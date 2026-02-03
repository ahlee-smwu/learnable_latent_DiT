import torch
import torch.distributed as dist
import torch.backends.cuda
import torch.backends.cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import math
import yaml
import json
import numpy as np
import logging
import os
import argparse
from time import time
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm

from diffusers.models import AutoencoderKL
from models.lightningdit import LightningDiT_models
from transport import create_transport, Sampler
from accelerate import Accelerator
from datasets.img_latent_dataset import ImgLatentDataset
import pickle
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import gridspec
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def sample_random_clusters_gmm(
        means: dict,
        covs: dict,
        weights: dict,
        pca_info: dict,
        use_weight: bool,
        batch_size: int,
        latent_shape: tuple,
        device=None,
):
    C, H, W = latent_shape
    # D = 8192, PCA_dim = 256 가정

    # 1. 데이터 통합
    all_means = torch.cat([m for m in means.values()], dim=0)  # (total_K, 8192)
    all_covs = torch.cat([c for c in covs.values()], dim=0)  # (total_K, 256)
    all_weights = torch.cat([w for w in weights.values()], dim=0)  # (total_K,)

    pca_components_list = []
    for cls, info in pca_info.items():
        num_clusters = means[cls].shape[0]
        # info["components"] shape: (256, 8192)
        pca_components_list.append(info["components"].unsqueeze(0).expand(num_clusters, -1, -1))

    all_pca_comps = torch.cat(pca_components_list, dim=0)  # (total_K, 256, 8192)

    # 2. 클러스터 선택
    # 1) Mixture weight 기반 샘플링
    if use_weight == 'true':
        probs = all_weights / all_weights.sum()
        idx = torch.multinomial(probs, batch_size, replacement=True)
    # 2) 랜덤 샘플링
    else:
        total_clusters = len(all_means)
        idx = torch.randint(0, total_clusters, (batch_size,), device=device)

    # 3. 선택된 클러스터 파라미터
    m_orig = all_means[idx]  # (batch_size, 8192)
    v_pca = all_covs[idx]  # (batch_size, 256)
    V = all_pca_comps[idx]  # (batch_size, 256, 8192)

    # 4. 결과 도출
    # Mean: 학습 코드에서 이미 inverse_transform 되었으므로 그대로 사용
    cluster_means = m_orig.view(batch_size, C, H, W)

    # Sigma: PCA 공간의 분산을 원본 공간으로 투영 (v_pca @ V^2)
    # v_pca.unsqueeze(1) shape: (B, 1, 256)
    # V**2 shape: (B, 256, 8192)
    # Result: (B, 1, 8192)
    v_orig = torch.bmm(v_pca.unsqueeze(1), V ** 2).squeeze(1)
    cluster_sigma = torch.sqrt(v_orig + 1e-6).view(batch_size, C, H, W)

    return cluster_means, cluster_sigma, idx

@torch.no_grad()
def load_model_and_transport(cfg, device, ckpt_path):
    # model
    downsample_ratio = cfg['vae'].get('downsample_ratio', 16)
    latent_size = cfg['data']['image_size'] // downsample_ratio

    model = LightningDiT_models[cfg['model']['model_type']](
        input_size=latent_size,
        num_classes=cfg['data']['num_classes'],
        use_qknorm=cfg['model']['use_qknorm'],
        use_swiglu=cfg['model'].get('use_swiglu', False),
        use_rope=cfg['model'].get('use_rope', False),
        use_rmsnorm=cfg['model'].get('use_rmsnorm', False),
        wo_shift=cfg['model'].get('wo_shift', False),
        in_channels=cfg['model'].get('in_chans', 4),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['ema'])  # ✅ EMA 쓰는 게 중요
    model.eval()

    transport = create_transport(
        cfg['transport']['path_type'],
        cfg['transport']['prediction'],
        cfg['transport']['loss_weight'],
        cfg['transport']['train_eps'],
        cfg['transport']['sample_eps'],
        use_cosine_loss=cfg['transport'].get('use_cosine_loss', False),
        use_lognorm=cfg['transport'].get('use_lognorm', False),
    )

    sampler = Sampler(transport)

    return model, sampler

@torch.no_grad()
def sample_reverse_trajectory_ode(
    model,
    sampler,
    x_T,
    y,
    num_steps=50,
):
    """
    Returns:
        traj: list[T][B,C,H,W]
    """

    ode_sampler = sampler.sample_ode(
        sampling_method="dopri5",   # or "euler"
        num_steps=num_steps,
        reverse=False,              # noise → data
    )

    xs = ode_sampler(
        x_T,
        model,
        y=y,
    )

    # xs is already the full trajectory
    traj = [x.cpu() for x in xs]
    return traj

def get_real_latents(cfg, batch_size):
    dataset = ImgLatentDataset(
        data_dir=cfg['data']['data_path'],
        latent_norm=cfg['data'].get('latent_norm', False),
        latent_multiplier=cfg['data'].get('latent_multiplier', 0.18215),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    return x, y


def get_gmm_noise_distribution(gmm_means, gmm_covs, gmm_pca, latent_shape, device, y_grid):
    """
    모든 GMM 클러스터의 평균과 공분산을 사용하여 통합된 노이즈 분포를 생성합니다.
    이 함수는 여러 봉우리를 가진 복합적인 분포를 KDE로 추정합니다.
    """
    C, H, W = latent_shape
    all_means = torch.cat([m for m in gmm_means.values()], dim=0)  # (total_K, 8192)
    all_covs = torch.cat([c for c in gmm_covs.values()], dim=0)  # (total_K, 256) (PCA 공간 분산)
    all_pca_comps = torch.cat([info["components"].unsqueeze(0).expand(m.shape[0], -1, -1)
                               for cls, m in gmm_means.items() for info in [gmm_pca[cls]]],
                              dim=0)  # (total_K, 256, 8192)

    # 각 클러스터별로 충분한 샘플을 생성하여 KDE에 전달
    num_samples_per_cluster = 100  # 각 GMM 클러스터에서 샘플링할 개수
    total_K = all_means.shape[0]

    samples_list = []
    for k in range(total_K):
        m_orig = all_means[k:k + 1]  # (1, 8192)
        v_pca = all_covs[k:k + 1]  # (1, 256)
        V = all_pca_comps[k:k + 1]  # (1, 256, 8192)

        # PCA 공간의 분산 v_pca를 원래 공간으로 투영하여 표준편차 계산
        # v_orig = torch.bmm(v_pca.unsqueeze(1), V**2).squeeze(1) # (1, 8192)
        # sigma_orig = torch.sqrt(v_orig + 1e-6) # (1, 8192)

        # NOTE: PCA 변환된 데이터로 KDE를 그렸으므로, VAE Latent-PCA 1st component로
        # 매핑된 값으로 직접 샘플링하는 것이 더 정확합니다.
        # 여기서는 단순화를 위해 PCA_dim (256)의 첫 번째 차원(주성분1)에 해당하는
        # GMM 평균과 분산을 활용하여 샘플링합니다.

        # GMM은 PCA 공간에서 피팅되었으므로, 여기서 샘플링할 때는 PCA 공간에서의 분산 정보를
        # 사용해야 합니다. 첫 번째 주성분(PCA-1)에 해당하는 분산 값만 가져와서 사용
        pca1_mean = project_time_series_global(m_orig.view(1, 1, C, H, W).cpu()).ys[0, 0]  # (1,)

        # PCA 공간의 분산 v_pca (K, 256)
        # 여기서는 PCA-1에 해당하는 대략적인 분산값을 사용 (이 부분이 GMM fit 방법에 따라 달라질 수 있음)
        # GMM fit 과정에서 PCA component 별로 분산을 가지고 있다면 해당 값을 직접 사용해야 합니다.
        # 임시로 VAE output의 전역 분산으로 가정
        pca1_std = torch.sqrt(v_pca[:, 0]).item() if v_pca.shape[1] > 0 else 1.0  # PCA-1의 분산 사용

        # PCA-1 값으로 샘플링
        cluster_samples = torch.normal(pca1_mean, pca1_std, size=(num_samples_per_cluster,)).numpy()
        samples_list.append(cluster_samples)

    all_gmm_samples_1d = np.concatenate(samples_list)
    kde_gmm = gaussian_kde(all_gmm_samples_1d, bw_method=0.1)  # 대역폭 조절로 봉우리 표현

    return kde_gmm(y_grid)


from sklearn.decomposition import PCA

def project_time_series(trajs, fit_start_ratio=0.7):
    """
    Args:
        trajs: list[T][B,C,H,W]
    Returns:
        times: (T,)
        ys: (T,B)  PCA-1 projected values
    """
    T = len(trajs)
    B = trajs[0].size(0)

    # ---- flatten
    trajs_flat = [
        t.view(B, -1).cpu().numpy()
        for t in trajs
    ]

    # ---- PCA fit on late steps (x0 region)
    start = int(T * fit_start_ratio)
    Z_fit = np.concatenate(trajs_flat[start:], axis=0)

    pca = PCA(n_components=1)
    pca.fit(Z_fit)

    # ---- project all
    ys = np.stack([
        pca.transform(z).squeeze(1)
        for z in trajs_flat
    ])  # (T, B)

    times = np.linspace(0, 1, T)
    return times, ys


from matplotlib.collections import LineCollection


def plot_latent_transport_density(times, ys, cluster_ys, gmm_means_raw, gmm_covs_raw, gmm_pca_raw, latent_shape,
                                  save_path):
    # 1. Y축 범위 최적화 (모든 데이터, 클러스터, GMM 분포를 아우르도록)
    # GMM 클러스터 평균들의 PCA-1 범위도 고려
    pca_proj_cluster_means = np.array(
        [project_time_series_global(m.view(1, 1, *latent_shape).cpu()).ys[0, 0] for m_dict in gmm_means_raw.values() for
         m in m_dict])

    all_y_values = np.concatenate([ys.flatten(), cluster_ys.flatten(), pca_proj_cluster_means])

    y_min, y_max = np.percentile(all_y_values, [1, 99])
    y_padding = (y_max - y_min) * 0.4  # 이전보다 더 넉넉하게
    y_limit = [y_min - y_padding, y_max + y_padding]

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[2.5, 6, 2], wspace=0.1)  # 왼쪽 분포 영역을 더 넓게

    ax_noise = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1])
    ax_data = fig.add_subplot(gs[2])

    # -------------------------------------------------
    # 중앙 Main Plot: 누적된 데이터의 밀도 및 모든 클러스터 궤적
    # -------------------------------------------------
    density, extent = compute_trajectory_density(times, ys, t_bins=500, y_bins=500, smooth_sigma=4.0)
    density = np.log1p(density * 100)
    ax_main.imshow(density, origin="lower", extent=[0, 1, ys.min(), ys.max()],
                   aspect="auto", cmap="magma", alpha=0.9)

    # 모든 GMM 클러스터(30개)의 대표 궤적을 선으로 그림
    for i in range(cluster_ys.shape[1]):
        ax_main.plot(times, cluster_ys[:, i], color="#00FF7F", linewidth=1.5, alpha=0.8, zorder=10)

    ax_main.set_ylim(y_limit)
    ax_main.axis('off')
    ax_main.set_title("Latent Space Transport Flow", fontsize=15, pad=20)

    # -------------------------------------------------
    # 왼쪽 Noise 분포 (모든 GMM 클러스터를 통합한 복합 분포)
    # -------------------------------------------------
    y_grid = np.linspace(y_limit[0], y_limit[1], 1000)
    # GMM 통합 분포 함수 호출
    dens_gmm_integrated = get_gmm_noise_distribution(gmm_means_raw, gmm_covs_raw, gmm_pca_raw, latent_shape, device,
                                                     y_grid)

    ax_noise.plot(dens_gmm_integrated, y_grid, color="#CF9FFF", linewidth=2.5)
    ax_noise.fill_betweenx(y_grid, 0, dens_gmm_integrated, color="#CF9FFF", alpha=0.4)
    ax_noise.set_ylim(y_limit)
    ax_noise.invert_xaxis()
    ax_noise.axis('off')
    ax_noise.set_title("GMM Noise Distribution", color="#CF9FFF", pad=10)

    # -------------------------------------------------
    # 오른쪽 Data 분포 (전체 데이터셋 분포)
    # -------------------------------------------------
    kde_data = gaussian_kde(ys[-1, :], bw_method=0.15)
    dens_data = kde_data(y_grid)

    ax_data.plot(dens_data, y_grid, color="#FFCC00", linewidth=2.5)
    ax_data.fill_betweenx(y_grid, 0, dens_data, color="#FFCC00", alpha=0.4)
    ax_data.set_ylim(y_limit)
    ax_data.axis('off')
    ax_data.set_title("Target Data Dist.", color="#FFCC00", pad=10)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


from sklearn.decomposition import PCA

def project_time_series_global(trajs):
    """
    trajs: Tensor [T,B,C,H,W]
    Returns:
        times: (T,)
        ys: (T,B)
    """
    T, B = trajs.shape[:2]

    Z = trajs.reshape(T * B, -1).numpy()

    pca = PCA(n_components=1)
    pca.fit(Z)              # 🔴 단 한 번만 fit

    Y = pca.transform(Z).reshape(T, B)

    times = np.linspace(0, 1, T)
    return times, Y


def compute_trajectory_density(times, ys, t_bins=300, y_bins=300, smooth_sigma=1.0):
    T, B = ys.shape
    t_min, t_max = times.min(), times.max()
    y_min, y_max = ys.min(), ys.max()

    # 모든 시점의 데이터를 2D 히스토그램으로 쌓음
    t_coords = np.repeat(times[:, None], B, axis=1).flatten()
    y_coords = ys.flatten()

    density, _, _ = np.histogram2d(
        t_coords, y_coords,
        bins=[t_bins, y_bins],
        range=[[t_min, t_max], [y_min, y_max]]
    )

    density = gaussian_filter(density.T, sigma=smooth_sigma)
    return density, [t_min, t_max, y_min, y_max]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='model2_xl_vavae_f16d32.yaml')
    parser.add_argument('--ckpt', type=str, default='output/1st_lightningdit_xl_vavae_f16d32_gmm30/checkpoints/0162000.pt')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=50)
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    # ✅ parser에서 받은 config 사용
    cfg = load_config(args.config)
    model, sampler = load_model_and_transport(
        cfg,
        device,
        ckpt_path=args.ckpt
    )
    model = accelerator.prepare(model)

    latent_size = cfg['data']['image_size'] // 16
    print("Load model and sampler: Done!")

    # real latent batch
    x0, y = get_real_latents(cfg, batch_size=args.batch_size)
    x0, y = accelerator.prepare(x0.to(device), y.to(device))
    print("Load latent: Done!")

    # GMM
    cluster_path = cfg['gmm']['load_dir']
    with open(os.path.join(cluster_path, "gmm_clusters.pkl"), "rb") as f:
        ckpt = pickle.load(f)
    # ---- Mean (μ): class -> (K, D)
    gmm_means = {
        cls: torch.from_numpy(mu).to(device=device, dtype=torch.float32)
        for cls, mu in ckpt["means"].items()
    }
    # ---- Covariance (Σ)
    # diag: (K, D)
    # full: (K, D, D)
    gmm_covs = {
        cls: torch.from_numpy(cov).to(device=device, dtype=torch.float32)
        for cls, cov in ckpt["covs"].items()
    }
    # ---- Mixture weight (π)
    gmm_weights = {
        cls: torch.from_numpy(w).to(device=device, dtype=torch.float32)
        for cls, w in ckpt["weights"].items()
    }
    # ---- PCA components (class-wise)
    gmm_pca = {
        cls: {
            "components": torch.from_numpy(pca_dict["components"])
            .to(device=device, dtype=torch.float32),
            "mean": torch.from_numpy(pca_dict["mean"])
            .to(device=device, dtype=torch.float32),
        }
        for cls, pca_dict in ckpt["pca_components"].items()
    }
    gmm_labels = ckpt.get("labels", None)
    gmm_means_raw = deepcopy(gmm_means)
    gmm_covs_raw = deepcopy(gmm_covs)
    gmm_pca_raw = deepcopy(gmm_pca)

    all_accumulated_ys = []
    num_iterations = 10



    print("Load GMM: Done!")

    # noise init
    # xT = torch.randn_like(x0) # normal gaussian
    # gmm
    base_model = model.module if hasattr(model, "module") else model
    cluster_means, cluster_sigma, cluster_ids = sample_random_clusters_gmm(
        gmm_means, gmm_covs, gmm_weights, gmm_pca,
        use_weight=False,
        batch_size=args.batch_size,
        latent_shape=(base_model.in_channels, latent_size, latent_size),
        device=device
    )
    cluster_sigma = torch.ones_like(cluster_means)

    eps = torch.randn_like(cluster_means)
    cluster = cluster_means + (eps * cluster_sigma)
    xT = accelerator.prepare((0.5 * cluster) + (0.5 * eps))

    trajs = sample_reverse_trajectory_ode(
        model=model,
        sampler=sampler,
        x_T=xT,
        y=y,
        num_steps=50,
    )
    # list[T][B,C,H,W] → tensor[T,B,C,H,W]
    trajs = torch.stack(trajs, dim=0).to(device)
    trajs = accelerator.gather(trajs)

    # sanity check
    print(trajs[0].std(), trajs[-1].std()) # trajs[0] = noise, trajs[-1] = data(x₀)
    print("sample trajectory: Done!")

    # trajs_2d = project_pca(trajs)
    # plot(trajs_2d, "traj_ode.png")

    if accelerator.is_main_process:
        trajs = trajs.cpu()
    else:
        del trajs
        torch.cuda.empty_cache()

    if accelerator.is_main_process:
        times, ys = project_time_series_global(trajs)

        plot_latent_transport_density(
            times,
            ys,
            save_path="latent_transport_density.png",
            num_sample_trajs=8,
        )

        print("Plot trajectory: Done!")
    accelerator.wait_for_everyone()