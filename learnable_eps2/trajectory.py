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
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA


# --- 기존 헬퍼 함수 유지 ---
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def sample_random_clusters_gmm(means, covs, weights, pca_info, use_weight, batch_size, latent_shape, device=None):
    C, H, W = latent_shape
    all_means = torch.cat([m for m in means.values()], dim=0)
    all_covs = torch.cat([c for c in covs.values()], dim=0)
    all_weights = torch.cat([w for w in weights.values()], dim=0)

    pca_components_list = []
    for cls, info in pca_info.items():
        num_clusters = means[cls].shape[0]
        pca_components_list.append(info["components"].unsqueeze(0).expand(num_clusters, -1, -1))
    all_pca_comps = torch.cat(pca_components_list, dim=0)

    if use_weight == 'true':
        probs = all_weights / all_weights.sum()
        idx = torch.multinomial(probs, batch_size, replacement=True)
    else:
        total_clusters = len(all_means)
        idx = torch.randint(0, total_clusters, (batch_size,), device=device)

    m_orig = all_means[idx]
    v_pca = all_covs[idx]
    V = all_pca_comps[idx]
    cluster_means = m_orig.view(batch_size, C, H, W)
    v_orig = torch.bmm(v_pca.unsqueeze(1), V ** 2).squeeze(1)
    cluster_sigma = torch.sqrt(v_orig + 1e-6).view(batch_size, C, H, W)
    return cluster_means, cluster_sigma, idx


@torch.no_grad()
def load_model_and_transport(cfg, device, ckpt_path):
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
    model.load_state_dict(ckpt['ema'])
    model.eval()
    transport = create_transport(
        cfg['transport']['path_type'], cfg['transport']['prediction'],
        cfg['transport']['loss_weight'], cfg['transport']['train_eps'],
        cfg['transport']['sample_eps'],
    )
    sampler = Sampler(transport)
    return model, sampler


@torch.no_grad()
def sample_reverse_trajectory_ode(model, sampler, x_T, y, num_steps=50):
    ode_sampler = sampler.sample_ode(sampling_method="dopri5", num_steps=num_steps, reverse=False)
    xs = ode_sampler(x_T, model, y=y)
    return xs  # Tensor [T, B, C, H, W]


def get_real_latents(cfg, batch_size):
    dataset = ImgLatentDataset(
        data_dir=cfg['data']['data_path'],
        latent_norm=cfg['data'].get('latent_norm', False),
        latent_multiplier=cfg['data'].get('latent_multiplier', 0.18215),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    return x, y


def project_time_series_global(trajs):
    T, B = trajs.shape[:2]
    Z = trajs.reshape(T * B, -1).numpy()
    pca = PCA(n_components=1)
    Y = pca.fit_transform(Z).reshape(T, B)
    times = np.linspace(0, 1, T)
    return times, Y


def compute_trajectory_density(times, ys, t_bins=300, y_bins=300, smooth_sigma=1.0):
    # ys shape: (T, B)
    T, B = ys.shape

    # 에러 방지: times의 길이를 T와 강제로 맞춤
    if len(times) != T:
        times = np.linspace(0, 1, T)

    t_min, t_max = times.min(), times.max()
    y_min, y_max = ys.min(), ys.max()

    # meshgrid 방식을 사용하여 정확히 T*B의 길이를 가진 좌표쌍 생성
    # times_grid: (T, B)
    times_grid = np.tile(times[:, None], (1, B))

    t_coords = times_grid.flatten()
    y_coords = ys.flatten()

    # 여기서 발생했던 "x and y must have the same length" 에러를 원천 차단
    density, _, _ = np.histogram2d(
        t_coords, y_coords,
        bins=[t_bins, y_bins],
        range=[[t_min, t_max], [y_min, y_max]]
    )

    density = gaussian_filter(density.T, sigma=smooth_sigma)
    return density, [t_min, t_max, y_min, y_max]


# --- 새로 추가/수정된 핵심 함수 ---

def get_all_gmm_clusters(means, latent_shape, device):
    """모든 클러스터(30개)의 평균을 가져옴"""
    C, H, W = latent_shape
    all_means = torch.cat([m for m in means.values()], dim=0)  # (30, D)
    num_clusters = all_means.shape[0]
    cluster_means = all_means.view(num_clusters, C, H, W).to(device)
    return cluster_means, num_clusters


def get_gmm_noise_distribution(gmm_means, gmm_covs, latent_shape, device, y_grid):
    """모든 GMM 클러스터를 합친 다중 봉우리 분포 생성 (PCA-1 공간)"""
    C, H, W = latent_shape
    all_means = torch.cat([m for m in gmm_means.values()], dim=0)
    all_covs = torch.cat([c for c in gmm_covs.values()], dim=0)

    num_samples_per_cluster = 200
    samples_list = []

    # 각 클러스터별로 PCA-1 공간상에서 샘플링 (시각화 일관성을 위함)
    for k in range(all_means.shape[0]):
        # PCA-1 공간으로 투영된 평균값 계산 (단일 포인트 투영)
        m_single = all_means[k:k + 1].view(1, 1, C, H, W).cpu()
        _, m_proj = project_time_series_global(m_single)
        pca1_mean = m_proj[0, 0]

        # PCA 공간의 첫번째 주성분 분산 사용
        pca1_std = torch.sqrt(all_covs[k, 0]).item()

        cluster_samples = np.random.normal(pca1_mean, pca1_std, num_samples_per_cluster)
        samples_list.append(cluster_samples)

    all_samples = np.concatenate(samples_list)
    kde = gaussian_kde(all_samples, bw_method=0.15)
    return kde(y_grid)


def plot_latent_transport_density(times, ys, cluster_ys, gmm_means_raw, gmm_covs_raw, latent_shape, save_path):
    # Y축 범위: 데이터와 클러스터 궤적 전체를 포함
    all_y = np.concatenate([ys.flatten(), cluster_ys.flatten()])
    y_min, y_max = np.percentile(all_y, [0.5, 99.5])
    y_padding = (y_max - y_min) * 0.4
    y_limit = [y_min - y_padding, y_max + y_padding]

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[2.5, 6, 2], wspace=0.1)

    ax_noise = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1])
    ax_data = fig.add_subplot(gs[2])

    # 1. 중앙 Main Plot (누적 밀도 + 30개 클러스터 대표 궤적)
    density, _ = compute_trajectory_density(times, ys, t_bins=500, y_bins=500, smooth_sigma=4.0)
    density = np.log1p(density * 100)
    ax_main.imshow(density, origin="lower", extent=[0, 1, y_limit[0], y_limit[1]],
                   aspect="auto", cmap="magma", alpha=0.9)

    # 30개 클러스터 대표 선 (눈에 띄는 녹색)
    for i in range(cluster_ys.shape[1]):
        ax_main.plot(times, cluster_ys[:, i], color="#00FF7F", linewidth=1.5, alpha=0.8, zorder=10)

    ax_main.set_ylim(y_limit)
    ax_main.set_title("Integrated Latent Transport Flow", fontsize=15)
    ax_main.axis('off')

    # 2. 왼쪽 Noise 분포 (통합 GMM - 다중 봉우리)
    y_grid = np.linspace(y_limit[0], y_limit[1], 1000)
    dens_gmm = get_gmm_noise_distribution(gmm_means_raw, gmm_covs_raw, latent_shape, device, y_grid)

    ax_noise.plot(dens_gmm, y_grid, color="#CF9FFF", linewidth=2.5)
    ax_noise.fill_betweenx(y_grid, 0, dens_gmm, color="#CF9FFF", alpha=0.4)
    ax_noise.set_ylim(y_limit)
    ax_noise.invert_xaxis()
    ax_noise.axis('off')
    ax_noise.set_title("Integrated GMM Noise", color="#CF9FFF")

    # 3. 오른쪽 Data 분포 (전체 누적 데이터의 마지막 시점)
    kde_data = gaussian_kde(ys[-1, :], bw_method=0.15)
    dens_data = kde_data(y_grid)
    ax_data.plot(dens_data, y_grid, color="#FFCC00", linewidth=2.5)
    ax_data.fill_betweenx(y_grid, 0, dens_data, color="#FFCC00", alpha=0.4)
    ax_data.set_ylim(y_limit)
    ax_data.axis('off')
    ax_data.set_title("Target Data Dist.", color="#FFCC00")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- 실행부 ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='model2_xl_vavae_f16d32.yaml')
    parser.add_argument('--ckpt', type=str,
                        default='output/1st_lightningdit_xl_vavae_f16d32_gmm30/checkpoints/0162000.pt')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=50)
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    cfg = load_config(args.config)

    model, sampler = load_model_and_transport(cfg, device, args.ckpt)
    model = accelerator.prepare(model)
    base_model = model.module if hasattr(model, "module") else model
    latent_size = cfg['data']['image_size'] // 16
    latent_shape = (base_model.in_channels, latent_size, latent_size)

    # GMM 데이터 로드
    cluster_path = cfg['gmm']['load_dir']
    with open(os.path.join(cluster_path, "gmm_clusters.pkl"), "rb") as f:
        ckpt_gmm = pickle.load(f)

    gmm_means = {cls: torch.from_numpy(mu).to(device) for cls, mu in ckpt_gmm["means"].items()}
    gmm_covs = {cls: torch.from_numpy(cov).to(device) for cls, cov in ckpt_gmm["covs"].items()}
    gmm_weights = {cls: torch.from_numpy(w).to(device) for cls, w in ckpt_gmm["weights"].items()}
    gmm_pca = {cls: {"components": torch.from_numpy(p["components"]).to(device)} for cls, p in
               ckpt_gmm["pca_components"].items()}

    # 시각화용 원본 복사
    gmm_means_raw = deepcopy(gmm_means)
    gmm_covs_raw = deepcopy(gmm_covs)

    # 1. 모든 클러스터(30개) 대표 궤적 생성
    print("Sampling 30 representative cluster trajectories...")
    cluster_init, num_c = get_all_gmm_clusters(gmm_means_raw, latent_shape, device)
    dummy_y = torch.zeros(num_c, dtype=torch.long).to(device)  # 레이블은 0으로 가정하거나 필요시 수정
    c_trajs = sample_reverse_trajectory_ode(model, sampler, cluster_init, dummy_y, args.steps)
    _, cluster_ys = project_time_series_global(c_trajs.cpu())

    # 2. 전체 데이터셋 누적 루프
    all_ys = []
    num_iterations = 10
    print(f"Accumulating {num_iterations} batches for global density...")

    for i in tqdm(range(num_iterations)):
        x0_real, y_real = get_real_latents(cfg, args.batch_size)
        y_real = y_real.to(device)

        c_means, c_sigma, _ = sample_random_clusters_gmm(
            gmm_means, gmm_covs, gmm_weights, gmm_pca, False, args.batch_size, latent_shape, device
        )
        eps = torch.randn_like(c_means)
        xT = (0.5 * c_means) + (0.5 * eps)

        # sample_reverse_trajectory_ode 결과는 list[T] 또는 tensor[T, B, C, H, W]
        batch_trajs = sample_reverse_trajectory_ode(model, sampler, xT, y_real, args.steps)
        if isinstance(batch_trajs, list):
            batch_trajs = torch.stack(batch_trajs, dim=0)  # [T, B, C, H, W]

        _, batch_ys = project_time_series_global(batch_trajs.cpu())  # batch_ys: (T, B)
        all_ys.append(batch_ys)

    # 누적된 데이터 합치기
    total_ys = np.concatenate(all_ys, axis=1)  # (T, Total_B)

    # [안전장치] 시각화 전 데이터 백업 (나중에 시각화만 다시 할 수 있음)
    if accelerator.is_main_process:
        np.save("accumulated_ys.npy", total_ys)
        np.save("cluster_ys.npy", cluster_ys)
        print(f"Data saved. total_ys shape: {total_ys.shape}")

    # 3. 시각화 실행
    if accelerator.is_main_process:
        # times를 total_ys의 첫 번째 차원 길이에 맞게 다시 생성
        final_times = np.linspace(0, 1, total_ys.shape[0])

        print("Saving global trajectory plot...")
        plot_latent_transport_density(
            times=final_times,
            ys=total_ys,
            cluster_ys=cluster_ys,
            gmm_means_raw=gmm_means_raw,
            gmm_covs_raw=gmm_covs_raw,
            latent_shape=latent_shape,
            save_path="global_latent_transport_density.png"
        )
        print("Done!")