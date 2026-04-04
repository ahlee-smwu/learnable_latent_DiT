"""
analyze_velocity_lanes.py

Reverse ODE로 x1 → x0를 역추적하여 covered / train_fail 그룹의
velocity lane이 분리되어 있는지 확인.

결과 해석:
  lane 분리 (ratio > 1.3, p < 0.05)
    → velocity field가 lane을 학습함
    → 추론 문제: prior가 train_fail lane에 도달 못 함
  lane 겹침 (ratio ≈ 1.0)
    → velocity field가 lane을 구분 못 함
    → 학습 문제: velocity averaging으로 lane 자체가 형성 안 됨
"""

import os
import yaml
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from sklearn.decomposition import PCA as skPCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mannwhitneyu, kruskal
import warnings
warnings.filterwarnings("ignore", message=r"Glyph .* missing from font\(s\) DejaVu Sans.")

import matplotlib
import matplotlib.font_manager as fm
for _fn in ['Noto Sans CJK JP', 'NanumGothic', 'Malgun Gothic', 'AppleGothic']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        matplotlib.rc('font', family=_fn)
        matplotlib.rcParams['axes.unicode_minus'] = False
        break


def load_model_from_ckpt(train_config, ckpt_path, device):
    """
    inference.py의 모델 로딩 패턴과 동일.
    EMA weight를 로드하고 eval 모드로 반환.
    """
    from models.lightningdit import LightningDiT_models

    downsample_ratio = train_config['vae'].get('downsample_ratio', 16)
    latent_size = train_config['data']['image_size'] // downsample_ratio

    model = LightningDiT_models[train_config['model']['model_type']](
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model'].get('use_swiglu', False),
        use_rope=train_config['model'].get('use_rope', False),
        use_rmsnorm=train_config['model'].get('use_rmsnorm', False),
        wo_shift=train_config['model'].get('wo_shift', False),
        in_channels=train_config['model'].get('in_chans', 4),
    )

    # inference.py 179-183줄과 동일한 패턴
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    print(f"[Model] Loaded EMA checkpoint: {ckpt_path}")
    print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model


def load_transport(train_config):
    """inference.py의 transport 생성 패턴과 동일."""
    from transport import create_transport, Sampler

    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss=train_config['transport'].get('use_cosine_loss', False),
        use_lognorm=train_config['transport'].get('use_lognorm', False),
    )
    return transport, Sampler(transport)


def run_reverse_ode(
    model,
    sampler,          # 이 실험에선 사용 안 함, 시그니처 유지용
    latents_flat,     # (N, D) numpy — real latent (x1)
    latent_shape,     # (C, H, W)
    device,
    label=0,
    num_steps=40,
    batch_size=32,
):
    """
    x1 → x0 reverse Euler.
    sample_ode(reverse=True)의 integrator assertion 문제를 우회.

    forward ODE: dx/dt = v(x, t),   t: 0 → 1
    reverse ODE: dx/dt = -v(x, t),  t: 1 → 0
    Euler:  x_{t - dt} = x_t - dt * v(x_t, t)
    """
    C, H, W = latent_shape
    N = len(latents_flat)

    # t=1 에서 t=0 으로 균등 분할
    # timesteps[0]=1.0, timesteps[-1]=0.0
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)   # (num_steps+1,)
    dt = (1.0 / num_steps)  # 각 step 크기 (양수)

    x0_list = []

    with torch.no_grad():
        for i in tqdm(range(0, N, batch_size), desc="  Reverse ODE"):
            batch_np = latents_flat[i:i + batch_size]          # (B, D)
            x = (torch.from_numpy(batch_np)
                 .float()
                 .view(-1, C, H, W)
                 .to(device))                                   # (B, C, H, W)

            y = torch.full((len(x),), label,
                           dtype=torch.long, device=device)

            # Euler reverse: t: 1 → 0
            for j in range(num_steps):
                t_val = timesteps[j].item()                     # 현재 t
                t_batch = torch.full(
                    (len(x),), t_val,
                    dtype=torch.float32, device=device
                )

                # velocity field: v(x_t, t)
                v = model(x, t_batch, y=y)                      # (B, C, H, W)

                # x_{t-dt} = x_t - dt * v
                x = x - dt * v

            x0_list.append(x.cpu())

    x0_all = torch.cat(x0_list, dim=0)        # (N, C, H, W)
    return x0_all.view(N, -1).numpy()          # (N, D)

'''compare covered vs train_fail in time step level'''
def run_reverse_ode_compare(
    model,
    x_cov_np,
    x_tf_np,
    latent_shape,
    device,
    label,
    num_steps,
):
    C, H, W = latent_shape

    x_cov = torch.from_numpy(x_cov_np).float().view(-1, C, H, W).to(device)
    x_tf  = torch.from_numpy(x_tf_np).float().view(-1, C, H, W).to(device)

    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
    dt = 1.0 / num_steps

    direction_stats = []

    for j in range(num_steps):
        t_val = timesteps[j].item()

        t_cov = torch.full((len(x_cov),), t_val, device=device)
        t_tf  = torch.full((len(x_tf),),  t_val, device=device)

        v_cov = model(x_cov, t_cov, y=torch.full((len(x_cov),), label, device=device))
        v_tf  = model(x_tf,  t_tf,  y=torch.full((len(x_tf),),  label, device=device))

        '''compare covered vs train_fail in time step level'''
        if j % 5 == 0:
            stats = analyze_velocity_direction_from_v(v_cov, v_tf)
            direction_stats.append((t_val, stats))

        # step
        x_cov = x_cov - dt * v_cov
        x_tf  = x_tf  - dt * v_tf

    return direction_stats

def analyze_velocity_lanes(
    model,
    sampler,
    covered_latents_flat,    # (N_cov, D) numpy — covered real x1
    train_fail_latents_flat, # (N_tf,  D) numpy — train_fail real x1
    gmm_means,               # dict {cls: (K, D)} — GMM means (numpy or tensor)
    target_k,
    save_path,
    device,
    latent_shape,            # (C, H, W)
    label=0,
    num_steps=40,
    batch_size=32,
    eps=1e-8,
):
    """
    Reverse ODE로 covered / train_fail x1의 x0 분포를 비교.
    """
    os.makedirs(save_path, exist_ok=True)
    print(f"\n[Velocity Lane Analysis] k={target_k}")
    print(f"  covered: {len(covered_latents_flat)} | train_fail: {len(train_fail_latents_flat)}")

    # ── 1. Reverse ODE ───────────────────────────────────────────────
    print("  Running reverse ODE for covered...")
    x0_cov = run_reverse_ode_compare(
        model, sampler, covered_latents_flat,
        latent_shape, device, label=label,
        num_steps=num_steps, batch_size=batch_size,
    )   # (N_cov, D)

    print("  Running reverse ODE for train_fail...")
    x0_tf = run_reverse_ode_compare(
        model, sampler, train_fail_latents_flat,
        latent_shape, device, label=label,
        num_steps=num_steps, batch_size=batch_size,
    )   # (N_tf, D)

    # ── 2. PCA-64 압축 ───────────────────────────────────────────────
    pca64 = skPCA(n_components=64, random_state=42)
    pca64.fit(np.vstack([x0_cov, x0_tf]))
    x0_cov_pca = pca64.transform(x0_cov)   # (N_cov, 64)
    x0_tf_pca  = pca64.transform(x0_tf)    # (N_tf,  64)

    # ── 3. Lane separation: train_fail x0 → covered x0 NN 거리 ──────
    nn_cov = NearestNeighbors(n_neighbors=5).fit(x0_cov_pca)
    d_tf_to_cov, _ = nn_cov.kneighbors(x0_tf_pca)
    d_tf_to_cov = d_tf_to_cov.mean(axis=1)   # (N_tf,)

    # covered self-NN (baseline density)
    nn_self = NearestNeighbors(n_neighbors=6).fit(x0_cov_pca)
    d_self, _ = nn_self.kneighbors(x0_cov_pca)
    d_self = d_self[:, 1:].mean(axis=1)      # (N_cov,)  자기 자신 제외

    _, p_lane = mannwhitneyu(d_tf_to_cov, d_self, alternative='greater')
    ratio_lane = np.median(d_tf_to_cov) / (np.median(d_self) + eps)

    print(f"\n  ── x0 Lane Separation (k={target_k}) ──")
    print(f"  covered  self-NN  median: {np.median(d_self):.4f}")
    print(f"  tf → covered NN  median: {np.median(d_tf_to_cov):.4f}")
    print(f"  Ratio (tf/cov):          {ratio_lane:.3f}")
    print(f"  Mann-Whitney p:          {p_lane:.3e}")

    if p_lane < 0.05 and ratio_lane > 1.3:
        verdict = "추론 문제"
        msg = [
            "  ⚠  train_fail x0가 covered x0와 분리된 영역에 위치",
            "     → velocity field가 lane을 학습함",
            "     → 추론 문제: prior가 train_fail lane에 도달하지 못함",
            "     → prior 시작점을 바꾸면 개선 가능성 있음",
        ]
    else:
        verdict = "학습 문제"
        msg = [
            "  ✅ train_fail x0가 covered x0와 겹치는 영역에 위치",
            "     → velocity field가 lane을 구분하지 못함",
            "     → 학습 문제: velocity averaging으로 lane 자체가 형성 안 됨",
            "     → prior를 바꿔도 근본적 해결 안 됨 → 학습 알고리즘 수정 필요",
        ]
    for m in msg:
        print(m)

    # ── 4. μ_k 기준 x0 거리 비교 ─────────────────────────────────────
    # "prior가 얼마나 각 lane에 도달할 수 있는가"
    cls_key  = sorted(gmm_means.keys())[0]
    mu_k_raw = gmm_means[cls_key][target_k]
    if isinstance(mu_k_raw, torch.Tensor):
        mu_k_raw = mu_k_raw.cpu().numpy()
    mu_k_pca64 = pca64.transform(mu_k_raw.reshape(1, -1))   # (1, 64)

    d_cov_to_mu = np.linalg.norm(x0_cov_pca - mu_k_pca64, axis=1)  # (N_cov,)
    d_tf_to_mu  = np.linalg.norm(x0_tf_pca  - mu_k_pca64, axis=1)  # (N_tf,)

    _, p_mu = mannwhitneyu(d_tf_to_mu, d_cov_to_mu, alternative='greater')
    ratio_mu = np.median(d_tf_to_mu) / (np.median(d_cov_to_mu) + eps)

    print(f"\n  ── x0 → μ_{target_k} 거리 비교 ──")
    print(f"  covered  x0 → μ_k  median: {np.median(d_cov_to_mu):.4f}")
    print(f"  tf       x0 → μ_k  median: {np.median(d_tf_to_mu):.4f}")
    print(f"  Ratio (tf/cov):             {ratio_mu:.3f}")
    print(f"  Mann-Whitney p:             {p_mu:.3e}")

    if p_mu < 0.05 and ratio_mu > 1.2:
        print(f"  → train_fail lane의 x0가 μ_k에서 더 멀리 있음")
        print(f"    prior N(μ_k, Σ_k)가 train_fail lane 시작점에 도달하기 어려움")
    else:
        print(f"  → covered와 train_fail의 x0가 μ_k와 비슷한 거리에 있음")
        print(f"    prior 위치 문제보다 velocity averaging이 주 원인")

    # ── 5. 시각화 ─────────────────────────────────────────────────────
    pca2 = skPCA(n_components=2, random_state=42)
    pca2.fit(np.vstack([x0_cov_pca, x0_tf_pca]))
    x0_cov_2d = pca2.transform(x0_cov_pca)
    x0_tf_2d  = pca2.transform(x0_tf_pca)
    mu_k_2d   = pca2.transform(mu_k_pca64)

    # x1 (원래 real latent)도 PCA-2D로 표시
    pca2_x1 = skPCA(n_components=2, random_state=42)
    pca2_x1.fit(np.vstack([covered_latents_flat, train_fail_latents_flat]))
    x1_cov_2d = pca2_x1.transform(covered_latents_flat)
    x1_tf_2d  = pca2_x1.transform(train_fail_latents_flat)

    C_COV, C_TF = '#1a6fa8', '#c0392b'

    fig = plt.figure(figsize=(22, 10), facecolor='white')
    fig.suptitle(
        f'Velocity Lane Analysis: k={target_k}  |  verdict: {verdict}\n'
        f'lane ratio={ratio_lane:.3f}  p={p_lane:.2e}  '
        f'|  μ_k dist ratio={ratio_mu:.3f}  p={p_mu:.2e}',
        fontsize=11, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    def _style(ax, title):
        ax.set_facecolor('white')
        ax.tick_params(colors='#333', labelsize=8)
        ax.set_title(title, fontsize=9, pad=4)
        for sp in ax.spines.values():
            sp.set_edgecolor('#cccccc')

    # Row 0: x1 공간 (원래 real latent)
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(x1_cov_2d[:,0], x1_cov_2d[:,1],
               s=5, alpha=0.3, color=C_COV, label=f'covered ({len(x1_cov_2d)})')
    ax.scatter(x1_tf_2d[:,0],  x1_tf_2d[:,1],
               s=5, alpha=0.3, color=C_TF,  label=f'train_fail ({len(x1_tf_2d)})')
    ax.legend(fontsize=7, framealpha=0.5)
    _style(ax, f'x1 공간 (real latent)\nk={target_k} 내 covered vs train_fail')

    # Row 0: x0 공간 (reverse ODE 결과)
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(x0_cov_2d[:,0], x0_cov_2d[:,1],
               s=5, alpha=0.3, color=C_COV, label=f'covered x0 ({len(x0_cov_2d)})')
    ax.scatter(x0_tf_2d[:,0],  x0_tf_2d[:,1],
               s=5, alpha=0.3, color=C_TF,  label=f'train_fail x0 ({len(x0_tf_2d)})')
    ax.scatter(mu_k_2d[0,0], mu_k_2d[0,1],
               s=200, marker='*', color='black', zorder=5, label=f'μ_{target_k}')
    ax.legend(fontsize=7, framealpha=0.5)
    _style(ax, f'x0 공간 (reverse ODE)\n분리 → lane 존재 / 겹침 → lane 없음')

    # Row 0: overlay (x1과 x0를 같은 PCA 공간에)
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(x0_cov_2d[:,0], x0_cov_2d[:,1],
               s=5, alpha=0.25, color=C_COV, marker='o', label='covered x0')
    ax.scatter(x0_tf_2d[:,0],  x0_tf_2d[:,1],
               s=5, alpha=0.25, color=C_TF,  marker='o', label='tf x0')
    ax.scatter(mu_k_2d[0,0], mu_k_2d[0,1],
               s=200, marker='*', color='black', zorder=5, label=f'μ_{target_k}')
    # μ_k를 중심으로 한 원 (prior의 도달 범위 시각화)
    theta  = np.linspace(0, 2*np.pi, 100)
    radius = np.median(d_cov_to_mu) * 0.05  # 시각적 스케일
    ax.plot(mu_k_2d[0,0] + radius*np.cos(theta),
            mu_k_2d[0,1] + radius*np.sin(theta),
            'k--', lw=0.8, alpha=0.5, label='prior reach (approx)')
    ax.legend(fontsize=7, framealpha=0.5)
    _style(ax, 'x0 공간 + μ_k 위치\nprior가 두 lane 모두 닿는가')

    # Row 1: NN distance histogram
    ax = fig.add_subplot(gs[1, 0])
    all_d = np.concatenate([d_self, d_tf_to_cov])
    bins  = np.linspace(np.percentile(all_d,1), np.percentile(all_d,99), 45)
    ax.hist(d_self,       bins=bins, density=True, alpha=0.6,
            color=C_COV, label=f'cov self-NN  med={np.median(d_self):.2f}')
    ax.hist(d_tf_to_cov,  bins=bins, density=True, alpha=0.6,
            color=C_TF,  label=f'tf→cov NN    med={np.median(d_tf_to_cov):.2f}')
    ax.axvline(np.median(d_self),      color=C_COV, lw=1.5, ls='--')
    ax.axvline(np.median(d_tf_to_cov), color=C_TF,  lw=1.5, ls='--')
    ax.set_xlabel('x0 NN distance (PCA-64)', color='#444', fontsize=8)
    ax.legend(fontsize=7, framealpha=0.5)
    _style(ax,
           f'x0 NN 거리  ratio={ratio_lane:.3f}\n'
           f'오른쪽으로 분리 → lane 존재')

    # Row 1: μ_k 거리 histogram
    ax = fig.add_subplot(gs[1, 1])
    all_mu = np.concatenate([d_cov_to_mu, d_tf_to_mu])
    bins_m = np.linspace(np.percentile(all_mu,1), np.percentile(all_mu,99), 45)
    ax.hist(d_cov_to_mu, bins=bins_m, density=True, alpha=0.6,
            color=C_COV, label=f'covered  med={np.median(d_cov_to_mu):.2f}')
    ax.hist(d_tf_to_mu,  bins=bins_m, density=True, alpha=0.6,
            color=C_TF,  label=f'tf       med={np.median(d_tf_to_mu):.2f}')
    ax.axvline(np.median(d_cov_to_mu), color=C_COV, lw=1.5, ls='--')
    ax.axvline(np.median(d_tf_to_mu),  color=C_TF,  lw=1.5, ls='--')
    ax.set_xlabel(f'x0 → μ_{target_k} distance (PCA-64)', color='#444', fontsize=8)
    ax.legend(fontsize=7, framealpha=0.5)
    _style(ax,
           f'x0 → μ_k 거리  ratio={ratio_mu:.3f}\n'
           f'분리 → prior가 tf lane에 못 닿음')

    # Row 1: CDF 비교
    ax = fig.add_subplot(gs[1, 2])
    for vals, label_str, color in [
        (d_self,       'cov self-NN', C_COV),
        (d_tf_to_cov,  'tf→cov NN',  C_TF),
    ]:
        sv = np.sort(vals)
        ax.plot(sv, np.arange(1, len(sv)+1)/len(sv),
                color=color, lw=1.8, label=label_str)
    ax.axhline(0.5, color='#888', lw=0.8, ls=':', alpha=0.6)
    ax.set_xlabel('x0 NN distance', color='#444', fontsize=8)
    ax.set_ylabel('CDF', color='#444', fontsize=8)
    ax.legend(fontsize=7, framealpha=0.5)
    ax.grid(alpha=0.2, color='#cccccc')
    _style(ax, 'x0 NN 거리 CDF\n두 곡선 분리 정도 확인')

    fig.tight_layout()
    fname = os.path.join(save_path, f'velocity_lane_k{target_k}.svg')
    fig.savefig(fname, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n✅ Saved: {fname}")

    return {
        'x0_covered_pca':    x0_cov_pca,
        'x0_train_fail_pca': x0_tf_pca,
        'd_tf_to_cov':       d_tf_to_cov,
        'd_self':            d_self,
        'd_cov_to_mu':       d_cov_to_mu,
        'd_tf_to_mu':        d_tf_to_mu,
        'ratio_lane':        ratio_lane,
        'ratio_mu':          ratio_mu,
        'p_lane':            p_lane,
        'p_mu':              p_mu,
        'verdict':           verdict,
    }

def analyze_velocity_direction_from_v(v_cov, v_tf):
    v_cov = v_cov.view(len(v_cov), -1)
    v_tf  = v_tf.view(len(v_tf), -1)

    v_cov_n = F.normalize(v_cov, dim=1)
    v_tf_n  = F.normalize(v_tf, dim=1)

    cos = (v_cov_n @ v_tf_n.T).mean().item()

    var_cov = v_cov_n.var(dim=0).mean().item()
    var_tf  = v_tf_n.var(dim=0).mean().item()

    return {
        "cosine": cos,
        "var_cov": var_cov,
        "var_tf": var_tf
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main: analyze_uncovered_real 결과를 받아서 velocity lane 분석
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── config 로드 ──────────────────────────────────────────────────
    train_config = load_config(args.train_config_path)
    ds_config    = load_config(args.ds_config_path)

    # ── latent shape 계산 (inference.py와 동일) ──────────────────────
    downsample_ratio = train_config['vae'].get('downsample_ratio', 16)
    latent_size      = train_config['data']['image_size'] // downsample_ratio
    C_latent         = train_config['model'].get('in_chans', 4)
    latent_shape     = (C_latent, latent_size, latent_size)
    D                = C_latent * latent_size * latent_size
    print(f"[Config] latent_shape={latent_shape}  D={D}")

    # ── 모델 로드 ────────────────────────────────────────────────────
    model = load_model_from_ckpt(train_config, args.ckpt_path, device)

    # ── transport ────────────────────────────────────────────────────
    transport, sampler = load_transport(train_config)

    # ── real latents 로드 (collect_real_latents와 동일 방식) ─────────
    from torch.utils.data import DataLoader
    from datasets.img_latent_dataset import ImgLatentDataset
    from tqdm import tqdm as _tqdm

    dataset = ImgLatentDataset(
        data_dir=ds_config['data']['data_path'],
        latent_norm=ds_config['data'].get('latent_norm', True),
        latent_multiplier=ds_config['data'].get('latent_multiplier', 0.18215),
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=True,
                        num_workers=ds_config['data']['num_workers'],
                        pin_memory=True)

    latents_list, labels_list = [], []
    with torch.no_grad():
        for x, y in _tqdm(loader, desc="Collect real latents"):
            latents_list.append(x.view(x.size(0), -1).numpy())
            labels_list.append(y.numpy())
    real_latents = np.concatenate(latents_list)   # (N, D) flattened
    real_labels  = np.concatenate(labels_list)

    # ── GMM 로드 ─────────────────────────────────────────────────────
    gmm_dir = (f"{ds_config['gmm']['output_dir']}/"
               f"{ds_config['gmm']['num_clusters']}_{ds_config['gmm']['cov_type']}")
    with open(os.path.join(gmm_dir, "gmm_clusters.pkl"), "rb") as f:
        gmm_ckpt = pickle.load(f)

    gmm_means   = gmm_ckpt["means"]
    gmm_covs    = gmm_ckpt["covs"]
    gmm_weights = gmm_ckpt["weights"]
    gmm_pca     = gmm_ckpt["pca_components"]

    # ── gen latents 로드 (collect_gen_latents_with_source_cluster) ───
    # analyze_uncovered_real을 재실행하여 covered/train_fail 얻기
    # (이미 SVG로 저장된 결과가 있다면 mask만 재계산)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analysis_cluster_w_GT import (
        compute_gmm_posterior,
        collect_gen_latents_with_source_cluster,
        analyze_uncovered_real,
    )
    from tokenizer.vavae import VA_VAE

    vae_wrapper = VA_VAE(args.vae_config_path)

    print("\n[Gen latents] Loading generated images...")
    gen_latents_by_cluster, gen_source_k = collect_gen_latents_with_source_cluster(
        generated_dir=args.generated_dir,
        vae_wrapper=vae_wrapper,
        ds_config=ds_config,
        device=device,
        max_images_per_cluster=3000,
    )
    gen_latents_all = gen_latents_by_cluster  # (N, D)

    # ── posterior로 real argmax 계산 ─────────────────────────────────
    real_post, real_argmax, _, _ = compute_gmm_posterior(
        real_latents, gmm_means, gmm_covs, gmm_weights, gmm_pca,
        use_weight=ds_config['gmm'].get('use_weight', True),
    )

    save_base = os.path.join(args.generated_dir, "../velocity_lane_analysis")

    for target_k in args.target_clusters:
        print(f"\n{'='*60}")
        print(f"  Target cluster: k={target_k}")
        print(f"{'='*60}")

        # ── npz 캐시 확인 ────────────────────────────────────────────
        npz_path = os.path.join(
            args.generated_dir, "../posterior_analysis",
            f"uncovered_masks_k{target_k}.npz"
        )

        if os.path.exists(npz_path):
            print(f"  [Cache] Loading masks from: {npz_path}")
            data = np.load(npz_path)
            covered_flat    = data['covered_latents']     # (N_cov, D)
            train_fail_flat = data['train_fail_latents']  # (N_tf,  D)
            print(f"  covered: {len(covered_flat)}, "
                  f"train_fail: {len(train_fail_flat)}")

        else:
            # npz 없으면 analyze_uncovered_real 재실행
            print(f"  [Cache] npz not found, running analyze_uncovered_real...")
            result = analyze_uncovered_real(
                gen_latents=gen_latents_all,
                gen_source_k=gen_source_k,
                real_latents=real_latents,
                gmm_means=gmm_means,
                gmm_covs=gmm_covs,
                gmm_weights=gmm_weights,
                gmm_pca=gmm_pca,
                target_k=target_k,
                save_path=os.path.join(
                    args.generated_dir, "../posterior_analysis"),
                use_weight=True,
            )
            mask_real_k     = (real_argmax == target_k)
            R_k             = real_latents[mask_real_k]
            covered_flat    = R_k[result['covered']]
            train_fail_flat = R_k[result['train_fail']]

        # ── subsample ────────────────────────────────────────────────
        max_n = args.max_samples_per_group
        if len(covered_flat) > max_n:
            idx = np.random.choice(len(covered_flat), max_n, replace=False)
            covered_flat = covered_flat[idx]
        if len(train_fail_flat) > max_n:
            idx = np.random.choice(len(train_fail_flat), max_n, replace=False)
            train_fail_flat = train_fail_flat[idx]

        print(f"  Subsampled → covered: {len(covered_flat)}, "
              f"train_fail: {len(train_fail_flat)}")

        analyze_velocity_lanes(
            model=model,
            sampler=sampler,
            covered_latents_flat=covered_flat,
            train_fail_latents_flat=train_fail_flat,
            gmm_means=gmm_means,
            target_k=target_k,
            save_path=save_base,
            device=device,
            latent_shape=latent_shape,
            label=0,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config_path", type=str,
                        default="model2_xl_vavae_f16d32.yaml")
    parser.add_argument("--ds_config_path", type=str,
                        default="model2_xl_vavae_f16d32.yaml")
    parser.add_argument("--vae_config_path", type=str,
                        default="model1_f16d32.yaml")
    parser.add_argument("--ckpt_path", type=str,
                        default="output/9th_lightningdit_xl_vavae_f16d32_gmm30_deterministic/checkpoints/0039440.pt")
    parser.add_argument("--generated_dir", type=str,
                        default="output/9th_lightningdit_xl_vavae_f16d32_gmm30_deterministic"
                                "/lightningdit-xl-1-ckpt-0039440-euler-40/class_0")
    parser.add_argument("--target_clusters", type=int, nargs='+',
                        default=[20, 6, 15])
    parser.add_argument("--max_samples_per_group", type=int, default=2000,
                        help="메모리/시간 절약을 위한 최대 샘플 수")
    parser.add_argument("--num_steps", type=int, default=40,
                        help="reverse ODE step 수")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args)