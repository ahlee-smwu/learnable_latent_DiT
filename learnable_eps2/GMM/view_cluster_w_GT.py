"""
t-SNE visualization:
- Real latents (cluster-colored)
- GMM cluster means (X marker)
- Generated image latents (+ marker)
"""

import os
import yaml
import pickle
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from datetime import datetime
from datasets.img_latent_dataset import ImgLatentDataset
from tokenizer.vavae import VA_VAE
from safetensors import safe_open
from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
import glob
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Glyph .* missing from font\(s\) DejaVu Sans."
)

# -------------------------------------------------
# Utils
# -------------------------------------------------
def make_distinct_colors(n):
    hues = np.linspace(0, 1, n, endpoint=False)
    return [mcolors.hsv_to_rgb((h, 0.85, 0.95)) for h in hues]

def _to_tensor(x):
    """numpy array 또는 torch tensor → float32 cpu tensor"""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x.float().cpu()

def _as_U_dD(comp, D):
    """(d,D) 또는 (D,d) → (d,D). train.py의 _as_U_dD와 동일."""
    comp = _to_tensor(comp)
    a, b = comp.shape
    if b == D:
        return comp
    if a == D:
        return comp.T
    raise ValueError(f"PCA components shape {comp.shape} incompatible with D={D}")

# -------------------------------------------------
# Real latents
# -------------------------------------------------
def collect_real_latents(loader, max_points=5000):
    latents, labels = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Collect real latents"):
            x = x.view(x.size(0), -1)
            latents.append(x.numpy())
            labels.append(y.numpy())

    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)

    if latents.shape[0] > max_points:
        idx = np.random.choice(latents.shape[0], max_points, replace=False)
        latents = latents[idx]
        labels = labels[idx]

    return latents, labels

def load_latent_stats(data_dir, device):
    stats_path = os.path.join(data_dir, "latents_stats.pt")
    stats = torch.load(stats_path, map_location=device)
    mean = stats["mean"].to(device)  # (1, C, 1, 1)
    std  = stats["std"].to(device)
    return mean, std

def normalize_latents(z, mean, std, latent_multiplier=1.0):
    """
    z: (B, C, H, W)  -- VAE posterior.sample()
    """
    z = (z - mean) / std
    z = z * latent_multiplier
    return z

# -------------------------------------------------
# Generated images → VAE latent
# -------------------------------------------------
def collect_generated_latents(
    args,
    ds_config,
    device,
    max_images=5000
):
    vae_wrapper = VA_VAE(args.config_path)
    vae = vae_wrapper.load().model.to(device)
    vae.eval()

    mean, std = load_latent_stats(ds_config["data"]["data_path"], device)
    latent_multiplier = ds_config["data"].get("latent_multiplier", 0.18215)

    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # ── ImageFolder 대신 직접 탐색 ────────────────────────────────────
    # cluster_* 폴더 안의 모든 png를 수집
    all_img_paths = []
    cluster_dirs = sorted(
        glob.glob(os.path.join(args.generated_dir, "cluster_*")),
        key=lambda p: int(os.path.basename(p).split("_")[1])  # 숫자 기준 정렬
    )

    for cluster_dir in cluster_dirs:
        paths = glob.glob(os.path.join(cluster_dir, "*.png"))
        all_img_paths.extend(paths)

    # 전체 셔플
    rng = np.random.default_rng(42)
    rng.shuffle(all_img_paths)

    if len(all_img_paths) > max_images:
        all_img_paths = all_img_paths[:max_images]

    print(f"[Generated] Total images to encode: {len(all_img_paths)}")

    # ── 배치 인코딩 ──────────────────────────────────────────────────
    latents_all = []
    batch_size = 512

    with torch.no_grad():
        for i in range(0, len(all_img_paths), batch_size):
            batch_paths = all_img_paths[i:i + batch_size]
            imgs = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                imgs.append(img_transform(img))
            imgs = torch.stack(imgs).to(device)

            z = vae.encode(imgs).sample()   # (B, C, H, W)
            z = (z - mean) / std
            z = z * latent_multiplier
            z = z.flatten(1)               # (B, D)

            latents_all.append(z.cpu().numpy())

            if (i // batch_size) % 50 == 0:
                print(f"[Generated] processed {min(i + batch_size, len(all_img_paths))}"
                      f"/{len(all_img_paths)}")

    latents_all = np.concatenate(latents_all, axis=0)
    print(f"[Generated] Final latent shape: {latents_all.shape}")
    return latents_all

# -------------------------------------------------
# t-SNE
# -------------------------------------------------
def tsne_real_gmm_generated(
    real_latents,      # (Nr, D)
    real_labels,       # (Nr,)
    gmm_means,         # dict: class -> (K, D)
    gen_latents,       # (Ng, D)
    save_path,
    pca_dim=128,
    seed=42
):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    np.random.seed(seed)

    # ------------------------------------------------
    # Flatten GMM means
    # ------------------------------------------------
    flat_means = []
    mean_cluster_ids = []

    for cls, means in gmm_means.items():
        for k in range(means.shape[0]):
            flat_means.append(means[k])
            mean_cluster_ids.append(f"{cls}_{k}")

    flat_means = np.stack(flat_means, axis=0)  # (Nm, D)

    # ------------------------------------------------
    # Assign real → nearest GMM mean (L2)
    # ------------------------------------------------
    dists = ((real_latents[:, None, :] - flat_means[None]) ** 2).sum(-1)
    assigns = dists.argmin(axis=1)
    cluster_ids = np.array(mean_cluster_ids)[assigns]

    # ------------------------------------------------
    # PCA (FIT ONLY on real + mean)
    # ------------------------------------------------
    X_pca_fit = np.concatenate([real_latents, flat_means], axis=0)
    pca = PCA(n_components=pca_dim, random_state=seed)
    pca.fit(X_pca_fit)

    real_pca = pca.transform(real_latents)
    mean_pca = pca.transform(flat_means)
    gen_pca  = pca.transform(gen_latents)

    # ------------------------------------------------
    # t-SNE (ON PCA SPACE)
    # ------------------------------------------------
    X_tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        verbose=1
    ).fit_transform(
        np.concatenate([real_pca, mean_pca, gen_pca], axis=0)
    )

    n_real = len(real_pca)
    n_mean = len(mean_pca)

    real_tsne = X_tsne[:n_real]
    mean_tsne = X_tsne[n_real:n_real + n_mean]
    gen_tsne  = X_tsne[n_real + n_mean:]

    # ------------------------------------------------
    # Colors (cluster-based)
    # ------------------------------------------------
    def make_distinct_colors(n):
        hues = np.linspace(0, 1, n, endpoint=False)
        return [(h, 0.85, 0.95) for h in hues]

    import matplotlib.colors as mcolors
    unique_clusters = sorted(set(mean_cluster_ids))
    colors = [mcolors.hsv_to_rgb(c) for c in make_distinct_colors(len(unique_clusters))]
    color_map = dict(zip(unique_clusters, colors))

    # ------------------------------------------------
    # Plot
    # ------------------------------------------------
    plt.figure(figsize=(11, 11))

    # Real latents
    for cid in unique_clusters:
        mask = cluster_ids == cid
        if mask.sum() == 0:
            continue
        plt.scatter(
            real_tsne[mask, 0],
            real_tsne[mask, 1],
            s=4,
            alpha=0.25,
            color=color_map[cid]
        )

    # GMM means
    for i, cid in enumerate(mean_cluster_ids):
        plt.scatter(
            mean_tsne[i, 0],
            mean_tsne[i, 1],
            s=140,
            marker="X",
            color=color_map[cid],
            edgecolors="black",
            linewidths=0.8
        )

    # Generated latents
    plt.scatter(
        gen_tsne[:, 0],
        gen_tsne[:, 1],
        s=18,
        marker="+",
        color="black",
        alpha=0.9,
        label="Generated"
    )

    plt.title("t-SNE: Real (clustered) + GMM Means + Generated")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Saved t-SNE: {save_path}")

# center score
def analyze_center_proximity(real_latents, gen_latents, flat_means, save_path):
    """
    flat_means : (Nm, D) — GMM centers (real 기준)
    각 real/gen point → 가장 가까운 center까지의 L2 거리를 비교
    """

    # 각 point → nearest center 거리
    real_dists = ((real_latents[:, None, :] - flat_means[None]) ** 2).sum(-1)  # (Nr, Nm)
    gen_dists  = ((gen_latents[:, None, :]  - flat_means[None]) ** 2).sum(-1)  # (Ng, Nm)

    real_dist = np.sqrt(real_dists.min(axis=1))  # (Nr,)
    gen_dist  = np.sqrt(gen_dists.min(axis=1))   # (Ng,)

    # 통계 검정
    ks_stat, ks_p = ks_2samp(real_dist, gen_dist)
    _, mw_p       = mannwhitneyu(real_dist, gen_dist, alternative='greater')
    wass          = wasserstein_distance(real_dist, gen_dist)
    median_ratio  = np.median(gen_dist) / (np.median(real_dist) + 1e-12)
    std_ratio     = gen_dist.std()      / (real_dist.std()      + 1e-12)

    # 수치 출력
    print("\n" + "=" * 60)
    print("  Center-Proximity Analysis")
    print("  GMM center ↔ Real pts  vs  GMM center ↔ Gen pts")
    print("=" * 60)
    for name, d in [("Real → nearest center", real_dist),
                    ("Gen  → nearest center", gen_dist)]:
        p = np.percentile(d, [10, 25, 50, 75, 90])
        print(f"\n  [{name}]")
        print(f"    n          = {len(d)}")
        print(f"    mean ± std = {d.mean():.4f} ± {d.std():.4f}")
        print(f"    P10/25/50/75/90 = {p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f}/{p[3]:.3f}/{p[4]:.3f}")

    print(f"\n  Median ratio  (Gen/Real) = {median_ratio:.4f}  {'⚠️  center 편향' if median_ratio < 0.8 else '✅'}")
    print(f"  Std ratio     (Gen/Real) = {std_ratio:.4f}  {'⚠️  다양성 부족' if std_ratio < 0.6 else '✅'}")
    print(f"  KS  stat / p            = {ks_stat:.4f} / {ks_p:.2e}  {'⚠️' if ks_p < 0.05 else '✅'}")
    print(f"  Mann-Whitney p          = {mw_p:.2e}  {'⚠️  gen이 center에 더 가까움' if mw_p < 0.05 else '✅'}")
    print(f"  Wasserstein             = {wass:.4f}")

    # 시각화
    C_REAL, C_GEN = '#1a6fa8', '#c0392b'  # 흰 배경용: 채도 낮춘 파랑/빨강

    fig = plt.figure(figsize=(18, 5), facecolor='white')
    fig.suptitle("GMM Center ↔ Real  vs  GMM Center ↔ Gen",
                 color='black', fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.33)

    def style(ax, title):
        ax.set_facecolor('white')
        ax.tick_params(colors='#333', labelsize=9)
        ax.set_title(title, color='black', fontsize=10.5, pad=6)
        for sp in ax.spines.values():
            sp.set_edgecolor('#cccccc')

    # 히스토그램
    ax = fig.add_subplot(gs[0])
    bins = np.linspace(0, max(real_dist.max(), gen_dist.max()) * 1.02, 60)
    ax.hist(real_dist, bins=bins, density=True, alpha=0.55, color=C_REAL,
            label=f'Real  med={np.median(real_dist):.3f}')
    ax.hist(gen_dist,  bins=bins, density=True, alpha=0.55, color=C_GEN,
            label=f'Gen   med={np.median(gen_dist):.3f}')
    ax.axvline(np.median(real_dist), color=C_REAL, lw=2, ls='--')
    ax.axvline(np.median(gen_dist),  color=C_GEN,  lw=2, ls='--')
    ax.set_xlabel('Distance to nearest center', color='#444')
    ax.set_ylabel('Density', color='#444')
    ax.legend(fontsize=9, framealpha=0.5, edgecolor='#ccc')
    style(ax, 'Distribution')

    # CDF
    ax = fig.add_subplot(gs[1])
    for dist, label, color in [(real_dist, 'Real', C_REAL), (gen_dist, 'Gen', C_GEN)]:
        sd = np.sort(dist)
        ax.plot(sd, np.arange(1, len(sd)+1) / len(sd), color=color, lw=2, label=label)
    ax.axhline(0.5, color='#888', lw=1, ls=':', alpha=0.6)
    ax.axvline(np.median(real_dist), color=C_REAL, lw=1.2, ls='--', alpha=0.7)
    ax.axvline(np.median(gen_dist),  color=C_GEN,  lw=1.2, ls='--', alpha=0.7)
    ax.set_xlabel('Distance to nearest center', color='#444')
    ax.set_ylabel('CDF', color='#444')
    ax.legend(fontsize=9, framealpha=0.5, edgecolor='#ccc')
    ax.grid(alpha=0.25, color='#cccccc')
    style(ax, 'CDF  (gen이 왼쪽 → center 편향)')

    # Violin
    ax = fig.add_subplot(gs[2])
    vp = ax.violinplot([real_dist, gen_dist], positions=[1, 2],
                       showmedians=True, showextrema=True)
    for body, c in zip(vp['bodies'], [C_REAL, C_GEN]):
        body.set_facecolor(c); body.set_alpha(0.55)
    vp['cmedians'].set_color('black'); vp['cmedians'].set_linewidth(2)
    for key in ['cmins', 'cmaxes', 'cbars']:
        vp[key].set_color('#555')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Real', 'Gen'], color='black', fontsize=11)
    ax.set_ylabel('Distance to nearest center', color='#444')
    style(ax, 'Violin')

    fig.savefig(os.path.join(save_path, 'center_analysis.svg'),
                format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ Saved: {save_path}")

def analyze_center_proximity_per_cluster(
    real_latents, gen_latents,
    flat_means,
    save_path,
    ncols=6,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance

    os.makedirs(save_path, exist_ok=True)

    C_REAL, C_GEN = '#1a6fa8', '#c0392b'  # 흰 배경용: 채도 낮춘 파랑/빨강

    # ════════════════════════════════════════════════════════════════════
    # STEP 1. 클러스터 라벨 할당 — 각 포인트의 nearest center index
    # ════════════════════════════════════════════════════════════════════
    real_labels = np.argmin(
        ((real_latents[:, None, :] - flat_means[None]) ** 2).sum(-1), axis=1
    )
    gen_labels = np.argmin(
        ((gen_latents[:, None, :] - flat_means[None]) ** 2).sum(-1), axis=1
    )

    n_clusters = flat_means.shape[0]
    nrows = int(np.ceil(n_clusters / ncols))

    # ════════════════════════════════════════════════════════════════════
    # STEP 2. 모든 클러스터 통계 검정
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Center-Proximity Analysis — Per Cluster Statistical Tests")
    print("=" * 70)

    cluster_data = {}
    for k in range(n_clusters):
        r_pts = real_latents[real_labels == k]
        g_pts = gen_latents[gen_labels == k]

        if len(r_pts) == 0 or len(g_pts) == 0:
            cluster_data[k] = None
            print(f"\n  [Cluster {k:02d}]  ⚠️  데이터 없음 (nr={len(r_pts)}, ng={len(g_pts)})")
            continue

        r_dists = np.sqrt(((r_pts[:, None, :] - flat_means[None]) ** 2).sum(-1).min(axis=1))
        g_dists = np.sqrt(((g_pts[:, None, :] - flat_means[None]) ** 2).sum(-1).min(axis=1))

        ks_stat, ks_p = ks_2samp(r_dists, g_dists)
        _, mw_p       = mannwhitneyu(r_dists, g_dists, alternative='greater')
        wass          = wasserstein_distance(r_dists, g_dists)
        med_ratio     = np.median(g_dists) / (np.median(r_dists) + 1e-12)
        std_ratio     = g_dists.std()      / (r_dists.std()      + 1e-12)

        cluster_data[k] = dict(
            r=r_dists, g=g_dists,
            ks_stat=ks_stat, ks_p=ks_p,
            mw_p=mw_p, wass=wass,
            med_ratio=med_ratio, std_ratio=std_ratio,
        )

        print(f"\n  [Cluster {k:02d}]  nr={len(r_pts)}  ng={len(g_pts)}")
        for name, d in [("Real → nearest center", r_dists),
                        ("Gen  → nearest center", g_dists)]:
            p = np.percentile(d, [10, 25, 50, 75, 90])
            print(f"    [{name}]  mean±std={d.mean():.4f}±{d.std():.4f}  "
                  f"P10/50/90={p[0]:.3f}/{p[2]:.3f}/{p[4]:.3f}")
        print(f"    Median ratio  (Gen/Real) = {med_ratio:.4f}  "
              f"{'⚠️  center 편향' if med_ratio < 0.8 else '✅'}")
        print(f"    Std ratio     (Gen/Real) = {std_ratio:.4f}  "
              f"{'⚠️  다양성 부족' if std_ratio < 0.6 else '✅'}")
        print(f"    KS  stat/p  = {ks_stat:.4f}/{ks_p:.2e}  "
              f"{'⚠️' if ks_p < 0.05 else '✅'}  "
              f"Mann-Whitney p = {mw_p:.2e}  "
              f"{'⚠️  gen이 center에 더 가까움' if mw_p < 0.05 else '✅'}  "
              f"Wasserstein = {wass:.4f}")

    print("\n" + "=" * 70)
    print("  통계 검정 완료. 이미지 저장 시작...")
    print("=" * 70)

    # ── 공통 axes 스타일 ──────────────────────────────────────────────────
    def _style(ax, title, subtitle=''):
        ax.set_facecolor('white')
        ax.tick_params(colors='#333', labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor('#cccccc')
        full_title = f'{title}\n{subtitle}' if subtitle else title
        ax.set_title(full_title, color='black', fontsize=7, pad=3, linespacing=1.3)

    def _empty(ax, k):
        ax.set_facecolor('white')
        ax.text(0.5, 0.5, f'Cluster {k}\n(no data)',
                ha='center', va='center', color='#aaa', fontsize=7,
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor('#cccccc')

    # ════════════════════════════════════════════════════════════════════
    # STEP 3. 히스토그램 그리드
    # ════════════════════════════════════════════════════════════════════
    fig_h, axes_h = plt.subplots(
        nrows, ncols, figsize=(ncols * 2.8, nrows * 2.4),
        facecolor='white',
    )
    fig_h.suptitle('Center-Proximity — Histogram  (Real vs Gen, per cluster)',
                   color='black', fontsize=13, fontweight='bold', y=1.01)
    axes_h = axes_h.flatten()

    for k in range(n_clusters):
        ax = axes_h[k]
        d  = cluster_data[k]
        if d is None:
            _empty(ax, k); continue

        r, g = d['r'], d['g']
        bins = np.linspace(0, max(r.max(), g.max()) * 1.02, 40)
        ax.hist(r, bins=bins, density=True, alpha=0.55, color=C_REAL,
                label=f'R {np.median(r):.2f}')
        ax.hist(g, bins=bins, density=True, alpha=0.55, color=C_GEN,
                label=f'G {np.median(g):.2f}')
        ax.axvline(np.median(r), color=C_REAL, lw=1.2, ls='--')
        ax.axvline(np.median(g), color=C_GEN,  lw=1.2, ls='--')
        ax.legend(fontsize=6, loc='upper right', framealpha=0.5, edgecolor='#ccc')
        ax.set_xlabel('Dist to nearest center', color='#444', fontsize=6)

        warn = '⚠' if d['med_ratio'] < 0.8 or d['ks_p'] < 0.05 else '✓'
        _style(ax, f'Cluster {k:02d}  {warn}',
               f'med_r={d["med_ratio"]:.2f} ks_p={d["ks_p"]:.1e}')

    for ax in axes_h[n_clusters:]:
        ax.set_visible(False)

    fig_h.tight_layout()
    path_h = os.path.join(save_path, 'proximity_histogram_grid.svg')
    fig_h.savefig(path_h, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig_h)
    print(f'\n✅ [1/3] Saved histogram grid → {path_h}')

    # ════════════════════════════════════════════════════════════════════
    # STEP 4. CDF 그리드
    # ════════════════════════════════════════════════════════════════════
    fig_c, axes_c = plt.subplots(
        nrows, ncols, figsize=(ncols * 2.8, nrows * 2.4),
        facecolor='white',
    )
    fig_c.suptitle('Center-Proximity — CDF  (Real vs Gen, per cluster)',
                   color='black', fontsize=13, fontweight='bold', y=1.01)
    axes_c = axes_c.flatten()

    for k in range(n_clusters):
        ax = axes_c[k]
        d  = cluster_data[k]
        if d is None:
            _empty(ax, k); continue

        for dist, label, color in [(d['r'], 'Real', C_REAL), (d['g'], 'Gen', C_GEN)]:
            sd = np.sort(dist)
            ax.plot(sd, np.arange(1, len(sd) + 1) / len(sd),
                    color=color, lw=1.3, label=label)
        ax.axhline(0.5, color='#888', lw=0.8, ls=':', alpha=0.6)
        ax.axvline(np.median(d['r']), color=C_REAL, lw=1.0, ls='--', alpha=0.6)
        ax.axvline(np.median(d['g']), color=C_GEN,  lw=1.0, ls='--', alpha=0.6)
        ax.legend(fontsize=6, loc='lower right', framealpha=0.5, edgecolor='#ccc')
        ax.set_xlabel('Dist to nearest center', color='#444', fontsize=6)
        ax.set_ylabel('CDF', color='#444', fontsize=6)
        ax.grid(alpha=0.20, color='#cccccc')

        warn = '⚠' if d['mw_p'] < 0.05 else '✓'
        _style(ax, f'Cluster {k:02d}  {warn}',
               f'mw_p={d["mw_p"]:.1e} W={d["wass"]:.3f}')

    for ax in axes_c[n_clusters:]:
        ax.set_visible(False)

    fig_c.tight_layout()
    path_c = os.path.join(save_path, 'proximity_cdf_grid.svg')
    fig_c.savefig(path_c, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig_c)
    print(f'✅ [2/3] Saved CDF grid       → {path_c}')

    # ════════════════════════════════════════════════════════════════════
    # STEP 5. 바이올린 그리드
    # ════════════════════════════════════════════════════════════════════
    fig_v, axes_v = plt.subplots(
        nrows, ncols, figsize=(ncols * 2.8, nrows * 2.4),
        facecolor='white',
    )
    fig_v.suptitle('Center-Proximity — Violin  (Real vs Gen, per cluster)',
                   color='black', fontsize=13, fontweight='bold', y=1.01)
    axes_v = axes_v.flatten()

    for k in range(n_clusters):
        ax = axes_v[k]
        d  = cluster_data[k]
        if d is None:
            _empty(ax, k); continue

        vp = ax.violinplot(
            [d['r'], d['g']], positions=[1, 2],
            showmedians=True, showextrema=True,
        )
        for body, c in zip(vp['bodies'], [C_REAL, C_GEN]):
            body.set_facecolor(c); body.set_alpha(0.55)
        vp['cmedians'].set_color('black'); vp['cmedians'].set_linewidth(1.5)
        for key in ['cmins', 'cmaxes', 'cbars']:
            vp[key].set_color('#555'); vp[key].set_linewidth(0.8)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Real', 'Gen'], color='black', fontsize=7)
        ax.set_ylabel('Dist to nearest center', color='#444', fontsize=6)

        warn = '⚠' if d['std_ratio'] < 0.6 else '✓'
        _style(ax, f'Cluster {k:02d}  {warn}',
               f'std_r={d["std_ratio"]:.2f} med_r={d["med_ratio"]:.2f}')

    for ax in axes_v[n_clusters:]:
        ax.set_visible(False)

    fig_v.tight_layout()
    path_v = os.path.join(save_path, 'proximity_violin_grid.svg')
    fig_v.savefig(path_v, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig_v)
    print(f'✅ [3/3] Saved violin grid    → {path_v}')

    return cluster_data

# ─────────────────────────────────────────────────────────────
# 핵심 유틸: GMM posterior r_k(x) 계산
# ─────────────────────────────────────────────────────────────
def compute_gmm_posterior(
    latents,
    gmm_means,
    gmm_covs,
    gmm_weights,
    gmm_pca,
    use_weight=True,
    eps=1e-8,
):
    """
    각 x ∈ latents에 대해 r_k(x) = P(k|x) 계산.
    get_cluster_gmm과 동일한 수치 연산.

    gmm_pca 형식:
        ckpt["pca_components"] 그대로 사용 가능
        {cls: {"components": (d,D)or(D,d), "mean": (D,)}}
        numpy array 또는 torch tensor 모두 허용.
    """

    # ── 모든 클래스 파라미터를 하나로 합치기 ─────────────────────────
    all_means    = []
    all_covs     = []
    all_weights  = []
    all_U        = []
    all_pca_mean = []
    cluster_info = []

    for cls in sorted(gmm_means.keys()):
        means   = _to_tensor(gmm_means[cls])    # (K, D)
        covs    = _to_tensor(gmm_covs[cls])     # (K, d)
        weights = _to_tensor(gmm_weights[cls])  # (K,)

        pca_dict = gmm_pca[cls]
        D        = means.shape[1]
        U        = _as_U_dD(_to_tensor(pca_dict["components"]), D)  # (d, D)
        pca_mean = _to_tensor(pca_dict["mean"])                      # (D,)

        K = means.shape[0]
        for k in range(K):
            all_means.append(means[k])
            all_covs.append(covs[k])
            all_weights.append(weights[k])
            all_U.append(U)
            all_pca_mean.append(pca_mean)
            cluster_info.append({"cls": cls, "k_local": k})

    K_total = len(all_means)

    means_t   = torch.stack(all_means,   dim=0)   # (K_total, D)
    covs_t    = torch.stack(all_covs,    dim=0)   # (K_total, d)
    weights_t = torch.stack(all_weights, dim=0)   # (K_total,)

    # LSUN-church는 class 0 하나 → U와 pca_mean이 모든 클러스터 동일
    U        = all_U[0]         # (d, D)
    pca_mean = all_pca_mean[0]  # (D,)

    # ── latent → tensor ──────────────────────────────────────────────
    x = _to_tensor(latents) if isinstance(latents, np.ndarray) \
        else latents.float().cpu()   # (N, D)

    # ── PCA 공간 투영 ─────────────────────────────────────────────────
    x_pca     = (x - pca_mean) @ U.T          # (N, d)
    means_pca = (means_t - pca_mean) @ U.T    # (K_total, d)

    # ── Mahalanobis + log_det (get_cluster_gmm과 동일) ───────────────
    v_safe  = torch.clamp(covs_t, min=eps)                          # (K_total, d)
    diff    = x_pca[:, None, :] - means_pca[None, :, :]            # (N, K_total, d)
    mahal   = (diff * diff / v_safe[None, :, :]).sum(dim=2)        # (N, K_total)
    log_det = torch.log(v_safe).sum(dim=1)                          # (K_total,)

    log_prob = -0.5 * (mahal + log_det[None, :])                   # (N, K_total)
    if use_weight:
        log_prob = log_prob + torch.log(weights_t[None, :] + eps)  # + log πk

    # ── Posterior ────────────────────────────────────────────────────
    posteriors  = F.softmax(log_prob.double(), dim=1).float()   # (N, K_total)
    assignments = posteriors.argmax(dim=1)     # (N,)

    return (
        posteriors.numpy(),    # (N, K_total)
        assignments.numpy(),   # (N,)
        log_prob.numpy(),      # (N, K_total)
        cluster_info,          # list of {"cls", "k_local"}
    )

def analyze_posterior_coverage(
    real_latents,   # (Nr, D) numpy
    gen_latents,    # (Ng, D) numpy
    gmm_means,
    gmm_covs,
    gmm_weights,
    gmm_pca,
    save_path,
    use_weight=True,
    eps=1e-8,
    ncols=6,
):
    """
    Posterior r_k(x) 기반으로 실제 데이터와 생성 데이터의 커버리지를 분석.

    분석 1. 전체 요약
        - 각 클러스터 k에 real/gen이 몇 개 할당됐는지
        - 클러스터별 MAP-posterior 값의 분포 (얼마나 확신하는가)
        - gen이 커버 못 한 클러스터 리스트

    분석 2. 클러스터별 posterior 분포 비교
        - r_k(x)의 히스토그램 (real vs gen)
        - CDF, violin

    분석 3. Entropy 기반 다양성
        - H(r(x)) = -Σk r_k(x) log r_k(x)
        - 낮으면 → x가 한 클러스터에 강하게 할당됨 (선명하지만 좁음)
        - 높으면 → x가 여러 클러스터에 걸침 (경계 포인트)
        - real vs gen의 entropy 분포 비교
    """
    os.makedirs(save_path, exist_ok=True)
    C_REAL, C_GEN = '#1a6fa8', '#c0392b'

    # ── 1. Posterior 계산 ─────────────────────────────────────────────
    print("Computing GMM posteriors for real latents...")
    real_post, real_assign, real_logp, cluster_info = compute_gmm_posterior(
        real_latents, gmm_means, gmm_covs, gmm_weights, gmm_pca,
        use_weight=use_weight, eps=eps,
    )
    print("Computing GMM posteriors for gen latents...")
    gen_post, gen_assign, gen_logp, _ = compute_gmm_posterior(
        gen_latents, gmm_means, gmm_covs, gmm_weights, gmm_pca,
        use_weight=use_weight, eps=eps,
    )

    K_total = real_post.shape[1]
    Nr, Ng  = len(real_latents), len(gen_latents)

    # ── 2. 클러스터별 할당 수 ────────────────────────────────────────
    real_counts = np.bincount(real_assign, minlength=K_total)  # (K,)
    gen_counts  = np.bincount(gen_assign,  minlength=K_total)  # (K,)

    # ── 3. 커버리지 요약 출력 ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Posterior-based Coverage Analysis")
    print(f"  Nr={Nr}  Ng={Ng}  K_total={K_total}  use_weight={use_weight}")
    print("=" * 70)
    print(f"\n{'Cluster':>10} {'cls':>5} {'k':>4} | {'Real N':>8} {'Real%':>7} | "
          f"{'Gen N':>8} {'Gen%':>7} | {'Gen/Real':>9} | {'Status':>10}")
    print("-" * 70)

    uncovered = []   # gen이 0개인 클러스터
    sparse    = []   # gen/real 비율이 0.3 미만

    for k, info in enumerate(cluster_info):
        r_n  = real_counts[k]
        g_n  = gen_counts[k]
        r_pct = 100.0 * r_n / Nr
        g_pct = 100.0 * g_n / Ng
        ratio = g_n / (r_n + eps)

        if g_n == 0:
            status = "❌ uncovered"
            uncovered.append(k)
        elif ratio < 0.3:
            status = "⚠️  sparse"
            sparse.append(k)
        else:
            status = "✅"

        print(f"  k={k:>4}   {info['cls']:>4} {info['k_local']:>4} | "
              f"{r_n:>8} {r_pct:>6.1f}% | "
              f"{g_n:>8} {g_pct:>6.1f}% | "
              f"{ratio:>9.3f} | {status}")

    print(f"\n  Uncovered clusters ({len(uncovered)}): {uncovered}")
    print(f"  Sparse clusters    ({len(sparse)}): {sparse}")

    # ── 4. MAP posterior 값 (얼마나 확신하는가) ────────────────────────
    # 각 포인트의 argmax 클러스터에서의 posterior 값
    real_map_conf = real_post[np.arange(Nr), real_assign]  # (Nr,)
    gen_map_conf  = gen_post[np.arange(Ng),  gen_assign]   # (Ng,)

    # ── 5. Entropy 계산 ──────────────────────────────────────────────
    def entropy(p, ax=1):
        p_safe = np.clip(p, 1e-12, 1.0)
        return -(p_safe * np.log(p_safe)).sum(axis=ax)

    real_entropy = entropy(real_post)   # (Nr,)
    gen_entropy  = entropy(gen_post)    # (Ng,)

    # ── 6. 전체 요약 시각화 ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='white')
    fig.suptitle("Posterior-based Coverage: Real vs Generated",
                 color='black', fontsize=13, fontweight='bold')

    # 6-1. 클러스터별 할당 수 bar chart
    ax = axes[0]
    x_pos = np.arange(K_total)
    ax.bar(x_pos - 0.2, real_counts / Nr * 100, 0.4,
           label='Real', color=C_REAL, alpha=0.7)
    ax.bar(x_pos + 0.2, gen_counts  / Ng * 100, 0.4,
           label='Gen',  color=C_GEN,  alpha=0.7)
    ax.set_xlabel('Cluster index', color='#444')
    ax.set_ylabel('Assignment %', color='#444')
    ax.set_title('Cluster assignment ratio\n(Real vs Gen)', fontsize=10)
    ax.legend(fontsize=9)
    ax.set_facecolor('white')

    # 6-2. MAP confidence 분포 (얼마나 확신하는 클러스터에 할당됐는가)
    ax = axes[1]
    bins = np.linspace(0, 1, 50)
    ax.hist(real_map_conf, bins=bins, density=True, alpha=0.6,
            color=C_REAL, label=f'Real  med={np.median(real_map_conf):.3f}')
    ax.hist(gen_map_conf,  bins=bins, density=True, alpha=0.6,
            color=C_GEN,  label=f'Gen   med={np.median(gen_map_conf):.3f}')
    ax.set_xlabel('MAP posterior r_k*(x)', color='#444')
    ax.set_ylabel('Density', color='#444')
    ax.set_title('MAP confidence\n(높을수록 클러스터에 명확히 속함)', fontsize=10)
    ax.legend(fontsize=9)
    ax.set_facecolor('white')

    # 6-3. Entropy 분포 (낮으면 collapse, 높으면 boundary)
    ax = axes[2]
    max_ent = np.log(K_total)  # 균등 분포일 때 최대 entropy
    bins_e  = np.linspace(0, max_ent, 50)
    ax.hist(real_entropy, bins=bins_e, density=True, alpha=0.6,
            color=C_REAL, label=f'Real  med={np.median(real_entropy):.3f}')
    ax.hist(gen_entropy,  bins=bins_e, density=True, alpha=0.6,
            color=C_GEN,  label=f'Gen   med={np.median(gen_entropy):.3f}')
    ax.axvline(max_ent, color='gray', ls='--', lw=1, alpha=0.5, label=f'max H={max_ent:.2f}')
    ax.set_xlabel('Posterior entropy H(r(x))', color='#444')
    ax.set_ylabel('Density', color='#444')
    ax.set_title('Posterior entropy\n(낮으면 collapse, 높으면 경계 포인트)', fontsize=10)
    ax.legend(fontsize=9)
    ax.set_facecolor('white')

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'posterior_coverage_summary.svg'),
                format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("\n✅ Saved: posterior_coverage_summary.svg")

    # ── 7. 클러스터별 posterior r_k(x) 분포 비교 ─────────────────────
    # 각 클러스터 k에 대해: real/gen 포인트들의 r_k(x) 값 분포
    # r_k(x)가 낮은 real 포인트 = k 클러스터에 약하게 속함 = 경계 포인트
    # gen의 r_k(x) 분포가 real보다 오른쪽으로 쏠림
    #   = gen이 k 클러스터의 중심부만 생성함 (collapse)
    nrows = int(np.ceil(K_total / ncols))

    fig_p, axes_p = plt.subplots(nrows, ncols,
                                  figsize=(ncols * 2.8, nrows * 2.4),
                                  facecolor='white')
    fig_p.suptitle('r_k(x): Posterior of cluster k  (Real vs Gen)',
                   color='black', fontsize=13, fontweight='bold', y=1.01)
    axes_p = axes_p.flatten()

    cluster_stats = {}
    for k in range(K_total):
        ax = axes_p[k]
        ax.set_facecolor('white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#cccccc')

        r_vals = real_post[:, k]   # (Nr,) — 모든 real 포인트의 k에 대한 posterior
        g_vals = gen_post[:, k]    # (Ng,)

        bins = np.linspace(0, max(r_vals.max(), g_vals.max()) * 1.02, 40)
        ax.hist(r_vals, bins=bins, density=True, alpha=0.55,
                color=C_REAL, label=f'R med={np.median(r_vals):.3f}')
        ax.hist(g_vals, bins=bins, density=True, alpha=0.55,
                color=C_GEN,  label=f'G med={np.median(g_vals):.3f}')
        ax.axvline(np.median(r_vals), color=C_REAL, lw=1.2, ls='--')
        ax.axvline(np.median(g_vals), color=C_GEN,  lw=1.2, ls='--')
        ax.legend(fontsize=6, loc='upper right', framealpha=0.5)
        ax.set_xlabel(f'r_{k}(x)', color='#444', fontsize=6)

        # 통계
        ks_stat, ks_p = ks_2samp(r_vals, g_vals)
        med_ratio = np.median(g_vals) / (np.median(r_vals) + eps)

        # gen median이 real보다 높으면 → gen이 이 클러스터 중심부에 집중
        # gen median이 real보다 낮으면 → gen이 이 클러스터를 회피
        warn = '⚠' if ks_p < 0.05 else '✓'
        direction = '↑집중' if med_ratio > 1.2 else ('↓회피' if med_ratio < 0.8 else '균형')
        ax.set_title(f'k={k:02d} {warn} {direction}\nks_p={ks_p:.1e}',
                     color='black', fontsize=7, pad=3)

        cluster_stats[k] = dict(
            real_median_rk=np.median(r_vals),
            gen_median_rk=np.median(g_vals),
            med_ratio=med_ratio,
            ks_stat=ks_stat, ks_p=ks_p,
            direction=direction,
            real_assign_n=int(real_counts[k]),
            gen_assign_n=int(gen_counts[k]),
        )

    for ax in axes_p[K_total:]:
        ax.set_visible(False)

    fig_p.tight_layout()
    fig_p.savefig(os.path.join(save_path, 'posterior_per_cluster.svg'),
                  format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig_p)
    print("✅ Saved: posterior_per_cluster.svg")

    # ── 8. 최종 요약 출력 ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Cluster-wise Summary (sorted by gen_assign_n)")
    print(f"  {'k':>4} | {'real_N':>7} {'gen_N':>7} | "
          f"{'real_med_rk':>12} {'gen_med_rk':>12} | "
          f"{'direction':>8} | {'ks_p':>10}")
    print("-" * 70)
    for k in sorted(cluster_stats, key=lambda k: cluster_stats[k]['gen_assign_n']):
        s = cluster_stats[k]
        print(f"  {k:>4} | {s['real_assign_n']:>7} {s['gen_assign_n']:>7} | "
              f"{s['real_median_rk']:>12.5f} {s['gen_median_rk']:>12.5f} | "
              f"{s['direction']:>8} | {s['ks_p']:>10.2e}")

    return {
        'real_post':     real_post,      # (Nr, K_total)
        'gen_post':      gen_post,       # (Ng, K_total)
        'real_assign':   real_assign,    # (Nr,)
        'gen_assign':    gen_assign,     # (Ng,)
        'real_entropy':  real_entropy,   # (Nr,)
        'gen_entropy':   gen_entropy,    # (Ng,)
        'cluster_info':  cluster_info,
        'cluster_stats': cluster_stats,
        'real_counts':   real_counts,
        'gen_counts':    gen_counts,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 추가 분석: "k를 넣었는데 j가 나왔다"
# 폴더 구조: generated_dir/class_{cls}/cluster_{k}/xxx.png
# ─────────────────────────────────────────────────────────────────────────────

def collect_gen_latents_with_source_cluster(
    generated_dir,
    vae_wrapper,
    ds_config,
    device,
    max_images_per_cluster=500,
):
    """
    cluster_k 폴더에서 이미지를 읽어 VAE 인코딩 후
    source cluster 레이블과 함께 반환.

    반환:
        gen_latents      : (N, D) numpy  — 전체 생성 latent
        gen_source_k     : (N,)   int    — 어떤 cluster prior로 생성됐는가
    """
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import glob

    vae = vae_wrapper.load().model.to(device)
    vae.eval()

    mean, std = load_latent_stats(ds_config["data"]["data_path"], device)
    latent_multiplier = ds_config["data"].get("latent_multiplier", 0.18215)

    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # ── 클러스터 폴더 탐색 ───────────────────────────────────────────
    # generated_dir/class_*/cluster_*/
    cluster_dirs = sorted(glob.glob(
        os.path.join(generated_dir, "cluster_*")
    ))

    all_latents  = []
    all_source_k = []

    for cluster_dir in cluster_dirs:
        # 폴더명에서 cluster id 파싱: "cluster_3" → 3
        k = int(os.path.basename(cluster_dir).split("_")[1])

        img_paths = glob.glob(os.path.join(cluster_dir, "*.png"))
        rng = np.random.default_rng(42)
        rng.shuffle(img_paths)

        if len(img_paths) == 0:
            continue
        if len(img_paths) > max_images_per_cluster:
            rng = np.random.default_rng(42)
            img_paths = list(rng.choice(img_paths, max_images_per_cluster, replace=False))

        # 배치 인코딩
        batch_size = 64
        latents_k = []

        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i:i + batch_size]
            imgs = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                imgs.append(img_transform(img))
            imgs = torch.stack(imgs).to(device)

            with torch.no_grad():
                z = vae.encode(imgs).sample()      # (B, C, H, W)
                z = (z - mean) / std
                z = z * latent_multiplier
                z = z.flatten(1)                   # (B, D)

            latents_k.append(z.cpu().numpy())

        latents_k = np.concatenate(latents_k, axis=0)  # (n_k, D)
        source_k  = np.full(len(latents_k), k, dtype=np.int64)

        all_latents.append(latents_k)
        all_source_k.append(source_k)
        print(f"  cluster {k:02d}: {len(latents_k)} images encoded")

    gen_latents  = np.concatenate(all_latents,  axis=0)   # (N, D)
    gen_source_k = np.concatenate(all_source_k, axis=0)   # (N,)

    print(f"\n[Gen] Total: {len(gen_latents)} latents from {len(cluster_dirs)} clusters")
    return gen_latents, gen_source_k


def analyze_posterior_drift(
    gen_latents,       # (N, D) numpy — 생성 latent
    gen_source_k,      # (N,)   int   — 어떤 cluster prior로 생성됐는가
    gmm_means,
    gmm_covs,
    gmm_weights,
    gmm_pca,
    save_path,
    use_weight=True,
    eps=1e-8,
    ncols=6,
):
    """
    "k를 넣었는데 posterior argmax가 j였다" 분석.

    기존 analyze_posterior_coverage()의 결과에 추가하는 분석.

    분석 1. Confusion matrix
        행 = source cluster k (prior로 넣은 것)
        열 = posterior argmax j (생성 결과가 어디로 갔는가)
        값 = 비율

    분석 2. 클러스터별 drift 통계
        - self-consistency: source k = argmax k 인 비율 (높을수록 좋음)
        - top-drift: 가장 많이 흘러간 j와 그 비율
        - r_k(x) at source k vs r_j(x) at argmax j 차이

    분석 3. 클러스터별 r_source(x) 분포
        - 각 k에서 생성된 샘플들의 r_k(x) 값
        - 높으면 → k 클러스터 중심에 맞는 샘플 생성
        - 낮으면 → k를 넣었지만 k답지 않은 샘플 생성 (drift)
    """
    os.makedirs(save_path, exist_ok=True)
    C_REAL, C_GEN = '#1a6fa8', '#c0392b'

    # ── 1. Posterior 계산 ─────────────────────────────────────────────
    print("\nComputing GMM posteriors for gen latents (with source cluster)...")
    gen_post, gen_argmax, _, cluster_info = compute_gmm_posterior(
        gen_latents, gmm_means, gmm_covs, gmm_weights, gmm_pca,
        use_weight=use_weight, eps=eps,
    )
    # gen_post  : (N, K_total)
    # gen_argmax: (N,)  — posterior 기준 가장 그럴듯한 클러스터

    K_total  = gen_post.shape[1]
    N        = len(gen_latents)
    k_values = sorted(np.unique(gen_source_k).tolist())

    # ── 2. Confusion matrix 계산 ──────────────────────────────────────
    # conf[k, j] = source k로 생성된 샘플 중 argmax가 j인 비율
    conf = np.zeros((K_total, K_total), dtype=np.float32)
    count_per_source = np.zeros(K_total, dtype=np.int64)

    for k in k_values:
        mask_k = gen_source_k == k
        n_k    = mask_k.sum()
        if n_k == 0:
            continue
        count_per_source[k] = n_k
        argmax_k = gen_argmax[mask_k]                    # (n_k,)
        for j in range(K_total):
            conf[k, j] = (argmax_k == j).sum() / n_k    # 비율

    # ── 3. 수치 출력 ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  Posterior Drift Analysis: 'k를 넣었는데 j가 나왔다'")
    print(f"  N={N}  K_total={K_total}  use_weight={use_weight}")
    print("=" * 80)
    print(f"\n{'src_k':>6} | {'n':>6} | {'self%':>7} | "
          f"{'top_j':>6} {'top%':>7} | "
          f"{'r_src med':>10} {'r_argmax med':>13} | {'drift_gap':>10}")
    print("-" * 80)

    drift_summary = {}

    for k in k_values:
        mask_k  = gen_source_k == k
        n_k     = mask_k.sum()
        if n_k == 0:
            continue

        argmax_k = gen_argmax[mask_k]                    # (n_k,)
        post_k   = gen_post[mask_k]                      # (n_k, K_total)

        # self-consistency
        self_rate = (argmax_k == k).mean() * 100         # %

        # top drift destination (k 제외)
        conf_row = conf[k].copy()
        conf_row[k] = 0.0                                # self 제외
        top_j    = int(conf_row.argmax())
        top_rate = conf_row[top_j] * 100

        # r_source(x): 생성 샘플의 source cluster k에서의 posterior 값
        r_source  = post_k[:, k]                         # (n_k,)
        # r_argmax(x): 각 샘플의 argmax 클러스터에서의 posterior 값
        r_argmax  = post_k[np.arange(n_k), argmax_k]    # (n_k,)
        # drift_gap = r_argmax - r_source  (양수 → argmax 쪽이 더 높음 = drift 발생)
        drift_gap = (r_argmax - r_source).mean()

        print(f"  k={k:>3}  | {n_k:>6} | {self_rate:>6.1f}% | "
              f"j={top_j:>3} {top_rate:>6.1f}% | "
              f"{np.median(r_source):>10.4f} {np.median(r_argmax):>13.4f} | "
              f"{drift_gap:>+10.4f}")

        drift_summary[k] = dict(
            n=n_k,
            self_rate=self_rate,
            top_j=top_j,
            top_rate=top_rate,
            r_source_median=np.median(r_source),
            r_argmax_median=np.median(r_argmax),
            drift_gap=drift_gap,
            conf_row=conf[k],
            r_source_vals=r_source,
            r_argmax_vals=r_argmax,
        )

    # ── 4. Confusion matrix 시각화 ────────────────────────────────────
    fig_c, ax = plt.subplots(1, 1, figsize=(12, 10), facecolor='white')
    active_k = [k for k in k_values if count_per_source[k] > 0]

    # 사용된 클러스터만 표시
    conf_sub = conf[np.ix_(active_k, active_k)]

    im = ax.imshow(conf_sub, cmap='Blues', vmin=0, vmax=conf_sub.max())
    ax.set_xticks(range(len(active_k)))
    ax.set_yticks(range(len(active_k)))
    ax.set_xticklabels([f'j={j}' for j in active_k], rotation=90, fontsize=7)
    ax.set_yticklabels([f'k={k}' for k in active_k], fontsize=7)
    ax.set_xlabel('Posterior argmax j  (결과가 어디로 갔나)', color='#444')
    ax.set_ylabel('Source cluster k  (무엇을 넣었나)', color='#444')
    ax.set_title('Posterior Drift Confusion Matrix\n'
                 '행=source k, 열=argmax j, 값=비율\n'
                 '대각선=self-consistent, 비대각선=drift',
                 fontsize=10, color='black')

    # 대각선 강조 및 수치 표시
    for i, ki in enumerate(active_k):
        for j, kj in enumerate(active_k):
            val = conf_sub[i, j]
            if val > 0.05:  # 5% 이상만 수치 표시
                color = 'white' if val > conf_sub.max() * 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label='Fraction')
    fig_c.tight_layout()
    fig_c.savefig(os.path.join(save_path, 'posterior_drift_confusion.svg'),
                  format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig_c)
    print("\n✅ Saved: posterior_drift_confusion.svg")

    # ── 5. 클러스터별 r_source vs r_argmax 분포 ───────────────────────
    nrows = int(np.ceil(len(active_k) / ncols))
    fig_r, axes_r = plt.subplots(nrows, ncols,
                                  figsize=(ncols * 2.8, nrows * 2.4),
                                  facecolor='white')
    fig_r.suptitle('r_source(x) vs r_argmax(x) per cluster\n'
                   '파랑=r_k(x) (넣은 클러스터의 posterior), '
                   '빨강=r_j(x) (argmax 클러스터의 posterior)',
                   fontsize=10, fontweight='bold', y=1.01)
    axes_r = axes_r.flatten()

    for idx, k in enumerate(active_k):
        ax = axes_r[idx]
        ax.set_facecolor('white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#cccccc')

        s = drift_summary[k]
        r_src = s['r_source_vals']
        r_arg = s['r_argmax_vals']

        bins = np.linspace(0, max(r_src.max(), r_arg.max()) * 1.02, 40)
        ax.hist(r_src, bins=bins, density=True, alpha=0.6,
                color=C_REAL, label=f'r_src med={np.median(r_src):.3f}')
        ax.hist(r_arg, bins=bins, density=True, alpha=0.6,
                color=C_GEN,  label=f'r_max med={np.median(r_arg):.3f}')
        ax.axvline(np.median(r_src), color=C_REAL, lw=1.2, ls='--')
        ax.axvline(np.median(r_arg), color=C_GEN,  lw=1.2, ls='--')
        ax.legend(fontsize=5.5, loc='upper left', framealpha=0.4)

        warn = '⚠' if s['self_rate'] < 50 else '✓'
        ax.set_title(
            f'k={k:02d} {warn}\n'
            f'self={s["self_rate"]:.0f}% drift→j={s["top_j"]}({s["top_rate"]:.0f}%)',
            fontsize=7, color='black', pad=3
        )
        ax.set_xlabel('Posterior value', color='#444', fontsize=6)
        ax.tick_params(labelsize=6, colors='#333')

    for ax in axes_r[len(active_k):]:
        ax.set_visible(False)

    fig_r.tight_layout()
    fig_r.savefig(os.path.join(save_path, 'posterior_drift_per_cluster.svg'),
                  format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig_r)
    print("✅ Saved: posterior_drift_per_cluster.svg")

    # ── 6. Self-consistency 요약 bar chart ────────────────────────────
    fig_s, ax = plt.subplots(1, 1, figsize=(14, 4), facecolor='white')
    self_rates = [drift_summary[k]['self_rate'] for k in active_k]
    colors_bar = ['#1a6fa8' if r >= 50 else '#c0392b' for r in self_rates]
    ax.bar(range(len(active_k)), self_rates, color=colors_bar, alpha=0.8)
    ax.axhline(50, color='gray', ls='--', lw=1, alpha=0.7, label='50% 기준선')
    ax.set_xticks(range(len(active_k)))
    ax.set_xticklabels([f'k={k}' for k in active_k], rotation=45, fontsize=8)
    ax.set_ylabel('Self-consistency (%)', color='#444')
    ax.set_title('Self-consistency per cluster\n'
                 '(source k = argmax j 인 비율, 높을수록 prior가 제대로 작동함)',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.set_facecolor('white')
    ax.set_ylim(0, 105)
    fig_s.tight_layout()
    fig_s.savefig(os.path.join(save_path, 'self_consistency.svg'),
                  format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig_s)
    print("✅ Saved: self_consistency.svg")

    return drift_summary

def analyze_uncovered_real(
    gen_latents,
    gen_source_k,
    real_latents,
    gmm_means,
    gmm_covs,
    gmm_weights,
    gmm_pca,
    target_k=20,
    save_path=None,
    use_weight=True,
    n_neighbors=5,
    eps=1e-8,
):
    """
    real k=target_k 데이터 중 생성으로 커버되지 않은 포인트를 찾고,
    그게 학습 문제인지 추론 문제인지 진단.

    핵심 아이디어:
      real k=20 각 포인트 x_r에 대해
      - gen(source=20) 중 nearest neighbor 거리 → 생성 커버리지
      - gen(source=any) 중 nearest neighbor 거리 → "어디서든 커버됐는가"

      만약:
        gen(source=20) NN 거리 크고 + gen(any) NN 거리도 큰
          → 모델이 이 영역을 아예 못 배움 (학습 문제)

        gen(source=20) NN 거리 크고 + gen(any) NN 거리 작음
          → 다른 prior로는 만들어짐 (추론/prior 문제)
    """
    os.makedirs(save_path, exist_ok=True)

    print(f"\n[Uncovered Real Analysis] k={target_k}")

    # ── 1. Posterior 계산 ─────────────────────────────────────────────
    real_post, real_argmax, _, _ = compute_gmm_posterior(
        real_latents, gmm_means, gmm_covs, gmm_weights, gmm_pca,
        use_weight=use_weight, eps=eps,
    )

    # real 중 k=target_k에 속하는 것
    mask_real_k = real_argmax == target_k
    R_k = real_latents[mask_real_k]         # (n_Rk, D)
    R_k_post = real_post[mask_real_k]       # (n_Rk, K)
    n_Rk = len(R_k)
    print(f"  real k={target_k}: {n_Rk}개")

    # gen 그룹
    mask_gen_k = gen_source_k == target_k
    G_k   = gen_latents[mask_gen_k]         # gen source=k
    G_any = gen_latents                     # gen 전체

    print(f"  gen source=k={target_k}: {len(G_k)}개")
    print(f"  gen 전체: {len(G_any)}개")

    # ── 2. PCA 64차원으로 압축 (NN 계산용) ───────────────────────────
    from sklearn.decomposition import PCA as skPCA
    from sklearn.neighbors import NearestNeighbors

    pca64 = skPCA(n_components=64, random_state=42)
    all_data = np.concatenate([R_k, G_k, G_any[:5000]], axis=0)
    pca64.fit(all_data)

    R_k_pca   = pca64.transform(R_k)
    G_k_pca   = pca64.transform(G_k)
    G_any_pca = pca64.transform(G_any)

    # ── 3. 각 real k 포인트에 대해 NN 거리 계산 ──────────────────────
    print("  Computing NN distances...")

    # gen(source=k) 기준 NN
    nn_src = NearestNeighbors(n_neighbors=n_neighbors).fit(G_k_pca)
    dist_src, _ = nn_src.kneighbors(R_k_pca)   # (n_Rk, k)
    d_src = dist_src.mean(axis=1)               # (n_Rk,) 각 real→gen_k 거리

    # gen(any) 기준 NN
    nn_any = NearestNeighbors(n_neighbors=n_neighbors).fit(G_any_pca)
    dist_any, _ = nn_any.kneighbors(R_k_pca)
    d_any = dist_any.mean(axis=1)               # (n_Rk,)

    # ── 4. 커버리지 판정 ─────────────────────────────────────────────
    # 임계값: 전체 거리 분포의 중앙값 기준
    thr_src = np.median(d_src)
    thr_any = np.median(d_any)

    # 네 가지 케이스로 분류
    covered      = (d_src <= thr_src)                        # gen_k로 커버됨
    train_fail   = (d_src > thr_src) & (d_any > thr_any)    # 어디서도 못 만듦 → 학습 문제
    infer_fail   = (d_src > thr_src) & (d_any <= thr_any)   # 다른 prior로는 가능 → 추론 문제

    print(f"\n  ── 커버리지 판정 (임계값: d_src={thr_src:.4f}, d_any={thr_any:.4f}) ──")
    print(f"  covered    (gen_k로 커버됨):        {covered.sum():4d} ({100*covered.mean():.1f}%)")
    print(f"  infer_fail (다른 prior로는 가능):   {infer_fail.sum():4d} ({100*infer_fail.mean():.1f}%)")
    print(f"  train_fail (어디서도 못 만듦):      {train_fail.sum():4d} ({100*train_fail.mean():.1f}%)")

    # ── 5. 각 케이스의 posterior 특성 분석 ───────────────────────────
    # 커버 못 한 real이 경계 포인트인가 중심 포인트인가
    r_k_vals = R_k_post[:, target_k]   # 각 real의 r_k(x)

    print(f"\n  ── r_{target_k}(x) 분포 (클러스터 귀속 강도) ──")
    print(f"  {'Group':>20} | {'mean':>8} {'std':>8} | {'의미'}")
    print(f"  {'-'*60}")
    for name, mask in [
        ('covered',    covered),
        ('infer_fail', infer_fail),
        ('train_fail', train_fail),
    ]:
        if mask.sum() == 0:
            continue
        vals = r_k_vals[mask]
        interpretation = (
            "클러스터 중심 데이터" if vals.mean() > r_k_vals.mean()
            else "클러스터 경계 데이터"
        )
        print(f"  {name:>20} | {vals.mean():>8.4f} {vals.std():>8.4f} | {interpretation}")

    # second-best posterior (k 다음으로 높은 클러스터)
    # 경계 포인트일수록 second-best가 높음
    post_copy = R_k_post.copy()
    post_copy[:, target_k] = 0
    second_best_k = post_copy.argmax(axis=1)       # 두 번째로 높은 클러스터
    second_best_v = post_copy.max(axis=1)           # 두 번째 posterior 값

    print(f"\n  ── uncovered real의 second-best 클러스터 분포 ──")
    print(f"  (경계 포인트일수록 특정 j로 second-best가 몰림)")
    for mask, name in [(infer_fail, 'infer_fail'), (train_fail, 'train_fail')]:
        if mask.sum() == 0:
            continue
        sb = second_best_k[mask]
        sv = second_best_v[mask]
        unique, counts = np.unique(sb, return_counts=True)
        top3 = sorted(zip(unique, counts), key=lambda x: -x[1])[:3]
        print(f"\n  [{name}] second-best 클러스터 top3:")
        for j, cnt in top3:
            mean_v = sv[sb == j].mean()
            print(f"    j={j:2d}: {cnt:3d}개 ({100*cnt/mask.sum():.1f}%)  "
                  f"r_{j}(x) mean={mean_v:.5f}")

    # ── 6. 시각화 ─────────────────────────────────────────────────────
    pca2 = skPCA(n_components=2, random_state=42)
    pca2.fit(np.concatenate([R_k, G_k], axis=0))

    R_k_2d = pca2.transform(R_k)
    G_k_2d = pca2.transform(G_k)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
    fig.suptitle(f'Uncovered Real Analysis: k={target_k}\n'
                 f'covered={covered.sum()} | '
                 f'infer_fail={infer_fail.sum()} | '
                 f'train_fail={train_fail.sum()}',
                 fontsize=11, fontweight='bold')

    colors = {'covered': '#1a6fa8', 'infer_fail': '#f39c12', 'train_fail': '#c0392b'}

    for ax_idx, (name, mask) in enumerate([
        ('covered', covered),
        ('infer_fail', infer_fail),
        ('train_fail', train_fail),
    ]):
        ax = axes[ax_idx]
        ax.set_facecolor('white')

        # gen_k 배경
        ax.scatter(G_k_2d[:,0], G_k_2d[:,1], s=4, alpha=0.15,
                   color='gray', label=f'gen k={target_k}')

        # 전체 real_k (흐리게)
        ax.scatter(R_k_2d[:,0], R_k_2d[:,1], s=8, alpha=0.2,
                   color='#aaaaaa', label='real k (all)')

        # 해당 그룹 강조
        if mask.sum() > 0:
            ax.scatter(R_k_2d[mask,0], R_k_2d[mask,1], s=20, alpha=0.8,
                       color=colors[name], label=f'{name} ({mask.sum()})')

        ax.set_title(
            f'{name}\n'
            f'r_k mean={r_k_vals[mask].mean():.4f}' if mask.sum() > 0
            else name,
            fontsize=9
        )
        ax.legend(fontsize=7)

    fig.tight_layout()
    fname = os.path.join(save_path, f'uncovered_real_k{target_k}.svg')
    fig.savefig(fname, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n✅ Saved: {fname}")

    return {
        'covered': covered,
        'infer_fail': infer_fail,
        'train_fail': train_fail,
        'd_src': d_src,
        'd_any': d_any,
        'r_k_vals': r_k_vals,
        'second_best_k': second_best_k,
    }

# -------------------------------------------------
# Main
# -------------------------------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- load config --------
    with open(args.ds_config_path) as f:
        ds_config = yaml.safe_load(f)

    # -------- real latent --------
    dataset = ImgLatentDataset(
        data_dir=ds_config["data"]["data_path"],
        latent_norm=ds_config["data"].get("latent_norm", True),
        latent_multiplier=ds_config["data"].get("latent_multiplier", 0.18215),
    )
    loader = DataLoader(
        dataset,
        batch_size= 256, #64,
        shuffle=True,
        num_workers=ds_config['data']['num_workers'],
        pin_memory=True
    )
    real_latents, real_labels = collect_real_latents(loader, max_points=50000)

    # -------- GMM --------
    gmm_dir = f"{ds_config['gmm']['output_dir']}/{ds_config['gmm']['num_clusters']}_{ds_config['gmm']['cov_type']}"
    with open(os.path.join(gmm_dir, "gmm_clusters.pkl"), "rb") as f:
        gmm_ckpt = pickle.load(f)

    gmm_means = gmm_ckpt["means"]
    gmm_covs = gmm_ckpt["covs"]
    gmm_weights = gmm_ckpt["weights"]
    gmm_pca = gmm_ckpt["pca_components"]

    gmm_params = dict(
        gmm_means=gmm_means,
        gmm_covs=gmm_covs,
        gmm_weights=gmm_weights,
        gmm_pca=gmm_pca,
    )

    # # -------- generated latent (전체, 클러스터 구분 없음) --------
    # gen_latents = collect_generated_latents(
    #     args=args,
    #     ds_config=ds_config,
    #     device=device,
    #     max_images=50000
    # )
    real_post, real_assign, real_logp, _ = compute_gmm_posterior(
        real_latents, gmm_means, gmm_covs, gmm_weights, gmm_pca, use_weight=True,
    )
    print("real argmax 분포:")
    unique, counts = np.unique(real_assign, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  k={u}: {c}")

    # k=0의 log_prob이 다른 것보다 얼마나 큰지 확인
    print(f"\nreal_latents[0]의 log_prob (top 5):")
    lp = real_logp[0]
    top5_idx = np.argsort(lp)[::-1][:5]
    for i in top5_idx:
        print(f"  k={i}: log_prob={lp[i]:.2f}")

    # -------- t-SNE --------
    # flat_means = np.vstack([means for means in gmm_means.values()])
    # tsne_real_gmm_generated(
    #     real_latents,
    #     real_labels,
    #     gmm_means,
    #     gen_latents,
    #     save_path=os.path.join(args.generated_dir, "../tsne_real_val.png")
    # )

    # -------- center proximity (L2 기반, 기존 분석) --------
    # flat_means = np.vstack([means for means in gmm_means.values()])
    # analyze_center_proximity(
    #     real_latents=real_latents,
    #     gen_latents=gen_latents,
    #     flat_means=flat_means,
    #     save_path=os.path.join(args.generated_dir, "../center_analysis"),
    # )
    # analyze_center_proximity_per_cluster(
    #     real_latents=real_latents,
    #     gen_latents=gen_latents,
    #     flat_means=flat_means,
    #     save_path=os.path.join(args.generated_dir, "../center_analysis"),
    # )

    # -------- 분석 1: posterior coverage (전체 gen, 클러스터 구분 없음) --------
    # analyze_posterior_coverage(
    #     real_latents=real_latents,
    #     gen_latents=gen_latents,
    #     save_path=os.path.join(args.generated_dir, "../posterior_analysis"),
    #     use_weight=True,
    #     **gmm_params,
    # )

    # -------- 분석 2: posterior drift (클러스터별 폴더에서 읽음) --------
    # # -------- load VAE --------
    if args.model_type == "vavae":
        from tokenizer.vavae import VA_VAE
        vae = VA_VAE(args.config_path)
    else:
        raise NotImplementedError
    gen_latents_by_cluster, gen_source_k = collect_gen_latents_with_source_cluster(
        generated_dir=args.generated_dir,
        vae_wrapper=vae,
        ds_config=ds_config,
        device=device,
        max_images_per_cluster=3000,
    )
    # analyze_posterior_drift(
    #     gen_latents=gen_latents_by_cluster,
    #     gen_source_k=gen_source_k,
    #     save_path=os.path.join(args.generated_dir, "../posterior_analysis"),
    #     use_weight=True,
    #     **gmm_params,
    # )
    for k in [20, 6, 15]:  # drift 심한 클러스터
        analyze_uncovered_real(
            gen_latents=gen_latents_by_cluster,
            gen_source_k=gen_source_k,
            real_latents=real_latents,
            gmm_means=gmm_means,
            gmm_covs=gmm_covs,
            gmm_weights=gmm_weights,
            gmm_pca=gmm_pca,
            target_k=k,
            save_path=os.path.join(args.generated_dir, "../posterior_analysis"),
            use_weight=True,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="model1_f16d32.yaml")
    parser.add_argument("--ds_config_path", type=str, default="model2_xl_vavae_f16d32.yaml")
    # parser.add_argument("--generated_dir", type=str, default='/mnt/SSD_raid1/lsun/church_outdoor_val')
    parser.add_argument("--generated_dir", type=str, default='output/9th_lightningdit_xl_vavae_f16d32_gmm30_deterministic/lightningdit-xl-1-ckpt-0039440-euler-40/class_0')
    parser.add_argument("--model_type", type=str, default="vavae")
    args = parser.parse_args()

    main(args)