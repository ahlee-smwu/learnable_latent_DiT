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

# -------------------------------------------------
# Utils
# -------------------------------------------------
def make_distinct_colors(n):
    hues = np.linspace(0, 1, n, endpoint=False)
    return [mcolors.hsv_to_rgb((h, 0.85, 0.95)) for h in hues]

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
    # --------------------------------
    # Create VAE (wrapper + model 분리)
    # --------------------------------
    vae_wrapper = VA_VAE(args.config_path)
    vae = vae_wrapper.load().model.to(device)
    vae.eval()

    # --------------------------------
    # Setup data (flip 2-view)
    # --------------------------------
    datasets = [
        ImageFolder(
            args.generated_dir,
            transform=vae_wrapper.img_transform(p_hflip=0.0)
        ),
        ImageFolder(
            args.generated_dir,
            transform=vae_wrapper.img_transform(p_hflip=1.0) #1.0
        )
    ]

    loaders = [
        DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False
        ) for dataset in datasets
    ]

    print(f"[Generated] Total images (with flip): {len(datasets[0]) * 2}")

    # --------------------------------
    # Encode images → latent
    # --------------------------------
    latents_all = []
    processed = 0

    mean, std = load_latent_stats(
        data_dir=ds_config["data"]["data_path"],
        device=device
    )
    latent_multiplier = ds_config["data"].get("latent_multiplier", 0.18215)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(zip(*loaders)):
            for x, _ in batch_data:
                x = x.to(device)

                posterior = vae.encode(x)
                z = posterior.sample()              # (B, C, H, W)
                # z = z.view(z.size(0), -1)           # flatten
                #
                # # latent norm & multiplier
                # # from datasets.img_latent_dataset import ImgLatentDataset
                # z = normalize_latents(z, mean, std, latent_multiplier)

                z = (z - mean) / std  # fused op
                z = z * latent_multiplier

                z = z.flatten(1)

                latents_all.append(z.cpu().numpy())
                processed += z.size(0)

                if processed >= max_images:
                    break

            if batch_idx % 50 == 0:
                print(f"[Generated] processed {processed}/{max_images}")

            if processed >= max_images:
                break

    latents_all = np.concatenate(latents_all, axis=0)

    if latents_all.shape[0] > max_images:
        latents_all = latents_all[:max_images]

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

# -------------------------------------------------
# Main
# -------------------------------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- load config --------
    with open(args.ds_config_path) as f:
        ds_config = yaml.safe_load(f)

    # # -------- load VAE --------
    # if args.model_type == "vavae":
    #     from tokenizer.vavae import VA_VAE
    #     vae = VA_VAE(args.config_path).load().model.to(device)
    # else:
    #     raise NotImplementedError
    #
    # vae.eval()

    # -------- real latent --------
    dataset = ImgLatentDataset(
        data_dir=ds_config["data"]["data_path"],
        latent_norm=ds_config["data"].get("latent_norm", False),
        latent_multiplier=ds_config["data"].get("latent_multiplier", 0.18215),
    )
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=ds_config['data']['num_workers'],
        pin_memory=True
    )
    real_latents, real_labels = collect_real_latents(loader)

    # -------- GMM --------
    gmm_dir = f"{ds_config['gmm']['output_dir']}/{ds_config['gmm']['num_clusters']}_{ds_config['gmm']['cov_type']}"
    with open(os.path.join(gmm_dir, "gmm_clusters.pkl"), "rb") as f:
        gmm_ckpt = pickle.load(f)

    gmm_means = gmm_ckpt["means"]

    # -------- generated latent --------
    gen_latents = collect_generated_latents(
        args=args,
        ds_config=ds_config,
        device=device,
        max_images=5000
    )

    # -------- t-SNE --------
    # tsne_real_gmm_generated(
    #     real_latents,
    #     real_labels,
    #     gmm_means,
    #     gen_latents,
    #     save_path=os.path.join(gmm_dir, "tsne_real_val.png")
    # )

    # center score
    flat_means = np.vstack([means for means in gmm_means.values()])

    analyze_center_proximity(
        real_latents=real_latents,
        gen_latents=gen_latents,
        flat_means=flat_means,
        save_path=os.path.join(gmm_dir, "center_analysis")
    ) # 전체 데이터셋 분석

    analyze_center_proximity_per_cluster(
        real_latents=real_latents,
        gen_latents=gen_latents,
        flat_means=flat_means,
        save_path=os.path.join(gmm_dir, "center_analysis")
    ) # per cluster 분석

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="model1_f16d32.yaml")
    parser.add_argument("--ds_config_path", type=str, default="model2_xl_vavae_f16d32.yaml")
    # parser.add_argument("--generated_dir", type=str, default='/mnt/SSD_raid1/lsun/church_outdoor_val')
    parser.add_argument("--generated_dir", type=str, default='output/5th_lightningdit_xl_vavae_f16d32_gmm30_use_weight/lightningdit-xl-1-ckpt-0063000-euler-20')
    parser.add_argument("--model_type", type=str, default="vavae")
    args = parser.parse_args()

    main(args)