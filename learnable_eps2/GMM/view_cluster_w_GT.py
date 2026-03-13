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
            transform=vae_wrapper.img_transform(p_hflip=0.0) #1.0
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
    from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance
    import matplotlib.gridspec as gridspec

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
    C_REAL, C_GEN = '#4fc3f7', '#ef5350'
    fig = plt.figure(figsize=(18, 5), facecolor='#0f0f1a')
    fig.suptitle("GMM Center ↔ Real  vs  GMM Center ↔ Gen",
                 color='white', fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.33)

    def style(ax, title):
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#aaa', labelsize=9)
        ax.set_title(title, color='white', fontsize=10.5, pad=6)
        for sp in ax.spines.values():
            sp.set_edgecolor('#333')

    # 히스토그램
    ax = fig.add_subplot(gs[0])
    bins = np.linspace(0, max(real_dist.max(), gen_dist.max()) * 1.02, 60)
    ax.hist(real_dist, bins=bins, density=True, alpha=0.55, color=C_REAL,
            label=f'Real  med={np.median(real_dist):.3f}')
    ax.hist(gen_dist,  bins=bins, density=True, alpha=0.55, color=C_GEN,
            label=f'Gen   med={np.median(gen_dist):.3f}')
    ax.axvline(np.median(real_dist), color=C_REAL, lw=2, ls='--')
    ax.axvline(np.median(gen_dist),  color=C_GEN,  lw=2, ls='--')
    ax.set_xlabel('Distance to nearest center', color='#aaa')
    ax.set_ylabel('Density', color='#aaa')
    ax.legend(fontsize=9)
    style(ax, 'Distribution')

    # CDF
    ax = fig.add_subplot(gs[1])
    for dist, label, color in [(real_dist, 'Real', C_REAL), (gen_dist, 'Gen', C_GEN)]:
        sd = np.sort(dist)
        ax.plot(sd, np.arange(1, len(sd)+1) / len(sd), color=color, lw=2, label=label)
    ax.axhline(0.5, color='#aaa', lw=1, ls=':', alpha=0.6)
    ax.axvline(np.median(real_dist), color=C_REAL, lw=1.2, ls='--', alpha=0.7)
    ax.axvline(np.median(gen_dist),  color=C_GEN,  lw=1.2, ls='--', alpha=0.7)
    ax.set_xlabel('Distance to nearest center', color='#aaa')
    ax.set_ylabel('CDF', color='#aaa')
    ax.legend(fontsize=9); ax.grid(alpha=0.12)
    style(ax, 'CDF  (gen이 왼쪽 → center 편향)')

    # Violin
    ax = fig.add_subplot(gs[2])
    vp = ax.violinplot([real_dist, gen_dist], positions=[1, 2],
                       showmedians=True, showextrema=True)
    for body, c in zip(vp['bodies'], [C_REAL, C_GEN]):
        body.set_facecolor(c); body.set_alpha(0.65)
    vp['cmedians'].set_color('white'); vp['cmedians'].set_linewidth(2)
    for key in ['cmins', 'cmaxes', 'cbars']:
        vp[key].set_color('#aaa')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Real', 'Gen'], color='white', fontsize=11)
    ax.set_ylabel('Distance to nearest center', color='#aaa')
    style(ax, 'Violin')

    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close(fig)
    print(f"✅ Saved: {save_path}")

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

    # -------- real dataset --------
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

    # -------- generated --------
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
        save_path=os.path.join(gmm_dir, "proximity_analysis.png")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="model1_f16d32.yaml")
    parser.add_argument("--ds_config_path", type=str, default="model2_xl_vavae_f16d32.yaml")
    # parser.add_argument("--generated_dir", type=str, default='/mnt/SSD_raid1/lsun/church_outdoor_val')
    parser.add_argument("--generated_dir", type=str, default='output/5th_lightningdit_xl_vavae_f16d32_gmm30_use_weight/lightningdit-xl-1-ckpt-0063000-euler-20')
    parser.add_argument("--model_type", type=str, default="vavae")
    args = parser.parse_args()

    main(args)