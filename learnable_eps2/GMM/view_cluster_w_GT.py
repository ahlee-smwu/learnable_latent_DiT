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
    tsne_real_gmm_generated(
        real_latents,
        real_labels,
        gmm_means,
        gen_latents,
        save_path=os.path.join(gmm_dir, "tsne_real_val.png")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="model1_f16d32.yaml")
    parser.add_argument("--ds_config_path", type=str, default="model2_xl_vavae_f16d32.yaml")
    parser.add_argument("--generated_dir", type=str, default='/mnt/SSD_raid1/lsun/church_outdoor_val')
    # parser.add_argument("--generated_dir", type=str, default='output/1st_lightningdit_xl_vavae_f16d32_gmm30/lightningdit-xl-1-ckpt-0159000-euler-20')
    parser.add_argument("--model_type", type=str, default="vavae")
    args = parser.parse_args()

    main(args)