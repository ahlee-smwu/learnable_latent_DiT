"""
View clusters
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from tools.calculate_fid import calculate_fid_given_paths
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchmetrics import StructuralSimilarityIndexMeasure
from models.lpips import LPIPS
from torchvision.datasets import ImageFolder
from torchvision import transforms
from diffusers.models import AutoencoderKL
import pickle
import math
from PIL import Image
from collections import defaultdict
import math
from datasets.img_latent_dataset import ImgLatentDataset
import yaml
import argparse
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import math
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def print_with_prefix(content, prefix='GMM visualization', rank=0):
    if rank == 0:
        print(f"\033[34m[{prefix}]\033[0m {content}")

def save_image(image, filename):
    Image.fromarray(image).save(filename)

def make_distinct_colors(n):
    hues = np.linspace(0, 1, n, endpoint=False)
    colors = [mcolors.hsv_to_rgb((h, 0.9, 0.9)) for h in hues]
    return colors

def decode_gmm_means(
    model,
    gmm_means,            # dict: class -> (K, D)
    latent_mean,
    latent_std,
    latent_multiplier,
    latent_shape=(32, 16, 16),
    device="cuda",
    save_root="./gmm_vis",
    grid_cols=5
):
    model.eval()
    os.makedirs(save_root, exist_ok=True)

    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device)

    gmm_mean_images = {}

    for cls, means in gmm_means.items():
        imgs = []
        gmm_mean_images[cls] = {}

        for k in range(means.shape[0]):
            z = torch.from_numpy(means[k]).float().to(device).view(1, *latent_shape)

            # inverse latent norm
            z = z / latent_multiplier
            z = z * latent_std + latent_mean

            with torch.no_grad():
                img = model.decode(z)
                img = torch.clamp(127.5 * img + 128.0, 0, 255)
                img = img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            pil_img = Image.fromarray(img)
            imgs.append(pil_img)
            gmm_mean_images[cls][k] = pil_img

        # ---- grid ----
        rows = math.ceil(len(imgs) / grid_cols)
        w, h = imgs[0].size
        grid = Image.new("RGB", (grid_cols * w, rows * h))
        for i, img in enumerate(imgs):
            grid.paste(img, ((i % grid_cols) * w, (i // grid_cols) * h))

        save_path = os.path.join(save_root, f"class_{cls}_gmm_means.png")
        grid.save(save_path)
        print_with_prefix(f"✅ Saved GMM mean grid: {save_path}")

    return gmm_mean_images

def collect_cluster_samples(
    loader,
    model,
    centers,
    latent_mean,
    latent_std,
    latent_multiplier,
    device,
    model_type='vavae',
    latent_shape=(32, 16, 16),
    max_samples_per_cluster=19
):
    """
    Returns:
        cluster_samples[class][cluster_id] = list[PIL.Image]
        cluster_counts[class][cluster_id] = int
    """
    model.eval()

    cluster_samples = defaultdict(lambda: defaultdict(list))
    cluster_counts = defaultdict(lambda: defaultdict(int))

    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device)

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Collecting cluster samples"):
            x = x.to(device)
            y = y.to(device)

            B, C, H, W = x.shape
            latents_flat = x.view(B, -1)  # [B, D]

            for i in range(B):
                cls = int(y[i].item())
                mu = centers[cls]  # (K, D)

                # L2 distance
                dists = ((latents_flat[i] - mu) ** 2).sum(dim=1)
                k = int(dists.argmin().item())

                # count always
                cluster_counts[cls][k] += 1

                # sample only if under limit
                if len(cluster_samples[cls][k]) < max_samples_per_cluster:
                    z = latents_flat[i].view(1, *latent_shape)
                    # inverse latent norm
                    z = z / latent_multiplier
                    z = z * latent_std + latent_mean

                    img = model.decode(z)
                    img = torch.clamp(127.5 * img + 128.0, 0, 255)
                    img = img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

                    cluster_samples[cls][k].append(Image.fromarray(img))

    return cluster_samples, cluster_counts

def tsne(
    latents,              # (N, D)
    labels,               # (N,) class labels
    gmm_means_dict,       # class -> (K, D)
    save_path,
    color_mode="cluster", # "class" or "cluster"
    pca_dim=128,
    max_points=None,
    seed=42
):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import colorsys

    np.random.seed(seed)

    # -----------------------------
    # Subsampling
    # -----------------------------
    if max_points is not None and latents.shape[0] > max_points:
        idx = np.random.choice(latents.shape[0], max_points, replace=False)
        latents = latents[idx]
        labels = labels[idx]

    # -----------------------------
    # Flatten GMM means
    # -----------------------------
    mean_vectors = []
    mean_ids = []  # class or class_cluster

    for cls, means in gmm_means_dict.items():
        for k in range(means.shape[0]):
            mean_vectors.append(means[k])
            if color_mode == "class":
                mean_ids.append(cls)
            else:
                mean_ids.append(f"{cls}_{k}")

    mean_vectors = np.stack(mean_vectors, axis=0)
    mean_ids = np.array(mean_ids)

    # -----------------------------
    # PCA + t-SNE
    # -----------------------------
    X = np.concatenate([latents, mean_vectors], axis=0)
    X = PCA(n_components=pca_dim).fit_transform(X)

    print_with_prefix("Starting t-SNE calculation (this may take a while)...")
    start_time = time.time()

    X_tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        verbose=1 #0,1,2
    ).fit_transform(X)

    end_time = time.time()
    elapsed = end_time - start_time
    print_with_prefix(f"t-SNE finished! Elapsed time: {elapsed:.2f} seconds")

    data_tsne = X_tsne[:len(latents)]
    mean_tsne = X_tsne[len(latents):]

    # -----------------------------
    # Generate DISTINCT colors (HSV)
    # -----------------------------
    unique_ids = sorted(np.unique(mean_ids))
    n_colors = len(unique_ids)

    color_dict = {}
    for i, uid in enumerate(unique_ids):
        # Hue 균등 분할 + 고정 Saturation / Value
        hue = i / n_colors
        color_dict[uid] = colorsys.hsv_to_rgb(hue, 0.85, 0.95)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(11, 11))

    # --------- Real data (NO legend) ---------
    if color_mode == "class":
        for cls in np.unique(labels):
            mask = labels == cls
            plt.scatter(
                data_tsne[mask, 0],
                data_tsne[mask, 1],
                s=4,
                alpha=0.25,
                color=color_dict[cls],
                marker="o"
            )
    else:
        # cluster assignment by nearest mean (L2)
        for cls, means in gmm_means_dict.items():
            means = np.asarray(means)
            cls_mask = labels == cls
            cls_latents = latents[cls_mask]

            dists = ((cls_latents[:, None, :] - means[None]) ** 2).sum(-1)
            assigns = dists.argmin(axis=1)

            global_idx = np.where(cls_mask)[0]
            for k in range(means.shape[0]):
                idxs = global_idx[assigns == k]
                if len(idxs) == 0:
                    continue
                key = f"{cls}_{k}"
                plt.scatter(
                    data_tsne[idxs, 0],
                    data_tsne[idxs, 1],
                    s=4,
                    alpha=0.25,
                    color=color_dict[key],
                    marker="o"
                )

    # --------- GMM means (WITH legend) ---------
    for i, uid in enumerate(mean_ids):
        plt.scatter(
            mean_tsne[i, 0],
            mean_tsne[i, 1],
            s=140,
            marker="X",
            color=color_dict[uid],
            edgecolors="black",
            linewidths=0.8,
            label=f"Mean {uid}"
        )

    plt.title(f"t-SNE ({color_mode}-colored, HSV distinct)")
    plt.legend(
        fontsize=7,
        markerscale=0.9,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        ncol=1
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Saved t-SNE: {save_path}")

def print_cluster_statistics(cluster_counts):
    print("\n📊 Cluster assignment statistics")
    for cls in sorted(cluster_counts.keys()):
        print(f"\nClass {cls}")
        for k in sorted(cluster_counts[cls].keys()):
            print(f"  Cluster {k:02d}: {cluster_counts[cls][k]} samples")

def save_cluster_sample_grids(
    cluster_samples,
    cluster_mean_images,
    save_root,
    grid_cols=5
):
    os.makedirs(save_root, exist_ok=True)

    for cls in cluster_samples:
        for k, imgs in cluster_samples[cls].items():
            if len(imgs) == 0:
                continue
            # -----------------------------
            # [mean 1장] + [real 최대 19장]
            # -----------------------------
            if cls in cluster_mean_images and k in cluster_mean_images[cls]:
                imgs = [cluster_mean_images[cls][k]] + imgs[:19]
            else:
                imgs = imgs[:20]  # fallback

            num_imgs = len(imgs)
            grid_rows = math.ceil(num_imgs / grid_cols)

            w, h = imgs[0].size
            grid_img = Image.new(
                "RGB",
                (grid_cols * w, grid_rows * h)
            )
            for idx, img in enumerate(imgs):
                row = idx // grid_cols
                col = idx % grid_cols
                grid_img.paste(img, (col * w, row * h))
            save_path = os.path.join(
                save_root,
                f"class_{cls}_cluster_{k}_mean_plus_real.png"
            )
            grid_img.save(save_path)
            print(f"✅ Saved mean+real grid: {save_path}")

def encode_images(model, images, model_type='vavae'):
    with torch.no_grad():
        posterior = {
            'vavae': lambda: model.encode(images),
            'marvae': lambda: model.encode(images),
            'sdvae': lambda: model.encode(images).latent_dist
        }[model_type]()
        return posterior.sample().to(torch.float32)

def decode_to_images(model, z):
    with torch.no_grad():
        images = model.decode(z)
        images = torch.clamp(127.5 * images + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    return images

def calculate_psnr(original, processed):
    mse = torch.mean((original - processed) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse)).item()

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return torch.tensor(np.array(image).transpose(2, 0, 1), dtype=torch.float32)

def calculate_psnr_for_pair(original_path, processed_path):
    return calculate_psnr(load_image(original_path), load_image(processed_path))

def calculate_psnr_between_folders(original_folder, processed_folder):
    original_files = sorted(os.listdir(original_folder))
    processed_files = sorted(os.listdir(processed_folder))

    if len(original_files) != len(processed_files):
        print("Warning: Mismatched number of images in folders")
        return []

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(calculate_psnr_for_pair,
                          os.path.join(original_folder, orig),
                          os.path.join(processed_folder, proc))
            for orig, proc in zip(original_files, processed_files)
        ]
        return [future.result() for future in as_completed(futures)]

def evaluate_gmm_scores(gmm, X):
    return {
        "log_likelihood": gmm.score(X),
        "BIC": gmm.bic(X),
        "AIC": gmm.aic(X),
    }

def gmm_entropy(gmm, X):
    resp = gmm.predict_proba(X)   # (N, K)
    entropy = -np.sum(resp * np.log(resp + 1e-9), axis=1)
    return entropy.mean()

def gmm_mode_usage(gmm, X):
    assignments = gmm.predict(X)
    counts = np.bincount(assignments, minlength=gmm.n_components)
    return counts / counts.sum()

def evaluate_all_classes(
    gmm_dict,          # class -> fitted gmm
    features_dict      # class -> (Nc, D)
):
    report = {}

    for cls in gmm_dict:
        gmm = gmm_dict[cls]
        X = features_dict[cls]

        report[cls] = {
            **evaluate_gmm_scores(gmm, X),
            "entropy": gmm_entropy(gmm, X),
            "mode_usage": gmm_mode_usage(gmm, X),
        }

    return report

def make_grid(images, rows, cols, img_size):
    """images: list[PIL.Image], length <= rows*cols"""
    grid = Image.new("RGB", (cols * img_size, rows * img_size), (0, 0, 0))
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        grid.paste(img.resize((img_size, img_size)), (c * img_size, r * img_size))
    return grid

def save_class_cluster_grid(
    cluster_mean_images,  # class -> dict[k -> PIL.Image]
    save_root,
    img_size=128
):
    os.makedirs(save_root, exist_ok=True)

    for cls, mean_dict in cluster_mean_images.items():
        mean_imgs = [mean_dict[k] for k in sorted(mean_dict.keys())]

        K = len(mean_imgs)
        cols = int(math.ceil(math.sqrt(K)))
        rows = int(math.ceil(K / cols))

        grid = make_grid(mean_imgs, rows, cols, img_size)
        out_path = os.path.join(save_root, f"class_{cls}_clusters.png")
        grid.save(out_path)
        print_with_prefix(f"✅ Saved class cluster grids: {out_path}")

def save_real_cluster_sample_grids(
    cluster_samples,       # class -> cluster -> list[PIL.Image]
    cluster_mean_images,   # class -> list[PIL.Image]
    save_root,
    img_size=128
):
    '''[cluster mean 1 + real samples 19] 그리드 저장'''

    base_dir = os.path.join(save_root, "real_cluster_samples")
    os.makedirs(base_dir, exist_ok=True)

    for cls, clusters in cluster_samples.items():
        for k, real_imgs in clusters.items():
            if len(real_imgs) == 0:
                continue

            mean_img = cluster_mean_images[cls][k]

            # mean 1 + real 19 = 최대 20
            images = [mean_img] + real_imgs[:19]

            rows, cols = 4, 5  # 20 images
            grid = make_grid(images, rows, cols, img_size)

            out_path = os.path.join(
                base_dir,
                f"class_{cls}_cluster_{k}.png"
            )
            grid.save(out_path)
            print_with_prefix(f"✅ Saved save_real_cluster_sample_grids: {out_path}")

def view_cluster(config_path, ds_config, model_type, tsne_color_mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_with_prefix("Loading model...")

    # -----------------------------
    # Load model
    # -----------------------------
    if model_type == 'vavae':
        from tokenizer.vavae import VA_VAE
        model = VA_VAE(config_path).load().model.to(device)
    elif model_type == 'sdvae':
        model = AutoencoderKL.from_pretrained(
            "path/to/your/sd-vae-ft-ema"
        ).to(device)
    elif model_type == 'marvae':
        from tokenizer.marvae import MAR_VAE
        model = MAR_VAE().load().model.to(device)

    model.eval()

    # -----------------------------
    # Dataset / Loader
    # -----------------------------
    dataset = ImgLatentDataset(
        data_dir=ds_config['data']['data_path'],
        latent_norm=ds_config['data'].get('latent_norm', False),
        latent_multiplier=ds_config['data'].get('latent_multiplier', 0.18215),
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=ds_config['data']['num_workers'],
        pin_memory=True
    )

    # -----------------------------
    # Load GMM checkpoint
    # -----------------------------
    output_path = f"{ds_config['gmm']['output_dir']}/{ds_config['gmm']['num_clusters']}_{ds_config['gmm']['cov_type']}"

    with open(os.path.join(output_path, "gmm_clusters.pkl"), "rb") as f:
        gmm_ckpt = pickle.load(f)

    gmm_means = gmm_ckpt["means"]         # class -> (K, D)

    # -----------------------------
    # Load latent stats
    # -----------------------------
    latent_stats = torch.load(
        os.path.join(
            "feature_output/model1_f16d32/lsun_train_256/",
            "latents_stats.pt"
        ),
        map_location="cpu"
    )

    # -----------------------------
    # (1) Decode GMM means & save mean grid
    # -----------------------------
    cluster_mean_images = decode_gmm_means(
        model=model,
        gmm_means=gmm_means,
        latent_mean=latent_stats["mean"],
        latent_std=latent_stats["std"],
        latent_multiplier=ds_config['data'].get('latent_multiplier', 0.18215),
        latent_shape=(32, 16, 16),
        device=device,
        save_root=output_path
    )

    # -----------------------------
    # (2) t-SNE
    # -----------------------------
    all_latents, all_labels = [], []

    with torch.no_grad():
        for latents, y in tqdm(loader, desc="Collect latents"):
            latents = latents.view(latents.size(0), -1)
            all_latents.append(latents.cpu().numpy())
            all_labels.append(y.numpy())

    all_latents = np.concatenate(all_latents, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    tsne(
        latents=all_latents,
        labels=all_labels,
        gmm_means_dict=gmm_means,
        save_path=os.path.join(output_path, f"tsne_{tsne_color_mode}.png"),
        color_mode=tsne_color_mode
    )

    # -----------------------------
    # (3) Collect real samples per GMM cluster & save real grid
    # -----------------------------
    cluster_samples, cluster_counts = collect_cluster_samples(
        loader=loader,
        model=model,
        centers={
            cls: torch.from_numpy(mu).float().to(device)
            for cls, mu in gmm_means.items()
        },
        latent_mean=latent_stats["mean"],
        latent_std=latent_stats["std"],
        latent_multiplier=ds_config['data'].get('latent_multiplier', 0.18215),
        device=device
    )

    save_real_cluster_sample_grids(
        cluster_samples,
        cluster_mean_images,
        save_root=output_path
    )

    # -----------------------------
    # (4) Statics
    # -----------------------------
    print_cluster_statistics(cluster_counts)

    print_with_prefix("GMM cluster visualization DONE ✅")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='model1_f16d32.yaml')
    parser.add_argument('--ds_config_path', type=str, default='model2_xl_vavae_f16d32.yaml')
    parser.add_argument('--model_type', type=str, default='vavae')
    parser.add_argument('--tsne_color_mode', type=str, default='cluster', choices=['cluster', 'class'])
    args = parser.parse_args()
    ds_config = load_config(args.ds_config_path)

    view_cluster(config_path=args.config_path, ds_config=ds_config, model_type=args.model_type, tsne_color_mode=args.tsne_color_mode)