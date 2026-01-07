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

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def print_with_prefix(content, prefix='Tokenizer Evaluation', rank=0):
    if rank == 0:
        print(f"\033[34m[{prefix}]\033[0m {content}")

def save_image(image, filename):
    Image.fromarray(image).save(filename)

def decode_cluster_means(
    model,
    cluster_centers,
    latent_mean,
    latent_std,
    latent_multiplier,
    latent_shape=(32, 16, 16),
    device="cuda",
    save_root="./cluster_vis",
    grid_cols=5
):
    """
    return:
        cluster_mean_images[class][k] = PIL.Image
    """
    model.eval()
    os.makedirs(save_root, exist_ok=True)

    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device)

    cluster_mean_images = {}

    for cls, centers in cluster_centers.items():
        imgs = []
        cluster_mean_images[cls] = {}

        for k, center in enumerate(centers):
            z = center.to(device).view(1, *latent_shape)

            # inverse latent norm
            z = z / latent_multiplier
            z = z * latent_std + latent_mean

            with torch.no_grad():
                img = model.decode(z)
                img = torch.clamp(127.5 * img + 128.0, 0, 255)
                img = img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            pil_img = Image.fromarray(img)

            imgs.append(pil_img)
            cluster_mean_images[cls][k] = pil_img

        # -----------------------------
        # make grid
        # -----------------------------
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
            save_root, f"class_{cls}_clusters.png"
        )
        grid_img.save(save_path)
        print(f"✅ Saved cluster grid: {save_path}")

    return cluster_mean_images

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

def view_cluster(config_path, ds_config, model_type):
    # -----------------------------
    # single-process setup
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_with_prefix("Loading model...")

    # Load model
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

    # Setup data
    dataset = ImgLatentDataset(
        data_dir=ds_config['data']['data_path'],
        latent_norm=ds_config['data']['latent_norm'] if 'latent_norm' in ds_config['data'] else False,
        latent_multiplier=ds_config['data']['latent_multiplier'] if 'latent_multiplier' in ds_config['data'] else 0.18215,
    )
    batch_size_per_gpu = 1
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=ds_config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    # -----------------------------
    # Load cluster centers
    # -----------------------------
    output_path = f"{ds_config['kmeans']['output_dir']}/{ds_config['kmeans']['num_clusters']}"

    with open(os.path.join(output_path, "kmeans_clusters.pkl"), "rb") as f:
        ckpt = pickle.load(f)

    cluster_centers = {
        cls: torch.from_numpy(mu).to(device=device, dtype=torch.float32)
        for cls, mu in ckpt["centers"].items()
    }
    # dict: class_id → (IPC, D)

    # -----------------------------
    # Load latent stats (for denorm)
    # -----------------------------
    latent_stats = torch.load(
        os.path.join(
            "feature_output/model1_f16d32/lsun_train_256/",
            "latents_stats.pt"
        ),
        map_location="cpu"
    )

    # -----------------------------
    # Decode & visualize clusters
    # -----------------------------
    cluster_mean_images = decode_cluster_means(
        model=model,
        cluster_centers=cluster_centers,
        latent_mean=latent_stats["mean"],
        latent_std=latent_stats["std"],
        latent_multiplier=ds_config['data']['latent_multiplier'] if 'latent_multiplier' in ds_config['data'] else 0.18215,
        latent_shape=(32, 16, 16),
        device=device,
        save_root=output_path
    )

    # -----------------------------
    # Collect real samples per cluster
    # -----------------------------
    cluster_samples, cluster_counts = collect_cluster_samples(
        loader=loader,
        model=model,
        centers=cluster_centers,
        latent_mean=latent_stats["mean"],
        latent_std=latent_stats["std"],
        latent_multiplier=ds_config['data']['latent_multiplier'] if 'latent_multiplier' in ds_config['data'] else 0.18215,
        device=device,
        model_type=model_type
    )

    print_cluster_statistics(cluster_counts)

    save_cluster_sample_grids(
        cluster_samples,
        cluster_mean_images,
        save_root=os.path.join(output_path, "real_cluster_samples")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='model1_f16d32.yaml')
    parser.add_argument('--ds_config_path', type=str, default='model2_xl_vavae_f16d32.yaml')
    parser.add_argument('--model_type', type=str, default='vavae')
    args = parser.parse_args()
    ds_config = load_config(args.ds_config_path)

    view_cluster(config_path=args.config_path, ds_config=ds_config, model_type=args.model_type)