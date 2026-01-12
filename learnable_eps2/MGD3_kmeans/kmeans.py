"""
Clustering KMeans for VA-VAE latent
"""

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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import defaultdict
import pickle


def do_kmeans(train_config, accelerator):
    """
    KMeans result_gmm on ImgLatentDataset.
    - Dataset handles latent_norm / latent_multiplier
    - This code ONLY clusters given latents
    """
    device = accelerator.device
    is_main = accelerator.is_main_process

    # --------------------------------------------------
    # Dataset / Loader
    # --------------------------------------------------
    dataset = ImgLatentDataset(
        data_dir=train_config['data']['data_path'],
        latent_norm=train_config['data'].get('latent_norm', True),
        latent_multiplier=train_config['data'].get('latent_multiplier', 1.0),
    )

    batch_size_per_gpu = int(
        train_config['train']['global_batch_size'] / accelerator.num_processes
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,      # ❗ clustering에서는 shuffle 금지
        num_workers=train_config['data']['num_workers'],
        pin_memory=True,
        drop_last=False
    )

    if is_main:
        print(f"[KMeans] Dataset size: {len(dataset)}")

    # --------------------------------------------------
    # Collect latents + labels (Case A)
    # --------------------------------------------------
    features = []
    labels = []

    for latents, y in tqdm(loader, disable=not is_main):
        # latents: [B, 4, H, W]
        latents = latents.to(device, non_blocking=True)
        latents = latents.flatten(start_dim=1)   # [B, D]

        features.append(latents.cpu())
        labels.append(y.cpu())

    features = torch.cat(features, dim=0).numpy()   # [N, D]
    labels = torch.cat(labels, dim=0).numpy()       # [N]

    if is_main:
        print(f"[KMeans] Features collected: {features.shape}")

    # --------------------------------------------------
    # KMeans config
    # --------------------------------------------------
    k_cfg = train_config['kmeans']
    ipc = k_cfg['num_clusters']
    use_pca = k_cfg.get('use_pca', False)
    pca_dim = k_cfg.get('pca_dim', 4)
    closest_point = k_cfg.get('closest_point', False)

    # --------------------------------------------------
    # Group features by class
    # --------------------------------------------------
    feats_per_class = defaultdict(list)

    for feat, cls in zip(features, labels):
        feats_per_class[int(cls)].append(feat)

    clusters_centers = {}
    mode_id_per_class = {}

    # --------------------------------------------------
    # Class-wise KMeans
    # --------------------------------------------------
    for cls in tqdm(sorted(feats_per_class.keys()), disable=not is_main):
        X = np.stack(feats_per_class[cls])  # (Nc, D)

        # optional PCA (purely for result_gmm stability)
        if use_pca:
            pca = PCA(n_components=pca_dim)
            X_km = pca.fit_transform(X)
        else:
            X_km = X

        kmeans = KMeans(
            n_clusters=ipc,
            random_state=0,
            n_init=10,
            verbose = 1
        ).fit(X_km)
        # lsun(120k): 56m

        mode_id_per_class[cls] = kmeans.labels_

        if closest_point:
            # use real latent closest to centroid
            centers = []
            for c in kmeans.cluster_centers_:
                idx = np.argmin(np.sum((X_km - c) ** 2, axis=1))
                centers.append(X[idx])
            clusters_centers[cls] = np.stack(centers)
        else:
            if use_pca:
                clusters_centers[cls] = pca.inverse_transform(
                    kmeans.cluster_centers_
                )
            else:
                clusters_centers[cls] = kmeans.cluster_centers_

    # --------------------------------------------------
    # Save (main process only)
    # --------------------------------------------------
    if is_main:
        save_dir = os.path.join(
            train_config['kmeans']['output_dir'],
            str(train_config['kmeans']['num_clusters'])
        )
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, "kmeans_clusters.pkl")

        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "centers": clusters_centers,   # class → (IPC, D)
                    "labels": mode_id_per_class,   # class → (Nc,)
                    "ipc": ipc,
                },
                f
            )

        print(f"✅ KMeans clusters saved to: {save_path}")


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config



if __name__ == "__main__":
    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='model2_xl_vavae_f16d32.yaml')
    args = parser.parse_args()

    accelerator = Accelerator()
    train_config = load_config(args.config)
    do_kmeans(train_config, accelerator)