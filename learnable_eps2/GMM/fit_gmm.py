import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import yaml
import pickle
from tqdm import tqdm
from collections import defaultdict

from accelerate import Accelerator
from datasets.img_latent_dataset import ImgLatentDataset
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


def do_gmm_clustering(train_config, accelerator):
    """
    ImgLatentDataset의 latent를 사용하여 클래스별 GMM 클러스터링 수행
    """
    device = accelerator.device
    is_main = accelerator.is_main_process

    # 1. Dataset / Loader (기존 KMeans 코드와 동일)
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
        shuffle=False,
        num_workers=train_config['data']['num_workers'],
        pin_memory=True,
        drop_last=False
    )

    if is_main:
        print(f"[GMM] Dataset size: {len(dataset)}")

    # 2. Latents 및 Labels 수집
    features = []
    labels = []

    for latents, y in tqdm(loader, disable=not is_main, desc="Collecting latents"):
        # latents shape: [B, 32, 16, 16] -> [B, 8192]
        latents = latents.to(device, non_blocking=True)
        latents = latents.flatten(start_dim=1)

        features.append(latents.cpu())
        labels.append(y.cpu())

    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    # 3. GMM 설정 (KMeans 섹션 설정 활용)
    g_cfg = train_config['gmm']
    num_components = g_cfg['num_clusters']
    use_pca = g_cfg.get('use_pca', False)
    pca_dim = g_cfg.get('pca_dim', 256)  # 8192차원이므로 높은 PCA 차원 권장
    cov_type = g_cfg.get('cov_type', 'diag')  # GMM 특화 설정 (diag/full)

    # KMeans 결과 불러오기
    kmeans_path = os.path.join(g_cfg['init_dir'], str(num_components), "kmeans_clusters.pkl")
    if os.path.exists(kmeans_path):
        with open(kmeans_path, "rb") as f:
            km_data = pickle.load(f)
        print(f"✅ Found existing KMeans results at {kmeans_path}")
    else:
        km_data = None
        print("⚠️ No KMeans results found. Running default initialization.")

    # 클래스별로 데이터 그룹화
    feats_per_class = defaultdict(list)
    for feat, cls in zip(features, labels):
        feats_per_class[int(cls)].append(feat)

    clusters_means = {}
    clusters_covs = {}
    clusters_weights = {}
    pca_components = {}
    mode_id_per_class = {}

    for cls in tqdm(sorted(feats_per_class.keys()), disable=not is_main):
        X = np.stack(feats_per_class[cls])

        if use_pca:
            pca = PCA(n_components=pca_dim)
            X_input = pca.fit_transform(X)
            pca_components[cls] = {
                "components": pca.components_,
                "mean": pca.mean_,
            }
        else:
            X_input = X

        gmm = GaussianMixture(
            n_components=num_components,
            covariance_type=cov_type,
            random_state=0,
            reg_covar=1e-6,
            max_iter=200,
            init_params='kmeans' if km_data is None else 'random',
            verbose=1
        )

        if km_data is not None and cls in km_data['centers']:
            init_means = km_data['centers'][cls]
            if use_pca:
                init_means = pca.transform(init_means)
            gmm.means_init = init_means

        gmm.fit(X_input)

        mode_id_per_class[cls] = gmm.predict(X_input)
        clusters_weights[cls] = gmm.weights_

        if use_pca:
            clusters_means[cls] = pca.inverse_transform(gmm.means_)
            clusters_covs[cls] = gmm.covariances_  # PCA-space
        else:
            clusters_means[cls] = gmm.means_
            clusters_covs[cls] = gmm.covariances_

    # 5. 결과 저장 (Main process only)
    if is_main:
        save_dir = os.path.join(
            g_cfg['output_dir'],
            f"{num_components}_{cov_type}"
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "gmm_clusters.pkl")

        with open(save_path, "wb") as f:
            pickle.dump({
                "means": clusters_means,  # class -> (K, D)
                "covs": clusters_covs,  # class -> (K, pca_dim) or (K, D)
                "weights": clusters_weights,  # class -> (K,)
                "labels": mode_id_per_class,
                "pca_components": pca_components if use_pca else None,
                "num_components": num_components,
                "pca_dim": pca_dim if use_pca else None,
                "cov_type": cov_type,
            }, f)

        print(f"✅ GMM results saved to: {save_path}")


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='model2_xl_vavae_f16d32.yaml')
    args = parser.parse_args()

    accelerator = Accelerator()
    train_config = load_config(args.config)
    do_gmm_clustering(train_config, accelerator)