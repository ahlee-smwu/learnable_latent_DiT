import torch
import pickle
import numpy as np
import os
import yaml
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.img_latent_dataset import ImgLatentDataset
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import cdist

# -----------------------------
# 1. Latent 및 Label 수집
# -----------------------------
def collect_all_latents(loader):
    all_latents = []
    all_labels = []
    for x, y in tqdm(loader, desc="Collecting latents"):
        # x shape: [B, C, H, W] -> flatten [B, D]
        all_latents.append(x.view(x.size(0), -1).numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_latents, axis=0), np.concatenate(all_labels, axis=0)


# -----------------------------
# 2. GMM 상세 평가 및 시각화
# -----------------------------
def evaluate_gmm_performance(latents, labels, gmm_params, save_path):
    os.makedirs(save_path, exist_ok=True)
    report = []

    # 클래스별로 루프 (GMM은 보통 class-wise로 학습됨)
    for cls in sorted(gmm_params['means'].keys()):
        mask = labels == cls
        if mask.sum() == 0:
            continue

        X = latents[mask]
        means = gmm_params['means'][cls]  # (K, D)
        weights = gmm_params['weights'][cls]  # (K,)

        # 클러스터 할당 (가장 가까운 Mean 기준)
        # GMM의 soft assignment 대신 명확한 경계 확인을 위해 hard assignment 수행
        dists = cdist(X, means, metric='euclidean')
        cluster_assigns = np.argmin(dists, axis=1)

        # --- (1) 클러스터 평가 지표 ---
        # 데이터가 너무 많으면 지표 계산이 매우 느리므로 샘플링 (최대 1만개)
        sample_size = min(10000, len(X))
        s_idx = np.random.choice(len(X), sample_size, replace=False)

        sil = silhouette_score(X[s_idx], cluster_assigns[s_idx])
        db_idx = davies_bouldin_score(X, cluster_assigns)
        ch_idx = calinski_harabasz_score(X[s_idx], cluster_assigns[s_idx])

        report.append({
            'Class': cls,
            'Silhouette': sil,
            'DB_Index': db_idx,
            'CH_Index': ch_idx,
            'Avg_Dist': np.min(dists, axis=1).mean()
        })

        # --- (2) Inter-cluster Heatmap (중심 간 거리) ---
        plt.figure(figsize=(10, 8))
        inter_dist = cdist(means, means, metric='euclidean')
        sns.heatmap(inter_dist, annot=True, fmt=".1f", cmap="YlGnBu")
        plt.title(f"Class {cls}: Inter-cluster Distance Heatmap")

        heatmap_path = os.path.join(save_path, f"class_{cls}_dist_heatmap.png")
        plt.savefig(heatmap_path, dpi=150)
        plt.close()

        # --- (3) Cluster Weight 분포 (GMM 특화) ---
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(weights)), weights, color='royalblue', alpha=0.7)
        plt.xticks(range(len(weights)))
        plt.title(f"Class {cls}: Cluster Mixing Coefficients (Weights)")
        plt.xlabel("Cluster ID")
        plt.ylabel("Weight")

        weight_path = os.path.join(save_path, f"class_{cls}_weights.png")
        plt.savefig(weight_path)
        plt.close()

    # --- 최종 결과 테이블 출력 ---
    print("\n" + "=" * 65)
    print(f"{'Class':<8} | {'Silh(↑)':<10} | {'DBI(↓)':<10} | {'CH(↑)':<10} | {'Dist(↓)':<10}")
    print("-" * 65)
    for r in report:
        print(
            f"{r['Class']:<8} | {r['Silhouette']:>10.4f} | {r['DB_Index']:>10.4f} | {r['CH_Index']:>10.2f} | {r['Avg_Dist']:>10.4f}")
    print("=" * 65)

# -----------------------------
# 3. Main 실행부
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GMM Cluster Evaluation")
    parser.add_argument('--config_path', type=str, default='model1_f16d32.yaml')
    parser.add_argument('--ds_config_path', type=str, default='model2_xl_vavae_f16d32.yaml')

    args = parser.parse_args()

    # Config 로드
    with open(args.ds_config_path, 'r') as f:
        ds_config = yaml.safe_load(f)
    output_path = f"{ds_config['gmm']['output_dir']}/{ds_config['gmm']['num_clusters']}_{ds_config['gmm']['cov_type']}"

    # 1. 데이터 로드
    print(f"[*] Loading dataset from: {ds_config['data']['data_path']}")
    dataset = ImgLatentDataset(
        data_dir=ds_config['data']['data_path'],
        latent_norm=ds_config['data'].get('latent_norm', False),
        latent_multiplier=ds_config['data'].get('latent_multiplier', 0.18215),
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

    latents, labels = collect_all_latents(loader)

    # 2. GMM 파라미터 로드
    gmm_file = os.path.join(output_path, "gmm_clusters.pkl")
    if not os.path.exists(gmm_file):
        raise FileNotFoundError(f"Cannot find {gmm_file}")

    with open(gmm_file, "rb") as f:
        gmm_params = pickle.load(f)

    # 3. 성능 평가 및 Heatmap 저장
    print("[*] Evaluating GMM performance metrics...")
    evaluate_gmm_performance(latents, labels, gmm_params, output_path)

    print(f"\n✅ Evaluation complete. Results saved in: {output_path}")
