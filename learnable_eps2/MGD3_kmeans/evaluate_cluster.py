"""
Cluster Evaluation + t-SNE Visualization
"""

import torch
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from datasets.img_latent_dataset import ImgLatentDataset
import yaml
import argparse
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1. Latent 수집
# -----------------------------
def collect_all_latents(loader, device="cuda"):
    all_latents = []
    all_labels = []
    for x, y in tqdm(loader, desc="Collecting latents"):
        all_latents.append(x.to(device))
        all_labels.append(y.to(device))
    all_latents = torch.cat(all_latents, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_latents, all_labels

# -----------------------------
# 2. Cluster 할당
# -----------------------------
def assign_clusters(latents, labels, centers, chunk_size=512, device="cuda"):
    """
    Assign each latent to the nearest cluster center using GPU in chunks.

    Args:
        latents: (N, D) tensor, flattened latent vectors
        labels: (N,) tensor, class labels
        centers: dict[class_id] -> (K, D) tensor of cluster centers
        chunk_size: int, number of samples to process per chunk
        device: str, "cuda" or "cpu"

    Returns:
        cluster_ids: (N,) tensor, cluster index for each latent
    """
    cluster_ids = torch.zeros(latents.size(0), dtype=torch.long, device=device)

    # 모든 centers를 GPU에 올림
    centers_gpu = {cls: mu.to(device) for cls, mu in centers.items()}

    for cls, cls_centers in centers_gpu.items():
        mask = labels == cls
        if mask.sum() == 0:
            continue

        cls_latents = latents[mask].to(device)  # chunk-wise로 GPU에 올림
        all_dists = []

        for i in range(0, cls_latents.size(0), chunk_size):
            lat_chunk = cls_latents[i:i+chunk_size]  # (chunk_size, D)
            dists_chunk = ((lat_chunk[:, None, :] - cls_centers[None, :, :]) ** 2).sum(dim=2)  # (chunk, K)
            all_dists.append(dists_chunk)

        dists = torch.cat(all_dists, dim=0)
        k_idx = dists.argmin(dim=1)
        cluster_ids[mask] = k_idx

    return cluster_ids

# -----------------------------
# 3. Plot color space
# -----------------------------
def get_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = int(i * 360 / n)
        # HSV → RGB 변환
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue/360, 0.8, 0.9)
        colors.append((int(r*255), int(g*255), int(b*255)))
    return colors

def get_tab20_colors(n):
    cmap = cm.get_cmap("tab20", n)
    colors = []
    for i in range(n):
        r, g, b, _ = cmap(i)
        colors.append((int(r*255), int(g*255), int(b*255)))
    return colors

# -----------------------------
# 3. Cluster 평가 + t-SNE
# -----------------------------
def evaluate_clusters_with_tsne(latents, cluster_labels, save_path, sample_size, image_size):
    """
    PIL만 사용해서 t-SNE 시각화, 범례, x/y 축 표시
    """
    X = latents.view(latents.shape[0], -1).cpu().numpy()
    labels = cluster_labels.cpu().numpy()

    # -----------------------------
    # 1) 샘플링
    # -----------------------------
    if X.shape[0] > sample_size:
        idx = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[idx]
        labels_sample = labels[idx]
    else:
        X_sample = X
        labels_sample = labels

    # -----------------------------
    # 2) t-SNE
    # -----------------------------
    tsne = TSNE(n_components=2, init='pca', random_state=42, learning_rate='auto', verbose=1)
    X_2d = tsne.fit_transform(X_sample)

    # -----------------------------
    # 3) 좌표 정규화 -> 이미지 좌표
    # -----------------------------
    x_min, x_max = X_2d[:, 0].min(), X_2d[:, 0].max()
    y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()
    X_norm = (X_2d - [x_min, y_min]) / ([x_max - x_min, y_max - y_min])
    pixel_coords = (X_norm * (image_size - 200)).astype(int) + 50  # 좌우/상하 여백 50

    # -----------------------------
    # 4) PIL 이미지 생성
    # -----------------------------
    img = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=12)  # 기본보다 큰 폰트
    except:
        font = ImageFont.load_default()

    # 각 cluster 색상
    unique_labels = np.unique(labels_sample)
    num_clusters = len(unique_labels)
    # palette = get_distinct_colors(num_clusters)
    palette = get_tab20_colors(num_clusters)

    for (x, y), label in zip(pixel_coords, labels_sample):
        color = palette[np.where(unique_labels == label)[0][0]]
        draw.rectangle([x, y, x + 2, y + 2], fill=color)

    # -----------------------------
    # 5) x/y 축 표시
    # -----------------------------
    for i in range(0, 11):
        # x-axis
        x = 50 + i * (image_size - 200)//10
        draw.line([(x, image_size-50), (x, image_size-45)], fill=(0,0,0))
        draw.text((x-10, image_size-45), f"{x_min + i*(x_max-x_min)/10:.2f}", fill=(0,0,0), font=font)
        # y-axis
        y = 50 + i * (image_size - 200)//10
        draw.line([(45, image_size-50-y + 50), (50, image_size-50-y + 50)], fill=(0,0,0))
        draw.text((0, image_size-55-y + 50), f"{y_min + i*(y_max-y_min)/10:.2f}", fill=(0,0,0), font=font)

    # -----------------------------
    # 6) 범례 표시
    # -----------------------------
    legend_x = image_size - 140
    legend_y = 50
    for i, label in enumerate(unique_labels):
        draw.rectangle([legend_x, legend_y + i*20, legend_x+15, legend_y + i*20 + 15], fill=palette[label % num_clusters])
        draw.text((legend_x+20, legend_y + i*20), f"Cluster {label}", fill=(0,0,0), font=font)

    tsne_path = f"{save_path}/tsne_clusters.png"
    img.save(tsne_path)
    print(f"✅ Saved t-SNE PIL plot to: {tsne_path}")

    # -----------------------------
    # cluster heatmap
    # -----------------------------
    centroids = np.array([X[labels == c].mean(axis=0) for c in unique_labels])
    dist_matrix = cdist(centroids, centroids, metric='euclidean')
    plt.figure(figsize=(12, 10))
    sns.heatmap(dist_matrix, annot=True, fmt=".2f", cmap="viridis", annot_kws={"size": 8})
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel("Cluster")
    plt.ylabel("Cluster")
    plt.title("Inter-cluster Distance Heatmap")
    plt.tight_layout()

    heatmap_path = f"{save_path}/heatmap_clusters.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"✅ Saved cluster distance heatmap to: {heatmap_path}")

    # -----------------------------
    # 7) 클러스터 지표
    # -----------------------------
    print(f"\n📊 Cluster Evaluation Metrics")
    silhouette = silhouette_score(X, labels)
    print(f"  Silhouette Score: {silhouette:.4f}")
    db_index = davies_bouldin_score(X, labels)
    print(f"  Davies-Bouldin Index: {db_index:.4f}")
    ch_score = calinski_harabasz_score(X, labels)
    print(f"  Calinski-Harabasz Index: {ch_score:.4f}")

# -----------------------------
# 4. Main 함수
# -----------------------------
def cluster_eval_main(ds_config, output_path, device="cuda"):
    # -----------------------------
    # 1) DataLoader
    # -----------------------------
    dataset = ImgLatentDataset(
        data_dir=ds_config['data']['data_path'],
        latent_norm=ds_config['data'].get('latent_norm', False),
        latent_multiplier=ds_config['data'].get('latent_multiplier', 0.18215),
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=ds_config['data'].get('num_workers', 4))

    # -----------------------------
    # 2) Cluster centers 로드
    # -----------------------------
    with open(f"{output_path}/kmeans_clusters.pkl", "rb") as f:
        ckpt = pickle.load(f)
    cluster_centers = {cls: torch.from_numpy(mu).to(device=device, dtype=torch.float32) for cls, mu in ckpt["centers"].items()}

    # -----------------------------
    # 3) Latent 수집 + Cluster 할당
    # -----------------------------
    latents, labels = collect_all_latents(loader, device=device)
    latents = latents.view(latents.shape[0],-1) # flatten
    cluster_labels = assign_clusters(latents, labels, cluster_centers, chunk_size=512, device=device)

    # -----------------------------
    # 4) 평가 + t-SNE
    # -----------------------------
    evaluate_clusters_with_tsne(latents, cluster_labels, save_path=output_path, sample_size=20000, image_size=1024)

# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_config_path', type=str, default='model2_xl_vavae_f16d32.yaml')
    parser.add_argument('--output_path', type=str, default='MGD3_kmeans/result_gmm/20')
    args = parser.parse_args()

    with open(args.ds_config_path, 'r') as f:
        ds_config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cluster_eval_main(ds_config, args.output_path, device=device)
