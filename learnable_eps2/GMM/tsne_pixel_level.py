import os
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from glob import glob


# -------------------------------------------------
# 1. 이미지 로드 및 전처리 유틸리티
# -------------------------------------------------
def get_transform(img_size=64):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def collect_real_images(real_dir, max_samples=2000, img_size=64):
    transform = get_transform(img_size)
    # ImageFolder는 하위 폴더가 하나라도 있어야 작동합니다.
    dataset = ImageFolder(real_dir, transform=transform)

    if len(dataset) > max_samples:
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    vectors = []

    print(f"[*] Loading Real images from {real_dir}...")
    for img, _ in tqdm(loader, desc="Real Images"):
        vectors.append(img.view(img.size(0), -1).numpy())

    return np.concatenate(vectors, axis=0)


def collect_gen_images_with_info(gen_base_dir, max_samples=2000, img_size=64):
    transform = get_transform(img_size)
    vectors = []
    info_labels = []

    # 모든 하위 이미지 파일 검색 (depth: gen_base_dir/class_i/cluster_j/*.png)
    image_paths = glob(os.path.join(gen_base_dir, "*", "*", "*.png")) + \
                  glob(os.path.join(gen_base_dir, "*", "*", "*.jpg"))

    if not image_paths:
        print(f"[!] No images found. Check path: {gen_base_dir} (Structure: class/cluster/img.png)")
        return None, None

    if len(image_paths) > max_samples:
        indices = np.random.choice(len(image_paths), max_samples, replace=False)
        image_paths = [image_paths[i] for i in indices]

    print(f"[*] Loading Generated images (Class > Cluster structure)...")
    for path in tqdm(image_paths, desc="Gen Images"):
        # OS 구분자 보정 (윈도우/리눅스 혼용 방지)
        normalized_path = os.path.normpath(path)
        parts = normalized_path.split(os.sep)

        # 구조: [..., class_name, cluster_name, image_name]
        class_name = parts[-3]
        cluster_name = parts[-2]
        label = f"{class_name}_{cluster_name}"

        try:
            img = Image.open(path).convert('RGB')
            img_vec = transform(img).view(-1).numpy()
            vectors.append(img_vec)
            info_labels.append(label)
        except Exception as e:
            continue

    return np.array(vectors), np.array(info_labels)


# -------------------------------------------------
# 2. t-SNE 시각화 로직
# -------------------------------------------------
def run_and_plot_tsne(real_vecs, gen_vecs, gen_labels, save_path, pca_dim=50):
    n_real = len(real_vecs)
    all_vecs = np.concatenate([real_vecs, gen_vecs], axis=0)

    # 1단계: PCA
    print(f"[*] Running PCA (target dim: {pca_dim})...")
    pca = PCA(n_components=pca_dim, random_state=42)
    all_pca = pca.fit_transform(all_vecs)

    # 2단계: t-SNE
    print("[*] Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=42,
        verbose=1
    )
    all_tsne = tsne.fit_transform(all_pca)

    real_tsne = all_tsne[:n_real]
    gen_tsne = all_tsne[n_real:]

    unique_labels = sorted(list(set(gen_labels)))
    num_colors = len(unique_labels)
    cmap = plt.cm.get_cmap('gist_rainbow', num_colors)
    color_map = {label: cmap(i) for i, label in enumerate(unique_labels)}

    plt.figure(figsize=(16, 10))

    # --- 수정된 부분: 그리는 순서와 zorder 조절 ---

    # 1. Real Data를 가장 먼저 그립니다 (zorder=1)
    plt.scatter(
        real_tsne[:, 0], real_tsne[:, 1],
        c='lightgray', label='Real Data',
        s=10, alpha=0.3, edgecolors='none',
        zorder=2  # 가장 아래 레이어
    )

    # 2. Generated Data를 나중에 그립니다 (zorder=2)
    for label in unique_labels:
        mask = (gen_labels == label)
        plt.scatter(
            gen_tsne[mask, 0], gen_tsne[mask, 1],
            color=color_map[label], label=f"Gen: {label}",
            s=45,  # 가독성을 위해 크기를 살짝 키웠습니다
            alpha=0.7,  # 더 선명하게 보이도록 불투명도 조절
            marker='+',
            linewidths=1.5,
            zorder=1  # 회색 점 위로 올라오게 설정
        )
    # ------------------------------------------

    plt.title("Pixel-level t-SNE: Real (Background) vs Generated (Top)", fontsize=15)

    # 범례 설정
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', markerscale=1.5,
               fontsize=8, ncol=2 if num_colors > 15 else 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] ✅ 시각화 레이어 수정 완료! 저장 경로: {save_path}")

# -------------------------------------------------
# 3. 메인 실행부
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, default="/mnt/HDD_raid1/lsun/church_outdoor_train")
    parser.add_argument("--gen_dir", type=str,default="output/1st_lightningdit_xl_vavae_f16d32_gmm30/lightningdit-xl-1-ckpt-0162000-euler-20")
    parser.add_argument("--save_dir", type=str, default="GMM/result_lsun/30_diag")
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--img_size", type=int, default=256, help="Memory safety: 128 is recommended over 256")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"tsne_pixel_s{args.img_size}_n{args.max_samples}.png")

    # 1. 수집
    real_vecs = collect_real_images(args.real_dir, max_samples=args.max_samples, img_size=args.img_size)
    gen_vecs, gen_labels = collect_gen_images_with_info(args.gen_dir, max_samples=args.max_samples,
                                                        img_size=args.img_size)

    if gen_vecs is not None:
        # 2. 실행
        run_and_plot_tsne(real_vecs, gen_vecs, gen_labels, save_path)


if __name__ == "__main__":
    main()