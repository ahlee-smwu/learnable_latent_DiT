import os
import shutil
import torch
from cleanfid import fid
from tqdm import tqdm
from cleanfid import fid
from torch.utils.data import DataLoader
import pickle

# 경로 설정
gen_base_dir = 'output/1st_lightningdit_xl_vavae_f16d32_gmm30_use_weight/lightningdit-xl-1-ckpt-0018000-euler-20/class_0'
real_base_dir = '/mnt/HDD_raid1/lsun/church_outdoor_train_gmm/30_diag/class_0/'
k_num = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 결과를 저장할 딕셔너리
fid_results = {}

# -------------------------------
# 1. cluster별 FID
# -------------------------------
print("Starting FID calculation for each cluster...")

for i in range(k_num):
    gen_cluster_path = os.path.join(gen_base_dir, f'cluster_{i}')
    real_cluster_path = os.path.join(real_base_dir, f'cluster_{i}')

    if (
        not os.path.exists(gen_cluster_path)
        or not os.path.exists(real_cluster_path)
        or len(os.listdir(gen_cluster_path)) == 0
        or len(os.listdir(real_cluster_path)) == 0
    ):
        print(f"Skip cluster_{i}")
        continue

    fid_score = fid.compute_fid(
        fdir1=gen_cluster_path,
        fdir2=real_cluster_path,
        mode="clean",
        num_workers=8,
        batch_size=128
    )

    fid_results[i] = fid_score
    print(f"[FID] cluster_{i}: {fid_score:.4f}")


# -------------------------------
# 2. 전체 클러스터 FID
# -------------------------------
print("\nCalculating FID for ALL clusters combined...")

all_cluster_dir = os.path.join(gen_base_dir, "all_clusters_tmp")

# 임시 폴더 생성
os.makedirs(all_cluster_dir, exist_ok=True)

# 모든 cluster 이미지 모으기
for i in range(k_num):
    cluster_path = os.path.join(gen_base_dir, f'cluster_{i}')
    if not os.path.exists(cluster_path):
        continue

    for fname in os.listdir(cluster_path):
        src = os.path.join(cluster_path, fname)
        if not os.path.isfile(src):
            continue

        # 파일명 충돌 방지
        dst = os.path.join(all_cluster_dir, f"cluster{i}_{fname}")
        shutil.copy(src, dst)

# 전체 FID 계산
fid_all = fid.compute_fid(
    fdir1=all_cluster_dir,
    dataset_name="lsun_church",
    dataset_res=256,
    mode="clean",
    dataset_split="train",
    num_workers=8,
    batch_size=32
)

print(f"[FID] ALL clusters: {fid_all:.4f}")

# 임시 폴더 삭제 (원하면 주석 처리)
shutil.rmtree(all_cluster_dir)

# -------------------------------
# 최종 결과 요약
# -------------------------------
print("\n" + "=" * 30)
print("Final Summary (Cluster FID)")
print("=" * 30)
for cluster_id, score in sorted(fid_results.items()):
    print(f"Cluster {cluster_id:02d}: {score:.4f}")

print("-" * 30)
print(f"ALL clusters FID: {fid_all:.4f}")


# -------------------------------
# 3. 선택된 클러스터만으로 전체 FID
# -------------------------------
# print("Starting FID calculation for top-k clusters...")
#
# selected_clusters = [
#     1, 3, 15, 20, 27, 6, 19, 25, 7, 11, 18, 9, 26, 0, 10, 24, 16
# ]
#
# selected_dir = os.path.join(gen_dir, "selected_clusters_tmp")
# os.makedirs(selected_dir, exist_ok=True)
#
# selected_image_count = 0
#
# for i in selected_clusters:
#     cluster_path = os.path.join(gen_dir, f"cluster_{i}")
#     if not os.path.exists(cluster_path):
#         continue
#
#     for fname in os.listdir(cluster_path):
#         src = os.path.join(cluster_path, fname)
#         if not os.path.isfile(src):
#             continue
#
#         # 파일명 충돌 방지
#         dst = os.path.join(selected_dir, f"cluster{i}_{fname}")
#         shutil.copy(src, dst)
#         selected_image_count += 1
#
# # FID 계산
# fid_selected = fid.compute_fid(
#     fdir1=selected_dir,
#     dataset_name="imagenet", #"lsun_church",
#     dataset_res=256,
#     mode="clean",
#     dataset_split="train",
#     num_workers=8,
#     batch_size=32
# )
#
# print(
#     f"[FID] SELECTED clusters ({len(selected_clusters)} clusters): "
#     f"{fid_selected:.4f}, found {selected_image_count} images"
# )
#
# # 임시 폴더 삭제
# shutil.rmtree(selected_dir)

# # -------------------------------
# # 4. org FID
# # -------------------------------
# import sys
# import os
#
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#
# gen_dir = '/home/ahlee/research/gen_AI/leanable_latent_DiT/learnable_eps2/output/org_lightningdit_xl_vavae_f16d32/lightningdit-xl-1-ckpt-lightningdit-xl-imagenet256-64ep-euler-20'
# ref_npz = "/home/ahlee/research/gen_AI/leanable_latent_DiT/pretrained_weight/VIRTUAL_imagenet256_labeled.npz"
#
# from tools.calculate_fid import calculate_fid_given_paths
# fid = calculate_fid_given_paths(
#                 [ref_npz, gen_dir],
#                 batch_size=50,
#                 dims=2048,
#                 device='cuda',
#                 num_workers=8,
#                 sp_len = 90112
#             )
#
# print("FID:", fid)