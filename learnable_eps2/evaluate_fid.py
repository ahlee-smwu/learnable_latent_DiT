import os
import shutil
from cleanfid import fid

# 경로 설정
gen_dir = 'output/1st_lightningdit_xl_vavae_f16d32_gmm30/lightningdit-xl-1-ckpt-0159000-euler-20/class_0'
k_num = 30

# 결과를 저장할 딕셔너리
fid_results = {}

'''
# -------------------------------
# 1. cluster별 FID
# -------------------------------
print("Starting FID calculation for each cluster...")

for i in range(k_num):
    cluster_path = os.path.join(gen_dir, f'cluster_{i}')

    if not os.path.exists(cluster_path) or len(os.listdir(cluster_path)) == 0:
        print(f"Skip cluster_{i}: Folder is empty or does not exist.")
        continue

    fid_score = fid.compute_fid(
        fdir1=cluster_path,
        dataset_name="lsun_church",
        dataset_res=256,
        mode="clean",
        dataset_split="train",
        num_workers=8,
        batch_size=128
    )

    fid_results[i] = fid_score
    print(f"[FID] cluster_{i}: {fid_score:.4f}")

# -------------------------------
# 2. 전체 클러스터 FID
# -------------------------------
print("\nCalculating FID for ALL clusters combined...")

all_cluster_dir = os.path.join(gen_dir, "all_clusters_tmp")

# 임시 폴더 생성
os.makedirs(all_cluster_dir, exist_ok=True)

# 모든 cluster 이미지 모으기
for i in range(k_num):
    cluster_path = os.path.join(gen_dir, f'cluster_{i}')
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
'''

# -------------------------------
# 3. 선택된 클러스터만으로 전체 FID
# -------------------------------
print("Starting FID calculation for top-k clusters...")

selected_clusters = [
    1, 3, 15, 20, 27, 6, 19, 25, 7, 11, 18, 9, 26, 0, 10, 24, 16
]

selected_dir = os.path.join(gen_dir, "selected_clusters_tmp")
os.makedirs(selected_dir, exist_ok=True)

selected_image_count = 0

for i in selected_clusters:
    cluster_path = os.path.join(gen_dir, f"cluster_{i}")
    if not os.path.exists(cluster_path):
        continue

    for fname in os.listdir(cluster_path):
        src = os.path.join(cluster_path, fname)
        if not os.path.isfile(src):
            continue

        # 파일명 충돌 방지
        dst = os.path.join(selected_dir, f"cluster{i}_{fname}")
        shutil.copy(src, dst)
        selected_image_count += 1

# FID 계산
fid_selected = fid.compute_fid(
    fdir1=selected_dir,
    dataset_name="lsun_church",
    dataset_res=256,
    mode="clean",
    dataset_split="train",
    num_workers=8,
    batch_size=32
)

print(
    f"[FID] SELECTED clusters ({len(selected_clusters)} clusters): "
    f"{fid_selected:.4f}, found {selected_image_count} images"
)

# 임시 폴더 삭제
shutil.rmtree(selected_dir)