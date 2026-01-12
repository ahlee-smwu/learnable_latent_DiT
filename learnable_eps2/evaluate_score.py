import os
import shutil
import torch
import numpy as np
from cleanfid import fid, features
from prdc import compute_prdc

# ---------------------------------------------------------
# 1. 멀티 GPU 및 경로 설정
# ---------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 원본 데이터 경로
real_dir = "/mnt/SSD_raid1/lsun/church_outdoor_train/church"

# 생성 데이터 소스 및 대상 경로
gen_base_dir = 'output/1st_lightningdit_xl_vavae_f16d32_gmm30/lightningdit-xl-1-ckpt-0159000-euler-20/class_0'
selected_clusters = [1, 3, 15, 20, 27, 6, 19, 25, 7, 11, 18, 9, 26, 0, 10, 24, 16]
selected_dir = os.path.join(gen_base_dir, "selected_clusters_tmp")


# ---------------------------------------------------------
# 2. 보조 함수들
# ---------------------------------------------------------
def prepare_gen_data():
    """클러스터 폴더들에서 이미지를 모아 임시 폴더 생성"""
    if os.path.exists(selected_dir):
        shutil.rmtree(selected_dir)
    os.makedirs(selected_dir, exist_ok=True)

    print(f"📂 Preparing generated images in {selected_dir}...")
    count = 0
    for i in selected_clusters:
        cluster_path = os.path.join(gen_base_dir, f"cluster_{i}")
        if not os.path.exists(cluster_path):
            continue
        for fname in os.listdir(cluster_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy(os.path.join(cluster_path, fname), os.path.join(selected_dir, f"c{i}_{fname}"))
                count += 1
    print(f"✅ Prepared {count} generated images.")
    return count


def main():
    try:
        # [Step 1] 데이터 준비
        img_count = prepare_gen_data()
        if img_count == 0:
            print("❌ Error: No generated images found.")
            return

        print(f"🚀 Initializing Inception-V3 on {device}...")
        feat_model = features.build_feature_extractor("clean", device)

        def get_images(folder):
            exts = ('.webp', '.png', '.jpg', '.jpeg', '.WEBP', '.PNG', '.JPG', '.JPEG')
            return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]

        real_files = get_images(real_dir)
        gen_files = get_images(selected_dir)

        # [Step 2] 특징 추출
        print(f"⚡ Extracting features for {len(real_files)} Real and {len(gen_files)} Gen images...")
        feat_real = fid.get_files_features(real_files, model=feat_model, device=device, batch_size=128)
        feat_gen = fid.get_files_features(gen_files, model=feat_model, device=device, batch_size=128)

        # [Step 3] 메모리 방어용 샘플링 (50,000장)
        # 12만 장 대 9만 장의 PRDC 계산은 일반적인 RAM에서 OOM(SIGKILL)을 유발합니다.
        max_samples = 50000
        if len(feat_real) > max_samples:
            print(f"♻️ Sampling Real features to {max_samples}...")
            feat_real = feat_real[np.random.choice(len(feat_real), max_samples, replace=False)]
        if len(feat_gen) > max_samples:
            print(f"♻️ Sampling Gen features to {max_samples}...")
            feat_gen = feat_gen[np.random.choice(len(feat_gen), max_samples, replace=False)]

        # [Step 4] 지표 계산
        print("📊 Calculating PRDC with optimized memory...")
        feat_real = feat_real.astype(np.float16)  # 메모리 점유율 절반으로 감소
        feat_gen = feat_gen.astype(np.float16)

        prdc_results = compute_prdc(real_features=feat_real, fake_features=feat_gen, nearest_k=5)

        # 결과 출력
        print("\n" + "=" * 50)
        print("📈 FINAL METRICS REPORT (Sampled 50k)")
        print("=" * 50)
        print(f"Precision: {prdc_results['precision']:.4f}")
        print(f"Recall:    {prdc_results['recall']:.4f}")
        print(f"Density:   {prdc_results['density']:.4f}")
        print(f"Coverage:  {prdc_results['coverage']:.4f}")
        print("=" * 50)

    except Exception as e:
        print(f"❌ An error occurred: {e}")

    finally:
        # [Step 5] 임시 파일 무조건 삭제
        if os.path.exists(selected_dir):
            print(f"\n🧹 Cleaning up temporary directory: {selected_dir}")
            shutil.rmtree(selected_dir)
            print("✨ Done.")


if __name__ == "__main__":
    main()