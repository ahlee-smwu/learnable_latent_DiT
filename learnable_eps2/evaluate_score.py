import os
import shutil
import argparse
import torch
import numpy as np
from cleanfid import fid, features
from prdc import compute_prdc

# ---------------------------------------------------------
# 1. Configure GPU and paths
# ---------------------------------------------------------
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------
def extract_features_from_dir(img_dir, feat_model, device, max_samples=50000):
    files = []
    for root, _, fs in os.walk(img_dir):
        for f in fs:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                files.append(os.path.join(root, f))

    if len(files) == 0:
        return None

    if len(files) > max_samples:
        files = list(np.random.choice(files, max_samples, replace=False))

    try:
        feats = fid.get_files_features(files, model=feat_model, device=device, batch_size=256)
        feats = feats.astype(np.float16)
    except Exception as e:
        print(f"[WARN] Skipped some images due to error: {e}")
        feats = None

    return feats.astype(np.float16)

def main(args):
    # Path to real images
    # real_dir = "/mnt/SSD_raid1/lsun/church_outdoor_train/church" # navi
    real_dir = args.real_dir
    real_cluster_base_dir = args.real_cluster_base_dir

    # Path to generated images (base directory)
    gen_base_dir = args.gen_base_dir
    # selected_clusters = [1, 3, 15, 20, 27, 6, 19, 25, 7, 11, 18, 9, 26, 0, 10, 24, 16] # for selected cluster
    selected_clusters = list(range(30))  # for all cluster
    selected_dir = os.path.join(gen_base_dir, "selected_clusters_tmp")
    gen_cluster_base_dir = gen_base_dir

    def prepare_gen_data():
        """Collect generated images for selected clusters into a temporary folder."""
        if os.path.exists(selected_dir):
            shutil.rmtree(selected_dir)
        os.makedirs(selected_dir, exist_ok=True)

        print(f"[INFO] Preparing generated images in {selected_dir}...")
        count = 0
        for i in selected_clusters:
            cluster_path = os.path.join(gen_base_dir, f"cluster_{i}")
            if not os.path.exists(cluster_path):
                continue
            for fname in os.listdir(cluster_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    shutil.copy(os.path.join(cluster_path, fname), os.path.join(selected_dir, f"c{i}_{fname}"))
                    count += 1
        print(f"[INFO] Prepared {count} generated images.")
        return count

    try:
        # [Step 1] Load images
        # 1) for cluster model
        img_count = prepare_gen_data()
        if img_count == 0:
            print("[ERROR] No generated images found.")
            return
        # 2) org diffusion model
        # selected_dir = '/home/ahlee/research/gen_AI/leanable_latent_DiT/learnable_eps2/output/org_lightningdit_xl_vavae_f16d32/lightningdit-xl-1-ckpt-lightningdit-xl-imagenet256-64ep-euler-20'
        # real_dir = "/mnt/SSD_raid1/imagenet/ILSVRC2012_train/data"

        print(f"[INFO] Initializing Inception-V3 on {device}...")
        feat_model = features.build_feature_extractor("clean", device)

        def get_images(root_dir):
            exts = ('.webp', '.png', '.jpg', '.jpeg')
            image_files = []
            for root, _, files in os.walk(root_dir):
                for f in files:
                    if f.lower().endswith(exts):
                        image_files.append(os.path.join(root, f))
            return image_files

        real_files = get_images(real_dir)
        gen_files = get_images(selected_dir)

        # [Step 2] Limit number of samples (50,000)
        max_samples = 50000
        max_samples_cluster = 90112

        if len(real_files) > max_samples:
            print(f"[INFO] Sampling Real images to {max_samples}...")
            real_files = list(
                np.random.choice(real_files, max_samples, replace=False)
            )
        if len(gen_files) > max_samples:
            print(f"[INFO] Sampling Gen images to {max_samples}...")
            gen_files = list(
                np.random.choice(gen_files, max_samples, replace=False)
            )

        # [Step 3] Extract features (from sampled files)
        print(f"[INFO] Extracting features for {len(real_files)} Real and {len(gen_files)} Gen images...")
        feat_real = fid.get_files_features(
            real_files, model=feat_model, device=device, batch_size=256
        )
        feat_gen = fid.get_files_features(
            gen_files, model=feat_model, device=device, batch_size=256
        )

        # [Step 4] Compute PRDC
        print("[INFO] Calculating PRDC with optimized memory...")
        feat_real = feat_real.astype(np.float16)  # Reduce memory usage
        feat_gen = feat_gen.astype(np.float16)

        prdc_results = compute_prdc(real_features=feat_real, fake_features=feat_gen, nearest_k=5)

        # Print results # ALL
        print("\n" + "=" * 50)
        print("[RESULT] FINAL METRICS REPORT (Sampled 50k)")
        print("=" * 50)
        print(f"Precision: {prdc_results['precision']:.4f}")
        print(f"Recall:    {prdc_results['recall']:.4f}")
        print(f"Density:   {prdc_results['density']:.4f}")
        print(f"Coverage:  {prdc_results['coverage']:.4f}")
        print("=" * 50)

        # Print results # Cluster
        print("\n" + "=" * 60)
        print("[INFO] Cluster-wise PRDC (gen cluster i ↔ real cluster i)")
        print("=" * 60)

        cluster_prdc_results = {}

        for i in selected_clusters:
            gen_cluster_dir = os.path.join(gen_cluster_base_dir, f"cluster_{i}")
            real_cluster_dir = os.path.join(real_cluster_base_dir, f"cluster_{i}")

            if not os.path.exists(gen_cluster_dir) or not os.path.exists(real_cluster_dir):
                print(f"[SKIP] cluster_{i}: missing folder")
                continue

            print(f"\n[INFO] Processing cluster_{i}...")

            feat_real_c = extract_features_from_dir(
                real_cluster_dir, feat_model, device
            )
            feat_gen_c = extract_features_from_dir(
                gen_cluster_dir, feat_model, device, max_samples_cluster
            )

            if feat_real_c is None or feat_gen_c is None:
                print(f"[SKIP] cluster_{i}: empty images")
                continue

            prdc_c = compute_prdc(
                real_features=feat_real_c,
                fake_features=feat_gen_c,
                nearest_k=5
            )

            cluster_prdc_results[i] = prdc_c

            print(
                f"[CLUSTER {i:02d}] "
                f"P={prdc_c['precision']:.4f} | "
                f"R={prdc_c['recall']:.4f} | "
                f"D={prdc_c['density']:.4f} | "
                f"C={prdc_c['coverage']:.4f}"
            )


    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")

    finally:
        pass
        # [Step 5] Remove temporary files
        if os.path.exists(selected_dir):
            print(f"\n[INFO] Cleaning up temporary directory: {selected_dir}")
            shutil.rmtree(selected_dir)
            print("[INFO] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--real_dir", type=str, default='/mnt/HDD_raid1/lsun/church_outdoor_train/church') #a6000
    parser.add_argument("--real_dir", type=str, default="/home/elicer/dataset/church_outdoor_train/church/") #elice
    # parser.add_argument("--real_cluster_base_dir", type=str, default="/mnt/HDD_raid1/lsun/church_outdoor_train_gmm/30_diag/class_0/") #a6000
    parser.add_argument("--real_cluster_base_dir", type=str, default="/home/elicer/dataset/church_outdoor_train_gmm/30_diag/class_0/") #elice
    # parser.add_argument("--gen_base_dir", type=str, default="output/5th_lightningdit_xl_vavae_f16d32_gmm30_use_weight/lightningdit-xl-1-ckpt-0063000-euler-20/class_0/")
    parser.add_argument("--gen_base_dir", type=str, default="output/9th_lightningdit_xl_vavae_f16d32_gmm30_deterministic/lightningdit-xl-1-ckpt-0039440-euler-40/class_0/")
    args = parser.parse_args()

    main(args)