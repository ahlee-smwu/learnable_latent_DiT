# assign_cluster_and_save_safetensors.py
#
# - safetensors (mu, sigma)를 입력으로
# - posterior-aware GMM으로 클러스터 할당
# - safetensors 하나당 결과 safetensors 하나 저장
#
# Output safetensors:
#   - cluster_id : (bs,) int64
#   - confidence : (bs,) float32
#
# Requirements:
#   pip install safetensors torch numpy scikit-learn joblib tqdm

import os
import glob
import argparse
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import joblib
from gmm_model import PosteriorDiagGMM

# -------------------------------------------------
# Posterior responsibility
# -------------------------------------------------

def posterior_log_resp(mu, sigma2, gmm):
    """
    mu, sigma2: (B, D)
    return: log responsibilities (B, K)
    """
    B = mu.shape[0]
    K = gmm.K
    log_r = np.zeros((B, K), dtype=np.float32)

    for k in range(K):
        diff2 = (mu - gmm.m[k]) ** 2
        log_r[:, k] = (
            np.log(gmm.pi[k] + 1e-12)
            - 0.5 * np.sum(
                (diff2 + sigma2) / gmm.s2[k]
                + np.log(gmm.s2[k]),
                axis=1
            )
        )
    return log_r


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default='../../feature_output/model1_6th_f16d32_vfdinov2_add_layer/lsun_train_128/')
    ap.add_argument("--pattern", type=str, default="*.safetensors")
    ap.add_argument("--model_dir", type=str, default='../../feature_output/model1_6th_f16d32_vfdinov2_add_layer/gmm2/',
                    help="standardize_stats.npz, ipca.joblib, posterior_gmm.joblib")
    ap.add_argument("--out_dir", type=str, default='../../feature_output/model1_6th_f16d32_vfdinov2_add_layer/gmm2/assign_cluster/')
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load trained artifacts
    stats = np.load(os.path.join(args.model_dir, "standardize_stats.npz"))
    mean = stats["mean"]
    std = stats["std"]

    ipca = joblib.load(os.path.join(args.model_dir, "ipca.joblib"))
    gmm = joblib.load(os.path.join(args.model_dir, "posterior_gmm.joblib"))

    W = ipca.components_.astype(np.float32)

    files = sorted(glob.glob(os.path.join(args.data_dir, args.pattern)))
    if not files:
        raise RuntimeError("No safetensors found")

    print(f"Found {len(files)} files")

    # ---- Process each safetensors independently
    for fp in tqdm(files, desc="Assign clusters"):
        obj = load_file(fp, device=args.device)

        mu = obj["mu"]
        sigma = obj["sigma"]

        if mu.ndim == 3:
            mu = mu.unsqueeze(0)
            sigma = sigma.unsqueeze(0)

        mu = mu.to(torch.float32)
        sigma = sigma.to(torch.float32)

        bs = mu.shape[0]
        mu = mu.reshape(bs, -1).cpu().numpy()
        sigma = sigma.reshape(bs, -1).cpu().numpy()

        # ---- Standardize
        mu_n = (mu - mean) / std
        sigma_n = sigma / std

        # ---- PCA transform
        mu_p = mu_n @ W.T
        sigma2_p = (sigma_n ** 2) @ (W ** 2).T

        # ---- Posterior responsibilities
        log_r = posterior_log_resp(mu_p, sigma2_p, gmm)
        r = np.exp(log_r - log_r.max(axis=1, keepdims=True))
        r /= r.sum(axis=1, keepdims=True)

        cluster_id = np.argmax(r, axis=1).astype(np.int64)
        confidence = np.max(r, axis=1).astype(np.float32)

        # ---- Save as safetensors
        out_fp = os.path.join(
            args.out_dir,
            os.path.basename(fp).replace(".safetensors", "_cluster.safetensors")
        )

        save_file(
            {
                "cluster_id": torch.from_numpy(cluster_id),
                "confidence": torch.from_numpy(confidence),
            },
            out_fp
        )

    print("Done.")


if __name__ == "__main__":
    main()
