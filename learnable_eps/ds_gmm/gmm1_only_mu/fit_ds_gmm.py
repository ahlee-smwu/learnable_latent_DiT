# fit_gmm_from_safetensors.py
# - 특정 디렉토리 아래의 latents_rank??_batch???????.safetensors들을 읽어서
# - key 'mu' (및 선택적으로 'sigma', 'labels') 중 mu만 사용해
# - (dataset-level) 표준화 -> IncrementalPCA -> diag-GMM 을 학습하고
# - 학습된 (표준화 통계, PCA, GMM)을 디스크에 저장합니다.
#
# Requirements:
#   pip install safetensors torch numpy scikit-learn joblib tqdm

import os
import glob
import argparse
import numpy as np
import torch
from safetensors.torch import load_file
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from sklearn.mixture import GaussianMixture
import joblib


def iter_mu_batches(files, batch_size_files=1024, device="cpu"):
    """
    Yields numpy arrays of shape (B, 768) in float32.
    Each safetensors may contain mu shape (bs, 16, 16, 3) or (1,16,16,3).
    We concatenate across files up to batch_size_files files per yield.
    """
    buf = []
    for fp in files:
        obj = load_file(fp, device=device)  # returns dict of torch tensors
        mu = obj["mu"]  # torch.Tensor
        # expected: (bs,16,16,3)
        if mu.ndim == 3:
            mu = mu.unsqueeze(0)
        elif mu.ndim != 4:
            raise ValueError(f"Unexpected mu.ndim={mu.ndim} in {fp}, shape={tuple(mu.shape)}")

        mu = mu.to(torch.float32)
        bs = mu.shape[0]
        mu_flat = mu.reshape(bs, -1).cpu().numpy().astype(np.float32)  # (bs, 768)
        buf.append(mu_flat)

        if len(buf) >= batch_size_files:
            yield np.concatenate(buf, axis=0)
            buf = []

    if buf:
        yield np.concatenate(buf, axis=0)


def welford_mean_std(files, batch_size_files=1024):
    """
    Streaming feature-wise mean/std for 768-dim vectors.
    Returns (mean (768,), std (768,))
    """
    count = 0
    mean = None
    M2 = None  # sum of squares of differences from the current mean

    for X in tqdm(iter_mu_batches(files, batch_size_files=batch_size_files),
                  desc="Pass1: mean/std", unit="batch"):
        # X: (B, 768)
        if mean is None:
            mean = np.zeros((X.shape[1],), dtype=np.float64)
            M2 = np.zeros((X.shape[1],), dtype=np.float64)

        for x in X.astype(np.float64):
            count += 1
            delta = x - mean
            mean += delta / count
            delta2 = x - mean
            M2 += delta * delta2

    if count < 2:
        raise RuntimeError("Not enough samples to compute std.")

    var = M2 / (count - 1)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents_dir", type=str, required=True,
                    help="Directory containing latents_rank??_batch*.safetensors")
    ap.add_argument("--pattern", type=str, default="latents_rank*_batch*.safetensors")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--gmm_components", type=int, default=128)
    ap.add_argument("--gmm_cov", type=str, default="diag", choices=["diag", "tied", "full", "spherical"])
    ap.add_argument("--reg_covar", type=float, default=1e-6)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--batch_files", type=int, default=1024,
                    help="How many safetensors files to merge per streaming batch")
    ap.add_argument("--ipca_batch", type=int, default=4096,
                    help="IncrementalPCA internal batch size (in samples)")
    ap.add_argument("--random_state", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0 etc (load_file device)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.latents_dir, args.pattern)))
    if not files:
        raise FileNotFoundError(f"No files matched: {os.path.join(args.latents_dir, args.pattern)}")
    print(f"Found {len(files)} safetensors files.")

    # ---- Pass 1: dataset-level mean/std (feature-wise) ----
    mean, std = welford_mean_std(files, batch_size_files=args.batch_files)
    std = np.maximum(std, 1e-6).astype(np.float32)

    np.savez(os.path.join(args.out_dir, "standardize_stats.npz"), mean=mean, std=std)
    print("Saved standardize stats:", os.path.join(args.out_dir, "standardize_stats.npz"))

    # ---- Pass 2: fit IncrementalPCA on standardized data ----
    ipca = IncrementalPCA(n_components=args.pca_dim, batch_size=args.ipca_batch)

    # We stream and partial_fit; but must ensure chunks are not too small.
    # We'll accumulate to ~ipca_batch samples before calling partial_fit.
    acc = []
    acc_n = 0

    for X in tqdm(iter_mu_batches(files, batch_size_files=args.batch_files),
                  desc="Pass2: IPCA fit", unit="batch"):
        Xn = (X - mean) / std
        acc.append(Xn)
        acc_n += Xn.shape[0]
        if acc_n >= args.ipca_batch:
            Xcat = np.concatenate(acc, axis=0)
            ipca.partial_fit(Xcat)
            acc, acc_n = [], 0

    if acc_n > 0:
        Xcat = np.concatenate(acc, axis=0)
        ipca.partial_fit(Xcat)

    joblib.dump(ipca, os.path.join(args.out_dir, "ipca.joblib"))
    print("Saved IPCA:", os.path.join(args.out_dir, "ipca.joblib"))

    # ---- Pass 3: transform all data to PCA space and store in memory ----
    # For N~120k and pca_dim=128, float32 array is ~61MB (manageable).
    # First, count samples to allocate.
    total = 0
    for X in iter_mu_batches(files, batch_size_files=args.batch_files):
        total += X.shape[0]
    print(f"Total samples: {total}")

    Hp = np.empty((total, args.pca_dim), dtype=np.float32)
    idx = 0

    for X in tqdm(iter_mu_batches(files, batch_size_files=args.batch_files),
                  desc="Pass3: IPCA transform", unit="batch"):
        Xn = (X - mean) / std
        Y = ipca.transform(Xn).astype(np.float32)
        Hp[idx:idx + Y.shape[0]] = Y
        idx += Y.shape[0]

    assert idx == total

    # ---- Fit GMM (MoG) ----
    gmm = GaussianMixture(
        n_components=args.gmm_components,
        covariance_type=args.gmm_cov,
        reg_covar=args.reg_covar,
        init_params="kmeans",
        max_iter=args.max_iter,
        random_state=args.random_state,
        verbose=1,
    ).fit(Hp)

    joblib.dump(gmm, os.path.join(args.out_dir, "gmm.joblib"))
    print("Saved GMM:", os.path.join(args.out_dir, "gmm.joblib"))

    # Optional: save PCA-space samples for quick sanity checks
    np.save(os.path.join(args.out_dir, "Hp_pca_space.npy"), Hp[:min(5000, total)])
    print("Done.")


if __name__ == "__main__":
    main()
