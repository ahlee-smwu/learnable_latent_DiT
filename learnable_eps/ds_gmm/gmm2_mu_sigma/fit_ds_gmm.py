# fit_posterior_gmm_from_safetensors.py
#
# - safetensors 안의 (mu, sigma)를 사용
# - dataset-level standardization
# - IncrementalPCA
# - PCA 공간에서 posterior-aware diag GMM (custom EM)
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
import joblib


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def iter_mu_sigma_batches(files, batch_size_files=512, device="cpu"):
    """
    Yield (mu, sigma) as numpy arrays of shape (B, 768)
    """
    buf_mu, buf_sigma = [], []

    for fp in files:
        obj = load_file(fp, device=device)
        mu = obj["mu"]
        sigma = obj["sigma"]

        if mu.ndim == 3:
            mu = mu.unsqueeze(0)
            sigma = sigma.unsqueeze(0)
        elif mu.ndim != 4:
            raise ValueError(f"Bad shape in {fp}: mu {mu.shape}")

        mu = mu.to(torch.float32)
        sigma = sigma.to(torch.float32)

        bs = mu.shape[0]
        mu = mu.reshape(bs, -1).cpu().numpy()
        sigma = sigma.reshape(bs, -1).cpu().numpy()

        buf_mu.append(mu)
        buf_sigma.append(sigma)

        if len(buf_mu) >= batch_size_files:
            yield np.concatenate(buf_mu), np.concatenate(buf_sigma)
            buf_mu, buf_sigma = [], []

    if buf_mu:
        yield np.concatenate(buf_mu), np.concatenate(buf_sigma)


def welford_mean_std(files, batch_size_files=512):
    """
    Streaming mean/std over mu only
    """
    count = 0
    mean = None
    M2 = None

    for mu, _ in tqdm(
        iter_mu_sigma_batches(files, batch_size_files),
        desc="Pass1: mean/std",
        unit="batch"
    ):
        if mean is None:
            mean = np.zeros(mu.shape[1], dtype=np.float64)
            M2 = np.zeros(mu.shape[1], dtype=np.float64)

        for x in mu.astype(np.float64):
            count += 1
            delta = x - mean
            mean += delta / count
            M2 += delta * (x - mean)

    var = M2 / (count - 1)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


# -------------------------------------------------
# Posterior-aware Diag GMM
# -------------------------------------------------

class PosteriorDiagGMM:
    def __init__(self, n_components, dim, reg=1e-6, seed=0):
        self.K = n_components
        self.dim = dim
        self.reg = reg
        self.rng = np.random.RandomState(seed)

    def init_params(self, mu):
        idx = self.rng.choice(mu.shape[0], self.K, replace=False)
        self.m = mu[idx].copy()
        self.s2 = np.ones((self.K, self.dim), dtype=np.float32)
        self.pi = np.ones(self.K, dtype=np.float32) / self.K

    def e_step(self, mu, sigma2):
        N = mu.shape[0]
        log_r = np.zeros((N, self.K), dtype=np.float32)

        for k in range(self.K):
            diff2 = (mu - self.m[k]) ** 2
            log_r[:, k] = (
                np.log(self.pi[k] + 1e-12)
                - 0.5 * np.sum(
                    (diff2 + sigma2) / self.s2[k]
                    + np.log(self.s2[k]),
                    axis=1
                )
            )

        log_r -= log_r.max(axis=1, keepdims=True)
        r = np.exp(log_r)
        r /= r.sum(axis=1, keepdims=True)
        return r

    def m_step(self, mu, sigma2, r):
        Nk = r.sum(axis=0) + 1e-8
        self.pi = Nk / Nk.sum()

        self.m = (r.T @ mu) / Nk[:, None]

        for k in range(self.K):
            diff2 = (mu - self.m[k]) ** 2
            self.s2[k] = (
                (r[:, k][:, None] * (diff2 + sigma2)).sum(axis=0)
                / Nk[k]
            )

        self.s2 = np.maximum(self.s2, self.reg)

    def fit(self, mu, sigma2, iters=50):
        self.init_params(mu)
        for it in range(iters):
            r = self.e_step(mu, sigma2)
            self.m_step(mu, sigma2, r)
            print(f"[EM] iter {it+1}/{iters} done")


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents_dir", type=str, default="/home/ahlee/learnable_latent_DiT/learnable_eps/feature_output/model1_6th_f16d32_vfdinov2_add_layer/lsun_train_128/")
    ap.add_argument("--pattern", type=str, default="latents_rank*_batch*.safetensors")
    ap.add_argument("--out_dir", type=str, default="/home/ahlee/learnable_latent_DiT/learnable_eps/feature_output/model1_6th_f16d32_vfdinov2_add_layer/gmm2/")
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--gmm_components", type=int, default=128)
    ap.add_argument("--em_iters", type=int, default=50)
    ap.add_argument("--batch_files", type=int, default=512)
    ap.add_argument("--ipca_batch", type=int, default=4096)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.latents_dir, args.pattern)))
    if not files:
        raise RuntimeError("No safetensors found")
    print(f"Found {len(files)} files")

    # ---- Pass 1: mean/std (mu only)
    mean, std = welford_mean_std(files, args.batch_files)
    std = np.maximum(std, 1e-6)

    np.savez(os.path.join(args.out_dir, "standardize_stats.npz"),
             mean=mean, std=std)

    # ---- Pass 2: Incremental PCA (mu only)
    ipca = IncrementalPCA(n_components=args.pca_dim,
                          batch_size=args.ipca_batch)

    acc, acc_n = [], 0
    for mu, _ in tqdm(
        iter_mu_sigma_batches(files, args.batch_files, args.device),
        desc="Pass2: IPCA fit"
    ):
        mu_n = (mu - mean) / std
        acc.append(mu_n)
        acc_n += mu_n.shape[0]
        if acc_n >= args.ipca_batch:
            ipca.partial_fit(np.concatenate(acc))
            acc, acc_n = [], 0

    if acc:
        ipca.partial_fit(np.concatenate(acc))

    joblib.dump(ipca, os.path.join(args.out_dir, "ipca.joblib"))

    # ---- Pass 3: transform all mu, sigma
    W = ipca.components_.astype(np.float32)
    Hp_mu, Hp_s2 = [], []

    for mu, sigma in tqdm(
        iter_mu_sigma_batches(files, args.batch_files, args.device),
        desc="Pass3: PCA transform"
    ):
        mu_n = (mu - mean) / std
        sigma_n = sigma / std

        mu_p = mu_n @ W.T
        s2_p = (sigma_n ** 2) @ (W ** 2).T

        Hp_mu.append(mu_p.astype(np.float32))
        Hp_s2.append(s2_p.astype(np.float32))

    Hp_mu = np.concatenate(Hp_mu)
    Hp_s2 = np.concatenate(Hp_s2)

    # ---- Pass 4: posterior-aware GMM
    gmm = PosteriorDiagGMM(
        n_components=args.gmm_components,
        dim=args.pca_dim,
        seed=args.seed
    )
    gmm.fit(Hp_mu, Hp_s2, iters=args.em_iters)

    joblib.dump(gmm, os.path.join(args.out_dir, "posterior_gmm.joblib"))

    print("Done.")


if __name__ == "__main__":
    main()
