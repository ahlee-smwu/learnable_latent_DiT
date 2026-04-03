import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.img_latent_dataset import ImgLatentDataset
import yaml

# ── 경로 수정 ──────────────────────────────────────────
CONFIG_PATH = "model2_xl_vavae_f16d32.yaml"   # ← 수정
GMM_PKL     = "GMM/result_proba_lsun/30_diag/gmm_clusters.pkl"  # ← 수정
# ───────────────────────────────────────────────────────

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# ── GMM 파라미터 로드 ────────────────────────────────────
with open(GMM_PKL, "rb") as f:
    ckpt = pickle.load(f)

# LSUN-Church는 class=0 하나
cls = list(ckpt["means"].keys())[0]
means   = ckpt["means"][cls]               # (30, 8192) data space
covs    = ckpt["covs"][cls]                # (30, 256)  PCA space diag var
weights = ckpt["weights"][cls]             # (30,)
U       = ckpt["pca_components"][cls]["components"]  # (256, 8192)
mu_pca  = ckpt["pca_components"][cls]["mean"]        # (8192,)
means_pca = (means - mu_pca) @ U.T        # (30, 256)

# ── 실제 latent 데이터 로드 ──────────────────────────────
dataset = ImgLatentDataset(
    data_dir=cfg['data']['data_path'],
    latent_norm=cfg['data'].get('latent_norm', True),
    latent_multiplier=cfg['data'].get('latent_multiplier', 1.0),
)
loader = DataLoader(dataset, batch_size=256, shuffle=False,
                    num_workers=4, pin_memory=True, drop_last=False)

print(f"Dataset size: {len(dataset)}")

features = []
for latents, _ in tqdm(loader, desc="Loading latents"):
    features.append(latents.flatten(start_dim=1).cpu().numpy())
features = np.concatenate(features, axis=0)   # (N, 8192)
print(f"Latents shape: {features.shape}")

# ── Posterior r_k(x1) 계산 ──────────────────────────────
def compute_posterior(x_flat, means_pca, covs, weights, U, mu_pca, eps=1e-8):
    """
    x_flat:    (N, 8192)
    returns r: (N, K)
    """
    x_pca  = (x_flat - mu_pca) @ U.T       # (N, 256)
    v_safe = np.clip(covs, eps, None)       # (K, 256)

    diff  = x_pca[:, None, :] - means_pca[None, :, :]   # (N, K, 256)
    mahal = (diff**2 / v_safe[None]).sum(-1)             # (N, K)
    log_det = np.log(v_safe).sum(-1)                     # (K,)

    log_p = -0.5*(mahal + log_det[None,:]) + np.log(weights[None,:]+eps)
    log_p -= log_p.max(axis=1, keepdims=True)            # numerical stability
    r = np.exp(log_p)
    r /= r.sum(axis=1, keepdims=True)
    return r                                              # (N, K)

print("\nComputing posteriors...")
# 메모리 절약: 배치로 계산
BATCH = 1000
r_list = []
for i in tqdm(range(0, len(features), BATCH)):
    r_list.append(compute_posterior(
        features[i:i+BATCH], means_pca, covs, weights, U, mu_pca
    ))
r = np.concatenate(r_list, axis=0)   # (N, 30)
max_r = r.max(axis=1)                # (N,)

# ── 분석 1: max posterior 분포 ───────────────────────────
print("\n" + "="*55)
print("분석 1: 실제 데이터 포인트의 max posterior 분포")
print("="*55)
print(f"  N = {len(max_r):,}")
print(f"  mean:              {max_r.mean():.4f}")
print(f"  median:            {np.median(max_r):.4f}")
print(f"  min:               {max_r.min():.4f}")
print(f"  max:               {max_r.max():.4f}")
print(f"  std:               {max_r.std():.4f}")
print(f"  < 0.9:  {(max_r<0.9).mean()*100:.1f}%")
print(f"  < 0.6 (경계):  {(max_r<0.6).mean()*100:.1f}%")
print(f"  < 0.4:         {(max_r<0.4).mean()*100:.1f}%")
print(f"  < 0.1:         {(max_r<0.1).mean()*100:.1f}%")

# 히스토그램
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
hist, _ = np.histogram(max_r, bins=bins)
print(f"\n  max_r 히스토그램:")
for i in range(len(hist)):
    bar = "█" * (hist[i] * 40 // max(hist))
    print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}]: {bar} {hist[i]:,} ({hist[i]/len(max_r)*100:.1f}%)")

# ── 분석 2: Soft vs Hard x0 차이 ────────────────────────
print("\n" + "="*55)
print("분석 2: Soft vs Hard x0 차이 (실제 데이터 기준)")
print("="*55)
hard_idx  = r.argmax(axis=1)           # (N,)
x0_hard   = means[hard_idx]            # (N, 8192)
x0_soft   = r @ means                  # (N, 8192)
diff_norm = np.linalg.norm(x0_soft - x0_hard, axis=1)   # (N,)

print(f"  ||x0_soft - x0_hard||:")
print(f"    mean:   {diff_norm.mean():.4f}")
print(f"    median: {np.median(diff_norm):.4f}")
print(f"    max:    {diff_norm.max():.4f}")
print(f"    std:    {diff_norm.std():.4f}")
print(f"\n  클러스터 간 평균 거리 대비: {diff_norm.mean()/32.19*100:.1f}%")
print(f"  노이즈(√8192=90.5) 대비:  {diff_norm.mean()/90.51*100:.1f}%")

# 경계 포인트만 따로 분석
boundary_mask = max_r < 0.6
if boundary_mask.sum() > 0:
    print(f"\n  경계 포인트(max_r<0.6) {boundary_mask.sum():,}개의 차이:")
    print(f"    mean: {diff_norm[boundary_mask].mean():.4f}")
    print(f"    max:  {diff_norm[boundary_mask].max():.4f}")

# ── 분석 3: 클러스터별 데이터 수 vs GMM weight ──────────
print("\n" + "="*55)
print("분석 3: 실제 클러스터 점유율 vs GMM weight")
print("(차이 클수록 marginal mismatch)")
print("="*55)
empirical = np.bincount(hard_idx, minlength=30) / len(hard_idx)
print(f"  클러스터  GMM_weight  실제비율  차이")
for k in range(30):
    diff_w = abs(weights[k] - empirical[k])
    flag = " ←" if diff_w > 0.01 else ""
    print(f"  k={k:02d}:  {weights[k]:.4f}      {empirical[k]:.4f}    {diff_w:.4f}{flag}")
print(f"\n  평균 |π_k - empirical_k|: {np.abs(weights - empirical).mean():.4f}")

# ── 최종 판정 ────────────────────────────────────────────
print("\n" + "="*55)
print("최종 판정")
print("="*55)
boundary_ratio = (max_r < 0.6).mean()
noise_ratio    = diff_norm.mean() / 90.51
weight_mismatch = np.abs(weights - empirical).mean()

print(f"  경계 포인트 비율:        {boundary_ratio*100:.1f}%")
print(f"  soft/hard 차이/노이즈:  {noise_ratio*100:.1f}%")
print(f"  marginal mismatch:      {weight_mismatch:.4f}")

if noise_ratio < 0.05:
    verdict = "Soft assignment 이득 없음 (차이 < 5% of noise)"
elif noise_ratio < 0.15:
    verdict = "Soft assignment 이득 제한적 (5~15% of noise)"
else:
    verdict = "Soft assignment 이득 유의미 (> 15% of noise)"
print(f"\n  → {verdict}")