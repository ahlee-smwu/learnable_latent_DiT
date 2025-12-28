import os
import math
import numpy as np
import joblib
from PIL import Image

def minmax_to_uint8(x, per_image=True, eps=1e-8):
    """
    x: (..., H, W, C) float
    per_image=True면 각 이미지별 min/max로 스케일(패턴 보기 좋음)
    per_image=False면 전체 텐서 글로벌 min/max로 스케일(비교 공정)
    """
    x = x.astype(np.float32)

    if per_image:
        # reshape to (N, -1) to compute per-image min/max
        N = x.shape[0]
        flat = x.reshape(N, -1)
        mn = flat.min(axis=1, keepdims=True)
        mx = flat.max(axis=1, keepdims=True)
        scaled = (flat - mn) / (mx - mn + eps)
        out = (scaled.reshape(x.shape) * 255.0).clip(0, 255).astype(np.uint8)
        return out
    else:
        mn = x.min()
        mx = x.max()
        out = ((x - mn) / (mx - mn + eps) * 255.0).clip(0, 255).astype(np.uint8)
        return out

def make_grid(images_uint8, nrow=None, pad=2, pad_value=0):
    """
    images_uint8: (N, H, W, C) uint8
    returns: (gridH, gridW, C) uint8
    """
    assert images_uint8.dtype == np.uint8
    N, H, W, C = images_uint8.shape
    if nrow is None:
        nrow = int(math.ceil(math.sqrt(N)))
    ncol = int(math.ceil(N / nrow))

    grid_h = ncol * H + (ncol - 1) * pad
    grid_w = nrow * W + (nrow - 1) * pad
    grid = np.full((grid_h, grid_w, C), pad_value, dtype=np.uint8)

    idx = 0
    for r in range(ncol):
        for c in range(nrow):
            if idx >= N:
                break
            y0 = r * (H + pad)
            x0 = c * (W + pad)
            grid[y0:y0+H, x0:x0+W, :] = images_uint8[idx]
            idx += 1
    return grid

def load_gmm_and_centers(model_dir):
    # stats
    stats = np.load(os.path.join(model_dir, "standardize_stats.npz"))
    mean = stats["mean"].astype(np.float32)  # (768,)
    std  = stats["std"].astype(np.float32)   # (768,)

    ipca = joblib.load(os.path.join(model_dir, "ipca.joblib"))
    gmm  = joblib.load(os.path.join(model_dir, "gmm.joblib"))

    # gmm.means_: (K, pca_dim)  -- PCA space component means
    Yp = gmm.means_.astype(np.float32)

    # inverse PCA -> standardized 768-d
    Hn = ipca.inverse_transform(Yp).astype(np.float32)  # (K, 768)

    # de-standardize -> original mu-space
    H = Hn * std[None, :] + mean[None, :]               # (K, 768)

    # reshape to (K, 16, 16, 3)
    Z = H.reshape(-1, 16, 16, 3).astype(np.float32)
    return Z, gmm

def save_gmm_representatives_grid(model_dir, out_png, per_image_minmax=True, pad=2):
    Z, gmm = load_gmm_and_centers(model_dir)
    K = Z.shape[0]
    print(f"Loaded {K} component means.")

    # (K,16,16,3) float -> uint8 for visualization
    imgs = minmax_to_uint8(Z, per_image=per_image_minmax)

    # grid 만들기 (보통 128이면 16x8 or 12x11 등 선택 가능)
    # 여기서는 16열로 고정하면 8행(128=16*8)
    nrow = 16 if K == 128 else None
    grid = make_grid(imgs, nrow=nrow, pad=pad, pad_value=0)

    Image.fromarray(grid).save(out_png)
    print("Saved:", out_png)

if __name__ == "__main__":
    model_dir = "/home/ahlee/learnable_latent_DiT/learnable_eps/feature_output/model1_6th_f16d32_vfdinov2_add_layer/"
    out_png = "/home/ahlee/learnable_latent_DiT/learnable_eps/feature_output/model1_6th_f16d32_vfdinov2_add_layer/gmm_component_means_grid.png"
    save_gmm_representatives_grid(model_dir, out_png, per_image_minmax=True, pad=2)
