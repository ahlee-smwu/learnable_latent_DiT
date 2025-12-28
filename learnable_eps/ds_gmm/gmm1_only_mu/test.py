import os
import math
import argparse
import numpy as np
import joblib
from PIL import Image, ImageDraw, ImageFont

# ---------- colormap: green(vmin) -> red(vmax) ----------
def green_to_red_colormap(x01: np.ndarray) -> np.ndarray:
    """
    x01: (H,W) float in [0,1]
    return: (H,W,3) uint8, 0->green, 1->red
    """
    x01 = np.clip(x01, 0.0, 1.0)
    r = (255.0 * x01).astype(np.uint8)
    g = (255.0 * (1.0 - x01)).astype(np.uint8)
    b = np.zeros_like(r, dtype=np.uint8)
    return np.stack([r, g, b], axis=-1)

def render_colorbar(height: int, vmin: float, vmax: float, width: int = 26) -> np.ndarray:
    """
    return: (height,width,3) uint8, top=red(vmax), bottom=green(vmin)
    """
    y = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None]
    bar = green_to_red_colormap(np.repeat(y, width, axis=1))
    return bar

def add_colorbar_and_labels(grid_rgb_u8: np.ndarray, vmin: float, vmax: float,
                            title: str = "", bar_width: int = 26, pad: int = 8) -> Image.Image:
    """
    grid_rgb_u8: (H,W,3) uint8
    returns PIL Image with a colorbar appended to the right.
    """
    H, W, _ = grid_rgb_u8.shape
    bar = render_colorbar(H, vmin=vmin, vmax=vmax, width=bar_width)

    canvas = np.zeros((H, W + pad + bar_width, 3), dtype=np.uint8)
    canvas[:, :W, :] = grid_rgb_u8
    canvas[:, W + pad:W + pad + bar_width, :] = bar

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()

    if title:
        draw.text((2, 2), title, fill=(255, 255, 255), font=font)

    draw.text((W + pad + 2, 2), f"{vmax:.3g}", fill=(255, 255, 255), font=font)
    draw.text((W + pad + 2, H - 12), f"{vmin:.3g}", fill=(255, 255, 255), font=font)

    return pil

# ---------- grid builder ----------
def make_grid_from_tiles_rgb(tiles_rgb_u8: np.ndarray, nrow: int, pad: int = 2, pad_value: int = 0) -> np.ndarray:
    """
    tiles_rgb_u8: (K,h,w,3) uint8
    returns grid (H,W,3)
    """
    K, h, w, _ = tiles_rgb_u8.shape
    ncol = int(math.ceil(K / nrow))

    H = ncol * h + (ncol - 1) * pad
    W = nrow * w + (nrow - 1) * pad
    grid = np.full((H, W, 3), pad_value, dtype=np.uint8)

    idx = 0
    for r in range(ncol):
        for c in range(nrow):
            if idx >= K:
                break
            y0 = r * (h + pad)
            x0 = c * (w + pad)
            grid[y0:y0 + h, x0:x0 + w, :] = tiles_rgb_u8[idx]
            idx += 1
    return grid

# ---------- vmin / vmax 자동 계산 ----------
def compute_vmin_vmax(x: np.ndarray, pmin: float, pmax: float):
    """
    x: arbitrary shape
    """
    vmin = np.percentile(x, pmin)
    vmax = np.percentile(x, pmax)
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)

def build_channel_grid_with_legend_colored(latents_khwc: np.ndarray,
                                           channel: int,
                                           vmin: float,
                                           vmax: float,
                                           nrow: int,
                                           pad: int,
                                           title: str) -> Image.Image:
    """
    latents_khwc: (K,16,16,3) float
    """
    x = latents_khwc[..., channel].astype(np.float32)
    x01 = (x - vmin) / (vmax - vmin + 1e-12)
    tiles_rgb = green_to_red_colormap(x01.reshape(-1, 16, 16)).reshape(-1, 16, 16, 3)
    grid = make_grid_from_tiles_rgb(tiles_rgb, nrow=nrow, pad=pad)
    return add_colorbar_and_labels(grid, vmin=vmin, vmax=vmax, title=title)

# ---------- GMM loading ----------
def load_mu_latents(model_dir: str):
    stats = np.load(os.path.join(model_dir, "standardize_stats.npz"))
    mean = stats["mean"].astype(np.float32)
    std = stats["std"].astype(np.float32)

    ipca = joblib.load(os.path.join(model_dir, "ipca.joblib"))
    gmm = joblib.load(os.path.join(model_dir, "gmm.joblib"))
    if gmm.covariance_type != "diag":
        raise ValueError("Expected diag GMM")

    mu_pca = gmm.means_.astype(np.float32)
    mu_std = ipca.inverse_transform(mu_pca).astype(np.float32)
    mu_orig = mu_std * std[None, :] + mean[None, :]
    mu_latents = mu_orig.reshape(-1, 16, 16, 3)
    return mu_latents, mean, std, ipca, gmm

def approx_sigma_latents(mean, std, ipca, gmm, n_samples=2000, seed=123):
    K, Dpca = gmm.means_.shape
    sigma_latents = np.empty((K, 16, 16, 3), dtype=np.float32)

    for k in range(K):
        rng = np.random.default_rng(seed + k)
        mu_p = gmm.means_[k].astype(np.float32)
        sig_p = np.sqrt(np.maximum(gmm.covariances_[k], 1e-12)).astype(np.float32)

        eps = rng.standard_normal((n_samples, Dpca)).astype(np.float32)
        Yp = mu_p[None] + eps * sig_p[None]

        Hn = ipca.inverse_transform(Yp).astype(np.float32)
        H = Hn * std[None] + mean[None]
        Z = H.reshape(n_samples, 16, 16, 3)

        sigma_latents[k] = Z.std(axis=0)

    return sigma_latents

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--nrow", type=int, default=16)
    ap.add_argument("--pad", type=int, default=2)
    ap.add_argument("--sigma_samples", type=int, default=2000)
    ap.add_argument("--sigma_seed", type=int, default=123)
    ap.add_argument("--pmin", type=float, default=1.0)
    ap.add_argument("--pmax", type=float, default=99.0)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.model_dir, "gmm_mu_sigma_channel_grids_colored_auto")
    os.makedirs(out_dir, exist_ok=True)

    mu_latents, mean, std, ipca, gmm = load_mu_latents(args.model_dir)
    sigma_latents = approx_sigma_latents(mean, std, ipca, gmm,
                                         n_samples=args.sigma_samples,
                                         seed=args.sigma_seed)

    # MU
    for ch in range(3):
        x = mu_latents[..., ch]
        vmin, vmax = compute_vmin_vmax(x, args.pmin, args.pmax)
        title = f"MU ch{ch} | p{args.pmin}-{args.pmax} [{vmin:.3g}, {vmax:.3g}]"
        img = build_channel_grid_with_legend_colored(
            mu_latents, ch, vmin, vmax, args.nrow, args.pad, title
        )
        img.save(os.path.join(out_dir, f"mu_ch{ch}_grid_colored.png"))

    # SIGMA
    for ch in range(3):
        x = sigma_latents[..., ch]
        vmin, vmax = compute_vmin_vmax(x, args.pmin, args.pmax)
        title = f"SIGMA ch{ch} | p{args.pmin}-{args.pmax} [{vmin:.3g}, {vmax:.3g}]"
        img = build_channel_grid_with_legend_colored(
            sigma_latents, ch, vmin, vmax, args.nrow, args.pad, title
        )
        img.save(os.path.join(out_dir, f"sigma_ch{ch}_grid_colored_ns{args.sigma_samples}.png"))

    print("Done. Output dir:", out_dir)

if __name__ == "__main__":
    main()

# python test.py \
#   --model_dir '/home/ahlee/learnable_latent_DiT/learnable_eps/feature_output/model1_6th_f16d32_vfdinov2_add_layer/' \
#   --pmin 1 \
#   --pmax 99 \
#   --nrow 16
