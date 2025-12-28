# score_and_visualize_posterior_gmm.py
#
# Output:
#  1) good_components.json
#  2) gmm_mu_grids.png
#  3) gmm_sigma_grids.png
#
# Requirements:
#  pip install safetensors torch numpy scikit-learn joblib tqdm pillow

import os, glob, json, math, argparse
import numpy as np
import torch
import joblib
from safetensors.torch import load_file
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from gmm_model import PosteriorDiagGMM


# -------------------------------------------------
# IO: stream mu (flattened)
# -------------------------------------------------

def iter_mu_flat(files, device="cpu"):
    for fp in files:
        obj = load_file(fp, device=device)
        mu = obj["mu"]
        if mu.ndim == 3:
            mu = mu.unsqueeze(0)
        mu = mu.to(torch.float32)
        B = mu.shape[0]
        yield mu.reshape(B, -1).cpu().numpy().astype(np.float32)


# -------------------------------------------------
# GMM scoring (posterior-aware)
# -------------------------------------------------

def score_components(model_dir, latents_dir, pattern, device,
                     min_count, min_mean_max_resp, max_mean_maha):

    stats = np.load(os.path.join(model_dir, "standardize_stats.npz"))
    mean, std = stats["mean"], stats["std"]

    ipca = joblib.load(os.path.join(model_dir, "ipca.joblib"))
    gmm  = joblib.load(os.path.join(model_dir, "posterior_gmm.joblib"))

    files = sorted(glob.glob(os.path.join(latents_dir, pattern)))
    if not files:
        raise RuntimeError("No latent files found")

    K, D = gmm.m.shape

    if max_mean_maha is None:
        max_mean_maha = float(D * 1.6)

    Nk = np.zeros(K)
    sum_resp = np.zeros(K)
    sum_maha = np.zeros(K)

    means = gmm.m
    inv_s2 = 1.0 / np.maximum(gmm.s2, 1e-12)

    for X in tqdm(iter_mu_flat(files, device), desc="Scoring"):
        Xn = (X - mean) / std
        Y  = ipca.transform(Xn).astype(np.float32)

        log_r = np.zeros((Y.shape[0], K), dtype=np.float32)
        for k in range(K):
            diff2 = (Y - means[k]) ** 2
            log_r[:, k] = (
                np.log(gmm.pi[k] + 1e-12)
                - 0.5 * np.sum(
                    diff2 * inv_s2[k] + np.log(gmm.s2[k]),
                    axis=1
                )
            )

        log_r -= log_r.max(axis=1, keepdims=True)
        r = np.exp(log_r)
        r /= r.sum(axis=1, keepdims=True)

        k_hat = r.argmax(axis=1)
        r_max = r[np.arange(len(k_hat)), k_hat]

        diff = Y - means[k_hat]
        maha = (diff * diff * inv_s2[k_hat]).sum(axis=1)

        Nk += np.bincount(k_hat, minlength=K)
        sum_resp += np.bincount(k_hat, weights=r_max, minlength=K)
        sum_maha += np.bincount(k_hat, weights=maha, minlength=K)

    mean_resp = np.where(Nk > 0, sum_resp / Nk, 0)
    mean_maha = np.where(Nk > 0, sum_maha / Nk, np.inf)

    good = [
        int(k) for k in range(K)
        if Nk[k] >= min_count
        and mean_resp[k] >= min_mean_max_resp
        and mean_maha[k] <= max_mean_maha
    ]

    good = sorted(
        good,
        key=lambda k: (mean_resp[k], -mean_maha[k], Nk[k]),
        reverse=True
    )

    report = {
        "K": int(K),
        "D_pca": int(D),
        "thresholds": {
            "min_count": min_count,
            "min_mean_max_resp": min_mean_max_resp,
            "max_mean_maha": max_mean_maha,
        },
        "good_components": good,
        "per_component": [
            {
                "k": int(k),
                "Nk": int(Nk[k]),
                "mean_max_resp": float(mean_resp[k]),
                "mean_maha": float(mean_maha[k]),
            } for k in range(K)
        ]
    }
    return report


# -------------------------------------------------
# Visualization utils
# -------------------------------------------------

def minmax_to_uint8(x, eps=1e-8):
    flat = x.reshape(x.shape[0], -1)
    mn = flat.min(axis=1)
    mx = flat.max(axis=1)
    x01 = (flat - mn[:, None]) / (mx[:, None] - mn[:, None] + eps)
    x01 = x01.reshape(x.shape)
    return (x01 * 255).astype(np.uint8), mn, mx


def make_grid(imgs, nrow=16, pad=2):
    N, H, W, C = imgs.shape
    ncol = int(math.ceil(N / nrow))
    grid = np.zeros(
        (ncol * (H + pad) - pad, nrow * (W + pad) - pad, C),
        dtype=np.uint8
    )
    idx = 0
    for r in range(ncol):
        for c in range(nrow):
            if idx >= N:
                break
            y = r * (H + pad)
            x = c * (W + pad)
            grid[y:y+H, x:x+W] = imgs[idx]
            idx += 1
    return grid


def add_legend(img, title, vmin, vmax):
    W, H = img.size
    canvas = Image.new("RGB", (W, H + 60), (255, 255, 255))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, H + 5), title, fill=(0, 0, 0))
    draw.text(
        (10, H + 30),
        f"min={vmin.min():.4f}   max={vmax.max():.4f}",
        fill=(0, 0, 0)
    )
    return canvas


def visualize_latent(lat, prefix, out_path):
    panels = []

    # RGB
    rgb_u8, mn, mx = minmax_to_uint8(lat)
    rgb_grid = make_grid(rgb_u8)
    panels.append(add_legend(Image.fromarray(rgb_grid), f"{prefix} RGB", mn, mx))

    # Channels
    for i, cname in enumerate(["R", "G", "B"]):
        ch = lat[..., i:i+1]
        ch_u8, mn, mx = minmax_to_uint8(ch)
        ch_u8 = np.repeat(ch_u8, 3, axis=-1)
        ch_grid = make_grid(ch_u8)
        panels.append(add_legend(
            Image.fromarray(ch_grid),
            f"{prefix} {cname}-channel",
            mn, mx
        ))

    W = max(p.width for p in panels)
    H = sum(p.height for p in panels)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))

    y = 0
    for p in panels:
        canvas.paste(p, (0, y))
        y += p.height

    canvas.save(out_path)
    print("Saved:", out_path)


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default='/home/ahlee/learnable_latent_DiT/learnable_eps/feature_output/model1_6th_f16d32_vfdinov2_add_layer/gmm2')
    ap.add_argument("--latents_dir", default='/home/ahlee/learnable_latent_DiT/learnable_eps/feature_output/model1_6th_f16d32_vfdinov2_add_layer/lsun_train_128')
    ap.add_argument("--pattern", default="*.safetensors")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min_count", type=int, default=400)
    ap.add_argument("--min_mean_max_resp", type=float, default=0.8)
    ap.add_argument("--max_mean_maha", type=float, default=None)
    args = ap.parse_args()

    report = score_components(
        args.model_dir,
        args.latents_dir,
        args.pattern,
        args.device,
        args.min_count,
        args.min_mean_max_resp,
        args.max_mean_maha
    )

    with open(os.path.join(args.model_dir, "good_components.json"), "w") as f:
        json.dump(report, f, indent=2)

    stats = np.load(os.path.join(args.model_dir, "standardize_stats.npz"))
    mean, std = stats["mean"], stats["std"]

    ipca = joblib.load(os.path.join(args.model_dir, "ipca.joblib"))
    gmm  = joblib.load(os.path.join(args.model_dir, "posterior_gmm.joblib"))

    mu_lat = ipca.inverse_transform(gmm.m) * std + mean
    sigma_lat = ipca.inverse_transform(np.sqrt(gmm.s2)) * std

    mu_lat = mu_lat.reshape(-1, 16, 16, 3)
    sigma_lat = sigma_lat.reshape(-1, 16, 16, 3)

    visualize_latent(
        mu_lat,
        "GMM Mean (μ)",
        os.path.join(args.model_dir, "gmm_mu_grids.png")
    )

    visualize_latent(
        sigma_lat,
        "GMM Std (σ)",
        os.path.join(args.model_dir, "gmm_sigma_grids.png")
    )


if __name__ == "__main__":
    main()
