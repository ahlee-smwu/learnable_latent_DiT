# score_and_visualize_good_gmm_components.py
# pip install safetensors torch numpy scikit-learn joblib tqdm pillow

import os, glob, json, math, argparse
import numpy as np
import torch
import joblib
from safetensors.torch import load_file
from tqdm import tqdm
from PIL import Image, ImageDraw

# ---------- IO: stream mu ----------
def iter_mu_flat(files, device="cpu"):
    """
    yields: numpy float32 (B, 768)
    each file has mu: (bs,16,16,3) or (1,16,16,3)
    """
    for fp in files:
        obj = load_file(fp, device=device)
        mu = obj["mu"]
        if mu.ndim == 3:
            mu = mu.unsqueeze(0)
        if mu.ndim != 4:
            raise ValueError(f"Unexpected mu shape in {fp}: {tuple(mu.shape)}")
        mu = mu.to(torch.float32)
        B = mu.shape[0]
        yield mu.reshape(B, -1).cpu().numpy().astype(np.float32)

# ---------- scoring ----------
def score_components(model_dir, latents_dir, pattern, device,
                     min_count, min_mean_max_resp, max_mean_maha):
    stats = np.load(os.path.join(model_dir, "standardize_stats.npz"))
    mean = stats["mean"].astype(np.float32)  # (768,)
    std  = stats["std"].astype(np.float32)   # (768,)
    ipca = joblib.load(os.path.join(model_dir, "ipca.joblib"))
    gmm  = joblib.load(os.path.join(model_dir, "gmm.joblib"))

    if gmm.covariance_type != "diag":
        raise ValueError(f"This script currently expects diag GMM, got {gmm.covariance_type}")

    files = sorted(glob.glob(os.path.join(latents_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matched: {os.path.join(latents_dir, pattern)}")

    K, D = gmm.means_.shape

    if max_mean_maha is None:
        # heuristic: for chi-square(D), mean ~ D. set cutoff a bit higher.
        max_mean_maha = float(D * 1.6)

    Nk = np.zeros((K,), dtype=np.int64)
    sum_max_resp = np.zeros((K,), dtype=np.float64)
    sum_maha = np.zeros((K,), dtype=np.float64)

    means = gmm.means_.astype(np.float32)           # (K,D)
    covs  = gmm.covariances_.astype(np.float32)     # (K,D) diag variances
    invcovs = 1.0 / np.maximum(covs, 1e-12)

    for X in tqdm(iter_mu_flat(files, device=device), total=len(files), desc="Scoring", unit="file"):
        Xn = (X - mean[None, :]) / std[None, :]
        Y  = ipca.transform(Xn).astype(np.float32)  # (B,D)

        R = gmm.predict_proba(Y).astype(np.float32) # (B,K)
        k_hat = R.argmax(axis=1)
        r_max = R[np.arange(R.shape[0]), k_hat]

        diff = Y - means[k_hat]
        maha = (diff * diff * invcovs[k_hat]).sum(axis=1)

        # accumulate (vectorized via bincount)
        # counts
        cnt = np.bincount(k_hat, minlength=K)
        Nk += cnt

        # sum r_max per k
        sum_r = np.bincount(k_hat, weights=r_max, minlength=K)
        sum_max_resp += sum_r

        # sum maha per k
        sum_m = np.bincount(k_hat, weights=maha, minlength=K)
        sum_maha += sum_m

    mean_max_resp = np.where(Nk > 0, sum_max_resp / Nk, 0.0)
    mean_maha     = np.where(Nk > 0, sum_maha / Nk, np.inf)

    good = [
        int(k) for k in range(K)
        if (Nk[k] >= min_count) and (mean_max_resp[k] >= min_mean_max_resp) and (mean_maha[k] <= max_mean_maha)
    ]

    # sort by quality proxy: resp high, maha low, count high
    good_sorted = sorted(good, key=lambda k: (mean_max_resp[k], -mean_maha[k], Nk[k]), reverse=True)

    report = {
        "K": int(K),
        "D_pca": int(D),
        "thresholds": {
            "min_count": int(min_count),
            "min_mean_max_resp": float(min_mean_max_resp),
            "max_mean_maha": float(max_mean_maha),
        },
        "good_components": good_sorted,
        "per_component": [
            {
                "k": int(k),
                "Nk": int(Nk[k]),
                "mean_max_resp": float(mean_max_resp[k]),
                "mean_maha": float(mean_maha[k]),
            } for k in range(K)
        ],
    }
    return report

# ---------- visualization ----------
def minmax_to_uint8(x, per_image=True, eps=1e-8):
    x = x.astype(np.float32)
    if per_image:
        N = x.shape[0]
        flat = x.reshape(N, -1)
        mn = flat.min(axis=1, keepdims=True)
        mx = flat.max(axis=1, keepdims=True)
        scaled = (flat - mn) / (mx - mn + eps)
        return (scaled.reshape(x.shape) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        mn, mx = x.min(), x.max()
        return ((x - mn) / (mx - mn + eps) * 255.0).clip(0, 255).astype(np.uint8)

def latent_to_uint8_fixed(x):
    # x assumed roughly in [-1, 1]
    y = (x + 1.0) * 0.5 * 255.0
    return np.clip(y, 0, 255).astype(np.uint8)


def load_component_means_as_latents(model_dir):
    stats = np.load(os.path.join(model_dir, "standardize_stats.npz"))
    mean = stats["mean"].astype(np.float32)  # (768,)
    std  = stats["std"].astype(np.float32)   # (768,)

    ipca = joblib.load(os.path.join(model_dir, "ipca.joblib"))
    gmm  = joblib.load(os.path.join(model_dir, "gmm.joblib"))

    Yp = gmm.means_.astype(np.float32)              # (K, pca_dim)
    Hn = ipca.inverse_transform(Yp).astype(np.float32)  # (K, 768)
    H  = Hn * std[None, :] + mean[None, :]          # (K, 768)
    Z  = H.reshape(-1, 16, 16, 3).astype(np.float32)
    return Z, gmm

def make_grid(images_uint8, nrow, pad=2, pad_value=0):
    N, H, W, C = images_uint8.shape
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
    geom = {"H": H, "W": W, "pad": pad, "nrow": nrow}
    return grid, geom

def draw_boxes_on_grid(grid_uint8, geom, indices, color=(255, 0, 0), width=2):
    img = Image.fromarray(grid_uint8)
    draw = ImageDraw.Draw(img)
    H, W, pad, nrow = geom["H"], geom["W"], geom["pad"], geom["nrow"]
    idx_set = set(int(i) for i in indices)

    for idx in idx_set:
        r = idx // nrow
        c = idx % nrow
        x0 = c * (W + pad)
        y0 = r * (H + pad)
        x1 = x0 + W - 1
        y1 = y0 + H - 1
        for t in range(width):
            draw.rectangle([x0 - t, y0 - t, x1 + t, y1 + t], outline=color)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default='/home/ahlee/learnable_latent_DiT/learnable_eps/feature_output/model1_6th_f16d32_vfdinov2_add_layer')
    ap.add_argument("--latents_dir", type=str, default="/home/ahlee/learnable_latent_DiT/learnable_eps/feature_output/model1_6th_f16d32_vfdinov2_add_layer/lsun_train_128/")
    ap.add_argument("--pattern", type=str, default="latents_rank*_batch*.safetensors")
    ap.add_argument("--device", type=str, default="cpu")

    # selection thresholds
    ap.add_argument("--min_count", type=int, default=400)
    ap.add_argument("--min_mean_max_resp", type=float, default=0.8)
    ap.add_argument("--max_mean_maha", type=float, default=160)

    # visualization
    ap.add_argument("--per_image_minmax", action="store_true", default=True,
                    help="If set, scale each tile individually for visibility. Otherwise global scaling.")
    ap.add_argument("--nrow", type=int, default=16)
    ap.add_argument("--pad", type=int, default=2)
    ap.add_argument("--box_width", type=int, default=2)
    ap.add_argument("--box_r", type=int, default=255)
    ap.add_argument("--box_g", type=int, default=0)
    ap.add_argument("--box_b", type=int, default=0)

    ap.add_argument("--out_report", type=str, default="good_components.json")
    ap.add_argument("--out_png", type=str, default="gmm_cluster.png")
    args = ap.parse_args()

    # (1) score + select good components
    report = score_components(
        model_dir=args.model_dir,
        latents_dir=args.latents_dir,
        pattern=args.pattern,
        device=args.device,
        min_count=args.min_count,
        min_mean_max_resp=args.min_mean_max_resp,
        max_mean_maha=args.max_mean_maha
    )

    report_path = os.path.join(args.model_dir, args.out_report)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    good = report["good_components"]
    print(f"Saved report: {report_path}")
    print(f"Good components: {len(good)}/{report['K']}")

    # (2) visualize component means grid + mark good ones
    Z, gmm = load_component_means_as_latents(args.model_dir)  # (K,16,16,3)
    imgs = minmax_to_uint8(Z, per_image=args.per_image_minmax)  # cluster-wise min-max normalize
    # imgs = latent_to_uint8_fixed(Z, per_image=args.per_image_minmax)  # regular -1~1 -> 255 normalize

    grid, geom = make_grid(imgs, nrow=args.nrow, pad=args.pad, pad_value=0)
    marked = draw_boxes_on_grid(
        grid, geom, good,
        color=(args.box_r, args.box_g, args.box_b),
        width=args.box_width
    )

    out_png = os.path.join(args.model_dir, args.out_png)
    marked.save(out_png)
    print(f"Saved marked grid: {out_png}")

if __name__ == "__main__":
    main()
