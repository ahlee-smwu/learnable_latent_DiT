import os
import json
import joblib
import numpy as np
from PIL import Image, ImageDraw
from gmm_model import PosteriorDiagGMM

# =================================================
# CONFIG
# =================================================

OUT_DIR = "/home/ahlee/learnable_latent_DiT/learnable_eps/feature_output/model1_6th_f16d32_vfdinov2_add_layer/gmm2/sampling"

GMM_PATH = "/home/ahlee/learnable_latent_DiT/learnable_eps/feature_output/model1_6th_f16d32_vfdinov2_add_layer/gmm2/posterior_gmm.joblib"
IPCA_PATH = "/home/ahlee/learnable_latent_DiT/learnable_eps/feature_output/model1_6th_f16d32_vfdinov2_add_layer/gmm2/ipca.joblib"
STAT_PATH = "/home/ahlee/learnable_latent_DiT/learnable_eps/feature_output/model1_6th_f16d32_vfdinov2_add_layer/gmm2/standardize_stats.npz"
GOOD_JSON_PATH = "/home/ahlee/learnable_latent_DiT/learnable_eps/feature_output/model1_6th_f16d32_vfdinov2_add_layer/gmm2/good_components.json"

N_GRIDS = 10
GRID_H, GRID_W = 8, 16          # 8 x 16 = 128
IMG_SHAPE = (3, 16, 16)        # ⚠️ latent reshape (필요 시 수정)
BORDER_WIDTH = 3               # good cluster 테두리 두께

SEED = 0

# =================================================
# SETUP
# =================================================

os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.RandomState(SEED)

# =================================================
# LOAD MODELS / STATS
# =================================================

print("[Load] GMM")
gmm = joblib.load(GMM_PATH)

print("[Load] IPCA")
ipca = joblib.load(IPCA_PATH)

print("[Load] Standardization stats")
stats = np.load(STAT_PATH)
mean = stats["mean"]
std = stats["std"]

print("[Load] Good components JSON")
with open(GOOD_JSON_PATH, "r") as f:
    good_info = json.load(f)

good_clusters = set(good_info["good_components"])

K = gmm.K
D = gmm.dim
assert K == GRID_H * GRID_W, f"K={K} != {GRID_H*GRID_W}"

# =================================================
# UTILITIES
# =================================================

def latent_to_image(x: np.ndarray) -> Image.Image:
    """
    x: (D,)
    return: PIL.Image
    """
    img = x.reshape(IMG_SHAPE)

    # min-max normalize for visualization
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)

    if IMG_SHAPE[0] == 1:
        return Image.fromarray(img[0], mode="L")
    else:
        img = np.transpose(img, (1, 2, 0))
        return Image.fromarray(img, mode="RGB")


def draw_border(img: Image.Image, color=(255, 0, 0), width=3):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for i in range(width):
        draw.rectangle(
            [i, i, w - i - 1, h - i - 1],
            outline=color
        )
    return img


# =================================================
# MAIN SAMPLING
# =================================================

for grid_idx in range(N_GRIDS):
    print(f"[Grid] {grid_idx + 1}/{N_GRIDS}")

    tiles = []

    for k in range(K):
        # ---- sample from k-th Gaussian (PCA space)
        z_pca = rng.randn(D) * np.sqrt(gmm.s2[k]) + gmm.m[k]

        z_norm = ipca.inverse_transform(z_pca)

        # ---- de-standardize
        z = z_norm * std + mean

        # ---- to image
        img = latent_to_image(z)

        # ---- mark good cluster
        if k in good_clusters:
            img = draw_border(img, color=(255, 0, 0), width=BORDER_WIDTH)

        tiles.append(img)

    # ---- build grid canvas
    tile_w, tile_h = tiles[0].size
    canvas = Image.new(
        "RGB",
        (GRID_W * tile_w, GRID_H * tile_h),
        color=(0, 0, 0)
    )

    for idx, img in enumerate(tiles):
        r = idx // GRID_W
        c = idx % GRID_W
        canvas.paste(img, (c * tile_w, r * tile_h))

    out_path = os.path.join(OUT_DIR, f"grid_{grid_idx:02d}.png")
    canvas.save(out_path)
    print(f"  -> saved {out_path}")

print("✅ All grids saved.")
