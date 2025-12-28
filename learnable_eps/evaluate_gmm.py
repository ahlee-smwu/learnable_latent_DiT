import os
import sys

# 절대경로로 vavae/DiT 폴더를 sys.path에 추가
vavae_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vavae'))
DiT_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(vavae_path)
sys.path.append(DiT_path)

import argparse
import json
import math
import numpy as np
import joblib
import torch
import torch.distributed as dist
from PIL import Image
from omegaconf import OmegaConf
from accelerate import Accelerator
from ldm.util import instantiate_from_config

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def sample_one_per_good_component(model_dir: str,
                                  good_json: str = "good_components.json",
                                  trunc: float | None = 2.5,
                                  seed: int | None = None) -> np.ndarray:
    """
    Returns:
        Z: (G, 16, 16, 3) float32, one sample per good component
    """
    rng = np.random.default_rng(seed)

    stats = np.load(os.path.join(model_dir, "standardize_stats.npz"))
    mean = stats["mean"].astype(np.float32)  # (768,)
    std = stats["std"].astype(np.float32)    # (768,)

    ipca = joblib.load(os.path.join(model_dir, "ipca.joblib"))
    gmm = joblib.load(os.path.join(model_dir, "gmm.joblib"))
    if gmm.covariance_type != "diag":
        raise ValueError(f"Expected diag GMM, got {gmm.covariance_type}")

    with open(os.path.join(model_dir, good_json), "r", encoding="utf-8") as f:
        rep = json.load(f)
    good = np.array(rep["good_components"], dtype=np.int64)
    if good.size == 0:
        raise ValueError("good_components is empty.")

    G = int(good.size)
    D = int(gmm.means_.shape[1])

    mu = gmm.means_[good].astype(np.float32)  # (G, D) in PCA space
    std_k = np.sqrt(np.maximum(gmm.covariances_[good], 1e-12)).astype(np.float32)  # (G, D)

    eps = rng.standard_normal(size=(G, D)).astype(np.float32)
    if trunc is not None:
        eps = np.clip(eps, -trunc, trunc)

    Yp = mu + std_k * eps  # (G, D)

    Hn = ipca.inverse_transform(Yp).astype(np.float32)  # (G, 768) standardized
    H = Hn * std[None, :] + mean[None, :]               # (G, 768) mu-space
    Z = H.reshape(G, 16, 16, 3).astype(np.float32)      # (G,16,16,3)
    return Z


def save_uint8_grid_bhwc(images_u8: np.ndarray,
                         out_path: str,
                         nrow: int | None = None,
                         pad: int = 2,
                         pad_value: int = 0):
    """
    images_u8: (G, H, W, 3) uint8
    out_path: must include extension like .png
    """
    assert images_u8.dtype == np.uint8, images_u8.dtype
    assert images_u8.ndim == 4 and images_u8.shape[-1] == 3, images_u8.shape

    _, ext = os.path.splitext(out_path)
    if ext.lower() not in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        raise ValueError(f"out_path must be a file path with image extension, got: {out_path}")

    G, H, W, _ = images_u8.shape
    if nrow is None:
        nrow = int(math.ceil(math.sqrt(G)))
    ncol = int(math.ceil(G / nrow))

    grid_h = ncol * H + (ncol - 1) * pad
    grid_w = nrow * W + (nrow - 1) * pad
    grid = np.full((grid_h, grid_w, 3), pad_value, dtype=np.uint8)

    idx = 0
    for r in range(ncol):
        for c in range(nrow):
            if idx >= G:
                break
            y0 = r * (H + pad)
            x0 = c * (W + pad)
            grid[y0:y0 + H, x0:x0 + W, :] = images_u8[idx]
            idx += 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    Image.fromarray(grid).save(out_path)
    return out_path


def ddp_setup(seed_base: int):
    """
    Try init DDP; fallback to local mode.
    Returns: rank, device, world_size, seed
    """
    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        world_size = dist.get_world_size()
        seed = seed_base + rank
        if rank == 0:
            print(f"Starting DDP rank={rank}, seed={seed}, world_size={world_size}.")
    except Exception:
        print("Failed to initialize DDP. Running in local mode.")
        rank = 0
        device = 0
        world_size = 1
        seed = seed_base

    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    return rank, device, world_size, seed


def main(args):
    assert torch.cuda.is_available(), "This script requires at least one GPU."

    rank, device, world_size, seed = ddp_setup(args.seed)

    output_dir = os.path.join(
        args.output_path
    )
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    accelerator = Accelerator()

    # ---- load model ----
    model1_config = OmegaConf.load(args.config)
    model1 = instantiate_from_config(model1_config.model)
    model1.to(device)

    # ---- patch layers (your original) ----
    model1.new_proj_vae = torch.nn.Conv2d(32 * 2, 3 * 2, kernel_size=1, bias=True)
    model1.new_proj_align1 = torch.nn.Conv2d(3, 64, kernel_size=1, bias=False)
    model1.new_proj_align2 = torch.nn.Conv2d(64, 1024, kernel_size=1, bias=False)
    model1.new_proj_align = torch.nn.Sequential(model1.new_proj_align1, model1.new_proj_align2)

    def new_forward(self, input, sample_posterior=True):
        from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
        h = self.encoder(input)
        moments = self.quant_conv(h)
        moments = self.new_proj_vae(moments)
        posterior = DiagonalGaussianDistribution(moments)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        dec = self.decode_eps(z)

        if self.use_vf is not None:
            aux_feature = self.foundation_model(input)
            if not self.reverse_proj:
                aux_feature = self.new_proj_align(aux_feature)
            else:
                z = self.new_proj_align(z)
            return dec, posterior, z, aux_feature

    import types
    model1.forward = types.MethodType(new_forward, model1)

    # ---- load checkpoint (filter linear_proj mismatch) ----
    try:
        ckpt = torch.load(model1_config.init_weight, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        bad_keys = []
        for k in list(state_dict.keys()):
            if k.endswith("linear_proj.weight") or k.endswith("linear_proj.bias"):
                bad_keys.append(k)
                del state_dict[k]

        missing, unexpected = model1.load_state_dict(state_dict, strict=False)

        if rank == 0:
            print(f"Loaded ckpt (filtered). Removed keys: {bad_keys}")
            print(f"Missing keys (first 20): {missing[:20]}")
            print(f"Unexpected keys (first 20): {unexpected[:20]}")
    except Exception:
        print("There is no initial weights to load or loading failed.")
        import traceback
        traceback.print_exc()

    model1.eval()

    # ---- output folder for grids ----
    if rank == 0:
        vis_dir = os.path.join(output_dir, "gmm_decode_vis")
        os.makedirs(vis_dir, exist_ok=True)

    # ---- main loop: iterations ----
    for it in range(args.iterations):
        # 1) sample Z from good components on rank0 and broadcast
        if rank == 0:
            Z_np = sample_one_per_good_component(
                model_dir=args.gmm_dir,
                good_json=args.good_json,
                trunc=args.trunc,
                seed=(args.seed + it) if args.use_deterministic_seed else None
            )  # (G,16,16,3)
            if Z_np.shape[0] != args.num_good:
                raise ValueError(f"JSON good_components has {Z_np.shape[0]} items, but --num_good={args.num_good}")
            Z_t = torch.from_numpy(Z_np)  # cpu float32
        else:
            Z_t = torch.empty((args.num_good, 16, 16, 3), dtype=torch.float32)

        if world_size > 1:
            dist.broadcast(Z_t, src=0)

        # 2) to device and decode
        z = Z_t.permute(0, 3, 1, 2).contiguous().to(device)  # (G,3,16,16)

        with torch.no_grad():
            decoded = model1.decode_eps(z)  # expected (G,3,128,128)

        # 3) to uint8 BHWC
        images = torch.clamp(127.5 * decoded + 128.0, 0, 255) \
            .permute(0, 2, 3, 1) \
            .to("cpu", dtype=torch.uint8) \
            .numpy()  # (G,H,W,3)

        # 4) save grid on rank0
        if rank == 0:
            out_png = os.path.join(vis_dir, f"gmm_good_samples_grid_iter{it:04d}.png")
            save_uint8_grid_bhwc(images, out_png, nrow=args.nrow, pad=args.pad)
            print(f"[rank0] saved: {out_png}")

    # ---- cleanup ----
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--data_split", type=str, default="lsun_train")
    parser.add_argument("--output_path", type=str, default="/home/ahlee/learnable_latent_DiT/learnable_eps/feature_output/model1_6th_f16d32_vfdinov2_add_layer")
    parser.add_argument("--config", type=str, default="model1_f16d32_vfdinov2_add_layer.yaml")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    # GMM paths
    parser.add_argument("--gmm_dir", type=str, default='/home/ahlee/learnable_latent_DiT/learnable_eps/feature_output/model1_6th_f16d32_vfdinov2_add_layer',
                        help="Directory containing gmm.joblib, ipca.joblib, standardize_stats.npz, good_components.json")
    parser.add_argument("--good_json", type=str, default="good_components.json")
    parser.add_argument("--num_good", type=int, default=21, help="Must match len(good_components).")
    parser.add_argument("--trunc", type=float, default=2.5, help="Truncation clip for GMM sampling. Use -1 for None.")
    parser.add_argument("--iterations", type=int, default=10, help="How many grid images to save.")
    # saving grid layout
    parser.add_argument("--nrow", type=int, default=7, help="Grid columns. For G=21, 7 makes 7x3.")
    parser.add_argument("--pad", type=int, default=2)
    # randomness control
    parser.add_argument("--use_deterministic_seed", action="store_true",
                        help="If set, use seed+iter for reproducible sampling. If not set, sampling is random each run.")

    args = parser.parse_args()
    if args.trunc < 0:
        args.trunc = None

    main(args)
