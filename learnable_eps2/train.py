"""
Training Codes of LightningDiT together with VA-VAE.
It envolves advanced training methods, sampling methods, 
architecture design methods, computation methods. We achieve
state-of-the-art FID 1.35 on ImageNet 256x256.

by Maple (Jingfeng Yao) from HUST-VL
"""
# # fix import
# ============================================================
# FIX: datasets name collision (HF datasets vs local datasets)
# - CODE ONLY
# - accelerate / DDP safe
# ============================================================

import sys
import site
import importlib.util
from pathlib import Path

# ------------------------------------------------------------
# 1) huggingface datasets를 site-packages에서 "패키지 import"
# ------------------------------------------------------------
_site_packages = site.getsitepackages()
_original_sys_path = sys.path.copy()

# site-packages를 sys.path 최우선으로
for sp in reversed(_site_packages):
    if sp in sys.path:
        sys.path.remove(sp)
    sys.path.insert(0, sp)

import datasets as hf_datasets  # ✅ 진짜 HF 패키지 import

# sys.modules에 datasets 고정
sys.modules["datasets"] = hf_datasets

# sys.path 복구
sys.path = _original_sys_path

# ------------------------------------------------------------
# 2) 로컬 ImgLatentDataset 직접 로드
# ------------------------------------------------------------
_local_ds_path = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "img_latent_dataset.py"
)

spec = importlib.util.spec_from_file_location(
    "local_img_latent_dataset",
    _local_ds_path
)
_local_ds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_local_ds)

ImgLatentDataset = _local_ds.ImgLatentDataset

# ============================================================
# 이후부터는 일반 import
# ============================================================
import torch
import torch.distributed as dist
import torch.backends.cuda
import torch.backends.cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import math
import yaml
import json
import numpy as np
import logging
import os
import argparse
from time import time
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
import pickle

import torch.nn.functional as F

from diffusers.models import AutoencoderKL
from accelerate import Accelerator

from models.lightningdit import LightningDiT_models
from transport import create_transport, Sampler


# # for common import
# import torch
# import torch.distributed as dist
# import torch.backends.cuda
# import torch.backends.cudnn
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
#
# import math
# import yaml
# import json
# import numpy as np
# import logging
# import os
# import argparse
# from time import time
# from glob import glob
# from copy import deepcopy
# from collections import OrderedDict
# from PIL import Image
# from tqdm import tqdm
#
# from diffusers.models import AutoencoderKL
# from models.lightningdit import LightningDiT_models
# from transport import create_transport, Sampler
# from accelerate import Accelerator
# from datasets.img_latent_dataset import ImgLatentDataset
# import pickle
# import torch.nn.functional as F

def do_train(train_config, accelerator):
    """
    Trains a LightningDiT.
    """
    # Setup accelerator:
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(train_config['train']['output_dir'], exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{train_config['train']['output_dir']}/*"))
        model_string_name = train_config['model']['model_type'].replace("/", "-")
        if train_config['train']['exp_name'] is None:
            exp_name = f'{experiment_index:03d}-{model_string_name}'
        else:
            exp_name = train_config['train']['exp_name']
        experiment_dir = f"{train_config['train']['output_dir']}/{exp_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        tensorboard_dir_log = f"tensorboard_logs/{exp_name}"
        os.makedirs(tensorboard_dir_log, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir_log)

        # add configs to tensorboard
        config_str=json.dumps(train_config, indent=4)
        writer.add_text('training configs', config_str, global_step=0)
    checkpoint_dir = f"{train_config['train']['output_dir']}/{train_config['train']['exp_name']}/checkpoints"

    # get rank
    rank = accelerator.local_process_index

    # Create model:
    if 'downsample_ratio' in train_config['vae']:
        downsample_ratio = train_config['vae']['downsample_ratio']
    else:
        downsample_ratio = 16
    assert train_config['data']['image_size'] % downsample_ratio == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = train_config['data']['image_size'] // downsample_ratio
    model = LightningDiT_models[train_config['model']['model_type']](
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model']['use_swiglu'] if 'use_swiglu' in train_config['model'] else False,
        use_rope=train_config['model']['use_rope'] if 'use_rope' in train_config['model'] else False,
        use_rmsnorm=train_config['model']['use_rmsnorm'] if 'use_rmsnorm' in train_config['model'] else False,
        wo_shift=train_config['model']['wo_shift'] if 'wo_shift' in train_config['model'] else False,
        in_channels=train_config['model']['in_chans'] if 'in_chans' in train_config['model'] else 4,
        use_checkpoint=train_config['model']['use_checkpoint'] if 'use_checkpoint' in train_config['model'] else False,
    )

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    # load pretrained model
    if 'weight_init' in train_config['train']:
        checkpoint = torch.load(train_config['train']['weight_init'], map_location=lambda storage, loc: storage)
        # remove the prefix 'module.' from the keys
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        model = load_weights_with_shape_check(model, checkpoint, rank=rank)
        ema = load_weights_with_shape_check(ema, checkpoint, rank=rank)
        if accelerator.is_main_process:
            logger.info(f"Loaded pretrained model from {train_config['train']['weight_init']}")
    requires_grad(ema, False)
    
    # model = DDP(model.to(device), device_ids=[rank]) # DDP init error
    model = model.to(device)

    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss = train_config['transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config['transport'] else False,
        use_lognorm = train_config['transport']['use_lognorm'] if 'use_lognorm' in train_config['transport'] else False,
    )  # default: velocity; 
    if accelerator.is_main_process:
        logger.info(f"LightningDiT Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        logger.info(f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}")
        logger.info(f'Use lognorm sampling: {train_config["transport"]["use_lognorm"]}')
        logger.info(f'Use cosine loss: {train_config["transport"]["use_cosine_loss"]}')
    opt = torch.optim.AdamW(model.parameters(), lr=train_config['optimizer']['lr'], weight_decay=0, betas=(0.9, train_config['optimizer']['beta2']))
    
    # Setup data
    dataset = ImgLatentDataset(
        data_dir=train_config['data']['data_path'],
        latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
        latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215,
    )
    batch_size_per_gpu = int(np.round(train_config['train']['global_batch_size'] / accelerator.num_processes))
    global_batch_size = batch_size_per_gpu * accelerator.num_processes
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=train_config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images {train_config['data']['data_path']}")
        logger.info(f"Batch size {batch_size_per_gpu} per gpu, with {global_batch_size} global batch size")
    
    if 'valid_path' in train_config['data']:
        valid_dataset = ImgLatentDataset(
            data_dir=train_config['data']['valid_path'],
            latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
            latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size_per_gpu,
            shuffle=True,
            num_workers=train_config['data']['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        if accelerator.is_main_process:
            logger.info(f"Validation Dataset contains {len(valid_dataset):,} images {train_config['data']['valid_path']}")

    train_config['train']['resume'] = train_config['train']['resume'] if 'resume' in train_config['train'] else False

    if train_config['train']['resume']:
        # check if the checkpoint exists
        checkpoint_files = glob(f"{checkpoint_dir}/*.pt")
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
            latest_checkpoint = checkpoint_files[-1]
            checkpoint = torch.load(latest_checkpoint, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'])
            # opt.load_state_dict(checkpoint['opt'])
            ema.load_state_dict(checkpoint['ema'])
            train_steps = int(latest_checkpoint.split('/')[-1].split('.')[0])
            if accelerator.is_main_process:
                logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
        else:
            if accelerator.is_main_process:
                logger.info("No checkpoint found. Starting training from scratch.")

    # Prepare models for training:
    model, opt, loader = accelerator.prepare(model, opt, loader)
    unwrapped_model = accelerator.unwrap_model(model)
    update_ema(ema, unwrapped_model, decay=0)  # Ensure EMA is initialized with synced weights
    accelerator.wait_for_everyone()
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    if not train_config['train']['resume']:
        train_steps = 0
    log_steps = 0
    running_loss = 0
    running_mse = 0
    running_cos = 0
    running_ot = 0

    start_time = time()
    use_checkpoint = train_config['train']['use_checkpoint'] if 'use_checkpoint' in train_config['train'] else True
    if accelerator.is_main_process:
        logger.info(f"Using checkpointing: {use_checkpoint}")

    '''Clustering'''
    cluster_type = train_config['cluster_type']
    if accelerator.process_index == 0:
        if accelerator.is_main_process:
            logger.info(f"Clustering: {cluster_type}")

    if cluster_type == 'kmeans':
        cluster_path = train_config['kmeans']['load_dir']
        with open(os.path.join(cluster_path, "kmeans_clusters.pkl"), "rb") as f:
            ckpt = pickle.load(f)
        kmeans_centers = {cls: torch.from_numpy(mu).to(device=device, dtype=torch.float32) for cls, mu in ckpt["centers"].items()}
        # ckpt["centers"]: dict: class → (K, D)
        if accelerator.process_index == 0:
            if accelerator.is_main_process:
                logger.info(f"Cluster data from: {cluster_path}")

    elif cluster_type == 'gmm':
        cluster_path = train_config['gmm']['load_dir']
        with open(os.path.join(cluster_path, "gmm_clusters.pkl"), "rb") as f:
            ckpt = pickle.load(f)
        # ---- Mean (μ): class -> (K, D) # [30, 8192]
        gmm_means = {
            cls: torch.from_numpy(mu).to(device=device, dtype=torch.float32)
            for cls, mu in ckpt["means"].items()
        }
        # ---- Covariance (Σ)
        # diag: (K, D)
        # full: (K, D, D)
        gmm_covs = {
            cls: torch.from_numpy(cov).to(device=device, dtype=torch.float32)
            for cls, cov in ckpt["covs"].items()
        }
        # ---- Mixture weight (π)
        gmm_weights = {
            cls: torch.from_numpy(w).to(device=device, dtype=torch.float32)
            for cls, w in ckpt["weights"].items()
        }
        # ---- PCA components (class-wise)
        gmm_pca = {
            cls: {
                "components": torch.from_numpy(pca_dict["components"])
                .to(device=device, dtype=torch.float32),
                "mean": torch.from_numpy(pca_dict["mean"])
                .to(device=device, dtype=torch.float32),
            }
            for cls, pca_dict in ckpt["pca_components"].items()
        }
        gmm_labels = ckpt.get("labels", None)
        gmm_use_weight = train_config['gmm']['use_weight']
        gmm_shell = torch.load(os.path.join(cluster_path, 'cluster_shell.pt'))
        reweight_lambda = train_config['gmm']['reweight_lambda']
        if accelerator.is_main_process:
            logger.info(f"Cluster data from: {cluster_path}")

    while True:
        pbar = tqdm(loader, desc=f"Epoch loop", dynamic_ncols=True)
        for x, y in pbar:
            # print(x.shape) # B,32,16,16
            if accelerator.mixed_precision == 'no':
                x = x.to(device, dtype=torch.float32)
                y = y
            else:
                x = x.to(device)
                y = y.to(device)
            model_kwargs = dict(y=y)

            '''Cluster matching'''
            if cluster_type == 'kmeans':
                cluster_means, cluster_ids = get_cluster_kmeans(x, y, kmeans_centers)
                cluster_sigma = torch.ones_like(x)
            elif cluster_type == 'gmm':
                cluster_means, cluster_sigma, cluster_ids = get_cluster_gmm(
                    x,y,
                    gmm_means, gmm_covs, gmm_weights, gmm_pca,
                    use_weight=gmm_use_weight,
                    stochastic=False #True
                )
                # cluster_sigma = torch.ones_like(x) # don't use gmm cov, use org I
                # cluster_sigma = cluster_sigma * 1.3 # x1.3 weighting
            ## normalize sigma
            cluster_sigma = norm_cluster_sigma(cluster_sigma)

            # learned mu/sigma
            # learned_sigma = torch.randn_like(x)
            # learned_mu = torch.zeros_like(x)
            # learned_sigma = torch.ones_like(x)
            # cluster_means = learned_mu
            # cluster_sigma = learned_sigma

            # loss_dict = transport.training_losses(model, x, model_kwargs)
            loss_dict = transport.training_losses_learnable_eps2(model, x, model_kwargs, learned_mu=cluster_means, learned_sigma=cluster_sigma) # for learnable eps

            '''reweighting fo data points by cluster shell'''
            distance = torch.norm(x.view(x.shape[0], -1) - cluster_means.view(x.shape[0], -1), dim=1)  # (B,)
            reweight = compute_reweight(distance, cluster_ids, gmm_shell, lambda_=reweight_lambda)

            # MSE Loss
            mse_loss = (loss_dict["loss"] * reweight).mean()
            loss = mse_loss
            # Cosine Loss
            if 'cos_loss' in loss_dict:
                loss = loss + (loss_dict["cos_loss"] * reweight).mean()
            # OT Loss
            if 'ot_loss' in loss_dict:
                lambda_ot = 0.005
                # loss = loss + (lambda_ot * loss_dict["ot_loss"].mean())

            opt.zero_grad()
            accelerator.backward(loss)
            if 'max_grad_norm' in train_config['optimizer']:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train_config['optimizer']['max_grad_norm'])
            opt.step()
            unwrapped_model = accelerator.unwrap_model(model)
            update_ema(ema, unwrapped_model)
            # update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            running_mse += mse_loss.item()
            if 'cos_loss' in loss_dict:
                running_cos += loss_dict["cos_loss"].mean().item()
            if 'ot_loss' in loss_dict:
                running_ot += loss_dict["ot_loss"].mean().item()

            log_steps += 1
            train_steps += 1

            if accelerator.is_main_process:
                pbar.set_postfix({"loss": f"{loss:.4f}"})

            if (train_steps) % train_config['train']['log_every'] == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                # 개별 로스들 텐서화 (all_reduce를 위함)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_mse = torch.tensor(running_mse / log_steps, device=device)
                avg_cos = torch.tensor(running_cos / log_steps, device=device)
                avg_ot = torch.tensor(running_ot / log_steps, device=device)

                # 모든 프로세스의 결과를 합산
                for l_tensor in [avg_loss, avg_mse, avg_cos, avg_ot]:
                    # dist.all_reduce(l_tensor, op=dist.ReduceOp.SUM)
                    l_tensor = accelerator.reduce(l_tensor, reduction="sum")
                    l_tensor.copy_(l_tensor)

                world_size = dist.get_world_size()
                avg_loss = avg_loss.item() / world_size
                avg_mse = avg_mse.item() / world_size
                avg_cos = avg_cos.item() / world_size
                avg_ot = avg_ot.item() / world_size

                if accelerator.is_main_process:
                    # 1. Logger (터미널) 출력
                    logger.info(
                        f"(step={train_steps:07d}) Loss: {avg_loss:.4f} (MSE: {avg_mse:.4f}, Cos: {avg_cos:.4f}, OT: {avg_ot:.4f})")

                    # 2. Tensorboard 기록 (기존 Loss/train 유지 + 개별 항목 추가)
                    writer.add_scalar('Loss/train', avg_loss, train_steps)
                    writer.add_scalar('Loss/mse', avg_mse, train_steps)
                    writer.add_scalar('Loss/cos', avg_cos, train_steps)
                    # writer.add_scalar('Loss/ot', avg_ot, train_steps)

                # --- 변수 초기화 (중요) ---
                running_loss = 0
                running_mse = 0
                running_cos = 0
                running_ot = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % train_config['train']['ckpt_every'] == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "config": train_config,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    if accelerator.is_main_process:
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

                # Evaluate on validation set
                if 'valid_path' in train_config['data']:
                    if accelerator.is_main_process:
                        logger.info(f"Start evaluating at step {train_steps}")
                    if cluster_type == 'kmeans':
                        val_loss = evaluate(model, valid_loader, device, transport, cluster_type="kmeans",kmeans_centers=kmeans_centers)
                    elif cluster_type == 'gmm':
                        val_loss = evaluate_eps(model,valid_loader,device,transport, cluster_type="gmm",gmm_means=gmm_means,gmm_covs=gmm_covs,gmm_weights=gmm_weights,gmm_pca=gmm_pca,gmm_use_weight=gmm_use_weight,)
                        # val_loss = evaluate(model,valid_loader,device,transport)
                    val_loss = torch.tensor(val_loss, device=device)
                    dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                    val_loss = val_loss.item() / dist.get_world_size()
                    if accelerator.is_main_process:
                        logger.info(f"Validation Loss: {val_loss:.4f}")
                        writer.add_scalar('Loss/validation', val_loss, train_steps)
                    model.train()
            if train_steps >= train_config['train']['max_steps']:
                break
        if train_steps >= train_config['train']['max_steps']:
            break

    if accelerator.is_main_process:
        logger.info("Done!")

    return accelerator

def load_weights_with_shape_check(model, checkpoint, rank=0):
    
    model_state_dict = model.state_dict()
    # check shape and load weights
    for name, param in checkpoint['model'].items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            elif name == 'x_embedder.proj.weight':
                # special case for x_embedder.proj.weight
                # the pretrained model is trained with 256x256 images
                # we can load the weights by resizing the weights
                # and keep the first 3 channels the same
                weight = torch.zeros_like(model_state_dict[name])
                weight[:, :16] = param[:, :16]
                model_state_dict[name] = weight
            else:
                if rank == 0:
                    print(f"Skipping loading parameter '{name}' due to shape mismatch: "
                        f"checkpoint shape {param.shape}, model shape {model_state_dict[name].shape}")
        else:
            if rank == 0:
                print(f"Parameter '{name}' not found in model, skipping.")
    # load state dict
    model.load_state_dict(model_state_dict, strict=False)
    
    return model

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0  # 분산 초기화가 안 된 경우 rank=0으로 가정

    if rank == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def get_cluster_kmeans(x, y, centers):
    """
    Compute cluster mean for each sample in the batch.

    Args:
        x: Latent tensor of shape [B, C, H, W]
        y: Class labels of shape [B]
        centers: dict of class_id -> (K, D) cluster centers, tensor on device

    Returns:
        cluster_means: Tensor of same shape as x
        cluster_ids: Tensor of shape [B] with assigned cluster index for each sample
    """
    B, C, H, W = x.shape
    latents_flat = x.view(B, -1)  # [B, D]
    cluster_means = torch.zeros_like(latents_flat)
    cluster_ids = torch.zeros(B, dtype=torch.long, device=x.device)

    start_idx = 0
    for cls, cls_centers in centers.items():
        mask = (y == cls)
        if mask.sum() == 0:
            continue
        cls_latents = latents_flat[mask]  # (B_cls, D)
        dists = ((cls_latents[:, None, :] - cls_centers[None, :, :]) ** 2).sum(dim=2)  # (B_cls, K)
        k_idx = dists.argmin(dim=1)  # (B_cls,)
        cluster_ids[mask] = k_idx
        cluster_means[mask] = cls_centers[k_idx]  # assign closest cluster center

    cluster_means = cluster_means.view(B, C, H, W)
    return cluster_means, cluster_ids

@torch.no_grad()
## 1st~4th에 쓴 함수, cov가 단일 scalar로 단순화됨(오류)
def get_cluster_gmm_bf(
        x, y, gmm_means, gmm_covs, gmm_weights, use_weight=True,
        stochastic=True, eps=1e-8
):
    device = x.device

    # Flatten
    if x.dim() == 4:
        B, C, H, W = x.shape
        D = C * H * W  # 8192
        x_flat = x.view(B, D)
    else:
        B, D = x.shape
        x_flat = x

    cluster_means_out = torch.empty_like(x_flat)
    cluster_sigma_out = torch.empty_like(x_flat)
    cluster_ids_out = torch.empty(B, dtype=torch.long, device=device)

    for cls_tensor in torch.unique(y):
        cls = int(cls_tensor.item())
        idx = (y == cls)
        if idx.sum() == 0: continue

        x_cls = x_flat[idx]  # (B_cls, 8192)
        B_cls = x_cls.shape[0]

        means = gmm_means[cls].to(device)  # (50, 8192)
        weights = gmm_weights[cls].to(device)  # (50,)
        covs_pca = gmm_covs[cls].to(device)  # (50, 256)

        K = means.shape[0]

        # ✅ Mahalanobis 거리 계산 (차원 완벽 호환)
        diff = x_cls[:, None, :] - means[None, :, :]  # (B_cls, K, 8192)
        sq_diff = diff ** 2  # (B_cls, K, 8192)

        # covs를 K로만 broadcasting
        covs_scale = covs_pca.mean(dim=1) + eps  # (K,)
        mahalanobis = sq_diff.sum(dim=2) / covs_scale[None, :]  # (B_cls, K,8192).sum -> (B_cls,K) / (K,) -> (B_cls,K)

        # Log-determinant 근사 (단순화)
        log_det = D * torch.log(covs_scale)  # (K,)

        log_prob = -0.5 * (mahalanobis + log_det[None, :]) + torch.log(weights[None, :] + eps)

        posterior = F.softmax(log_prob, dim=1)  # (B_cls, K)

        if stochastic:
            cluster_ids = torch.distributions.Categorical(posterior).sample()
        else:
            cluster_ids = posterior.argmax(dim=1)

        # 선택된 클러스터 파라미터
        sel_means = means[cluster_ids]  # (B_cls, 8192)
        sel_covs = covs_scale[cluster_ids]  # (B_cls,)
        sel_sigma = torch.sqrt(sel_covs[:, None]).expand_as(sel_means)  # (B_cls, 8192)

        cluster_means_out[idx] = sel_means
        cluster_sigma_out[idx] = sel_sigma
        cluster_ids_out[idx] = cluster_ids

    if x.dim() == 4:
        cluster_means_out = cluster_means_out.view(B, C, H, W)
        cluster_sigma_out = cluster_sigma_out.view(B, C, H, W)

    return cluster_means_out, cluster_sigma_out, cluster_ids_out

def _as_U_dD(components: torch.Tensor, D: int) -> torch.Tensor:
    """
    Return U with shape (d, D) where d=256, D=8192.
    We will use: x_pca = (x - pca_mean) @ U.T  -> (B, d)
    """
    if components.dim() != 2:
        raise ValueError("PCA components must be a 2D tensor.")
    a, b = components.shape
    if b == D:      # (d, D)
        return components
    if a == D:      # (D, d) -> transpose to (d, D)
        return components.T
    raise ValueError(f"Unexpected PCA components shape {components.shape}. Expected (d,D) or (D,d) with D={D}.")

## 5th~
def get_cluster_gmm(
    x, y,
    gmm_means, gmm_covs, gmm_weights, gmm_pca,
    use_weight=True,
    stochastic=True,
    eps=1e-8,
    # optional stabilizers (필요 없으면 기본값 그대로)
    var_floor=0.0,          # e.g. 1e-6 ~ 1e-3 (너 공간 스케일에 맞춰)
):
    device = x.device

    # Flatten
    if x.dim() == 4:
        B, C, H, W = x.shape
        D = C * H * W  # e.g. 8192
        x_flat = x.view(B, D)
    else:
        B, D = x.shape
        x_flat = x

    cluster_means_out = torch.empty_like(x_flat)
    cluster_sigma_out = torch.empty_like(x_flat)
    cluster_ids_out = torch.empty(B, dtype=torch.long, device=device)

    # Process per class to use class-wise GMM/PCA
    for cls_tensor in torch.unique(y):
        cls = int(cls_tensor.item())
        idx = (y == cls)
        if idx.sum() == 0:
            continue

        x_cls = x_flat[idx]  # (B_cls, D)

        means = gmm_means[cls].to(device)       # (K, D)  data-space means
        weights = gmm_weights[cls].to(device)   # (K,)
        v = gmm_covs[cls].to(device)            # (K, d=256) PCA-space diag variances

        pca_comp = gmm_pca[cls]["components"].to(device)  # (d,D) or (D,d)
        pca_mean = gmm_pca[cls]["mean"].to(device)        # (D,)

        U = _as_U_dD(pca_comp, D)               # (d, D)
        d = U.shape[0]
        K = means.shape[0]

        # --- 1) Project x and means to PCA space ---
        # x_pca = U (x - m)  implemented as (x-m) @ U.T
        x_pca = (x_cls - pca_mean) @ U.T        # (B_cls, d)
        means_pca = (means - pca_mean) @ U.T    # (K, d)

        # --- 2) Compute posterior in PCA space using diag Gaussian ---
        v_safe = torch.clamp(v, min=eps)        # (K, d)

        diff = x_pca[:, None, :] - means_pca[None, :, :]        # (B_cls, K, d)
        mahal = (diff * diff / v_safe[None, :, :]).sum(dim=2)   # (B_cls, K)
        log_det = torch.log(v_safe).sum(dim=1)                  # (K,)

        log_prob = -0.5 * (mahal + log_det[None, :])            # (B_cls, K)
        if use_weight:
            log_prob = log_prob + torch.log(weights[None, :] + eps)

        posterior = F.softmax(log_prob, dim=1)                  # (B_cls, K)

        if stochastic:
            cluster_ids = torch.distributions.Categorical(posterior).sample()  # (B_cls,)
        else:
            cluster_ids = posterior.argmax(dim=1)                               # (B_cls,)

        # --- 3) Select data-space mean ---
        sel_means = means[cluster_ids]  # (B_cls, D)

        # --- 4) Reverse PCA to get data-space diagonal sigma ---
        # diag_var_i = sum_j v_{k,j} * U_{j,i}^2
        # U2: (d, D), sel_v: (B_cls, d) -> diag_var: (B_cls, D)
        U2 = U * U
        sel_v = v_safe[cluster_ids]            # (B_cls, d)
        diag_var = sel_v @ U2                  # (B_cls, D)

        if var_floor > 0.0:
            diag_var = torch.clamp(diag_var, min=var_floor)

        sel_sigma = torch.sqrt(diag_var + eps)  # (B_cls, D)

        cluster_means_out[idx] = sel_means
        cluster_sigma_out[idx] = sel_sigma
        cluster_ids_out[idx] = cluster_ids

    if x.dim() == 4:
        cluster_means_out = cluster_means_out.view(B, C, H, W)
        cluster_sigma_out = cluster_sigma_out.view(B, C, H, W)

    return cluster_means_out, cluster_sigma_out, cluster_ids_out

def norm_cluster_sigma(  #TO-DO # 배치단위에서만 정규화됨,,, 의도한 건 전체 cluster에서의 정규화임,, 나중에 수정하자
    cluster_sigma,
    lambda_mix=1.0,      # 1.0 = full normalization, <1 = mix with I
    var_floor=0.0,       # optional lower bound on variance
    var_ceil=None,       # optional upper bound on variance
    eps=1e-8
):
    """
    Normalize cluster_sigma so that:
      - Global average variance = 1 (I-scale alignment)
      - Relative structure per sample preserved
      - Optional mixing with identity to avoid extreme values

    Input:
        cluster_sigma: (B, D)  data-space std
    Output:
        norm_cluster_sigma: (B, D)
    """

    # 1️⃣ Compute global mean variance
    var = cluster_sigma ** 2
    global_mean_var = var.mean()

    # 2️⃣ Global scaling to match I
    alpha = 1.0 / (global_mean_var + eps)
    scaled_var = alpha * var

    # 3️⃣ Optional mix with identity (variance=1)
    if lambda_mix < 1.0:
        scaled_var = (1.0 - lambda_mix) * 1.0 + lambda_mix * scaled_var

    # 4️⃣ Optional clipping
    if var_floor > 0.0:
        scaled_var = torch.clamp(scaled_var, min=var_floor)
    if var_ceil is not None:
        scaled_var = torch.clamp(scaled_var, max=var_ceil)

    # 5️⃣ Back to std
    norm_cluster_sigma = torch.sqrt(scaled_var + eps)

    return norm_cluster_sigma

def compute_reweight(dist, cluster_ids, gmm_shell, lambda_=0.6):
    # dist:        (B,)  float tensor
    # cluster_ids: (B,)  long tensor
    # gmm_shell:   dict {int: {'mean': float, 'std': float}}

    mu_d = torch.tensor(
        [gmm_shell[int(cid)]['mean'] for cid in cluster_ids],
        dtype=dist.dtype, device=dist.device
    )  # (B,)
    sig_d = torch.tensor(
        [gmm_shell[int(cid)]['std'] for cid in cluster_ids],
        dtype=dist.dtype, device=dist.device
    )  # (B,)
    z = (dist - mu_d) / (sig_d + 1e-6)  # (B,)
    w = 1.0 + lambda_ * z ** 2          # (B,)
    return w                             # (B,)

@torch.no_grad()
def evaluate_eps(
    model,
    valid_loader,
    device,
    transport,
    cluster_type,
    kmeans_centers=None,
    gmm_means=None,
    gmm_covs=None,
    gmm_weights=None,
    gmm_pca=None,
    gmm_use_weight=True,
    clip_range=(0.0, 1.0),
):
    """
    Evaluate model on the validation dataset using cluster-based learnable epsilon loss.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    disable_tqdm = not (
        dist.is_available() and dist.is_initialized() and dist.get_rank() == 0
    )

    for x, y in tqdm(
        valid_loader,
        desc="Validation",
        leave=False,
        dynamic_ncols=True,
        disable=disable_tqdm,
    ):
        x = x.to(device)
        y = y.to(device)
        model_kwargs = dict(y=y)

        # -------------------------
        # Cluster matching
        # -------------------------
        if cluster_type == "kmeans":
            cluster_means, cluster_ids = get_cluster_kmeans(
                x, y, kmeans_centers
            )
            cluster_sigma = torch.ones_like(x)
        elif cluster_type == "gmm":
            cluster_means, cluster_sigma, cluster_ids = get_cluster_gmm(
                x, y,
                gmm_means, gmm_covs, gmm_weights, gmm_pca,
                use_weight=gmm_use_weight,
                stochastic=False
            )
            # cluster_sigma = torch.ones_like(x)
        else:
            raise ValueError(f"Unknown cluster_type: {cluster_type}")

        # -------------------------
        # Compute loss
        # -------------------------
        loss_dict = transport.training_losses_learnable_eps2(
            model,
            x,
            model_kwargs,
            learned_mu=cluster_means,
            learned_sigma=cluster_sigma,
        )

        if "cos_loss" in loss_dict:
            loss = loss_dict["loss"].mean() + loss_dict["cos_loss"].mean()
        else:
            loss = loss_dict["loss"].mean()

        running_loss += loss.detach()
        num_batches += 1

    avg_loss = running_loss / num_batches
    return avg_loss
    
@torch.no_grad()
def evaluate(
    model,
    valid_loader,
    device,
    transport,
    clip_range=(0.0, 1.0),
):
    """
    Evaluate model on the validation dataset using cluster-based learnable epsilon loss.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    disable_tqdm = not (
        dist.is_available() and dist.is_initialized() and dist.get_rank() == 0
    )

    for x, y in tqdm(
        valid_loader,
        desc="Validation",
        leave=False,
        dynamic_ncols=True,
        disable=disable_tqdm,
    ):
        x = x.to(device)
        y = y.to(device)
        model_kwargs = dict(y=y)

        # -------------------------
        # Compute loss
        # -------------------------
        loss_dict = transport.training_losses(model, x, model_kwargs)


        if "cos_loss" in loss_dict:
            loss = loss_dict["loss"].mean() + loss_dict["cos_loss"].mean()
        else:
            loss = loss_dict["loss"].mean()

        running_loss += loss.detach()
        num_batches += 1

    avg_loss = running_loss / num_batches
    return avg_loss

if __name__ == "__main__":
    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/debug.yaml')
    args = parser.parse_args()

    accelerator = Accelerator()
    train_config = load_config(args.config)
    do_train(train_config, accelerator)