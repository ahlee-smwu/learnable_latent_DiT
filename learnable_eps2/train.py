"""
Training Codes of LightningDiT together with VA-VAE.
It envolves advanced training methods, sampling methods, 
architecture design methods, computation methods. We achieve
state-of-the-art FID 1.35 on ImageNet 256x256.

by Maple (Jingfeng Yao) from HUST-VL
"""

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

from diffusers.models import AutoencoderKL
from models.lightningdit import LightningDiT_models
from transport import create_transport, Sampler
from accelerate import Accelerator
from datasets.img_latent_dataset import ImgLatentDataset
import pickle

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
    start_time = time()
    use_checkpoint = train_config['train']['use_checkpoint'] if 'use_checkpoint' in train_config['train'] else True
    if accelerator.is_main_process:
        logger.info(f"Using checkpointing: {use_checkpoint}")

    '''Clustering'''
    cluster_path = train_config['kmeans']['load_dir']
    with open(os.path.join(cluster_path, "kmeans_clusters.pkl"), "rb") as f:
        ckpt = pickle.load(f)
    centers = {cls: torch.from_numpy(mu).to(device=device, dtype=torch.float32) for cls, mu in ckpt["centers"].items()}
    # ckpt["centers"]: dict: class → (K, D)

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

            # Cluster matching
            cluster_means, cluster_ids = get_cluster_means(x, y, centers)
            cluster_sigma = torch.ones_like(x)

            # # Cluster matching
            # B, C, H, W = x.shape
            # latents_flat = x.view(B, -1)  # [B, D]
            #
            # cluster_means_list = []
            # cluster_ids_list = torch.zeros(B, dtype=torch.long, device=device)
            #
            # # batch vectorized assignment
            # for cls in centers.keys():
            #     mask = (y == cls)
            #     if mask.sum() == 0:
            #         continue
            #     cls_latents = latents_flat[mask]  # (B_cls, D)
            #     cls_centers = centers[cls]  # (K, D)
            #
            #     # 거리 계산: (B_cls, K)
            #     dists = ((cls_latents[:, None, :] - cls_centers[None, :, :]) ** 2).sum(dim=2)
            #     k_idx = dists.argmin(dim=1)  # (B_cls,)
            #
            #     cluster_ids_list[mask] = k_idx
            #
            #     cluster_means_list.append(cls_centers[k_idx])  # list of (B_cls, D)
            #
            # # list → Tensor, 전체 batch 연결
            # cluster_means = torch.zeros_like(latents_flat)
            # start_idx = 0
            # for cls in centers.keys():
            #     mask = (y == cls)
            #     num_samples = mask.sum().item()
            #     if num_samples == 0:
            #         continue
            #     cluster_means[mask] = cluster_means_list[start_idx]
            #     start_idx += 1

            # learned mu/sigma
            learned_mu = torch.randn_like(x)
            learned_sigma = torch.randn_like(x)

            # loss_dict = transport.training_losses(model, x, model_kwargs)
            loss_dict = transport.training_losses_learnable_eps2(model, x, model_kwargs, learned_mu=cluster_means, learned_sigma=cluster_sigma) # for learnable eps

            if 'cos_loss' in loss_dict:
                mse_loss = loss_dict["loss"].mean()
                loss = loss_dict["cos_loss"].mean() + mse_loss
            else:
                loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            if 'max_grad_norm' in train_config['optimizer']:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train_config['optimizer']['max_grad_norm'])
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            if 'cos_loss' in loss_dict:
                running_loss += mse_loss.item()
            else:
                running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if accelerator.is_main_process:
                pbar.set_postfix({"loss": f"{loss:.4f}"})

            if train_steps % train_config['train']['log_every'] == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    writer.add_scalar('Loss/train', avg_loss, train_steps)
                # Reset monitoring variables:
                running_loss = 0
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
                    val_loss = evaluate(model, valid_loader, device, transport, centers, (0.0, 1.0))
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

def get_cluster_means(x, y, centers):
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
def evaluate(model, valid_loader, device, transport, centers, clip_range=(0.0, 1.0)):
    """
    Evaluate model on the validation dataset using cluster-based learnable epsilon loss.

    Args:
        model: PyTorch model to evaluate.
        valid_loader: DataLoader for validation dataset.
        device: Device to run evaluation on.
        transport: Transport module providing loss functions.
        centers: dict of class_id -> (K, D) cluster centers, tensor on device
        clip_range: optional tuple (min, max) to clip outputs (not used currently)

    Returns:
        avg_loss: Average validation loss (torch.Tensor on device)
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    disable_tqdm = not (dist.is_available() and dist.is_initialized() and dist.get_rank() == 0)
    for x, y in tqdm(valid_loader, desc="Validation", leave=False, dynamic_ncols=True, disable=disable_tqdm):
        x = x.to(device)
        y = y.to(device)
        model_kwargs = dict(y=y)

        # Compute cluster means
        cluster_means, cluster_ids = get_cluster_means(x, y, centers)
        cluster_sigma = torch.ones_like(x)

        # Compute loss
        loss_dict = transport.training_losses_learnable_eps2(
            model, x, model_kwargs, learned_mu=cluster_means, learned_sigma=cluster_sigma
        )

        # combine losses
        if 'cos_loss' in loss_dict:
            mse_loss = loss_dict["loss"].mean()
            loss = loss_dict["cos_loss"].mean() + mse_loss
        else:
            loss = loss_dict["loss"].mean()

        running_loss += loss
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