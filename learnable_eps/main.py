import os
import sys
# 절대경로로 vavae/DiT 폴더를 sys.path에 추가
vavae_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vavae'))
DiT_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(vavae_path)
sys.path.append(DiT_path)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only

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
import argparse
from time import time
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
import pytorch_lightning as pl

from diffusers.models import AutoencoderKL
from models.lightningdit import LightningDiT_models
from transport import create_transport, Sampler
from accelerate import Accelerator
from datasets.img_latent_dataset import ImgLatentDataset

# print("vavae_path:", vavae_path)
# print("sys.path:\n", "\n".join(sys.path))

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
from train import DataModuleFromConfig


def do_train(train_config, accelerator):
    """
    Trains a LightningDiT.
    """
    # Setup accelerator:
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(train_config['train']['output_dir'],
                    exist_ok=True)  # Make results folder (holds all experiment subfolders)
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
        config_str = json.dumps(train_config, indent=4)
        writer.add_text('training configs', config_str, global_step=0)
    checkpoint_dir = f"{train_config['train']['output_dir']}/{train_config['train']['exp_name']}/checkpoints"

    # get rank
    rank = accelerator.local_process_index

    # Create model:

    """
    Model 1: VA-VAE
    """
    model1_config_files = ["model1_f16d32_vfdinov2.yaml"]
    model1_configs = [OmegaConf.load(f) for f in model1_config_files]
    model1_config_merged = OmegaConf.merge(*model1_configs)
    model1 = instantiate_from_config(model1_config_merged.model)
    # model1.to(device)

    if 'downsample_ratio' in train_config['vae']:
        downsample_ratio = train_config['vae']['downsample_ratio']
    else:
        downsample_ratio = 16
    assert train_config['data'][
               'image_size'] % downsample_ratio == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = train_config['data']['image_size'] // downsample_ratio

    """
    Model 2: LightningDiT
    """
    model2 = LightningDiT_models[train_config['model']['model_type']](
        input_size=train_config['data']['image_size'], # latent x, direct image data
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model']['use_swiglu'] if 'use_swiglu' in train_config['model'] else False,
        use_rope=train_config['model']['use_rope'] if 'use_rope' in train_config['model'] else False,
        use_rmsnorm=train_config['model']['use_rmsnorm'] if 'use_rmsnorm' in train_config['model'] else False,
        wo_shift=train_config['model']['wo_shift'] if 'wo_shift' in train_config['model'] else False,
        in_channels=train_config['model']['in_chans'] if 'in_chans' in train_config['model'] else 4,
        use_checkpoint=train_config['model']['use_checkpoint'] if 'use_checkpoint' in train_config['model'] else False,
    )

    ema = deepcopy(model2).to(device)  # Create an EMA of the model for use after training

    # load pretrained model
    if 'weight_init' in train_config['train']:
        checkpoint = torch.load(train_config['train']['weight_init'], map_location=lambda storage, loc: storage)
        # remove the prefix 'module.' from the keys
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        model2 = load_weights_with_shape_check(model2, checkpoint, rank=rank)
        ema = load_weights_with_shape_check(ema, checkpoint, rank=rank)
        if accelerator.is_main_process:
            logger.info(f"Loaded pretrained model from {train_config['train']['weight_init']}")
    requires_grad(ema, False)

    # model = DDP(model.to(device), device_ids=[rank]) # DDP init error

    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss=train_config['transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config[
            'transport'] else False,
        use_lognorm=train_config['transport']['use_lognorm'] if 'use_lognorm' in train_config['transport'] else False,
    )  # default: velocity;
    if accelerator.is_main_process:
        logger.info(f"LightningDiT Parameters: {sum(p.numel() for p in model2.parameters()) / 1e6:.2f}M")
        logger.info(
            f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}")
        logger.info(f'Use lognorm sampling: {train_config["transport"]["use_lognorm"]}')
        logger.info(f'Use cosine loss: {train_config["transport"]["use_cosine_loss"]}')
    # model1 autoencoder, discriminator parameter
    param1_ae = (list(model1.encoder.parameters()) +
                 list(model1.decoder.parameters()) +
                 list(model1.quant_conv.parameters()) +
                 list(model1.post_quant_conv.parameters()))
    param1_disc = model1.loss.discriminator.parameters()

    # optimizer
    opt = torch.optim.AdamW(
        param1_ae + list(model2.parameters()),
        lr=train_config['optimizer']['lr'], #0.0002
        weight_decay=0,
        betas=(0.9, train_config['optimizer']['beta2'])
    )
    opt_disc = torch.optim.Adam(
        param1_disc,
        lr=train_config['optimizer']['lr'], # 0.0001
        weight_decay=0,
        betas=(0.9, train_config['optimizer']['beta2']) # 0.5, 0.9
    )

    '''data for DiT(model2)'''
    # dataset = ImgLatentDataset(
    #     data_dir=train_config['data']['data_path'],
    #     latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
    #     latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config[
    #         'data'] else 0.18215,
    # )
    '''data for VA-VAE(model1)'''
    data = instantiate_from_config(model1_config_merged.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    train_dataset = data.datasets["train"]
    loader1 = DataLoader(
        train_dataset,
        batch_size=data.batch_size,
        shuffle=True,
        num_workers=data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dataset = data.datasets["validation"]
    val_loader1 = DataLoader(
        val_dataset,
        batch_size=data.batch_size,
        shuffle=True,
        num_workers=data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    batch_size_per_gpu = int(np.round(train_config['train']['global_batch_size'] / accelerator.num_processes))
    global_batch_size = batch_size_per_gpu * accelerator.num_processes

    # loader = DataLoader(
    #     dataset,
    #     batch_size=batch_size_per_gpu,
    #     shuffle=True,
    #     num_workers=train_config['data']['num_workers'],
    #     pin_memory=True,
    #     drop_last=True
    # )
    # if accelerator.is_main_process:
    #     logger.info(f"Dataset contains {len(dataset):,} images {train_config['data']['data_path']}")
    #     logger.info(f"Batch size {batch_size_per_gpu} per gpu, with {global_batch_size} global batch size")

    if 'valid_path' in train_config['data']:
        valid_dataset = ImgLatentDataset(
            data_dir=train_config['data']['valid_path'],
            latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
            latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config[
                'data'] else 0.18215,
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
            logger.info(
                f"Validation Dataset contains {len(valid_dataset):,} images {train_config['data']['valid_path']}")

    # Prepare models for training:
    model1, model2, (opt,opt_disc), loader = accelerator.prepare(model1, model2, (opt,opt_disc), loader1)
    # DDP error로 acc로 대체함

    # update_ema(ema, model1.modules, decay=0)  # Ensure EMA is initialized with synced weights # check
    # update_ema(ema, model2.modules, decay=0)  # Ensure EMA is initialized with synced weights
    model1.train()  # important! This enables embedding dropout for classifier-free guidance
    model2.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    train_config['train']['resume'] = train_config['train']['resume'] if 'resume' in train_config['train'] else False

    if train_config['train']['resume']:
        # check if the checkpoint exists
        checkpoint_files = glob(f"{checkpoint_dir}/*.pt")
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: os.path.getsize(x))
            latest_checkpoint = checkpoint_files[-1]
            checkpoint = torch.load(latest_checkpoint, map_location=lambda storage, loc: storage)
            model2.load_state_dict(checkpoint['model'])
            # opt.load_state_dict(checkpoint['opt'])
            ema.load_state_dict(checkpoint['ema'])
            train_steps = int(latest_checkpoint.split('/')[-1].split('.')[0])
            if accelerator.is_main_process:
                logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
        else:
            if accelerator.is_main_process:
                logger.info("No checkpoint found. Starting training from scratch.")

    # Variables for monitoring/logging purposes:
    if not train_config['train']['resume']:
        train_steps = 0
    log_steps = 0
    running_loss2 = 0
    running_loss1_ae = 0
    running_loss1_disc = 0
    start_time = time()
    use_checkpoint = train_config['train']['use_checkpoint'] if 'use_checkpoint' in train_config['train'] else True
    if accelerator.is_main_process:
        logger.info(f"Using checkpointing: {use_checkpoint}")

    while True:
        for batch in loader:
            x = batch['image']
            y = batch['class_label']
            # print(batch.keys())
                # dict_keys(['image', 'relpath', 'synsets', 'class_label', 'human_label', 'file_path_'])

            if accelerator.mixed_precision == 'no':
                x = x.to(device, dtype=torch.float32)
                y = y
            else:
                x = x.to(device)
                y = y.to(device)
            model_kwargs = dict(y=y)
                # x: torch.Size([1, 256, 256, 3])

            """ loss1 """
            loss1_ae, loss1_disc, posterior = model1.module.training_step_eps(batch, batch_idx=None)
            # get mu, sigma from this VA-VAE(model1)

            """mu, sigma interpolate for usage as DiT eps"""
            learned_mu, learned_sigma = posterior.mu_sigma()
            learned_mu = learned_mu.permute(0, 2, 3, 1)     # torch.Size([1, 32, 16, 16]) -> torch.Size([1, 16, 16, 32])
            learned_sigma = learned_sigma.permute(0, 2, 3, 1)   # torch.Size([1, 32, 16, 16]) -> torch.Size([1, 16, 16, 32])
            # interpolate for learnable eps"""
            learned_mu = learned_mu.unsqueeze(2).repeat(1, 1, 16, 1, 1)  # [1, 16, 16, 16, 3]
            learned_mu = learned_mu.view(-1, 256, 16, 3)
            learned_mu = learned_mu.unsqueeze(3).repeat(1, 1, 1, 16, 1)  # [1, 256, 16, 16, 3]
            learned_mu = learned_mu.view(-1, 256, 256, 3)
            learned_sigma = learned_sigma.unsqueeze(2).repeat(1, 1, 16, 1, 1)  # [1, 16, 16, 16, 3]
            learned_sigma = learned_sigma.view(-1, 256, 16, 3)
            learned_sigma = learned_sigma.unsqueeze(3).repeat(1, 1, 1, 16, 1)  # [1, 256, 16, 16, 3]
            learned_sigma = learned_sigma.view(-1, 256, 256, 3)

            """ loss2 """
            # loss_dict2 = transport.training_losses(model2, x, model_kwargs)
            loss_dict2 = transport.training_losses_learnable_eps(model2, x, model_kwargs, learned_mu=learned_mu, learned_sigma=learned_sigma) # for learnable eps

            if 'cos_loss' in loss_dict2:
                mse_loss = loss_dict2["loss"].mean()
                loss2 = loss_dict2["cos_loss"].mean() + mse_loss
            else:
                loss2 = loss_dict2["loss"].mean()

            """final loss"""
            loss = loss1_ae + loss2 # check loss vf_weight
            opt.zero_grad()         # check optimizer
            accelerator.backward(loss)
            if 'max_grad_norm' in train_config['optimizer']:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model2.parameters(), train_config['optimizer']['max_grad_norm'])
            opt.step()
            update_ema(ema, model2.module)  # check ema, model1(ae)에는 굳이 안 써도 ㄱㅊ, 써서 효과가 있을 수도, model1(disc) 안 쓰는 게 나음

            """discriminator loss"""
            opt_disc.zero_grad()  # check optimizer
            accelerator.backward(loss1_disc)
            opt_disc.step()

            # Log loss values:
            if 'cos_loss' in loss_dict2:
                running_loss2 += mse_loss.item()
            else:
                running_loss2 += loss2.item()
            running_loss1_ae += loss1_ae.item()
            running_loss1_disc += loss1_disc.item()

            log_steps += 1
            train_steps += 1
            if train_steps % train_config['train']['log_every'] == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                '''loss2'''
                avg_loss2 = torch.tensor(running_loss2 / log_steps, device=device)
                dist.all_reduce(avg_loss2, op=dist.ReduceOp.SUM)
                avg_loss2 = avg_loss2.item() / dist.get_world_size()
                '''loss1_ae'''
                avg_loss1_ae = torch.tensor(running_loss1_ae / log_steps, device=device)
                dist.all_reduce(avg_loss1_ae, op=dist.ReduceOp.SUM)
                avg_loss1_ae = avg_loss1_ae.item() / dist.get_world_size()
                '''loss1_disc'''
                avg_loss1_disc = torch.tensor(running_loss1_disc / log_steps, device=device)
                dist.all_reduce(avg_loss1_disc, op=dist.ReduceOp.SUM)
                avg_loss1_disc = avg_loss1_disc.item() / dist.get_world_size()

                if accelerator.is_main_process:
                    logger.info(
                        f"(step={train_steps:07d}) Loss2: {avg_loss2:.4f}, Loss1_ae: {avg_loss1_ae:.4f}, Loss1_disc: {avg_loss1_disc:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    writer.add_scalar('Loss2/train', avg_loss2, train_steps)
                    writer.add_scalar('Loss1_ae/train', avg_loss1_ae, train_steps)
                    writer.add_scalar('Loss1_disc/train', avg_loss1_disc, train_steps)

                # Reset monitoring variables:
                running_loss2 = 0
                running_loss1_ae = 0
                running_loss1_disc = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % train_config['train']['ckpt_every'] == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model2.module.state_dict(),
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
                    val_loss = evaluate(model2, valid_loader, device, transport, (0.0, 1.0))
                    dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                    val_loss = val_loss.item() / dist.get_world_size()
                    if accelerator.is_main_process:
                        logger.info(f"Validation Loss: {val_loss:.4f}")
                        writer.add_scalar('Loss/validation', val_loss, train_steps)
                    model2.train()
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


if __name__ == "__main__":
    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/debug.yaml')
    args = parser.parse_args()

    accelerator = Accelerator()
    train_config = load_config(args.config)
    do_train(train_config, accelerator)