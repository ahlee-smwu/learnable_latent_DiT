import os
import sys
# 절대경로로 vavae/DiT 폴더를 sys.path에 추가
vavae_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vavae'))
DiT_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(vavae_path)
sys.path.append(DiT_path)

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import argparse
import os
from safetensors.torch import save_file
from datetime import datetime
from datasets.img_latent_dataset import ImgLatentDataset
from tokenizer.vavae import VA_VAE
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from accelerate import Accelerator
from ldm.models.autoencoder import AutoencoderKL

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def main(args):
    """
    Run a tokenizer on full dataset and save the features.
    """
    assert torch.cuda.is_available(), "Extract features currently requires at least one GPU."

    # Setup DDP:
    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        world_size = dist.get_world_size()
        seed = args.seed + rank
        if rank == 0:
            print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    except:
        print("Failed to initialize DDP. Running in local mode.")
        rank = 0
        device = 0
        world_size = 1
        seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup feature folders:
    output_dir = os.path.join(args.output_path, os.path.splitext(os.path.basename(args.config))[0], f'{args.data_split}_{args.image_size}')
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    accelerator = Accelerator()

    """
    Model 1: VA-VAE
    """
    model1_config_files = [args.config]
    model1_configs = [OmegaConf.load(f) for f in model1_config_files]
    model1_config_merged = OmegaConf.merge(*model1_configs)
    model1 = instantiate_from_config(model1_config_merged.model)
    model1.to(device)

    """ Model layer tunning """
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.shape}")

    model1.new_proj_vae = torch.nn.Conv2d(32 * 2, 3 * 2, kernel_size=1, bias=True)  # vae output dim: 32->3
    model1.new_proj_align1 = torch.nn.Conv2d(3, 64, kernel_size=1, bias=False)  # vf dim -> vae dim
    model1.new_proj_align2 = torch.nn.Conv2d(64, 1024, kernel_size=1, bias=False)  # vf dim -> vae dim
    model1.new_proj_align = torch.nn.Sequential(
        model1.new_proj_align1,
        model1.new_proj_align2
    )

    def new_forward(self, input, sample_posterior=True):
        '''encode'''
        from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
        h = self.encoder(input)
        moments = self.quant_conv(h)
        moments = self.new_proj_vae(moments)  # new layer
        posterior = DiagonalGaussianDistribution(moments)
        # post: (B, D*2(mean, std)) D: latent dimension
        '''sampling'''
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        '''decode'''
        dec = self.decode_eps(z)
        '''vf'''
        if self.use_vf is not None:
            aux_feature = self.foundation_model(input)
            if not self.reverse_proj:
                aux_feature = self.new_proj_align(aux_feature)
            else:
                z = self.new_proj_align(z)
            return dec, posterior, z, aux_feature
            # recon(=dec output), post(=enc output), sample of post, aux

    import types
    model1.forward = types.MethodType(new_forward, model1)

    try:
        model1.load_state_dict(torch.load(model1_config_merged.init_weight)['state_dict'], strict=True)
        print(f"Loaded initial weights1 from {model1_config_merged.init_weight}")
    except:
        print(f"There is no initial weights1 to load.")
        import traceback
        traceback.print_exc()

    '''data for VA-VAE(model1)'''
    data = instantiate_from_config(model1_config_merged.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()

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
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # datasets = [
    #     ImageFolder(args.data_path, transform=tokenizer.img_transform(p_hflip=0.0)),
    #     ImageFolder(args.data_path, transform=tokenizer.img_transform(p_hflip=1.0))
    # ]
    # samplers = [
    #     DistributedSampler(
    #         dataset,
    #         num_replicas=world_size,
    #         rank=rank,
    #         shuffle=False,
    #         seed=args.seed
    #     ) for dataset in datasets
    # ]
    # loaders = [
    #     DataLoader(
    #         dataset,
    #         batch_size=args.batch_size,
    #         shuffle=False,
    #         sampler=sampler,
    #         num_workers=args.num_workers,
    #         pin_memory=True,
    #         drop_last=False
    #     ) for dataset, sampler in zip(datasets, samplers)
    # ]
    # total_data_in_loop = len(loaders[0].dataset)
    # if rank == 0:
    #     print(f"Total data in one loop: {total_data_in_loop}")

    model1, loader1, val_loader1 = accelerator.prepare(model1, loader1, val_loader1)
    model1.eval()

    global_batch_idx = 0

    for batch_idx, batch in enumerate(loader1):
        if hasattr(batch, '__getitem__') and 'image' in batch:
            x = batch['image']
            y = batch['class_label']
        else:
            x = batch[0]
            x = x.permute(0, 2, 3, 1)
            y = batch[1]
            batch = {'image': x, 'class_label': y}

        if accelerator.mixed_precision == 'no':
            x = x.to(device, dtype=torch.float32)
        else:
            x = x.to(device)
            y = y.to(device)
        model_kwargs = dict(y=y)

        with torch.no_grad():
            _, _, posterior = model1.module.training_step_eps(batch, batch_idx=None)

            learned_mu, learned_sigma = posterior.mu_sigma()
            learned_mu = learned_mu.permute(0, 2, 3, 1).contiguous()
            learned_sigma = learned_sigma.permute(0, 2, 3, 1).contiguous()

        # --- 글로벌 배치 인덱스 기반으로 파일명 생성 ---
        save_dict = {
            'mu': learned_mu,
            'sigma': learned_sigma,
            'labels': y
        }
        save_filename = os.path.join(output_dir, f'latents_batch{global_batch_idx:06d}.safetensors')
        save_file(
            save_dict,
            save_filename,
            metadata={
                'batch_idx': str(global_batch_idx),
                'dtype': str(learned_mu.dtype),
                'shape_mu': str(learned_mu.shape),
                'shape_sigma': str(learned_sigma.shape),
                'shape_labels': str(y.shape)
            }
        )

        global_batch_idx += 1

        if rank == 0:
            print(
                f"Saved {save_filename} with shapes: mu={learned_mu.shape}, sigma={learned_sigma.shape}, labels={y.shape}")

    # save remainder latents that are fewer than 10000 images
    # if len(mu) > 0:
    #     mu = torch.cat(mu, dim=0)
    #     sigma = torch.cat(sigma, dim=0)
    #     labels = torch.cat(labels, dim=0)
    #     save_dict = {
    #         'mu': mu,
    #         'sigma': sigma,
    #         'labels': labels
    #     }
    #     for key in save_dict:
    #         if rank == 0:
    #             print(key, save_dict[key].shape)
    #     save_filename = os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors')
    #     save_file(
    #         save_dict,
    #         save_filename,
    #         metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}', 'device': f'{latents.device}'}
    #     )
    #     if rank == 0:
    #         print(f'Saved {save_filename}')

    # Calculate latents stats
    dist.barrier()
    if rank == 0:
        dataset = ImgLatentDataset(output_dir, latent_norm=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='')
    parser.add_argument("--data_split", type=str, default='imagenet_train')
    parser.add_argument("--output_path", type=str, default="feature_output")
    parser.add_argument("--config", type=str, default="config_details.yaml")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    main(args)