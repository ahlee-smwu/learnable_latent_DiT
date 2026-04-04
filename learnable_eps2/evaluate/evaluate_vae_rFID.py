import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from PIL import Image  # numpy → PIL 변환용
import argparse
import os
from datetime import datetime
from tokenizer.vavae import VA_VAE
from cleanfid import fid

def main(args):
    """
    Run VA_VAE on dataset and save RECONSTRUCTED images (exactly 5000) for FID evaluation.
    """
    assert torch.cuda.is_available(), "Requires GPU."

    '''
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
        print("Running in local mode.")
        rank = 0
        device = 0
        world_size = 1
        seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup output folder:
    output_dir = os.path.join(args.output_path, os.path.splitext(os.path.basename(args.config))[0],
                              f'{args.data_split}_{args.image_size}')
    recon_dir = os.path.join(output_dir, 'recon_images')
    if rank == 0:
        os.makedirs(recon_dir, exist_ok=True)
        print(f"Recon images → {recon_dir}")

    # Create model:
    tokenizer = VA_VAE(args.config)
    tokenizer.model.to(device).eval()

    # Setup data:
    datasets = [
        ImageFolder(args.data_path, transform=tokenizer.img_transform(p_hflip=0.0)),
        ImageFolder(args.data_path, transform=tokenizer.img_transform(p_hflip=1.0))
    ]
    samplers = [
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed)
        for dataset in datasets
    ]
    loaders = [
        DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                   pin_memory=True, drop_last=False)
        for dataset, sampler in zip(datasets, samplers)
    ]
    total_data = len(loaders[0].dataset)
    if rank == 0:
        print(f"Total images: {total_data}")

    # Processing loop
    run_images = 0
    global_idx = rank * total_data // world_size
    saved_count = 0
    TARGET_COUNT = args.fid_num

    for batch_idx, batch_data in enumerate(zip(*loaders)):
        run_images += batch_data[0][0].shape[0]
        if run_images % 100 == 0 and rank == 0:
            print(f"Processed {run_images}/{total_data}")

        for loader_idx, data in enumerate(batch_data):
            x = data[0].cuda()
            y = data[1]

            # Encode → Decode
            z = tokenizer.encode_images(x).detach()
            with torch.no_grad():
                recon_np = tokenizer.decode_to_images(z)  # (N,H,W,3) uint8 numpy

            if batch_idx == 0 and loader_idx == 0 and rank == 0:
                print(f"recon_np: {recon_np.shape}, {recon_np.dtype}, range [{recon_np.min()},{recon_np.max()}]")

            # 즉시 저장
            for i in range(recon_np.shape[0]):
                pil_img = Image.fromarray(recon_np[i])
                img_idx = global_idx + i + 11008
                pil_img.save(os.path.join(recon_dir, f'recon_{img_idx:08d}.png'))

            global_idx += x.shape[0]
            saved_count += x.shape[0]

            # 5000개 도달 시 종료
            if saved_count >= TARGET_COUNT:
                if rank == 0:
                    print(f"✅ Saved {TARGET_COUNT} recon images to {recon_dir}")
                return

    if rank == 0:
        print(f"Rank {rank}: Finished early with {saved_count} images")
    '''

    ## FID
    recon_dir = '/mnt/SSD_raid1/lsun/church_outdoor_train_recon/model1_f16d32/lsun_train_256/recon_images'
    fid_score = fid.compute_fid(
        fdir1=args.data_path,  # /mnt/SSD_raid1/lsun/church_outdoor_train
        fdir2=recon_dir,  # .../recon_images/
        dataset_name="lsun_church",
        dataset_res=256,
        mode="clean",
        dataset_split="train",  # 또는 val
        num_workers=8,
        batch_size=64,
        verbose=1
    )
    print(f"[FID]: {fid_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/mnt/SSD_raid1/lsun/church_outdoor_train")
    parser.add_argument("--data_split", type=str, default="lsun_train")
    parser.add_argument("--output_path", type=str, default="/mnt/SSD_raid1/lsun/church_outdoor_train_recon")
    parser.add_argument("--config", type=str, default="model1_f16d32.yaml")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--fid_num", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    main(args)
