from safetensors.torch import load_file
from PIL import Image
import torch
import os

# 저장한 safetensors 파일 경로
file_path = "/home/ahlee/learnable_latent_DiT/learnable_eps/output/14th_lightningdit_b2_vae_f16d3_lsun_interp05/lightningdit-b-2-ckpt-7500000-euler-250/000000.safetensors"
# 파일 로드
data = load_file(file_path)

# 저장된 키 확인
print("Keys:", list(data.keys()))
print("z raw shape:", data["z"].shape)
# print("z_org raw shape:", data["z_org"].shape)
print("output raw shape:", data["output"].shape)

def squeeze_or_fix(t: torch.Tensor):
    if t.shape[0] == 0:
        # batch가 0인 경우: 강제로 [C, H, W]
        return t.reshape(t.shape[1], t.shape[2], t.shape[3])
    else:
        # batch가 1 이상인 경우: 첫 번째 배치만 사용
        return t[0]

for key, tensor in data.items():
    t = tensor.float()  # 안전하게 float로 변환

    # 채널별 통계 (B, H, W 차원 제거)
    channel_mean = t.mean(dim=(0, 2, 3))  # shape: [C]
    channel_max = t.max(dim=0)[0].max(dim=1)[0].max(dim=1)[0]  # shape: [C] 직접 max 연산
    channel_min = t.min(dim=0)[0].min(dim=1)[0].min(dim=1)[0]  # shape: [C]

    print(f"[{key}] shape={tuple(t.shape)}")
    for i, c in enumerate(["R", "G", "B"]):
        print(f"  {c} channel → mean: {channel_mean[i]:.10f}, max: {channel_max[i]:.10f}, min: {channel_min[i]:.10f}")

# --- 2. z와 output 읽기 ---
z = squeeze_or_fix(data["z"])
# z_org = squeeze_or_fix(data["z_org"])
output = squeeze_or_fix(data["output"])

# --- 3. z 정규화 (-max~+max → 0~1) ---
max_abs = z.abs().max()
z_norm = (z + max_abs) / (2 * max_abs)
z_img = (z_norm * 255).permute(1, 2, 0).to(torch.uint8)  # [H, W, C]

# max_abs = z_org.abs().max()
# z_norm = (z_org + max_abs) / (2 * max_abs)
# z_org_img = (z_norm * 255).permute(1, 2, 0).to(torch.uint8)  # [H, W, C]

# -1~1 -> 정규화
# z_img = ((z + 1) / 2 * 255).permute(1, 2, 0).to(torch.uint8)
# max_abs = z_org.abs().max()
# z_norm = (z_org + max_abs) / (2 * max_abs)
# z_org_img = (z_norm * 255).permute(1, 2, 0).to(torch.uint8)  # [H, W, C]

# --- 4. output 정규화 (min-max → 0~1) ---
min_val = output.min()
max_val = output.max()
output_norm = (output - min_val) / (max_val - min_val)
output_img = (output_norm * 255).permute(1, 2, 0).to(torch.uint8)

# --- 5. PIL 이미지로 저장 ---
save_dir = "visualized_images"
os.makedirs(save_dir, exist_ok=True)
Image.fromarray(z_img.cpu().numpy()).save(os.path.join(save_dir, "13th_dit_z.png"))
# Image.fromarray(z_org_img.cpu().numpy()).save(os.path.join(save_dir, "13th_dit_z_org.png"))
Image.fromarray(output_img.cpu().numpy()).save(os.path.join(save_dir, "13th_dit_output.png"))

print(f"Images saved in folder: {save_dir}")

'''
# --- 1. safetensor 파일 로드 ---
file_path = "feature_output/model1_f16d32_vfdinov2_add_layer/lsun_train_128/latents_rank00_batch000001.safetensors" # VA-VAE z
data = load_file(file_path)

mu = data["mu"]       # [B, C, H, W] 혹은 [B, H, W, C]
sigma = data["sigma"]

# --- 3. batch 차원 확인 및 변환 (HWC → CHW 필요시) ---
# 만약 mu.shape = [B, H, W, C]라면 permute
if mu.shape[-1] == 3 and mu.ndim == 4:  # 마지막 채널이 3이면 HWC
    mu = mu.permute(0, 3, 1, 2)        # [B, C, H, W]
    sigma = sigma.permute(0, 3, 1, 2)

# --- 4. latent z 샘플링 ---
epsilon = torch.randn_like(mu)
z = mu + sigma * epsilon

# --- 5. batch별 min-max 정규화 및 이미지 저장 ---
for i in range(z.shape[0]):
    img = z[i]  # [C, H, W]

    # min-max 정규화
    img_min = img.min()
    img_max = img.max()
    img_norm = (img - img_min) / (img_max - img_min)

    # [C, H, W] → [H, W, C], 0~255, uint8
    img_uint8 = (img_norm * 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()

    # PIL 이미지로 저장
    # Image.fromarray(img_uint8).save(os.path.join(save_dir, f"vae_z_{i:03d}.png"))

print(f"Saved {z.shape[0]} images in folder: {save_dir}")


# 원본 latent channel-wise mean/std
latent_stats_cache_file = os.path.join('/home/ivpl-d26/ahlee/pycharm/winbuekbueq/research/generative_AI/leanable_latent_DiT/pretrained_weight/latents_stats/latents_stats.pt')
latent_stats = torch.load(latent_stats_cache_file)
print("mean",latent_stats['mean']) # torch.Size([1, 32, 1, 1]) -1~1 정도의 값
print("std",latent_stats['std']) # torch.Size([1, 32, 1, 1]) 3~4 정도의 값
'''