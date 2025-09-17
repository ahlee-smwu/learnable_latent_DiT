import torch
from safetensors.torch import load_file
import glob

files = glob.glob("feature_output/model1_f16d32_vfdinov2_add_layer/lsun_train_128/*.safetensors")
print("Found files:", len(files))

# mu shape: [16,16,3]
mu_sum = torch.zeros(16,16,3)
mu_sq_sum = torch.zeros(16,16,3)
count = 0

for f in files:
    data = load_file(f)
    mu = data["mu"].squeeze(0)  # [16,16,3]

    mu_sum += mu
    mu_sq_sum += mu ** 2
    count += 1

# 픽셀별 평균과 표준편차 (shape 유지)
mu_mean = mu_sum / count
mu_var = (mu_sq_sum / count) - (mu_mean ** 2)
mu_std = torch.sqrt(mu_var)

print("mu_mean shape:", mu_mean.shape)   # (16, 16, 3)
print("mu_std shape:", mu_std.shape)     # (16, 16, 3)

# --- 채널별 출력 ---
for c in range(3):
    print(f"\n=== Channel {c} ===")
    print("mu_mean:\n", mu_mean[..., c])
    print("mu_std:\n", mu_std[..., c])
