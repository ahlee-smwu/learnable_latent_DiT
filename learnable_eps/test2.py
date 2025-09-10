import torch

file_path = "../pretrained_weight/latents_stats/latents_stats.pt"   # 네 파일 경로
data = torch.load(file_path, map_location="cpu")

mean = data["mean"]   # torch.Tensor
std = data["std"]     # torch.Tensor

print(mean)
print(std)