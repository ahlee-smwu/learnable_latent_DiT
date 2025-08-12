import torch

ckpt_path = "/home/ahlee/research/gen_AI/leanable_latent_DiT/vavae/logs/2025-08-10T08-21-40_f16d32_vfdinov2/checkpoints/model-step=12.ckpt"
ckpt = torch.load(ckpt_path, map_location='cpu')  # CPU로 로드해도 됨

# PyTorch Lightning 체크포인트는 보통 'state_dict' 키 아래에 저장됨
state_dict = ckpt['state_dict']

print("체크포인트에 저장된 레이어(키) 목록:")
for k in state_dict.keys():
    print(k)
