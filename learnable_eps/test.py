import torch
import timm

outputs = {}

def hook_fn(module, input, output):
    outputs[module] = output.shape

hooks = []
# model = timm.create_model("hf-hub:timm/vit_base_patch14_dinov2.lvd142m", pretrained=True, dynamic_img_size=True) # dim: 768
model = timm.create_model("hf-hub:timm/vit_large_patch14_dinov2.lvd142m", pretrained=True, dynamic_img_size=True) # dim: 1024


for name, module in model.named_modules():
    # 관심있는 레이어만 hook 걸기 (예: Conv, Linear, LayerNorm 등)
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LayerNorm)):
        hooks.append(module.register_forward_hook(hook_fn))

# 임의 입력 (예: DINOv2는 일반적으로 224x224 RGB 이미지)
dummy_input = torch.randn(1, 3, 224, 224)

# forward 실행
model(dummy_input)

# hook 제거
for h in hooks:
    h.remove()

# 출력 결과 출력
for module, shape in outputs.items():
    print(f"{module}: output shape {shape}")
