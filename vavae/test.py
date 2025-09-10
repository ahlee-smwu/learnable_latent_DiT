import os
import re
import math

ckpt_dir = "logs/2025-09-01T11-08-23_model1_f16d32_vfdinov2_add_layer/checkpoints"
interval = 9860
tolerance = interval // 2   # 허용 오차: 절반(4930) 정도

# step ckpt 모으기
step_ckpts = []
for fname in os.listdir(ckpt_dir):
    if fname.startswith("model-step=") and fname.endswith(".ckpt"):
        match = re.search(r"model-step=(\d+)\.ckpt", fname)
        if match:
            step = int(match.group(1))
            step_ckpts.append((step, fname))

if step_ckpts:
    step_ckpts.sort()
    max_step, max_file = step_ckpts[-1]

    keep_files = set()
    # interval 단위 목표 step 구하기
    for target in range(interval, max_step + interval, interval):
        # target 근처에서 가장 가까운 step 선택
        closest = min(step_ckpts, key=lambda x: abs(x[0] - target))
        if abs(closest[0] - target) <= tolerance:
            keep_files.add(closest[1])

    # 항상 max step은 유지
    keep_files.add(max_file)

    # 삭제 진행
    for step, fname in step_ckpts:
        if fname not in keep_files:
            print("삭제:", fname)
            os.remove(os.path.join(ckpt_dir, fname))

print("정리 완료 ✅ (last.ckpt는 자동 보존됨)")
