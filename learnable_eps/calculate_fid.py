import os
import shutil
from PIL import Image
from cleanfid import fid

gen_dir = "/home/ahlee/learnable_latent_DiT/learnable_eps/output/14th_lightningdit_b2_vae_f16d3_lsun_interp05/lightningdit-b-2-ckpt-1762000-euler-5/"
ref_npz = "/home/ahlee/fid_npz/church_fid.npz"
resize = 128
tmp_dir = "tmp_resized"

# 4️⃣ 임시 폴더 삭제
shutil.rmtree(tmp_dir)

# 1️⃣ 임시 폴더 생성
# os.makedirs(tmp_dir, exist_ok=True)
#
# # 2️⃣ 이미지 리사이즈 후 저장
# for fname in os.listdir(gen_dir):
#     if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
#         continue
#     img = Image.open(os.path.join(gen_dir, fname)).convert('RGB')
#     img = img.resize((resize, resize), Image.LANCZOS)
#     img.save(os.path.join(tmp_dir, fname))
# print("resize done")

# 3️⃣ FID 계산
fid_score = fid.compute_fid(
    fdir1=gen_dir,
    fdir2=ref_npz,
    mode="clean",
    model_name="inception_v3"
)

print("FID:", fid_score)


