from PIL import Image
import numpy as np

img = Image.open("learnable_eps2/output/1st_lightningdit_xl_vavae_f16d32_kmeans30/lightningdit-xl-1-ckpt-0054000-euler-20/class_0/cluster_0/000158.png")
arr = np.array(img)

print("PIL mode:", img.mode)
print("dtype:", arr.dtype)
print("min/max:", arr.min(), arr.max())
print("shape:", arr.shape)
