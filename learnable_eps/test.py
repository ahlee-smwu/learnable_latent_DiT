import pandas as pd
import torch
import re
import numpy as np


csv_path = "feature_output/model1_f16d3_vfdinov2/imagenet_train_256/label_stats.csv"
output_csv_path = "feature_output/model1_f16d3_vfdinov2/imagenet_train_256/label_stats_with_kl.csv"


def str_to_tensor(s):
    """
    CSV 문자열 -> torch tensor (16x16x3)
    """
    # 불필요한 대괄호 제거
    s = s.replace('[', '').replace(']', '')
    # 공백으로 숫자 분리 후 float로 변환
    arr = np.fromstring(s, sep=' ', dtype=np.float32)
    return torch.tensor(arr.reshape(16, 16, 3), dtype=torch.float32)


def kl_div_map(mu, sigma):
    """
    mu, sigma: tensor (16x16x3)
    return: tensor KL map (16x16x3)
    KL divergence between N(mu, sigma^2) and N(0,1)
    """
    return 0.5 * (sigma ** 2 + mu ** 2 - 1 - torch.log(sigma ** 2 + 1e-8))


# CSV 읽기
df = pd.read_csv(csv_path)

kl_org_mean_list = []
kl_org_map_str_list = []
kl_0mu_mean_list = []
kl_0mu_map_str_list = []

for _, row in df.iterrows():
    label = row['label']
    mu = str_to_tensor(row['mu_mean'])
    sigma = str_to_tensor(row['sigma_mean'])

    kl_org_map = kl_div_map(mu, sigma)
    kl_0mu_map = kl_div_map(torch.zeros_like(mu), sigma)

    kl_org_mean_list.append(kl_org_map.mean().item())
    kl_org_map_str_list.append(kl_org_map.tolist())
    kl_0mu_mean_list.append(kl_0mu_map.mean().item())
    kl_0mu_map_str_list.append(kl_0mu_map.tolist())

# CSV에 KL 평균값 및 맵 추가
df['kl_org_mean'] = kl_org_mean_list
df['kl_org_map'] = kl_org_map_str_list
df['kl_0mu_mean'] = kl_0mu_mean_list
df['kl_0mu_map'] = kl_0mu_map_str_list
df.to_csv(output_csv_path, index=False)
print(f"CSV with KL map and mean saved to {output_csv_path}")
