import os
import torch
from safetensors.torch import load_file
import numpy as np
import glob
import pandas as pd

def load_latents_from_batches(directory):
    files = sorted(glob.glob(os.path.join(directory, 'latents_batch*.safetensors')))
    mu_list, sigma_list, label_list = [], [], []
    for f in files:
        data = load_file(f)
        mu_batch = data['mu'].float().cpu().numpy()
        sigma_batch = data['sigma'].float().cpu().numpy() # (64, batch)
        labels_batch = data['labels'].cpu().numpy()
        print("#################")
        print(mu_batch.shape)
        print(sigma_batch.shape)
        print(labels_batch.shape)
        mu_list.append(mu_batch)
        sigma_list.append(sigma_batch)
        label_list.append(labels_batch)
    mu = np.concatenate(mu_list, axis=0)       # (전체 샘플 수, latent_dim)
    sigma = np.concatenate(sigma_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    print("#################")
    print(mu.shape)
    print(sigma.shape)
    print(labels.shape)
    return mu, sigma, labels

output_dir = "/home/ivpl2/ahlee/learnable_latent_DiT/learnable_eps/feature_output/model1_f16d3_vfdinov2/imagenet_train_256"
mu, sigma, labels = load_latents_from_batches(output_dir)

unique_labels = np.unique(labels)
summary = []
for label in unique_labels:
    inds = labels == label
    summary.append({
        'label': label,
        'mu_mean': mu[inds].mean(axis=0),
        'mu_std': mu[inds].std(axis=0),
        'sigma_mean': sigma[inds].mean(axis=0),
        'sigma_std': sigma[inds].std(axis=0),
        'count': inds.sum()
    })
df = pd.DataFrame(summary)
# df.to_csv('label_stats.csv', index=False)
# print(df.head())
