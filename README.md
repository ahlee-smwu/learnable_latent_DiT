# RAPID: Robust Adaptive Prior Integration for Diffusion

#### Ah-Hyeon Lee and Byung-Gyu Kim

Intelligent Vision Processing Lab. (IVPL), Sookmyung Women's University, Seoul, Republic of Korea

----------------------------
#### This repository is the official PyTorch implementation of RAPID (Thesis for the Degree of Master by Ah-Hyeon Lee, Department of IT Engineering, Sookmyung Women's University, June 2026).
#### Related paper: Ah-Hyeon Lee, Hyeon-Su Lee, Byung-Gyu Kim*, "Adaptive Prior Diffusion: Dataset-Aware GMM Prior for High-Fidelity Image Generation," KMMS Spring Conference, vol. 29, no. 1, pp. 5-9, May 2026 (**Best Paper Award**).
----------------------------

## Summary of paper

#### Abstract

> _Diffusion generative models produce high-fidelity outputs but suffer from slow sampling due to the need to traverse the long and curved trajectory from uninformative Gaussian noise to data distributions. We propose **RAPID (Robust Adaptive Prior Integration for Diffusion)**, a new few-step generation framework that addresses this from the distinct perspective of prior design rather than model architecture or distillation. RAPID replaces the standard Gaussian prior with a GMM-based prior derived from latent feature clustering, injecting structural information in a coarse-to-fine manner. This prior is closer to the target distribution, therefore makes a shorter trajectory.
Unlike prior work that treats prior design, schedule compatibility, and trajectory efficiency as separate problems, RAPID resolves all three jointly. This is because substituting the prior alone would disrupt the denoising dynamics. Accordingly, the proposed thesis introduces two corresponding methods. First, a **data-aware prior** processes GMM cluster statistics into a diffusion-compatible noise. High-frequency artifacts in centroids are removed via low-pass filtering, and per-cluster covariance is normalized to restore the stochasticity. Second, a **step-adaptive noise**, analytically derived from SNR matching, controls how GMM influence is blended into the trajectory over time. This ensures the model operates optimally in a familiar signal regime of the diffusion noise schedule throughout training. From this novel approach, RAPID reduces velocity field complexity and enables high-fidelity generation with significantly fewer sampling steps.
Without introducing additional network modules, RAPID reduces the sampling steps by 92% at 20 NFE with a 9.8× throughput improvement, and by 84% at 20×2 NFE (CFG) with a 6.3× throughput improvement. RAPID achieves an FID of 5.40 on ImageNet-1K at 20×2 NFE, an FID of 4.65 on LSUN-Church-Outdoor, and a state-of-the-art Precision of 0.83 on FFHQ at 20 NFE._
>

**Keywords**: Diffusion Models, Adaptive Priors, Efficient AI, Few-step Sampling, Gaussian Mixture Models, Coarse-to-Fine Synthesis, Low-Pass Filter (LPF)

#### Method Overview

RAPID is a **plug-and-play prior replacement**. It adds no network module, no distillation, and no architectural change to the backbone, so NFE remains exactly equal to the number of sampling steps.

<p align="center">
  <img width="900" src="./images/img1.png">
</p>

**(1) GMM Clustering.** Latent features extracted by VA-VAE are compressed to 256 dimensions by PCA, and a per-class GMM with diagonal covariance is fitted. The goal is not hard partitioning but the extraction of coarse structural prototypes.

**(2) Data-Aware Prior.** Cluster centroids carry high-frequency residuals that behave as structured noise and corrupt the velocity field. A low-pass filter is applied to each centroid,

```
LPF(mu_k) = (1 - alpha) * mu_k + alpha * AvgPool(mu_k; r=3)
```

where the smoothing coefficient `alpha` is derived analytically from the spectral cutoff `f_c,min` defined by the -3 dB criterion `R(f) = P_mu(f) / P_data(f) <= 0.5`. Per-cluster covariance is then rescaled by a single global factor `s_global = 1 / sqrt(mean weighted diag variance)` so that the prior satisfies the unit-variance assumption of flow matching while preserving inter-cluster energy differences.

```
x0_gmm = LPF(mu_k) + s_global * sigma_k * eps
```

**(3) Step-Adaptive Noise.** GMM structure helps at early denoising steps and becomes a conflicting bias at later steps. The prior is therefore blended with standard Gaussian noise under a variance-preserving constraint,

```
x0_blend(t) = w(t) * x0_gmm + sqrt(1 - w(t)^2) * eps
x_t         = t * x1 + (1 - t) * x0_blend(t)
u_t         = x1 - x0_blend(t)
```

with the schedule `w(t) = q0 * exp(-alpha * t)`, `q0 = 0.5`, `alpha = 1.0`. `q0` follows from SNR matching at the structural formation timestep `t_target ~ 0.36`, and the exponential form is the unique decay with a constant relative rate, which keeps the velocity field regular across the whole trajectory.

#### Experimental Results

**ImageNet-1K 256x256.** `×2` denotes doubled NFE due to CFG. Competing methods are reported near 400k training iterations (80 epochs).

| Model | NFE | FID ↓ | Precision ↑ | Recall ↑ | Density ↑ | Coverage ↑ |
|:------|:---:|:-----:|:-----------:|:--------:|:---------:|:----------:|
| LightningDiT | 10 | 26.24 | 0.54 | 0.60 | 0.44 | 0.53 |
| LightningDiT | 20 | 10.25 | 0.71 | 0.65 | 0.72 | 0.79 |
| LightningDiT | 40 | 6.13 | 0.76 | 0.68 | 0.89 | 0.89 |
| CRS-vx | 50 | 9.34 | 0.64 | 0.63 | – | – |
| **RAPID** | 10 | 8.35 | 0.78 | 0.65 | 0.91 | 0.88 |
| **RAPID** | 20 | 6.55 | 0.79 | 0.66 | 0.98 | 0.90 |
| **RAPID** | **20×2** | **5.40** | **0.79** | 0.67 | 0.97 | **0.91** |
| **RAPID** | 40 | 6.27 | 0.80 | 0.66 | **0.99** | 0.90 |

**LSUN-Church-Outdoor 256x256.**

| Model | NFE | FID ↓ | Precision ↑ | Recall ↑ | Density ↑ | Coverage ↑ |
|:------|:---:|:-----:|:-----------:|:--------:|:---------:|:----------:|
| LightningDiT | 250 | 3.93 | 0.80 | 0.60 | 1.06 | 0.91 |
| LDM-8 | 200 | 4.02 | 0.64 | 0.52 | – | – |
| LightningDiT | 10 | 25.24 | 0.47 | 0.20 | 0.27 | 0.36 |
| LightningDiT | 20 | 10.30 | 0.69 | 0.42 | 0.62 | 0.71 |
| LightningDiT | 40 | 4.60 | 0.79 | 0.58 | 0.97 | 0.89 |
| **RAPID** | 10 | 6.14 | 0.79 | 0.52 | 0.99 | 0.83 |
| **RAPID** | **20** | **4.65** | 0.78 | 0.56 | **1.00** | 0.85 |
| **RAPID** | 40 | 4.53 | 0.77 | 0.58 | 0.96 | 0.85 |

**FFHQ 256x256.**

| Model | NFE | FID ↓ | Precision ↑ | Recall ↑ | Density ↑ | Coverage ↑ |
|:------|:---:|:-----:|:-----------:|:--------:|:---------:|:----------:|
| GPS | 250 | 4.48 | – | 0.65 | – | – |
| LFM | 88 | 4.55 | – | 0.48 | – | – |
| StyleNAT | 1 | 2.05 | 0.68 | 0.51 | – | – |
| **RAPID** | 10 | 15.20 | 0.80 | 0.50 | 1.06 | 0.65 |
| **RAPID** | **20** | 9.68 | **0.83** | 0.55 | **1.27** | 0.75 |
| **RAPID** | 40 | 8.97 | 0.82 | 0.55 | 1.25 | 0.76 |

**Inference efficiency.** Measured on a single NVIDIA RTX A6000, ODE Euler sampler, sampling batch size 256.

| Model | NFE ↓ | Prior gen (s) ↓ | Model sampling (s) ↓ | Throughput (imgs/s) ↑ |
|:------|:-----:|:---------------:|:--------------------:|:---------------------:|
| LightningDiT | 250 | 0.000 | 403.769 | 0.634 |
| **RAPID** | **20** | 0.026 | **41.493** | **6.234** |
| **RAPID** | 40 | 0.026 | 64.202 | 3.986 |

GMM prior construction takes 0.026 s, which is 0.06% of total inference time. The speedup comes entirely from the reduction in model evaluations, not from architectural change.

<p align="center">
  <img width="800" src="./images/img2.png">
</p>

----------------------------
## Getting Started

#### Dependencies and Installation

- Anaconda3
- Python == 3.10.12
- PyTorch (NVIDIA GPU + CUDA)

Run in `./`

```bash
conda env create -f rapid.yml
conda activate rapid
pip install -r requirements.txt
```

Additional packages used for evaluation:

```bash
pip install clean-fid prdc scikit-learn
```

#### Repository Structure

```
RAPID
├─adaptive-prior                  # all commands are executed from this directory
│  │  extract_features.py         # VA-VAE latent extraction
│  │  train.py                    # RAPID training
│  │  inference.py                # RAPID sampling
│  │  trajectory.py               # denoising trajectory / curvature analysis
│  │  model1_f16d32.yaml          # model 1: VA-VAE tokenizer config
│  │  model2_xl_vavae_f16d32.yaml # model 2: LightningDiT-XL/1 diffusion config
│  │
│  ├─GMM                          # step 2. GMM clustering
│  │      fit_gmm.py              # PCA(256) + diagonal-covariance GMM fitting
│  │      evaluate_cluster.py     # Silhouette, Neff/K, posterior entropy
│  │      view_cluster.py         # cluster centroid visualization
│  │      add_variable_to_pkl.py  # append derived statistics to gmm_clusters.pkl
│  │
│  ├─data-aware-prior             # step 3. data-aware prior
│  │      fc_min.py               # power-spectrum analysis -> f_c,min -> alpha*
│  │
│  ├─step-adaptive-noise          # step 4. step-adaptive noise
│  │      gmm_rho.py                       # rho (prior-data correlation) estimation
│  │      velocity_align_for_t-target.py   # structural formation timestep t_target
│  │      fit_gmm_traj_xt.py               # GMM fitting on intermediate x_t
│  │      trajectory_analysis.py           # blending schedule analysis
│  │      inference_t_start.py             # truncated-start sampling
│  │
│  ├─evaluate                     # step 7. evaluation
│  │  │  evaluate_score.py        # FID / Precision / Recall / Density / Coverage
│  │  │  analysis_fid.py          # overall and per-cluster FID
│  │  │  evaluate_is.py           # Inception Score
│  │  │  evaluate_lpips.py        # LPIPS
│  │  │  evaluate_vae_rFID.py     # tokenizer reconstruction FID
│  │  │  trajectory_distance.py   # trajectory length / distance
│  │  │  analysis_velocity_lanes.py
│  │  │  analysis_cluster_w_GT.py
│  │  │  analysis_loss.py
│  │  └─figure_nfe                # NFE vs FID / Precision plots
│  │
│  ├─imgnet-100                   # ImageNet-100 subset remapping utilities
│  ├─output                       # generated samples
│  └─tensorboard_logs             # training logs
│
├─datasets                        # latent dataset loaders
├─models                          # LightningDiT-XL/1 backbone
├─transport                       # flow matching core (RAPID prior injection)
│      path.py                    # plan_gmm_adaptive: w(t) schedule and x_t, u_t
│      transport.py               # training loss, LPF, sampling
│      integrators.py             # Euler / Heun / dopri5
├─tokenizer                       # VA-VAE
├─tools                           # FID utilities, npz export, latent visualization
├─vavae                           # VA-VAE training code (from baseline)
└─docs                            # baseline tutorial
```

The RAPID contribution is contained in `adaptive-prior/` and in the GMM prior handling inside `transport/path.py` and `transport/transport.py`. Everything else follows the LightningDiT baseline.

#### Dataset Preparation

We used ImageNet-1K, LSUN-Church-Outdoor, and FFHQ at 256x256 resolution.

| Dataset | Images | Classes | Download |
|:--------|:------:|:-------:|:---------|
| ImageNet-1K | 1,281,167 train / 50,000 val | 1,000 | [official website](https://www.image-net.org/) |
| LSUN-Church-Outdoor | 126,227 train / 300 val | 1 | [official website](https://www.yf.io/p/lsun) |
| FFHQ | 70,000 | 1 | [bitmind/ffhq-256](https://huggingface.co/datasets/bitmind/ffhq-256) |

Place each dataset in the standard `ImageFolder` layout and set the path in `data.data_path` of `model2_xl_vavae_f16d32.yaml`.

#### Model Zoo

RAPID uses the pre-trained VA-VAE tokenizer from the baseline. Diffusion models are trained from scratch.

| Component | Weight |
|:----------|:-------|
| Tokenizer (VA-VAE f16d32) | [vavae-imagenet256-f16d32-dinov2](https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/blob/main/vavae-imagenet256-f16d32-dinov2.pt) |
| Latent statistics | [latents_stats.pt](https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/blob/main/latents_stats.pt) |
| Baseline reference (LightningDiT-XL) | [lightningdit-xl-imagenet256-800ep](https://huggingface.co/hustvl/lightningdit-xl-imagenet256-800ep/blob/main/lightningdit-xl-imagenet256-800ep.pt) |

Save the tokenizer weight and set its path in `model1_f16d32.yaml`.

----------------------------
## Usage

**All commands are executed from the `adaptive-prior/` directory.**

```bash
cd adaptive-prior
```

### Step 1. Latent extraction

Encode the dataset into VA-VAE latents. For 256x256 RGB images the latent shape is 16 x 16 x 32.

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,3 \
NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 \
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1237 \
    --machine_rank 0 \
    --num_processes 2 \
    --num_machines 1 \
    --mixed_precision bf16 \
    extract_features.py --config model1_f16d32.yaml
```

### Step 2. GMM clustering

Fit a per-class GMM on the extracted latents. PCA compression to 256 dimensions is applied before fitting, and a diagonal covariance structure is used.

```bash
python GMM/fit_gmm.py
```

This produces `gmm_clusters.pkl` with the following entries, consumed directly by `train.py` and `inference.py`.

| Key | Shape | Description |
|:----|:------|:------------|
| `means` | {class: (K, D)} | cluster centroids in latent space (D = 8192) |
| `covs` | {class: (K, d)} | diagonal variance in PCA space (d = 256) |
| `weights` | {class: (K,)} | mixture weights |
| `means_pca` | {class: (K, d)} | centroids in PCA space |
| `diag_var` | {class: (K, D)} | diagonal variance projected back to latent space |
| `pca_components` | {class: {components, mean}} | PCA basis |
| `labels` | {class: (N,)} | hard assignments (optional) |

Cluster quality is checked with:

```bash
python GMM/evaluate_cluster.py
python GMM/view_cluster.py
```

The number of clusters `k` per class used in the thesis:

| Dataset | Clusters per class | Classes | Total clusters |
|:--------|:------------------:|:-------:|:--------------:|
| ImageNet-1K | 10 | 1,000 | 10,000 |
| LSUN-Church-Outdoor | 30 | 1 | 30 |
| FFHQ | 20 | 1 | 20 |

Reported clustering metrics. `Neff/K` above 0.75 in every case confirms balanced component utilisation with no dead components. Low silhouette is expected and acceptable, since the objective is structural abstraction rather than hard partitioning.

| Dataset | Silhouette ↑ | Neff/K ↑ | Entropy ↓ | rho ↑ |
|:--------|:------------:|:--------:|:---------:|:-----:|
| ImageNet-1K | 0.0182 | 0.908 | 0.030 | 0.7864 |
| LSUN-Church-Outdoor | 0.0105 | 0.767 | 0.244 | 0.8084 |
| FFHQ | 0.0053 | 0.826 | 0.223 | 0.7396 |

### Step 3. Data-aware prior

Derive the dataset-specific low-pass filter coefficient from the latent power spectrum. The script computes the spectral ratio `R(f) = P_mu(f) / P_data(f)`, locates the -3 dB cutoff `f_c,min`, and converts it into the smoothing coefficient `alpha*`.

```bash
python data-aware-prior/fc_min.py
```

The global covariance scale `s_global` is computed once over the full GMM at training start by `precompute_sigma_scale()` in `train.py`, so no separate script is required.

Dataset-specific values used in the thesis:

| Dataset | alpha* | s_global |
|:--------|:------:|:--------:|
| ImageNet-1K | 1.000 | 1.944 |
| LSUN-Church-Outdoor | 0.886 | 2.253 |
| FFHQ | 0.728 | 2.281 |

As dataset diversity increases, `alpha*` grows to suppress a wider high-frequency band, while `s_global` decreases because the broader covariance of diverse clusters already carries sufficient stochastic spread.

### Step 4. Step-adaptive noise

The blending schedule `w(t) = q0 * exp(-alpha * t)` with `q0 = 0.5`, `alpha = 1.0` is **dataset-agnostic** and requires no per-benchmark tuning. The scripts below reproduce its derivation.

```bash
python step-adaptive-noise/gmm_rho.py                     # rho: prior-data correlation
python step-adaptive-noise/velocity_align_for_t-target.py # t_target ~ 0.36
python step-adaptive-noise/trajectory_analysis.py         # schedule comparison
```

The schedule is implemented in `transport/path.py::plan_gmm_adaptive`.

### Step 5. Training

RAPID introduces no additional module or distillation. The backbone, optimiser, and schedule are identical to the baseline; only the starting distribution is replaced. Models are trained from scratch with AdamW, learning rate 2e-4, beta2 = 0.95.

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,3 \
NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 \
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1237 \
    --machine_rank 0 \
    --num_processes 2 \
    --num_machines 1 \
    --mixed_precision bf16 \
    train.py --config model2_xl_vavae_f16d32.yaml
```

Training logs are written to `tensorboard_logs/{dataset}/{exp_name}`.

```bash
tensorboard --logdir tensorboard_logs
```

### Step 6. Inference

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,3 \
NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 \
accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1237 \
    --machine_rank 0 \
    --num_processes 2 \
    --num_machines 1 \
    --mixed_precision bf16 \
    inference.py --config model2_xl_vavae_f16d32.yaml
```

Samples are written to

```
output/{exp_name}/{model_type}-ckpt-{step}-{sampling_method}-{num_sampling_steps}[-interval{...}-cfg{...}-shift{...}]/class_{c}/
```

Sampling behaviour is controlled entirely by the `sample` section of the config, so no code change is needed to switch solver or step count.

| Field | Description | Thesis setting |
|:------|:------------|:---------------|
| `mode` | `ODE` or `SDE` | `SDE` |
| `sampling_method` | `euler`, `Heun`, `dopri5` | `euler` |
| `num_sampling_steps` | NFE (equals sampling steps) | 20 |
| `cfg_scale` | classifier-free guidance scale, `>1.0` enables CFG | ImageNet-1K only |
| `cfg_interval_start` | CFG start interval | – |
| `timestep_shift` | timestep shift | – |
| `per_proc_batch_size` | per-process sampling batch | 256 total |
| `fid_num` | number of samples to generate | 50,000 |
| `fid_reference_file` | reference `.npz` for FID | – |

`Heun` consumes 2 NFE per step, so `num_sampling_steps` must be halved to match the NFE budget of `euler`. `dopri5` is adaptive and does not have a fixed NFE. DPM-Solver++ and UniPC are not supported, since they assume a VP-SDE `(alpha_t, sigma_t)` parameterisation, whereas this repository uses velocity prediction with a linear interpolant.

### Step 7. Evaluation

```bash
python evaluate/evaluate_score.py \
    --gen_base_dir output/{exp_name}/{folder_name}/class_0 \
    --real_dir /path/to/real/images

python evaluate/analysis_fid.py \
    --gen_base_dir output/{exp_name}/{folder_name}/class_0 \
    --real_base_dir /path/to/real/clustered/images \
    --real_dir /path/to/real/images

python evaluate/evaluate_is.py
python evaluate/evaluate_lpips.py
```

`evaluate_score.py` reports FID, Precision, Recall, Density, and Coverage using `clean-fid` and `prdc`. `analysis_fid.py` additionally reports per-cluster FID, which is useful for locating collapsed clusters.

NFE curves in the paper are reproduced with:

```bash
python evaluate/figure_nfe/figure-fid-lsun.py
python evaluate/figure_nfe/figure-pre-lsun.py
```

----------------------------
## Configuration

`model2_xl_vavae_f16d32.yaml` is the single entry point for training and inference. RAPID-specific fields are listed first.

```yaml
cluster_type: gmm            # gmm | kmeans

gmm:
  load_dir: /path/to/gmm     # directory containing gmm_clusters.pkl
  use_weight: true           # sample clusters by mixture weight
  reweight_lambda: 0.0       # distance-based loss reweighting (0 disables)

data:
  data_path: /path/to/latents
  valid_path: /path/to/valid_latents
  num_classes: 1             # 1000 for ImageNet-1K
  image_size: 256
  latent_norm: true
  latent_multiplier: 1.0
  fid_reference_file: /path/to/reference.npz

vae:
  downsample_ratio: 16

model:
  model_type: LightningDiT-XL/1
  in_chans: 32
  use_qknorm: false
  use_swiglu: true
  use_rope: true
  use_rmsnorm: true
  wo_shift: false
  use_checkpoint: true

train:
  exp_name: rapid_lsun_gmm30_q0.5_alpha1.0
  output_dir: output
  global_batch_size: 256
  max_steps: 400000
  ckpt_every: 10000
  log_every: 100
  resume: null

optimizer:
  lr: 2e-4
  beta2: 0.95
  max_grad_norm: 1.0

transport:
  path_type: Linear
  prediction: velocity
  loss_weight: null
  train_eps: null
  sample_eps: null
  use_lognorm: true
  use_cosine_loss: true

sample:
  mode: SDE
  sampling_method: euler
  num_sampling_steps: 20
  cfg_scale: 1.0
  fid_num: 50000
  per_proc_batch_size: 128
```

----------------------------
## Ablation Study

**Low-pass filtering** (LSUN-Church-Outdoor, ODE 40 steps). LPF must be applied during training, and the analytically derived cutoff outperforms a heuristic value.

| Configuration | FID ↓ | Precision ↑ | Recall ↑ | Density ↑ | Coverage ↑ |
|:--------------|:-----:|:-----------:|:--------:|:---------:|:----------:|
| w/o LPF | 13.97 | 0.73 | 0.41 | 0.75 | 0.63 |
| Inference-only (alpha = 0.7) | 10.21 | 0.70 | 0.51 | 0.67 | 0.68 |
| Training (alpha = 0.7) | 10.64 | 0.74 | 0.43 | 0.80 | 0.70 |
| **Training (alpha\* = 0.886)** | **10.40** | **0.74** | 0.43 | **0.81** | **0.71** |

**Sigma normalisation** (LSUN-Church-Outdoor, ODE 40 steps). Without normalisation the prior covariance is too narrow, and Coverage drops most sharply.

| sigma norm | FID ↓ | Precision ↑ | Recall ↑ | Density ↑ | Coverage ↑ |
|:-----------|:-----:|:-----------:|:--------:|:---------:|:----------:|
| w/o | 13.70 | 0.73 | 0.44 | 0.73 | 0.62 |
| **w/** | **10.40** | **0.74** | 0.43 | **0.81** | **0.71** |

**Blending schedule** (LSUN-Church-Outdoor). Only the exponential schedule with `alpha = 1.0` maintains a constant relative decay rate across `[0, 1]`.

| f(t) | q0 | alpha | FID ↓ | Precision ↑ | Recall ↑ | Density ↑ | Coverage ↑ |
|:-----|:--:|:-----:|:-----:|:-----------:|:--------:|:---------:|:----------:|
| **exp(-alpha t)** | **0.5** | **1.0** | **7.51** | **0.78** | 0.50 | **0.89** | **0.78** |
| exp(-alpha t) | 0.5 | 1.5 | 10.40 | 0.74 | 0.43 | 0.81 | 0.72 |
| 1 - t | 0.5 | – | 9.80 | 0.64 | 0.58 | 0.59 | 0.72 |
| cos^2(pi t / 2 t_c) | 0.6 | – | 15.63 | 0.66 | 0.42 | 0.57 | 0.58 |
| cos^2(pi t / 2 t_c) | 0.8 | – | 19.52 | 0.58 | 0.59 | 0.44 | 0.52 |

**ODE vs SDE coefficient** (LSUN-Church-Outdoor, 40 steps). `sigma = 0.2` gives the best balance and is the default.

| sigma | FID ↓ | Precision ↑ | Recall ↑ | Density ↑ | Coverage ↑ |
|:------|:-----:|:-----------:|:--------:|:---------:|:----------:|
| ODE | 7.51 | 0.78 | 0.50 | 0.89 | 0.78 |
| 0.1 | 4.61 | **0.79** | 0.57 | **1.02** | **0.86** |
| **0.2** | **4.53** | 0.77 | **0.58** | 0.96 | 0.85 |
| 0.3 | 5.11 | 0.75 | 0.57 | 0.87 | 0.83 |

----------------------------
## Acknowledgement

This repository is built on [LightningDiT / VA-VAE](https://github.com/hustvl/LightningDiT), which is itself built on [DiT](https://github.com/facebookresearch/DiT), [FastDiT](https://github.com/chuanyangjin/fast-DiT), [SiT](https://github.com/willisma/SiT), [LDM](https://github.com/CompVis/latent-diffusion), and [MAR](https://github.com/LTH14/mar). Evaluation uses [clean-fid](https://github.com/GaParmar/clean-fid) and [prdc](https://github.com/clovaai/generative-evaluation-prdc). We thank the authors for releasing their work.

```bash
@inproceedings{yao2025vavae,
  title={Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models},
  author={Yao, Jingfeng and Yang, Bin and Wang, Xinggang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

```bash
@article{yao2024fasterdit,
  title={FasterDiT: Towards faster diffusion transformers training without architecture modification},
  author={Yao, Jingfeng and Wang, Cheng and Liu, Wenyu and Wang, Xinggang},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={56166--56189},
  year={2024}
}
```

----------------------------
## Citation

If you find this work useful, please cite:

```bash
@mastersthesis{lee2026rapid,
  title={RAPID: Robust Adaptive Prior Integration for Diffusion},
  author={Ah-Hyeon Lee and Byung-Gyu Kim},
  school={Sookmyung Women's University},
  year={2026},
  month={June},
  type={Master's thesis}
}
```

----------------------------
## License

The baseline code follows the MIT License of LightningDiT (see `LICENSE-baseline`).
