#!/bin/bash

WORKDIR=/home/ahlee/learnable_latent_DiT/learnable_eps2
LOGDIR=$WORKDIR/evaluate_fid_logs

mkdir -p $LOGDIR

cd $WORKDIR

echo "Run1 start: $(date)"
CUDA_VISIBLE_DEVICES=1 python evaluate_fid.py \
    --gen_base_dir output/5th_lightningdit_xl_vavae_f16d32_gmm30_use_weight/lightningdit-xl-1-ckpt-0018000-euler-20/class_0 \
    > $LOGDIR/5th_0018000.txt 2>&1

echo "Run2 start: $(date)"
CUDA_VISIBLE_DEVICES=1 python evaluate_fid.py \
    --gen_base_dir output/5th_lightningdit_xl_vavae_f16d32_gmm30_use_weight/lightningdit-xl-1-ckpt-0024000-euler-20/class_0 \
    > $LOGDIR/5th_0024000.txt 2>&1

echo "Run3 start: $(date)"
CUDA_VISIBLE_DEVICES=1 python evaluate_fid.py \
    --gen_base_dir output/5th_lightningdit_xl_vavae_f16d32_gmm30_use_weight/lightningdit-xl-1-ckpt-0033000-euler-20/class_0 \
    > $LOGDIR/5th_0033000.txt 2>&1

echo "Run4 start: $(date)"
CUDA_VISIBLE_DEVICES=1 python evaluate_fid.py \
    --gen_base_dir output/5th_lightningdit_xl_vavae_f16d32_gmm30_use_weight/lightningdit-xl-1-ckpt-0048000-euler-20/class_0 \
    > $LOGDIR/5th_0048000.txt 2>&1

echo "Run5 start: $(date)"
CUDA_VISIBLE_DEVICES=1 python evaluate_fid.py \
    --gen_base_dir output/5th_lightningdit_xl_vavae_f16d32_gmm30_use_weight/lightningdit-xl-1-ckpt-0063000-euler-20/class_0 \
    > $LOGDIR/5th_0063000.txt 2>&1

echo "Run6 start: $(date)"
CUDA_VISIBLE_DEVICES=1 python evaluate_fid.py \
    --gen_base_dir output/5th_lightningdit_xl_vavae_f16d32_gmm30_use_weight/lightningdit-xl-1-ckpt-0078000-euler-20/class_0 \
    > $LOGDIR/5th_0078000.txt 2>&1