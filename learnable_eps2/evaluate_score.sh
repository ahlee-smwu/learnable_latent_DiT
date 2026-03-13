#!/bin/bash

WORKDIR=/home/ahlee/learnable_latent_DiT/learnable_eps2
LOGDIR=$WORKDIR/evaluate_score_logs
PREFIX=5th
ckpts=(0018000 0024000 0033000 0048000 0063000 0078000)

BASE_DIR=output/${PREFIX}_lightningdit_xl_vavae_f16d32_gmm30_use_weight

mkdir -p "$LOGDIR"
cd "$WORKDIR" || exit 1

for ckpt in "${ckpts[@]}"
do
    echo "${PREFIX} ckpt ${ckpt} start: $(date)"

    CUDA_VISIBLE_DEVICES=1 python -u evaluate_score.py \
        --gen_base_dir ${BASE_DIR}/lightningdit-xl-1-ckpt-${ckpt}-euler-20/class_0 \
        > "$LOGDIR/${PREFIX}_${ckpt}.txt" 2>&1

    sed -i '/%/d' $LOGDIR/${PREFIX}_${ckpt}.txt
done

