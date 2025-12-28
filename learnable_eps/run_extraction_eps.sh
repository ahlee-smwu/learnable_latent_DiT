CONFIG_PATH=$1

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-1235}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
PRECISION=${PRECISION:-bf16}

accelerate launch \
    --main_process_ip 127.0.0.1 \
    --main_process_port 1237 \
    --machine_rank 0 \
    --num_processes  1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    extract_features_eps.py \
    --config model1_f16d32_vfdinov2_add_layer.yaml