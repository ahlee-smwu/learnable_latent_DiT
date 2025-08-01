config_path=$CONFIG_PATH

torchrun --nproc_per_node=1 \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    vavae/train.py \
    --base $config_path \
    --train