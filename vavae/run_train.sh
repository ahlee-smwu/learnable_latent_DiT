config_path=$CONFIG_PATH

CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    main.py \
    --base $config_path \
    --train