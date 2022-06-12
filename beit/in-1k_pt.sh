OUTPUT_DIR='outputs'
DATA_PATH='../../../logs/train'
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"                                                                       
NNODES=$PADDLE_TRAINERS_NUM      
ADDR=`echo $PADDLE_TRAINERS | awk -F "," '{print $1}'`
RANK=$PADDLE_TRAINER_ID
PORT=8897
TOKENIZER_PATH='d_vae'

python3 -m torch.distributed.launch \
        --nproc_per_node=8 \
        --nnodes=$NNODES \
        --node_rank=$RANK \
        --master_addr=$ADDR \
        --master_port $PORT \
        run_beit_pretraining.py \
        --data_path ${DATA_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --num_mask_patches 75 \
        --model beit_base_patch16_224_8k_vocab --discrete_vae_weight_path ${TOKENIZER_PATH} \
        --batch_size 64 --lr 3e-3 --warmup_epochs 10 --epochs 800 \
        --clip_grad 3.0 --drop_path 0.1 --layer_scale_init_value 0.1 \
        --imagenet_default_mean_and_std >beit_base_pre-train_ep800.log 2>&1 &
