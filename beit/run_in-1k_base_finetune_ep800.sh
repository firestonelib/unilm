OUTPUT_DIR='outputs'
DATA_PATH='../../../logs/'
TOKENIZER_PATH='d_vae'
FINETUNE_MODEL='beit_base_ep800.pt'
NNODES=$PADDLE_TRAINERS_NUM      
ADDR=`echo $PADDLE_TRAINERS | awk -F "," '{print $1}'`
RANK=$PADDLE_TRAINER_ID
PORT=8899

python3 -m torch.distributed.launch  \
        --nproc_per_node=8  \
        --nnodes=$NNODES \
        --node_rank=$RANK \
        --master_addr=$ADDR \
        --master_port $PORT \
        run_class_finetuning.py \
        --data_path ${DATA_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --model beit_base_patch16_224 \
        --finetune ${FINETUNE_MODEL} \
        --layer_decay 0.65 --drop_path 0.1 \
        --batch_size 128 --lr 4e-3 --warmup_epochs 20 --epochs 100 \
        --nb_classes 1000 --weight_decay 0.05 --mixup 1.0 \
        --no_auto_resume >pd_ep800_ft.log 2>&1 &
