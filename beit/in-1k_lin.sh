GLOG_v=0 python3.7 -m torch.distributed.launch \
    --nproc_per_node=8 \
    run_linear_eval.py \                                                                                                                                                                                    
    --model beit_base_patch16_224 \
    --pretrained_weights beit_base_ep800.pt \
    --data_path ../../../logs \
    --lr 4e-3 \
    --output_dir output >beit_base_ep800lin.log 2>&1 & 
