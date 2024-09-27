#!/bin/bash

set -e

stage2_path=${1:-"work_dirs/etchat-stage-2"}
stage3_path="work_dirs/etchat-stage-3"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="./:$PYTHONPATH"

torchrun --nproc_per_node 8 etchat/train/train.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $stage2_path \
    --language_model phi3 \
    --conv_type phi3 \
    --fast_tokenizer True \
    --vision_tower eva_vit \
    --vision_processor clip_center_224 \
    --vision_output_layer -2 \
    --vision_output_token patch \
    --mm_projector qformer \
    --anno_path data/et_instruct_164k/et_instruct_164k_vid.json \
    --video_path data/et_instruct_164k/videos \
    --fps 1 \
    --lora_enable True \
    --lora_lr 5e-5 \
    --tuning_mode attention \
    --use_matching True \
    --use_time_tag False \
    --bi_attention True \
    --alpha 2.0 \
    --min_video_len 5 \
    --max_video_len 350 \
    --max_num_words 200 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --output_dir $stage3_path \
    --save_full_model True \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --fp16 True \
    --report_to tensorboard
