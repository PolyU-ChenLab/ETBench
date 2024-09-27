#!/bin/bash

set -e

stage1_path=${1:-"work_dirs/etchat-stage-1"}
stage2_path="work_dirs/etchat-stage-2"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="./:$PYTHONPATH"

torchrun --nproc_per_node 8 etchat/train/train.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $stage1_path \
    --language_model phi3 \
    --conv_type phi3 \
    --fast_tokenizer True \
    --vision_tower eva_vit \
    --vision_processor clip_center_224 \
    --vision_output_layer -2 \
    --vision_output_token patch \
    --mm_projector qformer \
    --anno_path data/llamavid/llava_v1_5_mix665k_with_video_chatgpt.json \
    --image_path data/llava_instruct \
    --video_path data/video_chatgpt \
    --image_pad_to_square True \
    --group_by_modality True \
    --fps 1 \
    --lora_enable True \
    --lora_lr 2e-4 \
    --tuning_mode qformer \
    --use_matching False \
    --use_time_tag False \
    --bi_attention False \
    --alpha 2.0 \
    --max_video_len 350 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --output_dir $stage2_path \
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
