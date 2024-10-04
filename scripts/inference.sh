#!/bin/bash

set -e

stage3_path=${1:-"work_dirs/etchat-stage-3"}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="./:$PYTHONPATH"

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python etchat/eval/infer_etbench.py \
        --anno_path data/etbench/annotations/vid \
        --data_path data/etbench/videos_compressed \
        --pred_path $stage3_path/etbench \
        --model_path $stage3_path \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose &
done

wait
