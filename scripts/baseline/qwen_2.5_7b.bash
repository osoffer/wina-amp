#!/bin/bash
MODEL_NAME=$1
#MODEL_NAME='/home/chensh/hub/model/qwen2.5-7b'
OUTPUT_PATH='outputs/teal/Qwen-2.5-7B'
MODEL_TYPE=Qwen-2.5-7B
SPARSE_MODE='teal'
mask_by='topk'

# lm-eval harness
python eval.py --base_model $MODEL_NAME --save_path $OUTPUT_PATH --sparsity 0 --sparse_mode $SPARSE_MODE --mask_by $mask_by