#!/bin/bash
MODEL_NAME=$1
#MODEL_NAME='/home/chensh/hub/model/llama-3-8b'
OUTPUT_PATH='outputs/teal/Llama-3-8B'
MODEL_TYPE=Llama-3-8B
SPARSE_MODE='teal'
mask_by='topk'

# lm-eval harness
python eval.py --base_model $MODEL_NAME --save_path $OUTPUT_PATH --sparsity 0 --sparse_mode $SPARSE_MODE --mask_by $mask_by