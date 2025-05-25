#!/bin/bash
MODEL_NAME=$1
#MODEL_NAME='/home/chensh/hub/model/phi-4'
OUTPUT_PATH='outputs/teal/Phi-4-14B'
MODEL_TYPE=Phi-4-14B
SPARSE_MODE='teal'
mask_by='topk'

# lm-eval harness
python eval.py --base_model $MODEL_NAME --save_path $OUTPUT_PATH --sparsity 0 --sparse_mode $SPARSE_MODE --mask_by $mask_by