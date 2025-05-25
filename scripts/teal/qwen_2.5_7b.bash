#!/bin/bash
MODEL_NAME=$1
#MODEL_NAME='/home/chensh/hub/model/qwen2.5-7b'
OUTPUT_PATH='outputs/teal/Qwen-2.5-7B'
MODEL_TYPE=Qwen-2.5-7B
SPARSE_MODE='teal'

# compute sparsity for each layer
python teal/grab_acts.py --model_name $MODEL_NAME --output_path $OUTPUT_PATH --sparse_mode $SPARSE_MODE
python teal/greedyopt.py --model_name $MODEL_NAME --output_path $OUTPUT_PATH --sparse_mode $SPARSE_MODE --model_type $MODEL_TYPE

# topk-based gate
mask_by='topk'
for sparsity in 0.25 0.4 0.5 0.65
do
    # lm-eval harness
    python eval.py --base_model $MODEL_NAME --save_path $OUTPUT_PATH --sparsity $sparsity --sparse_mode $SPARSE_MODE --mask_by $mask_by --greedy
done