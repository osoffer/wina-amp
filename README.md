# 🚀 WINA-AMP: WINA Activation Magnitude Profile optimization

[![Original WINA](https://img.shields.io/badge/Based%20on-Microsoft%2FWINA-blue.svg)](https://github.com/microsoft/wina)
[![Paper](https://img.shields.io/badge/arXiv-2505.19427-red.svg)](https://arxiv.org/abs/2505.19427)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)

*This repository builds on Microsoft's WINA framework (theoretical and implementation foundations) to develop cluster-specific sparse activation profiles for improved LLM inference efficiency.*

## What's WINA
**WINA (Weight-Informed Neuron Activation)** is a framework from Microsoft that accelerates LLM inference by selectively masking sparse activation neurons during computation. Previous methods, such as TEAL and CATS, score neuron importance by using **activation magnitude** `|x|`. In contrast, WINA introduces the criterion of:

**WINA Score** = `|activation × weight_norm|`. It cobines how active a neuron is (activation) with how important its connections are (weight_norm). 

This enables for a better estimation of neuron contribution to the final output, resulting in more effective pruning and faster inference without retraining: 60%-60% FLOP reduction with minimal performance loss, up to 2.94% better accuracy than TEAL according to the paper. All the aformentioned methods are training-free, requiring an activation analysis run.

## 🎯 Problem
Current sparse activation methods (e.g., WINA, TEAL) optimize for dataset-wide averages, but different inputs may hold fundamentally different neural computation patterns. This one-size-fits-all approach leaves performance on the table.

This project explores **cluster-based profile optimization** to push WINA's performance even further.

## 🔬 Research Questions
**Neural Pattern Discovery**  
Do similar inputs (e.g., by task or subject) naturally cluster by neuron activation patterns?

**Cluster-Profile Optimization**  
Can specialized WINA profiles per cluster outperform global, uniform pruning strategies?

**Input Neuron Profile Classification**  
What are effective and efficient ways to classify new samples into neural-behavioral clusters?

**Performance Trade-offs**  
What are the latency vs. accuracy trade-offs when using cluster-based profiles?

## 🔧 Technical Methodology
1. **Generate WINA-based score histograms** on a benchmark training set, with attribution to individual input samples.
2. **Evaluate uniform WINA optimization** on the same benchmark training\* set (accuracy, compute).
3. **Cluster input samples** based on similarity of their neural activation patterns.
4. **Generate new cluster-specific WINA score histograms** using the identified clusters, on the same benchmark training set.
5. **Evaluate cluster-based WINA optimization profiles** on the benchmark training\* set.

At this point, we've evaluated whether **cluster-specific profiles** outperform the **uniform profile** on the **training set**.  
This evaluation serves as a **basic validation**, and is subject to **data leakage**, since the same set is used for both clustering and evaluation. This is temporary measure, taken since we don't have an input profile classifier yet.

We currently need to find a strategy for classifying an input to cluster-profile. Potential strategies:
- Train an input-profile classifier, using input embeddings
- Freeze WINA scores for initial model layers, and use their results to cluster the inputs - requiring creating initial layer cluster profiles

<img src="path/to/image.svg" width="300" />

## Contents
- [Install](#Install)
- [Sparsity Allocation](#Sparsity-Allocation)
- [Evaluation](#Evaluation)
- [Reproduction](#Reproduction)

## Install

1. Clone the repo and navigate to wina:

```
git clone https://github.com/microsoft/wina.git
cd wina
```

2. Set up environment:

```bash
conda create -yn wina python=3.11
conda activate wina

pip install -r requirements.txt
```

## Sparsity Allocation
We assign sparsity for each weight matrices instead of a uniform sparsity across the model through the greedy algorithm proposed in TEAL.

💡Note: If you want to assign uniform sparsity for all matrices, you can **skip this part**.

1. Construct hidden states histograms. 
```bash
python wina/grab_acts.py --model_name [MODEL_PATH] --output_path [OUTPUT_PATH] --sparse_mode [SPARSE_MODE] --transform(Optional)
```
2. Allocate sparsity
```bash
python wina/greedyopt.py --model_name [MODEL_PATH] --output_path [OUTPUT_PATH] --sparse_mode [SPARSE_MODE] --model_type [MODEL_TYPE] --transform(Optional)
```
💡Notes: The '--transform' flag enables tensor transformation. Simply omit this flag if you wish to skip the transformation process.

ℹ️ Arguments
* [MODEL_PATH]: Path to base model, can be remote path or local path
* [OUTPUT_PATH]: Path to save outputs, including hidden states histograms and sparsity allocation results.
* [SPARSE_MODE]: Can be **WINA** or **TEAL**
* [MODEL_TYPE]: Can be one of the following four:
    * Llama-3-8B
    * Llama-2-7B
    * Qwen-2.5-7B
    * Phi-4-14B

## Evaluation
```bash
python eval.py --base_model [MODEL_PATH] --save_path [OUTPUT_PATH] --sparsity [sparsity] --sparse_mode [SPARSE_MODE] --greedy
```
💡Notes: The --greedy flag enables per-matrix sparsity assignment. Ensure [Sparsity Allocation](#sparsity-allocation) is completed before using this option.

## Reproduction
### Dense model

```bash
bash scripts/baseline/qwen-2.5-7b.bash [MODEL_PATH]
bash scripts/baseline/llama-2-7b.bash [MODEL_PATH]
bash scripts/baseline/llama-3-8b.bash [MODEL_PATH]
bash scripts/baseline/phi_4_14b.bash [MODEL_PATH]
```

### WINA Sparsification

```bash
bash scripts/wina/qwen_2.5_7b.bash [MODEL_PATH]
bash scripts/wina/llama_2_7b.bash [MODEL_PATH]
bash scripts/wina/llama_3_8b.bash [MODEL_PATH]
bash scripts/wina/phi_4_14b.bash [MODEL_PATH]
```

### TEAL Sparsification
```bash
bash scripts/teal/qwen_2.5_7b.bash [MODEL_PATH]
bash scripts/teal/llama_2_7b.bash [MODEL_PATH]
bash scripts/teal/llama_3_8b.bash [MODEL_PATH]
bash scripts/teal/phi_4_14b.bash [MODEL_PATH]
```

### TEAL-transform Sparsification

```bash
bash scripts/teal_transform/qwen_2.5_7b.bash [MODEL_PATH]
bash scripts/teal_transform/llama_2_7b.bash [MODEL_PATH]
bash scripts/teal_transform/llama_3_8b.bash [MODEL_PATH]
bash scripts/teal_transform/phi_4_14b.bash [MODEL_PATH]
```

## Citation
If you find this repo useful, please consider citing:
```
@misc{chen2025winaweightinformedneuron,
      title={WINA: Weight Informed Neuron Activation for Accelerating Large Language Model Inference}, 
      author={Sihan Chen and Dan Zhao and Jongwoo Ko and Colby Banbury and Huiping Zhuang and Luming Liang and Tianyi Chen},
      year={2025},
      eprint={2505.19427},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.19427}, 
}
```
