# 🚀 WINA-AMP: WINA Activation Magnitude Profile optimization

[![Original WINA](https://img.shields.io/badge/Based%20on-Microsoft%2FWINA-blue.svg)](https://github.com/microsoft/wina)
[![Paper](https://img.shields.io/badge/arXiv-2505.19427-red.svg)](https://arxiv.org/abs/2505.19427)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)

*This repository builds on Microsoft's WINA framework to develop cluster-specific WINA profiles for improved LLM inference efficiency.*
<br>
## What's WINA
**WINA (Weight-Informed Neuron Activation)** is a framework from Microsoft that accelerates LLM inference by selectively masking sparse activation neurons during computation. Previous methods, such as TEAL and CATS, score neuron importance by using **activation magnitude** `|x|`. In contrast, WINA introduces the criterion of:

**WINA Score** = `|activation × weight_norm|`.  
It cobines how active a neuron is (activation) with how important its connections are (weight_norm). 

This enables for a better estimation of neuron contribution to the final output, resulting in more effective pruning and faster inference without retraining: 60%-60% FLOP reduction with minimal performance loss, up to 2.94% better accuracy than TEAL according to the paper. All the aformentioned methods are training-free, requiring an activation analysis run.
<br><br>
## 🎯 Problem
Current sparse activation methods (e.g., WINA, TEAL) optimize for dataset-wide averages, but different inputs may hold fundamentally different neural computation patterns. This one-size-fits-all approach leaves performance on the table.

This project explores **cluster-based profile discovery & optimization** to push WINA's performance even further.
<br><br>
## 🔬 Research Questions
**Neural Pattern Discovery**  
Do similar inputs (e.g., by task or subject) naturally cluster by neuron activation patterns?

**Cluster-Profile Optimization**  
Can specialized WINA profiles per cluster outperform global, uniform pruning strategies?

**Input Neuron Profile Classification**  
What are effective and efficient ways to classify new samples into neural-behavioral clusters?

**Performance Trade-offs**  
What are the latency vs. accuracy trade-offs when using cluster-based profiles?
<br><br>
## 🔧 Technical Methodology
1.  **Generate WINA-based score histograms** on a benchmark training set, with attribution to individual input samples.  
2.  **Evaluate uniform WINA optimization** on benchmark training\* and test sets (accuracy, compute).  
3.  **Cluster input samples** based on similarity of their neural activation patterns.  
4. **Generate new cluster-specific WINA score histograms** using the identified clusters, on the same benchmark training set.  
5.  **Evaluate cluster-based WINA optimization profiles** on the benchmark training\* set.  

At this point, we've evaluated whether **cluster-specific profiles** outperform the **uniform profile** on the **training set**.  
This evaluation serves as a **basic validation**, and is subject to **data leakage**, since the same set is used for both clustering and evaluation. This is temporary measure, taken since we don't have an input profile classifier yet.

6. We currently need to find a strategy for classifying an input to a WINA profile:  
**Train an MoE-style input-profile classifier** based on input embeddings, lightweight enough for runtime.  
7.  Final Evaluation: **Compare WINA-AMP profiles to original uniform WINA prfoile** on test set, evaluate both accuracy & compute. 
<br><br><br>
<p align="center">
   <img src="figures/research_flow_1.drawio.svg" width="700" />
</p>
<br><br>

## Citation
If you find this repo useful, please consider citing the original paper:
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
