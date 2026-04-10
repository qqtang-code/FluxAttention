---
layout: home
title: Flux Attention
---

# Flux Attention

Context-Aware Hybrid Attention for Efficient LLMs Inference.

[Paper (arXiv)](https://arxiv.org/abs/2601.17367) | [Hugging Face Collection](https://huggingface.co/collections/QQTang1223/flux-attention) | [ModelScope Collection](https://modelscope.cn/collections/tang031223/Flux-Attention)

## Overview

Flux Attention dynamically allocates **Full Attention** and **Sparse Attention** at layer level with a Layer Router. It is designed to keep model quality while reducing inference cost on long-context tasks.

Core highlights:

- Efficient training for 8B-scale models.
- Strong long-context performance.
- Significant inference speedup with stable memory behavior.

## Quick Start

### Installation

```bash
conda create -n flux_attn python=3.11
conda activate flux_attn

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install modelscope
pip install -e .
```

### Training

```bash
chmod +x fluxattn/run_scripts/training.sh
cd fluxattn
bash run_scripts/training.sh
```

### Evaluation

We recommend [LOOM-Eval](https://github.com/LCM-Lab/LOOM-Eval) for long-context benchmarks.

## Repository

Full documentation, scripts, and source code are available in this repository:

- [README](https://github.com/qqtang-code/FluxAttention/blob/main/README.md)
- [Training scripts](https://github.com/qqtang-code/FluxAttention/tree/main/fluxattn/run_scripts)
- [Core attention code](https://github.com/qqtang-code/FluxAttention/tree/main/fluxattn/src)

## Citation

```bibtex
@misc{tang2026fluxattentiontesttimeadaptive,
  title={Flux Attention: Context-Aware Hybrid Attention for Efficient LLMs Inference},
  author={Quantong Qiu and Zhiyi Hong and Yi Yang and Haitian Wang and Kebin Liu and Qingqing Dang and Juntao Li and Min Zhang},
  year={2026},
  eprint={2601.17367},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2601.17367}
}
```
