---
layout: default
title: Flux Attention
---

<section id="top" class="hero">
  <p class="eyebrow">Long-Context LLM Acceleration</p>
  <h1>Flux Attention</h1>
  <p class="tagline">Context-Aware Hybrid Attention for efficient, high-fidelity inference at scale.</p>
  <div class="hero-chips">
    <span>Adaptive Layer Router</span>
    <span>Full + Sparse Hybrid</span>
    <span>Production-Oriented Inference</span>
  </div>
  <div class="hero-actions">
    <a class="btn btn-primary" href="https://arxiv.org/abs/2604.07394" target="_blank" rel="noopener">Read Paper</a>
    <a class="btn btn-ghost" href="https://github.com/qqtang-code/FluxAttention" target="_blank" rel="noopener">View on GitHub</a>
  </div>
  <div class="hero-links">
    <a href="https://huggingface.co/collections/QQTang1223/flux-attention" target="_blank" rel="noopener">Hugging Face Collection</a>
    <span>•</span>
    <a href="https://modelscope.cn/collections/tang031223/Flux-Attention" target="_blank" rel="noopener">ModelScope Collection</a>
  </div>
</section>

<section id="highlights" class="stat-grid">
  <article class="stat-card">
    <h3>12 Hours</h3>
    <p>Training for 8B-scale models on 8x A800 GPUs.</p>
  </article>
  <article class="stat-card">
    <h3>Long Context</h3>
    <p>Maintains strong retrieval quality across long sequences.</p>
  </article>
  <article class="stat-card">
    <h3>Practical Speedups</h3>
    <p>Higher sparsity with stable memory behavior in inference.</p>
  </article>
</section>

<section id="overview" class="section-block">
  <h2>What Makes It Different</h2>
  <p>
    Flux Attention uses a layer-level router to allocate <strong>Full Attention</strong> and <strong>Sparse Attention</strong>
    dynamically. Instead of fixed attention patterns, it adapts to input characteristics to preserve quality while reducing cost.
  </p>
</section>

<section id="architecture" class="section-block architecture reveal">
  <h2>Architecture At A Glance</h2>
  <p>The router decides attention mode per layer based on context complexity, balancing quality and compute in one forward pass.</p>
  <figure>
    <img src="{{ '/assets/images/arch.png' | relative_url }}" alt="Flux Attention architecture overview">
    <figcaption>Method overview from the Flux Attention paper.</figcaption>
  </figure>
</section>

<section id="flow" class="section-block reveal">
  <h2>Inference Flow</h2>
  <div class="flow-grid">
    <article>
      <h3>1. Context Profiling</h3>
      <p>Capture token distribution and locality signals from input context.</p>
    </article>
    <article>
      <h3>2. Layer Routing</h3>
      <p>Route each layer to Full or Sparse attention according to routing confidence.</p>
    </article>
    <article>
      <h3>3. Hybrid Execution</h3>
      <p>Execute mixed attention patterns while preserving stable memory behavior.</p>
    </article>
    <article>
      <h3>4. Output Generation</h3>
      <p>Decode with long-context fidelity while reducing end-to-end wall clock cost.</p>
    </article>
  </div>
</section>

<section id="quickstart" class="section-block reveal">
  <h2>Quick Start</h2>
  <div class="code-stack">
    <h3>Installation</h3>

```bash
conda create -n flux_attn python=3.11
conda activate flux_attn

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install modelscope
pip install -e .
```

    <h3>Training</h3>

```bash
chmod +x fluxattn/run_scripts/training.sh
cd fluxattn
bash run_scripts/training.sh
```

    <h3>Evaluation</h3>
    <p>For comprehensive long-context benchmarks, use <a href="https://github.com/LCM-Lab/LOOM-Eval" target="_blank" rel="noopener">LOOM-Eval</a>.</p>
  </div>
</section>

<section id="resources" class="section-block reveal">
  <h2>Resources</h2>
  <ul class="resource-list">
    <li><a href="https://github.com/qqtang-code/FluxAttention/blob/main/README.md" target="_blank" rel="noopener">Project README</a></li>
    <li><a href="https://github.com/qqtang-code/FluxAttention/tree/main/fluxattn/run_scripts" target="_blank" rel="noopener">Training Scripts</a></li>
    <li><a href="https://github.com/qqtang-code/FluxAttention/tree/main/fluxattn/src" target="_blank" rel="noopener">Core Attention Code</a></li>
  </ul>
</section>

<section id="citation" class="section-block citation reveal">
  <h2>Citation</h2>

```bibtex
@misc{qiu2026fluxattentioncontextawarehybrid,
  title={Flux Attention: Context-Aware Hybrid Attention for Efficient LLMs Inference},
  author={Quantong Qiu and Zhiyi Hong and Yi Yang and Haitian Wang and Kebin Liu and Qingqing Dang and Juntao Li and Min Zhang},
  year={2026},
  eprint={2604.07394},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2604.07394}
}
```
</section>
