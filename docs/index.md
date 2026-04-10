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

<section id="benchmarks" class="section-block">
  <h2>Performance Snapshot</h2>
  <p>Flux Attention is designed for long-context tasks where quality and runtime efficiency must co-exist.</p>
  <div class="table-wrap">
    <table class="benchmark-table">
      <thead>
        <tr>
          <th>Dimension</th>
          <th>Backbone Baseline</th>
          <th>Flux Attention</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Long-context retrieval quality</td>
          <td>Strong</td>
          <td>Comparable quality retention</td>
        </tr>
        <tr>
          <td>Inference compute</td>
          <td>High, uniform per layer</td>
          <td>Adaptive by layer routing</td>
        </tr>
        <tr>
          <td>Memory behavior</td>
          <td>Can be heavy on long input</td>
          <td>More stable under sparse routing</td>
        </tr>
        <tr>
          <td>Training cost for 8B setup</td>
          <td>Task dependent</td>
          <td>~12h on 8x A800 (reported)</td>
        </tr>
      </tbody>
    </table>
  </div>
</section>

<section id="overview" class="section-block">
  <h2>What Makes It Different</h2>
  <p>
    Flux Attention uses a layer-level router to allocate <strong>Full Attention</strong> and <strong>Sparse Attention</strong>
    dynamically. Instead of fixed attention patterns, it adapts to input characteristics to preserve quality while reducing cost.
  </p>
</section>

<section id="architecture" class="section-block architecture">
  <h2>Architecture At A Glance</h2>
  <p>The router decides attention mode per layer based on context complexity, balancing quality and compute in one forward pass.</p>
  <figure>
    <img src="{{ './assets/images/arch.png' | relative_url }}" alt="Flux Attention architecture overview">
    <figcaption>Method overview from the Flux Attention paper.</figcaption>
  </figure>
</section>

<section id="flow" class="section-block">
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

<section id="usecases" class="section-block">
  <h2>Where It Helps Most</h2>
  <div class="pill-grid">
    <span>Long document QA</span>
    <span>Multi-file code reasoning</span>
    <span>Conversation memory extension</span>
    <span>Retrieval-heavy generation</span>
    <span>Agentic planning with long traces</span>
    <span>Enterprise report summarization</span>
  </div>
</section>

<section id="quickstart" class="section-block">
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

<section id="resources" class="section-block">
  <h2>Resources</h2>
  <ul class="resource-list">
    <li><a href="https://github.com/qqtang-code/FluxAttention/blob/main/README.md" target="_blank" rel="noopener">Project README</a></li>
    <li><a href="https://github.com/qqtang-code/FluxAttention/tree/main/fluxattn/run_scripts" target="_blank" rel="noopener">Training Scripts</a></li>
    <li><a href="https://github.com/qqtang-code/FluxAttention/tree/main/fluxattn/src" target="_blank" rel="noopener">Core Attention Code</a></li>
  </ul>
</section>

<section id="roadmap" class="section-block">
  <h2>Project Roadmap</h2>
  <ul class="resource-list">
    <li>Release more checkpoints for different model sizes.</li>
    <li>Expand benchmark coverage on practical long-context scenarios.</li>
    <li>Improve deployment recipes for multi-GPU inference serving.</li>
    <li>Continue integration with evaluation toolchains.</li>
  </ul>
</section>

<section id="faq" class="section-block faq-block">
  <h2>FAQ</h2>
  <details>
    <summary>Does Flux Attention require model architecture changes?</summary>
    <p>It relies on router-guided hybrid attention behavior and corresponding implementation support, with checkpoints and code provided in this project.</p>
  </details>
  <details>
    <summary>Is this only useful for very long contexts?</summary>
    <p>The benefit is strongest on long input, but adaptive routing can still provide practical efficiency gains in mixed workloads.</p>
  </details>
  <details>
    <summary>How should I evaluate it?</summary>
    <p>Use long-context benchmarks (for example with LOOM-Eval) and compare both quality metrics and wall-clock runtime.</p>
  </details>
</section>

<section id="citation" class="section-block citation">
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
