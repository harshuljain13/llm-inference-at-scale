<p align="center">
  <img src="assets/banner.svg" alt="LLM Inference at Scale" width="100%">
</p>

<p align="center">
  <strong>The definitive guide to serving large language models in production.</strong>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-table-of-contents">Contents</a> •
  <a href="#-labs">Labs</a> •
  <a href="#-community">Community</a> •
  <a href="#-contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-All%20Rights%20Reserved-red.svg" alt="All Rights Reserved">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/vLLM-0.8+-orange.svg" alt="vLLM">
  <img src="https://img.shields.io/badge/SGLang-0.4+-purple.svg" alt="SGLang">
  <img src="https://img.shields.io/badge/TensorRT--LLM-0.15+-76B900.svg" alt="TensorRT-LLM">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.0+-76B900.svg" alt="CUDA">
</p>

---

## Why This Exists

LLM inference is hard. Not "read the docs and figure it out" hard — **fundamentally different from everything else in ML** hard.

Traditional ML inference is a solved problem. You batch requests, run a forward pass, return results. Latency is predictable, memory is fixed, scaling is linear.

LLM inference breaks all of these assumptions:
- **Latency is unpredictable** — a 10-token response takes 100ms, a 1000-token response takes 10 seconds
- **Memory grows during requests** — the KV cache expands with every generated token
- **Scaling is sub-linear** — communication overhead dominates as you add GPUs
- **Cost is 100x higher** — $0.001/request becomes $0.10/request

This handbook exists because we needed it and couldn't find it. The knowledge is scattered across papers, blog posts, tribal knowledge, and source code comments. We've consolidated years of production experience and research into one comprehensive resource.

**This is the guide we wish existed when we started.**

---

> 📬 **Follow the build** — New chapters, explained in plain English with production context.
> Subscribe to [The Engineer's Digest](https://harshuljain.substack.com) to get notified
> when new content drops. **[Subscribe free →](https://harshuljain.substack.com)**

💬 [Join the discussion](https://github.com/harshuljain13/llm-inference-at-scale/discussions) 
— questions, feedback, and corrections welcome
---

## 🎯 What You'll Learn

<table>
<tr>
<td width="50%">

### Foundations
- Why LLM inference costs 100x more than traditional ML
- The memory bandwidth wall and how to work around it
- Prefill vs decode: two completely different problems
- KV cache mechanics at the byte level

</td>
<td width="50%">

### Production Skills
- Choosing between vLLM, SGLang, and TensorRT-LLM
- Quantization tradeoffs (INT8, INT4, FP8, FP4)
- Tensor parallelism and multi-GPU serving
- Capacity planning and SLO management

</td>
</tr>
<tr>
<td width="50%">

### Optimization Techniques
- PagedAttention and memory-efficient serving
- Continuous batching for throughput
- Speculative decoding for latency
- FlashAttention and kernel-level optimizations

</td>
<td width="50%">

### Advanced Topics
- Disaggregated serving (prefill/decode separation)
- MoE inference and expert routing
- KV cache compression and eviction
- Edge deployment with llama.cpp

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA 12.0+ (for GPU labs)
- Basic PyTorch familiarity

### Installation

```bash
git clone https://github.com/harshuljain13/llm-inference-at-scale.git
cd llm-inference-at-scale

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Start Reading

```bash
# Open the first chapter
open content/00_foundations/00.0_transformer_anatomy_and_memory/transformer_anatomy_and_memory.md
```

Or browse the [Table of Contents](#-table-of-contents) below.

---

## 📚 Table of Contents

> **12 chapters. Each module is a focused 8-10 minute read with a companion lab.**

### Part I: Foundations

**Chapter 00: The Transformer at Inference Time** (4 modules)

| Module | Title |
|:------:|-------|
| 00.1 | [0.1 Transformer Architecture](content/00_transformer_at_inference_time/00.1_transformer_architecture/transformer_architecture.md) |
| 00.2 | [0.2 What Happens During Inference](content/00_transformer_at_inference_time/00.2_what_happens_during_inference/what_happens_during_inference.md) |
| 00.3 | [[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harshuljain13/llm-inference-at-scale/blob/master/content/00_transformer_at_inference_time/00.3_attention_and_kv_cache/lab.ipynb)](content/00_transformer_at_inference_time/00.3_attention_and_kv_cache/attention_and_kv_cache.md) |
| 00.4 | [[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harshuljain13/llm-inference-at-scale/blob/master/content/00_transformer_at_inference_time/00.4_why_llm_inference_is_different/lab.ipynb) [![Open In Molab](https://img.shields.io/badge/Open%20in-Molab-blue)](https://molab.marimo.io/github/harshuljain13/llm-inference-at-scale/blob/master/content/00_transformer_at_inference_time/00.4_why_llm_inference_is_different/lab.ipynb)](content/00_transformer_at_inference_time/00.4_why_llm_inference_is_different/why_llm_inference_is_different.md) |

**Chapter 01: GPU Hardware for Inference** (2 modules)

| Module | Title |
|:------:|-------|
| 01.1 | [1.1 GPU Memory Hierarchy](content/01_gpu_hardware/01.1_gpu_memory_hierarchy/gpu_memory.md) |
| 01.2 | [1.2 The Roofline Model](content/01_gpu_hardware/01.2_roofline_model/roofline_fundamentals.md) |

**Chapter 02: Sizing and Serving** (3 modules)

| Module | Title |
|:------:|-------|
| 02.1 | [2.1 VRAM Budget](content/02_sizing_and_serving/02.1_vram_budget/vram_budgeting.md) |
| 02.2 | [2.2 Batch Size and Throughput](content/02_sizing_and_serving/02.2_batch_size_and_throughput/batch_and_instance_selection.md) |
| 02.3 | [2.3 Serving Implications](content/02_sizing_and_serving/02.3_serving_implications/serving_implications.md) |

### Part II: Optimizations

**Chapter 03: Attention Variants** (7 modules)

| Module | Title |
|:------:|-------|
| 03.1 | [3.1 Multi-Head Attention (MHA)](content/03_attention_variants/03.1_mha/mha.md) |
| 03.2 | [3.2 MQA and GQA](content/03_attention_variants/03.2_mqa_gqa/mqa_gqa.md) |
| 03.3 | [3.3 GQA Deep Dive](content/03_attention_variants/03.3_gqa_deep_dive/gqa_deep_dive.md) |
| 03.4 | [3.4 Multi-Latent Attention (MLA)](content/03_attention_variants/03.4_mla/multi_latent_attention.md) |
| 03.5 | [3.5 FlashAttention: Why Standard Attention is Slow](content/03_attention_variants/03.5_flash_attention_problem/flash_attention_problem.md) |
| 03.6 | [3.6 FlashAttention: The Algorithm](content/03_attention_variants/03.6_flash_attention_algorithm/flash_attention_algorithm.md) |
| 03.7 | [3.7 FlashAttention: In Practice](content/03_attention_variants/03.7_flash_attention_practice/flash_attention_practice.md) |

**Chapter 04: KV Cache Engineering** (5 modules)

| Module | Title |
|:------:|-------|
| 04.1 | [4.1 PagedAttention](content/04_kv_cache_engineering/04.1_paged_attention/paged_attention.md) |
| 04.2 | [4.2 KV Cache Compression](content/04_kv_cache_engineering/04.2_kv_cache_compression/kv_cache_compression.md) |
| 04.3 | [4.3 Smart KV Caching](content/04_kv_cache_engineering/04.3_smart_kv_caching/smart_kv_caching.md) |
| 04.4 | [4.4 LMCache](content/04_kv_cache_engineering/04.4_lmcache/lmcache.md) |
| 04.5 | [4.5 Prefix Caching](content/04_kv_cache_engineering/04.5_prefix_caching/prefix_caching.md) |

**Chapter 05: Optimization Techniques** (6 modules)

| Module | Title |
|:------:|-------|
| 04.1 | [4.1 Quantization](content/05_optimization/04.1_quantization/quantization.md) |
| 04.2 | [4.2 TurboQuant](content/05_optimization/04.2_turboquant/turboquant.md) |
| 04.3 | [4.3 Continuous Batching](content/05_optimization/04.3_continuous_batching/continuous_batching.md) |
| 04.4 | [4.4 Advanced Speculative Decoding](content/05_optimization/04.4_speculative_decoding/speculative_decoding_advanced.md) |
| 04.5 | [4.5 Chunked Prefill](content/05_optimization/04.5_chunked_prefill/chunked_prefill.md) |
| 04.6 | [Inference-Time Compute: Spending More Tokens to Get Better Answers](content/05_optimization/04.6_inference_time_compute/inference_time_compute.md) |

### Part III: Engines & Scaling

**Chapter 06: Inference Engines** (4 modules)

| Module | Title |
|:------:|-------|
| 05.1 | [vLLM: The Production Standard for Open-Source LLM Serving](content/06_engines/05.1_vllm/vllm.md) |
| 05.2 | [SGLang: A Programming Language for Structured LLM Inference](content/06_engines/05.2_sglang/sglang.md) |
| 05.3 | [TensorRT-LLM: Compiler-First Inference on NVIDIA Hardware](content/06_engines/05.3_tensorrt_llm/tensorrt_llm.md) |
| 05.4 | [NVIDIA Dynamo: The Next-Generation Distributed Inference Framework](content/06_engines/05.4_nvidia_dynamo/nvidia_dynamo.md) |

**Chapter 07: Scaling** (2 modules)

| Module | Title |
|:------:|-------|
| 06.1 | [6.1 Tensor Parallelism](content/07_scaling/06.1_tensor_parallelism/tensor_parallelism.md) |
| 06.2 | [6.2 Mixture-of-Experts Inference](content/07_scaling/06.2_moe_inference/moe_inference.md) |
| 06.3 | [6.3 Distillation for Serving](content/07_scaling/06.3_distillation/distillation.md) |

### Part IV: Production

**Chapter 08: Serving Infrastructure** (7 modules)

| Module | Title |
|:------:|-------|
| 07.1 | [7.1 Ray Serve](content/08_serving/07.1_ray_serve/ray_serve.md) |
| 07.2 | [7.2 EKS and KServe](content/08_serving/07.2_eks_kserve/eks_kserve.md) |
| 07.3 | [7.3 SageMaker](content/08_serving/07.3_sagemaker/sagemaker.md) |
| 07.4 | [7.4 Disaggregated Serving](content/08_serving/07.4_disaggregated_serving/disaggregated_serving.md) |
| 07.5 | [7.5 Cold Start Optimization](content/08_serving/07.5_cold_start/cold_start.md) |
| 07.6 | [Cache-Aware Routing and Semantic Prompt Caching](content/08_serving/07.6_cache_aware_routing/cache_aware_routing.md) |
| 07.7 | [7.7 Kubernetes-Native LLM Inference Infrastructure](content/08_serving/07.7_kubernetes_inference_infrastructure/kubernetes_inference_infrastructure.md) |

**Chapter 09: Operations** (6 modules)

| Module | Title |
|:------:|-------|
| 08.1 | [8.1 Benchmarking and Metrics](content/09_operations/08.1_benchmarking/benchmarking.md) |
| 08.2 | [8.2 Structured Output and Guided Decoding](content/09_operations/08.2_structured_output/structured_output.md) |
| 08.3 | [8.3 Edge Deployment](content/09_operations/08.3_edge_deployment/edge_deployment.md) |
| 08.4 | [Inference Metrics, Goodput, and Production Monitoring](content/09_operations/08.4_inference_metrics/inference_metrics.md) |
| 08.5 | [Multi-Region Inference and KV Cache Locality](content/09_operations/08.5_multi_region_kv_locality/multi_region_kv_locality.md) |
| 08.6 | [Beyond GPUs: Custom Silicon for LLM Inference](content/09_operations/08.6_custom_silicon/custom_silicon.md) |

**Chapter 10: Production Stories** (2 modules)

| Module | Title |
|:------:|-------|
| 09.1 | [Meta's LLM Inference Infrastructure: Running Every Optimization Simultaneously at Planetary Scale](content/10_production_stories/09.1_meta_inference_platform/meta_inference_platform.md) |
| 09.2 | [Databricks Multi-Tenant LLM Serving: Model Units, LoRA Multiplexing, and Fleet Economics](content/10_production_stories/09.2_databricks_multi_tenant/databricks_multi_tenant.md) |
| 09.3 | [Managing Mixed LLM Inference Workloads on Shared Infrastructure](content/10_production_stories/09.3_mixed_workload_management/mixed_workload_management.md) |

**Chapter 11: System Designs** (5 modules)

| Module | Title |
|:------:|-------|
| 10.1 | [System Design: ChatGPT-Scale Conversational AI at 1M Concurrent Users](content/11_system_designs/10.1_chatgpt_scale_chatbot/chatgpt_scale_chatbot.md) |
| 10.2 | [09.2 System Design: Code Completion Copilot](content/11_system_designs/10.2_code_copilot/code_copilot.md) |
| 10.3 | [9.3 Enterprise RAG Service: Multi-Model Knowledge Assistant](content/11_system_designs/10.3_enterprise_rag/enterprise_rag.md) |
| 10.4 | [Multi-Model API Gateway: System Design](content/11_system_designs/10.4_multi_model_gateway/multi_model_gateway.md) |
| 10.5 | [09.5 System Design: Inference Infrastructure for Agentic AI Workloads](content/11_system_designs/10.5_agentic_workload/agentic_workload.md) |


---

**Total: 12 chapters, 56 modules.**

## 📐 Key Equations

Formulas you'll use constantly when working with LLM inference:

### Memory Bandwidth Ceiling

The theoretical maximum decode speed, limited by how fast you can read model weights:

```
max_tokens_per_second = memory_bandwidth / model_size_bytes
```

**Example:** Llama 8B (16GB FP16) on A100 (2 TB/s) → 125 tokens/sec maximum

### KV Cache Size

Memory required for the key-value cache:

```
kv_cache_bytes = 2 × num_layers × num_kv_heads × head_dim × seq_len × batch_size × dtype_bytes
```

**Example:** Llama 8B, batch=1, seq=4096, FP16 → 512 MB

### Arithmetic Intensity

Determines whether a workload is compute-bound or memory-bound:

```
arithmetic_intensity = FLOPs / bytes_transferred
```

**Rule of thumb:** Below the ridge point (~156 FLOPs/byte on A100) = memory-bound

---

## 🗂️ Repository Structure

```
llm-inference-at-scale/
├── content/                      # 📖 Handbook chapters
│   ├── 00_foundations/           #    Part I: Foundations
│   ├── 01_gpu_fundamentals/      #    Part II: GPU Fundamentals
│   ├── 02_attention_and_kv/      #    Part III: Attention & KV Cache
│   ├── 03_optimization/          #    Part IV: Optimization Techniques
│   ├── 04_engines/               #    Part V: Inference Engines
│   ├── 05_scaling/               #    Part VI: Scaling
│   ├── 06_serving/               #    Part VII: Production Serving
│   ├── 07_operations/            #    Part VIII: Operations
│   └── utils/                    #    Visualization utilities
├── labs/                         # 🧪 Hands-on exercises
├── reference/                    # 📋 Quick references
│   ├── cheat_sheet.md            #    One-page summary
│   ├── glossary.md               #    Terminology
│   ├── vllm_quick_reference.md   #    vLLM commands
│   └── cost_calculator.py        #    Inference cost estimation
├── assets/                       # 🎨 Images and diagrams
└── slides/                       # 📊 Presentation materials
```

---

## 🎓 Learning Paths

### 🏃 Speed Run (2 hours)

For engineers who need to deploy an LLM this week:

1. [0.0 Transformer Anatomy & Memory](content/00_foundations/00.0_transformer_anatomy_and_memory/transformer_anatomy_and_memory.md) — 15 min
2. [0.1 What is LLM Inference?](content/00_foundations/00.1_what_is_llm_inference/what_is_llm_inference.md) — 20 min
3. [3.1 Quantization](content/04_optimization/04.1_quantization/quantization.md) — 20 min
4. [4.1 vLLM](content/05_engines/05.1_vllm/vllm.md) — 30 min
5. [Lab 04: vLLM Deployment](labs/lab_04_vllm_deployment/) — 45 min

### 📚 Deep Dive (1 day)

For engineers building inference infrastructure:

- **Morning:** Part I (Foundations) + Part II (GPU Fundamentals)
- **Afternoon:** Part IV (Optimization) + Part V (Engines)
- **Labs:** 01, 02, 03, 04

### 🎯 Complete Coverage (3 days)

For teams standardizing on LLM serving:

- **Day 1:** Parts I, II, III — Foundations through KV Cache
- **Day 2:** Parts IV, V, VI — Optimization through Scaling
- **Day 3:** Parts VII, VIII — Production Serving and Operations
- **Labs:** All 10 labs

---

## 🌟 Community

### Presentations

This material has been presented at:

- *More talks coming — if you'd like this at your conference or meetup, open an issue.*

### Citation

If you use this material in research or internal documentation, please cite:

```bibtex
@misc{llm-inference-at-scale,
  title={LLM Inference at Scale: A Practitioner's Handbook},
  author={Jain, Harshul},
  year={2025},
  url={https://github.com/harshuljain13/llm-inference-at-scale}
}
```

### Star History

If you find this useful, please ⭐ the repo — it helps others discover it.

---

## 🤝 Contributing

Contributions are welcome. This is a living document.

### Ways to Contribute

- **Fix errors** — Typos, outdated information, incorrect formulas
- **Improve clarity** — Better explanations, additional examples
- **Add content** — New chapters, labs, or reference materials

### Process

1. Fork the repository
2. Create a feature branch (`git checkout -b improve-kv-cache-chapter`)
3. Make your changes
4. Submit a pull request

> To report errors or suggest corrections, open a GitHub Issue.

---

## 👤 About the Author

**Harshul Jain** is a Senior ML Infrastructure Engineer at Audible (Amazon), where he owns the ML Feature Store, a GenAI semantic search platform serving millions of customers, and real-time streaming pipelines at scale. He has been building and operating ML infrastructure in production for 4+ years and mentors 300+ engineers through an eMentoring program.

- GitHub: [@harshuljain13](https://github.com/harshuljain13)
- Newsletter: [The Engineer's Digest](https://harshuljain.substack.com) — LLM inference, deeply explained

---

## ⚠️ Disclaimer

The views, techniques, and opinions expressed in this handbook are solely those of the author and **do not represent the views of Audible, Amazon, or any affiliated organization**. No proprietary, confidential, or internal Amazon/Audible systems, data, or information has been included. All content is based on publicly available research, open-source tooling, and the author's independent experience and analysis.

This handbook is provided for **educational purposes only**. Production infrastructure decisions should be validated against your specific workload, hardware, and organizational constraints. The author makes no guarantees about the accuracy, completeness, or fitness for purpose of any content herein.

---

## 📄 License & Copyright

© 2026 Harshul Jain. All rights reserved.

No part of this work — including the framework, diagrams, models, terminology, chapter structure, or related materials — may be reproduced, distributed, modified, adapted, or used in whole or in part without prior written permission from the author. This includes but is not limited to use in courses, training programs, consulting engagements, publications, presentations, software, or organizational materials.

### Framework

The framework presented in this work is the intellectual property of Harshul Jain. It may not be copied, adapted, taught, commercialized, incorporated into derivative works, or used in any professional, commercial, or organizational context — including consulting, training, software, presentations, publications, or organizational materials — without prior written permission.

> To request permission, open a GitHub Issue or contact via the profile above.

---

## 🙏 Acknowledgments

This handbook builds on the work of many researchers and engineers:

- The [vLLM](https://github.com/vllm-project/vllm) team for PagedAttention and continuous batching
- The [SGLang](https://github.com/sgl-project/sglang) team for RadixAttention
- Tri Dao for [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- The authors of foundational papers: Attention Is All You Need, GQA, Medusa, EAGLE, and many others

---

📬 **Stay updated** — [Subscribe to The Engineer's Digest](https://harshuljain.substack.com) for chapter releases and build-in-public updates.

---

<p align="center">
  <strong>Built with ❤️ for the ML infrastructure community</strong>
</p>

<p align="center">
  <a href="#llm-inference-at-scale">Back to top ↑</a>
</p>
