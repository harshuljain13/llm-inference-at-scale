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
open content/00_foundations/00.0_what_is_llm_inference/what_is_llm_inference.md
```

Or browse the [Table of Contents](#-table-of-contents) below.

---

## 📚 Table of Contents

### Part I: Foundations

| Chapter | Title | Description |
|:-------:|-------|-------------|
| 0.0 | [What is LLM Inference?](content/00_foundations/00.0_what_is_llm_inference/what_is_llm_inference.md) | The four stages: tokenization → prefill → decode → detokenization. Key metrics: TTFT, ITL, throughput. |
| 0.1 | [Why LLM Inference is Different](content/00_foundations/00.1_why_llm_inference_is_different/why_llm_inference_is_different.md) | The 100x cost gap explained. Memory bandwidth wall. Why traditional ML rules don't apply. |
| 0.2 | [Transformer Inference Mechanics](content/00_foundations/00.2_transformer_inference_basics/transformer_inference_basics.md) | Byte-level attention walkthrough. KV cache math. GQA/MQA tradeoffs with real numbers. |

### Part II: GPU Fundamentals

| Chapter | Title | Description |
|:-------:|-------|-------------|
| 1.1 | [GPU Memory](content/01_gpu_fundamentals/01.1_gpu_memory/gpu_memory.md) | HBM architecture, memory hierarchy, VRAM budgeting for production. |
| 1.2 | [Roofline Model](content/01_gpu_fundamentals/01.2_roofline_model/roofline_model.md) | Compute vs memory bound analysis. Arithmetic intensity. Performance prediction. |
| 1.3 | [FlashAttention](content/01_gpu_fundamentals/01.3_flash_attention/flash_attention.md) | Why it's essential, how it works, integration with inference engines. |

### Part III: Attention & KV Cache

| Chapter | Title | Description |
|:-------:|-------|-------------|
| 2.1 | [KV Caching](content/02_attention_and_kv/02.1_kv_caching/kv_caching.md) | Why KV cache exists. Memory formulas. Growth patterns and limits. |
| 2.2 | [Attention Mechanisms](content/02_attention_and_kv/02.2_attention_mechanisms/attention_mechanisms.md) | MHA → MQA → GQA evolution. Quality vs memory tradeoffs. |
| 2.3 | [PagedAttention](content/02_attention_and_kv/02.3_paged_attention/paged_attention.md) | Virtual memory for KV cache. vLLM's breakthrough innovation. |
| 2.4 | [KV Cache Compression](content/02_attention_and_kv/02.4_kv_cache_compression/kv_cache_compression.md) | Quantized KV, eviction policies, cross-request sharing. |

### Part IV: Optimization Techniques

| Chapter | Title | Description |
|:-------:|-------|-------------|
| 3.1 | [Quantization](content/03_optimization/03.1_quantization/quantization.md) | INT8, INT4, FP8 deep dive. When each makes sense. Quality impact. |
| 3.2 | [TurboQuant](content/03_optimization/03.2_turboquant/turboquant.md) | FP4 and the frontier of aggressive quantization. |
| 3.3 | [Continuous Batching](content/03_optimization/03.3_continuous_batching/continuous_batching.md) | Dynamic batching for throughput. Iteration-level scheduling. |
| 3.4 | [Speculative Decoding](content/03_optimization/03.4_speculative_decoding/speculative_decoding.md) | Draft models, verification, acceptance rates. 2-3x speedup techniques. |
| 3.5 | [Chunked Prefill](content/03_optimization/03.5_chunked_prefill/chunked_prefill.md) | Balancing prefill and decode latency for mixed workloads. |

### Part V: Inference Engines

| Chapter | Title | Description |
|:-------:|-------|-------------|
| 4.1 | [vLLM](content/04_engines/04.1_vllm/vllm.md) | Architecture deep dive. Configuration guide. Production tuning. |
| 4.2 | [SGLang](content/04_engines/04.2_sglang/sglang.md) | RadixAttention, structured output, when to choose it over vLLM. |
| 4.3 | [TensorRT-LLM](content/04_engines/04.3_tensorrt_llm/tensorrt_llm.md) | NVIDIA's optimized runtime. Compilation tradeoffs. Best practices. |

### Part VI: Scaling

| Chapter | Title | Description |
|:-------:|-------|-------------|
| 5.1 | [Tensor Parallelism](content/05_scaling/05.1_tensor_parallelism/tensor_parallelism.md) | Splitting models across GPUs. Communication patterns. Efficiency limits. |
| 5.2 | [MoE Inference](content/05_scaling/05.2_moe_inference/moe_inference.md) | Expert routing at scale. Load balancing. DeepSeek-V2/V3 insights. |
| 5.3 | [Distillation](content/05_scaling/05.3_distillation/) | Compressing large models for efficient serving. |

### Part VII: Production Serving

| Chapter | Title | Description |
|:-------:|-------|-------------|
| 6.1 | [Ray Serve](content/06_serving/06.1_ray_serve/ray_serve.md) | Building scalable LLM services with Ray. |
| 6.2 | [EKS + KServe](content/06_serving/06.2_eks_kserve/eks_kserve.md) | Kubernetes-native LLM deployment on AWS. |
| 6.3 | [SageMaker](content/06_serving/06.3_sagemaker/sagemaker.md) | Managed inference with SageMaker LMI. |
| 6.4 | [Disaggregated Serving](content/06_serving/06.4_disaggregated_serving/disaggregated_serving.md) | Separating prefill and decode. 40-60% cost reduction. |
| 6.5 | [Cold Start](content/06_serving/06.5_cold_start/cold_start.md) | Minimizing startup latency for serverless LLMs. |

### Part VIII: Operations

| Chapter | Title | Description |
|:-------:|-------|-------------|
| 7.1 | [Benchmarking](content/07_operations/07.1_benchmarking/benchmarking.md) | TTFT, ITL, throughput measurement. Workload replay. |
| 7.2 | [Structured Output](content/07_operations/07.2_structured_output/) | JSON schemas, grammar-guided generation, Outlines. |
| 7.3 | [Edge Deployment](content/07_operations/07.3_edge_deployment/edge_deployment.md) | llama.cpp, GGUF, mobile inference, Apple Silicon. |

---

## 🧪 Labs

Hands-on exercises to reinforce each concept. Each lab includes starter code, step-by-step instructions, and solutions.

| Lab | Title | Prerequisites | Time |
|:---:|-------|---------------|:----:|
| 01 | [Transformer Forward Pass](labs/lab_01_transformer_forward_pass/) | Chapter 0.2 | 45 min |
| 02 | [VRAM Calculation](labs/lab_02_vram_calculation/) | Chapter 1.1 | 30 min |
| 03 | [Quantization Comparison](labs/lab_03_quantization_comparison/) | Chapter 3.1 | 60 min |
| 04 | [vLLM Deployment](labs/lab_04_vllm_deployment/) | Chapter 4.1 | 45 min |
| 05 | [SGLang Structured Output](labs/lab_05_sglang_structured_output/) | Chapter 4.2 | 45 min |
| 06 | [Tensor Parallelism](labs/lab_06_tensor_parallelism/) | Chapter 5.1 | 60 min |
| 07 | [Ray Serve Deployment](labs/lab_07_ray_serve_deployment/) | Chapter 6.1 | 60 min |
| 08 | [EKS + KServe](labs/lab_08_eks_kserve_deployment/) | Chapter 6.2 | 90 min |
| 09 | [SageMaker Production](labs/lab_09_sagemaker_production/) | Chapter 6.3 | 60 min |
| 10 | [Benchmarking Suite](labs/lab_10_benchmarking_monitoring/) | Chapter 7.1 | 45 min |

**Hardware requirements:** Most labs run on a single GPU (g5.xlarge or equivalent). Labs 06 and 08 require multi-GPU instances.

---

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

1. [0.0 What is LLM Inference?](content/00_foundations/00.0_what_is_llm_inference/what_is_llm_inference.md) — 15 min
2. [0.1 Why LLM Inference is Different](content/00_foundations/00.1_why_llm_inference_is_different/why_llm_inference_is_different.md) — 20 min
3. [3.1 Quantization](content/03_optimization/03.1_quantization/quantization.md) — 20 min
4. [4.1 vLLM](content/04_engines/04.1_vllm/vllm.md) — 30 min
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

<p align="center">
  <strong>Built with ❤️ for the ML infrastructure community</strong>
</p>

<p align="center">
  <a href="#llm-inference-at-scale">Back to top ↑</a>
</p>
