<p align="center">
  <img src="assets/banner.svg" alt="LLM Inference at Scale" width="100%">
</p>

<p align="center">
  <strong>The definitive guide to serving large language models in production.</strong>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-table-of-contents">Contents</a> •
  <a href="#-key-equations">Equations</a> •
  <a href="#-contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/vLLM-0.8+-orange.svg" alt="vLLM">
  <img src="https://img.shields.io/badge/SGLang-0.4+-purple.svg" alt="SGLang">
  <img src="https://img.shields.io/badge/TensorRT--LLM-0.15+-76B900.svg" alt="TensorRT-LLM">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.0+-76B900.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/license-All%20Rights%20Reserved-red.svg" alt="All Rights Reserved">
</p>

---

## Why This Exists

Serving an LLM is not like serving a traditional ML model. With traditional ML, you send a request, the model runs one forward pass, and you get a result. Fixed time. Fixed memory. Simple.

With LLMs, every request is different. A short reply takes 100ms. A long one takes 10 seconds. The model generates one word at a time, and each word requires reading the entire model from memory again. The longer the conversation, the more GPU memory it consumes. There is no fixed cost per request.

This makes LLM inference expensive, unpredictable, and hard to scale.

We wrote this handbook because the knowledge to solve these problems exists, but it is scattered across research papers, blog posts, source code comments, and tribal knowledge. No single resource connected the full picture: from how GPU memory works, to why decode is slow, to how production systems like vLLM actually solve it.

This is that resource.

---

## 🚀 Quick Start

```bash
git clone https://github.com/harshuljain13/llm-inference-at-scale.git
cd llm-inference-at-scale
pip install -e .
```

Start reading with [Chapter 00: The Transformer at Inference Time](content/00_transformer_at_inference_time/00.1_transformer_architecture/transformer_architecture.md).

---

## 📚 Table of Contents

> 12 chapters, 59 modules. Each module is a focused 8-10 minute read with a companion lab.

### Part I: Foundations

**Chapter 00: The Transformer at Inference Time** (4 modules)

| # | Module | Path |
|---|--------|------|
| 0.1 | Transformer Architecture | [transformer_architecture.md](content/00_transformer_at_inference_time/00.1_transformer_architecture/transformer_architecture.md) |
| 0.2 | What Happens During Inference | [what_happens_during_inference.md](content/00_transformer_at_inference_time/00.2_what_happens_during_inference/what_happens_during_inference.md) |
| 0.3 | Attention and KV Cache | [attention_and_kv_cache.md](content/00_transformer_at_inference_time/00.3_attention_and_kv_cache/attention_and_kv_cache.md) |
| 0.4 | Why LLM Inference is Different | [why_llm_inference_is_different.md](content/00_transformer_at_inference_time/00.4_why_llm_inference_is_different/why_llm_inference_is_different.md) |

**Chapter 01: GPU Hardware for Inference** (2 modules)

| # | Module | Path |
|---|--------|------|
| 1.1 | GPU Memory Hierarchy | [gpu_memory.md](content/01_gpu_hardware/01.1_gpu_memory_hierarchy/gpu_memory.md) |
| 1.2 | The Roofline Model | [roofline_fundamentals.md](content/01_gpu_hardware/01.2_roofline_model/roofline_fundamentals.md) |

**Chapter 02: Sizing and Serving** (3 modules)

| # | Module | Path |
|---|--------|------|
| 2.1 | Capacity Planning | [capacity_planning.md](content/02_sizing_and_serving/02.1_capacity_planning/capacity_planning.md) |
| 2.2 | Batch Size and Throughput | [batch_size_and_throughput.md](content/02_sizing_and_serving/02.2_batch_size_and_throughput/batch_size_and_throughput.md) |
| 2.3 | Instance Selection | [instance_selection.md](content/02_sizing_and_serving/02.3_instance_selection/instance_selection.md) |

### Part II: Optimizations

**Chapter 03: Attention Variants** (7 modules)

| # | Module | Path |
|---|--------|------|
| 3.1 | Multi-Head Attention (MHA) | [mha.md](content/03_attention_variants/03.1_mha/mha.md) |
| 3.2 | MQA and GQA | [mqa_gqa.md](content/03_attention_variants/03.2_mqa_gqa/mqa_gqa.md) |
| 3.3 | GQA Deep Dive | [gqa_deep_dive.md](content/03_attention_variants/03.3_gqa_deep_dive/gqa_deep_dive.md) |
| 3.4 | Multi-Latent Attention (MLA) | [multi_latent_attention.md](content/03_attention_variants/03.4_mla/multi_latent_attention.md) |
| 3.5 | FlashAttention: Why Standard Attention is Slow | [flash_attention_problem.md](content/03_attention_variants/03.5_flash_attention_problem/flash_attention_problem.md) |
| 3.6 | FlashAttention: The Algorithm | [flash_attention_algorithm.md](content/03_attention_variants/03.6_flash_attention_algorithm/flash_attention_algorithm.md) |
| 3.7 | FlashAttention: In Practice | [flash_attention_practice.md](content/03_attention_variants/03.7_flash_attention_practice/flash_attention_practice.md) |

**Chapter 04: KV Cache Engineering** (5 modules)

| # | Module | Path |
|---|--------|------|
| 4.1 | PagedAttention | [paged_attention.md](content/04_kv_cache_engineering/04.1_paged_attention/paged_attention.md) |
| 4.2 | KV Cache Compression | [kv_cache_compression.md](content/04_kv_cache_engineering/04.2_kv_cache_compression/kv_cache_compression.md) |
| 4.3 | Smart KV Caching | [smart_kv_caching.md](content/04_kv_cache_engineering/04.3_smart_kv_caching/smart_kv_caching.md) |
| 4.4 | LMCache | [lmcache.md](content/04_kv_cache_engineering/04.4_lmcache/lmcache.md) |
| 4.5 | Prefix Caching | [prefix_caching.md](content/04_kv_cache_engineering/04.5_prefix_caching/prefix_caching.md) |

**Chapter 05: Optimization Techniques** (6 modules)

| # | Module | Path |
|---|--------|------|
| 5.1 | Quantization | [quantization.md](content/05_optimization/04.1_quantization/quantization.md) |
| 5.2 | TurboQuant | [turboquant.md](content/05_optimization/04.2_turboquant/turboquant.md) |
| 5.3 | Continuous Batching | [continuous_batching.md](content/05_optimization/04.3_continuous_batching/continuous_batching.md) |
| 5.4 | Speculative Decoding | [speculative_decoding.md](content/05_optimization/04.4_speculative_decoding/speculative_decoding.md) |
| 5.5 | Chunked Prefill | [chunked_prefill.md](content/05_optimization/04.5_chunked_prefill/chunked_prefill.md) |
| 5.6 | Inference-Time Compute | [inference_time_compute.md](content/05_optimization/04.6_inference_time_compute/inference_time_compute.md) |

### Part III: Engines and Scaling

**Chapter 06: Inference Engines** (4 modules)

| # | Module | Path |
|---|--------|------|
| 6.1 | vLLM | [vllm.md](content/06_engines/05.1_vllm/vllm.md) |
| 6.2 | SGLang | [sglang.md](content/06_engines/05.2_sglang/sglang.md) |
| 6.3 | TensorRT-LLM | [tensorrt_llm.md](content/06_engines/05.3_tensorrt_llm/tensorrt_llm.md) |
| 6.4 | NVIDIA Dynamo | [nvidia_dynamo.md](content/06_engines/05.4_nvidia_dynamo/nvidia_dynamo.md) |

**Chapter 07: Scaling** (3 modules)

| # | Module | Path |
|---|--------|------|
| 7.1 | Tensor Parallelism | [tensor_parallelism.md](content/07_scaling/06.1_tensor_parallelism/tensor_parallelism.md) |
| 7.2 | Mixture-of-Experts Inference | [moe_inference.md](content/07_scaling/06.2_moe_inference/moe_inference.md) |
| 7.3 | Distillation for Serving | [distillation.md](content/07_scaling/06.3_distillation/distillation.md) |

### Part IV: Production

**Chapter 08: Serving Infrastructure** (7 modules)

| # | Module | Path |
|---|--------|------|
| 8.1 | Ray Serve | [ray_serve.md](content/08_serving/07.1_ray_serve/ray_serve.md) |
| 8.2 | EKS and KServe | [eks_kserve.md](content/08_serving/07.2_eks_kserve/eks_kserve.md) |
| 8.3 | SageMaker | [sagemaker.md](content/08_serving/07.3_sagemaker/sagemaker.md) |
| 8.4 | Disaggregated Serving | [disaggregated_serving.md](content/08_serving/07.4_disaggregated_serving/disaggregated_serving.md) |
| 8.5 | Cold Start Optimization | [cold_start.md](content/08_serving/07.5_cold_start/cold_start.md) |
| 8.6 | Cache-Aware Routing | [cache_aware_routing.md](content/08_serving/07.6_cache_aware_routing/cache_aware_routing.md) |
| 8.7 | Kubernetes Inference Infrastructure | [kubernetes_inference_infrastructure.md](content/08_serving/07.7_kubernetes_inference_infrastructure/kubernetes_inference_infrastructure.md) |

**Chapter 09: Operations** (6 modules)

| # | Module | Path |
|---|--------|------|
| 9.1 | Benchmarking and Metrics | [benchmarking.md](content/09_operations/08.1_benchmarking/benchmarking.md) |
| 9.2 | Structured Output and Guided Decoding | [structured_output.md](content/09_operations/08.2_structured_output/structured_output.md) |
| 9.3 | Edge Deployment | [edge_deployment.md](content/09_operations/08.3_edge_deployment/edge_deployment.md) |
| 9.4 | Inference Metrics and Monitoring | [inference_metrics.md](content/09_operations/08.4_inference_metrics/inference_metrics.md) |
| 9.5 | Multi-Region KV Locality | [multi_region_kv_locality.md](content/09_operations/08.5_multi_region_kv_locality/multi_region_kv_locality.md) |
| 9.6 | Custom Silicon | [custom_silicon.md](content/09_operations/08.6_custom_silicon/custom_silicon.md) |

**Chapter 10: Production Stories** (3 modules)

| # | Module | Path |
|---|--------|------|
| 10.1 | Meta's Inference Platform | [meta_inference_platform.md](content/10_production_stories/09.1_meta_inference_platform/meta_inference_platform.md) |
| 10.2 | Databricks Multi-Tenant Serving | [databricks_multi_tenant.md](content/10_production_stories/09.2_databricks_multi_tenant/databricks_multi_tenant.md) |
| 10.3 | Mixed Workload Management | [mixed_workload_management.md](content/10_production_stories/09.3_mixed_workload_management/mixed_workload_management.md) |

**Chapter 11: System Designs** (5 modules)

| # | Module | Path |
|---|--------|------|
| 11.1 | ChatGPT-Scale Chatbot | [chatgpt_scale_chatbot.md](content/11_system_designs/10.1_chatgpt_scale_chatbot/chatgpt_scale_chatbot.md) |
| 11.2 | Code Copilot | [code_copilot.md](content/11_system_designs/10.2_code_copilot/code_copilot.md) |
| 11.3 | Enterprise RAG | [enterprise_rag.md](content/11_system_designs/10.3_enterprise_rag/enterprise_rag.md) |
| 11.4 | Multi-Model Gateway | [multi_model_gateway.md](content/11_system_designs/10.4_multi_model_gateway/multi_model_gateway.md) |
| 11.5 | Agentic Workload | [agentic_workload.md](content/11_system_designs/10.5_agentic_workload/agentic_workload.md) |

**Total: 12 chapters, 55 modules.**

---


## 🤝 Contributing

Contributions are welcome. This is a living document.

- **Fix errors** — typos, outdated information, incorrect formulas
- **Improve clarity** — better explanations, additional examples
- **Add content** — new modules, labs, or reference materials

Fork the repo, create a branch, submit a PR. To report errors, open a GitHub Issue.

---

## 👤 About the Author

**Harshul Jain** is a Senior ML Infrastructure Engineer specializing in real-time ML systems, feature stores, and LLM serving infrastructure. He builds and operates ML platforms serving millions of users, mentors 300+ engineers through an eMentoring program, and is a recurring speaker at ML infrastructure conferences.

- GitHub: [@harshuljain13](https://github.com/harshuljain13)
- Newsletter: [The Engineer's Digest](https://harshuljain.substack.com)

---

## ⚠️ Disclaimer

The views, techniques, and opinions expressed in this handbook are solely those of the author and do not represent the views of any employer or affiliated organization. No proprietary or confidential information has been included. All content is based on publicly available research, open-source tooling, and independent analysis.

---

## 📄 License

© 2026 Harshul Jain. All rights reserved.

No part of this work may be reproduced, distributed, modified, or used without prior written permission from the author. To request permission, open a GitHub Issue.

---

## 🙏 Acknowledgments

This handbook builds on the work of many researchers and engineers:

- The [vLLM](https://github.com/vllm-project/vllm) team for PagedAttention and continuous batching
- The [SGLang](https://github.com/sgl-project/sglang) team for RadixAttention
- Tri Dao for [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- The authors of foundational papers: Attention Is All You Need, GQA, Medusa, EAGLE, and many others

---

<p align="center">
  <strong>Built with care for the ML infrastructure community</strong>
</p>
