# Why this roadmap exists

A structured self-study plan to prepare for running a 2-hour workshop covering the full LLM inference stack — from how a transformer generates tokens to production-grade serving with vLLM, SGLang, TensorRT-LLM, Ray, KServe, llm-d, and MoE.

---

# How to use this roadmap

- **Read the “8-layer map” once** to orient yourself.
- Then follow the **Phases (1 → 8)** in order.
- Each phase has:
  - **What to learn** (concepts)
  - **Hands-on** (what to build / measure)
  - **Primary resources** (minimal, high-signal)

---

# The 8-layer map (so you don’t miss anything)

1. **Transformer inference mechanics** — prefill vs decode, KV cache, attention variants
2. **GPU fundamentals for inference** — roofline, memory hierarchy, kernels, profiling intuition
3. **Memory engineering** — KV cache pressure, fragmentation, paging, eviction, VRAM “napkin math”
4. **Quantization & compression** — INT8/INT4/NF4/FP8, accuracy/perf tradeoffs, kernel implications
5. **Inference engines** — vLLM, SGLang, TensorRT-LLM (and when each wins)
6. **Distributed inference** — tensor/pipeline/expert parallelism, NCCL/collectives, interconnect ceilings
7. **Serving architecture** — Ray Serve, Kubernetes, KServe, llm-d; routing, autoscaling, rollouts
8. **Measurement & production ops** — TTFT/TBT/P95, workload replay, SLOs, capacity planning, cost

---

# What the workshop covers (aligned to the 8-layer map)

1. Transformer forward pass — token generation from first principles
2. Prefill vs decode — different bottlenecks, different hardware strategies
3. KV cache — what it stores, why it grows, why fragmentation hurts throughput
4. Attention variants — MHA, MQA, GQA, FlashAttention, PagedAttention
5. Quantization — INT8, INT4, NF4, FP8 tradeoffs
6. Speculative decoding — draft-verify loop, Medusa, EAGLE, n-gram
7. Batching strategies — static, continuous, chunked prefill
8. Inference engines — vLLM, SGLang, TensorRT-LLM (when to use which)
9. Model parallelism — tensor, pipeline, expert parallel
10. **MoE inference** — routing, expert parallelism, load balancing
11. Distributed serving with Ray — Ray Serve + vLLM, multi-node deployments
12. Cloud-native serving — KServe, llm-d on Kubernetes / EKS
13. Structured output & evaluation — guided decoding, benchmarking methodology
14. Edge deployment — quantization for mobile/edge, llama.cpp / Jetson / CoreML

---

# Phase 1 — Transformer & inference fundamentals (Layer 1)

> **Goal:** Explain token generation end-to-end from memory, without slides.

## What to learn

- Tokenization → embedding → attention → MLP → logits → sampling
- The autoregressive loop: one token per forward pass
- Prefill vs decode:
  - Prefill: prompt processed in one shot (often compute-bound)
  - Decode: one token at a time; repeated weight reads (often memory-bandwidth-bound)
- KV cache: what it stores and why it dominates memory at scale
- Why fragmentation hurts throughput and how paging helps

## Hands-on

- Implement a tiny transformer forward pass (enough to explain shapes + KV cache role).
- Draw (and narrate) the prefill vs decode split for a real workload.

## Resources

- **Video (essential):** [Karpathy — Neural Nets: Zero to Hero](https://youtube.com/@karpathy) (GPT episode minimum)
- **Video:** [Karpathy — Deep Dive into LLMs (3h31m)](https://youtube.com/watch?v=7xTGNNLPyMI)
- **Video:** [Vizuara AI YouTube](https://youtube.com/@vizuara)
- **Interactive:** [bbycroft.net/llm](http://bbycroft.net/llm)

---

# Phase 2 — GPU + performance fundamentals (Layers 2–3)

> **Goal:** Build “roofline thinking” and know what to profile first.

## What to learn

- Roofline model; arithmetic intensity; compute vs memory bound
- GPU memory hierarchy (registers/L1/L2/HBM) and why bandwidth dominates decode
- Kernel basics: fusion, launch overhead, occupancy (only what you need for inference)
- KV cache VRAM budgeting: weights + KV + overhead

## Hands-on

- Take a single model run and label the dominant bottleneck (HBM vs compute vs CPU/orchestration).
- Track VRAM usage as you scale: context length × batch × concurrency.

---

# Phase 3 — Core papers (Layer 1–3 vocabulary)

> **Goal:** Speak the standard inference engineering vocabulary.

| Paper                                     | What it solves                                                     | Link                                                 |
| ----------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------- |
| PagedAttention / vLLM (UC Berkeley, 2023) | KV cache fragmentation via OS-style memory paging                  | [arxiv 2309.06180](https://arxiv.org/abs/2309.06180) |
| FlashAttention (NeurIPS 2022)             | IO-aware attention — tiles Q/K/V in SRAM, eliminates N×N HBM reads | [arxiv 2205.14135](https://arxiv.org/abs/2205.14135) |
| FlashAttention-2 (2023)                   | Better warp partitioning                                           | [arxiv 2307.08691](https://arxiv.org/abs/2307.08691) |
| FlashAttention-3 (2024)                   | Async + FP8 for H100                                               | [arxiv 2407.08608](https://arxiv.org/abs/2407.08608) |
| Speculative Decoding (NeurIPS 2023)       | Draft model proposes k tokens; large model verifies in one pass    | [arxiv 2211.17192](https://arxiv.org/abs/2211.17192) |
| Medusa                                    | Multiple decoding heads for faster generation                      | [arxiv 2401.10774](https://arxiv.org/abs/2401.10774) |
| SGLang (Stanford)                         | RadixAttention + compiler-driven scheduling                        | [arxiv 2312.07104](https://arxiv.org/abs/2312.07104) |
| DeepSpeed-FastGen (Microsoft)             | SplitFuse for variable-length generation                           | [arxiv 2401.08671](https://arxiv.org/abs/2401.08671) |
| DeepSeek-V2 (MoE)                         | MoE routing + expert parallelism + economical inference            | [arxiv 2205.05198](https://arxiv.org/abs/2205.05198) |

**Curated paper list (500+ papers):** [github.com/xlite-dev/Awesome-LLM-Inference](http://github.com/xlite-dev/Awesome-LLM-Inference)

---

# Phase 4 — Inference engines deep dive (Layer 5)

> **Goal:** Know what each engine is optimized for and when to reach for which one.

## vLLM

The standard open-source choice. PagedAttention + continuous batching + FlashAttention-2/3. Tensor parallelism built in (`--tensor-parallel-size 8`). Supports speculative decoding (n-gram, EAGLE, Medusa), prefix caching, chunked prefill, multi-LoRA batching.

- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [Anatomy of vLLM (blog)](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- [Aleksa Gordić — Inside vLLM](https://www.aleksagordic.com/blog/vllm)

## SGLang

Often faster than vLLM for structured generation and multi-call programs. RadixAttention reuses KV cache across branching prompt trees.

- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [SGLang paper (Stanford)](https://arxiv.org/abs/2312.07104)

## TensorRT-LLM (NVIDIA)

Best raw throughput on NVIDIA GPUs, with compilation and tighter coupling to NVIDIA stack.

- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)

## When to use which

| Scenario                                               | Engine                   |
| ------------------------------------------------------ | ------------------------ |
| General open-source serving, fast iteration            | vLLM                     |
| Structured output, multi-step LLM programs             | SGLang                   |
| Max throughput on NVIDIA hardware, production-compiled | TensorRT-LLM             |
| Cost-optimised on AWS, avoiding GPU cost               | Inferentia2 + Neuron SDK |

---

# Phase 5 — Practical vLLM performance tuning (Layer 5 + Layer 8)

> **Goal:** A crisp “defaults leave throughput on the floor” segment + safe starter configs.

## Step 0: Make sure you’re on vLLM V1

V1 is meaningfully faster out of the box (async scheduling, chunked prefill defaults, better kernel fusions, torch.compile integration). If you’re on V0, upgrading is often the first win before touching knobs.

## The 6 knobs most teams never touch

1. `--max-num-batched-tokens`
   Default often ~2048 → **try 8192–32768** (largest throughput lever; defaults skew toward latency)
2. `--gpu-memory-utilization`
   Default ~0.90 → **try 0.95** (reclaims VRAM headroom)
3. `--max-num-seqs`
   Default: ~256 (V0) / ~1024 (V1) → **try 512–2048** (caps concurrency; bursty traffic silently queues)
4. `--enable-prefix-caching`
   Default: OFF → **turn ON** (free win if prompts/RAG chunks repeat)
5. `--enable-chunked-prefill`
   Default: OFF in V0; ON in V1 → **verify it’s ON**
6. CPU core allocation
   Rule of thumb: **≥ 2 + (#GPUs)** _physical_ cores (CPU starvation can make GPUs look “underutilized”)

## Two starter configs (then profile)

**Throughput-heavy (starting point):**
`--max-num-batched-tokens 16384 --gpu-memory-utilization 0.95 --enable-prefix-caching --enable-chunked-prefill`

**Latency-sensitive (starting point):**
`--max-num-batched-tokens 4096 --max-num-seqs 512 --enable-chunked-prefill`

**Caveat:** on very large models (70B+), KV cache pressure may force smaller practical values — benchmark with representative traffic.

---

# Phase 6 — Model parallelism & MoE (Layer 6)

> **Goal:** Understand how to scale beyond a single GPU and how MoE changes the serving problem.

## Parallelism types

- **Tensor parallelism:** split weight matrices across GPUs (vLLM: `--tensor-parallel-size N`)
- **Pipeline parallelism:** stage layers across GPUs (latency ↑, capacity ↑)
- **Data parallelism:** replicate model, split batch (simple if model fits per GPU)
- **Expert parallelism (MoE):** experts distributed across GPUs; tokens routed per-step

## MoE inference specifics

- Sparse activation: only k-of-N experts active per token (active params smaller than total params)
- Routing adds irregularity: load imbalance + all-to-all communication
- Practical bottlenecks: dispatch cost, imbalance, and network/interconnect ceilings
- Reference: [DeepSeek-V2 paper](https://arxiv.org/abs/2205.05198)

---

# Phase 7 — Distributed serving: Ray, KServe, llm-d (Layer 7)

> **Goal:** Go from “single box” to “org-scale serving”.

## Ray + Ray Serve

Ray is the distributed compute framework often used for multi-node deployments. Ray Serve adds HTTP routing, batching, and replica management.

- Guest lecture resource: Suman Debnath (AnyScale) + Seiji Eicher (vLLM & Ray Serve team) — covered in [Vizuara workshop](https://inference.vizuara.ai)
- [Ray Serve + vLLM docs](https://docs.ray.io/en/latest/serve/llm/vllm-deployment.html)

## KServe

Kubernetes-native model serving. Handles multi-model routing, canary deployments, autoscaling, and hardware abstraction (vLLM is the backend; KServe is the serving layer).

- [Cloud-Native AI Inference with KServe & llm-d](https://kserve.github.io/website/blog/cloud-native-ai-inference-kserve-llm-d)

## llm-d

Kubernetes controller purpose-built for LLM inference at scale. Focus areas include disaggregated prefill/decode and intelligent routing (often running vLLM as backend).

- [Production-Grade LLM Inference: KServe + llm-d + vLLM](https://llm-d.ai/blog/production-grade-llm-inference-at-scale-kserve-llm-d-vllm)

## The stack relationship

```jsx
User request
    ↓
KServe / llm-d  (routing, autoscaling, model management)
    ↓
Ray Serve       (replica mgmt, batching, HTTP)
    ↓
vLLM            (PagedAttention, continuous batching, GPU execution)
    ↓
FlashAttention / TensorRT kernels
```

---

# Phase 8 — Measurement, evaluation, and edge deployment (Layer 8 + Edge)

> **Goal:** Benchmark honestly, then optionally go edge.

## Performance measurement (don’t skip)

- Key metrics: TTFT, TBT, requests/s, tokens/s, P95 latency
- Throughput vs latency: optimize the right target
- Workload replay: representative prompts, output lengths, concurrency

## Structured output / guided decoding

Constrain generation to JSON schema, regex, or grammar. vLLM: `--guided-decoding-backend outlines`. SGLang has native structured generation.

## Edge deployment (optional)

- llama.cpp for CPU/mobile inference
- TensorRT-LLM on Jetson Orin Nano
- Apple: CoreML + on-device inference pipeline

---

# AWS experimentation path (optional but practical)

## Step 1 — EC2 g5.2xlarge + vLLM (~$1.20/hr)

NVIDIA A10G, 24GB VRAM. Run Llama 3.1 8B in FP16.

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 1 \
  --enable-prefix-caching
```

## Step 2 — Multi-GPU: p4d.24xlarge (8× A100)

Test tensor parallelism: `--tensor-parallel-size 8`. Run 70B models in FP16.

## Step 3 — SageMaker LMI containers

Managed endpoints with DJLServing + vLLM or TensorRT-LLM backend. Auto-scaling, no infra management.

## Step 4 — Inferentia2 (inf2 instances)

Compile with AWS Neuron SDK:

```bash
optimum-cli export neuron \
  --model meta-llama/Llama-3.1-8B \
  --task text-generation \
  --batch_size 1 \
  --sequence_length 1024 \
  --num_cores 8 \
  ./llama-3.1-8b-neuron/
```

- [Mixtral on SageMaker + Inferentia2](https://aws.amazon.com/blogs/machine-learning/optimizing-mixtral-8x7b-on-amazon-sagemaker-with-aws-inferentia2/)
- [High-perf Llama on Inferentia2 (PyTorch blog)](https://pytorch.org/blog/high-performance-llama/)

## Step 5 — EKS + vLLM + KServe + llm-d (production)

Full org-scale stack (routing/model mgmt + LLM-specific scheduling + inference backend).

---

# Self-study sequence (6 weeks)

| Week | Focus                                                | Done when…                                                                        |
| ---- | ---------------------------------------------------- | --------------------------------------------------------------------------------- |
| 1    | Transformer basics + prefill/decode + KV cache       | Can explain the full token-generation loop + prefill/decode split on a whiteboard |
| 2    | GPU/roofline + VRAM napkin math                      | Can predict the likely bottleneck for a workload and justify it                   |
| 3    | PagedAttention + vLLM internals                      | Can explain paging + blocks + why fragmentation kills throughput                  |
| 4    | FlashAttention + quantization + speculative decoding | Can explain when each helps (and when it doesn’t)                                 |
| 5    | Distributed inference + MoE                          | Can diagram tensor/pipeline/expert parallelism and key bottlenecks                |
| 6    | Ray/KServe/llm-d + measurement + dry run             | Full 2-hour rehearsal done; metrics story and tradeoffs are crisp                 |
