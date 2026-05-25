# Research Papers: LLM Inference at Scale (2025-2026)

> Critical analysis of recent papers from MLSys 2026, ICLR 2026, AISTATS 2026, and arXiv preprints relevant to the LLM Inference at Scale workshop.

---

## 1. Quantization & Compression

### 1.1 TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

| Field | Detail |
|-------|--------|
| **Authors** | Amir Zandieh (Google Research), Majid Daliri (NYU), Majid Hadian (Google DeepMind), Vahab Mirrokni (Google Research) |
| **Venue** | ICLR 2026 |
| **ArXiv** | https://arxiv.org/abs/2504.19874 |
| **Related** | PolarQuant (AISTATS 2026, arXiv:2502.02617), QJL (arXiv:2406.03482) |

**Core Innovation:**
- Data-oblivious, training-free vector quantization achieving near-optimal distortion rates (within ~2.7x of information-theoretic lower bounds)
- Randomly rotates input vectors to induce concentrated Beta distribution on coordinates, then applies optimal scalar quantizers per coordinate
- Two-stage: MSE quantizer + 1-bit QJL transform on residual → unbiased inner product estimation

**Key Results:**
- Absolute quality neutrality at **3.5 bits/channel**; marginal degradation at 2.5 bits
- Compresses KV cache to **3 bits** without training/fine-tuning and **no accuracy loss**
- 4-bit TurboQuant achieves **up to 8x speedup** over 32-bit unquantized keys on H100 GPUs
- Outperforms product quantization (PQ) and RabbiQ in nearest neighbor search recall

**Critical Assessment:**
- **Strengths**: Training-free (no calibration data needed), theoretically grounded, already integrated into vLLM, works across all bit-widths
- **Weaknesses**: Rotation overhead may matter for very short sequences; real-world gains depend on memory-bandwidth-bound regime
- **Workshop Relevance**: ★★★★★ — Directly extends Module 3 (Optimization Techniques) quantization section. Should be added as state-of-the-art KV cache quantization method alongside AWQ/GPTQ for weights.

---

### 1.2 PolarQuant (AISTATS 2026)

| Field | Detail |
|-------|--------|
| **ArXiv** | https://arxiv.org/abs/2502.02617 |
| **Relation** | Same research line as TurboQuant |

**Core Innovation:**
- Converts vectors to polar coordinates, eliminating normalization overhead
- Complementary to TurboQuant for scenarios where normalized representations are used

**Workshop Relevance**: ★★★☆☆ — Mention in advanced quantization section as alternative approach.

---

## 2. KV Cache Optimization

### 2.1 ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models

| Field | Detail |
|-------|--------|
| **Authors** | Akshat Ramachandran, Marina Neseem, Charbel Sakr, Rangharajan Venkatesan, Brucek Khailany, Tushar Krishna |
| **Venue** | ICLR 2026 (Oral) |
| **ArXiv** | https://arxiv.org/abs/2510.01290 |

**Core Innovation:**
- Identifies distinct thought types in Chain-of-Thought reasoning via attention sparsity patterns
- Applies hybrid quantization-eviction strategy: assigns token precision by thought importance
- Progressively evicts tokens from less critical thoughts

**Critical Assessment:**
- **Strengths**: Reasoning-aware (critical for o1/o3-style models), achieves significant compression without quality loss on CoT tasks
- **Weaknesses**: Requires attention pattern analysis overhead; may not generalize to non-CoT workloads
- **Workshop Relevance**: ★★★★★ — Essential for advanced content on reasoning model inference (DeepSeek-R1, o3, etc.)

---

### 2.2 Training Transformers for KV Cache Compressibility

| Field | Detail |
|-------|--------|
| **Published** | May 2026 |
| **ArXiv** | https://arxiv.org/abs/2605.05971 |

**Core Innovation:**
- Training-time approach making transformers inherently more compressible at KV cache level
- Addresses fundamental constraint: KV cache memory/decode-time costs scale linearly with prefix length
- Optimizes models during training for downstream compression

**Critical Assessment:**
- **Strengths**: Addresses root cause (model architecture) rather than post-hoc compression
- **Weaknesses**: Requires retraining; not applicable to existing models
- **Workshop Relevance**: ★★★☆☆ — Mention in future directions; relevant for teams training custom models

---

### 2.3 LeanKV: Unifying KV Cache Compression for Large Language Models

| Field | Detail |
|-------|--------|
| **ArXiv** | https://arxiv.org/abs/2412.03131 |
| **Implementation** | Built on vLLM |

**Key Results:**
- **3.0-5.0x compression** without accuracy loss (up to 11x with <5% loss)
- **1.9-6.9x throughput enhancement**
- Combines parallel KV compaction with differentiated memory management

**Critical Assessment:**
- **Strengths**: Production-ready (vLLM integration), unified framework, impressive throughput gains
- **Weaknesses**: Compression ratios vary significantly by model/task
- **Workshop Relevance**: ★★★★★ — Direct lab candidate; participants can benchmark LeanKV on vLLM

---

### 2.4 KVComp: High-Performance, LLM-Aware, Lossy Compression Framework

| Field | Detail |
|-------|--------|
| **Published** | Sep 2025 |
| **ArXiv** | https://arxiv.org/abs/2509.00579 |

**Core Innovation:**
- Lossy compression techniques specifically designed for KV cache data characteristics
- Co-designs compression algorithms with system architecture
- Maintains compatibility with growing KV cache

**Workshop Relevance**: ★★★★☆ — Good comparison point for KV cache compression landscape.

---

### 2.5 EchoKV: Efficient KV Cache Compression via Similarity-Based Reconstruction

| Field | Detail |
|-------|--------|
| **Published** | Mar 2026 |
| **ArXiv** | https://arxiv.org/abs/2603.22910 |

**Core Innovation:**
- On-demand transitions between standard and compressed inference
- Similarity-based reconstruction of evicted tokens
- Flexible compression that adapts to workload

**Workshop Relevance**: ★★★★☆ — Novel approach to dynamic compression; good for advanced module.

---

### 2.6 KV Policy (KVP): Learning to Evict from Key-Value Cache

| Field | Detail |
|-------|--------|
| **Published** | Feb 2026 |
| **ArXiv** | (2026 preprint) |

**Core Innovation:**
- Reframes KV cache eviction as a reinforcement learning problem
- Lightweight per-head RL agents trained on pre-computed generation traces
- Learns to rank tokens by importance for eviction decisions

**Critical Assessment:**
- **Strengths**: Learned policy outperforms heuristic eviction (H2O, StreamingLLM)
- **Weaknesses**: Training overhead for RL agents; per-model training required
- **Workshop Relevance**: ★★★★☆ — Excellent advanced topic showing ML-for-systems approach

---

### 2.7 Benchmarking Data-Dependent Low-Rank Compressibility of KV-Caches

| Field | Detail |
|-------|--------|
| **Published** | Feb 2026 |
| **ArXiv** | https://arxiv.org/abs/2602.05929 |

**Core Innovation:**
- First large-scale benchmark of KV-cache compressibility
- Principled evaluation framework for dynamic, data-aware compression
- Insights for data-centric model development

**Workshop Relevance**: ★★★☆☆ — Reference for benchmarking methodology in Module 7.

---

## 3. Speculative Decoding

### 3.1 Saguaro: Speculative Speculative Decoding (SSD)

| Field | Detail |
|-------|--------|
| **Authors** | Tanishq Kumar, Tri Dao, Avner May (Stanford, Princeton, Together AI) |
| **Venue** | ICLR 2026 |
| **ArXiv** | https://arxiv.org/abs/2603.03251 |

**Core Innovation:**
- Parallelizes speculation AND verification — while verification runs, draft model preemptively predicts verification outcomes and prepares next speculations
- Eliminates drafting overhead entirely when predictions hit
- **30% faster** than optimized speculative decoding baselines
- **Up to 5x faster** than autoregressive

**Critical Assessment:**
- **Strengths**: Elegant insight (speculate about speculation); significant speedup; from Tri Dao (FlashAttention author)
- **Weaknesses**: Requires careful synchronization; benefits diminish with low acceptance rates
- **Workshop Relevance**: ★★★★★ — Must-add to Module 3 speculative decoding section. Represents next generation of the technique.

---

### 3.2 Mirror Speculative Decoding (Mirror-SD)

| Field | Detail |
|-------|--------|
| **Authors** | Nikhil Bhendawade, Kumari Nishu, Arnav Kundu, Chris Bartels, Minsik Cho, Irina Belousova |
| **Published** | Oct 2025 |
| **ArXiv** | https://arxiv.org/abs/2510.13161 |

**Key Results:**
- Parallel rollouts across heterogeneous accelerators (GPU + NPU)
- Draft and target models simultaneously speculate for each other
- Speculative streaming (multi-token per step)
- **2.8x-5.8x wall-time speedups** on 14B-66B models
- **30% improvement** over EAGLE3

**Critical Assessment:**
- **Strengths**: Heterogeneous hardware utilization; impressive speedups on large models
- **Weaknesses**: Requires specific hardware configurations (GPU+NPU)
- **Workshop Relevance**: ★★★★☆ — Relevant for AWS deployments with mixed accelerators (GPU + Inferentia)

---

### 3.3 SpecKV: Adaptive Speculative Decoding with Compression-Aware Gamma Selection

| Field | Detail |
|-------|--------|
| **Authors** | Shikhar Shukla |
| **Published** | May 2026 |
| **ArXiv** | https://arxiv.org/abs/2605.02888 |

**Core Innovation:**
- Lightweight adaptive controller selecting speculation length γ per step
- Uses draft model confidence/entropy signals
- Profiles across 4 task types × 3 compression levels (FP16/INT8/NF4)
- **56% improvement** over fixed-γ=4 baseline with only 0.34ms overhead

**Critical Assessment:**
- **Strengths**: Practical, low-overhead, works with quantized models
- **Weaknesses**: Task-type profiling needed upfront
- **Workshop Relevance**: ★★★★★ — Bridges quantization + speculative decoding; perfect for advanced lab

---

### 3.4 Hierarchical Speculative Decoding

| Field | Detail |
|-------|--------|
| **Published** | Oct 2025 |
| **ArXiv** | https://arxiv.org/abs/2510.01336 |

**Core Innovation:**
- Multi-level draft hierarchy — cascading smaller models to improve draft quality while maintaining speed

**Workshop Relevance**: ★★★☆☆ — Mention in speculative decoding variants comparison.

---

## 4. Disaggregated Serving & Scheduling

### 4.1 DynaServe: Unified and Elastic Execution for Dynamic Disaggregated LLM Serving

| Field | Detail |
|-------|--------|
| **Authors** | Chaoyi Ruan, Yinhe Chen, Dongqi Tian, Yandong Shi, Yongji Wu, Jialin Li, Cheng Li |
| **Published** | Apr 2025 |
| **ArXiv** | https://arxiv.org/abs/2504.09285 |

**Core Innovation:**
- Micro-request abstraction splitting requests at arbitrary token boundaries
- Two-level scheduling (global + local) balances load across unified GPU instances
- **1.15x-3.07x serving capacity boost**
- **Up to 1.91x goodput improvement** over SOTA baselines

**Critical Assessment:**
- **Strengths**: Flexible granularity; works with heterogeneous GPUs; production-oriented
- **Weaknesses**: Scheduling complexity; potential for increased tail latency
- **Workshop Relevance**: ★★★★★ — Directly extends Module 6 (Production Serving) llm-d section

---

### 4.2 TaiChi: Unifying Prefill-Decode Aggregation and Disaggregation

| Field | Detail |
|-------|--------|
| **Authors** | Chao Wang, Pengfei Zuo, Zhangyu Chen, Yunkai Liang, Zhou Yu, Ming-Chang Yang |
| **Published** | Aug 2025 |

**Core Innovation:**
- Unifies PD aggregation and disaggregation with differentiated-capability GPU instances
- Three configurable sliders adapt to SLO regimes
- "Latency lending" — borrow decode capacity for prefill during bursts

**Critical Assessment:**
- **Strengths**: Unified framework (not either/or); SLO-aware; practical for mixed workloads
- **Weaknesses**: Configuration complexity; requires workload characterization
- **Workshop Relevance**: ★★★★★ — Advanced production architecture topic; extends llm-d discussion

---

### 4.3 ContextPilot: Fast Long-Context Inference via Context Reuse

| Field | Detail |
|-------|--------|
| **Authors** | Yinsicheng Jiang, Yeqi Huang, Liang Cheng, Cheng Deng, Xuan Sun, Luo Mai |
| **Venue** | MLSys 2026 |
| **ArXiv** | https://arxiv.org/abs/2511.03475 |
| **Code** | https://github.com/EfficientContext/ContextPilot |

**Core Innovation:**
- Context index identifies overlapping context blocks across users/turns
- Context ordering and de-duplication maximize KV-cache reuse
- Succinct context annotations preserve reasoning quality under reuse
- **Up to 3x prefill latency reduction** vs. state-of-the-art

**Critical Assessment:**
- **Strengths**: Open-sourced; addresses real multi-tenant scenario; preserves quality
- **Weaknesses**: Requires context indexing infrastructure; benefits depend on overlap
- **Workshop Relevance**: ★★★★★ — Extends prefix caching concept in Module 4; production-relevant for multi-tenant

---

### 4.4 FaaScale: Unlocking Fast LLM Scaling for Serverless Inference

| Field | Detail |
|-------|--------|
| **Venue** | MLSys 2026 |
| **ArXiv** | https://arxiv.org/abs/2502.09922 |

**Core Innovation:**
- Serverless LLM inference with fast scaling (cold start mitigation)
- Addresses the fundamental tension between serverless elasticity and LLM model loading times

**Workshop Relevance**: ★★★★☆ — Relevant for Module 6 production serving; serverless LLM pattern.

---

## 5. Long-Context & Attention

### 5.1 FlashAttention-3 (Reference)

| Field | Detail |
|-------|--------|
| **Authors** | Tri Dao et al. |
| **ArXiv** | https://arxiv.org/abs/2407.08608 |

Already covered in workshop Module 1. Key additions for 2026:
- FP8 support now production-ready on H100/H200
- Async execution pipeline fully integrated in vLLM V1
- New: FlashInfer library provides modular attention kernels

**Workshop Relevance**: ★★★★★ — Update existing FlashAttention section with FA3 production status.

---

## 6. Advanced Batching & Scheduling

### 6.1 Sarathi-Serve: Taming Throughput-Latency Tradeoff

| Field | Detail |
|-------|--------|
| **Authors** | Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nikita Kwatra, Bhargav Gulavani, Alexey Tumanov, Ramachandran Ramjee |
| **Venue** | OSDI 2024 |
| **ArXiv** | https://arxiv.org/abs/2403.02310 |

**Core Innovation:**
- **Chunked-prefills**: splits prefill requests into near-equal-sized chunks
- **Stall-free schedules**: adds new requests without pausing ongoing decodes
- Up to **2.6x throughput** on Mistral-7B (single A100), **6.9x on Falcon-180B** (8×A100) over Orca/vLLM

**Critical Assessment:**
- **Strengths**: Foundational work — chunked prefill is now default in vLLM V1. Proven at scale.
- **Weaknesses**: Fixed chunk sizes may not be optimal for all workloads; doesn't address KV cache pressure
- **Reproducibility**: ✅ Concepts integrated into vLLM, SGLang
- **Workshop Relevance**: ★★★★★ — Already referenced in Module 3; should cite as origin of chunked prefill

---

### 6.2 CONCUR: Congestion-Based Concurrency Control for Agentic Batch Inference

| Field | Detail |
|-------|--------|
| **Published** | Jan 2026 |
| **ArXiv** | https://arxiv.org/abs/2601.22705 |

**Core Innovation:**
- Addresses **KV-cache thrashing** in agentic batch workloads (multi-turn, tool-calling)
- Proactive agent-level admission control based on cache congestion signals
- Prevents throughput degradation before memory exhaustion
- **4.09x throughput** on Qwen3-32B, **1.9x on DeepSeek-V3**

**Critical Assessment:**
- **Strengths**: Addresses real production problem (agentic workloads); compatible with existing systems
- **Weaknesses**: Requires workload characterization; admission control adds latency for rejected requests
- **Failure Mode**: Over-aggressive admission control under bursty traffic → request starvation
- **Workshop Relevance**: ★★★★★ — Critical for agentic inference module; demonstrates why naive batching fails for agents

---

### 6.3 FairBatching: Fairness-Aware Batch Formation

| Field | Detail |
|-------|--------|
| **Published** | Oct 2025 |
| **ArXiv** | https://arxiv.org/abs/2510.14392 |

**Core Innovation:**
- Enforces fair resource allocation between prefill and decode tasks
- Adaptive batch capacity determination
- Reduces TTFT tail latency by **2.29x** while maintaining TPOT SLOs
- **20% single-node capacity improvement**, 54.3% cluster-level

**Critical Assessment:**
- **Strengths**: Addresses fairness (often ignored); improves tail latency significantly
- **Weaknesses**: "Fairness" definition is workload-dependent; may hurt throughput-optimized deployments
- **Workshop Relevance**: ★★★★☆ — Relevant for Module 7 (SLO management) and production operations

---

## 7. MoE Inference & DeepSeek

### 7.1 Quantifying the Double Penalty of MoE at Inference

| Field | Detail |
|-------|--------|
| **Published** | Mar 2026 |
| **ArXiv** | https://arxiv.org/abs/2603.08960 |

**Core Innovation:**
- Identifies **double penalty** for MoE during decoding:
  1. Expert routing fragments microbatches → reduced weight reuse
  2. Massive resident expert pools → reduced HBM headroom for KV cache
- Fine-grained MoEs suffer more; dense models dominate at long context
- Quantifies when MoE loses its efficiency advantage

**Critical Assessment:**
- **Strengths**: Rigorous analysis of MoE's hidden costs; challenges the "MoE is always cheaper" narrative
- **Weaknesses**: Analysis may not account for latest Wide-EP optimizations
- **Conflicting with**: Perplexity's claim that MoE achieves simultaneous higher throughput + lower latency with more GPUs
- **Resolution**: Both are correct — MoE wins with enough GPUs (Wide-EP), loses on constrained hardware
- **Workshop Relevance**: ★★★★★ — Must-add to Module 5 (MoE section); provides nuanced view

---

### 7.2 DeepSeek-V3 Architecture & Inference Optimizations

| Field | Detail |
|-------|--------|
| **ArXiv** | https://arxiv.org/abs/2512.02556 |
| **Architecture** | 671B total params, 37B active per token |

**Key Innovations for Inference:**
- **Multi-head Latent Attention (MLA)**: Reduces KV cache by **75%** vs standard GQA
- **DeepSeek Sparse Attention (DSA)**: Reduces computational complexity for long context
- **MoE with 256 experts, top-8 routing**: Fine-grained expert selection

**Production Performance (vLLM on GB300):**
- NVFP4 quantization + TP2: **7,360 tokens/GPU/second** (prefill-only)
- Mixed-context: **2,816 TGS**
- Per-user output: **230 TPS** (4x typical providers)

**Perplexity's Multi-Node Deployment:**
- MoE models achieve simultaneous higher throughput AND lower latency with more GPUs
- **10x faster** all-to-all communication for expert routing (optimized for AWS EFA)

**Critical Assessment:**
- **Strengths**: MLA is a genuine architectural innovation for KV cache efficiency; production-proven at scale
- **Weaknesses**: 256 experts creates extreme routing complexity; requires specialized infrastructure
- **Workshop Relevance**: ★★★★★ — Case study for Module 5 (MoE) and Module 2 (memory engineering with MLA)

---

## 8. Distillation for Inference Efficiency

### 8.1 SwiftKV: Fast Prefill-Optimized Inference via Knowledge-Preserving Transformation

| Field | Detail |
|-------|--------|
| **Authors** | Aurick Qiao, Zhewei Yao, Samyam Rajbhandari, Yuxiong He |
| **Published** | Oct 2024 (revised Jun 2025) |
| **ArXiv** | https://arxiv.org/abs/2410.03960 |

**Core Innovation:**
- Later layers skip prompt tokens by prefilling KV cache from earlier layers
- Reduces prefill FLOPs by **25-50%**
- **2x throughput**, **60% lower TTOT**
- 560 TFlops/GPU (16K tokens/s for Llama-3.1-70B)

**Critical Assessment:**
- **Strengths**: Open-sourced; directly targets prefill bottleneck; lightweight distillation procedure
- **Weaknesses**: Requires model-specific distillation; quality impact on long-context tasks unclear
- **Reproducibility**: ✅ Open-source
- **Workshop Relevance**: ★★★★★ — Novel optimization category (distillation for serving); extends Module 3

---

### 8.2 Llamba: Cross-Architecture Distillation (Transformer → Mamba)

| Field | Detail |
|-------|--------|
| **Authors** | Aviv Bick, Tobias Katsch, Nimit Sohoni, Arjun Desai, Albert Gu |
| **Published** | Feb 2025 |
| **ArXiv** | https://arxiv.org/abs/2502.14458 |

**Core Innovation:**
- Distills Llama-3.x into Mamba (recurrent) architecture using MOHAWK
- Produces 1B/3B/8B models with higher inference throughput and larger batch sizes
- Uses <0.1% of typical training data
- Eliminates KV cache entirely (recurrent state is fixed-size)

**Critical Assessment:**
- **Strengths**: Eliminates KV cache problem entirely; subquadratic inference; edge-friendly
- **Weaknesses**: Quality gap vs transformer on complex reasoning; limited model sizes available
- **Failure Mode**: Struggles with tasks requiring precise long-range retrieval (needle-in-haystack)
- **Workshop Relevance**: ★★★★☆ — Relevant for Module 10 (Edge) and future architectures discussion

---

### 8.3 Caprese: Scalable LLM Reasoning Acceleration with Low-rank Distillation

| Field | Detail |
|-------|--------|
| **Authors** | Harry Dong, Bilge Acun, Beidi Chen, Yuejie Chi |
| **Published** | May 2025 (revised Feb 2026) |
| **ArXiv** | https://arxiv.org/abs/2505.07861 |

**Core Innovation:**
- Recovers reasoning capabilities lost from pruning/sparsity via low-rank distillation
- Adds ~1% parameters via low-rank adapters in FFN blocks
- Cuts ~2B active parameters for 8-9B models
- **>16% time-to-next-token reduction**, up to 8.5% fewer generated tokens

**Critical Assessment:**
- **Strengths**: Distillation as post-hoc fix after aggressive optimization; practical for deployment
- **Weaknesses**: Requires reasoning-specific training data; benefits vary by task
- **Workshop Relevance**: ★★★★☆ — Bridges optimization and quality; relevant for advanced quantization discussion

---

## 9. NVIDIA Inference Stack

### 9.1 NVIDIA NIM (Inference Microservices)

**Key Facts:**
- Production-ready container with optimized engines (TensorRT-LLM, vLLM, SGLang)
- Auto-selects optimal engine and configuration per model/hardware
- Includes NVIDIA Dynamo for distributed orchestration
- Available on NGC catalog

**Critical Assessment:**
- **Strengths**: One-click deployment; auto-optimization; enterprise support
- **Weaknesses**: Vendor lock-in (NVIDIA GPUs only); opaque optimization decisions; licensing costs
- **Conflicting with**: Open-source vLLM/SGLang approach (more control, less convenience)
- **Workshop Relevance**: ★★★☆☆ — Mention as alternative to manual vLLM setup; not primary focus (workshop is AWS-centric)

---

## 10. Conflicting Approaches & When Each Wins

| Scenario | Winner | Loser | Why |
|----------|--------|-------|-----|
| Short sequences (<128 tokens) | Standard attention | TurboQuant | Rotation overhead exceeds savings |
| Long context + constrained memory | LeanKV/TurboQuant | Eviction-based (H2O) | Eviction loses information permanently |
| Reasoning models (CoT) | ThinKV | Uniform compression | Thought-adaptive preserves critical tokens |
| MoE on limited GPUs | Dense model | MoE | Double penalty (routing fragmentation + KV pressure) |
| MoE with Wide-EP (many GPUs) | MoE | Dense | Communication overhead amortized; active params low |
| High acceptance rate workloads | Saguaro | Standard spec decoding | Parallel speculation eliminates draft overhead |
| Low acceptance rate workloads | Standard autoregressive | Any speculation | Wasted draft computation exceeds savings |
| Heterogeneous hardware (GPU+NPU) | Mirror-SD | Single-device speculation | Parallel rollouts utilize both accelerators |
| Multi-tenant shared prefixes | ContextPilot | Per-request prefill | Prefix reuse eliminates redundant computation |
| Unique prompts (no sharing) | Standard prefill | ContextPilot | Indexing overhead with no reuse benefit |

---

## 11. Reproducibility Status

| Paper | Open Source | Framework Integration | Production-Ready |
|-------|-------------|----------------------|-----------------|
| TurboQuant | ✅ | vLLM | ✅ |
| ThinKV | ❓ | None yet | ❌ |
| LeanKV | ✅ | vLLM | ✅ |
| EchoKV | ❓ | None yet | ❌ |
| KV Policy | ❓ | None yet | ❌ |
| Saguaro | ❓ | Unknown | ❌ |
| Mirror-SD | ❓ | None yet | ❌ |
| SpecKV | ❓ | None yet | ❌ |
| ContextPilot | ✅ | GitHub | ⚠️ Research |
| DynaServe | ❓ | None yet | ❌ |
| TaiChi | ❓ | None yet | ❌ |
| FaaScale | ❓ | Unknown | ❌ |
| Sarathi-Serve | ✅ | vLLM (chunked prefill) | ✅ |
| SwiftKV | ✅ | Open-source | ⚠️ Research |
| Llamba | ✅ | HuggingFace | ⚠️ Research |
| CONCUR | ❓ | Compatible with vLLM | ❌ |
| DeepSeek-V3 | ✅ | vLLM, SGLang | ✅ |

**Key Insight**: Only ~5 of 17 papers have production-ready implementations. Workshop labs should focus on: TurboQuant, LeanKV, Sarathi-Serve (via vLLM chunked prefill), DeepSeek-V3, and SwiftKV.

---

## Summary: Top Papers by Workshop Impact

| Rank | Paper | Venue | Relevance | Module Impact | Reproducible? |
|------|-------|-------|-----------|---------------|---------------|
| 1 | TurboQuant | ICLR 2026 | ★★★★★ | Module 3 (Quantization) | ✅ vLLM |
| 2 | Saguaro (SSD) | ICLR 2026 | ★★★★★ | Module 3 (Speculative Decoding) | ❓ |
| 3 | DeepSeek-V3 MoE Double Penalty | arXiv 2026 | ★★★★★ | Module 5 (MoE) | ✅ Analysis |
| 4 | ThinKV | ICLR 2026 Oral | ★★★★★ | Module 3 (KV Cache), Advanced | ❓ |
| 5 | ContextPilot | MLSys 2026 | ★★★★★ | Module 4 (Prefix Caching) | ✅ GitHub |
| 6 | DynaServe | arXiv 2025 | ★★★★★ | Module 6 (Disaggregated) | ❓ |
| 7 | SwiftKV | arXiv 2025 | ★★★★★ | Module 3 (Distillation for Serving) | ✅ Open-source |
| 8 | CONCUR | arXiv 2026 | ★★★★★ | Advanced (Agentic Batching) | ❓ |
| 9 | TaiChi | arXiv 2025 | ★★★★★ | Module 6 (Production Architecture) | ❓ |
| 10 | SpecKV | arXiv 2026 | ★★★★★ | Module 3 (Adaptive Speculation) | ❓ |
| 11 | LeanKV | arXiv 2024 | ★★★★★ | Module 3 (KV Cache), Lab candidate | ✅ vLLM |
| 12 | Sarathi-Serve | OSDI 2024 | ★★★★★ | Module 3 (Chunked Prefill origin) | ✅ vLLM |
| 13 | FairBatching | arXiv 2025 | ★★★★☆ | Module 7 (SLO Fairness) | ❓ |
| 14 | Mirror-SD | arXiv 2025 | ★★★★☆ | Module 3 (Heterogeneous Spec) | ❓ |
| 15 | Llamba | arXiv 2025 | ★★★★☆ | Module 10 (Edge), Future Arch | ✅ HuggingFace |
| 16 | FaaScale | MLSys 2026 | ★★★★☆ | Module 6 (Serverless) | ❓ |
| 17 | Caprese | arXiv 2025 | ★★★★☆ | Module 3 (Quality Recovery) | ❓ |
