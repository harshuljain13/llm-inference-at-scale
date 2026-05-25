# Advanced Content Proposal: LLM Inference at Scale Workshop

> Based on critical analysis of 25+ research papers and 20+ industry blog posts from 2025-2026, this proposal outlines advanced content additions that align with and extend the existing workshop spec.

---

## Executive Summary

The LLM inference landscape has evolved significantly since the workshop was initially designed. Three major shifts demand new content:

1. **KV Cache as First-Class Optimization Target** — TurboQuant, ThinKV, LeanKV, and EchoKV represent a new generation of KV-specific optimizations beyond PagedAttention
2. **Disaggregated Serving Goes Production** — llm-d on AWS, DynaServe, TaiChi prove prefill/decode disaggregation is no longer research
3. **Speculative Decoding 2.0** — Saguaro, Mirror-SD, and SpecKV show speculation is becoming adaptive, parallel, and compression-aware

---

## Proposed Advanced Modules

### Advanced Module A: Next-Generation KV Cache Engineering

**Rationale**: Current Module 3 covers PagedAttention. The field has moved far beyond — KV cache compression, learned eviction, and training-time compressibility are now active production techniques.

**Duration**: 90 minutes | **Prerequisites**: Modules 1-3

#### Content Outline

1. **KV Cache Compression Taxonomy**
   - Quantization-based: TurboQuant (3-bit, training-free, near-optimal)
   - Eviction-based: KV Policy (RL-learned eviction), H2O, StreamingLLM
   - Hybrid: ThinKV (thought-adaptive quantization + eviction)
   - Reconstruction-based: EchoKV (similarity-based reconstruction)
   - Unified: LeanKV (parallel compaction + differentiated memory)

2. **TurboQuant Deep Dive**
   - Random rotation → Beta distribution → optimal scalar quantizers
   - Two-stage: MSE quantizer + QJL residual
   - 3-bit KV cache with zero accuracy loss
   - 8x speedup on H100 for attention computation
   - vLLM integration and configuration

3. **Reasoning-Aware Compression (ThinKV)**
   - Thought-type identification via attention sparsity
   - Differential precision assignment by thought importance
   - Progressive eviction of low-importance thoughts
   - Critical for o1/o3/DeepSeek-R1 style reasoning models

4. **Context Reuse at Scale (ContextPilot)**
   - Context indexing across users/turns
   - De-duplication and ordering for maximum KV reuse
   - 3x prefill latency reduction
   - Multi-tenant deployment patterns

5. **Hands-on Lab: KV Cache Compression Benchmark**
   - Compare: No compression vs TurboQuant vs LeanKV vs eviction
   - Measure: Memory savings, throughput gain, quality impact
   - Model: Llama 3.1 8B on g5.2xlarge

#### Alignment with Existing Spec
- Extends Requirement 4 (Optimization Techniques) acceptance criteria 2 (PagedAttention)
- Satisfies gap in Requirement 14 (FlashAttention) — attention-level optimizations
- Adds depth to Module 3 without restructuring

---

### Advanced Module B: Speculative Decoding 2.0

**Rationale**: Current Module 3 covers basic draft-verify, Medusa, EAGLE, n-gram. The field has advanced to parallel speculation, adaptive γ, heterogeneous hardware, and compression-aware speculation.

**Duration**: 60 minutes | **Prerequisites**: Module 3

#### Content Outline

1. **Evolution of Speculative Decoding**
   ```
   Gen 1: Draft-Verify (2022) → Fixed draft model, fixed γ
   Gen 2: EAGLE/Medusa (2024) → No separate draft model
   Gen 3: Saguaro/Mirror-SD (2026) → Parallel, adaptive, heterogeneous
   ```

2. **Saguaro: Speculative Speculative Decoding**
   - Insight: Speculate about the speculation outcome
   - While verification runs, draft model predicts acceptance and prepares next batch
   - Eliminates drafting overhead when predictions hit
   - 30% faster than optimized baselines; up to 5x vs autoregressive
   - From Tri Dao (FlashAttention author) — likely to be widely adopted

3. **Mirror-SD: Heterogeneous Parallel Speculation**
   - GPU + NPU parallel rollouts
   - Draft and target simultaneously speculate for each other
   - Speculative streaming (multi-token per step)
   - 2.8x-5.8x speedups on 14B-66B models
   - AWS relevance: GPU + Inferentia2 heterogeneous deployments

4. **SpecKV: Compression-Aware Adaptive Speculation**
   - Adaptive γ selection using confidence/entropy signals
   - Profiles across task types × compression levels
   - 56% improvement over fixed-γ with 0.34ms overhead
   - Bridges quantization and speculation — two optimizations that interact

5. **Production Speculative Decoding on AWS**
   - EAGLE on SageMaker (native LMI support)
   - Speculative decoding on Trainium/Inferentia2
   - When to use: decode-heavy workloads (short prompt, long output)
   - Configuration guide and performance expectations

6. **Hands-on Lab: Adaptive Speculative Decoding**
   - Deploy vLLM with EAGLE speculation
   - Benchmark fixed-γ vs adaptive-γ across workload types
   - Measure acceptance rates and speedup
   - Compare: No speculation vs EAGLE vs n-gram on SageMaker

#### Alignment with Existing Spec
- Extends Requirement 4 acceptance criteria 3 (speculative decoding)
- Adds depth beyond the existing variants table
- Connects to Requirement 10 (AWS-specific) with Trainium speculation

---

### Advanced Module C: Production Disaggregated Serving

**Rationale**: Current Module 6 introduces llm-d conceptually. AWS now officially supports it, Ray Serve implements it, and papers (DynaServe, TaiChi) provide the theory. This deserves dedicated advanced treatment.

**Duration**: 90 minutes | **Prerequisites**: Modules 5-6

#### Content Outline

1. **Why Disaggregate? The Economics**
   - Prefill: compute-bound → benefits from high FLOPS (H100, Trainium)
   - Decode: memory-bandwidth-bound → benefits from high BW/$ (Inferentia2)
   - Mixed workloads waste 30-50% of GPU capacity
   - Capital allocation framing (from Azure series)

2. **Architecture Patterns**
   - Pattern 1: Simple disaggregation (separate prefill/decode pools)
   - Pattern 2: DynaServe micro-requests (arbitrary split points)
   - Pattern 3: TaiChi unified (adaptive aggregation/disaggregation)
   - Pattern 4: Wide-EP for MoE (expert-level disaggregation)

3. **llm-d on AWS (Production)**
   - Official AWS architecture
   - KV cache transfer mechanisms (RDMA, EFA, shared memory)
   - Intelligent routing: request classification → pool assignment
   - Autoscaling prefill and decode pools independently
   - SageMaker and EKS deployment options

4. **Ray Serve Disaggregated Deployment**
   - Wide-EP configuration for MoE models
   - Custom routing for prefill/decode separation
   - Multi-node orchestration with Ray
   - 60% TTFT reduction with prefix-aware routing

5. **Cold Start Mitigation**
   - Model streaming (blob storage → GPU, bypassing disk)
   - 6x faster model loading (Run:AI Model Streamer pattern)
   - S3 → GPU streaming on AWS
   - Implications for autoscaling and spot recovery

6. **Serverless LLM Inference (FaaScale)**
   - Serverless elasticity vs model loading tension
   - Fast scaling techniques for LLM workloads
   - When serverless makes sense vs dedicated

7. **Hands-on Lab: Disaggregated Serving on AWS**
   - Deploy llm-d with separate prefill/decode pools
   - Configure KV cache transfer
   - Benchmark: aggregated vs disaggregated under mixed workloads
   - Measure: TTFT, throughput, GPU utilization per pool

#### Alignment with Existing Spec
- Extends Requirement 7 acceptance criteria 4 (llm-d disaggregated architecture)
- Extends Requirement 10 (AWS-specific) with production llm-d
- Adds cold start topic (gap in current spec)

---

### Advanced Module D: Agentic & Multi-Step Inference *(Optional/Future)*

**Rationale**: Agentic workloads (tool-calling, multi-step reasoning, multi-turn) have fundamentally different inference characteristics. Ray's blog on AI Agents + Azure's agentic workload discussion + CONCUR paper signal this is the next production challenge. *However, this is tangential to the core workshop spec and should be treated as optional future content.*

**Duration**: 60 minutes | **Prerequisites**: Modules 4, 6 | **Priority**: Phase 3 (Future)

#### Content Outline

1. **Agentic Inference Characteristics**
   - Multi-step: N inference calls per user request
   - Variable latency budget per step
   - Tool-calling: structured output → function execution → re-inference
   - Long context accumulation across steps
   - Branching: parallel tool calls, tree-of-thought

2. **Inference Patterns for Agents**
   - Streaming with early stopping (cancel if tool call detected)
   - Structured output for reliable tool calling (SGLang, guided decoding)
   - Prefix caching across agent steps (same system prompt + history)
   - Batching across agent instances (not just within one)

3. **Multi-LoRA for Agent Specialization**
   - Base model + specialized LoRA adapters per agent role
   - vLLM multi-LoRA serving (dozens of adapters, one GPU)
   - SageMaker multi-model endpoints
   - Dynamic adapter loading/unloading

4. **Serving Architecture for Agents**
   - Ray Serve multi-agent patterns
   - Request routing by agent type
   - Shared inference pool vs dedicated per-agent
   - Autoscaling based on agent step count, not just requests

5. **SLOs for Agentic Workloads**
   - End-to-end latency budget (sum of all steps)
   - Per-step TTFT targets (stricter for streaming steps)
   - Tool-call latency (inference + execution)
   - Cost per agent interaction (multiple inference calls)

#### Alignment with Existing Spec
- Extends Requirement 7 (Production Serving) with agentic patterns
- Extends Requirement 8 (Measurement) with agentic SLOs
- Connects to Requirement 13 (Structured Output) for tool calling

---

## Proposed Updates to Existing Modules

### NEW: Advanced Module E: MoE Inference & Distillation for Serving

**Rationale**: DeepSeek-V3's dominance, the MoE double penalty paper, and distillation-for-serving papers (SwiftKV, Llamba, Caprese) represent a new optimization category not covered in the current spec.

**Duration**: 75 minutes | **Prerequisites**: Modules 3, 5

#### Content Outline

1. **The MoE Inference Reality Check**
   - Double penalty: routing fragmentation + KV cache pressure (arXiv:2603.08960)
   - When MoE wins vs when dense wins (GPU count threshold)
   - DeepSeek-V3 case study: MLA (75% KV reduction) + 256 experts
   - Wide-EP: distributing experts across many GPUs (Anyscale/Ray)
   - Perplexity's 10x faster all-to-all communication

2. **Distillation as an Inference Optimization**
   - SwiftKV: skip prefill in later layers (2x throughput, 60% lower TTOT)
   - Caprese: recover reasoning quality after pruning via low-rank distillation
   - Llamba: cross-architecture distillation (Transformer → Mamba, eliminates KV cache)
   - When to distill vs when to quantize vs when to speculate

3. **Production MoE Deployment**
   - vLLM + DeepSeek-V3 on GB300: 7,360 tokens/GPU/s
   - Multi-node MoE on AWS (Perplexity approach)
   - Expert caching and prefetching strategies
   - Cost comparison: MoE vs dense at various scales

4. **Hands-on Lab: MoE vs Dense Inference Comparison**
   - Deploy Mixtral-8x7B with different TP configurations
   - Measure: throughput, latency, GPU utilization per expert
   - Compare with dense model of similar active params

#### Alignment with Existing Spec
- Extends Requirement 6 (Scaling) acceptance criteria 3 (MoE inference)
- Adds depth to Module 5 MoE section with production data
- New optimization category (distillation) not in current spec

---

### Updated Module Priorities

| Priority | Module | Confidence | Reproducible Labs? |
|----------|--------|------------|-------------------|
| **Phase 1** | Quick wins (update existing) | High | Yes (vLLM integrated) |
| **Phase 2a** | Module A: KV Cache Engineering | High | Yes (TurboQuant, LeanKV in vLLM) |
| **Phase 2b** | Module B: Speculative Decoding 2.0 | High | Partial (EAGLE on SageMaker) |
| **Phase 2c** | Module C: Disaggregated Serving | High | Yes (llm-d on AWS) |
| **Phase 2d** | Module E: MoE & Distillation | Medium | Partial (DeepSeek on vLLM) |
| **Phase 3** | Module D: Agentic Inference | Low | Limited (CONCUR not integrated) |

---

### Module 3: Optimization Techniques — Updates

| Section | Current Content | Proposed Addition |
|---------|----------------|-------------------|
| Quantization | AWQ, GPTQ, INT8, FP8 | Add TurboQuant (3-bit KV cache, training-free) |
| Speculative Decoding | Draft-verify, Medusa, EAGLE, n-gram | Add Saguaro (parallel speculation), SpecKV (adaptive γ) |
| PagedAttention | OS-style paging | Add LeanKV (unified compression), ContextPilot (prefix reuse) |
| Decision Matrix | Static table | Add compression-aware speculation row |

### Module 5: Scaling and Distribution — Updates

| Section | Current Content | Proposed Addition |
|---------|----------------|-------------------|
| MoE Inference | Basic routing, load balancing | Add Wide-EP (Ray/Anyscale), expert-level disaggregation |
| Interconnect | NVLink, EFA comparison | Add KV cache transfer bandwidth requirements for disaggregation |

### Module 6: Production Serving — Updates

| Section | Current Content | Proposed Addition |
|---------|----------------|-------------------|
| Ray Serve | Basic deployment | Add 88% latency reduction architecture, custom routing |
| llm-d | Conceptual diagram | Add AWS production deployment, DynaServe micro-requests |
| Cold Start | Not covered | Add model streaming, 6x faster loading |
| Autoscaling | Basic HPA | Add capacity-aware fallback (SageMaker), custom metrics |

### Module 7: Measurement & Operations — Updates

| Section | Current Content | Proposed Addition |
|---------|----------------|-------------------|
| SLOs | Chat, batch targets | Add voice (<200ms TTFT), agentic (per-step budgets) |
| Capacity Planning | Basic methodology | Add capital allocation framing (Azure series) |
| Troubleshooting | Common issues table | Add cold start diagnosis, KV cache pressure detection |

### Module 8: AWS Deep Dive — Updates

| Section | Current Content | Proposed Addition |
|---------|----------------|-------------------|
| Inferentia/Trainium | Basic Neuron SDK | Add speculative decoding on Trainium, multi-node (Rufus case study) |
| SageMaker | LMI deployment | Add multi-LoRA serving, EAGLE speculation, capacity-aware fallback |
| llm-d | Not detailed | Add full production deployment guide |
| Case Studies | None | Add Rufus Prime Day case study |

---

## Implementation Priority

### Phase 1: Quick Wins (Update existing modules)
1. Add TurboQuant to Module 3 quantization section
2. Add Saguaro/SpecKV to Module 3 speculative decoding
3. Update Module 6 Ray Serve with new performance numbers
4. Add Rufus case study to Module 8
5. Add cold start section to Module 6

### Phase 2: New Advanced Modules
1. Advanced Module A: Next-Gen KV Cache Engineering
2. Advanced Module B: Speculative Decoding 2.0
3. Advanced Module C: Production Disaggregated Serving

### Phase 3: Future-Looking Content
1. Advanced Module D: Agentic & Multi-Step Inference
2. Training for compressibility (emerging research)
3. Serverless LLM inference patterns

---

## References (New)

### Papers
| Paper | Venue | ArXiv |
|-------|-------|-------|
| TurboQuant | ICLR 2026 | arxiv:2504.19874 |
| ThinKV | ICLR 2026 (Oral) | arxiv:2510.01290 |
| Saguaro (SSD) | ICLR 2026 | arxiv:2603.03251 |
| ContextPilot | MLSys 2026 | arxiv:2511.03475 |
| FaaScale | MLSys 2026 | arxiv:2502.09922 |
| DynaServe | 2025 | arxiv:2504.09285 |
| TaiChi | 2025 | (preprint) |
| Mirror-SD | 2025 | arxiv:2510.13161 |
| SpecKV | 2026 | arxiv:2605.02888 |
| LeanKV | 2024/2025 | arxiv:2412.03131 |
| EchoKV | 2026 | arxiv:2603.22910 |
| KV Policy | 2026 | (preprint) |
| PolarQuant | AISTATS 2026 | arxiv:2502.02617 |

### Industry Posts
| Post | Source | Date |
|------|--------|------|
| Disaggregated Inference with llm-d | AWS | Apr 2026 |
| Ray Serve 88% Latency + 11.1x Throughput | Anyscale | Mar 2026 |
| Wide-EP + Disaggregated Serving | Anyscale | Dec 2025 |
| Enterprise Inference Series (3-part) | Azure | Dec 2025 - May 2026 |
| Rufus Multi-Node Trainium | AWS | Aug 2025 |
| Cold Start Elimination (6x) | Azure | May 2026 |
| EAGLE on SageMaker | AWS | May 2025 |
| Multi-LoRA on SageMaker | AWS | Mar 2026 |
| Speculative Decoding on Trainium | AWS | Apr 2026 |
| Custom Routing 60% TTFT | Anyscale | Sep 2025 |
