# 6.3 Knowledge Distillation for Serving

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harshuljain13/llm-inference-at-scale/blob/master/content/07_scaling/06.3_distillation/lab.ipynb)
[![Open In Molab](https://img.shields.io/badge/Open%20in-Molab-blue?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCI+PHJlY3Qgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSIjMjU2M2ViIiByeD0iNCIvPjx0ZXh0IHg9IjUiIHk9IjE3IiBmb250LXNpemU9IjEyIiBmaWxsPSJ3aGl0ZSI+TW88L3RleHQ+PC9zdmc+)](https://molab.marimo.io/github/harshuljain13/llm-inference-at-scale/blob/master/content/07_scaling/06.3_distillation/lab.ipynb)

Knowledge distillation enables 2-10x model compression with less than 5% quality loss, making it the most powerful technique for reducing serving costs without sacrificing capability. Unlike quantization (which preserves architecture) or pruning (which removes weights), distillation transfers learned behavior from a large teacher into a smaller, faster student.

## Why Distillation Matters for Inference

Quantization and pruning have diminishing returns below 4-bit precision. Distillation breaks through this wall by changing what the model computes rather than how precisely it computes.

```mermaid
flowchart LR
    subgraph Teacher["Teacher (70B, FP16)"]
        T1[Full Knowledge]
    end
    subgraph Student["Student (7B, optimized)"]
        S1[Distilled Knowledge]
    end
    Teacher -->|"Soft labels + hidden states"| Student
    Student -->|"2-10x faster<br/>< 5% quality loss"| Deploy[Production Serving]
    style Teacher fill:#ffe4e6,stroke:#000,color:#000
    style Student fill:#dcfce7,stroke:#000,color:#000
    style Deploy fill:#dbeafe,stroke:#000,color:#000
```

Three modern distillation strategies target different serving bottlenecks: SwiftKV skips redundant prefill computation, Caprese recovers quality after aggressive compression, and Llamba eliminates the KV cache entirely through cross-architecture transfer.

## SwiftKV: 2x Throughput via Layer Skipping

SwiftKV observes that later transformer layers contribute diminishing returns during prefill. By distilling knowledge from early layers into later layers' KV projections, it skips computation in the bottom half of the network during prefill.

```mermaid
flowchart LR
    subgraph Full["Standard Prefill (80 layers)"]
        F1["Layer 0-79:<br/>All compute Q,K,V"]
    end
    subgraph Swift["SwiftKV Prefill"]
        S1["Layer 0-39:<br/>Full compute"] --> S2["Layer 40-79:<br/>KV from Layer 39<br/>Only Q + Attention"]
    end
    Full -->|"100% FLOPs"| R1["Baseline<br/>Throughput"]
    Swift -->|"50% FLOPs"| R2["2x Throughput<br/>60% lower TTFT"]
    style Full fill:#ffe4e6,stroke:#000,color:#000
    style Swift fill:#dcfce7,stroke:#000,color:#000
    style R2 fill:#ccfbf1,stroke:#000,color:#000
```

Results on Llama-3.1-70B: 2x throughput improvement, 60% lower TTFT, 560 TFLOPs/GPU (16K tokens/s), with minimal quality degradation on standard benchmarks. The decode phase remains unchanged since it uses the already-cached KV.

## Caprese: Quality Recovery After Compression

When aggressive pruning or quantization drops reasoning quality by 5-10%, Caprese recovers most of it through lightweight low-rank distillation. It adds LoRA adapters (rank=64, roughly 1% extra parameters) and distills from the original model on reasoning data.

```mermaid
flowchart LR
    A["Original 8B<br/>MMLU: 65%"] -->|"Prune to 6B"| B["Pruned<br/>MMLU: 58%"]
    B -->|"LoRA distillation<br/>8 GPU-hours"| C["Recovered 6.08B<br/>MMLU: 63%"]
    C -->|"Result"| D[">16% faster TTNT<br/>8.5% fewer tokens"]
    style A fill:#dbeafe,stroke:#000,color:#000
    style B fill:#ffe4e6,stroke:#000,color:#000
    style C fill:#dcfce7,stroke:#000,color:#000
    style D fill:#ccfbf1,stroke:#000,color:#000
```

The key insight: distillation recovers 70%+ of lost quality at only 1% parameter overhead, making aggressive compression practical for production.

## Llamba: Cross-Architecture Distillation

The most radical approach distills a Transformer into a Mamba (SSM) architecture that eliminates the KV cache entirely. Mamba uses a fixed-size recurrent state regardless of sequence length: O(1) memory per token vs. O(n) for Transformers.

```mermaid
flowchart LR
    subgraph Trans["Transformer (Teacher)"]
        T1["KV Cache: 256 MB @ 1K<br/>2 GB @ 8K<br/>32 GB @ 128K"]
    end
    subgraph Mamba["Mamba (Student)"]
        M1["Fixed State: ~16 MB<br/>Any sequence length<br/>O(1) memory/token"]
    end
    Trans -->|"MOHAWK distillation<br/>< 0.1% pretraining data"| Mamba
    Mamba --> R["3-5x speedup @ long seq<br/>95-97% quality retained"]
    style Trans fill:#ffe4e6,stroke:#000,color:#000
    style Mamba fill:#dcfce7,stroke:#000,color:#000
    style R fill:#ccfbf1,stroke:#000,color:#000
```

Tradeoffs: eliminates KV cache and achieves subquadratic inference, but has a 3-5% quality gap on complex reasoning and struggles with precise long-range retrieval. Best suited for edge deployment and high-batch long-context serving.

## Technique Comparison

| Technique | Throughput Gain | Quality Impact | Training Cost | Best For |
|-----------|----------------|----------------|---------------|----------|
| SwiftKV | 2x | < 1% drop | Low (hours) | Prefill-heavy workloads |
| Caprese | 16%+ faster TTNT | Recovers 70%+ | Medium (8 GPU-hrs) | Post-pruning recovery |
| Llamba | 3-5x (long seq) | 5-10% drop | High (days) | Edge, long-context |
| INT4 Quant | 1.8x | 3% drop | None | Memory-constrained |
| Spec Decode | 2-3x latency | 0% drop | None | Latency-critical |

## Decision Framework

Start with quantization (free, no training needed). If quality or throughput remain insufficient, add distillation. Use speculative decoding only when quality cannot be compromised at all.

```mermaid
flowchart LR
    A["Serving<br/>Bottleneck?"] -->|"Memory"| B["Quantization first<br/>INT8 → INT4"]
    A -->|"Prefill latency"| C["SwiftKV<br/>Skip later layers"]
    A -->|"Long context"| D["Llamba<br/>Eliminate KV cache"]
    A -->|"Quality lost<br/>from compression"| E["Caprese<br/>LoRA recovery"]
    B -->|"Still too slow?"| F["Combine:<br/>Quant + Distill"]
    style A fill:#f3e8ff,stroke:#000,color:#000
    style B fill:#dbeafe,stroke:#000,color:#000
    style C fill:#dcfce7,stroke:#000,color:#000
    style D fill:#ccfbf1,stroke:#000,color:#000
    style E fill:#fef3c7,stroke:#000,color:#000
    style F fill:#ffedd5,stroke:#000,color:#000
```

Combined approaches (quantization + distillation) can achieve 2.5x throughput at 0.25x memory, but risk compounding quality loss. Monitor benchmark scores at each stage.

## FAQ

**Q: When should I distill vs. quantize?**
Quantize first (free, instant). Distill when you need further compression beyond INT4 or must recover quality lost from aggressive quantization.

**Q: How much training data does distillation need?**
SwiftKV needs a few hours on task-specific data. Caprese needs roughly 100B tokens for full recovery. Llamba (cross-architecture) needs less than 0.1% of pretraining data via MOHAWK.

**Q: Does distillation work with quantized models?**
Yes. You can distill from a full-precision teacher into a quantized student, or distill first then quantize the student. The order matters: distill-then-quantize generally preserves more quality.

**Q: What is the minimum teacher-student size ratio?**
Practical ratios range from 2x to 10x. Below 2x, distillation adds marginal value. Above 10x, the capacity gap makes transfer difficult without specialized techniques.

**Q: Can I distill proprietary API models?**
Yes, using only output logits (black-box distillation). Quality is lower than white-box (hidden state) distillation, but still effective. This is how many open-source models learn from GPT-4 outputs.

**Q: Does SwiftKV help decode latency?**
No. SwiftKV only accelerates prefill by skipping KV computation. Decode uses the cached KV normally and sees no speedup.

**Q: How do I detect when distillation quality is insufficient?**
Run your eval suite after each distillation stage. Flag any benchmark where the student drops more than 5% below the teacher. Focus recovery training on those specific capabilities.

## References

1. Qiao et al. "SwiftKV: Fast Prefill-Optimized Inference with Knowledge-Preserving Model Transformation" (arXiv:2410.03960, 2024)
2. Dong et al. "Caprese: Scalable and Efficient Reasoning Acceleration for LLMs" (arXiv:2505.07861, 2025)
3. Bick et al. "Llamba: Scaling Distilled Recurrence Beyond 100B Tokens" (arXiv:2502.14458, 2025)
4. Hinton et al. "Distilling the Knowledge in a Neural Network" (NeurIPS Workshop, 2015)
5. Gu and Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (arXiv:2312.00752, 2023)
6. Kim and Rush. "Sequence-Level Knowledge Distillation" (EMNLP, 2016)
