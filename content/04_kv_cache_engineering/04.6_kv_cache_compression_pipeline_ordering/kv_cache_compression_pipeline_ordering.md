# 4.6 KV Cache Compression Pipeline Ordering

Evict-first pipelines waste zero quantization cycles. They can still lose overall.
This single fact reveals why KV compression cannot be optimized one technique at a
time. Each stage in a pipeline changes the bytes remaining, the cost of downstream
processing, the quality already consumed, and the expected future reload penalty.
The right ordering depends on your workload regime. This module covers which pipeline
wins under which pressure, why minimizing a single metric is not sufficient, and what
rules survive across regimes.

---

## 4.6.1 Why Ordering Matters

Module 4.2 covers quantization, eviction, and tiering as individual techniques.
Production systems apply them as a sequence. The sequence is not neutral: each stage
modifies the state that downstream stages see.

Quantize then evict:
- Reduces byte volume early
- May spend compute quantizing regions that are immediately discarded
- Downstream eviction sees fewer bytes to process

Evict then quantize:
- Removes low-value regions before compressing
- Eliminates wasted quantization work on discarded data
- But survivors may have higher quality sensitivity, making quantization costlier

Tier then quantize:
- Moves cold data to CPU before compressing what remains
- Quantization work is concentrated on hot GPU-resident data
- PCIe transfer cost paid early; no compression benefit for tiered bytes

The ordering problem is not about which technique is better. It is about which
technique changes the remaining system state in a way that makes downstream stages
more effective.

```
flowchart LR
    subgraph Q_first["Quantize → Evict → Tier"]
        style Q_first fill:#fef3c7,stroke:#000,color:#000
        QA["Compress all regions<br/>including future evictions"] --> QB["Evict low-value<br/>already-compressed regions"] --> QC["Tier remainder<br/>to CPU"]
    end

    subgraph E_first["Evict → Quantize → Tier"]
        style E_first fill:#dbeafe,stroke:#000,color:#000
        EA["Remove low-value<br/>regions first"] --> EB["Compress survivors<br/>no wasted cycles"] --> EC["Tier remainder<br/>to CPU"]
    end

    subgraph M_first["Mixed-Precision → Evict → Tier"]
        style M_first fill:#dcfce7,stroke:#000,color:#000
        MA["Compress aggressively<br/>reduce byte volume early"] --> MB["Evict selectively<br/>from smaller pool"] --> MC["Tier cold remainder<br/>to CPU"]
    end

    style QA fill:#ffe4e6,stroke:#000,color:#000
    style QB fill:#fef3c7,stroke:#000,color:#000
    style QC fill:#f3f4f6,stroke:#000,color:#000
    style EA fill:#dcfce7,stroke:#000,color:#000
    style EB fill:#dbeafe,stroke:#000,color:#000
    style EC fill:#f3f4f6,stroke:#000,color:#000
    style MA fill:#dcfce7,stroke:#000,color:#000
    style MB fill:#dbeafe,stroke:#000,color:#000
    style MC fill:#f3f4f6,stroke:#000,color:#000
```

---

## 4.6.2 The Three Compression Stages

Each stage targets a different lever. Brief recap before pipeline analysis:

| Stage | Mechanism | Memory recovery | Quality cost | Future cost |
|---|---|---|---|---|
| Quantization (INT8) | Reduce precision per token | Moderate (2x) | Low | None |
| Mixed precision (INT8/INT4) | Selective precision per region | High (up to 4x) | Moderate | None |
| Eviction | Remove GPU-resident regions | High, immediate | High | Recomputation if region needed again |
| Tiering | Move regions to CPU | High, recoverable | None | PCIe reload latency |
| Prefetch | Pre-stage tiered regions | None (cost reducer) | None | Reduces tier reload penalty |

**Prefetch does not reduce memory pressure.** It reduces future reload cost for
tiered regions only. Its value is contingent on tiering being active in the pipeline.

---

## 4.6.3 Stage Activation Thresholds

One of the most practically important behaviors in pipeline benchmarking is stage
deactivation: an earlier stage may recover enough memory that downstream stages
become no-ops.

```
flowchart LR
    A["GPU KV pressure:<br/>above target budget"] --> B["Quantization stage runs"]
    B --> C{"GPU KV now<br/>below target?"}
    C -->|Yes| D["Eviction stage: no-op<br/>Tiering stage: no-op<br/>Wasted quant: zero"]
    C -->|No| E["Eviction stage runs"]
    E --> F{"GPU KV now<br/>below target?"}
    F -->|Yes| G["Tiering stage: no-op"]
    F -->|No| H["Tiering stage runs"]

    style A fill:#ffe4e6,stroke:#000,color:#000
    style B fill:#dbeafe,stroke:#000,color:#000
    style C fill:#fef3c7,stroke:#000,color:#000
    style D fill:#dcfce7,stroke:#000,color:#000
    style E fill:#dbeafe,stroke:#000,color:#000
    style F fill:#fef3c7,stroke:#000,color:#000
    style G fill:#dcfce7,stroke:#000,color:#000
    style H fill:#dbeafe,stroke:#000,color:#000
```

**What this unlocks:** In several normal-pressure workloads, quantization alone
reduces GPU KV usage below the target budget, making downstream eviction and tiering
no-ops. This means wasted quantization is zero in those regimes regardless of pipeline
ordering — the wasted quantization problem is pressure-gated.

---

## 4.6.4 Empirical Findings Across Pressure Regimes

The benchmark evaluates pipeline orderings across six synthetic workload regimes:
light pressure, moderate pressure, heavy pressure, long-context heavy, reuse-heavy,
and extreme overpressure. Source benchmark and methodology:
[github.com/JohnScheuer/kv-cache-compression-pipeline-bench](https://github.com/JohnScheuer/kv-cache-compression-pipeline-bench)

### Finding 1: No universally optimal pipeline

The best pipeline shifts across regimes. There is no single ordering that wins
across all workload types.

| Pressure Regime | Best Pipeline | Why |
|---|---|---|
| Light pressure | evict → mixed_precision → tier | Many evictable regions; no need to compress before removing |
| Moderate pressure | evict → mixed_precision → tier | Threshold effects make quant a no-op downstream |
| Long-context heavy | evict → mixed_precision → tier | High evictability; cold prefix segments removed first |
| Heavy pressure | mixed_precision → evict → tier | Byte volume dominates; early compression makes eviction cheaper |
| Reuse-heavy | mixed_precision → evict → tier | High reload cost penalizes eviction; compress first, evict selectively |
| Extreme overpressure | mixed_precision → evict → tier | Aggressive compression needed; evict-first wastes future cost budget |

### Finding 2: Compress-before-move is a robust rule

Quantize-then-tier consistently outperforms tier-only across regimes. Moving
uncompressed bytes across PCIe is wasteful: you pay transfer bandwidth on bytes
that compression could have eliminated. Applying quantization before tiering reduces
transfer cost and keeps future reload cheaper (fewer bytes to move back).

```
flowchart LR
    subgraph Tier_only["Tier without compression"]
        style Tier_only fill:#ffe4e6,stroke:#000,color:#000
        T1["Transfer FP16 bytes<br/>to CPU"] --> T2["Full bandwidth cost<br/>on reload"]
    end

    subgraph Quant_tier["Quantize then tier"]
        style Quant_tier fill:#dcfce7,stroke:#000,color:#000
        Q1["Compress to INT4/INT8<br/>on GPU"] --> Q2["Transfer smaller payload<br/>to CPU"] --> Q3["Reload cost reduced<br/>proportionally"]
    end

    style T1 fill:#ffe4e6,stroke:#000,color:#000
    style T2 fill:#ffe4e6,stroke:#000,color:#000
    style Q1 fill:#dcfce7,stroke:#000,color:#000
    style Q2 fill:#dcfce7,stroke:#000,color:#000
    style Q3 fill:#dcfce7,stroke:#000,color:#000
```

**Rule: always compress before moving across PCIe.** This holds regardless of which
compression mode is chosen and which pressure regime is active.

### Finding 3: Wasted quantization is real but pressure-gated

Wasted quantization occurs when a stage compresses a region that a downstream stage
then discards. The compute spent on compression is lost.

| Pipeline | Wasted quantization (extreme overpressure) |
|---|---|
| quantize → evict → tier | 23–36% of quantization work wasted |
| mixed_precision → evict → tier | Lower; mixed precision is more selective |
| evict → mixed_precision → tier | ~0%; compression only on survivors |

**What this unlocks:** Evict-first eliminates wasted quantization entirely. But this
advantage only appears under extreme overpressure, where downstream stages are
actually activated. In light or moderate regimes, downstream stages are no-ops and
wasted quantization is zero for all pipelines.

### Finding 4: Minimizing wasted quantization is not sufficient

This is the counterintuitive result. Evict-first wastes zero quantization cycles.
It still loses under heavy and reuse-heavy pressure.

Why: eviction carries a high future recompute penalty. Regions removed from GPU
memory must be recomputed from scratch if requested again. In reuse-heavy workloads,
evicted regions have high reload probability. The future cost of eviction-first
pipelines can exceed the quantization waste they avoid.

Mixed-precision-first recovers comparable memory with lower future penalty: tiered
regions are reloadable from CPU, while quantized survivors stay GPU-resident.
Eviction is applied selectively to the regions with lowest reload probability,
limiting expected recompute cost.

### Finding 5: Mixed-precision → evict → tier is the best balanced strategy under severe pressure

In heavy and extreme pressure regimes, `mixed_precision → evict → tier` dominates.
It reduces byte volume early (making downstream eviction more selective), limits
wasted quantization relative to `quantize → evict → tier`, and controls future
reload cost by avoiding aggressive eviction of high-reload-probability regions.

### Finding 6: Quantize-then-tier is a strong near-optimal baseline

In several workloads, `quantize → tier` (without eviction) sits within a small score
gap of the best pipeline while being operationally simpler. If your system needs one
default pipeline that avoids the complexity of regime detection:

- Use `quantize → tier` as the default
- Switch to `evict → mixed_precision → tier` for long-context or light workloads
- Switch to `mixed_precision → evict → tier` for heavy or extreme pressure

### Finding 7: Prefetch helps tiering, not eviction

Prefetch reduces reload stall by pre-staging CPU-tiered regions before they are
needed. This benefit is real for tiering-heavy pipelines. For eviction-heavy
pipelines, future cost is dominated by recomputation, not PCIe transfer. Prefetch
has limited effect when eviction is the primary pressure relief mechanism.

---

## 4.6.5 Practical Ordering Rules

Four rules that hold across all six regimes:

**Rule 1: Compress before moving across PCIe.**
Quantize or apply mixed precision before tiering. Always.

**Rule 2: Choose evict-first for light, moderate, and long-context workloads.**
When many regions are highly evictable and reload probability is low, removing them
before compressing avoids unnecessary compute.

**Rule 3: Choose mixed-precision-first for heavy, reuse-heavy, and extreme workloads.**
When future reload cost is high and byte volume is the dominant constraint, aggressive
compression before eviction reduces both wasted quantization and future penalty.

**Rule 4: Monitor stage activation to detect no-ops.**
If your monitoring shows eviction or tiering stages never activating, the earlier
stage is handling the full budget. You are paying pipeline overhead for no benefit.
Track `post_quant_to_target_ratio` and `evict_stage_active` signals to detect this.

---

## 4.6.6 Decision Framework

```
flowchart LR
    A{"What is your<br/>workload pressure?"}
    A -->|Light or moderate| B["Many evictable regions?"]
    A -->|Heavy or reuse-heavy| C["High reload probability?"]
    A -->|Extreme overpressure| D["mixed_precision → evict → tier"]

    B -->|Yes| E["evict → mixed_precision → tier"]
    B -->|No| F["quantize → tier<br/>(near-optimal baseline)"]

    C -->|Yes| G["mixed_precision → evict → tier<br/>(limit future cost)"]
    C -->|No| F

    style A fill:#fef3c7,stroke:#000,color:#000
    style D fill:#dcfce7,stroke:#000,color:#000
    style E fill:#dcfce7,stroke:#000,color:#000
    style F fill:#dbeafe,stroke:#000,color:#000
    style G fill:#dcfce7,stroke:#000,color:#000
```

---

## 4.6.7 Limitations

This module is based on a calibrated simulation benchmark, not end-to-end inference.
Key limitations to understand before applying these findings:

**Synthetic region distributions.** The benchmark uses synthetic workload regimes
with parameterized region properties (hotness, evictability, reload probability).
Real KV distributions inside production models may differ from these approximations.

**Proxy quality loss.** Quality degradation is modeled as an additive proxy per
stage rather than measured task accuracy. Results indicate relative ordering of
quality cost, not exact accuracy delta for a specific task.

**Simplified reload cost model.** Recomputation and PCIe reload penalties are
modeled as calibrated constants. Actual costs depend on model size, interconnect
bandwidth, and scheduler implementation.

**No multi-tenant modeling.** The benchmark assumes a single request context per
simulation run. Multi-tenant systems introduce interference effects on the block
pool that are not captured here.

**Extensions worth exploring:**
- Multi-tenant pipeline analysis with shared block pool contention
- Integration with admission control (pipeline ordering as function of queue depth)
- Adaptive pipeline selection using online regime detection signals
- Prefetch policies for tiering-heavy pipelines under varying reload probabilities

---

## FAQ

**Q: Can I apply this ordering logic without modifying the inference engine?**
Partially. The compress-before-move rule can be implemented at the KV management
layer without engine changes. Regime-adaptive ordering requires access to per-request
KV state signals (hotness, evictability estimates) that most engines do not expose
by default.

**Q: Does ordering matter if I am only using one technique?**
No. Ordering only has an effect when two or more stages are active in the same
pipeline. A quantization-only or eviction-only configuration has no ordering to
decide. These findings apply when you are combining techniques.

**Q: How does this interact with INT4 quantization from Module 4.2?**
Directly. The wasted quantization problem scales with compression aggressiveness.
INT4 wastes more compute per discarded region than INT8. Under extreme overpressure,
the case for evict-first (or mixed-precision-first) is stronger when INT4 is the
compression mode.

**Q: Should I always avoid eviction-first under heavy pressure?**
Not always avoid — avoid aggressive eviction of high-reload-probability regions.
Eviction of genuinely cold regions with low reload probability is still efficient
regardless of pressure. The regime classification is a heuristic; per-region
reload probability is the more precise signal.

**Q: What monitoring signals indicate I am in the wrong pipeline for my workload?**
Three signals: (1) high expected reload latency (eviction-first losing on future
cost), (2) high wasted quantization fraction (quantize-first under extreme pressure),
(3) downstream stages consistently inactive (pipeline overhead with no benefit).
Tracking these against net memory recovery gives regime-level feedback.

**Q: Is quantize-then-tier always safe as a default?**
Safe, but suboptimal in long-context workloads with many evictable regions. In those
cases, evict-first recovers comparable memory at lower compute cost. Quantize-then-tier
is the right default when you cannot afford regime-detection overhead or do not have
visibility into regional evictability.

---

## References

De Souza, J.F. "KV Cache Compression Pipeline Bench." 2026.
[github.com/JohnScheuer/kv-cache-compression-pipeline-bench](https://github.com/JohnScheuer/kv-cache-compression-pipeline-bench)

Zandieh et al. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." ICLR 2026. arXiv:2504.19874

Zhang et al. "H2O: Heavy-Hitter Oracle for Efficient Generative Inference." NeurIPS 2023. arXiv:2306.14048

Hooper et al. "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization." NeurIPS 2024. arXiv:2401.18079

Kwon, W. et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.