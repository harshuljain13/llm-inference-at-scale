# 2.3a GQA Deep Dive

---


### Memory Efficiency
- **MHA**: Baseline. Full KV cache. Maximum memory pressure.
- **GQA**: Configurable reduction. Group size is your dial.
- **MQA**: Maximum compression. Minimal cache footprint.

### Model Quality
- **MHA**: Best representational capacity. Each head sees unique KV projections.
- **GQA**: Near-MHA quality. Groups of heads share similar attention patterns anyway (empirically validated by Ainslie et al.).
- **MQA**: Measurable degradation on complex, long-context tasks. The single shared view bottlenecks information flow.

### Training Cost for Conversion
- **MHA → MHA**: Zero. Already trained.
- **MHA → GQA**: 5-10% additional pre-training compute for uptraining. Mean-pool KV weights, then continue training.
- **MHA → MQA**: 10-15% additional compute. Larger quality gap to recover.
- **Train GQA from scratch**: Same cost as MHA training. Just fewer KV parameters.
- **Train MQA from scratch**: Slightly cheaper than MHA (fewer parameters), same number of FLOPs.

The industry consensus as of 2024-2025 is clear: train with GQA from scratch. The quality is equivalent to MHA, the memory savings are substantial (4-16× depending on model size), and there is no conversion cost. MQA remains relevant only for latency-critical applications where every microsecond of KV cache loading matters (e.g., real-time voice assistants generating tokens at 100+ tokens/second).

---

## When to Use Which: A Decision Framework

### Use MHA When:
- **Training-only workloads** where inference memory is irrelevant
- **Small models (< 1B parameters)** where KV cache is already manageable
- **Research/experimentation** where you need maximum representational flexibility
- **Tasks where head specialization is critical** (e.g., multi-modal models where different heads attend to different modalities)

### Use GQA When:
- **Production serving** (this is the default choice in 2024-2025)
- **Any model > 7B parameters** where KV cache becomes a bottleneck
- **Long-context applications** (8K-128K tokens) where MQA degrades
- **Cost-sensitive deployments** where you need to maximize users per GPU
- **Converting existing MHA models** with minimal quality regression

### Use MQA When:
- **Ultra-low-latency serving** (voice assistants, real-time completion)
- **Extremely high concurrency** (thousands of simultaneous users on limited hardware)
- **Short-context applications** (< 512 tokens) where quality loss is minimal
- **Code completion** where outputs are short and speed matters more than long-range coherence

### The Production Default

If you are designing a new LLM for inference serving and have no other constraints, use GQA with these guidelines:

```
n_kv_heads = max(4, n_heads // group_size)
```

Where `group_size` depends on model size:
- **7-13B models**: group_size = 4 (like Llama 3.1 8B, Mistral 7B)
- **30-70B models**: group_size = 8 (like Llama 3.1 70B)
- **100B+ models**: group_size = 16 (like Llama 3.1 405B)

Larger models tolerate higher compression because their increased parameter count compensates for the shared KV representation.

---

## Forward Pointer: Beyond GQA

GQA reduces KV cache by sharing heads within groups, but the dimensionality of each KV head remains unchanged at `head_dim` (typically 128). Module 02.3 introduces Multi-Latent Attention (MLA), used in DeepSeek-V2, which takes compression further by projecting the KV cache into a low-rank latent space. Instead of storing full 128-dimensional vectors per KV head, MLA stores compressed latent vectors of 64 or fewer dimensions, achieving compression ratios that surpass even MQA while maintaining GQA-level quality. This represents the next frontier in attention mechanism design for inference efficiency.

---

## Mental Model

Think of attention mechanisms as a camera system:

- **MHA** gives each camera (head) its own unique lens (K) and film (V). Maximum information, maximum storage cost.
- **MQA** gives all cameras the same single lens and film. Cheap storage, but every camera sees the same thing.
- **GQA** groups cameras into clusters sharing a lens and film. Each cluster sees something different, but cameras within a cluster share a view.

The fundamental tension: **fewer KV heads = smaller cache = more concurrent users, but the question is always: at what quality cost?** GQA answered this definitively: for groups of 4-8, the cost is negligible.

---

## Matplotlib Chart for Lab

The companion `lab.ipynb` should include a visualization with these specifications:

**Chart 1: KV Cache Size vs Attention Mechanism**
- Bar chart showing KV cache per token (KB) for MHA, GQA-8, GQA-4, GQA-2, MQA
- Configuration: Llama 3.1 8B baseline (32 heads, 128 head_dim, 32 layers, FP16)
- Y-axis: KV cache per token (KB), log scale
- Annotate compression ratio above each bar
- Color: gradient from red (MHA) to green (MQA) showing memory improvement

**Chart 2: Quality-Memory Pareto Frontier**
- Scatter plot with X=KV memory per token (KB, log), Y=benchmark score (%)
- Plot points for MHA, GQA-8, GQA-4, MQA with model names
- Draw Pareto frontier line
- Highlight the "sweet spot" region (GQA-4 to GQA-8) with a shaded box

**Chart 3: Concurrent Users by Mechanism**
- Horizontal bar chart showing max concurrent users on A100-80GB at 4K context
- For each of: MHA, GQA-8, GQA-4, MQA
- Annotate with $/user/hour assuming $2/GPU-hour

These charts make the economic argument visceral: the attention mechanism directly determines your serving cost per user.

---

## Key Takeaways

1. **MHA creates independent KV projections per head**, making KV cache scale linearly with head count. This was designed for training, not serving.
2. **MQA (Shazeer, 2019) collapses all KV heads to one**, achieving maximum compression (32-96×) at a measurable quality cost on long-context tasks.
3. **GQA (Ainslie et al., 2023) groups query heads to share KV projections**, providing 4-16× compression with negligible quality loss. This is the industry standard.
4. **The group size scales with model size**: larger models tolerate more sharing because their capacity compensates.
5. **The economic impact is direct**: 4× smaller KV cache means 4× more concurrent users on the same GPU, which means 4× better unit economics.

---

## References

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). "Attention Is All You Need." NeurIPS 2017.
- Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need." arXiv:1911.02150.
- Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., Sanghai, S. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." EMNLP 2023.
- Touvron, H., et al. (2023). "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv:2307.09288.
- Dubey, A., et al. (2024). "The Llama 3 Herd of Models." arXiv:2407.21783.
- Jiang, A.Q., et al. (2023). "Mistral 7B." arXiv:2310.06825.


---

## Deep Dive: How GQA Affects Inference Kernels

The attention mechanism choice does not just affect memory. It fundamentally changes how inference kernels execute on GPU hardware, and understanding this connection explains why GQA delivers better latency in addition to better memory efficiency.

### Memory Bandwidth During Decoding

During the decode phase (generating one token at a time), the dominant operation is loading the KV cache from HBM to compute attention scores. The arithmetic intensity of this operation is extremely low: for each loaded KV vector, you perform only a dot product (head_dim multiplications and additions). This makes decode attention purely memory-bandwidth bound.

The time to generate one token's attention scores:

```
T_attention = KV_cache_size / HBM_bandwidth
```

For an A100 with 2 TB/s HBM bandwidth, serving Llama 3.1 8B at 4K context:

| Mechanism | KV to Load | Load Time | Relative |
|-----------|-----------|-----------|----------|
| MHA | 2,048 MB | 1.02 ms | 4.0× |
| GQA-8 | 512 MB | 0.26 ms | 1.0× (baseline) |
| MQA | 64 MB | 0.03 ms | 0.12× |

GQA-8 provides a 4× latency reduction over MHA purely from reduced memory bandwidth requirements. This is independent of the memory capacity savings for batching.

### Flash Attention Interaction

FlashAttention (Dao et al., 2022) optimizes the prefill phase by avoiding materializing the full attention matrix. However, FlashAttention's tiling strategy interacts differently with each attention variant:

- **MHA + FlashAttention**: Each head processes independently. Tile sizes are optimized per head. Full benefit.
- **GQA + FlashAttention**: The kernel must broadcast shared KV tiles across multiple query heads within a group. Modern FlashAttention-2 handles this efficiently with a dedicated GQA kernel path that loads KV tiles once and applies them to all query heads in the group.
- **MQA + FlashAttention**: Maximum KV reuse. The single KV tile is loaded once and reused across all query heads. This gives MQA the best compute-to-memory ratio during prefill.

In practice, FlashAttention-2 with GQA achieves within 5-10% of the theoretical bandwidth-optimal computation, making the attention mechanism choice the primary lever rather than kernel optimization.

### Tensor Parallelism Implications

When distributing a model across multiple GPUs using tensor parallelism (TP), the KV heads are sharded across devices. This creates a constraint: `n_kv_heads` must be divisible by `TP_degree`.

| TP Degree | MHA (32 KV heads) | GQA-8 | GQA-4 | MQA |
|-----------|-------------------|-------|-------|-----|
| TP=1 | 32 heads/GPU | 8 | 4 | 1 |
| TP=2 | 16 heads/GPU | 4 | 2 | ❌ |
| TP=4 | 8 heads/GPU | 2 | 1 | ❌ |
| TP=8 | 4 heads/GPU | 1 | ❌ | ❌ |

MQA with TP > 1 requires replicating the single KV head across devices (wasteful) or redesigning the parallelism strategy. GQA-8 cleanly supports up to TP=8, which is why Meta chose 8 KV heads: it aligns with their standard 8-GPU node configuration.

This is a practical engineering constraint that influenced the industry's convergence on GQA-8 over MQA. The parallelism-friendly nature of 8 KV heads across 8-GPU nodes made GQA the natural fit for large-scale deployment.

---

## Historical Timeline

Understanding when each mechanism was introduced and adopted reveals how quickly the field converges on proven techniques:

| Year | Event | Impact |
|------|-------|--------|
| 2017 | Vaswani et al. introduce MHA in "Attention Is All You Need" | Standard for 5 years |
| 2019 | Shazeer proposes MQA | Adopted by PaLM, Falcon, StarCoder |
| 2022 | PaLM ships with MQA at 540B scale | Proves MQA works for large models |
| 2023 Jan | Mistral 7B ships with GQA-8 | First popular open model with GQA |
| 2023 Jun | Ainslie et al. publish GQA paper | Formalizes the technique |
| 2023 Jul | Meta releases Llama 2 70B with GQA-8 | Industry adoption begins |
| 2024 Apr | Llama 3 family: all sizes use GQA-8 | GQA becomes universal default |
| 2024 May | DeepSeek-V2 introduces MLA | Next evolution (see Module 02.3) |

The adoption curve from Shazeer's 2019 paper to universal GQA adoption in 2024 took 5 years. The conversion from the GQA paper (June 2023) to industry default took less than 12 months. This acceleration reflects how critical KV cache efficiency became as context lengths grew from 2K to 128K tokens.

---

## Common Misconceptions

### "MQA is strictly worse than GQA"

Not true. For applications with short contexts (< 512 tokens) and extreme latency requirements, MQA's 32× compression can be the right choice. Code completion models (StarCoder, CodeGen) use MQA because outputs are short, latency matters more than long-range coherence, and the quality difference on short code completions is imperceptible.

### "GQA requires special training"

If training from scratch: no. You simply define fewer KV heads in your model config. The training procedure is identical to MHA. The "uptraining" mentioned in the GQA paper is only needed when converting an existing MHA checkpoint.

### "More KV heads always means better quality"

Not necessarily. Ainslie et al. showed that many MHA heads learn highly correlated KV representations. The heads within what would become a GQA group often attend to very similar patterns. Sharing was always implicit; GQA just makes it explicit and saves the redundant storage.

### "GQA only helps inference, not training"

Partially true. GQA primarily benefits inference by reducing KV cache. However, during training, fewer KV parameters mean slightly less memory for gradient states, which can allow larger batch sizes. The effect is modest (5-10% memory reduction during training) compared to the dramatic inference improvement.

---

## Practical Implementation Notes

### Detecting the Attention Type from Model Config

When working with HuggingFace models, you can determine the attention variant from the config:

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B")

n_heads = config.num_attention_heads        # 32
n_kv_heads = config.num_key_value_heads     # 8

if n_kv_heads == n_heads:
    print("MHA")
elif n_kv_heads == 1:
    print("MQA")
else:
    print(f"GQA-{n_kv_heads} (group_size={n_heads // n_kv_heads})")
# Output: GQA-8 (group_size=4)
```

### Adjusting Batch Size Based on Attention Type

When estimating how many requests you can batch together:

```python
def max_batch_size(
    gpu_memory_gb: float,
    model_memory_gb: float,
    n_kv_heads: int,
    head_dim: int,
    n_layers: int,
    max_seq_len: int,
    dtype_bytes: int = 2,
    overhead_factor: float = 1.2,  # 20% overhead for activations, fragmentation
) -> int:
    """Estimate maximum batch size given attention mechanism."""
    available_gb = gpu_memory_gb - model_memory_gb
    available_bytes = available_gb * (1024**3)
    
    kv_per_sequence = (
        2 * n_kv_heads * head_dim * n_layers * max_seq_len * dtype_bytes
    )
    
    return int(available_bytes / (kv_per_sequence * overhead_factor))

# Llama 3.1 8B on A100-80GB
batch = max_batch_size(
    gpu_memory_gb=80, model_memory_gb=16,
    n_kv_heads=8, head_dim=128, n_layers=32,
    max_seq_len=4096
)
print(f"Max batch size: {batch}")  # ~97
```

---

## Summary: The Three-Sentence Version

Multi-Head Attention gives every head independent KV projections, creating a KV cache that scales linearly with head count and becomes the primary memory bottleneck during inference. Multi-Query Attention (Shazeer, 2019) compresses this to a single shared KV head, achieving 32-96× reduction at a measurable quality cost. Grouped-Query Attention (Ainslie et al., 2023) strikes the optimal balance: groups of 4-8 query heads share KV projections, delivering 4-16× compression with negligible quality loss, making it the universal default for production LLM serving.
