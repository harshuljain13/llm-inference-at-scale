# 2.2 Multi-Head Attention (MHA)

Every token your model generates requires reading from the KV cache, and the size of that cache is determined entirely by your attention mechanism's design. This module traces the architectural evolution that took KV cache memory from "unaffordable at scale" to "hundreds of concurrent users on a single GPU." Understanding this progression is not optional for inference engineers: it is the single most impactful design decision affecting your serving costs.

## Back-Reference: What You Already Know

From Module 02.1, you know the KV cache stores K and V projection tensors for every layer and every token in the sequence. For Llama 3.1 8B with GQA (8 KV heads, 128 head dimension, 32 layers, FP16), each token costs approximately 128 KB of GPU memory. You saw how this grows linearly with sequence length and batch size, creating the fundamental memory pressure that limits concurrent serving.

What Module 02.1 did not explain is *why* Llama uses 8 KV heads instead of 32, or why earlier models like GPT-3 used 96 KV heads matching their query heads exactly. The answer lies in a three-stage evolution of the attention mechanism itself, each stage trading a small amount of model quality for a dramatic reduction in KV cache size.

---


![Attention Variants Memory Comparison](images/attention_variants_memory.png)

## Multi-Head Attention (MHA): The Original Design

Multi-Head Attention, introduced in "Attention Is All You Need" (Vaswani et al., 2017), gives every attention head its own independent Key and Value projections. If your model has `n_heads` query heads, it also has `n_heads` KV heads. Each head independently attends to different aspects of the input: one head might track syntactic relationships, another semantic similarity, another positional proximity.

### The Architecture

In MHA, the input hidden state `x` of dimension `d_model` is projected through three separate weight matrices per head:

```
Q_i = x @ W_Q_i    # shape: [seq_len, head_dim]
K_i = x @ W_K_i    # shape: [seq_len, head_dim]
V_i = x @ W_V_i    # shape: [seq_len, head_dim]
```

where `i` ranges from 1 to `n_heads`, and `head_dim = d_model / n_heads`.

Each head computes attention independently:

```
Attention_i = softmax(Q_i @ K_i^T / sqrt(head_dim)) @ V_i
```

The outputs are concatenated and projected back to `d_model`:

```
Output = Concat(Attention_1, ..., Attention_n) @ W_O
```

### KV Cache Memory Formula for MHA

During autoregressive inference, every generated token adds its K and V vectors to the cache for ALL heads across ALL layers:

```
KV_cache_per_token (MHA) = 2 × n_kv_heads × head_dim × n_layers × bytes_per_element
```

Since `n_kv_heads = n_heads` in MHA:

```
KV_cache_per_token (MHA) = 2 × n_heads × head_dim × n_layers × bytes_per_element
```

### Concrete Example: GPT-3 175B

GPT-3 uses pure MHA with these parameters:
- `n_heads = 96`
- `head_dim = 128`
- `n_layers = 96`
- `dtype = FP16 (2 bytes)`

```
KV per token = 2 × 96 × 128 × 96 × 2 = 4,718,592 bytes ≈ 4.5 MB/token
```

For a 2048-token context: `4.5 MB × 2048 = 9.2 GB` of KV cache per request. With 8 concurrent users at 2K context, you need 73.6 GB just for KV cache, consuming nearly all of an 80 GB A100.

### Why MHA Works for Training but Fails for Serving

During training, you process the entire sequence in parallel. The KV projections are computed once and used immediately. There is no cache because you already have all tokens. The memory cost is proportional to the batch size and sequence length, but it is transient.

During inference, the KV cache persists for the entire generation. Each new token requires reading the full cache from memory (memory-bandwidth bound), and the cache must remain allocated until the request completes. This is the fundamental asymmetry: a mechanism designed for parallel training creates an unbearable memory burden during sequential generation.

### Models Using MHA

| Model | Year | n_heads | head_dim | Layers | KV/token |
|-------|------|---------|----------|--------|----------|
| GPT-2 | 2019 | 12-25 | 64 | 12-48 | 36-600 KB |
| GPT-3 | 2020 | 96 | 128 | 96 | 4.5 MB |
| BERT-Large | 2018 | 16 | 64 | 24 | 49 KB |
| OPT-175B | 2022 | 96 | 128 | 96 | 4.5 MB |
| BLOOM-176B | 2022 | 112 | 128 | 70 | 4.0 MB |

The pattern is clear: as models scale, MHA's KV cache becomes the dominant memory consumer, leaving less room for batching concurrent requests.

---

## Multi-Query Attention (MQA): Radical Compression

In 2019, Noam Shazeer published "Fast Transformer Decoding: One Write-Head is All You Need," proposing a startlingly simple modification: instead of giving each attention head its own KV projections, share a single K and a single V across ALL query heads.

### The Key Insight

Shazeer observed that during inference, the memory bandwidth consumed by loading KV cache from HBM dominates the compute time. The attention computation itself is fast (it's just matrix multiplies on small head_dim vectors). The bottleneck is moving 4.5 MB of KV data per token from HBM to the compute units for every single generated token.

If all query heads share the same K and V, you only need to store and load one set of KV vectors per layer, regardless of how many query heads you have.

### The Architecture

```
# MQA: n_heads query projections, but only ONE KV projection
Q_i = x @ W_Q_i    # i = 1..n_heads, shape: [seq_len, head_dim]
K   = x @ W_K      # SINGLE shared K, shape: [seq_len, head_dim]
V   = x @ W_V      # SINGLE shared V, shape: [seq_len, head_dim]

# Each query head attends using the SAME K and V
Attention_i = softmax(Q_i @ K^T / sqrt(head_dim)) @ V
```

### KV Cache Memory Formula for MQA

```
KV_cache_per_token (MQA) = 2 × 1 × head_dim × n_layers × bytes_per_element
```

Notice `n_kv_heads = 1` regardless of how many query heads exist.

### Compression Ratio

For a model with `n_heads` query heads:

```
MQA_compression = n_heads / 1 = n_heads
```

A 32-head model gets 32× KV cache reduction. A 96-head model like GPT-3 would get 96× reduction.

### Concrete Example: PaLM (if MQA)

PaLM 540B uses MQA with:
- `n_heads = 48` (query heads)
- `n_kv_heads = 1` (MQA)
- `head_dim = 256`
- `n_layers = 118`
- `dtype = FP16`

```
KV per token = 2 × 1 × 256 × 118 × 2 = 120,832 bytes ≈ 118 KB/token
```

Compare to MHA equivalent: `2 × 48 × 256 × 118 × 2 = 5.8 MB/token`. That is a 48× reduction.

### The Quality Cost

The compression is not free. When all query heads share the same KV representation, the model loses the ability to attend to different aspects of the input with different head-specific Key/Value spaces. Empirical results from Shazeer (2019) and follow-up work show:

- **Short-context tasks (< 512 tokens)**: Negligible quality difference. The shared KV representation captures sufficient information for most heads.
- **Long-context tasks (> 2K tokens)**: Measurable degradation (0.5-2% on benchmarks). Different heads genuinely benefit from specialized KV projections when the sequence is long enough to contain diverse information.
- **Knowledge-intensive tasks**: Moderate degradation. Tasks like open-domain QA where the model must recall specific facts from long contexts suffer more than summarization tasks.

### Training Consideration

Training an MQA model from scratch requires no modification to the training procedure. The model simply has fewer parameters in the KV projection (by a factor of `n_heads`). However, converting an existing MHA checkpoint to MQA is non-trivial. Shazeer (2019) found that "uptrained" MQA models (pre-trained with MHA, then fine-tuned with shared KV) recover most but not all of the quality difference within 5-10% additional training compute.

### Models Using MQA

| Model | Year | Query Heads | KV Heads | Compression |
|-------|------|-------------|----------|-------------|
| PaLM | 2022 | 48 | 1 | 48× |
| Falcon-40B | 2023 | 64 | 1 | 64× |
| StarCoder | 2023 | 48 | 1 | 48× |
| MPT-30B | 2023 | 64 | 1 | 64× |

---

