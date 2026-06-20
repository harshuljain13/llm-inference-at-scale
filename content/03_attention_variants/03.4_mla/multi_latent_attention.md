# 3.4 Multi-Latent Attention (MLA)

Multi-Latent Attention (MLA) is an attention mechanism introduced by DeepSeek in their V2 model (May 2024) that compresses
both keys and values into a shared low-rank latent vector before caching them. Unlike previous approaches that merely reduce
the number of KV heads (GQA) or collapse them entirely (MQA), MLA projects the full KV representation into a dramatically
smaller latent space, achieving a 93.3% reduction in KV cache size while actually *improving* model quality compared to
standard Multi-Head Attention. This chapter explains how MLA works, derives the memory savings mathematically, and then
introduces LMCache, an open-source KV caching layer that enables cross-request KV reuse, disk/CPU offloading, and
prefix sharing in production serving stacks like vLLM and SGLang.

---

## Back-Reference: Where GQA Left Us

Recall from Module 2.2 that Grouped-Query Attention (GQA) addresses the KV cache bottleneck by sharing Key and Value heads
across groups of Query heads. In Llama 3.1 8B, for example, 32 query heads share 8 KV head groups, meaning each KV head
serves 4 query heads. This reduces the number of independent KV vectors from 32 (one per query head in full MHA) down to 8,
cutting KV cache by 4x compared to standard MHA.

The key insight of GQA is that adjacent query heads often attend to similar patterns, so sharing their KV projections
introduces minimal quality degradation. The architecture looks like this:

```
Standard MHA:  32 Query heads, 32 Key heads, 32 Value heads  -> Cache all 32 K + 32 V
GQA (Llama):   32 Query heads,  8 Key heads,  8 Value heads  -> Cache only 8 K + 8 V
```

This is a significant improvement. But there is a fundamental limitation that GQA does not address: each KV head still
stores a full `head_dim`-dimensional vector per token per layer. The cache size grows linearly with sequence length,
number of layers, and head dimension. For models with large head dimensions or many layers, even 8 KV heads can consume
substantial GPU memory.

The question MLA answers is: what if we could compress those KV vectors themselves into something much smaller, rather
than just reducing the number of heads that produce them?


## The Problem GQA Still Has

Let us quantify the remaining KV cache burden after GQA. The KV cache stores, for every past token in the sequence,
the Key and Value vectors at every layer. The formula for KV cache size per token in GQA is:

```
KV_cache_per_token_GQA = 2 * n_kv_heads * head_dim * n_layers * bytes_per_element
```

For Llama 3.1 8B with GQA:
- n_kv_heads = 8
- head_dim = 128
- n_layers = 32
- bytes_per_element = 2 (FP16)

```python
# GQA KV cache per token (Llama 3.1 8B)
n_kv_heads = 8
head_dim = 128
n_layers = 32
bytes_per_element = 2  # FP16

kv_per_token_gqa = 2 * n_kv_heads * head_dim * n_layers * bytes_per_element
print(f"GQA KV cache per token: {kv_per_token_gqa:,} bytes = {kv_per_token_gqa / 1024:.1f} KB")
# Output: GQA KV cache per token: 131,072 bytes = 128.0 KB
```

At 128 KB per token, a batch of 32 sequences each with 4096 tokens requires:

```python
batch_size = 32
seq_len = 4096
total_kv = kv_per_token_gqa * batch_size * seq_len
print(f"Total KV cache: {total_kv / (1024**3):.1f} GB")
# Output: Total KV cache: 16.0 GB
```

That is 16 GB of GPU memory consumed by KV cache alone, leaving limited headroom for model weights and activations on
a single 80 GB GPU. For longer contexts (32K, 128K tokens), GQA's linear scaling becomes the primary bottleneck for
batch size and throughput. This is the problem DeepSeek set out to solve.


## DeepSeek's Insight: Compress KV into a Low-Rank Latent Space

The fundamental observation behind MLA is that the Key and Value matrices across all heads contain significant redundancy.
If you stack all KV head vectors for a given token into a matrix and perform SVD (Singular Value Decomposition), you find
that most of the information is captured by a small number of singular values. In other words, the KV representation is
inherently low-rank.

DeepSeek's insight: instead of caching the full KV vectors (which are `n_kv_heads * head_dim` dimensional), project them
down into a compact latent vector of dimension `d_c` (where `d_c << n_kv_heads * head_dim`), cache only this latent
vector, and decompress it back to full KV dimensions on-the-fly during attention computation.

This is analogous to how JPEG compresses images: you store a compact representation and reconstruct the full image
when you need it. The tradeoff is extra compute (decompression) in exchange for dramatically less memory (smaller cache).

The architecture has three stages:

1. **Compress (Down-projection)**: Project the hidden state `h_t` into a low-dimensional latent `c_t^KV` using a
   learned down-projection matrix `W_DKV`.
2. **Cache**: Store only `c_t^KV` (the latent vector) in the KV cache, not the full keys and values.
3. **Decompress (Up-projection)**: At attention time, recover the full keys via `W_UK * c_t^KV` and values via
   `W_UV * c_t^KV`.

The critical realization is that during inference, the up-projection matrices `W_UK` and `W_UV` can be *absorbed* into
the query projection and output projection respectively (via matrix multiplication associativity). This means we never
actually need to materialize the full-size keys and values at all. We can compute attention scores directly between the
compressed query and the compressed KV latent, keeping everything in the low-dimensional space.


## The Math: Joint KV Compression

Let us formalize MLA step by step. We define the following notation (matching the DeepSeek-V2 paper, arXiv 2405.04434):

- `d`: model hidden dimension (embedding size)
- `n_h`: number of attention heads
- `d_h`: per-head dimension
- `d_c`: KV compression dimension (the latent size, where d_c << n_h * d_h)
- `d_c_prime`: query compression dimension
- `d_h_R`: per-head dimension for the decoupled RoPE key
- `h_t`: hidden state input at position t, shape [d]

### Step 1: Low-Rank KV Joint Compression

Instead of the standard KV projection (`K = W_K * h_t`, `V = W_V * h_t`), MLA first compresses into a shared
latent vector:

```
c_t^KV = W_DKV * h_t          # Down-project: [d] -> [d_c]
```

Where `W_DKV` is a matrix of shape `[d_c, d]`. This latent vector `c_t^KV` is the ONLY thing cached per token.

Then, to recover keys and values for attention computation:

```
k_t^C = W_UK * c_t^KV         # Up-project keys: [d_c] -> [n_h * d_h]
v_t^C = W_UV * c_t^KV         # Up-project values: [d_c] -> [n_h * d_h]
```

Where `W_UK` and `W_UV` are up-projection matrices of shape `[n_h * d_h, d_c]`.

### Step 2: Decoupled Rotary Position Embedding (RoPE)

There is a subtlety: Rotary Position Embedding (RoPE) is position-dependent and gets applied to keys. If we apply
RoPE to the compressed keys `k_t^C`, the up-projection matrix `W_UK` becomes coupled with a position-dependent
RoPE matrix and can no longer be absorbed into the query projection. This would force us to decompress all keys
at every generation step, destroying the efficiency gains.

DeepSeek's solution is to *decouple* the position information into a separate small key vector:

```
k_t^R = RoPE(W_KR * h_t)      # Decoupled RoPE key: [d] -> [d_h_R]
```

This `k_t^R` is a small vector (dimension `d_h_R`, typically `d_h / 2 = 64`) that carries all the positional
information. It is shared across all heads and must also be cached alongside `c_t^KV`.

The final key for each head is the concatenation:

```
k_t_i = [k_t_i^C ; k_t^R]    # Concat content key + position key
```

And the query gets a matching structure:

```
q_t_i = [q_t_i^C ; q_t_i^R]  # Concat content query + position query
```

### Step 3: The Attention Computation

With these definitions, the attention output for head i is:

```
o_t_i = sum_j softmax( q_t_i^T * k_j_i / sqrt(d_h + d_h_R) ) * v_j_i^C
```

And the final output combines all heads:

```
u_t = W_O * [o_t_1 ; o_t_2 ; ... ; o_t_nh]
```

### The Matrix Absorption Trick

The key efficiency insight: since `k_t_i^C = W_UK_i * c_t^KV`, the dot product `q_t_i^C^T * k_t_i^C` can be
rewritten as:

```
q_t_i^C^T * W_UK_i * c_t^KV = (W_UK_i^T * q_t_i^C)^T * c_t^KV
```

We can precompute `W_UK_i^T * W_UQ_i` as a single merged matrix. This means during inference, attention scores
are computed directly between the compressed query representation and the cached latent `c_t^KV`, without ever
materializing the full n_h * d_h dimensional keys.

Similarly, `W_UV` can be absorbed into `W_O`, so we never materialize full-size values either. The entire
attention computation operates in the compressed latent space.

### Dimensional Walkthrough of Absorption

Let us trace dimensions concretely for a single head `i` to verify the absorption trick works. The relevant shapes are:

- `q_t_i^C`: the content query for head i, shape `[d_h]` = `[128]`
- `W_UK_i`: the up-projection from latent to key for head i, shape `[d_h, d_c]` = `[128, 512]`
- `c_t^KV`: the cached latent vector, shape `[d_c]` = `[512]`

**Without absorption** (naive approach): For each of `S` cached tokens, reconstruct the full key then dot with query:

```
For each cached token j = 1..S:
    k_j_i^C = W_UK_i @ c_j^KV       # [128, 512] @ [512] -> [128]
    score_j = q_t_i^C^T @ k_j_i^C   # [128]^T @ [128] -> scalar
```

This requires `S` matrix-vector products of shape `[128, 512] @ [512]`, costing `O(S * d_h * d_c)` = `O(S * 128 * 512)` = `O(65,536 * S)` FLOPs. Worse, you must materialize all `S` full-size key vectors of dimension 128, defeating the purpose of caching small latents.

**With absorption** (the trick): Precompute a compressed query, then dot directly with cached latents:

```
# Precompute once per query token (not per cached token):
q_compressed = W_UK_i^T @ q_t_i^C   # [512, 128] @ [128] -> [512]

# For each cached token j = 1..S:
score_j = q_compressed^T @ c_j^KV   # [512]^T @ [512] -> scalar
```

The precomputation costs `O(d_c * d_h)` = `O(512 * 128)` = `O(65,536)` FLOPs once. Then each of the `S` cached tokens requires only a dot product of two `[512]` vectors: `O(d_c)` = `O(512)` per token, giving total cost `O(S * d_c)` = `O(512 * S)`.

**Comparison**: Without absorption costs `O(S * 128 * 512)`. With absorption costs `O(128 * 512 + S * 512)`. For any sequence length `S > 1`, absorption wins. At `S = 4096`, the ratio is:

```python
S = 4096
without = S * 128 * 512            # 268,435,456 FLOPs
with_abs = 128 * 512 + S * 512     # 65,536 + 2,097,152 = 2,162,688 FLOPs
print(f"Without absorption: {without:,} FLOPs")
print(f"With absorption:    {with_abs:,} FLOPs")
print(f"Speedup: {without / with_abs:.0f}x")
# Without absorption: 268,435,456 FLOPs
# With absorption:    2,162,688 FLOPs
# Speedup: 124x
```

The key mental model: absorption transforms the problem from "multiply a large matrix by each cached vector" into "compress the query once, then do cheap dot products against the already-compressed cache." This is why MLA never needs to decompress the latent during generation.


## Memory Savings Derivation

Now let us derive the concrete memory savings MLA provides. The KV cache per token for each approach:

### Standard MHA

```
Cache per token = 2 * n_h * d_h * n_layers    (elements, factor of 2 for K and V)
```

### GQA (with n_g groups)

```
Cache per token = 2 * n_g * d_h * n_layers    (elements)
```

### MLA (DeepSeek-V2)

MLA caches the latent vector `c_t^KV` plus the decoupled RoPE key `k_t^R`:

```
Cache per token = (d_c + d_h_R) * n_layers     (elements, just one vector, not K+V separately)
```

Note: there is no factor of 2 because both K and V are jointly encoded in the single latent `c_t^KV`.

### DeepSeek-V2 Concrete Numbers

For DeepSeek-V2 (236B total params, 21B activated):
- n_h = 128 (attention heads)
- d_h = 128 (per-head dimension)
- d_c = 512 (KV compression dimension, set to 4 * d_h)
- d_h_R = 64 (decoupled RoPE dimension, set to d_h / 2)
- n_layers = 60

```python
# DeepSeek-V2 parameters
n_h = 128
d_h = 128
d_c = 512       # 4 * d_h
d_h_R = 64      # d_h / 2
n_layers = 60

# MHA cache per token (elements)
mha_cache = 2 * n_h * d_h * n_layers
print(f"MHA cache per token: {mha_cache:,} elements")
# Output: 1,966,080 elements

# GQA cache per token (8 groups, like Llama)
n_g = 8
gqa_cache = 2 * n_g * d_h * n_layers
print(f"GQA (8 groups) cache per token: {gqa_cache:,} elements")
# Output: 122,880 elements

# MLA cache per token (elements)
mla_cache = (d_c + d_h_R) * n_layers
print(f"MLA cache per token: {mla_cache:,} elements")
# Output: 34,560 elements

# Compression ratios
print(f"\nMLA vs MHA: {mha_cache / mla_cache:.1f}x reduction ({(1 - mla_cache/mha_cache)*100:.1f}% smaller)")
print(f"MLA vs GQA-8: {gqa_cache / mla_cache:.1f}x reduction ({(1 - mla_cache/gqa_cache)*100:.1f}% smaller)")
# MLA vs MHA: 56.9x reduction (98.2% smaller)
# MLA vs GQA-8: 3.6x reduction (71.9% smaller)
```

The paper reports a 93.3% KV cache reduction compared to the original DeepSeek 67B (which used MHA). The
remaining cache corresponds to GQA with approximately 2.25 groups:

```python
# MLA is equivalent to GQA with how many groups?
equivalent_gqa_groups = mla_cache / (2 * d_h * n_layers)
print(f"MLA equivalent GQA groups: {equivalent_gqa_groups:.2f}")
# Output: 2.25 groups
```

This is remarkable: MLA uses cache equivalent to GQA with only 2.25 groups, yet achieves *better* quality than
full MHA with 128 heads. The DeepSeek-V2 ablation studies confirm this (Table 9 in the paper): on MMLU, BBH,
C-Eval, and CMMLU benchmarks, MLA consistently matches or exceeds MHA performance.

### Translating to Bytes: A Concrete Comparison

Let us compare KV cache memory for a practical scenario. Suppose we want to serve sequences of 4096 tokens
with a batch size of 64:

```python
import numpy as np

seq_len = 4096
batch_size = 64
bytes_per_elem = 2  # FP16

# For a Llama-like model (32 layers, GQA-8, head_dim=128)
llama_n_layers = 32
llama_n_kv_heads = 8
llama_head_dim = 128

llama_kv_bytes = 2 * llama_n_kv_heads * llama_head_dim * llama_n_layers * bytes_per_elem * seq_len * batch_size
print(f"Llama 8B GQA KV cache: {llama_kv_bytes / (1024**3):.1f} GB")

# For DeepSeek-V2 (60 layers, MLA with d_c=512, d_h_R=64)
ds_kv_bytes = (512 + 64) * 60 * bytes_per_elem * seq_len * batch_size
print(f"DeepSeek-V2 MLA KV cache: {ds_kv_bytes / (1024**3):.1f} GB")

# Despite having TWICE as many layers, DeepSeek-V2 uses less KV cache
print(f"\nDeepSeek-V2 (60 layers) vs Llama 8B (32 layers):")
print(f"  Llama: {llama_kv_bytes / (1024**3):.1f} GB")
print(f"  DeepSeek-V2: {ds_kv_bytes / (1024**3):.1f} GB")
print(f"  Ratio: {llama_kv_bytes / ds_kv_bytes:.2f}x more cache for Llama")
```

Expected output:
```
Llama 8B GQA KV cache: 16.0 GB
DeepSeek-V2 MLA KV cache: 18.1 GB  (for 60 layers vs 32!)
```

Despite having nearly twice as many layers (60 vs 32), DeepSeek-V2s total KV cache is only slightly larger than Llama 8Bs (18.1 GB vs 16.0 GB). On a per-layer basis, MLA stores 576 elements per token (d_c + d_h_R = 512 + 64) versus GQAs 2,048 elements (2 * 8 * 128), a 3.6x reduction. This is the fair comparison: layer-for-layer, MLA is dramatically more efficient.


## The Compute-Memory Tradeoff

MLA is not free. To recover keys and values from the latent vector during attention, the model must perform
matrix multiplications (the up-projections). However, DeepSeek demonstrates that this tradeoff is overwhelmingly
favorable for inference:

1. **Memory bandwidth is the bottleneck during generation, not compute.** The autoregressive decode phase is
   memory-bound (loading KV cache dominates latency). Reducing cache size directly reduces the bytes loaded
   per token, which directly improves generation throughput.

2. **The matrix absorption trick eliminates most of the decompression cost.** By absorbing `W_UK` into the
   query projection and `W_UV` into the output projection, we never actually decompress the latent. The
   attention computation happens directly in the compressed space.

3. **The throughput improvement is dramatic.** DeepSeek-V2 achieves 5.76x higher maximum generation throughput
   compared to DeepSeek 67B, primarily because the smaller KV cache enables much larger batch sizes.

The mental model to carry forward:

> **MLA trades compute (decompression via up-projection matrices) for memory (smaller KV cache). During
> generation, where memory bandwidth is the bottleneck, this trade is enormously profitable. The matrix
> absorption trick further minimizes the compute cost, making MLA strictly better than MHA in practice.**

This is the opposite tradeoff from standard attention, which uses large caches (memory) to avoid recomputation
(compute). MLA inverts the equation: use a tiny cache and spend a small amount of extra compute to recover what
you need. In the memory-bound regime of autoregressive generation, this is the correct engineering choice.


