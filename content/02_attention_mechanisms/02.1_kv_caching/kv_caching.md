# Module 1: Transformer Inference Mechanics

> The difference between understanding transformers and _really_ understanding them is knowing exactly which bytes move where, and why that determines everything about your inference costs.

---

## Learning Objectives

By the end of this module, you will:

- Trace tensor shapes through every operation and explain _why_ each shape is what it is
- Understand the KV cache at the byte level—not just "it caches K and V" but exactly what's stored and why
- Calculate KV cache memory requirements from first principles (and know when the formulas lie)
- Explain why GQA exists, what problem it solves, and the exact tradeoff it makes
- Understand the prefill/decode split at a level where you can predict which phase will bottleneck

---

## The Insight That Changes Everything

Here's what most tutorials get wrong: they explain transformers as a sequence of operations. But for inference engineering, you need to think about transformers as a **memory access pattern**.

Every forward pass is fundamentally:

1. Read weights from HBM → do math → write activations
2. Read activations from HBM → do math → write activations
3. Repeat

The "do math" part is almost free on modern GPUs. The reads and writes are what cost you.

**The single most important number in LLM inference:**

```
Llama 3.1 8B decode step:
- Weights to read: 16 GB (FP16)
- FLOPs to compute: ~16 billion
- A100 memory bandwidth: 2 TB/s
- A100 FP16 compute: 312 TFLOPS

Time to read weights: 16 GB / 2 TB/s = 8 ms
Time to compute = 0.05 ms

The GPU spends 99.4% of decode time waiting for memory.
```

This is why everything in LLM inference optimization is about memory—not compute.

---

## The Token Generation Pipeline: A Byte-Level View

Let's trace what actually happens when you generate one token. Not the conceptual flow—the actual bytes.

### Step 1: Tokenization (CPU, negligible)

```python
# Input: "The capital of France is"
# Output: [464, 3139, 286, 4881, 318]  (5 tokens)

# Memory: 5 × 4 bytes = 20 bytes (int32 token IDs)
# This is nothing. Tokenization is never your bottleneck.
```

### Step 2: Embedding Lookup (GPU, memory-bound)

```python
# Embedding table: [vocab_size, hidden_dim] = [128256, 4096]
# Size: 128256 × 4096 × 2 bytes = 1.05 GB

# Operation: Gather 5 rows from the table
# Output: [5, 4096] = 40 KB

# This is a pure memory operation—no compute.
# You read 5 × 4096 × 2 = 40 KB from a 1 GB table.
```

**Insight #1: The embedding table is 1 GB but you only ever read tiny slices of it.** This is why embedding tables don't benefit much from quantization—you're not reading the whole thing, just scattered rows. The memory bandwidth cost is the random access pattern, not the table size.

### Step 3: The Transformer Block (where the money is)

This is 95%+ of your inference time. Let's break it down operation by operation.

Let's trace what actually happens when you generate one token. Not the conceptual flow—the actual bytes.

### Step 1: Tokenization (CPU, negligible)

```python
# Input: "The capital of France is"
# Output: [464, 3139, 286, 4881, 318]  (5 tokens)

# Memory: 5 × 4 bytes = 20 bytes (int32 token IDs)
# This is nothing. Tokenization is never your bottleneck.
```

### Step 2: Embedding Lookup (GPU, memory-bound)

````python
# Embedding table: [vocab_size, hidden_dim] = [128256, 4096]
# Size: 128256 × 4096 × 2 bytes = 1.05 GB

# Operation: Gather 5 rows from the table
# Output: [5, 4096] = 40 KB

# This is a pure memory operation—no compute.
# You read 5 × 4096 × 2 = 40 KB from a 1 GB table.

#### 3a: RMSNorm (cheap, memory-bound)

```python
# Input: [batch, seq, hidden] = [1, 5, 4096]
# Weights: [hidden] = [4096] = 8 KB
# Output: [1, 5, 4096]

# Operation: x * rsqrt(mean(x²) + eps) * weight
# FLOPs: ~3 × batch × seq × hidden = 61K FLOPs
# Memory: Read 40 KB input + 8 KB weights, write 40 KB output

# This is so cheap it's essentially free.
````

#### 3b: Attention Projections (the first big memory read)

This is where things get expensive. You have four projection matrices:

```python
# Llama 3.1 8B uses GQA: 32 query heads, 8 KV heads, 128 dim per head

W_q: [4096, 4096]   = 32 MB   # 32 heads × 128 dim = 4096
W_k: [4096, 1024]   = 8 MB    # 8 KV heads × 128 dim = 1024
W_v: [4096, 1024]   = 8 MB    # 8 KV heads × 128 dim = 1024
W_o: [4096, 4096]   = 32 MB   # Output projection

Total per layer: 80 MB
Total for 32 layers: 2.56 GB  # Just the attention projections!
```

**Insight #2: The Q and O projections are 4× larger than K and V projections.** This is the GQA tradeoff—you save memory on KV cache but the projection weights are still dominated by Q and O. GQA doesn't reduce model size, only KV cache size.

```python
# The actual computation:
# Input: [1, 5, 4096]

Q = input @ W_q  # [1, 5, 4096] @ [4096, 4096] → [1, 5, 4096]
K = input @ W_k  # [1, 5, 4096] @ [4096, 1024] → [1, 5, 1024]
V = input @ W_v  # [1, 5, 4096] @ [4096, 1024] → [1, 5, 1024]

# Reshape for multi-head attention:
Q = Q.view(1, 5, 32, 128).transpose(1, 2)  # [1, 32, 5, 128]
K = K.view(1, 5, 8, 128).transpose(1, 2)   # [1, 8, 5, 128]
V = V.view(1, 5, 8, 128).transpose(1, 2)   # [1, 8, 5, 128]
```

#### 3c: The Attention Computation (where GQA gets interesting)

Here's where most explanations gloss over the details. With GQA, you have 32 query heads but only 8 KV heads. How does that work?

```python
# GQA: Each KV head serves 4 query heads (32/8 = 4)
#
# Query heads 0-3   share KV head 0
# Query heads 4-7   share KV head 1
# Query heads 8-11  share KV head 2
# ...
# Query heads 28-31 share KV head 7

# Implementation: Expand K and V to match Q's head count
K_expanded = K.repeat_interleave(4, dim=1)  # [1, 8, 5, 128] → [1, 32, 5, 128]
V_expanded = V.repeat_interleave(4, dim=1)  # [1, 8, 5, 128] → [1, 32, 5, 128]

# Now standard attention:
scores = Q @ K_expanded.transpose(-2, -1) / sqrt(128)  # [1, 32, 5, 5]
attn_weights = softmax(scores, dim=-1)                  # [1, 32, 5, 5]
attn_output = attn_weights @ V_expanded                 # [1, 32, 5, 128]
```

**Insight #3: The repeat_interleave doesn't actually copy memory in optimized implementations.** FlashAttention and PagedAttention use index arithmetic to avoid the expansion. But conceptually, each query head is attending to the same KV cache—just with different learned query projections.

**Insight #4: The attention matrix is [seq_len, seq_len].** For a 4096-token context, that's 16M elements per head, 512M elements total. This is why FlashAttention matters—it never materializes this full matrix.

#### 3d: The MLP Block (the other half of the parameters)

The MLP is deceptively simple but contains ~2/3 of the model's parameters:

```python
# Llama uses SwiGLU activation, which means 3 projections instead of 2:

W_gate: [4096, 14336]  = 117 MB
W_up:   [4096, 14336]  = 117 MB
W_down: [14336, 4096]  = 117 MB

Total per layer: 351 MB
Total for 32 layers: 11.2 GB  # The MLP is 70% of the model!
```

```python
# The computation:
gate = input @ W_gate           # [1, 5, 4096] @ [4096, 14336] → [1, 5, 14336]
up = input @ W_up               # [1, 5, 4096] @ [4096, 14336] → [1, 5, 14336]
hidden = silu(gate) * up        # Element-wise, [1, 5, 14336]
output = hidden @ W_down        # [1, 5, 14336] @ [14336, 4096] → [1, 5, 4096]
```

**Insight #5: The intermediate dimension (14336) is 3.5× the hidden dimension (4096).** This ratio is a design choice. Larger intermediate = more capacity but more memory. Llama chose 3.5×; GPT-2 used 4×. This is why MLP dominates model size.

**Insight #6: SwiGLU requires 3 weight matrices instead of 2.** Classic transformers use `relu(x @ W1) @ W2`. SwiGLU uses `silu(x @ W_gate) * (x @ W_up) @ W_down`. The extra matrix is why Llama's MLP is 50% larger than you'd expect from the intermediate dimension alone.

### Step 4: The LM Head (one more big matrix)

```python
# LM Head: [hidden, vocab] = [4096, 128256]
# Size: 4096 × 128256 × 2 = 1.05 GB

# This is the same size as the embedding table!
# In fact, many models tie these weights (share the same matrix).
# Llama 3.1 does NOT tie weights—it has separate embedding and LM head.

logits = final_hidden @ W_lm_head  # [1, 5, 4096] @ [4096, 128256] → [1, 5, 128256]

# For generation, you only need the last token's logits:
next_token_logits = logits[:, -1, :]  # [1, 128256]
```

**Insight #7: The LM head is 1 GB but you only use one row of output during generation.** You compute logits for all 128K vocab entries, but you only sample one token. This is unavoidable—you need the full distribution to sample from.

---

## The KV Cache: Why It Exists and What It Actually Stores

### The Problem KV Cache Solves

Without KV cache, generating 100 tokens requires:

```
Token 1:  Compute attention over 1 token
Token 2:  Compute attention over 2 tokens (recompute token 1's K,V)
Token 3:  Compute attention over 3 tokens (recompute tokens 1-2's K,V)
...
Token 100: Compute attention over 100 tokens (recompute tokens 1-99's K,V)

Total K,V computations: 1 + 2 + 3 + ... + 100 = 5,050
```

With KV cache:

```
Token 1:  Compute K,V for token 1, store in cache
Token 2:  Compute K,V for token 2, read token 1's K,V from cache
Token 3:  Compute K,V for token 3, read tokens 1-2's K,V from cache
...
Token 100: Compute K,V for token 100, read tokens 1-99's K,V from cache

Total K,V computations: 100 (one per token)
```

**The KV cache trades memory for compute: O(N) memory to avoid O(N²) compute.**

### What's Actually in the KV Cache

Let's be precise. For each layer, you store:

```python
# Per layer, per sequence:
K_cache: [num_kv_heads, seq_len, head_dim]  # [8, seq_len, 128]
V_cache: [num_kv_heads, seq_len, head_dim]  # [8, seq_len, 128]

# For Llama 3.1 8B with seq_len=4096:
K_cache per layer: 8 × 4096 × 128 × 2 bytes = 8 MB
V_cache per layer: 8 × 4096 × 128 × 2 bytes = 8 MB
Total per layer: 16 MB

# For all 32 layers:
Total KV cache: 32 × 16 MB = 512 MB per sequence
```

**Insight #8: The KV cache stores the _projected_ K and V, not the original hidden states.** You can't reconstruct the hidden states from the KV cache. This is important—if you wanted to branch the generation (beam search), you'd need to copy the KV cache, not recompute from hidden states.

### The KV Cache Memory Formula (and when it lies)

The standard formula:

```
KV_cache_bytes = 2 × num_layers × num_kv_heads × head_dim × seq_len × batch_size × dtype_bytes
```

Let's verify:

```python
# Llama 3.1 8B, batch=1, seq=4096, FP16
kv_cache = 2 × 32 × 8 × 128 × 4096 × 1 × 2
         = 536,870,912 bytes
         = 512 MB ✓
```

**When the formula lies:**

1. **PagedAttention adds overhead.** vLLM allocates KV cache in blocks (typically 16 tokens). If your sequence is 4097 tokens, you allocate 4112 tokens worth of cache. ~1% overhead for long sequences, more for short ones.

2. **FP16 KV cache even with INT4 weights.** Most quantized models still use FP16 for KV cache because quantizing KV hurts quality more than quantizing weights. Your "4-bit model" still has 16-bit KV cache.

3. **Speculative decoding multiplies KV cache.** If you're verifying 4 draft tokens at once, you need KV cache space for all 4 potential paths.

4. **Prefix caching shares memory.** If 10 requests share the same system prompt, vLLM stores one copy of that prefix's KV cache. The formula assumes no sharing.

### KV Cache Growth Visualization

```
PREFILL: Process "The capital of France is" (5 tokens)
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 0:  K[8, 5, 128]  V[8, 5, 128]   = 20 KB                      │
│ Layer 1:  K[8, 5, 128]  V[8, 5, 128]   = 20 KB                      │
│ ...                                                                 │
│ Layer 31: K[8, 5, 128]  V[8, 5, 128]   = 20 KB                      │
│                                                                     │
│ Total after prefill: 32 × 20 KB = 640 KB                            │
└─────────────────────────────────────────────────────────────────────┘

DECODE: Generate "Paris" (1 token)
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 0:  K[8, 6, 128]  V[8, 6, 128]   = 24 KB  (+4 KB)             │
│ Layer 1:  K[8, 6, 128]  V[8, 6, 128]   = 24 KB  (+4 KB)             │
│ ...                                                                 │
│ Layer 31: K[8, 6, 128]  V[8, 6, 128]   = 24 KB  (+4 KB)             │
│                                                                     │
│ Total after 1 decode: 32 × 24 KB = 768 KB  (+128 KB)                │
└─────────────────────────────────────────────────────────────────────┘

After 1000 generated tokens:
┌─────────────────────────────────────────────────────────────────────┐
│ Total KV cache: 32 × 8 × 1005 × 128 × 2 × 2 = 131 MB                │
│                                                                     │
│ That's 128 KB per generated token, or 128 MB per 1000 tokens.       │
└─────────────────────────────────────────────────────────────────────┘
```

**Insight #9: KV cache grows linearly with sequence length, but the _rate_ of growth depends on num_kv_heads.** Llama 3.1 8B grows at 128 KB/token. Llama 3.1 70B also grows at 128 KB/token (same 8 KV heads). But GPT-4 (estimated 96 KV heads) would grow at ~1.5 MB/token.

---

## Attention Variants: The Memory-Quality Tradeoff

### Why MQA and GQA Exist

The original transformer (2017) used Multi-Head Attention (MHA): each query head has its own K and V heads. This means KV cache scales with num_heads.

```
MHA KV cache per token = 2 × num_layers × num_heads × head_dim × dtype_bytes

GPT-3 175B (96 heads, 128 dim, 96 layers):
= 2 × 96 × 96 × 128 × 2 = 4.7 MB per token

At 4096 tokens: 19 GB of KV cache. Per sequence.
```

This is why Google invented Multi-Query Attention (MQA) in 2019: all query heads share a single K and V head.

```
MQA KV cache per token = 2 × num_layers × 1 × head_dim × dtype_bytes

Same model with MQA:
= 2 × 96 × 1 × 128 × 2 = 49 KB per token

96× reduction in KV cache!
```

**The problem:** MQA hurts model quality. All those query heads are fighting over one KV representation.

**The solution:** Grouped-Query Attention (GQA), introduced in 2023. A middle ground—groups of query heads share KV heads.

### GQA: The Math

```
GQA with G groups:
- num_query_heads query heads
- num_kv_heads = num_query_heads / G  KV heads
- Each KV head serves G query heads

Llama 3.1 8B: 32 query heads, 8 KV heads, G=4
Llama 3.1 70B: 64 query heads, 8 KV heads, G=8
```

**Insight #10: Larger models use more aggressive grouping.** Llama 8B uses G=4 (4× KV reduction). Llama 70B uses G=8 (8× KV reduction). The larger model can "afford" more sharing because it has more capacity elsewhere.

### The Quality Impact

From the GQA paper (Ainslie et al., 2023):

```
Model          Attention   MMLU    HellaSwag   Relative KV Cache
─────────────────────────────────────────────────────────────────
Llama 7B       MHA         35.1    76.1        1.0× (baseline)
Llama 7B       MQA         33.8    74.9        0.03× (32× smaller)
Llama 7B       GQA-4       34.9    75.8        0.125× (8× smaller)
Llama 7B       GQA-8       34.7    75.6        0.0625× (16× smaller)
```

**Insight #11: GQA-4 recovers almost all MHA quality while using 8× less KV cache.** The quality loss from MHA→GQA-4 is ~0.3% on MMLU. The memory savings is 8×. This is why every modern model uses GQA.

### Visualizing the Difference

```
MHA (32 query heads, 32 KV heads):
┌─────────────────────────────────────────────────────────────────────┐
│ Q₀ ─→ K₀,V₀    Q₈ ─→ K₈,V₈     Q₁₆─→ K₁₆,V₁₆   Q₂₄─→ K₂₄,V₂₄     │
│ Q₁ ─→ K₁,V₁    Q₉ ─→ K₉,V₉     Q₁₇─→ K₁₇,V₁₇   Q₂₅─→ K₂₅,V₂₅     │
│ Q₂ ─→ K₂,V₂    Q₁₀─→ K₁₀,V₁₀   Q₁₈─→ K₁₈,V₁₈   Q₂₆─→ K₂₆,V₂₆     │
│ Q₃ ─→ K₃,V₃    Q₁₁─→ K₁₁,V₁₁   Q₁₉─→ K₁₉,V₁₉   Q₂₇─→ K₂₇,V₂₇     │
│ Q₄ ─→ K₄,V₄    Q₁₂─→ K₁₂,V₁₂   Q₂₀─→ K₂₀,V₂₀   Q₂₈─→ K₂₈,V₂₈     │
│ Q₅ ─→ K₅,V₅    Q₁₃─→ K₁₃,V₁₃   Q₂₁─→ K₂₁,V₂₁   Q₂₉─→ K₂₉,V₂₉     │
│ Q₆ ─→ K₆,V₆    Q₁₄─→ K₁₄,V₁₄   Q₂₂─→ K₂₂,V₂₂   Q₃₀─→ K₃₀,V₃₀     │
│ Q₇ ─→ K₇,V₇    Q₁₅─→ K₁₅,V₁₅   Q₂₃─→ K₂₃,V₂₃   Q₃₁─→ K₃₁,V₃₁     │
│                                                                     │
│ KV cache: 32 K tensors + 32 V tensors = 64 tensors per layer        │
└─────────────────────────────────────────────────────────────────────┘

GQA-4 (32 query heads, 8 KV heads):
┌─────────────────────────────────────────────────────────────────────┐
│ Q₀ ─┐            Q₈ ─┐            Q₁₆─┐            Q₂₄─┐            │
│ Q₁ ─┼─→ K₀,V₀    Q₉ ─┼─→ K₂,V₂    Q₁₇─┼─→ K₄,V₄    Q₂₅─┼─→ K₆,V₆   │
│ Q₂ ─┤            Q₁₀─┤            Q₁₈─┤            Q₂₆─┤            │
│ Q₃ ─┘            Q₁₁─┘            Q₁₉─┘            Q₂₇─┘            │
│                                                                     │
│ Q₄ ─┐            Q₁₂─┐            Q₂₀─┐            Q₂₈─┐            │
│ Q₅ ─┼─→ K₁,V₁    Q₁₃─┼─→ K₃,V₃    Q₂₁─┼─→ K₅,V₅    Q₂₉─┼─→ K₇,V₇   │
│ Q₆ ─┤            Q₁₄─┤            Q₂₂─┤            Q₃₀─┤            │
│ Q₇ ─┘            Q₁₅─┘            Q₂₃─┘            Q₃₁─┘            │
│                                                                     │
│ KV cache: 8 K tensors + 8 V tensors = 16 tensors per layer          │
│ 4× smaller than MHA!                                                │
└─────────────────────────────────────────────────────────────────────┘

MQA (32 query heads, 1 KV head):
┌─────────────────────────────────────────────────────────────────────┐
│ Q₀ ─┐                                                               │
│ Q₁ ─┤                                                               │
│ Q₂ ─┤                                                               │
│ ... ├─────────────────────→ K₀,V₀                                   │
│ Q₃₀─┤                                                               │
│ Q₃₁─┘                                                               │
│                                                                     │
│ KV cache: 1 K tensor + 1 V tensor = 2 tensors per layer             │
│ 32× smaller than MHA! But quality suffers.                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Prefill vs Decode: Two Completely Different Problems

This is the most important section for inference engineering. Prefill and decode have different bottlenecks, different optimization strategies, and increasingly, different hardware.

### Prefill: Compute-Bound (Usually)

During prefill, you process the entire prompt in one forward pass:

```python
# Prefill for 1000-token prompt:
Input: [1, 1000, 4096]

# Attention computation:
Q @ K^T: [1, 32, 1000, 128] @ [1, 32, 128, 1000] → [1, 32, 1000, 1000]
         = 32 × 1000 × 128 × 1000 = 4.1 billion FLOPs

# MLP computation (per layer):
x @ W_gate: [1, 1000, 4096] @ [4096, 14336] = 58.7 billion FLOPs
x @ W_up:   [1, 1000, 4096] @ [4096, 14336] = 58.7 billion FLOPs
h @ W_down: [1, 1000, 14336] @ [14336, 4096] = 58.7 billion FLOPs

# Total per layer: ~180 billion FLOPs
# Total for 32 layers: ~5.8 trillion FLOPs
```

**Arithmetic intensity during prefill:**

```
FLOPs: 5.8 trillion
Bytes read: ~16 GB (model weights, read once)
Arithmetic intensity: 5.8T / 16G = 362 FLOPs/byte

A100 ridge point: 312 TFLOPS / 2 TB/s = 156 FLOPs/byte

362 > 156 → Prefill is compute-bound on A100
```

**Insight #12: Prefill arithmetic intensity scales with sequence length.** Longer prompts = more compute-bound. A 100-token prompt might be memory-bound; a 10,000-token prompt is definitely compute-bound.

### Decode: Memory-Bound (Always)

During decode, you generate one token at a time:

```python
# Decode for 1 new token (with 1000 tokens already in cache):
Input: [1, 1, 4096]

# Attention computation:
Q @ K^T: [1, 32, 1, 128] @ [1, 32, 128, 1001] → [1, 32, 1, 1001]
         = 32 × 1 × 128 × 1001 = 4.1 million FLOPs

# MLP computation (per layer):
x @ W_gate: [1, 1, 4096] @ [4096, 14336] = 58.7 million FLOPs
x @ W_up:   [1, 1, 4096] @ [4096, 14336] = 58.7 million FLOPs
h @ W_down: [1, 1, 14336] @ [14336, 4096] = 58.7 million FLOPs

# Total per layer: ~180 million FLOPs
# Total for 32 layers: ~5.8 billion FLOPs
```

**Arithmetic intensity during decode:**

```
FLOPs: 5.8 billion
Bytes read: ~16 GB (model weights) + ~130 MB (KV cache for 1000 tokens)
Arithmetic intensity: 5.8B / 16.1G = 0.36 FLOPs/byte

0.36 << 156 → Decode is extremely memory-bound
```

**Insight #13: Decode arithmetic intensity is ~1000× lower than prefill.** This is why decode is always memory-bound. You're reading the entire model to generate one token.

### The Batching Insight

Here's where it gets interesting. What if you batch multiple decode requests together?

```python
# Decode for 32 sequences simultaneously:
Input: [32, 1, 4096]

# FLOPs: 32 × 5.8 billion = 186 billion
# Bytes read: ~16 GB (weights, shared) + ~4.2 GB (32 × 130 MB KV cache)
# Arithmetic intensity: 186B / 20.2G = 9.2 FLOPs/byte

# Still memory-bound, but 25× better than batch=1!
```

**Insight #14: Batching amortizes weight reads across sequences.** With batch=32, you read the weights once but do 32× the compute. This is why high-throughput serving uses large batches.

**But there's a catch:** KV cache scales with batch size. At some point, you run out of memory for KV cache before you can batch enough to become compute-bound.

```
Llama 3.1 8B on A100 80GB:
- Model weights: 16 GB
- Available for KV cache: ~60 GB
- KV cache per sequence (4096 tokens): 512 MB
- Max batch size: 60 GB / 512 MB ≈ 117 sequences

At batch=117:
- Arithmetic intensity: 117 × 5.8B / (16G + 60G) = 8.9 FLOPs/byte
- Still memory-bound! But much better than batch=1.
```

**Insight #15: You can never make decode compute-bound through batching alone.** The KV cache grows with batch size, so you hit memory limits before reaching the ridge point. This is the fundamental constraint of LLM inference.

---

## The Memory Bandwidth Wall

Let's derive the theoretical maximum decode speed from first principles.

```
Llama 3.1 8B on A100 80GB:
- Model weights: 16 GB (FP16)
- Memory bandwidth: 2 TB/s
- Time to read weights: 16 GB / 2 TB/s = 8 ms

Theoretical max decode speed = 1 token / 8 ms = 125 tokens/sec

This is a HARD CEILING. No optimization can exceed this.
```

**Insight #16: The memory bandwidth wall is model_size / bandwidth.** This is the most important equation in LLM inference:

```
max_tokens_per_second = memory_bandwidth / model_size_bytes
```

Let's verify with real benchmarks:

| Model            | Size (FP16) | A100 BW | Theoretical Max | Actual (vLLM) | Efficiency |
| ---------------- | ----------- | ------- | --------------- | ------------- | ---------- |
| Llama 8B         | 16 GB       | 2 TB/s  | 125 tok/s       | 95-110 tok/s  | 76-88%     |
| Llama 70B        | 140 GB      | 2 TB/s  | 14.3 tok/s      | 11-13 tok/s   | 77-91%     |
| Llama 70B (TP=8) | 17.5 GB/GPU | 16 TB/s | 914 tok/s       | 650-750 tok/s | 71-82%     |

**Insight #17: Real systems achieve 70-90% of theoretical bandwidth.** The gap comes from:

- KV cache reads (not just weights)
- Kernel launch overhead
- Memory access patterns (not perfectly sequential)
- Synchronization in multi-GPU setups

### Breaking the Wall: Your Options

1. **Quantization**: Reduce model size → read fewer bytes
   - INT8: 2× faster theoretical max
   - INT4: 4× faster theoretical max
   - But: quality tradeoff

2. **Tensor Parallelism**: More GPUs → more bandwidth
   - TP=8 on A100: 16 TB/s aggregate bandwidth
   - But: communication overhead, diminishing returns

3. **Speculative Decoding**: Generate multiple tokens per weight read
   - Draft model proposes N tokens, target verifies in one pass
   - But: acceptance rate < 100%, draft model overhead

4. **Batching**: Amortize weight reads across sequences
   - But: KV cache limits batch size

---

## Putting It All Together: A Complete Example

Let's trace a real inference request end-to-end.

**Setup:**

- Model: Llama 3.1 8B
- Hardware: A100 80GB
- Prompt: 500 tokens
- Generation: 200 tokens
- Batch size: 1

**Prefill Phase:**

```
Input: [1, 500, 4096]

Memory reads:
- Model weights: 16 GB (read once)
- No KV cache yet

Compute:
- Attention: 500² × 32 × 128 × 32 layers = 32.8B FLOPs
- MLP: 500 × 4096 × 14336 × 3 × 32 layers = 2.8T FLOPs
- Total: ~2.8T FLOPs

Time estimate:
- Memory time: 16 GB / 2 TB/s = 8 ms
- Compute time: 2.8T / 312T = 9 ms
- Prefill is roughly balanced, ~17 ms total

Output:
- KV cache populated: 500 tokens × 128 KB/token = 64 MB
- First token generated
```

**Decode Phase (200 tokens):**

```
Per token:
- Memory reads: 16 GB weights + growing KV cache
- Compute: ~5.8B FLOPs
- Time: ~8-10 ms per token (memory-bound)

Total decode time: 200 × 9 ms = 1.8 seconds

Final KV cache: 700 tokens × 128 KB/token = 90 MB
```

**Total Request:**

- TTFT (Time to First Token): ~17 ms
- TBT (Time Between Tokens): ~9 ms
- Total time: 17 ms + 200 × 9 ms = 1.82 seconds
- Throughput: 200 tokens / 1.82 s = 110 tokens/sec

---

## Key Takeaways

1. **LLM inference is a memory bandwidth problem, not a compute problem.** The GPU spends most of decode time waiting for memory transfers.

2. **The KV cache is the critical resource.** It determines your max batch size, max sequence length, and memory efficiency.

3. **GQA reduces KV cache 4-8× with minimal quality loss.** This is why every modern model uses it.

4. **Prefill is compute-bound, decode is memory-bound.** They need different optimizations (and increasingly, different hardware).

5. **The memory bandwidth wall is `model_size / bandwidth`.** This is the theoretical maximum decode speed.

6. **Batching helps but doesn't solve the problem.** KV cache grows with batch size, limiting how much you can batch.

7. **The formulas are approximations.** Real systems have overhead from PagedAttention blocks, kernel launches, and synchronization.

---

## What's Next

In Module 2, we'll dive into GPU memory engineering:

- The roofline model and how to use it
- GPU memory hierarchy (registers → L1 → L2 → HBM)
- Why FlashAttention matters (hint: it's about L2 cache, not compute)
- VRAM budgeting for production deployments

In Lab 1, you'll implement a minimal transformer with KV cache and measure the memory/compute tradeoffs yourself.

---

## References

1. Vaswani et al. "Attention Is All You Need" (2017) - Original transformer
2. Shazeer "Fast Transformer Decoding: One Write-Head is All You Need" (2019) - MQA
3. Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models" (2023) - GQA
4. Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023) - vLLM
5. Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022)
6. Llama 3.1 Model Card - Meta AI
