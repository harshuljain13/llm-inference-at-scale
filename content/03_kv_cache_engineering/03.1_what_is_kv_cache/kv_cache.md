# 3.1 What the KV Cache Is

---

## The KV Cache: Why It Exists and What It Actually Stores

### The Problem KV Cache Solves

The fundamental issue with autoregressive generation is that attention is causal: each token attends to all previous tokens. Without caching, this means recomputing key and value projections for the entire history at every step.

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

**The KV cache trades memory for compute: O(N) memory to avoid O(N^2) compute.**

### What Is Actually in the KV Cache

Let us be precise about what gets stored. For each layer, the cache holds the projected key and value tensors for every token seen so far:

```python
# Per layer, per sequence:
K_cache: [num_kv_heads, seq_len, head_dim]  # [8, seq_len, 128]
V_cache: [num_kv_heads, seq_len, head_dim]  # [8, seq_len, 128]

# For Llama 3.1 8B with seq_len=4096:
K_cache per layer: 8 x 4096 x 128 x 2 bytes = 8 MB
V_cache per layer: 8 x 4096 x 128 x 2 bytes = 8 MB
Total per layer: 16 MB

# For all 32 layers:
Total KV cache: 32 x 16 MB = 512 MB per sequence
```

The KV cache stores the _projected_ K and V, not the original hidden states. You cannot reconstruct the hidden states from the KV cache. This is important: if you want to branch the generation (beam search), you need to copy the KV cache, not recompute from hidden states.

### The KV Cache Memory Formula (and when it lies)

The standard formula provides a first-order estimate of KV cache memory consumption. Understanding both the formula and its limitations is essential for capacity planning.

```
KV_cache_bytes = 2 x num_layers x num_kv_heads x head_dim x seq_len x batch_size x dtype_bytes
```

Let us verify with Llama 3.1 8B:

```python
# Llama 3.1 8B, batch=1, seq=4096, FP16
kv_cache = 2 x 32 x 8 x 128 x 4096 x 1 x 2
         = 536,870,912 bytes
         = 512 MB
```

**When the formula lies:**

1. **PagedAttention adds overhead.** vLLM allocates KV cache in blocks (typically 16 tokens). If your sequence is 4097 tokens, you allocate 4112 tokens worth of cache. This is roughly 1% overhead for long sequences, but more significant for short ones.

2. **FP16 KV cache even with INT4 weights.** Most quantized models still use FP16 for KV cache because quantizing KV hurts quality more than quantizing weights. Your "4-bit model" still has 16-bit KV cache.

3. **Speculative decoding multiplies KV cache.** If you are verifying 4 draft tokens at once, you need KV cache space for all 4 potential paths.

4. **Prefix caching shares memory.** If 10 requests share the same system prompt, vLLM stores one copy of that prefix's KV cache. The formula assumes no sharing.

### KV Cache Growth Visualization

The following diagram shows how the KV cache grows through prefill and decode phases. Notice that each generated token adds a fixed amount of memory across all 32 layers.

```
PREFILL: Process "The capital of France is" (5 tokens)
+---------------------------------------------------------------------+
| Layer 0:  K[8, 5, 128]  V[8, 5, 128]   = 20 KB                    |
| Layer 1:  K[8, 5, 128]  V[8, 5, 128]   = 20 KB                    |
| ...                                                                 |
| Layer 31: K[8, 5, 128]  V[8, 5, 128]   = 20 KB                    |
|                                                                     |
| Total after prefill: 32 x 20 KB = 640 KB                           |
+---------------------------------------------------------------------+

DECODE: Generate "Paris" (1 token)
+---------------------------------------------------------------------+
| Layer 0:  K[8, 6, 128]  V[8, 6, 128]   = 24 KB  (+4 KB)           |
| Layer 1:  K[8, 6, 128]  V[8, 6, 128]   = 24 KB  (+4 KB)           |
| ...                                                                 |
| Layer 31: K[8, 6, 128]  V[8, 6, 128]   = 24 KB  (+4 KB)           |
|                                                                     |
| Total after 1 decode: 32 x 24 KB = 768 KB  (+128 KB)               |
+---------------------------------------------------------------------+

After 1000 generated tokens:
+---------------------------------------------------------------------+
| Total KV cache: 32 x 8 x 1005 x 128 x 2 x 2 = 131 MB             |
|                                                                     |
| That is 128 KB per generated token, or 128 MB per 1000 tokens.     |
+---------------------------------------------------------------------+
```

KV cache grows linearly with sequence length, but the _rate_ of growth depends on num_kv_heads. Llama 3.1 8B grows at 128 KB/token. Llama 3.1 70B also grows at 128 KB/token (same 8 KV heads). But a model with 96 KV heads (estimated for GPT-4 class) would grow at roughly 1.5 MB/token.

---

## Attention Variants: The Memory-Quality Tradeoff

### Why MQA and GQA Exist

The original transformer (2017) used Multi-Head Attention (MHA): each query head has its own K and V heads. This means KV cache scales linearly with num_heads, which becomes prohibitive for large models.

```
MHA KV cache per token = 2 x num_layers x num_heads x head_dim x dtype_bytes

GPT-3 175B (96 heads, 128 dim, 96 layers):
= 2 x 96 x 96 x 128 x 2 = 4.7 MB per token

At 4096 tokens: 19 GB of KV cache. Per sequence.
```

This is why Google invented Multi-Query Attention (MQA) in 2019: all query heads share a single K and V head.

```
MQA KV cache per token = 2 x num_layers x 1 x head_dim x dtype_bytes

Same model with MQA:
= 2 x 96 x 1 x 128 x 2 = 49 KB per token

96x reduction in KV cache!
```

The problem with MQA is that it hurts model quality. All those query heads are fighting over one KV representation, which limits the model's ability to attend to different aspects of the input simultaneously.

The solution is Grouped-Query Attention (GQA), introduced in 2023. GQA provides a middle ground where groups of query heads share KV heads, balancing memory efficiency against representational capacity.

### GQA: The Math

```
GQA with G groups:
- num_query_heads query heads
- num_kv_heads = num_query_heads / G  KV heads
- Each KV head serves G query heads

Llama 3.1 8B: 32 query heads, 8 KV heads, G=4
Llama 3.1 70B: 64 query heads, 8 KV heads, G=8
```

Larger models use more aggressive grouping. Llama 8B uses G=4 (4x KV reduction). Llama 70B uses G=8 (8x KV reduction). The larger model can afford more sharing because it has more capacity elsewhere in the network to compensate.

### The Quality Impact

From the GQA paper (Ainslie et al., 2023):

```
Model          Attention   MMLU    HellaSwag   Relative KV Cache
---------------------------------------------------------------
Llama 7B       MHA         35.1    76.1        1.0x (baseline)
Llama 7B       MQA         33.8    74.9        0.03x (32x smaller)
Llama 7B       GQA-4       34.9    75.8        0.125x (8x smaller)
Llama 7B       GQA-8       34.7    75.6        0.0625x (16x smaller)
```

GQA-4 recovers almost all MHA quality while using 8x less KV cache. The quality loss from MHA to GQA-4 is roughly 0.3% on MMLU. The memory savings is 8x. This tradeoff is so favorable that every modern large language model uses GQA.

### Visualizing the Difference

The following diagrams show how query heads map to KV heads under each attention variant. The key insight is how dramatically the number of KV tensors stored per layer decreases as you move from MHA to GQA to MQA.

```
MHA (32 query heads, 32 KV heads):
+---------------------------------------------------------------------+
| Q0 -> K0,V0    Q8 -> K8,V8     Q16-> K16,V16   Q24-> K24,V24       |
| Q1 -> K1,V1    Q9 -> K9,V9     Q17-> K17,V17   Q25-> K25,V25       |
| Q2 -> K2,V2    Q10-> K10,V10   Q18-> K18,V18   Q26-> K26,V26       |
| Q3 -> K3,V3    Q11-> K11,V11   Q19-> K19,V19   Q27-> K27,V27       |
| Q4 -> K4,V4    Q12-> K12,V12   Q20-> K20,V20   Q28-> K28,V28       |
| Q5 -> K5,V5    Q13-> K13,V13   Q21-> K21,V21   Q29-> K29,V29       |
| Q6 -> K6,V6    Q14-> K14,V14   Q22-> K22,V22   Q30-> K30,V30       |
| Q7 -> K7,V7    Q15-> K15,V15   Q23-> K23,V23   Q31-> K31,V31       |
|                                                                     |
| KV cache: 32 K tensors + 32 V tensors = 64 tensors per layer       |
+---------------------------------------------------------------------+

GQA-4 (32 query heads, 8 KV heads):
+---------------------------------------------------------------------+
| Q0 -+            Q8 -+            Q16-+            Q24-+            |
| Q1 -+-> K0,V0   Q9 -+-> K2,V2   Q17-+-> K4,V4   Q25-+-> K6,V6    |
| Q2 -+            Q10-+            Q18-+            Q26-+            |
| Q3 -+            Q11-+            Q19-+            Q27-+            |
|                                                                     |
| Q4 -+            Q12-+            Q20-+            Q28-+            |
| Q5 -+-> K1,V1   Q13-+-> K3,V3   Q21-+-> K5,V5   Q29-+-> K7,V7    |
| Q6 -+            Q14-+            Q22-+            Q30-+            |
| Q7 -+            Q15-+            Q23-+            Q31-+            |
|                                                                     |
| KV cache: 8 K tensors + 8 V tensors = 16 tensors per layer         |
| 4x smaller than MHA!                                                |
+---------------------------------------------------------------------+

MQA (32 query heads, 1 KV head):
+---------------------------------------------------------------------+
| Q0 -+                                                               |
| Q1 -+                                                               |
| Q2 -+                                                               |
| ... +-------------------> K0,V0                                     |
| Q30-+                                                               |
| Q31-+                                                               |
|                                                                     |
| KV cache: 1 K tensor + 1 V tensor = 2 tensors per layer            |
| 32x smaller than MHA! But quality suffers.                          |
+---------------------------------------------------------------------+
```

---

