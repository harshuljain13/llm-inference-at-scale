# 3.7 FlashAttention: In Practice

---

## When FlashAttention Helps and When It Does Not

FlashAttention is not universally beneficial. Its advantage depends on the ratio between the quadratic attention matrix and the linear Q/K/V/O traffic. Understanding the boundary conditions prevents wasted engineering effort.

### Case 1: Short Sequences (N < 256)

For very short sequences, the attention matrix is small enough to fit in L2 cache even with standard attention. The N^2 term does not dominate:

```
N=128, d=128:
  Attention matrix: 128 x 128 = 16K elements = 32 KB
  A100 L2 cache: 40 MB

  Standard attention HBM traffic: 4*128*128 + 4*128^2 = 65K + 65K = 131K elements
  FlashAttention HBM traffic: roughly similar (tiling overhead for tiny matrices)

  Speedup from FlashAttention: 1.0-1.1x (negligible)
```

For sequences shorter than roughly 256 tokens, the overhead of FlashAttention's bookkeeping (online softmax state, tile management) can offset the IO savings. Standard fused attention kernels (like cuDNN's default) may match or exceed FlashAttention performance at these lengths.

### Case 2: Decode with KV Cache (N_q = 1)

During autoregressive decode, the query has length 1 while the KV cache has length N. The attention matrix is [1, N], which is a vector, not a matrix:

```
Decode attention: Q=[1, d], K=[N, d], V=[N, d]
  Attention scores: [1, N]  (a single row)
  Standard attention HBM: 2*d + 2*N*d + N = 2d + 2Nd + N
  No N^2 term! The attention 'matrix' is just a vector.

  FlashAttention advantage: minimal
  The bottleneck is reading the KV cache (2*N*d), not the attention scores.
```

For decode, the correct optimization is not FlashAttention but rather techniques that reduce KV cache reads: GQA (fewer KV heads), KV cache quantization (INT8/INT4 KV), or paged attention (efficient memory management). FlashAttention's tiling provides marginal benefit because there is no N^2 intermediate to eliminate.

This is why vLLM and SGLang use FlashAttention for prefill but use specialized decode kernels (like FlashDecoding or PagedAttention kernels) for token generation.

### Case 3: GQA Interaction

GQA reduces the number of KV heads, which changes the arithmetic intensity of attention:

```
MHA (32 Q heads, 32 KV heads):
  Each head: independent Q, K, V of size [N, d]
  Total KV reads per layer: 32 x 2 x N x d = 64Nd

GQA-4 (32 Q heads, 8 KV heads):
  Each KV head shared by 4 Q heads
  Total KV reads per layer: 8 x 2 x N x d = 16Nd
  But Q reads remain: 32 x N x d = 32Nd

GQA makes KV reads cheaper -> attention may become even more compute-bound
-> FlashAttention's benefit is LARGER with GQA (more compute per byte)
```

With GQA, each KV block is reused by multiple query heads. FlashAttention can exploit this by loading a KV tile once and processing multiple Q heads against it, further amortizing memory access. FlashAttention-2 and FlashAttention-3 include explicit GQA-aware tiling strategies.

### Case 4: Very Long Sequences (N > 32K)

For very long sequences, FlashAttention's benefit grows superlinearly because the N^2 standard attention traffic becomes enormous:

```
N=131072 (128K context), d=128, 32 heads:

Standard attention HBM traffic per head:
  4N^2 x 2 bytes = 4 x 131072^2 x 2 = 137 GB per head
  32 heads: 4.4 TB of HBM traffic (!!)
  This exceeds A100 HBM capacity. Standard attention physically cannot run.

FlashAttention HBM traffic per head:
  O(N^2 * d^2 / M) = 131072^2 * 128^2 / 96K
  = 4.6 GB per head
  32 heads: 147 GB

FlashAttention makes 128K-token attention POSSIBLE, not just faster.
Without it, the N^2 intermediate exceeds physical memory.
```

This is why every model supporting 128K+ context (Llama 3.1, Claude, GPT-4) requires FlashAttention or an equivalent tiled algorithm. It is not a performance optimization for long contexts; it is a correctness requirement.

### Summary Table

```
Scenario                     FlashAttention Benefit
-------------------------------------------------------
Short seq (N < 256)          Negligible (1.0-1.1x)
Medium seq (256-2048)        Significant (1.5-2.5x)
Long seq (2048-8192)         Large (2-4x)
Very long seq (8K-128K)      Essential (enables the computation)
Decode (N_q = 1)             Minimal (use FlashDecoding instead)
Prefill (N_q = N)            Full benefit
MHA                          Good benefit
GQA                          Even better (more compute per KV byte)
```

---

## Practical Usage in Inference Engines

Modern inference engines integrate FlashAttention deeply into their serving stack. Understanding how each engine uses it informs deployment decisions.

### vLLM

vLLM uses FlashAttention for prefill and a specialized paged-attention kernel for decode:

```
vLLM attention dispatch:

  if is_prefill:
      # FlashAttention-2 kernel (flash_attn_varlen_func)
      # Handles variable-length sequences in a batch efficiently
      # Uses cu_seqlens to pack multiple prompts without padding
      output = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k)
  else:
      # PagedAttention kernel (custom CUDA)
      # Reads KV from paged block tables
      # Optimized for N_q=1, large N_kv pattern
      output = paged_attention_v2(q, key_cache, value_cache, block_tables)
```

The `flash_attn_varlen_func` variant is critical for batched prefill: it packs multiple sequences of different lengths into a single kernel call using cumulative sequence length offsets (cu_seqlens), avoiding the waste of padding shorter sequences to the maximum length.

### SGLang

SGLang uses FlashInfer as its attention backend, which provides FlashAttention-style kernels with additional optimizations for radix-tree-based KV cache sharing:

```
SGLang attention features:

  1. Prefix-aware attention: When multiple requests share a common prefix,
     the KV cache for that prefix is computed once and shared via radix tree.
     FlashAttention processes only the unique suffix per request.

  2. Chunked prefill: Long prompts are split into chunks (e.g., 8192 tokens)
     and processed incrementally. Each chunk uses FlashAttention, appending
     its KV to the cache before the next chunk begins.

  3. Cascade attention: For very long KV caches (100K+ tokens),
     FlashInfer splits attention into a 'persistent' part (the cached prefix)
     and a 'local' part (the new tokens), computing them separately and merging.
```

### Integration Considerations

```
Engine        Prefill Kernel         Decode Kernel           GQA Support
--------------------------------------------------------------------------
vLLM          flash_attn_varlen      paged_attention_v2      Native
SGLang        FlashInfer             FlashInfer decode       Native
TensorRT-LLM  flash_attention        gpt_attention           Native
HuggingFace   flash_attn_2 or SDPA   flash_attn_2 or SDPA    Via config
```

All major engines now default to FlashAttention for prefill. The differentiation is in decode kernels and memory management, not the attention algorithm itself.

---

## The Backward Pass and Training

While this module focuses on inference, understanding FlashAttention's backward pass explains a design choice that affects inference: the decision not to store the attention matrix.

In standard attention training, the forward pass stores P (the softmax output) for use in the backward pass. This requires O(N^2) memory per head per layer. For a 70B model with 8K context during training, this intermediate storage alone would consume hundreds of gigabytes.

FlashAttention's forward pass discards P after use. During the backward pass, it recomputes P from Q, K, and the stored softmax statistics (m and l). This recomputation costs extra FLOPs but saves O(N^2) memory:

```
Memory savings from not storing P:

  Standard attention training memory for P:
    batch x heads x N^2 x sizeof(float)
    = 4 x 64 x 8192^2 x 2 = 34.4 GB

  FlashAttention stores only m, l per row:
    batch x heads x N x 2 x sizeof(float)
    = 4 x 64 x 8192 x 2 x 4 = 16.8 MB

  Reduction: 2048x less memory
```

For inference, the practical implication is that FlashAttention kernels only output O (the attention output) and optionally the log-sum-exp statistics (for debugging or gradient checkpointing). They never output the attention weights P, which means visualization tools that display attention patterns must use a separate, slower kernel.
---

## FlashDecoding: The Decode-Specific Variant

Standard FlashAttention parallelizes over Q blocks (rows). During decode, Q has only 1 row, so there is nothing to parallelize over in the outer loop. FlashDecoding (Dao et al., 2023) solves this by parallelizing over KV blocks instead:

```
FlashDecoding for decode (N_q=1, N_kv=large):

  Problem: Only 1 query row -> only 1 thread block in FA-2 outer loop
           -> 1 SM busy, 107 SMs idle on A100

  Solution: Split KV into S splits along sequence dimension
            Each split computes partial attention independently
            Final reduction combines partial results using online softmax merge

  Phase 1 (parallel): S thread blocks, each handles N_kv/S tokens of KV cache
    Block s: computes partial_o_s, partial_m_s, partial_l_s

  Phase 2 (reduction): Merge S partial results
    m_final = max(m_0, m_1, ..., m_(S-1))
    l_final = sum(exp(m_s - m_final) * l_s)
    o_final = sum(exp(m_s - m_final) * partial_o_s) / l_final
```

With S=32 splits and 32 heads, decode attention uses 32 x 32 = 1024 thread blocks, fully saturating the GPU. The reduction phase adds minimal overhead because it operates on [S, d] data (small).

```
Decode performance comparison (Llama 8B, A100, N_kv=4096):

  Standard attention decode:   12.3 us per layer
  PagedAttention v2:           8.1 us per layer
  FlashDecoding:               6.7 us per layer

  Improvement: 1.2x over PagedAttention, 1.8x over standard
```

---

## Common Misconceptions

### Misconception 1: FlashAttention is an Approximation

FlashAttention computes exact attention. The output is numerically identical to standard attention up to floating-point reordering (which affects the last few bits of precision due to non-associativity of floating-point addition). There is no sparsity, no low-rank approximation, no token dropping. Every query attends to every key with exact softmax weights.

### Misconception 2: FlashAttention Reduces FLOPs

FlashAttention performs the same number of FLOPs as standard attention (actually slightly more due to the online softmax bookkeeping). Its speedup comes entirely from reduced HBM access. It does more total computation but finishes faster because the GPU is no longer bottlenecked by memory bandwidth.

```
FLOP comparison:
  Standard attention: 2*N^2*d (for Q@K^T) + 2*N^2*d (for P@V) = 4*N^2*d
  FlashAttention:     4*N^2*d + online softmax overhead (~5% extra)

  FA does 5% MORE compute but runs 2-4x FASTER
  (because it eliminates the memory bottleneck)
```

### Misconception 3: FlashAttention Helps All Operations Equally

FlashAttention only optimizes the attention computation itself (Q@K^T, softmax, P@V). It does not accelerate the linear projections (W_q, W_k, W_v, W_o) or the MLP, which together account for 60-70% of inference time. For a full forward pass:

```
Llama 8B forward pass time breakdown (A100, seq=2048):

  Attention projections (Wq, Wk, Wv, Wo): 25%
  Attention computation (Q@K, softmax, P@V): 15%  <- FlashAttention helps HERE
  MLP (gate, up, down projections): 55%
  Norms, residuals, other: 5%

  If FlashAttention gives 3x speedup on attention computation:
  Overall speedup = 1 / (0.85 + 0.15/3) = 1 / 0.90 = 1.11x
```

The Amdahl's Law effect is significant. FlashAttention dramatically speeds up the attention kernel, but attention is not the dominant cost for moderate sequence lengths. At very long sequences (32K+), attention becomes a larger fraction and FlashAttention's impact on end-to-end latency increases.

### Misconception 4: You Need to Implement FlashAttention Yourself

FlashAttention is available as a drop-in replacement in all major frameworks:

```python
# PyTorch (>= 2.0): Automatic via SDPA
from torch.nn.functional import scaled_dot_product_attention
output = scaled_dot_product_attention(q, k, v)  # uses FlashAttention when possible

# Explicit FlashAttention package
from flash_attn import flash_attn_func
output = flash_attn_func(q, k, v, causal=True)

# HuggingFace Transformers
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    attn_implementation="flash_attention_2"
)
```

The kernel selection happens automatically based on input shapes, data types, and GPU architecture. PyTorch's SDPA dispatcher checks whether FlashAttention-2 is available and falls back to memory-efficient attention or the math implementation if not.

---

## End-to-End Impact: Before and After FlashAttention

To ground the theory in concrete numbers, here is a before/after comparison for a realistic inference workload:

```
Workload: Llama 3.1 8B, A100 80GB, batch=8, prompt=2048, generate=512

PREFILL (batch=8, seq=2048):

  Without FlashAttention:
    Attention HBM traffic: 8 x 32 heads x (4*128*2048 + 4*2048^2) x 2B = 8.7 GB
    Attention time: 8.7 GB / 2 TB/s + compute = 5.8 ms
    (memory-bound, GPU utilization ~45%)

  With FlashAttention-2:
    Attention HBM traffic: 8 x 32 x ~5.5 MB = 1.4 GB
    Attention time: compute-bound = 274.9B * 8 / 312T = 2.1 ms
    (compute-bound, GPU utilization ~87%)

  Prefill attention speedup: 2.8x
  Prefill end-to-end speedup: 1.3x (attention is 30% of prefill)

DECODE (not affected by FlashAttention):
  Uses FlashDecoding/PagedAttention kernels
  No meaningful change in per-token latency

MEMORY SAVINGS:
  Without FlashAttention: must allocate N^2 attention matrix in HBM
    8 x 32 x 2048^2 x 2B = 2.1 GB of intermediate storage
  With FlashAttention: zero intermediate storage
    Freed 2.1 GB -> can serve more concurrent sequences or longer contexts
```

---

## Key Takeaways

1. Standard attention materializes an N^2 matrix in HBM, creating quadratic memory traffic that makes attention memory-bound despite being a pure matmul operation.

2. FlashAttention tiles the computation so the N^2 matrix lives only in SRAM, reducing HBM accesses from Theta(Nd + N^2) to O(N^2 d^2 / M), transforming attention from memory-bound to compute-bound.

3. The online softmax trick enables correct incremental computation without ever seeing the full row of attention scores, which is what makes tiling possible despite softmax's global dependency.

4. FlashAttention-2 improves practical performance by 1.2-1.3x through reducing non-matmul operations, adding sequence-dimension parallelism, and eliminating warp synchronization.

5. FlashAttention-3 exploits Hopper's TMA for fully async loads, WGMMA for larger tile instructions, and FP8 for 2x throughput, achieving 1.3x over FA-2 in FP16 and 2.1x in FP8.

6. FlashAttention helps most during prefill (large N_q). For decode (N_q=1), use FlashDecoding which parallelizes over KV splits instead of Q blocks.

7. For very long contexts (32K+), FlashAttention is not an optimization but a necessity: the N^2 intermediate would exceed physical GPU memory without tiling.

---

## References

1. Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022)
2. Dao. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023)
3. Shah et al. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" (2024)
4. Milakov and Gimelshein. "Online normalizer calculation for softmax" (2018)
5. Dao et al. "FlashDecoding" (2023)
6. Rabe and Staats. "Self-attention Does Not Need O(n^2) Memory" (2021)
7. NVIDIA. "H100 Tensor Core GPU Architecture Whitepaper" (2022)
8. Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023)

---

## What Comes Next

Module 1.4 covers the roofline model in detail: how to determine whether any given kernel is compute-bound or memory-bound, and how to use this analysis to predict whether an optimization (like FlashAttention) will help for a specific workload configuration. The roofline framework generalizes the analysis we performed here for attention to any GPU kernel.