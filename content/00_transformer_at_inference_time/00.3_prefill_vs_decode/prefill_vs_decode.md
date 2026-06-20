# 0.3 Prefill vs Decode

---

## Prefill vs Decode: Two Completely Different Problems

This is the most important section for inference engineering. Prefill and decode have different bottlenecks, different optimization strategies, and increasingly, different hardware. Understanding this split is the key to making correct capacity planning decisions.

### Prefill: Compute-Bound (Usually)

During prefill, you process the entire prompt in one forward pass. Because you are computing attention over many tokens simultaneously, the arithmetic intensity is high enough to saturate the GPU's compute units.

```python
# Prefill for 1000-token prompt:
Input: [1, 1000, 4096]

# Attention computation:
Q @ K^T: [1, 32, 1000, 128] @ [1, 32, 128, 1000] -> [1, 32, 1000, 1000]
         = 32 x 1000 x 128 x 1000 = 4.1 billion FLOPs

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

362 > 156 -> Prefill is compute-bound on A100
```

Prefill arithmetic intensity scales with sequence length. Longer prompts produce more FLOPs for the same weight reads, pushing the operation further into compute-bound territory. A 100-token prompt might be memory-bound; a 10,000-token prompt is definitely compute-bound.

### Decode: Memory-Bound (Always)

During decode, you generate one token at a time. The arithmetic intensity drops by three orders of magnitude compared to prefill because you are reading the entire model's weights to produce a single output token.

```python
# Decode for 1 new token (with 1000 tokens already in cache):
Input: [1, 1, 4096]

# Attention computation:
Q @ K^T: [1, 32, 1, 128] @ [1, 32, 128, 1001] -> [1, 32, 1, 1001]
         = 32 x 1 x 128 x 1001 = 4.1 million FLOPs

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

0.36 << 156 -> Decode is extremely memory-bound
```

Decode arithmetic intensity is roughly 1000x lower than prefill. This is why decode is always memory-bound: you are reading the entire model to generate one token, and no amount of compute optimization can change that fundamental ratio.

### The Batching Insight

Batching multiple decode requests together is the primary mechanism for improving GPU utilization during decode. The key observation is that model weights are read once from HBM but used for all sequences in the batch.

```python
# Decode for 32 sequences simultaneously:
Input: [32, 1, 4096]

# FLOPs: 32 x 5.8 billion = 186 billion
# Bytes read: ~16 GB (weights, shared) + ~4.2 GB (32 x 130 MB KV cache)
# Arithmetic intensity: 186B / 20.2G = 9.2 FLOPs/byte

# Still memory-bound, but 25x better than batch=1!
```

Batching amortizes weight reads across sequences. With batch=32, you read the weights once but do 32x the compute. This is why high-throughput serving systems use large batches.

However, there is a fundamental tension: KV cache scales with batch size. At some point, you run out of memory for KV cache before you can batch enough to become compute-bound.

```
Llama 3.1 8B on A100 80GB:
- Model weights: 16 GB
- Available for KV cache: ~60 GB
- KV cache per sequence (4096 tokens): 512 MB
- Max batch size: 60 GB / 512 MB = 117 sequences

At batch=117:
- Arithmetic intensity: 117 x 5.8B / (16G + 60G) = 8.9 FLOPs/byte
- Still memory-bound! But much better than batch=1.
```

You can never make decode compute-bound through batching alone. The KV cache grows with batch size, so you hit memory limits before reaching the ridge point. This is the fundamental constraint of LLM inference and the reason why so much research focuses on reducing KV cache size.

---

## The Memory Bandwidth Wall

The memory bandwidth wall represents the theoretical maximum decode speed for any given model on any given hardware. It is derived from first principles and cannot be exceeded by any software optimization.

```
Llama 3.1 8B on A100 80GB:
- Model weights: 16 GB (FP16)
- Memory bandwidth: 2 TB/s
- Time to read weights: 16 GB / 2 TB/s = 8 ms

Theoretical max decode speed = 1 token / 8 ms = 125 tokens/sec

This is a HARD CEILING. No optimization can exceed this.
```

The memory bandwidth wall is defined by `model_size / bandwidth`. This is the most important equation in LLM inference:

```
max_tokens_per_second = memory_bandwidth / model_size_bytes
```

Let us verify with real benchmarks:

| Model            | Size (FP16) | A100 BW | Theoretical Max | Actual (vLLM) | Efficiency |
| ---------------- | ----------- | ------- | --------------- | ------------- | ---------- |
| Llama 8B         | 16 GB       | 2 TB/s  | 125 tok/s       | 95-110 tok/s  | 76-88%     |
| Llama 70B        | 140 GB      | 2 TB/s  | 14.3 tok/s      | 11-13 tok/s   | 77-91%     |
| Llama 70B (TP=8) | 17.5 GB/GPU | 16 TB/s | 914 tok/s       | 650-750 tok/s | 71-82%     |

Real systems achieve 70-90% of theoretical bandwidth utilization. The gap comes from KV cache reads (not just weights), kernel launch overhead, memory access patterns that are not perfectly sequential, and synchronization in multi-GPU setups.

### Breaking the Wall: Your Options

Given that the bandwidth wall is a hard physical constraint, the only ways to increase decode speed are to reduce the bytes read or increase the available bandwidth:

1. **Quantization**: Reduce model size to read fewer bytes
   - INT8: 2x faster theoretical max
   - INT4: 4x faster theoretical max
   - Tradeoff: quality degradation that varies by model and task

2. **Tensor Parallelism**: More GPUs provide more aggregate bandwidth
   - TP=8 on A100: 16 TB/s aggregate bandwidth
   - Tradeoff: communication overhead, diminishing returns past 8 GPUs

3. **Speculative Decoding**: Generate multiple tokens per weight read
   - Draft model proposes N tokens, target verifies in one pass
   - Tradeoff: acceptance rate below 100%, draft model overhead

4. **Batching**: Amortize weight reads across sequences
   - Tradeoff: KV cache limits batch size

---

## Putting It All Together: A Complete Example

The following end-to-end trace demonstrates how all the concepts in this module combine in a real inference request. Pay attention to how prefill and decode have completely different performance characteristics despite running on the same hardware.

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
- Attention: 500^2 x 32 x 128 x 32 layers = 32.8B FLOPs
- MLP: 500 x 4096 x 14336 x 3 x 32 layers = 2.8T FLOPs
- Total: ~2.8T FLOPs

Time estimate:
- Memory time: 16 GB / 2 TB/s = 8 ms
- Compute time: 2.8T / 312T = 9 ms
- Prefill is roughly balanced, ~17 ms total

Output:
- KV cache populated: 500 tokens x 128 KB/token = 64 MB
- First token generated
```

**Decode Phase (200 tokens):**

```
Per token:
- Memory reads: 16 GB weights + growing KV cache
- Compute: ~5.8B FLOPs
- Time: ~8-10 ms per token (memory-bound)

Total decode time: 200 x 9 ms = 1.8 seconds

Final KV cache: 700 tokens x 128 KB/token = 90 MB
```

**Total Request:**

- TTFT (Time to First Token): ~17 ms
- TBT (Time Between Tokens): ~9 ms
- Total time: 17 ms + 200 x 9 ms = 1.82 seconds
- Throughput: 200 tokens / 1.82 s = 110 tokens/sec

---

## Key Takeaways

1. **LLM inference is a memory bandwidth problem, not a compute problem.** The GPU spends most of decode time waiting for memory transfers.

2. **The KV cache is the critical resource.** It determines your max batch size, max sequence length, and memory efficiency.

3. **GQA reduces KV cache 4-8x with minimal quality loss.** This is why every modern model uses it.

4. **Prefill is compute-bound, decode is memory-bound.** They need different optimizations (and increasingly, different hardware).

5. **The memory bandwidth wall is `model_size / bandwidth`.** This is the theoretical maximum decode speed.

6. **Batching helps but does not solve the problem.** KV cache grows with batch size, limiting how much you can batch.

7. **The formulas are approximations.** Real systems have overhead from PagedAttention blocks, kernel launches, and synchronization.

---

## What's Next

In Module 02.2, we dive into the attention variants (MHA, MQA, GQA) in much greater depth, examining the training dynamics and architectural decisions that led to GQA becoming the universal standard.

In Lab 2.1, you will implement a minimal transformer with KV cache and measure the memory/compute tradeoffs yourself.

---

## References

1. Vaswani et al. "Attention Is All You Need" (2017) - Original transformer
2. Shazeer "Fast Transformer Decoding: One Write-Head is All You Need" (2019) - MQA
3. Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models" (2023) - GQA
4. Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023) - vLLM
5. Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022)
6. Llama 3.1 Model Card - Meta AI
