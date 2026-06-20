# 2.3 Serving Implications

The roofline derivations from Modules 1.4 and 1.5 translate directly into serving system design decisions.

---

## Practical Implications for Serving

The roofline analysis yields several concrete guidelines for inference system design.

### Implication 1: Decode Throughput is a Bandwidth Problem

Since decode lives at AI < 1% of the ridge point, optimizations that increase bandwidth utilization dominate. Optimizations that increase compute utilization provide negligible benefit for decode.

Effective optimizations (attack bandwidth):
- Quantization reduces model size, directly increasing tokens/second
- Tensor parallelism across N GPUs provides N x bandwidth
- Speculative decoding amortizes weight reads across multiple candidate tokens
- Operator fusion reduces redundant HBM round-trips

Ineffective optimizations (attack compute):
- Higher-TFLOPS GPU (H100 vs A100) provides only bandwidth-ratio speedup for decode
- Mixed-precision compute (FP8 tensor cores) provides minimal decode benefit
- Larger tile sizes or better GEMM schedules have negligible impact at AI = 1

### Implication 2: Quantization Provides Linear Speedup

Reducing model precision from FP16 to INT8 halves the bytes read per decode step. Since decode throughput is proportional to bandwidth / model_bytes:

```
FP16: tokens/s = BW / (params * 2)
INT8: tokens/s = BW / (params * 1) = 2x speedup
INT4: tokens/s = BW / (params * 0.5) = 4x speedup
```

This is the theoretical upper bound. Measured speedups are:
- FP16 to INT8: 1.7x to 1.9x (overhead from dequantization)
- FP16 to INT4: 2.5x to 3.5x (more dequantization overhead)

Quantization is the single most impactful optimization for decode throughput because it directly reduces the dominant cost (weight reads from HBM).

### Implication 3: Prefill and Decode Need Different Optimization Strategies

The roofline positions of prefill (AI > 400) and decode (AI < 50) are so different that a single hardware configuration cannot serve both optimally:

```
Prefill wants:               Decode wants:
- Maximum TFLOPS             - Maximum bandwidth
- Large batch of tokens      - Low latency per token
- Compute-dense hardware     - Bandwidth-dense hardware
- Benefits from H100 3.2x   - Benefits from H100 1.6x
```

This motivates disaggregated serving architectures (Splitwise, DistServe) that route prefill to compute-optimized nodes and decode to bandwidth-optimized nodes. A prefill node can be an H100 running at high utilization. A decode node benefits more from multiple smaller GPUs whose aggregate bandwidth exceeds a single large GPU.

### Implication 4: The Batch Size Sweet Spot

Too little batching wastes bandwidth (weights read for only one token). Too much batching wastes VRAM on KV cache and increases latency. The optimal batch size balances three constraints:

```
1. Latency constraint: Each token in the batch must complete within the
   latency SLA (e.g., 30ms inter-token latency)

2. VRAM constraint: Model + KV_cache(B) + overhead must fit in memory

3. Efficiency target: Higher B improves GPU utilization but with
   diminishing returns once KV cache dominates bandwidth
```

For a latency-sensitive serving system (TTFT < 500ms, ITL < 50ms), the practical sweet spot is typically:

| Model Size | Context Length | Optimal Batch Range | GPU |
|-----------|---------------|--------------------:|-----|
| 8B FP16 | 2K | 16-32 | A100 80GB |
| 8B INT4 | 4K | 32-64 | A100 80GB |
| 70B FP16 (TP=8) | 2K | 8-16 | 8x A100 80GB |
| 70B INT4 (TP=4) | 4K | 16-32 | 4x A100 80GB |

These ranges represent the point where adding more sequences to the batch provides less than 5% throughput improvement per sequence due to the KV cache overhead, while latency remains within acceptable bounds.

### Implication 5: GPU Utilization is a Misleading Metric

Standard GPU utilization (as reported by nvidia-smi) measures what fraction of time at least one kernel is running on the GPU. For memory-bound decode, a kernel is always running (reading weights and performing the matrix-vector multiply), so utilization reads 90-100%.

But the streaming multiprocessors (SMs) within those kernels are idle most of each cycle, waiting for data from HBM. The actual compute throughput might be:

```
Achieved TFLOPS during decode:
  = AI * Bandwidth_utilized
  = 1.0 * (0.80 * 2039 GB/s)
  = 1.63 TFLOPS

Fraction of peak compute used:
  = 1.63 / 312 = 0.52%
```

The GPU reports high "utilization" while achieving 0.5% of peak compute. The correct metric for memory-bound workloads is bandwidth utilization:

```
Bandwidth utilization = Measured_throughput * model_bytes / Bandwidth
                      = 100 tok/s * 15.56 GB / 2039 GB/s
                      = 76%
```

A decode system achieving 76% bandwidth utilization with 30% reported SM activity is performing well. Attempting to "improve GPU utilization" by other means is chasing the wrong metric.

### Implication 6: Hardware Generations Shift the Economics

Each GPU generation changes the ridge point, which affects the relative value of optimizations:

```
A100 -> H100: Ridge moves from 153 to 295
  - Memory-bound workloads get 1.64x speedup (bandwidth ratio)
  - Compute-bound workloads get 3.2x speedup (FLOPS ratio)
  - Decode remains firmly memory-bound

H100 -> B200: Ridge moves from 295 to 281
  - Memory-bound workloads get 2.4x speedup (bandwidth ratio)
  - Compute-bound workloads get 2.3x speedup (FLOPS ratio)
  - Blackwell prioritizes bandwidth; both regimes benefit equally
```

The B200 represents a strategic shift. NVIDIA recognized that the majority of inference revenue comes from memory-bound decode, and invested proportionally more in bandwidth (HBM3e at 8 TB/s) relative to compute. Future hardware trends will likely continue this rebalancing as LLM serving workloads dominate data center GPU demand.

---

## Roofline for the Attention Mechanism

The attention computation during decode deserves separate analysis because it reads from the KV cache rather than model weights.

### Attention FLOPs and Bytes

For a single attention head during decode (generating one new token, attending over context of length C):

```
Step 1: Compute attention scores
  q * K^T where q is [1, d_head] and K is [C, d_head]
  FLOPs = 2 * C * d_head
  Bytes read: K cache = C * d_head * 2 bytes (FP16)

Step 2: Apply softmax
  FLOPs = 5 * C (exp, sum, divide per element)
  Bytes: negligible (in registers)

Step 3: Multiply by V
  scores * V where scores is [1, C] and V is [C, d_head]
  FLOPs = 2 * C * d_head
  Bytes read: V cache = C * d_head * 2 bytes

Total per head:
  FLOPs = 4 * C * d_head + 5*C ~ 4 * C * d_head (for d_head >> 1)
  Bytes = 2 * C * d_head * 2 = 4 * C * d_head

AI for attention = (4 * C * d_head) / (4 * C * d_head) = 1.0 FLOPs/byte
```

Attention during decode also has AI of approximately 1.0, confirming it is equally memory-bound. The bottleneck is reading the KV cache from HBM, not computing the dot products.

### Why Longer Context Makes Attention Worse

As context length C grows, the KV cache grows proportionally. The FLOPs also grow proportionally (more tokens to attend over), maintaining AI = 1.0. But the absolute bandwidth consumed by attention increases:

```
Context = 2K:  KV read per head per layer = 2 * 2048 * 128 * 2 = 1 MB
Context = 32K: KV read per head per layer = 2 * 32768 * 128 * 2 = 16 MB
Context = 128K: KV read per head per layer = 2 * 131072 * 128 * 2 = 64 MB

Total KV reads for Llama 8B (32 layers, 8 KV heads):
  Context 2K:   32 * 8 * 1 MB = 256 MB per decode step
  Context 32K:  32 * 8 * 16 MB = 4 GB per decode step
  Context 128K: 32 * 8 * 64 MB = 16 GB per decode step
```

At 128K context, reading the KV cache alone consumes 16 GB of bandwidth per decode step, comparable to reading the model weights (15.56 GB). The total bandwidth demand roughly doubles, halving decode throughput. This is why long-context inference is significantly slower even on the same hardware: the KV cache bandwidth cost rivals the weight-read cost.

---

## Summary

The roofline model provides a quantitative framework for understanding LLM inference performance. The key results derived in this chapter:

1. Arithmetic intensity for matrix-vector multiplication (decode, batch=1) is approximately 1.0 FLOPs/byte, regardless of model dimension.

2. Ridge points for current GPUs range from 153 (A100) to 295 (H100) FLOPs/byte. Decode operates at less than 1% of the ridge point.

3. Prefill with sequences of 512+ tokens achieves AI > 400 FLOPs/byte, placing it firmly in the compute-bound regime.

4. Batching increases decode AI linearly at first, but KV cache bandwidth costs create diminishing returns. For long-context workloads, no practical batch size reaches the ridge point.

5. The batch size that fills VRAM is far below the batch size that would reach the ridge point, permanently locking long-context decode in the memory-bound regime.

6. Decode throughput equals memory bandwidth divided by model size. All optimizations that reduce bytes read (quantization, sparsity) provide proportional throughput gains. Optimizations that increase compute capacity provide no benefit.

7. Hardware generations that increase bandwidth proportionally more than compute (e.g., B200) provide more value for LLM serving than generations that emphasize compute growth.

---

## FAQ

**Q1: Should I optimize for latency or throughput?**

It depends on your serving objective. For interactive applications (chatbots, code completion), optimize for latency: minimize inter-token latency (ITL < 50ms) and time-to-first-token (TTFT < 500ms). For batch workloads (summarization pipelines, offline processing), optimize for throughput: maximize tokens per second per dollar by increasing batch size until VRAM is full. Most production systems need both, which is why continuous batching exists: it serves latency-sensitive requests immediately while filling idle cycles with batch work.

**Q2: When does tensor parallelism help vs. hurt?**

Tensor parallelism helps when decode throughput is bandwidth-limited and you need more aggregate HBM bandwidth. Splitting a model across N GPUs provides Nx bandwidth, giving near-linear decode speedup. It hurts when (1) the model already fits on one GPU with room for batching, making the NVLink communication overhead pure waste, (2) you are prefill-bound (compute-limited), where pipeline parallelism is more efficient, or (3) the per-GPU shard becomes so small that kernel launch overhead dominates. Rule of thumb: use TP when model_bytes > 0.6 * single_GPU_VRAM.

**Q3: How do I know if my serving setup is optimal?**

Measure bandwidth utilization, not GPU utilization. Compute: (measured_tokens_per_second * model_bytes_per_token) / peak_HBM_bandwidth. If this exceeds 70%, your decode path is well-optimized. If it is below 50%, investigate memory access inefficiencies (fragmented KV cache, unoptimized kernels, unnecessary copies). Also check: is your batch size in the sweet spot from Implication 4? Is quantization applied? Are prefill and decode contending for the same resources?

**Q4: Why does my 70B model not get 4x speedup from INT4 quantization?**

The theoretical 4x assumes zero overhead from dequantization and perfect bandwidth utilization. In practice, INT4 requires unpacking and converting weights back to FP16/BF16 for computation, adding extra instructions. Group quantization (e.g., GPTQ with group_size=128) adds per-group scale/zero-point reads. Measured INT4 speedups are typically 2.5x to 3.5x over FP16. The gap narrows further if your baseline FP16 implementation already had suboptimal bandwidth utilization, since quantization only helps with the bytes-read bottleneck.

**Q5: Does upgrading from A100 to H100 double my decode throughput?**

No. Decode is memory-bandwidth-bound, so the speedup equals the bandwidth ratio, not the FLOPS ratio. H100 provides 3.35 TB/s vs A100 at 2.04 TB/s, yielding approximately 1.64x decode speedup. The 3.2x FLOPS improvement benefits prefill (compute-bound) but is largely wasted on decode. If your workload is decode-heavy (long generation, short prompts), the H100 upgrade delivers only 60% more throughput for roughly 2x the cost. Quantization on A100 often provides better cost-efficiency than upgrading hardware.

**Q6: When should I use disaggregated prefill and decode (Splitwise/DistServe)?**

Disaggregated serving helps when prefill and decode have fundamentally different resource profiles and your traffic pattern creates contention. Specifically: (1) your prompts are long (>2K tokens), making prefill compute-heavy enough to starve decode of GPU cycles, (2) you have mixed traffic where some requests are prompt-heavy (RAG with large contexts) and others are generation-heavy (creative writing), (3) your latency SLAs are tight and prefill spikes cause decode latency violations. It adds complexity (routing, load balancing, network hops), so avoid it if your workload is homogeneous or latency requirements are relaxed.

## References

1. Williams, S., Waterman, A., Patterson, D. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." Communications of the ACM, 52(4), 2009.
2. NVIDIA. "NVIDIA A100 Tensor Core GPU Architecture." Whitepaper, 2020.
3. NVIDIA. "NVIDIA H100 Tensor Core GPU Architecture." Whitepaper, 2022.
4. NVIDIA. "NVIDIA Blackwell Architecture Technical Brief." 2024.
5. Ivanov, A., Dryden, N., Ben-Nun, T., Li, S., Hoefler, T. "Data Movement Is All You Need: A Case Study on Optimizing Transformers." MLSys, 2021.
6. Kim, S., Hooper, C., Wattanawong, T., Kang, M., Yan, R., Genc, H., Dinh, G., Huang, Q., Rawat, K., Shao, Y.S., Keutzer, K., Gholami, A. "Full Stack Optimization of Transformer Inference: A Survey." arXiv:2302.14017, 2023.
7. Aminabadi, R.Y., Rajbhandari, S., Awan, A.A., Li, C., Li, D., Zheng, E., Ruwase, O., Smith, S., Zhang, M., Rasley, J., He, Y. "DeepSpeed-Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale." SC, 2022.
8. Pope, R., Douglas, S., Chowdhery, A., Devlin, J., Bradbury, J., Heek, J., Xiao, K., Agrawal, S., Dean, J. "Efficiently Scaling Transformer Inference." MLSys, 2023.
