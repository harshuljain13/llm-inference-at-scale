# 1.5 Decode and Batching on the Roofline

The previous module proved that LLM prefill is compute-bound. This module proves decode is memory-bound and shows exactly how batching changes the picture.

---

## Where LLM Decode Lands: Memory-Bound Derivation

During decode, the model generates one token at a time. Each forward pass processes a single new token (or a single token per sequence in a batch). The input to each linear layer is a vector of shape [1, hidden_dim] (batch=1) or [B, hidden_dim] (batch=B).

### Single-Token Decode (Batch = 1)

For one decode step through the full Llama 3.1 8B model:

```
FLOPs per decode step:
  = 2 * num_params (each parameter participates in one multiply-add)
  = 2 * 7.78 * 10^9
  = 15.56 * 10^9 FLOPs

Bytes transferred per decode step:
  Model weights: 7.78 * 10^9 * 2 = 15.56 GB (must read entire model)
  KV cache read: 2 * 32 * 8 * 128 * context_len * 2
    At context_len = 2048: = 2 * 32 * 8 * 128 * 2048 * 2 = 268 MB
  Input/output vectors: negligible (< 1 MB)
  Total bytes: ~15.83 GB

Arithmetic intensity:
  AI = 15.56 * 10^9 / 15.83 * 10^9 = 0.98 FLOPs/byte
```

The arithmetic intensity of single-token decode is approximately 1.0 FLOPs/byte. Compare this to the ridge points:

```
A100 ridge point: 153 FLOPs/byte
H100 ridge point: 295 FLOPs/byte

Decode AI / A100 ridge = 0.98 / 153 = 0.0064 = 0.64%
Decode AI / H100 ridge = 0.98 / 295 = 0.0033 = 0.33%
```

Single-token decode operates at less than 1% of the ridge point. It is not slightly memory-bound; it is catastrophically memory-bound. The GPU's compute units are idle 99% of the time, waiting for model weights to arrive from HBM.

### Maximum Achievable Decode Throughput

Since decode is memory-bound, throughput is determined entirely by how fast the hardware can stream model weights:

```
Max tokens/second = Memory Bandwidth / Model Size (bytes)

A100: 2,039 GB/s / 15.56 GB = 131 tokens/second
H100: 3,352 GB/s / 15.56 GB = 215 tokens/second
B200: 8,000 GB/s / 15.56 GB = 514 tokens/second
```

These are theoretical upper bounds. Actual throughput is 70-85% of these values due to memory controller overhead and the KV cache reads that compete for bandwidth. Measured single-token decode speeds for Llama 8B FP16 are typically:

| GPU | Theoretical Max | Measured (typical) | Efficiency |
|-----|----------------:|-------------------:|-----------:|
| A100 | 131 tok/s | 95-110 tok/s | 73-84% |
| H100 | 215 tok/s | 160-185 tok/s | 74-86% |
| B200 | 514 tok/s | 380-440 tok/s | 74-86% |

The bandwidth ratio between GPUs predicts the decode speed ratio accurately. H100 is 1.64x faster than A100 for decode (matching the 3,352/2,039 = 1.64x bandwidth ratio), not 3.2x faster (which is the compute ratio). This confirms that decode speed is governed by bandwidth, not compute.


---

## How Batching Moves Decode Up the Roofline

Batching is the primary mechanism for improving decode efficiency. When B sequences are decoded simultaneously, the model weights are read once from HBM and reused B times. This multiplies the FLOPs without proportionally increasing the bytes transferred.

### The Arithmetic of Batched Decode

For batch size B through the full model:

```
FLOPs = 2 * num_params * B = 2 * 7.78 * 10^9 * B
Bytes = model_weights + KV_cache_reads + activations
      = 15.56 GB + B * kv_per_sequence + negligible

Where kv_per_sequence (Llama 8B, context 2048, FP16):
  = 2 * 32 * 8 * 128 * 2048 * 2 = 268 MB per sequence

Total bytes for batch B:
  = 15.56 * 10^9 + B * 268 * 10^6
```

Computing AI for various batch sizes:

```
B=1:   FLOPs = 15.56G,  Bytes = 15.56G + 0.27G = 15.83G,  AI = 0.98
B=4:   FLOPs = 62.2G,   Bytes = 15.56G + 1.07G = 16.63G,  AI = 3.74
B=16:  FLOPs = 248.9G,  Bytes = 15.56G + 4.29G = 19.85G,  AI = 12.5
B=32:  FLOPs = 497.8G,  Bytes = 15.56G + 8.58G = 24.14G,  AI = 20.6
B=64:  FLOPs = 995.5G,  Bytes = 15.56G + 17.2G = 32.72G,  AI = 30.4
B=128: FLOPs = 1991G,   Bytes = 15.56G + 34.3G = 49.90G,  AI = 39.9
B=256: FLOPs = 3982G,   Bytes = 15.56G + 68.6G = 84.20G,  AI = 47.3
```

Notice that AI does not scale linearly with B as it would for pure matrix-vector multiplication. The KV cache reads grow with B, adding to the denominator. Each additional sequence in the batch brings its own KV cache that must be read from HBM for the attention computation. This creates diminishing returns: the first few sequences in a batch provide large AI gains, while later sequences provide smaller gains.

### Visualizing Batched Decode on the Roofline

```
Achievable
TFLOPS
     |
 312 |_________________________________________________  A100 Compute Ceiling
     |                                  /
     |                                /
     |                              /
     |                            /
     |                          /    Ridge = 153
     |                        /
     |                      /
     |                    /          * B=256 (AI=47)
     |                  /        * B=128 (AI=40)
     |                /      * B=64 (AI=30)
     |              /    * B=32 (AI=21)
     |            /  * B=16 (AI=13)
     |          / * B=4 (AI=4)
     |        /
     |      /
     |    / * B=1 (AI=1)
     |  /
     |/
     +------------------------------------------------  AI (FLOPs/byte)
     0.1   1     10      100     153    1000
```

Even at B=256, the arithmetic intensity (47) remains well below the A100 ridge point (153). Decode stays memory-bound across all practical batch sizes because KV cache memory consumption prevents reaching the batch sizes that would push AI past the ridge point.

---

## Batch Size Needed to Hit the Ridge Point

We can solve for the batch size B that would push arithmetic intensity to the ridge point, assuming infinite VRAM:

```
Target: AI = Ridge Point

AI = (2 * params * B) / (model_bytes + B * kv_bytes_per_seq)

Setting AI = Ridge and solving for B:

Ridge * (model_bytes + B * kv_per_seq) = 2 * params * B
Ridge * model_bytes + Ridge * B * kv_per_seq = 2 * params * B
Ridge * model_bytes = B * (2 * params - Ridge * kv_per_seq)
B = (Ridge * model_bytes) / (2 * params - Ridge * kv_per_seq)
```

### A100 with Llama 8B (context = 2048)

```
Ridge = 153 FLOPs/byte
model_bytes = 15.56 * 10^9
params = 7.78 * 10^9
kv_per_seq = 268 * 10^6 bytes

Numerator = 153 * 15.56 * 10^9 = 2,381 * 10^9
Denominator = 2 * 7.78 * 10^9 - 153 * 268 * 10^6
            = 15.56 * 10^9 - 41.0 * 10^9
            = -25.4 * 10^9
```

The denominator is negative. This means there is no finite batch size that reaches the ridge point. The KV cache bandwidth cost per sequence (153 * 268 MB = 41 GB equivalent in compute terms) exceeds the FLOPs gained per sequence (2 * 7.78G = 15.56 GFLOPS). Adding more sequences to the batch increases both numerator and denominator of the AI fraction, but the denominator grows faster once KV cache reads dominate.

### What This Means

For Llama 8B at context length 2048 on A100, it is mathematically impossible to reach the ridge point through batching alone. The KV cache reads consume enough bandwidth that each additional sequence does not provide sufficient FLOPs to compensate.

Let us check with shorter context (context = 128, kv_per_seq = 16.8 MB):

```
Denominator = 15.56 * 10^9 - 153 * 16.8 * 10^6
            = 15.56 * 10^9 - 2.57 * 10^9
            = 12.99 * 10^9

B = (153 * 15.56 * 10^9) / (12.99 * 10^9) = 183 sequences
```

With very short context (128 tokens), you would need batch size 183 to reach the ridge point on A100. This requires:

```
VRAM for KV cache = 183 * 16.8 MB = 3.1 GB
VRAM for model = 15.56 GB
Total = ~19 GB (fits easily in 80 GB)
```

Short-context workloads (classification, embedding, short-form generation) can theoretically reach the ridge point. Long-context generative workloads cannot.

### H100 with Llama 8B (context = 128)

```
Ridge = 295 FLOPs/byte
kv_per_seq = 16.8 MB

Denominator = 15.56 * 10^9 - 295 * 16.8 * 10^6
            = 15.56 * 10^9 - 4.96 * 10^9
            = 10.6 * 10^9

B = (295 * 15.56 * 10^9) / (10.6 * 10^9) = 433 sequences
```

H100 requires batch size 433 at context=128 to reach its ridge point. The higher ridge point (295 vs 153) means more batching is needed, which partly offsets the bandwidth gains of moving to H100 for decode workloads.

### The Constraint Visualized

```
VRAM Budget (A100 80GB)
     |
80GB |========================================
     | [Overhead + Activations: ~5 GB       ]
     |========================================
     |                                        |
     | [KV Cache: grows linearly with B]      |
     |                                        |  VRAM runs out here
     |________________________________________| (B ~ 88 at ctx=4096)
     |                                        |
     |                                        |
     | [Model Weights: 15.56 GB fixed]        |
     |________________________________________|
     |
  0  +-----|-----|-----|-----|-----|-----|----->
     0    20    40    60    80   100   153
                   Batch Size
                                         ^
                              Ridge point batch (unreachable)
```

The batch size at which VRAM fills up (approximately 88 for Llama 8B at 4K context on A100) is far below the batch size that would reach the ridge point. This structural constraint means LLM decode with long contexts is permanently locked in the memory-bound regime on current hardware.

---

