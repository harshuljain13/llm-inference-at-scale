# 1.2 The Roofline Model

The roofline model is a performance analysis framework that maps any workload onto a two-dimensional space defined by hardware limits. One axis represents computational throughput (FLOPS). The other represents memory bandwidth (bytes per second). Every workload lands somewhere in this space, and the model tells you which hardware ceiling constrains it. For LLM inference, the roofline model explains why decode is slow, why batching helps, and exactly how much batching you need before compute becomes the bottleneck.

This chapter derives the roofline from first principles, applies it to matrix operations common in transformers, calculates ridge points for modern GPUs, and proves through worked arithmetic that LLM decode lives deep in the memory-bound regime.

---

## What Arithmetic Intensity Means

Arithmetic intensity is the ratio of floating-point operations performed to the number of bytes transferred between compute units and memory. It carries units of FLOPs per byte.

```
Arithmetic Intensity (AI) = Total FLOPs / Total Bytes Transferred
```

This single number characterizes a workload independently of any specific hardware. A workload with AI = 10 performs 10 floating-point operations for every byte it reads from or writes to memory. A workload with AI = 0.5 transfers two bytes for every operation it performs.

The significance of arithmetic intensity is that it determines which hardware resource limits performance. Hardware offers two resources: compute (measured in FLOPS) and memory bandwidth (measured in bytes/second). If a workload has low arithmetic intensity, it finishes its computations before memory can deliver the next batch of data. The compute units sit idle waiting for bytes. Conversely, if a workload has high arithmetic intensity, memory delivers data faster than compute can process it. The memory bus sits idle waiting for results to be written back.

### Calculating Arithmetic Intensity for Matrix Multiplication

Matrix multiplication is the dominant operation in transformer inference. Consider multiplying matrix A of shape [M, K] by matrix B of shape [K, N] to produce C of shape [M, N].

FLOPs count: Each element of C requires K multiply-accumulate operations. There are M x N elements in C, giving 2 x M x K x N FLOPs (the factor of 2 accounts for the separate multiply and add).

Bytes transferred: We must read A (M x K elements) and B (K x N elements) from memory, and write C (M x N elements) back. In FP16, each element is 2 bytes.

```
FLOPs = 2 * M * K * N
Bytes = 2 * (M*K + K*N + M*N)   [in FP16]
AI    = (2 * M * K * N) / (2 * (M*K + K*N + M*N))
      = (M * K * N) / (M*K + K*N + M*N)
```

For a square matrix where M = K = N = d:

```
AI = d^3 / (3 * d^2) = d / 3
```

With d = 4096 (typical hidden dimension): AI = 4096 / 3 = 1365 FLOPs/byte. This is extremely compute-bound. Large square matrix multiplications always saturate compute because the cubic growth in operations outpaces the quadratic growth in data movement.

### The Critical Case: Matrix-Vector Multiplication

During LLM decode with batch size 1, the input is a single token embedding, a vector of shape [1, K]. Multiplying by a weight matrix of shape [K, N]:

```
FLOPs = 2 * 1 * K * N = 2*K*N
Bytes = 2 * (1*K + K*N + 1*N) = 2*(K + K*N + N)

For K = N = 4096:
  FLOPs = 2 * 4096 * 4096 = 33,554,432
  Bytes = 2 * (4096 + 16,777,216 + 4096) = 33,570,816
  AI    = 33,554,432 / 33,570,816 = 1.0 FLOPs/byte
```

The weight matrix dominates the byte count. Reading K*N weights dwarfs the input vector and output vector. The result: matrix-vector multiplication has arithmetic intensity of approximately 1.0 regardless of the dimension size. This is the fundamental reason LLM decode is memory-bound.

### How Batch Size Changes Arithmetic Intensity

With batch size B, the input becomes [B, K] instead of [1, K]. The weight matrix is read once and reused across all B input vectors:

```
FLOPs = 2 * B * K * N
Bytes = 2 * (B*K + K*N + B*N)

For K = N = 4096, varying B:
  B=1:   AI = 2*1*4096^2 / 2*(1*4096 + 4096^2 + 1*4096)     = 1.0
  B=8:   AI = 2*8*4096^2 / 2*(8*4096 + 4096^2 + 8*4096)     = 7.9
  B=32:  AI = 2*32*4096^2 / 2*(32*4096 + 4096^2 + 32*4096)   = 30.5
  B=128: AI = 2*128*4096^2 / 2*(128*4096 + 4096^2 + 128*4096) = 107.8
  B=256: AI = 2*256*4096^2 / 2*(256*4096 + 4096^2 + 256*4096) = 186.2
```

The weight matrix K*N is read once regardless of batch size. The FLOPs scale linearly with B. Therefore arithmetic intensity scales approximately linearly with batch size until the input and output terms become significant relative to the weight term. This is the mechanism by which batching converts a memory-bound workload into a compute-bound one.

---

## The Two Ceilings

Every hardware platform imposes two performance limits. These form the "roof" in the roofline model.

### The Compute Ceiling

The compute ceiling is the maximum number of floating-point operations per second the hardware can execute. For GPUs, this is specified as peak TFLOPS for a given precision. The compute ceiling is a horizontal line on the roofline plot because it does not depend on arithmetic intensity. No matter how much data reuse a workload achieves, it cannot exceed the hardware's raw compute rate.

| GPU | FP16 Tensor Core Peak | FP8 Tensor Core Peak | TF32 Peak |
|-----|----------------------|---------------------|-----------|
| A100 SXM 80GB | 312 TFLOPS | N/A | 156 TFLOPS |
| H100 SXM 80GB | 990 TFLOPS | 1,979 TFLOPS | 495 TFLOPS |
| B200 SXM | 2,250 TFLOPS | 4,500 TFLOPS | 1,125 TFLOPS |

These are theoretical peaks. Sustained compute throughput is typically 60-80% of peak due to pipeline stalls, warp scheduling overhead, and instruction mix.

### The Memory Bandwidth Ceiling

The memory bandwidth ceiling is the maximum rate at which the hardware can transfer data between compute units and memory. On the roofline plot, this appears as a diagonal line with slope equal to the bandwidth. The achievable FLOPS equals arithmetic intensity multiplied by bandwidth:

```
Achievable FLOPS = AI * Bandwidth
```

This relationship holds until the compute ceiling is reached. On a log-log plot, the bandwidth ceiling is a line with slope 1, rising from the origin until it intersects the compute ceiling.

| GPU | HBM Bandwidth | HBM Generation | Memory Capacity |
|-----|--------------|----------------|-----------------|
| A100 SXM 80GB | 2,039 GB/s | HBM2e | 80 GB |
| H100 SXM 80GB | 3,352 GB/s | HBM3 | 80 GB |
| B200 SXM | 8,000 GB/s | HBM3e | 192 GB |

Sustained bandwidth is typically 75-85% of peak due to memory controller overhead, refresh cycles, and access pattern inefficiencies.

### The Roofline Shape

Combining both ceilings on a log-log plot produces the characteristic roofline shape:

```
log(TFLOPS)
     |
     |_________________________________________________  Compute ceiling
     |                                  /
     |                                /
     |                              /
     |                            /
     |                          /    <-- Ridge point
     |                        /
     |                      /
     |                    /   Memory bandwidth ceiling
     |                  /     (slope = bandwidth)
     |                /
     |              /
     |            /
     |          /
     |        /
     |      /
     |    /
     |  /
     |/
     +------------------------------------------------  log(AI)
```

Below the ridge point, performance is limited by memory bandwidth. The hardware cannot deliver data fast enough to keep compute units busy. Above the ridge point, performance is limited by compute. Data arrives faster than the hardware can process it.


---

## Ridge Point Calculation for Modern GPUs

The ridge point is the arithmetic intensity at which the bandwidth ceiling meets the compute ceiling. Below this point, the workload is memory-bound. Above it, the workload is compute-bound. The formula is:

```
Ridge Point = Peak Compute (FLOPS) / Memory Bandwidth (bytes/s)
```

### A100 SXM 80GB

```
Peak FP16 Compute = 312 TFLOPS = 312 * 10^12 FLOPS
Memory Bandwidth  = 2,039 GB/s = 2,039 * 10^9 bytes/s

Ridge Point = (312 * 10^12) / (2,039 * 10^9)
            = 312,000 / 2,039
            = 153 FLOPs/byte
```

Any workload with arithmetic intensity below 153 FLOPs/byte on A100 is memory-bound. Any workload above 153 FLOPs/byte is compute-bound.

### H100 SXM 80GB

```
Peak FP16 Compute = 990 TFLOPS = 990 * 10^12 FLOPS
Memory Bandwidth  = 3,352 GB/s = 3,352 * 10^9 bytes/s

Ridge Point = (990 * 10^12) / (3,352 * 10^9)
            = 990,000 / 3,352
            = 295 FLOPs/byte
```

H100 has a higher ridge point than A100. This means a workload must achieve nearly twice the data reuse to become compute-bound on H100. The compute ceiling grew faster (3.2x) than the bandwidth ceiling (1.6x), pushing the ridge point higher. Memory-bound workloads benefit less from the A100-to-H100 upgrade than compute-bound workloads do.

### B200 SXM

```
Peak FP16 Compute = 2,250 TFLOPS = 2,250 * 10^12 FLOPS
Memory Bandwidth  = 8,000 GB/s = 8,000 * 10^9 bytes/s

Ridge Point = (2,250 * 10^12) / (8,000 * 10^9)
            = 2,250,000 / 8,000
            = 281 FLOPs/byte
```

B200 brings the ridge point back down slightly relative to H100 because bandwidth grew proportionally faster (2.4x bandwidth vs 2.3x compute relative to H100). This is deliberate: NVIDIA recognized that memory-bound LLM workloads dominate GPU data center revenue, and invested disproportionately in bandwidth for Blackwell.

### Ridge Point Summary Table

| GPU | Peak FP16 (TFLOPS) | Bandwidth (GB/s) | Ridge Point (FLOPs/byte) |
|-----|--------------------:|------------------:|-------------------------:|
| A100 SXM | 312 | 2,039 | 153 |
| H100 SXM | 990 | 3,352 | 295 |
| B200 SXM | 2,250 | 8,000 | 281 |

The key insight: ridge points for modern GPUs are in the range of 150 to 300 FLOPs/byte. Any workload below this threshold gains nothing from additional compute capacity. It needs more bandwidth.

---

## Where LLM Prefill Lands: Compute-Bound Derivation

During prefill, the model processes the entire input prompt in parallel. If the prompt has S tokens, the input to each linear layer is a matrix of shape [S, hidden_dim]. This means prefill performs matrix-matrix multiplication, not matrix-vector multiplication.

### Derivation for a Single Linear Layer

Consider one linear projection in the attention block: input [S, d] multiplied by weight [d, d], where d is the hidden dimension.

```
FLOPs = 2 * S * d * d = 2 * S * d^2
Bytes = 2 * (S*d + d*d + S*d) = 2 * (2*S*d + d^2)   [FP16]
AI    = (2 * S * d^2) / (2 * (2*S*d + d^2))
      = (S * d^2) / (2*S*d + d^2)
      = (S * d) / (2*S + d)
```

For Llama 8B with d = 4096:

```
S = 512:   AI = (512 * 4096) / (2*512 + 4096)  = 2,097,152 / 5,120  = 410
S = 1024:  AI = (1024 * 4096) / (2*1024 + 4096) = 4,194,304 / 6,144 = 683
S = 2048:  AI = (2048 * 4096) / (2*2048 + 4096) = 8,388,608 / 8,192 = 1024
S = 4096:  AI = (4096 * 4096) / (2*4096 + 4096) = 16,777,216 / 12,288 = 1365
```

Even at modest sequence lengths of 512 tokens, the arithmetic intensity (410) far exceeds the ridge point of any current GPU (153 for A100, 295 for H100). Prefill is firmly compute-bound.

### Why Prefill is Compute-Bound

The mathematical explanation: with S tokens processed simultaneously, the weight matrix [d, d] is read once but used S times (once per token). The amortization factor is S. Since typical sequence lengths range from hundreds to thousands of tokens, and the ridge point is only 150 to 300, prefill achieves 2x to 10x more data reuse than needed to saturate compute.

This has a practical consequence: prefill throughput scales linearly with compute capacity. Doubling TFLOPS (e.g., A100 to H100) roughly doubles prefill speed. Doubling bandwidth has minimal impact on prefill.

### Full Model Prefill Calculation

For Llama 3.1 8B processing a 2048-token prompt:

```
Parameters breakdown:
  Attention projections (Q, K, V, O): 4 * d * d = 4 * 4096^2 per layer
  MLP (gate, up, down): 3 * d * d_ff = 3 * 4096 * 14336 per layer
  Total per layer: 4 * 4096^2 + 3 * 4096 * 14336 = 67M + 176M = 243M params
  Total model: 32 layers * 243M = 7.78B params (close to stated 8B)

Total FLOPs for prefill (2048 tokens):
  = 2 * num_params * seq_len
  = 2 * 7.78 * 10^9 * 2048
  = 31.9 * 10^12 FLOPs = 31.9 TFLOPS

Total bytes read (model weights, FP16):
  = 7.78 * 10^9 * 2 = 15.56 GB

Arithmetic intensity:
  AI = 31.9 * 10^12 / 15.56 * 10^9 = 2050 FLOPs/byte
```

At AI = 2050, prefill is 13x above the A100 ridge point (153) and 7x above the H100 ridge point (295). This confirms prefill is deeply compute-bound on all current hardware.

### Time to Complete Prefill

```
On A100 (312 TFLOPS peak, ~250 TFLOPS sustained at 80% efficiency):
  Time = 31.9 TFLOPS / 250 TFLOPS = 0.128 seconds = 128 ms

On H100 (990 TFLOPS peak, ~790 TFLOPS sustained):
  Time = 31.9 TFLOPS / 790 TFLOPS = 0.040 seconds = 40 ms
```

H100 prefills 3.2x faster than A100, tracking the compute ratio closely. This would not be the case for memory-bound workloads.

---


## FAQ

**Q1: What does arithmetic intensity physically mean?**

Arithmetic intensity measures how much computational work a kernel extracts from each byte it moves between memory and compute units. A value of 10 FLOPs/byte means for every byte transferred across the memory bus, the hardware performs 10 floating-point operations on it. Low arithmetic intensity (below the ridge point) means the workload starves compute units because memory cannot deliver data fast enough. High arithmetic intensity means compute is the bottleneck and the memory bus has idle cycles. It is a property of the algorithm and data layout, not the hardware.

**Q2: Why are there two ceilings and not one?**

Hardware imposes two independent physical limits: the maximum rate of arithmetic operations (compute ceiling, in FLOPS) and the maximum rate of data transfer (bandwidth ceiling, in bytes/second). These correspond to different silicon resources: tensor cores versus memory controllers. A workload cannot exceed either limit, but only one is active at a time for a given arithmetic intensity. The roofline model makes both visible simultaneously so you can identify which resource is the actual bottleneck without profiling.

**Q3: Is the ridge point the same for all workloads?**

No. The ridge point is a hardware property, not a workload property. It equals peak compute divided by peak bandwidth for a specific GPU and precision mode. All workloads on the same GPU share the same ridge point, but each workload has its own arithmetic intensity that determines which side of the ridge it falls on. Changing precision (FP16 to FP8) changes the compute ceiling and therefore shifts the ridge point. A100 FP16 has ridge point 153; H100 FP16 has ridge point 295. The same matmul kernel is memory-bound on one and compute-bound on neither depending on its AI relative to each GPU's ridge.

**Q4: How do I measure arithmetic intensity for my workload?**

Count total FLOPs (multiply-adds count as 2 operations) and total bytes transferred between HBM and the streaming multiprocessors for one kernel invocation. For matrix operations, use the formulas: matmul [M,K] x [K,N] has FLOPs = 2MKN and bytes = element_size * (MK + KN + MN). For fused kernels, sum FLOPs across all operations but count bytes only for HBM transfers (register and shared memory traffic is internal). Tools like NVIDIA Nsight Compute report both metrics directly via the `sm__sass_thread_inst_executed_op_*` and `dram__bytes` counters.

**Q5: Why does batch size improve arithmetic intensity for decode?**

During decode, weight matrices dominate the byte count because they are read from HBM regardless of batch size. With batch size B, FLOPs scale as 2*B*K*N (linear in B) while bytes remain approximately 2*K*N (the weight matrix read, which dominates). The ratio FLOPs/bytes therefore scales approximately linearly with B. Physically, you are reusing the same weight data across B independent token computations in a single kernel launch, amortizing the memory transfer cost across more useful work.

**Q6: Can a workload be both memory-bound and compute-bound simultaneously?**

Not for a single kernel at a single moment. Each kernel invocation is limited by exactly one resource at a time. However, a full inference pass contains many kernels: attention scores may be compute-bound while the subsequent softmax normalization is memory-bound. End-to-end latency is the sum of individual kernel times, each constrained by its own bottleneck. This is why optimizing inference requires kernel-level analysis, not just model-level arithmetic intensity averaging.

**Q7: How does quantization shift a workload on the roofline?**

Quantization (e.g., FP16 to INT8 or INT4) reduces the bytes transferred per parameter by 2x or 4x. Since arithmetic intensity = FLOPs / bytes, halving bytes doubles AI, pushing the workload rightward on the roofline plot toward the compute-bound regime. Simultaneously, lower-precision tensor cores often have higher peak FLOPS (H100 FP8 is 2x FP16), which raises the compute ceiling and shifts the ridge point. The net effect: quantized models move rightward faster than the ridge shifts, making them more likely to become compute-bound at smaller batch sizes.

---

## References

1. Williams, S., Waterman, A., & Patterson, D. (2009). "Roofline: An Insightful Visual Performance Model for Multicore Architectures." Communications of the ACM, 52(4), 65-76.
2. NVIDIA. (2023). "NVIDIA H100 Tensor Core GPU Architecture Whitepaper." NVIDIA Corporation.
3. NVIDIA. (2024). "NVIDIA Blackwell Architecture Technical Brief." NVIDIA Corporation.
4. Ofenbeck, G., Steinmann, R., Caparros, V., Leber, D., & Puschel, M. (2014). "Applying the Roofline Model." IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS).
5. Kim, S., et al. (2023). "Full Stack Optimization of Transformer Inference: a Survey." arXiv:2302.14017.
6. Pope, R., et al. (2022). "Efficiently Scaling Transformer Inference." MLSys 2023.
7. NVIDIA. (2024). "Nsight Compute Documentation: Roofline Analysis." NVIDIA Developer.
