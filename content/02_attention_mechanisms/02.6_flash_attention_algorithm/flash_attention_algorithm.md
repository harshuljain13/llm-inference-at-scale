# 2.6 FlashAttention: The Algorithm

---

## IO Complexity of FlashAttention

With the tiled algorithm, the HBM access pattern changes fundamentally:

```
Outer loop: ceil(N/B_r) iterations
  Read Q_i once: B_r x d elements

  Inner loop: ceil(N/B_c) iterations
    Read K_j: B_c x d elements
    Read V_j: B_c x d elements

  Write O_i once: B_r x d elements
```

Total HBM reads:

```
Q reads: Each Q block read once = N x d total
K reads: Each K block read ceil(N/B_r) times = N x d x ceil(N/B_r)
V reads: Each V block read ceil(N/B_r) times = N x d x ceil(N/B_r)
O writes: N x d total

Total = Nd + 2*Nd*(N/B_r) + Nd
      = 2Nd + 2N^2*d / B_r
      = O(N^2 * d / B_r)
```

Since B_r is chosen to maximize SRAM utilization, B_r = Theta(M/d) where M is SRAM size:

```
B_r = Theta(M / d)

HBM accesses = O(N^2 * d / (M/d)) = O(N^2 * d^2 / M)
```

This is the key result from Dao et al. (2022):

```
Standard attention HBM accesses: Theta(Nd + N^2)
FlashAttention HBM accesses:     Theta(N^2 * d^2 / M)
```

For typical values (d=128, M=192KB = 96K elements):

```
Ratio = (N^2 * d^2 / M) / (Nd + N^2)

For N=4096:
  Standard: 4096*128 + 4096^2 = 524K + 16.8M = 17.3M elements
  Flash:    4096^2 * 128^2 / 96K = 17.3M * 128/96K
          = 17.3M * 170.7 / (96K)  ... let's compute directly:
          = 4096^2 * 16384 / 98304 = 16.8M * 0.167 = 2.8M elements

Reduction factor: 17.3M / 2.8M = 6.2x fewer HBM accesses
```

For longer sequences, the reduction grows further because the N^2 term in standard attention grows faster than N^2*d^2/M (which also grows as N^2 but with a smaller constant when M >> d^2).

### Concrete Byte Comparison

For N=4096, d=128, 32 heads on A100:

```
Standard attention HBM traffic:
  Per head: (4*128*4096 + 4*4096^2) x 2 bytes = 138.4 MB
  32 heads: 4.4 GB

FlashAttention HBM traffic:
  Per head: ~22 MB (empirically measured)
  32 heads: ~710 MB

Reduction: 6.2x
```

With 6x fewer HBM accesses, FlashAttention shifts the arithmetic intensity above the ridge point:

```
FlashAttention arithmetic intensity:
  274.9B FLOPs / 710 MB = 387 FLOPs/byte

387 > 156 (A100 ridge point)

FlashAttention is COMPUTE-BOUND on A100!
```

This is the fundamental achievement: by eliminating quadratic HBM traffic, FlashAttention transforms attention from memory-bound to compute-bound, allowing the GPU to use its full compute throughput.
---

## FlashAttention-2: Reducing Non-Matmul FLOPs

FlashAttention-1 achieved the IO complexity breakthrough but left performance on the table. On A100, FlashAttention-1 reached about 72% of theoretical maximum FLOPS utilization. FlashAttention-2 (Dao, 2023) pushes this to approximately 73% by addressing three bottlenecks that have nothing to do with memory access.

### Problem 1: Non-Matmul FLOPs

GPU tensor cores are specialized for matrix multiplications. All other operations (exp, max, divide, compare) run on CUDA cores, which deliver 10-20x lower throughput. In FlashAttention-1, a significant fraction of time was spent on these non-matmul operations:

```
FlashAttention-1 FLOP breakdown (N=4096, d=128, one head):

  Matmul FLOPs (on tensor cores):
    Q @ K^T:         2 x B_r x B_c x d per tile = matmul-heavy
    P @ V:           2 x B_r x d x B_c per tile = matmul-heavy

  Non-matmul FLOPs (on CUDA cores):
    exp(S - m):      B_r x B_c per tile
    rowmax(S):       B_r x B_c per tile
    rowsum(exp):     B_r x B_c per tile
    rescale o_i:     B_r x d per tile
    divide o/l:      B_r x d (once at end)

Non-matmul fraction: approximately 25-30% of total execution time
```

### Solution: Defer Rescaling

FlashAttention-2 restructures the algorithm to minimize rescaling operations. Instead of normalizing by l_i at every step, it maintains unnormalized accumulators and applies the final division only once:

```
FlashAttention-1 inner loop (per tile):
  o_i = diag(exp(m_old - m_new))^(-1) @ o_i + P_ij @ V_j  # rescale EVERY tile

FlashAttention-2 inner loop (per tile):
  o_i = diag(exp(m_old - m_new)) @ o_i + P_ij @ V_j       # rescale without divide
  # Final: o_i = o_i / l_i  (ONCE, after all tiles)
```

By deferring the division, FlashAttention-2 eliminates one elementwise division per inner-loop iteration (B_r x d elements per tile, across ceil(N/B_c) tiles). For N=4096 and B_c=128, that removes 32 unnecessary division passes.

### Problem 2: Parallelism Across Sequence Length

FlashAttention-1 parallelizes over batch size and number of heads. Each thread block handles one (batch, head) pair and iterates sequentially over the sequence dimension. On A100 with 108 SMs, this means:

```
FlashAttention-1 parallelism:
  Parallel units = batch_size x num_heads
  Example: batch=4, heads=32 -> 128 parallel units
  A100 SMs: 108
  Occupancy: 128/108 = 1.19 waves (good)

Problem case: batch=1, heads=32 -> 32 parallel units
  Occupancy: 32/108 = 0.30 waves (bad! 70% of SMs idle)
```

### Solution: Parallelize Over Sequence

FlashAttention-2 adds parallelism along the sequence dimension in the OUTER loop (over Q blocks):

```
FlashAttention-2 parallelism:
  Parallel units = batch_size x num_heads x ceil(N/B_r)
  Example: batch=1, heads=32, N=4096, B_r=128
  -> 32 x 32 = 1024 parallel units
  Occupancy: 1024/108 = 9.5 waves (excellent)
```

This requires each thread block to independently compute its Q block's output by iterating over all K, V blocks. Since there are no cross-Q-block dependencies in the forward pass (each output row depends only on its own query and all keys/values), this parallelization is trivially correct.

### Problem 3: Warp Partitioning

Within a thread block, FlashAttention-1 splits work across warps by partitioning the K/V blocks (each warp handles a slice of the inner loop). This requires a reduction across warps to combine partial results, adding synchronization overhead.

FlashAttention-2 instead partitions across the Q block rows: each warp handles a subset of Q rows and independently iterates over all K/V blocks. No inter-warp communication is needed because different Q rows are independent.

```
FlashAttention-1 warp partitioning (4 warps):
  Warp 0: handles K_j[0:B_c/4]   -> partial o_i
  Warp 1: handles K_j[B_c/4:B_c/2] -> partial o_i
  Warp 2: handles K_j[B_c/2:3B_c/4] -> partial o_i
  Warp 3: handles K_j[3B_c/4:B_c] -> partial o_i
  SYNC: reduce partial o_i across warps (shared memory barrier)

FlashAttention-2 warp partitioning (4 warps):
  Warp 0: handles Q_i[0:B_r/4]     -> full o for those rows
  Warp 1: handles Q_i[B_r/4:B_r/2] -> full o for those rows
  Warp 2: handles Q_i[B_r/2:3B_r/4] -> full o for those rows
  Warp 3: handles Q_i[3B_r/4:B_r]  -> full o for those rows
  NO SYNC needed (rows are independent)
```

### FlashAttention-2 Performance Impact

```
A100 80GB, Llama-style attention (GQA, d=128):

  Sequence length   FA-1 (TFLOPS)   FA-2 (TFLOPS)   Speedup
  512               180              220              1.22x
  1024              200              240              1.20x
  2048              210              260              1.24x
  4096              215              270              1.26x
  8192              218              280              1.28x

Peak A100 FP16 TFLOPS: 312
FA-2 utilization at seq=8192: 280/312 = 89.7%
```

The improvements compound: fewer non-matmul FLOPs + better parallelism + no warp synchronization = approximately 1.2-1.3x wall-clock speedup over FlashAttention-1 across all sequence lengths.

---

## FlashAttention-3: Exploiting Hopper Architecture

FlashAttention-3 (Dao et al., 2024) targets the NVIDIA H100 (Hopper architecture), which introduces hardware features that FlashAttention-2 cannot exploit. The key insight: Hopper's new instructions allow overlapping memory transfers with computation at the hardware level, not just the software level.

### Hopper's New Hardware Features

```
Feature                  A100 (Ampere)          H100 (Hopper)
--------------------------------------------------------------------
Memory copy engine       Software-managed       TMA (Tensor Memory Accelerator)
Async capability         Limited                Full async TMA loads
FP8 tensor cores         No                     Yes (1.9 PFLOPS)
WGMMA instructions       No                     Yes (warp-group matmul)
Shared memory size       192 KB/SM              228 KB/SM
HBM bandwidth            2.0 TB/s               3.35 TB/s
FP16 TFLOPS              312                    989
```

### Asynchronous TMA Loads

On A100, loading K and V blocks from HBM to shared memory is a blocking operation: the SM stalls until data arrives. On H100, the TMA (Tensor Memory Accelerator) is a dedicated hardware unit that copies data asynchronously while the SM continues computing on previously loaded data.

```
FlashAttention-2 on A100 (simplified timeline for one Q block):

  Time: |--Load K0--|--Compute S0--|--Load V0--|--Compute P0@V0--|--Load K1--|...
                     ^^^^^^^^^^^^              ^^^^^^^^^^^^^^^^
                     SM active                 SM active
         ^^^^^^^^^^               ^^^^^^^^^^                    ^^^^^^^^^^
         SM stalls                SM stalls                     SM stalls

FlashAttention-3 on H100 (pipelined with TMA):

  TMA:  |--Load K0--|--Load V0--|--Load K1--|--Load V1--|--Load K2--|...
  SM:              |--Compute S0--|--P0@V0--|--Compute S1--|--P1@V1--|...
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                   SM never stalls (TMA runs ahead)
```

The TMA prefetches the next tile while the SM processes the current one. As long as compute takes longer than the memory transfer (which is true when we are compute-bound), the memory latency is completely hidden.

### FP8 Support

H100 tensor cores support FP8 (E4M3 and E5M2 formats) at 2x the throughput of FP16. FlashAttention-3 adds FP8 attention kernels:

```
FP8 FlashAttention-3 performance:
  FP16 peak on H100: 989 TFLOPS
  FP8 peak on H100:  1,979 TFLOPS (2x FP16)

  FA-3 FP16 achieved: ~740 TFLOPS (75% utilization)
  FA-3 FP8 achieved:  ~1,200 TFLOPS (61% utilization of FP8 peak)

  FP8 vs FP16 speedup: 1.62x
```

FP8 attention is particularly useful during prefill where the large matmuls can tolerate reduced precision. For decode with KV cache, the precision impact on generation quality requires careful evaluation per model.

### WGMMA Instructions

Hopper introduces warp-group matrix-multiply-accumulate (WGMMA) instructions that operate on larger tile sizes than Ampere's WMMA/MMA instructions. FlashAttention-3 uses WGMMA to process larger tiles per instruction, reducing instruction overhead and improving throughput.

```
Tile sizes per instruction:
  A100 (mma.sync): 16x8x16  (2048 FLOPs per instruction)
  H100 (wgmma):    64x256x16 (524,288 FLOPs per instruction)

256x more compute per instruction = fewer instructions = less overhead
```

### FlashAttention-3 Performance Summary

```
H100 SXM5, d=128, causal attention:

                   FA-2 on H100   FA-3 FP16    FA-3 FP8
  Seq 2048         520 TFLOPS     690 TFLOPS   1100 TFLOPS
  Seq 4096         560 TFLOPS     720 TFLOPS   1150 TFLOPS
  Seq 8192         580 TFLOPS     740 TFLOPS   1200 TFLOPS
  Seq 16384        590 TFLOPS     750 TFLOPS   1220 TFLOPS

FA-3 FP16 vs FA-2: 1.28-1.33x speedup on H100
FA-3 FP8 vs FA-3 FP16: 1.59-1.63x additional speedup
```
---

