# 2.5 FlashAttention: Why Standard Attention is Slow

> The gap between GPU compute capability and memory bandwidth grows every hardware generation. FlashAttention exploits this gap by restructuring attention to minimize HBM accesses, turning a memory-bound operation into a compute-bound one.

---

## The Problem: Standard Attention is Memory-Wasteful

Standard scaled dot-product attention computes the following for each head:

```
Output = softmax(Q @ K^T / sqrt(d)) @ V
```

where Q, K, V have shape [N, d] (sequence length N, head dimension d). The intermediate attention matrix S = Q @ K^T has shape [N, N]. For a 4096-token sequence with 128-dimensional heads, this matrix contains 16.7 million elements per head.

The critical issue is not the FLOPs. The critical issue is where those 16.7 million elements live during computation.

### Standard Attention: Step by Step HBM Traffic

On a GPU, HBM (High Bandwidth Memory) is the main memory (80 GB on A100), and SRAM is the fast on-chip memory (20 MB on A100, split across streaming multiprocessors). Standard attention proceeds as follows:

```
Step 1: Compute S = Q @ K^T
  Read:  Q [N, d] from HBM      -> N x d x 2 bytes
  Read:  K [N, d] from HBM      -> N x d x 2 bytes
  Write: S [N, N] to HBM        -> N x N x 2 bytes

Step 2: Compute P = softmax(S)
  Read:  S [N, N] from HBM      -> N x N x 2 bytes
  Write: P [N, N] to HBM        -> N x N x 2 bytes

Step 3: Compute O = P @ V
  Read:  P [N, N] from HBM      -> N x N x 2 bytes
  Read:  V [N, d] from HBM      -> N x d x 2 bytes
  Write: O [N, d] to HBM        -> N x d x 2 bytes
```

Each step launches a separate CUDA kernel. Between kernels, all intermediate results must round-trip through HBM. The attention matrix S and softmax output P each have N^2 elements, and both are written then read back. This is pure waste: these intermediates are consumed immediately and never needed again.

### Counting HBM Bytes

Total HBM reads and writes for standard attention (one head, FP16):

```
Reads:  N*d + N*d + N^2 + N^2 + N*d = 3Nd + 2N^2  elements
Writes: N^2 + N^2 + N*d             = Nd + 2N^2    elements
Total:  4Nd + 4N^2 elements
Bytes:  (4Nd + 4N^2) x 2  (FP16)
```

For N=4096, d=128:

```
Linear terms:    4Nd = 4 x 4096 x 128 = 2,097,152 elements
Quadratic terms: 4N^2 = 4 x 4096^2 = 67,108,864 elements
Total elements:  69,206,016
Total bytes:     138.4 MB per head
For 32 heads:    4.4 GB of HBM traffic for ONE attention layer
```

The quadratic terms are 32x larger than the linear terms. The attention matrix dominates HBM traffic.

Compare to the actual useful compute:

```
FLOPs for Q @ K^T:  2 x N x N x d = 2 x 4096^2 x 128 = 4.29 billion
FLOPs for P @ V:    2 x N x N x d = 4.29 billion
Total:              8.59 billion FLOPs per head
For 32 heads:       274.9 billion FLOPs
```

Arithmetic intensity of standard attention:

```
274.9 billion FLOPs / 4.4 GB = 62.5 FLOPs/byte

A100 ridge point: 312 TFLOPS / 2 TB/s = 156 FLOPs/byte

62.5 < 156  =>  Standard attention is MEMORY-BOUND on A100
```

This is paradoxical. Attention is a sequence of matrix multiplications, which should be compute-bound. But the mandatory materialization of the N^2 intermediate matrix forces so much HBM traffic that the operation becomes memory-bound. The GPU's compute units sit idle while bytes shuttle between HBM and the chip.

### IO Complexity of Standard Attention

Formally, standard attention requires:

```
HBM accesses = Theta(Nd + N^2)
```

The Nd term accounts for reading Q, K, V and writing O (unavoidable, as these tensors live in HBM). The N^2 term accounts for materializing S and P. Since d is typically 64-128 and N ranges from hundreds to hundreds of thousands, the condition N > d always holds in practice, making the N^2 term dominant.

The question FlashAttention answers: can we compute exact attention with only O(Nd) HBM accesses, eliminating the quadratic term entirely?

---

## The FlashAttention Algorithm: Tiling + Online Softmax

FlashAttention achieves sub-quadratic HBM access by never materializing the full N x N attention matrix. Instead, it computes attention in tiles that fit in SRAM, using an online softmax algorithm to accumulate correct results without needing the full row of attention scores.

### Why Tiling Alone is Insufficient

A naive tiling approach would partition the computation into blocks but still require the full attention matrix. The obstacle is softmax: computing softmax(row) requires knowing the maximum value in the entire row (for numerical stability) and the sum of exponentials across the entire row. Both require a full pass over all N attention scores for each query position.

```
Standard softmax for row i of attention matrix:

  s_ij = q_i @ k_j / sqrt(d)        for all j in [0, N)
  m_i  = max(s_i0, s_i1, ..., s_i(N-1))    # need ALL scores
  l_i  = sum(exp(s_ij - m_i))               # need ALL scores
  p_ij = exp(s_ij - m_i) / l_i
  o_i  = sum(p_ij * v_j)
```

If you only compute a tile of scores (say columns j=0..B_c), you do not know if a larger score exists in columns B_c..N. Your softmax would be wrong.

### The Online Softmax Trick

The key insight (Milakov and Gimelshein, 2018; refined by Dao et al., 2022) is that softmax can be computed incrementally. You maintain running statistics (maximum and sum of exponentials) that are corrected as new tiles arrive.
Consider processing tiles of K in order. After processing tile j (columns j*B_c to (j+1)*B_c - 1):

```
State after tile j:
  m_i^(j)  = max of all scores seen so far (across tiles 0..j)
  l_i^(j)  = sum of exp(s_ik - m_i^(j)) for all k seen so far
  o_i^(j)  = sum of exp(s_ik - m_i^(j)) * v_k / l_i^(j) for all k seen so far
```

When a new tile j+1 arrives with local scores s_i,new:

```
Algorithm: Online Softmax Update

1. Compute new tile scores: s_i,new = q_i @ K_new^T / sqrt(d)
2. Find new tile maximum:   m_new = max(s_i,new)
3. Update global maximum:   m_i^(j+1) = max(m_i^(j), m_new)
4. Correction factor:       alpha = exp(m_i^(j) - m_i^(j+1))
5. New tile exponentials:    exp_new = exp(s_i,new - m_i^(j+1))
6. Update running sum:      l_i^(j+1) = alpha * l_i^(j) + sum(exp_new)
7. Update output:           o_i^(j+1) = alpha * o_i^(j) + exp_new @ V_new
8. Final normalization:     output_i = o_i^(final) / l_i^(final)
```

The correction factor alpha rescales all previously accumulated values to account for the new maximum. If the new tile contains a larger score than anything seen before, previous contributions are downweighted. If the new tile has smaller scores, alpha equals 1 and previous values are unchanged.

This is mathematically exact. No approximation is involved. The final output is bit-for-bit identical to standard attention (up to floating-point associativity).

### The Full FlashAttention Forward Pass

With online softmax in hand, the algorithm tiles both Q (into blocks of B_r rows) and K, V (into blocks of B_c rows):

```
FlashAttention Forward (one head):

Input:  Q, K, V in HBM, each [N, d]
Output: O in HBM, [N, d]

Choose block sizes B_r, B_c such that:
  (2*B_r + B_c)*d + B_r*B_c fits in SRAM

for i = 0, 1, ..., ceil(N/B_r) - 1:
    Load Q_i = Q[i*B_r : (i+1)*B_r] from HBM to SRAM     # [B_r, d]
    Initialize: o_i = 0, l_i = 0, m_i = -inf              # in SRAM

    for j = 0, 1, ..., ceil(N/B_c) - 1:
        Load K_j = K[j*B_c : (j+1)*B_c] from HBM to SRAM # [B_c, d]
        Load V_j = V[j*B_c : (j+1)*B_c] from HBM to SRAM # [B_c, d]

        Compute S_ij = Q_i @ K_j^T / sqrt(d)              # [B_r, B_c] in SRAM
        Compute m_ij = rowmax(S_ij)                        # [B_r]
        Update m_new = max(m_i, m_ij)                      # [B_r]
        Compute P_ij = exp(S_ij - m_new)                   # [B_r, B_c] in SRAM
        Compute l_new = exp(m_i - m_new) * l_i + rowsum(P_ij)
        Update o_i = exp(m_i - m_new) * o_i + P_ij @ V_j
        Update m_i = m_new, l_i = l_new

    Write O_i = o_i / l_i to HBM                          # [B_r, d]
```

The attention matrix S_ij is computed in SRAM, used immediately for the matmul with V_j, and then discarded. It never touches HBM. The only HBM writes are the final output blocks O_i.

### SRAM Budget Calculation

For A100 with 192 KB shared memory per SM (M = 192 KB = 96K FP16 elements):

```
Required SRAM per tile (in FP16 elements):
  Q_i:   B_r x d
  K_j:   B_c x d
  V_j:   B_c x d
  S_ij:  B_r x B_c
  O_i:   B_r x d
  Stats: 2 x B_r  (m_i and l_i)

Total = (2*B_r + 2*B_c) * d + B_r * B_c + 2*B_r

With d=128, B_r=128, B_c=128:
  = (256 + 256) * 128 + 128*128 + 256
  = 65,536 + 16,384 + 256
  = 82,176 elements
  = 164,352 bytes (FP16)
  = 160.5 KB  <  192 KB  (fits!)
```

In practice, FlashAttention uses slightly smaller blocks to leave room for registers and other kernel state. The actual block sizes are tuned per GPU architecture.

---

