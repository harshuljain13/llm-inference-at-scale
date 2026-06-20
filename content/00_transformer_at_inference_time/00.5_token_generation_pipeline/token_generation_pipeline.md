# 0.5 The Token Generation Pipeline

> The difference between understanding transformers and _really_ understanding them is knowing exactly which bytes move where, and why that determines everything about your inference costs.

---

## Prerequisites and Context

From Module 00.0, you know the transformer has 32 layers with attention and MLP blocks. From Module 01.1, you know GPU memory is limited and the bandwidth wall constrains decode speed. This module explains the KV cache: the memory structure that grows with every generated token and ultimately determines how many users your GPU can serve.

---

## Learning Objectives

By the end of this module, you will:

- Trace tensor shapes through every operation and explain _why_ each shape is what it is
- Understand the KV cache at the byte level: not just "it caches K and V" but exactly what is stored and why
- Calculate KV cache memory requirements from first principles (and know when the formulas lie)
- Explain why GQA exists, what problem it solves, and the exact tradeoff it makes
- Understand the prefill/decode split at a level where you can predict which phase will bottleneck

---


![KV Cache Growth by Model](images/kv_cache_growth.png)

## The Insight That Changes Everything

Here is what most tutorials get wrong: they explain transformers as a sequence of operations. But for inference engineering, you need to think about transformers as a **memory access pattern**.

Every forward pass is fundamentally:

1. Read weights from HBM, do math, write activations
2. Read activations from HBM, do math, write activations
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

This is why everything in LLM inference optimization is about memory, not compute.

---

## The Token Generation Pipeline: A Byte-Level View

To understand the KV cache, you first need to see where it fits in the full generation pipeline. The following trace walks through what actually happens when you generate one token, not at a conceptual level but at the level of actual bytes moving through the GPU memory hierarchy.

### Step 1: Tokenization (CPU, negligible)

Tokenization converts raw text into integer IDs that index into the embedding table. This step runs on the CPU and is never a bottleneck for inference performance.

```python
# Input: "The capital of France is"
# Output: [464, 3139, 286, 4881, 318]  (5 tokens)

# Memory: 5 x 4 bytes = 20 bytes (int32 token IDs)
# This is nothing. Tokenization is never your bottleneck.
```

### Step 2: Embedding Lookup (GPU, memory-bound)

The embedding table maps each token ID to a dense vector. This is a pure gather operation: you read a few rows from a large table stored in HBM. No arithmetic is involved.

```python
# Embedding table: [vocab_size, hidden_dim] = [128256, 4096]
# Size: 128256 x 4096 x 2 bytes = 1.05 GB

# Operation: Gather 5 rows from the table
# Output: [5, 4096] = 40 KB

# This is a pure memory operation, no compute.
# You read 5 x 4096 x 2 = 40 KB from a 1 GB table.
```

The embedding table is 1 GB but you only ever read tiny slices of it. This is why embedding tables do not benefit much from quantization: you are not reading the whole thing, just scattered rows. The memory bandwidth cost comes from the random access pattern, not the table size.

### Step 3: The Transformer Block (where the money is)

The transformer block accounts for 95%+ of your inference time. Every layer applies the same sequence of operations: normalization, attention projections, the attention computation itself, and the MLP. The following breakdown traces each sub-operation with exact tensor shapes and byte counts so you can see precisely where time is spent.

#### 3a: RMSNorm (cheap, memory-bound)

RMSNorm is a lightweight normalization applied before each attention and MLP block. It stabilizes training and inference but contributes negligible time to the overall forward pass.

```python
# Input: [batch, seq, hidden] = [1, 5, 4096]
# Weights: [hidden] = [4096] = 8 KB
# Output: [1, 5, 4096]

# Operation: x * rsqrt(mean(x^2) + eps) * weight
# FLOPs: ~3 x batch x seq x hidden = 61K FLOPs
# Memory: Read 40 KB input + 8 KB weights, write 40 KB output

# This is so cheap it is essentially free.
```

#### 3b: Attention Projections (the first big memory read)

This is where things get expensive. The attention mechanism requires projecting the input hidden state into queries, keys, and values through four large weight matrices. These projections dominate the attention block's memory footprint.

```python
# Llama 3.1 8B uses GQA: 32 query heads, 8 KV heads, 128 dim per head

W_q: [4096, 4096]   = 32 MB   # 32 heads x 128 dim = 4096
W_k: [4096, 1024]   = 8 MB    # 8 KV heads x 128 dim = 1024
W_v: [4096, 1024]   = 8 MB    # 8 KV heads x 128 dim = 1024
W_o: [4096, 4096]   = 32 MB   # Output projection

Total per layer: 80 MB
Total for 32 layers: 2.56 GB  # Just the attention projections!
```

The Q and O projections are 4x larger than K and V projections. This is the GQA tradeoff: you save memory on KV cache but the projection weights are still dominated by Q and O. GQA does not reduce model size, only KV cache size.

```python
# The actual computation:
# Input: [1, 5, 4096]

Q = input @ W_q  # [1, 5, 4096] @ [4096, 4096] -> [1, 5, 4096]
K = input @ W_k  # [1, 5, 4096] @ [4096, 1024] -> [1, 5, 1024]
V = input @ W_v  # [1, 5, 4096] @ [4096, 1024] -> [1, 5, 1024]

# Reshape for multi-head attention:
Q = Q.view(1, 5, 32, 128).transpose(1, 2)  # [1, 32, 5, 128]
K = K.view(1, 5, 8, 128).transpose(1, 2)   # [1, 8, 5, 128]
V = V.view(1, 5, 8, 128).transpose(1, 2)   # [1, 8, 5, 128]
```

#### 3c: The Attention Computation (where GQA gets interesting)

Here is where most explanations gloss over the details. With GQA, you have 32 query heads but only 8 KV heads. The mechanism works by having groups of query heads share the same key-value representations, which is what makes the KV cache so much smaller than in traditional multi-head attention.

```python
# GQA: Each KV head serves 4 query heads (32/8 = 4)
#
# Query heads 0-3   share KV head 0
# Query heads 4-7   share KV head 1
# Query heads 8-11  share KV head 2
# ...
# Query heads 28-31 share KV head 7

# Implementation: Expand K and V to match Q's head count
K_expanded = K.repeat_interleave(4, dim=1)  # [1, 8, 5, 128] -> [1, 32, 5, 128]
V_expanded = V.repeat_interleave(4, dim=1)  # [1, 8, 5, 128] -> [1, 32, 5, 128]

# Now standard attention:
scores = Q @ K_expanded.transpose(-2, -1) / sqrt(128)  # [1, 32, 5, 5]
attn_weights = softmax(scores, dim=-1)                  # [1, 32, 5, 5]
attn_output = attn_weights @ V_expanded                 # [1, 32, 5, 128]
```

The `repeat_interleave` does not actually copy memory in optimized implementations. FlashAttention and PagedAttention use index arithmetic to avoid the expansion. Conceptually, each query head is attending to the same KV cache, just with different learned query projections.

The attention matrix has shape [seq_len, seq_len]. For a 4096-token context, that is 16M elements per head, 512M elements total. This is why FlashAttention matters: it never materializes this full matrix in HBM, instead computing attention tile-by-tile in SRAM.

#### 3d: The MLP Block (the other half of the parameters)

The MLP is deceptively simple but contains roughly two-thirds of the model's parameters. While attention gets most of the conceptual attention, the MLP is where the majority of HBM reads happen during each forward pass.

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
gate = input @ W_gate           # [1, 5, 4096] @ [4096, 14336] -> [1, 5, 14336]
up = input @ W_up               # [1, 5, 4096] @ [4096, 14336] -> [1, 5, 14336]
hidden = silu(gate) * up        # Element-wise, [1, 5, 14336]
output = hidden @ W_down        # [1, 5, 14336] @ [14336, 4096] -> [1, 5, 4096]
```

The intermediate dimension (14336) is 3.5x the hidden dimension (4096). This ratio is a design choice: larger intermediate means more capacity but more memory. Llama chose 3.5x while GPT-2 used 4x. This is why the MLP dominates model size.

SwiGLU requires 3 weight matrices instead of the 2 that classic transformers use. Classic transformers compute `relu(x @ W1) @ W2`. SwiGLU computes `silu(x @ W_gate) * (x @ W_up) @ W_down`. The extra matrix is why Llama's MLP is 50% larger than you would expect from the intermediate dimension alone.

### Step 4: The LM Head (one more big matrix)

The final layer projects from hidden dimension back to vocabulary size to produce logits. This matrix is the same shape as the embedding table, though Llama 3.1 keeps them as separate parameters rather than tying their weights.

```python
# LM Head: [hidden, vocab] = [4096, 128256]
# Size: 4096 x 128256 x 2 = 1.05 GB

# Many models tie these weights (share the same matrix as embedding).
# Llama 3.1 does NOT tie weights, it has separate embedding and LM head.

logits = final_hidden @ W_lm_head  # [1, 5, 4096] @ [4096, 128256] -> [1, 5, 128256]

# For generation, you only need the last token's logits:
next_token_logits = logits[:, -1, :]  # [1, 128256]
```

The LM head is 1 GB but you only use one row of output during generation. You compute logits for all 128K vocab entries, but you only sample one token. This is unavoidable because you need the full probability distribution to sample from.

---

