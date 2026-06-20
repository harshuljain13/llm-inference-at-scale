# 6.1 Tensor Parallelism

> Distributing LLM inference across multiple GPUs by splitting every layer

---

From Module 01.1, you know the bandwidth wall limits a single GPU to ~125 tokens/sec for 8B. For 70B, one GPU cannot even hold the weights. This module explains tensor parallelism: splitting a model across GPUs so each reads only a fraction of the weights per token. We will derive the math behind column and row parallel splits, quantify the communication cost of AllReduce, and show when NVLink versus PCIe determines whether TP is viable.

---

## Learning Objectives

By the end of this module, you will:

- Understand data, tensor, and pipeline parallelism and when each applies
- Configure multi-GPU vLLM deployments with the correct TP degree
- Calculate per-GPU memory requirements for distributed inference
- Choose the right parallelism strategy based on model size, latency target, and hardware

---

## Parallelism Strategies Overview

Before diving into tensor parallelism specifically, we need to understand the three fundamental approaches to distributing inference across GPUs. Each solves a different bottleneck: data parallelism addresses throughput for models that fit on one GPU, tensor parallelism addresses latency for models that do not, and pipeline parallelism addresses memory for models too large for a single node. The choice depends on whether your constraint is memory capacity, memory bandwidth, or request volume.

### Data Parallelism

Data parallelism is the simplest strategy and the one you should try first. If the model fits on a single GPU, you replicate it across N GPUs and route different requests to each replica independently. There is zero inter-GPU communication during the forward pass, which means scaling is nearly perfect: 4 replicas handle close to 4x the request volume.

The limitation is obvious: every replica must hold the full model. When Llama 70B requires 140 GB in FP16 and your GPU has 80 GB, data parallelism cannot help. That is where tensor parallelism comes in.

```
   Data Parallelism (DP):

   +----------+  +----------+  +----------+  +----------+
   |  GPU 0   |  |  GPU 1   |  |  GPU 2   |  |  GPU 3   |
   |  Model   |  |  Model   |  |  Model   |  |  Model   |
   | Batch 0  |  | Batch 1  |  | Batch 2  |  | Batch 3  |
   +----------+  +----------+  +----------+  +----------+
   No communication during forward pass.
```

### Tensor Parallelism

Tensor parallelism takes a fundamentally different approach: instead of replicating the model, it splits every layer across GPUs. Each GPU holds a slice of every weight matrix and computes a partial result. After each layer, GPUs synchronize via AllReduce to combine their partial outputs into the correct full result.

This means TP reduces per-token latency (each GPU does less work per layer) but introduces a hard dependency on interconnect speed. Every layer requires an AllReduce, so if your GPUs communicate over slow PCIe instead of NVLink, the synchronization cost dominates and TP becomes counterproductive beyond TP=2.

```
   Tensor Parallelism (TP):

   +-----------------------------------------------------------+
   |                      Single Layer                          |
   |  +--------+  +--------+  +--------+  +--------+          |
   |  | GPU 0  |  | GPU 1  |  | GPU 2  |  | GPU 3  |          |
   |  | Slice  |--|  Slice |--|  Slice |--|  Slice |          |
   |  |  0-24  |  | 25-49  |  | 50-74  |  | 75-99  |          |
   |  +--------+  +--------+  +--------+  +--------+          |
   |              <-- AllReduce after each layer -->            |
   +-----------------------------------------------------------+
```

### Pipeline Parallelism

Pipeline parallelism assigns different layers to different GPUs in sequence. GPU 0 holds layers 0-7, GPU 1 holds layers 8-15, and so on. A token's hidden state flows through the pipeline one stage at a time, with only point-to-point communication between adjacent stages.

The advantage over TP is that communication is minimal (one send/receive per stage boundary rather than AllReduce across all GPUs). The disadvantage is pipeline bubbles: when one stage is computing, the others sit idle unless you use micro-batching to keep the pipeline full. This makes PP better suited for throughput than latency.

```
   Pipeline Parallelism (PP):

   +----------+    +----------+    +----------+    +----------+
   |  GPU 0   |--->|  GPU 1   |--->|  GPU 2   |--->|  GPU 3   |
   |Layer 0-7 |    |Layer 8-15|    |Layer16-23|    |Layer24-31|
   +----------+    +----------+    +----------+    +----------+
```

The practical rule is: use TP within a node (where NVLink provides 600-900 GB/s), and PP across nodes (where InfiniBand provides 200-400 GB/s). This exploits the bandwidth hierarchy rather than fighting it.

---

## Tensor Parallelism Deep Dive

Now that you understand where TP sits in the parallelism landscape, let us examine exactly how it works at the matrix multiplication level. The key insight from Megatron-LM (Shoeybi et al., 2019) is that linear layers Y = XW can be split along either the column or row dimension, and by alternating between these two splits within each transformer block, you can avoid redundant communication.

### How Column Parallelism Works

Consider the first linear layer in the feed-forward network (FFN). It projects from hidden_size to ffn_hidden_size, typically a 4x expansion. Column parallelism splits the weight matrix W along its columns, giving each GPU a vertical slice:

```
   Column Parallel (for first linear in FFN):

   X [batch, hidden]  x  W [hidden, ffn_hidden]  =  Y [batch, ffn]

   Split W by columns:
   +-----------------------------------------------------------+
   |  GPU 0: X * W[:, :ffn/4]      -> Y[:, :ffn/4]            |
   |  GPU 1: X * W[:, ffn/4:ffn/2] -> Y[:, ffn/4:ffn/2]      |
   |  GPU 2: X * W[:, ffn/2:3ffn/4] -> Y[:, ffn/2:3ffn/4]    |
   |  GPU 3: X * W[:, 3ffn/4:]     -> Y[:, 3ffn/4:]           |
   +-----------------------------------------------------------+

   No communication needed! Each GPU has a partial output column.
```

The critical property here is that each GPU can compute its portion independently using the full input X. No synchronization is needed at this stage because concatenating the partial outputs would reconstruct the full Y. But we do not actually concatenate; instead, we feed these partial results directly into the next linear layer using row parallelism.

### How Row Parallelism Works

The second FFN linear layer projects from ffn_hidden_size back to hidden_size. Row parallelism splits W along its rows, which aligns perfectly with the column-split outputs from the previous layer:

```
   Row Parallel (for second linear in FFN):

   Y [batch, ffn_hidden]  x  W [ffn_hidden, hidden]  =  Z [batch, h]

   Split W by rows (and Y accordingly):
   +-----------------------------------------------------------+
   |  GPU 0: Y[:, :ffn/4] * W[:ffn/4, :]      -> Z_partial_0  |
   |  GPU 1: Y[:, ffn/4:ffn/2] * W[ffn/4:ffn/2, :] -> Z_partial_1 |
   |  GPU 2: Y[:, ffn/2:3ffn/4] * W[ffn/2:3ffn/4, :] -> Z_partial_2|
   |  GPU 3: Y[:, 3ffn/4:] * W[3ffn/4:, :]    -> Z_partial_3  |
   +-----------------------------------------------------------+

   AllReduce: Z = Z_partial_0 + Z_partial_1 + Z_partial_2 + Z_partial_3
```

This is where the communication happens. Each GPU holds a partial sum of the output, and AllReduce combines them so every GPU has the correct result to feed into the next layer. The elegance of column-then-row parallelism is that you need only one AllReduce per FFN block rather than two, halving the communication cost versus naive approaches.

### Attention Head Parallelism

Multi-head attention is naturally parallel because each head operates independently on its own Q, K, V projections. Tensor parallelism exploits this by assigning contiguous groups of heads to each GPU:

```
   Model: 32 attention heads, TP=4

   +-----------------------------------------------------------+
   |  GPU 0: Heads 0-7    (8 heads)                            |
   |  GPU 1: Heads 8-15   (8 heads)                            |
   |  GPU 2: Heads 16-23  (8 heads)                            |
   |  GPU 3: Heads 24-31  (8 heads)                            |
   +-----------------------------------------------------------+

   Each GPU:
   - Computes Q, K, V for its heads only
   - Runs attention (softmax, weighted sum) independently
   - Produces partial output

   After attention: AllReduce to combine outputs
```

Each GPU computes attention for its assigned heads using only its slice of the Q, K, V weight matrices. Since heads are independent, there is no communication during attention computation itself. The AllReduce happens only after the output projection, making attention the most naturally TP-friendly operation in the transformer.

The hard constraint is that TP must divide num_attention_heads evenly. A model with 32 heads supports TP of 1, 2, 4, 8, 16, or 32. A model with 40 heads (like Llama 65B) supports TP of 1, 2, 4, 5, 8, 10, 20, or 40. If your desired TP degree does not divide evenly, you cannot use it.

---

## NCCL Collective Operations

Tensor parallelism depends entirely on efficient GPU-to-GPU communication. The NVIDIA Collective Communication Library (NCCL) provides the primitives that make this possible. Understanding these operations is essential because the choice of collective determines both the communication pattern and the bandwidth requirement at each layer boundary.

### Key Operations for Distributed Inference

Three collectives matter for inference. AllReduce is the workhorse of tensor parallelism, used after every row-parallel layer. AllGather concatenates distributed tensors when you need the full result on every GPU. Broadcast distributes inputs from one GPU to all others at the start of inference.

```
   AllReduce (used in Tensor Parallelism):

   Before:  GPU0=[1,2]  GPU1=[3,4]  GPU2=[5,6]  GPU3=[7,8]
   After:   GPU0=[16,20] GPU1=[16,20] GPU2=[16,20] GPU3=[16,20]

   All GPUs get the element-wise sum of all inputs.
   Used after every row-parallel matmul to reconstruct the full hidden state.

   ---

   AllGather (used for gathering outputs):

   Before:  GPU0=[A]  GPU1=[B]  GPU2=[C]  GPU3=[D]
   After:   GPU0=[A,B,C,D] GPU1=[A,B,C,D] GPU2=[A,B,C,D] ...

   All GPUs get the concatenation of all inputs.
   Used when the full tensor is needed (e.g., vocabulary projection at the final layer).

   ---

   Broadcast (used for input distribution):

   Before:  GPU0=[data]  GPU1=[?]  GPU2=[?]  GPU3=[?]
   After:   GPU0=[data]  GPU1=[data]  GPU2=[data]  GPU3=[data]

   One GPU's data copied to all others.
   Used at the start of each forward pass to distribute input tokens.
```

### Communication Overhead: Why Interconnect Speed Determines TP Viability

The fundamental question for any TP deployment is whether the interconnect can keep up with the computation. Let us derive the exact bandwidth requirement for Llama 8B with TP=4 to make this concrete.

Each AllReduce transfers a tensor of shape [batch * seq_len, hidden_size] in FP16. During decode (one token per sequence), the data volume per AllReduce is:

```
   AllReduce data per layer (Llama 8B, TP=4):
   - Hidden size: 4096
   - Batch x Seq: 32 x 1 = 32 (decode, one new token per sequence)
   - Data: 32 x 4096 x 2 bytes = 256 KB per AllReduce
   - AllReduces per layer: 2 (one after attention, one after FFN)
   - Layers: 32
   - Total per token step: 256 KB x 2 x 32 = 16 MB

   At 100 tokens/sec decode speed: 1.6 GB/s sustained bandwidth needed
```

1.6 GB/s sounds trivial compared to NVLink's 900 GB/s. But this calculation assumes batch=32. At batch=512 (a busy production server), the requirement scales to 25.6 GB/s. More importantly, the latency of each AllReduce matters for decode speed: even at 0.5 microseconds per AllReduce, 64 AllReduces per forward pass add 32 microseconds of pure synchronization overhead per token.

The interconnect comparison makes the NVLink requirement clear:

```
   Interconnect Comparison:
   +-------------------+-----------+---------+-------------------+
   | Interconnect      | Bandwidth | Latency | Best For          |
   +-------------------+-----------+---------+-------------------+
   | PCIe 4.0 x16      | 32 GB/s   | ~1 us   | TP=2 (limited)    |
   | PCIe 5.0 x16      | 64 GB/s   | ~1 us   | TP=2-4            |
   | NVLink 3 (A100)   | 600 GB/s  | ~0.5 us | TP=2-8            |
   | NVLink 4 (H100)   | 900 GB/s  | ~0.5 us | TP=2-8            |
   | InfiniBand HDR    | 200 GB/s  | ~1 us   | Multi-node PP     |
   | InfiniBand NDR    | 400 GB/s  | ~1 us   | Multi-node PP     |
   +-------------------+-----------+---------+-------------------+

   Rule: Use NVLink for TP within a node, InfiniBand for PP across nodes.
```

The 10-20x bandwidth advantage of NVLink over PCIe is why TP efficiency drops sharply on PCIe-connected systems. At TP=8 on PCIe, you spend more time communicating than computing, negating the entire purpose of parallelism.

---

## Multi-GPU vLLM Configuration

With the theory established, let us see how tensor parallelism works in practice. vLLM makes TP deployment straightforward: you specify --tensor-parallel-size and the framework handles weight sharding, AllReduce insertion, and NCCL initialization automatically. But choosing the right TP degree requires understanding the memory arithmetic we derived above.

### Basic Multi-GPU Setup

The simplest configuration is specifying the TP degree. vLLM will shard model weights evenly across the available GPUs and insert AllReduce operations at the correct layer boundaries:

```bash
# 2 GPUs with Tensor Parallelism
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 2

# 4 GPUs with Tensor Parallelism
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4

# 8 GPUs (full p4d.24xlarge)
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 8
```

### Memory Calculation for Multi-GPU

When deciding the TP degree, you need to verify that each GPU can hold its share of weights plus the KV cache for your target batch size. The following calculation shows the key subtlety: model weights are divided by TP, but KV cache is NOT divided (each GPU needs the full cache for its assigned attention heads, and with GQA the KV heads may all fit on each GPU).

```python
def calculate_multi_gpu_vram(
    parameters_b: float,
    layers: int,
    kv_heads: int,
    head_dim: int,
    batch_size: int,
    sequence_length: int,
    tensor_parallel: int,
    quantization: str = "fp16"
) -> dict:
    """
    Calculate per-GPU VRAM for tensor parallel deployment.
    
    The key insight: weights scale as 1/TP, but KV cache does NOT
    because each GPU needs the full cache for its attention heads.
    """
    # Bytes per parameter
    quant_bytes = {"fp16": 2, "int8": 1, "int4": 0.5}
    param_bytes = quant_bytes.get(quantization, 2)

    # Model weights per GPU (divided by TP)
    total_model_gb = (parameters_b * 1e9 * param_bytes) / (1024**3)
    model_per_gpu_gb = total_model_gb / tensor_parallel

    # KV cache per GPU
    # Note: KV cache is NOT divided by TP (each GPU needs full cache for its heads)
    # But with GQA, we have fewer KV heads
    kv_heads_per_gpu = kv_heads  # Full KV cache on each GPU
    kv_cache_bytes = (
        2 * layers * kv_heads_per_gpu * head_dim *
        sequence_length * batch_size * 2  # FP16
    )
    kv_cache_gb = kv_cache_bytes / (1024**3)

    # Overhead (activation memory, NCCL buffers, framework state)
    overhead_gb = model_per_gpu_gb * 0.2

    # Total per GPU
    total_per_gpu = model_per_gpu_gb + kv_cache_gb + overhead_gb

    return {
        "model_per_gpu_gb": round(model_per_gpu_gb, 2),
        "kv_cache_gb": round(kv_cache_gb, 2),
        "overhead_gb": round(overhead_gb, 2),
        "total_per_gpu_gb": round(total_per_gpu, 2),
        "tensor_parallel": tensor_parallel,
    }


# Example: Llama 3.1 70B on 4 GPUs
result = calculate_multi_gpu_vram(
    parameters_b=70,
    layers=80,
    kv_heads=8,
    head_dim=128,
    batch_size=32,
    sequence_length=4096,
    tensor_parallel=4,
    quantization="fp16"
)
print(result)
# {'model_per_gpu_gb': 32.5, 'kv_cache_gb': 40.0, 'overhead_gb': 6.5,
#  'total_per_gpu_gb': 79.0, 'tensor_parallel': 4}
```

This result reveals a critical insight: even with TP=4, a 70B model in FP16 requires 79 GB per GPU, which barely fits on an 80 GB A100. The KV cache (40 GB) actually exceeds the model weight share (32.5 GB). This is why production deployments often combine TP with quantization: INT8 halves model_per_gpu_gb to 16.25 GB, freeing space for larger batches.

### Scaling Efficiency

The gap between theoretical and actual speedup reveals the communication overhead cost. On NVLink, TP=4 achieves 90% efficiency (3.6x instead of 4x). On PCIe, the same configuration achieves only 62.5% efficiency (2.5x). This difference compounds: at TP=8, NVLink delivers 81% efficiency while PCIe delivers only 44%.

```
   Theoretical vs Actual Speedup (Llama 70B):

   TP Size    Theoretical    Actual (NVLink)    Actual (PCIe)
   ----------------------------------------------------------------
      1          1.0x            1.0x              1.0x
      2          2.0x            1.9x              1.6x
      4          4.0x            3.6x              2.5x
      8          8.0x            6.5x              3.5x

   Why less than linear?
   - AllReduce communication overhead (grows with TP degree)
   - Synchronization barriers (all GPUs must wait for the slowest)
   - Load imbalance (embedding/final layers not perfectly divisible)

   NVLink is ESSENTIAL for efficient TP > 2
```

The transition from communication overhead to the MoE section is natural: if tensor parallelism is limited by AllReduce bandwidth, what happens when you have a model architecture that requires even more communication? Mixture-of-Experts models introduce AllToAll, a fundamentally different collective with worse scaling properties.

---

## Mixture of Experts (MoE) Inference

MoE models like Mixtral and DeepSeek-V3 present a unique parallelism challenge. Unlike dense models where every parameter participates in every token's computation, MoE models route each token to a small subset of "expert" sub-networks. This means total parameter count is much larger than active parameter count, creating a memory-vs-compute asymmetry that standard TP cannot solve alone.

### MoE Architecture

The core idea is replacing the monolithic FFN block with multiple smaller FFN "experts" plus a learned router that selects which experts process each token. The router produces a probability distribution over experts and selects the top-k (typically k=2) for each token.

```
   Standard FFN:
   Input -> FFN -> Output
   All parameters used for every token

   MoE FFN:
                    +---------+
                    | Router  |
                    +----+----+
                         | Select top-k experts
         +---------------+---------------+
         v               v               v
   +----------+   +----------+   +----------+
   | Expert 0 |   | Expert 1 |   | Expert 7 |  (8 experts)
   +----------+   +----------+   +----------+
         |               |               |
         +---------------+---------------+
                         v
                   Weighted Sum
```

The implication for inference is stark: Mixtral 8x7B has 46.7B total parameters but activates only ~13B per token. You need VRAM for all 46.7B (they must be resident for any token that might route to them), but you only perform 13B worth of FLOPs. This creates a model that is memory-capacity bound rather than compute bound, which fundamentally changes the parallelism strategy.

### Expert Parallelism

Expert parallelism (EP) distributes experts across GPUs rather than splitting each expert's weights. Each GPU holds a subset of experts and processes only the tokens routed to those experts:

```
   Mixtral 8x7B with EP=4:

   +-----------------------------------------------------------+
   |  GPU 0: Experts 0, 1    (2 experts)                       |
   |  GPU 1: Experts 2, 3    (2 experts)                       |
   |  GPU 2: Experts 4, 5    (2 experts)                       |
   |  GPU 3: Experts 6, 7    (2 experts)                       |
   +-----------------------------------------------------------+

   Routing flow:
   1. All GPUs compute router logits (cheap, small network)
   2. Determine which tokens go to which experts
   3. AllToAll: Send tokens to GPUs holding their assigned experts
   4. Each GPU processes tokens for its local experts
   5. AllToAll: Return processed tokens to originating GPUs
```

The critical difference from TP is the communication primitive. TP uses AllReduce (same data shape, element-wise sum). EP uses AllToAll (variable data, each GPU sends different amounts to different peers). AllToAll is harder to optimize and more sensitive to load imbalance: if the router sends 80% of tokens to 2 experts on one GPU, that GPU becomes a bottleneck while others sit idle.

### Wide Expert Parallelism (Wide-EP)

Standard EP assigns the minimum number of GPUs needed to hold all experts in memory. Wide-EP, introduced by Anyscale, deliberately uses MORE GPUs than necessary and applies tensor parallelism within each expert:

```
   Standard EP (8 experts, 8 GPUs):
   +------++------++------++------++------++------++------++------+
   | E0   || E1   || E2   || E3   || E4   || E5   || E6   || E7   |
   |GPU 0 ||GPU 1 ||GPU 2 ||GPU 3 ||GPU 4 ||GPU 5 ||GPU 6 ||GPU 7 |
   +------++------++------++------++------++------++------++------+

   Wide-EP (8 experts, 16 GPUs, TP=2 per expert):
   +------------++------------++------------++------------+
   |  Expert 0  ||  Expert 1  ||  Expert 2  ||  Expert 3  |
   |GPU 0|GPU 1 ||GPU 2|GPU 3 ||GPU 4|GPU 5 ||GPU 6|GPU 7 |
   +------------++------------++------------++------------+
   +------------++------------++------------++------------+
   |  Expert 4  ||  Expert 5  ||  Expert 6  ||  Expert 7  |
   |GPU 8|GPU 9 ||GPU10|GPU11 ||GPU12|GPU13 ||GPU14|GPU15 |
   +------------++------------++------------++------------+
```

Why does this help? The bottleneck for MoE inference is memory bandwidth: each expert's weights must be read from HBM for every token routed to it. By splitting each expert across 2 GPUs, you double the memory bandwidth available per expert, cutting the weight-read time in half. Anyscale measured 1.5-2x throughput improvement over standard EP because the fundamental constraint (HBM bandwidth per expert) is directly parallelized.

The trade-off is using more GPUs for the same model, but each GPU achieves higher utilization because the memory bandwidth bottleneck (not compute) is what limits MoE serving throughput.

### The MoE Double Penalty

A common misconception is that MoE models always outperform dense models of equivalent active compute. In practice, MoE models suffer two compounding penalties at inference time (arxiv:2603.08960):

**Penalty 1: Routing Fragmentation.** Tokens in a batch get routed to different experts. Each expert sees only a fraction of the batch, reducing GPU compute utilization. A dense model processes ALL tokens together, achieving full batch efficiency.

**Penalty 2: KV Cache Memory Pressure.** MoE models have the same attention layers as dense equivalents, but total model memory is much larger (all experts in VRAM). This leaves less VRAM for KV cache, meaning fewer concurrent sequences and lower throughput. Mixtral 8x7B uses 87 GB for weights versus 13 GB for a dense 13B with equivalent active compute.

```
   When Dense Models Win:
   +-----------------------------+--------+-----------------------+
   | Scenario                    | Winner | Why                   |
   +-----------------------------+--------+-----------------------+
   | High batch, short context   | Dense  | Routing fragmentation |
   | Memory-constrained deploy   | Dense  | KV cache pressure     |
   | Latency-critical (small TP) | Dense  | AllToAll overhead     |
   | Low batch, long context     | MoE    | Less compute/token    |
   | Quality-per-FLOP priority   | MoE    | More params, same cost|
   | Wide-EP available (many GPU)| MoE    | Penalties mitigated   |
   +-----------------------------+--------+-----------------------+

   Key insight: MoE advantage grows with GPU count. On few GPUs,
   the double penalty can make a dense model faster at serving.
```

### High-Performance AllToAll for Expert Routing

At scale (64+ GPUs), AllToAll becomes the dominant cost in MoE serving. Perplexity developed a custom AllToAll kernel that bypasses NCCL entirely, communicating directly between GPUs over AWS EFA (Elastic Fabric Adapter) using RDMA:

```
   Standard NCCL AllToAll:
   +----------+                              +----------+
   |  GPU 0   |---- NCCL (store-and-forward) |  GPU 8   |
   |  Node 0  |     through CPU/NIC stack    |  Node 1  |
   +----------+                              +----------+
   Latency: ~100-200 us per AllToAll

   Perplexity custom kernel:
   +----------+                              +----------+
   |  GPU 0   |---- Direct RDMA over EFA ----|  GPU 8   |
   |  Node 0  |     (GPU-initiated, async)   |  Node 1  |
   +----------+                              +----------+
   Latency: ~10-20 us per AllToAll
```

The 10x latency reduction enables efficient EP across 64-128 GPUs and makes Wide-EP practical at multi-node scale. This is critical for models like DeepSeek-V3 (256 experts) where standard NCCL AllToAll would make multi-node EP unviable.

### Counter-Intuitive MoE Scaling: More GPUs Improve Both Latency AND Throughput

Dense model scaling with TP involves a classic trade-off: more GPUs reduce latency (less compute per GPU) but also reduce throughput (more communication overhead). MoE with Wide-EP breaks this trade-off entirely because the bottleneck is memory bandwidth, not compute:

```
   Dense (TP=4 -> TP=8):
   +------------------------------------------------------------+
   | Compute: 2x faster (split across more GPUs)          OK    |
   | Communication: 2x more AllReduce overhead            BAD   |
   | Batch capacity: Same (KV cache per GPU unchanged)          |
   | Net: Latency down, Throughput DOWN                         |
   +------------------------------------------------------------+

   MoE (EP=8 -> Wide-EP=16):
   +------------------------------------------------------------+
   | Memory BW: 2x per expert (weight reads parallelized) GOOD  |
   | Communication: AllToAll (fast with custom kernels)    OK    |
   | Batch capacity: 2x more VRAM for KV cache            GOOD  |
   | Net: Latency down, Throughput UP                           |
   +------------------------------------------------------------+
```

Measured results from Anyscale with Mixtral 8x7B confirm this:

```
   +-------------+----------------+--------------------+---------+
   | Config      | Latency (TTFT) | Throughput (tok/s) | GPUs    |
   +-------------+----------------+--------------------+---------+
   | EP=8        | 45 ms          | 2,100              | 8       |
   | Wide-EP=16  | 28 ms          | 3,800              | 16      |
   | Wide-EP=32  | 19 ms          | 5,200              | 32      |
   +-------------+----------------+--------------------+---------+
```

The implication is that for MoE models, the cost-performance curve is fundamentally different from dense deployments. Spending more on GPUs improves both latency and throughput simultaneously, making the ROI calculation more favorable than it appears at first glance.

---

## Choosing Parallelism Strategy

With all three parallelism strategies and their MoE variants understood, the practical question is: given a model size, hardware budget, and latency requirement, which combination should you use? The decision tree below encodes the rules we derived throughout this module.

### Decision Guide

The first branch is whether the model fits on one GPU. If yes, you do not need parallelism at all, and should use data parallelism (multiple replicas) to scale throughput. If the model does not fit on one GPU but fits on one node (typically 8 GPUs with NVLink), use tensor parallelism. If it exceeds one node, combine TP within each node with PP across nodes.

```
   Model fits on 1 GPU?
   +-- Yes -> Use single GPU (no parallelism needed)
   |         Consider DP for throughput scaling
   |
   +-- No -> Model fits on 1 node (8 GPUs)?
             +-- Yes -> Use Tensor Parallelism
             |         TP = min(8, model_size / gpu_memory)
             |         Requires NVLink for efficiency
             |
             +-- No -> Need multi-node
                      Use TP within node + PP across nodes
                      Example: TP=8, PP=2 for 16 GPUs
```

For quick reference, here are the standard configurations used in production for common model sizes on AWS:

```
   Quick Reference:
   +------------+-----------------+------------------------------+
   | Model Size | Instance        | Parallelism                  |
   +------------+-----------------+------------------------------+
   | <=13B      | g5.xlarge       | None (single GPU)            |
   | 13-30B     | g5.12xlarge     | TP=2 or TP=4                |
   | 30-70B     | p4d.24xlarge    | TP=4 or TP=8                |
   | 70-140B    | p4d.24xlarge    | TP=8                         |
   | >140B      | 2x p4d.24xlarge | TP=8, PP=2                   |
   | 405B       | 4x p5.48xlarge  | TP=8, PP=4                   |
   +------------+-----------------+------------------------------+
```

Note that quantization shifts these boundaries. A 70B model in INT4 (35 GB) fits on a single 80 GB GPU, eliminating the need for TP entirely. Always calculate the actual memory requirement (using the formula from the Multi-GPU section above) before choosing your parallelism strategy.

---

## Key Takeaways

1. **Tensor Parallelism for latency.** Splits every layer across GPUs so each reads fewer weights per token. Requires fast interconnect (NVLink) because AllReduce happens after every layer.

2. **Pipeline Parallelism for throughput.** Assigns different layers to different GPUs in sequence. Works across nodes with InfiniBand because communication is only point-to-point between adjacent stages.

3. **NVLink is the TP enabler.** PCIe limits TP efficiency to ~44% at TP=8 versus 81% on NVLink. If your instance lacks NVLink, keep TP at 2 and use quantization to fit the model instead.

4. **TP must divide attention heads evenly.** Common valid values: 2, 4, 8. Check your model's num_attention_heads before choosing TP degree.

5. **MoE needs all experts in memory but activates only a subset.** This creates a memory-bandwidth bottleneck that standard TP cannot solve. Use Expert Parallelism or Wide-EP.

6. **Communication overhead limits scaling.** Expect 80-90% efficiency with NVLink TP. Beyond TP=8, returns diminish rapidly and PP becomes preferable.

7. **MoE scaling is counter-intuitive.** More GPUs improve BOTH latency and throughput (unlike dense models where it is a trade-off), because the bottleneck is memory bandwidth per expert.

---

## Lab Preview: Tensor Parallelism

In the accompanying lab, you will:

- Deploy Llama 70B with TP=2, TP=4, TP=8 and measure per-token latency at each configuration
- Quantify the scaling efficiency gap between NVLink and PCIe by comparing decode throughput
- Calculate the optimal TP degree for a given latency SLA and batch size using the memory formula above
- Visualize the communication overhead as a function of TP degree and batch size

---

## References

1. Shoeybi et al. "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019)
2. Narayanan et al. "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (2021)
3. Fedus et al. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (2021)
4. NVIDIA NCCL Documentation
5. Anyscale. "Wide Expert Parallelism for MoE Inference" (2024)
6. "The MoE Double Penalty" (arxiv:2603.08960, 2026)
7. Perplexity. "10x Faster AllToAll on AWS EFA" (2025)
