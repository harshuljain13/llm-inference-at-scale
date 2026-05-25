# Module 5: Scaling and Parallelism

> Distributing LLM inference across multiple GPUs and nodes

---

## Learning Objectives

By the end of this module, you will:

- Understand data, tensor, and pipeline parallelism
- Configure multi-GPU vLLM deployments
- Calculate memory requirements for distributed inference
- Choose the right parallelism strategy for your model

---

## Parallelism Strategies Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PARALLELISM STRATEGIES                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Data Parallelism (DP):                                            │
│   ══════════════════════                                            │
│   • Same model on each GPU                                          │
│   • Different data batches                                          │
│   • No communication during forward pass                            │
│   • Best for: Multiple independent requests                         │
│                                                                     │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐               │
│   │ GPU 0   │  │ GPU 1   │  │ GPU 2   │  │ GPU 3   │               │
│   │ Model   │  │ Model   │  │ Model   │  │ Model   │               │
│   │ Batch 0 │  │ Batch 1 │  │ Batch 2 │  │ Batch 3 │               │
│   └─────────┘  └─────────┘  └─────────┘  └─────────┘               │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   Tensor Parallelism (TP):                                          │
│   ════════════════════════                                          │
│   • Model layers split across GPUs                                  │
│   • Each GPU holds slice of every layer                             │
│   • AllReduce communication each layer                              │
│   • Best for: Large models, low latency                             │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                      Single Layer                           │   │
│   │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐             │   │
│   │  │ GPU 0  │  │ GPU 1  │  │ GPU 2  │  │ GPU 3  │             │   │
│   │  │ Slice  │──│ Slice  │──│ Slice  │──│ Slice  │             │   │
│   │  │  0-24  │  │ 25-49  │  │ 50-74  │  │ 75-99  │             │   │
│   │  └────────┘  └────────┘  └────────┘  └────────┘             │   │
│   │              ← AllReduce after each layer →                 │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   Pipeline Parallelism (PP):                                        │
│   ══════════════════════════                                        │
│   • Different layers on different GPUs                              │
│   • Sequential processing through pipeline                          │
│   • Point-to-point communication between stages                     │
│   • Best for: Very large models, high throughput                    │
│                                                                     │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐         │
│   │ GPU 0   │───►│ GPU 1   │───►│ GPU 2   │───►│ GPU 3   │         │
│   │Layer 0-7│    │Layer 8-15│   │Layer16-23│   │Layer24-31│        │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tensor Parallelism Deep Dive

### How Tensor Parallelism Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                 TENSOR PARALLELISM MECHANICS                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Linear Layer: Y = XW                                              │
│                                                                     │
│   Column Parallel (for first linear in FFN):                        │
│   ═══════════════════════════════════════════                       │
│                                                                     │
│   X [batch, hidden]  ×  W [hidden, ffn_hidden]  =  Y [batch, ffn]   │
│                                                                     │
│   Split W by columns:                                               │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  GPU 0: X × W[:, :ffn/4]     → Y[:, :ffn/4]                 │   │
│   │  GPU 1: X × W[:, ffn/4:ffn/2] → Y[:, ffn/4:ffn/2]           │   │
│   │  GPU 2: X × W[:, ffn/2:3ffn/4] → Y[:, ffn/2:3ffn/4]         │   │
│   │  GPU 3: X × W[:, 3ffn/4:]    → Y[:, 3ffn/4:]                │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   No communication needed! Each GPU has partial output.             │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   Row Parallel (for second linear in FFN):                          │
│   ═════════════════════════════════════════                         │
│                                                                     │
│   Y [batch, ffn_hidden]  ×  W [ffn_hidden, hidden]  =  Z [batch, h] │
│                                                                     │
│   Split W by rows (and Y accordingly):                              │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  GPU 0: Y[:, :ffn/4] × W[:ffn/4, :]     → Z_partial_0       │   │
│   │  GPU 1: Y[:, ffn/4:ffn/2] × W[ffn/4:ffn/2, :] → Z_partial_1 │   │
│   │  GPU 2: Y[:, ffn/2:3ffn/4] × W[ffn/2:3ffn/4, :] → Z_partial_2│  │
│   │  GPU 3: Y[:, 3ffn/4:] × W[3ffn/4:, :]   → Z_partial_3       │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   AllReduce: Z = Z_partial_0 + Z_partial_1 + Z_partial_2 + Z_partial_3
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Attention Head Parallelism

```
┌─────────────────────────────────────────────────────────────────────┐
│              ATTENTION HEAD PARALLELISM                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Model: 32 attention heads, TP=4                                   │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  GPU 0: Heads 0-7    (8 heads)                              │   │
│   │  GPU 1: Heads 8-15   (8 heads)                              │   │
│   │  GPU 2: Heads 16-23  (8 heads)                              │   │
│   │  GPU 3: Heads 24-31  (8 heads)                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   Each GPU:                                                         │
│   • Computes Q, K, V for its heads                                  │
│   • Computes attention for its heads                                │
│   • Produces partial output                                         │
│                                                                     │
│   After attention: AllReduce to combine outputs                     │
│                                                                     │
│   Constraint: TP must divide num_attention_heads evenly!            │
│   • 32 heads: TP can be 1, 2, 4, 8, 16, 32                          │
│   • 64 heads: TP can be 1, 2, 4, 8, 16, 32, 64                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## NCCL Collective Operations

### Key Operations for Distributed Inference

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NCCL COLLECTIVES                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   AllReduce (used in Tensor Parallelism):                           │
│   ═══════════════════════════════════════                           │
│                                                                     │
│   Before:  GPU0=[1,2]  GPU1=[3,4]  GPU2=[5,6]  GPU3=[7,8]          │
│   After:   GPU0=[16,20] GPU1=[16,20] GPU2=[16,20] GPU3=[16,20]     │
│                                                                     │
│   All GPUs get the sum of all inputs                                │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   AllGather (used for gathering outputs):                           │
│   ═══════════════════════════════════════                           │
│                                                                     │
│   Before:  GPU0=[A]  GPU1=[B]  GPU2=[C]  GPU3=[D]                  │
│   After:   GPU0=[A,B,C,D] GPU1=[A,B,C,D] GPU2=[A,B,C,D] ...        │
│                                                                     │
│   All GPUs get concatenation of all inputs                          │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   Broadcast (used for input distribution):                          │
│   ════════════════════════════════════════                          │
│                                                                     │
│   Before:  GPU0=[data]  GPU1=[?]  GPU2=[?]  GPU3=[?]               │
│   After:   GPU0=[data]  GPU1=[data]  GPU2=[data]  GPU3=[data]      │
│                                                                     │
│   One GPU's data copied to all others                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Communication Overhead

```
┌─────────────────────────────────────────────────────────────────────┐
│              INTERCONNECT BANDWIDTH REQUIREMENTS                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   AllReduce data per layer (Llama 8B, TP=4):                        │
│   • Hidden size: 4096                                               │
│   • Batch × Seq: 32 × 1 = 32 (decode)                               │
│   • Data: 32 × 4096 × 2 bytes = 256 KB per AllReduce                │
│   • AllReduces per layer: 2 (attention + FFN)                       │
│   • Layers: 32                                                      │
│   • Total per token: 256 KB × 2 × 32 = 16 MB                        │
│                                                                     │
│   At 100 tokens/sec: 1.6 GB/s bandwidth needed                      │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   Interconnect Comparison:                                          │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ Interconnect      │ Bandwidth │ Latency │ Best For          │   │
│   │───────────────────┼───────────┼─────────┼───────────────────│   │
│   │ PCIe 4.0 x16      │ 32 GB/s   │ ~1 μs   │ TP=2 (limited)    │   │
│   │ PCIe 5.0 x16      │ 64 GB/s   │ ~1 μs   │ TP=2-4            │   │
│   │ NVLink 3 (A100)   │ 600 GB/s  │ ~0.5 μs │ TP=2-8            │   │
│   │ NVLink 4 (H100)   │ 900 GB/s  │ ~0.5 μs │ TP=2-8            │   │
│   │ InfiniBand HDR    │ 200 GB/s  │ ~1 μs   │ Multi-node        │   │
│   │ InfiniBand NDR    │ 400 GB/s  │ ~1 μs   │ Multi-node        │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   Rule: Use NVLink for TP within node, InfiniBand for PP across     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Multi-GPU vLLM Configuration

### Basic Multi-GPU Setup

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

    # Overhead
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

### Scaling Efficiency

```
┌─────────────────────────────────────────────────────────────────────┐
│                 TENSOR PARALLELISM SCALING                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Theoretical vs Actual Speedup (Llama 70B):                        │
│                                                                     │
│   TP Size    Theoretical    Actual (NVLink)    Actual (PCIe)        │
│   ────────────────────────────────────────────────────────────      │
│      1          1.0x            1.0x              1.0x              │
│      2          2.0x            1.9x              1.6x              │
│      4          4.0x            3.6x              2.5x              │
│      8          8.0x            6.5x              3.5x              │
│                                                                     │
│   Why less than linear?                                             │
│   • AllReduce communication overhead                                │
│   • Synchronization barriers                                        │
│   • Load imbalance                                                  │
│                                                                     │
│   NVLink is ESSENTIAL for efficient TP > 2                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Mixture of Experts (MoE) Inference

### MoE Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MOE INFERENCE                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Standard FFN:                                                     │
│   ═════════════                                                     │
│   Input → FFN → Output                                              │
│   All parameters used for every token                               │
│                                                                     │
│   MoE FFN:                                                          │
│   ════════                                                          │
│                    ┌─────────┐                                      │
│                    │ Router  │                                      │
│                    └────┬────┘                                      │
│                         │ Select top-k experts                      │
│         ┌───────────────┼───────────────┐                           │
│         ▼               ▼               ▼                           │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐                        │
│   │ Expert 0 │   │ Expert 1 │   │ Expert 7 │  (8 experts)           │
│   └──────────┘   └──────────┘   └──────────┘                        │
│         │               │               │                           │
│         └───────────────┼───────────────┘                           │
│                         ▼                                           │
│                   Weighted Sum                                      │
│                                                                     │
│   Mixtral 8x7B:                                                     │
│   • 8 experts, top-2 routing                                        │
│   • Total params: 46.7B                                             │
│   • Active params: ~13B per token                                   │
│   • Memory: Need all 46.7B in VRAM                                  │
│   • Compute: Only 13B worth of FLOPs                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### MoE Parallelism: Expert Parallelism

```
┌─────────────────────────────────────────────────────────────────────┐
│                 EXPERT PARALLELISM                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Mixtral 8x7B with EP=4:                                           │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  GPU 0: Experts 0, 1    (2 experts)                         │   │
│   │  GPU 1: Experts 2, 3    (2 experts)                         │   │
│   │  GPU 2: Experts 4, 5    (2 experts)                         │   │
│   │  GPU 3: Experts 6, 7    (2 experts)                         │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   Routing:                                                          │
│   1. All GPUs compute router logits                                 │
│   2. Determine which tokens go to which experts                     │
│   3. AllToAll: Send tokens to GPUs with their experts               │
│   4. Each GPU processes tokens for its experts                      │
│   5. AllToAll: Return processed tokens                              │
│                                                                     │
│   Challenge: Load imbalance if routing is skewed                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Wide Expert Parallelism (Wide-EP)

```
┌─────────────────────────────────────────────────────────────────────┐
│              WIDE EXPERT PARALLELISM (Anyscale)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Problem: Standard EP distributes experts across minimum GPUs      │
│   needed for memory. But MoE inference is memory-bandwidth bound    │
│   — each expert's weights must be read from HBM for every token.    │
│                                                                     │
│   Wide-EP Insight: Spread experts across MORE GPUs than needed      │
│   for memory, using tensor parallelism WITHIN each expert.          │
│                                                                     │
│   Standard EP (8 experts, 8 GPUs):                                  │
│   ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐│
│   │ E0   ││ E1   ││ E2   ││ E3   ││ E4   ││ E5   ││ E6   ││ E7   ││
│   │GPU 0 ││GPU 1 ││GPU 2 ││GPU 3 ││GPU 4 ││GPU 5 ││GPU 6 ││GPU 7 ││
│   └──────┘└──────┘└──────┘└──────┘└──────┘└──────┘└──────┘└──────┘│
│                                                                     │
│   Wide-EP (8 experts, 16 GPUs, TP=2 per expert):                    │
│   ┌────────────┐┌────────────┐┌────────────┐┌────────────┐         │
│   │  Expert 0  ││  Expert 1  ││  Expert 2  ││  Expert 3  │         │
│   │GPU 0│GPU 1 ││GPU 2│GPU 3 ││GPU 4│GPU 5 ││GPU 6│GPU 7 │         │
│   └────────────┘└────────────┘└────────────┘└────────────┘         │
│   ┌────────────┐┌────────────┐┌────────────┐┌────────────┐         │
│   │  Expert 4  ││  Expert 5  ││  Expert 6  ││  Expert 7  │         │
│   │GPU 8│GPU 9 ││GPU10│GPU11 ││GPU12│GPU13 ││GPU14│GPU15 │         │
│   └────────────┘└────────────┘└────────────┘└────────────┘         │
│                                                                     │
│   Why it works:                                                     │
│   • Each expert's weight read is parallelized across 2 GPUs         │
│   • 2× memory bandwidth per expert → 2× faster expert execution    │
│   • Solves the "double penalty": MoE has more total params AND      │
│     lower arithmetic intensity than dense models                    │
│   • Anyscale showed 1.5-2× throughput improvement over standard EP  │
│                                                                     │
│   Trade-off: Uses more GPUs, but each GPU is better utilized        │
│   because memory bandwidth (the bottleneck) is parallelized.        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### The MoE Double Penalty (When Dense Beats MoE)

```
┌─────────────────────────────────────────────────────────────────────┐
│           THE MOE DOUBLE PENALTY (arxiv:2603.08960)                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   MoE models suffer two compounding penalties at inference time:    │
│                                                                     │
│   Penalty 1: Routing Fragmentation                                  │
│   ═══════════════════════════════                                   │
│   • Tokens in a batch get routed to different experts               │
│   • Each expert processes a SUBSET of the batch                     │
│   • Effective batch size per expert = total_batch / num_experts     │
│   • Smaller batches → lower GPU compute utilization                 │
│   • Dense model processes ALL tokens together (full batch)          │
│                                                                     │
│   Penalty 2: KV Cache Memory Pressure                               │
│   ═══════════════════════════════════                               │
│   • MoE models have same attention layers as dense equivalents      │
│   • But total model memory is much larger (all experts in VRAM)     │
│   • Less VRAM available for KV cache                                │
│   • Fewer concurrent sequences → lower throughput                   │
│   • Example: Mixtral 8x7B uses 87 GB for weights vs 13 GB for      │
│     a dense 13B with equivalent active compute                      │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   When Dense Models Win:                                            │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ Scenario                    │ Winner │ Why                   │   │
│   │─────────────────────────────┼────────┼───────────────────────│   │
│   │ High batch, short context   │ Dense  │ Routing fragmentation │   │
│   │ Memory-constrained deploy   │ Dense  │ KV cache pressure     │   │
│   │ Latency-critical (small TP) │ Dense  │ AllToAll overhead     │   │
│   │ Low batch, long context     │ MoE    │ Less compute/token    │   │
│   │ Quality-per-FLOP priority   │ MoE    │ More params, same cost│   │
│   │ Wide-EP available (many GPU)│ MoE    │ Penalties mitigated   │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   Key insight: MoE advantage grows with GPU count. On few GPUs,     │
│   the double penalty can make a dense model faster at serving.      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### High-Performance AllToAll for Expert Routing

```
┌─────────────────────────────────────────────────────────────────────┐
│        PERPLEXITY: 10× FASTER ALL-TO-ALL ON AWS EFA                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   The AllToAll bottleneck:                                          │
│   • Expert parallelism requires AllToAll to route tokens            │
│   • Standard NCCL AllToAll on multi-node is slow                    │
│   • Becomes the dominant cost at scale (>16 GPUs)                   │
│                                                                     │
│   Perplexity's approach:                                            │
│   • Custom AllToAll kernel bypassing NCCL                           │
│   • Direct GPU-to-GPU communication over AWS EFA                    │
│   • Overlaps communication with expert computation                  │
│   • Result: ~10× faster AllToAll vs NCCL baseline                   │
│                                                                     │
│   Standard NCCL AllToAll:                                           │
│   ┌─────────┐                              ┌─────────┐             │
│   │  GPU 0  │──── NCCL (store-and-forward)──│  GPU 8  │             │
│   │ Node 0  │     through CPU/NIC stack     │ Node 1  │             │
│   └─────────┘                              └─────────┘             │
│   Latency: ~100-200 μs per AllToAll                                 │
│                                                                     │
│   Perplexity custom kernel:                                         │
│   ┌─────────┐                              ┌─────────┐             │
│   │  GPU 0  │──── Direct RDMA over EFA ─────│  GPU 8  │             │
│   │ Node 0  │     (GPU-initiated, async)    │ Node 1  │             │
│   └─────────┘                              └─────────┘             │
│   Latency: ~10-20 μs per AllToAll                                   │
│                                                                     │
│   Impact on MoE serving:                                            │
│   • Enables efficient EP across 64-128 GPUs                         │
│   • Makes Wide-EP practical at multi-node scale                     │
│   • Removes communication as the scaling bottleneck                 │
│   • Critical for models like DeepSeek-V3 (256 experts)              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Counter-Intuitive MoE Scaling: More GPUs = Higher Throughput AND Lower Latency

```
┌─────────────────────────────────────────────────────────────────────┐
│         MOE SCALING: THE OPPOSITE OF DENSE MODELS                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Dense model scaling (TP):                                         │
│   • More GPUs → lower latency (compute split)                       │
│   • More GPUs → LOWER throughput (communication overhead)           │
│   • Classic trade-off: latency vs throughput                        │
│                                                                     │
│   MoE with Wide-EP:                                                 │
│   • More GPUs → lower latency (faster expert execution)             │
│   • More GPUs → HIGHER throughput (simultaneously!)                 │
│   • No trade-off — both improve together                            │
│                                                                     │
│   Why? The MoE bottleneck is memory bandwidth, not compute:         │
│                                                                     │
│   Dense (TP=4 → TP=8):                                              │
│   ┌────────────────────────────────────────────────────────────┐    │
│   │ Compute: 2× faster (split across more GPUs)        ✓       │    │
│   │ Communication: 2× more AllReduce overhead          ✗       │    │
│   │ Batch capacity: Same (KV cache per GPU unchanged)          │    │
│   │ Net: Latency ↓, Throughput ↓                               │    │
│   └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│   MoE (EP=8 → Wide-EP=16):                                          │
│   ┌────────────────────────────────────────────────────────────┐    │
│   │ Memory BW: 2× per expert (weight reads parallelized) ✓     │    │
│   │ Communication: AllToAll (fast with custom kernels)   ~     │    │
│   │ Batch capacity: 2× more VRAM for KV cache           ✓     │    │
│   │ Net: Latency ↓, Throughput ↑                               │    │
│   └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│   Measured results (Mixtral 8x7B, Anyscale):                        │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ Config      │ Latency (TTFT) │ Throughput (tok/s) │ GPUs    │   │
│   │─────────────┼────────────────┼────────────────────┼─────────│   │
│   │ EP=8        │ 45 ms          │ 2,100              │ 8       │   │
│   │ Wide-EP=16  │ 28 ms          │ 3,800              │ 16      │   │
│   │ Wide-EP=32  │ 19 ms          │ 5,200              │ 32      │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   Implication: For MoE models, throwing more GPUs at the problem    │
│   improves BOTH latency and throughput — making the cost/perf       │
│   curve fundamentally different from dense model deployments.       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Choosing Parallelism Strategy

### Decision Guide

```
┌─────────────────────────────────────────────────────────────────────┐
│              PARALLELISM STRATEGY SELECTION                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Model fits on 1 GPU?                                              │
│   ├── Yes → Use single GPU (no parallelism needed)                  │
│   │         Consider DP for throughput scaling                      │
│   │                                                                 │
│   └── No → Model fits on 1 node (8 GPUs)?                           │
│            ├── Yes → Use Tensor Parallelism                         │
│            │         TP = min(8, model_size / gpu_memory)           │
│            │         Requires NVLink for efficiency                 │
│            │                                                        │
│            └── No → Need multi-node                                 │
│                     Use TP within node + PP across nodes            │
│                     Example: TP=8, PP=2 for 16 GPUs                 │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   Quick Reference:                                                  │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ Model Size │ Instance        │ Parallelism                  │   │
│   │────────────┼─────────────────┼──────────────────────────────│   │
│   │ ≤13B       │ g5.xlarge       │ None (single GPU)            │   │
│   │ 13-30B     │ g5.12xlarge     │ TP=2 or TP=4                 │   │
│   │ 30-70B     │ p4d.24xlarge    │ TP=4 or TP=8                 │   │
│   │ 70-140B    │ p4d.24xlarge    │ TP=8                         │   │
│   │ >140B      │ 2× p4d.24xlarge │ TP=8, PP=2                   │   │
│   │ 405B       │ 4× p5.48xlarge  │ TP=8, PP=4                   │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **Tensor Parallelism for latency** - Splits layers across GPUs, requires fast interconnect

2. **Pipeline Parallelism for throughput** - Splits layers sequentially, works across nodes

3. **NVLink is essential** - PCIe limits TP efficiency significantly

4. **TP must divide attention heads** - Common values: 2, 4, 8

5. **MoE needs all experts in memory** - But only activates subset per token

6. **Communication overhead limits scaling** - Expect 80-90% efficiency with NVLink

---

## Lab Preview: Tensor Parallelism

In Lab 6, you will:

- Deploy Llama 70B with TP=2, TP=4, TP=8
- Measure scaling efficiency
- Compare NVLink vs PCIe performance
- Calculate optimal TP for your workload

---

## References

1. Shoeybi et al. "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019)
2. Narayanan et al. "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (2021)
3. Fedus et al. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (2021)
4. NVIDIA NCCL Documentation
