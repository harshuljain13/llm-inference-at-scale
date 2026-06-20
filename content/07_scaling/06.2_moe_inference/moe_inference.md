# 6.2 Mixture-of-Experts Inference

From Module 06.1, you know tensor parallelism splits each layer's weight matrices across GPUs so that a single operation (matrix multiply, layer norm) executes in parallel. Every GPU participates in every token's computation. Mixture-of-Experts (MoE) adds an entirely new dimension to the parallelism story: not every GPU needs to compute every token. Instead, a learned router sends each token to a small subset of "expert" sub-networks, and only those experts fire. The result is a model with 10-100x more total parameters than it activates per token, achieving the capacity of a dense giant at the inference cost of a much smaller model.

This module covers the unique serving challenges MoE architectures introduce: expert parallelism, all-to-all communication, load balancing, memory budgeting, and how modern inference engines like vLLM and SGLang handle MoE routing in production.

---

## 1. Why MoE Changes the Inference Game

Dense transformer models have a simple cost model: every parameter participates in every forward pass. A 70B dense model activates 70B parameters per token. Double the parameters, double the compute.

MoE breaks this relationship. DeepSeek-V3 has 671 billion total parameters but activates only 37 billion per token. Mixtral 8x7B has roughly 47B total parameters but activates approximately 13B per token (2 of 8 experts per layer, plus shared attention). The implications for inference are profound:

- **Compute scales with active parameters, not total parameters.** A 671B MoE model requires roughly the same FLOPs per token as a 37B dense model. Latency per token is comparable to a much smaller model.
- **Memory scales with total parameters, not active parameters.** You must store all 671B parameters in GPU memory because any token might route to any expert. You cannot predict which experts will be needed at serving time.
- **Communication patterns are irregular.** In tensor parallelism, every GPU exchanges the same amount of data every layer. In expert parallelism, the amount each GPU sends and receives depends on which experts the router selects, creating dynamic, unpredictable communication.

This asymmetry between compute and memory is what makes MoE inference uniquely challenging. You get dense-model latency at giant-model capacity, but you pay giant-model memory cost and face communication patterns that dense models never encounter.

### The MoE Landscape in 2024-2025

| Model | Total Params | Active Params/Token | Num Experts | Top-k | Architecture |
|-------|-------------|--------------------:|------------:|------:|--------------|
| Mixtral 8x7B | ~47B | ~13B | 8 | 2 | Standard MoE |
| Mixtral 8x22B | ~141B | ~39B | 8 | 2 | Standard MoE |
| DeepSeek-V2 | 236B | 21B | 160 | 6 | Fine-grained MoE |
| DeepSeek-V3 | 671B | 37B | 256 | 8 | Fine-grained MoE |
| DBRX | 132B | 36B | 16 | 4 | Fine-grained MoE |
| Qwen2-MoE-57B | 57B | 14B | 64 | 8 | Fine-grained MoE |

The trend is clear: models are moving toward more experts with finer granularity. DeepSeek-V2 introduced "fine-grained experts" where each expert is smaller but more numerous, improving load balance and routing flexibility. DeepSeek-V3 pushed this further with 256 experts selecting 8 per token.

---

## 2. How MoE Works During Inference

A standard transformer layer in a dense model computes:

```
output = Attention(x) + FFN(x)
```

In an MoE layer, the FFN (feed-forward network) is replaced with multiple expert FFNs and a router:

```
router_logits = Router(x)                    # [batch_size, num_experts]
expert_weights = TopK(Softmax(router_logits), k)  # Select top-k experts
output = sum(expert_weights[i] * Expert_i(x) for i in selected_experts)
```

### The Router

The router is a simple linear layer that maps each token's hidden state to a score for each expert:

```python
# Router: hidden_dim -> num_experts
router_logits = torch.matmul(hidden_states, router_weight)  # [tokens, num_experts]
routing_weights = torch.softmax(router_logits, dim=-1)
top_k_weights, top_k_indices = torch.topk(routing_weights, k=top_k, dim=-1)
```

For DeepSeek-V3 with 256 experts and top-8 routing:
- Each token gets a 256-dimensional score vector
- The top 8 scores are selected
- Only those 8 expert FFNs compute on that token
- Results are weighted-summed by the softmax scores

### The Forward Pass

During inference, the MoE forward pass for a single layer proceeds as:

1. **Route**: Compute router logits, select top-k experts per token
2. **Dispatch**: Group tokens by their selected experts (some tokens go to expert 0, others to expert 5, etc.)
3. **Compute**: Each expert processes only its assigned tokens
4. **Combine**: Gather results back, weight by router scores, sum

The dispatch and combine steps are what create the communication challenge in distributed settings. In a single-GPU scenario, dispatch is just indexing. Across multiple GPUs, dispatch becomes an all-to-all collective.

### Shared vs. Expert Parameters

Not all parameters in an MoE model are expert-specific. The attention layers, embeddings, layer norms, and the router itself are "shared" parameters that process every token. Only the FFN experts are conditionally activated.

For DeepSeek-V3:
- Shared parameters (attention, embeddings, norms): ~37B (always active)
- Expert parameters (256 expert FFNs per MoE layer): ~634B (8/256 active per token)
- Total: ~671B

This means the attention computation is identical to a 37B dense model. The MoE mechanism only applies to the FFN portion of each layer.

---

## 3. Expert Parallelism: Distributing Experts Across GPUs

Tensor parallelism (Module 06.1) splits each weight matrix across GPUs. Expert parallelism (EP) takes a different approach: it assigns entire experts to different GPUs. Each GPU holds a complete subset of experts and processes only the tokens routed to those experts.

### EP Layout

With 256 experts and 8 GPUs using EP=8:
- GPU 0 holds experts 0-31
- GPU 1 holds experts 32-63
- GPU 2 holds experts 64-95
- ...
- GPU 7 holds experts 224-255

Each GPU stores the full weight matrices for its 32 experts. When a token is routed to expert 47, it must be sent to GPU 1 for computation.

### The All-to-All Communication Pattern

Expert parallelism requires an "all-to-all" collective communication operation. Unlike all-reduce (where every GPU sends the same data to every other GPU), all-to-all sends different data to different GPUs based on the routing decisions:

```
Before all-to-all (each GPU has its local tokens with routing decisions):
  GPU 0: token_0 -> expert 5 (GPU 0), token_1 -> expert 47 (GPU 1), token_2 -> expert 200 (GPU 6)
  GPU 1: token_3 -> expert 5 (GPU 0), token_4 -> expert 100 (GPU 3), token_5 -> expert 250 (GPU 7)
  ...

After all-to-all (each GPU has received all tokens destined for its experts):
  GPU 0: token_0, token_3 (both routed to experts 0-31)
  GPU 1: token_1 (routed to experts 32-63)
  ...
```

This requires two all-to-all operations per MoE layer:
1. **Dispatch all-to-all**: Send tokens to the GPUs holding their selected experts
2. **Combine all-to-all**: Send computed results back to the original GPUs

### Communication Volume

The total bytes transferred in the dispatch all-to-all depends on the hidden dimension and number of tokens:

```
bytes_per_token = hidden_dim * dtype_size
total_dispatch_bytes = num_tokens * top_k * bytes_per_token
```

For DeepSeek-V3 with hidden_dim=7168, top-k=8, bfloat16, and a batch of 1024 tokens:
```
dispatch_volume = 1024 * 8 * 7168 * 2 = 117 MB per MoE layer
```

With 60 MoE layers in DeepSeek-V3, that is 7 GB of all-to-all traffic per forward pass for a single batch. This is why high-bandwidth interconnects (NVLink, InfiniBand) are essential for efficient MoE serving.

### EP vs TP: When to Use Each

| Dimension | Tensor Parallelism | Expert Parallelism |
|-----------|-------------------|-------------------|
| Communication pattern | All-reduce (symmetric) | All-to-all (asymmetric) |
| Communication per layer | 2 * hidden_dim * batch * dtype | top_k * hidden_dim * batch * dtype |
| Load balance | Perfect (same work per GPU) | Imperfect (depends on routing) |
| Memory per GPU | total_params / TP_degree | shared_params + expert_params / EP_degree |
| Scales with | Matrix dimensions | Number of experts |
| Best for | Dense models, small expert count | Many experts, fine-grained MoE |

For Mixtral 8x7B with only 8 experts, EP=8 puts one expert per GPU which is clean but wastes GPUs on the shared attention. For DeepSeek-V3 with 256 experts, EP is the natural fit because you can distribute 256 experts across 8-64 GPUs efficiently.

---

## 4. The Communication Problem: Why Interconnect Bandwidth Matters

The all-to-all communication in expert parallelism creates a fundamentally different performance profile than tensor parallelism. In TP, communication is predictable: every all-reduce transfers exactly `2 * hidden_dim * batch_size * dtype_size` bytes. In EP, communication is data-dependent: the volume and distribution depend on which experts the router selects.

### Bandwidth Requirements

Consider DeepSeek-V3 with EP=8 on a node with 8 GPUs:

- **NVLink (900 GB/s bidirectional per GPU pair in H100 SXM)**: The 117 MB per-layer dispatch completes in ~0.13 ms. Across 60 MoE layers: ~8 ms total communication time.
- **PCIe Gen5 (64 GB/s)**: Same dispatch takes ~1.8 ms per layer. Across 60 layers: ~110 ms. This is 14x slower.
- **InfiniBand HDR (200 Gbps = 25 GB/s)**: Cross-node dispatch takes ~4.7 ms per layer. Across 60 layers: ~280 ms. Tolerable only if experts are distributed within a single node.

The implication is clear: **MoE inference with expert parallelism is only practical on systems with high-bandwidth interconnects.** Running EP across PCIe or cross-node InfiniBand introduces communication overhead that can dominate the total latency.

### Latency Hiding Strategies

Several techniques reduce the impact of all-to-all latency:

**Overlapping communication with computation:**
While one MoE layer's combine all-to-all sends results back, the next layer's attention computation can proceed. This pipelines the communication behind useful work.

**Hierarchical all-to-all:**
On multi-node systems, perform intra-node all-to-all first (fast NVLink), then inter-node all-to-all (slower InfiniBand). This reduces the amount of data crossing the slow link.

**Expert replication:**
Place hot experts (those frequently selected) on multiple GPUs. Tokens routed to a replicated expert go to the nearest copy, reducing cross-GPU traffic. The cost is additional memory usage.

**Token dropping:**
If too many tokens route to a single expert (exceeding its capacity), excess tokens are "dropped" and processed by a shared residual path. This caps the maximum all-to-all volume at the expense of some quality degradation.

---

## 5. Load Balancing: The Expert Hotspot Problem

If the router learned to always send tokens to the same 2-3 experts, those experts' GPUs would be overloaded while others sit idle. Load imbalance is the primary operational challenge in MoE serving.

### Why Imbalance Happens

Language has structure. Tokens about code might consistently route to one expert. Tokens about math might route to another. During serving, a batch of similar tokens (e.g., a long code file) can create extreme routing skew where 80% of tokens go to 10% of experts.

### Auxiliary Loss During Training

The standard mitigation is an auxiliary load-balancing loss added during training (Fedus et al., 2022, "Switch Transformers"):

```python
# f_i = fraction of tokens dispatched to expert i
# P_i = fraction of router probability assigned to expert i
aux_loss = alpha * num_experts * sum(f_i * P_i for i in range(num_experts))
```

This loss encourages the router to spread tokens uniformly across experts. The coefficient `alpha` controls how aggressively balance is enforced. Too high and the model sacrifices quality for balance. Too low and routing becomes skewed.

DeepSeek-V3 uses a more sophisticated approach: they add per-expert bias terms that are adjusted during training to maintain balance without an explicit auxiliary loss, finding that removing the auxiliary loss improves model quality while maintaining acceptable balance.

### Capacity Factor During Serving

Even with balanced training, serving-time token distributions differ from training distributions. The "capacity factor" C defines how many tokens each expert can accept:

```
max_tokens_per_expert = C * (total_tokens * top_k / num_experts)
```

With C=1.0, each expert can handle exactly its fair share. With C=1.25, each expert gets 25% extra buffer. Tokens exceeding an expert's capacity are either dropped or sent to a fallback expert.

In practice:
- **C=1.0**: Strict capacity. Some tokens dropped. Minimum memory per expert.
- **C=1.25**: Small buffer. Rarely drops tokens. 25% more activation memory.
- **C=2.0**: Large buffer. Almost never drops. 2x activation memory per expert.

### Runtime Load Monitoring

Production MoE systems track expert utilization in real-time:

```python
# Per-expert token count over the last N batches
expert_load = torch.zeros(num_experts)
for batch in recent_batches:
    for token in batch:
        for expert_id in token.selected_experts:
            expert_load[expert_id] += 1

# Imbalance ratio: max_load / mean_load
imbalance_ratio = expert_load.max() / expert_load.mean()
# Healthy: < 1.5. Concerning: 1.5-3.0. Critical: > 3.0
```

When imbalance exceeds thresholds, operators can:
1. Increase the capacity factor (accepts more tokens per expert, costs memory)
2. Enable expert replication for hot experts
3. Adjust batch composition to mix diverse prompts

---

## 6. Combined Parallelism: Composing TP, EP, and DP

Real MoE deployments combine multiple parallelism strategies. The composition depends on model size, expert count, and hardware topology.

### DeepSeek-V3 Deployment Configuration

DeepSeek reports serving V3 with the following parallelism composition:

- **TP=1 within each expert**: Each expert is small enough (2.6B params) to fit on a single GPU. No need to split expert weights.
- **EP=8 across GPUs within a node**: 256 experts distributed across 8 GPUs. Each GPU holds 32 experts.
- **DP=N across nodes**: Multiple nodes handle different requests in parallel. Each node is a complete EP group that can independently serve any request.

This composition works because:
1. DeepSeek-V3's fine-grained experts are individually small (hidden_dim=7168, intermediate=2304 per expert). One expert is ~33M parameters = 66 MB in bfloat16. 32 experts per GPU = ~2.1 GB of expert weights.
2. The shared attention layers (~37B) are distributed with TP across the same 8 GPUs.
3. All-to-all communication stays within a single NVLink-connected node (fast).
4. Scaling throughput means adding more DP replicas (more nodes).

### Mixtral 8x7B Deployment

Mixtral has fewer, larger experts. A typical deployment:

- **TP=2 within each expert** (optional): Each expert is ~7B parameters = 14 GB in bfloat16. With TP=2, each half-expert is 7 GB, fitting alongside shared params.
- **EP=4 across GPUs**: 8 experts across 4 GPUs. Each GPU holds 2 experts.
- **Total GPUs**: 4-8 (EP=4 with optional TP=2 within experts doubles to 8 GPUs)

Alternatively, with large-memory GPUs (H100 80GB):
- **TP=1, EP=2**: Each GPU holds 4 experts (~28 GB) plus shared attention (~14 GB) = 42 GB. Fits in 80 GB with room for KV cache and activations.

### Parallelism Composition Rules

The total GPU count for an MoE model equals:

```
total_GPUs = TP * EP * DP
```

Where:
- **TP** splits weight matrices within a single expert or the shared attention
- **EP** distributes experts across GPUs
- **DP** replicates the entire model for throughput

Within a single node (8 GPUs), you must fit TP * EP ≤ 8. Across nodes, DP provides additional replicas.

### Decision Framework for Parallelism Choice

```
IF expert_size > single_GPU_memory:
    Use TP within experts (TP = expert_params / GPU_memory)
    
IF num_experts > 1 AND total_expert_params > single_GPU_memory:
    Use EP (EP = num_experts / experts_per_GPU)
    Constrain EP to NVLink domain for fast all-to-all
    
IF you need more throughput:
    Add DP replicas (DP = target_throughput / per_replica_throughput)
```

For most fine-grained MoE models (DeepSeek-V2/V3, Qwen2-MoE):
- Individual experts are small → TP=1 within experts
- Many experts total → EP=8 (one node)
- Throughput scaling → DP=N

For coarse-grained MoE models (Mixtral 8x7B):
- Individual experts are large → TP=1 or TP=2 within experts
- Few experts → EP=4 or EP=8
- Lower communication overhead (fewer experts = fewer routing decisions)

---

## 7. Memory Budget: The MoE Memory Paradox

The central confusion in MoE memory planning: "If only 37B parameters are active per token, why can't I fit DeepSeek-V3 on 37B-model hardware?"

### Why All Parameters Must Be Resident

The router's decision is made at runtime, per token. Any token might route to any expert. You cannot know in advance which experts will be needed for the next request. Therefore, all expert weights must be loaded in GPU memory before inference begins.

```
DeepSeek-V3 memory requirement:
  Shared params (attention, embeddings, norms): ~37B * 2 bytes = 74 GB
  Expert params (256 experts * FFN each):       ~634B * 2 bytes = 1268 GB
  Total model weights:                          ~1342 GB (bfloat16)
  
  Minimum GPUs (H100 80GB):
    1342 / 80 = 17 GPUs (just for weights, no KV cache or activations)
    Practical: 32-64 GPUs to leave room for KV cache and activation memory
```

Compare to a hypothetical 37B dense model:
```
  37B * 2 bytes = 74 GB → fits on a single H100 80GB
```

The MoE model uses the same compute per token as the 37B model but requires 18x more memory. This is the MoE memory paradox.

### Memory Breakdown Per GPU

For a DeepSeek-V3 deployment with EP=8 on 8x H100 80GB:

```
Per GPU:
  Shared attention weights (TP=8): 37B * 2 / 8 = 9.25 GB
  Expert weights (EP=8):           634B * 2 / 8 = 158.5 GB  ← DOES NOT FIT
```

This confirms that 8 GPUs are insufficient for DeepSeek-V3 in bfloat16. Options:

1. **More GPUs**: EP=32 → expert weights per GPU = 634B * 2 / 32 = 39.6 GB. Total per GPU with shared params (TP=32): 39.6 + 2.3 = 41.9 GB. Fits in 80 GB.
2. **Quantization**: INT4 weights reduce expert memory by 4x. 634B * 0.5 / 8 = 39.6 GB per GPU. Now 8 GPUs work with quantized experts.
3. **Expert offloading**: Keep only hot experts in GPU memory, page cold experts from CPU/NVMe. Adds latency for cache misses.

### Expert Offloading: Trading Latency for Memory

Since only top-k experts fire per token, you could store cold experts on CPU memory and load them on demand:

```
GPU memory: top-32 most-used experts (always resident)
CPU memory: remaining 224 experts (loaded on demand)

Cache hit (token routes to resident expert): 0 ms overhead
Cache miss (token routes to offloaded expert): 0.5-2 ms per miss (PCIe transfer)
```

This works for interactive serving where latency budgets are 50-100 ms per token. A 2 ms miss on 1-2 experts per layer, across 60 layers, adds 60-120 ms, which is too slow for real-time.

Expert offloading is practical only when:
- Batch sizes are large enough to amortize the transfer
- Expert access patterns are predictable (pre-fetch upcoming experts)
- Latency requirements are relaxed (batch processing, not interactive chat)

### Activation Memory in MoE

Beyond weights, MoE layers require activation memory for:
- Router logits: `batch * num_experts * dtype` per layer
- Dispatched tokens: `batch * top_k * hidden_dim * dtype` per expert group
- Expert intermediate activations: `batch * top_k * intermediate_dim * dtype`

For DeepSeek-V3 with batch=1024, hidden=7168, intermediate=2304:
```
Router logits: 1024 * 256 * 2 = 0.5 MB per layer
Dispatched tokens: 1024 * 8 * 7168 * 2 = 117 MB per layer
Expert intermediates: 1024 * 8 * 2304 * 2 = 37.7 MB per layer
Total per MoE layer: ~155 MB
Across 60 layers: ~9.3 GB
```

This is modest compared to the 1342 GB of weights, but adds up when combined with KV cache memory.

---

## 8. KV Cache in MoE Models

MoE models share the same KV cache characteristics as dense models because the attention layers are shared (not expert-gated). The KV cache size depends on the attention architecture, not the MoE structure.

### DeepSeek-V3's Multi-head Latent Attention (MLA)

DeepSeek-V3 uses Multi-head Latent Attention (covered in Module 02.3) which dramatically reduces KV cache size by compressing keys and values into a low-rank latent space:

```
Standard GQA KV cache per token:
  2 * num_kv_heads * head_dim * num_layers * dtype
  = 2 * 8 * 128 * 60 * 2 = 245 KB per token

DeepSeek-V3 MLA KV cache per token:
  (compressed_kv_dim + rope_dim) * num_layers * dtype
  = (512 + 64) * 60 * 2 = 69 KB per token
```

The MLA design reduces KV cache by ~3.5x compared to equivalent GQA, which is critical when the model weights already consume most of the GPU memory. With 1342 GB used for weights across the cluster, every byte saved on KV cache translates directly to higher batch sizes and throughput.

### KV Cache Budget Planning

For a DeepSeek-V3 deployment on 32x H100 80GB with INT4 expert quantization:

```
Per GPU budget (80 GB):
  Expert weights (INT4): 634B * 0.5 / 32 = 9.9 GB
  Shared weights (BF16, TP=32): 37B * 2 / 32 = 2.3 GB
  Activation memory: ~0.3 GB
  KV cache budget: 80 - 9.9 - 2.3 - 0.3 = 67.5 GB available

  Tokens supported at 69 KB/token: 67.5 GB / 69 KB ≈ 1,000,000 tokens
  At 4096 context length: ~244 concurrent sequences per GPU
  Total cluster capacity: 244 * 32 = 7,800 concurrent sequences
```

This is generous because MLA keeps KV cache small. Without MLA (standard GQA at 245 KB/token), the same budget supports only ~275,000 tokens total or ~67 sequences per GPU.

---

## 9. Inference Engine Support: vLLM and SGLang

Modern inference engines have added MoE-specific optimizations for routing, memory management, and load balancing.

### vLLM MoE Support

vLLM handles MoE models with the following key mechanisms:

**Expert-aware tensor parallelism:**
vLLM distributes experts across GPUs using EP within a TP group. For Mixtral 8x7B with TP=2:
- Each TP group of 2 GPUs holds all 8 experts, split by TP
- Alternative: EP=2 within TP=4, placing 4 experts per 2-GPU subgroup

**Fused MoE kernels:**
vLLM uses custom Triton kernels that fuse the routing, dispatch, expert compute, and combine into a single kernel launch. This eliminates intermediate memory allocations and reduces kernel launch overhead:

```python
# vLLM's fused MoE implementation (simplified)
# Single kernel handles: route -> permute -> expert matmul -> unpermute -> combine
from vllm.model_executor.layers.fused_moe import fused_moe

output = fused_moe(
    hidden_states,     # [num_tokens, hidden_dim]
    w1,               # [num_experts, intermediate_dim, hidden_dim]
    w2,               # [num_experts, hidden_dim, intermediate_dim]
    gating_output,    # [num_tokens, num_experts] (router logits)
    topk=2,
    renormalize=True,
)
```

**Expert quantization:**
vLLM supports INT8 and INT4 quantization specifically for expert weights using GPTQ, AWQ, or FP8 formats. Since experts are independent, each can be quantized separately without cross-expert calibration issues.

### SGLang MoE Support

SGLang (used heavily by DeepSeek) optimizes MoE serving with:

**Expert parallelism-first design:**
SGLang's architecture treats EP as a first-class parallelism mode, not an afterthought on top of TP. The scheduler is aware of expert placement and routes requests to minimize cross-GPU traffic.

**Expert-aware batching:**
SGLang groups tokens with similar routing patterns into the same micro-batch. If multiple requests route heavily to experts 0-31 (GPU 0), SGLang schedules them together to maximize data locality and minimize all-to-all volume.

**Overlapped communication:**
SGLang overlaps the combine all-to-all of layer N with the attention computation of layer N+1. Since attention uses the shared parameters (present on all GPUs), it can proceed while expert results are still in transit.

### Performance Comparison

For Mixtral 8x22B on 8x H100 SXM (NVLink):

| Engine | Prefill (tokens/s) | Decode (tokens/s) | Strategy |
|--------|-------------------:|------------------:|----------|
| vLLM (TP=8) | ~8,000 | ~2,500 | TP-only, fused kernels |
| SGLang (EP=4, TP=2) | ~9,200 | ~2,800 | EP+TP, overlapped comm |
| TensorRT-LLM (EP=8) | ~10,500 | ~3,100 | EP, custom NCCL kernels |

The performance differences come primarily from communication optimization. TensorRT-LLM's custom NCCL all-to-all kernels are highly tuned for specific GPU topologies.

---

## 10. Expert Caching and Speculative Expert Loading

A newer optimization treats expert weights like a cache: keep frequently-accessed experts in fast memory and load rare experts on demand.

### Expert Popularity Distribution

In practice, expert utilization follows a power law. Fedus et al. (2022) observed that even with load balancing, some experts receive 2-3x more tokens than others. This creates an opportunity:

```
Expert popularity (measured across 1M tokens of mixed workload):
  Top 10% experts: handle 25% of all token-expert pairs
  Top 50% experts: handle 80% of all token-expert pairs
  Bottom 20% experts: handle only 5% of all token-expert pairs
```

### Speculative Expert Pre-Loading

The router's decision depends on the hidden state, which is known one layer ahead. Some systems exploit this:

1. As layer N computes attention, predict which experts layer N+1 will need
2. Pre-fetch those expert weights into GPU cache before the MoE computation starts
3. If prediction is accurate (70-90% for adjacent layers), cold expert loads are hidden behind computation

This is analogous to hardware prefetching in CPUs but applied at the model layer level.

### Expert-Level KV-Aware Scheduling

An advanced strategy combines expert caching with request scheduling:

```python
# Pseudocode for expert-aware scheduling
def schedule_requests(pending_requests, gpu_expert_cache):
    """Prioritize requests whose experts are already cached."""
    scored = []
    for req in pending_requests:
        # Predict which experts this request will use
        # (based on prompt embedding similarity to expert prototypes)
        predicted_experts = predict_expert_routing(req.prompt)
        cache_hit_rate = len(predicted_experts & gpu_expert_cache) / len(predicted_experts)
        scored.append((cache_hit_rate, req))
    
    # Serve requests with highest cache hits first
    return sorted(scored, key=lambda x: -x[0])
```

---

## 11. Practical Deployment Considerations

### Sizing Hardware for MoE Models

Step-by-step sizing for DeepSeek-V3 production serving:

**Step 1: Minimum GPUs for weights**
```
BF16: 671B * 2 bytes = 1342 GB / 80 GB per H100 = 17 GPUs minimum
INT4: 671B * 0.5 bytes = 335 GB / 80 GB per H100 = 5 GPUs minimum (tight)
FP8:  671B * 1 byte = 671 GB / 80 GB per H100 = 9 GPUs minimum
```

**Step 2: Add KV cache budget**
Target: 64 concurrent sequences, 8192 context length
```
KV per sequence (MLA): 69 KB * 8192 = 565 MB
Total KV: 64 * 565 MB = 36 GB (distributed across GPUs)
```

**Step 3: Add activation memory**
```
~10 GB total (distributed across GPUs)
```

**Step 4: Choose GPU count**
For FP8 quantization targeting 64 concurrent sequences:
```
Total memory needed: 671 + 36 + 10 = 717 GB
GPUs at 80 GB: 717 / 80 = 9 → round up to 16 (2 nodes) or 8 (1 node tight)
With 70% utilization target: 717 / (80 * 0.7) = 13 → 16 GPUs (2 nodes)
```

**Step 5: Validate interconnect**
16 GPUs across 2 nodes. Intra-node: NVLink (fast). Inter-node: InfiniBand (slower).
EP should stay within a node. Use EP=8 per node, DP=2 across nodes.

### Monitoring MoE-Specific Metrics

Beyond standard serving metrics (TTFT, TPOT, throughput), MoE deployments require:

| Metric | Description | Healthy Range |
|--------|-------------|--------------|
| Expert imbalance ratio | max_load / mean_load | < 1.5 |
| Token drop rate | Tokens exceeding capacity factor | < 0.1% |
| All-to-all latency | Time spent in dispatch + combine | < 20% of layer time |
| Expert cache hit rate | Tokens routed to resident experts | > 95% (if using offloading) |
| Per-expert queue depth | Tokens waiting for a specific expert | < 2x mean |

### Cost-Performance Tradeoffs

MoE models offer a favorable cost profile when measured by quality-per-FLOP:

```
Dense 70B model:
  GPU cost (8x H100): ~$25/hour (cloud)
  Throughput: ~3000 tokens/s
  Quality: equivalent to 70B parameter capacity
  Cost per 1M tokens: $2.31

DeepSeek-V3 671B (MoE):
  GPU cost (32x H100): ~$100/hour (cloud)
  Throughput: ~4000 tokens/s (more params active, but larger model capacity)
  Quality: equivalent to 671B parameter capacity (10x the dense model)
  Cost per 1M tokens: $6.94

Quality-adjusted cost:
  Dense 70B: $2.31/M tokens at 70B-quality
  DeepSeek-V3: $6.94/M tokens at 671B-quality
  Per unit of model capacity: MoE is 3.3x cheaper
```

The MoE model costs 3x more in absolute terms but delivers 10x more model capacity, making it 3.3x more cost-effective per unit of quality.

---

## 12. Mental Model: The MoE Inference Summary

Think of MoE inference as a restaurant with 256 specialized chefs (experts). Each order (token) consults a maître d' (router) who assigns it to 8 chefs. The chefs work independently. The challenge is:

- **Memory**: You must pay rent for all 256 kitchens (all expert weights resident) even though only 8 are cooking at any moment.
- **Communication**: Orders must physically travel to whichever kitchen the maître d' selected (all-to-all dispatch). Closer kitchens (same NVLink node) are reached faster than distant ones (cross-node InfiniBand).
- **Load balance**: If the maître d' sends all orders to the pasta station, one chef is overwhelmed. Training teaches the maître d' to spread orders evenly, but serving-time distributions may differ from training data.
- **Parallelism composition**: You can split one kitchen across two counters (TP within expert), distribute kitchens across floors (EP), or open multiple restaurant locations (DP).

The one-line summary:

> **MoE gives you a 10x bigger model at 1x compute cost per token, but you pay 10x memory cost and face irregular all-to-all communication that demands high-bandwidth interconnects.**

---

## References

1. Fedus, W., Zoph, B., & Shazeer, N. (2022). "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." *JMLR*, 23(120), 1-39.
2. Jiang, A. Q., et al. (2024). "Mixtral of Experts." arXiv:2401.04088.
3. DeepSeek-AI. (2024). "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." arXiv:2405.04434.
4. DeepSeek-AI. (2024). "DeepSeek-V3 Technical Report." arXiv:2412.19437.
5. Lepikhin, D., et al. (2021). "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding." *ICLR 2021*.
6. Shazeer, N., et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." *ICLR 2017*.
7. Kim, S., et al. (2024). "vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention." (MoE extensions in vLLM v0.4+).
8. Zheng, L., et al. (2024). "SGLang: Efficient Execution of Structured Language Model Programs." (MoE-specific scheduling optimizations).

---

## What Comes Next

Module 06.3 covers **Pipeline Parallelism**, which splits model layers across GPUs sequentially. When combined with expert parallelism, pipeline parallelism assigns different layer groups to different pipeline stages, each of which internally uses EP for its MoE layers. This three-dimensional parallelism (TP + EP + PP) is how the largest MoE models (1T+ parameters) are served in production.
