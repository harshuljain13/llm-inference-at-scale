# Multi-Region Inference and KV Cache Locality

> **Mental model**: At global scale, KV cache becomes a distributed systems problem, not just a memory management problem. The challenge shifts from "how do I fit KV cache in one GPU" to "how do I get the right KV cache to the right GPU in the right region before the user notices."

## Why Single-Region Inference Breaks at Global Scale

Consider a user in Tokyo sending a request to your LLM inference service. If your GPUs live exclusively in us-east-1, that request travels approximately 11,000 km across the Pacific Ocean. Physics imposes a hard floor: the speed of light in fiber optic cable is roughly 200,000 km/s, which means the absolute minimum round-trip latency is ~110ms. In practice, with routing hops, TLS handshakes, and protocol overhead, you see 150-250ms before the first token even begins generation.

This matters enormously for interactive applications. A chatbot that takes 300ms to produce its first token feels sluggish. A code completion system with 400ms time-to-first-token (TTFT) disrupts the developer's flow. Multiply this across billions of daily requests from users distributed globally, and single-region inference becomes untenable for three fundamental reasons.

### Latency Physics

The speed of light is not negotiable. No amount of software optimization can reduce the propagation delay between continents. The table below shows minimum theoretical round-trip times between major cloud regions:

| Route | Distance (km) | Minimum RTT (ms) | Practical RTT (ms) |
|-------|---------------|-------------------|---------------------|
| us-east-1 → eu-west-1 | 5,500 | 55 | 80-120 |
| us-east-1 → ap-northeast-1 | 11,000 | 110 | 150-220 |
| eu-west-1 → ap-southeast-1 | 10,500 | 105 | 140-200 |
| us-west-2 → ap-south-1 | 13,000 | 130 | 170-250 |

These numbers represent floors. Real latency includes DNS resolution, TCP connection establishment (1.5 RTT for TLS 1.3), load balancer processing, and queuing delays. For streaming LLM responses where the user perceives each token individually, the initial connection latency dominates user experience.

### Compliance and Data Sovereignty

The European Union's GDPR, China's PIPL, India's DPDPA, and dozens of other regulatory frameworks impose constraints on where user data can be processed. When a user's prompt contains personal information (medical queries, financial data, legal questions), that data may be legally required to stay within specific geographic boundaries. Running all inference in a single region means either violating these regulations or refusing to serve users in regulated jurisdictions.

### Availability and Blast Radius

A single-region deployment creates a single point of failure. AWS us-east-1 has experienced multiple significant outages (2017, 2021, 2023). When your only inference cluster goes down, every user globally loses service simultaneously. Multi-region deployment limits the blast radius: a failure in one region affects only users routed to that region, while others continue uninterrupted.

**The implication is clear**: production LLM inference at global scale requires multi-region deployment. But this creates a problem that does not exist in traditional web services, a problem unique to the stateful nature of autoregressive generation.

## Back-Reference: Building on Prior Modules

From Module 5.1, you know that tensor parallelism requires NVLink bandwidth (900 GB/s on H100 SXM) to shard attention heads across GPUs within a single node. The key insight was that TP communication happens at microsecond granularity within a forward pass, making it impossible to distribute across regions where latencies are measured in milliseconds.

From Module 6.4, you learned that disaggregated serving separates the prefill phase (compute-bound, processes the entire prompt) from the decode phase (memory-bound, generates one token at a time). Systems like DistServe and Splitwise exploit this separation to use different hardware optimally: high-FLOPS GPUs for prefill, high-bandwidth-memory GPUs for decode.

From Module 3.4, you understand that KV cache grows linearly with sequence length and model dimension. For a 70B parameter model with GQA (8 KV heads), each token adds roughly 640 KB to the KV cache. A 4096-token context accumulates ~2.5 GB of KV state that must persist across all decode steps.

These three facts converge into the central tension of multi-region inference: the KV cache is large, it lives in GPU memory, and it must be present wherever decode happens. When the user who created that KV cache is in a different region than the GPU holding it, what do you do?

## The Fundamental Tension: KV Cache as Distributed State

In traditional stateless web services, multi-region deployment is straightforward. You replicate your application servers, put a global load balancer in front, and route users to the nearest region. Each request is independent; there is no state to synchronize.

LLM inference is fundamentally different because it is stateful within a conversation. The first user message triggers prefill, which produces a KV cache. Every subsequent token generation (decode) reads from and appends to this KV cache. The KV cache is not optional metadata; it is the core computational state without which generation cannot continue.

This creates a distributed systems problem with no perfect solution:

```
User in Tokyo → Request arrives at ap-northeast-1
                 ↓
           GPU in ap-northeast-1 runs prefill
                 ↓
           KV cache now lives in ap-northeast-1 GPU memory
                 ↓
    [User's next message in the same conversation]
                 ↓
           Must route back to the SAME GPU (or transfer KV cache)
```

The KV cache pins a conversation to a specific GPU in a specific region. This is "session affinity" taken to an extreme: not just sticky to a server, but sticky to a particular GPU's HBM. Breaking this affinity requires either discarding the KV cache (re-prefilling from scratch) or moving it (transferring gigabytes across a network).

Neither option is free. Re-prefilling a 4096-token prompt on an H100 takes approximately 200-400ms depending on the model. Transferring 2.5 GB across regions takes 50-500ms depending on available bandwidth. The right choice depends on the specific context: prompt length, available bandwidth, GPU utilization, and user latency tolerance.

## Option Space: Three Strategies for Multi-Region KV Locality

When a user's request could be served by multiple regions, you have three fundamental strategies. Each makes a different tradeoff between latency, cost, and complexity.

### Strategy A: Route to Nearest Region, Fresh Prefill

The simplest approach: always route the user to their geographically nearest region and run prefill from scratch for every conversation turn.

**How it works:**
1. User in Frankfurt sends a message
2. Global load balancer routes to eu-west-1 (nearest region with GPU capacity)
3. The full conversation history (all prior messages) is sent as the prompt
4. Prefill runs on the complete prompt, generating a fresh KV cache
5. Decode produces the response tokens
6. KV cache is discarded after the response completes (or kept for immediate follow-ups within the same region)

**Advantages:**
- Zero cross-region data transfer
- No distributed state to manage
- Simple routing logic
- Each region operates independently

**Disadvantages:**
- Prefill cost grows with conversation length (O(n²) in attention computation)
- A 10-turn conversation with 8K total tokens requires re-processing all 8K tokens on every turn
- GPU compute wasted on redundant prefill work
- TTFT increases with conversation length

**When this wins:** Short conversations (< 2K tokens), stateless single-turn queries (search, classification), and cases where inter-region bandwidth is limited or expensive.

### Strategy B: Transfer KV Cache Across Regions

When the user moves regions (or when load balancing shifts their traffic), transfer the existing KV cache to the new region's GPU rather than recomputing it.

**How it works:**
1. User's conversation has existing KV cache in us-east-1 (accumulated over several turns, 4K tokens, ~2.5 GB)
2. User's next request arrives and is routed to eu-west-1 (perhaps they traveled, or us-east-1 is congested)
3. System detects the KV cache miss in eu-west-1
4. Initiates transfer of 2.5 GB from us-east-1 GPU → eu-west-1 GPU
5. Once transfer completes, decode proceeds in eu-west-1 with the existing KV cache
6. New tokens are appended to the now-local KV cache

**Advantages:**
- Avoids redundant prefill computation
- Preserves full conversation context without recomputation
- TTFT is transfer_time + first_decode_step (can be faster than full re-prefill for long contexts)

**Disadvantages:**
- Requires high-bandwidth inter-region connectivity
- Complex orchestration (source GPU must hold KV cache until transfer completes)
- Transfer introduces latency that does not exist with local prefill for short prompts
- Failure handling is complex (what if transfer fails mid-way?)

**When this wins:** Long conversations (> 8K tokens) where re-prefill would take longer than transfer, and where inter-region bandwidth is abundant.

### Strategy C: Replicate Popular Prefixes Globally

For prompts that are shared across many users (system prompts, few-shot examples, RAG templates), pre-compute the KV cache and replicate it to all regions.

**How it works:**
1. Identify high-frequency prompt prefixes (e.g., ChatGPT's system message, a company's standard RAG template)
2. Run prefill once on these prefixes, producing a "prefix KV cache"
3. Distribute the prefix KV cache to GPU memory in every active region
4. When a user request arrives with a known prefix, load the pre-computed KV cache and only run prefill on the user-specific suffix
5. This is "prompt caching" applied globally

**Advantages:**
- Dramatic prefill savings for common prefixes (often 50-80% of prompt tokens are system/template)
- No per-request cross-region transfer needed
- Scales naturally with user base (cost is amortized)

**Disadvantages:**
- Only works for shared prefixes, not user-specific conversation history
- Requires identification and management of "hot" prefixes
- Storage cost of maintaining prefix caches in every region's GPU memory
- Cache invalidation when system prompts change

**When this wins:** High-traffic applications with standardized system prompts (customer service bots, coding assistants with standard instructions, RAG applications with fixed retrieval templates).

### Hybrid Approaches in Production

Real systems combine all three strategies. A production deployment might:
- Replicate the top 100 system prompt prefixes globally (Strategy C)
- Route new conversations to the nearest region with fresh prefill (Strategy A)
- Transfer KV cache for long-running sessions when users switch regions (Strategy B)

The decision engine considers: prompt length, existing KV cache size, inter-region bandwidth availability, GPU utilization in each region, and user latency SLA.

## KV Cache Transfer Protocols

When Strategy B is chosen, the implementation depends heavily on the network fabric connecting regions. Four primary protocols are used in production systems, each with distinct performance characteristics.

### RDMA over InfiniBand

Remote Direct Memory Access allows one machine to read from or write to another machine's memory without involving the remote CPU. InfiniBand networks (commonly HDR at 200 Gbps or NDR at 400 Gbps) provide RDMA natively.

**Mechanism:**
```
GPU A (Region 1)          Network Fabric          GPU B (Region 2)
     |                         |                        |
     | 1. Register memory      |                        |
     |    region for RDMA      |                        |
     |------------------------>|                        |
     |                         | 2. RDMA WRITE          |
     |                         |   (zero-copy,          |
     |                         |    bypasses CPU)        |
     |                         |----------------------->|
     |                         |                        | 3. KV cache
     |                         |                        |    appears in
     |                         |                        |    GPU B memory
```

**Performance:** At 400 Gbps (NDR InfiniBand), transferring 2.5 GB takes:
- Theoretical: 2.5 GB × 8 bits/byte ÷ 400 Gbps = 50ms
- Practical (with protocol overhead, ~85% efficiency): ~59ms

**Limitation:** InfiniBand is typically limited to intra-datacenter or campus-scale distances (< 10 km). Cross-region (hundreds to thousands of km) requires different transport.

### GPUDirect RDMA (GDR)

GPUDirect extends RDMA to operate directly on GPU memory without staging through host (CPU) memory. This eliminates two memory copies that would otherwise be required.

**Without GPUDirect:**
```
GPU HBM → PCIe → Host RAM → NIC → Network → NIC → Host RAM → PCIe → GPU HBM
         (copy 1)          (copy 2)        (copy 3)          (copy 4)
```

**With GPUDirect RDMA:**
```
GPU HBM → NIC → Network → NIC → GPU HBM
      (zero-copy, DMA directly from GPU memory)
```

This eliminates 2 memory copies and the associated host memory bandwidth bottleneck. For KV cache transfer, where the data originates in and must arrive in GPU HBM, GPUDirect saves approximately 30-40% in transfer time compared to staged transfers.

**Requirement:** Both the GPU and the NIC must support GPUDirect. NVIDIA H100/A100 GPUs with Mellanox ConnectX-7 NICs support this natively. The NIC must be on the same PCIe root complex as the GPU for optimal performance.

### TCP/NCCL over Wide-Area Networks

For cross-region transfers where InfiniBand is unavailable, NVIDIA's NCCL (Collective Communications Library) can operate over TCP/IP networks. Cloud providers offer dedicated inter-region backbone links:

- **AWS**: Inter-region networking at 25-100 Gbps per flow (via AWS backbone)
- **GCP**: Cross-region at up to 100 Gbps (Premium Tier)
- **Azure**: ExpressRoute Global Reach at 10-100 Gbps

**Performance at 100 Gbps:**
- 2.5 GB transfer: 2.5 × 8 ÷ 100 = 200ms theoretical
- Practical (TCP overhead, congestion, ~70% utilization): ~286ms

**NCCL optimization:** NCCL implements pipelining, chunking, and multi-path routing to maximize throughput. A 2.5 GB transfer is split into smaller chunks that can be processed in parallel across multiple TCP streams, reducing tail latency.

### S3 Express One Zone (Object Store Transfer)

For asynchronous or non-latency-critical KV cache movement, cloud object storage provides a reliable persistence layer:

**AWS S3 Express One Zone characteristics:**
- Single-digit millisecond first-byte latency
- Located in specific Availability Zones (co-located with compute)
- Up to 10× faster than standard S3 for small/medium objects
- Cost: ~$0.16/GB-month storage + $0.008/1000 PUT requests

**Use case:** Persist KV cache snapshots to S3 Express when a user session goes idle. If the user returns (potentially in a different region), fetch the KV cache from S3 rather than re-prefilling. This trades storage cost for compute savings.

**Performance:** S3 Express can deliver ~5-10 GB/s throughput to EC2 instances in the same AZ. Cross-AZ adds 1-2ms. Cross-region requires S3 Cross-Region Replication (minutes of delay), making it unsuitable for real-time transfer but viable for background pre-positioning.

```python
# Pseudocode: KV cache persistence decision
def should_persist_kv_cache(session):
    kv_size_gb = session.kv_cache_bytes / 1e9
    idle_probability = session.predict_idle_probability()
    prefill_cost_ms = estimate_prefill_time(session.total_tokens)
    storage_cost = kv_size_gb * 0.16 / 720  # per-hour cost
    
    # Persist if: expected savings > storage cost
    # Savings = idle_probability * prefill_cost * gpu_hourly_rate
    expected_savings = idle_probability * (prefill_cost_ms / 1000) * gpu_cost_per_second
    return expected_savings > storage_cost
```

Knowing the transfer mechanisms is necessary but not sufficient. The real question every platform team asks is: when does it actually pay to transfer rather than re-compute? The answer depends on a concrete numbers comparison that we can model precisely. Let us build that cost model now.

## Transfer Cost Analysis: The Crossover Point

The central question in multi-region KV cache management is: **when is transferring faster than re-computing?** This depends on a concrete comparison between transfer time and prefill time.

### Setting Up the Comparison

Consider a Llama 70B model with GQA (8 KV heads, head_dim=128, 80 layers):

**KV cache size per token:**
```
bytes_per_token = 2 × num_kv_heads × head_dim × num_layers × dtype_bytes
                = 2 × 8 × 128 × 80 × 2 (FP16)
                = 327,680 bytes ≈ 320 KB per token
```

**At various context lengths:**

| Context Length | KV Cache Size | Transfer @ 400 Gbps | Transfer @ 100 Gbps | Prefill Time (H100) |
|---------------|---------------|---------------------|---------------------|---------------------|
| 512 tokens | 160 MB | 3.2ms | 12.8ms | ~15ms |
| 2048 tokens | 640 MB | 12.8ms | 51.2ms | ~60ms |
| 4096 tokens | 1.28 GB | 25.6ms | 102.4ms | ~150ms |
| 8192 tokens | 2.56 GB | 51.2ms | 204.8ms | ~450ms |
| 16384 tokens | 5.12 GB | 102.4ms | 409.6ms | ~1400ms |
| 32768 tokens | 10.24 GB | 204.8ms | 819.2ms | ~4800ms |

**Key observations:**
1. At 400 Gbps (intra-region RDMA), transfer is faster than prefill for contexts > ~1K tokens
2. At 100 Gbps (inter-region TCP), the crossover happens around ~4K tokens
3. For very long contexts (32K+), transfer is 5-20× faster than re-prefill regardless of bandwidth
4. Prefill time grows quadratically with attention (O(n²)), while transfer grows linearly with size (O(n))

### The Crossover Formula

```python
def crossover_analysis(num_tokens, model_config, network_bandwidth_gbps, gpu_flops):
    """
    Determine whether transferring KV cache is faster than re-prefilling.
    
    Returns: (transfer_time_ms, prefill_time_ms, recommendation)
    """
    # KV cache size
    kv_bytes_per_token = (
        2 *  # K and V
        model_config.num_kv_heads *
        model_config.head_dim *
        model_config.num_layers *
        2  # FP16 = 2 bytes
    )
    total_kv_bytes = num_tokens * kv_bytes_per_token
    
    # Transfer time (including protocol overhead factor)
    protocol_efficiency = 0.75  # 75% of theoretical bandwidth achieved
    effective_bandwidth_bytes_per_sec = (
        network_bandwidth_gbps * 1e9 / 8 * protocol_efficiency
    )
    transfer_time_ms = (total_kv_bytes / effective_bandwidth_bytes_per_sec) * 1000
    
    # Prefill time (simplified: dominated by attention for long sequences)
    # Attention FLOPs ≈ 2 * num_layers * num_heads * seq_len² * head_dim
    attention_flops = (
        2 * model_config.num_layers * 
        model_config.num_attention_heads * 
        num_tokens ** 2 * 
        model_config.head_dim
    )
    # MLP FLOPs ≈ 2 * num_layers * seq_len * (8 * hidden_dim²)  [for SwiGLU]
    mlp_flops = (
        2 * model_config.num_layers * 
        num_tokens * 
        8 * model_config.hidden_dim ** 2
    )
    total_flops = attention_flops + mlp_flops
    
    # GPU utilization during prefill (typically 40-60% MFU)
    mfu = 0.50
    effective_flops = gpu_flops * mfu
    prefill_time_ms = (total_flops / effective_flops) * 1000
    
    recommendation = "TRANSFER" if transfer_time_ms < prefill_time_ms else "RE-PREFILL"
    return transfer_time_ms, prefill_time_ms, recommendation
```

### Why the Crossover Matters

The quadratic nature of attention computation means that as conversations grow longer, the advantage of transferring over re-computing increases dramatically. For a 32K-token context:
- Transfer at 100 Gbps: ~820ms
- Re-prefill on H100: ~4800ms

That is a 6× difference. For a 64K context (increasingly common with Claude, Gemini, GPT-4), the gap grows to 10-15×. This is why every major inference system at scale implements some form of KV cache migration.

## Decision Framework: Transfer vs. Re-Compute

Production systems need an automated decision engine that evaluates each request in real-time. The framework below integrates all relevant factors.

### Input Signals

```python
@dataclass
class RoutingDecision:
    """Inputs to the KV cache routing decision."""
    # User context
    user_region: str              # Where the user is (determined by IP/edge PoP)
    session_id: str               # Conversation identifier
    
    # KV cache state
    kv_cache_region: str          # Where existing KV cache lives (if any)
    kv_cache_size_bytes: int      # Current KV cache size
    total_tokens_cached: int      # Number of tokens already in KV cache
    new_prompt_tokens: int        # Tokens in the new user message
    
    # System state
    gpu_utilization: dict         # {region: utilization%}
    bandwidth_available: dict     # {(src, dst): current_gbps}
    queue_depth: dict             # {region: pending_requests}
    
    # SLA
    ttft_budget_ms: float         # Maximum acceptable time-to-first-token
```

### Decision Logic

```python
def route_request(decision: RoutingDecision) -> str:
    """
    Returns: 'local_prefill', 'transfer_kv', or 'prefix_cache_hit'
    """
    # Check 1: Is there a prefix cache hit in the user's region?
    if has_prefix_cache(decision.user_region, decision.session_id):
        return 'prefix_cache_hit'  # Best case: no transfer, partial prefill only
    
    # Check 2: No existing KV cache? Must do fresh prefill
    if decision.kv_cache_size_bytes == 0:
        return 'local_prefill'
    
    # Check 3: Existing KV cache is in user's region already
    if decision.kv_cache_region == decision.user_region:
        return 'local_prefill'  # Append new tokens to existing cache
    
    # Check 4: Compare transfer time vs re-prefill time
    transfer_time = estimate_transfer_time(
        decision.kv_cache_size_bytes,
        decision.bandwidth_available.get(
            (decision.kv_cache_region, decision.user_region), 0
        )
    )
    
    prefill_time = estimate_prefill_time(
        decision.total_tokens_cached + decision.new_prompt_tokens
    )
    
    # Include queue wait times in both regions
    transfer_total = transfer_time + decision.queue_depth[decision.user_region] * AVG_DECODE_TIME
    prefill_total = prefill_time + decision.queue_depth[decision.user_region] * AVG_DECODE_TIME
    
    # Decision with hysteresis (prefer local to avoid network dependency)
    HYSTERESIS_FACTOR = 1.3  # Transfer must be 30% faster to overcome complexity
    
    if transfer_total * HYSTERESIS_FACTOR < prefill_total:
        if transfer_total < decision.ttft_budget_ms:
            return 'transfer_kv'
    
    return 'local_prefill'
```

### The Hysteresis Factor

The `HYSTERESIS_FACTOR = 1.3` deserves explanation. Even when transfer is marginally faster, re-prefilling locally is preferred because:
1. **Reliability**: Local prefill has no network dependency; transfer can fail mid-way
2. **Simplicity**: No coordination between regions required
3. **GPU utilization**: The source region's GPU must hold the KV cache until transfer completes, reducing its capacity
4. **Tail latency**: Network jitter affects transfer but not local compute

Only when transfer is substantially faster (30%+ advantage) should the system pay the complexity cost. This threshold can be tuned based on operational experience.

The decision framework above handles individual requests reactively. But the highest-leverage optimization is proactive: if we can predict which KV caches will be needed globally and pre-position them, we eliminate the transfer-vs-recompute decision entirely for those requests. This is what prefix pools accomplish.

## Global Prefix Pools: Pre-Warming Common KV Caches

The highest-leverage optimization for multi-region inference is recognizing that many users share the same prompt prefix. In a customer service chatbot, every user gets the same 2000-token system prompt. In a coding assistant, the system instructions plus tool definitions might be 3000 tokens. In a RAG application, the retrieval template is identical across requests.

### How Prefix Pools Work

```
Central Prefix Manager
         |
    ┌────┼────┐────────┐────────┐
    ▼    ▼    ▼        ▼        ▼
 us-east eu-west ap-northeast ap-south us-west
    |    |    |        |        |
   GPU  GPU  GPU      GPU      GPU
  [Prefix KV cached in each region's GPU memory]
```

**Lifecycle:**
1. **Registration**: Application registers a prefix (e.g., system prompt) with the prefix pool manager
2. **Computation**: One region computes the KV cache for the prefix via prefill
3. **Distribution**: The computed KV cache is replicated to all active regions
4. **Serving**: When a request arrives with a matching prefix, the pre-computed KV cache is loaded and only the user-specific suffix requires prefill
5. **Invalidation**: When the prefix changes (system prompt update), all regions evict the stale cache and recompute

### Savings Quantification

For a typical enterprise chatbot:
- System prompt: 1500 tokens (fixed across all users)
- Few-shot examples: 800 tokens (fixed)
- Retrieved context: 500-2000 tokens (variable per query)
- User message: 50-200 tokens (unique)

With prefix caching, 2300 of the ~4000 total tokens (57%) have pre-computed KV caches. Prefill only runs on the remaining 1700 tokens, reducing prefill time by approximately 57% and TTFT proportionally.

**At scale (10M requests/day):**
```
Prefill savings = 10M × 2300 tokens × prefill_cost_per_token
                = 10M × 2300 × 0.00003 GPU-seconds
                = 690,000 GPU-seconds/day saved
                = 8 GPU-days/day saved
                ≈ $240/day on H100 instances (at $30/GPU-hour)
                ≈ $87,600/year
```

And that is for a single prefix. Applications with multiple prefixes (different system prompts for different products) multiply these savings.

### Implementation: Prefix Matching

```python
class PrefixPool:
    """
    Manages pre-computed KV caches for common prompt prefixes.
    Uses a trie for efficient prefix matching.
    """
    def __init__(self):
        self.prefix_trie = TokenTrie()  # Maps token sequences to KV cache refs
        self.kv_store = {}  # prefix_hash -> KV cache tensor
    
    def register_prefix(self, prefix_tokens: list[int], kv_cache: KVCache):
        """Register a pre-computed KV cache for a token prefix."""
        prefix_hash = hash_tokens(prefix_tokens)
        self.prefix_trie.insert(prefix_tokens, prefix_hash)
        self.kv_store[prefix_hash] = kv_cache
    
    def lookup(self, prompt_tokens: list[int]) -> tuple[KVCache | None, int]:
        """
        Find the longest matching prefix in the pool.
        Returns: (kv_cache, num_matched_tokens) or (None, 0)
        """
        matched_hash, match_length = self.prefix_trie.longest_match(prompt_tokens)
        if matched_hash is not None:
            return self.kv_store[matched_hash], match_length
        return None, 0
    
    def serve_request(self, prompt_tokens: list[int], gpu_id: int):
        """
        Serve a request using prefix cache if available.
        Only prefills the unmatched suffix.
        """
        cached_kv, matched_length = self.lookup(prompt_tokens)
        
        if cached_kv is not None:
            # Load pre-computed KV cache for the prefix
            load_kv_to_gpu(cached_kv, gpu_id)
            # Only prefill the suffix (tokens after the matched prefix)
            suffix_tokens = prompt_tokens[matched_length:]
            new_kv = run_prefill(suffix_tokens, existing_kv=cached_kv, gpu_id=gpu_id)
            return new_kv
        else:
            # No prefix match: full prefill required
            return run_prefill(prompt_tokens, gpu_id=gpu_id)
```

### Prefix Pool Challenges

**Cache invalidation**: When a system prompt is updated (even a single token change), all cached KV entries become invalid. The system must detect changes (via hashing the token sequence) and trigger global invalidation. This matters because stale prefix caches produce silently incorrect outputs: the model generates tokens conditioned on an outdated system prompt, potentially violating new safety rules or reflecting deprecated behavior. Unlike a database cache miss that simply returns no data, a KV cache hit on a stale prefix returns plausible but wrong continuations. Detection must happen at the token level (not text level) because tokenizer merges can produce different token IDs for text that appears identical to humans.

**Memory pressure**: Each cached prefix consumes GPU HBM. A 2000-token prefix for Llama 70B consumes ~640 MB. If you cache 50 popular prefixes, that is 32 GB of HBM, which is 40% of an H100's 80 GB. This directly competes with capacity for active decode sessions: every gigabyte allocated to prefix caches is a gigabyte unavailable for concurrent user requests. Eviction policies (LRU, frequency-based) are essential, but eviction decisions are non-trivial because evicting a highly-shared prefix forces re-prefill for all subsequent requests using it, creating a cascading latency spike that can overwhelm the prefill pool. The system must balance memory headroom against the amortized compute savings of keeping popular prefixes resident.

**Tokenizer sensitivity**: The same text can tokenize differently depending on what precedes it (due to BPE merge rules). Prefix matching must operate on token IDs, not text, and the prefix must be tokenized independently to ensure consistent boundaries. This is a subtle correctness issue: if the boundary between prefix and suffix falls mid-token (because the combined text would merge those characters into a single token), the cached KV values for the final prefix tokens are computed with incorrect positional context. The result is degraded generation quality that is difficult to diagnose because the outputs look reasonable but are subtly off-distribution. Production systems must enforce clean token boundaries by always tokenizing the prefix in isolation and only appending suffix tokens that begin at a valid BPE boundary.

## Session Stickiness Across Regions

Multi-turn conversations create session state (the accumulated KV cache) that must be tracked. When a user's network path changes, the system must handle it gracefully.

### Why Users Switch Regions

1. **Mobile roaming**: User starts a conversation on home WiFi (routed to us-east-1), continues on cellular during commute (routed to us-west-2 by carrier CDN). From a KV cache perspective, mobile roaming produces a partial warm state: the user's session KV cache exists in the original region and the conversation is likely mid-stream, meaning the cached context is valuable (several turns of history). However, the switch is often temporary and unpredictable, so the system faces a gamble: transfer the KV cache to the new region (expensive if the user switches back in minutes) or re-prefill locally (wasteful if the user stays in the new region). The optimal strategy is typically to re-prefill for the first request after roaming and only transfer if the user sends multiple consecutive requests from the new region.

2. **VPN switches**: User connects or disconnects a VPN, changing their apparent geography instantaneously. This triggers a full cold start from the KV cache perspective because there is no gradual transition: one moment the user appears to be in eu-west-1, the next in ap-southeast-1. The existing KV cache in the original region is perfectly valid but now completely inaccessible at low latency. Unlike mobile roaming where partial warming is possible, VPN switches create a binary state change that forces an immediate transfer-vs-recompute decision with no graceful middle ground.

3. **Load balancing**: Traffic shifts during peak hours redistribute users across regions. This is the most predictable scenario because load balancers typically shift traffic gradually and the system can proactively prepare. When a load balancer decides to redirect overflow traffic from us-east-1 to us-west-2, it can initiate background KV cache transfers for sessions likely to be redirected, pre-warming the destination region before the user's next request arrives. The KV cache implication is that emergency transfer bandwidth must be provisioned for these peak-hour migrations.

4. **Failover**: Region degradation triggers automatic failover routing. This is the worst case for KV cache locality because it combines urgency (the source region may be unreachable, making transfer impossible) with scale (all sessions in the degraded region need KV caches simultaneously). The KV cache implication is that failover requires emergency re-prefill capacity in backup regions, and the system must accept that all active sessions will experience a latency spike on their next request as KV caches are rebuilt from scratch. Pre-replication of at least the prefix pool to backup regions partially mitigates this by ensuring the shared portion of KV caches survives the failover.

### Session Tracking Architecture

```python
class SessionRouter:
    """
    Tracks active sessions and their KV cache locations.
    Backed by a globally replicated metadata store (e.g., DynamoDB Global Tables).
    """
    def __init__(self, metadata_store):
        self.store = metadata_store  # Global, strongly consistent reads
    
    def get_session_location(self, session_id: str) -> SessionMetadata | None:
        """Look up where a session's KV cache currently lives."""
        return self.store.get(f"session:{session_id}")
    
    def route_multi_turn(self, session_id: str, user_region: str) -> RoutingPlan:
        """
        Route a multi-turn request considering existing KV cache location.
        """
        metadata = self.get_session_location(session_id)
        
        if metadata is None:
            # New session: route to nearest region
            return RoutingPlan(
                target_region=user_region,
                action='fresh_prefill'
            )
        
        if metadata.kv_region == user_region:
            # Same region: append to existing KV cache
            return RoutingPlan(
                target_region=user_region,
                action='append',
                gpu_id=metadata.gpu_id
            )
        
        # Different region: evaluate transfer vs re-prefill
        decision = evaluate_transfer_vs_prefill(
            kv_size=metadata.kv_size_bytes,
            source_region=metadata.kv_region,
            target_region=user_region,
            bandwidth=get_inter_region_bandwidth(metadata.kv_region, user_region)
        )
        
        return RoutingPlan(
            target_region=user_region,
            action=decision,
            source_region=metadata.kv_region,
            source_gpu=metadata.gpu_id
        )
```

### Graceful Degradation

When transfer is not feasible (network partition, source GPU already freed the cache), the system must fall back to re-prefill without error. The user experiences slightly higher latency on that turn but the conversation continues. The new KV cache is then tracked at its new location for subsequent turns.

```
Turn 1: User in us-east-1 → Prefill → KV cache in us-east-1 (GPU 3)
Turn 2: User still in us-east-1 → Append to existing KV → Fast response
Turn 3: User now in eu-west-1 → Transfer attempted → 2.5 GB in 200ms → Success
Turn 4: User still in eu-west-1 → Append to transferred KV → Fast response
Turn 5: User back in us-east-1 → Old cache evicted → Re-prefill (conversation replay)
```

## Production Architectures: DistServe, Mooncake, and Splitwise

Three landmark systems have addressed multi-region KV cache locality with different architectural approaches. Understanding their designs reveals the design space.

### DistServe (Zhong et al., 2024)

**Core idea**: Disaggregate prefill and decode into separate GPU pools, connected by high-bandwidth KV cache transfer.

**Architecture:**
```
                    ┌─────────────────────┐
                    │   Request Router    │
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼                              ▼
    ┌─────────────────┐           ┌─────────────────┐
    │  Prefill Pool   │           │  Decode Pool    │
    │  (High-FLOPS    │  KV cache │  (High-MBW      │
    │   GPUs, e.g.    │──────────>│   GPUs, e.g.    │
    │   H100 SXM)     │  transfer │   A100/L40)     │
    └─────────────────┘           └─────────────────┘
```

**Key insight from the paper**: Prefill is compute-bound (high arithmetic intensity, benefits from FLOPS), while decode is memory-bandwidth-bound (low arithmetic intensity, benefits from memory bandwidth per dollar). Using the same GPU type for both wastes either FLOPS (during decode) or memory bandwidth (during prefill).

**KV cache transfer in DistServe:**
- After prefill completes, the KV cache is transferred from the prefill GPU to a decode GPU via RDMA
- The paper reports transfer overhead of 5-20ms for typical KV cache sizes (500 MB to 2 GB)
- This is acceptable because it happens once per request, and the decode latency savings (from using optimized decode hardware) amortize the transfer cost across dozens of decode steps

**Multi-region extension**: DistServe's architecture naturally extends to multi-region by placing prefill pools in regions with cheap, high-FLOPS GPUs and decode pools in regions closest to users. The KV cache transfer already exists in the architecture; extending it across regions changes only the transfer latency (from ~5ms intra-DC to ~50-200ms inter-region).

**Results (from paper):**
- 1.5-2.3× improvement in per-GPU goodput (tokens/second) compared to colocated serving
- P99 TTFT reduced by 2-4× due to elimination of decode interference with prefill scheduling
- KV cache transfer adds < 10% to end-to-end latency when using RDMA

### Mooncake (Moonshot AI, 2024)

**Core idea**: A KV-cache-centric disaggregated architecture where KV cache is treated as a first-class distributed object, stored in a dedicated "KVCache Pool" separate from both prefill and decode GPUs.

**Architecture:**
```
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  Prefill GPU │     │  Prefill GPU │     │  Prefill GPU │
    │    Pool      │     │    Pool      │     │    Pool      │
    └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
           │                    │                    │
           ▼                    ▼                    ▼
    ┌──────────────────────────────────────────────────────┐
    │              KVCache Pool (DRAM + SSD)                │
    │   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐         │
    │   │Node1│ │Node2│ │Node3│ │Node4│ │Node5│  ...      │
    │   └─────┘ └─────┘ └─────┘ └─────┘ └─────┘         │
    └──────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  Decode GPU  │     │  Decode GPU  │     │  Decode GPU  │
    │    Pool      │     │    Pool      │     │    Pool      │
    └──────────────┘     └──────────────┘     └──────────────┘
```

**Key innovations:**
1. **KV cache as a service**: KV cache is stored in CPU DRAM (not GPU HBM), allowing much larger capacity at lower cost. An 8-node pool with 512 GB DRAM each can store 4 TB of KV cache, equivalent to millions of active sessions.

2. **Prefix-aware scheduling**: Mooncake's scheduler routes requests to prefill GPUs that already have matching prefix KV caches loaded, maximizing cache hit rates. This is the global prefix pool idea implemented at the scheduler level.

3. **Chunk-based storage**: KV cache is divided into fixed-size chunks (e.g., 256 tokens per chunk). Chunks are individually addressable and can be composed to serve partial prefix matches.

4. **Multi-tier storage**: Hot KV caches live in DRAM for fast access (< 1ms). Warm caches spill to NVMe SSD (5-10ms access). Cold caches are evicted entirely.

**Multi-region implications**: Mooncake's separation of KV cache from GPU memory makes cross-region replication much more tractable. Replicating DRAM-resident KV cache between regions (CPU-to-CPU) is simpler and cheaper than GPU-to-GPU transfer. The KVCache Pool can span regions with eventual consistency for popular prefixes and on-demand replication for session-specific caches.

### Splitwise (Microsoft Research, 2024)

**Core idea**: Split inference between a "prompt machine" (optimized for prefill throughput) and a "token machine" (optimized for decode latency), with KV cache transferred between them.

**Distinguishing approach**: Splitwise focuses on heterogeneous hardware pairing. Rather than using the same GPU type in different roles, it pairs:
- **Prompt machines**: Dense compute (high SM count), less memory needed (KV cache is transferred out immediately after prefill)
- **Token machines**: High memory bandwidth, large HBM capacity (many concurrent decode sessions share the GPU)

**KV cache transfer mechanism**: Splitwise uses a pipeline overlap strategy where KV cache transfer begins for completed layers while prefill continues on later layers:

```
Layer 0 prefill complete → Begin transferring Layer 0 KV to token machine
Layer 1 prefill complete → Begin transferring Layer 1 KV to token machine
...
Layer 79 prefill complete → Begin transferring Layer 79 KV
                            (by now, Layers 0-60 already arrived at token machine)
```

This pipeline overlap hides most of the transfer latency behind ongoing prefill computation. The paper reports that with NVLink/RDMA connections, the effective overhead of KV transfer approaches zero for sequences longer than 1K tokens.

### Comparing the Three Architectures

| Aspect | DistServe | Mooncake | Splitwise |
|--------|-----------|----------|-----------|
| KV cache location | Transferred GPU→GPU | Stored in DRAM pool | Pipelined GPU→GPU |
| Multi-region readiness | Moderate (GPU transfer extends to WAN) | High (DRAM pool replicates easily) | Low (tight coupling required) |
| Hardware requirement | Homogeneous GPUs OK | CPU DRAM nodes + GPUs | Heterogeneous GPU types |
| Prefix sharing | Not native | Core feature | Not native |
| Scalability | Linear with GPU count | Massive (TB-scale KV pool) | Pair-based |
| Complexity | Medium | High | Medium |
| Best for | Mid-scale with uniform hardware | Hyperscale with shared prefixes | Cost-optimized heterogeneous clusters |

## Cost Analysis: Inter-Region Economics

Multi-region KV cache management introduces costs that do not exist in single-region deployments. Understanding these costs is essential for architecture decisions.

### Data Transfer Pricing

Cloud providers charge for inter-region data transfer:

| Provider | Intra-region | Cross-region (same continent) | Cross-continent |
|----------|-------------|-------------------------------|-----------------|
| AWS | $0.01/GB | $0.02/GB | $0.02-0.09/GB |
| GCP | Free (same zone) | $0.01/GB | $0.02-0.08/GB |
| Azure | Free (same zone) | $0.02/GB | $0.02-0.08/GB |

**Example**: Transferring a 2.5 GB KV cache cross-continent costs $0.05-0.22 per transfer. At 10,000 transfers/day, that is $500-2,200/day in data transfer alone.

### GPU Idle Time During Transfer

While a KV cache is being transferred, two GPUs are partially occupied:
- **Source GPU**: Must hold the KV cache in memory until transfer completes (cannot reclaim that HBM for other requests)
- **Destination GPU**: Waiting for the full KV cache before decode can begin (GPU cycles wasted)

For a 200ms transfer at 100 Gbps:
- Source GPU opportunity cost: 200ms × $0.0083/GPU-second (H100 on-demand) = $0.0017
- Destination GPU wait cost: 200ms × $0.0083/GPU-second = $0.0017
- Total GPU idle cost per transfer: ~$0.0034

At 10,000 transfers/day: $34/day in GPU idle costs. This is small compared to data transfer fees but scales linearly.

### Replication Storage for Prefix Pools

Maintaining pre-computed KV caches across N regions:
- 50 popular prefixes × 640 MB each = 32 GB per region
- Across 5 regions = 160 GB of GPU HBM dedicated to prefix caches
- At H100 pricing, 160 GB HBM ≈ 2 full GPUs worth of memory
- Monthly cost: 2 GPUs × $30/hour × 720 hours = $43,200/month

This is only justified when the compute savings from prefix caching exceed the storage cost. With 10M+ daily requests sharing those prefixes, the savings ($87,600/year as computed earlier) justify the cost for high-traffic applications.

### Total Cost Comparison

For a global application serving 1M requests/day across 3 regions:

| Strategy | Monthly Cost | TTFT (P50) | TTFT (P99) |
|----------|-------------|------------|------------|
| Single region (all traffic to us-east-1) | $45,000 (GPUs) | 80ms (local) / 300ms (remote) | 150ms / 500ms |
| Multi-region, always re-prefill | $135,000 (3× GPUs) | 80ms | 200ms |
| Multi-region + KV transfer | $140,000 (GPUs + transfer) | 60ms | 150ms |
| Multi-region + prefix pools | $155,000 (GPUs + prefix HBM) | 40ms | 100ms |
| Multi-region + prefix + transfer | $160,000 (full system) | 35ms | 90ms |

The full system costs 3.5× more than single-region but delivers 4-5× better tail latency globally. For user-facing applications where latency directly impacts engagement and revenue, this tradeoff is favorable.

## Mental Model: KV Cache as a Distributed Systems Problem

Throughout this module, we have seen how KV cache locality at global scale exhibits all the classic distributed systems challenges:

**CAP theorem applies**: You cannot have KV cache that is simultaneously consistent (same state everywhere), available (always accessible locally), and partition-tolerant (works when regions cannot communicate). Production systems choose AP (available + partition-tolerant) with eventual consistency for prefix caches, and CP (consistent + partition-tolerant) for session-specific caches.

**Replication strategies mirror databases**: Prefix pools use leader-follower replication (one region computes, others receive copies). Session caches use lazy replication (only replicated on demand when users move).

**Caching hierarchies echo CPU architecture**: L1 = GPU HBM (fastest, smallest), L2 = host DRAM (Mooncake's KV pool), L3 = NVMe SSD (spill tier), Remote = cross-region transfer (slowest, effectively unlimited capacity).

**Consistency models matter**: A stale prefix cache (outdated system prompt) produces incorrect outputs. A missing session cache produces correct but slow outputs (re-prefill). The failure modes are asymmetric, which informs the design: prefix caches need strong consistency (invalidation on change), session caches can tolerate relaxed consistency (worst case is redundant prefill).

The next time you think about KV cache, do not think of it as "GPU memory management." Think of it as a globally distributed data store with replication, caching tiers, consistency guarantees, and transfer protocols. That mental shift unlocks the full design space for multi-region inference systems.

## Summary and Key Takeaways

1. **Single-region inference breaks at global scale** due to latency physics (~150-250ms cross-continent), compliance requirements, and availability concerns.

2. **Three strategies exist**: route to nearest (fresh prefill), transfer KV cache, or replicate prefixes. Production systems use all three.

3. **The crossover point** between transfer and re-prefill depends on context length and bandwidth. Above ~4K tokens at 100 Gbps (or ~1K tokens at 400 Gbps), transfer wins.

4. **Global prefix pools** deliver the highest ROI for high-traffic applications with shared system prompts, saving 50-80% of prefill compute.

5. **DistServe, Mooncake, and Splitwise** represent three architectures: GPU-to-GPU transfer, DRAM-based KV pools, and pipelined heterogeneous transfer respectively.

6. **Cost is non-trivial**: data transfer fees, GPU idle time, and replication storage add 10-30% to inference infrastructure costs, justified by 4-5× latency improvement.

7. **Mental model**: KV cache at global scale is a distributed systems problem exhibiting CAP tradeoffs, replication strategies, and caching hierarchies analogous to databases and CPU architecture.
