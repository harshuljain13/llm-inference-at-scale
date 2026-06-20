# 4.4 LMCache

Every KV cache tensor you have encountered so far shares one fatal limitation: it lives inside the inference engine's GPU memory and vanishes the moment the request completes. The system prompt that cost 2,048 tokens of prefill computation? Gone. The multi-turn conversation history that required 800ms of attention computation? Evaporated when the user's session migrated to a different GPU. LMCache solves this by treating KV cache not as a transient runtime artifact but as a persistent, shareable, distributed data structure that outlives individual requests, spans multiple servers, and hierarchically spills across GPU HBM, CPU DRAM, NVMe SSDs, and remote object stores.

This module teaches you how LMCache works, how it integrates with vLLM through the KVConnector API, and how production systems (NVIDIA Dynamo, ByteDance AIBrix, Red Hat llm-d) use it to achieve 3-15x throughput improvements on prefix-heavy workloads.

## Connection to Prior Modules

From Module 02.3, you know that Multi-head Latent Attention (MLA) compresses KV representations into low-rank latent vectors, reducing the per-token memory footprint. LMCache is complementary to that optimization: while MLA reduces WHAT is stored per token, LMCache manages WHERE those compressed (or uncompressed) vectors live and HOW they are shared across requests, sessions, and physical machines. From Module 03.1, you understand PagedAttention and how vLLM manages KV cache in GPU memory using block tables. LMCache extends that management beyond the GPU boundary, adding external persistence and cross-request sharing that PagedAttention alone cannot provide. From Module 03.3, you know continuous batching keeps the GPU busy by interleaving requests. LMCache amplifies continuous batching's efficiency: when a new request arrives with a cached prefix, it skips prefill entirely and enters the decode phase immediately, freeing prefill slots for requests that genuinely need computation.

## The Problem: KV Cache is Ephemeral

Consider a production chatbot serving 10,000 concurrent users. Every user's first message triggers prefill of the system prompt (typically 1,000-4,000 tokens). With a 70B parameter model using GQA (8 KV heads, 128 dimensions, 80 layers), each token in the KV cache occupies:

```
bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
bytes_per_token = 2 * 80 * 8 * 128 * 2  (FP16)
bytes_per_token = 327,680 bytes = 320 KB per token
```

A 2,048-token system prompt therefore requires 640 MB of KV cache memory and roughly 200ms of prefill computation on an A100. Multiply by 10,000 users and the system spends 2,000 GPU-seconds per second just recomputing identical system prompts. This is pure waste.

The waste compounds across three dimensions:

**Dimension 1: Repeated prefill across requests.** When 500 users share the same RAG retrieval context (because they asked similar questions), the engine computes 500 identical KV caches. No mechanism exists within standard vLLM or TensorRT-LLM to detect this duplication and serve from a shared cache.

**Dimension 2: Session migration.** In disaggregated serving architectures (Module 05.4), prefill and decode run on separate GPU pools. After prefill completes on GPU-A, the KV cache must somehow reach GPU-B for decode. Without an external cache layer, the only options are: (a) re-run prefill on GPU-B (wasting the original computation), or (b) transfer raw tensors over the network with custom code that couples the scheduler to the hardware topology.

**Dimension 3: Multi-turn conversation continuity.** A user sends message 1, gets a response, waits 30 seconds, then sends message 2. If the load balancer routes message 2 to a different server (or even the same server but a different vLLM worker), the conversation's KV cache from message 1 no longer exists. The engine must re-prefill the entire conversation history (system prompt + message 1 + response 1) before processing message 2.

These three waste patterns share a root cause: the inference engine treats KV cache as a private, ephemeral, GPU-local resource. LMCache inverts this assumption.

## LMCache Architecture

LMCache introduces an external KV cache management layer that sits between the inference engine and the storage hierarchy. Its architecture has four core components:

### Component 1: KV Connector API

The KVConnector is the interface between the inference engine (vLLM) and LMCache. It defines five operations:

```python
class KVConnector(ABC):
    """Abstract interface for external KV cache management."""

    @abstractmethod
    def store(
        self,
        request_id: str,
        kv_tensors: List[Tuple[torch.Tensor, torch.Tensor]],
        token_ids: List[int],
        layer_indices: List[int],
    ) -> None:
        """Persist KV tensors after prefill completes."""
        pass

    @abstractmethod
    def lookup(
        self,
        token_ids: List[int],
    ) -> Optional[Tuple[List[torch.Tensor], int]]:
        """Check if KV cache exists for this token sequence.
        Returns (kv_tensors, num_matched_tokens) or None."""
        pass

    @abstractmethod
    def transfer(
        self,
        request_id: str,
        target_worker: str,
        kv_tensors: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Send KV tensors to another worker (disaggregated serving)."""
        pass

    @abstractmethod
    def evict(
        self,
        token_ids: List[int],
    ) -> None:
        """Remove cached KV tensors under memory pressure."""
        pass

    @abstractmethod
    def contains(
        self,
        token_ids: List[int],
    ) -> int:
        """Return number of prefix tokens cached for this sequence."""
        pass
```

The critical design decision here is that LMCache indexes by token sequence, not by request ID. This means two different requests with the same token prefix automatically share the cached KV tensors. The engine does not need to know they are related. Prefix matching happens at the token level.

### Component 2: CacheEngine

The CacheEngine is the orchestrator that decides where to place KV tensors and when to promote or demote them across the storage hierarchy. It implements a tiered eviction policy:

```
Tier 0: GPU HBM     (fastest, smallest, ~10-40 GB available for cache)
Tier 1: CPU DRAM    (fast, medium, ~100-500 GB)
Tier 2: NVMe SSD   (medium, large, ~1-4 TB)
Tier 3: Remote Store (slow, unlimited, S3/Redis/shared memory)
```

The CacheEngine tracks access frequency and recency for each cached prefix. When GPU memory pressure increases (monitored via torch.cuda.memory_allocated), it demotes cold entries to CPU DRAM. When CPU memory pressure increases, it further demotes to NVMe. The promotion path reverses: when a lookup hits a CPU-resident entry, the CacheEngine asynchronously prefetches it back to GPU HBM in anticipation of the next access.

```python
class CacheEngine:
    def __init__(self, config: LMCacheConfig):
        self.gpu_cache = GPUCacheStore(max_bytes=config.gpu_cache_bytes)
        self.cpu_cache = CPUCacheStore(max_bytes=config.cpu_cache_bytes)
        self.disk_cache = DiskCacheStore(path=config.disk_cache_path)
        self.remote_cache = RemoteCacheStore(endpoint=config.remote_endpoint)
        self.metadata = PrefixTree()  # Tracks which tokens are cached where

    def store(self, token_ids: List[int], kv_tensors: List[torch.Tensor]):
        # Store in GPU first (hot path)
        if self.gpu_cache.has_capacity(kv_tensors):
            self.gpu_cache.put(token_ids, kv_tensors)
            self.metadata.insert(token_ids, tier=0)
        else:
            # Evict coldest GPU entries to CPU, then store
            evicted = self.gpu_cache.evict_lru(needed_bytes=tensor_bytes(kv_tensors))
            self.cpu_cache.put(evicted.token_ids, evicted.tensors)
            self.metadata.update(evicted.token_ids, tier=1)
            self.gpu_cache.put(token_ids, kv_tensors)
            self.metadata.insert(token_ids, tier=0)

    def lookup(self, token_ids: List[int]) -> Optional[CacheLookupResult]:
        # Find longest prefix match
        matched_length, tier = self.metadata.longest_prefix_match(token_ids)
        if matched_length == 0:
            return None

        matched_ids = token_ids[:matched_length]
        if tier == 0:
            tensors = self.gpu_cache.get(matched_ids)
        elif tier == 1:
            tensors = self.cpu_cache.get(matched_ids)
            # Async promote to GPU for next access
            self._schedule_promotion(matched_ids, tensors)
        elif tier == 2:
            tensors = self.disk_cache.get(matched_ids)
        else:
            tensors = self.remote_cache.get(matched_ids)

        return CacheLookupResult(tensors=tensors, matched_tokens=matched_length)
```

### Component 3: PrefixTree (Token-Level Index)

The PrefixTree (trie) is the data structure that enables O(n) longest-prefix matching where n is the query sequence length. Each node in the tree represents a token, and edges represent token transitions. Leaf nodes (or intermediate nodes with cache entries) store metadata: which tier holds the KV tensors, last access timestamp, access count, and tensor byte size.

```
Root
+-- [BOS] -> [System] -> [You] -> [are] -> [a] -> [helpful] -> [assistant] ...
|                                                                +-- CACHED (Tier 0, 2048 tokens, 640MB)
+-- [BOS] -> [System] -> [You] -> [are] -> [an] -> [expert] -> [in] -> [Python] ...
|                                                                        +-- CACHED (Tier 1, 3072 tokens, 960MB)
+-- [BOS] -> [User] -> [context:] -> [Document] -> [1] ...
                                                     +-- CACHED (Tier 2, 8192 tokens, 2.5GB)
```

When a new request arrives with token sequence [BOS, System, You, are, a, helpful, assistant, ..., User, What, is, Python?], the PrefixTree traverses from the root, matching tokens until it diverges. If the first 2,048 tokens match the cached system prompt, LMCache returns those KV tensors and tells the engine: "skip prefill for the first 2,048 tokens, start computing from token 2,049."

### Component 4: Hierarchical Storage Backends

Each tier implements the same interface but with different performance characteristics:

| Tier | Backend | Latency (2048 tokens, 70B) | Bandwidth | Capacity |
|------|---------|---------------------------|-----------|----------|
| 0 | GPU HBM (CUDAMalloc) | <1ms | 3.35 TB/s (H100) | 10-40 GB |
| 1 | CPU DRAM (pinned memory) | 2-5ms | 50-100 GB/s (PCIe 5.0) | 100-500 GB |
| 2 | NVMe SSD (io_uring) | 10-50ms | 7-14 GB/s (Gen5) | 1-8 TB |
| 3 | Remote (RDMA/TCP) | 50-200ms | 25-100 Gb/s (InfiniBand) | Unlimited |

The key insight is that even the slowest tier (remote, 200ms for a 640MB transfer) is faster than recomputing prefill from scratch (200-800ms for 2,048 tokens on a 70B model). The crossover point depends on sequence length: for short prefixes (<256 tokens), recomputation is faster than remote fetch; for long prefixes (>1,024 tokens), cache retrieval wins even from NVMe.

## Key Operations in Detail

### Operation 1: Store (Post-Prefill Persistence)

After the engine completes prefill for a request, it calls `connector.store()` with the computed KV tensors. LMCache must decide:

1. **Should this be cached?** Not all prefills deserve caching. A unique, one-off query prefix will never be reused. LMCache uses an admission policy: only cache if the prefix matches a known "cacheable pattern" (system prompts, RAG contexts) or if the prefix length exceeds a configured threshold (default: 256 tokens).

2. **At what granularity?** LMCache stores KV cache in chunks (default: 256 tokens per chunk). This enables partial prefix matching. If a request matches 1,792 of 2,048 cached tokens, LMCache returns 7 chunks and the engine only prefills the remaining 256 tokens.

3. **In which format?** KV tensors are stored in their native dtype (FP16/BF16) for GPU and CPU tiers. For disk and remote tiers, optional compression (FP8 quantization or zstd on the tensor bytes) reduces storage by 40-60% with minimal accuracy impact.

```python
def store_after_prefill(self, request_id: str, kv_cache: KVCache, tokens: List[int]):
    chunk_size = self.config.chunk_size  # 256 tokens
    num_chunks = len(tokens) // chunk_size

    for i in range(num_chunks):
        chunk_tokens = tokens[i * chunk_size : (i + 1) * chunk_size]
        chunk_kv = kv_cache.slice(i * chunk_size, (i + 1) * chunk_size)

        # Admission check
        if not self.admission_policy.should_cache(chunk_tokens, access_count=1):
            continue

        # Compress if targeting lower tier
        if self.config.enable_compression:
            chunk_kv = self.compressor.compress(chunk_kv)

        self.cache_engine.store(
            token_ids=tokens[: (i + 1) * chunk_size],  # Full prefix up to this chunk
            kv_tensors=chunk_kv,
        )
```

### Operation 2: Lookup (Pre-Prefill Cache Check)

Before starting prefill, the engine calls `connector.lookup()` with the full input token sequence. LMCache returns the longest cached prefix:

```python
def lookup_before_prefill(self, tokens: List[int]) -> PrefillPlan:
    result = self.cache_engine.lookup(tokens)

    if result is None:
        return PrefillPlan(cached_tokens=0, kv_tensors=None, compute_from=0)

    # Validate cached KV dimensions match current model config
    if not self._validate_kv_shape(result.tensors):
        self.cache_engine.evict(tokens[:result.matched_tokens])
        return PrefillPlan(cached_tokens=0, kv_tensors=None, compute_from=0)

    return PrefillPlan(
        cached_tokens=result.matched_tokens,
        kv_tensors=result.tensors,
        compute_from=result.matched_tokens,  # Engine starts prefill here
    )
```

The engine then loads the returned KV tensors directly into its PagedAttention block table (no recomputation needed for those tokens) and runs prefill only on the remaining suffix tokens.

### Operation 3: Transfer (Disaggregated Serving)

In disaggregated architectures where prefill and decode happen on different GPUs (Module 05.4 covers NVIDIA Dynamo's approach), the KV cache must move from the prefill worker to the decode worker. Without LMCache, this requires point-to-point NCCL transfers tightly coupled to the scheduler. With LMCache, the transfer is decoupled:

1. Prefill worker completes computation and calls `connector.store()`.
2. Scheduler assigns a decode worker and sends it the request metadata (including token IDs).
3. Decode worker calls `connector.lookup()` and retrieves the KV cache from the shared tier (CPU DRAM on the same node, or remote store for cross-node).

This decoupling has a profound architectural benefit: the scheduler no longer needs to know the physical topology of GPU interconnects. It assigns work based on load, and LMCache handles the data movement transparently.

```python
# Prefill worker (GPU pool A)
async def handle_prefill(self, request: InferenceRequest):
    kv_cache = await self.engine.prefill(request.tokens)
    # Store to shared tier (CPU DRAM or RDMA-accessible memory)
    await self.connector.store(
        request_id=request.id,
        kv_tensors=kv_cache,
        token_ids=request.tokens,
        layer_indices=list(range(self.model.num_layers)),
    )
    # Notify scheduler that prefill is complete
    await self.scheduler.prefill_complete(request.id)

# Decode worker (GPU pool B)
async def handle_decode(self, request: InferenceRequest):
    # Retrieve KV from shared cache
    result = await self.connector.lookup(request.tokens)
    assert result is not None, "KV cache must exist after prefill"
    # Load into local block table
    self.engine.load_kv_from_external(result.tensors, result.matched_tokens)
    # Begin autoregressive decode
    async for token in self.engine.decode(request):
        yield token
```

### Operation 4: Evict (Memory Pressure Management)

Eviction triggers when any tier exceeds its configured capacity. The CacheEngine uses a weighted LRU policy that considers both recency and prefix sharing potential:

```
eviction_score = (1 / time_since_last_access) * sharing_factor * size_penalty
```

Where `sharing_factor` is the number of unique requests that have hit this prefix in the last minute (higher sharing = less likely to evict), and `size_penalty` penalizes very large entries that consume disproportionate capacity.

Eviction cascades downward: GPU eviction demotes to CPU, CPU eviction demotes to disk, disk eviction deletes. Remote tier entries have TTL-based expiration rather than active eviction (the remote store manages its own garbage collection).

## Integration with vLLM

LMCache integrates with vLLM as a plugin through the KVConnector interface, requiring zero modifications to vLLM's core engine code. The integration points are:

### Configuration

```yaml
# vllm serve configuration with LMCache enabled
model: meta-llama/Llama-3.1-70B-Instruct
tensor-parallel-size: 4
kv-connector: lmcache.vllm.LMCacheConnector
kv-connector-config:
  gpu_cache_bytes: 10737418240  # 10 GB
  cpu_cache_bytes: 107374182400  # 100 GB
  disk_cache_path: /mnt/nvme/lmcache
  chunk_size: 256
  enable_compression: true
  compression_dtype: fp8_e4m3
  admission_threshold: 256  # minimum tokens to cache
  eviction_policy: weighted_lru
```

### Execution Flow

```
Request arrives at vLLM scheduler
    |
    +-- Scheduler calls connector.contains(token_ids)
    |   +-- Returns: 1792 tokens cached
    |
    +-- Scheduler marks request as "partial prefill" (only 256 tokens to compute)
    |   +-- This affects batch scheduling: partial prefills are cheaper
    |
    +-- Before prefill, engine calls connector.lookup(token_ids)
    |   +-- Returns KV tensors for first 1792 tokens
    |   +-- Engine loads them into block table slots [0..6] (7 blocks of 256)
    |
    +-- Engine runs prefill ONLY on tokens[1792:2048]
    |   +-- 87.5% computation saved
    |
    +-- After prefill, engine calls connector.store(token_ids, full_kv)
        +-- CacheEngine updates metadata (now 2048 tokens cached for this prefix)
```

### Compatibility Notes

LMCache's vLLM integration works with:
- PagedAttention v1 and v2
- GQA and MQA attention (stores only the KV heads, not Q)
- Tensor parallelism (each TP rank stores/retrieves its own KV shard)
- Continuous batching (partial prefills integrate into the batch schedule)
- Speculative decoding (cached KV for the draft model prefix)

It does not currently support:
- Prefix caching internal to vLLM (must be disabled; LMCache replaces it)
- Chunked prefill with mid-chunk cache boundaries (chunks must align)
- Cross-model sharing (KV from Llama-70B cannot be used for Llama-8B)

## Production Deployments

### NVIDIA Dynamo: KV-Aware Routing

NVIDIA Dynamo (Module 05.4) uses LMCache as its KV transfer backbone for disaggregated prefill/decode. The key innovation is KV-aware routing: Dynamo's scheduler knows which decode workers already have warm caches for which prefixes. When a new request arrives:

1. Dynamo's router queries LMCache's metadata to find which workers hold relevant KV cache.
2. If a decode worker already has 80%+ of the prefix cached locally (in its GPU HBM via LMCache), the router sends the request there.
3. If no worker has a warm cache, the router picks the least-loaded prefill worker and relies on LMCache's transfer mechanism to move KV to the assigned decode worker.

This KV-aware routing reduces inter-node data transfer by 60-80% in production workloads with high prefix sharing (measured on chatbot and RAG workloads by NVIDIA).

### ByteDance AIBrix: Multi-Tenant Cache Sharing

ByteDance's AIBrix platform serves hundreds of internal teams from shared GPU clusters. Each team may deploy different system prompts but share the same base model. LMCache enables:

- **Cross-tenant prefix sharing**: If Team A and Team B both use Llama-70B with overlapping system prompt prefixes, the shared portion is cached once and served to both.
- **Tenant isolation**: Each tenant's unique suffix (after the shared prefix diverges) is cached separately with tenant-level access controls.
- **Cost attribution**: LMCache tracks cache hit rates per tenant, enabling accurate chargeback (teams with cache-friendly patterns pay less for compute).

AIBrix reports 5-8x throughput improvement on their multi-tenant RAG workloads, where 70-90% of prefill tokens are shared across tenants within the same time window.

### Red Hat llm-d: Kubernetes-Native Cache Management

Red Hat's llm-d project (Module 07.7) integrates LMCache with Kubernetes scheduling primitives. The key design decisions:

- **Cache locality as a scheduling constraint**: The K8s scheduler considers KV cache residency when placing pods. A decode pod preferentially schedules on a node where LMCache's CPU tier already holds relevant KV tensors.
- **Persistent volumes for disk tier**: LMCache's NVMe tier maps to K8s PersistentVolumeClaims, surviving pod restarts and rescheduling.
- **CRD-based configuration**: LMCache settings are expressed as K8s Custom Resources, enabling GitOps-style management of cache policies.

```yaml
apiVersion: llm-d.io/v1alpha1
kind: KVCachePolicy
metadata:
  name: chatbot-cache-policy
spec:
  model: meta-llama/Llama-3.1-70B-Instruct
  tiers:
    - type: gpu
      maxBytes: 10Gi
      evictionPolicy: lru
    - type: cpu
      maxBytes: 100Gi
      evictionPolicy: weighted_lru
    - type: disk
      storageClass: nvme-fast
      maxBytes: 1Ti
      compression: fp8
  admission:
    minTokens: 256
    patterns:
      - prefix: "system:"
        priority: high
      - prefix: "rag:"
        priority: medium
```

## Performance Analysis

The LMCache paper (arXiv 2510.09665) reports performance across several workload patterns:

### Throughput Improvement

| Workload Pattern | Cache Hit Rate | Throughput vs. Baseline | TTFT Reduction |
|-----------------|----------------|------------------------|----------------|
| Single system prompt, many users | 95-99% | 10-15x | 85-95% |
| RAG with shared document chunks | 60-80% | 3-8x | 50-75% |
| Multi-turn conversations | 40-70% | 2-5x | 30-60% |
| Diverse, unique queries | <10% | 1.0-1.2x | <5% |

The throughput improvement comes from two sources:
1. **Reduced prefill computation**: Cached prefixes skip attention computation entirely.
2. **Increased decode throughput**: With fewer prefill slots occupied in continuous batching, more slots are available for decode iterations, increasing the system's aggregate token generation rate.

### Latency Breakdown

For a cache hit on a 2,048-token prefix (Llama-70B, FP16):

```
Without LMCache:
  Prefill computation: 200-400ms (2048 tokens, A100)
  Total TTFT: 200-400ms

With LMCache (GPU tier hit):
  Cache lookup: 0.1ms (PrefixTree traversal)
  KV tensor copy (GPU->GPU within same device): 0.5ms
  Remaining prefill (0 tokens): 0ms
  Total TTFT: 0.6ms  (99.7% reduction)

With LMCache (CPU tier hit):
  Cache lookup: 0.1ms
  KV tensor transfer (CPU->GPU via PCIe 5.0): 4-8ms (640MB at 80 GB/s)
  Remaining prefill (0 tokens): 0ms
  Total TTFT: 4-8ms  (96-98% reduction)

With LMCache (NVMe tier hit):
  Cache lookup: 0.1ms
  Disk read: 45-90ms (640MB at 7-14 GB/s)
  CPU->GPU transfer: 4-8ms
  Total TTFT: 49-98ms  (50-75% reduction)
```

### Memory Overhead

LMCache's metadata (PrefixTree + tier tracking) consumes approximately:
- 64 bytes per cached chunk (256 tokens)
- For 100,000 cached chunks (~25.6M tokens): 6.4 MB of metadata
- Negligible compared to the KV tensors themselves

### Cache Efficiency Metrics

Production systems track these metrics to tune LMCache:

```python
# Key metrics for LMCache monitoring
cache_hit_rate = hits / (hits + misses)           # Target: >60% for ROI
prefix_sharing_ratio = unique_prefixes / total_requests  # Lower = more sharing
avg_matched_tokens = sum(matched) / hits          # Longer matches = more savings
tier_distribution = {tier: count for tier in tiers}  # GPU hits >> CPU hits >> disk
eviction_rate = evictions_per_second              # High = cache too small
promotion_rate = promotions_per_second            # High = access pattern changing
```

## When to Use LMCache

LMCache delivers maximum value when three conditions hold simultaneously:

### Condition 1: High Prefix Sharing

The workload must have significant token-level prefix overlap across requests. Measure this by sampling 1,000 requests and computing the average longest common prefix (LCP) between all pairs. If the average LCP exceeds 256 tokens, LMCache will provide meaningful speedup.

**High sharing examples:**
- Chatbots with fixed system prompts (LCP = system prompt length, typically 1,000-4,000 tokens)
- RAG systems where multiple users query the same documents (LCP = document chunk size, typically 512-2,048 tokens)
- Code completion with shared repository context (LCP = file preamble + imports, typically 200-1,000 tokens)
- Multi-turn conversations where early turns are shared (LCP grows with conversation depth)

**Low sharing examples:**
- Translation services (each request is unique text)
- Summarization of unique documents (no prefix overlap)
- Creative writing with unique prompts

### Condition 2: Prefill is the Bottleneck

If your workload is decode-bound (long output generation with short prompts), LMCache helps less because prefill is already a small fraction of total latency. LMCache matters most when:

- Input length >> Output length (e.g., RAG: 4,000 input tokens, 200 output tokens)
- TTFT SLA is tight (e.g., <100ms for interactive chat)
- Prefill computation consumes >30% of total GPU time in your batch

### Condition 3: Memory Budget Allows Caching

LMCache requires additional memory beyond what the engine uses for active inference. Budget rule of thumb:

```
Additional memory needed = avg_cached_prefixes * avg_prefix_length * bytes_per_token
Example: 1000 prefixes * 2048 tokens * 320KB/token (70B model) = 640 GB
```

This is why the hierarchical tier design matters: 640 GB does not fit in GPU HBM, but fits comfortably in CPU DRAM (hot prefixes in GPU, warm in CPU, cold on disk).

## Anti-Patterns and Failure Modes

### Anti-Pattern 1: Caching Everything

Caching every prefix regardless of reuse probability wastes memory and increases eviction churn. The admission policy must filter:
- Prefixes shorter than the chunk size (too small to matter)
- Prefixes that appear only once (unique queries)
- Prefixes from low-priority tenants (in multi-tenant settings)

### Anti-Pattern 2: Ignoring Cache Coherence

If the system prompt changes (deployment update), all cached KV tensors for the old prompt become stale. LMCache validates by token sequence, so changed tokens naturally miss. But partial overlaps are dangerous: if only the last 10 tokens of a 2,048-token prompt change, LMCache still returns the first 2,038 tokens from cache, which is correct. The issue arises if the model's attention patterns over early tokens depend on later tokens in the prompt (which they do, in causal attention, only for bidirectional models, not autoregressive LLMs). For standard autoregressive models, partial cache reuse is always correct because KV at position i depends only on tokens 0..i.

### Anti-Pattern 3: Misaligned Chunk Boundaries

If LMCache uses 256-token chunks but the system prompt is 300 tokens, only the first 256 tokens are cached (one full chunk). The remaining 44 tokens always require recomputation. Tune chunk size to align with your dominant prefix lengths. Some deployments use variable chunk sizes (128, 256, 512) and select the best fit per prefix.

### Anti-Pattern 4: Cross-Model Cache Pollution

Two different LoRA adapters on the same base model produce different KV values for the same input tokens (because LoRA modifies the attention weights). LMCache must namespace by model identity:

```
cache_key = hash(model_id + lora_adapter_id + token_ids)
```

Without this namespacing, serving LoRA-A's KV to LoRA-B produces incorrect outputs.

## Advanced: Combining LMCache with Other Optimizations

### LMCache + Chunked Prefill (Module 03.5)

When LMCache returns a partial match (e.g., 1,792 of 2,048 tokens cached), the remaining 256 tokens still need prefill. Chunked prefill processes these 256 tokens in smaller chunks (e.g., 4 chunks of 64) interleaved with decode batches. The combination means: LMCache eliminates most of the prefill computation, and chunked prefill ensures the residual computation does not monopolize the GPU.

### LMCache + Speculative Decoding (Module 04.2)

The draft model in speculative decoding also has a KV cache. LMCache can cache the draft model's KV alongside the target model's KV. Since draft models are smaller (7B vs 70B), their KV cache is 8-10x cheaper to store. Pre-caching the draft model's prefix KV eliminates draft model prefill latency, making the first speculative batch start faster.

### LMCache + MLA Compression (Module 02.3)

DeepSeek's MLA compresses K and V into a joint latent vector of dimension 512 (vs. the original 128 * num_heads). LMCache stores these compressed latents, which are 4-8x smaller per token. This dramatically increases effective cache capacity:

```
Standard GQA (8 KV heads, 128 dim): 320 KB/token
MLA (512-dim latent): 40-80 KB/token
Capacity improvement: 4-8x more tokens cached in same memory
```

### LMCache + Disaggregated Serving (Module 05.4)

NVIDIA Dynamo's disaggregated architecture (separate prefill and decode GPU pools) treats LMCache as the transport layer between pools. The prefill pool writes to LMCache; the decode pool reads from it. This eliminates the need for direct GPU-to-GPU NCCL links between prefill and decode workers, simplifying the network topology and enabling independent scaling of each pool.

## Mental Model

Think of LMCache as a CDN for KV cache. Just as a content delivery network caches HTTP responses close to users (edge nodes in GPU HBM, regional caches in CPU DRAM, origin servers on disk/remote), LMCache caches computed attention states close to where they will be consumed. The "origin server" is the prefill computation itself, which is expensive and should be invoked only when no cached copy exists. The "cache invalidation" problem (when does cached KV become stale?) is solved naturally by token-level addressing: if the tokens change, the cache key changes, and the old entry simply stops being hit.

LMCache turns KV cache from a runtime artifact into a first-class distributed data structure. This single conceptual shift, treating computed attention states as persistent, shareable data rather than ephemeral GPU state, unlocks an entire design space of cache-aware scheduling, disaggregated serving, and prefix-sharing optimizations that would be impossible with engine-internal KV management alone.

## Key Takeaways

1. **KV cache is reusable.** The mathematical output of attention computation for tokens 0..N is deterministic given the same model weights and input tokens. There is no reason to recompute it.

2. **Token-level indexing enables automatic sharing.** By keying on the token sequence itself (not request IDs or session IDs), LMCache discovers sharing opportunities without application-level coordination.

3. **Hierarchical tiers balance latency and capacity.** GPU HBM for hot prefixes (<1ms retrieval), CPU DRAM for warm prefixes (4-8ms), NVMe for cold prefixes (50-100ms). All faster than recomputation for prefixes longer than 256-1,024 tokens.

4. **The KVConnector API decouples cache management from the engine.** vLLM does not need to know about storage tiers, eviction policies, or remote transfers. It calls five methods and LMCache handles the rest.

5. **Production systems already depend on this.** NVIDIA Dynamo, ByteDance AIBrix, and Red Hat llm-d all use LMCache or equivalent external KV management. This is not theoretical; it is deployed infrastructure serving billions of tokens daily.

## Cache Warming Strategies

A cold LMCache provides zero benefit. On server startup or after a deployment, the cache is empty and every request pays full prefill cost. Production systems use three warming strategies to minimize the cold-start penalty:

### Strategy 1: Proactive Warming from Logs

Before a new server enters the load balancer rotation, it replays recent request prefixes from a log (sampled from the last hour of production traffic). The warming process runs prefill on these prefixes and stores the KV tensors in LMCache without generating any output tokens:

```python
async def warm_cache(self, recent_prefixes: List[List[int]], max_warmup_time_s: float = 60.0):
    start = time.time()
    # Sort by frequency (most common prefixes first)
    sorted_prefixes = sorted(recent_prefixes, key=lambda p: p[1], reverse=True)

    for prefix_tokens, frequency in sorted_prefixes:
        if time.time() - start > max_warmup_time_s:
            break
        # Run prefill (no decode, just compute KV)
        kv_tensors = await self.engine.prefill_only(prefix_tokens)
        await self.connector.store(
            request_id="warmup",
            kv_tensors=kv_tensors,
            token_ids=prefix_tokens,
            layer_indices=list(range(self.model.num_layers)),
        )
    logger.info(f"Warmed {len(sorted_prefixes)} prefixes in {time.time() - start:.1f}s")
```

This strategy is effective when the workload is predictable (same system prompts, same RAG documents rotating slowly). It wastes GPU compute on warming but recovers that cost within seconds of serving production traffic with cache hits.

### Strategy 2: Lazy Warming with Admission Boost

Instead of proactively computing KV, this strategy lets the first request for each prefix pay full cost, but aggressively caches the result with elevated priority (so it is not evicted before getting reuse). The admission policy tracks "first-seen" prefixes and gives them a grace period before they become eligible for eviction:

```python
class LazyWarmingAdmissionPolicy:
    def __init__(self, grace_period_s: float = 300.0):
        self.first_seen = {}  # token_hash -> timestamp
        self.grace_period = grace_period_s

    def should_cache(self, token_ids: List[int], access_count: int) -> bool:
        token_hash = hash(tuple(token_ids))
        if token_hash not in self.first_seen:
            self.first_seen[token_hash] = time.time()
            return True  # Always cache on first sight
        return True  # Cache all (eviction handles memory pressure)

    def eviction_eligible(self, token_hash: int) -> bool:
        if token_hash in self.first_seen:
            age = time.time() - self.first_seen[token_hash]
            return age > self.grace_period
        return True
```

### Strategy 3: Cross-Server Cache Migration

When scaling horizontally (adding new servers to handle increased load), the new server can pull hot cache entries from existing servers rather than computing them from scratch. This requires LMCache's remote tier to be accessible across servers (via shared memory, Redis, or RDMA):

1. New server joins the cluster and advertises itself to the cache coordinator.
2. Coordinator identifies the top-K hottest prefixes from its global metadata.
3. New server issues `lookup()` calls against the remote tier to pull these prefixes into its local GPU/CPU tiers.
4. New server enters the load balancer rotation with a warm cache.

This strategy reduces cold-start time from minutes (proactive warming) to seconds (network transfer of pre-computed tensors is faster than recomputing them).

## Mathematical Framework: When Does Caching Win?

The decision to cache a prefix is fundamentally an economic one: does the cost of storing and managing the cached KV tensors pay for itself in saved computation? We can formalize this.

### Cost of Recomputation (No Cache)

For a prefix of length L tokens on a model with P parameters, the prefill cost in FLOPs is approximately:

```
FLOPs_prefill = 2 * P * L  (forward pass, no KV reuse within the prefix)
```

Converting to wall-clock time on hardware with F FLOPS (accounting for MFU):

```
T_prefill = (2 * P * L) / (F * MFU)

Example (Llama-70B, L=2048, A100 at 312 TFLOPS, MFU=0.4):
T_prefill = (2 * 70e9 * 2048) / (312e12 * 0.4) = 2.30s
```

Note: This is the theoretical compute time. In practice, memory bandwidth limits prefill to ~200-400ms for 2048 tokens on A100 because attention is memory-bound at long sequences.

### Cost of Cache Retrieval

For a cache hit at tier T with bandwidth B_T and KV size S:

```
T_retrieval(T) = S / B_T

Where S = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes * L
For Llama-70B (FP16): S = 2 * 80 * 8 * 128 * 2 * L = 327,680 * L bytes
For L=2048: S = 640 MB
```

| Tier | Bandwidth | T_retrieval (640MB) |
|------|-----------|---------------------|
| GPU HBM | 3.35 TB/s | 0.19ms |
| CPU DRAM (PCIe 5.0) | 64 GB/s | 10ms |
| NVMe Gen5 | 14 GB/s | 46ms |
| Remote (100 Gb/s) | 12.5 GB/s | 51ms |

### Break-Even Analysis

Caching wins when:

```
T_retrieval(T) < T_prefill
```

But we also need to account for the opportunity cost of cache memory. Each cached prefix occupies S bytes that could serve active inference. The economic model becomes:

```
cache_value(prefix) = access_frequency * (T_prefill - T_retrieval) - memory_cost_per_second * S
```

Where `memory_cost_per_second` is the dollar cost of holding S bytes in tier T (derived from hardware cost amortized over lifetime). A prefix is worth caching when `cache_value > 0`.

For a system prompt accessed 100 times per second:
```
cache_value = 100 * (0.3s - 0.01s) - negligible_memory_cost
cache_value = 29 GPU-seconds saved per second
```

This means one cached system prompt frees 29 GPU-seconds of compute every second, equivalent to freeing an entire A100 for 29 seconds of decode work every second. The ROI is overwhelming for high-frequency prefixes.

### Minimum Reuse Threshold

Solving for the minimum access frequency where caching breaks even:

```
f_min = memory_cost_per_second * S / (T_prefill - T_retrieval)
```

For most practical configurations (GPU HBM, memory cost dominated by capacity constraints rather than dollar cost), the minimum reuse frequency is approximately 1 access per eviction cycle. In other words: if a prefix will be accessed at least once before it would be evicted, caching it is beneficial.

## Implementation Considerations for Custom Deployments

### Consideration 1: Tensor Serialization Format

LMCache must serialize KV tensors for storage in non-GPU tiers. The format choice affects both storage efficiency and retrieval latency:

- **Raw memcpy**: Zero-copy for GPU and CPU tiers. Fastest possible, but no compression.
- **FP8 quantization**: 50% size reduction with <0.1% perplexity impact (validated by NVIDIA for inference KV). Requires dequantization on retrieval (adds ~0.5ms for 640MB).
- **Token-dropping**: For very long prefixes, drop KV for attention-sink tokens (positions 4-64 in many models carry minimal information). Reduces size by 3-5% with no quality impact.
- **zstd compression**: General-purpose compression on top of tensor bytes. Achieves 20-40% reduction on FP16 KV (tensors have some redundancy across layers). Decompression adds 2-5ms for 640MB.

### Consideration 2: Consistency Under Model Updates

When deploying a new model version (different weights, same architecture), all cached KV tensors become invalid because KV values depend on weight matrices. LMCache must version its cache:

```python
class VersionedCacheKey:
    model_id: str          # e.g., "meta-llama/Llama-3.1-70B-Instruct"
    weight_hash: str       # SHA256 of model weights (first 1MB for speed)
    lora_adapter_id: str   # Empty string if no LoRA
    token_ids: Tuple[int]  # The actual prefix tokens

    def cache_key(self) -> str:
        return f"{self.model_id}:{self.weight_hash[:16]}:{self.lora_adapter_id}:{hash(self.token_ids)}"
```

On model update, the old cache entries naturally stop being hit (different weight_hash). They are garbage-collected by the normal eviction process rather than requiring an explicit flush.

### Consideration 3: Multi-GPU Sharding

With tensor parallelism (TP=4 for Llama-70B), each GPU holds 1/4 of the KV heads. LMCache must store and retrieve shards independently:

- Each TP rank has its own CacheEngine instance.
- `store()` and `lookup()` operate on the local shard only.
- The PrefixTree metadata is shared (replicated) across ranks for consistent cache hit/miss decisions.
- Eviction must be coordinated: if rank 0 evicts a prefix, ranks 1-3 must also evict it (otherwise partial KV retrieval on a future lookup produces incorrect attention output).

```python
class ShardedLMCache:
    def __init__(self, tp_rank: int, tp_size: int, config: LMCacheConfig):
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.local_engine = CacheEngine(config)
        self.eviction_sync = DistributedBarrier(tp_size)  # Ensures coordinated eviction

    def store(self, token_ids: List[int], kv_shard: torch.Tensor):
        # Each rank stores its own shard
        self.local_engine.store(token_ids, kv_shard)

    def evict(self, token_ids: List[int]):
        # Coordinated eviction across all TP ranks
        self.eviction_sync.barrier()  # Wait for all ranks
        self.local_engine.evict(token_ids)
```

### Consideration 4: Handling Variable-Length Responses in Multi-Turn

In multi-turn conversations, the KV cache grows with each turn. After turn N, the cached prefix includes all previous turns plus their responses. But response lengths vary, creating different cache keys for the same logical conversation:

```
Turn 1: [system_prompt] + [user_msg_1]           -> cached prefix A (2048 tokens)
Turn 1 response: 150 tokens (user A) vs 300 tokens (user B)
Turn 2: prefix_A + [response_150] + [user_msg_2] -> different cache key than
         prefix_A + [response_300] + [user_msg_2]
```

This means multi-turn cache reuse only works within the same conversation (same response history). Cross-conversation sharing only applies to the system prompt portion. Production systems handle this by:
- Caching the system prompt independently (shared across all conversations)
- Caching conversation history per-session (shared across turns within one session)
- Not attempting cross-session sharing of conversation-specific KV


## References

- LMCache: KV Cache Management for Efficient LLM Serving (arXiv 2510.09665)
- NVIDIA Dynamo Documentation: KV Cache Routing and Transfer
- vLLM KVConnector Interface: https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/base.py
- ByteDance AIBrix: Open-Source Inference Platform (GitHub: vllm-project/aibrix)
- Red Hat llm-d: Kubernetes-Native LLM Deployment (GitHub: llm-d/llm-d)
- NVIDIA NIXL: Network-Interconnect Transfer Library for KV cache movement
