# Cache-Aware Routing and Semantic Prompt Caching

## Why Routing Matters for Inference

When most engineers think about load balancing for LLM serving, they think about distributing requests evenly across GPU nodes to prevent any single machine from becoming overwhelmed. This is the traditional approach inherited from stateless web services: round-robin, least-connections, or weighted random selection. For LLM inference, this mental model is fundamentally incomplete.

The reason is that LLM inference is deeply stateful. Every request that passes through a model leaves behind a KV cache: the computed key and value tensors for each attention layer, for every token in the prompt. This cache represents substantial GPU memory (often gigabytes per request for long prompts) and significant compute (the entire prefill phase). When you route a request to a node that already holds relevant KV cache entries from prior requests with the same prefix, you skip that prefill computation entirely. The request starts generating tokens almost immediately. When you route that same request to a fresh node via round-robin, the node must perform full prefill from scratch, consuming hundreds of milliseconds of GPU time and delaying the first token.

This distinction transforms routing from a load distribution problem into a memory optimization problem. The question is no longer "which node is least busy?" but rather "which node already has the computation I need stored in GPU memory?" This chapter explores how modern serving systems answer that question through prefix-aware routing, session affinity, and semantic caching, building a complete picture of how intelligent routing reduces latency, saves compute, and cuts inference costs.

## Connection to Prior Modules

From Module 2.3, you know that PagedAttention stores KV blocks in GPU memory using a block table that maps logical positions to physical memory slots. Each block holds the keys and values for a fixed number of tokens (typically 16), and these blocks can be allocated non-contiguously. This paged layout is what makes prefix caching possible at all: shared prefix blocks can be referenced by multiple requests simultaneously without duplicating the underlying memory.

From Module 6.4 on disaggregated serving, you understand that modern architectures split prefill and decode across separate node pools. Prefill nodes handle the compute-heavy prompt processing, while decode nodes handle the memory-bound token generation. In this architecture, routing becomes even more critical: the prefill node you select determines whether cached prefix blocks can be reused, directly affecting the Time-to-First-Token (TTFT) that users experience.

From Module 3.4 on continuous batching, you know that GPU utilization depends on keeping the batch full. A request that arrives at a node with a warm cache enters the decode batch almost immediately. A request that requires full prefill occupies the prefill pipeline for hundreds of milliseconds before it can begin contributing to decode throughput. Cache-aware routing therefore improves not just the latency of individual requests but the throughput of the entire system.

## The Problem: Naive Round-Robin Wastes KV Cache

Consider a production deployment with 8 GPU nodes serving a customer support chatbot. The system prompt is 2,048 tokens long, and every request starts with this identical prefix. Under naive round-robin routing:

- Request 1 goes to Node 0. Full prefill of the 2,048-token system prompt. Takes ~410ms (Llama 3.1 70B on A100, batch size 1, ~5,000 tokens/sec during prefill).
- Request 2 goes to Node 1. Full prefill of the same 2,048 tokens. Another ~410ms.
- Request 3 goes to Node 2. Same story. 410ms of redundant computation.
- ...
- Request 8 goes to Node 7. By now, every node has computed the same KV cache independently.

After 8 requests, you have spent 8 x 410ms = 3,280ms of aggregate GPU compute on identical work. Worse, if the system prompt changes (as it does with A/B testing, feature flags, or per-tenant customization), nodes that cached the old prompt must recompute from scratch when they receive a request with the new prefix.

The waste scales with the number of nodes. In a 64-node cluster with round-robin routing, the first 64 requests all perform redundant prefill. At 1,000 requests per second, you are burning 7.7 seconds of GPU time per second on duplicate computation. This is compute you are paying for but receiving zero value from.

The quantitative impact is straightforward to derive. Let P be the prefill time for the shared prefix, N be the number of nodes, and R be the request rate. Under round-robin, the expected cache hit rate for the shared prefix is 0% for the first N requests and approaches (N-1)/N only after each node has seen at least one request. Under prefix-aware routing (all requests to one node), the cache hit rate is (R-1)/R from the second request onward, essentially 100% at scale. The prefill savings per request equal P multiplied by hit_rate. For a 2,048-token prefix on an A100 GPU processing approximately 5,000 tokens/second during prefill, P is roughly 410ms. Eliminating this computation for 99% of requests saves 406ms per request in TTFT.

## Prefix-Aware Routing: Exploiting Shared System Prompts

The first level of cache-aware routing targets exact prefix matches. The insight is simple: if multiple requests share the same starting tokens (system prompt, few-shot examples, RAG context), route them to the same node so the KV cache for that prefix is computed once and reused for all subsequent requests.

### How Prefix Routing Works

The router maintains a prefix table that maps token sequences (or their hashes) to node assignments. When a request arrives:

1. The router extracts the prompt prefix up to a configurable length (e.g., 2,048 tokens).
2. It hashes this prefix to produce a routing key.
3. It looks up the routing key in the prefix table to find which node(s) have cached this prefix.
4. If a match exists, route to that node (checking load constraints).
5. If no match exists, route to the least-loaded node and record the assignment.

```python
import hashlib
from typing import Dict, List, Optional, Tuple

class PrefixRouter:
    """Routes requests to nodes based on prefix cache locality."""

    def __init__(self, num_nodes: int, prefix_block_size: int = 512):
        self.num_nodes = num_nodes
        self.prefix_block_size = prefix_block_size
        # Maps prefix_hash -> (node_id, last_access_time)
        self.prefix_table: Dict[str, Tuple[int, float]] = {}
        # Tracks load per node (pending prefill tokens)
        self.node_load: List[int] = [0] * num_nodes
        self.max_load_threshold = 50000  # tokens

    def hash_prefix(self, tokens: List[int], length: int) -> str:
        """Hash the first `length` tokens to create a routing key."""
        prefix = tokens[:length]
        block_count = length // self.prefix_block_size
        hash_input = str(prefix[:block_count * self.prefix_block_size])
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def route(self, tokens: List[int], timestamp: float) -> int:
        """Select the best node for this request."""
        prefix_hash = self.hash_prefix(tokens, min(len(tokens), 2048))

        if prefix_hash in self.prefix_table:
            node_id, _ = self.prefix_table[prefix_hash]
            if self.node_load[node_id] < self.max_load_threshold:
                self.prefix_table[prefix_hash] = (node_id, timestamp)
                return node_id

        # Fallback: route to least-loaded node
        node_id = min(range(self.num_nodes), key=lambda i: self.node_load[i])
        self.prefix_table[prefix_hash] = (node_id, timestamp)
        return node_id
```

### Production Patterns Where Prefix Routing Excels

Prefix routing delivers the highest savings in workloads with high prefix sharing rates. Empirical measurements from production traces (Mooncake, FAST 2025) show:

- **Multi-turn conversations**: ~40% prefix sharing ratio. Each turn shares the full conversation history with the previous turn. Routing successive turns to the same node avoids recomputing the entire dialogue history.
- **Tool/Agent workloads**: ~59% prefix sharing ratio. Agents repeatedly invoke the same tools with identical system prompts and tool descriptions. Only the final user query differs between requests.
- **RAG pipelines**: The retrieved context (often 4,000-8,000 tokens) is frequently identical for related queries within a time window. Routing these to the same node reuses the context KV cache.
- **Batch few-shot learning**: All requests in a batch share the same few-shot examples. SGLang's benchmarks show up to 5x throughput improvement from RadixAttention cache reuse in few-shot workloads.

## Session Affinity: Multi-Turn Conversations on Warm Nodes

Session affinity extends prefix routing to the temporal dimension. In a multi-turn conversation, each new user message builds on the entire conversation history. If turn N was processed on Node 3, then Node 3 already holds the KV cache for turns 1 through N. Routing turn N+1 to Node 3 means the system only needs to compute KV entries for the new user message and the model's response to turn N, rather than reprocessing the entire conversation from scratch.

### The Mathematics of Session Affinity Savings

Consider a conversation with T turns, where each turn adds approximately L tokens (user message + model response). At turn T, the total context length is approximately T x L tokens.

Without session affinity (random routing):
- Expected prefill time at turn T: O(T x L) tokens multiplied by prefill_rate
- For T=10, L=200: 2,000 tokens x 0.2ms/token = 400ms TTFT

With session affinity (same node):
- Only the new tokens since last turn need prefill: O(L) tokens
- For L=200: 200 tokens x 0.2ms/token = 40ms TTFT
- Savings: 90% reduction in TTFT at turn 10

The savings compound with conversation length. By turn 20, the difference is 4,000 tokens vs 200 tokens of prefill, a 20x reduction. This is why session affinity is critical for chat applications where users expect sub-second responses.

### Implementing Session Affinity

```python
import time
from typing import Dict, List

class SessionAffinityRouter:
    """Routes multi-turn conversations to the same node."""

    def __init__(self, num_nodes: int, session_timeout: float = 3600.0):
        self.num_nodes = num_nodes
        self.session_timeout = session_timeout
        self.session_map: Dict[str, tuple] = {}
        self.node_load: List[int] = [0] * num_nodes
        self.critical_threshold = 80000  # tokens

    def route(self, session_id: str, tokens: List[int]) -> int:
        """Route based on session affinity with load-aware fallback."""
        now = time.time()

        if session_id in self.session_map:
            node_id, last_access = self.session_map[session_id]

            # Check session is still valid (not timed out)
            if now - last_access < self.session_timeout:
                # Check node is still healthy and not critically overloaded
                if self.node_load[node_id] < self.critical_threshold:
                    self.session_map[session_id] = (node_id, now)
                    return node_id

            # Session expired or node overloaded: evict and reassign
            del self.session_map[session_id]

        # New session: assign to least-loaded node
        node_id = min(range(self.num_nodes), key=lambda i: self.node_load[i])
        self.session_map[session_id] = (node_id, now)
        return node_id
```

### The Tension: Session Affinity vs Load Balancing

Session affinity creates a fundamental tension with load balancing. If one user has an extremely long conversation (50+ turns), their assigned node accumulates significant KV cache for that session, consuming GPU memory. Meanwhile, if popular sessions cluster on one node due to hash collisions, that node becomes a hotspot.

The DualMap paper (Yuan et al., 2026) formalizes this tension mathematically. They show that cache-affinity scheduling and load-balancing scheduling operate within a single mapping space, and optimizing one inherently degrades the other. Their solution uses the "power of two choices" principle: map each request to two candidate nodes via independent hash functions, then select the better candidate based on current conditions. This achieves cache hit rates close to pure affinity (within 2/m for m requests with the same prefix) while maintaining load balance bounded by O(log log n) deviation from the mean.

## Semantic Caching: Beyond Exact Prefix Matches

Prefix caching and session affinity handle exact matches: the same tokens in the same order. But what about requests that are semantically equivalent but textually different? Consider these two prompts:

- "Explain how photosynthesis works in simple terms"
- "Can you describe the process of photosynthesis in easy-to-understand language?"

These prompts will produce nearly identical responses, yet they share zero prefix tokens. A prefix cache treats them as completely unrelated. Semantic caching bridges this gap by comparing prompts in embedding space rather than token space.

### How Semantic Caching Works

The architecture introduces an embedding-based lookup layer before the LLM:

1. **Embed the query**: Convert the incoming prompt to a dense vector using a lightweight embedding model (e.g., sentence-transformers, ~100M parameters, <10ms latency).
2. **Search the cache**: Query a vector index (FAISS, Milvus, or similar) for the k-nearest neighbors within a similarity threshold.
3. **Return cached response**: If a sufficiently similar prompt exists in the cache (cosine similarity > threshold), return the cached completion directly without invoking the LLM.
4. **Cache miss**: If no match, forward to the LLM, then store both the embedding and the response in the cache.

```python
import numpy as np
from typing import List, Optional
import time

class SemanticCache:
    """Cache LLM responses indexed by prompt embedding similarity."""

    def __init__(self, embedding_dim: int = 768,
                 similarity_threshold: float = 0.95,
                 max_entries: int = 100_000):
        self.embedding_dim = embedding_dim
        self.threshold = similarity_threshold
        self.max_entries = max_entries

        # FAISS index for fast similarity search
        import faiss
        quantizer = faiss.IndexFlatIP(embedding_dim)
        self.index = faiss.IndexIVFFlat(
            quantizer, embedding_dim,
            min(256, max_entries // 100),
            faiss.METRIC_INNER_PRODUCT
        )

        self.responses: List[str] = []
        self.prompts: List[str] = []
        self.access_times: List[float] = []

    def lookup(self, query_embedding: np.ndarray) -> Optional[str]:
        """Find cached response for semantically similar prompt."""
        if self.index.ntotal == 0:
            return None

        # Normalize for cosine similarity via inner product
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        query_norm = query_norm.reshape(1, -1).astype(np.float32)

        similarities, indices = self.index.search(query_norm, k=1)

        if similarities[0][0] >= self.threshold:
            idx = indices[0][0]
            self.access_times[idx] = time.time()
            return self.responses[idx]

        return None

    def store(self, embedding: np.ndarray, prompt: str, response: str):
        """Cache a new prompt-response pair."""
        if self.index.ntotal >= self.max_entries:
            self._evict_lru(count=self.max_entries // 10)

        embedding_norm = embedding / np.linalg.norm(embedding)
        embedding_norm = embedding_norm.reshape(1, -1).astype(np.float32)

        self.index.add(embedding_norm)
        self.responses.append(response)
        self.prompts.append(prompt)
        self.access_times.append(time.time())
```

### Empirical Results from Semantic Caching

Research on semantic embedding caching (arxiv:2411.05276) demonstrates significant practical benefits:

- **API call reduction**: Up to 68.8% of requests served from cache across diverse query categories.
- **Cache hit accuracy**: Positive hit rates exceeding 97%, meaning cached responses are relevant and useful.
- **Latency improvement**: Cache hits return in <50ms vs 500-2000ms for full LLM inference.
- **Cost savings**: Directly proportional to hit rate. At 68.8% hit rate, inference costs drop by approximately 68.8%.

GPTCache (Zilliz, NLP-OSS 2023) implements this architecture with pluggable components: multiple embedding backends (OpenAI, Cohere, HuggingFace, ONNX, SentenceTransformers), multiple vector stores (FAISS, Milvus, ChromaDB), and configurable similarity thresholds. The modular design allows tuning the precision-recall tradeoff for different use cases.

## Levels of Caching: From Exact to Approximate

Cache-aware routing operates at multiple levels of specificity, each with different tradeoffs between hit rate, precision, and implementation complexity. Understanding these levels helps you choose the right approach for your workload.

### Level 1: Exact Prefix Match (Token-Level)

This is the simplest and most reliable form of caching. Two requests share a cached prefix if and only if their token sequences are identical up to some point.

**Implementation**: vLLM's Automatic Prefix Caching (APC) uses a hash-based block matching system. The KV cache is divided into fixed-size blocks (e.g., 16 tokens per block). Each block is identified by a hash of the tokens it contains plus the hash of the preceding block (creating a chain). When a new request arrives, the system walks the token sequence block by block, checking if each block's hash exists in the cache. The longest matching prefix of blocks is reused.

**Properties**:
- Hit rate: High for system prompts, conversation history, few-shot examples
- False positive rate: Zero. Exact match guarantees correctness.
- Overhead: Negligible. Hash computation is O(n) in prompt length.
- Memory cost: Only the block hash table (16 bytes per block) on the router side.

**When to use**: Always enable this. vLLM and SGLang both support it with zero configuration overhead. It is free performance for any workload with repeated prefixes.

### Level 2: Radix Tree Prefix Match (Structural)

SGLang's RadixAttention extends exact matching with a radix tree data structure that enables efficient prefix search, insertion, and eviction across arbitrary request patterns.

**How the radix tree works**: The tree stores token sequences as paths from root to leaves. Each edge is labeled with a subsequence of tokens (not just a single token, which distinguishes it from a naive trie). Nodes represent points where different requests diverge. The tree supports:

- **Prefix search**: Given a new token sequence, traverse the tree to find the longest matching prefix. This tells you exactly how many tokens of KV cache can be reused.
- **Insertion**: After processing a request, insert the full token sequence (prompt + generated tokens) into the tree. If the sequence shares a prefix with existing entries, the tree is extended from the divergence point.
- **LRU eviction**: When GPU memory pressure requires freeing cache, evict leaf nodes in LRU order. This naturally preserves frequently accessed prefixes (like system prompts) while evicting stale conversation histories.

**Properties**:
- Hit rate: Higher than block-level matching because it handles variable-length prefixes naturally
- Supports multi-program KV reuse: few-shot examples, tree-of-thought branching, self-consistency sampling
- SGLang achieves up to 5x throughput improvement over baseline vLLM on complex LLM programs
- Cache-aware scheduling complements the tree: the scheduler prioritizes requests that share prefixes with cached entries

### Level 3: Semantic Similarity Match (Approximate)

See the Semantic Caching section above for full architecture and FAISS IVF implementation. Key tradeoff:

| Property | Value |
|----------|-------|
| Hit rate | Highest. Captures paraphrases, rewordings, and semantically equivalent queries. |
| Latency overhead | Embedding computation (5-50ms) + vector search (1-10ms for FAISS IVF with 100K entries) |
| Memory cost | Embedding vectors (768-1536 floats per entry) + vector index overhead |

### Level 4: Hybrid (Exact + Semantic)

Production systems often combine levels. The request first checks for an exact prefix match (free, instant). On miss, it checks the semantic cache (5-50ms overhead). Only on double miss does the request proceed to full LLM inference. This layered approach maximizes hit rate while maintaining correctness for the highest-confidence matches.

## Architecture: Load Balancer, Router, and GPU Pool

A complete cache-aware serving architecture has three logical layers, each with distinct responsibilities and state management requirements.

### Layer 1: Load Balancer (Entry Point)

The load balancer handles TLS termination, rate limiting, and coarse-grained traffic distribution. It does NOT make cache-aware decisions. Its role is to direct traffic to the appropriate router cluster. In geo-distributed deployments, the load balancer routes to the nearest region.

### Layer 2: Cache-Aware Router (Intelligence Layer)

The router is the brain of the system. It maintains:

1. **Prefix table**: Maps prefix hashes to node assignments. Updated on every request completion.
2. **Session map**: Maps session IDs to node assignments for multi-turn conversations.
3. **Node health/load state**: Pending prefill tokens, GPU memory utilization, queue depth per node.
4. **Optional: Semantic index**: FAISS or similar for approximate prompt matching.

The router makes per-request decisions based on these signals:

```
                    +-----------------------------+
                    |      Cache-Aware Router     |
                    |                             |
  Request --------->  1. Extract prefix hash     |
                    |  2. Check prefix table      |
                    |  3. Check session map       |
                    |  4. Estimate TTFT per node  |
                    |  5. Select optimal node     |
                    |                             |
                    +-------------+--------------+
                                  |
              +-------------------+------------------+
              v                   v                  v
        +----------+       +----------+       +----------+
        |  Node 0  |       |  Node 1  |       |  Node 2  |
        | (cached  |       | (cached  |       | (least   |
        |  prefix  |       |  prefix  |       |  loaded) |
        |  A, B)   |       |  C, D)   |       |          |
        +----------+       +----------+       +----------+
```

### Layer 3: GPU Pool (Execution Layer)

Each GPU node runs the serving engine (vLLM, SGLang, TensorRT-LLM) with its local KV cache. Nodes report their cache state upstream to the router via lightweight heartbeats containing:
- List of cached prefix hashes and their sizes
- Current GPU memory utilization
- Queue depth and estimated TTFT for next request
- Number of active decode sequences

### The GORGO Model for Routing Decisions

The GORGO paper (Ricci Toniolo et al., 2025) formalizes the routing decision as minimizing estimated TTFT across all candidate nodes:

```
Cost(node) = NetworkLatency(node)
           + t_p * PrefillCost(node)
           + q_s * QueueWaitTime(node)
```

Where:
- `NetworkLatency(node)`: Round-trip time to the node (relevant in geo-distributed deployments)
- `PrefillCost(node)`: Number of tokens that must be recomputed (total prompt length minus cached prefix length), multiplied by per-token prefill time t_p
- `QueueWaitTime(node)`: Estimated wait time based on pending requests in the node's queue

The router selects argmin over all nodes of Cost(node). This formulation makes the tradeoff explicit: a node with high cache hit but long queue may lose to a node with lower cache hit but immediate availability.

### DualMap's Power-of-Two-Choices Approach

For single-region deployments, DualMap (Yuan et al., 2026) offers a more principled approach than simple hash-to-one-node routing. Each request prefix is hashed by two independent functions to produce two candidate nodes. The scheduler then selects between the two based on an SLO-aware strategy:

1. **Default**: Choose the candidate with higher cache hit (preserves affinity).
2. **Under load**: If the cache-affine candidate's estimated TTFT exceeds the SLO threshold, switch to the less-loaded candidate.
3. **Tie-breaking**: If both candidates have equal cache hit rates, choose the less-loaded one.

This achieves a provably tight bound on load imbalance. For m requests mapped to n instances with d=2 choices, the maximum load deviation is bounded by O(log log n), compared to O(sqrt(m log n / n)) for single-choice routing. Experiments show DualMap improves effective request capacity by up to 2.25x under the same TTFT SLO constraints compared to state-of-the-art schedulers.

## Tools and Implementations

### vLLM: Automatic Prefix Caching (APC)

vLLM's APC is enabled by default (since v0.8+) and requires no configuration. The implementation:

1. Divides the KV cache into fixed-size blocks (default 16 tokens per block).
2. Computes a content hash for each block: hash(block_tokens, previous_block_hash). The chaining ensures that two blocks with the same tokens but different preceding context produce different hashes.
3. Maintains a global hash table mapping block hashes to physical memory locations.
4. On new request arrival, walks the prompt tokens block by block, checking the hash table. The longest contiguous sequence of cache hits defines the reusable prefix.
5. Uses an LRU eviction policy when GPU memory is full: blocks not accessed recently are freed first.

**Key characteristics**:
- Zero-cost when there are no cache hits (no overhead from attempting lookup)
- Supports arbitrary prefix lengths (not limited to system prompts)
- Works across requests in the same batch and across batches
- Compatible with tensor parallelism (each TP rank caches its own shard of KV blocks)

### SGLang: RadixAttention with Radix Tree

SGLang's RadixAttention (Zheng et al., 2024) goes beyond block-level caching to support structured reuse patterns:

1. **Radix tree storage**: All cached token sequences are stored in a radix tree on CPU. Each leaf holds a reference to the corresponding KV cache pages on GPU.
2. **Automatic reuse detection**: When a new request arrives, the runtime traverses the radix tree to find the longest matching prefix. No manual annotation required.
3. **Multi-pattern support**: The tree naturally handles diverse reuse patterns:
   - Few-shot examples (shared across all queries in a batch)
   - Multi-turn chat (shared conversation history)
   - Tree-of-thought (shared reasoning prefix, branching at decision points)
   - Self-consistency (shared question, diverging sampled answers)
4. **Cache-aware scheduling**: The scheduler prioritizes requests that can reuse cached prefixes, maximizing GPU efficiency.
5. **LRU eviction on the tree**: Leaf nodes are evicted first in LRU order. This means frequently used prefixes (system prompts) survive while stale completions are freed.

**Performance**: SGLang achieves up to 5x higher throughput than vLLM (v0.2.5) on structured LLM programs that exhibit multiple reuse patterns. Even on simple single-call workloads, RadixAttention provides equivalent or better performance due to the zero-overhead design.

### GPTCache: Semantic Caching Layer

GPTCache (Zilliz, open-source) provides a full semantic caching stack:

1. **Embedding module**: Pluggable. Supports OpenAI embeddings, Cohere, HuggingFace models, ONNX runtime, or SentenceTransformers. Default uses a local ONNX model for zero-cost, zero-latency embedding.
2. **Vector store**: Pluggable. FAISS (local), Milvus (distributed), ChromaDB, or PostgreSQL with pgvector.
3. **Similarity evaluation**: Configurable threshold. Can use cosine similarity, L2 distance, or custom metrics.
4. **Cache storage**: Stores the actual LLM response text. Can use SQLite (local), Redis (distributed), or any key-value store.
5. **Eviction policy**: LRU, LFU, or FIFO on the response cache.

**Integration**: GPTCache wraps LangChain and LlamaIndex with a single-line configuration change. Any existing LLM application can add semantic caching without architectural changes.

### Custom Semantic Cache with FAISS

For teams that need fine-grained control, building a custom semantic cache with FAISS is straightforward:

```python
import faiss
import numpy as np
from typing import List, Optional
import time

class ProductionSemanticCache:
    """Production-grade semantic cache with FAISS backend."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 threshold: float = 0.92, max_size: int = 500_000):
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(model_name)
        self.threshold = threshold
        self.max_size = max_size

        # HNSW index: fast approximate search, good for production
        dim = self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexHNSWFlat(dim, 32)
        self.index.hnsw.efSearch = 64

        self.responses: List[str] = []
        self.metadata: List[dict] = []

    def query(self, prompt: str) -> Optional[str]:
        """Check cache for semantically similar prompt."""
        if self.index.ntotal == 0:
            return None

        embedding = self.encoder.encode([prompt], normalize_embeddings=True)

        scores, indices = self.index.search(
            embedding.astype(np.float32), k=1
        )

        if scores[0][0] >= self.threshold:
            return self.responses[indices[0][0]]
        return None

    def insert(self, prompt: str, response: str, metadata: dict = None):
        """Store a new prompt-response pair."""
        embedding = self.encoder.encode(
            [prompt], normalize_embeddings=True
        )
        self.index.add(embedding.astype(np.float32))
        self.responses.append(response)
        self.metadata.append(metadata or {})
```

## Tradeoffs: Cache Hit Rate vs Staleness

Every caching system faces the fundamental tension between hit rate and correctness. In the context of LLM serving, this manifests as several specific tradeoffs that system designers must navigate.

### Hit Rate vs Memory Pressure

Keeping more KV cache entries in GPU memory increases hit rates but reduces the memory available for active inference. Each cached prefix block consumes the same memory whether it is actively serving requests or sitting idle. For a Llama 3.1 70B model with 80 layers, GQA with 8 KV heads, and 128-dimensional heads per KV head, each token of KV cache consumes:

```
bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
               = 2 * 80 * 8 * 128 * 2  (for FP16)
               = 327,680 bytes ~ 320 KB per token
```

A 2,048-token cached prefix occupies ~640 MB. On an 80GB A100, caching 10 such prefixes consumes 6.4 GB, which is 8% of total GPU memory. This memory cannot be used for active KV cache of in-flight requests. The system must balance between caching more prefixes (higher hit rate) and maintaining headroom for concurrent request processing (higher throughput).

### Staleness and Cache Invalidation

Semantic caches have a unique correctness problem: the world changes, but cached responses do not. If a user asks "What is the current price of Bitcoin?" and the response is cached, returning a week-old cached answer is incorrect. This requires:

- **TTL (Time-to-Live)**: Expire cached entries after a configurable duration. Short TTL = lower staleness, lower hit rate. Long TTL = higher hit rate, higher staleness risk.
- **Content-aware expiration**: Queries containing temporal markers ("today", "current", "latest") should bypass the cache entirely or use very short TTLs.
- **Versioned caching**: When the underlying model or system prompt changes, invalidate all cached entries generated by the previous version.

### Cache Index Memory vs GPU Memory for KV

The semantic cache index itself requires memory. For 100,000 cached entries with 768-dimensional embeddings:

```
Index memory = 100,000 * 768 * 4 bytes (float32) = 307 MB
```

Plus the response storage (assuming average 500 tokens at 4 bytes per token):
```
Response storage = 100,000 * 2,000 bytes = 200 MB
```

Total: ~500 MB for the semantic cache infrastructure. This typically resides on CPU/DRAM (not GPU memory), so it does not compete with KV cache. However, the embedding model used for query encoding does require GPU compute (~5-10ms per query on a small GPU or CPU-only with ONNX). The tradeoff is whether this 5-10ms overhead per request is justified by the cache hit savings (which can save 500-2000ms when successful).

### Affinity vs Load Balance: The Fundamental Tension

The DualMap paper quantifies this precisely. Consider a cluster with n=8 instances:

- **Pure cache affinity**: Cache hit rate approaches 100%, but load CV (coefficient of variation) is high. Popular prefixes create hotspot nodes with 3-5x the average load.
- **Pure least-loaded**: CV approaches 0 (perfect balance), but cache hit rate drops to near the lower bound. Every request misses because it is scattered across nodes.
- **DualMap (power of two choices)**: Cache hit rate within 2/m of optimal (where m is requests per prefix), load deviation bounded by O(log log n). Experimentally achieves 2.25x higher effective request capacity than best-of baselines.

The sweet spot depends on your SLO constraints. If TTFT SLO is strict (e.g., <500ms), cache affinity dominates because prefill savings directly reduce TTFT. If throughput SLO dominates (e.g., sustain 1000 req/s), load balance matters more because hotspot nodes bottleneck the system.

## Metrics for Cache-Aware Routing

Measuring the effectiveness of cache-aware routing requires tracking metrics at multiple levels. These metrics guide threshold tuning and capacity planning.

### Cache Hit Rate

The primary metric. Measured as the fraction of prefill tokens that were served from cache vs computed from scratch:

```
cache_hit_rate = cached_prefix_tokens / total_prompt_tokens
```

Breakdown by hit type:
- **Exact prefix hits**: Tokens matched via hash in the prefix table
- **Semantic cache hits**: Full responses returned without LLM invocation
- **Partial hits**: Some prefix blocks cached, remainder requires computation

Target ranges (based on production workload characteristics):
- System-prompt-heavy workloads: 60-90% hit rate achievable
- Multi-turn chat: 40-70% hit rate (depending on conversation length)
- Diverse one-shot queries: 10-30% hit rate (semantic caching helps here)

### Prefill Savings (Milliseconds)

Directly measures the latency improvement from caching:

```
prefill_savings_ms = cached_tokens * per_token_prefill_time_ms
```

For an A100 at FP16 with Llama 3.1 70B:
- Per-token prefill time: ~0.2ms (at batch size 1)
- A 2,048-token prefix hit saves: 2,048 * 0.2ms = 409.6ms
- A 4,096-token prefix hit saves: 4,096 * 0.2ms = 819.2ms

These are direct TTFT improvements. For interactive applications where TTFT targets are 200-500ms, eliminating a 400ms prefill is the difference between meeting and violating SLO.

### Cost Reduction Per Request

Compute cost scales with FLOPs consumed. Prefill FLOPs for a transformer with L layers, hidden dimension H, and sequence length S:

```
prefill_flops ~ 2 * L * (12 * H^2 * S + 2 * H * S^2)
```

For cached tokens, these FLOPs are zero. The cost saving per request:

```
cost_saved = (cached_tokens / total_tokens) * prefill_cost_per_request
```

At $0.01 per 1M tokens for prefill (approximate GPU-hour cost amortized), a workload processing 1M requests/day with 2,048-token prefixes and 70% cache hit rate saves:

```
savings = 1M * 2,048 * 0.70 * $0.01 / 1M tokens
        = 1M * 0.001434
        = $1,434/day = $43,000/month
```

### Load Balance Coefficient of Variation

Measures how evenly requests are distributed:

```
CV = std(node_loads) / mean(node_loads)
```

Target: CV < 0.3 for acceptable balance. DualMap achieves CV < 0.1 while maintaining >90% of optimal cache hit rate.

### SLO Attainment Rate

The fraction of requests that meet TTFT targets:

```
slo_attainment = requests_within_ttft_slo / total_requests
```

This is the ultimate metric that cache-aware routing optimizes. DualMap demonstrates up to 2.25x improvement in effective request capacity (requests meeting SLO) compared to best baselines, by jointly optimizing cache affinity and load balance.

## Mental Model: Routing as Memory Optimization

The key insight of this chapter is a shift in how you think about request routing. Traditional load balancing treats the problem as: "Given N servers, distribute work evenly." Cache-aware routing treats the problem as: "Given N servers, each with unique cached computation, minimize total redundant work."

This reframing has deep implications:

1. **Routing state matters**: The router must track what each node has cached. Stateless routing is inherently suboptimal for LLM serving.

2. **History determines future performance**: A node that processed many requests with prefix A will be fast for future requests with prefix A. The routing decision today affects the routing quality tomorrow.

3. **Memory is a first-class routing signal**: GPU memory utilization, KV cache occupancy, and cache eviction pressure are all inputs to the routing function, not just load and latency.

4. **The cost of a miss is measurable**: Unlike web caches where a miss adds network latency (tens of milliseconds), a KV cache miss adds prefill compute (hundreds of milliseconds). The penalty is large enough that even modest improvements in hit rate yield significant user-visible improvements.

5. **Caching and routing co-design**: The best systems (SGLang, DualMap) design caching and routing together. The cache structure (radix tree, hash blocks) directly informs the routing algorithm (prefix-aware hashing, power-of-two-choices). Treating them independently leaves performance on the table.

When you design or evaluate an LLM serving system, ask: "For every request that arrives, how much of its computation has already been done somewhere in the cluster?" If the answer is "we do not know" or "we cannot use it even if we knew," you have identified a routing optimization opportunity that can yield 2-5x improvements in TTFT and throughput. The tools (vLLM APC, SGLang RadixAttention, DualMap, GORGO) exist today. The question is whether your routing layer is intelligent enough to exploit them.

## Key Takeaways

1. **Round-robin routing wastes GPU compute.** Every request routed to a cold node recomputes KV cache that may already exist elsewhere in the cluster. For system-prompt-heavy workloads, this waste exceeds 90% of prefill compute.

2. **Prefix-aware routing is free performance.** Both vLLM and SGLang support automatic prefix caching with zero configuration overhead. Enabling it and routing requests with shared prefixes to the same node eliminates redundant computation.

3. **Session affinity compounds over conversation length.** By turn 10, a session-affine request saves 90% of prefill compared to random routing. This is the single most impactful optimization for chat applications.

4. **Semantic caching extends beyond exact matches.** For FAQ-style workloads, embedding-based caching can serve 60-70% of requests without invoking the LLM, reducing costs proportionally.

5. **Cache affinity vs load balance is a solvable tension.** DualMap's power-of-two-choices achieves near-optimal cache hit rates with bounded load imbalance. The solution is not either-or but a principled combination.

6. **Routing is a memory optimization, not just a load balancing decision.** The fundamental mental model shift: GPU memory holding cached computation is the resource being optimized, not just CPU/GPU cycles.

## Production Deployment Considerations

### Warming the Cache on Cold Start

When a serving cluster starts fresh (after deployment, scaling event, or GPU failure recovery), all nodes have empty KV caches. The first wave of requests experiences full prefill regardless of routing strategy. Intelligent systems accelerate cache warming through several techniques:

- **Prioritized prefix pre-computation**: On startup, compute KV cache for known high-frequency prefixes (system prompts, common tool descriptions) before accepting traffic. This front-loads a few hundred milliseconds of compute to avoid repeated cold starts across the first N requests.
- **Cache transfer from DRAM/SSD**: Mooncake (FAST 2025) demonstrates that KV cache can be persisted to host DRAM or NVMe SSD and reloaded on restart. The cost of loading from SSD (~2 GB/s) is far less than recomputing from scratch (~0.2ms per token on GPU). A 2,048-token prefix reload takes ~0.3ms from DRAM vs ~410ms to recompute.
- **Gradual traffic ramp**: During scaling events, new nodes receive traffic gradually. The router sends a small fraction of requests to the new node until its cache warms for frequently-accessed prefixes, then increases the fraction. This avoids the latency spike that would occur from immediately routing 1/N of all traffic to an empty node.

### Handling Cache Eviction Under Memory Pressure

When GPU memory fills (common under high concurrency), the serving engine must evict cached prefixes to make room for active requests. This creates a feedback loop with routing:

1. Node A's cache is full. It evicts a prefix P to serve a new request.
2. The router still maps prefix P to Node A (stale entry in prefix table).
3. Next request with prefix P arrives at Node A, expecting a cache hit, but gets a miss.
4. Node A must recompute prefix P, consuming GPU time that delays other requests.

Mitigations:
- **Heartbeat-based cache state propagation**: Nodes report evictions to the router within 100ms. The router updates its prefix table and may reassign the prefix to a node with available memory.
- **Eviction-aware routing**: When a node reports high memory pressure, the router stops sending new prefix assignments to that node and redirects to nodes with headroom.
- **Two-tier caching**: Keep hot prefixes in GPU memory and warm prefixes in host DRAM. On cache miss for a warm prefix, reload from DRAM (~0.3ms) rather than recomputing from scratch (~400ms). This is the approach taken by Mooncake and NVIDIA NIM's KV cache reuse feature.

### Multi-Tenant Isolation

In multi-tenant deployments, different customers have different system prompts. Cache-aware routing must ensure:

- **No cross-tenant cache leakage**: Tenant A's cached responses must never be served to Tenant B, even if prompts are semantically similar. The semantic cache must be partitioned by tenant ID.
- **Fair cache allocation**: A high-traffic tenant should not evict cache entries belonging to low-traffic tenants. Per-tenant cache quotas or weighted LRU policies prevent this.
- **Prefix isolation**: Even with the same model, different tenants may have different system prompts that should not interfere. The prefix hash includes a tenant identifier to prevent collisions.

### Monitoring and Alerting

Key operational signals for cache-aware routing:

| Metric | Alert Threshold | Action |
|--------|----------------|--------|
| Cache hit rate drop >20% in 5min | Warning | Check for system prompt changes, model updates |
| Node load CV > 0.5 | Warning | Check for prefix hotspots, consider rebalancing |
| TTFT P99 > 2x target | Critical | Check for cache eviction storms, memory pressure |
| Prefix table size > 10M entries | Warning | Check for TTL misconfiguration, unbounded growth |
| Semantic cache false positive rate > 5% | Critical | Raise similarity threshold, audit cached entries |

### Future Directions: Learning-Based Routing

Current cache-aware routers use heuristic cost models (GORGO's linear cost function, DualMap's SLO threshold). An emerging direction is learning-based routing, where the router trains a small model to predict TTFT for each candidate node based on historical patterns:

- **Workload-aware prefix placement**: Instead of reactive caching, proactively replicate high-value prefixes to multiple nodes based on predicted traffic patterns. This is analogous to CDN content placement but for KV cache blocks.
- **Adaptive threshold tuning**: Automatically adjust similarity thresholds, SLO bounds, and eviction policies based on observed hit rates and latency distributions. The GORGO paper notes that its weights t_p and q_s are deployment parameters that could be learned online.
- **Cross-request dependency modeling**: In agentic workloads, the next request is often predictable from the current one (e.g., tool A is usually followed by tool B). A routing system that models these transitions can pre-warm caches on the target node before the next request arrives, achieving zero-latency cache hits.

These techniques move routing from reactive ("where is the cache?") to proactive ("where should the cache be?"), further closing the gap between actual and theoretical optimal TTFT.

The combination of exact prefix caching, semantic similarity matching, session affinity, and intelligent scheduling (DualMap, GORGO) represents the state of the art in 2024-2026. As models grow larger and prompts grow longer, the savings from cache-aware routing will only increase, making this one of the highest-leverage optimizations in the LLM serving stack.

## References and Further Reading

- Zheng et al., "Efficiently Programming Large Language Models using SGLang" (arXiv:2312.07104, 2024). Introduces RadixAttention and the radix tree for automatic KV cache reuse.
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023). Foundation for block-based KV cache management.
- Yuan et al., "DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving" (arXiv:2602.06502, 2026). Formalizes the cache-affinity vs load-balance tradeoff and provides the power-of-two-choices solution.
- Qin et al., "Mooncake: Trading More Storage for Less Computation" (FAST 2025). Production traces showing prefix sharing rates in multi-turn and tool-agent workloads.
- Ricci Toniolo et al., "GORGO: Maximizing KV-Cache Reuse While Minimizing Network Latency in Cross-Region LLM Load Balancing" (arXiv:2602.11688, 2025). Joint optimization of network latency, cache locality, and queue state.
- Bang et al., "Reducing LLM Costs and Latency via Semantic Embedding Caching" (arXiv:2411.05276, 2024). Empirical validation of semantic caching with 68.8% API call reduction.
- GPTCache documentation (gptcache.readthedocs.io). Open-source semantic caching framework with pluggable components.
- vLLM documentation on Automatic Prefix Caching (docs.vllm.ai). Implementation details of block-hash-based prefix reuse.
