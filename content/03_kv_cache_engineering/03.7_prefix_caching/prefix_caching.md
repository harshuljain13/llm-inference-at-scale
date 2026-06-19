# 3.7 Prefix Caching

Every conversation you have with ChatGPT begins identically. Before your message reaches the model, a system prompt of roughly 2000 tokens establishes the assistant's personality, safety guidelines, and behavioral constraints. That system prompt produces the same key-value tensors every single time. Without prefix caching, the serving infrastructure computes those 2000 tokens from scratch for every request, consuming GPU cycles, memory bandwidth, and latency budget on work that produces bit-identical results. Multiply this by thousands of concurrent users and you have a staggering waste: terabytes of redundant memory writes per second, millions of wasted FLOPs, and artificially inflated time-to-first-token for every user.

Prefix caching eliminates this waste through a deceptively simple insight: if two requests share the same token prefix, they produce identical KV cache entries for that prefix. Store those entries once, share them across requests, and you convert O(N) redundant computation into O(1) cached lookup. This is memoization applied to the attention mechanism itself, and it transforms the economics of LLM serving at scale.

This module traces prefix caching from its theoretical foundations through production implementations in vLLM and SGLang, quantifies the savings with concrete arithmetic, examines multi-tenant sharing patterns, and confronts the limitations that constrain real deployments. By the end, you will understand not just how prefix caching works, but when it helps, when it fails, and how commercial providers like Anthropic and OpenAI expose it to end users.


## 1. The Redundant Computation Problem

### 1.1 Anatomy of Repeated Prefixes

Consider a typical multi-turn chat application. Every API call includes:

```
[System Prompt: 2000 tokens] + [Conversation History: variable] + [New User Message: ~50 tokens]
```

The system prompt is identical across all users of the same application. The conversation history grows but shares structure with prior turns. Only the new user message is genuinely novel. Yet the standard inference pipeline treats every request as if it has never seen any of these tokens before.

From Module 03.1, we know that PagedAttention stores KV cache in fixed-size blocks. Each block holds a contiguous chunk of key-value pairs for a sequence of tokens. The critical observation is this: if two requests share the first N tokens, their first N blocks of KV cache are numerically identical. They contain the same floating-point values because the same tokens, processed by the same weights, produce the same activations through the deterministic forward pass.

Prefix caching takes PagedAttention one step further: if two requests share the first N blocks, point them at the same physical memory. Do not allocate separate blocks. Do not compute separate forward passes. Simply share the pointers.

### 1.2 Where Redundancy Occurs in Production

Redundant prefixes appear in multiple patterns:

**Same-application system prompts.** Every user of a chatbot application shares the same system prompt. For GPT-4-class deployments with 1500-3000 token system prompts, this represents the single largest source of redundancy.

**Multi-turn conversation prefixes.** Turn N+1 of a conversation includes all tokens from turns 1 through N. Without caching, the entire history is reprocessed. With caching, only the new tokens require fresh computation.

**Few-shot examples.** Applications using in-context learning prepend the same demonstration examples to every query. A code completion system might include 5 examples (2000 tokens) before each completion request.

**RAG document context.** When multiple users query the same retrieved document, the document tokens produce identical KV entries that could be shared.

**Batch processing pipelines.** Summarization or extraction tasks over a corpus share identical instruction prefixes across all documents.

### 1.3 Quantifying the Waste

For a Llama 3.1 70B model with:
- 80 layers, 8 KV heads (GQA), head dimension 128
- FP16 storage: 2 bytes per element
- Per-token KV size: 2 (K+V) x 80 layers x 8 heads x 128 dim x 2 bytes = 327,680 bytes = 320 KB

A 2000-token system prompt occupies: 2000 x 320 KB = 640 MB of KV cache.

At 1000 requests per second, all sharing the same system prompt, naive serving allocates:
- 1000 x 640 MB = 640 GB of new KV cache per second
- 1000 x 2000 tokens of redundant prefill computation per second = 2 million redundant tokens/second

With prefix caching, that 640 MB is computed once and shared across all 1000 requests. The savings are not incremental; they are transformational.


## 2. How Prefix Caching Works: Core Mechanisms

### 2.1 Hash-Based Prefix Identification

The fundamental question prefix caching must answer is: "Have I seen this exact sequence of tokens before?" The answer comes from hashing.

Given a sequence of tokens, prefix caching computes a hash over contiguous blocks of tokens. The hash function must satisfy three properties:

1. **Deterministic**: the same token sequence always produces the same hash.
2. **Collision-resistant**: different token sequences must not produce the same hash (with high probability).
3. **Incremental**: extending a prefix by one block should be efficient to hash.

The standard approach hashes at the block granularity. If PagedAttention uses blocks of 16 tokens, then the hash for block i depends on the tokens in that block AND the hash of all preceding blocks. This creates a hash chain:

```python
hash_0 = hash(tokens[0:16])
hash_1 = hash(hash_0, tokens[16:32])
hash_2 = hash(hash_1, tokens[32:48])
# ...
hash_i = hash(hash_{i-1}, tokens[i*16:(i+1)*16])
```

Each hash uniquely identifies the entire prefix up to that point. Two requests that produce the same hash for block i are guaranteed (with negligible collision probability) to share the identical first (i+1) blocks of tokens.

### 2.2 The Cache Lookup Flow

When a new request arrives, the serving engine:

1. Tokenizes the input and divides it into blocks of size B (e.g., B=16 tokens).
2. Computes the hash chain for each block.
3. Looks up each hash in the global prefix cache (a hash table mapping hash -> physical block pointer).
4. Finds the longest contiguous prefix where all blocks have cache hits.
5. Reuses those cached blocks (no computation needed).
6. Computes fresh KV only for the remaining suffix tokens.

The time-to-first-token drops proportionally: if 80% of the input is a cached prefix, you skip 80% of the prefill computation.

### 2.3 Copy-on-Write Semantics

Sharing physical blocks introduces a subtle problem: what happens when two requests share a prefix but then diverge? Request A has prefix "The cat sat on the" and request B has the same prefix. Block pointers are shared. But if request A generates the next token "mat" and request B generates "roof," their KV caches diverge from that point forward.

The solution is copy-on-write (CoW), borrowed directly from operating system memory management:

1. Shared blocks carry a reference count tracking how many sequences point to them.
2. While the block is only read (during attention over the cached prefix), sharing is safe.
3. When a sequence needs to write new KV entries into the next block, it gets a fresh private block.
4. The shared prefix blocks are never modified, only read.

This is safe because the KV cache for a prefix is immutable once computed. New tokens extend the cache by appending new blocks; they never modify existing blocks. The append-only nature of autoregressive generation makes copy-on-write particularly natural for KV caching.


## 3. Automatic Prefix Caching in vLLM

### 3.1 Architecture Overview

vLLM implements Automatic Prefix Caching (APC) as a transparent optimization layer within its PagedAttention memory manager. When enabled (via `--enable-prefix-caching`), vLLM automatically detects and deduplicates shared prefixes across all active sequences without any application-level code changes.

The key insight in vLLM's implementation: APC operates at the block level, reusing the same physical block abstraction that PagedAttention already uses for memory management. This means prefix caching adds no new memory allocation primitive; it simply changes when blocks are allocated versus shared.

### 3.2 The Global Block Hash Table

vLLM maintains a global hash table that maps (block_hash -> physical_block_id). When the scheduler processes a new request:

```python
# Simplified view of vLLM APC logic
def allocate_blocks_for_request(token_ids, block_size=16):
    blocks_needed = []
    hash_chain = None

    for i in range(0, len(token_ids), block_size):
        block_tokens = token_ids[i:i+block_size]
        # Hash includes all preceding context (hash chain)
        block_hash = compute_hash(hash_chain, block_tokens)
        hash_chain = block_hash

        # Check global cache
        cached_block = global_cache.lookup(block_hash)
        if cached_block is not None:
            # Reuse existing block (increment ref count)
            cached_block.ref_count += 1
            blocks_needed.append(cached_block)
        else:
            # Allocate new block, compute KV, insert into cache
            new_block = allocate_physical_block()
            blocks_needed.append(new_block)
            global_cache.insert(block_hash, new_block)

    return blocks_needed
```

### 3.3 Eviction Policy

When GPU memory is full and new blocks must be allocated, vLLM evicts cached prefix blocks using an LRU (Least Recently Used) policy. Blocks with ref_count > 1 (actively shared by running sequences) are never evicted. Only blocks with ref_count == 0 (cached but not currently in use by any active sequence) are candidates for eviction.

This creates a natural priority ordering:
1. **Active shared blocks** (ref_count > 1): never evicted, always available.
2. **Active private blocks** (ref_count == 1): evicted only when the sequence completes or is preempted.
3. **Cached-only blocks** (ref_count == 0): eviction candidates, ordered by last access time.

### 3.4 Performance Characteristics

vLLM's APC shines in workloads with high prefix overlap:

- **ChatGPT-like applications**: 60-90% of input tokens are shared system prompts. APC eliminates most prefill computation.
- **Multi-turn conversations**: each turn reuses the entire prior conversation as prefix. Turn 5 of a conversation only computes KV for the new user message.
- **Batch processing with shared instructions**: summarization pipelines where 100 documents share the same "Summarize this document:" prefix.

The overhead of APC is minimal: one hash computation per block (microseconds) and one hash table lookup per block (nanoseconds). The savings, denominated in avoided GPU compute and memory bandwidth, dwarf this overhead by orders of magnitude.

### 3.5 Enabling APC in Production

```bash
# Launch vLLM with APC enabled
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --enable-prefix-caching \
    --block-size 16 \
    --gpu-memory-utilization 0.95

# APC is transparent to clients -- no API changes needed
# The same OpenAI-compatible API works with or without APC
```

There is no client-side configuration. The system automatically identifies and caches shared prefixes. Applications benefit without modification, which is why vLLM calls it "automatic" prefix caching.


## 4. RadixAttention: Tree-Based Prefix Matching in SGLang

### 4.1 Limitations of Flat Hash Tables

vLLM's hash table approach has a structural limitation: it handles exact block-aligned prefix matches. If two requests share 31 tokens (just under 2 blocks of 16), the first block is shared but the second cannot be (it is incomplete). More critically, the flat hash table cannot efficiently answer: "What is the longest cached prefix that matches my input?"

SGLang's RadixAttention solves this with a radix tree (also called a radix trie or patricia trie): a tree data structure optimized for prefix matching.

### 4.2 Radix Tree Structure

A radix tree stores all cached token sequences as paths from root to leaves. Each edge in the tree corresponds to a sequence of tokens (not a single token, which would be a trie). Internal nodes represent shared prefixes, and branches represent divergence points.

```
                    ROOT
                     |
            [System Prompt: 2000 tokens]
                /          \
    [User A history]    [User B history]
        /      \              |
  [Turn 3a]  [Turn 3b]   [Turn 2]
```

When a new request arrives, it traverses the tree from the root, matching its token sequence against edges. The traversal stops at the deepest node where the tokens still match. That node represents the longest cached prefix.

### 4.3 Lookup Algorithm

```python
def find_longest_prefix(radix_tree, token_ids):
    """
    Traverse radix tree to find longest matching prefix.
    Returns (matched_length, cached_kv_blocks).
    """
    node = radix_tree.root
    matched = 0
    cached_blocks = []

    while matched < len(token_ids):
        # Find child edge that matches next tokens
        remaining = token_ids[matched:]
        child = node.find_matching_child(remaining)

        if child is None:
            break  # No matching prefix beyond this point

        # How much of this edge matches?
        edge_tokens = child.edge_label
        match_len = common_prefix_length(remaining, edge_tokens)

        if match_len < len(edge_tokens):
            # Partial edge match -- split node (lazy)
            break

        # Full edge match -- descend
        matched += match_len
        cached_blocks.extend(child.kv_blocks)
        node = child

    return matched, cached_blocks
```

The lookup complexity is O(L / B) where L is the input length and B is the average edge length. In practice, edges compress long shared sequences (the 2000-token system prompt is a single edge), making lookup nearly O(1) for the common case.

### 4.4 Insertion and Eviction

When a request completes and its KV cache should be retained for future reuse, SGLang inserts the full token sequence into the radix tree:

1. Traverse existing tree to find longest match (reusing existing nodes).
2. At the divergence point, split the edge if necessary.
3. Create a new leaf node storing the KV blocks for the novel suffix.

Eviction uses a reference-counted LRU policy similar to vLLM but operating on tree nodes:
- Leaf nodes with ref_count == 0 are eviction candidates.
- Evicting a leaf removes it from the tree. If the parent becomes a single-child node, edges are merged (tree compaction).
- The tree naturally shrinks under memory pressure, retaining the most frequently accessed prefixes.

### 4.5 Advantages Over Hash-Based Approaches

RadixAttention provides three structural advantages:

1. **Sub-block granularity**: matching is token-level, not block-level. A 17-token shared prefix matches 17 tokens, not just the first 16-token block.
2. **Efficient longest-prefix queries**: the tree structure directly answers "what is my longest cached prefix?" without scanning all hash entries.
3. **Natural hierarchy**: the tree reflects the hierarchical structure of real workloads (system prompt -> user -> conversation -> turn).

The tradeoff is implementation complexity: maintaining a concurrent radix tree with reference counting and eviction is significantly more complex than a hash table.


## 5. Savings Calculation: The Economics of Prefix Caching

### 5.1 Per-Request Memory Savings

For a concrete calculation, consider Llama 3.1 8B:
- 32 layers, 8 KV heads (GQA with 4:1 ratio), head dimension 128
- FP16: 2 bytes per element
- Per-token KV: 2 x 32 x 8 x 128 x 2 = 131,072 bytes = 128 KB/token

A 2000-token system prompt occupies:

```
2000 tokens x 128 KB/token = 256 MB per request
```

Without prefix caching, serving 1000 concurrent requests with the same system prompt requires:

```
1000 x 256 MB = 256 GB of KV cache just for system prompts
```

With prefix caching, that same workload requires:

```
1 x 256 MB = 256 MB for the shared prefix (plus per-request unique suffixes)
```

The memory reduction is 1000x for the shared portion. In practice, overall memory savings depend on the ratio of shared to unique tokens.

### 5.2 Compute Savings (Prefill FLOP Reduction)

The prefill phase processes input tokens through all transformer layers. For each token, the FLOPs scale with model dimension:

```
FLOPs per token (approx) = 2 x num_params
Llama 3.1 8B: ~16 GFLOPs per token
```

A 2000-token prefix requires:

```
2000 x 16 GFLOPs = 32 TFLOPs of prefill computation
```

At 1000 RPS, that is:

```
32 TFLOPs x 1000 = 32 PFLOPs/second of wasted computation
```

On an H100 (990 TFLOPS FP16 peak), this wasted computation would require 32 H100 GPUs just to handle the redundant system prompt computation. Prefix caching eliminates this entirely.

### 5.3 Latency Savings (Time-to-First-Token)

Time-to-first-token (TTFT) is dominated by prefill duration. If 80% of input tokens are a cached prefix, TTFT drops by approximately 80%:

```
Without caching: TTFT = prefill(2000 system + 500 user) = prefill(2500 tokens)
With caching:    TTFT = prefill(500 user tokens only)

Speedup: 2500/500 = 5x reduction in TTFT
```

For latency-sensitive applications (chatbots, code completion), this transforms user experience from noticeable delay to instantaneous response.

### 5.4 Throughput Improvement

Prefix caching increases throughput through two mechanisms:

1. **Freed GPU compute**: cycles not spent on redundant prefill can serve additional requests.
2. **Freed GPU memory**: memory not allocated to redundant KV cache can hold more concurrent sequences.

Both effects compound. A system that was GPU-compute-bound on prefill becomes decode-bound (which is memory-bandwidth-bound), unlocking a different scaling regime. A system that was memory-bound can now hold 2-5x more concurrent sequences.


## 6. Multi-Tenant Prefix Sharing

### 6.1 The Multi-Tenant Opportunity

Consider a platform hosting 100 different chatbot applications, each with its own system prompt. Without prefix caching, each user session independently computes and stores the system prompt KV. With prefix caching, all users of the same application share a single cached prefix.

The sharing pattern is hierarchical:

```
Level 0 (global):     Platform safety prefix (shared by ALL requests)
Level 1 (application): Application system prompt (shared by all users of app X)
Level 2 (session):    Conversation history (shared across turns of one session)
Level 3 (request):    Unique per-request tokens (never shared)
```

Each level multiplies the sharing benefit:
- Level 0: 10,000 users share 1 copy = 10,000x savings on safety prefix
- Level 1: 500 users per app share 1 copy = 500x savings on system prompt
- Level 2: 5 turns per session share prefix = 5x savings on conversation history

### 6.2 Cache Hierarchy Design

Production multi-tenant systems implement a tiered cache:

**L1 Cache (per-GPU, fastest):** Holds the hottest prefixes in HBM. For the top 10 applications by traffic, their system prompts live permanently in L1. No eviction pressure because these prefixes are accessed every millisecond.

**L2 Cache (cross-GPU, shared):** For prefixes shared across GPUs in the same node (e.g., via NVLink). Less common applications whose prompts are too cold for every GPU but too hot to recompute on every request.

**L3 Cache (cross-node, persistent):** This is where systems like LMCache (Module 03.4) operate. Prefixes stored in DRAM or SSD on remote nodes, transferred via RDMA when needed. Slower than L1/L2 but faster than recomputation.

### 6.3 Isolation and Security

Multi-tenant prefix sharing raises a critical question: can one tenant's cached KV blocks leak information to another tenant? The answer is no, by construction:

1. KV cache values depend only on the input tokens and model weights. Two users with the same tokens produce identical KV values regardless of who they are.
2. The cache is indexed by token content (hash), not by user identity. There is no way to access another user's unique suffix through the shared prefix.
3. Attention computation uses the shared prefix KV identically to privately computed KV. The math is the same; only the storage is deduplicated.

However, operational isolation still matters: one tenant's traffic pattern should not cause cache evictions that degrade another tenant's hit rate. Production systems implement per-tenant cache quotas or priority tiers.


## 7. Cross-Request vs Cross-Session Caching

### 7.1 Intra-Batch Prefix Sharing (Trivial Case)

The simplest form of prefix caching occurs within a single batch. If the scheduler groups 32 requests that share a system prompt into one batch, it can:

1. Compute the system prompt KV once.
2. Broadcast the result to all 32 sequences in the batch.
3. Compute unique suffixes independently.

This requires no persistent cache infrastructure. The KV blocks are computed fresh for the batch and shared within it. After the batch completes, the blocks are freed. This is "free" optimization with zero cache management overhead.

### 7.2 Inter-Batch Prefix Sharing (The Hard Problem)

The real value comes from sharing across batches, meaning across time. A request at time T computes a system prompt; a request at time T+5 seconds should reuse that computation rather than redoing it.

This requires persistent caching: the KV blocks must survive beyond the lifetime of the batch that created them. The challenges multiply:

**Memory pressure**: keeping cached blocks in GPU memory competes with blocks needed for active sequences. Too aggressive caching reduces batch size; too conservative caching misses reuse opportunities.

**Eviction policy**: when memory is full, which cached prefix should be evicted? LRU is the default, but frequency-based policies may be better for multi-tenant workloads where some applications have bursty traffic.

**Consistency**: if the model weights change (e.g., LoRA adapter swap), cached KV blocks become invalid. The cache must be keyed on (model_version, token_sequence) not just token_sequence.

### 7.3 Cross-Session Caching (LMCache Territory)

The most ambitious form extends caching across user sessions, potentially across server restarts. This is the domain addressed by LMCache (covered in Module 03.4): a persistent, distributed KV cache that stores computed prefixes in DRAM or SSD pools accessible to any serving instance.

Cross-session caching enables:
- A user returning 24 hours later to resume a conversation without recomputing the entire history.
- A newly provisioned GPU instance to immediately serve cached prefixes without warmup.
- Geographic replication of hot prefixes across data centers.

The tradeoff is complexity: network transfer latency for cache fills, consistency protocols for cache invalidation, and storage costs for persisting potentially terabytes of KV data.


## 8. Limitations and Failure Modes

### 8.1 Exact Prefix Match Requirement

The most fundamental limitation: prefix caching requires exact token-level match. A single different token at position i invalidates the cache for all positions >= i. This creates brittle failure modes:

**Timestamp injection**: if the system prompt includes "Current time: 2024-03-15 14:23:07", the prefix changes every second. No caching possible.

**User ID in prompt**: "You are helping user_id=12345" makes every user's prefix unique. Caching becomes per-user rather than per-application.

**Dynamic few-shot selection**: if few-shot examples are selected per-query (e.g., based on semantic similarity), every request has a unique prefix. No sharing occurs.

The mitigation is prompt engineering for cacheability: place static content (system instructions, safety guidelines) before dynamic content (timestamps, user context). This maximizes the shared prefix length.

```
GOOD (cacheable): [Static system prompt 2000 tokens] + [Dynamic context] + [User query]
BAD (uncacheable): [Dynamic timestamp] + [Static system prompt] + [User query]
```

### 8.2 Tokenizer Sensitivity

The same text can produce different token sequences depending on:

1. **Tokenizer version**: updating the tokenizer invalidates all cached prefixes.
2. **Whitespace handling**: trailing spaces, newlines, or encoding differences change tokenization.
3. **Chat template formatting**: different chat template implementations may insert special tokens differently.

A single byte difference in the input string can cascade into completely different token IDs starting from the divergence point, invalidating the entire suffix of the cache.

### 8.3 Cache Eviction Under Memory Pressure

Under high load, the GPU must choose between:
- Holding more active sequences (higher throughput but no caching benefit for future requests).
- Holding more cached prefixes (lower immediate throughput but faster future prefills).

This tension is fundamentally unresolvable without predicting future request patterns. Most systems default to prioritizing active sequences over cached blocks, which means cache hit rates degrade exactly when you need them most (during traffic spikes).

### 8.4 Cache Warmup and Cold Starts

A freshly started serving instance has an empty cache. The first request for each unique prefix must be computed from scratch. This creates:

- **Cold start latency**: first users of a new system prompt experience full prefill latency.
- **Warmup period**: cache effectiveness ramps up over time as common prefixes are computed and stored.
- **Scaling challenges**: adding new GPU instances to handle load spikes provides no immediate caching benefit.

Solutions include pre-warming the cache at startup (compute KV for known system prompts before accepting traffic) and cache migration (transfer cached blocks from existing instances to new ones).

### 8.5 Model Changes Invalidate Cache

Any change to model weights produces different KV values for the same input tokens. This means:

- Model updates require full cache invalidation.
- LoRA adapter swaps require per-adapter cache partitioning.
- Quantization changes (FP16 to FP8) produce different numerical values.

The cache must be keyed on (model_version, adapter_id, quantization_config, token_sequence). This increases key space and reduces effective sharing.


## 9. Commercial Provider Implementations

### 9.1 Anthropic Prompt Caching

Anthropic exposed prefix caching as an explicit API feature in 2024, giving users direct control over what gets cached. The API allows marking specific content blocks with `cache_control` to indicate they should be cached:

```json
{
  "model": "claude-sonnet-4-20250514",
  "messages": [],
  "system": [
    {
      "type": "text",
      "text": "You are a helpful assistant specialized in...",
      "cache_control": {"type": "ephemeral"}
    }
  ]
}
```

Key characteristics of Anthropic's implementation:

- **Minimum cacheable prefix**: 1024 tokens (shorter prefixes are not cached).
- **Cache lifetime**: 5 minutes from last use (ephemeral, not persistent).
- **Pricing**: cache writes cost 25% more than base input tokens; cache reads cost 90% less than base input tokens.
- **Break-even point**: if a cached prefix is reused 3+ times within 5 minutes, caching saves money.

The pricing structure makes caching profitable for any application with moderate request frequency. A chatbot handling even 1 request per minute to the same system prompt benefits.

### 9.2 OpenAI Automatic Caching

OpenAI implements prefix caching automatically (similar to vLLM's APC) without explicit cache_control markers:

- **Minimum cacheable prefix**: 1024 tokens.
- **Automatic detection**: any repeated prefix of sufficient length is cached transparently.
- **Pricing**: cached tokens are billed at 50% of input token price (compared to Anthropic's 90% discount).
- **No explicit API**: users cannot force caching or inspect cache state.

The tradeoff is simplicity versus control. OpenAI's approach requires zero application changes but provides no visibility into cache behavior. Anthropic's approach requires explicit markup but gives developers precise control.

### 9.3 Pricing Economics

The economics of prompt caching for API consumers:

```
Anthropic (Claude Sonnet):
  Base input:     $3.00 / million tokens
  Cache write:    $3.75 / million tokens (1.25x)
  Cache read:     $0.30 / million tokens (0.1x)
  Break-even:     ~4 reuses per cached prefix

OpenAI (GPT-4o):
  Base input:     $2.50 / million tokens
  Cached input:   $1.25 / million tokens (0.5x)
  Break-even:     2 reuses (automatic, no write overhead)
```

For high-volume applications, the savings are substantial. An application making 10,000 requests/day with a 2000-token system prompt:

```
Without caching: 10,000 x 2000 = 20M tokens/day x $3.00/M = $60/day
With caching:    10,000 x 2000 = 20M tokens/day x $0.30/M = $6/day
Savings:         $54/day = $1,620/month = $19,440/year
```

### 9.4 Design Implications for Application Developers

To maximize cache hit rates with commercial APIs:

1. **Front-load static content**: place system prompts, instructions, and examples before any dynamic content.
2. **Minimize dynamic prefix content**: avoid timestamps, random IDs, or per-request metadata in the cacheable prefix region.
3. **Batch similar requests**: group requests with the same prefix to maximize temporal locality.
4. **Monitor cache metrics**: track cache hit rates (where exposed) to validate caching effectiveness.
5. **Respect minimum thresholds**: ensure cacheable prefixes exceed the minimum length (1024 tokens for both providers).


## 10. Advanced Topics

### 10.1 Prefix Caching with Speculative Decoding

Prefix caching interacts with speculative decoding (Module 05.2) in an interesting way. The draft model and target model both process the same prefix, but:

- The draft model's prefix KV is much smaller (smaller model).
- The target model's prefix KV is what gets cached and shared.
- If both models share the same tokenizer and prefix, both can benefit from prefix caching independently.

The combined effect: cached prefix eliminates prefill for both models, making speculative decoding even more efficient because it only needs to handle the novel suffix tokens.

### 10.2 Prefix Caching with Chunked Prefill

From Module 03.5 (Sarathi-Serve pattern): chunked prefill breaks long prefills into smaller chunks interleaved with decode steps. Prefix caching eliminates the need to chunk the cached portion at all:

```
Without prefix cache: chunk([2000 system + 500 user], chunk_size=256) = 10 chunks
With prefix cache:    chunk([500 user only], chunk_size=256) = 2 chunks
```

This reduces the scheduling overhead of chunked prefill by eliminating chunks for the cached portion, improving both latency and scheduling efficiency.

### 10.3 Semantic Prefix Caching (Research Frontier)

Current prefix caching requires exact token match. Research explores approximate or semantic prefix caching:

- **Token-level edit distance**: cache the KV for a similar (but not identical) prefix and apply a correction factor. Explored in academic settings but not production-ready.
- **Learned prefix representations**: train a small model to predict KV values from prefix embeddings, enabling "soft" cache matches. Accuracy concerns limit practical deployment.
- **Prompt normalization**: automatically rewrite semantically equivalent prompts into a canonical form to maximize cache sharing. More practical but limited to simple transformations.

These approaches remain research-stage as of 2025-2026. Production systems rely exclusively on exact prefix matching.


## 11. Implementation Patterns and Best Practices

### 11.1 Prompt Structure for Maximum Cacheability

Design prompts with a clear hierarchy from most-shared to least-shared:

```
[MOST CACHED -- rarely changes]
  Platform safety instructions (shared by ALL applications)
  Application system prompt (shared by all users of one app)
  Few-shot examples (shared by all queries of one type)
  Retrieved documents (shared by queries about same topic)
  Conversation history (shared across turns)
  Current user message (NEVER cached, always unique)
[LEAST CACHED -- changes every request]
```

### 11.2 Monitoring and Observability

Key metrics to track for prefix caching effectiveness:

- **Cache hit rate**: percentage of input blocks served from cache vs computed fresh.
- **Prefix reuse ratio**: average number of requests sharing each cached prefix.
- **Cache memory utilization**: fraction of GPU memory dedicated to cached (ref_count=0) blocks.
- **TTFT reduction**: measured time-to-first-token with vs without caching.
- **Eviction rate**: how frequently cached blocks are evicted before reuse.

A healthy production system shows:
- Cache hit rate > 70% for chat applications with system prompts.
- Eviction rate < 5% for the top-10 most common prefixes.
- TTFT reduction of 3-5x for requests with long shared prefixes.

### 11.3 Configuration Tuning

```python
# vLLM configuration for optimal prefix caching
server_config = {
    "enable_prefix_caching": True,
    "block_size": 16,              # Standard block size
    "gpu_memory_utilization": 0.92, # Leave headroom for cache
    "max_num_seqs": 256,            # Balance concurrency vs cache pressure
    "swap_space": 4,                # GB of CPU swap for evicted blocks
}

# SGLang configuration for RadixAttention
sglang_config = {
    "enable_radix_cache": True,
    "radix_cache_max_memory": 0.3,  # 30% of GPU memory for cache
    "cache_eviction_policy": "lru",
    "min_prefix_len_to_cache": 64,  # Don't cache very short prefixes
}
```


## 12. Mental Model and Key Takeaways

### The Memoization Analogy

Prefix caching is memoization for attention. In traditional programming, memoization stores the result of expensive function calls so that future calls with the same arguments return instantly. Prefix caching does the same for the transformer's forward pass:

- **Function**: transformer attention computation
- **Arguments**: input token sequence
- **Result**: KV cache tensors
- **Cache key**: hash of token sequence (or position in radix tree)
- **Cache invalidation**: model weight changes, memory pressure

If you have seen this context before, do not recompute it. Store the result, share it across all requests that need it, and only compute the novel suffix. This one insight, applied systematically, transforms the economics and latency of LLM serving from O(total_tokens) to O(unique_tokens) per request.

### Decision Framework

| Factor | High Benefit | Low Benefit |
|--------|-------------|-------------|
| System prompt length | > 1000 tokens | < 100 tokens |
| Request rate | > 10 RPS with same prefix | < 1 RPM |
| Prefix stability | Static, unchanging | Dynamic per-request |
| Multi-tenancy | Many users, few apps | One user per prompt |
| Conversation depth | Many turns per session | Single-turn only |

### What to Remember

1. Prefix caching eliminates redundant KV computation for shared token prefixes.
2. vLLM uses hash-based automatic detection; SGLang uses radix tree for efficient longest-prefix matching.
3. Savings scale linearly with prefix length and request volume: 2000 tokens x 1000 RPS = 256 GB/s of avoided writes for Llama 3.1 8B.
4. Exact token match is required. One different token breaks the entire cache chain downstream.
5. Commercial APIs (Anthropic, OpenAI) expose prefix caching with 50-90% cost reduction on cached tokens.
6. Design prompts with static content first, dynamic content last, to maximize cacheable prefix length.


## 13. Prefix Caching in the Broader KV Cache Engineering Landscape

This module completes a progression through KV cache engineering that began with PagedAttention (Module 03.1) and continued through KV compression (Module 03.2), quantized KV (Module 03.3), and distributed KV stores like LMCache (Module 03.4). Each technique addresses a different facet of the same underlying challenge: the KV cache is the dominant memory consumer and the primary bottleneck in LLM serving.

The relationship between these techniques is compositional, not competitive. They form a stack:

- **PagedAttention** solves fragmentation (how to allocate KV blocks efficiently).
- **KV compression** solves size (how to represent KV in fewer bytes per token).
- **Prefix caching** solves redundancy (how to avoid recomputing shared context).
- **Distributed KV stores** solve persistence (how to retain KV across time and nodes).

A production system uses all four simultaneously. PagedAttention manages block allocation. GQA or MQA compresses the KV representation. Prefix caching deduplicates shared blocks. LMCache persists them across sessions. The compound effect is multiplicative: each technique enables the next to operate more efficiently.

Understanding prefix caching in isolation is necessary but insufficient. The system architect must reason about how cache hit rates interact with compression ratios, how eviction policies interact with persistence layers, and how block sizes affect both allocation efficiency and hash granularity. These interactions define the design space of production KV cache engineering, and mastering them is what separates a functioning demo from a system that serves millions of users at acceptable cost and latency.


## References

1. Zheng, L., et al. "Efficiently Programming Large Language Models using SGLang." arXiv:2312.07104, 2023. (RadixAttention)
2. vLLM Documentation. "Automatic Prefix Caching." https://docs.vllm.ai/en/latest/automatic_prefix_caching/apc.html
3. Anthropic. "Prompt Caching." https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching, 2024.
4. OpenAI. "Prompt Caching in the API." https://platform.openai.com/docs/guides/prompt-caching, 2024.
5. Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023. (Foundation for block-based KV caching)
6. Agrawal, A., et al. "Sarathi-Serve: ConcatChunks for Efficient Chunked-Prefill." arXiv:2403.02310, 2024. (Interaction with chunked prefill)
7. Liu, Z., et al. "CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving." SIGCOMM 2024. (Cache compression techniques)
