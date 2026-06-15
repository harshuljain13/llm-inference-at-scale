# System Design: ChatGPT-Scale Conversational AI at 1M Concurrent Users

> **Design Challenge**: You are the infrastructure lead at a company launching a global conversational AI product. Your mandate: serve 1 million concurrent users with ChatGPT-quality responses, sub-second first-token latency, and a cost structure that does not bankrupt the company. Every decision you make here draws on the foundations built throughout Chapters 00 through 08.

---

## 1. Requirements Analysis

### Traffic Profile

Before selecting hardware or models, we must understand what "1M concurrent users" actually means in terms of compute demand. This is not a simple throughput calculation; conversational AI has unique characteristics that shape every downstream decision.

**Concurrency Model:**
- 1,000,000 users with active sessions at any given moment
- Average conversation length: 10 turns (5 user messages, 5 assistant responses)
- Average context window per conversation: 4,096 tokens (accumulating across turns)
- Average generation per response: 256 tokens (typical assistant reply)
- User think time between turns: 30-60 seconds (this is crucial for batching)

**Request Rate Derivation:**

```
Requests/second = 1,000,000 / 45 = ~22,222 requests/second
```

At 256 tokens per response with target ITL of 80ms:

```
Tokens/second globally = 22,222 * 256 = ~5.7 million tokens/second generated
```

This is the raw demand. From Ch07.4, we learned that goodput (useful tokens delivered per second) is what matters, not raw throughput. Some requests will be preempted, some will hit cache, some will be retried. Our system must sustain approximately 5.7M tokens/second of goodput.

**Latency Requirements:**

| Metric | Target (p50) | Target (p99) | Rationale |
|--------|-------------|-------------|-----------|
| Time to First Token (TTFT) | < 300ms | < 1,000ms | User perceives system as "instant" |
| Inter-Token Latency (ITL) | < 50ms | < 150ms | Smooth streaming experience |
| End-to-end (256 tokens) | < 13s | < 40s | Acceptable conversation pace |

**Availability**: 99.9% (three nines) allows ~8.7 hours downtime per year. For a consumer product at this scale, four nines (99.99%) would be ideal but dramatically increases cost. We design for three nines with a path to four.

**Geographic Distribution:**
- North America: 40% of traffic (US-East, US-West)
- Europe: 35% of traffic (EU-West, EU-Central)
- Asia-Pacific: 25% of traffic (AP-Southeast, AP-Northeast)

Each region must independently handle its traffic share, with cross-region failover adding 15% headroom capacity.

**Budget Constraint**: Minimize cost per conversation. A 10-turn conversation generating ~2,560 tokens total should cost less than $0.01 to serve (target: $0.003-0.005/conversation).

---

## 2. Model Selection

### Why This Decision Comes First

From Ch00.0, we know that the model defines everything downstream: memory footprint, compute requirements, quality ceiling, and ultimately cost. The wrong model choice here would cascade into 10x cost differences.

### Candidate Analysis

**Llama 3.1 405B (Eliminated):**

From Ch02.3 (quantization), even at INT4 the 405B model requires:

```
405B parameters * 0.5 bytes (INT4) = ~202 GB for weights alone
```

At 1M concurrent users, the weight memory alone would require distributing across 3+ nodes per instance (even with 80GB GPUs), meaning PP or massive TP. From Ch04.2 (pipeline parallelism), PP introduces bubble overhead of 20-40%. At 22K requests/second, the latency penalty makes this infeasible for our TTFT target.

Cost estimate: ~$1.50/conversation at this scale. Eliminated.

**Llama 3.1 70B INT4 (Selected for Quality Tier):**

```
70B parameters * 0.5 bytes (INT4) = 35 GB weights
```

From Ch02.5, Multi-Latent Attention (MLA) and GQA dramatically reduce KV cache overhead for the 70B class. With GQA (8 KV heads), the KV cache per token is:

```
KV per token = 2 * num_layers * num_kv_heads * head_dim * precision
             = 2 * 80 * 8 * 128 * 2 bytes (FP16 cache)
             = 327,680 bytes = 320 KB per token
```

For a 4K context conversation:

```
KV cache per user = 4,096 * 320 KB = 1.28 GB per active conversation
```

This means on an 80GB A100 with 35GB weights, we have 45GB for KV cache, supporting ~35 concurrent users per GPU. This is workable.

**Llama 3.1 8B INT8 (Selected for Cost Tier):**

```
8B parameters * 1 byte (INT8) = 8 GB weights
```

The 8B model has 32 layers and uses GQA with 8 KV heads:

```
KV per token = 2 * 32 * 8 * 128 * 2 = 131,072 bytes = 128 KB per token
KV cache per user (4K) = 4,096 * 128 KB = 512 MB per conversation
```

On an 80GB GPU: 72GB for KV, supporting ~140 concurrent users. 4x more efficient per GPU.

### Two-Tier Architecture Decision

We deploy both models in a routing architecture (inspired by Ch08.2 Databricks model units):

| Tier | Model | Use Case | % Traffic |
|------|-------|----------|-----------|
| Quality | 70B INT4 | Complex reasoning, code, long context | 20% |
| Cost | 8B INT8 | Simple Q&A, summarization, translation | 80% |

A lightweight router (distilled BERT classifier, <5ms latency) categorizes incoming requests. From Ch08.3 (mixed workload management), we know this tiered approach reduces average cost by 60-70% while maintaining quality where it matters.

**Effective capacity needed:**
- Quality tier: 200K concurrent users on 70B -> ~5,700 GPUs (35 users/GPU)
- Cost tier: 800K concurrent users on 8B -> ~5,700 GPUs (140 users/GPU)
- Total: ~11,400 GPUs before optimization

This baseline gets dramatically reduced through the optimizations in sections 5-7.

---

## 3. Memory Budget

### Per-GPU Memory Allocation

From Ch00.0 (transformer anatomy) and Ch01.1 (GPU memory hierarchy), every byte of GPU memory must be accounted for. Unplanned memory pressure leads to OOM kills or, worse, silent performance degradation from excessive preemption.

**70B Quality Tier (A100 80GB):**

```
+---------------------------------------------------+
| A100 80GB Memory Budget (70B INT4)                |
+---------------------------------------------------+
| Model Weights (INT4):           35.0 GB           |
| KV Cache Pool:                  38.0 GB           |
| Activation Memory:               2.0 GB           |
| CUDA Workspace + Fragmentation:  3.0 GB           |
| Framework Overhead:              2.0 GB           |
| Total:                          80.0 GB           |
+---------------------------------------------------+
| KV Cache Capacity:                                |
|   38 GB / 1.28 GB per user = 29 active users     |
|   (Conservative: budget 25 for headroom)          |
+---------------------------------------------------+
```

**Why 25 and not 35?** From Ch03.1 (continuous batching), we learned that PagedAttention operates with page tables that have their own overhead (~5% of KV pool), and we need slack for incoming prefill requests that temporarily consume extra memory before settling into decode phase.

**8B Cost Tier (A100 80GB):**

```
+---------------------------------------------------+
| A100 80GB Memory Budget (8B INT8)                 |
+---------------------------------------------------+
| Model Weights (INT8):            8.0 GB           |
| KV Cache Pool:                  62.0 GB           |
| Activation Memory:               1.5 GB           |
| CUDA Workspace + Fragmentation:  3.0 GB           |
| Framework Overhead:              1.5 GB           |
| Reserved (prefix cache):         4.0 GB           |
| Total:                          80.0 GB           |
+---------------------------------------------------+
| KV Cache Capacity:                                |
|   62 GB / 0.5 GB per user = 124 active users     |
|   (Conservative: budget 100 for headroom)         |
+---------------------------------------------------+
```

### Aggregate Memory Across the Fleet

**Quality tier total KV memory:**

```
200,000 users * 1.28 GB = 256 TB of KV cache globally
Served by: 256 TB / 38 GB per GPU = ~6,737 GPUs worth of KV storage
But users are batched: 200,000 / 25 per GPU = 8,000 GPUs needed
```

**Cost tier total KV memory:**

```
800,000 users * 0.5 GB = 400 TB of KV cache globally
Served by: 800,000 / 100 per GPU = 8,000 GPUs needed
```

**Critical insight from Ch03.4 (PagedAttention):** Not all users are actively generating at once. With 45-second think time and 3-second generation time, only ~7% of users are in active decode at any instant. The rest have their KV cache parked. This is where PagedAttention's dynamic allocation shines: we can oversubscribe the KV pool by 3-4x safely because idle users' KV pages can be swapped to CPU memory (from Ch03.5 offloading patterns).

**Revised GPU count with oversubscription:**
- Quality tier: 8,000 / 3 = ~2,700 GPUs
- Cost tier: 8,000 / 3 = ~2,700 GPUs
- Total: ~5,400 GPUs (down from 11,400 with naive allocation)

This 2x reduction demonstrates why memory management (Ch03) is the most impactful optimization layer for serving at scale.

---

## 4. Hardware Selection

### GPU Comparison

From Ch01.2 (compute vs memory bound analysis), LLM decode is fundamentally memory-bandwidth bound. The GPU that moves bytes fastest wins for token generation.

| Spec | A100 80GB | H100 80GB | H200 141GB |
|------|-----------|-----------|------------|
| Memory Bandwidth | 2.0 TB/s | 3.35 TB/s | 4.8 TB/s |
| FP16 TFLOPS | 312 | 989 | 989 |
| Memory | 80 GB | 80 GB | 141 GB |
| TDP | 400W | 700W | 700W |
| List Price (cloud) | ~$2/hr | ~$3.5/hr | ~$5/hr |
| Tokens/$ (decode) | Baseline | 1.4x | 2.1x |

**Selection: H100 80GB for Quality Tier, A100 80GB for Cost Tier**

Rationale: The 70B model benefits most from H100's 67% higher memory bandwidth (faster decode per user). The 8B model on A100 already saturates compute per token at high batch sizes, making the H100 premium unnecessary for the cost tier.

From Ch01.1 (memory hierarchy), the H100's increased L2 cache (50MB vs 40MB) helps with the 70B model's larger weight matrices, reducing HBM accesses during decode.

### Fleet Sizing

With H100 for quality tier, higher bandwidth means each token takes less time, allowing higher batch utilization:

```
H100 decode throughput for 70B INT4:
  Model reads per token: 35 GB (full weight scan)
  Time per token at 3.35 TB/s: 35/3350 = 10.4ms per token per user
  Batch of 25 users: still 10.4ms (memory-bound, batch is free)
  Tokens/second per GPU: 25 * (1000/10.4) = ~2,400 tokens/second

A100 decode throughput for 8B INT8:
  Model reads per token: 8 GB
  Time per token at 2.0 TB/s: 8/2000 = 4ms per token per user
  Batch of 100 users: still 4ms (memory-bound)
  Tokens/second per GPU: 100 * (1000/4) = ~25,000 tokens/second
```

**Required fleet:**

```
Global tokens/second needed: 5.7M

Quality tier (20% = 1.14M tokens/s):
  1,140,000 / 2,400 = 475 H100 GPUs (in TP=4 groups = 119 instances)

Cost tier (80% = 4.56M tokens/s):
  4,560,000 / 25,000 = 183 A100 GPUs

Total: 475 + 183 = 658 GPUs minimum for decode
```

The discrepancy from the memory analysis comes from the memory budget being for ALL concurrent users (including thinking ones), while throughput only needs to serve actively generating users. With KV swapping:

```
Active generating users: 1M * (3s gen / 45s cycle) = ~67,000 at any instant
Queued/thinking users: 933,000 (KV in CPU memory, swapped in on demand)

Quality active: 13,400 users / 25 per GPU = 536 GPUs for active decode
Cost active: 53,600 users / 100 per GPU = 536 GPUs for active decode
Swap-in headroom (+20%): 128 additional GPUs
Total: ~1,200 GPUs
```

**Final fleet: ~1,200 GPUs** with aggressive KV management. We budget 1,500 for headroom and failure tolerance (20% spare).

### Cost Estimate

```
Quality: 500 H100s * $3.50/hr = $1,750/hr
Cost:    600 A100s * $2.00/hr = $1,200/hr
Networking/storage/CPU:         = $500/hr
Total:                          = $3,450/hr = $82,800/day

Conversations/day: 22,222 req/s * 86,400s / 10 turns = ~192M conversations/day
Cost per conversation: $82,800 / 192,000,000 = $0.00043
```

This is well under our $0.003 target. The aggressive batching and tiering reduced cost by ~10x from naive allocation.

---

## 5. Parallelism Strategy

### Why TP=4 for 70B, TP=1 for 8B

From Ch04.1 (tensor parallelism), we know TP splits weight matrices across GPUs, enabling models larger than single-GPU memory. The 70B INT4 at 35GB fits on a single 80GB GPU, so why use TP=4?

**Answer: Latency, not capacity.** From Ch04.1, TP reduces per-token latency linearly by distributing the memory bandwidth load:

```
Single GPU decode (70B INT4):
  35 GB / 3.35 TB/s = 10.4ms per token -> ITL = 10.4ms (under 80ms target)

But prefill at 4K tokens:
  Compute-bound: 4096 * 70B * 2 FLOPs / (989 TFLOPS) = 571ms -> TTFT = 571ms (exceeds 500ms!)
```

With TP=4:

```
Prefill: 571ms / 4 = 143ms -> TTFT = 143ms (well within target)
```

TP=4 also helps decode because each GPU only reads 35/4 = 8.75 GB per token:

```
TP=4 decode: 8.75 GB / 3.35 TB/s = 2.6ms per token + NVLink sync (~0.5ms) = 3.1ms
```

This allows deeper batching. We now fit fewer users per GPU set (4 GPUs share KV), but each generates faster.

**Revised capacity per TP=4 instance:**

```
4 H100s = 320 GB total
Weights: 35 GB (distributed, each GPU holds 8.75 GB)
KV cache: each GPU stores full KV for its batch (not split in TP)
Available for KV per GPU: 80 - 8.75 - 5 (overhead) = 66 GB
Users per instance: 66 GB / 1.28 GB = 51 users per 4-GPU instance
Tokens/s per instance: 51 / 3.1ms = ~16,400 tokens/second
```

**8B model: TP=1 is sufficient.**

```
8 GB weight reads in 4ms on A100. Prefill at 4K: 4096*8B*2/312T = 210ms (within target)
```

Single GPU handles everything. No parallelism overhead, maximum efficiency.

### No Pipeline Parallelism

From Ch04.2, PP introduces pipeline bubbles (idle GPU time between micro-batches). At 22K requests/second, we cannot afford the 20-30% throughput loss. TP=4 already makes the 70B model fit our latency targets. PP would only be needed for 405B+ models, which we eliminated.

### Data Parallelism for Scale

From Ch04.3, DP is simply replication. We replicate:
- 119 TP=4 instances of 70B (476 H100s serving quality tier)
- 600 standalone 8B instances (600 A100s serving cost tier)

A request router (Section 6) distributes traffic across replicas. No weight synchronization needed (inference only, no training gradients).

---

## 6. Serving Architecture

### Disaggregated Prefill/Decode

From Ch06.4, disaggregated serving is the most important architectural decision at this scale. The insight: prefill and decode have fundamentally different compute profiles, and mixing them on the same GPU creates interference.

```
Prefill: compute-bound, bursty, one-shot per request
Decode:  memory-bandwidth-bound, steady, iterative
```

**Architecture:**

```
+---------------------------------------------------------------------+
|                    Global Request Router                              |
|  (Route by: model tier, region, session affinity, load)              |
+--------+--------------------------------------------+----------------+
         |                                            |
  +------v--------+                          +--------v--------+
  | Prefill Pool  |                          |  Decode Pool    |
  | (compute-opt) |---- KV Transfer -------->| (memory-opt)    |
  | H100 TP=4     |    (RDMA/NVLink)         | H100 TP=4      |
  | High batch    |                          | Max KV capacity |
  +--------------+                           +-----------------+
```

**Prefill Pool Design:**
- Optimized for compute: pack many prefill requests into large batches
- Each prefill instance processes 4K token prompts and produces initial KV cache
- KV cache transferred to decode pool via RDMA (InfiniBand, ~12.5 GB/s per link)
- Transfer time for one user's KV: 1.28 GB / 12.5 GB/s = 102ms

**Decode Pool Design:**
- Optimized for memory capacity: maximize concurrent users per GPU
- Receives KV caches from prefill pool
- Runs continuous batching (Ch03.1) to keep utilization high
- Session-sticky: a user stays on the same decode instance across turns

**Why disaggregate?** From Ch06.4: a single long prefill (4K tokens) takes 143ms on TP=4 H100. During that time, all decode users on that GPU experience a 143ms stall. At p99, multiple prefills colliding could push ITL above 150ms. Disaggregation eliminates this interference entirely.

### Continuous Batching with vLLM

From Ch03.1, continuous batching (iteration-level scheduling) allows new requests to enter the batch after each decode step, rather than waiting for an entire batch to complete. This is non-negotiable at our scale:

```python
# vLLM-style scheduling (simplified)
while True:
    # Add new requests from queue (up to batch capacity)
    new_reqs = scheduler.get_pending(max=batch_size - len(active))
    for req in new_reqs:
        allocate_kv_pages(req)
        active.append(req)

    # Single forward pass for all active requests
    outputs = model.forward(active_tokens, kv_caches)

    # Remove completed requests, free their KV pages
    for req in active:
        if req.is_done():
            free_kv_pages(req)
            active.remove(req)
            send_response(req)
```

Without continuous batching, throughput drops by 3-5x because the GPU idles waiting for the slowest request in each batch.

### Request Router (from Ch06.6)

From Ch06.6 (cache-aware routing), the router is the brain of the system. It makes three decisions per request:

1. **Tier classification**: Quality (70B) or Cost (8B)?
2. **Region selection**: Route to nearest region with capacity
3. **Instance selection**: Which specific decode instance?

For decision 3, the router uses session affinity. Multi-turn conversations benefit enormously from hitting the same decode instance because their KV cache is already resident (no re-prefill needed):

```
Turn 1: Full prefill (4K tokens) -> 143ms
Turn 2 (same instance): Incremental prefill (new tokens only) -> ~5ms
Turn 2 (different instance): Full re-prefill -> 143ms again
```

Session affinity provides a 28x latency improvement on subsequent turns. From Ch06.6, consistent hashing on conversation_id achieves this while still allowing load balancing:

```python
def route_request(request):
    # Tier classification (lightweight BERT model)
    tier = classifier.predict(request.prompt)  # <5ms

    # Get pool for tier + region
    pool = get_pool(tier, request.region)

    # Session-affine routing with fallback
    if request.conversation_id:
        target = consistent_hash(request.conversation_id, pool)
        if target.has_kv_cache(request.conversation_id):
            return target  # Cache hit: no re-prefill needed

    # New conversation or cache miss: route to lowest-loaded instance
    return pool.least_loaded()
```

---

## 7. Caching Strategy

### Three-Layer Cache Architecture

From Ch06.6 (cache-aware routing and semantic caching), caching at this scale is the difference between profitability and bankruptcy. We implement three cache layers, each addressing different redundancy patterns.

**Layer 1: System Prompt Prefix Caching (Highest Impact)**

Every user shares the same system prompt (e.g., "You are a helpful assistant..."). At 500 tokens, this system prompt's KV cache is identical across all users:

```
System prompt KV cache: 500 tokens * 320 KB/token (70B) = 160 MB
Without prefix caching: computed 22,222 times/second = wasted
With prefix caching: computed once, shared read-only across all users
```

Savings: The prefill pool computes the system prompt KV once and stores it as a read-only prefix. New users "fork" from this prefix, only computing KV for their unique tokens. From Ch03.4 (PagedAttention), this is implemented as shared page references:

```
Prefill savings: 500/4096 = 12% of prefill compute eliminated for every request
At 22K req/s: 12% * 22,222 * 143ms = ~380 GPU-seconds/second saved
Equivalent to: ~380 GPUs worth of prefill eliminated
```

This single optimization removes the need for ~380 prefill GPUs from our fleet.

**Layer 2: Multi-Turn Session KV Persistence**

From Section 6 (session affinity), when a user sends turn N+1, their KV from turns 1..N is already on the decode instance. We extend this with a tiered eviction policy:

```
Hot (GPU HBM):  Active sessions (last message < 60s ago)
Warm (CPU RAM): Recent sessions (last message < 10 min ago)
Cold (NVMe):    Stale sessions (last message < 1 hour ago)
Evicted:        Sessions older than 1 hour (must re-prefill)
```

From Ch03.5, CPU-GPU swap takes ~50ms for a full user context (1.28 GB over PCIe Gen5 at 64 GB/s). This is much cheaper than re-prefilling (143ms of GPU compute).

Swap back latency: 1.28 GB / 64 GB/s = 20ms (PCIe Gen5 x16)

**Layer 3: Semantic Response Cache**

From Ch06.6 (semantic caching), many users ask identical or near-identical questions: "What is the capital of France?", "Explain quantum computing", "Write a hello world in Python". A semantic cache (embedding similarity > 0.95) returns cached responses directly:

```
Cache hit rate estimate: 5-15% of all requests (based on public ChatGPT usage patterns)
At 10% hit rate: 2,222 requests/second served from cache with ZERO GPU cost
Equivalent savings: ~100 GPUs worth of compute
```

Implementation: an embedding model (sentence-transformers, 33M params on CPU) converts each prompt to a 768-dim vector. ANN search (FAISS or ScaNN) finds similar cached prompts in <2ms. On hit, stream the cached response directly.

**Safety**: Only cache responses for prompts with no conversation history (turn 1) and no user-specific context. Multi-turn responses are never cached (context-dependent).

### Total Caching Impact

```
Before caching: ~1,500 GPUs needed
Prefix caching:  -380 GPUs (prefill savings)
Session KV:      -200 GPUs (avoided re-prefills for returning turns)
Semantic cache:  -100 GPUs (requests served without any compute)
After caching:   ~820 GPUs needed
```

Caching reduces our fleet by 45%. From Ch06.6, this is why cache-aware systems dominate inference economics at scale.

---

## 8. Monitoring & SLOs

### SLO Framework

From Ch07.4 (inference metrics and goodput), the metrics that matter for a conversational AI system are fundamentally different from traditional web services. HTTP 200 means nothing if the user experienced a 5-second stall mid-generation.

**Primary SLOs:**

| Metric | Target | Alert Threshold | Page Threshold |
|--------|--------|-----------------|----------------|
| TTFT p50 | < 300ms | > 400ms (5min) | > 800ms (1min) |
| TTFT p99 | < 1,000ms | > 1,200ms (5min) | > 2,000ms (1min) |
| ITL p50 | < 50ms | > 65ms (5min) | > 100ms (1min) |
| ITL p99 | < 150ms | > 180ms (5min) | > 300ms (1min) |
| Goodput | > 95% | < 93% (5min) | < 90% (1min) |
| Error rate | < 0.1% | > 0.2% (5min) | > 0.5% (1min) |

**Goodput Definition (from Ch07.4):**

```
Goodput = (Tokens delivered within SLO) / (Total tokens generated)
```

A token that arrives late (ITL > 150ms p99) or a request that gets preempted and restarted counts against goodput. This single metric captures the user experience better than any combination of traditional metrics.

### Infrastructure Metrics

From Ch07.4, the inference-specific metrics we instrument:

**KV Cache Utilization:**

```python
kv_utilization = allocated_kv_pages / total_kv_pages
# Alert at > 85%: approaching OOM territory
# Page at > 92%: preemptions imminent
```

When KV utilization exceeds 85%, the scheduler must begin preempting lower-priority requests (from Ch08.3, mixed workload management). Our preemption budget is:

```
Preemption rate target: < 5% of requests
Alert: > 5% sustained for 5 minutes
Page: > 10% sustained for 2 minutes
```

**Batch Utilization:**

```
batch_utilization = active_sequences / max_batch_size
Target: > 70% average (high throughput)
Alert if < 50%: GPUs underutilized, possible routing issue
```

**Prefill Queue Depth:**

```
prefill_queue_depth = waiting_prefill_requests
Target: < 100 queued (represents < 5s wait at current throughput)
Alert: > 500 (25s queue time, TTFT SLO at risk)
Page: > 1000 (approaching timeout)
```

**KV Transfer Latency (disaggregated serving):**

```
kv_transfer_p99 < 200ms (RDMA network health)
Alert: > 300ms (possible network congestion)
Page: > 500ms (TTFT SLO blown just from transfer)
```

### Dashboard Design

A production inference dashboard shows three panels:

1. **User Experience Panel**: TTFT distribution, ITL distribution, goodput percentage, active conversations
2. **Infrastructure Panel**: GPU utilization by tier, KV cache pressure, batch sizes, queue depths
3. **Cost Panel**: $/token by tier, cache hit rates, GPU efficiency (goodput per dollar)

### Alerting Philosophy

From Ch07.4, the key insight: alert on leading indicators, not trailing. KV cache at 85% predicts preemptions before they happen. Queue depth rising predicts TTFT degradation before users feel it. Proactive scaling (Section 9) triggers on these leading indicators.

---

## 9. Scaling & Cost

### Multi-Region Deployment (from Ch07.5)

From Ch07.5 (multi-region KV locality), global deployment is not just replication. User conversations must stay within their region for latency, but the system must handle uneven load and regional failures.

**Region Architecture:**

```
+-------------------------------------------------------------+
|              Global Load Balancer (DNS/Anycast)               |
+--------+--------------------------+---------------+----------+
         |                          |               |
   +-----v------+           +------v------+  +-----v------+
   |  US-East   |           |  EU-West    |  | AP-South   |
   |  400K usr  |           |  350K usr   |  |  250K usr  |
   |  ~500 GPUs |           |  ~430 GPUs  |  |  ~310 GPUs |
   |  +15% spare|           |  +15% spare |  |  +15% spare|
   +------------+           +-------------+  +------------+
```

Each region runs independently with its own prefill pool, decode pool, and router. Cross-region communication only happens for:
1. Failover (region goes down, traffic reroutes)
2. Model update propagation (new weights distributed)
3. Semantic cache synchronization (popular answers replicated globally)

From Ch07.5, KV caches are NEVER transferred cross-region (latency too high: 50-100ms RTT). If a user switches regions, they re-prefill from scratch. This is acceptable because region switches are rare (<0.1% of requests).

### Autoscaling Strategy

**Scaling signals (ordered by priority):**
1. Prefill queue depth > 200: scale prefill pool
2. KV utilization > 80%: scale decode pool
3. TTFT p99 > 800ms: emergency scale both
4. Predicted traffic surge (time-of-day model): pre-scale 15min ahead

**Scaling mechanics:**

```
Scale-up: Add pre-warmed instances from hot spare pool (model already loaded)
  Time to serve: 30-60 seconds (KV cache warm-up only)

Scale-down: Drain instance (stop accepting new sessions, wait for existing to complete)
  Time to drain: 5-10 minutes (longest active conversations finish)
  Never kill active sessions.
```

### Cost Optimization Layers

**1. Reserved vs. Spot Mix:**

```
Base load (p10 traffic): Reserved instances (60% discount)
Normal variation (p50-p90): On-demand
Burst (p90-p99): Spot instances with checkpointing

Cost mix:
  Reserved: 50% of fleet * 0.4x cost = 0.20x
  On-demand: 35% of fleet * 1.0x cost = 0.35x
  Spot: 15% of fleet * 0.3x cost = 0.045x
  Effective rate: 0.595x of full on-demand pricing (40% savings)
```

**2. Time-of-Day Optimization:**

Traffic follows strong diurnal patterns (from Ch08.3):

```
Peak: 10am-10pm local time -> full capacity
Trough: 2am-6am local time -> 30% of peak
```

By scaling down during trough: 30% of fleet idle 8 hours/day = 10% daily cost savings.

**3. Model Cascade Savings:**

From Section 2 (two-tier architecture):

```
If 100% traffic on 70B: $0.00043/conversation * 5x (no tiering) = $0.0022
With 80/20 tiering: $0.00043/conversation (already calculated)
Tiering savings: 80%
```

### Final Cost Summary

```
Monthly GPU cost: $3,450/hr * 720 hr * 0.6 (reserved/spot mix) * 0.9 (off-peak)
               = $1,341,360/month
Monthly conversations: 192M/day * 30 = 5.76 billion conversations/month
Cost per conversation: $1,341,360 / 5,760,000,000 = $0.000233

Revenue needed for profitability at $20/user/month:
  Users to break even: $1.34M / $20 = 67,000 paying users (out of millions on free tier)
```

This demonstrates why the ChatGPT model works: massive free tier for growth, small paying tier covers infrastructure.

---

## 10. Failure Modes & Resilience

### GPU Failures

At 1,200 GPUs, with typical failure rates of 2-3% annually (from Ch08.1, Meta's fleet experience with 100K GPUs), we expect:

```
Expected failures: 1,200 * 0.03 / 365 = ~0.1 GPUs/day (one every 10 days)
Impact of single failure: 25-100 users lose their decode instance
```

**Mitigation: Hot Spare Pool**

From Ch08.1 (Meta platform), maintain 5% fleet as hot spares (model loaded, waiting for traffic):

```
Hot spares: 60 GPUs (30 H100, 30 A100)
Recovery time: < 30 seconds (reassign affected sessions to spare)
User impact: Single TTFT spike on next turn (KV cache lost, must re-prefill)
```

**Graceful degradation on multi-GPU failure:**
- 1-3 GPUs down: Hot spare replacement, no user impact
- 4-10 GPUs down: Reduce batch sizes, accept higher latency temporarily
- 10+ GPUs down (catastrophic): Activate cross-region failover

### Network Partition

**Intra-region partition (prefill <-> decode):**
- Impact: KV transfers fail, new sessions cannot start
- Detection: KV transfer latency > 1s for 10 consecutive requests
- Response: Route new requests to instances that can self-prefill (sacrifice disaggregation temporarily)

**Cross-region partition:**
- Impact: One region becomes unreachable from global LB
- Detection: Health check failures from global LB (3 consecutive misses)
- Response: DNS failover to nearest healthy region within 30 seconds
- Capacity: Each region has 15% spare specifically for absorbing failed region's traffic

From Ch07.5, cross-region failover means users lose their conversation KV cache. The system stores conversation text (not KV) in a distributed database (DynamoDB Global Tables), allowing re-prefill from text on the new region. Users experience one slow turn (full re-prefill) then normal latency.

### Model Update Rollout

Deploying a new model version to 1,200 GPUs serving 1M users requires surgical precision:

**Canary Strategy:**

```
Stage 1: 1% of traffic (single instance) -> run 1 hour, compare quality metrics
Stage 2: 5% of traffic -> run 4 hours, compare TTFT/ITL distributions
Stage 3: 25% of traffic -> run 12 hours, compare goodput and user feedback
Stage 4: 100% rollout with instant rollback capability
```

**Eval Gate (from Ch08.3):**

At each stage, an automated eval suite runs:
- Factual accuracy benchmark (100 questions, must match baseline +/-2%)
- Safety benchmark (red-team prompts, zero regression tolerance)
- Latency regression (TTFT/ITL must not degrade >10% at same batch size)
- Memory regression (new model must not use more KV per token)

Any regression: automatic rollback to previous version in < 5 minutes.

**Zero-downtime mechanism:**

From Ch08.2 (Databricks model units), blue-green deployment:
1. Load new model on spare capacity (shadow fleet)
2. Gradually shift traffic from old to new
3. Keep old model loaded for 1 hour after full cutover (instant rollback)
4. Decommission old model, reclaim memory

### KV Cache OOM

The most common runtime failure. When KV utilization hits 95%, the system enters OOM danger:

**Graceful Preemption Protocol (from Ch03.1, Ch08.3):**

```python
def handle_kv_pressure(utilization):
    if utilization > 0.95:
        # Stage 1: Stop accepting new sessions on this instance
        instance.stop_accepting_new()

    if utilization > 0.97:
        # Stage 2: Preempt longest-idle sessions (swap to CPU)
        idle_sessions = sorted(active, key=lambda s: s.last_token_time)
        for session in idle_sessions[:10]:
            swap_to_cpu(session)
            if get_utilization() < 0.90:
                break

    if utilization > 0.99:
        # Stage 3: Emergency preemption (kill lowest-priority requests)
        low_priority = [s for s in active if s.tier == "free"]
        for session in low_priority[:20]:
            terminate_with_retry_header(session)
            # Client retries on different instance
```

**Priority hierarchy for preemption:**
1. Free tier, idle > 30s: preempt first
2. Free tier, active: preempt with retry
3. Paid tier, idle > 60s: swap to CPU
4. Paid tier, active: NEVER preempt (SLA violation)

### Cascading Failure Prevention

From Ch08.3, the most dangerous failure mode is cascade: one component fails, increasing load on others, causing them to fail:

```
Scenario: Prefill pool overloaded -> queue builds -> decode instances starve ->
          KV evictions start -> returning users need re-prefill ->
          more prefill load -> complete system failure
```

**Circuit breaker pattern:**

```
If prefill queue > 1000 requests:
  1. Return 503 to new sessions with Retry-After header
  2. Protect existing active sessions (they still decode normally)
  3. Clients implement exponential backoff
  4. System recovers organically as active sessions complete
```

**Load shedding priority:**
1. Protect active conversations (users mid-sentence)
2. Accept new paid-tier conversations
3. Queue new free-tier conversations
4. Reject with backoff if queue > capacity

---

## Summary: Architecture at a Glance

```
+-------------------------------------------------------------------------+
|                    COMPLETE SYSTEM ARCHITECTURE                          |
+-------------------------------------------------------------------------+
|                                                                         |
|  Users (1M concurrent) -> Global LB -> Regional Router                  |
|                                           |                             |
|                          +----------------+----------------+            |
|                          |                |                |            |
|                    +-----v-----+    +-----v-----+   +-----v-----+      |
|                    | Tier      |    | Session   |   | Semantic  |      |
|                    | Classifier|    | Affinity  |   | Cache     |      |
|                    +-----+-----+    +-----+-----+   +-----+-----+      |
|                          |                |           (hit->respond)    |
|                          v                v                             |
|                 +------------------------------+                        |
|                 |       Prefill Pool           |                        |
|                 | (H100 TP=4, compute-opt)     |                        |
|                 +-------------+----------------+                        |
|                               | KV Transfer (RDMA)                     |
|                               v                                        |
|                 +------------------------------+                        |
|                 |       Decode Pool            |                        |
|                 | (H100/A100, memory-opt)      |                        |
|                 | Continuous batching           |                        |
|                 | PagedAttention + swap         |                        |
|                 +------------------------------+                        |
|                                                                         |
|  Monitoring: Goodput, TTFT, ITL, KV%, Preemption rate                  |
|  Scaling: Queue depth -> prefill, KV% -> decode, predictive pre-scale  |
|  Resilience: Hot spares, region failover, graceful preemption          |
|                                                                         |
+-------------------------------------------------------------------------+
|  KEY NUMBERS                                                            |
|  Fleet: ~1,200 GPUs (500 H100 + 600 A100 + 100 spare)                 |
|  Cost: $0.00023/conversation ($1.34M/month)                            |
|  Latency: TTFT 143ms (prefill) + 20ms (swap) = 163ms p50              |
|  Throughput: 5.7M tokens/second sustained goodput                      |
|  Availability: 99.9% (hot spares + cross-region failover)              |
+-------------------------------------------------------------------------+
```

### Chapter Cross-References Summary

| Decision | Source Chapter | Key Insight Applied |
|----------|--------------|-------------------|
| Memory budget calculation | Ch00.0, Ch01.1 | Weights + KV + overhead = total |
| INT4 quantization choice | Ch02.3 | 2x memory reduction, <1% quality loss |
| GQA KV savings | Ch02.5 | 8 KV heads vs 64 attention heads |
| PagedAttention + swap | Ch03.1, Ch03.4, Ch03.5 | Dynamic allocation + 3x oversubscription |
| TP=4 for latency | Ch04.1 | Prefill speedup, not just capacity |
| Continuous batching | Ch03.1 | Iteration-level scheduling |
| Disaggregated serving | Ch06.4 | Eliminate prefill/decode interference |
| Cache-aware routing | Ch06.6 | Session affinity + semantic cache |
| Goodput metric | Ch07.4 | SLO-aware throughput measurement |
| Multi-region design | Ch07.5 | KV stays local, text replicates |
| Meta fleet patterns | Ch08.1 | Hot spares, failure rates, recovery |
| Databricks model units | Ch08.2 | Two-tier architecture, blue-green deploy |
| Mixed workload preemption | Ch08.3 | Priority-based graceful degradation |

---

*This system design demonstrates that serving LLMs at scale is not a single breakthrough but a composition of techniques from every layer of the stack. Each chapter in this book addresses one piece; this design shows how they combine into a coherent, production-ready architecture.*
