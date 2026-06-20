# Databricks Multi-Tenant LLM Serving: Model Units, LoRA Multiplexing, and Fleet Economics

When a single organization deploys one model on dedicated hardware, the optimization problem is straightforward: maximize throughput for that model's workload characteristics. Databricks faces a fundamentally harder challenge. They serve thousands of customers on a shared GPU fleet, where each customer runs different models (Llama, Mixtral, DBRX, custom fine-tunes), generates wildly different traffic patterns (batch embedding jobs at 3 AM, interactive chatbots at peak hours), and demands different SLOs (100ms P99 for autocomplete vs. 30s acceptable for document summarization). This chapter dissects how Databricks solved multi-tenant LLM inference at scale, drawing from their public engineering blogs spanning 2024-2026.

The techniques here build directly on foundations from earlier chapters. The continuous batching mechanisms from Chapter 3 become essential when multiplexing requests from hundreds of tenants into shared batch slots. The parallelism strategies from Chapter 5 determine how model replicas are sharded across the fleet. The serving system architectures from Chapter 6 provide the routing and load-balancing substrate. And the inference metrics framework from Chapter 7.4, particularly goodput and SLO attainment, becomes the optimization target rather than raw throughput.

What makes the Databricks story compelling for practitioners is that their solutions are not one-off hacks for hyperscaler-scale problems. The "Model Units" abstraction, fast LoRA serving, and MixAttention architecture each encode a reusable pattern that works at 10 GPUs or 10,000.

---

## 1. The Multi-Tenancy Problem Space

### Why Shared Fleets Are Hard

Traditional cloud computing solved multi-tenancy decades ago with virtual machines: give each tenant an isolated slice of CPU, memory, and disk. GPU inference breaks this model in several ways that make the problem qualitatively different.

First, GPU memory is not fungible the way CPU memory is. A 70B parameter model consumes roughly 140 GB in FP16, which means it occupies an entire 8-GPU node regardless of whether one customer or one thousand customers need it. You cannot give half a model to one tenant and half to another (unless you implement sophisticated model parallelism, which introduces its own coordination overhead as discussed in Chapter 5).

Second, inference workloads have extreme variance in resource consumption. A request generating 10 tokens costs perhaps 50ms of compute, while a request generating 4,096 tokens costs 20 seconds. Batch embedding requests can pack efficiently because all sequences have similar length, while chat completions have high variance in both input and output length. This variance makes static resource allocation wasteful.

Third, the KV cache (Chapter 2) creates a memory reservation problem. Each in-flight request holds KV cache memory proportional to its current sequence length. A long-context request (128K tokens) might hold 16 GB of KV cache for 30+ seconds, blocking that memory from other tenants even if the GPU compute is mostly idle waiting for the autoregressive generation to complete.

### The Naive Approach and Its Failure

The simplest multi-tenant serving strategy is to give each customer a dedicated endpoint: their own model replica on their own GPU(s). This is what Databricks initially offered with their "single-tenant" provisioned throughput endpoints. The problem is economic: if a customer needs 100 tokens/second average throughput but experiences 1,000 tokens/second bursts, you must provision for the burst, leaving 90% of GPU capacity idle on average.

Multiply this across thousands of customers and the fleet utilization becomes catastrophically low. Databricks reported that naive single-tenant provisioning achieved only 15-25% average GPU utilization across their fleet. At current GPU prices (-3/hour per A100), this inefficiency costs millions annually.

The multi-tenant serving challenge is therefore: how do you share GPU resources across tenants while providing each tenant the *illusion* of dedicated capacity with predictable performance?

---

## 2. Model Units: A VM Abstraction for GPU Inference

### The Core Insight

In their June 2026 blog post "Reliable LLM Inference at Scale," Databricks introduced the concept of **Model Units** as their fundamental resource abstraction for GPU inference. The insight is elegant: just as a virtual machine abstracts physical hardware into a portable, schedulable unit with guaranteed resources, a Model Unit abstracts GPU inference capacity into a portable, schedulable unit with guaranteed throughput.

A Model Unit represents a guaranteed amount of inference throughput for a specific model, measured in tokens per second. When a customer purchases N Model Units, they are guaranteed N tokens/second of sustained throughput regardless of what other tenants are doing on the same physical hardware.

### How Model Units Map to Physical Resources

The mapping from Model Units to physical GPUs is not one-to-one. Instead, Databricks uses a bin-packing algorithm that considers:

1. **Model memory footprint**: How much GPU memory does the model (weights + KV cache headroom) require?
2. **Compute density**: How many tokens/second can one GPU produce for this model?
3. **Isolation requirements**: Can this tenant tolerate co-location with other tenants on the same GPU?

For a concrete example, consider serving Llama 3.1 8B on A100 80GB GPUs:
- Model weights (FP16): ~16 GB
- KV cache headroom per concurrent request (4K context): ~0.5 GB
- Maximum concurrent requests per GPU: ~100 (limited by KV cache budget of ~60 GB)
- Peak throughput per GPU: ~4,000 output tokens/second with continuous batching

If Databricks defines one Model Unit as 100 tokens/second for Llama 3.1 8B, then a single A100 can host up to 40 Model Units. A customer purchasing 5 Model Units gets a guaranteed 500 tokens/second, backed by approximately 1/8 of a GPU's capacity.

### Statistical Multiplexing

The power of the Model Units abstraction comes from statistical multiplexing. Not all tenants hit their peak simultaneously. If each of 40 tenants on a GPU has 100 tokens/second guaranteed but their actual usage averages 30 tokens/second with occasional bursts to 150, the GPU can serve all 40 tenants comfortably most of the time while having burst capacity for the occasional spikes.

This is analogous to how airlines overbook flights or how ISPs oversubscribe bandwidth. The key engineering challenge is handling the rare cases when too many tenants burst simultaneously. Databricks addresses this through:

- **Priority queuing**: Tenants exceeding their Model Unit allocation get lower priority than tenants within their allocation
- **Burst pools**: A fraction of fleet capacity is reserved for handling burst traffic without impacting guaranteed throughput
- **Graceful degradation**: When contention occurs, latency increases smoothly rather than causing request failures

### Scheduling and Placement

The Model Units scheduler must solve a variant of the bin-packing problem: place Model Unit allocations onto physical GPUs such that:

1. No GPU is overcommitted beyond its physical capacity
2. Tenants with strict latency SLOs are not co-located with batch workloads
3. The same model running for different tenants shares weight memory (only KV caches are tenant-specific)
4. Failover capacity exists so that a GPU failure does not violate Model Unit guarantees

This last point is critical. If a customer has 10 Model Units backed by a single GPU, and that GPU fails, the customer's guaranteed throughput drops to zero until the system migrates their allocation to another GPU with the same model loaded. Databricks handles this through N+1 redundancy at the model level: for every N GPUs serving a model, at least one standby GPU has the model weights pre-loaded and ready to accept traffic.

---

## 3. Fast PEFT Serving: Hundreds of LoRA Adapters on One Base Model

### The LoRA Multiplexing Opportunity

In their February 2025 blog post on fast PEFT serving, Databricks described their system for serving hundreds of LoRA (Low-Rank Adaptation) adapters on a single base model instance. This is a natural fit for multi-tenant serving because many customers fine-tune the same base model (Llama, Mixtral) with their own data, producing adapters that differ only in a small set of low-rank matrices.

Recall from fine-tuning literature that a LoRA adapter modifies a weight matrix W as:



Where B is (d_model x rank) and A is (rank x d_model), with rank typically 8-64. For a 7B parameter model with rank 16, a LoRA adapter adds only ~33M parameters (0.5% of the base), consuming roughly 66 MB in FP16.

### Memory Architecture for Multi-LoRA Serving

The memory layout for serving N LoRA adapters on one GPU looks like:



For an A100 80 GB serving a 7B base model:
- Base weights: 14 GB
- 100 LoRA adapters loaded: 6.6 GB
- Available for KV cache: ~55 GB
- Maximum concurrent requests: ~110 (at 4K context each)

The critical insight is that LoRA adapters are tiny compared to both the base model and the KV cache budget. Loading 100 adapters costs only 6.6 GB, a fraction of what the base model alone requires. This means the marginal cost of adding another tenant (with their own adapter) is negligible in memory terms.

### Request Routing and Adapter Selection

When a request arrives, the serving system must:

1. Identify which adapter the request targets (from the endpoint URL or model name)
2. Verify the adapter is loaded in GPU memory (if not, trigger a load from host memory or storage)
3. Route the request to a GPU that has both the base model and the target adapter loaded
4. During attention and MLP forward passes, apply the correct adapter matrices

The routing challenge becomes interesting at scale. With 500 adapters distributed across 20 GPUs, not every GPU has every adapter. The router must balance:
- **Load balance**: Spread requests evenly across GPUs
- **Adapter affinity**: Route requests to GPUs that already have the target adapter loaded
- **Cache locality**: Prefer GPUs where the request can benefit from prefix caching (Chapter 6)

Databricks solves this with a two-level routing strategy:
1. **Adapter placement layer**: Decides which GPUs host which adapters based on traffic patterns
2. **Request routing layer**: For each request, selects the best GPU considering load, adapter availability, and cache state

### Batching Across Adapters

A key performance optimization is batching requests across different adapters. Since the base model forward pass is identical for all adapters, requests targeting different adapters can share the same batch for most of the computation. Only the LoRA delta computation differs.

The computation flow for a mixed-adapter batch:



This approach achieves near-linear scaling with batch size regardless of adapter diversity, because the LoRA computation is a small fraction of total FLOPs. For a 7B model with rank 16, the LoRA delta computation is approximately 0.5% of the total forward pass FLOPs, making the per-adapter overhead negligible.

### Adapter Hot-Swapping

Not all adapters can fit in GPU memory simultaneously. Databricks implements an LRU (Least Recently Used) eviction policy for adapters:

- Frequently-accessed adapters remain resident in GPU HBM
- Rarely-accessed adapters are evicted to host DRAM (loading from host takes ~10ms for a 66MB adapter over PCIe 5.0)
- Very cold adapters may be evicted entirely to SSD (loading takes ~100ms)

The latency impact of adapter loading is managed through prefetching. When the routing layer observes traffic ramping up for a cold adapter, it proactively loads the adapter before requests start queuing. This prefetching strategy connects back to the predictive cache warming concepts discussed in Chapter 6's serving systems.

---

## 4. MixAttention: Inference-Friendly Architecture for Variable Context

### The Short/Long Context Dilemma

In their September 2024 blog post, Databricks introduced MixAttention, an architecture designed to handle both short-context (< 4K tokens) and long-context (32K-128K tokens) workloads efficiently on the same infrastructure. This is essential for multi-tenant serving because different customers have radically different context length distributions.

The fundamental tension is:

- **Short-context optimization** favors large batch sizes (many concurrent requests), low KV cache per request, and compute-bound operation where GPU ALUs are fully utilized
- **Long-context optimization** favors small batch sizes (few concurrent requests), large KV cache per request, and memory-bandwidth-bound operation where data movement dominates

A traditional transformer architecture forces you to choose: either optimize your serving infrastructure for short context (high throughput, low per-request cost) or long context (high capability, high per-request cost). MixAttention removes this forced choice.

### Architecture Design

MixAttention uses a hybrid attention pattern where different layers use different attention mechanisms:



The key insight: sliding window layers only need KV cache for the most recent W tokens (where W is the window size). Full attention layers need KV cache for all tokens. By placing sliding window attention in the lower layers and full attention in the upper layers, MixAttention achieves:

1. **Reduced KV cache for short requests**: Short sequences (< 4K tokens) fit entirely within the sliding window, so all layers use the same amount of KV cache regardless of the attention type
2. **Reduced KV cache for long requests**: Only the full-attention layers (50% of layers in this example) maintain the full context length in their KV cache. Sliding window layers cap at W entries regardless of total sequence length.
3. **Quality preservation**: The full-attention layers in the upper portion of the network can still attend to all tokens, preserving long-range dependency modeling quality

### Memory Savings Quantified

For a 32-layer model with GQA (8 KV heads), hidden dim 4096, and head dim 128:

**Standard full attention (all 32 layers), 128K context:**


**MixAttention (16 sliding window + 16 full attention), 128K context:**


This is a 48% reduction in KV cache memory, which directly translates to either:
- Serving more concurrent long-context requests on the same hardware, or
- Freeing memory for additional short-context requests in a mixed workload

### Implications for Multi-Tenant Scheduling

MixAttention transforms the multi-tenant scheduling problem by making long-context requests less expensive to co-locate with short-context requests. In a standard architecture, one 128K-context request might consume 16.8 GB of KV cache, displacing 30+ short-context requests from the same GPU. With MixAttention, that long-context request consumes only 8.7 GB, displacing roughly half as many short-context requests.

This means the Model Units scheduler has more flexibility in placement decisions. A tenant running long-context workloads no longer requires as much dedicated GPU memory, making it easier to pack diverse workloads onto shared hardware without violating SLO guarantees.

---

## 5. Multi-Tenancy Challenges: Noisy Neighbors and Fair Scheduling

### The Noisy Neighbor Problem in GPU Inference

In CPU-based multi-tenant systems, the noisy neighbor problem manifests as cache pollution, memory bandwidth contention, and context switching overhead. In GPU inference, the problem is more severe because GPUs have less hardware-level isolation between concurrent workloads.

The primary noisy neighbor vectors in multi-tenant LLM serving are:

**KV Cache Pressure**: A tenant with many concurrent long-context requests can exhaust the KV cache pool, forcing other tenants' requests to wait for memory even when GPU compute cycles are available. This creates a situation where one tenant's memory-heavy workload degrades latency for all co-located tenants.

**Batch Slot Monopolization**: Continuous batching (Chapter 3) dynamically adds requests to running batches. If one tenant floods the system with requests, they can dominate the batch composition, causing other tenants' requests to wait longer for batch admission. Even with fair queuing, large batches from one tenant slow down the entire batch iteration because each forward pass takes longer with more sequences.

**Prefill Interference**: The prefill phase (processing the input prompt) is compute-intensive and memory-bandwidth-intensive. A tenant submitting a 100K-token prompt creates a prefill operation that monopolizes the GPU for hundreds of milliseconds, during which other tenants' decode iterations are stalled.

### Databricks' Isolation Mechanisms

Databricks addresses noisy neighbors through multiple layers of isolation:

**1. KV Cache Budgets**: Each tenant (Model Unit allocation) receives a guaranteed KV cache budget. Even if the overall KV cache pool has free space, a tenant cannot exceed their budget. This prevents one tenant's long-context workload from starving others.



**2. Scheduling Quanta**: Rather than processing all of one tenant's requests before moving to another, Databricks uses fair-share scheduling with quanta. Each tenant gets a guaranteed number of batch slots per scheduling quantum (e.g., per 100ms). If a tenant has more requests than their quantum allows, excess requests queue until the next quantum.

**3. Prefill Chunking**: Long prefills are broken into chunks of at most C tokens (typically 512-2048). Between chunks, the scheduler yields to other tenants' decode operations. This ensures that even a 100K-token prefill does not create a 500ms pause for other tenants.

This prefill chunking connects directly to the chunked prefill concept covered in Chapter 3's batching discussion. The multi-tenant dimension adds priority-awareness: which tenant's prefill chunks get scheduled first depends on their SLO urgency and remaining budget.

**4. Priority Classes**: Tenants are assigned priority classes based on their SLO tier:
- **P0 (latency-critical)**: Interactive chat endpoints with <2s TTFT guarantees. These get preemptive scheduling priority.
- **P1 (throughput-oriented)**: Batch processing with throughput SLOs. These fill in gaps when P0 requests are not saturating the GPU.
- **P2 (best-effort)**: Free tier or burst-beyond-allocation requests. These only execute when P0 and P1 headroom exists.

### SLO Isolation Through Admission Control

The strongest isolation guarantee comes from admission control: simply refusing to co-locate incompatible workloads on the same GPU. Databricks classifies workloads along two dimensions:

| Dimension | Values |
|-----------|--------|
| Latency sensitivity | Real-time (< 2s), Interactive (< 10s), Batch (minutes) |
| Context length | Short (< 4K), Medium (4K-32K), Long (32K+) |

Workloads in the same cell of this matrix are compatible for co-location. Workloads in different cells may interfere. For example, a real-time short-context chatbot should not share a GPU with a batch long-context summarization workload, because the summarization job's KV cache consumption and long decode times would create unacceptable latency variance for the chatbot.

---

## 6. Superhuman Case Study: 200K QPS at 60% Throughput Gains

### Partnership Context

In their June 2026 blog, Databricks described their collaboration with Superhuman (the email client) to build a 200K QPS inference platform. Superhuman uses LLMs pervasively: email composition, summarization, search, auto-categorization, and reply suggestions. Each of these features has different model size, latency, and throughput requirements, making it an ideal case study for multi-tenant serving on shared infrastructure.

### Architecture Decisions

The Superhuman deployment illustrates several key architectural choices:

**Model Diversity**: Rather than using one large model for everything, Superhuman deploys a family of models:
- Small models (1-3B parameters) for low-latency tasks: auto-complete, categorization
- Medium models (7-13B parameters) for quality-sensitive tasks: email composition, summarization
- Large models (70B+ parameters) for complex reasoning: search relevance, multi-hop queries

**Shared Base Models with Task-Specific Adapters**: For the medium model tier, Superhuman uses a single base model (likely Llama-family) with task-specific LoRA adapters for each feature. This directly leverages the fast PEFT serving infrastructure described in Section 3.

**Tiered Latency Targets**:
- Auto-complete: < 100ms P99 (user is typing, any perceptible delay breaks flow)
- Summarization: < 2s P99 (user opens email, summary should appear quickly)
- Search: < 500ms P99 (user expects near-instant results)
- Composition: < 5s P99 (user clicks "compose reply," expects a draft within seconds)

### How 60% Throughput Gains Were Achieved

The 60% throughput improvement (compared to Superhuman's previous serving infrastructure) came from three sources:

**1. Consolidation onto shared fleet (25% gain)**: Previously, each feature had dedicated GPU capacity provisioned for peak load. Consolidating onto Databricks' shared fleet allows statistical multiplexing: auto-complete peaks in the morning, summarization peaks when inboxes are full (Monday morning), composition peaks in the afternoon. These non-overlapping peaks mean the shared fleet needs less total capacity than the sum of individual peak requirements.

**2. Continuous batching with mixed requests (20% gain)**: By batching requests from different features (different models/adapters) that happen to target the same base model, the system achieves higher GPU utilization per forward pass. A batch might contain 32 auto-complete requests (short output) alongside 8 composition requests (long output), with the auto-complete requests completing quickly and freeing batch slots for new arrivals.

**3. MixAttention and KV cache efficiency (15% gain)**: The reduced KV cache footprint from MixAttention allows more concurrent requests per GPU, which increases throughput per GPU. With standard attention, a GPU might handle 64 concurrent requests; with MixAttention, it handles 90+.

### Scaling to 200K QPS

Reaching 200K queries per second requires careful capacity planning:

- Small models on A10G GPUs (cost-efficient for low-latency, low-compute tasks): ~50 GPUs
- Medium models on A100 GPUs (balance of throughput and capability): ~30 GPUs
- Large models on H100 GPUs (maximum throughput per dollar for large models): ~20 GPUs
- Routing and load balancing: Distributed across regions for latency minimization

The total GPU footprint is approximately 100 GPUs, serving 200K QPS. This translates to roughly 2,000 QPS per GPU on average, which is achievable because many queries are against small models with short outputs.

---

## 7. Provisioned Throughput: Guarantees Without Over-Provisioning

### The Provisioned Throughput Contract

Databricks' Provisioned Throughput endpoints offer customers a middle ground between fully shared (best-effort, unpredictable latency) and fully dedicated (guaranteed but expensive). The contract is:

- **Guaranteed throughput**: Customer is guaranteed X tokens/second sustained
- **Latency SLO**: P99 latency will not exceed Y milliseconds under normal load
- **Burst allowance**: Customer can temporarily exceed X tokens/second by up to 2x for Z seconds
- **No cold start**: The model is always loaded and ready (unlike serverless endpoints that may scale to zero)

### Implementation: Reservation Plus Overbooking

Internally, Provisioned Throughput is implemented through the Model Units abstraction with a reservation layer on top:



The 1.5x overbooking ratio is sustainable because customer traffic is bursty: at any given moment, most customers are below their guaranteed throughput. Databricks monitors the ratio between sold capacity and actual peak concurrent demand, adjusting the overbooking ratio dynamically based on observed traffic patterns.

When the overbooking ratio is exceeded (too many customers peaking simultaneously), the system falls back to strict priority enforcement: each customer gets exactly their guaranteed throughput, with no burst allowance, until contention subsides.

### Cost Efficiency Compared to Dedicated Endpoints

The economic advantage of Provisioned Throughput over dedicated endpoints comes from this overbooking. A customer needing 500 tokens/second peak with 100 tokens/second average:

- **Dedicated endpoint**: Provision for 500 tokens/second = 5 full GPUs. Utilization: 20%.
- **Provisioned Throughput**: Reserve 500 tokens/second guaranteed = 3.3 GPU-equivalents (at 1.5x overbooking). Utilization: 30%.

For the customer, Provisioned Throughput costs 34% less than a dedicated endpoint while providing the same peak guarantee. For Databricks, the fleet utilization increases from 20% to 30%, with the difference representing pure margin improvement.

---

## 8. Autoscaling: Scale-to-Zero and Burst Handling

### Scale-to-Zero for Inference Endpoints

Scale-to-zero is conceptually simple but operationally complex for LLM serving. Unlike web servers that can cold-start in seconds, LLM endpoints require:

1. **GPU allocation**: Finding an available GPU (seconds to minutes depending on fleet utilization)
2. **Model weight loading**: Transferring model weights from storage to GPU memory (30-180 seconds for models ranging from 7B to 70B parameters over network-attached storage)
3. **Model warm-up**: Running a few inference passes to populate GPU caches and JIT compilation (5-10 seconds)

Total cold-start time ranges from 45 seconds (small model, warm GPU available) to 5+ minutes (large model, GPU allocation queue). This is unacceptable for interactive workloads but acceptable for batch or scheduled workloads.

Databricks implements tiered scale-to-zero:

- **Serverless endpoints (batch, non-latency-sensitive)**: Full scale-to-zero. Customer pays nothing when idle. Cold start of 1-5 minutes is acceptable.
- **Provisioned Throughput endpoints**: Scale to "warm standby" (model stays loaded, minimum batch processing active). Customer pays a reduced rate when idle but gets instant response when traffic resumes.
- **Foundation Model API (shared multi-tenant)**: Never scales to zero. The shared fleet is always running, absorbing the cost across all tenants.

### Burst Handling Strategies

When a tenant's traffic suddenly spikes (e.g., a product launch drives 10x normal email volume for a Superhuman-like customer), the system must decide between several options:

**Option 1: Absorb in burst pool.** The shared fleet maintains unallocated burst capacity (typically 15-20% of total fleet). Spikes from any tenant can temporarily consume burst capacity without affecting other tenants' guaranteed throughput.

**Option 2: Queue and throttle.** If burst capacity is exhausted, excess requests queue with increasing latency. The system provides backpressure signals (HTTP 429 with Retry-After headers) so clients can implement exponential backoff.

**Option 3: Horizontal scale-out.** For sustained spikes (>5 minutes), the autoscaler provisions additional GPU capacity. This is the slowest response (minutes) but handles sustained demand increases that burst pools cannot absorb.

The autoscaler's decision logic:



### Cost vs. Latency Tradeoff

Autoscaling introduces a fundamental tradeoff that customers must configure:

- **Aggressive scaling** (scale up at 50% capacity, scale down after 30 min idle): Low latency, high cost due to GPU capacity sitting idle during scale-down cooldowns
- **Conservative scaling** (scale up at 85% capacity, scale down after 5 min idle): Higher latency during spikes, lower cost due to tight resource utilization
- **Predictive scaling** (scale based on historical traffic patterns): Best of both worlds when traffic is predictable, worst when traffic is unpredictable

Databricks exposes these as configuration options in their Provisioned Throughput endpoints, allowing customers to dial the cost-latency tradeoff based on their workload characteristics.

---

## 9. Lessons for Practitioners: Applying These Patterns at Smaller Scale

### Model Units at 10 GPUs

You do not need thousands of GPUs to benefit from the Model Units pattern. The core idea, abstracting GPU capacity into portable units of guaranteed throughput, works at any scale where you serve multiple models or multiple tenants.

**At 10 GPUs, the Model Units approach means:**

1. **Measure your baselines**: For each model you serve, determine the maximum sustainable tokens/second per GPU under your target latency SLO. This is your "tokens per GPU" constant.
2. **Define your unit**: Choose a Model Unit size that is a useful quantum for your customers. If one GPU produces 2,000 tokens/second for your 7B model, a Model Unit might be 200 tokens/second (1/10 of a GPU).
3. **Sell units, not GPUs**: Your internal teams or external customers request N units rather than N GPUs. This allows you to overbook and achieve higher utilization.
4. **Monitor utilization vs. commitment**: Track the ratio of actually-consumed tokens/second to guaranteed tokens/second. If your fleet runs at 40% average utilization, you can safely overbook by 2x.

### LoRA Multiplexing on Your Own Fleet

Multi-LoRA serving is immediately applicable for any organization with multiple fine-tuned models sharing the same base:

**Step 1: Consolidate base models.** If you have 5 teams each running their own fine-tune of Llama 3, move all to LoRA adapters on a shared base model instance. This immediately saves 4x the base model memory.

**Step 2: Implement adapter routing.** Use a simple request header or URL path to identify which adapter each request targets. Your load balancer routes to GPUs that have the target adapter loaded.

**Step 3: Batch across adapters.** Use a serving framework that supports multi-LoRA batching (vLLM supports this natively since v0.3.0 with the  flag). This gives you the batching efficiency of a single model with the customization of per-team fine-tunes.

**Step 4: Set memory budgets.** Decide how many adapters to keep hot (in GPU HBM), warm (in host DRAM), and cold (on disk). A good starting point: keep adapters with >1 request/minute hot, >1 request/hour warm, everything else cold.

### Memory and Cost Calculators

Here is how to estimate whether multi-LoRA serving makes sense for your deployment:



For a 7B model (14 GB) with rank-16 adapters (66 MB each):


Multi-LoRA wins with as few as 2 adapters. The savings increase linearly with the number of adapters: 10 adapters save 9x the base model memory compared to dedicated serving.

### Prioritization Without a Full Scheduler

If you cannot implement Databricks' full priority scheduling system, a simpler 80/20 approach:

1. **Separate queues by SLO class**: Use your load balancer to route latency-critical requests to one pool of GPUs and throughput-oriented requests to another. No sharing between pools.
2. **Limit concurrent requests per tenant**: A simple max-concurrency limit prevents any single tenant from monopolizing batch slots. Set this to .
3. **Chunked prefill**: If your serving framework supports it (vLLM does), enable chunked prefill with a max chunk size of 1024-2048 tokens. This automatically prevents long-prompt tenants from blocking others.

### When to Invest in Multi-Tenancy Infrastructure

Multi-tenancy infrastructure has significant engineering cost. It makes sense when:

- You serve 5+ distinct model variants or fine-tunes
- Your fleet utilization with dedicated provisioning is below 40%
- You have tenants with highly variable traffic (>5x peak-to-average ratio)
- Your GPU spend exceeds your engineering team's annual cost (i.e., the infrastructure cost justifies the engineering investment)

If none of these apply, dedicated serving with simple autoscaling is the pragmatic choice. Multi-tenancy is an optimization, not a requirement.

---

## 10. Connecting the Dots: Production Patterns Summary

The Databricks multi-tenant serving story illustrates how the theoretical concepts from earlier chapters combine in production:

| Chapter Concept | Databricks Application |
|----------------|----------------------|
| Continuous batching (Ch03) | Mixed-adapter batching across tenants |
| Tensor parallelism (Ch05) | Large model serving within Model Units |
| Prefix caching (Ch06) | Shared prompt caching across tenant requests |
| Goodput metrics (Ch07.4) | SLO-aware throughput measurement per Model Unit |
| KV cache management (Ch02) | Per-tenant KV cache budgets and eviction |

The overarching lesson is that production multi-tenant serving is not a single technique but an orchestration of many techniques, each operating at a different layer of the stack. Model Units provides the resource abstraction. LoRA multiplexing provides the model-level efficiency. MixAttention provides the architecture-level memory savings. Fair scheduling provides the workload isolation. And autoscaling provides the demand-elasticity.

For practitioners building their own serving infrastructure, the recommended adoption order is:

1. **First**: Continuous batching + basic request routing (immediate throughput gains, low complexity)
2. **Second**: Multi-LoRA serving if you have multiple fine-tunes (high memory savings, moderate complexity)
3. **Third**: Priority queuing and SLO-based scheduling (required once you have multiple workload types with different latency needs)
4. **Fourth**: Model Units abstraction and overbooking (required once fleet utilization optimization becomes a priority)
5. **Fifth**: Custom architectures like MixAttention (only justified at very large scale where KV cache memory is the binding constraint)

Each step builds on the previous and delivers diminishing returns. Most organizations will find that steps 1-3 capture 80% of the efficiency gains with 20% of the engineering effort.

---

## References

1. Databricks Engineering Blog, "Reliable LLM Inference at Scale," June 2026. Introduces Model Units abstraction, Superhuman case study (200K QPS, 60% throughput gains), and fleet scheduling architecture.
2. Databricks Engineering Blog, "Fast PEFT Serving: Serving Hundreds of LoRA Adapters," February 2025. Details multi-adapter serving architecture, memory layout, adapter routing, and batching across adapters.
3. Databricks Engineering Blog, "MixAttention: Efficient Inference for Mixed-Context Workloads," September 2024. Describes the hybrid sliding-window/full-attention architecture and its memory efficiency benefits.
4. S-LoRA: Serving Thousands of Concurrent LoRA Adapters (Sheng et al., 2023). Academic foundation for multi-LoRA serving systems.
5. vLLM documentation: Multi-LoRA serving with  flag. Open-source implementation of adapter multiplexing.

---

## 11. Deep Dive: The Model Units Scheduler Algorithm

### Formal Problem Statement

The Model Units scheduler solves a constrained optimization problem at every scheduling interval (typically every 100ms). The inputs are:

- A set of pending requests R, each with: tenant ID, model/adapter target, input length, expected output length, priority class, arrival time
- A set of active GPUs G, each with: loaded model, available KV cache memory, current batch size, current utilization
- A set of Model Unit allocations M, mapping each tenant to their guaranteed throughput and current consumption

The objective is to maximize total fleet goodput (SLO-meeting tokens/second) while ensuring no tenant falls below their guaranteed Model Unit throughput for sustained periods (>1 scheduling quantum).

### The Two-Phase Scheduling Approach

Databricks decomposes this into two phases that run at different timescales:

**Phase 1: Placement (runs every 5-10 minutes)**

This phase decides which models and adapters are loaded on which GPUs. It solves a variant of the bin-packing problem:

```
Minimize: total_gpus_used
Subject to:
  - Every model with active Model Unit allocations has at least
    ceil(total_model_units / units_per_gpu) GPUs assigned
  - Every model has at least one redundant GPU (N+1)
  - No GPU exceeds its memory capacity (weights + adapters + KV headroom)
  - Incompatible workload classes are not co-located
```

This is NP-hard in the general case, but Databricks uses a greedy first-fit-decreasing heuristic that runs in O(M log G) time, where M is the number of model allocations and G is the number of GPUs. The heuristic:

1. Sort model allocations by size (largest first)
2. For each allocation, find the GPU with the most remaining capacity that already has the target model loaded
3. If no suitable GPU exists, assign a new GPU and load the model
4. Verify N+1 redundancy for each model; add standby GPUs if needed

**Phase 2: Dispatch (runs every batch iteration, ~10-50ms)**

This phase decides which pending requests enter the next batch iteration on each GPU. It implements weighted fair queuing:

```
For each GPU g with available batch slots:
  1. Compute each tenant's deficit:
     deficit[t] = guaranteed_rate[t] - actual_rate[t] (over last 1s window)
  2. Sort tenants by deficit (highest deficit = most underserved)
  3. Fill batch slots by pulling from underserved tenants first
  4. If all tenants are at or above their guarantee, fill remaining
     slots round-robin (this is burst capacity)
```

This two-phase approach decouples the slow, heavy operation (model loading/placement) from the fast, lightweight operation (request dispatch), allowing the system to react to traffic changes at millisecond granularity without constantly reshuffling models across GPUs.

### Handling Failures and Rebalancing

When a GPU fails, the scheduler must react within the SLO window (typically <2 seconds for P0 traffic):

1. **Immediate**: All requests on the failed GPU are marked as failed and retried
2. **Within 100ms**: Traffic for affected tenants is redirected to standby GPUs with the same model
3. **Within 5 minutes**: The placement phase reruns to restore N+1 redundancy by loading the model on a new GPU

The failover latency is dominated by the time to detect the failure (health check interval, typically 500ms) plus the time to redirect traffic (single routing table update, <10ms). Because standby GPUs already have model weights loaded, there is no model loading delay during failover.

---

## 12. Observability in Multi-Tenant Serving

### Metrics That Matter

Multi-tenant serving requires per-tenant observability on top of fleet-level metrics. The key metrics Databricks tracks, connecting to the goodput framework from Chapter 7.4:

**Per-tenant metrics (reported to customers):**
- Tokens/second consumed vs. guaranteed (Model Unit utilization)
- P50/P95/P99 time-to-first-token (TTFT)
- P50/P95/P99 inter-token latency (ITL)
- Request success rate (non-429, non-5xx)
- SLO attainment: percentage of requests meeting the latency target

**Per-GPU metrics (internal fleet management):**
- Batch utilization: average batch size / maximum batch size
- KV cache utilization: allocated KV cache / total KV cache budget
- Compute utilization: actual TFLOPS / peak TFLOPS
- Memory bandwidth utilization: actual GB/s / peak GB/s
- Adapter cache hit rate: requests served from hot adapter / total requests

**Fleet-level metrics (capacity planning):**
- Total Model Units sold vs. total Model Units capacity
- Overbooking ratio: sold capacity / physical capacity
- Burst pool utilization: peak burst consumption / burst pool size
- Placement efficiency: actual GPUs used / theoretical minimum GPUs needed

### Debugging Multi-Tenant Performance Issues

When a tenant reports degraded performance, the debugging workflow follows a structured investigation:

1. **Check tenant's own metrics**: Is the tenant exceeding their Model Unit allocation? If so, throttling is expected behavior, not a bug.

2. **Check co-located tenants**: Are other tenants on the same GPU(s) experiencing similar degradation? If yes, the issue is likely GPU-level (hardware fault, thermal throttling, or misconfigured batch size).

3. **Check KV cache pressure**: Is the GPU's KV cache pool near capacity? If one tenant's long-context requests are consuming disproportionate KV cache, the per-tenant budget enforcement may need tightening.

4. **Check prefill interference**: Are there concurrent large prefill operations? The chunked prefill mechanism should prevent long stalls, but misconfigured chunk sizes can still cause 50-100ms delays.

5. **Check adapter loading**: Is the tenant's adapter being repeatedly evicted and reloaded? This creates periodic latency spikes every time the adapter must be fetched from host memory.

Each of these checks maps to specific metrics in the observability stack, allowing operators to quickly isolate whether a performance issue is caused by the tenant's own workload, a co-tenant's interference, or a system-level problem.

---

## 13. Comparison with Other Multi-Tenant Approaches

### Databricks vs. Anyscale (Ray Serve)

Anyscale's approach to multi-tenant serving differs in key architectural choices:

| Dimension | Databricks | Anyscale (Ray Serve) |
|-----------|-----------|---------------------|
| Resource abstraction | Model Units (throughput-based) | Replicas (instance-based) |
| Isolation | Shared GPU with priority scheduling | Separate actor processes per model |
| LoRA handling | Multi-LoRA batching on shared base | Separate replicas per adapter |
| Scaling unit | Fractional GPU (Model Units) | Full replica (1+ GPUs) |
| Overbooking | Yes (statistical multiplexing) | No (dedicated replicas) |

The Databricks approach achieves higher utilization through sharing and overbooking but requires more sophisticated scheduling. The Anyscale approach provides stronger isolation guarantees but at higher cost per tenant.

### Databricks vs. Together AI

Together AI's approach focuses on optimizing for the common case: many customers running the same popular open-source models (Llama, Mixtral, Qwen). Their FlashInfer-based serving system achieves efficiency through:

- Aggressive prefix caching (shared system prompts across tenants)
- Speculative decoding with shared draft models
- Custom CUDA kernels optimized for specific model families

Where Databricks differentiates is in supporting customer-specific fine-tunes (via LoRA multiplexing) and custom model architectures (via MixAttention and general model hosting). Together AI primarily serves base models and popular fine-tunes, which simplifies the multi-tenancy problem but limits customization.

### When Each Approach Wins

- **Databricks' Model Units approach** wins when: you have diverse models, customer-specific fine-tunes, variable traffic patterns, and SLO diversity. The overhead of the scheduling infrastructure is justified by the utilization gains.
- **Dedicated replicas (Anyscale-style)** win when: you have few models, predictable traffic, and strict isolation requirements (e.g., compliance reasons forbidding shared GPU execution).
- **Shared base model (Together-style)** wins when: all tenants use the same small set of models, traffic is high and steady, and prefix caching provides the majority of efficiency gains.

---

## 14. Future Directions

### Disaggregated Multi-Tenancy

The next frontier for multi-tenant serving, building on the disaggregated serving concepts from Chapter 6.4, is separating the prefill and decode phases across different GPU pools with independent multi-tenancy policies:

- **Prefill pool**: Optimized for throughput, handles large prompts from batch workloads. Multi-tenancy here focuses on fair bandwidth allocation.
- **Decode pool**: Optimized for latency, handles autoregressive generation for interactive workloads. Multi-tenancy here focuses on SLO isolation.

This disaggregation allows each pool to have different overbooking ratios, scheduling policies, and hardware configurations (prefill favors compute-dense GPUs like H100 SXM, decode favors memory-bandwidth-dense configurations).

### Predictive Scaling with Workload Forecasting

Current autoscaling is reactive: traffic increases, then capacity scales. Future systems will use tenant-specific traffic forecasting to predict demand hours or days in advance:

- Email platforms (like Superhuman) have strong weekly and daily seasonality
- Enterprise batch jobs run on predictable schedules
- Marketing campaigns drive known traffic spikes

By forecasting per-tenant demand and pre-positioning model replicas accordingly, the system can eliminate burst-pool consumption entirely for predictable workloads, freeing burst capacity for truly unpredictable spikes.

### Hardware-Aware Model Unit Pricing

As GPU generations evolve (A100 to H100 to B200), the tokens/second per GPU increases but so does the cost per GPU. Future Model Units pricing will need to account for the heterogeneous fleet:

- Older GPUs (A100): Lower throughput, lower cost per Model Unit, suitable for latency-tolerant workloads
- Current GPUs (H100): Higher throughput, moderate cost, suitable for mixed workloads
- Next-gen GPUs (B200): Highest throughput, highest cost, reserved for latency-critical P0 workloads

This creates a natural tiering where cost-sensitive tenants are automatically placed on older hardware while performance-sensitive tenants pay a premium for newer hardware, without requiring tenants to understand GPU hardware specifications.

---

## Summary

The Databricks multi-tenant serving story demonstrates that production LLM serving at scale is fundamentally a resource management problem, not just a model optimization problem. The key innovations:

1. **Model Units** transform GPU inference from a hardware problem into an abstract capacity planning problem, enabling statistical multiplexing and overbooking
2. **Fast LoRA serving** makes per-customer model customization nearly free in memory terms, turning fine-tuning from a deployment burden into a serving feature
3. **MixAttention** addresses the architectural tension between short-context efficiency and long-context capability, reducing KV cache costs by ~48%
4. **Priority-aware scheduling** with chunked prefill and per-tenant KV budgets provides isolation without dedication
5. **The Superhuman partnership** (200K QPS, 60% throughput gains) validates that these techniques compose into real-world production value

For practitioners, the actionable takeaway is not to replicate Databricks' full stack but to adopt the patterns in priority order: continuous batching first, then LoRA multiplexing, then SLO-based scheduling, and finally the full Model Units abstraction. Each step delivers measurable utilization improvements and can be implemented independently.

---

## 15. Quantifying the Economics: A Worked Example

### Scenario Setup

Consider a medium-scale deployment serving 50 enterprise customers on a fleet of 32 A100 80GB GPUs. Each customer has a fine-tuned Llama 3.1 8B model (via LoRA adapter). Traffic characteristics vary:

- 10 customers: high-traffic interactive chat (500 tokens/sec average, 2000 peak)
- 20 customers: moderate-traffic batch processing (200 tokens/sec average, 800 peak)
- 20 customers: low-traffic periodic jobs (50 tokens/sec average, 500 peak)

### Dedicated Serving Cost

With dedicated provisioning (one set of GPUs per customer, sized for peak):

```
High-traffic customers:  10 x ceil(2000 / 4000 tokens_per_A100) = 10 GPUs
Moderate customers:      20 x ceil(800 / 4000)  = 20 GPUs
Low-traffic customers:   20 x ceil(500 / 4000)  = 20 GPUs
Total GPUs needed:       50 GPUs
Average utilization:     (10x500 + 20x200 + 20x50) / (50 x 4000) = 5%
```

This is clearly untenable. Even with fractional GPU allocation, dedicated provisioning at 5% utilization wastes 95% of your GPU spend.

### Multi-Tenant Model Units Cost

With the Model Units approach (shared fleet, statistical multiplexing):

```
Total guaranteed throughput sold:
  10 x 2000 + 20 x 800 + 20 x 500 = 46,000 tokens/sec

Total average demand:
  10 x 500 + 20 x 200 + 20 x 50 = 10,000 tokens/sec

Peak concurrent demand (95th percentile, estimated):
  ~25,000 tokens/sec (not all tenants peak simultaneously)

GPUs needed for peak demand:
  ceil(25000 / 4000) = 7 GPUs for compute
  + adapter memory overhead: ~0.5 GPUs
  + N+1 redundancy: 1 GPU
  + burst buffer (20%): 2 GPUs
  Total: ~10 GPUs

Average utilization: 10000 / (10 x 4000) = 25%
Peak utilization: 25000 / (10 x 4000) = 62.5%
```

The multi-tenant approach requires 10 GPUs instead of 50: an 80% reduction in GPU fleet size with significantly higher utilization. The remaining headroom (62.5% peak vs. 100% capacity) provides the safety margin for handling burst-beyond-guarantee traffic without SLO violations.

### The Overbooking Math

Total guaranteed throughput sold: 46,000 tokens/sec
Physical capacity: 10 GPUs x 4000 = 40,000 tokens/sec

This means guaranteed throughput exceeds physical capacity by 15% (overbooking ratio 1.15x). This is safe because the probability of all 50 tenants simultaneously reaching their guaranteed throughput is negligible (by the law of large numbers, actual demand converges to the mean as tenant count increases).

However, the system must handle the case where realized demand exceeds capacity. The priority mechanism ensures:
- First 40,000 tokens/sec are served at guaranteed latency
- Excess demand (rare) is served with degraded latency or queued
- No tenant is ever completely starved (minimum service rate = 10% of guarantee even under extreme contention)

### ROI Calculation

```
Annual GPU cost (dedicated):     50 GPUs x $2.50/hr x 8760 hrs = $1,095,000
Annual GPU cost (multi-tenant):  10 GPUs x $2.50/hr x 8760 hrs = $219,000
Annual savings: $876,000 (80% reduction)

Engineering cost of multi-tenant infrastructure:
  Estimated: 2 senior engineers x 6 months = ~$400,000 (fully loaded)

Payback period: 5.5 months
```

This calculation demonstrates why Databricks invested heavily in multi-tenant infrastructure: the GPU cost savings at their scale (thousands of GPUs, not tens) run into tens of millions annually, making even a large engineering investment worthwhile.

### Sensitivity Analysis

The savings are most sensitive to:

1. **Traffic burstiness** (peak/average ratio): Higher burstiness = more savings from statistical multiplexing. At 10:1 peak/average (our example), savings are 80%. At 2:1 peak/average, savings drop to ~40%.

2. **Number of tenants**: More tenants = better statistical multiplexing (law of large numbers). With 5 tenants, peak overlap is likely; with 50 tenants, it is rare. The efficiency gains plateau around 30-50 tenants.

3. **SLO strictness**: Stricter SLOs require more headroom (burst buffer), reducing the overbooking ratio. With 99.9% SLO attainment target, you need ~30% burst buffer. With 99% target, ~15% suffices.

4. **Model diversity**: If all tenants use different base models (not just different adapters), the sharing benefits diminish because you cannot batch across different models. LoRA multiplexing only works when tenants share a base model.

These sensitivities determine whether multi-tenant serving is worth the engineering investment for your specific deployment. Run the numbers with your actual traffic patterns before committing to the architecture.
