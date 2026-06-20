# 09.5 System Design: Inference Infrastructure for Agentic AI Workloads

## Introduction

Agentic AI represents the most demanding inference workload pattern in production today. Unlike single-shot question-answering or batch summarization, an AI agent chains 5 to 20 LLM calls per task, with each call building upon the results of previous steps, tool invocations, and accumulated reasoning context. This creates a fundamentally different infrastructure challenge: the system must maintain state across multiple inference calls, manage exploding KV caches that grow with every step, route requests to preserve locality, and handle the unpredictable control flow that emerges when an LLM decides its own execution path.

Consider a code review agent that receives a pull request. Step 1: parse the diff and identify changed files. Step 2: retrieve relevant documentation for each changed module. Step 3: analyze each file for bugs, style violations, and security issues. Step 4: cross-reference findings across files. Step 5: synthesize a coherent review with prioritized feedback. Each step feeds its output into the next, context accumulates, and the KV cache grows from 2K tokens at step 1 to potentially 50K tokens by step 5. Multiply this by 10,000 concurrent agent sessions, and you face a memory management problem that no traditional serving system was designed to handle.

This chapter designs the complete inference infrastructure for agentic workloads, drawing on techniques from across this book: KV cache management (Ch02), speculative decoding (Ch03), serving architectures (Ch06), cache-aware routing (Ch06.6), LMCache for inter-step persistence (Ch02.5), and production monitoring (Ch07). The result is a system that delivers P95 total task completion under 30 seconds while serving 10,000 concurrent agent sessions cost-effectively.

---

## 1. Requirements Analysis

### Workload Characterization

Agentic workloads differ from traditional LLM serving in five critical dimensions:

**Multi-step execution with state accumulation.** Each agent task involves 5 to 20 sequential LLM calls. The output of step N becomes part of the input for step N+1. This creates a temporal dependency that prevents simple load-balancing across a stateless inference pool. Unlike batch processing where each request is independent, agent steps form a directed acyclic graph (or sometimes a cycle, when the agent retries a failed action).

**Variable-length chains with unpredictable depth.** A simple query might resolve in 3 steps, while a complex research task might require 15 or more. The system cannot pre-allocate resources based on a fixed chain length. Capacity planning must account for the long tail: if the mean is 7 steps but P99 is 18 steps, the infrastructure must handle bursts of deep chains without degrading latency for shallow ones.

**Tool-call interleaving.** Between LLM calls, the agent executes tools: web searches, code execution, API calls, database queries. Tool results inject variable-length content into the context. A search result might add 500 tokens; a code execution trace might add 3,000. This unpredictability makes memory budgeting significantly harder than for fixed-template workloads.

**Context growth pattern.** The token count per step follows a roughly linear growth pattern:

| Step | Accumulated Tokens | Primary Source |
|------|-------------------|----------------|
| 1 | 2,000 | System prompt + initial query |
| 3 | 6,000 | + 2 tool results + reasoning |
| 5 | 12,000 | + search results + code output |
| 10 | 30,000 | + multi-file analysis |
| 15 | 50,000 | + cross-reference + synthesis |
| 20 | 70,000 | + revision + final output |

**Bursty load patterns.** Agents do not distribute their compute needs uniformly. A "planning" step (short output, fast) followed by a "reasoning" step (long chain-of-thought, slow) creates load spikes. When 10,000 agents simultaneously hit their reasoning step, the cluster faces a thundering herd problem.

### Quantitative SLOs

The system must meet these targets under full load (10K concurrent sessions):

- **P50 per-step latency**: <800ms for 8B model, <2s for 70B model
- **P95 per-step latency**: <2s for 8B model, <5s for 70B model
- **P95 total task completion** (7-step average task): <30 seconds
- **KV cache hit rate** (inter-step): >95% (cache must survive between steps)
- **Tool-call overhead**: <500ms P95 (not counted against LLM latency budget)
- **Availability**: 99.9% (agent tasks are user-facing, not batch)
- **Concurrent sessions**: 10,000 sustained, 15,000 burst

### Derived Infrastructure Requirements

From these SLOs, we can derive hard constraints:

**Memory for KV cache.** At step 10 (30K tokens, 128 bytes per token per layer for typical 70B architecture), a single session requires approximately 3.84GB of KV cache. With 10K concurrent sessions at varying depths (average step 7, ~20K tokens), total KV cache memory is roughly 10,000 x 2.56GB = 25.6TB. This immediately rules out GPU-only storage and mandates a tiered memory architecture.

**Compute throughput.** 10K sessions, each making one LLM call every 3 seconds on average (including tool-call time), means ~3,333 inference requests per second. At an average of 200 output tokens per step, that is 666,000 tokens per second of generation throughput. For a 70B model at FP16 with TP=4, a single node generates approximately 800 tokens/second, requiring approximately 833 nodes for the large model alone if all requests went to 70B. Model routing (sending simple steps to 8B) is not optional; it is an economic necessity.

**Network bandwidth.** KV cache transfer between steps (if cache is not on the same GPU) requires moving gigabytes per second. With 3,333 requests/second and average KV cache of 2.56GB, naive approaches would require 8.5TB/s of network bandwidth. Session-aware routing (keeping the same session on the same GPU) reduces this to near zero for most requests.

---

## 2. Model Selection Strategy

### The Two-Model Architecture

Agentic workloads have a natural bifurcation in reasoning complexity. Some steps require deep chain-of-thought reasoning (planning, multi-step deduction, error recovery), while others are mechanical formatting tasks (converting a tool response into a structured action, generating a function call signature). A single model for all steps wastes either quality (too small) or compute (too large).

**The planning model (70B class).** Handles steps that require:
- Multi-hop reasoning over accumulated context
- Decision-making about which tool to call next
- Error recovery when a tool call fails
- Final synthesis of multi-step results
- Novel problem decomposition

Examples: Llama 3.1 70B, Qwen2 72B, Mixtral 8x22B (MoE variant). At FP16, the 70B model requires approximately 140GB of weights, necessitating TP=4 across 4x H100 80GB GPUs (leaving ~180GB aggregate for KV cache).

**The execution model (8B class).** Handles steps that require:
- Formatting a tool call from the LLM's decision
- Parsing structured tool output into the agent's expected format
- Simple classification (is this step done? does this need escalation?)
- Template-based responses where the plan is already determined

Examples: Llama 3.1 8B, Mistral 7B, Qwen2 7B. At FP16, requires ~16GB of weights, fitting on a single H100 with 64GB remaining for KV cache.

### Model Routing Logic

The routing decision happens at the agent orchestrator level, before the inference call:

```python
class AgentModelRouter:
    """Routes each agent step to the appropriate model based on step type."""
    
    def __init__(self, config: RouterConfig):
        self.planning_model = "llama-3.1-70b"
        self.execution_model = "llama-3.1-8b"
        self.complexity_threshold = 0.7
    
    def route(self, step: AgentStep) -> str:
        # Hard rules first
        if step.type in ("plan", "reason", "recover", "synthesize"):
            return self.planning_model
        if step.type in ("format_tool_call", "parse_response", "classify"):
            return self.execution_model
        
        # Soft routing based on context complexity
        complexity = self.estimate_complexity(step)
        if complexity > self.complexity_threshold:
            return self.planning_model
        return self.execution_model
    
    def estimate_complexity(self, step: AgentStep) -> float:
        """Heuristic complexity score based on step characteristics."""
        score = 0.0
        score += 0.3 if step.context_tokens > 10000 else 0.0
        score += 0.2 if step.requires_multi_hop else 0.0
        score += 0.2 if step.previous_step_failed else 0.0
        score += 0.15 if step.num_tool_results > 3 else 0.0
        score += 0.15 if step.is_final_synthesis else 0.0
        return score
```

### Speculative Model Escalation

A more sophisticated approach uses the 8B model speculatively for ambiguous steps:

1. Send the step to the 8B model with a confidence probe
2. If the model's output confidence (measured by token probabilities) drops below threshold, abort and escalate to 70B
3. This saves compute on the ~40% of "medium" steps that the 8B model can handle adequately

The escalation cost is one wasted 8B inference (cheap: ~100ms for 200 tokens) versus the savings of avoiding 70B on steps that do not need it (expensive: ~1.5s for 200 tokens). Even with a 20% false-start rate, this saves ~25% of 70B compute across a typical agent task.

### Model Distribution Across a Typical Agent Task

Analysis of production agent traces reveals the following distribution for a 10-step task:

| Step | Type | Model | Typical Latency |
|------|------|-------|-----------------|
| 1 | Plan initial approach | 70B | 1.8s |
| 2 | Format first tool call | 8B | 0.3s |
| 3 | Parse tool result | 8B | 0.2s |
| 4 | Reason about result | 70B | 2.1s |
| 5 | Format next tool call | 8B | 0.3s |
| 6 | Parse tool result | 8B | 0.2s |
| 7 | Reason + decide next action | 70B | 2.4s |
| 8 | Format tool call | 8B | 0.3s |
| 9 | Parse + classify completion | 8B | 0.4s |
| 10 | Final synthesis | 70B | 2.8s |

Total LLM time: ~10.8s. With tool-call overhead (~5s for 4 tool calls), total task time is ~16s, well within our 30s P95 SLO. The 70B model handles 4/10 steps but consumes ~83% of the LLM compute budget.

---

## 3. Memory Budget: The KV Cache Explosion

### The Fundamental Problem

KV cache in agentic workloads grows monotonically across steps. Unlike a stateless chatbot where each request starts fresh, an agent accumulates context: every tool result, every intermediate reasoning trace, every action taken. This creates the single hardest infrastructure problem in agentic inference.

**Per-session KV cache calculation for a 70B model (GQA with 8 KV heads, 128 head_dim, 80 layers):**

```
KV cache per token = 2 (K and V) x 8 (KV heads) x 128 (head_dim) x 80 (layers) x 2 (FP16 bytes)
                   = 2 x 8 x 128 x 80 x 2
                   = 327,680 bytes
                   ≈ 320 KB per token
```

At various step depths:

| Step | Tokens | KV Cache Size | Notes |
|------|--------|---------------|-------|
| 1 | 2,000 | 640 MB | System prompt + query |
| 5 | 12,000 | 3.84 GB | + tool results |
| 10 | 30,000 | 9.6 GB | + multi-file analysis |
| 15 | 50,000 | 16 GB | + cross-references |
| 20 | 70,000 | 22.4 GB | Approaching context limit |

### The Aggregate Problem

With 10,000 concurrent sessions, the total KV cache requirement depends on the distribution of session depths:

```python
def calculate_total_kv_cache(
    num_sessions: int = 10000,
    step_distribution: dict = None
) -> float:
    """Calculate total KV cache requirement in TB."""
    if step_distribution is None:
        # Empirical distribution from production traces
        step_distribution = {
            1: 0.05,   # 5% of sessions just started
            3: 0.15,   # 15% at step 3
            5: 0.25,   # 25% at step 5 (most common)
            7: 0.25,   # 25% at step 7
            10: 0.15,  # 15% at step 10
            15: 0.10,  # 10% deep chains
            20: 0.05,  # 5% very deep chains
        }
    
    tokens_per_step = {1: 2000, 3: 6000, 5: 12000, 7: 20000,
                       10: 30000, 15: 50000, 20: 70000}
    kv_bytes_per_token = 327680  # 320 KB for 70B model
    
    total_bytes = 0
    for step, fraction in step_distribution.items():
        sessions_at_step = num_sessions * fraction
        tokens = tokens_per_step[step]
        bytes_per_session = tokens * kv_bytes_per_token
        total_bytes += sessions_at_step * bytes_per_session
    
    total_tb = total_bytes / (1024**4)
    return total_tb

# Result: approximately 48.7 TB of KV cache for 10K sessions
```

**48.7 TB of KV cache.** This is physically impossible to store in GPU HBM (a cluster of 100 H100s has only 8TB of HBM total). The solution requires a tiered memory architecture that this chapter designs in detail.

### Tiered KV Cache Architecture

The solution draws directly from Ch02.5 (LMCache) and Ch02.4 (KV compression):

**Tier 1: GPU HBM (hot cache).** Stores KV cache for the currently-executing step of active sessions. Capacity: ~60GB per GPU (after model weights). At 320KB/token, this holds approximately 187,500 tokens per GPU, enough for ~6 sessions at step 10 (30K tokens each) or ~15 sessions at step 5 (12K tokens each).

**Tier 2: CPU DRAM (warm cache).** Stores KV cache for sessions between steps (waiting for tool calls). Capacity: 1-2TB per node. At 320KB/token, a single node with 2TB RAM holds KV cache for approximately 200 sessions at step 10. A 100-node cluster provides warm cache for all 10K sessions at average depth.

**Tier 3: NVMe SSD (cold cache).** Stores KV cache for idle sessions (user thinking time, long tool calls >10s). Capacity: 8-16TB per node. Sequential read at 7GB/s means restoring a 10GB KV cache (step 10) takes ~1.4 seconds, acceptable for sessions resuming after long pauses.

**Tier 4: Network storage (frozen cache).** For sessions that have been idle >5 minutes. Stored on distributed storage (e.g., Redis cluster or S3). Restoration takes 3-10 seconds but allows infinite session persistence.

```python
class TieredKVCacheManager:
    """Manages KV cache across memory tiers for agentic workloads."""
    
    def __init__(self, config: TierConfig):
        self.gpu_tier = GPUCacheTier(capacity_gb=60)
        self.cpu_tier = CPUCacheTier(capacity_gb=2000)
        self.nvme_tier = NVMeCacheTier(capacity_gb=16000)
        self.network_tier = NetworkCacheTier()  # Unbounded
        
        # Promotion/demotion thresholds
        self.idle_to_cpu_seconds = 2.0     # After 2s idle, move to CPU
        self.idle_to_nvme_seconds = 30.0   # After 30s idle, move to NVMe
        self.idle_to_network_seconds = 300  # After 5min, move to network
    
    async def get_kv_cache(self, session_id: str) -> KVCache:
        """Retrieve KV cache, promoting from lower tiers if necessary."""
        # Check tiers in order (fast to slow)
        if cache := self.gpu_tier.get(session_id):
            return cache
        if cache := self.cpu_tier.get(session_id):
            await self.promote_to_gpu(session_id, cache)
            return cache
        if cache := self.nvme_tier.get(session_id):
            await self.promote_to_cpu_then_gpu(session_id, cache)
            return cache
        if cache := self.network_tier.get(session_id):
            await self.full_promotion(session_id, cache)
            return cache
        return None  # Cache miss: must recompute from scratch
    
    async def promote_to_gpu(self, session_id: str, cache: KVCache):
        """Move cache from CPU to GPU. Latency: ~50ms for 3GB."""
        if not self.gpu_tier.has_space(cache.size_bytes):
            await self.evict_gpu_lru()
        await self.gpu_tier.store(session_id, cache)
        self.cpu_tier.remove(session_id)
```

### KV Cache Compression for Deep Chains

For sessions that reach step 15+ (50K tokens), even CPU DRAM becomes expensive at 16GB per session. Compression from Ch02.4 applies:

**Quantization from FP16 to INT4.** Reduces KV cache by 4x with minimal quality loss for earlier layers (the "stale" context from steps 1-3 is less precision-sensitive than the recent step's KV entries). This brings a step-15 session from 16GB to 4GB.

**Attention sink + eviction.** For very deep chains, tokens from early steps that received minimal attention in recent steps can be evicted entirely. The first 4 tokens (attention sinks) and the last 2,000 tokens (recent context) are always kept; middle tokens are candidates for eviction based on cumulative attention scores.

**Summarization checkpoints.** Every 5 steps, the agent generates a compressed summary of its progress so far. If KV cache must be fully evicted, the agent can resume from the summary rather than reprocessing all previous steps. This trades a small quality loss for a massive memory saving:

```python
class SummarizationCheckpoint:
    """Periodically summarize agent state to allow KV cache recovery."""
    
    def __init__(self, checkpoint_interval: int = 5):
        self.interval = checkpoint_interval
        self.checkpoints = {}  # session_id -> list of summaries
    
    async def maybe_checkpoint(self, session_id: str, step: int, 
                                context: str, model: InferenceClient):
        if step % self.interval != 0:
            return
        
        summary_prompt = (
            "Summarize the agent's progress so far in 500 tokens. "
            "Include: actions taken, results obtained, current hypothesis, "
            "and next planned steps. Be precise about data and findings."
        )
        summary = await model.generate(
            system=summary_prompt,
            user=context[-8000:],  # Last 8K tokens for summary context
            max_tokens=500
        )
        self.checkpoints.setdefault(session_id, []).append({
            "step": step,
            "summary": summary,
            "timestamp": time.time()
        })
```

---

## 4. Hardware Selection

### The Core Tension

Agentic workloads create a tension between two hardware requirements:

1. **Compute density** for fast token generation (want the fastest GPU possible)
2. **Memory capacity** for KV cache storage (want maximum memory per dollar)

Traditional LLM serving optimizes for (1) because KV cache is bounded by a single request's context window. Agentic workloads break this assumption because KV cache persists across multiple requests within a session.

### Recommended Hardware Architecture

**GPU nodes (compute tier): NVIDIA H100 80GB or H200 141GB**

Each H100 node (8x H100, 640GB aggregate HBM) serves:
- 70B model with TP=4: uses 2 of the 8 GPUs for weights, leaving 6 GPUs x 80GB = 480GB for KV cache
- Actually: with TP=4, weights are sharded across 4 GPUs at 35GB each, leaving 4 x 45GB = 180GB for KV cache on those GPUs
- The remaining 4 GPUs run the 8B model (16GB weights, 64GB for KV cache each = 256GB total for 8B sessions)

With H200 (141GB HBM per GPU), the budget improves dramatically:
- 70B model TP=4: 35GB weights per GPU, leaving 106GB per GPU x 4 = 424GB for KV cache
- This supports approximately 1,325 concurrent sessions at step 5 (320KB/token x 12K tokens = 3.84GB per session) per node

**CPU memory nodes (warm cache tier): High-memory instances**

AWS r7i.24xlarge (768GB RAM) or r7i.48xlarge (1.5TB RAM) serve as KV cache overflow. Connected to GPU nodes via high-bandwidth networking (EFA at 400 Gbps on AWS, InfiniBand at 400 Gbps on-prem). Each 1.5TB node stores warm cache for approximately 460 sessions at step 10.

**NVMe storage nodes (cold cache tier): Storage-optimized instances**

AWS i4i.16xlarge (15TB NVMe, 7GB/s sequential read) stores cold KV cache. Each node holds approximately 4,600 sessions at step 10. Restoration latency: 1.4s for a full step-10 KV cache.

### Cluster Sizing for 10K Concurrent Sessions

```python
def size_cluster(
    concurrent_sessions: int = 10000,
    avg_step: int = 7,
    tokens_at_avg_step: int = 20000,
    kv_bytes_per_token: int = 327680,
    gpu_kv_budget_gb: float = 180,  # Per TP=4 group after weights
    cpu_memory_per_node_gb: float = 1500,
    active_fraction: float = 0.3,   # 30% of sessions generating at any time
) -> dict:
    """Size the hardware cluster for agentic workloads."""
    
    active_sessions = int(concurrent_sessions * active_fraction)
    waiting_sessions = concurrent_sessions - active_sessions
    
    # GPU tier: must hold active sessions' KV cache
    kv_per_active_session_gb = (tokens_at_avg_step * kv_bytes_per_token) / (1024**3)
    gpu_nodes_needed = (active_sessions * kv_per_active_session_gb) / gpu_kv_budget_gb
    
    # CPU tier: holds waiting sessions' KV cache
    kv_per_waiting_session_gb = kv_per_active_session_gb  # Same size, just idle
    cpu_nodes_needed = (waiting_sessions * kv_per_waiting_session_gb) / cpu_memory_per_node_gb
    
    return {
        "gpu_nodes_h100_8x": int(gpu_nodes_needed) + 1,  # ceil
        "cpu_memory_nodes_1.5TB": int(cpu_nodes_needed) + 1,
        "active_sessions_on_gpu": active_sessions,
        "kv_per_session_gb": round(kv_per_active_session_gb, 2),
        "total_gpu_kv_tb": round(active_sessions * kv_per_active_session_gb / 1024, 2),
        "total_cpu_kv_tb": round(waiting_sessions * kv_per_waiting_session_gb / 1024, 2),
    }

# Result:
# gpu_nodes: ~107 H100 nodes (for compute + active KV)
# cpu_nodes: ~28 high-memory nodes (for warm KV)
# This is expensive but feasible for a production agentic platform
```

### H200 vs H100 Economic Analysis

The H200's 141GB HBM (vs H100's 80GB) provides 76% more KV cache budget per GPU. For agentic workloads where KV cache is the bottleneck, this translates directly to fewer nodes:

| Metric | H100 Cluster | H200 Cluster | Savings |
|--------|-------------|-------------|---------|
| GPU nodes needed | 107 | 62 | 42% fewer |
| HBM for KV cache | 180 GB/node | 424 GB/node | 2.4x more |
| Sessions per node | 28 | 66 | 2.4x more |
| Cost (estimated) | $107 x $300K = $32M | 62 x $450K = $28M | 13% cheaper |

The H200 is the clear winner for agentic workloads despite its higher per-unit cost, because the KV cache capacity drives node count more than raw FLOPS.

---

## 5. Parallelism Strategy

### Why Agentic Workloads Break Traditional Parallelism

Traditional LLM serving uses tensor parallelism (TP) to split a model across GPUs for single-request latency, and data parallelism (DP) for throughput. Agentic workloads complicate this because:

1. **TP is needed only for the 70B model.** The 8B model fits on a single GPU. Since ~60% of steps go to 8B, most requests need no TP overhead.
2. **DP must be session-aware.** Standard DP routes requests to any available replica. But agent sessions have KV cache affinity: step N+1 must go to the same replica that served step N (or pay the KV transfer cost).
3. **Load is temporally correlated.** When one agent hits its "reasoning" phase, it likely means similar agents (same type, started around the same time) are also hitting reasoning phases. This creates synchronized spikes on 70B replicas.

### Parallelism Architecture

**70B model: TP=4 within a node, DP across nodes.**

Each 70B serving group uses 4 GPUs on the same node (NVLink interconnect for fast all-reduce). The cluster has N such groups operating as independent DP replicas. Session routing ensures a session stays on the same DP replica across steps.

```
Node 1: [GPU0-GPU3] = 70B replica 1  |  [GPU4-GPU7] = 8B replicas (4 independent)
Node 2: [GPU0-GPU3] = 70B replica 2  |  [GPU4-GPU7] = 8B replicas (4 independent)
...
Node K: [GPU0-GPU3] = 70B replica K  |  [GPU4-GPU7] = 8B replicas (4 independent)
```

**8B model: No TP needed, pure DP.**

Each 8B model runs on a single GPU. With ~60% of steps going to 8B and 4 GPUs per node dedicated to 8B, each node handles approximately 4x the 8B request throughput of 70B. This naturally balances compute allocation with the expected request distribution.

### Session-Aware Load Balancing

The critical insight: standard round-robin or least-connections load balancing destroys KV cache locality. A session-aware balancer must route based on where the session's KV cache lives:

```python
class SessionAwareBalancer:
    """Routes agent steps to the GPU holding their KV cache."""
    
    def __init__(self, replicas: list[ModelReplica]):
        self.replicas = replicas
        self.session_affinity = {}  # session_id -> replica_id
        self.replica_load = {r.id: 0 for r in replicas}
    
    def route(self, request: AgentStepRequest) -> ModelReplica:
        session_id = request.session_id
        
        # Strong affinity: if session has cached KV on a replica, go there
        if session_id in self.session_affinity:
            replica_id = self.session_affinity[session_id]
            replica = self.get_replica(replica_id)
            if replica.is_healthy() and replica.has_kv_cache(session_id):
                return replica
        
        # Affinity broken (replica down or cache evicted): choose new replica
        # Prefer replicas with lowest load to prevent hotspots
        available = sorted(
            [r for r in self.replicas if r.is_healthy()],
            key=lambda r: self.replica_load[r.id]
        )
        chosen = available[0]
        self.session_affinity[session_id] = chosen.id
        self.replica_load[chosen.id] += 1
        return chosen
    
    def release(self, session_id: str):
        """Called when a session completes."""
        if session_id in self.session_affinity:
            replica_id = self.session_affinity.pop(session_id)
            self.replica_load[replica_id] -= 1
```

### Handling Thundering Herds

When many agents simultaneously escalate to the 70B model (the "reasoning surge"), the system needs overflow capacity:

**Strategy 1: Request queuing with priority.**  Reasoning steps that have already accumulated expensive KV cache get priority over new planning steps (which have minimal cache investment).

**Strategy 2: Dynamic 8B-to-70B overflow.** When 70B queue depth exceeds threshold, temporarily redirect "borderline" requests (complexity score 0.6-0.7) to 8B with extended chain-of-thought prompting. Quality degrades slightly but latency remains bounded.

**Strategy 3: Preemptive scaling.** Monitor the fraction of active sessions at "pre-reasoning" steps (steps 3-4 of a typical task). When this fraction exceeds threshold, proactively scale 70B capacity before the surge hits.



---

## 6. Serving Architecture: Session-Aware Inference

### Architecture Overview

The serving architecture for agentic workloads has three layers that do not exist in traditional LLM serving:

1. **Agent Orchestrator**: Manages the agent loop (decide action, call tool, feed result back). Owns session state, tool execution, and model routing decisions.
2. **Session Router**: Maintains the mapping of session_id to GPU/replica. Ensures KV cache locality across steps. Implements cache-aware routing from Ch06.6.
3. **Inference Pool**: Stateless model replicas that serve individual steps. Each replica manages its local KV cache and exposes APIs for cache status reporting.

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Orchestrator                         │
│  [Session State] [Tool Executor] [Model Router] [Checkpoints]│
└─────────────────────────┬───────────────────────────────────┘
                          │ (step request with session_id)
┌─────────────────────────▼───────────────────────────────────┐
│                    Session Router (Ch06.6)                    │
│  [Affinity Table] [Cache Registry] [Load Monitor] [Overflow] │
└────────┬────────────────┬──────────────────┬────────────────┘
         │                │                  │
┌────────▼──────┐ ┌──────▼───────┐ ┌───────▼──────┐
│ 70B Replica 1 │ │ 70B Replica 2│ │ 8B Replica N │
│ TP=4, H100x4  │ │ TP=4, H100x4 │ │ Single H100  │
│ [KV Cache Mgr]│ │ [KV Cache Mgr]│ │ [KV Cache]   │
└───────────────┘ └──────────────┘ └──────────────┘
```

### The Critical Insight: Steps Are Not Independent Requests

In traditional serving, each request arrives with its full context (prompt + chat history) and the server computes KV cache from scratch. For agentic workloads, this is catastrophically wasteful:

**Without inter-step KV persistence:** Step 10 arrives with 30K tokens of context. The server must process all 30K tokens through prefill (at ~50K tokens/second for 70B, that is 600ms of pure prefill). Every step pays this cost, even though 95% of the tokens are identical to the previous step.

**With inter-step KV persistence (LMCache):** Step 10 arrives, the server finds the session's KV cache from step 9 (28K tokens already computed), and only needs to process the delta (2K new tokens from step 9's output + tool result). Prefill drops from 600ms to 40ms, a 15x speedup on the prefill phase.

This is why session-aware routing is non-negotiable for agentic workloads. The entire latency budget depends on KV cache reuse between steps.

### Inter-Step KV Persistence with LMCache

Drawing from Ch02.5, the LMCache integration works as follows:

```python
class AgentInferenceServer:
    """Inference server with inter-step KV cache persistence."""
    
    def __init__(self, model: TransformerModel, cache_manager: TieredKVCacheManager):
        self.model = model
        self.cache_manager = cache_manager
        self.active_sessions = {}  # session_id -> CacheHandle
    
    async def serve_step(self, request: AgentStepRequest) -> AgentStepResponse:
        session_id = request.session_id
        
        # Attempt to load existing KV cache for this session
        cached_kv = await self.cache_manager.get_kv_cache(session_id)
        
        if cached_kv is not None:
            # Fast path: only compute KV for new tokens (delta from last step)
            new_tokens = request.full_context[cached_kv.num_tokens:]
            prefill_output = self.model.prefill_incremental(
                new_tokens=new_tokens,
                existing_kv=cached_kv.kv_tensors
            )
            metrics.record("prefill_tokens", len(new_tokens))
            metrics.record("cache_hit", 1)
        else:
            # Slow path: full prefill from scratch
            prefill_output = self.model.prefill_full(request.full_context)
            metrics.record("prefill_tokens", len(request.full_context))
            metrics.record("cache_miss", 1)
        
        # Generate response tokens
        output_tokens = await self.model.generate(
            prefill_state=prefill_output,
            max_tokens=request.max_output_tokens,
            stop_sequences=request.stop_sequences
        )
        
        # Persist updated KV cache for next step
        updated_kv = prefill_output.kv_cache  # Includes new tokens
        await self.cache_manager.store_kv_cache(
            session_id=session_id,
            kv_cache=updated_kv,
            tier="gpu"  # Start on GPU, demote later if idle
        )
        
        return AgentStepResponse(
            output=output_tokens,
            step_latency_ms=prefill_output.latency_ms + output_tokens.latency_ms,
            cache_hit=cached_kv is not None,
            total_kv_tokens=updated_kv.num_tokens
        )
```

### Handling Model Transitions

A unique challenge: when a session transitions from the 8B model (step N: format tool call) to the 70B model (step N+1: reason about results), the KV caches are incompatible. The 8B model's KV cache cannot be reused by the 70B model because they have different hidden dimensions, head counts, and layer counts.

**Solution: Dual-track KV caching.**

The system maintains the 70B model's KV cache even when intermediate steps run on the 8B model. When step N runs on 8B:
1. The 8B model processes only its own step (short, ~200 tokens output)
2. The 70B KV cache is not updated (it stays frozen at step N-1 state)
3. When step N+1 returns to 70B, the delta includes both step N's output AND the 8B's response, processed through 70B's prefill

This means the 70B model may need to process 2-3 steps of delta tokens (accumulated while 8B was active), but this is still far cheaper than full recomputation.

---

## 7. Caching Strategy: The Killer Feature

### Three Levels of Cache Reuse

Caching in agentic workloads operates at three distinct levels, each providing different savings:

**Level 1: Intra-session KV reuse (inter-step persistence).**  
As described in Section 6, the same session's KV cache is reused across steps. This provides the largest per-request savings: 90%+ of tokens are cache hits for steps 5+. Every production agentic system MUST implement this.

**Level 2: Cross-session prefix caching (agent-type sharing).**  
Many agents of the same type share identical system prompts and few-shot examples. A "code review" agent always starts with the same 1,500-token system prompt. Instead of computing KV for this prefix independently for each session, the system maintains a shared prefix cache:

```python
class PrefixCacheManager:
    """Manages shared prefix KV caches across agent sessions of the same type."""
    
    def __init__(self):
        self.prefix_cache = {}  # prefix_hash -> KVCache
        self.reference_count = {}  # prefix_hash -> int
    
    def get_or_compute_prefix(self, agent_type: str, 
                               system_prompt: str, 
                               model: TransformerModel) -> KVCache:
        prefix_hash = hashlib.sha256(
            f"{agent_type}:{system_prompt}".encode()
        ).hexdigest()[:16]
        
        if prefix_hash in self.prefix_cache:
            self.reference_count[prefix_hash] += 1
            return self.prefix_cache[prefix_hash].clone()  # Copy-on-write
        
        # Compute prefix KV cache once
        prefix_kv = model.prefill_full(system_prompt)
        self.prefix_cache[prefix_hash] = prefix_kv
        self.reference_count[prefix_hash] = 1
        return prefix_kv.clone()
```

With 10K concurrent sessions across 20 agent types (average 500 sessions per type), prefix caching eliminates redundant computation of system prompts. For a 1,500-token system prompt on a 70B model, this saves:
- Per session: ~30ms of prefill time
- Aggregate: 500 sessions x 30ms = 15 seconds of GPU time saved per agent type launch wave

**Level 3: Tool-result caching (semantic deduplication).**  
Multiple agents often call the same tools with similar or identical inputs. Ten "research" agents all searching for "kubernetes pod autoscaling best practices" will get nearly identical search results. Instead of injecting these results independently into each session's context, the system can:

1. Hash the tool call (function name + arguments)
2. If another session recently made the same call, reuse the formatted result
3. More importantly: if the formatted result is already in KV cache on a specific GPU, route similar sessions to that GPU to benefit from PagedAttention's physical page sharing

```python
class ToolResultCache:
    """Caches tool call results and tracks which GPUs hold their KV representations."""
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache = {}        # tool_call_hash -> (result, timestamp, gpu_locations)
        self.ttl = ttl_seconds
    
    def get_cached_result(self, tool_name: str, args: dict) -> Optional[CachedToolResult]:
        call_hash = self.hash_call(tool_name, args)
        if call_hash in self.cache:
            result, timestamp, gpu_locs = self.cache[call_hash]
            if time.time() - timestamp < self.ttl:
                return CachedToolResult(
                    result=result,
                    gpu_locations=gpu_locs,  # For routing hints
                    is_fresh=True
                )
        return None
    
    def store_result(self, tool_name: str, args: dict, 
                     result: str, gpu_id: str):
        call_hash = self.hash_call(tool_name, args)
        if call_hash in self.cache:
            _, timestamp, gpu_locs = self.cache[call_hash]
            gpu_locs.add(gpu_id)
        else:
            self.cache[call_hash] = (result, time.time(), {gpu_id})
```

### Quantifying Cache Savings

For a typical 10-step agent task:

| Cache Level | Tokens Saved | Prefill Time Saved | Memory Saved |
|------------|-------------|-------------------|--------------|
| L1: Inter-step | ~25K (steps 2-10 reuse) | ~500ms per step x 9 = 4.5s | 0 (same cache, just reused) |
| L2: Prefix | 1.5K per session | 30ms per session | ~480MB shared across 500 sessions |
| L3: Tool result | ~2K per duplicate call | 40ms per duplicate | Varies by dedup rate |

**Total per-task savings from caching: approximately 5 seconds of LLM time**, representing a 30% reduction in total task completion time compared to a naive stateless implementation.

---

## 8. Monitoring and SLOs

### Agent-Specific Metrics

Traditional LLM serving metrics (TTFT, TPOT, throughput) are necessary but insufficient for agentic workloads. The agent loop introduces new dimensions that must be monitored:

**Per-step metrics:**
```python
class AgentStepMetrics:
    """Metrics collected for every agent inference step."""
    
    step_number: int                    # Which step in the chain
    model_used: str                     # "70B" or "8B"
    prefill_tokens: int                 # Tokens processed in prefill
    cache_hit_tokens: int               # Tokens served from KV cache
    cache_hit_ratio: float              # cache_hit_tokens / total_input_tokens
    generation_tokens: int              # Output tokens generated
    prefill_latency_ms: float           # Time for prefill phase
    generation_latency_ms: float        # Time for decode phase
    total_step_latency_ms: float        # End-to-end step time
    kv_cache_size_gb: float             # Current session KV cache size
    kv_cache_tier: str                  # Where cache was loaded from
    cache_promotion_latency_ms: float   # Time to promote cache (0 if GPU hit)
```

**Per-task metrics (aggregated across all steps):**
```python
class AgentTaskMetrics:
    """Metrics for a complete agent task (all steps)."""
    
    session_id: str
    agent_type: str
    total_steps: int
    total_task_latency_seconds: float
    total_llm_latency_seconds: float    # Sum of all step latencies
    total_tool_latency_seconds: float   # Sum of all tool call durations
    tool_call_count: int
    model_distribution: dict            # {"70B": 4, "8B": 6}
    peak_kv_cache_gb: float             # Maximum KV cache size reached
    cache_evictions: int                # Times cache was evicted mid-task
    escalation_count: int               # Times 8B -> 70B escalation triggered
    early_termination: bool             # Did agent finish before max steps?
    success: bool                       # Did agent complete its objective?
```

### SLO Dashboard Design

The monitoring system must answer these operational questions:

1. **Which step is the bottleneck?** Distribution of per-step latency by step number reveals whether early steps (planning) or late steps (synthesis with large context) dominate.

2. **Is KV cache locality working?** Cache hit ratio should be >95% for steps 2+. If it drops, session routing is broken or cache is being evicted prematurely.

3. **Are tool calls the bottleneck?** If tool_latency / total_latency > 50%, the LLM infrastructure is not the constraint. Optimize tool execution instead.

4. **What is the cost per task?** Sum of GPU-seconds across all steps, weighted by model (70B costs 8x what 8B costs). This drives pricing decisions for agentic products.

```python
class AgentSLOMonitor:
    """Monitors SLO compliance for agentic inference workloads."""
    
    def __init__(self):
        self.slos = {
            "p95_task_completion_seconds": 30.0,
            "p95_step_latency_70b_ms": 5000,
            "p95_step_latency_8b_ms": 2000,
            "min_cache_hit_ratio": 0.95,
            "max_cache_eviction_rate": 0.02,  # <2% of steps should miss cache
            "max_escalation_rate": 0.20,      # <20% of 8B attempts should escalate
        }
        self.violations = []
    
    def check_task(self, metrics: AgentTaskMetrics):
        if metrics.total_task_latency_seconds > self.slos["p95_task_completion_seconds"]:
            self.violations.append(SLOViolation(
                slo="p95_task_completion",
                actual=metrics.total_task_latency_seconds,
                threshold=30.0,
                session_id=metrics.session_id,
                diagnosis=self.diagnose_slow_task(metrics)
            ))
    
    def diagnose_slow_task(self, metrics: AgentTaskMetrics) -> str:
        """Identify why a task breached SLO."""
        if metrics.total_tool_latency_seconds > 15:
            return "tool_bottleneck"
        if metrics.cache_evictions > 2:
            return "cache_thrashing"
        if metrics.total_steps > 12:
            return "excessive_steps"
        if metrics.peak_kv_cache_gb > 15:
            return "context_overflow"
        return "compute_bound"
```

### Alert Hierarchy

**P1 (page immediately):**
- Cache hit ratio drops below 80% (system is effectively stateless, latency will 3x)
- Task completion P95 exceeds 60s (2x SLO breach)
- KV cache tier-1 (GPU) utilization >95% (evictions imminent)

**P2 (alert within 5 minutes):**
- Task completion P95 exceeds 45s (1.5x SLO, degrading)
- Escalation rate exceeds 30% (model router threshold needs tuning)
- Any 70B replica queue depth >50 requests

**P3 (daily review):**
- Cost per task trending up >10% week-over-week
- Average steps per task increasing (agents getting less efficient)
- Tool cache hit rate below 20% (tool deduplication not working)

---

## 9. Scaling and Cost Optimization

### Cost Model for Agentic Inference

The cost of an agent task is the sum of GPU-seconds consumed across all steps:

```python
def calculate_task_cost(
    steps: list[StepMetrics],
    gpu_cost_per_hour: dict = None
) -> TaskCost:
    """Calculate the infrastructure cost of a single agent task."""
    if gpu_cost_per_hour is None:
        gpu_cost_per_hour = {
            "70B_tp4": 4 * 3.50,   # 4 H100s at $3.50/hr each = $14/hr
            "8B_single": 3.50,      # 1 H100 at $3.50/hr
        }
    
    total_cost = 0.0
    for step in steps:
        gpu_seconds = step.total_step_latency_ms / 1000.0
        model_key = "70B_tp4" if step.model_used == "70B" else "8B_single"
        hourly_rate = gpu_cost_per_hour[model_key]
        step_cost = (gpu_seconds / 3600) * hourly_rate
        total_cost += step_cost
    
    return TaskCost(
        total_usd=total_cost,
        cost_70b=sum(s.cost for s in steps if s.model == "70B"),
        cost_8b=sum(s.cost for s in steps if s.model == "8B"),
        cost_per_step=total_cost / len(steps)
    )

# Example: 10-step task
# 4 steps on 70B (avg 2s each): 4 x 2s x ($14/3600) = $0.031
# 6 steps on 8B (avg 0.3s each): 6 x 0.3s x ($3.50/3600) = $0.00175
# Total: ~$0.033 per task
# At 10K concurrent sessions, each completing ~3 tasks/hour:
# 30K tasks/hour x $0.033 = $990/hour = $23,760/day
```

### Cost Optimization Strategies

**Strategy 1: Early termination detection.**

Agents often reach their goal before the maximum step count. A task that could be "done" at step 5 but runs to step 10 costs 2x. The system should actively detect completion:

```python
class EarlyTerminationDetector:
    """Detects when an agent has completed its objective to prevent waste."""
    
    def __init__(self, model: str = "8B"):
        self.classifier_model = model  # Use cheap model for classification
    
    async def should_terminate(self, session: AgentSession) -> bool:
        # Check for explicit completion signals
        last_output = session.steps[-1].output
        if any(signal in last_output for signal in 
               ["TASK_COMPLETE", "DONE", "No further action needed"]):
            return True
        
        # Check for repetition (agent is looping)
        if self.detect_repetition(session.steps[-3:]):
            return True
        
        # Cheap classifier: is the objective met?
        if len(session.steps) >= 5:  # Only after minimum steps
            completion_score = await self.classify_completion(session)
            if completion_score > 0.9:
                return True
        
        return False
    
    def detect_repetition(self, recent_steps: list) -> bool:
        """Detect if the agent is repeating the same action."""
        if len(recent_steps) < 3:
            return False
        actions = [s.action_type for s in recent_steps]
        return len(set(actions)) == 1  # Same action 3 times = loop
```

Empirically, early termination saves 15-25% of total compute by catching tasks that would otherwise run to maximum depth.

**Strategy 2: Batch tool calls (prefetch).**

When the model's output indicates a tool call, the orchestrator can speculatively prefetch results for likely subsequent tool calls:

```python
class ToolPrefetcher:
    """Speculatively prefetch tool results to reduce inter-step wait time."""
    
    def __init__(self, prediction_model: ToolPredictor):
        self.predictor = prediction_model
        self.prefetch_cache = {}
    
    async def prefetch_likely_tools(self, session: AgentSession, 
                                     current_tool_call: ToolCall):
        """While current tool executes, prefetch predicted next tools."""
        predicted_next = self.predictor.predict_next_tools(
            agent_type=session.agent_type,
            current_step=session.current_step,
            current_tool=current_tool_call
        )
        
        # Launch predicted tool calls in parallel (fire and forget)
        tasks = []
        for predicted_call in predicted_next[:3]:  # Top 3 predictions
            if predicted_call.confidence > 0.6:
                tasks.append(self.execute_and_cache(predicted_call))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def execute_and_cache(self, call: ToolCall):
        result = await self.tool_executor.execute(call)
        self.prefetch_cache[call.hash()] = result
```

When prefetch hits (predicted tool matches actual next tool call), it eliminates the tool-call wait time entirely, saving 500ms-2s per hit. With 40-60% prediction accuracy on the top-3, this saves approximately 1-2 seconds per task.

**Strategy 3: Dynamic model allocation.**

During off-peak hours (fewer concurrent sessions), reallocate 8B GPUs to serve as additional 70B capacity (by joining TP groups). During peak, split back to maximize 8B throughput:

| Time Period | 70B Nodes | 8B GPUs | Rationale |
|-------------|-----------|---------|-----------|
| Peak (9am-6pm) | 60 | 240 | Maximum concurrency needed |
| Off-peak (6pm-9am) | 80 | 160 | Fewer sessions, but deeper chains (batch jobs) |
| Weekend | 40 | 100 | Minimal interactive usage |

### Scaling Triggers

```python
class AutoScaler:
    """Autoscaling for agentic inference based on session-aware metrics."""
    
    def evaluate_scaling(self, current_metrics: ClusterMetrics) -> ScalingDecision:
        # Scale up 70B if queue depth exceeds threshold
        if current_metrics.avg_70b_queue_depth > 20:
            return ScaleUp(model="70B", nodes=max(1, current_metrics.avg_70b_queue_depth // 10))
        
        # Scale up 8B if per-step latency exceeds SLO
        if current_metrics.p95_8b_latency_ms > 1500:
            return ScaleUp(model="8B", gpus=4)
        
        # Scale up CPU tier if KV promotion latency exceeds threshold
        if current_metrics.avg_kv_promotion_ms > 200:
            return ScaleUp(tier="cpu_memory", nodes=2)
        
        # Scale down if utilization drops below threshold
        if current_metrics.gpu_utilization < 0.4 and current_metrics.sessions < 5000:
            return ScaleDown(model="70B", nodes=1)
        
        return NoOp()
```

---

## 10. Failure Modes and Recovery

### Failure Mode 1: KV Cache Eviction Mid-Session

**What happens:** A session's KV cache is evicted from all tiers (GPU, CPU, NVMe) because of memory pressure from other sessions. When the session's next step arrives, there is no cached state.

**Impact:** The session must recompute KV cache from scratch. For a step-10 session (30K tokens), this adds ~600ms of prefill latency. For step-15 (50K tokens), it adds ~1000ms. Worse, if the full history exceeds the context window, tokens are permanently lost.

**Recovery strategy:**

```python
class KVEvictionRecovery:
    """Handles recovery when a session's KV cache is evicted."""
    
    async def recover_session(self, session_id: str, 
                               full_history: list[str]) -> RecoveryResult:
        # Strategy 1: Full recomputation (if history fits in context)
        total_tokens = sum(len(tokenize(h)) for h in full_history)
        if total_tokens <= self.max_context_tokens:
            kv = await self.model.prefill_full(full_history)
            return RecoveryResult(method="full_recompute", kv=kv, 
                                  quality_loss=0.0)
        
        # Strategy 2: Checkpoint recovery (if summarization checkpoint exists)
        checkpoint = self.checkpoint_manager.get_latest(session_id)
        if checkpoint:
            # Resume from checkpoint summary + recent steps only
            recovery_context = [checkpoint.summary] + full_history[checkpoint.step:]
            kv = await self.model.prefill_full(recovery_context)
            return RecoveryResult(method="checkpoint", kv=kv,
                                  quality_loss=0.05)  # Small loss from summarization
        
        # Strategy 3: Sliding window (last N tokens only)
        recent_history = full_history[-self.max_context_tokens:]
        kv = await self.model.prefill_full(recent_history)
        return RecoveryResult(method="sliding_window", kv=kv,
                              quality_loss=0.15)  # Loses early context
```

**Prevention:** The tiered cache architecture (Section 3) with network-tier persistence should prevent total eviction for active sessions. Set eviction policy to prioritize idle sessions (>5 min inactive) over active ones.

### Failure Mode 2: Tool Call Timeout

**What happens:** The agent calls a tool (web search, API call, code execution) that hangs or takes excessively long (>30 seconds). The agent session is stuck, consuming KV cache memory without making progress.

**Impact:** The session holds GPU/CPU memory while contributing nothing. If many sessions simultaneously hit slow tools, memory fills with stalled sessions, evicting active ones.

**Recovery strategy:**

```python
class ToolTimeoutHandler:
    """Handles stuck tool calls in agent sessions."""
    
    def __init__(self, timeout_seconds: int = 30, max_retries: int = 2):
        self.timeout = timeout_seconds
        self.max_retries = max_retries
    
    async def execute_with_timeout(self, tool_call: ToolCall, 
                                    session: AgentSession) -> ToolResult:
        for attempt in range(self.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self.tool_executor.execute(tool_call),
                    timeout=self.timeout
                )
                return result
            except asyncio.TimeoutError:
                if attempt < self.max_retries:
                    continue
                # All retries exhausted: return error to agent
                return ToolResult(
                    success=False,
                    error=f"Tool '{tool_call.name}' timed out after {self.timeout}s",
                    suggestion="Try a different approach or simpler query"
                )
        
        # Demote session's KV cache to CPU tier while waiting
        await self.cache_manager.demote(session.id, target_tier="cpu")
```

**Key insight:** When a tool times out, demote the session's KV cache to CPU tier immediately. This frees GPU memory for active sessions while preserving the stalled session's state for when the tool eventually responds or the agent adapts.

### Failure Mode 3: Infinite Agent Loops

**What happens:** The agent enters a loop, repeating the same action (or alternating between two actions) without making progress. Example: agent searches for X, gets no results, decides to search for X again with slightly different phrasing, gets no results, searches again indefinitely.

**Impact:** Each loop iteration costs one LLM inference call (GPU time) and grows the KV cache (memory). An undetected loop can consume 20+ inference calls before any timeout fires.

**Detection and mitigation:**

```python
class LoopDetector:
    """Detects and breaks infinite loops in agent execution."""
    
    def __init__(self, window_size: int = 5, similarity_threshold: float = 0.85):
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
    
    def check_for_loop(self, session: AgentSession) -> Optional[LoopBreaker]:
        recent_steps = session.steps[-self.window_size:]
        if len(recent_steps) < 3:
            return None
        
        # Check 1: Identical action repetition
        actions = [s.action_type + ":" + s.tool_name for s in recent_steps]
        if len(set(actions)) == 1:
            return LoopBreaker(
                type="identical_action",
                message="Agent repeated the same action {n} times. Injecting guidance.",
                intervention=self.create_intervention(session, "stuck_same_action")
            )
        
        # Check 2: Oscillation (A -> B -> A -> B)
        if len(recent_steps) >= 4:
            pairs = [(actions[i], actions[i+1]) for i in range(len(actions)-1)]
            if pairs[0] == pairs[2] and pairs[1] == pairs[3]:
                return LoopBreaker(
                    type="oscillation",
                    message="Agent oscillating between two actions.",
                    intervention=self.create_intervention(session, "oscillating")
                )
        
        # Check 3: Output similarity (different actions but same result)
        outputs = [s.output for s in recent_steps]
        avg_similarity = self.compute_avg_similarity(outputs)
        if avg_similarity > self.similarity_threshold:
            return LoopBreaker(
                type="similar_outputs",
                message="Agent producing nearly identical outputs.",
                intervention=self.create_intervention(session, "no_progress")
            )
        
        return None
    
    def create_intervention(self, session: AgentSession, 
                            loop_type: str) -> str:
        """Create a system message to break the loop."""
        interventions = {
            "stuck_same_action": (
                "You have attempted the same action multiple times without success. "
                "Please try a completely different approach or report that this subtask "
                "cannot be completed with available tools."
            ),
            "oscillating": (
                "You are alternating between two approaches without progress. "
                "Choose one approach and commit to it, or acknowledge the task "
                "requires different tools/information."
            ),
            "no_progress": (
                "Your recent outputs are very similar, suggesting no meaningful progress. "
                "Summarize what you have learned so far and either try a fundamentally "
                "different strategy or conclude with your best available answer."
            )
        }
        return interventions.get(loop_type, "Please try a different approach.")
```

### Failure Mode 4: Context Overflow

**What happens:** An agent session's accumulated context exceeds the model's maximum context window (e.g., 128K tokens for Llama 3.1). The model cannot process the full history.

**Impact:** Without intervention, the request either errors (hard cutoff) or silently truncates early context (soft cutoff), causing the agent to lose crucial information from earlier steps.

**Recovery strategy: Progressive summarization.**

```python
class ContextOverflowManager:
    """Manages context that exceeds the model's window."""
    
    def __init__(self, max_context_tokens: int = 128000, 
                 target_after_compression: int = 80000):
        self.max_tokens = max_context_tokens
        self.target = target_after_compression
        self.summary_budget = 2000  # Tokens for each checkpoint summary
    
    async def compress_context(self, session: AgentSession, 
                                model: InferenceClient) -> CompressedContext:
        total_tokens = session.total_tokens()
        
        if total_tokens <= self.max_tokens:
            return CompressedContext(
                text=session.full_context(),
                compressed=False
            )
        
        # Progressive compression: summarize oldest steps first
        steps = session.steps
        compressed_steps = []
        running_tokens = 0
        
        # Always keep: system prompt (step 0) and last 3 steps verbatim
        system_prompt = steps[0]
        recent_steps = steps[-3:]
        budget_for_middle = self.target - system_prompt.tokens - sum(s.tokens for s in recent_steps)
        
        # Summarize middle steps in groups of 3
        middle_steps = steps[1:-3]
        for i in range(0, len(middle_steps), 3):
            group = middle_steps[i:i+3]
            group_tokens = sum(s.tokens for s in group)
            
            if running_tokens + group_tokens <= budget_for_middle:
                # Keep verbatim if space allows
                compressed_steps.extend(group)
                running_tokens += group_tokens
            else:
                # Summarize this group
                group_text = "\n".join(s.text for s in group)
                summary = await model.generate(
                    system="Summarize these agent steps concisely. Keep all key findings, "
                           "decisions, and data points. Omit verbose tool outputs.",
                    user=group_text,
                    max_tokens=self.summary_budget
                )
                compressed_steps.append(AgentStep(
                    text=f"[Summary of steps {group[0].number}-{group[-1].number}]: {summary}",
                    tokens=self.summary_budget
                ))
                running_tokens += self.summary_budget
        
        final_context = [system_prompt] + compressed_steps + recent_steps
        return CompressedContext(
            text="\n".join(s.text for s in final_context),
            compressed=True,
            original_tokens=total_tokens,
            compressed_tokens=sum(s.tokens for s in final_context)
        )
```

### Failure Mode Summary Table

| Failure Mode | Detection | Impact | Recovery Time | Quality Loss |
|---|---|---|---|---|
| KV cache eviction | Cache miss on step start | +600ms-1s latency | Immediate (recompute) | 0-15% |
| Tool timeout | Timeout trigger at 30s | Session stalled | 30s + retry | 0% |
| Infinite loop | Repetition detector | Wasted compute | Immediate (intervention) | 5% |
| Context overflow | Token count monitor | Lost early context | ~500ms (summarization) | 10-20% |
| GPU node failure | Health check timeout | All sessions on node lost | 5-30s (failover) | 0-15% |
| Network partition | Heartbeat failure | KV cache unreachable | Depends on duration | 0-100% |

---

## Summary: Design Principles for Agentic Inference

Designing inference infrastructure for agentic workloads requires rethinking every layer of the traditional serving stack. The key principles:

1. **Sessions, not requests.** The fundamental unit is an agent session (5-20 steps), not an individual inference call. Every architectural decision must optimize for session-level metrics.

2. **KV cache is the scarce resource.** GPU FLOPS are plentiful relative to HBM capacity when sessions accumulate 10K-50K tokens of state. The tiered cache hierarchy (GPU -> CPU -> NVMe -> network) is the defining feature of the architecture.

3. **Routing is king.** Session-aware routing that maintains KV cache locality provides a 10-15x improvement in prefill latency versus stateless routing. This single optimization makes the difference between meeting and missing SLOs.

4. **Two models, not one.** The natural bifurcation between reasoning (70B) and execution (8B) steps means a single-model deployment wastes either quality or cost. Model routing based on step type provides 60-70% cost savings versus all-70B.

5. **Failure is expected.** Agent loops, tool timeouts, cache evictions, and context overflow are not edge cases; they are the steady state. The system must detect and recover from each gracefully, without human intervention.

6. **Cost scales with depth.** Unlike fixed-length inference where cost is predictable, agentic cost depends on how many steps the agent takes. Early termination detection and efficient model routing are the primary cost levers.

These principles, combined with the specific techniques from earlier chapters (LMCache for persistence, PagedAttention for memory efficiency, cache-aware routing for locality, speculative decoding for latency), compose into a production system capable of serving 10,000 concurrent agent sessions within a 30-second task completion SLO at approximately $24,000/day in infrastructure cost.

---

## References to Other Chapters

- **Ch02.4**: KV cache compression (quantization, eviction policies) -- applied to deep agent sessions
- **Ch02.5**: LMCache and inter-request KV persistence -- the foundation of inter-step caching
- **Ch03**: Speculative decoding -- used for speculative model escalation (8B -> 70B)
- **Ch06**: Serving architectures -- extended with session-aware routing
- **Ch06.6**: Cache-aware routing and semantic caching -- directly applied for tool-result deduplication
- **Ch07.4**: Inference metrics and goodput -- extended with agent-specific metrics
- **Ch07.5**: Multi-region KV locality -- relevant for globally distributed agent sessions
