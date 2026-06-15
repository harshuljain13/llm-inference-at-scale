# Multi-Model API Gateway: System Design

> **Design Brief**: A unified inference gateway that routes requests across a heterogeneous model fleet (405B, 70B, 8B, 1.5B parameters) based on SLOs, cost budgets, and quality requirements, serving 100K+ requests/day across multiple internal teams through a single API endpoint.

---

## 1. Requirements Analysis

### 1.1 Problem Statement

Large organizations deploying LLMs face a fundamental tension: no single model optimally serves all use cases. A customer support chatbot needs fast, cheap responses for routine queries but high-quality reasoning for escalations. A code review system needs deep understanding for architecture reviews but simple pattern matching for style checks. A document summarization pipeline needs different quality levels depending on whether the summary is internal-only or customer-facing.

The naive approach of deploying separate endpoints per model creates an explosion of integration points. If you have 4 models and 12 internal teams, you end up with 48 potential integration paths, each with its own authentication, rate limiting, monitoring, and billing. Teams must understand model capabilities to choose correctly, and they inevitably hard-code model choices that become stale as better models arrive.

A multi-model gateway solves this by presenting a single API surface while intelligently routing requests to the optimal model based on declared requirements rather than explicit model selection.

### 1.2 Functional Requirements

```
FR-1: Single API endpoint accepting standard chat/completion requests
FR-2: Automatic model selection based on request metadata and team policy
FR-3: Manual model override (team can pin to specific model if needed)
FR-4: Budget enforcement per team (monthly/weekly quotas in compute-dollars)
FR-5: Quality-aware routing (complexity estimation for incoming requests)
FR-6: Graceful degradation (fallback to cheaper models under load)
FR-7: Response streaming with consistent interface regardless of backend model
FR-8: Multi-turn conversation support with model consistency within sessions
```

### 1.3 Non-Functional Requirements

| Requirement | Target | Rationale |
|---|---|---|
| Gateway latency overhead | < 15ms p99 | Routing decision must be negligible vs inference time |
| Availability | 99.95% | Internal platform SLA |
| Throughput | 100K requests/day (peak 50 req/s) | Current organizational demand |
| Budget accuracy | ±2% of actual cost | Chargeback requires financial precision |
| Routing accuracy | > 90% optimal model selection | Measured by post-hoc quality evaluation |
| Fallback latency | < 100ms to detect and reroute | Users should not perceive model pool failures |

### 1.4 Model Fleet

The gateway manages four model tiers, each serving a distinct quality/cost/latency tradeoff:

| Model | Parameters | Use Case | Latency (TTFT) | Cost/1K tokens | Quality Score |
|---|---|---|---|---|---|
| Llama 405B | 405B | Complex reasoning, analysis, code generation | 800ms-2s | $0.015 | 9.2/10 |
| Llama 70B | 70B | General purpose, balanced quality/speed | 200-500ms | $0.004 | 7.8/10 |
| Llama 8B | 8B | Simple tasks, high throughput, drafts | 50-150ms | $0.0008 | 6.1/10 |
| Llama 1.5B | 1.5B | Classification, extraction, routing | 10-30ms | $0.0001 | 4.3/10 |

Quality scores are measured on an internal benchmark suite covering reasoning, factuality, instruction following, and code correctness. These scores are living metrics that update as models are fine-tuned or replaced.

### 1.5 Team Profiles and Policies

```python
# Example team policy configuration
team_policies = {
    "customer-support": {
        "default_model": "70B",
        "escalation_model": "405B",
        "budget_monthly_usd": 5000,
        "max_latency_ms": 3000,
        "min_quality_score": 7.0,
        "fallback_chain": ["70B", "8B"],  # Never fall below 8B
    },
    "code-review": {
        "default_model": "405B",
        "budget_monthly_usd": 15000,
        "max_latency_ms": 10000,  # Willing to wait for quality
        "min_quality_score": 8.5,
        "fallback_chain": ["405B", "70B"],  # Accept 70B but not lower
    },
    "internal-search": {
        "default_model": "8B",
        "budget_monthly_usd": 800,
        "max_latency_ms": 500,
        "min_quality_score": 5.0,
        "fallback_chain": ["8B", "1.5B"],
    },
    "document-processing": {
        "default_model": "1.5B",
        "escalation_trigger": "document_length > 10000",
        "escalation_model": "8B",
        "budget_monthly_usd": 200,
        "max_latency_ms": 1000,
        "min_quality_score": 4.0,
        "fallback_chain": ["1.5B"],
    },
}
```

This policy-driven approach means teams declare *what they need*, not *which model to use*. When a new model (say a 13B that matches 70B quality at 8B cost) enters the fleet, the gateway can route to it automatically without any team changing their integration.

---

## 2. Model Selection Strategy

### 2.1 The Routing Decision Problem

Model selection is a constrained optimization problem solved for every incoming request:

```
maximize: quality(model, request)
subject to:
    latency(model) <= team.max_latency_ms
    cost(model, request.tokens) <= team.remaining_budget
    quality(model) >= team.min_quality_score
    queue_depth(model) <= capacity_threshold
```

The challenge is that `quality(model, request)` depends on the specific request content. A 70B model might produce output indistinguishable from 405B for simple summarization, but noticeably worse for multi-step mathematical reasoning. The gateway must estimate request complexity *before* sending it to any model.

### 2.2 Complexity Classification

The gateway uses a lightweight classifier (running on the 1.5B model itself) to estimate request complexity on a 1-10 scale:

```python
class ComplexityClassifier:
    """
    Runs on 1.5B model with < 20ms latency.
    Classifies incoming requests into complexity tiers.
    """
    
    COMPLEXITY_PROMPT = """Rate the complexity of this request on a scale of 1-10:
    1-3: Simple factual lookup, classification, extraction
    4-6: Moderate reasoning, summarization, standard code
    7-8: Complex multi-step reasoning, analysis, architecture
    9-10: Novel problem-solving, research-level questions
    
    Request: {user_message}
    Complexity (number only):"""
    
    def __init__(self, model_pool_1_5b):
        self.model = model_pool_1_5b
        self.cache = LRUCache(maxsize=10000)  # Cache similar request patterns
    
    def classify(self, request: InferenceRequest) -> int:
        # Check cache first (semantic similarity on first 200 chars)
        cache_key = self._semantic_hash(request.messages[-1].content[:200])
        if cached := self.cache.get(cache_key):
            return cached
        
        # Fast inference on 1.5B
        score = self.model.generate(
            self.COMPLEXITY_PROMPT.format(user_message=request.messages[-1].content[:500]),
            max_tokens=3,
            temperature=0.0,
        )
        complexity = int(score.strip())
        self.cache.set(cache_key, complexity)
        return complexity
```

The classifier adds ~15ms overhead but saves significant cost by preventing simple requests from hitting expensive models. Empirically, 60-70% of requests score complexity 1-5 and can be served by the 8B or 70B model with no quality loss.

### 2.3 Model Selection Algorithm

```python
class ModelRouter:
    """
    Selects the optimal model for each request based on constraints.
    """
    
    def __init__(self, model_catalog: ModelCatalog, budget_service: BudgetService):
        self.catalog = model_catalog
        self.budget = budget_service
        self.quality_matrix = self._load_quality_matrix()
    
    def select_model(self, request: InferenceRequest, team: TeamPolicy) -> str:
        complexity = self.classifier.classify(request)
        
        # Step 1: Filter models by hard constraints
        candidates = []
        for model in self.catalog.active_models():
            if model.p50_latency_ms > team.max_latency_ms:
                continue  # Too slow for this team
            if model.quality_score < team.min_quality_score:
                continue  # Below quality floor
            estimated_cost = model.cost_per_token * request.estimated_tokens
            if not self.budget.can_afford(team.id, estimated_cost):
                continue  # Over budget
            if model.current_queue_depth > model.capacity_threshold:
                continue  # Model pool saturated
            candidates.append(model)
        
        if not candidates:
            # No model meets all constraints -- use fallback chain
            return self._fallback_selection(request, team)
        
        # Step 2: Score candidates by expected quality for this complexity
        scored = []
        for model in candidates:
            expected_quality = self.quality_matrix[model.name][complexity]
            cost_efficiency = expected_quality / model.cost_per_token
            scored.append((model, cost_efficiency))
        
        # Step 3: Select highest cost-efficiency (best quality per dollar)
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0].name
    
    def _fallback_selection(self, request: InferenceRequest, team: TeamPolicy) -> str:
        """When no model meets all constraints, relax constraints in priority order."""
        # Priority: availability > budget > latency > quality
        for model_name in team.fallback_chain:
            model = self.catalog.get(model_name)
            if model.current_queue_depth <= model.capacity_threshold * 1.5:
                return model_name
        raise NoModelAvailableError(team.id, request.id)
```

### 2.4 Quality Matrix

The quality matrix maps (model, complexity) pairs to expected output quality. This is built from offline evaluation and continuously updated with online feedback:

```
Quality Matrix (model x complexity):
              | C1-2 | C3-4 | C5-6 | C7-8 | C9-10 |
    405B      | 9.5  | 9.4  | 9.2  | 9.0  | 8.5   |
    70B       | 9.3  | 8.8  | 7.8  | 6.5  | 5.2   |
    8B        | 9.0  | 7.5  | 6.1  | 4.8  | 3.5   |
    1.5B      | 7.5  | 5.0  | 3.8  | 2.5  | 1.8   |
```

Key insight: for complexity 1-2 requests (simple extraction, classification), the 8B model scores 9.0, nearly matching the 405B at 1/19th the cost. The gateway exploits this by routing simple requests to smaller models aggressively.

### 2.5 Session Affinity

For multi-turn conversations, switching models mid-conversation creates incoherence. The gateway implements session affinity:

```python
class SessionManager:
    def get_model_for_session(self, session_id: str, request: InferenceRequest, team: TeamPolicy) -> str:
        if existing := self.session_store.get(session_id):
            # Check if current model is still available and within budget
            if self._model_still_viable(existing.model, team):
                return existing.model
            # Model no longer viable -- upgrade/downgrade with context carry
            new_model = self.router.select_model(request, team)
            self.session_store.update(session_id, model=new_model, carried_context=True)
            return new_model
        # New session -- fresh selection
        model = self.router.select_model(request, team)
        self.session_store.create(session_id, model=model)
        return model
```

---

## 3. Memory Budget

### 3.1 Per-Model Memory Requirements

Each model in the fleet has distinct memory characteristics. We calculate memory for inference (not training), accounting for model weights, KV cache, and activation memory.

**Llama 405B (FP8 quantized):**
```
Model weights: 405B params × 1 byte (FP8) = 405 GB
KV cache per request (4K context):
    - 126 layers × 2 (K+V) × 128 heads × 128 dim × 4096 tokens × 2 bytes (FP16)
    = 126 × 2 × 128 × 128 × 4096 × 2 = ~33.6 GB per batch of 32 requests
Activation memory: ~20 GB (for batch_size=32)
Framework overhead: ~15 GB (CUDA contexts, buffers)

Total per serving instance: ~475 GB
Hardware: 8× H100-80GB (640 GB total, 475 GB used = 74% utilization)
```

**Llama 70B (FP16):**
```
Model weights: 70B params × 2 bytes (FP16) = 140 GB
KV cache per request (8K context):
    - 80 layers × 2 × 64 heads × 128 dim × 8192 tokens × 2 bytes
    = ~20.5 GB per batch of 64 requests
Activation memory: ~12 GB
Framework overhead: ~8 GB

Total per serving instance: ~181 GB
Hardware: 4× A100-80GB (320 GB total, 181 GB used = 57% utilization)
```

**Llama 8B (FP16):**
```
Model weights: 8B params × 2 bytes = 16 GB
KV cache (16K context):
    - 32 layers × 2 × 32 heads × 128 dim × 16384 tokens × 2 bytes
    = ~8.6 GB per batch of 128 requests
Activation memory: ~3 GB
Framework overhead: ~3 GB

Total per serving instance: ~31 GB
Hardware: 1× A10G-24GB (with INT8 quantization: 8B × 1 byte = 8 GB weights)
    Quantized total: ~16 GB on single A10G-24GB
```

**Llama 1.5B (CPU, INT8):**
```
Model weights: 1.5B × 1 byte (INT8) = 1.5 GB
KV cache (4K context, batch 256): ~0.8 GB
Total: ~3 GB RAM
Hardware: Standard CPU instance (c7i.4xlarge, 32 GB RAM)
```

### 3.2 Fleet Memory Summary

| Model | Instances (min) | Memory/Instance | Total GPU Memory | Hardware |
|---|---|---|---|---|
| 405B | 2 (HA) | 475 GB | 1,280 GB (16× H100) | 2× 8-GPU nodes |
| 70B | 3 (capacity) | 181 GB | 960 GB (12× A100) | 3× 4-GPU nodes |
| 8B | 4 (throughput) | 16 GB | 96 GB (4× A10G) | 4× single-GPU |
| 1.5B | 2 (HA) | 3 GB RAM | N/A (CPU) | 2× c7i.4xlarge |
| **Total** | **11 instances** | | **2,336 GB GPU** | |

### 3.3 KV Cache Budget Allocation

The gateway must manage KV cache as a shared resource. Each model pool has a KV cache budget that determines maximum concurrent requests:

```python
class KVCacheBudgetManager:
    """
    Tracks KV cache memory usage per model pool.
    Rejects or queues requests when cache budget exhausted.
    """
    
    def __init__(self, model_configs: dict):
        self.budgets = {}
        for name, config in model_configs.items():
            total_gpu_memory = config.num_gpus * config.gpu_memory_gb
            weight_memory = config.weight_memory_gb
            overhead = config.overhead_gb
            self.budgets[name] = KVBudget(
                total_available_gb=total_gpu_memory - weight_memory - overhead,
                per_request_gb=config.kv_per_request_gb,
                max_concurrent=int(
                    (total_gpu_memory - weight_memory - overhead) / config.kv_per_request_gb
                ),
            )
    
    def can_accept(self, model: str, context_length: int) -> bool:
        budget = self.budgets[model]
        required_gb = budget.per_request_gb * (context_length / budget.base_context_length)
        return budget.current_used_gb + required_gb <= budget.total_available_gb
    
    def allocate(self, model: str, request_id: str, context_length: int) -> bool:
        if not self.can_accept(model, context_length):
            return False
        budget = self.budgets[model]
        required_gb = budget.per_request_gb * (context_length / budget.base_context_length)
        budget.current_used_gb += required_gb
        budget.active_requests[request_id] = required_gb
        return True
    
    def release(self, model: str, request_id: str):
        budget = self.budgets[model]
        if request_id in budget.active_requests:
            budget.current_used_gb -= budget.active_requests.pop(request_id)
```

### 3.4 Memory Oversubscription Strategy

In practice, not all concurrent requests use their full context window. The gateway implements memory oversubscription similar to airline overbooking:

```
Oversubscription ratio by model:
    405B: 1.2x (conservative -- expensive to OOM)
    70B:  1.5x (moderate -- can shed load to 8B)
    8B:   2.0x (aggressive -- cheap to restart, fast recovery)
    1.5B: 3.0x (CPU memory is abundant)
```

When actual usage exceeds the physical budget, the gateway activates memory pressure protocols: completing in-flight requests without accepting new ones, paging KV cache to host memory (adds ~50ms latency), or preempting low-priority requests.

---

## 4. Hardware Selection

### 4.1 Why Heterogeneous Hardware

A homogeneous fleet (all H100s) would simplify operations but waste money dramatically:

```
Cost analysis (AWS on-demand pricing, approximate):
    8× H100 node: ~$25/hr
    4× A100 node: ~$12/hr
    1× A10G:      ~$1.50/hr
    1× c7i.4xlarge: ~$0.60/hr

Serving 1.5B model on H100 node: $25/hr for work that needs $0.60/hr
    Waste factor: 42x overspend

Total fleet cost (heterogeneous):
    2× H100 nodes: $50/hr
    3× A100 nodes: $36/hr
    4× A10G:       $6/hr
    2× CPU:        $1.20/hr
    Total: $93.20/hr = $2,237/day

Homogeneous (all H100 nodes for equivalent capacity):
    ~8 nodes needed: $200/hr = $4,800/day
    
Savings from heterogeneous fleet: 53% ($2,563/day)
```

The principle: match hardware cost to model value. A 1.5B model producing $0.0001/token revenue should not occupy hardware costing $0.03/GPU-second.

### 4.2 Hardware-Model Mapping

| Hardware | GPU Memory | Interconnect | Best For | Why |
|---|---|---|---|---|
| H100 SXM 80GB | 80GB HBM3 | NVLink 900 GB/s | 405B (TP=8) | Highest bandwidth for all-reduce across 8 GPUs |
| A100 SXM 80GB | 80GB HBM2e | NVLink 600 GB/s | 70B (TP=4) | Sufficient bandwidth for 4-way TP, 40% cheaper than H100 |
| A10G | 24GB GDDR6 | PCIe Gen4 | 8B (quantized) | Adequate for single-GPU inference, 94% cheaper than H100 |
| CPU (c7i) | N/A (32GB RAM) | N/A | 1.5B (INT8) | No GPU needed, cheapest possible, still < 30ms TTFT |

### 4.3 Interconnect Requirements

Tensor parallelism performance depends critically on interconnect bandwidth because every transformer layer requires an all-reduce operation:

```python
def calculate_allreduce_overhead(model_hidden_dim: int, tp_degree: int, bandwidth_gbps: float) -> float:
    """
    All-reduce sends 2 * (tp-1)/tp * message_size bytes in ring-reduce.
    Called once per layer for the attention output and once for the MLP output.
    """
    message_size_bytes = model_hidden_dim * 2  # FP16
    allreduce_volume = 2 * (tp_degree - 1) / tp_degree * message_size_bytes
    time_seconds = allreduce_volume / (bandwidth_gbps * 1e9 / 8)
    return time_seconds * 1000  # Convert to ms

# 405B on H100 NVLink (900 GB/s):
#   hidden_dim=16384, tp=8
#   allreduce = 2 * 7/8 * 32768 = 57,344 bytes
#   time = 57344 / (900 * 1e9 / 8) = 0.0005ms per layer
#   126 layers × 2 (attn + MLP) = 0.13ms total allreduce overhead

# 405B on PCIe Gen4 (64 GB/s) -- hypothetical:
#   time = 57344 / (64 * 1e9 / 8) = 0.007ms per layer
#   126 layers × 2 = 1.8ms total -- 14x slower
#   At batch decode, this compounds to unacceptable latency
```

This is why H100 SXM (with NVLink) is non-negotiable for 405B: the interconnect overhead at PCIe bandwidth would add 14x more communication latency per token generated.

### 4.4 GPU Utilization Targets

| Model | GPU Type | Target Utilization | Reason |
|---|---|---|---|
| 405B | H100 | 70-80% | Leave headroom for burst; these are the most expensive GPUs |
| 70B | A100 | 80-90% | Workhouse model; high utilization is cost-efficient |
| 8B | A10G | 60-70% | Over-provisioned for burst absorption from fallback traffic |
| 1.5B | CPU | 40-50% | Classifier workload is bursty; need headroom for routing decisions |

The 8B tier is deliberately over-provisioned because it serves as the fallback target when higher-tier models are saturated. If 70B pool hits capacity and routes overflow to 8B, the 8B pool must have headroom to absorb the spike without cascading further.

---

## 5. Parallelism Strategy

### 5.1 Parallelism by Model Tier

Each model tier uses a different parallelism configuration optimized for its size and hardware:

```
405B: TP=8, PP=2 (16 GPUs total per replica)
    - TP=8 across one 8-GPU node (NVLink)
    - PP=2 across two nodes (inter-node networking)
    - Why: 405B weights alone need 405 GB; must spread across 8 GPUs minimum
    - PP=2 doubles throughput by pipelining micro-batches
    
70B: TP=4 (4 GPUs per replica)
    - TP=4 within one 4-GPU node
    - No PP needed: 4 A100s provide enough memory
    - Why: 140 GB weights fit in 4× 80GB with room for KV cache
    
8B: DP only (1 GPU per replica, N replicas)
    - Each replica is independent
    - Scale by adding more A10G instances
    - Why: 8B (quantized to INT8 = 8GB) fits on single GPU with large KV cache
    
1.5B: CPU batching (no GPU parallelism)
    - Dynamic batching up to 256 requests
    - Scale by adding more CPU instances
    - Why: Model is small enough for CPU inference; batching amortizes overhead
```

### 5.2 Pipeline Parallelism for 405B

The 405B model uses PP=2 to double throughput without requiring all 16 GPUs in a single NVLink domain:

```python
class PipelineScheduler:
    """
    1F1B (one-forward-one-backward) schedule for inference.
    In inference, there is no backward pass, so this becomes
    a simple micro-batch interleaving between pipeline stages.
    """
    
    def __init__(self, num_stages: int = 2, num_microbatches: int = 4):
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches
        # 405B has 126 layers: stage 0 gets layers 0-62, stage 1 gets 63-125
        self.layers_per_stage = 126 // num_stages  # 63 layers per stage
    
    def schedule_prefill(self, batch):
        """
        Pipeline bubble analysis for prefill:
            Without PP: 126 layers × T_layer = 126T
            With PP=2: (63 + 63) layers but pipelined
            Bubble fraction = (stages - 1) / (stages - 1 + microbatches)
                            = 1 / (1 + 4) = 20%
            Effective throughput gain: 1.6x over single-stage
        """
        microbatches = self._split_batch(batch, self.num_microbatches)
        # Stage 0 processes mb0, then mb1 while stage 1 processes mb0, etc.
        for i, mb in enumerate(microbatches):
            yield PipelineStep(stage=0, microbatch=i, data=mb)
            if i > 0:
                yield PipelineStep(stage=1, microbatch=i-1, data=None)  # Previous mb
```

### 5.3 Data Parallelism for 8B

The 8B tier achieves throughput through pure data parallelism:

```
4 replicas, each handling independent request streams

Throughput calculation:
    Single A10G with 8B INT8:
        Prefill: ~2000 tokens/second (batch=16)
        Decode: ~800 tokens/second per user (batch=16 concurrent)
    
    4 replicas total:
        Peak throughput: 4 × 800 = 3,200 decoded tokens/second
        At average 200 tokens/response: 16 responses/second = 57,600/hour
        
    For 100K requests/day, 8B handles ~60% (simple requests):
        60,000 requests / 24 hours = 2,500 requests/hour
        Well within capacity with 4 replicas (even 2 would suffice)
        Extra 2 replicas: headroom for fallback traffic from higher tiers
```

### 5.4 Gateway Abstraction Layer

The gateway completely hides parallelism complexity from API consumers. A user sends a request to `POST /v1/chat/completions` and receives a response. They never know whether their request was served by 1 GPU or 16:

```python
class ParallelismAbstraction:
    """
    Maps logical model endpoints to physical GPU configurations.
    Handles the complexity of addressing TP/PP/DP configurations.
    """
    
    def __init__(self):
        self.model_endpoints = {
            "405B": ModelEndpoint(
                replicas=[
                    TPPPReplica(tp_workers=8, pp_stages=2, node_ips=["10.0.1.1", "10.0.1.2"]),
                ],
                load_balancer="least_connections",
            ),
            "70B": ModelEndpoint(
                replicas=[
                    TPReplica(tp_workers=4, node_ip="10.0.2.1"),
                    TPReplica(tp_workers=4, node_ip="10.0.2.2"),
                    TPReplica(tp_workers=4, node_ip="10.0.2.3"),
                ],
                load_balancer="round_robin",
            ),
            "8B": ModelEndpoint(
                replicas=[
                    SingleGPUReplica(node_ip=f"10.0.3.{i}") for i in range(1, 5)
                ],
                load_balancer="least_queue_depth",
            ),
            "1.5B": ModelEndpoint(
                replicas=[
                    CPUReplica(node_ip=f"10.0.4.{i}") for i in range(1, 3)
                ],
                load_balancer="round_robin",
            ),
        }
    
    def route_request(self, model: str, request: InferenceRequest) -> str:
        endpoint = self.model_endpoints[model]
        replica = endpoint.load_balancer.select(endpoint.replicas)
        # Returns a single gRPC endpoint; the replica handles internal TP/PP coordination
        return replica.inference_endpoint
```

### 5.5 Parallelism Performance Comparison

```
Token generation latency (single request, 4K context):
    405B (TP=8, PP=2): ~45ms/token
        - Compute: 30ms (126 layers, split across 16 GPUs)
        - All-reduce: 0.13ms (NVLink, negligible)
        - Pipeline bubble: ~10ms (amortized)
        - Memory access: ~5ms
    
    70B (TP=4): ~15ms/token
        - Compute: 10ms (80 layers, split across 4 GPUs)
        - All-reduce: 0.08ms
        - Memory access: ~5ms
    
    8B (single GPU): ~8ms/token
        - Compute: 5ms (32 layers, single GPU)
        - Memory access: ~3ms
    
    1.5B (CPU): ~25ms/token
        - Compute: 20ms (CPU is slower per-op)
        - Memory access: ~5ms (DDR5 bandwidth)
        - But TTFT is fast because prefill is short for simple requests
```

The gateway uses these latency profiles to predict end-to-end response time and ensure SLO compliance before routing.


---

## 6. Serving Architecture

### 6.1 System Architecture Overview

The gateway architecture follows a layered design separating concerns into distinct, independently scalable components:

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway Layer                         │
│  (Authentication, Rate Limiting, Request Validation, Streaming) │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                         Router Layer                             │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ Complexity   │  │ Budget       │  │ SLO Matcher           │  │
│  │ Classifier   │  │ Checker      │  │ (latency + quality)   │  │
│  └─────────────┘  └──────────────┘  └───────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                     Model Pool Manager                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ 405B Pool │  │ 70B Pool │  │ 8B Pool  │  │ 1.5B Pool    │   │
│  │ (2 nodes) │  │ (3 nodes)│  │ (4 GPUs) │  │ (2 CPUs)     │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Request Flow

A complete request traverses these stages:

```python
class GatewayRequestFlow:
    """
    End-to-end request processing pipeline.
    Total gateway overhead target: < 15ms p99.
    """
    
    async def handle_request(self, raw_request: HTTPRequest) -> StreamingResponse:
        # Stage 1: Authentication + validation (< 1ms)
        team = self.auth.validate_api_key(raw_request.headers["Authorization"])
        request = self.validator.parse(raw_request.body)
        
        # Stage 2: Check semantic cache (< 3ms)
        if cached := await self.cache.lookup(request, team):
            self.metrics.record("cache_hit", model="cached", team=team.id)
            return StreamingResponse(cached.response)
        
        # Stage 3: Complexity classification (< 15ms, parallel with cache)
        complexity = await self.classifier.classify(request)
        
        # Stage 4: Model selection (< 1ms)
        model = self.router.select_model(request, team, complexity)
        
        # Stage 5: Queue management + admission control
        if not self.admission.accept(model, request):
            # Queue full -- attempt fallback
            model = self.router.fallback(model, team)
            if model is None:
                return HTTPResponse(status=429, body="All model pools at capacity")
        
        # Stage 6: Forward to model pool + stream response
        async for chunk in self.model_pools[model].generate(request):
            yield chunk
        
        # Stage 7: Post-processing (async, non-blocking)
        asyncio.create_task(self._post_process(request, model, team, complexity))
    
    async def _post_process(self, request, model, team, complexity):
        """Runs after response is delivered. Updates metrics, cache, and quality matrix."""
        self.metrics.record_completion(model, team.id, request)
        self.budget.deduct(team.id, self._calculate_cost(model, request))
        self.cache.store(request, model)  # For future semantic matches
```

### 6.3 Fallback Chain Implementation

The fallback mechanism is critical for maintaining availability when individual model pools are saturated:

```python
class FallbackChain:
    """
    Implements cascading fallback with circuit breakers.
    Each model pool has a circuit breaker that opens after N consecutive failures.
    """
    
    def __init__(self):
        self.circuit_breakers = {
            "405B": CircuitBreaker(failure_threshold=3, recovery_timeout_s=30),
            "70B": CircuitBreaker(failure_threshold=5, recovery_timeout_s=20),
            "8B": CircuitBreaker(failure_threshold=10, recovery_timeout_s=10),
            "1.5B": CircuitBreaker(failure_threshold=20, recovery_timeout_s=5),
        }
    
    def execute_with_fallback(self, request: InferenceRequest, team: TeamPolicy) -> str:
        """
        Try primary model, fall back through chain if unavailable.
        Returns the model that will actually serve this request.
        """
        for model_name in team.fallback_chain:
            cb = self.circuit_breakers[model_name]
            
            if cb.state == CircuitState.OPEN:
                continue  # Skip this model, circuit is open
            
            pool = self.model_pools[model_name]
            if pool.queue_depth < pool.max_queue_depth:
                return model_name
            else:
                cb.record_failure()  # Queue full counts as failure
                self.metrics.record("fallback_triggered", 
                    from_model=model_name, 
                    reason="queue_full",
                    team=team.id)
                continue
        
        # All models in chain exhausted
        raise AllModelsUnavailableError(team.id)
    
    def get_fallback_priority(self) -> dict:
        """
        Fallback rules:
            405B queue full -> route to 70B (quality acceptable for most)
            70B queue full -> route to 8B (quality trade-off, but available)
            8B queue full -> route to 1.5B (last resort, minimal quality)
            1.5B queue full -> reject with 429 (should never happen at CPU scale)
        """
        return {
            "405B": ["70B", "8B"],
            "70B": ["8B", "1.5B"],
            "8B": ["1.5B"],
            "1.5B": [],  # No fallback from cheapest model
        }
```

### 6.4 Circuit Breaker Pattern

```python
class CircuitBreaker:
    """
    Three states: CLOSED (normal), OPEN (rejecting), HALF_OPEN (testing recovery).
    
    State transitions:
        CLOSED -> OPEN: failure_count >= threshold
        OPEN -> HALF_OPEN: recovery_timeout elapsed
        HALF_OPEN -> CLOSED: test request succeeds
        HALF_OPEN -> OPEN: test request fails
    """
    
    def __init__(self, failure_threshold: int, recovery_timeout_s: int):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
    
    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout_s:
                self.state = CircuitState.HALF_OPEN
                return True  # Allow one test request
            return False
        if self.state == CircuitState.HALF_OPEN:
            return True  # Already allowing test
        return False
    
    def record_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

### 6.5 Load Balancing Within Model Pools

Different model tiers use different load-balancing strategies:

```python
LOAD_BALANCING_STRATEGIES = {
    "405B": "least_connections",     # Few replicas, expensive -- balance carefully
    "70B": "weighted_round_robin",   # Multiple replicas, even distribution
    "8B": "least_queue_depth",       # Many replicas, route to least loaded
    "1.5B": "round_robin",           # CPU instances are equivalent, simple rotation
}
```

For the 405B pool with only 2 replicas, least-connections ensures that a long-running generation (say, 2000 tokens at 45ms/token = 90 seconds) does not block the entire pool while the other replica sits idle.

---

## 7. Caching Strategy

### 7.1 Multi-Level Cache Architecture

The gateway implements three distinct caching layers, each operating at a different semantic level:

```
Level 1: Exact Match Cache (< 1ms lookup)
    - Hash of (system_prompt + messages) -> response
    - Hit rate: 5-8% (repeated identical queries)
    - Storage: Redis, 100GB, 7-day TTL
    
Level 2: Semantic Cache (< 5ms lookup)
    - Embedding similarity of user message -> closest cached response
    - Hit rate: 15-25% (paraphrased questions with same intent)
    - Storage: Vector DB (Qdrant), cosine similarity > 0.95 threshold
    - CRITICAL: must validate cached response still answers the new question
    
Level 3: Response Quality Cache (< 2ms lookup)
    - Maps (question_embedding, model_that_answered) -> quality_score
    - Purpose: if 8B already answered this question type well, skip 70B
    - Storage: Redis sorted sets, per-question-type
```

### 7.2 Cross-Model Semantic Cache

The most impactful optimization: if the 8B model already produced an accepted response for a semantically similar question, do not re-compute it on a larger model.

```python
class CrossModelSemanticCache:
    """
    Stores responses with quality scores. When a new request arrives,
    checks if any model has already produced a satisfactory answer.
    
    This saves the most money on common questions asked by different teams:
    Team A (using 405B) asks "What is our return policy?"
    Team B (using 70B) asks "Tell me about the return policy"
    Team C (using 8B) already answered this with score 8.5/10
    
    If team A's minimum quality threshold is 8.0, serve the cached 8B response.
    Cost savings: $0.015/K tokens -> $0 (cache hit)
    """
    
    def __init__(self, vector_db, quality_threshold: float = 0.85):
        self.vector_db = vector_db
        self.quality_threshold = quality_threshold
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast, 80ms
    
    async def lookup(self, request: InferenceRequest, team: TeamPolicy) -> Optional[CachedResponse]:
        # Embed the user message
        query_embedding = self.embedding_model.encode(request.messages[-1].content)
        
        # Search for similar past responses
        results = self.vector_db.search(
            collection="response_cache",
            vector=query_embedding,
            limit=5,
            score_threshold=0.95,  # Very high similarity required
        )
        
        for result in results:
            cached = CachedResponse.from_record(result)
            # Check if cached response meets this team's quality bar
            if cached.quality_score >= team.min_quality_score:
                # Validate the response still makes sense for this exact question
                if self._validate_response_relevance(request, cached):
                    self.metrics.record("semantic_cache_hit", 
                        original_model=cached.model,
                        would_have_used=self.router.select_model(request, team))
                    return cached
        
        return None
    
    async def store(self, request: InferenceRequest, response: str, model: str, quality_score: float):
        embedding = self.embedding_model.encode(request.messages[-1].content)
        self.vector_db.upsert(
            collection="response_cache",
            id=hash(request.messages[-1].content),
            vector=embedding,
            payload={
                "response": response,
                "model": model,
                "quality_score": quality_score,
                "timestamp": time.time(),
                "token_count": len(response.split()),
            }
        )
```

### 7.3 Prompt Template Cache

Many teams use the same system prompts repeatedly. Caching the KV states of these static prefixes eliminates redundant prefill computation:

```python
class PromptTemplateCache:
    """
    Pre-computes and caches KV states for common system prompts.
    
    Example: Customer support team sends the same 2000-token system prompt
    with every request. Without caching, each request re-processes those
    2000 tokens during prefill (~50ms on 70B). With caching, the KV state
    is pre-loaded and only the user message needs prefill.
    
    Savings: 2000/3000 tokens = 67% of prefill eliminated per request.
    At 2500 requests/hour for customer support: saves 34 GPU-hours/day.
    """
    
    def __init__(self, model_pools: dict):
        self.kv_cache_store = {}  # {(model, prompt_hash): KVState}
        self.access_counts = Counter()
    
    def get_or_compute(self, model: str, system_prompt: str) -> Optional[KVState]:
        prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()[:16]
        cache_key = (model, prompt_hash)
        
        self.access_counts[cache_key] += 1
        
        if cache_key in self.kv_cache_store:
            return self.kv_cache_store[cache_key]
        
        # Only cache prompts used > 100 times (to avoid filling memory with rare prompts)
        if self.access_counts[cache_key] > 100:
            kv_state = self.model_pools[model].compute_prefix_kv(system_prompt)
            self.kv_cache_store[cache_key] = kv_state
            return kv_state
        
        return None  # Not cached yet, compute normally
```

### 7.4 Cache Invalidation Strategy

```python
CACHE_POLICIES = {
    "exact_match": {
        "ttl_hours": 168,       # 7 days
        "max_entries": 1_000_000,
        "eviction": "LRU",
    },
    "semantic": {
        "ttl_hours": 72,        # 3 days (knowledge may change)
        "max_entries": 500_000,
        "eviction": "LRU + quality_score",  # Low-quality entries evicted first
        "revalidation": "periodic_sampling",  # Re-check 1% of cache daily
    },
    "prompt_template": {
        "ttl_hours": None,      # Never expires (system prompts are static)
        "max_entries": 1000,    # Limited to top-1000 most used prompts
        "eviction": "access_count",
        "invalidation": "on_model_update",  # Flush when model weights change
    },
}
```

---

## 8. Monitoring and SLOs

### 8.1 SLO Definitions

The gateway exposes three tiers of SLOs: platform-level, model-level, and team-level:

```yaml
# Platform SLOs (gateway team owns these)
platform_slos:
  availability: 99.95%  # < 22 minutes downtime/month
  routing_latency_p99: 15ms  # Gateway overhead excluding inference
  routing_accuracy: 90%  # Optimal model selected (measured by post-hoc eval)
  cache_hit_rate: 20%  # Minimum combined cache effectiveness

# Per-model SLOs (infra team owns these)
model_slos:
  "405B":
    ttft_p50: 800ms
    ttft_p99: 3000ms
    throughput_tokens_per_second: 50  # Per replica
    error_rate: < 0.1%
    availability: 99.9%
  "70B":
    ttft_p50: 200ms
    ttft_p99: 800ms
    throughput_tokens_per_second: 150
    error_rate: < 0.1%
    availability: 99.95%
  "8B":
    ttft_p50: 50ms
    ttft_p99: 200ms
    throughput_tokens_per_second: 500
    error_rate: < 0.05%
    availability: 99.99%  # Higher because it's the ultimate fallback
  "1.5B":
    ttft_p50: 10ms
    ttft_p99: 50ms
    throughput_tokens_per_second: 2000
    error_rate: < 0.01%
    availability: 99.99%

# Per-team SLOs (negotiated with each team)
team_slos:
  "customer-support":
    end_to_end_p99: 5000ms
    quality_floor: 7.0/10
    budget_adherence: ±5% of monthly allocation
```

### 8.2 Metrics Collection

```python
class GatewayMetricsCollector:
    """
    Emits metrics at three granularities: per-request, per-minute aggregates, per-hour rollups.
    All metrics are tagged with: model, team, complexity_tier, was_fallback, was_cached.
    """
    
    METRICS = {
        # Latency metrics
        "gateway.routing_latency_ms": "histogram",      # Time spent in routing decision
        "gateway.total_latency_ms": "histogram",        # End-to-end including inference
        "gateway.ttft_ms": "histogram",                 # Time to first token
        "gateway.tpot_ms": "histogram",                 # Time per output token
        
        # Throughput metrics
        "gateway.requests_total": "counter",            # Total requests received
        "gateway.tokens_generated_total": "counter",    # Total tokens produced
        "gateway.requests_in_flight": "gauge",          # Current active requests
        
        # Routing metrics
        "gateway.model_selected": "counter",            # Which model was chosen (by model tag)
        "gateway.fallback_triggered": "counter",        # Fallback events
        "gateway.fallback_depth": "histogram",          # How far down the chain (1=first fallback)
        "gateway.cache_hit": "counter",                 # Cache hits by level (L1/L2/L3)
        "gateway.routing_accuracy": "gauge",            # Post-hoc optimal selection rate
        
        # Cost metrics
        "gateway.cost_per_request_usd": "histogram",    # Dollar cost per request
        "gateway.team_budget_remaining_pct": "gauge",   # Budget burn rate per team
        "gateway.cost_savings_from_cache_usd": "counter",  # Money saved by cache hits
        
        # Model pool health
        "model_pool.queue_depth": "gauge",              # Per-model queue depth
        "model_pool.gpu_utilization_pct": "gauge",      # GPU compute utilization
        "model_pool.memory_utilization_pct": "gauge",   # GPU memory utilization
        "model_pool.error_rate": "gauge",               # Error rate per model
        "model_pool.circuit_breaker_state": "gauge",    # 0=closed, 1=half-open, 2=open
    }
    
    def record_request_complete(self, ctx: RequestContext):
        tags = {
            "model": ctx.model_used,
            "team": ctx.team_id,
            "complexity": ctx.complexity_tier,
            "was_fallback": ctx.was_fallback,
            "was_cached": ctx.was_cached,
        }
        self.emit("gateway.total_latency_ms", ctx.total_latency_ms, tags)
        self.emit("gateway.ttft_ms", ctx.ttft_ms, tags)
        self.emit("gateway.tokens_generated_total", ctx.output_tokens, tags)
        self.emit("gateway.cost_per_request_usd", ctx.cost_usd, tags)
```

### 8.3 Alerting Rules

```yaml
alerts:
  # Critical: page on-call
  - name: "model_pool_all_unhealthy"
    condition: "ALL circuit_breakers for a model are OPEN for > 60s"
    severity: critical
    action: "Page on-call, auto-scale if possible"
    
  - name: "gateway_error_rate_spike"
    condition: "gateway error rate > 5% for 2 consecutive minutes"
    severity: critical
    
  # Warning: Slack notification
  - name: "fallback_rate_high"
    condition: "fallback_triggered rate > 20% of requests for 5 minutes"
    severity: warning
    action: "Investigate primary model pool health"
    
  - name: "team_budget_near_exhaustion"
    condition: "team budget_remaining_pct < 10% AND days_remaining_in_period > 5"
    severity: warning
    action: "Notify team lead, suggest model tier adjustment"
    
  - name: "routing_accuracy_degraded"
    condition: "routing_accuracy < 80% over 1-hour window"
    severity: warning
    action: "Quality matrix may be stale, trigger recalibration"
    
  # Informational: dashboard only
  - name: "cache_hit_rate_low"
    condition: "combined cache_hit_rate < 15% over 6 hours"
    severity: info
    action: "Review cache configuration, TTLs may be too aggressive"
```

### 8.4 Quality Monitoring

Routing accuracy is measured through delayed evaluation: a random 5% sample of responses is re-evaluated by the 405B model to check if the routing decision was optimal:

```python
class QualityMonitor:
    """
    Runs async evaluation on sampled responses.
    Compares actual model output quality vs what the best model would have produced.
    Updates the quality matrix based on empirical results.
    """
    
    EVAL_PROMPT = """Rate this response quality on a 1-10 scale:
    Question: {question}
    Response: {response}
    
    Criteria: factual accuracy, completeness, coherence, helpfulness.
    Score (number only):"""
    
    async def evaluate_routing_decision(self, ctx: RequestContext):
        if random.random() > 0.05:  # 5% sampling
            return
        
        # Score the actual response
        actual_score = await self._evaluate(ctx.question, ctx.response)
        
        # If a cheaper model was used, check if quality was acceptable
        if ctx.model_used != "405B":
            # Would 405B have been significantly better?
            # (We don't actually re-run 405B -- too expensive. Use historical data.)
            expected_405b_score = self.quality_matrix["405B"][ctx.complexity_tier]
            quality_gap = expected_405b_score - actual_score
            
            if quality_gap > 1.5:
                # Routing was suboptimal -- update matrix
                self.quality_matrix[ctx.model_used][ctx.complexity_tier] = actual_score
                self.metrics.record("routing_accuracy", 0, tags={"model": ctx.model_used})
            else:
                self.metrics.record("routing_accuracy", 1, tags={"model": ctx.model_used})
```

---

## 9. Scaling and Cost

### 9.1 Cost Model

```python
class CostModel:
    """
    Calculates per-request cost including GPU time, memory, and platform overhead.
    Used for team chargeback and budget enforcement.
    """
    
    # Base costs per GPU-second (amortized hardware + electricity + ops)
    GPU_COST_PER_SECOND = {
        "H100": 0.0087,   # $31.25/hr / 3600
        "A100": 0.0042,   # $15.00/hr / 3600
        "A10G": 0.00053,  # $1.90/hr / 3600
        "CPU":  0.00017,  # $0.60/hr / 3600
    }
    
    # GPU-seconds per token by model (empirical, includes all overhead)
    TOKEN_GPU_SECONDS = {
        "405B": {"prefill": 0.0005, "decode": 0.045},  # 45ms/token on 8 H100s
        "70B": {"prefill": 0.0002, "decode": 0.015},
        "8B": {"prefill": 0.00005, "decode": 0.008},
        "1.5B": {"prefill": 0.00001, "decode": 0.025},  # CPU is slow per-token
    }
    
    def calculate_request_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        gpu_type = self.MODEL_TO_GPU[model]
        gpu_count = self.MODEL_TO_GPU_COUNT[model]
        
        prefill_time = input_tokens * self.TOKEN_GPU_SECONDS[model]["prefill"]
        decode_time = output_tokens * self.TOKEN_GPU_SECONDS[model]["decode"]
        total_gpu_seconds = (prefill_time + decode_time) * gpu_count
        
        raw_cost = total_gpu_seconds * self.GPU_COST_PER_SECOND[gpu_type]
        
        # Add platform overhead (routing, caching, monitoring): 15% markup
        return raw_cost * 1.15
```

### 9.2 Team Chargeback Model

```python
class ChargebackService:
    """
    Tracks spend per team, enforces budgets, generates monthly invoices.
    Budget enforcement happens in real-time at the routing layer.
    """
    
    def __init__(self, budget_store: BudgetStore):
        self.budget_store = budget_store
    
    def check_budget(self, team_id: str, estimated_cost: float) -> BudgetDecision:
        budget = self.budget_store.get(team_id)
        remaining = budget.monthly_allocation - budget.current_spend
        
        if remaining <= 0:
            return BudgetDecision(
                allowed=False,
                reason="monthly_budget_exhausted",
                suggested_action="downgrade_model",
                downgrade_to=budget.cheapest_allowed_model,
            )
        
        if remaining < estimated_cost * 10:  # Less than 10 requests worth
            return BudgetDecision(
                allowed=True,
                warning="budget_nearly_exhausted",
                remaining_usd=remaining,
                requests_remaining_estimate=int(remaining / estimated_cost),
            )
        
        return BudgetDecision(allowed=True)
    
    def enforce_budget_exhaustion(self, team_id: str, request: InferenceRequest) -> str:
        """
        When budget is exhausted, auto-downgrade to cheapest model in team's policy.
        Team gets a header in response: X-Budget-Exhausted: true
        """
        policy = self.team_policies[team_id]
        cheapest = policy.fallback_chain[-1]  # Last in chain is cheapest allowed
        
        self.metrics.record("budget_enforcement_triggered", team=team_id)
        self.notify_team_lead(team_id, 
            f"Monthly budget exhausted. Auto-downgrading to {cheapest}. "
            f"Quality may be reduced. Contact platform team for budget increase.")
        
        return cheapest
```

### 9.3 Cost Optimization Strategies

```
Strategy 1: Spot Instances for 8B and 1.5B
    - 8B on A10G spot: $0.57/hr (70% savings vs on-demand)
    - 1.5B on CPU spot: $0.18/hr (70% savings)
    - Interruption handling: pre-drain queue when spot termination notice arrives (2 min warning)
    - Never use spot for 405B/70B: warmup too slow, interruption too costly
    
    Annual savings: (4 × $0.93/hr + 2 × $0.42/hr) × 8760 = ~$40K/year

Strategy 2: Time-Based Scaling
    - Night (midnight-6am): scale 70B from 3 to 1 replica, 8B from 4 to 2
    - Weekend: scale 405B from 2 to 1 replica (batch-only mode)
    - This requires pre-warming: loading model weights takes 3-8 minutes
    
    Savings: ~30% on off-peak compute = ~$24K/year

Strategy 3: Request Batching
    - 405B: accumulate requests for 50ms before launching prefill batch
    - Batching 4 requests together: 1.8x throughput improvement
    - Latency trade-off: +50ms p50 TTFT (acceptable for high-quality tier)
    
    Savings: fewer GPU-hours needed for same throughput = ~$15K/year

Strategy 4: Aggressive Caching
    - At 20% cache hit rate across fleet:
    - Daily requests: 100,000 × 20% = 20,000 avoided inferences
    - Average inference cost: $0.002/request
    - Daily savings: $40/day = $14.6K/year
    
Total annual optimization: ~$94K/year on $815K base cost (11.5% reduction)
```

### 9.4 Fleet Scaling Rules

```python
AUTOSCALING_RULES = {
    "405B": {
        "scale_up": "queue_depth > 10 for 2 minutes",
        "scale_down": "gpu_utilization < 40% for 15 minutes",
        "min_replicas": 1,
        "max_replicas": 4,
        "warmup_time_minutes": 8,  # Loading 405GB of weights
        "cooldown_minutes": 30,    # Avoid thrashing expensive nodes
    },
    "70B": {
        "scale_up": "queue_depth > 20 for 1 minute",
        "scale_down": "gpu_utilization < 30% for 10 minutes",
        "min_replicas": 2,
        "max_replicas": 8,
        "warmup_time_minutes": 4,
        "cooldown_minutes": 15,
    },
    "8B": {
        "scale_up": "queue_depth > 50 for 30 seconds",
        "scale_down": "gpu_utilization < 20% for 5 minutes",
        "min_replicas": 2,
        "max_replicas": 16,
        "warmup_time_minutes": 1,  # Fast to load on single GPU
        "cooldown_minutes": 5,
    },
    "1.5B": {
        "scale_up": "queue_depth > 100 for 15 seconds",
        "scale_down": "cpu_utilization < 15% for 5 minutes",
        "min_replicas": 2,
        "max_replicas": 10,
        "warmup_time_minutes": 0.5,
        "cooldown_minutes": 2,
    },
}
```

---

## 10. Failure Modes

### 10.1 Model Pool Unavailable

**Scenario**: All replicas of the 70B model crash simultaneously (e.g., CUDA OOM from a bad batch, driver bug, or hardware failure).

```python
class ModelPoolFailureHandler:
    """
    Handles complete model pool failures with graceful degradation.
    """
    
    async def handle_pool_failure(self, failed_model: str):
        # 1. Open circuit breaker immediately
        self.circuit_breakers[failed_model].force_open()
        
        # 2. Drain in-flight requests (give 30s to complete)
        await self.drain_with_timeout(failed_model, timeout_s=30)
        
        # 3. Reroute all pending requests to fallback
        pending = self.queues[failed_model].drain_all()
        fallback_model = self.fallback_map[failed_model][0]
        
        for request in pending:
            if self.model_pools[fallback_model].can_accept():
                await self.model_pools[fallback_model].enqueue(request)
            else:
                request.respond_error(503, "Service temporarily degraded")
        
        # 4. Alert and begin recovery
        self.alert("critical", f"Model pool {failed_model} completely unavailable. "
                   f"Fallback to {fallback_model}. {len(pending)} requests rerouted.")
        
        # 5. Attempt auto-recovery (restart instances)
        asyncio.create_task(self._auto_recover(failed_model))
    
    async def _auto_recover(self, model: str):
        """Attempt to restart failed model pool replicas."""
        for attempt in range(3):
            try:
                await self.orchestrator.restart_pool(model)
                # Health check
                if await self.health_check(model):
                    self.circuit_breakers[model].reset()
                    self.alert("resolved", f"Model pool {model} recovered after {attempt+1} attempts")
                    return
            except Exception as e:
                await asyncio.sleep(30 * (attempt + 1))  # Exponential backoff
        
        self.alert("critical", f"Model pool {model} failed to recover after 3 attempts. "
                   "Manual intervention required.")
```

### 10.2 Budget Exhaustion Mid-Conversation

**Scenario**: A team is in the middle of a multi-turn conversation on 405B when their monthly budget runs out.

```python
class BudgetExhaustionHandler:
    """
    Handles the delicate case of budget running out mid-conversation.
    We cannot abruptly switch models without informing the user.
    """
    
    async def handle_mid_conversation_exhaustion(self, session: Session, team: TeamPolicy):
        # Option 1: Finish current response on 405B (honor in-flight commitment)
        # Option 2: Hard cut to cheaper model immediately
        # Decision: Option 1 -- finish the current response, then downgrade
        
        # Allow current generation to complete
        # But mark session for downgrade on NEXT turn
        session.budget_exhausted = True
        session.next_model = team.fallback_chain[0]  # Usually 70B
        
        # Inject system message in next response to inform user
        session.inject_system_note = (
            "[Note: This conversation has been moved to a smaller model due to "
            "team budget constraints. Response quality may differ slightly. "
            "Contact your team lead for budget increase.]"
        )
        
        # Log for visibility
        self.metrics.record("mid_conversation_downgrade",
            team=team.id,
            session_id=session.id,
            from_model=session.current_model,
            to_model=session.next_model,
            turns_completed=session.turn_count)
```

### 10.3 Quality Regression After Routing Change

**Scenario**: The quality matrix is updated and suddenly routes 40% of requests from 70B to 8B. Quality complaints spike.

```python
class QualityRegressionDetector:
    """
    Detects and auto-reverts routing changes that degrade quality.
    Uses a canary-like approach: route changes are applied gradually.
    """
    
    def __init__(self):
        self.baseline_quality = {}  # Per (team, complexity) -> rolling average
        self.change_log = []
    
    def apply_routing_change(self, change: RoutingChange):
        """
        Changes are applied to 5% of traffic initially.
        If quality holds for 1 hour, ramp to 25%, then 50%, then 100%.
        """
        self.change_log.append(change)
        change.traffic_pct = 5  # Start small
        
        # Schedule quality check
        asyncio.create_task(self._monitor_change(change))
    
    async def _monitor_change(self, change: RoutingChange):
        ramp_schedule = [(5, 3600), (25, 3600), (50, 1800), (100, 0)]
        
        for target_pct, hold_seconds in ramp_schedule:
            change.traffic_pct = target_pct
            await asyncio.sleep(hold_seconds)
            
            # Check quality metrics for affected traffic
            affected_quality = self.metrics.get_average_quality(
                team=change.affected_teams,
                since=time.time() - hold_seconds,
                traffic_group="canary"
            )
            baseline = self.baseline_quality[change.affected_key]
            
            if affected_quality < baseline - 0.5:  # Quality dropped > 0.5 points
                # ROLLBACK
                change.traffic_pct = 0
                self.alert("warning", 
                    f"Routing change reverted: quality dropped from {baseline:.1f} to "
                    f"{affected_quality:.1f} at {target_pct}% traffic")
                return
        
        # Fully rolled out successfully
        self.baseline_quality[change.affected_key] = affected_quality
```

### 10.4 Thundering Herd on 405B After 70B Failure

**Scenario**: 70B pool fails, all its traffic falls back to 405B, overwhelming the 405B pool which only has 2 replicas sized for its own traffic.

```python
class ThunderingHerdProtection:
    """
    Prevents cascading overload when a mid-tier model fails.
    
    Without protection:
        70B fails -> 3000 req/hr falls to 405B
        405B capacity: 1000 req/hr
        Result: 405B also fails -> everything falls to 8B -> quality disaster
    
    With protection:
        70B fails -> admission control limits 405B fallback to 50% of spare capacity
        Excess traffic goes to 8B (lower quality but available)
        405B stays healthy, serves most important requests
    """
    
    def __init__(self):
        self.fallback_admission_limits = {
            "405B": 0.5,  # Accept only 50% of fallback traffic (protect the crown jewel)
            "70B": 0.7,   # Accept 70% of fallback from above
            "8B": 1.0,    # Accept all fallback (designed for overflow)
        }
    
    def admit_fallback_request(self, target_model: str, request: InferenceRequest, 
                                team: TeamPolicy) -> bool:
        pool = self.model_pools[target_model]
        current_utilization = pool.queue_depth / pool.max_queue_depth
        fallback_headroom = 1.0 - current_utilization
        
        admission_limit = self.fallback_admission_limits[target_model]
        
        if fallback_headroom < (1 - admission_limit):
            # Not enough headroom -- prioritize by team importance
            if team.priority == "critical":
                return True  # Always admit critical teams
            if team.priority == "high" and random.random() < 0.5:
                return True  # 50% chance for high-priority
            return False  # Reject standard priority fallback
        
        return True
    
    def shed_load_gracefully(self, overloaded_model: str):
        """
        When thundering herd is detected, implement progressive load shedding:
        1. Stop accepting new fallback traffic
        2. Rate-limit new direct traffic to 80% of capacity
        3. Return 429 with Retry-After header for rejected requests
        4. As queue drains, gradually re-admit traffic
        """
        self.circuit_breakers[overloaded_model].set_half_open()
        self.rate_limiters[overloaded_model].reduce_to(0.8)
        
        self.alert("warning",
            f"Thundering herd detected on {overloaded_model}. "
            f"Load shedding active. Fallback traffic restricted.")
```

### 10.5 Failure Mode Summary

| Failure | Detection Time | Automatic Recovery | Impact During Recovery |
|---|---|---|---|
| Single model replica crash | < 5s (health check) | Restart in 1-8 min | Other replicas absorb load |
| Full model pool failure | < 30s (circuit breaker) | Restart attempts × 3 | Fallback chain activated |
| Budget exhaustion | Immediate (pre-check) | Auto-downgrade model | Lower quality, same availability |
| Quality regression | 1-6 hours (canary) | Auto-revert routing | 5% of users affected during canary |
| Thundering herd | < 30s (queue depth) | Load shedding + rate limit | Some requests get 429, retry later |
| Network partition | < 10s (heartbeat) | Reconnect + re-route | Brief queue buildup, then resolves |
| GPU OOM | Immediate (CUDA error) | Kill batch, reduce batch size | ~5s of dropped tokens, then recovery |

---

## Key Takeaways

1. **Route by need, not by name**: Teams declare quality/latency/cost requirements; the gateway selects models. This enables transparent model upgrades without integration changes.

2. **Heterogeneous hardware saves 53%**: Matching GPU cost to model value prevents expensive hardware from serving cheap inference.

3. **Complexity classification pays for itself**: 15ms overhead per request saves thousands of dollars daily by routing simple queries to cheap models.

4. **Cross-model caching is the highest-ROI optimization**: If any model already answered a question well, serve the cached response regardless of which tier the new request targets.

5. **Fallback chains must be depth-limited**: Cascading failures happen when every model falls back to the next. Admission control on fallback traffic prevents the thundering herd from taking down the entire fleet.

6. **Budget enforcement is a feature, not a bug**: Teams that exhaust budgets get auto-downgraded rather than cut off. This maintains availability while respecting cost constraints.

7. **Quality monitoring closes the loop**: Without measuring whether routing decisions were optimal, the system cannot improve. The 5% evaluation sample continuously refines the quality matrix.
