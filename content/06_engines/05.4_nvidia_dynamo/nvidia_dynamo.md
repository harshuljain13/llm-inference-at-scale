# NVIDIA Dynamo: The Next-Generation Distributed Inference Framework

## The Orchestration Problem No Single Engine Can Solve

You have trained a 70-billion-parameter model. You have optimized it with TensorRT-LLM, quantized it to INT8, enabled paged attention, and tuned every kernel. On a single 8xH100 node, it hums beautifully at 40 tokens per second per user. Then your traffic grows. You need three nodes. Then seven. Suddenly, the problems you face have nothing to do with kernel fusion or attention masks. They are problems of *distribution*: which node should handle this request? Where does the KV cache live? How do you move intermediate state between GPUs across a network fabric without the CPU becoming a bottleneck? How do you scale prefill capacity independently from decode capacity when their resource profiles differ by an order of magnitude?

These are the problems NVIDIA Dynamo was built to solve. Announced at GTC 2025 and released as open source, Dynamo is a *distributed inference orchestration framework* that sits above the engines you already know: vLLM, TensorRT-LLM, SGLang. It does not replace them. It coordinates them across nodes, manages KV cache routing at the cluster level, enables disaggregated prefill/decode architectures, and dynamically scales worker pools in response to load patterns.

If your inference workload fits on a single node, you do not need Dynamo. If it spans multiple nodes, or if you want the latency and cost benefits of separating prefill from decode, Dynamo provides the control plane that makes distributed inference practical rather than heroic.

## Connection to Prior Modules

From Module 05.3 (TensorRT-LLM), you understand how a single inference engine optimizes one model on one node. TRT-LLM handles kernel fusion, in-flight batching, paged KV cache management, and multi-GPU parallelism within a single machine. From Module 05.1 (vLLM), you understand PagedAttention and continuous batching. From Module 05.2 (SGLang), you understand RadixAttention and prefix-aware scheduling.

All three engines share a limitation: they think in terms of a single node. They manage GPUs within one machine, schedule requests within one process, and cache KV state in one machine's memory. Dynamo thinks in terms of a *cluster*. It manages engines across machines, schedules requests across nodes with awareness of where state lives, and moves KV cache between GPUs at hardware speed. The relationship is precise: Dynamo is to inference engines what Kubernetes is to containers. The container runtime (containerd, CRI-O) runs one container on one machine. Kubernetes orchestrates thousands of containers across hundreds of machines. Dynamo orchestrates inference engines the same way.

## Architecture: The Four Pillars of Dynamo

Dynamo's architecture decomposes distributed inference into four cooperating components. Each handles one dimension of the orchestration problem.

### The Planner: Deciding How to Decompose Work

The Planner is Dynamo's brain for workload decomposition. Given an incoming request, the Planner decides:

1. **Prefill or decode?** If this is a new prompt, it goes to a prefill worker. If it is a continuation (multi-turn conversation with existing KV cache), it routes to decode.
2. **Which parallelism strategy?** For very large prompts, the Planner may split prefill across multiple workers using pipeline or tensor parallelism.
3. **How to handle the P/D transition?** After prefill completes, the Planner coordinates KV cache transfer from the prefill worker to the assigned decode worker.

The Planner operates with global visibility. It knows the current load on every worker, the KV cache state across the cluster, and the hardware topology (which nodes are connected by NVLink vs InfiniBand vs Ethernet). This visibility enables decisions impossible for any single engine to make in isolation.

```
┌─────────────────────────────────────────────────────────────┐
│                     DYNAMO CONTROL PLANE                     │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐  │
│  │ Planner  │───▶│  Router  │───▶│  Worker Registry     │  │
│  │          │    │          │    │  (health, load, KV)  │  │
│  └──────────┘    └──────────┘    └──────────────────────┘  │
│       │               │                     │               │
└───────│───────────────│─────────────────────│───────────────┘
        │               │                     │
        ▼               ▼                     ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐
│Prefill Pool  │ │ Decode Pool  │ │     NIXL Fabric          │
│              │ │              │ │  (GPU-to-GPU transfer)   │
│ ┌──────────┐ │ │ ┌──────────┐ │ │                          │
│ │  vLLM    │ │ │ │ TRT-LLM  │ │ │  NVLink  InfiniBand     │
│ │  Engine  │ │ │ │  Engine  │ │ │  ───────  ──────────     │
│ └──────────┘ │ │ └──────────┘ │ │  900 GB/s  400 Gb/s     │
│ ┌──────────┐ │ │ ┌──────────┐ │ │                          │
│ │  vLLM    │ │ │ │ TRT-LLM  │ │ │  PCIe     Ethernet      │
│ │  Engine  │ │ │ │  Engine  │ │ │  ─────    ────────       │
│ └──────────┘ │ │ └──────────┘ │ │  64 GB/s   100 Gb/s     │
└──────────────┘ └──────────────┘ └──────────────────────────┘
```

### The Router: KV-Cache-Aware Request Dispatch

The Router is the most technically interesting component. Unlike a load balancer that distributes requests based on worker health and queue depth, Dynamo's Router maintains a *KV cache location map*: it knows which worker currently holds the KV cache for which conversation.

This knowledge enables three critical routing decisions:

**Affinity routing for multi-turn conversations.** When a user sends their fifth message in a conversation, the Router knows that Worker 7 holds the KV cache from the previous four turns. It routes the new request directly to Worker 7, avoiding a full re-prefill of the conversation history. For a 32K-token conversation, this saves ~800ms of prefill latency and gigabytes of redundant computation.

**Prefix-aware routing for shared context.** Many requests share a common system prompt or few-shot prefix. The Router tracks which workers have cached these common prefixes and routes matching requests to those workers, enabling cache reuse across unrelated conversations.

**Load-aware fallback.** When the affinity-preferred worker is overloaded, the Router makes a cost-benefit decision: is it cheaper to re-prefill the KV cache on an idle worker, or wait for the preferred worker to become available? This decision depends on conversation length, worker queue depth, and the NIXL transfer cost.

```python
# Simplified Router Decision Logic (conceptual)
class DynamoRouter:
    def __init__(self, worker_registry, kv_cache_map, nixl_fabric):
        self.registry = worker_registry
        self.kv_map = kv_cache_map
        self.nixl = nixl_fabric

    def route(self, request):
        # Step 1: Check KV cache affinity
        if request.conversation_id:
            preferred_worker = self.kv_map.get_location(
                request.conversation_id
            )
            if preferred_worker and not preferred_worker.overloaded:
                return RouteDecision(
                    target=preferred_worker,
                    action="append_decode",
                    kv_transfer=False
                )

        # Step 2: Check prefix cache matches
        prefix_hash = hash(request.system_prompt + request.few_shot)
        workers_with_prefix = self.kv_map.find_prefix(prefix_hash)
        if workers_with_prefix:
            least_loaded = min(workers_with_prefix, key=lambda w: w.queue_depth)
            if least_loaded.queue_depth < THRESHOLD:
                return RouteDecision(
                    target=least_loaded,
                    action="prefill_with_cached_prefix",
                    kv_transfer=False
                )

        # Step 3: Load-aware fallback
        if preferred_worker and self._should_wait(preferred_worker, request):
            return RouteDecision(
                target=preferred_worker,
                action="queue_and_wait",
                kv_transfer=False
            )

        # Step 4: Route to least-loaded worker, accept full prefill cost
        idle_worker = self.registry.get_least_loaded(pool="prefill")
        return RouteDecision(
            target=idle_worker,
            action="full_prefill",
            kv_transfer=True  # Will need NIXL transfer to decode worker
        )

    def _should_wait(self, worker, request):
        wait_time = worker.estimated_queue_time()
        reprefill_time = request.prompt_tokens / PREFILL_THROUGHPUT
        return wait_time < reprefill_time * 0.5
```

### Workers: Engine-Agnostic Execution Units

Dynamo workers are thin wrappers around existing inference engines. A worker consists of:

1. **An inference engine** (vLLM, TensorRT-LLM, or SGLang) handling the actual model execution
2. **A Dynamo agent** that communicates with the control plane, reports health/load metrics, and responds to orchestration commands
3. **A KV cache manager** that exposes local cache state to the Router and handles NIXL-mediated transfers

The worker abstraction is deliberately thin. Dynamo does not re-implement paged attention, continuous batching, or kernel fusion. It delegates all per-node inference to the wrapped engine and focuses exclusively on cross-node coordination.

```python
# Worker lifecycle in Dynamo
class DynamoWorker:
    def __init__(self, engine_type="vllm", role="prefill"):
        self.engine = self._init_engine(engine_type)
        self.role = role  # "prefill" or "decode"
        self.agent = DynamoAgent(self)
        self.kv_manager = KVCacheManager(self.engine)

    def _init_engine(self, engine_type):
        if engine_type == "vllm":
            return VLLMEngine(
                model="meta-llama/Llama-3.1-70B",
                tensor_parallel_size=8,  # Within this node
                gpu_memory_utilization=0.92
            )
        elif engine_type == "trtllm":
            return TRTLLMEngine(
                engine_dir="/models/llama-70b-trtllm",
                max_batch_size=256
            )

    def handle_prefill(self, request):
        """Execute prefill and report KV cache location."""
        kv_state = self.engine.prefill(request.tokens)
        self.kv_manager.register(
            conversation_id=request.conversation_id,
            kv_state=kv_state
        )
        # Notify control plane of new KV cache entry
        self.agent.report_kv_update(
            request.conversation_id,
            size_bytes=kv_state.size_bytes
        )
        return kv_state

    def handle_decode(self, request, kv_state=None):
        """Execute autoregressive decode with existing KV cache."""
        if kv_state is None:
            kv_state = self.kv_manager.get(request.conversation_id)
        return self.engine.decode(request.tokens, kv_state)
```

### NIXL: The GPU-to-GPU Transfer Fabric

NVIDIA Inference Transfer Library (NIXL) is the infrastructure that makes disaggregated serving practical at scale. Without NIXL, transferring KV cache between nodes requires:

1. Copy KV tensors from source GPU to source CPU (PCIe: ~64 GB/s)
2. Serialize and send over network (InfiniBand: ~50 GB/s effective)
3. Receive on destination CPU
4. Copy from destination CPU to destination GPU (PCIe: ~64 GB/s)

This four-hop path introduces latency that can negate the benefits of disaggregation. For a Llama 3.1 70B model with 32K context, the KV cache for one conversation is approximately:

```
KV cache size = 2 × num_layers × num_kv_heads × head_dim × seq_len × dtype_bytes
             = 2 × 80 × 8 × 128 × 32768 × 2  (FP16)
             = ~10.7 GB
```

At PCIe 5.0 speeds (64 GB/s), the CPU-mediated path takes:

```
Transfer time = 10.7 GB / 64 GB/s (GPU→CPU) + 10.7 GB / 50 GB/s (network)
              + 10.7 GB / 64 GB/s (CPU→GPU)
             ≈ 167ms + 214ms + 167ms = ~548ms
```

NIXL eliminates the CPU from this path entirely using RDMA (Remote Direct Memory Access) and GPUDirect technologies:

**NVLink path (intra-node, same DGX):** 900 GB/s bidirectional. Transfer time for 10.7 GB: ~12ms.

**InfiniBand + GPUDirect RDMA (inter-node):** 400 Gb/s (50 GB/s) directly between GPUs without CPU involvement. Transfer time for 10.7 GB: ~214ms (but no CPU copy overhead, reducing total to ~214ms vs ~548ms).

**NVLink + NVSwitch (multi-node NVLink domain, GB200 NVL72):** 1.8 TB/s aggregate. Transfer time for 10.7 GB: ~6ms.

```
┌────────────────────────────────────────────────────────────────────┐
│                    NIXL Transfer Paths                              │
├────────────────┬────────────────┬──────────────┬───────────────────┤
│ Interconnect   │ Bandwidth      │ 10.7GB KV    │ Use Case          │
│                │                │ Transfer     │                   │
├────────────────┼────────────────┼──────────────┼───────────────────┤
│ NVLink (intra) │ 900 GB/s       │ ~12ms        │ P/D split within  │
│                │                │              │ same DGX           │
├────────────────┼────────────────┼──────────────┼───────────────────┤
│ NVSwitch       │ 1.8 TB/s       │ ~6ms         │ GB200 NVL72       │
│ (GB200)        │ aggregate      │              │ rack-scale P/D    │
├────────────────┼────────────────┼──────────────┼───────────────────┤
│ IB GPUDirect   │ 400 Gb/s       │ ~214ms       │ Cross-rack P/D    │
│ RDMA           │ (50 GB/s)      │              │ disaggregation    │
├────────────────┼────────────────┼──────────────┼───────────────────┤
│ CPU-mediated   │ Limited by     │ ~548ms       │ Legacy (avoid)    │
│ (no NIXL)      │ PCIe + CPU     │              │                   │
└────────────────┴────────────────┴──────────────┴───────────────────┘
```

NIXL exposes a simple API to Dynamo workers:

```python
# NIXL transfer API (simplified)
class NIXLTransfer:
    def __init__(self, fabric_config):
        self.fabric = nixl.init(fabric_config)

    def transfer_kv_cache(
        self,
        source_gpu: GPUAddress,
        dest_gpu: GPUAddress,
        kv_tensors: List[torch.Tensor],
        async_mode: bool = True
    ) -> TransferHandle:
        """
        Direct GPU-to-GPU KV cache transfer.
        Selects optimal path automatically:
        - NVLink if same node
        - GPUDirect RDMA if cross-node with IB
        - Falls back to staged if no RDMA support
        """
        path = self.fabric.optimal_path(source_gpu, dest_gpu)
        handle = self.fabric.initiate_transfer(
            src=source_gpu,
            dst=dest_gpu,
            buffers=kv_tensors,
            path=path,
            async_op=async_mode
        )
        return handle  # Caller can await or poll completion
```

## Disaggregated Serving: Separating Prefill from Decode

Disaggregated serving is Dynamo's flagship capability and the primary reason to adopt it over simpler multi-node solutions. The insight is that prefill and decode have fundamentally different computational profiles:

### Why Disaggregate?

**Prefill is compute-bound.** Processing a 4096-token prompt requires a single forward pass through all layers with the full sequence in parallel. This saturates GPU compute (FLOPs) and benefits from maximum tensor parallelism. Prefill wants more compute, not more memory.

**Decode is memory-bandwidth-bound.** Generating each output token requires reading the entire KV cache from GPU memory but performs minimal computation (one token's worth of attention and FFN). Decode wants more memory bandwidth and capacity, not more FLOPs.

When you run both on the same GPU pool, you face a fundamental resource mismatch:

```
Mixed serving:
┌─────────────────────────────────────────────────┐
│ GPU Utilization During Mixed Serving             │
│                                                 │
│ Compute ████████████░░░░░░░░░░░░░░  45%        │
│ (FLOPs)  ▲ prefill  ▲ decode sits idle          │
│                                                 │
│ Memory  ░░░░░░████████████████████  78%         │
│ (BW)       ▲ decode ▲ prefill doesn't need it   │
│                                                 │
│ Neither resource is fully utilized!              │
└─────────────────────────────────────────────────┘

Disaggregated serving:
┌─────────────────────────────────────────────────┐
│ Prefill Workers (compute-optimized)             │
│                                                 │
│ Compute ██████████████████████████  92%         │
│ Memory  ████░░░░░░░░░░░░░░░░░░░░░  18%         │
│                                                 │
│ Decode Workers (bandwidth-optimized)            │
│                                                 │
│ Compute ███░░░░░░░░░░░░░░░░░░░░░░  12%         │
│ Memory  █████████████████████████  95%          │
└─────────────────────────────────────────────────┘
```

### The Disaggregated Pipeline

When a request arrives in a disaggregated Dynamo deployment:

**Step 1: Prefill Phase.** The Planner assigns the request to a prefill worker. The worker processes the full prompt through the model, generating the KV cache for all input tokens. This is compute-intensive but completes in one pass.

**Step 2: KV Cache Transfer.** Once prefill completes, NIXL transfers the generated KV cache from the prefill worker's GPU(s) to the assigned decode worker's GPU(s). With NVLink, this adds ~12ms for a 10GB cache. With InfiniBand GPUDirect, ~214ms.

**Step 3: Decode Phase.** The decode worker receives the KV cache and begins autoregressive generation. Each token is generated sequentially, reading the growing KV cache from memory. The decode worker is optimized for memory bandwidth with large batch sizes (hundreds of concurrent sequences sharing GPU memory bandwidth).

**Step 4: Subsequent Turns.** For multi-turn conversations, the KV cache stays on the decode worker. New messages route directly to the same decode worker (via Router affinity), which runs a minimal prefill of the new user message and continues decoding.

```python
# Disaggregated serving flow
class DisaggregatedPipeline:
    def __init__(self, planner, prefill_pool, decode_pool, nixl):
        self.planner = planner
        self.prefill_pool = prefill_pool
        self.decode_pool = decode_pool
        self.nixl = nixl

    async def serve_request(self, request):
        # Phase 1: Assign workers
        prefill_worker = self.planner.select_prefill_worker(
            prompt_length=request.prompt_tokens,
            available_workers=self.prefill_pool.healthy()
        )
        decode_worker = self.planner.select_decode_worker(
            expected_output_length=request.max_tokens,
            conversation_affinity=request.conversation_id,
            available_workers=self.decode_pool.healthy()
        )

        # Phase 2: Prefill
        kv_cache = await prefill_worker.prefill(request)

        # Phase 3: Transfer KV cache via NIXL
        transfer = self.nixl.transfer_kv_cache(
            source_gpu=prefill_worker.gpu_address,
            dest_gpu=decode_worker.gpu_address,
            kv_tensors=kv_cache.tensors
        )
        await transfer.wait()  # 6-214ms depending on interconnect

        # Phase 4: Decode
        async for token in decode_worker.decode_stream(request, kv_cache):
            yield token

        # Phase 5: Register KV location for future turns
        self.planner.register_kv_location(
            conversation_id=request.conversation_id,
            worker=decode_worker,
            kv_size=kv_cache.size_bytes
        )
```

### Dynamic Pool Scaling

The disaggregated architecture enables independent scaling of each pool. During peak hours when many new conversations start simultaneously, prefill demand spikes. During sustained usage periods, decode demand dominates as existing conversations generate long responses.

Dynamo's autoscaler monitors:
- **Prefill queue depth:** How many prompts are waiting for prefill workers
- **Decode batch utilization:** How full are decode workers' batches
- **Time-to-first-token (TTFT):** If rising, need more prefill capacity
- **Inter-token latency (ITL):** If rising, need more decode capacity or smaller batches

```python
# Autoscaler logic
class DynamoAutoscaler:
    def __init__(self, target_ttft_ms=500, target_itl_ms=30):
        self.target_ttft = target_ttft_ms
        self.target_itl = target_itl_ms

    def evaluate(self, metrics):
        decisions = []

        # Scale prefill pool based on TTFT
        if metrics.p95_ttft > self.target_ttft * 1.5:
            decisions.append(ScaleAction(
                pool="prefill",
                direction="up",
                reason=f"TTFT {metrics.p95_ttft}ms > target {self.target_ttft}ms"
            ))
        elif metrics.p95_ttft < self.target_ttft * 0.3:
            decisions.append(ScaleAction(
                pool="prefill",
                direction="down",
                reason=f"TTFT {metrics.p95_ttft}ms well below target"
            ))

        # Scale decode pool based on ITL
        if metrics.p95_itl > self.target_itl * 1.5:
            decisions.append(ScaleAction(
                pool="decode",
                direction="up",
                reason=f"ITL {metrics.p95_itl}ms > target {self.target_itl}ms"
            ))

        # Cross-pool rebalancing: convert idle prefill to decode
        if (metrics.prefill_utilization < 0.2 and
            metrics.decode_utilization > 0.85):
            decisions.append(ScaleAction(
                pool="prefill_to_decode",
                direction="rebalance",
                reason="Prefill idle, decode saturated"
            ))

        return decisions
```

## KV-Cache-Aware Routing: The Intelligence Layer

Traditional load balancers treat inference requests as stateless. Every request can go to any worker. This assumption breaks catastrophically for LLM serving because conversations carry state (the KV cache). Routing a multi-turn conversation to a new worker means either:

1. **Re-prefilling the entire conversation history** (wasting compute, adding latency)
2. **Transferring the KV cache** to the new worker (adding network overhead)

Dynamo's Router eliminates this false choice by maintaining a global KV cache location index:

### The KV Cache Map

```python
# Simplified KV Cache Map structure
kv_cache_map = {
    "conv_abc123": {
        "worker": "decode-worker-07",
        "gpu_ids": [4, 5, 6, 7],
        "size_bytes": 10_737_418_240,  # 10 GB
        "seq_len": 32768,
        "last_accessed": 1719500000,
        "model": "llama-3.1-70b",
        "prefix_hash": "sha256:9f3a..."  # For prefix matching
    },
    "conv_def456": {
        "worker": "decode-worker-03",
        "gpu_ids": [0, 1, 2, 3],
        "size_bytes": 2_684_354_560,   # 2.5 GB
        "seq_len": 8192,
        "last_accessed": 1719499800,
        "model": "llama-3.1-70b",
        "prefix_hash": "sha256:9f3a..."  # Same system prompt!
    }
}
```

### Routing Strategies

Dynamo supports multiple routing strategies that the operator configures based on workload:

**1. Strict Affinity:** Always route to the worker holding the KV cache. If that worker is down, re-prefill on another. Best for long conversations where re-prefill cost is high.

**2. Soft Affinity with Threshold:** Route to the preferred worker unless its queue exceeds a threshold. If threshold is exceeded, transfer KV cache via NIXL to a less-loaded worker. Best for balanced latency.

**3. Prefix-First Routing:** Group requests by shared prefix (system prompt hash) and route to workers that have that prefix cached. Different conversations with the same system prompt share cache entries. Best for API platforms serving many apps with distinct system prompts.

**4. Cost-Based Routing:** The Router computes the expected latency for each option (wait for preferred worker, transfer via NIXL, re-prefill on idle worker) and picks the lowest-cost path. Best for mixed workloads with varying conversation lengths.

```python
# Cost-based routing calculation
def compute_route_cost(self, request, options):
    costs = {}

    for option in options:
        if option.type == "wait_for_preferred":
            costs[option] = (
                option.worker.estimated_queue_time() +
                request.expected_decode_time
            )
        elif option.type == "nixl_transfer":
            kv_size = self.kv_map.get_size(request.conversation_id)
            transfer_time = kv_size / option.bandwidth
            costs[option] = (
                transfer_time +
                request.expected_decode_time
            )
        elif option.type == "full_reprefill":
            costs[option] = (
                request.prompt_tokens / PREFILL_THROUGHPUT +
                request.expected_decode_time
            )

    return min(costs, key=costs.get)
```

## Integration with Existing Engines

Dynamo's value proposition depends on a critical design choice: it does not reimplement inference. It wraps existing engines. This means you can adopt Dynamo without abandoning your current vLLM or TRT-LLM deployment. You keep your per-node optimizations, your quantization choices, your batching configurations. Dynamo adds the orchestration layer on top.

### The Engine Adapter Interface

```python
# Dynamo's engine adapter interface
class EngineAdapter(ABC):
    """Abstract interface that any inference engine must implement
    to work as a Dynamo worker."""

    @abstractmethod
    async def prefill(
        self,
        tokens: List[int],
        sampling_params: SamplingParams
    ) -> KVCacheHandle:
        """Process input tokens, return handle to generated KV cache."""
        ...

    @abstractmethod
    async def decode_step(
        self,
        kv_handle: KVCacheHandle,
        sampling_params: SamplingParams
    ) -> Tuple[int, KVCacheHandle]:
        """Generate one token given existing KV cache."""
        ...

    @abstractmethod
    def export_kv_cache(
        self,
        kv_handle: KVCacheHandle
    ) -> List[torch.Tensor]:
        """Export KV tensors for NIXL transfer."""
        ...

    @abstractmethod
    def import_kv_cache(
        self,
        tensors: List[torch.Tensor],
        metadata: KVMetadata
    ) -> KVCacheHandle:
        """Import KV tensors received via NIXL."""
        ...

    @abstractmethod
    def get_metrics(self) -> WorkerMetrics:
        """Report current load, memory usage, batch size."""
        ...
```

### Engine-Specific Adapters

**vLLM Adapter:** Leverages vLLM's AsyncLLMEngine API and PagedAttention block manager. KV export/import maps to vLLM's internal block tables, allowing seamless transfer of paged KV cache blocks.

**TensorRT-LLM Adapter:** Uses TRT-LLM's Executor API. KV cache export accesses the underlying CUDA tensors through TRT-LLM's KV cache manager. Import allocates new KV cache slots and copies received tensors.

**SGLang Adapter:** Integrates with SGLang's RadixAttention cache. Prefix sharing in SGLang maps naturally to Dynamo's prefix-aware routing, creating a synergy where SGLang's local prefix cache is orchestrated globally by Dynamo.

### Deployment Configuration

```yaml
# dynamo-deployment.yaml
cluster:
  name: "llama-70b-production"
  model: "meta-llama/Llama-3.1-70B-Instruct"

control_plane:
  planner:
    strategy: "disaggregated"
    prefill_timeout_ms: 5000
    max_kv_transfer_wait_ms: 300
  router:
    strategy: "cost_based"
    affinity_weight: 0.7
    load_weight: 0.3
    prefix_cache_enabled: true

pools:
  prefill:
    engine: "vllm"
    engine_config:
      tensor_parallel_size: 8
      gpu_memory_utilization: 0.90
      max_num_seqs: 64
    min_workers: 2
    max_workers: 8
    scale_metric: "ttft_p95"
    scale_target_ms: 500

  decode:
    engine: "vllm"
    engine_config:
      tensor_parallel_size: 4  # Less TP, more memory per GPU
      gpu_memory_utilization: 0.95
      max_num_seqs: 512  # Large batch for decode
    min_workers: 4
    max_workers: 16
    scale_metric: "itl_p95"
    scale_target_ms: 30

nixl:
  transport: "gpudirect_rdma"
  fallback: "staged_cpu"
  max_concurrent_transfers: 32
  compression: "none"  # KV cache not compressible effectively

autoscaler:
  evaluation_interval_s: 10
  cooldown_s: 60
  scale_up_threshold: 1.5  # 1.5x target latency
  scale_down_threshold: 0.3  # 0.3x target latency
```

## When to Use Dynamo vs Plain vLLM

The decision to introduce Dynamo is a function of scale and architectural need. Dynamo adds operational complexity (a control plane, NIXL fabric, worker registration). This complexity pays off only when distributed inference is genuinely necessary.

### Decision Matrix

```
┌────────────────────────────────────────────────────────────────────────┐
│ Scenario                          │ Recommendation   │ Why             │
├───────────────────────────────────┼──────────────────┼─────────────────┤
│ Model fits on 1 node (≤8 GPUs)   │ Plain vLLM       │ No cross-node   │
│ with acceptable throughput        │                  │ coordination    │
│                                   │                  │ needed          │
├───────────────────────────────────┼──────────────────┼─────────────────┤
│ Model fits on 1 node but traffic  │ Multiple vLLM    │ Replicate, don't│
│ exceeds single-node capacity      │ replicas + LB    │ distribute      │
├───────────────────────────────────┼──────────────────┼─────────────────┤
│ Model requires multi-node TP/PP   │ Dynamo           │ Need cross-node │
│ (>8 GPUs for one model)           │                  │ coordination    │
├───────────────────────────────────┼──────────────────┼─────────────────┤
│ Need P/D disaggregation for       │ Dynamo           │ Core capability │
│ TTFT vs ITL optimization          │                  │ of Dynamo       │
├───────────────────────────────────┼──────────────────┼─────────────────┤
│ Long multi-turn conversations     │ Dynamo           │ KV affinity     │
│ with state preservation           │                  │ routing saves   │
│                                   │                  │ re-prefill cost │
├───────────────────────────────────┼──────────────────┼─────────────────┤
│ Multiple models on shared cluster │ Dynamo           │ Unified control │
│ with dynamic resource allocation  │                  │ plane, pooling  │
├───────────────────────────────────┼──────────────────┼─────────────────┤
│ Latency-sensitive with SLA on     │ Dynamo           │ Disaggregated   │
│ both TTFT and token throughput    │                  │ pools tuned     │
│                                   │                  │ independently   │
└────────────────────────────────────────────────────────────────────────┘
```

### Scale Threshold Analysis

The crossover point where Dynamo becomes worthwhile depends on your workload:

**Compute-heavy workloads (long prompts, code generation):** Dynamo pays off at 2+ nodes because prefill dominates and benefits from dedicated compute-optimized workers.

**Memory-heavy workloads (long conversations, large batches):** Dynamo pays off at 3+ nodes because KV affinity routing avoids expensive re-prefills that grow linearly with conversation length.

**Mixed workloads (API platform, varying request sizes):** Dynamo pays off at 4+ nodes because the autoscaler can dynamically rebalance prefill/decode capacity as traffic patterns shift throughout the day.

## Comparison: Dynamo vs Alternatives

The distributed inference space has multiple solutions. Understanding where each fits prevents over-engineering.

### Feature Comparison

```
┌──────────────────┬─────────────┬──────────────┬─────────────┬──────────────┐
│ Feature          │ Dynamo      │ Ray Serve    │ llm-d       │ KServe       │
├──────────────────┼─────────────┼──────────────┼─────────────┼──────────────┤
│ P/D disagg.      │ ✅ Native   │ ❌ Manual    │ ✅ Native   │ ❌ Not       │
│                  │             │              │             │ supported    │
├──────────────────┼─────────────┼──────────────┼─────────────┼──────────────┤
│ KV-aware routing │ ✅ Built-in │ ❌ No        │ ✅ Built-in │ ❌ No        │
├──────────────────┼─────────────┼──────────────┼─────────────┼──────────────┤
│ GPU-to-GPU       │ ✅ NIXL     │ ❌ CPU only  │ ⚠️ Planned  │ ❌ No        │
│ KV transfer      │             │              │             │              │
├──────────────────┼─────────────┼──────────────┼─────────────┼──────────────┤
│ Engine support   │ vLLM,       │ Any          │ vLLM        │ Any via      │
│                  │ TRT-LLM,    │              │ primarily   │ container    │
│                  │ SGLang      │              │             │              │
├──────────────────┼─────────────┼──────────────┼─────────────┼──────────────┤
│ Autoscaling      │ ✅ Per-pool │ ✅ General   │ ⚠️ Basic    │ ✅ KPA/HPA   │
├──────────────────┼─────────────┼──────────────┼─────────────┼──────────────┤
│ K8s native       │ ⚠️ Not yet  │ ⚠️ KubeRay  │ ✅ K8s-     │ ✅ K8s-      │
│                  │ (planned)   │              │ native      │ native       │
├──────────────────┼─────────────┼──────────────┼─────────────┼──────────────┤
│ Hardware req.    │ NVLink/IB   │ Any          │ Any         │ Any          │
│                  │ for best    │              │             │              │
│                  │ performance │              │             │              │
├──────────────────┼─────────────┼──────────────┼─────────────┼──────────────┤
│ Maturity         │ Early       │ Production   │ Early       │ Production   │
│                  │ (2025)      │ (2022+)      │ (2025)      │ (2021+)      │
├──────────────────┼─────────────┼──────────────┼─────────────┼──────────────┤
│ Open source      │ ✅ Apache 2 │ ✅ Apache 2  │ ✅ Apache 2 │ ✅ Apache 2  │
├──────────────────┼─────────────┼──────────────┼─────────────┼──────────────┤
│ Best for         │ Multi-node  │ General ML   │ K8s-native  │ Multi-model  │
│                  │ LLM with    │ serving with │ LLM serving │ serving with │
│                  │ disagg. P/D │ complex      │ with P/D    │ standard     │
│                  │             │ pipelines    │ on K8s      │ MLOps        │
└──────────────────┴─────────────┴──────────────┴─────────────┴──────────────┘
```

### When Each Solution Fits

**Dynamo:** You have NVIDIA hardware with NVLink/InfiniBand, you need disaggregated P/D serving, and you want maximum hardware utilization through GPU-direct KV transfer. You are willing to operate a dedicated inference control plane.

**Ray Serve:** You have a polyglot ML pipeline (pre-processing, multiple models, post-processing) and need a general-purpose serving framework. LLM inference is one component among many. You value Ray's ecosystem (training, data, tune).

**llm-d (from the Kubernetes AI WG):** You are Kubernetes-native, want P/D disaggregation through Kubernetes primitives (CRDs, operators), and prefer infrastructure managed through kubectl and Helm. You want integration with the Gateway API for routing.

**KServe:** You have a multi-framework model serving platform (TensorFlow, PyTorch, XGBoost, LLMs) and need a unified serving layer with standard APIs (V2 Inference Protocol), autoscaling via Knative, and multi-model serving.

## Production Deployment Patterns

### Pattern 1: Single Large Model, Multi-Node

Deploy Llama 3.1 405B across 8 nodes (64 H100s total) with tensor parallelism spanning nodes:

```yaml
# Pattern: Large model multi-node
pools:
  prefill:
    workers: 2  # Each worker spans 4 nodes (32 GPUs, TP=32)
    engine_config:
      tensor_parallel_size: 32
      pipeline_parallel_size: 1
  decode:
    workers: 2  # Each worker spans 2 nodes (16 GPUs, TP=16)
    engine_config:
      tensor_parallel_size: 16
      pipeline_parallel_size: 1
      max_num_seqs: 256
```

### Pattern 2: Medium Model, High Throughput

Deploy Llama 3.1 70B with disaggregated P/D for a chatbot serving millions of daily conversations:

```yaml
# Pattern: High-throughput chatbot
pools:
  prefill:
    workers: 4  # Each worker: 1 node, TP=8
    engine_config:
      tensor_parallel_size: 8
      max_num_seqs: 64
  decode:
    workers: 12  # Each worker: 1 node, TP=4 (more memory per GPU)
    engine_config:
      tensor_parallel_size: 4
      max_num_seqs: 512
      gpu_memory_utilization: 0.95

router:
  strategy: "strict_affinity"  # Long conversations, avoid re-prefill
  eviction_policy: "lru"
  max_kv_age_s: 3600  # Evict KV after 1 hour idle
```

### Pattern 3: Multi-Model Gateway

Serve multiple models (70B main, 8B fast, coding-specialized) on a shared GPU cluster:

```yaml
# Pattern: Multi-model shared cluster
models:
  - name: "llama-3.1-70b"
    pools:
      prefill: {workers: 3, tp: 8}
      decode: {workers: 8, tp: 4}
    priority: "high"

  - name: "llama-3.1-8b"
    pools:
      prefill: {workers: 1, tp: 2}
      decode: {workers: 2, tp: 1}
    priority: "medium"

  - name: "codellama-34b"
    pools:
      prefill: {workers: 1, tp: 4}
      decode: {workers: 3, tp: 2}
    priority: "medium"

resource_sharing:
  enabled: true
  preemption: true  # High-priority model can preempt low-priority
  rebalance_interval_s: 30
```

## Performance Characteristics and Benchmarks

Based on NVIDIA's published results from GTC 2025 and early adopter reports:

### Disaggregated vs Aggregated Serving (Llama 3.1 70B, 8xH100 per node)

```
┌────────────────────────┬─────────────────┬──────────────────┬─────────────┐
│ Metric                 │ Aggregated      │ Disaggregated    │ Improvement │
│                        │ (4 nodes mixed) │ (1P + 3D nodes)  │             │
├────────────────────────┼─────────────────┼──────────────────┼─────────────┤
│ TTFT (P50)             │ 1,200 ms        │ 450 ms           │ 2.7x        │
├────────────────────────┼─────────────────┼──────────────────┼─────────────┤
│ TTFT (P95)             │ 3,800 ms        │ 980 ms           │ 3.9x        │
├────────────────────────┼─────────────────┼──────────────────┼─────────────┤
│ ITL (P50)              │ 28 ms           │ 22 ms            │ 1.3x        │
├────────────────────────┼─────────────────┼──────────────────┼─────────────┤
│ ITL (P95)              │ 85 ms           │ 35 ms            │ 2.4x        │
├────────────────────────┼─────────────────┼──────────────────┼─────────────┤
│ Throughput (tok/s)     │ 12,400          │ 18,600           │ 1.5x        │
├────────────────────────┼─────────────────┼──────────────────┼─────────────┤
│ GPU utilization (avg)  │ 58%             │ 87%              │ 1.5x        │
├────────────────────────┼─────────────────┼──────────────────┼─────────────┤
│ Cost per 1M tokens     │ $0.82           │ $0.55            │ 33% savings │
└────────────────────────┴─────────────────┴──────────────────┴─────────────┘
```

The improvements come from three sources:
1. **Prefill workers stay compute-saturated** (no interference from decode memory reads)
2. **Decode workers batch more sequences** (no interference from prefill compute bursts)
3. **KV affinity routing eliminates redundant prefills** for multi-turn conversations

### NIXL Transfer Overhead

The cost of disaggregation is the KV transfer step. NIXL minimizes but cannot eliminate this:

```
KV Transfer Overhead as % of Total Request Latency:
┌─────────────────────────┬────────────┬────────────┬────────────┐
│ Context Length           │ NVLink     │ IB RDMA    │ CPU-staged │
├─────────────────────────┼────────────┼────────────┼────────────┤
│ 2K tokens  (~335 MB)    │ 0.4 ms     │ 6.7 ms     │ 18 ms      │
│ 8K tokens  (~1.3 GB)    │ 1.5 ms     │ 27 ms      │ 72 ms      │
│ 32K tokens (~10.7 GB)   │ 12 ms      │ 214 ms     │ 548 ms     │
│ 128K tokens (~42.8 GB)  │ 48 ms      │ 856 ms     │ 2,192 ms   │
└─────────────────────────┴────────────┴────────────┴────────────┘

Impact on TTFT (added latency from disaggregation):
- NVLink: negligible (< 50ms even at 128K)
- IB RDMA: acceptable at ≤32K, challenging at 128K
- CPU-staged: unacceptable -- defeats purpose of disaggregation
```

This is why Dynamo requires NVLink or InfiniBand GPUDirect RDMA to achieve its performance targets. Without direct GPU-to-GPU transfer, the KV cache movement overhead erases the gains from disaggregation.

## Limitations and Current State

Dynamo was announced at GTC 2025 and is in active development. Understanding its current limitations is essential for deployment planning:

**Maturity:** Early-stage open source. Production deployments are limited to NVIDIA's internal workloads and select partners. API stability is not guaranteed across releases.

**Hardware dependency:** Optimal performance requires NVIDIA's NVLink and InfiniBand ecosystem. Dynamo on non-NVIDIA hardware or Ethernet-only networks loses its primary advantage (NIXL-based GPU-direct transfer).

**Kubernetes integration:** Not yet Kubernetes-native. Deployment requires manual orchestration of the control plane and worker pools. Integration with Kubernetes operators is planned but not available. llm-d (the K8s AI WG project) fills this gap for Kubernetes-native environments.

**Single-vendor:** Dynamo is NVIDIA-specific. If your inference strategy spans NVIDIA and non-NVIDIA accelerators (AMD MI300X, Google TPUs, AWS Trainium), Dynamo cannot orchestrate the heterogeneous fleet.

**Operational complexity:** Running Dynamo adds a control plane (Planner, Router, Registry) that must be monitored, scaled, and maintained. For teams that struggle to operate plain vLLM, adding Dynamo is premature.

**KV cache format standardization:** Different engines represent KV cache differently internally. Dynamo's adapters handle translation, but cross-engine transfers (e.g., prefill on TRT-LLM, decode on vLLM) add format conversion overhead and are not yet production-ready.

## Mental Model: The Kubernetes Analogy

The clearest way to internalize Dynamo's role:

```
Container Runtime (containerd/CRI-O)     =  Inference Engine (vLLM/TRT-LLM)
  - Runs one container on one machine       - Runs one model on one node
  - Manages cgroups, namespaces, storage    - Manages GPU memory, batching, KV cache
  - No cluster awareness                    - No cross-node awareness

Kubernetes                                =  NVIDIA Dynamo
  - Orchestrates containers across cluster  - Orchestrates engines across cluster
  - Scheduler places pods on nodes          - Planner places requests on workers
  - Service mesh routes traffic             - Router directs requests with KV awareness
  - HPA scales deployments                  - Autoscaler scales P/D pools
  - CNI handles pod networking              - NIXL handles GPU-to-GPU data movement

kubectl / Helm                            =  dynamo CLI / YAML configs
  - Declarative desired state               - Declarative pool configurations
  - Rolling updates, canaries               - Engine version upgrades, pool rebalancing
```

This analogy is deliberately precise. Just as you would not deploy Kubernetes for a single-server application, you should not deploy Dynamo for a single-node inference workload. The orchestration layer's complexity is justified only when the coordination problem exists.

## Key Takeaways

1. **Dynamo is an orchestration layer, not an inference engine.** It wraps vLLM, TRT-LLM, or SGLang and adds distributed coordination. You keep your engine, you add Dynamo when you need multi-node orchestration.

2. **Disaggregated prefill/decode is the primary value.** Separating compute-bound prefill from memory-bound decode improves GPU utilization from ~58% to ~87% and reduces cost per token by ~33%.

3. **NIXL makes disaggregation practical.** Without GPU-direct KV transfer, the overhead of moving 10+ GB of KV cache between nodes erases the benefits. NIXL on NVLink adds only 12ms; on InfiniBand RDMA, ~214ms.

4. **KV-aware routing saves multi-turn conversations.** By knowing which worker holds which KV cache, the Router avoids redundant re-prefills that would otherwise add hundreds of milliseconds to every conversational turn.

5. **The adoption threshold is clear.** Single node: use vLLM directly. Multiple nodes without P/D separation: consider Ray Serve or KServe. Multiple nodes with P/D disaggregation on NVIDIA hardware: Dynamo is purpose-built for this.

6. **Dynamic scaling separates pools.** Prefill and decode scale independently based on TTFT and ITL metrics respectively. The autoscaler can even rebalance workers between pools as traffic patterns shift.

7. **Current limitation: NVIDIA-only, early-stage.** Dynamo requires NVIDIA hardware with NVLink/IB for optimal performance, is not yet Kubernetes-native, and is still maturing. Plan for API changes and missing features.

## References

- NVIDIA Dynamo GitHub Repository: https://github.com/ai-dynamo/dynamo
- NVIDIA NIXL (Inference Transfer Library): https://github.com/ai-dynamo/nixl
- GTC 2025: "NVIDIA Dynamo: A Datacenter Scale Distributed Inference Framework" (Session S72451)
- NVIDIA Technical Blog: "Disaggregated Inference with NVIDIA Dynamo" (March 2025)
- Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving" (OSDI 2024): foundational paper on P/D disaggregation that Dynamo builds upon
- Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting" (ISCA 2024): phase-aware scheduling that influenced Dynamo's design
- Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve" (OSDI 2024): chunked prefill techniques related to Dynamo's prefill worker design
