# 7.7 Kubernetes-Native LLM Inference Infrastructure

## The Impedance Mismatch: Why Kubernetes Wasn't Built for This

Kubernetes was designed in 2014 for a world of stateless HTTP microservices. A pod serves a request, forgets about it, and the next request can land on any replica. Scaling means adding identical pods behind a load balancer. Scheduling means bin-packing CPU and memory into nodes. This model powered a decade of web infrastructure, but it fundamentally breaks when you try to serve large language models.

LLM inference is different in three ways that matter:

**It is stateful.** Every ongoing generation maintains a KV cache that can consume 10-40 GB of GPU memory per active request. Rerouting a request mid-generation means either discarding that cache (adding seconds of latency for re-prefill) or migrating it across the network (consuming precious GPU interconnect bandwidth). The "cattle not pets" philosophy of Kubernetes directly conflicts with the economics of KV cache preservation.

**It is GPU-bound with non-fungible resources.** Not all GPUs are equal. An H100 with NVLink is not interchangeable with an A100 on PCIe. Multi-node tensor parallelism requires specific topology awareness. The default Kubernetes scheduler sees GPUs as opaque integer resources ("give me 4 GPUs") without understanding memory capacity, interconnect topology, or compute capability differences.

**It is latency-sensitive with variable cost.** A single inference request might consume anywhere from 100 tokens to 100,000 tokens, making capacity planning fundamentally different from web services where requests have roughly predictable cost. Token-level rate limiting, model-aware routing, and prefill/decode separation all require infrastructure primitives that simply did not exist in Kubernetes before 2025.

This module covers the new Kubernetes primitives and projects that emerged in 2025-2026 specifically to address these gaps. By the end, you will understand how NVIDIA, Google, Microsoft, and the broader CNCF community are rebuilding the Kubernetes scheduling, networking, and orchestration layers for inference workloads.

> **Back-reference:** From Module 07.2 (EKS + KServe), you know how to deploy a model on Kubernetes using InferenceService custom resources, autoscaling based on GPU metrics, and basic request routing. This module goes deeper into the inference-specific extensions that make Kubernetes actually good at serving LLMs, rather than merely capable of it.

---

## KAI Scheduler: GPU-Aware Scheduling for AI Workloads

### Why the Default Scheduler Fails

The default `kube-scheduler` makes placement decisions based on resource requests: CPU cores, memory bytes, and integer device counts. When a pod requests `nvidia.com/gpu: 2`, the scheduler finds a node with 2 available GPUs and places the pod there. This is fundamentally insufficient for inference workloads for three reasons:

1. **No fractional GPU support.** A small model (7B parameters in FP16, ~14 GB) on an 80 GB H100 wastes 82% of the GPU memory. You cannot request 0.2 of a GPU through standard device plugins.

2. **No topology awareness.** Tensor parallelism across 4 GPUs connected via NVLink is 10x faster than across 4 GPUs on separate PCIe switches. The default scheduler has no concept of GPU interconnect topology.

3. **No workload prioritization.** Production inference serving a customer-facing application should preempt a background batch fine-tuning job. Standard Kubernetes PriorityClasses are too coarse for AI workload management.

4. **No queue-based fairness.** When multiple teams share a GPU cluster, you need fair-share allocation with borrowing and lending semantics, not just hard resource quotas.

### How KAI Scheduler Works

KAI Scheduler (formerly known as the RunAI Scheduler, donated to CNCF Sandbox in early 2026 by NVIDIA after their RunAI acquisition) replaces `kube-scheduler` for GPU workloads with a purpose-built scheduling engine. It introduces several concepts absent from vanilla Kubernetes:

**Fractional GPU Allocation.** KAI allows pods to request GPU memory in granular units (e.g., 16 GB of an 80 GB GPU) rather than whole devices. Multiple inference pods can share a single physical GPU, each with guaranteed memory isolation enforced at the driver level. This is critical for serving multiple small models (7B-13B parameters) on expensive hardware.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: llama-7b-inference
  annotations:
    kai.scheduler/gpu-memory: "16Gi"
    kai.scheduler/gpu-fraction: "0.2"
spec:
  containers:
  - name: inference
    image: vllm/vllm-openai:latest
    resources:
      limits:
        nvidia.com/gpu-fraction: "0.2"  # KAI-specific resource type
```

**Queue-Based Resource Management.** Instead of Kubernetes namespaces with ResourceQuotas (which are binary: you either fit or you don't), KAI introduces hierarchical queues with borrowing semantics. A team's inference workloads can temporarily borrow unused GPUs from another team's allocation, then gracefully yield them back when the owner needs them.

```yaml
apiVersion: kai.nvidia.com/v1
kind: Queue
metadata:
  name: recommendation-team
spec:
  parent: ml-platform
  guaranteedGpus: 8
  maxGpus: 16          # Can borrow up to 16 when cluster has spare capacity
  priority: 100
  preemptible: false   # Production inference cannot be preempted
---
apiVersion: kai.nvidia.com/v1
kind: Queue
metadata:
  name: research-team
spec:
  parent: ml-platform
  guaranteedGpus: 4
  maxGpus: 32          # Research can burst aggressively on spare capacity
  priority: 50
  preemptible: true    # But yields GPUs when production needs them
```

**Workload-Type Scheduling Policies.** KAI distinguishes between inference (latency-sensitive, long-running, needs guaranteed resources), training (throughput-oriented, can be preempted and checkpointed), and interactive (Jupyter notebooks, short-lived, best-effort). Each workload type gets different scheduling behavior:

| Workload Type | Preemptible | Gang Scheduled | Topology Aware | Consolidation |
|---|---|---|---|---|
| Inference | No | Optional (TP) | Yes (NVLink) | Spread (HA) |
| Training | Yes (checkpoint) | Yes (all-or-nothing) | Yes (NVLink+IB) | Pack (locality) |
| Interactive | Yes (immediate) | No | No | Pack (efficiency) |

**Topology-Aware Placement.** When a pod requests 4 GPUs for tensor parallelism, KAI ensures those GPUs are on the same NVLink domain rather than scattered across PCIe switches. It maintains a real-time topology map of every node's GPU interconnect structure and factors this into scheduling decisions. The performance difference is not marginal: NVLink provides 900 GB/s bidirectional bandwidth versus ~64 GB/s for PCIe Gen5, making correct placement a 14x throughput difference for communication-bound operations like all-reduce in tensor parallelism.

### Deployment Considerations

KAI Scheduler runs as a secondary scheduler alongside (or replacing) `kube-scheduler`. Pods that need GPU-aware scheduling specify `schedulerName: kai-scheduler` in their spec. Non-GPU pods continue using the default scheduler unchanged. This makes adoption incremental: you can migrate inference workloads to KAI without disrupting the rest of your cluster.

The scheduler requires NVIDIA's GPU Operator for device visibility and integrates with NVIDIA's DCGM (Data Center GPU Manager) for real-time GPU health and utilization metrics. In clusters running mixed CPU and GPU workloads, KAI handles only the GPU scheduling decisions, delegating CPU/memory placement to the standard scheduler.

---

## Dynamic Resource Allocation (DRA): The Future of GPU Management in Kubernetes

### The Device Plugin Limitation

Since Kubernetes 1.8 (2017), GPUs have been exposed to pods through the device plugin framework. A device plugin runs as a DaemonSet on each node, advertises available GPUs as extended resources (`nvidia.com/gpu: 8`), and handles device assignment when pods are scheduled. This mechanism has a fundamental limitation: it treats GPUs as opaque integers with no structured attributes.

You cannot express "I need 2 GPUs with at least 80 GB memory each, connected via NVLink, on a node with at least 400 GB/s aggregate memory bandwidth." The device plugin API has no vocabulary for these constraints. Workarounds involve node labels (`gpu-type: h100-sxm`), node affinity rules, and complex admission webhooks, creating a fragile tower of configuration that breaks when clusters have heterogeneous GPU hardware.

### What DRA Changes

Dynamic Resource Allocation (graduated to beta in Kubernetes 1.31, GA-track in 1.32) introduces structured, claim-based resource management that replaces the integer-counter model with a rich attribute system. NVIDIA donated their DRA driver to the CNCF in 2025, making it the community-standard way to manage GPU resources.

The conceptual shift is from "give me N GPUs" to "give me GPU resources matching these specific requirements":

```yaml
apiVersion: resource.k8s.io/v1beta1
kind: ResourceClaim
metadata:
  name: inference-gpus
spec:
  devices:
    requests:
    - name: gpu-group
      deviceClassName: gpu.nvidia.com
      count: 4
      selectors:
      - cel:
          expression: >
            device.attributes["gpu.nvidia.com"].memory >= 80 &&
            device.attributes["gpu.nvidia.com"].architecture == "hopper" &&
            device.attributes["gpu.nvidia.com"].nvlinkConnected == true
    constraints:
    - requests: ["gpu-group"]
      matchAttribute: "gpu.nvidia.com/nvlinkDomain"  # All 4 on same NVLink domain
```

This CEL (Common Expression Language) based selector system allows expressing arbitrarily complex constraints that were previously impossible with device plugins. The DRA controller evaluates these constraints against the cluster's actual GPU inventory and either satisfies the claim or reports it unsatisfiable.

### Key DRA Capabilities for Inference

**Fractional GPU Sharing Under Community Governance.** Unlike vendor-specific fractional GPU solutions (NVIDIA MPS, MIG, RunAI fractions), DRA provides a standardized API for GPU partitioning that any vendor can implement. This means your workload definitions are portable across NVIDIA, AMD, and Intel GPU drivers.

**Multi-Node Resource Claims.** For very large models requiring tensor parallelism across multiple nodes (e.g., Llama 405B across 8 nodes with 8 GPUs each), DRA supports claims that span node boundaries. The scheduler co-schedules pods across nodes while ensuring interconnect requirements (InfiniBand fabric, specific switch topology) are met.

**GPU Health and Lifecycle Management.** DRA drivers can report GPU health status (ECC errors, thermal throttling, memory degradation) as structured attributes. Scheduling decisions incorporate hardware health: a GPU showing early signs of failure gets deprioritized for production inference while remaining available for best-effort research workloads.

**Preparation and Cleanup Hooks.** Before a pod starts using an allocated GPU, the DRA driver can execute preparation steps (loading specific firmware, configuring MIG partitions, setting power profiles for inference versus training). When the pod terminates, cleanup hooks reset the GPU state. This lifecycle management was impossible with device plugins.

### Migration Path from Device Plugins

The DRA and device plugin systems coexist during the transition period. A practical migration path:

1. Deploy the NVIDIA DRA driver alongside existing device plugins
2. New inference workloads use ResourceClaims with structured attributes
3. Existing workloads continue using `nvidia.com/gpu` integer requests
4. Gradually migrate as you gain confidence in DRA scheduling decisions
5. Remove device plugins once all workloads use DRA claims

The key insight is that DRA does not just replace device plugins with a fancier API. It fundamentally changes the scheduling model from "find a node with free GPUs" to "find GPUs anywhere in the cluster that match my requirements," enabling much more efficient utilization of heterogeneous GPU fleets.

---

## Gateway API Inference Extension: Model-Aware Routing at the Network Layer

### Why Standard Ingress Fails for Inference

Kubernetes Ingress and even the newer Gateway API route traffic based on HTTP attributes: hostname, path, headers. An inference platform needs routing based on attributes that do not exist in the HTTP layer:

- **Model identity.** Route requests for `llama-3.1-70b` to one pool and `mistral-7b` to another, potentially based on the `model` field inside a JSON request body.
- **Token-aware load balancing.** A request generating 4,000 tokens consumes 100x more compute than one generating 40 tokens. Round-robin load balancing creates severe imbalance.
- **KV cache affinity.** Multi-turn conversations benefit from routing to the same backend that holds the KV cache from previous turns.
- **Capacity-aware admission.** When all backends for a model are at capacity, requests should queue with backpressure rather than overloading backends and causing latency spikes.

Standard Kubernetes networking primitives have no concept of these requirements.

### The InferencePool and InferenceModel CRDs

The Gateway API Inference Extension (developed under Kubernetes SIG Network, reaching v0.3 in early 2026) introduces two custom resources that bridge the gap between Kubernetes networking and inference serving:

```yaml
apiVersion: inference.networking.k8s.io/v1alpha1
kind: InferencePool
metadata:
  name: large-models
spec:
  selector:
    matchLabels:
      pool: large-gpu
  targetPortNumber: 8000
  endpointPickerConfig:
    extensionRef:
      name: kv-aware-picker    # Custom endpoint selection logic
---
apiVersion: inference.networking.k8s.io/v1alpha1
kind: InferenceModel
metadata:
  name: llama-70b
spec:
  modelName: meta-llama/Llama-3.1-70B-Instruct
  poolRef:
    name: large-models
  criticality: Critical        # Never shed this model's traffic
  targetRequests: 100          # Desired concurrent requests per pod
```

**InferencePool** defines a group of serving endpoints (pods) with a shared GPU class and an endpoint picker strategy. The picker can be a simple algorithm (least-connections, least-tokens-in-flight) or a sophisticated extension that considers KV cache state, queue depth, and model-specific metrics.

**InferenceModel** maps a logical model name to a pool, with criticality levels that determine shedding behavior under overload. Critical models never shed traffic (they queue instead), while best-effort models can be temporarily unavailable to protect critical model SLOs.

### Integration with Gateway API HTTPRoutes

The inference extension composes with standard Gateway API resources. An HTTPRoute routes to an InferencePool backend, and the pool's endpoint picker handles the inference-specific routing logic:

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: inference-api
spec:
  parentRefs:
  - name: inference-gateway
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /v1/chat/completions
    backendRefs:
    - group: inference.networking.k8s.io
      kind: InferencePool
      name: large-models
```

This design means existing Gateway API implementations (Envoy Gateway, Istio, NGINX Gateway Fabric) can support inference routing by implementing the InferencePool backend type, rather than requiring a completely separate infrastructure stack. The community chose composition over reinvention.

### Endpoint Picker Architecture

The endpoint picker is the most innovative component. Rather than hardcoding a single load-balancing algorithm, the extension defines a gRPC interface that custom pickers implement. This allows organizations to build routing logic specific to their inference backends:

- **vLLM-aware picker:** Queries each backend's `/metrics` endpoint for `vllm:num_requests_running` and `vllm:gpu_cache_usage_perc`, routing to the backend with most available KV cache capacity.
- **Prefix-cache-aware picker:** Maintains a prefix tree of cached prompts across backends, routing requests to the backend most likely to have a cache hit for the request's system prompt.
- **Cost-aware picker:** Routes to spot-instance backends first, falling back to on-demand instances only when spot capacity is exhausted.

The picker runs as a sidecar or standalone deployment, not inline in the data path. The gateway queries it for routing decisions, keeping the picker's latency out of the critical request path when decisions can be cached.

---

## Envoy AI Gateway: Token-Level Traffic Management

### Beyond HTTP-Level Proxying

Envoy AI Gateway (part of the Envoy Proxy ecosystem, backed by Tetrate and the broader Envoy community) extends Envoy's already-powerful L7 proxy capabilities with AI-specific features. While the Gateway API Inference Extension defines the Kubernetes-native control plane, Envoy AI Gateway provides the data plane implementation with features that go beyond what standard HTTP proxying can achieve.

The key insight driving Envoy AI Gateway is that AI traffic has fundamentally different characteristics from web traffic. A single HTTP request to an LLM can generate a streaming response lasting 30+ seconds, consuming GPU resources proportional to the output token count rather than the request count. Traditional rate limiting (requests per second) and load balancing (round-robin, least-connections) produce poor outcomes because they ignore the variable cost of each request.

### Token-Aware Rate Limiting

Envoy AI Gateway introduces rate limiting based on token consumption rather than request count. This requires parsing the LLM response stream (OpenAI-compatible SSE format) and counting tokens in real-time:

```yaml
apiVersion: aigateway.envoyproxy.io/v1alpha1
kind: AIGatewayRoute
metadata:
  name: production-api
spec:
  rules:
  - backends:
    - name: vllm-cluster
  rateLimiting:
    tokenBucket:
      inputTokensPerMinute: 100000    # Limit based on actual token consumption
      outputTokensPerMinute: 50000     # Output tokens are 10x more expensive
      requestsPerMinute: 1000          # Also cap request count as safety valve
    perClient: true                    # Per API key, not global
    spillover:
      action: queue                    # Queue excess rather than reject
      maxQueueTime: 30s
```

This token-based accounting means a client sending many short requests and a client sending few long requests are treated fairly based on actual resource consumption, not request count. The gateway maintains running token counts per client identity and enforces limits in real-time as the streaming response is proxied.

### Provider Failover and Model Fallback

For organizations using multiple inference backends (self-hosted vLLM, NVIDIA NIM, cloud providers as overflow), Envoy AI Gateway provides intelligent failover with model compatibility awareness:

```yaml
apiVersion: aigateway.envoyproxy.io/v1alpha1
kind: AIGatewayRoute
metadata:
  name: multi-provider
spec:
  rules:
  - match:
      model: "gpt-4-equivalent"
    backends:
    - name: self-hosted-llama-70b
      priority: 1
      weight: 90
    - name: azure-openai-gpt4
      priority: 2                    # Failover target
      weight: 10                     # Also gets 10% of normal traffic (warm standby)
    failover:
      triggers:
      - type: latency
        threshold: 5s                # Failover if p99 exceeds 5s
      - type: errorRate
        threshold: 0.05              # Failover if >5% errors
      - type: queueDepth
        threshold: 100               # Failover if >100 requests queued
```

The failover logic is not simple binary switching. It implements graduated traffic shifting: as the primary backend's latency increases, the gateway progressively shifts more traffic to the secondary, avoiding thundering herd problems that binary failover creates.

### KV-Cache-Aware Endpoint Picking

The most sophisticated feature for inference-specific routing is KV-cache-aware endpoint selection. When integrated with inference engines that expose cache metadata (vLLM's `/v1/cache/status` endpoint, SGLang's RadixAttention tree), the gateway can route multi-turn conversations to the same backend that holds the previous turn's KV cache:

```yaml
spec:
  endpointPicker:
    type: kv-cache-aware
    config:
      cacheMetadataEndpoint: /v1/cache/status
      affinityKey: "x-session-id"       # HTTP header identifying the conversation
      affinityTimeout: 300s             # Cache affinity expires after 5 minutes idle
      fallbackStrategy: least-tokens    # When affinity target is overloaded
```

When a request arrives with a session ID that maps to a backend holding its KV cache, the gateway routes directly to that backend. If the backend is overloaded (queue depth exceeds threshold), the gateway falls back to the least-tokens-in-flight strategy, accepting the cache miss penalty rather than increasing latency by queuing.

The latency savings from KV cache hits are substantial. For a multi-turn conversation with 8,000 tokens of context, a cache hit eliminates the prefill phase entirely (saving 2-4 seconds on large models), making the difference between acceptable and unacceptable user experience.

---

## llm-d on Kubernetes: Disaggregated Inference as a Kubernetes-Native Architecture

### The Disaggregation Thesis

Module 07.4 (Disaggregated Serving) covered the theoretical case for separating prefill and decode phases. The key insight: prefill is compute-bound (processing all input tokens in parallel) while decode is memory-bandwidth-bound (generating one token at a time, reading the full KV cache each step). Running both phases on the same GPU means neither is optimally utilized, because the hardware cannot simultaneously maximize compute throughput and memory bandwidth utilization.

llm-d (open-sourced by Red Hat and IBM in early 2026) takes this theoretical architecture and makes it deployable on standard Kubernetes clusters. Rather than requiring a custom orchestration layer, llm-d maps disaggregated inference directly onto Kubernetes primitives, making it operable by platform teams who already know Kubernetes.

### Architecture: Prefill and Decode as Separate Workloads

llm-d deploys prefill and decode as distinct Kubernetes workload types with different resource profiles and scaling characteristics:

```yaml
# Prefill workers: compute-optimized, stateless, horizontally scalable
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-70b-prefill
spec:
  replicas: 4
  template:
    spec:
      containers:
      - name: prefill-worker
        image: llm-d/prefill:latest
        resources:
          requests:
            nvidia.com/gpu: 4        # Needs high compute (TP=4 for large context)
        env:
        - name: LLM_D_ROLE
          value: "prefill"
        - name: LLM_D_KV_TRANSFER_PROTOCOL
          value: "rdma"              # RDMA for fast KV transfer to decode pods
---
# Decode workers: memory-optimized, stateful, need persistent KV cache
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: llama-70b-decode
spec:
  replicas: 8
  template:
    spec:
      containers:
      - name: decode-worker
        image: llm-d/decode:latest
        resources:
          requests:
            nvidia.com/gpu: 2        # Fewer GPUs but more memory per GPU
        env:
        - name: LLM_D_ROLE
          value: "decode"
        - name: LLM_D_KV_CACHE_CAPACITY
          value: "65536"             # Max concurrent tokens in KV cache
        volumeMounts:
        - name: kv-cache-backing     # Optional: SSD-backed overflow for KV cache
          mountPath: /kv-overflow
  volumeClaimTemplates:
  - metadata:
      name: kv-cache-backing
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
      storageClassName: local-nvme   # Fast local NVMe for KV overflow
```

The critical design choice is using a StatefulSet for decode workers. Decode pods maintain KV cache state that should survive pod restarts (within reason). The StatefulSet provides stable network identities (`llama-70b-decode-0`, `llama-70b-decode-1`, etc.) that the router uses for session affinity, and optional persistent volumes for KV cache overflow to NVMe when GPU memory is exhausted.

Prefill workers are a standard Deployment because they are stateless: they process input tokens, generate the KV cache, transfer it to a decode worker, and are immediately available for the next request. Horizontal pod autoscaling based on queue depth works naturally.

### KV Cache Transfer: The Critical Data Path

The performance of disaggregated inference depends entirely on how fast KV cache transfers from prefill to decode workers. For a Llama 70B model with 8,000 token context, the KV cache is approximately 5 GB. At 10 Gbps network speed, this transfer takes 4 seconds, which is unacceptable. llm-d supports three transfer modes:

| Transfer Mode | Bandwidth | Latency (5GB) | Requirement |
|---|---|---|---|
| TCP/gRPC | 10-25 Gbps | 1.6-4s | Standard K8s networking |
| RDMA/RoCEv2 | 100-400 Gbps | 0.1-0.4s | RDMA-capable NICs + network config |
| GPU Direct RDMA | 400-900 Gbps | 0.04-0.1s | NVLink/InfiniBand fabric |

For production deployments, RDMA is the minimum viable option. llm-d configures RDMA networking through Kubernetes Network Attachment Definitions (Multus CNI), attaching a secondary high-speed network interface to pods specifically for KV cache transfer:

```yaml
apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: rdma-net
spec:
  config: |
    {
      "cniVersion": "0.3.1",
      "type": "host-device",
      "device": "mlx5_0",
      "ipam": { "type": "whereabouts", "range": "10.0.100.0/24" }
    }
```

### The Router Component

llm-d includes a lightweight router that sits between the client and the prefill/decode workers. The router's responsibilities:

1. **Accept client requests** via standard OpenAI-compatible API
2. **Select a prefill worker** based on current queue depth and GPU utilization
3. **Select a decode worker** based on available KV cache capacity and session affinity
4. **Coordinate the handoff:** prefill worker processes input, transfers KV cache to the selected decode worker, decode worker begins token generation
5. **Stream tokens** back to the client from the decode worker

The router itself is stateless and horizontally scalable. It makes decisions based on real-time metrics scraped from both prefill and decode workers via a lightweight metrics protocol.

### Scaling Characteristics

The beauty of disaggregated inference on Kubernetes is that prefill and decode scale independently based on different signals:

- **Prefill scales on input queue depth.** More incoming requests with long prompts? Add prefill replicas. HPA with custom metrics works naturally.
- **Decode scales on KV cache pressure.** When decode workers approach memory capacity (active generations filling available KV cache), add decode replicas. This is a memory-pressure signal, not a compute signal.

This independent scaling means you are not forced to add expensive full-stack replicas (each with both prefill and decode capacity) when only one phase is the bottleneck. In practice, most production workloads are decode-bottlenecked (many concurrent long-running generations), so you end up with more decode pods than prefill pods.

---

## NVIDIA Grove: GPU Cluster Orchestration for AI Workloads

### Beyond Single-Node Scheduling

KAI Scheduler handles GPU-aware scheduling within a Kubernetes cluster. But large-scale inference deployments face challenges at a higher level: orchestrating multiple models across a fleet of GPU nodes, managing model placement for optimal utilization, and handling the lifecycle of inference services as a fleet operation rather than individual deployments.

NVIDIA Grove (open-sourced in Q1 2026) provides a Kubernetes-native API for fleet-level AI workload orchestration. It operates above the scheduler, making decisions about which models should run where, how many replicas each model needs, and when to scale or relocate models based on demand patterns.

### Core Concepts

**ModelDeployment:** A declaration of intent: "I want this model available with these SLOs." Grove handles the mechanics of making it happen.

```yaml
apiVersion: grove.nvidia.com/v1
kind: ModelDeployment
metadata:
  name: llama-70b-chat
spec:
  model:
    source: huggingface
    name: meta-llama/Llama-3.1-70B-Instruct
    quantization: fp8                    # Grove handles quantization at deploy time
  serving:
    engine: vllm
    tensorParallelism: 4
    maxConcurrentRequests: 256
  scaling:
    minReplicas: 2
    maxReplicas: 16
    targetLatencyP99: 2000ms            # Scale to maintain this SLO
    scaleDownDelay: 600s                 # Cooldown before removing replicas
  placement:
    preferredGPU: h100-sxm
    fallbackGPU: a100-80gb              # Acceptable but not optimal
    antiAffinity:
    - modelDeployment: llama-70b-chat    # Spread replicas across failure domains
      topologyKey: topology.kubernetes.io/zone
```

**GPUCluster:** A representation of the available GPU fleet with topology and capability information that Grove uses for intelligent placement decisions.

```yaml
apiVersion: grove.nvidia.com/v1
kind: GPUCluster
metadata:
  name: us-east-1-inference
spec:
  nodes:
  - selector:
      matchLabels:
        gpu-class: h100-sxm-8
    gpusPerNode: 8
    interconnect: nvlink
    count: 32
  - selector:
      matchLabels:
        gpu-class: a100-80gb-8
    gpusPerNode: 8
    interconnect: pcie-gen4
    count: 64
  costModel:
    h100-sxm: 3.5               # Relative cost units per GPU-hour
    a100-80gb: 2.0
```

### Fleet-Level Optimization

Grove continuously optimizes model placement across the fleet based on:

- **Demand prediction:** Historical request patterns predict which models need more capacity at which times. Grove pre-scales before demand arrives rather than reacting to SLO violations.
- **Cost minimization:** When multiple GPU types can serve a model, Grove places on the cheapest option that still meets the latency SLO. Only during traffic spikes does it overflow to premium hardware.
- **Bin packing with SLO awareness:** Multiple small models can share a node, but Grove ensures that co-located models do not interfere with each other's latency SLOs through resource isolation and GPU memory partitioning.
- **Graceful model migration:** When a model needs to move (node maintenance, rebalancing), Grove performs a warm migration: spin up the new replica, transfer KV cache if applicable, shift traffic, then terminate the old replica. Zero downtime for model relocations.

### Integration with KAI and DRA

Grove sits above KAI Scheduler and DRA in the stack:
- Grove decides WHAT models should run WHERE (fleet orchestration)
- KAI Scheduler places individual pods with topology awareness (node scheduling)
- DRA provides structured GPU claims that both Grove and KAI use for hardware matching

This layered architecture means each component handles its level of abstraction without reimplementing the others.

---

## Microsoft AI Runway: Inference Orchestration with Hugging Face Integration

### The Problem AI Runway Solves

Platform teams running Kubernetes clusters for ML teams face a recurring challenge: data scientists want to deploy a model by name ("deploy Phi-3-medium on our cluster"), but the actual deployment requires answering dozens of infrastructure questions. Which GPU type fits this model? How much memory does it need? What's the optimal tensor parallelism degree? What serving engine should we use? What quantization gives acceptable quality?

Microsoft AI Runway (contributed to the CNCF ecosystem in 2026) provides a Kubernetes API that handles these questions automatically through integration with Hugging Face model metadata and GPU hardware specifications.

### Automatic Hardware Fitting

When you create an AIModel resource, AI Runway calculates the GPU memory requirements from model metadata (parameter count, architecture, precision) and matches against available cluster hardware:

```yaml
apiVersion: airunway.microsoft.com/v1alpha1
kind: AIModel
metadata:
  name: phi-3-medium
spec:
  source:
    huggingFace:
      modelId: microsoft/Phi-3-medium-128k-instruct
      revision: main
  inference:
    targetLatencyMs: 500
    maxBatchSize: 32
  resources:
    budget:
      maxGPUs: 4
      preferCostOptimal: true     # Choose cheapest config that meets latency target
```

AI Runway resolves this declaration by:
1. Querying Hugging Face for model config (14B parameters, 40 layers, 5120 hidden dim)
2. Computing memory requirements: ~28 GB in FP16, ~14 GB in INT8, ~7 GB in INT4
3. Evaluating available GPU types against memory + compute requirements
4. Selecting optimal configuration: 1x A100-80GB at FP16, or 1x L4-24GB at INT4
5. Generating the actual Kubernetes Deployment, Service, and HPA resources

### GPU Memory Fit Indicators

AI Runway exposes a "fit indicator" as a status condition, giving platform teams clear visibility into whether their cluster can serve a model before attempting deployment:

```yaml
status:
  conditions:
  - type: GPUFit
    status: "True"
    message: "Model fits on 2x A100-80GB with TP=2 (56GB model + 24GB KV cache headroom)"
  - type: CostEstimate
    status: "True"
    message: "Estimated $2.80/hour at current spot pricing"
  - type: LatencyEstimate
    status: "True"
    message: "Projected P99 latency: 380ms (within 500ms target)"
  selectedConfiguration:
    gpuType: a100-80gb
    gpuCount: 2
    quantization: fp16
    tensorParallelism: 2
    estimatedThroughput: "45 requests/sec at batch_size=16"
```

This pre-deployment analysis prevents the common failure mode where a team deploys a model, waits 10 minutes for weights to download, only to discover the GPU runs out of memory during the first request because KV cache allocation was not accounted for.

### Cost Estimation and Chargeback

AI Runway integrates with cloud provider pricing APIs to provide real-time cost estimates for inference deployments:

- Per-model cost attribution based on actual GPU utilization
- Comparison between deployment options (fewer expensive GPUs vs. more cheap GPUs with quantization)
- Projected monthly cost at current request volume
- Alert thresholds when cost exceeds budget

This cost visibility is critical for organizations where ML teams deploy models but platform teams pay the infrastructure bill. AI Runway makes the cost of each model deployment visible to the team requesting it, enabling informed decisions about model selection and optimization.

---

## Decision Framework: Choosing the Right K8s Inference Stack

The proliferation of Kubernetes-native inference tools creates a genuine selection problem. This framework maps workload characteristics to the appropriate tool combination.

### Dimension 1: Model Scale and Complexity

| Model Size | Recommended Stack | Reasoning |
|---|---|---|
| Small (< 13B) | KServe + KAI fractional GPU | Models fit on fractional GPUs, standard serving is sufficient |
| Medium (13B-70B) | vLLM/SGLang + Gateway Inference Extension | Need tensor parallelism, model-aware routing, but single-node |
| Large (70B-405B) | llm-d + KAI topology-aware scheduling | Multi-node TP requires disaggregation for efficiency |
| Mixture-of-Experts | NVIDIA NIM + Grove fleet management | Complex routing between expert groups benefits from NIM optimization |

### Dimension 2: Operational Maturity

| Team Capability | Recommended Approach | Why |
|---|---|---|
| "We just want to deploy models" | AI Runway + KServe | Minimal config, auto-fitting, standard interfaces |
| "We have a platform team" | Gateway Inference Extension + KAI | Custom routing logic, fair-share scheduling, full control |
| "We run large-scale inference" | llm-d + Grove + Envoy AI Gateway | Disaggregated serving, fleet orchestration, token-level traffic management |

### Dimension 3: Latency Requirements

| Latency Target | Architecture | Key Trade-off |
|---|---|---|
| < 100ms TTFT | Prefill-optimized (chunked prefill, prefix cache) | Dedicate GPU compute to prefill, accept lower decode throughput |
| < 500ms TTFT | Standard consolidated serving | Simpler operations, good enough for most applications |
| < 2s TTFT | Disaggregated with queue-based routing | Maximize throughput, accept queuing delay for cost efficiency |
| Best-effort | Spot instances + aggressive batching | Minimize cost, latency is secondary |

### Dimension 4: Multi-tenancy Requirements

| Isolation Need | Solution | What It Provides |
|---|---|---|
| Hard isolation (compliance) | Separate namespaces + NetworkPolicy + KAI queues | Full resource and network isolation per tenant |
| Soft isolation (fairness) | KAI queue-based quotas + Envoy token rate limiting | Fair sharing with borrowing, no hard boundaries |
| No isolation (single team) | Standard Deployment + HPA | Simplest possible setup, no overhead |

### Composability Matrix

These tools are designed to compose, not compete. A production stack typically includes multiple layers:

```
┌─────────────────────────────────────────────────────────────┐
│  Envoy AI Gateway (token rate limiting, failover)           │  ← L7 Traffic Management
├─────────────────────────────────────────────────────────────┤
│  Gateway API Inference Extension (model routing, pools)     │  ← K8s-native Routing
├─────────────────────────────────────────────────────────────┤
│  Grove / AI Runway (fleet orchestration, auto-fitting)      │  ← Workload Orchestration  
├─────────────────────────────────────────────────────────────┤
│  llm-d / vLLM / NIM (inference engine)                      │  ← Serving Engine
├─────────────────────────────────────────────────────────────┤
│  KAI Scheduler (topology-aware placement, fractional GPU)   │  ← GPU Scheduling
├─────────────────────────────────────────────────────────────┤
│  DRA (structured GPU claims, multi-node)                    │  ← Resource Allocation
├─────────────────────────────────────────────────────────────┤
│  Kubernetes (pods, services, networking)                     │  ← Container Orchestration
└─────────────────────────────────────────────────────────────┘
```

Not every deployment needs every layer. Start with the bottom layers (DRA + KAI for better GPU utilization), add routing (Gateway Inference Extension) when you have multiple models, and add fleet orchestration (Grove) when managing tens or hundreds of model deployments across a large cluster.

---

## Mental Model: Where We Are in the Infrastructure Evolution

Kubernetes for LLM inference in 2026 is where Kubernetes for microservices was in 2018. The parallels are striking:

**2018 microservices:** Kubernetes provided pods and services, but you needed Istio for traffic management, Prometheus for observability, cert-manager for TLS, and external-dns for discovery. These were all separate projects solving problems that Kubernetes itself did not address. By 2022, many of these patterns were absorbed into Kubernetes itself (Gateway API, built-in topology-aware routing, improved service mesh integration).

**2026 inference:** Kubernetes provides pods and services, but you need KAI for GPU scheduling, DRA for structured resource claims, Gateway Inference Extension for model routing, and Envoy AI Gateway for token management. These are separate projects solving problems that vanilla Kubernetes does not address. By 2028-2029, we should expect many of these patterns to be absorbed into Kubernetes core or become standard CRDs that every distribution supports.

The implication for practitioners is clear: invest in learning these new primitives now. They will become as fundamental to AI infrastructure as Ingress and HPA are to web infrastructure today. The teams that build competency with KAI Scheduler, DRA claims, and inference-aware routing will have a significant operational advantage as LLM inference becomes a standard workload type rather than a novel one.

**Key principle:** The goal is not to make Kubernetes "work" for inference through heroic workarounds (node affinity hacks, custom admission webhooks, hand-tuned resource limits). The goal is to use the new purpose-built primitives that make Kubernetes genuinely good at inference orchestration. The tools described in this module exist precisely because the workaround era is ending.

---

## Summary

| Component | Owner | Status (mid-2026) | Primary Function |
|---|---|---|---|
| KAI Scheduler | NVIDIA (CNCF Sandbox) | Production-ready | Fractional GPU, topology-aware scheduling, queue fairness |
| DRA | Kubernetes SIG Node | GA (K8s 1.32) | Structured GPU claims, multi-node resources |
| Gateway API Inference Extension | Kubernetes SIG Network | v0.3 alpha | Model-aware routing, endpoint pickers |
| Envoy AI Gateway | Envoy community | Beta | Token rate limiting, KV-cache routing, failover |
| llm-d | Red Hat + IBM | Early production | Disaggregated prefill/decode on K8s |
| NVIDIA Grove | NVIDIA | GA | Fleet-level model orchestration |
| Microsoft AI Runway | Microsoft | Alpha | Auto-fitting, cost estimation, HF integration |

The Kubernetes ecosystem for inference is moving fast. New projects appear quarterly, existing ones merge or get absorbed. The architectural patterns, however, are stable: structured resource claims, model-aware routing, disaggregated serving, and fleet orchestration. Learn the patterns, and the specific tools become implementation details.

---

## Further Reading

- KAI Scheduler: NVIDIA/KAI-Scheduler on GitHub (CNCF Sandbox project page)
- DRA specification: Kubernetes Enhancement Proposal KEP-4381
- Gateway API Inference Extension: kubernetes-sigs/gateway-api-inference-extension
- Envoy AI Gateway: envoyproxy/ai-gateway
- llm-d architecture: llm-d/llm-d on GitHub
- NVIDIA Grove: NVIDIA/grove on GitHub
- KubeCon EU 2026 AI Day keynotes on inference infrastructure evolution
