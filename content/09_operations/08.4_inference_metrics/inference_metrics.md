# Inference Metrics, Goodput, and Production Monitoring

## Why Standard Metrics Fail LLM Inference

Every distributed system ships with metrics: latency, throughput, error rate. These three numbers have guided capacity planning for decades. But LLM inference breaks the assumptions behind all of them.

A REST API that returns JSON has a single latency distribution. The request arrives, computation happens, the response leaves. One number captures the user experience. LLM inference generates tokens one at a time over seconds or minutes. The user sees the first token (time-to-first-token), then experiences the streaming rate (inter-token latency), then waits for completion. A single "latency" number conflates three distinct experiences into one meaningless average.

Throughput is equally misleading. A system processing 1000 tokens/second sounds impressive until you learn that 800 of those tokens belong to requests that will eventually timeout and be discarded. The system is busy, but not productive. Standard throughput counts all work equally, whether it satisfies user expectations or not.

From Module 0.2, we established that decode is memory-bound while prefill is compute-bound. A single latency number hides two different bottlenecks operating simultaneously. A system might have excellent prefill performance (low TTFT) but terrible decode throughput (high inter-token latency), or vice versa. Without separating these phases in your metrics, you cannot diagnose which resource is the bottleneck.

This chapter builds the metrics hierarchy you need to actually understand what your inference system is doing, introduces goodput as the production metric that matters, and shows you how to instrument, monitor, and alert on a real deployment.

---

## The Core Metrics Hierarchy

LLM inference metrics form a natural hierarchy. At the top are user-facing metrics that directly impact experience. Below them are system metrics that explain why user-facing metrics behave as they do. At the bottom are hardware metrics that constrain everything above.

### Time-to-First-Token (TTFT)

TTFT measures the elapsed time from when a request arrives at the inference server to when the first generated token is returned to the client. This metric captures the combined cost of:

1. **Queue wait time**: How long the request sits in the scheduler queue waiting for a slot
2. **Prefill computation**: Processing all input tokens through the model to build the KV cache
3. **First decode step**: Generating the actual first output token

For interactive applications (chatbots, code assistants, search), TTFT dominates perceived responsiveness. Research on user experience shows that delays beyond 200-500ms feel "slow" to users, regardless of how fast subsequent tokens arrive. A system with 2-second TTFT but 20ms inter-token latency feels sluggish, while a system with 200ms TTFT and 50ms inter-token latency feels snappy.

TTFT scales with input sequence length because prefill is O(n²) in attention computation (or O(n) with flash attention, but still linear in sequence length for the KV cache build). A 128-token prompt might have 50ms TTFT while a 4096-token prompt might have 400ms TTFT on the same hardware, purely from prefill cost.

### Time Per Output Token (TPOT) / Inter-Token Latency (ITL)

TPOT measures the average time between consecutive generated tokens after the first token. ITL is the per-step measurement. The distinction matters: TPOT is the mean across all decode steps, while ITL captures the distribution, including the variance that users perceive as "stuttering."

During continuous batching, ITL can spike when new requests join the batch (their prefill computation delays the decode step for existing requests). This creates a sawtooth pattern: steady ITL punctuated by spikes when the scheduler inserts new prefills. Understanding this pattern is critical for SLO design, because your p99 ITL will be dominated by these interference events rather than by steady-state decode performance.

For streaming applications, human reading speed provides a natural SLO boundary. Most people read at 200-300 words per minute, or roughly 4-5 words per second. Since the average token is approximately 0.75 words, a streaming rate of 6-7 tokens/second (140-170ms ITL) matches comfortable reading speed. Anything faster than this provides no perceptible benefit for chat-style interfaces.

### End-to-End Latency

End-to-end latency measures total time from request arrival to final token delivery. It equals TTFT + (number_of_generated_tokens × TPOT). This metric matters for batch/offline workloads where the user waits for the complete response (summarization, translation, code generation with compilation).

For streaming applications, end-to-end latency matters less than TTFT and ITL because the user is consuming tokens as they arrive. But for API-style integrations where downstream systems wait for the full response, end-to-end latency determines your timeout budget and retry strategy.

### Throughput: Tokens Per Second

System throughput measures total tokens generated per second across all concurrent requests. This is your capacity metric: how much work can the system do? But throughput alone is dangerous because it does not distinguish between useful and wasted work.

A system generating 5000 tokens/second might be:
- Serving 50 concurrent users at 100 tokens/sec each (healthy)
- Serving 500 concurrent users at 10 tokens/sec each (everyone is slow)
- Generating 4000 tokens for requests that will timeout (wasteful)

Throughput must always be reported alongside per-request latency metrics to be interpretable.

### Normalized Metrics for Fair Comparison

Raw TTFT and throughput numbers are meaningless without normalization. Three normalized metrics enable apples-to-apples comparison:

**Normalized Latency** = End-to-end latency / output tokens generated. This lets you compare a 10-token response (fast) against a 500-token response (slow) fairly. Without normalization, the 500-token response always looks worse even if it's running at higher efficiency.

**Model FLOPs Utilization (MFU)** = Achieved FLOPS / Theoretical Peak FLOPS. From Module 01.2's roofline model, you know peak A100 FP16 is 312 TFLOPS. If your serving system achieves 50 TFLOPS sustained, MFU = 16%. This explains SM utilization: low MFU during decode is expected (memory-bound), high MFU during prefill means your batching is working.

**Model Bandwidth Utilization (MBU)** = Achieved bandwidth / Peak HBM bandwidth. For decode, MBU is the metric that matters. A100 peak is 2 TB/s. If you're sustaining 1.8 TB/s during decode, MBU = 90% -- you're near the hardware limit.

---

## Percentiles Matter: Why Averages Lie

Average latency is the most dangerous metric in production systems. It hides the experience of your worst-served users behind the comfortable majority.

Consider a system with these TTFT measurements across 1000 requests:
- 950 requests: 100-200ms (fast, healthy)
- 40 requests: 500-800ms (noticeable delay)
- 10 requests: 3000-5000ms (terrible experience)

The average TTFT is ~230ms. This looks excellent. But 1% of your users (p99) are waiting 3-5 seconds for the first token. If you serve 10 million requests per day, that is 100,000 users per day having a terrible experience.

### The Percentile Hierarchy

| Percentile | What It Tells You | When It Matters |
|---|---|---|
| p50 (median) | Typical user experience | Baseline performance monitoring |
| p75 | Where degradation starts showing | Early warning indicator |
| p90 | Impact during load spikes | Capacity planning threshold |
| p95 | Tail latency under normal load | SLO boundary for most services |
| p99 | Worst 1% experience | Critical for high-traffic services |
| p99.9 | Extreme outliers | Debugging specific failure modes |

For LLM inference specifically, the gap between p50 and p99 is often 10-50x (compared to 2-5x for typical web services). This happens because:

1. **Input length variance**: A 32-token prompt vs a 4096-token prompt produces wildly different TTFT
2. **Batch interference**: Requests that arrive during a large prefill get delayed
3. **KV cache pressure**: When KV cache is near capacity, the scheduler preempts requests, adding queue time
4. **Output length variance**: A 10-token response vs a 2048-token response creates massive end-to-end latency spread

### Histogram Buckets for LLM Metrics

Standard Prometheus histogram buckets (5ms, 10ms, 25ms, 50ms, 100ms...) are wrong for LLM inference. You need buckets that match the actual distribution:

```python
# TTFT buckets (milliseconds): bimodal distribution expected
TTFT_BUCKETS = [50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000]

# ITL buckets (milliseconds): tight cluster with occasional spikes  
ITL_BUCKETS = [10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 1000]

# End-to-end buckets (seconds): wide range due to output length variance
E2E_BUCKETS = [0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30, 45, 60, 90, 120]
```

---

## Goodput: The Production Metric That Matters

### The Problem with Raw Throughput

A serving system reporting 10,000 tokens/second at 80% GPU utilization looks healthy by every traditional metric. But examine the requests more closely:

- 30% of tokens belong to requests that exceeded TTFT SLO (user already saw an error)
- 15% of tokens belong to requests that will be cancelled before completion (user navigated away)
- 10% of tokens are being regenerated after preemption (wasted compute)

Actual useful output: 4,500 tokens/second. The system is 55% efficient despite looking 100% busy.

### Defining Goodput

Goodput, as formalized by the Anyscale team, is the rate of output tokens that satisfy all SLO requirements. The formal definition:

```
Goodput = (tokens generated for requests meeting ALL SLOs) / time
```

A request "meets SLOs" only if:
1. TTFT is within the TTFT SLO bound
2. Every ITL measurement is within the ITL SLO bound (or the mean ITL is within bound)
3. The request completes without preemption/timeout
4. The output is delivered to the client successfully

This is a strict definition. A request with perfect TTFT but one ITL spike above the SLO threshold contributes zero tokens to goodput. This strictness is intentional: it measures what the user actually experiences as acceptable service.

### Why Goodput Changes Everything

Traditional capacity planning asks: "How many tokens/second can this GPU produce?" This leads to overloading systems because raw throughput increases with batch size even as per-request latency degrades.

Goodput-driven capacity planning asks: "How many tokens/second can this GPU produce while keeping all requests within SLO?" This question has a fundamentally different answer. There exists an optimal batch size beyond which adding more requests increases raw throughput but decreases goodput (because the additional requests cause existing requests to miss their SLOs).

```python
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RequestMetrics:
    """Metrics for a single completed request."""
    request_id: str
    ttft_ms: float
    itl_values_ms: List[float]
    total_tokens_generated: int
    completed: bool
    preempted: bool


@dataclass
class SLOConfig:
    """Service Level Objective configuration."""
    ttft_ms: float        # Max acceptable TTFT
    itl_p99_ms: float     # Max acceptable p99 ITL
    itl_mean_ms: float    # Max acceptable mean ITL
    completion_required: bool  # Must complete without preemption


def compute_goodput(
    requests: List[RequestMetrics],
    slo: SLOConfig,
    window_seconds: float
) -> dict:
    """
    Compute goodput and related metrics over a time window.
    
    Returns dict with:
      - goodput_tokens_per_sec: tokens/sec from SLO-meeting requests
      - raw_throughput_tokens_per_sec: total tokens/sec regardless of SLO
      - goodput_ratio: goodput / raw_throughput
      - slo_attainment: fraction of requests meeting all SLOs
      - violation_breakdown: which SLO is most commonly violated
    """
    total_tokens = 0
    good_tokens = 0
    violations = {"ttft": 0, "itl_p99": 0, "itl_mean": 0, "completion": 0}
    
    for req in requests:
        total_tokens += req.total_tokens_generated
        
        # Check each SLO condition
        ttft_ok = req.ttft_ms <= slo.ttft_ms
        
        if req.itl_values_ms:
            itl_p99 = np.percentile(req.itl_values_ms, 99)
            itl_mean = np.mean(req.itl_values_ms)
        else:
            itl_p99 = 0
            itl_mean = 0
            
        itl_p99_ok = itl_p99 <= slo.itl_p99_ms
        itl_mean_ok = itl_mean <= slo.itl_mean_ms
        completion_ok = req.completed and not req.preempted if slo.completion_required else True
        
        # Track violations
        if not ttft_ok:
            violations["ttft"] += 1
        if not itl_p99_ok:
            violations["itl_p99"] += 1
        if not itl_mean_ok:
            violations["itl_mean"] += 1
        if not completion_ok:
            violations["completion"] += 1
        
        # Request contributes to goodput only if ALL SLOs met
        if ttft_ok and itl_p99_ok and itl_mean_ok and completion_ok:
            good_tokens += req.total_tokens_generated
    
    raw_throughput = total_tokens / window_seconds
    goodput = good_tokens / window_seconds
    
    return {
        "goodput_tokens_per_sec": goodput,
        "raw_throughput_tokens_per_sec": raw_throughput,
        "goodput_ratio": goodput / raw_throughput if raw_throughput > 0 else 0,
        "slo_attainment": good_tokens / total_tokens if total_tokens > 0 else 0,
        "violation_breakdown": violations,
        "total_requests": len(requests),
        "good_requests": sum(
            1 for r in requests
            if r.ttft_ms <= slo.ttft_ms
            and (np.percentile(r.itl_values_ms, 99) <= slo.itl_p99_ms if r.itl_values_ms else True)
            and (np.mean(r.itl_values_ms) <= slo.itl_mean_ms if r.itl_values_ms else True)
            and (r.completed and not r.preempted if slo.completion_required else True)
        )
    }


# Example: Measure goodput for a deployment
slo = SLOConfig(
    ttft_ms=500,
    itl_p99_ms=150,
    itl_mean_ms=80,
    completion_required=True
)

# Simulated request data (in production, pull from your metrics store)
np.random.seed(42)
requests = []
for i in range(1000):
    # Simulate bimodal TTFT (fast prefills + occasional queue delays)
    ttft = np.random.choice(
        [np.random.normal(150, 50), np.random.normal(800, 200)],
        p=[0.9, 0.1]
    )
    ttft = max(20, ttft)
    
    num_tokens = int(np.random.lognormal(4, 0.8))
    num_tokens = min(max(num_tokens, 5), 2048)
    
    # ITL values: mostly steady with occasional spikes from batch interference
    itl_base = np.random.normal(45, 10, size=num_tokens)
    # 5% of steps get interference spikes
    spike_mask = np.random.random(num_tokens) < 0.05
    itl_base[spike_mask] *= np.random.uniform(2, 5, size=spike_mask.sum())
    itl_values = np.clip(itl_base, 10, 500).tolist()
    
    requests.append(RequestMetrics(
        request_id=f"req-{i}",
        ttft_ms=ttft,
        itl_values_ms=itl_values,
        total_tokens_generated=num_tokens,
        completed=np.random.random() > 0.03,  # 3% don't complete
        preempted=np.random.random() < 0.05    # 5% preempted
    ))

result = compute_goodput(requests, slo, window_seconds=60.0)
print(f"Raw throughput: {result['raw_throughput_tokens_per_sec']:.0f} tok/s")
print(f"Goodput:        {result['goodput_tokens_per_sec']:.0f} tok/s")
print(f"Goodput ratio:  {result['goodput_ratio']:.1%}")
print(f"SLO attainment: {result['slo_attainment']:.1%}")
print(f"Violations:     {result['violation_breakdown']}")
```

The goodput ratio is the single most important number for a production deployment. A ratio below 0.7 means you are wasting more than 30% of your GPU compute on work that does not satisfy users.

---

## SLO-Driven Metrics Design

### Define the Contract First

Before instrumenting anything, define what "good" means for your application. Different use cases demand radically different SLOs:

| Application | TTFT SLO | ITL SLO | Completion SLO |
|---|---|---|---|
| Interactive chat | < 300ms p95 | < 80ms p95 | Not required (streaming) |
| Code completion | < 200ms p99 | < 50ms p95 | Full suggestion required |
| Batch summarization | < 5000ms p95 | < 200ms p95 | Must complete |
| RAG pipeline (internal) | < 1000ms p95 | < 100ms p95 | Must complete |
| Voice assistant | < 150ms p99 | < 40ms p99 | First sentence required |

Notice the SLO percentile varies by application. Interactive chat can tolerate occasional slow responses (p95) because users retry naturally. Code completion must be consistently fast (p99) because even rare delays break flow. Voice assistants need p99 because any stutter is audible.

### SLO Compliance Tracking

```python
from dataclasses import dataclass, field
from collections import deque
import time
from typing import Optional


@dataclass
class SLOTracker:
    """
    Tracks rolling SLO compliance over a configurable window.
    
    This is what your monitoring system should compute continuously.
    Alert when compliance drops below threshold.
    """
    ttft_slo_ms: float
    itl_slo_ms: float
    window_seconds: int = 300  # 5-minute rolling window
    
    # Internal state
    _ttft_measurements: deque = field(default_factory=deque)
    _itl_measurements: deque = field(default_factory=deque)
    _timestamps: deque = field(default_factory=deque)
    
    def record_ttft(self, ttft_ms: float, timestamp: Optional[float] = None):
        ts = timestamp or time.time()
        self._ttft_measurements.append((ts, ttft_ms))
        self._evict_old(ts)
    
    def record_itl(self, itl_ms: float, timestamp: Optional[float] = None):
        ts = timestamp or time.time()
        self._itl_measurements.append((ts, itl_ms))
        self._evict_old(ts)
    
    def _evict_old(self, now: float):
        cutoff = now - self.window_seconds
        while self._ttft_measurements and self._ttft_measurements[0][0] < cutoff:
            self._ttft_measurements.popleft()
        while self._itl_measurements and self._itl_measurements[0][0] < cutoff:
            self._itl_measurements.popleft()
    
    @property
    def ttft_compliance(self) -> float:
        """Fraction of TTFT measurements within SLO."""
        if not self._ttft_measurements:
            return 1.0
        good = sum(1 for _, v in self._ttft_measurements if v <= self.ttft_slo_ms)
        return good / len(self._ttft_measurements)
    
    @property
    def itl_compliance(self) -> float:
        """Fraction of ITL measurements within SLO."""
        if not self._itl_measurements:
            return 1.0
        good = sum(1 for _, v in self._itl_measurements if v <= self.itl_slo_ms)
        return good / len(self._itl_measurements)
    
    @property
    def overall_compliance(self) -> float:
        """Combined SLO compliance (both must be met)."""
        return min(self.ttft_compliance, self.itl_compliance)
    
    def should_alert(self, threshold: float = 0.95) -> bool:
        """Returns True if compliance dropped below threshold."""
        return self.overall_compliance < threshold
```

---

## GPU-Side Metrics: What the Hardware Tells You

User-facing metrics tell you WHAT is happening. GPU metrics tell you WHY.

### Streaming Multiprocessor (SM) Utilization

SM utilization measures what fraction of the GPU's compute cores are actively executing kernels. For LLM inference:

- **During prefill**: SM utilization should be high (70-95%) because prefill is compute-bound matrix multiplications
- **During decode**: SM utilization is typically low (10-40%) because decode is memory-bandwidth-bound, waiting for weight loads

If you see low SM utilization during prefill, your batch size is too small to saturate the GPU. If you see high SM utilization during decode, something unusual is happening (possibly very large batch sizes where the matrix multiplications become compute-bound again).

### Memory Bandwidth Utilization

This is the critical metric during decode. Each decode step must load the full model weights from HBM to compute one token per request. Memory bandwidth utilization tells you how close you are to the theoretical maximum:

- **A100 80GB**: 2 TB/s HBM bandwidth
- **H100 80GB**: 3.35 TB/s HBM bandwidth
- **Llama 7B FP16**: ~14 GB weights per decode step
- **Theoretical minimum decode time**: 14 GB / 2 TB/s = 7ms on A100

If your actual decode step takes 15ms on A100, you are at ~47% bandwidth utilization. The gap comes from kernel launch overhead, attention computation, and memory access patterns that do not achieve full bandwidth.

### KV Cache Occupancy

KV cache occupancy is the fraction of allocated KV cache memory currently in use. This is arguably the most important GPU-side metric for LLM serving because it directly determines:

1. **Whether new requests can be admitted** (if occupancy is 100%, new requests must queue)
2. **Whether existing requests will be preempted** (if a new request needs space, the scheduler evicts others)
3. **The effective maximum batch size** (each concurrent request consumes KV cache proportional to its sequence length)

```python
def compute_kv_cache_occupancy(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,  # 2 for FP16, 1 for FP8
    active_sequences: list,  # List of (prompt_len + generated_len) per request
    total_gpu_memory_bytes: int,
    model_weights_bytes: int,
    activation_memory_bytes: int
) -> dict:
    """
    Compute KV cache occupancy and remaining capacity.
    
    Each layer stores K and V tensors for each token in the sequence.
    KV cache per token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    """
    kv_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    
    available_for_kv = total_gpu_memory_bytes - model_weights_bytes - activation_memory_bytes
    max_tokens_possible = available_for_kv // kv_per_token
    
    current_tokens = sum(active_sequences)
    occupancy = current_tokens / max_tokens_possible
    
    remaining_tokens = max_tokens_possible - current_tokens
    
    # How many new requests of average length can we admit?
    avg_seq_len = sum(active_sequences) / len(active_sequences) if active_sequences else 512
    remaining_capacity_requests = remaining_tokens / avg_seq_len
    
    return {
        "kv_per_token_bytes": kv_per_token,
        "total_kv_budget_tokens": max_tokens_possible,
        "current_tokens_stored": current_tokens,
        "occupancy_pct": occupancy * 100,
        "remaining_token_slots": remaining_tokens,
        "estimated_remaining_requests": int(remaining_capacity_requests),
        "memory_breakdown": {
            "model_weights_gb": model_weights_bytes / 1e9,
            "kv_cache_used_gb": (current_tokens * kv_per_token) / 1e9,
            "kv_cache_free_gb": (remaining_tokens * kv_per_token) / 1e9,
            "activations_gb": activation_memory_bytes / 1e9,
        }
    }


# Example: Llama 70B on 4x A100-80GB (tensor parallel)
result = compute_kv_cache_occupancy(
    num_layers=80,
    num_kv_heads=8,          # GQA: 8 KV heads
    head_dim=128,
    dtype_bytes=2,           # FP16
    active_sequences=[1024, 2048, 512, 768, 1536, 896, 2048, 1024,
                      640, 1280, 768, 512, 1024, 1536, 896, 2048],
    total_gpu_memory_bytes=4 * 80 * (1024**3),   # 320 GB total
    model_weights_bytes=140 * (1024**3),          # ~140 GB in FP16
    activation_memory_bytes=8 * (1024**3)         # ~8 GB activations
)
print(f"KV cache occupancy: {result['occupancy_pct']:.1f}%")
print(f"Can admit ~{result['estimated_remaining_requests']} more requests")
print(f"Memory: {result['memory_breakdown']}")
```

### Batch Queue Depth

Queue depth measures how many requests are waiting for a slot in the running batch. A consistently growing queue means your system is overloaded: requests arrive faster than they complete. Queue depth directly predicts TTFT because queued requests accumulate wait time before their prefill even begins.

The relationship is approximately: `TTFT ≈ queue_wait + prefill_time`. When queue depth is 0, TTFT equals pure prefill time. When queue depth is 10, TTFT includes the time for 10 prior requests to vacate slots.

---

## Request-Level Metrics: The Full Lifecycle

Every request passes through distinct phases. Instrumenting each phase separately reveals exactly where time is spent.

### The Request Timeline

```
Request arrives → [Queue Wait] → [Prefill] → [Decode token 1] → [Decode token 2] → ... → [Final token] → Response complete
                  ←── TTFT ────→
                  ←────────────────────── End-to-End Latency ──────────────────────────────────────────→
```

### Instrumentation Points

```python
import time
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class StopReason(Enum):
    EOS_TOKEN = "eos"          # Model generated end-of-sequence
    MAX_TOKENS = "max_tokens"  # Hit token limit
    TIMEOUT = "timeout"        # Request timed out
    PREEMPTED = "preempted"    # Evicted from KV cache
    CLIENT_DISCONNECT = "disconnect"  # Client cancelled
    OOM = "oom"                # Out of memory during generation


@dataclass
class RequestTrace:
    """Complete trace of a single request's lifecycle."""
    request_id: str
    model: str
    
    # Timestamps (all in seconds since epoch)
    arrived_at: float = field(default_factory=time.time)
    scheduled_at: Optional[float] = None      # When prefill started
    first_token_at: Optional[float] = None    # When first token generated
    completed_at: Optional[float] = None      # When generation finished
    
    # Token counts
    prompt_tokens: int = 0
    generated_tokens: int = 0
    
    # Per-step ITL values
    itl_values_ms: List[float] = field(default_factory=list)
    
    # Outcome
    stop_reason: Optional[StopReason] = None
    
    # Computed metrics
    @property
    def queue_time_ms(self) -> Optional[float]:
        if self.scheduled_at:
            return (self.scheduled_at - self.arrived_at) * 1000
        return None
    
    @property
    def prefill_time_ms(self) -> Optional[float]:
        if self.scheduled_at and self.first_token_at:
            return (self.first_token_at - self.scheduled_at) * 1000
        return None
    
    @property
    def ttft_ms(self) -> Optional[float]:
        if self.first_token_at:
            return (self.first_token_at - self.arrived_at) * 1000
        return None
    
    @property
    def decode_time_ms(self) -> Optional[float]:
        if self.first_token_at and self.completed_at:
            return (self.completed_at - self.first_token_at) * 1000
        return None
    
    @property
    def total_time_ms(self) -> Optional[float]:
        if self.completed_at:
            return (self.completed_at - self.arrived_at) * 1000
        return None
    
    @property
    def mean_itl_ms(self) -> Optional[float]:
        if self.itl_values_ms:
            return sum(self.itl_values_ms) / len(self.itl_values_ms)
        return None
    
    @property
    def tokens_per_second(self) -> Optional[float]:
        if self.decode_time_ms and self.generated_tokens > 0:
            return self.generated_tokens / (self.decode_time_ms / 1000)
        return None
    
    def to_metrics_dict(self) -> dict:
        """Export for Prometheus/logging."""
        return {
            "request_id": self.request_id,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "queue_time_ms": self.queue_time_ms,
            "prefill_time_ms": self.prefill_time_ms,
            "ttft_ms": self.ttft_ms,
            "decode_time_ms": self.decode_time_ms,
            "total_time_ms": self.total_time_ms,
            "mean_itl_ms": self.mean_itl_ms,
            "tokens_per_second": self.tokens_per_second,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
        }
```

### Stop Reason Analysis

The stop reason distribution is a powerful diagnostic signal:

- **High EOS rate (>90%)**: Healthy. Model is finishing naturally.
- **High max_tokens rate (>20%)**: Users are hitting limits. Consider increasing max output length or this indicates prompt injection/abuse.
- **Any preemption (>0%)**: KV cache pressure. Scale up or reduce concurrent requests.
- **Any timeout (>0%)**: Requests taking too long. Check queue depth and batch sizes.
- **High disconnect rate (>5%)**: Users are giving up. TTFT or ITL is too slow.

---

## System-Level Metrics: Fleet Health

Beyond individual requests, you need metrics that describe the system as a whole.

### Requests Per Second (RPS)

The inbound request rate determines whether your system is approaching capacity. Unlike web services where RPS directly maps to CPU usage, LLM serving has a more complex relationship: RPS × average_sequence_length × average_output_length determines actual load.

Two workloads with identical RPS can have 10x different resource demands:
- 100 RPS with 50 input tokens, 20 output tokens = low load
- 100 RPS with 4000 input tokens, 500 output tokens = extreme load

Always report RPS alongside token rate for meaningful capacity measurement.

### Concurrent Requests (Batch Size)

The number of requests being simultaneously processed determines memory usage (KV cache), throughput (more batching = higher tokens/sec), and per-request latency (more batching = higher ITL).

The optimal concurrent request count is where goodput peaks: enough batching to amortize memory bandwidth across multiple requests, but not so much that individual requests miss their SLOs.

### Error Rate and Error Classification

Not all errors are equal:

| Error Type | Severity | Root Cause | Action |
|---|---|---|---|
| OOM during prefill | Critical | Input too long for available memory | Reject or chunk input |
| OOM during decode | Critical | KV cache exhausted | Reduce batch size |
| Timeout | High | System overloaded | Scale out or shed load |
| Model error (NaN) | Critical | Numerical instability | Check quantization |
| Client disconnect | Low | User cancelled | Informational only |
| Rate limit | Medium | Traffic spike | Expected behavior |

### Preemption Events

Preemption occurs when the scheduler must evict a partially-generated request from the KV cache to make room for a higher-priority request (or because the cache is full). Each preemption means:

1. Wasted compute: all tokens generated so far are discarded
2. Increased latency: the request must be re-queued and re-prefilled
3. User impact: if streaming, the user sees a stall or error

Preemption rate is a leading indicator of capacity exhaustion. Any preemption rate above 0% warrants immediate investigation.

---

## Monitoring Stack: Prometheus + Grafana for vLLM

vLLM exposes a comprehensive Prometheus metrics endpoint. Here is how to configure monitoring for a production deployment.

### Extracting vLLM Metrics

```python
"""
vLLM Prometheus metrics extraction and custom goodput computation.

vLLM exposes metrics at /metrics endpoint. Key metrics:
- vllm:time_to_first_token_seconds (histogram)
- vllm:time_per_output_token_seconds (histogram)
- vllm:e2e_request_latency_seconds (histogram)
- vllm:num_requests_running (gauge)
- vllm:num_requests_waiting (gauge)
- vllm:gpu_cache_usage_perc (gauge)
- vllm:num_preemptions_total (counter)
- vllm:prompt_tokens_total (counter)
- vllm:generation_tokens_total (counter)
"""
import requests
from prometheus_client.parser import text_string_to_metric_families


def scrape_vllm_metrics(endpoint: str = "http://localhost:8000/metrics") -> dict:
    """
    Scrape vLLM's Prometheus endpoint and return structured metrics.
    
    Returns a dict with categorized metrics ready for dashboarding.
    """
    response = requests.get(endpoint, timeout=5)
    response.raise_for_status()
    
    metrics = {}
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            key = sample.name
            labels = sample.labels
            value = sample.value
            
            if key not in metrics:
                metrics[key] = []
            metrics[key].append({"labels": labels, "value": value})
    
    # Extract key operational metrics
    result = {
        "requests_running": _get_gauge(metrics, "vllm:num_requests_running"),
        "requests_waiting": _get_gauge(metrics, "vllm:num_requests_waiting"),
        "gpu_cache_usage_pct": _get_gauge(metrics, "vllm:gpu_cache_usage_perc") * 100,
        "preemptions_total": _get_counter(metrics, "vllm:num_preemptions_total"),
        "prompt_tokens_total": _get_counter(metrics, "vllm:prompt_tokens_total"),
        "generation_tokens_total": _get_counter(metrics, "vllm:generation_tokens_total"),
        "ttft_p50_ms": _get_histogram_percentile(metrics, "vllm:time_to_first_token_seconds", 0.5) * 1000,
        "ttft_p95_ms": _get_histogram_percentile(metrics, "vllm:time_to_first_token_seconds", 0.95) * 1000,
        "ttft_p99_ms": _get_histogram_percentile(metrics, "vllm:time_to_first_token_seconds", 0.99) * 1000,
        "tpot_p50_ms": _get_histogram_percentile(metrics, "vllm:time_per_output_token_seconds", 0.5) * 1000,
        "tpot_p95_ms": _get_histogram_percentile(metrics, "vllm:time_per_output_token_seconds", 0.95) * 1000,
        "tpot_p99_ms": _get_histogram_percentile(metrics, "vllm:time_per_output_token_seconds", 0.99) * 1000,
    }
    
    return result


def _get_gauge(metrics: dict, name: str) -> float:
    samples = metrics.get(name, [])
    return samples[0]["value"] if samples else 0.0


def _get_counter(metrics: dict, name: str) -> float:
    # Counters have _total suffix in exposition format
    samples = metrics.get(f"{name}_total", metrics.get(name, []))
    return samples[0]["value"] if samples else 0.0


def _get_histogram_percentile(metrics: dict, name: str, percentile: float) -> float:
    """
    Approximate percentile from Prometheus histogram buckets.
    
    Uses linear interpolation between bucket boundaries.
    """
    bucket_key = f"{name}_bucket"
    buckets = metrics.get(bucket_key, [])
    
    if not buckets:
        return 0.0
    
    # Sort by upper bound
    sorted_buckets = sorted(
        [(float(b["labels"].get("le", "inf")), b["value"]) for b in buckets],
        key=lambda x: x[0] if x[0] != float("inf") else 1e18
    )
    
    total = sorted_buckets[-1][1]  # +Inf bucket has total count
    if total == 0:
        return 0.0
    
    target = percentile * total
    
    prev_bound = 0
    prev_count = 0
    for bound, count in sorted_buckets:
        if count >= target:
            # Linear interpolation within this bucket
            fraction = (target - prev_count) / max(count - prev_count, 1)
            return prev_bound + fraction * (bound - prev_bound)
        prev_bound = bound
        prev_count = count
    
    return sorted_buckets[-2][0]  # Return last finite bucket


# Example Grafana dashboard queries (PromQL)
GRAFANA_QUERIES = {
    "ttft_p99": 'histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[5m]))',
    "tpot_p95": 'histogram_quantile(0.95, rate(vllm:time_per_output_token_seconds_bucket[5m]))',
    "throughput_tokens_per_sec": 'rate(vllm:generation_tokens_total[1m])',
    "cache_pressure": 'vllm:gpu_cache_usage_perc',
    "queue_depth": 'vllm:num_requests_waiting',
    "preemption_rate": 'rate(vllm:num_preemptions_total[5m])',
    "request_rate": 'rate(vllm:request_success_total[1m]) + rate(vllm:request_failure_total[1m])',
    "error_rate": 'rate(vllm:request_failure_total[1m]) / (rate(vllm:request_success_total[1m]) + rate(vllm:request_failure_total[1m]))',
}
```

### Grafana Dashboard Layout

A production LLM inference Grafana dashboard should have these panels arranged in order of diagnostic priority:

**Row 1: SLO Compliance (the "is everything OK?" row)**
- Goodput ratio (single stat, green >0.9, yellow >0.7, red <0.7)
- TTFT SLO compliance % (single stat)
- ITL SLO compliance % (single stat)
- Active alerts count

**Row 2: User-Facing Latency**
- TTFT p50/p95/p99 over time (line chart)
- ITL p50/p95/p99 over time (line chart)
- E2E latency heatmap (by output length bucket)

**Row 3: System Capacity**
- KV cache occupancy % (gauge, alert at 90%)
- Requests running vs waiting (stacked area)
- Throughput tokens/sec (line)
- Preemption rate (should be 0)

**Row 4: GPU Utilization**
- SM utilization % per GPU (line chart)
- Memory bandwidth utilization (line chart)
- GPU memory used vs total (stacked bar)

---

## Alerting: What Deserves a Page

Not every metric deviation warrants human attention. The goal of alerting is to notify when user experience is degrading and automated mitigation has not resolved it.

### Critical Alerts (Page immediately)

```yaml
# Alert: KV cache approaching capacity
- alert: KVCacheNearCapacity
  expr: vllm:gpu_cache_usage_perc > 0.90
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "KV cache at {{ $value | humanizePercentage }}, preemptions imminent"
    runbook: "Scale out replicas or reduce max_concurrent_requests"

# Alert: TTFT SLO violation sustained
- alert: TTFTSLOViolation
  expr: |
    histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[5m])) > 0.5
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "TTFT p99 is {{ $value }}s, SLO is 500ms"
    runbook: "Check queue depth; if high, scale out. If low, check prefill batch size."

# Alert: Preemptions occurring
- alert: PreemptionRateHigh
  expr: rate(vllm:num_preemptions_total[5m]) > 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Preemptions detected: {{ $value }}/sec. Users experiencing interrupted generation."
    runbook: "Immediately reduce max batch size or scale out."
```

### Warning Alerts (Investigate within 15 minutes)

```yaml
# Alert: Queue building up
- alert: QueueDepthRising
  expr: vllm:num_requests_waiting > 10
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "{{ $value }} requests in queue. TTFT will degrade."

# Alert: Goodput ratio dropping
- alert: GoodputDegrading
  expr: |
    (rate(vllm:generation_tokens_total{slo_met="true"}[5m]) 
     / rate(vllm:generation_tokens_total[5m])) < 0.8
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Goodput ratio at {{ $value | humanizePercentage }}. >20% of work is wasted."

# Alert: Error rate spike
- alert: InferenceErrorRate
  expr: |
    rate(vllm:request_failure_total[5m]) 
    / (rate(vllm:request_success_total[5m]) + rate(vllm:request_failure_total[5m])) > 0.01
  for: 3m
  labels:
    severity: warning
  annotations:
    summary: "Error rate at {{ $value | humanizePercentage }}. Check for OOM or model issues."
```

### Alert Anti-Patterns

**Do not alert on:**
- Instantaneous metric spikes (use `for:` duration to filter transients)
- GPU utilization being "too low" (decode is memory-bound by design)
- Individual request timeouts (track rate, not individual events)
- Raw throughput drops without corresponding SLO violations (might be lower traffic)

**Do alert on:**
- Sustained SLO violations (>5 minutes)
- Any preemption (always indicates capacity issues)
- KV cache >90% (preemptions are seconds away)
- Queue depth growing monotonically (system cannot keep up)
- Goodput ratio declining (wasting GPU spend)

---

## Cost Metrics: Dollars Per Token

GPU inference is expensive. Understanding cost-per-token enables rational capacity decisions and pricing.

### Computing Cost Per Million Tokens

```python
from dataclasses import dataclass


@dataclass
class CostModel:
    """
    Compute inference cost per token for a deployment.
    
    Key insight: cost/token improves with utilization but degrades
    when over-loaded (because wasted tokens from preemptions/timeouts
    still consume GPU time).
    """
    # Hardware costs
    gpu_hourly_cost: float          # $/hour per GPU (e.g., $3.50 for A100 on-demand)
    num_gpus: int                   # GPUs in this deployment
    
    # Observed metrics
    goodput_tokens_per_sec: float   # From goodput calculation
    raw_tokens_per_sec: float       # Total tokens including wasted
    gpu_utilization_pct: float      # Average across GPUs during decode
    
    @property
    def cost_per_hour(self) -> float:
        return self.gpu_hourly_cost * self.num_gpus
    
    @property
    def cost_per_million_tokens_raw(self) -> float:
        """Cost per 1M tokens (counting all tokens including wasted)."""
        tokens_per_hour = self.raw_tokens_per_sec * 3600
        if tokens_per_hour == 0:
            return float('inf')
        return (self.cost_per_hour / tokens_per_hour) * 1_000_000
    
    @property
    def cost_per_million_tokens_good(self) -> float:
        """Cost per 1M GOOD tokens (only counting SLO-meeting tokens)."""
        good_tokens_per_hour = self.goodput_tokens_per_sec * 3600
        if good_tokens_per_hour == 0:
            return float('inf')
        return (self.cost_per_hour / good_tokens_per_hour) * 1_000_000
    
    @property
    def waste_cost_per_hour(self) -> float:
        """Dollars/hour spent on tokens that don't meet SLOs."""
        wasted_fraction = 1 - (self.goodput_tokens_per_sec / self.raw_tokens_per_sec)
        return self.cost_per_hour * wasted_fraction
    
    @property
    def idle_cost_per_hour(self) -> float:
        """Dollars/hour of idle GPU capacity (not generating any tokens)."""
        # Rough: if GPU is X% utilized, (100-X)% is idle
        return self.cost_per_hour * (1 - self.gpu_utilization_pct / 100)


# Example: 4x A100 deployment
model = CostModel(
    gpu_hourly_cost=3.50,      # On-demand A100 80GB
    num_gpus=4,
    goodput_tokens_per_sec=4500,
    raw_tokens_per_sec=6000,
    gpu_utilization_pct=65
)

print(f"Deployment cost: ${model.cost_per_hour:.2f}/hour")
print(f"Cost per 1M tokens (raw): ${model.cost_per_million_tokens_raw:.2f}")
print(f"Cost per 1M tokens (good): ${model.cost_per_million_tokens_good:.2f}")
print(f"Waste (SLO violations): ${model.waste_cost_per_hour:.2f}/hour")
print(f"Idle capacity cost: ${model.idle_cost_per_hour:.2f}/hour")
print(f"Effective cost premium from waste: {model.cost_per_million_tokens_good / model.cost_per_million_tokens_raw:.1f}x")
```

### Cost Optimization Levers

The cost equation has three terms you can optimize:

1. **Reduce idle cost**: Right-size instances, use spot/preemptible where SLOs allow, autoscale based on queue depth
2. **Reduce waste cost**: Improve goodput ratio by avoiding overload (the cheapest token is the one you don't generate)
3. **Reduce per-token compute cost**: Quantization (FP16 → INT8 → INT4), smaller models for easier tasks, speculative decoding for throughput

The relationship between batch size and cost is non-linear:
- Too few requests: GPU is idle, high $/token from underutilization
- Optimal batch: GPU is busy, all requests meet SLO, lowest $/good-token
- Too many requests: GPU is busy but goodput drops, $/good-token rises despite high utilization

---

## Benchmarking Methodology

### Open-Loop vs Closed-Loop

This distinction is critical and frequently confused.

**Closed-loop benchmarking**: Send a request, wait for completion, send next request. This models a single user typing queries sequentially. It underestimates real-world load because it never builds a queue.

**Open-loop benchmarking**: Send requests at a fixed rate regardless of whether prior requests have completed. This models real traffic where users arrive independently. It reveals queuing behavior and SLO violations under load.

Production traffic is open-loop. Always benchmark with open-loop methodology. Closed-loop benchmarks will show artificially good latency numbers because the system never becomes overloaded.

```python
import asyncio
import time
import numpy as np
from typing import Callable, Awaitable, List


async def open_loop_benchmark(
    send_request: Callable[[], Awaitable[dict]],
    target_rps: float,
    duration_seconds: float,
    warmup_seconds: float = 10.0
) -> dict:
    """
    Open-loop benchmark: sends requests at fixed rate independent of completions.
    
    Args:
        send_request: async function that sends one request and returns metrics dict
        target_rps: desired requests per second
        duration_seconds: total benchmark duration (excluding warmup)
        warmup_seconds: initial period whose results are discarded
        
    Returns:
        dict with latency percentiles, throughput, and goodput measurements
    """
    interval = 1.0 / target_rps
    results: List[dict] = []
    warmup_end = time.time() + warmup_seconds
    benchmark_end = warmup_end + duration_seconds
    
    async def launch_request(request_time: float):
        result = await send_request()
        result["launched_at"] = request_time
        result["is_warmup"] = request_time < warmup_end
        results.append(result)
    
    # Launch requests at fixed intervals using Poisson arrivals
    tasks = []
    start = time.time()
    
    while time.time() < benchmark_end:
        # Poisson inter-arrival time (more realistic than fixed interval)
        wait = np.random.exponential(interval)
        await asyncio.sleep(wait)
        
        task = asyncio.create_task(launch_request(time.time()))
        tasks.append(task)
    
    # Wait for all in-flight requests to complete (with timeout)
    await asyncio.wait(tasks, timeout=60.0)
    
    # Filter out warmup results
    measured = [r for r in results if not r.get("is_warmup", True)]
    
    if not measured:
        return {"error": "No measurements collected after warmup"}
    
    ttfts = [r["ttft_ms"] for r in measured if "ttft_ms" in r]
    itls = [r["mean_itl_ms"] for r in measured if "mean_itl_ms" in r]
    e2e = [r["total_ms"] for r in measured if "total_ms" in r]
    
    return {
        "target_rps": target_rps,
        "actual_rps": len(measured) / duration_seconds,
        "total_requests": len(measured),
        "ttft": {
            "p50_ms": np.percentile(ttfts, 50),
            "p95_ms": np.percentile(ttfts, 95),
            "p99_ms": np.percentile(ttfts, 99),
            "mean_ms": np.mean(ttfts),
        } if ttfts else None,
        "itl": {
            "p50_ms": np.percentile(itls, 50),
            "p95_ms": np.percentile(itls, 95),
            "p99_ms": np.percentile(itls, 99),
            "mean_ms": np.mean(itls),
        } if itls else None,
        "e2e_latency": {
            "p50_ms": np.percentile(e2e, 50),
            "p95_ms": np.percentile(e2e, 95),
            "p99_ms": np.percentile(e2e, 99),
            "mean_ms": np.mean(e2e),
        } if e2e else None,
        "errors": sum(1 for r in measured if r.get("error")),
        "timeouts": sum(1 for r in measured if r.get("timeout")),
    }
```

### Warmup and Statistical Significance

**Warmup**: The first 10-30 seconds of any benchmark produce unreliable numbers. The model is loading, CUDA kernels are being JIT-compiled, the KV cache is empty, and the batch is building. Always discard warmup measurements.

**Statistical significance**: A single benchmark run is noise. You need:
- Minimum 1000 requests per configuration (more for tail percentiles)
- At least 3 independent runs with different random seeds
- Report confidence intervals, not just point estimates
- For p99, you need 10,000+ requests (99th percentile of 100 requests is a single data point)

**Confounding variables to control:**
- Input length distribution (use realistic distribution, not fixed length)
- Output length distribution (model behavior varies with prompt)
- Request arrival pattern (Poisson, not fixed interval)
- System temperature (GPU thermal throttling after sustained load)
- Other workloads on the same machine (isolate benchmarks)

### Common Benchmarking Mistakes

1. **Using fixed input/output lengths**: Real traffic has high variance. A system tuned for 512-input/256-output may collapse at 4096-input/1024-output.

2. **Reporting only average latency**: Hides tail latency that affects real users. Always report p50, p95, p99.

3. **Closed-loop testing at high "throughput"**: Shows excellent latency because the system never queues. Tells you nothing about production behavior.

4. **Ignoring warmup**: First 100 requests will have high latency from compilation and cache warming. Including them skews results.

5. **Benchmarking on idle systems**: Production systems have background tasks (health checks, model reloads, logging). Benchmark under realistic conditions.

6. **Comparing engines at different batch sizes**: vLLM at batch=32 vs TGI at batch=8 is not a fair comparison. Sweep batch sizes and compare at the same SLO constraint.

### Production Benchmarking Tools

Building benchmarks from scratch (as shown above) teaches you the mechanics. In production, use established tools:

- **vLLM built-in benchmark**: `python -m vllm.entrypoints.openai.bench_serving` -- ships with vLLM, supports ShareGPT datasets
- **GenAI-Perf (NVIDIA)**: Part of Triton Inference Server, supports streaming, multi-model, custom token distributions
- **LLMPerf (Anyscale)**: Measures TTFT, ITL, throughput with configurable concurrency ramps. Defines goodput.
- **ShareGPT dataset**: Real conversation traces (variable prompt/completion lengths) -- standard for realistic benchmarks vs synthetic fixed-length inputs

---

## Putting It All Together: The Goodput Mental Model

After reading this chapter, you should carry one mental model forward:

**Goodput is the only metric that matters in production.**

Raw throughput is a vanity metric. GPU utilization is a cost metric. Latency percentiles are diagnostic tools. But goodput, the rate at which you produce tokens that satisfy your users, is the single number that combines all of them into an actionable answer: "Is my deployment doing its job?"

Every optimization decision should be evaluated against goodput:
- Does increasing batch size improve goodput? (Yes, until SLOs start breaking)
- Does quantization improve goodput? (Yes, if latency drops without quality degradation)
- Does adding a GPU improve goodput? (Only if the bottleneck is capacity, not a software bug)
- Does continuous batching improve goodput? (Yes, by filling idle decode slots)

When someone asks "how is inference performing?", the answer is not "we are at 80% GPU utilization" or "throughput is 10K tokens/sec." The answer is "our goodput is 8,200 tokens/sec with 96.5% SLO attainment." That single statement tells you the system is productive, the users are served, and the remaining 3.5% gap indicates where to focus next.

This metric connects directly back to cost: if your goodput is 8,200 tokens/sec and your deployment costs $14/hour, your effective cost is $0.47 per million good tokens. That is the number that determines whether your deployment is economically viable and how it compares to managed API pricing.

---

## Summary

| Metric | Layer | Purpose | Alert Threshold |
|---|---|---|---|
| TTFT p99 | User | First impression speed | > SLO |
| ITL p99 | User | Streaming smoothness | > SLO |
| Goodput ratio | User | Overall quality of service | < 0.8 |
| KV cache occupancy | GPU | Capacity headroom | > 90% |
| Queue depth | System | Incoming overload | Growing monotonically |
| Preemption rate | System | Capacity exhaustion | > 0 |
| $/M good tokens | Cost | Economic efficiency | > budget |
| Stop reason distribution | Diagnostic | Why requests end | Abnormal shift |

The metrics hierarchy is simple: start with goodput to know if things are OK, drill into TTFT/ITL percentiles to understand what is wrong, check GPU/system metrics to understand why, and use cost metrics to determine the economic tradeoff of fixing it. This chapter gives you the instrumentation and methodology to do all four.
