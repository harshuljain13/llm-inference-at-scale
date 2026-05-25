# Module 7: Measurement and Operations

> Benchmarking, monitoring, and troubleshooting LLM inference systems

---

## Learning Objectives

By the end of this module, you will:

- Define and measure key LLM inference metrics
- Set up comprehensive monitoring dashboards
- Benchmark inference performance accurately
- Troubleshoot common production issues

---

## Key Metrics

### Metric Definitions

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM INFERENCE METRICS                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   LATENCY METRICS:                                                  │
│   ════════════════                                                  │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                                                             │   │
│   │   Request ──► Queue ──► Prefill ──► Decode ──► Response     │   │
│   │      │         │          │           │           │         │   │
│   │      └─────────┴──────────┴───────────┴───────────┘         │   │
│   │            │                    │              │            │   │
│   │         Queue              TTFT           Total             │   │
│   │         Time          (Time to First    Latency            │   │
│   │                         Token)                              │   │
│   │                                                             │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   • Queue Time: Time waiting before processing starts               │
│   • TTFT: Time from request to first token (includes prefill)       │
│   • ITL: Inter-Token Latency (time between tokens)                  │
│   • Total Latency: End-to-end request time                          │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   THROUGHPUT METRICS:                                               │
│   ═══════════════════                                               │
│                                                                     │
│   • Tokens/second: Total tokens generated per second                │
│   • Requests/second: Completed requests per second                  │
│   • Tokens/second/GPU: Normalized throughput per GPU                │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   RESOURCE METRICS:                                                 │
│   ═════════════════                                                 │
│                                                                     │
│   • GPU Utilization: % of GPU compute used                          │
│   • GPU Memory: VRAM usage (model + KV cache)                       │
│   • KV Cache Utilization: % of allocated KV cache used              │
│   • Batch Size: Current number of sequences in batch                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Metric Targets by Use Case

| Use Case            | TTFT Target | ITL Target | Throughput Priority |
| ------------------- | ----------- | ---------- | ------------------- |
| Real-time Chat      | < 500ms     | < 50ms     | Medium              |
| Streaming Response  | < 1s        | < 30ms     | Medium              |
| Batch Processing    | < 5s        | < 100ms    | High                |
| Code Completion     | < 200ms     | < 20ms     | Low                 |
| Document Processing | < 2s        | < 50ms     | High                |

---

## Benchmarking

### Benchmark Suite Implementation

```python
import asyncio
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Optional
import aiohttp
import json


@dataclass
class BenchmarkResult:
    """Results from a single benchmark request."""
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float  # Time to first token
    total_latency_ms: float
    tokens_per_second: float
    success: bool
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark results."""
    total_requests: int
    successful_requests: int
    failed_requests: int

    # Latency stats (ms)
    ttft_p50: float
    ttft_p90: float
    ttft_p99: float
    ttft_mean: float

    total_latency_p50: float
    total_latency_p90: float
    total_latency_p99: float
    total_latency_mean: float

    # Throughput
    tokens_per_second: float
    requests_per_second: float

    # Duration
    total_duration_seconds: float


async def benchmark_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int = 100,
) -> BenchmarkResult:
    """Execute a single benchmark request with streaming."""

    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    start_time = time.perf_counter()
    ttft = None
    completion_tokens = 0

    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            async for line in response.content:
                if ttft is None:
                    ttft = (time.perf_counter() - start_time) * 1000

                line = line.decode('utf-8').strip()
                if line.startswith('data: ') and line != 'data: [DONE]':
                    data = json.loads(line[6:])
                    if data.get('choices', [{}])[0].get('delta', {}).get('content'):
                        completion_tokens += 1

        total_latency = (time.perf_counter() - start_time) * 1000

        return BenchmarkResult(
            prompt_tokens=len(prompt.split()),  # Approximate
            completion_tokens=completion_tokens,
            ttft_ms=ttft or total_latency,
            total_latency_ms=total_latency,
            tokens_per_second=completion_tokens / (total_latency / 1000) if total_latency > 0 else 0,
            success=True,
        )

    except Exception as e:
        return BenchmarkResult(
            prompt_tokens=0,
            completion_tokens=0,
            ttft_ms=0,
            total_latency_ms=0,
            tokens_per_second=0,
            success=False,
            error=str(e),
        )


async def run_benchmark(
    url: str,
    prompts: List[str],
    concurrency: int = 10,
    max_tokens: int = 100,
) -> BenchmarkSummary:
    """Run benchmark with specified concurrency."""

    start_time = time.perf_counter()
    results: List[BenchmarkResult] = []

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_request(prompt):
            async with semaphore:
                return await benchmark_request(session, url, prompt, max_tokens)

        tasks = [bounded_request(p) for p in prompts]
        results = await asyncio.gather(*tasks)

    total_duration = time.perf_counter() - start_time

    # Calculate statistics
    successful = [r for r in results if r.success]

    if not successful:
        raise ValueError("All requests failed!")

    ttfts = [r.ttft_ms for r in successful]
    latencies = [r.total_latency_ms for r in successful]
    total_tokens = sum(r.completion_tokens for r in successful)

    def percentile(data, p):
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]

    return BenchmarkSummary(
        total_requests=len(results),
        successful_requests=len(successful),
        failed_requests=len(results) - len(successful),

        ttft_p50=percentile(ttfts, 50),
        ttft_p90=percentile(ttfts, 90),
        ttft_p99=percentile(ttfts, 99),
        ttft_mean=statistics.mean(ttfts),

        total_latency_p50=percentile(latencies, 50),
        total_latency_p90=percentile(latencies, 90),
        total_latency_p99=percentile(latencies, 99),
        total_latency_mean=statistics.mean(latencies),

        tokens_per_second=total_tokens / total_duration,
        requests_per_second=len(successful) / total_duration,

        total_duration_seconds=total_duration,
    )


# Example usage
async def main():
    prompts = [
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about programming.",
    ] * 100  # 300 total requests

    summary = await run_benchmark(
        url="http://localhost:8000",
        prompts=prompts,
        concurrency=32,
        max_tokens=100,
    )

    print(f"Benchmark Results:")
    print(f"  Requests: {summary.successful_requests}/{summary.total_requests}")
    print(f"  TTFT P50/P90/P99: {summary.ttft_p50:.0f}/{summary.ttft_p90:.0f}/{summary.ttft_p99:.0f} ms")
    print(f"  Latency P50/P90/P99: {summary.total_latency_p50:.0f}/{summary.total_latency_p90:.0f}/{summary.total_latency_p99:.0f} ms")
    print(f"  Throughput: {summary.tokens_per_second:.0f} tokens/s, {summary.requests_per_second:.1f} req/s")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Monitoring Dashboard

### CloudWatch Dashboard Specification

```yaml
# cloudwatch-dashboard.yaml
AWSTemplateFormatVersion: "2010-09-09"
Description: LLM Inference Monitoring Dashboard

Resources:
  LLMDashboard:
    Type: AWS::CloudWatch::Dashboard
    Properties:
      DashboardName: LLM-Inference-Dashboard
      DashboardBody: !Sub |
        {
          "widgets": [
            {
              "type": "metric",
              "x": 0, "y": 0, "width": 12, "height": 6,
              "properties": {
                "title": "Request Latency",
                "metrics": [
                  ["LLMInference", "TTFT_P50", {"label": "TTFT P50"}],
                  [".", "TTFT_P99", {"label": "TTFT P99"}],
                  [".", "TotalLatency_P50", {"label": "Total P50"}],
                  [".", "TotalLatency_P99", {"label": "Total P99"}]
                ],
                "period": 60,
                "stat": "Average"
              }
            },
            {
              "type": "metric",
              "x": 12, "y": 0, "width": 12, "height": 6,
              "properties": {
                "title": "Throughput",
                "metrics": [
                  ["LLMInference", "TokensPerSecond", {"label": "Tokens/s"}],
                  [".", "RequestsPerSecond", {"label": "Requests/s", "yAxis": "right"}]
                ],
                "period": 60
              }
            },
            {
              "type": "metric",
              "x": 0, "y": 6, "width": 8, "height": 6,
              "properties": {
                "title": "GPU Utilization",
                "metrics": [
                  ["LLMInference", "GPUUtilization", {"label": "GPU %"}],
                  [".", "GPUMemoryUsed", {"label": "Memory GB", "yAxis": "right"}]
                ],
                "period": 60
              }
            },
            {
              "type": "metric",
              "x": 8, "y": 6, "width": 8, "height": 6,
              "properties": {
                "title": "Queue & Batch",
                "metrics": [
                  ["LLMInference", "QueueDepth", {"label": "Queue Depth"}],
                  [".", "BatchSize", {"label": "Batch Size"}],
                  [".", "KVCacheUtilization", {"label": "KV Cache %", "yAxis": "right"}]
                ],
                "period": 60
              }
            },
            {
              "type": "metric",
              "x": 16, "y": 6, "width": 8, "height": 6,
              "properties": {
                "title": "Errors",
                "metrics": [
                  ["LLMInference", "ErrorRate", {"label": "Error Rate %"}],
                  [".", "TimeoutRate", {"label": "Timeout Rate %"}]
                ],
                "period": 60
              }
            }
          ]
        }
```

### Prometheus Metrics Export

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
REQUEST_LATENCY = Histogram(
    'llm_request_latency_seconds',
    'Request latency in seconds',
    ['model', 'status'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

TTFT_LATENCY = Histogram(
    'llm_ttft_seconds',
    'Time to first token in seconds',
    ['model'],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
)

TOKENS_GENERATED = Counter(
    'llm_tokens_generated_total',
    'Total tokens generated',
    ['model']
)

REQUESTS_TOTAL = Counter(
    'llm_requests_total',
    'Total requests',
    ['model', 'status']
)

GPU_UTILIZATION = Gauge(
    'llm_gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)

KV_CACHE_UTILIZATION = Gauge(
    'llm_kv_cache_utilization_percent',
    'KV cache utilization percentage'
)

BATCH_SIZE = Gauge(
    'llm_current_batch_size',
    'Current batch size'
)


class MetricsMiddleware:
    """Middleware to collect inference metrics."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def record_request(
        self,
        ttft_seconds: float,
        total_latency_seconds: float,
        tokens_generated: int,
        success: bool
    ):
        status = "success" if success else "error"

        REQUEST_LATENCY.labels(
            model=self.model_name,
            status=status
        ).observe(total_latency_seconds)

        TTFT_LATENCY.labels(model=self.model_name).observe(ttft_seconds)

        TOKENS_GENERATED.labels(model=self.model_name).inc(tokens_generated)

        REQUESTS_TOTAL.labels(
            model=self.model_name,
            status=status
        ).inc()

    def update_gpu_metrics(self, gpu_id: int, utilization: float):
        GPU_UTILIZATION.labels(gpu_id=str(gpu_id)).set(utilization)

    def update_batch_metrics(self, batch_size: int, kv_cache_util: float):
        BATCH_SIZE.set(batch_size)
        KV_CACHE_UTILIZATION.set(kv_cache_util)


# Start metrics server
start_http_server(9090)
```

---

## Troubleshooting Guide

### Common Issues and Solutions

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TROUBLESHOOTING GUIDE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   SYMPTOM: High TTFT (Time to First Token)                          │
│   ═══════════════════════════════════════                           │
│   Possible Causes:                                                  │
│   • Long prompts causing slow prefill                               │
│   • GPU memory pressure                                             │
│   • High queue depth                                                │
│                                                                     │
│   Solutions:                                                        │
│   1. Enable chunked prefill: --enable-chunked-prefill               │
│   2. Reduce max_num_seqs to lower queue depth                       │
│   3. Check GPU memory utilization                                   │
│   4. Consider prompt caching for repeated prefixes                  │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   SYMPTOM: High ITL (Inter-Token Latency)                           │
│   ═══════════════════════════════════════                           │
│   Possible Causes:                                                  │
│   • Too many concurrent sequences                                   │
│   • Memory bandwidth saturation                                     │
│   • Large batch causing slow iterations                             │
│                                                                     │
│   Solutions:                                                        │
│   1. Reduce max_num_seqs                                            │
│   2. Use quantization (INT8/INT4) to reduce memory reads            │
│   3. Add tensor parallelism for more bandwidth                      │
│   4. Set max_num_batched_tokens to limit batch size                 │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   SYMPTOM: OOM (Out of Memory) Errors                               │
│   ═════════════════════════════════════                             │
│   Possible Causes:                                                  │
│   • KV cache too large                                              │
│   • Too many concurrent sequences                                   │
│   • Model + KV cache exceeds VRAM                                   │
│                                                                     │
│   Solutions:                                                        │
│   1. Reduce gpu_memory_utilization (e.g., 0.85)                     │
│   2. Reduce max_num_seqs                                            │
│   3. Reduce max_model_len                                           │
│   4. Use quantization                                               │
│   5. Add more GPUs with tensor parallelism                          │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   SYMPTOM: Low GPU Utilization                                      │
│   ════════════════════════════                                      │
│   Possible Causes:                                                  │
│   • Not enough concurrent requests                                  │
│   • Memory-bound decode phase                                       │
│   • Inefficient batching                                            │
│                                                                     │
│   Solutions:                                                        │
│   1. Increase load (more concurrent requests)                       │
│   2. Increase max_num_seqs to allow larger batches                  │
│   3. This is NORMAL for decode phase (memory-bound)                 │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   SYMPTOM: Requests Timing Out                                      │
│   ════════════════════════════                                      │
│   Possible Causes:                                                  │
│   • Queue too deep                                                  │
│   • Generation too slow                                             │
│   • Deadlock or hang                                                │
│                                                                     │
│   Solutions:                                                        │
│   1. Check queue depth metrics                                      │
│   2. Reduce max_num_seqs                                            │
│   3. Add more replicas                                              │
│   4. Check for CUDA errors in logs                                  │
│   5. Restart if hung (last resort)                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Diagnostic Commands

```bash
# Check GPU status
nvidia-smi

# Watch GPU utilization
watch -n 0.5 nvidia-smi

# Check vLLM metrics
curl http://localhost:8000/metrics

# Check vLLM health
curl http://localhost:8000/health

# View vLLM logs
docker logs -f vllm-container

# Check CUDA errors
dmesg | grep -i nvidia

# Memory usage breakdown
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
```

---

## Key Takeaways

1. **Measure what matters** - TTFT, ITL, throughput, not just average latency

2. **P99 is critical** - Tail latency affects user experience

3. **Monitor GPU metrics** - Utilization, memory, KV cache

4. **Benchmark realistically** - Use production-like prompts and concurrency

5. **Troubleshoot systematically** - Check metrics, identify bottleneck, apply fix

6. **Automate alerting** - Set thresholds for latency, errors, resource usage

---

## Lab Preview: Benchmarking and Monitoring

In Lab 10, you will:

- Run comprehensive benchmarks
- Set up Prometheus + Grafana monitoring
- Create CloudWatch dashboards
- Practice troubleshooting scenarios

---

## References

1. vLLM Metrics Documentation
2. Prometheus Best Practices
3. AWS CloudWatch Documentation
4. NVIDIA DCGM for GPU Monitoring
