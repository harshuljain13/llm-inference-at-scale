# Lab 10: Benchmarking and Monitoring

## Overview

Learn to benchmark LLM inference systems and set up production monitoring with key metrics, dashboards, and alerting.

## Learning Objectives

- Measure TTFT, TBT, throughput, and latency percentiles
- Design representative benchmark workloads
- Set up CloudWatch dashboards
- Configure alerts for SLO violations

## Prerequisites

- Completed Labs 1-9
- Running LLM inference endpoint
- AWS CloudWatch access

## Setup

```bash
pip install aiohttp numpy matplotlib boto3
```

## Duration

60-90 minutes

## AWS Cost

~$2-5 (endpoint running + CloudWatch)

## Key Metrics

| Metric      | Definition          | Target (Chatbot) |
| ----------- | ------------------- | ---------------- |
| TTFT        | Time to first token | < 500ms P95      |
| TBT         | Time between tokens | < 50ms P95       |
| E2E Latency | Total request time  | < 10s P95        |
| Throughput  | Tokens per second   | > 50 tok/s       |

## Exercises

1. **Benchmark Suite**: Implement async benchmark client
2. **Workload Design**: Create representative prompt distributions
3. **Metric Collection**: Gather TTFT, TBT, throughput
4. **Dashboard Creation**: Build CloudWatch dashboard
5. **Alert Configuration**: Set up SLO-based alerts

## Benchmark Code

```python
async def benchmark_request(session, url, prompt):
    start = time.perf_counter()
    first_token_time = None

    async with session.post(url, json={"prompt": prompt, "stream": True}) as resp:
        async for chunk in resp.content:
            if first_token_time is None:
                first_token_time = time.perf_counter()

    return {
        "ttft_ms": (first_token_time - start) * 1000,
        "e2e_ms": (time.perf_counter() - start) * 1000,
    }
```

## Dashboard Specification

```yaml
widgets:
  - title: "TTFT P95"
    metric: ttft_p95
    threshold: 500ms
    alarm: true

  - title: "Throughput"
    metric: tokens_per_second
    threshold: 100

  - title: "Error Rate"
    metric: error_rate_percent
    threshold: 1%
    alarm: true
```

## Validation Checkpoints

- [ ] Benchmark suite runs successfully
- [ ] Metrics are collected accurately
- [ ] Dashboard displays real-time data
- [ ] Alerts trigger on threshold violations

## Troubleshooting Guide

| Symptom        | Likely Cause                | Solution                                  |
| -------------- | --------------------------- | ----------------------------------------- |
| High TTFT      | Long prefill queue          | Increase replicas, enable chunked prefill |
| High TBT       | Memory bandwidth saturation | Reduce batch size, use quantization       |
| Low throughput | Small batch sizes           | Increase max_num_batched_tokens           |

## Workshop Complete!

Congratulations on completing all labs. You now have hands-on experience with the full LLM inference stack.
