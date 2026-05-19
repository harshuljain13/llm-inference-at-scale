# Lab 4: vLLM Deployment

## Overview

Deploy vLLM with production-optimized configurations. Learn the 6 critical tuning knobs that most teams never touch.

## Learning Objectives

- Configure vLLM for throughput vs latency optimization
- Understand the impact of each tuning knob
- Benchmark different configurations
- Set up OpenAI-compatible API server

## Prerequisites

- Completed Labs 1-3
- AWS g5.2xlarge or larger instance
- HuggingFace token

## Setup

```bash
pip install vllm aiohttp
```

## Duration

60-90 minutes

## AWS Cost

~$2.00 (g5.2xlarge for ~90 minutes)

## The 6 Critical Knobs

1. `--max-num-batched-tokens` (default: 2048 → try 8192-32768)
2. `--gpu-memory-utilization` (default: 0.90 → try 0.95)
3. `--max-num-seqs` (default: 256/1024 → try 512-2048)
4. `--enable-prefix-caching` (default: OFF → turn ON)
5. `--enable-chunked-prefill` (default: OFF in V0 → verify ON)
6. CPU core allocation (≥ 2 + #GPUs physical cores)

## Exercises

1. **Baseline Deployment**: Start vLLM with defaults
2. **Throughput Optimization**: Apply throughput-focused config
3. **Latency Optimization**: Apply latency-focused config
4. **Benchmark Comparison**: Measure TTFT, TBT, throughput

## Starter Configurations

**Throughput-heavy:**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --enable-chunked-prefill
```

**Latency-sensitive:**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 512 \
    --enable-chunked-prefill
```

## Validation Checkpoints

- [ ] vLLM server starts successfully
- [ ] OpenAI-compatible API responds to requests
- [ ] Throughput config shows higher tokens/s
- [ ] Latency config shows lower TTFT

## Next Steps

Proceed to Lab 5: SGLang Structured Output for constrained generation.
