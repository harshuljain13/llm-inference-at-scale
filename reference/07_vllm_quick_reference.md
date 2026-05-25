# vLLM Quick Reference

> Complete configuration guide for vLLM - the high-throughput LLM inference engine

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Server Configuration](#server-configuration)
- [Critical Tuning Knobs](#critical-tuning-knobs)
- [Configuration Profiles](#configuration-profiles)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Basic Installation

```bash
pip install vllm
```

### With CUDA 12.1

```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121
```

### From Source (Latest Features)

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

### Docker (Recommended for Production)

```bash
docker pull vllm/vllm-openai:latest
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.1-8B-Instruct
```

---

## Quick Start

### Python API

```python
from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# Configure sampling
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256
)

# Generate
outputs = llm.generate(["Hello, how are you?"], sampling_params)
print(outputs[0].outputs[0].text)
```

### OpenAI-Compatible Server

```bash
# Start server
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

# Query with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Server Configuration

### Essential Arguments

| Argument                   | Description                                  | Default  |
| -------------------------- | -------------------------------------------- | -------- |
| `--model`                  | HuggingFace model ID or path                 | Required |
| `--port`                   | Server port                                  | 8000     |
| `--host`                   | Server host                                  | 0.0.0.0  |
| `--dtype`                  | Data type (auto, float16, bfloat16, float32) | auto     |
| `--tensor-parallel-size`   | Number of GPUs for tensor parallelism        | 1        |
| `--pipeline-parallel-size` | Number of GPUs for pipeline parallelism      | 1        |

### Memory Configuration

| Argument                   | Description                                        | Default     |
| -------------------------- | -------------------------------------------------- | ----------- |
| `--gpu-memory-utilization` | Fraction of GPU memory to use                      | 0.9         |
| `--max-model-len`          | Maximum sequence length                            | Model's max |
| `--kv-cache-dtype`         | KV cache data type (auto, fp8, fp8_e5m2, fp8_e4m3) | auto        |
| `--block-size`             | Token block size for PagedAttention                | 16          |

### Batching Configuration

| Argument                     | Description                   | Default |
| ---------------------------- | ----------------------------- | ------- |
| `--max-num-seqs`             | Maximum concurrent sequences  | 256     |
| `--max-num-batched-tokens`   | Maximum tokens per batch      | None    |
| `--enable-chunked-prefill`   | Enable chunked prefill        | False   |
| `--max-num-partial-prefills` | Max partial prefills per step | 1       |

### Quantization

| Argument         | Description                                      | Default |
| ---------------- | ------------------------------------------------ | ------- |
| `--quantization` | Quantization method (awq, gptq, squeezellm, fp8) | None    |
| `--load-format`  | Model loading format                             | auto    |

---

## Critical Tuning Knobs

### The 6 Knobs That Matter Most

```
┌─────────────────────────────────────────────────────────────────┐
│                    vLLM TUNING HIERARCHY                        │
├─────────────────────────────────────────────────────────────────┤
│  1. gpu-memory-utilization  │  How much VRAM to use            │
│  2. max-num-seqs            │  Concurrent request capacity     │
│  3. max-num-batched-tokens  │  Tokens processed per iteration  │
│  4. enable-chunked-prefill  │  Interleave prefill/decode       │
│  5. tensor-parallel-size    │  Multi-GPU distribution          │
│  6. quantization            │  Memory/quality tradeoff         │
└─────────────────────────────────────────────────────────────────┘
```

### 1. gpu-memory-utilization (0.0 - 1.0)

```bash
# Conservative (leaves room for other processes)
--gpu-memory-utilization 0.8

# Aggressive (maximum throughput)
--gpu-memory-utilization 0.95

# Default
--gpu-memory-utilization 0.9
```

**When to adjust:**

- Lower if OOM errors occur
- Lower if running multiple models
- Higher for dedicated inference servers

### 2. max-num-seqs

```bash
# Low latency (fewer concurrent requests)
--max-num-seqs 32

# High throughput (many concurrent requests)
--max-num-seqs 512

# Default
--max-num-seqs 256
```

**Impact:**

- Higher = more throughput, higher latency
- Lower = lower latency, less throughput
- Memory scales with this value

### 3. max-num-batched-tokens

```bash
# Limit batch size for latency
--max-num-batched-tokens 4096

# Unlimited (let vLLM decide)
# Don't set this flag
```

**When to use:**

- Set when you need predictable latency
- Leave unset for maximum throughput

### 4. enable-chunked-prefill

```bash
# Enable for better latency during high load
--enable-chunked-prefill
--max-num-partial-prefills 1

# With specific chunk size
--enable-chunked-prefill
--num-scheduler-steps 1
```

**Benefits:**

- Reduces latency spikes from long prompts
- Better interleaving of prefill and decode
- Recommended for production

### 5. tensor-parallel-size

```bash
# Single GPU
--tensor-parallel-size 1

# 2 GPUs (e.g., for 70B models)
--tensor-parallel-size 2

# 4 GPUs
--tensor-parallel-size 4

# 8 GPUs (full node)
--tensor-parallel-size 8
```

**Rules:**

- Model size / TP = VRAM per GPU needed
- Use NVLink for best performance
- TP should divide number of attention heads

### 6. quantization

```bash
# AWQ (recommended for quality)
--quantization awq

# GPTQ
--quantization gptq

# FP8 (Hopper GPUs)
--quantization fp8

# No quantization (highest quality)
# Don't set this flag
```

---

## Configuration Profiles

### Profile 1: Latency-Optimized (Real-time Chat)

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 4096 \
    --enable-chunked-prefill \
    --max-num-partial-prefills 1
```

**Use case:** Interactive chatbots, real-time applications
**Expected:** TTFT < 200ms, ITL < 30ms

### Profile 2: Throughput-Optimized (Batch Processing)

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 512 \
    --enable-chunked-prefill
```

**Use case:** Batch inference, offline processing
**Expected:** Maximum tokens/second

### Profile 3: Memory-Constrained (Limited VRAM)

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --gpu-memory-utilization 0.9 \
    --quantization awq \
    --max-model-len 4096 \
    --max-num-seqs 64
```

**Use case:** Smaller GPUs, cost optimization
**Expected:** 2-3x memory reduction

### Profile 4: Multi-GPU Large Model

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 128 \
    --enable-chunked-prefill
```

**Use case:** 70B+ models on multiple GPUs
**Expected:** Linear scaling with NVLink

### Profile 5: Production with Monitoring

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256 \
    --enable-chunked-prefill \
    --disable-log-requests \
    --uvicorn-log-level warning
```

---

## API Reference

### Completions API

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "prompt": "The capital of France is",
        "max_tokens": 50,
        "temperature": 0.7
    }
)
```

### Chat Completions API

```python
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 256,
        "temperature": 0.7,
        "stream": False
    }
)
```

### Streaming

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

### Structured Output (JSON Mode)

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    city: str

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "Generate a person"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": Person.model_json_schema()
            }
        }
    }
)
```

---

## Troubleshooting

### Common Errors

#### OOM (Out of Memory)

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**

1. Reduce `--gpu-memory-utilization`
2. Reduce `--max-num-seqs`
3. Reduce `--max-model-len`
4. Enable quantization
5. Use tensor parallelism

#### Model Too Large

```
ValueError: The model's max seq len (X) is larger than the maximum number of tokens that can be stored in KV cache
```

**Solutions:**

1. Set `--max-model-len` to a smaller value
2. Use quantization
3. Add more GPUs with tensor parallelism

#### Slow First Request

```
First request takes 30+ seconds
```

**Cause:** Model loading and CUDA graph compilation

**Solutions:**

1. Use `--disable-custom-all-reduce` if not using NVLink
2. Pre-warm with a dummy request
3. Use Docker with pre-built images

#### High Latency Under Load

```
Latency increases significantly with concurrent requests
```

**Solutions:**

1. Enable `--enable-chunked-prefill`
2. Reduce `--max-num-seqs`
3. Set `--max-num-batched-tokens`
4. Add more GPUs

### Performance Debugging

#### Check GPU Utilization

```bash
watch -n 0.5 nvidia-smi
```

#### Check vLLM Metrics

```bash
curl http://localhost:8000/metrics
```

#### Key Metrics to Monitor

- `vllm:num_requests_running` - Active requests
- `vllm:num_requests_waiting` - Queued requests
- `vllm:gpu_cache_usage_perc` - KV cache utilization
- `vllm:avg_generation_throughput_toks_per_s` - Throughput

---

## Version Compatibility

| vLLM Version | Key Features            |
| ------------ | ----------------------- |
| 0.4.x        | Stable, PagedAttention  |
| 0.5.x        | Chunked prefill, FP8    |
| 0.6.x        | V1 architecture preview |
| 0.7.x        | Improved multi-GPU      |

---

## Quick Commands Cheat Sheet

```bash
# Basic serve
vllm serve MODEL_NAME

# With quantization
vllm serve MODEL_NAME --quantization awq

# Multi-GPU
vllm serve MODEL_NAME --tensor-parallel-size 4

# Check available models
vllm serve --help

# Benchmark
python -m vllm.entrypoints.openai.api_server --model MODEL_NAME &
python -m vllm.benchmark_serving --model MODEL_NAME
```
