# LLM Inference Cheat Sheet

## Quick Reference for Production LLM Serving

---

## VRAM Calculation Formulas

### Model Weights

```
Weights (GB) = num_params (B) × bytes_per_param

FP32: bytes = 4    → 8B model = 32 GB
FP16: bytes = 2    → 8B model = 16 GB
INT8: bytes = 1    → 8B model = 8 GB
INT4: bytes = 0.5  → 8B model = 4 GB
```

### KV Cache

```
KV Cache (GB) = 2 × layers × kv_heads × head_dim × seq_len × batch × 2 bytes / 1e9

Llama 3.1 8B (batch=1, seq=4096):
= 2 × 32 × 8 × 128 × 4096 × 1 × 2 / 1e9 = 0.54 GB
```

### Total VRAM

```
Total ≈ Weights + KV Cache + Activations (10% of weights) + Overhead (1 GB)
```

---

## vLLM Configuration Quick Reference

### Throughput-Optimized

```bash
--max-num-batched-tokens 16384
--gpu-memory-utilization 0.95
--enable-prefix-caching
--enable-chunked-prefill
```

### Latency-Optimized

```bash
--max-num-batched-tokens 4096
--max-num-seqs 512
--enable-chunked-prefill
```

### The 6 Critical Knobs

| Knob                   | Default  | Throughput | Latency |
| ---------------------- | -------- | ---------- | ------- |
| max-num-batched-tokens | 2048     | 16384      | 4096    |
| gpu-memory-utilization | 0.90     | 0.95       | 0.90    |
| max-num-seqs           | 256/1024 | 2048       | 512     |
| enable-prefix-caching  | OFF      | ON         | ON      |
| enable-chunked-prefill | OFF/ON   | ON         | ON      |
| CPU cores              | varies   | ≥2+GPUs    | ≥2+GPUs |

---

## Engine Selection Guide

```
General serving, fast iteration     → vLLM
Structured output, multi-step       → SGLang
Max throughput on NVIDIA            → TensorRT-LLM
Cost-optimized on AWS               → Inferentia2
```

---

## AWS Instance Selection

| Model Size | Instance            | VRAM   | Cost/hr |
| ---------- | ------------------- | ------ | ------- |
| < 15B FP16 | g5.2xlarge          | 24 GB  | $1.21   |
| < 15B INT4 | g5.xlarge           | 24 GB  | $1.01   |
| 15-40B     | p4d.24xlarge (TP=4) | 320 GB | $32.77  |
| 40-80B     | p4d.24xlarge (TP=8) | 320 GB | $32.77  |
| > 80B      | p5.48xlarge         | 640 GB | $98.32  |
| Cost-opt   | inf2.xlarge         | 32 GB  | $0.76   |

---

## Key Metrics & Targets

| Metric     | Chatbot    | Batch       |
| ---------- | ---------- | ----------- |
| TTFT P95   | < 500ms    | < 5s        |
| TBT P95    | < 50ms     | N/A         |
| E2E P95    | < 10s      | < 60s       |
| Throughput | > 50 tok/s | > 500 tok/s |

---

## Parallelism Decision

```
Model fits 1 GPU?           → TP=1, scale replicas
Model fits 1 node (8 GPU)?  → TP=8
Model needs multiple nodes? → TP=8 + PP=N
```

---

## Quantization Selection

| Method | Memory | Quality   | Use Case           |
| ------ | ------ | --------- | ------------------ |
| FP16   | 2×     | Baseline  | Default            |
| AWQ    | 8×     | <1% loss  | Production INT4    |
| GPTQ   | 8×     | 1-3% loss | Memory-constrained |
| FP8    | 4×     | Minimal   | H100 only          |

---

## Troubleshooting Quick Reference

| Symptom           | Cause          | Fix                             |
| ----------------- | -------------- | ------------------------------- |
| High TTFT         | Prefill queue  | More replicas, chunked prefill  |
| High TBT          | Memory BW      | Smaller batch, quantization     |
| OOM               | KV cache       | Reduce max_num_seqs             |
| Low throughput    | Small batch    | Increase max_num_batched_tokens |
| GPU underutilized | CPU bottleneck | More CPU cores                  |

---

## Common Commands

### Start vLLM Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --port 8000
```

### Test Endpoint

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "prompt": "Hello", "max_tokens": 50}'
```

### Check GPU Memory

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```
