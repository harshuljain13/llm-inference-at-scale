# Lab 6: Tensor Parallelism

## Overview

Scale LLM inference across multiple GPUs using tensor parallelism. Learn to serve 70B+ models that don't fit on a single GPU.

## Learning Objectives

- Configure tensor parallelism in vLLM
- Understand weight distribution across GPUs
- Measure scaling efficiency
- Choose parallelism strategy based on model size

## Prerequisites

- Completed Labs 1-5
- AWS p4d.24xlarge (8× A100) or g5.12xlarge (4× A10G)
- Understanding of NCCL collectives

## Setup

```bash
pip install vllm
```

## Duration

60-90 minutes

## AWS Cost

~$50-100 (p4d.24xlarge for ~90 minutes)

## Exercises

1. **Single GPU Baseline**: Measure throughput on 1 GPU
2. **TP=2 Configuration**: Split model across 2 GPUs
3. **TP=4 Configuration**: Split model across 4 GPUs
4. **TP=8 Configuration**: Full 8-GPU deployment for 70B model
5. **Scaling Efficiency**: Calculate efficiency vs linear scaling

## Configuration Examples

```bash
# TP=2
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 2

# TP=8 for 70B model
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.95
```

## Parallelism Decision Guide

| Model Size | Single GPU | Recommended TP |
| ---------- | ---------- | -------------- |
| < 15B      | 24GB+      | TP=1           |
| 15-40B     | 80GB+      | TP=2-4         |
| 40-80B     | 160GB+     | TP=4-8         |
| > 80B      | 320GB+     | TP=8 + PP      |

## Validation Checkpoints

- [ ] TP=2 shows ~1.8x throughput vs TP=1
- [ ] 70B model loads successfully with TP=8
- [ ] NCCL communication overhead is acceptable
- [ ] GPU utilization is balanced across devices

## Cleanup

```bash
# Terminate expensive multi-GPU instance
aws ec2 terminate-instances --instance-ids <instance-id>
```

## Next Steps

Proceed to Lab 7: Ray Serve Deployment for distributed serving.
