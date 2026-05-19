# Lab 2: VRAM Calculation

## Overview

Learn to calculate VRAM requirements for LLM inference using "napkin math" formulas. This skill is essential for capacity planning and instance selection.

## Learning Objectives

- Calculate model weight memory for different precisions
- Estimate KV cache memory for various batch sizes and sequence lengths
- Predict total VRAM requirements
- Identify compute vs memory bottlenecks

## Prerequisites

- Completed Lab 1 (understanding of KV cache)
- Python 3.10+

## Setup

```bash
pip install torch numpy pandas matplotlib
```

## Duration

30-45 minutes

## AWS Cost

$0 (runs locally)

## Exercises

1. **Model Weights Calculation**: Calculate memory for Llama 3.1 8B and 70B
2. **KV Cache Scaling**: Plot KV cache growth vs sequence length
3. **Batch Size Impact**: Analyze memory vs batch size tradeoffs
4. **Instance Selection**: Match models to AWS instance types

## Key Formulas

```
Model Weights = num_params × bytes_per_param

KV Cache = 2 × num_layers × num_kv_heads × head_dim × seq_length × batch_size × bytes

Total VRAM ≈ Weights + KV Cache + Activations (10% of weights) + Overhead (1GB)
```

## Validation Checkpoints

- [ ] Llama 3.1 8B FP16 weights ≈ 16 GB
- [ ] Llama 3.1 70B FP16 weights ≈ 140 GB
- [ ] KV cache scales linearly with sequence length

## Next Steps

Proceed to Lab 3: Quantization Comparison to learn how quantization reduces memory requirements.
