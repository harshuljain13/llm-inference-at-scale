# Lab 1: Transformer Forward Pass - KV Cache Deep Dive

## The Big Picture

This lab isn't about implementing attention for the sake of it. It's about building the mental model you need to understand why LLM inference is expensive and where the optimization opportunities are.

**The key insight:** LLM inference is a memory bandwidth problem, not a compute problem. By the end of this lab, you'll understand exactly why.

## What You'll Learn

1. **How KV cache actually works** - Not just "it caches K and V" but exactly what bytes are stored and why
2. **Why GQA exists** - The specific tradeoff it makes and why every modern model uses it
3. **The prefill/decode split** - Why these phases have completely different bottlenecks
4. **The memory bandwidth wall** - The hard ceiling on decode speed that no optimization can exceed

## Prerequisites

- Python 3.10+
- PyTorch 2.0+
- Basic understanding of matrix multiplication
- GPU recommended but not required (CPU works for small examples)

## Setup

```bash
pip install torch numpy
```

## Duration

45-60 minutes

## AWS Cost

$0 (runs locally). GPU recommended for timing experiments but not required.

## Files

- `lab_01_kv_cache_deep_dive.py` - Main lab script with 5 experiments
- `solutions.py` - Reference solutions with detailed comments

## The Experiments

### Experiment 1: KV Cache Growth

Watch the KV cache grow token by token during generation. You'll see exactly how much memory each token adds.

### Experiment 2: MHA vs GQA Memory Comparison

Compare memory usage between Multi-Head Attention and Grouped-Query Attention. Key insight: GQA reduces KV cache dramatically but weight size only decreases slightly.

### Experiment 3: Prefill vs Decode Timing

Measure the time difference between processing a prompt (prefill) and generating tokens (decode). You'll see why prefill is compute-bound and decode is memory-bound.

### Experiment 4: Memory Bandwidth Wall

Calculate the theoretical maximum decode speed for different models and GPUs. This is the hard ceiling that no optimization can exceed.

### Experiment 5: KV Cache Memory Budget

Calculate how many concurrent sequences you can serve given VRAM constraints. You'll see how KV cache can exceed model weights at high batch sizes.

## Running the Lab

```bash
python lab_01_kv_cache_deep_dive.py
```

## Expected Output

```
EXPERIMENT 1: KV Cache Growth During Generation
======================================================================
Config: hidden=512, heads=8, kv_heads=2
Head dim: 64
GQA group size: 4

PREFILL (10 tokens):
  K cache shape: [1, 2, 10, 64]
  KV cache size: 5.00 KB

DECODE (one token at a time):
  Step 1: shape [1, 2, 11, 64], size = 5.50 KB
  Step 2: shape [1, 2, 12, 64], size = 6.00 KB
  ...
```

## Validation Checkpoints

- [ ] KV cache grows by exactly 1 token per decode step
- [ ] GQA-4 uses 4× less KV cache than MHA
- [ ] Prefill time scales sub-linearly with prompt length
- [ ] You can calculate max decode speed from memory bandwidth

## Key Numbers to Remember

| Model      | KV Cache per Token (FP16) | Max Decode Speed (A100) |
| ---------- | ------------------------- | ----------------------- |
| Llama 8B   | 128 KB                    | 125 tok/s               |
| Llama 70B  | 320 KB                    | 14 tok/s                |
| Llama 405B | 504 KB                    | 2.5 tok/s               |

## Troubleshooting

| Issue                                 | Solution                                                        |
| ------------------------------------- | --------------------------------------------------------------- |
| Shape mismatch in attention           | Check head_dim = hidden_dim // num_heads                        |
| KV cache not growing                  | Ensure return_cache=True and passing cache between calls        |
| OOM on large sequences                | Reduce batch_size or seq_length                                 |
| Timing experiments show no difference | Need GPU for accurate timing; CPU hides the memory-bound nature |

## Challenge Extensions

1. **Implement RoPE**: Add rotary position embeddings to the attention
2. **Measure actual memory bandwidth**: Use torch.cuda.memory_stats() to track bytes transferred
3. **Implement prefix caching**: Share KV cache across requests with common prefixes

## Next Steps

After completing this lab, proceed to:

- **Module 2**: GPU Memory Engineering - understand the roofline model and memory hierarchy
- **Lab 2**: VRAM Calculation - estimate memory requirements for production models
