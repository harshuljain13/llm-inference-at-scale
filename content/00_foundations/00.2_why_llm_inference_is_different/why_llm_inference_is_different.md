# Module 0.2: Why LLM Inference is Different

> You now know what a transformer looks like inside (Module 0.0) and how inference works — tokenization, prefill, decode, KV cache (Module 0.1). This module answers: **why is all of this so expensive?**

---

## Learning Objectives

By the end of this module, you will:

- Understand why LLM inference costs 100x more than traditional ML inference
- Grasp the fundamental difference between traditional ML and autoregressive generation
- Know the two phases of LLM inference (prefill and decode) at a high level
- Understand the memory bandwidth wall that limits decode speed

---

## The Uncomfortable Truth

Here's what nobody tells you when you start working on LLM inference:

**Traditional ML inference is a solved problem. LLM inference is not.**

| Aspect   | Traditional ML       | LLM Inference                   |
| -------- | -------------------- | ------------------------------- |
| Latency  | Predictable (5-20ms) | Unpredictable (100ms-10s)       |
| Memory   | Fixed per request    | Grows during request            |
| Batching | Trivial              | Requires continuous batching    |
| Scaling  | Linear with GPUs     | Sub-linear, communication-bound |
| Cost     | $0.001 per request   | $0.01-0.10 per request          |

The difference isn't 2x or 5x—it's 100x. And the reasons are fundamental, not incidental.

---

## Why LLMs Are Fundamentally Different

### The Sequential Nature of Text Generation

The core difference comes down to one word: **autoregressive**.

In traditional ML, inference is a single forward pass. You feed an image into ResNet, the data flows through the network once, and you get your classification. Done.

![Traditional ML Inference](images/traditional_ml.png)
*Traditional ML inference: one input, one forward pass, one output. Time is fixed, memory is constant, and batching is trivial.*

LLMs work completely differently. When you ask "What is the capital of France?", the model doesn't produce the answer in one shot. It generates one token at a time: "The" → "capital" → "of" → "France" → "is" → "Paris". Each token requires a separate forward pass through the entire model.

![LLM Inference](images/llm_inference.png)
*LLM inference: each output token requires its own forward pass. Token N cannot be generated until tokens 1 through N-1 exist.*

This isn't a limitation to be engineered away—it's how autoregressive language models work by design. The probability distribution for token 5 depends on what tokens 1-4 actually are.


### You Read the Entire Model for Every Token

In traditional ML, you load the model once and run millions of requests through it — the weights stay in GPU memory and get reused efficiently. With LLMs, something counterintuitive happens: even though the weights never change, the GPU must physically read them from memory on every single token generation. Here's why that's devastating:

```
Llama 3.1 8B generating 100 tokens:

Each token generation requires a full forward pass through the model.
A forward pass means reading ALL 8 billion parameters from memory.

  - Token 1:   Read 16 GB of weights
  - Token 2:   Read 16 GB of weights again
  - Token 3:   Read 16 GB of weights again
  - ...
  - Token 100: Read 16 GB of weights again

  Total memory reads: 16 GB × 100 = 1.6 TB
```

Neural networks don't "remember" their weights between operations. Every matrix multiplication requires loading the weight matrix from GPU memory (HBM) into the compute units. Generate 100 tokens, load the weights 100 times.

### The Memory Bandwidth Wall

If the GPU must read 16 GB of weights for every token, the speed of generation is fundamentally limited by how fast the GPU can read from its own memory. This isn't a software bottleneck you can optimize away — it's a hardware constraint called the memory bandwidth wall:

```
A100 memory bandwidth: 2 TB/s
Model size (FP16):     16 GB
Time to read model:    16 GB / 2 TB/s = 8 ms

Maximum decode speed = 1 token / 8 ms = 125 tokens/second
```

**This is a hard ceiling.** No software optimization can exceed it. The only ways past this wall are:
- Reduce model size (quantization)
- Increase memory bandwidth (better hardware or more GPUs)
- Generate multiple tokens per weight read (speculative decoding)

---

## The Two Phases: Prefill and Decode

Not all parts of inference are equally slow. When you send a prompt, the model first processes all your input tokens in parallel (fast), then generates the response one token at a time (slow). These two phases have completely different performance characteristics, and understanding this split is the key to every optimization in the rest of this book:

| Phase | What Happens | Bottleneck | When It Runs |
|-------|--------------|------------|--------------|
| **Prefill** | Process entire prompt at once | Compute (TFLOPS) | Once per request |
| **Decode** | Generate one token at a time | Memory bandwidth (TB/s) | Once per output token |

### Prefill: The Compute-Bound Phase

During prefill, all prompt tokens are processed in parallel through the model. This is the "reading comprehension" step — the model digests your entire input and builds the KV cache. Because many tokens are processed simultaneously, the GPU's compute units stay busy.

**Why prefill is compute-bound:** The GPU sees large matrices (e.g., [1000, 4096] × [4096, 4096] for a 1000-token prompt). There's enough parallel work to keep the compute units busy. The bottleneck is how many FLOPs the GPU can execute per second.

### Decode: The Memory-Bound Phase

During decode, the model generates one token at a time. Each token requires reading the full model from memory, but produces just a single output. This is the bottleneck that dominates most real-world LLM serving costs.

**Why decode is memory-bound:** The GPU sees tiny matrices (e.g., [1, 4096] × [4096, 4096]). There's not enough parallel work to keep the compute units busy. The GPU spends most of its time waiting for data to arrive from memory.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE DECODE BOTTLENECK                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   What the GPU CAN do:     312 TFLOPS (312 trillion ops/second)     │
│   What the GPU DOES do:    ~16 GFLOPS (limited by memory bandwidth) │
│                                                                     │
│   GPU Utilization during decode: ~2 TFLOP/s sustained vs 312 TFLOP/s peak ≈ 0.6%  │
│                                                                     │
│   The GPU is 99.995% IDLE during decode!                            │
│                                                                     │
│   This is why:                                                      │
│   • Decode is slow despite "less work"                              │
│   • Faster GPUs don't help much (memory bandwidth is similar)       │
│   • Batching is critical (amortize weight reads across requests)    │
│   • Quantization helps (smaller weights = faster reads)             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Visualizing the Parallelism Difference

```
                    PREFILL: Embarrassingly Parallel
    
    GPU sees large matrices → High utilization
    
    ┌─────────────────────────────────────────────────────────────┐
    │  Processing 1000 tokens simultaneously:                     │
    │                                                             │
    │  Q matrix: [1000, 4096]    K matrix: [1000, 4096]          │
    │  ┌─────────────────────┐   ┌─────────────────────┐         │
    │  │█████████████████████│   │█████████████████████│         │
    │  │█████████████████████│   │█████████████████████│         │
    │  │█████████████████████│   │█████████████████████│         │
    │  │█████████████████████│ × │█████████████████████│         │
    │  │█████████████████████│   │█████████████████████│         │
    │  │█████████████████████│   │█████████████████████│         │
    │  └─────────────────────┘   └─────────────────────┘         │
    │                                                             │
    │  → Millions of multiply-adds happening in parallel          │
    │  → GPU cores fully utilized                                 │
    │  → Compute-bound: limited by TFLOPS                         │
    └─────────────────────────────────────────────────────────────┘

                    DECODE: Fundamentally Sequential
    
    GPU sees tiny vectors → Low utilization
    
    ┌─────────────────────────────────────────────────────────────┐
    │  Processing 1 token at a time:                              │
    │                                                             │
    │  Q vector: [1, 4096]       K matrix: [4096, seq_len]       │
    │  ┌─────────────────────┐   ┌─────────────────────┐         │
    │  │█                    │   │█████████████████████│         │
    │  └─────────────────────┘ × │█████████████████████│         │
    │   (just one row!)          │█████████████████████│         │
    │                            │█████████████████████│         │
    │                            └─────────────────────┘         │
    │                                                             │
    │  → Most GPU cores sit idle                                  │
    │  → Waiting for memory reads                                 │
    │  → Memory-bound: limited by TB/s                            │
    └─────────────────────────────────────────────────────────────┘
```

### The Fundamental Asymmetry

Here's the counterintuitive part:

```
Example: 1000-token prompt, generate 100 tokens

Prefill:
  • Process 1000 tokens in ONE forward pass
  • Time: ~50ms (compute-bound)

Decode:
  • Process 1 token per forward pass × 100 passes
  • Time: 100 × 8ms = 800ms (memory-bound)

Total: 850ms
  • Prefill: 6% of time (processed 1000 tokens)
  • Decode: 94% of time (processed 100 tokens)
```

**Decode dominates even though it processes 10× fewer tokens.** This asymmetry drives everything in LLM inference optimization.

---

## The Roofline Model

The roofline model is a visual tool that explains why prefill and decode behave so differently on the same hardware. It plots achievable performance against computational intensity (how much math you do per byte of memory read). Workloads that read a lot of data but do little math sit on the left (memory-bound); workloads that do heavy math on small data sit on the right (compute-bound).

![Roofline Model for LLM Inference on A100](images/roofline_model_a100.png)
*The roofline model shows why prefill and decode have fundamentally different bottlenecks. Decode sits deep in the memory-bound region, while prefill operates in the compute-bound region.*

**Key insight:** Workloads left of the ridge point (156 FLOPs/byte on A100) are memory-bound. Workloads to the right are compute-bound. Decode has an arithmetic intensity of ~1 FLOP/byte; prefill has ~1000 FLOPs/byte.

This is why **batching helps decode**—processing multiple requests together increases arithmetic intensity, moving you up the diagonal toward better efficiency.

---

## Key Takeaways

1. **LLM inference is fundamentally sequential** — each token depends on all previous tokens

2. **Two phases, two bottlenecks:**
   - Prefill: compute-bound, parallel, efficient
   - Decode: memory-bound, sequential, inefficient

3. **Memory bandwidth is the wall** — decode speed is limited by `model_size / bandwidth`

4. **100x more expensive than traditional ML** — this isn't going away with better software

5. **Different optimizations for different phases** — prefill needs compute, decode needs bandwidth

---

## What's Next

You now have the complete foundation: what the machine is (0.0), how it runs (0.1), and why it's hard (0.2). The rest of this book is about solving these problems:

- **Chapter 01: GPU Fundamentals** — Memory hierarchy, roofline analysis, and hardware selection
- **Chapter 02: Attention and KV Cache** — PagedAttention, cache compression, memory management
- **Chapter 03: Optimization** — Quantization, continuous batching, speculative decoding
- **Chapters 04-07** — Engines, scaling, serving, and operations

---

## References

1. Vaswani et al. "Attention Is All You Need" (2017) — The original transformer paper
2. Pope et al. "Efficiently Scaling Transformer Inference" (2022) — Google's analysis of inference scaling
3. Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023) — vLLM
