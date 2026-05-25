# LLM Inference Glossary

> Comprehensive terminology reference for LLM inference at scale

---

## A

### Arithmetic Intensity

The ratio of compute operations (FLOPs) to memory operations (bytes transferred). LLM inference is typically **memory-bound** with low arithmetic intensity during the decode phase.

```
Arithmetic Intensity = FLOPs / Bytes Transferred
```

### Attention

The mechanism that allows tokens to "attend" to other tokens in the sequence. Computes weighted combinations of values based on query-key similarity.

### Autoregressive Generation

The process of generating tokens one at a time, where each new token depends on all previously generated tokens. This is why LLM inference is fundamentally sequential.

### AWQ (Activation-aware Weight Quantization)

A quantization method that preserves important weights based on activation patterns. Typically achieves INT4 precision with minimal quality loss.

---

## B

### Batch Size

The number of sequences processed simultaneously. In LLM inference, this is dynamic due to continuous batching.

### BF16 (Brain Float 16)

A 16-bit floating-point format with the same exponent range as FP32 but reduced mantissa precision. Preferred for training and inference on modern GPUs.

### Block

In PagedAttention, a fixed-size chunk of KV cache memory (typically 16 tokens). Enables efficient memory management similar to OS virtual memory.

---

## C

### Chunked Prefill

A technique that breaks long prompt processing into smaller chunks, interleaving with decode operations to reduce latency spikes.

### Continuous Batching

Dynamic batching that adds new requests and removes completed ones without waiting for the entire batch to finish. Key innovation in vLLM.

### Context Length

The maximum number of tokens a model can process in a single forward pass. Includes both prompt and generated tokens.

---

## D

### Decode Phase

The token-by-token generation phase after prefill. Each step generates one token and is memory-bound due to low arithmetic intensity.

### Disaggregated Serving

Architecture that separates prefill and decode into different GPU pools, allowing independent scaling. Used in llm-d.

### Draft Model

A smaller, faster model used in speculative decoding to generate candidate tokens that are verified by the main model.

---

## E

### EAGLE

A speculative decoding variant that uses a lightweight draft head instead of a separate model.

### EOS (End of Sequence)

A special token that signals the model to stop generating.

---

## F

### Flash Attention

An optimized attention implementation that reduces memory usage from O(n²) to O(n) by using tiling and recomputation.

### FLOPs (Floating Point Operations)

A measure of computational work. Used to estimate GPU utilization and compare model complexity.

### FP8 (8-bit Floating Point)

A compact floating-point format supported on Hopper GPUs (H100). Offers 2x memory savings over FP16 with minimal quality loss.

### FP16 (Half Precision)

16-bit floating-point format. Standard for inference, offering good balance of precision and memory efficiency.

---

## G

### GGUF (GPT-Generated Unified Format)

A file format for quantized models, commonly used with llama.cpp for CPU and edge inference.

### GQA (Grouped Query Attention)

An attention variant where multiple query heads share a single key-value head. Reduces KV cache size while maintaining quality. Used in Llama 2 70B and Llama 3.

### GPU Memory Utilization

The fraction of GPU VRAM allocated for model weights, KV cache, and activations. Typically set to 0.9 in vLLM.

---

## H

### HBM (High Bandwidth Memory)

The high-speed memory used in modern GPUs. A100 has 2TB/s bandwidth, H100 has 3.35TB/s.

### Head (Attention Head)

A single attention computation unit. Multi-head attention runs multiple heads in parallel, each learning different patterns.

---

## I

### INT4 / INT8

4-bit and 8-bit integer quantization formats. Reduce memory by 4x/2x compared to FP16.

### Inter-Token Latency (ITL)

The time between generating consecutive tokens during the decode phase. Target: <50ms for interactive applications.

### Iteration

One forward pass through the model. In continuous batching, each iteration may process different numbers of tokens for different sequences.

---

## K

### KServe

A Kubernetes-native model serving framework that provides autoscaling, canary deployments, and inference graphs.

### KV Cache

Storage for key and value tensors from previous tokens, avoiding recomputation during autoregressive generation. Major memory consumer during inference.

```
KV Cache Size = 2 × layers × heads × head_dim × seq_len × batch_size × dtype_bytes
```

---

## L

### Latency

Time from request submission to response completion. Composed of queue time, TTFT, and generation time.

### llm-d

A disaggregated LLM serving system that separates prefill and decode workers for independent scaling.

### LoRA (Low-Rank Adaptation)

A parameter-efficient fine-tuning method. vLLM supports serving multiple LoRA adapters simultaneously.

---

## M

### Medusa

A speculative decoding variant that adds multiple prediction heads to generate several candidate tokens in parallel.

### Memory Bandwidth

The rate at which data can be transferred between GPU memory and compute units. Often the bottleneck in LLM inference.

### MHA (Multi-Head Attention)

Standard attention with separate key-value heads for each query head. Most memory-intensive attention variant.

### MoE (Mixture of Experts)

Architecture where only a subset of parameters (experts) are activated for each token. Enables larger models with similar compute cost.

### MQA (Multi-Query Attention)

Attention variant where all query heads share a single key-value head. Minimizes KV cache but may reduce quality.

---

## N

### NCCL (NVIDIA Collective Communications Library)

Library for multi-GPU communication. Provides AllReduce, AllGather, and other collective operations.

### Neuron SDK

AWS's SDK for running models on Inferentia and Trainium chips.

### NVLink

High-bandwidth interconnect between GPUs. Essential for efficient tensor parallelism.

---

## P

### PagedAttention

vLLM's memory management technique that stores KV cache in non-contiguous blocks, similar to OS virtual memory. Eliminates memory fragmentation.

### Pipeline Parallelism (PP)

Distributing model layers across GPUs. Each GPU processes a subset of layers sequentially.

### Prefill Phase

The initial phase where the model processes the entire prompt in parallel. Compute-bound and typically faster per token than decode.

---

## Q

### Quantization

Reducing the precision of model weights and/or activations to decrease memory usage and increase throughput.

### Queue Time

Time a request spends waiting before processing begins. Should be minimized for interactive applications.

---

## R

### RadixAttention

SGLang's prefix caching mechanism that uses a radix tree to efficiently share KV cache across requests with common prefixes.

### Ray Serve

A scalable model serving library built on Ray. Supports autoscaling and multi-model deployments.

### Roofline Model

A visual model for understanding whether a workload is compute-bound or memory-bound based on arithmetic intensity.

---

## S

### SageMaker

AWS's managed ML platform. Supports LLM inference through Large Model Inference (LMI) containers.

### Sampling

The process of selecting the next token from the model's probability distribution. Includes temperature, top-p, top-k.

### Sequence Length

The total number of tokens in a sequence, including both prompt and generated tokens.

### SGLang

An inference engine optimized for structured generation and complex LLM programs. Features RadixAttention for prefix caching.

### Speculative Decoding

A technique that uses a draft model to generate multiple candidate tokens, then verifies them in parallel with the main model.

---

## T

### Temperature

A sampling parameter that controls randomness. Higher values (>1) increase diversity, lower values (<1) increase determinism.

### Tensor Parallelism (TP)

Distributing individual layers across GPUs. Each GPU holds a slice of every layer and communicates via AllReduce.

### TensorRT-LLM

NVIDIA's optimized inference library. Requires model compilation but achieves highest performance on NVIDIA GPUs.

### Throughput

The rate of token generation, typically measured in tokens per second. Key metric for batch processing.

### Time to First Token (TTFT)

The time from request submission to receiving the first generated token. Dominated by prefill time.

### Token

The basic unit of text processing. Typically 3-4 characters on average for English text.

### Top-k Sampling

Sampling only from the k most likely tokens.

### Top-p (Nucleus) Sampling

Sampling from the smallest set of tokens whose cumulative probability exceeds p.

---

## V

### vLLM

A high-throughput LLM inference engine featuring PagedAttention and continuous batching.

### VRAM (Video RAM)

GPU memory. The primary constraint for LLM inference, determining maximum model size and batch capacity.

---

## W

### Weight

Model parameters learned during training. Stored in GPU memory during inference.

### Weight-Only Quantization

Quantizing only model weights while keeping activations in higher precision. Simpler than full quantization.

---

## Quick Reference: Key Formulas

### Model Memory

```
Model Memory (GB) = Parameters (B) × Bytes per Parameter
                  = 8B × 2 (FP16) = 16 GB
```

### KV Cache Memory

```
KV Cache (GB) = 2 × L × H × D × S × B × dtype / 1e9

Where:
  L = number of layers
  H = number of KV heads
  D = head dimension
  S = sequence length
  B = batch size
  dtype = bytes per element (2 for FP16)
```

### Tokens per Second (Decode)

```
Tokens/sec ≈ Memory Bandwidth / Bytes per Token
           ≈ Memory Bandwidth / (2 × Model Parameters)
```

### GPU Memory Requirement

```
Total VRAM = Model Weights + KV Cache + Activations + Overhead
           ≈ Model Weights × 1.2 + KV Cache
```
