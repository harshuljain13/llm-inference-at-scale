# Beyond GPUs: Custom Silicon for LLM Inference

GPUs dominate LLM inference today. Not because they are optimal for the workload, but because decades of ecosystem investment in CUDA, cuDNN, and frameworks like PyTorch created an unassailable software moat. Every optimization technique in this book, from FlashAttention to continuous batching, exists because researchers had to work around GPU architectural limitations rather than redesign the hardware itself.

But a new generation of custom silicon is challenging this dominance. Companies like Groq, Cerebras, AWS, and Meta have built chips purpose-designed for the specific computational patterns of neural network inference. Their approach is radical: instead of adapting software to general-purpose hardware, they adapt hardware to the specific workload. The results are striking. Groq's Language Processing Unit delivers 300+ tokens per second on Llama 2 70B. Cerebras' Wafer-Scale Engine achieves 2500 tokens per second on models up to 400B parameters. These numbers represent 10-50x improvements over NVIDIA's best GPUs on decode-heavy workloads.

This module surveys the custom silicon landscape for LLM inference. You will understand why these chips exist, how they achieve their performance advantages, where they fall short, and when deploying on custom silicon makes economic sense versus staying on GPUs.

From Module 01.1, you know the memory bandwidth wall limits decode throughput to approximately `bandwidth / model_size` tokens per second on a GPU. An H100 with 3.35 TB/s of HBM3 bandwidth serving a 70B model (140 GB in FP16) achieves roughly 24 tokens per second per sequence during autoregressive decode. This is a hard physical limit: each token generation requires reading every model parameter once. Custom silicon attacks this constraint directly, either by eliminating off-chip memory entirely (Groq), providing orders-of-magnitude more on-chip bandwidth (Cerebras), or optimizing the memory hierarchy specifically for transformer access patterns (Trainium).

## Why Custom Silicon Exists: The GPU Efficiency Gap

A GPU is a general-purpose parallel processor. It handles graphics rendering, scientific simulation, cryptocurrency mining, and neural network computation on the same transistor budget. This generality comes at a cost: silicon area dedicated to features irrelevant to LLM inference.

Consider what an H100 GPU actually contains:

- **Streaming Multiprocessors (SMs)**: 132 SMs with FP32/FP64/INT units, tensor cores, and register files
- **L1/L2 Cache Hierarchy**: 256 KB L1 per SM, 50 MB shared L2
- **HBM3 Memory Controller**: 6 stacks, 80 GB total, 3.35 TB/s bandwidth
- **NVLink/PCIe interfaces**: Inter-GPU and host communication
- **Graphics pipeline**: Rasterization, ray tracing, texture units (unused for inference)
- **Warp schedulers**: Thread-level parallelism management for arbitrary kernels

For LLM inference specifically, the workload is remarkably predictable:

1. **Prefill phase**: Large matrix multiplications (compute-bound, high arithmetic intensity)
2. **Decode phase**: Repeated matrix-vector products (memory-bound, low arithmetic intensity)
3. **Attention**: KV cache reads with softmax reduction (memory-bound)
4. **Communication**: All-reduce for tensor parallelism (network-bound)

The decode phase, which dominates inference latency for interactive applications, uses less than 1% of the GPU's peak FLOPS. The chip is waiting for memory reads 99% of the time. This is the efficiency gap: you pay for 989 TFLOPS of FP16 compute but use 10 TFLOPS during decode.

Custom silicon closes this gap by allocating transistor budget differently:

| Resource | GPU (H100) | Inference ASIC (ideal) |
|----------|-----------|----------------------|
| Compute units | 989 TFLOPS | 50-200 TFLOPS (sufficient) |
| On-chip SRAM | 50 MB L2 | 200 MB - 44 GB |
| Memory bandwidth | 3.35 TB/s (HBM) | 21-200 TB/s (on-chip) |
| Power efficiency | ~3 TFLOPS/W | 10-50 TFLOPS/W |
| Flexibility | Any workload | Transformer-specific |

The tradeoff is explicit: sacrifice generality to gain 10-50x efficiency on the specific workload that matters. This is not a new idea. Google's TPU (2016) proved the concept for training. What is new is applying this philosophy to the autoregressive decode bottleneck.

### The Memory Bandwidth Arithmetic

To understand why custom silicon achieves such dramatic speedups, trace the arithmetic for decode on different architectures.

For a 70B parameter model in FP16 (140 GB):

**NVIDIA H100 (HBM3, 3.35 TB/s):**
```
Decode throughput = bandwidth / model_size
                  = 3,350 GB/s / 140 GB
                  = 23.9 tokens/second/sequence
```

**Groq LPU (on-chip SRAM, ~80 TB/s internal bandwidth):**
```
Decode throughput = bandwidth / model_size
                  = 80,000 GB/s / 140 GB
                  = 571 tokens/second/sequence (theoretical)
                  = ~300 tokens/second (measured, with overhead)
```

**Cerebras WSE-3 (44 GB on-chip SRAM, 21 PB/s):**
```
Decode throughput = bandwidth / model_size
                  = 21,000,000 GB/s / 140 GB
                  = 150,000 tokens/second (theoretical)
                  = ~2,500 tokens/second (measured, with routing overhead)
```

The gap between theoretical and measured reflects routing overhead, synchronization costs, and the fact that real transformer computation is not a single monolithic memory read. But even at 5-20% efficiency, these architectures dramatically outperform GPUs on decode.

### Why Now: The Inference Cost Inflection

Custom silicon for inference became economically viable because of a market shift. In 2022, inference was a small fraction of AI compute spend. By 2025, inference exceeded training in total GPU-hours consumed globally. This shift changes the economics:

- **Training**: Happens once per model. Cost amortized over the model's lifetime. GPU flexibility matters because architectures change rapidly.
- **Inference**: Happens billions of times per day. Cost scales linearly with users. Efficiency directly impacts unit economics.

When inference dominates spend, even a 2x efficiency improvement justifies building custom silicon. When the improvement is 10-50x, the investment case is overwhelming for high-volume deployments.


## Groq LPU: Deterministic Inference at SRAM Speed

Groq's Language Processing Unit (LPU) represents the most radical departure from GPU architecture in the inference silicon landscape. Founded by Jonathan Ross (designer of Google's first TPU), Groq built a chip that eliminates HBM entirely, stores the entire model in on-chip SRAM, and executes inference with deterministic, cycle-accurate timing.

In December 2025, NVIDIA announced the acquisition of Groq, signaling that even the GPU incumbent recognizes the value of purpose-built inference hardware. The acquisition gives NVIDIA access to Groq's deterministic execution model and SRAM-first architecture for integration into future inference products.

### Architecture: No HBM, No Bottleneck

The Groq LPU architecture is built on a single principle: if the memory bandwidth wall is the bottleneck, eliminate off-chip memory entirely.

**Key architectural decisions:**

1. **All-SRAM memory**: Each LPU chip contains 230 MB of on-chip SRAM. No HBM, no DRAM. Every byte the chip reads comes from SRAM running at register-file speeds.

2. **Deterministic execution**: Unlike GPUs where execution timing depends on cache hits, warp scheduling, and memory controller arbitrage, the LPU executes every instruction in a fixed number of cycles. You can predict exactly when each computation completes.

3. **Temporal Streaming Architecture (TSP)**: Instructions flow through the chip in a predetermined schedule. There is no instruction fetch, no branch prediction, no speculative execution. The compiler pre-computes the entire execution schedule.

4. **Software-defined networking**: Multiple LPU chips connect via a deterministic network fabric. The compiler schedules inter-chip communication at compile time, eliminating runtime coordination overhead.

**Why deterministic execution matters for inference:**

Autoregressive decode is inherently sequential: token N+1 depends on token N. On a GPU, each token generation involves:
- Launching CUDA kernels (microseconds of overhead)
- Waiting for memory controller to service requests (variable latency)
- Synchronizing across warps (barrier overhead)
- All-reduce across GPUs for tensor parallelism (network jitter)

These overheads compound. A single H100 token generation takes 40-50ms for a 70B model, but only 30% of that time is useful computation. The rest is overhead.

On the Groq LPU, token generation is a single deterministic pipeline flush. No kernel launches, no variable-latency memory, no synchronization. The overhead approaches zero because the entire execution is pre-scheduled.

### Performance Characteristics

Groq published benchmarks on their GroqCloud inference API (publicly verifiable via the API):

| Model | Tokens/second (output) | Latency to first token |
|-------|----------------------|----------------------|
| Llama 2 70B | 300+ tok/s | <100ms |
| Mixtral 8x7B | 500+ tok/s | <80ms |
| Llama 3.1 8B | 1000+ tok/s | <50ms |
| Llama 3.1 70B | 250+ tok/s | <150ms |

For comparison, an H100 serving Llama 2 70B with vLLM and continuous batching achieves approximately 20-30 tokens per second per sequence during decode. Groq achieves 10x this rate.

**The prefill tradeoff:**

Groq's architecture excels at decode (memory-bound) but is less advantageous for prefill (compute-bound). During prefill, GPUs can utilize their massive FLOPS (989 TFLOPS on H100) effectively because the arithmetic intensity is high. Groq's lower compute density means prefill is not dramatically faster than GPU, but decode is where users experience latency, making this tradeoff favorable for interactive applications.

### Scaling: The Multi-Chip Challenge

A single Groq LPU chip has 230 MB of SRAM. A 70B model in FP16 requires 140 GB. This means serving a 70B model requires approximately 600 LPU chips working together with the model sharded across them.

Groq addresses this with their deterministic interconnect:

1. **Model parallelism at compile time**: The compiler partitions the model across chips and pre-schedules all inter-chip transfers.
2. **No runtime coordination**: Because execution is deterministic, chips know exactly when data will arrive from neighbors. No handshaking, no flow control.
3. **GroqRack**: A single rack contains 576 LPUs with a fully connected internal fabric. One rack serves one 70B model instance.

The economic question is whether the cost of 600 chips (with lower per-chip cost than a GPU but higher chip count) beats 8 H100s for the same model. At high utilization and high query volume, Groq's throughput advantage means lower cost per token despite higher hardware cost per deployment.

### Limitations

1. **Model size ceiling**: Total SRAM across a rack limits the maximum model size. Very large models (400B+) require multiple racks.
2. **No fine-tuning on-device**: The LPU is inference-only. Training and fine-tuning happen on GPUs, then models are compiled for LPU deployment.
3. **Compilation overhead**: Each model requires a custom compilation pass that maps the computation graph onto the deterministic schedule. New model architectures may require compiler updates.
4. **Batch size constraints**: The deterministic schedule is optimized for specific batch sizes. Dynamic batching (continuous batching with variable sequence lengths) is harder to implement than on GPUs.
5. **NVIDIA acquisition**: Post-acquisition, Groq's standalone roadmap is uncertain. The technology may be integrated into future NVIDIA products rather than remaining an independent platform.


## Cerebras WSE-3: The Wafer-Scale Approach

Cerebras took the opposite approach to the chip-scaling problem. Instead of building small, efficient chips and connecting many together, they build one enormous chip: an entire silicon wafer as a single processor. The Wafer-Scale Engine 3 (WSE-3) is the largest chip ever built, containing 4 trillion transistors on a single 300mm wafer.

### Architecture: One Wafer, One Chip

The WSE-3 specifications are staggering in their departure from conventional chip design:

| Specification | WSE-3 | H100 (for comparison) |
|--------------|-------|----------------------|
| Transistors | 4 trillion | 80 billion |
| Cores | 900,000 | 16,896 CUDA + 528 Tensor |
| On-chip SRAM | 44 GB | 50 MB (L2) |
| Memory bandwidth | 21 PB/s (on-chip) | 3.35 TB/s (HBM) |
| Die area | 46,225 mm² | 814 mm² |
| Interconnect | On-wafer fabric | NVLink (inter-chip) |
| Process node | TSMC 5nm | TSMC 4nm |

The critical number is memory bandwidth: 21 PB/s versus 3.35 TB/s. That is a 6,000x advantage in raw bandwidth. Even accounting for the fact that not all bandwidth is usable simultaneously (due to routing topology and access patterns), the effective bandwidth advantage for inference workloads is 100-1000x over a GPU.

### How Wafer-Scale Inference Works

The WSE-3 distributes model weights across its 44 GB of on-chip SRAM. Each of the 900,000 cores has local SRAM and communicates with neighbors via a 2D mesh interconnect fabric woven directly into the silicon.

**Inference execution pattern:**

1. **Weight distribution**: Model layers are mapped spatially across the wafer. Each core holds a small tile of weights.
2. **Activation flow**: Input activations stream across the wafer from left to right (conceptually), encountering each layer's weights in sequence.
3. **Dataflow execution**: The chip operates in a dataflow paradigm. Cores fire when their input data arrives, no central scheduler required.
4. **Pipelining**: Multiple tokens can be in-flight simultaneously at different stages of the network, achieving pipeline parallelism without explicit coordination.

This approach eliminates three bottlenecks simultaneously:
- **Memory bandwidth**: Weights are local to compute, no off-chip reads
- **Communication latency**: Inter-layer data movement is on-chip (nanoseconds, not microseconds)
- **Synchronization**: Dataflow execution eliminates barriers and kernel launch overhead

### Performance: Inference at Scale

Cerebras published inference benchmarks through their Cerebras Inference API (publicly accessible):

| Model | Tokens/second (output) | Context window |
|-------|----------------------|----------------|
| Llama 3.1 70B | 2,100 tok/s | 8K |
| Llama 3.1 8B | 4,000+ tok/s | 8K |
| Llama 3.3 70B | 2,500 tok/s | 8K |

These numbers are measured on the CS-3 system (the server housing a WSE-3 chip). For context, achieving 2,500 tok/s on a 70B model with H100 GPUs would require approximately 80-100 GPUs running with perfect continuous batching at maximum occupancy, serving many concurrent sequences. Cerebras achieves comparable aggregate throughput on a single system for individual sequences with much lower latency.

### The Wafer Yield Problem (And How They Solved It)

The obvious question: how do you manufacture a chip the size of an entire wafer? Conventional chips are small (reticle-sized, ~800 mm²) precisely because defects in silicon manufacturing are random. A larger chip has more area for defects to land, reducing yield toward zero.

Cerebras' solutions:

1. **Redundant cores**: The wafer contains more cores than specified. Defective cores are disabled at test time and their workload redistributed to neighbors.
2. **Redundant interconnect**: The mesh fabric has multiple paths between any two points. Failed links are routed around.
3. **Coarse granularity**: Each core is simple (not a complex CPU), so losing a few cores has minimal impact on total compute.
4. **Custom packaging**: The wafer sits on a custom interconnect substrate with thousands of I/O connections for power delivery and external memory attachment.

This approach works because inference is embarrassingly parallel at the core level. Losing 1% of cores reduces throughput by 1%, not functionality.

### The CS-3 System

A WSE-3 chip alone cannot serve large models. The 44 GB of on-chip SRAM fits models up to approximately 20B parameters in FP16 without external memory. For larger models, the CS-3 system attaches external MemoryX units:

- **MemoryX**: High-bandwidth external memory modules that stream weights to the wafer
- **SwarmX**: Networking fabric connecting multiple CS-3 systems for model parallelism
- **Weight streaming**: For models larger than on-chip SRAM, weights stream from MemoryX through the wafer at high bandwidth

For models that fit entirely on-chip (up to ~20B in FP16, or ~70B in INT4/INT8 with quantization), the full bandwidth advantage applies. For larger models, the weight-streaming approach still outperforms GPU HBM but at reduced advantage.

### Limitations

1. **Cost per system**: A CS-3 system costs millions of dollars. The economics only work at very high utilization.
2. **Model flexibility**: Switching between models requires redistributing weights across the wafer. Not as fast as GPU model loading.
3. **Long context**: The 44 GB on-chip SRAM must store both weights AND KV cache. Long sequences reduce the model size that fits on-chip.
4. **Ecosystem**: Limited framework support compared to CUDA. Cerebras provides their own SDK and compiler stack.
5. **Availability**: CS-3 systems are available through Cerebras Inference API (cloud) or as on-premise hardware. Not available on major cloud providers.


## AWS Trainium and Inferentia: Cloud-Native Custom Silicon

Amazon Web Services took a different approach to custom silicon: build chips optimized for their own cloud infrastructure and offer them as managed services at dramatically lower cost than GPU instances. Unlike Groq and Cerebras, which sell performance, AWS sells cost reduction with acceptable performance.

### Inferentia2: The Inference Chip

AWS Inferentia2 (launched 2022) is designed specifically for inference workloads on AWS:

**Architecture highlights:**

- **NeuronCores**: Custom matrix multiplication engines optimized for transformer attention patterns
- **32 GB HBM per chip**: Unlike Groq's SRAM-only approach, Inferentia uses HBM but with a custom memory controller optimized for sequential weight reads
- **Hardware support for dynamic shapes**: Unlike many ASICs, Inferentia handles variable sequence lengths without padding waste
- **2 chips per Inf2 instance**: Each inf2.xlarge instance provides 2 Inferentia2 chips

**Performance characteristics:**

| Instance | Model | Throughput | Cost/1M tokens |
|----------|-------|-----------|---------------|
| inf2.48xlarge | Llama 2 70B | ~40 tok/s | ~$0.20 |
| inf2.24xlarge | Llama 2 13B | ~100 tok/s | ~$0.05 |
| p5.48xlarge (H100) | Llama 2 70B | ~30 tok/s | ~$0.60 |

The value proposition is clear: comparable throughput at 50-70% lower cost per token. The tradeoff is less flexibility (limited operator support) and higher latency to production (compilation through the Neuron SDK).

### Trainium2: Training and Inference Convergence

AWS Trainium2 (2024) blurs the line between training and inference chips:

- **Designed for both workloads**: Unlike Inferentia (inference-only), Trainium2 handles training, fine-tuning, and inference
- **UltraServer configuration**: 64 Trainium2 chips per server with custom high-bandwidth interconnect
- **NeuronLink**: Chip-to-chip communication fabric (analogous to NVLink) enabling tensor and pipeline parallelism
- **FP8 support**: Native support for FP8 formats (E4M3, E5M2) matching H100 transformer engine capabilities

**Key differentiator: Neuron SDK**

The Neuron SDK is AWS's answer to CUDA. It provides:

1. **PyTorch integration**: `torch_neuronx` provides a near-drop-in replacement for CUDA tensors
2. **Transformers NeuronX**: Pre-optimized transformer inference kernels (analogous to TensorRT-LLM)
3. **Continuous batching**: Built-in support for dynamic batching without external orchestrators
4. **SVD compression**: Hardware-accelerated Singular Value Decomposition for model compression, reducing memory footprint by 30-50% with minimal quality loss

```python
# Example: Deploying on Inferentia2 with Neuron SDK
import torch_neuronx
from transformers_neuronx import LlamaForSampling

# Compile model for Neuron hardware
model = LlamaForSampling.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    batch_size=4,
    tp_degree=24,  # tensor parallelism across 24 NeuronCores
    amp='bf16'
)
model.to_neuron()

# Inference runs on Inferentia2 hardware
output = model.sample(
    input_ids,
    sequence_length=2048,
    top_k=50
)
```

### The Cloud Lock-In Calculation

AWS custom silicon is only available on AWS. This creates a strategic consideration:

**Arguments for Trainium/Inferentia:**
- 50%+ cost reduction on inference at scale
- Deep integration with SageMaker endpoints (auto-scaling, monitoring, A/B testing)
- No capital expenditure on hardware
- AWS handles chip failures, upgrades, and capacity planning

**Arguments against:**
- Neuron SDK has fewer supported operations than CUDA (some models require operator decomposition)
- Compilation can take hours for large models (versus minutes for TensorRT)
- Portability: code written for Neuron does not run on GPU without modification
- New model architectures may not be supported for weeks/months after release

For organizations already committed to AWS and running inference at scale (millions of tokens per day), the cost savings typically justify the ecosystem lock-in. For organizations requiring multi-cloud or rapid model iteration, GPUs remain more practical.

## Meta MTIA: Internal Custom Silicon

Meta (formerly Facebook) developed the Meta Training and Inference Accelerator (MTIA) for their own internal workloads. Unlike the other chips in this module, MTIA is not available externally. It is included here because it represents the strategic direction of the largest AI deployer in the world.

### Why Meta Built Custom Silicon

Meta's inference workload is unique in several ways:

1. **Scale**: Meta serves recommendations, ads ranking, content understanding, and generative AI to 3+ billion daily active users
2. **Diversity**: Workloads range from small ranking models (millions of parameters) to large generative models (70B+)
3. **Latency requirements**: Ranking models must return in single-digit milliseconds for real-time ad serving
4. **Cost sensitivity**: At Meta's scale, even 10% efficiency improvement saves hundreds of millions of dollars annually

### MTIA Architecture

MTIA v1 (announced 2023) was designed for ranking and recommendation inference:

- **Dense compute**: Optimized for small-to-medium matrix multiplications (typical of recommendation models)
- **High memory bandwidth per FLOP**: Recommendation models are memory-bound (like LLM decode), so MTIA prioritizes bandwidth over raw compute
- **On-chip SRAM**: Large local SRAM for embedding table lookups (the primary bottleneck in recommendation models)
- **INT8/INT4 native**: Recommendation models tolerate aggressive quantization

MTIA v2 (2024) extends to generative AI:

- **Larger model support**: Designed to handle Meta's Llama models internally
- **Increased compute density**: More tensor cores for prefill-heavy workloads
- **Custom attention hardware**: Dedicated silicon for scaled-dot-product attention with KV cache management

### Lessons from Meta's Approach

Meta's custom silicon strategy offers insights for the broader industry:

1. **Start with your highest-volume workload**: Meta built MTIA for ranking first (billions of inferences per second), not for the largest models
2. **Complement, don't replace GPUs**: Meta still uses NVIDIA GPUs for training and for workloads where flexibility matters. MTIA handles the predictable, high-volume tail.
3. **Internal deployment de-risks**: By deploying internally first, Meta validates the hardware without betting customer-facing products on unproven silicon
4. **Vertical integration**: Owning the chip, the framework (PyTorch), and the deployment platform gives Meta optimization opportunities unavailable to companies using off-the-shelf hardware


## Head-to-Head Comparison

The following table compares the four custom silicon approaches against the GPU baseline across dimensions that matter for production inference deployments:

| Dimension | NVIDIA H100 (GPU) | Groq LPU | Cerebras WSE-3 | AWS Inferentia2 | Meta MTIA |
|-----------|-------------------|----------|----------------|-----------------|-----------|
| **Memory bandwidth** | 3.35 TB/s (HBM3) | ~80 TB/s (SRAM) | 21 PB/s (on-chip) | ~2.4 TB/s (HBM) | Not disclosed |
| **On-chip memory** | 50 MB L2 | 230 MB SRAM | 44 GB SRAM | 32 GB HBM | Not disclosed |
| **Decode tok/s (70B)** | 20-30 | 250-300 | 2,100-2,500 | 35-45 | Internal only |
| **Prefill advantage** | Baseline | Moderate | Large | Comparable | Unknown |
| **Cost per M tokens** | ~$0.60 | ~$0.30 (API) | ~$0.20 (API) | ~$0.20 | Internal |
| **Ecosystem maturity** | Excellent (CUDA) | Limited (GroqWare) | Limited (Cerebras SDK) | Good (Neuron SDK) | Internal (PyTorch) |
| **Model support** | Universal | Llama, Mixtral, etc. | Llama, custom | Major HF models | Internal models |
| **Availability** | All clouds + on-prem | GroqCloud API | Cerebras API + on-prem | AWS only | Meta internal |
| **Flexibility** | Any workload | Inference only | Training + inference | Training + inference | Inference primarily |
| **Max model size** | Unlimited (multi-GPU) | ~600 GB (rack) | ~2 TB (weight streaming) | ~140 GB (instance) | Unknown |
| **Dynamic batching** | Excellent (vLLM) | Limited | Supported | Supported (Neuron) | Supported |
| **Time to deploy new model** | Hours | Days-weeks | Days-weeks | Days-weeks | Weeks-months |

### Reading the Comparison

Several patterns emerge:

**Decode speed**: Custom silicon dominates. Every alternative beats GPU on raw decode throughput because they all address the memory bandwidth wall. Cerebras leads by 100x due to sheer on-chip bandwidth, Groq by 10x through deterministic execution, and Inferentia achieves modest gains through cost optimization rather than raw speed.

**Ecosystem**: GPU dominates overwhelmingly. Every new model architecture works on CUDA first. Custom silicon requires porting, compilation, and often operator-level debugging. The gap between "model released" and "model available on custom silicon" ranges from days (Inferentia for popular models) to weeks (Groq, Cerebras for new architectures).

**Economics**: The comparison depends heavily on utilization. At 90%+ utilization with a single model, custom silicon wins on cost per token. At variable load with multiple models, GPU flexibility wins because you can serve different models on the same hardware without recompilation.

## When Custom Silicon Makes Sense

Custom silicon is not universally better than GPUs. It excels in specific deployment scenarios and fails in others.

### Deploy on Custom Silicon When:

1. **High-volume, single-model serving**: You serve one model (or a small set) at millions of queries per day. The compilation cost amortizes over billions of tokens.

2. **Latency is the primary constraint**: Interactive applications (chatbots, coding assistants, real-time translation) where time-to-first-token and inter-token latency directly impact user experience.

3. **Cost optimization at scale**: After validating product-market fit on GPUs, you need to reduce inference cost by 50%+ to achieve unit economics.

4. **Predictable workload patterns**: Traffic follows known patterns (peak hours, geographic distribution), allowing capacity planning without GPU-style dynamic scaling.

5. **Long-running deployments**: The model will be served for months without architecture changes. The upfront compilation and optimization cost pays back over extended deployment.

### Stay on GPUs When:

1. **Research and development**: You iterate on model architectures weekly. Recompiling for custom silicon after each change is impractical.

2. **Multi-model serving**: You serve dozens of models on shared infrastructure, swapping them based on demand. GPU memory management (like vLLM's model swapping) enables this; custom silicon generally does not.

3. **Cutting-edge architectures**: You deploy models with novel attention patterns, custom operators, or experimental architectures that custom silicon compilers have not implemented yet.

4. **Small scale**: Below approximately 1 million tokens per day, the overhead of deploying and managing custom silicon infrastructure exceeds the cost savings.

5. **Rapid iteration**: Product requirements change frequently, requiring different model sizes, quantization levels, or serving configurations. GPU flexibility enables experimentation without recompilation.

### The Hybrid Approach

Many organizations adopt a hybrid strategy:

- **GPUs for development and experimentation**: Use H100/A100 clusters for training, evaluation, and initial deployment
- **Custom silicon for production at scale**: Once a model is validated and traffic is predictable, migrate to Inferentia/Groq/Cerebras for cost reduction
- **GPU as fallback**: Maintain GPU capacity for traffic spikes, new model rollouts, and workloads that custom silicon cannot handle

This mirrors the historical pattern in other industries. CPUs handle general computation while DSPs, FPGAs, and ASICs handle specific high-volume workloads (video encoding, network packet processing, cryptocurrency mining).

## The Ecosystem Problem: Why GPUs Will Dominate for Years

Despite performance and cost advantages, custom silicon faces a formidable barrier: the CUDA ecosystem. Understanding this barrier explains why GPU dominance persists even when alternatives are technically superior.

### The CUDA Moat

CUDA is not just a programming language. It is an ecosystem comprising:

1. **Libraries**: cuBLAS, cuDNN, NCCL, cuSPARSE, Thrust (thousands of optimized primitives)
2. **Frameworks**: PyTorch, TensorFlow, JAX all target CUDA as primary backend
3. **Tools**: NSight profiler, cuda-gdb, Tensor Core instrumentation
4. **Knowledge**: Millions of developers trained on CUDA. Every ML paper includes CUDA benchmarks.
5. **Serving stacks**: vLLM, TensorRT-LLM, TGI, SGLang all assume NVIDIA GPUs
6. **Hardware ecosystem**: NVLink, NVSwitch, InfiniBand (Mellanox/NVIDIA), DGX systems

For a custom silicon vendor to compete, they must replicate not just the chip performance but the entire software stack. This is why every alternative ships with their own SDK:

| Vendor | SDK | Maturity |
|--------|-----|----------|
| Groq | GroqWare, GroqFlow | Early (2023+) |
| Cerebras | Cerebras SDK, ModelZoo | Growing (2020+) |
| AWS | Neuron SDK, transformers-neuronx | Mature for supported models (2019+) |
| Google | JAX/XLA on TPU | Mature (2016+) |

### The Operator Coverage Gap

The most practical barrier is operator coverage. A typical LLM uses 50-100 distinct operators (matmul, softmax, layer_norm, rotary embedding, etc.). GPUs support all of them through CUDA kernels. Custom silicon supports a subset:

- **Well-supported**: matmul, softmax, layer_norm, GELU, element-wise ops
- **Partially supported**: Rotary positional encoding, grouped-query attention, mixture-of-experts routing
- **Often missing**: Custom attention patterns (sliding window, dilated), novel activation functions, sparse operations

When a model uses an unsupported operator, the deployment path is:
1. Decompose the operator into supported primitives (performance loss)
2. Wait for the vendor to implement native support (days to months)
3. Fall back to GPU for that model

This creates a chicken-and-egg problem. Model developers target GPU because it supports everything. Custom silicon vendors prioritize the most popular models. Novel architectures always hit GPU first.

### Breaking the Ecosystem Lock-In

Several forces are slowly eroding CUDA dominance:

1. **OpenAI Triton**: A Python-based compiler that generates kernels for multiple backends (CUDA, ROCm, and potentially custom silicon). Reduces the porting effort for new operators.

2. **MLIR/StableHLO**: Compiler intermediate representations that enable "write once, compile anywhere" for ML workloads. Google's TPU, AMD's ROCm, and custom silicon vendors all consume MLIR.

3. **ONNX Runtime**: Model exchange format that abstracts hardware-specific execution. Models exported to ONNX can target multiple backends.

4. **Cloud API abstraction**: GroqCloud, Cerebras Inference, and SageMaker endpoints expose inference via REST APIs. Application developers never see the underlying hardware, reducing switching cost to zero at the API level.

The trajectory is clear: hardware diversity is increasing, but the timeline for CUDA parity is measured in years, not months.

## Mental Model: The Flexibility-Efficiency Tradeoff

Custom silicon trades flexibility for efficiency. This is the fundamental lens through which to evaluate every alternative to GPU:

```
                    High Flexibility
                         |
              CPU -------|
                         |
              GPU -------|----------- Moderate Efficiency
                         |
              TPU -------|
                         |
          Inferentia ----|
                         |
           Groq LPU ----|----------- High Efficiency
                         |
        Cerebras WSE ----|
                         |
         Full ASIC ------|----------- Maximum Efficiency
                         |
                    Low Flexibility
```

**The decision framework:**

1. If your workload is unpredictable or rapidly changing, stay near the top (GPU). The flexibility premium is worth paying.
2. If your workload is predictable and high-volume, move toward the bottom. The 10x efficiency gain pays for the ecosystem cost.
3. Most production deployments land in the middle: GPUs for development, custom silicon for the highest-volume inference paths.

The memory bandwidth wall is a physics problem, not an engineering problem. GPUs cannot solve it without fundamental architecture changes (which would make them less general-purpose). Custom silicon solves it by design. As inference becomes the dominant AI compute workload, the economic pressure toward specialized hardware will only increase.

The question is not whether custom silicon will matter. It is whether the ecosystem cost of adoption falls fast enough to match the pace of model innovation. For now, GPUs remain the safe default. But for high-volume, latency-sensitive inference at scale, the custom silicon alternative is already compelling.

---


## Google TPU: The Pioneer That Proved the Concept

No discussion of custom AI silicon is complete without acknowledging Google's Tensor Processing Unit, the chip that proved purpose-built hardware could outperform GPUs for neural network workloads. While TPUs are primarily used for training (Google's own models: Gemini, PaLM, BERT were all trained on TPUs), their architectural innovations directly influenced every chip in this module.

### TPU Architecture Evolution

| Generation | Year | Key Innovation | Inference Relevance |
|-----------|------|---------------|-------------------|
| TPU v1 | 2016 | Matrix multiply unit, 8-bit inference | First proof custom silicon beats GPU |
| TPU v2 | 2017 | bfloat16, HBM, training capability | Established custom silicon for scale |
| TPU v3 | 2018 | Liquid cooling, 2x compute | Demonstrated thermal scaling |
| TPU v4 | 2021 | Optical interconnect, 4096-chip pods | Proved custom interconnect matters |
| TPU v5e | 2023 | Cost-optimized inference | Direct inference cost competition |
| TPU v5p | 2023 | 3x bandwidth over v4 | Memory-bound workload focus |
| Trillium (v6e) | 2024 | 4.7x compute over v5e | Generative AI inference at scale |

**TPU v5e** is specifically relevant to this module: it is Google's cost-optimized inference chip, designed to serve Gemini and PaLM at massive scale in Google Cloud. Key specs:

- 256 GB HBM per chip (vs 80 GB for H100)
- Optimized for serving, not training
- Available via Google Cloud (Vertex AI endpoints)
- Supports JAX and TensorFlow models natively

**Lessons from Google's TPU program:**

1. **Vertical integration wins**: Google controls the chip, the compiler (XLA), the framework (JAX/TensorFlow), and the deployment platform (Vertex AI). This enables optimizations impossible for third-party chip vendors.
2. **Software matters more than hardware**: TPU v1 was architecturally simple. Its advantage came from XLA compiler optimizations that mapped TensorFlow graphs efficiently onto the hardware.
3. **Scale justifies specialization**: Google processes billions of inference requests daily. At that scale, even 20% efficiency improvement justifies multi-billion dollar chip development programs.

## The Economics of Custom Silicon Deployment

Understanding when custom silicon makes economic sense requires modeling the total cost of ownership, not just the chip cost or the per-token API price.

### Total Cost Model

For an on-premise deployment, the cost equation is:

```
Total annual cost = Hardware + Power + Cooling + Software + Personnel + Opportunity cost

Where:
  Hardware = (chip_cost × num_chips) / amortization_years
  Power = chips × TDP_watts × $/kWh × 8,760 hours × PUE
  Cooling = ~30% of power cost (for liquid cooling)
  Software = engineering_hours × hourly_rate (porting, optimization, maintenance)
  Personnel = dedicated_ops_team × annual_salary (custom silicon requires specialized ops)
  Opportunity cost = time_to_deploy × daily_revenue_loss (compilation + validation time)
```

### Break-Even Analysis

Consider a concrete scenario: serving Llama 3.1 70B at 10 million tokens per day.

**GPU deployment (8x H100 DGX):**
```
Hardware: $300K / 3 years = $100K/year
Power: 10 kW × $0.10/kWh × 8,760 × 1.3 PUE = $11.4K/year
Throughput: ~30 tok/s × 86,400s = 2.6M tok/day (need 4 instances)
Total: ~$450K/year for 10M tok/day capacity
Cost per million tokens: $0.12
```

**Inferentia2 (AWS managed, inf2.48xlarge):**
```
Instance cost: $12.98/hour × 24 × 365 = $113.7K/year
Throughput: ~40 tok/s × 86,400s = 3.5M tok/day (need 3 instances)
Total: ~$341K/year for 10M tok/day capacity
Cost per million tokens: $0.09
Savings vs GPU: 24%
```

**GroqCloud API (pay per token):**
```
API pricing: ~$0.27 per million tokens (output, as of early 2026)
10M tokens/day × 365 = 3.65B tokens/year
Total: ~$986K/year
Cost per million tokens: $0.27
```

The API pricing is higher because it includes margins, but requires zero infrastructure management. For organizations without ML infrastructure teams, the API approach may still be more economical when accounting for personnel costs.

### The Utilization Sensitivity

The economics of custom silicon are extremely sensitive to utilization:

```
At 90% utilization: Custom silicon wins by 50%+ on cost/token
At 50% utilization: Roughly breakeven with GPU
At 30% utilization: GPU wins (flexibility allows serving other workloads during idle time)
```

This is why custom silicon excels for high-volume, predictable workloads and fails for variable traffic patterns. A GPU cluster at 30% utilization for your primary model can serve other models, run fine-tuning jobs, or handle batch processing during idle periods. An idle Groq rack is just burning power.

## Future Directions: Where Custom Silicon Is Heading

The custom silicon landscape is evolving rapidly. Several trends will shape the next 2-3 years:

### Trend 1: Disaggregated Prefill and Decode

From Module 04.5 (Sarathi-Serve) and Module 06.4, you know that prefill and decode have fundamentally different compute profiles. Custom silicon is ideally positioned for disaggregated serving:

- **GPUs handle prefill**: Compute-bound, benefits from high FLOPS, variable input lengths
- **Custom silicon handles decode**: Memory-bound, benefits from high bandwidth, predictable execution

This hybrid architecture gives each hardware type its optimal workload. NVIDIA's acquisition of Groq may target exactly this use case: Groq LPUs as dedicated decode accelerators attached to H100 prefill clusters.

### Trend 2: Inference-Time Compute (Test-Time Scaling)

Models like OpenAI's o1 and DeepSeek-R1 generate many internal reasoning tokens before producing output. This dramatically increases decode compute per query:

- Standard chatbot: 100-500 output tokens per query
- Reasoning model: 5,000-50,000 internal tokens per query

Custom silicon's decode speed advantage becomes 10x more valuable when each query requires 10x more decode tokens. A reasoning query on H100 (30 tok/s) takes 167 seconds for 5000 tokens. On Cerebras (2500 tok/s), it takes 2 seconds.

### Trend 3: Edge Inference ASICs

While this module focuses on datacenter chips, the same principles apply to edge devices:

- **Apple Neural Engine**: Custom silicon in every iPhone/Mac for on-device LLM inference
- **Qualcomm Hexagon NPU**: Mobile inference acceleration for Android devices
- **Intel Meteor Lake NPU**: Laptop-class inference for Copilot workloads

Edge ASICs follow the same tradeoff: sacrifice generality for efficiency on the specific inference workload. The key difference is power constraint (5-15W vs 300-700W).

### Trend 4: Photonic and Analog Compute

Emerging approaches that may disrupt digital custom silicon:

- **Lightmatter**: Uses photonic (light-based) interconnect and compute for matrix multiplications at near-zero energy cost
- **Analog Inference chips**: Store weights as analog voltages and compute matrix multiplications via physical current summation (no digital multiply-accumulate)

These are 3-5 years from production viability but represent the ultimate endpoint of the specialization spectrum: hardware where the physics of the device directly implements the computation.

---


### Trend 5: Chiplet and Multi-Die Architectures

Rather than building monolithic dies (limited by reticle size) or entire wafers (Cerebras), several vendors are exploring chiplet architectures for inference:

- **AMD MI300X**: Uses chiplet design with separate compute and memory dies on a single package. 192 GB HBM3 with 5.3 TB/s bandwidth.
- **Intel Gaudi 3**: Modular architecture connecting compute tiles with memory tiles via high-bandwidth bridges.
- **NVIDIA Blackwell B200**: Dual-die design (two compute dies on one package) with 192 GB HBM3e at 8 TB/s.

Chiplets represent a middle ground: more bandwidth than monolithic GPU (multiple memory stacks per package) without the manufacturing complexity of wafer-scale. For inference specifically, chiplet architectures enable placing SRAM cache dies directly adjacent to compute dies, creating a fast local memory layer that mitigates the HBM bandwidth wall without eliminating HBM entirely.

### Trend 6: Compiler-First Silicon Design

A notable shift in custom silicon development: newer chips are designed compiler-first rather than hardware-first. The hardware architecture is co-designed with the compiler to ensure that the theoretical peak performance is actually achievable by the software stack.

Groq's TSP architecture exemplifies this: the hardware is intentionally simple (no caches, no branch prediction, no out-of-order execution) because the compiler handles all scheduling at compile time. This inverts the traditional GPU model where complex hardware (warp schedulers, cache hierarchies, memory controllers) compensates for compiler limitations.

The implication for practitioners: when evaluating custom silicon, the compiler quality matters as much as the hardware specs. A chip with 2x the theoretical bandwidth but 50% compiler efficiency delivers the same real-world performance as a chip with 1x bandwidth and 100% efficiency.

## Summary

Custom silicon for LLM inference is not a future speculation. It is deployed today, serving production traffic, and delivering 10-50x performance improvements over GPUs for decode-heavy workloads. The four approaches covered in this module represent different points on the flexibility-efficiency spectrum:

1. **Groq LPU**: Maximum decode speed through deterministic SRAM-only execution. Best for latency-critical applications.
2. **Cerebras WSE-3**: Maximum bandwidth through wafer-scale integration. Best for absolute throughput on large models.
3. **AWS Inferentia/Trainium**: Maximum cost efficiency through cloud-native optimization. Best for organizations already on AWS at scale.
4. **Meta MTIA**: Vertical integration for the world's largest inference workload. Not externally available but indicative of industry direction.

The GPU will remain the default choice for most organizations for years to come, protected by the CUDA ecosystem moat and the flexibility premium that matters during rapid model iteration. But for the minority of deployments that serve millions of users with stable models (which accounts for the majority of global inference compute), custom silicon is already the economically rational choice.

The memory bandwidth wall is physics. Custom silicon is the engineering response.


---


### Decision Checklist

Before evaluating custom silicon for your deployment, answer these questions:

1. **Volume**: Do you serve more than 5 million tokens per day on a single model? (If no, GPU is almost certainly more cost-effective.)
2. **Stability**: Will this model be in production for 3+ months without architecture changes? (If no, compilation overhead dominates.)
3. **Latency SLA**: Is sub-100ms time-to-first-token a hard requirement? (If yes, Groq/Cerebras offer guarantees GPU cannot.)
4. **Operator coverage**: Does your model use only standard transformer operations? (If it uses custom ops, verify vendor support first.)
5. **Cloud strategy**: Are you single-cloud (AWS) or multi-cloud? (If single AWS, Inferentia is low-risk. If multi-cloud, API-based services reduce lock-in.)

If you answered "yes" to questions 1, 2, and at least one of 3-5, custom silicon deserves a proof-of-concept evaluation. The 50%+ cost reduction or 10x latency improvement can transform the economics of your inference deployment.

## Further Reading

- Groq: "Groq LPU Inference Engine" (groq.com/technology)
- Cerebras: "Wafer-Scale Engine Architecture" (cerebras.net/product-chip)
- AWS: "AWS Neuron SDK Documentation" (awsdocs-neuron.readthedocs-hosted.com)
- Meta: "MTIA: Meta's Training and Inference Accelerator" (ai.meta.com/blog/meta-training-inference-accelerator-AI-MTIA)
- Google: "TPU v5e: Cost-Optimized Inference" (cloud.google.com/tpu)
- Jouppi et al., "TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning" (2023)
- Patterson, "A Domain-Specific Architecture for Deep Neural Networks" (2018)
- NVIDIA: "NVIDIA to Acquire Groq" (December 2025 announcement)

