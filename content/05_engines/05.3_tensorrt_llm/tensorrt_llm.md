# TensorRT-LLM: Compiler-First Inference on NVIDIA Hardware

When you need the absolute highest throughput from NVIDIA GPUs, you reach for TensorRT-LLM. Unlike framework-based serving engines that interpret model graphs at runtime, TensorRT-LLM is fundamentally a compiler: it takes your model definition, analyzes every operation, fuses kernels, selects optimal data layouts, and produces a binary engine file that squeezes every FLOP from the hardware. The tradeoff is explicit. You pay compilation time upfront (minutes to hours depending on model size) in exchange for inference that runs closer to the theoretical peak of your GPU than any other approach.

TensorRT-LLM sits atop NVIDIA's TensorRT deep learning compiler, extending it with LLM-specific optimizations: inflight batching, paged KV caches, speculative decoding, and custom attention kernels purpose-built for the decode phase where memory bandwidth dominates. For organizations committed to NVIDIA hardware and willing to invest in the build step, it delivers throughput that justifies the operational complexity.

## Connection to Prior Modules

From Module 01.3, you know FlashAttention fuses the Q, K, V multiply and softmax into a single kernel pass, reducing HBM traffic from O(N^2) to O(N) in sequence length. TRT-LLM applies this principle to the entire model: every layer, every residual connection, every normalization operation is analyzed for fusion opportunities. Where FlashAttention optimizes one operation, TRT-LLM optimizes the complete forward pass as a unified compilation unit. The result is not just faster attention but faster everything: layer norms folded into preceding operations, residual additions merged with subsequent kernels, and quantization nodes placed at mathematically optimal boundaries.

From Module 04.1, you understand how batching amortizes the cost of weight loading across multiple sequences. TRT-LLM's inflight batching implementation takes this further by allowing new sequences to enter the batch at every decode step, not just at batch boundaries. This eliminates the "batch drain" problem where short sequences finish but their slots sit empty until the entire batch completes.

From Module 05.1 on vLLM, you know PagedAttention solved KV cache fragmentation through virtual memory techniques. TRT-LLM implements its own paged KV cache with a critical difference: the page table lookups are compiled into the attention kernel itself, eliminating the overhead of a separate memory management layer.

## Architecture: From Python to Optimized Engine

TRT-LLM's architecture has three distinct phases, each with different performance characteristics and operational implications.

### Phase 1: Model Definition in Python

Every model in TRT-LLM starts as a Python class that defines the network topology. This is not a PyTorch model in the traditional sense. Instead, it uses TRT-LLM's own tensor operations that build a computation graph rather than executing eagerly:

```python
import tensorrt_llm
from tensorrt_llm.layers import (
    Attention, MLP, LayerNorm, Embedding, Linear
)
from tensorrt_llm.models import PretrainedModel

class LlamaDecoderLayer(Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.input_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=config.norm_epsilon
        )
        self.attention = Attention(
            local_layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.causal,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size
        )
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size
        )

    def forward(self, hidden_states, attention_mask, kv_cache_params):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params
        )
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
```

The key insight: this code never runs inference. It constructs a symbolic graph that the compiler will optimize. Every `tensorrt_llm` operation records what should happen, not what does happen. This deferred execution model is what enables whole-graph optimization.

### Phase 2: Compilation (The Build Step)

The build step transforms the symbolic graph into an optimized TensorRT engine. This is where the real magic happens:

```bash
# Build command for Llama 3.1 70B with FP8 quantization
trtllm-build \
    --checkpoint_dir ./llama-70b-checkpoint/ \
    --output_dir ./llama-70b-engine/ \
    --gemm_plugin fp8 \
    --gpt_attention_plugin fp8 \
    --max_batch_size 256 \
    --max_input_len 4096 \
    --max_seq_len 8192 \
    --tp_size 4 \
    --pp_size 1 \
    --use_paged_context_fmha enable \
    --workers 4
```

During compilation, TensorRT performs:

1. **Layer fusion**: Adjacent operations that can share memory are merged. A typical transformer layer might go from 40+ individual CUDA kernel launches to 5-8 fused mega-kernels.

2. **Kernel auto-tuning**: For each fused operation, TensorRT benchmarks multiple implementation strategies (different tile sizes, different memory access patterns, different thread configurations) and selects the fastest for the specific GPU architecture.

3. **Memory planning**: The compiler computes the exact memory layout for every intermediate tensor, eliminating dynamic allocation during inference. Every buffer address is known at compile time.

4. **Precision calibration**: When using INT8 or FP8, the compiler inserts quantize/dequantize nodes at optimal positions, minimizing accuracy loss while maximizing throughput from lower-precision tensor cores.

5. **Shape specialization**: The engine is compiled for specific input shape ranges (min/opt/max batch size, sequence length). Operations are specialized for these ranges, avoiding the overhead of dynamic shape handling.

Build times scale with model complexity:

| Model | GPU | TP | Build Time | Engine Size |
|-------|-----|-----|-----------|-------------|
| Llama 3.1 8B | 1x H100 | 1 | ~8 min | 16 GB |
| Llama 3.1 70B | 4x H100 | 4 | ~45 min | 140 GB |
| Llama 3.1 405B | 8x H100 | 8 | ~3 hours | 810 GB |

### Phase 3: Runtime Execution

Once compiled, the engine file is loaded and executed by the TRT-LLM runtime:

```python
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir(
    engine_dir="./llama-70b-engine/",
    rank=tensorrt_llm.mpi_rank()
)

outputs = runner.generate(
    batch_input_ids=input_ids,
    max_new_tokens=512,
    end_id=tokenizer.eos_token_id,
    pad_id=tokenizer.pad_token_id,
    temperature=0.7,
    top_p=0.9,
    streaming=True
)
```

The runtime is thin by design. It manages the KV cache, schedules inflight batching, and feeds tokens to the compiled engine. No graph interpretation, no operator dispatch, no JIT compilation. Every kernel launch was decided at build time.

## Key Optimizations

### XQA Kernel: 2.4x Decode Throughput

The decode phase of LLM inference is fundamentally memory-bandwidth bound: each token generation requires reading the entire KV cache but performs very little computation per byte loaded. Standard attention kernels waste bandwidth because they are designed for the prefill phase where the compute-to-memory ratio is favorable.

TRT-LLM's XQA (Cross-Query Attention) kernel is purpose-built for decode. The key innovations:

1. **Persistent thread blocks**: Instead of launching a new kernel for each attention head, XQA keeps thread blocks resident on the SM. This eliminates kernel launch overhead and enables the threads to maintain their register state across multiple heads.

2. **KV cache streaming**: The kernel loads KV cache pages in a streaming pattern optimized for L2 cache hit rates. Pages accessed by multiple heads are loaded once and shared through L2, not re-fetched from HBM.

3. **Fused softmax and output projection**: The attention weights, softmax normalization, and value projection are computed without writing intermediate results to HBM.

4. **GQA-native**: For models using Grouped Query Attention (e.g., Llama 3.1 with 8 KV heads shared across 64 query heads), XQA avoids redundant KV cache reads by computing all query heads sharing a KV group in the same thread block.

Benchmark results on H100 SXM (Llama 3.1 70B, batch size 128, sequence length 4096):

| Kernel | Decode Throughput | HBM Bandwidth Utilization |
|--------|------------------|--------------------------|
| Standard MHA | 1,850 tok/s | 62% |
| FlashDecoding | 2,900 tok/s | 78% |
| XQA | 4,440 tok/s | 91% |

The 2.4x improvement over standard attention comes from approaching the theoretical bandwidth limit of H100 (3.35 TB/s). At 91% bandwidth utilization, the kernel is nearly memory-bandwidth optimal.

### Inflight Batching

Traditional static batching groups requests into a batch and processes them together until all complete. This creates two problems: padding waste (short sequences padded to max length) and batch drain (completed sequences hold slots until the slowest finishes).

TRT-LLM's inflight batching solves both:

```
Time Step 0: [Seq A prefill] [Seq B prefill] [Seq C prefill]
Time Step 1: [Seq A decode]  [Seq B decode]  [Seq C decode]
Time Step 2: [Seq A decode]  [Seq B DONE]    [Seq C decode]  [Seq D prefill] <- D enters immediately
Time Step 3: [Seq A decode]  [Seq D decode]  [Seq C decode]
```

The scheduler operates at token granularity:
- Every decode step, check if any sequence has finished (EOS or max length)
- Immediately reclaim that sequence's KV cache pages
- If waiting requests exist, begin their prefill in the freed slots
- Prefill and decode operations for different sequences execute concurrently within the same batch

This keeps GPU utilization consistently above 85% even with highly variable sequence lengths, compared to 50-60% for static batching under the same workload.

### FP8 and NVFP4 Quantization

TRT-LLM leverages NVIDIA's hardware quantization support to reduce memory footprint and increase throughput without significant accuracy loss.

**FP8 (E4M3)** on H100/H200:
- 4-bit exponent, 3-bit mantissa
- Dynamic range: 0.001953 to 448
- Per-tensor or per-channel scaling factors stored in FP32
- Tensor Core throughput: 2x FP16 (1,979 TFLOPS vs 989 TFLOPS on H100)

```python
# FP8 quantization during build
from tensorrt_llm.quantization import QuantAlgo

config = {
    'quantization': {
        'quant_algo': QuantAlgo.FP8,
        'kv_cache_quant_algo': QuantAlgo.FP8,
    }
}
```

**NVFP4 (Blackwell architecture)**:
- 4-bit floating point with block-level scaling
- Achievable only on B100/B200 GPUs
- 4x memory reduction over FP16 for weights
- Combined with FP8 KV cache: model fits in 1/4 the GPU memory

Memory savings for Llama 3.1 70B:

| Precision | Weight Memory | KV Cache (4K seq, batch 128) | Total |
|-----------|--------------|------------------------------|-------|
| FP16 | 140 GB | 80 GB | 220 GB |
| FP8 | 70 GB | 40 GB | 110 GB |
| NVFP4 weights + FP8 KV | 35 GB | 40 GB | 75 GB |

The practical impact: FP8 allows serving Llama 3.1 70B on 2x H100 instead of 4x, halving infrastructure cost while maintaining >99% of FP16 accuracy on standard benchmarks.

### EAGLE Speculative Decoding

Speculative decoding exploits a fundamental asymmetry: verifying multiple token candidates in parallel costs almost the same as generating one token, because prefill (parallel verification) is compute-bound while decode (sequential generation) is memory-bandwidth-bound. TRT-LLM integrates EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) as its primary speculative decoding method.

The EAGLE approach differs from draft-model speculation:

**Draft-model speculation** (used by others):
1. Run a small model (e.g., 1B params) to generate K candidate tokens
2. Verify all K candidates with the large model in one forward pass
3. Accept the longest matching prefix

**EAGLE speculation** (TRT-LLM's approach):
1. Train a lightweight feature-extrapolation head on top of the main model's hidden states
2. The head predicts the next hidden state (not the next token directly)
3. From predicted hidden states, generate a tree of candidate continuations
4. Verify the entire tree in one forward pass using tree attention masks

The EAGLE head is approximately 0.5-1% of the main model's parameters (350M params for a 70B model). Because it operates on hidden states rather than making independent token predictions, it achieves higher acceptance rates than draft models:

| Method | Acceptance Rate | Speedup (Llama 70B) | Extra Memory |
|--------|----------------|--------------------:|--------------|
| Draft model (7B) | 60-70% | 1.5-1.8x | +14 GB |
| Medusa heads | 55-65% | 1.4-1.6x | +2 GB |
| EAGLE | 75-85% | 2.0-2.8x | +1.4 GB |

TRT-LLM compiles the EAGLE head alongside the main model, fusing the speculation and verification into a single engine:

```python
# Enable EAGLE speculative decoding in build config
build_config = BuildConfig(
    max_batch_size=64,
    max_beam_width=1,
    speculative_decoding_mode=SpeculativeDecodingMode.EAGLE,
    max_draft_len=7,  # Generate up to 7 speculative tokens per step
)
```

The compiled engine handles the tree attention masking internally, avoiding the overhead of separate kernel launches for speculation and verification. On latency-sensitive workloads (single user, long generations), EAGLE delivers 2.5x speedup at the cost of ~1% additional memory.

### Paged KV Cache

TRT-LLM implements paged KV cache with a twist: the page table is compiled into the attention kernel. Unlike vLLM's runtime page table (Module 05.1), where a Python-level scheduler manages block allocation and passes page tables as kernel arguments, TRT-LLM's approach bakes the page table lookup into the kernel's memory access pattern.

The implementation uses fixed-size pages (typically 64 or 128 tokens per page):

```
Physical Memory Layout:
[Page Pool: N pages x page_size x num_layers x 2 x num_kv_heads x head_dim]

Logical Sequence View:
Seq A: [Page 7] -> [Page 2] -> [Page 15] -> [Page 8]  (256 tokens)
Seq B: [Page 0] -> [Page 12]                           (128 tokens)
Seq C: [Page 3] -> [Page 9] -> [Page 1]                (192 tokens)
```

Benefits of compilation-integrated paging:
1. Zero runtime overhead for page table lookups (address computation is in the kernel binary)
2. Pages can be pre-fetched into L2 based on the known access pattern
3. Free pages are managed by a simple bitmap, no garbage collection needed
4. Cross-sequence page sharing for common prefixes (system prompts) requires zero copies

The tradeoff: page size is fixed at compile time. If your workload has highly variable sequence lengths, you may waste memory on partially-filled final pages. In practice, with 64-token pages, average waste is 32 tokens per sequence (negligible for sequences >1K tokens).

## PyTorch-First Backend (v1.0+)

TRT-LLM v1.0 (released early 2025) introduced a paradigm shift: the PyTorch-first backend. Previously, users had to define models using TRT-LLM's custom Python API (shown in Phase 1 above). The new backend accepts standard PyTorch models and compiles them automatically.

The motivation was clear from user feedback: the custom API created a maintenance burden. Every new model architecture required manual translation from HuggingFace PyTorch code to TRT-LLM's symbolic graph API. With hundreds of new models released monthly, this was unsustainable.

### How PyTorch-First Works

```python
import torch
from transformers import AutoModelForCausalLM
import tensorrt_llm

# Load a standard HuggingFace model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B",
    torch_dtype=torch.float16
)

# TRT-LLM compiles it directly
engine = tensorrt_llm.compile(
    model,
    backend="torch_trt",  # Use PyTorch-TensorRT path
    max_batch_size=128,
    max_seq_len=8192,
    quantization="fp8",
    tp_size=4
)

# Serve immediately
runner = tensorrt_llm.runtime.ModelRunner(engine)
```

Under the hood, the PyTorch-first backend:
1. Traces the PyTorch model using `torch.export` to capture the computation graph
2. Converts the exported graph to TensorRT's internal representation
3. Applies LLM-specific fusion passes (attention fusion, MLP fusion, RoPE fusion)
4. Compiles with the same optimization pipeline as the manual API

### Performance Comparison

| Approach | Build Complexity | Build Time | Runtime Performance |
|----------|-----------------|-----------|-------------------|
| Manual TRT-LLM API | High (custom code) | Baseline | 100% |
| PyTorch-first (torch_trt) | Low (load HF model) | +15-20% | 95-98% |
| PyTorch eager (no compile) | None | None | 40-50% |

The 2-5% performance gap between PyTorch-first and manual API comes from edge cases where the manual API's explicit graph construction enables optimizations that automatic tracing cannot infer (e.g., custom fusion patterns for novel architectures). For standard architectures (Llama, Mistral, GPT-NeoX), the gap is negligible.

### When to Use Which Backend

**Use PyTorch-first when:**
- Rapid prototyping with new model architectures
- You need to serve a model within hours, not days
- The model architecture is well-supported (Llama, Mistral, Phi, Gemma)
- You want to iterate on model modifications without rewriting TRT-LLM layers

**Use manual API when:**
- Custom architectures with non-standard attention patterns
- You need the last 2-3% of performance
- Production deployment where the model will run unchanged for months
- Novel quantization schemes not yet supported by automatic conversion

## Multi-GPU Inference

TRT-LLM provides native multi-GPU support compiled directly into the engine, not bolted on as a runtime layer. When you specify `tp_size=4`, the compiler partitions the model and generates four separate engine files, each containing only the operations for its partition.

### Tensor Parallelism (TP)

Tensor parallelism splits individual layers across GPUs. For a transformer layer:
- Attention: Q, K, V projections split by head across GPUs (each GPU owns num_heads/tp_size heads)
- MLP: First linear split column-wise, second linear split row-wise
- All-reduce after attention output projection and after MLP second linear

```python
# Build with tensor parallelism
trtllm-build \
    --checkpoint_dir ./model/ \
    --tp_size 4 \
    --output_dir ./engine_tp4/

# This produces 4 engine files:
# engine_tp4/rank0.engine (layers split, handles heads 0-15)
# engine_tp4/rank1.engine (layers split, handles heads 16-31)
# engine_tp4/rank2.engine (layers split, handles heads 32-47)
# engine_tp4/rank3.engine (layers split, handles heads 48-63)
```

Because the parallelism is compiled in, the all-reduce communication pattern is fixed at build time. TRT-LLM uses NCCL with optimized ring/tree topologies selected based on the NVLink/NVSwitch topology detected during compilation.

### Pipeline Parallelism (PP)

Pipeline parallelism assigns complete layers to different GPUs:

```
GPU 0: Layers 0-19  (prefill and decode for its portion)
GPU 1: Layers 20-39 (receives activations from GPU 0)
GPU 2: Layers 40-59 (receives activations from GPU 1)
GPU 3: Layers 60-79 (produces final logits)
```

PP reduces per-GPU memory but introduces pipeline bubbles. TRT-LLM mitigates this with:
- **Microbatching**: Split the batch into microbatches that pipeline through stages
- **Interleaved scheduling**: Begin the next microbatch's prefill while the current one decodes
- **Compiled communication**: Send/receive operations are fused into the engine, overlapping computation with data transfer

### Expert Parallelism (EP) for MoE

For Mixture-of-Experts models (Mixtral, DBRX, DeepSeek-V3), TRT-LLM adds expert parallelism:

```python
# Mixtral 8x22B: 8 experts, using EP=4 (2 experts per GPU)
trtllm-build \
    --checkpoint_dir ./mixtral-8x22b/ \
    --tp_size 2 \
    --ep_size 4 \
    --output_dir ./engine_ep4/
```

With EP=4 on Mixtral (8 experts), each GPU holds 2 experts. The router's top-2 selection determines which GPUs need to process each token. TRT-LLM compiles the all-to-all communication pattern for expert dispatch:

1. Router selects top-K experts for each token
2. All-to-all shuffle sends tokens to the GPUs holding their selected experts
3. Each GPU processes tokens routed to its local experts
4. All-to-all gather returns results to originating GPUs

The compiled dispatch avoids the overhead of dynamic routing decisions at runtime. The router weights are the only dynamic component; the communication topology is fixed.

### Combined Parallelism

For very large models, combine all three:

```
DeepSeek-V3 (671B MoE, 256 experts):
- TP=8 (within a single node, 8x H100 via NVLink)
- EP=4 (across 4 nodes, 64 experts per group)  
- PP=2 (2 pipeline stages for memory)
- Total: 64 GPUs across 8 nodes
```

The build step computes the optimal communication schedule: TP all-reduces use NVLink (900 GB/s), EP all-to-all uses InfiniBand (400 Gb/s per port), and PP point-to-point uses the lowest-latency available interconnect.

## Disaggregated Serving with NIXL

Module 04.5 introduced the concept of disaggregated prefill and decode: separating the compute-bound prefill phase from the memory-bandwidth-bound decode phase onto different GPU pools. TRT-LLM implements this natively with NVIDIA's NIXL (NVIDIA Inference Xfer Library) handling the GPU-to-GPU KV cache transfer.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Request Router                               │
│   (routes new requests to prefill pool, decode to decode pool)       │
└────────────────────┬───────────────────────────────┬────────────────┘
                     │                               │
         ┌───────────▼───────────┐       ┌──────────▼──────────┐
         │    Prefill Pool       │       │    Decode Pool       │
         │  (Compute-optimized)  │       │  (Memory-optimized)  │
         │                       │       │                       │
         │  H100 SXM, TP=4      │       │  H100 SXM, TP=2      │
         │  Batch size: 32       │       │  Batch size: 512      │
         │  High arithmetic      │       │  High bandwidth       │
         │  intensity workload   │       │  utilization          │
         └───────────┬───────────┘       └──────────▲──────────┘
                     │                               │
                     │    NIXL KV Cache Transfer     │
                     │    (RDMA, GPU-Direct)         │
                     └───────────────────────────────┘
```

### NIXL: GPU-to-GPU KV Transfer

NIXL provides zero-copy KV cache transfer between GPUs across nodes:

```python
# Disaggregated serving configuration
executor_config = ExecutorConfig(
    disaggregated_serving=DisaggregatedServingConfig(
        prefill_config=PrefillConfig(
            max_batch_size=32,
            max_input_len=32768,
        ),
        decode_config=DecodeConfig(
            max_batch_size=512,
            max_new_tokens=4096,
        ),
        kv_transfer=KvTransferConfig(
            backend="nixl",  # GPU-Direct RDMA
            # Alternative: "nccl" for within-node
        )
    )
)
```

NIXL transfer characteristics:
- **Latency**: 50-100 microseconds for page transfer over NVLink, 200-500 microseconds over InfiniBand RDMA
- **Bandwidth**: Saturates available interconnect (900 GB/s NVLink, 50 GB/s per IB port)
- **Overlap**: Transfers happen concurrently with ongoing decode computation on the decode pool

For a Llama 3.1 70B model with 4K context length, the KV cache for one sequence is approximately 2.5 GB (80 layers x 8 KV heads x 128 head_dim x 4096 tokens x 2 bytes FP16 x 2 for K and V). Over NVLink, this transfers in ~2.8 ms. Over 4x InfiniBand (200 GB/s aggregate), approximately 12.5 ms. Both are fast enough that the decode pool can begin generating tokens within one decode step of prefill completion.

### When Disaggregation Wins

Disaggregated serving provides the highest throughput when:
1. **Long prefills**: Input sequences >4K tokens create prefill jobs lasting 100+ ms, blocking decode slots
2. **Heterogeneous hardware**: Use compute-dense GPUs (H100) for prefill, memory-optimized GPUs for decode
3. **SLA diversity**: Latency-critical decode (chatbots) coexists with throughput-optimized prefill (batch processing)

The break-even point: disaggregation adds NIXL transfer overhead but removes prefill/decode interference. For workloads where average input length exceeds 2K tokens and batch sizes exceed 64, disaggregation improves total system throughput by 30-50% compared to colocated serving.

## When to Choose TRT-LLM

### TRT-LLM is the Right Choice When:

**Maximum single-node throughput is the goal.** TRT-LLM consistently benchmarks 15-30% higher throughput than vLLM on identical NVIDIA hardware at the same precision. The gap comes from kernel compilation (fused operations vs. individual kernel launches) and XQA's near-optimal bandwidth utilization.

**FP8 precision is required.** TRT-LLM's FP8 support is the most mature in the ecosystem, with per-tensor calibration, mixed-precision (FP8 compute with FP16 master weights), and compiled quantize/dequantize placement that minimizes accuracy loss.

**Latency-critical serving with stable models.** If you deploy a model and serve it unchanged for weeks or months, the upfront compilation cost amortizes to zero. The resulting engine provides deterministic, low-jitter latency.

**NVIDIA-only infrastructure.** If your fleet is exclusively H100/H200/B100, TRT-LLM extracts hardware-specific performance that portable frameworks leave on the table.

**Speculative decoding with EAGLE.** TRT-LLM's compiled EAGLE integration achieves the highest acceptance rates and lowest overhead of any speculative decoding implementation.

### Choose Something Else When:

**Rapid model iteration.** If you swap models weekly (fine-tuning experiments, A/B testing), the build step becomes a bottleneck. vLLM's dynamic execution is better suited.

**Multi-vendor hardware.** TRT-LLM produces NVIDIA-only engines. If you serve on AMD MI300X, Intel Gaudi, or AWS Trainium alongside NVIDIA, you need a portable solution.

**Simple deployment with minimal ops.** TRT-LLM requires understanding build configurations, TP/PP layouts, and engine management. vLLM's single-command deployment is operationally simpler.

**Custom sampling or post-processing.** TRT-LLM's compiled engine has limited flexibility for custom token selection logic. If you need complex constrained decoding (grammar-guided, JSON schemas), check that your specific constraints are supported before committing.

## Performance Benchmarks

Comparative benchmarks on H100 80GB SXM (Llama 3.1 70B, FP8, TP=4):

### Throughput (output tokens/second, batch size 128, input 2048, output 512)

| Engine | Throughput | Latency P50 | Latency P99 | GPU Util |
|--------|-----------|-------------|-------------|----------|
| TRT-LLM 1.0 | 12,800 tok/s | 38 ms/tok | 52 ms/tok | 89% |
| vLLM 0.6 | 10,200 tok/s | 48 ms/tok | 71 ms/tok | 78% |
| SGLang 0.3 | 11,100 tok/s | 43 ms/tok | 63 ms/tok | 82% |

### Time-to-First-Token (TTFT, single request, input 4096 tokens)

| Engine | TTFT | Notes |
|--------|------|-------|
| TRT-LLM 1.0 | 180 ms | Compiled prefill kernel |
| vLLM 0.6 | 220 ms | PagedAttention prefill |
| SGLang 0.3 | 195 ms | RadixAttention prefill |

The throughput advantage compounds with longer sequences and larger batches because TRT-LLM's compiled memory management eliminates the per-token overhead that framework-based engines accumulate.

## Tradeoffs and Limitations

### Build Time is Non-Trivial

The compilation step is the primary operational cost:
- Cannot hot-swap models in production (each requires a full rebuild)
- Build must be repeated for each GPU architecture (H100 engine does not run on A100)
- Configuration changes (max batch size, sequence length) require rebuild
- CI/CD pipelines must include engine building as a deployment step

Mitigation: build engines during off-peak hours, maintain a cache of pre-built engines for common configurations, and use the PyTorch-first backend for development with manual API engines for production.

### NVIDIA Lock-In

Engines are compiled for specific NVIDIA architectures:
- sm_89 (L40S, RTX 4090)
- sm_90 (H100, H200)
- sm_100 (B100, B200)

An engine built for sm_90 will not run on sm_89 or sm_100. This means:
- No portability to AMD, Intel, or custom accelerators
- GPU generation upgrades require full rebuilds
- Mixed-generation fleets need multiple engine builds per model

### Debugging Complexity

When inference produces incorrect results:
- The compiled engine is a binary blob (not human-readable)
- No easy way to inspect intermediate activations
- Quantization errors require rebuilding with different calibration
- Layer fusion may hide the source of numerical issues

Best practice: validate model accuracy at FP16 first, then enable quantization and compare outputs. Use TRT-LLM's `--strongly_typed` flag during development to catch precision issues early.

### Memory Overhead from Compilation

The build process requires significantly more memory than the final engine:
- Building Llama 70B requires ~300 GB system RAM (not GPU memory)
- The auto-tuning phase temporarily allocates GPU memory for benchmarking kernel variants
- Engine file sizes are larger than raw weights due to embedded metadata and kernel binaries

## Mental Model

Think of TRT-LLM as an ahead-of-time compiler for neural network inference, analogous to how gcc compiles C code:

```
Source Code  (Python model definition)
     │
     ▼
Compiler     (TensorRT builder: fuses, optimizes, specializes)
     │
     ▼
Binary       (Engine file: hardware-specific, fast, not portable)
     │
     ▼
Runtime      (Thin executor: feeds data, manages memory, returns results)
```

You pay compile time upfront for faster inference forever after. Just as a compiled C binary runs faster than interpreted Python, a compiled TRT-LLM engine runs faster than dynamically-dispatched PyTorch. The cost is flexibility: every change requires recompilation.

The right mental model for choosing TRT-LLM: if your model deployment lifecycle looks like "build once, serve for months," TRT-LLM's compilation cost amortizes to zero and its runtime advantage dominates. If your lifecycle looks like "experiment daily, swap models weekly," the compilation tax makes vLLM or SGLang more productive choices.


## Practical Deployment Walkthrough

A complete deployment of Llama 3.1 70B on 4x H100 with FP8, from checkpoint to serving:

### Step 1: Quantize the Checkpoint

```bash
# Convert HuggingFace checkpoint to TRT-LLM format with FP8 calibration
python convert_checkpoint.py \
    --model_dir meta-llama/Llama-3.1-70B \
    --output_dir ./llama-70b-ckpt/ \
    --dtype float16 \
    --tp_size 4 \
    --use_fp8 \
    --calib_dataset ./calibration_data.json \
    --calib_batch_size 32 \
    --calib_size 512
```

Calibration uses 512 representative samples to compute per-tensor scaling factors. These factors are stored alongside the weights and baked into the compiled engine.

### Step 2: Build the Engine

```bash
trtllm-build \
    --checkpoint_dir ./llama-70b-ckpt/ \
    --output_dir ./llama-70b-engine/ \
    --max_batch_size 256 \
    --max_input_len 8192 \
    --max_seq_len 16384 \
    --gemm_plugin fp8 \
    --gpt_attention_plugin fp8 \
    --use_paged_context_fmha enable \
    --paged_kv_cache enable \
    --tokens_per_block 64 \
    --workers 4 \
    --max_num_tokens 16384
```

Critical parameters:
- `max_batch_size`: Sets the upper bound. Over-provisioning wastes memory; under-provisioning limits throughput.
- `max_input_len` vs `max_seq_len`: Input is the prompt length, seq is total (input + output). Setting seq > input enables long-generation workloads.
- `tokens_per_block`: KV cache page size. 64 is optimal for most workloads (good granularity without excessive page table overhead).
- `max_num_tokens`: Total token budget across all sequences in a batch. Limits memory consumption.

### Step 3: Serve with Triton Inference Server

```bash
# Launch Triton with TRT-LLM backend
docker run --gpus all -p 8000:8000 -p 8001:8001 \
    -v ./llama-70b-engine:/models/llama70b/1/ \
    -v ./triton_config:/models/llama70b/ \
    nvcr.io/nvidia/tritonserver:24.09-trtllm-python-py3 \
    tritonserver --model-repository /models/
```

Triton handles HTTP/gRPC ingress, request queuing, and health checks. TRT-LLM's inflight batching scheduler operates underneath, managing the actual GPU execution.

### Step 4: Validate

```python
import requests

response = requests.post("http://localhost:8000/v2/models/llama70b/generate", json={
    "text_input": "Explain the attention mechanism in transformers:",
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": True
})

# Verify: output quality, latency, throughput under load
```

### Operational Considerations

**Engine versioning**: Store engines in object storage (S3) with metadata tags: model version, GPU architecture, build config hash, calibration dataset hash. Automate rebuilds when any input changes.

**Graceful rollout**: Deploy new engines to a canary group first. Compare output quality (perplexity on held-out set) and latency P99 before full rollout.

**Monitoring**: Export TRT-LLM metrics (inflight batch size, KV cache utilization, queue depth) to Prometheus. Alert on KV cache pressure >90% (indicates max_seq_len or batch_size need adjustment).

**Failure modes**: If an engine produces garbage output, the most common causes are (1) calibration data mismatch (calibrated on English, serving multilingual), (2) max_seq_len exceeded at runtime, or (3) memory corruption from OOM during KV cache allocation.

## Summary

TRT-LLM represents the maximum-performance end of the inference engine spectrum. Its compiler-first architecture delivers throughput that framework-based engines cannot match on NVIDIA hardware, at the cost of build complexity and vendor lock-in. The v1.0 PyTorch-first backend reduces the adoption barrier significantly, making compilation accessible without manual graph construction.

Key facts to carry forward:
- XQA kernel achieves 91% HBM bandwidth utilization in decode (near theoretical limit)
- FP8 compilation halves memory footprint with <1% accuracy loss on standard benchmarks
- EAGLE speculative decoding delivers 2.0-2.8x latency reduction for interactive workloads
- Disaggregated serving with NIXL separates prefill and decode onto optimized hardware pools
- Build time ranges from 8 minutes (8B model) to 3 hours (405B model)
- PyTorch-first achieves 95-98% of manual API performance with dramatically less engineering effort

In the next module, we examine NVIDIA Dynamo (Module 05.4), the orchestration layer that manages fleets of TRT-LLM engines, routing requests across disaggregated prefill/decode pools and handling the operational complexity of multi-engine deployments.

---

## References

1. NVIDIA TensorRT-LLM Documentation. https://nvidia.github.io/TensorRT-LLM/
2. NVIDIA. "TensorRT-LLM: A Toolkit for High-Performance LLM Inference." GTC 2024.
3. NVIDIA. "XQA Kernel: Optimizing Decode-Phase Attention for LLMs." TRT-LLM Technical Blog, 2024.
4. Li et al. "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty." ICML 2024.
5. NVIDIA. "NIXL: NVIDIA Inference Transfer Library." https://github.com/ai-dynamo/nixl
6. NVIDIA. "TensorRT-LLM v1.0: PyTorch-First Backend." GTC 2025.
7. NVIDIA. "FP8 Training and Inference." H100 Whitepaper, 2023.
8. Agrawal et al. "Sarathi-Serve: Efficient LLM Inference with Chunked Prefills and Stall-Free Scheduling." OSDI 2024.
9. NVIDIA. "Disaggregated Serving in TensorRT-LLM." AI Dynamo Documentation, 2025.
10. NVIDIA. "Blackwell Architecture: NVFP4 and Next-Generation Inference." GTC 2025.
