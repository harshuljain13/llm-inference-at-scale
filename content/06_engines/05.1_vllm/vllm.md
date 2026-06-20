# vLLM: The Production Standard for Open-Source LLM Serving

From Module 03.1, you know PagedAttention manages KV cache as virtual memory blocks, eliminating fragmentation and enabling near-optimal memory utilization. vLLM is the engine that implements this idea end-to-end: a complete serving system built around PagedAttention, continuous batching, and an iteration-level scheduler that together define the modern standard for open-source LLM inference.

vLLM is the most widely deployed open-source inference engine. Built at UC Berkeley by Woosuk Kwon and collaborators, it introduced PagedAttention and continuous batching to production serving when it launched in 2023. The PagedAttention paper (SOSP 2023) demonstrated 2-4x throughput improvements over state-of-the-art systems by solving the KV cache memory fragmentation problem that plagued earlier serving frameworks. Since then, vLLM has become the default engine behind dozens of inference providers, internal platforms at major tech companies, and the reference implementation that newer engines benchmark against.

This module dissects vLLM's architecture from scheduler to GPU kernel, explains how to configure it for production workloads, and establishes when vLLM is the right choice versus specialized alternatives covered in subsequent modules.

---

## 1. Architecture Overview

vLLM's architecture separates concerns into four core components that communicate through well-defined interfaces. Understanding this separation is essential for debugging performance issues and configuring the engine correctly.

### 1.1 The Four Core Components

**Scheduler.** The scheduler operates at iteration granularity, not request granularity. Every forward pass, it decides which sequences participate in the current batch. This iteration-level scheduling is what enables continuous batching: new requests join the running batch without waiting for existing requests to complete. The scheduler maintains three queues: waiting (new requests), running (active generation), and swapped (preempted to CPU). The scheduling policy runs in microseconds because it only manipulates metadata (block tables, sequence states) rather than tensors.

**KV Cache Manager.** The cache manager implements PagedAttention's block-based allocation. It maintains a block table per sequence that maps logical KV cache positions to physical GPU memory blocks. Blocks are allocated on demand as tokens are generated, freed immediately when sequences complete, and shared across sequences through copy-on-write semantics when prefix caching is enabled. The manager also tracks reference counts for shared blocks and coordinates swap operations between GPU and CPU memory pools.

**Worker.** Each worker owns a GPU and executes model forward passes. In tensor-parallel configurations, multiple workers coordinate through NCCL for all-reduce operations. The worker receives a batch specification from the scheduler (which sequences, which token positions) and returns logits for the next token. Workers are stateless from the scheduler's perspective: they execute whatever batch the scheduler assembles, making horizontal scaling straightforward.

**Tokenizer.** The tokenizer handles encoding input prompts and decoding output tokens. vLLM supports HuggingFace tokenizers natively and adds streaming detokenization that returns partial tokens as they become available, critical for time-to-first-token optimization in chat applications. The tokenizer runs asynchronously on CPU, overlapping with GPU computation to avoid becoming a bottleneck at high request rates.

### 1.2 Request Lifecycle

A request flows through vLLM in these stages:

1. **Arrival**: The API server receives an HTTP request and enqueues it in the scheduler's waiting queue with metadata (priority, max tokens, sampling parameters).
2. **Prefill scheduling**: The scheduler selects the request for prefill, allocating KV cache blocks for the entire prompt length upfront. If insufficient blocks exist, the request remains in the waiting queue.
3. **Prefill execution**: The worker processes all prompt tokens in one forward pass (or multiple chunks if chunked prefill is enabled), populating the KV cache blocks with computed key-value pairs.
4. **Decode iterations**: Each iteration, the scheduler includes this sequence in the running batch. The worker generates one token per iteration. The scheduler allocates one additional block every 16 tokens (at block boundaries).
5. **Completion**: When the sequence hits a stop condition (EOS token, max length, stop string), the scheduler removes it from the running queue and frees all KV cache blocks immediately. The freed blocks become available for new sequences in the very next iteration.

This lifecycle reveals why vLLM achieves high throughput: step 4 repeats across many sequences simultaneously, and the scheduler can inject new prefills between decode steps without stalling the running batch. The GPU never sits idle waiting for a batch to complete.

### 1.3 Memory Layout

vLLM pre-allocates a contiguous GPU memory pool at startup, then carves it into fixed-size blocks (default 16 tokens per block). The total number of blocks determines the maximum concurrent sequences the engine can serve. Understanding this calculation is essential for capacity planning:

```
available_blocks = (total_gpu_memory - model_weights - activation_memory) / block_bytes

block_bytes = block_size * 2 * num_layers * head_dim * num_kv_heads * dtype_bytes
            = 16 * 2 * num_layers * head_dim * num_kv_heads * dtype_bytes
```

For Llama 3.1 8B in FP16 with 16-token blocks on an 80GB A100:
- Model weights: ~16 GB (8B params * 2 bytes)
- Activation scratch space: ~1-2 GB (intermediate tensors during forward pass)
- Remaining for KV cache: ~62 GB
- Block size in bytes: 16 tokens * 2 (K+V) * 32 layers * 128 head_dim * 8 kv_heads * 2 bytes = 2.1 MB per block
- Available blocks: ~29,500 blocks = ~472,000 tokens of KV cache capacity

This means the engine can serve roughly 230 concurrent sequences at 2048 tokens each, or 115 sequences at 4096 tokens, all sharing the same memory pool without fragmentation. Compare this to naive contiguous allocation where a sequence reserving 4096 tokens blocks that entire buffer even if it only generates 200 tokens before stopping.

### 1.4 The Engine Loop

At the highest level, vLLM runs an async engine loop that coordinates all components:

```python
# Simplified engine loop (conceptual)
while True:
    # 1. Scheduler decides what to run this iteration
    scheduler_output = scheduler.schedule()
    # scheduler_output contains: sequences to prefill, sequences to decode,
    # sequences to swap in/out, blocks to free

    # 2. Execute swaps (async, overlapped with compute when possible)
    if scheduler_output.swaps_in:
        cache_engine.swap_in(scheduler_output.swaps_in)
    if scheduler_output.swaps_out:
        cache_engine.swap_out(scheduler_output.swaps_out)

    # 3. Worker executes the batch
    output = worker.execute_model(scheduler_output.scheduled_sequences)

    # 4. Process outputs (sampling, detokenization)
    for seq, logits in zip(scheduler_output.scheduled_sequences, output):
        token = sampler.sample(logits, seq.sampling_params)
        seq.append_token(token)
        if seq.is_finished():
            scheduler.free(seq)
            yield seq.output  # Stream to client
```

This loop executes once per iteration (one token per decode sequence). On an A100 with batch size 128 and Llama 8B, each iteration takes approximately 15-25ms, yielding 40-65 iterations per second.

---

## 2. PagedAttention Implementation

PagedAttention is the foundational innovation that makes vLLM's memory efficiency possible. While Module 03.1 covered the concept, here we examine vLLM's specific implementation choices and their performance implications.

### 2.1 Block Allocation Strategy

vLLM maintains a free block pool initialized at startup. When a sequence needs a new block (either during prefill or at a block boundary during decode), the allocator pops from the free list in O(1). When a sequence completes, all its blocks return to the free list in O(number_of_blocks). This constant-time allocation avoids the fragmentation that plagued contiguous-allocation systems where a 2048-token sequence needed a single contiguous 2048-token buffer.

The block table is a per-sequence data structure mapping logical block indices to physical block IDs:

```python
# Logical view: sequence has tokens at positions 0-47 (3 blocks of 16 tokens each)
# Physical view: blocks allocated non-contiguously from the free pool
block_table = {
    seq_id_42: [physical_block_7, physical_block_103, physical_block_42]
    # Logical block 0 -> physical 7 (tokens 0-15)
    # Logical block 1 -> physical 103 (tokens 16-31)
    # Logical block 2 -> physical 42 (tokens 32-47)
}
```

The custom PagedAttention CUDA kernel reads this block table to gather the correct K and V tensors from scattered physical locations, making non-contiguous allocation completely transparent to the attention computation. The kernel uses the block table as an indirection layer: instead of reading K/V from a contiguous buffer at offset `seq_id * max_len * head_dim`, it reads from `physical_blocks[block_table[seq_id][logical_idx]] + offset_within_block`.

The memory waste from this scheme is bounded: at most one partially-filled block per sequence (average waste = block_size/2 = 8 tokens per sequence). For 200 concurrent sequences, total waste is ~200 * 8 * per_token_kv_bytes, which is negligible compared to the total pool.

### 2.2 Copy-on-Write for Beam Search and Parallel Sampling

When multiple sequences share a common prefix (beam search candidates, parallel samples from the same prompt), vLLM avoids duplicating KV cache blocks through copy-on-write (CoW). This mechanism works identically to CoW in operating system virtual memory:

1. All sequences sharing a prefix point to the same physical blocks in their block tables.
2. A reference counter tracks how many sequences use each physical block.
3. When a sequence needs to write to a shared block (appending a new token to a block with ref_count > 1), the system copies the block to a new physical location, decrements the old block's reference count, and updates that sequence's block table to point to the new copy.

The memory savings are substantial for parallel decoding strategies:

```
Without CoW (naive beam search, beam_width=4, seq_len=2048):
  Memory = 4 * 2048 * per_token_kv_bytes = 4x

With CoW (shared prefix of 1900 tokens, 148 divergent tokens each):
  Shared blocks = ceil(1900/16) = 119 blocks (stored once)
  Divergent blocks = 4 * ceil(148/16) = 4 * 10 = 40 blocks
  Total = 119 + 40 = 159 blocks vs 4 * 128 = 512 blocks (naive)
  Savings: 69% memory reduction
```

For parallel sampling (best-of-N), the savings are even more dramatic because all N samples share the entire prompt's KV cache until divergence.

### 2.3 Prefix Caching

vLLM extends CoW into prefix caching: when multiple requests share a system prompt or few-shot examples, the KV cache for the shared prefix is computed once and reused across all requests that share it. The implementation uses a hash-based lookup keyed on the actual token content of each block:

```
# Prefix caching lookup (simplified)
for block_idx in range(num_blocks_in_prompt):
    block_tokens = prompt_tokens[block_idx*16 : (block_idx+1)*16]
    block_hash = hash(tuple(block_tokens))

    if block_hash in prefix_cache:
        # Reuse existing physical block, increment ref count
        block_table[seq_id].append(prefix_cache[block_hash])
        prefix_cache[block_hash].ref_count += 1
    else:
        # Allocate new block, compute KV, store in cache
        new_block = allocate_block()
        compute_kv(block_tokens, new_block)
        prefix_cache[block_hash] = new_block
        block_table[seq_id].append(new_block)
```

For workloads with repeated system prompts (chat applications with identical system instructions across users), prefix caching eliminates redundant prefill computation entirely. Consider the economics: a 1500-token system prompt at approximately 3-4ms per token for prefill represents 4.5-6 seconds of GPU time per request. With prefix caching, the second and all subsequent requests with that system prompt skip prefill entirely for the cached portion, reducing time-to-first-token from seconds to milliseconds for the prefix portion.

The cache hit rate depends on workload characteristics:
- **Chat with fixed system prompt**: 95%+ hit rate (all requests share the same prefix)
- **Few-shot learning with template**: 80-90% hit rate (shared examples, variable query)
- **RAG with retrieved context**: 20-40% hit rate (retrieved documents vary per query)
- **Unique prompts**: 0% hit rate (no benefit, disable to save hash computation overhead)

### 2.4 Automatic Prefix Caching (APC)

vLLM v0.4+ introduced Automatic Prefix Caching that requires no user intervention or application-level changes. The engine automatically detects shared prefixes across requests by hashing token sequences at block boundaries. Blocks whose token content matches existing cached blocks are shared automatically, making prefix reuse completely transparent to the application layer.

Enable with a single flag: `--enable-prefix-caching`

The cache uses an LRU eviction policy when memory pressure requires reclaiming prefix blocks for active sequences. Eviction is safe because prefix blocks can always be recomputed if needed later. The eviction priority considers both recency and sharing degree: blocks shared by many active sequences are evicted last.

APC interacts with the scheduler's memory management: when the scheduler needs blocks for a new sequence but the free pool is empty, it first evicts unused prefix cache entries before resorting to preemption of running sequences. This creates a natural priority hierarchy: active sequences > recently-used prefix cache > stale prefix cache.

### 2.5 PagedAttention Kernel Performance

The PagedAttention kernel introduces overhead compared to contiguous attention (FlashAttention operating on a dense buffer) because of the indirection through block tables. However, this overhead is small in practice:

- **Prefill phase**: vLLM uses FlashAttention for prefill (prompt tokens are contiguous), so there is zero overhead during the compute-heavy prefill phase.
- **Decode phase**: The PagedAttention kernel handles the scattered reads during decode. The overhead is approximately 3-5% compared to contiguous decode, dominated by the irregular memory access pattern when gathering K/V blocks.

This design choice is deliberate: prefill is compute-bound (benefits from FlashAttention's optimized fused kernels), while decode is memory-bound (the 3-5% overhead from scattered reads is dwarfed by the memory bandwidth bottleneck). The memory savings from paging (supporting 2-4x more concurrent sequences) far outweigh the minor kernel overhead.

---

## 3. Continuous Batching: The Scheduler in Detail

Traditional static batching waits for all sequences in a batch to complete before starting new ones. This wastes GPU cycles: short sequences finish early and their slots sit idle until the longest sequence completes. The waste is proportional to the variance in output lengths within a batch. For a batch where the shortest output is 10 tokens and the longest is 500, the GPU utilization for short sequences drops to 2% (10/500).

Continuous batching solves this by allowing the scheduler to add and remove sequences at every iteration, keeping all batch slots productive at all times.

### 3.1 Iteration-Level Scheduling

Every forward pass (every ~20ms), vLLM's scheduler executes this decision loop:

1. **Check running sequences**: Remove any that have completed (hit EOS, max tokens, or stop string). Free their blocks immediately.
2. **Process completions**: For completed sequences, trigger the output callback (stream the final token to the client, close the SSE connection).
3. **Check for freed capacity**: Count available blocks after completions.
4. **Attempt to schedule waiting requests**: For each waiting request (in FCFS or priority order), estimate the blocks needed for prefill (prompt_length / block_size). If enough free blocks exist, allocate them and move the request from waiting to running.
5. **Check memory pressure**: If the running set collectively needs more blocks than available for their next decode step, do not schedule new requests even if blocks appear free (reserve headroom for decode growth).
6. **Preemption (if critical)**: If running sequences cannot continue because no blocks remain for their next token, preempt the lowest-priority sequence by swapping or recomputing its KV cache.

The scheduler's decision takes microseconds because it operates on metadata only (block counts, sequence states, priority values). No tensor operations occur during scheduling.

### 3.2 Preemption and Swapping

When GPU memory is exhausted and running sequences cannot allocate their next block, vLLM preempts sequences rather than rejecting new requests or crashing. Preemption has two modes:

**Recompute mode** (default for sequences with short prefill, < 512 tokens):
- Discard the sequence's KV cache blocks entirely (free them back to the pool)
- Move the sequence to the waiting queue
- When capacity frees up, re-prefill from scratch
- Cost: redundant prefill computation
- Benefit: zero CPU memory usage, zero PCIe bandwidth consumption

**Swap mode** (default for sequences with long prefill, >= 512 tokens):
- Copy the sequence's KV cache blocks from GPU to pre-allocated CPU memory via PCIe
- Move the sequence to the swapped queue
- When GPU blocks free up, copy blocks back from CPU and resume decode from where it stopped
- Cost: PCIe bandwidth (bidirectional, ~25 GB/s on PCIe Gen4 x16)
- Benefit: avoids expensive re-prefill for long contexts

The crossover point between recompute and swap depends on prefill length relative to PCIe bandwidth. For Llama 8B on A100 (PCIe Gen4):
- Recompute 512 tokens: ~2ms (GPU compute is fast for short prefills)
- Swap 512 tokens of KV cache: ~0.5ms (small data transfer)
- Recompute 4096 tokens: ~16ms
- Swap 4096 tokens of KV cache: ~4ms

For sequences longer than ~256 tokens, swap is almost always cheaper than recompute.

### 3.3 Chunked Prefill

Long prompts create a scheduling problem: a 32K-token prefill monopolizes the GPU for hundreds of milliseconds (32K tokens at ~4ms/1K tokens = ~128ms), blocking all decode iterations and spiking latency for every running sequence. Users waiting for their next token see a 128ms stall because the GPU was busy computing attention for someone else's long prompt.

Chunked prefill solves this by splitting long prefills into fixed-size chunks (default 512 tokens) that interleave with decode steps:

```
Without chunked prefill (32K prompt arrives):
  Iteration 1: [32K prefill for seq A] -> 128ms, all other seqs stalled
  Iteration 2: [decode for all seqs] -> normal

With chunked prefill (512-token chunks):
  Iteration 1: [prefill chunk 0-511 for seq A] + [decode for seqs B,C,D] -> ~22ms
  Iteration 2: [prefill chunk 512-1023 for seq A] + [decode for seqs B,C,D] -> ~22ms
  ...
  Iteration 64: [prefill final chunk for seq A] + [decode for seqs B,C,D] -> ~22ms
```

The tradeoff: seq A's time-to-first-token increases (64 iterations * 22ms = 1.4s vs 128ms without chunking), but every other sequence maintains consistent ~22ms inter-token latency throughout. For interactive applications where predictable latency matters more than individual TTFT, this tradeoff is correct.

Enable with `--enable-chunked-prefill`. Adjust chunk size with `--max-num-batched-tokens` (the maximum tokens processed per iteration across all sequences, including both prefill chunks and decode tokens).

### 3.4 Priority Scheduling

vLLM supports priority-based scheduling where requests carry numeric priority levels. The scheduler uses priority to make three decisions:

1. **Scheduling order**: Higher-priority requests in the waiting queue are scheduled before lower-priority ones, even if lower-priority requests arrived first.
2. **Preemption order**: Under memory pressure, the lowest-priority running sequence is preempted first.
3. **Swap-in order**: When GPU memory frees up, the highest-priority swapped sequence is restored first.

Production use cases for priority scheduling:
- **Tiered pricing**: Premium users get priority 10, free tier gets priority 1. Under load, free-tier requests queue while premium requests cut ahead.
- **Request type differentiation**: Interactive chat (priority 10) over batch summarization (priority 1). Short expected outputs (priority 8) over long generation (priority 3).
- **Deadline-aware**: Requests approaching their SLA timeout get priority boosted to avoid violations.

Priority is set per-request via the API (custom header or request body field) and requires no engine restart to change.

---

## 4. Configuration Guide

vLLM exposes dozens of configuration parameters through its CLI and Python API. Most have sensible defaults, but production deployments require tuning a handful of critical knobs. This section groups parameters by their primary effect and explains when to change each from its default.

### 4.1 Memory Parameters

**`--gpu-memory-utilization`** (default: 0.90)
Fraction of total GPU memory vLLM claims for the KV cache pool after model weights are loaded. The remaining percentage is reserved for CUDA context overhead, activation memory spikes during prefill, and safety margin against OOM. Tuning guidance:
- Dedicated inference machine, single model: increase to 0.93-0.95
- Shared GPU (monitoring, other processes): decrease to 0.80-0.85
- Large batch sizes with long prefills: keep at 0.90 (prefill activations spike)
- Never set above 0.97: CUDA runtime needs ~2-3% for internal allocations

**`--max-model-len`** (default: model's max position embeddings)
Maximum sequence length the engine will accept. This parameter directly affects capacity planning because the scheduler must reserve blocks for sequences that could grow up to this length. If your application never produces outputs longer than 4096 tokens, setting this to 4096 even on a 128K-capable model dramatically increases concurrent capacity:
- 128K max-model-len: worst case reserves 8000 blocks per sequence
- 4096 max-model-len: worst case reserves 256 blocks per sequence
- Result: 31x more potential concurrent sequences

**`--block-size`** (default: 16)
Tokens per KV cache block. Smaller blocks reduce internal fragmentation (less wasted memory in partially-filled blocks) but increase the block table size and indirection overhead. Larger blocks reduce overhead but waste more memory. The default of 16 is empirically optimal for most workloads. Only change this if you have unusual sequence length distributions (e.g., all sequences are exactly 7 tokens: block-size=8 wastes less).

**`--swap-space`** (default: 4 GB)
CPU memory reserved for swapped KV cache blocks. The engine pre-allocates this at startup. Increase if:
- Your workload has many long concurrent sequences that trigger frequent preemption
- You see "No swap space available" errors in logs
- You want to avoid recompute-mode preemption for long sequences

For most deployments, 4 GB is sufficient. Each swapped sequence for Llama 8B at 2048 tokens uses ~34 MB of swap space (2048 tokens * 2 * 32 layers * 128 * 8 * 2 bytes / (1024^2)).

### 4.2 Throughput Parameters

**`--max-num-seqs`** (default: 256)
Maximum sequences in a single running batch. This is the primary throughput lever:
- Offline batch processing (latency irrelevant): set to 512-1024
- Interactive chat (latency sensitive): set to 64-128
- Mixed workload: set to 256 (default) and rely on priority scheduling to protect latency-sensitive requests

Higher values increase throughput but also increase per-token latency because more sequences share the same forward pass. The relationship is sublinear: doubling batch size from 128 to 256 increases throughput by ~60-70% (not 100%) due to memory bandwidth saturation.

**`--max-num-batched-tokens`** (default: computed from max-num-seqs * max-model-len)
Total tokens across all sequences in one iteration. This caps the compute work per forward pass. Set explicitly when:
- Using chunked prefill (controls chunk size: max-num-batched-tokens - running_decode_tokens = available prefill chunk)
- You want to bound worst-case iteration latency regardless of batch composition
- You observe latency spikes when many long prefills arrive simultaneously

**`--enable-prefix-caching`**
Enables Automatic Prefix Caching. Always enable for:
- Chat applications with system prompts (95%+ cache hit rate)
- Few-shot inference with fixed examples
- Multi-turn conversations (prior turns are a shared prefix)

Overhead is minimal (one hash per block during scheduling). Disable only for workloads where every prompt is unique (zero reuse) and you want to eliminate even the hash computation overhead.

**`--enable-chunked-prefill`**
Enables chunked prefill for long prompts. Enable when:
- Serving a mix of long-context (>4K) and short-context requests
- P99 inter-token latency matters more than individual TTFT
- Using models with 32K+ context that receive variable-length inputs

### 4.3 Latency Parameters

**`--disable-log-stats`**
Disables per-iteration statistics logging to stdout. In high-throughput deployments (>1000 req/s), the string formatting and I/O overhead of per-iteration logging is measurable (~1-2% throughput cost). Always disable in production; use Prometheus metrics endpoint instead.

**`--disable-log-requests`**
Disables individual request logging. At high QPS, request logging generates significant I/O. Disable in production; sample with external observability tools instead.

**`--response-role`** (default: "assistant")
Sets the role field in chat completion responses. No performance impact, but important for API compatibility with downstream consumers.

### 4.4 Model-Specific Parameters

**`--quantization`** (choices: awq, gptq, squeezellm, fp8, marlin, bitsandbytes)
Specifies the quantization format for model weights. Each format has different requirements:
- **awq/gptq**: Requires pre-quantized model checkpoint (quantization happens offline)
- **fp8**: On-the-fly quantization on H100/L40S GPUs with FP8 Tensor Cores. No pre-quantization needed.
- **marlin**: Optimized 4-bit kernel that accelerates AWQ/GPTQ inference on A100/H100. Requires pre-quantized model + Marlin-compatible format.
- **bitsandbytes**: Dynamic quantization via bitsandbytes library. Easiest to set up but lowest performance.

**`--dtype`** (choices: auto, float16, bfloat16, float32)
Compute and storage precision for model weights and KV cache:
- **auto** (default): Uses the model's training precision from config.json
- **bfloat16**: Preferred for Llama-family models (trained in BF16). Better numerical range than FP16, avoids overflow in attention for very long sequences.
- **float16**: Works for most models. May require attention scaling adjustments for sequences >16K tokens due to FP16's limited exponent range.

**`--tensor-parallel-size`** (TP)
Number of GPUs for tensor parallelism. Set to the minimum GPUs needed to fit model weights in memory:
- Llama 8B FP16 (16 GB): TP=1 on A100-80GB
- Llama 70B FP16 (140 GB): TP=2 on A100-80GB (minimum), TP=4 for more KV cache headroom
- Llama 405B FP16 (810 GB): TP=8 on A100-80GB (fills 8 GPUs almost entirely with weights)

**`--pipeline-parallel-size`** (PP)
Number of stages for pipeline parallelism. Use when you have more GPUs than needed for TP and want to increase throughput by overlapping pipeline stages. PP=2 with TP=4 uses 8 GPUs with 4-way tensor parallelism across 2 pipeline stages. Pipeline parallelism adds latency per token (pipeline bubble) but increases aggregate throughput.

**`--enforce-eager`**
Disables CUDA graph capture. Use during debugging or when the model has dynamic control flow that CUDA graphs cannot capture. In production, always allow CUDA graphs (do not set this flag) for 10-20% decode throughput improvement from reduced kernel launch overhead.

---

## 5. Performance Tuning

### 5.1 Throughput Optimization

The primary throughput lever is batch size (controlled by `--max-num-seqs`). vLLM's throughput scales with batch size because decode is memory-bandwidth-bound: adding more sequences to the batch amortizes the cost of loading model weights from HBM. Each sequence adds only its KV cache reads to the memory traffic, while the weight-loading cost is shared.

Empirical throughput scaling for Llama 3.1 8B on A100-80GB (output length 256, input length 512):

```
max-num-seqs=16:    ~800 tokens/sec    (GPU utilization ~15%)
max-num-seqs=32:    ~1,500 tokens/sec  (GPU utilization ~25%)
max-num-seqs=64:    ~2,800 tokens/sec  (GPU utilization ~45%)
max-num-seqs=128:   ~4,800 tokens/sec  (GPU utilization ~70%)
max-num-seqs=256:   ~7,500 tokens/sec  (GPU utilization ~88%)
max-num-seqs=512:   ~9,000 tokens/sec  (GPU utilization ~93%, diminishing returns)
max-num-seqs=1024:  ~9,800 tokens/sec  (GPU utilization ~96%, compute-bound)
```

The diminishing returns at 512+ indicate the transition from memory-bound to compute-bound operation. Beyond this point, adding more sequences increases latency without proportional throughput gain because the GPU's math units are fully utilized.

To find your workload's optimal batch size: start at max-num-seqs=256, measure throughput, then double. When doubling yields <15% throughput gain, you have found the saturation point.

### 5.2 Speculative Decoding

Speculative decoding uses a small draft model to propose N tokens that the target model verifies in a single forward pass. When proposals are accepted (draft matches what the target would have generated), multiple tokens are produced per iteration. Rejection is free: the target model's logits from the verification pass provide the correct next token regardless.

Configuration:
```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --speculative-model meta-llama/Llama-3.1-8B-Instruct \
  --num-speculative-tokens 5 \
  --speculative-max-model-len 4096
```

Expected speedup depends on acceptance rate:
- 90% acceptance (highly predictable text, code completion): 2.0-2.5x tokens/sec
- 75% acceptance (general chat): 1.5-1.8x tokens/sec
- 50% acceptance (creative writing, diverse outputs): 1.1-1.3x (barely worth the overhead)

Key tuning decisions:
- **Draft model size**: Smaller drafts are faster to run but have lower acceptance rates. 8B drafting for 70B typically achieves 70-85% acceptance on English text.
- **Number of speculative tokens** (`--num-speculative-tokens`): Higher values amortize verification cost but reduce overall acceptance probability (must accept ALL N tokens for full benefit). Sweet spot is 3-5 for most workloads. Beyond 7, acceptance probability drops exponentially.
- **When NOT to use**: At high batch sizes (>128), the draft model's GPU time competes with serving more sequences via continuous batching. Speculative decoding shines at low batch sizes where the GPU is underutilized during decode.

### 5.3 LoRA Adapter Serving

vLLM serves multiple LoRA adapters simultaneously from a single base model, enabling multi-tenant fine-tuned model serving without loading separate model copies per adapter:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --lora-modules customer-support=/path/to/lora1 code-review=/path/to/lora2 \
  --max-loras 8 \
  --max-lora-rank 64
```

How it works internally:
1. Base model weights are loaded once and shared across all requests.
2. LoRA adapter weights (A and B matrices) are loaded into GPU memory separately.
3. During forward pass, the engine computes `base_output + alpha * (B @ A @ input)` for sequences using that adapter.
4. Sequences using different adapters can coexist in the same batch because LoRA computation is per-sequence.

Memory overhead per adapter (rank 64, Llama 8B, all linear layers adapted):
- A matrices: 64 * hidden_dim * num_layers * num_adapted_layers * dtype_bytes
- For Llama 8B: ~100-150 MB per adapter (negligible vs 16 GB base model)

Requests specify their adapter via the `model` field in the OpenAI-compatible API:
```json
{"model": "customer-support", "messages": [...]}
```

This enables routing different users or applications to different fine-tuned variants through a single endpoint, dramatically simplifying multi-model deployment.

### 5.4 Quantization Performance Tradeoffs

Quantization reduces memory footprint and can increase throughput by fitting more sequences in the same GPU memory (more batching headroom) and by reducing memory bandwidth consumption per token:

| Method | Bits | Weight Memory | Throughput vs FP16 | Quality Impact | Hardware Requirement |
|--------|------|--------------|-------------------|----------------|---------------------|
| FP16 (baseline) | 16 | 100% | 1.0x | None | Any GPU |
| BF16 | 16 | 100% | 1.0x | None | Ampere+ (A100, H100) |
| FP8 | 8 | 50% | 1.3-1.5x | Negligible | H100, L40S (FP8 TC) |
| AWQ | 4 | 25% | 1.8-2.5x batch capacity | <1% perplexity | Any GPU |
| GPTQ | 4 | 25% | 1.8-2.5x batch capacity | <1% perplexity | Any GPU |
| Marlin kernel | 4 | 25% | Best 4-bit perf | <1% perplexity | A100, H100 |
| GGUF (via integration) | 2-8 | Variable | Lower than native | Varies | CPU + GPU |

Production recommendations by hardware:
- **H100/H200**: Use FP8 (hardware-native, no pre-quantization step, minimal quality loss)
- **A100-80GB**: Use AWQ with Marlin kernels (best 4-bit decode throughput)
- **A100-40GB**: Use AWQ/GPTQ (essential to fit 70B models on 2 GPUs instead of 4)
- **L40S/A10G**: Use AWQ (widest compatibility, fits larger models on smaller GPUs)

### 5.5 CUDA Graphs

vLLM captures CUDA graphs for the decode forward pass to eliminate kernel launch overhead. Without CUDA graphs, each of the hundreds of kernel launches per iteration incurs ~5-10us of CPU-side overhead, totaling 0.5-1ms per iteration. CUDA graphs record the entire kernel sequence once and replay it in ~50us, yielding 10-20% throughput improvement for decode-heavy workloads.

CUDA graphs are captured at startup for each unique batch size (powers of 2 up to max-num-seqs). The capture takes 30-60 seconds for large models. Disable with `--enforce-eager` only during development/debugging.

Limitations: CUDA graphs cannot handle dynamic shapes. vLLM works around this by capturing graphs for discrete batch sizes and padding to the next captured size. If your workload has a very stable batch size, you can improve efficiency by setting max-num-seqs to a power of 2 to minimize padding waste.

---

## 6. Deployment Patterns

### 6.1 Docker

The official vLLM Docker image includes all dependencies, CUDA drivers, and the OpenAI-compatible API server:

```bash
docker run --gpus all -p 8000:8000 \
  -v /path/to/models:/models \
  vllm/vllm-openai:v0.5.0 \
  --model /models/Llama-3.1-8B-Instruct \
  --gpu-memory-utilization 0.92 \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --enable-prefix-caching
```

Production Docker best practices:
- Pin a specific version tag (never use `latest` in production)
- Mount model weights from a local volume or pull from S3/GCS at startup to avoid baking 16+ GB weights into the image
- Set `--disable-log-stats` and `--disable-log-requests` to reduce I/O overhead
- Use `--uvicorn-log-level warning` to silence per-request HTTP logs
- Set container resource limits: `--shm-size=8g` for shared memory (used by NCCL in multi-GPU setups)

### 6.2 Kubernetes

vLLM deploys on Kubernetes with GPU node pools. The key consideration is GPU scheduling: each vLLM pod claims exclusive GPU(s) through the device plugin.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama-8b
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vllm-llama-8b
  template:
    metadata:
      labels:
        app: vllm-llama-8b
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.5.0
        args:
        - "--model"
        - "meta-llama/Llama-3.1-8B-Instruct"
        - "--gpu-memory-utilization"
        - "0.92"
        - "--max-num-seqs"
        - "128"
        - "--enable-prefix-caching"
        - "--disable-log-stats"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
          requests:
            nvidia.com/gpu: 1
            memory: 24Gi
        ports:
        - containerPort: 8000
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120  # Model loading time
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 180
          periodSeconds: 30
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-A100-SXM4-80GB
```

Scaling strategies:
- **HPA on GPU utilization**: Scale when `nvidia_gpu_utilization > 85%` sustained for 2 minutes
- **HPA on queue depth**: Scale when `vllm:num_requests_waiting > 50` sustained for 30 seconds (more responsive than GPU util)
- **KEDA with Prometheus**: Custom metric scaling using vLLM's Prometheus endpoint
- **KAI Scheduler** (Module 07.7): Intelligent GPU time-sharing across vLLM replicas when GPUs are fractionally utilized

### 6.3 Ray Serve Integration

For multi-GPU deployments requiring elastic autoscaling and multi-model routing, vLLM integrates with Ray Serve:

```python
from vllm import LLM, SamplingParams
from ray import serve
import ray

@serve.deployment(
    ray_actor_options={"num_gpus": 2},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 8,
        "target_num_ongoing_requests_per_replica": 20,
    },
)
class VLLMDeployment:
    def __init__(self):
        self.llm = LLM(
            model="meta-llama/Llama-3.1-70B-Instruct",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.92,
            max_num_seqs=128,
            enable_prefix_caching=True,
        )

    async def generate(self, request):
        prompt = request.query_params.get("prompt")
        params = SamplingParams(temperature=0.7, max_tokens=512)
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text

app = VLLMDeployment.bind()
```

Ray Serve adds capabilities beyond standalone vLLM:
- **Autoscaling**: Scale replicas based on request queue depth with configurable thresholds. Ray handles GPU allocation across the cluster.
- **Multi-model composition**: Route requests to different model deployments based on model field, enabling a single gateway for multiple vLLM instances.
- **Zero-downtime updates**: Deploy new model versions through deployment graph versioning. Ray drains old replicas while spinning up new ones.
- **Heterogeneous hardware**: Mix A100 and H100 nodes in the same cluster. Ray schedules deployments to appropriate hardware based on resource requirements.

### 6.4 SageMaker Deployment

AWS SageMaker hosts vLLM through either the HuggingFace DLC (managed) or custom containers (full control):

```python
from sagemaker.huggingface import HuggingFaceModel

model = HuggingFaceModel(
    model_data="s3://my-bucket/llama-3.1-8b/model.tar.gz",
    role=role,
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi2.0-gpu-py310-cu121-ubuntu22.04",
    env={
        "HF_MODEL_ID": "/opt/ml/model",
        "SM_NUM_GPUS": "1",
        "MAX_INPUT_LENGTH": "4096",
        "MAX_TOTAL_TOKENS": "8192",
        "MAX_BATCH_PREFILL_TOKENS": "8192",
    }
)
predictor = model.deploy(
    instance_type="ml.g5.2xlarge",
    initial_instance_count=1,
    endpoint_name="llama-8b-vllm"
)
```

For direct vLLM (not TGI) on SageMaker:
1. Build a custom container with vLLM's OpenAI server
2. Expose port 8080 (SageMaker's default inference port)
3. Implement `/ping` health check endpoint (SageMaker requirement)
4. Use `/invocations` or configure custom routing to vLLM's `/v1/completions`
5. Set `SAGEMAKER_PROGRAM` environment variable to your serving script

Instance selection guide:
- **ml.g5.2xlarge** (1x A10G 24GB): Llama 8B AWQ, light traffic
- **ml.g5.12xlarge** (4x A10G): Llama 70B AWQ with TP=4
- **ml.p4d.24xlarge** (8x A100 80GB): Llama 70B FP16 with TP=2, high throughput
- **ml.p5.48xlarge** (8x H100 80GB): Llama 405B or maximum throughput scenarios

---

## 7. Observability and Monitoring

### 7.1 Built-in Prometheus Metrics

vLLM exposes a comprehensive metrics endpoint at `/metrics` in Prometheus format. These metrics are the foundation for production monitoring and autoscaling:

**Capacity metrics** (are we running out of room?):
| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `vllm:num_requests_running` | Active sequences in current batch | Info only |
| `vllm:num_requests_waiting` | Queued requests awaiting scheduling | > 50 sustained 60s |
| `vllm:num_requests_swapped` | Preempted sequences in CPU swap | > 20 |
| `vllm:gpu_cache_usage_perc` | KV cache pool utilization | > 95% |
| `vllm:cpu_cache_usage_perc` | Swap space utilization | > 80% |
| `vllm:num_preemptions_total` | Cumulative preemption count | Rate > 5/min |

**Performance metrics** (how fast are we serving?):
| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `vllm:avg_generation_throughput_toks_per_s` | Aggregate output throughput | < baseline * 0.7 |
| `vllm:avg_prompt_throughput_toks_per_s` | Aggregate prefill throughput | < baseline * 0.7 |
| `vllm:e2e_request_latency_seconds` | End-to-end latency histogram | P99 > SLA |
| `vllm:time_to_first_token_seconds` | TTFT histogram | P99 > 5s |
| `vllm:time_per_output_token_seconds` | Inter-token latency histogram | P99 > 100ms |

**Model metrics** (what is the engine doing?):
| Metric | Description |
|--------|-------------|
| `vllm:num_generation_tokens_total` | Total tokens generated |
| `vllm:num_prompt_tokens_total` | Total prompt tokens processed |
| `vllm:request_success_total` | Successful completions |
| `vllm:request_failure_total` | Failed requests (OOM, timeout) |

### 7.2 Grafana Dashboard Design

A production vLLM dashboard should have four panels:

1. **Traffic panel**: Requests/sec (running + waiting + completed), split by success/failure
2. **Capacity panel**: GPU cache utilization, swap utilization, batch size over time
3. **Latency panel**: TTFT P50/P95/P99, inter-token P50/P95/P99, E2E P50/P95/P99
4. **Throughput panel**: Tokens/sec (generation + prompt), requests completed/sec

### 7.3 Key Alerts for Production

Set these alerts to catch issues before they impact users:

| Condition | Severity | Action |
|-----------|----------|--------|
| `num_requests_waiting > 50` for 60s | Warning | Scale up replicas |
| `num_requests_waiting > 200` for 30s | Critical | Immediate scale-up, page on-call |
| `gpu_cache_usage_perc > 95%` for 120s | Warning | Reduce max-num-seqs or scale |
| `num_preemptions rate > 10/min` | Warning | Memory pressure, reduce load |
| `time_to_first_token P99 > 5s` | Warning | Enable chunked prefill or reduce max-model-len |
| `request_failure_total rate > 1/min` | Critical | Investigate OOM or model errors |
| `generation_throughput < baseline * 0.5` | Critical | GPU thermal throttling or hardware issue |

---

## 8. When to Choose vLLM

vLLM is the right choice when:

**You need the widest model support.** vLLM supports more model architectures than any other open-source engine: Llama, Mistral, Mixtral, Qwen, Falcon, GPT-NeoX, Phi, Gemma, Command-R, DeepSeek, StarCoder, and dozens more. New models are typically supported within days of release because the community is large enough to contribute implementations quickly.

**You want a battle-tested production system.** Hundreds of companies run vLLM in production today, from startups to large enterprises. Edge cases, race conditions, and memory leaks are found and fixed by a community of thousands of contributors. The issue tracker contains solutions to problems you have not encountered yet.

**Your workload is general-purpose.** Mixed batch and streaming requests, variable sequence lengths (100 to 32K tokens in the same deployment), multiple sampling strategies (greedy, beam search, top-p/top-k), and diverse use cases (chat, completion, embedding) all work through a single deployment.

**You need LoRA multi-tenancy.** Serve 8+ fine-tuned variants from one base model without duplicating 16+ GB of weights per variant. No other engine handles multi-LoRA as seamlessly.

**You want ecosystem integration.** First-class support for HuggingFace models (load any model hub checkpoint directly), Ray Serve (elastic autoscaling), OpenAI-compatible API (drop-in replacement for applications built against GPT-4), Prometheus (standard observability), and major cloud platforms.

**You prioritize operational simplicity.** `pip install vllm && vllm serve model-name` gets you a production-grade server in one command. No model conversion, no compilation step, no custom config files.

### When vLLM is NOT the best choice:

**Maximum decode throughput on NVIDIA hardware is the only metric.** TensorRT-LLM (Module 05.2) achieves 20-40% higher raw throughput through aggressive kernel fusion, custom CUDA kernels, and compile-time graph optimization. If you control the hardware, have engineering resources for the build/convert pipeline, and are optimizing solely for tokens/dollar, TensorRT-LLM wins.

**You need state-of-the-art prefix caching for complex multi-turn workloads.** SGLang (Module 05.3) pioneered RadixAttention, which maintains a radix tree of all cached prefixes and achieves more granular sharing than vLLM's block-hash approach. For applications with deep multi-turn histories and branching conversation trees, SGLang's cache hit rate can be 10-20% higher.

**Edge or mobile deployment.** vLLM targets server GPUs with 16+ GB VRAM. For edge inference on consumer GPUs, laptops, or mobile devices, llama.cpp (Module 05.5) provides CPU/Metal/Vulkan inference with aggressive quantization (2-4 bit) that vLLM does not support.

**Disaggregated prefill and decode.** If your workload has extreme prefill/decode imbalance (very long prompts but short outputs, or vice versa), systems that separate prefill and decode onto different GPU pools (Splitwise architecture, Mooncake, DistServe) achieve better resource utilization than vLLM's co-located approach. vLLM is adding disaggregated serving support but it remains experimental as of mid-2025.

---

## 9. Comparison with Other Engines

Understanding vLLM's position relative to alternatives helps you make informed deployment decisions. Module 05.2-05.5 cover each alternative in depth; this table provides the decision-making summary:

| Dimension | vLLM | TensorRT-LLM | SGLang | llama.cpp |
|-----------|------|--------------|--------|-----------|
| **Primary strength** | Widest compatibility | Maximum throughput | Best prefix caching | Edge/CPU inference |
| **Model support** | 50+ architectures | NVIDIA-optimized subset | 30+ architectures | GGUF ecosystem |
| **Setup complexity** | pip install | Build + convert (hours) | pip install | cmake build |
| **Raw throughput** | Good (baseline) | Best (+20-40%) | Good-Great | Low (CPU-bound) |
| **Prefix caching** | APC (hash-based) | Manual/limited | RadixAttention (best) | None |
| **Multi-LoRA** | Native, seamless | Limited support | Native | Not supported |
| **Community size** | Largest (25K+ stars) | NVIDIA-maintained | Growing fast (10K+) | Massive (50K+ stars) |
| **API** | OpenAI-compatible | Triton Inference Server | OpenAI-compatible | OpenAI-compatible |
| **Speculative decoding** | Supported | Supported | Supported | Supported |
| **Quantization** | AWQ/GPTQ/FP8/Marlin | FP8/INT8/INT4 (native) | AWQ/GPTQ/FP8 | GGUF Q2-Q8 |
| **Hardware** | NVIDIA, AMD (experimental) | NVIDIA only | NVIDIA, AMD | CPU, Metal, Vulkan, CUDA |
| **Typical user** | Most production deployments | Performance-critical, NVIDIA-only | Research, complex pipelines | Local/edge, privacy-first |

The trend in 2024-2025 is convergence: vLLM adopts SGLang's ideas (chunked prefill, improved prefix caching, disaggregated serving), SGLang adopts vLLM's stability patterns and broader model support, and TensorRT-LLM remains the performance ceiling for NVIDIA hardware at the cost of developer experience. The competitive landscape pushes all engines toward feature parity, with differentiation increasingly coming from community size, operational maturity, and ecosystem integration rather than fundamental architectural differences.

---

## 10. Mental Model

Think of vLLM as the Linux of inference engines. It is not always the fastest at any single benchmark. TensorRT-LLM will beat it on raw NVIDIA throughput by 20-40%. SGLang may handle complex prefix patterns more efficiently. llama.cpp runs on hardware vLLM cannot touch. But vLLM offers the widest model support, the largest community finding and fixing issues, the most predictable production behavior, and the lowest barrier from zero to production.

When you are unsure which engine to choose, start with vLLM. It establishes a strong baseline that specialized engines must justify their additional complexity against. Most production workloads never need to leave vLLM because its combination of PagedAttention, continuous batching, and automatic prefix caching already captures 80-90% of the theoretical throughput improvement over naive serving.

The key insight is not about raw performance. vLLM won adoption by being fast enough across every dimension while being the easiest to deploy, debug, and operate. In production systems, operational simplicity compounds into reliability. A system you can deploy in 5 minutes, monitor with standard tools, debug with readable logs, and scale with standard Kubernetes patterns is worth more than 20% additional throughput from a system that requires a week of setup, custom monitoring, and specialized debugging expertise.

Start with vLLM. Only move to alternatives when you have concrete evidence (benchmarks on your workload, with your models, at your scale) that the alternative's advantage justifies its additional operational complexity. For most teams, that evidence never materializes.

---

## References

1. Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., and Stoica, I. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." *Proceedings of the 29th Symposium on Operating Systems Principles (SOSP 2023)*.
2. vLLM Documentation. https://docs.vllm.ai/
3. vLLM GitHub Repository. https://github.com/vllm-project/vllm
4. Yu, G., Jeong, J., Kim, G., Kim, S., and Chun, B. (2022). "Orca: A Distributed Serving System for Transformer-Based Generative Models." *OSDI 2022*. (Continuous batching concept that vLLM implements and extends.)
5. Zheng, L., Yin, L., Xie, Z., et al. (2024). "SGLang: Efficient Execution of Structured Language Model Programs." (RadixAttention comparison system.)
6. NVIDIA TensorRT-LLM Documentation. https://nvidia.github.io/TensorRT-LLM/ (Performance ceiling comparison.)
7. Agrawal, A., et al. (2024). "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve." *OSDI 2024*. (Chunked prefill technique adopted by vLLM.)
