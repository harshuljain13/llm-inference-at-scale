# Module 4: Inference Engines Deep Dive

> The difference between inference engines isn't just performance—it's architecture. vLLM, SGLang, and TensorRT-LLM make fundamentally different tradeoffs between flexibility, performance, and complexity. Understanding these tradeoffs lets you pick the right tool and tune it correctly.

---

## Learning Objectives

By the end of this module, you will:

- Understand vLLM's internal architecture: how the scheduler, block manager, and workers coordinate
- Explain why SGLang's RadixAttention outperforms vLLM for certain workloads (and when it doesn't)
- Know exactly what TensorRT-LLM compilation does and why it's worth the complexity
- Make informed engine selection decisions based on your specific workload characteristics
- Tune the 6 critical vLLM knobs with understanding, not guesswork

---

## The Engine Landscape

Before diving into internals, let's understand what each engine optimizes for:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ENGINE DESIGN PHILOSOPHIES                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   vLLM: "Make the common case fast with minimal setup"              │
│   ─────────────────────────────────────────────────────────────    │
│   • PagedAttention for memory efficiency                            │
│   • Continuous batching for throughput                              │
│   • Python scheduler, CUDA kernels for hot paths                    │
│   • Philosophy: 80% of optimal performance with 20% of effort       │
│                                                                     │
│   SGLang: "Optimize for LLM programs, not just single calls"        │
│   ─────────────────────────────────────────────────────────────    │
│   • RadixAttention for prefix sharing across requests               │
│   • Native structured output with constrained decoding              │
│   • Compiler for multi-step LLM programs                            │
│   • Philosophy: Maximize reuse across related requests              │
│                                                                     │
│   TensorRT-LLM: "Maximum performance, pay the compilation cost"     │
│   ─────────────────────────────────────────────────────────────    │
│   • Ahead-of-time compilation to TensorRT engines                   │
│   • Kernel fusion, quantization, graph optimization                 │
│   • C++ runtime with minimal Python overhead                        │
│   • Philosophy: Squeeze every FLOP, accept complexity               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

\*\*Insight #1: Engine choice is a tradeoff between iteration speed and rhare structure.

---

## vLLM Architecture: The Deep Dive

### The Request Lifecycle

Let's trace exactly what happens when a request hits vLLM:

**Insight #1: Engine choice is a tradeoff between iteration speed and runtime performance.** vLLM lets you change models in seconds. TensorRT-LLM requires hours of compilation but runs 30-50% faster. SGLang sits in between, optimizing for workloads where requests share structure.

---

## vLLM Architecture: The Deep Dive

### The Request Lifecycle

Let's trace exactly what happens when a request hits vLLM:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    vLLM REQUEST LIFECYCLE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   1. API LAYER (FastAPI)                                            │
│      ─────────────────────────────────────────────────────────     │
│      POST /v1/completions                                           │
│      │                                                              │
│      ▼                                                              │
│      Tokenize prompt → Create SequenceGroup                         │
│      │                                                              │
│      ▼                                                              │
│      Add to AsyncLLMEngine.add_request() queue                      │
│                                                                     │
│   2. SCHEDULER (Python, runs every step)                            │
│      ─────────────────────────────────────────────────────────     │
│      │                                                              │
│      ├─► Check waiting queue for new requests                       │
│      │   └─► Can we allocate KV cache blocks? (Block Manager)       │
│      │       └─► Yes: Move to running, allocate blocks              │
│      │       └─► No: Keep waiting (or preempt if priority)          │
│      │                                                              │
│      ├─► For each running request:                                  │
│      │   └─► Allocate 1 new block if current block is full          │
│      │                                                              │
│      └─► Build SchedulerOutput:                                     │
│          • Which sequences to run this step                         │
│          • Block tables (physical → logical mapping)                │
│          • Prefill vs decode classification                         │
│                                                                     │
│   3. MODEL EXECUTOR (dispatches to workers)                         │
│      ─────────────────────────────────────────────────────────     │
│      │                                                              │
│      ├─► Single GPU: Direct execution                               │
│      └─► Multi-GPU: Ray or multiprocessing workers                  │
│          └─► Each worker has model shard + KV cache shard           │
│                                                                     │
│   4. MODEL FORWARD PASS (CUDA kernels)                              │
│      ─────────────────────────────────────────────────────────     │
│      │                                                              │
│      ├─► Embedding lookup                                           │
│      ├─► For each layer:                                            │
│      │   ├─► LayerNorm                                              │
│      │   ├─► QKV projection                                         │
│      │   ├─► PagedAttention kernel (reads KV cache via block table) │
│      │   ├─► Output projection                                      │
│      │   ├─► LayerNorm                                              │
│      │   └─► MLP (gate, up, down projections)                       │
│      ├─► Final LayerNorm                                            │
│      └─► LM head → logits                                           │
│                                                                     │
│   5. SAMPLING (Python + CUDA)                                       │
│      ─────────────────────────────────────────────────────────     │
│      │                                                              │
│      ├─► Apply temperature, top-p, top-k                            │
│      ├─► Sample next token                                          │
│      └─► Check stopping conditions (EOS, max_tokens)                │
│                                                                     │
│   6. OUTPUT (back to API)                                           │
│      ─────────────────────────────────────────────────────────     │
│      │                                                              │
│      ├─► Streaming: Yield token immediately                         │
│      └─► Non-streaming: Accumulate until done                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Insight #2: The scheduler runs in Python and executes every decode step.** This is vLLM's main overhead source. At 100 tokens/second decode rate, the scheduler runs 100 times per second per request. V1 architecture moves scheduling to Rust to reduce this overhead.

### The Block Manager: vLLM's Memory Allocator

The Block Manager is the heart of PagedAttention. Let's understand exactly how it works:

```python
# Simplified Block Manager logic (actual vLLM code is more complex)

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size  # Tokens per block (default: 16)
        self.num_blocks = num_blocks  # Total physical blocks

        # Free block pool
        self.free_blocks = list(range(num_blocks))

        # Mapping: sequence_id → list of physical block indices
        self.block_tables: dict[int, list[int]] = {}

    def can_allocate(self, num_tokens: int) -> bool:
        """Check if we can allocate blocks for a new sequence."""
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        return len(self.free_blocks) >= blocks_needed

    def allocate(self, seq_id: int, num_tokens: int) -> list[int]:
        """Allocate blocks for a new sequence (prefill)."""
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        allocated = []
        for _ in range(blocks_needed):
            block_id = self.free_blocks.pop()
            allocated.append(block_id)

        self.block_tables[seq_id] = allocated
        return allocated

    def append_slot(self, seq_id: int) -> int:
        """Allocate one more slot for decode. May need new block."""
        blocks = self.block_tables[seq_id]
        current_block = blocks[-1]

        # Check if current block has space
        # (In reality, we track slots_used per block)
        slots_in_last_block = len(blocks) * self.block_size - self._get_seq_len(seq_id)

        if slots_in_last_block > 0:
            # Current block has space
            return current_block
        else:
            # Need new block
            new_block = self.free_blocks.pop()
            blocks.append(new_block)
            return new_block

    def free(self, seq_id: int):
        """Free all blocks for a completed sequence."""
        blocks = self.block_tables.pop(seq_id)
        self.free_blocks.extend(blocks)
```

**Insight #3: Block allocation is O(1) for decode (just check if current block has space) but O(blocks_needed) for prefill.** This is why prefill of a 4K prompt is more expensive than 4K decode steps—not just compute, but also allocation overhead.

### The PagedAttention Kernel

The magic happens in the PagedAttention CUDA kernel. Here's what it does:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PAGEDATTENTION KERNEL                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input:                                                            │
│   • Query tensor: [batch, num_heads, 1, head_dim] (decode)          │
│   • Block tables: [batch, max_blocks] - physical block indices      │
│   • KV cache: [num_blocks, 2, num_kv_heads, block_size, head_dim]   │
│   • Context lengths: [batch] - how many tokens each sequence has    │
│                                                                     │
│   The kernel does:                                                  │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   for each sequence in batch:                                       │
│       for each query head:                                          │
│           # Determine which KV head (for GQA)                       │
│           kv_head = query_head // num_kv_groups                     │
│                                                                     │
│           # Iterate through blocks in block table                   │
│           for block_idx in block_table[sequence]:                   │
│               # Load K, V from physical block                       │
│               K_block = kv_cache[block_idx, 0, kv_head]  # [16, 128]│
│               V_block = kv_cache[block_idx, 1, kv_head]  # [16, 128]│
│                                                                     │
│               # Compute attention scores for this block             │
│               scores = Q @ K_block.T  # [1, 16]                     │
│               # Accumulate weighted values                          │
│               output += softmax(scores) @ V_block                   │
│                                                                     │
│   Key optimizations:                                                │
│   • Blocks are processed in parallel across thread blocks           │
│   • Softmax is computed in a numerically stable streaming fashion   │
│   • Memory access is coalesced within blocks                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Insight #4: PagedAttention's overhead comes from the block table indirection.** Instead of reading KV cache from contiguous memory, we follow pointers. This adds ~5-10% overhead compared to contiguous attention, but the memory savings (60-80% less waste) more than compensate.

### vLLM V0 vs V1: What Changed

```
┌─────────────────────────────────────────────────────────────────────┐
│                    vLLM V0 vs V1 ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   V0 (Current Stable):                                              │
│   ─────────────────────────────────────────────────────────────    │
│   • Python scheduler runs synchronously every step                  │
│   • Single-step execution: schedule → execute → schedule → ...      │
│   • Chunked prefill: optional, manual enable                        │
│   • torch.compile: limited support                                  │
│                                                                     │
│   Timeline:                                                         │
│   [Sched][Execute][Sched][Execute][Sched][Execute]...               │
│      ↑                ↑                                             │
│      Python overhead  Python overhead                               │
│                                                                     │
│   V1 (Preview, becoming default):                                   │
│   ─────────────────────────────────────────────────────────────    │
│   • Rust scheduler (faster, async)                                  │
│   • Multi-step execution: schedule once, execute N steps            │
│   • Chunked prefill: default ON                                     │
│   • torch.compile: full integration                                 │
│                                                                     │
│   Timeline:                                                         │
│   [Sched][Execute][Execute][Execute][Sched][Execute][Execute]...    │
│      ↑                                   ↑                          │
│      Less frequent scheduling            Amortized overhead         │
│                                                                     │
│   Performance impact:                                               │
│   • 20-40% throughput improvement                                   │
│   • Lower P99 latency (less scheduling jitter)                      │
│   • Better GPU utilization                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Insight #5: V1's multi-step scheduling is the biggest win.** Instead of Python scheduler running 100 times/second, it runs 10-20 times/second with 5-10 steps batched. This alone accounts for most of the 20-40% improvement.

### The 6 Critical vLLM Tuning Knobs (With Understanding)

Most vLLM tuning guides list parameters without explaining the tradeoffs. Let's fix that:

```python
# vLLM Configuration Deep Dive

from vllm import LLM, SamplingParams

# ═══════════════════════════════════════════════════════════════════
# KNOB 1: gpu_memory_utilization (0.0 - 1.0)
# ═══════════════════════════════════════════════════════════════════
#
# What it controls: How much GPU memory vLLM reserves for KV cache
#
# The math:
#   Total GPU memory = Model weights + KV cache + Activations + Overhead
#   KV cache budget = GPU memory × utilization - (weights + activations + overhead)
#
# Example: 80GB H100, Llama 8B (16GB weights), 0.9 utilization
#   KV cache budget = 80 × 0.9 - 16 - 2 - 1 = 53 GB
#   At 512 MB per sequence (4K context), that's ~100 concurrent sequences
#
# Tradeoff:
#   Higher → More KV cache → Higher batch size → Higher throughput
#   Lower → Less KV cache → Room for other processes → Lower throughput
#
# Recommendation:
#   Production (dedicated GPU): 0.90-0.95
#   Shared GPU: 0.70-0.85
#   Development: 0.80

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    gpu_memory_utilization=0.90,  # Default
)

# ═══════════════════════════════════════════════════════════════════
# KNOB 2: max_num_seqs
# ═══════════════════════════════════════════════════════════════════
#
# What it controls: Maximum concurrent sequences in a batch
#
# Why it matters:
#   Each sequence needs KV cache memory
#   More sequences = more memory pressure
#   But also = better GPU utilization (more tokens per forward pass)
#
# The constraint:
#   max_num_seqs × avg_kv_cache_per_seq ≤ KV cache budget
#
# Tradeoff:
#   Higher → Better throughput (more parallelism)
#   Higher → Higher latency (more competition for GPU)
#   Lower → Lower latency (less queuing)
#   Lower → Lower throughput (underutilized GPU)
#
# Recommendation:
#   Throughput-focused: 256-1024
#   Latency-focused: 32-128
#   Balanced: 128-256

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_num_seqs=256,  # Default in V0
)

# ═══════════════════════════════════════════════════════════════════
# KNOB 3: max_num_batched_tokens
# ═══════════════════════════════════════════════════════════════════
#
# What it controls: Maximum tokens processed in one forward pass
#
# This is the MOST IMPORTANT throughput knob.
#
# Why it matters:
#   Prefill: processes prompt_length tokens
#   Decode: processes 1 token per sequence
#   Mixed batch: prefill_tokens + num_decode_sequences
#
# The constraint:
#   sum(tokens_per_sequence_this_step) ≤ max_num_batched_tokens
#
# Tradeoff:
#   Higher → Better throughput (larger batches)
#   Higher → Higher latency (longer forward passes)
#   Higher → More activation memory
#   Lower → Lower latency
#   Lower → Lower throughput
#
# Recommendation:
#   Throughput-focused: 16384-32768
#   Latency-focused: 2048-4096
#   Balanced: 4096-8192

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_num_batched_tokens=8192,
)

# ═══════════════════════════════════════════════════════════════════
# KNOB 4: enable_chunked_prefill
# ═══════════════════════════════════════════════════════════════════
#
# What it controls: Whether long prefills are split into chunks
#
# Why it matters:
#   Without chunking: 8K prompt blocks ALL decode for ~400ms
#   With chunking: 8K prompt processed in 4 × 2K chunks, interleaved
#
# The mechanism:
#   Prefill is split into chunks of ~max_num_batched_tokens
#   Between chunks, decode steps for other sequences run
#   Prevents "starvation" of existing requests
#
# Tradeoff:
#   Enabled → Consistent latency for all requests
#   Enabled → Slightly higher TTFT for long prompts
#   Disabled → Lower TTFT for long prompts (if no other requests)
#   Disabled → Latency spikes for existing requests
#
# Recommendation:
#   Production: ALWAYS enable
#   Benchmarking single requests: Can disable

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_chunked_prefill=True,  # Default ON in V1
)

# ═══════════════════════════════════════════════════════════════════
# KNOB 5: enable_prefix_caching
# ═══════════════════════════════════════════════════════════════════
#
# What it controls: Whether KV cache is shared for common prefixes
#
# Why it matters:
#   System prompt: "You are a helpful assistant..." (100 tokens)
#   Without caching: Computed 1000× for 1000 requests
#   With caching: Computed once, reused 1000×
#
# The mechanism:
#   Hash prompt tokens → Check if KV cache exists
#   If exists: Reuse cached KV blocks (copy-on-write)
#   If not: Compute and cache for future requests
#
# Tradeoff:
#   Enabled → Faster TTFT for repeated prefixes
#   Enabled → Memory overhead for cache management
#   Enabled → Slight overhead for cache lookups
#   Disabled → No overhead, but no reuse
#
# When it helps:
#   • Chatbots with system prompts
#   • RAG with repeated context
#   • Few-shot learning with same examples
#
# When it doesn't help:
#   • Every prompt is unique
#   • Batch processing with no repetition

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_prefix_caching=True,
)

# ═══════════════════════════════════════════════════════════════════
# KNOB 6: tensor_parallel_size
# ═══════════════════════════════════════════════════════════════════
#
# What it controls: How many GPUs share the model via tensor parallelism
#
# Why it matters:
#   70B model needs ~140GB in FP16
#   Single H100: 80GB → Doesn't fit
#   TP=2: 70GB per GPU → Fits
#
# The mechanism:
#   Weight matrices split across GPUs
#   Each GPU computes partial result
#   AllReduce to combine results
#
# Tradeoff:
#   Higher TP → Fits larger models
#   Higher TP → More memory bandwidth (good for decode)
#   Higher TP → AllReduce overhead
#   Higher TP → Requires NVLink for efficiency
#
# Scaling efficiency:
#   TP=2: ~1.8× throughput (90% efficiency)
#   TP=4: ~3.2× throughput (80% efficiency)
#   TP=8: ~5.6× throughput (70% efficiency)
#
# Recommendation:
#   Use minimum TP that fits your model
#   Prefer TP=2 or TP=4 over TP=8 if possible
#   TP > 8 rarely makes sense (use pipeline parallelism instead)

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
)
```

**Insight #6: The three most impactful knobs are max_num_batched_tokens (throughput), max_num_seqs (concurrency), and enable_chunked_prefill (latency consistency).** gpu_memory_utilization and tensor_parallel_size are usually set once based on hardware. prefix_caching depends on workload.

### Configuration Profiles for Common Workloads

```bash
# ═══════════════════════════════════════════════════════════════════
# PROFILE 1: Real-time Chat (Latency-Optimized)
# ═══════════════════════════════════════════════════════════════════
# Goal: Minimize TTFT and ITL for interactive users
# Tradeoff: Lower throughput

vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 4096 \
    --enable-chunked-prefill \
    --enable-prefix-caching

# Why these values:
# • Lower max-num-seqs: Less competition, lower queuing delay
# • Lower max-num-batched-tokens: Smaller forward passes, lower latency
# • Chunked prefill: Prevents long prompts from blocking
# • Prefix caching: System prompts are reused

# ═══════════════════════════════════════════════════════════════════
# PROFILE 2: Batch Processing (Throughput-Optimized)
# ═══════════════════════════════════════════════════════════════════
# Goal: Maximize tokens/second for offline processing
# Tradeoff: Higher latency per request

vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 512 \
    --max-num-batched-tokens 32768 \
    --enable-chunked-prefill

# Why these values:
# • Higher gpu-memory-utilization: Use all available memory
# • Higher max-num-seqs: Maximum parallelism
# • Higher max-num-batched-tokens: Larger batches
# • No prefix caching: Prompts don't repeat in batch processing

# ═══════════════════════════════════════════════════════════════════
# PROFILE 3: RAG Application (Balanced)
# ═══════════════════════════════════════════════════════════════════
# Goal: Good latency with high throughput, repeated context
# Tradeoff: Balanced

vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching

# Why these values:
# • Prefix caching: RAG context often repeats
# • Moderate batch sizes: Balance latency and throughput
# • Chunked prefill: RAG prompts can be long (retrieved docs)

# ═══════════════════════════════════════════════════════════════════
# PROFILE 4: Large Model Multi-GPU (70B on 4× H100)
# ═══════════════════════════════════════════════════════════════════

vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching

# Why these values:
# • TP=4: 70B needs ~140GB, 4×80GB H100 = 320GB (plenty of room)
# • Lower max-num-seqs: KV cache per sequence is larger for 70B
# • Prefix caching: Even more valuable for expensive prefills
```

**Insight #7: There's no universal "best" configuration. The optimal settings depend on your latency SLA, throughput requirements, and workload characteristics.** Start with a profile, measure, and iterate.

---

## SGLang: RadixAttention and Beyond

### Why SGLang Exists

vLLM optimizes for independent requests. But many LLM workloads have structure:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE PROBLEM SGLANG SOLVES                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Scenario: Multi-turn chatbot with 1000 concurrent users           │
│                                                                     │
│   System prompt (shared by ALL users):                              │
│   "You are a helpful assistant for Audible. You help users with     │
│    audiobook recommendations, account issues, and app support.      │
│    Always be friendly and concise."                                 │
│   = 50 tokens                                                       │
│                                                                     │
│   With vLLM (basic prefix caching):                                 │
│   ─────────────────────────────────────────────────────────────    │
│   • System prompt KV cache: computed once, shared                   │
│   • But: Each user's conversation history is separate               │
│   • User A turn 3 and User B turn 3 share nothing                   │
│                                                                     │
│   With SGLang (RadixAttention):                                     │
│   ─────────────────────────────────────────────────────────────    │
│   • System prompt: shared (same as vLLM)                            │
│   • Common conversation patterns: also shared!                      │
│   • "User: How do I cancel?" appears in 100 conversations           │
│   • That prefix is computed once, reused 100×                       │
│                                                                     │
│   The insight: Real workloads have MORE sharing than just           │
│   system prompts. RadixAttention captures ALL prefix sharing.       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Insight #8: RadixAttention is a generalization of prefix caching.** vLLM's prefix caching shares exact prefix matches. RadixAttention shares ANY common prefix, building a tree of all seen prefixes.

### RadixAttention: How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RADIXATTENTION DATA STRUCTURE                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   A Radix Tree (also called Patricia Trie) for KV cache:            │
│                                                                     │
│                         [ROOT]                                      │
│                            │                                        │
│              ┌─────────────┴─────────────┐                          │
│              ▼                           ▼                          │
│   ["You are a helpful"]         ["The capital of"]                  │
│   KV: blocks [0,1,2]            KV: blocks [10,11]                  │
│              │                           │                          │
│     ┌────────┴────────┐          ┌───────┴───────┐                  │
│     ▼                 ▼          ▼               ▼                  │
│ ["assistant"]    ["expert in"]  ["France"]    ["Japan"]             │
│ KV: block [3]    KV: block [4]  KV: block[12] KV: block[13]         │
│     │                 │              │               │              │
│     ▼                 ▼              ▼               ▼              │
│ ["User:"]        ["Python"]     ["is Paris"]    ["is Tokyo"]        │
│ KV: block[5]     KV: block[6]   KV: block[14]   KV: block[15]       │
│                                                                     │
│   Request: "You are a helpful assistant. User: Hello"               │
│   ─────────────────────────────────────────────────────────────    │
│   1. Traverse tree: ROOT → "You are a helpful" → "assistant"        │
│      → "User:"                                                      │
│   2. Found! Reuse KV blocks [0,1,2,3,5]                             │
│   3. Only compute KV for "Hello" (new suffix)                       │
│                                                                     │
│   Request: "You are a helpful expert in Python. How do I..."        │
│   ─────────────────────────────────────────────────────────────    │
│   1. Traverse: ROOT → "You are a helpful" → "expert in" → "Python"  │
│   2. Reuse KV blocks [0,1,2,4,6]                                    │
│   3. Compute KV for "How do I..." (new suffix)                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Insight #9: RadixAttention's tree structure means prefix matching is O(prefix_length), not O(num_cached_prefixes).** vLLM's hash-based prefix cache is O(1) for exact matches but can't find partial matches. RadixAttention finds the longest matching prefix efficiently.

### When SGLang Beats vLLM

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SGLANG vs vLLM: WHEN TO USE WHICH                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   SGLang WINS (use it):                                             │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   1. Multi-turn conversations with shared patterns                  │
│      • Chatbots where users ask similar questions                   │
│      • Customer support with common issues                          │
│      • Speedup: 20-50% from prefix reuse                            │
│                                                                     │
│   2. Structured output generation                                   │
│      • JSON with schema constraints                                 │
│      • Code with syntax constraints                                 │
│      • SGLang's constrained decoding is native, not bolted on       │
│      • Speedup: 10-30% from efficient constraint checking           │
│                                                                     │
│   3. Multi-step LLM programs                                        │
│      • Chain-of-thought with multiple generations                   │
│      • Tool use with interleaved calls                              │
│      • SGLang compiles the program, optimizes across steps          │
│      • Speedup: 30-100% from cross-step optimization                │
│                                                                     │
│   4. Tree-based generation                                          │
│      • Beam search                                                  │
│      • Best-of-N sampling                                           │
│      • RadixAttention shares prefixes across branches               │
│      • Speedup: 50-80% from branch sharing                          │
│                                                                     │
│   vLLM WINS (use it):                                               │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   1. Simple request/response with unique prompts                    │
│      • Batch processing of diverse inputs                           │
│      • No prefix sharing opportunity                                │
│      • vLLM has lower overhead                                      │
│                                                                     │
│   2. Maximum model compatibility                                    │
│      • vLLM supports more models out of the box                     │
│      • Newer architectures often land in vLLM first                 │
│                                                                     │
│   3. Simpler deployment                                             │
│      • vLLM's API is more mature                                    │
│      • Better documentation and community support                   │
│                                                                     │
│   4. When you need speculative decoding                             │
│      • vLLM's speculative decoding is more mature                   │
│      • SGLang's is catching up but less tested                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Insight #10: The decision isn't "SGLang vs vLLM" but "does my workload have exploitable structure?"** If yes, SGLang's RadixAttention and program compiler can provide significant speedups. If no, vLLM's simpler architecture has less overhead.

### SGLang's Constrained Decoding

SGLang's structured output is fundamentally different from vLLM's JSON mode:

```python
# vLLM JSON mode: Post-hoc validation
# ─────────────────────────────────────────────────────────────────
# vLLM generates tokens, then validates JSON at the end
# If invalid, you get an error or malformed output

from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
params = SamplingParams(
    temperature=0.7,
    # guided_json validates but doesn't constrain during generation
)

# SGLang constrained decoding: Token-level constraints
# ─────────────────────────────────────────────────────────────────
# SGLang masks invalid tokens BEFORE sampling
# Output is GUARANTEED to match the constraint

import sglang as sgl

@sgl.function
def extract_person(s, text):
    s += f"Extract person info from: {text}\n"
    s += "Output JSON:\n"

    # This regex constraint is applied at EVERY token
    # Invalid tokens are masked to probability 0
    s += sgl.gen(
        "result",
        regex=r'\{"name": "[A-Za-z ]+", "age": \d+, "city": "[A-Za-z ]+"\}'
    )

# The difference in practice:
# ─────────────────────────────────────────────────────────────────
# vLLM: Model might generate {"name": "John", "age": "thirty", ...}
#       → Invalid! "thirty" is not a number
#       → You need retry logic
#
# SGLang: After generating "age": , only digit tokens are allowed
#         → Model MUST generate a number
#         → No retries needed, 100% valid output
```

**Insight #11: SGLang's constrained decoding is compile-time, not runtime.** The regex/grammar is compiled into a finite state machine. At each token, SGLang computes which tokens are valid transitions and masks the rest. This adds ~5% overhead but guarantees valid output.

### SGLang Program Compilation

SGLang's most powerful feature is its program compiler:

````python
import sglang as sgl

# A multi-step LLM program
@sgl.function
def analyze_code(s, code):
    # Step 1: Identify the language
    s += f"Code:\n```\n{code}\n```\n\n"
    s += "Programming language: "
    s += sgl.gen("language", max_tokens=10, stop="\n")

    # Step 2: Find bugs (depends on step 1)
    s += f"\n\nBugs in this {s['language']} code:\n"
    s += sgl.gen("bugs", max_tokens=200)

    # Step 3: Suggest fixes (depends on step 2)
    s += "\n\nSuggested fixes:\n"
    s += sgl.gen("fixes", max_tokens=200)

# What SGLang's compiler does:
# ─────────────────────────────────────────────────────────────────
# 1. Analyzes the program structure
# 2. Identifies that steps 1, 2, 3 share a growing prefix
# 3. Schedules KV cache to be retained across steps
# 4. Batches multiple programs together when possible
#
# Without compilation (naive approach):
#   Step 1: Prefill "Code:...", generate "language"
#   Step 2: Prefill "Code:... language... Bugs:", generate "bugs"
#           ↑ Recomputes KV for "Code:... language"!
#   Step 3: Prefill everything again
#           ↑ Recomputes KV for everything!
#
# With SGLang compilation:
#   Step 1: Prefill "Code:...", generate "language", KEEP KV cache
#   Step 2: Prefill only "Bugs:", generate "bugs", KEEP KV cache
#   Step 3: Prefill only "Suggested fixes:", generate "fixes"
#
# Speedup: 2-3× for multi-step programs
````

**Insight #12: SGLang's compiler transforms sequential LLM calls into a single optimized execution plan.** This is similar to how SQL query optimizers transform queries—the logical program is the same, but the physical execution is much more efficient.

---

## TensorRT-LLM: The Compilation Approach

### Why Compilation Matters

vLLM and SGLang use PyTorch with custom CUDA kernels for hot paths. TensorRT-LLM takes a different approach: compile the entire model to an optimized TensorRT engine.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PYTORCH vs TENSORRT EXECUTION                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   PyTorch (vLLM, SGLang):                                           │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   Python                                                            │
│      │                                                              │
│      ▼                                                              │
│   PyTorch dispatcher (decides which kernel to call)                 │
│      │                                                              │
│      ▼                                                              │
│   CUDA kernel 1 (e.g., LayerNorm)                                   │
│      │                                                              │
│      ▼                                                              │
│   Back to Python/PyTorch                                            │
│      │                                                              │
│      ▼                                                              │
│   CUDA kernel 2 (e.g., Linear)                                      │
│      │                                                              │
│      ▼                                                              │
│   ... (repeat for every operation)                                  │
│                                                                     │
│   Overhead: Kernel launch latency (~5-10μs per kernel)              │
│   A transformer layer has ~20 kernels → 100-200μs overhead/layer    │
│   32 layers → 3-6ms overhead per forward pass                       │
│                                                                     │
│   TensorRT:                                                         │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   C++ Runtime                                                       │
│      │                                                              │
│      ▼                                                              │
│   TensorRT Engine (pre-compiled, fused kernels)                     │
│      │                                                              │
│      ▼                                                              │
│   Fused CUDA kernel (LayerNorm + Linear + Activation)               │
│      │                                                              │
│      ▼                                                              │
│   Fused CUDA kernel (Attention)                                     │
│      │                                                              │
│      ▼                                                              │
│   ... (fewer, larger kernels)                                       │
│                                                                     │
│   Overhead: Minimal kernel launch overhead                          │
│   Fused kernels: 5-10 per layer instead of 20                       │
│   32 layers → 0.5-1ms overhead per forward pass                     │
│                                                                     │
│   Speedup from reduced overhead: 10-20%                             │
│   Speedup from kernel fusion: 10-30%                                │
│   Total: 20-50% faster than PyTorch                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Insight #13: TensorRT's speedup comes from two sources: fewer kernel launches (fusion) and optimized kernel implementations.** The compilation process analyzes the entire graph and finds opportunities that per-operator optimization misses.

### The TensorRT-LLM Compilation Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TENSORRT-LLM BUILD PIPELINE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   STEP 1: Convert Checkpoint                                        │
│   ─────────────────────────────────────────────────────────────    │
│   Input: HuggingFace model (PyTorch weights)                        │
│   Output: TensorRT-LLM checkpoint (reorganized weights)             │
│                                                                     │
│   python convert_checkpoint.py \                                    │
│       --model_dir ./llama-3.1-8b \                                  │
│       --output_dir ./trt_ckpt \                                     │
│       --dtype float16 \                                             │
│       --tp_size 1                                                   │
│                                                                     │
│   What happens:                                                     │
│   • Weights are transposed/reshaped for TensorRT's layout           │
│   • Quantization calibration (if using INT8/FP8)                    │
│   • Tensor parallel sharding (if tp_size > 1)                       │
│                                                                     │
│   STEP 2: Build Engine                                              │
│   ─────────────────────────────────────────────────────────────    │
│   Input: TensorRT-LLM checkpoint                                    │
│   Output: Optimized TensorRT engine (.engine file)                  │
│                                                                     │
│   trtllm-build \                                                    │
│       --checkpoint_dir ./trt_ckpt \                                 │
│       --output_dir ./trt_engine \                                   │
│       --gemm_plugin float16 \                                       │
│       --gpt_attention_plugin float16 \                              │
│       --max_batch_size 64 \                                         │
│       --max_input_len 2048 \                                        │
│       --max_output_len 512 \                                        │
│       --max_num_tokens 8192                                         │
│                                                                     │
│   What happens (this is the slow part, 10-60 minutes):              │
│   • Graph optimization (constant folding, dead code elimination)    │
│   • Kernel fusion (combine adjacent operations)                     │
│   • Kernel auto-tuning (try different implementations, pick best)   │
│   • Memory planning (optimize tensor lifetimes)                     │
│   • Quantization (if enabled)                                       │
│                                                                     │
│   STEP 3: Run Inference                                             │
│   ─────────────────────────────────────────────────────────────    │
│   Input: TensorRT engine + input tokens                             │
│   Output: Generated tokens                                          │
│                                                                     │
│   # Python API                                                      │
│   from tensorrt_llm import LLM                                      │
│   llm = LLM(model="./trt_engine")                                   │
│   output = llm.generate("Hello, world!")                            │
│                                                                     │
│   # Or use Triton Inference Server                                  │
│   # (recommended for production)                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Insight #14: The max_batch_size, max_input_len, max_output_len parameters are BAKED INTO the engine.** Unlike vLLM where you can change these at runtime, TensorRT engines are compiled for specific shapes. If you need different shapes, you need to rebuild.

### TensorRT-LLM's Key Optimizations

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TENSORRT-LLM OPTIMIZATIONS                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   1. KERNEL FUSION                                                  │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   Before fusion:                                                    │
│   LayerNorm → Q_proj → K_proj → V_proj → Reshape → Attention        │
│   (6 kernel launches, 6 memory round-trips)                         │
│                                                                     │
│   After fusion:                                                     │
│   FusedQKVLayerNorm → FusedAttention                                │
│   (2 kernel launches, 2 memory round-trips)                         │
│                                                                     │
│   Benefit: 3× fewer kernel launches, 3× less memory traffic         │
│                                                                     │
│   2. GEMM PLUGIN                                                    │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   TensorRT's default GEMM vs TensorRT-LLM's GEMM plugin:            │
│   • Plugin uses cuBLAS with LLM-specific tuning                     │
│   • Optimized for the specific shapes in transformers               │
│   • FP8 support on H100 with tensor cores                           │
│                                                                     │
│   Benefit: 10-20% faster matrix multiplications                     │
│                                                                     │
│   3. IN-FLIGHT BATCHING                                             │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   TensorRT-LLM's version of continuous batching:                    │
│   • Implemented in C++ (lower overhead than Python)                 │
│   • Integrated with TensorRT's memory management                    │
│   • Supports paged KV cache (similar to vLLM)                       │
│                                                                     │
│   Benefit: Same throughput benefits as vLLM, lower overhead         │
│                                                                     │
│   4. QUANTIZATION                                                   │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   TensorRT-LLM supports:                                            │
│   • FP8 (H100 native, best quality/speed tradeoff)                  │
│   • INT8 with SmoothQuant (good quality, wide hardware support)     │
│   • INT4 with AWQ/GPTQ (maximum compression)                        │
│   • Mixed precision (different layers at different precision)       │
│                                                                     │
│   Benefit: Same memory savings as vLLM, but with compiled kernels   │
│                                                                     │
│   5. SPECULATIVE DECODING                                           │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   TensorRT-LLM supports draft model speculation:                    │
│   • Draft and target models both compiled                           │
│   • Verification is a single fused kernel                           │
│   • Lower overhead than Python-based verification                   │
│                                                                     │
│   Benefit: 10-20% better speculative decoding speedup               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Insight #15: TensorRT-LLM's optimizations are the same as vLLM's (PagedAttention, continuous batching, quantization) but implemented with compiled kernels instead of Python + custom CUDA.** The 20-50% speedup comes from lower overhead, not fundamentally different algorithms.

### When TensorRT-LLM Makes Sense

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TENSORRT-LLM DECISION GUIDE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   USE TensorRT-LLM when:                                            │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   ✓ Model is stable (not changing frequently)                       │
│     → Compilation takes 10-60 minutes per configuration             │
│     → Changing model = recompile                                    │
│                                                                     │
│   ✓ Workload shapes are predictable                                 │
│     → max_batch_size, max_seq_len are baked in                      │
│     → Different shapes = different engines                          │
│                                                                     │
│   ✓ Running on NVIDIA GPUs (especially H100)                        │
│     → TensorRT is NVIDIA-only                                       │
│     → H100 FP8 support is excellent                                 │
│                                                                     │
│   ✓ Maximum performance is worth the complexity                     │
│     → 20-50% faster than vLLM                                       │
│     → But: More complex deployment, less flexibility                │
│                                                                     │
│   ✓ Using Triton Inference Server                                   │
│     → TensorRT-LLM integrates natively with Triton                  │
│     → Production-grade serving with model management                │
│                                                                     │
│   DON'T USE TensorRT-LLM when:                                      │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   ✗ Rapid iteration needed                                          │
│     → vLLM: change model in seconds                                 │
│     → TensorRT-LLM: recompile for 30 minutes                        │
│                                                                     │
│   ✗ Workload shapes vary widely                                     │
│     → Would need multiple engines for different shapes              │
│     → Memory overhead of multiple engines                           │
│                                                                     │
│   ✗ Using non-NVIDIA hardware                                       │
│     → TensorRT is NVIDIA-only                                       │
│     → Use vLLM or SGLang instead                                    │
│                                                                     │
│   ✗ Need structured output / constrained decoding                   │
│     → TensorRT-LLM's support is limited                             │
│     → SGLang is much better for this                                │
│                                                                     │
│   ✗ Team lacks TensorRT expertise                                   │
│     → Debugging compiled engines is harder                          │
│     → vLLM errors are more interpretable                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Insight #16: TensorRT-LLM is the "production at scale" choice, not the "getting started" choice.** Start with vLLM, measure your performance, and only move to TensorRT-LLM if you need that extra 20-50% and can accept the complexity.
