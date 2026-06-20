# 09.2 System Design: Code Completion Copilot

## Introduction

A code completion copilot represents one of the most demanding LLM inference workloads in production today. Unlike chatbots where users tolerate 1-2 second response times, code completion must feel instantaneous: every keystroke triggers a request, and if the suggestion arrives after the developer has already typed the next character, it is worthless. This creates a brutal constraint: sub-200ms time-to-first-token (TTFT) while processing up to 128K tokens of context (the current file, imported modules, open tabs, and repository structure).

GitHub Copilot, Cursor, Cody, and Supermaven all face this same fundamental tension: the model needs maximum context to produce high-quality completions, but processing that context takes time proportional to its length. The system design challenge is resolving this tension at scale for 10 million daily active users generating 500 million completion requests per day.

This chapter walks through the complete system design, from model selection through failure modes, showing how every technique from earlier chapters (speculative decoding, KV compression, context parallelism, RadixAttention, chunked prefill) converges in a single production system.

---

## 1. Requirements Analysis

### Latency Requirements

The fundamental constraint is human perception during typing:

| Metric | Target | Rationale |
|--------|--------|-----------|
| TTFT p50 | <100ms | Feels instantaneous at typing speed |
| TTFT p95 | <200ms | Must arrive before next keystroke |
| TTFT p99 | <500ms | Acceptable for complex completions |
| Total generation time | <1s | Short outputs (50 tokens avg) |
| Suggestion staleness | <300ms | Beyond this, user has moved on |

Why 200ms is the hard boundary: average typing speed is 200ms between keystrokes for a proficient developer. If TTFT exceeds the inter-keystroke interval, the suggestion arrives after the user has already typed the next character, making it stale. The system must either display the completion before the next keystroke or discard it entirely.

### Throughput Requirements

```
Daily active users:        10,000,000
Completions per user/day:  50 (conservative; power users hit 200+)
Total requests per day:    500,000,000
Sustained RPS:             500M / 86,400 = ~5,787 RPS
Peak RPS (3x sustained):   ~17,400 RPS
```

The traffic pattern is heavily diurnal and timezone-dependent. US business hours (9am-6pm ET) see 4-5x the off-peak rate. Monday mornings have the highest sustained load. The system must handle bursts during "morning standup ends, everyone starts coding" events.

### Context Requirements

A code completion copilot needs extensive context to produce useful suggestions:

| Context Source | Typical Size | Priority |
|----------------|-------------|----------|
| Current file (cursor position) | 2K-8K tokens | Critical |
| Import/dependency files | 4K-16K tokens | High |
| Open editor tabs | 8K-32K tokens | Medium |
| Repository structure/types | 4K-16K tokens | Medium |
| Recent edits (diff context) | 2K-8K tokens | Medium |
| Language server info (types, signatures) | 1K-4K tokens | Low |
| **Total potential context** | **21K-84K tokens** | - |

The system must support up to 128K token context windows to accommodate large codebases, but most requests will use 16K-64K tokens. The distribution is heavy-tailed: 80% of requests need <32K tokens, but the remaining 20% (complex files with many dependencies) need the full 128K.

### Output Characteristics

Code completions are fundamentally different from chat responses:

- **Short outputs**: Average 50 tokens (1-3 lines of code), 95th percentile is 200 tokens (a function body)
- **High rejection rate**: Only 25-30% of suggestions are accepted by users
- **Speculative nature**: Most completions are generated and discarded without being shown
- **Prefix-heavy**: 80% of value is in the first line; multi-line completions have diminishing acceptance rates

This output profile means **prefill dominates cost**, not decode. A 128K input generating 50 output tokens spends 99%+ of compute on prefill. This is the opposite of chatbot workloads where long outputs dominate.

### Functional Requirements

1. **Multi-language support**: Python, TypeScript, Java, Go, Rust, C++ (minimum). Each language has different completion patterns.
2. **Fill-in-the-middle (FIM)**: Not just next-token prediction. Must support inserting code between existing lines.
3. **Multi-line completions**: When the model is confident, generate entire function bodies.
4. **Context-aware cancellation**: If the user types ahead, cancel in-flight requests immediately.
5. **Graceful degradation**: Under load, fall back to shorter context rather than failing entirely.

---

## 2. Model Selection

### Why Small Models Win

The latency budget of <200ms TTFT eliminates large models entirely:

| Model Size | FP16 Memory | Prefill Time (32K ctx, 1xH100) | Verdict |
|-----------|-------------|-------------------------------|---------|
| 70B | 140 GB | ~800ms | ❌ Too slow |
| 34B | 68 GB | ~400ms | ❌ Marginal |
| 8B | 16 GB | ~100ms | ✅ Feasible |
| 3B | 6 GB | ~40ms | ✅ Fast |
| 1B | 2 GB | ~15ms | ✅ Draft model |

The 8B parameter class (DeepSeek-Coder-V2-Lite, CodeLlama-8B, StarCoder2-7B) provides the best quality/latency tradeoff for code completion. These models achieve 70-80% of the quality of 70B models on code benchmarks (HumanEval, MBPP) while fitting within the latency budget.

### Speculative Decoding Architecture

The optimal architecture combines a small draft model with a larger verification model:

```
Draft Model:  1B parameters (DeepSeek-Coder-1.3B or similar)
Verify Model: 8B parameters (DeepSeek-Coder-V2-Lite-Instruct)

Workflow:
1. Draft model generates K=5 candidate tokens (fast, ~3ms/token)
2. Verify model checks all K tokens in one forward pass
3. Accept matching tokens, reject and regenerate from first mismatch
4. Net speedup: 2-3x decode throughput with identical output quality
```

Why speculative decoding is perfect for code completion:
- Code is highly predictable (variable names, syntax patterns, common idioms)
- Draft acceptance rate for code is 70-85% (much higher than natural language at 50-60%)
- Short outputs (50 tokens) mean the speedup compounds: 50 tokens at 3x speed = done in ~50ms decode time

### Model Quantization Strategy

For the verification model (8B), use INT8 quantization (not INT4):

```
FP16:  16 GB model weights, baseline quality
INT8:  8 GB model weights, <1% quality loss on code benchmarks
INT4:  4 GB model weights, 3-5% quality loss (unacceptable for code)
```

INT4 is too aggressive for code completion because small errors in token probabilities cause incorrect variable names, wrong operators, and syntax errors. INT8 preserves quality while halving memory, allowing more KV cache budget on the same GPU.

### Fill-in-the-Middle (FIM) Training

The model must be trained with FIM objective, not just autoregressive next-token:

```
Standard autoregressive: prefix -> completion
FIM format:             <prefix>code before cursor<suffix>code after cursor<middle>

Example:
<prefix>def fibonacci(n):
    if n <= 1:
        return n
    <suffix>
    return result
<middle>result = fibonacci(n-1) + fibonacci(n-2)
```

This is critical because developers often type a function signature, skip to the closing brace, then want the body filled in. Without FIM training, the model can only complete from the end of the visible text.

---

## 3. Memory Budget

### The KV Cache Crisis

The KV cache is the single biggest challenge in code copilot design. For an 8B model with 128K context:

```python
# KV cache memory calculation for 8B model (32 layers, 32 heads, 128 head_dim)
layers = 32
heads = 32  # for keys and values
head_dim = 128
seq_len = 128_000  # 128K context
bytes_per_param = 2  # FP16

kv_per_token = 2 * layers * heads * head_dim * bytes_per_param
# = 2 * 32 * 32 * 128 * 2 = 524,288 bytes = 512 KB per token

total_kv = kv_per_token * seq_len
# = 512 KB * 128,000 = 65,536 MB = 64 GB per request!
```

This is catastrophic: a single 128K request needs 64 GB of KV cache, which exceeds the entire memory of an H100 (80 GB) when combined with model weights (8 GB INT8). Serving even ONE concurrent request at full context is barely possible, let alone thousands.

### KV Compression Strategies (from Ch02.4)

Three approaches to tame the KV cache:

**Strategy 1: Multi-Latent Attention (MLA)**
DeepSeek-V2/V3 compresses KV by projecting to a low-rank latent space:
```
Standard KV: 512 KB/token
MLA (rank 512): 512 KB/token -> 4 KB/token (128x compression)
128K context: 64 GB -> 500 MB ✓
```
This requires the model to be trained with MLA from scratch (DeepSeek-Coder-V2 uses this).

**Strategy 2: Grouped Query Attention (GQA)**
Reduce KV heads from 32 to 8 (4x compression):
```
GQA-8: 512 KB/token -> 128 KB/token (4x compression)
128K context: 64 GB -> 16 GB
```
Still large but manageable with 2 GPUs. Most 8B models (Llama 3, Mistral) use GQA.

**Strategy 3: Sliding Window + Sparse Attention**
Keep full KV for recent tokens, compress older tokens:
```
Full attention window: last 4K tokens (2 GB KV)
Sparse attention: tokens 4K-128K (sample every 4th token = 8K effective, 4 GB KV)
Total: 6 GB KV per request
```
Loses some long-range precision but code locality means recent tokens matter most.

### Practical Memory Layout (2xH100, TP=2)

```
Component                    | Per-GPU Memory | Total (2 GPUs)
-----------------------------|---------------|---------------
Model weights (8B INT8)      | 4 GB          | 8 GB
Draft model (1B FP16)        | 1 GB          | 2 GB
KV cache (GQA-8, 128K, 1 req)| 8 GB         | 16 GB
KV cache pool (64 concurrent)| 256 GB needed | IMPOSSIBLE
                             |               |
With sliding window + sparse:|               |
KV cache (6 GB/req, 64 conc) | 192 GB       | Still too much
                             |               |
With RadixAttention sharing: |               |
Shared prefix pool (80%)     | 40 GB         | 80 GB
Per-request unique KV        | 20 GB         | 40 GB
Total KV budget              | 60 GB         | 120 GB -> fits in 2xH100
```

The key insight: in a code copilot, 80%+ of KV cache is shared across users editing the same file or using the same libraries. RadixAttention (Section 7) exploits this massively.

### Memory Budget Summary

```
Per 2xH100 node:
  Model weights (INT8):     8 GB
  Draft model (FP16):       2 GB
  CUDA overhead:            4 GB
  KV cache pool:           120 GB (RadixAttention shared)
  Activations/workspace:    26 GB
  --------------------------------
  Total:                   160 GB (= 2 x 80 GB H100)
  
  Max concurrent requests: ~80-120 (depending on context length distribution)
```

---

## 4. Hardware Selection

### Why H100 Over A100

The code copilot workload is **memory-bandwidth bound during decode** and **compute-bound during prefill**:

| Metric | A100 80GB | H100 80GB | Improvement |
|--------|-----------|-----------|-------------|
| HBM bandwidth | 2 TB/s | 3.35 TB/s | 1.67x |
| FP16 compute | 312 TFLOPS | 989 TFLOPS | 3.2x |
| FP8 compute | N/A | 1,979 TFLOPS | ∞ |
| Memory | 80 GB | 80 GB | Same |
| NVLink bandwidth | 600 GB/s | 900 GB/s | 1.5x |

The 3.2x compute improvement directly translates to 3.2x faster prefill for 128K context. This is the difference between 320ms prefill (A100, too slow) and 100ms prefill (H100, within budget).

The 1.67x bandwidth improvement means decode tokens arrive faster: at 8B INT8, each token requires reading 8 GB of weights, taking 8GB/3.35TB/s = 2.4ms on H100 vs 4ms on A100.

### Tensor Parallelism = 2

Why TP=2, not TP=4 or TP=1:

```
TP=1 (single H100):
  Prefill 128K tokens: ~200ms (too close to budget)
  Max KV cache: ~60 GB (limited concurrency)
  
TP=2 (2x H100 NVLinked):
  Prefill 128K tokens: ~100ms ✓ (halved via parallelism)
  Max KV cache: ~120 GB (good concurrency)
  NVLink overhead: ~5ms per sync (negligible)
  
TP=4 (4x H100):
  Prefill 128K tokens: ~55ms (overkill)
  Communication overhead starts to matter
  Cost doubles for diminishing returns
```

TP=2 is the sweet spot: it halves prefill time (the bottleneck) while keeping communication overhead minimal over NVLink.

### Cost Per Completion

```
H100 cost (cloud): ~$3.50/hour per GPU
TP=2 node cost: $7.00/hour

Throughput per node: ~80 concurrent requests
                     ~400 completions/second (50 token avg, speculative decoding)
                     ~1.44M completions/hour

Cost per completion: $7.00 / 1,440,000 = $0.0000049 = $0.005 per 1000 completions

At 500M completions/day:
  Nodes needed (sustained): 500M / (1.44M * 24) = ~14.5 nodes = 15 nodes
  Peak (3x): 45 nodes
  Daily cost (sustained): 15 * $7 * 24 = $2,520/day
  Monthly cost: ~$75,000 (sustained) to ~$225,000 (provisioned for peak)
```

This is remarkably cost-effective at scale. The key enablers are: (1) short outputs mean fast request turnaround, (2) RadixAttention shares prefill cost across users, (3) speculative decoding speeds decode by 2-3x.

---

## 5. Parallelism Strategy

### Data Parallelism for Throughput

Each TP=2 node handles requests independently. Scale horizontally with data parallelism:

```
Architecture:
  Load Balancer (smart routing)
       |
  ┌────┼────┬────┬────┬────┐
  DP0  DP1  DP2  DP3  ... DP44
  (TP=2)(TP=2)(TP=2)(TP=2)   (TP=2)
  
  Total GPUs: 45 nodes x 2 GPUs = 90 H100s (peak provisioning)
```

### Context Parallelism for Long Prefill

When a request arrives with 128K context, even TP=2 takes ~100ms for prefill. For the rare cases needing <50ms TTFT (autocomplete in fast-typing scenarios), use context parallelism:

```
Standard prefill (TP=2, 128K):
  GPU0: processes tokens 0-128K (first half of layers)
  GPU1: processes tokens 0-128K (second half of layers)
  Time: ~100ms

Context parallel prefill (TP=2, CP=2, 128K):
  GPU0: processes tokens 0-64K (all layers, first half of sequence)
  GPU1: processes tokens 64K-128K (all layers, second half of sequence)
  Ring attention sync at each layer boundary
  Time: ~55ms (not quite 2x due to sync overhead)
```

Context parallelism (from Ch05.4) is only activated for requests with >64K context tokens. Below that threshold, standard TP=2 already meets the 100ms prefill budget.

### Prefill/Decode Disaggregation

The most important architectural decision: **separate prefill and decode into different GPU pools**.

```
Prefill pool (compute-optimized):
  - 20 nodes (40 H100s)
  - Handles all 128K token prefills
  - Processes context, generates KV cache
  - Transfers KV cache to decode pool via RDMA

Decode pool (bandwidth-optimized):
  - 25 nodes (50 H100s)  
  - Receives pre-computed KV cache
  - Generates 50-token completions
  - Speculative decoding (draft + verify)
  - Higher concurrency (shorter KV per request after prefix sharing)
```

Why disaggregate for code copilots:
1. Prefill is compute-bound (128K tokens of matrix multiplications)
2. Decode is bandwidth-bound (reading 8 GB of weights per token)
3. A single GPU doing both wastes compute during decode and wastes bandwidth during prefill
4. Prefill nodes can run at higher batch sizes (amortize compute across requests)
5. Decode nodes maintain lower latency (no prefill blocking the pipeline)

### KV Cache Transfer

After prefill completes, the KV cache must reach the decode node:

```
KV cache for one request (GQA-8, 128K tokens):
  Size: 8 GB (with GQA compression)
  
Transfer options:
  NVLink (same node): 900 GB/s -> 9ms ✓
  InfiniBand (cross-node): 400 GB/s -> 20ms ✓
  Ethernet (fallback): 100 GB/s -> 80ms (too slow)
```

InfiniBand RDMA between prefill and decode pools adds 20ms to TTFT but enables the disaggregated architecture. Total TTFT budget: 100ms prefill + 20ms transfer + 5ms first decode token = 125ms (within 200ms target).

---

## 6. Serving Architecture

### SGLang with RadixAttention

SGLang (from Ch06) is the optimal serving engine for code copilots because of RadixAttention: a radix tree that stores KV cache prefixes and shares them across requests.

```
Why RadixAttention is perfect for code:

User A opens main.py:    [import numpy...] [def train():...] [cursor here]
User B opens main.py:    [import numpy...] [def train():...] [different cursor]
User A types next char:  [import numpy...] [def train():...] [cursor here + 1 char]

Without RadixAttention:
  3 separate 128K prefills = 3 x 100ms compute

With RadixAttention:
  First request: full 128K prefill (100ms)
  Second request: match prefix in radix tree, only compute delta (5ms)
  Third request: match prefix, compute 1-token delta (<1ms)
```

In a code copilot, successive keystrokes in the same file share 99.9% of their context. RadixAttention reduces the amortized prefill cost from 100ms to <5ms for ongoing editing sessions.

### Chunked Prefill (from Ch03.5)

Long prefill (128K tokens) takes ~100ms even on H100. During this time, decode requests from other users are blocked. Chunked prefill interleaves prefill and decode:

```
Without chunked prefill:
  t=0ms:   Start 128K prefill for User A
  t=100ms: User A prefill complete, start decode
  t=0-100ms: Users B,C,D decode requests BLOCKED (TTFT violated!)

With chunked prefill (chunk_size=8192):
  t=0ms:   Process chunk 1 (tokens 0-8K) for User A [6ms]
  t=6ms:   Process 1 decode step for Users B,C,D [2ms]
  t=8ms:   Process chunk 2 (tokens 8K-16K) for User A [6ms]
  t=14ms:  Process 1 decode step for Users B,C,D [2ms]
  ...
  t=100ms: User A prefill complete
  
  Users B,C,D experience: 8ms inter-token latency (vs blocked)
```

Chunk size of 8192 tokens is optimal: large enough to amortize kernel launch overhead, small enough to give decode slots every 6-8ms.

### Speculative Decoding Pipeline (from Ch03.4)

The decode phase uses speculative decoding with the 1B draft model:

```
Speculative decoding for code completion:

Step 1: Draft model generates K=5 tokens
  "result = fibonacci(n" -> ["-", "1", ")", " ", "+"]  (3ms total)

Step 2: Verify model checks all 5 in one forward pass
  Verify: ["-", "1", ")", " ", "+"] -> all match! Accept all 5. (5ms)
  
Step 3: Net result: 5 tokens in 8ms = 1.6ms/token
  Without spec decoding: 5 tokens x 5ms/token = 25ms
  Speedup: 3.1x
```

Code has high token-level predictability (variable names repeat, syntax is constrained, common patterns like `for i in range(n)` are near-deterministic). Draft acceptance rates for code average 75-85%, making speculative decoding exceptionally effective.

### Request Lifecycle

Complete request flow from keystroke to suggestion:

```
t=0ms:    User types character in IDE
t=5ms:    IDE extension debounces, sends request to gateway
t=10ms:   Gateway routes to nearest region, hits load balancer
t=12ms:   Load balancer checks RadixAttention prefix cache
t=13ms:   CACHE HIT: 99.5% prefix match (same file, 1 new token)
t=14ms:   Incremental prefill: process 1 new token (0.1ms)
t=15ms:   Decode begins (speculative, K=5)
t=23ms:   First 5 tokens generated
t=31ms:   Next 5 tokens generated
t=50ms:   50 tokens complete, return to client
t=55ms:   IDE displays suggestion

Total TTFT: 15ms (cache hit case)
Total completion time: 55ms
```

For cache-miss cases (first time opening a file, or cold start):
```
t=0ms:    Request arrives
t=12ms:   CACHE MISS: no prefix match
t=13ms:   Route to prefill pool
t=113ms:  128K prefill complete, KV cache stored in radix tree
t=133ms:  KV transferred to decode node
t=138ms:  First decode token generated (TTFT = 138ms ✓)
t=188ms:  50 tokens complete via speculative decoding
```

---

## 7. Caching Strategy

### RadixAttention Tree Structure

The radix tree stores KV cache at the token level, enabling prefix sharing:

```
Root
├── "import numpy as np\nimport torch\n" (common ML file header)
│   ├── "class Model(nn.Module):\n    def __init__" (User A's file)
│   │   ├── "(self, hidden=512):" (User A, keystroke 1)
│   │   └── "(self, hidden=1024):" (User B, same template)
│   └── "def train_loop(model, data):\n" (User C's file)
└── "package main\nimport \"fmt\"\n" (Go file header)
    └── "func main() {\n    fmt.Println(" (User D)
```

Each node stores the KV cache tensors for its token span. When a new request arrives, the engine walks the tree to find the longest matching prefix, then only computes KV for the remaining tokens.

### Session-Level Caching

A code editing session (one file open in one IDE) generates hundreds of requests as the user types. The caching strategy exploits this:

```
Session lifecycle:
  1. File opened: Full 128K prefill (cold start, 100ms)
  2. User types char 1: Incremental prefill of 1 token (0.1ms)
  3. User types char 2: Incremental prefill of 1 token (0.1ms)
  ...
  N. User types char N: Incremental prefill of 1 token (0.1ms)
  
  Amortized TTFT across session: (100ms + N * 0.1ms) / (N+1)
  For N=100 keystrokes: ~1.1ms average TTFT
```

This is why the cache hit rate is so critical: a code copilot with warm caches has sub-5ms TTFT for 95%+ of requests, far exceeding the 200ms target.

### Cache Eviction Policy

With limited GPU memory, the system must evict stale KV cache entries:

```
Eviction priority (lowest priority evicted first):
  1. Sessions with no request in >5 minutes (user stopped typing)
  2. Prefix nodes with only 1 reference (unique to one user)
  3. Short prefix nodes (< 1K tokens, cheap to recompute)
  
  NEVER evict:
  - Active session prefixes (user is currently typing)
  - Highly shared prefixes (>10 references, e.g. common imports)
  - Long prefixes (>64K tokens, expensive to recompute)
```

### Cross-User Prefix Sharing

In a large organization, many developers work on the same codebase:

```
Scenario: 50 developers working on the same Python project

Shared prefixes:
  - requirements.txt imports: ~500 tokens (shared by all 50)
  - Base class definitions: ~2K tokens (shared by 30)
  - Utility module: ~4K tokens (shared by 20)
  - Same file open: ~8K tokens (shared by 5)

Without sharing: 50 x 64K avg = 3.2M tokens of KV stored
With sharing:    500 + 2K + 4K + 8K shared + 50 x 16K unique = 814K tokens
Compression ratio: 3.9x memory savings
```

### Distributed Cache Tier

For multi-region deployments, add an L2 cache on CPU/DRAM:

```
L1: GPU HBM (hot cache, per-node, ~60 GB)
    - Active sessions, <100ms access
    
L2: Host DRAM (warm cache, per-machine, ~512 GB)
    - Recently idle sessions, ~5ms to reload to GPU
    - 8x more capacity than L1
    
L3: Distributed Redis/Memcached (cold cache, cross-region)
    - Sessions idle >10 min, ~20ms to fetch
    - Only stores compressed KV (quantized to INT4 for storage)
    - Full recomputation if miss here
```

Promotion/demotion:
- GPU -> DRAM: session idle >30 seconds
- DRAM -> Redis: session idle >5 minutes  
- Redis -> evict: session idle >30 minutes
- Any tier -> GPU: new request arrives for cached session

---

## 8. Monitoring and SLOs

### The Only Metric That Matters: TTFT

For a code copilot, TTFT is the single most important SLO. Everything else is secondary:

```
SLO Hierarchy:
  CRITICAL: TTFT p95 < 200ms
  HIGH:     TTFT p99 < 500ms
  MEDIUM:   Acceptance rate > 25%
  LOW:      Total latency p95 < 1s
  INFO:     Token throughput, GPU utilization
```

If TTFT exceeds 200ms, the completion is useless regardless of quality. A fast bad completion is better than a slow good one (user can dismiss bad suggestions instantly, but cannot un-wait).

### TTFT Decomposition Dashboard

Break TTFT into components to identify bottlenecks:

```
TTFT = Network + Queue + Prefix_Match + Prefill + Transfer + First_Decode

Typical breakdown (cache hit):
  Network: 5ms (client to edge)
  Queue: 2ms (scheduler picks up request)
  Prefix match: 1ms (radix tree lookup)
  Prefill delta: 1ms (1-10 new tokens)
  Transfer: 0ms (same node)
  First decode: 3ms
  TOTAL: 12ms ✓

Typical breakdown (cache miss, 128K context):
  Network: 5ms
  Queue: 5ms (may need to wait for GPU slot)
  Prefix match: 1ms (miss)
  Prefill: 100ms (full 128K computation)
  Transfer: 20ms (prefill node -> decode node)
  First decode: 5ms
  TOTAL: 136ms ✓

Degraded case (overloaded, no cache):
  Network: 10ms
  Queue: 50ms (queuing behind other prefills)
  Prefix match: 1ms
  Prefill: 100ms
  Transfer: 20ms
  First decode: 5ms
  TOTAL: 186ms (close to limit!)
```

### Acceptance Rate Monitoring

Acceptance rate measures whether suggestions are useful:

```python
# Acceptance rate calculation
acceptance_rate = accepted_completions / shown_completions

# Breakdown by signal:
# - Full accept: user tabs to accept entire suggestion
# - Partial accept: user accepts first line, modifies rest
# - Implicit accept: user types exactly what was suggested (without tab)
# - Reject: user continues typing, ignoring suggestion
# - Stale: suggestion arrived after user moved on (not counted in denominator)

# Target: >25% full accept rate, >40% full+partial
```

A/B test suggestion quality by routing 5% of traffic to a new model and comparing acceptance rates. Statistically significant difference (p<0.05) requires ~100K suggestions per variant (achievable in hours at this scale).

### Staleness Detection

A "stale" suggestion arrives after the user has already typed past the suggestion point:

```
Staleness rate = stale_suggestions / total_generated

Stale if: suggestion.arrival_time > next_keystroke.time
         OR suggestion.prefix != current_editor_state

Target: <5% staleness rate
Action if >10%: increase prefill capacity, enable priority queuing for active typers
```

### GPU Monitoring

```
Key GPU metrics:
  - KV cache utilization: % of allocated KV pool in use (target: 60-80%)
  - Prefill queue depth: requests waiting for prefill slot (target: <5)
  - Decode batch size: concurrent decode requests (target: 40-80)
  - Radix cache hit rate: % of requests with prefix match (target: >90%)
  - Speculative acceptance rate: draft tokens accepted (target: >70%)
  
Alerts:
  - KV cache >90%: trigger eviction sweep, page alert
  - Prefill queue >20: scale up prefill pool
  - Cache hit rate <80%: investigate cache eviction, possible workload shift
  - Spec acceptance <60%: draft model may be out of distribution
```

### Goodput vs Throughput

From Ch07.4, goodput measures useful work:

```
Throughput: total tokens generated per second (includes rejected, stale, cancelled)
Goodput: tokens in ACCEPTED completions per second

For code copilots:
  Throughput: 400 completions/sec/node x 50 tokens = 20,000 tokens/sec/node
  Acceptance rate: 28%
  Shown rate: 60% (40% cancelled before showing)
  Goodput: 20,000 x 0.60 x 0.28 = 3,360 useful tokens/sec/node
  
  Waste ratio: 83% of generated tokens are never used
  This is NORMAL for code copilots -- speculative generation is inherently wasteful
```

---

## 9. Scaling and Cost

### Traffic Analysis

```
500M requests/day distribution:
  - 80% cache hits (400M): <5ms TTFT, minimal GPU cost
  - 15% partial hits (75M): 10-50ms TTFT, partial prefill
  - 5% cache misses (25M): 100-150ms TTFT, full prefill

Compute demand (in GPU-seconds/day):
  Cache hits: 400M x 0.005s decode = 2M GPU-seconds
  Partial hits: 75M x (0.030s prefill + 0.005s decode) = 2.6M GPU-seconds
  Cache misses: 25M x (0.100s prefill + 0.005s decode) = 2.6M GPU-seconds
  Total: 7.2M GPU-seconds/day = 83 GPU-days
  
With overhead (scheduling, transfers, idle): 83 x 1.5 = 125 GPU-days
At 24 hours: need ~125 GPUs sustained (63 TP=2 nodes)
Peak (3x): ~190 nodes
```

### Cost Optimization Strategies

**Strategy 1: Aggressive caching reduces prefill compute by 80%**
Without caching, every request needs full prefill:
```
No cache: 500M x 0.100s = 50M GPU-seconds = 579 GPU-days = $48,600/day
With RadixAttention: 7.2M GPU-seconds = 83 GPU-days = $7,000/day
Savings: 86% cost reduction from caching alone
```

**Strategy 2: Short-circuit low-confidence completions**
If the model's first-token probability is below threshold, abort immediately:
```
Confidence threshold: top_p < 0.3 for first token -> abort
Abort rate: ~30% of requests (model unsure what code comes next)
Savings: 30% fewer decode operations
```

**Strategy 3: Adaptive context length**
Not every request needs 128K context:
```
Heuristic context sizing:
  - Single-line completion (common): 8K context sufficient
  - Multi-line function body: 32K context
  - Complex refactoring: full 128K context
  
  Route based on request type:
  - 60% of requests: 8K context (6ms prefill, cache miss)
  - 30% of requests: 32K context (25ms prefill, cache miss)
  - 10% of requests: 128K context (100ms prefill, cache miss)
  
  Average prefill cost drops 4x vs always using 128K
```

**Strategy 4: Off-peak model upgrades**
During low-traffic hours (nights, weekends), route to higher-quality 34B model:
```
Peak hours (9am-6pm): 8B model only (latency priority)
Off-peak (6pm-9am): 34B model for complex completions (quality priority)
  - 34B fits in TP=4, TTFT ~100ms with lower traffic
  - Acceptance rate improves 10-15% with larger model
  - No additional hardware needed (same GPUs, less traffic)
```

### Auto-Scaling Policy

```yaml
scaling_policy:
  metric: prefill_queue_depth
  target: 5  # requests waiting for prefill
  
  scale_up:
    threshold: queue_depth > 10 for 60 seconds
    action: add 2 TP=2 nodes to prefill pool
    cooldown: 300 seconds
    max_nodes: 100
  
  scale_down:
    threshold: queue_depth < 2 for 300 seconds  
    action: remove 1 node from prefill pool
    cooldown: 600 seconds
    min_nodes: 15
  
  # Predictive scaling for known patterns
  scheduled_scale:
    - cron: "0 8 * * MON-FRI"  # 8am weekdays
      action: scale to 60 nodes (anticipate morning coding)
    - cron: "0 18 * * MON-FRI"  # 6pm weekdays
      action: scale to 25 nodes (evening drop)
```

### Monthly Cost Summary

```
Infrastructure (45 nodes average, 90 H100s):
  GPU compute: 90 x $3.50/hr x 720 hrs = $226,800/month
  Networking (InfiniBand): ~$20,000/month
  Storage (model weights, logs): ~$5,000/month
  Load balancers + edge: ~$15,000/month
  Total infrastructure: ~$267,000/month

Per-user cost:
  $267,000 / 10M users = $0.027/user/month = 2.7 cents per user per month

Per-completion cost:
  $267,000 / (500M x 30) = $0.000018 per completion

Revenue (at $10/user/month individual, $19/user/month business):
  Even at 1M paying users: $10M-19M/month revenue
  Gross margin: >95% (infrastructure is tiny fraction of revenue)
```

---

## 10. Failure Modes and Mitigations

### Failure Mode 1: Context Too Long (Truncation Strategy)

When context exceeds 128K tokens (large monorepo files with many imports):

```
Truncation priority (keep highest priority):
  1. Current function/block around cursor (ALWAYS keep, ~2K tokens)
  2. Import statements and type definitions (~4K tokens)
  3. Same-file functions called by current function (~8K tokens)
  4. Open tab contents, ordered by recency (~16K tokens)
  5. Repository-level type stubs (~8K tokens)
  6. Distant same-file code (TRUNCATE FIRST)

Algorithm:
  total_budget = 128_000
  allocated = 0
  for source in priority_order:
    if allocated + source.tokens <= total_budget:
      include(source)
      allocated += source.tokens
    else:
      include(source.truncate(total_budget - allocated))
      break
```

The key insight: the code immediately surrounding the cursor is infinitely more valuable than distant code. A developer typing inside `def process_data()` needs the function signature, local variables, and called functions. Code 500 lines away in the same file has near-zero predictive value.

### Failure Mode 2: Model Quality for Niche Languages

The 8B model's training data is dominated by Python, JavaScript, and Java. Performance degrades on niche languages:

```
Language tier by model quality:
  Tier 1 (excellent): Python, JavaScript, TypeScript, Java, C++, Go
  Tier 2 (good): Rust, Ruby, PHP, C#, Kotlin, Swift
  Tier 3 (mediocre): Haskell, OCaml, Elixir, Clojure, Zig
  Tier 4 (poor): COBOL, Fortran, Assembly, Verilog

Mitigations:
  Tier 1-2: Standard 8B model, full pipeline
  Tier 3: Route to 34B model (better few-shot learning), accept higher latency
  Tier 4: Disable multi-line completions, single-token only, or disable entirely
  
  Per-language acceptance rate monitoring:
  If acceptance_rate(language) < 15% for 7 days:
    -> Alert model team
    -> Consider fine-tuning on that language's corpus
    -> Reduce suggestion frequency for that language (don't annoy users)
```

### Failure Mode 3: Stale Suggestions

The user typed ahead while the suggestion was being generated:

```
Staleness detection pipeline:
  1. Client sends request with cursor_position=P and file_version=V
  2. Server generates completion for position P
  3. Before sending response, server checks:
     - Has client sent a newer request? (cursor_position=P+1)
     - If yes: DISCARD current response, process newer request
  4. Client-side validation:
     - On receiving suggestion, check current cursor position
     - If cursor has moved past suggestion insertion point: discard
     
Implementation:
  - Client maintains a monotonically increasing request_id
  - Server keeps only the LATEST request_id per session
  - When a new request arrives, all in-flight requests for that session are cancelled
  - Cancelled requests free their decode slots immediately
```

Cancel propagation is critical: a cancelled request that continues generating wastes GPU cycles. The decode scheduler must support immediate preemption when a newer request arrives for the same session.

### Failure Mode 4: GPU Memory Pressure (OOM)

When KV cache pool is exhausted:

```
Graceful degradation ladder:
  Level 0 (normal): Full 128K context, all features enabled
  Level 1 (pressure): Reduce max context to 64K, evict idle sessions aggressively
  Level 2 (critical): Reduce max context to 32K, disable multi-line completions
  Level 3 (emergency): Reduce max context to 8K, single-line only, drop 50% of requests
  Level 4 (OOM imminent): Reject all new requests, serve only active decode operations
  
  Recovery:
  - Each level has a 30-second cooldown before stepping down
  - Auto-scaling triggers at Level 1 (add nodes in 2-3 minutes)
  - Level 3+ pages on-call immediately
```

### Failure Mode 5: Cascading Failures

A prefill node failure causes a thundering herd on remaining nodes:

```
Scenario: 1 of 15 prefill nodes dies during peak
  - 1/15 = 6.7% of traffic rerouted to remaining 14 nodes
  - Each node now handles 7.1% more traffic
  - If nodes were at 80% capacity: now at 85.7% (manageable)
  
Scenario: 3 of 15 prefill nodes die simultaneously  
  - 20% of traffic rerouted to remaining 12 nodes
  - Each node handles 25% more traffic
  - If nodes were at 80%: now at 100% -> queue buildup -> TTFT violation
  
Mitigations:
  1. Headroom: never run >70% capacity (30% margin for failures)
  2. Circuit breaker: if queue_depth > 50, shed load (return empty completion)
  3. Priority: paying users get priority, free tier gets degraded first
  4. Cross-region failover: route to secondary region if primary is impaired
```

### Failure Mode 6: Cold Start Problem

New users or new files have no cached KV:

```
Cold start scenarios:
  1. First request of the day: full 128K prefill (100ms TTFT)
  2. Switching to a new file: new prefill for different context
  3. After IDE restart: all session caches invalidated
  
Mitigations:
  1. Predictive pre-warming: when user opens a file in IDE, immediately 
     start prefill in background (before first keystroke)
  2. Project-level caching: cache common imports/headers at project level
     (shared across all files in same repo)
  3. Warm pool: maintain pre-computed KV for top 1000 most common file headers
     (numpy imports, React boilerplate, etc.)
```

### Failure Mode 7: Model Drift and Quality Regression

Deploying a new model version that subtly degrades quality:

```
Canary deployment:
  1. Route 2% of traffic to new model version
  2. Monitor acceptance rate vs control (98% on old model)
  3. If acceptance_rate_new < acceptance_rate_old - 2%: auto-rollback
  4. If acceptance_rate_new >= acceptance_rate_old: gradually increase to 100%
  5. Full rollout takes 48 hours minimum
  
Shadow mode (pre-deployment):
  - Run new model on 100% of traffic in shadow (don't show results)
  - Compare output quality offline against ground truth (what user actually typed)
  - Only promote to canary if shadow metrics are >= current model
```

---

## Summary: Complete System Architecture

```
IDE Extension (client)
    |
    | HTTPS (5ms)
    v
Edge CDN / API Gateway
    |
    | Route by region (5ms)
    v
Smart Load Balancer
    |
    | Radix tree prefix check (1ms)
    |
    ├── CACHE HIT (95%) ──────> Decode Pool (TP=2 nodes)
    |                              - Speculative decoding (draft 1B + verify 8B)
    |                              - 50 tokens in 35ms
    |                              - RadixAttention for prefix reuse
    |
    └── CACHE MISS (5%) ──────> Prefill Pool (TP=2 nodes)
                                   - Full 128K context processing (100ms)
                                   - KV cache stored in radix tree
                                   - Transfer KV to decode pool (20ms)
                                   - Then decode as above

    Scaling: 45 TP=2 nodes (90 H100s) peak
    Cost: $267K/month for 10M users ($0.027/user/month)
    TTFT: 12ms (cache hit), 138ms (cache miss), p95 < 200ms ✓
```

### Key Design Decisions Recap

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Model size | 8B (verify) + 1B (draft) | Only size class meeting <200ms TTFT |
| Quantization | INT8 | Best quality/memory tradeoff for code |
| Parallelism | TP=2 per node, DP across nodes | Halves prefill time, simple scaling |
| Serving engine | SGLang | RadixAttention is transformative for code |
| Caching | Radix tree + session persistence | 95% cache hit rate = 12ms avg TTFT |
| Disaggregation | Separate prefill and decode pools | Each optimized for its bottleneck |
| Decode speedup | Speculative decoding K=5 | 3x faster decode, perfect for predictable code |
| Context budget | 128K max, adaptive sizing | Most requests need <32K, save compute |
| Eviction | Session-aware, priority-based | Never evict active typing sessions |
| Failure handling | Graceful degradation ladder | Shorter context > no response |

### Cross-Chapter References

This system design integrates techniques from across the book:
- **Ch02.4**: KV compression (GQA, MLA) to fit 128K context in memory
- **Ch03.4**: Speculative decoding for 3x decode speedup
- **Ch03.5**: Chunked prefill to prevent long prefills from blocking decode
- **Ch05.4**: Context parallelism (ring attention) for >64K prefill acceleration
- **Ch06.6**: Cache-aware routing and RadixAttention for prefix sharing
- **Ch07.4**: Goodput metrics and staleness monitoring
- **Ch08.1**: Meta's disaggregated serving architecture pattern
- **Ch08.3**: Mixed workload scheduling (prefill vs decode priority)

The code copilot is the quintessential inference system design problem because it simultaneously demands:
- Ultra-low latency (typing speed constraints)
- Long context (full file + dependencies)
- High throughput (500M requests/day)
- Extreme cache efficiency (repetitive keystroke pattern)
- Graceful degradation (useless if slow, better to skip than lag)

Every optimization technique in this book finds direct application in making code completion fast, accurate, and cost-effective at scale.
