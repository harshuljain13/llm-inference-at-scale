# AIE Workshop 2026: LLM Inference at Scale
## Presentation Narrative (39 slides, 8 demos, 2 hours)

Each slide ends with a question or implication that pulls the audience into the next.
Transitions are explicit. The audience should never wonder "why are we talking about this now?"

---

## OPENING: "Your model is slow. Let's find out why." (15 min)

### Slide 1: Title
**LLM Inference at Scale: From Memory Equation to Production Engines**

Repo QR code. Molab links for attendees.

> "By the end of these 2 hours, you will know exactly why your model is slow, and how to make it 20x faster on the same hardware."

---

### Slide 2: The Cost Problem
**Inference costs more than training. And it never stops.**

Training GPT-3 cost $4.6M. Once.
Serving it costs more than that every single week. Forever.

> "Training is a fixed cost. Inference is a running bill. And it scales with your users."

**→ So who's paying this bill?**

---

### Slide 3: Inference Is Everywhere
**Every interaction with AI is an inference call. The scale is staggering.**

- **AI coding agents:** 52% of code is now AI-authored (DX, Q2 2026). 1 in 7 PRs involve AI agents (Pullflow). 84-91% of developers use AI coding tools (2025-2026 surveys).
- **AI products:** ChatGPT, Claude, Gemini, Copilot — every chat message, every autocomplete, every agent tool call = inference.
- **The market:** AI inference-as-a-service: $18.6B in 2025 → $23.4B in 2026 (Precedence Research). Cost per million tokens dropped 1,200x in 5 years ($60 in 2020 → $0.05 in 2025) — but volume grew even faster.
- **The real cost:** One 50-turn coding session consumes ~1M tokens (Vantage, 2026). A 25-person team: $72K/year on Opus.

Whether you raise PRs, chat with AI, or build agent features — you're doing LLM inference. And it's expensive at scale.
---

### Slide 4: The Field is Moving Fast
**Key breakthroughs in LLM inference (2022-2026).**

```mermaid
flowchart LR
    A["2022<br/>FlashAttention<br/>6x memory savings"] --> B["2022<br/>Orca<br/>continuous batching"]
    B --> C["2023<br/>vLLM<br/>PagedAttention"]
    C --> D["2024<br/>SGLang<br/>RadixAttention"]
    D --> E["2024<br/>DistServe<br/>disaggregated P/D"]
    E --> F["2025<br/>DeepSeek V3<br/>MLA: 90% cost cut"]
    F --> G["2026<br/>NVIDIA Dynamo<br/>agentic inference"]
    F --> H["2026<br/>Stanford M*<br/>multimodal serving"]

    style A fill:#f3f4f6,stroke:#000,color:#1e293b
    style B fill:#f3f4f6,stroke:#000,color:#1e293b
    style C fill:#fef3c7,stroke:#000,color:#1e293b
    style D fill:#fef3c7,stroke:#000,color:#1e293b
    style E fill:#dcfce7,stroke:#000,color:#1e293b
    style F fill:#dcfce7,stroke:#000,color:#1e293b
    style G fill:#dbeafe,stroke:#000,color:#1e293b
    style H fill:#dbeafe,stroke:#000,color:#1e293b
```

Each innovation solved a specific bottleneck:
- **FlashAttention** (2022): tiling attention in SRAM, no N² matrix in HBM
- **Orca** (2022): iteration-level scheduling, continuous batching
- **vLLM** (2023): PagedAttention, virtual memory for KV cache, 24x over HF
- **SGLang** (2024): radix tree prefix cache, 5x cache hit rate
- **DistServe** (2024): separate prefill and decode onto different GPUs
- **DeepSeek V3** (2025): MLA compresses KV 56x, 90% cost reduction
- **NVIDIA Dynamo** (2026): KV-aware routing for agentic sessions
- **Stanford M*** (2026): Walk Graphs for multimodal composite models

> "This workshop gives you the foundations to understand ALL of these. Let's start."

> "This workshop gives you the foundations to evaluate any new paper or tool. Let's start by experiencing the problem."

---

### Slide 5: What We Cover
**Foundations → Attention → KV Cache → Optimizations → Engines**

Chapters 00-06 of the book. Chapters 07-11 (multimodal, K8s, Ray) are in the repo for you to explore.

---



---

### DEMO A: Feel the Pain
> `demos/demo_a_feel_the_pain.ipynb` (5 min)
> 
> What the audience sees:
> 1. "14.5 GB consumed just by weights"
> 2. "TTFT: grows from 90ms to 5000ms as context scales to 16K"
> 3. "5 users queue up: last one waits 5x longer than first"
> 4. "KV cache grows ~1 GB per user at 8K context"
>
> End with: "Why? Let's find out."

---

## PART 1: Foundations (25 min)

> **Transition:** "We saw the problems. Now let's understand the machine that causes them."

### GROUP 1: What's inside the model

### Slide 7: The Inference Pipeline
**Text in → tokens → 32 layers → next token out. Runs once PER token.**

```mermaid
flowchart LR
    A["Input Text"] --> B["Tokenizer"]
    B --> C["Embedding<br/>(128K x 4096)"]
    C --> D["32 Transformer<br/>Layers"]
    D --> E["LM Head<br/>(4096 x 128K)"]
    E --> F["Next Token"]

    style A fill:#ffe4e6,stroke:#000,color:#1e293b
    style B fill:#f3f4f6,stroke:#000,color:#1e293b
    style C fill:#dbeafe,stroke:#000,color:#1e293b
    style D fill:#dcfce7,stroke:#000,color:#1e293b
    style E fill:#f3e8ff,stroke:#000,color:#1e293b
    style F fill:#fef3c7,stroke:#000,color:#1e293b
```

To generate 100 tokens, this runs 100 times. The 32 layers are where 95% of memory lives.

---

### Slide 8: Inside One Transformer Layer
**Each layer: Attention + MLP. Same structure x32, different learned weights.**

```mermaid
flowchart LR
    A["Input"] --> B["Attention<br/>(which tokens matter?)"]
    B --> C["MLP<br/>(4096 to 14336 to 4096)"]
    C --> D["Output"]

    style A fill:#ffe4e6,stroke:#000,color:#1e293b
    style B fill:#dbeafe,stroke:#000,color:#1e293b
    style C fill:#dcfce7,stroke:#000,color:#1e293b
    style D fill:#ccfbf1,stroke:#000,color:#1e293b
```

- **Attention:** For the current token, which earlier tokens should influence its meaning?
- **MLP:** Transform the representation. Stores factual knowledge. Largest component.

---

### Slide 9: Weight Distribution
**Where the 14.5 GB lives.**

| Component | Size | % |
|-----------|------|---|
| Attention (Q,K,V,O x 32 layers) | 8.0 GB | 50% |
| MLP (gate, up, down x 32 layers) | 6.8 GB | 42% |
| Embedding + LM Head | 1.0 GB | 6% |
| **Total** | **~16 GB** | |

---

### 🎯 DEMO B: Architecture + Weights (5 min)
> `demos/demo_b_memory_equation.ipynb`
> 1. Load Mistral-7B config: 32 layers, 4096 hidden, 8 KV heads
> 2. Derive weight sizes: Attention 8 GB, MLP 6.8 GB, Embedding 1 GB
> 3. Pie chart showing 50%/42%/6% distribution
> 4. Total: 7.24B params x 2 bytes = 14.5 GB
>
> Proves: the model is 14.5 GB fixed. This is what gets read every decode step.

---

### GROUP 2: Why decode is slow

### Slide 10: Prefill vs Decode
**Two phases. Prefill is fast (parallel). Decode is the bottleneck (sequential).**

**Prefill:** All input tokens processed in parallel.
- GPU computes attention across all positions at once. Compute-bound. Fast.

**Decode:** One token per step. Each step reads ALL 14.5 GB of weights from memory.
- Sequential. Memory-bound. This is THE bottleneck.

---

### Slide 11: GPU Memory Hierarchy
**The model streams from HBM. That's the speed ceiling.**

| Memory Level | Size | Bandwidth |
|-------|------|-----------|
| SRAM (on-chip) | 20 MB | 19 TB/s |
| HBM (GPU memory) | 80 GB | **2 TB/s** |

Model is 700x larger than SRAM. Must stream from HBM.
Each decode step: `14.5 GB / 2 TB/s = 7.25 ms per token = ~138 tok/s ceiling`

---

### Slide 12: Roofline: Decode is Memory-Bound
**80x below compute peak. More FLOPS won't help.**

- Arithmetic intensity of decode: **2 FLOP/byte**
- Ridgeline (compute bottleneck): **156 FLOP/byte**
- Gap: **80x** of compute sits idle

Only two escapes: read LESS data, or serve MORE users per read.

---

### 🎯 DEMO C: Prefill vs Decode (5 min)
> `demos/demo_c_prefill_vs_decode.ipynb`
> 1. Measures prefill time at 128-8K tokens (shows linear scaling, fast)
> 2. Measures per-token decode latency (shows ~constant, memory-bound)
> 3. Side-by-side chart: prefill (green, parallel) vs decode (red, sequential)
>
> Proves: decode reads 14.5 GB every step. GPU compute sits 80% idle.: TTFT scaling proved prefill cost.
> Per-token decode timing proved the bandwidth wall.
> Speaker walks through those results explaining the roofline connection.

---

### GROUP 3: KV cache and capacity

### Slide 13: Why the KV Cache Exists
**Without it: recompute attention for ALL previous tokens every step. O(n squared).**

During decode, attention needs K and V vectors from ALL previous tokens.
- Without cache: recompute everything. Gets quadratically slower.
- With cache: store K,V vectors. O(n) but costs memory per user.

Every production system uses the KV cache. The question: how much memory?

---

### Slide 14: KV Cache Growth
**Every token x every layer x every head. Adds up fast.**

```
KV per token = 2 (K + V) x 32 layers x 8 KV heads x 128 dim x 2 bytes = 131 KB
```

| Scenario | KV Memory |
|----------|-----------|
| 1 user, 4K context | 524 MB |
| 1 user, 16K context | 2.1 GB |
| 80 users, 4K context | **42 GB** |
| 80 users, 16K context | 168 GB (OOM) |

Weights: fixed. KV cache: the variable that kills you.

---

### Slide 15: The Capacity Equation
**Three numbers decide max users.**

```
Available = GPU_total - weights - overhead
Max users = Available / (131 KB x context_length)
```

A100-80GB: 80 - 14.5 - 8 = 57.5 GB available
- At 4K context: **109 users**
- At 16K context: **27 users**

---

### Slide 16: Latency SLO Determines Your Batch Size
**You can't just max out batch size. Users have latency expectations.**

Two SLO metrics:
- **TTFT** (Time to First Token): user waits before seeing anything. Target: < 500ms.
- **ITL** (Inter-Token Latency): time between streamed tokens. Target: < 50ms.

Higher batch = better throughput but worse per-user latency.
Your SLO caps the batch size, which caps your throughput and revenue per GPU.

```
Throughput = batch_size x (1000 / ITL_ms) tokens/sec
Max batch = whatever keeps ITL < SLO target
```

---

### Slide 17: GPU Selection

| GPU | VRAM | Bandwidth | Max Users (4K) | $/M tokens |
|-----|------|-----------|------|-----------|
| A100-80 | 80 GB | 2.0 TB/s | 109 | $0.28 |
| H100-80 | 80 GB | 3.4 TB/s | 109 | $0.16 |
| H200-141 | 141 GB | 4.8 TB/s | 237 | $0.13 |

More VRAM + more bandwidth = cheaper per token.

---

### 🎯 DEMO D: KV Cache Growth + Capacity Calculator (5 min)
> `demos/demo_d_capacity_calculator.ipynb`
> 1. Derive: KV per token = 2 x 32 x 8 x 128 x 2 = 131 KB
> 2. Plot: KV cache growth curve as users x context scales
> 3. Show: 80 users at 4K = 42 GB (fills the GPU)
> 4. Interactive: pick model, precision, GPU, context. Watch memory fill.
> 5. GPU comparison: $/M tokens across A100, H100, H200
>
> Proves: KV cache is the variable cost. Capacity equation works.
> Interactive: pick model, precision, GPU, context. Watch memory fill. See $/M tokens.

> **End of Part 1:** "Now we know the machine, why it's slow, and what limits capacity. Next: how to push those limits."

---

## PART 2: "The first free lunch: attention architecture." (12 min)

> **Transition:** "We're stuck at 107 users on A100-80. The 131 KB per token is the bottleneck. Can we reduce it?"

### Slide 15: GQA Already Saved You
**Without GQA, that 131 KB would be 524 KB.**

```mermaid
flowchart LR
    A[MHA<br>32 KV heads<br>524 KB/tok] --> B[GQA-8<br>8 KV heads<br>131 KB/tok]
    B --> C[MLA<br>compressed<br>9.3 KB/tok]
    style A fill:#ffe4e6,stroke:#000
    style B fill:#dcfce7,stroke:#000
    style C fill:#dbeafe,stroke:#000
```

GQA groups 4 query heads per KV head. 4x savings. Already in every modern model.

> "This was free. You didn't have to do anything. But can we go further?"

---

### Slide 16: FlashAttention
**Same math. 6x less memory traffic.**

Standard attention writes N² scores to HBM. FlashAttention tiles in SRAM.
Never materializes the full attention matrix. Exact same output.

> "Also free. Enabled by default. What about the KV cache itself?"

---

### Slide 17: MLA (DeepSeek)
**Compress before caching. 56x savings.**

Store a 512-dim latent instead of full K (128d) + V (128d) per head.
Decompress at attention time. Why DeepSeek-V3 runs at 90% lower cost.

> "GQA is today. MLA is tomorrow. Both: same quality, less memory."

---

### 🎯 DEMO D: Attention Comparison (4 min)
> `demos/demo_e_attention_comparison.ipynb`
> Bar chart: MHA 524 KB vs GQA 131 KB vs MLA 9.3 KB.
> End with: "OK, we've reduced the per-token cost. Now: can we reduce the number of tokens stored?"

---

## PART 3: "Four ways to fix the KV cache." (20 min)

> **Transition:** "Attention architecture determines cost per token. These four techniques reduce what's actually stored."

### Slide 18: Four Levers

```mermaid
flowchart LR
    A[PagedAttention<br>No fragmentation] --> B[Prefix Cache<br>Reuse across users]
    B --> C[Smart Eviction<br>Keep only what matters]
    C --> D[KV Quantization<br>Fewer bits per value]
    style A fill:#dbeafe,stroke:#000
    style B fill:#dcfce7,stroke:#000
    style C fill:#f3e8ff,stroke:#000
    style D fill:#fef3c7,stroke:#000
```

> "Each gives 2-10x. They're orthogonal. Stack all four."

---

### Slide 19: PagedAttention
**Like virtual memory for your GPU.**

```mermaid
flowchart LR
    subgraph BEFORE["Without PagedAttention"]
        U1B[User 1<br>allocated 4K] ~~~ GAP1[GAP<br>wasted]
        GAP1 ~~~ U2B[User 2<br>allocated 4K]
        U2B ~~~ GAP2[GAP<br>wasted]
    end
    subgraph AFTER["With PagedAttention"]
        PT[Page Table] --> P1[Block 1<br>User 1]
        PT --> P2[Block 2<br>User 2]
        PT --> P3[Block 3<br>User 1]
        PT --> P4[Block 4<br>User 2]
    end
    style GAP1 fill:#ffe4e6,stroke:#000
    style GAP2 fill:#ffe4e6,stroke:#000
    style PT fill:#dbeafe,stroke:#000
    style P1 fill:#dcfce7,stroke:#000
    style P2 fill:#fef3c7,stroke:#000
    style P3 fill:#dcfce7,stroke:#000
    style P4 fill:#fef3c7,stroke:#000
```

Problem: pre-allocating contiguous KV blocks wastes 50-70% to fragmentation.
Solution: page table with small blocks. Non-contiguous. Allocated on demand.

Result: **4x more concurrent users** on same hardware.
Built into vLLM. Zero code change.

> "Fragmentation fixed. But every user still pays full prefill. Can we avoid that?"

---

### Slide 20: Prefix Caching
**System prompt: computed once. Reused forever.**

Request 1: [system prompt 2000 tokens] → prefill (200ms) + decode
Request 2-N: [system prompt] → CACHE HIT (0ms) + decode only

Anthropic charges 10% for cached tokens. They save 90%.
Stripe agents: 85-97% cache hit rate.

> "From 200ms to 0ms for returning users. But what about the tokens we ARE caching? Do we need all of them?"

---

### 🎯 DEMO E: Prefix Caching (5 min)
> `demos/demo_f_prefix_caching.ipynb`
> 10 requests cold (all ~200ms) → warm (requests 2-10 at ~12ms).
> Bar chart showing the 15x improvement.

---

### Slide 21: NVIDIA Dynamo's Agentic Pattern
**Cache as a distributed session store.**

Agent: tool call → SUSPEND (KV cached with TTL) → resume → tool call → SUSPEND

11.7x read/write ratio. KV-aware routing. Priority eviction.
The frontier: treating KV cache like a database, not a buffer.

---

### Slide 22: Smart Eviction (H2O)
**90% of tokens contribute almost nothing.**

Attention follows a power law: 5% of tokens get 80% of the attention weight.
Heavy Hitter Oracle: keep those + a recent sliding window. Evict the rest.

Budget = 10-20% of full cache. Fits 5-10x longer context.

> "Kept 10% of the cache, lost <2% quality. But what about the values themselves? Can we make them smaller?"

---

### 🎯 DEMO F: Smart Eviction (5 min)
> `demos/demo_g_smart_eviction.ipynb`
> Real attention heatmap showing power law. H2O quality comparison.

---

### Slide 23: KV Quantization
**FP16 → INT8: 2x more users. One config flag.**

`--kv-cache-dtype fp8` in vLLM. Done.
Quality loss < 0.1% on most tasks (KV values are less sensitive than weights).

> "OK. We've squeezed the KV cache 4 ways. What about the model weights and the decode loop itself?"

---

## PART 4: "Three more levers beyond the KV cache." (18 min)

> **Transition:** "KV cache is one axis. These three optimize the model weights and the generation loop."

### Slide 24: Weight Quantization
**Same model. 4x less memory. Minor quality loss.**

FP16: 14.5 GB (baseline)
INT8: 7.2 GB (2x compression)
INT4 (NF4): 3.6 GB (4x compression)

More room for KV cache = more users.
Minor quality loss at INT4 on reasoning. For serving: quantize.

> "Smaller weights. But we're still generating one token at a time in a loop. Can we break that?"

---

### Slide 25: Continuous Batching
**Static batching wastes 50%+ of GPU time.**

```mermaid
flowchart LR
    A[Request 1<br>needs 16 tokens] --> B[Padded to 128<br>112 wasted]
    C[Request 2<br>needs 128 tokens] --> D[Full 128<br>0 wasted]
    style B fill:#ffe4e6,stroke:#000
    style D fill:#dcfce7,stroke:#000
```

Static: all requests padded to max length. Short ones waste compute on padding.
Continuous: each request finishes independently. Slot in new ones immediately.

Result: 2-3x throughput improvement. Built into vLLM/SGLang.

> "No more padding waste. But decode is still sequential. One token per step. Can we go faster?"

---

### Slide 26: Speculative Decoding
**Draft fast. Verify cheap.**

```mermaid
flowchart LR
    subgraph DRAFT["Draft Model (1B, fast)"]
        D1[tok1] --> D2[tok2] --> D3[tok3] --> D4[tok4] --> D5[tok5]
    end
    subgraph VERIFY["Main Model (7B, one pass)"]
        V[Verify all 5<br>in parallel]
    end
    D5 --> V
    V --> ACC[4/5 accepted ✓]
    V --> REJ[1 rejected ✗<br>resample]
    style DRAFT fill:#dbeafe,stroke:#000
    style V fill:#fef3c7,stroke:#000
    style ACC fill:#dcfce7,stroke:#000
    style REJ fill:#ffe4e6,stroke:#000
```

- Draft model generates 5 candidates sequentially (cheap, ~1ms total)
- Main model verifies all 5 in ONE forward pass (same cost as generating 1 token)
- Accepted tokens: free. Rejected: fall back to normal decode from that point.
- Acceptance rate: 70-85%. Net speedup: 2-3x. **Same output quality** (mathematically guaranteed).

> "Three more levers, all orthogonal. Let's prove them live."

---

### 🎯 DEMO H: Quantization + Batching (8 min)
> `demos/demo_i_quantization_batching.ipynb`
> 1. Real FP16 vs INT8 vs INT4: memory + throughput side by side
> 2. Real static vs continuous batching: measure padding waste
> 3. Speculative decoding (vLLM if available, AWS alternative otherwise)

---

## PART 5: "Who implements all of this?" (12 min)

> **Transition:** "You've seen 7 optimizations. The good news: you don't implement them yourself. Engines do."

### Slide 27: Three Engines, One Decision

| Engine | Implements | Best For |
|--------|-----------|----------|
| **vLLM** | PagedAttention + continuous batching + KV quant | General purpose |
| **SGLang** | RadixAttention (tree prefix cache) + structured output | Agents, multi-turn |
| **TensorRT-LLM** | AOT compilation + CUDA graphs + XQA kernels | Max throughput |

> "All three implement everything we discussed. They differ in what they're best at."

---

### Slide 28: vLLM Architecture
**The default choice. One command.**

```mermaid
flowchart LR
    subgraph API["API Layer"]
        R1[Request 1]
        R2[Request 2]
        R3[Request 3]
    end
    subgraph SCHED["Scheduler"]
        CB[Continuous<br>Batching]
    end
    subgraph MEM["Memory Management"]
        KV[KV Block Manager<br>PagedAttention]
        PT[Page Table<br>non-contiguous blocks]
    end
    subgraph GPU["GPU Workers"]
        W1[Worker 1]
        W2[Worker 2]
    end
    R1 --> CB
    R2 --> CB
    R3 --> CB
    CB --> KV
    KV --> PT
    PT --> W1
    PT --> W2
    style CB fill:#dbeafe,stroke:#000
    style KV fill:#dcfce7,stroke:#000
    style PT fill:#dcfce7,stroke:#000
    style W1 fill:#fef3c7,stroke:#000
    style W2 fill:#fef3c7,stroke:#000
```

- Requests arrive asynchronously → scheduler forms dynamic batches
- KV Block Manager allocates/frees pages as requests start/finish
- No fragmentation, no padding, continuous insertion of new requests
- 24x throughput over HuggingFace

---

### Slide 29: SGLang Architecture
**RadixAttention: a tree of cached prefixes.**

```mermaid
flowchart LR
    subgraph TREE["Radix Tree Cache"]
        ROOT["You are a helpful<br>assistant..."] --> BR1["Explain X"]
        ROOT --> BR2["Summarize Y"]
        ROOT --> BR3["Translate Z"]
    end
    subgraph ENGINE["SGLang Engine"]
        MATCH[Prefix Matcher<br>longest match lookup]
        SCHED2[Scheduler<br>+ constrained decoding]
    end
    subgraph OUT["Output"]
        O1[Response 1]
        O2[Response 2]
    end
    BR1 --> MATCH
    BR2 --> MATCH
    MATCH --> SCHED2
    SCHED2 --> O1
    SCHED2 --> O2
    style ROOT fill:#dcfce7,stroke:#000
    style MATCH fill:#dbeafe,stroke:#000
    style SCHED2 fill:#f3e8ff,stroke:#000
```

- All requests with shared system prompt share ONE cached prefix (green)
- New request: find longest matching prefix in tree → skip that prefill
- 5x cache hit rate over vLLM's hash-based approach
- Also: constrained decoding for structured output (JSON, regex)

Best for: agents with system prompts, multi-turn conversations, structured output.

---

### Slide 30: TensorRT-LLM Architecture
**Compile once. Run at hardware peak.**

```mermaid
flowchart LR
    subgraph BUILD["Build Phase (offline)"]
        M[Model<br>Checkpoint] --> OPT[Optimizer<br>layer fusion<br>kernel selection]
        OPT --> ENG[TRT Engine<br>binary]
    end
    subgraph RUN["Runtime Phase"]
        ENG --> CG[CUDA Graphs<br>zero launch overhead]
        CG --> XQA[XQA Kernels<br>per-architecture]
        XQA --> FP8[FP8 Compute<br>2x throughput]
    end
    style M fill:#f3f4f6,stroke:#000
    style OPT fill:#fef3c7,stroke:#000
    style ENG fill:#dcfce7,stroke:#000
    style CG fill:#dbeafe,stroke:#000
    style XQA fill:#dbeafe,stroke:#000
    style FP8 fill:#dbeafe,stroke:#000
```

- Build: fuses layers, selects optimal kernels for YOUR specific GPU
- Runtime: CUDA graphs eliminate kernel launch overhead entirely
- XQA: custom attention kernels per architecture (Hopper, Ada, Ampere)
- FP8: native on H100/H200, 2x throughput over FP16
- Tradeoff: compile step (minutes), NVIDIA-only

---

### Slide 31: Decision Flow

```mermaid
flowchart LR
    Q1{Multi-turn<br>agents?} -->|Yes| SG[SGLang]
    Q1 -->|No| Q2{Max single-GPU<br>throughput?}
    Q2 -->|Yes| TRT[TensorRT-LLM]
    Q2 -->|No| VL[vLLM]
    style SG fill:#dcfce7,stroke:#000
    style TRT fill:#fef3c7,stroke:#000
    style VL fill:#dbeafe,stroke:#000
```

> "Most teams start with vLLM. Switch to SGLang if you're building agents."

---

### 🎯 DEMO G: Engine Benchmark (5 min)
> `demos/demo_h_engine_comparison.ipynb`
> Same model, same prompt: HF → vLLM → SGLang. Chart + cost table.
> End with: "Same hardware. 20x cheaper per token."

---

## CLOSING: "You now know the full stack." (13 min)

### Slide 32: Stack Everything
**Waterfall: baseline to 20x.**

| Optimization | Cumulative Improvement |
|---|---|
| Baseline (HF, FP16) | 1x |
| + GQA (4x less KV) | ~1x (already in model) |
| + INT4 weights | ~4x (more users fit) |
| + PagedAttention | ~8x (no fragmentation) |
| + Prefix cache | ~8x + 15x TTFT |
| + Continuous batching | ~16x throughput |
| + Speculative decode | ~20x |

> "Same GPU. Same model. 20x more capacity."

---

### Slide 33: The Decision Framework
**What you take home.**

```mermaid
flowchart LR
    A[Pick Model] --> B[Pick Precision<br>FP16/INT8/INT4]
    B --> C[Pick GPU<br>A100/H100/H200]
    C --> D[Set SLO<br>TTFT + ITL]
    D --> E[Max Batch<br>from equation]
    E --> F[Pick Engine<br>vLLM/SGLang/TRT]
    F --> G[Cost per<br>M tokens]
    style A fill:#f3f4f6,stroke:#000
    style G fill:#dcfce7,stroke:#000
```

---

### Slide 34: What's Next (In the Repo)
**Chapters 07-11. Explore on your own.**

- Multimodal serving (Stanford M*, Walk Graphs, VoxServe)
- Kubernetes (KServe, llm-d, KAI Scheduler)
- Ray Serve, disaggregated serving (DistServe, Mooncake)
- Custom silicon (Groq LPU, Cerebras WSE-3)

---

### Slide 35: Numbers You Proved Today

| Number | What It Is |
|--------|-----------|
| 14.5 GB | Mistral-7B FP16 weight memory |
| 131 KB | KV cache cost per token |
| 80x | How far decode sits below compute peak |
| 4x | GQA savings over MHA |
| 4x | INT4 weight compression |
| 15x | Prefix caching TTFT improvement |
| 2-3x | Speculative decoding speedup |
| 24x | vLLM throughput over HuggingFace |

> "Every number was computed live. Reproducible in the notebooks."

---

### Slide 36: Resources + Q&A
**Go build.**

- GitHub: `github.com/harshuljain13/llm-inference-at-scale` (QR code)
- Molab: links for all 8 demo notebooks
- Book: Manning "LLM Inference in Action" (2027)

> "55+ modules, 12 chapters. Every concept has a lab. Questions?"

---

## Narrative Thread Summary

The story in one paragraph:

> We loaded a model and it was slow (Demo A). We learned it's because inference is memory-bound, not compute-bound (Part 1). We saw that attention architecture determines the KV cost per token, and modern models already give us 4x savings for free (Part 2). We discovered 4 techniques to optimize what's stored in the cache: page it, reuse prefixes, evict useless tokens, quantize values (Part 3). We found 3 more levers for the weights and decode loop: quantize weights, batch continuously, and decode speculatively (Part 4). And we saw that production engines implement ALL of this in one package, giving 20x improvement over naive HuggingFace (Part 5).

Each transition answers: "OK, but..." 
- "OK memory is used. But why is it SLOW?" → bandwidth wall
- "OK it's slow. But can we reduce what's stored?" → attention optimizations  
- "OK per-token cost is lower. But can we cache smarter?" → KV engineering
- "OK cache is optimized. But what about weights and the loop?" → quant + batching + speculative
- "OK there are 7 techniques. Do I implement them?" → engines do it for you
