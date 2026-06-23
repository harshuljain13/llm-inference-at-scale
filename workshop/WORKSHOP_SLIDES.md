# AIE Workshop 2026: LLM Inference at Scale
## Slide Deck Content + Speaker Notes + Demo Cues

---

## PART 1: FOUNDATIONS (30 min)

---

### SLIDE 1 — Title Slide

**Heading:** LLM Inference at Scale: From First Principles to Production

**Visual:** Dark background. Large bold text. Repo QR code bottom right.
```
github.com/harshuljain13/llm-inference-at-scale
```

**Speaker notes:**
> Welcome. Today is hands-on. Every number we discuss, we verify live in a notebook. By the end you'll be able to pick a GPU, configure a serving stack, and calculate your cost per million tokens from scratch. Let's start from what actually happens inside these models.

---

### SLIDE 2 — The Transformer Layer (Animation: 4 steps)

**Heading:** One Transformer Layer

**Visual — Step 1:** Empty box labeled "Input Token"
**Visual — Step 2:** Arrow to "Multi-Head Attention" block (highlight blue)
**Visual — Step 3:** Arrow to "FFN" block (highlight green)
**Visual — Step 4:** Arrow out labeled "Output Token"

Below each block show byte sizes:
- Attention weights: `32 × 128 × 128 × 4 = 67MB per layer`
- FFN weights: `4096 × 16384 × 2 = 134MB per layer`

**Speaker notes:**
> A transformer layer is two operations: attention then FFN. The attention block computes relationships between tokens. The FFN applies a learned transformation. Both are matrix multiplications. For Mistral-7B with 32 layers, the weights alone are 14.5 GB. This is your fixed cost. It doesn't change with load.

---

### SLIDE 3 — Where Memory Goes (Animation: 3-part pie chart building)

**Heading:** GPU Memory = Weights + KV Cache + Overhead

**Visual — Step 1:** Pie slice labeled "Weights: 14.5 GB" (blue, large)
**Visual — Step 2:** Second slice appears "KV Cache: grows with users" (amber, expanding)
**Visual — Step 3:** Third slice "Overhead: ~8 GB" (gray, fixed)

Full circle shows 80 GB total (A100 line)

**Speaker notes:**
> Three buckets. Weights are fixed — 14.5 GB whether you serve 1 user or 1000. Overhead is fixed — CUDA context, activations, framework buffers, ~10%. KV cache is the variable. It scales with every user, every token. This is where the math gets interesting.

---

### SLIDE 4 — Prefill vs Decode (Animation: 2 phases)

**Heading:** Two Phases, Two Problems

**Visual — Step 1:** 
Left: "PREFILL" — show tokens processing in parallel, all arrows going simultaneously
Label: "Compute-bound, fast, one-shot"

**Visual — Step 2:**
Right: "DECODE" — show single token being generated, then another, then another (sequential)
Label: "Memory-bound, slow, sequential"

**Visual — Step 3:**
Timeline bar showing: [PREFILL 180ms] [token1 20ms] [token2 20ms] [token3 20ms]...

**Speaker notes:**
> Prefill processes your entire prompt at once. It's parallel and fast — compute-bound. Decode generates one token at a time. It MUST be sequential — each token depends on the previous. This is why you can throw more GPU compute at prefill but not at decode. Decode is memory-bound. That's the core constraint of everything we'll talk about today.

---

### SLIDE 5 — The KV Cache (Animation: token-by-token)

**Heading:** Why the KV Cache Exists

**Visual — Step 1:** Show sequence "The cat sat on the..." — highlight "The"
Attention block reads all previous tokens' K,V vectors

**Visual — Step 2:** "Without KV cache" → recompute all K,V at every step → O(n²) operations

**Visual — Step 3:** "With KV cache" → store K,V vectors as you go → O(n) at each step

**Visual — Step 4:** Show the cache growing: 1 token → 131 KB, 100 tokens → 13 MB, 4096 tokens → 537 MB

**Speaker notes:**
> At each decode step, the model needs the key and value vectors for every previous token to compute attention. Without caching, you'd recompute them all — that's quadratic. The KV cache stores them so you only compute once. The trade: memory grows linearly with context. For Mistral-7B with GQA: 131 KB per token. 4K context per user: 537 MB. 80 users: 43 GB just for KV.

**🎯 DEMO CUE:** Switch to `workshop/01_memory_equation.ipynb`
- Cell 3: Load Mistral-7B config, print architecture numbers
- Cell 5-6: Derive 131 KB/token formula live
- Cell 7: Plot KV cache growing as context × users scales
- *Time: 4 minutes*

---

### SLIDE 6 — Why LLM Inference is Different (Animation: comparison)

**Heading:** Not Like Any Other Web Service

**Visual:** Two columns

Left — "Traditional API":
- Request in → Process → Response out
- Stateless
- Scale = more instances

Right — "LLM Inference":
- Request in → [KV cache allocated] → [tokens stream out one by one]
- Stateful (KV cache lives in GPU memory)
- Scale = more memory management complexity

**Speaker notes:**
> Traditional services are stateless. Each request is independent. LLM inference is stateful — the KV cache for an ongoing conversation must stay in GPU memory until the conversation ends. You can't just add more instances to handle burst traffic without also moving state. This is what makes LLM serving a hard systems problem.

---

### SLIDE 7 — GPU Memory Hierarchy (Animation: 3 tiers)

**Heading:** Your GPU Has Three Memory Tiers

**Visual — Step 1:** Large box "HBM (80 GB, 2 TB/s)" — the big slow layer
**Visual — Step 2:** Medium box "L2 Cache (40 MB, 12 TB/s)" — medium
**Visual — Step 3:** Small box "SRAM / Registers (20 MB, 19 TB/s per SM)" — tiny fast

Show data flowing: model weights live in HBM → pulled into L2 → computation in SRAM

**Speaker notes:**
> The A100 has 80 GB of HBM — High Bandwidth Memory. That's where your model weights and KV cache live. Reading from HBM is fast by CPU standards — 2 TB per second — but it's the bottleneck for decode. The SRAM inside each SM is 200x faster but only 20 MB total. You can't fit your model there. Every decode step is fundamentally limited by how fast you can move data out of HBM.

---

### SLIDE 8 — The Roofline (Animation: plot building)

**Heading:** Where Decode Lives on the Roofline

**Visual — Step 1:** Empty axes. X = Arithmetic Intensity (FLOP/byte), Y = Performance (TFLOPS)

**Visual — Step 2:** Draw bandwidth line from bottom-left to top-right
Label: "Memory bandwidth limit (2 TB/s)"

**Visual — Step 3:** Draw horizontal roof line
Label: "Compute limit (312 TFLOPS)"

**Visual — Step 4:** Mark ridgeline intersection at 156 FLOP/byte

**Visual — Step 5:** Place red dot at far left (2 FLOP/byte) — label "Decode"
Place green dot near ridgeline — label "Prefill"
Place blue dot far right — label "Training"

**Speaker notes:**
> This is the roofline model. Left of the ridgeline: you're memory-bound. Right: compute-bound. Decode sits at 2 FLOP/byte — we do 2 multiplications for every byte we read. Training sits near the ridgeline. The distance between where decode sits and where compute maxes out represents idle GPU compute. We're using maybe 1% of the 312 TFLOPS available. Adding more compute doesn't help decode at all.

**🎯 DEMO CUE:** Switch to Ch01 roofline lab
- Show roofline plot with decode marked
- Show arithmetic intensity calculation live
- *Time: 3 minutes*

---

### SLIDE 9 — The Batch-Latency Tradeoff (Animation: curve building)

**Heading:** More Users = Slower Per User

**Visual — Step 1:** Formula: `ITL = (weights + batch × KV_per_user) / bandwidth`

**Visual — Step 2:** Plot builds: X = batch size, Y = inter-token latency (ms)
Line rises from left (8ms at batch=1) to right

**Visual — Step 3:** Horizontal SLO line appears at 20ms
Green region below = SLO met. Red region above = SLO violated.

**Visual — Step 4:** Dot appears at intersection: "Optimal batch = 80"
Show annotation: "4,000 tokens/sec at 20ms ITL"

**Speaker notes:**
> Each additional user in the batch means more KV cache to read per decode step. ITL — inter-token latency — grows linearly with batch size. Your SLO draws a line. Everything below it is acceptable. The optimal batch is the largest batch that stays under your SLO. That number determines your throughput and your cost.

**🎯 DEMO CUE:** Switch to Ch02 capacity planning lab
- Run capacity_calc widget: Mistral-7B, A100-80, FP16, 2048 tokens
- Slide num_users: watch the stacked bar fill up
- Switch to GPU selection widget: show cost comparison chart
- *Time: 5 minutes*

---

## PART 2: ATTENTION OPTIMIZATIONS (25 min)

---

### SLIDE 10 — The Attention Zoo (Animation: 4 variants)

**Heading:** MHA → MQA → GQA → MLA: The Memory Story

**Visual — Step 1:** MHA — 32 query heads, 32 KV heads, all separate
Show: 32 × 128 × 32 × 2 = 524 KB/token

**Visual — Step 2:** MQA — 32 query heads, 1 KV head shared
Show: 1 × 128 × 32 × 2 = 16 KB/token (33x savings)

**Visual — Step 3:** GQA — 32 query heads, 8 KV heads (groups of 4)
Show: 8 × 128 × 32 × 2 = 131 KB/token (4x savings vs MHA)

**Visual — Step 4:** MLA — single compressed latent vector (512 dim)
Show: 512 + 64 = 576 values = 1.1 KB/token (474x savings vs MHA)

**Speaker notes:**
> Four variants, four different memory profiles. MHA stores full K,V for every head. MQA goes extreme — one shared K,V. GQA is the production sweet spot: groups of 4 query heads share one K,V pair. Mistral, Llama 3, and every major 2024 model uses GQA-8. That 131 KB/token you saw? That's GQA already in action. Without it you'd be at 524 KB. MLA is DeepSeek's innovation — they compress the KV into a latent vector. Still experimental, not in mainstream engines yet.

---

### SLIDE 11 — FlashAttention (Animation: memory traffic)

**Heading:** Same Math, 6x Less Memory Traffic

**Visual — Step 1:** Standard attention — show 3 kernel passes:
Pass 1: Q×K → write S to HBM (N² matrix)
Pass 2: softmax(S) → read S, write P to HBM (N² matrix)  
Pass 3: P×V → read P, write output

Total HBM writes: 3N² elements

**Visual — Step 2:** FlashAttention — show 1 tiled pass:
Load tile of Q, K, V into SRAM → compute attention for this tile → accumulate into output
Never write N² matrix to HBM

Total HBM writes: N elements (output only)

**Visual — Step 3:** Bar chart: Standard = 3×4096² = 50M writes. Flash = 4096 = 4K writes. 12,000x less.

**Speaker notes:**
> FlashAttention computes the same result with the same floating point operations. The difference is memory traffic. Standard attention materializes the full N×N score matrix — that's quadratic in sequence length. At 32K context this is 4 GB of HBM writes per attention layer per step. FlashAttention tiles the computation into SRAM — never writes the intermediate matrix. Same output, 12,000x less HBM traffic for the attention computation. This is now default in PyTorch SDPA and all major engines.

---

### SLIDE 12 — Attention Comparison (Demo slide)

**Heading:** Let's Measure It

**Visual:** Split screen — left side for slides, right for notebook

**Speaker notes:**
> Let's verify all four variants with real GPU measurements.

**🎯 DEMO CUE:** Switch to Ch03 comparison lab (03.6)
- Cell 1: Load Mistral-7B
- Cell 5: Measure real GQA KV cache memory
- Cell 7: Simulate MHA expansion — show 4x memory difference
- Cell 13: Final comparison chart — GQA vs MHA vs FlashAttention
- *Time: 8 minutes*

---

## PART 3: KV CACHE ENGINEERING (30 min)

---

### SLIDE 13 — Four Levers (Overview)

**Heading:** Four Ways to Serve More Users

**Visual:** 2×2 grid

Top-left: **PagedAttention** — "No wasted VRAM" → 4x concurrent requests
Top-right: **KV Quantization** — "Smaller values" → 2-4x memory savings
Bottom-left: **Smart Eviction** — "Forget irrelevant tokens" → 10x longer context
Bottom-right: **Prefix Caching** — "Pay once, reuse forever" → 15x TTFT improvement

**Speaker notes:**
> Four independent optimizations. Each attacks a different aspect of the KV cache problem. They compose — you can stack all four. Let's take each one.

---

### SLIDE 14 — PagedAttention (Animation: memory blocks)

**Heading:** PagedAttention: Virtual Memory for KV Cache

**Visual — Step 1:** Show naive allocation: Request A gets 512MB reserved block. Uses only 50MB. 90% waste.

**Visual — Step 2:** Show paged allocation: KV cache split into 16-token blocks. Blocks allocated on demand.

**Visual — Step 3:** Block table: logical sequence → physical block mapping (like OS page table)

**Visual — Step 4:** Bar chart: Naive (10 users, 90% fragmentation) vs Paged (40+ users, <4% fragmentation)

**Speaker notes:**
> Before PagedAttention, serving engines pre-allocated the maximum context length for every request. A request that generates 50 tokens still locked 4096 tokens of memory. PagedAttention, from the vLLM paper, manages KV cache the way an OS manages RAM — virtual pages mapped to physical blocks, allocated on demand. This alone is why vLLM could serve 4x more requests than naive HuggingFace inference on the same hardware. It's now in every production serving engine.

---

### SLIDE 15 — KV Quantization (Animation: precision ladder)

**Heading:** Quantize the Cache, Not Just the Weights

**Visual — Step 1:** FP16 KV vector (16 bits per value) → baseline memory, baseline quality

**Visual — Step 2:** INT8 KV → 2x fewer bytes → 2x more users → quality impact: <0.1% on most benchmarks

**Visual — Step 3:** INT4 KV → 4x fewer bytes → 4x more users → quality impact: 0.5-1% on reasoning

**Visual — Step 4:** Bar chart showing: users per GPU at each precision level for Mistral-7B on A100

**Speaker notes:**
> KV values are less sensitive to quantization than weights. Weights need to be precise because errors accumulate through the network. KV values are intermediate results — quantization error doesn't compound the same way. INT8 KV is essentially free quality-wise. Enable it in vLLM with `--kv-cache-dtype fp8`. One flag, 2x capacity.

**🎯 DEMO CUE:** Switch to Ch04 compression lab (04.2)
- Show memory vs quality tradeoff chart
- Highlight INT8 as the sweet spot
- *Time: 3 minutes*

---

### SLIDE 16 — Smart Eviction (Animation: attention heatmap)

**Heading:** 90% of Tokens Get Less Than 5% of Attention

**Visual — Step 1:** Show attention heatmap for a layer: most values near zero, few bright spots

**Visual — Step 2:** Sorted bar chart: attention weight distribution — power law shape, long tail

**Visual — Step 3:** H2O algorithm: keep "heavy hitters" (top K by accumulated score) + recent window
Show memory budget: 20% of full cache, fits 5x longer context

**Visual — Step 4:** Quality comparison: H2O vs full cache on generation task — nearly identical

**Speaker notes:**
> Attention is not uniform. In any long context, a small set of tokens — the system prompt, key facts, recent context — absorb most of the attention weight. The rest contribute almost nothing. H2O (Heavy Hitter Oracle) tracks which tokens are "heavy hitters" and evicts the rest. With 20% of the KV budget, it retains >95% of generation quality on most tasks. This is what allows 128K context on a 24GB A10G.

**🎯 DEMO CUE:** Switch to Ch03 smart caching lab (03.3)
- Load Mistral-7B, run forward pass with output_attentions=True
- Show attention distribution: "See the power law"
- Show H2O vs random eviction comparison
- *Time: 4 minutes*

---

### SLIDE 17 — Prefix Caching (Animation: request flow)

**Heading:** Pay Once. Reuse Forever.

**Visual — Step 1:** Standard flow: User 1 sends [system prompt + query]. System prompt prefilled: 180ms.
User 2 sends [same system prompt + different query]. Same 180ms again.
100 users = 100 × 180ms = 18,000ms of wasted prefill.

**Visual — Step 2:** With prefix caching: User 1's system prompt KV stored in radix trie.
User 2 arrives → cache hit → skip prefill entirely.
Users 2-100: 12ms TTFT instead of 180ms.

**Visual — Step 3:** Show Anthropic pricing: "Cached: $0.30/M tokens. Uncached: $3.00/M tokens."
"10x price difference = 10x serving cost difference"

**Speaker notes:**
> If your application has a system prompt — and most do — every user is paying the full prefill cost for identical computation. Prefix caching is memoization for the attention mechanism. The first user pays. Every subsequent user gets the cached KV tensors. Anthropic exposes this to customers and charges 10x less for cached tokens. That price difference reflects the real serving cost. Enable in vLLM: `--enable-prefix-caching`. Enable in SGLang: on by default.

**🎯 DEMO CUE:** Switch to Ch04 prefix caching lab (04.5)
- Show cold vs warm prefill timing: "190ms vs 12ms"
- Show how prefix length determines speedup
- *Time: 4 minutes*

---

## PART 4: THE DEPLOYMENT DECISION (15 min)

---

### SLIDE 18 — Stack All Four (Animation: multipliers)

**Heading:** Compose the Optimizations

**Visual — Step 1:** Baseline: Mistral-7B, A100, FP16, no optimizations → 80 users max

**Visual — Step 2:** + GQA (already in model): 80 users (was 20 without GQA)

**Visual — Step 3:** + PagedAttention: 120 users (reduced fragmentation)

**Visual — Step 4:** + INT8 KV: 240 users (2x KV capacity)

**Visual — Step 5:** + Prefix caching: 240 users but 15x better TTFT for returning users

**Visual — Step 6:** + H2O at 50% budget: 480 users at shorter effective context

**Speaker notes:**
> These optimizations are mostly independent. GQA is already in your model. PagedAttention is in vLLM by default. INT8 KV is one flag. Prefix caching is one flag. H2O requires a custom serving setup but is available in SGLang. Stack them all: you've gone from 20 users to 480 on the same hardware with no model changes.

---

### SLIDE 19 — The Decision Framework (Animation: flowchart)

**Heading:** How to Pick Your Config

**Visual — Step 1:** Input: model, context length, SLO (max ITL ms)

**Visual — Step 2:** Formula builds:
```
weight_gb = params × bytes_per_param
kv_per_user = 2 × kv_heads × head_dim × layers × bytes × tokens
max_users = (GPU_vram - weights - overhead) / kv_per_user
max_batch_slo = SLO_ms × bandwidth / (weight_gb + kv_per_user)
throughput = batch × (1000 / ITL)
cost_per_M = ($/hr / throughput) × 277,800
```

**Visual — Step 3:** Table: GPU options ranked by $/M tokens for this config

**Speaker notes:**
> This is the complete decision process. Five formulas. You can run this in 30 seconds with any model and GPU. Let's do it live.

**🎯 DEMO CUE:** Switch to Ch02 GPU selection widget
- Set Mistral-7B, FP16, 2048 tokens, 20ms SLO
- Show table: which GPUs pass, which fail
- Show cost chart: H200 at $0.13/M tokens wins
- Slide SLO from 20ms to 50ms: watch more GPUs become viable
- *Time: 6 minutes*

---

### SLIDE 20 — Where to Go Next

**Heading:** You've Covered Chapters 00-04. Here's What's Next.

**Visual:** Repo structure showing chapters with brief descriptions

| Chapter | Topic |
|---------|-------|
| Ch05 | Quantization deep dive |
| Ch06 | Serving engines (vLLM, SGLang, TRT-LLM) |
| Ch07 | Scaling: tensor parallelism, pipeline parallelism |
| Ch08 | Production: monitoring, autoscaling, cost optimization |
| Ch09-11 | Advanced: speculative decoding, disaggregated serving |

Below: QR code to repo

**Speaker notes:**
> Everything we covered today is chapters 00-04 in the repo. 55 modules total, each with a lab notebook. The repo is the full curriculum. Today was the tasting menu. Every number we computed is reproducible — open any lab on Molab, run it yourself. Links on screen. Questions?

---

## APPENDIX: SPEAKER DEMO CHECKLIST

Before the workshop, verify these demos run end-to-end on Molab:

- [ ] `workshop/01_memory_equation.ipynb` — cells 3, 5-7 (no vLLM needed)
- [ ] `content/02_sizing_and_serving/02.1_capacity_planning/lab.ipynb` — cells 3, 5, 7 (widgets)
- [ ] `content/01_gpu_hardware/01.2_roofline_model/lab.ipynb` — roofline chart
- [ ] `content/03_attention_variants/03.6_comparison/lab.ipynb` — GQA vs MHA chart
- [ ] `content/03_attention_variants/03.3_gqa_deep_dive/lab.ipynb` — KV growth chart
- [ ] `content/04_kv_cache_engineering/04.2_kv_cache_compression/lab.ipynb` — compression chart
- [ ] `content/04_kv_cache_engineering/04.3_smart_kv_caching/lab.ipynb` — attention power law
- [ ] `content/04_kv_cache_engineering/04.5_prefix_caching/lab.ipynb` — TTFT cold vs warm

## SLIDE COUNT SUMMARY

| Part | Slides |
|------|--------|
| Foundations (Ch00-02) | 9 slides + 3 demos |
| Attention optimizations (Ch03) | 3 slides + 1 demo |
| KV cache engineering (Ch04) | 5 slides + 3 demos |
| Deployment decision | 3 slides + 1 demo |
| **Total** | **20 slides + 8 demos** |

