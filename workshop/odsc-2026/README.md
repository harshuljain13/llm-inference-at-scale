# Serving LLMs Efficiently at Scale: Workshop — ODSC 2026

By [Harshul Jain](https://github.com/harshuljain13) |
[Slides](https://drive.google.com/drive/folders/1j7GYjcLetHQjrRp9dhnS5wiLOT3dtz7A?usp=sharing)

**Status:** in preparation. Duration and final running order TBD.

---

## Quick start

```bash
git clone https://github.com/harshuljain13/llm-inference-at-scale.git
cd llm-inference-at-scale/workshop/odsc-2026/demos
```

Both demos auto-detect hardware: a CUDA GPU with ≥ 18 GB VRAM runs the 7B path,
anything smaller (or CPU-only) falls back to a 1.1B model with smaller knobs.
Same shape of result, different absolute numbers. Demo B §5 (quantization) is
the one part that genuinely requires CUDA + bitsandbytes.

## Demos

Every file is prefixed `odsc_2026_` so a notebook pulled into Colab, Molab, or a
student's downloads folder is still identifiable as ODSC material.

| Demo | The question | Hardware |
|------|--------------|----------|
| [`odsc_2026_demo_a_model_memory_prefill_decode.ipynb`](demos/odsc_2026_demo_a_model_memory_prefill_decode.ipynb) | Why is serving an LLM hard? Download → memory → latency → prefill vs decode (with and without KV cache) → the bandwidth floor → the API bill | GPU preferred, CPU fallback |
| [`odsc_2026_demo_b_attention_and_quantization.ipynb`](demos/odsc_2026_demo_b_attention_and_quantization.ipynb) | Given that floor, how do you lower it? Shrink the KV cache (GQA/MQA/MLA) and shrink the weights (INT8/INT4) — and how many users each buys | §2–4 any device; §5 needs CUDA + bitsandbytes |

**A establishes the bottleneck; B attacks it.** That is the arc, and B's opening
paragraph picks up A's closing number deliberately.

### Demo A is one argument, not nine topics

Each section exists because the previous one raised a question:

| § | Measures | Which raises |
|---|----------|--------------|
| 1 | Load time, params, **weights in GB** | What else needs memory? |
| 2 | Every tensor bucketed; free VRAM; KV bytes/token *predicted* | What does a user experience? |
| 3 | **TTFT across 5 short + 4 long prompts**, median of N with range | Where did that latency go? |
| 4 | Prefill vs decode for all 9 prompts | Can we skip prefill? |
| 5 | **Similar (shared-prefix) vs dissimilar long prompts**, length-matched | Why is decode flat? |
| 6 | Cached vs uncached decode, over enough steps for O(n) to show | Flat but slow — waiting on what? |
| 7 | This GPU's bandwidth (measured) vs measured per-token latency | What does the cache weigh? |
| 8 | Real caches weighed; §2's prediction settled | What does it cost in money? |
| 9 | Providers' output:input price ratio vs **our measured** time ratio | Floor → Demo B |

### Benchmark discipline, enforced in code

An earlier draft produced numbers that looked broken — a short prompt showing *no*
KV-cache benefit, and an uncached long prompt whose per-step cost appeared to
*shrink*. The measurements were right; the experiment was wrong. n=1 per class,
no repeats, and a growth test run over a range where the sequence grew 0.3%.

The `repeat()` harness in §0 now enforces three rules, and every table carries `n`
and a low–high range so the noise is visible:

1. **Synchronize** around every timed region — otherwise you time a launch queue.
2. **Warm up** first — the first call compiles kernels and allocates peaks.
3. **Repeat, report the median with its range** — one sample is an anecdote.

Design consequences worth keeping:

- **Nine prompts, not two.** Short prompts are dominated by fixed per-call
  overhead, which is why a 9-token prompt measures ~400 tok/s of "prefill" and a
  4096-token one measures ~14,000. The `ms per input token` column makes that
  visible instead of mysterious.
- **§6 starts from a short prompt** and decodes 64 steps, so the sequence
  multiplies several times and O(n) growth is actually detectable. The notebook
  states plainly why the same test on a 4096-token prompt shows nothing.

### §5: the experiment that replaced the cache on/off toggle

Real traffic shares prefixes — RAG resends the same document, agents replay the
same system prompt, chatbots resend the conversation. So §5 measures two families
at **matched total input length**:

| Family | Construction | Reuse available? |
|--------|--------------|------------------|
| similar | one shared prefix + a different short suffix per request | yes — prefix K/V computed once |
| dissimilar | a different source text per request, same total length | no — every token is new |

Identical length, identical model; the only variable is prefix sharing. The
conclusion is one most capacity planning misses: **TTFT depends on your traffic
pattern, and routing is part of your latency budget** — round-robin throws away a
warm prefix. This is the mechanism behind vLLM prefix caching and SGLang's
RadixAttention.

### Conventions these two establish

- **No simulation.** Every figure is a timed run on the machine executing the
  notebook. The one predicted value (KV bytes/token from `config.json`) is
  labelled a prediction in A §2 and weighed against the real cache in A §6.
- **No hardcoded GPU specs.** Bandwidth comes from a device-to-device copy
  benchmark, so the narration stays true on whatever GPU is in the room.
- **Prefill and decode are driven by hand**, not via `generate()` — the only way
  to time the phases apart.
- **Helpers on top, story underneath.** Both notebooks open with one identical
  `# --- Plot helpers` cell (`bars`, `hbars`, `stacked`, `grouped`, `curves`,
  `panels`) plus a measurement-helper cell (`timed`, `prompt_of`, `cache_bytes`).
  After that, a chart is two or three lines and a reader can see the point
  instead of the axis bookkeeping. Keep it that way: if a story cell is growing
  matplotlib boilerplate, the helper is missing an argument.
- **Derived numbers are labelled derived.** Demo B §4 (MLA) is arithmetic from
  DeepSeek-V2's published config and says so in the output, the chart, and the
  prose.

### Demo B merges both AIEWF "demo E" notebooks

AIEWF had two files sharing the `demo_e` prefix — one slot, two levers:

| § | Section | What gets measured |
|---|---------|--------------------|
| 2 | Controlled: MHA vs GQA-4 vs GQA-2 vs MQA | Four identical Llama models, only `num_key_value_heads` differs — real prefill, real cache weighed, real decode timing |
| 3 | Real checkpoints | GPT-2 (MHA) vs TinyLlama (GQA), then checked against the config formula |
| 4 | MLA | **Derived from published config, not measured** — with the 56x-vs-14x baseline caveat |
| 5 | Quantization | FP16 / INT8 / INT4: weight memory, throughput, ms/token |
| 6 | Both levers | Capacity from measured VRAM + measured weights + the §3-validated formula |

Rebuilding the attention half was necessary. The AIEWF original claimed in its
own header to "benchmark MHA vs GQA vs MLA vs FlashAttention on a real GPU" and
to "load Mistral-7B and measure actual memory and latency" — **it loaded no model
and ran no benchmark.** It was arithmetic over hardcoded constants with
`gpu_bw_gbs = 2039` baked in.

## Deleted from this folder

- `demo_b_memory_equation.ipynb` — absorbed. Its equation is A §2 (now measured,
  with the formula as a checked prediction); its users × context chart is B §6,
  built from measured inputs.
- `demo_c_prefill_vs_decode.ipynb` — absorbed into A §4, which also fixes it: the
  original's decode loop re-forwarded the whole sequence every step with no KV
  cache, so its "decode latency" grew with sequence length, contradicting its own
  "~constant per token" summary. In A §4 that uncached loop is still run — but
  deliberately, as the *contrast*, which is the only thing it was ever good for.
- `demo_e_quantization_batching.ipynb` — merged into Demo B §5.
- `module1_prefill_vs_decode.ipynb` — its experiments now live in Demo A: the
  cached/uncached prefill-decode behaviour in §4, token counts and latency in
  §3–4, and its **cost mapping** as the new §7. Deliberately *not* carried over:
  the PyTorch profiler section, the reflection prompts, and the HTML/PDF report
  export — course scaffolding for a multi-week class, not a conference slot.

All three originals remain in [`../aiewf-2026/demos/`](../aiewf-2026/demos/).

## Still to decide

1. **Running order and time budget** once the slot length is confirmed.
2. **Batching is the missing lever — and A §7 now proves it quantitatively.**
   Our measured single-stream output:input *time* ratio comes out ~100x or more,
   while providers charge only ~4–5x. That gap is batching: a real server's one
   weight read produces a token for every user in the batch, not just one. A §7
   names this and points at it; nothing in A or B measures it. This is the
   strongest case for a Demo C.
3. **Whether the throughput-ceiling story returns.** AIEWF's `demo_a` ended with
   5 sequential requests and a queue visualization — the setup for batching.
   Still in [`../aiewf-2026/demos/demo_a_feel_the_pain.ipynb`](../aiewf-2026/demos/demo_a_feel_the_pain.ipynb).
4. **Engines** — `demo_f_vllm`, `demo_g_sglang`, and `demo_d_capacity_calculator`
   (whose capacity widgets now overlap B §6) are still only in `../aiewf-2026/`.
   Copy (don't move) any that make the cut, and repoint their badge paths.
5. **Molab/Colab hosted copies.** Badge links resolve against `master` once
   merged; the `molab.marimo.io/notebooks/nb_...` links in the AIEWF README are
   hosted forks of the *AIEWF* notebooks and need re-forking for ODSC.

---

## Setup

TBD. See [`../aiewf-2026/README.md`](../aiewf-2026/README.md) for the Molab +
Lightning.ai setup used previously.
