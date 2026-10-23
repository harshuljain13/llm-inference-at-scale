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

**Both demos detect hardware, because Demo A runs on Molab.** A CUDA GPU with
≥ 18 GB VRAM takes the Mistral-7B path; anything smaller — including CPU-only —
falls back to TinyLlama-1.1B with smaller knobs, and every prompt length, decode
step count and precision follows from that one switch. Same shapes, different
absolute numbers. Two consequences worth knowing before you present:

- **On the fallback path, §07 also prints the Mistral-7B-on-80GiB arithmetic**
  (128 KiB/token → 80 users × 16K = 160 GiB), derived from published config and
  labelled as derived. That is the number the slides quote, so it has to survive
  on a machine that cannot measure it.
- **With no CUDA there is no free-VRAM reading**, so §02 falls back to a *stated*
  budget (80 GiB minus the loaded weights) and carries the label
  `ASSUMED: ...` through every table and chart that consumes it.

Mistral-7B is gated, so on the 7B path cell 00 checks `whoami()` and tells you to
log in rather than failing part-way into a download. Demo B §5 (quantization) is
the one part that genuinely requires bitsandbytes.

## Demos

Every file is prefixed `odsc_2026_` so a notebook pulled into Colab, Molab, or a
student's downloads folder is still identifiable as ODSC material.

| Demo | The question | Hardware |
|------|--------------|----------|
| [`odsc_2026_demo_a_model_memory_prefill_decode.ipynb`](demos/odsc_2026_demo_a_model_memory_prefill_decode.ipynb) | Why is serving an LLM hard? Model shape → memory → TTFT → ITL → prompt length → **how the KV cache grows** → how many users fit | Molab, Colab, or any CUDA GPU; CPU fallback |
| [`odsc_2026_demo_b_attention_and_quantization.ipynb`](demos/odsc_2026_demo_b_attention_and_quantization.ipynb) | Given that floor, how do you lower it? Shrink the KV cache (GQA/MQA/MLA) and shrink the weights (INT8/INT4) — and how many users each buys | §2–4 any device; §5 needs CUDA + bitsandbytes |

**A establishes the bottleneck; B attacks it.** That is the arc, and B's opening
paragraph picks up A's closing number deliberately.

### Demo A is a presentation, not a reader

Demo A is the version you *run on stage*. It merges AIEWF's
[`demo_b_memory_equation`](../aiewf-2026/demos/demo_b_memory_equation.ipynb) and
[`demo_c_prefill_vs_decode`](../aiewf-2026/demos/demo_c_prefill_vs_decode.ipynb)
under a single constraint: **one markdown header, one short code cell, one printed
result, one chart per section — and every cell under 30 seconds.** Nothing in it
needs the audience to read ahead.

| § | Measured | Visual |
|---|----------|--------|
| 01 | Layers, hidden size, Q/KV heads, head dim, params | Query heads coloured by which KV head they share |
| 02 | Weights, device total, free remainder (or a labelled stated budget) | Text bar + one stacked bar: weights \| free |
| 03 | One warmed, synchronized forward pass over the prompt | TTFT as a timeline bar — the "user sees nothing" window |
| 04 | Single-token steps with `past_key_values`; mean and p50 ITL | Per-step line with mean/p50 rules |
| 04b | The same steps with `use_cache=False` (optional) | Cache on vs off overlay + mean bars |
| 05 | TTFT and mean ITL across four prompt lengths | Two panels: TTFT climbs, ITL flat |
| 06 | KV bytes/token predicted from config, **weighed against the real cache** | — |
| 06a | **The cache weighed at six context lengths** | Measured points on the predicted line: exactly linear |
| 06b | Max context per user count, from the budget | Curve family + budget line, **X where each crosses** |
| 07 | Users that fit at 4K and 16K; the 80-user shortfall | Fits/OOM grid against the budget |
| 08 | Every number above, in one table | — |

The spine is §05 and §06. §05 is the one slide that pays for the whole demo: 64x
more prompt moves TTFT a lot and ITL almost nothing, so prefill and decode are two
bottlenecks and you can never quote one average. §06 then shows the KV cache
growing in both directions — **linear in context, measured; linear in users,
arithmetic** — and §07 lands on a number that does not fit, which is the handoff
to Demo B.

`§06a` is the one to not cut for time. It is the only place in either notebook
where a prediction and a measurement are drawn on the same axes and coincide, and
it is what earns the right to do pure arithmetic in `§06b` and `§07`.

Three things it deliberately does **not** do, all of which the previous draft did:

- **No `repeat()` median harness.** Warm up once, then time one synchronized pass.
  Individual numbers therefore carry single-sample noise; the per-step ITL line in
  §04 is what makes that noise visible instead of hidden behind a median.
- **No bandwidth roofline, no prefix caching, no API pricing.** Those need a reader
  who can pause. See *Deleted from this folder* below for where they went.
- **No plot-helper layer.** Each chart is 10–15 plain matplotlib lines in the
  section it belongs to, so a presenter can point at the code that drew the thing
  on screen.

### Conventions these two establish

- **No simulation.** Every figure is a timed run on the machine executing the
  notebook. The one predicted value (KV bytes/token from `config.json`) is
  labelled a prediction in A §06 and weighed against the real `past_key_values`
  in the same cell, with the error printed as a percentage.
- **No hardcoded GPU specs.** Total and free VRAM come from
  `torch.cuda.mem_get_info()`, so every capacity claim is true of the GPU in the
  room rather than of an A100 in a blog post. Where there is no GPU to ask, the
  budget is a *stated* number carrying an `ASSUMED:` label into every table and
  chart downstream of it.
- **Prefill and decode are driven by hand**, not via `generate()` — the only way
  to time the phases apart.
- **Synchronize and warm up, always.** `timed()` calls
  `torch.cuda.synchronize()` on both sides, and every shape is run once before it
  is measured. Skip either and you are timing a launch queue or a kernel compile.
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

- `demo_b_memory_equation.ipynb` — absorbed, and extended. Its equation is A §06
  with the formula now checked against the real cache; its users × context chart
  is A §06b and §07, built from the detected budget instead of a hardcoded
  `57.5`. What the original never did — and A §06a now does — is *weigh* the
  cache at a range of context lengths to show the growth is linear rather than
  asserting it.
- `demo_c_prefill_vs_decode.ipynb` — absorbed into A §03–04b, which also fixes it:
  the original's decode loop re-forwarded the whole sequence every step with no KV
  cache, so its "decode latency" grew with sequence length, contradicting its own
  "~constant per token" summary. In A §04b that uncached loop is still run — but
  deliberately, as the *contrast*, which is the only thing it was ever good for.
- `demo_e_quantization_batching.ipynb` — merged into Demo B §5.
- `module1_prefill_vs_decode.ipynb` — its prefill/decode experiments now live in
  A §03–04. Deliberately *not* carried over: the PyTorch profiler section, the
  reflection prompts, and the HTML/PDF report export — course scaffolding for a
  multi-week class, not a conference slot.

All of the above remain in [`../aiewf-2026/demos/`](../aiewf-2026/demos/).

### Cut from Demo A in the stage rewrite

The previous nine-section Demo A is the last version at commit `c9ba1c6`. Three of
its sections have no home yet and exist nowhere else, so to read or recover them:

```bash
git show c9ba1c6:workshop/odsc-2026/demos/odsc_2026_demo_a_model_memory_prefill_decode.ipynb \
  > /tmp/demo_a_nine_section.ipynb
```

| Was | What it did | Verdict |
|-----|-------------|---------|
| §5 | Shared-prefix vs dissimilar prompts at matched length — the mechanism behind vLLM prefix caching and SGLang RadixAttention. Concluded that **TTFT depends on your traffic pattern, so routing is part of your latency budget** | Worth reviving, probably as its own demo |
| §7 | Measured this GPU's bandwidth by device-to-device copy, then compared `(weights + KV) / bandwidth` against measured ms/token | Strong, but needs a reader; a chart of it may survive as a slide |
| §9 | Providers' output:input price ratio vs our measured single-stream time ratio | The best motivation for batching we have — see *Still to decide* |

Also cut: the `repeat()` median harness, the nine-prompt TTFT sweep, and the
shared plot-helper layer. (The CPU fallback was cut and then restored — Demo A
runs on Molab.)

## Still to decide

1. **Running order and time budget** once the slot length is confirmed.
2. **Batching is the missing lever, and nothing now in the folder names it.**
   The old A §9 measured it indirectly: single-stream output:input *time* ratio
   comes out ~100x or more, while providers charge only ~4–5x, and that gap is
   batching — a real server's one weight read produces a token for every user in
   the batch, not just one. That section now exists only at `c9ba1c6`. Either it
   returns as a closing slide in A, or it becomes the opening of a Demo C. It is
   the strongest case for a Demo C either way.
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
