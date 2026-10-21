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
| [`odsc_2026_demo_a_model_memory_prefill_decode.ipynb`](demos/odsc_2026_demo_a_model_memory_prefill_decode.ipynb) | Why is serving an LLM hard? Download → memory → latency → prefill vs decode (with and without KV cache) → the bandwidth floor | GPU preferred, CPU fallback |
| [`odsc_2026_demo_b_attention_and_quantization.ipynb`](demos/odsc_2026_demo_b_attention_and_quantization.ipynb) | Given that floor, how do you lower it? Shrink the KV cache (GQA/MQA/MLA) and shrink the weights (INT8/INT4) — and how many users each buys | §2–4 any device; §5 needs CUDA + bitsandbytes |

**A establishes the bottleneck; B attacks it.** That is the arc, and B's opening
paragraph picks up A's closing number deliberately.

### Demo A is one argument, not six topics

Each section exists because the previous one raised a question:

| § | Measures | Which raises |
|---|----------|--------------|
| 1 | Load time, params, **weights in GB** | What else needs memory, and how much is left? |
| 2 | Every tensor bucketed; allocator vs sum; free VRAM; KV bytes/token *predicted* | Fine — but what does a user experience? |
| 3 | TTFT and total, short vs long prompt | Where inside that wait did the time go? |
| 4 | **2 prompts × 2 cache settings = 4 runs**, prefill and decode timed separately | Decode is flat but slow. What is it waiting on? |
| 5 | This GPU's bandwidth (measured), vs measured per-token latency | So what does that cache weigh? |
| 6 | Real cache weighed; §2's prediction settled | Floor = (weights + KV) / bandwidth → Demo B |

§4 is the hinge, and it runs the full grid on purpose. The two-dimensional
comparison shows something neither axis alone does:

- **Prefill is identical with and without the cache** — it processes the prompt
  in one parallel pass either way; the only difference is whether K and V are
  *kept*. So the cache is a decode optimization, not a prompt one.
- **Decode lives or dies by it**, and the long prompt is hit far harder, because
  every uncached step re-forwards its entire context.
- **Cached per-step times are flat; uncached ones climb.** The table prints
  `First step` and `Last step` side by side so this is visible as numbers, not
  only as a chart shape.
- **Even cached, decode is far slower per token than prefill** — which is what
  sends you into §5.

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

All three originals remain in [`../aiewf-2026/demos/`](../aiewf-2026/demos/).

## Still here, still undecided

- `demos/module1_prefill_vs_decode.ipynb` — the course notebook, kept for one
  reason: its **OpenAI input-vs-output cost mapping** did not make it into Demo A,
  and API pricing charging 3–5× more for output tokens is the business-facing
  readout of the asymmetry A §4 measures. Either fold that section into A §7 or
  delete this file; it should not stay as a loose third notebook.

## Still to decide

1. **Running order and time budget** once the slot length is confirmed.
2. **Batching is the missing lever.** A shows one weight read per token for one
   user; B shrinks that read. Neither shows amortizing a single read across many
   users' tokens — which is what the engines actually do, and the natural Demo C.
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
