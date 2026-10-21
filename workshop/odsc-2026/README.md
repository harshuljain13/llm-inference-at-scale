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

Demo A auto-detects your hardware: a CUDA GPU with ≥ 18 GB VRAM runs the 7B
path, anything smaller (or CPU-only) falls back to a 1.1B model with smaller
knobs. The story is identical either way; the absolute numbers are not.

## Demos

| Demo | The question it answers | Hardware |
|------|------------------------|----------|
| [`demo_a_model_memory_prefill_decode.ipynb`](demos/demo_a_model_memory_prefill_decode.ipynb) | Download a model → where its memory goes → why a long prompt is slower → prefill vs. decode → the bandwidth floor → the KV cache | GPU preferred, CPU fallback |
| [`demo_e_attention_and_quantization.ipynb`](demos/demo_e_attention_and_quantization.ipynb) | The two ways to lower that floor: shrink the KV cache (GQA/MQA/MLA) and shrink the weights (INT8/INT4) — then how many users each buys | §2–4 any device; §5 needs CUDA + bitsandbytes |

Demo A establishes the bottleneck; Demo E attacks it. That is the whole arc so far.

### Demo A is the spine

It replaces AIEWF's `demo_a` + `demo_b` + `demo_c` and the prefill/decode work
from
[`llm-inference-experiments/module1`](https://github.com/harshuljain13/llm-inference-experiments/tree/main/module1-prefill-vs-decode),
told as one continuous measurement session:

| § | Section | What gets measured |
|---|---------|--------------------|
| 1 | What did we just download? | Load time, on-disk cache size, parameter count, architecture from `config.json` |
| 2 | Where does the memory go? | Every parameter tensor bucketed by component; allocator view vs. tensor sum; GPU headroom |
| 3 | First inference | TTFT and total latency, short prompt vs. long prompt |
| 4 | Splitting the clock | Prefill (one parallel pass) vs. decode (one pass per token), both prompts, plus a prompt-length sweep |
| 5 | Why can't decode go faster? | Achievable HBM bandwidth, measured on the spot, vs. the measured per-token latency |
| 6 | The KV cache | Decode with `past_key_values` vs. `use_cache=False`; real cache weighed against the §2 prediction |

### Demo E merges both AIEWF "demo E" notebooks

AIEWF had two files sharing the `demo_e` prefix — one workshop slot, two levers. They
are now one notebook, because they answer the same question (how do you read fewer
bytes per token?) from opposite ends:

| § | Section | What gets measured |
|---|---------|--------------------|
| 2 | Controlled: MHA vs GQA-4 vs GQA-2 vs MQA | Four identical Llama models, only `num_key_value_heads` differs, randomly initialized — real prefill, real cache weighed, real decode timing |
| 3 | Real checkpoints | GPT-2 (MHA) vs TinyLlama (GQA): measured cache, then checked against the config formula |
| 4 | MLA | **Derived from DeepSeek-V2's published config, not measured** — and labelled so, with the 56x-vs-14x baseline caveat kept |
| 5 | Quantization | Same checkpoint at FP16 / INT8 / INT4: measured weight memory, throughput, ms/token |
| 6 | Both levers together | Capacity from measured VRAM + measured weights + the §3-validated KV formula |

Rebuilding the attention half was necessary, not cosmetic. The AIEWF original claimed
in its own header to "benchmark MHA vs GQA vs MLA vs FlashAttention on a real GPU" and
to "load Mistral-7B and measure actual memory and latency" — **it loaded no model and
ran no benchmark.** It was arithmetic over hardcoded constants with `gpu_bw_gbs = 2039`
(an A100 figure) baked in. §2 now does the experiment the old header promised, and does
it *controlled*, which the original could not have done even in principle by comparing
published configs.

Two other inherited defects, fixed:

- The quantization half promised perplexity in prose ("to show it barely changes") and
  never computed it. Demo E now says plainly that it measures **no** quality cost, and
  that INT4 capacity gains shouldn't be quoted without one.
- Its max-users arithmetic hardcoded an 80 GB GPU and 131 KB/token. §6 reads VRAM from
  the device and weights from §5.

Ground rules it follows, and that any new demo should too:

- **No simulation.** Every figure is a timed run on the machine executing the
  notebook. Where a value is *predicted* from `config.json` (KV bytes per token),
  it is labelled a prediction in §2 and then checked against the real cache in §6.
- **No hardcoded GPU specs.** Bandwidth is measured with a device-to-device copy
  benchmark rather than quoting a spec sheet, so the narration stays true on
  whatever GPU is in the room.
- **Prefill and decode are driven by hand**, not via `generate()` — that is the
  only way to time the two phases separately.
- Tables for numbers, charts for shape, and headings that state the question.

## Still to decide

1. **Running order and time budget** once the slot length is confirmed.
2. **Whether the throughput-ceiling story returns.** AIEWF's `demo_a` ended with
   5 sequential requests and a queue visualization — the setup for batching. It
   is *not* in the merged Demo A. It is still in
   [`../aiewf-2026/demos/demo_a_feel_the_pain.ipynb`](../aiewf-2026/demos/demo_a_feel_the_pain.ipynb)
   if ODSC wants it, probably as its own demo alongside batching.
3. **Molab/Colab hosted copies.** The badge links in these notebooks resolve
   against `master` once merged; the `molab.marimo.io/notebooks/nb_...` links in
   the AIEWF README are hosted forks of the *AIEWF* notebooks and need
   re-forking for ODSC.
4. **Engines** — `demo_f_vllm` and `demo_g_sglang` are still only in
   `../aiewf-2026/`, as is `demo_d_capacity_calculator` (whose interactive
   capacity widgets now overlap Demo E §6). Copy (don't move) any that make the
   cut, and repoint their badge paths to `workshop/odsc-2026/demos/...`.
5. **Batching** is the obvious missing lever. Demo A shows one weight read per
   token for one user; Demo E shrinks that read. Neither shows amortizing a
   single read across many users' tokens — which is the thing the engines
   actually do and the natural Demo F.

## Deleted from this folder

- `demo_b_memory_equation.ipynb` — fully absorbed. Its memory equation is Demo A
  §2 (now *measured*, with the config formula as a labelled prediction that §6
  checks), and its users × context scaling chart is Demo E §6, built from
  measured inputs instead of assumed ones.
- `demo_e_quantization_batching.ipynb` — merged into
  `demo_e_attention_and_quantization.ipynb` §5. Both originals remain in
  `../aiewf-2026/demos/`.

## Superseded sources (safe to delete after a verified run)

These two are the inputs that were merged into Demo A. They are kept only until
Demo A has been run end to end on real hardware:

- `demos/demo_c_prefill_vs_decode.ipynb` — AIEWF original. **Its decode
  measurement is wrong:** the loop re-forwards the whole sequence every step with
  no KV cache, so its "decode latency" grows with sequence length, contradicting
  the notebook's own "~constant per token" summary. Demo A §4 fixes this by
  feeding back `past_key_values`, and §6 keeps the uncached loop deliberately —
  as the *contrast*, which is the only thing it is good for.
- `demos/module1_prefill_vs_decode.ipynb` — course notebook with the correct
  cached decode path, the profiler section, and the OpenAI input-vs-output cost
  mapping. **The cost section did not make it into Demo A** and is worth adding
  back: API pricing charging 3–5× more for output tokens is the business-facing
  readout of the hardware asymmetry §4 measures.

---

## Setup

TBD. See [`../aiewf-2026/README.md`](../aiewf-2026/README.md) for the Molab +
Lightning.ai setup used previously.
