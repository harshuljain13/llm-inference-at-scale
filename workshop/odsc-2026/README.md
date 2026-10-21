# LLM Inference at Scale: Workshop — ODSC 2026

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
| [`demo_b_memory_equation.ipynb`](demos/demo_b_memory_equation.ipynb) | The memory equation from `config.json`: weights vs. KV, scaled across users × context | Any (config math, no model load) |
| [`demo_e_quantization_batching.ipynb`](demos/demo_e_quantization_batching.ipynb) | FP16 vs. INT8 vs. INT4: memory, throughput, implied max users | CUDA + bitsandbytes |

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
4. **Attention variants and engines** — `demo_e_attention_comparison`,
   `demo_d_capacity_calculator`, `demo_f_vllm`, `demo_g_sglang` are still only in
   `../aiewf-2026/`. Copy (don't move) any that make the cut, and repoint their
   badge paths to `workshop/odsc-2026/demos/...`.

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
