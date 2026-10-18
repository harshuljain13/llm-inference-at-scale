# LLM Inference at Scale: Workshop — ODSC 2026

**Status:** in preparation. Duration and final outline TBD.

By [Harshul Jain](https://github.com/harshuljain13)

---

## Scope so far

One topic is committed: **prefill vs. decode** — an LLM call is two different
workloads glued together, a compute-bound prefill and a memory-bound decode, and
that asymmetry explains the rest of the stack.

Two existing notebooks cover it from different angles. Both are copied into
[`demos/`](demos/) **to be combined into a single ODSC notebook** — they are not
both meant to ship.

| Notebook | Source | Hardware | Covers |
|----------|--------|----------|--------|
| [`demos/demo_c_prefill_vs_decode.ipynb`](demos/demo_c_prefill_vs_decode.ipynb) | `../aiewf-2026/demos/` | **CUDA required**, Mistral-7B fp16 | Prefill sweep 128→8192 tokens; per-step decode timing; roofline plot |
| [`demos/module1_prefill_vs_decode.ipynb`](demos/module1_prefill_vs_decode.ipynb) | [`llm-inference-experiments`](https://github.com/harshuljain13/llm-inference-experiments/tree/main/module1-prefill-vs-decode) @ `aeb6bdb` | CPU / MPS / CUDA auto-detect, GPT-2 / TinyLlama | Timed prefill+decode; **with vs. without KV cache**; PyTorch profiler; **OpenAI input-vs-output cost mapping**; results table; HTML/PDF export; reflection |

Outputs and the orphaned `metadata.widgets` blob were stripped from
`module1_prefill_vs_decode.ipynb`; the version with a recorded run lives in the
source repo. Its companion write-up is
[`module1-prefill-vs-decode/overview.md`](https://github.com/harshuljain13/llm-inference-experiments/blob/main/module1-prefill-vs-decode/overview.md).

## Combining the two

What each one uniquely contributes:

- **Keep from `demo_c`:** the prefill length sweep (the parallel-scaling point
  needs more than one prompt length) and the roofline plot (the clearest "decode
  cannot use the compute you paid for" visual).
- **Keep from `module1`:** device auto-detect, the with/without-KV-cache
  comparison, and the cost section — API providers charging 3–5× more for output
  tokens is a direct readout of the hardware asymmetry, and it lands with any
  audience.

Decisions to make before merging:

1. **Hardware floor.** `demo_c` hard-requires an A100-class GPU and a gated
   7B model; `module1` runs on a laptop. Pick one target, or make the sweep
   degrade to GPT-2 so nobody is blocked at minute five.
2. **Reconcile the numbers.** `demo_c` hardcodes A100 specs (312 TFLOPS,
   2 TB/s, "reads 14.5 GB per step") in prose and in the roofline. On any other
   GPU the narration is wrong. Derive these from the loaded model and
   `torch.cuda.get_device_properties` instead of asserting them.
3. **Trim the course scaffolding.** `module1`'s reflection, report export, and
   results-table-across-sessions sections serve a multi-week class, not a
   conference slot.
4. **Scale to the slot.** 11 cells vs. 40; the merged notebook needs a stated
   time budget before it can be cut to fit.

Known defects to fix while merging, not inherit:

- `demo_c` cell 1 computes `gpu_bw_tbs` from `total_memory` — that is capacity,
  not bandwidth, and the variable is never used.
- `demo_c`'s roofline cell builds `roofline_vals` and then discards it in favour
  of `roof_perf`.
- `demo_c` Experiment 2 re-runs a **full forward pass over the whole sequence**
  each step with no KV cache, so its "decode latency" grows with sequence length
  and overstates per-token cost. `module1`'s cached decode path is the correct
  measurement — and `demo_c`'s own summary claim ("~constant per token") is what
  the uncached loop fails to show.

## Carry-over candidates (not yet copied)

From [`../aiewf-2026/`](../aiewf-2026/): `demo_a_feel_the_pain`,
`demo_b_memory_equation`, `demo_d_capacity_calculator`,
`demo_e_attention_comparison`, `demo_e_quantization_batching`, `demo_f_vllm`,
`demo_g_sglang`. Copy (don't move) any of these in, and update their Colab/Molab
badge paths to `workshop/odsc-2026/demos/...`.

---

## Setup

TBD. See [`../aiewf-2026/README.md`](../aiewf-2026/README.md) for the Molab +
Lightning.ai setup used previously.
