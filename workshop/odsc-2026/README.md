# Serving LLMs Efficiently at Scale: Workshop — ODSC 2026

By [Harshul Jain](https://github.com/harshuljain13) |
[Slides](https://drive.google.com/drive/folders/1j7GYjcLetHQjrRp9dhnS5wiLOT3dtz7A?usp=sharing)

**Status:** in preparation. Duration and final running order TBD.

Four modules, one argument: **find the wall, lower it, build against it, then
decide who gets turned away when you hit it anyway.** Every number is measured on
the machine running the notebook — nothing here is simulated.

---

## Setup

```bash
git clone https://github.com/harshuljain13/llm-inference-at-scale.git
cd llm-inference-at-scale/workshop/odsc-2026
```

**Start at [`SETUP.md`](SETUP.md)** — the only link the setup slide needs. Modules
1–2 need a HuggingFace account and nothing else. For the cluster half it hands off
to Dodona's
**[`PREREQUISITES.md`](https://github.com/harshuljain13/Dodona/blob/main/PREREQUISITES.md)**,
which is the single source of truth for the GPU box, the keys and the `.env`.

## Modules

| Module | The question it answers |
|--------|-------------------------|
| 1 · [`odsc_2026_module1_memory_and_latency.ipynb`](demos/odsc_2026_module1_memory_and_latency.ipynb) | Why is serving an LLM hard? Model shape → memory → TTFT → ITL → prompt length → **how the KV cache grows** → how many users fit |
| 2 · [`odsc_2026_module2_attention_and_quantization.ipynb`](demos/odsc_2026_module2_attention_and_quantization.ipynb) | Given that floor, how do you lower it? Shrink the KV cache (GQA/MQA/MLA) and shrink the weights (INT8/INT4) — and how many users each buys |
| 3 & 4 · [`odsc_2026_module3_and_4_dodona_cluster.md`](demos/odsc_2026_module3_and_4_dodona_cluster.md) | **Markdown, not a notebook.** Running it for real: prefill/decode split across one sliced card (3), then **admission control** under load (4) |

**1 establishes the bottleneck, 2 attacks it, 3 builds it for real, 4 decides who
gets turned away.** Each module's opening picks up the previous one's closing
number deliberately.

Modules 3–4 are markdown because they need a real cluster, which lives in a
separate public repo — **[Dodona](https://github.com/harshuljain13/Dodona)**. The
module file links out rather than restating, and documents the no-GPU path for
presenting those modules cold.

Every notebook keeps the `odsc_2026_` prefix so a copy pulled into Colab, Molab or
a downloads folder is still identifiable as ODSC material.

## Where things live

One job per file. If something here already belongs to another file, link to it
instead of restating it — duplicated prose drifts, and setup prose drifts fastest.

| File | Owns |
|---|---|
| `README.md` (this) | The running order and the arc. Nothing else |
| [`SETUP.md`](SETUP.md) | Everything before the session: accounts, keys, hardware, the GPU box |
| [`DECISIONS.md`](DECISIONS.md) | The design log — why each module is shaped this way, what was cut, what is still open |
| Each notebook | Its own section index and every number it measures. They open standalone in Colab/Molab, so they never depend on this file |
| [`demos/…module3_and_4…md`](demos/odsc_2026_module3_and_4_dodona_cluster.md) | Modules 3–4, and the rules for referring to Dodona |
| [Dodona](https://github.com/harshuljain13/Dodona) | The cluster itself: prerequisites, bring-up, benchmarking |

The previous Molab + Lightning.ai setup is in
[`../aiewf-2026/README.md`](../aiewf-2026/README.md).
