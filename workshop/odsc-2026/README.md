# Serving LLMs Efficiently at Scale: Workshop

**ODSC 2026**

By [Harshul Jain](https://github.com/harshuljain13) | [All Artifacts](https://drive.google.com/drive/folders/1j7GYjcLetHQjrRp9dhnS5wiLOT3dtz7A?usp=sharing)

---

## Quick Start (Do This Before the Workshop)

### 1. Clone the repo
```bash
git clone https://github.com/harshuljain13/llm-inference-at-scale.git
cd llm-inference-at-scale/workshop/odsc-2026/demos
```

### 2. Accounts and keys

Open these now, not on the day. Each one has a wait in it somewhere — an email
confirmation, a licence click, a card on file — and all of them are free or
free-tier except the GPU box.

| | What for | Needed by |
|---|---|---|
| [Molab](https://molab.marimo.io/notebooks) | Runs modules 1–2 in the browser, on a free GPU | **1, 2** |
| [HuggingFace → Access Tokens](https://huggingface.co/settings/tokens) | Downloading the model weights | **1, 2** |
| [RunPod](https://console.runpod.io/) | The 2 × A100-80 box the cluster runs on, billed by the hour | **3, 4** |
| [Deepgram](https://console.deepgram.com/) | Speech-to-text for Dodona's voice agent | **3, 4** (voice demo only) |

**HuggingFace, in full:** modules 1 and 2 load Mistral-7B, which is gated.
[Accept the licence](https://huggingface.co/mistralai/Mistral-7B-v0.1) — one click —
then Settings → Access Tokens → **New token**, type *read*. Do it now: without it
the download fails halfway through.

**RunPod** is the only one that costs real money. Two A100-80s run a few dollars
an hour, and the meter does not stop when you close the tab — stop the pod in the
console when you are done. Modules 1 and 2 need no GPU of your own at all.

**Deepgram** is needed only if you want to *talk* to the agent. Modules 3 and 4
work entirely through text and the dashboards without it.

---

## Demo Notebooks with Workshop links

| Module | What it proves | Run on |
|--------|----------------|--------|
| `odsc_2026_module1_memory_and_latency.ipynb` | Weights are fixed, KV cache is per-user; TTFT grows with the prompt, ITL does not; the cache is linear in context **and** in users | [Colab](https://colab.research.google.com/github/harshuljain13/llm-inference-at-scale/blob/master/workshop/odsc-2026/demos/odsc_2026_module1_memory_and_latency.ipynb) · [Molab](https://molab.marimo.io/github/harshuljain13/llm-inference-at-scale/blob/master/workshop/odsc-2026/demos/odsc_2026_module1_memory_and_latency.ipynb) |
| `odsc_2026_module2_attention_and_quantization.ipynb` | GQA/MQA shrink the cache in proportion to KV heads, INT8/INT4 shrink the weights; both stack, and they buy capacity rather than speed | [Colab](https://colab.research.google.com/github/harshuljain13/llm-inference-at-scale/blob/master/workshop/odsc-2026/demos/odsc_2026_module2_attention_and_quantization.ipynb) · [Molab](https://molab.marimo.io/github/harshuljain13/llm-inference-at-scale/blob/master/workshop/odsc-2026/demos/odsc_2026_module2_attention_and_quantization.ipynb) |
| `odsc_2026_module2_vllm_engine.ipynb` | What continuous batching, prefix caching, KV quantization and speculative decoding buy over pure PyTorch | vLLM server (offline) |
| `odsc_2026_module3_and_4_dodona_cluster.md` | Two replicas of a 32B, one per card: routing a request to the less loaded one, then admission control — when both are full, who gets turned away | [Dodona](https://github.com/harshuljain13/Dodona) on a GPU box |

**1 establishes the bottleneck, 2 attacks it with the model and the engine, 3 builds
it for real, 4 decides who gets turned away.**

---

## Requirements

**For modules 1 and 2 (Molab / Colab):**
- Browser only. No local install needed.
- A CUDA GPU with ≥ 18 GB runs Mistral-7B; anything smaller, including CPU-only,
  falls back to TinyLlama-1.1B. Same shapes, different absolute numbers.
- HuggingFace read token (step 2 above).

**For the vLLM engine module:**
- A GPU box with vLLM, and a server already running:
  ```bash
  vllm serve mistralai/Mistral-7B-v0.1 --port 8000
  ```
- The notebook talks to `localhost:8000`. It does not start the server.

**For modules 3 and 4 (Dodona):**
- A GPU box you ssh to, billed by the hour — 2 × A100-80 on RunPod, one vLLM
  replica of the 32B per card.
- Setup is [`PREREQUISITES.md`](https://github.com/harshuljain13/Dodona/blob/main/PREREQUISITES.md)
  in the Dodona repo — the box, the keys, the `.env`. Do it before the session.
- No GPU? Two of the four benchmarks can be followed cold, because their reports
  and Grafana screenshots are committed: `01-health` and `02-baseline`.
  `03-capacity` and `04-routing-admission` ship the notebook but no output, so
  they need a live fleet.

### Open notebooks on Molab (GPU, free)
Go to [molab.marimo.io](https://molab.marimo.io) and fork any notebook:
- Click "Open from GitHub"
- Paste the notebook URL from the table above
- Select a GPU runtime

---

## After the Workshop

The full book is in the [`content/`](../../content) folder at the repo root — a `.md`
explainer plus a runnable `lab.ipynb` per module.
