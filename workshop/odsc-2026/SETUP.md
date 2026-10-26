# Setup — ODSC 2026

**Do this before the session starts.** The two halves need different things, and
only the second takes real time.

| Half | Runs on | Setup |
|---|---|---|
| **Modules 1 and 2** — model shape, KV cache, attention, quantization | Molab / Colab, or any CUDA GPU | §1 below, two minutes |
| **Modules 3 and 4** — [Dodona](https://github.com/harshuljain13/Dodona): a real cluster, admission control, a voice agent | A GPU box you ssh to | [**PREREQUISITES.md**](https://github.com/harshuljain13/Dodona/blob/main/PREREQUISITES.md), about twenty minutes |

---

## 1 · For modules 1 and 2 — a HuggingFace account

1. Create an account at [huggingface.co](https://huggingface.co).
2. **Accept the Mistral-7B licence** at
   [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
   — one click, instant. Do it now: without it the download fails halfway
   through, which is five wasted minutes in the middle of Module 1.
3. Settings → Access Tokens → **New token**, type *read*.

Both notebooks detect their hardware. A CUDA GPU with ≥ 18 GB takes the
Mistral-7B path; anything smaller — including CPU-only — falls back to
TinyLlama-1.1B. Same shapes, different absolute numbers.

## 2 · For modules 3 and 4 — everything else

→ [**PREREQUISITES.md**](https://github.com/harshuljain13/Dodona/blob/main/PREREQUISITES.md)

Ten numbered steps, ordered by when you can do them rather than by topic:

- **Now** — a RunPod account and **claim your credits**. It is the one step that
  can block on somebody else, and a code that silently failed looks exactly like
  one that worked until you try to launch.
- **Any evening** — clone the repo, copy `.env`, make an SSH key, and add it to
  RunPod *before* launching anything.
- **On the day** — launch the box (this is what starts billing), then collect
  the Deepgram and Firecrawl keys while it boots.

The HuggingFace token from §1 is reused there, so do §1 first.

Two things from that page worth knowing before you book anything:

- **A standard RunPod *Pod* will not work.** It is a container, and the cluster
  installs k3s as a systemd service. Book an **Instant Cluster / bare metal**.
- **Do not bring the cluster up before the session.** It is the first thing we
  do together, it takes fifteen to twenty minutes, and it bills by the hour.

---

The cluster setup lives in the Dodona repo rather than being copied here: it
describes that code and has to version with it. This page owns only what the
notebooks need; the module-by-module framing lives in
[`demos/odsc_2026_module3_and_4_dodona_cluster.md`](demos/odsc_2026_module3_and_4_dodona_cluster.md).
