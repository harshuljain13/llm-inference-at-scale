# Workshop Sprint: 5 Hours to Done

**Date:** June 25, 2026
**Workshop:** June 29, 2026 (4 days away)
**Goal:** All demos tested, slides generated, backups recorded, pushed to GitHub.

---

## Task 1: Lightning.ai Studio Setup (30 min)

- [x] Go to https://lightning.ai and create a new Studio
- [x] Select GPU: A10G (24 GB) or L4
- [x] Install vLLM: `pip install vllm`
- [x] Download Mistral-7B weights: `hf download mistralai/Mistral-7B-v0.1`
- [x] Test server starts:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-v0.1 \
    --dtype float16 \
    --gpu-memory-utilization 0.90 \
    --port 8000
```
- [x] Verify: `curl http://localhost:8000/v1/models` returns model list
- [x] If vLLM fails: try `pip install vllm==0.6.0` (older stable version)

### Troubleshooting (issues encountered)

**Issue 1: `huggingface-cli` deprecated**
- Lightning.ai uses new `hf` CLI instead of `huggingface-cli`
- Fix: `hf download mistralai/Mistral-7B-v0.1`

**Issue 2: NumPy/SciPy version conflict**
- Error: `ImportError: cannot import name 'Inf' from 'numpy'`
- Cause: NumPy 2.x removed `numpy.Inf`, old SciPy still references it
- Fix: `pip install "numpy<2" "scipy>=1.14"`
- Then retry vLLM server command

---

## Task 2: Test Demo F on Lightning.ai (60 min)

Notebook: `workshop/demos/demo_f_engines_kv_optimizations.ipynb`

### Experiment 1: PagedAttention
- [ ] With server running, run the 50-request benchmark cell
- [ ] Verify: all 50 requests complete without OOM
- [ ] Note throughput number: _____ tok/s

### Experiment 2: Prefix Caching
- [ ] Stop server. Restart with `--enable-prefix-caching`
- [ ] Run cold batch (10 requests): note avg TTFT: _____ ms
- [ ] Run warm batch (10 requests): note avg TTFT: _____ ms
- [ ] Verify speedup is 5x+

### Experiment 3: SGLang (OPTIONAL, skip if short on time)
- [ ] Install: `pip install "sglang[all]"`
- [ ] Start: `python -m sglang.launch_server --model-path mistralai/Mistral-7B-v0.1 --port 8000`
- [ ] Run same prefix benchmark
- [ ] Note: if SGLang install fails, skip entirely. vLLM prefix caching is enough.

### Experiment 4: Speculative Decoding (OPTIONAL)
- [ ] Restart vLLM with `--speculative-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --num-speculative-tokens 5`
- [ ] Run 100-token generation benchmark
- [ ] Note: needs 20GB+ VRAM. If OOM, skip this experiment.

---

## Task 3: Demo E - Model Optimizations (45 min)

Platform: Molab

Current file: `workshop/demos/demo_e_attention_comparison.ipynb`

Needs to include:
- [ ] Part 1: MHA vs GQA vs MLA memory comparison (already there)
- [ ] Part 2: Weight quantization (ADD THIS)
  - Load Mistral-7B FP16, measure memory
  - Load INT8 (bitsandbytes), measure memory
  - Load INT4 (NF4), measure memory
  - Bar chart comparing all three
- [ ] Test on Molab: all cells run without error
- [ ] Check: no variable name collisions between Part 1 and Part 2

---

## Task 4: Verify Demo G Smart Eviction (20 min)

Platform: Molab

File: `workshop/demos/demo_g_smart_eviction.ipynb`

- [ ] Run all cells on Molab
- [ ] Verify: attention heatmap displays correctly
- [ ] Verify: power law distribution chart shows
- [ ] If broken: check if model loads (Mistral-7B, token=False)

---

## Task 5: Record Backup Screenshots (30 min)

For each demo, screenshot the key output in case of live failure:

- [ ] Demo A: TTFT scaling chart + queue Gantt chart
- [ ] Demo B: weight distribution pie chart
- [ ] Demo C: prefill vs decode side-by-side + roofline
- [ ] Demo D: KV growth chart + GPU comparison bars
- [ ] Demo E: attention comparison bars + quantization bars
- [ ] Demo F: prefix caching cold vs warm numbers
- [ ] Demo G: attention heatmap

Save to: `workshop/backup_screenshots/`

---

## Task 6: Push to GitHub (15 min)

```bash
cd /local/home/jharshul/work/llm-inference/llm-inference-at-scale
git add workshop/
git status  # review what's being added
git commit -m "workshop: final demos for AIE World's Fair June 29"
git push origin master
```

- [ ] Verify on GitHub: https://github.com/harshuljain13/llm-inference-at-scale
- [ ] Check: all notebooks render on GitHub
- [ ] Check: SLIDES.md renders with mermaid diagrams

---

## Task 7: Generate Slides (60 min)

- [ ] Open SLIDES.md
- [ ] Copy content into Gamma.app (or Google Slides)
- [ ] Generate slides from the markdown content
- [ ] For mermaid diagrams: render them at https://mermaid.live, screenshot, paste into slides
- [ ] Add your headshot + title slide
- [ ] Export as PDF backup

---

## Cut List (Skip if Running Behind)

| Skip | Impact |
|------|--------|
| SGLang experiment (Task 2.3) | Low: vLLM prefix caching proves the concept |
| Speculative decoding (Task 2.4) | Low: mention in slides, don't demo live |
| Demo G re-test (Task 4) | Low: show attention heatmap from backup screenshot |
| Streamlit dashboard | Zero: never promised to audience |

---

## Key Numbers to Verify (Fill in During Testing)

| Metric | Expected | Actual |
|--------|----------|--------|
| Model memory (FP16) | ~14.5 GB | |
| TTFT at 128 tokens | ~90 ms | |
| TTFT at 16K tokens | ~5000 ms | |
| KV per token | 131 KB | |
| Prefix cache cold TTFT | ~200 ms | |
| Prefix cache warm TTFT | ~12 ms | |
| vLLM throughput (50 users) | ~500+ tok/s | |
| INT4 memory | ~3.6 GB | |

---

## Done Checklist

- [ ] All demos tested on correct platform (Molab or Lightning)
- [ ] Backup screenshots saved
- [ ] Pushed to GitHub
- [ ] Slides generated
- [ ] Lightning.ai Studio URL saved (for workshop day)
- [ ] Speaker notes reviewed (SLIDES.md transitions)
