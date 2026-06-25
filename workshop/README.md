# LLM Inference at Scale: Workshop

**AI Engineering World's Fair 2026 | June 29 | 2 hours**

By [Harshul Jain](https://github.com/harshuljain13) | [Book repo](https://github.com/harshuljain13/llm-inference-at-scale)

---

## Quick Start (Do This Before the Workshop)

### 1. Clone the repo
```bash
git clone https://github.com/harshuljain13/llm-inference-at-scale.git
cd llm-inference-at-scale/workshop/demos
```

### 2. Open notebooks on Molab (GPU, free)
Go to [molab.marimo.io](https://molab.marimo.io) and fork any notebook:
- Click "Open from GitHub"
- Paste: `https://github.com/harshuljain13/llm-inference-at-scale/blob/master/workshop/demos/demo_a_feel_the_pain.ipynb`
- Select GPU runtime (RTX Pro 6000, 96GB)

Demos A through E run on Molab. No install needed.

### 3. (Optional) Lightning.ai for engine demos
Demos F and G (vLLM, SGLang) require a Lightning.ai Studio:
1. Sign up at [lightning.ai](https://lightning.ai)
2. Create a Studio with **A100 GPU**
3. Install engines:
```bash
pip install vllm openai tqdm matplotlib
pip install "numpy<2" "scipy>=1.14"
```
4. Download model:
```bash
hf download mistralai/Mistral-7B-v0.1
```

---

## Workshop Structure

| Part | Time | Demos | Platform |
|------|------|-------|----------|
| 0: Problem Statement | 15 min | A (feel the pain) | Molab |
| 1: Foundations | 25 min | B, C, D | Molab |
| 2: Model Optimizations | 20 min | E (attention + quantization) | Molab |
| 3: KV Cache + Engines | 45 min | F (vLLM), G (SGLang) | Lightning.ai |
| Closing | 10 min | 20x waterfall | Slides |

---

## Demo Notebooks

| Demo | What it proves | Run on |
|------|---------------|--------|
| `demo_a_feel_the_pain.ipynb` | Memory, TTFT, queue, KV growth | Molab |
| `demo_b_memory_equation.ipynb` | Architecture, weights, KV derivation | Molab |
| `demo_c_prefill_vs_decode.ipynb` | Prefill fast, decode slow, roofline | Molab |
| `demo_d_capacity_calculator.ipynb` | Max users, GPU selection (widgets) | Molab |
| `demo_e_attention_comparison.ipynb` | GQA/MLA comparison + INT4/INT8 quantization | Molab |
| `demo_e_quantization_batching.ipynb` | GQA/MLA comparison + INT4/INT8 quantization | Molab |
| `demo_f_vllm.ipynb` | vLLM: PagedAttention, prefix, KV quant, speculative | Lightning.ai |
| `demo_g_sglang.ipynb` | SGLang RadixAttention vs vLLM hash-based prefix | Lightning.ai |

---

## Requirements

**For Demos A-E (Molab):**
- Browser only. No local install needed.
- Molab provides free GPU (RTX Pro 6000, 96GB)

**For Demos F-G (Lightning.ai):**
- Lightning.ai account with GPU access (A100 recommended)
- Python 3.10+
- vLLM, openai SDK, matplotlib, tqdm

---

## Slides

`SLIDES.md` contains the full slide deck with mermaid diagrams, speaker notes, and narrative transitions. Feed to Gamma.app or Google Slides for presentation.

---

## After the Workshop

The full book (59 modules, 12 chapters) is in the `content/` folder:
```
content/
├── 00_transformer_at_inference_time/  (4 modules)
├── 01_gpu_hardware/                   (3 modules)
├── 02_sizing_and_serving/             (1 module)
├── 03_attention_variants/             (7 modules)
├── 04_kv_cache_engineering/           (5 modules)
├── 05_optimization/                   (6 modules)
├── 06_engines/                        (5 modules)
├── 07_scaling/                        (3 modules)
├── 08_serving/                        (7 modules)
├── 09_operations/                     (6 modules)
├── 10_production_stories/             (3 modules)
└── 11_system_designs/                 (5 modules)
```

Each module has a `.md` explainer + `lab.ipynb` you can run.
