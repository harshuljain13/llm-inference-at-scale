# Workshop: LLM Inference at Scale

**AI Engineering World's Fair 2026 | June 29 | 2 hours**

## Structure

```
workshop/
├── README.md           ← You are here
├── SLIDES.md           ← 39 slides with visuals + speaker notes
└── demos/              ← 8 demo notebooks (real experiments on Molab GPU)
    ├── demo_a_feel_the_pain.ipynb         ← Load model, show the problem
    ├── demo_b_memory_equation.ipynb       ← Derive 131 KB/token
    ├── demo_c_capacity_calculator.ipynb   ← Interactive GPU selection
    ├── demo_d_attention_comparison.ipynb  ← MHA vs GQA vs MLA memory
    ├── demo_e_prefix_caching.ipynb        ← Cold vs warm TTFT (real timing)
    ├── demo_f_smart_eviction.ipynb        ← Attention power law + H2O
    ├── demo_g_engine_comparison.ipynb     ← HF vs vLLM vs SGLang benchmark
    └── demo_h_quantization_batching.ipynb ← FP16 vs INT4 + batching waste
```

## Arc (110 min + 10 min Q&A)

| Section | Time | Demos |
|---------|------|-------|
| Opening: Surface the Problem | 15 min | Demo A |
| Part 1: Foundations (Ch00-02) | 20 min | Demo B + C |
| Part 2: Attention (Ch03) | 12 min | Demo D |
| Part 3: KV Cache Engineering (Ch04) | 20 min | Demo E + F |
| Part 4: Optimizations (Ch05) | 18 min | Demo H |
| Part 5: Engines (Ch06) | 12 min | Demo G |
| Closing + Q&A | 13 min | -- |

## Demo Philosophy

Every demo is a real experiment, not a simulation:
- Load actual model weights on GPU
- Measure actual memory, TTFT, throughput
- Apply actual optimization (quantization, prefix reuse, etc.)
- Measure again and show the improvement

If vLLM or other engines fail to install on Molab, the notebooks explain
how to run on AWS (SageMaker ml.g5.xlarge) and include space for pasting
pre-recorded results.

## Dependencies

All demos use Mistral-7B-v0.1 (non-gated, no auth token needed).

| Demo | Core Deps | GPU | Notes |
|------|-----------|-----|-------|
| A | transformers, torch | Yes | The "hook" demo |
| B | transformers, matplotlib | No | Config only, math |
| C | matplotlib, ipywidgets | No | Interactive widgets |
| D | matplotlib | No | Pure calculation |
| E | transformers, torch | Yes | Real prefill timing |
| F | transformers, torch | Yes | Real attention weights |
| G | transformers, vLLM (optional) | Yes | Falls back to pre-computed |
| H | transformers, bitsandbytes | Yes | Real INT4/INT8 loading |

## Pre-Workshop Checklist

- [ ] All 8 demos run end-to-end on Molab GPU
- [ ] Mistral-7B weights pre-cached on Molab
- [ ] TinyLlama-1.1B pre-cached (speculative decoding)
- [ ] Slides generated from SLIDES.md
- [ ] AWS fallback recordings for Demo G + H speculative section
- [ ] Backup screenshots for GPU-dependent cells
- [ ] Attendee Molab access instructions confirmed
