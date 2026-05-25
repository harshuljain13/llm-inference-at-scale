#!/usr/bin/env python3
"""Build lab.ipynb for 00.2 Transformer Inference Basics."""
import json

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n"), "id": None}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source.split("\n"), "outputs": [], "execution_count": None, "id": None}

cells = []

# Cell 1: Title
cells.append(md("""# Lab 00.2: Transformer Inference Basics — Hands-On Experiments

**Objective:** Measure and visualize the fundamental mechanics of autoregressive LLM inference:
- Autoregressive decoding loop implementation and timing
- Prefill vs decode phase characteristics
- KV cache mechanics and memory growth
- End-to-end latency breakdown

> Run on a GPU instance (A10G/A100/H100) for meaningful results."""))

# Cell 2: Setup
cells.append(code("""import sys
sys.path.insert(0, '../../..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass

from utils import gpu_info, benchmark, latency

# Detect GPU
gpu = gpu_info.detect_gpu()
gpu_info.print_gpu_info(gpu)
print(f"\\nPyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'"""))

# Cell 3: Theory - Model config
cells.append(md("""## 1. Model Configuration

We'll build a minimal transformer decoder matching Llama 3.1 8B dimensions to measure real inference behavior.
Key parameters:
- `hidden_dim=4096`, `num_heads=32`, `num_kv_heads=8` (GQA)
- `intermediate_dim=14336` (SwiGLU MLP)
- `num_layers=4` (subset for memory — scale analysis to 32 layers)"""))

# Cell 4: Config + minimal transformer
cells.append(code("""@dataclass
class ModelConfig:
    hidden_dim: int = 4096
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_dim: int = 14336
    num_layers: int = 4
    vocab_size: int = 32000
    max_seq_len: int = 2048
    dtype: torch.dtype = torch.float16

cfg = ModelConfig()
print(f"Config: {cfg.num_layers} layers, {cfg.hidden_dim}d, GQA {cfg.num_heads}q/{cfg.num_kv_heads}kv")
print(f"Params per layer (attn): {(cfg.hidden_dim * cfg.hidden_dim + 2 * cfg.hidden_dim * cfg.num_kv_heads * cfg.head_dim + cfg.hidden_dim * cfg.hidden_dim) * 2 / 1e6:.1f} MB")
print(f"Params per layer (MLP):  {3 * cfg.hidden_dim * cfg.intermediate_dim * 2 / 1e6:.1f} MB")"""))

# Cell 5: Theory - Autoregressive decoding
cells.append(md("""## 2. Autoregressive Decoding Loop

The core inference loop: each new token depends on ALL previous tokens via the attention mechanism.
Without KV cache, cost is O(n²) in sequence length. With KV cache, each step is O(n) but requires storing past keys/values.

**Key insight:** The decoding loop is inherently sequential — you cannot parallelize token generation for a single sequence."""))

# Cell 6: Minimal attention with KV cache
cells.append(code("""class GQAAttention(nn.Module):
    \"\"\"Grouped-Query Attention with KV cache support.\"\"\"
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.groups = cfg.num_heads // cfg.num_kv_heads

        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.num_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_dim, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_dim, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_dim, bias=False)

    def forward(self, x, kv_cache=None, use_cache=True):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        new_cache = (k, v) if use_cache else None

        # Expand KV for GQA
        k_exp = k.repeat_interleave(self.groups, dim=1)
        v_exp = v.repeat_interleave(self.groups, dim=1)

        scale = self.head_dim ** -0.5
        attn = (q @ k_exp.transpose(-2, -1)) * scale

        # Causal mask
        T = k_exp.shape[2]
        mask = torch.triu(torch.ones(S, T, device=x.device), diagonal=T - S + 1).bool()
        attn.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v_exp).transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), new_cache"""))

# Cell 7: Transformer block + model
cells.append(code("""class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = GQAAttention(cfg)
        self.norm1 = nn.RMSNorm(cfg.hidden_dim)
        self.norm2 = nn.RMSNorm(cfg.hidden_dim)
        self.gate_proj = nn.Linear(cfg.hidden_dim, cfg.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_dim, cfg.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_dim, cfg.hidden_dim, bias=False)

    def forward(self, x, kv_cache=None, use_cache=True):
        h = self.norm1(x)
        attn_out, new_cache = self.attn(h, kv_cache, use_cache)
        x = x + attn_out
        h = self.norm2(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x, new_cache


class MiniLlama(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = nn.RMSNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

    def forward(self, input_ids, kv_caches=None, use_cache=True):
        x = self.embed(input_ids)
        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None
            x, new_cache = layer(x, cache, use_cache)
            new_caches.append(new_cache)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_caches if use_cache else None

model = MiniLlama(cfg).to(device=device, dtype=cfg.dtype)
param_mb = sum(p.numel() * 2 for p in model.parameters()) / 1e6
print(f"Model: {param_mb:.0f} MB ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")"""))

with open('/local/home/jharshul/work/llm-inference-at-scale/content/00_foundations/00.2_transformer_inference_basics/_cells_part1.json', 'w') as f:
    json.dump(cells, f)
print(f"Part 1: {len(cells)} cells written")
