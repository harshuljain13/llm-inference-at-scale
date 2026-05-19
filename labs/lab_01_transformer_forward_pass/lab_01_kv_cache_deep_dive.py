"""
Lab 1: Transformer Forward Pass - Understanding KV Cache at the Byte Level

Duration: 45-60 minutes

Objective: Build intuition for transformer inference by implementing attention 
with KV cache from scratch, then measuring exactly where time and memory go.

What You'll Learn:
1. How KV cache actually works (not just "it caches K and V")
2. Why GQA reduces memory but not compute
3. The exact memory/compute tradeoff in prefill vs decode
4. How to predict whether your workload is memory-bound or compute-bound

Run this script: python lab_01_kv_cache_deep_dive.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import time
import math
from dataclasses import dataclass

# ============================================================================
# PART 1: Setup and Device Detection
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Memory: {gpu_mem:.1f} GB")
print()

# ============================================================================
# PART 2: Attention Implementation with Explicit KV Cache
# ============================================================================

class AttentionWithKVCache(nn.Module):
    """
    Multi-head attention with explicit KV cache.
    
    This implementation shows exactly what happens at each step,
    rather than hiding details in optimized kernels.
    
    Key insight: The KV cache stores the PROJECTED K and V tensors,
    not the original hidden states. You cannot reconstruct hidden states
    from the KV cache.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,  # For GQA: num_kv_heads < num_heads
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.num_kv_groups = num_heads // num_kv_heads

        # Projection matrices
        # Note the size difference between Q and K/V for GQA!
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)
        
        # Stats tracking
        self.stats: Dict[str, float] = {}
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Handle KV cache
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        new_cache = (k, v) if return_cache else None
        
        # Track stats
        self.stats['kv_cache_tokens'] = k.shape[2]
        self.stats['kv_cache_bytes'] = k.numel() * k.element_size() * 2
        
        # Expand K, V for GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Compute attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Causal mask
        q_len, kv_len = q.shape[2], k.shape[2]
        causal_mask = torch.triu(
            torch.ones(q_len, kv_len, device=q.device, dtype=torch.bool),
            diagonal=kv_len - q_len + 1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)
        
        return output, new_cache


# ============================================================================
# PART 3: KV Cache Growth Demonstration
# ============================================================================

def demonstrate_kv_cache_growth():
    """Show exactly how KV cache grows during generation."""
    print("=" * 70)
    print("EXPERIMENT 1: KV Cache Growth During Generation")
    print("=" * 70)
    
    hidden_dim = 512
    num_heads = 8
    num_kv_heads = 2  # GQA: 4 Q heads per KV head
    
    attn = AttentionWithKVCache(hidden_dim, num_heads, num_kv_heads).to(device)
    
    print(f"Config: hidden={hidden_dim}, heads={num_heads}, kv_heads={num_kv_heads}")
    print(f"Head dim: {hidden_dim // num_heads}")
    print(f"GQA group size: {num_heads // num_kv_heads}")
    print()
    
    # PREFILL
    prompt_len = 10
    prompt = torch.randn(1, prompt_len, hidden_dim, device=device)
    output, kv_cache = attn(prompt, kv_cache=None)
    k_cache, v_cache = kv_cache
    
    print(f"PREFILL ({prompt_len} tokens):")
    print(f"  K cache shape: {list(k_cache.shape)}")
    print(f"  KV cache size: {attn.stats['kv_cache_bytes'] / 1024:.2f} KB")
    print()
    
    # DECODE
    print("DECODE (one token at a time):")
    for step in range(5):
        new_token = torch.randn(1, 1, hidden_dim, device=device)
        output, kv_cache = attn(new_token, kv_cache=kv_cache)
        k_cache, _ = kv_cache
        print(f"  Step {step + 1}: shape {list(k_cache.shape)}, "
              f"size = {attn.stats['kv_cache_bytes'] / 1024:.2f} KB")
    
    bytes_per_token = num_kv_heads * (hidden_dim // num_heads) * 2 * 4
    print(f"\nGrowth rate: {bytes_per_token} bytes/token (FP32)")
    print()


# ============================================================================
# PART 4: MHA vs GQA Memory Comparison
# ============================================================================

def compare_mha_vs_gqa():
    """
    Compare memory usage between MHA and GQA.
    
    Key insight: GQA reduces KV cache size but NOT the Q and O projection
    weight sizes. The memory savings are purely in the KV cache.
    """
    print("=" * 70)
    print("EXPERIMENT 2: MHA vs GQA Memory Comparison")
    print("=" * 70)
    
    hidden_dim = 4096  # Llama-scale
    num_heads = 32
    seq_len = 2048
    batch_size = 1
    
    configs = [
        ("MHA (32 KV heads)", 32),
        ("GQA-4 (8 KV heads)", 8),
        ("GQA-8 (4 KV heads)", 4),
        ("MQA (1 KV head)", 1),
    ]
    
    print(f"Config: hidden={hidden_dim}, heads={num_heads}, seq={seq_len}")
    print()
    print(f"{'Variant':<25} {'KV Cache':<15} {'Reduction':<12} {'Weight Size':<15}")
    print("-" * 70)
    
    baseline_kv = None
    for name, num_kv_heads in configs:
        attn = AttentionWithKVCache(hidden_dim, num_heads, num_kv_heads)
        
        # Calculate KV cache size
        head_dim = hidden_dim // num_heads
        kv_cache_bytes = 2 * num_kv_heads * seq_len * head_dim * 4  # FP32
        kv_cache_mb = kv_cache_bytes / (1024 * 1024)
        
        if baseline_kv is None:
            baseline_kv = kv_cache_bytes
            reduction = "1.0x"
        else:
            reduction = f"{baseline_kv / kv_cache_bytes:.1f}x"
        
        # Calculate weight sizes
        q_params = hidden_dim * num_heads * head_dim
        k_params = hidden_dim * num_kv_heads * head_dim
        v_params = hidden_dim * num_kv_heads * head_dim
        o_params = num_heads * head_dim * hidden_dim
        total_params = q_params + k_params + v_params + o_params
        weight_mb = total_params * 4 / (1024 * 1024)
        
        print(f"{name:<25} {kv_cache_mb:>10.1f} MB   {reduction:<12} {weight_mb:>10.1f} MB")
    
    print()
    print("Key insight: GQA reduces KV cache dramatically but weight size")
    print("only decreases slightly (K and V projections are smaller, but")
    print("Q and O projections stay the same size).")
    print()


# ============================================================================
# PART 5: Prefill vs Decode Timing
# ============================================================================

def measure_prefill_vs_decode():
    """
    Measure the time difference between prefill and decode.
    
    Key insight: Prefill processes many tokens in parallel (compute-bound).
    Decode processes one token at a time (memory-bound).
    """
    print("=" * 70)
    print("EXPERIMENT 3: Prefill vs Decode Timing")
    print("=" * 70)
    
    if device.type != "cuda":
        print("This experiment requires a GPU for accurate timing.")
        print("Skipping...")
        print()
        return
    
    hidden_dim = 2048
    num_heads = 16
    num_kv_heads = 4
    num_layers = 8  # Simulate multiple layers
    
    # Create multiple attention layers
    layers = nn.ModuleList([
        AttentionWithKVCache(hidden_dim, num_heads, num_kv_heads)
        for _ in range(num_layers)
    ]).to(device)
    
    prompt_lengths = [128, 512, 1024, 2048]
    decode_steps = 50
    
    print(f"Config: hidden={hidden_dim}, heads={num_heads}, layers={num_layers}")
    print()
    print(f"{'Prompt Len':<12} {'Prefill (ms)':<15} {'Decode/tok (ms)':<18} {'Ratio':<10}")
    print("-" * 60)
    
    for prompt_len in prompt_lengths:
        # Warmup
        x = torch.randn(1, prompt_len, hidden_dim, device=device)
        caches = [None] * num_layers
        for i, layer in enumerate(layers):
            x, caches[i] = layer(x, caches[i])
        torch.cuda.synchronize()
        
        # Measure prefill
        x = torch.randn(1, prompt_len, hidden_dim, device=device)
        caches = [None] * num_layers
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i, layer in enumerate(layers):
            x, caches[i] = layer(x, caches[i])
        torch.cuda.synchronize()
        prefill_time = (time.perf_counter() - start) * 1000
        
        # Measure decode
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(decode_steps):
            x = torch.randn(1, 1, hidden_dim, device=device)
            for i, layer in enumerate(layers):
                x, caches[i] = layer(x, caches[i])
        torch.cuda.synchronize()
        decode_time = (time.perf_counter() - start) * 1000 / decode_steps
        
        ratio = prefill_time / (prompt_len * decode_time)
        print(f"{prompt_len:<12} {prefill_time:<15.2f} {decode_time:<18.3f} {ratio:<10.2f}")
    
    print()
    print("Key insight: Prefill time scales sub-linearly with prompt length")
    print("(parallel processing), while decode time is roughly constant per token.")
    print("The ratio shows how much more efficient prefill is per token.")
    print()


# ============================================================================
# PART 6: Memory Bandwidth Calculation
# ============================================================================

def calculate_memory_bandwidth_limit():
    """
    Calculate the theoretical maximum decode speed based on memory bandwidth.
    
    Key insight: Decode speed is limited by how fast you can read model weights.
    This is the "memory bandwidth wall" - no optimization can exceed this.
    """
    print("=" * 70)
    print("EXPERIMENT 4: Memory Bandwidth Wall Calculation")
    print("=" * 70)
    
    # Model configurations
    models = [
        ("Llama 3.1 8B", 8e9, 32, 8, 128),
        ("Llama 3.1 70B", 70e9, 80, 8, 128),
        ("Mistral 7B", 7e9, 32, 8, 128),
    ]
    
    # GPU configurations (memory bandwidth in GB/s)
    gpus = [
        ("A10G (g5)", 600),
        ("A100 80GB", 2000),
        ("H100 80GB", 3350),
    ]
    
    print("Theoretical maximum decode speed (tokens/second):")
    print()
    print(f"{'Model':<20}", end="")
    for gpu_name, _ in gpus:
        print(f"{gpu_name:<15}", end="")
    print()
    print("-" * 65)
    
    for model_name, params, layers, kv_heads, head_dim in models:
        model_size_gb = params * 2 / 1e9  # FP16
        print(f"{model_name:<20}", end="")
        
        for gpu_name, bandwidth in gpus:
            # Time to read model weights
            read_time_s = model_size_gb / bandwidth
            max_tokens_per_sec = 1 / read_time_s
            print(f"{max_tokens_per_sec:<15.0f}", end="")
        print()
    
    print()
    print("Formula: max_tokens/sec = memory_bandwidth / model_size")
    print()
    print("Key insight: This is a HARD CEILING. Real systems achieve 70-90%")
    print("of this due to KV cache reads, kernel overhead, and sync costs.")
    print()
    
    # Show the impact of quantization
    print("Impact of quantization on Llama 8B + A100:")
    print()
    model_params = 8e9
    bandwidth = 2000  # GB/s
    
    precisions = [
        ("FP16", 2),
        ("INT8", 1),
        ("INT4", 0.5),
    ]
    
    for prec_name, bytes_per_param in precisions:
        model_size_gb = model_params * bytes_per_param / 1e9
        max_tps = bandwidth / model_size_gb
        print(f"  {prec_name}: {model_size_gb:.1f} GB → {max_tps:.0f} tokens/sec theoretical max")
    print()


# ============================================================================
# PART 7: KV Cache Memory Budget Calculator
# ============================================================================

@dataclass
class ModelConfig:
    name: str
    params_b: float
    layers: int
    kv_heads: int
    head_dim: int


def calculate_kv_cache_budget():
    """
    Calculate how many concurrent sequences you can serve given VRAM constraints.
    
    Key insight: KV cache often consumes MORE memory than model weights
    at high batch sizes or long sequences.
    """
    print("=" * 70)
    print("EXPERIMENT 5: KV Cache Memory Budget")
    print("=" * 70)
    
    # Model configs
    llama_8b = ModelConfig("Llama 8B", 8, 32, 8, 128)
    llama_70b = ModelConfig("Llama 70B", 70, 80, 8, 128)
    
    # GPU VRAM options
    vram_options = [24, 40, 80]  # GB
    
    # Sequence length
    seq_len = 4096
    
    for model in [llama_8b, llama_70b]:
        print(f"\n{model.name} (FP16 weights, FP16 KV cache):")
        print(f"  Sequence length: {seq_len} tokens")
        print()
        
        # Model weights
        model_weights_gb = model.params_b * 2  # FP16
        
        # KV cache per sequence
        kv_per_seq_bytes = 2 * model.layers * model.kv_heads * model.head_dim * seq_len * 2
        kv_per_seq_gb = kv_per_seq_bytes / 1e9
        
        print(f"  Model weights: {model_weights_gb:.1f} GB")
        print(f"  KV cache per sequence: {kv_per_seq_gb:.2f} GB")
        print()
        
        print(f"  {'VRAM':<10} {'Available':<12} {'Max Batch':<12} {'Total KV':<12}")
        print("  " + "-" * 50)
        
        for vram in vram_options:
            available = vram - model_weights_gb - 2  # 2 GB overhead
            if available <= 0:
                print(f"  {vram} GB     Model doesn't fit!")
                continue
            
            max_batch = int(available / kv_per_seq_gb)
            total_kv = max_batch * kv_per_seq_gb
            
            print(f"  {vram} GB     {available:.1f} GB       {max_batch:<12} {total_kv:.1f} GB")
    
    print()
    print("Key insight: At batch=32 with 4K context, KV cache for Llama 8B")
    print("is 16 GB - the same as the model weights! This is why KV cache")
    print("management (PagedAttention) is so critical.")
    print()


# ============================================================================
# PART 8: Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LAB 1: TRANSFORMER FORWARD PASS - KV CACHE DEEP DIVE")
    print("=" * 70 + "\n")
    
    demonstrate_kv_cache_growth()
    compare_mha_vs_gqa()
    measure_prefill_vs_decode()
    calculate_memory_bandwidth_limit()
    calculate_kv_cache_budget()
    
    print("=" * 70)
    print("LAB COMPLETE")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("1. KV cache grows linearly with sequence length")
    print("2. GQA reduces KV cache 4-8x with minimal quality loss")
    print("3. Prefill is compute-bound, decode is memory-bound")
    print("4. Memory bandwidth sets a hard ceiling on decode speed")
    print("5. KV cache can exceed model weights at high batch/sequence")
    print()
    print("Next: Module 2 - GPU Memory Engineering")
