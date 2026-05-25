"""Timing harness for GPU benchmarks with proper CUDA synchronization."""
from dataclasses import dataclass
import torch
import time


@dataclass
class BenchmarkResult:
    name: str
    mode: str  # "prefill" or "decode"
    latency_ms: float
    tokens_per_sec: float
    gflops: float
    arithmetic_intensity: float  # FLOP/byte
    kv_cache_gib: float
    batch_size: int
    context_len: int


def time_cuda(fn, warmup: int = 3, iterations: int = 10) -> float:
    """Time a CUDA function with proper synchronization. Returns ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iterations * 1000
    return elapsed


def benchmark_attention(
    module: torch.nn.Module,
    batch_size: int,
    context_len: int,
    embed_dim: int,
    num_kv_heads: int,
    head_dim: int,
    num_layers: int = 24,
    dtype=torch.float16,
    warmup: int = 3,
    iterations: int = 10,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """Benchmark an attention module in both prefill and decode modes."""
    device = "cuda"

    # Prefill: full context
    x_prefill = torch.randn(batch_size, context_len, embed_dim, dtype=dtype, device=device)
    prefill_ms = time_cuda(lambda: module(x_prefill), warmup, iterations)
    prefill_tps = batch_size * context_len / (prefill_ms / 1000)

    # Decode: single token with KV cache
    x_decode = torch.randn(batch_size, 1, embed_dim, dtype=dtype, device=device)
    # Pre-fill cache
    with torch.no_grad():
        module(x_prefill)
    decode_ms = time_cuda(lambda: module(x_decode), warmup, iterations)
    decode_tps = batch_size / (decode_ms / 1000)

    # Estimate FLOPs and arithmetic intensity
    bytes_per_elem = 2 if dtype == torch.float16 else 4
    # Prefill: 2 * B * S * D * D (per projection, 4 projections)
    prefill_flops = 2 * batch_size * context_len * embed_dim * embed_dim * 4
    prefill_bytes = embed_dim * embed_dim * 4 * bytes_per_elem  # weight reads
    prefill_ai = prefill_flops / prefill_bytes

    # Decode: same FLOPs per token but reads KV cache too
    decode_flops = 2 * batch_size * 1 * embed_dim * embed_dim * 4
    kv_bytes = 2 * num_kv_heads * head_dim * context_len * batch_size * bytes_per_elem
    decode_bytes = prefill_bytes + kv_bytes
    decode_ai = decode_flops / decode_bytes

    # KV cache estimate for full model
    kv_gib = 2 * num_layers * num_kv_heads * head_dim * context_len * batch_size * bytes_per_elem / (1024**3)

    prefill_result = BenchmarkResult(
        name=module.__class__.__name__, mode="prefill",
        latency_ms=prefill_ms, tokens_per_sec=prefill_tps,
        gflops=prefill_flops / (prefill_ms / 1000) / 1e9,
        arithmetic_intensity=prefill_ai, kv_cache_gib=kv_gib,
        batch_size=batch_size, context_len=context_len,
    )
    decode_result = BenchmarkResult(
        name=module.__class__.__name__, mode="decode",
        latency_ms=decode_ms, tokens_per_sec=decode_tps,
        gflops=decode_flops / (decode_ms / 1000) / 1e9,
        arithmetic_intensity=decode_ai, kv_cache_gib=kv_gib,
        batch_size=batch_size, context_len=context_len,
    )
    return prefill_result, decode_result
