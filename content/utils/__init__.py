"""LLM Inference Experiment Utilities — reusable across all notebooks."""
from .gpu_info import detect_gpu, print_gpu_info
from .roofline import plot_roofline, overlay_points
from .latency import plot_latency_distribution, plot_ttft_tbt
from .kv_efficiency import plot_kv_efficiency, kv_cache_size_gib
from .batch_scaling import plot_batch_scaling
from .benchmark import time_cuda, benchmark_attention, BenchmarkResult
