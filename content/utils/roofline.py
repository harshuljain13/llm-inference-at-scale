"""GPU roofline model plotting with benchmark point overlay."""
import matplotlib.pyplot as plt
import numpy as np
from .gpu_info import GPUInfo
from .benchmark import BenchmarkResult

ATTENTION_COLORS = {"MHA": "tab:blue", "MQA": "tab:orange", "GQA": "tab:green", "MLA": "tab:red"}
MODE_MARKERS = {"prefill": "^", "decode": "o"}


def plot_roofline(gpu: GPUInfo, output: str = None, title: str = None):
    """Plot the GPU roofline model."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ai = np.logspace(-1, 5, 500)
    memory_roof = gpu.bw_gbs * ai  # GFLOP/s = BW * AI
    compute_roof = np.full_like(ai, gpu.tflops_fp16 * 1000)  # GFLOP/s
    achieved = np.minimum(memory_roof, compute_roof)

    ax.loglog(ai, achieved, "k-", linewidth=2.5, label="Roofline")
    ax.axvline(gpu.ridge_point, color="gray", linestyle="--", alpha=0.7, label=f"Ridge: {gpu.ridge_point:.0f} FLOP/byte")
    ax.fill_between(ai, 0, achieved, where=(ai < gpu.ridge_point), alpha=0.08, color="red", label="Memory-bound")
    ax.fill_between(ai, 0, achieved, where=(ai >= gpu.ridge_point), alpha=0.08, color="blue", label="Compute-bound")

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Performance (GFLOP/s)", fontsize=12)
    ax.set_title(title or f"{gpu.name} FP16 Roofline", fontsize=14)
    ax.set_xlim(0.1, 1e5)
    ax.set_ylim(1, gpu.tflops_fp16 * 2000)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
    return fig, ax


def overlay_points(ax, results: list[BenchmarkResult], label_style: str = "short"):
    """Overlay benchmark results on an existing roofline plot."""
    for r in results:
        color = ATTENTION_COLORS.get(r.name.replace("Attention", "").upper(), "tab:purple")
        marker = MODE_MARKERS.get(r.mode, "o")
        ax.scatter(r.arithmetic_intensity, r.gflops, color=color, marker=marker, s=100, zorder=5, edgecolors="white", linewidths=0.8)
        if label_style == "short":
            ax.annotate(f"{r.name}\n{r.mode}", (r.arithmetic_intensity, r.gflops),
                       textcoords="offset points", xytext=(8, 4), fontsize=8, color=color)
    return ax
