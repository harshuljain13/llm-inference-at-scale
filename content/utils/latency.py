"""Latency distribution and TTFT/TBT plotting."""
import matplotlib.pyplot as plt
import numpy as np


def plot_latency_distribution(latencies_ms: list[float], title: str = "Latency Distribution", output: str = None):
    """Histogram + percentile lines for latency measurements."""
    arr = np.array(latencies_ms)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(arr, bins=40, color="tab:blue", alpha=0.7, edgecolor="white")
    ax.axvline(p50, color="green", linestyle="-", linewidth=2, label=f"P50: {p50:.1f}ms")
    ax.axvline(p95, color="orange", linestyle="--", linewidth=2, label=f"P95: {p95:.1f}ms")
    ax.axvline(p99, color="red", linestyle=":", linewidth=2, label=f"P99: {p99:.1f}ms")
    ax.set_xlabel("Latency (ms)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_ttft_tbt(ttft_ms: list[float], tbt_ms: list[float], title: str = "TTFT vs TBT", output: str = None):
    """Side-by-side TTFT and TBT distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, name, color in [(ax1, ttft_ms, "TTFT", "tab:blue"), (ax2, tbt_ms, "TBT", "tab:orange")]:
        arr = np.array(data)
        p50, p95 = np.percentile(arr, [50, 95])
        ax.hist(arr, bins=30, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(p50, color="green", linewidth=2, label=f"P50: {p50:.1f}ms")
        ax.axvline(p95, color="red", linewidth=2, linestyle="--", label=f"P95: {p95:.1f}ms")
        ax.set_xlabel(f"{name} (ms)")
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
    return fig
