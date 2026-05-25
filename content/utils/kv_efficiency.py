"""KV cache efficiency metrics and plots."""
import matplotlib.pyplot as plt
from .benchmark import BenchmarkResult


def kv_cache_size_gib(num_layers: int, kv_heads: int, head_dim: int, context: int, batch: int, bytes_per_elem: int = 2) -> float:
    """Calculate KV cache size in GiB for a full model."""
    return 2 * num_layers * kv_heads * head_dim * context * batch * bytes_per_elem / (1024**3)


def kv_efficiency_score(tokens_per_sec: float, kv_gib: float) -> float:
    """KV Efficiency = decode tok/s per GiB of KV cache."""
    return tokens_per_sec / kv_gib if kv_gib > 0 else 0


def plot_kv_efficiency(results: list[BenchmarkResult], output: str = None):
    """Bar chart of KV efficiency scores across attention mechanisms."""
    decode_results = [r for r in results if r.mode == "decode"]
    decode_results.sort(key=lambda r: kv_efficiency_score(r.tokens_per_sec, r.kv_cache_gib))

    names = [r.name for r in decode_results]
    scores = [kv_efficiency_score(r.tokens_per_sec, r.kv_cache_gib) for r in decode_results]
    colors = ["tab:blue", "tab:green", "tab:red", "tab:orange"][:len(names)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, scores, color=colors, edgecolor="black", linewidth=0.8)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + max(scores) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{score:,.0f}", va="center", fontsize=10, fontweight="semibold")

    ax.set_xlabel("KV Efficiency (decode tok/s per GiB)", fontsize=12)
    ax.set_title("KV Cache Efficiency Score", fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
    return fig, ax
