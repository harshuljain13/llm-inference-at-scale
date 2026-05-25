"""Batch scaling curve plots."""
import matplotlib.pyplot as plt
from .benchmark import BenchmarkResult

ATTENTION_COLORS = {"MHA": "tab:blue", "MQA": "tab:orange", "GQA": "tab:green", "MLA": "tab:red"}


def plot_batch_scaling(results: dict[str, list[BenchmarkResult]], output: str = None, title: str = None):
    """
    Plot decode throughput vs batch size for each attention type.
    results: {"MHA": [result_b1, result_b2, ...], "GQA": [...], ...}
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for attn_name, attn_results in results.items():
        decode_only = sorted([r for r in attn_results if r.mode == "decode"], key=lambda r: r.batch_size)
        if not decode_only:
            continue
        batches = [r.batch_size for r in decode_only]
        tps = [r.tokens_per_sec for r in decode_only]
        color = ATTENTION_COLORS.get(attn_name, "tab:purple")
        ax.plot(batches, tps, marker="o", linewidth=2.2, color=color, label=attn_name)

        # Mark best point
        best_idx = max(range(len(tps)), key=tps.__getitem__)
        ax.scatter([batches[best_idx]], [tps[best_idx]], s=100, color=color, edgecolor="black", zorder=5)

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Decode Tokens/sec", fontsize=12)
    ax.set_title(title or "Decode Throughput vs Batch Size", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
    return fig, ax
