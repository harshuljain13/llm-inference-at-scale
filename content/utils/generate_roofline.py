#!/usr/bin/env python3
"""Generate research-quality roofline model visualization for LLM inference."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set up research-oriented style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

def plot_roofline():
    """Generate roofline model plot for A100 GPU with LLM workloads."""
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # A100 specifications
    peak_compute = 312  # TFLOPS (FP16 Tensor Core)
    memory_bandwidth = 2.0  # TB/s
    ridge_point = peak_compute / memory_bandwidth  # 156 FLOPs/byte
    
    # Arithmetic intensity range (log scale)
    ai = np.logspace(-1, 4, 1000)  # 0.1 to 10000 FLOPs/byte
    
    # Roofline: min of memory ceiling and compute ceiling
    memory_ceiling = ai * memory_bandwidth
    compute_ceiling = np.full_like(ai, peak_compute)
    roofline = np.minimum(memory_ceiling, compute_ceiling)
    
    # Plot the roofline
    ax.loglog(ai, roofline, 'b-', linewidth=2.5, label='Roofline (A100 80GB)', zorder=5)
    
    # Fill regions
    mask_mem = ai < ridge_point
    mask_comp = ai >= ridge_point
    ax.fill_between(ai[mask_mem], roofline[mask_mem], 0.01, 
                    alpha=0.15, color='red', label='Memory-bound region')
    ax.fill_between(ai[mask_comp], roofline[mask_comp], 0.01, 
                    alpha=0.15, color='green', label='Compute-bound region')

    # Mark the ridge point
    ax.axvline(x=ridge_point, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.annotate('Ridge Point\n(156 FLOPs/byte)', 
                xy=(ridge_point, peak_compute * 0.7),
                xytext=(ridge_point * 2.5, peak_compute * 0.4),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))

    # LLM Workload points - Decode (memory-bound) and Prefill (compute-bound)
    workloads = [
        # Decode workloads - memory bound region (left of ridge)
        ('Decode\n(batch=1)', 1.0, 2.0, 'o', '#d62728', (0.25, 3.5)),
        ('Decode\n(batch=8)', 8.0, 16.0, 's', '#ff7f0e', (0.3, 4.0)),
        ('Decode\n(batch=32)', 32.0, 64.0, '^', '#e377c2', (0.35, 3.0)),
        # Prefill workloads - compute bound region (right of ridge)
        ('Prefill\n(N=256)', 256, 290, 'D', '#2ca02c', (0.4, 0.4)),
        ('Prefill\n(N=1000)', 1000, 305, 'p', '#1f77b4', (1.5, 0.5)),
    ]
    
    for name, ai_val, perf, marker, color, (x_mult, y_mult) in workloads:
        ax.scatter(ai_val, perf, s=180, marker=marker, c=color, 
                   edgecolors='black', linewidths=1.5, zorder=10)
        
        xytext = (ai_val * x_mult, perf * y_mult)
        ax.annotate(name, xy=(ai_val, perf), xytext=xytext,
                   fontsize=9, ha='center', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
    
    # Add ceiling labels
    ax.text(0.15, memory_bandwidth * 0.15 * 0.7, 
            'Memory Bandwidth\nCeiling (2 TB/s)', 
            fontsize=10, rotation=45, ha='left', va='bottom', color='#555555',
            style='italic')
    ax.text(1000, peak_compute * 1.15, 
            'Compute Ceiling (312 TFLOPS)', 
            fontsize=10, ha='center', va='bottom', color='#555555',
            style='italic')
    
    # Formatting
    ax.set_xlabel('Arithmetic Intensity (FLOPs/byte)', fontweight='bold')
    ax.set_ylabel('Attainable Performance (TFLOPS)', fontweight='bold')
    ax.set_title('Roofline Model: LLM Inference on NVIDIA A100 80GB', 
                 fontweight='bold', pad=15)
    
    ax.set_xlim(0.1, 10000)
    ax.set_ylim(0.1, 500)
    
    # Legend
    ax.legend(loc='lower right', framealpha=0.95, edgecolor='gray')
    
    # GPU specs box
    specs_text = ("NVIDIA A100 80GB SXM\n"
                  "Peak FP16: 312 TFLOPS\n"
                  "Memory BW: 2.0 TB/s\n"
                  "Ridge Point: 156 FLOPs/byte")
    ax.text(0.02, 0.98, specs_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', 
                     edgecolor='gray', alpha=0.9))
    
    # Insight box
    insight_text = ("Key Insight:\n"
                    "- Decode (batch=1): <1% GPU utilization\n"
                    "- Batching improves decode efficiency\n"
                    "- Prefill achieves near-peak compute")
    ax.text(0.98, 0.02, insight_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcyan', 
                     edgecolor='gray', alpha=0.9))
    
    plt.tight_layout()
    
    # Save
    output_path = Path(__file__).parent / 'images' / 'roofline_model_a100.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")
    
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, facecolor='white', edgecolor='none')
    print(f"Saved: {pdf_path}")
    
    plt.close()

if __name__ == '__main__':
    plot_roofline()
