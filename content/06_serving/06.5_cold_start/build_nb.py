#!/usr/bin/env python3
"""Build cold start mitigation notebook."""
import json

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n"), "id": None}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source.split("\n"), "outputs": [], "execution_count": None, "id": None}

cells = []

# Cell 1: Intro
cells.append(md("""# Cold Start Mitigation for LLM Inference
\n## Overview
\nCold start is the latency penalty when a new inference instance must:
1. **Pull model weights** from storage (S3/EFS/local NVMe)
2. **Load into GPU memory** (VRAM allocation + tensor deserialization)
3. **Warm up** (JIT compilation, CUDA graph capture, KV cache pre-allocation)
\nFor large models (70B+ parameters), cold start can exceed **5-10 minutes**, making autoscaling
and spot recovery critical challenges for production serving.
\nThis notebook benchmarks cold start components and builds mitigation strategies:
- Streaming vs bulk loading
- Warm pool sizing under traffic patterns
- Spot recovery with pre-warming
- Cost vs latency Pareto optimization"""))

# Cell 2: Imports
cells.append(code("""import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('.')), '..', '..'))

try:
    from utils.benchmark import Timer
    from utils.latency import plot_latency_distribution
except ImportError:
    pass

np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')
print("Environment ready.")"""))

# Cell 3: Model loading time calculator
cells.append(code("""@dataclass
class ModelConfig:
    name: str
    params_b: float  # billions
    bytes_per_param: float = 2.0  # FP16

    @property
    def size_gb(self) -> float:
        return self.params_b * self.bytes_per_param

MODELS = [
    ModelConfig("7B", 7), ModelConfig("13B", 13),
    ModelConfig("34B", 34), ModelConfig("70B", 70),
    ModelConfig("180B", 180), ModelConfig("405B", 405, 1.0),  # INT8
]

# Storage throughput (GB/s) - realistic values
STORAGE = {
    "S3 single-stream": 0.7,
    "S3 multi-stream (16x)": 8.0,
    "EFS burst": 3.0,
    "Local NVMe": 25.0,
    "NVMe RAID-0 (4x)": 80.0,
    "Streaming (first token)": 0.7,  # only need first layers
}

# Calculate load times
print(f"{'Model':<8} {'Size(GB)':<10}", end="")
for s in STORAGE:
    print(f"{s:<22}", end="")
print()
print("-" * 150)

for m in MODELS:
    print(f"{m.name:<8} {m.size_gb:<10.1f}", end="")
    for s, bw in STORAGE.items():
        if s == "Streaming (first token)":
            # Only need ~2 layers for first token
            t = (m.size_gb / (m.params_b * 0.5) * 2) / bw
        else:
            t = m.size_gb / bw
        print(f"{t:<22.1f}s", end="")
    print()

# Visualization
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
x = [m.name for m in MODELS]
for label, bw in list(STORAGE.items())[:5]:
    times = [m.size_gb / bw for m in MODELS]
    ax.plot(x, times, 'o-', label=label)
ax.set_xlabel("Model Size")
ax.set_ylabel("Load Time (seconds)")
ax.set_title("Model Loading Time by Storage Backend")
ax.legend()
ax.set_yscale('log')
plt.tight_layout()
plt.show()"""))

# Cell 4: Streaming simulation
cells.append(code("""def simulate_streaming_load(model_gb: float, layer_count: int,
                           bandwidth_gbps: float, streams: int = 1) -> dict:
    \"\"\"Simulate layer-by-layer streaming with time-to-first-token tracking.\"\"\"
    layer_size = model_gb / layer_count
    effective_bw = bandwidth_gbps * min(streams, 8)  # diminishing returns past 8

    layer_times = []
    cumulative = 0.0
    for i in range(layer_count):
        # Add jitter for network variance
        jitter = np.random.exponential(0.05)
        t = (layer_size / effective_bw) + jitter
        cumulative += t
        layer_times.append(cumulative)

    return {
        "total_time": cumulative,
        "time_to_first_layer": layer_times[0],
        "time_to_half": layer_times[layer_count // 2],
        "layer_times": layer_times,
    }

# Compare streaming strategies for 70B model
configs = [
    ("1 stream", 1), ("4 streams", 4),
    ("8 streams", 8), ("16 streams", 16),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for label, streams in configs:
    result = simulate_streaming_load(140.0, 80, 0.7, streams)
    axes[0].plot(result["layer_times"], label=f"{label} (total={result['total_time']:.1f}s)")

axes[0].set_xlabel("Layer Index")
axes[0].set_ylabel("Cumulative Time (s)")
axes[0].set_title("70B FP16: Layer Loading Progress")
axes[0].axhline(y=5.0, color='r', linestyle='--', alpha=0.5, label='5s SLA')
axes[0].legend()

# Time to first inference capability
ttfi = []
for streams in range(1, 17):
    r = simulate_streaming_load(140.0, 80, 0.7, streams)
    ttfi.append(r["time_to_first_layer"])
axes[1].bar(range(1, 17), ttfi)
axes[1].set_xlabel("Number of Streams")
axes[1].set_ylabel("Time to First Layer (s)")
axes[1].set_title("Parallelism vs Time-to-First-Layer")
plt.tight_layout()
plt.show()"""))

# Cell 5: Autoscaling cold start impact
cells.append(code("""def simulate_autoscaling(duration_min: int, base_rps: float, spike_mult: float,
                        cold_start_s: float, scale_threshold: float = 0.8,
                        max_rps_per_instance: float = 10.0) -> dict:
    \"\"\"Simulate autoscaling with cold start delays under traffic spikes.\"\"\"
    dt = 1.0  # 1-second steps
    steps = int(duration_min * 60 / dt)
    instances = 1
    pending_instances = []  # (ready_at_step, count)

    metrics = {"rps": [], "capacity": [], "dropped": [], "instances": [], "queue": []}
    queue = 0.0

    for step in range(steps):
        t = step * dt / 60.0  # minutes
        # Traffic: base + spike in middle third
        if duration_min * 0.33 < t < duration_min * 0.66:
            rps = base_rps * spike_mult + np.random.normal(0, base_rps * 0.1)
        else:
            rps = base_rps + np.random.normal(0, base_rps * 0.05)
        rps = max(0, rps)

        # Check pending instances
        new_pending = []
        for ready_at, count in pending_instances:
            if step >= ready_at:
                instances += count
            else:
                new_pending.append((ready_at, count))
        pending_instances = new_pending

        capacity = instances * max_rps_per_instance
        utilization = rps / capacity if capacity > 0 else 1.0

        # Scale up decision
        if utilization > scale_threshold:
            needed = int(np.ceil((rps - capacity) / max_rps_per_instance)) + 1
            pending_instances.append((step + int(cold_start_s / dt), needed))

        # Process requests
        queue += rps * dt
        served = min(queue, capacity * dt)
        queue -= served
        dropped = max(0, queue - capacity * 30)  # drop if queue > 30s backlog
        queue -= dropped

        metrics["rps"].append(rps)
        metrics["capacity"].append(capacity)
        metrics["dropped"].append(dropped)
        metrics["instances"].append(instances)
        metrics["queue"].append(queue)

    return metrics

# Compare cold start durations
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
cold_starts = [10, 60, 180, 600]  # seconds

for ax, cs in zip(axes.flat, cold_starts):
    m = simulate_autoscaling(30, 50, 5.0, cs)
    t = np.arange(len(m["rps"])) / 60.0
    ax.plot(t, m["rps"], 'b-', alpha=0.5, label='Traffic')
    ax.plot(t, m["capacity"], 'g-', label='Capacity')
    ax.fill_between(t, 0, m["dropped"], alpha=0.3, color='red', label='Dropped')
    ax.set_title(f"Cold Start = {cs}s")
    ax.set_xlabel("Time (min)")
    ax.legend(loc='upper left')

plt.suptitle("Autoscaling Response Under 5x Traffic Spike", fontsize=14)
plt.tight_layout()
plt.show()

total_dropped = {cs: sum(simulate_autoscaling(30, 50, 5.0, cs)["dropped"]) for cs in cold_starts}
print("\\nTotal dropped requests by cold start duration:")
for cs, d in total_dropped.items():
    print(f"  {cs:>4}s cold start -> {d:,.0f} dropped requests")"""))

with open("/local/home/jharshul/work/llm-inference-at-scale/content/06_serving/06.5_cold_start/lab.ipynb", "w") as f:
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": cells
    }
    # Write partial - we'll append remaining cells
    json.dump(nb, f, indent=1)

print("Part 1 written (cells 1-5)")
