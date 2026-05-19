"""
GPU Monitoring and Visualization Toolkit for LLM Inference Workshop.

This module provides real-time GPU metrics collection and visualization,
including animated displays of memory usage, utilization, and inference metrics.

Usage:
    from utils.gpu_monitor import GPUMonitor, InferenceProfiler
    
    # Real-time GPU monitoring
    with GPUMonitor() as monitor:
        # Your inference code here
        pass
    monitor.plot_timeline()
    
    # Animated visualization
    monitor.animate_memory_usage()
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable
from collections import deque
import json

# Conditional imports for environments without GPU
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class GPUSnapshot:
    """Single point-in-time GPU metrics."""
    timestamp: float
    memory_used_mb: float
    memory_total_mb: float
    gpu_utilization: float  # 0-100
    memory_utilization: float  # 0-100
    temperature: float
    power_draw_w: float
    sm_clock_mhz: int
    memory_clock_mhz: int
    
    @property
    def memory_used_gb(self) -> float:
        return self.memory_used_mb / 1024
    
    @property
    def memory_free_gb(self) -> float:
        return (self.memory_total_mb - self.memory_used_mb) / 1024
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "memory_used_gb": round(self.memory_used_gb, 2),
            "memory_total_gb": round(self.memory_total_mb / 1024, 2),
            "gpu_util_pct": self.gpu_utilization,
            "mem_util_pct": self.memory_utilization,
            "temp_c": self.temperature,
            "power_w": self.power_draw_w,
        }
