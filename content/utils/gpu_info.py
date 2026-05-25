"""GPU detection and hardware specs."""
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

# Known GPU specs (peak FP16 TFLOPS, memory BW in GB/s)
GPU_CATALOG = {
    "A10G": {"tflops_fp16": 125, "bw_gbs": 600, "vram_gb": 24},
    "A100": {"tflops_fp16": 312, "bw_gbs": 2039, "vram_gb": 80},
    "A100-80GB": {"tflops_fp16": 312, "bw_gbs": 2039, "vram_gb": 80},
    "H100": {"tflops_fp16": 990, "bw_gbs": 3350, "vram_gb": 80},
    "RTX 4090": {"tflops_fp16": 330, "bw_gbs": 1008, "vram_gb": 24},
    "RTX 3090": {"tflops_fp16": 142, "bw_gbs": 936, "vram_gb": 24},
    "L4": {"tflops_fp16": 121, "bw_gbs": 300, "vram_gb": 24},
    "L40S": {"tflops_fp16": 366, "bw_gbs": 864, "vram_gb": 48},
}


@dataclass
class GPUInfo:
    name: str
    vram_gb: float
    tflops_fp16: float
    bw_gbs: float
    ridge_point: float  # FLOP/byte where compute meets memory roof

    def to_dict(self):
        return asdict(self)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


def detect_gpu() -> GPUInfo:
    """Detect current GPU and return specs."""
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. Run on a GPU instance.")

    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9

    # Match against catalog
    specs = None
    for key, val in GPU_CATALOG.items():
        if key.lower() in name.lower():
            specs = val
            break

    if specs is None:
        print(f"⚠️  GPU '{name}' not in catalog. Using detected VRAM + estimates.")
        specs = {"tflops_fp16": 200, "bw_gbs": 900, "vram_gb": vram}

    ridge = (specs["tflops_fp16"] * 1e3) / specs["bw_gbs"]  # FLOP/byte

    return GPUInfo(
        name=name,
        vram_gb=specs["vram_gb"],
        tflops_fp16=specs["tflops_fp16"],
        bw_gbs=specs["bw_gbs"],
        ridge_point=ridge,
    )


def print_gpu_info(gpu: GPUInfo):
    """Pretty-print GPU info."""
    print(f"GPU: {gpu.name}")
    print(f"  VRAM:       {gpu.vram_gb:.0f} GB")
    print(f"  FP16 Peak:  {gpu.tflops_fp16} TFLOPS")
    print(f"  Memory BW:  {gpu.bw_gbs} GB/s")
    print(f"  Ridge Point: {gpu.ridge_point:.1f} FLOP/byte")
    print(f"  → Below ridge = memory-bound, above = compute-bound")
