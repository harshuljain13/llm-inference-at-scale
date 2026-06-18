# 8.3 Edge Deployment

> Running LLMs on consumer hardware: llama.cpp, GGUF, and Apple Silicon

---

## Learning Objectives

By the end of this module, you will:

- Understand llama.cpp architecture and GGUF format
- Configure quantization for edge devices
- Deploy models on Apple Silicon with MLX
- Compare edge deployment options

---

## Why Edge Deployment?

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EDGE DEPLOYMENT USE CASES                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Use Case              │ Benefit                                   │
│   ══════════════════════│═══════════════════════════════════════    │
│                         │                                           │
│   Privacy-sensitive     │ Data never leaves device                  │
│   Offline operation     │ No internet required                      │
│   Low latency           │ No network round-trip                     │
│   Cost reduction        │ No cloud compute costs                    │
│   Development/testing   │ Fast iteration without GPU servers        │
│   Embedded systems      │ IoT, robotics, automotive                 │
│                         │                                           │
│   ─────────────────────────────────────────────────────────────    │
│                         │                                           │
│   Tradeoffs:            │                                           │
│   ══════════                                                        │
│                         │                                           │
│   • Lower throughput than GPU servers                               │
│   • Limited model size (constrained by RAM)                         │
│   • Aggressive quantization may reduce quality                      │
│   • No batching benefits (typically single user)                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Edge Deployment Options

### Platform Comparison

| Platform             | Framework      | Memory    | Throughput   | Use Case             |
| -------------------- | -------------- | --------- | ------------ | -------------------- |
| CPU (x86)            | llama.cpp      | 8-64 GB   | 10-30 tok/s  | Development, servers |
| Apple Silicon        | MLX, llama.cpp | 16-192 GB | 30-100 tok/s | Mac development      |
| NVIDIA Jetson        | TensorRT-LLM   | 8-64 GB   | 20-50 tok/s  | Edge AI, robotics    |
| Mobile (iOS/Android) | llama.cpp, MLC | 4-8 GB    | 5-15 tok/s   | On-device apps       |
| Raspberry Pi 5       | llama.cpp      | 8 GB      | 1-5 tok/s    | Hobby, education     |

---

## llama.cpp

### Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLAMA.CPP ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Key Features:                                                     │
│   ══════════════                                                    │
│                                                                     │
│   • Pure C/C++ implementation (no Python dependencies)              │
│   • GGUF format for efficient model loading                         │
│   • Extensive quantization support (Q2 to Q8)                       │
│   • CPU optimizations (AVX, AVX2, AVX-512, ARM NEON)                │
│   • GPU acceleration (CUDA, Metal, Vulkan, OpenCL)                  │
│   • Memory-mapped model loading                                     │
│   • Speculative decoding support                                    │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   Architecture:                                                     │
│   ═════════════                                                     │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                      llama.cpp                              │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │   │
│   │  │   GGUF      │  │  Quantized  │  │   Backend   │          │   │
│   │  │   Loader    │──│   Weights   │──│   (CPU/GPU) │          │   │
│   │  └─────────────┘  └─────────────┘  └─────────────┘          │   │
│   │         │                                  │                │   │
│   │         ▼                                  ▼                │   │
│   │  ┌─────────────┐                   ┌─────────────┐          │   │
│   │  │   Memory    │                   │   SIMD      │          │   │
│   │  │   Mapping   │                   │   Kernels   │          │   │
│   │  └─────────────┘                   └─────────────┘          │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Installation and Setup

```bash
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build for CPU (with optimizations)
make -j$(nproc)

# Build with CUDA support
make LLAMA_CUDA=1 -j$(nproc)

# Build with Metal support (macOS)
make LLAMA_METAL=1 -j$(nproc)

# Build with Vulkan support
make LLAMA_VULKAN=1 -j$(nproc)
```

### Download GGUF Models

```bash
# Download pre-quantized models from HuggingFace
# Example: Llama 3.1 8B in Q4_K_M quantization

# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download \
    TheBloke/Llama-3.1-8B-Instruct-GGUF \
    llama-3.1-8b-instruct.Q4_K_M.gguf \
    --local-dir ./models

# Or using wget
wget https://huggingface.co/TheBloke/Llama-3.1-8B-Instruct-GGUF/resolve/main/llama-3.1-8b-instruct.Q4_K_M.gguf \
    -O ./models/llama-3.1-8b-instruct.Q4_K_M.gguf
```

### Running Inference

```bash
# Basic inference
./llama-cli \
    -m ./models/llama-3.1-8b-instruct.Q4_K_M.gguf \
    -p "What is machine learning?" \
    -n 100

# Interactive chat mode
./llama-cli \
    -m ./models/llama-3.1-8b-instruct.Q4_K_M.gguf \
    --interactive \
    --color \
    -r "User:" \
    --in-prefix " " \
    -n 256

# Server mode (OpenAI-compatible API)
./llama-server \
    -m ./models/llama-3.1-8b-instruct.Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -c 4096 \
    -t 8

# Performance tuning
./llama-cli \
    -m ./models/llama-3.1-8b-instruct.Q4_K_M.gguf \
    -p "Hello" \
    -n 100 \
    -t 8 \           # Number of threads
    -ngl 35 \        # Layers to offload to GPU
    --mlock \        # Lock model in RAM
    -c 4096          # Context size
```

---

## GGUF Format

### Quantization Options

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GGUF QUANTIZATION LEVELS                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Quant    │ Bits │ Size (7B) │ Quality │ Speed │ Use Case          │
│   ═════════│══════│═══════════│═════════│═══════│═══════════════    │
│            │      │           │         │       │                   │
│   F16      │ 16   │ 14 GB     │ Best    │ Slow  │ Reference         │
│   Q8_0     │ 8    │ 7.2 GB    │ Excellent│ Fast │ Quality-focused   │
│   Q6_K     │ 6    │ 5.5 GB    │ Very Good│ Fast │ Balanced          │
│   Q5_K_M   │ 5    │ 4.8 GB    │ Good    │ Fast  │ Recommended       │
│   Q4_K_M   │ 4    │ 4.1 GB    │ Good    │ Faster│ Most popular      │
│   Q4_K_S   │ 4    │ 3.9 GB    │ OK      │ Faster│ Memory-limited    │
│   Q3_K_M   │ 3    │ 3.3 GB    │ Usable  │ Fast  │ Very constrained  │
│   Q2_K     │ 2    │ 2.7 GB    │ Poor    │ Fastest│ Extreme limits   │
│            │      │           │         │       │                   │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   Naming Convention:                                                │
│   ══════════════════                                                │
│                                                                     │
│   Q4_K_M = Q4 (4-bit) + K (k-quant method) + M (medium quality)     │
│                                                                     │
│   Suffixes:                                                         │
│   • _S = Small (faster, lower quality)                              │
│   • _M = Medium (balanced)                                          │
│   • _L = Large (slower, higher quality)                             │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   Recommendation:                                                   │
│   ═══════════════                                                   │
│                                                                     │
│   • Start with Q4_K_M (best balance)                                │
│   • Use Q5_K_M if quality matters more                              │
│   • Use Q3_K_M only if RAM is very limited                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Converting Models to GGUF

```bash
# Convert HuggingFace model to GGUF
cd llama.cpp

# Install Python dependencies
pip install -r requirements.txt

# Convert to F16 GGUF
python convert_hf_to_gguf.py \
    /path/to/huggingface/model \
    --outfile ./models/model-f16.gguf \
    --outtype f16

# Quantize to Q4_K_M
./llama-quantize \
    ./models/model-f16.gguf \
    ./models/model-q4_k_m.gguf \
    Q4_K_M

# Quantize with importance matrix (better quality)
./llama-quantize \
    ./models/model-f16.gguf \
    ./models/model-q4_k_m.gguf \
    Q4_K_M \
    --imatrix ./imatrix.dat
```

---

## Apple Silicon / MLX

### MLX Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MLX ON APPLE SILICON                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Apple Silicon Advantages:                                         │
│   ══════════════════════════                                        │
│                                                                     │
│   • Unified Memory Architecture (UMA)                               │
│     - CPU and GPU share same memory                                 │
│     - No PCIe transfer bottleneck                                   │
│     - Can use all system RAM for models                             │
│                                                                     │
│   • High Memory Bandwidth                                           │
│     - M1 Max: 400 GB/s                                              │
│     - M2 Ultra: 800 GB/s                                            │
│     - M3 Max: 400 GB/s                                              │
│                                                                     │
│   • Large Memory Options                                            │
│     - M2 Ultra: up to 192 GB                                        │
│     - M3 Max: up to 128 GB                                          │
│     - Can run 70B models!                                           │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   MLX Framework:                                                    │
│   ══════════════                                                    │
│                                                                     │
│   • Apple's ML framework for Apple Silicon                          │
│   • NumPy-like API                                                  │
│   • Lazy evaluation                                                 │
│   • Automatic differentiation                                       │
│   • Optimized for Metal GPU                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### MLX Installation and Usage

```bash
# Install MLX
pip install mlx mlx-lm

# Download and run a model
mlx_lm.generate \
    --model mlx-community/Llama-3.1-8B-Instruct-4bit \
    --prompt "What is machine learning?" \
    --max-tokens 100

# Start a server
mlx_lm.server \
    --model mlx-community/Llama-3.1-8B-Instruct-4bit \
    --port 8080
```

### MLX Python API

```python
# mlx_inference.py
"""LLM inference with MLX on Apple Silicon."""

from mlx_lm import load, generate


def run_inference(
    model_path: str = "mlx-community/Llama-3.1-8B-Instruct-4bit",
    prompt: str = "What is machine learning?",
    max_tokens: int = 100,
    temperature: float = 0.7,
) -> str:
    """Run inference with MLX."""

    # Load model and tokenizer
    model, tokenizer = load(model_path)

    # Generate response
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=temperature,
    )

    return response


def stream_inference(
    model_path: str,
    prompt: str,
    max_tokens: int = 100,
):
    """Stream tokens as they're generated."""

    from mlx_lm import load, stream_generate

    model, tokenizer = load(model_path)

    for token in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
    ):
        print(token, end="", flush=True)


# Performance comparison
APPLE_SILICON_PERFORMANCE = {
    "M1": {
        "memory": "16 GB",
        "bandwidth": "68 GB/s",
        "llama_8b_q4": "~15 tok/s",
    },
    "M1 Max": {
        "memory": "64 GB",
        "bandwidth": "400 GB/s",
        "llama_8b_q4": "~50 tok/s",
        "llama_70b_q4": "~8 tok/s",
    },
    "M2 Ultra": {
        "memory": "192 GB",
        "bandwidth": "800 GB/s",
        "llama_8b_q4": "~80 tok/s",
        "llama_70b_q4": "~15 tok/s",
    },
    "M3 Max": {
        "memory": "128 GB",
        "bandwidth": "400 GB/s",
        "llama_8b_q4": "~60 tok/s",
        "llama_70b_q4": "~10 tok/s",
    },
}


if __name__ == "__main__":
    response = run_inference(
        prompt="Explain quantum computing in simple terms.",
        max_tokens=200,
    )
    print(response)
```

---

## NVIDIA Jetson Deployment

### Jetson Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NVIDIA JETSON FOR EDGE AI                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Jetson Models:                                                    │
│   ══════════════                                                    │
│                                                                     │
│   Model          │ GPU Cores │ Memory │ Power │ LLM Capability      │
│   ════════════════│═══════════│════════│═══════│════════════════    │
│   Jetson Nano    │ 128       │ 4 GB   │ 10W   │ Small models only   │
│   Jetson Xavier  │ 512       │ 32 GB  │ 30W   │ 7B quantized        │
│   Jetson Orin    │ 2048      │ 64 GB  │ 60W   │ 13B+ quantized      │
│   Orin Nano      │ 1024      │ 8 GB   │ 15W   │ 7B Q4               │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   Use Cases:                                                        │
│   ══════════                                                        │
│                                                                     │
│   • Robotics (local language understanding)                         │
│   • Automotive (in-vehicle assistants)                              │
│   • Industrial IoT (edge analytics)                                 │
│   • Smart cameras (vision + language)                               │
│   • Drones (autonomous navigation)                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Jetson Deployment with llama.cpp

```bash
# On Jetson device

# Install dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake

# Clone and build llama.cpp with CUDA
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_CUDA=1 -j$(nproc)

# Download quantized model
wget https://huggingface.co/TheBloke/Llama-3.1-8B-Instruct-GGUF/resolve/main/llama-3.1-8b-instruct.Q4_K_M.gguf

# Run inference
./llama-cli \
    -m llama-3.1-8b-instruct.Q4_K_M.gguf \
    -p "Hello, how are you?" \
    -n 100 \
    -ngl 99  # Offload all layers to GPU
```

---

## Mobile Deployment

### iOS/Android Options

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MOBILE LLM DEPLOYMENT                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Framework       │ Platform    │ Size Limit │ Performance          │
│   ════════════════│═════════════│════════════│═══════════════════   │
│   llama.cpp       │ iOS/Android │ ~4 GB      │ 5-15 tok/s           │
│   MLC LLM         │ iOS/Android │ ~4 GB      │ 10-20 tok/s          │
│   Core ML         │ iOS only    │ ~8 GB      │ 15-30 tok/s          │
│   MediaPipe       │ iOS/Android │ ~2 GB      │ 5-10 tok/s           │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   Practical Limits:                                                 │
│   ═════════════════                                                 │
│                                                                     │
│   • iPhone 15 Pro: 8 GB RAM → ~3B model Q4                          │
│   • iPhone 15 Pro Max: 8 GB RAM → ~3B model Q4                      │
│   • High-end Android: 12 GB RAM → ~7B model Q4                      │
│                                                                     │
│   Recommended Models:                                               │
│   ═══════════════════                                               │
│                                                                     │
│   • Phi-3 Mini (3.8B) - Best quality/size ratio                     │
│   • Gemma 2B - Google's small model                                 │
│   • TinyLlama (1.1B) - Very fast, lower quality                     │
│   • Qwen2 1.5B - Good multilingual support                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### MLC LLM for Mobile

```bash
# Install MLC LLM
pip install mlc-llm

# Convert model for mobile
mlc_llm convert \
    --model HuggingFaceH4/zephyr-7b-beta \
    --quantization q4f16_1 \
    --device android

# Package for Android
mlc_llm package \
    --model zephyr-7b-beta-q4f16_1 \
    --target android
```

---

## Performance Optimization

### CPU Optimization Tips

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CPU OPTIMIZATION TIPS                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   1. Thread Count                                                   │
│   ═══════════════                                                   │
│   • Use physical cores, not logical (hyperthreading)                │
│   • Example: 8-core CPU → use -t 8, not -t 16                       │
│   • Too many threads = context switching overhead                   │
│                                                                     │
│   2. Memory Locking                                                 │
│   ════════════════                                                  │
│   • Use --mlock to prevent swapping                                 │
│   • Requires sufficient RAM                                         │
│   • May need: sudo sysctl -w vm.max_locked_memory=unlimited         │
│                                                                     │
│   3. NUMA Awareness                                                 │
│   ════════════════                                                  │
│   • On multi-socket systems, pin to one NUMA node                   │
│   • numactl --cpunodebind=0 --membind=0 ./llama-cli ...             │
│                                                                     │
│   4. Batch Size                                                     │
│   ═════════════                                                     │
│   • Larger batch = better throughput, higher latency                │
│   • For interactive: batch=1                                        │
│   • For batch processing: batch=8-32                                │
│                                                                     │
│   5. Context Size                                                   │
│   ══════════════                                                    │
│   • Smaller context = faster, less memory                           │
│   • Use minimum needed for your use case                            │
│   • -c 2048 instead of -c 8192 if possible                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Benchmarking Edge Performance

```python
# benchmark_edge.py
"""Benchmark LLM performance on edge devices."""

import subprocess
import time
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkResult:
    model: str
    quantization: str
    prompt_tokens: int
    generated_tokens: int
    prompt_eval_time_ms: float
    generation_time_ms: float
    tokens_per_second: float
    memory_used_mb: float


def benchmark_llama_cpp(
    model_path: str,
    prompt: str = "What is machine learning?",
    n_tokens: int = 100,
    threads: int = 8,
    gpu_layers: int = 0,
) -> BenchmarkResult:
    """Benchmark llama.cpp inference."""

    cmd = [
        "./llama-cli",
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_tokens),
        "-t", str(threads),
        "-ngl", str(gpu_layers),
        "--no-display-prompt",
    ]

    start = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start

    # Parse output for timing info
    output = result.stderr

    # Extract metrics from llama.cpp output
    prompt_eval = re.search(r'prompt eval time =\s+([\d.]+) ms', output)
    generation = re.search(r'eval time =\s+([\d.]+) ms', output)
    tokens_per_sec = re.search(r'([\d.]+) tokens per second', output)

    return BenchmarkResult(
        model=model_path,
        quantization=model_path.split('.')[-2],  # e.g., Q4_K_M
        prompt_tokens=len(prompt.split()),
        generated_tokens=n_tokens,
        prompt_eval_time_ms=float(prompt_eval.group(1)) if prompt_eval else 0,
        generation_time_ms=float(generation.group(1)) if generation else elapsed * 1000,
        tokens_per_second=float(tokens_per_sec.group(1)) if tokens_per_sec else n_tokens / elapsed,
        memory_used_mb=0,  # Would need to measure separately
    )


def compare_quantizations(
    model_base: str,
    quantizations: list = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
    prompt: str = "Explain quantum computing.",
    n_tokens: int = 100,
):
    """Compare different quantization levels."""

    results = []
    for quant in quantizations:
        model_path = f"{model_base}.{quant}.gguf"
        try:
            result = benchmark_llama_cpp(model_path, prompt, n_tokens)
            results.append(result)
            print(f"{quant}: {result.tokens_per_second:.1f} tok/s")
        except Exception as e:
            print(f"{quant}: Error - {e}")

    return results


if __name__ == "__main__":
    # Single benchmark
    result = benchmark_llama_cpp(
        model_path="./models/llama-3.1-8b-instruct.Q4_K_M.gguf",
        prompt="What is the meaning of life?",
        n_tokens=100,
        threads=8,
    )
    print(f"Tokens/sec: {result.tokens_per_second:.1f}")

    # Compare quantizations
    compare_quantizations("./models/llama-3.1-8b-instruct")
```

---

## Key Takeaways

1. **llama.cpp is the standard** - Works everywhere, extensive quantization support

2. **GGUF is the format** - Efficient, self-contained, widely supported

3. **Q4_K_M is the sweet spot** - Best balance of quality and size

4. **Apple Silicon excels** - Unified memory enables large models on laptops

5. **Mobile is limited** - Stick to 1-3B models for practical use

6. **Optimize for your hardware** - Thread count, memory locking, GPU offload

---

## Quick Reference

### Model Size vs RAM Requirements

| Model Size | Q4_K_M Size | Min RAM | Recommended RAM |
| ---------- | ----------- | ------- | --------------- |
| 1B         | ~0.6 GB     | 2 GB    | 4 GB            |
| 3B         | ~1.8 GB     | 4 GB    | 8 GB            |
| 7B         | ~4.1 GB     | 8 GB    | 16 GB           |
| 13B        | ~7.4 GB     | 12 GB   | 24 GB           |
| 34B        | ~19 GB      | 24 GB   | 48 GB           |
| 70B        | ~40 GB      | 48 GB   | 96 GB           |

### Platform Recommendations

| Use Case           | Platform       | Model           | Expected Performance |
| ------------------ | -------------- | --------------- | -------------------- |
| Mac development    | MLX            | Llama 3.1 8B Q4 | 30-60 tok/s          |
| Linux server (CPU) | llama.cpp      | Llama 3.1 8B Q4 | 10-20 tok/s          |
| Raspberry Pi       | llama.cpp      | Phi-3 Mini Q4   | 2-5 tok/s            |
| iPhone             | MLC LLM        | Phi-3 Mini Q4   | 5-10 tok/s           |
| Jetson Orin        | llama.cpp+CUDA | Llama 3.1 8B Q4 | 20-40 tok/s          |

---

## References

1. [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
2. [MLX Documentation](https://ml-explore.github.io/mlx/)
3. [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
4. [MLC LLM](https://mlc.ai/mlc-llm/)
5. [TheBloke's Quantized Models](https://huggingface.co/TheBloke)
