# Lab 3: Quantization Comparison

## Overview

Compare different quantization methods (FP16, INT8, INT4, AWQ) and measure their impact on accuracy, throughput, and memory usage.

## Learning Objectives

- Understand quantization precision formats
- Measure accuracy/throughput tradeoffs
- Configure vLLM with different quantization backends
- Select appropriate quantization for your use case

## Prerequisites

- Completed Labs 1-2
- AWS account with GPU access (g5.xlarge recommended)
- HuggingFace token for model access

## Setup

```bash
pip install vllm transformers accelerate
```

## Duration

60-90 minutes

## AWS Cost

~$1.50 (g5.xlarge for ~90 minutes)

## Exercises

1. **Baseline FP16**: Deploy Llama 3.1 8B in FP16
2. **AWQ INT4**: Deploy AWQ-quantized model
3. **Benchmark Comparison**: Measure throughput and latency
4. **Accuracy Evaluation**: Compare output quality

## Quantization Options

| Method | vLLM Flag             | Memory | Quality             |
| ------ | --------------------- | ------ | ------------------- |
| FP16   | (default)             | 16 GB  | Baseline            |
| AWQ    | `quantization="awq"`  | 4 GB   | <1% loss            |
| GPTQ   | `quantization="gptq"` | 4 GB   | 1-3% loss           |
| FP8    | `quantization="fp8"`  | 8 GB   | Minimal (H100 only) |

## Validation Checkpoints

- [ ] FP16 model loads successfully
- [ ] AWQ model uses ~4x less memory
- [ ] Throughput improves with quantization
- [ ] Output quality remains acceptable

## Cleanup

```bash
# Terminate EC2 instance when done
aws ec2 terminate-instances --instance-ids <instance-id>
```

## Next Steps

Proceed to Lab 4: vLLM Deployment to learn production configuration tuning.
