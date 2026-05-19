#!/usr/bin/env python3
"""
LLM Inference Cost Calculator

Calculate and compare costs for different AWS deployment options.
Supports EC2 GPU instances, SageMaker endpoints, and Bedrock.

Usage:
    python cost_calculator.py --model llama-3.1-8b --requests-per-day 100000
    python cost_calculator.py --interactive
"""

import argparse
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class InstanceType(Enum):
    """AWS GPU instance types for LLM inference."""
    # G5 instances (A10G GPUs)
    G5_XLARGE = "g5.xlarge"      # 1x A10G, 24GB
    G5_2XLARGE = "g5.2xlarge"    # 1x A10G, 24GB
    G5_4XLARGE = "g5.4xlarge"    # 1x A10G, 24GB
    G5_12XLARGE = "g5.12xlarge"  # 4x A10G, 96GB
    G5_48XLARGE = "g5.48xlarge"  # 8x A10G, 192GB
    
    # P4d instances (A100 GPUs)
    P4D_24XLARGE = "p4d.24xlarge"  # 8x A100 40GB, 320GB
    
    # P5 instances (H100 GPUs)
    P5_48XLARGE = "p5.48xlarge"    # 8x H100 80GB, 640GB
    
    # Inferentia2
    INF2_XLARGE = "inf2.xlarge"    # 1x Inferentia2
    INF2_8XLARGE = "inf2.8xlarge"  # 1x Inferentia2
    INF2_24XLARGE = "inf2.24xlarge"  # 6x Inferentia2
    INF2_48XLARGE = "inf2.48xlarge"  # 12x Inferentia2


@dataclass
class InstanceSpec:
    """Specifications for an AWS instance type."""
    instance_type: str
    gpu_count: int
    gpu_memory_gb: int
    hourly_cost_usd: float  # On-demand pricing (us-east-1)
    memory_bandwidth_gbps: float
    gpu_type: str
    
    @property
    def total_gpu_memory_gb(self) -> int:
        return self.gpu_count * self.gpu_memory_gb


# AWS pricing as of 2024 (us-east-1, on-demand)
INSTANCE_SPECS = {
    "g5.xlarge": InstanceSpec("g5.xlarge", 1, 24, 1.006, 600, "A10G"),
    "g5.2xlarge": InstanceSpec("g5.2xlarge", 1, 24, 1.212, 600, "A10G"),
    "g5.4xlarge": InstanceSpec("g5.4xlarge", 1, 24, 1.624, 600, "A10G"),
    "g5.12xlarge": InstanceSpec("g5.12xlarge", 4, 24, 5.672, 600, "A10G"),
    "g5.48xlarge": InstanceSpec("g5.48xlarge", 8, 24, 16.288, 600, "A10G"),
    "p4d.24xlarge": InstanceSpec("p4d.24xlarge", 8, 40, 32.77, 2000, "A100"),
    "p5.48xlarge": InstanceSpec("p5.48xlarge", 8, 80, 98.32, 3350, "H100"),
    "inf2.xlarge": InstanceSpec("inf2.xlarge", 1, 32, 0.758, 190, "Inferentia2"),
    "inf2.8xlarge": InstanceSpec("inf2.8xlarge", 1, 32, 1.968, 190, "Inferentia2"),
    "inf2.24xlarge": InstanceSpec("inf2.24xlarge", 6, 32, 6.49, 190, "Inferentia2"),
    "inf2.48xlarge": InstanceSpec("inf2.48xlarge", 12, 32, 12.98, 190, "Inferentia2"),
}


@dataclass
class ModelSpec:
    """Specifications for an LLM model."""
    name: str
    parameters_b: float  # Billions of parameters
    fp16_memory_gb: float  # Memory for FP16 weights
    int8_memory_gb: float  # Memory for INT8 weights
    int4_memory_gb: float  # Memory for INT4 weights
    kv_heads: int  # Number of KV heads (for GQA)
    layers: int
    head_dim: int
    recommended_instance: str
    
    def kv_cache_per_token_bytes(self, dtype_bytes: int = 2) -> float:
        """Calculate KV cache memory per token in bytes."""
        return 2 * self.layers * self.kv_heads * self.head_dim * dtype_bytes


# Common model specifications
MODEL_SPECS = {
    "llama-3.1-8b": ModelSpec(
        "Llama 3.1 8B", 8.0, 16.0, 8.0, 4.0,
        kv_heads=8, layers=32, head_dim=128,
        recommended_instance="g5.xlarge"
    ),
    "llama-3.1-70b": ModelSpec(
        "Llama 3.1 70B", 70.0, 140.0, 70.0, 35.0,
        kv_heads=8, layers=80, head_dim=128,
        recommended_instance="p4d.24xlarge"
    ),
    "llama-3.1-405b": ModelSpec(
        "Llama 3.1 405B", 405.0, 810.0, 405.0, 202.5,
        kv_heads=8, layers=126, head_dim=128,
        recommended_instance="p5.48xlarge"
    ),
    "mistral-7b": ModelSpec(
        "Mistral 7B", 7.0, 14.0, 7.0, 3.5,
        kv_heads=8, layers=32, head_dim=128,
        recommended_instance="g5.xlarge"
    ),
    "mixtral-8x7b": ModelSpec(
        "Mixtral 8x7B", 46.7, 93.4, 46.7, 23.35,
        kv_heads=8, layers=32, head_dim=128,
        recommended_instance="g5.48xlarge"
    ),
}


@dataclass
class BedrockPricing:
    """Bedrock pricing per 1K tokens."""
    model_id: str
    input_per_1k: float
    output_per_1k: float


# Bedrock pricing (as of 2024)
BEDROCK_PRICING = {
    "llama-3.1-8b": BedrockPricing("meta.llama3-1-8b-instruct-v1:0", 0.0003, 0.0006),
    "llama-3.1-70b": BedrockPricing("meta.llama3-1-70b-instruct-v1:0", 0.00265, 0.0035),
    "claude-3-sonnet": BedrockPricing("anthropic.claude-3-sonnet-20240229-v1:0", 0.003, 0.015),
    "claude-3-haiku": BedrockPricing("anthropic.claude-3-haiku-20240307-v1:0", 0.00025, 0.00125),
}


def calculate_vram_requirement(
    model: ModelSpec,
    batch_size: int,
    sequence_length: int,
    quantization: str = "fp16"
) -> dict:
    """
    Calculate total VRAM requirement for a model deployment.
    
    Args:
        model: Model specification
        batch_size: Maximum concurrent sequences
        sequence_length: Maximum sequence length
        quantization: One of 'fp16', 'int8', 'int4'
    
    Returns:
        Dictionary with memory breakdown
    """
    # Model weights
    if quantization == "fp16":
        model_memory_gb = model.fp16_memory_gb
        kv_dtype_bytes = 2
    elif quantization == "int8":
        model_memory_gb = model.int8_memory_gb
        kv_dtype_bytes = 2  # KV cache usually stays FP16
    elif quantization == "int4":
        model_memory_gb = model.int4_memory_gb
        kv_dtype_bytes = 2
    else:
        raise ValueError(f"Unknown quantization: {quantization}")
    
    # KV cache
    kv_cache_bytes = (
        model.kv_cache_per_token_bytes(kv_dtype_bytes) 
        * sequence_length 
        * batch_size
    )
    kv_cache_gb = kv_cache_bytes / (1024**3)
    
    # Activations and overhead (rough estimate: 10-20% of model size)
    overhead_gb = model_memory_gb * 0.15
    
    total_gb = model_memory_gb + kv_cache_gb + overhead_gb
    
    return {
        "model_weights_gb": model_memory_gb,
        "kv_cache_gb": kv_cache_gb,
        "overhead_gb": overhead_gb,
        "total_gb": total_gb,
        "quantization": quantization,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
    }


def estimate_throughput(
    instance: InstanceSpec,
    model: ModelSpec,
    quantization: str = "fp16"
) -> dict:
    """
    Estimate inference throughput for a given instance and model.
    
    This is a simplified estimate based on memory bandwidth.
    Actual throughput depends on many factors.
    """
    # Bytes per parameter
    if quantization == "fp16":
        bytes_per_param = 2
    elif quantization == "int8":
        bytes_per_param = 1
    elif quantization == "int4":
        bytes_per_param = 0.5
    else:
        bytes_per_param = 2
    
    # Memory bandwidth per GPU (GB/s)
    bandwidth_per_gpu = instance.memory_bandwidth_gbps
    total_bandwidth = bandwidth_per_gpu * instance.gpu_count
    
    # Bytes to read per token (simplified: just model weights)
    bytes_per_token = model.parameters_b * 1e9 * bytes_per_param
    
    # Theoretical max tokens/sec (decode phase, memory-bound)
    theoretical_tps = (total_bandwidth * 1e9) / bytes_per_token
    
    # Apply efficiency factor (typically 50-70% of theoretical)
    efficiency = 0.6
    estimated_tps = theoretical_tps * efficiency
    
    return {
        "theoretical_tokens_per_sec": theoretical_tps,
        "estimated_tokens_per_sec": estimated_tps,
        "memory_bandwidth_gbps": total_bandwidth,
        "bytes_per_token": bytes_per_token,
    }


def calculate_ec2_cost(
    instance: InstanceSpec,
    requests_per_day: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    model: ModelSpec,
    quantization: str = "fp16",
    utilization: float = 0.7
) -> dict:
    """
    Calculate EC2 deployment cost.
    
    Args:
        instance: Instance specification
        requests_per_day: Number of requests per day
        avg_input_tokens: Average input tokens per request
        avg_output_tokens: Average output tokens per request
        model: Model specification
        quantization: Quantization method
        utilization: Expected GPU utilization (0-1)
    
    Returns:
        Cost breakdown dictionary
    """
    throughput = estimate_throughput(instance, model, quantization)
    tokens_per_sec = throughput["estimated_tokens_per_sec"] * utilization
    
    # Total tokens per day
    total_tokens_per_day = requests_per_day * (avg_input_tokens + avg_output_tokens)
    
    # Time needed per day (seconds)
    seconds_per_day = total_tokens_per_day / tokens_per_sec if tokens_per_sec > 0 else float('inf')
    hours_per_day = seconds_per_day / 3600
    
    # Number of instances needed (assuming 24/7 operation)
    instances_needed = max(1, int(hours_per_day / 24) + 1)
    
    # Daily and monthly costs
    daily_cost = instances_needed * instance.hourly_cost_usd * 24
    monthly_cost = daily_cost * 30
    
    # Cost per 1M tokens
    tokens_per_month = total_tokens_per_day * 30
    cost_per_1m_tokens = (monthly_cost / tokens_per_month) * 1e6 if tokens_per_month > 0 else 0
    
    return {
        "instance_type": instance.instance_type,
        "instances_needed": instances_needed,
        "hourly_cost_per_instance": instance.hourly_cost_usd,
        "daily_cost": daily_cost,
        "monthly_cost": monthly_cost,
        "cost_per_1m_tokens": cost_per_1m_tokens,
        "estimated_tokens_per_sec": tokens_per_sec,
        "utilization": utilization,
    }


def calculate_bedrock_cost(
    model_key: str,
    requests_per_day: int,
    avg_input_tokens: int,
    avg_output_tokens: int
) -> dict:
    """
    Calculate Bedrock cost for a given usage pattern.
    """
    if model_key not in BEDROCK_PRICING:
        return {"error": f"Model {model_key} not available in Bedrock pricing"}
    
    pricing = BEDROCK_PRICING[model_key]
    
    # Daily tokens
    input_tokens_per_day = requests_per_day * avg_input_tokens
    output_tokens_per_day = requests_per_day * avg_output_tokens
    
    # Daily cost
    daily_input_cost = (input_tokens_per_day / 1000) * pricing.input_per_1k
    daily_output_cost = (output_tokens_per_day / 1000) * pricing.output_per_1k
    daily_cost = daily_input_cost + daily_output_cost
    
    # Monthly cost
    monthly_cost = daily_cost * 30
    
    # Cost per 1M tokens (blended)
    total_tokens = input_tokens_per_day + output_tokens_per_day
    cost_per_1m_tokens = (daily_cost / total_tokens) * 1e6 if total_tokens > 0 else 0
    
    return {
        "model_id": pricing.model_id,
        "daily_cost": daily_cost,
        "monthly_cost": monthly_cost,
        "cost_per_1m_tokens": cost_per_1m_tokens,
        "input_cost_per_1k": pricing.input_per_1k,
        "output_cost_per_1k": pricing.output_per_1k,
    }


def compare_deployment_options(
    model_key: str,
    requests_per_day: int,
    avg_input_tokens: int = 500,
    avg_output_tokens: int = 200
) -> dict:
    """
    Compare costs across different deployment options.
    """
    results = {
        "model": model_key,
        "requests_per_day": requests_per_day,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "options": []
    }
    
    # Get model spec
    if model_key in MODEL_SPECS:
        model = MODEL_SPECS[model_key]
        
        # EC2 with recommended instance
        instance = INSTANCE_SPECS.get(model.recommended_instance)
        if instance:
            ec2_cost = calculate_ec2_cost(
                instance, requests_per_day, 
                avg_input_tokens, avg_output_tokens,
                model, "fp16"
            )
            results["options"].append({
                "type": "EC2",
                "details": ec2_cost
            })
            
            # EC2 with INT8 quantization
            ec2_int8_cost = calculate_ec2_cost(
                instance, requests_per_day,
                avg_input_tokens, avg_output_tokens,
                model, "int8"
            )
            results["options"].append({
                "type": "EC2 (INT8)",
                "details": ec2_int8_cost
            })
    
    # Bedrock
    if model_key in BEDROCK_PRICING:
        bedrock_cost = calculate_bedrock_cost(
            model_key, requests_per_day,
            avg_input_tokens, avg_output_tokens
        )
        results["options"].append({
            "type": "Bedrock",
            "details": bedrock_cost
        })
    
    return results


def print_comparison(comparison: dict):
    """Pretty print comparison results."""
    print("\n" + "="*70)
    print(f"Cost Comparison: {comparison['model']}")
    print(f"Requests/day: {comparison['requests_per_day']:,}")
    print(f"Avg tokens: {comparison['avg_input_tokens']} input, {comparison['avg_output_tokens']} output")
    print("="*70)
    
    for option in comparison["options"]:
        print(f"\n{option['type']}:")
        details = option["details"]
        if "error" in details:
            print(f"  {details['error']}")
            continue
            
        print(f"  Monthly cost: ${details['monthly_cost']:,.2f}")
        print(f"  Cost per 1M tokens: ${details['cost_per_1m_tokens']:.4f}")
        
        if "instance_type" in details:
            print(f"  Instance: {details['instance_type']} x {details['instances_needed']}")
            print(f"  Throughput: {details['estimated_tokens_per_sec']:.0f} tokens/sec")


def interactive_mode():
    """Run interactive cost calculator."""
    print("\n" + "="*50)
    print("LLM Inference Cost Calculator")
    print("="*50)
    
    # List available models
    print("\nAvailable models:")
    for i, (key, model) in enumerate(MODEL_SPECS.items(), 1):
        print(f"  {i}. {key} ({model.name})")
    
    # Get model selection
    model_key = input("\nEnter model name (or number): ").strip()
    if model_key.isdigit():
        model_key = list(MODEL_SPECS.keys())[int(model_key) - 1]
    
    # Get usage parameters
    requests_per_day = int(input("Requests per day: "))
    avg_input_tokens = int(input("Average input tokens [500]: ") or "500")
    avg_output_tokens = int(input("Average output tokens [200]: ") or "200")
    
    # Calculate and display
    comparison = compare_deployment_options(
        model_key, requests_per_day,
        avg_input_tokens, avg_output_tokens
    )
    print_comparison(comparison)


def main():
    parser = argparse.ArgumentParser(description="LLM Inference Cost Calculator")
    parser.add_argument("--model", type=str, help="Model name (e.g., llama-3.1-8b)")
    parser.add_argument("--requests-per-day", type=int, help="Number of requests per day")
    parser.add_argument("--input-tokens", type=int, default=500, help="Average input tokens")
    parser.add_argument("--output-tokens", type=int, default=200, help="Average output tokens")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-instances", action="store_true", help="List available instances")
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\nAvailable Models:")
        print("-" * 60)
        for key, model in MODEL_SPECS.items():
            print(f"{key:20} {model.name:25} {model.parameters_b}B params")
        return
    
    if args.list_instances:
        print("\nAvailable Instances:")
        print("-" * 80)
        print(f"{'Instance':20} {'GPUs':6} {'Memory':10} {'$/hr':10} {'GPU Type':10}")
        print("-" * 80)
        for key, spec in INSTANCE_SPECS.items():
            print(f"{spec.instance_type:20} {spec.gpu_count:6} {spec.total_gpu_memory_gb:8}GB ${spec.hourly_cost_usd:8.3f} {spec.gpu_type:10}")
        return
    
    if args.interactive:
        interactive_mode()
        return
    
    if args.model and args.requests_per_day:
        comparison = compare_deployment_options(
            args.model,
            args.requests_per_day,
            args.input_tokens,
            args.output_tokens
        )
        print_comparison(comparison)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
