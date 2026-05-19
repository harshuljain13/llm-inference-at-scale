# Lab 9: SageMaker Production Deployment

## Overview

Deploy LLM inference on Amazon SageMaker using LMI (Large Model Inference) containers with vLLM backend for managed, production-grade serving.

## Learning Objectives

- Deploy models using SageMaker LMI containers
- Configure vLLM backend options
- Set up autoscaling policies
- Monitor endpoint performance

## Prerequisites

- Completed Labs 1-8
- AWS account with SageMaker permissions
- SageMaker execution role
- HuggingFace token

## Setup

```bash
pip install sagemaker boto3
```

## Duration

60-90 minutes

## AWS Cost

~$5-10 (SageMaker endpoint for ~90 minutes)

## Exercises

1. **LMI Container Deployment**: Deploy with DJLModel
2. **vLLM Backend Configuration**: Optimize for throughput
3. **Autoscaling Setup**: Configure scaling policies
4. **Endpoint Testing**: Benchmark the endpoint
5. **Multi-Model Endpoint**: Deploy multiple models (optional)

## Deployment Example

```python
from sagemaker.djl_inference import DJLModel

model = DJLModel(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    role=role,
    task="text-generation",
    env={
        "OPTION_ROLLING_BATCH": "vllm",
        "OPTION_MAX_ROLLING_BATCH_SIZE": "64",
        "OPTION_TENSOR_PARALLEL_DEGREE": "1",
        "OPTION_GPU_MEMORY_UTILIZATION": "0.95",
    },
)

predictor = model.deploy(
    instance_type="ml.g5.2xlarge",
    initial_instance_count=1,
)
```

## Autoscaling Configuration

```python
client.put_scaling_policy(
    PolicyName="llm-scaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 5.0,  # requests per instance
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
        },
    },
)
```

## Validation Checkpoints

- [ ] Endpoint deploys successfully
- [ ] Inference requests return valid responses
- [ ] Autoscaling triggers on load
- [ ] CloudWatch metrics are populated

## Cleanup

```python
predictor.delete_endpoint()
predictor.delete_model()
```

## Next Steps

Proceed to Lab 10: Benchmarking and Monitoring for production operations.
