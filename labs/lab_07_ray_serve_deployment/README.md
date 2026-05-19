# Lab 7: Ray Serve Deployment

## Overview

Deploy LLM inference with Ray Serve for production-grade serving with autoscaling, batching, and replica management.

## Learning Objectives

- Configure Ray Serve with vLLM backend
- Set up autoscaling based on request load
- Implement health checks and monitoring
- Handle multi-replica deployments

## Prerequisites

- Completed Labs 1-6
- AWS g5.2xlarge or larger
- Understanding of Ray basics

## Setup

```bash
pip install "ray[serve]" vllm
```

## Duration

60-90 minutes

## AWS Cost

~$2-5 (depending on scaling tests)

## Exercises

1. **Basic Deployment**: Deploy vLLM with Ray Serve
2. **Autoscaling Configuration**: Set up request-based scaling
3. **Load Testing**: Test autoscaling behavior
4. **Multi-Replica**: Deploy multiple model replicas

## Deployment Example

```python
from ray import serve
from vllm import LLM, SamplingParams

@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_num_ongoing_requests_per_replica": 5,
    },
)
class VLLMDeployment:
    def __init__(self):
        self.llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

    async def __call__(self, request):
        prompt = request.query_params.get("prompt", "")
        outputs = self.llm.generate([prompt])
        return {"text": outputs[0].outputs[0].text}

app = VLLMDeployment.bind()
serve.run(app)
```

## Validation Checkpoints

- [ ] Ray Serve deployment starts successfully
- [ ] Requests are handled correctly
- [ ] Autoscaling triggers on load increase
- [ ] Health checks pass

## Next Steps

Proceed to Lab 8: EKS + KServe Deployment for Kubernetes-native serving.
