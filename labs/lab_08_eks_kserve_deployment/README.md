# Lab 8: EKS + KServe Deployment

## Overview

Deploy LLM inference on Amazon EKS with KServe for Kubernetes-native model serving with autoscaling, canary deployments, and model routing.

## Learning Objectives

- Set up EKS cluster with GPU nodes
- Install and configure KServe
- Deploy vLLM as InferenceService
- Configure autoscaling and canary deployments

## Prerequisites

- Completed Labs 1-7
- AWS account with EKS permissions
- kubectl and eksctl installed
- Helm 3.x installed

## Setup

```bash
# Install eksctl
brew install eksctl  # macOS

# Install kubectl
brew install kubectl

# Install Helm
brew install helm
```

## Duration

90-120 minutes

## AWS Cost

~$20-50 (EKS cluster + GPU nodes for ~2 hours)

## Exercises

1. **EKS Cluster Setup**: Create cluster with GPU node group
2. **KServe Installation**: Install KServe with Istio
3. **InferenceService Deployment**: Deploy vLLM model
4. **Autoscaling Configuration**: Set up HPA for LLM workloads
5. **Canary Deployment**: Deploy new model version with traffic split

## Cluster Creation

```bash
eksctl create cluster \
    --name llm-inference \
    --region us-west-2 \
    --nodegroup-name gpu-nodes \
    --node-type g5.2xlarge \
    --nodes 2 \
    --nodes-min 1 \
    --nodes-max 4
```

## KServe InferenceService

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama-3-8b
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 4
    containers:
      - name: kserve-container
        image: vllm/vllm-openai:latest
        args:
          - --model=meta-llama/Llama-3.1-8B-Instruct
        resources:
          limits:
            nvidia.com/gpu: 1
```

## Validation Checkpoints

- [ ] EKS cluster is running with GPU nodes
- [ ] KServe is installed and healthy
- [ ] InferenceService is ready
- [ ] Autoscaling responds to load
- [ ] Canary deployment routes traffic correctly

## Cleanup

```bash
# Delete cluster to avoid ongoing charges
eksctl delete cluster --name llm-inference
```

## Next Steps

Proceed to Lab 9: SageMaker Production Deployment for managed inference.
