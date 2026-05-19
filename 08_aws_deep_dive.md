# Module 8: AWS Deep Dive

> Deploying LLM inference on AWS: EC2, SageMaker, Inferentia2, and Bedrock

---

## Learning Objectives

By the end of this module, you will:

- Deploy LLM inference on EC2 GPU instances with vLLM
- Configure SageMaker endpoints with LMI containers
- Understand Inferentia2 compilation and deployment
- Compare Bedrock vs self-hosted for different use cases
- Select the right AWS service for your workload

---

## AWS Instance Selection Guide

### Decision Flowchart

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AWS INSTANCE SELECTION                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   START: What's your model size?                                    │
│   ═══════════════════════════════                                   │
│                                                                     │
│                    ┌─────────────┐                                  │
│                    │ Model Size? │                                  │
│                    └──────┬──────┘                                  │
│           ┌───────────────┼───────────────┐                         │
│           ▼               ▼               ▼                         │
│      < 15B params    15-70B params    > 70B params                  │
│           │               │               │                         │
│           ▼               ▼               ▼                         │
│      ┌────────┐      ┌────────┐      ┌────────┐                     │
│      │ Small  │      │ Medium │      │ Large  │                     │
│      └───┬────┘      └───┬────┘      └───┬────┘                     │
│          │               │               │                          │
│   ┌──────┴──────┐   ┌────┴────┐    ┌────┴────┐                      │
│   ▼             ▼   ▼         ▼    ▼         ▼                      │
│ Dev/Test    Production  Standard  Cost-Opt  Max Perf  Cost-Opt      │
│   │             │        │         │         │         │            │
│   ▼             ▼        ▼         ▼         ▼         ▼            │
│ g5.xlarge  g5.2xlarge  p4d.24xl  inf2.24xl  p5.48xl  inf2.48xl     │
│ $1.01/hr   $1.21/hr    $32.77/hr $6.49/hr   $98.32/hr $12.98/hr    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### AWS GPU Instance Comparison

| Instance      | GPU      | VRAM   | Memory BW | FP16 TFLOPS | $/hr (On-Demand) | Best For                |
| ------------- | -------- | ------ | --------- | ----------- | ---------------- | ----------------------- |
| g5.xlarge     | 1× A10G  | 24 GB  | 600 GB/s  | 125         | $1.01            | Dev/test, small models  |
| g5.2xlarge    | 1× A10G  | 24 GB  | 600 GB/s  | 125         | $1.21            | Production small models |
| g5.12xlarge   | 4× A10G  | 96 GB  | 2.4 TB/s  | 500         | $5.67            | Multi-GPU small models  |
| g5.48xlarge   | 8× A10G  | 192 GB | 4.8 TB/s  | 1000        | $16.29           | TP=8 for medium models  |
| p4d.24xlarge  | 8× A100  | 320 GB | 16 TB/s   | 2496        | $32.77           | Large models, TP=8      |
| p4de.24xlarge | 8× A100  | 640 GB | 16 TB/s   | 2496        | $40.97           | 80GB A100s              |
| p5.48xlarge   | 8× H100  | 640 GB | 26.8 TB/s | 15936       | $98.32           | Maximum throughput      |
| inf2.xlarge   | 1× Inf2  | 32 GB  | 820 GB/s  | 190         | $0.76            | Cost-optimized small    |
| inf2.8xlarge  | 1× Inf2  | 32 GB  | 820 GB/s  | 190         | $1.97            | More vCPU/memory        |
| inf2.24xlarge | 6× Inf2  | 192 GB | 4.9 TB/s  | 1140        | $6.49            | Medium models           |
| inf2.48xlarge | 12× Inf2 | 384 GB | 9.8 TB/s  | 2280        | $12.98           | Large models            |

### Model-to-Instance Mapping

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MODEL → INSTANCE MAPPING                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Model                    FP16 VRAM    Recommended Instance        │
│   ═════                    ═════════    ════════════════════        │
│                                                                     │
│   Llama 3.1 8B             ~16 GB       g5.xlarge (24 GB)           │
│   Llama 3.1 8B + batch     ~24 GB       g5.2xlarge (24 GB)          │
│                                                                     │
│   Mistral 7B               ~14 GB       g5.xlarge (24 GB)           │
│   Mixtral 8x7B             ~90 GB       g5.48xlarge (192 GB)        │
│                                                                     │
│   Llama 3.1 70B            ~140 GB      p4d.24xlarge (320 GB)       │
│   Llama 3.1 70B INT4       ~35 GB       g5.12xlarge (96 GB)         │
│                                                                     │
│   Llama 3.1 405B           ~810 GB      p5.48xlarge × 2 (1.28 TB)   │
│   Llama 3.1 405B FP8       ~405 GB      p5.48xlarge (640 GB)        │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   Quick VRAM Formula:                                               │
│   VRAM ≈ Parameters (B) × 2 bytes (FP16) + KV Cache + Overhead      │
│                                                                     │
│   Example: 70B model                                                │
│   = 70B × 2 bytes = 140 GB (weights only)                           │
│   + ~20 GB KV cache (batch=8, seq=4096)                             │
│   + ~5 GB overhead                                                  │
│   = ~165 GB total                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## EC2 + vLLM Deployment

### Launch Script

```bash
#!/bin/bash
# deploy_vllm_ec2.sh
# Deploy vLLM on EC2 g5.2xlarge

# Variables
INSTANCE_TYPE="g5.2xlarge"
AMI_ID="ami-0123456789abcdef0"  # Deep Learning AMI (Ubuntu)
KEY_NAME="your-key-pair"
SECURITY_GROUP="sg-xxx"
SUBNET_ID="subnet-xxx"
INSTANCE_NAME="vllm-inference-server"

# Launch instance
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SECURITY_GROUP \
    --subnet-id $SUBNET_ID \
    --block-device-mappings '[{
        "DeviceName": "/dev/sda1",
        "Ebs": {
            "VolumeSize": 200,
            "VolumeType": "gp3",
            "Iops": 3000,
            "Throughput": 125
        }
    }]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Launched instance: $INSTANCE_ID"

# Wait for instance to be running
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Instance IP: $PUBLIC_IP"
```

### Instance Setup Script

```bash
#!/bin/bash
# setup_vllm.sh - Run on the EC2 instance

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker (if not using Deep Learning AMI)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Set HuggingFace token
export HF_TOKEN="your-huggingface-token"

# Run vLLM container
docker run -d \
    --name vllm-server \
    --gpus all \
    --shm-size=16g \
    -p 8000:8000 \
    -e HF_TOKEN=$HF_TOKEN \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.95 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 256 \
    --enable-prefix-caching \
    --enable-chunked-prefill

# Check logs
docker logs -f vllm-server
```

### Production vLLM Configuration

```python
# vllm_production_config.py
"""Production vLLM configuration for different workloads."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VLLMConfig:
    """vLLM server configuration."""
    model: str
    gpu_memory_utilization: float = 0.95
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 256
    max_model_len: Optional[int] = None
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    tensor_parallel_size: int = 1
    quantization: Optional[str] = None

    def to_args(self) -> list:
        """Convert to command line arguments."""
        args = [
            f"--model={self.model}",
            f"--gpu-memory-utilization={self.gpu_memory_utilization}",
            f"--max-num-batched-tokens={self.max_num_batched_tokens}",
            f"--max-num-seqs={self.max_num_seqs}",
            f"--tensor-parallel-size={self.tensor_parallel_size}",
        ]
        if self.max_model_len:
            args.append(f"--max-model-len={self.max_model_len}")
        if self.enable_prefix_caching:
            args.append("--enable-prefix-caching")
        if self.enable_chunked_prefill:
            args.append("--enable-chunked-prefill")
        if self.quantization:
            args.append(f"--quantization={self.quantization}")
        return args


# Configuration profiles
CONFIGS = {
    # Throughput-optimized for batch processing
    "throughput": VLLMConfig(
        model="meta-llama/Llama-3.1-8B-Instruct",
        gpu_memory_utilization=0.95,
        max_num_batched_tokens=32768,
        max_num_seqs=512,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    ),

    # Latency-optimized for real-time chat
    "latency": VLLMConfig(
        model="meta-llama/Llama-3.1-8B-Instruct",
        gpu_memory_utilization=0.90,
        max_num_batched_tokens=8192,
        max_num_seqs=128,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    ),

    # Memory-constrained (quantized)
    "memory_constrained": VLLMConfig(
        model="casperhansen/llama-3.1-8b-instruct-awq",
        gpu_memory_utilization=0.95,
        max_num_batched_tokens=16384,
        max_num_seqs=256,
        quantization="awq",
    ),

    # Multi-GPU for 70B model
    "multi_gpu_70b": VLLMConfig(
        model="meta-llama/Llama-3.1-70B-Instruct",
        gpu_memory_utilization=0.95,
        max_num_batched_tokens=8192,
        max_num_seqs=64,
        tensor_parallel_size=8,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    ),
}
```

### CloudFormation Template for EC2

```yaml
# cloudformation/vllm-ec2.yaml
AWSTemplateFormatVersion: "2010-09-09"
Description: vLLM Inference Server on EC2

Parameters:
  InstanceType:
    Type: String
    Default: g5.2xlarge
    AllowedValues:
      - g5.xlarge
      - g5.2xlarge
      - g5.4xlarge
      - g5.12xlarge
      - p4d.24xlarge
    Description: EC2 instance type

  KeyName:
    Type: AWS::EC2::KeyPair::KeyName
    Description: SSH key pair

  VpcId:
    Type: AWS::EC2::VPC::Id
    Description: VPC ID

  SubnetId:
    Type: AWS::EC2::Subnet::Id
    Description: Subnet ID

  HuggingFaceToken:
    Type: String
    NoEcho: true
    Description: HuggingFace API token

Resources:
  SecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: vLLM inference server
      VpcId: !Ref VpcId
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 22
          ToPort: 22
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 8000
          ToPort: 8000
          CidrIp: 0.0.0.0/0

  InstanceRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Effect: Allow
            Principal:
              Service: ec2.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
        - arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy

  InstanceProfile:
    Type: AWS::IAM::InstanceProfile
    Properties:
      Roles:
        - !Ref InstanceRole

  Instance:
    Type: AWS::EC2::Instance
    Properties:
      InstanceType: !Ref InstanceType
      ImageId: ami-0123456789abcdef0 # Deep Learning AMI
      KeyName: !Ref KeyName
      SubnetId: !Ref SubnetId
      SecurityGroupIds:
        - !Ref SecurityGroup
      IamInstanceProfile: !Ref InstanceProfile
      BlockDeviceMappings:
        - DeviceName: /dev/sda1
          Ebs:
            VolumeSize: 200
            VolumeType: gp3
      UserData:
        Fn::Base64: !Sub |
          #!/bin/bash
          export HF_TOKEN=${HuggingFaceToken}
          docker run -d --gpus all --shm-size=16g -p 8000:8000 \
            -e HF_TOKEN=$HF_TOKEN \
            vllm/vllm-openai:latest \
            --model meta-llama/Llama-3.1-8B-Instruct \
            --gpu-memory-utilization 0.95
      Tags:
        - Key: Name
          Value: vllm-inference-server

Outputs:
  InstanceId:
    Value: !Ref Instance
  PublicIP:
    Value: !GetAtt Instance.PublicIp
  Endpoint:
    Value: !Sub "http://${Instance.PublicIp}:8000"
```

---

## SageMaker LMI Deployment

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SAGEMAKER LMI ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐     ┌─────────────────────────────────────────┐   │
│   │   Client    │────▶│           SageMaker Endpoint            │   │
│   └─────────────┘     │  ┌─────────────────────────────────────┐│   │
│                       │  │         Load Balancer               ││   │
│                       │  └──────────────┬──────────────────────┘│   │
│                       │                 │                       │   │
│                       │    ┌────────────┼────────────┐          │   │
│                       │    ▼            ▼            ▼          │   │
│                       │ ┌──────┐    ┌──────┐    ┌──────┐        │   │
│                       │ │Inst 1│    │Inst 2│    │Inst N│        │   │
│                       │ │      │    │      │    │      │        │   │
│                       │ │ LMI  │    │ LMI  │    │ LMI  │        │   │
│                       │ │ +    │    │ +    │    │ +    │        │   │
│                       │ │vLLM  │    │vLLM  │    │vLLM  │        │   │
│                       │ └──────┘    └──────┘    └──────┘        │   │
│                       └─────────────────────────────────────────┘   │
│                                                                     │
│   LMI = Large Model Inference Container                             │
│   Supports: vLLM, TensorRT-LLM, DeepSpeed backends                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### SageMaker Deployment Code

```python
# sagemaker_lmi_deployment.py
"""Deploy LLM on SageMaker with LMI container."""

import sagemaker
from sagemaker.djl_inference import DJLModel
import boto3
import json


def deploy_llm_endpoint(
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    instance_type: str = "ml.g5.2xlarge",
    endpoint_name: str = "llama-3-8b-vllm",
    initial_instance_count: int = 1,
) -> sagemaker.Predictor:
    """Deploy LLM using SageMaker LMI with vLLM backend."""

    role = sagemaker.get_execution_role()
    sess = sagemaker.Session()

    # Get LMI container image
    image_uri = sagemaker.image_uris.retrieve(
        framework="djl-lmi",
        region=sess.boto_region_name,
        version="0.28.0",  # Use latest stable version
    )

    # Model configuration
    model = DJLModel(
        model_id=model_id,
        role=role,
        image_uri=image_uri,
        env={
            # Backend selection
            "OPTION_ROLLING_BATCH": "vllm",

            # vLLM configuration
            "OPTION_MAX_ROLLING_BATCH_SIZE": "64",
            "OPTION_TENSOR_PARALLEL_DEGREE": "1",
            "OPTION_MAX_MODEL_LEN": "4096",
            "OPTION_GPU_MEMORY_UTILIZATION": "0.95",

            # Performance tuning
            "OPTION_ENABLE_PREFIX_CACHING": "true",
            "OPTION_ENABLE_CHUNKED_PREFILL": "true",

            # Logging
            "OPTION_OUTPUT_FORMATTER": "jsonlines",
        },
    )

    # Deploy endpoint
    predictor = model.deploy(
        instance_type=instance_type,
        initial_instance_count=initial_instance_count,
        endpoint_name=endpoint_name,
        container_startup_health_check_timeout=900,  # 15 minutes for model loading
    )

    return predictor


def configure_autoscaling(
    endpoint_name: str,
    min_capacity: int = 1,
    max_capacity: int = 10,
    target_value: float = 5.0,  # Target concurrent requests per instance
):
    """Configure autoscaling for SageMaker endpoint."""

    client = boto3.client("application-autoscaling")

    # Register scalable target
    client.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=f"endpoint/{endpoint_name}/variant/AllTraffic",
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=min_capacity,
        MaxCapacity=max_capacity,
    )

    # Create scaling policy
    client.put_scaling_policy(
        PolicyName=f"{endpoint_name}-scaling-policy",
        ServiceNamespace="sagemaker",
        ResourceId=f"endpoint/{endpoint_name}/variant/AllTraffic",
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": target_value,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
            },
            "ScaleInCooldown": 300,   # 5 minutes
            "ScaleOutCooldown": 60,   # 1 minute
        },
    )

    print(f"Autoscaling configured: {min_capacity}-{max_capacity} instances")


def invoke_endpoint(
    endpoint_name: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
) -> str:
    """Invoke SageMaker endpoint."""

    runtime = boto3.client("sagemaker-runtime")

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": True,
        },
    }

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    result = json.loads(response["Body"].read().decode())
    return result[0]["generated_text"]


def cleanup_endpoint(endpoint_name: str):
    """Delete SageMaker endpoint and associated resources."""

    sm_client = boto3.client("sagemaker")

    # Delete endpoint
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    print(f"Deleted endpoint: {endpoint_name}")

    # Delete endpoint config
    sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
    print(f"Deleted endpoint config: {endpoint_name}")


# Example usage
if __name__ == "__main__":
    # Deploy
    predictor = deploy_llm_endpoint(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        instance_type="ml.g5.2xlarge",
        endpoint_name="llama-3-8b-demo",
    )

    # Configure autoscaling
    configure_autoscaling(
        endpoint_name="llama-3-8b-demo",
        min_capacity=1,
        max_capacity=5,
    )

    # Test
    response = invoke_endpoint(
        endpoint_name="llama-3-8b-demo",
        prompt="What is machine learning?",
    )
    print(response)
```

### SageMaker CloudFormation Template

```yaml
# cloudformation/sagemaker-endpoint.yaml
AWSTemplateFormatVersion: "2010-09-09"
Description: SageMaker LLM Inference Endpoint

Parameters:
  ModelId:
    Type: String
    Default: meta-llama/Llama-3.1-8B-Instruct
    Description: HuggingFace model ID

  InstanceType:
    Type: String
    Default: ml.g5.2xlarge
    AllowedValues:
      - ml.g5.xlarge
      - ml.g5.2xlarge
      - ml.g5.4xlarge
      - ml.g5.12xlarge
      - ml.p4d.24xlarge
    Description: SageMaker instance type

  InitialInstanceCount:
    Type: Number
    Default: 1
    MinValue: 1
    MaxValue: 10

  MaxInstanceCount:
    Type: Number
    Default: 5
    MinValue: 1
    MaxValue: 20

Resources:
  ExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Effect: Allow
            Principal:
              Service: sagemaker.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
        - arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

  Model:
    Type: AWS::SageMaker::Model
    Properties:
      ModelName: !Sub "${AWS::StackName}-model"
      ExecutionRoleArn: !GetAtt ExecutionRole.Arn
      PrimaryContainer:
        Image: !Sub "763104351884.dkr.ecr.${AWS::Region}.amazonaws.com/djl-inference:0.28.0-lmi"
        Environment:
          HF_MODEL_ID: !Ref ModelId
          OPTION_ROLLING_BATCH: vllm
          OPTION_MAX_ROLLING_BATCH_SIZE: "64"
          OPTION_GPU_MEMORY_UTILIZATION: "0.95"

  EndpointConfig:
    Type: AWS::SageMaker::EndpointConfig
    Properties:
      EndpointConfigName: !Sub "${AWS::StackName}-config"
      ProductionVariants:
        - VariantName: AllTraffic
          ModelName: !GetAtt Model.ModelName
          InstanceType: !Ref InstanceType
          InitialInstanceCount: !Ref InitialInstanceCount
          ContainerStartupHealthCheckTimeoutInSeconds: 900

  Endpoint:
    Type: AWS::SageMaker::Endpoint
    Properties:
      EndpointName: !Sub "${AWS::StackName}-endpoint"
      EndpointConfigName: !GetAtt EndpointConfig.EndpointConfigName

  ScalableTarget:
    Type: AWS::ApplicationAutoScaling::ScalableTarget
    Properties:
      MaxCapacity: !Ref MaxInstanceCount
      MinCapacity: !Ref InitialInstanceCount
      ResourceId: !Sub "endpoint/${Endpoint.EndpointName}/variant/AllTraffic"
      RoleARN: !Sub "arn:aws:iam::${AWS::AccountId}:role/aws-service-role/sagemaker.application-autoscaling.amazonaws.com/AWSServiceRoleForApplicationAutoScaling_SageMakerEndpoint"
      ScalableDimension: sagemaker:variant:DesiredInstanceCount
      ServiceNamespace: sagemaker

  ScalingPolicy:
    Type: AWS::ApplicationAutoScaling::ScalingPolicy
    Properties:
      PolicyName: !Sub "${AWS::StackName}-scaling-policy"
      PolicyType: TargetTrackingScaling
      ScalingTargetId: !Ref ScalableTarget
      TargetTrackingScalingPolicyConfiguration:
        TargetValue: 5.0
        PredefinedMetricSpecification:
          PredefinedMetricType: SageMakerVariantInvocationsPerInstance
        ScaleInCooldown: 300
        ScaleOutCooldown: 60

Outputs:
  EndpointName:
    Value: !GetAtt Endpoint.EndpointName
  EndpointArn:
    Value: !Ref Endpoint
```

---

## Inferentia2 Deployment

### Inferentia2 Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INFERENTIA2 ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   AWS Inferentia2 Chip:                                             │
│   ═════════════════════                                             │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    Inferentia2 Chip                         │   │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │   │
│   │  │NeuronCore│  │NeuronCore│  │NeuronCore│  │NeuronCore│    │   │
│   │  │    0     │  │    1     │  │    2     │  │    3     │    │   │
│   │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │   │
│   │                                                             │   │
│   │  • 32 GB HBM per chip                                       │   │
│   │  • 820 GB/s memory bandwidth                                │   │
│   │  • 190 TFLOPS FP16                                          │   │
│   │  • NeuronLink for multi-chip                                │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   Instance Types:                                                   │
│   ═══════════════                                                   │
│                                                                     │
│   inf2.xlarge    │ 1 chip  │ 32 GB  │ $0.76/hr  │ Small models     │
│   inf2.8xlarge   │ 1 chip  │ 32 GB  │ $1.97/hr  │ More CPU/memory  │
│   inf2.24xlarge  │ 6 chips │ 192 GB │ $6.49/hr  │ Medium models    │
│   inf2.48xlarge  │ 12 chips│ 384 GB │ $12.98/hr │ Large models     │
│                                                                     │
│   Cost Comparison (70B model):                                      │
│   ════════════════════════════                                      │
│   p4d.24xlarge (8× A100): $32.77/hr                                 │
│   inf2.48xlarge (12× Inf2): $12.98/hr  ← 60% cheaper                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Neuron SDK Compilation

```python
# inferentia2_compile.py
"""Compile and deploy model on Inferentia2."""

from optimum.neuron import NeuronModelForCausalLM
from transformers import AutoTokenizer
import torch


def compile_model_for_inferentia(
    model_id: str,
    output_dir: str,
    batch_size: int = 1,
    sequence_length: int = 2048,
    num_cores: int = 2,
    auto_cast_type: str = "bf16",
):
    """Compile HuggingFace model for Inferentia2."""

    print(f"Compiling {model_id} for Inferentia2...")

    # Export and compile model
    model = NeuronModelForCausalLM.from_pretrained(
        model_id,
        export=True,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_cores=num_cores,
        auto_cast_type=auto_cast_type,
    )

    # Save compiled model
    model.save_pretrained(output_dir)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_dir)

    print(f"Model compiled and saved to {output_dir}")
    return output_dir


def run_inference_inferentia(
    model_dir: str,
    prompt: str,
    max_new_tokens: int = 100,
) -> str:
    """Run inference on compiled Inferentia2 model."""

    # Load compiled model
    model = NeuronModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
    )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Compilation options for different models
COMPILATION_CONFIGS = {
    "llama-3-8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "batch_size": 1,
        "sequence_length": 2048,
        "num_cores": 2,
        "auto_cast_type": "bf16",
    },
    "llama-3-8b-throughput": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "batch_size": 4,
        "sequence_length": 2048,
        "num_cores": 4,
        "auto_cast_type": "bf16",
    },
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "batch_size": 1,
        "sequence_length": 4096,
        "num_cores": 2,
        "auto_cast_type": "bf16",
    },
}


if __name__ == "__main__":
    # Compile Llama 3.1 8B
    compile_model_for_inferentia(
        **COMPILATION_CONFIGS["llama-3-8b"],
        output_dir="./llama-3-8b-neuron",
    )

    # Test inference
    response = run_inference_inferentia(
        model_dir="./llama-3-8b-neuron",
        prompt="What is machine learning?",
    )
    print(response)
```

### Inferentia2 Deployment Script

```bash
#!/bin/bash
# deploy_inferentia2.sh

# Launch inf2.xlarge instance
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id ami-0123456789abcdef0 \  # Neuron DLAMI
    --instance-type inf2.xlarge \
    --key-name your-key \
    --security-group-ids sg-xxx \
    --subnet-id subnet-xxx \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Launched: $INSTANCE_ID"

# On the instance:
# Install Neuron SDK
pip install optimum[neuronx]
pip install transformers

# Compile model (one-time)
python -c "
from optimum.neuron import NeuronModelForCausalLM
model = NeuronModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.1-8B-Instruct',
    export=True,
    batch_size=1,
    sequence_length=2048,
    num_cores=2,
)
model.save_pretrained('./llama-3-8b-neuron')
"

# Run inference server
python -m optimum.neuron.server \
    --model ./llama-3-8b-neuron \
    --port 8000
```

---

## Bedrock vs Self-Hosted Comparison

### Decision Matrix

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BEDROCK vs SELF-HOSTED                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Dimension          │ Bedrock           │ Self-Hosted              │
│   ═══════════════════│═══════════════════│══════════════════════    │
│                      │                   │                          │
│   Setup Time         │ Minutes           │ Hours to days            │
│   Operational Burden │ None              │ High                     │
│   Cost (Low Volume)  │ Higher per token  │ Higher fixed cost        │
│   Cost (High Volume) │ Higher per token  │ Lower per token          │
│   Latency            │ ~500ms TTFT       │ ~200ms TTFT (tuned)      │
│   Customization      │ Limited           │ Full control             │
│   Model Selection    │ Bedrock models    │ Any model                │
│   Fine-tuning        │ Limited           │ Full support             │
│   Data Privacy       │ AWS managed       │ Full control             │
│   Scaling            │ Automatic         │ Manual/configured        │
│                      │                   │                          │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   COST COMPARISON (1M tokens/day):                                  │
│   ════════════════════════════════                                  │
│                                                                     │
│   Bedrock (Claude 3 Sonnet):                                        │
│   • Input: $3/M tokens × 0.5M = $1.50/day                           │
│   • Output: $15/M tokens × 0.5M = $7.50/day                         │
│   • Total: ~$9/day = ~$270/month                                    │
│                                                                     │
│   Self-Hosted (Llama 3.1 8B on g5.2xlarge):                         │
│   • Instance: $1.21/hr × 24 × 30 = ~$871/month                      │
│   • Can handle 10M+ tokens/day                                      │
│   • Cost per 1M tokens: ~$2.90                                      │
│                                                                     │
│   Break-even: ~3M tokens/day                                        │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   DECISION GUIDE:                                                   │
│   ═══════════════                                                   │
│                                                                     │
│   USE BEDROCK when:                                                 │
│   • Low volume (< 1M tokens/day)                                    │
│   • Need Claude/Anthropic models                                    │
│   • Want zero operational burden                                    │
│   • Prototyping/experimentation                                     │
│                                                                     │
│   USE SELF-HOSTED when:                                             │
│   • High volume (> 3M tokens/day)                                   │
│   • Need custom/fine-tuned models                                   │
│   • Strict latency requirements (< 300ms TTFT)                      │
│   • Data privacy requirements                                       │
│   • Cost optimization is priority                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Bedrock Integration Example

```python
# bedrock_integration.py
"""AWS Bedrock integration for LLM inference."""

import boto3
import json
from typing import Generator


def invoke_bedrock(
    prompt: str,
    model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
    max_tokens: int = 1000,
    temperature: float = 0.7,
) -> str:
    """Invoke Bedrock model (non-streaming)."""

    client = boto3.client("bedrock-runtime")

    # Format for Claude
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def invoke_bedrock_streaming(
    prompt: str,
    model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
    max_tokens: int = 1000,
    temperature: float = 0.7,
) -> Generator[str, None, None]:
    """Invoke Bedrock model with streaming."""

    client = boto3.client("bedrock-runtime")

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    response = client.invoke_model_with_response_stream(
        modelId=model_id,
        body=json.dumps(body),
    )

    for event in response["body"]:
        chunk = json.loads(event["chunk"]["bytes"])
        if chunk["type"] == "content_block_delta":
            yield chunk["delta"]["text"]


def invoke_bedrock_llama(
    prompt: str,
    model_id: str = "meta.llama3-1-8b-instruct-v1:0",
    max_tokens: int = 1000,
    temperature: float = 0.7,
) -> str:
    """Invoke Bedrock Llama model."""

    client = boto3.client("bedrock-runtime")

    body = {
        "prompt": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "max_gen_len": max_tokens,
        "temperature": temperature,
    }

    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
    )

    result = json.loads(response["body"].read())
    return result["generation"]


# Example usage
if __name__ == "__main__":
    # Non-streaming
    response = invoke_bedrock("What is machine learning?")
    print(response)

    # Streaming
    print("\nStreaming response:")
    for chunk in invoke_bedrock_streaming("Explain quantum computing."):
        print(chunk, end="", flush=True)
```

---

## AWS Architecture Patterns

### Pattern 1: Simple (Bedrock)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PATTERN 1: BEDROCK SIMPLE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐     ┌─────────────┐     ┌──────────┐                 │
│   │  Client  │────▶│ API Gateway │────▶│  Lambda  │                 │
│   └──────────┘     └─────────────┘     └────┬─────┘                 │
│                                             │                       │
│                                             ▼                       │
│                                        ┌──────────┐                 │
│                                        │ Bedrock  │                 │
│                                        └──────────┘                 │
│                                                                     │
│   Pros: Simple, no infrastructure, auto-scaling                     │
│   Cons: Higher cost at scale, limited model selection               │
│   Best for: Prototypes, low volume, Claude models                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Pattern 2: SageMaker Production

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PATTERN 2: SAGEMAKER PRODUCTION                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐     ┌─────────────┐     ┌─────────────────────────┐  │
│   │  Client  │────▶│     ALB     │────▶│   SageMaker Endpoint    │  │
│   └──────────┘     └─────────────┘     │  ┌─────────────────────┐│  │
│                                        │  │   Auto Scaling      ││  │
│                                        │  │  ┌─────┐ ┌─────┐    ││  │
│                                        │  │  │ g5  │ │ g5  │... ││  │
│                                        │  │  │ LMI │ │ LMI │    ││  │
│                                        │  │  └─────┘ └─────┘    ││  │
│                                        │  └─────────────────────┘│  │
│                                        └─────────────────────────┘  │
│                                                                     │
│   Pros: Managed, auto-scaling, monitoring, A/B testing              │
│   Cons: Higher cost than EC2, less control                          │
│   Best for: Production workloads, teams without infra expertise     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Pattern 3: EKS + KServe

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PATTERN 3: EKS + KSERVE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐     ┌─────────────┐     ┌─────────────────────────┐  │
│   │  Client  │────▶│     ALB     │────▶│      EKS Cluster        │  │
│   └──────────┘     └─────────────┘     │  ┌─────────────────────┐│  │
│                                        │  │   Istio Ingress     ││  │
│                                        │  └──────────┬──────────┘│  │
│                                        │             │           │  │
│                                        │  ┌──────────▼──────────┐│  │
│                                        │  │      KServe         ││  │
│                                        │  │  ┌─────┐ ┌─────┐    ││  │
│                                        │  │  │vLLM │ │vLLM │... ││  │
│                                        │  │  │ Pod │ │ Pod │    ││  │
│                                        │  │  └─────┘ └─────┘    ││  │
│                                        │  └─────────────────────┘│  │
│                                        │                         │  │
│                                        │  ┌─────────────────────┐│  │
│                                        │  │    Karpenter        ││  │
│                                        │  │  (GPU Node Scaling) ││  │
│                                        │  └─────────────────────┘│  │
│                                        └─────────────────────────┘  │
│                                                                     │
│   Pros: Full control, multi-model, canary deployments               │
│   Cons: Complex setup, requires K8s expertise                       │
│   Best for: Large teams, multi-model serving, advanced routing      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Pattern 4: Cost-Optimized (Inferentia2)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PATTERN 4: INFERENTIA2 COST-OPT                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐     ┌─────────────┐     ┌─────────────────────────┐  │
│   │  Client  │────▶│     ALB     │────▶│   SageMaker Endpoint    │  │
│   └──────────┘     └─────────────┘     │  ┌─────────────────────┐│  │
│                                        │  │   Auto Scaling      ││  │
│                                        │  │  ┌─────┐ ┌─────┐    ││  │
│                                        │  │  │inf2 │ │inf2 │... ││  │
│                                        │  │  │Neuron│ │Neuron│   ││  │
│                                        │  │  └─────┘ └─────┘    ││  │
│                                        │  └─────────────────────┘│  │
│                                        └─────────────────────────┘  │
│                                                                     │
│   Pros: 60% cost savings vs GPU, good throughput                    │
│   Cons: Compilation required, less flexible, longer cold start      │
│   Best for: High-volume, cost-sensitive, stable models              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Cost Optimization Strategies

### Spot Instances for Development

```python
# spot_instance_deployment.py
"""Deploy vLLM on Spot instances for cost savings."""

import boto3


def launch_spot_instance(
    instance_type: str = "g5.2xlarge",
    max_price: str = "0.50",  # 50% of on-demand
) -> str:
    """Launch a Spot instance for vLLM."""

    ec2 = boto3.client("ec2")

    response = ec2.request_spot_instances(
        InstanceCount=1,
        Type="one-time",
        SpotPrice=max_price,
        LaunchSpecification={
            "ImageId": "ami-0123456789abcdef0",  # Deep Learning AMI
            "InstanceType": instance_type,
            "KeyName": "your-key",
            "SecurityGroupIds": ["sg-xxx"],
            "SubnetId": "subnet-xxx",
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "VolumeSize": 200,
                        "VolumeType": "gp3",
                    },
                }
            ],
        },
    )

    return response["SpotInstanceRequests"][0]["SpotInstanceRequestId"]


# Spot pricing comparison
SPOT_SAVINGS = {
    "g5.xlarge": {"on_demand": 1.01, "spot_avg": 0.35, "savings": "65%"},
    "g5.2xlarge": {"on_demand": 1.21, "spot_avg": 0.42, "savings": "65%"},
    "g5.12xlarge": {"on_demand": 5.67, "spot_avg": 1.98, "savings": "65%"},
    "p4d.24xlarge": {"on_demand": 32.77, "spot_avg": 11.47, "savings": "65%"},
}
```

### Reserved Capacity Planning

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RESERVED CAPACITY PLANNING                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Commitment Options:                                               │
│   ═══════════════════                                               │
│                                                                     │
│   1-Year Reserved:     ~30% savings vs on-demand                    │
│   3-Year Reserved:     ~50% savings vs on-demand                    │
│   Savings Plans:       ~30% savings, more flexible                  │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   Example: g5.2xlarge for production                                │
│                                                                     │
│   On-Demand:    $1.21/hr × 24 × 365 = $10,600/year                  │
│   1-Year RI:    $0.85/hr × 24 × 365 = $7,446/year  (30% savings)    │
│   3-Year RI:    $0.61/hr × 24 × 365 = $5,344/year  (50% savings)    │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   Recommendation:                                                   │
│   • Dev/Test: Spot instances (65% savings)                          │
│   • Production baseline: Reserved (30-50% savings)                  │
│   • Production burst: On-demand or Spot                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **Match instance to model size** - Use VRAM calculations to select appropriate instance

2. **SageMaker for managed** - Best for teams without dedicated infra expertise

3. **EKS for control** - Best for advanced routing, multi-model, canary deployments

4. **Inferentia2 for cost** - 60% savings but requires compilation

5. **Bedrock for simplicity** - Zero ops but higher per-token cost

6. **Spot for dev/test** - 65% savings for non-production workloads

7. **Reserved for production** - 30-50% savings for predictable workloads

---

## Lab Preview: SageMaker Production Deployment

In Lab 9, you will:

- Deploy Llama 3.1 8B on SageMaker with LMI
- Configure autoscaling policies
- Set up CloudWatch monitoring
- Test endpoint performance
- Clean up resources

---

## References

1. [AWS SageMaker LMI Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference.html)
2. [AWS Neuron SDK Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
3. [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
4. [EC2 GPU Instance Types](https://aws.amazon.com/ec2/instance-types/)
5. [SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)
