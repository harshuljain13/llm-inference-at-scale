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


## Advanced AWS Inference Patterns

### Disaggregated Inference with llm-d on AWS

AWS officially supports **llm-d** (April 2026) for disaggregated inference — separating prefill and decode into independent, independently-scalable pools connected via high-speed KV cache transfer over EFA.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    llm-d DISAGGREGATED INFERENCE                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐     ┌─────────────┐                                  │
│   │  Client  │────▶│  Router /   │                                  │
│   └──────────┘     │  Scheduler  │                                  │
│                    └──────┬──────┘                                  │
│                           │                                         │
│              ┌────────────┴────────────┐                            │
│              ▼                         ▼                            │
│   ┌─────────────────────┐   ┌─────────────────────┐                │
│   │   PREFILL POOL      │   │   DECODE POOL       │                │
│   │  (compute-bound)    │   │  (memory-bound)     │                │
│   │                     │   │                     │                │
│   │  • p5.48xlarge      │   │  • inf2.48xlarge    │                │
│   │  • High FLOPS       │   │  • High mem BW      │                │
│   │  • Batch prefills   │   │  • Token-by-token   │                │
│   │  • Scale on queue   │   │  • Scale on active  │                │
│   └──────────┬──────────┘   └──────────▲──────────┘                │
│              │                         │                            │
│              │    KV Cache Transfer    │                            │
│              └────────────────────────▶│                            │
│                   via EFA (400 Gbps)                                │
│                                                                     │
│   Benefits:                                                         │
│   • Prefill pool uses compute-optimized GPUs (H100)                 │
│   • Decode pool uses memory-BW-optimized chips (Inferentia2)        │
│   • Each pool scales independently based on its bottleneck          │
│   • KV cache transfer via EFA avoids recomputation                  │
│   • 40-60% cost reduction vs monolithic serving at scale            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key configuration:**

```yaml
# llm-d deployment config (EKS-based)
apiVersion: llm-d.aws/v1alpha1
kind: InferencePool
metadata:
  name: llama-70b-disaggregated
spec:
  model: meta-llama/Llama-3.1-70B-Instruct
  prefillPool:
    instanceType: p5.48xlarge
    replicas: 2
    maxBatchSize: 32
    maxSequenceLength: 8192
  decodePool:
    instanceType: inf2.48xlarge
    replicas: 4
    maxConcurrentSequences: 256
  kvCacheTransfer:
    transport: efa          # 400 Gbps RDMA
    compression: none       # Lossless for accuracy
    maxCacheSizeGB: 64
  router:
    strategy: least-kv-pending
    prefillQueueThreshold: 16
```

---

### EAGLE-Based Adaptive Speculative Decoding on SageMaker

SageMaker LMI natively supports **EAGLE** (Extrapolation Algorithm for Greater Language-model Efficiency) — an adaptive speculative decoding method that uses a lightweight draft head trained on the target model's hidden states rather than a separate draft model.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EAGLE SPECULATIVE DECODING                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Traditional Speculative Decoding:                                 │
│   Draft Model (1.3B) → Speculate K tokens → Verify with 70B        │
│   Problem: Draft model quality varies, fixed speculation length     │
│                                                                     │
│   EAGLE Approach:                                                   │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  Target Model (70B)                                         │   │
│   │  ┌─────────────────────────────────────────────────────┐    │   │
│   │  │  Hidden States (layer N-1)                          │    │   │
│   │  └────────────────────┬────────────────────────────────┘    │   │
│   │                       │                                     │   │
│   │                       ▼                                     │   │
│   │  ┌─────────────────────────────────────────────────────┐    │   │
│   │  │  EAGLE Draft Head (~0.5B params, trained on target) │    │   │
│   │  │  • Extrapolates next hidden states                  │    │   │
│   │  │  • Tree-structured speculation (not linear)         │    │   │
│   │  │  • Adaptive depth: 2-8 tokens based on confidence   │    │   │
│   │  └────────────────────┬────────────────────────────────┘    │   │
│   │                       │                                     │   │
│   │                       ▼                                     │   │
│   │  Verify all candidates in single forward pass               │   │
│   │  Accept rate: 70-85% (vs 50-60% for separate draft model)  │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   Speedup: 2.5-3.5× for greedy, 2-2.8× for sampling                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**SageMaker LMI deployment with EAGLE:**

```python
# sagemaker_eagle_deployment.py
"""Deploy LLM with EAGLE speculative decoding on SageMaker."""

from sagemaker.djl_inference import DJLModel
import sagemaker

def deploy_eagle_endpoint(
    model_id: str = "meta-llama/Llama-3.1-70B-Instruct",
    eagle_model_id: str = "yuhuili/EAGLE-LLaMA3-Instruct-70B",
    instance_type: str = "ml.p4d.24xlarge",
    endpoint_name: str = "llama-70b-eagle",
):
    """Deploy with EAGLE speculative decoding enabled."""

    role = sagemaker.get_execution_role()
    image_uri = sagemaker.image_uris.retrieve(
        framework="djl-lmi", region="us-east-1", version="0.29.0"
    )

    model = DJLModel(
        model_id=model_id,
        role=role,
        image_uri=image_uri,
        env={
            "OPTION_ROLLING_BATCH": "vllm",
            "OPTION_TENSOR_PARALLEL_DEGREE": "8",
            "OPTION_MAX_MODEL_LEN": "4096",
            "OPTION_GPU_MEMORY_UTILIZATION": "0.92",
            # EAGLE configuration
            "OPTION_SPECULATIVE_MODEL": eagle_model_id,
            "OPTION_SPECULATIVE_METHOD": "eagle",
            "OPTION_NUM_SPECULATIVE_TOKENS": "5",
            "OPTION_SPECULATIVE_DISABLE_BY_BATCH_SIZE": "8",
        },
    )

    return model.deploy(
        instance_type=instance_type,
        initial_instance_count=1,
        endpoint_name=endpoint_name,
        container_startup_health_check_timeout=1200,
    )
```

> **Note:** EAGLE automatically disables speculation when batch size exceeds the threshold (default 8), since the verification overhead outweighs the latency benefit at high concurrency.

---

### Speculative Decoding on Trainium/Inferentia2 with vLLM

vLLM's Neuron backend (v0.6+) supports speculative decoding on Trainium and Inferentia2, enabling latency reduction on cost-optimized silicon. The Neuron compiler handles draft/target model co-location across NeuronCores.

```
┌─────────────────────────────────────────────────────────────────────┐
│            SPECULATIVE DECODING ON NEURON DEVICES                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   inf2.24xlarge (6 chips × 4 cores = 24 NeuronCores)               │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  Chip 0-1 (8 cores)         │  Chip 2-5 (16 cores)         │   │
│   │  ┌───────────────────────┐  │  ┌───────────────────────┐   │   │
│   │  │  Draft Model (8B)     │  │  │  Target Model (70B)   │   │   │
│   │  │  • 2 NeuronCores      │  │  │  • 16 NeuronCores     │   │   │
│   │  │  • Speculate K=5      │  │  │  • Verify + accept    │   │   │
│   │  │  • ~5ms per draft     │  │  │  • ~40ms per verify   │   │   │
│   │  └───────────────────────┘  │  └───────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   Without speculation: 70B decode = ~80ms/token                     │
│   With speculation:    5 drafts + 1 verify = ~45ms → accept ~3.5    │
│   Effective: ~13ms/token (6× improvement in tokens/sec)             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Deployment with vLLM on Neuron:**

```bash
# Launch vLLM with speculative decoding on inf2
docker run -d --device=/dev/neuron0 --device=/dev/neuron1 \
    --device=/dev/neuron2 --device=/dev/neuron3 \
    --device=/dev/neuron4 --device=/dev/neuron5 \
    -p 8000:8000 \
    vllm/vllm-neuron:latest \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.1-8B-Instruct \
    --num-speculative-tokens 5 \
    --device neuron \
    --tensor-parallel-size 16 \
    --speculative-draft-tensor-parallel-size 2 \
    --max-num-seqs 32 \
    --block-size 8
```

> **Constraint:** Both draft and target models must be compiled with matching sequence lengths. Recompilation is required when changing speculation depth or batch size.

---

### Case Study: Amazon Rufus — Multi-Node Trainium Inference at Scale

Amazon Rufus (the AI shopping assistant) runs multi-node Trainium inference to serve hundreds of millions of queries, demonstrating production-grade LLM serving on custom silicon at Prime Day scale.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AMAZON RUFUS ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Scale: Hundreds of millions of queries/day (Prime Day 2025)       │
│   Result: 2× inference speed vs prior GPU-based deployment          │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    Inference Cluster                         │   │
│   │                                                             │   │
│   │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │   │
│   │   │ trn1.32 │  │ trn1.32 │  │ trn1.32 │  │ trn1.32 │      │   │
│   │   │ 16 chips│  │ 16 chips│  │ 16 chips│  │ 16 chips│      │   │
│   │   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘      │   │
│   │        │             │             │             │          │   │
│   │        └─────────────┴──────┬──────┴─────────────┘          │   │
│   │                             │                               │   │
│   │                    EFA Fabric (400 Gbps)                     │   │
│   │                    NeuronLink cross-node                     │   │
│   │                                                             │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   Key Design Decisions:                                             │
│   • Multi-node tensor parallelism across trn1.32xlarge instances    │
│   • EFA for all-reduce and KV cache synchronization                 │
│   • Neuron compiler optimizations: operator fusion, layout xforms   │
│   • Dynamic batching tuned for shopping query latency SLAs          │
│   • Graceful degradation: automatic fallback to smaller model       │
│     under extreme load (Prime Day traffic spikes)                   │
│                                                                     │
│   Results:                                                          │
│   • 2× inference throughput vs equivalent GPU deployment            │
│   • 50%+ cost reduction per query                                   │
│   • Sub-200ms P99 latency for product recommendations              │
│   • Zero downtime during Prime Day 2025 traffic surge               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Lessons for practitioners:**

1. **Compile once, serve forever** — Rufus pre-compiles model graphs for fixed batch/sequence configs, eliminating JIT overhead
2. **Multi-node TP over EFA** — For models exceeding single-node memory, EFA provides near-local bandwidth for tensor parallel communication
3. **Heterogeneous fallback** — Under extreme load, route overflow traffic to a smaller quantized model rather than dropping requests
4. **Warmup pools** — Pre-warmed Trainium instances eliminate cold-start latency during traffic spikes

---

### Multi-LoRA Serving on SageMaker

SageMaker LMI supports serving **dozens of LoRA adapters** on a single endpoint, sharing the base model weights while dynamically loading task-specific adapters per request. This eliminates the need for separate endpoints per fine-tuned model.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-LORA ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Single SageMaker Endpoint (ml.g5.12xlarge)                        │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  Base Model: Llama 3.1 8B (shared, loaded once)             │   │
│   │  ═══════════════════════════════════════════════             │   │
│   │                                                             │   │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │   │
│   │  │ LoRA #1  │ │ LoRA #2  │ │ LoRA #3  │ │ LoRA #N  │       │   │
│   │  │ Customer │ │ Medical  │ │ Legal    │ │  ...     │       │   │
│   │  │ Support  │ │ Summary  │ │ Extract  │ │          │       │   │
│   │  │ ~16 MB   │ │ ~16 MB   │ │ ~16 MB   │ │ ~16 MB   │       │   │
│   │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │   │
│   │                                                             │   │
│   │  Adapter selection: per-request header                       │   │
│   │  Hot-swap latency: < 1ms (adapters cached in GPU memory)     │   │
│   │  Max adapters: limited by GPU memory (~50-100 for rank-16)   │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   Cost Impact:                                                      │
│   • Without Multi-LoRA: 20 models × $1.21/hr = $24.20/hr           │
│   • With Multi-LoRA:    1 endpoint × $5.67/hr = $5.67/hr           │
│   • Savings: 77%                                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Deployment configuration:**

```python
# multi_lora_sagemaker.py
"""Deploy multi-LoRA endpoint on SageMaker."""

from sagemaker.djl_inference import DJLModel
import sagemaker
import json
import boto3


def deploy_multi_lora_endpoint(
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    lora_adapters: dict = None,
    instance_type: str = "ml.g5.12xlarge",
    endpoint_name: str = "multi-lora-endpoint",
):
    """Deploy base model with multiple LoRA adapters."""

    if lora_adapters is None:
        lora_adapters = {
            "customer-support": "s3://my-bucket/loras/customer-support/",
            "medical-summary": "s3://my-bucket/loras/medical-summary/",
            "legal-extraction": "s3://my-bucket/loras/legal-extraction/",
        }

    role = sagemaker.get_execution_role()
    image_uri = sagemaker.image_uris.retrieve(
        framework="djl-lmi", region="us-east-1", version="0.29.0"
    )

    model = DJLModel(
        model_id=base_model,
        role=role,
        image_uri=image_uri,
        env={
            "OPTION_ROLLING_BATCH": "vllm",
            "OPTION_TENSOR_PARALLEL_DEGREE": "4",
            "OPTION_GPU_MEMORY_UTILIZATION": "0.90",
            "OPTION_MAX_LORAS": "32",
            "OPTION_MAX_LORA_RANK": "16",
            "OPTION_ENABLE_LORA": "true",
            "OPTION_MAX_CPU_LORAS": "64",  # Cache more on CPU
        },
    )

    return model.deploy(
        instance_type=instance_type,
        initial_instance_count=1,
        endpoint_name=endpoint_name,
        container_startup_health_check_timeout=900,
    )


def invoke_with_lora(endpoint_name: str, prompt: str, adapter_name: str):
    """Invoke endpoint with a specific LoRA adapter."""

    runtime = boto3.client("sagemaker-runtime")

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "adapter_name": adapter_name,  # Select LoRA per request
        },
    }

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    return json.loads(response["Body"].read().decode())
```

---

### Capacity-Aware Inference with Automatic Instance Fallback

Production LLM deployments must handle GPU capacity constraints gracefully. A capacity-aware routing layer automatically falls back across instance types when primary capacity is unavailable — critical during GPU shortages or regional outages.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CAPACITY-AWARE FALLBACK                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Request → Router                                                  │
│              │                                                      │
│              ├─── Try: p5.48xlarge (H100, lowest latency)           │
│              │    └── Available? ✓ → Route here                     │
│              │                   ✗ ↓                                │
│              ├─── Try: p4d.24xlarge (A100, good perf)               │
│              │    └── Available? ✓ → Route here                     │
│              │                   ✗ ↓                                │
│              ├─── Try: g5.48xlarge (A10G, acceptable)               │
│              │    └── Available? ✓ → Route here (quantized model)   │
│              │                   ✗ ↓                                │
│              └─── Try: inf2.48xlarge (Inferentia2, cost-opt)        │
│                   └── Available? ✓ → Route here (compiled model)   │
│                                  ✗ → Queue + alert                 │
│                                                                     │
│   Each tier has pre-deployed model variants:                        │
│   • p5/p4d: Full precision, max batch size                          │
│   • g5: AWQ INT4 quantized, reduced batch                           │
│   • inf2: Neuron-compiled, fixed batch/seq                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Implementation with SageMaker inference components:**

```python
# capacity_aware_routing.py
"""Capacity-aware inference with multi-instance fallback."""

import boto3
import json
from typing import Optional


class CapacityAwareRouter:
    """Routes inference requests across instance tiers with fallback."""

    def __init__(self, endpoint_name: str):
        self.runtime = boto3.client("sagemaker-runtime")
        self.endpoint_name = endpoint_name
        # Ordered by preference (best performance first)
        self.variants = [
            "h100-primary",
            "a100-fallback",
            "a10g-quantized",
            "inf2-compiled",
        ]

    def invoke(self, prompt: str, max_tokens: int = 256) -> Optional[str]:
        """Try each variant in priority order until one succeeds."""

        payload = json.dumps({
            "inputs": prompt,
            "parameters": {"max_new_tokens": max_tokens},
        })

        for variant in self.variants:
            try:
                response = self.runtime.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType="application/json",
                    Body=payload,
                    TargetVariant=variant,
                )
                result = json.loads(response["Body"].read().decode())
                return result[0]["generated_text"]
            except self.runtime.exceptions.ModelError:
                continue  # Variant overloaded, try next
            except Exception:
                continue  # Capacity unavailable, try next

        return None  # All variants exhausted


# SageMaker endpoint config with multiple variants
ENDPOINT_CONFIG = {
    "EndpointConfigName": "capacity-aware-config",
    "ProductionVariants": [
        {
            "VariantName": "h100-primary",
            "ModelName": "llama-70b-fp16",
            "InstanceType": "ml.p5.48xlarge",
            "InitialInstanceCount": 2,
            "InitialVariantWeight": 80,
        },
        {
            "VariantName": "a100-fallback",
            "ModelName": "llama-70b-fp16",
            "InstanceType": "ml.p4d.24xlarge",
            "InitialInstanceCount": 1,
            "InitialVariantWeight": 15,
        },
        {
            "VariantName": "a10g-quantized",
            "ModelName": "llama-70b-awq-int4",
            "InstanceType": "ml.g5.48xlarge",
            "InitialInstanceCount": 1,
            "InitialVariantWeight": 4,
        },
        {
            "VariantName": "inf2-compiled",
            "ModelName": "llama-70b-neuron",
            "InstanceType": "ml.inf2.48xlarge",
            "InitialInstanceCount": 1,
            "InitialVariantWeight": 1,
        },
    ],
}
```

> **Production tip:** Combine capacity-aware routing with SageMaker's built-in auto-scaling. Set aggressive scale-out on the primary variant and conservative scale-out on fallback variants. Use CloudWatch alarms on `Invocation4XXErrors` to detect capacity exhaustion early.

---

## Key Takeaways

1. **Match instance to model size** - Use VRAM calculations to select appropriate instance

2. **SageMaker for managed** - Best for teams without dedicated infra expertise

3. **EKS for control** - Best for advanced routing, multi-model, canary deployments

4. **Inferentia2 for cost** - 60% savings but requires compilation

5. **Bedrock for simplicity** - Zero ops but higher per-token cost

6. **Spot for dev/test** - 65% savings for non-production workloads

7. **Reserved for production** - 30-50% savings for predictable workloads

8. **Disaggregated inference (llm-d)** - Separate prefill/decode pools for independent scaling, 40-60% cost reduction

9. **EAGLE speculative decoding** - 2.5-3.5× speedup with native SageMaker LMI support

10. **Multi-LoRA serving** - Dozens of fine-tuned models on one endpoint, 77% cost savings vs separate endpoints

11. **Capacity-aware routing** - Automatic fallback across instance tiers for resilience during GPU shortages

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
6. [llm-d: Disaggregated Serving for LLMs](https://github.com/llm-d/llm-d)
7. [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
8. [vLLM on AWS Neuron](https://docs.aws.amazon.com/neuron/latest/frameworks/vllm-index.html)
9. [Amazon Rufus — AI Shopping Assistant](https://aws.amazon.com/blogs/machine-learning/how-amazon-rufus-uses-trainium/)
10. [SageMaker Multi-LoRA Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-lora.html)
