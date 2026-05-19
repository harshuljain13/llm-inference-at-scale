# Pre-Workshop Setup Guide

> Complete this setup before the workshop to ensure a smooth experience

---

## Quick Checklist

- [ ] AWS account with GPU instance access
- [ ] HuggingFace account and access token
- [ ] Python 3.10+ installed
- [ ] Git installed
- [ ] Workshop repository cloned
- [ ] (Optional) Local GPU with CUDA 12.1+

---

## 1. AWS Account Setup

### 1.1 Account Requirements

You need an AWS account with the following:

- Ability to launch GPU instances (g5, p4d)
- SageMaker access
- Service quota for GPU instances

### 1.2 Request GPU Instance Quotas

GPU instances have default quotas of 0. Request increases:

1. Go to **Service Quotas** in AWS Console
2. Search for "Amazon EC2"
3. Request quota increases for:
   - `Running On-Demand G and VT instances` → 48 vCPUs minimum
   - `Running On-Demand P instances` → 96 vCPUs (for p4d labs)

**Note:** Quota increases can take 24-48 hours. Request early!

### 1.3 Create IAM User/Role

For the workshop, you need permissions for:

- EC2 (launch, terminate instances)
- SageMaker (create endpoints, models)
- S3 (read/write model artifacts)
- CloudWatch (view metrics)

Recommended: Use `PowerUserAccess` managed policy for the workshop.

### 1.4 Configure AWS CLI

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
# Enter your Access Key ID, Secret Access Key, and region (us-east-1 recommended)

# Verify
aws sts get-caller-identity
```

---

## 2. HuggingFace Setup

### 2.1 Create Account

1. Go to https://huggingface.co/join
2. Create an account
3. Verify your email

### 2.2 Accept Model Licenses

Some models require accepting a license:

1. Go to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Click "Agree and access repository"
3. Repeat for other models you plan to use

### 2.3 Create Access Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: `workshop-token`
4. Type: `Read`
5. Click "Generate"
6. Copy and save the token securely

### 2.4 Configure Token

```bash
# Option 1: Environment variable
export HF_TOKEN="hf_your_token_here"

# Option 2: HuggingFace CLI
pip install huggingface_hub
huggingface-cli login
# Paste your token when prompted

# Verify
python -c "from huggingface_hub import whoami; print(whoami())"
```

---

## 3. Python Environment

### 3.1 Install Python 3.10+

```bash
# Check version
python --version  # Should be 3.10 or higher

# macOS with Homebrew
brew install python@3.11

# Ubuntu
sudo apt update
sudo apt install python3.11 python3.11-venv
```

### 3.2 Create Virtual Environment

```bash
# Create environment
python -m venv llm-workshop
source llm-workshop/bin/activate  # Linux/macOS
# or
.\llm-workshop\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip
```

### 3.3 Install Core Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate
pip install jupyter notebook
pip install matplotlib pandas numpy
```

### 3.4 Install Inference Engines (GPU Required)

```bash
# vLLM
pip install vllm

# SGLang
pip install sglang[all]

# Verify installation
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

---

## 4. Clone Workshop Repository

```bash
# Clone the repository
git clone https://github.com/your-org/llm-inference-workshop.git
cd llm-inference-workshop

# Install workshop dependencies
pip install -r requirements.txt
```

---

## 5. Verify Setup

### 5.1 Run Verification Script

```bash
python scripts/verify_setup.py
```

Expected output:

```
✓ Python 3.11.0
✓ PyTorch 2.2.0 with CUDA 12.1
✓ Transformers 4.40.0
✓ vLLM 0.4.0
✓ HuggingFace token configured
✓ AWS credentials configured
✓ All checks passed!
```

### 5.2 Manual Verification

```python
# Test PyTorch + CUDA
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Test HuggingFace
from huggingface_hub import whoami
print(f"HuggingFace user: {whoami()['name']}")

# Test AWS
import boto3
sts = boto3.client('sts')
print(f"AWS Account: {sts.get_caller_identity()['Account']}")
```

---

## 6. Pre-Download Models (Optional)

To save time during the workshop, pre-download models:

```bash
# Download Llama 3.1 8B (requires HF token and license acceptance)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = 'meta-llama/Llama-3.1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id)
# This downloads the model weights
print('Model downloaded successfully')
"

# Alternative: Use huggingface-cli
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
```

---

## 7. Launch Test Instance (Optional)

Test your AWS setup by launching a GPU instance:

```bash
# Using CloudFormation template
aws cloudformation create-stack \
    --stack-name workshop-test \
    --template-body file://labs/infrastructure/cloudformation/g5-instance.yaml \
    --parameters ParameterKey=InstanceType,ParameterValue=g5.xlarge

# Wait for stack creation
aws cloudformation wait stack-create-complete --stack-name workshop-test

# Get instance IP
aws cloudformation describe-stacks --stack-name workshop-test \
    --query 'Stacks[0].Outputs[?OutputKey==`PublicIP`].OutputValue' --output text

# Clean up after testing
aws cloudformation delete-stack --stack-name workshop-test
```

---

## Troubleshooting

### CUDA Not Found

```bash
# Check NVIDIA driver
nvidia-smi

# If not installed, install CUDA toolkit
# Ubuntu
sudo apt install nvidia-cuda-toolkit

# Verify
nvcc --version
```

### HuggingFace Token Issues

```bash
# Clear cached token
rm -rf ~/.cache/huggingface/token

# Re-login
huggingface-cli login
```

### AWS Quota Errors

```
Error: You have requested more vCPU capacity than your current vCPU limit
```

Solution: Request quota increase in AWS Console → Service Quotas

### vLLM Installation Fails

```bash
# Try installing from source
pip uninstall vllm
pip install vllm --no-cache-dir

# Or use Docker
docker pull vllm/vllm-openai:latest
```

---

## Support

If you encounter issues during setup:

1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Post in the workshop Slack channel
4. Email: workshop-support@example.com

---

## What to Bring

- Laptop with terminal access
- AWS credentials configured
- HuggingFace token ready
- Curiosity and questions!
