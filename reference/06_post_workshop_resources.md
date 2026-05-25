# Post-Workshop Resources

> Continue your LLM inference learning journey

---

## Continued Learning Paths

### Path 1: Deep Dive into vLLM

1. **Read the vLLM paper**: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
2. **Explore vLLM internals**: [Anatomy of vLLM](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
3. **Join the community**: [vLLM Discord](https://discord.gg/vllm)
4. **Contribute**: [vLLM GitHub](https://github.com/vllm-project/vllm)

### Path 2: Production ML Systems

1. **Ray ecosystem**: [Ray Documentation](https://docs.ray.io/)
2. **Kubernetes for ML**: [KServe Documentation](https://kserve.github.io/website/)
3. **MLOps practices**: [Made With ML](https://madewithml.com/)

### Path 3: Model Optimization

1. **Quantization deep dive**: [HuggingFace Quantization Guide](https://huggingface.co/docs/transformers/quantization)
2. **FlashAttention papers**: [FlashAttention-2](https://arxiv.org/abs/2307.08691)
3. **Speculative decoding**: [Fast Inference from Transformers](https://arxiv.org/abs/2211.17192)

---

## Essential Papers

### Foundational

| Paper                     | Topic                      | Link                                                         |
| ------------------------- | -------------------------- | ------------------------------------------------------------ |
| Attention Is All You Need | Transformer architecture   | [arXiv](https://arxiv.org/abs/1706.03762)                    |
| GPT-2                     | Language model pretraining | [OpenAI](https://openai.com/research/better-language-models) |
| Llama 2                   | Open-source LLM            | [arXiv](https://arxiv.org/abs/2307.09288)                    |

### Inference Optimization

| Paper                 | Topic                  | Link                                      |
| --------------------- | ---------------------- | ----------------------------------------- |
| PagedAttention (vLLM) | Memory management      | [arXiv](https://arxiv.org/abs/2309.06180) |
| FlashAttention        | Efficient attention    | [arXiv](https://arxiv.org/abs/2205.14135) |
| FlashAttention-2      | Improved attention     | [arXiv](https://arxiv.org/abs/2307.08691) |
| SGLang                | Structured generation  | [arXiv](https://arxiv.org/abs/2312.07104) |
| Speculative Decoding  | Fast inference         | [arXiv](https://arxiv.org/abs/2211.17192) |
| Medusa                | Multi-head speculation | [arXiv](https://arxiv.org/abs/2401.10774) |

### Quantization

| Paper | Topic                         | Link                                      |
| ----- | ----------------------------- | ----------------------------------------- |
| GPTQ  | Post-training quantization    | [arXiv](https://arxiv.org/abs/2210.17323) |
| AWQ   | Activation-aware quantization | [arXiv](https://arxiv.org/abs/2306.00978) |
| QLoRA | Efficient fine-tuning         | [arXiv](https://arxiv.org/abs/2305.14314) |

---

## Community Resources

### Discord Servers

- **vLLM**: Technical discussions, troubleshooting
- **Hugging Face**: Model sharing, transformers library
- **MLOps Community**: Production ML systems
- **NVIDIA Developer**: GPU optimization

### GitHub Repositories

```
Essential repos to star:

vllm-project/vllm          - Main inference engine
sgl-project/sglang         - Structured generation
ggerganov/llama.cpp        - Edge deployment
huggingface/transformers   - Model library
ray-project/ray            - Distributed computing
```

### Blogs and Newsletters

- [vLLM Blog](https://blog.vllm.ai/)
- [Hugging Face Blog](https://huggingface.co/blog)
- [The Gradient](https://thegradient.pub/)
- [Sebastian Raschka's Newsletter](https://magazine.sebastianraschka.com/)

---

## AWS-Specific Resources

### Documentation

- [SageMaker LMI](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference.html)
- [Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/)
- [Bedrock](https://docs.aws.amazon.com/bedrock/)
- [EC2 GPU Instances](https://aws.amazon.com/ec2/instance-types/)

### Workshops and Tutorials

- [AWS ML Workshop](https://catalog.workshops.aws/ml-on-aws)
- [SageMaker Examples](https://github.com/aws/amazon-sagemaker-examples)
- [Neuron Samples](https://github.com/aws-neuron/aws-neuron-samples)

### Cost Optimization

- [Spot Instance Advisor](https://aws.amazon.com/ec2/spot/instance-advisor/)
- [AWS Pricing Calculator](https://calculator.aws/)
- [Savings Plans](https://aws.amazon.com/savingsplans/)

---

## Tools and Utilities

### Benchmarking

```bash
# vLLM benchmark
python -m vllm.entrypoints.openai.api_server --model ... &
python benchmark_serving.py --backend vllm

# llm-perf
pip install llm-perf
llm-perf benchmark --model meta-llama/Llama-3.1-8B-Instruct

# locust for load testing
pip install locust
locust -f locustfile.py --host http://localhost:8000
```

### Monitoring

- **Prometheus + Grafana**: Metrics visualization
- **CloudWatch**: AWS-native monitoring
- **Weights & Biases**: Experiment tracking
- **MLflow**: Model registry and tracking

### Development

- **Jupyter**: Interactive development
- **VS Code + Remote SSH**: Remote GPU development
- **Docker**: Containerized deployments
- **Terraform/CDK**: Infrastructure as code

---

## Certification Paths

### AWS

1. **AWS Certified Machine Learning - Specialty**
2. **AWS Certified Solutions Architect - Professional**

### NVIDIA

1. **NVIDIA Deep Learning Institute - LLM Deployment**
2. **NVIDIA Certified Associate - AI Infrastructure**

### General ML

1. **Google Cloud Professional ML Engineer**
2. **Coursera Deep Learning Specialization**

---

## Stay Updated

### Newsletters

- **The Batch** (deeplearning.ai) - Weekly AI news
- **Import AI** - Research summaries
- **Last Week in AI** - Comprehensive roundup

### Podcasts

- **Practical AI** - Applied ML discussions
- **TWIML AI** - Research interviews
- **Gradient Dissent** - W&B podcast

### Conferences

- **NeurIPS** - December
- **ICML** - July
- **ACL** - Annual (NLP focus)
- **AWS re:Invent** - November/December

---

## Next Steps

1. **Practice**: Deploy a model on your own AWS account
2. **Benchmark**: Compare vLLM vs SGLang for your use case
3. **Optimize**: Try different quantization levels
4. **Scale**: Set up multi-GPU inference
5. **Monitor**: Build a production monitoring dashboard
6. **Share**: Write about your learnings

---

## Contact and Support

### Workshop Materials

All workshop materials are available at:
`learnings/ml_infra/llm_inference_at_scale/`

### Questions?

- Open an issue in the workshop repository
- Join the team Slack channel
- Schedule office hours with the ML Platform team

---

_Happy inferencing! 🚀_
