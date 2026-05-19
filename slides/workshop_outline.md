# LLM Inference at Scale - Workshop Outline

> Comprehensive workshop for ML platform engineers on production LLM inference

---

## Workshop Formats

### Format A: 2-Day Full Workshop (16 hours)

Complete coverage with all labs and deep dives.

### Format B: 1-Day Intensive (8 hours)

Core concepts with selected labs.

### Format C: Half-Day Deep Dive (4 hours)

Focused session on specific topics.

---

## Format A: 2-Day Full Workshop

### Day 1: Foundations & Optimization (8 hours)

| Time  | Duration | Topic                                     | Type       |
| ----- | -------- | ----------------------------------------- | ---------- |
| 09:00 | 30 min   | Welcome & Setup Verification              | Setup      |
| 09:30 | 45 min   | Module 0: Why LLM Inference is Different  | Lecture    |
| 10:15 | 15 min   | Break                                     | -          |
| 10:30 | 60 min   | Module 1: Transformer Inference Mechanics | Lecture    |
| 11:30 | 45 min   | Lab 1: Transformer Forward Pass           | Hands-on   |
| 12:15 | 60 min   | Lunch                                     | -          |
| 13:15 | 45 min   | Module 2: GPU & Memory Engineering        | Lecture    |
| 14:00 | 45 min   | Lab 2: VRAM Calculation                   | Hands-on   |
| 14:45 | 15 min   | Break                                     | -          |
| 15:00 | 60 min   | Module 3: Optimization Techniques         | Lecture    |
| 16:00 | 60 min   | Lab 3: Quantization Comparison            | Hands-on   |
| 17:00 | 30 min   | Day 1 Wrap-up & Q&A                       | Discussion |

**Day 1 Learning Outcomes:**

- Understand autoregressive generation and KV cache
- Calculate VRAM requirements for any model
- Apply quantization and understand tradeoffs
- Identify compute vs memory bottlenecks

### Day 2: Engines, Scaling & Production (8 hours)

| Time  | Duration | Topic                                 | Type     |
| ----- | -------- | ------------------------------------- | -------- |
| 09:00 | 15 min   | Day 1 Recap                           | Review   |
| 09:15 | 60 min   | Module 4: Inference Engines Deep Dive | Lecture  |
| 10:15 | 15 min   | Break                                 | -        |
| 10:30 | 60 min   | Lab 4: vLLM Deployment                | Hands-on |
| 11:30 | 45 min   | Lab 5: SGLang Structured Output       | Hands-on |
| 12:15 | 60 min   | Lunch                                 | -        |
| 13:15 | 45 min   | Module 5: Scaling & Parallelism       | Lecture  |
| 14:00 | 45 min   | Lab 6: Tensor Parallelism             | Hands-on |
| 14:45 | 15 min   | Break                                 | -        |
| 15:00 | 45 min   | Module 6: Production Serving          | Lecture  |
| 15:45 | 45 min   | Lab 7: Ray Serve Deployment           | Hands-on |
| 16:30 | 45 min   | Module 8: AWS Deep Dive               | Lecture  |
| 17:15 | 30 min   | Lab 9: SageMaker Production           | Hands-on |
| 17:45 | 15 min   | Workshop Wrap-up & Resources          | Closing  |

**Day 2 Learning Outcomes:**

- Configure vLLM for different workloads
- Implement structured output generation
- Scale models across multiple GPUs
- Deploy production-ready inference services
- Choose appropriate AWS services

---

## Format B: 1-Day Intensive (8 hours)

| Time  | Duration | Topic                                              | Type     |
| ----- | -------- | -------------------------------------------------- | -------- |
| 09:00 | 30 min   | Welcome & Module 0: Why LLM Inference is Different | Lecture  |
| 09:30 | 45 min   | Module 1: Transformer Mechanics (condensed)        | Lecture  |
| 10:15 | 15 min   | Break                                              | -        |
| 10:30 | 45 min   | Module 2: GPU & Memory Engineering                 | Lecture  |
| 11:15 | 45 min   | Lab 2: VRAM Calculation                            | Hands-on |
| 12:00 | 60 min   | Lunch                                              | -        |
| 13:00 | 45 min   | Module 3: Optimization Techniques                  | Lecture  |
| 13:45 | 45 min   | Module 4: Inference Engines (vLLM focus)           | Lecture  |
| 14:30 | 15 min   | Break                                              | -        |
| 14:45 | 60 min   | Lab 4: vLLM Deployment                             | Hands-on |
| 15:45 | 30 min   | Module 5: Scaling & Parallelism                    | Lecture  |
| 16:15 | 45 min   | Module 8: AWS Deep Dive                            | Lecture  |
| 17:00 | 45 min   | Lab 9: SageMaker Production                        | Hands-on |
| 17:45 | 15 min   | Wrap-up & Resources                                | Closing  |

**1-Day Learning Outcomes:**

- Understand LLM inference fundamentals
- Calculate and optimize memory usage
- Deploy vLLM for production workloads
- Use AWS services for LLM inference

---

## Format C: Half-Day Deep Dives (4 hours each)

### Option C1: Optimization & Performance

Focus: Getting maximum performance from your hardware

| Time  | Duration | Topic                                |
| ----- | -------- | ------------------------------------ |
| 09:00 | 30 min   | Memory Hierarchy & Bottlenecks       |
| 09:30 | 45 min   | Quantization Deep Dive               |
| 10:15 | 15 min   | Break                                |
| 10:30 | 45 min   | Lab: Quantization Comparison         |
| 11:15 | 45 min   | vLLM Tuning Knobs                    |
| 12:00 | 45 min   | Lab: vLLM Configuration Optimization |
| 12:45 | 15 min   | Q&A                                  |

### Option C2: Production Deployment

Focus: Running LLMs in production on AWS

| Time  | Duration | Topic                            |
| ----- | -------- | -------------------------------- |
| 09:00 | 30 min   | Production Architecture Patterns |
| 09:30 | 45 min   | AWS Service Selection            |
| 10:15 | 15 min   | Break                            |
| 10:30 | 45 min   | Lab: SageMaker LMI Deployment    |
| 11:15 | 45 min   | Monitoring & Operations          |
| 12:00 | 45 min   | Lab: Benchmarking & Monitoring   |
| 12:45 | 15 min   | Q&A                              |

### Option C3: Multi-GPU & Scaling

Focus: Scaling to large models and high throughput

| Time  | Duration | Topic                        |
| ----- | -------- | ---------------------------- |
| 09:00 | 30 min   | Parallelism Strategies       |
| 09:30 | 45 min   | Tensor Parallelism Deep Dive |
| 10:15 | 15 min   | Break                        |
| 10:30 | 45 min   | Lab: Multi-GPU vLLM          |
| 11:15 | 45 min   | Disaggregated Serving        |
| 12:00 | 45 min   | Lab: Ray Serve Deployment    |
| 12:45 | 15 min   | Q&A                          |

---

## Module Dependencies

```
Module 0 (Why Different)
    │
    ▼
Module 1 (Transformer Mechanics) ──────┐
    │                                   │
    ▼                                   ▼
Module 2 (GPU/Memory) ────────► Module 3 (Optimization)
    │                                   │
    └───────────────┬───────────────────┘
                    │
                    ▼
            Module 4 (Engines)
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    Module 5    Module 6    Module 8
    (Scaling)   (Serving)   (AWS)
        │           │           │
        └───────────┴───────────┘
                    │
                    ▼
            Module 7 (Operations)
```

---

## Prerequisites

### Required Knowledge

- Python programming (intermediate)
- Basic ML/DL concepts
- Command line familiarity
- AWS basics (EC2, S3)

### Required Setup

- AWS account with GPU instance access
- HuggingFace account and token
- Python 3.10+ environment
- CUDA-compatible GPU (for local labs)

### Recommended Reading

- Attention Is All You Need (Vaswani et al.)
- vLLM paper (Kwon et al.)
- FlashAttention paper (Dao et al.)

---

## Lab Requirements

| Lab    | Instance Type | Min VRAM | Duration |
| ------ | ------------- | -------- | -------- |
| Lab 1  | Any           | 8GB      | 45 min   |
| Lab 2  | Any           | 4GB      | 45 min   |
| Lab 3  | g5.xlarge     | 24GB     | 60 min   |
| Lab 4  | g5.xlarge     | 24GB     | 60 min   |
| Lab 5  | g5.xlarge     | 24GB     | 45 min   |
| Lab 6  | g5.12xlarge   | 96GB     | 45 min   |
| Lab 7  | g5.xlarge     | 24GB     | 45 min   |
| Lab 8  | EKS cluster   | -        | 60 min   |
| Lab 9  | SageMaker     | -        | 45 min   |
| Lab 10 | g5.xlarge     | 24GB     | 45 min   |

---

## Instructor Notes

### Pacing Guidelines

- Allow extra time for setup issues
- Labs often take 20% longer than estimated
- Build in buffer for Q&A
- Have backup exercises for fast finishers

### Common Issues

1. HuggingFace token not configured
2. CUDA version mismatches
3. Insufficient GPU memory
4. Network timeouts downloading models

### Key Concepts to Emphasize

1. Memory bandwidth is usually the bottleneck
2. KV cache grows with sequence length AND batch size
3. Quantization is almost always worth it
4. Continuous batching is essential for throughput
5. TTFT and ITL are different optimization targets

---

## Post-Workshop Resources

### Continued Learning

- vLLM documentation and GitHub
- SGLang documentation
- AWS ML Blog
- Hugging Face course

### Community

- vLLM Discord
- r/LocalLLaMA
- AWS ML community

### Certification Path

- AWS ML Specialty
- NVIDIA Deep Learning Institute
