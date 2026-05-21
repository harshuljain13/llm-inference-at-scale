# Requirements Document

## Introduction

This document defines the requirements for a comprehensive LLM Inference at Scale Workshop designed for ML platform engineers at Audible. The workshop covers the full LLM inference stack—from transformer mechanics to production-grade serving—with a strong emphasis on AWS services, hands-on labs, and practical production considerations. The workshop follows a "start small, go big and deep" philosophy, building foundational understanding before tackling advanced optimization and scaling topics.

## Glossary

- **Workshop_System**: The complete workshop delivery system including documentation, labs, code examples, and supporting materials
- **Module**: A self-contained unit of workshop content covering a specific topic area
- **Lab_Exercise**: A hands-on practical exercise with clear instructions, code, and expected outcomes
- **Deep_Dive_Document**: A comprehensive technical document following the established style (Mermaid diagrams, code examples, comparison tables)
- **Participant**: An ML platform engineer attending the workshop
- **Instructor**: The person delivering the workshop content
- **TTFT**: Time To First Token—latency from request to first generated token
- **TBT**: Time Between Tokens—latency between consecutive generated tokens
- **KV_Cache**: Key-Value cache storing attention states for autoregressive generation
- **Prefill_Phase**: Initial processing of the input prompt (compute-bound)
- **Decode_Phase**: Autoregressive token generation (memory-bandwidth-bound)

## Requirements

### Requirement 1: Workshop Foundation Module (Module 0)

**User Story:** As a workshop participant, I want to understand why LLM inference is fundamentally different from traditional ML inference, so that I can appreciate the unique challenges and optimization opportunities.

#### Acceptance Criteria

1. WHEN a participant completes Module 0, THE Workshop_System SHALL have explained the key differences between LLM inference and traditional ML inference (autoregressive generation, variable output length, memory-bound decode phase)
2. WHEN presenting the motivation, THE Workshop_System SHALL include real-world cost and latency examples from production LLM deployments
3. THE Workshop_System SHALL provide a visual comparison of inference patterns (single forward pass vs. autoregressive loop)
4. WHEN introducing the workshop structure, THE Workshop_System SHALL present a clear roadmap showing how each module builds on previous concepts
5. THE Deep_Dive_Document SHALL include a Mermaid diagram showing the end-to-end inference request lifecycle

### Requirement 2: Transformer Inference Mechanics Module (Module 1)

**User Story:** As a workshop participant, I want to understand transformer inference from first principles, so that I can reason about performance bottlenecks and optimization opportunities.

#### Acceptance Criteria

1. WHEN explaining token generation, THE Workshop_System SHALL cover the complete pipeline: tokenization → embedding → attention → MLP → logits → sampling
2. THE Workshop_System SHALL explain the prefill vs. decode phases with clear diagrams showing compute vs. memory bandwidth characteristics
3. WHEN covering the KV cache, THE Workshop_System SHALL explain what it stores, why it grows with sequence length, and how it dominates memory at scale
4. THE Workshop_System SHALL include attention variant comparisons (MHA, MQA, GQA) with memory footprint calculations
5. WHEN presenting FlashAttention, THE Workshop_System SHALL explain the IO-aware tiling strategy and why it eliminates N×N HBM reads
6. THE Workshop_System SHALL include a hands-on exercise implementing a minimal transformer forward pass to demonstrate KV cache mechanics
7. THE Deep_Dive_Document SHALL include shape annotations for all tensor operations in the forward pass

### Requirement 3: GPU and Memory Engineering Module (Module 2)

**User Story:** As a workshop participant, I want to develop "roofline thinking" for GPU inference, so that I can quickly identify whether a workload is compute-bound or memory-bound.

#### Acceptance Criteria

1. THE Workshop_System SHALL explain the roofline model with arithmetic intensity calculations for prefill and decode phases
2. WHEN covering GPU memory hierarchy, THE Workshop_System SHALL explain registers, L1/L2 cache, and HBM with bandwidth numbers for relevant GPU types (A10G, A100, H100, Inferentia2)
3. THE Workshop_System SHALL provide VRAM "napkin math" formulas for calculating memory requirements: model weights + KV cache + activations + overhead
4. WHEN explaining memory fragmentation, THE Workshop_System SHALL demonstrate how fragmentation reduces effective batch size and throughput
5. THE Workshop_System SHALL include a comparison table of AWS GPU instance types (g5, p4d, p5, inf2) with memory, bandwidth, and cost metrics
6. THE Lab_Exercise SHALL have participants calculate VRAM requirements for Llama 3.1 8B and 70B at various batch sizes and sequence lengths
7. IF a participant provides model parameters and batch configuration, THEN THE Workshop_System SHALL enable them to predict the dominant bottleneck (compute vs. memory bandwidth)

### Requirement 4: Optimization Techniques Module (Module 3)

**User Story:** As a workshop participant, I want to understand the key optimization techniques for LLM inference, so that I can apply them appropriately based on workload characteristics.

#### Acceptance Criteria

1. WHEN covering quantization, THE Workshop_System SHALL explain INT8, INT4, NF4, FP8, and AWQ with accuracy/performance tradeoff data
2. THE Workshop_System SHALL explain PagedAttention with OS-style memory paging analogies and block allocation diagrams
3. WHEN presenting speculative decoding, THE Workshop_System SHALL cover draft-verify loop, Medusa, EAGLE, and n-gram approaches with speedup expectations
4. THE Workshop_System SHALL explain continuous batching vs. static batching with throughput comparison diagrams
5. WHEN covering chunked prefill, THE Workshop_System SHALL explain how it prevents long prompts from blocking decode requests
6. THE Workshop_System SHALL include a decision matrix for when to apply each optimization technique based on workload characteristics
7. THE Lab_Exercise SHALL have participants benchmark the same model with different quantization levels and measure accuracy/throughput tradeoffs

### Requirement 5: Inference Engines Deep Dive Module (Module 4)

**User Story:** As a workshop participant, I want to understand the major inference engines (vLLM, SGLang, TensorRT-LLM), so that I can select the right engine for my use case.

#### Acceptance Criteria

1. WHEN covering vLLM, THE Workshop_System SHALL explain PagedAttention implementation, continuous batching, and the V0 vs. V1 architecture differences
2. THE Workshop_System SHALL provide vLLM tuning guidance for the 6 key knobs: max-num-batched-tokens, gpu-memory-utilization, max-num-seqs, prefix-caching, chunked-prefill, and CPU allocation
3. WHEN covering SGLang, THE Workshop_System SHALL explain RadixAttention and when it outperforms vLLM (structured generation, multi-call programs)
4. WHEN covering TensorRT-LLM, THE Workshop_System SHALL explain the compilation approach and NVIDIA-specific optimizations
5. THE Workshop_System SHALL include a comparison table of engines across dimensions: ease of use, throughput, latency, model support, and AWS compatibility
6. THE Lab_Exercise SHALL have participants deploy the same model on vLLM and measure throughput with different configuration profiles
7. WHEN discussing AWS options, THE Workshop_System SHALL cover SageMaker LMI containers with vLLM/TensorRT-LLM backends

### Requirement 6: Scaling and Distribution Module (Module 5)

**User Story:** As a workshop participant, I want to understand how to scale LLM inference beyond a single GPU, so that I can serve large models and handle high throughput requirements.

#### Acceptance Criteria

1. THE Workshop_System SHALL explain tensor parallelism, pipeline parallelism, and data parallelism with clear diagrams showing weight and activation distribution
2. WHEN covering tensor parallelism, THE Workshop_System SHALL explain NCCL collectives (all-reduce, all-gather) and interconnect bandwidth requirements
3. THE Workshop_System SHALL explain MoE (Mixture of Experts) inference including routing, expert parallelism, and load balancing challenges
4. WHEN discussing interconnect, THE Workshop_System SHALL compare NVLink, NVSwitch, and EFA bandwidth with implications for parallelism strategies
5. THE Workshop_System SHALL include VRAM calculations for 70B+ models across different parallelism configurations
6. THE Lab_Exercise SHALL have participants configure tensor parallelism on a multi-GPU instance and measure scaling efficiency
7. IF a model exceeds single-GPU memory, THEN THE Workshop_System SHALL provide a decision framework for choosing parallelism strategy

### Requirement 7: Production Serving Architecture Module (Module 6)

**User Story:** As a workshop participant, I want to understand production serving architectures for LLM inference, so that I can design scalable, reliable serving systems.

#### Acceptance Criteria

1. THE Workshop_System SHALL explain the serving stack layers: inference engine → Ray Serve → KServe/llm-d → load balancer
2. WHEN covering Ray Serve, THE Workshop_System SHALL explain replica management, batching, and multi-node deployment patterns
3. WHEN covering KServe, THE Workshop_System SHALL explain model routing, canary deployments, and autoscaling on Kubernetes/EKS
4. THE Workshop_System SHALL explain llm-d's disaggregated prefill/decode architecture and intelligent routing
5. WHEN discussing AWS options, THE Workshop_System SHALL cover SageMaker endpoints, EKS with KServe, and Bedrock for comparison
6. THE Workshop_System SHALL include architecture diagrams for three deployment patterns: single-model, multi-model routing, and disaggregated prefill/decode
7. THE Lab_Exercise SHALL have participants deploy a model on SageMaker with autoscaling configured
8. THE Workshop_System SHALL cover security considerations: authentication, rate limiting, prompt injection defense, and data privacy

### Requirement 8: Measurement and Operations Module (Module 7)

**User Story:** As a workshop participant, I want to understand how to measure, monitor, and operate LLM inference systems in production, so that I can maintain SLOs and optimize costs.

#### Acceptance Criteria

1. THE Workshop_System SHALL define and explain key metrics: TTFT, TBT, tokens/second, requests/second, P50/P95/P99 latencies
2. WHEN covering benchmarking, THE Workshop_System SHALL explain workload replay methodology with representative prompt distributions
3. THE Workshop_System SHALL provide SLO guidance with example targets for different use cases (chatbot, code completion, batch processing)
4. WHEN covering cost optimization, THE Workshop_System SHALL include cost-per-token calculations across AWS instance types and Bedrock
5. THE Workshop_System SHALL explain capacity planning methodology: traffic modeling, headroom calculation, and burst handling
6. THE Workshop_System SHALL include a monitoring dashboard specification with key metrics and alert thresholds
7. THE Lab_Exercise SHALL have participants run a benchmark suite and analyze the results to identify bottlenecks
8. THE Workshop_System SHALL include a troubleshooting guide for common production issues (high latency, OOM, throughput degradation)

### Requirement 9: Hands-on Labs Module (Module 8)

**User Story:** As a workshop participant, I want comprehensive hands-on labs with clear instructions, so that I can apply the concepts learned in a practical setting.

#### Acceptance Criteria

1. THE Workshop_System SHALL provide at least 6 hands-on labs covering: transformer mechanics, VRAM calculation, quantization benchmarking, vLLM deployment, multi-GPU scaling, and SageMaker deployment
2. WHEN providing lab instructions, THE Workshop_System SHALL include prerequisites, step-by-step commands, expected outputs, and troubleshooting tips
3. THE Lab_Exercise SHALL include AWS cost estimates and cleanup instructions for each lab
4. THE Workshop_System SHALL provide starter code templates and Jupyter notebooks for each lab
5. WHEN a lab requires AWS resources, THE Workshop_System SHALL include CloudFormation or CDK templates for infrastructure setup
6. THE Lab_Exercise SHALL include validation checkpoints where participants can verify their progress
7. IF a participant encounters an error, THEN THE Workshop_System SHALL provide a troubleshooting section with common issues and solutions
8. THE Workshop_System SHALL provide a "challenge" extension for each lab for participants who finish early

### Requirement 10: AWS-Specific Deep Dive Content

**User Story:** As an Audible ML platform engineer, I want AWS-specific guidance for LLM inference, so that I can leverage our existing AWS infrastructure effectively.

#### Acceptance Criteria

1. THE Workshop_System SHALL include a dedicated section on AWS Inferentia2 covering Neuron SDK, compilation workflow, and performance characteristics
2. WHEN covering SageMaker, THE Workshop_System SHALL explain LMI containers, endpoint configuration, autoscaling policies, and multi-model endpoints
3. THE Workshop_System SHALL compare Bedrock vs. self-hosted inference with cost, latency, and flexibility tradeoffs
4. WHEN discussing EKS deployment, THE Workshop_System SHALL cover GPU node pools, Karpenter autoscaling, and KServe installation
5. THE Workshop_System SHALL include IAM and security best practices for LLM serving on AWS
6. THE Deep_Dive_Document SHALL include architecture diagrams for recommended AWS deployment patterns
7. THE Workshop_System SHALL provide cost optimization strategies specific to AWS (Spot instances, Savings Plans, right-sizing)

### Requirement 11: Documentation Deliverables

**User Story:** As a workshop instructor, I want comprehensive documentation following the established deep-dive style, so that participants have high-quality reference materials.

#### Acceptance Criteria

1. THE Workshop_System SHALL produce one Deep_Dive_Document per module following the established style (Mermaid diagrams, code examples, comparison tables)
2. WHEN creating diagrams, THE Workshop_System SHALL use Mermaid syntax for architecture diagrams, sequence diagrams, and flowcharts
3. THE Deep_Dive_Document SHALL include code examples in Python with clear comments and type hints
4. THE Workshop_System SHALL include comparison tables for all major technology choices (engines, instance types, parallelism strategies)
5. WHEN presenting metrics, THE Workshop_System SHALL include specific numbers and benchmarks rather than vague qualitative statements
6. THE Deep_Dive_Document SHALL include a "Key Takeaways" section summarizing the most important points
7. THE Workshop_System SHALL provide a glossary of terms used throughout the workshop
8. THE Deep_Dive_Document SHALL include references to primary sources (papers, documentation, blog posts)

### Requirement 12: Supporting Materials

**User Story:** As a workshop instructor, I want supporting materials (slides, cheat sheets, reference cards), so that I can deliver an effective workshop.

#### Acceptance Criteria

1. THE Workshop_System SHALL provide a workshop outline with timing estimates for each module
2. WHEN creating reference materials, THE Workshop_System SHALL include a one-page cheat sheet summarizing key formulas and decision frameworks
3. THE Workshop_System SHALL provide a "quick reference" card for vLLM configuration options
4. THE Workshop_System SHALL include a cost calculator spreadsheet for AWS instance selection
5. WHEN providing code examples, THE Workshop_System SHALL include a GitHub repository with all lab code organized by module
6. THE Workshop_System SHALL provide pre-workshop setup instructions for participants
7. THE Workshop_System SHALL include post-workshop resources for continued learning

### Requirement 13: Structured Output and Guided Decoding

**User Story:** As a workshop participant, I want to understand structured output generation and guided decoding, so that I can constrain LLM outputs to valid formats (JSON, regex, grammars).

#### Acceptance Criteria

1. THE Workshop_System SHALL explain guided decoding concepts: constrained generation, grammar-based decoding, and schema enforcement
2. WHEN covering vLLM structured output, THE Workshop_System SHALL explain the `--guided-decoding-backend` options (outlines, lm-format-enforcer)
3. THE Workshop_System SHALL explain SGLang's native structured generation capabilities and when it outperforms vLLM for this use case
4. THE Workshop_System SHALL include examples of JSON schema enforcement, regex constraints, and context-free grammar constraints
5. THE Lab_Exercise SHALL have participants implement structured JSON output generation with schema validation
6. WHEN discussing production use cases, THE Workshop_System SHALL cover API response formatting, function calling, and tool use patterns

### Requirement 14: FlashAttention Deep Dive

**User Story:** As a workshop participant, I want to understand the FlashAttention family of optimizations in depth, so that I can reason about attention performance across different hardware.

#### Acceptance Criteria

1. THE Workshop_System SHALL explain FlashAttention-1's IO-aware tiling strategy and why it eliminates N×N HBM reads
2. WHEN covering FlashAttention-2, THE Workshop_System SHALL explain the improved warp partitioning and parallelism strategies
3. THE Workshop_System SHALL explain FlashAttention-3's async execution and FP8 support for H100 GPUs
4. THE Workshop_System SHALL include memory access pattern diagrams comparing standard attention vs FlashAttention
5. WHEN discussing hardware compatibility, THE Workshop_System SHALL specify which FlashAttention versions work on which GPU architectures (Ampere, Hopper, etc.)
6. THE Deep_Dive_Document SHALL include arithmetic intensity calculations showing the transition from memory-bound to compute-bound

### Requirement 15: Edge and Resource-Constrained Deployment (Optional Module)

**User Story:** As a workshop participant, I want to understand edge deployment options for LLM inference, so that I can deploy models on resource-constrained devices when needed.

#### Acceptance Criteria

1. THE Workshop_System SHALL explain llama.cpp architecture, quantization formats (GGUF), and CPU/GPU inference modes
2. WHEN covering mobile deployment, THE Workshop_System SHALL explain Apple CoreML integration and on-device inference pipelines
3. THE Workshop_System SHALL explain NVIDIA Jetson deployment with TensorRT-LLM optimizations
4. THE Workshop_System SHALL include a comparison table of edge deployment options with memory requirements, latency, and supported models
5. IF a participant needs edge deployment, THEN THE Workshop_System SHALL provide guidance on model selection and quantization strategies for target hardware
6. THE Lab_Exercise SHALL include an optional exercise running a quantized model with llama.cpp on CPU
