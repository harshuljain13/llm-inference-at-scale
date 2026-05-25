# Industry Blogs & Articles: LLM Inference at Scale (2025-2026)

> Critical analysis of blog posts from Ray/Anyscale, AWS, and Azure/Microsoft relevant to the LLM Inference at Scale workshop.

---

## 1. Ray / Anyscale

### 1.1 Major Upgrades to Ray Serve: 88% Lower Latency and 11.1x Higher Throughput (Mar 2026)

**URL**: https://www.anyscale.com/blog/ray-serve-inference-lower-latency-higher-throughput-haproxy

**Key Technical Insights:**
- HAProxy integration for external load balancing at >5K req/s
- Zero-copy data path eliminates serialization overhead
- Async request handling pipeline
- **88% latency reduction + 11.1x throughput** vs. previous Ray Serve versions

**Critical Assessment:**
- Validates Ray Serve as production-grade for LLM serving at scale
- HAProxy integration pattern is directly applicable to workshop Module 6
- Numbers are impressive but likely measured on specific workloads — participants should benchmark their own

**Workshop Relevance**: ★★★★★ — Update Module 6 Ray Serve section with new architecture and performance numbers.

---

### 1.2 Wide-EP and Disaggregated Serving with vLLM (Dec 2025)

**URL**: https://www.anyscale.com/blog/ray-serve-llm-anyscale-apis-wide-ep-disaggregated-serving-vllm

**Key Technical Insights:**
- Wide Expert Parallelism (Wide-EP): distributes MoE experts across many GPUs for better load balancing
- Disaggregated prefill/decode with Ray Serve orchestration
- vLLM as inference engine with Ray Serve managing replicas and routing

**Critical Assessment:**
- Directly relevant to Module 5 (MoE inference) and Module 6 (disaggregated serving)
- Wide-EP is the production answer to MoE load imbalance discussed in workshop
- Demonstrates Ray Serve + vLLM as the dominant open-source serving stack

**Workshop Relevance**: ★★★★★ — Must-add for MoE serving and disaggregated architecture sections.

---

### 1.3 Reduce LLM Inference Latency by 60% with Custom Request Routing (Sep 2025)

**URL**: https://www.anyscale.com/blog/ray-serve-faster-first-token-custom-routing

**Key Technical Insights:**
- Custom routing policies that route prefill-heavy requests to idle replicas
- Prefix-aware routing: route requests sharing prefixes to same replica for cache hits
- **60% TTFT reduction** with intelligent routing

**Critical Assessment:**
- Practical optimization that requires no model changes
- Directly applicable to production deployments
- Complements ContextPilot paper's prefix reuse approach

**Workshop Relevance**: ★★★★★ — Add to Module 6 as production optimization technique.

---

### 1.4 DeepSeek on Kubernetes with vLLM and Ray Serve (Aug 2025)

**URL**: https://www.anyscale.com/blog/deepseek-vllm-ray-google-kubernetes

**Key Technical Insights:**
- End-to-end deployment guide for DeepSeek (MoE model) on Kubernetes
- vLLM + Ray Serve + Kubernetes orchestration
- Multi-node tensor parallelism configuration

**Workshop Relevance**: ★★★★☆ — Reference deployment guide for Lab 8 (EKS + KServe).

---

### 1.5 AI Agents on Ray Serve: Single to Multi-Agent Architecture (May 2026)

**URL**: https://www.anyscale.com/blog/ai-agents-on-ray-serve-single-to-multi-agent-architecture

**Key Technical Insights:**
- Serving agentic workloads (multi-step, tool-calling) on Ray Serve
- Architecture patterns for single-agent vs. multi-agent serving
- Implications for inference: variable-length, multi-turn, tool-calling patterns

**Critical Assessment:**
- Agentic inference is the next frontier — different workload characteristics than simple chat
- Higher TTFT tolerance but stricter TBT requirements for streaming
- Multi-step means multiple inference calls per user request

**Workshop Relevance**: ★★★★☆ — Advanced topic for proposed new module on agentic inference.

---

### 1.6 Advancing Flexibility: Async Inference, Custom Request Routing, and Custom Autoscaling (Nov 2025)

**URL**: https://www.anyscale.com/blog/ray-serve-autoscaling-async-inference-custom-routing

**Key Technical Insights:**
- Custom autoscaling policies based on GPU utilization, queue depth, or custom metrics
- Async inference patterns for non-blocking request handling
- Pluggable routing for A/B testing, canary, and weighted routing

**Workshop Relevance**: ★★★★☆ — Extends Module 6 autoscaling and routing sections.

---

## 2. AWS / Amazon

### 2.1 Introducing Disaggregated Inference on AWS powered by llm-d (Apr 2026)

**URL**: https://aws.amazon.com/blogs/machine-learning/introducing-disaggregated-inference-on-aws-powered-by-llm-d/

**Key Technical Insights:**
- Official AWS support for llm-d disaggregated prefill/decode
- Separate prefill and decode pools with KV cache transfer
- Intelligent routing based on request characteristics
- Integration with SageMaker and EKS

**Critical Assessment:**
- Validates disaggregated serving as production-ready on AWS
- llm-d is now a first-class AWS offering, not just research
- Directly aligns with workshop Module 6 content

**Workshop Relevance**: ★★★★★ — Update Module 6 and Module 8 (AWS) with official llm-d support.

---

### 2.2 Capacity-aware Inference: Automatic Instance Fallback for SageMaker AI Endpoints (May 2026)

**URL**: https://aws.amazon.com/blogs/machine-learning/capacity-aware-inference-automatic-instance-fallback-for-sagemaker-ai-endpoints/

**Key Technical Insights:**
- Automatic fallback to alternative instance types when primary capacity unavailable
- Addresses GPU scarcity problem in production
- Transparent to application — SageMaker handles routing

**Critical Assessment:**
- Production resilience pattern — critical for high-availability LLM serving
- Relevant for capacity planning discussion in Module 7

**Workshop Relevance**: ★★★★☆ — Add to Module 7 (Operations) and Module 8 (AWS).

---

### 2.3 Efficiently Serve Dozens of Fine-tuned Models with vLLM on SageMaker (Mar 2026)

**URL**: https://aws.amazon.com/blogs/machine-learning/efficiently-serve-dozens-of-fine-tuned-models-with-vllm-on-amazon-sagemaker-ai-and-amazon-bedrock/

**Key Technical Insights:**
- Multi-LoRA serving with vLLM on SageMaker
- Shared base model + multiple LoRA adapters
- Cost efficiency: serve dozens of models on single endpoint

**Critical Assessment:**
- Multi-LoRA is increasingly important for enterprise (per-customer fine-tunes)
- vLLM's LoRA support is production-ready on SageMaker
- Significant cost savings vs. dedicated endpoints per model

**Workshop Relevance**: ★★★★★ — New topic for Module 8 (AWS) and advanced content proposal.

---

### 2.4 Accelerating Decode-Heavy LLM Inference with Speculative Decoding on AWS Trainium (Apr 2026)

**URL**: https://aws.amazon.com/blogs/machine-learning/accelerating-decode-heavy-llm-inference-with-speculative-decoding-on-aws-trainium-and-vllm/

**Key Technical Insights:**
- Speculative decoding on Trainium (not just GPU)
- vLLM integration with Neuron SDK for speculation
- Decode-heavy workloads (short prompts, long outputs) benefit most

**Critical Assessment:**
- Extends speculative decoding to custom silicon — important for cost optimization
- Validates Trainium as viable for advanced inference techniques

**Workshop Relevance**: ★★★★★ — Update Module 8 Inferentia/Trainium section with speculative decoding support.

---

### 2.5 How Amazon Scaled Rufus with Multi-Node Inference on Trainium (Aug 2025)

**URL**: https://aws.amazon.com/blogs/machine-learning/how-amazon-scaled-rufus-by-building-multi-node-inference-using-aws-trainium-chips-and-vllm/

**Key Technical Insights:**
- Real production case study: Amazon Rufus (shopping assistant)
- Multi-node inference on Trainium with vLLM
- Handled Prime Day traffic spikes
- **2x inference speed** with parallel decoding

**Critical Assessment:**
- Best real-world case study available for AWS LLM inference at scale
- Demonstrates multi-node Trainium viability for large models
- Prime Day scale = millions of concurrent users

**Workshop Relevance**: ★★★★★ — Case study for Module 8; demonstrates production scale on AWS custom silicon.

---

### 2.6 EAGLE-based Adaptive Speculative Decoding on SageMaker (May 2025)

**URL**: https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-ai-introduces-eagle-based-adaptive-speculative-decoding-to-accelerate-generative-ai-inference/

**Key Technical Insights:**
- EAGLE speculative decoding natively supported in SageMaker LMI
- Adaptive γ selection based on acceptance rate history
- No separate draft model needed (EAGLE uses feature extrapolation)

**Workshop Relevance**: ★★★★★ — Update Module 3 speculative decoding with EAGLE on SageMaker.

---

### 2.7 Build Real-time Voice Applications with SageMaker and vLLM (May 2026)

**URL**: https://aws.amazon.com/blogs/machine-learning/build-real-time-voice-applications-with-amazon-sagemaker-ai-and-vllm/

**Key Technical Insights:**
- Ultra-low latency requirements for voice (< 200ms TTFT)
- Streaming inference with vLLM for real-time applications
- SageMaker endpoint configuration for voice workloads

**Workshop Relevance**: ★★★★☆ — Advanced use case for Module 7 (SLO targets for voice).

---

### 2.8 Accelerating LLM Inference with AWQ and GPTQ on SageMaker (Jan 2026)

**URL**: https://aws.amazon.com/blogs/machine-learning/accelerating-llm-inference-with-post-training-weight-and-activation-using-awq-and-gptq-on-amazon-sagemaker-ai/

**Key Technical Insights:**
- Production deployment of AWQ and GPTQ quantized models on SageMaker
- Benchmark comparisons: AWQ vs GPTQ vs FP16 on g5 and p4d instances
- Best practices for quantization selection

**Workshop Relevance**: ★★★★★ — Directly supports Lab 3 (Quantization Comparison) with AWS-specific benchmarks.

---

## 3. Azure / Microsoft

### 3.1 Enterprise LLM Inference Series (3-part, Dec 2025 - May 2026)

#### Part 1: Why LLM Inference Is a Capital Allocation Problem
**URL**: https://techcommunity.microsoft.com/blog/appsonazureblog/part-1-inference-at-enterprise-scale-why-llm-inference-is-a-capital-allocation-p/4498754

**Key Insights:**
- Frames inference as capital allocation, not just infrastructure
- Pareto frontier: accuracy/latency/cost tradeoff
- Two-phase bottleneck analysis (prefill vs decode)
- KV cache pressure at enterprise scale
- Agentic workloads changing inference patterns

#### Part 2: The LLM Inference Optimization Stack
**URL**: https://techcommunity.microsoft.com/blog/appsonazureblog/the-llm-inference-optimization-stack-a-prioritized-playbook-for-enterprise-teams/4498818

**Key Insights:**
- Prioritized optimization framework (what to do first)
- Continuous batching → quantization → speculative decoding → disaggregated prefill
- AKS + vLLM + PagedAttention as recommended stack

#### Part 3: Building a Controllable Inference Platform on Kubernetes with AI Runway
**URL**: https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-a-controllable-inference-platform-on-kubernetes-with-ai-runway/4520590

**Key Insights:**
- Enterprise governance for inference platforms
- Controllability, observability, and policy enforcement
- Moving from demos to production workloads

**Critical Assessment (Series):**
- Excellent enterprise framing — complements our workshop's production focus
- "Capital allocation" framing is powerful for executive buy-in
- Prioritized optimization stack aligns with our Module 3 structure
- AI Runway concept relevant for enterprise governance

**Workshop Relevance**: ★★★★★ — Informs Module 0 (motivation), Module 3 (optimization priority), Module 6 (governance).

---

### 3.2 Eliminate LLM Cold Starts: 6x Faster Model Loading (May 2026)

**URL**: https://devblogs.microsoft.com/azure-sdk/eliminate-llm-cold-starts-load-models-up-to-6x-faster-with-azure-blob-storage-and-runai-model-streamer/

**Key Technical Insights:**
- Cold start tax from auto-scaling, spot eviction, rolling deploys, model swaps
- Streams model weights directly from blob storage to GPU memory (bypasses disk)
- **6x faster model loading**
- Critical for multi-tenant serving and spot VM recovery

**Critical Assessment:**
- Cold start is a real production pain point not well-covered in current workshop
- Model streaming is the solution — applicable to AWS (S3 → GPU) too
- Run:AI Model Streamer is open-source

**Workshop Relevance**: ★★★★★ — New topic for Module 6 (cold start mitigation) and Module 7 (operations).

---

### 3.3 vLLM Performance on ND-H100-v5

**URL**: https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/performance-of-llama-3-1-8b-ai-inference-using-vllm-on-nd-h100-v5/...

**Key Insights:**
- H100 benchmark data for Llama 3.1 8B with vLLM
- Performance characteristics at various batch sizes and sequence lengths
- Comparison with A100 baseline

**Workshop Relevance**: ★★★★☆ — Benchmark reference data for Module 2 and Module 4.

---

## 4. NVIDIA / DeepSeek Ecosystem

### 4.1 DeepSeek-V3.2 on GB300 (vLLM Blog, Jun 2026)

**URL**: https://vllm.ai/blog/gb300-deepseek

**Key Technical Insights:**
- NVFP4 quantization + TP2 on GB300 (Blackwell Ultra)
- **7,360 tokens/GPU/second** (prefill-only)
- **2,816 TGS** mixed-context (ISL=2k, OSL=1k)
- Per-user output: **230 TPS** (4x typical providers)

**Critical Assessment:**
- GB300 is bleeding-edge hardware (not yet widely available on AWS)
- NVFP4 is NVIDIA-specific — not portable to Trainium/Inferentia
- Numbers represent ceiling of what's possible, not typical deployment

**Workshop Relevance**: ★★★★☆ — Reference benchmark for "what's possible"; future hardware section.

---

### 4.2 Perplexity: Multi-Node DeepSeek Deployment (Apr 2025)

**URL**: https://www.perplexity.ai/hub/blog/lower-latency-and-higher-throughput-with-multi-node-deepseek-deployment

**Key Technical Insights:**
- MoE models achieve simultaneous higher throughput AND lower latency with more GPUs
- Contrary to dense model behavior (more GPUs = diminishing returns)
- Multi-node deployment is the correct strategy for large MoE

**Workshop Relevance**: ★★★★★ — Challenges conventional wisdom; must-add to Module 5.

---

### 4.3 Perplexity: 10x Faster MoE Communication (Feb 2026)

**URL**: https://www.perplexity.ai/hub/blog/efficient-and-portable-mixture-of-experts-communication

**Key Technical Insights:**
- **10x faster** all-to-all communication for expert routing
- Optimized for AWS EFA and other network fabrics
- Portable across cloud providers

**Critical Assessment:**
- Directly addresses the MoE double penalty (communication overhead)
- AWS EFA optimization is directly relevant for workshop participants
- Open question: how much of this is reproducible without Perplexity's custom stack?

**Workshop Relevance**: ★★★★★ — Critical for Module 5 MoE section; demonstrates communication optimization.

---

### 4.4 NVIDIA NIM (Inference Microservices)

**URL**: https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/llm-nim

**Key Technical Insights:**
- Production container with TensorRT-LLM, vLLM, SGLang backends
- Auto-selects optimal engine per model/hardware
- NVIDIA Dynamo for distributed orchestration
- Enterprise support and licensing

**Critical Assessment:**
- **Strengths**: One-click deployment; auto-optimization; handles complexity
- **Weaknesses**: NVIDIA GPU lock-in; opaque decisions; licensing costs; not available on Inferentia
- **vs. Open-source**: Less control, more convenience. Enterprise teams may prefer NIM; ML platform teams building custom stacks should use vLLM/SGLang directly.
- **Marketing vs Reality**: "Optimized" often means TensorRT-LLM compilation which any team can do manually

**Workshop Relevance**: ★★★☆☆ — Mention as alternative; workshop is AWS-centric and favors open-source stack.

---

### 4.5 SGLang on GB300 NVL72 (LMSYS, Feb 2026)

**URL**: https://lmsys.org/blog/2026-02-19-gb300-longctx/

**Key Technical Insights:**
- Long-context (128K/8K) peak: **226.2 TPS/GPU** on GB300 NVL72
- 1.53x advantage over GB200
- Multi-Token Prediction (MTP): **1.87x TPS/User** under same throughput

**Workshop Relevance**: ★★★★☆ — SGLang performance reference; MTP as emerging technique.

---

## Summary: Top Blog Posts by Workshop Impact

| Rank | Post | Source | Relevance | Module Impact |
|------|------|--------|-----------|---------------|
| 1 | Disaggregated Inference with llm-d | AWS | ★★★★★ | Module 6, 8 |
| 2 | Enterprise Inference Series (3-part) | Azure | ★★★★★ | Module 0, 3, 6 |
| 3 | Wide-EP + Disaggregated Serving | Ray | ★★★★★ | Module 5, 6 |
| 4 | Ray Serve 88% Latency Reduction | Ray | ★★★★★ | Module 6 |
| 5 | Rufus Multi-Node Trainium | AWS | ★★★★★ | Module 8 (case study) |
| 6 | 10x Faster MoE Communication | Perplexity | ★★★★★ | Module 5 |
| 7 | Multi-LoRA Serving on SageMaker | AWS | ★★★★★ | Module 8, Advanced |
| 8 | 60% TTFT with Custom Routing | Ray | ★★★★★ | Module 6 |
| 9 | Cold Start Elimination (6x) | Azure | ★★★★★ | Module 6, 7 |
| 10 | Speculative Decoding on Trainium | AWS | ★★★★★ | Module 8 |
| 11 | DeepSeek-V3.2 on GB300 | vLLM | ★★★★☆ | Module 5 (benchmark) |
| 12 | EAGLE on SageMaker | AWS | ★★★★★ | Module 3, 8 |
| 13 | NVIDIA NIM | NVIDIA | ★★★☆☆ | Module 4 (mention) |
