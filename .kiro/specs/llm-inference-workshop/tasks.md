# Implementation Tasks

## Task 1: Project Setup and Infrastructure

- [ ] 1.1 Create workshop directory structure
  - [ ] 1.1.1 Create main directory `learnings/ml_infra/llm_inference_at_scale/`
  - [ ] 1.1.2 Create `labs/` subdirectory with lab folders (lab_01 through lab_10)
  - [ ] 1.1.3 Create `labs/infrastructure/` with cloudformation/, cdk/, terraform/ subdirectories
  - [ ] 1.1.4 Create `reference/` subdirectory for cheat sheets and quick references
  - [ ] 1.1.5 Create `slides/` subdirectory for presentation materials
- [ ] 1.2 Create workshop configuration file
  - [ ] 1.2.1 Implement WorkshopConfig schema in Python
  - [ ] 1.2.2 Create workshop_config.yaml with module and lab definitions
  - [ ] 1.2.3 Add validation script for configuration

## Task 2: Module 0 - Why LLM Inference is Different

- [ ] 2.1 Create `00_why_llm_inference_is_different.md`
  - [ ] 2.1.1 Write introduction explaining autoregressive generation
  - [ ] 2.1.2 Add Mermaid diagram comparing traditional ML vs LLM inference patterns
  - [ ] 2.1.3 Create cost reality check comparison table (compute, memory, latency, cost per 1M requests)
  - [ ] 2.1.4 Add end-to-end request lifecycle sequence diagram
  - [ ] 2.1.5 Write Key Takeaways section
  - [ ] 2.1.6 Add references to foundational papers

## Task 3: Module 1 - Transformer Inference Mechanics

- [ ] 3.1 Create `01_transformer_inference_mechanics.md`
  - [ ] 3.1.1 Write token generation pipeline explanation with Mermaid flowchart
  - [ ] 3.1.2 Add tensor shape annotations for all operations (input_ids through logits)
  - [ ] 3.1.3 Create KV cache deep dive section with prefill/decode diagrams
  - [ ] 3.1.4 Write KV cache memory formula with worked example
  - [ ] 3.1.5 Add attention variants comparison table (MHA, MQA, GQA)
  - [ ] 3.1.6 Create Mermaid diagrams showing MHA vs GQA vs MQA head arrangements
  - [ ] 3.1.7 Write Key Takeaways section
- [ ] 3.2 Create Lab 1: Transformer Forward Pass
  - [ ] 3.2.1 Create `labs/lab_01_transformer_forward_pass/notebook.ipynb`
  - [ ] 3.2.2 Implement MinimalAttention class with KV cache support
  - [ ] 3.2.3 Implement demonstrate_kv_cache_growth() function
  - [ ] 3.2.4 Add exercises for participants to modify and experiment
  - [ ] 3.2.5 Create `labs/lab_01_transformer_forward_pass/solutions.py`
  - [ ] 3.2.6 Create `labs/lab_01_transformer_forward_pass/README.md` with prerequisites and instructions

## Task 4: Module 2 - GPU and Memory Engineering

- [ ] 4.1 Create `02_gpu_memory_engineering.md`
  - [ ] 4.1.1 Write roofline model explanation with arithmetic intensity calculations
  - [ ] 4.1.2 Create Mermaid diagram for GPU memory hierarchy
  - [ ] 4.1.3 Add AWS GPU instance comparison table (g5, p4d, p5, inf2)
  - [ ] 4.1.4 Write VRAM napkin math formulas section
  - [ ] 4.1.5 Implement calculate_vram_requirements() function with examples
  - [ ] 4.1.6 Add memory fragmentation explanation with diagrams
  - [ ] 4.1.7 Write Key Takeaways section
- [ ] 4.2 Create Lab 2: VRAM Calculation
  - [ ] 4.2.1 Create `labs/lab_02_vram_calculation/notebook.ipynb`
  - [ ] 4.2.2 Implement interactive VRAM calculator
  - [ ] 4.2.3 Add exercises for Llama 3.1 8B and 70B calculations
  - [ ] 4.2.4 Create bottleneck prediction exercises
  - [ ] 4.2.5 Create `labs/lab_02_vram_calculation/README.md`

## Task 5: Module 3 - Optimization Techniques

- [ ] 5.1 Create `03_optimization_techniques.md`
  - [ ] 5.1.1 Write quantization deep dive with precision formats diagram
  - [ ] 5.1.2 Create quantization methods comparison table (FP16, FP8, INT8, INT4, AWQ, NF4)
  - [ ] 5.1.3 Add vLLM quantization code examples
  - [ ] 5.1.4 Write PagedAttention explanation with block allocation diagrams
  - [ ] 5.1.5 Create continuous vs static batching sequence diagram
  - [ ] 5.1.6 Write speculative decoding section with draft-verify flowchart
  - [ ] 5.1.7 Add speculative decoding variants table (Draft Model, Medusa, EAGLE, N-gram)
  - [ ] 5.1.8 Write chunked prefill explanation with diagrams
  - [ ] 5.1.9 Create optimization decision matrix
  - [ ] 5.1.10 Write Key Takeaways section
- [ ] 5.2 Create Lab 3: Quantization Comparison
  - [ ] 5.2.1 Create `labs/lab_03_quantization_comparison/notebook.ipynb`
  - [ ] 5.2.2 Implement benchmark script for FP16 vs INT8 vs INT4
  - [ ] 5.2.3 Add accuracy evaluation code
  - [ ] 5.2.4 Create throughput measurement code
  - [ ] 5.2.5 Add visualization for accuracy/throughput tradeoffs
  - [ ] 5.2.6 Create `labs/lab_03_quantization_comparison/README.md`

## Task 6: Module 4 - Inference Engines Deep Dive

- [ ] 6.1 Create `04_inference_engines_deep_dive.md`
  - [ ] 6.1.1 Write vLLM architecture section with Mermaid flowchart
  - [ ] 6.1.2 Create vLLM V0 vs V1 comparison table
  - [ ] 6.1.3 Document the 6 critical vLLM tuning knobs with code examples
  - [ ] 6.1.4 Write SGLang architecture section with RadixAttention diagram
  - [ ] 6.1.5 Add SGLang code examples for structured generation
  - [ ] 6.1.6 Write TensorRT-LLM section with compilation workflow
  - [ ] 6.1.7 Create engine comparison matrix (ease of use, throughput, latency, etc.)
  - [ ] 6.1.8 Add decision guide flowchart for engine selection
  - [ ] 6.1.9 Write Key Takeaways section
- [ ] 6.2 Create Lab 4: vLLM Deployment
  - [ ] 6.2.1 Create `labs/lab_04_vllm_deployment/notebook.ipynb`
  - [ ] 6.2.2 Add vLLM installation and setup instructions
  - [ ] 6.2.3 Implement throughput-optimized configuration example
  - [ ] 6.2.4 Implement latency-optimized configuration example
  - [ ] 6.2.5 Add benchmarking code to compare configurations
  - [ ] 6.2.6 Create `labs/lab_04_vllm_deployment/README.md`
- [ ] 6.3 Create Lab 5: SGLang Structured Output
  - [ ] 6.3.1 Create `labs/lab_05_sglang_structured_output/notebook.ipynb`
  - [ ] 6.3.2 Implement JSON schema-constrained generation example
  - [ ] 6.3.3 Implement regex-constrained generation example
  - [ ] 6.3.4 Add multi-step reasoning example
  - [ ] 6.3.5 Create `labs/lab_05_sglang_structured_output/README.md`

## Task 7: Module 5 - Scaling and Distribution

- [ ] 7.1 Create `05_scaling_and_parallelism.md`
  - [ ] 7.1.1 Write parallelism strategies section with DP/TP/PP diagrams
  - [ ] 7.1.2 Create tensor parallelism deep dive with column/row parallel explanation
  - [ ] 7.1.3 Add NCCL collectives diagram (AllReduce, AllGather, AllToAll)
  - [ ] 7.1.4 Create interconnect bandwidth requirements table
  - [ ] 7.1.5 Add multi-GPU vLLM configuration examples
  - [ ] 7.1.6 Write MoE inference section with routing diagram
  - [ ] 7.1.7 Implement calculate_multi_gpu_vram() function
  - [ ] 7.1.8 Write Key Takeaways section
- [ ] 7.2 Create Lab 6: Tensor Parallelism
  - [ ] 7.2.1 Create `labs/lab_06_tensor_parallelism/notebook.ipynb`
  - [ ] 7.2.2 Add multi-GPU setup instructions
  - [ ] 7.2.3 Implement TP=2, TP=4, TP=8 configuration examples
  - [ ] 7.2.4 Add scaling efficiency measurement code
  - [ ] 7.2.5 Create `labs/lab_06_tensor_parallelism/README.md`

## Task 8: Module 6 - Production Serving Architecture

- [ ] 8.1 Create `06_production_serving_architecture.md`
  - [ ] 8.1.1 Write production serving stack diagram
  - [ ] 8.1.2 Add Ray Serve deployment code example
  - [ ] 8.1.3 Create KServe InferenceService YAML example
  - [ ] 8.1.4 Write llm-d disaggregated prefill/decode section with diagram
  - [ ] 8.1.5 Add deployment patterns diagrams (single-model, multi-model, canary)
  - [ ] 8.1.6 Write security considerations section with checklist
  - [ ] 8.1.7 Write Key Takeaways section
- [ ] 8.2 Create Lab 7: Ray Serve Deployment
  - [ ] 8.2.1 Create `labs/lab_07_ray_serve_deployment/notebook.ipynb`
  - [ ] 8.2.2 Implement VLLMDeployment class with autoscaling
  - [ ] 8.2.3 Add load testing code
  - [ ] 8.2.4 Create `labs/lab_07_ray_serve_deployment/README.md`
- [ ] 8.3 Create Lab 8: EKS + KServe Deployment
  - [ ] 8.3.1 Create `labs/lab_08_eks_kserve_deployment/` directory
  - [ ] 8.3.2 Create EKS cluster CloudFormation template
  - [ ] 8.3.3 Add KServe installation instructions
  - [ ] 8.3.4 Create InferenceService deployment manifests
  - [ ] 8.3.5 Add autoscaling configuration examples
  - [ ] 8.3.6 Create `labs/lab_08_eks_kserve_deployment/README.md`

## Task 9: Module 7 - Measurement and Operations

- [ ] 9.1 Create `07_measurement_and_operations.md`
  - [ ] 9.1.1 Write key metrics section with timeline diagram
  - [ ] 9.1.2 Create metrics definition table with targets
  - [ ] 9.1.3 Implement benchmark_suite.py with BenchmarkResult dataclass
  - [ ] 9.1.4 Add monitoring dashboard specification YAML
  - [ ] 9.1.5 Create troubleshooting guide table
  - [ ] 9.1.6 Write Key Takeaways section
- [ ] 9.2 Create Lab 10: Benchmarking and Monitoring
  - [ ] 9.2.1 Create `labs/lab_10_benchmarking_monitoring/notebook.ipynb`
  - [ ] 9.2.2 Implement run_benchmark() async function
  - [ ] 9.2.3 Add visualization code for latency distributions
  - [ ] 9.2.4 Create CloudWatch dashboard deployment script
  - [ ] 9.2.5 Create `labs/lab_10_benchmarking_monitoring/README.md`

## Task 10: Module 8 - AWS Deep Dive

- [ ] 10.1 Create `08_aws_deep_dive.md`
  - [ ] 10.1.1 Create AWS instance selection guide flowchart
  - [ ] 10.1.2 Add EC2 + vLLM deployment script
  - [ ] 10.1.3 Write SageMaker LMI deployment section with Python code
  - [ ] 10.1.4 Add SageMaker autoscaling configuration code
  - [ ] 10.1.5 Write Inferentia2 deployment section with Neuron SDK examples
  - [ ] 10.1.6 Create Bedrock vs Self-Hosted comparison table
  - [ ] 10.1.7 Add AWS architecture patterns diagrams
  - [ ] 10.1.8 Write Key Takeaways section
- [ ] 10.2 Create Lab 9: SageMaker Production Deployment
  - [ ] 10.2.1 Create `labs/lab_09_sagemaker_production/notebook.ipynb`
  - [ ] 10.2.2 Implement DJLModel deployment with vLLM backend
  - [ ] 10.2.3 Add autoscaling policy configuration
  - [ ] 10.2.4 Create endpoint testing code
  - [ ] 10.2.5 Add cleanup instructions
  - [ ] 10.2.6 Create `labs/lab_09_sagemaker_production/README.md`

## Task 11: Module 9 - Structured Output and Guided Decoding

- [ ] 11.1 Create `09_structured_output_guided_decoding.md`
  - [ ] 11.1.1 Write guided decoding approaches diagram
  - [ ] 11.1.2 Add vLLM structured output code examples with Pydantic
  - [ ] 11.1.3 Add SGLang structured generation examples
  - [ ] 11.1.4 Write function calling patterns section
  - [ ] 11.1.5 Write Key Takeaways section

## Task 12: Module 10 - Edge Deployment (Optional)

- [ ] 12.1 Create `10_edge_deployment.md`
  - [ ] 12.1.1 Create edge deployment options comparison table
  - [ ] 12.1.2 Add llama.cpp deployment instructions
  - [ ] 12.1.3 Write GGUF quantization format explanation
  - [ ] 12.1.4 Add Apple Silicon / MLX section
  - [ ] 12.1.5 Write Key Takeaways section

## Task 13: Infrastructure Templates

- [ ] 13.1 Create CloudFormation templates
  - [ ] 13.1.1 Create `labs/infrastructure/cloudformation/g5-instance.yaml`
  - [ ] 13.1.2 Create `labs/infrastructure/cloudformation/p4d-instance.yaml`
  - [ ] 13.1.3 Create `labs/infrastructure/cloudformation/eks-cluster.yaml`
  - [ ] 13.1.4 Create `labs/infrastructure/cloudformation/sagemaker-endpoint.yaml`
- [ ] 13.2 Create CDK templates
  - [ ] 13.2.1 Create `labs/infrastructure/cdk/llm_inference_stack.py`
  - [ ] 13.2.2 Add EC2 GPU instance construct
  - [ ] 13.2.3 Add SageMaker endpoint construct
  - [ ] 13.2.4 Add EKS cluster construct

## Task 14: Reference Materials

- [ ] 14.1 Create cheat sheet
  - [ ] 14.1.1 Create `reference/cheat_sheet.md` with key formulas
  - [ ] 14.1.2 Add VRAM calculation quick reference
  - [ ] 14.1.3 Add optimization decision framework
  - [ ] 14.1.4 Add engine selection guide
- [ ] 14.2 Create vLLM quick reference
  - [ ] 14.2.1 Create `reference/vllm_quick_reference.md`
  - [ ] 14.2.2 Document all configuration options
  - [ ] 14.2.3 Add common configuration profiles
- [ ] 14.3 Create glossary
  - [ ] 14.3.1 Create `reference/glossary.md`
  - [ ] 14.3.2 Add all technical terms with definitions
- [ ] 14.4 Create cost calculator
  - [ ] 14.4.1 Create `reference/cost_calculator.py` or spreadsheet
  - [ ] 14.4.2 Add AWS instance cost data
  - [ ] 14.4.3 Add cost-per-token calculations

## Task 15: Workshop Delivery Materials

- [ ] 15.1 Create workshop outline
  - [ ] 15.1.1 Create `slides/workshop_outline.md` with timing estimates
  - [ ] 15.1.2 Add 2-day full workshop schedule
  - [ ] 15.1.3 Add 1-day condensed workshop schedule
  - [ ] 15.1.4 Add 2-hour deep dive session options
- [ ] 15.2 Create pre-workshop setup guide
  - [ ] 15.2.1 Create `reference/pre_workshop_setup.md`
  - [ ] 15.2.2 Add AWS account requirements
  - [ ] 15.2.3 Add software prerequisites
  - [ ] 15.2.4 Add HuggingFace token setup instructions
- [ ] 15.3 Create post-workshop resources
  - [ ] 15.3.1 Create `reference/post_workshop_resources.md`
  - [ ] 15.3.2 Add continued learning paths
  - [ ] 15.3.3 Add community resources and links

## Task 16: Testing and Validation

- [ ] 16.1 Create content validation tests
  - [ ] 16.1.1 Create `tests/test_module_content.py` for Property 1 (module completeness)
  - [ ] 16.1.2 Create `tests/test_lab_structure.py` for Property 2 (lab structure)
  - [ ] 16.1.3 Create `tests/test_code_quality.py` for Property 3 (code examples)
  - [ ] 16.1.4 Create `tests/test_diagrams.py` for Property 4 (Mermaid syntax)
- [ ] 16.2 Create calculation validation tests
  - [ ] 16.2.1 Create `tests/test_vram_calculator.py` for Property 5
  - [ ] 16.2.2 Add known model configuration test cases
  - [ ] 16.2.3 Add edge case tests
- [ ] 16.3 Create benchmark schema tests
  - [ ] 16.3.1 Create `tests/test_benchmark_schema.py` for Property 13
  - [ ] 16.3.2 Add sample benchmark output validation
