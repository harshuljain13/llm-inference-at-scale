# Module 6: Production Serving Architecture

> Deploying LLMs in production with Ray Serve, KServe, and llm-d

---

## Learning Objectives

By the end of this module, you will:

- Design production-ready LLM serving architectures
- Deploy with Ray Serve and KServe
- Understand disaggregated serving with llm-d
- Implement security best practices

---

## Production Serving Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                 PRODUCTION LLM SERVING STACK                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                      Load Balancer                          │   │
│   │              (ALB / NLB / Istio Gateway)                    │   │
│   └─────────────────────────────┬───────────────────────────────┘   │
│                                 │                                   │
│   ┌─────────────────────────────▼───────────────────────────────┐   │
│   │                      API Gateway                            │   │
│   │         (Rate limiting, Auth, Request routing)              │   │
│   └─────────────────────────────┬───────────────────────────────┘   │
│                                 │                                   │
│   ┌─────────────────────────────▼───────────────────────────────┐   │
│   │                   Inference Router                          │   │
│   │        (Model selection, A/B testing, Canary)               │   │
│   └─────────────────────────────┬───────────────────────────────┘   │
│                                 │                                   │
│         ┌───────────────────────┼───────────────────────┐           │
│         ▼                       ▼                       ▼           │
│   ┌───────────┐           ┌───────────┐           ┌───────────┐     │
│   │  Model A  │           │  Model B  │           │  Model C  │     │
│   │  Replica  │           │  Replica  │           │  Replica  │     │
│   │    1-N    │           │    1-N    │           │    1-N    │     │
│   └───────────┘           └───────────┘           └───────────┘     │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    Observability                            │   │
│   │     (Prometheus, Grafana, CloudWatch, Datadog)              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Ray Serve Deployment

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAY SERVE ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                     Ray Cluster                             │   │
│   │                                                             │   │
│   │   ┌─────────────┐                                           │   │
│   │   │  Head Node  │                                           │   │
│   │   │  • GCS      │                                           │   │
│   │   │  • Dashboard│                                           │   │
│   │   └─────────────┘                                           │   │
│   │                                                             │   │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │   │
│   │   │ Worker Node │  │ Worker Node │  │ Worker Node │         │   │
│   │   │   (GPU)     │  │   (GPU)     │  │   (GPU)     │         │   │
│   │   │             │  │             │  │             │         │   │
│   │   │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │         │   │
│   │   │ │ Replica │ │  │ │ Replica │ │  │ │ Replica │ │         │   │
│   │   │ │  (vLLM) │ │  │ │  (vLLM) │ │  │ │  (vLLM) │ │         │   │
│   │   │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │         │   │
│   │   └─────────────┘  └─────────────┘  └─────────────┘         │   │
│   │                                                             │   │
│   │   ┌─────────────────────────────────────────────────────┐   │   │
│   │   │              Ray Serve Controller                   │   │   │
│   │   │  • Autoscaling                                      │   │   │
│   │   │  • Health checks                                    │   │   │
│   │   │  • Traffic routing                                  │   │   │
│   │   └─────────────────────────────────────────────────────┘   │   │
│   │                                                             │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Ray Serve vLLM Deployment

```python
from ray import serve
from vllm import LLM, SamplingParams
from starlette.requests import Request
import json

@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_num_ongoing_requests_per_replica": 10,
    },
)
class VLLMDeployment:
    def __init__(self):
        self.llm = LLM(
            model="meta-llama/Llama-3.1-8B-Instruct",
            gpu_memory_utilization=0.9,
            max_num_seqs=64,
            enable_chunked_prefill=True,
        )
        self.default_sampling = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
        )

    async def __call__(self, request: Request) -> dict:
        body = await request.json()
        prompt = body.get("prompt", "")

        # Override sampling params if provided
        sampling = SamplingParams(
            temperature=body.get("temperature", 0.7),
            top_p=body.get("top_p", 0.9),
            max_tokens=body.get("max_tokens", 256),
        )

        outputs = self.llm.generate([prompt], sampling)

        return {
            "text": outputs[0].outputs[0].text,
            "usage": {
                "prompt_tokens": len(outputs[0].prompt_token_ids),
                "completion_tokens": len(outputs[0].outputs[0].token_ids),
            }
        }


# Deploy
deployment = VLLMDeployment.bind()
serve.run(deployment, host="0.0.0.0", port=8000)
```

### Multi-Model Deployment

```python
from ray import serve

@serve.deployment(ray_actor_options={"num_gpus": 1})
class SmallModel:
    def __init__(self):
        self.llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

    async def __call__(self, request):
        # Handle request
        pass

@serve.deployment(ray_actor_options={"num_gpus": 4})
class LargeModel:
    def __init__(self):
        self.llm = LLM(
            model="meta-llama/Llama-3.1-70B-Instruct",
            tensor_parallel_size=4
        )

    async def __call__(self, request):
        # Handle request
        pass

@serve.deployment
class Router:
    def __init__(self, small_model, large_model):
        self.small = small_model
        self.large = large_model

    async def __call__(self, request):
        body = await request.json()
        model = body.get("model", "small")

        if model == "large":
            return await self.large.remote(request)
        return await self.small.remote(request)


# Compose deployments
small = SmallModel.bind()
large = LargeModel.bind()
router = Router.bind(small, large)

serve.run(router, host="0.0.0.0", port=8000)
```

---

## KServe Deployment

### InferenceService Definition

```yaml
# kserve-llm.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama-3-8b
  namespace: llm-serving
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 4
    scaleTarget: 10 # Target concurrent requests
    scaleMetric: concurrency

    containers:
      - name: kserve-container
        image: vllm/vllm-openai:latest
        args:
          - --model
          - meta-llama/Llama-3.1-8B-Instruct
          - --gpu-memory-utilization
          - "0.9"
          - --max-num-seqs
          - "64"
          - --enable-chunked-prefill

        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
          requests:
            nvidia.com/gpu: 1
            memory: 24Gi

        env:
          - name: HF_TOKEN
            valueFrom:
              secretKeyRef:
                name: huggingface-secret
                key: token

        ports:
          - containerPort: 8000
            protocol: TCP

        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10

        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 30
```

### Autoscaling Configuration

```yaml
# kserve-autoscaling.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama-3-8b-autoscale
  annotations:
    # Knative autoscaling annotations
    autoscaling.knative.dev/class: kpa.autoscaling.knative.dev
    autoscaling.knative.dev/metric: concurrency
    autoscaling.knative.dev/target: "10"
    autoscaling.knative.dev/minScale: "1"
    autoscaling.knative.dev/maxScale: "10"
    # Scale down delay
    autoscaling.knative.dev/scaleDownDelay: "5m"
spec:
  predictor:
    containers:
      - name: kserve-container
        image: vllm/vllm-openai:latest
        # ... rest of config
```

### Canary Deployment

```yaml
# kserve-canary.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama-canary
spec:
  predictor:
    # Production model (90% traffic)
    canaryTrafficPercent: 10
    containers:
      - name: kserve-container
        image: vllm/vllm-openai:latest
        args:
          - --model
          - meta-llama/Llama-3.1-8B-Instruct

  # Canary model (10% traffic)
  canary:
    predictor:
      containers:
        - name: kserve-container
          image: vllm/vllm-openai:latest
          args:
            - --model
            - meta-llama/Llama-3.1-8B-Instruct
            - --quantization
            - awq # Testing quantized version
```

---

## Disaggregated Serving with llm-d

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM-D ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Traditional (Aggregated):                                         │
│   ═════════════════════════                                         │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  GPU Worker                                                 │   │
│   │  ┌─────────────────────────────────────────────────────┐    │   │
│   │  │  Prefill + Decode (same GPU)                        │    │   │
│   │  │  • Prefill: compute-bound, bursty                   │    │   │
│   │  │  • Decode: memory-bound, steady                     │    │   │
│   │  │  • Resource contention!                             │    │   │
│   │  └─────────────────────────────────────────────────────┘    │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   Disaggregated (llm-d):                                            │
│   ══════════════════════                                            │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                      Router                                 │   │
│   └─────────────────────────────┬───────────────────────────────┘   │
│                                 │                                   │
│         ┌───────────────────────┴───────────────────────┐           │
│         ▼                                               ▼           │
│   ┌─────────────────────┐                   ┌─────────────────────┐ │
│   │   Prefill Workers   │                   │   Decode Workers    │ │
│   │   (Compute-optimized)│                  │   (Memory-optimized)│ │
│   │                     │                   │                     │ │
│   │   • High compute    │   KV Cache        │   • High bandwidth  │ │
│   │   • Bursty workload │ ───Transfer───►   │   • Steady workload │ │
│   │   • Scale for TTFT  │                   │   • Scale for ITL   │ │
│   │                     │                   │                     │ │
│   └─────────────────────┘                   └─────────────────────┘ │
│                                                                     │
│   Benefits:                                                         │
│   • Independent scaling of prefill and decode                       │
│   • Better resource utilization                                     │
│   • Optimized hardware for each phase                               │
│   • Lower tail latency                                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### When to Use Disaggregated Serving

```
┌─────────────────────────────────────────────────────────────────────┐
│           DISAGGREGATED SERVING DECISION GUIDE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   USE disaggregated serving when:                                   │
│   ✓ High request volume with variable prompt lengths                │
│   ✓ Strict latency SLAs (both TTFT and ITL)                         │
│   ✓ Cost optimization is critical                                   │
│   ✓ Have infrastructure for KV cache transfer                       │
│                                                                     │
│   DON'T USE disaggregated serving when:                             │
│   ✗ Low request volume                                              │
│   ✗ Simple deployment requirements                                  │
│   ✗ Limited infrastructure complexity budget                        │
│   ✗ Uniform prompt lengths                                          │
│                                                                     │
│   Typical Improvements:                                             │
│   • 20-40% better GPU utilization                                   │
│   • 30-50% lower P99 latency                                        │
│   • 15-25% cost reduction at scale                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Patterns

### Single Model Deployment

```
┌─────────────────────────────────────────────────────────────────────┐
│                 SINGLE MODEL DEPLOYMENT                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    Load Balancer                            │   │
│   └─────────────────────────────┬───────────────────────────────┘   │
│                                 │                                   │
│         ┌───────────────────────┼───────────────────────┐           │
│         ▼                       ▼                       ▼           │
│   ┌───────────┐           ┌───────────┐           ┌───────────┐     │
│   │ Replica 1 │           │ Replica 2 │           │ Replica N │     │
│   │  (GPU)    │           │  (GPU)    │           │  (GPU)    │     │
│   └───────────┘           └───────────┘           └───────────┘     │
│                                                                     │
│   Use case: Single model, horizontal scaling                        │
│   Scaling: Add replicas based on load                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Multi-Model Deployment

```
┌─────────────────────────────────────────────────────────────────────┐
│                 MULTI-MODEL DEPLOYMENT                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    API Gateway                              │   │
│   │              (Route by model parameter)                     │   │
│   └─────────────────────────────┬───────────────────────────────┘   │
│                                 │                                   │
│         ┌───────────────────────┼───────────────────────┐           │
│         ▼                       ▼                       ▼           │
│   ┌───────────┐           ┌───────────┐           ┌───────────┐     │
│   │  Model A  │           │  Model B  │           │  Model C  │     │
│   │  (8B)     │           │  (70B)    │           │  (8B-FT)  │     │
│   │  1 GPU    │           │  4 GPUs   │           │  1 GPU    │     │
│   └───────────┘           └───────────┘           └───────────┘     │
│                                                                     │
│   Use case: Different models for different tasks                    │
│   Routing: Based on request parameter or endpoint                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### A/B Testing / Canary

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CANARY DEPLOYMENT                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                  Traffic Splitter                           │   │
│   │              (90% stable, 10% canary)                       │   │
│   └─────────────────────────────┬───────────────────────────────┘   │
│                                 │                                   │
│               ┌─────────────────┴─────────────────┐                 │
│               ▼                                   ▼                 │
│   ┌───────────────────────┐           ┌───────────────────────┐     │
│   │    Stable (90%)       │           │    Canary (10%)       │     │
│   │    Model v1.0         │           │    Model v1.1         │     │
│   │    FP16               │           │    INT8 (testing)     │     │
│   └───────────────────────┘           └───────────────────────┘     │
│                                                                     │
│   Use case: Safe rollout of new models/configs                      │
│   Metrics: Compare latency, quality, errors                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Security Considerations

### Security Checklist

```
┌─────────────────────────────────────────────────────────────────────┐
│                 LLM SERVING SECURITY CHECKLIST                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Authentication & Authorization:                                   │
│   ☐ API key or OAuth2 authentication                                │
│   ☐ Role-based access control (RBAC)                                │
│   ☐ Per-user/team rate limiting                                     │
│   ☐ Audit logging of all requests                                   │
│                                                                     │
│   Network Security:                                                 │
│   ☐ TLS/HTTPS for all endpoints                                     │
│   ☐ VPC isolation for inference servers                             │
│   ☐ Security groups limiting ingress                                │
│   ☐ Private endpoints where possible                                │
│                                                                     │
│   Input Validation:                                                 │
│   ☐ Maximum prompt length limits                                    │
│   ☐ Input sanitization                                              │
│   ☐ Prompt injection detection                                      │
│   ☐ Content filtering (if required)                                 │
│                                                                     │
│   Output Safety:                                                    │
│   ☐ Output content filtering                                        │
│   ☐ PII detection and redaction                                     │
│   ☐ Response length limits                                          │
│                                                                     │
│   Infrastructure:                                                   │
│   ☐ Secrets management (HF tokens, API keys)                        │
│   ☐ Container image scanning                                        │
│   ☐ Regular security updates                                        │
│   ☐ Backup and disaster recovery                                    │
│                                                                     │
│   Monitoring:                                                       │
│   ☐ Anomaly detection on request patterns                           │
│   ☐ Cost monitoring and alerts                                      │
│   ☐ Error rate monitoring                                           │
│   ☐ Latency SLA monitoring                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Example: Secure API Gateway Configuration

```yaml
# api-gateway-config.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: llm-api-gateway
  annotations:
    # Rate limiting
    nginx.ingress.kubernetes.io/limit-rps: "100"
    nginx.ingress.kubernetes.io/limit-connections: "50"

    # Request size limits
    nginx.ingress.kubernetes.io/proxy-body-size: "1m"

    # Timeouts
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"

    # SSL
    nginx.ingress.kubernetes.io/ssl-redirect: "true"

    # Auth (external auth service)
    nginx.ingress.kubernetes.io/auth-url: "https://auth.example.com/validate"
    nginx.ingress.kubernetes.io/auth-signin: "https://auth.example.com/login"
spec:
  tls:
    - hosts:
        - llm-api.example.com
      secretName: llm-api-tls
  rules:
    - host: llm-api.example.com
      http:
        paths:
          - path: /v1
            pathType: Prefix
            backend:
              service:
                name: llm-service
                port:
                  number: 8000
```

---

## Key Takeaways

1. **Ray Serve for flexibility** - Easy autoscaling, multi-model, Python-native

2. **KServe for Kubernetes** - Standard interface, built-in autoscaling, canary support

3. **Disaggregated serving for scale** - Separate prefill/decode for better utilization

4. **Security is non-negotiable** - Auth, rate limiting, input validation, monitoring

5. **Start simple, scale up** - Single model → Multi-model → Disaggregated

6. **Monitor everything** - Latency, throughput, errors, costs

---

## Lab Preview

### Lab 7: Ray Serve Deployment

- Deploy vLLM with Ray Serve
- Configure autoscaling
- Implement load testing

### Lab 8: EKS + KServe Deployment

- Set up EKS cluster with GPU nodes
- Deploy KServe InferenceService
- Configure autoscaling and canary

---

## References

1. Ray Serve Documentation: https://docs.ray.io/en/latest/serve/
2. KServe Documentation: https://kserve.github.io/website/
3. llm-d Paper: "Disaggregated LLM Inference" (2024)
4. OWASP LLM Security Guidelines
