# 9.3 Enterprise RAG Service: Multi-Model Knowledge Assistant

## Why This Design Matters

Enterprise Retrieval-Augmented Generation (RAG) represents the most common production LLM deployment pattern today. Unlike consumer chatbots optimized for a single interaction style, enterprise RAG must simultaneously handle diverse query types (factual lookup, summarization, analysis, code generation), operate under strict accuracy requirements (wrong answers erode trust faster than slow answers), and do so within budgets that make CFOs comfortable.

This system design tackles the hardest variant: a mixed-workload enterprise RAG service that combines real-time interactive chat with batch document processing, uses multiple models of different sizes to balance quality against cost, and serves thousands of concurrent users against a corpus of millions of documents. Every decision here, from hardware selection to caching strategy, flows from one core tension: **accuracy demands large models, but cost and latency demand small ones**.

The architecture we develop solves this with a model cascade, semantic caching, and intelligent routing, achieving >90% domain QA accuracy at <$0.01 per query and <2 second end-to-end latency.

---

## Part 1: Requirements Analysis

### Functional Requirements

The system serves as the primary knowledge interface for a large enterprise:

**Document Corpus:**
- 10 million documents spanning internal wikis, technical documentation, policy manuals, code repositories, Slack archives, and email threads
- Documents range from 100 tokens (Slack messages) to 50,000 tokens (technical specifications)
- Average document length: ~2,000 tokens
- Total corpus size: ~20 billion tokens of raw text
- After chunking (512-token chunks with 64-token overlap): ~45 million chunks

**User Population:**
- 5,000 employees querying simultaneously during peak hours (9 AM - 5 PM across time zones)
- Average queries per user per day: 15-25
- Peak QPS: ~200 queries/second (5,000 users × 2.5 queries/minute during bursts)
- Sustained QPS: ~50 queries/second during business hours
- Off-peak QPS: ~5 queries/second

**Query Types (distribution from production logs):**
- Factual lookup (40%): "What is our PTO policy for employees with 5+ years?"
- Summarization (25%): "Summarize the Q3 architecture review decisions"
- Analysis (20%): "Compare our authentication approach vs. the NIST recommendations"
- Code/technical (15%): "Show me how to configure the payment gateway retry logic"

### Non-Functional Requirements

| Requirement | Target | Rationale |
|---|---|---|
| End-to-end latency (P50) | <1.5s | Conversational UX expectation |
| End-to-end latency (P99) | <3.0s | Users abandon after 5s |
| Accuracy (domain QA) | >90% | Below this, users revert to manual search |
| Faithfulness | >95% | Hallucinated enterprise info is dangerous |
| Monthly budget | <$50,000 | Must justify ROI vs. hiring more support staff |
| Availability | 99.9% | Business-critical during work hours |
| Batch indexing | Complete nightly | New docs searchable within 24 hours |
| Security | Document-level ACL | Users only see docs they have permission to access |

### Latency Budget Breakdown

The 2-second end-to-end budget must be distributed across the pipeline stages:

```
Query arrives ──► Embedding (50ms) ──► Vector Search (100ms) ──► Reranking (150ms)
     ──► Context Assembly (20ms) ──► LLM Generation (1,200ms) ──► Post-processing (80ms)
     Total: ~1,600ms (P50), leaving 400ms of headroom
```

This budget immediately constrains our architecture: generation gets 60% of the total time budget. Any caching strategy that eliminates the generation step saves 1.2 seconds, making the response feel instant.

### Query Complexity Classification

Not all queries need a 70B model. We define complexity tiers:

| Tier | Examples | Model | Expected Accuracy |
|---|---|---|---|
| Simple | Policy lookups, definitions, dates | 8B | 92% |
| Medium | Comparisons, multi-doc synthesis | 70B | 94% |
| Complex | Analysis, recommendations, novel questions | 70B + chain-of-thought | 91% |

The router's job is to classify incoming queries into these tiers with >85% accuracy. Misclassification costs either quality (sending complex queries to 8B) or money (sending simple queries to 70B).

---

## Part 2: Model Selection

### The Model Cascade Architecture

Rather than deploying a single model, we use a three-model architecture where each model serves a distinct role:

**Primary Generation: Llama 3.1 70B (INT4)**
- Role: High-quality answers for complex queries
- Why 70B: Enterprise QA requires reasoning over multiple document chunks simultaneously. 8B models struggle with multi-hop reasoning ("Find the policy, then apply it to this specific scenario").
- Why INT4: Full FP16 requires 140GB (two A100-80GB). INT4 fits in 35GB, enabling deployment on 4×A10G GPUs with room for KV cache.
- Quality tradeoff: INT4 quantization loses ~1-2% on MMLU benchmarks, but enterprise QA (factual, grounded in context) is less sensitive to quantization than creative tasks.

**Fast Generation: Llama 3.1 8B (FP16)**
- Role: Simple lookups, factual questions with single-source answers
- Why 8B: 40% of queries are simple lookups where 8B achieves comparable accuracy to 70B (within 2%) when provided with the correct retrieved context.
- Why FP16: At 16GB, 8B in FP16 fits comfortably on a single A10G with abundant KV cache space. No need to sacrifice quality via quantization.
- Latency advantage: 8B generates at 80-120 tokens/second on A10G vs. 20-30 tokens/second for 70B INT4 across 4 GPUs.

**Embedding Model: BGE-Large-en-v1.5 (FP16)**
- Role: Query and document embedding for vector retrieval
- Parameters: 335M (0.67GB in FP16)
- Embedding dimension: 1024
- Why this model: Top-5 on MTEB leaderboard at time of deployment, excellent cost/quality tradeoff for enterprise text.
- Throughput: ~500 embeddings/second on A10G (batched), ~2000/second on CPU with ONNX optimization.

### Model Cascade Decision Logic

```python
class QueryRouter:
    """Routes queries to appropriate model based on complexity signals."""

    def __init__(self, complexity_classifier, confidence_threshold=0.75):
        self.classifier = complexity_classifier  # Fine-tuned DistilBERT
        self.confidence_threshold = confidence_threshold
        self.escalation_threshold = 0.6  # 8B confidence below this triggers 70B

    def route(self, query: str, retrieved_chunks: list[str]) -> ModelTier:
        # Step 1: Classify query complexity
        complexity = self.classifier.predict(query)

        if complexity == "simple":
            return ModelTier.SMALL  # 8B handles it

        if complexity == "complex":
            return ModelTier.LARGE  # Skip 8B entirely

        # Step 2: For "medium" queries, try 8B first
        return ModelTier.CASCADE  # 8B first, escalate if low confidence

    def should_escalate(self, response: str, query: str) -> bool:
        """Check if 8B response needs escalation to 70B."""
        signals = [
            self._check_hedging_language(response),  # "I'm not sure", "might be"
            self._check_response_length(response),    # Very short = possibly incomplete
            self._check_source_coverage(response),    # Did it use multiple chunks?
            self._check_contradiction(response, query) # Internal consistency
        ]
        confidence = 1.0 - (sum(signals) / len(signals))
        return confidence < self.escalation_threshold
```

The cascade saves significant cost: if 40% of queries are simple (handled by 8B at 1/10th the cost) and another 20% of medium queries succeed on 8B without escalation, we route only 40-50% of queries to the expensive 70B model.

### Cost Per Query by Model

| Model | GPU Cost/Hour | Tokens/Second | Avg Response (500 tok) | Cost/Query |
|---|---|---|---|---|
| 70B INT4 (4×A10G) | $4.00/hr | 25 tok/s | 20s generation | $0.022 |
| 8B FP16 (1×A10G) | $1.00/hr | 100 tok/s | 5s generation | $0.0014 |
| Embedding (1×A10G) | $1.00/hr | 500 emb/s | 2ms/query | $0.0000006 |

With the cascade routing 50% to 8B and 50% to 70B, the blended generation cost is:
```
0.50 × $0.0014 + 0.50 × $0.022 = $0.0117/query
```

Adding retrieval, reranking, and infrastructure overhead brings us to ~$0.015/query, well under the $0.01 target if we achieve even modest cache hit rates (see Part 7).

---

## Part 3: Memory Budget

### GPU Memory Allocation

Memory planning is the most critical constraint in multi-model deployments. A single miscalculation leads to OOM kills during peak load.

**70B INT4 Model (deployed across 4×A10G-24GB GPUs):**

```
Model weights (INT4):
  70B parameters × 4 bits/param = 35 GB
  Distributed across 4 GPUs: 8.75 GB/GPU

KV Cache per GPU:
  Layers per GPU: 80 / 4 = 20 layers
  KV size per layer per token: 2 × 8192 × 2 bytes = 32 KB (FP16 values)
  Per-token KV across 20 layers: 20 × 32 KB = 640 KB
  Max sequence length: 8,192 tokens (context window)
  Max concurrent sequences: 8
  Total KV per GPU: 8 × 8,192 × 640 KB = 41.9 GB ← EXCEEDS BUDGET

  ──► Must limit: either sequences or context length
  Practical limit: 4 concurrent sequences × 4,096 context = 10.5 GB/GPU
```

**Revised 70B memory layout per GPU (24GB A10G):**

| Component | Memory | % of 24GB |
|---|---|---|
| Model weights (sharded) | 8.75 GB | 36.5% |
| KV cache (4 seq × 4K ctx) | 10.5 GB | 43.7% |
| Activations + workspace | 2.0 GB | 8.3% |
| CUDA overhead + fragmentation | 2.75 GB | 11.5% |
| **Total** | **24.0 GB** | **100%** |

This gives us 4 concurrent requests with 4K context. For our RAG use case, 4K context is sufficient: system prompt (~200 tokens) + retrieved chunks (5 chunks × 512 tokens = 2,560 tokens) + query (~100 tokens) + generation budget (1,140 tokens) = 4,000 tokens.

**8B FP16 Model (single A10G-24GB GPU):**

```
Model weights (FP16):
  8B parameters × 2 bytes = 16 GB

KV Cache:
  Layers: 32
  KV size per layer per token: 2 × 4096 × 2 bytes = 16 KB
  Per-token KV across 32 layers: 512 KB
  Concurrent sequences: 16
  Context per sequence: 4,096
  Total KV: 16 × 4,096 × 512 KB = 33.6 GB ← EXCEEDS

  Practical limit: 8 sequences × 4,096 = 16.8 GB ← Still exceeds
  Final: 8 sequences × 2,048 context = 8.4 GB
```

**8B memory layout (24GB A10G):**

| Component | Memory | % of 24GB |
|---|---|---|
| Model weights | 16.0 GB | 66.7% |
| KV cache (8 seq × 2K ctx) | 4.2 GB | 17.5% |
| Activations + workspace | 1.5 GB | 6.25% |
| CUDA overhead | 2.3 GB | 9.6% |
| **Total** | **24.0 GB** | **100%** |

Wait, 8 sequences × 2K context is tight for RAG. Let us reconsider: with 8B, we can use INT8 quantization (negligible quality loss at this scale):

**8B INT8 Model (optimized layout):**

| Component | Memory | % of 24GB |
|---|---|---|
| Model weights (INT8) | 8.0 GB | 33.3% |
| KV cache (16 seq × 4K ctx) | 8.4 GB | 35.0% |
| Activations + workspace | 2.0 GB | 8.3% |
| CUDA overhead | 2.6 GB | 10.8% |
| Prefix cache (shared) | 3.0 GB | 12.5% |
| **Total** | **24.0 GB** | **100%** |

This is much better: 16 concurrent sequences with full 4K context, plus 3GB dedicated to prefix caching (shared system prompt + common document prefixes).

**Embedding Model (shared GPU or CPU):**

| Component | Memory |
|---|---|
| Model weights (FP16) | 0.67 GB |
| Batch buffer (128 queries) | 0.5 GB |
| ONNX runtime overhead | 0.3 GB |
| **Total** | **1.47 GB** |

The embedding model is small enough to co-locate on the 8B model's GPU (if using INT8) or run on CPU with ONNX optimization.

### Total GPU Fleet Memory Summary

| Component | GPUs | Type | Total VRAM |
|---|---|---|---|
| 70B INT4 generation | 4 | A10G-24GB | 96 GB |
| 8B INT8 generation (×2 replicas) | 2 | A10G-24GB | 48 GB |
| Embedding + Reranker | 1 | A10G-24GB | 24 GB |
| **Total** | **7** | | **168 GB** |

---

## Part 4: Hardware Selection

### Instance Type Analysis

AWS provides several GPU instance families. For enterprise RAG, we need a mix optimized for different workload characteristics:

**Generation Tier (70B model):**

| Instance | GPUs | VRAM | vCPUs | RAM | Cost/hr | Fit? |
|---|---|---|---|---|---|---|
| g5.12xlarge | 4×A10G | 96GB | 48 | 192GB | $5.67 | ✅ Perfect |
| g5.48xlarge | 8×A10G | 192GB | 192 | 768GB | $16.29 | Overkill |
| p4d.24xlarge | 8×A100 | 640GB | 96 | 1.1TB | $32.77 | Way overkill |
| g6.12xlarge | 4×L4 | 96GB | 48 | 192GB | $4.60 | ✅ 19% cheaper |

**Decision**: g5.12xlarge for 70B. The g6 (L4 GPUs) offers 19% cost savings but L4 has lower memory bandwidth (300 GB/s vs. A10G's 600 GB/s), which directly impacts token generation speed for large models. For the 70B model where we are memory-bandwidth-bound, A10G wins.

**Fast Generation Tier (8B model):**

| Instance | GPUs | VRAM | Cost/hr | Throughput | $/1K tokens |
|---|---|---|---|---|---|
| g5.xlarge | 1×A10G | 24GB | $1.01 | 100 tok/s | $0.0028 |
| g6.xlarge | 1×L4 | 24GB | $0.80 | 85 tok/s | $0.0026 |
| g5.2xlarge | 1×A10G | 24GB | $1.21 | 100 tok/s | $0.0034 |

**Decision**: g5.xlarge for 8B. Minimal instance with full A10G access. We deploy 2 replicas for redundancy and throughput.

**Retrieval Tier (Embedding + Reranking):**

Embeddings are compute-bound (matrix multiplications on short sequences) rather than memory-bandwidth-bound. Options:

| Approach | Instance | Cost/hr | Throughput | Latency |
|---|---|---|---|---|
| GPU embedding | g5.xlarge | $1.01 | 500 emb/s | 2ms |
| CPU embedding (ONNX) | c6i.4xlarge | $0.68 | 200 emb/s | 5ms |
| CPU embedding (ONNX) | c6i.8xlarge | $1.36 | 400 emb/s | 2.5ms |

**Decision**: One g5.xlarge shared between embedding model and cross-encoder reranker. At 200 QPS peak, we need to embed 200 queries/second. A single A10G handles this at 500 emb/s with headroom. The reranker (scoring 20 chunks per query) needs 4,000 scores/second, which fits on the same GPU when batched.

### Cost Breakdown: Monthly Infrastructure

| Component | Instance | Count | Hours/Month | Cost/Month |
|---|---|---|---|---|
| 70B Generation | g5.12xlarge | 1 | 730 (24/7) | $4,139 |
| 8B Generation | g5.xlarge | 2 | 730 (24/7) | $1,475 |
| Embedding + Rerank | g5.xlarge | 1 | 730 (24/7) | $738 |
| Vector DB (OpenSearch) | r6g.2xlarge | 3 | 730 (24/7) | $2,847 |
| Batch Indexing | g5.xlarge (spot) | 2 | 240 (8hr/night) | $242 |
| Load Balancer + API | m6i.xlarge | 2 | 730 (24/7) | $280 |
| Storage (S3 + EBS) | Various | - | - | $500 |
| **Total** | | | | **$10,221** |

We are well under the $50K/month budget at $10.2K, leaving room for scaling during traffic spikes and adding redundancy.

### Heterogeneous Fleet Management

The key insight is that different pipeline stages have fundamentally different compute profiles:

```
Embedding:    Compute-bound, short sequences, high batch parallelism
Reranking:    Compute-bound, pair-wise scoring, moderate batch
8B Generation: Memory-bandwidth-bound, autoregressive, moderate concurrency
70B Generation: Memory-bandwidth-bound, autoregressive, low concurrency
Batch Indexing: Throughput-optimized, no latency requirement, spot-tolerant
```

This heterogeneity justifies a mixed fleet rather than uniform large instances.

---

## Part 5: Parallelism Strategy

### 70B Model: Tensor Parallelism Across 4 GPUs

The 70B model requires distribution across 4 A10G GPUs on a single g5.12xlarge instance. Tensor Parallelism (TP) is the only viable strategy here because:

1. **Pipeline Parallelism** adds latency (each stage must complete before the next starts), unacceptable for interactive RAG.
2. **Data Parallelism** requires the full model on each GPU (impossible at 35GB on 24GB GPUs).
3. **Tensor Parallelism** splits each layer's computation across GPUs, maintaining single-sequence latency while distributing memory.

**TP=4 Layout for Llama 70B:**

```
Layer Structure (80 transformer layers):
Each layer distributed across 4 GPUs:

GPU 0: heads 0-19  (20/80 attention heads)
GPU 1: heads 20-39
GPU 2: heads 40-59
GPU 3: heads 60-79

MLP split (each GPU handles 1/4 of intermediate dimension):
Full intermediate: 28,672
Per GPU: 7,168

All-reduce after each attention + MLP block:
  NVLink bandwidth: 600 GB/s (g5.12xlarge uses PCIe, not NVLink)
  PCIe Gen4 bandwidth: 32 GB/s bidirectional
  All-reduce per layer: 2 × hidden_size × 2 bytes = 2 × 8192 × 2 = 32 KB
  80 layers × 32 KB × 2 (attn + MLP) = 5.12 MB per token

  At PCIe bandwidth: 5.12 MB / 32 GB/s = 0.16 ms per generated token
  This adds ~13ms for a full 80-token response ← acceptable
```

**Critical Note on NVLink vs PCIe**: The g5.12xlarge connects its 4 A10G GPUs via PCIe, not NVLink. This limits all-reduce bandwidth to ~32 GB/s vs. NVLink's 600 GB/s. For 70B with TP=4, this works because per-layer communication is small (32KB). For larger models or TP=8, this would become a bottleneck.

**vLLM Configuration for 70B:**

```python
# vLLM engine configuration for 70B INT4 on g5.12xlarge
engine_config = {
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "quantization": "awq",  # 4-bit AWQ quantization
    "tensor_parallel_size": 4,
    "max_model_len": 4096,
    "max_num_seqs": 4,       # Limited by KV cache budget
    "gpu_memory_utilization": 0.92,
    "enable_prefix_caching": True,
    "block_size": 16,
    "swap_space": 4,  # GB of CPU swap for KV cache overflow
    "enforce_eager": False,  # Use CUDA graphs for speed
}
```

### 8B Model: No Parallelism Required

The 8B model in INT8 (8GB weights) fits entirely on one A10G with room for 16 concurrent sequences. No parallelism needed.

**Scaling strategy**: Horizontal replication. Two g5.xlarge instances each running an independent 8B model, with a load balancer distributing requests. This provides:
- 2× throughput (32 concurrent sequences total)
- Fault tolerance (one instance can handle full load during rolling updates)
- Simple scaling (add more replicas as query volume grows)

```python
# vLLM configuration for 8B INT8
engine_config_8b = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "quantization": "gptq",  # INT8 quantization
    "tensor_parallel_size": 1,
    "max_model_len": 4096,
    "max_num_seqs": 16,
    "gpu_memory_utilization": 0.90,
    "enable_prefix_caching": True,
    "block_size": 16,
    "max_num_batched_tokens": 16384,  # Aggressive batching
}
```

### Embedding Model: Batched Inference

The embedding model processes queries and (during indexing) document chunks. Two modes:

**Real-time query embedding (latency-sensitive):**
- Batch size: 1-8 (as queries arrive)
- Target: <50ms per query
- Strategy: Dynamic batching with 10ms wait window

**Batch document indexing (throughput-sensitive):**
- Batch size: 128-256
- Target: Process 45M chunks in <8 hours
- Required throughput: 45M / (8 × 3600) = 1,562 chunks/second
- Single A10G at batch=128: ~500 embeddings/second
- Need 3-4 GPUs during batch indexing (use spot instances)

```python
# Triton Inference Server config for embedding model
config = {
    "model": "BAAI/bge-large-en-v1.5",
    "instance_group": [
        {"count": 1, "kind": "KIND_GPU"},
    ],
    "dynamic_batching": {
        "preferred_batch_size": [32, 64, 128],
        "max_queue_delay_microseconds": 10000,  # 10ms batching window
    },
    "optimization": {
        "execution_accelerators": {
            "gpu_execution_accelerator": [{"name": "tensorrt"}]
        }
    }
}
```

---

## Part 6: Serving Architecture

### Three-Tier Pipeline Design

The serving architecture implements a staged pipeline where each tier can be independently scaled, cached, and monitored:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Enterprise RAG Pipeline                             │
│                                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌─────────────────┐   │
│  │  Query   │───►│  Retrieval   │───►│ Reranker │───►│   Generation    │   │
│  │  Router  │    │  (Emb + VDB) │    │ (Cross-  │    │  (8B or 70B)    │   │
│  │          │    │              │    │  Encoder) │    │                 │   │
│  └──────────┘    └──────────────┘    └──────────┘    └─────────────────┘   │
│       │                                                       │             │
│       │          ┌──────────────┐                             │             │
│       └─────────►│ Semantic     │◄────────────────────────────┘             │
│                  │ Cache        │   (cache miss → generate → store)         │
│                  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tier 1: Query Processing & Retrieval

When a query arrives, the system executes these steps in sequence:

**Step 1: Query Understanding (5ms)**
- Intent classification (lookup / summarize / analyze / code)
- Complexity estimation (simple / medium / complex)
- Query expansion: generate 2-3 paraphrases for better recall

**Step 2: Embedding Generation (50ms)**
- Encode query into 1024-dim vector using BGE-Large
- Apply query prefix instruction: "Represent this sentence for searching relevant passages:"
- Normalize to unit vector for cosine similarity

**Step 3: Vector Search (100ms)**
- Search OpenSearch with k-NN plugin (HNSW index)
- Retrieve top-50 candidates using approximate nearest neighbor
- Apply document-level ACL filter (user can only see permitted documents)
- Parameters: ef_search=256, providing 95%+ recall

**Step 4: Hybrid Search Fusion (20ms)**
- Combine vector results with BM25 keyword search
- Reciprocal Rank Fusion (RRF) with k=60:
  ```
  RRF_score(d) = Σ 1/(k + rank_i(d)) for each retrieval method i
  ```
- Hybrid search catches keyword-specific queries that semantic search misses (e.g., exact error codes, policy numbers)

```python
class RetrievalTier:
    """Tier 1: Retrieve candidate chunks from vector DB."""

    def __init__(self, opensearch_client, embedding_model):
        self.vector_db = opensearch_client
        self.embedder = embedding_model
        self.bm25_weight = 0.3
        self.vector_weight = 0.7

    async def retrieve(self, query: str, user_id: str, top_k: int = 50) -> list[Chunk]:
        # Parallel: vector search + BM25 keyword search
        query_embedding = await self.embedder.encode(query)

        vector_results, bm25_results = await asyncio.gather(
            self.vector_db.knn_search(
                vector=query_embedding,
                k=top_k,
                filter=self._build_acl_filter(user_id),
                params={"ef_search": 256}
            ),
            self.vector_db.bm25_search(
                query=query,
                size=top_k,
                filter=self._build_acl_filter(user_id)
            )
        )

        # Reciprocal Rank Fusion
        fused = self._rrf_fusion(vector_results, bm25_results, k=60)
        return fused[:top_k]

    def _rrf_fusion(self, *result_lists, k=60) -> list[Chunk]:
        scores = defaultdict(float)
        for results in result_lists:
            for rank, chunk in enumerate(results):
                scores[chunk.id] += 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: -x[1])
```

### Tier 2: Reranking

The reranker applies a cross-encoder model that sees both query and document together (unlike the bi-encoder used for initial retrieval):

**Cross-Encoder Reranking (150ms):**
- Model: bge-reranker-v2-m3 (568M parameters)
- Input: (query, chunk) pairs for top-50 candidates
- Output: relevance score 0-1 for each pair
- Select top-5 chunks by reranker score
- Batch all 50 pairs for efficient GPU inference

The reranker is critical because:
1. Bi-encoder retrieval uses independent embeddings (approximate matching)
2. Cross-encoder sees both texts together (exact relevance modeling)
3. Reranking typically improves MRR@5 by 15-25% over embedding-only retrieval

```python
class RerankingTier:
    """Tier 2: Rerank candidates with cross-encoder."""

    def __init__(self, reranker_model, top_k_final: int = 5):
        self.reranker = reranker_model  # bge-reranker-v2-m3
        self.top_k = top_k_final

    async def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        # Batch score all (query, chunk) pairs
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = await self.reranker.score_batch(pairs)

        # Sort by relevance score, return top-k
        scored_chunks = sorted(
            zip(chunks, scores), key=lambda x: -x[1]
        )
        return [chunk for chunk, score in scored_chunks[:self.top_k]]
```

### Tier 3: Generation

The generation tier assembles context and routes to the appropriate model:

**Context Assembly:**
```
System prompt (fixed, ~200 tokens):
  "You are an enterprise knowledge assistant. Answer questions
   using ONLY the provided context. If the context doesn't
   contain the answer, say so explicitly."

Retrieved context (variable, ~2,560 tokens):
  [Document 1: {title}]
  {chunk_text}

  [Document 2: {title}]
  {chunk_text}
  ... (5 chunks × 512 tokens)

User query (~100 tokens):
  {original_question}
```

**Routing Decision:**
Based on the complexity classification from Tier 1:
- Simple queries → 8B model (direct generation)
- Medium queries → 8B model first, with escalation check
- Complex queries → 70B model (skip 8B entirely)

```python
class GenerationTier:
    """Tier 3: Generate answer using routed model."""

    def __init__(self, model_8b, model_70b, router):
        self.model_8b = model_8b
        self.model_70b = model_70b
        self.router = router

    async def generate(self, query: str, context: list[Chunk],
                       complexity: str) -> Response:
        prompt = self._build_prompt(query, context)

        if complexity == "simple":
            return await self.model_8b.generate(prompt, max_tokens=500)

        if complexity == "complex":
            return await self.model_70b.generate(prompt, max_tokens=1000)

        # Medium: cascade strategy
        response_8b = await self.model_8b.generate(prompt, max_tokens=500)

        if self.router.should_escalate(response_8b.text, query):
            # 8B was uncertain, escalate to 70B
            return await self.model_70b.generate(prompt, max_tokens=1000)

        return response_8b
```

### Request Flow: End-to-End Example

Let us trace a query through the full pipeline:

```
User: "What is our data retention policy for EU customers under GDPR?"

1. Query Router (5ms):
   - Intent: factual_lookup
   - Complexity: medium (regulatory + regional scope)
   - Decision: CASCADE (try 8B first)

2. Embedding (50ms):
   - Vector: [0.023, -0.187, 0.442, ...] (1024-dim)

3. Vector Search (100ms):
   - Top-50 chunks from "policies", "compliance", "gdpr" indexed docs
   - ACL filter: user has access to compliance wiki

4. Hybrid Fusion (20ms):
   - BM25 boosts chunks containing exact phrase "data retention"
   - RRF combines vector similarity + keyword match

5. Reranking (150ms):
   - Cross-encoder scores 50 pairs
   - Top-5: GDPR data retention policy v3.2, EU compliance FAQ,
     Data classification guide, Customer data lifecycle, DPO guidelines

6. Generation - 8B first (500ms):
   - Generates: "Under our GDPR compliance policy (v3.2, updated
     March 2024), EU customer data must be retained for..."
   - Confidence check: response cites specific document, no hedging
   - Decision: NO ESCALATION needed

7. Post-processing (80ms):
   - Citation extraction: links back to source documents
   - Faithfulness check: all claims grounded in provided context
   - Response formatting

Total: ~905ms (well under 2s budget)
```

---

## Part 7: Caching Strategy

### Why Caching is Critical for Enterprise RAG

Enterprise environments have a unique property that makes caching exceptionally effective: **employees ask the same questions repeatedly**. Analysis of enterprise search logs shows:

- 30-40% of queries are semantically identical to a previous query (exact or near-duplicate)
- 60-70% of queries retrieve the same top-5 documents as a recent query
- System prompts are identical across all requests (perfect prefix caching candidate)

These patterns enable three complementary caching layers:

### Layer 1: Semantic Query Cache

Unlike exact-match caches, semantic caching matches queries by meaning. "What's our PTO policy?" and "How many vacation days do I get?" should return the same cached answer.

```python
class SemanticCache:
    """Cache answers by query semantic similarity."""

    def __init__(self, embedding_model, similarity_threshold=0.92):
        self.embedder = embedding_model
        self.threshold = similarity_threshold
        self.cache = {}  # embedding -> (answer, timestamp, hit_count)
        self.index = None  # FAISS index for fast similarity search

    async def lookup(self, query: str) -> Optional[CachedAnswer]:
        query_vec = await self.embedder.encode(query)

        # Search for semantically similar cached queries
        distances, indices = self.index.search(
            query_vec.reshape(1, -1), k=5
        )

        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1 - dist  # cosine distance to similarity
            if similarity >= self.threshold:
                cached = self.cache[idx]
                # Check freshness: invalidate if source docs updated
                if not self._is_stale(cached):
                    cached.hit_count += 1
                    return cached.answer

        return None  # Cache miss

    def _is_stale(self, cached: CachedAnswer) -> bool:
        """Check if any source documents have been updated since caching."""
        for doc_id in cached.source_doc_ids:
            if self.doc_registry.last_modified(doc_id) > cached.timestamp:
                return True  # Source changed, invalidate
        return False
```

**Cache sizing calculation:**
- Unique queries per day: ~50,000 (5,000 users × 20 queries, minus duplicates)
- Average cached response: 2KB
- Embedding per query: 4KB (1024 dims × 4 bytes)
- 7-day retention: 350,000 entries × 6KB = 2.1 GB (fits in RAM)
- Expected hit rate: 35-45% (based on enterprise search log analysis)

**Impact on cost:** If 40% of queries hit the semantic cache, generation cost drops from $0.0117 to $0.007/query, bringing us well under $0.01/query.

### Layer 2: Document-Chunk KV Cache

When multiple users ask different questions about the same documents, they share the same retrieval context. Instead of re-computing KV cache entries for identical document chunks, we cache the KV states:

```python
class DocumentKVCache:
    """Cache KV states for frequently-retrieved document chunks."""

    def __init__(self, max_entries=10000, max_memory_gb=3.0):
        self.cache = LRUCache(max_entries)
        self.memory_budget = max_memory_gb * 1024**3

    def get_prefix_kv(self, chunk_ids: list[str], model: str) -> Optional[KVState]:
        """Look up pre-computed KV states for document chunks."""
        cache_key = (tuple(sorted(chunk_ids)), model)
        return self.cache.get(cache_key)

    def store_prefix_kv(self, chunk_ids: list[str], model: str,
                        kv_state: KVState):
        """Store KV state after computing for a new chunk combination."""
        cache_key = (tuple(sorted(chunk_ids)), model)
        if kv_state.memory_bytes + self.current_usage < self.memory_budget:
            self.cache.put(cache_key, kv_state)
```

**Why this works for RAG specifically:**
- The top-5 popular document chunks appear in 20%+ of all queries
- Pre-computing their KV cache entries saves 60-70% of prefill computation
- For the 8B model: 5 chunks × 512 tokens × 512 KB/token = 1.3 GB of KV state (fits in our 3GB prefix cache budget)

### Layer 3: vLLM Automatic Prefix Caching

vLLM natively supports prefix caching, which we leverage for the system prompt shared across all requests:

```python
# vLLM handles this automatically when enable_prefix_caching=True
# The system prompt (~200 tokens) is computed once and reused for all requests

# Savings calculation:
# 200 tokens × prefill_time_per_token(~0.5ms) = 100ms saved per request
# At 200 QPS: 20,000 prefill-ms saved per second
# Equivalent to: one fewer GPU needed for the same throughput
```

### Cache Invalidation Strategy

Enterprise RAG caches must invalidate when source documents change:

```python
class CacheInvalidator:
    """Event-driven cache invalidation on document updates."""

    def __init__(self, semantic_cache, kv_cache):
        self.semantic_cache = semantic_cache
        self.kv_cache = kv_cache

    async def on_document_updated(self, doc_id: str):
        """Called by the indexing pipeline when a document is re-indexed."""
        # 1. Invalidate all semantic cache entries citing this document
        invalidated = self.semantic_cache.invalidate_by_source(doc_id)

        # 2. Invalidate KV cache entries containing chunks from this doc
        chunk_ids = await self.get_chunks_for_doc(doc_id)
        for chunk_id in chunk_ids:
            self.kv_cache.invalidate_containing(chunk_id)

        logger.info(f"Invalidated {invalidated} cache entries for doc {doc_id}")

    async def on_batch_index_complete(self):
        """Called after nightly batch indexing finishes."""
        # Warm cache with top-100 most-queried document combinations
        popular_combos = await self.analytics.get_top_chunk_combinations(100)
        for chunk_ids in popular_combos:
            await self._warm_kv_cache(chunk_ids)
```

### Combined Cache Impact

| Cache Layer | Hit Rate | Latency Saved | Cost Saved |
|---|---|---|---|
| Semantic query cache | 35-40% | 1,200ms (skip generation) | $0.004/query |
| Document KV cache | 25-30% | 600ms (skip prefill) | $0.002/query |
| Prefix cache (system prompt) | 100% | 100ms (always applies) | Minimal |
| **Combined effective** | **~55%** | **avg 700ms** | **$0.005/query** |

With combined caching, our effective cost per query drops to approximately $0.007, well under the $0.01 target.

---

## Part 8: Monitoring & SLOs

### SLO Framework

Enterprise RAG requires monitoring across four dimensions: latency, quality, cost, and reliability. Each has defined SLOs with graduated alerting:

**Latency SLOs:**

| Metric | Target | Warning | Critical |
|---|---|---|---|
| E2E P50 | <1.5s | >1.8s | >2.5s |
| E2E P99 | <3.0s | >3.5s | >5.0s |
| Retrieval P99 | <200ms | >300ms | >500ms |
| Reranking P99 | <250ms | >350ms | >500ms |
| 8B generation P99 | <1.0s | >1.5s | >2.0s |
| 70B generation P99 | <2.0s | >2.5s | >3.5s |
| Cache lookup P99 | <10ms | >20ms | >50ms |

**Quality SLOs:**

| Metric | Target | Measurement | Alert Threshold |
|---|---|---|---|
| Faithfulness | >95% | LLM-as-judge on sample | <92% (hourly sample) |
| Answer relevance | >90% | User feedback + auto-eval | <87% (daily) |
| Retrieval recall@5 | >80% | Curated test queries | <75% (daily) |
| Hallucination rate | <3% | Claims not in context | >5% (hourly) |
| Cascade accuracy | >85% | Router correct decisions | <80% (daily) |

**Cost SLOs:**

| Metric | Target | Alert |
|---|---|---|
| Cost per query (blended) | <$0.01 | >$0.012 |
| Monthly infrastructure | <$50,000 | >$40,000 (capacity planning) |
| Cache hit rate | >35% | <25% (cache degradation) |
| 70B utilization rate | <60% | >75% (scale signal) |

### Monitoring Implementation

```python
class RAGMetricsCollector:
    """Collects and reports metrics for all pipeline stages."""

    def __init__(self, prometheus_client):
        self.latency = prometheus_client.Histogram(
            'rag_latency_seconds',
            'End-to-end RAG latency',
            ['stage', 'model', 'complexity'],
            buckets=[0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        )
        self.quality = prometheus_client.Gauge(
            'rag_quality_score',
            'Answer quality metrics',
            ['metric_type']  # faithfulness, relevance, hallucination
        )
        self.cost = prometheus_client.Counter(
            'rag_cost_dollars',
            'Cumulative cost tracking',
            ['component']  # retrieval, generation_8b, generation_70b, cache
        )
        self.cache_hits = prometheus_client.Counter(
            'rag_cache_hits_total',
            'Cache hit/miss tracking',
            ['layer', 'result']  # semantic/kv/prefix, hit/miss
        )

    async def record_request(self, request_trace: RequestTrace):
        """Record metrics for a completed request."""
        # Latency breakdown
        for stage, duration in request_trace.stage_durations.items():
            self.latency.labels(
                stage=stage,
                model=request_trace.model_used,
                complexity=request_trace.complexity
            ).observe(duration)

        # Cost attribution
        compute_cost = self._calculate_cost(request_trace)
        self.cost.labels(component=request_trace.model_used).inc(compute_cost)

        # Cache efficiency
        for layer, hit in request_trace.cache_results.items():
            self.cache_hits.labels(
                layer=layer,
                result="hit" if hit else "miss"
            ).inc()
```

### Answer Quality Evaluation Pipeline

Quality monitoring cannot rely solely on user feedback (sparse, biased toward negative). We implement continuous automated evaluation:

```python
class QualityEvaluator:
    """Automated quality scoring using LLM-as-judge."""

    def __init__(self, judge_model, sample_rate=0.05):
        self.judge = judge_model  # Use 70B as judge for sampled responses
        self.sample_rate = sample_rate

    async def evaluate_faithfulness(self, query: str, context: list[str],
                                     response: str) -> float:
        """Score whether response is grounded in provided context."""
        prompt = f"""Score the faithfulness of this response on a scale of 1-10.
A faithful response ONLY contains claims supported by the provided context.

Context: {context}
Question: {query}
Response: {response}

Score (1-10) and brief justification:"""

        result = await self.judge.generate(prompt)
        score = self._extract_score(result)
        return score / 10.0

    async def detect_hallucination(self, response: str,
                                    context: list[str]) -> list[str]:
        """Identify specific claims not grounded in context."""
        prompt = f"""List any claims in the response that are NOT supported
by the provided context. If all claims are supported, respond with "NONE".

Context: {context}
Response: {response}

Unsupported claims:"""

        result = await self.judge.generate(prompt)
        if "NONE" in result.upper():
            return []
        return self._parse_claims(result)
```

### Alerting and Dashboards

The monitoring system produces three operational dashboards:

**Dashboard 1: Real-Time Operations**
- QPS by model (8B vs 70B), latency percentiles, active requests, queue depth
- Cache hit rates (semantic, KV, prefix) with trend lines
- Error rates by stage (retrieval failures, generation timeouts, OOMs)

**Dashboard 2: Quality Tracking**
- Faithfulness score (hourly moving average)
- Hallucination incidents (count and examples)
- User feedback sentiment (thumbs up/down ratio)
- Retrieval recall on test queries (daily benchmark run)

**Dashboard 3: Cost and Capacity**
- Cost per query trend (7-day rolling)
- GPU utilization by instance (memory, compute, bandwidth)
- Traffic forecast vs. current capacity
- Monthly projected spend vs. budget

---

## Part 9: Scaling & Cost Optimization

### Traffic Patterns and Scaling Strategy

Enterprise RAG has highly predictable traffic patterns:

```
QPS Profile (typical weekday):
  00:00-06:00 UTC:  ~5 QPS   (overnight, automated queries only)
  06:00-09:00 UTC:  ~50 QPS  (EU morning ramp)
  09:00-14:00 UTC:  ~150 QPS (EU + US East peak)
  14:00-17:00 UTC:  ~200 QPS (full global peak)
  17:00-21:00 UTC:  ~80 QPS  (US wind-down)
  21:00-00:00 UTC:  ~15 QPS  (evening stragglers)
```

This predictability enables aggressive time-based scaling rather than reactive autoscaling:

### Scheduled Scaling Policy

```python
scaling_policy = {
    "70B_instances": {
        "00:00-06:00": 1,  # Minimum: 1 g5.12xlarge (maintains availability)
        "06:00-09:00": 1,  # Single instance handles 50 QPS with queueing
        "09:00-17:00": 2,  # Peak: 2 instances (8 concurrent 70B requests)
        "17:00-21:00": 1,
        "21:00-00:00": 1,
    },
    "8B_instances": {
        "00:00-06:00": 1,  # Minimum for availability
        "06:00-09:00": 2,
        "09:00-17:00": 3,  # Peak: 48 concurrent 8B requests
        "17:00-21:00": 2,
        "21:00-00:00": 1,
    },
}
```

### Batch Indexing on Spot Instances

Nightly batch indexing (re-embedding updated documents, rebuilding HNSW index segments) runs on spot instances:

**Batch Indexing Requirements:**
- Process ~100,000 updated/new documents per night (typical corporate velocity)
- 100K docs × 5 chunks/doc = 500K chunks to embed
- At 500 embeddings/second per GPU: 1,000 seconds = ~17 minutes on single GPU
- With 2 spot GPUs in parallel: ~8.5 minutes

**Spot Instance Strategy:**
- Use g5.xlarge spot instances ($0.38/hr, 62% savings vs. on-demand)
- Launch 2 instances at midnight, terminate after indexing
- Checkpointing every 50K chunks (if spot interruption, resume from checkpoint)
- Fallback: if spot unavailable for >30 minutes, launch on-demand (still cheap, ~$2 for the job)

```python
class BatchIndexingPipeline:
    """Nightly document re-indexing pipeline."""

    def __init__(self, config):
        self.chunk_size = 512
        self.chunk_overlap = 64
        self.batch_size = 256
        self.checkpoint_interval = 50000

    async def run_nightly_index(self):
        """Full nightly indexing job."""
        # 1. Identify changed documents since last index
        changed_docs = await self.doc_store.get_modified_since(
            self.last_index_timestamp
        )
        logger.info(f"Indexing {len(changed_docs)} changed documents")

        # 2. Chunk documents
        chunks = self.chunker.chunk_documents(
            changed_docs,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap
        )

        # 3. Embed in batches (distributed across spot GPUs)
        embeddings = await self.embed_distributed(
            chunks,
            batch_size=self.batch_size,
            checkpoint_every=self.checkpoint_interval
        )

        # 4. Upsert into vector DB
        await self.vector_db.bulk_upsert(chunks, embeddings)

        # 5. Invalidate affected caches
        affected_doc_ids = [doc.id for doc in changed_docs]
        await self.cache_invalidator.invalidate_batch(affected_doc_ids)

        # 6. Update last index timestamp
        self.last_index_timestamp = datetime.utcnow()
```

### Cost Breakdown: Monthly Detail

| Category | Component | Monthly Cost | % of Total |
|---|---|---|---|
| **Generation** | 70B (1.5 avg instances × $5.67/hr) | $6,208 | 42% |
| **Generation** | 8B (2 avg instances × $1.01/hr) | $1,475 | 10% |
| **Retrieval** | Embedding + Reranker (1 × $1.01/hr) | $738 | 5% |
| **Storage** | OpenSearch (3 × r6g.2xlarge) | $2,847 | 19% |
| **Batch** | Spot indexing (2 × $0.38/hr × 8hr × 30) | $182 | 1.2% |
| **Compute** | API + LB (2 × m6i.xlarge) | $280 | 1.9% |
| **Storage** | S3 (document store, 5TB) | $115 | 0.8% |
| **Storage** | EBS (model weights, caches) | $200 | 1.4% |
| **Network** | Data transfer, VPC endpoints | $350 | 2.4% |
| **Other** | CloudWatch, secrets, DNS | $150 | 1% |
| **Contingency** | 15% buffer for traffic spikes | $1,877 | 12.7% |
| | **Total** | **$14,422** | **100%** |

### Cost Per Query Calculation

```
Monthly queries: 50 QPS average × 3600 × 24 × 30 = 129.6M queries
Monthly cost: $14,422

Gross cost/query: $14,422 / 129,600,000 = $0.000111

Wait, that seems too low. Let us recalculate with realistic sustained QPS:

Business hours only (14 hrs/day × 22 working days):
  Peak queries: 150 QPS × 14 hrs × 3600 × 22 days = 166.3M queries/month

Actually correct sustained estimate:
  Average QPS across full month: ~50
  Monthly queries: 50 × 2,592,000 seconds = 129.6M

Cost per query: $14,422 / 129,600,000 ≈ $0.00011/query ← infrastructure amortized

But the meaningful metric is marginal cost per query (compute consumed):
  70B query (20s GPU): $5.67/3600 × 20 = $0.0315 ← expensive
  8B query (5s GPU): $1.01/3600 × 5 = $0.0014
  Cached query: ~$0.0001 (lookup only)

  Blended (40% cached, 30% 8B, 30% 70B):
  0.40 × $0.0001 + 0.30 × $0.0014 + 0.30 × $0.0315 = $0.0098/query
```

We achieve the target of <$0.01/query through the combination of semantic caching (eliminating 40% of generation calls) and model cascade routing (sending only truly complex queries to 70B).

### Reserved Instance Strategy

For the always-on components (70B and 8B generation), 1-year Reserved Instances provide significant savings:

| Component | On-Demand Monthly | 1yr RI Monthly | Savings |
|---|---|---|---|
| g5.12xlarge (70B) | $4,139 | $2,690 | 35% |
| g5.xlarge × 2 (8B) | $1,475 | $958 | 35% |
| r6g.2xlarge × 3 (OpenSearch) | $2,847 | $1,850 | 35% |

With RIs: monthly cost drops from $14,422 to ~$10,800, providing even more budget headroom.

---

## Part 10: Failure Modes & Mitigation

### Failure Mode 1: Retrieval Returns Irrelevant Documents

**Symptom**: LLM generates confident-sounding answers based on wrong context, producing hallucinations that are hard for users to detect.

**Why it happens:**
- Embedding model conflates semantically similar but contextually different content
- HNSW approximate search misses true nearest neighbors (recall < 100%)
- Query ambiguity: "What's the retention policy?" (data retention? employee retention? customer retention?)

**Detection:**
```python
class RetrievalQualityMonitor:
    """Detect when retrieval is returning low-quality results."""

    def check_retrieval_quality(self, query: str, chunks: list[Chunk],
                                 reranker_scores: list[float]) -> Alert:
        # Signal 1: Reranker scores are uniformly low
        if max(reranker_scores) < 0.3:
            return Alert(
                level="warning",
                message="No highly relevant chunks found",
                action="prepend 'I could not find specific information about this' to response"
            )

        # Signal 2: Large gap between top-1 and top-2 score
        if reranker_scores[0] - reranker_scores[1] > 0.5:
            # Only one relevant document found, might be insufficient
            return Alert(level="info", message="Single-source answer")

        # Signal 3: Query-chunk embedding similarity is low even for top results
        similarities = [cosine_sim(query_emb, chunk.embedding) for chunk in chunks[:5]]
        if max(similarities) < 0.6:
            return Alert(
                level="critical",
                message="Potential out-of-domain query",
                action="respond with 'I don't have information about this topic'"
            )
```

**Mitigation:**
1. Hybrid search (BM25 + vector) catches keyword-specific queries
2. Reranker filters out false-positive retrievals
3. Generation prompt explicitly instructs "say you don't know if context is insufficient"
4. Query clarification: when retrieval confidence is low, ask user to rephrase
5. Human escalation: route low-confidence answers to subject matter experts

### Failure Mode 2: 70B GPU OOM During Peak Load

**Symptom**: vLLM rejects requests with "No available slots" or OOM kills the process.

**Why it happens:**
- More concurrent requests than KV cache budget allows (>4 on our config)
- Single long response consuming excessive KV cache
- Memory fragmentation after running for days without restart

**Detection:**
```python
# vLLM exposes metrics for monitoring
GPU_MEMORY_ALERTS = {
    "gpu_cache_usage_perc > 0.90": "WARNING: KV cache nearly full",
    "num_requests_waiting > 10": "WARNING: Request queue growing",
    "avg_generation_throughput_toks < 15": "CRITICAL: Generation slowdown (OOM pressure)",
}
```

**Mitigation:**
1. **Request queue with backpressure**: When 70B queue exceeds 10 requests, route new medium-complexity queries to 8B instead of waiting
2. **Max sequence length enforcement**: Hard cap at 4,096 tokens (truncate retrieved context if needed)
3. **Graceful degradation**: If 70B is overloaded, respond with 8B + "Note: this answer used our faster model. For more detailed analysis, please retry in a few minutes."
4. **Proactive scaling**: When GPU memory utilization exceeds 85% for >5 minutes, launch second g5.12xlarge
5. **Daily restart**: Schedule maintenance window at 3 AM UTC for process restart (clears fragmentation)

```python
class LoadSheddingPolicy:
    """Graceful degradation when 70B is overloaded."""

    def __init__(self, queue_threshold=10, memory_threshold=0.90):
        self.queue_threshold = queue_threshold
        self.memory_threshold = memory_threshold

    async def check_and_shed(self, request: Request) -> RoutingDecision:
        queue_depth = await self.model_70b.get_queue_depth()
        memory_usage = await self.model_70b.get_memory_utilization()

        if queue_depth > self.queue_threshold or memory_usage > self.memory_threshold:
            if request.complexity == "medium":
                # Downgrade medium queries to 8B during overload
                return RoutingDecision(
                    model="8b",
                    reason="load_shedding",
                    metadata={"add_disclaimer": True}
                )
            elif request.complexity == "complex":
                # Queue complex queries but with timeout
                return RoutingDecision(
                    model="70b",
                    reason="queued",
                    timeout_ms=5000,
                    fallback="8b"
                )

        return RoutingDecision(model="70b", reason="normal")
```

### Failure Mode 3: Index Staleness (New Docs Not Yet Searchable)

**Symptom**: Users upload a document and immediately try to query it, but get "I don't have information about this" because nightly indexing has not yet processed it.

**Why it happens:**
- Batch indexing runs once nightly; documents uploaded during the day are invisible until next index run
- Even with real-time indexing, HNSW segment merging introduces 5-30 minute delays

**Mitigation: Tiered Indexing Strategy**

```python
class TieredIndexingPipeline:
    """Three-tier indexing for different freshness requirements."""

    async def on_document_created(self, doc: Document):
        # Tier 1: Immediate (within 30 seconds)
        # Add to a "hot" index that gets searched alongside main index
        chunks = self.chunker.chunk(doc)
        embeddings = await self.embedder.encode_batch(chunks)
        await self.hot_index.upsert(chunks, embeddings)

        # The hot index is small (<10K chunks) and uses brute-force search
        # (no HNSW, so updates are instant but search is O(n))

    async def run_hourly_merge(self):
        # Tier 2: Hourly merge of hot index into warm segment
        hot_chunks = await self.hot_index.drain_all()
        await self.warm_index.bulk_insert(hot_chunks)
        # Warm index uses HNSW but with smaller ef_construction for speed

    async def run_nightly_optimize(self):
        # Tier 3: Nightly full optimization
        await self.warm_index.merge_into(self.main_index)
        await self.main_index.optimize_segments()  # Rebuild HNSW graph
        # After optimization: best recall, best search performance
```

**Search-time fusion:**
```python
async def hybrid_search(self, query_embedding, top_k=50):
    # Search all three tiers in parallel
    hot_results, warm_results, main_results = await asyncio.gather(
        self.hot_index.search(query_embedding, k=10),     # Brute force, instant freshness
        self.warm_index.search(query_embedding, k=20),    # HNSW, hourly freshness
        self.main_index.search(query_embedding, k=top_k), # Optimized HNSW, nightly
    )
    # Merge and deduplicate
    return self._merge_results(hot_results, warm_results, main_results)[:top_k]
```

### Failure Mode 4: Cascade Timeout (8B to 70B Escalation Adds Latency)

**Symptom**: Medium-complexity queries that escalate from 8B to 70B take 2x the latency budget because they run through both models sequentially.

**Quantitative impact:**
- 8B generation: 500ms
- Escalation decision: 50ms
- 70B generation: 1,500ms
- Total for escalated query: 2,050ms (exceeds 2s budget)

**Mitigation strategies:**

1. **Speculative execution**: For "medium" queries, start 70B generation in parallel with 8B. Cancel 70B if 8B is confident. Wastes some 70B compute but guarantees latency.

```python
class SpeculativeCascade:
    """Start both models for medium queries, cancel the unnecessary one."""

    async def generate_medium(self, prompt: str) -> Response:
        # Launch both in parallel
        task_8b = asyncio.create_task(self.model_8b.generate(prompt))
        task_70b = asyncio.create_task(self.model_70b.generate(prompt))

        # Wait for 8B (faster)
        response_8b = await task_8b

        if self.router.is_confident(response_8b):
            task_70b.cancel()  # Save 70B compute
            return response_8b
        else:
            # 8B uncertain, wait for 70B (already running!)
            response_70b = await task_70b
            return response_70b
```

2. **Confidence-weighted timeout**: If 8B does not finish within 400ms, start 70B immediately (do not wait for 8B confidence check).

3. **Predictive routing improvement**: Invest in the complexity classifier. If it correctly routes 95% of queries on the first try (instead of 85%), the cascade path triggers only for 5% of queries, making the latency impact negligible.

### Failure Mode Summary Table

| Failure Mode | Frequency | Impact | Detection Time | Recovery Time |
|---|---|---|---|---|
| Irrelevant retrieval | 5-8% of queries | Hallucinated answers | Real-time (reranker scores) | Immediate (fallback response) |
| 70B OOM | 1-2× per week | Rejected requests | <1 minute (metrics) | 2-5 min (auto-scale) |
| Index staleness | Continuous (by design) | Missing recent docs | N/A (known limitation) | <30s (hot index) |
| Cascade timeout | 10-15% of medium queries | Latency SLO breach | Real-time (P99 tracking) | Immediate (speculative exec) |
| Vector DB failure | Monthly (rare) | Full service outage | <30s (health check) | 2-5 min (failover) |
| Embedding model crash | Rare | Cannot embed new queries | <10s (health check) | 30s (container restart) |

---

## Key Takeaways

This enterprise RAG design demonstrates several principles that generalize to any production LLM system:

1. **Model cascades beat single-model deployments** on both cost and latency. Not every query needs your largest model, and routing intelligence pays for itself immediately.

2. **Caching is the single highest-ROI optimization** for enterprise workloads. The combination of semantic caching, prefix caching, and document KV caching reduces effective generation cost by 40-55%.

3. **Memory budgeting determines your architecture.** The difference between 4 concurrent sequences and 16 concurrent sequences is the difference between needing 1 instance and needing 4. Always compute memory budgets before selecting hardware.

4. **Heterogeneous fleets match heterogeneous workloads.** Embedding, reranking, and generation have fundamentally different compute profiles. Using the right instance type for each stage saves 30-40% vs. uniform large instances.

5. **Graceful degradation preserves user trust.** When the 70B model is overloaded, serving a slightly worse answer from 8B with a disclaimer is far better than a 10-second timeout. Users forgive quality variation; they do not forgive unresponsiveness.

6. **Freshness and accuracy trade off against each other.** The three-tier indexing strategy (hot/warm/main) lets you tune this tradeoff per deployment, from "always fresh but slightly lower recall" to "optimized recall but 24-hour indexing lag."

The total cost of $14,422/month (or ~$10,800 with Reserved Instances) serves 5,000 concurrent users at <2s latency and >90% accuracy, delivering approximately $0.01 per query. For an enterprise, this replaces multiple support staff, reduces ticket volume, and makes institutional knowledge accessible to every employee instantly.
