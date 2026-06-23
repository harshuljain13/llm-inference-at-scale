# 6.1 Tensor Parallelism

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harshuljain13/llm-inference-at-scale/blob/master/content/07_scaling/06.1_tensor_parallelism/lab.ipynb)
[![Open In Molab](https://raw.githubusercontent.com/marimo-team/marimo/main/docs/_static/marimo-badge.svg)](https://molab.marimo.io/open?url=https://github.com/harshuljain13/llm-inference-at-scale/blob/master/content/07_scaling/06.1_tensor_parallelism/lab.ipynb)

> When a model exceeds one GPU's memory, tensor parallelism splits every layer across GPUs so each reads only a fraction of weights per token.

Llama 70B in FP16 needs 140 GB. No single GPU holds that. Tensor parallelism (TP) from Megatron-LM (Shoeybi et al., 2019) solves this by sharding weight matrices across GPUs within a node, using AllReduce to recombine partial results after each layer. The result: lower per-token latency (less compute per GPU) at the cost of requiring fast interconnect (NVLink) between GPUs.

---

## Where TP Fits Among Parallelism Strategies

```mermaid
flowchart LR
    subgraph DP["Data Parallelism"]
        style DP fill:#dcfce7,stroke:#000,color:#000
        D1[GPU 0: Full Model<br/>Batch 0]
        D2[GPU 1: Full Model<br/>Batch 1]
    end
    subgraph TP["Tensor Parallelism"]
        style TP fill:#dbeafe,stroke:#000,color:#000
        T1[GPU 0: Layer Slice]
        T2[GPU 1: Layer Slice]
        T1 -->|AllReduce| T2
    end
    subgraph PP["Pipeline Parallelism"]
        style PP fill:#f3e8ff,stroke:#000,color:#000
        P1[GPU 0: Layers 0-15] --> P2[GPU 1: Layers 16-31]
    end
```

**Data Parallelism**: replicate model, split requests. No communication. Use when model fits on one GPU.
**Tensor Parallelism**: split every layer, AllReduce per layer. Use within a node (NVLink).
**Pipeline Parallelism**: assign layer ranges to GPUs in sequence. Use across nodes (InfiniBand).

The rule: TP within a node (600-900 GB/s NVLink), PP across nodes (200-400 GB/s InfiniBand).

---

## Column-Parallel Split

The first linear in the FFN projects hidden_size to ffn_hidden_size. Column parallelism splits the weight matrix W along columns. Each GPU computes a slice of the output independently, requiring no communication.

```mermaid
flowchart LR
    subgraph Input
        style Input fill:#f3f4f6,stroke:#000,color:#000
        X["X [batch, 4096]"]
    end
    subgraph ColSplit["Column-Parallel: W split by columns"]
        style ColSplit fill:#dbeafe,stroke:#000,color:#000
        G0["GPU 0: W[:, :4096]<br/>→ Y[:, :4096]"]
        G1["GPU 1: W[:, 4096:8192]<br/>→ Y[:, 4096:8192]"]
        G2["GPU 2: W[:, 8192:12288]<br/>→ Y[:, 8192:12288]"]
        G3["GPU 3: W[:, 12288:]<br/>→ Y[:, 12288:]"]
    end
    X --> G0
    X --> G1
    X --> G2
    X --> G3
```

Each GPU receives the full input X and multiplies by its column slice. The outputs are disjoint portions of Y. No AllReduce needed at this stage because the next layer uses row parallelism that consumes these partial outputs directly.

---

## Row-Parallel Split and AllReduce

The second FFN linear projects ffn_hidden_size back to hidden_size. Row parallelism splits W along rows, aligning with column-parallel outputs from the previous layer. Each GPU computes a partial sum, then AllReduce combines them.

```mermaid
flowchart LR
    subgraph Partials["Row-Parallel: partial sums"]
        style Partials fill:#fef3c7,stroke:#000,color:#000
        R0["GPU 0: Y_0 × W[0:4096, :]<br/>→ Z_partial_0"]
        R1["GPU 1: Y_1 × W[4096:8192, :]<br/>→ Z_partial_1"]
        R2["GPU 2: Y_2 × W[8192:12288, :]<br/>→ Z_partial_2"]
        R3["GPU 3: Y_3 × W[12288:, :]<br/>→ Z_partial_3"]
    end
    subgraph AR["AllReduce"]
        style AR fill:#ffe4e6,stroke:#000,color:#000
        Z["Z = sum of partials<br/>(all GPUs get result)"]
    end
    R0 --> Z
    R1 --> Z
    R2 --> Z
    R3 --> Z
```

The column-then-row pattern requires only ONE AllReduce per FFN block. This halves communication versus naive approaches where each matmul would need its own synchronization.

---

## Attention Head Distribution

Multi-head attention is naturally parallel: each head operates independently. TP assigns contiguous groups of heads to GPUs. Communication happens only after the output projection.

```mermaid
flowchart LR
    subgraph Heads["Llama 70B: 64 heads, TP=8"]
        style Heads fill:#ccfbf1,stroke:#000,color:#000
        H0["GPU 0<br/>Heads 0-7"]
        H1["GPU 1<br/>Heads 8-15"]
        H2["GPU 2<br/>Heads 16-23"]
        H3["..."]
        H7["GPU 7<br/>Heads 56-63"]
    end
    subgraph Sync["Output Projection"]
        style Sync fill:#ffe4e6,stroke:#000,color:#000
        AR2["AllReduce"]
    end
    H0 --> AR2
    H1 --> AR2
    H2 --> AR2
    H3 --> AR2
    H7 --> AR2
```

Each GPU computes Q, K, V for its heads only, runs attention independently, then AllReduce merges outputs. Constraint: TP must divide num_attention_heads evenly (32 heads → TP of 1, 2, 4, 8, 16, 32).

---

## Communication Cost: Why NVLink is Mandatory

Every layer requires 2 AllReduces (attention + FFN). The ring AllReduce algorithm transfers 2×(N-1)/N × message_size. For Llama 70B decode at batch=32:

- Hidden: 8192, FP16 → message = 32 × 8192 × 2 = 512 KB per AllReduce
- Per forward pass: 512 KB × 2 × 80 layers = 80 MB total
- At 100 tok/s decode: 8 GB/s sustained bandwidth needed

| Interconnect | Bandwidth | TP=4 Efficiency | TP=8 Efficiency |
|---|---|---|---|
| PCIe 4.0 | 32 GB/s | ~65% | ~44% |
| PCIe 5.0 | 64 GB/s | ~78% | ~62% |
| NVLink 3 (A100) | 600 GB/s | ~92% | ~85% |
| NVLink 4 (H100) | 900 GB/s | ~95% | ~90% |

NVLink provides 10-28x more bandwidth than PCIe. Beyond TP=2, PCIe systems spend more time communicating than computing. This is why all cloud GPU instances designed for large model serving (p4d, p5) use NVLink internally.

---

## Choosing TP Degree

| Model Size | Instance | TP | Notes |
|---|---|---|---|
| ≤13B | g5.xlarge (A10G) | 1 | Fits single GPU with INT4/INT8 |
| 13-30B | g5.12xlarge | 2-4 | NVLink within node |
| 30-70B | p4d.24xlarge | 4-8 | 8×A100 NVLink |
| 70-140B | p4d.24xlarge | 8 | Full node required |
| >140B | 2×p4d.24xlarge | TP=8, PP=2 | Multi-node |
| 405B | 4×p5.48xlarge | TP=8, PP=4 | Multi-node |

Always calculate: per-GPU memory = (model_bytes / TP) + KV_cache + overhead. If total exceeds GPU VRAM, increase TP or apply quantization.

---

## FAQ

**Q: Can I use TP across nodes (InfiniBand)?**
Technically yes, but efficiency drops below 50% due to latency. Use PP across nodes instead.

**Q: Why not just use data parallelism for everything?**
DP requires each replica to hold the full model. When the model exceeds one GPU, DP cannot help.

**Q: Does quantization reduce the need for TP?**
Yes. Llama 70B in INT4 (35 GB) fits on one 80 GB GPU, eliminating TP entirely.

**Q: What about GQA models with fewer KV heads?**
GQA reduces KV cache memory but does not change the TP constraint on Q heads. TP must still divide num_attention_heads.

**Q: How does vLLM handle TP?**
Set `--tensor-parallel-size N`. vLLM shards weights, inserts AllReduce ops, and manages NCCL automatically.

---

## References

1. Shoeybi et al. "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019)
2. Narayanan et al. "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (2021)
3. NVIDIA NCCL Documentation: https://docs.nvidia.com/deeplearning/nccl/
4. vLLM Documentation: Distributed Inference: https://docs.vllm.ai/
5. Pope et al. "Efficiently Scaling Transformer Inference" (2022)
