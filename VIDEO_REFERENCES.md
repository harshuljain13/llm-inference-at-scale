# Video References

Curated technical talks and lectures mapped to each chapter. Watch these alongside reading for deeper understanding.

## Part I: Foundations

### Ch00 — Transformer Anatomy
- [Andrej Karpathy: Deep Dive into LLMs like ChatGPT](https://youtube.com/watch?v=7xTGNNLPyMI) (~3.5 hrs) — Transformer internals, token generation, parameters as memory
- [Mark Moyou (NVIDIA): Mastering LLM Inference Optimization](https://www.youtube.com/results?search_query=Mark+Moyou+LLM+Inference+Optimization+NVIDIA) (~45 min) — Prefill vs decode, memory bandwidth, hardware sizing

### Ch01 — GPU Fundamentals
- [CUDA MODE Lecture 4: Compute and Memory Basics](https://www.youtube.com/results?search_query=CUDA+MODE+lecture+4+compute+memory) (~1.5 hrs) — GPU memory hierarchy, roofline, kernel fusion
- [Bill Dally (NVIDIA): Trends in Deep Learning Hardware](https://www.youtube.com/results?search_query=Bill+Dally+trends+deep+learning+hardware+CUDA+MODE) (~1 hr) — GPU architecture evolution, bandwidth constraints

### Ch02 — Attention Mechanisms
- [Tri Dao: FlashAttention — Fast and Memory-Efficient Exact Attention](https://www.youtube.com/results?search_query=Tri+Dao+FlashAttention+Stanford) (~1 hr) — Online softmax, tiling, IO complexity
- [Google: Efficiently Scaling Transformer Inference (MLSys 2023)](https://www.youtube.com/results?search_query=MLSys+2023+Efficiently+Scaling+Transformer+Inference) (~20 min) — MQA, partitioning for inference

## Part II: Optimizations

### Ch04 — Quantization and Speculative Decoding
- [NVIDIA GTC: Optimize GenAI Inference with Quantization in TensorRT-LLM](https://www.nvidia.com/en-us/on-demand/session/gtc24-s63213/) (~40 min) — INT4/INT8/FP8 practical guide
- [Anyscale: Optimizing vLLM Performance Through Quantization (Ray Summit 2024)](https://www.youtube.com/results?search_query=Optimizing+vLLM+Performance+Quantization+Ray+Summit+2024) (~30 min)
- [PyTorch Blog: A Hitchhiker's Guide to Speculative Decoding](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/) — Draft-verify mechanics, EAGLE, Medusa

### Ch05 — Inference Engines
- [Databricks: vLLM Optimization for Cost-Effective LLM Inference (Ray Summit 2024)](https://www.youtube.com/results?search_query=Databricks+vLLM+Optimization+Ray+Summit+2024) (~30 min)
- [The Evolution of Multi-GPU Inference in vLLM (Ray Summit 2024)](https://www.youtube.com/results?search_query=Evolution+Multi-GPU+Inference+vLLM+Ray+Summit+2024) (~25 min)
- [Lianmin Zheng: SGLang — Efficient Execution of Structured LLM Programs (NeurIPS 2024)](https://www.youtube.com/results?search_query=SGLang+Lianmin+Zheng+NeurIPS+2024) (~15 min)

### Ch06 — Parallelism
- [Anyscale: Accelerated LLM Inference (Ray Summit 2024)](https://www.youtube.com/results?search_query=Accelerated+LLM+Inference+Anyscale+Ray+Summit+2024) (~30 min) — Multi-GPU scaling patterns

## Part III: Operationalization

### Ch07 — Serving
- [NVIDIA GTC 2025: Disaggregated Serving with TensorRT-LLM](https://www.youtube.com/results?search_query=NVIDIA+GTC+2025+disaggregated+serving+TensorRT-LLM) — Prefill/decode separation at scale
- [KubeCon EU 2026: KV-Cache Wins You Can Feel (IBM Research)](https://www.youtube.com/results?search_query=KubeCon+2026+KV+Cache+routing+IBM) — Cache-aware K8s routing

### Ch09 — Production Stories
- [Meta Engineering: Scaling LLM Inference (Oct 2025)](https://www.youtube.com/results?search_query=Meta+Scaling+LLM+Inference+2025) — TP+CP+EP at 100K GPUs
- [NVIDIA GTC: Scaling DeepSeek-V3 Inference](https://www.youtube.com/results?search_query=NVIDIA+GTC+DeepSeek-V3+inference+scaling) — MoE expert parallelism at scale

---

*Note: Some links are search queries rather than direct URLs due to video availability. Search YouTube with the provided query to find the talk.*
