# SGLang: A Programming Language for Structured LLM Inference

## Why Another Serving Engine?

vLLM solved the memory problem. TensorRT-LLM solved the kernel problem. But neither solved the *programming* problem: how do you express complex generation patterns (constrained JSON output, multi-step reasoning chains, parallel tool calls) without fighting the serving layer at every step?

SGLang answers this question by treating LLM inference as a programmable computation, not just a request/response API. Built at UC Berkeley by Lianmin Zheng (co-creator of vLLM, Vicuna, and LMSYS Chatbot Arena), SGLang combines two innovations that make it fundamentally different from other serving engines:

1. **RadixAttention**: a tree-based KV cache that shares prefixes across arbitrary request patterns, not just sequential conversations.
2. **Native structured generation**: constrained decoding (JSON schemas, regex grammars) integrated at the scheduling layer, not bolted on as post-processing.

The result: 5-10x throughput improvement on structured output workloads and 2-3x improvement on prefix-heavy workloads like RAG and multi-turn chat. These are not synthetic benchmarks; they reflect the workloads that dominate production inference today: agents calling tools, APIs returning JSON, and chatbots sharing long system prompts across thousands of concurrent users.

From Module 03.5, you know prefix caching avoids redundant prefill computation by reusing KV cache entries for shared prompt prefixes. SGLang takes this further with RadixAttention: a tree-based cache that shares prefixes across ANY request pattern, not just the linear prefix chains that hash-based approaches handle. Where vLLM's Automatic Prefix Caching (APC) matches exact hash blocks, RadixAttention matches the longest common prefix in O(log n) time across a tree of all active and recently evicted sequences.

---

## RadixAttention: Tree-Based KV Cache Sharing

### The Limitation of Hash-Based Prefix Caching

vLLM's APC (Automatic Prefix Caching) works by hashing fixed-size blocks of tokens and storing their KV cache entries in a hash table. When a new request arrives, the engine checks if the hash of each block already exists:

```
Request A: [system_prompt | user_msg_1 | assistant_1 | user_msg_2]
Request B: [system_prompt | user_msg_1 | assistant_1 | user_msg_3]

Hash blocks: [block_0][block_1][block_2][block_3]...
Match: blocks 0-2 are identical -> reuse their KV cache
```

This works well for linear prefix chains (chat continuations). But it fails in three important scenarios:

1. **Branching conversations**: when one prompt spawns multiple completions (beam search, parallel sampling), hash-based matching cannot share across branches efficiently.
2. **Partial block matches**: if two requests share 95% of a block but differ in the last few tokens, the entire block is recomputed.
3. **Cross-request prefix sharing**: when unrelated requests happen to share prefixes (common in RAG where multiple queries retrieve the same documents), hash tables require exact block boundary alignment.

### How RadixAttention Solves This

RadixAttention organizes the KV cache as a radix tree (also called a Patricia trie or compressed trie). Each edge in the tree represents a sequence of tokens, and each node stores the corresponding KV cache tensors:

```
Root
 |
 [system_prompt tokens 0-512]  <- shared by ALL requests
 |
 +-- [user_msg_1 tokens 513-600]
 |    |
 |    +-- [assistant_1 tokens 601-750]
 |    |    |
 |    |    +-- [user_msg_2 tokens 751-800]  <- Request A continues here
 |    |    +-- [user_msg_3 tokens 751-820]  <- Request B branches here
 |    |
 |    +-- [assistant_2 tokens 601-780]      <- Different response, same prefix
 |
 +-- [user_msg_X tokens 513-590]            <- Completely different conversation
```

The tree structure provides several advantages over flat hash tables:

**Longest prefix matching in O(log n)**. When a new request arrives, the engine traverses the tree from root to find the longest matching prefix. This is a single tree traversal, not a sequence of hash lookups. The complexity is O(log n) where n is the number of unique token sequences, compared to O(k) hash lookups where k is the number of blocks.

**Variable-length sharing**. Because edges represent variable-length token sequences (not fixed blocks), two requests that share 513 tokens get full reuse even if 513 is not a multiple of the block size.

**Automatic deduplication across request patterns**. The tree naturally deduplicates: if 1000 requests share the same system prompt, only one copy of those KV tensors exists in GPU memory. No explicit deduplication logic is needed.

**Cache-aware scheduling**. The scheduler can prioritize requests whose prefixes are already cached (hot nodes in the tree), improving overall hit rates.

### RadixAttention vs. vLLM APC: Quantitative Comparison

| Dimension | vLLM APC | SGLang RadixAttention |
|-----------|----------|----------------------|
| Data structure | Hash table (block-level) | Radix tree (token-level) |
| Match granularity | Fixed block size (16 tokens default) | Variable length, any boundary |
| Lookup complexity | O(k) where k = num blocks | O(log n) tree traversal |
| Cross-request sharing | Only if blocks align exactly | Automatic via tree structure |
| Branching support | Limited (each branch is independent) | Native (branches share parent path) |
| Memory overhead | Hash table entries per block | Tree nodes (more compact for shared prefixes) |
| Eviction policy | LRU per block | LRU with tree-aware eviction (evict leaves first) |

### Tree-Aware Eviction

When GPU memory pressure requires evicting cached entries, SGLang's eviction policy respects the tree structure. Leaf nodes (recently completed requests with no children) are evicted first because they have no dependents. Internal nodes (shared prefixes) are evicted last because removing them invalidates all downstream entries. This contrasts with flat LRU eviction where a frequently shared prefix block might be evicted simply because no single request accessed it recently.

The eviction algorithm assigns each node a "sharing score":

```python
# Simplified eviction scoring
def eviction_priority(node):
    # Nodes with more children are more valuable
    sharing_factor = len(node.children) + sum(
        eviction_priority(child) for child in node.children
    )
    # Recent access time also matters
    recency = time.time() - node.last_access_time
    # Higher score = evict later (more valuable)
    return sharing_factor / (1 + recency * decay_rate)
```

This means a system prompt prefix shared by 500 active conversations will survive eviction pressure even if some individual conversations are idle, because the sharing factor dominates the recency component.

### Implementation Detail: Cache-Aware Scheduling

SGLang's scheduler integrates with the radix tree to make scheduling decisions. When multiple requests are waiting, the scheduler computes a "cache locality score" for each:

```python
# Simplified cache-aware scheduling logic
def schedule_next_batch(waiting_requests, radix_tree):
    scored = []
    for req in waiting_requests:
        # How many tokens of this request are already cached?
        cached_prefix_len = radix_tree.longest_prefix_match(req.tokens)
        # Ratio of cached to total prefix
        locality_score = cached_prefix_len / len(req.tokens)
        scored.append((locality_score, req))
    
    # Prefer requests with high cache locality
    scored.sort(reverse=True)
    return select_batch(scored, max_batch_tokens=budget)
```

This means that after processing a batch of requests with the same system prompt, subsequent requests with that prompt are prioritized because their KV cache is "hot." The effect is a natural batching of similar workloads without explicit routing logic.

---

## Structured Generation at the Engine Level

### Why Post-Hoc Masking Is Slow

Most frameworks implement constrained decoding as a logit processor: after the model produces logits for the next token, a mask is applied to zero out tokens that would violate the constraint (e.g., produce invalid JSON). This approach has three problems:

1. **Serial overhead per token**. Computing the valid token mask requires evaluating the constraint grammar against the current partial output. For complex JSON schemas or regex patterns, this can take 1-5ms per token, which is comparable to the model forward pass itself for small models.

2. **No batch optimization**. When different requests in the same batch have different constraints, the masking logic runs independently for each request. There is no opportunity to share computation across requests with similar schemas.

3. **Scheduling blindness**. The scheduler has no visibility into constraint state. It cannot predict which tokens will be masked, leading to wasted computation: the model generates logits for tokens that will never be selected.

### SGLang's Integrated Approach

SGLang moves structured generation into the engine's core loop. Instead of masking after logit computation, the engine:

1. **Pre-computes constraint automata**. When a request specifies a JSON schema or regex, SGLang compiles it into a finite state automaton (FSA) at request submission time. This compilation happens once, not per token.

2. **Batches constraint evaluation**. Requests with identical schemas share the same compiled FSA. The engine groups these requests and evaluates constraint transitions in parallel using CUDA kernels.

3. **Integrates with the scheduler**. The scheduler knows which tokens are valid for each request at each step. It can skip unnecessary logit computation for tokens that are predetermined by the grammar (e.g., the opening `{` of a JSON object).

4. **Jump-forward optimization**. When the grammar uniquely determines the next several tokens (e.g., after `{"name":` the next token must be `"`), SGLang skips the model forward pass entirely and directly appends those tokens. This can skip 20-40% of decode steps for structured outputs.

```python
# Constrained decoding with SGLang (user-facing API)
import sglang as sgl

@sgl.function
def extract_person(s, text):
    s += "Extract person info from: " + text + "\n"
    s += "Output JSON:\n"
    s += sgl.gen("result", 
                 max_tokens=256,
                 regex=r'\{"name": "[^"]+", "age": \d+, "city": "[^"]+"\}')

# The regex is compiled to an FSA once, then reused across all calls
```

### The Jump-Forward Mechanism in Detail

Jump-forward is SGLang's most impactful optimization for structured workloads. Consider generating a JSON object with schema `{"name": string, "age": integer}`:

```
Token 1: {           <- determined by grammar (only valid start)
Token 2: "           <- determined (object key must start with quote)  
Token 3: n           <- determined (only field is "name")
Token 4: a           <- determined
Token 5: m           <- determined
Token 6: e           <- determined
Token 7: "           <- determined (close quote)
Token 8: :           <- determined (key-value separator)
Token 9: " or space  <- determined (string value starts)
Token 10-N: [model generates actual name]  <- ACTUAL GENERATION NEEDED
Token N+1: "         <- determined (close string)
Token N+2: ,         <- determined (more fields follow)
...
```

Of the ~25 tokens in a typical 2-field JSON response, only 5-8 require actual model inference. The rest are deterministic given the schema. Jump-forward skips all deterministic tokens in a single step, reducing decode iterations by 60-75% for highly structured outputs.

The savings compound with schema complexity: a nested 3-level JSON object with 15 fields might have 120 total tokens but only 30 that require model generation.

### Performance Impact of Engine-Level Constraints

The difference between post-hoc masking and engine-level integration is dramatic for structured workloads:

| Workload | Post-hoc masking (tokens/s) | SGLang integrated (tokens/s) | Speedup |
|----------|----------------------------|------------------------------|---------|
| Simple JSON (5 fields) | 850 | 4,200 | 4.9x |
| Nested JSON (3 levels) | 620 | 3,800 | 6.1x |
| Complex regex (email + phone) | 740 | 5,100 | 6.9x |
| SQL query generation | 580 | 4,500 | 7.8x |
| Tool call (function + args) | 690 | 4,800 | 7.0x |

*Benchmarks from SGLang paper (arXiv 2312.07104), Llama-2 7B, A100 80GB, batch size 32.*

The speedup comes from three sources: (1) jump-forward skipping model calls for deterministic tokens, (2) batched FSA evaluation on GPU, and (3) elimination of CPU-side grammar evaluation overhead.

---

## The SGLang Programming Model

### Beyond Request/Response: Programs as First-Class Constructs

Most serving engines expose a simple interface: send a prompt, get a completion. SGLang exposes a programming model where you define generation *programs* that combine multiple generation calls, control flow, and constraints into a single optimizable unit:

```python
import sglang as sgl

@sgl.function
def multi_step_reasoning(s, question):
    # Step 1: Generate initial reasoning
    s += f"Question: {question}\n"
    s += "Let me think step by step.\n"
    s += sgl.gen("thinking", max_tokens=256, temperature=0.7)
    
    # Step 2: Extract the answer from reasoning
    s += "\nTherefore, the answer is: "
    s += sgl.gen("answer", max_tokens=50, temperature=0.0)
    
    # Step 3: Verify with a different prompt structure
    s += "\n\nVerification: Is this correct? "
    s += sgl.gen("verification", 
                 choices=["Yes, this is correct.", "No, let me reconsider."])
```

The key insight: because all three generation steps are part of the same program, SGLang can:
- Keep the KV cache from step 1 alive for steps 2 and 3 (no re-prefill).
- Schedule all three steps as a unit, avoiding the overhead of three separate API calls.
- Optimize the combined program (e.g., if step 3 always selects "Yes", future runs can skip it).

Without a programming model, the same workflow requires three separate API calls. Each call re-sends the full context (or relies on external session management), and the serving engine has no visibility into the relationship between calls. The accumulated overhead of re-prefilling shared context across multiple calls is substantial: for a 2000-token context with 3 steps, you pay 6000 prefill tokens instead of 2000.

### Fork and Join: Parallel Generation

SGLang supports forking a generation into multiple parallel branches, then joining the results. This is essential for techniques like best-of-N sampling, tree-of-thought reasoning, and parallel tool calling:

```python
@sgl.function
def parallel_tool_calls(s, user_query):
    s += f"User: {user_query}\n"
    s += "I need to call multiple tools:\n"
    
    # Fork into 3 parallel tool calls
    # All share the prefix KV cache via RadixAttention
    forks = s.fork(3)
    
    forks[0] += "Tool: search\nQuery: "
    forks[0] += sgl.gen("search_query", max_tokens=50)
    
    forks[1] += "Tool: calculator\nExpression: "
    forks[1] += sgl.gen("calc_expr", max_tokens=30, 
                        regex=r'[\d+\-*/().\s]+')
    
    forks[2] += "Tool: calendar\nAction: "
    forks[2] += sgl.gen("calendar_action", max_tokens=40,
                        regex=r'\{"action": "(create|read|update)", "date": "\d{4}-\d{2}-\d{2}"\}')
    
    # Join results back
    s += sgl.join(forks)
    s += "\nBased on tool results, my answer is: "
    s += sgl.gen("final_answer", max_tokens=200)
```

The fork operation creates branches that share the parent's KV cache via RadixAttention. This means the system prompt and conversation history (which might be 2000+ tokens) are computed once and shared across all three tool call branches. Without RadixAttention, each branch would need to re-prefill the entire prefix.

The memory savings from fork/join are proportional to the prefix length and number of branches:

```
Memory without fork: prefix_tokens * num_branches * 2 * d_model * num_layers
Memory with fork:    prefix_tokens * 1 * 2 * d_model * num_layers + branch_tokens * num_branches * 2 * d_model * num_layers

For Llama-3 70B with 2000-token prefix, 3 branches of 100 tokens each:
  Without: 2000 * 3 * 2 * 8192 * 80 = 7.5 GB
  With:    2000 * 1 * 2 * 8192 * 80 + 100 * 3 * 2 * 8192 * 80 = 2.9 GB (61% savings)
```

### Select: Efficient Classification

The `select` primitive generates a token sequence that matches one of a predefined set of options. This is more efficient than generating freely and checking:

```python
@sgl.function
def classify_intent(s, message):
    s += f"Classify the intent of: '{message}'\n"
    s += "Intent: "
    s += sgl.select("intent", 
                    choices=["question", "command", "feedback", "complaint"])
```

Under the hood, `select` uses the log-probabilities of each choice's token sequence to pick the most likely option without generating tokens one by one. For N choices of average length L, this reduces from O(N*L) sequential decode steps to a single batched forward pass that evaluates all choices simultaneously.

The implementation tokenizes each choice, runs a single forward pass to get logits for the next position, then follows each choice's token sequence through the model to compute cumulative log-probability. The choice with the highest total log-probability wins. For 4 choices of 3 tokens each, this requires 1 forward pass instead of potentially 12 sequential decode steps.

### Composition: Building Complex Agents

These primitives compose naturally into agent-like patterns:

```python
@sgl.function
def react_agent(s, task, tools, max_steps=5):
    s += f"Task: {task}\n"
    s += f"Available tools: {', '.join(tools)}\n\n"
    
    for step in range(max_steps):
        # Decide action
        s += f"Step {step + 1}:\n"
        s += "Thought: "
        s += sgl.gen("thought", max_tokens=150, stop="\n")
        
        # Select tool or finish
        s += "\nAction: "
        action = sgl.select("action", choices=tools + ["finish"])
        s += action
        
        if action == "finish":
            s += "\nFinal Answer: "
            s += sgl.gen("answer", max_tokens=200)
            break
        
        # Generate structured tool input
        s += "\nAction Input: "
        s += sgl.gen("input", max_tokens=100, 
                     regex=r'\{[^}]+\}')  # Must be valid JSON
        
        # Simulate tool response (in production, call actual tool)
        s += "\nObservation: [tool result]\n\n"
```

This entire agent loop runs as a single SGLang program. The KV cache grows incrementally across steps (no re-prefill between steps), structured constraints ensure tool inputs are valid JSON, and the select primitive efficiently picks tools without wasteful generation.

### Batch Execution: Processing Many Inputs

SGLang programs can be batched across multiple inputs, sharing the compiled program structure:

```python
@sgl.function
def extract_entities(s, text):
    s += "Extract all named entities from the following text.\n"
    s += f"Text: {text}\n"
    s += "Entities (JSON array):\n"
    s += sgl.gen("entities", max_tokens=200,
                 regex=r'\[("([^"]+)"(, )?)*\]')

# Process 1000 texts in one batch
# SGLang automatically:
# 1. Shares the instruction prefix across all 1000 requests
# 2. Compiles the regex FSA once
# 3. Schedules based on cache locality
texts = load_texts()  # 1000 documents
results = extract_entities.run_batch(
    [{"text": t} for t in texts],
    num_threads=64,
    progress_bar=True
)
```

---

## Performance Architecture

### Why SGLang Is Fast: The Full Picture

SGLang's performance advantage comes from the interaction of multiple optimizations, not any single feature:

```
                    +---------------------------------------------+
                    |         SGLang Performance Stack             |
                    +---------------------------------------------+
                    |  Programming Model (fork/join/select)        |
                    |  -> Reduces total generation steps           |
                    +---------------------------------------------+
                    |  Jump-Forward Optimization                   |
                    |  -> Skips deterministic tokens               |
                    +---------------------------------------------+
                    |  Batched FSA Evaluation (GPU)                |
                    |  -> Constraint checking at GPU speed         |
                    +---------------------------------------------+
                    |  RadixAttention (tree cache)                 |
                    |  -> Eliminates redundant prefill             |
                    +---------------------------------------------+
                    |  Continuous Batching + PagedAttention        |
                    |  -> Standard serving optimizations           |
                    +---------------------------------------------+
```

Each layer multiplies the effect of the layers below it. RadixAttention reduces prefill work. Jump-forward reduces decode steps. Batched FSA reduces per-step overhead. The programming model reduces the total number of separate requests.

### Benchmark: Structured Output Workloads

For workloads that require JSON/structured output (increasingly common with tool-calling agents), SGLang shows dramatic advantages:

**Agent tool-calling benchmark** (Llama-2 70B, 8xA100):
- 1000 requests, each requiring 2-4 tool calls with JSON-formatted arguments
- All requests share the same 1500-token system prompt

| Engine | Throughput (req/s) | Median latency (ms) | P99 latency (ms) |
|--------|-------------------|---------------------|-------------------|
| vLLM (no constraints) | 42 | 890 | 2,100 |
| vLLM + outlines | 28 | 1,340 | 3,200 |
| TGI + grammar | 31 | 1,210 | 2,900 |
| SGLang | 186 | 320 | 780 |

The 4.4x throughput advantage over vLLM (without constraints) and 6.6x over vLLM+outlines comes from:
- RadixAttention reuses the 1500-token system prompt across all 1000 requests (saves 1.5M prefill tokens).
- Jump-forward skips approximately 30% of decode steps (JSON delimiters, field names).
- Fork/join executes multiple tool calls without re-sending the conversation prefix.

### Benchmark: Prefix-Heavy Workloads

For RAG and chatbot workloads where many requests share long prefixes:

**RAG benchmark** (Mixtral 8x7B, 4xA100):
- 500 queries, each retrieving 3 documents (avg 800 tokens each)
- 60% of documents are shared across queries (realistic for popular topics)

| Engine | Throughput (req/s) | Cache hit rate | GPU memory used |
|--------|-------------------|----------------|-----------------|
| vLLM (no APC) | 38 | 0% | 76 GB |
| vLLM (APC enabled) | 52 | 41% | 68 GB |
| SGLang | 124 | 78% | 54 GB |

SGLang achieves higher cache hit rates because the radix tree matches partial prefixes that vLLM's block-aligned hashing misses. The memory savings compound: less memory spent on duplicate KV cache entries means more memory available for larger batches, which further increases throughput.

### Benchmark: Multi-Turn Chat at Scale

**Chatbot benchmark** (Llama-3 8B, single A100):
- 200 concurrent conversations, average 8 turns each
- 800-token system prompt shared across all conversations
- Average user message: 50 tokens, average response: 150 tokens

| Engine | Throughput (turns/s) | Avg TTFT (ms) | Memory efficiency |
|--------|---------------------|---------------|-------------------|
| vLLM | 89 | 245 | 1.0x (baseline) |
| SGLang | 203 | 112 | 1.4x |

The TTFT (time to first token) improvement comes directly from RadixAttention: on turn 8 of a conversation, SGLang only prefills the new user message (50 tokens) because the entire conversation history is already in the radix tree. vLLM with APC gets partial benefit but misses matches when conversation turns do not align with block boundaries.

### Worked Example: RadixAttention Savings for a RAG Pipeline

Consider a production RAG system serving 100 concurrent users. Each query retrieves 3 documents (average 600 tokens each) and generates a response conditioned on those documents plus a 400-token system prompt.

**Without RadixAttention (vLLM, no APC):**
Each request prefills independently:
```
Per request: 400 (system) + 3 * 600 (docs) + 50 (query) = 2,250 tokens prefill
100 concurrent: 225,000 total prefill tokens
At 50,000 tokens/s prefill throughput: 4.5 seconds to process batch
```

**With vLLM APC:**
System prompt is cached (400 tokens saved per request). Documents are cached only if they align with block boundaries AND were recently accessed in the same block order:
```
Per request: 0 (system cached) + ~1,200 (2 of 3 docs partially cached) + 50 (query) = 1,250 tokens
100 concurrent: 125,000 total prefill tokens (44% reduction)
```

**With SGLang RadixAttention:**
The radix tree shares the system prompt AND any document that was recently retrieved by any user, regardless of retrieval order or block alignment. In a typical RAG workload, 60% of retrieved documents are "popular" (shared across users):
```
Per request: 0 (system) + 600 (1 unique doc) + 50 (query) = 650 tokens average
100 concurrent: 65,000 total prefill tokens (71% reduction)
GPU memory for shared docs: stored once, referenced by tree edges
```

The 71% reduction in prefill tokens translates directly to:
- 3.5x higher throughput (same GPU processes 3.5x more requests per second)
- 60% lower time-to-first-token (less prefill work before generation starts)
- 40% less GPU memory for KV cache (shared entries stored once)

This advantage grows with the "sharing factor" of the workload. A customer support chatbot where 95% of conversations use the same 2000-token system prompt sees even larger gains. A code completion engine where every request shares a 5000-token repository context sees massive savings.

### SGLang vs. vLLM: Architecture Comparison

Understanding where SGLang differs architecturally from vLLM clarifies when each is the right choice:

| Component | vLLM | SGLang |
|-----------|------|--------|
| KV cache management | PagedAttention (paged blocks) | PagedAttention + RadixAttention (tree overlay) |
| Scheduling | FCFS or priority queue | Cache-aware (LPM policy) |
| Prefix caching | APC (hash-based, block-aligned) | Radix tree (variable-length, O(log n)) |
| Structured output | External (outlines, guidance) | Native (compiled FSA, jump-forward) |
| Multi-step workflows | Multiple API calls | Single program (fork/join/select) |
| API compatibility | OpenAI-compatible | OpenAI-compatible + native SGLang API |
| Model support | Broadest (100+ architectures) | Growing (major architectures covered) |
| Quantization | AWQ, GPTQ, FP8, INT8 | AWQ, GPTQ, FP8 (slightly fewer options) |
| Speculative decoding | Supported | Supported |
| LoRA serving | Dynamic LoRA loading | Dynamic LoRA loading |

Both engines build on the same foundation (PagedAttention for memory efficiency, continuous batching for throughput). SGLang adds the radix tree layer and programming model on top. The trade-off is clear: SGLang offers higher performance for structured/prefix-heavy workloads at the cost of slightly narrower model support.


---

## Deployment and Operations

### Installation and Basic Serving

```bash
# Install SGLang
pip install "sglang[all]"

# Launch server (OpenAI-compatible API)
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --port 30000 \
    --tp 1 \
    --mem-fraction-static 0.85
```

SGLang exposes an OpenAI-compatible API, making it a drop-in replacement for vLLM or TGI in most deployments. The structured generation features are accessed via additional parameters:

```python
import openai

client = openai.Client(base_url="http://localhost:30000/v1")

# Standard completion (works like any OpenAI-compatible server)
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Hello"}]
)

# Structured output via JSON schema
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Extract: John is 30, lives in NYC"}],
    extra_body={
        "json_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"}
            },
            "required": ["name", "age", "city"]
        }
    }
)
```

### Multi-GPU and Tensor Parallelism

```bash
# Tensor parallelism across 4 GPUs
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-70B-Instruct \
    --tp 4 \
    --port 30000

# Data parallelism (multiple independent replicas)
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --dp 4 \
    --port 30000

# Combined: 2 replicas, each using 4 GPUs for tensor parallelism
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-70B-Instruct \
    --tp 4 --dp 2 \
    --port 30000
```

### Configuration for Production

Key parameters that affect RadixAttention and structured generation performance:

```bash
python -m sglang.launch_server \
    --model-path $MODEL \
    --tp 4 \
    --mem-fraction-static 0.85 \        # Fraction of GPU memory for KV cache
    --max-prefill-tokens 16384 \        # Max tokens per prefill batch
    --schedule-policy lpm \             # Longest Prefix Match scheduling
    --chunk-prefill-size 4096 \         # Chunked prefill for long prompts
    --enable-mixed-chunk \              # Mix prefill and decode in same batch
    --max-running-requests 256           # Concurrency limit
```

The `--schedule-policy lpm` (Longest Prefix Match) is critical for RadixAttention performance. It tells the scheduler to prioritize requests whose prefixes are already in the radix tree, maximizing cache hit rates. Without this flag, the scheduler uses FCFS (first come, first served), which can evict hot prefixes before similar requests arrive.

### Monitoring RadixAttention Cache Performance

SGLang exposes metrics for monitoring cache effectiveness:

```python
import requests

# Get runtime metrics
metrics = requests.get("http://localhost:30000/get_server_info").json()

# Key metrics to monitor:
# - cache_hit_rate: fraction of prefill tokens served from cache
# - radix_tree_size: number of nodes in the tree
# - eviction_count: how often cache entries are evicted
# - avg_prefix_match_len: average tokens matched per request

# Healthy indicators:
# cache_hit_rate > 0.6 for chatbot workloads
# cache_hit_rate > 0.4 for RAG workloads
# eviction_count should be stable (not growing linearly)
```

If cache hit rates are below expectations, common causes include:
- Insufficient `--mem-fraction-static` (not enough memory for the tree)
- High request diversity (few shared prefixes in the workload)
- Using FCFS scheduling instead of LPM

---

## When to Choose SGLang

### Ideal Workloads

SGLang provides the strongest advantages for these workload patterns:

**1. Agent/tool-calling systems**. Agents generate structured tool calls (JSON arguments), receive results, and generate more calls. SGLang's combination of structured generation (ensuring valid JSON), RadixAttention (sharing system prompts and conversation history), and fork/join (parallel tool calls) directly targets this pattern. Expected improvement: 5-7x throughput over vLLM+outlines.

**2. High prefix-sharing deployments**. Any scenario where many concurrent requests share long prefixes: chatbots with the same system prompt, RAG systems retrieving from the same document corpus, or batch processing tasks with identical instructions. Expected improvement: 2-3x throughput, 30-50% memory reduction.

**3. Complex generation pipelines**. Workloads that require multiple generation steps (chain-of-thought, self-verification, extract-then-summarize) benefit from SGLang programs that keep KV cache alive across steps. Expected improvement: 2-4x end-to-end latency reduction from eliminated re-prefills.

**4. Structured output at scale**. APIs that must return valid JSON, SQL, or other constrained formats. The jump-forward optimization alone can double throughput for highly structured outputs. Expected improvement: 5-10x for schema-heavy workloads.

**5. Batch inference with shared context**. Processing thousands of inputs with the same instruction template (data extraction, classification, transformation). The combination of RadixAttention (shares the instruction prefix) and batch execution mode makes this dramatically faster than sequential API calls.

### When NOT to Use SGLang

**Simple completion tasks with diverse prompts**. If your workload is "send unique prompt, get text back" with no constraints and minimal prefix sharing, vLLM is simpler to deploy and has a larger ecosystem of integrations. The radix tree adds overhead when there is nothing to share.

**Maximum model architecture coverage**. SGLang supports fewer model architectures than vLLM. If you need to serve a niche model (custom architectures, rare quantization formats), vLLM or HuggingFace TGI are safer choices. Check SGLang's supported model list before committing.

**Latency-optimized single-request serving**. For scenarios where you care about the absolute latency of a single isolated request (not throughput under load), TensorRT-LLM with custom kernels may offer lower per-request latency due to kernel-level optimizations.

**Existing well-tuned vLLM infrastructure**. If you have a production vLLM deployment that meets your SLOs and your workload characteristics do not strongly favor SGLang's features, the migration cost may not justify the improvement. Measure before migrating.

**Extremely long sequences with no sharing**. For workloads dominated by unique 100K+ token contexts (long document summarization with unique documents), the radix tree provides minimal benefit since each request's prefix is unique.

### Decision Framework

```
Is your workload primarily structured output (JSON, SQL, regex)?
  YES -> SGLang (5-10x advantage from jump-forward + batched FSA)
  NO  -> continue

Do many requests share long prefixes (>500 tokens)?
  YES -> SGLang (2-3x from RadixAttention vs hash-based APC)
  NO  -> continue

Do you need multi-step generation (agent loops, chain-of-thought)?
  YES -> SGLang (programs keep KV cache alive across steps)
  NO  -> continue

Is it simple text completion with diverse prompts?
  YES -> vLLM (simpler deployment, broader model support)

Do you need maximum single-request latency?
  YES -> TensorRT-LLM (custom CUDA kernels for every op)
```

---

## Mental Model

Think of the serving engine landscape in terms of what each engine optimizes:

- **vLLM** optimizes *memory efficiency* (PagedAttention) and *deployment simplicity* (broad model support, OpenAI API compatibility). It answers: "How do I serve any model without running out of GPU memory?"
- **TensorRT-LLM** optimizes *kernel performance* (custom CUDA kernels for every operation, maximum single-request speed). It answers: "How do I make each individual inference call as fast as physically possible?"
- **SGLang** optimizes *computation reuse* (RadixAttention eliminates redundant prefill) and *generation programs* (structured output, multi-step workflows, fork/join patterns). It answers: "How do I avoid doing the same work twice, and how do I express complex generation patterns efficiently?"

SGLang is a programming language for LLM inference, not just a serving engine. Where vLLM gives you an API endpoint and TensorRT-LLM gives you optimized kernels, SGLang gives you a way to express what you want the model to produce and how different generation steps relate to each other. The engine then optimizes the entire program, not just individual requests.

As workloads shift from simple chat completion toward agentic patterns (tool calling, structured output, multi-step reasoning), the programming model becomes the bottleneck. An engine that understands the structure of your workload can eliminate redundant computation that request-level optimizations cannot see. This is why SGLang's throughput advantage grows with workload complexity: the more structure in your generation pattern, the more the engine can optimize.

The trajectory is clear: today's agent frameworks make multiple independent API calls per user request. Tomorrow's frameworks will express entire agent programs as optimizable computation graphs. SGLang is building that future today.

### Ecosystem and Community

SGLang is developed at UC Berkeley's Sky Computing Lab, the same group that produced vLLM, Vicuna, and Chatbot Arena. The project has seen rapid adoption since its ICLR 2024 oral presentation:

- Used internally by LMSYS to power Chatbot Arena's multi-model inference (serving 20+ models simultaneously with shared infrastructure).
- Adopted by several startups building agent platforms where structured output reliability is critical.
- Growing contributor community with bi-weekly releases and active Discord.
- Integration with major model providers: supports Llama, Mistral, Qwen, Gemma, and other popular architectures.

The project's roadmap focuses on three areas: (1) expanding model architecture support to match vLLM's breadth, (2) deeper integration with agent frameworks (LangChain, LlamaIndex, DSPy), and (3) multi-node serving with distributed radix trees for cluster-scale prefix sharing.


---

## References

1. Zheng, L., Yin, L., Xie, Z., et al. "SGLang: Efficient Execution of Structured Language Model Programs." arXiv:2312.07104, 2023.
2. Zheng, L., et al. "Efficiently Programming Large Language Models using SGLang." ICLR 2024 (oral presentation).
3. SGLang GitHub repository: https://github.com/sgl-project/sglang
4. RadixAttention: described in SGLang paper Section 3.1, extending classical radix tree data structures to KV cache management.
5. Willard, B., Louf, R. "Efficient Guided Generation for Large Language Models." arXiv:2307.09702, 2023. (Outlines framework, comparison baseline for structured generation.)
6. Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023. (vLLM, the baseline system SGLang builds upon.)
7. Zheng, L., et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." NeurIPS 2023. (LMSYS Chatbot Arena, built using SGLang for efficient multi-model inference.)
