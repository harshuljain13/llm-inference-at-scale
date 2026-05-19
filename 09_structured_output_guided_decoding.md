# Module 9: Structured Output and Guided Decoding

> Constraining LLM outputs for reliable, parseable responses

---

## Learning Objectives

By the end of this module, you will:

- Implement JSON schema-constrained generation
- Configure guided decoding backends in vLLM and SGLang
- Use regex and grammar constraints for structured output
- Apply structured output for function calling and tool use

---

## Why Structured Output Matters

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE STRUCTURED OUTPUT PROBLEM                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Without Constraints:                                              │
│   ════════════════════                                              │
│                                                                     │
│   Prompt: "Extract the name and age from: John is 25 years old"     │
│                                                                     │
│   Response 1: {"name": "John", "age": 25}           ✓ Valid JSON    │
│   Response 2: The name is John and age is 25        ✗ Not JSON      │
│   Response 3: {"name": "John", "age": "25"}         ✗ Wrong type    │
│   Response 4: {name: John, age: 25}                 ✗ Invalid JSON  │
│                                                                     │
│   Problem: ~20-30% of responses may be malformed                    │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   With Guided Decoding:                                             │
│   ═════════════════════                                             │
│                                                                     │
│   • 100% valid JSON guaranteed                                      │
│   • Correct types enforced                                          │
│   • Required fields always present                                  │
│   • No post-processing/retry needed                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Guided Decoding Approaches

### Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GUIDED DECODING METHODS                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Method          │ Constraint Type    │ Use Case                   │
│   ════════════════│════════════════════│════════════════════════    │
│                   │                    │                            │
│   JSON Schema     │ Pydantic/JSON      │ Structured data extraction │
│   Regex           │ Pattern matching   │ Formats (email, phone)     │
│   Choice          │ Enum selection     │ Classification             │
│   Grammar (CFG)   │ Context-free       │ Code, complex structures   │
│                   │                    │                            │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   How It Works:                                                     │
│   ══════════════                                                    │
│                                                                     │
│   1. Define constraint (schema, regex, grammar)                     │
│   2. At each token generation step:                                 │
│      a. Get logits from model                                       │
│      b. Mask invalid tokens (set logits to -inf)                    │
│      c. Sample from valid tokens only                               │
│   3. Result: 100% compliant output                                  │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  Logits: [0.1, 0.3, 0.2, 0.4, ...]                          │   │
│   │            ↓                                                │   │
│   │  Mask:   [  ✓,   ✗,   ✓,   ✗, ...]  (based on constraint)  │   │
│   │            ↓                                                │   │
│   │  Result: [0.1, -∞, 0.2, -∞, ...]                            │   │
│   │            ↓                                                │   │
│   │  Sample from valid tokens only                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Backend Comparison

| Backend            | Engine         | Speed   | Features            | Best For            |
| ------------------ | -------------- | ------- | ------------------- | ------------------- |
| outlines           | vLLM (default) | Fast    | JSON, regex, CFG    | General use         |
| lm-format-enforcer | vLLM           | Medium  | JSON, regex         | Alternative         |
| xgrammar           | vLLM, SGLang   | Fastest | JSON, regex, CFG    | High throughput     |
| SGLang native      | SGLang         | Fast    | JSON, regex, choice | Multi-step programs |

---

## vLLM Structured Output

### JSON Schema with Pydantic

```python
# vllm_structured_output.py
"""Structured output generation with vLLM."""

from vllm import LLM, SamplingParams
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum


# Define output schemas using Pydantic
class EntityType(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    DATE = "DATE"
    MONEY = "MONEY"


class Entity(BaseModel):
    """A named entity extracted from text."""
    name: str = Field(description="The entity text")
    type: EntityType = Field(description="The entity type")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")


class ExtractionResult(BaseModel):
    """Result of entity extraction."""
    entities: List[Entity] = Field(description="List of extracted entities")
    summary: str = Field(max_length=200, description="Brief summary")


class SentimentResult(BaseModel):
    """Sentiment analysis result."""
    sentiment: Literal["positive", "negative", "neutral"]
    score: float = Field(ge=-1.0, le=1.0)
    reasoning: str = Field(max_length=100)


# Initialize vLLM
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    guided_decoding_backend="outlines",  # or "lm-format-enforcer", "xgrammar"
)


def extract_entities(text: str) -> ExtractionResult:
    """Extract entities with guaranteed valid JSON output."""

    prompt = f"""Extract named entities from the following text.
Output as JSON matching the schema.

Text: {text}

JSON:"""

    sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=500,
    )

    outputs = llm.generate(
        [prompt],
        sampling_params,
        guided_options_request={
            "guided_json": ExtractionResult.model_json_schema(),
        },
    )

    # Parse guaranteed-valid JSON
    result = ExtractionResult.model_validate_json(
        outputs[0].outputs[0].text
    )
    return result


def analyze_sentiment(text: str) -> SentimentResult:
    """Analyze sentiment with constrained output."""

    prompt = f"""Analyze the sentiment of the following text.

Text: {text}

JSON:"""

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=200,
    )

    outputs = llm.generate(
        [prompt],
        sampling_params,
        guided_options_request={
            "guided_json": SentimentResult.model_json_schema(),
        },
    )

    return SentimentResult.model_validate_json(
        outputs[0].outputs[0].text
    )


# Example usage
if __name__ == "__main__":
    # Entity extraction
    text = "Apple Inc. CEO Tim Cook announced a $100 billion investment in California on January 15, 2024."
    result = extract_entities(text)
    print(f"Entities: {result.entities}")
    print(f"Summary: {result.summary}")

    # Sentiment analysis
    review = "This product exceeded my expectations! Great quality and fast shipping."
    sentiment = analyze_sentiment(review)
    print(f"Sentiment: {sentiment.sentiment} ({sentiment.score})")
```

### Regex Constraints

```python
# vllm_regex_constraints.py
"""Regex-constrained generation with vLLM."""

from vllm import LLM, SamplingParams


llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")


def generate_email(context: str) -> str:
    """Generate a valid email address."""

    prompt = f"""Based on the context, generate an appropriate email address.
Context: {context}
Email:"""

    # Email regex pattern
    email_pattern = r'[a-z][a-z0-9._]{2,20}@[a-z]{2,10}\.(com|org|net|edu)'

    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.7, max_tokens=50),
        guided_options_request={"guided_regex": email_pattern},
    )

    return outputs[0].outputs[0].text


def generate_phone(context: str) -> str:
    """Generate a valid US phone number."""

    prompt = f"""Generate a phone number for: {context}
Phone:"""

    # US phone pattern
    phone_pattern = r'\(\d{3}\) \d{3}-\d{4}'

    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.7, max_tokens=20),
        guided_options_request={"guided_regex": phone_pattern},
    )

    return outputs[0].outputs[0].text


def generate_date(context: str) -> str:
    """Generate a valid ISO date."""

    prompt = f"""Generate an appropriate date for: {context}
Date:"""

    # ISO date pattern
    date_pattern = r'20[2-3][0-9]-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])'

    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.7, max_tokens=15),
        guided_options_request={"guided_regex": date_pattern},
    )

    return outputs[0].outputs[0].text


# Example usage
if __name__ == "__main__":
    print(generate_email("John Smith, software engineer at TechCorp"))
    # Output: john.smith@techcorp.com

    print(generate_phone("Customer support line"))
    # Output: (800) 555-1234

    print(generate_date("Next quarterly review"))
    # Output: 2024-04-15
```

### Choice Constraints

```python
# vllm_choice_constraints.py
"""Choice-constrained generation with vLLM."""

from vllm import LLM, SamplingParams
from typing import List


llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")


def classify_intent(query: str, intents: List[str]) -> str:
    """Classify user intent from predefined options."""

    prompt = f"""Classify the user's intent.

Query: {query}

Intent:"""

    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.1, max_tokens=20),
        guided_options_request={"guided_choice": intents},
    )

    return outputs[0].outputs[0].text


def route_request(query: str) -> str:
    """Route request to appropriate handler."""

    handlers = [
        "billing_support",
        "technical_support",
        "sales_inquiry",
        "general_question",
        "complaint",
    ]

    prompt = f"""Route this customer request to the appropriate department.

Request: {query}

Department:"""

    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.1, max_tokens=20),
        guided_options_request={"guided_choice": handlers},
    )

    return outputs[0].outputs[0].text


# Example usage
if __name__ == "__main__":
    # Intent classification
    intents = ["search", "purchase", "return", "track_order", "other"]
    query = "Where is my package?"
    intent = classify_intent(query, intents)
    print(f"Intent: {intent}")  # Output: track_order

    # Request routing
    request = "I was charged twice for my subscription"
    department = route_request(request)
    print(f"Route to: {department}")  # Output: billing_support
```

---

## SGLang Structured Generation

### RadixAttention for Multi-Step Programs

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SGLANG RADIXATTENTION                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Traditional Approach:                                             │
│   ═════════════════════                                             │
│                                                                     │
│   Step 1: [System + User + Gen1] → KV cache discarded               │
│   Step 2: [System + User + Gen1 + Gen2] → Recompute all             │
│   Step 3: [System + User + Gen1 + Gen2 + Gen3] → Recompute all      │
│                                                                     │
│   Problem: Redundant computation for shared prefixes                │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   SGLang RadixAttention:                                            │
│   ══════════════════════                                            │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    Radix Tree (KV Cache)                    │   │
│   │                                                             │   │
│   │                      [System Prompt]                        │   │
│   │                            │                                │   │
│   │              ┌─────────────┼─────────────┐                  │   │
│   │              ▼             ▼             ▼                  │   │
│   │         [User A]      [User B]      [User C]                │   │
│   │              │             │             │                  │   │
│   │         [Gen A1]      [Gen B1]      [Gen C1]                │   │
│   │              │             │                                │   │
│   │         [Gen A2]      [Gen B2]                              │   │
│   │                                                             │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   Benefit: Shared prefixes computed once, reused across requests    │
│   Speedup: 2-5x for multi-turn conversations                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### SGLang Structured Generation Examples

```python
# sglang_structured.py
"""Structured generation with SGLang."""

import sglang as sgl
from sglang import RuntimeEndpoint


# Connect to SGLang server
# Start server: python -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --port 30000
runtime = RuntimeEndpoint("http://localhost:30000")


@sgl.function
def extract_entities_sglang(s, text: str):
    """Extract entities with regex-constrained JSON."""

    s += f"Extract named entities from: {text}\n"
    s += "Output as JSON array:\n"

    # Regex for JSON array of entities
    entity_pattern = r'\[\s*(\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"type"\s*:\s*"(PERSON|ORG|LOCATION)"\s*\}\s*,?\s*)*\]'

    s += sgl.gen("entities", regex=entity_pattern, max_tokens=500)


@sgl.function
def classify_with_reasoning(s, text: str, categories: list):
    """Classify with chain-of-thought reasoning."""

    s += f"Classify the following text into one of these categories: {categories}\n\n"
    s += f"Text: {text}\n\n"
    s += "Reasoning: "
    s += sgl.gen("reasoning", max_tokens=100, stop="\n")
    s += "\nCategory: "
    s += sgl.gen("category", choices=categories)


@sgl.function
def multi_step_extraction(s, document: str):
    """Multi-step document processing."""

    s += f"Document: {document}\n\n"

    # Step 1: Extract summary
    s += "Step 1 - Summary (one sentence):\n"
    s += sgl.gen("summary", max_tokens=50, stop="\n")

    # Step 2: Extract key points
    s += "\n\nStep 2 - Key points (JSON array):\n"
    s += sgl.gen(
        "key_points",
        regex=r'\[\s*"[^"]+"\s*(,\s*"[^"]+"\s*)*\]',
        max_tokens=200
    )

    # Step 3: Sentiment
    s += "\n\nStep 3 - Overall sentiment:\n"
    s += sgl.gen("sentiment", choices=["positive", "negative", "neutral"])


@sgl.function
def function_calling(s, user_query: str):
    """Function calling with structured output."""

    s += f"User query: {user_query}\n\n"
    s += "Select the appropriate function to call:\n"

    # First, select function
    s += sgl.gen(
        "function",
        choices=["search_web", "get_weather", "send_email", "calculate", "none"]
    )

    # Then, generate parameters based on function
    with sgl.match("function"):
        with sgl.case("search_web"):
            s += "\nSearch query: "
            s += sgl.gen("query", max_tokens=50, stop="\n")

        with sgl.case("get_weather"):
            s += "\nLocation: "
            s += sgl.gen(
                "location",
                regex=r'[A-Za-z\s]+,\s*[A-Z]{2}',  # City, ST format
                max_tokens=30
            )

        with sgl.case("send_email"):
            s += "\nRecipient: "
            s += sgl.gen(
                "recipient",
                regex=r'[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}',
                max_tokens=50
            )
            s += "\nSubject: "
            s += sgl.gen("subject", max_tokens=30, stop="\n")

        with sgl.case("calculate"):
            s += "\nExpression: "
            s += sgl.gen(
                "expression",
                regex=r'[\d\s\+\-\*\/\(\)\.]+',
                max_tokens=50
            )

        with sgl.case("none"):
            s += "\nResponse: "
            s += sgl.gen("response", max_tokens=100)


# Batch processing with SGLang
@sgl.function
def batch_classify(s, texts: list, categories: list):
    """Batch classification for multiple texts."""

    s += f"Categories: {categories}\n\n"

    results = []
    for i, text in enumerate(texts):
        s += f"Text {i+1}: {text}\n"
        s += f"Category {i+1}: "
        s += sgl.gen(f"cat_{i}", choices=categories)
        s += "\n"


# Example usage
if __name__ == "__main__":
    # Entity extraction
    result = extract_entities_sglang.run(
        text="Apple CEO Tim Cook met with Microsoft's Satya Nadella in Seattle.",
        backend=runtime,
    )
    print(f"Entities: {result['entities']}")

    # Classification with reasoning
    result = classify_with_reasoning.run(
        text="The stock market crashed today amid recession fears.",
        categories=["business", "sports", "technology", "politics"],
        backend=runtime,
    )
    print(f"Reasoning: {result['reasoning']}")
    print(f"Category: {result['category']}")

    # Function calling
    result = function_calling.run(
        user_query="What's the weather like in New York?",
        backend=runtime,
    )
    print(f"Function: {result['function']}")
    if result['function'] == 'get_weather':
        print(f"Location: {result['location']}")
```

### SGLang Server Configuration

```bash
# Start SGLang server with structured output support
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --host 0.0.0.0 \
    --tp 1 \
    --enable-torch-compile \
    --grammar-backend xgrammar  # Fast grammar backend
```

---

## Function Calling Patterns

### OpenAI-Compatible Function Calling

```python
# function_calling_openai.py
"""OpenAI-compatible function calling with vLLM."""

from vllm import LLM, SamplingParams
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import json


# Define function schemas
class SearchWebArgs(BaseModel):
    query: str = Field(description="Search query")
    num_results: int = Field(default=5, ge=1, le=20)


class GetWeatherArgs(BaseModel):
    location: str = Field(description="City and state")
    units: str = Field(default="fahrenheit", pattern="^(celsius|fahrenheit)$")


class SendEmailArgs(BaseModel):
    to: str = Field(description="Recipient email")
    subject: str = Field(max_length=100)
    body: str = Field(max_length=1000)


class FunctionCall(BaseModel):
    name: str = Field(description="Function name")
    arguments: Union[SearchWebArgs, GetWeatherArgs, SendEmailArgs]


class ToolResponse(BaseModel):
    tool_calls: List[FunctionCall] = Field(default_factory=list)
    content: Optional[str] = Field(default=None)


# Function definitions for the model
FUNCTIONS = [
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": SearchWebArgs.model_json_schema(),
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": GetWeatherArgs.model_json_schema(),
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "parameters": SendEmailArgs.model_json_schema(),
    },
]


llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")


def process_with_tools(user_message: str) -> ToolResponse:
    """Process user message with function calling."""

    # Format prompt with function definitions
    functions_str = json.dumps(FUNCTIONS, indent=2)

    prompt = f"""You are a helpful assistant with access to the following functions:

{functions_str}

Based on the user's request, either:
1. Call one or more functions by outputting a JSON with "tool_calls" array
2. Respond directly with "content" if no function is needed

User: {user_message}

Response (JSON):"""

    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.1, max_tokens=500),
        guided_options_request={
            "guided_json": ToolResponse.model_json_schema(),
        },
    )

    return ToolResponse.model_validate_json(outputs[0].outputs[0].text)


# Example usage
if __name__ == "__main__":
    # Weather query
    response = process_with_tools("What's the weather in San Francisco?")
    print(f"Tool calls: {response.tool_calls}")

    # Direct response
    response = process_with_tools("What is 2 + 2?")
    print(f"Content: {response.content}")

    # Multi-function
    response = process_with_tools(
        "Search for Python tutorials and email the results to john@example.com"
    )
    print(f"Tool calls: {response.tool_calls}")
```

---

## Performance Considerations

### Overhead Analysis

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GUIDED DECODING OVERHEAD                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Overhead Sources:                                                 │
│   ═════════════════                                                 │
│                                                                     │
│   1. Grammar/Schema Compilation (one-time)                          │
│      • JSON schema → FSM: ~10-100ms                                 │
│      • Regex → FSM: ~1-10ms                                         │
│      • Cached after first use                                       │
│                                                                     │
│   2. Token Masking (per token)                                      │
│      • Compute valid tokens: ~0.1-1ms                               │
│      • Apply mask to logits: ~0.01ms                                │
│      • Total: ~1-5% latency overhead                                │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   Backend Performance Comparison:                                   │
│   ═══════════════════════════════                                   │
│                                                                     │
│   Backend              │ Compile Time │ Per-Token │ Best For        │
│   ─────────────────────│──────────────│───────────│─────────────    │
│   outlines             │ Medium       │ Fast      │ General use     │
│   lm-format-enforcer   │ Fast         │ Medium    │ Simple schemas  │
│   xgrammar             │ Fast         │ Fastest   │ High throughput │
│                                                                     │
│   ─────────────────────────────────────────────────────────────    │
│                                                                     │
│   Optimization Tips:                                                │
│   ══════════════════                                                │
│                                                                     │
│   1. Cache compiled schemas                                         │
│   2. Use simpler constraints when possible                          │
│   3. Prefer choice over regex for enums                             │
│   4. Use xgrammar backend for high throughput                       │
│   5. Batch requests with same schema                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Benchmarking Structured Output

```python
# benchmark_structured.py
"""Benchmark structured output performance."""

import time
from vllm import LLM, SamplingParams
from pydantic import BaseModel
from typing import List


class SimpleResult(BaseModel):
    answer: str
    confidence: float


class ComplexResult(BaseModel):
    entities: List[dict]
    summary: str
    sentiment: str
    keywords: List[str]


def benchmark_structured_output(
    llm: LLM,
    schema: type,
    num_requests: int = 100,
) -> dict:
    """Benchmark structured output generation."""

    prompt = "Analyze this text: The quick brown fox jumps over the lazy dog."

    # Warmup (compile schema)
    llm.generate(
        [prompt],
        SamplingParams(max_tokens=100),
        guided_options_request={"guided_json": schema.model_json_schema()},
    )

    # Benchmark
    start = time.perf_counter()

    for _ in range(num_requests):
        llm.generate(
            [prompt],
            SamplingParams(max_tokens=200),
            guided_options_request={"guided_json": schema.model_json_schema()},
        )

    elapsed = time.perf_counter() - start

    return {
        "total_time": elapsed,
        "avg_latency_ms": (elapsed / num_requests) * 1000,
        "requests_per_second": num_requests / elapsed,
    }


if __name__ == "__main__":
    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

    # Benchmark simple schema
    simple_results = benchmark_structured_output(llm, SimpleResult, 50)
    print(f"Simple schema: {simple_results['avg_latency_ms']:.1f}ms avg")

    # Benchmark complex schema
    complex_results = benchmark_structured_output(llm, ComplexResult, 50)
    print(f"Complex schema: {complex_results['avg_latency_ms']:.1f}ms avg")
```

---

## Key Takeaways

1. **Guaranteed valid output** - Guided decoding ensures 100% schema compliance

2. **Multiple constraint types** - JSON schema, regex, choice, grammar

3. **Backend selection matters** - xgrammar for speed, outlines for features

4. **SGLang for multi-step** - RadixAttention enables efficient multi-turn programs

5. **Minimal overhead** - 1-5% latency increase for most use cases

6. **Cache schemas** - Compilation is one-time cost per schema

---

## Lab Preview: SGLang Structured Output

In Lab 5, you will:

- Set up SGLang server
- Implement JSON schema-constrained generation
- Build a multi-step extraction pipeline
- Create a function calling system
- Benchmark structured vs unstructured output

---

## References

1. [Outlines Documentation](https://outlines-dev.github.io/outlines/)
2. [SGLang Documentation](https://sgl-project.github.io/)
3. [vLLM Guided Decoding](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#guided-decoding)
4. [XGrammar Paper](https://arxiv.org/abs/2411.15100)
