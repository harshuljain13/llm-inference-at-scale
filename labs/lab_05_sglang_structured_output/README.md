# Lab 5: SGLang Structured Output

## Overview

Use SGLang for structured output generation with JSON schemas, regex constraints, and multi-step LLM programs.

## Learning Objectives

- Configure SGLang server
- Implement JSON schema-constrained generation
- Use regex patterns for output formatting
- Build multi-step LLM programs with shared context

## Prerequisites

- Completed Labs 1-4
- AWS g5.2xlarge instance
- HuggingFace token

## Setup

```bash
pip install sglang[all]
```

## Duration

45-60 minutes

## AWS Cost

~$1.50 (g5.2xlarge for ~60 minutes)

## Exercises

1. **SGLang Server Setup**: Start SGLang runtime
2. **JSON Schema Enforcement**: Generate valid JSON with Pydantic schemas
3. **Regex Constraints**: Use regex patterns for formatting
4. **Multi-Step Programs**: Build programs with RadixAttention benefits

## Example: Entity Extraction

```python
import sglang as sgl

@sgl.function
def extract_entities(s, text):
    s += f"Extract entities from: {text}\n"
    s += "Output JSON:\n"
    s += sgl.gen("result",
                 regex=r'\{"entities": \[.*\]\}')
```

## When SGLang Outperforms vLLM

- Multi-turn conversations with shared context
- Structured output generation (JSON, code)
- Branching/tree-based generation
- Complex LLM programs with multiple calls

## Validation Checkpoints

- [ ] SGLang server starts successfully
- [ ] JSON output validates against schema
- [ ] Regex constraints produce valid formats
- [ ] Multi-step programs share KV cache efficiently

## Next Steps

Proceed to Lab 6: Tensor Parallelism for multi-GPU scaling.
