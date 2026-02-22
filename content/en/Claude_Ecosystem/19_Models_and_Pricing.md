# Models, Pricing, and Optimization

**Previous**: [18. Building Custom Agents](./18_Building_Custom_Agents.md) | **Next**: [20. Advanced Development Workflows](./20_Advanced_Workflows.md)

---

Understanding Claude's model tiers, pricing structure, and cost optimization strategies is essential for building cost-effective AI applications. This lesson provides a comprehensive guide to choosing the right model for each task, leveraging caching and batching for significant savings, and estimating costs for real-world workflows.

**Difficulty**: ⭐⭐

**Prerequisites**:
- Basic understanding of the Claude API ([Lesson 15](./15_Claude_API_Fundamentals.md))
- Familiarity with token-based pricing concepts
- Experience making API calls to Claude

**Learning Objectives**:
- Compare Claude model tiers by intelligence, speed, cost, and capabilities
- Calculate costs for API usage across different models
- Implement prompt caching to reduce costs by up to 90%
- Use the Batch API for 50% savings on non-time-sensitive work
- Design model selection strategies for multi-tier architectures
- Estimate costs for common development and production workflows
- Choose appropriate subscription plans for individual and team use

---

## Table of Contents

1. [Claude Model Family Overview](#1-claude-model-family-overview)
2. [Model Capabilities Comparison](#2-model-capabilities-comparison)
3. [Pricing Structure](#3-pricing-structure)
4. [Prompt Caching](#4-prompt-caching)
5. [Batch API](#5-batch-api)
6. [Model Selection Strategies](#6-model-selection-strategies)
7. [Token Efficiency Techniques](#7-token-efficiency-techniques)
8. [Subscription Plans](#8-subscription-plans)
9. [Cost Estimation for Common Workflows](#9-cost-estimation-for-common-workflows)
10. [Exercises](#10-exercises)

---

## 1. Claude Model Family Overview

Claude is available in three model tiers, each optimized for different use cases along the intelligence-speed-cost spectrum.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Claude Model Family                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────┐                 │
│  │  Claude Opus 4                                 │                 │
│  │  "The Thinker"                                │                 │
│  │                                               │                 │
│  │  • Most capable model in the Claude family    │                 │
│  │  • Best for complex reasoning, math, coding   │                 │
│  │  • Excels at multi-step analysis              │                 │
│  │  • Ideal for: research, architecture design,  │                 │
│  │    difficult debugging, nuanced writing       │                 │
│  └───────────────────────────────────────────────┘                 │
│                                                                     │
│  ┌───────────────────────────────────────────────┐                 │
│  │  Claude Sonnet 4                               │                 │
│  │  "The Workhorse"                              │                 │
│  │                                               │                 │
│  │  • Balanced intelligence and speed            │                 │
│  │  • Great for most everyday coding tasks       │                 │
│  │  • Strong at following instructions           │                 │
│  │  • Ideal for: code generation, refactoring,   │                 │
│  │    translation, summarization, analysis       │                 │
│  └───────────────────────────────────────────────┘                 │
│                                                                     │
│  ┌───────────────────────────────────────────────┐                 │
│  │  Claude Haiku                                  │                 │
│  │  "The Speedster"                              │                 │
│  │                                               │                 │
│  │  • Fastest and most cost-efficient            │                 │
│  │  • Good for simple, high-volume tasks         │                 │
│  │  • Near-instant responses                     │                 │
│  │  • Ideal for: classification, extraction,     │                 │
│  │    simple Q&A, data processing                │                 │
│  └───────────────────────────────────────────────┘                 │
│                                                                     │
│  Intelligence ─────▶  Haiku ─── Sonnet ─────── Opus               │
│  Speed ────────────▶  Opus ──── Sonnet ─────── Haiku              │
│  Cost ─────────────▶  Haiku ─── Sonnet ─────── Opus               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.1 When to Use Each Model

**Claude Opus 4** -- use when the task demands the highest reasoning capability:
- Complex multi-step mathematical proofs
- Architectural decisions involving many trade-offs
- Debugging subtle concurrency or memory issues
- Long-form technical writing requiring deep domain expertise
- Tasks where getting it right matters more than speed

**Claude Sonnet 4** -- the default choice for most development work:
- Code generation and refactoring
- Code review and bug identification
- Document summarization and translation
- API integration and boilerplate generation
- Interactive coding sessions in Claude Code

**Claude Haiku** -- use for high-volume, latency-sensitive tasks:
- Classifying support tickets or user intent
- Extracting structured data from text
- Simple question answering over known content
- Data validation and formatting
- Real-time user-facing features where speed is critical

---

## 2. Model Capabilities Comparison

```
┌──────────────────────┬──────────────┬──────────────┬──────────────┐
│ Capability           │ Claude Opus 4│Claude Sonnet4│ Claude Haiku │
├──────────────────────┼──────────────┼──────────────┼──────────────┤
│ Intelligence Level   │ Highest      │ High         │ Good         │
│ Reasoning Depth      │ ★★★★★       │ ★★★★☆       │ ★★★☆☆       │
│ Coding Ability       │ ★★★★★       │ ★★★★☆       │ ★★★☆☆       │
│ Speed (tokens/sec)   │ Moderate     │ Fast         │ Fastest      │
│ Context Window       │ 200K tokens  │ 200K tokens  │ 200K tokens  │
│ Max Output Tokens    │ 32,000       │ 16,000       │ 8,192        │
│ Vision (images)      │ Yes          │ Yes          │ Yes          │
│ Extended Thinking    │ Yes          │ Yes          │ No           │
│ Tool Use             │ Yes          │ Yes          │ Yes          │
│ Streaming            │ Yes          │ Yes          │ Yes          │
│ Batch API            │ Yes          │ Yes          │ Yes          │
│ Prompt Caching       │ Yes          │ Yes          │ Yes          │
├──────────────────────┼──────────────┼──────────────┼──────────────┤
│ Best For             │ Complex      │ Everyday     │ High-volume  │
│                      │ reasoning    │ coding       │ simple tasks │
└──────────────────────┴──────────────┴──────────────┴──────────────┘
```

### 2.1 Context Window Deep Dive

All Claude models share a 200K token context window (approximately 150,000 words or 500 pages of text). Understanding how to use this effectively is critical for cost management.

```python
import anthropic

client = anthropic.Anthropic()

# Check model capabilities programmatically
# The context window applies to the sum of input + output tokens
# For a 200K context window:
#   - Input tokens + Output tokens <= 200,000 (approximate)
#   - Practical input limit depends on desired output length

# Example: estimating token count before sending
def estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 characters per token for English text."""
    return len(text) // 4

# A 200K context window means you can fit approximately:
examples = {
    "A short prompt": 50,
    "A typical code file (500 lines)": 2_000,
    "A full project README": 1_500,
    "10 source files for context": 20_000,
    "An entire small codebase": 100_000,
    "Maximum practical input": 180_000,  # Leave room for output
}

print("Token budget for 200K context window:")
print(f"{'Content':<40} {'Tokens':>10} {'% of Budget':>12}")
print("-" * 65)
for desc, tokens in examples.items():
    pct = tokens / 200_000 * 100
    print(f"{desc:<40} {tokens:>10,} {pct:>11.1f}%")
```

### 2.2 Extended Thinking

Extended thinking allows Claude to reason step-by-step before responding, significantly improving performance on complex tasks. It is available on Opus and Sonnet.

```python
import anthropic

client = anthropic.Anthropic()

# Using extended thinking for a complex reasoning task
response = client.messages.create(
    model="claude-opus-4-20250514",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # Allow up to 10K tokens for thinking
    },
    messages=[{
        "role": "user",
        "content": (
            "Analyze this distributed system design for potential "
            "consistency issues when handling concurrent writes "
            "across three data centers with eventual consistency."
        )
    }]
)

# The response includes both thinking and text blocks
for block in response.content:
    if block.type == "thinking":
        print(f"[Thinking] ({len(block.thinking)} chars)")
        print(block.thinking[:200] + "...")
    elif block.type == "text":
        print(f"\n[Response]")
        print(block.text)

# Note: thinking tokens are billed at a reduced rate
# but still count toward context window usage
```

---

## 3. Pricing Structure

Claude API pricing is based on tokens -- the units of text that models process. Input tokens (what you send) and output tokens (what Claude generates) are priced differently.

### 3.1 Per-Token Pricing

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Claude API Pricing (per million tokens)           │
├──────────────────┬──────────────┬──────────────┬───────────────────┤
│                  │ Claude Opus 4│Claude Sonnet4│ Claude Haiku      │
├──────────────────┼──────────────┼──────────────┼───────────────────┤
│ Input tokens     │   $15.00     │    $3.00     │     $0.80         │
│ Output tokens    │   $75.00     │   $15.00     │     $4.00         │
├──────────────────┼──────────────┼──────────────┼───────────────────┤
│ Prompt Caching:  │              │              │                   │
│  Cache write     │   $18.75     │    $3.75     │     $1.00         │
│  Cache read      │    $1.50     │    $0.30     │     $0.08         │
├──────────────────┼──────────────┼──────────────┼───────────────────┤
│ Batch API:       │              │              │                   │
│  Input tokens    │    $7.50     │    $1.50     │     $0.40         │
│  Output tokens   │   $37.50     │    $7.50     │     $2.00         │
└──────────────────┴──────────────┴──────────────┴───────────────────┘

Notes:
- 1 million tokens (MTok) ≈ 750,000 words ≈ 2,500 pages of text
- Output tokens cost 5x input tokens (models generate more carefully)
- Prompt caching: first write costs 1.25x, subsequent reads cost 0.1x
- Batch API: 50% discount on all token costs
```

### 3.2 Understanding Token Costs

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelPricing:
    """Pricing per million tokens for a Claude model."""
    name: str
    input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: float
    cache_read_per_mtok: float
    batch_input_per_mtok: float
    batch_output_per_mtok: float

# Define pricing for each model
PRICING = {
    "opus": ModelPricing(
        name="Claude Opus 4",
        input_per_mtok=15.00,
        output_per_mtok=75.00,
        cache_write_per_mtok=18.75,
        cache_read_per_mtok=1.50,
        batch_input_per_mtok=7.50,
        batch_output_per_mtok=37.50,
    ),
    "sonnet": ModelPricing(
        name="Claude Sonnet 4",
        input_per_mtok=3.00,
        output_per_mtok=15.00,
        cache_write_per_mtok=3.75,
        cache_read_per_mtok=0.30,
        batch_input_per_mtok=1.50,
        batch_output_per_mtok=7.50,
    ),
    "haiku": ModelPricing(
        name="Claude Haiku",
        input_per_mtok=0.80,
        output_per_mtok=4.00,
        cache_write_per_mtok=1.00,
        cache_read_per_mtok=0.08,
        batch_input_per_mtok=0.40,
        batch_output_per_mtok=2.00,
    ),
}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
    use_batch: bool = False,
) -> dict:
    """Calculate the cost for a Claude API call.

    Args:
        model: One of "opus", "sonnet", "haiku"
        input_tokens: Number of non-cached input tokens
        output_tokens: Number of output tokens
        cached_input_tokens: Number of tokens read from cache
        cache_write_tokens: Number of tokens written to cache
        use_batch: Whether to use Batch API pricing

    Returns:
        Dictionary with cost breakdown
    """
    pricing = PRICING[model]

    if use_batch:
        input_cost = (input_tokens / 1_000_000) * pricing.batch_input_per_mtok
        output_cost = (output_tokens / 1_000_000) * pricing.batch_output_per_mtok
        # Caching is not typically combined with batch, but for completeness:
        cache_read_cost = 0
        cache_write_cost = 0
    else:
        input_cost = (input_tokens / 1_000_000) * pricing.input_per_mtok
        output_cost = (output_tokens / 1_000_000) * pricing.output_per_mtok
        cache_read_cost = (cached_input_tokens / 1_000_000) * pricing.cache_read_per_mtok
        cache_write_cost = (cache_write_tokens / 1_000_000) * pricing.cache_write_per_mtok

    total = input_cost + output_cost + cache_read_cost + cache_write_cost

    return {
        "model": pricing.name,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cache_read_cost": cache_read_cost,
        "cache_write_cost": cache_write_cost,
        "total_cost": total,
        "batch_discount": use_batch,
    }


# Example: Compare costs across models for a typical coding task
# ~2000 input tokens (prompt + code context), ~1000 output tokens
print("Cost comparison: typical coding task (2K input, 1K output)")
print("=" * 60)
for model in ["opus", "sonnet", "haiku"]:
    result = calculate_cost(model, input_tokens=2000, output_tokens=1000)
    print(f"\n{result['model']}:")
    print(f"  Input:  ${result['input_cost']:.4f}")
    print(f"  Output: ${result['output_cost']:.4f}")
    print(f"  Total:  ${result['total_cost']:.4f}")

# Example: Large context with caching
print("\n\nLarge context with caching (50K cached, 2K new input, 1K output)")
print("=" * 60)
for model in ["opus", "sonnet", "haiku"]:
    # Without caching: all 52K tokens are input
    no_cache = calculate_cost(model, input_tokens=52000, output_tokens=1000)
    # With caching: 50K cached (read), 2K fresh input
    with_cache = calculate_cost(
        model,
        input_tokens=2000,
        output_tokens=1000,
        cached_input_tokens=50000,
    )
    savings = (1 - with_cache["total_cost"] / no_cache["total_cost"]) * 100

    print(f"\n{PRICING[model].name}:")
    print(f"  Without cache: ${no_cache['total_cost']:.4f}")
    print(f"  With cache:    ${with_cache['total_cost']:.4f}")
    print(f"  Savings:       {savings:.1f}%")
```

### 3.3 Cost Ratio Across Models

To put the pricing in perspective:

```
Relative cost for the same task:

  Opus    ████████████████████████████████████████  $1.00 (baseline)
  Sonnet  ████████                                  $0.20 (5x cheaper)
  Haiku   ██                                        $0.05 (20x cheaper)

For the cost of ONE Opus call, you can make:
  - 5 Sonnet calls, or
  - 20 Haiku calls

This makes model selection one of the most impactful cost levers.
```

---

## 4. Prompt Caching

Prompt caching allows you to cache frequently used context (system prompts, large documents, code files) and reuse it across multiple API calls. Cached tokens are read at 90% discount compared to regular input tokens.

### 4.1 How Prompt Caching Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Prompt Caching Flow                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  First Request (cache miss → cache write):                          │
│  ┌──────────────────────────────────────────────────┐              │
│  │  System Prompt (2K tokens)     ← cache_control   │              │
│  │  Large Document (50K tokens)   ← cache_control   │              │
│  │  User Message (500 tokens)     ← NOT cached      │              │
│  └──────────────────────────────────────────────────┘              │
│  Cost: 52K × write_price + 500 × input_price + output_price       │
│                                                                     │
│  Subsequent Requests (cache hit → cache read):                      │
│  ┌──────────────────────────────────────────────────┐              │
│  │  System Prompt (2K tokens)     ← CACHE HIT ✓    │              │
│  │  Large Document (50K tokens)   ← CACHE HIT ✓    │              │
│  │  User Message (800 tokens)     ← NOT cached      │              │
│  └──────────────────────────────────────────────────┘              │
│  Cost: 52K × read_price + 800 × input_price + output_price        │
│                                                                     │
│  Cache TTL: 5 minutes (refreshed on each cache hit)                │
│  Min cacheable: 1,024 tokens (shorter content not worth caching)   │
│  Cache key: exact prefix match (any change invalidates)            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Python Implementation

```python
import anthropic

client = anthropic.Anthropic()

# A large system prompt with project context (worth caching)
SYSTEM_PROMPT = """You are an expert Python developer working on our e-commerce
platform. The codebase uses FastAPI, SQLAlchemy, and PostgreSQL.

Here are the key architectural decisions:
... (imagine 2000+ tokens of context here) ...

Follow these coding standards:
- PEP 8 style
- Type hints on all functions
- Docstrings for public methods
- 90% test coverage minimum
"""

# A large reference document (worth caching)
API_SPEC = """
OpenAPI 3.0 Specification for our REST API:
... (imagine 10,000+ tokens of API spec here) ...
"""

def query_with_caching(user_message: str) -> str:
    """Send a query using prompt caching for the system prompt and API spec."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}  # Cache this block
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": API_SPEC,
                        "cache_control": {"type": "ephemeral"}  # Cache this block
                    },
                    {
                        "type": "text",
                        "text": user_message  # This varies per request
                    }
                ]
            }
        ],
    )

    # Check cache performance from usage stats
    usage = response.usage
    print(f"Input tokens:        {usage.input_tokens}")
    print(f"Cache creation:      {getattr(usage, 'cache_creation_input_tokens', 0)}")
    print(f"Cache read:          {getattr(usage, 'cache_read_input_tokens', 0)}")
    print(f"Output tokens:       {usage.output_tokens}")

    return response.content[0].text


# First call: cache miss (writes to cache)
print("=== First call (cache write) ===")
result1 = query_with_caching("Add a new endpoint GET /api/v1/products/{id}/reviews")
# Output: Cache creation: ~12000, Cache read: 0

# Second call within 5 minutes: cache hit (reads from cache)
print("\n=== Second call (cache hit) ===")
result2 = query_with_caching("Now add pagination to the reviews endpoint")
# Output: Cache creation: 0, Cache read: ~12000 (90% cheaper!)

# Third call: still hitting cache (TTL refreshed by second call)
print("\n=== Third call (cache hit) ===")
result3 = query_with_caching("Add rate limiting to the reviews endpoint")
# Output: Cache creation: 0, Cache read: ~12000
```

### 4.3 TypeScript Implementation

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const SYSTEM_CONTEXT = `You are a senior TypeScript developer...
${/* Imagine 2000+ tokens of context */ ""}`;

const CODE_BASE_CONTEXT = `// Current codebase structure:
${/* Imagine 10,000+ tokens of code */ ""}`;

async function queryWithCache(userMessage: string): Promise<string> {
  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 4096,
    system: [
      {
        type: "text",
        text: SYSTEM_CONTEXT,
        cache_control: { type: "ephemeral" },
      },
    ],
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: CODE_BASE_CONTEXT,
            cache_control: { type: "ephemeral" },
          },
          {
            type: "text",
            text: userMessage,
          },
        ],
      },
    ],
  });

  // Log cache performance
  const usage = response.usage as any;
  console.log(`Cache write: ${usage.cache_creation_input_tokens ?? 0}`);
  console.log(`Cache read:  ${usage.cache_read_input_tokens ?? 0}`);

  const textBlock = response.content.find((b) => b.type === "text");
  return textBlock?.text ?? "";
}
```

### 4.4 Caching Best Practices

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Prompt Caching Best Practices                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  DO:                                                                │
│  ✓ Cache large, stable context (system prompts, reference docs)    │
│  ✓ Place cached content BEFORE varying content (prefix matching)   │
│  ✓ Cache content ≥ 1,024 tokens (minimum for caching benefit)     │
│  ✓ Reuse cache within 5-minute TTL window                          │
│  ✓ Structure prompts: [cached prefix] + [variable suffix]          │
│  ✓ Monitor cache hit rates in production                           │
│                                                                     │
│  DON'T:                                                             │
│  ✗ Cache content that changes every request                        │
│  ✗ Put variable content before cached content (breaks prefix)      │
│  ✗ Cache tiny prompts (< 1,024 tokens) — overhead not worth it    │
│  ✗ Assume cache persists forever (5-minute TTL)                    │
│  ✗ Use different models with the same cache (separate caches)      │
│                                                                     │
│  Cost math:                                                         │
│  - Write: 1.25x normal input price (first call only)              │
│  - Read:  0.10x normal input price (subsequent calls)              │
│  - Break-even: 2 reads pay for the write overhead                  │
│  - 10 reads: ~87% savings on cached portion                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Batch API

The Batch API allows you to send large numbers of requests at a 50% discount. Batches are processed asynchronously within a 24-hour window, making them ideal for non-time-sensitive workloads.

### 5.1 How Batches Work

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Batch API Workflow                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Create batch with multiple requests                             │
│     ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │
│     │Request 1│ │Request 2│ │Request 3│ │Request N│              │
│     └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘              │
│          └───────────┴───────────┴───────────┘                     │
│                          │                                          │
│  2. Submit batch ────────▼─────────────────────────                │
│     POST /v1/messages/batches                                      │
│     Status: "in_progress"                                          │
│                          │                                          │
│  3. Processing ──────────▼─────────────────────────                │
│     (up to 24 hours, typically much faster)                        │
│                          │                                          │
│  4. Results ready ───────▼─────────────────────────                │
│     Status: "ended"                                                │
│     ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │
│     │Result 1 │ │Result 2 │ │Result 3 │ │Result N │              │
│     └─────────┘ └─────────┘ └─────────┘ └─────────┘              │
│                                                                     │
│  Pricing: 50% of standard API rates                                │
│  Max requests per batch: 10,000                                    │
│  Processing window: up to 24 hours                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Creating a Batch

```python
import anthropic
import json
import time

client = anthropic.Anthropic()

# Prepare batch requests — each is a standard Messages API call
# wrapped with a custom_id for tracking
batch_requests = []

# Example: translate 100 product descriptions
product_descriptions = [
    {"id": f"product_{i}", "text": f"Sample product description #{i}"}
    for i in range(100)
]

for product in product_descriptions:
    batch_requests.append({
        "custom_id": product["id"],
        "params": {
            "model": "claude-haiku-3-5-20241022",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Translate this product description to Korean. "
                        f"Return only the translation:\n\n{product['text']}"
                    )
                }
            ]
        }
    })

# Create the batch
batch = client.messages.batches.create(requests=batch_requests)

print(f"Batch created: {batch.id}")
print(f"Status: {batch.processing_status}")
print(f"Request counts: {batch.request_counts}")
```

### 5.3 Polling for Results

```python
import anthropic
import time

client = anthropic.Anthropic()

def wait_for_batch(batch_id: str, poll_interval: int = 30) -> None:
    """Poll batch status until completion."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status

        succeeded = batch.request_counts.succeeded
        errored = batch.request_counts.errored
        processing = batch.request_counts.processing
        total = succeeded + errored + processing

        print(
            f"Status: {status} | "
            f"Succeeded: {succeeded}/{total} | "
            f"Errored: {errored}"
        )

        if status == "ended":
            print("Batch processing complete!")
            break

        time.sleep(poll_interval)

    return batch


def retrieve_results(batch_id: str) -> list:
    """Retrieve all results from a completed batch."""
    results = []
    for result in client.messages.batches.results(batch_id):
        results.append({
            "custom_id": result.custom_id,
            "type": result.result.type,
            "message": (
                result.result.message.content[0].text
                if result.result.type == "succeeded"
                else None
            ),
            "error": (
                result.result.error
                if result.result.type == "errored"
                else None
            ),
        })
    return results


# Usage
batch_id = "batch_abc123"  # From the creation step
batch = wait_for_batch(batch_id)
results = retrieve_results(batch_id)

# Process results
succeeded = [r for r in results if r["type"] == "succeeded"]
failed = [r for r in results if r["type"] == "errored"]

print(f"\nResults: {len(succeeded)} succeeded, {len(failed)} failed")
for result in succeeded[:3]:
    print(f"\n[{result['custom_id']}]: {result['message'][:100]}...")
```

### 5.4 Batch API Use Cases

```
Good candidates for Batch API:               Poor candidates:
─────────────────────────────                 ─────────────────
✓ Bulk content translation                    ✗ Real-time chat
✓ Dataset labeling / classification           ✗ Interactive coding
✓ Generating test data                        ✗ User-facing features
✓ Overnight code analysis                     ✗ CI/CD blocking steps
✓ Document summarization pipelines            ✗ Time-sensitive alerts
✓ Monthly report generation                   ✗ Streaming responses
✓ Content moderation backlogs
✓ Embedding generation
```

---

## 6. Model Selection Strategies

### 6.1 Tiered Architecture

The most cost-effective approach uses multiple models in a tiered architecture, routing tasks to the appropriate model based on complexity.

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import anthropic

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

@dataclass
class ModelConfig:
    model_id: str
    max_tokens: int
    temperature: float = 0.0

# Model configurations
MODEL_TIERS = {
    TaskComplexity.SIMPLE: ModelConfig(
        model_id="claude-haiku-3-5-20241022",
        max_tokens=1024,
    ),
    TaskComplexity.MODERATE: ModelConfig(
        model_id="claude-sonnet-4-20250514",
        max_tokens=4096,
    ),
    TaskComplexity.COMPLEX: ModelConfig(
        model_id="claude-opus-4-20250514",
        max_tokens=8192,
    ),
}


def classify_task_complexity(task_description: str) -> TaskComplexity:
    """Classify a task to determine which model tier to use.

    This is a simplified heuristic. In production, you might use
    a small model (Haiku) to classify tasks before routing.
    """
    complex_indicators = [
        "architect", "design system", "debug concurrency",
        "mathematical proof", "security audit", "trade-offs",
        "multi-step reasoning", "analyze entire codebase",
    ]
    moderate_indicators = [
        "refactor", "implement", "write tests", "code review",
        "generate", "convert", "translate", "summarize",
    ]

    task_lower = task_description.lower()

    if any(indicator in task_lower for indicator in complex_indicators):
        return TaskComplexity.COMPLEX
    elif any(indicator in task_lower for indicator in moderate_indicators):
        return TaskComplexity.MODERATE
    else:
        return TaskComplexity.SIMPLE


def route_to_model(task: str, user_message: str) -> str:
    """Route a task to the appropriate model tier."""
    client = anthropic.Anthropic()
    complexity = classify_task_complexity(task)
    config = MODEL_TIERS[complexity]

    print(f"Task: {task}")
    print(f"Complexity: {complexity.value} → Model: {config.model_id}")

    response = client.messages.create(
        model=config.model_id,
        max_tokens=config.max_tokens,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text


# Examples of routing decisions
tasks = [
    ("Extract the email from this text", "simple"),
    ("Refactor this function to use async/await", "moderate"),
    ("Design a distributed cache with consistency guarantees", "complex"),
]

for task, expected in tasks:
    complexity = classify_task_complexity(task)
    model = MODEL_TIERS[complexity]
    print(f"  '{task}'")
    print(f"    → {complexity.value} ({expected}) → {model.model_id}")
    print()
```

### 6.2 Cascade Pattern

Start with a cheaper model and escalate to a more capable one only if the result is unsatisfactory.

```python
import anthropic
import json

client = anthropic.Anthropic()

def cascade_query(
    user_message: str,
    system_prompt: str = "",
    validation_fn=None,
) -> dict:
    """Try cheaper models first, escalate if quality is insufficient.

    Args:
        user_message: The user's request
        system_prompt: Optional system prompt
        validation_fn: Optional function to validate the response.
                       Returns True if the response is acceptable.
    """
    models = [
        ("claude-haiku-3-5-20241022", 1024),    # Try cheapest first
        ("claude-sonnet-4-20250514", 4096),      # Escalate to mid-tier
        ("claude-opus-4-20250514", 8192),        # Final escalation
    ]

    for model_id, max_tokens in models:
        print(f"Trying {model_id}...")

        kwargs = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user_message}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = client.messages.create(**kwargs)
        result = response.content[0].text

        # If no validation function, accept the first result
        if validation_fn is None:
            return {"model": model_id, "response": result, "escalations": 0}

        # Validate the response
        if validation_fn(result):
            return {"model": model_id, "response": result}
        else:
            print(f"  → Response from {model_id} failed validation, escalating...")

    # If all models tried, return the last result
    return {"model": models[-1][0], "response": result, "note": "max escalation"}


# Example: validate that a code generation response compiles
def validate_json_output(response: str) -> bool:
    """Check if the response contains valid JSON."""
    try:
        # Try to extract JSON from the response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            json.loads(response[start:end])
            return True
    except json.JSONDecodeError:
        pass
    return False


result = cascade_query(
    user_message="Generate a JSON schema for a User object with name, email, age, and roles fields.",
    validation_fn=validate_json_output,
)
print(f"\nFinal model used: {result['model']}")
print(f"Response: {result['response'][:200]}...")
```

---

## 7. Token Efficiency Techniques

Reducing token usage directly reduces cost. Here are practical techniques for writing more efficient prompts.

### 7.1 Concise Prompts

```python
# BAD: Verbose prompt (wastes tokens on unnecessary words)
verbose_prompt = """
I would really appreciate it if you could help me with something.
I have a Python function that I need you to look at. What I'm hoping
you can do is review the code and tell me if there are any bugs or
issues with it. The function is supposed to calculate the factorial
of a number. Could you please take a look at it and let me know
what you think? Here is the code:

def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

Thank you so much for your help with this!
"""
# ~100 tokens

# GOOD: Concise prompt (same information, fewer tokens)
concise_prompt = """
Review this factorial function for bugs:

```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
```

List any issues found.
"""
# ~40 tokens (60% reduction)
```

### 7.2 Structured Output Requests

```python
# Requesting structured output reduces unnecessary prose in responses
import anthropic

client = anthropic.Anthropic()

# BAD: Unstructured request → long, chatty response
bad_prompt = "What do you think about this code? Is it good or bad?"

# GOOD: Structured request → concise, actionable response
good_prompt = """Review this code. Respond in this exact JSON format:
{
  "issues": [{"line": N, "severity": "high|medium|low", "description": "..."}],
  "suggestions": ["..."],
  "overall_rating": "good|acceptable|needs_work"
}"""

# GOOD: Using prefilled assistant response to constrain output
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    messages=[
        {"role": "user", "content": good_prompt + "\n\n```python\ndef add(a, b): return a + b\n```"},
        {"role": "assistant", "content": "{"}  # Force JSON output
    ],
)
```

### 7.3 Context Pruning

```python
def prune_context_for_task(files: dict, task: str) -> dict:
    """Include only relevant files in the context.

    Instead of sending your entire codebase, send only the files
    that are relevant to the current task.
    """
    # Simple keyword-based relevance scoring
    task_keywords = set(task.lower().split())
    scored_files = []

    for filepath, content in files.items():
        # Score based on keyword overlap
        file_words = set(filepath.lower().replace("/", " ").replace("_", " ").split())
        content_sample = set(content[:500].lower().split())
        overlap = len(task_keywords & (file_words | content_sample))
        scored_files.append((overlap, filepath, content))

    # Sort by relevance and take top files
    scored_files.sort(reverse=True)
    relevant = {fp: content for score, fp, content in scored_files[:5] if score > 0}

    total_tokens_saved = sum(
        len(content) // 4
        for score, fp, content in scored_files[5:]
        if score == 0
    )
    print(f"Included {len(relevant)}/{len(files)} files")
    print(f"Estimated tokens saved: ~{total_tokens_saved:,}")

    return relevant
```

---

## 8. Subscription Plans

For interactive use through claude.ai and the Claude apps (not the API), Anthropic offers subscription plans.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Claude Subscription Plans                         │
├─────────────────┬──────────────────────────────────────────────────┤
│                 │                                                    │
│  Claude Free    │  $0/month                                         │
│                 │  • Access to Claude Sonnet (limited)              │
│                 │  • Basic conversation features                    │
│                 │  • Limited daily message allowance                │
│                 │  • No priority access during high traffic         │
│                 │                                                    │
├─────────────────┼──────────────────────────────────────────────────┤
│                 │                                                    │
│  Claude Pro     │  $20/month                                        │
│                 │  • Access to Opus, Sonnet, and Haiku              │
│                 │  • 5x more usage than Free tier                   │
│                 │  • Priority access during high traffic            │
│                 │  • Access to Claude Projects                      │
│                 │  • Extended thinking mode                         │
│                 │  • Early access to new features                   │
│                 │                                                    │
├─────────────────┼──────────────────────────────────────────────────┤
│                 │                                                    │
│  Claude Team    │  $25/user/month (minimum 5 users)                │
│                 │  • Everything in Pro, plus:                       │
│                 │  • Higher usage limits than Pro                   │
│                 │  • Team workspace with shared Projects            │
│                 │  • Admin dashboard for usage monitoring           │
│                 │  • 500K context window for all models             │
│                 │  • Data not used for training                     │
│                 │                                                    │
├─────────────────┼──────────────────────────────────────────────────┤
│                 │                                                    │
│  Claude         │  Custom pricing (contact sales)                   │
│  Enterprise     │  • Everything in Team, plus:                      │
│                 │  • SSO (SAML) and SCIM provisioning              │
│                 │  • Domain capture and admin controls              │
│                 │  • Audit logs and compliance features             │
│                 │  • Custom data retention policies                 │
│                 │  • Dedicated support and SLA                      │
│                 │  • Higher rate limits                             │
│                 │  • Data not used for training                     │
│                 │                                                    │
└─────────────────┴──────────────────────────────────────────────────┘

Note: API usage is billed separately based on per-token pricing.
Subscription plans cover interactive use of claude.ai and Claude apps.
Claude Code (CLI) uses API credits, not subscription quotas.
```

---

## 9. Cost Estimation for Common Workflows

### 9.1 Daily Developer Workflow

```python
def estimate_daily_developer_cost():
    """Estimate daily API costs for a typical developer using Claude Code."""

    # Typical daily usage pattern
    daily_activities = [
        {
            "activity": "Morning code review (3 files)",
            "model": "sonnet",
            "input_tokens": 15000,   # 3 files × ~5K tokens each
            "output_tokens": 3000,    # Review comments
            "calls": 3,
        },
        {
            "activity": "Feature implementation (5 iterations)",
            "model": "sonnet",
            "input_tokens": 8000,    # Context + prompt per iteration
            "output_tokens": 4000,   # Generated code
            "calls": 5,
        },
        {
            "activity": "Bug debugging session",
            "model": "opus",          # Complex debugging → Opus
            "input_tokens": 20000,   # Large context for debugging
            "output_tokens": 5000,   # Detailed analysis
            "calls": 2,
        },
        {
            "activity": "Test generation",
            "model": "sonnet",
            "input_tokens": 5000,
            "output_tokens": 8000,   # Tests are usually longer
            "calls": 3,
        },
        {
            "activity": "Quick questions / lookups",
            "model": "haiku",         # Simple questions → Haiku
            "input_tokens": 1000,
            "output_tokens": 500,
            "calls": 10,
        },
    ]

    total_cost = 0
    print("Daily Developer Cost Estimate")
    print("=" * 70)
    print(f"{'Activity':<40} {'Model':<10} {'Calls':>5} {'Cost':>8}")
    print("-" * 70)

    for activity in daily_activities:
        pricing = PRICING[activity["model"]]
        per_call_cost = (
            (activity["input_tokens"] / 1_000_000) * pricing.input_per_mtok +
            (activity["output_tokens"] / 1_000_000) * pricing.output_per_mtok
        )
        activity_cost = per_call_cost * activity["calls"]
        total_cost += activity_cost

        print(
            f"{activity['activity']:<40} "
            f"{activity['model']:<10} "
            f"{activity['calls']:>5} "
            f"${activity_cost:>7.2f}"
        )

    print("-" * 70)
    print(f"{'Daily total':<40} {'':>10} {'':>5} ${total_cost:>7.2f}")
    print(f"{'Monthly (22 working days)':<40} {'':>10} {'':>5} ${total_cost * 22:>7.2f}")
    print(f"{'Yearly estimate':<40} {'':>10} {'':>5} ${total_cost * 260:>7.2f}")

    # Compare with and without optimization
    print(f"\nWith prompt caching (est. 40% savings): ${total_cost * 22 * 0.6:.2f}/month")
    print(f"With model tiering (est. 30% savings):  ${total_cost * 22 * 0.7:.2f}/month")
    print(f"With both optimizations:                ${total_cost * 22 * 0.42:.2f}/month")

estimate_daily_developer_cost()
```

### 9.2 Production Pipeline Costs

```python
def estimate_production_pipeline_cost():
    """Estimate monthly costs for a production AI pipeline."""

    monthly_volumes = {
        "Customer support classification": {
            "model": "haiku",
            "monthly_requests": 100_000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 50,
        },
        "Content moderation": {
            "model": "haiku",
            "monthly_requests": 500_000,
            "avg_input_tokens": 200,
            "avg_output_tokens": 30,
        },
        "Document summarization": {
            "model": "sonnet",
            "monthly_requests": 10_000,
            "avg_input_tokens": 5000,
            "avg_output_tokens": 500,
        },
        "Code review automation": {
            "model": "sonnet",
            "monthly_requests": 5_000,
            "avg_input_tokens": 10000,
            "avg_output_tokens": 2000,
        },
        "Architecture analysis": {
            "model": "opus",
            "monthly_requests": 500,
            "avg_input_tokens": 30000,
            "avg_output_tokens": 5000,
        },
    }

    total_monthly = 0
    print("Production Pipeline Monthly Cost Estimate")
    print("=" * 80)
    print(f"{'Pipeline':<35} {'Model':<8} {'Requests':>10} {'Cost':>10}")
    print("-" * 80)

    for pipeline, config in monthly_volumes.items():
        pricing = PRICING[config["model"]]
        total_input = config["monthly_requests"] * config["avg_input_tokens"]
        total_output = config["monthly_requests"] * config["avg_output_tokens"]

        cost = (
            (total_input / 1_000_000) * pricing.input_per_mtok +
            (total_output / 1_000_000) * pricing.output_per_mtok
        )
        total_monthly += cost

        print(
            f"{pipeline:<35} "
            f"{config['model']:<8} "
            f"{config['monthly_requests']:>10,} "
            f"${cost:>9.2f}"
        )

    print("-" * 80)
    print(f"{'Monthly total':<35} {'':>8} {'':>10} ${total_monthly:>9.2f}")
    print(f"\nWith Batch API for non-realtime (est.): ${total_monthly * 0.65:>9.2f}")
    print(f"With caching for repeated contexts:      ${total_monthly * 0.50:>9.2f}")

estimate_production_pipeline_cost()
```

---

## 10. Exercises

### Exercise 1: Cost Calculation (Beginner)

Calculate the cost for each scenario using the pricing table:

1. A single Opus API call with 10,000 input tokens and 2,000 output tokens.
2. 1,000 Haiku API calls, each with 500 input tokens and 100 output tokens.
3. A Sonnet call with 50,000 cached input tokens, 2,000 fresh input tokens, and 1,000 output tokens.
4. A batch of 500 Sonnet requests, each with 3,000 input tokens and 1,000 output tokens.

### Exercise 2: Model Selection (Intermediate)

For each task below, recommend the best model and explain your reasoning:

1. Classifying 50,000 customer emails into 5 categories
2. Debugging a race condition in a distributed system
3. Generating unit tests for a CRUD API
4. Translating a 200-page technical manual
5. Designing a database schema for a new application

### Exercise 3: Caching Strategy (Intermediate)

You have an application that makes 100 API calls per hour, all sharing the same 30,000-token system prompt but with different 500-token user messages. Calculate:

1. Hourly cost without caching (using Sonnet)
2. Hourly cost with prompt caching
3. Monthly savings from caching (assume 8 hours/day, 22 days/month)
4. What happens if calls are spaced more than 5 minutes apart?

### Exercise 4: Build a Cost Dashboard (Advanced)

Write a Python script that:
1. Reads API usage logs (simulate with generated data)
2. Calculates costs by model, by day, and by endpoint
3. Identifies the top 3 cost drivers
4. Recommends optimization opportunities (model downgrades, caching candidates)

### Exercise 5: Tiered Architecture Design (Advanced)

Design a model routing system for a customer support application that handles:
- Simple FAQ responses (70% of queries)
- Product troubleshooting (20% of queries)
- Complex complaint resolution (10% of queries)

Specify which model handles each tier, estimate monthly costs for 50,000 queries/month, and compare with a single-model approach using only Sonnet.

---

## References

- Anthropic Pricing - https://www.anthropic.com/pricing
- Prompt Caching Documentation - https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- Batch API Documentation - https://docs.anthropic.com/en/docs/build-with-claude/message-batches
- Claude Models Overview - https://docs.anthropic.com/en/docs/about-claude/models

---

## Next Steps

[20. Advanced Development Workflows](./20_Advanced_Workflows.md) covers multi-file refactoring, TDD with Claude, CI/CD integration, and strategies for exploring large codebases -- putting these cost-optimized models to work in real development scenarios.
