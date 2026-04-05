# Prompt Caching and Batch API

**Previous**: [23. Vision Agents](./23_Vision_Agents.md) | **Next**: [25. RAG Patterns](./25_RAG_Patterns.md)

---

At scale, API costs and latency become critical concerns. Anthropic provides two powerful mechanisms for reducing both: **prompt caching**, which avoids reprocessing repeated content, and the **Message Batches API**, which offers a 50% discount for asynchronous workloads. This lesson covers both features in depth, including how to combine them for maximum savings.

**Difficulty**: ⭐⭐⭐

**Prerequisites**:
- Claude API fundamentals ([Lesson 15](./15_Claude_API_Fundamentals.md))
- Tool use and function calling ([Lesson 16](./16_Tool_Use_and_Function_Calling.md))
- Models and pricing ([Lesson 19](./19_Models_and_Pricing.md))

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement prompt caching with `cache_control` breakpoints
2. Identify cacheable content types and optimize cache hit rates
3. Calculate cost savings from caching (up to 90% reduction on cached tokens)
4. Manage cache TTL and implement warming strategies
5. Create and monitor batch requests with the Message Batches API
6. Combine caching with batching for maximum cost efficiency
7. Build real-world pipelines for document processing and evaluation

---

## Table of Contents

1. [Prompt Caching Fundamentals](#1-prompt-caching-fundamentals)
2. [Cacheable Content Types](#2-cacheable-content-types)
3. [Cache Hit Rates and Cost Savings](#3-cache-hit-rates-and-cost-savings)
4. [TTL Management and Cache Warming](#4-ttl-management-and-cache-warming)
5. [Message Batches API Overview](#5-message-batches-api-overview)
6. [Creating and Monitoring Batch Requests](#6-creating-and-monitoring-batch-requests)
7. [Batch API Pricing and Limits](#7-batch-api-pricing-and-limits)
8. [Combining Caching with Batching](#8-combining-caching-with-batching)
9. [Real-World Patterns](#9-real-world-patterns)
10. [Exercises](#10-exercises)

---

## 1. Prompt Caching Fundamentals

Prompt caching allows you to mark portions of your request that are likely to remain the same across multiple API calls. When Claude encounters a cache hit, it skips the processing of those tokens, resulting in:

- **90% cost reduction** on cached input tokens
- **Significantly lower latency** (cached tokens are not reprocessed)
- **No change to output quality** — the response is identical whether cached or not

### 1.1 How It Works

```
Request 1: [System Prompt (5000 tokens)] + [User Message (100 tokens)]
            ├── cache_control: ephemeral ──┘
            └── Cache WRITE: 5000 tokens stored, write cost applies

Request 2: [System Prompt (5000 tokens)] + [Different User Message (150 tokens)]
            ├── cache_control: ephemeral ──┘
            └── Cache HIT: 5000 tokens read from cache (90% cheaper)
```

### 1.2 Basic Usage

```python
import anthropic

client = anthropic.Anthropic()

# The system prompt is cached across requests
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are a legal expert assistant. You have deep knowledge of "
                    "contract law, intellectual property, and corporate governance. "
                    "Always cite relevant statutes and case law when applicable. "
                    "Format responses with clear headings and numbered points."
                    + "\n\n" + large_legal_reference_text,  # e.g., 4000+ tokens
            "cache_control": {"type": "ephemeral"},
        }
    ],
    messages=[
        {"role": "user", "content": "What are the key elements of a valid contract?"}
    ],
)

# Check cache performance in the usage stats
print(f"Input tokens: {response.usage.input_tokens}")
print(f"Cache creation tokens: {response.usage.cache_creation_input_tokens}")
print(f"Cache read tokens: {response.usage.cache_read_input_tokens}")
```

### 1.3 Cache Control Breakpoints

The `cache_control` field with `"type": "ephemeral"` marks the end of a cacheable prefix. Key rules:

- You can place up to **4 cache breakpoints** per request
- Content before a breakpoint is eligible for caching
- The minimum cacheable prefix is **1,024 tokens** (for Claude Sonnet) or **2,048 tokens** (for Claude Haiku)
- Breakpoints must be placed at content block boundaries

```python
# Multiple breakpoints example
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": base_system_prompt,  # Breakpoint 1: rarely changes
            "cache_control": {"type": "ephemeral"},
        }
    ],
    tools=tools_with_cache,  # Breakpoint 2: tool definitions
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": long_document,  # Breakpoint 3: document context
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": "Summarize the key findings.",  # Not cached (varies)
                },
            ],
        }
    ],
)
```

---

## 2. Cacheable Content Types

### 2.1 System Prompts

The most common caching target. System prompts rarely change between requests:

```python
SYSTEM_PROMPT = {
    "type": "text",
    "text": (
        "You are a customer support agent for Acme Corp.\n\n"
        "## Product Catalog\n"
        + product_catalog_text    # 3000+ tokens of product data
        + "\n\n## Support Policies\n"
        + support_policies_text   # 2000+ tokens of policies
    ),
    "cache_control": {"type": "ephemeral"},
}

# Every customer query reuses the cached system prompt
for query in customer_queries:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=[SYSTEM_PROMPT],
        messages=[{"role": "user", "content": query}],
    )
```

### 2.2 Tool Definitions

Large tool sets benefit significantly from caching:

```python
# Cache the last tool in the list to cache all tools
tools = [
    {"name": "search_products", "description": "...", "input_schema": {...}},
    {"name": "check_inventory", "description": "...", "input_schema": {...}},
    {"name": "process_return", "description": "...", "input_schema": {...}},
    {"name": "create_ticket", "description": "...", "input_schema": {...}},
    {
        "name": "send_email",
        "description": "...",
        "input_schema": {...},
        "cache_control": {"type": "ephemeral"},  # Caches ALL tools above
    },
]
```

### 2.3 Long Context Documents

When asking multiple questions about the same document:

```python
def multi_question_analysis(document: str, questions: list[str]) -> list[str]:
    """Ask multiple questions about the same document, caching it."""
    answers = []
    for question in questions:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Document:\n\n{document}",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": f"Question: {question}",
                        },
                    ],
                }
            ],
        )
        answers.append(response.content[0].text)
    return answers
```

### 2.4 Conversation History

In multi-turn conversations, cache previous turns:

```python
def chat_with_caching(messages: list[dict], new_message: str) -> str:
    """Continue a conversation with cached history."""
    # Mark the last existing message for caching
    cached_messages = []
    for i, msg in enumerate(messages):
        if i == len(messages) - 1:
            # Add cache_control to last existing message
            cached_msg = {
                "role": msg["role"],
                "content": [
                    {
                        "type": "text",
                        "text": msg["content"],
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
            cached_messages.append(cached_msg)
        else:
            cached_messages.append(msg)

    # Add the new message (not cached)
    cached_messages.append({"role": "user", "content": new_message})

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=cached_messages,
    )
    return response.content[0].text
```

---

## 3. Cache Hit Rates and Cost Savings

### 3.1 Pricing Model

| Token Type | Cost (Sonnet) | Relative to Base |
|---|---|---|
| Regular input tokens | Base price | 1.0x |
| Cache write tokens | 1.25x base | 25% premium |
| Cache read tokens | 0.1x base | **90% discount** |

### 3.2 Break-Even Analysis

Caching is profitable when you reuse cached content at least **twice**:

```python
def calculate_cache_savings(
    cached_tokens: int,
    num_requests: int,
    base_input_price_per_mtok: float = 3.0,  # Sonnet pricing example
) -> dict:
    """Calculate cost savings from prompt caching."""
    # Without caching
    no_cache_cost = cached_tokens * num_requests * base_input_price_per_mtok / 1_000_000

    # With caching: 1 write + (N-1) reads
    write_cost = cached_tokens * (base_input_price_per_mtok * 1.25) / 1_000_000
    read_cost = cached_tokens * (num_requests - 1) * (base_input_price_per_mtok * 0.1) / 1_000_000
    cache_cost = write_cost + read_cost

    savings = no_cache_cost - cache_cost
    savings_pct = (savings / no_cache_cost) * 100

    return {
        "without_caching": round(no_cache_cost, 4),
        "with_caching": round(cache_cost, 4),
        "savings": round(savings, 4),
        "savings_percent": round(savings_pct, 1),
        "break_even_requests": 2,  # Always breaks even at 2 requests
    }


# Example: 10,000 cached tokens, 50 requests
result = calculate_cache_savings(10_000, 50)
print(f"Without caching: ${result['without_caching']}")
print(f"With caching:    ${result['with_caching']}")
print(f"Savings:         ${result['savings']} ({result['savings_percent']}%)")
# Without caching: $1.5000
# With caching:    $0.1845
# Savings:         $1.3155 (87.7%)
```

### 3.3 Monitoring Cache Performance

```python
class CacheMonitor:
    """Track cache hit rates and cost savings across requests."""

    def __init__(self):
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_writes = 0
        self.total_cached_tokens_read = 0
        self.total_cached_tokens_written = 0

    def record(self, usage):
        """Record usage stats from an API response."""
        self.total_requests += 1
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

        if cache_read > 0:
            self.cache_hits += 1
            self.total_cached_tokens_read += cache_read
        if cache_write > 0:
            self.cache_writes += 1
            self.total_cached_tokens_written += cache_write

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    def report(self) -> str:
        return (
            f"Requests: {self.total_requests}\n"
            f"Cache hits: {self.cache_hits} ({self.hit_rate:.1%})\n"
            f"Cache writes: {self.cache_writes}\n"
            f"Tokens read from cache: {self.total_cached_tokens_read:,}\n"
            f"Tokens written to cache: {self.total_cached_tokens_written:,}"
        )


# Usage
monitor = CacheMonitor()

for query in queries:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": query}],
    )
    monitor.record(response.usage)

print(monitor.report())
```

---

## 4. TTL Management and Cache Warming

### 4.1 Cache Lifetime

Cached content has a **5-minute TTL** (Time To Live). The TTL resets each time the cached prefix is used:

```
t=0:00  Request 1 → Cache WRITE (TTL starts: 5 minutes)
t=2:00  Request 2 → Cache HIT  (TTL resets: 5 minutes from now)
t=5:00  Request 3 → Cache HIT  (TTL resets: 5 minutes from now)
t=11:00 (no requests for 6 min) → Cache EXPIRED
t=11:00 Request 4 → Cache WRITE (new cache entry)
```

### 4.2 Cache Warming Strategy

For workloads with gaps between requests, proactively warm the cache:

```python
import time
import threading


class CacheWarmer:
    """Keep a cache warm by sending periodic lightweight requests."""

    def __init__(self, system_prompt: list[dict], interval: int = 240):
        """
        Args:
            system_prompt: The system prompt with cache_control.
            interval: Seconds between warming requests (default 4 min).
        """
        self.client = anthropic.Anthropic()
        self.system_prompt = system_prompt
        self.interval = interval
        self._running = False
        self._thread = None

    def start(self):
        """Start the cache warming background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._warm_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the cache warming thread."""
        self._running = False
        if self._thread:
            self._thread.join()

    def _warm_loop(self):
        while self._running:
            try:
                # Minimal request to keep cache alive
                self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1,  # Minimize output cost
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": "ping"}],
                )
            except Exception:
                pass  # Log in production
            time.sleep(self.interval)


# Usage
warmer = CacheWarmer(
    system_prompt=[{
        "type": "text",
        "text": large_system_prompt,
        "cache_control": {"type": "ephemeral"},
    }],
    interval=240,  # Warm every 4 minutes (before 5-min TTL)
)
warmer.start()

# ... do work ...

warmer.stop()
```

### 4.3 Cache-Friendly Request Design

```python
# BAD: Dynamic content before static content breaks caching
messages = [
    {"role": "user", "content": f"Today is {date}. Analyze: {document}"}
]

# GOOD: Static content first with cache_control, dynamic content after
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Document:\n{document}",  # Static, cacheable
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": f"Today is {date}. Please analyze the document above.",  # Dynamic
            },
        ],
    }
]
```

---

## 5. Message Batches API Overview

The Message Batches API lets you send large volumes of requests asynchronously at a **50% discount**. Batch results are guaranteed within 24 hours but typically complete much faster.

### 5.1 When to Use Batches

| Use Case | Real-time API | Batch API |
|---|---|---|
| Interactive chat | Yes | No |
| Document processing (100+ docs) | Possible | **Recommended** |
| Evaluation pipelines | Possible | **Recommended** |
| Data labeling/classification | Possible | **Recommended** |
| Content generation at scale | Possible | **Recommended** |
| Latency-sensitive applications | Yes | No |

### 5.2 Batch Lifecycle

```
CREATE → in_progress → ended
                     ↗
         canceling →
```

- **created**: Batch accepted, waiting to process
- **in_progress**: Requests are being processed
- **canceling**: Cancel requested, finishing in-flight requests
- **ended**: All requests completed (check results_counts)

---

## 6. Creating and Monitoring Batch Requests

### 6.1 Creating a Batch

```python
import anthropic

client = anthropic.Anthropic()

# Define batch requests
batch_requests = [
    {
        "custom_id": f"doc-{i}",
        "params": {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": f"Summarize this document:\n\n{doc}"}
            ],
        },
    }
    for i, doc in enumerate(documents)
]

# Create the batch
batch = client.messages.batches.create(requests=batch_requests)

print(f"Batch ID: {batch.id}")
print(f"Status: {batch.processing_status}")
```

### 6.2 Monitoring Progress

```python
import time


def wait_for_batch(batch_id: str, poll_interval: int = 30) -> dict:
    """Poll a batch until completion."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)

        print(
            f"Status: {batch.processing_status} | "
            f"Succeeded: {batch.request_counts.succeeded} | "
            f"Errored: {batch.request_counts.errored} | "
            f"Processing: {batch.request_counts.processing}"
        )

        if batch.processing_status == "ended":
            return batch

        time.sleep(poll_interval)


batch_result = wait_for_batch(batch.id)
```

### 6.3 Retrieving Results

```python
def get_batch_results(batch_id: str) -> dict[str, str]:
    """Retrieve all results from a completed batch."""
    results = {}

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id

        if result.result.type == "succeeded":
            message = result.result.message
            text = message.content[0].text
            results[custom_id] = text
        elif result.result.type == "errored":
            error = result.result.error
            results[custom_id] = f"ERROR: {error.type} - {error.message}"
        elif result.result.type == "expired":
            results[custom_id] = "EXPIRED: Request did not complete in time"

    return results


results = get_batch_results(batch.id)
for doc_id, summary in sorted(results.items()):
    print(f"\n{'='*60}")
    print(f"Document: {doc_id}")
    print(f"Summary: {summary[:200]}...")
```

### 6.4 Canceling a Batch

```python
# Cancel a running batch
client.messages.batches.cancel(batch.id)

# Cancellation is async — already-processing requests will complete
# Check final state:
final = wait_for_batch(batch.id)
print(f"Succeeded: {final.request_counts.succeeded}")
print(f"Canceled: {final.request_counts.canceled}")
```

---

## 7. Batch API Pricing and Limits

### 7.1 Pricing

| Feature | Standard API | Batch API |
|---|---|---|
| Input token price | Base price | **50% of base** |
| Output token price | Base price | **50% of base** |
| Prompt caching | Available | Available |
| Cache write tokens | 1.25x base | **0.625x base** (50% off) |
| Cache read tokens | 0.1x base | **0.05x base** (50% off) |

### 7.2 Limits

- Maximum **10,000 requests** per batch (may vary, check documentation)
- Each request follows standard model limits (max tokens, etc.)
- Batches expire after **24 hours** if not completed
- Rate limits are separate from real-time API

### 7.3 Cost Calculation Example

```python
def estimate_batch_cost(
    num_requests: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    cached_tokens: int = 0,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Estimate costs for batch vs real-time processing."""
    # Sonnet pricing (per million tokens)
    prices = {
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-haiku-4-20250514": {"input": 0.80, "output": 4.0},
    }
    p = prices[model]

    uncached_tokens = avg_input_tokens - cached_tokens

    # Real-time cost
    rt_input_cost = (uncached_tokens * num_requests * p["input"]) / 1_000_000
    rt_cache_write = (cached_tokens * p["input"] * 1.25) / 1_000_000
    rt_cache_read = (cached_tokens * (num_requests - 1) * p["input"] * 0.1) / 1_000_000
    rt_output_cost = (avg_output_tokens * num_requests * p["output"]) / 1_000_000
    rt_total = rt_input_cost + rt_cache_write + rt_cache_read + rt_output_cost

    # Batch cost (50% of everything)
    batch_total = rt_total * 0.5

    return {
        "realtime_cost": round(rt_total, 4),
        "batch_cost": round(batch_total, 4),
        "batch_savings": round(rt_total - batch_total, 4),
        "batch_savings_pct": 50.0,
    }
```

---

## 8. Combining Caching with Batching

The most powerful cost optimization combines both: cache shared context and run via batch.

### 8.1 Batch with Cached System Prompt

```python
# All batch requests share the same system prompt → cache it
system_prompt = [
    {
        "type": "text",
        "text": large_instructions + "\n\n" + reference_data,
        "cache_control": {"type": "ephemeral"},
    }
]

batch_requests = [
    {
        "custom_id": f"item-{i}",
        "params": {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": f"Process: {item}"}
            ],
        },
    }
    for i, item in enumerate(items)
]

batch = client.messages.batches.create(requests=batch_requests)
```

### 8.2 Batch with Cached Document

```python
def batch_analyze_document(document: str, questions: list[str]) -> dict:
    """Ask many questions about a document using batch + caching."""
    batch_requests = [
        {
            "custom_id": f"q-{i}",
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2048,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Document:\n\n{document}",
                                "cache_control": {"type": "ephemeral"},
                            },
                            {
                                "type": "text",
                                "text": question,
                            },
                        ],
                    }
                ],
            },
        }
        for i, question in enumerate(questions)
    ]

    batch = client.messages.batches.create(requests=batch_requests)
    final = wait_for_batch(batch.id)
    return get_batch_results(batch.id)
```

### 8.3 Cost Comparison Table

For 1,000 requests with 5,000 cached tokens and 500 uncached tokens each:

| Strategy | Relative Cost |
|---|---|
| No caching, real-time | 100% (baseline) |
| Caching only | ~25% |
| Batching only | ~50% |
| **Caching + Batching** | **~12.5%** |

---

## 9. Real-World Patterns

### 9.1 Document Processing Pipeline

```python
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


class DocumentPipeline:
    """Process a large set of documents using batch + caching."""

    def __init__(self, extraction_schema: dict):
        self.client = anthropic.Anthropic()
        self.schema = extraction_schema
        self.system_prompt = [
            {
                "type": "text",
                "text": (
                    "You are a document processing assistant.\n"
                    f"Extract data according to this schema:\n"
                    f"{json.dumps(extraction_schema, indent=2)}\n\n"
                    "Return ONLY valid JSON matching the schema. No explanation."
                ),
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def process_batch(self, documents: dict[str, str]) -> dict[str, dict]:
        """Process documents via batch API with caching."""
        batch_requests = [
            {
                "custom_id": doc_id,
                "params": {
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 2048,
                    "system": self.system_prompt,
                    "messages": [
                        {"role": "user", "content": f"Extract from:\n\n{content}"}
                    ],
                },
            }
            for doc_id, content in documents.items()
        ]

        # Split into batches of 10,000
        all_results = {}
        for i in range(0, len(batch_requests), 10_000):
            chunk = batch_requests[i : i + 10_000]
            batch = self.client.messages.batches.create(requests=chunk)
            final = wait_for_batch(batch.id)
            raw_results = get_batch_results(batch.id)

            for doc_id, text in raw_results.items():
                try:
                    all_results[doc_id] = json.loads(text)
                except json.JSONDecodeError:
                    all_results[doc_id] = {"error": "Failed to parse", "raw": text}

        return all_results


# Usage
pipeline = DocumentPipeline(
    extraction_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "author": {"type": "string"},
            "date": {"type": "string"},
            "key_findings": {"type": "array", "items": {"type": "string"}},
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        },
    }
)

docs = {f"doc-{i}": text for i, text in enumerate(all_documents)}
results = pipeline.process_batch(docs)
```

### 9.2 LLM Evaluation Pipeline

```python
class EvalPipeline:
    """Evaluate LLM outputs using Claude as a judge, via batch API."""

    def __init__(self, rubric: str):
        self.client = anthropic.Anthropic()
        self.system_prompt = [
            {
                "type": "text",
                "text": (
                    "You are an LLM output evaluator.\n\n"
                    f"## Rubric\n{rubric}\n\n"
                    "Score each response on the criteria in the rubric.\n"
                    "Return JSON: {\"scores\": {\"criterion\": score}, \"total\": N, \"reasoning\": \"...\"}"
                ),
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def evaluate(self, test_cases: list[dict]) -> list[dict]:
        """
        Evaluate test cases via batch.

        Each test_case: {"id": str, "prompt": str, "response": str}
        """
        batch_requests = [
            {
                "custom_id": tc["id"],
                "params": {
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "system": self.system_prompt,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"## Prompt\n{tc['prompt']}\n\n"
                                f"## Response to Evaluate\n{tc['response']}"
                            ),
                        }
                    ],
                },
            }
            for tc in test_cases
        ]

        batch = self.client.messages.batches.create(requests=batch_requests)
        final = wait_for_batch(batch.id)
        raw = get_batch_results(batch.id)

        results = []
        for tc_id, text in raw.items():
            try:
                parsed = json.loads(text)
                parsed["id"] = tc_id
                results.append(parsed)
            except json.JSONDecodeError:
                results.append({"id": tc_id, "error": text})

        return sorted(results, key=lambda r: r.get("total", 0), reverse=True)
```

### 9.3 Hybrid Real-Time + Batch Architecture

```python
class HybridProcessor:
    """Use real-time for urgent requests, batch for bulk processing."""

    def __init__(self, system_prompt: list[dict]):
        self.client = anthropic.Anthropic()
        self.system_prompt = system_prompt
        self.pending_batch = []

    def process_urgent(self, content: str) -> str:
        """Process a single item in real-time with caching."""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=self.system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text

    def queue_for_batch(self, item_id: str, content: str):
        """Queue an item for batch processing."""
        self.pending_batch.append({
            "custom_id": item_id,
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "system": self.system_prompt,
                "messages": [{"role": "user", "content": content}],
            },
        })

    def flush_batch(self) -> str:
        """Submit all queued items as a batch. Returns batch ID."""
        if not self.pending_batch:
            return None
        batch = self.client.messages.batches.create(requests=self.pending_batch)
        self.pending_batch = []
        return batch.id
```

---

## 10. Exercises

### Exercise 1: Cache Performance Tracker

Build a wrapper that tracks and reports cache performance:

```python
"""
Exercise 1 starter code — build a caching-aware API wrapper.
"""
import anthropic


class CachedClient:
    """API wrapper that manages caching and tracks performance."""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_writes": 0,
            "tokens_saved": 0,
            "estimated_savings_usd": 0.0,
        }

    def create_message(self, system_text: str, user_message: str, **kwargs) -> str:
        """
        Send a message with automatic system prompt caching.

        - Wrap system_text with cache_control
        - Track cache hits/misses
        - Calculate cost savings
        """
        # TODO: Implement caching logic
        # TODO: Record usage stats
        # TODO: Calculate savings
        pass

    def report(self) -> str:
        """Return a formatted report of cache performance."""
        # TODO: Format and return stats
        pass
```

### Exercise 2: Batch Document Classifier

Create a batch pipeline that classifies documents into categories:

```python
"""
Exercise 2 starter code — batch document classification.
"""


class BatchClassifier:
    """Classify documents in bulk using the Batch API."""

    def __init__(self, categories: list[str]):
        self.client = anthropic.Anthropic()
        self.categories = categories

    def classify(self, documents: dict[str, str]) -> dict[str, dict]:
        """
        Classify each document into one of the categories.

        Args:
            documents: {doc_id: doc_text} mapping

        Returns:
            {doc_id: {"category": str, "confidence": float, "reasoning": str}}
        """
        # TODO: Build batch requests with cached system prompt
        # TODO: Submit batch
        # TODO: Wait for completion
        # TODO: Parse and return results
        pass
```

### Exercise 3: Cache Warming Service

Implement a service that keeps multiple cache entries warm:

```python
"""
Exercise 3 starter code — multi-entry cache warming service.
"""
import threading


class CacheWarmingService:
    """Keep multiple cache entries warm simultaneously."""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.entries = {}  # name -> system_prompt
        self._running = False

    def register(self, name: str, system_prompt: list[dict], interval: int = 240):
        """Register a cache entry to keep warm."""
        # TODO: Store entry configuration
        pass

    def unregister(self, name: str):
        """Remove a cache entry from warming."""
        # TODO: Remove entry
        pass

    def start(self):
        """Start warming all registered entries."""
        # TODO: Start background threads
        pass

    def stop(self):
        """Stop all warming threads."""
        # TODO: Clean shutdown
        pass

    def status(self) -> dict:
        """Return the status of all cache entries."""
        # TODO: Report last warm time, hit rate, etc.
        pass
```

### Exercise 4: Cost Optimizer

Build a tool that analyzes API usage and recommends optimizations:

```python
"""
Exercise 4 starter code — API usage cost optimizer.
"""


class CostOptimizer:
    """Analyze API usage patterns and recommend cost optimizations."""

    def __init__(self):
        self.usage_log = []

    def log_request(self, usage: dict, request_metadata: dict):
        """Log a request's usage for analysis."""
        # TODO: Store usage data
        pass

    def analyze(self) -> dict:
        """
        Analyze logged usage and return recommendations.

        Returns:
            {
                "total_cost": float,
                "potential_savings": float,
                "recommendations": [
                    {
                        "type": "enable_caching" | "use_batch" | "switch_model",
                        "description": str,
                        "estimated_savings": float,
                    }
                ]
            }
        """
        # TODO: Identify requests with repeated system prompts (→ caching)
        # TODO: Identify non-urgent bulk requests (→ batching)
        # TODO: Identify simple tasks using expensive models (→ model switch)
        pass
```

---

**Previous**: [23. Vision Agents](./23_Vision_Agents.md) | **Next**: [25. RAG Patterns](./25_RAG_Patterns.md)
