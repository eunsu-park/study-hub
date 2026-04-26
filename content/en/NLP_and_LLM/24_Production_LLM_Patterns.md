# 24. Production LLM Patterns

## Learning Objectives

- Design robust LLM application architectures for production workloads
- Implement caching strategies (semantic cache and exact match) to reduce cost and latency
- Build fallback, retry, and multi-model routing patterns for reliability
- Set up observability with LangSmith and Phoenix for debugging and monitoring
- Apply a comprehensive deployment checklist for production-ready LLM systems

---

## Theory & Principles

A production LLM system has three concerns that prototype code can ignore: **cost** (LLM calls are expensive at scale), **latency** (user-facing apps demand sub-second responses), and **reliability** (LLMs occasionally fail, models are deprecated, providers go down). The patterns in this lesson — caching, fallbacks, routing, observability, A/B testing — are the operational layer that turns a working prototype into a system that runs 24/7 at scale without burning a hole in your budget.

This section covers:

- **(A) Cost economics** — where the money goes, the per-token math, the orders of magnitude that matter.
- **(B) Caching** — exact-match cache (free wins), semantic cache (the trade-off), prompt caching (provider-side).
- **(C) Latency optimization** — the wait-for-token hierarchy: TTFB, streaming, parallelization.
- **(D) Fallback and retry patterns** — circuit breaker, multi-provider failover, exponential backoff.
- **(E) Multi-model routing** — using a small model when possible, large model when necessary.
- **(F) Observability** — distributed tracing for LLM apps, the things you must log, LangSmith / Phoenix.
- **(G) A/B testing and gradual rollout** — measuring whether changes actually help in production.
- **(H) Rate limiting and quotas** — protecting yourself and your users from cost runaway.

### A. Cost Economics

Per-token pricing varies by model and provider. As of late 2025:
- Frontier models (GPT-4 class): $2-15 / million input tokens, $10-60 / million output tokens.
- Mid-tier (Claude Sonnet, GPT-4o-mini): $0.15-3 / million input tokens.
- Cheap (GPT-4o-mini, open-source): $0.05-0.5 / million tokens.

A typical RAG query: ~1500 input tokens (system prompt + retrieved chunks + user query), ~300 output tokens. At GPT-4 prices: `(1500 · $5 + 300 · $15) / 10^6 = $0.012 / query`. At 1M queries/day, that is $12K/day — $4M/year — *just for the LLM calls*. Embedding, retrieval, infrastructure are extra.

This is the central economic reality of LLM apps. Every optimization in the lesson — caching, routing, smaller models — exists because of this. Doubling the cache hit rate halves the LLM bill.

### B. Caching

**B.1 Exact-match cache.** Hash the input (prompt + parameters), look it up, return cached response if hit. Trivial to implement. Hit rate depends on input distribution: high (30-50%) for FAQ-style apps, low (<5%) for open-ended chat.

**B.2 Semantic cache.** Embed the query, look up similar past queries (cosine similarity > threshold), return their cached responses if similar enough. Trades exactness for higher hit rate. Risk: returning a cached response to a query that *seems* similar but actually requires a different answer.

```
exact_cache[hash(query)] = response  # easy
semantic_cache: embed(query) → top-1 in past queries → if sim > 0.97, return cached response
```

The threshold (0.95-0.99) is the lever. Higher = safer but lower hit rate. Production systems usually set conservative thresholds and verify with sampling.

**B.3 Provider-side prompt caching** (Anthropic, OpenAI). When you make the same prompt prefix multiple times (e.g., a long system prompt + retrieved chunks that don't change), the provider caches the prefix's KV-cache and reuses it. Reduces input-token cost by 90% on cached portions and improves TTFB significantly. Free to enable, just mark the prefix as cacheable.

### C. Latency Optimization

User-perceived latency in an LLM app:

```
total = retrieve_latency + LLM_TTFB + decode_time
```

- **Retrieve latency**: 10-200ms for vector search; 50-500ms for web search APIs. Optimization: index tuning, smaller embedding models, caching.
- **LLM TTFB**: 200ms-2s. Dominated by prompt length (prefill) and provider load. Optimization: shorter prompts, prompt caching, smaller models, dedicated capacity.
- **Decode time**: number of output tokens × per-token speed (~50-200 tokens/sec). Optimization: shorter responses, streaming.

**Streaming changes UX more than absolute latency.** A response that starts in 0.5s and streams smoothly feels faster than one that waits 1.5s and dumps. Always stream user-facing responses.

### D. Fallback and Retry Patterns

**D.1 Retry with exponential backoff.** Transient errors (rate limit, timeout, 5xx) — wait 1s, 2s, 4s, 8s, then give up. Standard.

**D.2 Circuit breaker.** If a provider has been failing for a while, stop trying and immediately route to a fallback. Prevents cascading failures.

**D.3 Multi-provider failover.** Primary provider fails → secondary → tertiary. Each can be a different LLM (GPT → Claude → Gemini → open-source). Cost: each provider needs auth keys, schema differences must be abstracted.

**D.4 Static fallback.** If all LLM calls fail, return a canned response: "I'm having trouble right now, please try again later." Better than a 500 error to the user.

The pattern: **always have a working response path**, even if it's degraded.

### E. Multi-Model Routing

Most production traffic is simple. Routing easy queries to a cheap model and reserving the expensive one for hard queries cuts cost by 5-10× without quality loss.

**E.1 Heuristic routing.** Rule-based: short queries → cheap model, long queries → expensive. Crude but effective at the extremes.

**E.2 Classifier-based routing.** A small classifier predicts difficulty; route accordingly. Trained on a labeled dataset (e.g., from past A/B tests). Better than heuristic but adds training/maintenance cost.

**E.3 Cascading.** Try the cheap model first; if confidence is low or output fails validation, retry with the expensive model. Standard pattern.

**E.4 Model-as-a-router.** A small LLM (or the cheap model itself) inspects the query and decides which downstream model handles it. Most flexible, adds an extra LLM call.

### F. Observability

LLM apps are harder to debug than traditional apps because the "logic" is implicit in prompts and weights. Observability requires logging:

- Full input prompt (with all retrieved chunks, etc.)
- Model identity and parameters (temperature, top_p, etc.)
- Full output
- Latency breakdown (retrieve / TTFB / decode)
- Token counts, cost
- Tool calls and their results (if agents)
- User feedback (thumbs up/down) when collected

**LangSmith** (LangChain), **Phoenix** (Arize), **Langfuse**, **Helicone**: hosted/OSS observability platforms for LLM apps. Each instruments your LLM calls (one-line setup with most LLM frameworks) and provides trace UIs, latency breakdowns, cost dashboards, and error analysis.

The cost of observability is small (one HTTP call per LLM call to the trace backend); the value is enormous (debugging a production bug without it is essentially impossible).

### G. A/B Testing and Gradual Rollout

Changes that look good in offline eval often don't help in production (or actively hurt). The only way to know: serve both versions to real users, compare engagement metrics.

Standard approach:
1. Deploy the candidate alongside the production version.
2. Route a small fraction (1-10%) of traffic to the candidate.
3. Track engagement metrics (response acceptance, follow-up rate, user ratings, downstream conversion).
4. Use statistical hypothesis tests to determine significance.
5. If the candidate wins (or doesn't lose), gradually ramp up traffic.

For LLM-specific changes, also track: cost per request, latency p50/p95/p99, error rate, refusal rate.

### H. Rate Limiting and Quotas

**Per-user limits**: prevent any one user from monopolizing capacity or causing surprise bills. Token-based (e.g., 100K tokens/day) is more aligned with cost than request-based.

**Global limits**: protect against runaway scenarios — a bug that loops, a viral incident. Hard cap on total tokens per minute / hour / day.

**Cost ceiling**: a separate budget guard that disables LLM calls if cost exceeds a threshold. The last line of defense.

### From Theory to the Functions Below

- §1 (architecture) — frames the §A-§H concerns at a system level.
- §2 (caching) — implements §B.1 exact and §B.2 semantic cache.
- §3 (cost optimization) — applies §A and §E to reduce per-query cost.
- §4 (fallback/retry) — implements §D patterns.
- §5 (A/B testing) — implements §G with traffic splitting and significance testing.
- §6 (observability) — wires LangSmith and Phoenix per §F.
- §7 (rate limiting / routing) — combines §E multi-model routing with §H rate limits.
- §8 (deployment checklist) — synthesizes §A-§H into a launch-ready checklist.

---

## 1. LLM Application Architecture

### Production Architecture Overview

> **Production LLM Stack**
>
> ```
> Client Request
>     |
>     v
> [Rate Limiter] -> [Input Validator] -> [Cache Layer]
>     |                                        |
>     | (cache miss)                     (cache hit)
>     v                                        |
> [Router] -> [Model A / Model B / ...]        |
>     |                                        |
>     v                                        |
> [Output Validator] -> [Content Filter]       |
>     |                                        |
>     v                                        v
> [Response Logger] ----------------------> Client
> ```

### Architecture Comparison

| Pattern | Latency | Cost | Complexity | Reliability |
|---------|---------|------|------------|-------------|
| Direct API Call | High | High | Low | Low |
| + Caching | Medium | Medium | Medium | Medium |
| + Fallback Models | Medium | Medium | Medium | High |
| + Multi-Model Router | Low-Medium | Low-Medium | High | Very High |
| Full Production Stack | Low-Medium | Low | High | Very High |

### Base Application Structure

```python
from dataclasses import dataclass, field
from typing import Any
import time
import uuid
import logging

logger = logging.getLogger(__name__)

@dataclass
class LLMRequest:
    """Standardized request object."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[dict] = field(default_factory=list)
    model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 2048
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class LLMResponse:
    """Standardized response object."""
    request_id: str
    content: str
    model: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    cached: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output

    @property
    def estimated_cost(self) -> float:
        """Estimate cost based on model pricing (approximate)."""
        pricing = {
            "gpt-4o": (2.50, 10.00),           # per 1M tokens (input, output)
            "gpt-4o-mini": (0.15, 0.60),
            "claude-sonnet-4-20250514": (3.00, 15.00),
            "claude-haiku-4-20250514": (0.25, 1.25),
        }
        input_rate, output_rate = pricing.get(self.model, (5.0, 15.0))
        return (
            self.tokens_input * input_rate / 1_000_000
            + self.tokens_output * output_rate / 1_000_000
        )
```

---

## 2. Caching Strategies

### Exact Match Cache

```python
import hashlib
import json
import sqlite3
import time
from pathlib import Path

class ExactMatchCache:
    """SQLite-backed exact match cache for LLM responses."""

    def __init__(self, db_path: str = "llm_cache.db", ttl_hours: float = 24):
        self.db_path = db_path
        self.ttl_seconds = ttl_hours * 3600
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    hit_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON cache(created_at)
            """)

    def _make_key(self, messages: list[dict], model: str,
                  temperature: float) -> str:
        """Create deterministic cache key from request parameters."""
        payload = json.dumps({
            "messages": messages,
            "model": model,
            "temperature": temperature,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, messages: list[dict], model: str,
            temperature: float) -> str | None:
        """Retrieve cached response if available and not expired."""
        key = self._make_key(messages, model, temperature)
        cutoff = time.time() - self.ttl_seconds

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT response FROM cache WHERE cache_key = ? AND created_at > ?",
                (key, cutoff),
            ).fetchone()

            if row:
                conn.execute(
                    "UPDATE cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                    (key,),
                )
                return row[0]
        return None

    def put(self, messages: list[dict], model: str,
            temperature: float, response: str):
        """Store a response in the cache."""
        key = self._make_key(messages, model, temperature)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (cache_key, response, model, created_at) "
                "VALUES (?, ?, ?, ?)",
                (key, response, model, time.time()),
            )

    def evict_expired(self):
        """Remove expired entries."""
        cutoff = time.time() - self.ttl_seconds
        with sqlite3.connect(self.db_path) as conn:
            deleted = conn.execute(
                "DELETE FROM cache WHERE created_at < ?", (cutoff,)
            ).rowcount
            logger.info(f"Evicted {deleted} expired cache entries")

    def stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            total_hits = conn.execute("SELECT SUM(hit_count) FROM cache").fetchone()[0] or 0
            return {"total_entries": total, "total_hits": total_hits}
```

### Semantic Cache

```python
import numpy as np
from openai import OpenAI

client = OpenAI()

class SemanticCache:
    """Cache that matches semantically similar queries."""

    def __init__(self, similarity_threshold: float = 0.92,
                 max_entries: int = 10000):
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self.entries: list[dict] = []  # In production: use a vector DB
        self.embeddings: list[np.ndarray] = []

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text string."""
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return np.array(response.data[0].embedding)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _query_to_key(self, messages: list[dict]) -> str:
        """Extract the meaningful query from messages."""
        # Use the last user message as the cache key
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        return user_messages[-1] if user_messages else ""

    def get(self, messages: list[dict]) -> str | None:
        """Find a semantically similar cached response."""
        if not self.entries:
            return None

        query = self._query_to_key(messages)
        query_embedding = self._get_embedding(query)

        # Find most similar entry
        best_score = 0.0
        best_idx = -1

        for i, emb in enumerate(self.embeddings):
            score = self._cosine_similarity(query_embedding, emb)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= self.threshold:
            logger.info(f"Semantic cache hit (similarity={best_score:.4f})")
            return self.entries[best_idx]["response"]

        return None

    def put(self, messages: list[dict], response: str):
        """Store a query-response pair in the semantic cache."""
        query = self._query_to_key(messages)
        embedding = self._get_embedding(query)

        self.entries.append({
            "query": query,
            "response": response,
            "timestamp": time.time(),
        })
        self.embeddings.append(embedding)

        # Evict oldest if over limit
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)
            self.embeddings.pop(0)

# Combined cache strategy
class TieredCache:
    """Two-tier cache: exact match first, then semantic."""

    def __init__(self):
        self.exact = ExactMatchCache(ttl_hours=48)
        self.semantic = SemanticCache(similarity_threshold=0.93)

    def get(self, messages: list[dict], model: str,
            temperature: float) -> tuple[str | None, str]:
        """Try exact match first, then semantic. Returns (response, cache_type)."""
        # Tier 1: Exact match
        exact_result = self.exact.get(messages, model, temperature)
        if exact_result:
            return exact_result, "exact"

        # Tier 2: Semantic match (only for low-temperature requests)
        if temperature <= 0.3:
            semantic_result = self.semantic.get(messages)
            if semantic_result:
                return semantic_result, "semantic"

        return None, "miss"

    def put(self, messages: list[dict], model: str,
            temperature: float, response: str):
        self.exact.put(messages, model, temperature, response)
        if temperature <= 0.3:
            self.semantic.put(messages, response)
```

---

## 3. Cost and Latency Optimization

### Cost Tracking and Budgets

```python
from collections import defaultdict
from datetime import datetime, timedelta
import threading

class CostTracker:
    """Track and enforce LLM spending budgets."""

    PRICING = {
        # (input_per_1M_tokens, output_per_1M_tokens)
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "claude-sonnet-4-20250514": (3.00, 15.00),
        "claude-haiku-4-20250514": (0.25, 1.25),
    }

    def __init__(self, daily_budget: float = 50.0, monthly_budget: float = 1000.0):
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        self._daily_spend: dict[str, float] = defaultdict(float)  # date -> spend
        self._monthly_spend: dict[str, float] = defaultdict(float)  # month -> spend
        self._lock = threading.Lock()

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Record token usage and return cost."""
        input_rate, output_rate = self.PRICING.get(model, (5.0, 15.0))
        cost = (
            input_tokens * input_rate / 1_000_000
            + output_tokens * output_rate / 1_000_000
        )

        today = datetime.now().strftime("%Y-%m-%d")
        month = datetime.now().strftime("%Y-%m")

        with self._lock:
            self._daily_spend[today] += cost
            self._monthly_spend[month] += cost

        return cost

    def check_budget(self) -> dict:
        """Check if within budget limits."""
        today = datetime.now().strftime("%Y-%m-%d")
        month = datetime.now().strftime("%Y-%m")

        daily_spend = self._daily_spend.get(today, 0)
        monthly_spend = self._monthly_spend.get(month, 0)

        return {
            "daily_spend": round(daily_spend, 4),
            "daily_budget": self.daily_budget,
            "daily_remaining": round(self.daily_budget - daily_spend, 4),
            "daily_exceeded": daily_spend >= self.daily_budget,
            "monthly_spend": round(monthly_spend, 4),
            "monthly_budget": self.monthly_budget,
            "monthly_remaining": round(self.monthly_budget - monthly_spend, 4),
            "monthly_exceeded": monthly_spend >= self.monthly_budget,
        }

    def can_proceed(self) -> bool:
        """Check if a new request is within budget."""
        budget = self.check_budget()
        return not budget["daily_exceeded"] and not budget["monthly_exceeded"]
```

### Prompt Optimization

```python
class PromptOptimizer:
    """Reduce token count without losing quality."""

    @staticmethod
    def compress_system_prompt(prompt: str) -> str:
        """Remove unnecessary verbosity from system prompts."""
        # Remove excessive whitespace
        import re
        prompt = re.sub(r"\n{3,}", "\n\n", prompt)
        prompt = re.sub(r" {2,}", " ", prompt)
        return prompt.strip()

    @staticmethod
    def truncate_context(messages: list[dict], max_context_tokens: int = 4000,
                         preserve_last_n: int = 4) -> list[dict]:
        """Truncate conversation history while preserving recent messages."""
        if len(messages) <= preserve_last_n:
            return messages

        # Always keep system message and last N messages
        system = [m for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]

        preserved = non_system[-preserve_last_n:]

        # Summarize dropped messages
        dropped = non_system[:-preserve_last_n]
        if dropped:
            summary_msg = {
                "role": "user",
                "content": f"[Previous {len(dropped)} messages summarized: "
                           f"The conversation covered various topics. "
                           f"Please continue from the recent context below.]",
            }
            return system + [summary_msg] + preserved

        return system + preserved

    @staticmethod
    def select_model_by_complexity(messages: list[dict]) -> str:
        """Route to cheaper model for simple tasks."""
        last_user_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_user_msg = m["content"]
                break

        # Simple heuristics for model selection
        word_count = len(last_user_msg.split())

        # Short, simple queries -> cheap model
        if word_count < 30 and "?" in last_user_msg:
            return "gpt-4o-mini"

        # Complex reasoning -> powerful model
        complex_keywords = ["analyze", "compare", "design", "architect",
                          "debug", "optimize", "explain in detail"]
        if any(kw in last_user_msg.lower() for kw in complex_keywords):
            return "gpt-4o"

        return "gpt-4o-mini"  # Default to cheaper model
```

---

## 4. Fallback and Retry Patterns

### Multi-Provider Fallback

```python
from openai import OpenAI
from anthropic import Anthropic
import time

class LLMRouter:
    """Route requests across multiple providers with fallback."""

    def __init__(self):
        self.openai = OpenAI()
        self.anthropic = Anthropic()
        self.cost_tracker = CostTracker()

    def _call_openai(self, messages: list[dict], model: str,
                     temperature: float, max_tokens: int) -> LLMResponse:
        start = time.time()
        response = self.openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = (time.time() - start) * 1000
        usage = response.usage
        self.cost_tracker.record(model, usage.prompt_tokens, usage.completion_tokens)
        return LLMResponse(
            request_id="",
            content=response.choices[0].message.content,
            model=model,
            tokens_input=usage.prompt_tokens,
            tokens_output=usage.completion_tokens,
            latency_ms=latency,
        )

    def _call_anthropic(self, messages: list[dict], model: str,
                        temperature: float, max_tokens: int) -> LLMResponse:
        start = time.time()
        # Convert OpenAI format to Anthropic format
        system = ""
        anthropic_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                anthropic_messages.append(m)

        response = self.anthropic.messages.create(
            model=model,
            system=system,
            messages=anthropic_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = (time.time() - start) * 1000
        self.cost_tracker.record(
            model, response.usage.input_tokens, response.usage.output_tokens
        )
        return LLMResponse(
            request_id="",
            content=response.content[0].text,
            model=model,
            tokens_input=response.usage.input_tokens,
            tokens_output=response.usage.output_tokens,
            latency_ms=latency,
        )

    def call(self, request: LLMRequest) -> LLMResponse:
        """Call LLM with automatic fallback chain."""
        # Define fallback chain
        fallback_chain = [
            ("openai", request.model, request.temperature),
            ("openai", "gpt-4o-mini", request.temperature),
            ("anthropic", "claude-sonnet-4-20250514", request.temperature),
            ("anthropic", "claude-haiku-4-20250514", request.temperature),
        ]

        last_error = None

        for provider, model, temp in fallback_chain:
            try:
                if provider == "openai":
                    response = self._call_openai(
                        request.messages, model, temp, request.max_tokens
                    )
                else:
                    response = self._call_anthropic(
                        request.messages, model, temp, request.max_tokens
                    )

                response.request_id = request.request_id
                logger.info(
                    f"[{request.request_id}] Success: {provider}/{model} "
                    f"({response.latency_ms:.0f}ms)"
                )
                return response

            except Exception as e:
                last_error = e
                logger.warning(
                    f"[{request.request_id}] Failed {provider}/{model}: {e}"
                )
                continue

        raise RuntimeError(
            f"All providers failed for request {request.request_id}. "
            f"Last error: {last_error}"
        )
```

### Retry with Exponential Backoff

```python
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log,
)

class RetryableLLMClient:
    """LLM client with configurable retry behavior."""

    def __init__(self):
        self.client = OpenAI()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((
            ConnectionError,
            TimeoutError,
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def call(self, messages: list[dict], model: str = "gpt-4o",
             temperature: float = 0.3) -> str:
        """Make an LLM call with automatic retry on transient failures."""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=30.0,
        )
        return response.choices[0].message.content

    def call_with_budget_check(self, request: LLMRequest,
                               cost_tracker: CostTracker) -> LLMResponse | None:
        """Only proceed if within budget."""
        if not cost_tracker.can_proceed():
            logger.error("Budget exceeded. Request blocked.")
            return None
        content = self.call(
            request.messages, request.model, request.temperature
        )
        return LLMResponse(
            request_id=request.request_id,
            content=content,
            model=request.model,
            tokens_input=0,
            tokens_output=0,
            latency_ms=0,
        )
```

---

## 5. A/B Testing LLM Responses

### A/B Testing Framework

```python
import random
import hashlib
from dataclasses import dataclass

@dataclass
class Variant:
    name: str
    model: str
    temperature: float
    system_prompt: str
    weight: float = 0.5  # Traffic allocation

class ABTestManager:
    """A/B test different LLM configurations."""

    def __init__(self):
        self.experiments: dict[str, list[Variant]] = {}
        self.results: dict[str, list[dict]] = defaultdict(list)

    def create_experiment(self, name: str, variants: list[Variant]):
        """Create a new A/B test experiment."""
        total_weight = sum(v.weight for v in variants)
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Variant weights must sum to 1.0, got {total_weight}")
        self.experiments[name] = variants

    def assign_variant(self, experiment: str, user_id: str) -> Variant:
        """Deterministically assign a user to a variant (sticky assignment)."""
        variants = self.experiments[experiment]
        # Hash user_id for deterministic assignment
        hash_val = int(hashlib.md5(
            f"{experiment}:{user_id}".encode()
        ).hexdigest(), 16)
        threshold = hash_val % 1000 / 1000.0

        cumulative = 0.0
        for variant in variants:
            cumulative += variant.weight
            if threshold < cumulative:
                return variant

        return variants[-1]  # Fallback

    def record_result(self, experiment: str, variant_name: str,
                      metrics: dict):
        """Record experiment result."""
        self.results[experiment].append({
            "variant": variant_name,
            "timestamp": time.time(),
            **metrics,
        })

    def get_summary(self, experiment: str) -> dict:
        """Get experiment results summary."""
        results = self.results[experiment]
        by_variant = defaultdict(list)
        for r in results:
            by_variant[r["variant"]].append(r)

        summary = {}
        for variant_name, variant_results in by_variant.items():
            latencies = [r.get("latency_ms", 0) for r in variant_results]
            ratings = [r.get("user_rating", 0) for r in variant_results
                       if "user_rating" in r]
            costs = [r.get("cost", 0) for r in variant_results]

            summary[variant_name] = {
                "total_requests": len(variant_results),
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
                "avg_rating": sum(ratings) / len(ratings) if ratings else 0,
                "total_cost": sum(costs),
                "avg_cost": sum(costs) / len(costs) if costs else 0,
            }
        return summary

# Usage
ab = ABTestManager()
ab.create_experiment("prompt-style-v2", [
    Variant("concise", "gpt-4o-mini", 0.2,
            "Be concise and direct. Answer in 2-3 sentences.", weight=0.5),
    Variant("detailed", "gpt-4o", 0.3,
            "Provide thorough, detailed answers with examples.", weight=0.5),
])

# Per-request routing
user_id = "user_12345"
variant = ab.assign_variant("prompt-style-v2", user_id)
print(f"User {user_id} assigned to variant: {variant.name}")
```

---

## 6. Observability

### LangSmith Integration

```python
import os
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Configure LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "production-llm-app"
# os.environ["LANGCHAIN_API_KEY"] = "your-key"

langsmith = Client()

# All LangChain calls are automatically traced
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}"),
])

chain = prompt | llm

# This call is automatically logged to LangSmith
result = chain.invoke({"question": "What is retrieval-augmented generation?"})

# Manual trace annotation
from langsmith import traceable

@traceable(name="custom-rag-pipeline", run_type="chain")
def rag_pipeline(query: str) -> str:
    """Custom RAG pipeline with LangSmith tracing."""
    # Each sub-step is automatically traced
    docs = retrieve_documents(query)
    context = format_context(docs)
    answer = generate_answer(query, context)
    return answer

@traceable(run_type="retriever")
def retrieve_documents(query: str) -> list[str]:
    return ["doc1 content", "doc2 content"]

@traceable(run_type="chain")
def format_context(docs: list[str]) -> str:
    return "\n\n".join(docs)

@traceable(run_type="llm")
def generate_answer(query: str, context: str) -> str:
    response = llm.invoke(f"Context: {context}\n\nQuestion: {query}")
    return response.content
```

### Phoenix (Arize) Integration

```python
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor

# Start Phoenix for local observability
session = px.launch_app()
print(f"Phoenix UI: {session.url}")

# Auto-instrument OpenAI calls
tracer_provider = register(project_name="llm-production")
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# All OpenAI calls now appear in the Phoenix dashboard
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain vector databases"}],
)

# Phoenix provides:
# - Trace visualization (spans, latency, tokens)
# - Evaluation metrics
# - Embedding visualization
# - Dataset management
```

### Custom Observability

```python
import json
import time
from collections import defaultdict
from datetime import datetime

class LLMObserver:
    """Lightweight observability for LLM applications."""

    def __init__(self):
        self.traces: list[dict] = []
        self.metrics = defaultdict(list)

    def trace(self, func):
        """Decorator to trace LLM calls."""
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            trace_id = str(uuid.uuid4())
            start = time.time()

            trace_entry = {
                "trace_id": trace_id,
                "function": func.__name__,
                "start_time": datetime.now().isoformat(),
                "args_preview": str(args)[:200],
            }

            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start) * 1000

                trace_entry.update({
                    "status": "success",
                    "duration_ms": round(duration, 2),
                    "result_preview": str(result)[:200],
                })
                self.metrics["latency_ms"].append(duration)
                self.metrics["success"].append(1)
                return result

            except Exception as e:
                duration = (time.time() - start) * 1000
                trace_entry.update({
                    "status": "error",
                    "duration_ms": round(duration, 2),
                    "error": str(e),
                })
                self.metrics["errors"].append(str(e))
                self.metrics["success"].append(0)
                raise

            finally:
                self.traces.append(trace_entry)

        return wrapper

    def dashboard(self) -> dict:
        """Get observability dashboard data."""
        latencies = self.metrics.get("latency_ms", [])
        successes = self.metrics.get("success", [])

        return {
            "total_requests": len(successes),
            "success_rate": sum(successes) / len(successes) if successes else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "p50_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0,
            "p99_latency_ms": (
                sorted(latencies)[int(len(latencies) * 0.99)]
                if latencies else 0
            ),
            "total_errors": len(self.metrics.get("errors", [])),
            "recent_errors": self.metrics.get("errors", [])[-5:],
        }

# Usage
observer = LLMObserver()

@observer.trace
def llm_call(query: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}],
    )
    return response.choices[0].message.content

# Make calls
llm_call("What is Python?")
llm_call("Explain Docker")

# View dashboard
print(json.dumps(observer.dashboard(), indent=2))
```

---

## 7. Rate Limiting and Multi-Model Routing

### Token Bucket Rate Limiter

```python
import threading
import time

class TokenBucketRateLimiter:
    """Rate limiter using the token bucket algorithm."""

    def __init__(self, requests_per_minute: int = 60,
                 tokens_per_minute: int = 100_000):
        self.rpm_limit = requests_per_minute
        self.tpm_limit = tokens_per_minute
        self.request_tokens = requests_per_minute
        self.token_tokens = tokens_per_minute
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.request_tokens = min(
            self.rpm_limit,
            self.request_tokens + elapsed * (self.rpm_limit / 60),
        )
        self.token_tokens = min(
            self.tpm_limit,
            self.token_tokens + elapsed * (self.tpm_limit / 60),
        )
        self.last_refill = now

    def acquire(self, estimated_tokens: int = 1000) -> bool:
        """Try to acquire capacity for a request."""
        with self._lock:
            self._refill()
            if self.request_tokens >= 1 and self.token_tokens >= estimated_tokens:
                self.request_tokens -= 1
                self.token_tokens -= estimated_tokens
                return True
            return False

    def wait_and_acquire(self, estimated_tokens: int = 1000,
                         timeout: float = 60.0) -> bool:
        """Wait until capacity is available."""
        start = time.time()
        while time.time() - start < timeout:
            if self.acquire(estimated_tokens):
                return True
            time.sleep(0.1)
        return False
```

### Intelligent Multi-Model Router

```python
class ModelRouter:
    """Route requests to optimal model based on task characteristics."""

    MODEL_PROFILES = {
        "gpt-4o": {
            "provider": "openai",
            "capabilities": ["reasoning", "coding", "creative", "analysis"],
            "speed": "medium",
            "cost": "high",
            "context_window": 128_000,
        },
        "gpt-4o-mini": {
            "provider": "openai",
            "capabilities": ["classification", "extraction", "simple_qa"],
            "speed": "fast",
            "cost": "low",
            "context_window": 128_000,
        },
        "claude-sonnet-4-20250514": {
            "provider": "anthropic",
            "capabilities": ["reasoning", "coding", "analysis", "long_context"],
            "speed": "medium",
            "cost": "high",
            "context_window": 200_000,
        },
        "claude-haiku-4-20250514": {
            "provider": "anthropic",
            "capabilities": ["classification", "extraction", "simple_qa"],
            "speed": "fast",
            "cost": "low",
            "context_window": 200_000,
        },
    }

    def __init__(self):
        self.rate_limiters = {
            "openai": TokenBucketRateLimiter(rpm=500, tpm=800_000),
            "anthropic": TokenBucketRateLimiter(rpm=400, tpm=400_000),
        }

    def classify_task(self, messages: list[dict]) -> dict:
        """Classify the task to determine routing."""
        last_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_msg = m["content"]
                break

        total_tokens = sum(len(m.get("content", "").split()) * 1.3
                          for m in messages)

        # Simple heuristic classification
        task_type = "simple_qa"
        if any(kw in last_msg.lower() for kw in ["code", "implement", "debug", "function"]):
            task_type = "coding"
        elif any(kw in last_msg.lower() for kw in ["analyze", "compare", "evaluate"]):
            task_type = "analysis"
        elif any(kw in last_msg.lower() for kw in ["write", "create", "draft", "story"]):
            task_type = "creative"
        elif any(kw in last_msg.lower() for kw in ["classify", "categorize", "label"]):
            task_type = "classification"
        elif any(kw in last_msg.lower() for kw in ["extract", "parse", "find"]):
            task_type = "extraction"

        return {
            "task_type": task_type,
            "estimated_tokens": int(total_tokens),
            "needs_long_context": total_tokens > 50_000,
        }

    def select_model(self, messages: list[dict],
                     prefer_fast: bool = False,
                     prefer_cheap: bool = False) -> str:
        """Select the optimal model for the request."""
        task = self.classify_task(messages)

        candidates = []
        for model, profile in self.MODEL_PROFILES.items():
            # Check capability match
            capability_match = task["task_type"] in profile["capabilities"]

            # Check context window
            fits_context = task["estimated_tokens"] < profile["context_window"]

            # Check rate limit availability
            provider = profile["provider"]
            available = self.rate_limiters[provider].acquire(
                task["estimated_tokens"]
            ) if provider in self.rate_limiters else True

            if fits_context:
                score = 0
                if capability_match:
                    score += 10
                if prefer_fast and profile["speed"] == "fast":
                    score += 5
                if prefer_cheap and profile["cost"] == "low":
                    score += 5
                if available:
                    score += 3

                candidates.append((model, score, available))

        # Sort by score descending
        candidates.sort(key=lambda x: (-x[1], -int(x[2])))

        if candidates:
            return candidates[0][0]

        return "gpt-4o-mini"  # Ultimate fallback

# Usage
router = ModelRouter()
model = router.select_model(
    messages=[{"role": "user", "content": "Classify this ticket as bug or feature: 'Login page crashes'"}],
    prefer_cheap=True,
)
print(f"Selected model: {model}")  # Likely gpt-4o-mini or claude-haiku
```

---

## Deployment Checklist

| Category | Item | Priority |
|----------|------|----------|
| **Reliability** | Multi-provider fallback configured | Critical |
| **Reliability** | Retry with exponential backoff | Critical |
| **Reliability** | Request timeout configured (30s) | Critical |
| **Reliability** | Circuit breaker for provider outages | High |
| **Performance** | Exact-match cache enabled | High |
| **Performance** | Semantic cache for repeated queries | Medium |
| **Performance** | Streaming responses for user-facing APIs | High |
| **Performance** | Async I/O for tool calls and retrieval | High |
| **Cost** | Cost tracking per request | Critical |
| **Cost** | Daily/monthly budget enforcement | Critical |
| **Cost** | Model routing (cheap model for simple tasks) | High |
| **Cost** | Prompt token optimization | Medium |
| **Security** | Input sanitization (prompt injection defense) | Critical |
| **Security** | Output filtering (PII, harmful content) | Critical |
| **Security** | Rate limiting per user/API key | Critical |
| **Security** | Secrets management (no hardcoded API keys) | Critical |
| **Observability** | Request/response logging | Critical |
| **Observability** | Latency, token, cost metrics | High |
| **Observability** | Error alerting | High |
| **Observability** | Trace visualization (LangSmith/Phoenix) | Medium |
| **Testing** | Unit tests for tool implementations | High |
| **Testing** | Integration tests with mock LLM | High |
| **Testing** | A/B testing framework for prompt changes | Medium |
| **Testing** | Red team evaluation before launch | High |

---

## Next Steps

This lesson concludes the production deployment section of the NLP and LLM course. For further study, revisit [12_Advanced_RAG.md](./12_Advanced_RAG.md) to apply these production patterns to RAG systems, or explore [15_Multi_Agent_Systems.md](./15_Multi_Agent_Systems.md) for deploying multi-agent workflows at scale.
