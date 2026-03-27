"""
24. Production LLM Patterns Example

Caching, cost tracking, rate limiting, A/B testing, and observability
"""

import hashlib
import json
import time
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Callable

print("=" * 60)
print("Production LLM Patterns")
print("=" * 60)


# ============================================
# 1. Request/Response Models
# ============================================
print("\n[1] Standard Request/Response")
print("-" * 40)


@dataclass
class LLMRequest:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    messages: list[dict] = field(default_factory=list)
    model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 2048


@dataclass
class LLMResponse:
    request_id: str
    content: str
    model: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    cached: bool = False

    @property
    def estimated_cost(self) -> float:
        pricing = {"gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60)}
        in_rate, out_rate = pricing.get(self.model, (5.0, 15.0))
        return self.tokens_input * in_rate / 1e6 + self.tokens_output * out_rate / 1e6


req = LLMRequest(messages=[{"role": "user", "content": "Hello"}])
resp = LLMResponse(req.request_id, "Hi there!", "gpt-4o", 10, 5, 250.0)
print(f"Request ID: {req.request_id}")
print(f"Response: {resp.content}, Cost: ${resp.estimated_cost:.6f}")


# ============================================
# 2. Exact Match Cache
# ============================================
print("\n[2] Exact Match Cache")
print("-" * 40)


class ExactMatchCache:
    """In-memory exact match cache with TTL."""

    def __init__(self, ttl_seconds: float = 3600):
        self.ttl = ttl_seconds
        self._store: dict[str, tuple[str, float]] = {}
        self._hits = 0
        self._misses = 0

    def _make_key(self, messages: list[dict], model: str, temperature: float) -> str:
        payload = json.dumps({"messages": messages, "model": model, "temperature": temperature}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def get(self, messages: list[dict], model: str, temperature: float) -> str | None:
        key = self._make_key(messages, model, temperature)
        if key in self._store:
            response, created = self._store[key]
            if time.time() - created < self.ttl:
                self._hits += 1
                return response
            del self._store[key]
        self._misses += 1
        return None

    def put(self, messages: list[dict], model: str, temperature: float, response: str):
        key = self._make_key(messages, model, temperature)
        self._store[key] = (response, time.time())

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {"hits": self._hits, "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0,
                "size": len(self._store)}


cache = ExactMatchCache(ttl_seconds=60)

msgs = [{"role": "user", "content": "What is Python?"}]
cache.put(msgs, "gpt-4o", 0.3, "Python is a programming language.")

# Cache hit
result = cache.get(msgs, "gpt-4o", 0.3)
print(f"Cache hit: {result}")

# Cache miss (different query)
result = cache.get([{"role": "user", "content": "What is Java?"}], "gpt-4o", 0.3)
print(f"Cache miss: {result}")

# Cache miss (different temp)
result = cache.get(msgs, "gpt-4o", 0.7)
print(f"Cache miss (diff temp): {result}")

print(f"Stats: {cache.stats()}")


# ============================================
# 3. Cost Tracker
# ============================================
print("\n[3] Cost Tracking")
print("-" * 40)


class CostTracker:
    """Track LLM spending and enforce budgets."""

    PRICING = {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
    }

    def __init__(self, daily_budget: float = 50.0):
        self.daily_budget = daily_budget
        self._daily_spend: dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        in_rate, out_rate = self.PRICING.get(model, (5.0, 15.0))
        cost = input_tokens * in_rate / 1e6 + output_tokens * out_rate / 1e6
        today = datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            self._daily_spend[today] += cost
        return cost

    def can_proceed(self) -> bool:
        today = datetime.now().strftime("%Y-%m-%d")
        return self._daily_spend.get(today, 0) < self.daily_budget

    def summary(self) -> dict:
        today = datetime.now().strftime("%Y-%m-%d")
        spend = self._daily_spend.get(today, 0)
        return {"daily_spend": round(spend, 6), "budget": self.daily_budget,
                "remaining": round(self.daily_budget - spend, 6)}


tracker = CostTracker(daily_budget=10.0)

# Simulate usage
costs = [
    tracker.record("gpt-4o", 1000, 500),
    tracker.record("gpt-4o-mini", 2000, 1000),
    tracker.record("gpt-4o", 5000, 2000),
]
for i, c in enumerate(costs):
    print(f"  Call {i+1} cost: ${c:.6f}")

print(f"Summary: {tracker.summary()}")
print(f"Can proceed: {tracker.can_proceed()}")


# ============================================
# 4. Rate Limiter
# ============================================
print("\n[4] Token Bucket Rate Limiter")
print("-" * 40)


class TokenBucketRateLimiter:
    """Rate limiter using token bucket algorithm."""

    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self.tokens = float(requests_per_minute)
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.rpm, self.tokens + elapsed * (self.rpm / 60))
        self.last_refill = now

    def acquire(self) -> bool:
        with self._lock:
            self._refill()
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


limiter = TokenBucketRateLimiter(requests_per_minute=10)

acquired = sum(1 for _ in range(15) if limiter.acquire())
print(f"Acquired {acquired}/15 requests (limit: 10 RPM)")


# ============================================
# 5. A/B Testing
# ============================================
print("\n[5] A/B Testing")
print("-" * 40)


@dataclass
class Variant:
    name: str
    model: str
    temperature: float
    weight: float = 0.5


class ABTest:
    """Simple A/B test manager."""

    def __init__(self, name: str, variants: list[Variant]):
        self.name = name
        self.variants = variants
        self.results: dict[str, list[dict]] = defaultdict(list)

    def assign(self, user_id: str) -> Variant:
        hash_val = int(hashlib.md5(f"{self.name}:{user_id}".encode()).hexdigest(), 16)
        threshold = (hash_val % 1000) / 1000.0
        cumulative = 0.0
        for v in self.variants:
            cumulative += v.weight
            if threshold < cumulative:
                return v
        return self.variants[-1]

    def record(self, variant_name: str, latency_ms: float, rating: float):
        self.results[variant_name].append({"latency_ms": latency_ms, "rating": rating})

    def summary(self) -> dict:
        summary = {}
        for name, results in self.results.items():
            latencies = [r["latency_ms"] for r in results]
            ratings = [r["rating"] for r in results]
            summary[name] = {
                "count": len(results),
                "avg_latency": round(sum(latencies) / len(latencies), 1) if latencies else 0,
                "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else 0,
            }
        return summary


ab = ABTest("prompt-v2", [
    Variant("concise", "gpt-4o-mini", 0.2, 0.5),
    Variant("detailed", "gpt-4o", 0.3, 0.5),
])

# Simulate 20 users
import random
random.seed(42)
for i in range(20):
    user_id = f"user_{i}"
    variant = ab.assign(user_id)
    ab.record(variant.name, random.uniform(100, 500), random.uniform(3, 5))

print(f"A/B test: {ab.name}")
for name, stats in ab.summary().items():
    print(f"  {name}: {stats}")


# ============================================
# 6. Observability
# ============================================
print("\n[6] LLM Observability")
print("-" * 40)


class LLMObserver:
    """Lightweight observability for LLM apps."""

    def __init__(self):
        self.metrics: dict[str, list] = defaultdict(list)

    def trace(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start) * 1000
                self.metrics["latency_ms"].append(duration)
                self.metrics["success"].append(1)
                return result
            except Exception as e:
                duration = (time.time() - start) * 1000
                self.metrics["latency_ms"].append(duration)
                self.metrics["success"].append(0)
                self.metrics["errors"].append(str(e))
                raise
        return wrapper

    def dashboard(self) -> dict:
        latencies = self.metrics.get("latency_ms", [])
        successes = self.metrics.get("success", [])
        return {
            "total_requests": len(successes),
            "success_rate": round(sum(successes) / len(successes), 3) if successes else 0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 2) if latencies else 0,
            "errors": len(self.metrics.get("errors", [])),
        }


observer = LLMObserver()


@observer.trace
def simulated_llm_call(query: str) -> str:
    time.sleep(random.uniform(0.001, 0.01))
    if "error" in query.lower():
        raise RuntimeError("Simulated error")
    return f"Answer to: {query}"


for q in ["What is Python?", "Explain Docker", "How does React work?", "Tell me about Rust"]:
    simulated_llm_call(q)

try:
    simulated_llm_call("This will error out")
except RuntimeError:
    pass

print(f"Dashboard: {json.dumps(observer.dashboard(), indent=2)}")


# ============================================
# 7. Model Router
# ============================================
print("\n[7] Model Router")
print("-" * 40)


class ModelRouter:
    """Route to optimal model based on task."""

    SIMPLE_KEYWORDS = {"classify", "categorize", "label", "yes or no", "true or false"}
    COMPLEX_KEYWORDS = {"analyze", "compare", "design", "debug", "explain in detail", "architect"}

    def select(self, query: str, prefer_cheap: bool = False) -> str:
        query_lower = query.lower()
        words = len(query.split())

        if prefer_cheap or words < 20:
            if any(kw in query_lower for kw in self.SIMPLE_KEYWORDS):
                return "gpt-4o-mini"

        if any(kw in query_lower for kw in self.COMPLEX_KEYWORDS):
            return "gpt-4o"

        return "gpt-4o-mini" if prefer_cheap else "gpt-4o"


router = ModelRouter()

queries = [
    ("Classify this: bug or feature?", True),
    ("Analyze the performance of our ML pipeline and suggest optimizations", False),
    ("What is Python?", True),
    ("Design a distributed system for real-time analytics", False),
]

for query, cheap in queries:
    model = router.select(query, prefer_cheap=cheap)
    print(f"  [{model:20s}] {query[:60]}...")

print("\n" + "=" * 60)
print("Production LLM Patterns example complete!")
print("=" * 60)
