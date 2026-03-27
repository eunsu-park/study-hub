"""
Exercises for Lesson 24: Production LLM Patterns
Topic: NLP_and_LLM

Practice problems for caching, cost tracking, rate limiting, and observability.
"""

import hashlib
import json
import math
import time
import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Callable
from functools import wraps


# === Exercise 1: Multi-Tier Cache ===
# Problem: Implement a two-tier cache (exact match + fuzzy match)
# with statistics tracking and TTL-based eviction.

def exercise_1():
    """Build a multi-tier caching system."""
    print("=" * 60)
    print("Exercise 1: Multi-Tier Cache")
    print("=" * 60)

    class TieredCache:
        def __init__(self, exact_ttl: float = 60, fuzzy_threshold: float = 0.8,
                     max_size: int = 100):
            self.exact_ttl = exact_ttl
            self.fuzzy_threshold = fuzzy_threshold
            self.max_size = max_size
            self._exact: dict[str, tuple[str, float]] = {}
            self._fuzzy: list[tuple[str, str, float]] = []  # (query, response, timestamp)
            self.stats = {"exact_hits": 0, "fuzzy_hits": 0, "misses": 0}

        def _hash_key(self, messages: list[dict], model: str) -> str:
            payload = json.dumps({"messages": messages, "model": model}, sort_keys=True)
            return hashlib.sha256(payload.encode()).hexdigest()[:16]

        def _similarity(self, a: str, b: str) -> float:
            """Simple Jaccard similarity between two strings."""
            words_a = set(a.lower().split())
            words_b = set(b.lower().split())
            if not words_a or not words_b:
                return 0.0
            return len(words_a & words_b) / len(words_a | words_b)

        # TODO: Implement get with exact then fuzzy fallback
        def get(self, messages: list[dict], model: str) -> tuple[str | None, str]:
            now = time.time()

            # Tier 1: Exact match
            key = self._hash_key(messages, model)
            if key in self._exact:
                response, created = self._exact[key]
                if now - created < self.exact_ttl:
                    self.stats["exact_hits"] += 1
                    return response, "exact"
                del self._exact[key]

            # Tier 2: Fuzzy match
            query = messages[-1]["content"] if messages else ""
            for stored_query, response, created in self._fuzzy:
                if now - created < self.exact_ttl:
                    sim = self._similarity(query, stored_query)
                    if sim >= self.fuzzy_threshold:
                        self.stats["fuzzy_hits"] += 1
                        return response, "fuzzy"

            self.stats["misses"] += 1
            return None, "miss"

        # TODO: Implement put for both tiers
        def put(self, messages: list[dict], model: str, response: str):
            key = self._hash_key(messages, model)
            self._exact[key] = (response, time.time())

            query = messages[-1]["content"] if messages else ""
            self._fuzzy.append((query, response, time.time()))

            # Evict if over max size
            if len(self._fuzzy) > self.max_size:
                self._fuzzy = self._fuzzy[-self.max_size:]

    cache = TieredCache(fuzzy_threshold=0.6)

    # Store a response
    msgs1 = [{"role": "user", "content": "What is machine learning?"}]
    cache.put(msgs1, "gpt-4o", "ML is a subset of AI that learns from data.")

    # Exact hit
    result, tier = cache.get(msgs1, "gpt-4o")
    print(f"Exact match:  tier={tier}, result='{result[:40]}...'")

    # Fuzzy hit (similar query)
    msgs2 = [{"role": "user", "content": "What is machine learning about?"}]
    result, tier = cache.get(msgs2, "gpt-4o")
    print(f"Fuzzy match:  tier={tier}, result={repr(result)[:50] if result else 'None'}")

    # Miss (different query)
    msgs3 = [{"role": "user", "content": "Explain quantum computing"}]
    result, tier = cache.get(msgs3, "gpt-4o")
    print(f"Miss:         tier={tier}")

    print(f"Stats: {cache.stats}")


# === Exercise 2: Cost Budget Manager ===
# Problem: Build a cost budget manager with per-model tracking,
# alerting thresholds, and automatic model downgrade.

def exercise_2():
    """Build a cost budget manager."""
    print("\n" + "=" * 60)
    print("Exercise 2: Cost Budget Manager")
    print("=" * 60)

    class BudgetManager:
        PRICING = {
            "gpt-4o": (2.50, 10.00),       # per 1M tokens
            "gpt-4o-mini": (0.15, 0.60),
        }
        DOWNGRADE_MAP = {"gpt-4o": "gpt-4o-mini"}

        def __init__(self, daily_budget: float = 10.0, alert_threshold: float = 0.8):
            self.daily_budget = daily_budget
            self.alert_threshold = alert_threshold
            self._spend: dict[str, float] = defaultdict(float)  # model -> spend
            self._total: float = 0.0
            self.alerts: list[str] = []

        # TODO: Record usage and check budget
        def record(self, model: str, input_tokens: int, output_tokens: int) -> dict:
            in_rate, out_rate = self.PRICING.get(model, (5.0, 15.0))
            cost = input_tokens * in_rate / 1e6 + output_tokens * out_rate / 1e6

            self._spend[model] += cost
            self._total += cost

            # Check alert threshold
            if self._total >= self.daily_budget * self.alert_threshold:
                self.alerts.append(f"ALERT: {self._total / self.daily_budget:.0%} of daily budget used")

            return {"cost": round(cost, 6), "total": round(self._total, 6),
                    "budget_remaining": round(self.daily_budget - self._total, 6)}

        # TODO: Select model with automatic downgrade when budget is low
        def select_model(self, preferred: str) -> str:
            remaining_pct = (self.daily_budget - self._total) / self.daily_budget
            if remaining_pct < 0.2 and preferred in self.DOWNGRADE_MAP:
                downgraded = self.DOWNGRADE_MAP[preferred]
                self.alerts.append(f"DOWNGRADE: {preferred} -> {downgraded} (budget low)")
                return downgraded
            return preferred

        def summary(self) -> dict:
            return {
                "total_spend": round(self._total, 6),
                "by_model": {k: round(v, 6) for k, v in self._spend.items()},
                "budget_used_pct": round(self._total / self.daily_budget * 100, 1),
                "alerts": self.alerts[-3:],
            }

    budget = BudgetManager(daily_budget=0.01)  # Very low budget for demo

    # Simulate heavy usage
    for i in range(5):
        model = budget.select_model("gpt-4o")
        result = budget.record(model, 1000, 500)
        print(f"  Call {i+1}: model={model}, cost=${result['cost']:.6f}, "
              f"remaining=${result['budget_remaining']:.6f}")

    print(f"\nSummary: {json.dumps(budget.summary(), indent=2)}")


# === Exercise 3: Adaptive Rate Limiter ===
# Problem: Build a rate limiter that adapts based on error rates
# (backs off when errors increase).

def exercise_3():
    """Build an adaptive rate limiter."""
    print("\n" + "=" * 60)
    print("Exercise 3: Adaptive Rate Limiter")
    print("=" * 60)

    class AdaptiveRateLimiter:
        def __init__(self, base_rpm: int = 60, min_rpm: int = 10):
            self.base_rpm = base_rpm
            self.min_rpm = min_rpm
            self.current_rpm = base_rpm
            self.tokens = float(base_rpm)
            self.last_refill = time.time()
            self.error_count = 0
            self.success_count = 0
            self._lock = threading.Lock()

        def _refill(self):
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.current_rpm, self.tokens + elapsed * (self.current_rpm / 60))
            self.last_refill = now

        # TODO: Acquire with adaptive rate
        def acquire(self) -> bool:
            with self._lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
                return False

        # TODO: Report success/error to adjust rate
        def report_success(self):
            self.success_count += 1
            # Gradually increase rate back to base
            if self.success_count % 10 == 0 and self.current_rpm < self.base_rpm:
                self.current_rpm = min(self.base_rpm, self.current_rpm + 5)

        def report_error(self):
            self.error_count += 1
            # Reduce rate on errors (exponential backoff style)
            self.current_rpm = max(self.min_rpm, int(self.current_rpm * 0.7))

        def stats(self) -> dict:
            total = self.success_count + self.error_count
            return {
                "current_rpm": self.current_rpm,
                "base_rpm": self.base_rpm,
                "success_count": self.success_count,
                "error_count": self.error_count,
                "error_rate": round(self.error_count / total, 3) if total > 0 else 0,
            }

    limiter = AdaptiveRateLimiter(base_rpm=60, min_rpm=10)

    # Simulate: some successes, then errors, then recovery
    for i in range(10):
        limiter.report_success()
    print(f"After 10 successes: {limiter.stats()}")

    for i in range(5):
        limiter.report_error()
    print(f"After 5 errors: {limiter.stats()}")

    for i in range(20):
        limiter.report_success()
    print(f"After 20 more successes: {limiter.stats()}")


# === Exercise 4: Observability Dashboard ===
# Problem: Build a comprehensive observability system that tracks
# latency, tokens, costs, and error rates per model.

def exercise_4():
    """Build an LLM observability dashboard."""
    print("\n" + "=" * 60)
    print("Exercise 4: Observability Dashboard")
    print("=" * 60)

    class LLMDashboard:
        def __init__(self):
            self.records: list[dict] = []

        # TODO: Record a request with all relevant metrics
        def record(self, model: str, latency_ms: float, input_tokens: int,
                   output_tokens: int, success: bool, cached: bool = False):
            pricing = {"gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60)}
            in_rate, out_rate = pricing.get(model, (5.0, 15.0))
            cost = input_tokens * in_rate / 1e6 + output_tokens * out_rate / 1e6

            self.records.append({
                "model": model, "latency_ms": latency_ms,
                "input_tokens": input_tokens, "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost": cost, "success": success, "cached": cached,
                "timestamp": time.time(),
            })

        # TODO: Generate comprehensive dashboard summary
        def summary(self) -> dict:
            if not self.records:
                return {}

            by_model = defaultdict(list)
            for r in self.records:
                by_model[r["model"]].append(r)

            dashboard = {"total_requests": len(self.records)}

            # Overall metrics
            latencies = [r["latency_ms"] for r in self.records]
            dashboard["overall"] = {
                "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
                "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 1),
                "p99_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 1),
                "total_cost": round(sum(r["cost"] for r in self.records), 6),
                "total_tokens": sum(r["total_tokens"] for r in self.records),
                "success_rate": round(sum(1 for r in self.records if r["success"]) / len(self.records), 3),
                "cache_rate": round(sum(1 for r in self.records if r["cached"]) / len(self.records), 3),
            }

            # Per-model metrics
            dashboard["by_model"] = {}
            for model, records in by_model.items():
                lats = [r["latency_ms"] for r in records]
                dashboard["by_model"][model] = {
                    "count": len(records),
                    "avg_latency_ms": round(sum(lats) / len(lats), 1),
                    "total_cost": round(sum(r["cost"] for r in records), 6),
                    "success_rate": round(sum(1 for r in records if r["success"]) / len(records), 3),
                }

            return dashboard

    import random
    random.seed(42)
    dashboard = LLMDashboard()

    # Simulate 50 requests
    for i in range(50):
        model = random.choice(["gpt-4o", "gpt-4o-mini", "gpt-4o-mini"])
        latency = random.uniform(50, 500) if model == "gpt-4o" else random.uniform(20, 200)
        input_tok = random.randint(100, 2000)
        output_tok = random.randint(50, 1000)
        success = random.random() > 0.05  # 95% success rate
        cached = random.random() < 0.2  # 20% cache rate

        if cached:
            latency = random.uniform(1, 10)

        dashboard.record(model, latency, input_tok, output_tok, success, cached)

    result = dashboard.summary()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
