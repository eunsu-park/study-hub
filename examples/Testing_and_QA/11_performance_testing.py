#!/usr/bin/env python3
"""Example: Performance Testing

Demonstrates load testing patterns, benchmarking with pytest-benchmark,
and Locust-style load test concepts for web services.
Related lesson: 11_Performance_Testing.md
"""

# =============================================================================
# WHY PERFORMANCE TESTING?
#
# Functional tests verify correctness; performance tests verify speed and
# capacity. A feature that works but takes 30 seconds is broken in practice.
#
# Key types:
#   1. Benchmarking — measure execution time of individual functions
#   2. Load testing — simulate concurrent users hitting an API
#   3. Stress testing — push beyond expected load to find breaking points
#   4. Soak testing — sustained load over time to detect memory leaks
# =============================================================================

import pytest
import time
import statistics
from dataclasses import dataclass, field
from typing import Callable


# =============================================================================
# PRODUCTION CODE — THINGS WE WANT TO BENCHMARK
# =============================================================================

def fibonacci_recursive(n: int) -> int:
    """Naive recursive Fibonacci — O(2^n), intentionally slow."""
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci_iterative(n: int) -> int:
    """Iterative Fibonacci — O(n), fast."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def sort_with_builtin(data: list) -> list:
    """Python's Timsort — O(n log n) average."""
    return sorted(data)


def sort_with_bubble(data: list) -> list:
    """Bubble sort — O(n^2), intentionally slow for comparison."""
    result = list(data)
    n = len(result)
    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
    return result


# =============================================================================
# SIMPLE BENCHMARKING WITHOUT EXTERNAL LIBRARIES
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a simple benchmark run."""
    name: str
    iterations: int
    times: list = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.mean(self.times)

    @property
    def median(self) -> float:
        return statistics.median(self.times)

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.times) if len(self.times) > 1 else 0.0

    def __str__(self) -> str:
        return (
            f"{self.name}: mean={self.mean:.6f}s, "
            f"median={self.median:.6f}s, stdev={self.stdev:.6f}s "
            f"({self.iterations} iterations)"
        )


def simple_benchmark(func: Callable, *args, iterations: int = 100) -> BenchmarkResult:
    """Run a function multiple times and collect timing stats.
    This is a minimal benchmark — real projects use pytest-benchmark or pyperf."""
    result = BenchmarkResult(name=func.__name__, iterations=iterations)
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        elapsed = time.perf_counter() - start
        result.times.append(elapsed)
    return result


# =============================================================================
# TESTS — BENCHMARKING PATTERNS
# =============================================================================

class TestBenchmarkComparison:
    """Compare algorithm performance using simple timing assertions."""

    def test_iterative_faster_than_recursive(self):
        """Iterative Fibonacci should be orders of magnitude faster."""
        n = 25
        rec_bench = simple_benchmark(fibonacci_recursive, n, iterations=5)
        iter_bench = simple_benchmark(fibonacci_iterative, n, iterations=5)

        # Iterative should be at least 100x faster for n=25
        assert iter_bench.mean < rec_bench.mean / 100
        # Both must produce the same result
        assert fibonacci_recursive(n) == fibonacci_iterative(n)

    def test_builtin_sort_faster_than_bubble(self):
        """Timsort should dominate bubble sort on realistic data."""
        import random
        random.seed(42)
        data = [random.randint(0, 10000) for _ in range(1000)]

        builtin_bench = simple_benchmark(sort_with_builtin, data, iterations=20)
        bubble_bench = simple_benchmark(sort_with_bubble, data, iterations=5)

        assert builtin_bench.mean < bubble_bench.mean
        assert sort_with_builtin(data) == sort_with_bubble(data)

    def test_performance_regression_guard(self):
        """Set a time budget to catch regressions.
        In CI, this prevents merging code that makes things slower."""
        data = list(range(10000, 0, -1))  # worst case for many sorts
        bench = simple_benchmark(sort_with_builtin, data, iterations=50)

        # Timsort on 10k items should complete well under 10ms
        assert bench.mean < 0.01, f"Sort too slow: {bench.mean:.4f}s"


# =============================================================================
# LOCUST-STYLE LOAD TEST SIMULATION
# =============================================================================
# In real projects, you'd use Locust (locust.io) which provides a web UI and
# distributed load generation. Here we simulate the pattern.

@dataclass
class RequestResult:
    """Simulated HTTP request result."""
    status_code: int
    response_time_ms: float
    success: bool


class FakeHttpClient:
    """Simulates an HTTP service for demonstration purposes."""

    def __init__(self, base_latency_ms: float = 5.0):
        self.base_latency = base_latency_ms / 1000
        self.request_count = 0

    def get(self, path: str) -> RequestResult:
        """Simulate a GET request with realistic latency."""
        self.request_count += 1
        start = time.perf_counter()
        # Simulate work — latency increases under load
        time.sleep(self.base_latency)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return RequestResult(
            status_code=200,
            response_time_ms=elapsed_ms,
            success=True,
        )


class TestLoadTestPatterns:
    """Demonstrate load testing concepts without external dependencies."""

    def test_response_time_under_threshold(self):
        """Verify p95 response time stays under SLA threshold."""
        client = FakeHttpClient(base_latency_ms=2.0)
        results = [client.get("/api/health") for _ in range(50)]

        times = [r.response_time_ms for r in results]
        p95 = sorted(times)[int(len(times) * 0.95)]

        # SLA: p95 response time < 50ms
        assert p95 < 50, f"p95 latency {p95:.1f}ms exceeds 50ms SLA"
        assert all(r.success for r in results)

    def test_throughput_minimum(self):
        """Verify the service handles minimum required throughput."""
        client = FakeHttpClient(base_latency_ms=1.0)
        start = time.perf_counter()
        results = [client.get("/api/data") for _ in range(20)]
        duration = time.perf_counter() - start

        throughput = len(results) / duration  # requests per second
        assert throughput > 5, f"Throughput {throughput:.1f} rps below minimum"


# =============================================================================
# LOCUST FILE EXAMPLE (REFERENCE)
# =============================================================================

LOCUSTFILE_EXAMPLE = """
# === locustfile.py ===
# Run with: locust -f locustfile.py --host=http://localhost:8000
# Then open http://localhost:8089 for the web UI.

from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    # Wait 1-3 seconds between tasks (simulates real user think time)
    wait_time = between(1, 3)

    @task(3)  # Weight 3 — this runs 3x more often than weight-1 tasks
    def view_homepage(self):
        self.client.get("/")

    @task(1)
    def view_api(self):
        self.client.get("/api/data")

    def on_start(self):
        # Called once per simulated user at startup
        self.client.post("/login", json={"user": "test", "pass": "test"})
"""

# =============================================================================
# RUNNING THIS FILE
# =============================================================================
# Basic run:
#   pytest 11_performance_testing.py -v
#
# With pytest-benchmark (install: pip install pytest-benchmark):
#   pytest 11_performance_testing.py -v --benchmark-only
#
# Locust (install: pip install locust):
#   locust -f locustfile.py --host=http://localhost:8000

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
