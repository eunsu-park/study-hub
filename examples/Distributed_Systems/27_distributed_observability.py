"""
Distributed Observability: Tracing, Logging, and Metrics

Implements a distributed tracing system with correlation IDs, structured
logging, and metrics collection. Demonstrates trace propagation across
service boundaries and how to diagnose latency issues.

Key concepts:
- Distributed tracing: spans, traces, and context propagation
- Correlation IDs for cross-service request tracking
- Structured logging with trace context
- RED metrics: Rate, Errors, Duration
- Trace sampling strategies

Usage:
    python 27_distributed_observability.py
"""

from __future__ import annotations

import random
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Distributed Tracing
# ---------------------------------------------------------------------------

@dataclass
class Span:
    """A span represents a unit of work in a distributed trace."""
    trace_id: str
    span_id: str
    parent_span_id: str | None
    service: str
    operation: str
    start_time: float
    end_time: float = 0.0
    status: str = "ok"
    tags: dict[str, str] = field(default_factory=dict)
    logs: list[tuple[float, str]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    def __repr__(self) -> str:
        return (f"Span({self.service}/{self.operation}, "
                f"{self.duration_ms:.1f}ms, {self.status})")


class TraceCollector:
    """Collects and stores spans from all services."""

    def __init__(self):
        self._spans: list[Span] = []
        self._by_trace: dict[str, list[Span]] = defaultdict(list)

    def record(self, span: Span) -> None:
        self._spans.append(span)
        self._by_trace[span.trace_id].append(span)

    def get_trace(self, trace_id: str) -> list[Span]:
        return sorted(self._by_trace.get(trace_id, []),
                      key=lambda s: s.start_time)

    def all_traces(self) -> dict[str, list[Span]]:
        return dict(self._by_trace)


class Tracer:
    """Creates and manages spans for a service."""

    def __init__(self, service: str, collector: TraceCollector,
                 seed: int = 42):
        self.service = service
        self.collector = collector
        self._rng = random.Random(seed)

    def start_span(self, operation: str, trace_id: str | None = None,
                   parent_span_id: str | None = None,
                   start_time: float = 0.0) -> Span:
        if trace_id is None:
            trace_id = uuid.UUID(int=self._rng.getrandbits(128)).hex[:16]
        span_id = uuid.UUID(int=self._rng.getrandbits(128)).hex[:8]

        return Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            service=self.service,
            operation=operation,
            start_time=start_time,
        )

    def finish_span(self, span: Span, end_time: float,
                    status: str = "ok") -> None:
        span.end_time = end_time
        span.status = status
        self.collector.record(span)


# ---------------------------------------------------------------------------
# Structured Logging
# ---------------------------------------------------------------------------

class StructuredLogger:
    """Structured logging with trace context."""

    def __init__(self, service: str):
        self.service = service
        self.entries: list[dict] = []

    def log(self, level: str, message: str, trace_id: str = "",
            span_id: str = "", extra: dict = None) -> dict:
        entry = {
            "level": level,
            "service": self.service,
            "message": message,
            "trace_id": trace_id,
            "span_id": span_id,
            **(extra or {}),
        }
        self.entries.append(entry)
        return entry


# ---------------------------------------------------------------------------
# Metrics Collection
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Collects RED metrics: Rate, Errors, Duration."""

    def __init__(self):
        self.request_count: dict[str, int] = defaultdict(int)
        self.error_count: dict[str, int] = defaultdict(int)
        self.durations: dict[str, list[float]] = defaultdict(list)

    def record_request(self, service: str, operation: str,
                       duration_ms: float, error: bool = False) -> None:
        key = f"{service}/{operation}"
        self.request_count[key] += 1
        self.durations[key].append(duration_ms)
        if error:
            self.error_count[key] += 1

    def summary(self, key: str) -> dict:
        durations = self.durations.get(key, [])
        if not durations:
            return {"count": 0}

        durations_sorted = sorted(durations)
        n = len(durations_sorted)

        return {
            "count": self.request_count.get(key, 0),
            "errors": self.error_count.get(key, 0),
            "error_rate": self.error_count.get(key, 0) / max(1, self.request_count.get(key, 0)),
            "p50_ms": durations_sorted[n // 2],
            "p95_ms": durations_sorted[int(n * 0.95)] if n >= 20 else durations_sorted[-1],
            "p99_ms": durations_sorted[int(n * 0.99)] if n >= 100 else durations_sorted[-1],
            "avg_ms": sum(durations) / n,
        }


# ---------------------------------------------------------------------------
# Simulated Microservice Architecture
# ---------------------------------------------------------------------------

def simulate_request(collector: TraceCollector, metrics: MetricsCollector,
                     rng: random.Random, request_id: int) -> str:
    """
    Simulate a request flowing through:
    API Gateway -> Auth Service -> Order Service -> Payment Service -> DB
    """
    t = request_id * 0.1

    # API Gateway
    gw_tracer = Tracer("api-gateway", collector, seed=request_id)
    gw_span = gw_tracer.start_span("handle_request", start_time=t)
    trace_id = gw_span.trace_id

    # Auth Service
    auth_tracer = Tracer("auth-service", collector, seed=request_id + 100)
    auth_span = auth_tracer.start_span("verify_token", trace_id,
                                        gw_span.span_id, t + 0.001)
    auth_latency = rng.uniform(1, 10)
    auth_error = rng.random() < 0.05  # 5% error rate
    auth_tracer.finish_span(auth_span, t + 0.001 + auth_latency / 1000,
                            "error" if auth_error else "ok")
    metrics.record_request("auth-service", "verify_token", auth_latency, auth_error)

    if auth_error:
        gw_tracer.finish_span(gw_span, t + 0.002 + auth_latency / 1000, "error")
        return trace_id

    # Order Service
    order_tracer = Tracer("order-service", collector, seed=request_id + 200)
    order_span = order_tracer.start_span("create_order", trace_id,
                                          gw_span.span_id,
                                          t + 0.002 + auth_latency / 1000)
    order_latency = rng.uniform(5, 50)
    order_tracer.finish_span(order_span,
                              order_span.start_time + order_latency / 1000)
    metrics.record_request("order-service", "create_order", order_latency)

    # Payment Service
    pay_tracer = Tracer("payment-service", collector, seed=request_id + 300)
    pay_span = pay_tracer.start_span("process_payment", trace_id,
                                      order_span.span_id,
                                      order_span.end_time)
    pay_latency = rng.uniform(10, 200)
    pay_error = rng.random() < 0.02  # 2% error rate
    pay_tracer.finish_span(pay_span, pay_span.start_time + pay_latency / 1000,
                           "error" if pay_error else "ok")
    metrics.record_request("payment-service", "process_payment",
                           pay_latency, pay_error)

    # Finish gateway span
    total_end = pay_span.end_time + 0.001
    status = "error" if pay_error else "ok"
    gw_tracer.finish_span(gw_span, total_end, status)
    total_ms = (total_end - t) * 1000
    metrics.record_request("api-gateway", "handle_request", total_ms,
                           pay_error or auth_error)

    return trace_id


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_distributed_tracing() -> None:
    """Demonstrate distributed tracing across services."""
    print("=" * 70)
    print("Distributed Tracing")
    print("=" * 70)

    collector = TraceCollector()
    metrics = MetricsCollector()
    rng = random.Random(42)

    # Simulate one request
    trace_id = simulate_request(collector, metrics, rng, request_id=0)
    trace = collector.get_trace(trace_id)

    print(f"\n  Trace {trace_id}:")
    print(f"  {'Service':<20} {'Operation':<20} {'Duration':>10} {'Status':>8}")
    print("  " + "-" * 62)

    for span in trace:
        indent = "  " if span.parent_span_id else ""
        print(f"  {indent}{span.service:<18} {span.operation:<20} "
              f"{span.duration_ms:>9.1f}ms {span.status:>8}")

    # Show trace tree
    print(f"\n  Trace waterfall:")
    root = trace[0]
    total_ms = root.duration_ms
    for span in trace:
        offset = (span.start_time - root.start_time) * 1000
        width = max(1, int(span.duration_ms / total_ms * 40))
        padding = int(offset / total_ms * 40)
        bar = " " * padding + "#" * width
        print(f"    {span.service:>18}: {bar} {span.duration_ms:.1f}ms")


def demo_metrics() -> None:
    """Demonstrate RED metrics collection."""
    print("\n" + "=" * 70)
    print("RED Metrics: Rate, Errors, Duration")
    print("=" * 70)

    collector = TraceCollector()
    metrics = MetricsCollector()
    rng = random.Random(42)

    # Simulate 100 requests
    for i in range(100):
        simulate_request(collector, metrics, rng, request_id=i)

    print(f"\n  100 simulated requests:\n")
    print(f"  {'Service/Op':<35} {'Count':>6} {'Errors':>7} "
          f"{'Err%':>6} {'p50':>8} {'p95':>8} {'p99':>8}")
    print("  " + "-" * 80)

    for key in sorted(metrics.request_count.keys()):
        s = metrics.summary(key)
        print(f"  {key:<35} {s['count']:>6} {s['errors']:>7} "
              f"{s['error_rate']*100:>5.1f}% "
              f"{s['p50_ms']:>7.1f}ms {s['p95_ms']:>7.1f}ms "
              f"{s['p99_ms']:>7.1f}ms")


def demo_structured_logging() -> None:
    """Demonstrate structured logging with trace context."""
    print("\n" + "=" * 70)
    print("Structured Logging with Trace Context")
    print("=" * 70)

    logger = StructuredLogger("order-service")

    trace_id = "abc123def456"
    span_id = "span7890"

    logger.log("INFO", "Received order request",
               trace_id, span_id, {"user_id": "user-42"})
    logger.log("INFO", "Validated order items",
               trace_id, span_id, {"item_count": 3})
    logger.log("WARN", "Inventory low for item SKU-100",
               trace_id, span_id, {"sku": "SKU-100", "remaining": 2})
    logger.log("INFO", "Order created successfully",
               trace_id, span_id, {"order_id": "ORD-789"})

    print(f"\n  Structured log entries (JSON-like):")
    for entry in logger.entries:
        print(f"    {entry}")

    print(f"\n  Benefits of structured logging:")
    print(f"    - Searchable by trace_id: find all logs for one request")
    print(f"    - Filterable by level, service, or any field")
    print(f"    - Correlate logs with traces and metrics")
    print(f"    - Machine-parseable (unlike free-text logs)")


def demo_sampling() -> None:
    """Demonstrate trace sampling strategies."""
    print("\n" + "=" * 70)
    print("Trace Sampling Strategies")
    print("=" * 70)

    print("""
  At high traffic, collecting every trace is too expensive.
  Sampling strategies:

  ┌───────────────────┬──────────────────────────────────────────┐
  │ Strategy          │ Description                              │
  ├───────────────────┼──────────────────────────────────────────┤
  │ Fixed rate        │ Sample 1% of all traces                  │
  │ Adaptive          │ Adjust rate based on traffic volume      │
  │ Priority          │ Always sample errors and slow requests   │
  │ Head-based        │ Decide at trace start (all-or-nothing)   │
  │ Tail-based        │ Decide after trace completes (expensive) │
  └───────────────────┴──────────────────────────────────────────┘
""")

    rng = random.Random(42)
    n_requests = 10000

    # Fixed 1% sampling
    fixed_sampled = sum(1 for _ in range(n_requests) if rng.random() < 0.01)

    # Priority: always sample errors (5% error rate) + 0.5% of normal
    rng2 = random.Random(42)
    priority_sampled = 0
    for _ in range(n_requests):
        is_error = rng2.random() < 0.05
        if is_error or rng2.random() < 0.005:
            priority_sampled += 1

    print(f"  {n_requests} requests:")
    print(f"    Fixed 1%:    {fixed_sampled} traces sampled")
    print(f"    Priority:    {priority_sampled} traces "
          f"(all errors + 0.5% normal)")
    print(f"\n  Priority sampling captures all interesting traces")
    print(f"  while keeping volume manageable")


if __name__ == "__main__":
    demo_distributed_tracing()
    demo_metrics()
    demo_structured_logging()
    demo_sampling()
    print("Done.")
