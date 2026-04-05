# Lesson 27: Distributed Observability

[Overview](./00_Overview.md) | [Previous: Distributed Testing](./26_Distributed_Testing.md) | [Next: Capstone — Distributed KV Store](./28_Capstone_Distributed_KV.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement distributed tracing with context propagation across service boundaries
2. Design correlation ID schemes for request tracking through microservices
3. Build structured distributed logging with centralized aggregation
4. Implement metric collection and anomaly detection for distributed systems
5. Debug production distributed systems using traces, logs, and metrics together

---

## Table of Contents

1. [Observability Fundamentals](#1-observability-fundamentals)
2. [Distributed Tracing](#2-distributed-tracing)
3. [Context Propagation](#3-context-propagation)
4. [Correlation IDs](#4-correlation-ids)
5. [Distributed Logging](#5-distributed-logging)
6. [Metrics Collection](#6-metrics-collection)
7. [Anomaly Detection](#7-anomaly-detection)
8. [Debugging Distributed Systems](#8-debugging-distributed-systems)
9. [Real-World Observability Stacks](#9-real-world-observability-stacks)
10. [Summary and Key Takeaways](#10-summary-and-key-takeaways)
11. [Practice Problems](#11-practice-problems)
12. [References](#12-references)

---

## 1. Observability Fundamentals

### 1.1 The Three Pillars

```python
import time
import json
import uuid
import random
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


def explain_three_pillars():
    """Explain the three pillars of observability."""
    print("=== Three Pillars of Observability ===\n")

    pillars = {
        "Traces": {
            "what": "End-to-end request path across services",
            "answers": "Where did the request go? What was slow?",
            "tools": "Jaeger, Zipkin, AWS X-Ray, OpenTelemetry",
        },
        "Logs": {
            "what": "Timestamped text records from each service",
            "answers": "What happened? What was the error message?",
            "tools": "ELK Stack, Loki, CloudWatch Logs",
        },
        "Metrics": {
            "what": "Numeric measurements over time",
            "answers": "How many? How fast? What percentage?",
            "tools": "Prometheus, Datadog, CloudWatch Metrics",
        },
    }

    for name, info in pillars.items():
        print(f"  {name}:")
        for k, v in info.items():
            print(f"    {k}: {v}")
        print()


explain_three_pillars()
```

---

## 2. Distributed Tracing

### 2.1 Trace and Span Model

```python
@dataclass
class Span:
    """
    A single unit of work in a distributed trace.

    A span represents an operation within a service:
    - An HTTP request handler
    - A database query
    - A message processing step
    """
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    service_name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    duration_ms: float = 0.0
    status: str = "ok"  # ok, error
    tags: dict = field(default_factory=dict)
    logs: list = field(default_factory=list)

    def finish(self):
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000

    def log(self, message: str, fields: dict = None):
        self.logs.append({
            "timestamp": time.time(),
            "message": message,
            "fields": fields or {},
        })

    def set_tag(self, key: str, value: Any):
        self.tags[key] = value


class Tracer:
    """
    Distributed tracing system.

    Creates and manages spans, propagates trace context
    across service boundaries, and collects completed spans.
    """

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.active_spans: Dict[str, Span] = {}
        self.completed_spans: list[Span] = []

    def start_span(self, operation_name: str, parent: Optional[Span] = None,
                   trace_id: Optional[str] = None) -> Span:
        """Start a new span."""
        span = Span(
            trace_id=trace_id or parent.trace_id if parent else str(uuid.uuid4())[:16],
            span_id=str(uuid.uuid4())[:16],
            parent_span_id=parent.span_id if parent else None,
            operation_name=operation_name,
            service_name=self.service_name,
        )
        self.active_spans[span.span_id] = span
        return span

    def finish_span(self, span: Span):
        """Finish and record a span."""
        span.finish()
        self.active_spans.pop(span.span_id, None)
        self.completed_spans.append(span)

    def inject_context(self, span: Span) -> dict:
        """
        Inject trace context into a carrier (e.g., HTTP headers).

        This allows the trace to continue across service boundaries.
        """
        return {
            "x-trace-id": span.trace_id,
            "x-span-id": span.span_id,
            "x-parent-span-id": span.parent_span_id or "",
        }

    def extract_context(self, carrier: dict) -> Tuple[str, str]:
        """Extract trace context from a carrier."""
        trace_id = carrier.get("x-trace-id", "")
        parent_span_id = carrier.get("x-span-id", "")
        return trace_id, parent_span_id


class TraceCollector:
    """Centralized trace collector that assembles complete traces."""

    def __init__(self):
        self.spans: list[Span] = []
        self.traces: Dict[str, list[Span]] = defaultdict(list)

    def collect(self, span: Span):
        self.spans.append(span)
        self.traces[span.trace_id].append(span)

    def get_trace(self, trace_id: str) -> list[Span]:
        spans = self.traces.get(trace_id, [])
        return sorted(spans, key=lambda s: s.start_time)

    def render_trace(self, trace_id: str) -> str:
        """Render a trace as a visual tree."""
        spans = self.get_trace(trace_id)
        if not spans:
            return "No spans found"

        lines = [f"Trace: {trace_id}"]
        root_spans = [s for s in spans if s.parent_span_id is None]
        children = defaultdict(list)
        for s in spans:
            if s.parent_span_id:
                children[s.parent_span_id].append(s)

        def render_span(span, indent=0):
            prefix = "  " * indent + ("└─ " if indent > 0 else "")
            status = "✓" if span.status == "ok" else "✗"
            lines.append(
                f"{prefix}[{status}] {span.service_name}/{span.operation_name} "
                f"({span.duration_ms:.1f}ms)"
            )
            for child in sorted(children.get(span.span_id, []),
                              key=lambda s: s.start_time):
                render_span(child, indent + 1)

        for root in root_spans:
            render_span(root)

        return "\n".join(lines)


def demonstrate_distributed_tracing():
    """Demonstrate distributed tracing across services."""
    print("=== Distributed Tracing ===\n")

    collector = TraceCollector()

    # Service A: API Gateway
    tracer_a = Tracer("api-gateway")
    root_span = tracer_a.start_span("POST /orders")
    root_span.set_tag("http.method", "POST")
    root_span.set_tag("http.url", "/orders")

    # Service A calls Service B
    headers = tracer_a.inject_context(root_span)
    time.sleep(0.01)

    # Service B: Order Service
    tracer_b = Tracer("order-service")
    trace_id, parent_id = tracer_b.extract_context(headers)
    span_b = tracer_b.start_span("createOrder", trace_id=trace_id)
    span_b.parent_span_id = parent_id

    # Service B calls Service C (database)
    span_db = tracer_b.start_span("INSERT orders", parent=span_b)
    span_db.set_tag("db.type", "postgresql")
    time.sleep(0.005)
    tracer_b.finish_span(span_db)

    # Service B calls Service D (payment)
    headers_d = tracer_b.inject_context(span_b)
    tracer_d = Tracer("payment-service")
    trace_id_d, parent_id_d = tracer_d.extract_context(headers_d)
    span_d = tracer_d.start_span("chargeCard", trace_id=trace_id_d)
    span_d.parent_span_id = parent_id_d
    time.sleep(0.015)
    tracer_d.finish_span(span_d)

    time.sleep(0.005)
    tracer_b.finish_span(span_b)

    time.sleep(0.002)
    tracer_a.finish_span(root_span)

    # Collect all spans
    for tracer in [tracer_a, tracer_b, tracer_d]:
        for span in tracer.completed_spans:
            collector.collect(span)

    # Render trace
    print(collector.render_trace(root_span.trace_id))
    print(f"\nTotal spans: {len(collector.spans)}")
    total_time = root_span.duration_ms
    print(f"Total time: {total_time:.1f}ms")


demonstrate_distributed_tracing()
```

---

## 3. Context Propagation

### 3.1 W3C Trace Context

```python
class W3CTraceContext:
    """
    W3C Trace Context propagation format.

    traceparent: {version}-{trace-id}-{parent-id}-{trace-flags}
    tracestate: vendor-specific key-value pairs
    """

    VERSION = "00"

    @staticmethod
    def create_traceparent(trace_id: str, span_id: str,
                           sampled: bool = True) -> str:
        flags = "01" if sampled else "00"
        return f"{W3CTraceContext.VERSION}-{trace_id}-{span_id}-{flags}"

    @staticmethod
    def parse_traceparent(header: str) -> Optional[dict]:
        parts = header.split("-")
        if len(parts) != 4:
            return None
        return {
            "version": parts[0],
            "trace_id": parts[1],
            "parent_id": parts[2],
            "sampled": parts[3] == "01",
        }

    @staticmethod
    def create_tracestate(entries: dict) -> str:
        return ",".join(f"{k}={v}" for k, v in entries.items())


def demonstrate_context_propagation():
    """Demonstrate W3C trace context propagation."""
    print("=== Context Propagation ===\n")

    trace_id = uuid.uuid4().hex[:32]
    span_id = uuid.uuid4().hex[:16]

    # Create traceparent header
    traceparent = W3CTraceContext.create_traceparent(trace_id, span_id)
    print(f"traceparent: {traceparent}")

    # Parse it back
    parsed = W3CTraceContext.parse_traceparent(traceparent)
    print(f"Parsed: {json.dumps(parsed, indent=2)}")

    # Tracestate for vendor-specific data
    tracestate = W3CTraceContext.create_tracestate({
        "vendor1": "value1",
        "vendor2": "value2",
    })
    print(f"tracestate: {tracestate}")


demonstrate_context_propagation()
```

---

## 4. Correlation IDs

### 4.1 Request Correlation

```python
class CorrelationContext:
    """
    Correlation context for tracking requests through a distributed system.

    Unlike trace context (which tracks spans), correlation context
    carries business-level identifiers like request ID, user ID,
    and session ID through the entire request chain.
    """

    def __init__(self):
        self.request_id: str = str(uuid.uuid4())
        self.correlation_id: str = str(uuid.uuid4())[:8]
        self.user_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.baggage: dict = {}

    def to_headers(self) -> dict:
        headers = {
            "x-request-id": self.request_id,
            "x-correlation-id": self.correlation_id,
        }
        if self.user_id:
            headers["x-user-id"] = self.user_id
        for k, v in self.baggage.items():
            headers[f"x-baggage-{k}"] = v
        return headers

    @classmethod
    def from_headers(cls, headers: dict) -> 'CorrelationContext':
        ctx = cls()
        ctx.request_id = headers.get("x-request-id", ctx.request_id)
        ctx.correlation_id = headers.get("x-correlation-id", ctx.correlation_id)
        ctx.user_id = headers.get("x-user-id")
        for k, v in headers.items():
            if k.startswith("x-baggage-"):
                ctx.baggage[k[10:]] = v
        return ctx


def demonstrate_correlation_ids():
    """Demonstrate correlation ID propagation."""
    print("=== Correlation IDs ===\n")

    # Client creates initial context
    ctx = CorrelationContext()
    ctx.user_id = "user-123"
    ctx.baggage["tenant"] = "acme-corp"

    headers = ctx.to_headers()
    print("Request headers:")
    for k, v in headers.items():
        print(f"  {k}: {v}")

    # Downstream service extracts context
    downstream_ctx = CorrelationContext.from_headers(headers)
    print(f"\nDownstream service sees:")
    print(f"  correlation_id: {downstream_ctx.correlation_id}")
    print(f"  user_id: {downstream_ctx.user_id}")
    print(f"  tenant: {downstream_ctx.baggage.get('tenant')}")

    # All logs from this request share the correlation ID
    print(f"\n  All log entries tagged with correlation_id={downstream_ctx.correlation_id}")


demonstrate_correlation_ids()
```

---

## 5. Distributed Logging

### 5.1 Structured Logging

```python
class StructuredLogger:
    """
    Structured logging for distributed systems.

    All log entries are JSON-formatted with consistent fields
    for correlation, timing, and context.
    """

    def __init__(self, service_name: str, instance_id: str):
        self.service_name = service_name
        self.instance_id = instance_id
        self.logs: list[dict] = []

    def log(self, level: str, message: str, context: Optional[CorrelationContext] = None,
            **extra):
        entry = {
            "timestamp": time.time(),
            "level": level,
            "service": self.service_name,
            "instance": self.instance_id,
            "message": message,
        }
        if context:
            entry["correlation_id"] = context.correlation_id
            entry["request_id"] = context.request_id
            entry["user_id"] = context.user_id

        entry.update(extra)
        self.logs.append(entry)
        return entry

    def info(self, message: str, **kwargs):
        return self.log("INFO", message, **kwargs)

    def error(self, message: str, **kwargs):
        return self.log("ERROR", message, **kwargs)

    def warn(self, message: str, **kwargs):
        return self.log("WARN", message, **kwargs)


class LogAggregator:
    """
    Centralized log aggregator.

    Collects logs from all services and provides
    correlation-based search.
    """

    def __init__(self):
        self.logs: list[dict] = []

    def ingest(self, logs: list[dict]):
        self.logs.extend(logs)

    def search_by_correlation(self, correlation_id: str) -> list[dict]:
        return [
            log for log in self.logs
            if log.get("correlation_id") == correlation_id
        ]

    def search_by_level(self, level: str, since: float = 0) -> list[dict]:
        return [
            log for log in self.logs
            if log.get("level") == level and log.get("timestamp", 0) >= since
        ]

    def timeline(self, correlation_id: str) -> list[dict]:
        logs = self.search_by_correlation(correlation_id)
        return sorted(logs, key=lambda l: l.get("timestamp", 0))


def demonstrate_distributed_logging():
    """Demonstrate distributed structured logging."""
    print("=== Distributed Logging ===\n")

    aggregator = LogAggregator()
    ctx = CorrelationContext()
    ctx.user_id = "user-456"

    # API Gateway logs
    gw_logger = StructuredLogger("api-gateway", "gw-01")
    gw_logger.info("Request received", context=ctx, path="/api/orders", method="POST")
    gw_logger.info("Routing to order-service", context=ctx)

    # Order Service logs
    order_logger = StructuredLogger("order-service", "order-02")
    order_logger.info("Processing order", context=ctx, order_total=99.99)
    order_logger.info("Calling payment service", context=ctx)

    # Payment Service logs
    pay_logger = StructuredLogger("payment-service", "pay-01")
    pay_logger.info("Charging card", context=ctx, amount=99.99)
    pay_logger.error("Payment declined", context=ctx, reason="insufficient_funds")

    # Order Service gets error
    order_logger.error("Payment failed", context=ctx, error="payment_declined")

    # Aggregate all logs
    for logger in [gw_logger, order_logger, pay_logger]:
        aggregator.ingest(logger.logs)

    # Search by correlation ID
    print(f"Request timeline (correlation_id={ctx.correlation_id}):")
    for log in aggregator.timeline(ctx.correlation_id):
        level = log["level"]
        svc = log["service"]
        msg = log["message"]
        print(f"  [{level:5s}] {svc:20s}: {msg}")

    # Error search
    errors = aggregator.search_by_level("ERROR")
    print(f"\nRecent errors: {len(errors)}")


demonstrate_distributed_logging()
```

---

## 6. Metrics Collection

### 6.1 Metric Types

```python
class Counter:
    """Monotonically increasing counter."""
    def __init__(self, name: str, labels: dict = None):
        self.name = name
        self.labels = labels or {}
        self.value: float = 0

    def inc(self, amount: float = 1.0):
        self.value += amount


class Gauge:
    """Value that can go up or down."""
    def __init__(self, name: str, labels: dict = None):
        self.name = name
        self.labels = labels or {}
        self.value: float = 0

    def set(self, value: float):
        self.value = value

    def inc(self, amount: float = 1.0):
        self.value += amount

    def dec(self, amount: float = 1.0):
        self.value -= amount


class Histogram:
    """Distribution of values with configurable buckets."""
    def __init__(self, name: str, buckets: list[float] = None, labels: dict = None):
        self.name = name
        self.labels = labels or {}
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        self.counts: Dict[float, int] = {b: 0 for b in self.buckets}
        self.counts[float('inf')] = 0
        self.sum: float = 0
        self.count: int = 0

    def observe(self, value: float):
        self.sum += value
        self.count += 1
        for bucket in self.buckets:
            if value <= bucket:
                self.counts[bucket] += 1
        self.counts[float('inf')] += 1

    def percentile(self, p: float) -> float:
        """Calculate approximate percentile."""
        target = self.count * p
        cumulative = 0
        for bucket in sorted(self.counts.keys()):
            cumulative += self.counts[bucket]
            if cumulative >= target:
                return bucket
        return self.buckets[-1]


class MetricsRegistry:
    """Central metrics registry for a service."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}

    def counter(self, name: str, **labels) -> Counter:
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        if key not in self.counters:
            self.counters[key] = Counter(name, labels)
        return self.counters[key]

    def gauge(self, name: str, **labels) -> Gauge:
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        if key not in self.gauges:
            self.gauges[key] = Gauge(name, labels)
        return self.gauges[key]

    def histogram(self, name: str, buckets: list[float] = None, **labels) -> Histogram:
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        if key not in self.histograms:
            self.histograms[key] = Histogram(name, buckets, labels)
        return self.histograms[key]

    def snapshot(self) -> dict:
        return {
            "service": self.service_name,
            "counters": {k: c.value for k, c in self.counters.items()},
            "gauges": {k: g.value for k, g in self.gauges.items()},
            "histograms": {
                k: {"count": h.count, "sum": h.sum,
                    "p50": h.percentile(0.5), "p99": h.percentile(0.99)}
                for k, h in self.histograms.items()
            },
        }


def demonstrate_metrics():
    """Demonstrate metrics collection for distributed systems."""
    print("=== Metrics Collection ===\n")

    registry = MetricsRegistry("order-service")

    # Request counter
    req_counter = registry.counter("http_requests_total", method="POST", path="/orders")
    err_counter = registry.counter("http_errors_total", method="POST", path="/orders")

    # Active connections gauge
    connections = registry.gauge("active_connections")

    # Request duration histogram
    duration = registry.histogram("http_request_duration_seconds",
                                   buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0])

    # Simulate traffic
    for _ in range(1000):
        req_counter.inc()
        connections.inc()

        # Simulate request duration
        latency = random.expovariate(10)  # ~100ms average
        duration.observe(latency)

        if random.random() < 0.02:  # 2% error rate
            err_counter.inc()

        connections.dec()

    snapshot = registry.snapshot()
    print("Metrics snapshot:")
    print(f"  Requests: {snapshot['counters']}")
    print(f"  Connections: {snapshot['gauges']}")
    print(f"  Latency: {snapshot['histograms']}")


demonstrate_metrics()
```

---

## 7. Anomaly Detection

### 7.1 Statistical Anomaly Detection

```python
class AnomalyDetector:
    """
    Simple anomaly detection for distributed system metrics.

    Uses moving average and standard deviation to detect
    anomalies in time series data.
    """

    def __init__(self, window_size: int = 30, threshold_sigmas: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold_sigmas
        self.values: list[float] = []
        self.anomalies: list[dict] = []

    def observe(self, value: float, timestamp: float = None) -> bool:
        """Record a value and check for anomaly."""
        self.values.append(value)
        ts = timestamp or time.time()

        if len(self.values) < self.window_size:
            return False

        window = self.values[-self.window_size:]
        mean = sum(window) / len(window)
        std_dev = (sum((x - mean) ** 2 for x in window) / len(window)) ** 0.5

        if std_dev == 0:
            return False

        z_score = abs(value - mean) / std_dev
        is_anomaly = z_score > self.threshold

        if is_anomaly:
            self.anomalies.append({
                "timestamp": ts,
                "value": value,
                "mean": round(mean, 3),
                "std_dev": round(std_dev, 3),
                "z_score": round(z_score, 2),
            })

        return is_anomaly


def demonstrate_anomaly_detection():
    """Demonstrate anomaly detection on distributed system metrics."""
    print("=== Anomaly Detection ===\n")

    detector = AnomalyDetector(window_size=20, threshold_sigmas=2.5)

    # Normal latency: ~100ms with 10ms std dev
    for i in range(50):
        latency = random.gauss(100, 10)
        is_anomaly = detector.observe(latency)

    # Inject anomaly: sudden latency spike
    for i in range(5):
        latency = random.gauss(500, 50)  # 5x normal
        is_anomaly = detector.observe(latency)
        if is_anomaly:
            print(f"  ANOMALY at t={50+i}: latency={latency:.0f}ms "
                  f"(z={detector.anomalies[-1]['z_score']})")

    # Recovery
    for i in range(20):
        latency = random.gauss(100, 10)
        detector.observe(latency)

    print(f"\nTotal anomalies detected: {len(detector.anomalies)}")


demonstrate_anomaly_detection()
```

---

## 8. Debugging Distributed Systems

### 8.1 Debugging Workflow

```python
class DistributedDebugger:
    """
    Debugging toolkit for distributed systems.

    Combines traces, logs, and metrics to diagnose issues.
    """

    def __init__(self, trace_collector: TraceCollector, log_aggregator: LogAggregator):
        self.traces = trace_collector
        self.logs = log_aggregator

    def diagnose_slow_request(self, trace_id: str) -> dict:
        """Diagnose a slow request using trace data."""
        spans = self.traces.get_trace(trace_id)
        if not spans:
            return {"error": "Trace not found"}

        root = spans[0]
        total_time = root.duration_ms

        # Find the slowest span
        slowest = max(spans, key=lambda s: s.duration_ms)

        # Find error spans
        errors = [s for s in spans if s.status == "error"]

        # Find critical path
        critical_path = self._find_critical_path(spans)

        return {
            "trace_id": trace_id,
            "total_time_ms": total_time,
            "num_spans": len(spans),
            "slowest_span": {
                "service": slowest.service_name,
                "operation": slowest.operation_name,
                "duration_ms": slowest.duration_ms,
            },
            "errors": [
                {"service": e.service_name, "operation": e.operation_name}
                for e in errors
            ],
            "critical_path": critical_path,
        }

    def _find_critical_path(self, spans: list[Span]) -> list[dict]:
        """Find the critical path (longest sequential chain)."""
        children = defaultdict(list)
        for s in spans:
            if s.parent_span_id:
                children[s.parent_span_id].append(s)

        def longest_path(span):
            child_spans = children.get(span.span_id, [])
            if not child_spans:
                return [span]
            longest = max(
                (longest_path(c) for c in child_spans),
                key=lambda p: sum(s.duration_ms for s in p),
            )
            return [span] + longest

        root = next((s for s in spans if s.parent_span_id is None), spans[0])
        path = longest_path(root)
        return [{"service": s.service_name, "op": s.operation_name,
                 "ms": round(s.duration_ms, 1)} for s in path]

    def correlate_error(self, correlation_id: str) -> dict:
        """Correlate logs from all services for a failed request."""
        logs = self.logs.search_by_correlation(correlation_id)
        errors = [l for l in logs if l.get("level") == "ERROR"]

        services_involved = list(set(l.get("service") for l in logs))
        error_chain = []
        for error in errors:
            error_chain.append({
                "service": error.get("service"),
                "message": error.get("message"),
                "timestamp": error.get("timestamp"),
            })

        return {
            "correlation_id": correlation_id,
            "services_involved": services_involved,
            "total_log_entries": len(logs),
            "error_chain": sorted(error_chain, key=lambda e: e.get("timestamp", 0)),
        }


def demonstrate_debugging():
    """Demonstrate distributed system debugging."""
    print("=== Debugging Distributed Systems ===\n")

    # Setup
    collector = TraceCollector()
    aggregator = LogAggregator()
    debugger = DistributedDebugger(collector, aggregator)

    # Simulate a slow request trace
    trace_id = "abc123"
    spans = [
        Span(trace_id=trace_id, span_id="s1", operation_name="POST /api",
             service_name="gateway", duration_ms=250),
        Span(trace_id=trace_id, span_id="s2", parent_span_id="s1",
             operation_name="createOrder", service_name="order-svc", duration_ms=200),
        Span(trace_id=trace_id, span_id="s3", parent_span_id="s2",
             operation_name="INSERT", service_name="postgres", duration_ms=150),
        Span(trace_id=trace_id, span_id="s4", parent_span_id="s2",
             operation_name="charge", service_name="payment-svc", duration_ms=30),
    ]
    for span in spans:
        collector.collect(span)

    diagnosis = debugger.diagnose_slow_request(trace_id)
    print("Slow request diagnosis:")
    print(f"  Total time: {diagnosis['total_time_ms']}ms")
    print(f"  Slowest: {diagnosis['slowest_span']}")
    print(f"  Critical path:")
    for step in diagnosis['critical_path']:
        print(f"    {step['service']}/{step['op']}: {step['ms']}ms")


demonstrate_debugging()
```

---

## 9. Real-World Observability Stacks

```python
def compare_observability_stacks():
    """Compare real-world observability stacks."""
    print("=== Observability Stack Comparison ===\n")

    stacks = [
        {"name": "OpenTelemetry + Jaeger + Prometheus + ELK",
         "type": "Open source",
         "traces": "Jaeger/Tempo", "metrics": "Prometheus", "logs": "Elasticsearch"},
        {"name": "Datadog",
         "type": "SaaS",
         "traces": "Datadog APM", "metrics": "Datadog Metrics", "logs": "Datadog Logs"},
        {"name": "Grafana Stack (LGTM)",
         "type": "Open source",
         "traces": "Tempo", "metrics": "Mimir", "logs": "Loki"},
        {"name": "AWS Native",
         "type": "Cloud",
         "traces": "X-Ray", "metrics": "CloudWatch", "logs": "CloudWatch Logs"},
    ]

    for stack in stacks:
        print(f"  {stack['name']} ({stack['type']}):")
        print(f"    Traces: {stack['traces']}")
        print(f"    Metrics: {stack['metrics']}")
        print(f"    Logs: {stack['logs']}")
        print()


compare_observability_stacks()
```

---

## 10. Summary and Key Takeaways

### Observability Checklist

> **DISTRIBUTED OBSERVABILITY CHECKLIST**
>
> ☐ Distributed tracing with W3C context propagation
> ☐ Correlation IDs in all service-to-service calls
> ☐ Structured JSON logging with consistent fields
> ☐ RED metrics (Rate, Errors, Duration) per service
> ☐ Centralized log aggregation with full-text search
> ☐ Trace-to-log correlation for debugging
> ☐ Anomaly detection on latency and error rate
> ☐ Dashboards for each service and the overall system

### Key Principles

1. **Correlation is king**: Without correlation IDs, debugging across services is impossible.
2. **Structured logs > unstructured logs**: JSON logs enable machine parsing and search.
3. **Traces show the path, logs show the detail**: Use traces to find the slow service, logs to find why.
4. **RED metrics for every service**: Rate, Errors, Duration — the minimum viable metrics.
5. **OpenTelemetry is the standard**: Vendor-neutral, widely supported, future-proof.

---

## 11. Practice Problems

### Problem 1: Trace Analysis

Given a trace with 12 spans across 4 services, identify the critical path and calculate the percentage of time spent in each service.

### Problem 2: Log Correlation Design

Design a structured logging format that includes: timestamp, level, service, instance, correlation_id, trace_id, span_id, user_id, and arbitrary key-value fields. Implement log aggregation search.

### Problem 3: Metric Dashboard

Design a dashboard for a 3-tier application (web → API → DB). Include: request rate, error rate, latency percentiles, active connections, and queue depth. Define alerting thresholds.

### Problem 4: Implementation Challenge

Build an observability library with: Tracer (span creation and context propagation), StructuredLogger (JSON logging with correlation), MetricsRegistry (counters, gauges, histograms), all linked by correlation IDs.

### Problem 5: Debugging Exercise

Given logs from 5 services with a common correlation_id showing an intermittent 500 error, design a systematic debugging procedure. What information do you need? In what order do you investigate?

---

## 12. References

1. Sridharan, C. (2018). *Distributed Systems Observability*. O'Reilly Media.
2. OpenTelemetry documentation: https://opentelemetry.io/docs/
3. Sigelman, B. et al. (2010). "Dapper, a Large-Scale Distributed Systems Tracing Infrastructure." Google Technical Report.
4. W3C Trace Context specification: https://www.w3.org/TR/trace-context/
5. Beyer, B. et al. (2016). *Site Reliability Engineering*. O'Reilly Media.
6. Majors, C. et al. (2022). *Observability Engineering*. O'Reilly Media.
7. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Ch. 4. O'Reilly Media.

---

[Next: Lesson 28 — Capstone: Distributed KV Store](./28_Capstone_Distributed_KV.md)
