#!/usr/bin/env python3
"""Example: Observability Engineering — The Three Pillars Unified

Demonstrates observability fundamentals: structured telemetry collection,
context propagation across metrics/logs/traces, cardinality management,
and a unified observability data model.
Related lesson: 19_Observability_Engineering.md (conceptual overview)
"""

# =============================================================================
# WHY OBSERVABILITY ENGINEERING?
# Monitoring tells you WHEN something is broken. Observability lets you ask
# arbitrary questions about WHY. It combines high-cardinality metrics,
# structured logs, and distributed traces into a single correlated view
# so you can debug novel failures without deploying new instrumentation.
# =============================================================================

import time
import random
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# =============================================================================
# 1. UNIFIED TELEMETRY DATA MODEL
# =============================================================================

class SignalType(Enum):
    METRIC = "metric"
    LOG = "log"
    TRACE = "trace"


@dataclass
class TelemetryContext:
    """Shared context propagated across all signal types."""
    trace_id: str = ""
    span_id: str = ""
    service: str = ""
    environment: str = "production"
    attributes: dict[str, str] = field(default_factory=dict)

    def derive_span(self) -> "TelemetryContext":
        """Create a child span context."""
        new_span = hashlib.md5(
            f"{self.span_id}-{time.monotonic()}".encode()
        ).hexdigest()[:16]
        return TelemetryContext(
            trace_id=self.trace_id,
            span_id=new_span,
            service=self.service,
            environment=self.environment,
            attributes=self.attributes.copy(),
        )


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: float
    metric_type: str = "gauge"  # gauge, counter, histogram
    labels: dict[str, str] = field(default_factory=dict)
    context: Optional[TelemetryContext] = None


@dataclass
class LogEntry:
    """A structured log entry."""
    message: str
    level: str = "INFO"
    timestamp: float = field(default_factory=time.time)
    fields: dict[str, Any] = field(default_factory=dict)
    context: Optional[TelemetryContext] = None


@dataclass
class Span:
    """A distributed trace span."""
    name: str
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    status: str = "OK"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    context: Optional[TelemetryContext] = None

    @property
    def duration_ms(self) -> float:
        if self.end_time == 0:
            return 0.0
        return (self.end_time - self.start_time) * 1000


# =============================================================================
# 2. TELEMETRY COLLECTOR
# =============================================================================

@dataclass
class TelemetryCollector:
    """Collects and correlates all telemetry signals."""
    metrics: list[MetricPoint] = field(default_factory=list)
    logs: list[LogEntry] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)

    def record_metric(self, name: str, value: float,
                      labels: dict[str, str] | None = None,
                      ctx: TelemetryContext | None = None) -> None:
        self.metrics.append(MetricPoint(
            name=name, value=value, timestamp=time.time(),
            labels=labels or {}, context=ctx,
        ))

    def record_log(self, message: str, level: str = "INFO",
                   fields: dict[str, Any] | None = None,
                   ctx: TelemetryContext | None = None) -> None:
        self.logs.append(LogEntry(
            message=message, level=level,
            fields=fields or {}, context=ctx,
        ))

    def start_span(self, name: str, ctx: TelemetryContext | None = None) -> Span:
        span = Span(name=name, context=ctx)
        self.spans.append(span)
        return span

    def end_span(self, span: Span, status: str = "OK") -> None:
        span.end_time = time.time()
        span.status = status

    def correlate_by_trace(self, trace_id: str) -> dict[str, list]:
        """Find all signals with a given trace ID."""
        return {
            "metrics": [m for m in self.metrics
                        if m.context and m.context.trace_id == trace_id],
            "logs": [l for l in self.logs
                     if l.context and l.context.trace_id == trace_id],
            "spans": [s for s in self.spans
                      if s.context and s.context.trace_id == trace_id],
        }


# =============================================================================
# 3. CARDINALITY ANALYZER
# =============================================================================

def analyze_cardinality(metrics: list[MetricPoint]) -> dict[str, Any]:
    """Analyze metric cardinality to detect explosion risks."""
    label_values: dict[str, set[str]] = {}
    for m in metrics:
        for k, v in m.labels.items():
            label_values.setdefault(k, set()).add(v)

    analysis = {}
    for label, values in label_values.items():
        card = len(values)
        risk = "HIGH" if card > 100 else "MEDIUM" if card > 20 else "LOW"
        analysis[label] = {"cardinality": card, "risk": risk}

    total_series = 1
    for vals in label_values.values():
        total_series *= max(len(vals), 1)
    return {
        "labels": analysis,
        "total_series_estimate": total_series,
        "explosion_risk": total_series > 10000,
    }


# =============================================================================
# 4. RED/USE METHOD CALCULATORS
# =============================================================================

def compute_red_metrics(spans: list[Span]) -> dict[str, float]:
    """Compute RED metrics (Rate, Errors, Duration) from spans."""
    if not spans:
        return {"rate": 0, "error_rate": 0, "p50_ms": 0, "p99_ms": 0}
    total = len(spans)
    errors = sum(1 for s in spans if s.status != "OK")
    durations = sorted(s.duration_ms for s in spans if s.duration_ms > 0)
    p50_idx = max(0, int(len(durations) * 0.50) - 1)
    p99_idx = max(0, int(len(durations) * 0.99) - 1)
    return {
        "rate": total,
        "error_rate": errors / total if total else 0,
        "p50_ms": round(durations[p50_idx], 2) if durations else 0,
        "p99_ms": round(durations[p99_idx], 2) if durations else 0,
    }


def compute_use_metrics(resource_samples: list[dict]) -> dict[str, Any]:
    """Compute USE metrics (Utilization, Saturation, Errors) for a resource."""
    if not resource_samples:
        return {}
    utilizations = [s["utilization"] for s in resource_samples]
    saturations = [s["saturation"] for s in resource_samples]
    error_counts = [s.get("errors", 0) for s in resource_samples]
    return {
        "avg_utilization": round(sum(utilizations) / len(utilizations), 2),
        "max_utilization": max(utilizations),
        "avg_saturation": round(sum(saturations) / len(saturations), 2),
        "total_errors": sum(error_counts),
    }


# =============================================================================
# 5. DEMO
# =============================================================================

if __name__ == "__main__":
    random.seed(42)
    collector = TelemetryCollector()

    # Simulate a request flowing through multiple services
    print("=" * 60)
    print("Unified Observability — Request Flow")
    print("=" * 60)
    root_ctx = TelemetryContext(
        trace_id="abc123def456", span_id="root0001",
        service="api-gateway", environment="production",
    )

    # API Gateway span
    gw_span = collector.start_span("api-gateway.handle", ctx=root_ctx)
    collector.record_log("Received request", "INFO",
                         {"path": "/api/orders", "method": "POST"}, root_ctx)
    collector.record_metric("http_requests_total", 1,
                            {"method": "POST", "path": "/api/orders"}, root_ctx)

    # Order service span (child)
    order_ctx = root_ctx.derive_span()
    order_ctx.service = "order-svc"
    order_span = collector.start_span("order-svc.create_order", ctx=order_ctx)
    collector.record_log("Creating order", "INFO", {"user_id": "u42"}, order_ctx)
    time.sleep(0.01)
    collector.end_span(order_span)

    # Payment service span (child)
    pay_ctx = root_ctx.derive_span()
    pay_ctx.service = "payment-svc"
    pay_span = collector.start_span("payment-svc.charge", ctx=pay_ctx)
    collector.record_metric("payment_amount", 49.99, {"currency": "USD"}, pay_ctx)
    time.sleep(0.005)
    collector.end_span(pay_span)

    collector.end_span(gw_span)

    # Correlate by trace ID
    correlated = collector.correlate_by_trace("abc123def456")
    print(f"  Trace abc123def456:")
    print(f"    Spans: {len(correlated['spans'])}")
    print(f"    Logs:  {len(correlated['logs'])}")
    print(f"    Metrics: {len(correlated['metrics'])}")
    for span in correlated["spans"]:
        print(f"    - {span.name}: {span.duration_ms:.2f}ms [{span.status}]")

    # --- RED Metrics ---
    print(f"\n{'=' * 60}")
    print("RED Metrics")
    print("=" * 60)
    # Generate synthetic spans
    for i in range(200):
        ctx = TelemetryContext(trace_id=f"trace-{i}", service="order-svc")
        s = collector.start_span("order-svc.handle", ctx=ctx)
        s.end_time = s.start_time + random.uniform(0.005, 0.2)
        s.status = "ERROR" if random.random() < 0.05 else "OK"
    red = compute_red_metrics(collector.spans)
    print(f"  Rate: {red['rate']} requests")
    print(f"  Error rate: {red['error_rate']:.2%}")
    print(f"  p50 latency: {red['p50_ms']:.2f}ms")
    print(f"  p99 latency: {red['p99_ms']:.2f}ms")

    # --- Cardinality Analysis ---
    print(f"\n{'=' * 60}")
    print("Cardinality Analysis")
    print("=" * 60)
    # Simulate high-cardinality metric
    for i in range(150):
        collector.record_metric("http_requests", 1, {
            "method": random.choice(["GET", "POST"]),
            "path": f"/api/users/{random.randint(1, 500)}",  # High cardinality!
            "status": random.choice(["200", "201", "400", "500"]),
        })
    card = analyze_cardinality(collector.metrics)
    for label, info in card["labels"].items():
        print(f"  {label}: cardinality={info['cardinality']} risk={info['risk']}")
    print(f"  Total series estimate: {card['total_series_estimate']}")
    print(f"  Explosion risk: {card['explosion_risk']}")
