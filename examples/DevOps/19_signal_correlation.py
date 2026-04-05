#!/usr/bin/env python3
"""Example: Signal Correlation — Linking Metrics, Logs & Traces

Demonstrates cross-signal correlation: exemplar-based metric-to-trace
linking, log-to-trace enrichment, automated root-cause narrowing by
correlating anomalous signals across services.
Related lesson: 21_Signal_Correlation.md
"""

# =============================================================================
# WHY SIGNAL CORRELATION?
# Individual metrics, logs, and traces only tell part of the story. Signal
# correlation connects a latency spike (metric) to the specific slow trace
# and its error logs, enabling engineers to jump from "something is wrong"
# to "here is why" in seconds rather than hours.
# =============================================================================

import random
import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional


# =============================================================================
# 1. TELEMETRY SIGNAL MODELS
# =============================================================================

@dataclass
class Exemplar:
    """Links a metric sample to a specific trace for drill-down."""
    trace_id: str
    span_id: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """A time-series metric with exemplar support."""
    name: str
    labels: dict[str, str]
    samples: list[tuple[float, float]] = field(default_factory=list)  # (timestamp, value)
    exemplars: list[Exemplar] = field(default_factory=list)

    def add_sample(self, value: float, exemplar: Exemplar | None = None) -> None:
        ts = time.time()
        self.samples.append((ts, value))
        if exemplar:
            self.exemplars.append(exemplar)


@dataclass
class CorrelatedLog:
    """A log entry with trace context for correlation."""
    message: str
    level: str
    service: str
    trace_id: str = ""
    span_id: str = ""
    timestamp: float = field(default_factory=time.time)
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceSpan:
    """A span in a distributed trace."""
    trace_id: str
    span_id: str
    parent_span_id: str = ""
    service: str = ""
    operation: str = ""
    start_time: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    status: str = "OK"
    attributes: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 2. SIGNAL CORRELATION ENGINE
# =============================================================================

@dataclass
class CorrelationEngine:
    """Correlates signals across metrics, logs, and traces."""
    metrics: list[MetricSeries] = field(default_factory=list)
    logs: list[CorrelatedLog] = field(default_factory=list)
    spans: list[TraceSpan] = field(default_factory=list)

    def trace_from_exemplar(self, exemplar: Exemplar) -> list[TraceSpan]:
        """Given a metric exemplar, find the associated trace spans."""
        return [s for s in self.spans if s.trace_id == exemplar.trace_id]

    def logs_for_trace(self, trace_id: str) -> list[CorrelatedLog]:
        """Find all logs associated with a trace."""
        return sorted(
            [l for l in self.logs if l.trace_id == trace_id],
            key=lambda l: l.timestamp,
        )

    def exemplars_for_anomaly(self, metric_name: str,
                              threshold: float) -> list[Exemplar]:
        """Find exemplars for metric values exceeding a threshold."""
        results = []
        for series in self.metrics:
            if series.name == metric_name:
                for ex in series.exemplars:
                    if ex.value > threshold:
                        results.append(ex)
        return results

    def correlate_anomaly(self, metric_name: str,
                          threshold: float) -> list[dict[str, Any]]:
        """Full correlation: metric anomaly -> traces -> logs."""
        correlations = []
        exemplars = self.exemplars_for_anomaly(metric_name, threshold)
        for ex in exemplars:
            spans = self.trace_from_exemplar(ex)
            logs = self.logs_for_trace(ex.trace_id)
            error_logs = [l for l in logs if l.level in ("ERROR", "WARN")]
            slowest_span = max(spans, key=lambda s: s.duration_ms) if spans else None
            correlations.append({
                "trace_id": ex.trace_id,
                "metric_value": ex.value,
                "span_count": len(spans),
                "error_log_count": len(error_logs),
                "slowest_operation": slowest_span.operation if slowest_span else "",
                "slowest_duration_ms": slowest_span.duration_ms if slowest_span else 0,
                "root_cause_hint": (
                    error_logs[0].message if error_logs else
                    f"Slow operation: {slowest_span.operation}" if slowest_span else
                    "Unknown"
                ),
            })
        return correlations


# =============================================================================
# 3. SERVICE DEPENDENCY CORRELATION
# =============================================================================

def build_service_graph(spans: list[TraceSpan]) -> dict[str, set[str]]:
    """Build a service dependency graph from trace spans."""
    span_map = {s.span_id: s for s in spans}
    graph: dict[str, set[str]] = {}
    for span in spans:
        if span.parent_span_id and span.parent_span_id in span_map:
            parent = span_map[span.parent_span_id]
            if parent.service != span.service:
                graph.setdefault(parent.service, set()).add(span.service)
    return graph


def find_error_propagation(spans: list[TraceSpan]) -> list[str]:
    """Trace error propagation path through services."""
    error_spans = [s for s in spans if s.status == "ERROR"]
    if not error_spans:
        return []
    # Sort by start time to find the origin
    error_spans.sort(key=lambda s: s.start_time)
    return [f"{s.service}/{s.operation} ({s.duration_ms:.1f}ms)" for s in error_spans]


# =============================================================================
# 4. SYNTHETIC DATA GENERATOR
# =============================================================================

def generate_test_data(engine: CorrelationEngine, num_traces: int = 20) -> None:
    """Generate realistic correlated telemetry data."""
    services = ["api-gw", "order-svc", "payment-svc", "inventory-svc"]
    random.seed(42)

    for i in range(num_traces):
        trace_id = hashlib.md5(f"trace-{i}".encode()).hexdigest()[:16]
        is_slow = random.random() < 0.15
        has_error = random.random() < 0.1

        base_latency = random.uniform(50, 150) if not is_slow else random.uniform(500, 2000)
        parent_span_id = ""

        for svc_idx, svc in enumerate(services[:random.randint(2, 4)]):
            span_id = hashlib.md5(f"{trace_id}-{svc}".encode()).hexdigest()[:16]
            svc_latency = base_latency * random.uniform(0.2, 0.5)
            span = TraceSpan(
                trace_id=trace_id, span_id=span_id,
                parent_span_id=parent_span_id, service=svc,
                operation=f"{svc}.handle",
                duration_ms=svc_latency,
                status="ERROR" if has_error and svc_idx == len(services) - 2 else "OK",
            )
            engine.spans.append(span)
            parent_span_id = span_id

            # Correlated log
            if has_error and span.status == "ERROR":
                engine.logs.append(CorrelatedLog(
                    message=f"Connection timeout to downstream",
                    level="ERROR", service=svc,
                    trace_id=trace_id, span_id=span_id,
                ))
            else:
                engine.logs.append(CorrelatedLog(
                    message=f"Request processed",
                    level="INFO", service=svc,
                    trace_id=trace_id, span_id=span_id,
                ))

        # Metric with exemplar
        latency_series = MetricSeries(
            name="http_request_duration_ms",
            labels={"service": "api-gw"},
        )
        latency_series.add_sample(base_latency, Exemplar(
            trace_id=trace_id, span_id="root",
            value=base_latency,
        ))
        engine.metrics.append(latency_series)


# =============================================================================
# 5. DEMO
# =============================================================================

if __name__ == "__main__":
    engine = CorrelationEngine()
    generate_test_data(engine, num_traces=30)

    print("=" * 60)
    print("Signal Correlation Engine")
    print("=" * 60)
    print(f"  Loaded: {len(engine.spans)} spans, {len(engine.logs)} logs, "
          f"{len(engine.metrics)} metric series")

    # --- Anomaly Correlation ---
    print(f"\n{'=' * 60}")
    print("Anomaly Correlation (latency > 500ms)")
    print("=" * 60)
    correlations = engine.correlate_anomaly("http_request_duration_ms", 500)
    for c in correlations[:5]:
        print(f"  Trace {c['trace_id']}:")
        print(f"    Latency: {c['metric_value']:.1f}ms")
        print(f"    Spans: {c['span_count']}, Error logs: {c['error_log_count']}")
        print(f"    Slowest: {c['slowest_operation']} ({c['slowest_duration_ms']:.1f}ms)")
        print(f"    Hint: {c['root_cause_hint']}")

    # --- Service Dependency Graph ---
    print(f"\n{'=' * 60}")
    print("Service Dependency Graph (from traces)")
    print("=" * 60)
    graph = build_service_graph(engine.spans)
    for svc, deps in graph.items():
        print(f"  {svc} -> {', '.join(deps)}")

    # --- Error Propagation ---
    print(f"\n{'=' * 60}")
    print("Error Propagation Path")
    print("=" * 60)
    # Find a trace with errors
    error_traces = {s.trace_id for s in engine.spans if s.status == "ERROR"}
    for tid in list(error_traces)[:2]:
        trace_spans = [s for s in engine.spans if s.trace_id == tid]
        path = find_error_propagation(trace_spans)
        print(f"  Trace {tid}: {' -> '.join(path)}")
