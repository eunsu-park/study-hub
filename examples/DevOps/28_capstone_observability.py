#!/usr/bin/env python3
"""Example: Capstone — Full-Stack Observability Platform Simulation

Demonstrates a complete observability stack integration: service instrumentation,
telemetry pipeline, backend storage simulation, dashboard data model,
SLO monitoring, and automated incident detection — tying together concepts
from the entire DevOps observability track.
Related lesson: 28_Capstone_Full_Stack_Observability.md
"""

# =============================================================================
# WHY A CAPSTONE?
# Individual observability tools (metrics, logs, traces, profiles) are most
# powerful when integrated. This capstone builds a miniature observability
# platform that instruments services, collects correlated telemetry, evaluates
# SLOs, and triggers incidents — demonstrating the full feedback loop.
# =============================================================================

import random
import time
import math
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional


# =============================================================================
# 1. SERVICE SIMULATOR
# =============================================================================

@dataclass
class ServiceEndpoint:
    """A single API endpoint of a service."""
    path: str
    method: str = "GET"
    base_latency_ms: float = 20.0
    error_rate: float = 0.01
    dependencies: list[str] = field(default_factory=list)


@dataclass
class Service:
    """A microservice in the platform."""
    name: str
    endpoints: list[ServiceEndpoint] = field(default_factory=list)
    healthy: bool = True
    degradation_factor: float = 1.0  # Multiplier for latency

    def handle_request(self, endpoint: ServiceEndpoint) -> dict[str, Any]:
        """Simulate handling a request."""
        latency = endpoint.base_latency_ms * self.degradation_factor
        latency *= random.uniform(0.5, 2.0)  # Natural variance
        is_error = random.random() < endpoint.error_rate * (
            5.0 if not self.healthy else 1.0
        )
        return {
            "service": self.name,
            "path": endpoint.path,
            "method": endpoint.method,
            "status": 500 if is_error else 200,
            "latency_ms": round(latency, 2),
            "timestamp": time.time(),
        }


# =============================================================================
# 2. TELEMETRY COLLECTOR
# =============================================================================

@dataclass
class MetricSample:
    name: str
    value: float
    labels: dict[str, str]
    timestamp: float


@dataclass
class LogEntry:
    level: str
    message: str
    service: str
    trace_id: str
    timestamp: float
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanRecord:
    trace_id: str
    span_id: str
    service: str
    operation: str
    duration_ms: float
    status: str
    timestamp: float
    parent_id: str = ""


@dataclass
class TelemetryBackend:
    """Unified telemetry storage backend."""
    metrics: list[MetricSample] = field(default_factory=list)
    logs: list[LogEntry] = field(default_factory=list)
    spans: list[SpanRecord] = field(default_factory=list)

    def ingest_request(self, request_data: dict[str, Any],
                       trace_id: str, span_id: str) -> None:
        """Ingest all telemetry signals from a single request."""
        ts = request_data["timestamp"]
        svc = request_data["service"]
        status = request_data["status"]

        # Metric
        self.metrics.append(MetricSample(
            name="http_request_duration_ms",
            value=request_data["latency_ms"],
            labels={"service": svc, "path": request_data["path"],
                    "method": request_data["method"], "status": str(status)},
            timestamp=ts,
        ))
        self.metrics.append(MetricSample(
            name="http_requests_total",
            value=1,
            labels={"service": svc, "status": str(status)},
            timestamp=ts,
        ))

        # Log
        level = "ERROR" if status >= 500 else "INFO"
        self.logs.append(LogEntry(
            level=level,
            message=f"{request_data['method']} {request_data['path']} -> {status}",
            service=svc, trace_id=trace_id, timestamp=ts,
            fields={"latency_ms": request_data["latency_ms"]},
        ))

        # Span
        self.spans.append(SpanRecord(
            trace_id=trace_id, span_id=span_id,
            service=svc, operation=f"{svc}.handle",
            duration_ms=request_data["latency_ms"],
            status="ERROR" if status >= 500 else "OK",
            timestamp=ts,
        ))


# =============================================================================
# 3. SLO EVALUATOR
# =============================================================================

@dataclass
class SLOConfig:
    name: str
    service: str
    target: float  # e.g., 0.999
    window_seconds: float = 3600  # 1 hour
    sli_type: str = "availability"  # availability, latency


@dataclass
class SLOEvaluator:
    """Evaluate SLOs from collected telemetry."""
    config: SLOConfig
    backend: TelemetryBackend

    def evaluate(self) -> dict[str, Any]:
        cutoff = time.time() - self.config.window_seconds
        relevant = [
            m for m in self.backend.metrics
            if m.name == "http_requests_total"
            and m.labels.get("service") == self.config.service
            and m.timestamp >= cutoff
        ]
        if not relevant:
            return {"slo": self.config.name, "status": "NO_DATA"}

        total = len(relevant)
        good = sum(1 for m in relevant if m.labels.get("status", "200") < "500")
        sli = good / total if total else 0
        error_budget_total = 1 - self.config.target
        error_budget_consumed = (1 - sli) / error_budget_total if error_budget_total else 0
        return {
            "slo": self.config.name,
            "service": self.config.service,
            "target": self.config.target,
            "current_sli": round(sli, 5),
            "meeting_slo": sli >= self.config.target,
            "error_budget_consumed_pct": round(error_budget_consumed * 100, 1),
            "total_requests": total,
            "good_requests": good,
        }


# =============================================================================
# 4. DASHBOARD DATA MODEL
# =============================================================================

@dataclass
class DashboardPanel:
    """A single panel in an observability dashboard."""
    title: str
    panel_type: str  # timeseries, stat, table, heatmap
    query: str
    thresholds: list[dict] = field(default_factory=list)


@dataclass
class Dashboard:
    """An observability dashboard."""
    title: str
    panels: list[DashboardPanel] = field(default_factory=list)
    refresh_interval: str = "30s"

    def generate_data(self, backend: TelemetryBackend) -> list[dict[str, Any]]:
        """Generate panel data from the backend."""
        results = []
        for panel in self.panels:
            if "request_rate" in panel.query:
                value = len(backend.metrics) / max(1, 60)
            elif "error_rate" in panel.query:
                errors = sum(1 for m in backend.metrics
                             if m.labels.get("status", "") == "500")
                total = sum(1 for m in backend.metrics
                            if m.name == "http_requests_total")
                value = errors / total if total else 0
            elif "p99_latency" in panel.query:
                latencies = sorted(m.value for m in backend.metrics
                                   if m.name == "http_request_duration_ms")
                idx = int(len(latencies) * 0.99) if latencies else 0
                value = latencies[idx] if latencies else 0
            else:
                value = 0
            results.append({"panel": panel.title, "value": round(value, 3)})
        return results


def create_golden_signals_dashboard(service: str) -> Dashboard:
    """Create a Golden Signals dashboard for a service."""
    return Dashboard(
        title=f"{service} — Golden Signals",
        panels=[
            DashboardPanel("Request Rate", "timeseries",
                           f"request_rate{{service='{service}'}}"),
            DashboardPanel("Error Rate", "stat",
                           f"error_rate{{service='{service}'}}",
                           [{"value": 0.01, "color": "green"},
                            {"value": 0.05, "color": "red"}]),
            DashboardPanel("P99 Latency", "timeseries",
                           f"p99_latency{{service='{service}'}}",
                           [{"value": 200, "color": "green"},
                            {"value": 1000, "color": "red"}]),
            DashboardPanel("Active Traces", "table",
                           f"traces{{service='{service}'}}"),
        ],
    )


# =============================================================================
# 5. AUTOMATED INCIDENT DETECTOR
# =============================================================================

@dataclass
class IncidentAlert:
    service: str
    severity: str
    title: str
    details: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


def detect_incidents(slo_results: list[dict], backend: TelemetryBackend
                     ) -> list[IncidentAlert]:
    """Detect incidents from SLO breaches and anomalies."""
    alerts = []
    for result in slo_results:
        if not result.get("meeting_slo", True):
            consumed = result.get("error_budget_consumed_pct", 0)
            severity = "SEV1" if consumed > 100 else "SEV2" if consumed > 50 else "SEV3"
            alerts.append(IncidentAlert(
                service=result["service"],
                severity=severity,
                title=f"SLO breach: {result['slo']}",
                details=result,
            ))

    # Check for error spikes
    services = {m.labels.get("service") for m in backend.metrics
                if m.name == "http_requests_total"}
    for svc in services:
        svc_metrics = [m for m in backend.metrics
                       if m.name == "http_requests_total"
                       and m.labels.get("service") == svc]
        errors = sum(1 for m in svc_metrics if m.labels.get("status") == "500")
        if svc_metrics and errors / len(svc_metrics) > 0.1:
            alerts.append(IncidentAlert(
                service=svc, severity="SEV2",
                title=f"High error rate: {svc}",
                details={"error_rate": errors / len(svc_metrics),
                         "error_count": errors},
            ))
    return alerts


# =============================================================================
# 6. DEMO — FULL PLATFORM SIMULATION
# =============================================================================

if __name__ == "__main__":
    random.seed(42)

    # --- Define Services ---
    services = {
        "api-gateway": Service(
            name="api-gateway",
            endpoints=[
                ServiceEndpoint("/api/orders", "POST", 30, 0.02, ["order-svc"]),
                ServiceEndpoint("/api/users", "GET", 15, 0.005, ["user-svc"]),
            ],
        ),
        "order-svc": Service(
            name="order-svc",
            endpoints=[ServiceEndpoint("/orders", "POST", 50, 0.03, ["payment-svc", "db"])],
        ),
        "payment-svc": Service(
            name="payment-svc",
            endpoints=[ServiceEndpoint("/charge", "POST", 100, 0.01)],
            healthy=False, degradation_factor=3.0,  # Degraded!
        ),
        "user-svc": Service(
            name="user-svc",
            endpoints=[ServiceEndpoint("/users", "GET", 10, 0.005)],
        ),
    }

    # --- Simulate Traffic ---
    print("=" * 60)
    print("Full-Stack Observability Platform Simulation")
    print("=" * 60)
    backend = TelemetryBackend()
    trace_count = 0

    for i in range(500):
        # Pick a random service and endpoint
        svc_name = random.choice(list(services.keys()))
        svc = services[svc_name]
        endpoint = random.choice(svc.endpoints)
        trace_id = hashlib.md5(f"trace-{i}".encode()).hexdigest()[:16]
        span_id = hashlib.md5(f"span-{i}-{svc_name}".encode()).hexdigest()[:16]

        result = svc.handle_request(endpoint)
        backend.ingest_request(result, trace_id, span_id)

        # Simulate downstream calls
        for dep in endpoint.dependencies:
            if dep in services:
                dep_svc = services[dep]
                dep_endpoint = dep_svc.endpoints[0]
                dep_result = dep_svc.handle_request(dep_endpoint)
                dep_span = hashlib.md5(f"span-{i}-{dep}".encode()).hexdigest()[:16]
                backend.ingest_request(dep_result, trace_id, dep_span)
        trace_count += 1

    print(f"  Simulated {trace_count} traces")
    print(f"  Collected: {len(backend.metrics)} metrics, {len(backend.logs)} logs, "
          f"{len(backend.spans)} spans")

    # --- SLO Evaluation ---
    print(f"\n{'=' * 60}")
    print("SLO Evaluation")
    print("=" * 60)
    slo_results = []
    for svc_name in services:
        slo = SLOConfig(
            name=f"{svc_name}-availability",
            service=svc_name,
            target=0.99,
            window_seconds=999999,  # Use all data
        )
        evaluator = SLOEvaluator(config=slo, backend=backend)
        result = evaluator.evaluate()
        slo_results.append(result)
        status = "PASS" if result.get("meeting_slo", False) else "FAIL"
        print(f"  [{status}] {svc_name}: SLI={result.get('current_sli', 0):.4f} "
              f"(target={slo.target}), budget consumed={result.get('error_budget_consumed_pct', 0):.1f}%")

    # --- Dashboard ---
    print(f"\n{'=' * 60}")
    print("Golden Signals Dashboard")
    print("=" * 60)
    for svc_name in ["api-gateway", "payment-svc"]:
        dash = create_golden_signals_dashboard(svc_name)
        data = dash.generate_data(backend)
        print(f"  {dash.title}:")
        for panel in data:
            print(f"    {panel['panel']}: {panel['value']}")

    # --- Incident Detection ---
    print(f"\n{'=' * 60}")
    print("Automated Incident Detection")
    print("=" * 60)
    incidents = detect_incidents(slo_results, backend)
    if incidents:
        for inc in incidents:
            print(f"  [{inc.severity}] {inc.title}")
            for k, v in list(inc.details.items())[:3]:
                print(f"    {k}: {v}")
    else:
        print("  No incidents detected")

    # --- Trace Drill-Down ---
    print(f"\n{'=' * 60}")
    print("Trace Drill-Down (error traces)")
    print("=" * 60)
    error_traces = {s.trace_id for s in backend.spans if s.status == "ERROR"}
    for tid in list(error_traces)[:3]:
        spans = [s for s in backend.spans if s.trace_id == tid]
        logs = [l for l in backend.logs if l.trace_id == tid]
        print(f"  Trace {tid}:")
        for s in spans:
            print(f"    [{s.status:>5}] {s.service}/{s.operation} {s.duration_ms:.1f}ms")
        for l in logs:
            if l.level == "ERROR":
                print(f"    LOG: {l.message}")
