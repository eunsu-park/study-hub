#!/usr/bin/env python3
"""Example: API Monitoring

Demonstrates API monitoring and observability patterns:
- Health check endpoints (liveness, readiness, startup)
- SLI/SLO tracking (latency, error rate, availability)
- Structured logging with correlation IDs
- Metrics collection and alerting logic
- Dependency health checks

Related lesson: 23_API_Monitoring.md

Run:
    pip install "fastapi[standard]"
    uvicorn 23_api_monitoring:app --reload --port 8000

    # Health:      GET /health/live, /health/ready, /health/startup
    # Metrics:     GET /metrics
    # SLO status:  GET /slo/status
"""

import logging
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("monitoring")

# =============================================================================
# METRICS COLLECTOR — In production, use Prometheus client library
# =============================================================================

@dataclass
class MetricsCollector:
    """Collects API metrics in-memory. In production, export to Prometheus,
    Datadog, or CloudWatch instead."""

    request_count: int = 0
    error_count: int = 0
    latencies_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    status_codes: dict = field(default_factory=lambda: {})
    endpoint_counts: dict = field(default_factory=lambda: {})

    def record_request(self, path: str, status_code: int, latency_ms: float):
        self.request_count += 1
        if status_code >= 500:
            self.error_count += 1
        self.latencies_ms.append(latency_ms)
        self.status_codes[status_code] = self.status_codes.get(status_code, 0) + 1
        self.endpoint_counts[path] = self.endpoint_counts.get(path, 0) + 1

    @property
    def error_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count

    @property
    def p50_latency(self) -> float:
        return self._percentile(50)

    @property
    def p95_latency(self) -> float:
        return self._percentile(95)

    @property
    def p99_latency(self) -> float:
        return self._percentile(99)

    def _percentile(self, p: int) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * p / 100)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def to_dict(self) -> dict:
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": round(self.error_rate, 4),
            "latency_p50_ms": round(self.p50_latency, 2),
            "latency_p95_ms": round(self.p95_latency, 2),
            "latency_p99_ms": round(self.p99_latency, 2),
            "status_codes": dict(self.status_codes),
            "top_endpoints": dict(sorted(
                self.endpoint_counts.items(), key=lambda x: -x[1]
            )[:10]),
        }


metrics = MetricsCollector()

# =============================================================================
# SLO DEFINITIONS — Service Level Objectives
# =============================================================================
# SLIs (indicators) are measured; SLOs (objectives) are targets.

SLO_DEFINITIONS = {
    "availability": {
        "description": "Percentage of non-5xx responses",
        "target": 99.9,
        "window": "30d",
    },
    "latency_p95": {
        "description": "95th percentile latency",
        "target_ms": 200,
        "window": "30d",
    },
    "error_rate": {
        "description": "Percentage of 5xx responses",
        "target_max": 0.1,
        "window": "30d",
    },
}

# =============================================================================
# DEPENDENCY HEALTH CHECKS
# =============================================================================

def check_database() -> dict:
    """Simulate a database health check (ping query)."""
    # In production: run "SELECT 1" with a short timeout
    return {"name": "database", "status": "healthy", "latency_ms": 2.1}


def check_cache() -> dict:
    """Simulate a Redis cache health check."""
    return {"name": "cache", "status": "healthy", "latency_ms": 0.8}


def check_external_api() -> dict:
    """Simulate external API dependency check."""
    # Randomly degrade to show unhealthy state
    healthy = random.random() > 0.1
    return {
        "name": "payment_api",
        "status": "healthy" if healthy else "degraded",
        "latency_ms": 45.0 if healthy else 2000.0,
    }


# =============================================================================
# APPLICATION
# =============================================================================

app = FastAPI(title="API Monitoring Demo", version="1.0.0")
_startup_complete = True


# =============================================================================
# MIDDLEWARE — Metrics collection and correlation ID
# =============================================================================

@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    # Assign correlation ID for request tracing across services
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4())[:8])

    start = time.monotonic()
    response = await call_next(request)
    latency_ms = (time.monotonic() - start) * 1000

    # Skip health endpoints from metrics
    path = request.url.path
    if not path.startswith("/health"):
        metrics.record_request(path, response.status_code, latency_ms)

    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Response-Time-Ms"] = f"{latency_ms:.1f}"

    # Structured log entry
    logger.info(
        f"method={request.method} path={path} status={response.status_code} "
        f"latency_ms={latency_ms:.1f} correlation_id={correlation_id}"
    )
    return response


# =============================================================================
# HEALTH ENDPOINTS — Kubernetes probe convention
# =============================================================================

@app.get("/health/live", tags=["Health"])
def liveness():
    """Liveness probe: is the process running? Returns 200 if yes.
    Kubernetes restarts the pod if this fails."""
    return {"status": "alive"}


@app.get("/health/ready", tags=["Health"])
def readiness():
    """Readiness probe: can the service handle requests?
    Checks all dependencies. Kubernetes removes from load balancer if unhealthy."""
    deps = [check_database(), check_cache(), check_external_api()]
    all_healthy = all(d["status"] == "healthy" for d in deps)
    status_code = 200 if all_healthy else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if all_healthy else "not_ready", "dependencies": deps},
    )


@app.get("/health/startup", tags=["Health"])
def startup():
    """Startup probe: has initialization completed?
    Kubernetes waits for this before sending liveness probes."""
    if _startup_complete:
        return {"status": "started"}
    return JSONResponse(status_code=503, content={"status": "starting"})


# =============================================================================
# METRICS ENDPOINT — Prometheus-style export
# =============================================================================

@app.get("/metrics", tags=["Observability"])
def get_metrics():
    """Return current API metrics."""
    return metrics.to_dict()


# =============================================================================
# SLO STATUS ENDPOINT
# =============================================================================

@app.get("/slo/status", tags=["Observability"])
def slo_status():
    """Report current SLO compliance."""
    m = metrics
    availability = (1 - m.error_rate) * 100 if m.request_count > 0 else 100.0

    return {
        "availability": {
            **SLO_DEFINITIONS["availability"],
            "current": round(availability, 3),
            "met": availability >= SLO_DEFINITIONS["availability"]["target"],
        },
        "latency_p95": {
            **SLO_DEFINITIONS["latency_p95"],
            "current_ms": round(m.p95_latency, 2),
            "met": m.p95_latency <= SLO_DEFINITIONS["latency_p95"]["target_ms"],
        },
        "error_rate": {
            **SLO_DEFINITIONS["error_rate"],
            "current_pct": round(m.error_rate * 100, 3),
            "met": (m.error_rate * 100) <= SLO_DEFINITIONS["error_rate"]["target_max"],
        },
    }


# =============================================================================
# SAMPLE API ENDPOINT — Generates traffic for metrics
# =============================================================================

@app.get("/api/v1/data", tags=["API"])
def get_data():
    """Sample endpoint that generates some latency variation."""
    time.sleep(random.uniform(0.005, 0.05))
    return {"data": "sample", "timestamp": datetime.now(timezone.utc).isoformat()}


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("23_api_monitoring:app", host="127.0.0.1", port=8000, reload=True)
