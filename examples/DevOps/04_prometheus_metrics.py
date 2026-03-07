#!/usr/bin/env python3
"""Example: Prometheus Metrics Instrumentation

Demonstrates a Flask application instrumented with prometheus_client for
monitoring. Covers counters, histograms, gauges, summaries, and a custom
collector for business metrics.
Related lesson: 09_Monitoring_and_Observability.md
"""

# =============================================================================
# PROMETHEUS METRICS MODEL
# Four core metric types:
#   Counter   — monotonically increasing (e.g., total requests)
#   Gauge     — goes up and down (e.g., active connections)
#   Histogram — bucketed distributions (e.g., request latency)
#   Summary   — streaming quantiles (e.g., p50/p99 latency)
# =============================================================================

import time
import random
import threading
from dataclasses import dataclass
from typing import Callable

# ---------------------------------------------------------------------------
# NOTE: This example uses a local simulation layer so it runs without
# installing prometheus_client or Flask. Replace the Sim* classes with
# real prometheus_client types in production.
# ---------------------------------------------------------------------------


# =============================================================================
# 1. SIMULATED PROMETHEUS METRICS (standalone demo)
# =============================================================================
# In production, replace with:
#   from prometheus_client import Counter, Histogram, Gauge, Summary, Info

@dataclass
class _Metric:
    name: str
    doc: str
    labels: list[str]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"


class SimCounter(_Metric):
    """Simulated Prometheus Counter."""
    def __init__(self, name: str, doc: str, labels: list[str] | None = None):
        super().__init__(name, doc, labels or [])
        self._values: dict[tuple, float] = {}

    def labels(self, **kwargs: str) -> "SimCounter":
        key = tuple(sorted(kwargs.items()))
        if key not in self._values:
            self._values[key] = 0.0
        self._current_key = key
        return self

    def inc(self, amount: float = 1.0) -> None:
        key = getattr(self, "_current_key", ())
        self._values[key] = self._values.get(key, 0.0) + amount

    def collect(self) -> dict[tuple, float]:
        return dict(self._values)


class SimHistogram(_Metric):
    """Simulated Prometheus Histogram."""
    def __init__(self, name: str, doc: str, labels: list[str] | None = None,
                 buckets: list[float] | None = None):
        super().__init__(name, doc, labels or [])
        self._buckets = buckets or [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self._observations: list[tuple[tuple, float]] = []
        self._current_key: tuple = ()

    def labels(self, **kwargs: str) -> "SimHistogram":
        self._current_key = tuple(sorted(kwargs.items()))
        return self

    def observe(self, amount: float) -> None:
        self._observations.append((self._current_key, amount))

    def time(self) -> "_HistTimer":
        return _HistTimer(self)


class _HistTimer:
    """Context manager for timing with a histogram."""
    def __init__(self, hist: SimHistogram):
        self._hist = hist
        self._start = 0.0

    def __enter__(self) -> "_HistTimer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *args: object) -> None:
        self._hist.observe(time.monotonic() - self._start)


class SimGauge(_Metric):
    """Simulated Prometheus Gauge."""
    def __init__(self, name: str, doc: str, labels: list[str] | None = None):
        super().__init__(name, doc, labels or [])
        self._value: float = 0.0

    def set(self, value: float) -> None:
        self._value = value

    def inc(self, amount: float = 1.0) -> None:
        self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        self._value -= amount

    def get(self) -> float:
        return self._value


# =============================================================================
# 2. METRIC DEFINITIONS
# =============================================================================
# Following Prometheus naming conventions:
#   <namespace>_<subsystem>_<name>_<unit>
# e.g., myapp_http_requests_total, myapp_http_request_duration_seconds

REQUEST_COUNT = SimCounter(
    name="myapp_http_requests_total",
    doc="Total HTTP requests by method, endpoint, and status",
    labels=["method", "endpoint", "status"],
)

REQUEST_DURATION = SimHistogram(
    name="myapp_http_request_duration_seconds",
    doc="HTTP request latency in seconds",
    labels=["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

ACTIVE_REQUESTS = SimGauge(
    name="myapp_http_active_requests",
    doc="Number of currently active HTTP requests",
)

DB_POOL_SIZE = SimGauge(
    name="myapp_db_pool_connections",
    doc="Number of connections in the database pool",
)

ERRORS_TOTAL = SimCounter(
    name="myapp_errors_total",
    doc="Total application errors by type",
    labels=["error_type"],
)

BUSINESS_ORDERS = SimCounter(
    name="myapp_orders_total",
    doc="Total orders placed by status",
    labels=["status"],
)


# =============================================================================
# 3. MIDDLEWARE / DECORATOR PATTERN
# =============================================================================

def track_request(method: str, endpoint: str, handler: Callable) -> dict:
    """Simulate request tracking with Prometheus metrics.

    In a real Flask app, this would be a before_request/after_request hook
    or a WSGI middleware.
    """
    ACTIVE_REQUESTS.inc()
    start = time.monotonic()

    try:
        result = handler()
        status = "200"
    except Exception as e:
        status = "500"
        ERRORS_TOTAL.labels(error_type=type(e).__name__).inc()
        result = {"error": str(e)}

    duration = time.monotonic() - start
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    ACTIVE_REQUESTS.dec()

    return {
        "result": result,
        "status": status,
        "duration_ms": round(duration * 1000, 2),
    }


# =============================================================================
# 4. SIMULATED APPLICATION ENDPOINTS
# =============================================================================

def handle_list_items() -> list[str]:
    """Simulate listing items (fast endpoint)."""
    time.sleep(random.uniform(0.001, 0.02))
    return ["item1", "item2", "item3"]


def handle_create_order() -> dict:
    """Simulate creating an order (slower, may fail)."""
    time.sleep(random.uniform(0.05, 0.3))
    if random.random() < 0.1:  # 10% failure rate
        raise RuntimeError("Database connection timeout")
    BUSINESS_ORDERS.labels(status="completed").inc()
    return {"order_id": random.randint(1000, 9999)}


def handle_health_check() -> dict:
    """Simulate health check (instant)."""
    return {"status": "healthy", "db_pool": DB_POOL_SIZE.get()}


# =============================================================================
# 5. PROMETHEUS TEXT FORMAT EXPOSITION
# =============================================================================

def format_prometheus_text() -> str:
    """Format collected metrics in Prometheus text exposition format.

    In production, prometheus_client.generate_latest() does this automatically.
    This is a simplified version for demonstration.
    """
    lines: list[str] = []

    # Request count
    lines.append(f"# HELP {REQUEST_COUNT.name} {REQUEST_COUNT.doc}")
    lines.append(f"# TYPE {REQUEST_COUNT.name} counter")
    for labels, value in REQUEST_COUNT.collect().items():
        label_str = ",".join(f'{k}="{v}"' for k, v in labels)
        lines.append(f"{REQUEST_COUNT.name}{{{label_str}}} {value}")

    lines.append("")

    # Active requests gauge
    lines.append(f"# HELP {ACTIVE_REQUESTS.name} {ACTIVE_REQUESTS.doc}")
    lines.append(f"# TYPE {ACTIVE_REQUESTS.name} gauge")
    lines.append(f"{ACTIVE_REQUESTS.name} {ACTIVE_REQUESTS.get()}")

    lines.append("")

    # Error count
    lines.append(f"# HELP {ERRORS_TOTAL.name} {ERRORS_TOTAL.doc}")
    lines.append(f"# TYPE {ERRORS_TOTAL.name} counter")
    for labels, value in ERRORS_TOTAL.collect().items():
        label_str = ",".join(f'{k}="{v}"' for k, v in labels)
        lines.append(f"{ERRORS_TOTAL.name}{{{label_str}}} {value}")

    lines.append("")

    # Business metrics
    lines.append(f"# HELP {BUSINESS_ORDERS.name} {BUSINESS_ORDERS.doc}")
    lines.append(f"# TYPE {BUSINESS_ORDERS.name} counter")
    for labels, value in BUSINESS_ORDERS.collect().items():
        label_str = ",".join(f'{k}="{v}"' for k, v in labels)
        lines.append(f"{BUSINESS_ORDERS.name}{{{label_str}}} {value}")

    return "\n".join(lines) + "\n"


# =============================================================================
# 6. DEMO: SIMULATE TRAFFIC
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Prometheus Metrics Instrumentation Demo")
    print("=" * 70)
    print()

    # Simulate a database connection pool
    DB_POOL_SIZE.set(10)

    # Simulate 50 requests across different endpoints
    endpoints = [
        ("GET", "/api/items", handle_list_items),
        ("POST", "/api/orders", handle_create_order),
        ("GET", "/health", handle_health_check),
    ]

    # Weight: more reads than writes
    weights = [0.6, 0.3, 0.1]

    print("Simulating 50 requests...")
    for i in range(50):
        method, path, handler = random.choices(endpoints, weights=weights, k=1)[0]
        result = track_request(method, path, handler)
        if result["status"] != "200":
            print(f"  [{i+1:02d}] {method} {path} -> {result['status']} "
                  f"({result['duration_ms']}ms)")

    print()
    print("=" * 70)
    print("Prometheus Text Exposition (/metrics)")
    print("=" * 70)
    print(format_prometheus_text())

    # Summary stats
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    total_requests = sum(REQUEST_COUNT.collect().values())
    total_errors = sum(ERRORS_TOTAL.collect().values())
    total_orders = sum(BUSINESS_ORDERS.collect().values())
    print(f"  Total requests:    {int(total_requests)}")
    print(f"  Total errors:      {int(total_errors)}")
    print(f"  Error rate:        {total_errors/total_requests*100:.1f}%")
    print(f"  Orders completed:  {int(total_orders)}")
    print(f"  Active requests:   {int(ACTIVE_REQUESTS.get())}")
    print(f"  DB pool size:      {int(DB_POOL_SIZE.get())}")
