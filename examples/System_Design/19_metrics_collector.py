"""
Metrics Collector

Demonstrates:
- Counter, Gauge, Histogram metric types
- Percentile calculation (P50, P95, P99)
- Sliding window aggregation
- Alert thresholds

Theory:
- Observability requires collecting and analyzing system metrics.
- Counter: monotonically increasing value (e.g., total requests).
- Gauge: point-in-time value (e.g., CPU usage, queue depth).
- Histogram: distribution of values (e.g., response latencies).
- Percentiles (P50, P95, P99): indicate what percentage of
  observations fall below a given value.
- Sliding windows: aggregate metrics over recent time periods
  to detect trends and anomalies.

Adapted from System Design Lesson 19.
"""

import math
import random
import bisect
from collections import deque
from dataclasses import dataclass, field
from typing import Any


# ── Counter ────────────────────────────────────────────────────────────

# Why: Counters are monotonically increasing by design — they only go up. This
# makes them robust to restarts: you compute rates by differencing two samples,
# so even if the counter resets to 0, the monitoring system detects the reset.
class Counter:
    """Monotonically increasing counter."""

    def __init__(self, name: str):
        self.name = name
        self.value: float = 0.0
        self.labels: dict[str, "Counter"] = {}

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount

    def with_labels(self, **labels: str) -> "Counter":
        key = str(sorted(labels.items()))
        if key not in self.labels:
            self.labels[key] = Counter(f"{self.name}{labels}")
        return self.labels[key]

    def rate(self, seconds: float) -> float:
        """Calculate per-second rate over given duration."""
        return self.value / seconds if seconds > 0 else 0.0


# ── Gauge ──────────────────────────────────────────────────────────────

class Gauge:
    """Point-in-time value that can go up or down."""

    def __init__(self, name: str):
        self.name = name
        self.value: float = 0.0
        self.history: list[tuple[float, float]] = []  # (timestamp, value)

    def set(self, value: float, timestamp: float = 0.0) -> None:
        self.value = value
        self.history.append((timestamp, value))

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount

    def dec(self, amount: float = 1.0) -> None:
        self.value -= amount


# ── Histogram ──────────────────────────────────────────────────────────

# Why: Histograms capture the full distribution of values, not just averages.
# Averages hide tail latency — a service with 50ms average might have P99 at 2s.
# Percentiles (P50, P95, P99) reveal the experience of the worst-affected users.
class Histogram:
    """Distribution of observed values with percentile computation."""

    def __init__(self, name: str, buckets: list[float] | None = None):
        self.name = name
        self.values: list[float] = []
        self.count = 0
        self.total = 0.0
        # Why: Pre-defined buckets enable O(1) "what percentage of requests
        # are under Xms?" queries without sorting. Bucket boundaries should
        # match your SLA thresholds (e.g., 100ms, 500ms, 1000ms).
        self.buckets = buckets or [5, 10, 25, 50, 100, 250, 500, 1000]
        self.bucket_counts: dict[float, int] = {b: 0 for b in self.buckets}

    def observe(self, value: float) -> None:
        self.values.append(value)
        self.count += 1
        self.total += value
        for b in self.buckets:
            if value <= b:
                self.bucket_counts[b] += 1

    def percentile(self, p: float) -> float:
        """Calculate the p-th percentile (0-100)."""
        if not self.values:
            return 0.0
        # Why: Sorting on each percentile query is O(n log n) — acceptable for
        # monitoring dashboards that refresh every few seconds. For hot-path use,
        # a streaming approximation (t-digest, DDSketch) would be more efficient.
        sorted_vals = sorted(self.values)
        idx = (p / 100.0) * (len(sorted_vals) - 1)
        lower = int(math.floor(idx))
        upper = int(math.ceil(idx))
        if lower == upper:
            return sorted_vals[lower]
        # Linear interpolation
        frac = idx - lower
        return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def min(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def max(self) -> float:
        return max(self.values) if self.values else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "count": self.count,
            "mean": self.mean,
            "min": self.min,
            "max": self.max,
            "p50": self.percentile(50),
            "p90": self.percentile(90),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
        }


# ── Sliding Window ────────────────────────────────────────────────────

# Why: Sliding windows focus on recent behavior, which is what operators care
# about during incidents. A 5-minute window catches spikes that an all-time
# average would smooth away, enabling timely alerting.
class SlidingWindowMetric:
    """Aggregate metrics over a sliding time window."""

    def __init__(self, window_seconds: float):
        self.window = window_seconds
        self.observations: deque[tuple[float, float]] = deque()

    def add(self, value: float, timestamp: float) -> None:
        self.observations.append((timestamp, value))
        self._prune(timestamp)

    def _prune(self, current_time: float) -> None:
        cutoff = current_time - self.window
        while self.observations and self.observations[0][0] < cutoff:
            self.observations.popleft()

    def stats(self, current_time: float) -> dict[str, float]:
        self._prune(current_time)
        values = [v for _, v in self.observations]
        if not values:
            return {"count": 0, "mean": 0, "min": 0, "max": 0}
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "sum": sum(values),
        }


# ── Alert Rule ─────────────────────────────────────────────────────────

@dataclass
class AlertRule:
    name: str
    metric_name: str
    condition: str  # "gt", "lt", "gte", "lte"
    threshold: float
    window_seconds: float = 60.0
    triggered: bool = False

    def evaluate(self, value: float) -> bool:
        ops = {
            "gt": lambda v, t: v > t,
            "lt": lambda v, t: v < t,
            "gte": lambda v, t: v >= t,
            "lte": lambda v, t: v <= t,
        }
        return ops[self.condition](value, self.threshold)


# ── Metrics Registry ──────────────────────────────────────────────────

# Why: A central registry prevents metric name collisions and provides a single
# point for alert evaluation. This mirrors how Prometheus client libraries work —
# metrics are registered once and shared across the application.
class MetricsRegistry:
    """Central registry for all metrics."""

    def __init__(self):
        self.counters: dict[str, Counter] = {}
        self.gauges: dict[str, Gauge] = {}
        self.histograms: dict[str, Histogram] = {}
        self.alerts: list[AlertRule] = []

    def counter(self, name: str) -> Counter:
        if name not in self.counters:
            self.counters[name] = Counter(name)
        return self.counters[name]

    def gauge(self, name: str) -> Gauge:
        if name not in self.gauges:
            self.gauges[name] = Gauge(name)
        return self.gauges[name]

    def histogram(self, name: str, **kwargs: Any) -> Histogram:
        if name not in self.histograms:
            self.histograms[name] = Histogram(name, **kwargs)
        return self.histograms[name]

    def add_alert(self, rule: AlertRule) -> None:
        self.alerts.append(rule)

    def check_alerts(self) -> list[str]:
        """Check all alert rules and return triggered alerts."""
        triggered = []
        for rule in self.alerts:
            if rule.metric_name in self.histograms:
                h = self.histograms[rule.metric_name]
                value = h.percentile(95)
            elif rule.metric_name in self.gauges:
                value = self.gauges[rule.metric_name].value
            elif rule.metric_name in self.counters:
                value = self.counters[rule.metric_name].value
            else:
                continue

            if rule.evaluate(value):
                rule.triggered = True
                triggered.append(
                    f"ALERT: {rule.name} — {rule.metric_name} "
                    f"{rule.condition} {rule.threshold} (current: {value:.1f})"
                )
        return triggered


# ── Demos ──────────────────────────────────────────────────────────────

def demo_histogram():
    print("=" * 60)
    print("HISTOGRAM & PERCENTILES")
    print("=" * 60)

    h = Histogram("request_duration_ms")

    # Simulate realistic latency distribution
    random.seed(42)
    for _ in range(1000):
        # 90% normal requests (10-100ms)
        if random.random() < 0.9:
            latency = random.gauss(50, 20)
        # 9% slow requests (100-500ms)
        elif random.random() < 0.9:
            latency = random.gauss(200, 50)
        # 1% very slow (500-2000ms)
        else:
            latency = random.gauss(1000, 200)
        h.observe(max(1, latency))

    summary = h.summary()
    print(f"\n  {h.name} ({int(summary['count'])} observations):\n")
    print(f"    {'Metric':<12} {'Value':>10}")
    print(f"    {'-'*12} {'-'*10}")
    for key, val in summary.items():
        if key != "count":
            print(f"    {key:<12} {val:>9.1f}ms")

    # Bucket distribution
    print(f"\n  Bucket distribution:")
    prev = 0
    for bucket in h.buckets:
        count = h.bucket_counts[bucket]
        pct = count / h.count * 100
        bar = "█" * int(pct / 2)
        print(f"    ≤{bucket:>6.0f}ms  {count:>5} ({pct:>5.1f}%) {bar}")


def demo_counter_and_gauge():
    print("\n" + "=" * 60)
    print("COUNTER & GAUGE")
    print("=" * 60)

    # Counter
    req_counter = Counter("http_requests_total")
    err_counter = Counter("http_errors_total")

    random.seed(42)
    for _ in range(1000):
        req_counter.inc()
        if random.random() < 0.05:  # 5% error rate
            err_counter.inc()

    duration = 60.0  # 60 seconds
    print(f"\n  Counters (over {duration:.0f}s):")
    print(f"    Total requests: {req_counter.value:.0f}")
    print(f"    Total errors:   {err_counter.value:.0f}")
    print(f"    Request rate:   {req_counter.rate(duration):.1f}/s")
    print(f"    Error rate:     {err_counter.rate(duration):.1f}/s")
    print(f"    Error ratio:    {err_counter.value/req_counter.value*100:.1f}%")

    # Gauge
    cpu_gauge = Gauge("cpu_usage_percent")
    random.seed(42)
    print(f"\n  Gauge (CPU usage over 10 samples):")
    for t in range(10):
        usage = 30 + random.gauss(0, 15)
        usage = max(0, min(100, usage))
        cpu_gauge.set(usage, timestamp=float(t))
        bar = "█" * int(usage / 2)
        print(f"    t={t:>2}: {usage:>5.1f}% {bar}")


def demo_sliding_window():
    print("\n" + "=" * 60)
    print("SLIDING WINDOW AGGREGATION")
    print("=" * 60)

    sw = SlidingWindowMetric(window_seconds=5.0)

    random.seed(42)
    print(f"\n  5-second sliding window of latencies:\n")
    print(f"  {'Time':>6}  {'Value':>8}  {'Win Count':>10}  {'Win Mean':>10}  {'Win Max':>10}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")

    for t in range(15):
        t_float = float(t)
        value = random.gauss(50, 15)
        sw.add(value, t_float)
        stats = sw.stats(t_float)
        print(f"  {t:>5.0f}s  {value:>7.1f}  {stats['count']:>10.0f}  "
              f"{stats['mean']:>9.1f}  {stats['max']:>9.1f}")


def demo_alerts():
    print("\n" + "=" * 60)
    print("ALERT RULES")
    print("=" * 60)

    registry = MetricsRegistry()

    # Create metrics
    h = registry.histogram("api_latency_ms")
    g = registry.gauge("cpu_percent")

    # Add alert rules
    registry.add_alert(AlertRule(
        name="High P95 Latency",
        metric_name="api_latency_ms",
        condition="gt",
        threshold=200.0,
    ))
    registry.add_alert(AlertRule(
        name="High CPU",
        metric_name="cpu_percent",
        condition="gt",
        threshold=80.0,
    ))

    # Simulate normal metrics
    random.seed(42)
    for _ in range(100):
        h.observe(random.gauss(50, 20))
    g.set(45.0)

    print(f"\n  Normal state:")
    print(f"    P95 latency: {h.percentile(95):.1f}ms")
    print(f"    CPU: {g.value:.1f}%")
    alerts = registry.check_alerts()
    print(f"    Alerts: {alerts if alerts else 'None'}")

    # Simulate degraded state
    for _ in range(50):
        h.observe(random.gauss(300, 100))
    g.set(85.0)

    print(f"\n  Degraded state:")
    print(f"    P95 latency: {h.percentile(95):.1f}ms")
    print(f"    CPU: {g.value:.1f}%")
    alerts = registry.check_alerts()
    for alert in alerts:
        print(f"    {alert}")


if __name__ == "__main__":
    demo_histogram()
    demo_counter_and_gauge()
    demo_sliding_window()
    demo_alerts()
