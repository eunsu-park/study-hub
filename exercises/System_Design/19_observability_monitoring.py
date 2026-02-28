"""
Exercises for Lesson 19: Observability and Monitoring
Topic: System_Design

Solutions to practice problems from the lesson.
Covers monitoring design, alerting strategy, and cost optimization.
"""

import time
import random
import math
from collections import defaultdict
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# === Exercise 1: Design Monitoring for Microservices Platform ===
# Problem: Design observability stack for e-commerce platform with 20 microservices.

@dataclass
class MetricPoint:
    name: str
    value: float
    labels: Dict[str, str]
    timestamp: float


class REDMetricsCollector:
    """Collects RED metrics (Rate, Errors, Duration) per service."""

    def __init__(self):
        self.metrics = defaultdict(list)

    def record_request(self, service, method, status, duration_ms):
        """Record a request metric."""
        ts = time.time()
        labels = {"service": service, "method": method, "status": str(status)}

        # Rate
        self.metrics[f"http_requests_total"].append(
            MetricPoint("http_requests_total", 1, labels, ts)
        )

        # Duration
        self.metrics[f"http_request_duration_ms"].append(
            MetricPoint("http_request_duration_ms", duration_ms, labels, ts)
        )

    def get_rate(self, service, window_seconds=60):
        """Get request rate for a service."""
        now = time.time()
        cutoff = now - window_seconds
        count = sum(
            1 for m in self.metrics["http_requests_total"]
            if m.labels["service"] == service and m.timestamp > cutoff
        )
        return count / window_seconds

    def get_error_rate(self, service, window_seconds=60):
        """Get error rate for a service."""
        now = time.time()
        cutoff = now - window_seconds
        total = sum(
            1 for m in self.metrics["http_requests_total"]
            if m.labels["service"] == service and m.timestamp > cutoff
        )
        errors = sum(
            1 for m in self.metrics["http_requests_total"]
            if m.labels["service"] == service
            and m.timestamp > cutoff
            and m.labels["status"].startswith("5")
        )
        return errors / total if total > 0 else 0

    def get_duration_percentiles(self, service, window_seconds=60):
        """Get duration percentiles for a service."""
        now = time.time()
        cutoff = now - window_seconds
        durations = sorted([
            m.value for m in self.metrics["http_request_duration_ms"]
            if m.labels["service"] == service and m.timestamp > cutoff
        ])
        if not durations:
            return {"p50": 0, "p95": 0, "p99": 0}

        def percentile(data, p):
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data) - 1)]

        return {
            "p50": percentile(durations, 50),
            "p95": percentile(durations, 95),
            "p99": percentile(durations, 99),
        }


@dataclass
class SLO:
    name: str
    target: float
    window_days: int
    metric_type: str  # "availability" or "latency"
    budget_remaining: float = 1.0

    def check(self, current_value):
        if self.metric_type == "availability":
            return current_value >= self.target
        elif self.metric_type == "latency":
            return current_value <= self.target


def exercise_1():
    """Monitoring design for microservices platform."""
    print("Monitoring Design: E-Commerce Platform (20 services):")
    print("=" * 60)

    collector = REDMetricsCollector()

    # Simulate traffic for checkout service
    random.seed(42)
    services = ["checkout", "payment", "inventory", "user-auth", "catalog"]

    for _ in range(500):
        service = random.choice(services)
        method = random.choice(["GET", "POST"])
        # Simulate latency (checkout is slower)
        base_latency = 200 if service == "checkout" else 50
        latency = base_latency + random.gauss(0, base_latency * 0.3)
        # Simulate errors (payment has higher error rate)
        error_rate = 0.05 if service == "payment" else 0.01
        status = 500 if random.random() < error_rate else 200
        collector.record_request(service, method, status, max(1, latency))

    print("\n  RED Metrics Summary:")
    print(f"  {'Service':<15} {'Rate (rps)':>12} {'Error Rate':>12} "
          f"{'p50 (ms)':>10} {'p99 (ms)':>10}")
    print("  " + "-" * 65)

    for service in services:
        rate = collector.get_rate(service, window_seconds=9999)
        error_rate = collector.get_error_rate(service, window_seconds=9999)
        percentiles = collector.get_duration_percentiles(service, window_seconds=9999)
        print(f"  {service:<15} {rate:>12.2f} {error_rate:>11.1%} "
              f"{percentiles['p50']:>10.0f} {percentiles['p99']:>10.0f}")

    # SLO definitions
    print("\n  Checkout Service SLOs:")
    slos = [
        SLO("Availability", 0.9995, 30, "availability"),
        SLO("Latency p99", 2000, 30, "latency"),
        SLO("Latency p50", 500, 30, "latency"),
    ]

    checkout_error_rate = collector.get_error_rate("checkout", window_seconds=9999)
    checkout_p99 = collector.get_duration_percentiles("checkout", window_seconds=9999)["p99"]
    checkout_p50 = collector.get_duration_percentiles("checkout", window_seconds=9999)["p50"]

    current_values = [1 - checkout_error_rate, checkout_p99, checkout_p50]
    for slo, current in zip(slos, current_values):
        passed = slo.check(current)
        status = "PASS" if passed else "FAIL"
        print(f"    {slo.name}: target={slo.target}, current={current:.4f} [{status}]")

    # Logging strategy
    print("\n  Logging Strategy:")
    print("    - Structured JSON with trace_id in every log line")
    print("    - Centralized via Grafana Loki or Elasticsearch")
    print("    - Retention: 7d hot, 30d warm, 90d cold (S3)")

    # Tracing strategy
    print("\n  Tracing Strategy:")
    print("    - Tail-based sampling: keep all errors + p99 latencies")
    print("    - Head-based: 10% sample for normal traffic")
    print("    - Backend: Jaeger or Grafana Tempo")
    print(f"    - Error budget: ~{0.05 * 30 * 24 * 60:.1f} min/month at 99.95% SLO")


# === Exercise 2: Alert Design ===
# Problem: Design alerting that avoids alert fatigue.

@dataclass
class Alert:
    name: str
    severity: str  # "critical", "warning", "info"
    condition: str
    runbook_url: str
    routing: str
    cooldown_minutes: int = 10


class AlertManager:
    """Alert manager with multi-burn-rate alerting."""

    def __init__(self, slo_target=0.9995, window_days=30):
        self.slo_target = slo_target
        self.error_budget = 1 - slo_target  # 0.05%
        self.window_seconds = window_days * 86400
        self.alerts_fired = []

    def check_burn_rate(self, error_rate_1h, error_rate_6h, error_rate_3d):
        """Multi-burn-rate alerting."""
        budget_per_hour = self.error_budget / (30 * 24)  # Per hour budget

        # Fast burn (critical): 2% budget in 1 hour
        if error_rate_1h > budget_per_hour * (30 * 24 * 0.02):
            return Alert(
                "High error rate (fast burn)",
                "critical",
                f"2% error budget consumed in 1 hour (rate={error_rate_1h:.4f})",
                "https://runbooks.internal/checkout-errors",
                "PagerDuty -> on-call engineer",
                cooldown_minutes=5
            )

        # Medium burn (warning): 5% budget in 6 hours
        if error_rate_6h > budget_per_hour * (30 * 24 * 0.05) / 6:
            return Alert(
                "Elevated error rate (medium burn)",
                "warning",
                f"5% error budget consumed in 6 hours (rate={error_rate_6h:.4f})",
                "https://runbooks.internal/checkout-errors",
                "Slack #alerts -> team lead",
                cooldown_minutes=30
            )

        # Slow burn (info): 10% budget in 3 days
        if error_rate_3d > budget_per_hour * (30 * 24 * 0.10) / 72:
            return Alert(
                "Sustained error rate (slow burn)",
                "info",
                f"10% error budget consumed in 3 days (rate={error_rate_3d:.4f})",
                "https://runbooks.internal/checkout-errors",
                "Jira ticket -> backlog",
                cooldown_minutes=60
            )

        return None


def exercise_2():
    """Alert design to avoid alert fatigue."""
    print("Alert Design (Multi-Burn-Rate):")
    print("=" * 60)

    manager = AlertManager(slo_target=0.9995)

    # Simulate different scenarios
    scenarios = [
        {
            "name": "Major outage (sudden spike)",
            "error_rate_1h": 0.05,
            "error_rate_6h": 0.02,
            "error_rate_3d": 0.005,
        },
        {
            "name": "Degraded service (gradual)",
            "error_rate_1h": 0.002,
            "error_rate_6h": 0.003,
            "error_rate_3d": 0.002,
        },
        {
            "name": "Normal operation",
            "error_rate_1h": 0.0003,
            "error_rate_6h": 0.0003,
            "error_rate_3d": 0.0002,
        },
        {
            "name": "Slow degradation",
            "error_rate_1h": 0.0005,
            "error_rate_6h": 0.0005,
            "error_rate_3d": 0.001,
        },
    ]

    for scenario in scenarios:
        alert = manager.check_burn_rate(
            scenario["error_rate_1h"],
            scenario["error_rate_6h"],
            scenario["error_rate_3d"],
        )
        print(f"\n  Scenario: {scenario['name']}")
        print(f"    Error rates: 1h={scenario['error_rate_1h']:.4f}, "
              f"6h={scenario['error_rate_6h']:.4f}, "
              f"3d={scenario['error_rate_3d']:.4f}")
        if alert:
            print(f"    ALERT: [{alert.severity.upper()}] {alert.name}")
            print(f"    Routing: {alert.routing}")
            print(f"    Runbook: {alert.runbook_url}")
        else:
            print(f"    No alert (within budget)")

    print("\n  Alert Requirements:")
    print("    Every alert MUST have:")
    print("      - Runbook link (step-by-step remediation)")
    print("      - Dashboard link (visualization)")
    print("      - Expected impact description")
    print("      - Suggested first response action")


# === Exercise 3: Observability Cost Optimization ===
# Problem: Reduce $50K/month observability cost by 40%.

def exercise_3():
    """Observability cost optimization."""
    print("Observability Cost Optimization:")
    print("=" * 60)

    # Current costs
    current_costs = {
        "Metrics (Datadog)": 15000,
        "Logs (ELK Cloud)": 20000,
        "Traces (Datadog APM)": 10000,
        "Infrastructure": 5000,
    }

    total_current = sum(current_costs.values())
    target_savings = 0.40
    target_cost = total_current * (1 - target_savings)

    print(f"\n  Current monthly costs: ${total_current:,}")
    for item, cost in current_costs.items():
        print(f"    {item}: ${cost:,}")

    print(f"\n  Target: Reduce by {target_savings:.0%} to ${target_cost:,.0f}")

    # Optimization strategies
    optimizations = [
        {
            "area": "Metrics",
            "current": 15000,
            "actions": [
                "Drop unused metrics (audit dashboards): -30%",
                "Reduce cardinality (fewer label values): -15%",
                "Increase scrape interval for non-critical (15s->60s): -10%",
            ],
            "new_cost": 15000 * 0.45,  # 55% reduction
        },
        {
            "area": "Logs",
            "current": 20000,
            "actions": [
                "Switch from ELK Cloud to Grafana Loki: -50%",
                "Reduce log verbosity (DEBUG->INFO in prod): -20%",
                "Shorter retention (90d->30d for non-regulated): -15%",
                "Stop full-text indexing: -10%",
            ],
            "new_cost": 20000 * 0.30,  # 70% reduction
        },
        {
            "area": "Traces",
            "current": 10000,
            "actions": [
                "Tail-based sampling (keep 100% errors, 1% success): -70%",
                "Use Grafana Tempo instead of Datadog APM: -30%",
            ],
            "new_cost": 10000 * 0.25,  # 75% reduction
        },
        {
            "area": "Infrastructure",
            "current": 5000,
            "actions": [
                "Self-host OpenTelemetry Collector: -20%",
                "Use S3 for cold storage: -15%",
                "Aggregate metrics at collector level: -10%",
            ],
            "new_cost": 5000 * 0.65,  # 35% reduction
        },
    ]

    total_new = 0
    print("\n  Optimization Plan:")
    print("  " + "=" * 58)
    for opt in optimizations:
        savings = opt["current"] - opt["new_cost"]
        pct = savings / opt["current"]
        total_new += opt["new_cost"]
        print(f"\n  {opt['area']} (${opt['current']:,.0f} -> ${opt['new_cost']:,.0f}, "
              f"save {pct:.0%})")
        for action in opt["actions"]:
            print(f"    - {action}")

    total_saved = total_current - total_new
    actual_savings_pct = total_saved / total_current

    print(f"\n  Summary:")
    print(f"  {'':>20} {'Before':>10} {'After':>10} {'Saved':>10}")
    print("  " + "-" * 55)
    for opt in optimizations:
        saved = opt["current"] - opt["new_cost"]
        print(f"  {opt['area']:>20} ${opt['current']:>8,.0f} ${opt['new_cost']:>8,.0f} "
              f"${saved:>8,.0f}")
    print("  " + "-" * 55)
    print(f"  {'TOTAL':>20} ${total_current:>8,} ${total_new:>8,.0f} "
          f"${total_saved:>8,.0f}")
    print(f"\n  Savings: {actual_savings_pct:.0%} "
          f"({'MEETS' if actual_savings_pct >= target_savings else 'MISSES'} "
          f"{target_savings:.0%} target)")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Monitoring Design ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Alert Design ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Cost Optimization ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
