#!/usr/bin/env python3
"""Example: Advanced Metrics Architecture — Histograms, Aggregation & Federation

Demonstrates advanced metrics patterns: histogram bucketing and quantile
estimation, recording rules for pre-aggregation, metric federation across
clusters, and storage cost modeling.
Related lesson: 22_Advanced_Metrics_Architecture.md
"""

# =============================================================================
# WHY ADVANCED METRICS?
# Basic counters and gauges are a start, but production systems need
# histograms for latency distributions, pre-aggregation to tame query load,
# federation to unify multi-cluster data, and cardinality-aware design to
# keep storage costs manageable.
# =============================================================================

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# 1. HISTOGRAM IMPLEMENTATION
# =============================================================================

# Default Prometheus-style latency buckets (seconds)
DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]


@dataclass
class Histogram:
    """A cumulative histogram with configurable buckets."""
    name: str
    buckets: list[float] = field(default_factory=lambda: DEFAULT_BUCKETS.copy())
    counts: list[int] = field(init=False)
    total_sum: float = 0.0
    total_count: int = 0

    def __post_init__(self):
        self.buckets = sorted(self.buckets) + [float("inf")]
        self.counts = [0] * len(self.buckets)

    def observe(self, value: float) -> None:
        """Record an observation into the histogram."""
        self.total_sum += value
        self.total_count += 1
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[i] += 1

    def quantile(self, q: float) -> float:
        """Estimate a quantile from the histogram (linear interpolation)."""
        if self.total_count == 0:
            return 0.0
        target = q * self.total_count
        for i, count in enumerate(self.counts):
            if count >= target:
                # Linear interpolation within the bucket
                lower = self.buckets[i - 1] if i > 0 else 0.0
                upper = self.buckets[i]
                if upper == float("inf"):
                    return lower
                prev_count = self.counts[i - 1] if i > 0 else 0
                fraction = (target - prev_count) / max(count - prev_count, 1)
                return lower + fraction * (upper - lower)
        return self.buckets[-2]  # Last finite bucket

    def to_prometheus(self) -> str:
        """Format as Prometheus exposition text."""
        lines = []
        for i, bound in enumerate(self.buckets):
            le = "+Inf" if bound == float("inf") else str(bound)
            lines.append(f'{self.name}_bucket{{le="{le}"}} {self.counts[i]}')
        lines.append(f"{self.name}_sum {self.total_sum}")
        lines.append(f"{self.name}_count {self.total_count}")
        return "\n".join(lines)


# =============================================================================
# 2. RECORDING RULES (PRE-AGGREGATION)
# =============================================================================

@dataclass
class RecordingRule:
    """A pre-aggregation recording rule (Prometheus-style)."""
    record: str          # Name of the new metric
    expr: str            # PromQL-like expression (descriptive)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class RecordingRuleGroup:
    """A group of recording rules evaluated together."""
    name: str
    interval_seconds: int = 30
    rules: list[RecordingRule] = field(default_factory=list)

    def to_yaml_dict(self) -> dict:
        return {
            "name": self.name,
            "interval": f"{self.interval_seconds}s",
            "rules": [
                {"record": r.record, "expr": r.expr, "labels": r.labels}
                for r in self.rules
            ],
        }


def generate_slo_recording_rules(service: str, slo_target: float) -> RecordingRuleGroup:
    """Generate standard SLO recording rules for a service."""
    return RecordingRuleGroup(
        name=f"{service}:slo_rules",
        rules=[
            RecordingRule(
                record=f"{service}:http_requests:rate5m",
                expr=f'rate(http_requests_total{{service="{service}"}}[5m])',
            ),
            RecordingRule(
                record=f"{service}:http_errors:rate5m",
                expr=f'rate(http_requests_total{{service="{service}",code=~"5.."}}[5m])',
            ),
            RecordingRule(
                record=f"{service}:availability:ratio5m",
                expr=f'1 - ({service}:http_errors:rate5m / {service}:http_requests:rate5m)',
            ),
            RecordingRule(
                record=f"{service}:error_budget:remaining",
                expr=f'1 - (({service}:http_errors:rate5m / {service}:http_requests:rate5m) / {1 - slo_target})',
            ),
            RecordingRule(
                record=f"{service}:latency:p99_5m",
                expr=f'histogram_quantile(0.99, rate(http_request_duration_bucket{{service="{service}"}}[5m]))',
            ),
        ],
    )


# =============================================================================
# 3. METRIC FEDERATION
# =============================================================================

@dataclass
class MetricStore:
    """Represents a Prometheus-like metric store in a cluster."""
    cluster_name: str
    series: dict[str, list[tuple[float, float]]] = field(default_factory=dict)

    def add(self, metric_name: str, value: float, ts: float | None = None) -> None:
        ts = ts or time.time()
        self.series.setdefault(metric_name, []).append((ts, value))

    def query_latest(self, metric_name: str) -> float | None:
        samples = self.series.get(metric_name)
        if not samples:
            return None
        return samples[-1][1]


@dataclass
class FederationAggregator:
    """Aggregates metrics across multiple cluster stores."""
    stores: list[MetricStore] = field(default_factory=list)

    def federated_query(self, metric_name: str) -> dict[str, float | None]:
        """Query a metric across all federated clusters."""
        return {
            s.cluster_name: s.query_latest(metric_name) for s in self.stores
        }

    def aggregate_sum(self, metric_name: str) -> float:
        values = self.federated_query(metric_name)
        return sum(v for v in values.values() if v is not None)

    def aggregate_avg(self, metric_name: str) -> float:
        values = [v for v in self.federated_query(metric_name).values() if v is not None]
        return sum(values) / len(values) if values else 0.0


# =============================================================================
# 4. STORAGE COST ESTIMATOR
# =============================================================================

def estimate_storage_cost(
    active_series: int,
    scrape_interval_s: int = 15,
    retention_days: int = 30,
    bytes_per_sample: int = 2,  # Compressed (Prometheus TSDB)
) -> dict[str, Any]:
    """Estimate metric storage requirements."""
    samples_per_day = active_series * (86400 / scrape_interval_s)
    total_samples = samples_per_day * retention_days
    storage_bytes = total_samples * bytes_per_sample
    storage_gb = storage_bytes / (1024 ** 3)
    # Rough cost estimate (cloud block storage ~$0.10/GB/month)
    cost_per_month = storage_gb * 0.10
    return {
        "active_series": active_series,
        "samples_per_day": int(samples_per_day),
        "total_samples": int(total_samples),
        "storage_gb": round(storage_gb, 2),
        "estimated_cost_monthly_usd": round(cost_per_month, 2),
    }


# =============================================================================
# 5. DEMO
# =============================================================================

if __name__ == "__main__":
    random.seed(42)

    # --- Histogram ---
    print("=" * 60)
    print("Histogram — HTTP Request Duration")
    print("=" * 60)
    hist = Histogram(name="http_request_duration_seconds")
    for _ in range(10000):
        # Simulate bimodal latency distribution
        if random.random() < 0.95:
            latency = random.gauss(0.05, 0.02)
        else:
            latency = random.gauss(0.5, 0.2)  # Slow requests
        hist.observe(max(0.001, latency))
    print(f"  Total requests: {hist.total_count}")
    print(f"  Mean latency: {hist.total_sum / hist.total_count:.4f}s")
    for q in [0.5, 0.9, 0.95, 0.99]:
        print(f"  p{int(q*100)}: {hist.quantile(q):.4f}s")
    print(f"\n  Prometheus format (first 5 lines):")
    for line in hist.to_prometheus().split("\n")[:5]:
        print(f"    {line}")

    # --- Recording Rules ---
    print(f"\n{'=' * 60}")
    print("SLO Recording Rules")
    print("=" * 60)
    rules = generate_slo_recording_rules("order-svc", 0.999)
    for rule in rules.rules:
        print(f"  {rule.record}")
        print(f"    expr: {rule.expr[:70]}...")

    # --- Federation ---
    print(f"\n{'=' * 60}")
    print("Metric Federation")
    print("=" * 60)
    fed = FederationAggregator()
    for cluster in ["us-east-1", "eu-west-1", "ap-southeast-1"]:
        store = MetricStore(cluster_name=cluster)
        store.add("http_requests_total", random.uniform(1000, 5000))
        store.add("error_rate", random.uniform(0.001, 0.01))
        fed.stores.append(store)
    print(f"  Federated http_requests_total: {fed.federated_query('http_requests_total')}")
    print(f"  Global sum: {fed.aggregate_sum('http_requests_total'):.0f}")
    print(f"  Global avg error_rate: {fed.aggregate_avg('error_rate'):.4f}")

    # --- Storage Cost ---
    print(f"\n{'=' * 60}")
    print("Storage Cost Estimation")
    print("=" * 60)
    for series_count in [10_000, 100_000, 1_000_000]:
        est = estimate_storage_cost(series_count)
        print(f"  {series_count:>10,} series: {est['storage_gb']:>8.1f} GB, "
              f"${est['estimated_cost_monthly_usd']:>8.2f}/month")
