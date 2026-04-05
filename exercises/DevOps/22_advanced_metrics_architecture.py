#!/usr/bin/env python3
"""Exercises for Lesson 22: Advanced Metrics Architecture
Topic: DevOps
"""


def exercise_1():
    """Architecture design: cardinality estimation and tool selection."""
    print("=== Exercise 1: Metrics Architecture Design ===\n")
    services = 200
    metrics_per_service = 500
    clusters = 3
    label_combos = 5
    estimated_series = services * metrics_per_service * clusters * label_combos
    print(f"Cardinality estimate: {services} × {metrics_per_service} × {clusters} × {label_combos}")
    print(f"  = {estimated_series:,} active time series")
    print(f"  With recording rules: ~{int(estimated_series * 1.33):,}")
    print()
    print("Architecture choice: Thanos")
    print("  - Minimal disruption to existing Prometheus")
    print("  - Sidecar model, no config changes")
    print("  - S3 object storage for cost-effective retention")
    print("  - Compactor: raw=14d, 5m=90d, 1h=365d")


def exercise_2():
    """Cardinality explosion diagnosis."""
    print("\n=== Exercise 2: Cardinality Explosion ===\n")
    labels = {
        "method": 5,
        "path": 1000,
        "status": 20,
        "trace_id": "∞",
        "user_agent": 500,
        "source_ip": 10000,
    }
    print("Original labels:")
    for label, card in labels.items():
        print(f"  {label}: {card}")
    print(f"\n  Theoretical max: ∞ (trace_id is unbounded)")
    print(f"  Without trace_id: 5 × 1000 × 20 × 500 × 10000 = 500,000,000,000")
    print()
    print("Fix:")
    print("  REMOVE: trace_id, source_ip, user_agent")
    print("  MODIFY: path → route templates (~30), status → status_class (5)")
    fixed = 5 * 30 * 5
    print(f"\n  Fixed cardinality: 5 × 30 × 5 = {fixed}")


def exercise_3():
    """Recording rules for e-commerce platform."""
    print("\n=== Exercise 3: Recording Rules ===\n")
    rules = [
        ("job:http_requests:rate5m", "sum by (job) (rate(http_requests_total[5m]))"),
        ("job:http_errors:ratio_rate5m", "error_rate / total_rate"),
        ("job:http_request_duration:p99_5m", "histogram_quantile(0.99, ...)"),
        ("instance:node_cpu:ratio_rate5m", "1 - avg by (instance) (rate(idle[5m]))"),
        ("business:orders:rate1m", "sum(rate(orders_created_total[1m])) * 60"),
        ("business:payment_success:ratio_rate5m", "success / total"),
    ]
    for name, expr in rules:
        print(f"  - record: {name}")
        print(f"    expr: {expr}")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
