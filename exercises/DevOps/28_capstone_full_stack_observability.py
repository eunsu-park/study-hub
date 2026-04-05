#!/usr/bin/env python3
"""Exercises for Lesson 28: Capstone - Full-Stack Observability
Topic: DevOps
"""


def exercise_1():
    """Platform design for a startup."""
    print("=== Exercise 1: Observability Platform Design ===\n")

    print("Startup: 10 Python/FastAPI services, 1 K8s cluster, $3K/mo budget\n")

    print("Tool selection: Grafana Cloud Pro (managed, no platform team needed)")
    costs = {
        "Metrics (50K series × $8/1K)": 400,
        "Traces (20GB × $0.50)": 10,
        "Logs (100GB × $0.50)": 50,
        "Profiles (10 services)": 50,
        "OTel Collector infra (3 pods)": 25,
        "PagerDuty (free tier)": 0,
    }
    total = sum(costs.values())
    print("Cost breakdown:")
    for item, cost in costs.items():
        print(f"  {item}: ${cost}")
    print(f"  TOTAL: ${total}/month (well under $3,000)")
    print()

    print("Architecture:")
    print("  Apps (auto-instrumented) → OTel DaemonSet → Grafana Cloud OTLP endpoint")
    print()

    print("SLOs:")
    slos = [
        ("payment-service", "99.9% availability, 99% < 500ms"),
        ("auth-service", "99.99% availability, 99% < 200ms"),
        ("order-service", "99.9% availability, 95% < 1000ms"),
    ]
    for svc, slo in slos:
        print(f"  {svc}: {slo}")


def exercise_2():
    """Incident simulation walkthrough."""
    print("\n=== Exercise 2: Incident Simulation ===\n")

    timeline = [
        ("03:00", "Checkout SLO burn rate alert fires (14.4x)"),
        ("03:00", "OrderService error rate 15%, InventoryService latency 5s"),
        ("03:02", "On-call acknowledges, declares SEV1"),
        ("03:05", "Exemplar → trace: order → inventory [TIMEOUT 5s]"),
        ("03:08", "Inventory DB connections: 50/50 EXHAUSTED"),
        ("03:10", "Logs: 'Lock wait timeout exceeded'"),
        ("03:12", "pg_stat_activity: ALTER TABLE running 2h (holding lock)"),
        ("03:15", "Kill ALTER TABLE query → connections resume"),
        ("03:20", "Error rate drops to 0%, SLO recovering"),
    ]
    print("Timeline:")
    for time, event in timeline:
        print(f"  {time} - {event}")

    print("\nRoot cause: Cron job at 03:00 ran ALTER TABLE ADD COLUMN")
    print("which acquired ACCESS EXCLUSIVE lock, blocking all reads.")
    print()

    print("Action items:")
    actions = [
        ("P0", "Use lock-free ALTER TABLE (PG 11+: ADD COLUMN DEFAULT NULL)"),
        ("P0", "Add PostgreSQL lock wait monitoring (alert if > 30s)"),
        ("P1", "Schedule migrations during maintenance windows"),
        ("P1", "Add application-level query timeout (5s max)"),
        ("P2", "Add circuit breaker between order-service and inventory"),
        ("P1", "Require migration review checklist (lock analysis)"),
    ]
    for priority, action in actions:
        print(f"  [{priority}] {action}")


def exercise_3():
    """Cost optimization plan."""
    print("\n=== Exercise 3: Cost Optimization ===\n")
    print("Current: $27,050/month. Target: $15,000/month.\n")

    optimizations = [
        ("Drop unused metrics (30%)", "Metrics", 7200),
        ("Reduce histogram buckets (10%)", "Metrics", 2400),
        ("Pre-aggregate per-pod → per-deployment (10%)", "Metrics", 2400),
        ("Increase trace sampling to 5%", "Traces", 25),
        ("Filter DEBUG/TRACE logs", "Logs", 500),
        ("Right-size Collector pods", "Infra", 400),
    ]
    total_saved = 0
    print(f"{'Action':<50} {'Signal':<10} {'Savings':>8}")
    print("-" * 70)
    for action, signal, savings in optimizations:
        total_saved += savings
        print(f"{action:<50} {signal:<10} ${savings:>7,}")
    print("-" * 70)
    print(f"{'Total savings':<50} {'':10} ${total_saved:>7,}")
    print(f"{'New monthly cost':<50} {'':10} ${27050 - total_saved:>7,}")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
