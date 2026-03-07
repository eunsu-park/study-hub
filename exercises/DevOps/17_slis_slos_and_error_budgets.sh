#!/bin/bash
# Exercises for Lesson 17: SLIs, SLOs, and Error Budgets
# Topic: DevOps
# Solutions to practice problems from the lesson.

# === Exercise 1: Define SLIs for a Service ===
# Problem: Define appropriate SLIs for an e-commerce checkout service,
# choosing the right measurement method for each.
exercise_1() {
    echo "=== Exercise 1: Define SLIs for a Service ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
sli_definitions = {
    "Availability SLI": {
        "definition": "Proportion of requests that return non-5xx status codes",
        "formula": "good_requests / total_requests",
        "measurement": "Server-side (load balancer access logs or Prometheus counter)",
        "good_event": "HTTP status 2xx or 3xx or 4xx (client errors are not our fault)",
        "bad_event": "HTTP status 5xx",
        "example_query": 'sum(rate(http_requests_total{status!~"5.."}[5m])) / sum(rate(http_requests_total[5m]))',
    },
    "Latency SLI": {
        "definition": "Proportion of requests faster than a threshold",
        "formula": "requests_below_threshold / total_requests",
        "measurement": "Server-side histogram (request duration)",
        "good_event": "Response time < 300ms",
        "bad_event": "Response time >= 300ms",
        "example_query": 'histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))',
    },
    "Correctness SLI": {
        "definition": "Proportion of checkout operations that produce correct results",
        "formula": "correct_checkouts / total_checkouts",
        "measurement": "Application-level metric (reconciliation check)",
        "good_event": "Order total matches cart total + tax + shipping",
        "bad_event": "Mismatch detected in order reconciliation",
        "example_query": 'sum(rate(checkout_correct_total[5m])) / sum(rate(checkout_total[5m]))',
    },
    "Freshness SLI": {
        "definition": "Proportion of data reads that return sufficiently recent data",
        "formula": "fresh_reads / total_reads",
        "measurement": "Probe that checks data age against threshold",
        "good_event": "Product price updated within last 60 seconds",
        "bad_event": "Stale price served (cache > 60s old)",
        "example_query": 'sum(rate(data_reads_fresh_total[5m])) / sum(rate(data_reads_total[5m]))',
    },
}

for name, sli in sli_definitions.items():
    print(f"\n{name}")
    print(f"  Definition:  {sli['definition']}")
    print(f"  Formula:     {sli['formula']}")
    print(f"  Good event:  {sli['good_event']}")
    print(f"  Bad event:   {sli['bad_event']}")
    print(f"  Measurement: {sli['measurement']}")

# SLI specification best practices:
# 1. Use ratios (good/total), not absolute counts
# 2. Measure at the boundary closest to the user (LB > app > DB)
# 3. Exclude maintenance windows and planned downtime
# 4. Count client errors (4xx) as good events (user's fault, not ours)
SOLUTION
}

# === Exercise 2: Error Budget Calculation ===
# Problem: Calculate error budget for a 99.9% SLO over 30 days
# and determine if the team can ship new features.
exercise_2() {
    echo "=== Exercise 2: Error Budget Calculation ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
def calculate_error_budget(
    slo_target: float,
    window_days: int,
    total_requests: int,
    failed_requests: int,
) -> dict:
    """Calculate error budget status."""
    error_budget_ratio = 1.0 - slo_target
    budget_total = total_requests * error_budget_ratio
    budget_consumed = failed_requests
    budget_remaining = max(0, budget_total - budget_consumed)
    budget_remaining_pct = budget_remaining / budget_total * 100 if budget_total > 0 else 0

    current_sli = 1.0 - (failed_requests / total_requests) if total_requests > 0 else 1.0

    # Time-based budget (in minutes)
    total_minutes = window_days * 24 * 60
    allowed_downtime_min = total_minutes * error_budget_ratio
    consumed_downtime_min = allowed_downtime_min * (budget_consumed / budget_total) if budget_total > 0 else 0
    remaining_downtime_min = allowed_downtime_min - consumed_downtime_min

    return {
        "slo_target": f"{slo_target:.3%}",
        "current_sli": f"{current_sli:.4%}",
        "slo_met": current_sli >= slo_target,
        "budget_total_requests": int(budget_total),
        "budget_consumed_requests": budget_consumed,
        "budget_remaining_pct": f"{budget_remaining_pct:.1f}%",
        "allowed_downtime_min": f"{allowed_downtime_min:.1f}",
        "remaining_downtime_min": f"{remaining_downtime_min:.1f}",
    }

# Scenario 1: Healthy service
healthy = calculate_error_budget(
    slo_target=0.999,
    window_days=30,
    total_requests=10_000_000,
    failed_requests=3_000,
)

# Scenario 2: Service with recent incident
incident = calculate_error_budget(
    slo_target=0.999,
    window_days=30,
    total_requests=10_000_000,
    failed_requests=9_500,
)

for name, result in [("Healthy Service", healthy), ("Post-Incident", incident)]:
    print(f"\n{name}:")
    for key, val in result.items():
        print(f"  {key:30s}: {val}")

# Error budget policy decision:
print("\nError Budget Policy:")
print("  Budget > 50%: Ship features at normal velocity")
print("  Budget 20-50%: Prioritize reliability work")
print("  Budget < 20%: Feature freeze, focus on stability")
print("  Budget 0%: Hard freeze, all hands on reliability")

# SLO examples for different service tiers:
print("\nCommon SLO Targets:")
print("  99.99% (4 nines): 4.3 min/month downtime  -> Payment, auth")
print("  99.9%  (3 nines): 43.2 min/month downtime  -> Core API")
print("  99.5%            : 3.6 hr/month downtime    -> Internal tools")
print("  99.0%            : 7.2 hr/month downtime    -> Batch processing")
SOLUTION
}

# === Exercise 3: Burn Rate Alerting ===
# Problem: Implement multi-window burn rate alerting following
# Google SRE best practices.
exercise_3() {
    echo "=== Exercise 3: Burn Rate Alerting ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Multi-window, multi-burn-rate alerting for a 30-day SLO window

# Burn rate = observed error rate / SLO error budget rate
# If SLO = 99.9%, error budget rate = 0.1%
# If observed error rate = 1.44%, burn rate = 1.44/0.1 = 14.4x
# At 14.4x, you'll burn through 30 days of budget in ~2 days

# Prometheus alerting rules:

alert_rules = [
    {
        "name": "ErrorBudgetFastBurn",
        "severity": "critical",
        "short_window": "5m",
        "long_window": "1h",
        "burn_rate": 14.4,
        "budget_consumed_pct": "2% in 1 hour",
        "action": "Page on-call immediately",
        "promql": """
- alert: ErrorBudgetFastBurn
  expr: |
    (
      sum(rate(http_requests_total{status=~"5.."}[5m]))
      / sum(rate(http_requests_total[5m]))
    ) / 0.001 > 14.4
    AND
    (
      sum(rate(http_requests_total{status=~"5.."}[1h]))
      / sum(rate(http_requests_total[1h]))
    ) / 0.001 > 14.4
  for: 2m
  labels:
    severity: critical
""",
    },
    {
        "name": "ErrorBudgetSlowBurn",
        "severity": "critical",
        "short_window": "30m",
        "long_window": "6h",
        "burn_rate": 6.0,
        "budget_consumed_pct": "5% in 6 hours",
        "action": "Page on-call",
        "promql": "(similar pattern with 30m/6h windows and 6.0x threshold)",
    },
    {
        "name": "ErrorBudgetSteadyBurn",
        "severity": "warning",
        "short_window": "2h",
        "long_window": "1d",
        "burn_rate": 3.0,
        "budget_consumed_pct": "10% in 1 day",
        "action": "Create ticket, investigate during business hours",
        "promql": "(similar pattern with 2h/1d windows and 3.0x threshold)",
    },
    {
        "name": "ErrorBudgetSlowLeak",
        "severity": "warning",
        "short_window": "6h",
        "long_window": "3d",
        "burn_rate": 1.0,
        "budget_consumed_pct": "10% in 10 days",
        "action": "Add to sprint backlog",
        "promql": "(similar pattern with 6h/3d windows and 1.0x threshold)",
    },
]

print("Multi-Window Burn Rate Alerts (30-day SLO: 99.9%)")
print(f"{'Alert':<25} {'Burn Rate':>10} {'Short':>7} {'Long':>7} {'Budget Consumed':>20} {'Action':<25}")
print("-" * 100)
for alert in alert_rules:
    print(f"{alert['name']:<25} {alert['burn_rate']:>8.1f}x "
          f"{alert['short_window']:>7} {alert['long_window']:>7} "
          f"{alert['budget_consumed_pct']:>20} {alert['action']}")

# Why multi-window?
# Short window alone: too noisy (fires on brief spikes)
# Long window alone: too slow (misses fast outages)
# Both must exceed threshold: reduces false positives significantly
SOLUTION
}

# === Exercise 4: SLO Document Template ===
# Problem: Write an SLO document for a service following Google SRE template.
exercise_4() {
    echo "=== Exercise 4: SLO Document Template ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
slo_document = {
    "service": "checkout-api",
    "owner": "Commerce Team",
    "last_reviewed": "2025-01-15",
    "review_cadence": "Quarterly",

    "slos": [
        {
            "name": "Checkout Availability",
            "sli": "Proportion of checkout requests returning non-5xx responses",
            "target": "99.95%",
            "window": "30 days, rolling",
            "measurement": "Load balancer logs (status code)",
            "error_budget": "~2,160 errors per 4.32M monthly requests",
            "alert": "Burn rate > 6x for 30 minutes -> page on-call",
        },
        {
            "name": "Checkout Latency",
            "sli": "Proportion of checkout requests completing within 2 seconds",
            "target": "99%",
            "window": "30 days, rolling",
            "measurement": "Server-side histogram (request duration)",
            "error_budget": "~43,200 slow requests per 4.32M monthly",
            "alert": "p99 latency > 3s for 5 minutes -> page on-call",
        },
    ],

    "error_budget_policy": {
        "budget_remaining > 50%": "Normal feature velocity",
        "budget_remaining 20-50%": "Prioritize reliability, limit risky changes",
        "budget_remaining < 20%": "Feature freeze, all effort on reliability",
        "budget_exhausted": "Hard freeze, daily SLO review, escalate to VP",
    },

    "exclusions": [
        "Planned maintenance windows (announced 48h in advance)",
        "Load testing traffic (tagged with x-load-test header)",
        "Client errors (4xx are not counted as SLO violations)",
    ],
}

print(f"SLO Document: {slo_document['service']}")
print(f"Owner: {slo_document['owner']}")
print(f"Review: {slo_document['review_cadence']}")
print()
for slo in slo_document["slos"]:
    print(f"  SLO: {slo['name']}")
    print(f"    Target: {slo['target']} over {slo['window']}")
    print(f"    SLI:    {slo['sli']}")
    print(f"    Budget: {slo['error_budget']}")
    print()
print("Error Budget Policy:")
for condition, action in slo_document["error_budget_policy"].items():
    print(f"  {condition}: {action}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 17: SLIs, SLOs, and Error Budgets"
echo "================================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
