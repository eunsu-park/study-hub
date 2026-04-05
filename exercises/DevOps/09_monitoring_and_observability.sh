#!/bin/bash
# Exercises for Lesson 09: Monitoring and Observability
# Topic: DevOps
# Solutions to practice problems from the lesson.

# === Exercise 1: Three Pillars of Observability ===
# Problem: Explain the three pillars (metrics, logs, traces) and when
# each is most useful for debugging a production issue.
exercise_1() {
    echo "=== Exercise 1: Three Pillars of Observability ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
three_pillars = {
    "Metrics": {
        "what": "Numeric time-series data aggregated over intervals",
        "examples": ["CPU utilization", "Request rate", "Error count", "p99 latency"],
        "tools": ["Prometheus", "Datadog", "CloudWatch", "Grafana"],
        "best_for": "Detecting THAT something is wrong (alerting)",
        "format": "metric_name{label=value} value timestamp",
        "storage": "Time-series DB (efficient, compact, long retention)",
        "limitation": "Low cardinality — cannot drill into individual requests",
    },
    "Logs": {
        "what": "Timestamped text records of discrete events",
        "examples": ["Request received", "Query executed", "Error occurred"],
        "tools": ["ELK Stack", "Loki + Grafana", "Splunk", "CloudWatch Logs"],
        "best_for": "Understanding WHAT happened (investigation)",
        "format": "JSON lines: {timestamp, level, message, correlation_id, ...}",
        "storage": "Log aggregation system (expensive at scale, retention matters)",
        "limitation": "High volume — need structured format + indexing to be useful",
    },
    "Traces": {
        "what": "End-to-end request flow across services with timing",
        "examples": ["API -> auth -> DB -> cache -> response"],
        "tools": ["Jaeger", "Zipkin", "Tempo", "Datadog APM", "AWS X-Ray"],
        "best_for": "Finding WHERE the bottleneck is (distributed debugging)",
        "format": "Spans with trace_id, span_id, parent_id, duration",
        "storage": "Trace backend (sampling required — typically 1-10% of traffic)",
        "limitation": "Sampling means you may miss rare issues",
    },
}

for pillar, details in three_pillars.items():
    print(f"\n{pillar}")
    print(f"  What:       {details['what']}")
    print(f"  Best for:   {details['best_for']}")
    print(f"  Tools:      {', '.join(details['tools'])}")
    print(f"  Limitation: {details['limitation']}")

# Debugging workflow:
# 1. Alert fires (METRICS) -> "Error rate > 5% on order-api"
# 2. Check dashboard (METRICS) -> See spike at 14:32 UTC
# 3. Search logs (LOGS) -> Filter by time window + error level
#    -> "Database connection timeout" in order-api
# 4. Trace a request (TRACES) -> See 3s delay in DB span
#    -> Root cause: connection pool exhausted
SOLUTION
}

# === Exercise 2: Prometheus PromQL Queries ===
# Problem: Write PromQL queries for common monitoring scenarios.
exercise_2() {
    echo "=== Exercise 2: Prometheus PromQL Queries ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
promql_queries = {
    "Request rate (per second)": {
        "query": 'rate(http_requests_total[5m])',
        "explanation": "Average requests/sec over last 5 minutes",
    },
    "Error rate (percentage)": {
        "query": 'sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) * 100',
        "explanation": "5xx errors as % of total requests",
    },
    "p99 latency": {
        "query": 'histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))',
        "explanation": "99th percentile request duration from histogram",
    },
    "Top 5 endpoints by error count": {
        "query": 'topk(5, sum by (endpoint) (rate(http_requests_total{status=~"5.."}[1h])))',
        "explanation": "Highest error-producing endpoints in the last hour",
    },
    "CPU usage by pod": {
        "query": 'sum(rate(container_cpu_usage_seconds_total{namespace="production"}[5m])) by (pod)',
        "explanation": "CPU cores used per pod in production namespace",
    },
    "Memory usage percentage": {
        "query": 'container_memory_working_set_bytes / container_spec_memory_limit_bytes * 100',
        "explanation": "Memory used as % of limit (OOM risk if near 100%)",
    },
    "Availability (last 30 days)": {
        "query": '1 - (sum(increase(http_requests_total{status=~"5.."}[30d])) / sum(increase(http_requests_total[30d])))',
        "explanation": "Success ratio over 30 days (SLI for availability SLO)",
    },
}

for name, details in promql_queries.items():
    print(f"\n{name}:")
    print(f"  Query: {details['query']}")
    print(f"  Note:  {details['explanation']}")

# PromQL key concepts:
# rate()     — per-second rate of counter increase (use for counters only)
# increase() — total increase in counter value over time window
# sum by ()  — aggregate across labels (GROUP BY equivalent)
# histogram_quantile() — compute percentiles from histogram buckets
# topk()     — top N results by value
SOLUTION
}

# === Exercise 3: Alerting Rules ===
# Problem: Design alerting rules following best practices: severity levels,
# meaningful descriptions, and avoiding alert fatigue.
exercise_3() {
    echo "=== Exercise 3: Alerting Rules ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Prometheus alerting rules (alerting_rules.yml)
groups:
  - name: sla_alerts
    rules:
      # CRITICAL: Page the on-call engineer immediately
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m]))
          / sum(rate(http_requests_total[5m])) > 0.05
        for: 2m              # Must persist for 2 minutes (avoid flapping)
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "Error rate above 5% for {{ $labels.service }}"
          description: "Current error rate: {{ $value | humanizePercentage }}"
          runbook: "https://runbooks.example.com/high-error-rate"
          dashboard: "https://grafana.example.com/d/svc-overview"

      # WARNING: Create a ticket, fix during business hours
      - alert: HighLatencyP99
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service)
          ) > 1.0
        for: 5m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "p99 latency above 1s for {{ $labels.service }}"
          description: "Current p99: {{ $value | humanizeDuration }}"

      # WARNING: Resource exhaustion approaching
      - alert: PodMemoryHigh
        expr: |
          container_memory_working_set_bytes
          / container_spec_memory_limit_bytes > 0.85
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Pod {{ $labels.pod }} memory at {{ $value | humanizePercentage }}"

# Alert design best practices:
best_practices = [
    "Every alert must have a runbook link (what to do when it fires)",
    "Use 'for' duration to avoid alerting on transient spikes",
    "Critical = page (wake someone up); Warning = ticket (business hours)",
    "Alert on symptoms (error rate) not causes (CPU usage)",
    "Include dashboard links in annotations for quick investigation",
    "Test alerts with promtool: promtool check rules alerting_rules.yml",
    "Review alert fatigue monthly — delete alerts nobody acts on",
]

for bp in best_practices:
    print(f"  - {bp}")
SOLUTION
}

# === Exercise 4: Dashboard Design ===
# Problem: Design a Grafana dashboard layout for a microservice,
# following the USE/RED methods.
exercise_4() {
    echo "=== Exercise 4: Dashboard Design ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Dashboard methodology:
# RED method (for request-driven services): Rate, Errors, Duration
# USE method (for resources): Utilization, Saturation, Errors

dashboard_panels = {
    "Row 1 — Overview (RED)": [
        {"title": "Request Rate", "type": "graph",
         "query": "sum(rate(http_requests_total[5m])) by (service)"},
        {"title": "Error Rate %", "type": "graph",
         "query": 'sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) * 100'},
        {"title": "p50 / p95 / p99 Latency", "type": "graph",
         "query": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))"},
    ],
    "Row 2 — Resources (USE)": [
        {"title": "CPU Utilization", "type": "graph",
         "query": "rate(container_cpu_usage_seconds_total[5m]) / container_spec_cpu_quota * 100"},
        {"title": "Memory Usage", "type": "graph",
         "query": "container_memory_working_set_bytes / container_spec_memory_limit_bytes * 100"},
        {"title": "Pod Count", "type": "stat",
         "query": 'count(kube_pod_status_ready{condition="true"}) by (deployment)'},
    ],
    "Row 3 — Dependencies": [
        {"title": "Database Query Duration", "type": "graph",
         "query": "histogram_quantile(0.99, rate(db_query_duration_seconds_bucket[5m]))"},
        {"title": "Redis Hit Rate", "type": "gauge",
         "query": "rate(redis_hits_total[5m]) / (rate(redis_hits_total[5m]) + rate(redis_misses_total[5m]))"},
        {"title": "External API Latency", "type": "graph",
         "query": "histogram_quantile(0.99, rate(external_request_duration_seconds_bucket[5m]))"},
    ],
    "Row 4 — Business Metrics": [
        {"title": "Orders/min", "type": "stat",
         "query": "sum(rate(orders_total[5m])) * 60"},
        {"title": "Revenue/hour", "type": "stat",
         "query": "sum(increase(order_revenue_total[1h]))"},
        {"title": "Cart Abandonment Rate", "type": "gauge",
         "query": "1 - rate(checkout_completed_total[1h]) / rate(cart_created_total[1h])"},
    ],
}

for row, panels in dashboard_panels.items():
    print(f"\n{row}")
    for p in panels:
        print(f"  [{p['type']:6s}] {p['title']}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 09: Monitoring and Observability"
echo "==============================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
