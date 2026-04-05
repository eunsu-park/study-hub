#!/usr/bin/env python3
"""Exercises for Lesson 19: Observability Engineering
Topic: DevOps
Solutions to practice problems from the lesson.
"""

# === Exercise 1: Observability Maturity Assessment ===
# Problem: Assess your current observability maturity level and propose a roadmap.

def exercise_1():
    """Observability maturity model assessment."""
    print("=== Exercise 1: Observability Maturity Assessment ===\n")

    maturity_levels = {
        "L0_Reactive": {
            "description": "Basic health checks, no structured telemetry",
            "capabilities": ["up/down monitoring", "manual log checking"],
            "tools": ["ping", "basic nagios"],
        },
        "L1_Informed": {
            "description": "Metrics + dashboards, golden signals monitored",
            "capabilities": ["Prometheus metrics", "Grafana dashboards", "threshold alerts"],
            "tools": ["Prometheus", "Grafana", "Alertmanager"],
        },
        "L2_Investigative": {
            "description": "Correlated signals: metrics, logs, traces linked",
            "capabilities": ["distributed tracing", "trace-to-log linking", "exemplars"],
            "tools": ["OpenTelemetry", "Jaeger/Tempo", "Loki"],
        },
        "L3_Proactive": {
            "description": "SLO-driven decisions, anomaly detection",
            "capabilities": ["SLO dashboards", "error budgets", "dynamic alerting"],
            "tools": ["Sloth", "Grafana ML", "burn rate alerts"],
        },
        "L4_Predictive": {
            "description": "Capacity forecasting, automated remediation",
            "capabilities": ["trend analysis", "auto-scaling", "predictive alerts"],
            "tools": ["AIOps platforms", "ML models", "auto-remediation"],
        },
    }

    for level, info in maturity_levels.items():
        print(f"{level}: {info['description']}")
        print(f"  Capabilities: {', '.join(info['capabilities'])}")
        print(f"  Tools: {', '.join(info['tools'])}")
        print()


# === Exercise 2: Instrumentation Design ===
# Problem: Design instrumentation for a user registration endpoint.

def exercise_2():
    """Instrumentation design for a user registration endpoint."""
    print("=== Exercise 2: Instrumentation Design ===\n")

    spans = [
        {"name": "register_user", "parent": "root",
         "attributes": ["user.email_domain", "user.source"],
         "justification": "Top-level span for the entire operation"},
        {"name": "validate_input", "parent": "register_user",
         "attributes": ["validation.error_count", "validation.fields"],
         "justification": "Business-relevant validation failures"},
        {"name": "check_duplicate_email", "parent": "register_user",
         "attributes": ["db.system=postgresql", "user.duplicate_found"],
         "justification": "Auto-instrumented DB span + custom attribute"},
        {"name": "hash_password", "parent": "register_user",
         "attributes": ["bcrypt.cost_factor=12"],
         "justification": "CPU-intensive; important to track latency contribution"},
        {"name": "create_user_record", "parent": "register_user",
         "attributes": ["db.system=postgresql", "user.id"],
         "justification": "Auto-instrumented DB operation"},
        {"name": "send_welcome_email", "parent": "register_user",
         "attributes": ["email.provider=ses", "email.template=welcome"],
         "justification": "External dependency; may fail independently"},
    ]

    print("Spans:")
    for span in spans:
        print(f"  {span['name']} (parent: {span['parent']})")
        print(f"    Attributes: {', '.join(span['attributes'])}")
        print(f"    Justification: {span['justification']}")
    print()

    metrics = [
        ("user.registrations_total", "Counter", ["status", "source"], "Business KPI"),
        ("user.registration_duration_seconds", "Histogram", ["source"], "Performance"),
        ("user.password_hash_duration_seconds", "Histogram", [], "CPU-bound tracking"),
    ]

    print("Metrics:")
    for name, mtype, labels, reason in metrics:
        print(f"  {name} ({mtype}) labels={labels} -- {reason}")


# === Exercise 3: Cardinality Analysis ===
# Problem: Analyze cardinality of a proposed metric.

def exercise_3():
    """Cardinality analysis of a proposed metric."""
    print("=== Exercise 3: Cardinality Analysis ===\n")

    # Original metric labels and their cardinalities
    original = {
        "method": 5,
        "endpoint": 200,
        "status_code": 50,
        "user_id": 100_000,
        "request_id": float("inf"),
        "region": 3,
    }

    print("Original labels and cardinality:")
    product = 1
    for label, card in original.items():
        print(f"  {label}: {card}")
        if card != float("inf"):
            product *= card
    print(f"  Theoretical max (without request_id): {product:,} (UNBOUNDED with request_id)")
    print()

    # Labels to remove
    print("Labels to REMOVE:")
    print("  - request_id: unbounded cardinality, use trace ID in traces")
    print("  - user_id: 100K unique values, use traces for per-user analysis")
    print()

    # Labels to modify
    print("Labels to MODIFY:")
    print("  - status_code → status_class: bucket into 2xx/3xx/4xx/5xx (5 values)")
    print("  - endpoint → route: group parameterized paths (200 → ~30 templates)")
    print()

    # Revised cardinality
    revised = {"method": 5, "route": 30, "status_class": 5, "region": 3}
    revised_product = 1
    for card in revised.values():
        revised_product *= card
    print(f"Revised cardinality: {' × '.join(str(v) for v in revised.values())} = {revised_product:,}")
    print(f"Reduction: {product:,} → {revised_product:,}")


if __name__ == "__main__":
    exercise_1()
    print("\n" + "=" * 70 + "\n")
    exercise_2()
    print("\n" + "=" * 70 + "\n")
    exercise_3()
