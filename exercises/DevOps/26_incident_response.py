#!/usr/bin/env python3
"""Exercises for Lesson 26: Incident Response
Topic: DevOps
"""


def exercise_1():
    """Severity classification for various scenarios."""
    print("=== Exercise 1: Severity Classification ===\n")
    scenarios = [
        ("Marketing homepage typo", "SEV4", "No functional impact, cosmetic only"),
        ("Login service 503 for 30% of requests", "SEV2",
         "Core functionality degraded, 30% users affected"),
        ("DB migration data loss (2h of registrations)", "SEV1",
         "Data loss is ALWAYS SEV1 regardless of scope"),
        ("Internal wiki down during business hours", "SEV3",
         "No external user impact, internal productivity only"),
        ("Credit card numbers in application logs", "SEV1",
         "Security/PCI-DSS breach, always SEV1"),
    ]
    for scenario, sev, reason in scenarios:
        print(f"  {scenario}")
        print(f"    → {sev}: {reason}")
        print()


def exercise_2():
    """Postmortem action items for search service incident."""
    print("=== Exercise 2: Postmortem Action Items ===\n")
    actions = [
        ("P0", "Detection", "Add search index freshness monitoring (alert if > 1h stale)"),
        ("P0", "Prevention", "Fix cron job error handling (structured error + Slack alert)"),
        ("P1", "Detection", "Add ES cluster health alerting (yellow/red > 10min)"),
        ("P1", "Detection", "Deploy synthetic search freshness check"),
        ("P1", "Prevention", "Fix unassigned replica shard + write runbook"),
        ("P2", "Mitigation", "Add user-facing staleness indicator banner"),
    ]
    print("| # | Priority | Category | Action |")
    print("|---|----------|----------|--------|")
    for i, (priority, category, action) in enumerate(actions, 1):
        print(f"| {i} | {priority} | {category} | {action} |")


def exercise_3():
    """Runbook design for database connection pool exhaustion."""
    print("\n=== Exercise 3: Runbook Design ===\n")
    print("Runbook: Database Connection Pool Exhausted")
    print()
    steps = [
        "Step 1: Check pool metrics (kubectl exec ... curl /metrics | grep db_pool)",
        "Step 2: Find long-running queries (pg_stat_activity WHERE duration > 30s)",
        "Step 3: Check for connection leaks (growing connections over time)",
        "Step 4: Check DB server health (max_connections vs active, lock contention)",
    ]
    for step in steps:
        print(f"  {step}")

    print("\nMitigation actions:")
    mitigations = [
        "Long-running query found → pg_terminate_backend(pid)",
        "Connection leak → kubectl rollout restart",
        "Traffic spike → Scale up replicas + increase pool size",
        "DB overloaded → Enable PgBouncer connection pooler",
    ]
    for m in mitigations:
        print(f"  - {m}")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
