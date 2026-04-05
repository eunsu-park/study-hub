#!/bin/bash
# Exercises for Lesson 15: Incident Management
# Topic: DevOps
# Solutions to practice problems from the lesson.

# === Exercise 1: Incident Response Process ===
# Problem: Define an incident response process with severity levels,
# roles, and communication templates.
exercise_1() {
    echo "=== Exercise 1: Incident Response Process ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
severity_levels = {
    "SEV1 — Critical": {
        "criteria": "Complete service outage or data loss affecting all users",
        "response_time": "15 minutes",
        "communication": "Every 30 minutes to stakeholders",
        "examples": ["Payment processing down", "Database corruption", "Security breach"],
        "who_paged": "On-call engineer + engineering manager + VP Eng",
    },
    "SEV2 — Major": {
        "criteria": "Significant degradation affecting >10% of users",
        "response_time": "30 minutes",
        "communication": "Every 1 hour",
        "examples": ["Search unavailable", "50% increased latency", "Region failover"],
        "who_paged": "On-call engineer + team lead",
    },
    "SEV3 — Minor": {
        "criteria": "Partial degradation, workaround available",
        "response_time": "4 hours",
        "communication": "Daily update in incident channel",
        "examples": ["One microservice degraded", "Non-critical feature broken"],
        "who_paged": "On-call engineer (Slack notification, no page)",
    },
    "SEV4 — Low": {
        "criteria": "Minor issue, no user impact",
        "response_time": "Next business day",
        "communication": "Ticket update",
        "examples": ["UI cosmetic bug", "Log noise increase"],
        "who_paged": "Ticket assigned to team backlog",
    },
}

for sev, details in severity_levels.items():
    print(f"\n{sev}")
    print(f"  Criteria:     {details['criteria']}")
    print(f"  Response:     {details['response_time']}")
    print(f"  Paged:        {details['who_paged']}")
    print(f"  Examples:     {', '.join(details['examples'][:2])}")

# Incident roles:
print("\nIncident Roles:")
print("  Incident Commander (IC): Coordinates response, makes decisions")
print("  Communications Lead:     Updates stakeholders and status page")
print("  Operations Lead:         Hands-on debugging and mitigation")
print("  Scribe:                  Documents timeline and actions taken")
SOLUTION
}

# === Exercise 2: Runbook Design ===
# Problem: Write a runbook for a common incident scenario: database
# connection pool exhaustion.
exercise_2() {
    echo "=== Exercise 2: Runbook Design ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# RUNBOOK: Database Connection Pool Exhaustion
# Severity: SEV2
# Last updated: 2025-01-15
# Owner: Platform Team

# SYMPTOMS:
# - Alert: "DatabaseConnectionPoolExhausted" firing
# - Application logs: "could not obtain connection from pool"
# - Error rate spike on endpoints that query the database
# - Healthy endpoints (cache-only) still working

# IMPACT:
# - All database-dependent endpoints return 500 errors
# - User-facing: checkout, order history, account pages affected

# DIAGNOSIS:
# Step 1: Confirm the alert
kubectl get pods -l app=order-api -n production
# Check for pods in CrashLoopBackOff or high restart counts

# Step 2: Check database connection count
psql -h db.internal -U admin -c "SELECT count(*) FROM pg_stat_activity;"
# Normal: <100, Problem: approaching max_connections (200)

# Step 3: Identify connection holders
psql -c "SELECT pid, now() - pg_stat_activity.query_start AS duration,
         query, state FROM pg_stat_activity WHERE state != 'idle'
         ORDER BY duration DESC LIMIT 20;"
# Look for: long-running queries, idle-in-transaction connections

# MITIGATION:
# Option A: Kill long-running queries
psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity
         WHERE duration > interval '5 minutes' AND state != 'idle';"

# Option B: Restart application pods (releases connections)
kubectl rollout restart deployment/order-api -n production
# Pods restart one at a time (rolling update), releasing connections

# Option C: Temporarily increase max_connections (if DB can handle it)
psql -c "ALTER SYSTEM SET max_connections = 300;"
# Requires: SELECT pg_reload_conf(); or DB restart

# PREVENTION:
# 1. Set pool size = min(max_connections / num_instances, 20)
# 2. Set connection timeout to 5 seconds (fail fast)
# 3. Add connection pool metrics to monitoring (pool_active, pool_idle)
# 4. Alert when pool utilization > 80%

# RESOLUTION CHECKLIST:
# [ ] Connections below normal threshold
# [ ] Error rate returned to baseline
# [ ] All pods healthy and serving traffic
# [ ] Root cause identified and documented
# [ ] Postmortem scheduled if SEV1/SEV2
SOLUTION
}

# === Exercise 3: Blameless Postmortem ===
# Problem: Write a blameless postmortem for a production incident.
exercise_3() {
    echo "=== Exercise 3: Blameless Postmortem ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# POSTMORTEM: Order API Outage — 2025-01-15
# Severity: SEV1 | Duration: 47 minutes | Users affected: ~12,000

# SUMMARY:
# The order-api service experienced a complete outage from 14:32 to 15:19 UTC
# due to database connection pool exhaustion triggered by a slow query
# introduced in release v1.4.2.

# TIMELINE (UTC):
# 14:00  v1.4.2 deployed to production (new search feature)
# 14:28  Slow query starts consuming connections (query plan regression)
# 14:32  Alert: DatabaseConnectionPoolExhausted fires
# 14:35  On-call engineer acknowledges, opens incident channel
# 14:38  IC declared, SEV1 escalation
# 14:42  Root cause identified: new ORDER BY clause missing index
# 14:48  Decision: rollback v1.4.2
# 14:52  Rollback initiated (kubectl rollout undo)
# 15:05  Pods restarted with v1.4.1, connections releasing
# 15:15  Error rate drops below 1%
# 15:19  All-clear declared, monitoring continues

# ROOT CAUSE:
# The search feature in v1.4.2 added an ORDER BY on a non-indexed column.
# Under production load (10x staging), the query took 30+ seconds instead
# of <100ms. Each slow query held a DB connection, exhausting the pool.

# CONTRIBUTING FACTORS (not root causes):
# - No query performance test in CI/CD for the affected endpoint
# - Staging database had 1/10th the production data volume
# - Connection pool had no timeout for idle-in-transaction connections

# WHAT WENT WELL:
# - Alert fired within 4 minutes of impact
# - IC assigned quickly, clear communication in Slack
# - Rollback executed cleanly in 7 minutes

# WHAT WENT WRONG:
# - 13 minutes from alert to IC declaration (target: 5 min)
# - No automated canary analysis would have caught the latency spike
# - Staging did not reproduce the issue (data volume mismatch)

# ACTION ITEMS:
actions = [
    ("Add index on orders.search_score column", "P0", "DB Team", "2025-01-16"),
    ("Add query performance tests to CI (explain analyze)", "P1", "Platform", "2025-01-22"),
    ("Seed staging DB with production-scale data sample", "P1", "Data Eng", "2025-01-31"),
    ("Set idle_in_transaction_session_timeout = 30s on DB", "P1", "DB Team", "2025-01-17"),
    ("Add canary analysis for p99 latency regression", "P2", "Platform", "2025-02-15"),
]

print("Action Items:")
print(f"  {'Action':<55} {'Priority':>8} {'Owner':>10} {'Due':>12}")
print("-" * 90)
for action, priority, owner, due in actions:
    print(f"  {action:<55} {priority:>8} {owner:>10} {due:>12}")

# BLAMELESS CULTURE NOTE:
# This postmortem focuses on SYSTEMS, not INDIVIDUALS.
# The developer who wrote the query is not at fault — the system
# lacked guardrails (no perf tests, no prod-scale staging data).
# We fix the system so this class of issue cannot recur.
SOLUTION
}

# === Exercise 4: On-Call Best Practices ===
# Problem: Design an on-call rotation with escalation policies,
# handoff procedures, and toil budgets.
exercise_4() {
    echo "=== Exercise 4: On-Call Best Practices ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
on_call_design = {
    "Rotation": {
        "schedule": "Weekly rotation, 2 engineers per shift (primary + secondary)",
        "handoff": "30-minute handoff meeting at rotation boundary",
        "compensation": "Comp time or on-call stipend per shift",
        "minimum_team_size": "6 engineers (max 1 week per 6 weeks)",
    },
    "Escalation Policy": {
        "level_1": "Primary on-call: paged immediately (PagerDuty/Opsgenie)",
        "level_2": "Secondary on-call: paged after 5 min no-ack",
        "level_3": "Team lead: paged after 15 min no-ack",
        "level_4": "Engineering manager: paged after 30 min no-ack",
    },
    "Handoff Checklist": [
        "Review open incidents and ongoing issues",
        "Check recent deployments (what changed this week?)",
        "Review alert noise (any alerts to silence/fix?)",
        "Verify PagerDuty schedule and phone numbers",
        "Confirm VPN/SSH access to production",
    ],
    "Toil Budget": {
        "target": "Max 50% of on-call time spent on toil (Google SRE guideline)",
        "track": "Log all on-call tasks in a shared spreadsheet",
        "reduce": "Automate recurring tasks, fix noisy alerts, improve runbooks",
        "escalate": "If toil exceeds 50%, raise to management for headcount/tooling",
    },
}

for area, details in on_call_design.items():
    print(f"\n{area}:")
    if isinstance(details, dict):
        for key, val in details.items():
            if isinstance(val, list):
                for item in val:
                    print(f"    - {item}")
            else:
                print(f"  {key}: {val}")
    elif isinstance(details, list):
        for item in details:
            print(f"  - {item}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 15: Incident Management"
echo "======================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
