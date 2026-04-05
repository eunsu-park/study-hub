#!/bin/bash
# Exercises for Lesson 10: Logging and Log Management
# Topic: DevOps
# Solutions to practice problems from the lesson.

# === Exercise 1: Structured Logging Implementation ===
# Problem: Convert plain-text logging to structured JSON logging with
# correlation IDs for a Python web application.
exercise_1() {
    echo "=== Exercise 1: Structured Logging Implementation ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import logging
import json
import uuid
from datetime import datetime, timezone

class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID from extra fields
        if hasattr(record, "correlation_id"):
            log_entry["correlation_id"] = record.correlation_id

        # Add any additional extra fields
        for key in ("user_id", "request_id", "duration_ms", "status_code"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry, default=str)

# Setup
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger("myapp")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage
cid = str(uuid.uuid4())
logger.info("Request received", extra={
    "correlation_id": cid,
    "user_id": 42,
    "request_id": "req-abc123",
})
# Output: {"timestamp": "2025-01-15T10:30:00+00:00", "level": "INFO",
#          "message": "Request received", "correlation_id": "abc...",
#          "user_id": 42, ...}
SOLUTION
}

# === Exercise 2: Log Aggregation Architecture ===
# Problem: Design a log aggregation pipeline using ELK or Loki.
exercise_2() {
    echo "=== Exercise 2: Log Aggregation Architecture ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Two common log aggregation stacks:

# === Option A: ELK Stack (Elasticsearch, Logstash, Kibana) ===
# App -> Filebeat -> Logstash -> Elasticsearch -> Kibana
#
# Pros: Full-text search, rich query language, mature ecosystem
# Cons: Resource-heavy (Elasticsearch needs RAM), operational complexity
#
# Elasticsearch sizing (rule of thumb):
#   - 1 GB ingested/day -> ~1.5 GB storage (with replicas)
#   - 100 GB/day -> need 3+ data nodes, 64GB RAM each

# === Option B: Loki + Grafana (lightweight, label-based) ===
# App -> Promtail -> Loki -> Grafana
#
# Pros: Low cost (indexes labels only, not content), integrates with Prometheus
# Cons: No full-text indexing (grep-like queries on content)
#
# Loki is the "Prometheus for logs" — same label model

architecture = {
    "Collection": {
        "agents": ["Filebeat", "Promtail", "Fluentd", "Vector"],
        "role": "Read logs from containers/files, enrich with labels, ship to aggregator",
    },
    "Processing": {
        "tools": ["Logstash", "Vector", "Fluentd"],
        "role": "Parse, filter, transform, route logs to storage",
    },
    "Storage": {
        "options": ["Elasticsearch", "Loki", "S3 (archive)"],
        "role": "Index and store logs for querying",
    },
    "Visualization": {
        "tools": ["Kibana", "Grafana"],
        "role": "Search, filter, dashboard, alert on log patterns",
    },
}

for stage, details in architecture.items():
    print(f"\n{stage}:")
    tools = details.get("agents", details.get("tools", details.get("options", [])))
    print(f"  Tools: {', '.join(tools)}")
    print(f"  Role:  {details['role']}")

# Retention policy:
print("\nRetention Policy:")
print("  Hot:    0-7 days  (fast SSD storage, full indexing)")
print("  Warm:   7-30 days (cheaper storage, reduced replicas)")
print("  Cold:   30-90 days (S3/object storage, compressed)")
print("  Delete: >90 days  (unless compliance requires longer)")
SOLUTION
}

# === Exercise 3: Log-Based Alerting ===
# Problem: Create alerting rules based on log patterns to detect
# security events and application errors.
exercise_3() {
    echo "=== Exercise 3: Log-Based Alerting ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Log-based alerts trigger on patterns in log content,
# as opposed to metric-based alerts on numeric thresholds.

log_alerts = [
    {
        "name": "AuthenticationBruteForce",
        "severity": "critical",
        "pattern": 'level="ERROR" AND message="authentication_failed"',
        "condition": "count > 50 in 5 minutes from same IP",
        "action": "Page security team, auto-block IP via WAF",
    },
    {
        "name": "DatabaseConnectionExhausted",
        "severity": "critical",
        "pattern": '"connection pool exhausted" OR "too many connections"',
        "condition": "count > 5 in 1 minute",
        "action": "Page on-call, check connection pool settings",
    },
    {
        "name": "UnhandledException",
        "severity": "warning",
        "pattern": 'level="ERROR" AND exception.type != "expected_errors"',
        "condition": "count > 10 in 5 minutes",
        "action": "Create ticket, notify development team",
    },
    {
        "name": "SensitiveDataExposure",
        "severity": "critical",
        "pattern": 'message=~"(password|secret|token|api_key).*logged"',
        "condition": "any occurrence",
        "action": "Page security team, rotate exposed credentials",
    },
    {
        "name": "DiskSpaceLow",
        "severity": "warning",
        "pattern": '"No space left on device" OR "disk usage above 90%"',
        "condition": "any occurrence",
        "action": "Create ticket, clean up logs or expand volume",
    },
]

for alert in log_alerts:
    print(f"\n[{alert['severity'].upper()}] {alert['name']}")
    print(f"  Pattern:   {alert['pattern']}")
    print(f"  Condition: {alert['condition']}")
    print(f"  Action:    {alert['action']}")

# Best practice: Log-based alerts should be a supplement to metrics,
# not a replacement. Use logs for pattern detection (security events,
# specific error messages) and metrics for threshold alerting.
SOLUTION
}

# === Exercise 4: Log Analysis with Command Line ===
# Problem: Demonstrate essential CLI tools for log analysis: jq, grep, awk.
exercise_4() {
    echo "=== Exercise 4: Log Analysis with Command Line ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Analyzing JSON logs with jq, grep, and awk

# 1. Parse JSON logs with jq
# Extract all error messages:
# cat app.log | jq -r 'select(.level == "ERROR") | .message'

# Count errors by type:
# cat app.log | jq -r 'select(.level == "ERROR") | .exception.type' | sort | uniq -c | sort -rn

# Filter by time range:
# cat app.log | jq 'select(.timestamp >= "2025-01-15T10:00:00" and .timestamp <= "2025-01-15T11:00:00")'

# 2. Find slow requests (>1 second)
# cat app.log | jq 'select(.duration_ms > 1000) | {timestamp, message, duration_ms, endpoint}'

# 3. Group by correlation ID (trace a full request):
# cat app.log | jq 'select(.correlation_id == "abc-123-def")' | jq -s 'sort_by(.timestamp)'

# 4. Error rate calculation:
# TOTAL=$(cat app.log | wc -l)
# ERRORS=$(cat app.log | jq 'select(.level == "ERROR")' | wc -l)
# echo "Error rate: $(echo "scale=2; $ERRORS * 100 / $TOTAL" | bc)%"

# 5. Top 10 slowest endpoints:
# cat app.log | jq -r 'select(.duration_ms != null) | [.endpoint, .duration_ms] | @tsv' \
#   | sort -t$'\t' -k2 -rn | head -10

# 6. Aggregate logs across multiple files:
# zcat /var/log/app/*.log.gz | jq 'select(.level == "ERROR")' | wc -l

# 7. Real-time log monitoring:
# tail -f /var/log/app/app.log | jq --unbuffered 'select(.level == "ERROR")'

# 8. Using grep for quick pattern search (when jq is overkill):
# grep -c '"level":"ERROR"' app.log           # Count errors
# grep '"status_code":500' app.log | tail -5  # Last 5 500 errors
# grep -P '"duration_ms":\d{4,}' app.log      # Duration >= 1000ms

print("Essential log analysis commands:")
print("  jq 'select(.level == \"ERROR\")'         # Filter by field")
print("  jq -r '.message'                         # Extract field")
print("  jq -s 'group_by(.endpoint) | map(...)'   # Aggregate")
print("  grep -c 'pattern' file                   # Count matches")
print("  tail -f file | jq --unbuffered '...'     # Real-time filter")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 10: Logging and Log Management"
echo "============================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
