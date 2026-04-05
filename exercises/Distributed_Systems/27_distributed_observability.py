"""
Exercises for Lesson 27: Distributed Observability
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import time
import json
import uuid
import random
from typing import Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass, field


# === Exercise 1: Trace Critical Path ===
def exercise_1():
    """Analyze a 12-span trace across 4 services."""
    print("=== Exercise 1: Trace Critical Path ===\n")

    spans = [
        {"id": "s1", "parent": None, "svc": "gateway", "op": "POST /order",
         "start": 0, "dur": 250},
        {"id": "s2", "parent": "s1", "svc": "order-svc", "op": "createOrder",
         "start": 5, "dur": 200},
        {"id": "s3", "parent": "s2", "svc": "order-svc", "op": "validateCart",
         "start": 10, "dur": 20},
        {"id": "s4", "parent": "s2", "svc": "inventory-svc", "op": "checkStock",
         "start": 35, "dur": 30},
        {"id": "s5", "parent": "s2", "svc": "pricing-svc", "op": "calculateTotal",
         "start": 35, "dur": 25},
        {"id": "s6", "parent": "s2", "svc": "order-svc", "op": "saveOrder",
         "start": 70, "dur": 40},
        {"id": "s7", "parent": "s6", "svc": "db", "op": "INSERT orders",
         "start": 75, "dur": 30},
        {"id": "s8", "parent": "s2", "svc": "payment-svc", "op": "charge",
         "start": 115, "dur": 80},
        {"id": "s9", "parent": "s8", "svc": "payment-svc", "op": "callStripe",
         "start": 120, "dur": 70},
        {"id": "s10", "parent": "s2", "svc": "order-svc", "op": "sendConfirmation",
         "start": 200, "dur": 5},
        {"id": "s11", "parent": "s10", "svc": "notification-svc", "op": "sendEmail",
         "start": 201, "dur": 3},
        {"id": "s12", "parent": "s1", "svc": "gateway", "op": "serialize",
         "start": 210, "dur": 10},
    ]

    # Service time breakdown
    svc_time = defaultdict(float)
    for s in spans:
        svc_time[s["svc"]] += s["dur"]

    total = spans[0]["dur"]
    print(f"  Total trace time: {total}ms\n")
    print(f"  Service breakdown:")
    for svc, dur in sorted(svc_time.items(), key=lambda x: -x[1]):
        pct = dur / total * 100
        print(f"    {svc:20s}: {dur:5.0f}ms ({pct:5.1f}%)")

    # Critical path: s1 → s2 → s8 → s9 (gateway → order → payment → stripe)
    print(f"\n  Critical path:")
    critical = ["s1", "s2", "s8", "s9"]
    for sid in critical:
        s = next(sp for sp in spans if sp["id"] == sid)
        print(f"    {s['svc']}/{s['op']}: {s['dur']}ms")


exercise_1()


# === Exercise 2: Structured Log Format ===
def exercise_2():
    """Design and implement structured logging format."""
    print("\n=== Exercise 2: Structured Logging ===\n")

    class Logger:
        def __init__(self, service, instance):
            self.service = service
            self.instance = instance
            self.entries = []

        def log(self, level, msg, **kwargs):
            entry = {
                "timestamp": time.time(),
                "level": level,
                "service": self.service,
                "instance": self.instance,
                "message": msg,
                **kwargs,
            }
            self.entries.append(entry)
            return entry

    class Aggregator:
        def __init__(self):
            self.logs = []

        def ingest(self, entries):
            self.logs.extend(entries)

        def search(self, **filters):
            results = self.logs
            for key, val in filters.items():
                results = [l for l in results if l.get(key) == val]
            return sorted(results, key=lambda l: l.get("timestamp", 0))

    agg = Aggregator()
    cid = "corr-abc123"

    for svc in ["gateway", "order-svc", "payment-svc"]:
        logger = Logger(svc, f"{svc}-01")
        logger.log("INFO", f"Processing request", correlation_id=cid, user_id="u42")
        if svc == "payment-svc":
            logger.log("ERROR", "Payment declined", correlation_id=cid,
                       error_code="INSUFFICIENT_FUNDS")
        agg.ingest(logger.entries)

    results = agg.search(correlation_id=cid)
    print(f"  Logs for {cid}:")
    for r in results:
        print(f"    [{r['level']:5s}] {r['service']:15s}: {r['message']}")


exercise_2()


# === Exercise 3: Metric Dashboard Design ===
def exercise_3():
    """Design dashboard for 3-tier application."""
    print("\n=== Exercise 3: Metric Dashboard ===\n")

    dashboard = {
        "Web Tier": {
            "metrics": ["request_rate", "error_rate_5xx", "latency_p50_p99",
                       "active_connections", "cache_hit_rate"],
            "alerts": [
                "error_rate > 1% for 5 min",
                "p99_latency > 500ms for 5 min",
            ],
        },
        "API Tier": {
            "metrics": ["request_rate_by_endpoint", "error_rate", "latency_p50_p99",
                       "thread_pool_utilization", "outbound_connection_count"],
            "alerts": [
                "error_rate > 0.5% for 5 min",
                "p99_latency > 200ms for 5 min",
                "thread_pool > 80% for 1 min",
            ],
        },
        "Database Tier": {
            "metrics": ["query_rate", "slow_query_rate", "connection_pool_usage",
                       "replication_lag", "disk_usage_pct"],
            "alerts": [
                "replication_lag > 5s",
                "connection_pool > 90%",
                "slow_queries > 10/min",
            ],
        },
    }

    for tier, info in dashboard.items():
        print(f"  {tier}:")
        print(f"    Metrics: {', '.join(info['metrics'])}")
        for alert in info["alerts"]:
            print(f"    Alert: {alert}")
        print()


exercise_3()


# === Exercise 4: Observability Library ===
def exercise_4():
    """Build observability library linking traces, logs, metrics."""
    print("\n=== Exercise 4: Observability Library ===\n")

    class ObservabilityContext:
        def __init__(self, service):
            self.service = service
            self.trace_id = uuid.uuid4().hex[:16]
            self.span_id = uuid.uuid4().hex[:8]
            self.correlation_id = uuid.uuid4().hex[:8]
            self.spans = []
            self.logs = []
            self.counters = defaultdict(int)

        def start_span(self, op):
            span = {"trace_id": self.trace_id, "span_id": uuid.uuid4().hex[:8],
                    "parent": self.span_id, "op": op, "start": time.time()}
            self.spans.append(span)
            return span

        def end_span(self, span):
            span["end"] = time.time()
            span["dur_ms"] = (span["end"] - span["start"]) * 1000

        def log(self, level, msg):
            self.logs.append({
                "level": level, "msg": msg,
                "correlation_id": self.correlation_id,
                "trace_id": self.trace_id,
            })

        def inc_counter(self, name):
            self.counters[name] += 1

        def summary(self):
            return {
                "spans": len(self.spans),
                "logs": len(self.logs),
                "counters": dict(self.counters),
                "trace_id": self.trace_id,
            }

    ctx = ObservabilityContext("order-service")
    span = ctx.start_span("processOrder")
    ctx.log("INFO", "Processing order #123")
    ctx.inc_counter("orders_processed")
    ctx.end_span(span)

    print(f"  Summary: {ctx.summary()}")
    print(f"  All linked by trace_id={ctx.trace_id}")


exercise_4()


# === Exercise 5: Debugging Procedure ===
def exercise_5():
    """Systematic debugging procedure for intermittent errors."""
    print("\n=== Exercise 5: Debugging Procedure ===\n")

    steps = [
        "1. GATHER: Collect correlation_id from the 500 error response",
        "2. TRACE: Find the distributed trace for this correlation_id",
        "3. IDENTIFY: Locate the span with error status in the trace",
        "4. NARROW: Determine which service and operation failed",
        "5. LOGS: Search logs by correlation_id + service + time window",
        "6. ERROR: Find the ERROR log with stack trace / error message",
        "7. CONTEXT: Check surrounding INFO logs for request parameters",
        "8. METRICS: Check service metrics (error rate, latency) at that time",
        "9. PATTERN: Check if error correlates with load spikes or deployments",
        "10. ROOT CAUSE: Identify whether it's timeout, OOM, bad input, or bug",
    ]

    print("  Systematic debugging procedure:")
    for step in steps:
        print(f"    {step}")

    print("\n  Information needed:")
    print("    - Correlation ID from the failing request")
    print("    - Distributed trace showing all service calls")
    print("    - Structured logs from all 5 services")
    print("    - Service metrics dashboard (error rate, latency)")
    print("    - Deployment history (recent changes)")


exercise_5()


if __name__ == "__main__":
    print("\nAll exercises completed.")
