#!/usr/bin/env python3
"""Exercises for Lesson 21: Signal Correlation
Topic: DevOps
"""

import json
import logging
from datetime import datetime, timezone


# === Exercise 1: Trace-to-Log Linking ===

def exercise_1():
    """Implement trace-to-log linking with a custom formatter."""
    print("=== Exercise 1: Trace-to-Log Linking ===\n")

    class TraceContextFormatter(logging.Formatter):
        """Custom formatter that injects trace context into JSON log records."""
        def format(self, record):
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "trace_id": getattr(record, "trace_id", "0" * 32),
                "span_id": getattr(record, "span_id", "0" * 16),
            }
            return json.dumps(log_entry)

    handler = logging.StreamHandler()
    handler.setFormatter(TraceContextFormatter())
    logger = logging.getLogger("demo")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)

    # Simulate with trace context
    record = logger.makeRecord("demo", logging.INFO, "app.py", 42,
                               "Payment processing started", (), None)
    record.trace_id = "0af7651916cd43dd8448eb211c80319c"
    record.span_id = "b7ad6b7169203331"
    logger.handle(record)

    # Without trace context
    logger.info("No trace context here")


# === Exercise 2: Exemplar Analysis ===

def exercise_2():
    """Analyze histogram with exemplars."""
    print("\n=== Exercise 2: Exemplar Analysis ===\n")

    buckets = {
        0.1: {"count": 9500, "exemplar": ("aaa", 0.08)},
        0.5: {"count": 9900, "exemplar": ("bbb", 0.35)},
        1.0: {"count": 9980, "exemplar": ("ccc", 0.72)},
        5.0: {"count": 9998, "exemplar": ("ddd", 3.10)},
        float("inf"): {"count": 10000, "exemplar": ("eee", 12.5)},
    }
    total = 10000

    print("Histogram buckets:")
    for le, data in buckets.items():
        trace_id, val = data["exemplar"]
        print(f"  le={le}: count={data['count']}, exemplar=({trace_id}, {val}s)")

    # (a) Requests between 0.5s and 1.0s
    between = buckets[1.0]["count"] - buckets[0.5]["count"]
    print(f"\n(a) Requests between 0.5s and 1.0s: {between}")

    # (b) p95 latency
    p95_rank = 0.95 * total
    print(f"(b) p95 (rank {p95_rank}): ~0.1s (falls at le=0.1 boundary)")

    # (c) Which trace_id for tail latency
    print(f"(c) Investigate trace_id='eee' (12.5s) -- most extreme outlier")

    # (d) Percentage within 100ms
    pct = buckets[0.1]["count"] / total * 100
    print(f"(d) Requests within 100ms: {pct}%")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
