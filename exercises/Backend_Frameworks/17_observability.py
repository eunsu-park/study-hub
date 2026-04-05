# Exercise: Observability
# Practice with structured logging, metrics collection, trace propagation, and alerting.

import json
import time
import uuid
from datetime import datetime, timezone
from collections import defaultdict
from typing import Optional


# Exercise 1: Structured Log Formatter
class LogFormatter:
    """Format log entries as JSON with consistent schema."""

    def __init__(self, service: str, default_fields: dict | None = None):
        self.service = service
        self.default_fields = default_fields or {}

    def format(self, level: str, message: str, **extra) -> str:
        """Return a JSON string with: timestamp, level, service, message,
        all default_fields, and any extra kwargs.

        Timestamp should be ISO 8601 UTC.
        """
        # TODO: Implement
        pass

    def with_context(self, **fields) -> "LogFormatter":
        """Return a new formatter with additional default fields merged."""
        # TODO: Implement
        pass


# Test
# fmt = LogFormatter("order-svc", {"env": "prod"})
# entry = json.loads(fmt.format("INFO", "order created", order_id="abc"))
# assert entry["service"] == "order-svc"
# assert entry["env"] == "prod"
# assert entry["order_id"] == "abc"
# child = fmt.with_context(trace_id="t1")
# entry2 = json.loads(child.format("WARN", "slow query"))
# assert entry2["trace_id"] == "t1"


# Exercise 2: Metrics Counter and Histogram
class MetricsRegistry:
    """Simple metrics registry supporting counters and histograms."""

    def __init__(self):
        self._counters = defaultdict(float)
        self._histograms = defaultdict(list)

    def counter_inc(self, name: str, value: float = 1.0, labels: dict | None = None):
        """Increment a counter. Labels are included in the key."""
        # TODO: Implement
        pass

    def histogram_observe(self, name: str, value: float, labels: dict | None = None):
        """Record an observation in a histogram."""
        # TODO: Implement
        pass

    def get_counter(self, name: str, labels: dict | None = None) -> float:
        """Get current counter value."""
        # TODO: Implement
        pass

    def get_histogram_stats(self, name: str, labels: dict | None = None) -> dict:
        """Get histogram statistics: count, sum, avg, min, max, p50, p99.

        Return empty dict if no observations.
        """
        # TODO: Implement
        pass

    def exposition(self) -> str:
        """Render all metrics in Prometheus text exposition format."""
        # TODO: Implement
        pass


# Test
# m = MetricsRegistry()
# m.counter_inc("http_requests", labels={"method": "GET", "status": "200"})
# m.counter_inc("http_requests", labels={"method": "GET", "status": "200"})
# assert m.get_counter("http_requests", {"method": "GET", "status": "200"}) == 2.0
# for v in [0.1, 0.2, 0.15, 0.5, 1.0]:
#     m.histogram_observe("request_duration", v)
# stats = m.get_histogram_stats("request_duration")
# assert stats["count"] == 5
# assert 0.38 < stats["avg"] < 0.40


# Exercise 3: Trace Context Propagation
class TraceContext:
    """W3C Trace Context implementation."""

    def __init__(self, trace_id: str | None = None, span_id: str | None = None,
                 parent_span_id: str | None = None):
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self.span_id = span_id or uuid.uuid4().hex[:8]
        self.parent_span_id = parent_span_id

    def create_child(self) -> "TraceContext":
        """Create a child span context (same trace_id, new span_id, this span as parent)."""
        # TODO: Implement
        pass

    def to_traceparent(self) -> str:
        """Serialize to W3C traceparent header: '00-{trace_id}-{span_id}-01'."""
        # TODO: Implement
        pass

    @classmethod
    def from_traceparent(cls, header: str) -> "TraceContext":
        """Parse a traceparent header string. Raise ValueError if malformed."""
        # TODO: Implement
        pass


# Test
# ctx = TraceContext()
# child = ctx.create_child()
# assert child.trace_id == ctx.trace_id
# assert child.parent_span_id == ctx.span_id
# header = ctx.to_traceparent()
# parsed = TraceContext.from_traceparent(header)
# assert parsed.trace_id == ctx.trace_id


# Exercise 4: Alert Rule Engine
class AlertRule:
    """Evaluate alert conditions against metric values."""

    def __init__(self, name: str, metric: str, operator: str, threshold: float,
                 for_seconds: int = 60):
        """
        operator: one of ">", "<", ">=", "<=", "=="
        for_seconds: condition must be true for this duration before firing
        """
        self.name = name
        self.metric = metric
        self.operator = operator
        self.threshold = threshold
        self.for_seconds = for_seconds
        self._first_triggered: Optional[float] = None

    def evaluate(self, value: float, now: float | None = None) -> dict:
        """Evaluate the rule against a metric value.

        Returns: {
            "name": str,
            "firing": bool,       # True if condition met for >= for_seconds
            "pending": bool,      # True if condition met but not yet for_seconds
            "value": float,
            "threshold": float,
        }
        """
        # TODO: Implement
        pass


# Test
# rule = AlertRule("high_latency", "p99_latency", ">", 1.0, for_seconds=0)
# result = rule.evaluate(1.5)
# assert result["firing"] is True
# result2 = rule.evaluate(0.5)
# assert result2["firing"] is False


if __name__ == "__main__":
    print("Observability Exercise")
    print("Implement each class/function and verify with the test cases.")
