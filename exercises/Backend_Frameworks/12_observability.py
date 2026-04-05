# Exercise: Observability
# Practice with structured logging, metrics, and tracing.

import json
import time
import logging
from datetime import datetime


# Exercise 1: Structured Logger
# Create a logger that outputs JSON-formatted log entries.

class StructuredLogger:
    def __init__(self, service: str, version: str = "1.0.0"):
        self.service = service
        self.version = version
        self.context = {}

    def bind(self, **kwargs) -> "StructuredLogger":
        """Return a new logger with additional context fields."""
        # TODO: Implement (returns new instance with merged context)
        pass

    def _log(self, level: str, message: str, **kwargs):
        """Output a JSON log entry."""
        # TODO: Implement
        # Fields: timestamp (ISO), level, message, service, version,
        #         + context fields + extra kwargs
        pass

    def info(self, message: str, **kwargs):
        self._log("INFO", message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log("ERROR", message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log("WARNING", message, **kwargs)


# Test
# logger = StructuredLogger("my-api")
# req_logger = logger.bind(request_id="abc-123", user_id="user-1")
# req_logger.info("Request received", method="GET", path="/users")
# Expected JSON:
# {"timestamp":"2024-...","level":"INFO","message":"Request received",
#  "service":"my-api","version":"1.0.0","request_id":"abc-123",
#  "user_id":"user-1","method":"GET","path":"/users"}


# Exercise 2: Metrics Collector
# Implement a simple Prometheus-style metrics collector.

class Counter:
    """A monotonically increasing counter."""
    def __init__(self, name: str, help: str, labels: list[str] = None):
        self.name = name
        self.help = help
        self.labels = labels or []
        # TODO: Initialize storage

    def inc(self, value: float = 1.0, **label_values):
        """Increment the counter."""
        # TODO: Implement
        pass

    def get(self, **label_values) -> float:
        """Get current counter value for given labels."""
        # TODO: Implement
        pass


class Histogram:
    """Tracks distribution of values."""
    def __init__(self, name: str, help: str, buckets: list[float] = None):
        self.name = name
        self.help = help
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        # TODO: Initialize storage

    def observe(self, value: float):
        """Record a value."""
        # TODO: Implement
        pass

    def get_percentile(self, p: float) -> float:
        """Get approximate percentile value (e.g., 0.95 for p95)."""
        # TODO: Implement
        pass


# Exercise 3: Request Tracing
# Implement a simple distributed tracing system.

class Span:
    def __init__(self, trace_id: str, span_id: str, operation: str, parent_id: str = None):
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_id = parent_id
        self.operation = operation
        self.start_time = None
        self.end_time = None
        self.tags = {}
        self.logs = []

    def start(self):
        # TODO: Record start time
        pass

    def finish(self):
        # TODO: Record end time
        pass

    def set_tag(self, key: str, value: str):
        # TODO: Add tag
        pass

    def log(self, message: str):
        # TODO: Add timestamped log entry
        pass

    def duration_ms(self) -> float:
        # TODO: Calculate duration in milliseconds
        pass

    def to_dict(self) -> dict:
        # TODO: Serialize to dict
        pass


class Tracer:
    def __init__(self):
        self.spans = []

    def start_span(self, operation: str, parent: Span = None) -> Span:
        """Start a new span, optionally as a child of parent."""
        # TODO: Generate trace_id (or inherit from parent), generate span_id
        pass

    def report(self) -> list[dict]:
        """Get all completed spans as dicts."""
        # TODO: Implement
        pass


# Exercise 4: Middleware Integration
# Create a middleware function that combines logging, metrics, and tracing.

def observability_middleware(logger: StructuredLogger, counter: Counter, histogram: Histogram):
    """Returns a middleware function that:
    1. Creates a request-scoped logger with request_id
    2. Increments request counter with method and path labels
    3. Records request duration in histogram
    4. Logs request completion with status code and duration
    """
    def middleware(request, response_fn):
        # TODO: Implement
        pass
    return middleware


if __name__ == "__main__":
    print("Observability Exercise")
    print("Implement each class and test with the provided examples.")
