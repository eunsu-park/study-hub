#!/usr/bin/env python3
"""Example: Structured Logging with JSON Formatter & Correlation IDs

Demonstrates production-grade structured logging: JSON output, correlation
ID propagation, log levels, contextual fields, and log aggregation patterns.
Related lesson: 10_Logging_and_Log_Management.md
"""

# =============================================================================
# WHY STRUCTURED LOGGING?
# Plain-text logs ("INFO: user logged in") are hard to parse and search at
# scale. Structured logs emit machine-readable JSON so that log aggregation
# systems (ELK, Loki, Datadog) can index, filter, and alert on any field.
# =============================================================================

import json
import logging
import sys
import time
import uuid
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generator


# =============================================================================
# 1. CORRELATION ID CONTEXT
# =============================================================================
# A correlation ID (aka trace ID, request ID) ties all log entries from a
# single request together. It propagates through threads and service calls.

_context: threading.local = threading.local()


def get_correlation_id() -> str:
    """Get the current correlation ID, or generate one if absent."""
    return getattr(_context, "correlation_id", str(uuid.uuid4()))


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current thread."""
    _context.correlation_id = cid


@contextmanager
def correlation_context(cid: str | None = None) -> Generator[str, None, None]:
    """Context manager that sets (and restores) a correlation ID."""
    prev = getattr(_context, "correlation_id", None)
    new_cid = cid or str(uuid.uuid4())
    set_correlation_id(new_cid)
    try:
        yield new_cid
    finally:
        if prev is not None:
            set_correlation_id(prev)
        elif hasattr(_context, "correlation_id"):
            del _context.correlation_id


# =============================================================================
# 2. JSON LOG FORMATTER
# =============================================================================
# Outputs each log record as a single JSON line. Fields include:
#   timestamp, level, message, logger, correlation_id, and any extras.

class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    STANDARD_FIELDS = {
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "exc_info", "exc_text", "stack_info",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process", "message",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": get_correlation_id(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Include any extra fields attached to the log record
        for key, value in record.__dict__.items():
            if key not in self.STANDARD_FIELDS and not key.startswith("_"):
                log_entry[key] = value

        # Include exception info if present
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(log_entry, default=str)


# =============================================================================
# 3. LOGGER FACTORY
# =============================================================================

def get_logger(
    name: str,
    level: int = logging.INFO,
    stream: Any = None,
) -> logging.Logger:
    """Create a logger with JSON formatting and correlation ID support.

    Usage:
        logger = get_logger(__name__)
        logger.info("User logged in", extra={"user_id": 42})
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    return logger


# =============================================================================
# 4. STRUCTURED LOG HELPERS
# =============================================================================
# Wrappers that attach common context automatically.

class StructuredLogger:
    """Logger wrapper that attaches default context fields."""

    def __init__(self, name: str, default_context: dict[str, Any] | None = None):
        self._logger = get_logger(name)
        self._context = default_context or {}

    def _merge_extra(self, extra: dict[str, Any] | None) -> dict[str, Any]:
        merged = {**self._context}
        if extra:
            merged.update(extra)
        return merged

    def info(self, msg: str, **extra: Any) -> None:
        self._logger.info(msg, extra=self._merge_extra(extra))

    def warning(self, msg: str, **extra: Any) -> None:
        self._logger.warning(msg, extra=self._merge_extra(extra))

    def error(self, msg: str, exc_info: bool = False, **extra: Any) -> None:
        self._logger.error(msg, exc_info=exc_info, extra=self._merge_extra(extra))

    def debug(self, msg: str, **extra: Any) -> None:
        self._logger.debug(msg, extra=self._merge_extra(extra))

    @contextmanager
    def timed(self, operation: str, **extra: Any) -> Generator[None, None, None]:
        """Context manager that logs operation duration."""
        start = time.monotonic()
        self.info(f"{operation} started", **extra)
        try:
            yield
        except Exception:
            duration_ms = (time.monotonic() - start) * 1000
            self.error(
                f"{operation} failed",
                exc_info=True,
                duration_ms=round(duration_ms, 2),
                **extra,
            )
            raise
        else:
            duration_ms = (time.monotonic() - start) * 1000
            self.info(
                f"{operation} completed",
                duration_ms=round(duration_ms, 2),
                **extra,
            )


# =============================================================================
# 5. APPLICATION SIMULATION
# =============================================================================

def simulate_request_processing(logger: StructuredLogger, user_id: int) -> None:
    """Simulate a web request with structured logging throughout."""

    # Log the incoming request
    logger.info(
        "Request received",
        user_id=user_id,
        method="POST",
        path="/api/orders",
        ip="192.168.1.42",
    )

    # Simulate database query
    with logger.timed("database_query", table="orders", user_id=user_id):
        time.sleep(0.02)  # Simulate DB latency

    # Simulate external API call
    with logger.timed("payment_api_call", provider="stripe", user_id=user_id):
        time.sleep(0.05)  # Simulate API latency

    # Log business event
    logger.info(
        "Order created",
        user_id=user_id,
        order_id="ORD-12345",
        amount=99.99,
        currency="USD",
    )

    # Log the response
    logger.info(
        "Request completed",
        user_id=user_id,
        status_code=201,
        response_size_bytes=256,
    )


def simulate_error_scenario(logger: StructuredLogger) -> None:
    """Simulate an error with full context in structured logs."""
    logger.warning(
        "Rate limit approaching",
        client_id="client-abc",
        current_rate=95,
        limit=100,
    )

    try:
        # Simulate an error
        result = 1 / 0
    except ZeroDivisionError:
        logger.error(
            "Calculation failed",
            exc_info=True,
            operation="discount_calculation",
            input_value=0,
        )


# =============================================================================
# 6. LOG FILTERING AND AGGREGATION PATTERNS
# =============================================================================

def demonstrate_log_levels() -> None:
    """Show how log levels map to operational concerns."""
    log = StructuredLogger("levels_demo", {"service": "order-api"})

    print("\n--- Log Level Guide ---")
    print("DEBUG:    Detailed diagnostic info (disabled in production)")
    print("INFO:     Normal operational events (request served, job completed)")
    print("WARNING:  Something unexpected but recoverable (retry, fallback)")
    print("ERROR:    Operation failed, needs attention (exception, timeout)")
    print("CRITICAL: System is in a broken state (out of memory, data loss)\n")

    log.debug("Cache miss for key user:42", cache="redis", ttl_remaining=0)
    log.info("Order processed successfully", order_id="ORD-100")
    log.warning("Slow query detected", query_time_ms=2500, threshold_ms=1000)
    log.error("Payment gateway timeout", gateway="stripe", timeout_s=30)


# =============================================================================
# 7. DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Structured Logging Demo")
    print("=" * 70)
    print()

    logger = StructuredLogger(
        "myapp.api",
        default_context={"service": "order-api", "version": "1.2.3"},
    )

    # Simulate two requests with different correlation IDs
    print("--- Request 1 ---")
    with correlation_context() as cid:
        print(f"[Correlation ID: {cid}]")
        simulate_request_processing(logger, user_id=42)

    print()
    print("--- Request 2 (with error) ---")
    with correlation_context() as cid:
        print(f"[Correlation ID: {cid}]")
        simulate_error_scenario(logger)

    print()
    print("--- Log Levels ---")
    demonstrate_log_levels()

    print()
    print("=" * 70)
    print("Key Takeaways")
    print("=" * 70)
    print("1. Every log line is valid JSON -> easy to parse with jq, ELK, Loki")
    print("2. correlation_id ties all logs from one request together")
    print("3. Extra fields (user_id, order_id) enable precise filtering")
    print("4. Timed context manager auto-logs duration for performance tracking")
    print("5. Exception info is structured, not buried in a text blob")
