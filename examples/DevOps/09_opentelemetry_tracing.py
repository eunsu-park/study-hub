#!/usr/bin/env python3
"""Example: OpenTelemetry Tracing Instrumentation

Demonstrates distributed tracing concepts using a simulated OpenTelemetry
SDK. Covers spans, context propagation, attributes, events, status codes,
and trace export.
Related lesson: 09_Monitoring_and_Observability.md
"""

# =============================================================================
# OPENTELEMETRY CONCEPTS
#   Trace    — end-to-end journey of a request across services
#   Span     — a single unit of work within a trace (has start/end time)
#   Context  — carries trace/span IDs across service boundaries
#   Exporter — sends trace data to a backend (Jaeger, Zipkin, OTLP)
# =============================================================================

import time
import random
import uuid
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generator


# =============================================================================
# 1. CORE TRACING MODEL (simulated OpenTelemetry SDK)
# =============================================================================

class SpanKind(Enum):
    INTERNAL = "INTERNAL"
    SERVER = "SERVER"
    CLIENT = "CLIENT"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"


class StatusCode(Enum):
    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


@dataclass
class SpanEvent:
    """An event (log) attached to a span."""
    name: str
    timestamp: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanLink:
    """A link to another span (causal but not parent-child)."""
    trace_id: str
    span_id: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """A unit of work in a distributed trace."""
    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    kind: SpanKind = SpanKind.INTERNAL
    start_time: float = 0.0
    end_time: float = 0.0
    status: StatusCode = StatusCode.UNSET
    status_message: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    links: list[SpanLink] = field(default_factory=list)
    service_name: str = ""

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append(SpanEvent(
            name=name,
            timestamp=time.time(),
            attributes=attributes or {},
        ))

    def set_status(self, code: StatusCode, message: str = "") -> None:
        self.status = code
        self.status_message = message

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status.value,
            "attributes": self.attributes,
            "events": [
                {"name": e.name, "attributes": e.attributes}
                for e in self.events
            ],
            "service": self.service_name,
        }


# =============================================================================
# 2. TRACER AND CONTEXT MANAGEMENT
# =============================================================================

_trace_context = threading.local()


def _gen_id(length: int = 16) -> str:
    return uuid.uuid4().hex[:length]


class Tracer:
    """Simplified tracer that creates and manages spans."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._spans: list[Span] = []

    @contextmanager
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        """Context manager that creates, activates, and finishes a span."""
        parent = getattr(_trace_context, "current_span", None)

        trace_id = parent.trace_id if parent else _gen_id(32)
        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=_gen_id(16),
            parent_span_id=parent.span_id if parent else None,
            kind=kind,
            start_time=time.time(),
            service_name=self.service_name,
            attributes=attributes or {},
        )

        # Push span onto context
        prev_span = parent
        _trace_context.current_span = span

        try:
            yield span
            if span.status == StatusCode.UNSET:
                span.set_status(StatusCode.OK)
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.add_event("exception", {
                "exception.type": type(e).__name__,
                "exception.message": str(e),
            })
            raise
        finally:
            span.end_time = time.time()
            self._spans.append(span)
            _trace_context.current_span = prev_span

    def get_spans(self) -> list[Span]:
        return list(self._spans)

    def clear(self) -> None:
        self._spans.clear()


# =============================================================================
# 3. SPAN ENRICHMENT HELPERS
# =============================================================================
# Semantic conventions from OpenTelemetry spec.

def add_http_server_attributes(span: Span, method: str, url: str,
                                status_code: int) -> None:
    """Add standard HTTP server span attributes."""
    span.set_attribute("http.method", method)
    span.set_attribute("http.url", url)
    span.set_attribute("http.status_code", status_code)
    span.set_attribute("http.scheme", "https")


def add_db_attributes(span: Span, system: str, statement: str,
                       operation: str) -> None:
    """Add standard database span attributes."""
    span.set_attribute("db.system", system)
    span.set_attribute("db.statement", statement)
    span.set_attribute("db.operation", operation)


def add_rpc_attributes(span: Span, system: str, service: str,
                        method: str) -> None:
    """Add standard RPC span attributes."""
    span.set_attribute("rpc.system", system)
    span.set_attribute("rpc.service", service)
    span.set_attribute("rpc.method", method)


# =============================================================================
# 4. SIMULATED APPLICATION WITH TRACING
# =============================================================================
# Simulates a 3-service architecture:
#   API Gateway -> Order Service -> Payment Service + Database

def simulate_api_gateway(tracer: Tracer) -> None:
    """API Gateway: receives HTTP request, forwards to order service."""
    with tracer.start_span("HTTP GET /api/orders", kind=SpanKind.SERVER) as span:
        add_http_server_attributes(span, "GET", "/api/orders", 200)
        span.set_attribute("http.user_agent", "Mozilla/5.0")
        span.set_attribute("enduser.id", "user-42")

        # Authenticate
        with tracer.start_span("authenticate") as auth_span:
            auth_span.set_attribute("auth.method", "JWT")
            time.sleep(0.002)  # 2ms auth
            auth_span.add_event("token_validated", {"token.claims": "user-42"})

        # Forward to order service
        simulate_order_service(tracer)


def simulate_order_service(tracer: Tracer) -> None:
    """Order service: fetches orders from DB and calls payment service."""
    with tracer.start_span("order-service.get_orders", kind=SpanKind.SERVER) as span:
        span.set_attribute("service.name", "order-service")

        # Database query
        with tracer.start_span("SELECT orders", kind=SpanKind.CLIENT) as db_span:
            add_db_attributes(
                db_span,
                system="postgresql",
                statement="SELECT * FROM orders WHERE user_id = $1 LIMIT 10",
                operation="SELECT",
            )
            db_span.set_attribute("db.name", "orders_db")
            time.sleep(random.uniform(0.005, 0.015))  # 5-15ms query
            db_span.add_event("query_completed", {"db.rows_affected": 5})

        # Call payment service for each order
        with tracer.start_span("payment-check", kind=SpanKind.CLIENT) as pay_span:
            add_rpc_attributes(pay_span, "grpc", "PaymentService", "GetStatus")
            simulate_payment_service(tracer)


def simulate_payment_service(tracer: Tracer) -> None:
    """Payment service: checks payment status (may be slow)."""
    with tracer.start_span("payment-service.get_status", kind=SpanKind.SERVER) as span:
        span.set_attribute("service.name", "payment-service")
        span.set_attribute("payment.provider", "stripe")

        # Simulate occasional slow response
        latency = random.uniform(0.01, 0.05)
        if random.random() < 0.1:  # 10% slow
            latency += 0.2
            span.add_event("slow_response", {"delay_ms": round(latency * 1000)})

        time.sleep(latency)

        # Cache lookup
        with tracer.start_span("redis.get", kind=SpanKind.CLIENT) as cache_span:
            cache_span.set_attribute("db.system", "redis")
            cache_span.set_attribute("db.operation", "GET")
            cache_hit = random.random() < 0.7
            cache_span.set_attribute("cache.hit", cache_hit)
            time.sleep(0.001)  # 1ms cache


# =============================================================================
# 5. TRACE VISUALIZATION (ASCII)
# =============================================================================

def print_trace(spans: list[Span]) -> None:
    """Print a trace as an ASCII timeline."""
    if not spans:
        print("  (no spans)")
        return

    # Group by trace
    traces: dict[str, list[Span]] = {}
    for span in spans:
        traces.setdefault(span.trace_id, []).append(span)

    for trace_id, trace_spans in traces.items():
        # Sort by start time
        trace_spans.sort(key=lambda s: s.start_time)
        earliest = trace_spans[0].start_time
        total_ms = (trace_spans[-1].end_time - earliest) * 1000

        print(f"\n  Trace ID: {trace_id}")
        print(f"  Total duration: {total_ms:.1f}ms")
        print(f"  Spans: {len(trace_spans)}")
        print()

        for span in trace_spans:
            offset_ms = (span.start_time - earliest) * 1000
            depth = 0
            # Count depth by walking parent chain
            parent_id = span.parent_span_id
            while parent_id:
                depth += 1
                parent_span = next(
                    (s for s in trace_spans if s.span_id == parent_id), None
                )
                parent_id = parent_span.parent_span_id if parent_span else None

            indent = "  " * depth
            status_icon = "OK" if span.status == StatusCode.OK else "ERR"
            print(
                f"    {indent}[{status_icon}] {span.name} "
                f"({span.duration_ms:.1f}ms) "
                f"kind={span.kind.value}"
            )
            if span.events:
                for event in span.events:
                    print(f"    {indent}  ^ {event.name}: {event.attributes}")


# =============================================================================
# 6. DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("OpenTelemetry Distributed Tracing Demo")
    print("=" * 70)

    tracer = Tracer(service_name="api-gateway")

    # Run 3 traced requests
    for i in range(3):
        print(f"\n{'─' * 50}")
        print(f"Request {i + 1}")
        print("─" * 50)
        simulate_api_gateway(tracer)
        # Print the trace for this request
        spans = tracer.get_spans()
        # Get only spans from the latest trace
        if spans:
            latest_trace_id = spans[-1].trace_id
            trace_spans = [s for s in spans if s.trace_id == latest_trace_id]
            print_trace(trace_spans)

    # Export as JSON (what an OTLP exporter would send)
    print(f"\n{'=' * 70}")
    print("Sample OTLP Export (JSON)")
    print("=" * 70)
    all_spans = tracer.get_spans()
    if all_spans:
        sample = all_spans[-1]
        print(json.dumps(sample.to_dict(), indent=2))

    print(f"\n{'=' * 70}")
    print("Key Takeaways")
    print("=" * 70)
    print("1. Every span has a trace_id (shared) and span_id (unique)")
    print("2. parent_span_id creates the tree structure within a trace")
    print("3. Span attributes follow semantic conventions (http.method, db.system)")
    print("4. Events mark notable points within a span (cache hit, slow query)")
    print("5. StatusCode.ERROR + exception event captures failures")
    print("6. Context propagation carries trace_id across service boundaries")
