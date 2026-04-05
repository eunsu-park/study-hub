"""
Exercises for Lesson 14: Microservices Patterns
Topic: System_Design

Solutions to practice problems from the lesson.
Covers circuit breaker design, service mesh selection,
and distributed tracing implementation.
"""

import time
import random
import enum
from collections import defaultdict
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field


# === Exercise 1: Circuit Breaker Design ===
# Problem: Design a circuit breaker for a payment service.

class CircuitState(enum.Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject all requests
    HALF_OPEN = "half_open" # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for a payment service.

    State transitions:
    CLOSED -> OPEN: when failure_count >= failure_threshold
    OPEN -> HALF_OPEN: after timeout_duration
    HALF_OPEN -> CLOSED: on success
    HALF_OPEN -> OPEN: on failure
    """

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout_duration: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_duration = timeout_duration
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0

        # Metrics
        self.total_calls = 0
        self.total_failures = 0
        self.total_fallbacks = 0
        self.state_changes = []

    def _change_state(self, new_state, reason=""):
        old = self.state
        self.state = new_state
        self.state_changes.append((time.time(), old.value, new_state.value, reason))
        print(f"    [{self.service_name}] {old.value} -> {new_state.value}: {reason}")

    def call(self, operation: Callable, fallback: Callable = None):
        """Execute operation through circuit breaker."""
        self.total_calls += 1

        if self.state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if time.time() - self.last_failure_time >= self.timeout_duration:
                self._change_state(CircuitState.HALF_OPEN, "Timeout elapsed, testing")
                self.half_open_calls = 0
                self.success_count = 0
            else:
                self.total_fallbacks += 1
                if fallback:
                    return fallback()
                raise Exception(f"Circuit OPEN for {self.service_name}")

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                self.total_fallbacks += 1
                if fallback:
                    return fallback()
                raise Exception(f"Circuit HALF_OPEN, max test calls reached")
            self.half_open_calls += 1

        try:
            result = operation()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            self.total_fallbacks += 1
            if fallback:
                return fallback()
            raise

    def _on_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._change_state(CircuitState.CLOSED, "Enough successes in HALF_OPEN")
                self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # Reset on success

    def _on_failure(self):
        self.total_failures += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self._change_state(CircuitState.OPEN, "Failure in HALF_OPEN")
        elif self.state == CircuitState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self._change_state(CircuitState.OPEN,
                                   f"{self.failure_count} consecutive failures")


def exercise_1():
    """Circuit breaker design for payment service."""
    print("Circuit Breaker: Payment Service:")
    print("=" * 60)

    print("\nConfiguration:")
    print("  Failure threshold: 5 consecutive failures -> OPEN")
    print("  Timeout: 30s in OPEN before testing")
    print("  Success threshold: 3 successes in HALF_OPEN -> CLOSED")
    print("  Fallback: Return cached/default response")

    # Simulate with a short timeout for demo
    cb = CircuitBreaker(
        "PaymentService",
        failure_threshold=5,
        success_threshold=3,
        timeout_duration=0.1,  # Short for demo
        half_open_max_calls=3,
    )

    # Simulate payment calls
    call_count = {"n": 0}

    def payment_call():
        call_count["n"] += 1
        n = call_count["n"]
        # Fail calls 6-15, succeed otherwise
        if 6 <= n <= 15:
            raise Exception("Payment gateway timeout")
        return {"status": "success", "transaction_id": f"TXN-{n}"}

    def fallback():
        return {"status": "queued", "message": "Payment will be retried"}

    print("\nSimulation:")
    results = []
    for i in range(25):
        try:
            result = cb.call(payment_call, fallback)
            results.append(("OK" if result.get("status") == "success" else "FALLBACK",
                            cb.state.value))
        except Exception as e:
            results.append(("ERROR", cb.state.value))

        if i == 14:
            time.sleep(0.15)  # Wait for timeout

    print(f"\n  Results summary:")
    print(f"    Total calls: {cb.total_calls}")
    print(f"    Failures: {cb.total_failures}")
    print(f"    Fallbacks: {cb.total_fallbacks}")
    print(f"    State changes: {len(cb.state_changes)}")

    print("\n  Fallback strategies for payment service:")
    print("    1. Queue payment for async retry (return 'processing')")
    print("    2. Use backup payment provider")
    print("    3. Cache last known payment status")
    print("    4. Return degraded response (allow order but delay confirmation)")


# === Exercise 2: Service Mesh Selection ===
# Problem: Choose service mesh for given requirements.

def exercise_2():
    """Service mesh selection."""
    print("Service Mesh Selection:")
    print("=" * 60)

    print("\nRequirements:")
    print("  - 10 microservices")
    print("  - Kubernetes environment")
    print("  - mTLS required")
    print("  - Canary deployment needed")
    print("  - Team: intermediate K8s experience")

    options = [
        {
            "name": "Istio",
            "recommendation": "RECOMMENDED",
            "pros": [
                "Most mature and feature-rich",
                "Built-in mTLS (auto-enabled)",
                "Advanced traffic management (canary, A/B, mirroring)",
                "Large community and documentation",
            ],
            "cons": [
                "Higher resource overhead",
                "Steeper learning curve",
                "Complex configuration",
            ],
        },
        {
            "name": "Linkerd",
            "recommendation": "Strong alternative",
            "pros": [
                "Lightweight and simple",
                "Easier learning curve",
                "Lower resource footprint",
                "Good mTLS support",
            ],
            "cons": [
                "Fewer traffic management features",
                "Smaller ecosystem than Istio",
                "Limited canary support (needs Flagger)",
            ],
        },
        {
            "name": "Consul Connect",
            "recommendation": "If already using Consul",
            "pros": [
                "Multi-platform (K8s + VMs)",
                "Service discovery built-in",
                "Good mTLS support",
            ],
            "cons": [
                "Less mature for K8s-native features",
                "Canary deployment support limited",
            ],
        },
    ]

    for opt in options:
        marker = f" [{opt['recommendation']}]"
        print(f"\n  {opt['name']}{marker}")
        print(f"    Pros:")
        for p in opt['pros']:
            print(f"      + {p}")
        print(f"    Cons:")
        for c in opt['cons']:
            print(f"      - {c}")

    print("\n  Decision: Istio")
    print("  Rationale:")
    print("    1. mTLS is auto-enabled (meets security requirement)")
    print("    2. VirtualService + DestinationRule for canary deployments")
    print("    3. Team has intermediate K8s -> Istio learning curve is manageable")
    print("    4. 10 services is within Istio's sweet spot")


# === Exercise 3: Distributed Tracing Design ===
# Problem: Design distributed tracing for order processing system.

@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    service: str
    operation: str
    start_time: float
    end_time: float = 0
    tags: Dict = field(default_factory=dict)
    status: str = "OK"


class DistributedTracer:
    """Simplified distributed tracing system."""

    def __init__(self, sample_rate=0.1):
        self.traces = defaultdict(list)  # trace_id -> [spans]
        self.sample_rate = sample_rate
        self._span_counter = 0

    def _gen_id(self):
        self._span_counter += 1
        return f"span-{self._span_counter:04d}"

    def should_sample(self, trace_id=None):
        """Tail-based: always sample errors and slow requests."""
        # For head-based: random.random() < self.sample_rate
        return True  # Collect all, decide later

    def start_span(self, trace_id, service, operation, parent_span_id=None, tags=None):
        span = Span(
            trace_id=trace_id,
            span_id=self._gen_id(),
            parent_span_id=parent_span_id,
            service=service,
            operation=operation,
            start_time=time.time(),
            tags=tags or {},
        )
        return span

    def end_span(self, span, status="OK"):
        span.end_time = time.time()
        span.status = status
        self.traces[span.trace_id].append(span)

    def print_trace(self, trace_id):
        spans = self.traces.get(trace_id, [])
        if not spans:
            print("  No spans found")
            return

        # Build tree
        root_spans = [s for s in spans if s.parent_span_id is None]
        child_map = defaultdict(list)
        for s in spans:
            if s.parent_span_id:
                child_map[s.parent_span_id].append(s)

        def print_span(span, indent=0):
            duration = (span.end_time - span.start_time) * 1000
            prefix = "  " * indent
            status_marker = "X" if span.status != "OK" else "."
            print(f"  {prefix}[{status_marker}] {span.service}/{span.operation} "
                  f"({duration:.1f}ms) {span.tags}")
            for child in child_map.get(span.span_id, []):
                print_span(child, indent + 1)

        for root in root_spans:
            print_span(root)


def exercise_3():
    """Distributed tracing for order processing."""
    print("Distributed Tracing: Order Processing:")
    print("=" * 60)

    tracer = DistributedTracer(sample_rate=0.1)
    trace_id = "trace-order-001"

    # Simulate order processing flow
    # 1. API Gateway receives request
    gateway_span = tracer.start_span(
        trace_id, "api-gateway", "POST /orders",
        tags={"http.method": "POST", "http.url": "/orders", "user_id": "user_123"}
    )

    # 2. Order Service creates order
    order_span = tracer.start_span(
        trace_id, "order-service", "createOrder",
        parent_span_id=gateway_span.span_id,
        tags={"order_id": "ORD-001", "items": 3}
    )

    # 3. Order Service calls Inventory Service
    inventory_span = tracer.start_span(
        trace_id, "inventory-service", "reserveStock",
        parent_span_id=order_span.span_id,
        tags={"product_ids": ["P1", "P2", "P3"]}
    )
    time.sleep(0.001)
    tracer.end_span(inventory_span)

    # 4. Order Service calls Payment Service
    payment_span = tracer.start_span(
        trace_id, "payment-service", "processPayment",
        parent_span_id=order_span.span_id,
        tags={"amount": 150.00, "currency": "USD", "method": "credit_card"}
    )
    time.sleep(0.002)
    tracer.end_span(payment_span)

    # 5. Payment Service calls external payment gateway
    gateway_call = tracer.start_span(
        trace_id, "payment-service", "stripe.charge",
        parent_span_id=payment_span.span_id,
        tags={"provider": "stripe", "charge_id": "ch_xxx"}
    )
    time.sleep(0.003)
    tracer.end_span(gateway_call)

    # End parent spans
    time.sleep(0.001)
    tracer.end_span(order_span)
    tracer.end_span(gateway_span, status="OK")

    print("\n  Trace visualization:")
    tracer.print_trace(trace_id)

    # Sampling strategy
    print("\n  Sampling Strategy:")
    print("    Head-based: 10% of all traces (random)")
    print("    Tail-based: Keep ALL traces with:")
    print("      - Status != OK (errors)")
    print("      - Latency > p99 threshold")
    print("      - Specific user IDs (VIP customers)")
    print("    In production: tail-based recommended")

    # Key tags/metadata
    print("\n  Important Tags per Span:")
    print("    All spans: trace_id, span_id, parent_span_id, service, operation")
    print("    HTTP spans: method, url, status_code, user_agent")
    print("    DB spans: db.type, db.statement, db.duration")
    print("    Business: user_id, order_id, payment_id, amount")
    print("    Error: error.type, error.message, error.stack")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Circuit Breaker Design ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Service Mesh Selection ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Distributed Tracing Design ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
