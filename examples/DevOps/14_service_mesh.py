#!/usr/bin/env python3
"""Example: Service Mesh & Networking — Traffic Routing and Resilience

Demonstrates service mesh concepts: sidecar proxy model, traffic splitting,
circuit breaking, retry policies, mutual TLS simulation, and observability
header propagation.
Related lesson: 09_Service_Mesh_and_Networking.md
"""

# =============================================================================
# WHY A SERVICE MESH?
# In microservice architectures, every service must handle retries, timeouts,
# circuit breaking, mTLS, and observability. A service mesh (Istio, Linkerd)
# offloads these cross-cutting concerns to sidecar proxies so application
# code stays focused on business logic.
# =============================================================================

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


# =============================================================================
# 1. SERVICE AND REQUEST MODELS
# =============================================================================

@dataclass
class Request:
    """An HTTP-like request flowing through the mesh."""
    method: str = "GET"
    path: str = "/"
    headers: dict[str, str] = field(default_factory=dict)
    trace_id: str = ""
    attempt: int = 1


@dataclass
class Response:
    """An HTTP-like response."""
    status_code: int = 200
    body: str = ""
    latency_ms: float = 0.0


class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject requests
    HALF_OPEN = "half-open" # Testing recovery


# =============================================================================
# 2. CIRCUIT BREAKER
# =============================================================================

@dataclass
class CircuitBreaker:
    """Implements the circuit breaker pattern for upstream protection."""
    failure_threshold: int = 5
    recovery_timeout_s: float = 30.0
    half_open_max_calls: int = 1

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    half_open_calls: int = 0

    def allow_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            elapsed = time.monotonic() - self.last_failure_time
            if elapsed >= self.recovery_timeout_s:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        # Half-open: allow limited probing
        if self.half_open_calls < self.half_open_max_calls:
            self.half_open_calls += 1
            return True
        return False

    def record_success(self) -> None:
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
        self.failure_count = 0

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN


# =============================================================================
# 3. RETRY POLICY
# =============================================================================

@dataclass
class RetryPolicy:
    """Configurable retry with exponential backoff."""
    max_retries: int = 3
    base_delay_ms: float = 100.0
    max_delay_ms: float = 5000.0
    retryable_codes: set[int] = field(default_factory=lambda: {502, 503, 504})

    def should_retry(self, response: Response, attempt: int) -> bool:
        return (attempt <= self.max_retries and
                response.status_code in self.retryable_codes)

    def delay_ms(self, attempt: int) -> float:
        delay = self.base_delay_ms * (2 ** (attempt - 1))
        jitter = random.uniform(0, delay * 0.1)
        return min(delay + jitter, self.max_delay_ms)


# =============================================================================
# 4. TRAFFIC SPLITTING (CANARY / A-B ROUTING)
# =============================================================================

@dataclass
class TrafficRoute:
    """A weighted traffic routing rule."""
    destination: str
    weight: int  # Percentage 0-100
    headers_match: dict[str, str] = field(default_factory=dict)


@dataclass
class VirtualService:
    """Defines traffic routing rules (modeled after Istio VirtualService)."""
    name: str
    host: str
    routes: list[TrafficRoute] = field(default_factory=list)

    def resolve(self, request: Request) -> str:
        """Select a destination based on headers or weighted routing."""
        # Header-based routing takes priority (e.g., canary header)
        for route in self.routes:
            if route.headers_match:
                if all(request.headers.get(k) == v
                       for k, v in route.headers_match.items()):
                    return route.destination

        # Weighted random selection
        roll = random.randint(1, 100)
        cumulative = 0
        for route in self.routes:
            if not route.headers_match:
                cumulative += route.weight
                if roll <= cumulative:
                    return route.destination
        return self.routes[-1].destination


# =============================================================================
# 5. SIDECAR PROXY
# =============================================================================

@dataclass
class SidecarProxy:
    """Simulates an Envoy-style sidecar proxy with mesh features."""
    service_name: str
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    mtls_enabled: bool = True
    request_log: list[dict] = field(default_factory=list)

    def handle_request(self, request: Request,
                       upstream_fn: Callable[[Request], Response]) -> Response:
        """Process a request through the sidecar with full mesh features."""
        # Inject tracing headers
        if not request.trace_id:
            request.trace_id = f"trace-{random.randint(10000, 99999)}"
        request.headers["x-request-id"] = request.trace_id
        request.headers["x-envoy-downstream-service"] = self.service_name

        # mTLS indicator
        if self.mtls_enabled:
            request.headers["x-forwarded-client-cert"] = f"By={self.service_name}"

        # Circuit breaker check
        if not self.circuit_breaker.allow_request():
            return Response(status_code=503, body="Circuit breaker OPEN")

        # Execute with retry
        attempt = 0
        response = Response(status_code=500, body="No attempts made")
        while True:
            attempt += 1
            request.attempt = attempt
            response = upstream_fn(request)

            if response.status_code < 500:
                self.circuit_breaker.record_success()
                break
            else:
                self.circuit_breaker.record_failure()
                if not self.retry_policy.should_retry(response, attempt):
                    break
                # Backoff before retry (simulated)
                _ = self.retry_policy.delay_ms(attempt)

        # Log for observability
        self.request_log.append({
            "trace_id": request.trace_id,
            "path": request.path,
            "status": response.status_code,
            "attempts": attempt,
            "latency_ms": response.latency_ms,
            "circuit_state": self.circuit_breaker.state.value,
        })
        return response


# =============================================================================
# 6. DEMO
# =============================================================================

def make_flaky_upstream(failure_rate: float = 0.4) -> Callable[[Request], Response]:
    """Create an upstream handler that fails at a given rate."""
    def handler(request: Request) -> Response:
        if random.random() < failure_rate:
            return Response(status_code=503, body="Service Unavailable", latency_ms=50.0)
        return Response(status_code=200, body="OK", latency_ms=random.uniform(5, 25))
    return handler


if __name__ == "__main__":
    random.seed(42)

    # --- Traffic Splitting ---
    print("=" * 60)
    print("Traffic Splitting (90/10 canary)")
    print("=" * 60)
    vs = VirtualService(
        name="payment-svc",
        host="payment.default.svc.cluster.local",
        routes=[
            TrafficRoute(destination="payment-v1", weight=90),
            TrafficRoute(destination="payment-v2", weight=10),
            TrafficRoute(destination="payment-v2",
                         headers_match={"x-canary": "true"}, weight=0),
        ],
    )
    counts: dict[str, int] = {"payment-v1": 0, "payment-v2": 0}
    for _ in range(100):
        dest = vs.resolve(Request())
        counts[dest] += 1
    print(f"  100 requests -> {counts}")

    # Force canary via header
    canary_req = Request(headers={"x-canary": "true"})
    print(f"  With x-canary header -> {vs.resolve(canary_req)}")

    # --- Sidecar Proxy with Circuit Breaker ---
    print(f"\n{'=' * 60}")
    print("Sidecar Proxy — Circuit Breaker + Retry")
    print("=" * 60)
    proxy = SidecarProxy(
        service_name="order-svc",
        circuit_breaker=CircuitBreaker(failure_threshold=3, recovery_timeout_s=0.5),
        retry_policy=RetryPolicy(max_retries=2),
    )
    upstream = make_flaky_upstream(0.6)

    for i in range(12):
        req = Request(path=f"/api/orders/{i}")
        resp = proxy.handle_request(req, upstream)
        log_entry = proxy.request_log[-1]
        print(f"  Req {i+1:2d}: status={resp.status_code} "
              f"attempts={log_entry['attempts']} "
              f"circuit={log_entry['circuit_state']}")

    # --- Mesh Observability ---
    print(f"\n{'=' * 60}")
    print("Request Log (observability)")
    print("=" * 60)
    for entry in proxy.request_log[:5]:
        print(f"  trace={entry['trace_id']} path={entry['path']} "
              f"status={entry['status']} latency={entry['latency_ms']:.1f}ms")
