#!/usr/bin/env python3
"""Example: API Gateway

Demonstrates core API gateway patterns using FastAPI as a reverse proxy:
- Request routing to backend services
- Load balancing (round-robin)
- Circuit breaker for fault tolerance
- Request/response transformation
- Centralized logging middleware

Related lesson: 12_API_Gateway.md

Run:
    pip install "fastapi[standard]" httpx
    uvicorn 12_api_gateway:app --reload --port 8000
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gateway")

# =============================================================================
# SERVICE REGISTRY — Maps route prefixes to backend instances
# =============================================================================
# In production this comes from service discovery (Consul, Kubernetes DNS).

SERVICE_REGISTRY: dict[str, list[str]] = {
    "users":    ["http://users-svc:8001", "http://users-svc:8002"],
    "orders":   ["http://orders-svc:8003"],
    "products": ["http://products-svc:8004", "http://products-svc:8005"],
}


# =============================================================================
# LOAD BALANCER — Round-robin across healthy instances
# =============================================================================

class RoundRobinBalancer:
    """Distributes requests evenly across backend instances.

    Round-robin is simple but effective when instances have similar capacity.
    For heterogeneous backends, consider weighted or least-connections.
    """

    def __init__(self):
        self._counters: dict[str, int] = defaultdict(int)

    def next(self, service: str) -> str:
        instances = SERVICE_REGISTRY.get(service, [])
        if not instances:
            raise HTTPException(status_code=503, detail=f"No instances for {service}")
        idx = self._counters[service] % len(instances)
        self._counters[service] += 1
        return instances[idx]


balancer = RoundRobinBalancer()


# =============================================================================
# CIRCUIT BREAKER — Prevent cascade failures
# =============================================================================
# States: CLOSED (normal) -> OPEN (failing) -> HALF_OPEN (probing)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Per-service circuit breaker implementing the three-state pattern.

    - CLOSED: requests flow normally, failures are counted.
    - OPEN: requests are rejected immediately (fail fast).
    - HALF_OPEN: one probe request is allowed to test recovery.
    """
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0

    def allow_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.monotonic() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker -> HALF_OPEN (probing)")
                return True
            return False
        # HALF_OPEN: allow one probe
        return True

    def record_success(self):
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("Circuit breaker -> CLOSED (recovered)")

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker -> OPEN (too many failures)")


# Per-service circuit breakers
_breakers: dict[str, CircuitBreaker] = defaultdict(CircuitBreaker)


# =============================================================================
# APPLICATION
# =============================================================================

app = FastAPI(title="API Gateway", version="1.0.0")


# =============================================================================
# MIDDLEWARE — Centralized request logging and timing
# =============================================================================

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} ({elapsed_ms:.1f}ms)"
    )
    response.headers["X-Gateway-Latency-Ms"] = f"{elapsed_ms:.1f}"
    return response


# =============================================================================
# PROXY ROUTE — Routes all /api/<service>/* to the correct backend
# =============================================================================

@app.api_route(
    "/api/{service}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    tags=["Gateway"],
    summary="Reverse-proxy to backend services",
)
async def proxy(service: str, path: str, request: Request):
    """Route requests to backend services via load balancer and circuit breaker.

    In this demo, actual HTTP calls are simulated. In production, use
    httpx.AsyncClient to forward the request.
    """
    # Check circuit breaker
    breaker = _breakers[service]
    if not breaker.allow_request():
        raise HTTPException(
            status_code=503,
            detail=f"Service '{service}' circuit open — try again later",
        )

    # Resolve backend instance
    try:
        backend = balancer.next(service)
    except HTTPException:
        breaker.record_failure()
        raise

    target_url = f"{backend}/{path}"
    logger.info(f"Routing to {target_url}")

    # --- Simulated backend response ---
    # In production, replace with:
    #   async with httpx.AsyncClient() as client:
    #       resp = await client.request(request.method, target_url, ...)
    simulated = {
        "gateway": True,
        "service": service,
        "backend": backend,
        "path": f"/{path}",
        "method": request.method,
        "circuit_state": breaker.state.value,
    }
    breaker.record_success()
    return JSONResponse(content=simulated)


# =============================================================================
# HEALTH / STATUS ENDPOINTS
# =============================================================================

@app.get("/gateway/health", tags=["Gateway"])
def gateway_health():
    """Gateway liveness check."""
    return {"status": "ok", "services": list(SERVICE_REGISTRY.keys())}


@app.get("/gateway/circuits", tags=["Gateway"])
def circuit_status():
    """Report the state of all circuit breakers."""
    return {
        svc: {"state": b.state.value, "failures": b.failure_count}
        for svc, b in _breakers.items()
    }


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("12_api_gateway:app", host="127.0.0.1", port=8000, reload=True)
