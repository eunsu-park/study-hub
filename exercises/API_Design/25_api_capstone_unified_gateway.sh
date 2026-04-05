#!/bin/bash
# Exercises for Lesson 25: API Capstone — Unified Gateway
# Topic: API_Design
# Solutions to practice problems from the lesson.

exercise_1() {
    echo "=== Exercise 1: Protocol Translation ==="
    cat << 'SOLUTION'
import httpx
import base64
import strawberry

def _encode_cursor(page: int, limit: int) -> str:
    return base64.b64encode(f"{page}:{limit}".encode()).decode()

def _decode_cursor(cursor: str) -> tuple[int, int]:
    decoded = base64.b64decode(cursor.encode()).decode()
    page, limit = decoded.split(":")
    return int(page), int(limit)

@strawberry.type
class Query:
    @strawberry.field
    async def products(
        self,
        info,
        query: str | None = None,
        category: str | None = None,
        first: int = 10,
        after: str | None = None,
    ) -> "ProductConnection":
        # Convert cursor to REST page/limit
        page = 1
        if after:
            prev_page, _ = _decode_cursor(after)
            page = prev_page + 1

        # Call REST endpoint
        params = {"limit": first, "page": page}
        if query:
            params["q"] = query
        if category:
            params["category"] = category

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{PRODUCTS_URL}/api/products",
                params=params,
            )
            data = resp.json()

        # Convert to GraphQL Connection
        products = [Product(**p) for p in data["items"]]
        edges = [
            ProductEdge(
                node=p,
                cursor=_encode_cursor(page, first),
            )
            for p in products
        ]

        return ProductConnection(
            edges=edges,
            page_info=PageInfo(
                has_next_page=data.get("has_next", False),
                has_previous_page=page > 1,
                start_cursor=edges[0].cursor if edges else None,
                end_cursor=edges[-1].cursor if edges else None,
            ),
            total_count=data.get("total", 0),
        )
SOLUTION
}

exercise_2() {
    echo "=== Exercise 2: Gateway Rate Limiter ==="
    cat << 'SOLUTION'
import aioredis
import time
import json

class GatewayRateLimiter:
    TIERS = {
        "anonymous": {"cpm": 200},
        "free":      {"cpm": 500},
        "pro":       {"cpm": 5000},
        "enterprise":{"cpm": 50000},
    }
    COST_MULTIPLIERS = {
        "query": 1,
        "mutation": 5,
        "subscription": 10,
    }

    def __init__(self, redis_url):
        self.redis = aioredis.from_url(redis_url)

    async def check(self, client_id, tier, operation_type, query_cost=1):
        multiplier = self.COST_MULTIPLIERS.get(operation_type, 1)
        total_cost = query_cost * multiplier
        limits = self.TIERS.get(tier, self.TIERS["free"])
        window = 60
        now = time.time()
        key = f"rl:{client_id}:{int(now // window)}"

        pipe = self.redis.pipeline()
        pipe.incrby(key, total_cost)
        pipe.expire(key, window + 1)
        results = await pipe.execute()

        current = results[0]
        limit = limits["cpm"]
        remaining = max(0, limit - current)
        reset_at = (int(now // window) + 1) * window

        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(reset_at)),
            "X-RateLimit-Cost": str(total_cost),
        }

        return {
            "allowed": current <= limit,
            "remaining": remaining,
            "headers": headers,
        }
SOLUTION
}

exercise_3() {
    echo "=== Exercise 3: Circuit Breaker ==="
    cat << 'SOLUTION'
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("circuit_breaker")

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, name, failure_threshold=5, recovery_timeout=30):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = timedelta(seconds=recovery_timeout)
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.last_failure = None
        self._cached_response = None

    def can_execute(self):
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if datetime.now() - self.last_failure > self.recovery_timeout:
                self._transition(CircuitState.HALF_OPEN)
                return True
            return False
        return True  # HALF_OPEN

    def record_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self._transition(CircuitState.CLOSED)
        self.failures = 0

    def record_failure(self):
        self.failures += 1
        self.last_failure = datetime.now()
        if self.failures >= self.failure_threshold:
            self._transition(CircuitState.OPEN)

    def _transition(self, new_state):
        old = self.state
        self.state = new_state
        logger.warning(f"Circuit '{self.name}': {old.value} -> {new_state.value}")
        # Emit metric
        metrics.gauge(f"circuit.{self.name}.state", new_state.value)

    def get_fallback(self):
        """Return cached response when circuit is open."""
        if self._cached_response:
            return self._cached_response
        return {"error": "Service unavailable", "status": 503}

    def cache_response(self, response):
        self._cached_response = response
SOLUTION
}

exercise_4() {
    echo "=== Exercise 4: Migration Plan ==="
    cat << 'SOLUTION'
# REST API with 30 endpoints → GraphQL Migration Plan

# Phase 1: Analysis (Week 1-2)
# Map endpoints to GraphQL operations:
#   Queries (GET endpoints):
#     GET /users          → Query.users
#     GET /users/:id      → Query.user(id)
#     GET /products       → Query.products
#     GET /products/:id   → Query.product(id)
#     GET /orders         → Query.myOrders
#     ...
#
#   Mutations (POST/PUT/DELETE):
#     POST /users         → Mutation.createUser
#     PUT /users/:id      → Mutation.updateUser
#     DELETE /users/:id   → Mutation.deleteUser
#     POST /orders        → Mutation.createOrder
#     ...

# Phase 2: Coexistence (Week 3-6)
# - Deploy GraphQL endpoint at /graphql
# - GraphQL resolvers internally call REST services
# - New features built in GraphQL only
# - Add Sunset header to REST responses:
#   Sunset: Sat, 01 Jun 2025 00:00:00 GMT
#   Deprecation: true
#   Link: </docs/migration>; rel="sunset"

# Phase 3: Client Migration (Week 7-14)
# Priority order:
#   1. Mobile apps (biggest bandwidth savings)
#   2. Internal tools (most control)
#   3. Web frontend (page-by-page)
#   4. Partner integrations (need migration guide)

# Phase 4: REST Deprecation (Week 15-18)
# - Monitor REST traffic (target: <5% of total)
# - Send deprecation emails to remaining consumers
# - Return 299 Warning header on REST responses

# Phase 5: Sunset (Week 19+)
# - Return 410 Gone for deprecated endpoints
# - Remove REST code
# - GraphQL as sole API

# Sunset Header Implementation:
from datetime import datetime
from starlette.middleware.base import BaseHTTPMiddleware

SUNSET_DATE = "Sat, 01 Jun 2025 00:00:00 GMT"

class SunsetHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/v1/"):
            response.headers["Sunset"] = SUNSET_DATE
            response.headers["Deprecation"] = "true"
            response.headers["Link"] = '</docs/graphql-migration>; rel="sunset"'
        return response

# Client Migration Guide outline:
# 1. Install GraphQL client library
# 2. Replace fetch('/api/v1/users/1') with GraphQL query
# 3. Map REST response fields to GraphQL selections
# 4. Replace pagination (page/limit → first/after)
# 5. Replace POST mutations with GraphQL mutations
# 6. Update error handling (HTTP status → errors array)
# 7. Test thoroughly
# 8. Remove REST client code
SOLUTION
}

main() { exercise_1; echo ""; exercise_2; echo ""; exercise_3; echo ""; exercise_4; }
main "$@"
