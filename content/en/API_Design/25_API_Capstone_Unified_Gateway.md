# 25. API Capstone — Unified Gateway

**Previous**: [GraphQL Testing and Tooling](./24_GraphQL_Testing_and_Tooling.md) | **Next**: None (Final Lesson)

**Difficulty**: ⭐⭐⭐⭐

---

## Learning Objectives

- Design a unified API gateway that routes traffic to REST, GraphQL, and gRPC backends
- Implement protocol translation between GraphQL queries and REST/gRPC service calls
- Apply request routing, authentication, and rate limiting at the gateway layer
- Plan and execute a migration strategy from REST to GraphQL without breaking existing clients
- Evaluate gateway patterns: API gateway, Backend-for-Frontend (BFF), and schema stitching
- Build a working prototype gateway using Python that combines all three protocols

---

## Table of Contents

1. [The Unified Gateway Concept](#1-the-unified-gateway-concept)
2. [Gateway Architecture Patterns](#2-gateway-architecture-patterns)
3. [Protocol Translation](#3-protocol-translation)
4. [Building the Gateway](#4-building-the-gateway)
5. [Authentication at the Gateway](#5-authentication-at-the-gateway)
6. [Rate Limiting and Traffic Management](#6-rate-limiting-and-traffic-management)
7. [REST to GraphQL Migration](#7-rest-to-graphql-migration)
8. [GraphQL Schema Stitching](#8-graphql-schema-stitching)
9. [Observability and Monitoring](#9-observability-and-monitoring)
10. [Capstone Project](#10-capstone-project)
11. [Exercises](#11-exercises)
12. [References](#12-references)

---

## 1. The Unified Gateway Concept

Modern APIs often combine multiple protocols. A unified gateway provides a single entry point that routes requests to the appropriate backend.

### Why a Unified Gateway?

| Challenge | Gateway Solution |
|-----------|-----------------|
| Clients must know multiple endpoints | Single entry point |
| Different auth mechanisms per service | Centralized authentication |
| Inconsistent rate limiting | Unified policy enforcement |
| Cross-cutting concerns duplicated | Applied once at the gateway |
| Protocol-specific clients needed | Protocol translation |

### Architecture Overview

```
                         ┌──────────────────────┐
    Mobile App ────────▶ │                      │ ──▶ REST Service (Users)
    Web App ───────────▶ │   Unified Gateway    │ ──▶ GraphQL Service (Content)
    Partner API ───────▶ │                      │ ──▶ gRPC Service (Orders)
    Internal Tool ─────▶ │  - Auth              │ ──▶ gRPC Service (Inventory)
                         │  - Rate Limiting      │ ──▶ REST Service (Payments)
                         │  - Protocol Routing   │ ──▶ Event Bus (Notifications)
                         │  - Logging            │
                         └──────────────────────┘
```

### Protocol Strengths

| Protocol | Best For | Exposed Via Gateway |
|----------|----------|-------------------|
| REST | Public APIs, CRUD, caching | Direct passthrough |
| GraphQL | Complex queries, mobile clients | Query endpoint |
| gRPC | Internal services, streaming | Translated to REST/GraphQL |
| WebSocket | Real-time subscriptions | Proxied |

---

## 2. Gateway Architecture Patterns

### Pattern 1: Simple Reverse Proxy

Route by URL path:

```
/api/users/*    → REST Users Service
/api/graphql    → GraphQL Content Service
/api/orders/*   → gRPC Orders Service (via grpc-gateway)
```

```python
# Simple routing with FastAPI
from fastapi import FastAPI, Request
import httpx

app = FastAPI(title="API Gateway")

ROUTES = {
    "/api/users": "http://users-service:8001",
    "/api/graphql": "http://graphql-service:8002",
    "/api/orders": "http://orders-service:8003",
}


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, path: str):
    """Route requests to the appropriate backend service."""
    for prefix, upstream in ROUTES.items():
        if f"/api/{path}".startswith(prefix):
            async with httpx.AsyncClient() as client:
                url = f"{upstream}/{path}"
                response = await client.request(
                    method=request.method,
                    url=url,
                    headers=dict(request.headers),
                    content=await request.body(),
                )
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
    return JSONResponse({"error": "Not found"}, status_code=404)
```

### Pattern 2: Backend-for-Frontend (BFF)

Each client type gets a tailored gateway:

```
Mobile BFF ──▶ REST (users) + GraphQL (content, optimized for mobile)
Web BFF    ──▶ REST (users) + GraphQL (content, full data)
Partner BFF──▶ REST (users, rate-limited) + REST (orders, read-only)
```

```python
# Mobile BFF: aggregates and optimizes for mobile
@app.get("/mobile/home")
async def mobile_home(request: Request):
    """Aggregated home screen data for mobile clients."""
    async with httpx.AsyncClient() as client:
        # Parallel requests to multiple services
        user_task = client.get(f"{USERS_URL}/api/users/me",
                               headers=forward_auth(request))
        feed_task = client.post(f"{GRAPHQL_URL}/graphql", json={
            "query": """
                query MobileFeed {
                    feed(first: 10) {
                        edges { node { id title thumbnail } }
                    }
                }
            """
        })

        user_resp, feed_resp = await asyncio.gather(
            user_task, feed_task
        )

    return {
        "user": user_resp.json(),
        "feed": feed_resp.json()["data"]["feed"],
    }
```

### Pattern 3: GraphQL Gateway (Schema Stitching / Federation)

GraphQL as the unified query layer:

```
Client → GraphQL Gateway → REST Service (wrapped in resolvers)
                         → GraphQL Subgraph
                         → gRPC Service (wrapped in resolvers)
```

### Pattern Comparison

| Pattern | Complexity | Flexibility | Best For |
|---------|-----------|-------------|----------|
| Reverse Proxy | Low | Low | Simple routing |
| BFF | Medium | High | Multiple client types |
| GraphQL Gateway | High | Very High | Complex data requirements |
| Service Mesh | Very High | Very High | Kubernetes microservices |

---

## 3. Protocol Translation

### REST to GraphQL Wrapping

Expose REST endpoints as GraphQL fields:

```python
@strawberry.type
class Query:
    @strawberry.field
    async def user(self, info, id: strawberry.ID) -> "User":
        """Wraps REST GET /api/users/{id} as a GraphQL field."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{USERS_SERVICE_URL}/api/users/{id}",
                headers={"Authorization": info.context.auth_header},
            )
            resp.raise_for_status()
            data = resp.json()
            return User(
                id=data["id"],
                username=data["username"],
                email=data["email"],
            )

    @strawberry.field
    async def products(
        self, info, category: str | None = None, limit: int = 10
    ) -> list["Product"]:
        """Wraps REST GET /api/products as a GraphQL field."""
        params = {"limit": limit}
        if category:
            params["category"] = category

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{PRODUCTS_SERVICE_URL}/api/products",
                params=params,
            )
            return [Product(**p) for p in resp.json()["items"]]
```

### gRPC to GraphQL Wrapping

```python
import grpc
from orders_pb2 import GetOrderRequest
from orders_pb2_grpc import OrderServiceStub


@strawberry.type
class Query:
    @strawberry.field
    async def order(self, info, id: strawberry.ID) -> "Order":
        """Wraps gRPC GetOrder as a GraphQL field."""
        channel = grpc.aio.insecure_channel("orders-service:50051")
        stub = OrderServiceStub(channel)

        request = GetOrderRequest(order_id=str(id))
        response = await stub.GetOrder(request)

        return Order(
            id=str(response.id),
            status=response.status,
            total=float(response.total_cents) / 100,
            items=[
                OrderItem(
                    product_id=str(item.product_id),
                    quantity=item.quantity,
                    price=float(item.price_cents) / 100,
                )
                for item in response.items
            ],
        )
```

### GraphQL to REST Translation

Expose GraphQL queries as REST endpoints for legacy clients:

```python
@app.get("/api/v1/users/{user_id}")
async def get_user_rest(user_id: str, fields: str = "id,username,email"):
    """REST endpoint that delegates to GraphQL internally."""
    requested_fields = fields.split(",")
    graphql_fields = " ".join(requested_fields)

    result = await schema.execute(
        f'query {{ user(id: "{user_id}") {{ {graphql_fields} }} }}'
    )

    if result.errors:
        return JSONResponse(
            {"error": result.errors[0].message},
            status_code=404,
        )

    return result.data["user"]


@app.get("/api/v1/users/{user_id}/posts")
async def get_user_posts_rest(user_id: str, limit: int = 10, offset: int = 0):
    """REST endpoint for user's posts, backed by GraphQL."""
    result = await schema.execute(
        """
        query GetUserPosts($userId: ID!, $limit: Int!, $offset: Int!) {
            user(id: $userId) {
                posts(first: $limit, offset: $offset) {
                    id
                    title
                    status
                    createdAt
                }
            }
        }
        """,
        variable_values={
            "userId": user_id,
            "limit": limit,
            "offset": offset,
        },
    )

    if result.errors:
        return JSONResponse({"error": "Not found"}, status_code=404)

    return {
        "items": result.data["user"]["posts"],
        "limit": limit,
        "offset": offset,
    }
```

---

## 4. Building the Gateway

### Complete Gateway Implementation

```python
# gateway/main.py
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import httpx
import structlog

from gateway.auth import authenticate, AuthContext
from gateway.rate_limit import RateLimiter
from gateway.graphql_schema import schema
from strawberry.fastapi import GraphQLRouter

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources."""
    app.state.http_client = httpx.AsyncClient(timeout=30.0)
    app.state.rate_limiter = RateLimiter(redis_url="redis://localhost:6379")
    yield
    await app.state.http_client.aclose()


app = FastAPI(title="Unified API Gateway", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.example.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# GraphQL endpoint
graphql_router = GraphQLRouter(schema)
app.include_router(graphql_router, prefix="/graphql")


# REST proxy endpoints
@app.api_route(
    "/api/v1/{service}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def rest_proxy(
    request: Request,
    service: str,
    path: str,
    auth: AuthContext = Depends(authenticate),
):
    """Proxy REST requests to upstream services."""
    service_map = {
        "users": "http://users-service:8001",
        "products": "http://products-service:8002",
        "payments": "http://payments-service:8003",
    }

    upstream = service_map.get(service)
    if not upstream:
        return JSONResponse({"error": "Unknown service"}, status_code=404)

    # Rate limit check
    allowed = await request.app.state.rate_limiter.check(
        client_id=auth.client_id,
        cost=1,
    )
    if not allowed:
        return JSONResponse({"error": "Rate limit exceeded"}, status_code=429)

    # Forward request
    client = request.app.state.http_client
    upstream_url = f"{upstream}/api/{path}"

    response = await client.request(
        method=request.method,
        url=upstream_url,
        headers={
            "Authorization": request.headers.get("Authorization", ""),
            "X-Request-ID": request.headers.get("X-Request-ID", ""),
            "X-Forwarded-For": request.client.host,
        },
        params=dict(request.query_params),
        content=await request.body() if request.method != "GET" else None,
    )

    logger.info(
        "rest_proxy",
        service=service,
        path=path,
        method=request.method,
        status=response.status_code,
    )

    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type"),
    )


# Health check
@app.get("/health")
async def health():
    return {"status": "healthy", "services": ["users", "products", "payments", "graphql"]}
```

### Gateway Configuration

```yaml
# gateway/config.yaml
gateway:
  port: 4000
  cors:
    origins: ["https://app.example.com"]

services:
  users:
    url: http://users-service:8001
    protocol: rest
    timeout: 10s
    retries: 2

  content:
    url: http://content-service:8002
    protocol: graphql
    timeout: 30s

  orders:
    url: orders-service:50051
    protocol: grpc
    timeout: 15s

  payments:
    url: http://payments-service:8003
    protocol: rest
    timeout: 30s
    retries: 3

rate_limiting:
  default:
    requests_per_minute: 100
    cost_per_minute: 1000
  tiers:
    free: { requests_per_minute: 60, cost_per_minute: 500 }
    pro: { requests_per_minute: 600, cost_per_minute: 5000 }
    enterprise: { requests_per_minute: 6000, cost_per_minute: 50000 }

auth:
  jwt_secret: ${JWT_SECRET}
  api_key_header: X-API-Key
```

---

## 5. Authentication at the Gateway

### Centralized Authentication

```python
# gateway/auth.py
from dataclasses import dataclass
from fastapi import Request, HTTPException
import jwt


@dataclass
class AuthContext:
    client_id: str
    user_id: str | None
    roles: list[str]
    tier: str
    auth_method: str  # "jwt", "api_key", "anonymous"


async def authenticate(request: Request) -> AuthContext:
    """Extract and validate authentication from the request."""
    # Try JWT first
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            return AuthContext(
                client_id=payload["sub"],
                user_id=payload["sub"],
                roles=payload.get("roles", []),
                tier=payload.get("tier", "free"),
                auth_method="jwt",
            )
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

    # Try API key
    api_key = request.headers.get("X-API-Key")
    if api_key:
        key_data = await validate_api_key(api_key)
        if key_data:
            return AuthContext(
                client_id=key_data["client_id"],
                user_id=None,
                roles=key_data.get("roles", []),
                tier=key_data.get("tier", "free"),
                auth_method="api_key",
            )
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Anonymous access
    return AuthContext(
        client_id=request.client.host,
        user_id=None,
        roles=[],
        tier="anonymous",
        auth_method="anonymous",
    )
```

### Header Forwarding

```python
def build_upstream_headers(request: Request, auth: AuthContext) -> dict:
    """Build headers to forward to upstream services."""
    return {
        "Authorization": request.headers.get("Authorization", ""),
        "X-Request-ID": request.headers.get("X-Request-ID", str(uuid4())),
        "X-Client-ID": auth.client_id,
        "X-User-ID": auth.user_id or "",
        "X-User-Roles": ",".join(auth.roles),
        "X-Forwarded-For": request.client.host,
        "X-Forwarded-Proto": request.url.scheme,
    }
```

---

## 6. Rate Limiting and Traffic Management

### Multi-Tier Rate Limiting

```python
# gateway/rate_limit.py
import aioredis
import time
from dataclasses import dataclass


@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    reset_at: float
    limit: int


class RateLimiter:
    TIERS = {
        "anonymous": {"rpm": 30, "cpm": 200},
        "free": {"rpm": 60, "cpm": 500},
        "pro": {"rpm": 600, "cpm": 5000},
        "enterprise": {"rpm": 6000, "cpm": 50000},
    }

    def __init__(self, redis_url: str):
        self.redis = aioredis.from_url(redis_url)

    async def check(
        self, client_id: str, tier: str = "free", cost: int = 1
    ) -> RateLimitResult:
        """Sliding window rate limit check."""
        limits = self.TIERS.get(tier, self.TIERS["free"])
        window = 60  # 1 minute
        now = time.time()
        key = f"ratelimit:{client_id}:{int(now // window)}"

        pipe = self.redis.pipeline()
        pipe.incrby(key, cost)
        pipe.expire(key, window + 1)
        results = await pipe.execute()

        current = results[0]
        limit = limits["cpm"]
        remaining = max(0, limit - current)
        reset_at = (int(now // window) + 1) * window

        return RateLimitResult(
            allowed=current <= limit,
            remaining=remaining,
            reset_at=reset_at,
            limit=limit,
        )
```

### Circuit Breaker

```python
from enum import Enum
from datetime import datetime, timedelta


class CircuitState(Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = timedelta(seconds=recovery_timeout)
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.last_failure: datetime | None = None

    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if datetime.now() - self.last_failure > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN: allow one request

    def record_success(self):
        self.failures = 0
        self.state = CircuitState.CLOSED

    def record_failure(self):
        self.failures += 1
        self.last_failure = datetime.now()
        if self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN


# Per-service circuit breakers
circuit_breakers = {
    "users": CircuitBreaker(),
    "products": CircuitBreaker(),
    "orders": CircuitBreaker(),
}
```

---

## 7. REST to GraphQL Migration

### Migration Phases

```
Phase 1: Coexistence (Weeks 1-4)
  ├── Deploy GraphQL endpoint alongside REST
  ├── GraphQL resolvers call REST services internally
  ├── New features built in GraphQL only
  └── REST endpoints unchanged

Phase 2: Client Migration (Weeks 5-12)
  ├── Mobile apps switch to GraphQL (biggest benefit)
  ├── Web app migrates page by page
  ├── Monitor REST vs. GraphQL traffic ratio
  └── REST endpoints still active

Phase 3: REST Deprecation (Weeks 13-16)
  ├── Add Sunset headers to REST endpoints
  ├── Contact remaining REST consumers
  ├── Provide migration guides
  └── Set deprecation deadline

Phase 4: REST Sunset (Week 17+)
  ├── Redirect REST endpoints to documentation
  ├── Return 410 Gone after grace period
  ├── Remove REST code
  └── GraphQL as sole API
```

### Wrapping REST During Migration

```python
# Phase 1: GraphQL resolvers that delegate to REST
class UserRESTDataSource:
    """Data source that calls the existing REST API."""

    def __init__(self, base_url: str, http_client: httpx.AsyncClient):
        self.base_url = base_url
        self.client = http_client

    async def get_user(self, id: str) -> dict:
        resp = await self.client.get(f"{self.base_url}/api/v1/users/{id}")
        resp.raise_for_status()
        return resp.json()

    async def get_users(self, limit: int = 10, offset: int = 0) -> list[dict]:
        resp = await self.client.get(
            f"{self.base_url}/api/v1/users",
            params={"limit": limit, "offset": offset},
        )
        return resp.json()["items"]

    async def create_user(self, data: dict) -> dict:
        resp = await self.client.post(
            f"{self.base_url}/api/v1/users",
            json=data,
        )
        resp.raise_for_status()
        return resp.json()


# Phase 2: Gradually move to direct DB access
class UserDirectDataSource:
    """Data source that queries the database directly."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user(self, id: str) -> User | None:
        return await self.db.get(UserModel, id)
```

### Traffic Splitting

```python
# Feature flag: gradually shift traffic from REST to GraphQL
GRAPHQL_TRAFFIC_PERCENT = 50  # Start at 10%, increase to 100%

@app.get("/api/v1/users/{user_id}")
async def get_user(user_id: str, request: Request):
    """REST endpoint with gradual GraphQL migration."""
    import random

    if random.randint(1, 100) <= GRAPHQL_TRAFFIC_PERCENT:
        # Delegate to GraphQL
        result = await schema.execute(
            "query($id: ID!) { user(id: $id) { id username email } }",
            variable_values={"id": user_id},
        )
        if not result.errors:
            return result.data["user"]

    # Fallback to REST
    return await legacy_get_user(user_id)
```

---

## 8. GraphQL Schema Stitching

### Remote Schema Stitching

Combine multiple GraphQL endpoints into one:

```python
# Fetch and merge remote schemas
from graphql import build_schema, print_schema
from graphql.utilities import merge_schemas


async def fetch_remote_schema(url: str) -> str:
    """Fetch SDL from a remote GraphQL endpoint."""
    async with httpx.AsyncClient() as client:
        result = await client.post(url, json={
            "query": """
                query {
                    __schema {
                        types { name kind }
                        queryType { name }
                        mutationType { name }
                    }
                }
            """
        })
        return result.json()


# Stitching vs. Federation comparison
# +------------------+---------------------+
# | Schema Stitching | Federation          |
# +------------------+---------------------+
# | Gateway merges   | Router composes     |
# | schemas          | subgraph schemas    |
# | Manual conflict  | Directive-based     |
# | resolution       | resolution          |
# | Any GraphQL      | Requires federation |
# | server           | spec support        |
# | Simpler setup    | Better for large    |
# |                  | organizations       |
# +------------------+---------------------+
```

---

## 9. Observability and Monitoring

### Distributed Tracing

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

tracer = trace.get_tracer("api-gateway")


@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    with tracer.start_as_current_span(
        "gateway.request",
        attributes={
            "http.method": request.method,
            "http.url": str(request.url),
            "gateway.service": _get_target_service(request),
        },
    ) as span:
        response = await call_next(request)
        span.set_attribute("http.status_code", response.status_code)
        return response
```

### Gateway Metrics Dashboard

| Metric | Description | Alert |
|--------|-------------|-------|
| `gateway.requests.total` | Total requests per service | - |
| `gateway.requests.latency` | Request latency histogram | p99 > 5s |
| `gateway.requests.errors` | Error rate per service | > 5% |
| `gateway.circuit.state` | Circuit breaker state | OPEN |
| `gateway.ratelimit.rejections` | Rate-limited requests | > 100/min |
| `gateway.upstream.latency` | Per-service upstream latency | p95 > 2s |

### Health Check Aggregation

```python
@app.get("/health")
async def health(request: Request):
    """Aggregate health from all upstream services."""
    client = request.app.state.http_client
    services = {
        "users": "http://users-service:8001/health",
        "products": "http://products-service:8002/health",
        "orders": "http://orders-service:8003/health",
    }

    results = {}
    for name, url in services.items():
        try:
            resp = await client.get(url, timeout=5.0)
            results[name] = "healthy" if resp.status_code == 200 else "degraded"
        except Exception:
            results[name] = "unhealthy"

    overall = "healthy" if all(v == "healthy" for v in results.values()) else "degraded"
    status_code = 200 if overall == "healthy" else 503

    return JSONResponse(
        {"status": overall, "services": results},
        status_code=status_code,
    )
```

---

## 10. Capstone Project

### Project: Build a Unified API Gateway

Combine everything you have learned in lessons 01-24 to build a working unified API gateway.

### Requirements

1. **Three backend services**:
   - Users service (REST/FastAPI) — CRUD for users
   - Content service (GraphQL/Strawberry) — posts, comments
   - Orders service (gRPC or REST) — order management

2. **Gateway features**:
   - Single entry point at port 4000
   - GraphQL endpoint at `/graphql` that wraps all services
   - REST proxy at `/api/v1/{service}/{path}`
   - JWT authentication
   - Cost-based rate limiting
   - Circuit breaker per service
   - Request logging with structured output
   - Health check endpoint

3. **GraphQL schema** that unifies all services:
   ```graphql
   type Query {
     user(id: ID!): User
     me: User
     post(id: ID!): Post
     posts(first: Int, after: String): PostConnection!
     order(id: ID!): Order
     myOrders: [Order!]!
   }

   type User {
     id: ID!
     username: String!
     email: String!
     posts: [Post!]!
     orders: [Order!]!
   }

   type Post {
     id: ID!
     title: String!
     content: String!
     author: User!
     comments: [Comment!]!
   }

   type Order {
     id: ID!
     customer: User!
     items: [OrderItem!]!
     total: Float!
     status: OrderStatus!
   }
   ```

4. **Testing**:
   - Unit tests for GraphQL resolvers
   - Integration test for cross-service queries
   - Rate limiter tests
   - Circuit breaker tests

5. **Docker Compose** to run everything locally

### Suggested Implementation Order

```
Day 1: Set up three backend services with Docker Compose
Day 2: Build basic gateway with REST proxying
Day 3: Add GraphQL endpoint with resolvers wrapping REST/gRPC
Day 4: Implement authentication and rate limiting
Day 5: Add circuit breaker, health checks, logging
Day 6: Write tests (unit + integration)
Day 7: Documentation and review
```

---

## 11. Exercises

### Exercise 1: Protocol Translation

Implement a GraphQL resolver that wraps a REST endpoint for product search:
- REST endpoint: `GET /api/products?q=query&category=cat&limit=10&page=2`
- GraphQL query: `products(query: String, category: String, first: Int, after: String)`
- Convert REST pagination (page/limit) to cursor-based pagination

### Exercise 2: Gateway Rate Limiter

Implement a rate limiter for the gateway that:
- Uses Redis sliding window
- Applies different limits per auth tier (anonymous, free, pro, enterprise)
- Returns `X-RateLimit-*` headers
- Applies cost multipliers (mutations cost 5x, subscriptions cost 10x)

### Exercise 3: Circuit Breaker

Implement a circuit breaker that:
- Opens after 5 consecutive failures
- Half-opens after 30 seconds
- Returns a cached response or 503 when open
- Emits metrics when state changes

### Exercise 4: Migration Plan

Write a detailed migration plan for moving a REST API with 30 endpoints to GraphQL:
- Identify which endpoints map to queries vs. mutations
- Design the GraphQL schema
- Plan the phase timeline
- Write the `Sunset` header implementation
- Create a client migration guide

---

## 12. References

### API Gateways
- [Kong Gateway](https://docs.konghq.com/) — Open-source API gateway
- [AWS API Gateway](https://aws.amazon.com/api-gateway/)
- [Apollo Router](https://www.apollographql.com/docs/router/) — GraphQL-native gateway

### Architecture
- "Building Microservices" by Sam Newman (O'Reilly) — API Gateway pattern
- "API Architecture" by Matthias Biehl — BFF pattern
- [microservices.io — API Gateway Pattern](https://microservices.io/patterns/apigateway.html)

### Migration
- "From REST to GraphQL" — GitHub Engineering Blog
- "Migrating to GraphQL at Airbnb" — Airbnb Engineering
- "GraphQL at PayPal" — PayPal Engineering Blog

### Observability
- [OpenTelemetry Python SDK](https://opentelemetry.io/docs/languages/python/)
- [Grafana + Prometheus for API Monitoring](https://grafana.com/)

---

**License**: CC BY-NC 4.0
