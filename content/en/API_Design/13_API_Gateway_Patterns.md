# 13. API Gateway Patterns

**Previous**: [Webhooks and Callbacks](./12_Webhooks_and_Callbacks.md) | **Next**: [gRPC and Protocol Buffers](./14_gRPC_and_Protocol_Buffers.md)

**Difficulty**: ⭐⭐⭐

---

## Learning Objectives

- Explain the responsibilities of an API gateway and its role in microservices architecture
- Implement the Backend for Frontend (BFF) pattern to tailor APIs for different client types
- Design API composition and aggregation strategies that combine data from multiple services
- Compare service mesh and API gateway approaches for cross-cutting concerns
- Configure popular API gateways (Kong, AWS API Gateway) for routing, authentication, and rate limiting
- Evaluate trade-offs between centralized gateway and distributed gateway patterns

---

## Table of Contents

1. [What Is an API Gateway?](#1-what-is-an-api-gateway)
2. [Gateway Responsibilities](#2-gateway-responsibilities)
3. [Backend for Frontend (BFF) Pattern](#3-backend-for-frontend-bff-pattern)
4. [API Composition and Aggregation](#4-api-composition-and-aggregation)
5. [Service Mesh vs. API Gateway](#5-service-mesh-vs-api-gateway)
6. [Kong API Gateway](#6-kong-api-gateway)
7. [AWS API Gateway](#7-aws-api-gateway)
8. [Building a Gateway with FastAPI](#8-building-a-gateway-with-fastapi)
9. [Exercises](#9-exercises)
10. [References](#10-references)

---

## 1. What Is an API Gateway?

An API gateway is a server that sits between clients and backend services. It acts as a single entry point for all API requests, handling cross-cutting concerns so that individual services do not have to.

### Without a Gateway

```
Mobile App ──────────→ User Service
Mobile App ──────────→ Order Service
Mobile App ──────────→ Payment Service
Mobile App ──────────→ Notification Service

Web App ─────────────→ User Service
Web App ─────────────→ Order Service
Web App ─────────────→ Payment Service

Partner API ─────────→ User Service
Partner API ─────────→ Order Service
```

Problems: Clients must know every service address. Each service must implement authentication, rate limiting, and CORS independently. Service changes break all clients.

### With a Gateway

```
Mobile App ──┐
Web App ─────┤
Partner API ─┘
      │
      ▼
┌──────────────────┐
│   API Gateway     │
│                   │
│  • Routing        │
│  • Authentication │
│  • Rate Limiting  │
│  • Load Balancing │
│  • Logging        │
└────────┬─────────┘
         │
    ┌────┴────┐────────┐────────────┐
    ▼         ▼        ▼            ▼
User Svc  Order Svc  Payment Svc  Notif Svc
```

Benefits: Single entry point, centralized cross-cutting concerns, simplified client logic, service-level isolation from external traffic.

---

## 2. Gateway Responsibilities

### Routing

The gateway routes incoming requests to the appropriate backend service based on URL path, HTTP method, or headers.

```python
# Route configuration (conceptual)
routes = {
    "/api/users/**":        "http://user-service:8001",
    "/api/orders/**":       "http://order-service:8002",
    "/api/payments/**":     "http://payment-service:8003",
    "/api/notifications/**": "http://notification-service:8004",
}
```

### Authentication

Centralize authentication at the gateway so individual services do not need to verify tokens:

```
Client → [Gateway: verify JWT] → Service (receives user_id in header)
```

The gateway validates the JWT and passes user identity as an internal header:

```python
# Gateway middleware: JWT validation
async def auth_middleware(request: Request, call_next):
    token = request.headers.get("Authorization", "").removeprefix("Bearer ")

    if not token:
        return JSONResponse(status_code=401, content={"detail": "Missing token"})

    try:
        payload = jwt.decode(token, PUBLIC_KEY, algorithms=["RS256"])
    except jwt.InvalidTokenError:
        return JSONResponse(status_code=401, content={"detail": "Invalid token"})

    # Forward user identity to backend services via internal headers
    request.state.headers_to_forward = {
        "X-User-Id": str(payload["sub"]),
        "X-User-Role": payload.get("role", "user"),
        "X-Request-Id": request.headers.get("X-Request-Id", str(uuid.uuid4())),
    }

    return await call_next(request)
```

### Rate Limiting

Apply rate limits globally at the gateway level:

```python
# Different limits for different route categories
RATE_LIMITS = {
    "/api/search/**":    {"limit": 20, "window": 60},     # Expensive
    "/api/auth/**":      {"limit": 10, "window": 60},     # Brute-force protection
    "/api/**":           {"limit": 100, "window": 60},    # Default
}
```

### Request/Response Transformation

The gateway can transform requests and responses to maintain backward compatibility or adapt to client needs:

```python
async def transform_response(response_body: dict, api_version: str) -> dict:
    """Transform the response based on the client's API version.

    v1 clients expect a flat structure; v2 clients expect nested.
    """
    if api_version == "v1":
        # Flatten nested price object for v1 compatibility
        if "price" in response_body and isinstance(response_body["price"], dict):
            response_body["price"] = response_body["price"]["amount"]
    return response_body
```

### Logging and Observability

```python
import time
import uuid
import logging

logger = logging.getLogger("gateway")


async def observability_middleware(request: Request, call_next):
    """Add tracing headers and log request metrics."""
    request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
    start = time.perf_counter()

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Structured log for aggregation
    logger.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round(elapsed_ms, 2),
            "client_ip": request.client.host,
            "user_agent": request.headers.get("User-Agent", ""),
        },
    )

    # Add tracing headers to response
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"

    return response
```

---

## 3. Backend for Frontend (BFF) Pattern

Different client types (web, mobile, IoT) have different needs. A mobile app on a slow network needs minimal data; a web dashboard needs rich, detailed responses. The **BFF pattern** creates a dedicated gateway for each client type.

### Architecture

```
Mobile App ──→ [Mobile BFF] ──→ User Svc, Order Svc
Web App ─────→ [Web BFF] ────→ User Svc, Order Svc, Analytics Svc
Partner API ─→ [Partner BFF] ─→ User Svc, Order Svc
```

Each BFF is tailored to its client's specific needs:

```python
from fastapi import FastAPI
import httpx

# Mobile BFF: minimal data, optimized for bandwidth
mobile_app = FastAPI(title="Mobile BFF")


@mobile_app.get("/home")
async def mobile_home(user_id: int):
    """Mobile home screen: minimal data in a single request.

    Aggregates user profile and recent orders into one response,
    reducing the number of network round-trips from the mobile app.
    """
    async with httpx.AsyncClient() as client:
        user_resp, orders_resp = await asyncio.gather(
            client.get(f"http://user-service:8001/users/{user_id}"),
            client.get(
                f"http://order-service:8002/users/{user_id}/orders",
                params={"limit": 3, "fields": "id,status,total"},
            ),
        )

    user = user_resp.json()
    orders = orders_resp.json()

    # Return a compact response optimized for mobile
    return {
        "user": {
            "name": user["name"],
            "avatar_url": user["avatar_url"],
        },
        "recent_orders": [
            {"id": o["id"], "status": o["status"], "total": o["total"]}
            for o in orders["data"]
        ],
    }


# Web BFF: rich data for desktop dashboard
web_app = FastAPI(title="Web BFF")


@web_app.get("/dashboard")
async def web_dashboard(user_id: int):
    """Web dashboard: rich data including analytics.

    Desktop clients have more bandwidth, so include detailed
    user profile, full order history, and analytics summary.
    """
    async with httpx.AsyncClient() as client:
        user_resp, orders_resp, analytics_resp = await asyncio.gather(
            client.get(f"http://user-service:8001/users/{user_id}"),
            client.get(
                f"http://order-service:8002/users/{user_id}/orders",
                params={"limit": 20},
            ),
            client.get(
                f"http://analytics-service:8005/users/{user_id}/summary"
            ),
        )

    return {
        "user": user_resp.json(),
        "orders": orders_resp.json(),
        "analytics": analytics_resp.json(),
    }
```

### When to Use BFF

| Scenario | BFF Appropriate? | Reason |
|----------|-----------------|--------|
| Web + mobile with very different UIs | Yes | Different data needs |
| Web + mobile with similar UIs | No | Shared gateway is simpler |
| Public + partner APIs | Yes | Different auth, rate limits, data exposure |
| Single client type | No | Adds unnecessary complexity |

---

## 4. API Composition and Aggregation

The gateway can combine responses from multiple services into a single response, reducing round-trips for the client.

### Sequential Composition

When data from one service is needed to query another:

```python
@app.get("/users/{user_id}/order-summary")
async def get_order_summary(user_id: int):
    """Compose data from user and order services sequentially.

    First fetch the user (to get their name), then fetch their orders.
    Sequential because the order query depends on the user existing.
    """
    async with httpx.AsyncClient() as client:
        # Step 1: Get user
        user_resp = await client.get(
            f"http://user-service:8001/users/{user_id}"
        )
        if user_resp.status_code == 404:
            raise HTTPException(status_code=404, detail="User not found")
        user = user_resp.json()

        # Step 2: Get orders (depends on user existing)
        orders_resp = await client.get(
            f"http://order-service:8002/users/{user_id}/orders"
        )
        orders = orders_resp.json()

    return {
        "user_name": user["name"],
        "total_orders": len(orders["data"]),
        "total_spent": sum(o["total"] for o in orders["data"]),
        "orders": orders["data"],
    }
```

### Parallel Composition

When queries are independent, execute them concurrently:

```python
@app.get("/products/{product_id}/details")
async def get_product_details(product_id: int):
    """Compose data from multiple services in parallel.

    Product details, reviews, and inventory are independent,
    so we fetch them concurrently to minimize latency.
    """
    async with httpx.AsyncClient() as client:
        product_task = client.get(
            f"http://product-service:8001/products/{product_id}"
        )
        reviews_task = client.get(
            f"http://review-service:8002/products/{product_id}/reviews",
            params={"limit": 5, "sort": "-rating"},
        )
        inventory_task = client.get(
            f"http://inventory-service:8003/products/{product_id}/stock"
        )

        # Execute all three requests concurrently
        product_resp, reviews_resp, inventory_resp = await asyncio.gather(
            product_task, reviews_task, inventory_task,
            return_exceptions=True,
        )

    # Handle partial failures gracefully
    product = product_resp.json() if not isinstance(product_resp, Exception) else None
    if product is None:
        raise HTTPException(status_code=502, detail="Product service unavailable")

    reviews = (
        reviews_resp.json()["data"]
        if not isinstance(reviews_resp, Exception)
        else []
    )

    inventory = (
        inventory_resp.json()
        if not isinstance(inventory_resp, Exception)
        else {"in_stock": None, "quantity": None}
    )

    return {
        "product": product,
        "reviews": reviews,
        "inventory": inventory,
    }
```

### Partial Failure Handling

When composing responses from multiple services, some services may fail while others succeed. Design your aggregation to degrade gracefully:

```python
from dataclasses import dataclass
from enum import Enum


class ServiceStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass
class ServiceResult:
    """Result from a backend service call."""
    status: ServiceStatus
    data: dict | None = None
    error: str | None = None


async def call_service_safely(
    client: httpx.AsyncClient,
    url: str,
    timeout: float = 5.0,
) -> ServiceResult:
    """Call a backend service with timeout and error handling.

    Returns a ServiceResult regardless of success or failure,
    allowing the aggregator to decide how to handle partial data.
    """
    try:
        response = await client.get(url, timeout=timeout)
        if response.status_code == 200:
            return ServiceResult(status=ServiceStatus.OK, data=response.json())
        else:
            return ServiceResult(
                status=ServiceStatus.DEGRADED,
                error=f"HTTP {response.status_code}",
            )
    except httpx.TimeoutException:
        return ServiceResult(
            status=ServiceStatus.UNAVAILABLE,
            error="Service timed out",
        )
    except httpx.ConnectError:
        return ServiceResult(
            status=ServiceStatus.UNAVAILABLE,
            error="Service unreachable",
        )
```

---

## 5. Service Mesh vs. API Gateway

Both service meshes and API gateways handle cross-cutting concerns, but they operate at different levels.

### API Gateway: North-South Traffic

Handles traffic between external clients and internal services.

```
External Client
      │
      ▼
[API Gateway]  ← North-South (ingress)
      │
      ▼
  Services
```

### Service Mesh: East-West Traffic

Handles traffic between internal services via sidecar proxies.

```
[Service A] ←→ [Sidecar A] ←→ [Sidecar B] ←→ [Service B]
                     │                │
                     └──── Mesh ──────┘
                     (mTLS, retries,
                      circuit breaking,
                      observability)
```

### Comparison

| Aspect | API Gateway | Service Mesh |
|--------|-------------|--------------|
| Traffic | North-south (external → internal) | East-west (internal → internal) |
| Deployed as | Centralized server | Sidecar per service |
| Primary use | External API management | Service-to-service communication |
| Auth | JWT, API keys, OAuth | mTLS (mutual TLS) |
| Rate limiting | Per-client, per-API | Per-service |
| Protocol | HTTP/REST, gRPC | Any (TCP, HTTP, gRPC) |
| Examples | Kong, AWS API Gateway, Nginx | Istio, Linkerd, Envoy |

### When to Use Each

```
Small team, few services:
  → API Gateway only (Kong, Nginx)

Growing microservices (5-20 services):
  → API Gateway for external traffic
  → Direct service-to-service calls with shared auth library

Large microservices (20+ services):
  → API Gateway for external traffic
  → Service mesh for internal traffic (Istio, Linkerd)
```

---

## 6. Kong API Gateway

Kong is an open-source API gateway built on Nginx. It supports plugins for authentication, rate limiting, logging, and more.

### Declarative Configuration

```yaml
# kong.yml
_format_version: "3.0"

services:
  - name: user-service
    url: http://user-service:8001
    routes:
      - name: user-routes
        paths:
          - /api/users
        strip_path: false

  - name: order-service
    url: http://order-service:8002
    routes:
      - name: order-routes
        paths:
          - /api/orders
        strip_path: false

plugins:
  # Global plugins (apply to all routes)
  - name: cors
    config:
      origins:
        - "https://app.example.com"
      methods:
        - GET
        - POST
        - PUT
        - DELETE
      headers:
        - Authorization
        - Content-Type
      max_age: 3600

  - name: rate-limiting
    config:
      minute: 100
      policy: redis
      redis_host: redis
      redis_port: 6379

  - name: jwt
    config:
      key_claim_name: sub
      claims_to_verify:
        - exp

  - name: request-transformer
    config:
      add:
        headers:
          - "X-Gateway: kong"
          - "X-Request-Start: $(now)"

# Per-consumer rate limiting
consumers:
  - username: mobile-app
    plugins:
      - name: rate-limiting
        config:
          minute: 200

  - username: partner-api
    plugins:
      - name: rate-limiting
        config:
          minute: 1000
```

### Docker Compose Setup

```yaml
# docker-compose.yml
version: "3.8"

services:
  kong:
    image: kong:3.6
    environment:
      KONG_DATABASE: "off"
      KONG_DECLARATIVE_CONFIG: /kong/kong.yml
      KONG_PROXY_ACCESS_LOG: /dev/stdout
      KONG_ADMIN_ACCESS_LOG: /dev/stdout
      KONG_PROXY_ERROR_LOG: /dev/stderr
    ports:
      - "8000:8000"    # Proxy port
      - "8001:8001"    # Admin API
    volumes:
      - ./kong.yml:/kong/kong.yml:ro

  user-service:
    build: ./services/user
    expose:
      - "8001"

  order-service:
    build: ./services/order
    expose:
      - "8002"

  redis:
    image: redis:7-alpine
    expose:
      - "6379"
```

---

## 7. AWS API Gateway

AWS API Gateway is a managed service for creating, deploying, and managing APIs at scale.

### REST API with Lambda Integration

```python
# infrastructure/api_gateway.py (AWS CDK)
from aws_cdk import (
    aws_apigateway as apigw,
    aws_lambda as lambda_,
    Stack,
)
from constructs import Construct


class ApiGatewayStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # Lambda function for handling requests
        handler = lambda_.Function(
            self, "ApiHandler",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="main.handler",
            code=lambda_.Code.from_asset("lambda"),
        )

        # API Gateway
        api = apigw.RestApi(
            self, "BookstoreApi",
            rest_api_name="Bookstore API",
            description="API Gateway for the Bookstore",
            deploy_options=apigw.StageOptions(
                stage_name="prod",
                throttling_rate_limit=100,
                throttling_burst_limit=200,
            ),
        )

        # Resources and methods
        books = api.root.add_resource("books")
        books.add_method(
            "GET",
            apigw.LambdaIntegration(handler),
        )
        books.add_method(
            "POST",
            apigw.LambdaIntegration(handler),
            authorization_type=apigw.AuthorizationType.IAM,
        )

        book = books.add_resource("{book_id}")
        book.add_method("GET", apigw.LambdaIntegration(handler))
        book.add_method("PUT", apigw.LambdaIntegration(handler))
        book.add_method("DELETE", apigw.LambdaIntegration(handler))

        # Usage plan with API key
        plan = api.add_usage_plan(
            "ProPlan",
            name="Pro",
            throttle=apigw.ThrottleSettings(
                rate_limit=500,
                burst_limit=1000,
            ),
            quota=apigw.QuotaSettings(
                limit=100_000,
                period=apigw.Period.MONTH,
            ),
        )

        api_key = api.add_api_key("ProApiKey")
        plan.add_api_key(api_key)
        plan.add_api_stage(stage=api.deployment_stage)
```

### HTTP API (Lightweight)

```python
# AWS CDK: HTTP API (cheaper, simpler than REST API)
from aws_cdk import aws_apigatewayv2 as apigwv2
from aws_cdk.aws_apigatewayv2_integrations import HttpUrlIntegration


class HttpApiStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # HTTP API with direct service integration
        api = apigwv2.HttpApi(
            self, "BookstoreHttpApi",
            api_name="Bookstore HTTP API",
            cors_preflight=apigwv2.CorsPreflightOptions(
                allow_origins=["https://app.example.com"],
                allow_methods=[
                    apigwv2.CorsHttpMethod.GET,
                    apigwv2.CorsHttpMethod.POST,
                ],
            ),
        )

        # Route to ECS/Fargate service
        api.add_routes(
            path="/api/books/{proxy+}",
            methods=[apigwv2.HttpMethod.ANY],
            integration=HttpUrlIntegration(
                "BookService",
                "http://book-service.internal:8001/{proxy}",
            ),
        )
```

---

## 8. Building a Gateway with FastAPI

For smaller architectures, you can build a lightweight API gateway with FastAPI. This gives you full control and avoids the complexity of dedicated gateway infrastructure.

```python
import httpx
import asyncio
import uuid
import time
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, JSONResponse
from dataclasses import dataclass

logger = logging.getLogger("gateway")


@dataclass
class ServiceRoute:
    """Configuration for a backend service route."""
    prefix: str
    upstream: str
    timeout: float = 10.0
    strip_prefix: bool = True


# Route configuration
ROUTES = [
    ServiceRoute(prefix="/api/users", upstream="http://user-service:8001"),
    ServiceRoute(prefix="/api/orders", upstream="http://order-service:8002"),
    ServiceRoute(prefix="/api/payments", upstream="http://payment-service:8003"),
]


app = FastAPI(title="API Gateway")


def find_route(path: str) -> tuple[ServiceRoute, str] | None:
    """Match a request path to a backend service route."""
    for route in ROUTES:
        if path.startswith(route.prefix):
            if route.strip_prefix:
                remaining = path[len(route.prefix):]
                upstream_path = remaining or "/"
            else:
                upstream_path = path
            return route, upstream_path
    return None


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def gateway_proxy(request: Request, path: str):
    """Reverse proxy: forward requests to the appropriate backend service."""
    full_path = f"/{path}"
    match = find_route(full_path)

    if match is None:
        raise HTTPException(status_code=404, detail="Route not found")

    route, upstream_path = match
    upstream_url = f"{route.upstream}{upstream_path}"

    # Forward query parameters
    if request.url.query:
        upstream_url += f"?{request.url.query}"

    # Build forwarded headers
    forward_headers = dict(request.headers)
    forward_headers.pop("host", None)
    forward_headers["X-Request-Id"] = request.headers.get(
        "X-Request-Id", str(uuid.uuid4())
    )
    forward_headers["X-Forwarded-For"] = request.client.host
    forward_headers["X-Forwarded-Proto"] = request.url.scheme

    # Read request body
    body = await request.body()

    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=route.timeout) as client:
            response = await client.request(
                method=request.method,
                url=upstream_url,
                headers=forward_headers,
                content=body,
            )
    except httpx.TimeoutException:
        logger.error(f"Upstream timeout: {upstream_url}")
        return JSONResponse(
            status_code=504,
            content={"detail": "Upstream service timed out"},
        )
    except httpx.ConnectError:
        logger.error(f"Upstream unreachable: {upstream_url}")
        return JSONResponse(
            status_code=502,
            content={"detail": "Upstream service unavailable"},
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {full_path} → {upstream_url} "
        f"status={response.status_code} latency={elapsed_ms:.1f}ms"
    )

    # Build response with upstream headers
    response_headers = dict(response.headers)
    response_headers["X-Request-Id"] = forward_headers["X-Request-Id"]
    response_headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
    response_headers.pop("transfer-encoding", None)

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=response_headers,
    )
```

### Circuit Breaker

Prevent cascading failures when a backend service is down:

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Failing, reject requests
    HALF_OPEN = "half_open"    # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker to protect against cascading failures.

    States:
    - CLOSED: Requests pass through normally. Track failures.
    - OPEN: All requests are rejected immediately (503).
    - HALF_OPEN: Allow one test request. If it succeeds, close.
                 If it fails, reopen.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0  # seconds
    _state: CircuitState = CircuitState.CLOSED
    _failure_count: int = 0
    _last_failure_time: float = 0.0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def record_success(self):
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN

    def can_request(self) -> bool:
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            return True  # Allow one test request
        return False  # OPEN: reject


# One circuit breaker per backend service
circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(service_name: str) -> CircuitBreaker:
    if service_name not in circuit_breakers:
        circuit_breakers[service_name] = CircuitBreaker()
    return circuit_breakers[service_name]
```

---

## 9. Exercises

### Exercise 1: Build a Mini API Gateway

Build a FastAPI-based API gateway that:

- Routes requests to 3 backend services based on URL prefix
- Adds `X-Request-Id` and `X-Forwarded-For` headers
- Logs every request with method, path, upstream, status code, and latency
- Returns 502 when a backend service is unreachable
- Returns 504 when a backend service times out (5-second timeout)

Test with mock backend services (use `uvicorn` to run simple FastAPI apps on different ports).

### Exercise 2: BFF Implementation

Design and implement two BFF gateways for an e-commerce platform:

**Mobile BFF** (`/mobile/...`):
- `GET /mobile/home` — Aggregates: user profile (name, avatar only), 3 recent orders (id, status only), 5 recommended products (id, name, price, thumbnail only)
- Optimized for bandwidth: minimal fields, small payloads

**Web BFF** (`/web/...`):
- `GET /web/dashboard` — Aggregates: full user profile, 20 recent orders with line items, analytics summary, notification count
- Rich data for desktop display

Both BFFs should call the same backend services but return differently shaped responses.

### Exercise 3: Circuit Breaker

Implement a circuit breaker for the gateway from Exercise 1. The circuit breaker should:

- Track failures per backend service independently
- Open the circuit after 5 consecutive failures
- Return 503 with `Retry-After` header when the circuit is open
- Enter half-open state after 30 seconds
- Close the circuit after 3 consecutive successes in half-open state
- Expose a `GET /admin/circuits` endpoint showing the state of all circuit breakers

### Exercise 4: API Composition with Partial Failure

Build a product detail page aggregation endpoint that calls 4 backend services:

- Product service (required): product details
- Review service (optional): user reviews
- Inventory service (optional): stock availability
- Recommendation service (optional): related products

If the product service fails, return 502. If any optional service fails, return what you have with a `_degraded` field listing the unavailable services. Implement concurrent requests with a 3-second timeout per service.

### Exercise 5: Gateway Configuration DSL

Design a configuration format (YAML) for an API gateway that supports:

- Route definitions with path patterns, upstream URLs, and HTTP methods
- Authentication requirements per route (none, JWT, API key)
- Rate limiting per route
- Request/response transformations (header addition/removal, body field mapping)
- Circuit breaker settings per upstream

Write a Python parser that reads this configuration and generates the FastAPI gateway code (or configures a generic gateway application).

---

## 10. References

- [Kong Gateway Documentation](https://docs.konghq.com/)
- [AWS API Gateway Developer Guide](https://docs.aws.amazon.com/apigateway/)
- [Nginx as API Gateway](https://www.nginx.com/blog/deploying-nginx-plus-as-an-api-gateway-part-1/)
- [Sam Newman: Building Microservices, Chapter 8 (API Gateways)](https://www.oreilly.com/library/view/building-microservices/9781492034018/)
- [Microsoft: Gateway Pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/gateway-routing)
- [BFF Pattern (Sam Newman)](https://samnewman.io/patterns/architectural/bff/)
- [Istio Service Mesh](https://istio.io/latest/docs/)
- [Envoy Proxy](https://www.envoyproxy.io/)
- [Martin Fowler: Circuit Breaker](https://martinfowler.com/bliki/CircuitBreaker.html)

---

**Previous**: [Webhooks and Callbacks](./12_Webhooks_and_Callbacks.md) | [Overview](./00_Overview.md) | **Next**: [gRPC and Protocol Buffers](./14_gRPC_and_Protocol_Buffers.md)

**License**: CC BY-NC 4.0
