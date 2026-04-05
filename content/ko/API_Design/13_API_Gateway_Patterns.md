# 13. API Gateway 패턴(API Gateway Patterns)

**이전**: [Webhook과 콜백](./12_Webhooks_and_Callbacks.md) | **다음**: [gRPC와 Protocol Buffers](./14_gRPC_and_Protocol_Buffers.md)

**난이도**: ⭐⭐⭐

---

## 학습 목표

- API gateway의 역할과 마이크로서비스 아키텍처에서의 책임을 설명할 수 있다
- Backend for Frontend (BFF) 패턴을 구현하여 각 클라이언트 유형에 맞는 API를 제공할 수 있다
- 여러 서비스의 데이터를 결합하는 API 합성(composition) 및 집계(aggregation) 전략을 설계할 수 있다
- 서비스 메시와 API gateway의 횡단 관심사(cross-cutting concerns) 처리 방식을 비교할 수 있다
- 인기 있는 API gateway(Kong, AWS API Gateway)의 라우팅, 인증, 속도 제한을 설정할 수 있다
- 중앙 집중식 gateway와 분산 gateway 패턴의 트레이드오프를 평가할 수 있다

---

## 목차

1. [API Gateway란?](#1-api-gateway란)
2. [Gateway의 책임](#2-gateway의-책임)
3. [Backend for Frontend (BFF) 패턴](#3-backend-for-frontend-bff-패턴)
4. [API 합성과 집계](#4-api-합성과-집계)
5. [서비스 메시 vs. API Gateway](#5-서비스-메시-vs-api-gateway)
6. [Kong API Gateway](#6-kong-api-gateway)
7. [AWS API Gateway](#7-aws-api-gateway)
8. [FastAPI로 Gateway 구축](#8-fastapi로-gateway-구축)
9. [연습 문제](#9-연습-문제)
10. [참고 자료](#10-참고-자료)

---

## 1. API Gateway란?

API gateway는 클라이언트와 백엔드 서비스 사이에 위치하는 서버입니다. 모든 API 요청의 단일 진입점 역할을 하며, 개별 서비스가 직접 처리할 필요 없는 횡단 관심사를 담당합니다.

### Gateway 없이

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

문제점: 클라이언트가 모든 서비스 주소를 알아야 합니다. 각 서비스가 인증, 속도 제한, CORS를 독립적으로 구현해야 합니다. 서비스 변경 시 모든 클라이언트가 영향을 받습니다.

### Gateway 사용

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

장점: 단일 진입점, 중앙 집중화된 횡단 관심사, 단순화된 클라이언트 로직, 외부 트래픽으로부터의 서비스 수준 격리.

---

## 2. Gateway의 책임

### 라우팅

Gateway는 URL 경로, HTTP 메서드 또는 헤더를 기반으로 수신 요청을 적절한 백엔드 서비스로 라우팅합니다.

```python
# Route configuration (conceptual)
routes = {
    "/api/users/**":        "http://user-service:8001",
    "/api/orders/**":       "http://order-service:8002",
    "/api/payments/**":     "http://payment-service:8003",
    "/api/notifications/**": "http://notification-service:8004",
}
```

### 인증

Gateway에서 인증을 중앙 집중화하여 개별 서비스가 토큰을 검증할 필요가 없도록 합니다:

```
Client → [Gateway: verify JWT] → Service (receives user_id in header)
```

Gateway가 JWT를 검증하고 사용자 ID를 내부 헤더로 전달합니다:

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

### 속도 제한

Gateway 수준에서 전역 속도 제한을 적용합니다:

```python
# Different limits for different route categories
RATE_LIMITS = {
    "/api/search/**":    {"limit": 20, "window": 60},     # Expensive
    "/api/auth/**":      {"limit": 10, "window": 60},     # Brute-force protection
    "/api/**":           {"limit": 100, "window": 60},    # Default
}
```

### 요청/응답 변환

Gateway는 하위 호환성을 유지하거나 클라이언트 요구 사항에 맞추기 위해 요청과 응답을 변환할 수 있습니다:

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

### 로깅과 관측성(Observability)

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

## 3. Backend for Frontend (BFF) 패턴

클라이언트 유형(웹, 모바일, IoT)마다 요구 사항이 다릅니다. 느린 네트워크의 모바일 앱은 최소한의 데이터가 필요하고, 웹 대시보드는 풍부하고 상세한 응답이 필요합니다. **BFF 패턴**은 각 클라이언트 유형에 전용 gateway를 생성합니다.

### 아키텍처

```
Mobile App ──→ [Mobile BFF] ──→ User Svc, Order Svc
Web App ─────→ [Web BFF] ────→ User Svc, Order Svc, Analytics Svc
Partner API ─→ [Partner BFF] ─→ User Svc, Order Svc
```

각 BFF는 해당 클라이언트의 특정 요구 사항에 맞게 조정됩니다:

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

### BFF 사용 시점

| 시나리오 | BFF 적합? | 이유 |
|----------|----------|------|
| 매우 다른 UI를 가진 웹 + 모바일 | 예 | 다른 데이터 요구 사항 |
| 유사한 UI를 가진 웹 + 모바일 | 아니오 | 공유 gateway가 더 단순 |
| 공개 + 파트너 API | 예 | 다른 인증, 속도 제한, 데이터 노출 |
| 단일 클라이언트 유형 | 아니오 | 불필요한 복잡성 추가 |

---

## 4. API 합성과 집계

Gateway는 여러 서비스의 응답을 하나로 결합하여 클라이언트의 왕복 횟수를 줄일 수 있습니다.

### 순차 합성(Sequential Composition)

한 서비스의 데이터가 다른 서비스를 쿼리하는 데 필요한 경우:

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

### 병렬 합성(Parallel Composition)

쿼리가 독립적인 경우 동시에 실행합니다:

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

### 부분 실패 처리

여러 서비스에서 응답을 합성할 때, 일부 서비스는 실패하고 다른 서비스는 성공할 수 있습니다. 집계 로직이 우아하게 성능을 저하시키도록 설계합니다:

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

## 5. 서비스 메시 vs. API Gateway

서비스 메시와 API gateway 모두 횡단 관심사를 처리하지만, 서로 다른 수준에서 동작합니다.

### API Gateway: 남북 트래픽(North-South Traffic)

외부 클라이언트와 내부 서비스 간의 트래픽을 처리합니다.

```
External Client
      │
      ▼
[API Gateway]  ← North-South (ingress)
      │
      ▼
  Services
```

### 서비스 메시: 동서 트래픽(East-West Traffic)

사이드카 프록시를 통해 내부 서비스 간의 트래픽을 처리합니다.

```
[Service A] ←→ [Sidecar A] ←→ [Sidecar B] ←→ [Service B]
                     │                │
                     └──── Mesh ──────┘
                     (mTLS, retries,
                      circuit breaking,
                      observability)
```

### 비교

| 항목 | API Gateway | 서비스 메시 |
|------|-------------|------------|
| 트래픽 | 남북 (외부 → 내부) | 동서 (내부 → 내부) |
| 배포 방식 | 중앙 집중형 서버 | 서비스당 사이드카 |
| 주요 용도 | 외부 API 관리 | 서비스 간 통신 |
| 인증 | JWT, API 키, OAuth | mTLS (상호 TLS) |
| 속도 제한 | 클라이언트별, API별 | 서비스별 |
| 프로토콜 | HTTP/REST, gRPC | 모든 프로토콜 (TCP, HTTP, gRPC) |
| 예시 | Kong, AWS API Gateway, Nginx | Istio, Linkerd, Envoy |

### 각각의 사용 시기

```
소규모 팀, 적은 서비스:
  → API Gateway만 사용 (Kong, Nginx)

성장 중인 마이크로서비스 (5-20개 서비스):
  → 외부 트래픽에 API Gateway
  → 공유 인증 라이브러리를 사용한 직접 서비스 간 호출

대규모 마이크로서비스 (20개 이상 서비스):
  → 외부 트래픽에 API Gateway
  → 내부 트래픽에 서비스 메시 (Istio, Linkerd)
```

---

## 6. Kong API Gateway

Kong은 Nginx 기반으로 구축된 오픈 소스 API gateway입니다. 인증, 속도 제한, 로깅 등을 위한 플러그인을 지원합니다.

### 선언적 설정

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

### Docker Compose 설정

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

AWS API Gateway는 대규모 API를 생성, 배포, 관리하기 위한 관리형 서비스입니다.

### Lambda 통합 REST API

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

### HTTP API (경량)

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

## 8. FastAPI로 Gateway 구축

소규모 아키텍처에서는 FastAPI로 경량 API gateway를 구축할 수 있습니다. 전용 gateway 인프라의 복잡성을 피하면서 완전한 제어를 할 수 있습니다.

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

### 서킷 브레이커(Circuit Breaker)

백엔드 서비스가 다운되었을 때 연쇄 실패(cascading failure)를 방지합니다:

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

## 9. 연습 문제

### 문제 1: 미니 API Gateway 구축

다음 기능을 갖춘 FastAPI 기반 API gateway를 구축하세요:

- URL 접두사를 기반으로 3개의 백엔드 서비스로 요청 라우팅
- `X-Request-Id`와 `X-Forwarded-For` 헤더 추가
- 모든 요청에 대해 메서드, 경로, upstream, 상태 코드, 지연 시간 로깅
- 백엔드 서비스에 접근할 수 없을 때 502 반환
- 백엔드 서비스 타임아웃 시 504 반환 (5초 타임아웃)

모의 백엔드 서비스로 테스트하세요 (`uvicorn`을 사용하여 다른 포트에서 간단한 FastAPI 앱 실행).

### 문제 2: BFF 구현

전자상거래 플랫폼을 위한 두 개의 BFF gateway를 설계하고 구현하세요:

**Mobile BFF** (`/mobile/...`):
- `GET /mobile/home` — 집계: 사용자 프로필 (이름, 아바타만), 최근 3개 주문 (id, 상태만), 추천 상품 5개 (id, 이름, 가격, 썸네일만)
- 대역폭 최적화: 최소 필드, 작은 페이로드

**Web BFF** (`/web/...`):
- `GET /web/dashboard` — 집계: 전체 사용자 프로필, 라인 아이템이 포함된 최근 20개 주문, 분석 요약, 알림 수
- 데스크톱 디스플레이를 위한 풍부한 데이터

두 BFF 모두 동일한 백엔드 서비스를 호출하지만 다른 형태의 응답을 반환해야 합니다.

### 문제 3: 서킷 브레이커

문제 1의 gateway에 서킷 브레이커를 구현하세요. 서킷 브레이커는 다음을 수행해야 합니다:

- 백엔드 서비스별로 독립적으로 실패 추적
- 5회 연속 실패 후 서킷 개방
- 서킷이 열려 있을 때 `Retry-After` 헤더와 함께 503 반환
- 30초 후 반개방(half-open) 상태로 전환
- 반개방 상태에서 3회 연속 성공 후 서킷 닫기
- 모든 서킷 브레이커의 상태를 보여주는 `GET /admin/circuits` 엔드포인트 제공

### 문제 4: 부분 실패를 다루는 API 합성

4개의 백엔드 서비스를 호출하는 상품 상세 페이지 집계 엔드포인트를 구축하세요:

- 상품 서비스 (필수): 상품 상세 정보
- 리뷰 서비스 (선택): 사용자 리뷰
- 재고 서비스 (선택): 재고 현황
- 추천 서비스 (선택): 관련 상품

상품 서비스가 실패하면 502를 반환합니다. 선택 서비스가 실패하면 사용 가능한 데이터를 반환하되, 사용 불가능한 서비스를 나열하는 `_degraded` 필드를 포함합니다. 서비스당 3초 타임아웃으로 동시 요청을 구현하세요.

### 문제 5: Gateway 설정 DSL

다음을 지원하는 API gateway의 설정 형식(YAML)을 설계하세요:

- 경로 패턴, upstream URL, HTTP 메서드를 포함한 라우트 정의
- 라우트별 인증 요구 사항 (없음, JWT, API 키)
- 라우트별 속도 제한
- 요청/응답 변환 (헤더 추가/제거, 본문 필드 매핑)
- upstream별 서킷 브레이커 설정

이 설정을 읽고 FastAPI gateway 코드를 생성하는 Python 파서를 작성하세요 (또는 범용 gateway 애플리케이션을 구성).

---

## 10. 참고 자료

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

**이전**: [Webhook과 콜백](./12_Webhooks_and_Callbacks.md) | [개요](./00_Overview.md) | **다음**: [gRPC와 Protocol Buffers](./14_gRPC_and_Protocol_Buffers.md)

**License**: CC BY-NC 4.0
