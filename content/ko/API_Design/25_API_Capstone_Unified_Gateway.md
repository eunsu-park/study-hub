# 25. API 캡스톤 — 통합 게이트웨이(API Capstone — Unified Gateway)

**이전**: [GraphQL 테스팅과 도구](./24_GraphQL_Testing_and_Tooling.md) | **다음**: 없음 (최종 레슨)

**난이도**: ⭐⭐⭐⭐

---

## 학습 목표

- REST, GraphQL, gRPC 백엔드로 트래픽을 라우팅하는 통합 API 게이트웨이를 설계할 수 있다
- GraphQL 쿼리와 REST/gRPC 서비스 호출 간의 프로토콜 변환을 구현할 수 있다
- 게이트웨이 레이어에서 요청 라우팅, 인증, 속도 제한을 적용할 수 있다
- 기존 클라이언트를 깨뜨리지 않고 REST에서 GraphQL로의 마이그레이션 전략을 계획하고 실행할 수 있다
- 게이트웨이 패턴(API 게이트웨이, BFF, 스키마 스티칭)을 평가할 수 있다
- 세 가지 프로토콜을 결합하는 Python 프로토타입 게이트웨이를 구축할 수 있다

---

## 목차

1. [통합 게이트웨이 개념](#1-통합-게이트웨이-개념)
2. [게이트웨이 아키텍처 패턴](#2-게이트웨이-아키텍처-패턴)
3. [프로토콜 변환](#3-프로토콜-변환)
4. [게이트웨이 구축](#4-게이트웨이-구축)
5. [게이트웨이에서의 인증](#5-게이트웨이에서의-인증)
6. [속도 제한과 트래픽 관리](#6-속도-제한과-트래픽-관리)
7. [REST에서 GraphQL로 마이그레이션](#7-rest에서-graphql로-마이그레이션)
8. [GraphQL 스키마 스티칭](#8-graphql-스키마-스티칭)
9. [관찰 가능성과 모니터링](#9-관찰-가능성과-모니터링)
10. [캡스톤 프로젝트](#10-캡스톤-프로젝트)
11. [연습 문제](#11-연습-문제)
12. [참고 자료](#12-참고-자료)

---

## 1. 통합 게이트웨이 개념

현대 API는 종종 여러 프로토콜을 결합합니다. 통합 게이트웨이는 요청을 적절한 백엔드로 라우팅하는 단일 진입점을 제공합니다.

### 통합 게이트웨이의 필요성

| 과제 | 게이트웨이 해결책 |
|------|----------------|
| 클라이언트가 여러 엔드포인트를 알아야 함 | 단일 진입점 |
| 서비스별 다른 인증 메커니즘 | 중앙 집중식 인증 |
| 일관되지 않은 속도 제한 | 통합 정책 적용 |
| 횡단 관심사 중복 | 게이트웨이에서 한 번 적용 |
| 프로토콜별 클라이언트 필요 | 프로토콜 변환 |

### 아키텍처 개요

```
                         ┌──────────────────────┐
    모바일 앱 ──────────▶ │                      │ ──▶ REST 서비스 (Users)
    웹 앱 ────────────▶ │   통합 게이트웨이      │ ──▶ GraphQL 서비스 (Content)
    파트너 API ────────▶ │                      │ ──▶ gRPC 서비스 (Orders)
    내부 도구 ──────────▶ │  - 인증              │ ──▶ gRPC 서비스 (Inventory)
                         │  - 속도 제한          │ ──▶ REST 서비스 (Payments)
                         │  - 프로토콜 라우팅     │ ──▶ 이벤트 버스 (Notifications)
                         │  - 로깅              │
                         └──────────────────────┘
```

### 프로토콜 강점(Protocol Strengths)

| 프로토콜 | 최적 사용 사례 | 게이트웨이 노출 방식 |
|----------|--------------|-------------------|
| REST | 공개 API, CRUD, 캐싱 | 직접 통과 |
| GraphQL | 복잡한 쿼리, 모바일 클라이언트 | 쿼리 엔드포인트 |
| gRPC | 내부 서비스, 스트리밍 | REST/GraphQL로 변환 |
| WebSocket | 실시간 구독(subscription) | 프록시 |

---

## 2. 게이트웨이 아키텍처 패턴

### 패턴 1: 단순 리버스 프록시

URL 경로로 라우팅:

```
/api/users/*    → REST Users 서비스
/api/graphql    → GraphQL Content 서비스
/api/orders/*   → gRPC Orders 서비스 (grpc-gateway 경유)
```

```python
# FastAPI를 사용한 단순 라우팅
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
    """요청을 적절한 백엔드 서비스로 라우팅합니다."""
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

### 패턴 2: Backend-for-Frontend (BFF)

각 클라이언트 유형에 맞춤형 게이트웨이를 제공합니다:

```
모바일 BFF ──▶ REST (users) + GraphQL (content, 모바일 최적화)
웹 BFF    ──▶ REST (users) + GraphQL (content, 전체 데이터)
파트너 BFF──▶ REST (users, 속도 제한) + REST (orders, 읽기 전용)
```

```python
# 모바일 BFF: 모바일 클라이언트에 최적화된 집계
@app.get("/mobile/home")
async def mobile_home(request: Request):
    """모바일 클라이언트를 위한 집계된 홈 화면 데이터."""
    async with httpx.AsyncClient() as client:
        # 여러 서비스에 병렬 요청
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

### 패턴 3: GraphQL 게이트웨이 (스키마 스티칭 / 페더레이션)

```
Client → GraphQL Gateway → REST 서비스 (리졸버로 래핑)
                         → GraphQL 서브그래프
                         → gRPC 서비스 (리졸버로 래핑)
```

### 패턴 비교

| 패턴 | 복잡도 | 유연성 | 적합 사례 |
|------|--------|--------|----------|
| 리버스 프록시 | 낮음 | 낮음 | 단순 라우팅 |
| BFF | 중간 | 높음 | 다중 클라이언트 유형 |
| GraphQL 게이트웨이 | 높음 | 매우 높음 | 복잡한 데이터 요구사항 |
| 서비스 메시 | 매우 높음 | 매우 높음 | Kubernetes 마이크로서비스 |

---

## 3. 프로토콜 변환

### REST를 GraphQL로 래핑

REST 엔드포인트를 GraphQL 필드로 노출합니다:

```python
@strawberry.type
class Query:
    @strawberry.field
    async def user(self, info, id: strawberry.ID) -> "User":
        """REST GET /api/users/{id}를 GraphQL 필드로 래핑합니다."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{USERS_SERVICE_URL}/api/users/{id}",
                headers={"Authorization": info.context.auth_header},
            )
            resp.raise_for_status()
            data = resp.json()
            return User(id=data["id"], username=data["username"], email=data["email"])
```

### gRPC를 GraphQL로 래핑

```python
@strawberry.type
class Query:
    @strawberry.field
    async def order(self, info, id: strawberry.ID) -> "Order":
        """gRPC GetOrder를 GraphQL 필드로 래핑합니다."""
        channel = grpc.aio.insecure_channel("orders-service:50051")
        stub = OrderServiceStub(channel)
        request = GetOrderRequest(order_id=str(id))
        response = await stub.GetOrder(request)
        return Order(
            id=str(response.id),
            status=response.status,
            total=float(response.total_cents) / 100,
        )
```

### GraphQL을 REST로 변환

레거시 클라이언트를 위해 GraphQL 쿼리를 REST 엔드포인트로 노출합니다:

```python
@app.get("/api/v1/users/{user_id}")
async def get_user_rest(user_id: str, fields: str = "id,username,email"):
    """내부적으로 GraphQL에 위임하는 REST 엔드포인트."""
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
    """GraphQL로 지원되는 사용자 게시글 REST 엔드포인트."""
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

## 4. 게이트웨이 구축

### 완전한 게이트웨이 구현

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
    """공유 리소스를 초기화합니다."""
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


# GraphQL 엔드포인트
graphql_router = GraphQLRouter(schema)
app.include_router(graphql_router, prefix="/graphql")


# REST 프록시 엔드포인트
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
    """REST 요청을 업스트림 서비스로 프록시합니다."""
    service_map = {
        "users": "http://users-service:8001",
        "products": "http://products-service:8002",
        "payments": "http://payments-service:8003",
    }

    upstream = service_map.get(service)
    if not upstream:
        return JSONResponse({"error": "Unknown service"}, status_code=404)

    # 속도 제한 확인
    allowed = await request.app.state.rate_limiter.check(
        client_id=auth.client_id,
        cost=1,
    )
    if not allowed:
        return JSONResponse({"error": "Rate limit exceeded"}, status_code=429)

    # 요청 전달
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


# 헬스 체크
@app.get("/health")
async def health():
    return {"status": "healthy", "services": ["users", "products", "payments", "graphql"]}
```

### 게이트웨이 구성

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

## 5. 게이트웨이에서의 인증

### 중앙 집중식 인증

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
    """요청에서 인증 정보를 추출하고 검증합니다."""
    # JWT 먼저 시도
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

    # API 키 시도
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

    # 익명 접근
    return AuthContext(
        client_id=request.client.host,
        user_id=None,
        roles=[],
        tier="anonymous",
        auth_method="anonymous",
    )
```

### 헤더 전달(Header Forwarding)

```python
def build_upstream_headers(request: Request, auth: AuthContext) -> dict:
    """업스트림 서비스로 전달할 헤더를 구성합니다."""
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

## 6. 속도 제한과 트래픽 관리

### 다중 티어 속도 제한(Multi-Tier Rate Limiting)

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
        """슬라이딩 윈도우 속도 제한 확인."""
        limits = self.TIERS.get(tier, self.TIERS["free"])
        window = 60  # 1분
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

### 서킷 브레이커

```python
from enum import Enum
from datetime import datetime, timedelta


class CircuitState(Enum):
    CLOSED = "closed"        # 정상 작동
    OPEN = "open"            # 실패 중, 요청 거부
    HALF_OPEN = "half_open"  # 복구 테스트 중


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
        return True  # HALF_OPEN: 요청 하나 허용

    def record_success(self):
        self.failures = 0
        self.state = CircuitState.CLOSED

    def record_failure(self):
        self.failures += 1
        self.last_failure = datetime.now()
        if self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN


# 서비스별 서킷 브레이커
circuit_breakers = {
    "users": CircuitBreaker(),
    "products": CircuitBreaker(),
    "orders": CircuitBreaker(),
}
```

---

## 7. REST에서 GraphQL로 마이그레이션

### 마이그레이션 단계

```
1단계: 공존 (1-4주)
  ├── REST와 함께 GraphQL 엔드포인트 배포
  ├── GraphQL 리졸버가 내부적으로 REST 서비스 호출
  ├── 새 기능은 GraphQL로만 구축
  └── REST 엔드포인트 변경 없음

2단계: 클라이언트 마이그레이션 (5-12주)
  ├── 모바일 앱이 GraphQL로 전환 (가장 큰 이점)
  ├── 웹 앱이 페이지별로 마이그레이션
  ├── REST vs. GraphQL 트래픽 비율 모니터링
  └── REST 엔드포인트 여전히 활성

3단계: REST 폐기 (13-16주)
  ├── REST 엔드포인트에 Sunset 헤더 추가
  ├── 남은 REST 소비자 연락
  ├── 마이그레이션 가이드 제공
  └── 폐기 기한 설정

4단계: REST 일몰 (17주+)
  ├── REST 엔드포인트를 문서로 리다이렉트
  ├── 유예 기간 후 410 Gone 반환
  ├── REST 코드 제거
  └── GraphQL이 유일한 API
```

### 마이그레이션 중 REST 래핑

```python
# 1단계: 기존 REST API를 호출하는 GraphQL 리졸버
class UserRESTDataSource:
    """기존 REST API를 호출하는 데이터 소스."""

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


# 2단계: 직접 DB 접근으로 점진적 전환
class UserDirectDataSource:
    """데이터베이스에 직접 쿼리하는 데이터 소스."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user(self, id: str) -> User | None:
        return await self.db.get(UserModel, id)
```

### 트래픽 분할(Traffic Splitting)

```python
# 피처 플래그: REST에서 GraphQL로 트래픽을 점진적으로 전환
GRAPHQL_TRAFFIC_PERCENT = 50  # 10%에서 시작하여 100%까지 증가

@app.get("/api/v1/users/{user_id}")
async def get_user(user_id: str, request: Request):
    """점진적 GraphQL 마이그레이션이 적용된 REST 엔드포인트."""
    import random

    if random.randint(1, 100) <= GRAPHQL_TRAFFIC_PERCENT:
        # GraphQL에 위임
        result = await schema.execute(
            "query($id: ID!) { user(id: $id) { id username email } }",
            variable_values={"id": user_id},
        )
        if not result.errors:
            return result.data["user"]

    # REST 폴백
    return await legacy_get_user(user_id)
```

---

## 8. GraphQL 스키마 스티칭

### 원격 스키마 스티칭

여러 GraphQL 엔드포인트를 하나로 결합합니다:

```python
# 원격 스키마 페치 및 병합
from graphql import build_schema, print_schema
from graphql.utilities import merge_schemas


async def fetch_remote_schema(url: str) -> str:
    """원격 GraphQL 엔드포인트에서 SDL을 가져옵니다."""
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
```

### 스티칭 vs. 페더레이션 비교

| 스키마 스티칭 | 페더레이션 |
|-------------|----------|
| 게이트웨이가 스키마 병합 | 라우터가 서브그래프 스키마 합성 |
| 수동 충돌 해결 | 디렉티브 기반 해결 |
| 모든 GraphQL 서버 | 페더레이션 사양 지원 필요 |
| 더 단순한 설정 | 대규모 조직에 더 적합 |

---

## 9. 관찰 가능성과 모니터링

### 분산 추적

```python
from opentelemetry import trace

tracer = trace.get_tracer("api-gateway")

@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    with tracer.start_as_current_span(
        "gateway.request",
        attributes={
            "http.method": request.method,
            "http.url": str(request.url),
        },
    ) as span:
        response = await call_next(request)
        span.set_attribute("http.status_code", response.status_code)
        return response
```

### 게이트웨이 메트릭 대시보드

| 메트릭 | 설명 | 알림 |
|--------|------|------|
| `gateway.requests.total` | 서비스별 총 요청 | - |
| `gateway.requests.latency` | 요청 지연시간 히스토그램 | p99 > 5초 |
| `gateway.requests.errors` | 서비스별 오류율 | > 5% |
| `gateway.circuit.state` | 서킷 브레이커 상태 | OPEN |
| `gateway.ratelimit.rejections` | 속도 제한된 요청 | > 100/분 |

### 헬스 체크 집계

```python
@app.get("/health")
async def health(request: Request):
    """모든 업스트림 서비스의 헬스를 집계합니다."""
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

## 10. 캡스톤 프로젝트

### 프로젝트: 통합 API 게이트웨이 구축

레슨 01-24에서 배운 모든 것을 결합하여 작동하는 통합 API 게이트웨이를 구축하세요.

### 요구사항

1. **세 가지 백엔드 서비스**:
   - Users 서비스 (REST/FastAPI) — 사용자 CRUD
   - Content 서비스 (GraphQL/Strawberry) — 게시글, 댓글
   - Orders 서비스 (gRPC 또는 REST) — 주문 관리

2. **게이트웨이 기능**:
   - 포트 4000의 단일 진입점
   - 모든 서비스를 래핑하는 `/graphql` GraphQL 엔드포인트
   - `/api/v1/{service}/{path}` REST 프록시
   - JWT 인증
   - 비용 기반 속도 제한
   - 서비스별 서킷 브레이커
   - 구조화 출력 요청 로깅
   - 헬스 체크 엔드포인트

3. **모든 서비스를 통합하는 GraphQL 스키마**:
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

4. **테스트**:
   - GraphQL 리졸버 단위 테스트
   - 서비스 간 쿼리 통합 테스트
   - 속도 제한기 테스트
   - 서킷 브레이커 테스트

5. **Docker Compose**: 로컬에서 모든 것을 실행

### 권장 구현 순서

```
1일차: Docker Compose로 세 개의 백엔드 서비스 설정
2일차: REST 프록시가 있는 기본 게이트웨이 구축
3일차: REST/gRPC를 래핑하는 리졸버가 있는 GraphQL 엔드포인트 추가
4일차: 인증과 속도 제한 구현
5일차: 서킷 브레이커, 헬스 체크, 로깅 추가
6일차: 테스트 작성 (단위 + 통합)
7일차: 문서화 및 리뷰
```

---

## 11. 연습 문제

### 연습 1: 프로토콜 변환

제품 검색 REST 엔드포인트를 래핑하는 GraphQL 리졸버를 구현하세요:
- REST: `GET /api/products?q=query&category=cat&limit=10&page=2`
- GraphQL: `products(query: String, category: String, first: Int, after: String)`
- REST 페이지네이션(page/limit)을 커서 기반 페이지네이션으로 변환

### 연습 2: 게이트웨이 속도 제한기

게이트웨이용 속도 제한기를 구현하세요:
- Redis 슬라이딩 윈도우 사용
- 인증 티어별 다른 제한 (anonymous, free, pro, enterprise)
- `X-RateLimit-*` 헤더 반환
- 비용 곱수 적용 (뮤테이션 5배, 서브스크립션 10배)

### 연습 3: 서킷 브레이커

서킷 브레이커를 구현하세요:
- 5회 연속 실패 후 열림
- 30초 후 반개방
- 열린 상태에서 캐시된 응답 또는 503 반환
- 상태 변경 시 메트릭 방출

### 연습 4: 마이그레이션 계획

30개 엔드포인트를 가진 REST API를 GraphQL로 이동하기 위한 상세 마이그레이션 계획을 작성하세요.

---

## 12. 참고 자료

### API 게이트웨이
- [Kong Gateway](https://docs.konghq.com/)
- [AWS API Gateway](https://aws.amazon.com/api-gateway/)
- [Apollo Router](https://www.apollographql.com/docs/router/)

### 아키텍처
- "Building Microservices" by Sam Newman (O'Reilly)
- [microservices.io — API Gateway Pattern](https://microservices.io/patterns/apigateway.html)

### 마이그레이션
- "From REST to GraphQL" — GitHub Engineering Blog
- "Migrating to GraphQL at Airbnb" — Airbnb Engineering

### 관찰 가능성
- [OpenTelemetry Python SDK](https://opentelemetry.io/docs/languages/python/)
- [Grafana + Prometheus](https://grafana.com/)

---

**License**: CC BY-NC 4.0
