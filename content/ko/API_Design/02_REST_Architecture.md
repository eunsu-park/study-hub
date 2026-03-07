# 레슨 2: REST 아키텍처

**이전**: [API Design 기초](01_API_Design_Fundamentals.md) | [개요](00_Overview.md) | **다음**: [URL 설계와 네이밍](03_URL_Design_and_Naming.md)

## 학습 목표(Learning Objectives)

이 레슨을 마치면 다음을 할 수 있습니다:

1. 여섯 가지 REST 아키텍처 제약 조건과 그 실질적 의미를 설명할 수 있다
2. Richardson 성숙도 모델(레벨 0-3)을 사용하여 API를 분류할 수 있다
3. 도메인 개념을 REST 리소스 및 하위 리소스로 모델링할 수 있다
4. API 응답에 HATEOAS 링크를 구현할 수 있다
5. 수평 확장이 가능한 무상태, 캐시 가능 엔드포인트를 설계할 수 있다
6. "RESTful"과 "REST 기반(REST-inspired)" API 설계를 구분할 수 있다

---

REST(Representational State Transfer)는 프로토콜이나 라이브러리가 아닙니다 -- Roy Fielding이 2000년 박사 논문에서 정의한 **아키텍처 스타일**입니다. "RESTful"이라고 자칭하는 대부분의 API는 실제로 REST 제약 조건의 일부만 구현합니다. 전체 모델을 이해하면 우연이 아닌 의도적인 트레이드오프를 할 수 있습니다.

> **비유:** REST는 잘 운영되는 도서관의 규칙과 같습니다. 책(리소스)에는 카탈로그 항목(표현)이 있습니다. 표준 시스템(HTTP 메서드)을 사용하여 책을 찾고, 빌리고, 반납합니다. 도서관은 지난번에 무엇을 찾고 있었는지 기억하지 않으며(무상태), 방문할 때마다 도서관 카드(토큰)를 가져와야 합니다.

## 목차
1. [REST 제약 조건](#rest-제약-조건)
2. [Richardson 성숙도 모델](#richardson-성숙도-모델)
3. [리소스 모델링](#리소스-모델링)
4. [HATEOAS 실습](#hateoas-실습)
5. [무상태성](#무상태성)
6. [캐시 가능성](#캐시-가능성)
7. [균일한 인터페이스](#균일한-인터페이스)
8. [연습 문제](#연습-문제)

---

## REST 제약 조건

Fielding은 여섯 가지 제약 조건을 정의했습니다. 진정한 RESTful API가 되려면 여섯 가지 모두를 충족해야 합니다.

### 1. 클라이언트-서버

클라이언트와 서버는 독립적입니다. 서버는 UI에 관심이 없고, 클라이언트는 데이터 저장에 관심이 없습니다.

```
Client (React, Mobile, CLI)      Server (FastAPI, Flask)
┌─────────────────────┐          ┌─────────────────────┐
│ UI / Presentation   │◄────────►│ Business Logic      │
│ User Interaction    │  HTTP    │ Data Storage        │
└─────────────────────┘          └─────────────────────┘
```

**이점:** 클라이언트와 서버가 독립적으로 발전할 수 있습니다. 백엔드를 건드리지 않고 프론트엔드를 재구축할 수 있으며, 그 반대도 마찬가지입니다.

### 2. 무상태성

모든 요청은 서버가 처리하는 데 필요한 **모든** 정보를 포함해야 합니다. 서버는 요청 간에 세션 상태를 저장하지 않습니다.

```python
from fastapi import FastAPI, Header, HTTPException

app = FastAPI()

# STATELESS: Token carries all auth context
@app.get("/api/me")
async def get_current_user(authorization: str = Header()):
    """Each request is self-contained. No server-side session."""
    token = authorization.removeprefix("Bearer ")
    user = decode_and_verify_token(token)  # all state is in the token
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"id": user["sub"], "name": user["name"]}

# STATEFUL (anti-pattern): Server remembers session
# Session ID maps to server-side state -- breaks horizontal scaling
# @app.get("/api/me")
# async def get_current_user(session_id: str = Cookie()):
#     user = session_store[session_id]  # server must share session state
#     return user
```

**이점:** 어떤 서버 인스턴스든 어떤 요청이든 처리할 수 있습니다. 수평 확장이 간단합니다.

### 3. 캐시 가능성

응답은 자신이 캐시 가능한지 불가능한지를 선언해야 합니다. 이를 통해 클라이언트와 중개자(CDN, 프록시)가 응답을 재사용할 수 있습니다.

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/api/products/{product_id}")
async def get_product(product_id: int):
    product = {"id": product_id, "name": "Widget", "price": 9.99}
    return JSONResponse(
        content=product,
        headers={
            "Cache-Control": "public, max-age=3600",  # cache for 1 hour
            "ETag": f'"{hash(str(product))}"',         # fingerprint
        }
    )

@app.get("/api/users/me")
async def get_current_user():
    """User-specific data should not be cached publicly."""
    return JSONResponse(
        content={"id": 1, "name": "Alice"},
        headers={
            "Cache-Control": "private, no-store",  # never cache
        }
    )
```

### 4. 계층형 시스템

클라이언트는 서버에 직접 연결되어 있는지 중개자(로드 밸런서, CDN, API 게이트웨이)에 연결되어 있는지 알 수 없습니다. 각 계층은 자신이 상호작용하는 계층만 알고 있습니다.

```
Client → CDN → API Gateway → Load Balancer → App Server → Database
         │        │                │
         │        │                └── Health checks, routing
         │        └── Auth, rate limiting, logging
         └── Static content, cached responses
```

### 5. 균일한 인터페이스

가장 근본적인 제약 조건입니다. 모든 리소스는 네 가지 하위 제약 조건을 가진 **균일하고** 표준화된 인터페이스를 통해 접근됩니다:

1. **리소스 식별** -- 각 리소스는 고유한 URI를 가짐
2. **표현을 통한 조작** -- 클라이언트는 리소스 자체가 아닌 JSON/XML 표현으로 작업
3. **자체 설명적 메시지** -- 각 메시지는 이해에 충분한 메타데이터(Content-Type, 상태 코드)를 포함
4. **HATEOAS** -- 애플리케이션 상태 엔진으로서의 하이퍼미디어 (응답에 관련 동작 링크 포함)

### 6. 코드 온 디맨드 (선택 사항)

서버가 실행 가능한 코드(예: JavaScript)를 클라이언트에 전송할 수 있습니다. 이것은 유일한 선택적 제약 조건이며 JSON API에서는 거의 사용되지 않습니다.

---

## Richardson 성숙도 모델

Leonard Richardson은 API가 얼마나 "RESTful"한지를 분류하는 네 가지 레벨 모델을 제안했습니다.

### 레벨 0: POX(Plain Old XML/JSON)의 늪

단일 엔드포인트가 모든 것을 처리합니다. URL과 HTTP 메서드는 무관하며, 동작이 요청 본문에 포함됩니다.

```python
# Level 0: Single endpoint, action in body
@app.post("/api")
async def handle_request(action: str, data: dict):
    if action == "getUser":
        return {"id": 1, "name": "Alice"}
    elif action == "createUser":
        return {"id": 2, "name": data["name"]}
    elif action == "deleteUser":
        return {"deleted": True}
    # Everything goes through POST /api
```

**문제점:** 캐싱 불가(모든 것이 POST), 표준 탐색 없음, HTTP 시맨틱 없음.

### 레벨 1: 리소스

리소스별로 개별 URI를 사용하지만, 여전히 단일 HTTP 메서드(보통 POST)를 사용합니다.

```python
# Level 1: Different URLs, but still only POST
@app.post("/api/users")
async def users_endpoint(action: str, data: dict = None):
    if action == "list":
        return [{"id": 1, "name": "Alice"}]
    elif action == "create":
        return {"id": 2, "name": data["name"]}

@app.post("/api/orders")
async def orders_endpoint(action: str, data: dict = None):
    if action == "list":
        return [{"id": 10, "total": 99.99}]
```

**개선점:** 리소스에 정체성이 부여됩니다. 그러나 HTTP 메서드가 의미론적으로 사용되지 않습니다.

### 레벨 2: HTTP 동사

리소스가 URI로 식별되**고** 올바른 HTTP 메서드로 조작됩니다.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class UserCreate(BaseModel):
    name: str
    email: str

# Level 2: Proper HTTP methods + status codes
@app.get("/api/users")
async def list_users():
    """GET for retrieval -- cacheable, safe, idempotent."""
    return {"data": [{"id": 1, "name": "Alice"}]}

@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    return {"id": user_id, "name": "Alice"}

@app.post("/api/users", status_code=201)
async def create_user(user: UserCreate):
    """POST for creation -- returns 201 Created."""
    return {"id": 2, "name": user.name, "email": user.email}

@app.put("/api/users/{user_id}")
async def replace_user(user_id: int, user: UserCreate):
    """PUT for full replacement -- idempotent."""
    return {"id": user_id, "name": user.name, "email": user.email}

@app.patch("/api/users/{user_id}")
async def update_user(user_id: int, name: str = None):
    """PATCH for partial update."""
    return {"id": user_id, "name": name or "Alice"}

@app.delete("/api/users/{user_id}", status_code=204)
async def delete_user(user_id: int):
    """DELETE for removal -- returns 204 No Content."""
    return None
```

**대부분의 프로덕션 API가 이 수준에 해당합니다.** 레벨 2는 캐싱, 표준 도구, 명확한 시맨틱을 제공합니다.

### 레벨 3: 하이퍼미디어 컨트롤 (HATEOAS)

응답에 클라이언트가 다음으로 할 수 있는 것을 알려주는 **링크**가 포함됩니다. 클라이언트는 URL을 하드코딩하지 않고 링크를 따릅니다.

```python
@app.get("/api/users/{user_id}")
async def get_user_with_links(user_id: int):
    """Level 3: Response includes navigation links."""
    return {
        "id": user_id,
        "name": "Alice",
        "email": "alice@example.com",
        "status": "active",
        "_links": {
            "self": {"href": f"/api/users/{user_id}", "method": "GET"},
            "update": {"href": f"/api/users/{user_id}", "method": "PATCH"},
            "delete": {"href": f"/api/users/{user_id}", "method": "DELETE"},
            "orders": {"href": f"/api/users/{user_id}/orders", "method": "GET"},
            "deactivate": {"href": f"/api/users/{user_id}/deactivate", "method": "POST"},
        }
    }
```

**이점:** 클라이언트가 URL 구조에서 분리됩니다. 서버가 클라이언트를 깨뜨리지 않고 URL을 변경할 수 있습니다. 이것이 가장 성숙한 레벨이지만 실제로 완전히 구현되는 경우는 드뭅니다.

### 성숙도 레벨 요약

| 레벨 | 이름 | 특징 | 캐싱 | 탐색 가능성 |
|------|------|------|------|------------|
| 0 | POX의 늪 | 단일 엔드포인트, 본문에 동작 | 없음 | 없음 |
| 1 | 리소스 | 리소스별 고유 URI | 제한적 | URL 기반 |
| 2 | HTTP 동사 | 올바른 메서드 + 상태 코드 | 가능 | 규칙 기반 |
| 3 | 하이퍼미디어 | 응답에 링크 (HATEOAS) | 가능 | 자체 설명적 |

---

## 리소스 모델링

리소스 모델링은 도메인 개념을 API 리소스에 매핑하는 과정입니다.

### 리소스 식별

```python
# Domain: E-commerce platform
# Nouns become resources:

# Primary resources (independent)
# /api/users
# /api/products
# /api/orders
# /api/categories

# Sub-resources (belong to a parent)
# /api/users/{id}/addresses
# /api/orders/{id}/items
# /api/products/{id}/reviews

# Avoid turning actions into resources:
# BAD:  /api/create-order     (verb as resource)
# GOOD: POST /api/orders      (HTTP method carries the verb)
```

### 리소스 관계

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# --- One-to-Many: User -> Orders ---
@app.get("/api/users/{user_id}/orders")
async def get_user_orders(user_id: int, page: int = 1):
    """Sub-resource: orders belonging to a user."""
    return {
        "data": [
            {"id": 101, "user_id": user_id, "total": 49.99},
            {"id": 102, "user_id": user_id, "total": 129.99},
        ],
        "meta": {"page": page, "per_page": 20, "total": 2}
    }

# --- Many-to-Many: Order -> Products (via line items) ---
@app.get("/api/orders/{order_id}/items")
async def get_order_items(order_id: int):
    """Sub-resource: line items in an order."""
    return {
        "data": [
            {"product_id": 1, "name": "Widget", "quantity": 2, "unit_price": 9.99},
            {"product_id": 5, "name": "Gadget", "quantity": 1, "unit_price": 29.99},
        ]
    }

# --- Singleton sub-resource ---
@app.get("/api/users/{user_id}/profile")
async def get_user_profile(user_id: int):
    """Singleton: each user has exactly one profile."""
    return {
        "user_id": user_id,
        "bio": "Software developer",
        "avatar_url": "/avatars/alice.jpg"
    }
```

### 리소스 세분화

적절한 수준의 세분화로 리소스를 설계합니다:

```python
# TOO FINE: Every field is a resource (chatty API)
# GET /api/users/1/name          -> "Alice"
# GET /api/users/1/email         -> "alice@example.com"
# GET /api/users/1/created_at    -> "2025-01-01"
# Problem: 3 requests for basic user info

# TOO COARSE: One giant resource (over-fetching)
# GET /api/users/1
# Returns: user + all orders + all reviews + all addresses + ...
# Problem: Huge response, slow, wastes bandwidth

# JUST RIGHT: Logical groupings
# GET /api/users/1               -> core user fields
# GET /api/users/1/orders        -> separate collection when needed
# GET /api/users/1?expand=orders -> optional expansion for convenience
```

---

## HATEOAS 실습

HATEOAS(Hypermedia As The Engine Of Application State)는 API 응답에 관련 리소스 및 사용 가능한 동작에 대한 링크를 포함하는 것을 의미합니다.

### 기본 구현

```python
from fastapi import FastAPI

app = FastAPI()

def add_links(resource: dict, resource_type: str, resource_id: int) -> dict:
    """Helper to add standard HATEOAS links to a resource."""
    base = f"/api/{resource_type}/{resource_id}"
    resource["_links"] = {
        "self": {"href": base, "method": "GET"},
        "update": {"href": base, "method": "PATCH"},
        "delete": {"href": base, "method": "DELETE"},
    }
    return resource

@app.get("/api/orders/{order_id}")
async def get_order(order_id: int):
    order = {
        "id": order_id,
        "status": "shipped",
        "total": 59.99,
        "user_id": 1,
    }

    # Add standard links
    add_links(order, "orders", order_id)

    # Add context-specific links based on state
    if order["status"] == "shipped":
        order["_links"]["track"] = {
            "href": f"/api/orders/{order_id}/tracking",
            "method": "GET"
        }
    elif order["status"] == "pending":
        order["_links"]["cancel"] = {
            "href": f"/api/orders/{order_id}/cancel",
            "method": "POST"
        }
        order["_links"]["pay"] = {
            "href": f"/api/orders/{order_id}/payments",
            "method": "POST"
        }

    # Add relationship links
    order["_links"]["items"] = {
        "href": f"/api/orders/{order_id}/items",
        "method": "GET"
    }
    order["_links"]["user"] = {
        "href": f"/api/users/{order['user_id']}",
        "method": "GET"
    }

    return order
```

### 페이지네이션이 포함된 컬렉션 링크

```python
@app.get("/api/products")
async def list_products(page: int = 1, per_page: int = 20):
    total = 95
    total_pages = (total + per_page - 1) // per_page

    products = [
        {"id": i, "name": f"Product {i}", "price": 9.99 + i}
        for i in range((page - 1) * per_page + 1, min(page * per_page, total) + 1)
    ]

    response = {
        "data": products,
        "meta": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
        },
        "_links": {
            "self": {"href": f"/api/products?page={page}&per_page={per_page}"},
            "first": {"href": f"/api/products?page=1&per_page={per_page}"},
            "last": {"href": f"/api/products?page={total_pages}&per_page={per_page}"},
        }
    }

    if page > 1:
        response["_links"]["prev"] = {
            "href": f"/api/products?page={page - 1}&per_page={per_page}"
        }
    if page < total_pages:
        response["_links"]["next"] = {
            "href": f"/api/products?page={page + 1}&per_page={per_page}"
        }

    return response
```

---

## 무상태성

### 무상태성이 중요한 이유

```python
# STATELESS server: any instance can handle any request
#
# Client ──► Load Balancer ──► Server A  (handles request 1)
#                          ──► Server B  (handles request 2)
#                          ──► Server C  (handles request 3)
#
# Each request carries its own authentication token.
# No sticky sessions. Easy to scale horizontally.

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

app = FastAPI()
security = HTTPBearer()
SECRET_KEY = "your-secret-key"

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Extract user from JWT -- no server-side session needed."""
    try:
        payload = jwt.decode(
            credentials.credentials, SECRET_KEY, algorithms=["HS256"]
        )
        return {"id": payload["sub"], "role": payload["role"]}
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/api/dashboard")
async def dashboard(user: dict = Depends(get_current_user)):
    """Any server can handle this -- the token is self-contained."""
    return {"message": f"Welcome, user {user['id']}!", "role": user["role"]}
```

### 상태가 필요한 경우

일부 연산은 본질적으로 상태가 필요합니다(예: 다단계 위저드, 파일 업로드). 다음과 같은 방법으로 처리합니다:

1. 서버 메모리가 아닌 **데이터베이스에 상태를 저장**
2. 재시도가 안전하도록 **멱등성 키 사용**
3. **각 단계를 별도의 리소스로 분리** (예: `/api/uploads/{upload_id}/parts/{part_num}`)

---

## 캐시 가능성

### Cache-Control 전략

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
import hashlib
import json

app = FastAPI()

@app.get("/api/catalog")
async def get_catalog(request: Request):
    """Public, cacheable catalog data."""
    catalog = {"categories": ["Electronics", "Books", "Clothing"]}

    # Generate ETag from content
    content_bytes = json.dumps(catalog, sort_keys=True).encode()
    etag = f'"{hashlib.md5(content_bytes).hexdigest()}"'

    # Check If-None-Match for conditional request
    if_none_match = request.headers.get("if-none-match")
    if if_none_match == etag:
        return JSONResponse(status_code=304, content=None)

    return JSONResponse(
        content=catalog,
        headers={
            "Cache-Control": "public, max-age=3600, stale-while-revalidate=60",
            "ETag": etag,
            "Last-Modified": "Sat, 01 Jan 2025 00:00:00 GMT",
        }
    )

@app.get("/api/users/{user_id}/balance")
async def get_user_balance(user_id: int):
    """Private, user-specific data -- do not cache publicly."""
    return JSONResponse(
        content={"user_id": user_id, "balance": 150.00},
        headers={
            "Cache-Control": "private, no-store",
        }
    )

@app.get("/api/exchange-rates")
async def get_exchange_rates():
    """Semi-volatile data -- cache briefly."""
    return JSONResponse(
        content={"USD_EUR": 0.92, "USD_GBP": 0.79},
        headers={
            "Cache-Control": "public, max-age=60",  # refresh every minute
            "Vary": "Accept",  # cache varies by Accept header
        }
    )
```

### Cache-Control 지시자 요약

| 지시자 | 의미 |
|--------|------|
| `public` | 모든 캐시(CDN, 브라우저)가 응답을 저장할 수 있음 |
| `private` | 최종 사용자의 브라우저만 캐시할 수 있음 |
| `no-store` | 전혀 캐시하지 않음 |
| `no-cache` | 캐시하되 사용 전에 서버와 재검증 |
| `max-age=N` | 캐시가 N초 동안 유효 |
| `stale-while-revalidate=N` | 백그라운드에서 재검증하는 동안 오래된 콘텐츠 제공 |
| `must-revalidate` | 만료되면 제공 전에 반드시 재검증 |

---

## 균일한 인터페이스

### 종합 예제

```python
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()

class ArticleCreate(BaseModel):
    title: str
    body: str
    tags: list[str] = []

class Article(BaseModel):
    id: int
    title: str
    body: str
    tags: list[str]
    created_at: str
    updated_at: str

# Uniform interface demonstration:
# 1. Resource identification via URI
# 2. Manipulation through representations (JSON)
# 3. Self-descriptive messages (Content-Type, status codes)
# 4. HATEOAS (links in responses)

@app.get("/api/articles")
async def list_articles(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    tag: str | None = None,
):
    """Uniform list endpoint with filtering and pagination."""
    articles = [
        {"id": 1, "title": "REST Deep Dive", "tags": ["api", "rest"]},
        {"id": 2, "title": "GraphQL Intro", "tags": ["api", "graphql"]},
    ]

    if tag:
        articles = [a for a in articles if tag in a["tags"]]

    return {
        "data": articles,
        "meta": {"page": page, "per_page": per_page, "total": len(articles)},
        "_links": {
            "self": {"href": f"/api/articles?page={page}&per_page={per_page}"},
            "create": {"href": "/api/articles", "method": "POST"},
        }
    }

@app.post("/api/articles", status_code=201)
async def create_article(article: ArticleCreate):
    """Create returns 201 + Location header."""
    now = datetime.now().isoformat()
    created = {
        "id": 3,
        "title": article.title,
        "body": article.body,
        "tags": article.tags,
        "created_at": now,
        "updated_at": now,
        "_links": {
            "self": {"href": "/api/articles/3"},
            "collection": {"href": "/api/articles"},
        }
    }
    from fastapi.responses import JSONResponse
    return JSONResponse(
        content=created,
        status_code=201,
        headers={"Location": "/api/articles/3"}
    )

@app.get("/api/articles/{article_id}")
async def get_article(article_id: int):
    """Self-descriptive response with HATEOAS links."""
    return {
        "id": article_id,
        "title": "REST Deep Dive",
        "body": "Full article content...",
        "tags": ["api", "rest"],
        "created_at": "2025-01-15T10:00:00",
        "updated_at": "2025-01-15T10:00:00",
        "_links": {
            "self": {"href": f"/api/articles/{article_id}"},
            "update": {"href": f"/api/articles/{article_id}", "method": "PATCH"},
            "delete": {"href": f"/api/articles/{article_id}", "method": "DELETE"},
            "collection": {"href": "/api/articles"},
        }
    }
```

---

## 연습 문제

### 연습 1: API 성숙도 분류

다음 API 인터랙션이 주어졌을 때, 각각의 Richardson 성숙도 레벨을 식별하십시오:

1. `POST /service` (본문: `{"method": "getUserById", "params": {"id": 1}}`)
2. `POST /users/1` (본문: `{"action": "delete"}`)
3. `DELETE /api/users/1` (반환: `204 No Content`)
4. `GET /api/users/1` (반환: `{"id": 1, "name": "Alice", "_links": {"orders": {"href": "/api/users/1/orders"}}}`)

### 연습 2: 도메인을 리소스로 모델링

**영화 스트리밍 플랫폼**의 리소스 모델을 설계하십시오:
- 영화, 배우, 감독, 장르
- 사용자 시청 목록 및 시청 기록
- 리뷰 및 평점

최소 5개의 엔드포인트에 대한 URI, HTTP 메서드, 예시 응답을 정의합니다. 하위 리소스 관계와 HATEOAS 링크를 포함합니다.

### 연습 3: 조건부 캐싱 구현

다음을 지원하는 제품 카탈로그용 FastAPI 엔드포인트를 구축하십시오:
- 조건부 GET을 위한 `ETag`와 `If-None-Match`
- 적절한 `max-age`를 가진 `Cache-Control` 헤더
- `Last-Modified`와 `If-Modified-Since` 헤더
- 콘텐츠가 변경되지 않았을 때 `304 Not Modified` 반환

### 연습 4: 무상태 인증

세션 기반 Flask 애플리케이션을 무상태 JWT 인증으로 리팩터링하십시오:
- `POST /api/auth/login`에서 JWT 발급
- 모든 보호된 엔드포인트에서 JWT 검증
- 토큰 페이로드에 사용자 역할과 권한 포함
- 명확한 오류 응답으로 토큰 만료를 우아하게 처리

### 연습 5: 레벨 3 API 구축

완전한 HATEOAS를 갖춘 간단한 태스크 관리자 FastAPI 애플리케이션을 작성하십시오:
- `GET /api/tasks` -- 페이지네이션 링크가 포함된 태스크 목록
- `POST /api/tasks` -- 태스크 생성, 새 리소스에 대한 링크 반환
- `GET /api/tasks/{id}` -- 상태에 따른 링크가 포함된 태스크 조회 (예: 상태가 "pending"일 때만 "complete" 링크)
- `PATCH /api/tasks/{id}` -- 태스크 업데이트
- `POST /api/tasks/{id}/complete` -- 완료 처리, "complete" 링크 제거, "reopen" 링크 추가

---

## 요약

이 레슨에서 다룬 내용:
1. 여섯 가지 REST 아키텍처 제약 조건 (클라이언트-서버, 무상태, 캐시 가능, 계층형, 균일한 인터페이스, 코드 온 디맨드)
2. Richardson 성숙도 모델: 레벨 0(POX의 늪)에서 레벨 3(하이퍼미디어)까지
3. 리소스 모델링: 리소스, 관계, 세분화 식별
4. HATEOAS: 응답에 탐색 링크와 상태 의존 동작 포함
5. 무상태성: 수평 확장을 위한 JWT와 자체 완결 요청 사용
6. 캐시 가능성: Cache-Control 지시자, ETag, 조건부 요청

---

**이전**: [API Design 기초](01_API_Design_Fundamentals.md) | [개요](00_Overview.md) | **다음**: [URL 설계와 네이밍](03_URL_Design_and_Naming.md)

**License**: CC BY-NC 4.0
