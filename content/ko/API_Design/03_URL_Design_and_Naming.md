# 레슨 3: URL 설계와 네이밍

**이전**: [REST 아키텍처](02_REST_Architecture.md) | [개요](00_Overview.md) | **다음**: [요청과 응답 설계](04_Request_and_Response_Design.md)

## 학습 목표(Learning Objectives)

이 레슨을 마치면 다음을 할 수 있습니다:

1. 직관적이고 일관성 있는 리소스 네이밍 규칙을 적용할 수 있다
2. 리소스 관계를 반영하는 계층적 URL을 설계할 수 있다
3. 올바른 복수형 사용 및 케이싱 전략을 선택할 수 있다
4. 필터링, 정렬, 필드 선택을 위한 쿼리 파라미터를 구현할 수 있다
5. 일반적인 URL 안티패턴을 인식하고 회피할 수 있다
6. API 버전 간에 안정적으로 유지되는 URL을 구성할 수 있다

---

URL은 API에서 가장 눈에 띄는 부분입니다. 개발자가 문서를 읽거나 응답 본문을 확인하기 전에 먼저 URL을 봅니다. 잘 설계된 URL 구조는 의도를 즉시 전달하고, 잘못 설계된 구조는 어떤 문서화로도 완전히 보완할 수 없는 혼란을 초래합니다. 이 레슨에서는 API를 예측 가능하고 사용하기 좋게 만드는 규칙을 확립합니다.

> **비유:** URL은 도로 주소와 같습니다. 좋은 주소 체계(서울시 강남구 테헤란로 123, 4층)는 지도 없이도 위치를 찾을 수 있게 합니다. 나쁜 주소("오래된 참나무 지나서 파란 집")는 현지 지식이 필요합니다. 일관되고 계층적인 URL은 API의 도로 체계입니다.

## 목차
1. [리소스 네이밍 규칙](#리소스-네이밍-규칙)
2. [계층적 URL](#계층적-url)
3. [복수형 규칙](#복수형-규칙)
4. [쿼리 파라미터](#쿼리-파라미터)
5. [필터링 패턴](#필터링-패턴)
6. [URL 안티패턴](#url-안티패턴)
7. [실용 가이드라인](#실용-가이드라인)
8. [연습 문제](#연습-문제)

---

## 리소스 네이밍 규칙

### 동사가 아닌 명사를 사용

리소스는 **동작**이 아닌 **사물**을 나타냅니다. HTTP 메서드가 동사 역할을 합니다.

```python
# GOOD: Nouns as resources, HTTP methods as verbs
# GET    /api/users          -- list users
# POST   /api/users          -- create a user
# GET    /api/users/42       -- get user 42
# PATCH  /api/users/42       -- update user 42
# DELETE /api/users/42       -- delete user 42

# BAD: Verbs in URLs (RPC-style, not RESTful)
# GET    /api/getUsers
# POST   /api/createUser
# POST   /api/deleteUser/42
# GET    /api/fetchUserById/42
```

### 소문자와 하이픈 사용

URL은 소문자여야 합니다. 단어를 구분할 때 밑줄이나 camelCase가 아닌 하이픈(`-`)을 사용합니다.

```python
# GOOD: Lowercase with hyphens
# /api/order-items
# /api/user-profiles
# /api/shipping-addresses

# BAD: Mixed casing or underscores in URLs
# /api/orderItems          -- camelCase
# /api/OrderItems          -- PascalCase
# /api/order_items         -- underscores (acceptable but less standard)
# /api/Order-Items         -- mixed case
```

> **참고:** 하이픈이 URL에서 가장 일반적인 규칙이지만, 일부 주요 API(예: Google)는 밑줄을 사용합니다. 핵심 규칙은 **하나를 선택하고 일관되게 사용하는 것**입니다.

### 구체적인 이름 사용

일반적인 이름보다 구체적이고 도메인 관련성이 높은 이름을 선호합니다.

```python
from fastapi import FastAPI

app = FastAPI()

# GOOD: Specific, meaningful resource names
@app.get("/api/invoices")
async def list_invoices():
    pass

@app.get("/api/shipments")
async def list_shipments():
    pass

@app.get("/api/subscriptions")
async def list_subscriptions():
    pass

# BAD: Generic, meaningless names
# /api/items       -- items of what?
# /api/objects      -- too abstract
# /api/entities     -- conveys no domain meaning
# /api/records      -- what kind of records?
```

---

## 계층적 URL

### 부모-자식 관계

중첩을 사용하여 **소유** 또는 **포함** 관계를 표현합니다.

```python
from fastapi import FastAPI

app = FastAPI()

# A user HAS many orders -- orders belong to a user
@app.get("/api/users/{user_id}/orders")
async def get_user_orders(user_id: int):
    """List orders belonging to a specific user."""
    return {
        "data": [
            {"id": 101, "total": 49.99, "status": "shipped"},
            {"id": 102, "total": 129.99, "status": "pending"},
        ]
    }

# An order HAS many items -- items belong to an order
@app.get("/api/orders/{order_id}/items")
async def get_order_items(order_id: int):
    """List line items within an order."""
    return {
        "data": [
            {"product_id": 1, "name": "Widget", "quantity": 2},
            {"product_id": 5, "name": "Gadget", "quantity": 1},
        ]
    }

# A project HAS many members
@app.get("/api/projects/{project_id}/members")
async def get_project_members(project_id: int):
    """List members of a project."""
    return {
        "data": [
            {"user_id": 1, "role": "owner"},
            {"user_id": 2, "role": "contributor"},
        ]
    }
```

### 중첩 깊이 제한

중첩은 **최대 두 단계**로 유지합니다. 더 깊은 중첩은 URL을 길고 취약하게 만듭니다.

```python
# GOOD: Maximum two levels of nesting
# /api/users/{user_id}/orders
# /api/orders/{order_id}/items
# /api/projects/{project_id}/tasks

# BAD: Three or more levels -- too deep
# /api/users/{user_id}/orders/{order_id}/items/{item_id}/reviews
# Problem: Long URL, tightly coupled, hard to cache

# SOLUTION: Flatten deep relationships
# Instead of /api/users/1/orders/101/items/5/reviews
# Use:
# GET /api/reviews?order_item_id=5
# or
# GET /api/order-items/5/reviews
```

### 중첩 vs 평탄화 기준

```python
# NEST when:
# - The child resource does not make sense without the parent
# - You always need the parent context to access the child
# - The relationship is strictly one-to-many ownership

# FLATTEN when:
# - The child resource can be accessed independently
# - You need to query across parents (e.g., "all reviews by any user")
# - The nesting would exceed two levels

# Example: Both valid approaches for different use cases
@app.get("/api/users/{user_id}/reviews")
async def get_reviews_by_user(user_id: int):
    """Nested: reviews in context of a specific user."""
    return {"data": [{"id": 1, "rating": 5}]}

@app.get("/api/reviews")
async def list_all_reviews(user_id: int | None = None, product_id: int | None = None):
    """Flat: query reviews across all users/products."""
    return {"data": [{"id": 1, "user_id": 1, "product_id": 3, "rating": 5}]}
```

---

## 복수형 규칙

### 컬렉션에는 항상 복수형 사용

```python
# GOOD: Plural nouns for collections
# /api/users              -- collection of users
# /api/users/42           -- single user from the collection
# /api/orders             -- collection of orders
# /api/orders/101         -- single order

# BAD: Mixing singular and plural
# /api/user               -- singular for collection?
# /api/user/42            -- inconsistent
# /api/order              -- singular feels like a singleton
```

### 싱글톤 리소스

일부 리소스는 본질적으로 단수입니다(부모 컨텍스트당 정확히 하나만 존재). 이 경우 단수형을 사용합니다.

```python
# Singleton: Each user has exactly one profile
@app.get("/api/users/{user_id}/profile")
async def get_user_profile(user_id: int):
    """Singleton sub-resource -- no ID needed."""
    return {"bio": "Developer", "avatar_url": "/avatars/42.jpg"}

# Singleton: Current user's settings
@app.get("/api/settings")
async def get_settings():
    """Global settings -- only one instance exists."""
    return {"theme": "dark", "language": "en", "timezone": "UTC"}

# Singleton: Application health check
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "uptime": "72h"}
```

### 불규칙 복수형 처리

```python
# Standard plurals -- straightforward
# /api/users, /api/orders, /api/products, /api/categories

# Irregular plurals -- use the correct English plural
# /api/people (not /api/persons)
# /api/analyses (not /api/analysiss)
# /api/criteria (not /api/criterias)

# Uncountable nouns -- use plural form anyway for consistency
# /api/feedback (acceptable as-is, since "feedbacks" is awkward)
# /api/media (acceptable as-is)
# /api/metadata (acceptable as-is)

# When in doubt, pick the conventional plural and document it.
```

---

## 쿼리 파라미터

### 표준 쿼리 파라미터 패턴

```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/api/products")
async def list_products(
    # Pagination
    page: int = Query(default=1, ge=1, description="Page number"),
    per_page: int = Query(default=20, ge=1, le=100, description="Items per page"),

    # Filtering
    category: str | None = Query(default=None, description="Filter by category"),
    min_price: float | None = Query(default=None, ge=0, description="Minimum price"),
    max_price: float | None = Query(default=None, ge=0, description="Maximum price"),
    in_stock: bool | None = Query(default=None, description="Filter by stock status"),

    # Sorting
    sort: str = Query(default="created_at", description="Sort field"),
    order: str = Query(default="desc", regex="^(asc|desc)$", description="Sort order"),

    # Search
    q: str | None = Query(default=None, min_length=2, description="Search query"),

    # Field selection
    fields: str | None = Query(default=None, description="Comma-separated fields to include"),
):
    """
    List products with filtering, sorting, pagination, and field selection.

    Example: GET /api/products?category=electronics&min_price=10&sort=price&order=asc&page=2
    """
    return {
        "data": [{"id": 1, "name": "Widget", "price": 9.99, "category": "electronics"}],
        "meta": {
            "page": page,
            "per_page": per_page,
            "total": 150,
            "filters_applied": {
                "category": category,
                "min_price": min_price,
                "max_price": max_price,
            }
        }
    }
```

### 경로 파라미터 vs 쿼리 파라미터

```python
# PATH PARAMETERS: Identify a specific resource
# /api/users/42              -- "user 42" is the identity
# /api/orders/101/items/5    -- "item 5 in order 101"

# QUERY PARAMETERS: Modify the response (filter, sort, paginate, search)
# /api/users?role=admin            -- filter users
# /api/users?sort=name&order=asc   -- sort users
# /api/users?page=3&per_page=10    -- paginate users
# /api/users?fields=id,name,email  -- select fields

# RULE OF THUMB:
# - Use path params for resource identification (required)
# - Use query params for everything else (optional)

# BAD: Query params for resource identity
# /api/users?id=42                 -- should be /api/users/42

# BAD: Path params for optional filters
# /api/users/admin                 -- is "admin" a user ID or a role filter?
# Use: /api/users?role=admin
```

---

## 필터링 패턴

### 단순 동등 필터

```python
# Direct field equality
# GET /api/products?color=red
# GET /api/products?status=active
# GET /api/users?role=admin&department=engineering
```

### 범위 필터

```python
# Prefix convention for ranges
# GET /api/products?min_price=10&max_price=50
# GET /api/events?start_after=2025-01-01&start_before=2025-12-31
# GET /api/logs?since=2025-01-01T00:00:00Z

# Alternative: operator suffix
# GET /api/products?price_gte=10&price_lte=50
# GET /api/orders?created_at_gt=2025-01-01
```

### 다중 값 필터

```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/api/products")
async def list_products(
    # Comma-separated values (most common)
    # GET /api/products?category=electronics,books,toys
    category: str | None = Query(default=None),

    # Repeated parameters (also valid)
    # GET /api/products?tag=sale&tag=new
    tag: list[str] = Query(default=[]),

    # Comma-separated IDs
    # GET /api/products?ids=1,2,3,4,5
    ids: str | None = Query(default=None),
):
    categories = category.split(",") if category else []
    id_list = [int(x) for x in ids.split(",")] if ids else []

    return {"filters": {"categories": categories, "tags": tag, "ids": id_list}}
```

### 복잡한 필터 구문

고급 필터링이 필요한 API의 경우 구조화된 접근 방식을 고려합니다:

```python
# LHS bracket syntax (used by Stripe, many others)
# GET /api/products?price[gte]=10&price[lte]=50
# GET /api/orders?created[after]=2025-01-01&status[in]=pending,processing

# Filter parameter with JSON (more flexible but less readable)
# GET /api/products?filter={"price":{"$gte":10,"$lte":50},"status":"active"}

# RSQL / FIQL syntax (standardized, powerful)
# GET /api/products?filter=price>=10;price<=50;status==active

# For most APIs, simple equality + range prefixes are sufficient.
# Only adopt complex syntax when your consumers genuinely need it.
```

### FastAPI에서 브래킷 필터 구현

```python
from fastapi import FastAPI, Request

app = FastAPI()

def parse_bracket_filters(query_params: dict) -> dict:
    """Parse LHS bracket filter syntax: price[gte]=10 -> {"price": {"gte": 10}}."""
    import re
    filters = {}
    pattern = re.compile(r"^(\w+)\[(\w+)\]$")
    for key, value in query_params.items():
        match = pattern.match(key)
        if match:
            field, operator = match.groups()
            if field not in filters:
                filters[field] = {}
            filters[field][operator] = value
        else:
            filters[key] = value
    return filters

@app.get("/api/products")
async def list_products(request: Request):
    """
    Supports bracket filter syntax:
    GET /api/products?price[gte]=10&price[lte]=50&category=electronics
    """
    filters = parse_bracket_filters(dict(request.query_params))
    # filters = {"price": {"gte": "10", "lte": "50"}, "category": "electronics"}
    return {"applied_filters": filters, "data": []}
```

---

## URL 안티패턴

### 1. URL에 동사 사용

```python
# BAD: Verbs as URL segments
# POST /api/users/create
# GET  /api/users/list
# PUT  /api/users/42/update
# POST /api/users/42/delete

# GOOD: HTTP methods carry the verb
# POST   /api/users              -- create
# GET    /api/users              -- list
# PATCH  /api/users/42           -- update
# DELETE /api/users/42           -- delete

# EXCEPTION: Action endpoints for non-CRUD operations
# POST /api/orders/42/cancel     -- acceptable for domain actions
# POST /api/users/42/deactivate  -- acceptable for state transitions
```

### 2. URL에 파일 확장자 사용

```python
# BAD: File extensions
# /api/users.json
# /api/users.xml
# /api/reports/monthly.pdf

# GOOD: Content negotiation via Accept header
# GET /api/users
# Accept: application/json        -> returns JSON
# Accept: application/xml         -> returns XML
# Accept: text/csv                -> returns CSV

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import csv
import io

app = FastAPI()

@app.get("/api/users")
async def list_users(request: Request):
    users = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    accept = request.headers.get("accept", "application/json")

    if "text/csv" in accept:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["id", "name"])
        writer.writeheader()
        writer.writerows(users)
        return Response(content=output.getvalue(), media_type="text/csv")

    return JSONResponse(content={"data": users})
```

### 3. 불필요한 래퍼

```python
# BAD: Redundant path segments
# /api/v1/service/resource/users
# /api/rest/v2/data/users

# GOOD: Clean, direct paths
# /api/v1/users
# /api/users
```

### 4. 일관성 없는 네이밍

```python
# BAD: Mixed conventions across endpoints
# /api/users              -- plural
# /api/order              -- singular
# /api/productList        -- camelCase with "List" suffix
# /api/get-categories     -- verb + hyphen
# /api/User_Profiles      -- PascalCase + underscore

# GOOD: Uniform conventions
# /api/users
# /api/orders
# /api/products
# /api/categories
# /api/user-profiles
```

### 5. 구현 세부 정보 노출

```python
# BAD: Leaking database/implementation details
# /api/tbl_users                  -- table prefix
# /api/mysql/users                -- database engine
# /api/users?sql=SELECT * FROM... -- raw SQL exposure
# /api/v2/legacy/users            -- "legacy" is internal context

# GOOD: Abstract over implementation
# /api/users
# The client does not need to know about your database schema.
```

### 6. 후행 슬래시 불일치

```python
from fastapi import FastAPI
from fastapi.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse

app = FastAPI()

# Choose ONE convention and enforce it.
# Most common: No trailing slash.

# FastAPI strips trailing slashes by default.
# /api/users and /api/users/ both route to the same handler.

# If you need strict enforcement:
class TrailingSlashMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path != "/" and request.url.path.endswith("/"):
            url = str(request.url).rstrip("/")
            return RedirectResponse(url=url, status_code=301)
        return await call_next(request)

# app.add_middleware(TrailingSlashMiddleware)
```

---

## 실용 가이드라인

### URL 설계 체크리스트

```
1. Resources are plural nouns:              /api/users, /api/orders
2. IDs in path for specific resources:      /api/users/42
3. Sub-resources for ownership:             /api/users/42/orders
4. Max two levels of nesting:               /api/users/42/orders (not deeper)
5. Query params for filtering:              /api/users?role=admin
6. Query params for pagination:             /api/users?page=2&per_page=20
7. Query params for sorting:                /api/users?sort=name&order=asc
8. Lowercase with hyphens:                  /api/order-items (not orderItems)
9. No verbs in URLs:                        POST /api/users (not /api/createUser)
10. No file extensions:                     /api/users (not /api/users.json)
11. Consistent trailing slash policy:       /api/users (no trailing slash)
12. Meaningful, domain-specific names:      /api/invoices (not /api/documents)
```

### 완전한 URL 설계 예제

```python
from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI()

# --- E-commerce API URL Design ---

# Products (collection)
# GET    /api/products                         -- list products
# POST   /api/products                         -- create product
# GET    /api/products/{id}                    -- get product
# PATCH  /api/products/{id}                    -- update product
# DELETE /api/products/{id}                    -- delete product

# Product reviews (sub-resource)
# GET    /api/products/{id}/reviews            -- list product reviews
# POST   /api/products/{id}/reviews            -- add review

# Product variants (sub-resource)
# GET    /api/products/{id}/variants           -- list variants
# POST   /api/products/{id}/variants           -- create variant

# Categories (independent resource)
# GET    /api/categories                       -- list categories
# GET    /api/categories/{id}                  -- get category
# GET    /api/categories/{id}/products         -- products in category

# Orders (independent resource)
# GET    /api/orders                           -- list orders
# POST   /api/orders                           -- create order
# GET    /api/orders/{id}                      -- get order
# POST   /api/orders/{id}/cancel               -- cancel order (action)

# Order items (sub-resource)
# GET    /api/orders/{id}/items                -- list items in order

# Users (independent resource)
# GET    /api/users/{id}                       -- get user
# GET    /api/users/{id}/orders                -- user's orders
# GET    /api/users/{id}/profile               -- user's profile (singleton)

# Search (cross-resource)
# GET    /api/search?q=widget&type=products    -- search across resources

# Example implementation
@app.get("/api/products")
async def list_products(
    category: str | None = None,
    min_price: float | None = Query(default=None, ge=0),
    max_price: float | None = Query(default=None, ge=0),
    sort: str = Query(default="created_at", regex="^(name|price|created_at|rating)$"),
    order: str = Query(default="desc", regex="^(asc|desc)$"),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    q: str | None = Query(default=None, min_length=2),
):
    """Full-featured product listing with clean URL design."""
    return {
        "data": [
            {"id": 1, "name": "Widget", "price": 9.99, "category": "tools"}
        ],
        "meta": {
            "page": page,
            "per_page": per_page,
            "total": 1,
        },
        "_links": {
            "self": {"href": f"/api/products?page={page}&per_page={per_page}"},
        }
    }
```

### Flask 비교

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.get("/api/products")
def list_products():
    """Flask version of the product listing endpoint."""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    category = request.args.get("category")
    sort = request.args.get("sort", "created_at")
    order = request.args.get("order", "desc")
    q = request.args.get("q")

    products = [{"id": 1, "name": "Widget", "price": 9.99}]

    if category:
        products = [p for p in products if p.get("category") == category]

    return jsonify({
        "data": products,
        "meta": {"page": page, "per_page": per_page, "total": len(products)},
    })

@app.get("/api/products/<int:product_id>")
def get_product(product_id):
    """Flask: path parameter with type converter."""
    return jsonify({"id": product_id, "name": "Widget", "price": 9.99})

@app.get("/api/products/<int:product_id>/reviews")
def get_product_reviews(product_id):
    """Flask: nested sub-resource."""
    return jsonify({
        "data": [{"id": 1, "product_id": product_id, "rating": 5, "text": "Great!"}]
    })
```

---

## 연습 문제

### 연습 1: URL 수정하기

각 URL을 REST 네이밍 규칙에 맞게 다시 작성하십시오:

1. `GET /api/getAllUsers`
2. `POST /api/user/create`
3. `PUT /api/updateProduct/42`
4. `DELETE /api/remove-order?id=101`
5. `GET /api/Users/42/OrderList`
6. `GET /api/tbl_categories`
7. `POST /api/users/42/orders/101/items/5/reviews/create`

### 연습 2: URL 계층 설계

다음을 포함하는 **프로젝트 관리** 애플리케이션의 완전한 URL 계층을 설계하십시오:
- 조직, 프로젝트, 태스크, 댓글, 첨부파일
- 태스크 할당 (사용자-태스크 관계)
- 프로젝트 마일스톤

각 리소스에 대해 다음을 정의합니다:
- 컬렉션 URL과 싱글톤 URL
- 하위 리소스인 것과 최상위 리소스인 것
- 필터링 및 정렬을 위한 쿼리 파라미터

### 연습 3: 필터링 구현

다음을 지원하는 `/api/events` FastAPI 엔드포인트를 구축하십시오:
- 날짜 범위 필터링 (`start_after`, `start_before`)
- 카테고리 필터링 (콤마로 구분된 다중 카테고리)
- 위치 필터링 (`city`, `country`)
- 전문 검색 (`q`)
- `date`, `name`, `popularity`로 정렬
- `page`와 `per_page`로 페이지네이션

### 연습 4: URL 설계 리뷰

작업 중인 기존 API의 URL 설계를 감사하십시오:
- 리소스가 일관되게 명명되어 있는가 (복수형 명사, 소문자)?
- 중첩 깊이가 적절한가?
- 필터링에는 쿼리 파라미터, 식별에는 경로 파라미터가 올바르게 사용되고 있는가?
- 안티패턴이 있는가 (URL에 동사, 파일 확장자, 일관성 없는 케이싱)?

구체적인 권장 사항을 포함한 간략한 보고서를 작성합니다.

---

## 요약

이 레슨에서 다룬 내용:
1. 리소스 네이밍 규칙: 명사, 소문자, 하이픈
2. 부모-자식 중첩(최대 두 단계)을 활용한 계층적 URL 설계
3. 컬렉션과 싱글톤에 대한 복수형 규칙
4. 필터링, 정렬, 페이지네이션, 필드 선택을 위한 쿼리 파라미터 패턴
5. 단순 동등 필터부터 브래킷 구문까지의 필터링 접근 방식
6. 일반적인 URL 안티패턴과 회피 방법

---

**이전**: [REST 아키텍처](02_REST_Architecture.md) | [개요](00_Overview.md) | **다음**: [요청과 응답 설계](04_Request_and_Response_Design.md)

**License**: CC BY-NC 4.0
