# 레슨 5: 페이지네이션과 필터링

**이전**: [요청과 응답 설계](04_Request_and_Response_Design.md) | [개요](00_Overview.md) | **다음**: [인증과 인가](06_Authentication_and_Authorization.md)

## 학습 목표(Learning Objectives)

이 레슨을 마치면 다음을 할 수 있습니다:

1. 오프셋 기반, 커서 기반, 키셋 페이지네이션을 비교하고 적합한 전략을 선택할 수 있다
2. HATEOAS 탐색 링크가 포함된 페이지네이션을 구현할 수 있다
3. 동등, 범위, 다중 값 쿼리를 위한 필터링 구문을 설계할 수 있다
4. 다중 필드 지원과 방향 제어가 포함된 정렬을 추가할 수 있다
5. 응답 크기를 줄이기 위한 필드 선택(희소 필드셋)을 구현할 수 있다
6. 빈 결과, 잘못된 파라미터, 대규모 데이터셋 등의 엣지 케이스를 처리할 수 있다

---

소수 이상의 항목을 반환할 수 있는 모든 컬렉션 엔드포인트에는 페이지네이션이 필요합니다. 이것이 없으면 단순한 `GET /api/users`가 수백만 개의 레코드를 반환하려고 시도하여 서버를 다운시키거나 클라이언트를 압도할 수 있습니다. 페이지네이션, 필터링, 정렬, 필드 선택이 함께 작동하여 소비자에게 어떤 데이터를 얼마나 받을지에 대한 정밀한 제어를 제공합니다.

> **비유:** 페이지네이션은 도서관 카탈로그와 같습니다. 모든 책을 바닥에 쏟아놓고 찾지 않습니다 -- 페이지별로 탐색하고, 장르별로 필터링하고, 저자별로 정렬하고, 필요한 것만 대출합니다. 잘 설계된 API는 소비자에게 동일한 수준의 제어를 제공합니다.

## 목차
1. [오프셋 페이지네이션](#오프셋-페이지네이션)
2. [커서 페이지네이션](#커서-페이지네이션)
3. [키셋 페이지네이션](#키셋-페이지네이션)
4. [페이지네이션 비교](#페이지네이션-비교)
5. [HATEOAS 페이지네이션 링크](#hateoas-페이지네이션-링크)
6. [필터링](#필터링)
7. [정렬](#정렬)
8. [필드 선택](#필드-선택)
9. [연습 문제](#연습-문제)

---

## 오프셋 페이지네이션

오프셋 페이지네이션은 `page`와 `per_page`(또는 `offset`과 `limit`) 파라미터를 사용하여 결과 집합을 분할합니다.

### 기본 구현

```python
from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI()

# Simulated database
ALL_PRODUCTS = [
    {"id": i, "name": f"Product {i}", "price": round(9.99 + i * 0.5, 2), "category": "electronics"}
    for i in range(1, 201)
]

@app.get("/api/products")
async def list_products(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    per_page: int = Query(default=20, ge=1, le=100, description="Items per page"),
):
    """
    Offset pagination with page/per_page.

    Example: GET /api/products?page=3&per_page=10
    Returns items 21-30 of the total set.
    """
    total = len(ALL_PRODUCTS)
    total_pages = (total + per_page - 1) // per_page

    # Clamp page to valid range
    if page > total_pages and total > 0:
        page = total_pages

    start = (page - 1) * per_page
    end = start + per_page
    items = ALL_PRODUCTS[start:end]

    return {
        "data": items,
        "meta": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
        }
    }
```

### Offset/Limit 변형

```python
@app.get("/api/logs")
async def list_logs(
    offset: int = Query(default=0, ge=0, description="Number of items to skip"),
    limit: int = Query(default=50, ge=1, le=200, description="Max items to return"),
):
    """
    Offset/limit pagination (0-indexed).

    Example: GET /api/logs?offset=100&limit=50
    Returns items 101-150.
    """
    total = 5000
    # In practice: SELECT * FROM logs ORDER BY created_at DESC OFFSET {offset} LIMIT {limit}
    items = [{"id": i, "message": f"Log entry {i}"} for i in range(offset + 1, min(offset + limit + 1, total + 1))]

    return {
        "data": items,
        "meta": {
            "offset": offset,
            "limit": limit,
            "total": total,
            "has_more": offset + limit < total,
        }
    }
```

### 오프셋 페이지네이션의 단점

```
Problem 1: Page drift (data inconsistency)
─────────────────────────────────────────
Request 1: GET /api/users?page=1&per_page=10  (gets users 1-10)
            -- Someone deletes user 5 --
Request 2: GET /api/users?page=2&per_page=10  (gets users 11-20)
            But user 11 shifted to position 10, so it was already in page 1!
            Result: user 11 is skipped entirely.

Problem 2: Performance degradation
─────────────────────────────────
Page 1:    SELECT * FROM users LIMIT 10 OFFSET 0      -- fast
Page 100:  SELECT * FROM users LIMIT 10 OFFSET 990    -- still OK
Page 10000: SELECT * FROM users LIMIT 10 OFFSET 99990 -- SLOW!
            The database must scan and discard 99,990 rows.
```

---

## 커서 페이지네이션

커서 페이지네이션은 결과 집합의 특정 위치를 가리키는 불투명 토큰(커서)을 사용합니다. 클라이언트는 다음 페이지를 얻기 위해 커서를 전송합니다.

### 구현

```python
from fastapi import FastAPI, Query
import base64
import json

app = FastAPI()

# Simulated database
ALL_USERS = [
    {"id": i, "name": f"User {i}", "created_at": f"2025-01-{i:02d}T00:00:00Z"}
    for i in range(1, 101)
]

def encode_cursor(data: dict) -> str:
    """Encode pagination state as an opaque base64 cursor."""
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()

def decode_cursor(cursor: str) -> dict:
    """Decode an opaque cursor back to pagination state."""
    return json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())

@app.get("/api/users")
async def list_users(
    limit: int = Query(default=20, ge=1, le=100),
    after: str | None = Query(default=None, description="Cursor for next page"),
    before: str | None = Query(default=None, description="Cursor for previous page"),
):
    """
    Cursor-based pagination.

    Example:
        GET /api/users?limit=10             -> first page
        GET /api/users?limit=10&after=eyJ.. -> next page (using cursor from previous response)
    """
    # Determine starting position from cursor
    start_index = 0
    if after:
        cursor_data = decode_cursor(after)
        start_index = cursor_data["index"]
    elif before:
        cursor_data = decode_cursor(before)
        start_index = max(0, cursor_data["index"] - limit)

    # Fetch one extra item to determine if there are more
    items = ALL_USERS[start_index:start_index + limit + 1]
    has_next = len(items) > limit
    items = items[:limit]  # trim the extra item
    has_prev = start_index > 0

    # Build cursors
    response = {
        "data": items,
        "meta": {
            "has_next": has_next,
            "has_prev": has_prev,
            "count": len(items),
        },
        "cursors": {}
    }

    if has_next:
        response["cursors"]["after"] = encode_cursor({"index": start_index + limit})
    if has_prev:
        response["cursors"]["before"] = encode_cursor({"index": start_index})

    return response
```

### 커서 페이지네이션의 장점

```
1. No page drift: Cursors point to a specific item, not a position.
   Even if items are inserted or deleted, the cursor remains valid.

2. Consistent performance: The database uses an index scan from the
   cursor position. No matter how deep you paginate, performance
   is constant (O(1) seek vs O(n) offset scan).

3. Opaque tokens: Clients cannot manipulate cursors to jump to
   arbitrary pages. This prevents abuse and simplifies the API contract.
```

---

## 키셋 페이지네이션

키셋 페이지네이션("seek method"라고도 함)은 현재 페이지의 마지막 항목의 **정렬 키 값**을 사용하여 다음 페이지를 가져옵니다. 커서 페이지네이션의 투명한 버전입니다.

### 구현

```python
from fastapi import FastAPI, Query
from datetime import datetime

app = FastAPI()

@app.get("/api/events")
async def list_events(
    limit: int = Query(default=20, ge=1, le=100),
    after_id: int | None = Query(default=None, description="Return events after this ID"),
    after_date: str | None = Query(default=None, description="Return events after this date"),
):
    """
    Keyset pagination using the sort key directly.

    Example:
        GET /api/events?limit=20
        -> Returns first 20 events, last one has id=20, date=2025-01-20

        GET /api/events?limit=20&after_id=20&after_date=2025-01-20
        -> Returns next 20 events starting after id=20
    """
    # SQL equivalent:
    # SELECT * FROM events
    # WHERE (date, id) > (:after_date, :after_id)
    # ORDER BY date ASC, id ASC
    # LIMIT :limit + 1

    # Simulated data
    all_events = [
        {"id": i, "title": f"Event {i}", "date": f"2025-01-{i:02d}"}
        for i in range(1, 101)
    ]

    # Apply keyset filter
    if after_id is not None:
        all_events = [e for e in all_events if e["id"] > after_id]

    # Fetch limit + 1 to detect if more exist
    items = all_events[:limit + 1]
    has_more = len(items) > limit
    items = items[:limit]

    response = {
        "data": items,
        "meta": {
            "count": len(items),
            "has_more": has_more,
        }
    }

    if has_more and items:
        last = items[-1]
        response["meta"]["next_params"] = {
            "after_id": last["id"],
            "after_date": last["date"],
        }

    return response
```

### 키셋 vs 커서

```
Keyset Pagination:
  - Sort key values are visible (e.g., after_id=42)
  - Client can construct their own starting point
  - Transparent and debuggable
  - Client must know the sort fields

Cursor Pagination:
  - Opaque token (e.g., after=eyJpZCI6NDJ9)
  - Server controls the format entirely
  - More flexible (can change internal representation without breaking clients)
  - Client cannot jump to arbitrary positions

Most APIs use cursor pagination (opaque) for public APIs
and keyset pagination (transparent) for internal APIs.
```

---

## 페이지네이션 비교

| 특성 | 오프셋 | 커서 | 키셋 |
|------|--------|------|------|
| N 페이지로 이동 | 가능 | 불가 | 불가 |
| 총 개수 | 가능 (비용 발생) | 선택적 | 선택적 |
| 깊은 페이지 성능 | 저하 (O(n)) | 일정 (O(1)) | 일정 (O(1)) |
| 데이터 일관성 | 페이지 드리프트 가능 | 안정적 | 안정적 |
| 구현 난이도 | 간단 | 보통 | 보통 |
| 클라이언트 복잡도 | 낮음 | 낮음 | 낮음-보통 |
| 사용 사례 | 소규모 데이터셋, 관리자 UI | 소셜 피드, 타임라인 | 로그 항목, 감사 추적 |
| 실제 사례 | GitHub (repos), Shopify | Twitter, Facebook, Slack | Stripe (events), Datadog |

### 언제 어떤 것을 사용할 것인가

```python
def choose_pagination(
    dataset_size: str,      # "small" | "medium" | "large"
    needs_page_jump: bool,  # Does the UI need "go to page 42"?
    real_time_inserts: bool, # Are new items inserted frequently?
) -> str:
    if dataset_size == "small" and needs_page_jump:
        return "offset"  # simple, page jumps supported
    if real_time_inserts:
        return "cursor"  # no page drift
    if dataset_size == "large":
        return "keyset"  # constant performance
    return "cursor"  # safe default
```

---

## HATEOAS 페이지네이션 링크

페이지네이션 링크를 통해 클라이언트가 URL을 직접 구성하지 않고도 탐색할 수 있습니다.

### 표준 링크 관계

```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/api/articles")
async def list_articles(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
):
    total = 250
    total_pages = (total + per_page - 1) // per_page

    items = [
        {"id": i, "title": f"Article {i}"}
        for i in range((page - 1) * per_page + 1, min(page * per_page, total) + 1)
    ]

    # Build HATEOAS pagination links
    base = "/api/articles"
    links = {
        "self": {"href": f"{base}?page={page}&per_page={per_page}"},
        "first": {"href": f"{base}?page=1&per_page={per_page}"},
        "last": {"href": f"{base}?page={total_pages}&per_page={per_page}"},
    }
    if page > 1:
        links["prev"] = {"href": f"{base}?page={page - 1}&per_page={per_page}"}
    if page < total_pages:
        links["next"] = {"href": f"{base}?page={page + 1}&per_page={per_page}"}

    return {
        "data": items,
        "meta": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
        },
        "_links": links,
    }
```

### 커서 기반 탐색 링크

```python
@app.get("/api/feed")
async def list_feed(
    limit: int = Query(default=20, ge=1, le=100),
    after: str | None = None,
):
    """Cursor pagination with HATEOAS links."""
    # ... fetch items ...

    items = [{"id": i, "text": f"Post {i}"} for i in range(1, limit + 1)]
    next_cursor = encode_cursor({"index": 20})  # from cursor helper

    return {
        "data": items,
        "meta": {
            "count": len(items),
            "has_more": True,
        },
        "_links": {
            "self": {"href": f"/api/feed?limit={limit}" + (f"&after={after}" if after else "")},
            "next": {"href": f"/api/feed?limit={limit}&after={next_cursor}"},
        }
    }
```

### Link 헤더 (RFC 8288)

일부 API는 응답 본문 대신 또는 추가로 HTTP `Link` 헤더를 사용합니다:

```python
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/api/repos")
async def list_repos(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=30, ge=1, le=100),
):
    """GitHub-style pagination using Link header."""
    total_pages = 10
    base = "/api/repos"

    # Build Link header (RFC 8288)
    links = []
    if page < total_pages:
        links.append(f'<{base}?page={page + 1}&per_page={per_page}>; rel="next"')
    if page > 1:
        links.append(f'<{base}?page={page - 1}&per_page={per_page}>; rel="prev"')
    links.append(f'<{base}?page=1&per_page={per_page}>; rel="first"')
    links.append(f'<{base}?page={total_pages}&per_page={per_page}>; rel="last"')

    return JSONResponse(
        content={"data": [{"id": 1, "name": "repo-1"}]},
        headers={"Link": ", ".join(links)},
    )
```

---

## 필터링

### 동등 필터

```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/api/products")
async def list_products(
    category: str | None = Query(default=None, description="Filter by category"),
    brand: str | None = Query(default=None, description="Filter by brand"),
    in_stock: bool | None = Query(default=None, description="Filter by availability"),
    color: str | None = Query(default=None, description="Filter by color (comma-separated)"),
):
    """
    Simple equality filters.

    Examples:
        GET /api/products?category=electronics
        GET /api/products?brand=acme&in_stock=true
        GET /api/products?color=red,blue,green
    """
    # Build query conditions
    conditions = {}
    if category:
        conditions["category"] = category
    if brand:
        conditions["brand"] = brand
    if in_stock is not None:
        conditions["in_stock"] = in_stock
    if color:
        conditions["color__in"] = color.split(",")

    return {"data": [], "filters_applied": conditions}
```

### 범위 필터

```python
@app.get("/api/orders")
async def list_orders(
    min_total: float | None = Query(default=None, ge=0, description="Minimum order total"),
    max_total: float | None = Query(default=None, ge=0, description="Maximum order total"),
    created_after: str | None = Query(default=None, description="Orders created after (ISO 8601)"),
    created_before: str | None = Query(default=None, description="Orders created before (ISO 8601)"),
    status: str | None = Query(default=None, description="Comma-separated statuses"),
):
    """
    Range and multi-value filters.

    Examples:
        GET /api/orders?min_total=50&max_total=200
        GET /api/orders?created_after=2025-01-01&created_before=2025-06-30
        GET /api/orders?status=pending,processing
    """
    filters = {}
    if min_total is not None:
        filters["total__gte"] = min_total
    if max_total is not None:
        filters["total__lte"] = max_total
    if created_after:
        filters["created_at__gt"] = created_after
    if created_before:
        filters["created_at__lt"] = created_before
    if status:
        filters["status__in"] = status.split(",")

    return {"data": [], "filters_applied": filters}
```

### 전문 검색

```python
@app.get("/api/search")
async def search(
    q: str = Query(..., min_length=2, max_length=200, description="Search query"),
    type: str | None = Query(default=None, description="Resource type: products, users, orders"),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
):
    """
    Cross-resource search endpoint.

    Example: GET /api/search?q=wireless+headphones&type=products
    """
    # In practice: use PostgreSQL FTS, Elasticsearch, or similar
    return {
        "data": [
            {"type": "product", "id": 1, "title": "Wireless Headphones", "score": 0.95},
            {"type": "product", "id": 7, "title": "Wireless Earbuds", "score": 0.82},
        ],
        "meta": {
            "query": q,
            "type": type,
            "page": page,
            "per_page": per_page,
            "total": 2,
        }
    }
```

---

## 정렬

### 단일 필드 정렬

```python
@app.get("/api/products")
async def list_products_sorted(
    sort: str = Query(
        default="created_at",
        description="Field to sort by",
        regex="^(name|price|created_at|rating|popularity)$",
    ),
    order: str = Query(
        default="desc",
        description="Sort direction",
        regex="^(asc|desc)$",
    ),
):
    """
    Single-field sorting.

    Examples:
        GET /api/products?sort=price&order=asc      -- cheapest first
        GET /api/products?sort=rating&order=desc     -- highest rated first
        GET /api/products?sort=name&order=asc        -- alphabetical
    """
    return {
        "data": [{"id": 1, "name": "Widget", "price": 9.99}],
        "meta": {"sort": sort, "order": order},
    }
```

### 다중 필드 정렬

```python
@app.get("/api/employees")
async def list_employees(
    sort: str = Query(
        default="-created_at",
        description="Comma-separated sort fields. Prefix with - for descending.",
    ),
):
    """
    Multi-field sorting with direction prefix.

    Convention: "-" prefix means descending, no prefix means ascending.

    Examples:
        GET /api/employees?sort=department,-salary    -- by dept asc, then salary desc
        GET /api/employees?sort=-hire_date             -- newest hires first
        GET /api/employees?sort=last_name,first_name   -- alphabetical by full name
    """
    # Parse sort fields
    allowed_fields = {"first_name", "last_name", "department", "salary",
                      "hire_date", "created_at"}
    sort_specs = []
    for field_spec in sort.split(","):
        field_spec = field_spec.strip()
        if field_spec.startswith("-"):
            field = field_spec[1:]
            direction = "desc"
        else:
            field = field_spec
            direction = "asc"

        if field not in allowed_fields:
            continue  # or raise 400
        sort_specs.append({"field": field, "direction": direction})

    return {
        "data": [{"id": 1, "first_name": "Alice", "department": "Engineering"}],
        "meta": {"sort": sort_specs},
    }
```

---

## 필드 선택

필드 선택(희소 필드셋)은 클라이언트가 필요한 필드만 요청하여 페이로드 크기를 줄일 수 있게 합니다.

### 구현

```python
@app.get("/api/users")
async def list_users_with_fields(
    fields: str | None = Query(
        default=None,
        description="Comma-separated list of fields to include",
    ),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
):
    """
    Sparse fieldsets: return only requested fields.

    Examples:
        GET /api/users?fields=id,name,email         -- only id, name, email
        GET /api/users?fields=id,name                -- minimal response
        GET /api/users                                -- all fields (default)
    """
    # Full user objects
    users = [
        {
            "id": 1,
            "name": "Alice",
            "email": "alice@example.com",
            "role": "admin",
            "department": "Engineering",
            "phone": "+1234567890",
            "avatar_url": "/avatars/alice.jpg",
            "created_at": "2025-01-15T10:00:00Z",
            "updated_at": "2025-03-01T14:30:00Z",
        },
    ]

    # Apply field selection
    if fields:
        requested = set(fields.split(","))
        # Always include "id" for resource identity
        requested.add("id")
        allowed = {"id", "name", "email", "role", "department", "phone",
                    "avatar_url", "created_at", "updated_at"}
        valid_fields = requested & allowed

        users = [
            {k: v for k, v in user.items() if k in valid_fields}
            for user in users
        ]

    return {
        "data": users,
        "meta": {
            "page": page,
            "per_page": per_page,
            "total": len(users),
            "fields": fields.split(",") if fields else "all",
        }
    }
```

### 리소스 확장

필드 선택의 반대 개념으로 -- 관련 리소스를 응답에 포함시킵니다:

```python
@app.get("/api/orders/{order_id}")
async def get_order(
    order_id: int,
    expand: str | None = Query(
        default=None,
        description="Comma-separated resources to embed: items, user, shipping",
    ),
):
    """
    Resource expansion: embed related resources to reduce round trips.

    Examples:
        GET /api/orders/1                              -- basic order
        GET /api/orders/1?expand=items                 -- order + line items
        GET /api/orders/1?expand=items,user,shipping   -- fully expanded
    """
    order = {
        "id": order_id,
        "status": "shipped",
        "total": 149.99,
        "user_id": 1,
    }

    expansions = set(expand.split(",")) if expand else set()

    if "items" in expansions:
        order["items"] = [
            {"product_id": 10, "name": "Widget", "quantity": 2, "price": 9.99},
            {"product_id": 20, "name": "Gadget", "quantity": 1, "price": 129.99},
        ]

    if "user" in expansions:
        order["user"] = {"id": 1, "name": "Alice", "email": "alice@example.com"}

    if "shipping" in expansions:
        order["shipping"] = {
            "carrier": "FedEx",
            "tracking_number": "FX123456789",
            "estimated_delivery": "2025-02-01",
        }

    return {"data": order}
```

---

## 연습 문제

### 연습 1: 커서 페이지네이션 구현

커서 기반 페이지네이션이 포함된 `/api/messages` FastAPI 엔드포인트를 구축하십시오:
- `limit` (기본값 20, 최대 100)와 `after` (커서) 파라미터 수신
- `created_at` 내림차순(최신 순)으로 정렬된 메시지 반환
- 다음 페이지를 위한 `cursors.after` 포함
- `meta.has_more` 불리언 포함
- base64 인코딩된 JSON 커서 사용
- 500개의 메시지 데이터셋으로 테스트

### 연습 2: 고급 필터링

다음을 지원하는 `/api/products` 엔드포인트를 생성하십시오:
- 동등: `?category=electronics`
- 다중 값: `?color=red,blue,green`
- 범위: `?min_price=10&max_price=100`
- 날짜 범위: `?created_after=2025-01-01`
- 불리언: `?in_stock=true`
- 전문 검색: `?q=wireless`

소비자가 적용된 내용을 확인할 수 있도록 `meta` 객체에 적용된 필터를 반환합니다.

### 연습 3: 다중 필드 정렬

다음을 지원하는 정렬을 구현하십시오:
- 단일 필드: `?sort=price`
- 방향: `?sort=-price` (내림차순)
- 다중 필드: `?sort=category,-price` (카테고리 오름차순, 가격 내림차순)
- 유효성 검사: 알 수 없는 필드를 400 오류로 거부
- 기본 정렬: `-created_at` (최신 순)

### 연습 4: 희소 필드셋과 확장

다음을 지원하는 `/api/users/{id}` 엔드포인트를 구축하십시오:
- `?fields=id,name,email` -- 지정된 필드만 반환
- `?expand=orders,profile` -- 관련 리소스 포함
- 둘 다 결합: `?fields=id,name&expand=orders`
- 필드명과 확장 대상 유효성 검사

### 연습 5: 페이지네이션 전략 비교

`/api/logs` 엔드포인트의 세 가지 버전을 생성하십시오:
1. 오프셋 페이지네이션 (`page`, `per_page`)
2. 커서 페이지네이션 (`after`, `limit`)
3. 키셋 페이지네이션 (`after_id`, `after_timestamp`, `limit`)

10,000개의 로그 항목으로 채우고, 다음 상황에서의 성능과 동작을 비교합니다:
- 500페이지까지 페이지네이션
- 페이지네이션 중 항목 삽입
- 정렬 순서 변경

---

## 요약

이 레슨에서 다룬 내용:
1. 오프셋 페이지네이션: 간단하지만 페이지 드리프트와 깊은 페이지에서의 성능 저하 문제
2. 커서 페이지네이션: 안정적이고 성능 좋은 깊은 페이지네이션을 위한 불투명 토큰
3. 키셋 페이지네이션: 일정한 성능을 가진 투명한 키 기반 탐색
4. HATEOAS 페이지네이션 링크: self, first, last, prev, next 탐색
5. 필터링 패턴: 동등, 범위, 다중 값, 전문 검색
6. 정렬: 방향 접두사를 사용한 단일 및 다중 필드 정렬
7. 필드 선택: 응답 최적화를 위한 희소 필드셋과 리소스 확장

---

**이전**: [요청과 응답 설계](04_Request_and_Response_Design.md) | [개요](00_Overview.md) | **다음**: [인증과 인가](06_Authentication_and_Authorization.md)

**License**: CC BY-NC 4.0
