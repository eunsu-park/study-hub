# 레슨 4: 요청과 응답 설계

**이전**: [URL 설계와 네이밍](03_URL_Design_and_Naming.md) | [개요](00_Overview.md) | **다음**: [페이지네이션과 필터링](05_Pagination_and_Filtering.md)

## 학습 목표(Learning Objectives)

이 레슨을 마치면 다음을 할 수 있습니다:

1. 각 HTTP 메서드(GET, POST, PUT, PATCH, DELETE)를 정의된 시맨틱에 따라 적용할 수 있다
2. 모든 응답 시나리오에 대해 올바른 HTTP 상태 코드를 선택할 수 있다
3. 다중 응답 포맷을 지원하는 콘텐츠 협상을 구현할 수 있다
4. 재시도가 안전한 멱등 엔드포인트를 설계할 수 있다
5. 일관된 요청 및 응답 엔벨로프를 구성할 수 있다
6. JSON Merge Patch와 JSON Patch를 사용한 부분 업데이트를 처리할 수 있다

---

요청-응답 사이클은 모든 HTTP API 인터랙션의 원자적 단위입니다. 메서드, 상태 코드, 헤더, 본문 구조를 올바르게 설정하는 것이 전문적인 API와 즉흥적인 엔드포인트 모음을 구분합니다. 이 레슨에서는 클라이언트가 서버와 대화하고 서버가 응답하는 전체 메커니즘을 다룹니다.

> **비유:** HTTP 메서드는 우체국의 다른 창구와 같습니다. GET은 "수령" 창구(아무것도 변경하지 않고 정보를 가져감)입니다. POST는 "접수" 창구(새로운 것을 제출)입니다. PUT은 전체 소포를 교체합니다. PATCH는 배송 라벨을 수정합니다. DELETE는 시스템에서 소포를 제거합니다. 올바른 창구를 사용하는 것이 중요합니다.

## 목차
1. [HTTP 메서드](#http-메서드)
2. [메서드 속성](#메서드-속성)
3. [HTTP 상태 코드](#http-상태-코드)
4. [콘텐츠 협상](#콘텐츠-협상)
5. [멱등성](#멱등성)
6. [요청과 응답 엔벨로프](#요청과-응답-엔벨로프)
7. [부분 업데이트](#부분-업데이트)
8. [연습 문제](#연습-문제)

---

## HTTP 메서드

### GET -- 리소스 조회

GET은 **안전**(부작용 없음)하고 **멱등**(여러 번 호출해도 한 번 호출한 것과 같은 효과)합니다. 서버 상태를 절대 수정해서는 안 됩니다.

```python
from fastapi import FastAPI, Query, HTTPException

app = FastAPI()

# GET collection
@app.get("/api/books")
async def list_books(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    genre: str | None = None,
):
    """Retrieve a paginated list of books. No side effects."""
    books = [
        {"id": 1, "title": "Dune", "genre": "sci-fi"},
        {"id": 2, "title": "1984", "genre": "dystopian"},
    ]
    if genre:
        books = [b for b in books if b["genre"] == genre]

    return {
        "data": books,
        "meta": {"page": page, "per_page": per_page, "total": len(books)},
    }

# GET single resource
@app.get("/api/books/{book_id}")
async def get_book(book_id: int):
    """Retrieve a single book by ID."""
    if book_id > 100:
        raise HTTPException(status_code=404, detail="Book not found")
    return {"id": book_id, "title": "Dune", "author": "Frank Herbert", "genre": "sci-fi"}
```

### POST -- 리소스 생성

POST는 **안전하지 않으며** **멱등하지 않습니다**. 각 호출이 새 리소스를 생성할 수 있습니다.

```python
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

class BookCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    author: str = Field(..., min_length=1)
    isbn: str | None = Field(default=None, pattern=r"^\d{13}$")

@app.post("/api/books", status_code=201)
async def create_book(book: BookCreate):
    """Create a new book. Returns 201 with Location header."""
    new_book = {"id": 42, "title": book.title, "author": book.author, "isbn": book.isbn}
    return JSONResponse(
        content=new_book,
        status_code=201,
        headers={"Location": f"/api/books/{new_book['id']}"},
    )
```

### PUT -- 리소스 교체

PUT은 **멱등**합니다 -- 동일한 PUT 요청을 여러 번 보내도 같은 결과를 생성합니다. **전체** 리소스를 교체합니다.

```python
class BookUpdate(BaseModel):
    """PUT requires ALL fields (full replacement)."""
    title: str
    author: str
    isbn: str | None = None
    genre: str
    published_year: int

@app.put("/api/books/{book_id}")
async def replace_book(book_id: int, book: BookUpdate):
    """
    Full replacement of a book resource.
    Every field must be provided -- missing fields are set to their default/null.
    """
    return {
        "id": book_id,
        "title": book.title,
        "author": book.author,
        "isbn": book.isbn,
        "genre": book.genre,
        "published_year": book.published_year,
    }
```

### PATCH -- 부분 업데이트

PATCH는 **부분** 업데이트를 위한 것입니다. 요청 본문에 포함된 필드만 수정됩니다.

```python
class BookPatch(BaseModel):
    """PATCH: all fields optional (partial update)."""
    title: str | None = None
    author: str | None = None
    isbn: str | None = None
    genre: str | None = None
    published_year: int | None = None

@app.patch("/api/books/{book_id}")
async def update_book(book_id: int, updates: BookPatch):
    """
    Partial update of a book resource.
    Only provided fields are modified; others remain unchanged.
    """
    # In practice, fetch existing book and merge updates
    existing = {"id": book_id, "title": "Dune", "author": "Frank Herbert",
                "genre": "sci-fi", "published_year": 1965}

    update_data = updates.model_dump(exclude_unset=True)
    merged = {**existing, **update_data}
    return merged
```

### DELETE -- 리소스 삭제

DELETE는 **멱등**합니다 -- 이미 삭제된 리소스를 다시 삭제해도 실패하지 않아야 합니다.

```python
from fastapi import Response

@app.delete("/api/books/{book_id}", status_code=204)
async def delete_book(book_id: int):
    """
    Delete a book. Returns 204 No Content (no response body).
    Idempotent: deleting a non-existent book also returns 204.
    """
    # Perform deletion in database
    # If already deleted, still return 204 (idempotent)
    return Response(status_code=204)
```

### HEAD와 OPTIONS

```python
@app.head("/api/books/{book_id}")
async def head_book(book_id: int):
    """
    HEAD: Identical to GET but returns only headers, no body.
    Useful for checking existence or getting metadata without downloading content.
    """
    return Response(
        headers={
            "Content-Type": "application/json",
            "X-Book-Title": "Dune",
            "Last-Modified": "Sat, 15 Jan 2025 10:00:00 GMT",
        }
    )

@app.options("/api/books")
async def options_books():
    """
    OPTIONS: Returns allowed methods for the resource.
    Often handled automatically by frameworks (CORS preflight).
    """
    return Response(
        headers={
            "Allow": "GET, POST, HEAD, OPTIONS",
            "Access-Control-Allow-Methods": "GET, POST",
        }
    )
```

---

## 메서드 속성

| 메서드 | 안전 | 멱등 | 캐시 가능 | 본문 (요청) | 본문 (응답) |
|--------|------|------|-----------|------------|------------|
| GET | 예 | 예 | 예 | 아니오 | 예 |
| HEAD | 예 | 예 | 예 | 아니오 | 아니오 |
| POST | 아니오 | 아니오 | 드물게 | 예 | 예 |
| PUT | 아니오 | 예 | 아니오 | 예 | 예 |
| PATCH | 아니오 | 아니오* | 아니오 | 예 | 예 |
| DELETE | 아니오 | 예 | 아니오 | 드물게 | 드물게 |
| OPTIONS | 예 | 예 | 아니오 | 아니오 | 예 |

\* PATCH는 멱등할 **수** 있지만(JSON Merge Patch) 필수는 아닙니다.

### 안전 vs 멱등

```python
# SAFE: No side effects. The resource is not modified.
# GET /api/books         -- safe: just reading
# HEAD /api/books/1      -- safe: just checking headers

# IDEMPOTENT: Multiple identical calls produce the same result.
# PUT /api/books/1       -- idempotent: same PUT = same state
# DELETE /api/books/1    -- idempotent: delete twice = book still gone
# GET /api/books/1       -- idempotent: same GET = same response

# NOT IDEMPOTENT:
# POST /api/books        -- not idempotent: each POST may create a new book
# PATCH /api/books/1     -- not guaranteed idempotent (depends on patch format)
```

---

## HTTP 상태 코드

### 2xx -- 성공

```python
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

app = FastAPI()

# 200 OK -- General success (GET, PATCH)
@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    return {"id": user_id, "name": "Alice"}  # implicit 200

# 201 Created -- Resource created (POST, PUT)
@app.post("/api/users", status_code=201)
async def create_user(name: str, email: str):
    new_user = {"id": 1, "name": name, "email": email}
    return JSONResponse(
        content=new_user,
        status_code=201,
        headers={"Location": "/api/users/1"},
    )

# 202 Accepted -- Request accepted for async processing
@app.post("/api/reports/generate", status_code=202)
async def generate_report():
    """Long-running operation. Check status later."""
    return {
        "job_id": "rpt_abc123",
        "status": "processing",
        "status_url": "/api/reports/jobs/rpt_abc123",
    }

# 204 No Content -- Success with no response body (DELETE)
@app.delete("/api/users/{user_id}", status_code=204)
async def delete_user(user_id: int):
    return Response(status_code=204)
```

### 3xx -- 리다이렉션

```python
from fastapi.responses import RedirectResponse

# 301 Moved Permanently
@app.get("/api/v1/users")
async def old_users_endpoint():
    """Resource has permanently moved to a new URL."""
    return RedirectResponse(url="/api/v2/users", status_code=301)

# 304 Not Modified (conditional GET)
@app.get("/api/products/{product_id}")
async def get_product(product_id: int, request: Request):
    etag = '"abc123"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304)
    return JSONResponse(
        content={"id": product_id, "name": "Widget"},
        headers={"ETag": etag},
    )
```

### 4xx -- 클라이언트 오류

```python
from fastapi import HTTPException

# 400 Bad Request -- Malformed request
@app.post("/api/orders")
async def create_order(items: list[dict]):
    if not items:
        raise HTTPException(
            status_code=400,
            detail={"code": "EMPTY_ORDER", "message": "Order must contain at least one item"}
        )
    return {"id": 1, "items": items}

# 401 Unauthorized -- Missing or invalid authentication
@app.get("/api/me")
async def get_current_user():
    raise HTTPException(
        status_code=401,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )

# 403 Forbidden -- Authenticated but not authorized
@app.delete("/api/admin/users/{user_id}")
async def admin_delete_user(user_id: int):
    raise HTTPException(
        status_code=403,
        detail="You do not have permission to delete users"
    )

# 404 Not Found -- Resource does not exist
@app.get("/api/books/{book_id}")
async def get_book_example(book_id: int):
    raise HTTPException(status_code=404, detail=f"Book {book_id} not found")

# 405 Method Not Allowed
# Automatically handled by frameworks when method is not registered

# 409 Conflict -- State conflict (e.g., duplicate)
@app.post("/api/users")
async def create_user_example(email: str):
    raise HTTPException(
        status_code=409,
        detail={"code": "DUPLICATE_EMAIL", "message": f"A user with email '{email}' already exists"}
    )

# 422 Unprocessable Entity -- Validation error
# FastAPI returns this automatically for Pydantic validation failures

# 429 Too Many Requests -- Rate limit exceeded
@app.get("/api/search")
async def search():
    raise HTTPException(
        status_code=429,
        detail="Rate limit exceeded",
        headers={"Retry-After": "60"},
    )
```

### 5xx -- 서버 오류

```python
# 500 Internal Server Error -- Unexpected failure
# Never return raw stack traces to clients!
@app.get("/api/data")
async def get_data():
    try:
        # ... operation that might fail
        result = 1 / 0
    except Exception:
        raise HTTPException(
            status_code=500,
            detail={"code": "INTERNAL_ERROR", "message": "An unexpected error occurred"}
        )

# 502 Bad Gateway -- Upstream service failure
@app.get("/api/exchange-rates")
async def get_rates():
    raise HTTPException(
        status_code=502,
        detail="Unable to reach the exchange rate provider"
    )

# 503 Service Unavailable -- Server temporarily unavailable
@app.get("/api/health")
async def health():
    return JSONResponse(
        content={"status": "maintenance"},
        status_code=503,
        headers={"Retry-After": "300"},  # try again in 5 minutes
    )
```

### 상태 코드 결정 트리

```
Is the request valid?
├── No → Is it a client error?
│        ├── Bad syntax/format     → 400 Bad Request
│        ├── Missing auth          → 401 Unauthorized
│        ├── Insufficient perms    → 403 Forbidden
│        ├── Resource not found    → 404 Not Found
│        ├── Wrong method          → 405 Method Not Allowed
│        ├── Conflict / duplicate  → 409 Conflict
│        ├── Validation failure    → 422 Unprocessable Entity
│        └── Rate limited          → 429 Too Many Requests
│
└── Yes → Did the operation succeed?
         ├── Yes → What was the operation?
         │        ├── GET/PATCH        → 200 OK
         │        ├── POST (created)   → 201 Created
         │        ├── Async job        → 202 Accepted
         │        └── DELETE           → 204 No Content
         │
         └── No → Server error
                  ├── Bug / crash      → 500 Internal Server Error
                  ├── Upstream down    → 502 Bad Gateway
                  └── Maintenance      → 503 Service Unavailable
```

---

## 콘텐츠 협상

콘텐츠 협상은 클라이언트가 원하는 포맷으로 데이터를 요청할 수 있게 합니다.

### Accept 헤더

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import csv
import io
import xml.etree.ElementTree as ET

app = FastAPI()

@app.get("/api/users")
async def list_users(request: Request):
    """Returns JSON, CSV, or XML based on Accept header."""
    users = [
        {"id": 1, "name": "Alice", "email": "alice@example.com"},
        {"id": 2, "name": "Bob", "email": "bob@example.com"},
    ]
    accept = request.headers.get("accept", "application/json")

    # JSON (default)
    if "application/json" in accept or accept == "*/*":
        return JSONResponse(content={"data": users})

    # CSV
    if "text/csv" in accept:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["id", "name", "email"])
        writer.writeheader()
        writer.writerows(users)
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=users.csv"},
        )

    # XML
    if "application/xml" in accept:
        root = ET.Element("users")
        for user in users:
            elem = ET.SubElement(root, "user")
            for key, value in user.items():
                child = ET.SubElement(elem, key)
                child.text = str(value)
        xml_str = ET.tostring(root, encoding="unicode")
        return Response(content=xml_str, media_type="application/xml")

    # Unsupported format
    return JSONResponse(
        content={"error": f"Unsupported media type: {accept}"},
        status_code=406,  # 406 Not Acceptable
    )
```

### 요청의 Content-Type

```python
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()

@app.post("/api/import")
async def import_data(request: Request):
    """Accept different input formats via Content-Type header."""
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        data = await request.json()
        return {"format": "json", "records": len(data)}

    if "text/csv" in content_type:
        body = await request.body()
        lines = body.decode().strip().split("\n")
        return {"format": "csv", "records": len(lines) - 1}  # minus header

    raise HTTPException(
        status_code=415,  # 415 Unsupported Media Type
        detail=f"Unsupported Content-Type: {content_type}. Use application/json or text/csv."
    )
```

---

## 멱등성

### 멱등성이 중요한 이유

네트워크 장애는 발생합니다. POST 요청이 타임아웃되었을 때, 서버가 처리했는지 알 수 없습니다. 멱등성 없이는 클라이언트가 안전하게 재시도할 수 없습니다.

### 멱등성 키

```python
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import uuid

app = FastAPI()

# In-memory store for demo (use Redis/database in production)
processed_keys: dict[str, dict] = {}

class PaymentCreate(BaseModel):
    amount: float
    currency: str
    recipient: str

@app.post("/api/payments", status_code=201)
async def create_payment(
    payment: PaymentCreate,
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
):
    """
    Idempotent payment creation using Idempotency-Key header.

    Usage:
        POST /api/payments
        Idempotency-Key: unique-client-generated-uuid
        Content-Type: application/json
        {"amount": 100.00, "currency": "USD", "recipient": "alice@example.com"}

    If the same Idempotency-Key is sent again, the original response is returned.
    """
    # Check if this key was already processed
    if idempotency_key in processed_keys:
        return processed_keys[idempotency_key]  # return cached response

    # Process the payment
    result = {
        "id": f"pay_{uuid.uuid4().hex[:12]}",
        "amount": payment.amount,
        "currency": payment.currency,
        "recipient": payment.recipient,
        "status": "completed",
    }

    # Cache the result
    processed_keys[idempotency_key] = result
    return result
```

### 메서드별 멱등성 요약

```python
# GET    -- naturally idempotent (no side effects)
# PUT    -- naturally idempotent (same input = same state)
# DELETE -- naturally idempotent (deleting twice = same result)
# PATCH  -- may not be idempotent (e.g., "increment by 1")
# POST   -- NOT idempotent by default -- use Idempotency-Key!

# Making PATCH idempotent:
# BAD:  {"operation": "increment", "field": "views", "value": 1}
#       Calling this 3 times increments by 3 -- not idempotent.

# GOOD: {"views": 42}
#       Calling this 3 times sets views to 42 each time -- idempotent.
```

---

## 요청과 응답 엔벨로프

### 일관된 응답 구조

모든 API 응답을 감싸는 표준 엔벨로프를 정의합니다:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any
from datetime import datetime

app = FastAPI()

# --- Standard Response Envelope ---

class Meta(BaseModel):
    """Pagination and metadata for list responses."""
    page: int = 1
    per_page: int = 20
    total: int = 0
    total_pages: int = 0

class ApiResponse(BaseModel):
    """Standard envelope for all API responses."""
    data: Any
    meta: Meta | None = None
    errors: list[dict] | None = None
    request_id: str | None = None

# --- Single resource response ---
@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    return ApiResponse(
        data={
            "id": user_id,
            "name": "Alice",
            "email": "alice@example.com",
            "created_at": "2025-01-15T10:00:00Z",
        },
        request_id="req_abc123",
    )

# --- Collection response ---
@app.get("/api/users")
async def list_users(page: int = 1, per_page: int = 20):
    total = 95
    return ApiResponse(
        data=[
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ],
        meta=Meta(
            page=page,
            per_page=per_page,
            total=total,
            total_pages=(total + per_page - 1) // per_page,
        ),
        request_id="req_def456",
    )

# --- Error response (same envelope) ---
@app.get("/api/users/{user_id}/orders")
async def get_user_orders(user_id: int):
    # Example: user not found
    raise HTTPException(
        status_code=404,
        detail={
            "data": None,
            "errors": [{
                "code": "USER_NOT_FOUND",
                "message": f"User {user_id} does not exist",
                "field": None,
            }],
            "request_id": "req_ghi789",
        }
    )
```

### 요청 엔벨로프 (배치 연산용)

```python
class BatchAction(BaseModel):
    action: str  # "create" | "update" | "delete"
    resource_id: int | None = None
    data: dict | None = None

@app.post("/api/users/batch")
async def batch_users(actions: list[BatchAction]):
    """
    Batch endpoint for multiple operations in a single request.
    Each action in the batch is processed independently.
    """
    results = []
    for action in actions:
        if action.action == "create":
            results.append({"status": "created", "id": 99})
        elif action.action == "delete":
            results.append({"status": "deleted", "id": action.resource_id})
        else:
            results.append({"status": "error", "message": f"Unknown action: {action.action}"})

    return {"data": results, "meta": {"total_actions": len(actions)}}
```

---

## 부분 업데이트

### JSON Merge Patch (RFC 7396)

가장 단순한 접근 방식입니다: 업데이트할 필드만 포함된 JSON 객체를 전송합니다. `null`로 설정된 필드는 제거됩니다.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class UserMergePatch(BaseModel):
    """JSON Merge Patch: send only fields to update."""
    name: str | None = None
    email: str | None = None
    bio: str | None = None
    avatar_url: str | None = None

    model_config = {"extra": "forbid"}  # reject unknown fields

@app.patch("/api/users/{user_id}")
async def merge_patch_user(user_id: int, patch: UserMergePatch):
    """
    JSON Merge Patch (Content-Type: application/merge-patch+json).

    Example:
        PATCH /api/users/1
        {"name": "Alice Smith", "bio": null}

        Result: name updated, bio removed, email and avatar unchanged.
    """
    # Simulated existing user
    existing = {
        "id": user_id,
        "name": "Alice",
        "email": "alice@example.com",
        "bio": "Developer",
        "avatar_url": "/avatars/alice.jpg",
    }

    # Apply merge patch
    updates = patch.model_dump(exclude_unset=True)
    for key, value in updates.items():
        if value is None:
            existing.pop(key, None)  # null means "remove this field"
        else:
            existing[key] = value

    return existing
```

### JSON Patch (RFC 6902)

연산(add, remove, replace, move, copy, test)을 표현하는 더 강력한 형식입니다.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class JsonPatchOp(BaseModel):
    op: str       # "add" | "remove" | "replace" | "move" | "copy" | "test"
    path: str     # JSON Pointer (e.g., "/name", "/address/city")
    value: str | int | float | bool | None = None  # required for add/replace/test
    from_path: str | None = None  # required for move/copy (aliased as "from")

def apply_json_patch(document: dict, operations: list[JsonPatchOp]) -> dict:
    """Apply RFC 6902 JSON Patch operations to a document."""
    result = document.copy()
    for op in operations:
        # Strip leading "/" and handle nested paths
        path_parts = op.path.strip("/").split("/")
        key = path_parts[-1]

        if op.op == "replace":
            result[key] = op.value
        elif op.op == "add":
            result[key] = op.value
        elif op.op == "remove":
            result.pop(key, None)
        elif op.op == "test":
            if result.get(key) != op.value:
                raise HTTPException(
                    status_code=409,
                    detail=f"Test failed: {key} is {result.get(key)}, expected {op.value}"
                )
        # move and copy omitted for brevity
    return result

@app.patch("/api/users/{user_id}")
async def json_patch_user(user_id: int, operations: list[JsonPatchOp]):
    """
    JSON Patch (Content-Type: application/json-patch+json).

    Example:
        PATCH /api/users/1
        [
            {"op": "test", "path": "/name", "value": "Alice"},
            {"op": "replace", "path": "/name", "value": "Alice Smith"},
            {"op": "add", "path": "/nickname", "value": "Ali"},
            {"op": "remove", "path": "/bio"}
        ]
    """
    existing = {
        "id": user_id,
        "name": "Alice",
        "email": "alice@example.com",
        "bio": "Developer",
    }
    result = apply_json_patch(existing, operations)
    return result
```

### PUT vs PATCH 결정

```python
# Use PUT when:
# - You have the complete resource representation
# - You want atomic replacement
# - Missing fields should be cleared/defaulted

# Use PATCH when:
# - You only know the fields that changed
# - You want to minimize payload size
# - You want to avoid accidentally clearing fields

# Example: User has 20 fields. You want to update the name.
# PUT:   Send all 20 fields (19 unchanged + 1 updated)  -- heavy
# PATCH: Send only {"name": "New Name"}                  -- light

# Most real-world APIs use PATCH for updates and rarely use PUT.
```

---

## 연습 문제

### 연습 1: 메서드 매핑

각 시나리오에 대해 올바른 HTTP 메서드와 상태 코드를 식별하십시오:

1. 클라이언트가 제품 목록을 조회하려고 함
2. 클라이언트가 새 주문을 생성하려고 함
3. 클라이언트가 사용자의 이메일 주소만 업데이트하려고 함
4. 클라이언트가 전체 제품 레코드를 교체하려고 함
5. 클라이언트가 댓글을 삭제하려고 함
6. 클라이언트가 리소스의 존재 여부를 다운로드 없이 확인하려고 함
7. 장시간 실행되는 보고서 생성이 요청됨

### 연습 2: CRUD API 구축

다음을 갖춘 `Task` 리소스용 완전한 FastAPI 애플리케이션을 구현하십시오:
- GET `/api/tasks` -- 페이지네이션이 포함된 목록 (200)
- GET `/api/tasks/{id}` -- 단일 태스크 (200, 404)
- POST `/api/tasks` -- 생성 (201 + Location 헤더)
- PUT `/api/tasks/{id}` -- 전체 교체 (200, 404)
- PATCH `/api/tasks/{id}` -- merge patch를 사용한 부분 업데이트 (200, 404)
- DELETE `/api/tasks/{id}` -- 삭제 (204)

각 엔드포인트에 대해 적절한 상태 코드, 유효성 검사, 오류 응답을 포함합니다.

### 연습 3: 콘텐츠 협상 구현

동일한 데이터를 세 가지 포맷으로 제공하는 엔드포인트를 구축하십시오:
- `application/json` (기본값)
- `text/csv`
- `application/xml`

지원하지 않는 포맷에 대해 406 Not Acceptable을 반환합니다. 다음으로 테스트합니다:
```bash
http GET localhost:8000/api/users Accept:application/json
http GET localhost:8000/api/users Accept:text/csv
http GET localhost:8000/api/users Accept:application/xml
http GET localhost:8000/api/users Accept:text/yaml
```

### 연습 4: 멱등 결제

다음을 갖춘 결제 엔드포인트를 구축하십시오:
- `Idempotency-Key` 헤더 수신
- 첫 번째 호출에서 결제 처리
- 동일한 키로 이후 호출 시 캐시된 응답 반환
- 멱등성 키 없는 요청 거부 (400)
- 24시간 후 멱등성 키 만료

### 연습 5: 부분 업데이트 비교

`Article` 리소스(필드: title, body, tags, published)에 대해 JSON Merge Patch와 JSON Patch를 모두 구현하십시오. 비교 내용:
- 각각이 기존 목록에 태그를 추가하는 방법
- 각각이 필드를 제거하는 방법
- 사용 사례에 더 간단한 것은 무엇이고 그 이유

---

## 요약

이 레슨에서 다룬 내용:
1. HTTP 메서드(GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS)와 시맨틱 속성
2. 클래스별 상태 코드 (2xx 성공, 3xx 리다이렉션, 4xx 클라이언트 오류, 5xx 서버 오류)
3. Accept 및 Content-Type 헤더를 사용한 콘텐츠 협상
4. 안전한 재시도를 위한 멱등성 원칙과 멱등성 키 패턴
5. 일관된 요청/응답 엔벨로프 설계
6. 부분 업데이트 전략: JSON Merge Patch vs JSON Patch

---

**이전**: [URL 설계와 네이밍](03_URL_Design_and_Naming.md) | [개요](00_Overview.md) | **다음**: [페이지네이션과 필터링](05_Pagination_and_Filtering.md)

**License**: CC BY-NC 4.0
