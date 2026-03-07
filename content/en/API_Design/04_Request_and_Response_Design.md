# Lesson 4: Request and Response Design

**Previous**: [URL Design and Naming](03_URL_Design_and_Naming.md) | [Overview](00_Overview.md) | **Next**: [Pagination and Filtering](05_Pagination_and_Filtering.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply each HTTP method (GET, POST, PUT, PATCH, DELETE) according to its defined semantics
2. Select the correct HTTP status code for every response scenario
3. Implement content negotiation to support multiple response formats
4. Design idempotent endpoints that are safe to retry
5. Structure consistent request and response envelopes
6. Handle partial updates with JSON Merge Patch and JSON Patch

---

The request-response cycle is the atomic unit of every HTTP API interaction. Getting the methods, status codes, headers, and body structures right is what separates a professional API from an ad-hoc collection of endpoints. This lesson covers the full mechanics of how clients talk to servers and how servers talk back.

> **Analogy:** HTTP methods are like the different counters at a post office. GET is the "pick up" window (you take information without changing anything). POST is the "drop off" window (you submit something new). PUT replaces an entire package. PATCH updates the shipping label. DELETE removes a package from the system. Using the right counter matters.

## Table of Contents
1. [HTTP Methods](#http-methods)
2. [Method Properties](#method-properties)
3. [HTTP Status Codes](#http-status-codes)
4. [Content Negotiation](#content-negotiation)
5. [Idempotency](#idempotency)
6. [Request and Response Envelopes](#request-and-response-envelopes)
7. [Partial Updates](#partial-updates)
8. [Exercises](#exercises)

---

## HTTP Methods

### GET -- Retrieve a Resource

GET is **safe** (no side effects) and **idempotent** (calling it multiple times has the same effect as calling it once). It must never modify server state.

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

### POST -- Create a Resource

POST is **not safe** and **not idempotent**. Each call may create a new resource.

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

### PUT -- Replace a Resource

PUT is **idempotent** -- sending the same PUT request multiple times produces the same result. It replaces the **entire** resource.

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

### PATCH -- Partial Update

PATCH is for **partial** updates. Only the fields included in the request body are modified.

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

### DELETE -- Remove a Resource

DELETE is **idempotent** -- deleting a resource that is already deleted should not fail.

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

### HEAD and OPTIONS

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

## Method Properties

| Method | Safe | Idempotent | Cacheable | Has Body (Request) | Has Body (Response) |
|--------|------|------------|-----------|-------------------|-------------------|
| GET | Yes | Yes | Yes | No | Yes |
| HEAD | Yes | Yes | Yes | No | No |
| POST | No | No | Rarely | Yes | Yes |
| PUT | No | Yes | No | Yes | Yes |
| PATCH | No | No* | No | Yes | Yes |
| DELETE | No | Yes | No | Rarely | Rarely |
| OPTIONS | Yes | Yes | No | No | Yes |

\* PATCH **can** be idempotent (JSON Merge Patch) but is not required to be.

### Safe vs Idempotent

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

## HTTP Status Codes

### 2xx -- Success

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

### 3xx -- Redirection

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

### 4xx -- Client Errors

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

### 5xx -- Server Errors

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

### Status Code Decision Tree

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

## Content Negotiation

Content negotiation lets clients request data in their preferred format.

### Accept Header

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

### Content-Type for Requests

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

## Idempotency

### Why Idempotency Matters

Network failures happen. When a POST request times out, did the server process it? Without idempotency, the client cannot safely retry.

### Idempotency Keys

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

### Method Idempotency Summary

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

## Request and Response Envelopes

### Consistent Response Structure

Define a standard envelope that wraps all API responses:

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

### Request Envelope (for Batch Operations)

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

## Partial Updates

### JSON Merge Patch (RFC 7396)

The simplest approach: send a JSON object with only the fields to update. Fields set to `null` are removed.

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

A more powerful format that expresses operations (add, remove, replace, move, copy, test).

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

### PUT vs PATCH Decision

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

## Exercises

### Exercise 1: Method Mapping

For each scenario, identify the correct HTTP method and status code:

1. A client wants to fetch a list of products
2. A client wants to create a new order
3. A client wants to update a user's email address only
4. A client wants to replace an entire product record
5. A client wants to delete a comment
6. A client wants to check if a resource exists without downloading it
7. A long-running report generation is requested

### Exercise 2: Build a CRUD API

Implement a complete FastAPI application for a `Task` resource with:
- GET `/api/tasks` -- list with pagination (200)
- GET `/api/tasks/{id}` -- single task (200, 404)
- POST `/api/tasks` -- create (201 + Location header)
- PUT `/api/tasks/{id}` -- full replacement (200, 404)
- PATCH `/api/tasks/{id}` -- partial update using merge patch (200, 404)
- DELETE `/api/tasks/{id}` -- remove (204)

Include proper status codes, validation, and error responses for each endpoint.

### Exercise 3: Implement Content Negotiation

Build an endpoint that serves the same data in three formats:
- `application/json` (default)
- `text/csv`
- `application/xml`

Return 406 Not Acceptable for unsupported formats. Test with:
```bash
http GET localhost:8000/api/users Accept:application/json
http GET localhost:8000/api/users Accept:text/csv
http GET localhost:8000/api/users Accept:application/xml
http GET localhost:8000/api/users Accept:text/yaml
```

### Exercise 4: Idempotent Payments

Build a payment endpoint that:
- Accepts an `Idempotency-Key` header
- Processes the payment on the first call
- Returns the cached response on subsequent calls with the same key
- Rejects requests without an idempotency key (400)
- Expires idempotency keys after 24 hours

### Exercise 5: Partial Update Comparison

Implement both JSON Merge Patch and JSON Patch for an `Article` resource (with fields: title, body, tags, published). Compare:
- How each handles adding a tag to the existing list
- How each handles removing a field
- Which is simpler for your use case and why

---

## Summary

This lesson covered:
1. HTTP methods (GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS) and their semantic properties
2. Status codes organized by class (2xx success, 3xx redirect, 4xx client error, 5xx server error)
3. Content negotiation using Accept and Content-Type headers
4. Idempotency principles and idempotency key patterns for safe retries
5. Consistent request/response envelope design
6. Partial update strategies: JSON Merge Patch vs JSON Patch

---

**Previous**: [URL Design and Naming](03_URL_Design_and_Naming.md) | [Overview](00_Overview.md) | **Next**: [Pagination and Filtering](05_Pagination_and_Filtering.md)

**License**: CC BY-NC 4.0
