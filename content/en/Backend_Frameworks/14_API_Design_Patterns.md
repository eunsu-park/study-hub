# 14. API Design Patterns

**Previous**: [Django Advanced](./13_Django_Advanced.md) | **Next**: [Authentication Patterns](./15_Authentication_Patterns.md)

**Difficulty**: ⭐⭐⭐

## Learning Objectives

- Design RESTful APIs using resource-oriented naming and proper HTTP method semantics
- Implement pagination, filtering, sorting, and field selection for scalable endpoints
- Compare API versioning strategies and select the appropriate one for a given project
- Structure consistent error responses following RFC 7807 Problem Details
- Apply rate limiting headers and HATEOAS principles to production APIs

## Table of Contents

Before the patterns reference, read [**Theory & Principles**](#theory--principles) — the mathematical definition of HTTP idempotency, the RFC 7807 Problem Details specification, and how REST/GraphQL/gRPC compare on the same axes.

1. [RESTful Resource Design](#1-restful-resource-design)
2. [HTTP Methods and Idempotency](#2-http-methods-and-idempotency)
3. [Status Code Conventions](#3-status-code-conventions)
4. [Pagination Patterns](#4-pagination-patterns)
5. [Filtering, Sorting, and Field Selection](#5-filtering-sorting-and-field-selection)
6. [API Versioning Strategies](#6-api-versioning-strategies)
7. [HATEOAS and Hypermedia](#7-hateoas-and-hypermedia)
8. [Error Response Format](#8-error-response-format)
9. [Rate Limiting Headers](#9-rate-limiting-headers)
10. [Practice Problems](#10-practice-problems)

---

## Theory & Principles

API design patterns look like a list of conventions, but they are downstream of three deeper ideas. Once you internalize each one, the rest follows mechanically.

- **(A) HTTP method semantics as a mathematical contract** — safe, idempotent, and cacheable are precise properties, not vibes.
- **(B) RFC 7807 Problem Details and machine-readable errors** — the spec for "structured error responses".
- **(C) REST vs RPC vs GraphQL vs gRPC** — four architectural styles compared on the same axes.

### A. The Mathematical Definition of Idempotency

The HTTP/1.1 spec (RFC 9110) defines two properties of methods that constrain how middleboxes (proxies, retries, browsers) treat them. Both are mathematical properties of the *server's behavior*, not the wire format.

#### A.1 Safe methods

A method is **safe** if its semantics are essentially read-only — calling it does not modify server state. Formally:

```
∀ request r: state(server, after r) = state(server, before r)
```

`GET`, `HEAD`, and `OPTIONS` are safe. `POST`, `PUT`, `PATCH`, `DELETE` are not. The contract matters because:

- Browsers and crawlers freely pre-fetch safe methods.
- Caches store safe-method responses without asking.
- Retries on transport errors are unconditionally OK for safe methods.

A "safe" method that secretly logs a write violates the spec — and silently breaks intermediaries. The discipline: a `GET` handler must not modify state. Period.

#### A.2 Idempotent methods

A method is **idempotent** if making the same request multiple times has the same effect on the server as making it once. Formally:

```
state(server, after [r, r, r, ...]) = state(server, after [r])
```

Idempotency does not mean "returns the same response every time" — it means the *server state* converges to the same value. `DELETE /users/42` is idempotent: after one call the user is gone; after a hundred calls the user is still gone (subsequent calls return `404`, but the state is the same).

The HTTP method classification:

| Method | Safe | Idempotent |
|--------|------|-----------|
| GET, HEAD, OPTIONS | yes | yes |
| PUT, DELETE | no | yes |
| POST, PATCH (sometimes) | no | no |

Why it matters: idempotent methods are safe to *retry*. A `PUT` that times out can be re-sent — the worst case is you do the same write twice. A non-idempotent `POST` cannot — retrying might create a second order, charge twice, etc.

#### A.3 Idempotency keys: making POST idempotent at the application layer

Real-world APIs need idempotent payment, order, and message endpoints — but the operations are POST-shaped (creates). The standard solution is the `Idempotency-Key` header (RFC draft, Stripe convention):

```
POST /payments
Idempotency-Key: 7e3b1c8d-4f9a-4...

# First call: server processes, stores result keyed by the key
# Repeat call (same key): server returns the stored result; no new charge
```

The server must remember the key for at least the request retry window (typically 24 h). This pushes the idempotency property into the application layer where the spec can't help.

### B. RFC 7807: Problem Details for HTTP APIs

Before RFC 7807, every API invented its own error response format: `{"error": "..."}`, `{"errors": [...]}`, `{"code": 42, "message": "..."}`. The spec standardizes a small set of fields with well-defined meanings.

#### B.1 The required fields

```json
{
  "type": "https://example.com/probs/out-of-credit",
  "title": "You do not have enough credit.",
  "status": 403,
  "detail": "Your balance is 30, but the requested amount is 50.",
  "instance": "/account/12345/transactions/67890"
}
```

- `type` — A URI identifying the problem class. Two responses with the same `type` mean the same kind of problem. URIs (not opaque codes) so they can be dereferenced for documentation.
- `title` — A short human-readable summary, ideally constant for a given `type`.
- `status` — The HTTP status code, duplicated for clients reading only the body.
- `detail` — The instance-specific human-readable explanation.
- `instance` — A URI identifying *this specific occurrence* of the problem.

The Content-Type is `application/problem+json` (or `+xml`). Any extension fields are allowed — e.g., `balance: 30, requested: 50` for the example above.

#### B.2 Why `type` is a URI, not a code

The temptation is to invent integer codes (`E001`, `E002`). RFC 7807 chose URIs because:

- URIs are globally unique without coordination.
- URIs can be dereferenced — pointing a browser at the URI returns documentation.
- Codes drift across versions; URIs are versioned independently.

In practice, most APIs use a path under their own domain (`https://example.com/probs/out-of-credit`) and serve documentation at that URL.

#### B.3 Validation errors: the structured-array extension

RFC 7807 itself does not specify how to represent *multiple* validation errors. The de facto convention adds an `errors` array:

```json
{
  "type": "https://example.com/probs/validation",
  "title": "Validation failed",
  "status": 422,
  "errors": [
    {"field": "email", "code": "invalid", "message": "Not a valid email address"},
    {"field": "age", "code": "too_low", "message": "Must be at least 18"}
  ]
}
```

This shape lets clients render per-field error messages on a form without parsing free text.

### C. Four Architectural Styles Compared

REST is one of four widely-used styles for backend APIs. Each is a different point in the design space; choosing intelligently means knowing the axes.

#### C.1 The comparison matrix

| Property | REST | RPC (JSON-RPC) | GraphQL | gRPC |
|----------|------|---------------|---------|------|
| Transport | HTTP | HTTP | HTTP | HTTP/2 + Protobuf |
| Discovery | URL conventions | Method directory | Schema | Protobuf .proto |
| Over-fetch | Possible | Method per shape | Client picks | Method per shape |
| Versioning | URL/header | Method name | Schema deprecation | Package version |
| Streaming | SSE / chunked | Bespoke | Subscriptions | First-class |
| Tooling | Browser, cURL | OpenRPC | GraphiQL | grpcurl |
| Browser support | Native | Native | Native | grpc-web (limited) |

#### C.2 REST's strengths and weaknesses

**Strengths:** browser-native (no extra libraries), cacheable at every layer (URL is the cache key), well-understood semantics (every developer knows `GET`/`POST`).

**Weaknesses:** over- and under-fetching (a `GET /users/42` returns the full user; you usually need only the name). N+1 round-trips for nested data. Verb mismatch when the operation does not fit CRUD (`POST /orders/42/cancel/` is a workaround).

#### C.3 GraphQL's pitch

The client sends a query specifying *exactly* the fields it needs:

```graphql
query {
  user(id: 42) {
    name
    orders(last: 5) {
      total
    }
  }
}
```

One round-trip, no over-fetching, no under-fetching. The cost: the server must implement resolvers for every field at every level (with N+1 risk if the resolvers do per-row queries — DataLoader is the standard fix). Caching is harder because the query is the cache key, not the URL.

GraphQL wins when clients are diverse (mobile + web + dashboard) or when data shape varies dramatically per view.

#### C.4 gRPC's pitch

gRPC uses HTTP/2 + Protobuf. The schema lives in `.proto` files; code generation produces typed clients in many languages. Streaming (server-side, client-side, bidirectional) is first-class.

Wins when latency, type safety across languages, and binary efficiency matter — typical for service-to-service communication inside a microservice architecture. Loses for browser clients (gRPC-Web is a partial workaround) and for human-debuggable APIs (Protobuf is binary, not curl-friendly).

#### C.5 The choice principle

Pick the style that minimizes friction for the *primary* consumer:

- **External, public API** → REST (browser-friendly, well-documented, well-cached).
- **Mobile + web app with rich shared backend** → GraphQL (one schema, many shapes).
- **Internal microservices** → gRPC (type-safe, fast, streaming).
- **Action-heavy domains where CRUD is awkward** → JSON-RPC or REST with action endpoints.

### From Theory to the Patterns Below

Each section that follows operationalizes one piece of this framework:

- §1 (RESTful resource design) is the noun-over-verb principle from §C.2's "REST as the default" baseline.
- §2 (HTTP methods and idempotency) is §A.1, §A.2, §A.3 in concrete code.
- §3 (Status code conventions) is the precise use of the 5 status classes from Lesson 01 §A.2.
- §4 (Pagination patterns) is over-fetch defense at the list-endpoint level — offset, cursor, keyset.
- §5 (Filtering, sorting, field selection) is the partial-version of GraphQL's promise (§C.3) in REST-shaped queries.
- §6 (API versioning) is how to evolve the §C.5 contract without breaking clients (URL prefix, Accept header, parameter).
- §7 (HATEOAS) is the original REST §uniform-interface constraint expressed as link relations in responses.
- §8 (Error response format) is RFC 7807 from §B in concrete JSON.
- §9 (Rate limiting headers) is the convention `X-RateLimit-Remaining`, `Retry-After` for client-side back-off.

---

## 1. RESTful Resource Design

REST APIs model resources as nouns, not actions. The URL identifies *what* you are operating on; the HTTP method specifies *how*.

**Bad (verb-oriented):**

```
POST /getUsers
POST /createUser
POST /deleteUser/5
```

**Good (resource-oriented):**

```
GET    /users          # list users
POST   /users          # create a user
GET    /users/5        # retrieve user 5
PUT    /users/5        # replace user 5
PATCH  /users/5        # partially update user 5
DELETE /users/5        # delete user 5
```

### Naming Conventions

- Use **plural nouns** for collections: `/users`, `/posts`, `/comments`
- Use **nested resources** for relationships: `/users/5/posts` (posts by user 5)
- Keep nesting shallow (max 2 levels). Beyond that, use query parameters or top-level resources with filters
- Use **kebab-case** for multi-word resources: `/blog-posts`, `/order-items`
- Avoid file extensions in URLs: use `Accept` headers instead of `/users.json`

### Sub-resources vs. Query Parameters

```
# Sub-resource: strong ownership relationship
GET /users/5/posts          # posts belonging to user 5

# Query parameter: filtering across collections
GET /posts?author_id=5      # posts filtered by author
GET /posts?status=published  # posts filtered by status
```

The sub-resource pattern implies that the child cannot exist without the parent. If a post can exist independently of a user context, prefer the query parameter approach.

---

## 2. HTTP Methods and Idempotency

An operation is **idempotent** if calling it multiple times produces the same result as calling it once. This matters for retry logic and network reliability.

| Method  | Purpose              | Idempotent | Safe | Request Body |
|---------|----------------------|------------|------|--------------|
| GET     | Retrieve resource    | Yes        | Yes  | No           |
| POST    | Create resource      | **No**     | No   | Yes          |
| PUT     | Replace resource     | Yes        | No   | Yes          |
| PATCH   | Partial update       | **No***    | No   | Yes          |
| DELETE  | Remove resource      | Yes        | No   | Optional     |
| HEAD    | Headers only (no body) | Yes      | Yes  | No           |
| OPTIONS | Supported methods    | Yes        | Yes  | No           |

> *PATCH can be idempotent if the patch document specifies absolute values (e.g., `{"status": "active"}`), but it is not guaranteed to be idempotent by specification (e.g., `{"op": "increment", "path": "/count", "value": 1}`).

### PUT vs. PATCH

```python
# PUT: full replacement — client sends the complete resource
# Missing fields are set to null/default
PUT /users/5
{
    "name": "Alice",
    "email": "alice@example.com",
    "bio": "Developer"
}

# PATCH: partial update — client sends only changed fields
PATCH /users/5
{
    "bio": "Senior Developer"
}
```

---

## 3. Status Code Conventions

Use status codes consistently so clients can handle responses programmatically.

### 2xx Success

| Code | Meaning             | When to Use                                  |
|------|---------------------|----------------------------------------------|
| 200  | OK                  | Successful GET, PUT, PATCH                   |
| 201  | Created             | Successful POST that creates a resource      |
| 204  | No Content          | Successful DELETE (no response body)         |

### 4xx Client Errors

| Code | Meaning             | When to Use                                  |
|------|---------------------|----------------------------------------------|
| 400  | Bad Request         | Malformed syntax, invalid input              |
| 401  | Unauthorized        | Missing or invalid authentication            |
| 403  | Forbidden           | Authenticated but lacks permission           |
| 404  | Not Found           | Resource does not exist                      |
| 409  | Conflict            | Duplicate resource, version conflict         |
| 422  | Unprocessable Entity| Valid syntax but semantic errors (validation) |
| 429  | Too Many Requests   | Rate limit exceeded                          |

### 5xx Server Errors

| Code | Meaning             | When to Use                                  |
|------|---------------------|----------------------------------------------|
| 500  | Internal Server Error | Unexpected server failure                  |
| 502  | Bad Gateway         | Upstream service returned invalid response   |
| 503  | Service Unavailable | Server overloaded or in maintenance          |
| 504  | Gateway Timeout     | Upstream service timed out                   |

### FastAPI Example

```python
from fastapi import FastAPI, HTTPException, status

app = FastAPI()

@app.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    if await user_exists(user.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this email already exists",
        )
    return await save_user(user)

@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int):
    user = await get_user(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    await remove_user(user_id)
```

---

## 4. Pagination Patterns

Every endpoint that returns a list must support pagination. Unbounded queries degrade performance and can crash both server and client.

### Offset-Based Pagination

The simplest approach. The client specifies `offset` (or `page`) and `limit`.

```
GET /posts?offset=20&limit=10
```

```json
{
    "data": [...],
    "pagination": {
        "offset": 20,
        "limit": 10,
        "total": 153
    }
}
```

**Pros:** Easy to implement, supports jumping to arbitrary pages.
**Cons:** Expensive for large offsets (database must scan and discard rows), inconsistent results if data changes between pages.

### Cursor-Based Pagination

The server returns an opaque cursor (typically a base64-encoded identifier) that the client passes to fetch the next page.

```
GET /posts?limit=10&cursor=eyJpZCI6IDIwfQ==
```

```json
{
    "data": [...],
    "pagination": {
        "next_cursor": "eyJpZCI6IDMwfQ==",
        "has_more": true
    }
}
```

**Pros:** Consistent results even when data changes, efficient for large datasets.
**Cons:** Cannot jump to arbitrary pages, cursor must be treated as opaque.

### Keyset Pagination

Similar to cursor-based but uses explicit column values instead of opaque tokens.

```
GET /posts?limit=10&created_after=2025-01-15T10:30:00Z&id_after=150
```

**Pros:** Transparent, efficient (uses indexed columns), no scanning.
**Cons:** Requires a unique, sequential sort key; cannot jump to arbitrary pages.

### Comparison

| Aspect              | Offset       | Cursor       | Keyset       |
|---------------------|--------------|--------------|--------------|
| Jump to page N      | Yes          | No           | No           |
| Consistent paging   | No           | Yes          | Yes          |
| Large dataset perf  | Poor         | Good         | Good         |
| Implementation      | Simple       | Moderate     | Moderate     |
| Best for            | Admin panels | Social feeds | Time-series  |

---

## 5. Filtering, Sorting, and Field Selection

### Filtering

Use query parameters named after the resource fields:

```
GET /posts?status=published&author_id=5
GET /posts?created_after=2025-01-01&created_before=2025-06-01
GET /posts?tags=python,fastapi     # comma-separated for IN queries
```

For complex filtering, consider a structured syntax:

```
GET /posts?filter[status]=published&filter[rating][gte]=4
```

### Sorting

Use a `sort` parameter with field names. Prefix with `-` for descending:

```
GET /posts?sort=-created_at          # newest first
GET /posts?sort=author,-created_at   # by author ascending, then newest
```

### Field Selection (Sparse Fieldsets)

Allow clients to request only the fields they need, reducing payload size:

```
GET /posts?fields=id,title,created_at
GET /users/5?fields=name,email
```

### FastAPI Implementation

```python
from fastapi import FastAPI, Query
from typing import Optional

app = FastAPI()

@app.get("/posts")
async def list_posts(
    status: Optional[str] = None,
    author_id: Optional[int] = None,
    sort: str = Query(default="-created_at"),
    fields: Optional[str] = None,
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
):
    query = select(Post)

    # Apply filters
    if status:
        query = query.where(Post.status == status)
    if author_id:
        query = query.where(Post.author_id == author_id)

    # Apply sorting
    for field in sort.split(","):
        if field.startswith("-"):
            query = query.order_by(getattr(Post, field[1:]).desc())
        else:
            query = query.order_by(getattr(Post, field).asc())

    # Apply pagination
    query = query.offset(offset).limit(limit)

    results = await db.execute(query)
    posts = results.scalars().all()

    # Apply field selection
    if fields:
        field_list = fields.split(",")
        posts = [
            {k: v for k, v in post.model_dump().items() if k in field_list}
            for post in posts
        ]

    return {"data": posts, "pagination": {"offset": offset, "limit": limit}}
```

---

## 6. API Versioning Strategies

APIs evolve. Versioning ensures existing clients continue to work while new clients use updated endpoints.

| Strategy         | Example                         | Pros                          | Cons                               |
|------------------|---------------------------------|-------------------------------|-------------------------------------|
| URL path         | `/v1/users`, `/v2/users`       | Explicit, easy to route       | URL pollution, hard to sunset       |
| Query parameter  | `/users?version=2`             | Optional, backward-compatible | Easy to forget, caching issues      |
| Header           | `Accept: application/vnd.api+json;version=2` | Clean URLs       | Hidden, harder to test in browser   |
| Content negotiation | `Accept: application/vnd.myapp.v2+json` | RESTful, precise  | Complex, unfamiliar to many devs    |

### Recommendation

**URL path versioning** is the most common and practical choice for most teams. It is explicit, easy to understand, and straightforward to implement with framework routers.

```python
# FastAPI: URL path versioning with routers
from fastapi import APIRouter

v1_router = APIRouter(prefix="/v1")
v2_router = APIRouter(prefix="/v2")

@v1_router.get("/users")
async def list_users_v1():
    """Returns user list without email (legacy)."""
    return [{"id": u.id, "name": u.name} for u in users]

@v2_router.get("/users")
async def list_users_v2():
    """Returns user list with email and avatar."""
    return [
        {"id": u.id, "name": u.name, "email": u.email, "avatar_url": u.avatar}
        for u in users
    ]

app.include_router(v1_router)
app.include_router(v2_router)
```

---

## 7. HATEOAS and Hypermedia

**HATEOAS** (Hypermedia as the Engine of Application State) means API responses include links that tell the client what actions are available next. The client navigates the API through these links, rather than hard-coding URL patterns.

```json
{
    "id": 42,
    "title": "API Design Patterns",
    "status": "draft",
    "_links": {
        "self": {"href": "/posts/42"},
        "author": {"href": "/users/5"},
        "publish": {"href": "/posts/42/publish", "method": "POST"},
        "comments": {"href": "/posts/42/comments"}
    }
}
```

### Pagination with HATEOAS

```json
{
    "data": [...],
    "_links": {
        "self": {"href": "/posts?page=3&limit=10"},
        "first": {"href": "/posts?page=1&limit=10"},
        "prev": {"href": "/posts?page=2&limit=10"},
        "next": {"href": "/posts?page=4&limit=10"},
        "last": {"href": "/posts?page=15&limit=10"}
    },
    "_meta": {
        "total": 150,
        "page": 3,
        "limit": 10
    }
}
```

In practice, full HATEOAS is rarely implemented. Most production APIs adopt a pragmatic subset: pagination links and a `self` link per resource.

---

## 8. Error Response Format

Consistent error responses are critical for developer experience. **RFC 7807** (Problem Details for HTTP APIs) defines a standard format.

### RFC 7807 Structure

```json
{
    "type": "https://api.example.com/errors/validation",
    "title": "Validation Error",
    "status": 422,
    "detail": "The 'email' field must be a valid email address.",
    "instance": "/users",
    "errors": [
        {
            "field": "email",
            "message": "Not a valid email address",
            "value": "not-an-email"
        }
    ]
}
```

| Field      | Required | Description                                      |
|------------|----------|--------------------------------------------------|
| `type`     | Yes      | URI identifying the error type                   |
| `title`    | Yes      | Short human-readable summary                     |
| `status`   | Yes      | HTTP status code                                 |
| `detail`   | No       | Human-readable explanation of this occurrence     |
| `instance` | No       | URI of the specific request that caused the error |

### FastAPI Exception Handler

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        })

    return JSONResponse(
        status_code=422,
        content={
            "type": "https://api.example.com/errors/validation",
            "title": "Validation Error",
            "status": 422,
            "detail": f"{len(errors)} validation error(s) in request",
            "instance": str(request.url),
            "errors": errors,
        },
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": f"https://api.example.com/errors/{exc.status_code}",
            "title": exc.detail,
            "status": exc.status_code,
            "instance": str(request.url),
        },
    )
```

---

## 9. Rate Limiting Headers

Rate limiting protects your API from abuse and ensures fair usage. Standard headers communicate limits to clients.

### Standard Headers

```
HTTP/1.1 200 OK
X-RateLimit-Limit: 100         # Max requests per window
X-RateLimit-Remaining: 42      # Requests remaining in window
X-RateLimit-Reset: 1706140800  # Unix timestamp when window resets
Retry-After: 30                # Seconds to wait (on 429 response)
```

### FastAPI with slowapi

```python
from fastapi import FastAPI
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter

@app.get("/posts")
@limiter.limit("100/minute")
async def list_posts(request: Request):
    return await fetch_posts()

@app.get("/search")
@limiter.limit("20/minute")
async def search(request: Request, q: str):
    """Search is more expensive, so it has a lower limit."""
    return await perform_search(q)
```

### Rate Limiting Strategies

| Strategy       | Description                              | Use Case                |
|----------------|------------------------------------------|-------------------------|
| Fixed window   | Reset counter every N minutes            | Simple, general purpose |
| Sliding window | Rolling window based on request times    | Smoother rate limiting  |
| Token bucket   | Tokens refill at a constant rate         | Allows short bursts     |
| Leaky bucket   | Requests processed at a constant rate    | Strict rate enforcement |

---

## 10. Practice Problems

### Problem 1: Resource Modeling

Design the URL structure for a university course management system with the following entities: departments, courses, semesters, enrollments, and grades. Write out all the endpoints (method + URL) and specify which status codes each should return.

### Problem 2: Pagination Implementation

Implement cursor-based pagination for a `/comments` endpoint using FastAPI and SQLAlchemy. The cursor should encode the comment's `created_at` and `id` fields. Include proper response formatting with `next_cursor` and `has_more`.

### Problem 3: Error Handler

Create a comprehensive error handling system for a FastAPI application that:
- Returns RFC 7807-compliant error responses
- Handles validation errors, not-found errors, and permission errors
- Includes a custom exception class for business logic errors (e.g., "InsufficientBalance")
- Logs errors with appropriate severity levels

### Problem 4: Versioning Migration

You have an existing `/v1/products` endpoint that returns `{"name": "Widget", "price": 9.99}`. In v2, you need to split `price` into `price.amount` and `price.currency`. Design both the v2 response format and a migration strategy that allows v1 clients to continue working. Implement the solution in FastAPI with shared business logic.

### Problem 5: Rate Limiter Design

Design a rate limiting middleware that supports per-user limits (authenticated via JWT) with different tiers:
- Free tier: 60 requests/minute
- Pro tier: 600 requests/minute
- Enterprise tier: 6000 requests/minute

Implement it using a Redis-backed sliding window algorithm. Include the standard rate limit response headers.

---

## References

- Fielding, R. (2000). *Architectural Styles and the Design of Network-based Software Architectures* (Doctoral dissertation). Chapter 5: REST.
- [RFC 7807: Problem Details for HTTP APIs](https://tools.ietf.org/html/rfc7807)
- [RFC 6585: Additional HTTP Status Codes (429)](https://tools.ietf.org/html/rfc6585)
- [Microsoft REST API Guidelines](https://github.com/microsoft/api-guidelines)
- [JSON:API Specification](https://jsonapi.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Previous**: [Django Advanced](./13_Django_Advanced.md) | **Next**: [Authentication Patterns](./15_Authentication_Patterns.md)
