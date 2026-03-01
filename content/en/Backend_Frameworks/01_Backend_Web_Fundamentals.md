# 01. Backend Web Fundamentals

**Previous**: [Overview](./00_Overview.md) | **Next**: [FastAPI Basics](./02_FastAPI_Basics.md)

**Difficulty**: ⭐⭐

---

## Learning Objectives

- Explain the HTTP request/response cycle including methods, headers, status codes, and body formats
- Apply REST principles to design resource-oriented APIs with proper CRUD mapping
- Compare WSGI and ASGI server models and identify when each is appropriate
- Design a consistent API versioning strategy for evolving services
- Trace the request lifecycle through a typical web framework from routing to response

---

## Table of Contents

1. [The HTTP Request/Response Cycle](#1-the-http-requestresponse-cycle)
2. [REST Principles](#2-rest-principles)
3. [WSGI vs ASGI](#3-wsgi-vs-asgi)
4. [JSON as the Lingua Franca of APIs](#4-json-as-the-lingua-franca-of-apis)
5. [API Versioning Strategies](#5-api-versioning-strategies)
6. [Request Lifecycle in a Web Framework](#6-request-lifecycle-in-a-web-framework)
7. [Practice Problems](#7-practice-problems)
8. [References](#8-references)

---

## 1. The HTTP Request/Response Cycle

Every interaction between a client (browser, mobile app, CLI tool) and a backend server follows the **HTTP request/response** model. The client sends a request; the server processes it and returns a response.

### Anatomy of an HTTP Request

```
POST /api/users HTTP/1.1          <-- Request line: METHOD PATH VERSION
Host: api.example.com             <-- Headers begin
Content-Type: application/json
Authorization: Bearer eyJhbG...
Content-Length: 56
                                  <-- Blank line separates headers from body
{"name": "Alice", "email": "alice@example.com"}   <-- Body (optional)
```

### HTTP Methods

| Method | Purpose | Idempotent | Safe | Has Body |
|--------|---------|-----------|------|----------|
| `GET` | Retrieve a resource | Yes | Yes | No |
| `POST` | Create a new resource | No | No | Yes |
| `PUT` | Replace a resource entirely | Yes | No | Yes |
| `PATCH` | Partially update a resource | No | No | Yes |
| `DELETE` | Remove a resource | Yes | No | Optional |
| `HEAD` | Like GET but no body returned | Yes | Yes | No |
| `OPTIONS` | Discover allowed methods (CORS) | Yes | Yes | No |

**Idempotent** means calling the same request multiple times produces the same result. `PUT /users/42` with the same body always yields the same state. `POST /users` may create a duplicate each time.

### Anatomy of an HTTP Response

```
HTTP/1.1 201 Created             <-- Status line: VERSION CODE REASON
Content-Type: application/json
Location: /api/users/42          <-- Where the new resource lives
X-Request-Id: abc-123
                                 <-- Blank line
{"id": 42, "name": "Alice", "email": "alice@example.com"}
```

### Status Code Families

| Range | Category | Common Codes |
|-------|----------|-------------|
| `1xx` | Informational | `101 Switching Protocols` (WebSocket upgrade) |
| `2xx` | Success | `200 OK`, `201 Created`, `204 No Content` |
| `3xx` | Redirection | `301 Moved Permanently`, `304 Not Modified` |
| `4xx` | Client Error | `400 Bad Request`, `401 Unauthorized`, `403 Forbidden`, `404 Not Found`, `422 Unprocessable Entity`, `429 Too Many Requests` |
| `5xx` | Server Error | `500 Internal Server Error`, `502 Bad Gateway`, `503 Service Unavailable` |

A helpful rule: if the client sent a bad request, return `4xx`. If the server failed, return `5xx`. Never return `200` with an error body -- this confuses clients and monitoring tools.

### Request Flow Diagram

```
  Client                           Server
    |                                |
    |  ---- HTTP Request ----------> |
    |       Method + URL             |
    |       Headers                  |
    |       Body (optional)          |
    |                                |
    |                          [ Process ]
    |                          [ Route   ]
    |                          [ Logic   ]
    |                          [ DB call ]
    |                                |
    |  <--- HTTP Response ---------- |
    |       Status Code              |
    |       Headers                  |
    |       Body (optional)          |
    |                                |
```

---

## 2. REST Principles

**REST** (Representational State Transfer) is an architectural style, not a protocol. It provides guidelines for designing web APIs that are predictable, scalable, and easy to consume.

### Core Principles

1. **Resources, not actions**: URLs identify nouns (`/users/42`), not verbs (`/getUser?id=42`)
2. **Statelessness**: Each request carries all information needed to process it. The server stores no client session state between requests.
3. **Uniform interface**: Use standard HTTP methods consistently across all resources.
4. **HATEOAS** (Hypermedia As The Engine Of Application State): Responses include links to related resources. In practice, many APIs skip this.

### CRUD Mapping

| Operation | HTTP Method | URL Pattern | Response Code |
|-----------|------------|-------------|---------------|
| Create | `POST` | `/api/users` | `201 Created` |
| Read (list) | `GET` | `/api/users` | `200 OK` |
| Read (detail) | `GET` | `/api/users/42` | `200 OK` |
| Update (full) | `PUT` | `/api/users/42` | `200 OK` |
| Update (partial) | `PATCH` | `/api/users/42` | `200 OK` |
| Delete | `DELETE` | `/api/users/42` | `204 No Content` |

### Resource Naming Conventions

```
# Good - plural nouns, hierarchical relationships
GET  /api/users/42/orders          # Orders belonging to user 42
GET  /api/users/42/orders/7        # Specific order
POST /api/users/42/orders          # Create order for user 42

# Bad - verbs in URLs, flat structure
GET  /api/getUserOrders?userId=42
POST /api/createOrder
```

### Filtering, Sorting, and Pagination

```
# Filtering with query parameters
GET /api/users?role=admin&status=active

# Sorting (prefix with - for descending)
GET /api/users?sort=-created_at,name

# Pagination (offset-based)
GET /api/users?page=2&per_page=25

# Pagination (cursor-based -- better for large datasets)
GET /api/users?cursor=eyJpZCI6NDJ9&limit=25
```

---

## 3. WSGI vs ASGI

Python web servers need a standard interface between the web server and the application. Two standards exist: **WSGI** (synchronous) and **ASGI** (asynchronous).

### WSGI (Web Server Gateway Interface)

WSGI was defined in PEP 3333 (2003). It handles one request at a time per worker process.

```python
# Minimal WSGI application
# Each call blocks until the response is ready
def application(environ: dict, start_response):
    """WSGI expects a callable that receives the request environment
    and a callback to begin the response."""
    status = "200 OK"
    headers = [("Content-Type", "text/plain")]
    start_response(status, headers)
    return [b"Hello, WSGI!"]
```

**WSGI servers**: Gunicorn, uWSGI, mod_wsgi
**WSGI frameworks**: Flask, Django (traditional mode)

### ASGI (Asynchronous Server Gateway Interface)

ASGI was introduced to support async/await, WebSockets, and long-lived connections. It handles many concurrent connections per worker.

```python
# Minimal ASGI application
# Uses async/await -- can handle thousands of concurrent connections
async def application(scope: dict, receive, send):
    """ASGI uses a three-argument callable:
    scope = connection metadata, receive = incoming messages,
    send = outgoing messages."""
    if scope["type"] == "http":
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain")],
        })
        await send({
            "type": "http.response.body",
            "body": b"Hello, ASGI!",
        })
```

**ASGI servers**: Uvicorn, Hypercorn, Daphne
**ASGI frameworks**: FastAPI, Starlette, Django (with ASGI mode)

### Comparison

```
WSGI (Synchronous)                    ASGI (Asynchronous)
┌──────────────────┐                  ┌──────────────────┐
│  Worker Process   │                  │  Event Loop       │
│                   │                  │                   │
│  Request 1 ████   │ (blocking)       │  Req 1 ██  ██    │ (non-blocking)
│  Request 2   ████ │ (waits)          │  Req 2  ██  ██   │ (interleaved)
│  Request 3     ████                  │  Req 3 ██  ██    │
└──────────────────┘                  │  Req 4  ██  ██   │
                                      └──────────────────┘
Needs N workers for N concurrent       One worker handles many
requests. Good for CPU-bound work.     connections. Good for I/O-bound.
```

### When to Use Each

| Scenario | Recommendation |
|----------|---------------|
| Simple CRUD API | Either works; ASGI is more future-proof |
| High-concurrency I/O (DB, external APIs) | ASGI |
| WebSockets, SSE, long polling | ASGI (WSGI cannot support these) |
| CPU-intensive computation | WSGI with multiple workers, or ASGI with thread pool |
| Legacy application | WSGI if already built with Flask/Django |

---

## 4. JSON as the Lingua Franca of APIs

**JSON** (JavaScript Object Notation) is the dominant data format for web APIs. It is human-readable, language-agnostic, and natively supported in JavaScript.

### JSON Data Types

```json
{
    "string": "hello",
    "integer": 42,
    "float": 3.14,
    "boolean": true,
    "null_value": null,
    "array": [1, 2, 3],
    "nested_object": {
        "key": "value"
    }
}
```

### Python JSON Handling

```python
import json
from datetime import datetime, date
from decimal import Decimal

# Python's json module handles basic types, but datetime and Decimal
# need custom serialization since they aren't JSON primitives
class APIEncoder(json.JSONEncoder):
    """Custom encoder for types that json.dumps() doesn't handle natively."""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()  # ISO 8601 is the standard for dates in APIs
        if isinstance(obj, Decimal):
            return float(obj)  # Trade precision for JSON compatibility
        return super().default(obj)

data = {"created_at": datetime.now(), "price": Decimal("19.99")}
json_string = json.dumps(data, cls=APIEncoder, indent=2)
print(json_string)
# {"created_at": "2025-01-15T14:30:00", "price": 19.99}
```

### Content Negotiation

Clients tell the server what format they accept via the `Accept` header. Servers respond with the chosen format in `Content-Type`:

```
# Client requests JSON
GET /api/users/42 HTTP/1.1
Accept: application/json

# Server responds with JSON
HTTP/1.1 200 OK
Content-Type: application/json; charset=utf-8
```

### Alternatives to JSON

| Format | Pros | Cons | Use Case |
|--------|------|------|----------|
| JSON | Universal, human-readable | Verbose, no schema | General APIs |
| MessagePack | Compact binary JSON | Not human-readable | High-throughput internal |
| Protocol Buffers | Strongly typed, fast | Needs `.proto` files | gRPC microservices |
| XML | Self-describing, schemas | Very verbose | Legacy SOAP, config files |

---

## 5. API Versioning Strategies

APIs evolve. Breaking changes are inevitable. Versioning lets you introduce changes without breaking existing clients.

### Strategy 1: URL Path Versioning

```
GET /api/v1/users/42
GET /api/v2/users/42
```

**Pros**: Explicit, easy to understand, easy to route
**Cons**: URL pollution, harder to deprecate
**Used by**: GitHub, Stripe, Twitter

### Strategy 2: Header Versioning

```
GET /api/users/42
Accept: application/vnd.myapi.v2+json
```

**Pros**: Clean URLs, follows HTTP semantics
**Cons**: Less discoverable, harder to test in browser
**Used by**: GitHub (also supports this)

### Strategy 3: Query Parameter Versioning

```
GET /api/users/42?version=2
```

**Pros**: Easy to add, optional (default to latest)
**Cons**: Easy to forget, pollutes query string
**Used by**: Google APIs, Amazon

### Practical Recommendation

For most projects, **URL path versioning** (`/api/v1/`) is the best default. It is explicit, simple, and works with every HTTP client and caching layer. Only introduce a new version when you have breaking changes.

```python
# FastAPI example: organizing versions with routers
from fastapi import APIRouter, FastAPI

app = FastAPI()

# Each version gets its own router with independent logic
v1_router = APIRouter(prefix="/api/v1")
v2_router = APIRouter(prefix="/api/v2")

@v1_router.get("/users/{user_id}")
async def get_user_v1(user_id: int):
    return {"id": user_id, "name": "Alice"}  # v1: flat response

@v2_router.get("/users/{user_id}")
async def get_user_v2(user_id: int):
    return {"data": {"id": user_id, "name": "Alice"}, "meta": {"version": 2}}

app.include_router(v1_router)
app.include_router(v2_router)
```

---

## 6. Request Lifecycle in a Web Framework

When a request arrives, it passes through a pipeline of components before reaching your business logic.

### The Pipeline

```
Client Request
     │
     ▼
┌─────────────────┐
│  Web Server      │  Uvicorn / Gunicorn receives raw HTTP
│  (ASGI/WSGI)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Middleware       │  Runs BEFORE the handler
│  Stack           │  - CORS headers
│                   │  - Authentication check
│                   │  - Request logging
│                   │  - Rate limiting
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Router          │  Matches URL pattern to handler function
│                   │  /api/users/{id} → get_user(id)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Dependency      │  Resolves dependencies declared by handler
│  Injection       │  - Database session
│                   │  - Current authenticated user
│                   │  - Configuration values
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Request         │  Parses and validates:
│  Validation      │  - Path parameters (type conversion)
│                   │  - Query parameters
│                   │  - Request body (JSON → Pydantic model)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Handler         │  Your business logic runs here
│  (View/Endpoint) │  - Query database
│                   │  - Process data
│                   │  - Return response object
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Response        │  Serializes return value to JSON
│  Serialization   │  Applies response model filtering
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Middleware       │  Runs AFTER the handler
│  (post-process)  │  - Add response headers
│                   │  - Compress response body
│                   │  - Log response status + timing
└────────┬────────┘
         │
         ▼
    HTTP Response
```

### Middleware Example

```python
import time
from fastapi import FastAPI, Request

app = FastAPI()

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Measure how long each request takes and add it as a header.
    This is useful for performance monitoring without modifying
    every endpoint."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    # X- prefix is conventional for custom headers
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response
```

### Error Handling in the Pipeline

When an exception occurs at any stage, the framework catches it and converts it to an HTTP error response:

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    user = await find_user(user_id)
    if user is None:
        # HTTPException short-circuits the pipeline and returns
        # an error response immediately without further processing
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found"
        )
    return user
```

---

## 7. Practice Problems

### Problem 1: Design a REST API

Design the URL structure and HTTP methods for a **blog** application with the following resources: posts, comments, and tags. Include:
- CRUD operations for posts
- Nested comments under posts
- Tag assignment to posts
- Filtering posts by tag
- Pagination for post listings

Write out each endpoint as `METHOD /path -> status_code`.

### Problem 2: Status Code Selection

For each scenario, choose the most appropriate HTTP status code and explain why:

1. User submits a registration form with an invalid email format
2. Server's database connection pool is exhausted
3. User requests `DELETE /api/users/99` but user 99 does not exist
4. User requests a resource but their JWT token has expired
5. A `POST` request succeeds but the created resource will be available later (async processing)

### Problem 3: WSGI vs ASGI Analysis

You are building a chat application that:
- Serves a REST API for user profiles and chat history
- Maintains real-time WebSocket connections for live messaging
- Calls an external translation API for multilingual support

Which server model (WSGI or ASGI) would you choose? Justify your answer by addressing each of the three requirements above.

### Problem 4: Middleware Pipeline

Write a Python function (pseudo-middleware) that:
1. Checks for an `X-API-Key` header
2. Returns `401 Unauthorized` if the key is missing
3. Returns `403 Forbidden` if the key is invalid (not in a predefined set)
4. Logs the request method, path, and response status code
5. Adds an `X-Request-Id` header (UUID) to every response

### Problem 5: JSON Serialization Edge Cases

Given the following Python data, write a custom JSON encoder that handles all types correctly:
```python
data = {
    "id": uuid.UUID("12345678-1234-5678-1234-567812345678"),
    "amount": Decimal("99.95"),
    "created_at": datetime(2025, 6, 15, 10, 30),
    "tags": frozenset({"python", "api"}),
    "metadata": None
}
```

---

## 8. References

- [MDN HTTP Reference](https://developer.mozilla.org/en-US/docs/Web/HTTP)
- [RFC 7231 - HTTP/1.1 Semantics and Content](https://datatracker.ietf.org/doc/html/rfc7231)
- [Roy Fielding's REST Dissertation](https://www.ics.uci.edu/~fielding/pubs/dissertation/rest_arch_style.htm)
- [PEP 3333 - Python Web Server Gateway Interface](https://peps.python.org/pep-3333/)
- [ASGI Specification](https://asgi.readthedocs.io/en/latest/specs/main.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [JSON Specification (RFC 8259)](https://datatracker.ietf.org/doc/html/rfc8259)

---

**Previous**: [Overview](./00_Overview.md) | **Next**: [FastAPI Basics](./02_FastAPI_Basics.md)
