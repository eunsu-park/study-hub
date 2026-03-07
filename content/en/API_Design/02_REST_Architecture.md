# Lesson 2: REST Architecture

**Previous**: [API Design Fundamentals](01_API_Design_Fundamentals.md) | [Overview](00_Overview.md) | **Next**: [URL Design and Naming](03_URL_Design_and_Naming.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the six REST architectural constraints and their practical implications
2. Classify APIs using the Richardson Maturity Model (levels 0-3)
3. Model domain concepts as REST resources and sub-resources
4. Implement HATEOAS links in API responses
5. Design stateless, cacheable endpoints that scale horizontally
6. Distinguish between "RESTful" and "REST-inspired" API designs

---

REST (Representational State Transfer) is not a protocol or a library -- it is an **architectural style** defined by Roy Fielding in his 2000 doctoral dissertation. Most APIs that call themselves "RESTful" actually implement only a subset of REST's constraints. Understanding the full model helps you make deliberate trade-offs rather than accidental ones.

> **Analogy:** REST is like the rules of a well-run library. Books (resources) have catalog entries (representations). You use a standard system (HTTP methods) to find, borrow, and return them. The library does not remember what you were looking for last time (statelessness) -- you bring your library card (token) each visit.

## Table of Contents
1. [REST Constraints](#rest-constraints)
2. [Richardson Maturity Model](#richardson-maturity-model)
3. [Resource Modeling](#resource-modeling)
4. [HATEOAS in Practice](#hateoas-in-practice)
5. [Statelessness](#statelessness)
6. [Cacheability](#cacheability)
7. [Uniform Interface](#uniform-interface)
8. [Exercises](#exercises)

---

## REST Constraints

Fielding defined six constraints. An API must satisfy all six to be truly RESTful.

### 1. Client-Server

The client and server are independent. The server does not care about the UI; the client does not care about data storage.

```
Client (React, Mobile, CLI)      Server (FastAPI, Flask)
┌─────────────────────┐          ┌─────────────────────┐
│ UI / Presentation   │◄────────►│ Business Logic      │
│ User Interaction    │  HTTP    │ Data Storage        │
└─────────────────────┘          └─────────────────────┘
```

**Benefit:** Client and server evolve independently. You can rebuild the frontend without touching the backend and vice versa.

### 2. Statelessness

Every request must contain **all** the information the server needs to process it. The server does not store session state between requests.

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

**Benefit:** Any server instance can handle any request. Horizontal scaling is trivial.

### 3. Cacheability

Responses must declare themselves as cacheable or non-cacheable. This allows clients and intermediaries (CDNs, proxies) to reuse responses.

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

### 4. Layered System

The client cannot tell whether it is connected directly to the server or to an intermediary (load balancer, CDN, API gateway). Each layer only knows about the layer it interacts with.

```
Client → CDN → API Gateway → Load Balancer → App Server → Database
         │        │                │
         │        │                └── Health checks, routing
         │        └── Auth, rate limiting, logging
         └── Static content, cached responses
```

### 5. Uniform Interface

The most fundamental constraint. All resources are accessed through a **uniform**, standardized interface with four sub-constraints:

1. **Resource identification** -- Each resource has a unique URI
2. **Manipulation through representations** -- Clients work with JSON/XML representations, not the resource itself
3. **Self-descriptive messages** -- Each message includes enough metadata (Content-Type, status code) to understand it
4. **HATEOAS** -- Hypermedia as the engine of application state (responses include links to related actions)

### 6. Code on Demand (Optional)

The server can send executable code to the client (e.g., JavaScript). This is the only optional constraint and is rarely used in JSON APIs.

---

## Richardson Maturity Model

Leonard Richardson proposed a model with four levels to classify how "RESTful" an API is.

### Level 0: The Swamp of POX (Plain Old XML/JSON)

A single endpoint handles everything. The URL and HTTP method are irrelevant -- the action is embedded in the request body.

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

**Problems:** No caching (everything is POST), no standard discovery, no HTTP semantics.

### Level 1: Resources

Individual URIs for different resources, but still using a single HTTP method (usually POST).

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

**Improvement:** Resources have identity. But HTTP methods are not used semantically.

### Level 2: HTTP Verbs

Resources are identified by URIs **and** manipulated using the correct HTTP methods.

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

**This is where most production APIs land.** Level 2 gives you caching, standard tooling, and clear semantics.

### Level 3: Hypermedia Controls (HATEOAS)

Responses include **links** that tell the client what it can do next. The client does not hard-code URLs -- it follows links.

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

**Benefit:** Clients are decoupled from URL structures. The server can change URLs without breaking clients. This is the most mature level but is rarely fully implemented in practice.

### Maturity Level Summary

| Level | Name | Characteristics | Caching | Discoverability |
|-------|------|----------------|---------|-----------------|
| 0 | Swamp of POX | Single endpoint, action in body | None | None |
| 1 | Resources | Unique URIs per resource | Limited | By URL |
| 2 | HTTP Verbs | Correct methods + status codes | Yes | By convention |
| 3 | Hypermedia | Links in responses (HATEOAS) | Yes | Self-describing |

---

## Resource Modeling

Resource modeling is the process of mapping domain concepts to API resources.

### Identifying Resources

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

### Resource Relationships

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

### Resource Granularity

Design resources at the right level of granularity:

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

## HATEOAS in Practice

HATEOAS (Hypermedia As The Engine Of Application State) means that API responses include links to related resources and available actions.

### Basic Implementation

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

### Collection Links with Pagination

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

## Statelessness

### Why Statelessness Matters

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

### When State Is Necessary

Some operations inherently require state (e.g., multi-step wizards, file uploads). Handle these by:

1. **Storing state in the database**, not in server memory
2. **Using idempotency keys** so retries are safe
3. **Making each step a separate resource** (e.g., `/api/uploads/{upload_id}/parts/{part_num}`)

---

## Cacheability

### Cache-Control Strategies

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

### Cache-Control Directives Summary

| Directive | Meaning |
|-----------|---------|
| `public` | Any cache (CDN, browser) may store the response |
| `private` | Only the end-user's browser may cache |
| `no-store` | Do not cache at all |
| `no-cache` | Cache but revalidate with server before using |
| `max-age=N` | Cache is valid for N seconds |
| `stale-while-revalidate=N` | Serve stale content while revalidating in background |
| `must-revalidate` | Once stale, must revalidate before serving |

---

## Uniform Interface

### Putting It All Together

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

## Exercises

### Exercise 1: Classify API Maturity

Given the following API interactions, identify the Richardson Maturity level of each:

1. `POST /service` with body `{"method": "getUserById", "params": {"id": 1}}`
2. `POST /users/1` with body `{"action": "delete"}`
3. `DELETE /api/users/1` returning `204 No Content`
4. `GET /api/users/1` returning `{"id": 1, "name": "Alice", "_links": {"orders": {"href": "/api/users/1/orders"}}}`

### Exercise 2: Model a Domain as Resources

Design the resource model for a **movie streaming platform**:
- Movies, actors, directors, genres
- User watchlists and viewing history
- Reviews and ratings

Define the URIs, HTTP methods, and example responses for at least 5 endpoints. Include sub-resource relationships and HATEOAS links.

### Exercise 3: Implement Conditional Caching

Build a FastAPI endpoint for a product catalog that supports:
- `ETag` and `If-None-Match` for conditional GET
- `Cache-Control` headers with appropriate `max-age`
- `Last-Modified` and `If-Modified-Since` headers
- Returns `304 Not Modified` when content has not changed

### Exercise 4: Stateless Authentication

Refactor a session-based Flask application to use stateless JWT authentication:
- Issue a JWT on login with `POST /api/auth/login`
- Validate the JWT on every protected endpoint
- Include user role and permissions in the token payload
- Handle token expiration gracefully with a clear error response

### Exercise 5: Build a Level 3 API

Create a FastAPI application for a simple task manager with full HATEOAS:
- `GET /api/tasks` -- list tasks with pagination links
- `POST /api/tasks` -- create a task, return links to the new resource
- `GET /api/tasks/{id}` -- get a task with state-dependent links (e.g., "complete" link only if status is "pending")
- `PATCH /api/tasks/{id}` -- update a task
- `POST /api/tasks/{id}/complete` -- mark as complete, remove "complete" link, add "reopen" link

---

## Summary

This lesson covered:
1. The six REST architectural constraints (client-server, stateless, cacheable, layered, uniform interface, code on demand)
2. The Richardson Maturity Model: from Level 0 (swamp of POX) to Level 3 (hypermedia)
3. Resource modeling: identifying resources, relationships, and granularity
4. HATEOAS: embedding navigational links and state-dependent actions in responses
5. Statelessness: using JWTs and self-contained requests for horizontal scaling
6. Cacheability: Cache-Control directives, ETags, and conditional requests

---

**Previous**: [API Design Fundamentals](01_API_Design_Fundamentals.md) | [Overview](00_Overview.md) | **Next**: [URL Design and Naming](03_URL_Design_and_Naming.md)

**License**: CC BY-NC 4.0
