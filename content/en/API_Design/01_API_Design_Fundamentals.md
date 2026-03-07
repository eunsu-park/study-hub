# Lesson 1: API Design Fundamentals

**Previous**: - | [Overview](00_Overview.md) | **Next**: [REST Architecture](02_REST_Architecture.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define what an API is and explain why API design matters
2. Compare and contrast major API paradigms (REST, RPC, GraphQL, event-driven)
3. Articulate core design principles: consistency, simplicity, and discoverability
4. Distinguish between contract-first and code-first design approaches
5. Evaluate APIs as products and identify the qualities of a great developer experience

---

APIs are the connective tissue of modern software. Every mobile app, single-page application, IoT device, and microservice communicates through APIs. Yet a poorly designed API creates friction that compounds over time -- confusing names, inconsistent conventions, missing error details, and breaking changes drive away consumers and slow down teams. Learning to design APIs well is one of the highest-leverage skills a backend developer can acquire.

> **Analogy:** An API is a restaurant menu. The kitchen (server) can cook many things, but the menu (API) defines what guests (clients) can order, how they order it, and what they should expect to receive. A clear, well-organized menu makes for happy guests; a confusing one sends them to a different restaurant.

## Table of Contents
1. [What Is an API?](#what-is-an-api)
2. [Why API Design Matters](#why-api-design-matters)
3. [API Paradigms](#api-paradigms)
4. [Core Design Principles](#core-design-principles)
5. [Contract-First vs Code-First](#contract-first-vs-code-first)
6. [API as a Product](#api-as-a-product)
7. [Choosing the Right Paradigm](#choosing-the-right-paradigm)
8. [Exercises](#exercises)

---

## What Is an API?

An **Application Programming Interface** is a contract that defines how two pieces of software communicate. In the context of web development, an API typically refers to an HTTP-based interface that accepts requests and returns structured responses (usually JSON).

### Anatomy of an API Call

```
Client                          Server
  |                               |
  |  POST /api/orders             |
  |  Content-Type: application/json
  |  { "item": "widget", "qty": 3 }
  |  ─────────────────────────►   |
  |                               |  (validate, process, persist)
  |  201 Created                  |
  |  Location: /api/orders/42     |
  |  { "id": 42, "status": "pending" }
  |  ◄─────────────────────────   |
```

### Key Terminology

| Term | Definition |
|------|-----------|
| **Endpoint** | A specific URL path that accepts requests (e.g., `/api/users`) |
| **Resource** | A domain object exposed through the API (e.g., User, Order) |
| **Method** | The HTTP verb indicating the action (GET, POST, PUT, DELETE) |
| **Payload** | The data sent in the request or response body |
| **Header** | Metadata attached to requests/responses (auth tokens, content type) |
| **Status Code** | A numeric code indicating the result (200 OK, 404 Not Found) |

---

## Why API Design Matters

### The Cost of Poor Design

Poor API design creates compounding costs:

1. **Integration time** -- Developers spend hours reading inconsistent docs and reverse-engineering behavior
2. **Support burden** -- Confusing APIs generate more support tickets
3. **Breaking changes** -- Poorly planned APIs require frequent, disruptive version bumps
4. **Security holes** -- Inconsistent auth patterns leave gaps attackers exploit

### The Value of Good Design

Well-designed APIs deliver measurable benefits:

- **Faster integration** -- Consistent patterns let developers predict behavior
- **Lower support costs** -- Self-describing responses reduce questions
- **Longer lifespan** -- Thoughtful versioning and extensibility defer rewrites
- **Ecosystem growth** -- Delightful DX attracts third-party integrations

### Real-World Example: Stripe

Stripe is frequently cited as the gold standard for API design:

```python
# Stripe's API is predictable and consistent:
# - Resources are nouns: /v1/customers, /v1/charges, /v1/invoices
# - CRUD maps to HTTP methods: GET (read), POST (create), DELETE (remove)
# - Errors follow a consistent structure
# - Every object has an "id" and "object" field
# - List endpoints always return {"data": [...], "has_more": bool}
```

---

## API Paradigms

### 1. REST (Representational State Transfer)

REST models the API as a collection of **resources** accessed via standard HTTP methods.

```python
from fastapi import FastAPI

app = FastAPI()

# REST: Resources are nouns, methods are verbs
@app.get("/api/books")
async def list_books():
    """List all books (collection resource)."""
    return {"data": [{"id": 1, "title": "API Design Patterns"}]}

@app.get("/api/books/{book_id}")
async def get_book(book_id: int):
    """Get a single book (singleton resource)."""
    return {"id": book_id, "title": "API Design Patterns"}

@app.post("/api/books", status_code=201)
async def create_book(title: str):
    """Create a new book."""
    return {"id": 2, "title": title}
```

**Strengths:** Widely understood, cacheable, stateless, tooling-rich.
**Weaknesses:** Over-fetching/under-fetching, multiple round trips for complex queries.

### 2. RPC (Remote Procedure Call)

RPC exposes **actions** (verbs) rather than resources. The client calls a named procedure on the server.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TransferRequest(BaseModel):
    from_account: str
    to_account: str
    amount: float

# RPC: Endpoints are actions/verbs
@app.post("/api/transfer_funds")
async def transfer_funds(req: TransferRequest):
    """Execute a fund transfer between accounts."""
    return {
        "success": True,
        "transaction_id": "txn_abc123",
        "message": f"Transferred ${req.amount}"
    }

@app.post("/api/send_notification")
async def send_notification(user_id: str, message: str):
    """Send a push notification to a user."""
    return {"delivered": True}
```

**Strengths:** Natural for action-oriented operations, simple function-call semantics.
**Weaknesses:** No standard for discovery, easy to create inconsistent interfaces, less cacheable.

### 3. GraphQL

GraphQL lets clients specify **exactly** which fields they need in a single request.

```python
# GraphQL query -- client asks for precisely what it needs
"""
query {
  book(id: 1) {
    title
    author {
      name
    }
    reviews(limit: 5) {
      rating
      comment
    }
  }
}
"""

# Server schema definition (Strawberry library for Python)
import strawberry

@strawberry.type
class Author:
    name: str

@strawberry.type
class Review:
    rating: int
    comment: str

@strawberry.type
class Book:
    title: str
    author: Author
    reviews: list[Review]

@strawberry.type
class Query:
    @strawberry.field
    def book(self, id: int) -> Book:
        # Resolve book from database
        return Book(
            title="API Design Patterns",
            author=Author(name="JJ Geewax"),
            reviews=[Review(rating=5, comment="Excellent")]
        )
```

**Strengths:** No over-fetching, single endpoint, strong typing, self-documenting schema.
**Weaknesses:** Complexity (caching, authorization per field), N+1 query risk, learning curve.

### 4. Event-Driven (Webhooks / Async APIs)

Event-driven APIs push data to consumers when something happens, rather than waiting for a poll.

```python
from fastapi import FastAPI, Request
import hmac
import hashlib

app = FastAPI()

# --- Provider side: sending a webhook ---
async def send_webhook(event: str, payload: dict, target_url: str):
    """Deliver an event to a subscriber's webhook endpoint."""
    import httpx
    body = {"event": event, "data": payload}
    signature = hmac.new(
        b"webhook_secret", str(body).encode(), hashlib.sha256
    ).hexdigest()
    async with httpx.AsyncClient() as client:
        await client.post(
            target_url,
            json=body,
            headers={"X-Signature": signature}
        )

# --- Consumer side: receiving a webhook ---
@app.post("/webhooks/orders")
async def handle_order_webhook(request: Request):
    """Receive and process an order event from a provider."""
    body = await request.json()
    event = body["event"]
    if event == "order.completed":
        # Process the completed order
        order_id = body["data"]["order_id"]
        return {"received": True, "order_id": order_id}
    return {"received": True, "ignored": True}
```

**Strengths:** Real-time, decoupled, reduces polling overhead.
**Weaknesses:** Delivery reliability, debugging difficulty, requires idempotent consumers.

---

## Core Design Principles

### 1. Consistency

Every endpoint should follow the same conventions for naming, casing, pagination, errors, and authentication. Consistency reduces cognitive load.

```python
# CONSISTENT: Same patterns everywhere
# - Plural nouns for collections
# - snake_case for JSON fields
# - Same pagination structure
# - Same error format

# GET /api/users         -> {"data": [...], "meta": {"total": 100}}
# GET /api/orders        -> {"data": [...], "meta": {"total": 50}}
# GET /api/products      -> {"data": [...], "meta": {"total": 200}}

# INCONSISTENT: Avoid this
# GET /api/users         -> {"users": [...]}
# GET /api/getOrders     -> {"orderList": [...]}
# GET /api/product/list  -> [...]
```

### 2. Simplicity

A good API makes simple things easy and complex things possible. Start with the minimal surface area and expand based on real demand.

```python
# Simple: One obvious way to create a user
@app.post("/api/users", status_code=201)
async def create_user(name: str, email: str):
    return {"id": 1, "name": name, "email": email}

# Avoid: Multiple endpoints doing the same thing
# POST /api/users/create      -- redundant
# POST /api/users/new          -- redundant
# POST /api/create_user         -- RPC style mixed with REST
```

### 3. Discoverability

Consumers should be able to explore the API without reading every page of documentation. Provide links, schemas, and self-describing responses.

```python
@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    return {
        "id": user_id,
        "name": "Alice",
        "email": "alice@example.com",
        "_links": {
            "self": {"href": f"/api/users/{user_id}"},
            "orders": {"href": f"/api/users/{user_id}/orders"},
            "profile": {"href": f"/api/users/{user_id}/profile"},
        }
    }
```

### 4. Predictability

Given knowledge of one endpoint, a developer should be able to guess the shape of another. Use uniform patterns:

```python
# If GET /api/users returns:
# {"data": [{"id": 1, "name": "Alice"}], "meta": {"total": 1}}

# Then GET /api/orders should return:
# {"data": [{"id": 10, "total": 99.99}], "meta": {"total": 1}}

# Same envelope, same meta, same pagination keys.
```

### 5. Robustness (Postel's Law)

> "Be liberal in what you accept, and conservative in what you send."

```python
from pydantic import BaseModel, field_validator

class CreateUserRequest(BaseModel):
    name: str
    email: str
    role: str = "member"  # sensible default

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower()  # accept "  Alice@Example.COM  "
```

---

## Contract-First vs Code-First

### Contract-First (Design-First)

Write the API specification (OpenAPI/Swagger) **before** writing any code. Teams review the contract, agree on it, then implement.

```yaml
# openapi.yaml -- written before any Python code
openapi: "3.1.0"
info:
  title: Bookstore API
  version: "1.0.0"
paths:
  /api/books:
    get:
      summary: List all books
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
      responses:
        "200":
          description: A paginated list of books
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: "#/components/schemas/Book"
                  meta:
                    $ref: "#/components/schemas/PaginationMeta"

components:
  schemas:
    Book:
      type: object
      properties:
        id:
          type: integer
        title:
          type: string
        author:
          type: string
      required: [id, title, author]

    PaginationMeta:
      type: object
      properties:
        page:
          type: integer
        per_page:
          type: integer
        total:
          type: integer
```

**Advantages:**
- Frontend and backend can work in parallel
- API is reviewed as a design artifact before implementation
- Client SDKs can be generated immediately from the spec
- Forces deliberate design decisions

**Disadvantages:**
- Spec can drift from implementation if not enforced
- Slower to start for small projects

### Code-First

Write the implementation, then generate the specification from code annotations.

```python
from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI(title="Bookstore API", version="1.0.0")

class Book(BaseModel):
    id: int
    title: str
    author: str

class BookList(BaseModel):
    data: list[Book]
    meta: dict

@app.get("/api/books", response_model=BookList)
async def list_books(page: int = Query(default=1, ge=1)):
    """List all books with pagination."""
    # FastAPI auto-generates OpenAPI spec from this code
    return BookList(
        data=[Book(id=1, title="API Design Patterns", author="JJ Geewax")],
        meta={"page": page, "per_page": 20, "total": 1}
    )

# Access the generated spec at GET /openapi.json
# Access Swagger UI at GET /docs
# Access ReDoc at GET /redoc
```

**Advantages:**
- Spec is always in sync with code (single source of truth)
- Faster to get started
- Leverages framework features (FastAPI, Django REST Framework)

**Disadvantages:**
- Design decisions happen during coding, not during review
- Harder to get early stakeholder feedback

### Which to Choose?

| Factor | Contract-First | Code-First |
|--------|---------------|------------|
| Team size | Large / cross-functional | Small / single team |
| API audience | Public / external | Internal / private |
| Iteration speed | Slower initial, faster long-term | Faster initial |
| Spec accuracy | Requires enforcement tooling | Automatic |
| Recommended for | Platform APIs, partner integrations | Microservices, MVPs |

---

## API as a Product

When you treat your API as a product, you focus on the developer experience (DX) of your consumers.

### Product Thinking for APIs

```
Traditional Thinking          Product Thinking
─────────────────────         ──────────────────
"Ship endpoints"              "Solve developer problems"
"Document the API"            "Enable self-service onboarding"
"Fix bugs"                    "Measure time-to-first-call"
"Add features"                "Understand use cases, then add features"
```

### Key DX Metrics

1. **Time to First Call (TTFC)** -- How long until a developer makes their first successful API call?
2. **Time to Working App (TTWA)** -- How long from signup to a functioning integration?
3. **Error Resolution Time** -- How quickly can a developer understand and fix an error?
4. **Self-Service Rate** -- What percentage of developers integrate without contacting support?

### Building a Great DX

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Bookstore API",
    description="A simple API for managing a bookstore inventory.",
    version="1.0.0",
    docs_url="/docs",           # Swagger UI
    redoc_url="/redoc",         # ReDoc
    openapi_url="/openapi.json" # Machine-readable spec
)

class BookCreate(BaseModel):
    """Request body for creating a book."""
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        examples=["The Pragmatic Programmer"],
        description="The book's title. Must be unique within the store."
    )
    author: str = Field(
        ...,
        min_length=1,
        examples=["David Thomas, Andrew Hunt"],
        description="The author's full name."
    )
    isbn: str | None = Field(
        default=None,
        pattern=r"^\d{13}$",
        examples=["9780135957059"],
        description="13-digit ISBN. Optional but recommended."
    )

class BookResponse(BaseModel):
    """A book in the bookstore inventory."""
    id: int
    title: str
    author: str
    isbn: str | None
    created_at: str

    model_config = {"json_schema_extra": {
        "examples": [{
            "id": 1,
            "title": "The Pragmatic Programmer",
            "author": "David Thomas, Andrew Hunt",
            "isbn": "9780135957059",
            "created_at": "2025-01-15T10:30:00Z"
        }]
    }}

@app.post(
    "/api/books",
    response_model=BookResponse,
    status_code=201,
    summary="Create a book",
    responses={
        409: {"description": "A book with this ISBN already exists"},
        422: {"description": "Validation error in request body"},
    }
)
async def create_book(book: BookCreate):
    """
    Add a new book to the inventory.

    - **title**: Must be unique within the store
    - **author**: Author's full name
    - **isbn**: Optional 13-digit ISBN for deduplication
    """
    return BookResponse(
        id=1,
        title=book.title,
        author=book.author,
        isbn=book.isbn,
        created_at="2025-01-15T10:30:00Z"
    )
```

---

## Choosing the Right Paradigm

There is no single "best" paradigm. The choice depends on your use case:

```
Use Case                        Recommended Paradigm
──────────────────────────────  ──────────────────────
CRUD on domain objects          REST
Complex queries, mobile apps    GraphQL
Server-to-server actions        RPC (gRPC for performance)
Real-time notifications         Event-driven (webhooks/SSE)
File uploads                    REST (multipart) or tus protocol
Streaming data                  gRPC streaming or WebSockets
Internal microservices          gRPC or REST
Public developer platform       REST (with OpenAPI)
```

### Decision Framework

```python
def choose_paradigm(
    audience: str,       # "public" | "internal" | "partner"
    data_shape: str,     # "simple_crud" | "complex_graph" | "actions"
    performance: str,    # "standard" | "high_throughput" | "real_time"
) -> str:
    """Heuristic for choosing an API paradigm."""
    if performance == "real_time":
        return "event-driven (WebSockets, SSE, webhooks)"
    if performance == "high_throughput" and audience == "internal":
        return "gRPC"
    if data_shape == "complex_graph":
        return "GraphQL"
    if data_shape == "actions":
        return "RPC"
    return "REST"  # default for most cases
```

---

## Exercises

### Exercise 1: Classify Existing APIs

Visit three public APIs (e.g., GitHub, Stripe, Twitter/X) and classify each as REST, RPC, GraphQL, or hybrid. For each, identify:
- How resources/actions are named
- What HTTP methods are used
- How errors are structured
- Whether they follow REST constraints strictly

### Exercise 2: Design an API Contract

Choose a domain (e.g., a library system, a food delivery app, a task manager) and write a contract-first OpenAPI specification with at least:
- 3 resources (e.g., Books, Authors, Loans)
- CRUD endpoints for each resource
- Consistent naming and response envelopes
- Error response schemas

### Exercise 3: Compare Paradigms

For a social media application with users, posts, comments, and likes, sketch out:
1. A REST API (resource endpoints)
2. An RPC API (action endpoints)
3. A GraphQL schema (types and queries)

Compare the number of requests needed for a "user profile page" that shows the user, their 10 latest posts, and the comment count on each post.

### Exercise 4: Evaluate DX

Pick a public API you have used and evaluate its developer experience:
- How long did it take to make your first successful call?
- Was the documentation clear?
- Were error messages helpful?
- What would you improve?

---

## Summary

This lesson covered:
1. What APIs are and why design quality matters
2. Four major paradigms: REST, RPC, GraphQL, and event-driven
3. Core design principles: consistency, simplicity, discoverability, predictability, robustness
4. Contract-first vs code-first design approaches
5. Treating APIs as products with measurable DX metrics
6. A decision framework for choosing the right paradigm

---

**Previous**: - | [Overview](00_Overview.md) | **Next**: [REST Architecture](02_REST_Architecture.md)

**License**: CC BY-NC 4.0
