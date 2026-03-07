# 10. API Documentation

**Previous**: [Rate Limiting and Throttling](./09_Rate_Limiting_and_Throttling.md) | **Next**: [API Testing and Validation](./11_API_Testing_and_Validation.md)

**Difficulty**: ⭐⭐

---

## Learning Objectives

- Write OpenAPI 3.x specifications that fully describe API endpoints, schemas, and authentication
- Configure Swagger UI and ReDoc for interactive API documentation
- Auto-generate documentation from code using FastAPI's built-in OpenAPI support
- Apply API description best practices including clear summaries, examples, and error documentation
- Create useful request and response examples that serve as both documentation and test fixtures
- Maintain documentation accuracy as the API evolves

---

## Table of Contents

1. [Why Documentation Matters](#1-why-documentation-matters)
2. [OpenAPI 3.x Specification](#2-openapi-3x-specification)
3. [Swagger UI and ReDoc](#3-swagger-ui-and-redoc)
4. [Auto-Generation with FastAPI](#4-auto-generation-with-fastapi)
5. [Description Best Practices](#5-description-best-practices)
6. [Examples in Documentation](#6-examples-in-documentation)
7. [Documenting Authentication](#7-documenting-authentication)
8. [Documenting Errors](#8-documenting-errors)
9. [Keeping Documentation in Sync](#9-keeping-documentation-in-sync)
10. [Exercises](#10-exercises)
11. [References](#11-references)

---

## 1. Why Documentation Matters

API documentation is the primary interface between your API and its consumers. Poor documentation leads to:

- Higher support burden (developers cannot self-serve)
- Slower adoption and integration
- Misuse of endpoints (wrong parameters, missing headers)
- Frustration and abandonment

Good documentation answers three questions for every endpoint:

1. **What does this endpoint do?** (summary and description)
2. **What do I send?** (request parameters, headers, body)
3. **What do I get back?** (response schema, status codes, examples)

### Documentation-First vs. Code-First

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| Design-first | Write OpenAPI spec before coding | Forces good design, enables parallel work | Spec can drift from implementation |
| Code-first | Generate spec from annotated code | Always accurate, lower maintenance | May lead to design shortcuts |

FastAPI excels at the code-first approach: your Pydantic models and type hints automatically generate accurate OpenAPI documentation.

---

## 2. OpenAPI 3.x Specification

The **OpenAPI Specification** (formerly Swagger) is the industry standard for describing REST APIs. It is a machine-readable format (YAML or JSON) that describes endpoints, parameters, request/response schemas, and authentication.

### Structure Overview

```yaml
openapi: 3.1.0
info:
  title: Bookstore API
  description: API for managing books, authors, and reviews
  version: 1.0.0
  contact:
    name: API Support
    email: api-support@example.com
  license:
    name: MIT

servers:
  - url: https://api.bookstore.example.com/v1
    description: Production
  - url: https://staging-api.bookstore.example.com/v1
    description: Staging

paths:
  /books:
    get:
      summary: List all books
      operationId: listBooks
      tags:
        - Books
      parameters:
        - name: genre
          in: query
          description: Filter by genre
          schema:
            type: string
            enum: [fiction, non-fiction, science, history]
        - name: limit
          in: query
          description: Maximum number of results
          schema:
            type: integer
            default: 20
            minimum: 1
            maximum: 100
      responses:
        "200":
          description: A paginated list of books
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/BookList"
        "400":
          $ref: "#/components/responses/BadRequest"

    post:
      summary: Create a new book
      operationId: createBook
      tags:
        - Books
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/BookCreate"
            example:
              title: "API Design Patterns"
              author_id: 42
              isbn: "978-1617295850"
              genre: "non-fiction"
              price: 49.99
      responses:
        "201":
          description: Book created successfully
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/Book"
        "422":
          $ref: "#/components/responses/ValidationError"

  /books/{book_id}:
    get:
      summary: Get a book by ID
      operationId: getBook
      tags:
        - Books
      parameters:
        - name: book_id
          in: path
          required: true
          schema:
            type: integer
      responses:
        "200":
          description: Book details
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/Book"
        "404":
          $ref: "#/components/responses/NotFound"

components:
  schemas:
    Book:
      type: object
      required: [id, title, author_id, isbn]
      properties:
        id:
          type: integer
          description: Unique book identifier
          example: 1
        title:
          type: string
          description: Book title
          example: "API Design Patterns"
        author_id:
          type: integer
          description: ID of the author
          example: 42
        isbn:
          type: string
          pattern: "^978-\\d{10}$"
          example: "978-1617295850"
        genre:
          type: string
          enum: [fiction, non-fiction, science, history]
        price:
          type: number
          format: float
          minimum: 0
          example: 49.99
        created_at:
          type: string
          format: date-time

    BookCreate:
      type: object
      required: [title, author_id, isbn]
      properties:
        title:
          type: string
          minLength: 1
          maxLength: 500
        author_id:
          type: integer
        isbn:
          type: string
        genre:
          type: string
        price:
          type: number
          minimum: 0

    BookList:
      type: object
      properties:
        data:
          type: array
          items:
            $ref: "#/components/schemas/Book"
        pagination:
          $ref: "#/components/schemas/Pagination"

    Pagination:
      type: object
      properties:
        total:
          type: integer
        offset:
          type: integer
        limit:
          type: integer

  responses:
    BadRequest:
      description: Invalid request parameters
      content:
        application/json:
          schema:
            $ref: "#/components/schemas/Error"
    NotFound:
      description: Resource not found
      content:
        application/json:
          schema:
            $ref: "#/components/schemas/Error"
    ValidationError:
      description: Request body validation failed
      content:
        application/json:
          schema:
            $ref: "#/components/schemas/Error"

    Error:
      type: object
      properties:
        type:
          type: string
        title:
          type: string
        status:
          type: integer
        detail:
          type: string

  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| `paths` | Endpoint definitions with HTTP methods |
| `components/schemas` | Reusable data models (JSON Schema) |
| `$ref` | References to reusable components |
| `operationId` | Unique identifier for each operation (used by code generators) |
| `tags` | Groups for organizing endpoints |
| `securitySchemes` | Authentication mechanism definitions |

---

## 3. Swagger UI and ReDoc

Two popular renderers transform OpenAPI specs into interactive documentation.

### Swagger UI

Swagger UI provides an interactive interface where developers can read documentation and execute API calls directly from the browser.

FastAPI includes Swagger UI by default at `/docs`:

```python
from fastapi import FastAPI

app = FastAPI(
    title="Bookstore API",
    description="API for managing books, authors, and reviews",
    version="1.0.0",
    docs_url="/docs",          # Swagger UI (default)
    redoc_url="/redoc",        # ReDoc (default)
    openapi_url="/openapi.json",  # Raw OpenAPI spec
)
```

### ReDoc

ReDoc provides a clean, three-panel documentation layout. It is read-only (no "Try it out" feature) but excels at long-form documentation with markdown support.

FastAPI includes ReDoc by default at `/redoc`.

### Customizing Documentation UI

```python
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


app = FastAPI()


def custom_openapi():
    """Override the default OpenAPI schema to add custom metadata."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Bookstore API",
        version="1.0.0",
        description="""
## Overview

The Bookstore API allows you to manage books, authors, and reviews.

### Authentication

All write operations require a Bearer token. Obtain a token via `POST /auth/token`.

### Rate Limits

| Tier | Requests/min | Daily Limit |
|------|-------------|-------------|
| Free | 60 | 1,000 |
| Pro  | 600 | 50,000 |
        """,
        routes=app.routes,
    )

    # Add custom logo for ReDoc
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/logo.png"
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
```

### Disabling Docs in Production

```python
import os

docs_url = "/docs" if os.getenv("ENVIRONMENT") != "production" else None
redoc_url = "/redoc" if os.getenv("ENVIRONMENT") != "production" else None

app = FastAPI(docs_url=docs_url, redoc_url=redoc_url)
```

---

## 4. Auto-Generation with FastAPI

FastAPI automatically generates OpenAPI documentation from your Python code. Every type hint, Pydantic model, and docstring contributes to the generated specification.

### Pydantic Models as Schemas

```python
from pydantic import BaseModel, Field
from datetime import datetime


class BookCreate(BaseModel):
    """Schema for creating a new book.

    This description appears in the OpenAPI schema documentation.
    """

    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The title of the book",
        examples=["API Design Patterns"],
    )
    author_id: int = Field(
        ...,
        gt=0,
        description="ID of the book's author",
        examples=[42],
    )
    isbn: str = Field(
        ...,
        pattern=r"^978-\d{10}$",
        description="ISBN-13 identifier",
        examples=["978-1617295850"],
    )
    genre: str | None = Field(
        default=None,
        description="Book genre category",
        examples=["non-fiction"],
    )
    price: float = Field(
        default=0.0,
        ge=0,
        description="Price in USD",
        examples=[49.99],
    )


class BookResponse(BaseModel):
    """Complete book record returned from the API."""

    id: int
    title: str
    author_id: int
    isbn: str
    genre: str | None = None
    price: float
    created_at: datetime

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": 1,
                    "title": "API Design Patterns",
                    "author_id": 42,
                    "isbn": "978-1617295850",
                    "genre": "non-fiction",
                    "price": 49.99,
                    "created_at": "2025-01-15T10:30:00Z",
                }
            ]
        }
    }
```

### Route Documentation

```python
from fastapi import FastAPI, Query, Path, HTTPException, status

app = FastAPI()


@app.get(
    "/books",
    response_model=list[BookResponse],
    summary="List all books",
    description="Retrieve a paginated list of books with optional filtering by genre.",
    tags=["Books"],
    responses={
        200: {
            "description": "A list of books matching the query parameters",
        },
        400: {
            "description": "Invalid query parameters",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Invalid genre: 'sci-fi'. Must be one of: fiction, non-fiction, science, history"
                    }
                }
            },
        },
    },
)
async def list_books(
    genre: str | None = Query(
        default=None,
        description="Filter books by genre",
        examples=["fiction", "science"],
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Number of books to return",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of books to skip",
    ),
):
    """Retrieve a paginated list of books.

    Supports filtering by genre and pagination via offset/limit.
    Results are sorted by creation date (newest first).
    """
    ...


@app.post(
    "/books",
    response_model=BookResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new book",
    tags=["Books"],
    responses={
        201: {"description": "Book created successfully"},
        409: {
            "description": "Book with this ISBN already exists",
            "content": {
                "application/json": {
                    "example": {"detail": "Book with ISBN 978-1617295850 already exists"}
                }
            },
        },
        422: {"description": "Validation error in request body"},
    },
)
async def create_book(book: BookCreate):
    """Create a new book record.

    The ISBN must be unique. If a book with the same ISBN already exists,
    a 409 Conflict error is returned.
    """
    ...
```

### Tags and Grouping

```python
from fastapi import FastAPI

# Tags organize endpoints into logical groups in the documentation
tags_metadata = [
    {
        "name": "Books",
        "description": "Operations for managing the book catalog.",
    },
    {
        "name": "Authors",
        "description": "Operations for managing author profiles.",
    },
    {
        "name": "Reviews",
        "description": "Operations for managing book reviews. Requires authentication.",
        "externalDocs": {
            "description": "Review guidelines",
            "url": "https://docs.example.com/review-guidelines",
        },
    },
]

app = FastAPI(openapi_tags=tags_metadata)
```

---

## 5. Description Best Practices

### Writing Effective Summaries

The `summary` field is a short label displayed in the endpoint list. The `description` field provides detailed context.

```python
# Bad: vague, does not explain behavior
@app.get("/users", summary="Get users")
async def get_users(): ...

# Good: specific, explains filtering and pagination
@app.get(
    "/users",
    summary="List users with filtering",
    description=(
        "Returns a paginated list of users. Supports filtering by role "
        "and status. Results are sorted by creation date (newest first). "
        "Requires `read:users` scope."
    ),
)
async def list_users(): ...
```

### Documentation Checklist

For every endpoint, document:

1. **Summary**: One-line description (shown in endpoint list)
2. **Description**: What the endpoint does, any side effects, required permissions
3. **Parameters**: Type, constraints, defaults, and examples for every parameter
4. **Request body**: Schema with field descriptions and a complete example
5. **Response codes**: Every possible status code with description and example body
6. **Authentication**: What credentials are needed (if any)
7. **Rate limits**: Applicable rate limit tier (if different from default)

### Markdown in Descriptions

OpenAPI supports Markdown in description fields. Use it for formatting:

```python
@app.post(
    "/orders",
    description="""
Create a new order for the authenticated user.

### Side Effects

- Decrements inventory for each item in the order
- Sends a confirmation email to the user
- Creates a payment intent via Stripe

### Required Permissions

- `create:orders`

### Notes

> Orders with a total above $1,000 require manual approval
> and will be created with status `pending_review`.
    """,
)
async def create_order(order: OrderCreate): ...
```

---

## 6. Examples in Documentation

Examples are the most valuable part of API documentation. Developers often skip the prose and jump straight to examples.

### Request Examples

```python
from pydantic import BaseModel, Field


class OrderCreate(BaseModel):
    items: list[dict] = Field(
        ...,
        description="List of items to order",
    )
    shipping_address: str = Field(
        ...,
        description="Full shipping address",
    )
    notes: str | None = Field(
        default=None,
        description="Optional order notes",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Simple order",
                    "description": "A basic order with one item",
                    "value": {
                        "items": [
                            {"book_id": 1, "quantity": 2}
                        ],
                        "shipping_address": "123 Main St, New York, NY 10001",
                        "notes": None,
                    },
                },
                {
                    "summary": "Multi-item order with notes",
                    "description": "An order with multiple items and special instructions",
                    "value": {
                        "items": [
                            {"book_id": 1, "quantity": 1},
                            {"book_id": 5, "quantity": 3},
                        ],
                        "shipping_address": "456 Oak Ave, San Francisco, CA 94102",
                        "notes": "Please gift-wrap all items",
                    },
                },
            ]
        }
    }
```

### Response Examples

```python
@app.get(
    "/books/{book_id}",
    responses={
        200: {
            "description": "Book found",
            "content": {
                "application/json": {
                    "examples": {
                        "fiction": {
                            "summary": "A fiction book",
                            "value": {
                                "id": 1,
                                "title": "The Great Gatsby",
                                "author_id": 10,
                                "isbn": "978-0743273565",
                                "genre": "fiction",
                                "price": 12.99,
                                "created_at": "2025-01-01T00:00:00Z",
                            },
                        },
                        "technical": {
                            "summary": "A technical book",
                            "value": {
                                "id": 2,
                                "title": "Designing Data-Intensive Applications",
                                "author_id": 20,
                                "isbn": "978-1449373320",
                                "genre": "non-fiction",
                                "price": 45.99,
                                "created_at": "2025-03-15T10:00:00Z",
                            },
                        },
                    }
                }
            },
        },
        404: {
            "description": "Book not found",
            "content": {
                "application/json": {
                    "example": {
                        "type": "https://api.example.com/errors/not-found",
                        "title": "Not Found",
                        "status": 404,
                        "detail": "Book with ID 999 not found",
                    }
                }
            },
        },
    },
)
async def get_book(book_id: int = Path(..., gt=0)):
    ...
```

### cURL Examples in Descriptions

Include cURL examples for developers who prefer the command line:

```python
@app.post(
    "/auth/token",
    description="""
Authenticate and receive an access token.

### Example Request

```bash
curl -X POST https://api.example.com/v1/auth/token \\
  -H "Content-Type: application/x-www-form-urlencoded" \\
  -d "username=alice&password=secret123"
```

### Example Response

```json
{
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "token_type": "bearer",
    "expires_in": 900
}
```
    """,
)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    ...
```

---

## 7. Documenting Authentication

### Security Schemes

```python
from fastapi import FastAPI, Depends, Security
from fastapi.security import (
    HTTPBearer,
    OAuth2PasswordBearer,
    APIKeyHeader,
)

app = FastAPI()

# Bearer token authentication
bearer_scheme = HTTPBearer(
    description="JWT access token obtained from `POST /auth/token`"
)

# OAuth2 with password flow
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/token",
    scopes={
        "read:books": "Read access to books",
        "write:books": "Create and update books",
        "admin": "Full administrative access",
    },
)

# API key authentication
api_key_scheme = APIKeyHeader(
    name="X-API-Key",
    description="API key for server-to-server communication",
)


@app.get(
    "/books",
    summary="List books (requires read:books scope)",
)
async def list_books(token: str = Security(oauth2_scheme, scopes=["read:books"])):
    """This endpoint requires the `read:books` OAuth2 scope."""
    ...


@app.get(
    "/internal/stats",
    summary="Internal statistics (API key required)",
)
async def get_stats(api_key: str = Security(api_key_scheme)):
    """This endpoint requires a valid API key."""
    ...
```

---

## 8. Documenting Errors

Every endpoint should document its possible error responses. This helps client developers write robust error handling.

### Centralized Error Responses

```python
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Standard error response following RFC 7807."""

    type: str = Field(description="URI identifying the error type")
    title: str = Field(description="Short human-readable summary")
    status: int = Field(description="HTTP status code")
    detail: str | None = Field(
        default=None, description="Detailed explanation of the error"
    )
    instance: str | None = Field(
        default=None, description="URI of the request that caused the error"
    )


class ValidationErrorResponse(ErrorResponse):
    """Error response for validation failures (422)."""

    errors: list[dict] = Field(
        description="List of individual field validation errors"
    )


# Reusable response definitions
ERROR_RESPONSES = {
    400: {
        "model": ErrorResponse,
        "description": "Bad request — malformed syntax or invalid parameters",
    },
    401: {
        "model": ErrorResponse,
        "description": "Unauthorized — missing or invalid authentication",
    },
    403: {
        "model": ErrorResponse,
        "description": "Forbidden — insufficient permissions",
    },
    404: {
        "model": ErrorResponse,
        "description": "Not found — the requested resource does not exist",
    },
    422: {
        "model": ValidationErrorResponse,
        "description": "Validation error — request body contains invalid data",
    },
    429: {
        "model": ErrorResponse,
        "description": "Rate limit exceeded — try again later",
    },
}


# Apply common errors to all routes
@app.post(
    "/books",
    responses={
        **ERROR_RESPONSES,
        201: {"description": "Book created"},
        409: {
            "model": ErrorResponse,
            "description": "Conflict — ISBN already exists",
        },
    },
)
async def create_book(book: BookCreate): ...
```

---

## 9. Keeping Documentation in Sync

### Automated Validation

Ensure your documentation matches your implementation by validating the OpenAPI schema in CI:

```python
# tests/test_openapi.py
import json
from openapi_spec_validator import validate

from app.main import app


def test_openapi_schema_is_valid():
    """Verify the generated OpenAPI schema is valid."""
    schema = app.openapi()
    validate(schema)


def test_all_routes_have_descriptions():
    """Every route must have a summary and description."""
    schema = app.openapi()
    for path, methods in schema["paths"].items():
        for method, details in methods.items():
            if method in ("get", "post", "put", "patch", "delete"):
                assert "summary" in details, (
                    f"{method.upper()} {path} is missing a summary"
                )
                assert "description" in details or details.get("summary"), (
                    f"{method.upper()} {path} is missing a description"
                )


def test_all_routes_have_response_examples():
    """Every route should document at least the success response."""
    schema = app.openapi()
    for path, methods in schema["paths"].items():
        for method, details in methods.items():
            if method in ("get", "post", "put", "patch", "delete"):
                responses = details.get("responses", {})
                assert len(responses) > 0, (
                    f"{method.upper()} {path} has no documented responses"
                )


def test_openapi_snapshot():
    """Detect unexpected changes to the API schema.

    Run `pytest --snapshot-update` to update the snapshot
    when changes are intentional.
    """
    schema = app.openapi()
    snapshot_path = "tests/snapshots/openapi.json"

    with open(snapshot_path) as f:
        expected = json.load(f)

    assert schema == expected, (
        "OpenAPI schema has changed. If intentional, update the snapshot."
    )
```

### Exporting the Spec

```python
# scripts/export_openapi.py
"""Export the OpenAPI spec to a file for external tools."""

import json
import yaml
from app.main import app


def export_openapi():
    schema = app.openapi()

    # JSON format
    with open("docs/openapi.json", "w") as f:
        json.dump(schema, f, indent=2)

    # YAML format
    with open("docs/openapi.yaml", "w") as f:
        yaml.dump(schema, f, default_flow_style=False, sort_keys=False)

    print(f"Exported OpenAPI spec: {len(schema['paths'])} paths")


if __name__ == "__main__":
    export_openapi()
```

---

## 10. Exercises

### Exercise 1: Write an OpenAPI Spec

Write a complete OpenAPI 3.1 specification (in YAML) for a **Task Management API** with the following endpoints:

- `GET /tasks` — List tasks with filtering (status, priority, assignee) and pagination
- `POST /tasks` — Create a task (title, description, priority, assignee_id)
- `GET /tasks/{id}` — Get a single task
- `PATCH /tasks/{id}` — Update task fields
- `DELETE /tasks/{id}` — Delete a task
- `POST /tasks/{id}/comments` — Add a comment to a task

Include: schemas for all models, examples for requests and responses, authentication via Bearer token, and error responses for 400, 401, 404, and 422.

### Exercise 2: FastAPI Documentation Enhancement

Given the following FastAPI endpoint, add complete documentation including summary, description, parameter descriptions, response models, response examples (success and error), and tags:

```python
@app.get("/users/{user_id}/orders")
async def get_user_orders(user_id: int, status: str = None, limit: int = 20):
    ...
```

### Exercise 3: API Changelog Generator

Build a Python script that compares two OpenAPI JSON files (old and new versions) and generates a human-readable changelog. The changelog should identify:

- Added endpoints
- Removed endpoints
- Changed request parameters (added, removed, type changed)
- Changed response schemas
- Changed authentication requirements

Format the output as Markdown suitable for a release note.

### Exercise 4: Documentation Linter

Create a pytest test suite that validates your FastAPI application's documentation quality. Check for:

- Every endpoint has a non-empty summary (max 80 characters)
- Every endpoint has a description
- Every path parameter has a description
- Every query parameter has a description and example
- Every POST/PUT/PATCH endpoint has a request body example
- Every response code has a description
- No endpoint uses generic "string" types without format or pattern constraints

### Exercise 5: Interactive Examples

Add a `POST /books/search` endpoint to a FastAPI application that accepts a complex search query body. Provide at least 3 named examples in the OpenAPI schema:

1. Simple title search
2. Advanced search with genre, price range, and date filters
3. Full-text search with pagination and sorting

Ensure all examples render correctly in both Swagger UI and ReDoc.

---

## 11. References

- [OpenAPI 3.1.0 Specification](https://spec.openapis.org/oas/v3.1.0)
- [Swagger UI](https://swagger.io/tools/swagger-ui/)
- [ReDoc](https://redocly.com/redoc/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/tutorial/metadata/)
- [OpenAPI Generator](https://openapi-generator.tech/)
- [Redocly CLI (Linting)](https://redocly.com/docs/cli/)
- [openapi-spec-validator (Python)](https://github.com/python-openapi/openapi-spec-validator)
- [Stripe API Documentation](https://stripe.com/docs/api) — industry-leading example

---

**Previous**: [Rate Limiting and Throttling](./09_Rate_Limiting_and_Throttling.md) | [Overview](./00_Overview.md) | **Next**: [API Testing and Validation](./11_API_Testing_and_Validation.md)

**License**: CC BY-NC 4.0
