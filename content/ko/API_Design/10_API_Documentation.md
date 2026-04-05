# 10. API 문서화

**이전**: [Rate Limiting과 Throttling](./09_Rate_Limiting_and_Throttling.md) | **다음**: [API 테스팅과 검증](./11_API_Testing_and_Validation.md)

**난이도**: ⭐⭐

---

## 학습 목표

- API 엔드포인트, 스키마, 인증을 완전히 설명하는 OpenAPI 3.x 명세 작성하기
- 대화형 API 문서를 위한 Swagger UI와 ReDoc 설정하기
- FastAPI의 내장 OpenAPI 지원을 사용하여 코드에서 문서를 자동 생성하기
- 명확한 요약, 예제, 오류 문서를 포함한 API 설명 모범 사례 적용하기
- 문서와 테스트 픽스처를 겸하는 유용한 요청 및 응답 예제 생성하기
- API가 발전함에 따라 문서의 정확성 유지하기

---

## 목차

1. [문서화가 중요한 이유](#1-문서화가-중요한-이유)
2. [OpenAPI 3.x 명세](#2-openapi-3x-명세)
3. [Swagger UI와 ReDoc](#3-swagger-ui와-redoc)
4. [FastAPI를 사용한 자동 생성](#4-fastapi를-사용한-자동-생성)
5. [설명 작성 모범 사례](#5-설명-작성-모범-사례)
6. [문서의 예제](#6-문서의-예제)
7. [인증 문서화](#7-인증-문서화)
8. [오류 문서화](#8-오류-문서화)
9. [문서 동기화 유지](#9-문서-동기화-유지)
10. [연습 문제](#10-연습-문제)
11. [참고 자료](#11-참고-자료)

---

## 1. 문서화가 중요한 이유

API 문서는 API와 소비자 사이의 주요 인터페이스입니다. 부실한 문서는 다음을 초래합니다:

- 높은 지원 부담 (개발자가 스스로 해결할 수 없음)
- 느린 도입 및 통합
- 엔드포인트의 오용 (잘못된 파라미터, 누락된 헤더)
- 좌절감과 이탈

좋은 문서는 모든 엔드포인트에 대해 세 가지 질문에 답합니다:

1. **이 엔드포인트는 무엇을 하는가?** (요약과 설명)
2. **무엇을 보내야 하는가?** (요청 파라미터, 헤더, 본문)
3. **무엇을 돌려받는가?** (응답 스키마, 상태 코드, 예제)

### Documentation-First vs Code-First

| 접근 방식 | 설명 | 장점 | 단점 |
|----------|-------------|------|------|
| Design-first | 코딩 전에 OpenAPI 명세 작성 | 좋은 설계를 강제, 병렬 작업 가능 | 명세가 구현과 불일치할 수 있음 |
| Code-first | 어노테이션된 코드에서 명세 생성 | 항상 정확, 낮은 유지보수 | 설계 지름길로 이어질 수 있음 |

FastAPI는 code-first 접근 방식에서 탁월합니다: Pydantic 모델과 타입 힌트가 자동으로 정확한 OpenAPI 문서를 생성합니다.

---

## 2. OpenAPI 3.x 명세

**OpenAPI Specification**(이전 Swagger)은 REST API를 설명하기 위한 업계 표준입니다. 엔드포인트, 파라미터, 요청/응답 스키마, 인증을 설명하는 기계 판독 가능한 형식(YAML 또는 JSON)입니다.

### 구조 개요

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

### 핵심 개념

| 개념 | 설명 |
|---------|-------------|
| `paths` | HTTP 메서드를 포함한 엔드포인트 정의 |
| `components/schemas` | 재사용 가능한 데이터 모델 (JSON Schema) |
| `$ref` | 재사용 가능한 컴포넌트에 대한 참조 |
| `operationId` | 각 작업의 고유 식별자 (코드 생성기에서 사용) |
| `tags` | 엔드포인트를 그룹으로 정리 |
| `securitySchemes` | 인증 메커니즘 정의 |

---

## 3. Swagger UI와 ReDoc

OpenAPI 명세를 대화형 문서로 변환하는 두 가지 인기 있는 렌더러입니다.

### Swagger UI

Swagger UI는 개발자가 문서를 읽고 브라우저에서 직접 API 호출을 실행할 수 있는 대화형 인터페이스를 제공합니다.

FastAPI는 기본적으로 `/docs`에서 Swagger UI를 포함합니다:

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

ReDoc은 깔끔한 3단 패널 문서 레이아웃을 제공합니다. 읽기 전용이지만("Try it out" 기능 없음) 마크다운을 지원하는 장문의 문서에서 뛰어납니다.

FastAPI는 기본적으로 `/redoc`에서 ReDoc을 포함합니다.

### 문서 UI 커스터마이징

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

### 프로덕션에서 문서 비활성화

```python
import os

docs_url = "/docs" if os.getenv("ENVIRONMENT") != "production" else None
redoc_url = "/redoc" if os.getenv("ENVIRONMENT") != "production" else None

app = FastAPI(docs_url=docs_url, redoc_url=redoc_url)
```

---

## 4. FastAPI를 사용한 자동 생성

FastAPI는 Python 코드에서 자동으로 OpenAPI 문서를 생성합니다. 모든 타입 힌트, Pydantic 모델, docstring이 생성된 명세에 기여합니다.

### 스키마로서의 Pydantic 모델

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

### 라우트 문서화

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

### 태그와 그룹화

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

## 5. 설명 작성 모범 사례

### 효과적인 요약 작성

`summary` 필드는 엔드포인트 목록에 표시되는 짧은 레이블입니다. `description` 필드는 상세한 컨텍스트를 제공합니다.

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

### 문서 체크리스트

모든 엔드포인트에 대해 다음을 문서화하십시오:

1. **요약(Summary)**: 한 줄 설명 (엔드포인트 목록에 표시)
2. **설명(Description)**: 엔드포인트가 하는 일, 부수 효과, 필요한 권한
3. **파라미터(Parameters)**: 모든 파라미터의 타입, 제약 조건, 기본값, 예제
4. **요청 본문(Request body)**: 필드 설명과 완전한 예제가 포함된 스키마
5. **응답 코드(Response codes)**: 가능한 모든 상태 코드와 설명 및 예제 본문
6. **인증(Authentication)**: 필요한 자격 증명 (있는 경우)
7. **Rate limit**: 적용 가능한 rate limit 등급 (기본값과 다른 경우)

### 설명에서 Markdown 사용

OpenAPI는 설명 필드에서 Markdown을 지원합니다. 서식 지정에 활용하십시오:

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

## 6. 문서의 예제

예제는 API 문서에서 가장 가치 있는 부분입니다. 개발자는 종종 설명을 건너뛰고 바로 예제로 이동합니다.

### 요청 예제

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

### 응답 예제

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

### 설명에 cURL 예제 포함

커맨드 라인을 선호하는 개발자를 위해 cURL 예제를 포함하십시오:

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

## 7. 인증 문서화

### Security Scheme

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

## 8. 오류 문서화

모든 엔드포인트는 가능한 오류 응답을 문서화해야 합니다. 이는 클라이언트 개발자가 강건한 오류 처리를 작성하는 데 도움이 됩니다.

### 중앙화된 오류 응답

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

## 9. 문서 동기화 유지

### 자동화된 검증

CI에서 OpenAPI 스키마를 검증하여 문서가 구현과 일치하는지 확인하십시오:

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

### 명세 내보내기

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

## 10. 연습 문제

### 연습 문제 1: OpenAPI 명세 작성

다음 엔드포인트를 가진 **Task Management API**에 대한 완전한 OpenAPI 3.1 명세를 YAML로 작성하십시오:

- `GET /tasks` -- 필터링(status, priority, assignee)과 페이지네이션을 포함한 태스크 목록
- `POST /tasks` -- 태스크 생성 (title, description, priority, assignee_id)
- `GET /tasks/{id}` -- 단일 태스크 조회
- `PATCH /tasks/{id}` -- 태스크 필드 업데이트
- `DELETE /tasks/{id}` -- 태스크 삭제
- `POST /tasks/{id}/comments` -- 태스크에 댓글 추가

포함 사항: 모든 모델의 스키마, 요청 및 응답 예제, Bearer 토큰 인증, 400, 401, 404, 422에 대한 오류 응답.

### 연습 문제 2: FastAPI 문서 개선

다음 FastAPI 엔드포인트에 summary, description, 파라미터 설명, 응답 모델, 응답 예제(성공과 오류), 태그를 포함한 완전한 문서를 추가하십시오:

```python
@app.get("/users/{user_id}/orders")
async def get_user_orders(user_id: int, status: str = None, limit: int = 20):
    ...
```

### 연습 문제 3: API Changelog 생성기

두 개의 OpenAPI JSON 파일(이전 버전과 새 버전)을 비교하여 사람이 읽을 수 있는 changelog를 생성하는 Python 스크립트를 구축하십시오. Changelog는 다음을 식별해야 합니다:

- 추가된 엔드포인트
- 제거된 엔드포인트
- 변경된 요청 파라미터 (추가, 제거, 타입 변경)
- 변경된 응답 스키마
- 변경된 인증 요구사항

출력을 릴리스 노트에 적합한 Markdown으로 포맷하십시오.

### 연습 문제 4: 문서 Linter

FastAPI 애플리케이션의 문서 품질을 검증하는 pytest 테스트 스위트를 생성하십시오. 다음을 확인하십시오:

- 모든 엔드포인트에 비어있지 않은 summary가 있음 (최대 80자)
- 모든 엔드포인트에 description이 있음
- 모든 path 파라미터에 description이 있음
- 모든 query 파라미터에 description과 예제가 있음
- 모든 POST/PUT/PATCH 엔드포인트에 request body 예제가 있음
- 모든 응답 코드에 description이 있음
- 어떤 엔드포인트도 format이나 pattern 제약 없이 일반적인 "string" 타입을 사용하지 않음

### 연습 문제 5: 대화형 예제

FastAPI 애플리케이션에 복잡한 검색 쿼리 본문을 수용하는 `POST /books/search` 엔드포인트를 추가하십시오. OpenAPI 스키마에 최소 3개의 이름 있는 예제를 제공하십시오:

1. 간단한 제목 검색
2. 장르, 가격 범위, 날짜 필터를 포함한 고급 검색
3. 페이지네이션과 정렬을 포함한 전문 검색

모든 예제가 Swagger UI와 ReDoc 모두에서 올바르게 렌더링되는지 확인하십시오.

---

## 11. 참고 자료

- [OpenAPI 3.1.0 Specification](https://spec.openapis.org/oas/v3.1.0)
- [Swagger UI](https://swagger.io/tools/swagger-ui/)
- [ReDoc](https://redocly.com/redoc/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/tutorial/metadata/)
- [OpenAPI Generator](https://openapi-generator.tech/)
- [Redocly CLI (Linting)](https://redocly.com/docs/cli/)
- [openapi-spec-validator (Python)](https://github.com/python-openapi/openapi-spec-validator)
- [Stripe API Documentation](https://stripe.com/docs/api) -- 업계 최고 수준의 예제

---

**이전**: [Rate Limiting과 Throttling](./09_Rate_Limiting_and_Throttling.md) | [개요](./00_Overview.md) | **다음**: [API 테스팅과 검증](./11_API_Testing_and_Validation.md)

**License**: CC BY-NC 4.0
