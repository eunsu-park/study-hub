# 레슨 1: API Design 기초

**이전**: - | [개요](00_Overview.md) | **다음**: [REST 아키텍처](02_REST_Architecture.md)

## 학습 목표(Learning Objectives)

이 레슨을 마치면 다음을 할 수 있습니다:

1. API가 무엇인지 정의하고 API 설계가 중요한 이유를 설명할 수 있다
2. 주요 API 패러다임(REST, RPC, GraphQL, 이벤트 기반)을 비교 대조할 수 있다
3. 핵심 설계 원칙인 일관성, 단순성, 탐색 가능성을 설명할 수 있다
4. 계약 우선 설계와 코드 우선 설계 접근 방식을 구분할 수 있다
5. API를 제품으로 평가하고 훌륭한 개발자 경험의 특성을 파악할 수 있다

---

API는 현대 소프트웨어의 결합 조직입니다. 모든 모바일 앱, 싱글 페이지 애플리케이션, IoT 디바이스, 마이크로서비스가 API를 통해 통신합니다. 그러나 잘못 설계된 API는 시간이 지남에 따라 복합적인 마찰을 일으킵니다 -- 혼란스러운 이름, 일관성 없는 규칙, 부족한 오류 세부 정보, 파괴적 변경 사항은 소비자를 멀리하고 팀의 속도를 늦춥니다. API를 잘 설계하는 법을 배우는 것은 백엔드 개발자가 습득할 수 있는 가장 높은 레버리지 기술 중 하나입니다.

> **비유:** API는 레스토랑 메뉴와 같습니다. 주방(서버)은 많은 것을 요리할 수 있지만, 메뉴(API)는 손님(클라이언트)이 무엇을 주문할 수 있고, 어떻게 주문하며, 무엇을 받을 수 있는지를 정의합니다. 명확하고 잘 정리된 메뉴는 손님을 행복하게 만들고, 혼란스러운 메뉴는 손님을 다른 레스토랑으로 보냅니다.

## 목차
1. [API란 무엇인가?](#api란-무엇인가)
2. [API 설계가 중요한 이유](#api-설계가-중요한-이유)
3. [API 패러다임](#api-패러다임)
4. [핵심 설계 원칙](#핵심-설계-원칙)
5. [계약 우선 vs 코드 우선](#계약-우선-vs-코드-우선)
6. [제품으로서의 API](#제품으로서의-api)
7. [적합한 패러다임 선택](#적합한-패러다임-선택)
8. [연습 문제](#연습-문제)

---

## API란 무엇인가?

**Application Programming Interface**(애플리케이션 프로그래밍 인터페이스)는 두 소프트웨어가 어떻게 통신하는지를 정의하는 계약입니다. 웹 개발 맥락에서 API는 일반적으로 요청을 받아들이고 구조화된 응답(보통 JSON)을 반환하는 HTTP 기반 인터페이스를 의미합니다.

### API 호출의 구조

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

### 핵심 용어

| 용어 | 정의 |
|------|------|
| **Endpoint** | 요청을 받아들이는 특정 URL 경로 (예: `/api/users`) |
| **Resource** | API를 통해 노출되는 도메인 객체 (예: User, Order) |
| **Method** | 동작을 나타내는 HTTP 동사 (GET, POST, PUT, DELETE) |
| **Payload** | 요청 또는 응답 본문에 전송되는 데이터 |
| **Header** | 요청/응답에 첨부되는 메타데이터 (인증 토큰, 콘텐츠 타입) |
| **Status Code** | 결과를 나타내는 숫자 코드 (200 OK, 404 Not Found) |

---

## API 설계가 중요한 이유

### 잘못된 설계의 비용

잘못된 API 설계는 복합적인 비용을 발생시킵니다:

1. **통합 시간** -- 개발자가 일관성 없는 문서를 읽고 동작을 역설계하느라 수 시간을 소비합니다
2. **지원 부담** -- 혼란스러운 API는 더 많은 지원 티켓을 생성합니다
3. **파괴적 변경** -- 제대로 계획되지 않은 API는 빈번하고 파괴적인 버전 변경을 필요로 합니다
4. **보안 취약점** -- 일관성 없는 인증 패턴은 공격자가 악용할 수 있는 틈을 남깁니다

### 좋은 설계의 가치

잘 설계된 API는 측정 가능한 이점을 제공합니다:

- **빠른 통합** -- 일관된 패턴으로 개발자가 동작을 예측할 수 있습니다
- **낮은 지원 비용** -- 자체 설명적인 응답이 질문을 줄입니다
- **긴 수명** -- 신중한 버전 관리와 확장성이 재작성을 지연시킵니다
- **생태계 성장** -- 우수한 DX가 서드파티 통합을 유치합니다

### 실제 사례: Stripe

Stripe는 API 설계의 모범 사례로 자주 인용됩니다:

```python
# Stripe's API is predictable and consistent:
# - Resources are nouns: /v1/customers, /v1/charges, /v1/invoices
# - CRUD maps to HTTP methods: GET (read), POST (create), DELETE (remove)
# - Errors follow a consistent structure
# - Every object has an "id" and "object" field
# - List endpoints always return {"data": [...], "has_more": bool}
```

---

## API 패러다임

### 1. REST (Representational State Transfer)

REST는 API를 표준 HTTP 메서드로 접근하는 **리소스**의 컬렉션으로 모델링합니다.

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

**장점:** 널리 이해됨, 캐시 가능, 무상태, 풍부한 도구 지원.
**단점:** 과다 페칭/과소 페칭, 복잡한 쿼리에 여러 번의 왕복 필요.

### 2. RPC (Remote Procedure Call)

RPC는 리소스가 아닌 **동작**(동사)을 노출합니다. 클라이언트가 서버의 명명된 프로시저를 호출합니다.

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

**장점:** 동작 지향 연산에 자연스러움, 단순한 함수 호출 의미론.
**단점:** 탐색 표준 부재, 일관성 없는 인터페이스 생성이 쉬움, 캐시 어려움.

### 3. GraphQL

GraphQL은 클라이언트가 단일 요청에서 **정확히** 필요한 필드를 지정할 수 있게 합니다.

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

**장점:** 과다 페칭 없음, 단일 엔드포인트, 강력한 타입 지정, 자체 문서화 스키마.
**단점:** 복잡성(캐싱, 필드별 인가), N+1 쿼리 위험, 학습 곡선.

### 4. 이벤트 기반 (Webhooks / Async APIs)

이벤트 기반 API는 폴링을 기다리는 대신 무언가 발생했을 때 소비자에게 데이터를 푸시합니다.

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

**장점:** 실시간, 느슨한 결합, 폴링 오버헤드 감소.
**단점:** 전달 신뢰성, 디버깅 어려움, 멱등성을 갖춘 소비자 필요.

---

## 핵심 설계 원칙

### 1. 일관성

모든 엔드포인트는 네이밍, 케이싱, 페이지네이션, 오류, 인증에 대해 동일한 규칙을 따라야 합니다. 일관성은 인지 부하를 줄입니다.

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

### 2. 단순성

좋은 API는 간단한 것은 쉽게, 복잡한 것은 가능하게 만듭니다. 최소한의 표면적으로 시작하고 실제 수요에 따라 확장합니다.

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

### 3. 탐색 가능성

소비자가 문서의 모든 페이지를 읽지 않고도 API를 탐색할 수 있어야 합니다. 링크, 스키마, 자체 설명적 응답을 제공합니다.

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

### 4. 예측 가능성

하나의 엔드포인트를 알면 다른 엔드포인트의 형태를 추측할 수 있어야 합니다. 일관된 패턴을 사용합니다:

```python
# If GET /api/users returns:
# {"data": [{"id": 1, "name": "Alice"}], "meta": {"total": 1}}

# Then GET /api/orders should return:
# {"data": [{"id": 10, "total": 99.99}], "meta": {"total": 1}}

# Same envelope, same meta, same pagination keys.
```

### 5. 견고성 (Postel의 법칙)

> "받는 것에 관대하고, 보내는 것에 보수적이어라."

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

## 계약 우선 vs 코드 우선

### 계약 우선 (설계 우선)

코드를 작성하기 **전에** API 사양(OpenAPI/Swagger)을 먼저 작성합니다. 팀이 계약을 검토하고 합의한 후 구현합니다.

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

**장점:**
- 프론트엔드와 백엔드가 병렬로 작업 가능
- 구현 전에 API가 설계 산출물로 검토됨
- 사양에서 즉시 클라이언트 SDK 생성 가능
- 의도적인 설계 결정을 강제함

**단점:**
- 강제 도구가 없으면 사양이 구현과 불일치할 수 있음
- 소규모 프로젝트에서는 시작이 느림

### 코드 우선

구현을 먼저 작성한 후, 코드 어노테이션에서 사양을 생성합니다.

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

**장점:**
- 사양이 항상 코드와 동기화됨 (단일 진실 소스)
- 시작이 빠름
- 프레임워크 기능 활용 (FastAPI, Django REST Framework)

**단점:**
- 설계 결정이 검토 단계가 아닌 코딩 중에 이루어짐
- 초기 이해관계자 피드백을 받기 어려움

### 어떤 것을 선택할 것인가?

| 요소 | 계약 우선 | 코드 우선 |
|------|----------|----------|
| 팀 규모 | 대규모 / 교차 기능 팀 | 소규모 / 단일 팀 |
| API 대상 | 공개 / 외부 | 내부 / 비공개 |
| 반복 속도 | 초기에 느림, 장기적으로 빠름 | 초기에 빠름 |
| 사양 정확도 | 강제 도구 필요 | 자동 |
| 권장 대상 | 플랫폼 API, 파트너 통합 | 마이크로서비스, MVP |

---

## 제품으로서의 API

API를 제품으로 취급하면 소비자의 개발자 경험(DX)에 집중하게 됩니다.

### API를 위한 제품적 사고

```
Traditional Thinking          Product Thinking
─────────────────────         ──────────────────
"Ship endpoints"              "Solve developer problems"
"Document the API"            "Enable self-service onboarding"
"Fix bugs"                    "Measure time-to-first-call"
"Add features"                "Understand use cases, then add features"
```

### 핵심 DX 지표

1. **Time to First Call (TTFC)** -- 개발자가 첫 번째 성공적인 API 호출을 하기까지 얼마나 걸리는가?
2. **Time to Working App (TTWA)** -- 가입에서 작동하는 통합까지 얼마나 걸리는가?
3. **Error Resolution Time** -- 개발자가 오류를 이해하고 수정하는 데 얼마나 걸리는가?
4. **Self-Service Rate** -- 지원에 연락하지 않고 통합하는 개발자의 비율은 얼마인가?

### 훌륭한 DX 구축하기

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

## 적합한 패러다임 선택

"최고의" 패러다임은 하나가 아닙니다. 선택은 사용 사례에 따라 달라집니다:

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

### 의사결정 프레임워크

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

## 연습 문제

### 연습 1: 기존 API 분류

세 가지 공개 API(예: GitHub, Stripe, Twitter/X)를 방문하고 각각을 REST, RPC, GraphQL 또는 하이브리드로 분류하십시오. 각 API에 대해 다음을 파악합니다:
- 리소스/동작의 네이밍 방식
- 사용되는 HTTP 메서드
- 오류 구조
- REST 제약 조건의 엄격한 준수 여부

### 연습 2: API 계약 설계

도메인(예: 도서관 시스템, 음식 배달 앱, 태스크 관리자)을 선택하고 다음을 포함하는 계약 우선 OpenAPI 사양을 작성하십시오:
- 3개 이상의 리소스 (예: Books, Authors, Loans)
- 각 리소스에 대한 CRUD 엔드포인트
- 일관된 네이밍 및 응답 엔벨로프
- 오류 응답 스키마

### 연습 3: 패러다임 비교

사용자, 게시물, 댓글, 좋아요가 있는 소셜 미디어 애플리케이션에 대해 다음을 스케치하십시오:
1. REST API (리소스 엔드포인트)
2. RPC API (동작 엔드포인트)
3. GraphQL 스키마 (타입 및 쿼리)

사용자, 최근 게시물 10개, 각 게시물의 댓글 수를 보여주는 "사용자 프로필 페이지"에 필요한 요청 수를 비교합니다.

### 연습 4: DX 평가

사용해 본 공개 API를 선택하고 개발자 경험을 평가하십시오:
- 첫 번째 성공적인 호출을 하기까지 얼마나 걸렸는가?
- 문서가 명확했는가?
- 오류 메시지가 도움이 되었는가?
- 무엇을 개선하겠는가?

---

## 요약

이 레슨에서 다룬 내용:
1. API가 무엇이고 설계 품질이 중요한 이유
2. 네 가지 주요 패러다임: REST, RPC, GraphQL, 이벤트 기반
3. 핵심 설계 원칙: 일관성, 단순성, 탐색 가능성, 예측 가능성, 견고성
4. 계약 우선 vs 코드 우선 설계 접근 방식
5. 측정 가능한 DX 지표를 통한 제품으로서의 API 관점
6. 적합한 패러다임 선택을 위한 의사결정 프레임워크

---

**이전**: - | [개요](00_Overview.md) | **다음**: [REST 아키텍처](02_REST_Architecture.md)

**License**: CC BY-NC 4.0
