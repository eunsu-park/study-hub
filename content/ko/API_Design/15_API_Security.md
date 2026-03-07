# 15. API 보안(API Security)

**이전**: [gRPC와 Protocol Buffers](./14_gRPC_and_Protocol_Buffers.md) | **다음**: [API 생명주기 관리](./16_API_Lifecycle_Management.md)

**난이도**: ⭐⭐⭐

---

## 학습 목표

- OWASP API Security Top 10 취약점을 식별하고 완화할 수 있다
- 인젝션 공격과 데이터 손상을 방지하기 위한 견고한 입력 검증을 구현할 수 있다
- 교차 출처(cross-origin) API 접근에 대해 보안과 사용성의 균형을 맞추는 CORS 정책을 설정할 수 있다
- 쿠키 기반 인증을 사용하는 API에 CSRF 보호 전략을 적용할 수 있다
- 적절한 응답 필터링과 로깅 관행을 통해 민감한 데이터 노출을 방지할 수 있다
- API 엔드포인트에 함수 수준 및 객체 수준 인가 검사를 구현할 수 있다

---

## 목차

1. [OWASP API Security Top 10](#1-owasp-api-security-top-10)
2. [입력 검증](#2-입력-검증)
3. [CORS (Cross-Origin Resource Sharing)](#3-cors-cross-origin-resource-sharing)
4. [CSRF 보호](#4-csrf-보호)
5. [인젝션 방지](#5-인젝션-방지)
6. [민감한 데이터 노출](#6-민감한-데이터-노출)
7. [객체 수준 인가 결함 (BOLA)](#7-객체-수준-인가-결함-bola)
8. [함수 수준 인가 결함](#8-함수-수준-인가-결함)
9. [보안 헤더와 강화](#9-보안-헤더와-강화)
10. [연습 문제](#10-연습-문제)
11. [참고 자료](#11-참고-자료)

---

## 1. OWASP API Security Top 10

OWASP API Security Top 10 (2023년 에디션)은 API의 가장 중요한 보안 위험을 식별합니다. 이러한 위험을 이해하는 것이 안전한 API 설계의 기초입니다.

| # | 위험 | 설명 |
|---|------|------|
| API1 | 객체 수준 인가 결함(Broken Object-Level Authorization) | 객체 ID를 조작하여 다른 사용자의 데이터에 접근 |
| API2 | 인증 결함(Broken Authentication) | 약하거나 누락된 인증 메커니즘 |
| API3 | 객체 속성 수준 인가 결함(Broken Object Property-Level Authorization) | 민감한 객체 속성의 노출 또는 수락 |
| API4 | 무제한 리소스 소비(Unrestricted Resource Consumption) | 요청 크기, 속도 또는 리소스 사용에 대한 제한 없음 |
| API5 | 함수 수준 인가 결함(Broken Function-Level Authorization) | 일반 사용자가 관리자 기능에 접근 |
| API6 | 민감한 비즈니스 흐름에 대한 무제한 접근(Unrestricted Access to Sensitive Business Flows) | 사람 대상 흐름의 자동화 (티켓 스캘핑, 크리덴셜 스터핑) |
| API7 | 서버 측 요청 위조(Server-Side Request Forgery, SSRF) | 서버가 내부 리소스를 요청하도록 만듦 |
| API8 | 보안 설정 오류(Security Misconfiguration) | 기본 자격 증명, 장황한 오류, 누락된 헤더 |
| API9 | 부적절한 인벤토리 관리(Improper Inventory Management) | 추적되지 않거나 더 이상 사용되지 않는 섀도 API |
| API10 | 서드파티 API의 안전하지 않은 사용(Unsafe Consumption of Third-Party APIs) | 외부 API 데이터를 검증 없이 신뢰 |

---

## 2. 입력 검증

클라이언트로부터 오는 모든 데이터는 신뢰할 수 없습니다. 경로 파라미터, 쿼리 파라미터, 헤더, 요청 본문 등 모든 것을 검증해야 합니다.

### 심층 방어(Defense in Depth)

```python
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, Query, Path, HTTPException
import re
import bleach

app = FastAPI()


class UserCreate(BaseModel):
    """User creation schema with comprehensive validation.

    Every field has type constraints, length limits, and
    format validation. This prevents injection attacks
    and data corruption at the schema level.
    """

    username: str = Field(
        ...,
        min_length=3,
        max_length=32,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Alphanumeric with underscores and hyphens",
    )
    email: str = Field(
        ...,
        max_length=254,
        pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    )
    display_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )
    bio: str | None = Field(
        default=None,
        max_length=500,
    )
    role: str = Field(
        default="user",
        pattern=r"^(user|moderator)$",
        description="Only 'user' or 'moderator' allowed at creation",
    )

    @field_validator("display_name")
    @classmethod
    def sanitize_display_name(cls, v):
        """Strip HTML to prevent stored XSS."""
        sanitized = bleach.clean(v, tags=[], strip=True)
        if sanitized != v:
            raise ValueError("HTML is not allowed in display name")
        return sanitized.strip()

    @field_validator("bio")
    @classmethod
    def sanitize_bio(cls, v):
        """Allow limited HTML in bio but strip dangerous tags."""
        if v is None:
            return v
        return bleach.clean(
            v,
            tags=["b", "i", "em", "strong", "a"],
            attributes={"a": ["href"]},
            strip=True,
        )

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v):
        """Normalize email to lowercase."""
        return v.lower().strip()


@app.post("/users", status_code=201)
async def create_user(user: UserCreate):
    """Create a user with validated input.

    Pydantic enforces all constraints before the handler runs.
    If validation fails, FastAPI returns 422 with field-level errors.
    """
    ...
```

### 파라미터 검증

```python
@app.get("/users/{user_id}/posts")
async def get_user_posts(
    user_id: int = Path(
        ...,
        gt=0,
        lt=2**31,
        description="User ID must be a positive 32-bit integer",
    ),
    status: str | None = Query(
        default=None,
        regex=r"^(draft|published|archived)$",
        description="Filter by post status",
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Number of posts to return (max 100)",
    ),
    sort: str = Query(
        default="-created_at",
        regex=r"^-?(created_at|updated_at|title)$",
        description="Sort field (prefix with - for descending)",
    ),
):
    """Validated parameters prevent:
    - Integer overflow (gt=0, lt=2^31)
    - Arbitrary field names in sort (injection vector)
    - Excessive result sets (le=100)
    - Invalid enum values (regex constraint)
    """
    ...
```

### 요청 크기 제한

```python
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()

MAX_BODY_SIZE = 1 * 1024 * 1024  # 1 MB


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Reject requests with bodies larger than the limit.

    Prevents denial-of-service attacks that send enormous payloads
    to exhaust server memory.
    """
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_SIZE:
        return JSONResponse(
            status_code=413,
            content={
                "detail": f"Request body too large. Maximum: {MAX_BODY_SIZE} bytes"
            },
        )
    return await call_next(request)
```

---

## 3. CORS (Cross-Origin Resource Sharing)

CORS는 어떤 웹사이트가 브라우저에서 여러분의 API를 호출할 수 있는지를 제어합니다. 적절한 CORS 설정이 없으면 어떤 웹사이트든 사용자를 대신하여 인증된 요청을 할 수 있습니다.

### CORS 동작 원리

```
Browser (https://app.example.com)      API (https://api.example.com)
   |                                      |
   |-- OPTIONS /api/users (preflight) --->|
   |   Origin: https://app.example.com    |
   |   Access-Control-Request-Method: POST|
   |   Access-Control-Request-Headers:    |
   |     Authorization, Content-Type      |
   |                                      |
   |<-- 204 No Content ------------------|
   |   Access-Control-Allow-Origin:       |
   |     https://app.example.com          |
   |   Access-Control-Allow-Methods:      |
   |     GET, POST, PUT, DELETE           |
   |   Access-Control-Allow-Headers:      |
   |     Authorization, Content-Type      |
   |   Access-Control-Max-Age: 3600       |
   |                                      |
   |-- POST /api/users ----------------->|
   |   Origin: https://app.example.com    |
   |                                      |
   |<-- 201 Created --------------------|
   |   Access-Control-Allow-Origin:       |
   |     https://app.example.com          |
```

### FastAPI CORS 설정

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Production: explicit origins only
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.example.com",
        "https://admin.example.com",
    ],
    allow_credentials=True,  # Allow cookies/Authorization header
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Request-Id"],
    expose_headers=["X-RateLimit-Remaining", "X-Request-Id"],
    max_age=3600,  # Cache preflight response for 1 hour
)
```

### 흔한 CORS 실수

```python
# DANGEROUS: Allow all origins with credentials
# Any website can make authenticated requests to your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # BAD: any website
    allow_credentials=True,         # BAD: with cookies!
)

# SECURE: Explicit origins with credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.example.com"],  # Explicit
    allow_credentials=True,
)

# ACCEPTABLE: Public API without credentials
# If your API does not use cookies and requires API keys,
# wildcard origins are acceptable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # No cookies
)
```

### 멀티 테넌트를 위한 동적 CORS

```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class DynamicCORSMiddleware(BaseHTTPMiddleware):
    """CORS middleware that validates origins against a database.

    Useful for multi-tenant platforms where each tenant has
    their own allowed origin domain.
    """

    async def dispatch(self, request, call_next):
        origin = request.headers.get("origin")

        if origin and await self.is_allowed_origin(origin):
            if request.method == "OPTIONS":
                # Preflight response
                response = Response(status_code=204)
            else:
                response = await call_next(request)

            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = (
                "GET, POST, PUT, DELETE, OPTIONS"
            )
            response.headers["Access-Control-Allow-Headers"] = (
                "Authorization, Content-Type"
            )
            response.headers["Access-Control-Max-Age"] = "3600"
            return response

        return await call_next(request)

    async def is_allowed_origin(self, origin: str) -> bool:
        """Check if the origin is registered for any tenant."""
        # In production, cache this lookup
        return await db.scalar(
            select(Tenant).where(Tenant.allowed_origin == origin)
        ) is not None
```

---

## 4. CSRF 보호

교차 사이트 요청 위조(Cross-Site Request Forgery, CSRF)는 인증된 사용자의 브라우저를 속여 원치 않는 요청을 하도록 만듭니다. 쿠키 기반 인증을 사용하는 API가 취약합니다.

### API에 CSRF 보호가 필요한 이유

```
1. User logs into https://app.example.com (session cookie set)
2. User visits https://evil-site.com
3. Evil site contains: <form action="https://api.example.com/transfer" method="POST">
4. Form auto-submits with the user's session cookie
5. API processes the request as if the user intended it
```

### CSRF 토큰 패턴

```python
import secrets
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse

app = FastAPI()


def generate_csrf_token() -> str:
    """Generate a cryptographically secure CSRF token."""
    return secrets.token_urlsafe(32)


@app.get("/csrf-token")
async def get_csrf_token(request: Request):
    """Issue a CSRF token for the current session.

    The client stores this token and includes it in the
    X-CSRFToken header on all state-changing requests.
    """
    token = generate_csrf_token()
    # Store in session for later verification
    request.session["csrf_token"] = token
    return {"csrf_token": token}


async def verify_csrf_token(request: Request):
    """Verify CSRF token on state-changing requests.

    Only applies to POST, PUT, PATCH, DELETE — not GET or OPTIONS.
    """
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return

    # Token-based auth (Bearer) is not vulnerable to CSRF
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return

    # For cookie-based auth, verify the CSRF token
    expected = request.session.get("csrf_token")
    received = request.headers.get("X-CSRFToken")

    if not expected or not received or not secrets.compare_digest(expected, received):
        raise HTTPException(
            status_code=403,
            detail="CSRF token missing or invalid",
        )


@app.post("/transfer", dependencies=[Depends(verify_csrf_token)])
async def transfer_money(amount: float, to_account: str):
    """This endpoint is protected by CSRF verification."""
    ...
```

### SameSite 쿠키

최신 브라우저는 기본 CSRF 보호를 제공하는 `SameSite` 쿠키 속성을 지원합니다:

```python
response.set_cookie(
    key="session_id",
    value=session_id,
    httponly=True,
    secure=True,
    samesite="lax",     # Prevents CSRF for POST/PUT/DELETE
    max_age=3600,
)
```

| SameSite 값 | 동작 | CSRF 보호 |
|-------------|------|----------|
| `Strict` | 쿠키가 교차 사이트에서 전송되지 않음 | 완전 (그러나 합법적인 링크를 깨뜨릴 수 있음) |
| `Lax` | 최상위 GET 네비게이션에서만 쿠키 전송 | 양호 (권장 기본값) |
| `None` | 쿠키가 항상 전송됨 (`Secure` 필요) | 없음 (CSRF 토큰을 사용해야 함) |

---

## 5. 인젝션 방지

### SQL 인젝션

사용자 입력을 연결하여 SQL 쿼리를 구성하지 마세요.

```python
# VULNERABLE: SQL injection
@app.get("/users")
async def search_users_vulnerable(name: str):
    # An attacker can send: name = "'; DROP TABLE users; --"
    query = f"SELECT * FROM users WHERE name = '{name}'"  # NEVER DO THIS
    return await db.execute(query)


# SAFE: Parameterized queries (SQLAlchemy)
from sqlalchemy import select
from sqlalchemy.orm import Session

@app.get("/users")
async def search_users_safe(name: str, db: Session = Depends(get_db)):
    """SQLAlchemy parameterizes all values automatically.
    The input is never interpreted as SQL code.
    """
    query = select(User).where(User.name == name)
    result = await db.execute(query)
    return result.scalars().all()


# SAFE: Raw SQL with parameters (when ORM is not available)
@app.get("/search")
async def search_raw(q: str, db: Session = Depends(get_db)):
    """Use :param syntax for parameterized raw queries."""
    result = await db.execute(
        text("SELECT * FROM books WHERE title ILIKE :query"),
        {"query": f"%{q}%"},
    )
    return result.fetchall()
```

### NoSQL 인젝션

MongoDB 및 기타 NoSQL 데이터베이스도 취약합니다:

```python
# VULNERABLE: MongoDB operator injection
# Attacker sends: {"username": {"$gt": ""}, "password": {"$gt": ""}}
# This matches any document where username > "" and password > ""
@app.post("/login")
async def login_vulnerable(credentials: dict):
    user = await db.users.find_one(credentials)  # DANGEROUS
    ...


# SAFE: Explicitly extract and type-check values
@app.post("/login")
async def login_safe(credentials: LoginRequest):
    """Pydantic ensures username and password are strings,
    preventing operator injection."""
    user = await db.users.find_one({
        "username": credentials.username,  # Always a string
        "password_hash": hash_password(credentials.password),
    })
    ...
```

### 명령 인젝션

```python
import subprocess
import shlex

# VULNERABLE: Command injection
@app.get("/ping")
async def ping_vulnerable(host: str):
    # Attacker sends: host = "google.com; rm -rf /"
    result = subprocess.run(f"ping -c 1 {host}", shell=True, capture_output=True)
    return {"output": result.stdout.decode()}


# SAFE: Use list arguments (no shell interpolation)
@app.get("/ping")
async def ping_safe(host: str):
    """Using list arguments prevents command injection.
    Also validate the input format."""
    # Validate: only allow hostnames and IP addresses
    if not re.match(r"^[a-zA-Z0-9.-]+$", host):
        raise HTTPException(status_code=400, detail="Invalid hostname")

    result = subprocess.run(
        ["ping", "-c", "1", host],   # List arguments, no shell=True
        capture_output=True,
        timeout=5,
    )
    return {"output": result.stdout.decode()}
```

---

## 6. 민감한 데이터 노출

### 응답 필터링

클라이언트가 필요로 하는 것보다 더 많은 데이터를 반환하지 마세요:

```python
from pydantic import BaseModel, Field


class UserInternal(BaseModel):
    """Full user model used internally."""
    id: int
    username: str
    email: str
    password_hash: str
    ssn: str
    role: str
    api_key: str
    created_at: datetime


class UserPublicResponse(BaseModel):
    """Response model that excludes sensitive fields.

    Using response_model ensures that password_hash, ssn,
    and api_key are NEVER included in the response,
    even if the internal model contains them.
    """
    id: int
    username: str
    email: str
    role: str
    created_at: datetime


class UserAdminResponse(UserPublicResponse):
    """Admin response includes more fields but still excludes secrets."""
    ssn_last_four: str  # Only last 4 digits
    api_key_prefix: str  # Only first 8 characters


@app.get("/users/{user_id}", response_model=UserPublicResponse)
async def get_user(user_id: int):
    """response_model acts as a security filter.

    Even though find_user returns the full internal model
    with password_hash and SSN, FastAPI only serializes
    the fields defined in UserPublicResponse.
    """
    user = await find_user(user_id)
    if not user:
        raise HTTPException(status_code=404)
    return user  # Pydantic filters out sensitive fields
```

### 로깅 살균(Sanitization)

```python
import logging
import re

logger = logging.getLogger(__name__)


class SensitiveDataFilter(logging.Filter):
    """Redact sensitive data from log messages.

    Prevents accidental exposure of tokens, passwords,
    and other secrets in application logs.
    """

    PATTERNS = [
        (re.compile(r'(Authorization:\s*Bearer\s+)\S+'), r'\1[REDACTED]'),
        (re.compile(r'(password["\s:=]+)\S+', re.I), r'\1[REDACTED]'),
        (re.compile(r'(api[_-]?key["\s:=]+)\S+', re.I), r'\1[REDACTED]'),
        (re.compile(r'(secret["\s:=]+)\S+', re.I), r'\1[REDACTED]'),
        (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[SSN-REDACTED]'),  # SSN
        (re.compile(r'\b\d{16}\b'), '[CARD-REDACTED]'),  # Credit card
    ]

    def filter(self, record):
        if isinstance(record.msg, str):
            for pattern, replacement in self.PATTERNS:
                record.msg = pattern.sub(replacement, record.msg)
        return True


# Apply the filter
handler = logging.StreamHandler()
handler.addFilter(SensitiveDataFilter())
logger.addHandler(handler)

# These will be redacted automatically
logger.info("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9...")
# → "Authorization: Bearer [REDACTED]"

logger.info("User login with password=secret123")
# → "User login with password=[REDACTED]"
```

### 오류 메시지 보안

```python
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Never expose internal details in error responses.

    In production, return a generic error message. Log the
    full exception details server-side for debugging.
    """
    # Log full details for debugging
    logger.exception(
        f"Unhandled exception on {request.method} {request.url.path}",
        exc_info=exc,
    )

    # Return generic message to client
    return JSONResponse(
        status_code=500,
        content={
            "type": "https://api.example.com/errors/internal",
            "title": "Internal Server Error",
            "status": 500,
            "detail": "An unexpected error occurred. Please try again later.",
            # NEVER include: stack traces, SQL queries, file paths,
            # internal service names, or database error details
        },
    )
```

---

## 7. 객체 수준 인가 결함 (BOLA)

BOLA(Broken Object-Level Authorization, IDOR -- Insecure Direct Object Reference라고도 함)는 API 보안 위험 1위입니다. 사용자가 리소스 ID를 변경하여 다른 사용자의 데이터에 접근할 수 있을 때 발생합니다.

### 취약한 엔드포인트

```python
# VULNERABLE: No ownership check
@app.get("/users/{user_id}/orders")
async def get_orders(user_id: int):
    """Any authenticated user can view any other user's orders
    by changing the user_id in the URL."""
    orders = await db.execute(
        select(Order).where(Order.user_id == user_id)
    )
    return orders.scalars().all()


# VULNERABLE: Direct object access without authorization
@app.get("/documents/{doc_id}")
async def get_document(doc_id: int):
    """Any user can access any document by guessing or enumerating IDs."""
    doc = await db.get(Document, doc_id)
    return doc
```

### 안전한 구현

```python
from fastapi import Depends


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Extract and validate the current user from the JWT token."""
    ...


@app.get("/users/{user_id}/orders")
async def get_orders(
    user_id: int,
    current_user: User = Depends(get_current_user),
):
    """Enforce that users can only access their own orders.

    Admin users can access any user's orders.
    """
    if current_user.id != user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="You can only access your own orders",
        )

    orders = await db.execute(
        select(Order).where(Order.user_id == user_id)
    )
    return orders.scalars().all()


@app.get("/documents/{doc_id}")
async def get_document(
    doc_id: int,
    current_user: User = Depends(get_current_user),
):
    """Check document ownership or shared access."""
    doc = await db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404)

    # Check ownership or sharing
    if doc.owner_id != current_user.id:
        # Check if document is shared with this user
        shared = await db.execute(
            select(DocumentShare).where(
                DocumentShare.document_id == doc_id,
                DocumentShare.user_id == current_user.id,
            )
        )
        if not shared.scalar_one_or_none():
            raise HTTPException(
                status_code=404,  # 404 instead of 403 to avoid
                                  # confirming the resource exists
                detail="Document not found",
            )

    return doc
```

### 순차 ID 대신 UUID 사용

순차 정수 ID는 열거하기 쉽습니다. UUID는 추측하기 훨씬 어렵습니다:

```python
import uuid
from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import UUID


class Document(Base):
    __tablename__ = "documents"

    # UUID primary key — not enumerable
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String)
    owner_id = Column(UUID(as_uuid=True), nullable=False)
```

---

## 8. 함수 수준 인가 결함

함수 수준 인가(Function-level authorization)는 사용자가 자신의 역할에 적합한 API 엔드포인트에만 접근할 수 있도록 보장합니다. 일반 사용자가 관리자 엔드포인트를 호출할 수 없어야 합니다.

### 역할 기반 접근 제어 (RBAC)

```python
from enum import Enum
from functools import wraps
from fastapi import Depends, HTTPException


class Role(str, Enum):
    USER = "user"
    MODERATOR = "moderator"
    ADMIN = "admin"


# Permission mapping
ROLE_PERMISSIONS = {
    Role.USER: {
        "read:own_profile", "update:own_profile",
        "read:books", "create:reviews",
    },
    Role.MODERATOR: {
        "read:own_profile", "update:own_profile",
        "read:books", "create:reviews",
        "delete:reviews", "read:users",
    },
    Role.ADMIN: {
        "read:own_profile", "update:own_profile",
        "read:books", "create:reviews",
        "delete:reviews", "read:users",
        "create:books", "delete:books",
        "manage:users", "read:admin_dashboard",
    },
}


def require_permission(permission: str):
    """Dependency that checks if the current user has a specific permission."""
    async def check(current_user: User = Depends(get_current_user)):
        user_permissions = ROLE_PERMISSIONS.get(Role(current_user.role), set())
        if permission not in user_permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{permission}' required",
            )
        return current_user
    return check


# Usage: protect endpoints with specific permissions
@app.get("/admin/dashboard")
async def admin_dashboard(
    user: User = Depends(require_permission("read:admin_dashboard")),
):
    """Only admins can access this endpoint."""
    return await get_dashboard_stats()


@app.delete("/books/{book_id}")
async def delete_book(
    book_id: int,
    user: User = Depends(require_permission("delete:books")),
):
    """Only admins can delete books."""
    await db.delete(await db.get(Book, book_id))
    await db.commit()


@app.get("/users")
async def list_users(
    user: User = Depends(require_permission("read:users")),
):
    """Moderators and admins can list users."""
    return await db.execute(select(User))
```

### 권한 상승 방지

```python
@app.patch("/users/{user_id}")
async def update_user(
    user_id: int,
    update: UserUpdate,
    current_user: User = Depends(get_current_user),
):
    """Prevent users from escalating their own privileges."""
    # Only admins can change roles
    if update.role is not None and current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Only admins can change user roles",
        )

    # Users can only update their own profile (unless admin)
    if user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Cannot update other users' profiles",
        )

    # Additional check: admins cannot demote themselves
    if user_id == current_user.id and update.role and update.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Cannot demote yourself. Another admin must do this.",
        )

    ...
```

---

## 9. 보안 헤더와 강화

### 보안 헤더 미들웨어

```python
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all API responses.

    These headers instruct browsers to enforce security
    policies that mitigate common web attacks.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent clickjacking (embedding in iframes)
        response.headers["X-Frame-Options"] = "DENY"

        # Enable browser XSS filter
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Control how much referrer information is sent
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Content Security Policy (for APIs serving HTML)
        response.headers["Content-Security-Policy"] = "default-src 'none'"

        # Strict Transport Security (HTTPS only)
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )

        # Permissions Policy (disable browser features)
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )

        # Remove server identification headers
        response.headers.pop("Server", None)
        response.headers.pop("X-Powered-By", None)

        return response


app = FastAPI()
app.add_middleware(SecurityHeadersMiddleware)
```

### API 키 보안

```python
import hashlib
import secrets


def generate_api_key() -> tuple[str, str]:
    """Generate an API key and its hash.

    Returns (plaintext_key, key_hash).
    Store only the hash in the database. Show the plaintext
    to the user exactly once.
    """
    # Prefix for identification (not a secret)
    prefix = "sk_live"

    # Cryptographically secure random key
    raw_key = secrets.token_hex(32)

    # Full key shown to user
    plaintext_key = f"{prefix}_{raw_key}"

    # Hash for storage (one-way)
    key_hash = hashlib.sha256(plaintext_key.encode()).hexdigest()

    return plaintext_key, key_hash


async def verify_api_key(key: str) -> dict | None:
    """Verify an API key by hashing and looking up the hash.

    The database never stores the plaintext key, so even if
    the database is compromised, the keys cannot be used.
    """
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    result = await db.execute(
        select(ApiKey).where(
            ApiKey.key_hash == key_hash,
            ApiKey.active == True,
            ApiKey.expires_at > datetime.utcnow(),
        )
    )
    return result.scalar_one_or_none()
```

---

## 10. 연습 문제

### 문제 1: 보안 감사

다음 FastAPI 엔드포인트의 보안 취약점을 감사하세요. 최소 5가지 문제를 식별하고 수정된 코드를 제공하세요:

```python
@app.get("/search")
async def search(q: str, db=Depends(get_db)):
    results = await db.execute(f"SELECT * FROM products WHERE name LIKE '%{q}%'")
    return {"results": results.fetchall(), "query": q}

@app.post("/users")
async def create_user(data: dict):
    user = User(**data)
    db.add(user)
    await db.commit()
    return {"id": user.id, "password": user.password, "email": user.email, "ssn": user.ssn}
```

### 문제 2: BOLA 방지

다음 엔드포인트를 모두 BOLA로부터 적절히 보호하는 FastAPI 애플리케이션을 구축하세요:

- `GET /notes/{note_id}` -- 노트 조회 (소유자 또는 공유된 사용자만)
- `PUT /notes/{note_id}` -- 노트 수정 (소유자만)
- `DELETE /notes/{note_id}` -- 노트 삭제 (소유자만)
- `POST /notes/{note_id}/share` -- 다른 사용자와 노트 공유 (소유자만)
- `GET /notes` -- 현재 사용자의 노트 목록 (소유 + 공유)

사용자가 다른 사람의 노트에 접근할 수 없음, 공유된 사용자가 읽기만 가능하고 수정할 수 없음, 관리자가 모든 노트에 접근할 수 있음을 검증하는 테스트를 포함하세요.

### 문제 3: CORS 설정

세 가지 시나리오에 대한 CORS 설정을 설계하세요:

1. **단일 SPA**: `api.example.com`의 API가 `app.example.com`만 서비스
2. **멀티 테넌트**: 각 테넌트가 자체 도메인을 보유 (`tenant1.example.com`, `tenant2.example.com`)
3. **공개 API**: 모든 서드파티 웹사이트에서 사용되지만 API 키 인증이 필요

각 시나리오에 대해 FastAPI CORS 미들웨어 설정을 작성하고 각 설정이 선택된 이유를 설명하세요.

### 문제 4: 입력 검증 강화

다음을 검증하는 **결제** 엔드포인트용 Pydantic 모델을 작성하세요:

- 신용카드 번호 (Luhn 알고리즘 검사)
- 유효기간 (만료되지 않음, 10년 이후가 아님)
- CVV (3자리 또는 4자리, 로그나 응답에 절대 포함되지 않음)
- 금액 (양수, 소수점 이하 최대 2자리, 최대 $10,000)
- 통화 (ISO 4217 코드, USD, EUR, GBP만 허용)
- 청구 주소 (국가 코드 ISO 3166-1, 우편번호 형식은 국가별로 다름)

유효한 입력과 유효하지 않은 입력에 대한 테스트를 포함하고, 오류 메시지에 민감한 데이터가 나타나지 않도록 보장하세요.

### 문제 5: 보안 헤더 감사

API에 요청을 보내고 다음 보안 헤더가 존재하며 올바르게 설정되어 있는지 검증하는 pytest 테스트 스위트를 작성하세요:

- `Strict-Transport-Security` (최소 1년, 하위 도메인 포함)
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Content-Security-Policy` (비어 있지 않음)
- `Referrer-Policy` (`unsafe-url`이 아님)
- `Server` 또는 `X-Powered-By` 헤더 없음 (정보 노출)
- 모든 `Set-Cookie` 헤더에 `HttpOnly`, `Secure`, `SameSite` 포함

---

## 11. 참고 자료

- [OWASP API Security Top 10 (2023)](https://owasp.org/API-Security/editions/2023/en/0x00-header/)
- [OWASP Cheat Sheet Series](https://cheatsheetseries.owasp.org/)
- [MDN CORS Guide](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS)
- [MDN CSRF Prevention](https://developer.mozilla.org/en-US/docs/Web/Security/Practical_implementation_guides/CSRF_prevention)
- [FastAPI Security Tutorial](https://fastapi.tiangolo.com/tutorial/security/)
- [RFC 6749: OAuth 2.0](https://tools.ietf.org/html/rfc6749)
- [CWE-89: SQL Injection](https://cwe.mitre.org/data/definitions/89.html)
- [CWE-79: Cross-Site Scripting (XSS)](https://cwe.mitre.org/data/definitions/79.html)
- [SecurityHeaders.com](https://securityheaders.com/)
- [Bleach Documentation](https://bleach.readthedocs.io/)

---

**이전**: [gRPC와 Protocol Buffers](./14_gRPC_and_Protocol_Buffers.md) | [개요](./00_Overview.md) | **다음**: [API 생명주기 관리](./16_API_Lifecycle_Management.md)

**License**: CC BY-NC 4.0
