# Lesson 8: 오류 처리

**이전**: [API Versioning](07_API_Versioning.md) | [개요](00_Overview.md) | **다음**: [Rate Limiting과 Throttling](09_Rate_Limiting_and_Throttling.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. RFC 7807 Problem Details 표준을 따르는 구조화된 오류 응답 설계하기
2. 기계 파싱 가능하고 사람이 읽을 수 있는 계층적 오류 코드 시스템 구축하기
3. 풍부한 클라이언트 피드백을 위한 필드별 상세 검증 오류 포매팅하기
4. 다국어 API 소비자를 지원하기 위한 오류 현지화 구현하기
5. 일시적 오류에 대해 Retry-After 헤더와 지수 백오프 지침 사용하기
6. 전체 API에 걸쳐 일관된 오류 처리 프레임워크 생성하기

---

오류 처리는 API의 진정한 품질이 드러나는 지점입니다. 모든 것이 정상적으로 작동할 때는 어떤 API든 괜찮아 보입니다. 그러나 문제가 발생했을 때 -- 그리고 반드시 발생합니다 -- 난해한 `{"error": "Something went wrong"}`과 구조화되고 실행 가능한 오류 응답의 차이는, 개발자가 문제를 몇 분 안에 해결하느냐 아니면 몇 시간 동안 좌절하느냐의 차이입니다. 이 레슨에서는 오류를 API의 최고 기능으로 만드는 방법을 알려드립니다.

> **비유:** 오류 메시지는 도로 표지판과 같습니다. "도로 폐쇄"라고만 적힌 표지판은 도움이 되지 않습니다. "도로 폐쇄: 교량 수리. Oak Street로 우회. 재개통 예정: 3월 15일"이라고 적힌 표지판은 무슨 일이 일어났는지, 무엇을 해야 하는지, 언제 다시 시도해야 하는지를 알려줍니다. API 오류도 두 번째 종류여야 합니다.

## 목차
1. [임의적 오류의 문제점](#임의적-오류의-문제점)
2. [RFC 7807 Problem Details](#rfc-7807-problem-details)
3. [오류 코드 계층 구조](#오류-코드-계층-구조)
4. [구조화된 오류 응답](#구조화된-오류-응답)
5. [검증 오류](#검증-오류)
6. [오류 현지화](#오류-현지화)
7. [Retry-After와 일시적 오류](#retry-after와-일시적-오류)
8. [오류 처리 프레임워크](#오류-처리-프레임워크)
9. [연습 문제](#연습-문제)

---

## 임의적 오류의 문제점

### 일관성 없는 오류 형식

```python
# Real examples of inconsistent errors from the same API:

# Endpoint 1: Plain string
{"error": "Not found"}

# Endpoint 2: Nested object
{"error": {"message": "Invalid email", "code": 422}}

# Endpoint 3: Array of strings
{"errors": ["Name is required", "Email is invalid"]}

# Endpoint 4: HTTP status only (empty body)
# 500 Internal Server Error

# Endpoint 5: HTML error page
# <html><body><h1>500 Server Error</h1></body></html>

# Problems:
# - Clients cannot parse errors consistently
# - No machine-readable error codes for programmatic handling
# - No guidance on how to fix the issue
# - No request correlation for debugging
```

### 좋은 오류의 모습

```json
{
    "type": "https://api.example.com/errors/validation-failed",
    "title": "Validation Failed",
    "status": 422,
    "detail": "The request body contains 2 validation errors.",
    "instance": "/api/users",
    "request_id": "req_abc123def456",
    "errors": [
        {
            "field": "email",
            "code": "INVALID_FORMAT",
            "message": "Must be a valid email address.",
            "rejected_value": "not-an-email"
        },
        {
            "field": "age",
            "code": "OUT_OF_RANGE",
            "message": "Must be between 0 and 150.",
            "rejected_value": -5
        }
    ]
}
```

---

## RFC 7807 Problem Details

RFC 7807(RFC 9457로 업데이트됨)은 HTTP API 오류 응답을 위한 표준 형식을 정의합니다. 미디어 타입 `application/problem+json`을 사용합니다.

### 표준 필드

| 필드 | 타입 | 필수 여부 | 설명 |
|-------|------|----------|-------------|
| `type` | URI | 예 | 오류 유형을 식별하는 URI (문서 링크) |
| `title` | string | 예 | 짧은 사람이 읽을 수 있는 요약 (이 유형의 모든 인스턴스에 동일) |
| `status` | integer | 예 | HTTP 상태 코드 |
| `detail` | string | 아니요 | 이 특정 발생에 대한 사람이 읽을 수 있는 설명 |
| `instance` | URI | 아니요 | 특정 발생을 식별하는 URI (요청 경로 또는 추적 ID) |

### 기본 구현

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

app = FastAPI()

class ProblemDetail(BaseModel):
    """RFC 7807 Problem Details response."""
    type: str
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None

def problem_response(
    status: int,
    error_type: str,
    title: str,
    detail: str | None = None,
    instance: str | None = None,
    extra: dict | None = None,
) -> JSONResponse:
    """Create an RFC 7807 Problem Details response."""
    body = {
        "type": f"https://api.example.com/errors/{error_type}",
        "title": title,
        "status": status,
    }
    if detail:
        body["detail"] = detail
    if instance:
        body["instance"] = instance
    if extra:
        body.update(extra)

    return JSONResponse(
        content=body,
        status_code=status,
        media_type="application/problem+json",
    )

# Usage in endpoints
@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    if user_id > 100:
        return problem_response(
            status=404,
            error_type="resource-not-found",
            title="Resource Not Found",
            detail=f"No user exists with ID {user_id}.",
            instance=f"/api/users/{user_id}",
        )
    return {"id": user_id, "name": "Alice"}

@app.post("/api/orders")
async def create_order(request: Request):
    body = await request.json()
    if not body.get("items"):
        return problem_response(
            status=422,
            error_type="validation-failed",
            title="Validation Failed",
            detail="The order must contain at least one item.",
            instance="/api/orders",
            extra={
                "errors": [{
                    "field": "items",
                    "code": "REQUIRED",
                    "message": "At least one item is required.",
                }]
            },
        )
    return {"id": 1, "status": "created"}
```

### 전역 예외 핸들러

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import traceback
import uuid

app = FastAPI()

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Convert all HTTPExceptions to RFC 7807 format."""
    error_type = {
        400: "bad-request",
        401: "unauthorized",
        403: "forbidden",
        404: "resource-not-found",
        405: "method-not-allowed",
        409: "conflict",
        422: "validation-failed",
        429: "rate-limit-exceeded",
    }.get(exc.status_code, "error")

    body = {
        "type": f"https://api.example.com/errors/{error_type}",
        "title": exc.detail if isinstance(exc.detail, str) else error_type.replace("-", " ").title(),
        "status": exc.status_code,
        "instance": str(request.url.path),
        "request_id": f"req_{uuid.uuid4().hex[:12]}",
    }

    if isinstance(exc.detail, dict):
        body.update(exc.detail)

    return JSONResponse(
        content=body,
        status_code=exc.status_code,
        media_type="application/problem+json",
        headers=exc.headers,
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Convert Pydantic validation errors to RFC 7807 format."""
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"] if loc != "body")
        errors.append({
            "field": field,
            "code": error["type"].upper().replace(".", "_"),
            "message": error["msg"],
        })

    return JSONResponse(
        content={
            "type": "https://api.example.com/errors/validation-failed",
            "title": "Validation Failed",
            "status": 422,
            "detail": f"The request body contains {len(errors)} validation error(s).",
            "instance": str(request.url.path),
            "request_id": f"req_{uuid.uuid4().hex[:12]}",
            "errors": errors,
        },
        status_code=422,
        media_type="application/problem+json",
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all handler for unexpected errors."""
    request_id = f"req_{uuid.uuid4().hex[:12]}"

    # Log the full exception with request_id for debugging
    # logger.error(f"[{request_id}] Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        content={
            "type": "https://api.example.com/errors/internal-error",
            "title": "Internal Server Error",
            "status": 500,
            "detail": "An unexpected error occurred. Please contact support with the request_id.",
            "instance": str(request.url.path),
            "request_id": request_id,
            # NEVER expose stack traces or internal details to clients
        },
        status_code=500,
        media_type="application/problem+json",
    )
```

---

## 오류 코드 계층 구조

### 코드 시스템 설계

```python
# Error codes should be:
# 1. Machine-parseable (consistent format)
# 2. Hierarchical (general -> specific)
# 3. Documented (each code has a description and resolution)

ERROR_CODES = {
    # Authentication errors
    "AUTH_001": {
        "title": "Missing Authentication",
        "detail": "No authentication credentials were provided.",
        "resolution": "Include a valid Bearer token in the Authorization header.",
        "status": 401,
    },
    "AUTH_002": {
        "title": "Invalid Token",
        "detail": "The provided authentication token is invalid or malformed.",
        "resolution": "Obtain a new token via POST /api/auth/login.",
        "status": 401,
    },
    "AUTH_003": {
        "title": "Expired Token",
        "detail": "The authentication token has expired.",
        "resolution": "Refresh your token via POST /api/auth/refresh.",
        "status": 401,
    },
    "AUTH_004": {
        "title": "Insufficient Permissions",
        "detail": "Your token does not have the required scopes.",
        "resolution": "Request additional scopes or contact an administrator.",
        "status": 403,
    },

    # Validation errors
    "VAL_001": {
        "title": "Required Field Missing",
        "detail": "A required field was not provided.",
        "status": 422,
    },
    "VAL_002": {
        "title": "Invalid Format",
        "detail": "A field does not match the expected format.",
        "status": 422,
    },
    "VAL_003": {
        "title": "Value Out of Range",
        "detail": "A field value is outside the acceptable range.",
        "status": 422,
    },

    # Resource errors
    "RES_001": {
        "title": "Resource Not Found",
        "detail": "The requested resource does not exist.",
        "status": 404,
    },
    "RES_002": {
        "title": "Resource Conflict",
        "detail": "The operation conflicts with the current state of the resource.",
        "status": 409,
    },
    "RES_003": {
        "title": "Resource Gone",
        "detail": "The requested resource has been permanently removed.",
        "status": 410,
    },

    # Rate limiting errors
    "RATE_001": {
        "title": "Rate Limit Exceeded",
        "detail": "You have exceeded the rate limit for this endpoint.",
        "resolution": "Wait and retry after the Retry-After period.",
        "status": 429,
    },

    # Server errors
    "SRV_001": {
        "title": "Internal Server Error",
        "detail": "An unexpected error occurred on the server.",
        "resolution": "Retry the request. If the issue persists, contact support.",
        "status": 500,
    },
    "SRV_002": {
        "title": "Service Unavailable",
        "detail": "The service is temporarily unavailable.",
        "resolution": "Retry after the period specified in the Retry-After header.",
        "status": 503,
    },
}
```

### 응답에서 오류 코드 사용

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

class ApiError(HTTPException):
    """Custom exception that maps to RFC 7807 + error codes."""

    def __init__(self, code: str, detail: str | None = None, extra: dict | None = None):
        error_def = ERROR_CODES.get(code, ERROR_CODES["SRV_001"])
        self.error_code = code
        self.error_body = {
            "type": f"https://api.example.com/errors/{code}",
            "title": error_def["title"],
            "status": error_def["status"],
            "detail": detail or error_def["detail"],
            "code": code,
        }
        if "resolution" in error_def:
            self.error_body["resolution"] = error_def["resolution"]
        if extra:
            self.error_body.update(extra)

        super().__init__(status_code=error_def["status"], detail=self.error_body)

# Usage
@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    if user_id > 100:
        raise ApiError(
            code="RES_001",
            detail=f"No user found with ID {user_id}.",
            extra={"instance": f"/api/users/{user_id}"},
        )
    return {"id": user_id, "name": "Alice"}

@app.post("/api/users")
async def create_user(email: str):
    # Check for duplicate
    existing = True  # simulated
    if existing:
        raise ApiError(
            code="RES_002",
            detail=f"A user with email '{email}' already exists.",
            extra={
                "instance": "/api/users",
                "conflicting_field": "email",
                "conflicting_value": email,
            },
        )
```

---

## 구조화된 오류 응답

### 단일 오류

```python
@app.get("/api/products/{product_id}")
async def get_product(product_id: int):
    """Single error response example."""
    return JSONResponse(
        status_code=404,
        content={
            "type": "https://api.example.com/errors/resource-not-found",
            "title": "Resource Not Found",
            "status": 404,
            "detail": f"Product with ID {product_id} does not exist.",
            "instance": f"/api/products/{product_id}",
            "request_id": "req_abc123",
            "code": "RES_001",
            "resolution": "Verify the product ID and try again. Use GET /api/products to list available products.",
        },
        media_type="application/problem+json",
    )
```

### 다중 오류 (배치/검증)

```python
@app.post("/api/users")
async def create_user_with_validation(request: Request):
    """Multiple validation errors collected and returned together."""
    body = await request.json()
    errors = []

    # Validate each field
    if not body.get("name"):
        errors.append({
            "field": "name",
            "code": "VAL_001",
            "message": "Name is required.",
            "rejected_value": body.get("name"),
        })
    elif len(body.get("name", "")) < 2:
        errors.append({
            "field": "name",
            "code": "VAL_003",
            "message": "Name must be at least 2 characters long.",
            "rejected_value": body.get("name"),
            "constraint": {"min_length": 2},
        })

    email = body.get("email", "")
    if not email:
        errors.append({
            "field": "email",
            "code": "VAL_001",
            "message": "Email is required.",
            "rejected_value": None,
        })
    elif "@" not in email:
        errors.append({
            "field": "email",
            "code": "VAL_002",
            "message": "Must be a valid email address.",
            "rejected_value": email,
            "expected_format": "user@domain.com",
        })

    age = body.get("age")
    if age is not None and (age < 0 or age > 150):
        errors.append({
            "field": "age",
            "code": "VAL_003",
            "message": "Age must be between 0 and 150.",
            "rejected_value": age,
            "constraint": {"min": 0, "max": 150},
        })

    if errors:
        return JSONResponse(
            status_code=422,
            content={
                "type": "https://api.example.com/errors/validation-failed",
                "title": "Validation Failed",
                "status": 422,
                "detail": f"The request body contains {len(errors)} validation error(s).",
                "instance": "/api/users",
                "request_id": "req_xyz789",
                "errors": errors,
            },
            media_type="application/problem+json",
        )

    return JSONResponse(status_code=201, content={"id": 1, "name": body["name"]})
```

---

## 검증 오류

### Pydantic 통합

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Annotated

app = FastAPI()

class CreateUserRequest(BaseModel):
    name: str = Field(
        ..., min_length=2, max_length=100,
        description="User's full name",
        examples=["Alice Smith"],
    )
    email: str = Field(
        ..., pattern=r"^[\w.+-]+@[\w-]+\.[\w.]+$",
        description="Valid email address",
        examples=["alice@example.com"],
    )
    age: int | None = Field(
        default=None, ge=0, le=150,
        description="User's age (0-150)",
    )
    role: str = Field(
        default="member",
        description="User role",
        examples=["member"],
    )

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed = {"member", "editor", "admin"}
        if v not in allowed:
            raise ValueError(f"Role must be one of: {', '.join(sorted(allowed))}")
        return v

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower()

@app.post("/api/users", status_code=201)
async def create_user(user: CreateUserRequest):
    """
    FastAPI + Pydantic automatically validate and return 422 errors.

    The global exception handler (defined earlier) converts these
    to RFC 7807 format with per-field error details.
    """
    return {"id": 1, "name": user.name, "email": user.email}
```

### 중첩 객체 검증

```python
from pydantic import BaseModel, Field

class Address(BaseModel):
    street: str = Field(..., min_length=1)
    city: str = Field(..., min_length=1)
    state: str = Field(..., min_length=2, max_length=2)
    zip_code: str = Field(..., pattern=r"^\d{5}(-\d{4})?$")
    country: str = Field(default="US", min_length=2, max_length=2)

class CreateOrderRequest(BaseModel):
    items: list[dict] = Field(..., min_length=1)
    shipping_address: Address
    billing_address: Address | None = None
    notes: str | None = Field(default=None, max_length=500)

@app.post("/api/orders", status_code=201)
async def create_order(order: CreateOrderRequest):
    """
    Nested validation errors include the full field path:

    {
        "errors": [
            {
                "field": "shipping_address.zip_code",
                "code": "STRING_PATTERN_MISMATCH",
                "message": "String should match pattern '^\\d{5}(-\\d{4})?$'"
            }
        ]
    }
    """
    return {"id": 1, "status": "pending"}
```

---

## 오류 현지화

### 다국어 오류 메시지

```python
from fastapi import FastAPI, Header

app = FastAPI()

ERROR_MESSAGES = {
    "en": {
        "RES_001": "The requested resource was not found.",
        "VAL_001": "This field is required.",
        "VAL_002": "The value does not match the expected format.",
        "AUTH_001": "Authentication credentials are required.",
        "RATE_001": "You have exceeded the rate limit. Please try again later.",
    },
    "ko": {
        "RES_001": "요청한 리소스를 찾을 수 없습니다.",
        "VAL_001": "이 필드는 필수입니다.",
        "VAL_002": "값이 예상된 형식과 일치하지 않습니다.",
        "AUTH_001": "인증 자격 증명이 필요합니다.",
        "RATE_001": "요청 제한을 초과했습니다. 잠시 후 다시 시도해 주세요.",
    },
    "es": {
        "RES_001": "El recurso solicitado no fue encontrado.",
        "VAL_001": "Este campo es obligatorio.",
        "VAL_002": "El valor no coincide con el formato esperado.",
        "AUTH_001": "Se requieren credenciales de autenticacion.",
        "RATE_001": "Ha excedido el limite de solicitudes. Intente de nuevo mas tarde.",
    },
}

def get_localized_message(code: str, lang: str = "en") -> str:
    """Get error message in the requested language."""
    messages = ERROR_MESSAGES.get(lang, ERROR_MESSAGES["en"])
    return messages.get(code, ERROR_MESSAGES["en"].get(code, "An error occurred."))

@app.get("/api/users/{user_id}")
async def get_user(
    user_id: int,
    accept_language: str = Header(default="en", alias="Accept-Language"),
):
    """
    Localized error messages based on Accept-Language header.

    Usage:
        GET /api/users/999
        Accept-Language: ko

        Response:
        {
            "title": "Resource Not Found",
            "detail": "요청한 리소스를 찾을 수 없습니다.",
            "code": "RES_001"
        }
    """
    # Parse Accept-Language (simplified -- use a library for production)
    lang = accept_language.split(",")[0].split("-")[0].strip().lower()
    if lang not in ERROR_MESSAGES:
        lang = "en"

    if user_id > 100:
        return JSONResponse(
            status_code=404,
            content={
                "type": "https://api.example.com/errors/resource-not-found",
                "title": "Resource Not Found",
                "status": 404,
                "detail": get_localized_message("RES_001", lang),
                "code": "RES_001",
                "instance": f"/api/users/{user_id}",
            },
            media_type="application/problem+json",
        )

    return {"id": user_id, "name": "Alice"}
```

### 현지화된 검증 오류

```python
FIELD_MESSAGES = {
    "en": {
        "required": "{field} is required.",
        "min_length": "{field} must be at least {min} characters.",
        "max_length": "{field} must be at most {max} characters.",
        "invalid_email": "{field} must be a valid email address.",
        "out_of_range": "{field} must be between {min} and {max}.",
    },
    "ko": {
        "required": "{field}은(는) 필수입니다.",
        "min_length": "{field}은(는) 최소 {min}자 이상이어야 합니다.",
        "max_length": "{field}은(는) 최대 {max}자까지 가능합니다.",
        "invalid_email": "{field}은(는) 유효한 이메일 주소여야 합니다.",
        "out_of_range": "{field}은(는) {min}에서 {max} 사이여야 합니다.",
    },
}

def localized_field_error(
    field: str, error_type: str, lang: str = "en", **kwargs
) -> str:
    """Generate a localized field validation message."""
    messages = FIELD_MESSAGES.get(lang, FIELD_MESSAGES["en"])
    template = messages.get(error_type, "{field}: validation error")
    return template.format(field=field, **kwargs)

# Usage:
# localized_field_error("email", "required", "ko")
# -> "email은(는) 필수입니다."
# localized_field_error("name", "min_length", "en", min=2)
# -> "name must be at least 2 characters."
```

---

## Retry-After와 일시적 오류

### Retry-After 헤더

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from datetime import datetime, timezone, timedelta

app = FastAPI()

@app.get("/api/search")
async def search():
    """Rate limit exceeded -- tell client when to retry."""
    retry_after_seconds = 60
    retry_at = datetime.now(timezone.utc) + timedelta(seconds=retry_after_seconds)

    return JSONResponse(
        status_code=429,
        content={
            "type": "https://api.example.com/errors/rate-limit-exceeded",
            "title": "Rate Limit Exceeded",
            "status": 429,
            "detail": "You have made too many requests. Please wait before trying again.",
            "code": "RATE_001",
            "retry_after": retry_after_seconds,
            "retry_at": retry_at.isoformat(),
            "rate_limit": {
                "limit": 100,
                "remaining": 0,
                "reset": retry_at.isoformat(),
            },
        },
        headers={
            "Retry-After": str(retry_after_seconds),
            "X-RateLimit-Limit": "100",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(retry_at.timestamp())),
        },
        media_type="application/problem+json",
    )

@app.get("/api/external-data")
async def get_external_data():
    """Upstream service unavailable -- suggest retry with backoff."""
    return JSONResponse(
        status_code=503,
        content={
            "type": "https://api.example.com/errors/service-unavailable",
            "title": "Service Unavailable",
            "status": 503,
            "detail": "The upstream data provider is temporarily unavailable.",
            "code": "SRV_002",
            "retry_after": 30,
            "backoff_strategy": "exponential",
            "max_retries": 3,
        },
        headers={
            "Retry-After": "30",
        },
        media_type="application/problem+json",
    )
```

### 클라이언트 측 지수 백오프 재시도

```python
import httpx
import asyncio
import random

async def api_call_with_retry(
    url: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> dict:
    """
    Make an API call with exponential backoff on transient errors.

    Retry on: 429 (rate limited), 500, 502, 503, 504
    Do NOT retry on: 400, 401, 403, 404, 422
    """
    retryable_statuses = {429, 500, 502, 503, 504}

    async with httpx.AsyncClient() as client:
        for attempt in range(max_retries + 1):
            try:
                response = await client.get(url)

                if response.status_code < 400:
                    return response.json()

                if response.status_code not in retryable_statuses:
                    # Non-retryable error -- fail immediately
                    return {"error": response.json(), "status": response.status_code}

                if attempt == max_retries:
                    return {"error": response.json(), "status": response.status_code}

                # Calculate delay
                if response.status_code == 429:
                    # Use server's Retry-After if available
                    retry_after = response.headers.get("retry-after")
                    if retry_after:
                        delay = float(retry_after)
                    else:
                        delay = base_delay * (2 ** attempt)
                else:
                    # Exponential backoff with jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    delay += random.uniform(0, delay * 0.1)  # add 10% jitter

                print(f"Attempt {attempt + 1} failed ({response.status_code}). "
                      f"Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)

            except httpx.RequestError as e:
                if attempt == max_retries:
                    raise
                delay = base_delay * (2 ** attempt)
                print(f"Connection error: {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)

    return {"error": "Max retries exceeded"}
```

---

## 오류 처리 프레임워크

### 완전한 오류 프레임워크

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from typing import Any
import uuid
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

# --- Custom Exception Classes ---

class ApiException(Exception):
    """Base exception for all API errors."""

    def __init__(
        self,
        code: str,
        status: int,
        title: str,
        detail: str | None = None,
        errors: list[dict] | None = None,
        headers: dict | None = None,
    ):
        self.code = code
        self.status = status
        self.title = title
        self.detail = detail
        self.errors = errors
        self.headers = headers or {}
        super().__init__(detail or title)

class NotFoundError(ApiException):
    def __init__(self, resource: str, resource_id: Any):
        super().__init__(
            code="RES_001",
            status=404,
            title="Resource Not Found",
            detail=f"{resource} with ID '{resource_id}' does not exist.",
        )

class ConflictError(ApiException):
    def __init__(self, detail: str):
        super().__init__(
            code="RES_002",
            status=409,
            title="Conflict",
            detail=detail,
        )

class ValidationError(ApiException):
    def __init__(self, errors: list[dict]):
        super().__init__(
            code="VAL_000",
            status=422,
            title="Validation Failed",
            detail=f"The request contains {len(errors)} validation error(s).",
            errors=errors,
        )

class AuthenticationError(ApiException):
    def __init__(self, detail: str = "Authentication required."):
        super().__init__(
            code="AUTH_001",
            status=401,
            title="Unauthorized",
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )

class ForbiddenError(ApiException):
    def __init__(self, detail: str = "You do not have permission to perform this action."):
        super().__init__(
            code="AUTH_004",
            status=403,
            title="Forbidden",
            detail=detail,
        )

class RateLimitError(ApiException):
    def __init__(self, retry_after: int = 60):
        super().__init__(
            code="RATE_001",
            status=429,
            title="Rate Limit Exceeded",
            detail=f"Too many requests. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

# --- Exception Handlers ---

@app.exception_handler(ApiException)
async def api_exception_handler(request: Request, exc: ApiException):
    """Handle all custom API exceptions."""
    request_id = f"req_{uuid.uuid4().hex[:12]}"

    body = {
        "type": f"https://api.example.com/errors/{exc.code}",
        "title": exc.title,
        "status": exc.status,
        "detail": exc.detail,
        "code": exc.code,
        "instance": str(request.url.path),
        "request_id": request_id,
    }
    if exc.errors:
        body["errors"] = exc.errors

    logger.warning(
        f"[{request_id}] {exc.code}: {exc.detail}",
        extra={"path": request.url.path, "method": request.method},
    )

    return JSONResponse(
        content=body,
        status_code=exc.status,
        headers=exc.headers,
        media_type="application/problem+json",
    )

@app.exception_handler(RequestValidationError)
async def pydantic_validation_handler(request: Request, exc: RequestValidationError):
    """Convert Pydantic errors to our error format."""
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    errors = []
    for error in exc.errors():
        field_path = ".".join(str(loc) for loc in error["loc"] if loc != "body")
        errors.append({
            "field": field_path or "(root)",
            "code": error["type"].upper().replace(".", "_"),
            "message": error["msg"],
        })

    return JSONResponse(
        content={
            "type": "https://api.example.com/errors/validation-failed",
            "title": "Validation Failed",
            "status": 422,
            "detail": f"The request contains {len(errors)} validation error(s).",
            "code": "VAL_000",
            "instance": str(request.url.path),
            "request_id": request_id,
            "errors": errors,
        },
        status_code=422,
        media_type="application/problem+json",
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions."""
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    logger.error(f"[{request_id}] Unhandled: {exc}", exc_info=True)

    return JSONResponse(
        content={
            "type": "https://api.example.com/errors/internal-error",
            "title": "Internal Server Error",
            "status": 500,
            "detail": "An unexpected error occurred. Please try again or contact support.",
            "code": "SRV_001",
            "instance": str(request.url.path),
            "request_id": request_id,
        },
        status_code=500,
        media_type="application/problem+json",
    )

# --- Usage in Endpoints ---

@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    if user_id > 100:
        raise NotFoundError("User", user_id)
    return {"id": user_id, "name": "Alice"}

@app.post("/api/users")
async def create_user(email: str):
    raise ConflictError(f"A user with email '{email}' already exists.")

@app.delete("/api/admin/data")
async def admin_delete():
    raise ForbiddenError("Only super-admins can delete data.")
```

---

## 연습 문제

### 연습 문제 1: RFC 7807 구현

기존 API를 모든 오류에 RFC 7807 Problem Details를 사용하도록 변환하십시오:
- `HTTPException`, `RequestValidationError`, `Exception`에 대한 전역 예외 핸들러 생성
- `application/problem+json` 미디어 타입 사용
- 모든 오류에 `type`, `title`, `status`, `detail`, `instance`, `request_id` 포함
- 스택 트레이스나 내부 세부 정보가 클라이언트에 노출되지 않도록 보장

### 연습 문제 2: 오류 코드 레지스트리

오류 코드 레지스트리 시스템을 구축하십시오:
- 카테고리(AUTH, VAL, RES, RATE, SRV) 전반에 걸쳐 최소 15개의 오류 코드 정의
- 모든 오류 코드와 설명을 나열하는 `/api/errors` 엔드포인트 생성
- 특정 코드에 대한 전체 문서를 반환하는 `/api/errors/{code}` 엔드포인트 생성
- 각 오류 코드에 대한 해결 지침 포함

### 연습 문제 3: 검증 오류 포매터

다음 기능을 가진 검증 미들웨어를 생성하십시오:
- 모든 검증 오류를 수집 (첫 번째에서 멈추지 않음)
- 거부된 값과 함께 필드별 오류 세부 정보 반환
- 중첩 객체 검증 지원 (예: `shipping_address.zip_code`)
- 각 오류에 대한 제약 조건 정보(min, max, pattern) 포함

### 연습 문제 4: 현지화된 오류

오류 현지화를 구현하십시오:
- 영어, 한국어, 추가 1개 언어 지원
- 응답 언어를 결정하기 위해 `Accept-Language` 헤더 파싱
- 오류 제목과 필드 수준 검증 메시지 모두 현지화
- 지원하지 않는 언어에 대해 영어로 대체
- 각 오류 코드에 번역이 있는지 확인하는 테스트 스위트 생성

### 연습 문제 5: 재시도 프레임워크

다음 기능을 가진 클라이언트 측 재시도 프레임워크를 구축하십시오:
- `Retry-After` 헤더 준수
- 지터가 포함된 지수 백오프 구현
- 일시적 오류(429, 500, 502, 503, 504)에서만 재시도
- 클라이언트 오류(400, 401, 403, 404, 422)에서는 즉시 실패
- 각 재시도 시도를 지연 시간과 사유와 함께 로깅
- 설정 가능한 최대 재시도 횟수와 최대 지연 시간

---

## 요약

이 레슨에서 다룬 내용:
1. RFC 7807 Problem Details: 구조화된 API 오류 응답을 위한 표준 형식
2. 오류 코드 계층 구조: 카테고리별로 구성된 기계 판독 가능한 코드 (AUTH, VAL, RES, RATE, SRV)
3. 구조화된 오류 응답: 단일 오류, 다중 검증 오류, 중첩 필드 오류
4. 검증 오류: Pydantic 통합, 필드별 세부 정보, 거부된 값, 제약 조건
5. 오류 현지화: Accept-Language 헤더를 통한 다국어 오류 메시지
6. Retry-After 헤더: 일시적 실패 시 재시도 시점에 대한 클라이언트 안내
7. 커스텀 예외와 전역 핸들러를 갖춘 완전한 오류 처리 프레임워크

---

**이전**: [API Versioning](07_API_Versioning.md) | [개요](00_Overview.md) | **다음**: [Rate Limiting과 Throttling](09_Rate_Limiting_and_Throttling.md)

**License**: CC BY-NC 4.0
