# Lesson 8: Error Handling

**Previous**: [API Versioning](07_API_Versioning.md) | [Overview](00_Overview.md) | **Next**: [Rate Limiting and Throttling](09_Rate_Limiting_and_Throttling.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Design structured error responses following the RFC 7807 Problem Details standard
2. Build a hierarchical error code system that is machine-parseable and human-readable
3. Format validation errors with per-field detail for rich client feedback
4. Implement error localization to support multi-language API consumers
5. Use Retry-After headers and exponential backoff guidance for transient errors
6. Create a consistent error handling framework across an entire API

---

Error handling is where the true quality of an API reveals itself. When everything works, any API feels fine. But when things go wrong -- and they will -- the difference between a cryptic `{"error": "Something went wrong"}` and a structured, actionable error response is the difference between a developer solving their problem in minutes versus spending hours in frustration. This lesson teaches you to make errors your API's best feature.

> **Analogy:** Error messages are like road signs. A sign that says "Road Closed" is unhelpful. A sign that says "Road Closed: Bridge Repair. Detour via Oak Street. Expected reopening: March 15" tells you what happened, what to do, and when to try again. Your API errors should be the second kind.

## Table of Contents
1. [The Problem with Ad-Hoc Errors](#the-problem-with-ad-hoc-errors)
2. [RFC 7807 Problem Details](#rfc-7807-problem-details)
3. [Error Code Hierarchy](#error-code-hierarchy)
4. [Structured Error Responses](#structured-error-responses)
5. [Validation Errors](#validation-errors)
6. [Error Localization](#error-localization)
7. [Retry-After and Transient Errors](#retry-after-and-transient-errors)
8. [Error Handling Framework](#error-handling-framework)
9. [Exercises](#exercises)

---

## The Problem with Ad-Hoc Errors

### Inconsistent Error Formats

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

### What a Good Error Looks Like

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

RFC 7807 (updated by RFC 9457) defines a standard format for HTTP API error responses. It uses the media type `application/problem+json`.

### Standard Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | URI | Yes | A URI that identifies the error type (documentation link) |
| `title` | string | Yes | A short, human-readable summary (same for all instances of this type) |
| `status` | integer | Yes | The HTTP status code |
| `detail` | string | No | A human-readable explanation specific to this occurrence |
| `instance` | URI | No | A URI that identifies the specific occurrence (request path or tracking ID) |

### Basic Implementation

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

### Global Exception Handlers

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

## Error Code Hierarchy

### Designing a Code System

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

### Using Error Codes in Responses

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

## Structured Error Responses

### Single Error

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

### Multiple Errors (Batch/Validation)

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

## Validation Errors

### Pydantic Integration

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

### Nested Object Validation

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

## Error Localization

### Multi-Language Error Messages

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

### Localized Validation Errors

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

## Retry-After and Transient Errors

### Retry-After Header

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

### Client-Side Retry with Exponential Backoff

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

## Error Handling Framework

### Complete Error Framework

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

## Exercises

### Exercise 1: RFC 7807 Implementation

Convert an existing API to use RFC 7807 Problem Details for all errors:
- Create global exception handlers for `HTTPException`, `RequestValidationError`, and `Exception`
- Use `application/problem+json` media type
- Include `type`, `title`, `status`, `detail`, `instance`, and `request_id` in every error
- Ensure no stack traces or internal details leak to clients

### Exercise 2: Error Code Registry

Build an error code registry system:
- Define at least 15 error codes across categories (AUTH, VAL, RES, RATE, SRV)
- Create a `/api/errors` endpoint that lists all error codes with descriptions
- Create a `/api/errors/{code}` endpoint that returns full documentation for a specific code
- Include resolution guidance for each error code

### Exercise 3: Validation Error Formatter

Create a validation middleware that:
- Collects ALL validation errors (does not stop at the first one)
- Returns per-field error details with rejected values
- Supports nested object validation (e.g., `shipping_address.zip_code`)
- Includes constraint information (min, max, pattern) for each error

### Exercise 4: Localized Errors

Implement error localization:
- Support English, Korean, and one additional language
- Parse the `Accept-Language` header to determine the response language
- Localize both error titles and field-level validation messages
- Fall back to English for unsupported languages
- Create a test suite that verifies each error code has translations

### Exercise 5: Retry Framework

Build a client-side retry framework that:
- Respects `Retry-After` headers
- Implements exponential backoff with jitter
- Only retries on transient errors (429, 500, 502, 503, 504)
- Immediately fails on client errors (400, 401, 403, 404, 422)
- Logs each retry attempt with delay and reason
- Has configurable max retries and max delay

---

## Summary

This lesson covered:
1. RFC 7807 Problem Details: a standard format for structured API error responses
2. Error code hierarchy: machine-readable codes organized by category (AUTH, VAL, RES, RATE, SRV)
3. Structured error responses: single errors, multiple validation errors, nested field errors
4. Validation errors: Pydantic integration, per-field details, rejected values, and constraints
5. Error localization: multi-language error messages via Accept-Language header
6. Retry-After headers: guiding clients on when to retry transient failures
7. A complete error handling framework with custom exceptions and global handlers

---

**Previous**: [API Versioning](07_API_Versioning.md) | [Overview](00_Overview.md) | **Next**: [Rate Limiting and Throttling](09_Rate_Limiting_and_Throttling.md)

**License**: CC BY-NC 4.0
