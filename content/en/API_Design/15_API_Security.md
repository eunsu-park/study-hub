# 15. API Security

**Previous**: [gRPC and Protocol Buffers](./14_gRPC_and_Protocol_Buffers.md) | **Next**: [API Lifecycle Management](./16_API_Lifecycle_Management.md)

**Difficulty**: ⭐⭐⭐

---

## Learning Objectives

- Identify and mitigate the OWASP API Security Top 10 vulnerabilities
- Implement robust input validation to prevent injection attacks and data corruption
- Configure CORS policies that balance security with usability for cross-origin API access
- Apply CSRF protection strategies for APIs that use cookie-based authentication
- Prevent sensitive data exposure through proper response filtering and logging practices
- Implement function-level and object-level authorization checks in API endpoints

---

## Table of Contents

1. [OWASP API Security Top 10](#1-owasp-api-security-top-10)
2. [Input Validation](#2-input-validation)
3. [CORS (Cross-Origin Resource Sharing)](#3-cors-cross-origin-resource-sharing)
4. [CSRF Protection](#4-csrf-protection)
5. [Injection Prevention](#5-injection-prevention)
6. [Sensitive Data Exposure](#6-sensitive-data-exposure)
7. [Broken Object-Level Authorization (BOLA)](#7-broken-object-level-authorization-bola)
8. [Broken Function-Level Authorization](#8-broken-function-level-authorization)
9. [Security Headers and Hardening](#9-security-headers-and-hardening)
10. [Exercises](#10-exercises)
11. [References](#11-references)

---

## 1. OWASP API Security Top 10

The OWASP API Security Top 10 (2023 edition) identifies the most critical security risks for APIs. Understanding these risks is the foundation of secure API design.

| # | Risk | Description |
|---|------|-------------|
| API1 | Broken Object-Level Authorization | Accessing other users' data by manipulating object IDs |
| API2 | Broken Authentication | Weak or missing authentication mechanisms |
| API3 | Broken Object Property-Level Authorization | Exposing or accepting sensitive object properties |
| API4 | Unrestricted Resource Consumption | No limits on request size, rate, or resource usage |
| API5 | Broken Function-Level Authorization | Accessing admin functions as a regular user |
| API6 | Unrestricted Access to Sensitive Business Flows | Automating flows meant for humans (ticket scalping, credential stuffing) |
| API7 | Server-Side Request Forgery (SSRF) | Making the server request internal resources |
| API8 | Security Misconfiguration | Default credentials, verbose errors, missing headers |
| API9 | Improper Inventory Management | Untracked, deprecated, or shadow APIs |
| API10 | Unsafe Consumption of Third-Party APIs | Trusting external API data without validation |

---

## 2. Input Validation

Every piece of data from the client is untrusted. Validate everything: path parameters, query parameters, headers, and request bodies.

### Defense in Depth

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

### Parameter Validation

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

### Request Size Limits

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

CORS controls which websites can call your API from a browser. Without proper CORS configuration, any website could make authenticated requests on behalf of your users.

### How CORS Works

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

### FastAPI CORS Configuration

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

### Common CORS Mistakes

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

### Dynamic CORS for Multi-Tenant

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

## 4. CSRF Protection

Cross-Site Request Forgery (CSRF) tricks an authenticated user's browser into making unwanted requests. APIs that use cookie-based authentication are vulnerable.

### Why APIs Need CSRF Protection

```
1. User logs into https://app.example.com (session cookie set)
2. User visits https://evil-site.com
3. Evil site contains: <form action="https://api.example.com/transfer" method="POST">
4. Form auto-submits with the user's session cookie
5. API processes the request as if the user intended it
```

### CSRF Token Pattern

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

### SameSite Cookies

Modern browsers support the `SameSite` cookie attribute, which provides built-in CSRF protection:

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

| SameSite Value | Behavior | CSRF Protection |
|----------------|----------|-----------------|
| `Strict` | Cookie never sent cross-site | Full (but may break legitimate links) |
| `Lax` | Cookie sent on top-level GET navigations only | Good (recommended default) |
| `None` | Cookie always sent (requires `Secure`) | None (must use CSRF tokens) |

---

## 5. Injection Prevention

### SQL Injection

Never construct SQL queries by concatenating user input.

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

### NoSQL Injection

MongoDB and other NoSQL databases are also vulnerable:

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

### Command Injection

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

## 6. Sensitive Data Exposure

### Response Filtering

Never return more data than the client needs:

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

### Logging Sanitization

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

### Error Message Security

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

## 7. Broken Object-Level Authorization (BOLA)

BOLA (also known as IDOR — Insecure Direct Object Reference) is the #1 API security risk. It occurs when users can access other users' data by changing the resource ID.

### Vulnerable Endpoint

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

### Secure Implementation

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

### Use UUIDs Instead of Sequential IDs

Sequential integer IDs are easy to enumerate. UUIDs are much harder to guess:

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

## 8. Broken Function-Level Authorization

Function-level authorization ensures that users can only access API endpoints appropriate for their role. Regular users should not be able to call admin endpoints.

### Role-Based Access Control (RBAC)

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

### Preventing Privilege Escalation

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

## 9. Security Headers and Hardening

### Security Headers Middleware

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

### API Key Security

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

## 10. Exercises

### Exercise 1: Security Audit

Audit the following FastAPI endpoint for security vulnerabilities. Identify at least 5 issues and provide the corrected code:

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

### Exercise 2: BOLA Prevention

Build a FastAPI application with the following endpoints, all properly protected against BOLA:

- `GET /notes/{note_id}` — Get a note (owner or shared users only)
- `PUT /notes/{note_id}` — Update a note (owner only)
- `DELETE /notes/{note_id}` — Delete a note (owner only)
- `POST /notes/{note_id}/share` — Share a note with another user (owner only)
- `GET /notes` — List the current user's notes (owned + shared)

Include tests that verify: users cannot access others' notes, shared users can read but not modify, and admins can access all notes.

### Exercise 3: CORS Configuration

Design CORS configurations for three scenarios:

1. **Single SPA**: Your API at `api.example.com` serves `app.example.com` only
2. **Multi-tenant**: Each tenant has their own domain (`tenant1.example.com`, `tenant2.example.com`)
3. **Public API**: Your API is consumed by any third-party website, but requires API key authentication

For each scenario, write the FastAPI CORS middleware configuration and explain why each setting is chosen.

### Exercise 4: Input Validation Hardening

Create Pydantic models for a **Payment** endpoint that validates:

- Credit card number (Luhn algorithm check)
- Expiration date (not expired, not more than 10 years in the future)
- CVV (3 or 4 digits, never logged or returned in responses)
- Amount (positive, max 2 decimal places, maximum $10,000)
- Currency (ISO 4217 code, allow only USD, EUR, GBP)
- Billing address (country code ISO 3166-1, ZIP code format varies by country)

Include tests for valid and invalid inputs, and ensure no sensitive data appears in error messages.

### Exercise 5: Security Headers Audit

Write a pytest test suite that makes requests to your API and verifies the following security headers are present and correctly configured:

- `Strict-Transport-Security` (min 1 year, includes subdomains)
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Content-Security-Policy` (not empty)
- `Referrer-Policy` (not `unsafe-url`)
- No `Server` or `X-Powered-By` header (information disclosure)
- All `Set-Cookie` headers include `HttpOnly`, `Secure`, and `SameSite`

---

## 11. References

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

**Previous**: [gRPC and Protocol Buffers](./14_gRPC_and_Protocol_Buffers.md) | [Overview](./00_Overview.md) | **Next**: [API Lifecycle Management](./16_API_Lifecycle_Management.md)

**License**: CC BY-NC 4.0
