# Project: Building a Secure REST API

**Previous**: [14. Incident Response and Forensics](14_Incident_Response.md) | **Next**: [16. Project: Building a Vulnerability Scanner](16_Project_Vulnerability_Scanner.md)

---

This project lesson walks through building a production-ready secure REST API from scratch using FastAPI. We will implement every security layer -- from proper password hashing with Argon2, to JWT authentication with refresh tokens, role-based access control, input validation, rate limiting, security headers, CORS configuration, structured logging, and safe error handling. By the end, you will have a complete, deployable API that follows security best practices.

## Learning Objectives

- Build a REST API with security as a first-class concern
- Implement Argon2 password hashing with proper configuration
- Create JWT authentication with access and refresh token rotation
- Design and implement role-based access control (RBAC)
- Validate all input using Pydantic models
- Add rate limiting, security headers, and CORS
- Implement structured security logging
- Handle errors without leaking sensitive information
- Write security-focused tests
- Prepare for production deployment

---

## 1. Project Overview and Architecture

### 1.1 What We Are Building

```
┌─────────────────────────────────────────────────────────────────┐
│                  Secure REST API Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Client Request                                                  │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────────┐                                           │
│  │  Rate Limiter     │  ← Prevent brute force / DoS             │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  Security Headers │  ← HSTS, CSP, X-Frame-Options           │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  CORS Middleware  │  ← Cross-origin request control          │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  Authentication   │  ← JWT verification                      │
│  │  Middleware       │                                           │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  Authorization    │  ← RBAC permission check                 │
│  │  (RBAC)          │                                           │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  Input Validation │  ← Pydantic schema validation            │
│  │  (Pydantic)      │                                           │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  Business Logic   │  ← Route handlers                        │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  Database Layer   │  ← Parameterized queries only            │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  Security Logger  │  ← Audit trail (no PII in logs)          │
│  └──────────────────┘                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Project Structure

```
secure_api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration management
│   ├── database.py          # Database setup (SQLAlchemy)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py          # User database model
│   │   └── token.py         # Refresh token model
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── user.py          # Pydantic user schemas
│   │   ├── auth.py          # Auth request/response schemas
│   │   └── common.py        # Shared schemas
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── auth.py          # Authentication endpoints
│   │   ├── users.py         # User management endpoints
│   │   └── admin.py         # Admin-only endpoints
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── security_headers.py
│   │   ├── rate_limiter.py
│   │   └── request_logging.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── auth_service.py  # Authentication logic
│   │   ├── user_service.py  # User CRUD operations
│   │   └── token_service.py # JWT creation/validation
│   └── utils/
│       ├── __init__.py
│       ├── password.py      # Argon2 hashing
│       ├── security.py      # Security helpers
│       └── logging.py       # Structured logging
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Test fixtures
│   ├── test_auth.py         # Auth endpoint tests
│   ├── test_users.py        # User endpoint tests
│   └── test_security.py     # Security-specific tests
├── alembic/                 # Database migrations
│   └── ...
├── requirements.txt
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

### 1.3 Dependencies

```
# requirements.txt
fastapi==0.109.2
uvicorn[standard]==0.27.1
sqlalchemy==2.0.27
alembic==1.13.1
asyncpg==0.29.0
python-jose[cryptography]==3.3.0
argon2-cffi==23.1.0
pydantic[email]==2.6.1
pydantic-settings==2.1.0
python-multipart==0.0.9
slowapi==0.1.9
structlog==24.1.0
httpx==0.27.0
pytest==8.0.1
pytest-asyncio==0.23.5
```

---

## 2. Configuration Management

### 2.1 Secure Configuration

```python
"""
app/config.py - Application configuration.
All secrets come from environment variables, never hardcoded.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Never hardcode secrets - use .env file for development,
    environment variables or secret manager for production.
    """

    # Application
    APP_NAME: str = "Secure API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"  # development, staging, production

    # Database
    DATABASE_URL: str = Field(
        ...,  # Required - must be provided
        description="PostgreSQL connection string"
    )

    # JWT Configuration
    JWT_SECRET_KEY: str = Field(
        ...,  # Required
        min_length=32,
        description="Secret key for JWT signing (min 32 chars)"
    )
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15      # Short-lived
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7         # Longer-lived

    # Password Hashing (Argon2)
    ARGON2_TIME_COST: int = 3          # Number of iterations
    ARGON2_MEMORY_COST: int = 65536    # Memory in KiB (64 MB)
    ARGON2_PARALLELISM: int = 4        # Number of threads
    ARGON2_HASH_LENGTH: int = 32       # Hash output length
    ARGON2_SALT_LENGTH: int = 16       # Salt length

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    LOGIN_RATE_LIMIT: str = "5/minute"     # Stricter for login
    REGISTER_RATE_LIMIT: str = "3/minute"  # Stricter for registration

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["GET", "POST", "PUT", "DELETE"]
    CORS_ALLOW_HEADERS: list[str] = ["Authorization", "Content-Type"]

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or console

    # Security
    ALLOWED_HOSTS: list[str] = ["localhost", "127.0.0.1"]
    SECURE_COOKIES: bool = True
    MAX_LOGIN_ATTEMPTS: int = 5
    ACCOUNT_LOCKOUT_MINUTES: int = 30

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
```

### 2.2 Environment File

```bash
# .env.example - Copy to .env and fill in values
# NEVER commit .env to version control!

# Application
APP_NAME="Secure API"
DEBUG=false
ENVIRONMENT=development

# Database (use a strong password)
DATABASE_URL=postgresql+asyncpg://user:strong_password_here@localhost:5432/secure_api

# JWT (generate with: python -c "import secrets; print(secrets.token_hex(32))")
JWT_SECRET_KEY=your-secret-key-at-least-32-characters-long-change-this

# CORS
CORS_ORIGINS=["http://localhost:3000","http://localhost:8080"]

# Logging
LOG_LEVEL=INFO
```

---

## 3. Password Hashing with Argon2

### 3.1 Why Argon2?

```
┌─────────────────────────────────────────────────────────────────┐
│              Password Hashing Algorithm Comparison                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Algorithm      Memory   GPU       Status                       │
│                 Hard?    Resistant?                               │
│  ─────────────  ───────  ────────  ──────────────────────       │
│  MD5            No       No        BROKEN - never use            │
│  SHA-256        No       No        NOT for passwords             │
│  bcrypt         No       Partial   Good, but aging               │
│  scrypt         Yes      Yes       Good                          │
│  Argon2id       Yes      Yes       RECOMMENDED (PHC winner)     │
│                                                                  │
│  Argon2 variants:                                                │
│  - Argon2d:  Data-dependent (GPU resistant, side-channel risk)  │
│  - Argon2i:  Data-independent (side-channel safe, less GPU-R)   │
│  - Argon2id: Hybrid (RECOMMENDED - best of both)                │
│                                                                  │
│  Argon2id is the winner of the Password Hashing Competition     │
│  (PHC) and is recommended by OWASP for password storage.        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Password Hashing Implementation

```python
"""
app/utils/password.py - Secure password hashing with Argon2id.
"""

from argon2 import PasswordHasher
from argon2.exceptions import (
    VerifyMismatchError,
    VerificationError,
    InvalidHashError,
)
from app.config import get_settings

settings = get_settings()

# Configure Argon2id hasher with secure parameters
# OWASP recommends: time_cost=2, memory_cost=19456 (19 MiB) minimum
# We use stronger parameters for better security
_hasher = PasswordHasher(
    time_cost=settings.ARGON2_TIME_COST,       # iterations
    memory_cost=settings.ARGON2_MEMORY_COST,   # KiB (64 MB)
    parallelism=settings.ARGON2_PARALLELISM,   # threads
    hash_len=settings.ARGON2_HASH_LENGTH,      # output length
    salt_len=settings.ARGON2_SALT_LENGTH,       # salt length
    type=2,                                     # 2 = Argon2id
)


def hash_password(password: str) -> str:
    """
    Hash a password using Argon2id.

    Args:
        password: The plaintext password to hash.

    Returns:
        Argon2id hash string including algorithm parameters and salt.

    Example output:
        $argon2id$v=19$m=65536,t=3,p=4$c2FsdHNhbHQ$hash...
    """
    return _hasher.hash(password)


def verify_password(hash: str, password: str) -> bool:
    """
    Verify a password against an Argon2id hash.

    This function is constant-time to prevent timing attacks.

    Args:
        hash: The stored Argon2id hash string.
        password: The plaintext password to verify.

    Returns:
        True if password matches, False otherwise.
    """
    try:
        return _hasher.verify(hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def check_needs_rehash(hash: str) -> bool:
    """
    Check if a hash needs to be re-computed with updated parameters.

    Call this after successful password verification. If parameters
    have been updated (stronger settings), rehash the password.

    Args:
        hash: The stored Argon2id hash string.

    Returns:
        True if the hash should be recomputed.
    """
    return _hasher.check_needs_rehash(hash)


# ─── Password Strength Validation ───

import re

class PasswordStrengthError(ValueError):
    """Raised when password does not meet strength requirements."""
    pass


def validate_password_strength(password: str) -> None:
    """
    Validate password meets minimum strength requirements.

    Requirements:
    - Minimum 8 characters (NIST recommends allowing up to 64)
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    - Not a commonly breached password

    Raises:
        PasswordStrengthError if password is too weak.
    """
    if len(password) < 8:
        raise PasswordStrengthError(
            "Password must be at least 8 characters long"
        )

    if len(password) > 128:
        raise PasswordStrengthError(
            "Password must not exceed 128 characters"
        )

    if not re.search(r'[A-Z]', password):
        raise PasswordStrengthError(
            "Password must contain at least one uppercase letter"
        )

    if not re.search(r'[a-z]', password):
        raise PasswordStrengthError(
            "Password must contain at least one lowercase letter"
        )

    if not re.search(r'\d', password):
        raise PasswordStrengthError(
            "Password must contain at least one digit"
        )

    if not re.search(r'[!@#$%^&*()_+\-=\[\]{};:\'",.<>?/\\|`~]', password):
        raise PasswordStrengthError(
            "Password must contain at least one special character"
        )

    # Check against common passwords (simplified list)
    # In production, use a larger list (e.g., Have I Been Pwned API)
    COMMON_PASSWORDS = {
        'password', 'password1', '12345678', 'qwerty123',
        'admin123', 'letmein1', 'welcome1', 'monkey123',
        'dragon12', 'master12', 'password123', 'abc12345',
    }

    if password.lower() in COMMON_PASSWORDS:
        raise PasswordStrengthError(
            "This password is too common. Please choose a stronger password."
        )
```

---

## 4. JWT Authentication with Refresh Tokens

### 4.1 Token Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                JWT Token Architecture                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Access Token (short-lived: 15 min)                             │
│  ┌─────────────────────────────────────────────────┐            │
│  │  Header: {"alg": "HS256", "typ": "JWT"}        │            │
│  │  Payload: {                                      │            │
│  │    "sub": "user-uuid",                           │            │
│  │    "role": "user",                               │            │
│  │    "type": "access",                             │            │
│  │    "exp": 1706000000,     ← 15 min expiry       │            │
│  │    "iat": 1706000000,                            │            │
│  │    "jti": "unique-token-id"                      │            │
│  │  }                                               │            │
│  │  Signature: HMAC-SHA256(header + payload, secret)│            │
│  └─────────────────────────────────────────────────┘            │
│                                                                  │
│  Refresh Token (long-lived: 7 days)                             │
│  ┌─────────────────────────────────────────────────┐            │
│  │  Stored in: HTTP-only secure cookie              │            │
│  │  Payload: {                                      │            │
│  │    "sub": "user-uuid",                           │            │
│  │    "type": "refresh",                            │            │
│  │    "exp": 1706600000,     ← 7 day expiry        │            │
│  │    "jti": "unique-token-id"                      │            │
│  │  }                                               │            │
│  │  Also tracked in DB for revocation               │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                  │
│  Token Flow:                                                     │
│                                                                  │
│  1. Login → Get access_token + refresh_token                    │
│  2. API calls → Send access_token in Authorization header       │
│  3. access_token expires → Use refresh_token to get new pair    │
│  4. refresh_token expires → Must login again                    │
│  5. Logout → Revoke refresh_token in DB                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Token Service Implementation

```python
"""
app/services/token_service.py - JWT token creation and validation.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import jwt, JWTError, ExpiredSignatureError
from pydantic import BaseModel

from app.config import get_settings

settings = get_settings()


class TokenPayload(BaseModel):
    """Decoded token payload."""
    sub: str                # Subject (user ID)
    role: str = "user"      # User role
    type: str = "access"    # Token type: access or refresh
    exp: datetime           # Expiration time
    iat: datetime           # Issued at
    jti: str                # JWT ID (unique identifier)


class TokenPair(BaseModel):
    """Access + refresh token pair."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int         # Access token expiry in seconds


def create_access_token(
    user_id: str,
    role: str = "user",
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a short-lived access token.

    Args:
        user_id: The user's unique identifier.
        role: The user's role for RBAC.
        expires_delta: Optional custom expiration time.

    Returns:
        Encoded JWT string.
    """
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    ))

    payload = {
        "sub": user_id,
        "role": role,
        "type": "access",
        "exp": expire,
        "iat": now,
        "jti": str(uuid.uuid4()),
    }

    return jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def create_refresh_token(
    user_id: str,
    expires_delta: Optional[timedelta] = None,
) -> tuple[str, str]:
    """
    Create a long-lived refresh token.

    Returns:
        Tuple of (encoded JWT string, token JTI for DB storage).
    """
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(
        days=settings.REFRESH_TOKEN_EXPIRE_DAYS
    ))

    jti = str(uuid.uuid4())
    payload = {
        "sub": user_id,
        "type": "refresh",
        "exp": expire,
        "iat": now,
        "jti": jti,
    }

    token = jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    return token, jti


def create_token_pair(user_id: str, role: str = "user") -> TokenPair:
    """Create both access and refresh tokens."""
    access_token = create_access_token(user_id, role)
    refresh_token, _ = create_refresh_token(user_id)

    return TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


def decode_token(token: str, expected_type: str = "access") -> TokenPayload:
    """
    Decode and validate a JWT token.

    Args:
        token: The JWT string to decode.
        expected_type: Expected token type ("access" or "refresh").

    Returns:
        Decoded token payload.

    Raises:
        ValueError: If token is invalid, expired, or wrong type.
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except ExpiredSignatureError:
        raise ValueError("Token has expired")
    except JWTError as e:
        raise ValueError(f"Invalid token: {e}")

    # Validate token type
    token_type = payload.get("type")
    if token_type != expected_type:
        raise ValueError(
            f"Invalid token type: expected {expected_type}, got {token_type}"
        )

    # Validate required fields
    if not payload.get("sub"):
        raise ValueError("Token missing subject claim")

    return TokenPayload(
        sub=payload["sub"],
        role=payload.get("role", "user"),
        type=payload["type"],
        exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
        iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
        jti=payload.get("jti", ""),
    )
```

---

## 5. Role-Based Access Control (RBAC)

### 5.1 RBAC Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    RBAC Permission Matrix                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Role         Permissions                                        │
│  ──────────   ──────────────────────────────────────────        │
│  admin        users:read, users:write, users:delete,            │
│               admin:read, admin:write, system:manage             │
│                                                                  │
│  moderator    users:read, users:write,                           │
│               content:read, content:write, content:delete        │
│                                                                  │
│  user         self:read, self:write,                             │
│               content:read, content:write                        │
│                                                                  │
│  viewer       self:read, content:read                            │
│                                                                  │
│                                                                  │
│  Endpoint              Required Permission     Roles Allowed     │
│  ───────────────────   ──────────────────     ───────────────   │
│  GET  /users           users:read              admin, moderator  │
│  POST /users           users:write             admin             │
│  GET  /users/me        self:read               all               │
│  PUT  /users/me        self:write              all               │
│  DEL  /users/{id}      users:delete            admin             │
│  GET  /admin/stats     admin:read              admin             │
│  POST /admin/config    admin:write             admin             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 RBAC Implementation

```python
"""
app/utils/security.py - RBAC and authentication dependencies.
"""

from enum import Enum
from typing import Annotated
from functools import wraps

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.services.token_service import decode_token, TokenPayload


# ─── Role and Permission Definitions ───

class Role(str, Enum):
    """User roles in order of privilege."""
    ADMIN = "admin"
    MODERATOR = "moderator"
    USER = "user"
    VIEWER = "viewer"


class Permission(str, Enum):
    """Fine-grained permissions."""
    USERS_READ = "users:read"
    USERS_WRITE = "users:write"
    USERS_DELETE = "users:delete"
    SELF_READ = "self:read"
    SELF_WRITE = "self:write"
    CONTENT_READ = "content:read"
    CONTENT_WRITE = "content:write"
    CONTENT_DELETE = "content:delete"
    ADMIN_READ = "admin:read"
    ADMIN_WRITE = "admin:write"
    SYSTEM_MANAGE = "system:manage"


# Role → Permissions mapping
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.ADMIN: {
        Permission.USERS_READ, Permission.USERS_WRITE,
        Permission.USERS_DELETE, Permission.SELF_READ,
        Permission.SELF_WRITE, Permission.CONTENT_READ,
        Permission.CONTENT_WRITE, Permission.CONTENT_DELETE,
        Permission.ADMIN_READ, Permission.ADMIN_WRITE,
        Permission.SYSTEM_MANAGE,
    },
    Role.MODERATOR: {
        Permission.USERS_READ, Permission.USERS_WRITE,
        Permission.SELF_READ, Permission.SELF_WRITE,
        Permission.CONTENT_READ, Permission.CONTENT_WRITE,
        Permission.CONTENT_DELETE,
    },
    Role.USER: {
        Permission.SELF_READ, Permission.SELF_WRITE,
        Permission.CONTENT_READ, Permission.CONTENT_WRITE,
    },
    Role.VIEWER: {
        Permission.SELF_READ, Permission.CONTENT_READ,
    },
}


def has_permission(role: Role, permission: Permission) -> bool:
    """Check if a role has a specific permission."""
    return permission in ROLE_PERMISSIONS.get(role, set())


# ─── FastAPI Dependencies ───

security_scheme = HTTPBearer(
    scheme_name="JWT",
    description="Enter your JWT access token",
)


async def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials,
        Depends(security_scheme)
    ],
) -> TokenPayload:
    """
    Dependency: Extract and validate JWT from Authorization header.
    Returns the decoded token payload.
    """
    try:
        payload = decode_token(credentials.credentials, expected_type="access")
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


def require_role(*allowed_roles: Role):
    """
    Dependency factory: Require user to have one of the specified roles.

    Usage:
        @router.get("/admin", dependencies=[Depends(require_role(Role.ADMIN))])
        async def admin_endpoint():
            ...
    """
    async def role_checker(
        current_user: Annotated[TokenPayload, Depends(get_current_user)],
    ) -> TokenPayload:
        try:
            user_role = Role(current_user.role)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid role",
            )

        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )

        return current_user

    return role_checker


def require_permission(permission: Permission):
    """
    Dependency factory: Require user to have a specific permission.

    Usage:
        @router.delete("/users/{id}",
                       dependencies=[Depends(require_permission(Permission.USERS_DELETE))])
    """
    async def permission_checker(
        current_user: Annotated[TokenPayload, Depends(get_current_user)],
    ) -> TokenPayload:
        try:
            user_role = Role(current_user.role)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid role",
            )

        if not has_permission(user_role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )

        return current_user

    return permission_checker


# Type alias for convenience
CurrentUser = Annotated[TokenPayload, Depends(get_current_user)]
```

---

## 6. Input Validation with Pydantic

### 6.1 User Schemas

```python
"""
app/schemas/user.py - Pydantic schemas for user data validation.
"""

import re
from datetime import datetime
from typing import Optional

from pydantic import (
    BaseModel,
    EmailStr,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)


class UserCreate(BaseModel):
    """Schema for user registration."""

    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        pattern=r'^[a-zA-Z0-9_-]+$',
        description="Username (alphanumeric, underscores, hyphens only)",
        examples=["john_doe"],
    )
    email: EmailStr = Field(
        ...,
        description="Valid email address",
        examples=["john@example.com"],
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Password (min 8 chars, must include upper, lower, digit, special)",
    )
    full_name: Optional[str] = Field(
        None,
        max_length=100,
        description="Full name (optional)",
    )

    @field_validator('username')
    @classmethod
    def username_not_reserved(cls, v: str) -> str:
        """Ensure username is not a reserved name."""
        reserved = {
            'admin', 'administrator', 'root', 'system',
            'api', 'www', 'mail', 'support', 'help',
            'null', 'undefined', 'true', 'false',
        }
        if v.lower() in reserved:
            raise ValueError(f"Username '{v}' is reserved")
        return v

    @field_validator('email')
    @classmethod
    def email_normalize(cls, v: str) -> str:
        """Normalize email to lowercase."""
        return v.lower().strip()

    @field_validator('full_name')
    @classmethod
    def sanitize_full_name(cls, v: Optional[str]) -> Optional[str]:
        """Sanitize full name to prevent injection."""
        if v is None:
            return v
        # Remove any HTML tags
        v = re.sub(r'<[^>]*>', '', v)
        # Remove control characters
        v = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', v)
        return v.strip()


class UserUpdate(BaseModel):
    """Schema for updating user profile."""

    full_name: Optional[str] = Field(
        None, max_length=100
    )
    email: Optional[EmailStr] = None

    @field_validator('full_name')
    @classmethod
    def sanitize_full_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = re.sub(r'<[^>]*>', '', v)
        v = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', v)
        return v.strip()


class UserResponse(BaseModel):
    """Schema for user data in responses. Never include password hash."""

    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    role: str
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    # IMPORTANT: No password_hash field! Never expose hashes.


class UserListResponse(BaseModel):
    """Paginated user list response."""
    users: list[UserResponse]
    total: int
    page: int
    per_page: int
    pages: int


class PasswordChange(BaseModel):
    """Schema for password change request."""

    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)

    @model_validator(mode='after')
    def passwords_different(self):
        """Ensure new password differs from current."""
        if self.current_password == self.new_password:
            raise ValueError("New password must be different from current password")
        return self
```

### 6.2 Auth Schemas

```python
"""
app/schemas/auth.py - Authentication request/response schemas.
"""

from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    """Login request schema."""
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


class TokenResponse(BaseModel):
    """Token response after successful authentication."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    # refresh_token is sent as HTTP-only cookie, not in body


class RefreshRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str


class MessageResponse(BaseModel):
    """Generic message response."""
    message: str
    detail: str = ""
```

---

## 7. Middleware: Rate Limiting

### 7.1 Rate Limiter

```python
"""
app/middleware/rate_limiter.py - Rate limiting middleware.
"""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request


def get_client_ip(request: Request) -> str:
    """
    Get the real client IP, considering proxy headers.

    IMPORTANT: Only trust X-Forwarded-For if behind a trusted proxy.
    In production, configure this based on your infrastructure.
    """
    # If behind a trusted reverse proxy (nginx, AWS ALB, etc.)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP (client's real IP)
        return forwarded_for.split(",")[0].strip()

    # Direct connection
    if request.client:
        return request.client.host

    return "unknown"


# Create limiter instance
limiter = Limiter(
    key_func=get_client_ip,
    default_limits=["60/minute"],  # Global default
    storage_uri="memory://",       # Use Redis in production:
    # storage_uri="redis://localhost:6379/0"
)
```

---

## 8. Middleware: Security Headers

### 8.1 Security Headers Middleware

```python
"""
app/middleware/security_headers.py - Add security headers to all responses.
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to every response.
    These headers protect against common web attacks.
    """

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Control referrer information
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; "
            "frame-ancestors 'none'"
        )

        # Permissions Policy (disable unnecessary browser features)
        response.headers["Permissions-Policy"] = (
            "camera=(), "
            "microphone=(), "
            "geolocation=(), "
            "payment=()"
        )

        # Strict Transport Security (HTTPS only)
        # Only enable in production with HTTPS
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )

        # Prevent caching of authenticated responses
        if request.headers.get("Authorization"):
            response.headers["Cache-Control"] = (
                "no-store, no-cache, must-revalidate, private"
            )
            response.headers["Pragma"] = "no-cache"

        # Remove server identification header
        if "server" in response.headers:
            del response.headers["server"]

        return response
```

---

## 9. Middleware: Request Logging

### 9.1 Structured Security Logging

```python
"""
app/utils/logging.py - Structured security logging.
"""

import sys
import time
import uuid
from typing import Any

import structlog
from fastapi import Request

from app.config import get_settings

settings = get_settings()


def setup_logging() -> None:
    """Configure structured logging."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
            # JSON output for production, console for development
            structlog.processors.JSONRenderer()
            if settings.LOG_FORMAT == "json"
            else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_security_logger() -> structlog.BoundLogger:
    """Get a logger configured for security events."""
    return structlog.get_logger("security")


# ─── Specific Security Event Loggers ───

def log_authentication_event(
    event_type: str,
    email: str,
    success: bool,
    ip_address: str,
    user_agent: str = "",
    reason: str = "",
) -> None:
    """
    Log authentication events (login, logout, token refresh).

    IMPORTANT: Never log passwords or tokens.
    """
    logger = get_security_logger()
    log_func = logger.info if success else logger.warning

    log_func(
        "authentication_event",
        event_type=event_type,
        email=_mask_email(email),
        success=success,
        ip_address=ip_address,
        user_agent=user_agent[:200],  # Truncate long user agents
        reason=reason,
    )


def log_authorization_event(
    user_id: str,
    resource: str,
    action: str,
    granted: bool,
    ip_address: str,
) -> None:
    """Log authorization decisions."""
    logger = get_security_logger()
    log_func = logger.info if granted else logger.warning

    log_func(
        "authorization_event",
        user_id=user_id,
        resource=resource,
        action=action,
        granted=granted,
        ip_address=ip_address,
    )


def log_security_violation(
    violation_type: str,
    details: str,
    ip_address: str,
    user_id: str = "",
    severity: str = "HIGH",
) -> None:
    """Log security violations (rate limiting, invalid tokens, etc.)."""
    logger = get_security_logger()
    logger.error(
        "security_violation",
        violation_type=violation_type,
        details=details,
        ip_address=ip_address,
        user_id=user_id,
        severity=severity,
    )


def log_data_access(
    user_id: str,
    resource_type: str,
    resource_id: str,
    action: str,
    ip_address: str,
) -> None:
    """Log data access for audit trail."""
    logger = get_security_logger()
    logger.info(
        "data_access",
        user_id=user_id,
        resource_type=resource_type,
        resource_id=resource_id,
        action=action,
        ip_address=ip_address,
    )


def _mask_email(email: str) -> str:
    """Mask email for logging (show first char and domain only)."""
    if '@' not in email:
        return '***'
    local, domain = email.split('@', 1)
    if len(local) <= 1:
        masked_local = '*'
    else:
        masked_local = local[0] + '*' * (len(local) - 1)
    return f"{masked_local}@{domain}"
```

### 9.2 Request Logging Middleware

```python
"""
app/middleware/request_logging.py - Log all HTTP requests.
"""

import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.utils.logging import get_security_logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with timing and security-relevant info."""

    async def dispatch(self, request: Request, call_next):
        # Generate unique request ID
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Add request ID to response headers
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Log the request
        logger = get_security_logger()
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query": str(request.query_params) if request.query_params else "",
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "ip_address": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "")[:200],
        }

        if response.status_code >= 500:
            logger.error("http_request", **log_data)
        elif response.status_code >= 400:
            logger.warning("http_request", **log_data)
        else:
            logger.info("http_request", **log_data)

        return response
```

---

## 10. Error Handling Without Information Leakage

### 10.1 Safe Error Handling

```python
"""
app/main.py - Application setup with safe error handling.
"""

import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.middleware.security_headers import SecurityHeadersMiddleware
from app.middleware.rate_limiter import limiter
from app.middleware.request_logging import RequestLoggingMiddleware
from app.routers import auth, users, admin
from app.utils.logging import setup_logging, get_security_logger, log_security_violation

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown events."""
    setup_logging()
    logger = get_security_logger()
    logger.info("application_start", environment=settings.ENVIRONMENT)
    yield
    logger.info("application_shutdown")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    # Disable docs in production
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan,
)

# ─── Middleware (order matters: last added = first executed) ───

# 1. Request logging (outermost)
app.add_middleware(RequestLoggingMiddleware)

# 2. Security headers
app.add_middleware(SecurityHeadersMiddleware)

# 3. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
    expose_headers=["X-Request-ID"],
)

# 4. Rate limiter
app.state.limiter = limiter


# ─── Error Handlers ───

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors."""
    ip = request.client.host if request.client else "unknown"
    log_security_violation(
        violation_type="RATE_LIMIT_EXCEEDED",
        details=f"Rate limit exceeded on {request.url.path}",
        ip_address=ip,
    )
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "rate_limit_exceeded",
            "message": "Too many requests. Please try again later.",
        },
        headers={"Retry-After": "60"},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
):
    """
    Handle validation errors.

    IMPORTANT: Return generic message. Do not expose internal
    validation details that could help an attacker craft payloads.
    """
    # Log full details internally
    logger = get_security_logger()
    logger.warning(
        "validation_error",
        path=request.url.path,
        errors=str(exc.errors())[:500],
        ip_address=request.client.host if request.client else "unknown",
    )

    # Return sanitized error to client
    # In production: generic message
    # In development: include field-level details
    if settings.DEBUG:
        # Development: helpful error messages
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "message": "Request validation failed",
                "details": [
                    {
                        "field": ".".join(str(x) for x in err["loc"]),
                        "message": err["msg"],
                    }
                    for err in exc.errors()
                ],
            },
        )
    else:
        # Production: minimal information
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "message": "Invalid request data",
            },
        )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Catch-all error handler.

    CRITICAL: Never expose stack traces, internal paths,
    database details, or any implementation details to clients.
    """
    # Log full error internally
    logger = get_security_logger()
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        error_type=type(exc).__name__,
        error_message=str(exc)[:500],
        traceback=traceback.format_exc()[:2000],
        ip_address=request.client.host if request.client else "unknown",
    )

    # Return generic error to client
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again later.",
        },
    )


# ─── Routers ───

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])


# ─── Health Check ───

@app.get("/health", tags=["Health"])
async def health_check():
    """Public health check endpoint (no auth required)."""
    return {"status": "healthy", "version": settings.APP_VERSION}
```

---

## 11. Authentication Endpoints

### 11.1 Auth Router

```python
"""
app/routers/auth.py - Authentication endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.middleware.rate_limiter import limiter
from app.schemas.auth import LoginRequest, TokenResponse, MessageResponse
from app.schemas.user import UserCreate, UserResponse
from app.services.auth_service import AuthService
from app.utils.logging import log_authentication_event
from app.utils.password import validate_password_strength, PasswordStrengthError

settings = get_settings()
router = APIRouter()


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
@limiter.limit(settings.REGISTER_RATE_LIMIT)
async def register(
    request: Request,
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new user account.

    Rate limited to prevent abuse.
    """
    # Validate password strength
    try:
        validate_password_strength(user_data.password)
    except PasswordStrengthError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    auth_service = AuthService(db)

    # Check if user already exists
    existing = await auth_service.get_user_by_email(user_data.email)
    if existing:
        # IMPORTANT: Don't reveal if email exists
        # Use a generic message for both "exists" and "created"
        # to prevent user enumeration
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to create account with provided information",
        )

    # Create user
    user = await auth_service.create_user(user_data)

    ip = request.client.host if request.client else "unknown"
    log_authentication_event(
        event_type="registration",
        email=user_data.email,
        success=True,
        ip_address=ip,
        user_agent=request.headers.get("user-agent", ""),
    )

    return user


@router.post("/login", response_model=TokenResponse)
@limiter.limit(settings.LOGIN_RATE_LIMIT)
async def login(
    request: Request,
    response: Response,
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Authenticate user and return JWT tokens.

    Access token in response body, refresh token as HTTP-only cookie.
    """
    ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "")

    auth_service = AuthService(db)
    result = await auth_service.authenticate(
        login_data.email, login_data.password
    )

    if not result:
        log_authentication_event(
            event_type="login",
            email=login_data.email,
            success=False,
            ip_address=ip,
            user_agent=user_agent,
            reason="invalid_credentials",
        )

        # IMPORTANT: Generic error message - don't reveal
        # whether email exists or password is wrong
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user, token_pair = result

    # Set refresh token as HTTP-only secure cookie
    response.set_cookie(
        key="refresh_token",
        value=token_pair.refresh_token,
        httponly=True,             # JavaScript cannot access
        secure=settings.SECURE_COOKIES,  # HTTPS only in production
        samesite="lax",           # CSRF protection
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        path="/api/v1/auth",      # Only sent to auth endpoints
    )

    log_authentication_event(
        event_type="login",
        email=login_data.email,
        success=True,
        ip_address=ip,
        user_agent=user_agent,
    )

    return TokenResponse(
        access_token=token_pair.access_token,
        expires_in=token_pair.expires_in,
    )


@router.post("/refresh", response_model=TokenResponse)
@limiter.limit("10/minute")
async def refresh_token(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    """
    Refresh access token using refresh token from cookie.

    Implements token rotation: old refresh token is invalidated,
    new refresh token is issued.
    """
    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token not found",
        )

    auth_service = AuthService(db)
    result = await auth_service.refresh_tokens(refresh_token)

    if not result:
        # Clear the invalid cookie
        response.delete_cookie("refresh_token", path="/api/v1/auth")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    new_token_pair = result

    # Rotate refresh token (set new cookie)
    response.set_cookie(
        key="refresh_token",
        value=new_token_pair.refresh_token,
        httponly=True,
        secure=settings.SECURE_COOKIES,
        samesite="lax",
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        path="/api/v1/auth",
    )

    return TokenResponse(
        access_token=new_token_pair.access_token,
        expires_in=new_token_pair.expires_in,
    )


@router.post("/logout", response_model=MessageResponse)
async def logout(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    """
    Logout: revoke refresh token and clear cookie.
    """
    refresh_token = request.cookies.get("refresh_token")
    if refresh_token:
        auth_service = AuthService(db)
        await auth_service.revoke_refresh_token(refresh_token)

    # Clear the refresh token cookie
    response.delete_cookie("refresh_token", path="/api/v1/auth")

    ip = request.client.host if request.client else "unknown"
    log_authentication_event(
        event_type="logout",
        email="",
        success=True,
        ip_address=ip,
        user_agent=request.headers.get("user-agent", ""),
    )

    return MessageResponse(message="Successfully logged out")
```

---

## 12. User Endpoints

### 12.1 User Router

```python
"""
app/routers/users.py - User management endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.user import (
    UserResponse, UserUpdate, UserListResponse, PasswordChange,
)
from app.services.user_service import UserService
from app.utils.security import (
    CurrentUser, require_role, require_permission,
    Role, Permission,
)
from app.utils.password import validate_password_strength, PasswordStrengthError
from app.utils.logging import log_data_access

router = APIRouter()


# ─── Self-Service Endpoints (any authenticated user) ───

@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    request: Request,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Get current user's profile."""
    user_service = UserService(db)
    user = await user_service.get_by_id(current_user.sub)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return user


@router.put("/me", response_model=UserResponse)
async def update_current_user_profile(
    request: Request,
    update_data: UserUpdate,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Update current user's profile."""
    user_service = UserService(db)
    user = await user_service.update(current_user.sub, update_data)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    ip = request.client.host if request.client else "unknown"
    log_data_access(
        user_id=current_user.sub,
        resource_type="user",
        resource_id=current_user.sub,
        action="update_profile",
        ip_address=ip,
    )

    return user


@router.post("/me/change-password", response_model=dict)
async def change_password(
    request: Request,
    password_data: PasswordChange,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Change current user's password."""
    # Validate new password strength
    try:
        validate_password_strength(password_data.new_password)
    except PasswordStrengthError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    user_service = UserService(db)
    success = await user_service.change_password(
        user_id=current_user.sub,
        current_password=password_data.current_password,
        new_password=password_data.new_password,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    ip = request.client.host if request.client else "unknown"
    log_data_access(
        user_id=current_user.sub,
        resource_type="user",
        resource_id=current_user.sub,
        action="change_password",
        ip_address=ip,
    )

    return {"message": "Password changed successfully"}


# ─── Admin Endpoints ───

@router.get(
    "/",
    response_model=UserListResponse,
    dependencies=[Depends(require_permission(Permission.USERS_READ))],
)
async def list_users(
    request: Request,
    page: int = 1,
    per_page: int = 20,
    current_user: CurrentUser = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """List all users (admin/moderator only)."""
    if per_page > 100:
        per_page = 100  # Cap maximum page size

    user_service = UserService(db)
    users, total = await user_service.list_users(
        page=page, per_page=per_page
    )

    ip = request.client.host if request.client else "unknown"
    log_data_access(
        user_id=current_user.sub,
        resource_type="user_list",
        resource_id=f"page={page}",
        action="list",
        ip_address=ip,
    )

    return UserListResponse(
        users=users,
        total=total,
        page=page,
        per_page=per_page,
        pages=(total + per_page - 1) // per_page,
    )


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    dependencies=[Depends(require_permission(Permission.USERS_READ))],
)
async def get_user(
    user_id: str,
    request: Request,
    current_user: CurrentUser = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific user (admin/moderator only)."""
    user_service = UserService(db)
    user = await user_service.get_by_id(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return user


@router.delete(
    "/{user_id}",
    response_model=dict,
    dependencies=[Depends(require_permission(Permission.USERS_DELETE))],
)
async def delete_user(
    user_id: str,
    request: Request,
    current_user: CurrentUser = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """Delete a user (admin only)."""
    # Prevent self-deletion
    if user_id == current_user.sub:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account via this endpoint",
        )

    user_service = UserService(db)
    success = await user_service.delete(user_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    ip = request.client.host if request.client else "unknown"
    log_data_access(
        user_id=current_user.sub,
        resource_type="user",
        resource_id=user_id,
        action="delete",
        ip_address=ip,
    )

    return {"message": "User deleted successfully"}
```

---

## 13. Security Tests

### 13.1 Test Configuration

```python
"""
tests/conftest.py - Test fixtures and configuration.
"""

import asyncio
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)

from app.main import app
from app.database import get_db, Base
from app.config import get_settings

# Use SQLite for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create test HTTP client with dependency overrides."""

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def registered_user(client: AsyncClient) -> dict:
    """Create a registered user and return credentials."""
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "TestPass123!",
        "full_name": "Test User",
    }
    response = await client.post("/api/v1/auth/register", json=user_data)
    assert response.status_code == 201
    return user_data


@pytest_asyncio.fixture
async def auth_headers(client: AsyncClient, registered_user: dict) -> dict:
    """Get authentication headers for a registered user."""
    login_data = {
        "email": registered_user["email"],
        "password": registered_user["password"],
    }
    response = await client.post("/api/v1/auth/login", json=login_data)
    assert response.status_code == 200

    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
```

### 13.2 Security-Specific Tests

```python
"""
tests/test_security.py - Security-focused tests.
"""

import pytest
from httpx import AsyncClient


class TestAuthenticationSecurity:
    """Test authentication security measures."""

    @pytest.mark.asyncio
    async def test_login_wrong_password_generic_error(self, client: AsyncClient,
                                                       registered_user: dict):
        """Login with wrong password should return generic error."""
        response = await client.post("/api/v1/auth/login", json={
            "email": registered_user["email"],
            "password": "WrongPassword123!",
        })
        assert response.status_code == 401
        # Should NOT say "wrong password" - just generic message
        assert "Invalid email or password" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_login_nonexistent_email_generic_error(self, client: AsyncClient):
        """Login with non-existent email should return same error."""
        response = await client.post("/api/v1/auth/login", json={
            "email": "nonexistent@example.com",
            "password": "SomePassword123!",
        })
        assert response.status_code == 401
        # Same message as wrong password (prevents user enumeration)
        assert "Invalid email or password" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_registration_duplicate_email_generic_error(
        self, client: AsyncClient, registered_user: dict
    ):
        """Duplicate registration should not reveal email exists."""
        response = await client.post("/api/v1/auth/register", json={
            "username": "different_user",
            "email": registered_user["email"],
            "password": "NewPassword123!",
        })
        assert response.status_code == 400
        # Should NOT say "email already exists"
        assert "Unable to create account" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_weak_password_rejected(self, client: AsyncClient):
        """Weak passwords should be rejected."""
        weak_passwords = [
            "short1!",          # Too short
            "alllowercase1!",   # No uppercase
            "ALLUPPERCASE1!",   # No lowercase
            "NoDigitsHere!",    # No digit
            "NoSpecial123",     # No special char
            "password123!",     # Common password
        ]

        for password in weak_passwords:
            response = await client.post("/api/v1/auth/register", json={
                "username": "testuser2",
                "email": "test2@example.com",
                "password": password,
            })
            assert response.status_code == 400, \
                f"Weak password accepted: {password}"


class TestAuthorizationSecurity:
    """Test authorization and access control."""

    @pytest.mark.asyncio
    async def test_unauthenticated_access_denied(self, client: AsyncClient):
        """Protected endpoints should reject unauthenticated requests."""
        protected_endpoints = [
            ("GET", "/api/v1/users/me"),
            ("PUT", "/api/v1/users/me"),
            ("GET", "/api/v1/users/"),
            ("GET", "/api/v1/admin/stats"),
        ]

        for method, path in protected_endpoints:
            response = await client.request(method, path)
            assert response.status_code in (401, 403), \
                f"{method} {path} accessible without auth"

    @pytest.mark.asyncio
    async def test_expired_token_rejected(self, client: AsyncClient):
        """Expired tokens should be rejected."""
        # Create a token with 0 second expiry
        from app.services.token_service import create_access_token
        from datetime import timedelta

        expired_token = create_access_token(
            user_id="test-id",
            role="user",
            expires_delta=timedelta(seconds=-1),  # Already expired
        )

        response = await client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_malformed_token_rejected(self, client: AsyncClient):
        """Malformed tokens should be rejected."""
        malformed_tokens = [
            "not-a-jwt",
            "eyJ.eyJ.invalid",
            "",
            "Bearer ",
            "null",
        ]

        for token in malformed_tokens:
            response = await client.get(
                "/api/v1/users/me",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == 401 or response.status_code == 403


class TestInputValidation:
    """Test input validation and sanitization."""

    @pytest.mark.asyncio
    async def test_sql_injection_in_login(self, client: AsyncClient):
        """SQL injection payloads should be handled safely."""
        payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "admin'--",
        ]

        for payload in payloads:
            response = await client.post("/api/v1/auth/login", json={
                "email": f"{payload}@example.com",
                "password": payload,
            })
            # Should return 401 or 422, NOT 500
            assert response.status_code in (401, 422), \
                f"SQLi payload caused error: {payload}"

    @pytest.mark.asyncio
    async def test_xss_in_username(self, client: AsyncClient):
        """XSS payloads in username should be rejected."""
        response = await client.post("/api/v1/auth/register", json={
            "username": "<script>alert(1)</script>",
            "email": "xss@example.com",
            "password": "SafePassword123!",
        })
        # Should fail validation (username pattern: alphanumeric only)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_oversized_input_rejected(self, client: AsyncClient):
        """Extremely large inputs should be rejected."""
        response = await client.post("/api/v1/auth/register", json={
            "username": "a" * 10000,
            "email": "big@example.com",
            "password": "BigPassword123!",
        })
        assert response.status_code == 422


class TestSecurityHeaders:
    """Test security headers are present."""

    @pytest.mark.asyncio
    async def test_security_headers_present(self, client: AsyncClient):
        """All security headers should be set."""
        response = await client.get("/health")

        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert "strict-origin" in response.headers.get("Referrer-Policy", "")
        assert response.headers.get("Content-Security-Policy") is not None

    @pytest.mark.asyncio
    async def test_no_server_header(self, client: AsyncClient):
        """Server header should not reveal implementation details."""
        response = await client.get("/health")
        server = response.headers.get("server", "")
        # Should not reveal uvicorn, gunicorn, etc.
        assert "uvicorn" not in server.lower()
        assert "python" not in server.lower()


class TestErrorHandling:
    """Test error handling doesn't leak information."""

    @pytest.mark.asyncio
    async def test_404_no_info_leak(self, client: AsyncClient, auth_headers: dict):
        """404 errors should not reveal internal paths."""
        response = await client.get(
            "/api/v1/nonexistent",
            headers=auth_headers,
        )
        body = response.json()
        # Should not contain file paths or stack traces
        assert "/app/" not in str(body)
        assert "traceback" not in str(body).lower()
        assert "Traceback" not in str(body)

    @pytest.mark.asyncio
    async def test_500_no_stack_trace(self, client: AsyncClient):
        """500 errors should not expose stack traces."""
        # This tests the global exception handler
        response = await client.get("/health")
        # Health should work, but verify error format
        if response.status_code == 500:
            body = response.json()
            assert "traceback" not in str(body).lower()
            assert "File " not in str(body)
```

---

## 14. Deployment Considerations

### 14.1 Production Deployment Checklist

```
┌──────────────────────────────────────────────────────────────────┐
│              Production Deployment Checklist                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  HTTPS / TLS:                                                     │
│  [ ] TLS 1.2+ only (disable TLS 1.0, 1.1)                      │
│  [ ] Strong cipher suites                                        │
│  [ ] Valid SSL certificate (Let's Encrypt or CA-signed)         │
│  [ ] HSTS header enabled                                         │
│  [ ] HTTP → HTTPS redirect                                       │
│                                                                   │
│  Reverse Proxy (nginx):                                          │
│  [ ] Hide backend server headers                                 │
│  [ ] Rate limiting at proxy level                                │
│  [ ] Request body size limits                                    │
│  [ ] Timeout configurations                                      │
│  [ ] Access logging                                               │
│                                                                   │
│  Application:                                                     │
│  [ ] DEBUG = False                                                │
│  [ ] API docs disabled in production                             │
│  [ ] Secrets from environment variables / secret manager         │
│  [ ] All dependencies pinned and audited                         │
│  [ ] Error handling: no stack traces exposed                     │
│  [ ] Logging: structured, no PII/secrets in logs                │
│                                                                   │
│  Database:                                                        │
│  [ ] Strong password for DB user                                 │
│  [ ] Minimal DB privileges (no admin/superuser)                  │
│  [ ] Encrypted connections (SSL/TLS)                             │
│  [ ] Regular backups                                              │
│  [ ] Connection pooling configured                               │
│                                                                   │
│  Infrastructure:                                                  │
│  [ ] Firewall: only necessary ports open                         │
│  [ ] SSH key-based auth only                                     │
│  [ ] Regular OS patching                                          │
│  [ ] Monitoring and alerting configured                          │
│  [ ] Log aggregation (ELK, Datadog, etc.)                       │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 14.2 Nginx Reverse Proxy Configuration

```nginx
# /etc/nginx/sites-available/secure-api

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name api.example.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name api.example.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security Headers (additional to app-level headers)
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Hide nginx version
    server_tokens off;

    # Request limits
    client_max_body_size 10m;          # Max request body size
    client_body_timeout 12;             # Body read timeout
    client_header_timeout 12;           # Header read timeout

    # Rate limiting zone
    limit_req_zone $binary_remote_addr zone=api:10m rate=60r/m;

    location / {
        # Apply rate limiting
        limit_req zone=api burst=20 nodelay;

        # Proxy to FastAPI
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Proxy timeouts
        proxy_connect_timeout 5s;
        proxy_read_timeout 30s;
        proxy_send_timeout 30s;

        # Hide proxy headers
        proxy_hide_header X-Powered-By;
    }

    # Block common attack paths
    location ~ /\.(git|svn|env|htaccess) {
        deny all;
        return 404;
    }

    # Access logging
    access_log /var/log/nginx/api_access.log;
    error_log /var/log/nginx/api_error.log;
}
```

### 14.3 Docker Deployment

```dockerfile
# Dockerfile - Multi-stage build for smaller image

# Stage 1: Build dependencies
FROM python:3.12-slim as builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Production image
FROM python:3.12-slim

# Security: Run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Copy dependencies from builder
COPY --from=builder /install /usr/local

# Copy application
WORKDIR /app
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Do NOT expose port 8000 directly - use reverse proxy
```

---

## 15. Exercises

### Exercise 1: Implement the Auth Service

Complete the `AuthService` class that handles:
1. User creation with Argon2 password hashing
2. Authentication (email + password verification)
3. Token pair creation and refresh token storage in DB
4. Refresh token rotation and revocation
5. Account lockout after N failed attempts

### Exercise 2: Add Two-Factor Authentication

Extend the authentication system to support TOTP-based 2FA:
1. Add a `/auth/2fa/setup` endpoint that returns a QR code URL
2. Add a `/auth/2fa/verify` endpoint to confirm setup
3. Modify login to require TOTP code when 2FA is enabled
4. Add backup codes for account recovery

### Exercise 3: API Key Authentication

Add support for API key authentication alongside JWT:
1. Create an endpoint to generate API keys
2. Support API key authentication via `X-API-Key` header
3. Implement per-key rate limiting
4. Add API key rotation support
5. Log all API key usage

### Exercise 4: Complete Test Suite

Write additional tests covering:
1. Token refresh flow (including rotation)
2. Account lockout after failed attempts
3. RBAC: verify each role can only access permitted endpoints
4. Rate limiting: verify limits are enforced
5. CORS: verify only allowed origins can access the API

### Exercise 5: Security Audit

Conduct a security audit of the complete project:
1. Run Bandit against all source files
2. Run pip-audit against requirements.txt
3. Check for hardcoded secrets
4. Review all error responses for information leakage
5. Verify all input validation is comprehensive
6. Document any findings and create fixes

### Exercise 6: Production Deployment

Deploy the API to a cloud provider (or local Docker):
1. Set up PostgreSQL with SSL
2. Configure nginx as reverse proxy with TLS
3. Set up structured log aggregation
4. Configure monitoring and alerting
5. Run a security scan (ZAP baseline) against the deployed API
6. Document the deployment architecture

---

## Summary

```
┌──────────────────────────────────────────────────────────────────┐
│            Secure API Key Takeaways                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. Defense in depth: Multiple security layers, not just one    │
│  2. Argon2id: Use proper password hashing, not SHA/MD5/bcrypt   │
│  3. JWT: Short-lived access tokens + rotated refresh tokens     │
│  4. RBAC: Fine-grained permissions, principle of least privilege│
│  5. Input validation: Validate and sanitize ALL external input  │
│  6. Rate limiting: Protect login, registration, and all APIs    │
│  7. Security headers: HSTS, CSP, X-Frame-Options on every      │
│     response                                                     │
│  8. Error handling: Generic errors to users, detailed logs      │
│     internally                                                   │
│  9. Logging: Structured audit logs, never log secrets or PII    │
│ 10. Testing: Security tests are as important as functional tests│
│ 11. Deployment: HTTPS, reverse proxy, non-root containers      │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

**Previous**: [14. Incident Response and Forensics](14_Incident_Response.md) | **Next**: [16. Project: Building a Vulnerability Scanner](16_Project_Vulnerability_Scanner.md)
