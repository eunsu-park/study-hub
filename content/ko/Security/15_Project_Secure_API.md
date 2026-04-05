# 프로젝트: 안전한 REST API 구축

**이전**: [사고 대응](./14_Incident_Response.md) | **다음**: [프로젝트: 취약점 스캐너](./16_Project_Vulnerability_Scanner.md)

---

이 프로젝트 레슨은 FastAPI를 사용하여 프로덕션 레벨의 안전한 REST API를 처음부터 구축하는 과정을 안내합니다. Argon2를 사용한 적절한 비밀번호 해싱, 리프레시 토큰을 사용한 JWT 인증, 역할 기반 접근 제어, 입력 검증, 속도 제한, 보안 헤더, CORS 설정, 구조화된 로깅, 안전한 오류 처리 등 모든 보안 계층을 구현할 것입니다. 레슨을 마치면 보안 모범 사례를 따르는 완전하고 배포 가능한 API를 갖게 됩니다.

## 학습 목표

- 보안을 최우선 관심사로 하는 REST API 구축
- 적절한 설정으로 Argon2 비밀번호 해싱 구현
- 액세스 토큰 및 리프레시 토큰 로테이션을 사용한 JWT 인증 생성
- 역할 기반 접근 제어(RBAC) 설계 및 구현
- Pydantic 모델을 사용한 모든 입력 검증
- 속도 제한, 보안 헤더, CORS 추가
- 구조화된 보안 로깅 구현
- 민감한 정보를 노출하지 않는 오류 처리
- 보안 중심 테스트 작성
- 프로덕션 배포 준비

---

## 1. 프로젝트 개요 및 아키텍처

### 1.1 구축할 내용

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

### 1.2 프로젝트 구조

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

### 1.3 의존성

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

## 2. 설정 관리

### 2.1 안전한 설정

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

### 2.2 환경 파일

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

## 3. Argon2를 사용한 비밀번호 해싱

### 3.1 Argon2를 사용하는 이유

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

### 3.2 비밀번호 해싱 구현

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

## 4. 리프레시 토큰을 사용한 JWT 인증

### 4.1 토큰 아키텍처

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

### 4.2 토큰 서비스 구현

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

## 5. 역할 기반 접근 제어 (RBAC)

### 5.1 RBAC 설계

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

### 5.2 RBAC 구현

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

## 6. Pydantic을 사용한 입력 검증

### 6.1 사용자 스키마

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
        v = re.sub(r'<[^>]*>', '', v)
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

### 6.2 인증 스키마

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


class RefreshRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str


class MessageResponse(BaseModel):
    """Generic message response."""
    message: str
    detail: str = ""
```

---

## 7. 미들웨어: 속도 제한

### 7.1 속도 제한기

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
    프록시 헤더를 고려하여 실제 클라이언트 IP를 가져옵니다.

    중요: 신뢰할 수 있는 프록시 뒤에 있을 때만 X-Forwarded-For를 신뢰하세요.
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    if request.client:
        return request.client.host

    return "unknown"


limiter = Limiter(
    key_func=get_client_ip,
    default_limits=["60/minute"],
    storage_uri="memory://",
)
```

---

## 8. 미들웨어: 보안 헤더

### 8.1 보안 헤더 미들웨어

```python
"""
app/middleware/security_headers.py - Add security headers to all responses.
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """모든 응답에 보안 헤더를 추가합니다."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)

        # 클릭재킹 방지
        response.headers["X-Frame-Options"] = "DENY"

        # MIME 타입 스니핑 방지
        response.headers["X-Content-Type-Options"] = "nosniff"

        # 리퍼러 정보 제어
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # 콘텐츠 보안 정책
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; "
            "frame-ancestors 'none'"
        )

        # 권한 정책 (불필요한 브라우저 기능 비활성화)
        response.headers["Permissions-Policy"] = (
            "camera=(), "
            "microphone=(), "
            "geolocation=(), "
            "payment=()"
        )

        # HSTS (HTTPS 전용)
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )

        # 인증된 응답의 캐싱 방지
        if request.headers.get("Authorization"):
            response.headers["Cache-Control"] = (
                "no-store, no-cache, must-revalidate, private"
            )
            response.headers["Pragma"] = "no-cache"

        # 서버 식별 헤더 제거
        if "server" in response.headers:
            del response.headers["server"]

        return response
```

---

## 9. 미들웨어: 요청 로깅

### 9.1 구조화된 보안 로깅

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
    """구조화된 로깅을 설정합니다."""
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
    """보안 이벤트용 로거를 가져옵니다."""
    return structlog.get_logger("security")


# ─── 보안 이벤트별 로거 ───

def log_authentication_event(
    event_type: str,
    email: str,
    success: bool,
    ip_address: str,
    user_agent: str = "",
    reason: str = "",
) -> None:
    """
    인증 이벤트를 로깅합니다 (로그인, 로그아웃, 토큰 갱신).

    중요: 비밀번호나 토큰은 절대 로깅하지 않습니다.
    """
    logger = get_security_logger()
    log_func = logger.info if success else logger.warning

    log_func(
        "authentication_event",
        event_type=event_type,
        email=_mask_email(email),
        success=success,
        ip_address=ip_address,
        user_agent=user_agent[:200],
        reason=reason,
    )


def log_authorization_event(
    user_id: str,
    resource: str,
    action: str,
    granted: bool,
    ip_address: str,
) -> None:
    """인가 결정을 로깅합니다."""
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
    """보안 위반을 로깅합니다 (속도 제한, 무효 토큰 등)."""
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
    """감사 추적을 위한 데이터 접근을 로깅합니다."""
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
    """로깅용 이메일 마스킹 (첫 글자와 도메인만 표시)."""
    if '@' not in email:
        return '***'
    local, domain = email.split('@', 1)
    if len(local) <= 1:
        masked_local = '*'
    else:
        masked_local = local[0] + '*' * (len(local) - 1)
    return f"{masked_local}@{domain}"
```

### 9.2 요청 로깅 미들웨어

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
    """모든 HTTP 요청을 타이밍 및 보안 관련 정보와 함께 로깅합니다."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        duration_ms = (time.time() - start_time) * 1000

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

## 10. 정보 유출 없는 에러 처리

### 10.1 안전한 에러 처리

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
    """애플리케이션 시작/종료 이벤트."""
    setup_logging()
    logger = get_security_logger()
    logger.info("application_start", environment=settings.ENVIRONMENT)
    yield
    logger.info("application_shutdown")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan,
)

# ─── 미들웨어 (순서 중요: 마지막에 추가된 것이 먼저 실행) ───

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
    expose_headers=["X-Request-ID"],
)

app.state.limiter = limiter


# ─── 에러 핸들러 ───

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """속도 제한 초과 에러를 처리합니다."""
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
    검증 에러를 처리합니다.

    중요: 일반적인 메시지를 반환합니다. 공격자가 페이로드를 만드는 데
    도움이 될 수 있는 내부 검증 세부사항을 노출하지 않습니다.
    """
    logger = get_security_logger()
    logger.warning(
        "validation_error",
        path=request.url.path,
        errors=str(exc.errors())[:500],
        ip_address=request.client.host if request.client else "unknown",
    )

    if settings.DEBUG:
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
    포괄적 에러 핸들러.

    중요: 스택 트레이스, 내부 경로, 데이터베이스 세부사항 또는
    구현 세부사항을 클라이언트에 절대 노출하지 않습니다.
    """
    logger = get_security_logger()
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        error_type=type(exc).__name__,
        error_message=str(exc)[:500],
        traceback=traceback.format_exc()[:2000],
        ip_address=request.client.host if request.client else "unknown",
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again later.",
        },
    )


# ─── 라우터 ───

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])


# ─── 헬스 체크 ───

@app.get("/health", tags=["Health"])
async def health_check():
    """공개 헬스 체크 엔드포인트 (인증 불필요)."""
    return {"status": "healthy", "version": settings.APP_VERSION}
```

---

## 11. 인증 엔드포인트

### 11.1 인증 라우터

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
    새 사용자 계정을 등록합니다.

    남용 방지를 위해 속도 제한이 적용됩니다.
    """
    try:
        validate_password_strength(user_data.password)
    except PasswordStrengthError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    auth_service = AuthService(db)

    existing = await auth_service.get_user_by_email(user_data.email)
    if existing:
        # 중요: 이메일 존재 여부를 노출하지 않습니다
        # 사용자 열거를 방지하기 위해 일반적인 메시지를 사용합니다
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to create account with provided information",
        )

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
    사용자를 인증하고 JWT 토큰을 반환합니다.

    액세스 토큰은 응답 본문에, 리프레시 토큰은 HTTP-only 쿠키로 전달됩니다.
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

        # 중요: 일반적인 에러 메시지 - 이메일 존재 여부나
        # 비밀번호 오류를 노출하지 않습니다
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user, token_pair = result

    # 리프레시 토큰을 HTTP-only 보안 쿠키로 설정
    response.set_cookie(
        key="refresh_token",
        value=token_pair.refresh_token,
        httponly=True,             # JavaScript 접근 불가
        secure=settings.SECURE_COOKIES,  # 프로덕션에서 HTTPS 전용
        samesite="lax",           # CSRF 보호
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        path="/api/v1/auth",      # 인증 엔드포인트에만 전송
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
    쿠키의 리프레시 토큰을 사용하여 액세스 토큰을 갱신합니다.

    토큰 로테이션 구현: 기존 리프레시 토큰은 무효화되고
    새 리프레시 토큰이 발급됩니다.
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
        response.delete_cookie("refresh_token", path="/api/v1/auth")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    new_token_pair = result

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
    """로그아웃: 리프레시 토큰을 폐기하고 쿠키를 삭제합니다."""
    refresh_token = request.cookies.get("refresh_token")
    if refresh_token:
        auth_service = AuthService(db)
        await auth_service.revoke_refresh_token(refresh_token)

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

## 12. 사용자 엔드포인트

### 12.1 사용자 라우터

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


# ─── 셀프 서비스 엔드포인트 (인증된 모든 사용자) ───

@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    request: Request,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """현재 사용자의 프로필을 가져옵니다."""
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
    """현재 사용자의 프로필을 업데이트합니다."""
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
    """현재 사용자의 비밀번호를 변경합니다."""
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


# ─── 관리자 엔드포인트 ───

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
    """모든 사용자를 조회합니다 (관리자/모더레이터 전용)."""
    if per_page > 100:
        per_page = 100

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
    """특정 사용자를 조회합니다 (관리자/모더레이터 전용)."""
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
    """사용자를 삭제합니다 (관리자 전용)."""
    # 자기 자신 삭제 방지
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

## 13. 보안 테스트

### 13.1 테스트 설정

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

TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"


@pytest.fixture(scope="session")
def event_loop():
    """비동기 테스트를 위한 이벤트 루프를 생성합니다."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """테스트 데이터베이스 세션을 생성합니다."""
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
    """의존성 오버라이드가 포함된 테스트 HTTP 클라이언트를 생성합니다."""

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def registered_user(client: AsyncClient) -> dict:
    """등록된 사용자를 생성하고 자격 증명을 반환합니다."""
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
    """등록된 사용자의 인증 헤더를 가져옵니다."""
    login_data = {
        "email": registered_user["email"],
        "password": registered_user["password"],
    }
    response = await client.post("/api/v1/auth/login", json=login_data)
    assert response.status_code == 200

    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
```

### 13.2 보안 전용 테스트

```python
"""
tests/test_security.py - Security-focused tests.
"""

import pytest
from httpx import AsyncClient


class TestAuthenticationSecurity:
    """인증 보안 조치를 테스트합니다."""

    @pytest.mark.asyncio
    async def test_login_wrong_password_generic_error(self, client: AsyncClient,
                                                       registered_user: dict):
        """잘못된 비밀번호로 로그인 시 일반적인 에러를 반환해야 합니다."""
        response = await client.post("/api/v1/auth/login", json={
            "email": registered_user["email"],
            "password": "WrongPassword123!",
        })
        assert response.status_code == 401
        assert "Invalid email or password" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_login_nonexistent_email_generic_error(self, client: AsyncClient):
        """존재하지 않는 이메일로 로그인 시 동일한 에러를 반환해야 합니다."""
        response = await client.post("/api/v1/auth/login", json={
            "email": "nonexistent@example.com",
            "password": "SomePassword123!",
        })
        assert response.status_code == 401
        assert "Invalid email or password" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_registration_duplicate_email_generic_error(
        self, client: AsyncClient, registered_user: dict
    ):
        """중복 등록 시 이메일 존재 여부를 노출하지 않아야 합니다."""
        response = await client.post("/api/v1/auth/register", json={
            "username": "different_user",
            "email": registered_user["email"],
            "password": "NewPassword123!",
        })
        assert response.status_code == 400
        assert "Unable to create account" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_weak_password_rejected(self, client: AsyncClient):
        """취약한 비밀번호는 거부되어야 합니다."""
        weak_passwords = [
            "short1!",          # 너무 짧음
            "alllowercase1!",   # 대문자 없음
            "ALLUPPERCASE1!",   # 소문자 없음
            "NoDigitsHere!",    # 숫자 없음
            "NoSpecial123",     # 특수문자 없음
            "password123!",     # 일반적인 비밀번호
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
    """인가 및 접근 제어를 테스트합니다."""

    @pytest.mark.asyncio
    async def test_unauthenticated_access_denied(self, client: AsyncClient):
        """보호된 엔드포인트는 미인증 요청을 거부해야 합니다."""
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
        """만료된 토큰은 거부되어야 합니다."""
        from app.services.token_service import create_access_token
        from datetime import timedelta

        expired_token = create_access_token(
            user_id="test-id",
            role="user",
            expires_delta=timedelta(seconds=-1),
        )

        response = await client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_malformed_token_rejected(self, client: AsyncClient):
        """잘못된 형식의 토큰은 거부되어야 합니다."""
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
    """입력 검증 및 살균을 테스트합니다."""

    @pytest.mark.asyncio
    async def test_sql_injection_in_login(self, client: AsyncClient):
        """SQL 인젝션 페이로드가 안전하게 처리되어야 합니다."""
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
            assert response.status_code in (401, 422), \
                f"SQLi payload caused error: {payload}"

    @pytest.mark.asyncio
    async def test_xss_in_username(self, client: AsyncClient):
        """사용자명의 XSS 페이로드가 거부되어야 합니다."""
        response = await client.post("/api/v1/auth/register", json={
            "username": "<script>alert(1)</script>",
            "email": "xss@example.com",
            "password": "SafePassword123!",
        })
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_oversized_input_rejected(self, client: AsyncClient):
        """과도하게 큰 입력은 거부되어야 합니다."""
        response = await client.post("/api/v1/auth/register", json={
            "username": "a" * 10000,
            "email": "big@example.com",
            "password": "BigPassword123!",
        })
        assert response.status_code == 422


class TestSecurityHeaders:
    """보안 헤더 존재 여부를 테스트합니다."""

    @pytest.mark.asyncio
    async def test_security_headers_present(self, client: AsyncClient):
        """모든 보안 헤더가 설정되어야 합니다."""
        response = await client.get("/health")

        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert "strict-origin" in response.headers.get("Referrer-Policy", "")
        assert response.headers.get("Content-Security-Policy") is not None

    @pytest.mark.asyncio
    async def test_no_server_header(self, client: AsyncClient):
        """Server 헤더가 구현 세부사항을 노출하지 않아야 합니다."""
        response = await client.get("/health")
        server = response.headers.get("server", "")
        assert "uvicorn" not in server.lower()
        assert "python" not in server.lower()


class TestErrorHandling:
    """에러 처리가 정보를 유출하지 않는지 테스트합니다."""

    @pytest.mark.asyncio
    async def test_404_no_info_leak(self, client: AsyncClient, auth_headers: dict):
        """404 에러가 내부 경로를 노출하지 않아야 합니다."""
        response = await client.get(
            "/api/v1/nonexistent",
            headers=auth_headers,
        )
        body = response.json()
        assert "/app/" not in str(body)
        assert "traceback" not in str(body).lower()
        assert "Traceback" not in str(body)

    @pytest.mark.asyncio
    async def test_500_no_stack_trace(self, client: AsyncClient):
        """500 에러가 스택 트레이스를 노출하지 않아야 합니다."""
        response = await client.get("/health")
        if response.status_code == 500:
            body = response.json()
            assert "traceback" not in str(body).lower()
            assert "File " not in str(body)
```

---

## 14. 배포 고려사항

### 14.1 프로덕션 배포 체크리스트

```
┌──────────────────────────────────────────────────────────────────┐
│              프로덕션 배포 체크리스트                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  HTTPS / TLS:                                                     │
│  [ ] TLS 1.2+ 전용 (TLS 1.0, 1.1 비활성화)                     │
│  [ ] 강력한 암호화 스위트                                         │
│  [ ] 유효한 SSL 인증서 (Let's Encrypt 또는 CA 서명)              │
│  [ ] HSTS 헤더 활성화                                             │
│  [ ] HTTP → HTTPS 리다이렉트                                      │
│                                                                   │
│  리버스 프록시 (nginx):                                           │
│  [ ] 백엔드 서버 헤더 숨김                                        │
│  [ ] 프록시 레벨 속도 제한                                        │
│  [ ] 요청 본문 크기 제한                                          │
│  [ ] 타임아웃 설정                                                │
│  [ ] 접근 로깅                                                    │
│                                                                   │
│  애플리케이션:                                                     │
│  [ ] DEBUG = False                                                │
│  [ ] 프로덕션에서 API 문서 비활성화                                │
│  [ ] 환경 변수 / 시크릿 매니저에서 비밀 관리                      │
│  [ ] 모든 의존성 고정 및 감사 완료                                 │
│  [ ] 에러 처리: 스택 트레이스 미노출                               │
│  [ ] 로깅: 구조화, PII/비밀 미포함                                │
│                                                                   │
│  데이터베이스:                                                     │
│  [ ] DB 사용자에 강력한 비밀번호                                   │
│  [ ] 최소 DB 권한 (admin/superuser 아님)                          │
│  [ ] 암호화된 연결 (SSL/TLS)                                      │
│  [ ] 정기 백업                                                    │
│  [ ] 커넥션 풀링 설정                                             │
│                                                                   │
│  인프라:                                                           │
│  [ ] 방화벽: 필요한 포트만 개방                                    │
│  [ ] SSH 키 기반 인증만 허용                                       │
│  [ ] 정기적인 OS 패치                                              │
│  [ ] 모니터링 및 알림 설정                                         │
│  [ ] 로그 집계 (ELK, Datadog 등)                                  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 14.2 Nginx 리버스 프록시 설정

```nginx
# /etc/nginx/sites-available/secure-api

# HTTP를 HTTPS로 리다이렉트
server {
    listen 80;
    server_name api.example.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS 서버
server {
    listen 443 ssl http2;
    server_name api.example.com;

    # SSL 설정
    ssl_certificate /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # 보안 헤더
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    server_tokens off;

    # 요청 제한
    client_max_body_size 10m;
    client_body_timeout 12;
    client_header_timeout 12;

    # 속도 제한 존
    limit_req_zone $binary_remote_addr zone=api:10m rate=60r/m;

    location / {
        limit_req zone=api burst=20 nodelay;

        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_connect_timeout 5s;
        proxy_read_timeout 30s;
        proxy_send_timeout 30s;

        proxy_hide_header X-Powered-By;
    }

    # 일반적인 공격 경로 차단
    location ~ /\.(git|svn|env|htaccess) {
        deny all;
        return 404;
    }

    access_log /var/log/nginx/api_access.log;
    error_log /var/log/nginx/api_error.log;
}
```

### 14.3 Docker 배포

```dockerfile
# Dockerfile - 더 작은 이미지를 위한 멀티 스테이지 빌드

# 1단계: 의존성 빌드
FROM python:3.12-slim as builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# 2단계: 프로덕션 이미지
FROM python:3.12-slim

# 보안: 비루트 사용자로 실행
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

COPY --from=builder /install /usr/local

WORKDIR /app
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini .

RUN chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

---

## 15. 연습 문제

### 연습 문제 1: Auth Service 구현

다음을 처리하는 `AuthService` 클래스를 완성하세요:
1. Argon2 비밀번호 해싱을 사용한 사용자 생성
2. 인증 (이메일 + 비밀번호 확인)
3. 토큰 쌍 생성 및 리프레시 토큰 DB 저장
4. 리프레시 토큰 로테이션 및 폐기
5. N회 실패 후 계정 잠금

### 연습 문제 2: 2단계 인증 추가

인증 시스템을 TOTP 기반 2FA를 지원하도록 확장하세요:
1. QR 코드 URL을 반환하는 `/auth/2fa/setup` 엔드포인트 추가
2. 설정을 확인하는 `/auth/2fa/verify` 엔드포인트 추가
3. 2FA 활성화 시 로그인에 TOTP 코드 요구하도록 수정
4. 계정 복구용 백업 코드 추가

### 연습 문제 3: API 키 인증

JWT와 함께 API 키 인증 지원을 추가하세요:
1. API 키를 생성하는 엔드포인트 작성
2. `X-API-Key` 헤더를 통한 API 키 인증 지원
3. 키별 속도 제한 구현
4. API 키 로테이션 지원 추가
5. 모든 API 키 사용 로깅

### 연습 문제 4: 완전한 테스트 스위트

다음을 다루는 추가 테스트를 작성하세요:
1. 토큰 갱신 플로우 (로테이션 포함)
2. 실패 후 계정 잠금
3. RBAC: 각 역할이 허용된 엔드포인트에만 접근 가능한지 확인
4. 속도 제한: 제한이 적용되는지 확인
5. CORS: 허용된 출처만 API에 접근 가능한지 확인

### 연습 문제 5: 보안 감사

전체 프로젝트의 보안 감사를 수행하세요:
1. 모든 소스 파일에 대해 Bandit 실행
2. requirements.txt에 대해 pip-audit 실행
3. 하드코딩된 비밀 검사
4. 모든 에러 응답의 정보 유출 검토
5. 모든 입력 검증이 포괄적인지 확인
6. 발견 사항을 문서화하고 수정 방법 작성

### 연습 문제 6: 프로덕션 배포

API를 클라우드 제공자 (또는 로컬 Docker)에 배포하세요:
1. SSL이 적용된 PostgreSQL 설정
2. TLS가 적용된 nginx 리버스 프록시 설정
3. 구조화된 로그 집계 설정
4. 모니터링 및 알림 설정
5. 배포된 API에 대해 보안 스캔 (ZAP 베이스라인) 실행
6. 배포 아키텍처 문서화

---

## 요약

```
┌──────────────────────────────────────────────────────────────────┐
│            보안 API 핵심 요약                                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. 심층 방어: 단일이 아닌 다중 보안 계층                         │
│  2. Argon2id: SHA/MD5/bcrypt가 아닌 적절한 비밀번호 해싱 사용    │
│  3. JWT: 단기 액세스 토큰 + 로테이션되는 리프레시 토큰            │
│  4. RBAC: 세분화된 권한, 최소 권한 원칙                           │
│  5. 입력 검증: 모든 외부 입력을 검증하고 살균                     │
│  6. 속도 제한: 로그인, 등록, 모든 API 보호                        │
│  7. 보안 헤더: HSTS, CSP, X-Frame-Options를 모든 응답에 적용      │
│  8. 에러 처리: 사용자에게 일반적 에러, 내부적으로 상세 로그       │
│  9. 로깅: 구조화된 감사 로그, 비밀이나 PII 미로깅                 │
│ 10. 테스팅: 보안 테스트는 기능 테스트만큼 중요                    │
│ 11. 배포: HTTPS, 리버스 프록시, 비루트 컨테이너                   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

**이전**: [14. 사고 대응과 포렌식](14_Incident_Response.md) | **다음**: [16. 프로젝트: 취약점 스캐너 구축](16_Project_Vulnerability_Scanner.md)
