# Lesson 6: 인증과 인가

**이전**: [Pagination과 Filtering](05_Pagination_and_Filtering.md) | [개요](00_Overview.md) | **다음**: [API Versioning](07_API_Versioning.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 인증(Authentication, 당신이 누구인지)과 인가(Authorization, 무엇을 할 수 있는지)를 구분하기
2. 안전한 생성 및 교체가 가능한 API key 인증 구현하기
3. OAuth 2.0 플로우를 설명하고 각 사용 사례에 적합한 grant type 선택하기
4. JSON Web Token(JWT) 생성, 검증, 갱신하기
5. Scope 기반 및 Role 기반 권한 시스템 설계하기
6. 토큰 관리 및 key 교체를 위한 보안 모범 사례 적용하기

---

보안은 나중에 추가하는 기능이 아닙니다 -- 모든 설계 결정에 영향을 미치는 근본적인 관심사입니다. 단 하나의 인증 우회나 토큰 유출이 전체 시스템을 노출시킬 수 있습니다. 이 레슨에서는 API 보안의 세 가지 기둥을 다룹니다: 간단한 기계 간 접근을 위한 API key, 위임된 인가를 위한 OAuth 2.0, 상태 비저장(stateless) 토큰 기반 인증을 위한 JWT입니다.

> **비유:** 인증은 문 앞에서 신분증을 보여주는 것(당신이 누구인지 증명)입니다. 인가는 내부에서 받는 팔찌(어떤 구역에 접근할 수 있는지 결정)입니다. API key는 건물 출입 카드와 같습니다 -- 통제된 접근을 위해 간단하고 효과적입니다. OAuth는 발렛 파킹 티켓과 같습니다 -- 자동차 열쇠를 넘기지 않고도 제3자에게 제한된 권한을 부여합니다.

## 목차
1. [Authentication vs Authorization](#authentication-vs-authorization)
2. [API Key 인증](#api-key-인증)
3. [OAuth 2.0](#oauth-20)
4. [JSON Web Tokens (JWT)](#json-web-tokens-jwt)
5. [Scope와 권한](#scope와-권한)
6. [토큰 관리](#토큰-관리)
7. [API Key 교체](#api-key-교체)
8. [연습 문제](#연습-문제)

---

## Authentication vs Authorization

```
Authentication (AuthN)              Authorization (AuthZ)
──────────────────────              ────────────────────
"Who are you?"                      "What can you do?"
Verifies identity                   Verifies permissions
Happens FIRST                       Happens AFTER authentication
Methods: API key, OAuth, JWT        Methods: RBAC, ABAC, scopes
Result: Known user/service          Result: Allowed/denied action

Example:
  POST /api/admin/users
  Authorization: Bearer <token>

  Step 1 (AuthN): Decode token → user_id=42, verified ✓
  Step 2 (AuthZ): user_id=42 has role="admin" → allowed ✓
```

### 일반적인 인증 방법

| 방법 | 사용 사례 | 복잡도 | 보안 수준 |
|--------|----------|------------|----------|
| API Key | 서버 간 통신, 간단한 연동 | 낮음 | 중간 |
| Basic Auth | 내부 도구, 개발 환경 | 낮음 | 낮음 |
| OAuth 2.0 | 제3자 접근, 사용자 위임 | 높음 | 높음 |
| JWT (Bearer Token) | Stateless API, 마이크로서비스 | 중간 | 높음 |
| mTLS | 서비스 메시, 높은 보안 요구 | 높음 | 매우 높음 |

---

## API Key 인증

### 안전한 API Key 생성

```python
import secrets
import hashlib
from datetime import datetime, timezone

def generate_api_key(prefix: str = "sk") -> tuple[str, str]:
    """
    Generate a secure API key with a prefix.

    Returns:
        (full_key, hashed_key)
        - full_key: shown to the user once (e.g., "sk_live_abc123...")
        - hashed_key: stored in the database
    """
    # Generate 32 bytes of random data (256 bits)
    random_part = secrets.token_urlsafe(32)
    full_key = f"{prefix}_live_{random_part}"

    # Store only the hash in the database
    hashed_key = hashlib.sha256(full_key.encode()).hexdigest()

    return full_key, hashed_key

# Example usage:
# full_key, hashed_key = generate_api_key("sk")
# Show full_key to the user ONCE: "sk_live_EXAMPLE_KEY_REPLACE_ME..."
# Store hashed_key in database: "a3f2b8c4..."
```

### FastAPI에서 API Key 검증

```python
from fastapi import FastAPI, Security, HTTPException, Depends
from fastapi.security import APIKeyHeader
import hashlib

app = FastAPI()

# Define the API key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Simulated database of hashed keys
API_KEY_DB = {
    # hash of "sk_live_test123" -> user info
    hashlib.sha256(b"sk_live_test123").hexdigest(): {
        "user_id": 1,
        "name": "Test App",
        "scopes": ["read:products", "write:orders"],
        "rate_limit": 1000,
    }
}

async def verify_api_key(api_key: str = Security(api_key_header)) -> dict:
    """Validate the API key and return the associated user/app info."""
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={"code": "MISSING_API_KEY", "message": "X-API-Key header is required"},
        )

    # Hash the provided key and look it up
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    app_info = API_KEY_DB.get(key_hash)

    if not app_info:
        raise HTTPException(
            status_code=401,
            detail={"code": "INVALID_API_KEY", "message": "The provided API key is invalid"},
        )

    return app_info

@app.get("/api/products")
async def list_products(app_info: dict = Depends(verify_api_key)):
    """Protected endpoint -- requires valid API key."""
    return {
        "data": [{"id": 1, "name": "Widget"}],
        "authenticated_as": app_info["name"],
    }
```

### Flask에서 API Key 사용

```python
from flask import Flask, request, jsonify, g
from functools import wraps
import hashlib

app = Flask(__name__)

def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return jsonify({
                "error": {"code": "MISSING_API_KEY", "message": "X-API-Key header is required"}
            }), 401

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        # Look up in database
        app_info = lookup_api_key(key_hash)  # your DB lookup function
        if not app_info:
            return jsonify({
                "error": {"code": "INVALID_API_KEY", "message": "Invalid API key"}
            }), 401

        g.app_info = app_info
        return f(*args, **kwargs)
    return decorated

@app.get("/api/products")
@require_api_key
def list_products():
    return jsonify({"data": [], "authenticated_as": g.app_info["name"]})
```

---

## OAuth 2.0

OAuth 2.0은 **위임된 인가**를 위한 프레임워크입니다 -- 제3자 애플리케이션이 사용자의 비밀번호를 공유하지 않고도 사용자를 대신하여 리소스에 접근할 수 있도록 합니다.

### OAuth 2.0 역할

```
Resource Owner (User)         -- The person who owns the data
Client (Third-Party App)      -- The application requesting access
Authorization Server          -- Issues tokens (e.g., Auth0, Keycloak)
Resource Server (Your API)    -- Protects the data, validates tokens
```

### Authorization Code Flow (가장 일반적)

```
User        Client App       Auth Server       Resource Server
 |               |                |                    |
 |  1. Click "Login with X"      |                    |
 |──────────────►|                |                    |
 |               |  2. Redirect to auth server         |
 |               |───────────────►|                    |
 |  3. User logs in & consents   |                    |
 |───────────────────────────────►|                    |
 |               |  4. Auth code  |                    |
 |               |◄───────────────|                    |
 |               |  5. Exchange code for token         |
 |               |───────────────►|                    |
 |               |  6. Access token + refresh token    |
 |               |◄───────────────|                    |
 |               |  7. API call with access token      |
 |               |────────────────────────────────────►|
 |               |  8. Protected resource              |
 |               |◄────────────────────────────────────|
```

### Token 교환 구현 (서버 측)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

app = FastAPI()

OAUTH_CONFIG = {
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "token_url": "https://auth.example.com/oauth/token",
    "redirect_uri": "https://yourapp.com/callback",
}

class TokenRequest(BaseModel):
    code: str         # authorization code from step 4
    redirect_uri: str

@app.post("/api/auth/callback")
async def oauth_callback(token_req: TokenRequest):
    """Exchange authorization code for access token."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            OAUTH_CONFIG["token_url"],
            data={
                "grant_type": "authorization_code",
                "code": token_req.code,
                "redirect_uri": token_req.redirect_uri,
                "client_id": OAUTH_CONFIG["client_id"],
                "client_secret": OAUTH_CONFIG["client_secret"],
            },
        )

    if response.status_code != 200:
        raise HTTPException(status_code=401, detail="Token exchange failed")

    token_data = response.json()
    return {
        "access_token": token_data["access_token"],
        "token_type": "Bearer",
        "expires_in": token_data["expires_in"],
        "refresh_token": token_data.get("refresh_token"),
        "scope": token_data.get("scope"),
    }
```

### Client Credentials Flow (기계 간 통신)

```python
@app.post("/api/auth/token")
async def client_credentials():
    """
    Client Credentials flow: machine-to-machine authentication.
    No user involved -- the client authenticates directly.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            OAUTH_CONFIG["token_url"],
            data={
                "grant_type": "client_credentials",
                "client_id": OAUTH_CONFIG["client_id"],
                "client_secret": OAUTH_CONFIG["client_secret"],
                "scope": "read:products write:orders",
            },
        )

    if response.status_code != 200:
        raise HTTPException(status_code=401, detail="Authentication failed")

    return response.json()
```

### OAuth 2.0 Grant Type 요약

| Grant Type | 사용 사례 | 사용자 관여 여부 |
|-----------|----------|----------------|
| Authorization Code | 웹 앱, 모바일 앱 | 예 |
| Authorization Code + PKCE | SPA, 네이티브 앱 (client secret 없음) | 예 |
| Client Credentials | 서버 간 통신, 마이크로서비스 | 아니요 |
| Device Code | 스마트 TV, CLI 도구, IoT | 예 (다른 기기에서) |
| Refresh Token | 재로그인 없이 접근 연장 | 아니요 (기존 grant 사용) |

---

## JSON Web Tokens (JWT)

### JWT 구조

JWT는 점(`.`)으로 구분된 세 부분으로 구성됩니다: `header.payload.signature`

```python
import jwt
import json
import base64
from datetime import datetime, timedelta, timezone

# --- JWT Structure ---
# Header: algorithm and token type
header = {"alg": "HS256", "typ": "JWT"}

# Payload: claims (data)
payload = {
    "sub": "user_42",               # subject (user ID)
    "name": "Alice",                # custom claim
    "email": "alice@example.com",   # custom claim
    "role": "admin",                # custom claim
    "iat": 1706000000,              # issued at (Unix timestamp)
    "exp": 1706003600,              # expiration (1 hour later)
    "iss": "https://api.example.com",  # issuer
    "aud": "https://api.example.com",  # audience
}

# Signature: HMAC-SHA256(base64url(header) + "." + base64url(payload), secret)
# The signature ensures the token has not been tampered with.
```

### JWT 생성 및 검증

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt
from datetime import datetime, timedelta, timezone

app = FastAPI()
security = HTTPBearer()

SECRET_KEY = "your-256-bit-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 30

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int

def create_access_token(user_id: str, role: str, scopes: list[str]) -> str:
    """Create a short-lived access token."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "role": role,
        "scopes": scopes,
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        "iss": "https://api.example.com",
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(user_id: str) -> str:
    """Create a long-lived refresh token."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "type": "refresh",
        "iat": now,
        "exp": now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        "iss": "https://api.example.com",
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

@app.post("/api/auth/login", response_model=TokenResponse)
async def login(credentials: LoginRequest):
    """Authenticate user and issue JWT tokens."""
    # In practice: verify credentials against database
    if credentials.email != "alice@example.com" or credentials.password != "secret":
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user_id = "user_42"
    role = "admin"
    scopes = ["read:users", "write:users", "read:orders"]

    access_token = create_access_token(user_id, role, scopes)
    refresh_token = create_refresh_token(user_id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
```

### JWT 검증 미들웨어

```python
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Validate JWT and extract user information."""
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            issuer="https://api.example.com",
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail={"code": "TOKEN_EXPIRED", "message": "Access token has expired"},
            headers={"WWW-Authenticate": "Bearer error='invalid_token'"},
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=401,
            detail={"code": "INVALID_TOKEN", "message": str(e)},
            headers={"WWW-Authenticate": "Bearer"},
        )

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=401,
            detail={"code": "WRONG_TOKEN_TYPE", "message": "Expected an access token"},
        )

    return {
        "user_id": payload["sub"],
        "role": payload["role"],
        "scopes": payload.get("scopes", []),
    }

@app.get("/api/me")
async def get_profile(user: dict = Depends(get_current_user)):
    """Protected endpoint: requires valid JWT."""
    return {"user_id": user["user_id"], "role": user["role"]}
```

### Token 갱신

```python
class RefreshRequest(BaseModel):
    refresh_token: str

@app.post("/api/auth/refresh", response_model=TokenResponse)
async def refresh_tokens(body: RefreshRequest):
    """Exchange a refresh token for new access and refresh tokens."""
    try:
        payload = jwt.decode(
            body.refresh_token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            issuer="https://api.example.com",
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail={"code": "REFRESH_EXPIRED", "message": "Refresh token expired. Please log in again."},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Expected a refresh token")

    user_id = payload["sub"]
    # In practice: look up user from database for current role/scopes
    role = "admin"
    scopes = ["read:users", "write:users", "read:orders"]

    # Issue new token pair (rotate refresh token for security)
    new_access = create_access_token(user_id, role, scopes)
    new_refresh = create_refresh_token(user_id)

    return TokenResponse(
        access_token=new_access,
        refresh_token=new_refresh,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
```

---

## Scope와 권한

### Scope 기반 접근 제어

```python
from functools import wraps

def require_scopes(*required_scopes: str):
    """Dependency that checks if the user has the required scopes."""
    async def check_scopes(user: dict = Depends(get_current_user)):
        user_scopes = set(user.get("scopes", []))
        missing = set(required_scopes) - user_scopes
        if missing:
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "INSUFFICIENT_SCOPE",
                    "message": f"Missing required scopes: {', '.join(missing)}",
                    "required": list(required_scopes),
                    "granted": list(user_scopes),
                },
            )
        return user
    return check_scopes

# Usage: protect endpoints with specific scopes
@app.get("/api/users")
async def list_users(user: dict = Depends(require_scopes("read:users"))):
    """Requires read:users scope."""
    return {"data": [{"id": 1, "name": "Alice"}]}

@app.post("/api/users")
async def create_user(user: dict = Depends(require_scopes("write:users"))):
    """Requires write:users scope."""
    return {"id": 2, "name": "Bob"}

@app.delete("/api/users/{user_id}")
async def delete_user(
    user_id: int,
    user: dict = Depends(require_scopes("write:users", "admin:users")),
):
    """Requires BOTH write:users AND admin:users scopes."""
    return {"deleted": True}
```

### Role 기반 접근 제어 (RBAC)

```python
from enum import Enum

class Role(str, Enum):
    VIEWER = "viewer"
    EDITOR = "editor"
    ADMIN = "admin"

# Role hierarchy: admin > editor > viewer
ROLE_HIERARCHY = {
    Role.VIEWER: 0,
    Role.EDITOR: 1,
    Role.ADMIN: 2,
}

def require_role(minimum_role: Role):
    """Dependency that checks the user's role against a minimum level."""
    async def check_role(user: dict = Depends(get_current_user)):
        user_role = Role(user["role"])
        if ROLE_HIERARCHY[user_role] < ROLE_HIERARCHY[minimum_role]:
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "INSUFFICIENT_ROLE",
                    "message": f"Requires {minimum_role.value} role or higher",
                    "your_role": user_role.value,
                },
            )
        return user
    return check_role

@app.get("/api/reports")
async def view_reports(user: dict = Depends(require_role(Role.VIEWER))):
    """Any authenticated user can view reports."""
    return {"data": []}

@app.post("/api/reports")
async def create_report(user: dict = Depends(require_role(Role.EDITOR))):
    """Only editors and admins can create reports."""
    return {"id": 1, "title": "Monthly Report"}

@app.delete("/api/reports/{report_id}")
async def delete_report(
    report_id: int,
    user: dict = Depends(require_role(Role.ADMIN)),
):
    """Only admins can delete reports."""
    return {"deleted": True}
```

### Scope 네이밍 규칙

```python
# Common scope naming patterns:

# resource:action
SCOPES = {
    "read:users":      "View user profiles",
    "write:users":     "Create and update users",
    "delete:users":    "Delete users",
    "read:orders":     "View orders",
    "write:orders":    "Create and update orders",
    "admin:settings":  "Modify system settings",
}

# Wildcard scopes
# "users:*"     -- all user permissions
# "*:read"      -- read access to everything
# "*"           -- full access (dangerous, use sparingly)
```

---

## 토큰 관리

### 보안 모범 사례

```python
# 1. Short-lived access tokens (15-30 minutes)
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 2. Long-lived refresh tokens (30 days) stored securely
REFRESH_TOKEN_EXPIRE_DAYS = 30

# 3. Token rotation: issue a new refresh token with each refresh
# This limits the damage window if a refresh token is compromised.

# 4. Token revocation: maintain a blocklist for logged-out tokens
TOKEN_BLOCKLIST: set[str] = set()  # use Redis in production

async def get_current_user_with_revocation(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """JWT validation with revocation check."""
    token = credentials.credentials

    # Check blocklist before decoding
    if token in TOKEN_BLOCKLIST:
        raise HTTPException(status_code=401, detail="Token has been revoked")

    # ... normal JWT validation ...
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return {"user_id": payload["sub"], "role": payload["role"]}

@app.post("/api/auth/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Revoke the current access token."""
    TOKEN_BLOCKLIST.add(credentials.credentials)
    return {"message": "Successfully logged out"}

# 5. Never store tokens in:
#    - URL query parameters (logged in server logs, browser history)
#    - localStorage (vulnerable to XSS)
#
# Preferred storage:
#    - httpOnly, secure, SameSite cookies (for web apps)
#    - Secure storage (Keychain/Keystore for mobile apps)
#    - Memory only (for SPAs, with silent refresh)
```

### Token Claims 모범 사례

```python
# DO include:
payload = {
    "sub": "user_42",         # unique identifier
    "role": "editor",         # authorization level
    "scopes": ["read:data"],  # fine-grained permissions
    "iat": 1706000000,        # issued at
    "exp": 1706001800,        # expiration
    "iss": "api.example.com", # issuer
    "aud": "api.example.com", # audience
}

# DO NOT include:
bad_payload = {
    "password": "hashed_pw",  # NEVER put credentials in JWT
    "ssn": "123-45-6789",     # NEVER put PII in JWT
    "credit_card": "4111...", # NEVER put financial data in JWT
    # JWTs are encoded (base64), NOT encrypted.
    # Anyone can decode and read the payload.
}
```

---

## API Key 교체

### 다운타임 없는 Key 교체

```python
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timezone, timedelta
import secrets
import hashlib

app = FastAPI()

class ApiKeyRecord(BaseModel):
    key_hash: str
    prefix: str        # first 8 chars for identification
    created_at: str
    expires_at: str | None
    status: str        # "active" | "deprecated" | "revoked"
    label: str

# Simulated key store
KEY_STORE: dict[str, ApiKeyRecord] = {}

@app.post("/api/keys/rotate")
async def rotate_api_key(
    old_key_prefix: str,
    user: dict = Depends(get_current_user),
):
    """
    Rotate an API key with zero downtime:
    1. Create a new key (active)
    2. Mark the old key as deprecated (still works for grace period)
    3. After grace period, revoke the old key
    """
    # Generate new key
    new_key = f"sk_live_{secrets.token_urlsafe(32)}"
    new_hash = hashlib.sha256(new_key.encode()).hexdigest()
    now = datetime.now(timezone.utc)

    # Create new key record
    KEY_STORE[new_hash] = ApiKeyRecord(
        key_hash=new_hash,
        prefix=new_key[:12],
        created_at=now.isoformat(),
        expires_at=None,
        status="active",
        label="Primary key",
    )

    # Deprecate old key (grace period: 7 days)
    for key_hash, record in KEY_STORE.items():
        if record.prefix == old_key_prefix and record.status == "active":
            record.status = "deprecated"
            record.expires_at = (now + timedelta(days=7)).isoformat()

    return {
        "new_key": new_key,  # show ONCE
        "message": "New key created. Old key will remain valid for 7 days.",
        "old_key_expires_at": (now + timedelta(days=7)).isoformat(),
    }

@app.get("/api/keys")
async def list_api_keys(user: dict = Depends(get_current_user)):
    """List all API keys (showing only prefixes, never full keys)."""
    return {
        "data": [
            {
                "prefix": record.prefix,
                "status": record.status,
                "created_at": record.created_at,
                "expires_at": record.expires_at,
                "label": record.label,
            }
            for record in KEY_STORE.values()
        ]
    }
```

---

## 연습 문제

### 연습 문제 1: API Key 인증 시스템

다음 기능을 갖춘 완전한 API key 관리 시스템을 구축하십시오:
- `POST /api/keys` -- 새 API key 생성 (한 번만 표시, 해시 저장)
- `GET /api/keys` -- 모든 key 목록 조회 (prefix만)
- `DELETE /api/keys/{prefix}` -- key 폐기
- 모든 보호된 엔드포인트에서 `X-API-Key` 헤더 검증
- 각 key의 마지막 사용 시각 기록

### 연습 문제 2: JWT 인증

FastAPI에서 완전한 JWT 인증 플로우를 구현하십시오:
- `POST /api/auth/register` -- 새 사용자 생성 (bcrypt로 비밀번호 해시)
- `POST /api/auth/login` -- 자격 증명 검증, access + refresh 토큰 반환
- `POST /api/auth/refresh` -- refresh 토큰을 새 토큰 쌍으로 교환
- `POST /api/auth/logout` -- 현재 토큰 폐기
- `GET /api/me` -- 현재 사용자 프로필 반환 (보호됨)

### 연습 문제 3: Scope 기반 인가

프로젝트 관리 API를 위한 scope 시스템을 설계하고 구현하십시오:
- Scope 정의: `read:projects`, `write:projects`, `read:tasks`, `write:tasks`, `admin:*`
- 세 가지 role 생성: viewer (읽기 전용), member (읽기 + 쓰기), admin (모든 권한)
- 각 엔드포인트에 적절한 scope 적용
- 누락된 scope를 나열하는 명확한 403 오류 반환

### 연습 문제 4: OAuth 2.0 클라이언트

OAuth 2.0 클라이언트 역할을 하는 FastAPI 애플리케이션을 구축하십시오:
- 사용자를 GitHub로 리디렉션하여 인증
- Authorization code 교환으로 콜백 처리
- Access token 저장 및 사용자의 GitHub 프로필 가져오기
- 사용자의 리포지토리 표시

### 연습 문제 5: Key 교체

다음 기능을 갖춘 API key 교체를 구현하십시오:
- 다운타임 없는 교체 (유예 기간 동안 이전 key 사용 가능)
- 설정 가능한 유예 기간 (기본 7일)
- 유예 기간 만료 후 자동 폐기
- 더 이상 사용되지 않는 key 사용 시 알림 (로그 메시지)
- 모든 key 작업의 감사 로그 (생성, 지원 중단, 폐기)

---

## 요약

이 레슨에서 다룬 내용:
1. 인증(신원)과 인가(권한)의 구분
2. API key 인증: 생성, 검증, 저장, 교체
3. OAuth 2.0 플로우: authorization code, client credentials, token refresh
4. JWT 구조, 생성, 검증, refresh token 패턴
5. Scope 기반 및 role 기반 접근 제어와 계층적 권한
6. 보안 모범 사례: 단기 토큰, 교체, 폐기, 안전한 저장

---

**이전**: [Pagination과 Filtering](05_Pagination_and_Filtering.md) | [개요](00_Overview.md) | **다음**: [API Versioning](07_API_Versioning.md)

**License**: CC BY-NC 4.0
