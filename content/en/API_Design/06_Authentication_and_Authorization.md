# Lesson 6: Authentication and Authorization

**Previous**: [Pagination and Filtering](05_Pagination_and_Filtering.md) | [Overview](00_Overview.md) | **Next**: [API Versioning](07_API_Versioning.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between authentication (who you are) and authorization (what you can do)
2. Implement API key authentication with secure generation and rotation
3. Explain OAuth 2.0 flows and choose the right grant type for each use case
4. Create, validate, and refresh JSON Web Tokens (JWTs)
5. Design scope-based and role-based permission systems
6. Apply security best practices for token management and key rotation

---

Security is not a feature you add later -- it is a foundational concern that shapes every design decision. A single authentication bypass or token leak can expose your entire system. This lesson covers the three pillars of API security: API keys for simple machine-to-machine access, OAuth 2.0 for delegated authorization, and JWTs for stateless token-based authentication.

> **Analogy:** Authentication is showing your ID at the door (proving who you are). Authorization is the wristband you get inside (determining which areas you can access). API keys are like building keycards -- simple and effective for controlled access. OAuth is like a valet ticket -- you give limited permission to a third party without handing over your car keys.

## Table of Contents
1. [Authentication vs Authorization](#authentication-vs-authorization)
2. [API Key Authentication](#api-key-authentication)
3. [OAuth 2.0](#oauth-20)
4. [JSON Web Tokens (JWT)](#json-web-tokens-jwt)
5. [Scopes and Permissions](#scopes-and-permissions)
6. [Token Management](#token-management)
7. [API Key Rotation](#api-key-rotation)
8. [Exercises](#exercises)

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

### Common Authentication Methods

| Method | Use Case | Complexity | Security |
|--------|----------|------------|----------|
| API Key | Server-to-server, simple integrations | Low | Medium |
| Basic Auth | Internal tools, development | Low | Low |
| OAuth 2.0 | Third-party access, user delegation | High | High |
| JWT (Bearer Token) | Stateless APIs, microservices | Medium | High |
| mTLS | Service mesh, high-security | High | Very High |

---

## API Key Authentication

### Generating Secure API Keys

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

### Validating API Keys in FastAPI

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

### API Keys in Flask

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

OAuth 2.0 is a framework for **delegated authorization** -- it allows a third-party application to access resources on behalf of a user, without the user sharing their password.

### OAuth 2.0 Roles

```
Resource Owner (User)         -- The person who owns the data
Client (Third-Party App)      -- The application requesting access
Authorization Server          -- Issues tokens (e.g., Auth0, Keycloak)
Resource Server (Your API)    -- Protects the data, validates tokens
```

### Authorization Code Flow (Most Common)

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

### Implementing the Token Exchange (Server-Side)

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

### Client Credentials Flow (Machine-to-Machine)

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

### OAuth 2.0 Grant Types Summary

| Grant Type | Use Case | User Involved? |
|-----------|----------|----------------|
| Authorization Code | Web apps, mobile apps | Yes |
| Authorization Code + PKCE | SPAs, native apps (no client secret) | Yes |
| Client Credentials | Server-to-server, microservices | No |
| Device Code | Smart TVs, CLI tools, IoT | Yes (on another device) |
| Refresh Token | Extend access without re-login | No (uses existing grant) |

---

## JSON Web Tokens (JWT)

### JWT Structure

A JWT consists of three parts separated by dots: `header.payload.signature`

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

### Creating and Validating JWTs

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

### JWT Validation Middleware

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

### Token Refresh

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

## Scopes and Permissions

### Scope-Based Access Control

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

### Role-Based Access Control (RBAC)

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

### Scope Naming Conventions

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

## Token Management

### Security Best Practices

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

### Token Claims Best Practices

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

## API Key Rotation

### Zero-Downtime Key Rotation

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

## Exercises

### Exercise 1: API Key Authentication System

Build a complete API key management system with:
- `POST /api/keys` -- generate a new API key (show it once, store the hash)
- `GET /api/keys` -- list all keys (prefixes only)
- `DELETE /api/keys/{prefix}` -- revoke a key
- All protected endpoints validate the `X-API-Key` header
- Log the last used timestamp for each key

### Exercise 2: JWT Authentication

Implement a full JWT authentication flow in FastAPI:
- `POST /api/auth/register` -- create a new user (hash password with bcrypt)
- `POST /api/auth/login` -- verify credentials, return access + refresh tokens
- `POST /api/auth/refresh` -- exchange refresh token for new token pair
- `POST /api/auth/logout` -- revoke the current token
- `GET /api/me` -- return current user profile (protected)

### Exercise 3: Scope-Based Authorization

Design and implement a scope system for a project management API:
- Define scopes: `read:projects`, `write:projects`, `read:tasks`, `write:tasks`, `admin:*`
- Create three roles: viewer (read only), member (read + write), admin (everything)
- Protect each endpoint with the appropriate scopes
- Return clear 403 errors listing missing scopes

### Exercise 4: OAuth 2.0 Client

Build a FastAPI application that acts as an OAuth 2.0 client:
- Redirect users to GitHub for authentication
- Handle the callback with authorization code exchange
- Store the access token and use it to fetch the user's GitHub profile
- Display the user's repositories

### Exercise 5: Key Rotation

Implement API key rotation with:
- Zero-downtime rotation (old key works during grace period)
- Configurable grace period (default 7 days)
- Automatic revocation after grace period expires
- Notification (log message) when a deprecated key is used
- Audit log of all key operations (create, deprecate, revoke)

---

## Summary

This lesson covered:
1. The distinction between authentication (identity) and authorization (permissions)
2. API key authentication: generation, validation, storage, and rotation
3. OAuth 2.0 flows: authorization code, client credentials, and token refresh
4. JWT structure, creation, validation, and refresh token patterns
5. Scope-based and role-based access control with hierarchical permissions
6. Security best practices: short-lived tokens, rotation, revocation, and secure storage

---

**Previous**: [Pagination and Filtering](05_Pagination_and_Filtering.md) | [Overview](00_Overview.md) | **Next**: [API Versioning](07_API_Versioning.md)

**License**: CC BY-NC 4.0
