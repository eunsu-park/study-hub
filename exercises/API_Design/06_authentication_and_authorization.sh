#!/bin/bash
# Exercises for Lesson 06: Authentication and Authorization
# Topic: API_Design
# Solutions to practice problems from the lesson.

# === Exercise 1: API Key Authentication ===
# Problem: Implement API key authentication middleware that validates keys
# from a header and returns proper 401/403 responses.
exercise_1() {
    echo "=== Exercise 1: API Key Authentication ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

app = FastAPI()

# API key registry (in production, store hashed keys in a database)
API_KEYS = {
    "sk_live_abc123": {"name": "Frontend App", "scopes": ["read"]},
    "sk_live_xyz789": {"name": "Admin Dashboard", "scopes": ["read", "write", "admin"]},
}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(
    api_key: str = Security(api_key_header),
) -> dict:
    """Validate API key and return the associated client info."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include X-API-Key header.",
        )
    client = API_KEYS.get(api_key)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    return client


def require_scope(required: str):
    """Factory for scope-checking dependencies."""
    def checker(client: dict = Depends(get_api_key)):
        if required not in client["scopes"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Scope '{required}' required. Client has: {client['scopes']}",
            )
        return client
    return checker


@app.get("/api/v1/data")
def read_data(client: dict = Depends(require_scope("read"))):
    """Read-only endpoint — requires 'read' scope."""
    return {"data": [1, 2, 3], "client": client["name"]}


@app.post("/api/v1/data")
def write_data(client: dict = Depends(require_scope("write"))):
    """Write endpoint — requires 'write' scope."""
    return {"status": "created", "client": client["name"]}


@app.delete("/api/v1/data/{item_id}")
def delete_data(item_id: str, client: dict = Depends(require_scope("admin"))):
    """Admin endpoint — requires 'admin' scope."""
    return {"status": "deleted", "client": client["name"]}

# Test:
# http GET :8000/api/v1/data X-API-Key:sk_live_abc123  → 200 OK
# http POST :8000/api/v1/data X-API-Key:sk_live_abc123 → 403 (no write scope)
# http POST :8000/api/v1/data X-API-Key:sk_live_xyz789 → 200 OK
SOLUTION
}

# === Exercise 2: JWT Token Lifecycle ===
# Problem: Implement JWT access + refresh token flow including token
# expiration, refresh, and revocation.
exercise_2() {
    echo "=== Exercise 2: JWT Token Lifecycle ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import jwt
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

app = FastAPI()
SECRET = "change-me-in-production"

# Token blacklist for revocation (in production, use Redis)
revoked_tokens: set[str] = set()
security = HTTPBearer()


def create_tokens(user_id: str, role: str) -> dict:
    """Create access + refresh token pair."""
    now = datetime.now(timezone.utc)

    access_token = jwt.encode(
        {"sub": user_id, "role": role, "type": "access",
         "exp": now + timedelta(minutes=15), "iat": now},
        SECRET, algorithm="HS256",
    )
    refresh_token = jwt.encode(
        {"sub": user_id, "type": "refresh",
         "exp": now + timedelta(days=7), "iat": now,
         "jti": f"refresh-{user_id}-{now.timestamp()}"},
        SECRET, algorithm="HS256",
    )
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": 900,
    }


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Validate access token."""
    token = creds.credentials

    # Check revocation
    if token in revoked_tokens:
        raise HTTPException(status_code=401, detail="Token has been revoked")

    try:
        payload = jwt.decode(token, SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Not an access token")

    return {"user_id": payload["sub"], "role": payload["role"]}


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/v1/auth/login")
def login(body: LoginRequest):
    # Validate credentials (simplified)
    if body.username == "alice" and body.password == "password":
        return create_tokens("user_1", "admin")
    raise HTTPException(status_code=401, detail="Invalid credentials")


class RefreshRequest(BaseModel):
    refresh_token: str


@app.post("/api/v1/auth/refresh")
def refresh(body: RefreshRequest):
    """Exchange refresh token for new token pair."""
    try:
        payload = jwt.decode(body.refresh_token, SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired — login again")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=400, detail="Not a refresh token")

    # Revoke old refresh token (rotation)
    revoked_tokens.add(body.refresh_token)

    return create_tokens(payload["sub"], "admin")


@app.post("/api/v1/auth/logout")
def logout(creds: HTTPAuthorizationCredentials = Depends(security)):
    """Revoke the current access token."""
    revoked_tokens.add(creds.credentials)
    return {"message": "Logged out"}


@app.get("/api/v1/protected")
def protected(user: dict = Depends(get_current_user)):
    return {"message": f"Hello {user['user_id']}", "role": user["role"]}
SOLUTION
}

# === Exercise 3: OAuth 2.0 Scopes ===
# Problem: Design an OAuth 2.0 scope hierarchy for a project management API
# and implement scope-based access control.
exercise_3() {
    echo "=== Exercise 3: OAuth 2.0 Scopes ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

app = FastAPI()

# OAuth 2.0 Scope Hierarchy for a Project Management API
SCOPE_DEFINITIONS = {
    "projects:read":     "View projects and their details",
    "projects:write":    "Create and update projects",
    "projects:delete":   "Delete projects",
    "tasks:read":        "View tasks",
    "tasks:write":       "Create and update tasks",
    "tasks:assign":      "Assign tasks to team members",
    "users:read":        "View user profiles",
    "users:manage":      "Create/update/delete users (admin)",
    "admin":             "Full administrative access",
}

# Scope hierarchy: admin includes everything
SCOPE_HIERARCHY = {
    "admin": set(SCOPE_DEFINITIONS.keys()),
    "projects:write": {"projects:read"},    # write implies read
    "projects:delete": {"projects:read"},
    "tasks:write": {"tasks:read"},
    "tasks:assign": {"tasks:read", "users:read"},
    "users:manage": {"users:read"},
}


def resolve_scopes(granted: set[str]) -> set[str]:
    """Expand scope hierarchy — write implies read, admin implies all."""
    resolved = set(granted)
    for scope in granted:
        if scope in SCOPE_HIERARCHY:
            resolved |= SCOPE_HIERARCHY[scope]
    return resolved


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


def require_scopes(*required_scopes: str):
    """Dependency that checks if the token has the required scopes."""
    def checker(token: str = Depends(oauth2_scheme)):
        # In production, decode JWT and extract scopes
        # For demo, simulate token → scopes mapping
        token_scopes = {"projects:read", "tasks:write"}  # From JWT
        effective = resolve_scopes(token_scopes)

        missing = set(required_scopes) - effective
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing scopes: {', '.join(missing)}",
                headers={"WWW-Authenticate": f'Bearer scope="{" ".join(required_scopes)}"'},
            )
        return token
    return checker


@app.get("/api/v1/projects")
def list_projects(_: str = Depends(require_scopes("projects:read"))):
    return {"projects": []}


@app.post("/api/v1/projects")
def create_project(_: str = Depends(require_scopes("projects:write"))):
    return {"created": True}


@app.delete("/api/v1/projects/{id}")
def delete_project(id: str, _: str = Depends(require_scopes("projects:delete"))):
    return {"deleted": True}


@app.post("/api/v1/tasks/{id}/assign")
def assign_task(id: str, _: str = Depends(require_scopes("tasks:assign"))):
    return {"assigned": True}
SOLUTION
}

# === Exercise 4: Rate-Limited Auth Endpoint ===
# Problem: Implement login with brute-force protection (account lockout
# after N failed attempts).
exercise_4() {
    echo "=== Exercise 4: Rate-Limited Auth Endpoint ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import time
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 300  # 5 minutes

# Track failed login attempts per username
login_attempts: dict[str, dict] = defaultdict(
    lambda: {"count": 0, "locked_until": 0}
)

users = {"alice": {"password_hash": "hashed_password", "id": "1"}}


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/v1/auth/login")
def login(body: LoginRequest):
    """Login with brute-force protection.

    After 5 failed attempts, the account is locked for 5 minutes.
    This prevents credential stuffing and brute-force attacks.
    """
    attempt = login_attempts[body.username]

    # Check lockout
    if attempt["locked_until"] > time.time():
        remaining = int(attempt["locked_until"] - time.time())
        raise HTTPException(
            status_code=429,
            detail=f"Account locked. Try again in {remaining} seconds.",
            headers={"Retry-After": str(remaining)},
        )

    # Validate credentials
    user = users.get(body.username)
    if not user or body.password != "correct_password":  # Simplified check
        attempt["count"] += 1

        if attempt["count"] >= MAX_ATTEMPTS:
            attempt["locked_until"] = time.time() + LOCKOUT_SECONDS
            attempt["count"] = 0
            raise HTTPException(
                status_code=429,
                detail=f"Too many failed attempts. Account locked for {LOCKOUT_SECONDS}s.",
                headers={"Retry-After": str(LOCKOUT_SECONDS)},
            )

        remaining_attempts = MAX_ATTEMPTS - attempt["count"]
        raise HTTPException(
            status_code=401,
            detail=f"Invalid credentials. {remaining_attempts} attempts remaining.",
        )

    # Success — reset counter
    attempt["count"] = 0
    attempt["locked_until"] = 0

    return {"access_token": "jwt_token_here", "token_type": "bearer"}
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 06: Authentication and Authorization"
echo "===================================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
