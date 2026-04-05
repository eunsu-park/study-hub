#!/usr/bin/env python3
"""Example: JWT Authentication

Demonstrates JWT-based API authentication with FastAPI:
- User registration and login
- Access token + refresh token generation
- Protected endpoints with dependency injection
- Token refresh flow
- Role-based access control (RBAC)

Related lesson: 06_Authentication_and_Authorization.md

Run:
    pip install "fastapi[standard]" pyjwt "passlib[bcrypt]"
    uvicorn 03_authentication_jwt:app --reload --port 8000

Test:
    # Register
    http POST :8000/api/v1/auth/register username=alice password=secret123 email=alice@example.com

    # Login (get tokens)
    http POST :8000/api/v1/auth/login username=alice password=secret123

    # Access protected endpoint
    http GET :8000/api/v1/users/me "Authorization:Bearer <access_token>"
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field

# =============================================================================
# CONFIGURATION
# =============================================================================
# In production, load from environment variables or a secrets manager.
# Never hardcode secrets in source code.

SECRET_KEY = "your-256-bit-secret-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# =============================================================================
# PASSWORD HASHING
# =============================================================================
# Always hash passwords before storing. bcrypt is the recommended algorithm
# because it includes a salt and is deliberately slow (resistant to brute-force).

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# =============================================================================
# TOKEN CREATION
# =============================================================================

def create_token(data: dict, expires_delta: timedelta) -> str:
    """Create a JWT with claims and expiration.

    Standard claims (RFC 7519):
    - sub: Subject (user identifier)
    - exp: Expiration time
    - iat: Issued at
    - type: Custom claim to distinguish access vs refresh tokens
    """
    now = datetime.now(timezone.utc)
    payload = {
        **data,
        "iat": now,
        "exp": now + expires_delta,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_access_token(user_id: str, role: str) -> str:
    return create_token(
        {"sub": user_id, "role": role, "type": "access"},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )


def create_refresh_token(user_id: str) -> str:
    return create_token(
        {"sub": user_id, "type": "refresh"},
        timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )


def decode_token(token: str) -> dict:
    """Decode and validate a JWT. Raises HTTPException on failure."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# =============================================================================
# SCHEMAS
# =============================================================================

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    email: str = Field(..., examples=["alice@example.com"])
    role: str = Field("user", pattern="^(user|admin)$")


class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    role: str
    created_at: str


# =============================================================================
# IN-MEMORY USER STORE
# =============================================================================

users_db: dict[str, dict] = {}
next_user_id = 1


# =============================================================================
# DEPENDENCIES — FastAPI's dependency injection for auth
# =============================================================================
# Dependencies are the recommended way to handle cross-cutting concerns
# like authentication. They compose cleanly and are easy to test.

security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Extract and validate the current user from the Authorization header.

    Usage in route: def endpoint(user: dict = Depends(get_current_user))
    """
    payload = decode_token(credentials.credentials)

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type — use an access token",
        )

    user_id = payload.get("sub")
    user = users_db.get(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Role-based access control — only admins can access this endpoint.

    Chain dependencies: require_admin depends on get_current_user, so the
    user is already authenticated before we check the role.
    """
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


# =============================================================================
# APPLICATION
# =============================================================================

app = FastAPI(title="JWT Auth API", version="1.0.0")


@app.post(
    "/api/v1/auth/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Auth"],
)
def register(body: UserRegister):
    """Register a new user account.

    Passwords are hashed with bcrypt before storage. Never store plain-text
    passwords — if your database is compromised, hashed passwords remain safe.
    """
    global next_user_id

    # Check for duplicate username
    for u in users_db.values():
        if u["username"] == body.username:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Username '{body.username}' already exists",
            )

    user_id = str(next_user_id)
    next_user_id += 1

    user = {
        "id": user_id,
        "username": body.username,
        "email": body.email,
        "password_hash": hash_password(body.password),
        "role": body.role,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    users_db[user_id] = user

    return UserResponse(
        id=user["id"],
        username=user["username"],
        email=user["email"],
        role=user["role"],
        created_at=user["created_at"],
    )


@app.post("/api/v1/auth/login", response_model=TokenResponse, tags=["Auth"])
def login(body: UserLogin):
    """Authenticate and receive access + refresh tokens.

    The access token is short-lived (30 min) for security. The refresh token
    is longer-lived (7 days) and can only be used to get new access tokens.
    """
    # Find user by username
    user = None
    for u in users_db.values():
        if u["username"] == body.username:
            user = u
            break

    if not user or not verify_password(body.password, user["password_hash"]):
        # Use the same error message for both "user not found" and "wrong password"
        # to prevent username enumeration attacks.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return TokenResponse(
        access_token=create_access_token(user["id"], user["role"]),
        refresh_token=create_refresh_token(user["id"]),
    )


@app.post("/api/v1/auth/refresh", response_model=TokenResponse, tags=["Auth"])
def refresh(body: RefreshRequest):
    """Exchange a refresh token for a new access + refresh token pair.

    This enables seamless session extension without re-entering credentials.
    The old refresh token should be invalidated (not implemented here for
    simplicity — in production, use a token blacklist or rotation).
    """
    payload = decode_token(body.refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid token type — provide a refresh token",
        )

    user_id = payload["sub"]
    user = users_db.get(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return TokenResponse(
        access_token=create_access_token(user["id"], user["role"]),
        refresh_token=create_refresh_token(user["id"]),
    )


# =============================================================================
# PROTECTED ENDPOINTS
# =============================================================================

@app.get("/api/v1/users/me", response_model=UserResponse, tags=["Users"])
def get_current_user_profile(user: dict = Depends(get_current_user)):
    """Get the authenticated user's profile.

    The `Depends(get_current_user)` dependency handles token extraction,
    validation, and user lookup. If any step fails, an appropriate HTTP
    error is returned before this function body executes.
    """
    return UserResponse(
        id=user["id"],
        username=user["username"],
        email=user["email"],
        role=user["role"],
        created_at=user["created_at"],
    )


@app.get("/api/v1/admin/users", response_model=list[UserResponse], tags=["Admin"])
def list_all_users(admin: dict = Depends(require_admin)):
    """List all users — admin only.

    Demonstrates role-based access control. The `require_admin` dependency
    first authenticates (via `get_current_user`), then checks the role.
    Regular users receive 403 Forbidden.
    """
    return [
        UserResponse(
            id=u["id"],
            username=u["username"],
            email=u["email"],
            role=u["role"],
            created_at=u["created_at"],
        )
        for u in users_db.values()
    ]


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("03_authentication_jwt:app", host="127.0.0.1", port=8000, reload=True)
