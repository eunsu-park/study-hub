"""
Authentication Patterns — JWT, OAuth2, Session Management
Demonstrates: JWT creation/verification, refresh tokens, OAuth2 password flow,
              and session-based auth with secure cookies.

Run: pip install fastapi uvicorn python-jose[cryptography] passlib[bcrypt]
     uvicorn 15_authentication_patterns:app --reload
"""

from fastapi import FastAPI, Depends, HTTPException, status, Response, Cookie
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import jwt, JWTError
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from typing import Optional
import secrets

app = FastAPI(title="Auth Patterns", version="1.0.0")

# --- Config ---
SECRET_KEY = "change-me-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE = timedelta(minutes=15)
REFRESH_TOKEN_EXPIRE = timedelta(days=7)

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# --- In-memory stores ---
USERS_DB = {
    "alice": {"username": "alice", "hashed_pw": pwd_ctx.hash("secret123"), "role": "admin"},
    "bob": {"username": "bob", "hashed_pw": pwd_ctx.hash("pass456"), "role": "user"},
}
REFRESH_TOKENS: dict[str, str] = {}  # token -> username
SESSIONS: dict[str, dict] = {}  # session_id -> user info


# --- 1. JWT Access + Refresh Tokens ---

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


def create_access_token(sub: str, role: str) -> str:
    payload = {
        "sub": sub,
        "role": role,
        "exp": datetime.now(timezone.utc) + ACCESS_TOKEN_EXPIRE,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(username: str) -> str:
    token = secrets.token_urlsafe(32)
    REFRESH_TOKENS[token] = username
    return token


@app.post("/auth/token", response_model=TokenResponse)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 password flow: exchange credentials for JWT pair."""
    user = USERS_DB.get(form.username)
    if not user or not pwd_ctx.verify(form.password, user["hashed_pw"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(
        access_token=create_access_token(user["username"], user["role"]),
        refresh_token=create_refresh_token(user["username"]),
    )


@app.post("/auth/refresh", response_model=TokenResponse)
async def refresh(refresh_token: str):
    """Rotate refresh token and issue new access token."""
    username = REFRESH_TOKENS.pop(refresh_token, None)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    user = USERS_DB[username]
    return TokenResponse(
        access_token=create_access_token(username, user["role"]),
        refresh_token=create_refresh_token(username),
    )


# --- 2. Token Verification Dependency ---

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None or username not in USERS_DB:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"username": username, "role": payload.get("role")}
    except JWTError:
        raise HTTPException(status_code=401, detail="Token expired or invalid")


def require_role(role: str):
    """Factory for role-based authorization dependency."""
    async def checker(user: dict = Depends(get_current_user)):
        if user["role"] != role:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return checker


@app.get("/users/me")
async def read_me(user: dict = Depends(get_current_user)):
    return user


@app.get("/admin/dashboard")
async def admin_dashboard(user: dict = Depends(require_role("admin"))):
    return {"message": f"Welcome admin {user['username']}"}


# --- 3. Session-Based Auth (Cookie) ---

@app.post("/session/login")
async def session_login(form: OAuth2PasswordRequestForm = Depends(), response: Response = None):
    user = USERS_DB.get(form.username)
    if not user or not pwd_ctx.verify(form.password, user["hashed_pw"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    session_id = secrets.token_urlsafe(32)
    SESSIONS[session_id] = {"username": user["username"], "role": user["role"]}
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=3600,
    )
    return {"message": "Logged in"}


@app.get("/session/me")
async def session_me(session_id: Optional[str] = Cookie(None)):
    if not session_id or session_id not in SESSIONS:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return SESSIONS[session_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
