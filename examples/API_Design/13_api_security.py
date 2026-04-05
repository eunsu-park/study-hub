#!/usr/bin/env python3
"""Example: API Security

Demonstrates essential API security patterns with FastAPI:
- CORS configuration
- Content Security Policy (CSP) headers
- Input validation and sanitization
- SQL injection prevention
- Rate-based brute-force protection
- OWASP API Top-10 mitigations

Related lesson: 13_API_Security.md

Run:
    pip install "fastapi[standard]"
    uvicorn 13_api_security:app --reload --port 8000
"""

import html
import re
import time
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# =============================================================================
# APPLICATION
# =============================================================================

app = FastAPI(title="Secure API Demo", version="1.0.0")


# =============================================================================
# CORS — Cross-Origin Resource Sharing
# =============================================================================
# CORS controls which origins (domains) can call your API from a browser.
# Without it, browsers block cross-origin requests by default.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://myapp.example.com"],  # Never use ["*"] in production
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=True,
    max_age=3600,  # Cache preflight for 1 hour
)


# =============================================================================
# SECURITY HEADERS MIDDLEWARE — Defense-in-depth
# =============================================================================

@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)

    # Content Security Policy — prevents XSS by restricting script sources
    response.headers["Content-Security-Policy"] = "default-src 'self'"

    # Prevent MIME-type sniffing attacks
    response.headers["X-Content-Type-Options"] = "nosniff"

    # Clickjacking protection
    response.headers["X-Frame-Options"] = "DENY"

    # Enforce HTTPS
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    # Hide server implementation details
    response.headers["X-Powered-By"] = ""

    return response


# =============================================================================
# BRUTE-FORCE PROTECTION — IP-based rate limiting
# =============================================================================
# Track failed login attempts per IP. Lock out after threshold.

_login_attempts: dict[str, list[float]] = defaultdict(list)
MAX_ATTEMPTS = 5
WINDOW_SECONDS = 300  # 5 minutes


def check_brute_force(ip: str):
    """Reject if too many failed attempts in the time window."""
    now = time.monotonic()
    attempts = _login_attempts[ip]
    # Prune old entries
    _login_attempts[ip] = [t for t in attempts if now - t < WINDOW_SECONDS]
    if len(_login_attempts[ip]) >= MAX_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail="Too many failed attempts. Try again later.",
        )


def record_failed_attempt(ip: str):
    _login_attempts[ip].append(time.monotonic())


# =============================================================================
# INPUT VALIDATION — Pydantic + custom validators
# =============================================================================
# OWASP API3:2023 — Broken Object Property Level Authorization
# Validate and sanitize every input field explicitly.

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=30, pattern=r"^[a-zA-Z0-9_]+$")
    email: str = Field(..., max_length=254)
    bio: Optional[str] = Field(None, max_length=500)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        # Basic email format check (use email-validator package in production)
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError("Invalid email format")
        return v.lower()

    @field_validator("bio")
    @classmethod
    def sanitize_bio(cls, v: Optional[str]) -> Optional[str]:
        """Escape HTML to prevent stored XSS."""
        if v:
            return html.escape(v)
        return v


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=8, max_length=128)


# =============================================================================
# SQL INJECTION PREVENTION — Parameterized queries only
# =============================================================================
# Never use f-strings or string concatenation with user input in SQL.

def safe_query_example():
    """Demonstrate parameterized queries (pseudocode)."""
    username = "user_input"

    # BAD — vulnerable to SQL injection:
    # cursor.execute(f"SELECT * FROM users WHERE name = '{username}'")

    # GOOD — parameterized query (driver escapes automatically):
    # cursor.execute("SELECT * FROM users WHERE name = %s", (username,))

    # GOOD — ORM (SQLAlchemy):
    # session.query(User).filter(User.name == username).first()
    pass


# =============================================================================
# ROUTES
# =============================================================================

@app.post("/api/v1/users", status_code=201, tags=["Users"])
def create_user(body: UserCreate):
    """Create a user with validated and sanitized input.

    OWASP mitigations applied:
    - API3: strict field validation prevents mass assignment
    - API8: input sanitization prevents stored XSS
    """
    return {
        "id": "u-123",
        "username": body.username,
        "email": body.email,
        "bio": body.bio,
    }


@app.post("/api/v1/login", tags=["Auth"])
def login(body: LoginRequest, request: Request):
    """Login endpoint with brute-force protection.

    OWASP API2:2023 — Broken Authentication: rate-limit login attempts.
    """
    client_ip = request.client.host if request.client else "unknown"
    check_brute_force(client_ip)

    # Simulated auth check
    if body.username == "admin" and body.password == "correct_password":
        _login_attempts.pop(client_ip, None)  # Reset on success
        return {"token": "eyJ...simulated"}

    record_failed_attempt(client_ip)
    raise HTTPException(status_code=401, detail="Invalid credentials")


@app.get("/api/v1/users/{user_id}", tags=["Users"])
def get_user(user_id: str):
    """Demonstrates BOLA prevention (OWASP API1:2023).

    In production, verify the authenticated user has access to this user_id.
    Never trust path parameters as authorization proof.
    """
    # Authorization check would go here:
    # if current_user.id != user_id and not current_user.is_admin:
    #     raise HTTPException(403, "Forbidden")
    return {"id": user_id, "username": "demo_user"}


@app.get("/api/v1/search", tags=["Search"])
def search(
    q: str = Query(..., min_length=1, max_length=200, description="Search query"),
):
    """Search with validated query parameter to prevent injection."""
    sanitized = html.escape(q)
    return {"query": sanitized, "results": []}


# =============================================================================
# OWASP API TOP-10 QUICK REFERENCE
# =============================================================================

OWASP_REFERENCE = """
OWASP API Security Top 10 (2023)
=================================
API1  Broken Object Level Authorization (BOLA)
API2  Broken Authentication
API3  Broken Object Property Level Authorization
API4  Unrestricted Resource Consumption
API5  Broken Function Level Authorization
API6  Unrestricted Access to Sensitive Business Flows
API7  Server Side Request Forgery (SSRF)
API8  Security Misconfiguration
API9  Improper Inventory Management
API10 Unsafe Consumption of APIs
"""


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print(OWASP_REFERENCE)
    uvicorn.run("13_api_security:app", host="127.0.0.1", port=8000, reload=True)
