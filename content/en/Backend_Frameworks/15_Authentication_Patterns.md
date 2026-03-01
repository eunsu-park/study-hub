# 15. Authentication Patterns

**Previous**: [API Design Patterns](./14_API_Design_Patterns.md) | **Next**: [Production Deployment](./16_Production_Deployment.md)

**Difficulty**: ⭐⭐⭐

## Learning Objectives

- Explain the difference between session-based authentication and token-based authentication, including trade-offs
- Implement JWT authentication with access tokens and refresh tokens in a backend application
- Compare OAuth2 authorization flows and select the appropriate flow for a given client type
- Apply secure password hashing using bcrypt or argon2 with proper configuration
- Describe multi-factor authentication concepts and their integration points in a backend system

## Table of Contents

1. [Session-Based Authentication](#1-session-based-authentication)
2. [JWT Authentication](#2-jwt-authentication)
3. [API Keys](#3-api-keys)
4. [OAuth2 Flows](#4-oauth2-flows)
5. [Comparison: Sessions vs. JWT vs. API Keys](#5-comparison-sessions-vs-jwt-vs-api-keys)
6. [Password Hashing](#6-password-hashing)
7. [Multi-Factor Authentication](#7-multi-factor-authentication)
8. [Practice Problems](#8-practice-problems)

---

## 1. Session-Based Authentication

Session-based authentication stores user state on the server. After login, the server creates a session record and sends the client a session ID via a cookie. Every subsequent request includes this cookie, and the server looks up the session to identify the user.

### How It Works

```
Client                          Server
  |                               |
  |-- POST /login (credentials) ->|
  |                               |-- Validate credentials
  |                               |-- Create session in store
  |<- Set-Cookie: session_id=abc -|
  |                               |
  |-- GET /profile               ->|
  |   Cookie: session_id=abc       |
  |                               |-- Look up session "abc"
  |                               |-- Return user data
  |<- 200 OK { user data }       -|
```

### FastAPI with Session Store (Redis)

```python
import uuid
import redis
import json
from fastapi import FastAPI, Request, Response, HTTPException

app = FastAPI()
session_store = redis.Redis(host="localhost", port=6379, db=0)

SESSION_TTL = 3600  # 1 hour

@app.post("/login")
async def login(response: Response, credentials: LoginRequest):
    user = await authenticate(credentials.username, credentials.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create session
    session_id = str(uuid.uuid4())
    session_data = {"user_id": user.id, "username": user.username}
    session_store.setex(
        f"session:{session_id}",
        SESSION_TTL,
        json.dumps(session_data),
    )

    # Set cookie — HttpOnly prevents JavaScript access
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,     # Not accessible via document.cookie
        secure=True,       # HTTPS only
        samesite="lax",    # CSRF protection
        max_age=SESSION_TTL,
    )
    return {"message": "Logged in"}

@app.get("/profile")
async def get_profile(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    session_data = session_store.get(f"session:{session_id}")
    if not session_data:
        raise HTTPException(status_code=401, detail="Session expired")

    session = json.loads(session_data)
    user = await get_user_by_id(session["user_id"])
    return {"id": user.id, "username": user.username, "email": user.email}

@app.post("/logout")
async def logout(request: Request, response: Response):
    session_id = request.cookies.get("session_id")
    if session_id:
        session_store.delete(f"session:{session_id}")
    response.delete_cookie("session_id")
    return {"message": "Logged out"}
```

### Security Considerations

- **HttpOnly cookies**: Prevent XSS attacks from stealing session IDs
- **Secure flag**: Ensures cookies are only sent over HTTPS
- **SameSite attribute**: Mitigates CSRF attacks (`Lax` or `Strict`)
- **Session expiration**: Set reasonable TTLs and implement idle timeout
- **Session fixation**: Regenerate session ID after login

---

## 2. JWT Authentication

JSON Web Tokens (JWT) are self-contained tokens that carry user claims. Unlike sessions, the server does not need to store state --- the token itself contains all necessary information, verified through a cryptographic signature.

### JWT Structure

A JWT consists of three base64url-encoded parts separated by dots:

```
eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiI1In0.signature
|---- Header ----|.|---- Payload ----|.|-- Sig --|
```

**Header**: Algorithm and token type
```json
{"alg": "HS256", "typ": "JWT"}
```

**Payload**: Claims (user data and metadata)
```json
{
    "sub": "5",
    "username": "alice",
    "role": "admin",
    "iat": 1706140800,
    "exp": 1706144400
}
```

**Signature**: HMAC-SHA256(base64url(header) + "." + base64url(payload), secret)

### Access Token + Refresh Token Pattern

Short-lived access tokens minimize the damage window if a token is stolen. Refresh tokens allow obtaining new access tokens without re-authenticating.

```python
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

app = FastAPI()

SECRET_KEY = "your-secret-key-store-in-env"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def create_token(data: dict, expires_delta: timedelta) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_access_token(user_id: int, username: str) -> str:
    return create_token(
        {"sub": str(user_id), "username": username, "type": "access"},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )


def create_refresh_token(user_id: int) -> str:
    return create_token(
        {"sub": str(user_id), "type": "refresh"},
        timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Dependency that extracts and validates the current user from JWT."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        token_type = payload.get("type")
        if user_id is None or token_type != "access":
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = await get_user_by_id(int(user_id))
    if user is None:
        raise credentials_exception
    return user


@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "access_token": create_access_token(user.id, user.username),
        "refresh_token": create_refresh_token(user.id),
        "token_type": "bearer",
    }


@app.post("/token/refresh")
async def refresh(refresh_token: str):
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user_id = int(payload["sub"])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return {
        "access_token": create_access_token(user.id, user.username),
        "token_type": "bearer",
    }


@app.get("/profile")
async def get_profile(current_user=Depends(get_current_user)):
    return {"id": current_user.id, "username": current_user.username}
```

### JWT Security Best Practices

- **Short expiration**: Access tokens should expire in 5-30 minutes
- **Store refresh tokens securely**: HttpOnly cookies or encrypted storage, never localStorage
- **Validate all claims**: Check `exp`, `iss`, `aud`, and token type
- **Use asymmetric signing (RS256)** for distributed systems where multiple services verify tokens
- **Implement token revocation**: Maintain a blocklist for compromised tokens (e.g., in Redis)
- **Never store sensitive data** in the payload --- JWTs are encoded, not encrypted

---

## 3. API Keys

API keys are long, random strings that identify the calling application rather than a specific user. They are best suited for server-to-server communication and service identification.

### When to Use API Keys

- **Machine-to-machine** communication (backend services, CI/CD pipelines)
- **Public data APIs** where user identity is not needed (e.g., weather data)
- **Usage tracking and billing** per application or customer

### When NOT to Use API Keys

- **User authentication** --- API keys do not represent a user session
- **Browser-based clients** --- keys would be exposed in client-side code

### Implementation

```python
from fastapi import FastAPI, Security, HTTPException
from fastapi.security import APIKeyHeader

app = FastAPI()
api_key_header = APIKeyHeader(name="X-API-Key")

# In production, store keys in a database with associated metadata
API_KEYS = {
    "sk_live_abc123def456": {"client": "mobile-app", "tier": "pro"},
    "sk_live_xyz789ghi012": {"client": "partner-service", "tier": "enterprise"},
}

async def verify_api_key(api_key: str = Security(api_key_header)):
    key_data = API_KEYS.get(api_key)
    if not key_data:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key_data

@app.get("/data")
async def get_data(key_data: dict = Security(verify_api_key)):
    return {"client": key_data["client"], "data": "..."}
```

### Security Practices

- Prefix keys to indicate type: `sk_live_`, `sk_test_`, `pk_` (publishable)
- Hash keys in the database --- store only the hash, not the plaintext
- Support key rotation: allow multiple active keys per client
- Log key usage for audit trails
- Set expiration dates and require periodic rotation

---

## 4. OAuth2 Flows

OAuth2 is an **authorization** framework (not authentication) that enables third-party applications to access user resources without exposing credentials. Different flows suit different client types.

### Authorization Code Flow (with PKCE)

The most secure flow, used by web apps and mobile apps. PKCE (Proof Key for Code Exchange) prevents authorization code interception.

```
User          Client App       Auth Server       Resource Server
 |                |                 |                  |
 |-- Click Login ->|                |                  |
 |                |-- Redirect ---->|                  |
 |                |   (+ code_verifier hash)           |
 |<-------------- Auth Page --------|                  |
 |-- Login + Consent -------------->|                  |
 |                |<-- Auth Code ---|                  |
 |                |-- Exchange Code + code_verifier -->|
 |                |<-- Access Token + Refresh Token ---|
 |                |-- GET /resource (Bearer token) --->|
 |                |<-- Resource Data ------------------|
```

### Client Credentials Flow

For server-to-server communication where no user is involved.

```python
import httpx

async def get_service_token():
    """Exchange client credentials for an access token."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://auth.example.com/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": "my-service-id",
                "client_secret": "my-service-secret",
                "scope": "read:users",
            },
        )
        token_data = response.json()
        return token_data["access_token"]
```

### Flow Selection Guide

| Flow                       | Client Type              | User Involved | Security Level |
|----------------------------|--------------------------|---------------|----------------|
| Authorization Code + PKCE  | Web apps, mobile apps    | Yes           | High           |
| Client Credentials         | Server-to-server         | No            | High           |
| Device Code                | Smart TVs, CLI tools     | Yes           | Medium         |
| Implicit (deprecated)      | SPAs (legacy)            | Yes           | Low            |

> **Note**: The Implicit flow is deprecated in OAuth 2.1. Use Authorization Code + PKCE for all browser-based applications.

### FastAPI OAuth2 Integration (GitHub Example)

```python
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
import httpx
import secrets

app = FastAPI()

GITHUB_CLIENT_ID = "your-client-id"
GITHUB_CLIENT_SECRET = "your-client-secret"

# Store state parameters to prevent CSRF
pending_states: dict[str, bool] = {}

@app.get("/auth/github")
async def github_login():
    state = secrets.token_urlsafe(32)
    pending_states[state] = True
    return RedirectResponse(
        f"https://github.com/login/oauth/authorize"
        f"?client_id={GITHUB_CLIENT_ID}"
        f"&scope=user:email"
        f"&state={state}"
    )

@app.get("/auth/github/callback")
async def github_callback(code: str, state: str):
    # Verify state to prevent CSRF
    if state not in pending_states:
        raise HTTPException(status_code=400, detail="Invalid state")
    del pending_states[state]

    # Exchange code for access token
    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            "https://github.com/login/oauth/access_token",
            json={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code,
            },
            headers={"Accept": "application/json"},
        )
        access_token = token_response.json()["access_token"]

        # Fetch user info
        user_response = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        github_user = user_response.json()

    # Create or update local user, issue session/JWT
    user = await get_or_create_user(
        github_id=github_user["id"],
        username=github_user["login"],
        email=github_user.get("email"),
    )
    token = create_access_token(user.id, user.username)
    return {"access_token": token, "token_type": "bearer"}
```

---

## 5. Comparison: Sessions vs. JWT vs. API Keys

| Aspect              | Sessions              | JWT                      | API Keys              |
|---------------------|-----------------------|--------------------------|-----------------------|
| State               | Server-side (stateful)| Client-side (stateless)  | Server-side (stateful)|
| Storage             | Redis, DB, memory     | Cookie or Authorization header | Database       |
| Scalability         | Requires shared store | Scales horizontally      | Simple lookup         |
| Revocation          | Easy (delete session) | Hard (needs blocklist)   | Easy (delete key)     |
| Identifies          | User session          | User claims              | Application/client    |
| Best for            | Traditional web apps  | SPAs, mobile, microservices | Service-to-service |
| CSRF vulnerability  | Yes (cookie-based)    | No (if in Auth header)   | No                    |
| XSS vulnerability   | Low (HttpOnly cookie) | High (if in localStorage)| Low (server-side)     |

### Choosing the Right Strategy

- **Server-rendered web app** (Django templates, Jinja2): Sessions
- **SPA + API backend**: JWT (access + refresh tokens)
- **Mobile app**: JWT with secure storage (Keychain/Keystore)
- **Microservice-to-microservice**: JWT or mutual TLS
- **Third-party API access**: API keys
- **Third-party user login**: OAuth2

---

## 6. Password Hashing

Never store passwords in plaintext. Use a purpose-built password hashing function that is intentionally slow to resist brute-force attacks.

### bcrypt

The most widely deployed password hashing algorithm. Uses a configurable cost factor (work factor) to control computational expense.

```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """Hash a password with bcrypt (cost factor 12 by default)."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

# Usage
hashed = hash_password("my-secure-password")
# "$2b$12$LJ3m4ys3Lp..."

assert verify_password("my-secure-password", hashed) is True
assert verify_password("wrong-password", hashed) is False
```

### argon2

The winner of the Password Hashing Competition (2015). Recommended for new projects. Offers protection against both GPU and side-channel attacks.

```python
from passlib.context import CryptContext

# argon2id is the recommended variant
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# Same API as bcrypt
hashed = pwd_context.hash("my-secure-password")
is_valid = pwd_context.verify("my-secure-password", hashed)
```

### Comparison

| Algorithm | GPU Resistance | Memory-Hard | Side-Channel Resistant | Maturity |
|-----------|---------------|-------------|----------------------|----------|
| bcrypt    | Good          | No          | No                   | High     |
| scrypt    | Good          | Yes         | No                   | Medium   |
| argon2id  | Excellent     | Yes         | Yes                  | Growing  |

**Recommendation**: Use **argon2id** for new projects. Use **bcrypt** if argon2 is not available or if compatibility with existing systems is needed.

### Password Policy

```python
import re

def validate_password(password: str) -> list[str]:
    """Return a list of policy violations. Empty list means valid."""
    errors = []
    if len(password) < 12:
        errors.append("Password must be at least 12 characters")
    if not re.search(r"[A-Z]", password):
        errors.append("Password must contain an uppercase letter")
    if not re.search(r"[a-z]", password):
        errors.append("Password must contain a lowercase letter")
    if not re.search(r"\d", password):
        errors.append("Password must contain a digit")
    # Check against common password lists in production
    return errors
```

---

## 7. Multi-Factor Authentication

Multi-factor authentication (MFA) requires users to provide two or more independent verification factors. The factors are categorized as:

- **Something you know**: Password, PIN
- **Something you have**: Phone (TOTP app), hardware key (YubiKey)
- **Something you are**: Fingerprint, face recognition

### TOTP (Time-Based One-Time Password)

The most common MFA method. The server and client share a secret key. Both generate a 6-digit code based on the current time, which changes every 30 seconds.

```python
import pyotp
import qrcode
from io import BytesIO

def generate_totp_secret() -> str:
    """Generate a new TOTP secret for a user."""
    return pyotp.random_base32()

def get_totp_uri(secret: str, username: str, issuer: str = "MyApp") -> str:
    """Generate the otpauth:// URI for QR code generation."""
    totp = pyotp.TOTP(secret)
    return totp.provisioning_uri(name=username, issuer_name=issuer)

def verify_totp(secret: str, code: str) -> bool:
    """Verify a TOTP code. Allows +/- 1 time step for clock skew."""
    totp = pyotp.TOTP(secret)
    return totp.verify(code, valid_window=1)

# Enrollment flow
secret = generate_totp_secret()
uri = get_totp_uri(secret, "alice@example.com")
# User scans QR code with authenticator app (Google Authenticator, Authy, etc.)
# Store `secret` in user record (encrypted)

# Verification flow
user_code = "123456"  # from authenticator app
if verify_totp(stored_secret, user_code):
    print("MFA verified")
```

### Integration in Login Flow

```python
@app.post("/login")
async def login(credentials: LoginRequest):
    user = await authenticate(credentials.username, credentials.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if user.mfa_enabled:
        # Issue a short-lived, limited-scope token for MFA step
        mfa_token = create_token(
            {"sub": str(user.id), "type": "mfa_pending"},
            timedelta(minutes=5),
        )
        return {"mfa_required": True, "mfa_token": mfa_token}

    # No MFA: issue full access token
    return {
        "access_token": create_access_token(user.id, user.username),
        "token_type": "bearer",
    }

@app.post("/login/mfa")
async def verify_mfa(mfa_token: str, code: str):
    try:
        payload = jwt.decode(mfa_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "mfa_pending":
            raise HTTPException(status_code=401)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid MFA token")

    user = await get_user_by_id(int(payload["sub"]))
    if not verify_totp(user.totp_secret, code):
        raise HTTPException(status_code=401, detail="Invalid MFA code")

    return {
        "access_token": create_access_token(user.id, user.username),
        "refresh_token": create_refresh_token(user.id),
        "token_type": "bearer",
    }
```

### Backup Codes

Always provide backup codes during MFA enrollment in case the user loses access to their authenticator device. Generate 8-10 single-use codes, hash them, and store in the database.

---

## 8. Practice Problems

### Problem 1: Session Security Audit

You inherit a session-based authentication system with the following cookie configuration:
```python
response.set_cookie(key="session_id", value=session_id, max_age=86400)
```
Identify all the security problems with this configuration and rewrite it with proper security settings. Explain why each setting matters.

### Problem 2: JWT Refresh Token Rotation

Implement a refresh token rotation system where each time a refresh token is used, a new refresh token is issued and the old one is invalidated. If a previously-used refresh token is submitted (indicating potential theft), revoke the entire token family. Use Redis for the token family tracking.

### Problem 3: OAuth2 PKCE Implementation

Implement the Authorization Code flow with PKCE for a mobile app. Write:
1. The code verifier and code challenge generation
2. The authorization redirect URL construction
3. The token exchange endpoint that validates the code verifier
4. Explain why PKCE is necessary even when using HTTPS

### Problem 4: Password Migration

Your application currently stores passwords with MD5 (a critical vulnerability). Design and implement a migration strategy to argon2id that:
- Does not require users to reset their passwords immediately
- Transparently rehashes passwords on next successful login
- Tracks migration progress
- Forces reset for accounts inactive longer than 90 days

### Problem 5: Multi-Factor Authentication System

Design a complete MFA enrollment and verification flow for a FastAPI application. Include:
- TOTP secret generation and QR code delivery
- Backup code generation (10 codes, single-use)
- Login flow with MFA step
- MFA recovery when the authenticator device is lost
- Rate limiting on MFA code attempts (max 5 attempts per token)

---

## References

- [RFC 7519: JSON Web Token (JWT)](https://tools.ietf.org/html/rfc7519)
- [RFC 6749: OAuth 2.0 Authorization Framework](https://tools.ietf.org/html/rfc6749)
- [RFC 7636: PKCE for OAuth](https://tools.ietf.org/html/rfc7636)
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [OWASP Session Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html)
- [Password Hashing Competition](https://www.password-hashing.net/)
- [python-jose Documentation](https://python-jose.readthedocs.io/)
- [passlib Documentation](https://passlib.readthedocs.io/)

---

**Previous**: [API Design Patterns](./14_API_Design_Patterns.md) | **Next**: [Production Deployment](./16_Production_Deployment.md)
