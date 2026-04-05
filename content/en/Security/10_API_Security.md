# API Security

**Previous**: [09_Web_Security_Headers.md](./09_Web_Security_Headers.md) | **Next**: [11_Secrets_Management.md](./11_Secrets_Management.md)

---

APIs are the backbone of modern software systems. They connect microservices, power mobile applications, and enable third-party integrations. Because APIs expose application logic and data directly, they are a prime target for attackers. A single misconfigured endpoint can expose millions of records. This lesson covers the essential security practices for building, deploying, and maintaining secure APIs — from authentication and rate limiting to input validation and CORS configuration.

## Learning Objectives

- Implement robust API authentication using API keys, OAuth 2.0, and JWT
- Design and deploy rate limiting strategies to prevent abuse
- Validate and sanitize all API inputs to prevent injection attacks
- Configure CORS correctly to control cross-origin access
- Secure GraphQL endpoints against common attack patterns
- Define security schemes in OpenAPI/Swagger specifications
- Apply API gateway security patterns for production environments

---

## 1. API Threat Landscape

### 1.1 OWASP API Security Top 10 (2023)

```
┌─────────────────────────────────────────────────────────────────────┐
│                 OWASP API Security Top 10 (2023)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  API1  Broken Object Level Authorization (BOLA)                      │
│        Accessing resources belonging to other users                  │
│        GET /api/users/OTHER_USER_ID/orders                           │
│                                                                      │
│  API2  Broken Authentication                                         │
│        Weak authentication mechanisms, token leakage                 │
│                                                                      │
│  API3  Broken Object Property Level Authorization                    │
│        Mass assignment, exposing sensitive properties                 │
│                                                                      │
│  API4  Unrestricted Resource Consumption                             │
│        No rate limiting, large payloads, expensive queries           │
│                                                                      │
│  API5  Broken Function Level Authorization                           │
│        Accessing admin functions as regular user                     │
│        POST /api/admin/users/delete                                  │
│                                                                      │
│  API6  Unrestricted Access to Sensitive Business Flows               │
│        Automated abuse of business features                          │
│                                                                      │
│  API7  Server-Side Request Forgery (SSRF)                            │
│        Fetching URLs from user input without validation              │
│                                                                      │
│  API8  Security Misconfiguration                                     │
│        Missing headers, verbose errors, default creds                │
│                                                                      │
│  API9  Improper Inventory Management                                 │
│        Unmanaged API versions, shadow APIs                           │
│                                                                      │
│  API10 Unsafe Consumption of APIs                                    │
│        Trusting third-party API responses blindly                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 API Attack Surface

```
┌─────────────────────────────────────────────────────────────────────┐
│                    API Attack Surface                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │  Client  │───▶│ Network  │───▶│ API      │───▶│ Database │      │
│  │          │    │          │    │ Server   │    │          │      │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│                                                                      │
│  Attack vectors at each layer:                                       │
│                                                                      │
│  Client:     Tampered requests, stolen tokens, replay attacks        │
│  Network:    Man-in-the-middle, eavesdropping, DNS hijacking         │
│  API Server: Injection, broken auth, BOLA, mass assignment           │
│  Database:   SQL injection, data exfiltration, unauthorized access   │
│                                                                      │
│  Cross-cutting concerns:                                             │
│  ├── Authentication    (Who are you?)                                │
│  ├── Authorization     (What can you access?)                        │
│  ├── Input validation  (Is this request safe?)                       │
│  ├── Rate limiting     (Are you abusing the API?)                    │
│  ├── Encryption        (Is data protected in transit?)               │
│  └── Logging/Monitoring (Can we detect attacks?)                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. API Authentication Patterns

### 2.1 API Keys

```python
"""
API Key authentication — simple but limited.
Suitable for server-to-server communication and public APIs
with usage tracking.
"""
import secrets
import hashlib
from flask import Flask, request, jsonify, abort
from functools import wraps
from datetime import datetime

app = Flask(__name__)

# ── API Key Generation ───────────────────────────────────────────
def generate_api_key() -> tuple[str, str]:
    """Generate an API key and its hash for storage."""
    # Generate a cryptographically secure key
    # Prefix helps identify the key type and version
    raw_key = f"sk_live_{secrets.token_urlsafe(32)}"

    # Store only the hash — never store the raw key
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    return raw_key, key_hash

# raw_key: sk_live_Abc123...  (sent to client once, never stored)
# key_hash: e3b0c4...         (stored in database)


# ── Simulated key storage (use a real database in production) ────
API_KEYS = {
    # hash -> metadata
    hashlib.sha256(b"sk_live_test_key_12345").hexdigest(): {
        "client_id": "client_001",
        "name": "Test Client",
        "rate_limit": 100,       # requests per minute
        "scopes": ["read", "write"],
        "created_at": "2025-01-01",
        "last_used": None,
    }
}


def require_api_key(f):
    """Decorator to require a valid API key."""
    @wraps(f)
    def decorated(*args, **kwargs):
        # Check multiple locations for the API key
        api_key = (
            request.headers.get('X-API-Key') or
            request.headers.get('Authorization', '').replace('Bearer ', '') or
            request.args.get('api_key')  # Less secure, avoid if possible
        )

        if not api_key:
            return jsonify({
                "error": "missing_api_key",
                "message": "API key is required. "
                           "Pass it in the X-API-Key header."
            }), 401

        # Hash the provided key and look it up
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_data = API_KEYS.get(key_hash)

        if not key_data:
            return jsonify({
                "error": "invalid_api_key",
                "message": "The provided API key is not valid."
            }), 401

        # Update last used timestamp
        key_data["last_used"] = datetime.utcnow().isoformat()

        # Store key metadata in request context
        request.api_client = key_data
        return f(*args, **kwargs)

    return decorated


@app.route('/api/data')
@require_api_key
def get_data():
    """Protected endpoint requiring API key."""
    client = request.api_client
    return jsonify({
        "client": client["name"],
        "data": [1, 2, 3]
    })


# ── API Key Security Best Practices ─────────────────────────────
"""
1. Transmission:
   - Always use HTTPS
   - Prefer headers over query parameters (query params appear in logs)
   - Use X-API-Key header or Authorization: Bearer <key>

2. Storage:
   - Hash keys before storing (SHA-256 minimum)
   - Never log full API keys
   - Show only last 4 characters in UI: sk_live_****5678

3. Rotation:
   - Support multiple active keys per client
   - Allow key rotation without downtime
   - Set expiration dates on keys

4. Scoping:
   - Assign scopes/permissions to each key
   - Use separate keys for read vs write operations
   - Use separate keys for test vs production

5. Limitations:
   - API keys identify applications, not users
   - They lack built-in expiration (unlike tokens)
   - They cannot be easily scoped per-request
   - Use OAuth 2.0 for user-context authorization
"""
```

### 2.2 OAuth 2.0

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OAuth 2.0 Authorization Code Flow                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. User clicks "Login with Provider"                                │
│     Client ──────────────────────────▶ Auth Server                   │
│     GET /authorize?response_type=code                                │
│         &client_id=CLIENT_ID                                         │
│         &redirect_uri=CALLBACK_URL                                   │
│         &scope=read+write                                            │
│         &state=RANDOM_STATE                                          │
│                                                                      │
│  2. User authenticates and approves scopes                           │
│     Auth Server ─────────────────────▶ Client Callback               │
│     GET /callback?code=AUTH_CODE&state=RANDOM_STATE                  │
│                                                                      │
│  3. Client exchanges code for tokens                                 │
│     Client ──────────────────────────▶ Auth Server                   │
│     POST /token                                                      │
│         grant_type=authorization_code                                │
│         &code=AUTH_CODE                                              │
│         &client_id=CLIENT_ID                                         │
│         &client_secret=CLIENT_SECRET                                 │
│         &redirect_uri=CALLBACK_URL                                   │
│                                                                      │
│  4. Auth server returns tokens                                       │
│     Auth Server ─────────────────────▶ Client                        │
│     { "access_token": "...",                                         │
│       "refresh_token": "...",                                        │
│       "token_type": "Bearer",                                        │
│       "expires_in": 3600 }                                           │
│                                                                      │
│  5. Client uses access token                                         │
│     Client ──────────────────────────▶ Resource Server               │
│     Authorization: Bearer ACCESS_TOKEN                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
"""
OAuth 2.0 implementation with Flask (server-side).
"""
import requests
import secrets
from flask import Flask, redirect, request, session, jsonify
from urllib.parse import urlencode

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# OAuth 2.0 configuration (example: GitHub)
OAUTH_CONFIG = {
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "authorize_url": "https://github.com/login/oauth/authorize",
    "token_url": "https://github.com/login/oauth/access_token",
    "api_url": "https://api.github.com/user",
    "redirect_uri": "http://localhost:5000/callback",
    "scope": "read:user user:email",
}


@app.route('/login')
def login():
    """Initiate OAuth 2.0 Authorization Code flow."""
    # Generate state parameter to prevent CSRF
    state = secrets.token_urlsafe(32)
    session['oauth_state'] = state

    # Build authorization URL
    params = {
        "client_id": OAUTH_CONFIG["client_id"],
        "redirect_uri": OAUTH_CONFIG["redirect_uri"],
        "scope": OAUTH_CONFIG["scope"],
        "state": state,
        "response_type": "code",
    }

    auth_url = f"{OAUTH_CONFIG['authorize_url']}?{urlencode(params)}"
    return redirect(auth_url)


@app.route('/callback')
def callback():
    """Handle OAuth 2.0 callback."""
    # ── Verify state parameter ───────────────────────────────────
    state = request.args.get('state')
    if state != session.pop('oauth_state', None):
        return jsonify({"error": "Invalid state parameter"}), 400

    # ── Check for error response ─────────────────────────────────
    error = request.args.get('error')
    if error:
        return jsonify({
            "error": error,
            "description": request.args.get('error_description', '')
        }), 400

    # ── Exchange authorization code for tokens ───────────────────
    code = request.args.get('code')
    token_response = requests.post(
        OAUTH_CONFIG["token_url"],
        data={
            "client_id": OAUTH_CONFIG["client_id"],
            "client_secret": OAUTH_CONFIG["client_secret"],
            "code": code,
            "redirect_uri": OAUTH_CONFIG["redirect_uri"],
            "grant_type": "authorization_code",
        },
        headers={"Accept": "application/json"},
        timeout=10,
    )
    token_data = token_response.json()

    if "error" in token_data:
        return jsonify(token_data), 400

    access_token = token_data["access_token"]

    # ── Fetch user information ───────────────────────────────────
    user_response = requests.get(
        OAUTH_CONFIG["api_url"],
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        },
        timeout=10,
    )
    user_data = user_response.json()

    # Store in session (or create/update user in database)
    session['user'] = {
        "id": user_data["id"],
        "name": user_data.get("name", user_data["login"]),
        "email": user_data.get("email"),
    }

    return redirect('/dashboard')


# ── OAuth 2.0 Security Checklist ────────────────────────────────
"""
1. Always use the state parameter (prevents CSRF)
2. Validate redirect_uri exactly (no open redirect)
3. Use Authorization Code flow (not Implicit for server apps)
4. Store tokens securely (encrypted, server-side)
5. Use PKCE for public clients (SPAs, mobile apps)
6. Validate token scopes on every request
7. Use short-lived access tokens + refresh tokens
8. Revoke tokens on logout
"""
```

### 2.3 JWT (JSON Web Tokens)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    JWT Structure                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  eyJhbGci... . eyJzdWIi... . SflKxwRJ...                           │
│  ─────────     ─────────     ──────────                             │
│   Header        Payload       Signature                              │
│                                                                      │
│  Header (base64url):                                                 │
│  {                                                                   │
│    "alg": "RS256",                                                   │
│    "typ": "JWT",                                                     │
│    "kid": "key-id-123"                                               │
│  }                                                                   │
│                                                                      │
│  Payload (base64url):                                                │
│  {                                                                   │
│    "sub": "user_123",        // Subject (user ID)                    │
│    "iss": "api.example.com", // Issuer                               │
│    "aud": "app.example.com", // Audience                             │
│    "exp": 1700000000,        // Expiration time                      │
│    "iat": 1699996400,        // Issued at                            │
│    "nbf": 1699996400,        // Not before                           │
│    "jti": "unique-token-id", // JWT ID (for revocation)              │
│    "scope": "read write",    // Custom claims                        │
│    "role": "admin"                                                   │
│  }                                                                   │
│                                                                      │
│  Signature:                                                          │
│  RSASHA256(base64url(header) + "." + base64url(payload), key)       │
│                                                                      │
│  IMPORTANT: Payload is NOT encrypted — it is only base64url         │
│  encoded. Anyone can read it. Never store secrets in JWT.            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
"""
JWT authentication with PyJWT — secure implementation.
"""
import jwt
import time
import uuid
from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify
from functools import wraps

app = Flask(__name__)

# ── Key configuration ────────────────────────────────────────────
# Use RS256 (asymmetric) for production
# Private key signs tokens; public key verifies them
# This allows microservices to verify without the private key

# For this example, we use HS256 (symmetric) for simplicity
JWT_SECRET = "your-256-bit-secret-change-this"  # Use env var in production
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=15)
JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)


# ── Token blacklist (use Redis in production) ────────────────────
revoked_tokens = set()


def create_access_token(user_id: str, role: str = "user",
                        scopes: list[str] = None) -> str:
    """Create a short-lived access token."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "iss": "api.example.com",
        "aud": "app.example.com",
        "iat": now,
        "exp": now + JWT_ACCESS_TOKEN_EXPIRES,
        "nbf": now,
        "jti": str(uuid.uuid4()),        # Unique ID for revocation
        "type": "access",
        "role": role,
        "scopes": scopes or ["read"],
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    """Create a long-lived refresh token."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "iss": "api.example.com",
        "iat": now,
        "exp": now + JWT_REFRESH_TOKEN_EXPIRES,
        "jti": str(uuid.uuid4()),
        "type": "refresh",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            issuer="api.example.com",
            audience="app.example.com",
            options={
                "require": ["exp", "iat", "sub", "iss", "aud", "jti"],
                "verify_exp": True,
                "verify_iss": True,
                "verify_aud": True,
                "verify_nbf": True,
            }
        )

        # Check if token has been revoked
        if payload["jti"] in revoked_tokens:
            raise jwt.InvalidTokenError("Token has been revoked")

        return payload

    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidAudienceError:
        raise ValueError("Invalid token audience")
    except jwt.InvalidIssuerError:
        raise ValueError("Invalid token issuer")
    except jwt.InvalidTokenError as e:
        raise ValueError(f"Invalid token: {e}")


def require_auth(scopes: list[str] = None):
    """Decorator to require JWT authentication with optional scope check."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            # Extract token from Authorization header
            auth_header = request.headers.get('Authorization', '')
            if not auth_header.startswith('Bearer '):
                return jsonify({
                    "error": "missing_token",
                    "message": "Authorization header with Bearer token required"
                }), 401

            token = auth_header.split(' ', 1)[1]

            try:
                payload = decode_token(token)
            except ValueError as e:
                return jsonify({
                    "error": "invalid_token",
                    "message": str(e)
                }), 401

            # Verify token type
            if payload.get("type") != "access":
                return jsonify({
                    "error": "wrong_token_type",
                    "message": "Access token required"
                }), 401

            # Check scopes
            if scopes:
                token_scopes = set(payload.get("scopes", []))
                required_scopes = set(scopes)
                if not required_scopes.issubset(token_scopes):
                    missing = required_scopes - token_scopes
                    return jsonify({
                        "error": "insufficient_scope",
                        "message": f"Missing scopes: {', '.join(missing)}"
                    }), 403

            request.user = payload
            return f(*args, **kwargs)
        return decorated
    return decorator


# ── Routes ───────────────────────────────────────────────────────
@app.route('/api/login', methods=['POST'])
def login():
    """Authenticate and return JWT tokens."""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    # Validate credentials (use bcrypt/argon2 hash comparison)
    # This is simplified — see Authentication lesson for details
    user = authenticate_user(username, password)
    if not user:
        # Use consistent timing to prevent user enumeration
        return jsonify({
            "error": "invalid_credentials",
            "message": "Invalid username or password"
        }), 401

    # Generate token pair
    access_token = create_access_token(
        user_id=user["id"],
        role=user["role"],
        scopes=user["scopes"]
    )
    refresh_token = create_refresh_token(user_id=user["id"])

    return jsonify({
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "expires_in": int(JWT_ACCESS_TOKEN_EXPIRES.total_seconds()),
    })


@app.route('/api/refresh', methods=['POST'])
def refresh():
    """Exchange a refresh token for a new access token."""
    data = request.get_json()
    refresh_token = data.get('refresh_token')

    if not refresh_token:
        return jsonify({"error": "missing_refresh_token"}), 400

    try:
        # Decode without audience check (refresh tokens may differ)
        payload = jwt.decode(
            refresh_token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            issuer="api.example.com",
            options={"require": ["exp", "sub", "jti", "iss"]}
        )

        if payload.get("type") != "refresh":
            raise ValueError("Not a refresh token")

        if payload["jti"] in revoked_tokens:
            raise ValueError("Refresh token has been revoked")

    except (jwt.InvalidTokenError, ValueError) as e:
        return jsonify({"error": "invalid_refresh_token",
                        "message": str(e)}), 401

    # Revoke old refresh token (rotation)
    revoked_tokens.add(payload["jti"])

    # Issue new token pair
    user_id = payload["sub"]
    # Look up current user roles/scopes from database
    user = get_user_by_id(user_id)

    new_access = create_access_token(
        user_id=user_id,
        role=user["role"],
        scopes=user["scopes"]
    )
    new_refresh = create_refresh_token(user_id=user_id)

    return jsonify({
        "access_token": new_access,
        "refresh_token": new_refresh,
        "token_type": "Bearer",
        "expires_in": int(JWT_ACCESS_TOKEN_EXPIRES.total_seconds()),
    })


@app.route('/api/logout', methods=['POST'])
@require_auth()
def logout():
    """Revoke the current access token."""
    revoked_tokens.add(request.user["jti"])
    return jsonify({"message": "Successfully logged out"}), 200


@app.route('/api/protected')
@require_auth(scopes=["read"])
def protected_resource():
    """Protected endpoint requiring 'read' scope."""
    return jsonify({
        "user_id": request.user["sub"],
        "message": "You have access to this protected resource"
    })


# Placeholder functions for completeness
def authenticate_user(username, password):
    """Placeholder — implement with proper password hashing."""
    return None

def get_user_by_id(user_id):
    """Placeholder — implement with database lookup."""
    return {"id": user_id, "role": "user", "scopes": ["read"]}
```

### 2.4 JWT Security Best Practices

```python
"""
JWT security best practices and common pitfalls.
"""

# ── PITFALL 1: Using 'none' algorithm ───────────────────────────
# Attack: Change header to {"alg": "none"} and remove signature
# Defense: Always specify allowed algorithms explicitly
payload = jwt.decode(
    token, key,
    algorithms=["RS256"],  # NEVER include "none" or allow all
)

# ── PITFALL 2: Confusing HS256 and RS256 ────────────────────────
# Attack: If server uses RS256, attacker might:
#   1. Get the public key (it is public)
#   2. Sign a new token with HS256 using the public key as secret
#   3. Server treats public key as HMAC secret
# Defense: Strictly validate algorithm in header
payload = jwt.decode(
    token, rsa_public_key,
    algorithms=["RS256"],  # Only allow expected algorithm
)

# ── PITFALL 3: Missing expiration ───────────────────────────────
# Tokens without exp claim live forever
# Defense: Always set short expiration
payload = {
    "sub": "user_123",
    "exp": datetime.now(timezone.utc) + timedelta(minutes=15),
}

# ── PITFALL 4: Storing sensitive data in payload ─────────────────
# JWT payload is base64url-encoded, NOT encrypted
# Anyone can decode it without the key
import base64
header, payload_b64, signature = token.split('.')
decoded = base64.urlsafe_b64decode(payload_b64 + '==')
# Entire payload is now readable!

# NEVER include: passwords, SSNs, credit cards, PII in JWT
# Only include: user ID, role, scopes, expiration

# ── PITFALL 5: No token revocation mechanism ─────────────────────
# JWTs are self-contained — the server cannot invalidate them
# Solutions:
# 1. Short-lived access tokens (15 min) + refresh token rotation
# 2. Token blacklist (Redis with TTL = token exp)
# 3. Token versioning (store token version in DB, check on each request)
# 4. Change signing key (invalidates ALL tokens — nuclear option)

# ── RS256 Setup (Production Recommended) ────────────────────────
"""
# Generate RSA key pair
openssl genrsa -out private.pem 2048
openssl rsa -in private.pem -pubout -out public.pem

# Private key: Used by auth server to SIGN tokens
# Public key:  Used by all services to VERIFY tokens
# Share public key freely; guard private key
"""

from cryptography.hazmat.primitives import serialization

# Load keys
with open('private.pem', 'rb') as f:
    private_key = serialization.load_pem_private_key(f.read(), password=None)

with open('public.pem', 'rb') as f:
    public_key = serialization.load_pem_public_key(f.read())

# Sign with private key
token = jwt.encode(payload, private_key, algorithm="RS256")

# Verify with public key (any service can do this)
decoded = jwt.decode(token, public_key, algorithms=["RS256"])
```

---

## 3. Rate Limiting

### 3.1 Rate Limiting Algorithms

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Rate Limiting Algorithms                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Fixed Window                                                     │
│     ┌──────────┐┌──────────┐┌──────────┐                            │
│     │ Window 1 ││ Window 2 ││ Window 3 │                            │
│     │ ■■■■■    ││ ■■■      ││ ■■■■■■■  │                           │
│     │ 5/10     ││ 3/10     ││ 7/10     │  (limit: 10 per window)   │
│     └──────────┘└──────────┘└──────────┘                            │
│     Pro: Simple.  Con: Burst at window boundary (2x limit)          │
│                                                                      │
│  2. Sliding Window Log                                               │
│     Time: ─────────[====current window====]────────                  │
│     Track exact timestamp of each request                            │
│     Count requests within sliding window                             │
│     Pro: Accurate.  Con: Memory-intensive (stores all timestamps)    │
│                                                                      │
│  3. Sliding Window Counter                                           │
│     Combine fixed window counts with weighted overlap                │
│     weight = (window_size - elapsed) / window_size                   │
│     count = prev_count * weight + current_count                      │
│     Pro: Memory-efficient.  Con: Approximate                         │
│                                                                      │
│  4. Token Bucket                                                     │
│     ┌─────────┐                                                      │
│     │ ● ● ● ● │ ← Bucket (capacity: 10 tokens)                     │
│     │ ● ● ●   │                                                     │
│     └─────────┘                                                      │
│         ↑                                                            │
│     Refill: 1 token/second                                           │
│     Each request consumes 1 token                                    │
│     Pro: Allows bursts.  Con: More state to manage                   │
│                                                                      │
│  5. Leaky Bucket                                                     │
│     Requests enter a queue (bucket)                                  │
│     Processed at a fixed rate (leak rate)                            │
│     Overflow is rejected                                             │
│     Pro: Smooth output rate.  Con: Delays even when capacity exists  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Flask Rate Limiting with Flask-Limiter

```python
"""
Rate limiting implementation with Flask-Limiter.
pip install Flask-Limiter
"""
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# ── Basic setup ──────────────────────────────────────────────────
limiter = Limiter(
    app=app,
    key_func=get_remote_address,       # Rate limit by IP address
    default_limits=["200 per day", "50 per hour"],
    storage_uri="redis://localhost:6379",  # Use Redis for distributed
    # storage_uri="memory://",           # In-memory for development
    strategy="fixed-window-elastic-expiry",
)


# ── Global rate limit (applies to all routes) ───────────────────
# Already set via default_limits above

# ── Per-route rate limits ────────────────────────────────────────
@app.route('/api/search')
@limiter.limit("10 per minute")
def search():
    """Search endpoint — stricter limit to prevent scraping."""
    query = request.args.get('q', '')
    return jsonify({"query": query, "results": []})


@app.route('/api/login', methods=['POST'])
@limiter.limit("5 per minute")         # Prevent brute force
def login():
    """Login endpoint with aggressive rate limiting."""
    return jsonify({"message": "login"})


@app.route('/api/data')
@limiter.limit("100 per hour")
@limiter.limit("10 per minute")        # Multiple limits
def get_data():
    """Data endpoint with tiered rate limits."""
    return jsonify({"data": []})


# ── Dynamic rate limit based on API key tier ────────────────────
def get_rate_limit_by_tier():
    """Return rate limit string based on the client's API tier."""
    api_key = request.headers.get('X-API-Key', '')
    # Look up tier from database (simplified)
    tiers = {
        "free": "100 per hour",
        "pro": "1000 per hour",
        "enterprise": "10000 per hour",
    }
    tier = get_tier_for_key(api_key)
    return tiers.get(tier, "50 per hour")  # Default to free


@app.route('/api/premium')
@limiter.limit(get_rate_limit_by_tier)
def premium_endpoint():
    """Endpoint with tier-based rate limiting."""
    return jsonify({"data": "premium"})


# ── Rate limit headers ──────────────────────────────────────────
# Flask-Limiter automatically adds these headers:
# X-RateLimit-Limit: 100
# X-RateLimit-Remaining: 95
# X-RateLimit-Reset: 1699999999
# Retry-After: 60 (when limit exceeded)

# ── Custom error handler ────────────────────────────────────────
@app.errorhandler(429)
def ratelimit_handler(e):
    """Custom response when rate limit is exceeded."""
    return jsonify({
        "error": "rate_limit_exceeded",
        "message": "Too many requests. Please try again later.",
        "retry_after": e.description,
    }), 429


# Placeholder
def get_tier_for_key(api_key):
    return "free"
```

### 3.3 Custom Token Bucket Implementation

```python
"""
Token bucket rate limiter implementation from scratch.
"""
import time
import threading
from dataclasses import dataclass, field

@dataclass
class TokenBucket:
    """Token bucket rate limiter.

    Args:
        capacity: Maximum number of tokens in the bucket
        refill_rate: Tokens added per second
    """
    capacity: int
    refill_rate: float
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self):
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()

    def _refill(self):
        """Add tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if allowed."""
        with self.lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def wait_time(self) -> float:
        """Returns seconds until at least 1 token is available."""
        with self.lock:
            self._refill()
            if self.tokens >= 1:
                return 0.0
            return (1 - self.tokens) / self.refill_rate


class RateLimiterStore:
    """Manage rate limiters per client key."""

    def __init__(self, capacity: int = 100, refill_rate: float = 10.0):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def get_bucket(self, key: str) -> TokenBucket:
        """Get or create a token bucket for the given key."""
        if key not in self._buckets:
            with self._lock:
                if key not in self._buckets:
                    self._buckets[key] = TokenBucket(
                        capacity=self.capacity,
                        refill_rate=self.refill_rate
                    )
        return self._buckets[key]

    def is_allowed(self, key: str, tokens: int = 1) -> bool:
        """Check if a request is allowed for the given key."""
        bucket = self.get_bucket(key)
        return bucket.consume(tokens)


# ── Usage with Flask ─────────────────────────────────────────────
from flask import Flask, request, jsonify
from functools import wraps

app = Flask(__name__)
rate_limiter = RateLimiterStore(capacity=100, refill_rate=10.0)

def rate_limit(capacity=100, refill_rate=10.0):
    """Custom rate limit decorator."""
    store = RateLimiterStore(capacity=capacity, refill_rate=refill_rate)

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            # Use IP + endpoint as the rate limit key
            client_ip = request.remote_addr
            key = f"{client_ip}:{request.endpoint}"

            bucket = store.get_bucket(key)
            if not bucket.consume():
                wait = bucket.wait_time()
                return jsonify({
                    "error": "rate_limit_exceeded",
                    "retry_after": round(wait, 2)
                }), 429

            response = f(*args, **kwargs)
            return response
        return decorated
    return decorator


@app.route('/api/resource')
@rate_limit(capacity=10, refill_rate=1.0)  # 10 burst, 1/sec sustained
def get_resource():
    return jsonify({"data": "resource"})
```

---

## 4. Input Validation and Sanitization

### 4.1 Validation Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Input Validation Layers                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Request ─┬── Layer 1: Schema Validation                             │
│           │   (structure, types, required fields)                     │
│           │                                                          │
│           ├── Layer 2: Business Validation                            │
│           │   (ranges, formats, consistency)                         │
│           │                                                          │
│           ├── Layer 3: Sanitization                                   │
│           │   (trim whitespace, normalize, encode)                   │
│           │                                                          │
│           └── Layer 4: Parameterized Operations                      │
│               (SQL parameterization, template escaping)              │
│                                                                      │
│  Principle: Validate early, fail fast, never trust client data.      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Schema Validation with Marshmallow

```python
"""
API input validation using Marshmallow schemas.
pip install marshmallow
"""
from marshmallow import (
    Schema, fields, validate, validates, validates_schema,
    ValidationError, pre_load, RAISE
)
from flask import Flask, request, jsonify

app = Flask(__name__)


# ── Schema Definition ────────────────────────────────────────────
class UserCreateSchema(Schema):
    """Schema for creating a new user."""

    class Meta:
        # Raise error on unknown fields (prevents mass assignment)
        unknown = RAISE

    username = fields.String(
        required=True,
        validate=[
            validate.Length(min=3, max=30),
            validate.Regexp(
                r'^[a-zA-Z0-9_]+$',
                error="Username must contain only letters, numbers, underscores"
            ),
        ]
    )
    email = fields.Email(required=True)
    password = fields.String(
        required=True,
        load_only=True,  # Never include in serialized output
        validate=validate.Length(min=12, max=128),
    )
    age = fields.Integer(
        validate=validate.Range(min=13, max=150),
        load_default=None,
    )
    role = fields.String(
        validate=validate.OneOf(["user", "moderator"]),
        load_default="user",
        # Note: "admin" is not allowed via API — only via database
    )

    @validates('password')
    def validate_password_complexity(self, value):
        """Enforce password complexity requirements."""
        errors = []
        if not any(c.isupper() for c in value):
            errors.append("Must contain at least one uppercase letter")
        if not any(c.islower() for c in value):
            errors.append("Must contain at least one lowercase letter")
        if not any(c.isdigit() for c in value):
            errors.append("Must contain at least one digit")
        if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in value):
            errors.append("Must contain at least one special character")
        if errors:
            raise ValidationError(errors)

    @pre_load
    def normalize_input(self, data, **kwargs):
        """Normalize input data before validation."""
        if 'email' in data:
            data['email'] = data['email'].strip().lower()
        if 'username' in data:
            data['username'] = data['username'].strip()
        return data


class SearchQuerySchema(Schema):
    """Schema for search queries."""

    class Meta:
        unknown = RAISE

    q = fields.String(
        required=True,
        validate=validate.Length(min=1, max=200),
    )
    page = fields.Integer(
        validate=validate.Range(min=1, max=1000),
        load_default=1,
    )
    per_page = fields.Integer(
        validate=validate.Range(min=1, max=100),
        load_default=20,
    )
    sort = fields.String(
        validate=validate.OneOf(["relevance", "date", "name"]),
        load_default="relevance",
    )
    order = fields.String(
        validate=validate.OneOf(["asc", "desc"]),
        load_default="desc",
    )


# ── Validation decorator ────────────────────────────────────────
def validate_input(schema_class, location="json"):
    """Decorator to validate request input against a schema."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            schema = schema_class()

            if location == "json":
                data = request.get_json(silent=True)
                if data is None:
                    return jsonify({
                        "error": "invalid_request",
                        "message": "Request body must be valid JSON"
                    }), 400
            elif location == "args":
                data = request.args.to_dict()
            elif location == "form":
                data = request.form.to_dict()
            else:
                raise ValueError(f"Unknown location: {location}")

            try:
                validated = schema.load(data)
            except ValidationError as err:
                return jsonify({
                    "error": "validation_error",
                    "messages": err.messages,
                }), 422

            request.validated_data = validated
            return f(*args, **kwargs)
        return decorated
    return decorator


# ── Using the validation decorator ───────────────────────────────
@app.route('/api/users', methods=['POST'])
@validate_input(UserCreateSchema, location="json")
def create_user():
    """Create a new user with validated input."""
    data = request.validated_data
    # data is guaranteed to be valid at this point
    return jsonify({
        "message": "User created",
        "username": data["username"],
        "email": data["email"],
    }), 201


@app.route('/api/search')
@validate_input(SearchQuerySchema, location="args")
def search():
    """Search with validated query parameters."""
    data = request.validated_data
    return jsonify({
        "query": data["q"],
        "page": data["page"],
        "per_page": data["per_page"],
    })
```

### 4.3 Pydantic Validation (Alternative)

```python
"""
API validation using Pydantic v2.
pip install pydantic
"""
from pydantic import (
    BaseModel, Field, field_validator, model_validator,
    EmailStr, ConfigDict
)
from typing import Optional
import re

class UserCreate(BaseModel):
    """Pydantic model for user creation."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra='forbid',  # Reject unknown fields
    )

    username: str = Field(
        min_length=3,
        max_length=30,
        pattern=r'^[a-zA-Z0-9_]+$',
    )
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)
    age: Optional[int] = Field(default=None, ge=13, le=150)
    role: str = Field(default="user", pattern=r'^(user|moderator)$')

    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if not re.search(r'[A-Z]', v):
            raise ValueError('Must contain uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Must contain lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Must contain digit')
        if not re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]', v):
            raise ValueError('Must contain special character')
        return v

    @field_validator('email')
    @classmethod
    def normalize_email(cls, v):
        return v.lower()


# ── Usage in Flask ───────────────────────────────────────────────
from flask import Flask, request, jsonify
from pydantic import ValidationError as PydanticValidationError

app = Flask(__name__)

@app.route('/api/users', methods=['POST'])
def create_user():
    try:
        user = UserCreate(**request.get_json())
    except PydanticValidationError as e:
        return jsonify({
            "error": "validation_error",
            "details": e.errors(),
        }), 422

    return jsonify({"username": user.username}), 201
```

---

## 5. CORS (Cross-Origin Resource Sharing)

### 5.1 How CORS Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CORS Flow                                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Simple Request (GET, HEAD, POST with simple headers):               │
│                                                                      │
│  Browser ────GET /api/data──────▶ Server                            │
│  (origin: https://app.com)                                          │
│                                                                      │
│  Server ────Response─────────────▶ Browser                           │
│  Access-Control-Allow-Origin: https://app.com                        │
│  ─── Browser allows response ✓                                      │
│                                                                      │
│  ─────────────────────────────────────────────────────────────       │
│                                                                      │
│  Preflight Request (PUT, DELETE, custom headers, JSON):              │
│                                                                      │
│  Step 1: Browser sends OPTIONS preflight                             │
│  Browser ────OPTIONS /api/data───▶ Server                            │
│  Origin: https://app.com                                             │
│  Access-Control-Request-Method: PUT                                  │
│  Access-Control-Request-Headers: Content-Type, Authorization         │
│                                                                      │
│  Step 2: Server responds with allowed methods/headers                │
│  Server ────204 No Content───────▶ Browser                           │
│  Access-Control-Allow-Origin: https://app.com                        │
│  Access-Control-Allow-Methods: GET, POST, PUT, DELETE                │
│  Access-Control-Allow-Headers: Content-Type, Authorization           │
│  Access-Control-Max-Age: 86400                                       │
│                                                                      │
│  Step 3: Browser sends actual request                                │
│  Browser ────PUT /api/data───────▶ Server                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Flask CORS Configuration

```python
"""
CORS configuration in Flask.
pip install flask-cors
"""
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)

# ── Option 1: Allow specific origins (RECOMMENDED) ──────────────
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://app.example.com",
            "https://admin.example.com",
        ],
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["X-Request-Id", "X-RateLimit-Remaining"],
        "supports_credentials": True,
        "max_age": 86400,
    }
})

# ── Option 2: Different CORS for different routes ────────────────
app2 = Flask(__name__)

# Public API: allow any origin (no credentials)
CORS(app2, resources={
    r"/api/public/*": {
        "origins": "*",
        "methods": ["GET"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": False,  # MUST be False with origin: *
    }
})

# Private API: specific origins with credentials
CORS(app2, resources={
    r"/api/private/*": {
        "origins": ["https://app.example.com"],
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "max_age": 86400,
    }
})

# ── Option 3: Manual CORS implementation ────────────────────────
from flask import Flask, request, make_response

app3 = Flask(__name__)

ALLOWED_ORIGINS = {
    "https://app.example.com",
    "https://admin.example.com",
}

@app3.after_request
def add_cors_headers(response):
    origin = request.headers.get('Origin')

    if origin in ALLOWED_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Methods'] = (
            'GET, POST, PUT, DELETE, OPTIONS'
        )
        response.headers['Access-Control-Allow-Headers'] = (
            'Content-Type, Authorization, X-Requested-With'
        )
        response.headers['Access-Control-Expose-Headers'] = (
            'X-Request-Id, X-RateLimit-Remaining'
        )
        response.headers['Access-Control-Max-Age'] = '86400'
        # Vary: Origin tells caches that response varies by origin
        response.headers.add('Vary', 'Origin')

    return response

@app3.route('/api/data', methods=['OPTIONS'])
def preflight():
    """Handle CORS preflight requests."""
    return make_response('', 204)


# ── CORS Security Rules ─────────────────────────────────────────
"""
1. NEVER use Access-Control-Allow-Origin: * with credentials
   - This is actually blocked by browsers
   - If you need credentials, list specific origins

2. NEVER reflect the Origin header as Allow-Origin without checking
   - This is equivalent to allowing all origins
   - BAD:  response.headers['ACAO'] = request.headers['Origin']
   - GOOD: Check against an allowlist first

3. Limit Access-Control-Allow-Methods to what is actually needed
   - Don't allow DELETE if the route doesn't support it

4. Set Access-Control-Max-Age to reduce preflight requests
   - 86400 (24 hours) is reasonable

5. Use Access-Control-Expose-Headers for custom headers
   - By default, only simple headers are readable by JavaScript

6. Always add Vary: Origin when ACAO changes per request
   - Prevents cache poisoning
"""
```

---

## 6. GraphQL Security

### 6.1 GraphQL-Specific Threats

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GraphQL Security Threats                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Query Depth Attack                                               │
│     query { user { posts { comments { author { posts { ... } } } } }│
│     ──▶ Deeply nested queries cause exponential DB load              │
│                                                                      │
│  2. Query Breadth Attack                                             │
│     query { user1: user(id:1) {...} user2: user(id:2) {...} ... }   │
│     ──▶ Many aliases multiplies query cost                           │
│                                                                      │
│  3. Introspection Abuse                                              │
│     query { __schema { types { name fields { name } } } }           │
│     ──▶ Exposes entire API schema to attackers                       │
│                                                                      │
│  4. Batching Attack                                                  │
│     [{"query": "..."}, {"query": "..."}, ... x 1000]                │
│     ──▶ Multiple queries in a single request                         │
│                                                                      │
│  5. Injection via Variables                                          │
│     query ($id: String!) { user(id: $id) { ... } }                  │
│     variables: { "id": "1 OR 1=1" }                                 │
│     ──▶ SQL injection through unvalidated variables                  │
│                                                                      │
│  6. Information Disclosure                                           │
│     Verbose error messages revealing internal details                │
│     Field suggestions: "Did you mean 'secretAdminField'?"           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 GraphQL Security Mitigations

```python
"""
GraphQL security measures with graphene (Python).
pip install graphene flask-graphql
"""

# ── 1. Query Depth Limiting ─────────────────────────────────────
class DepthAnalyzer:
    """Analyze and limit GraphQL query depth."""

    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth

    def analyze(self, query_ast, depth: int = 0) -> int:
        """Calculate the maximum depth of a query."""
        if depth > self.max_depth:
            raise ValueError(
                f"Query depth {depth} exceeds maximum allowed ({self.max_depth})"
            )

        max_child_depth = depth
        if hasattr(query_ast, 'selection_set') and query_ast.selection_set:
            for selection in query_ast.selection_set.selections:
                child_depth = self.analyze(selection, depth + 1)
                max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth


# ── 2. Query Cost Analysis ──────────────────────────────────────
class QueryCostAnalyzer:
    """Estimate the cost of a GraphQL query."""

    # Define cost per field type
    FIELD_COSTS = {
        "user": 1,
        "posts": 5,       # List field, potentially expensive
        "comments": 3,
        "search": 10,     # Full-text search is expensive
    }

    def __init__(self, max_cost: int = 1000):
        self.max_cost = max_cost

    def calculate_cost(self, query_ast, multiplier: int = 1) -> int:
        """Calculate estimated query cost."""
        total_cost = 0

        if hasattr(query_ast, 'selection_set') and query_ast.selection_set:
            for selection in query_ast.selection_set.selections:
                field_name = selection.name.value
                field_cost = self.FIELD_COSTS.get(field_name, 1)

                # Check for pagination arguments that multiply cost
                args = {
                    arg.name.value: arg.value.value
                    for arg in (selection.arguments or [])
                    if hasattr(arg.value, 'value')
                }
                limit = int(args.get('first', args.get('limit', 1)))

                cost = field_cost * multiplier * max(limit, 1)
                total_cost += cost

                # Recurse into child selections
                total_cost += self.calculate_cost(selection, limit)

        if total_cost > self.max_cost:
            raise ValueError(
                f"Query cost {total_cost} exceeds maximum ({self.max_cost})"
            )

        return total_cost


# ── 3. Disable Introspection in Production ──────────────────────
"""
Introspection reveals your entire API schema.
Disable it in production.
"""
from graphql import GraphQLError

class DisableIntrospection:
    """Middleware to disable introspection queries in production."""

    def resolve(self, next, root, info, **kwargs):
        # Block __schema and __type queries
        if info.field_name in ('__schema', '__type'):
            raise GraphQLError("Introspection is disabled")
        return next(root, info, **kwargs)


# ── 4. Rate Limit + Batch Limiting ──────────────────────────────
from flask import Flask, request, jsonify

app = Flask(__name__)

MAX_BATCH_SIZE = 5  # Maximum queries per batch request

@app.before_request
def limit_batch_queries():
    """Limit the number of queries in a batch request."""
    if request.is_json:
        data = request.get_json(silent=True)
        if isinstance(data, list):
            if len(data) > MAX_BATCH_SIZE:
                return jsonify({
                    "error": "batch_limit_exceeded",
                    "message": f"Maximum {MAX_BATCH_SIZE} queries per batch"
                }), 400


# ── 5. Persisted Queries ────────────────────────────────────────
"""
Instead of accepting arbitrary queries, only accept pre-registered
query hashes. This prevents injection and query manipulation.
"""
import hashlib

# Pre-registered queries (build-time generated)
PERSISTED_QUERIES = {
    "abc123": "query { users { id name } }",
    "def456": "query GetUser($id: ID!) { user(id: $id) { id name email } }",
}

@app.route('/graphql', methods=['POST'])
def graphql_endpoint():
    data = request.get_json()

    # Only accept persisted queries in production
    query_hash = data.get('extensions', {}).get('persistedQuery', {}).get('sha256Hash')

    if query_hash:
        query = PERSISTED_QUERIES.get(query_hash)
        if not query:
            return jsonify({"error": "query_not_found"}), 404
    else:
        # In development, allow arbitrary queries
        # In production, reject:
        return jsonify({
            "error": "persisted_queries_only",
            "message": "Only persisted queries are accepted"
        }), 400

    # Execute the query...
    return jsonify({"data": {}})
```

---

## 7. API Gateway Security

### 7.1 Gateway Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    API Gateway Security Architecture                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Client                                                              │
│    │                                                                 │
│    ▼                                                                 │
│  ┌────────────────────────────────────────────────┐                  │
│  │              API Gateway                        │                 │
│  │                                                 │                 │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐       │                 │
│  │  │   TLS    │ │   Auth   │ │  Rate    │       │                 │
│  │  │Termination│ │Validation│ │ Limiting │       │                 │
│  │  └──────────┘ └──────────┘ └──────────┘       │                 │
│  │                                                 │                 │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐       │                 │
│  │  │  Input   │ │ Request  │ │  CORS    │       │                 │
│  │  │Validation│ │ Logging  │ │ Handling │       │                 │
│  │  └──────────┘ └──────────┘ └──────────┘       │                 │
│  │                                                 │                 │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐       │                 │
│  │  │  WAF     │ │ IP Allow/│ │ Response │       │                 │
│  │  │  Rules   │ │ Blocklist│ │ Filtering│       │                 │
│  │  └──────────┘ └──────────┘ └──────────┘       │                 │
│  └────────────────────────────────────────────────┘                  │
│           │              │              │                             │
│           ▼              ▼              ▼                             │
│    ┌──────────┐   ┌──────────┐   ┌──────────┐                       │
│    │ Service A│   │ Service B│   │ Service C│                       │
│    │ (Users)  │   │ (Orders) │   │ (Search) │                       │
│    └──────────┘   └──────────┘   └──────────┘                       │
│                                                                      │
│  Security benefits of centralized gateway:                           │
│  • Single point for auth enforcement                                 │
│  • Consistent rate limiting across services                          │
│  • Centralized logging and monitoring                                │
│  • Backend services don't handle TLS                                 │
│  • Simplified security policy management                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Request/Response Filtering

```python
"""
API gateway request/response filtering middleware.
"""
from flask import Flask, request, jsonify, g
import re
import uuid
import time
import logging

app = Flask(__name__)
logger = logging.getLogger('api_gateway')


# ── Request ID tracking ─────────────────────────────────────────
@app.before_request
def add_request_id():
    """Add a unique request ID for tracing."""
    g.request_id = request.headers.get(
        'X-Request-Id',
        str(uuid.uuid4())
    )
    g.request_start = time.monotonic()


@app.after_request
def add_response_headers(response):
    """Add security and tracking headers to response."""
    response.headers['X-Request-Id'] = g.request_id
    # Add timing
    elapsed = time.monotonic() - g.request_start
    response.headers['X-Response-Time'] = f"{elapsed:.3f}s"
    return response


# ── Request size limiting ───────────────────────────────────────
MAX_CONTENT_LENGTH = 1 * 1024 * 1024  # 1 MB
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

@app.before_request
def check_content_length():
    """Reject oversized requests."""
    content_length = request.content_length
    if content_length and content_length > MAX_CONTENT_LENGTH:
        return jsonify({
            "error": "payload_too_large",
            "max_bytes": MAX_CONTENT_LENGTH,
        }), 413


# ── IP allowlist/blocklist ──────────────────────────────────────
BLOCKED_IPS = {"192.168.1.100", "10.0.0.50"}
ADMIN_ALLOWED_IPS = {"10.0.0.1", "10.0.0.2"}

@app.before_request
def check_ip():
    """Block requests from banned IPs."""
    client_ip = request.remote_addr

    if client_ip in BLOCKED_IPS:
        logger.warning(f"Blocked request from banned IP: {client_ip}")
        return jsonify({"error": "forbidden"}), 403

    # Admin routes require specific IPs
    if request.path.startswith('/admin/'):
        if client_ip not in ADMIN_ALLOWED_IPS:
            return jsonify({"error": "forbidden"}), 403


# ── Response data filtering ─────────────────────────────────────
SENSITIVE_FIELDS = {'password', 'ssn', 'credit_card', 'secret_key',
                    'token', 'api_key'}

def filter_sensitive_data(data):
    """Recursively remove sensitive fields from response data."""
    if isinstance(data, dict):
        return {
            k: filter_sensitive_data(v)
            for k, v in data.items()
            if k.lower() not in SENSITIVE_FIELDS
        }
    elif isinstance(data, list):
        return [filter_sensitive_data(item) for item in data]
    return data


# ── Audit logging ───────────────────────────────────────────────
@app.after_request
def audit_log(response):
    """Log every API request for audit trail."""
    logger.info(
        "API Request: method=%s path=%s status=%s "
        "ip=%s user_agent=%s request_id=%s duration=%s",
        request.method,
        request.path,
        response.status_code,
        request.remote_addr,
        request.user_agent.string[:100],
        g.request_id,
        response.headers.get('X-Response-Time', 'N/A'),
    )
    return response
```

---

## 8. OpenAPI Security Definitions

### 8.1 Security Schemes in OpenAPI 3.0

```yaml
# openapi.yaml - Security scheme definitions
openapi: 3.0.3
info:
  title: Secure API
  version: 1.0.0

# ── Security Scheme Definitions ──────────────────────────────────
components:
  securitySchemes:
    # API Key authentication
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key
      description: API key for server-to-server communication

    # JWT Bearer token
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: JWT access token

    # OAuth 2.0
    OAuth2:
      type: oauth2
      flows:
        authorizationCode:
          authorizationUrl: https://auth.example.com/authorize
          tokenUrl: https://auth.example.com/token
          refreshUrl: https://auth.example.com/refresh
          scopes:
            read: Read access to resources
            write: Write access to resources
            admin: Administrative access

    # OpenID Connect
    OpenIdConnect:
      type: openIdConnect
      openIdConnectUrl: https://auth.example.com/.well-known/openid-configuration

# ── Global security (applies to all endpoints) ──────────────────
security:
  - BearerAuth: []

# ── Per-endpoint security ────────────────────────────────────────
paths:
  /api/public/status:
    get:
      summary: Health check (no auth required)
      security: []  # Override: no authentication
      responses:
        '200':
          description: OK

  /api/users:
    get:
      summary: List users
      security:
        - BearerAuth: []
        - ApiKeyAuth: []  # Alternative auth (OR)
      responses:
        '200':
          description: User list

  /api/admin/settings:
    put:
      summary: Update settings (admin only)
      security:
        - OAuth2: [admin]  # Requires 'admin' scope
      responses:
        '200':
          description: Settings updated

  /api/data:
    post:
      summary: Create data (requires read + write)
      security:
        - OAuth2: [read, write]  # Requires BOTH scopes
      responses:
        '201':
          description: Data created
```

### 8.2 API Versioning Security

```python
"""
API versioning considerations for security.
"""

# ── URL path versioning ──────────────────────────────────────────
# /api/v1/users  (old, may have known vulnerabilities)
# /api/v2/users  (current, patched)

# ── Header versioning ───────────────────────────────────────────
# Accept: application/vnd.example.v2+json

# ── Security concerns with versioning ───────────────────────────
"""
1. Old API versions may have known vulnerabilities
   - Set deprecation dates and enforce them
   - Return Deprecation and Sunset headers

2. Don't maintain security patches for deprecated versions
   - Force migration to latest version

3. Monitor usage of deprecated versions
   - Alert when old versions are still in use
"""

from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

API_VERSIONS = {
    "v1": {
        "status": "deprecated",
        "sunset": "2025-06-01",
        "successor": "v2",
    },
    "v2": {
        "status": "current",
        "sunset": None,
        "successor": None,
    },
}

@app.before_request
def check_api_version():
    """Warn or block deprecated API versions."""
    # Extract version from URL path
    path_parts = request.path.strip('/').split('/')
    if len(path_parts) >= 2 and path_parts[0] == 'api':
        version = path_parts[1]
        version_info = API_VERSIONS.get(version)

        if not version_info:
            return jsonify({
                "error": "invalid_version",
                "supported": list(API_VERSIONS.keys()),
            }), 400

        if version_info["status"] == "deprecated":
            sunset_date = version_info["sunset"]
            if sunset_date:
                # Check if past sunset date
                if datetime.now() > datetime.fromisoformat(sunset_date):
                    return jsonify({
                        "error": "version_retired",
                        "message": f"API {version} was retired on {sunset_date}",
                        "upgrade_to": version_info["successor"],
                    }), 410  # 410 Gone


@app.after_request
def add_deprecation_headers(response):
    """Add deprecation warnings to response headers."""
    path_parts = request.path.strip('/').split('/')
    if len(path_parts) >= 2 and path_parts[0] == 'api':
        version = path_parts[1]
        version_info = API_VERSIONS.get(version, {})

        if version_info.get("status") == "deprecated":
            response.headers['Deprecation'] = 'true'
            if version_info.get("sunset"):
                response.headers['Sunset'] = version_info["sunset"]
            response.headers['Link'] = (
                f'</api/{version_info["successor"]}>; rel="successor-version"'
            )

    return response
```

---

## 9. Request/Response Encryption

### 9.1 Transport Layer Security

```python
"""
TLS configuration and certificate pinning.
"""

# ── Flask with TLS (development) ────────────────────────────────
# Generate self-signed cert for development:
# openssl req -x509 -newkey rsa:4096 -nodes \
#   -out cert.pem -keyout key.pem -days 365

from flask import Flask
app = Flask(__name__)

# Run with TLS (development only)
# In production, TLS is handled by reverse proxy (nginx, load balancer)
if __name__ == '__main__':
    app.run(
        ssl_context=('cert.pem', 'key.pem'),
        host='0.0.0.0',
        port=443,
    )

# ── Certificate Pinning (for API clients) ───────────────────────
import requests
import hashlib
import ssl

def verify_certificate_pin(host: str, expected_pin: str) -> bool:
    """Verify server certificate matches expected pin."""
    import socket

    context = ssl.create_default_context()
    with socket.create_connection((host, 443)) as sock:
        with context.wrap_socket(sock, server_hostname=host) as ssock:
            cert_der = ssock.getpeercert(True)
            cert_hash = hashlib.sha256(cert_der).hexdigest()
            return cert_hash == expected_pin


# ── Payload-level encryption (for sensitive fields) ─────────────
from cryptography.fernet import Fernet

# Generate key (store securely, not in code)
encryption_key = Fernet.generate_key()
cipher = Fernet(encryption_key)

def encrypt_sensitive_fields(data: dict, fields: list[str]) -> dict:
    """Encrypt specific fields in a response payload."""
    result = data.copy()
    for field in fields:
        if field in result:
            value = str(result[field]).encode()
            result[field] = cipher.encrypt(value).decode()
    return result

def decrypt_sensitive_fields(data: dict, fields: list[str]) -> dict:
    """Decrypt specific fields in a request payload."""
    result = data.copy()
    for field in fields:
        if field in result:
            value = result[field].encode()
            result[field] = cipher.decrypt(value).decode()
    return result
```

---

## 10. Exercises

### Exercise 1: Secure JWT Authentication Service

Build a complete JWT authentication service with Flask that includes:

1. User registration with password hashing (argon2 or bcrypt)
2. Login endpoint that returns access + refresh tokens
3. Token refresh endpoint with refresh token rotation
4. Logout endpoint with token blacklisting (use Redis or in-memory set)
5. A protected endpoint that requires the "admin" scope
6. Proper error handling for expired, malformed, and revoked tokens
7. Rate limiting on the login endpoint (5 attempts per minute per IP)

### Exercise 2: CORS Security Audit

Write a Python script that:

1. Takes a list of API URLs
2. Sends requests with various Origin headers
3. Tests if the API reflects arbitrary origins (vulnerability)
4. Checks if credentials are allowed with wildcard origins
5. Tests preflight handling for common methods and headers
6. Generates a security report for each endpoint

### Exercise 3: GraphQL Security Middleware

Implement a GraphQL security middleware that:

1. Limits query depth to a configurable maximum (default: 10)
2. Calculates query cost and rejects expensive queries
3. Disables introspection in production
4. Limits batch query size
5. Logs all queries with their cost and execution time
6. Implements persisted queries with a hash allowlist

### Exercise 4: Rate Limiter with Sliding Window

Implement a sliding window log rate limiter that:

1. Uses Redis as the backing store (or an in-memory simulation)
2. Tracks exact timestamps of each request
3. Supports configurable windows (per second, minute, hour)
4. Supports different limits for different API key tiers
5. Returns proper rate limit headers (X-RateLimit-Limit, Remaining, Reset)
6. Handles distributed deployment (multiple server instances)

### Exercise 5: API Input Validation Framework

Build a reusable validation framework that:

1. Validates JSON request bodies against schemas
2. Validates query parameters with type coercion
3. Validates path parameters
4. Supports nested object validation
5. Returns consistent error response format (RFC 7807)
6. Protects against mass assignment (reject unknown fields)
7. Sanitizes string inputs (trim, normalize Unicode)

### Exercise 6: API Security Scanner

Create a security scanning tool that tests an API for:

1. Missing authentication on endpoints
2. Broken object-level authorization (BOLA/IDOR)
3. Missing rate limiting
4. Verbose error messages revealing internal details
5. Missing security headers
6. CORS misconfiguration
7. Generate a report with severity ratings

---

## Summary

### API Security Checklist

| Category | Item | Priority |
|----------|------|----------|
| Authentication | Use OAuth 2.0 or JWT (not API keys alone for users) | Critical |
| Authentication | Short-lived access tokens (15 min) | Critical |
| Authentication | Refresh token rotation | High |
| Authorization | Check object-level permissions (prevent BOLA) | Critical |
| Authorization | Validate scopes on every request | Critical |
| Rate Limiting | Per-IP and per-user rate limits | High |
| Rate Limiting | Stricter limits on auth endpoints | High |
| Input Validation | Schema validation on all inputs | Critical |
| Input Validation | Reject unknown fields | High |
| CORS | Specific origin allowlist (no wildcard with credentials) | Critical |
| CORS | Vary: Origin header | Medium |
| Encryption | TLS everywhere (HTTPS only) | Critical |
| Logging | Log all requests with unique request IDs | High |
| Versioning | Deprecate old versions with sunset dates | Medium |
| Gateway | Centralized auth and rate limiting | Recommended |

### Key Takeaways

1. **Defense in depth** — apply security at every layer (network, gateway, application, database)
2. **Never trust client input** — validate everything, on the server, every time
3. **Use established libraries** — do not roll your own authentication or encryption
4. **Monitor and alert** — security without monitoring is incomplete
5. **Document your API security** — use OpenAPI security schemes for clarity

---

**Previous**: [09_Web_Security_Headers.md](./09_Web_Security_Headers.md) | **Next**: [11_Secrets_Management.md](./11_Secrets_Management.md)
