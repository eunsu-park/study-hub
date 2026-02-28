"""
Exercise Solutions: API Security
================================
Lesson 10 from Security topic.

Covers JWT auth service, CORS auditing, GraphQL security,
sliding window rate limiting, input validation, and API scanning.
"""

import hashlib
import hmac
import json
import os
import re
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Exercise 1: Secure JWT Authentication Service
# ---------------------------------------------------------------------------

class JWTAuthService:
    """
    Complete JWT authentication service with token rotation and blacklisting.
    Uses a simplified JWT implementation for portability.
    """

    def __init__(self, secret: str):
        self._secret = secret.encode()
        self._users: dict[str, dict] = {}
        self._refresh_tokens: dict[str, dict] = {}  # token -> {user_id, expires}
        self._blacklist: set[str] = set()
        self._login_attempts: dict[str, list[float]] = defaultdict(list)

    def _hash_password(self, password: str) -> str:
        salt = os.urandom(16)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600_000)
        return f"{salt.hex()}:{dk.hex()}"

    def _verify_password(self, stored: str, password: str) -> bool:
        salt_hex, dk_hex = stored.split(":")
        salt = bytes.fromhex(salt_hex)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600_000)
        return hmac.compare_digest(dk.hex(), dk_hex)

    def _create_token(self, user_id: str, scopes: list[str],
                      expires_in: int = 900) -> str:
        """Create a simple signed token (simulated JWT)."""
        payload = {
            "sub": user_id,
            "scopes": scopes,
            "exp": int(time.time()) + expires_in,
            "jti": secrets.token_urlsafe(16),
            "iat": int(time.time()),
        }
        payload_b64 = json.dumps(payload)
        sig = hmac.new(self._secret, payload_b64.encode(), hashlib.sha256).hexdigest()
        return f"{payload_b64}|{sig}"

    def _verify_token(self, token: str) -> Optional[dict]:
        """Verify and decode a token."""
        if "|" not in token:
            return None
        payload_str, sig = token.rsplit("|", 1)
        expected = hmac.new(
            self._secret, payload_str.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(payload_str)
        if payload.get("jti") in self._blacklist:
            return None
        if payload.get("exp", 0) < time.time():
            return None
        return payload

    def register(self, user_id: str, password: str, scopes: list[str]) -> dict:
        """Register a new user."""
        if user_id in self._users:
            return {"error": "User already exists"}
        self._users[user_id] = {
            "password_hash": self._hash_password(password),
            "scopes": scopes,
        }
        return {"status": "registered", "user_id": user_id}

    def login(self, user_id: str, password: str, ip: str = "0.0.0.0") -> dict:
        """Login with rate limiting."""
        # Rate limiting: 5 attempts per minute per IP
        now = time.time()
        attempts = self._login_attempts[ip]
        # Clean old attempts
        attempts[:] = [t for t in attempts if now - t < 60]
        if len(attempts) >= 5:
            return {"error": "Too many login attempts. Try again later."}

        attempts.append(now)

        user = self._users.get(user_id)
        if not user or not self._verify_password(user["password_hash"], password):
            return {"error": "Invalid credentials"}

        # Generate tokens
        access_token = self._create_token(user_id, user["scopes"], expires_in=900)
        refresh_token = secrets.token_urlsafe(64)
        self._refresh_tokens[refresh_token] = {
            "user_id": user_id,
            "expires": now + 7 * 86400,
        }

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": 900,
            "token_type": "Bearer",
        }

    def refresh(self, refresh_token: str) -> dict:
        """Refresh access token with rotation."""
        stored = self._refresh_tokens.pop(refresh_token, None)
        if not stored or stored["expires"] < time.time():
            return {"error": "Invalid or expired refresh token"}

        user_id = stored["user_id"]
        user = self._users.get(user_id)
        if not user:
            return {"error": "User not found"}

        # Issue new tokens (rotation)
        access_token = self._create_token(user_id, user["scopes"], expires_in=900)
        new_refresh = secrets.token_urlsafe(64)
        self._refresh_tokens[new_refresh] = {
            "user_id": user_id,
            "expires": time.time() + 7 * 86400,
        }

        return {
            "access_token": access_token,
            "refresh_token": new_refresh,
            "expires_in": 900,
        }

    def logout(self, token: str) -> dict:
        """Blacklist a token."""
        payload = self._verify_token(token)
        if payload:
            self._blacklist.add(payload["jti"])
        return {"status": "logged out"}

    def authorize(self, token: str, required_scope: str) -> Optional[dict]:
        """Verify token and check scope."""
        payload = self._verify_token(token)
        if not payload:
            return None
        if required_scope not in payload.get("scopes", []):
            return None
        return payload


def exercise_1_jwt_auth_service():
    """Demonstrate the JWT auth service."""
    svc = JWTAuthService(secret=secrets.token_hex(32))

    # Register users
    svc.register("alice", "SecurePass123!", ["user", "admin"])
    svc.register("bob", "AnotherPass456!", ["user"])

    # Login
    result = svc.login("alice", "SecurePass123!")
    print(f"Login: {list(result.keys())}")
    access_token = result["access_token"]
    refresh_token = result["refresh_token"]

    # Authorize with admin scope
    payload = svc.authorize(access_token, "admin")
    print(f"Admin access: {payload is not None}")

    # Bob does not have admin scope
    bob_result = svc.login("bob", "AnotherPass456!")
    bob_payload = svc.authorize(bob_result["access_token"], "admin")
    print(f"Bob admin access: {bob_payload is not None}")

    # Refresh token
    new_tokens = svc.refresh(refresh_token)
    print(f"Token refreshed: {'access_token' in new_tokens}")

    # Old refresh token no longer works (rotation)
    reuse = svc.refresh(refresh_token)
    print(f"Old refresh reuse blocked: {'error' in reuse}")

    # Logout (blacklist)
    svc.logout(access_token)
    revoked = svc.authorize(access_token, "user")
    print(f"Token after logout: {revoked is None}")


# ---------------------------------------------------------------------------
# Exercise 2: CORS Security Audit
# ---------------------------------------------------------------------------

def exercise_2_cors_audit():
    """
    CORS security audit script (analysis-only, does not make real requests).
    """
    print("CORS Security Audit Methodology")
    print("=" * 60)

    test_cases = [
        {
            "test": "Reflect arbitrary Origin",
            "origin": "https://evil.com",
            "expected": "Should NOT see Access-Control-Allow-Origin: https://evil.com",
            "severity": "Critical",
        },
        {
            "test": "Null origin reflection",
            "origin": "null",
            "expected": "Should NOT see Access-Control-Allow-Origin: null",
            "severity": "High",
        },
        {
            "test": "Wildcard with credentials",
            "origin": "https://any-origin.com",
            "expected": "Should NOT see * with Access-Control-Allow-Credentials: true",
            "severity": "Critical",
        },
        {
            "test": "Subdomain bypass",
            "origin": "https://evil.trusted-domain.com",
            "expected": "Should validate exact origin, not just suffix",
            "severity": "High",
        },
        {
            "test": "Preflight caching",
            "origin": "https://trusted.com",
            "expected": "Access-Control-Max-Age should be reasonable (< 86400)",
            "severity": "Low",
        },
    ]

    for tc in test_cases:
        print(f"\n  Test: {tc['test']}")
        print(f"  Origin: {tc['origin']}")
        print(f"  Expected: {tc['expected']}")
        print(f"  Severity: {tc['severity']}")

    # Secure CORS configuration
    print("\n\nSecure CORS Configuration:")
    print('  ALLOWED_ORIGINS = {"https://app.example.com", "https://admin.example.com"}')
    print("  Always validate against exact allowlist")
    print("  Never reflect arbitrary origins")
    print("  Use Vary: Origin header when origin-dependent")
    print("  Never combine wildcard (*) with credentials")


# ---------------------------------------------------------------------------
# Exercise 3: GraphQL Security Middleware
# ---------------------------------------------------------------------------

class GraphQLSecurityMiddleware:
    """Security middleware for GraphQL APIs."""

    def __init__(self, max_depth: int = 10, max_cost: int = 1000,
                 max_batch: int = 5, disable_introspection: bool = True):
        self.max_depth = max_depth
        self.max_cost = max_cost
        self.max_batch = max_batch
        self.disable_introspection = disable_introspection
        self._persisted_queries: dict[str, str] = {}

    def check_depth(self, query: str, current_depth: int = 0) -> int:
        """Estimate query depth by counting nesting levels."""
        depth = 0
        max_found = current_depth
        for char in query:
            if char == "{":
                depth += 1
                max_found = max(max_found, current_depth + depth)
            elif char == "}":
                depth -= 1
        return max_found

    def estimate_cost(self, query: str) -> int:
        """Estimate query cost based on field count and nesting."""
        field_count = query.count("\n") + query.count(",")
        nesting = self.check_depth(query)
        return field_count * (nesting + 1)

    def validate_query(self, query: str, is_production: bool = True) -> tuple[bool, str]:
        """Validate a GraphQL query against security rules."""
        # Check introspection
        if self.disable_introspection and is_production:
            if "__schema" in query or "__type" in query:
                return False, "Introspection is disabled in production"

        # Check depth
        depth = self.check_depth(query)
        if depth > self.max_depth:
            return False, f"Query depth {depth} exceeds maximum {self.max_depth}"

        # Check cost
        cost = self.estimate_cost(query)
        if cost > self.max_cost:
            return False, f"Query cost {cost} exceeds maximum {self.max_cost}"

        return True, f"OK (depth={depth}, cost={cost})"

    def validate_batch(self, queries: list[str]) -> tuple[bool, str]:
        """Validate a batch of queries."""
        if len(queries) > self.max_batch:
            return False, f"Batch size {len(queries)} exceeds maximum {self.max_batch}"
        for i, q in enumerate(queries):
            ok, msg = self.validate_query(q)
            if not ok:
                return False, f"Query {i}: {msg}"
        return True, "OK"

    def register_persisted_query(self, query_hash: str, query: str):
        """Register a persisted query."""
        self._persisted_queries[query_hash] = query

    def get_persisted_query(self, query_hash: str) -> Optional[str]:
        """Look up a persisted query by hash."""
        return self._persisted_queries.get(query_hash)


def exercise_3_graphql_security():
    """Demonstrate GraphQL security middleware."""
    middleware = GraphQLSecurityMiddleware(max_depth=5, max_cost=50)

    # Normal query
    ok, msg = middleware.validate_query("{ users { name email } }")
    print(f"Normal query: {ok} ({msg})")

    # Deep nested query
    deep_query = "{ a { b { c { d { e { f { g { h } } } } } } } }"
    ok, msg = middleware.validate_query(deep_query)
    print(f"Deep query: {ok} ({msg})")

    # Introspection attempt
    ok, msg = middleware.validate_query("{ __schema { types { name } } }")
    print(f"Introspection: {ok} ({msg})")

    # Batch limit
    batch = ["{ users { name } }"] * 10
    ok, msg = middleware.validate_batch(batch)
    print(f"Large batch: {ok} ({msg})")


# ---------------------------------------------------------------------------
# Exercise 4: Sliding Window Rate Limiter
# ---------------------------------------------------------------------------

class SlidingWindowRateLimiter:
    """Sliding window log rate limiter."""

    def __init__(self):
        self._logs: dict[str, list[float]] = defaultdict(list)
        self._tiers = {
            "free": {"per_second": 1, "per_minute": 30, "per_hour": 500},
            "basic": {"per_second": 5, "per_minute": 100, "per_hour": 3000},
            "premium": {"per_second": 20, "per_minute": 500, "per_hour": 10000},
        }

    def _clean_logs(self, key: str, window: float):
        """Remove entries outside the window."""
        cutoff = time.time() - window
        self._logs[key] = [t for t in self._logs[key] if t > cutoff]

    def is_allowed(self, client_id: str, tier: str = "free") -> tuple[bool, dict]:
        """Check if a request is allowed under rate limits."""
        now = time.time()
        limits = self._tiers.get(tier, self._tiers["free"])

        # Check each window
        for window_name, (window_secs, limit) in [
            ("per_second", (1, limits["per_second"])),
            ("per_minute", (60, limits["per_minute"])),
            ("per_hour", (3600, limits["per_hour"])),
        ]:
            key = f"{client_id}:{window_name}"
            self._clean_logs(key, window_secs)

            if len(self._logs[key]) >= limit:
                # Calculate reset time
                oldest = min(self._logs[key]) if self._logs[key] else now
                reset = oldest + window_secs

                return False, {
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(reset)),
                    "Retry-After": str(int(reset - now)),
                    "violated_window": window_name,
                }

        # Record request in all windows
        for window_name in ["per_second", "per_minute", "per_hour"]:
            key = f"{client_id}:{window_name}"
            self._logs[key].append(now)

        # Calculate remaining for per_minute
        minute_key = f"{client_id}:per_minute"
        remaining = limits["per_minute"] - len(self._logs[minute_key])

        return True, {
            "X-RateLimit-Limit": str(limits["per_minute"]),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(now + 60)),
        }


def exercise_4_rate_limiter():
    """Demonstrate the sliding window rate limiter."""
    limiter = SlidingWindowRateLimiter()

    # Normal requests (free tier: 1/second)
    for i in range(3):
        allowed, headers = limiter.is_allowed("client-1", "free")
        print(f"Request {i+1}: {'allowed' if allowed else 'BLOCKED'} "
              f"(remaining: {headers.get('X-RateLimit-Remaining')})")

    # Premium tier (20/second)
    print("\nPremium tier:")
    for i in range(25):
        allowed, headers = limiter.is_allowed("client-2", "premium")
    print(f"After 25 requests: {'allowed' if allowed else 'BLOCKED'} "
          f"(remaining: {headers.get('X-RateLimit-Remaining')})")


# ---------------------------------------------------------------------------
# Exercise 5: API Input Validation Framework
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """RFC 7807 Problem Details error."""
    def __init__(self, errors: list[dict]):
        self.errors = errors

    def to_rfc7807(self) -> dict:
        return {
            "type": "https://api.example.com/errors/validation",
            "title": "Validation Error",
            "status": 400,
            "detail": "One or more fields failed validation",
            "errors": self.errors,
        }


class APIValidator:
    """Reusable API input validation framework."""

    @staticmethod
    def validate_string(value: Any, field: str, min_len: int = 0,
                        max_len: int = 1000, pattern: str = None) -> str:
        """Validate a string field."""
        if not isinstance(value, str):
            raise ValidationError([{
                "field": field, "message": "Must be a string"
            }])
        value = value.strip()
        # Normalize Unicode
        import unicodedata
        value = unicodedata.normalize("NFC", value)

        if len(value) < min_len:
            raise ValidationError([{
                "field": field,
                "message": f"Minimum length is {min_len}"
            }])
        if len(value) > max_len:
            raise ValidationError([{
                "field": field,
                "message": f"Maximum length is {max_len}"
            }])
        if pattern and not re.match(pattern, value):
            raise ValidationError([{
                "field": field,
                "message": f"Does not match required pattern"
            }])
        return value

    @staticmethod
    def validate_email(value: Any, field: str = "email") -> str:
        """Validate an email address."""
        if not isinstance(value, str):
            raise ValidationError([{"field": field, "message": "Must be a string"}])
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, value.strip()):
            raise ValidationError([{"field": field, "message": "Invalid email"}])
        return value.strip().lower()

    @staticmethod
    def validate_integer(value: Any, field: str, min_val: int = None,
                         max_val: int = None) -> int:
        """Validate an integer field with type coercion."""
        try:
            result = int(value)
        except (ValueError, TypeError):
            raise ValidationError([{"field": field, "message": "Must be an integer"}])
        if min_val is not None and result < min_val:
            raise ValidationError([{
                "field": field, "message": f"Minimum value is {min_val}"
            }])
        if max_val is not None and result > max_val:
            raise ValidationError([{
                "field": field, "message": f"Maximum value is {max_val}"
            }])
        return result

    @staticmethod
    def reject_unknown_fields(data: dict, allowed: set[str], context: str = "body"):
        """Reject unknown fields (mass assignment protection)."""
        unknown = set(data.keys()) - allowed
        if unknown:
            raise ValidationError([{
                "field": f,
                "message": f"Unknown field in {context}"
            } for f in unknown])


def exercise_5_input_validation():
    """Demonstrate the API input validation framework."""
    v = APIValidator()

    # Valid input
    try:
        name = v.validate_string("  Alice Smith  ", "name", min_len=1, max_len=100)
        email = v.validate_email("alice@example.com")
        age = v.validate_integer("25", "age", min_val=0, max_val=150)
        print(f"Valid: name={name}, email={email}, age={age}")
    except ValidationError as e:
        print(f"Error: {e.to_rfc7807()}")

    # Invalid inputs
    for test_name, test_fn in [
        ("Empty name", lambda: v.validate_string("", "name", min_len=1)),
        ("Bad email", lambda: v.validate_email("not-an-email")),
        ("Age too high", lambda: v.validate_integer(200, "age", max_val=150)),
    ]:
        try:
            test_fn()
        except ValidationError as e:
            print(f"{test_name}: {e.errors[0]['message']}")

    # Mass assignment protection
    try:
        data = {"name": "Alice", "email": "a@b.com", "is_admin": True}
        v.reject_unknown_fields(data, {"name", "email"})
    except ValidationError as e:
        print(f"Unknown fields rejected: {e.errors}")


# ---------------------------------------------------------------------------
# Exercise 6: API Security Scanner (Conceptual)
# ---------------------------------------------------------------------------

def exercise_6_api_scanner():
    """API security scanning methodology and checks."""
    checks = [
        {
            "name": "Missing Authentication",
            "method": "Send requests without Authorization header",
            "expected": "Should receive 401 Unauthorized",
            "severity": "Critical",
        },
        {
            "name": "Broken Object-Level Authorization (BOLA/IDOR)",
            "method": "Authenticate as User A, access User B's resources",
            "expected": "Should receive 403 Forbidden",
            "severity": "Critical",
        },
        {
            "name": "Missing Rate Limiting",
            "method": "Send 100 requests in 1 second",
            "expected": "Should receive 429 Too Many Requests",
            "severity": "High",
        },
        {
            "name": "Verbose Error Messages",
            "method": "Send malformed requests, check error responses",
            "expected": "Should NOT reveal stack traces or internal details",
            "severity": "Medium",
        },
        {
            "name": "Missing Security Headers",
            "method": "Check response headers for CSP, HSTS, etc.",
            "expected": "All recommended security headers present",
            "severity": "Medium",
        },
        {
            "name": "CORS Misconfiguration",
            "method": "Send requests with various Origin headers",
            "expected": "Should NOT reflect arbitrary origins",
            "severity": "High",
        },
    ]

    print("API Security Scanner Checks")
    print("=" * 60)
    for c in checks:
        print(f"\n  [{c['severity']}] {c['name']}")
        print(f"  Method: {c['method']}")
        print(f"  Expected: {c['expected']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: JWT Auth Service")
    print("=" * 70)
    exercise_1_jwt_auth_service()

    print("\n" + "=" * 70)
    print("Exercise 2: CORS Security Audit")
    print("=" * 70)
    exercise_2_cors_audit()

    print("\n" + "=" * 70)
    print("Exercise 3: GraphQL Security Middleware")
    print("=" * 70)
    exercise_3_graphql_security()

    print("\n" + "=" * 70)
    print("Exercise 4: Sliding Window Rate Limiter")
    print("=" * 70)
    exercise_4_rate_limiter()

    print("\n" + "=" * 70)
    print("Exercise 5: API Input Validation Framework")
    print("=" * 70)
    exercise_5_input_validation()

    print("\n" + "=" * 70)
    print("Exercise 6: API Security Scanner")
    print("=" * 70)
    exercise_6_api_scanner()
