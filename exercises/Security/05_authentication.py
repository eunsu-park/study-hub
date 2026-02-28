"""
Exercise Solutions: Authentication Systems
==========================================
Lesson 05 from Security topic.

Covers secure password storage, TOTP-based 2FA, JWT security audit,
OAuth 2.0 with PKCE, and password reset security.

Dependencies:
    pip install cryptography PyJWT pyotp
"""

import hashlib
import hmac
import json
import os
import secrets
import time
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlencode, urlparse, parse_qs


# ---------------------------------------------------------------------------
# Exercise 1: Secure Password Storage (UserService)
# ---------------------------------------------------------------------------

# Simulated argon2-like password hashing for portability
# In production, use: from argon2 import PasswordHasher

class SimplePasswordHasher:
    """Simulated secure password hasher (in production, use argon2-cffi)."""

    @staticmethod
    def hash(password: str) -> str:
        salt = os.urandom(16)
        # PBKDF2 with high iteration count as argon2 stand-in
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600_000)
        return f"$pbkdf2${salt.hex()}${dk.hex()}"

    @staticmethod
    def verify(stored_hash: str, password: str) -> bool:
        parts = stored_hash.split("$")
        salt = bytes.fromhex(parts[2])
        expected = bytes.fromhex(parts[3])
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600_000)
        return hmac.compare_digest(dk, expected)


# Common password list (small sample for demonstration)
COMMON_PASSWORDS = {
    "password", "123456", "qwerty", "letmein", "admin",
    "welcome", "monkey", "dragon", "master", "login",
}


@dataclass
class User:
    id: int
    username: str
    email: str
    password_hash: str
    failed_attempts: int = 0
    locked_until: float = 0.0
    sessions: list[str] = field(default_factory=list)


class UserService:
    """Complete user registration and authentication service."""

    def __init__(self):
        self._users: dict[str, User] = {}
        self._emails: dict[str, str] = {}  # email -> username
        self._next_id = 1
        self._hasher = SimplePasswordHasher()
        self._max_failed_attempts = 5
        self._lockout_duration = 300  # 5 minutes

    def _validate_password(self, password: str) -> tuple[bool, str]:
        """Validate password strength."""
        if len(password) < 12:
            return False, "Password must be at least 12 characters"
        if password.lower() in COMMON_PASSWORDS:
            return False, "Password is too common"
        if not re.search(r"[A-Z]", password):
            return False, "Password must contain an uppercase letter"
        if not re.search(r"[a-z]", password):
            return False, "Password must contain a lowercase letter"
        if not re.search(r"\d", password):
            return False, "Password must contain a digit"
        return True, "OK"

    def register(self, username: str, email: str, password: str) -> dict:
        """Register a new user with validated password."""
        # Input validation
        if not username or not email:
            return {"error": "Username and email are required"}

        # Check uniqueness
        if username in self._users:
            return {"error": "Username already exists"}
        if email in self._emails:
            return {"error": "Email already registered"}

        # Validate password strength
        valid, msg = self._validate_password(password)
        if not valid:
            return {"error": msg}

        # Hash password with argon2id (simulated)
        password_hash = self._hasher.hash(password)

        user = User(
            id=self._next_id,
            username=username,
            email=email,
            password_hash=password_hash,
        )
        self._users[username] = user
        self._emails[email] = username
        self._next_id += 1

        return {"user_id": user.id, "username": user.username, "email": user.email}

    def login(self, username: str, password: str) -> dict:
        """Authenticate with account lockout and session management."""
        user = self._users.get(username)
        if not user:
            # Same timing as a real check to prevent user enumeration
            self._hasher.hash("dummy-timing-equalization")
            return {"error": "Invalid credentials"}

        # Check lockout
        if user.locked_until > time.time():
            remaining = int(user.locked_until - time.time())
            return {"error": f"Account locked. Try again in {remaining}s"}

        # Verify password
        if not self._hasher.verify(user.password_hash, password):
            user.failed_attempts += 1
            if user.failed_attempts >= self._max_failed_attempts:
                user.locked_until = time.time() + self._lockout_duration
                user.failed_attempts = 0
                return {"error": "Account locked due to too many failed attempts"}
            return {"error": "Invalid credentials"}

        # Success: reset failures, generate session
        user.failed_attempts = 0
        session_id = secrets.token_urlsafe(32)
        user.sessions.append(session_id)

        # Generate tokens
        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(64)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "session_id": session_id,
        }

    def change_password(self, user_id: int, old_password: str,
                        new_password: str) -> bool:
        """Change password, invalidating all existing sessions."""
        user = next(
            (u for u in self._users.values() if u.id == user_id), None
        )
        if not user:
            return False

        # Verify old password
        if not self._hasher.verify(user.password_hash, old_password):
            return False

        # Validate new password
        valid, msg = self._validate_password(new_password)
        if not valid:
            return False

        # Update password and invalidate all sessions
        user.password_hash = self._hasher.hash(new_password)
        user.sessions.clear()
        return True


def exercise_1_secure_password_storage():
    """Demonstrate the UserService."""
    svc = UserService()

    # Register
    result = svc.register("alice", "alice@example.com", "MyStr0ng!Pass2025")
    print(f"Register: {result}")

    # Weak password rejected
    result = svc.register("bob", "bob@example.com", "password")
    print(f"Weak password: {result}")

    # Login
    result = svc.login("alice", "MyStr0ng!Pass2025")
    print(f"Login success: {list(result.keys())}")

    # Wrong password
    result = svc.login("alice", "wrong-password")
    print(f"Wrong password: {result}")

    # Change password
    ok = svc.change_password(1, "MyStr0ng!Pass2025", "NewSecure!Pass2025")
    print(f"Password changed: {ok}")


# ---------------------------------------------------------------------------
# Exercise 2: TOTP Integration
# ---------------------------------------------------------------------------

class TOTPManager:
    """TOTP-based 2FA manager (simulated without pyotp for portability)."""

    @staticmethod
    def generate_secret() -> str:
        """Generate a base32-encoded TOTP secret."""
        import base64
        return base64.b32encode(os.urandom(20)).decode().rstrip("=")

    @staticmethod
    def generate_totp(secret: str, time_step: int = 30) -> str:
        """Generate a TOTP code (RFC 6238 simplified)."""
        import base64
        import struct
        key = base64.b32decode(secret + "=" * (-len(secret) % 8))
        counter = int(time.time()) // time_step
        msg = struct.pack(">Q", counter)
        h = hmac.new(key, msg, hashlib.sha1).digest()
        offset = h[-1] & 0x0F
        code = struct.unpack(">I", h[offset:offset + 4])[0] & 0x7FFFFFFF
        return f"{code % 1_000_000:06d}"

    @staticmethod
    def verify_totp(secret: str, code: str, window: int = 1) -> bool:
        """Verify TOTP code with time window tolerance."""
        import base64
        import struct
        key = base64.b32decode(secret + "=" * (-len(secret) % 8))
        current_counter = int(time.time()) // 30

        for offset in range(-window, window + 1):
            counter = current_counter + offset
            msg = struct.pack(">Q", counter)
            h = hmac.new(key, msg, hashlib.sha1).digest()
            off = h[-1] & 0x0F
            expected = struct.unpack(">I", h[off:off + 4])[0] & 0x7FFFFFFF
            if code == f"{expected % 1_000_000:06d}":
                return True
        return False

    @staticmethod
    def generate_backup_codes(count: int = 8) -> list[str]:
        """Generate single-use backup codes."""
        return [secrets.token_hex(4).upper() for _ in range(count)]


def exercise_2_totp_integration():
    """Demonstrate TOTP 2FA setup and verification."""
    totp = TOTPManager()

    # Setup 2FA
    secret = totp.generate_secret()
    backup_codes = totp.generate_backup_codes()
    qr_uri = f"otpauth://totp/MyApp:alice?secret={secret}&issuer=MyApp"

    print("2FA Setup:")
    print(f"  Secret: {secret}")
    print(f"  QR URI: {qr_uri}")
    print(f"  Backup codes: {backup_codes[:3]}...")

    # Generate and verify current code
    current_code = totp.generate_totp(secret)
    print(f"\nCurrent TOTP code: {current_code}")
    print(f"Verification: {totp.verify_totp(secret, current_code)}")

    # Wrong code
    print(f"Wrong code check: {totp.verify_totp(secret, '000000')}")


# ---------------------------------------------------------------------------
# Exercise 3: JWT Security Audit
# ---------------------------------------------------------------------------

def exercise_3_jwt_security_audit():
    """
    Identify and explain all security issues in the vulnerable JWT code.
    """
    issues = [
        {
            "issue": 'SECRET = "mysecret"',
            "problem": "Hardcoded, short, predictable secret. Can be brute-forced.",
            "fix": "Use os.environ['JWT_SECRET'] with a 256-bit random value.",
        },
        {
            "issue": '"password": get_user_password(user_id)',
            "problem": "Password stored in JWT payload! JWTs are base64-encoded, "
                       "not encrypted. Anyone can read the payload.",
            "fix": "Never put sensitive data in JWT claims. Store only user_id and role.",
        },
        {
            "issue": 'jwt.encode(payload, SECRET) — no algorithm specified',
            "problem": "Defaults may include 'none' algorithm, allowing token forgery.",
            "fix": "Always specify algorithm: jwt.encode(payload, SECRET, algorithm='HS256')",
        },
        {
            "issue": 'algorithms=["HS256", "none"]',
            "problem": "Allowing 'none' algorithm means an attacker can create unsigned "
                       "tokens that will be accepted as valid.",
            "fix": "Only allow specific algorithms: algorithms=['HS256']",
        },
        {
            "issue": "except: (bare except)",
            "problem": "Catches ALL exceptions silently, hiding programming errors "
                       "and making debugging impossible.",
            "fix": "Catch specific exceptions: except (jwt.ExpiredSignatureError, "
                   "jwt.InvalidTokenError) as e:",
        },
        {
            "issue": 'if payload.get("admin")',
            "problem": "Authorization based on JWT claim that user controls. "
                       "An attacker can forge a token with admin=True if they know "
                       "the weak secret.",
            "fix": "Look up roles from the database using the user_id from the token. "
                   "Never trust client-supplied role claims alone.",
        },
        {
            "issue": "No expiration (exp) claim",
            "problem": "Tokens never expire, giving attackers unlimited time to use "
                       "a stolen token.",
            "fix": "Add 'exp': datetime.utcnow() + timedelta(minutes=15)",
        },
        {
            "issue": "No token revocation mechanism",
            "problem": "Even after logout or password change, old tokens remain valid.",
            "fix": "Use a token blacklist (Redis) or short-lived tokens with refresh rotation.",
        },
    ]

    print("JWT Security Audit")
    print("=" * 60)
    for i, item in enumerate(issues, 1):
        print(f"\nIssue {i}: {item['issue']}")
        print(f"  Problem: {item['problem']}")
        print(f"  Fix:     {item['fix']}")

    # Corrected implementation
    print("\n\n--- Corrected JWT Implementation ---")
    corrected = '''
import jwt
import os
import datetime

SECRET = os.environ["JWT_SECRET"]  # 256-bit random, from env

def create_token(user_id: int) -> str:
    payload = {
        "sub": user_id,
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=15),
        "jti": secrets.token_urlsafe(16),  # Unique token ID for revocation
    }
    return jwt.encode(payload, SECRET, algorithm="HS256")

def verify_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, SECRET, algorithms=["HS256"])
        # Check if token is blacklisted
        # if is_blacklisted(payload["jti"]): return None
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def protected_route(token: str):
    payload = verify_token(token)
    if not payload:
        return "Unauthorized", 401
    # Look up role from database, NOT from token
    user = db.get_user(payload["sub"])
    if user and user.role == "admin":
        return admin_dashboard()
    return user_dashboard(payload["sub"])
'''
    print(corrected)


# ---------------------------------------------------------------------------
# Exercise 4: OAuth 2.0 with PKCE
# ---------------------------------------------------------------------------

class OAuthClientPKCE:
    """OAuth 2.0 Authorization Code flow with PKCE."""

    def __init__(self, client_id: str, auth_url: str,
                 token_url: str, redirect_uri: str):
        self.client_id = client_id
        self.auth_url = auth_url
        self.token_url = token_url
        self.redirect_uri = redirect_uri
        self._state: Optional[str] = None
        self._code_verifier: Optional[str] = None

    def _generate_code_verifier(self) -> str:
        """Generate a PKCE code_verifier (43-128 chars, URL-safe)."""
        return secrets.token_urlsafe(32)

    def _generate_code_challenge(self, verifier: str) -> str:
        """Generate a PKCE code_challenge (S256)."""
        import base64
        digest = hashlib.sha256(verifier.encode()).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

    def start_auth_flow(self) -> str:
        """Generate the authorization URL with PKCE and state."""
        self._state = secrets.token_urlsafe(32)
        self._code_verifier = self._generate_code_verifier()
        code_challenge = self._generate_code_challenge(self._code_verifier)

        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "openid profile email",
            "state": self._state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }

        return f"{self.auth_url}?{urlencode(params)}"

    def handle_callback(self, callback_url: str) -> dict:
        """Handle the OAuth callback, verifying state and exchanging code."""
        parsed = urlparse(callback_url)
        params = parse_qs(parsed.query)

        # Verify state parameter (CSRF protection)
        received_state = params.get("state", [None])[0]
        if not hmac.compare_digest(received_state or "", self._state or ""):
            return {"error": "State parameter mismatch (possible CSRF attack)"}

        # Check for error response
        if "error" in params:
            return {"error": params["error"][0]}

        code = params.get("code", [None])[0]
        if not code:
            return {"error": "No authorization code received"}

        # Exchange code for tokens (simulated — in real code, use httpx/requests)
        token_request = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "code_verifier": self._code_verifier,
        }

        # Simulated response
        return {
            "access_token": secrets.token_urlsafe(32),
            "refresh_token": secrets.token_urlsafe(64),
            "expires_in": 3600,
            "token_type": "Bearer",
            "note": "Token exchange payload would be sent to: " + self.token_url,
            "request_body": token_request,
        }

    def refresh_access_token(self, refresh_token: str) -> dict:
        """Refresh an expired access token (simulated)."""
        return {
            "access_token": secrets.token_urlsafe(32),
            "refresh_token": secrets.token_urlsafe(64),  # Rotation
            "expires_in": 3600,
            "token_type": "Bearer",
        }


def exercise_4_oauth_pkce():
    """Demonstrate OAuth 2.0 with PKCE."""
    client = OAuthClientPKCE(
        client_id="my-app-id",
        auth_url="https://auth.example.com/authorize",
        token_url="https://auth.example.com/token",
        redirect_uri="https://myapp.com/callback",
    )

    # Start auth flow
    auth_url = client.start_auth_flow()
    print(f"Authorization URL:\n  {auth_url[:100]}...")

    # Simulate callback
    callback = f"https://myapp.com/callback?code=abc123&state={client._state}"
    tokens = client.handle_callback(callback)
    print(f"\nTokens received: {list(tokens.keys())}")

    # Refresh
    refreshed = client.refresh_access_token(tokens.get("refresh_token", ""))
    print(f"Refreshed tokens: {list(refreshed.keys())}")


# ---------------------------------------------------------------------------
# Exercise 5: Password Reset Security Review
# ---------------------------------------------------------------------------

def exercise_5_password_reset_review():
    """
    Identify all security vulnerabilities in the password reset code
    and provide the corrected version.
    """
    vulnerabilities = [
        {
            "issue": 'return "Email not found", 404',
            "problem": "User enumeration — attacker can discover which emails "
                       "are registered by checking for 404 vs 200 responses.",
            "fix": "Always return the same response regardless of whether email exists.",
        },
        {
            "issue": "random.choices(string.digits, k=4)",
            "problem": "4-digit code has only 10,000 possibilities — trivially "
                       "brute-forced. Also uses random module (not cryptographic).",
            "fix": "Use secrets.token_urlsafe(32) for a 256-bit token.",
        },
        {
            "issue": "reset_codes[email] = code (no expiration)",
            "problem": "Reset codes never expire. An old code remains valid forever.",
            "fix": "Store expiration timestamp; reject codes older than 30 minutes.",
        },
        {
            "issue": "reset_codes[email] (stored in plaintext)",
            "problem": "If server memory is compromised, all reset tokens are exposed.",
            "fix": "Store only the hash of the token; compare with hash on verification.",
        },
        {
            "issue": "reset_codes.get(email) == code (timing vulnerable)",
            "problem": "String comparison leaks information via timing side-channel.",
            "fix": "Use hmac.compare_digest() for constant-time comparison.",
        },
        {
            "issue": "user.password = new_password (plaintext storage)",
            "problem": "Password stored as plaintext, not hashed.",
            "fix": "Hash with argon2id/bcrypt before storing.",
        },
        {
            "issue": "No rate limiting on /reset endpoint",
            "problem": "Attacker can brute-force the 4-digit code with 10K requests.",
            "fix": "Rate-limit to 3-5 attempts per email per hour.",
        },
        {
            "issue": "Code not invalidated after use",
            "problem": "Same code can be reused multiple times.",
            "fix": "Delete the code from storage immediately after successful use.",
        },
        {
            "issue": "No password validation on new password",
            "problem": "User can set a weak password during reset.",
            "fix": "Apply the same password strength validation as registration.",
        },
    ]

    print("Password Reset Security Review")
    print("=" * 60)
    for i, v in enumerate(vulnerabilities, 1):
        print(f"\n{i}. Code: {v['issue']}")
        print(f"   Problem: {v['problem']}")
        print(f"   Fix:     {v['fix']}")

    print("\n\n--- Corrected Implementation ---")
    corrected = '''
import secrets
import hashlib
import hmac as _hmac
import time

reset_tokens = {}  # email -> {"hash": token_hash, "expires": timestamp}

@app.route("/forgot", methods=["POST"])
@rate_limit("3/hour")
def forgot_password():
    email = request.form.get("email", "").strip().lower()
    # Always return same response (prevent enumeration)
    user = db.find_user(email=email)
    if user:
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        reset_tokens[email] = {
            "hash": token_hash,
            "expires": time.time() + 1800,  # 30 minutes
        }
        send_email(email, f"Reset link: https://app.com/reset?token={token}&email={email}")
    return "If the email exists, a reset link has been sent.", 200

@app.route("/reset", methods=["POST"])
@rate_limit("5/hour")
def reset_password():
    email = request.form.get("email", "").strip().lower()
    token = request.form.get("token", "")
    new_password = request.form.get("password", "")

    stored = reset_tokens.get(email)
    if not stored or stored["expires"] < time.time():
        return "Invalid or expired reset link", 400

    token_hash = hashlib.sha256(token.encode()).hexdigest()
    if not _hmac.compare_digest(stored["hash"], token_hash):
        return "Invalid or expired reset link", 400

    # Validate password strength
    valid, msg = validate_password(new_password)
    if not valid:
        return msg, 400

    # Update password (hashed) and invalidate token
    user = db.find_user(email=email)
    user.password_hash = password_hasher.hash(new_password)
    db.save(user)
    del reset_tokens[email]
    # Invalidate all existing sessions
    db.invalidate_sessions(user.id)
    return "Password updated", 200
'''
    print(corrected)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: Secure Password Storage")
    print("=" * 70)
    exercise_1_secure_password_storage()

    print("\n" + "=" * 70)
    print("Exercise 2: TOTP Integration")
    print("=" * 70)
    exercise_2_totp_integration()

    print("\n" + "=" * 70)
    print("Exercise 3: JWT Security Audit")
    print("=" * 70)
    exercise_3_jwt_security_audit()

    print("\n" + "=" * 70)
    print("Exercise 4: OAuth 2.0 with PKCE")
    print("=" * 70)
    exercise_4_oauth_pkce()

    print("\n" + "=" * 70)
    print("Exercise 5: Password Reset Security Review")
    print("=" * 70)
    exercise_5_password_reset_review()
