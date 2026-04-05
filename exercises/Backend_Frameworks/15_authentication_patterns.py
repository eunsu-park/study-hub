# Exercise: Authentication Patterns
# Practice with JWT tokens, password hashing, RBAC, and session management.

import json
import time
import hmac
import hashlib
import base64
import secrets
from typing import Optional


# Exercise 1: JWT Encoder/Decoder (simplified)
# Implement HS256 JWT without external libraries.

def jwt_encode(payload: dict, secret: str) -> str:
    """Create a JWT token (header.payload.signature).

    - Header: {"alg": "HS256", "typ": "JWT"}
    - Payload: add "iat" (issued-at, unix timestamp) to the given payload
    - Signature: HMAC-SHA256 of "header_b64.payload_b64" with secret
    - Use base64url encoding (no padding)
    """
    # TODO: Implement
    pass


def jwt_decode(token: str, secret: str) -> dict:
    """Verify and decode a JWT token.

    - Split token into 3 parts
    - Verify HMAC signature
    - Return payload dict
    - Raise ValueError if signature invalid or token malformed
    """
    # TODO: Implement
    pass


# Test
# token = jwt_encode({"sub": "alice", "role": "admin"}, "mysecret")
# parts = token.split(".")
# assert len(parts) == 3
# payload = jwt_decode(token, "mysecret")
# assert payload["sub"] == "alice"
# try: jwt_decode(token, "wrongsecret"); assert False
# except ValueError: pass


# Exercise 2: Password Hasher (PBKDF2)
class PasswordHasher:
    """Hash and verify passwords using PBKDF2-HMAC-SHA256."""

    def __init__(self, iterations: int = 100_000, salt_length: int = 16):
        self.iterations = iterations
        self.salt_length = salt_length

    def hash(self, password: str) -> str:
        """Return a string in format: "salt_hex$hash_hex"."""
        # TODO: Implement using hashlib.pbkdf2_hmac
        pass

    def verify(self, password: str, stored: str) -> bool:
        """Verify a password against a stored hash string.

        Parse salt from stored string, re-derive hash, compare securely.
        Use hmac.compare_digest for timing-safe comparison.
        """
        # TODO: Implement
        pass


# Test
# hasher = PasswordHasher(iterations=1000)  # low for testing
# hashed = hasher.hash("mypassword")
# assert "$" in hashed
# assert hasher.verify("mypassword", hashed) is True
# assert hasher.verify("wrongpassword", hashed) is False


# Exercise 3: Role-Based Access Control
class RBAC:
    """Simple role-based access control system."""

    def __init__(self):
        self._roles = {}       # role -> set of permissions
        self._user_roles = {}  # username -> set of roles

    def define_role(self, role: str, permissions: list[str]):
        """Define a role with its permissions."""
        # TODO: Implement
        pass

    def assign_role(self, username: str, role: str):
        """Assign a role to a user. Raise ValueError if role not defined."""
        # TODO: Implement
        pass

    def has_permission(self, username: str, permission: str) -> bool:
        """Check if user has permission through any of their roles."""
        # TODO: Implement
        pass

    def get_permissions(self, username: str) -> set[str]:
        """Get all permissions for a user (union of all role permissions)."""
        # TODO: Implement
        pass


# Test
# rbac = RBAC()
# rbac.define_role("editor", ["read", "write", "publish"])
# rbac.define_role("viewer", ["read"])
# rbac.assign_role("alice", "editor")
# rbac.assign_role("bob", "viewer")
# assert rbac.has_permission("alice", "publish") is True
# assert rbac.has_permission("bob", "publish") is False
# assert rbac.get_permissions("alice") == {"read", "write", "publish"}


# Exercise 4: Session Store
class SessionStore:
    """Server-side session management with expiration."""

    def __init__(self, max_age: int = 3600):
        self.max_age = max_age
        self._sessions = {}  # session_id -> {"data": dict, "created_at": float, "last_access": float}

    def create(self, data: dict) -> str:
        """Create a new session. Return the session ID (random token)."""
        # TODO: Implement
        pass

    def get(self, session_id: str) -> Optional[dict]:
        """Get session data. Return None if expired or not found.
        Update last_access on successful get.
        """
        # TODO: Implement
        pass

    def destroy(self, session_id: str) -> bool:
        """Delete a session. Return True if it existed."""
        # TODO: Implement
        pass

    def cleanup_expired(self) -> int:
        """Remove all expired sessions. Return count of removed sessions."""
        # TODO: Implement
        pass


# Test
# store = SessionStore(max_age=1)
# sid = store.create({"user": "alice"})
# assert store.get(sid) == {"user": "alice"}
# time.sleep(1.1)
# assert store.get(sid) is None


if __name__ == "__main__":
    print("Authentication Patterns Exercise")
    print("Implement each class/function and verify with the test cases.")
