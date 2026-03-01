# Exercise: Authentication Patterns
# Practice with JWT, session management, and OAuth2 concepts.

import hashlib
import hmac
import json
import base64
import time


# Exercise 1: JWT Implementation (simplified)
# Implement JWT creation and verification without external libraries.
# Use HMAC-SHA256 for signing.

def create_jwt(payload: dict, secret: str, expires_in: int = 3600) -> str:
    """Create a JWT token.

    Args:
        payload: Claims dict (e.g., {"sub": "user123", "role": "admin"})
        secret: HMAC secret key
        expires_in: Token TTL in seconds

    Returns: "header.payload.signature" string
    """
    # TODO: Implement
    # 1. Create header: {"alg": "HS256", "typ": "JWT"}
    # 2. Add "iat" and "exp" to payload
    # 3. Base64url encode header and payload
    # 4. Sign with HMAC-SHA256
    # 5. Return "header.payload.signature"
    pass


def verify_jwt(token: str, secret: str) -> dict | None:
    """Verify and decode a JWT token.

    Returns: payload dict if valid, None if invalid/expired.
    """
    # TODO: Implement
    # 1. Split token into parts
    # 2. Verify signature
    # 3. Check expiration
    # 4. Return decoded payload
    pass


# Test
# token = create_jwt({"sub": "alice", "role": "admin"}, "my-secret")
# payload = verify_jwt(token, "my-secret")
# assert payload["sub"] == "alice"


# Exercise 2: Password Hashing
# Implement password hashing with salt (simplified version).

def hash_password(password: str) -> str:
    """Hash a password with a random salt.

    Returns: "salt:hash" string
    """
    # TODO: Implement using hashlib.pbkdf2_hmac
    pass


def verify_password(password: str, stored: str) -> bool:
    """Verify a password against a stored hash.

    Args:
        password: Plain text password
        stored: "salt:hash" string from hash_password
    """
    # TODO: Implement
    pass


# Exercise 3: Session Store
# Implement an in-memory session store with expiration.

class SessionStore:
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self.sessions = {}

    def create(self, user_id: str, data: dict = None) -> str:
        """Create a new session. Returns session ID."""
        # TODO: Generate unique session ID, store with expiration
        pass

    def get(self, session_id: str) -> dict | None:
        """Get session data. Returns None if expired/missing."""
        # TODO: Check expiration, return data
        pass

    def destroy(self, session_id: str) -> bool:
        """Destroy a session. Returns True if existed."""
        # TODO: Implement
        pass

    def cleanup(self) -> int:
        """Remove all expired sessions. Returns count removed."""
        # TODO: Implement
        pass


# Exercise 4: API Key Manager
# Implement API key generation, storage, and validation.

class APIKeyManager:
    def __init__(self):
        self.keys = {}  # key_hash -> {name, created_at, last_used, scopes}

    def create_key(self, name: str, scopes: list[str]) -> str:
        """Generate a new API key. Returns the key (shown only once)."""
        # TODO: Generate secure random key, store its hash
        pass

    def validate_key(self, key: str) -> dict | None:
        """Validate an API key. Returns key metadata or None."""
        # TODO: Hash the key, look up, update last_used
        pass

    def has_scope(self, key: str, scope: str) -> bool:
        """Check if a key has a specific scope."""
        # TODO: Implement
        pass

    def revoke_key(self, key: str) -> bool:
        """Revoke an API key."""
        # TODO: Implement
        pass


if __name__ == "__main__":
    print("Authentication Patterns Exercise")
    print("Implement each class/function and test manually.")
