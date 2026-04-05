#!/usr/bin/env python3
"""Example: Secrets Management — Vault Patterns, Rotation & Encryption

Demonstrates secrets management fundamentals: envelope encryption, secret
rotation, lease-based access, audit logging, and a minimal Vault-like
key-value secrets engine.
Related lesson: 15_Secrets_Management.md
"""

# =============================================================================
# WHY SECRETS MANAGEMENT?
# Hard-coded passwords, API keys, and certificates in code or env vars are a
# top security risk. A secrets manager provides encryption at rest, access
# control, audit trails, automatic rotation, and dynamic credential generation.
# =============================================================================

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional


# =============================================================================
# 1. ENCRYPTION PRIMITIVES (SIMPLIFIED)
# =============================================================================

@dataclass
class EncryptionKey:
    """A symmetric encryption key for envelope encryption."""
    key_id: str
    key_bytes: bytes = field(default_factory=lambda: os.urandom(32))
    created_at: float = field(default_factory=time.time)
    version: int = 1

    def encrypt(self, plaintext: str) -> str:
        """XOR-based encryption (demo only — use AES-GCM in production)."""
        data = plaintext.encode()
        key_stream = (self.key_bytes * ((len(data) // len(self.key_bytes)) + 1))[:len(data)]
        ciphertext = bytes(a ^ b for a, b in zip(data, key_stream))
        return base64.b64encode(ciphertext).decode()

    def decrypt(self, ciphertext_b64: str) -> str:
        """Decrypt a base64-encoded ciphertext."""
        ciphertext = base64.b64decode(ciphertext_b64)
        key_stream = (self.key_bytes * ((len(ciphertext) // len(self.key_bytes)) + 1))[:len(ciphertext)]
        plaintext = bytes(a ^ b for a, b in zip(ciphertext, key_stream))
        return plaintext.decode()


# =============================================================================
# 2. SECRET STORE (VAULT-LIKE KV ENGINE)
# =============================================================================

@dataclass
class SecretVersion:
    """A versioned secret entry."""
    data: dict[str, str]
    version: int
    created_at: float = field(default_factory=time.time)
    destroyed: bool = False


@dataclass
class SecretEntry:
    """A secret with version history (KV v2 semantics)."""
    path: str
    versions: dict[int, SecretVersion] = field(default_factory=dict)
    max_versions: int = 10
    current_version: int = 0


@dataclass
class SecretsEngine:
    """Minimal Vault-like KV secrets engine with versioning and encryption."""
    name: str
    encryption_key: EncryptionKey = field(default_factory=lambda: EncryptionKey("master-key"))
    secrets: dict[str, SecretEntry] = field(default_factory=dict)
    audit_log: list[dict] = field(default_factory=list)

    def _audit(self, operation: str, path: str, accessor: str = "system") -> None:
        self.audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "path": path,
            "accessor": accessor,
        })

    def put(self, path: str, data: dict[str, str], accessor: str = "system") -> int:
        """Write a new version of a secret."""
        # Encrypt values before storage
        encrypted_data = {
            k: self.encryption_key.encrypt(v) for k, v in data.items()
        }
        if path not in self.secrets:
            self.secrets[path] = SecretEntry(path=path)
        entry = self.secrets[path]
        entry.current_version += 1
        entry.versions[entry.current_version] = SecretVersion(
            data=encrypted_data, version=entry.current_version
        )
        # Prune old versions
        while len(entry.versions) > entry.max_versions:
            oldest = min(entry.versions)
            del entry.versions[oldest]
        self._audit("write", path, accessor)
        return entry.current_version

    def get(self, path: str, version: int = 0, accessor: str = "system") -> Optional[dict[str, str]]:
        """Read a secret (latest version if version=0)."""
        entry = self.secrets.get(path)
        if not entry:
            self._audit("read_miss", path, accessor)
            return None
        v = version or entry.current_version
        sv = entry.versions.get(v)
        if not sv or sv.destroyed:
            self._audit("read_miss", path, accessor)
            return None
        # Decrypt values
        decrypted = {
            k: self.encryption_key.decrypt(v) for k, v in sv.data.items()
        }
        self._audit("read", path, accessor)
        return decrypted

    def delete(self, path: str, version: int, accessor: str = "system") -> bool:
        """Soft-delete a specific version."""
        entry = self.secrets.get(path)
        if not entry or version not in entry.versions:
            return False
        entry.versions[version].destroyed = True
        self._audit("delete", path, accessor)
        return True

    def list_secrets(self, prefix: str = "") -> list[str]:
        """List secret paths matching a prefix."""
        return [p for p in self.secrets if p.startswith(prefix)]


# =============================================================================
# 3. SECRET ROTATION
# =============================================================================

@dataclass
class RotationPolicy:
    """Defines how and when secrets should be rotated."""
    path: str
    ttl_seconds: int = 86400  # 24 hours
    last_rotated: float = 0.0
    rotation_count: int = 0

    def needs_rotation(self) -> bool:
        if self.last_rotated == 0:
            return True
        return (time.time() - self.last_rotated) >= self.ttl_seconds


def generate_password(length: int = 32) -> str:
    """Generate a cryptographically random password."""
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%"
    return "".join(alphabet[b % len(alphabet)] for b in os.urandom(length))


def rotate_secret(engine: SecretsEngine, policy: RotationPolicy) -> dict[str, Any]:
    """Rotate a secret if the TTL has expired."""
    if not policy.needs_rotation():
        return {"rotated": False, "reason": "TTL not expired"}
    new_password = generate_password()
    version = engine.put(
        policy.path,
        {"password": new_password, "rotated_at": datetime.now(timezone.utc).isoformat()},
        accessor="rotation-agent",
    )
    policy.last_rotated = time.time()
    policy.rotation_count += 1
    return {
        "rotated": True,
        "path": policy.path,
        "new_version": version,
        "rotation_count": policy.rotation_count,
    }


# =============================================================================
# 4. ACCESS POLICY (SIMPLIFIED RBAC)
# =============================================================================

@dataclass
class AccessPolicy:
    """Path-based access control for secrets."""
    name: str
    allowed_paths: list[str]  # Glob-like patterns
    capabilities: list[str] = field(default_factory=lambda: ["read"])

    def allows(self, path: str, operation: str) -> bool:
        if operation not in self.capabilities:
            return False
        for pattern in self.allowed_paths:
            if pattern.endswith("*"):
                if path.startswith(pattern[:-1]):
                    return True
            elif path == pattern:
                return True
        return False


# =============================================================================
# 5. DEMO
# =============================================================================

if __name__ == "__main__":
    # --- Secrets Engine ---
    print("=" * 60)
    print("Secrets Engine (KV v2)")
    print("=" * 60)
    engine = SecretsEngine(name="kv")

    # Write secrets
    v1 = engine.put("database/prod", {"username": "admin", "password": "s3cret!"})
    v2 = engine.put("database/prod", {"username": "admin", "password": "n3w-s3cret!"})
    engine.put("api/stripe", {"api_key": "sk_live_abc123"})
    print(f"  database/prod: v{v1}, v{v2}")
    print(f"  Secrets stored: {engine.list_secrets()}")

    # Read latest
    secret = engine.get("database/prod", accessor="app-server")
    print(f"  Read latest: {secret}")

    # Read specific version
    old = engine.get("database/prod", version=1)
    print(f"  Read v1: {old}")

    # --- Secret Rotation ---
    print(f"\n{'=' * 60}")
    print("Secret Rotation")
    print("=" * 60)
    policy = RotationPolicy(path="database/prod", ttl_seconds=0)  # Force rotation
    for i in range(3):
        result = rotate_secret(engine, policy)
        policy.last_rotated = 0  # Force next rotation for demo
        print(f"  Rotation {i+1}: version={result.get('new_version')}")

    # --- Access Policy ---
    print(f"\n{'=' * 60}")
    print("Access Policy (RBAC)")
    print("=" * 60)
    dev_policy = AccessPolicy(
        name="dev-readonly",
        allowed_paths=["database/*"],
        capabilities=["read"],
    )
    admin_policy = AccessPolicy(
        name="admin-full",
        allowed_paths=["*"],
        capabilities=["read", "write", "delete"],
    )
    checks = [
        ("dev-readonly", "database/prod", "read"),
        ("dev-readonly", "database/prod", "write"),
        ("dev-readonly", "api/stripe", "read"),
        ("admin-full", "api/stripe", "delete"),
    ]
    for policy_name, path, op in checks:
        p = dev_policy if policy_name == "dev-readonly" else admin_policy
        allowed = p.allows(path, op)
        status = "ALLOW" if allowed else "DENY"
        print(f"  [{status}] {policy_name}: {op} {path}")

    # --- Audit Log ---
    print(f"\n{'=' * 60}")
    print("Audit Log (last 5 entries)")
    print("=" * 60)
    for entry in engine.audit_log[-5:]:
        print(f"  {entry['timestamp']} | {entry['operation']:10s} | "
              f"{entry['path']} | by {entry['accessor']}")
