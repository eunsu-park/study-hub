"""
CIA Triad Demonstration

Demonstrates:
- Confidentiality: encryption, access control
- Integrity: hashing, digital signatures, checksums
- Availability: redundancy, failover simulation

Theory:
- The CIA Triad is the foundational model for information security.
- Confidentiality: prevent unauthorized access to information.
  Methods: encryption (AES, RSA), access control lists (ACL),
  role-based access control (RBAC).
- Integrity: ensure data is not tampered with or corrupted.
  Methods: cryptographic hashes (SHA-256), HMACs, checksums,
  digital signatures.
- Availability: ensure systems and data are accessible when needed.
  Methods: redundancy, failover, load balancing, backups.

Adapted from Security Lesson 01.
"""

import hashlib
import hmac
import os
import base64
from dataclasses import dataclass, field


# ── Confidentiality ───────────────────────────────────────────────────

class SimpleXORCipher:
    """XOR cipher for educational purposes (NOT cryptographically secure)."""

    def __init__(self, key: bytes):
        self.key = key

    def encrypt(self, plaintext: bytes) -> bytes:
        return bytes(p ^ self.key[i % len(self.key)]
                     for i, p in enumerate(plaintext))

    def decrypt(self, ciphertext: bytes) -> bytes:
        return self.encrypt(ciphertext)  # XOR is its own inverse


@dataclass
class AccessControlEntry:
    subject: str
    resource: str
    permissions: set[str]


class AccessControlList:
    """Simple ACL-based access control."""

    def __init__(self):
        self.entries: list[AccessControlEntry] = []

    def grant(self, subject: str, resource: str,
              permissions: set[str]) -> None:
        for entry in self.entries:
            if entry.subject == subject and entry.resource == resource:
                entry.permissions |= permissions
                return
        self.entries.append(AccessControlEntry(subject, resource, permissions))

    def check(self, subject: str, resource: str, action: str) -> bool:
        for entry in self.entries:
            if entry.subject == subject and entry.resource == resource:
                return action in entry.permissions
        return False

    def revoke(self, subject: str, resource: str,
               permissions: set[str]) -> None:
        for entry in self.entries:
            if entry.subject == subject and entry.resource == resource:
                entry.permissions -= permissions


class RBAC:
    """Role-Based Access Control."""

    def __init__(self):
        self.roles: dict[str, set[str]] = {}          # role → permissions
        self.user_roles: dict[str, set[str]] = {}     # user → roles

    def create_role(self, role: str, permissions: set[str]) -> None:
        self.roles[role] = permissions

    def assign_role(self, user: str, role: str) -> None:
        self.user_roles.setdefault(user, set()).add(role)

    def check(self, user: str, permission: str) -> bool:
        for role in self.user_roles.get(user, set()):
            if permission in self.roles.get(role, set()):
                return True
        return False

    def get_permissions(self, user: str) -> set[str]:
        perms = set()
        for role in self.user_roles.get(user, set()):
            perms |= self.roles.get(role, set())
        return perms


# ── Integrity ─────────────────────────────────────────────────────────

def compute_checksum(data: bytes) -> str:
    """SHA-256 hash for integrity verification."""
    return hashlib.sha256(data).hexdigest()


def verify_checksum(data: bytes, expected: str) -> bool:
    return compute_checksum(data) == expected


def compute_hmac(data: bytes, key: bytes) -> str:
    """HMAC-SHA256 for authenticated integrity."""
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def verify_hmac(data: bytes, key: bytes, expected: str) -> bool:
    computed = compute_hmac(data, key)
    return hmac.compare_digest(computed, expected)


@dataclass
class IntegrityProtectedFile:
    """File with integrity metadata."""
    filename: str
    content: bytes
    checksum: str = ""
    hmac_tag: str = ""

    def sign(self, key: bytes) -> None:
        self.checksum = compute_checksum(self.content)
        self.hmac_tag = compute_hmac(self.content, key)

    def verify(self, key: bytes) -> dict[str, bool]:
        return {
            "checksum_valid": verify_checksum(self.content, self.checksum),
            "hmac_valid": verify_hmac(self.content, key, self.hmac_tag),
        }


# ── Availability ──────────────────────────────────────────────────────

@dataclass
class Server:
    name: str
    alive: bool = True
    data: dict = field(default_factory=dict)
    request_count: int = 0

    def handle_request(self, key: str) -> str | None:
        if not self.alive:
            raise ConnectionError(f"{self.name} is down")
        self.request_count += 1
        return self.data.get(key)


class RedundantCluster:
    """Simulates high-availability with failover."""

    def __init__(self, replicas: int = 3):
        self.servers = [Server(f"server-{i}") for i in range(replicas)]
        self.primary = 0

    def write(self, key: str, value: str) -> int:
        """Write to all replicas (synchronous replication)."""
        written = 0
        for server in self.servers:
            if server.alive:
                server.data[key] = value
                written += 1
        return written

    def read(self, key: str) -> tuple[str | None, str]:
        """Read with automatic failover."""
        # Try primary first
        order = [self.primary] + [i for i in range(len(self.servers))
                                   if i != self.primary]
        for idx in order:
            server = self.servers[idx]
            try:
                result = server.handle_request(key)
                return result, server.name
            except ConnectionError:
                continue

        return None, "ALL_DOWN"

    def kill(self, index: int) -> None:
        self.servers[index].alive = False

    def revive(self, index: int) -> None:
        self.servers[index].alive = True


# ── Demos ─────────────────────────────────────────────────────────────

def demo_confidentiality():
    print("=" * 60)
    print("CONFIDENTIALITY")
    print("=" * 60)

    # Encryption
    print("\n  1. Encryption (XOR cipher — educational only):")
    key = os.urandom(16)
    cipher = SimpleXORCipher(key)
    message = b"Secret: account balance is $50,000"
    encrypted = cipher.encrypt(message)
    decrypted = cipher.decrypt(encrypted)
    print(f"    Plaintext:  {message.decode()}")
    print(f"    Encrypted:  {base64.b64encode(encrypted).decode()[:40]}...")
    print(f"    Decrypted:  {decrypted.decode()}")
    print(f"    Match: {message == decrypted}")

    # ACL
    print("\n  2. Access Control List (ACL):")
    acl = AccessControlList()
    acl.grant("alice", "/data/reports", {"read", "write"})
    acl.grant("bob", "/data/reports", {"read"})
    acl.grant("bob", "/data/logs", {"read", "write"})

    checks = [
        ("alice", "/data/reports", "read"),
        ("alice", "/data/reports", "write"),
        ("bob", "/data/reports", "write"),
        ("bob", "/data/logs", "write"),
        ("charlie", "/data/reports", "read"),
    ]
    for subject, resource, action in checks:
        result = acl.check(subject, resource, action)
        status = "ALLOW" if result else "DENY"
        print(f"    {subject:>8} → {action:>5} {resource:<16} → {status}")

    # RBAC
    print("\n  3. Role-Based Access Control (RBAC):")
    rbac = RBAC()
    rbac.create_role("admin", {"read", "write", "delete", "manage_users"})
    rbac.create_role("editor", {"read", "write"})
    rbac.create_role("viewer", {"read"})

    rbac.assign_role("alice", "admin")
    rbac.assign_role("bob", "editor")
    rbac.assign_role("charlie", "viewer")

    for user in ["alice", "bob", "charlie"]:
        perms = rbac.get_permissions(user)
        can_delete = rbac.check(user, "delete")
        print(f"    {user:>8}: {sorted(perms)}"
              f"  can_delete={can_delete}")


def demo_integrity():
    print("\n" + "=" * 60)
    print("INTEGRITY")
    print("=" * 60)

    # Hash-based integrity
    print("\n  1. Checksum verification:")
    original = b"Transfer $1000 to account 12345"
    checksum = compute_checksum(original)
    print(f"    Original:  {original.decode()}")
    print(f"    SHA-256:   {checksum[:32]}...")
    print(f"    Valid:     {verify_checksum(original, checksum)}")

    tampered = b"Transfer $9999 to account 12345"
    print(f"    Tampered:  {tampered.decode()}")
    print(f"    Valid:     {verify_checksum(tampered, checksum)}")

    # HMAC
    print("\n  2. HMAC (authenticated integrity):")
    key = b"shared-secret-key"
    data = b"Important message"
    tag = compute_hmac(data, key)
    print(f"    Data:      {data.decode()}")
    print(f"    HMAC:      {tag[:32]}...")
    print(f"    Verify (correct key):  {verify_hmac(data, key, tag)}")
    print(f"    Verify (wrong key):    "
          f"{verify_hmac(data, b'wrong-key', tag)}")

    # File integrity
    print("\n  3. File integrity protection:")
    key = b"file-integrity-key"
    f = IntegrityProtectedFile("report.pdf", b"Annual report data...")
    f.sign(key)
    print(f"    File: {f.filename}")
    print(f"    Checksum: {f.checksum[:32]}...")
    result = f.verify(key)
    print(f"    Verification: {result}")

    # Simulate tampering
    f.content = b"TAMPERED report data..."
    result = f.verify(key)
    print(f"    After tampering: {result}")


def demo_availability():
    print("\n" + "=" * 60)
    print("AVAILABILITY")
    print("=" * 60)

    cluster = RedundantCluster(replicas=3)

    # Normal operation
    print("\n  1. Normal operation (3 replicas):")
    cluster.write("config", "v1.0")
    cluster.write("status", "active")
    value, server = cluster.read("config")
    print(f"    Write 'config'='v1.0' to all replicas")
    print(f"    Read 'config' → '{value}' (from {server})")

    # Single failure
    print("\n  2. Single server failure:")
    cluster.kill(0)
    print(f"    server-0 is DOWN")
    value, server = cluster.read("config")
    print(f"    Read 'config' → '{value}' (failover to {server})")
    written = cluster.write("config", "v2.0")
    print(f"    Write 'config'='v2.0' → replicated to {written}/3 servers")

    # Majority failure
    print("\n  3. Majority failure:")
    cluster.kill(1)
    print(f"    server-1 is DOWN (2 of 3 servers down)")
    value, server = cluster.read("config")
    print(f"    Read 'config' → '{value}' (from {server})")
    print(f"    System degraded but still available!")

    # Total failure
    print("\n  4. Total failure:")
    cluster.kill(2)
    print(f"    server-2 is DOWN (all servers down)")
    value, server = cluster.read("config")
    print(f"    Read 'config' → {value} (from {server})")
    print(f"    Availability LOST — no servers reachable")

    # Recovery
    print("\n  5. Recovery:")
    cluster.revive(1)
    cluster.revive(2)
    value, server = cluster.read("config")
    print(f"    Revived server-1 and server-2")
    print(f"    Read 'config' → '{value}' (from {server})")
    print(f"    Service restored!")


def demo_triad_summary():
    print("\n" + "=" * 60)
    print("CIA TRIAD SUMMARY")
    print("=" * 60)

    print(f"""
  ┌─────────────────────────────────────────────────┐
  │                 CIA Triad                        │
  │                                                 │
  │           Confidentiality                       │
  │           /            \\                        │
  │      Encryption    Access Control               │
  │      (AES, RSA)    (ACL, RBAC)                  │
  │                                                 │
  │     Integrity ──────────── Availability         │
  │     /        \\             /          \\         │
  │  Hashing   Signatures  Redundancy  Failover     │
  │  (SHA-256) (HMAC)      (Replicas)  (Backup)     │
  └─────────────────────────────────────────────────┘

  Threat vs Control:
    {'Threat':<25} {'Control':<20} {'Pillar':<15}
    {'-'*25} {'-'*20} {'-'*15}
    {'Eavesdropping':<25} {'Encryption':<20} {'Confidentiality':<15}
    {'Unauthorized access':<25} {'RBAC / ACL':<20} {'Confidentiality':<15}
    {'Data tampering':<25} {'Hashing / HMAC':<20} {'Integrity':<15}
    {'Man-in-the-middle':<25} {'Digital signatures':<20} {'Integrity':<15}
    {'Server crash':<25} {'Redundancy':<20} {'Availability':<15}
    {'DDoS attack':<25} {'Load balancing':<20} {'Availability':<15}

  Trade-offs:
    - Strong encryption (C) adds latency, reducing availability (A)
    - Multiple replicas (A) increase attack surface, risking C and I
    - Strict access control (C) can impede legitimate access (A)
    - Balance depends on threat model and business requirements""")


if __name__ == "__main__":
    demo_confidentiality()
    demo_integrity()
    demo_availability()
    demo_triad_summary()
