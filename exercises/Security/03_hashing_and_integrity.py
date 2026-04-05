"""
Exercise Solutions: Hashing and Data Integrity
===============================================
Lesson 03 from Security topic.

Covers hash comparison, password hashing defense, Merkle trees,
HMAC-based API authentication, content-addressable storage,
and timing attack prevention.

Dependencies:
    pip install cryptography bcrypt argon2-cffi
"""

import hashlib
import hmac
import os
import time
import json
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Exercise 1: Hash Explorer
# ---------------------------------------------------------------------------

def exercise_1_hash_explorer():
    """
    Compute multiple hash algorithms on sample data and compare speeds.
    In a real scenario this would operate on files; here we use synthetic data.
    """
    # Create sample data (simulating a 1 MB file for speed comparison)
    sample_data = os.urandom(1_000_000)

    algorithms = {
        "SHA-256": lambda d: hashlib.sha256(d).hexdigest(),
        "SHA-3-256": lambda d: hashlib.sha3_256(d).hexdigest(),
        "BLAKE2b-256": lambda d: hashlib.blake2b(d, digest_size=32).hexdigest(),
    }

    # Optional: BLAKE3 if available
    try:
        import blake3
        algorithms["BLAKE3"] = lambda d: blake3.blake3(d).hexdigest()
    except ImportError:
        print("  (blake3 not installed, skipping BLAKE3)")

    print("Hash values for 1 MB random data:")
    results = {}
    for name, hash_fn in algorithms.items():
        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            digest = hash_fn(sample_data)
        elapsed = (time.perf_counter() - start) / iterations * 1000
        results[name] = {"digest": digest[:32] + "...", "time_ms": elapsed}
        print(f"  {name:12s}: {digest[:32]}...  ({elapsed:.3f} ms)")

    # Integrity verification demo
    print("\nIntegrity verification:")
    original_hash = hashlib.sha256(sample_data).hexdigest()
    verify_hash = hashlib.sha256(sample_data).hexdigest()
    print(f"  Original hash matches recomputed hash: {original_hash == verify_hash}")

    # Tampered data
    tampered = bytearray(sample_data)
    tampered[0] ^= 0x01  # Flip one bit
    tampered_hash = hashlib.sha256(bytes(tampered)).hexdigest()
    print(f"  Tampered hash matches original:        {original_hash == tampered_hash}")

    return results


# ---------------------------------------------------------------------------
# Exercise 2: Password Cracker Defense
# ---------------------------------------------------------------------------

def exercise_2_password_cracker_defense():
    """
    Simulate password hashing with different methods and compare
    resistance to dictionary attacks.

    We use a small scale for demonstration (10 accounts, 50 passwords).
    """
    import secrets

    # Small word list for demonstration
    common_passwords = [
        "password", "123456", "qwerty", "letmein", "admin",
        "welcome", "monkey", "dragon", "master", "login",
        "abc123", "football", "shadow", "sunshine", "trustno1",
        "iloveyou", "batman", "passw0rd", "hello", "charlie",
    ]

    # Create 10 user accounts (some with common passwords, some with strong ones)
    users = {}
    for i in range(10):
        if i < 5:
            # Weak password from common list
            password = common_passwords[i]
        else:
            # Strong random password
            password = secrets.token_urlsafe(16)
        users[f"user_{i}"] = password

    # Method 1: Plain SHA-256 (no salt) -- INSECURE
    print("Method 1: SHA-256 (no salt) -- INSECURE")
    sha256_hashes = {}
    for user, pw in users.items():
        sha256_hashes[user] = hashlib.sha256(pw.encode()).hexdigest()

    # "Crack" by comparing dictionary hashes
    precomputed = {hashlib.sha256(pw.encode()).hexdigest(): pw
                   for pw in common_passwords}
    cracked_sha256 = 0
    for user, h in sha256_hashes.items():
        if h in precomputed:
            cracked_sha256 += 1
    print(f"  Cracked: {cracked_sha256}/{len(users)} "
          f"({cracked_sha256/len(users)*100:.0f}%)")

    # Method 2: SHA-256 with unique salt
    print("\nMethod 2: SHA-256 with unique salt -- still not ideal for passwords")
    salted_hashes = {}
    for user, pw in users.items():
        salt = os.urandom(16)
        h = hashlib.sha256(salt + pw.encode()).hexdigest()
        salted_hashes[user] = (salt, h)

    # Must re-hash each dictionary word per user (much slower)
    cracked_salted = 0
    start = time.perf_counter()
    for user, (salt, stored_hash) in salted_hashes.items():
        for guess in common_passwords:
            if hashlib.sha256(salt + guess.encode()).hexdigest() == stored_hash:
                cracked_salted += 1
                break
    elapsed = time.perf_counter() - start
    print(f"  Cracked: {cracked_salted}/{len(users)} in {elapsed:.4f}s")

    # Method 3: bcrypt (cost=10) -- RECOMMENDED
    try:
        import bcrypt
        print("\nMethod 3: bcrypt (cost=10) -- RECOMMENDED")
        bcrypt_hashes = {}
        for user, pw in users.items():
            bcrypt_hashes[user] = bcrypt.hashpw(
                pw.encode(), bcrypt.gensalt(rounds=10)
            )

        cracked_bcrypt = 0
        start = time.perf_counter()
        for user, stored_hash in bcrypt_hashes.items():
            for guess in common_passwords[:5]:  # Limit for speed
                if bcrypt.checkpw(guess.encode(), stored_hash):
                    cracked_bcrypt += 1
                    break
        elapsed = time.perf_counter() - start
        print(f"  Checked 5 guesses/user in {elapsed:.2f}s "
              f"(~{elapsed/len(users)/5*1000:.0f} ms/check)")
        print(f"  Cracked: {cracked_bcrypt}/{len(users)} (with only 5 guesses each)")
    except ImportError:
        print("\n  (bcrypt not installed, skipping)")

    # Summary
    print("\nSummary:")
    print("  SHA-256 (no salt): Instant precomputed attack")
    print("  SHA-256 (salted):  Per-user attack, but still fast (GPU-friendly)")
    print("  bcrypt/Argon2:     Deliberately slow, resistant to GPU attacks")


# ---------------------------------------------------------------------------
# Exercise 3: Merkle Tree File Verifier
# ---------------------------------------------------------------------------

class MerkleTree:
    """A Merkle tree for file integrity verification."""

    def __init__(self):
        self.leaves: list[tuple[str, bytes]] = []  # (path, hash)
        self.tree: list[list[bytes]] = []
        self.root: bytes = b""

    @staticmethod
    def _hash_pair(left: bytes, right: bytes) -> bytes:
        return hashlib.sha256(left + right).digest()

    @staticmethod
    def _hash_leaf(data: bytes) -> bytes:
        # Prefix with 0x00 to distinguish leaf from internal nodes
        return hashlib.sha256(b"\x00" + data).digest()

    @staticmethod
    def _hash_node(left: bytes, right: bytes) -> bytes:
        # Prefix with 0x01 for internal nodes
        return hashlib.sha256(b"\x01" + left + right).digest()

    def build(self, file_data: dict[str, bytes]):
        """
        Build the Merkle tree from a dictionary of {path: content}.
        Files are sorted by path for deterministic ordering.
        """
        sorted_paths = sorted(file_data.keys())
        self.leaves = [
            (path, self._hash_leaf(file_data[path]))
            for path in sorted_paths
        ]

        if not self.leaves:
            self.root = hashlib.sha256(b"empty").digest()
            return

        # Build tree bottom-up
        current_level = [h for _, h in self.leaves]
        self.tree = [current_level[:]]

        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                next_level.append(self._hash_node(left, right))
            self.tree.append(next_level[:])
            current_level = next_level

        self.root = current_level[0]

    def get_proof(self, index: int) -> list[tuple[str, bytes]]:
        """
        Generate a Merkle proof for the leaf at the given index.
        Returns a list of (direction, hash) pairs.
        """
        proof = []
        for level in self.tree[:-1]:  # Skip root level
            if index % 2 == 0:
                sibling_idx = index + 1
                direction = "right"
            else:
                sibling_idx = index - 1
                direction = "left"

            if sibling_idx < len(level):
                proof.append((direction, level[sibling_idx]))
            else:
                proof.append((direction, level[index]))  # Duplicate

            index //= 2

        return proof

    @staticmethod
    def verify_proof(leaf_hash: bytes, proof: list[tuple[str, bytes]],
                     expected_root: bytes) -> bool:
        """Verify a Merkle proof against an expected root hash."""
        current = leaf_hash
        for direction, sibling in proof:
            if direction == "right":
                current = MerkleTree._hash_node(current, sibling)
            else:
                current = MerkleTree._hash_node(sibling, current)
        return current == expected_root


def exercise_3_merkle_tree():
    """Build a Merkle tree and verify individual files."""
    files = {
        "src/main.py": b"print('hello')",
        "src/utils.py": b"def helper(): pass",
        "README.md": b"# My Project",
        "config.yaml": b"debug: false",
    }

    tree = MerkleTree()
    tree.build(files)
    print(f"Merkle root: {tree.root.hex()[:32]}...")
    print(f"Tree levels: {len(tree.tree)}")

    # Verify a specific file
    file_index = 0  # First file (sorted: README.md)
    proof = tree.get_proof(file_index)
    leaf_hash = tree.leaves[file_index][1]

    verified = MerkleTree.verify_proof(leaf_hash, proof, tree.root)
    print(f"\nVerification of '{tree.leaves[file_index][0]}': {verified}")
    print(f"  Proof length: {len(proof)} hashes (O(log n))")

    # Tampered file
    tampered_hash = MerkleTree._hash_leaf(b"print('HACKED')")
    tampered_result = MerkleTree.verify_proof(tampered_hash, proof, tree.root)
    print(f"  Tampered file verification: {tampered_result}")


# ---------------------------------------------------------------------------
# Exercise 4: HMAC-based API Authentication
# ---------------------------------------------------------------------------

class HMACAuthServer:
    """Server-side HMAC-based API authentication."""

    def __init__(self):
        self.clients: dict[str, bytes] = {}  # api_key -> secret
        self.nonce_cache: set[str] = set()
        self.max_clock_skew = 300  # 5 minutes

    def register_client(self, api_key: str) -> bytes:
        """Issue an API key and secret to a client."""
        secret = os.urandom(32)
        self.clients[api_key] = secret
        return secret

    def verify_request(self, api_key: str, method: str, path: str,
                       body: bytes, timestamp: int, nonce: str,
                       signature: str) -> tuple[bool, str]:
        """Verify an HMAC-signed API request."""
        # Check API key exists
        if api_key not in self.clients:
            return False, "Unknown API key"

        # Check timestamp freshness
        now = int(time.time())
        if abs(now - timestamp) > self.max_clock_skew:
            return False, f"Timestamp too old/new (skew: {abs(now - timestamp)}s)"

        # Check nonce uniqueness (replay protection)
        if nonce in self.nonce_cache:
            return False, "Nonce already used (replay attack)"

        # Compute expected signature
        body_hash = hashlib.sha256(body).hexdigest()
        message = f"{method}\n{path}\n{timestamp}\n{body_hash}\n{nonce}"
        expected = hmac.new(
            self.clients[api_key],
            message.encode(),
            hashlib.sha256,
        ).hexdigest()

        # Constant-time comparison
        if not hmac.compare_digest(signature, expected):
            return False, "Invalid signature"

        # Accept and cache nonce
        self.nonce_cache.add(nonce)
        return True, "Authenticated"


class HMACAuthClient:
    """Client-side HMAC request signing."""

    def __init__(self, api_key: str, secret: bytes):
        self.api_key = api_key
        self.secret = secret

    def sign_request(self, method: str, path: str, body: bytes = b"") -> dict:
        """Sign an API request and return headers."""
        timestamp = int(time.time())
        nonce = os.urandom(16).hex()
        body_hash = hashlib.sha256(body).hexdigest()
        message = f"{method}\n{path}\n{timestamp}\n{body_hash}\n{nonce}"
        signature = hmac.new(
            self.secret, message.encode(), hashlib.sha256
        ).hexdigest()

        return {
            "X-API-Key": self.api_key,
            "X-Timestamp": str(timestamp),
            "X-Nonce": nonce,
            "X-Signature": signature,
        }


def exercise_4_hmac_api_auth():
    """Demonstrate HMAC-based API authentication."""
    server = HMACAuthServer()

    # Register a client
    secret = server.register_client("client-001")
    client = HMACAuthClient("client-001", secret)

    # Sign and verify a request
    headers = client.sign_request("POST", "/api/orders", b'{"item": "widget"}')
    ok, msg = server.verify_request(
        api_key=headers["X-API-Key"],
        method="POST",
        path="/api/orders",
        body=b'{"item": "widget"}',
        timestamp=int(headers["X-Timestamp"]),
        nonce=headers["X-Nonce"],
        signature=headers["X-Signature"],
    )
    print(f"Valid request:   {ok} ({msg})")

    # Replay attack
    ok, msg = server.verify_request(
        api_key=headers["X-API-Key"],
        method="POST",
        path="/api/orders",
        body=b'{"item": "widget"}',
        timestamp=int(headers["X-Timestamp"]),
        nonce=headers["X-Nonce"],
        signature=headers["X-Signature"],
    )
    print(f"Replay attempt:  {ok} ({msg})")

    # Tampered body
    headers2 = client.sign_request("POST", "/api/orders", b'{"item": "widget"}')
    ok, msg = server.verify_request(
        api_key=headers2["X-API-Key"],
        method="POST",
        path="/api/orders",
        body=b'{"item": "TAMPERED"}',  # Different body
        timestamp=int(headers2["X-Timestamp"]),
        nonce=headers2["X-Nonce"],
        signature=headers2["X-Signature"],
    )
    print(f"Tampered body:   {ok} ({msg})")


# ---------------------------------------------------------------------------
# Exercise 5: Content-Addressable File Sync
# ---------------------------------------------------------------------------

class ContentAddressableStore:
    """A simple content-addressable store using SHA-256."""

    def __init__(self, name: str):
        self.name = name
        self.store: dict[str, bytes] = {}  # hash -> content

    def add(self, content: bytes) -> str:
        """Add content and return its hash key."""
        key = hashlib.sha256(content).hexdigest()
        self.store[key] = content
        return key

    def get(self, key: str) -> Optional[bytes]:
        return self.store.get(key)

    def has(self, key: str) -> bool:
        return key in self.store

    def keys(self) -> set[str]:
        return set(self.store.keys())


def sync_stores(source: ContentAddressableStore,
                dest: ContentAddressableStore) -> int:
    """
    Sync from source to dest by transferring only missing blocks.
    Returns the number of blocks transferred.
    """
    source_keys = source.keys()
    dest_keys = dest.keys()
    missing = source_keys - dest_keys

    transferred = 0
    for key in missing:
        content = source.get(key)
        if content is not None:
            # Verify integrity after "transfer"
            received_key = hashlib.sha256(content).hexdigest()
            assert received_key == key, "Integrity check failed!"
            dest.store[key] = content
            transferred += 1

    return transferred


def exercise_5_content_addressable_sync():
    """Demonstrate content-addressable file synchronization."""
    store_a = ContentAddressableStore("Node A")
    store_b = ContentAddressableStore("Node B")

    # Node A has some files
    store_a.add(b"File 1 content")
    store_a.add(b"File 2 content")
    store_a.add(b"Shared file content")

    # Node B has some files, including one shared with A
    store_b.add(b"Shared file content")  # Same content, same hash
    store_b.add(b"File 3 content")

    print(f"Node A has {len(store_a.keys())} blocks")
    print(f"Node B has {len(store_b.keys())} blocks")
    print(f"Common blocks: {len(store_a.keys() & store_b.keys())}")

    transferred = sync_stores(store_a, store_b)
    print(f"\nSync A -> B: {transferred} blocks transferred")
    print(f"Node B now has {len(store_b.keys())} blocks")
    print(f"All A blocks in B: {store_a.keys().issubset(store_b.keys())}")


# ---------------------------------------------------------------------------
# Exercise 6: Timing Attack Lab (Defense)
# ---------------------------------------------------------------------------

def insecure_compare(a: str, b: str) -> bool:
    """INSECURE: String comparison that leaks timing information."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x != y:
            return False  # Short-circuits on first mismatch
    return True


def secure_compare(a: str, b: str) -> bool:
    """SECURE: Constant-time comparison using hmac.compare_digest."""
    return hmac.compare_digest(a.encode(), b.encode())


def exercise_6_timing_attack_defense():
    """
    Demonstrate the difference between insecure and secure comparison.

    NOTE: We only demonstrate the DEFENSE. We do not provide actual
    timing attack exploit code. The key lesson is: always use
    hmac.compare_digest() for comparing secrets.
    """
    secret_token = "a1b2c3d4e5f6g7h8"

    # Insecure comparison -- vulnerable to timing analysis
    print("Insecure comparison (== or character-by-character):")
    print(f"  Correct token:  {insecure_compare(secret_token, secret_token)}")
    print(f"  Wrong token:    {insecure_compare(secret_token, 'wrong-token-here')}")
    print("  WARNING: An attacker can measure response time differences")
    print("  to guess the token one character at a time.")

    # Secure comparison -- constant time
    print("\nSecure comparison (hmac.compare_digest):")
    print(f"  Correct token:  {secure_compare(secret_token, secret_token)}")
    print(f"  Wrong token:    {secure_compare(secret_token, 'wrong-token-here')}")
    print("  Both comparisons take the same amount of time regardless")
    print("  of how many characters match.")

    print("\nOther timing side channels to be aware of:")
    print("  - Cache timing attacks (e.g., on AES lookup tables)")
    print("  - Power analysis (measuring CPU power consumption)")
    print("  - Branch prediction side channels (Spectre/Meltdown)")
    print("  - Network response time differences in remote services")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: Hash Explorer")
    print("=" * 70)
    exercise_1_hash_explorer()

    print("\n" + "=" * 70)
    print("Exercise 2: Password Cracker Defense")
    print("=" * 70)
    exercise_2_password_cracker_defense()

    print("\n" + "=" * 70)
    print("Exercise 3: Merkle Tree File Verifier")
    print("=" * 70)
    exercise_3_merkle_tree()

    print("\n" + "=" * 70)
    print("Exercise 4: HMAC-based API Authentication")
    print("=" * 70)
    exercise_4_hmac_api_auth()

    print("\n" + "=" * 70)
    print("Exercise 5: Content-Addressable File Sync")
    print("=" * 70)
    exercise_5_content_addressable_sync()

    print("\n" + "=" * 70)
    print("Exercise 6: Timing Attack Defense")
    print("=" * 70)
    exercise_6_timing_attack_defense()
