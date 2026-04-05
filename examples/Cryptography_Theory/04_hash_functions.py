"""
Hash Functions, Birthday Attacks, HMAC, and Merkle Trees
========================================================
Covers the theory and practice of cryptographic hash functions:
collision resistance, the birthday paradox, message authentication,
and hash-based data structures.
"""

from __future__ import annotations
import hashlib
import hmac as hmac_stdlib
import math
import os
import random
import struct
import time


# ---------------------------------------------------------------------------
# Simple Educational Hash Function
# ---------------------------------------------------------------------------

# Why: Real hash functions (SHA-256) are opaque — you can't see the
# internal state evolve. This toy hash is intentionally weak (small output)
# so we can study collision behavior and the birthday attack directly.
def simple_hash(data: bytes, output_bits: int = 16) -> int:
    """A simple hash function with configurable output size.

    NOT cryptographically secure — designed for educational study.
    Uses a Merkle-Damgard-like structure with a compression function.
    """
    # Why: Merkle-Damgard is the construction used by MD5, SHA-1, SHA-2.
    # We process the input in fixed-size blocks, maintaining an internal
    # state (the "chaining value") that accumulates the hash.
    BLOCK_SIZE = 4
    mask = (1 << output_bits) - 1

    # Initialize state (IV)
    state = 0x5A3C & mask

    # Pad to block boundary
    padded = data + b"\x80"
    while len(padded) % BLOCK_SIZE != 0:
        padded += b"\x00"
    padded += struct.pack(">I", len(data))
    while len(padded) % BLOCK_SIZE != 0:
        padded += b"\x00"

    # Process blocks
    for i in range(0, len(padded), BLOCK_SIZE):
        block_val = int.from_bytes(padded[i : i + BLOCK_SIZE], "big") & mask
        # Why: The compression function mixes the block with the state.
        # We use XOR + multiplication + rotation — crude but illustrative.
        state ^= block_val
        state = ((state * 0x9E37) + 0xB5C1) & mask
        state ^= (state >> 5)
        state = (state * 0x27D4) & mask

    return state


# ---------------------------------------------------------------------------
# Birthday Attack Simulation
# ---------------------------------------------------------------------------

# Why: The birthday attack exploits the birthday paradox — in a hash with
# n-bit output, you expect a collision after ~2^(n/2) random hashes, NOT
# 2^n. For SHA-256 (256 bits), that's 2^128 — still infeasible. But for
# our 16-bit toy hash, it's only ~2^8 = 256 attempts. This is why hash
# output must be large enough.
def birthday_attack(output_bits: int = 16) -> tuple[int, bytes, bytes]:
    """Find a collision in simple_hash using the birthday attack.

    Returns (attempts, message1, message2) where both hash to the same value.
    """
    seen: dict[int, bytes] = {}
    attempts = 0

    while True:
        msg = random.getrandbits(64).to_bytes(8, "big")
        h = simple_hash(msg, output_bits)
        attempts += 1

        if h in seen and seen[h] != msg:
            return attempts, seen[h], msg

        seen[h] = msg


def birthday_probability(n: int, k: int) -> float:
    """Probability of at least one collision among k items in n buckets.

    Uses the exact formula: 1 - product(1 - i/n for i in range(k))
    """
    # Why: This is the mathematical basis of the birthday paradox.
    # For n=365 (days), k=23 people gives ~50% collision probability.
    # For hashes: n=2^bits, so k ~ sqrt(n) = 2^(bits/2) for 50%.
    prob_no_collision = 1.0
    for i in range(k):
        prob_no_collision *= (n - i) / n
        if prob_no_collision <= 0:
            return 1.0
    return 1.0 - prob_no_collision


# ---------------------------------------------------------------------------
# HMAC (Hash-based Message Authentication Code)
# ---------------------------------------------------------------------------

# Why: A hash alone doesn't provide authentication — an attacker who knows
# H(m) can compute H(m || extra) due to length-extension attacks on
# Merkle-Damgard hashes. HMAC fixes this by keying the hash:
#   HMAC(K, m) = H((K XOR opad) || H((K XOR ipad) || m))
# This construction is provably secure if the hash is a PRF.
def hmac_sha256(key: bytes, message: bytes) -> bytes:
    """Compute HMAC-SHA256 from scratch (RFC 2104)."""
    BLOCK_SIZE = 64  # SHA-256 block size

    # Why: If the key is longer than the block size, we hash it first
    # to bring it down to a manageable size. If shorter, we pad with zeros.
    if len(key) > BLOCK_SIZE:
        key = hashlib.sha256(key).digest()
    key = key.ljust(BLOCK_SIZE, b"\x00")

    # Why: ipad and opad are fixed constants (0x36 and 0x5C) chosen to
    # create two distinct "domains" for the inner and outer hash calls.
    # This prevents an attacker from relating the inner hash to the outer.
    o_key_pad = bytes(k ^ 0x5C for k in key)  # outer padding
    i_key_pad = bytes(k ^ 0x36 for k in key)  # inner padding

    inner_hash = hashlib.sha256(i_key_pad + message).digest()
    return hashlib.sha256(o_key_pad + inner_hash).digest()


# ---------------------------------------------------------------------------
# Merkle Tree
# ---------------------------------------------------------------------------

# Why: Merkle trees allow efficient verification of data integrity.
# To verify one leaf in a tree with N leaves, you only need log(N) hashes
# (the "proof path"), not the entire dataset. This is why blockchains
# (Bitcoin, Ethereum) and certificate transparency logs use them.
class MerkleTree:
    """A binary Merkle tree built from a list of data blocks."""

    def __init__(self, data_blocks: list[bytes]) -> None:
        self.leaves = [self._hash_leaf(d) for d in data_blocks]
        self.tree: list[list[bytes]] = [self.leaves[:]]
        self._build()

    @staticmethod
    def _hash_leaf(data: bytes) -> bytes:
        """Hash a leaf node (prefix 0x00 to domain-separate from branches)."""
        return hashlib.sha256(b"\x00" + data).digest()

    @staticmethod
    def _hash_branch(left: bytes, right: bytes) -> bytes:
        """Hash a branch node (prefix 0x01 to prevent second-preimage attacks)."""
        # Why: Domain separation (different prefix for leaves vs branches)
        # prevents an attacker from creating a fake shorter tree that
        # hashes to the same root. Without this, a leaf "L" could be
        # reinterpreted as a branch "hash(A||B)" and vice versa.
        return hashlib.sha256(b"\x01" + left + right).digest()

    def _build(self) -> None:
        """Build tree bottom-up."""
        current = self.tree[0]
        while len(current) > 1:
            next_level = []
            for i in range(0, len(current), 2):
                left = current[i]
                # Why: If odd number of nodes, duplicate the last one.
                # This is the standard approach in Bitcoin's Merkle tree.
                right = current[i + 1] if i + 1 < len(current) else left
                next_level.append(self._hash_branch(left, right))
            self.tree.append(next_level)
            current = next_level

    @property
    def root(self) -> bytes:
        """The Merkle root hash."""
        return self.tree[-1][0]

    def get_proof(self, index: int) -> list[tuple[str, bytes]]:
        """Get the authentication path for leaf at `index`.

        Returns list of (side, hash) pairs needed to reconstruct the root.
        """
        proof = []
        for level in self.tree[:-1]:
            if index % 2 == 0:
                sibling_idx = index + 1
                side = "right"
            else:
                sibling_idx = index - 1
                side = "left"
            if sibling_idx < len(level):
                proof.append((side, level[sibling_idx]))
            else:
                proof.append((side, level[index]))  # duplicate
            index //= 2
        return proof

    @staticmethod
    def verify_proof(
        leaf_data: bytes, proof: list[tuple[str, bytes]], root: bytes
    ) -> bool:
        """Verify a Merkle proof against the expected root."""
        current = MerkleTree._hash_leaf(leaf_data)
        for side, sibling in proof:
            if side == "right":
                current = MerkleTree._hash_branch(current, sibling)
            else:
                current = MerkleTree._hash_branch(sibling, current)
        return current == root


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  Hash Functions, Birthday Attack, HMAC & Merkle Trees")
    print("=" * 65)

    # --- Simple hash ---
    print("\n[Simple Hash Function (16-bit output)]")
    for msg in [b"hello", b"Hello", b"hello!", b"hell0"]:
        h = simple_hash(msg)
        print(f"  H({msg.decode():>8}) = 0x{h:04X}")

    # --- Birthday paradox probability ---
    print("\n[Birthday Paradox Probability]")
    bits = 16
    n = 2**bits
    for k in [10, 50, 100, 200, 300, 400]:
        p = birthday_probability(n, k)
        print(f"  {k:>4} hashes in {bits}-bit space: "
              f"P(collision) = {p:.4f}")
    theoretical_50 = math.sqrt(2 * n * math.log(2))
    print(f"  ~50% collision expected at k ~ {theoretical_50:.0f} "
          f"(sqrt(2 * 2^{bits} * ln2))")

    # --- Birthday attack ---
    print("\n[Birthday Attack on 16-bit Hash]")
    trials = 5
    total_attempts = 0
    for i in range(trials):
        attempts, m1, m2 = birthday_attack(16)
        total_attempts += attempts
        if i == 0:
            h = simple_hash(m1, 16)
            print(f"  Collision found after {attempts} attempts!")
            print(f"    msg1 = {m1.hex()}")
            print(f"    msg2 = {m2.hex()}")
            print(f"    H(msg1) = H(msg2) = 0x{h:04X}")
    avg = total_attempts / trials
    print(f"  Average over {trials} trials: {avg:.0f} attempts "
          f"(expected ~{theoretical_50:.0f})")

    # --- HMAC ---
    print("\n[HMAC-SHA256]")
    key = b"my-secret-key"
    message = b"Important message to authenticate"
    our_hmac = hmac_sha256(key, message)
    std_hmac = hmac_stdlib.new(key, message, hashlib.sha256).digest()
    print(f"  Key:     {key.decode()}")
    print(f"  Message: {message.decode()}")
    print(f"  Our HMAC:    {our_hmac.hex()}")
    print(f"  stdlib HMAC: {std_hmac.hex()}")
    print(f"  Match: {our_hmac == std_hmac}")

    # Tamper detection
    tampered = b"Important message to authenticatE"
    tampered_hmac = hmac_sha256(key, tampered)
    print(f"\n  Tampered message HMAC: {tampered_hmac.hex()[:32]}...")
    print(f"  Matches original?     {tampered_hmac == our_hmac}")

    # --- Merkle Tree ---
    print("\n[Merkle Tree]")
    blocks = [f"Transaction {i}".encode() for i in range(8)]
    tree = MerkleTree(blocks)
    print(f"  Built tree with {len(blocks)} leaves")
    print(f"  Tree depth: {len(tree.tree)} levels")
    print(f"  Root: {tree.root.hex()[:32]}...")

    # Verify a leaf
    idx = 3
    proof = tree.get_proof(idx)
    valid = MerkleTree.verify_proof(blocks[idx], proof, tree.root)
    print(f"\n  Verifying leaf {idx} ('{blocks[idx].decode()}'):")
    print(f"    Proof length: {len(proof)} hashes (log2({len(blocks)}) = "
          f"{math.log2(len(blocks)):.0f})")
    print(f"    Valid: {valid}")

    # Tampered leaf
    fake_data = b"Fake Transaction 3"
    valid_fake = MerkleTree.verify_proof(fake_data, proof, tree.root)
    print(f"\n  Verifying tampered leaf ('{fake_data.decode()}'):")
    print(f"    Valid: {valid_fake}")

    print(f"\n{'=' * 65}")
