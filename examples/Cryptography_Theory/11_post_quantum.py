"""
Post-Quantum Cryptography: NIST Standards and Migration
========================================================
Demonstrates the quantum threat to classical cryptosystems and
introduces post-quantum approaches. Includes a simplified
CRYSTALS-Kyber-like key encapsulation mechanism (KEM) concept
and crypto-agility patterns.
"""

from __future__ import annotations
import random
import hashlib


# ---------------------------------------------------------------------------
# 1. Quantum Threat Demonstration
# ---------------------------------------------------------------------------

# Why: Shor's algorithm breaks RSA and ECC in polynomial time on a quantum
# computer. Grover's algorithm halves the effective security of symmetric
# ciphers (AES-128 → 64-bit security). This section illustrates the impact.

def quantum_threat_summary() -> None:
    """Summarize the quantum impact on current cryptosystems."""
    threats = [
        ("RSA-2048", "Factoring", "Shor's", "Broken", "Migrate to ML-KEM"),
        ("ECDSA P-256", "Discrete log", "Shor's", "Broken", "Migrate to ML-DSA"),
        ("Diffie-Hellman", "Discrete log", "Shor's", "Broken", "Migrate to ML-KEM"),
        ("AES-128", "Brute force", "Grover's", "Weakened (64-bit)", "Use AES-256"),
        ("SHA-256", "Preimage", "Grover's", "Weakened (128-bit)", "Use SHA-384+"),
        ("AES-256", "Brute force", "Grover's", "Still secure (128-bit)", "No change"),
    ]

    print("  Quantum Impact on Current Cryptosystems:")
    print(f"  {'Algorithm':<16} {'Problem':<14} {'Attack':<10} {'Status':<22} {'Recommendation'}")
    print("  " + "-" * 90)
    for alg, problem, attack, status, rec in threats:
        print(f"  {alg:<16} {problem:<14} {attack:<10} {status:<22} {rec}")


# ---------------------------------------------------------------------------
# 2. NIST PQC Standards Overview
# ---------------------------------------------------------------------------

def nist_pqc_standards() -> None:
    """List the NIST post-quantum cryptography standards."""
    standards = [
        {
            "name": "ML-KEM (CRYSTALS-Kyber)",
            "type": "Key Encapsulation (KEM)",
            "family": "Lattice (Module-LWE)",
            "standard": "FIPS 203",
            "params": {"ML-KEM-512": 128, "ML-KEM-768": 192, "ML-KEM-1024": 256},
        },
        {
            "name": "ML-DSA (CRYSTALS-Dilithium)",
            "type": "Digital Signature",
            "family": "Lattice (Module-LWE/SIS)",
            "standard": "FIPS 204",
            "params": {"ML-DSA-44": 128, "ML-DSA-65": 192, "ML-DSA-87": 256},
        },
        {
            "name": "SLH-DSA (SPHINCS+)",
            "type": "Digital Signature",
            "family": "Hash-based",
            "standard": "FIPS 205",
            "params": {"SLH-DSA-128s": 128, "SLH-DSA-192s": 192, "SLH-DSA-256s": 256},
        },
    ]

    print("\n  NIST Post-Quantum Cryptography Standards:")
    for s in standards:
        print(f"\n  {s['name']} ({s['standard']})")
        print(f"    Type: {s['type']}, Family: {s['family']}")
        for param, security in s["params"].items():
            print(f"    {param}: {security}-bit security")


# ---------------------------------------------------------------------------
# 3. Simplified Lattice-Based KEM (Learning With Errors)
# ---------------------------------------------------------------------------

# Why: ML-KEM (Kyber) is the primary NIST standard for key exchange.
# This is a simplified toy version to illustrate the LWE concept.

def simple_lwe_kem(n: int = 8, q: int = 97) -> None:
    """Demonstrate a toy LWE-based key encapsulation mechanism."""
    print(f"\n  Toy LWE-KEM (n={n}, q={q}):")

    # Key generation
    s = [random.randint(0, q - 1) for _ in range(n)]  # Secret key
    A = [[random.randint(0, q - 1) for _ in range(n)] for _ in range(n)]
    e = [random.randint(-1, 1) for _ in range(n)]  # Small error

    # Public key: b = A·s + e (mod q)
    b = [(sum(A[i][j] * s[j] for j in range(n)) + e[i]) % q for i in range(n)]

    print(f"  Secret key s: {s[:4]}...")
    print(f"  Public key b: {b[:4]}...")

    # Encapsulation (sender)
    r = [random.randint(-1, 1) for _ in range(n)]  # Random vector
    e1 = [random.randint(-1, 1) for _ in range(n)]
    e2 = random.randint(-1, 1)

    u = [(sum(r[i] * A[i][j] for i in range(n)) + e1[j]) % q for j in range(n)]
    message_bit = random.randint(0, 1)
    v = (sum(r[i] * b[i] for i in range(n)) + e2 + message_bit * (q // 2)) % q

    print(f"  Original message bit: {message_bit}")
    print(f"  Ciphertext u: {u[:4]}...")
    print(f"  Ciphertext v: {v}")

    # Decapsulation (receiver)
    dec = (v - sum(s[j] * u[j] for j in range(n))) % q
    recovered = 1 if abs(dec - q // 2) < q // 4 else 0

    print(f"  Decrypted value: {dec}")
    print(f"  Recovered bit: {recovered}")
    print(f"  Correct: {recovered == message_bit}")


# ---------------------------------------------------------------------------
# 4. Hybrid Key Exchange Pattern
# ---------------------------------------------------------------------------

def hybrid_key_exchange() -> None:
    """Demonstrate the hybrid classical + post-quantum key exchange pattern."""
    print("\n  Hybrid Key Exchange (X25519 + ML-KEM-768):")
    print("  Step 1: Perform classical X25519 key exchange → shared_secret_1")
    print("  Step 2: Perform ML-KEM-768 encapsulation     → shared_secret_2")
    print("  Step 3: Combined = KDF(shared_secret_1 || shared_secret_2)")
    print()
    print("  Security: secure if EITHER algorithm is unbroken")
    print("  TLS: Already supported via X25519Kyber768Draft00")

    # Simulate
    ss1 = hashlib.sha256(b"x25519_shared_secret").digest()
    ss2 = hashlib.sha256(b"mlkem768_shared_secret").digest()
    combined = hashlib.sha256(ss1 + ss2).digest()
    print(f"\n  Classical SS:  {ss1[:8].hex()}...")
    print(f"  PQ SS:         {ss2[:8].hex()}...")
    print(f"  Combined key:  {combined[:8].hex()}...")


# ---------------------------------------------------------------------------
# 5. Crypto-Agility Pattern
# ---------------------------------------------------------------------------

def crypto_agility_demo() -> None:
    """Demonstrate crypto-agility: ability to swap algorithms without rewriting."""
    print("\n  Crypto-Agility Pattern:")

    algorithms = {
        "kem_v1": {"name": "X25519", "type": "classical", "status": "active"},
        "kem_v2": {"name": "X25519+ML-KEM-768", "type": "hybrid", "status": "active"},
        "kem_v3": {"name": "ML-KEM-1024", "type": "post-quantum", "status": "planned"},
    }

    for version, algo in algorithms.items():
        print(f"  {version}: {algo['name']} ({algo['type']}) — {algo['status']}")

    print()
    print("  Migration checklist:")
    checklist = [
        "Inventory all cryptographic dependencies",
        "Identify harvest-now-decrypt-later risks",
        "Deploy hybrid KEM for key exchange first",
        "Migrate signatures (ML-DSA) — less urgent",
        "Test with NIST reference implementations",
        "Monitor NIST round 4 candidates (code-based, etc.)",
    ]
    for i, item in enumerate(checklist, 1):
        print(f"    {i}. {item}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Post-Quantum Cryptography")
    print("=" * 60)
    quantum_threat_summary()
    nist_pqc_standards()
    simple_lwe_kem()
    hybrid_key_exchange()
    crypto_agility_demo()
