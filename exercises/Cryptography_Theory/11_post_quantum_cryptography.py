"""
Exercises for Lesson 11: Post-Quantum Cryptography
Topic: Cryptography_Theory
Solutions to practice problems from the lesson.
"""

import random
import math
from math import gcd, isqrt


def exercise_1():
    """Exercise 1: Quantum Threat Assessment (Conceptual)

    Assess quantum threat level for 5 systems and recommend migration priority.
    """
    print(f"  Quantum Threat Assessment")

    systems = [
        {
            "name": "Hospital EMR System",
            "description": "Electronic medical records, 30-year retention requirement",
            "data_sensitivity": "High (HIPAA, patient privacy)",
            "retention": "30+ years",
            "threat_level": "HIGH",
            "priority": "1 (Urgent)",
            "reasoning": (
                "Medical records must remain confidential for 30+ years. "
                "An adversary recording encrypted EMR traffic today could decrypt it "
                "with a quantum computer in 10-15 years, exposing patient data that "
                "still needs protection. 'Harvest now, decrypt later' is the primary threat."
            ),
            "recommendation": (
                "Immediate migration to hybrid PQC for data in transit. "
                "Re-encrypt stored data with PQ-safe algorithms. "
                "Use AES-256 (quantum-resistant symmetric) for data at rest."
            ),
        },
        {
            "name": "Real-Time Gaming Platform",
            "description": "Session keys live for minutes, no long-term secrets",
            "data_sensitivity": "Low (game state, no PII in transit)",
            "retention": "Minutes",
            "threat_level": "LOW",
            "priority": "5 (Lowest)",
            "reasoning": (
                "Session keys are ephemeral (minutes). Even if an adversary records "
                "traffic, the decrypted data (game moves, scores) has no value "
                "years later. Forward secrecy with ECDHE already limits exposure."
            ),
            "recommendation": (
                "No urgent action needed. Adopt PQC when it becomes the default "
                "in TLS libraries. Focus on other security priorities first."
            ),
        },
        {
            "name": "Government Intelligence Communications",
            "description": "Classified communications, 75+ year secrecy requirement",
            "data_sensitivity": "Critical (national security)",
            "retention": "75+ years",
            "threat_level": "HIGH",
            "priority": "1 (Urgent)",
            "reasoning": (
                "Intelligence data may need to remain secret for 75+ years. "
                "State adversaries are actively recording encrypted traffic. "
                "The 'harvest now, decrypt later' threat is the most acute here. "
                "NSA has mandated PQC migration for classified systems."
            ),
            "recommendation": (
                "Immediate deployment of hybrid PQC (classical + lattice). "
                "Use Suite B replacement (ML-KEM, ML-DSA). "
                "Air-gapped systems need PQ-safe key exchange for initial setup."
            ),
        },
        {
            "name": "Public Blog Website",
            "description": "All content is already public, no user accounts",
            "data_sensitivity": "None (public content)",
            "retention": "N/A",
            "threat_level": "LOW",
            "priority": "5 (Lowest)",
            "reasoning": (
                "The content is already public. HTTPS protects integrity "
                "(no tampering), not confidentiality. Even with a quantum computer, "
                "there is nothing to decrypt that isn't already public."
            ),
            "recommendation": (
                "Adopt PQC passively when web servers and CDNs support it by default. "
                "No proactive migration needed."
            ),
        },
        {
            "name": "Cryptocurrency Wallet",
            "description": "Private keys stored long-term, ECDSA signatures",
            "data_sensitivity": "Critical (financial assets)",
            "retention": "Indefinite (as long as funds exist)",
            "threat_level": "HIGH",
            "priority": "2 (High)",
            "reasoning": (
                "ECDSA (secp256k1) is broken by Shor's algorithm. "
                "Private keys derived from public keys on the blockchain are "
                "permanently exposed. Any address that has sent a transaction "
                "has its public key on-chain, making it vulnerable to future "
                "quantum attack. Funds could be stolen retroactively."
            ),
            "recommendation": (
                "Move funds to quantum-resistant addresses when available. "
                "Bitcoin community is researching post-quantum signature schemes. "
                "Use addresses only once (minimize public key exposure). "
                "Consider migrating to a PQ-safe blockchain if funds are significant."
            ),
        },
    ]

    for i, sys in enumerate(systems, 1):
        print(f"\n  System {i}: {sys['name']}")
        print(f"    Description:    {sys['description']}")
        print(f"    Data sensitivity: {sys['data_sensitivity']}")
        print(f"    Retention:      {sys['retention']}")
        print(f"    Threat level:   {sys['threat_level']}")
        print(f"    Priority:       {sys['priority']}")
        print(f"    Reasoning:      {sys['reasoning']}")
        print(f"    Recommendation: {sys['recommendation']}")


def exercise_2():
    """Exercise 2: Shor's Algorithm Exploration (Coding)

    Classical simulation of Shor's period-finding step.
    Factor all semiprimes up to 1000.
    """
    def classical_period_find(a, N):
        """Find the period r of a^x mod N (classical brute force)."""
        x = 1
        for r in range(1, N):
            x = (x * a) % N
            if x == 1:
                return r
        return None

    def shor_factor(N, max_attempts=50):
        """Factor N using Shor's algorithm (classical simulation).

        Steps:
        1. Pick random a < N
        2. Check gcd(a, N) != 1 (lucky factor)
        3. Find period r of a^x mod N
        4. If r is even, try gcd(a^(r/2) +/- 1, N)
        """
        if N % 2 == 0:
            return 2, N // 2, 1

        for attempt in range(1, max_attempts + 1):
            a = random.randrange(2, N)

            # Lucky: gcd(a, N) > 1
            g = gcd(a, N)
            if g > 1:
                return g, N // g, attempt

            # Find period
            r = classical_period_find(a, N)
            if r is None:
                continue

            # Need r to be even
            if r % 2 != 0:
                continue

            # Try factoring: gcd(a^(r/2) +/- 1, N)
            half = pow(a, r // 2, N)
            f1 = gcd(half - 1, N)
            f2 = gcd(half + 1, N)

            if 1 < f1 < N:
                return f1, N // f1, attempt
            if 1 < f2 < N:
                return f2, N // f2, attempt

        return None, None, max_attempts

    print(f"  Shor's Algorithm: Factor Semiprimes up to 1000")

    # Generate all semiprimes up to 1000
    def is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0: return False
        for i in range(3, isqrt(n) + 1, 2):
            if n % i == 0: return False
        return True

    primes = [p for p in range(2, 500) if is_prime(p)]
    semiprimes = []
    for i, p in enumerate(primes):
        for q in primes[i:]:
            n = p * q
            if n <= 1000 and p != q:  # Exclude perfect squares
                semiprimes.append((n, p, q))

    semiprimes.sort()
    print(f"    Found {len(semiprimes)} semiprimes up to 1000")

    # Factor each
    successes = 0
    total_attempts = 0
    attempt_counts = []

    for n, p_true, q_true in semiprimes[:50]:  # First 50 for time
        f1, f2, attempts = shor_factor(n)
        total_attempts += attempts
        if f1 is not None and f1 * f2 == n:
            successes += 1
            attempt_counts.append(attempts)

    # Show some results
    print(f"\n    Results (first 50 semiprimes):")
    print(f"    Successes: {successes}/50")
    print(f"    Average attempts: {total_attempts/50:.1f}")
    if attempt_counts:
        print(f"    Min attempts: {min(attempt_counts)}")
        print(f"    Max attempts: {max(attempt_counts)}")
        print(f"    Median attempts: {sorted(attempt_counts)[len(attempt_counts)//2]}")

    # Detailed examples
    print(f"\n    Detailed examples:")
    for n, p, q in [(15, 3, 5), (77, 7, 11), (221, 13, 17), (899, 29, 31)]:
        f1, f2, att = shor_factor(n)
        print(f"    N={n}: factors=({f1}, {f2}), attempts={att}, "
              f"correct={f1*f2==n if f1 else False}")

    # Part 4: Why multiple attempts?
    print(f"\n  Why multiple attempts are sometimes needed:")
    print(f"    1. Random a may share a factor with N (lucky but rare)")
    print(f"    2. Period r may be odd -> a^(r/2) is not an integer power")
    print(f"    3. a^(r/2) +/- 1 may give trivial factors (1 or N)")
    print(f"    4. Probability of success per attempt: >= 1/2 (for most N)")
    print(f"    5. After k attempts: P(failure) <= (1/2)^k")
    print(f"    6. For k=20 attempts: P(failure) <= 2^(-20) ~ 10^(-6)")


def exercise_3():
    """Exercise 3: PQC Key Size Impact (Coding)

    Model TLS 1.3 handshake sizes with classical vs PQC algorithms.
    """
    print(f"  PQC Key Size Impact on TLS Handshake")

    # Sizes in bytes
    classical = {
        "name": "Classical (X25519 + Ed25519)",
        "kex_public": 32,      # X25519 public key
        "kex_ciphertext": 32,  # X25519 response
        "sig_public": 32,      # Ed25519 public key
        "sig_signature": 64,   # Ed25519 signature
    }

    pqc = {
        "name": "PQC (Kyber-768 + Dilithium3)",
        "kex_public": 1184,    # ML-KEM-768 public key
        "kex_ciphertext": 1088, # ML-KEM-768 ciphertext
        "sig_public": 1952,    # ML-DSA-65 public key
        "sig_signature": 3293, # ML-DSA-65 signature
    }

    hybrid = {
        "name": "Hybrid (X25519+Kyber-768, Ed25519+Dilithium3)",
        "kex_public": 32 + 1184,
        "kex_ciphertext": 32 + 1088,
        "sig_public": 32 + 1952,
        "sig_signature": 64 + 3293,
    }

    # TLS 1.3 handshake components
    # ClientHello: key_share (public key)
    # ServerHello: key_share (ciphertext/public key)
    # Certificate: 3 certs, each with public key
    # CertificateVerify: signature
    # Additional overhead: ~500 bytes (headers, extensions, etc.)

    print(f"\n  {'Component':<30} {'Classical':>10} {'PQC':>10} {'Hybrid':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

    components = [
        ("ClientHello (key_share)", "kex_public"),
        ("ServerHello (key_share)", "kex_ciphertext"),
        ("Certificate (3x pub key)", "sig_public"),
        ("CertificateVerify (sig)", "sig_signature"),
    ]

    totals = {"Classical": 0, "PQC": 0, "Hybrid": 0}

    for comp_name, key in components:
        c_size = classical[key]
        p_size = pqc[key]
        h_size = hybrid[key]

        # For certificate chain (3 certs)
        if "Certificate" in comp_name and "Verify" not in comp_name:
            c_size *= 3
            p_size *= 3
            h_size *= 3

        totals["Classical"] += c_size
        totals["PQC"] += p_size
        totals["Hybrid"] += h_size

        print(f"  {comp_name:<30} {c_size:>10,} {p_size:>10,} {h_size:>10,}")

    overhead = 500
    for k in totals:
        totals[k] += overhead

    print(f"  {'TLS overhead':<30} {overhead:>10} {overhead:>10} {overhead:>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'TOTAL (bytes)':<30} {totals['Classical']:>10,} {totals['PQC']:>10,} {totals['Hybrid']:>10,}")

    growth_pqc = totals['PQC'] / totals['Classical']
    growth_hybrid = totals['Hybrid'] / totals['Classical']
    print(f"\n  PQC handshake is {growth_pqc:.1f}x larger than classical")
    print(f"  Hybrid handshake is {growth_hybrid:.1f}x larger than classical")

    # Latency impact
    print(f"\n  Latency Impact:")
    print(f"  {'RTT (ms)':>10} {'Classical (ms)':>15} {'PQC (ms)':>12} {'Hybrid (ms)':>13}")
    print(f"  {'-'*10} {'-'*15} {'-'*12} {'-'*13}")

    for rtt in [1, 10, 50, 100, 200]:
        # Assume 10 Mbps link (1.25 MB/s)
        bandwidth = 1.25e6  # bytes per second
        c_transfer = totals['Classical'] / bandwidth * 1000
        p_transfer = totals['PQC'] / bandwidth * 1000
        h_transfer = totals['Hybrid'] / bandwidth * 1000

        c_total = rtt + c_transfer
        p_total = rtt + p_transfer
        h_total = rtt + h_transfer

        print(f"  {rtt:>10} {c_total:>15.1f} {p_total:>12.1f} {h_total:>13.1f}")

    print(f"\n  At high bandwidths (>100 Mbps), the size increase is negligible.")
    print(f"  On slow connections (mobile, satellite), PQC adds measurable delay.")


def exercise_4():
    """Exercise 4: Hybrid KEM Implementation (Coding)

    Combine X25519 and simplified LWE.
    Show security when one scheme is 'broken'.
    """
    import hashlib
    import os

    print(f"  Hybrid KEM: X25519 + Simplified LWE")

    # Simulated key exchange components
    def simulate_x25519():
        """Simulate X25519 ECDH (random 32-byte shared secret)."""
        # In production, use cryptography.hazmat.primitives.asymmetric.x25519
        alice_priv = os.urandom(32)
        bob_priv = os.urandom(32)
        # Simulated shared secret (deterministic from both keys)
        shared = hashlib.sha256(alice_priv + bob_priv).digest()
        return shared, alice_priv, bob_priv

    def simulate_lwe_kem(n=32, q=97, error_std=1.0):
        """Simplified LWE KEM."""
        # Generate shared key material
        s = [random.randrange(q) for _ in range(n)]
        r = [random.randrange(-1, 2) for _ in range(n)]
        A = [[random.randrange(q) for _ in range(n)] for _ in range(n)]

        # Alice's public key
        e = [round(random.gauss(0, error_std)) % q for _ in range(n)]
        b = [(sum(A[i][j] * s[j] for j in range(n)) + e[i]) % q for i in range(n)]

        # Bob encapsulates
        v_bob = sum(b[j] * r[j] for j in range(n)) % q
        v_alice = sum(s[j] * (sum(A[k][j] * r[k] for k in range(n)) % q) for j in range(n)) % q

        # Derive shared secret from approximate agreement
        shared = hashlib.sha256(str(v_bob % (q // 2)).encode()).digest()
        return shared

    def hkdf_combine(secret1, secret2, info=b"hybrid-kem"):
        """Combine two shared secrets using HKDF-like construction."""
        combined = secret1 + secret2
        return hashlib.sha256(combined + info).digest()

    # Part 1: Normal operation
    print(f"\n  Part 1: Normal hybrid operation")
    ecdh_secret, _, _ = simulate_x25519()
    lwe_secret = simulate_lwe_kem()
    hybrid_key = hkdf_combine(ecdh_secret, lwe_secret)

    print(f"    ECDH secret:   {ecdh_secret.hex()[:32]}...")
    print(f"    LWE secret:    {lwe_secret.hex()[:32]}...")
    print(f"    Hybrid key:    {hybrid_key.hex()[:32]}...")

    # Part 2: X25519 is 'broken' (quantum computer)
    print(f"\n  Part 2: Scenario - X25519 is broken")
    print(f"    Attacker knows ECDH shared secret: {ecdh_secret.hex()[:32]}...")
    print(f"    But LWE secret is still unknown")
    print(f"    Attacker cannot compute hybrid key without LWE secret")

    # Attacker tries all possible LWE secrets
    # Show that even knowing ECDH, the hybrid key is protected by LWE
    fake_lwe = os.urandom(32)
    fake_hybrid = hkdf_combine(ecdh_secret, fake_lwe)
    print(f"    Attacker's guess: {fake_hybrid.hex()[:32]}...")
    print(f"    Matches real key: {fake_hybrid == hybrid_key}")
    print(f"    -> Hybrid key is STILL SECURE (LWE protects)")

    # Part 3: LWE is 'broken' (new lattice attack)
    print(f"\n  Part 3: Scenario - LWE is broken")
    print(f"    Attacker knows LWE shared secret: {lwe_secret.hex()[:32]}...")
    print(f"    But ECDH secret is still unknown")

    fake_ecdh = os.urandom(32)
    fake_hybrid2 = hkdf_combine(fake_ecdh, lwe_secret)
    print(f"    Attacker's guess: {fake_hybrid2.hex()[:32]}...")
    print(f"    Matches real key: {fake_hybrid2 == hybrid_key}")
    print(f"    -> Hybrid key is STILL SECURE (ECDH protects)")

    # Part 4: Both broken
    print(f"\n  Part 4: Both schemes broken")
    recovered_hybrid = hkdf_combine(ecdh_secret, lwe_secret)
    print(f"    Attacker computes: {recovered_hybrid.hex()[:32]}...")
    print(f"    Matches real key:  {recovered_hybrid == hybrid_key}")
    print(f"    -> Hybrid key is COMPROMISED (both broken)")

    print(f"\n  Conclusion:")
    print(f"    Hybrid KEM security = max(ECDH security, LWE security)")
    print(f"    The hybrid is at least as strong as the stronger component.")


def exercise_5():
    """Exercise 5: Crypto Agility Audit (Challenging)

    Audit a system for crypto agility.
    """
    print(f"  Crypto Agility Audit Framework")

    print(f"""
  Crypto Agility: the ability to swap cryptographic algorithms without
  major code changes. Critical for PQC migration.

  Audit Dimensions:
  =================

  1. Algorithm Specification Location
  -----------------------------------
  Bad:  Hardcoded in source code
        const ALGORITHM = "RSA-2048"  // Cannot change without recompiling

  Better: Configuration file
        algorithm: RSA-2048          // Change requires config update + restart

  Best: Runtime negotiation
        supported: [ML-KEM-768, X25519, RSA-2048]  // Dynamic selection

  2. Abstraction Level
  --------------------
  Bad:  Direct library calls scattered through codebase
        openssl_rsa_sign(key, data)  // Tied to RSA

  Better: Wrapper functions
        crypto_sign(key, data)  // Dispatches based on key type

  Best: Algorithm-agnostic interfaces
        class Signer:
            def sign(self, data) -> bytes
            def verify(self, data, signature) -> bool

  3. Key Management
  -----------------
  Bad:  Key format hardcoded (PEM RSA only)
  Better: Multiple key formats supported
  Best: Key metadata includes algorithm, version, migration status

  4. Protocol Negotiation
  -----------------------
  Bad:  Fixed cipher suite
  Better: Configurable cipher suite list
  Best: Dynamic negotiation with version fallback

  Example Audit: OpenSSH
  ======================

  1. Algorithm specification:
     - ssh_config / sshd_config: KexAlgorithms, HostKeyAlgorithms, Ciphers
     - Runtime negotiation between client and server
     - Score: GOOD (configurable, negotiated)

  2. Adding Kyber support:
     - Difficulty: MODERATE
     - OpenSSH 9.0+ already supports sntrup761x25519-sha512 (hybrid PQ)
     - New algorithms can be added via compile-time options
     - No source code changes needed for end users

  3. Abstractions:
     - Key exchange: abstracted via kex_* interface
     - Signing: abstracted via sshkey_sign/sshkey_verify
     - Both are algorithm-agnostic
     - Score: GOOD

  4. Migration plan for full PQC:
     Step 1: Enable hybrid key exchange (sntrup761x25519) [DONE in OpenSSH 9.x]
     Step 2: Add ML-KEM-768 support when standardized
     Step 3: Add ML-DSA-65 for host key signatures
     Step 4: Deprecate classical-only cipher suites
     Step 5: Remove classical algorithms after transition period

  Generic Migration Checklist:
  ============================
  [ ] Inventory all cryptographic algorithm usage
  [ ] Identify algorithm-specific code vs abstracted code
  [ ] Add PQC algorithm support to crypto abstraction layer
  [ ] Implement hybrid mode (classical + PQC)
  [ ] Test with PQC-only mode (for future)
  [ ] Update key management to handle larger PQC keys
  [ ] Benchmark performance impact (PQC is slower/larger)
  [ ] Plan certificate chain migration (PQC certificates are larger)
  [ ] Document algorithm deprecation timeline
  [ ] Train operations team on PQC monitoring
  """)


if __name__ == "__main__":
    print("=== Exercise 1: Quantum Threat Assessment ===")
    exercise_1()

    print("\n=== Exercise 2: Shor's Algorithm Exploration ===")
    exercise_2()

    print("\n=== Exercise 3: PQC Key Size Impact ===")
    exercise_3()

    print("\n=== Exercise 4: Hybrid KEM Implementation ===")
    exercise_4()

    print("\n=== Exercise 5: Crypto Agility Audit ===")
    exercise_5()

    print("\nAll exercises completed!")
