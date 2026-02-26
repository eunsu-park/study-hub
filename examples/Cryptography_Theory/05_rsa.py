"""
RSA Cryptosystem — From Scratch
================================
Key generation, encryption, decryption, and digital signatures.
Demonstrates why textbook RSA is insecure (deterministic encryption)
and how the system relies on the difficulty of integer factorization.
"""

from __future__ import annotations
import hashlib
import math
import random
import sys
import os

# Import from our number theory module (with fallback)
try:
    from importlib import import_module
    _nt = import_module("01_number_theory")
    miller_rabin = _nt.miller_rabin
    mod_inverse = _nt.mod_inverse
    mod_pow = _nt.mod_pow
    generate_prime = _nt.generate_prime
except (ImportError, ModuleNotFoundError):
    # Fallback: inline implementations for standalone execution
    def mod_pow(base: int, exp: int, mod: int) -> int:
        result = 1
        base %= mod
        while exp > 0:
            if exp & 1:
                result = (result * base) % mod
            exp >>= 1
            base = (base * base) % mod
        return result

    def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
        old_r, r = a, b
        old_s, s = 1, 0
        old_t, t = 0, 1
        while r:
            q = old_r // r
            old_r, r = r, old_r - q * r
            old_s, s = s, old_s - q * s
            old_t, t = t, old_t - q * t
        return old_r, old_s, old_t

    def mod_inverse(a: int, m: int) -> int:
        g, x, _ = _extended_gcd(a % m, m)
        if g != 1:
            raise ValueError(f"No inverse: gcd({a},{m})={g}")
        return x % m

    def miller_rabin(n: int, rounds: int = 20) -> bool:
        if n < 4:
            return n >= 2
        if n % 2 == 0:
            return False
        s, d = 0, n - 1
        while d % 2 == 0:
            d //= 2
            s += 1
        for _ in range(rounds):
            a = random.randrange(2, n - 1)
            x = mod_pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = mod_pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def generate_prime(bits: int = 512) -> int:
        while True:
            c = random.getrandbits(bits) | (1 << (bits - 1)) | 1
            if miller_rabin(c):
                return c


# ---------------------------------------------------------------------------
# RSA Key Generation
# ---------------------------------------------------------------------------

# Why: RSA security rests on the assumption that factoring n = p*q is hard
# when p and q are large primes. The public exponent e and private exponent d
# satisfy e*d = 1 (mod lambda(n)), where lambda(n) = lcm(p-1, q-1).
# Anyone who can factor n can compute d and break the system.
def rsa_keygen(bits: int = 512) -> tuple[tuple[int, int], tuple[int, int]]:
    """Generate an RSA key pair.

    Returns ((n, e), (n, d)) = (public_key, private_key).
    """
    # Why: We use bits//2 for each prime so that n is approximately `bits` bits.
    p = generate_prime(bits // 2)
    q = generate_prime(bits // 2)

    # Why: p != q is critical. If p == q, then n = p^2 and anyone can
    # compute p = sqrt(n), breaking the factoring assumption.
    while p == q:
        q = generate_prime(bits // 2)

    n = p * q

    # Why: We use lambda(n) = lcm(p-1, q-1) instead of phi(n) = (p-1)*(q-1).
    # Lambda is smaller and gives a smaller d, which is correct and efficient.
    # Both work mathematically, but lambda(n) is the "right" modulus.
    phi_n = (p - 1) * (q - 1)
    lambda_n = phi_n // math.gcd(p - 1, q - 1)

    # Why: e = 65537 (0x10001) is the standard public exponent because:
    # 1) It's prime (so gcd(e, lambda) = 1 almost always)
    # 2) It has only two 1-bits in binary, making encryption very fast
    # 3) It's large enough to resist small-exponent attacks
    e = 65537
    if math.gcd(e, lambda_n) != 1:
        # Extremely rare; fall back to finding a suitable e
        e = 3
        while math.gcd(e, lambda_n) != 1:
            e += 2

    d = mod_inverse(e, lambda_n)

    return (n, e), (n, d)


# ---------------------------------------------------------------------------
# Textbook RSA Encrypt / Decrypt
# ---------------------------------------------------------------------------

# Why: "Textbook RSA" means raw modular exponentiation: c = m^e mod n.
# This is INSECURE for real use because:
# 1) It's deterministic — same message always gives same ciphertext
# 2) It's malleable — Enc(m1) * Enc(m2) = Enc(m1*m2)
# 3) Small messages can be recovered by taking the e-th root
# Real RSA uses padding schemes like OAEP to fix these issues.
def rsa_encrypt(message: int, public_key: tuple[int, int]) -> int:
    """Textbook RSA encryption: c = m^e mod n."""
    n, e = public_key
    if message >= n:
        raise ValueError("Message must be less than modulus n")
    return mod_pow(message, e, n)


def rsa_decrypt(ciphertext: int, private_key: tuple[int, int]) -> int:
    """Textbook RSA decryption: m = c^d mod n."""
    n, d = private_key
    return mod_pow(ciphertext, d, n)


# ---------------------------------------------------------------------------
# RSA Digital Signature
# ---------------------------------------------------------------------------

# Why: RSA signatures use the private key to "encrypt" a hash of the message.
# Anyone with the public key can verify by checking that the "decrypted"
# signature matches the hash. Security: only the private key holder can
# produce a valid signature.
def rsa_sign(message: bytes, private_key: tuple[int, int]) -> int:
    """Sign a message: signature = H(message)^d mod n."""
    # Why: We hash the message first because:
    # 1) It compresses arbitrary-length messages to a fixed size
    # 2) It prevents existential forgery attacks on textbook RSA
    msg_hash = int.from_bytes(hashlib.sha256(message).digest(), "big")
    n, d = private_key
    msg_hash %= n  # ensure hash < n
    return mod_pow(msg_hash, d, n)


def rsa_verify(
    message: bytes, signature: int, public_key: tuple[int, int]
) -> bool:
    """Verify an RSA signature: check H(message) == sig^e mod n."""
    n, e = public_key
    msg_hash = int.from_bytes(hashlib.sha256(message).digest(), "big")
    msg_hash %= n
    recovered = mod_pow(signature, e, n)
    return recovered == msg_hash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bytes_to_int(data: bytes) -> int:
    """Convert bytes to integer for RSA operations."""
    return int.from_bytes(data, "big")


def int_to_bytes(n: int, length: int | None = None) -> bytes:
    """Convert integer back to bytes."""
    if length is None:
        length = (n.bit_length() + 7) // 8
    return n.to_bytes(length, "big")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  RSA Cryptosystem — From Scratch")
    print("=" * 65)

    # Key generation
    print("\n[Key Generation (512-bit modulus for speed)]")
    pub, priv = rsa_keygen(512)
    n, e = pub
    _, d = priv
    print(f"  n = {n}")
    print(f"  e = {e}")
    print(f"  d = {d}")
    print(f"  Modulus bit length: {n.bit_length()}")

    # Encryption / Decryption
    print("\n[Textbook RSA Encrypt/Decrypt]")
    message_text = "Hello, RSA!"
    m = bytes_to_int(message_text.encode())
    print(f"  Message: '{message_text}'")
    print(f"  As integer: {m}")

    c = rsa_encrypt(m, pub)
    print(f"  Ciphertext: {c}")

    recovered_m = rsa_decrypt(c, priv)
    recovered_text = int_to_bytes(recovered_m).decode()
    print(f"  Decrypted: '{recovered_text}'")
    print(f"  Match: {message_text == recovered_text}")

    # Digital Signature
    print("\n[RSA Digital Signature]")
    doc = b"I agree to pay Bob $100"
    sig = rsa_sign(doc, priv)
    print(f"  Document: '{doc.decode()}'")
    print(f"  Signature: {sig}")

    valid = rsa_verify(doc, sig, pub)
    print(f"  Verification: {'VALID' if valid else 'INVALID'}")

    # Tampered document
    tampered = b"I agree to pay Bob $10000"
    valid_tampered = rsa_verify(tampered, sig, pub)
    print(f"\n  Tampered:  '{tampered.decode()}'")
    print(f"  Verification: {'VALID' if valid_tampered else 'INVALID'}")

    # Why textbook RSA is insecure: deterministic encryption
    print("\n[Why Textbook RSA Is Insecure]")
    print("  Encrypting the same message twice:")
    c1 = rsa_encrypt(m, pub)
    c2 = rsa_encrypt(m, pub)
    print(f"  c1 = {c1}")
    print(f"  c2 = {c2}")
    print(f"  c1 == c2: {c1 == c2}")
    print("  ^ Same ciphertext! An eavesdropper can detect repeated messages.")

    # Multiplicative homomorphism
    print("\n  Multiplicative homomorphism (malleability):")
    m1, m2 = 42, 7
    c1 = rsa_encrypt(m1, pub)
    c2 = rsa_encrypt(m2, pub)
    c_product = (c1 * c2) % n
    decrypted_product = rsa_decrypt(c_product, priv)
    print(f"  Enc({m1}) * Enc({m2}) mod n -> Dec = {decrypted_product}")
    print(f"  {m1} * {m2} = {m1 * m2}")
    print(f"  Match: {decrypted_product == m1 * m2}")
    print("  ^ Attacker can manipulate ciphertexts without knowing plaintext!")

    print(f"\n{'=' * 65}")
