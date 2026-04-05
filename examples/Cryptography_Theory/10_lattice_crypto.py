"""
Lattice-Based Cryptography: LWE Encryption
============================================
Introduces lattices, the Learning With Errors (LWE) problem, and
a simple LWE-based encryption scheme. Lattice cryptography is the
leading candidate for post-quantum security.
"""

from __future__ import annotations
import random
import math


# ---------------------------------------------------------------------------
# 2D Lattice Visualization
# ---------------------------------------------------------------------------

# Why: A lattice is the set of all integer linear combinations of basis
# vectors. In 2D, think of a grid tilted and stretched by the basis.
# Lattice problems (SVP, CVP) are believed to be hard even for quantum
# computers, unlike factoring (RSA) and discrete log (ECC).
def visualize_2d_lattice(
    b1: tuple[int, int],
    b2: tuple[int, int],
    bounds: int = 5,
) -> None:
    """Print a 2D lattice defined by basis vectors b1 and b2."""
    print(f"  Basis: b1 = {b1}, b2 = {b2}")

    # Generate lattice points
    points: set[tuple[int, int]] = set()
    for i in range(-bounds, bounds + 1):
        for j in range(-bounds, bounds + 1):
            x = i * b1[0] + j * b2[0]
            y = i * b1[1] + j * b2[1]
            if -15 <= x <= 15 and -15 <= y <= 15:
                points.add((x, y))

    # ASCII plot
    for y in range(10, -11, -1):
        row = ""
        for x in range(-15, 16):
            if (x, y) in points:
                row += "*"
            elif x == 0 and y == 0:
                row += "+"
            elif x == 0:
                row += "|"
            elif y == 0:
                row += "-"
            else:
                row += " "
        # Only print rows with content
        if "*" in row or "+" in row:
            print(f"    {row}")


# ---------------------------------------------------------------------------
# LWE Parameters and Helpers
# ---------------------------------------------------------------------------

# Why: The Learning With Errors (LWE) problem asks: given pairs (a_i, b_i)
# where b_i = <a_i, s> + e_i (mod q), with s a secret vector and e_i
# small noise, find s. This is believed to be as hard as worst-case
# lattice problems (by Regev's reduction, 2005). LWE is the foundation
# of most post-quantum crypto schemes including CRYSTALS-Kyber (ML-KEM).

# Why: Small parameters for educational demonstration. Real LWE uses
# n ~ 1024, q ~ 2^23, and carefully chosen error distributions.
# These small values let us see the noise and verify correctness by hand.
LWE_N = 4       # secret vector dimension
LWE_Q = 97      # modulus (prime for simplicity)
LWE_ERROR_BOUND = 3  # error sampled from [-ERROR_BOUND, ERROR_BOUND]


def sample_error() -> int:
    """Sample a small error value from a discrete distribution."""
    # Why: In real LWE, errors follow a discrete Gaussian distribution
    # centered at 0. We use uniform [-bound, bound] for simplicity.
    # The error must be small enough that decryption succeeds (< q/4).
    return random.randint(-LWE_ERROR_BOUND, LWE_ERROR_BOUND)


def sample_vector(n: int, q: int) -> list[int]:
    """Sample a random vector in Z_q^n."""
    return [random.randrange(q) for _ in range(n)]


def inner_product(a: list[int], b: list[int], q: int) -> int:
    """Compute <a, b> mod q."""
    return sum(x * y for x, y in zip(a, b)) % q


# ---------------------------------------------------------------------------
# LWE Key Generation
# ---------------------------------------------------------------------------

# Why: The public key is a set of LWE samples (A, b = As + e mod q).
# The secret key is the vector s. Anyone with the public key can encrypt,
# but only the holder of s can decrypt (by removing the noise).
def lwe_keygen(
    n: int = LWE_N,
    q: int = LWE_Q,
    m: int | None = None,
) -> tuple[tuple[list[list[int]], list[int]], list[int]]:
    """Generate LWE key pair.

    Returns (public_key=(A, b), secret_key=s).
    """
    if m is None:
        m = n + 4  # number of LWE samples

    # Secret key
    s = sample_vector(n, q)

    # Public key: A is m x n random matrix, b = A*s + e (mod q)
    A = [sample_vector(n, q) for _ in range(m)]
    e = [sample_error() for _ in range(m)]
    b = [(inner_product(A[i], s, q) + e[i]) % q for i in range(m)]

    return (A, b), s


# ---------------------------------------------------------------------------
# LWE Encryption / Decryption
# ---------------------------------------------------------------------------

# Why: To encrypt a single bit mu in {0, 1}:
# 1. Choose a random subset S of the public key rows
# 2. Sum the selected rows: u = sum(a_i for i in S), v = sum(b_i for i in S)
# 3. If mu = 1, add q//2 to v (encode the bit in the "upper half" of Z_q)
# Decryption: compute v - <u, s> mod q. If close to 0, bit is 0;
# if close to q//2, bit is 1. The accumulated noise is small enough
# (with high probability) that the bit can be recovered.

def lwe_encrypt_bit(
    bit: int,
    public_key: tuple[list[list[int]], list[int]],
    n: int = LWE_N,
    q: int = LWE_Q,
) -> tuple[list[int], int]:
    """Encrypt a single bit (0 or 1) using LWE.

    Returns (u, v) where u is a vector and v is a scalar.
    """
    A, b = public_key
    m = len(A)

    # Why: We select a random subset by flipping coins. This randomized
    # subset-sum is what makes each ciphertext look different, even for
    # the same plaintext bit — achieving semantic security.
    subset = [random.randint(0, 1) for _ in range(m)]

    u = [0] * n
    v = 0

    for i in range(m):
        if subset[i]:
            for j in range(n):
                u[j] = (u[j] + A[i][j]) % q
            v = (v + b[i]) % q

    # Why: Encoding the bit as 0 or q//2 maximizes the "distance" between
    # the two possible values. Since noise accumulates to at most ~sqrt(m)*bound,
    # the decoder can distinguish 0 from q//2 as long as noise < q/4.
    if bit == 1:
        v = (v + q // 2) % q

    return u, v


def lwe_decrypt_bit(
    ciphertext: tuple[list[int], int],
    secret_key: list[int],
    q: int = LWE_Q,
) -> int:
    """Decrypt an LWE ciphertext to recover the bit."""
    u, v = ciphertext
    n = len(secret_key)

    # Why: v - <u, s> = sum(e_i for i in S) + bit * (q//2)
    # The error sum is small, so we just check if the result is
    # closer to 0 (bit=0) or to q//2 (bit=1).
    inner = inner_product(u, secret_key, q)
    noisy_bit = (v - inner) % q

    # Decode: closer to 0 -> bit 0, closer to q//2 -> bit 1
    if noisy_bit > q // 4 and noisy_bit < 3 * q // 4:
        return 1
    else:
        return 0


def lwe_encrypt_message(
    message: str,
    public_key: tuple[list[list[int]], list[int]],
) -> list[tuple[list[int], int]]:
    """Encrypt a string bit-by-bit using LWE."""
    ciphertexts = []
    for char in message:
        for bit_pos in range(7, -1, -1):  # MSB first
            bit = (ord(char) >> bit_pos) & 1
            ct = lwe_encrypt_bit(bit, public_key)
            ciphertexts.append(ct)
    return ciphertexts


def lwe_decrypt_message(
    ciphertexts: list[tuple[list[int], int]],
    secret_key: list[int],
    length: int,
) -> str:
    """Decrypt LWE ciphertexts back to a string."""
    bits = [lwe_decrypt_bit(ct, secret_key) for ct in ciphertexts]
    chars = []
    for i in range(0, len(bits), 8):
        byte_bits = bits[i : i + 8]
        if len(byte_bits) == 8:
            val = 0
            for b in byte_bits:
                val = (val << 1) | b
            chars.append(chr(val))
    return "".join(chars[:length])


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  Lattice-Based Cryptography: LWE")
    print("=" * 65)

    # --- 2D Lattice ---
    print("\n[2D Lattice Visualization]")
    print("\n  Standard basis (grid):")
    visualize_2d_lattice((1, 0), (0, 1), bounds=10)

    print("\n  Skewed basis (same lattice, harder to solve SVP):")
    visualize_2d_lattice((3, 1), (1, 2), bounds=4)

    # --- LWE Encryption ---
    print(f"\n[LWE Parameters]")
    print(f"  Dimension n = {LWE_N}")
    print(f"  Modulus q = {LWE_Q}")
    print(f"  Error bound = +/-{LWE_ERROR_BOUND}")

    print(f"\n[LWE Key Generation]")
    pub, sec = lwe_keygen()
    A, b = pub
    print(f"  Secret key s = {sec}")
    print(f"  Public key: {len(A)} LWE samples (A rows + b values)")
    print(f"  First sample: a = {A[0]}, b = {b[0]}")
    expected = inner_product(A[0], sec, LWE_Q)
    error = (b[0] - expected) % LWE_Q
    if error > LWE_Q // 2:
        error -= LWE_Q
    print(f"    <a, s> mod q = {expected}, b - <a,s> = {error} (noise)")

    # Single bit encryption
    print(f"\n[LWE Single Bit Encryption]")
    for bit in [0, 1]:
        ct = lwe_encrypt_bit(bit, pub)
        decrypted = lwe_decrypt_bit(ct, sec)
        u, v = ct
        inner = inner_product(u, sec, LWE_Q)
        noisy = (v - inner) % LWE_Q
        print(f"  Encrypt({bit}): v - <u,s> mod q = {noisy:>3}  "
              f"(q/2 = {LWE_Q//2})  -> Decrypt = {decrypted}  "
              f"{'OK' if decrypted == bit else 'ERROR'}")

    # Message encryption
    print(f"\n[LWE Message Encryption]")
    message = "Hi!"
    cts = lwe_encrypt_message(message, pub)
    decrypted = lwe_decrypt_message(cts, sec, len(message))
    print(f"  Original:  '{message}'")
    print(f"  Encrypted: {len(cts)} ciphertexts ({len(message)} chars x 8 bits)")
    print(f"  Decrypted: '{decrypted}'")
    print(f"  Match: {message == decrypted}")

    # Noise analysis
    print(f"\n[Noise Analysis]")
    errors = 0
    total = 1000
    for _ in range(total):
        bit = random.randint(0, 1)
        ct = lwe_encrypt_bit(bit, pub)
        dec = lwe_decrypt_bit(ct, sec)
        if dec != bit:
            errors += 1
    print(f"  Tested {total} random bit encryptions")
    print(f"  Decryption errors: {errors}/{total} "
          f"({errors/total*100:.1f}%)")
    print(f"  Error rate depends on noise accumulation vs q/4 threshold")

    # Why lattice crypto matters
    print(f"\n[Why Lattice Cryptography?]")
    print(f"  - RSA/ECC broken by quantum computers (Shor's algorithm)")
    print(f"  - Lattice problems are believed quantum-resistant")
    print(f"  - NIST post-quantum standards (2024):")
    print(f"    ML-KEM (Kyber)    - key encapsulation (based on MLWE)")
    print(f"    ML-DSA (Dilithium) - digital signatures (based on MLWE)")
    print(f"    SLH-DSA (SPHINCS+) - hash-based signatures (backup)")

    print(f"\n{'=' * 65}")
