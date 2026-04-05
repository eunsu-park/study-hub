"""
Example: Authentication Attacks
================================
Password hash identification, entropy calculation, brute-force timing,
and credential-stuffing simulation.

IMPORTANT: For authorized security testing and CTF only.
"""

import hashlib
import math
import string
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Hash Identification
# ---------------------------------------------------------------------------

HASH_PATTERNS = {
    "MD5": (32, "hex"),
    "SHA1": (40, "hex"),
    "SHA256": (64, "hex"),
    "bcrypt": (60, "$2b$"),
    "NTLM": (32, "hex_upper"),
}


def identify_hash(h: str) -> str:
    """Identify hash algorithm from hash string."""
    if h.startswith("$2b$") or h.startswith("$2a$"):
        return "bcrypt"
    if len(h) == 32 and all(c in "0123456789ABCDEF" for c in h):
        return "NTLM"
    if len(h) == 32 and all(c in string.hexdigits for c in h):
        return "MD5"
    if len(h) == 40 and all(c in string.hexdigits for c in h):
        return "SHA1"
    if len(h) == 64 and all(c in string.hexdigits for c in h):
        return "SHA256"
    return "Unknown"


# ---------------------------------------------------------------------------
# Password Entropy
# ---------------------------------------------------------------------------

def charset_size(password: str) -> int:
    """Calculate effective charset size from password characters."""
    size = 0
    if any(c in string.ascii_lowercase for c in password):
        size += 26
    if any(c in string.ascii_uppercase for c in password):
        size += 26
    if any(c in string.digits for c in password):
        size += 10
    if any(c in string.punctuation for c in password):
        size += 32
    return size


def entropy_bits(password: str) -> float:
    """Calculate password entropy in bits."""
    cs = charset_size(password)
    if cs == 0:
        return 0.0
    return len(password) * math.log2(cs)


def crack_time_seconds(entropy: float, rate: float = 1e10) -> float:
    """Estimate crack time at given guesses/second."""
    return (2 ** entropy) / (2 * rate)


@dataclass
class PasswordAnalysis:
    password: str
    charset: int
    entropy: float
    crack_seconds: float

    @property
    def crack_human(self) -> str:
        s = self.crack_seconds
        if s < 1:
            return "instant"
        if s < 60:
            return f"{s:.1f} seconds"
        if s < 3600:
            return f"{s / 60:.1f} minutes"
        if s < 86400:
            return f"{s / 3600:.1f} hours"
        if s < 86400 * 365:
            return f"{s / 86400:.1f} days"
        return f"{s / (86400 * 365):.1f} years"


# ---------------------------------------------------------------------------
# Brute-Force Timing Calculator
# ---------------------------------------------------------------------------

def brute_force_keyspace(charset_sz: int, min_len: int, max_len: int) -> int:
    """Total keyspace for brute-force over length range."""
    return sum(charset_sz ** l for l in range(min_len, max_len + 1))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("Authentication Attacks Examples")
    print("=" * 50)

    # Hash identification
    hashes = [
        "5f4dcc3b5aa765d61d8327deb882cf99",
        "$2b$12$LJ3m4ys3Lg7E5S0Rq5Y8eOmZr7kGHs9kXaLTqVKB1HV2vGndm6a",
        "5baa61e4c9b93f3f0682250b6cf8331b7ee68fd8",
        "8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92",
    ]
    print("\nHash Identification:")
    for h in hashes:
        print(f"  {h[:40]}... -> {identify_hash(h)}")

    # Password entropy
    passwords = ["password", "P@ssw0rd!", "correcthorsebatterystaple", "xK#9mQ$2pL"]
    print("\nPassword Entropy Analysis:")
    for pw in passwords:
        e = entropy_bits(pw)
        analysis = PasswordAnalysis(pw, charset_size(pw), e, crack_time_seconds(e))
        print(f"  {pw:30s} charset={analysis.charset:3d} "
              f"entropy={analysis.entropy:5.1f} bits  crack={analysis.crack_human}")

    # Brute-force keyspace
    print("\nBrute-force Keyspace (lowercase 1-6 chars):")
    ks = brute_force_keyspace(26, 1, 6)
    print(f"  Total combinations: {ks:,}")
    print(f"  At 10B/sec: {ks / 1e10:.4f} seconds")

    # Hash computation
    print("\nMD5 Hash Demo:")
    for word in ["password", "123456", "admin"]:
        h = hashlib.md5(word.encode()).hexdigest()
        print(f"  MD5({word}) = {h}")


if __name__ == "__main__":
    demo()
