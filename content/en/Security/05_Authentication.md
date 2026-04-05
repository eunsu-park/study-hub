# 05. Authentication Systems

**Previous**: [04. TLS/SSL and Public Key Infrastructure](./04_TLS_and_PKI.md) | **Next**: [06. Authorization and Access Control](06_Authorization.md)

---

Authentication is the process of verifying that a user or system is who they claim to be. It answers the question "Who are you?" and is the foundation upon which all access control decisions rest. A poorly implemented authentication system can render even the most sophisticated authorization and encryption useless. This lesson covers password-based authentication, multi-factor authentication, token-based systems, OAuth 2.0/OIDC, session management, and biometric approaches, with practical Python examples throughout.

## Learning Objectives

- Implement secure password storage using salting and key stretching
- Understand and implement multi-factor authentication (TOTP, FIDO2/WebAuthn)
- Describe OAuth 2.0 and OpenID Connect authorization/authentication flows
- Manage sessions securely using cookies, tokens, and JWTs
- Identify and avoid common JWT pitfalls
- Design secure password reset flows
- Understand biometric authentication concepts and tradeoffs

---

## 1. Password-Based Authentication

### 1.1 The Password Problem

Passwords remain the most common authentication method despite their well-known weaknesses.

```
┌─────────────────────────────────────────────────────────────────┐
│               Password Authentication Flow                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   User                    Server                                 │
│   ┌──────┐               ┌──────────┐                           │
│   │ Form │───────────────▶│ Receive  │                          │
│   │ user │  username +    │ username │                           │
│   │ pass │  password      │ + pass   │                          │
│   └──────┘               └────┬─────┘                           │
│                                │                                 │
│                                ▼                                 │
│                         ┌──────────────┐                        │
│                         │  Hash the    │                        │
│                         │  provided    │                        │
│                         │  password    │                        │
│                         └──────┬───────┘                        │
│                                │                                 │
│                                ▼                                 │
│                    ┌────────────────────┐                       │
│                    │  Compare hash to   │                       │
│                    │  stored hash in DB │                       │
│                    └────────┬───────────┘                       │
│                             │                                    │
│                      ┌──────┴──────┐                            │
│                      │             │                             │
│                   Match?        No Match?                        │
│                      │             │                             │
│                      ▼             ▼                             │
│                  ┌────────┐   ┌────────┐                        │
│                  │ Grant  │   │ Deny   │                        │
│                  │ Access │   │ Access │                        │
│                  └────────┘   └────────┘                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Why Not Store Plaintext Passwords?

If an attacker gains access to your database (via SQL injection, backup theft, insider threat, etc.), plaintext passwords are immediately compromised. The principle of **defense in depth** requires that even a database breach does not directly expose user credentials.

```
┌──────────────────────────────────────────────────────────────┐
│  NEVER DO THIS:                                               │
│                                                               │
│  users table:                                                 │
│  ┌──────────┬────────────────┐                               │
│  │ username │ password       │                                │
│  ├──────────┼────────────────┤                                │
│  │ alice    │ MyP@ssw0rd!    │  ← Plaintext = catastrophe    │
│  │ bob      │ hunter2        │                                │
│  └──────────┴────────────────┘                               │
│                                                               │
│  INSTEAD:                                                     │
│                                                               │
│  users table:                                                 │
│  ┌──────────┬──────────────────────────────────────────────┐ │
│  │ username │ password_hash                                │  │
│  ├──────────┼──────────────────────────────────────────────┤ │
│  │ alice    │ $2b$12$LJ3m4ys3Lk0aB...  (bcrypt hash)     │  │
│  │ bob      │ $argon2id$v=19$m=65536... (argon2 hash)     │  │
│  └──────────┴──────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 1.3 Hashing, Salting, and Key Stretching

**Hashing** converts a password into a fixed-length string. But simple hashing (MD5, SHA-256) is vulnerable to rainbow tables and brute force.

**Salting** adds a unique random value to each password before hashing, defeating rainbow tables.

**Key Stretching** applies the hash function thousands or millions of times, making brute force computationally expensive.

```
┌─────────────────────────────────────────────────────────────────┐
│                  Password Hashing Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   "MyP@ssw0rd!"                                                  │
│        │                                                         │
│        ▼                                                         │
│   ┌──────────────┐                                              │
│   │  Generate    │ ──▶  salt = "x9Kp2mQ..."  (random, unique)  │
│   │  Random Salt │                                               │
│   └──────┬───────┘                                              │
│          │                                                       │
│          ▼                                                       │
│   ┌──────────────────┐                                          │
│   │  Concatenate     │ ──▶  "x9Kp2mQ..." + "MyP@ssw0rd!"      │
│   │  salt + password │                                           │
│   └──────┬───────────┘                                          │
│          │                                                       │
│          ▼                                                       │
│   ┌──────────────────┐                                          │
│   │  Key Stretching  │ ──▶  Apply hash 100,000+ iterations     │
│   │  (bcrypt/argon2) │      or memory-hard function              │
│   └──────┬───────────┘                                          │
│          │                                                       │
│          ▼                                                       │
│   "$2b$12$x9Kp2mQ.../hashed_output"                            │
│   (salt + hash stored together)                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Recommended Algorithms (in order of preference):**

| Algorithm | Type | Key Feature | Recommended Parameters |
|-----------|------|-------------|----------------------|
| Argon2id | Memory-hard | GPU/ASIC resistant | m=65536, t=3, p=4 |
| bcrypt | CPU-hard | Widely supported | cost factor 12+ |
| scrypt | Memory-hard | Good alternative | N=2^15, r=8, p=1 |
| PBKDF2 | Iteration-based | NIST approved | 600,000+ iterations (SHA-256) |

### 1.4 Python Implementation: Password Hashing

```python
"""
password_hashing.py - Secure password storage with bcrypt and argon2
"""
import bcrypt
import hashlib
import os
import secrets


# ==============================================================
# Method 1: bcrypt (most widely used)
# ==============================================================

def hash_password_bcrypt(password: str) -> str:
    """Hash a password using bcrypt with automatic salting."""
    # bcrypt automatically generates a salt and includes it in the output
    # The cost factor (rounds) controls computation time: 2^rounds iterations
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt(rounds=12)  # 2^12 = 4096 iterations
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password_bcrypt(password: str, stored_hash: str) -> bool:
    """Verify a password against a bcrypt hash."""
    password_bytes = password.encode('utf-8')
    stored_bytes = stored_hash.encode('utf-8')
    return bcrypt.checkpw(password_bytes, stored_bytes)


# ==============================================================
# Method 2: Argon2 (recommended by OWASP)
# ==============================================================

# pip install argon2-cffi
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

def hash_password_argon2(password: str) -> str:
    """Hash a password using Argon2id."""
    ph = PasswordHasher(
        time_cost=3,        # Number of iterations
        memory_cost=65536,   # 64 MB of memory
        parallelism=4,       # Number of parallel threads
        hash_len=32,         # Length of the hash output
        salt_len=16          # Length of the random salt
    )
    return ph.hash(password)


def verify_password_argon2(password: str, stored_hash: str) -> bool:
    """Verify a password against an Argon2 hash."""
    ph = PasswordHasher()
    try:
        return ph.verify(stored_hash, password)
    except VerifyMismatchError:
        return False


# ==============================================================
# Method 3: PBKDF2 (built into Python, no external deps)
# ==============================================================

def hash_password_pbkdf2(password: str) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256."""
    salt = os.urandom(32)  # 32-byte random salt
    iterations = 600_000   # OWASP recommended minimum for SHA-256

    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        iterations,
        dklen=32
    )

    # Store salt + iterations + hash together
    # Format: iterations$salt_hex$hash_hex
    return f"{iterations}${salt.hex()}${key.hex()}"


def verify_password_pbkdf2(password: str, stored: str) -> bool:
    """Verify a password against a PBKDF2 hash."""
    iterations_str, salt_hex, hash_hex = stored.split('$')
    iterations = int(iterations_str)
    salt = bytes.fromhex(salt_hex)
    stored_key = bytes.fromhex(hash_hex)

    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        iterations,
        dklen=32
    )

    # Use constant-time comparison to prevent timing attacks
    return secrets.compare_digest(key, stored_key)


# ==============================================================
# Demo
# ==============================================================

if __name__ == "__main__":
    test_password = "MySecureP@ssw0rd!"

    # bcrypt
    print("=== bcrypt ===")
    hashed = hash_password_bcrypt(test_password)
    print(f"Hash: {hashed}")
    print(f"Verify (correct): {verify_password_bcrypt(test_password, hashed)}")
    print(f"Verify (wrong):   {verify_password_bcrypt('wrong', hashed)}")

    # Argon2
    print("\n=== Argon2id ===")
    hashed = hash_password_argon2(test_password)
    print(f"Hash: {hashed}")
    print(f"Verify (correct): {verify_password_argon2(test_password, hashed)}")
    print(f"Verify (wrong):   {verify_password_argon2('wrong', hashed)}")

    # PBKDF2
    print("\n=== PBKDF2 ===")
    hashed = hash_password_pbkdf2(test_password)
    print(f"Hash: {hashed}")
    print(f"Verify (correct): {verify_password_pbkdf2(test_password, hashed)}")
    print(f"Verify (wrong):   {verify_password_pbkdf2('wrong', hashed)}")
```

### 1.5 Password Policies

Strong passwords alone are insufficient. A comprehensive password policy includes:

```
┌─────────────────────────────────────────────────────────────────┐
│               Modern Password Policy (NIST SP 800-63B)           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  DO:                                                             │
│  ✓ Minimum 8 characters (12+ recommended)                       │
│  ✓ Maximum 64+ characters allowed                               │
│  ✓ Allow all printable ASCII + Unicode characters                │
│  ✓ Check against breached password lists (haveibeenpwned.com)    │
│  ✓ Check against common passwords (password, 123456, etc.)      │
│  ✓ Allow paste into password fields (for password managers)      │
│  ✓ Show password strength meter                                  │
│                                                                  │
│  DON'T:                                                          │
│  ✗ Force arbitrary complexity rules (uppercase + number + ...)   │
│  ✗ Force periodic password rotation (unless breach suspected)    │
│  ✗ Use password hints or knowledge-based questions               │
│  ✗ Truncate passwords silently                                   │
│  ✗ Use SMS for password recovery (SIM swapping attacks)          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
"""
password_policy.py - Modern password validation per NIST guidelines
"""
import re
import hashlib
import requests
from typing import Tuple, List


# Common passwords list (top 20 - in practice, use a much larger list)
COMMON_PASSWORDS = {
    "password", "123456", "123456789", "12345678", "12345",
    "1234567", "qwerty", "abc123", "password1", "111111",
    "iloveyou", "1234567890", "123123", "admin", "letmein",
    "welcome", "monkey", "dragon", "master", "000000",
}


def check_password_strength(password: str) -> Tuple[bool, List[str]]:
    """
    Validate password against modern security guidelines.
    Returns (is_valid, list_of_issues).
    """
    issues = []

    # Length check (NIST minimum: 8, recommended: 12+)
    if len(password) < 8:
        issues.append("Password must be at least 8 characters long")
    elif len(password) < 12:
        issues.append("Warning: 12+ characters recommended for better security")

    # Common password check
    if password.lower() in COMMON_PASSWORDS:
        issues.append("This is a commonly used password")

    # Repetitive pattern check
    if re.match(r'^(.)\1+$', password):
        issues.append("Password cannot be a single repeated character")

    # Sequential pattern check
    if re.search(r'(012|123|234|345|456|567|678|789|890)', password):
        issues.append("Warning: Contains sequential number pattern")

    # Context-specific check (would include username, email, etc.)
    # In production, also check against the user's personal info

    is_valid = not any(
        not issue.startswith("Warning") for issue in issues
    )

    return is_valid, issues


def check_breached_password(password: str) -> bool:
    """
    Check if password appears in known breaches using the
    Have I Been Pwned API (k-anonymity model - only first 5
    chars of SHA-1 hash are sent).
    """
    sha1 = hashlib.sha1(password.encode('utf-8')).hexdigest().upper()
    prefix = sha1[:5]
    suffix = sha1[5:]

    try:
        response = requests.get(
            f"https://api.pwnedpasswords.com/range/{prefix}",
            timeout=5
        )
        response.raise_for_status()

        # Response contains lines of "SUFFIX:COUNT"
        for line in response.text.splitlines():
            hash_suffix, count = line.split(':')
            if hash_suffix == suffix:
                return True  # Password has been breached

        return False  # Password not found in breaches
    except requests.RequestException:
        # If API is unavailable, fail open (but log the error)
        return False


if __name__ == "__main__":
    test_passwords = [
        "short",
        "password",
        "MyStr0ng&Secure!Pass",
        "aaaaaaaaaaaa",
        "12345678",
    ]

    for pwd in test_passwords:
        is_valid, issues = check_password_strength(pwd)
        status = "PASS" if is_valid else "FAIL"
        print(f"\n[{status}] '{pwd}'")
        for issue in issues:
            print(f"  - {issue}")
```

---

## 2. Multi-Factor Authentication (MFA)

### 2.1 Authentication Factors

```
┌─────────────────────────────────────────────────────────────────┐
│                  Authentication Factors                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Factor 1: Something You KNOW                                    │
│  ├── Password                                                    │
│  ├── PIN                                                         │
│  └── Security questions (discouraged)                            │
│                                                                  │
│  Factor 2: Something You HAVE                                    │
│  ├── Smartphone (TOTP app)                                       │
│  ├── Hardware security key (YubiKey, Titan)                      │
│  ├── Smart card                                                  │
│  └── SMS (weak - SIM swapping risk)                              │
│                                                                  │
│  Factor 3: Something You ARE                                     │
│  ├── Fingerprint                                                 │
│  ├── Face recognition                                            │
│  ├── Iris scan                                                   │
│  └── Voice recognition                                           │
│                                                                  │
│  MFA = Combining 2+ different factors                            │
│  (password + TOTP = 2FA, but two passwords ≠ 2FA)               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 TOTP (Time-Based One-Time Password)

TOTP generates a short-lived code based on a shared secret and the current time. Defined in RFC 6238.

```
┌─────────────────────────────────────────────────────────────────┐
│                     TOTP Algorithm                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Setup (one-time):                                               │
│  1. Server generates random secret key (base32 encoded)          │
│  2. Server shares secret with user via QR code                   │
│  3. User scans QR code with authenticator app                    │
│                                                                  │
│  Verification (each login):                                      │
│                                                                  │
│       Shared Secret              Current Time                    │
│            │                          │                          │
│            ▼                          ▼                          │
│       ┌─────────┐            ┌──────────────┐                   │
│       │ Secret  │            │  T = floor   │                   │
│       │  Key    │            │  (time / 30) │                   │
│       └────┬────┘            └──────┬───────┘                   │
│            │                        │                            │
│            └────────┬───────────────┘                            │
│                     │                                            │
│                     ▼                                            │
│              ┌─────────────┐                                    │
│              │ HMAC-SHA1   │                                    │
│              │ (secret, T) │                                    │
│              └──────┬──────┘                                    │
│                     │                                            │
│                     ▼                                            │
│              ┌─────────────┐                                    │
│              │ Truncate to │                                    │
│              │  6 digits   │                                    │
│              └──────┬──────┘                                    │
│                     │                                            │
│                     ▼                                            │
│                  "482916"   (valid for 30 seconds)              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
"""
totp_example.py - TOTP implementation using pyotp
pip install pyotp qrcode[pil]
"""
import pyotp
import qrcode
import time
import io


class TOTPManager:
    """Manage TOTP-based two-factor authentication."""

    def __init__(self):
        self.secrets = {}  # In production, store in encrypted database

    def enroll_user(self, username: str, issuer: str = "MyApp") -> str:
        """
        Generate a TOTP secret for a new user.
        Returns the provisioning URI for QR code generation.
        """
        # Generate a random base32 secret (160 bits)
        secret = pyotp.random_base32()
        self.secrets[username] = secret

        # Generate provisioning URI for authenticator apps
        totp = pyotp.TOTP(secret)
        uri = totp.provisioning_uri(
            name=username,
            issuer_name=issuer
        )

        return uri, secret

    def generate_qr_code(self, uri: str, filename: str = "totp_qr.png"):
        """Generate a QR code image from the provisioning URI."""
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(uri)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(filename)
        print(f"QR code saved to {filename}")

    def verify_totp(self, username: str, code: str) -> bool:
        """
        Verify a TOTP code for the given user.
        Allows 1 period of clock drift (±30 seconds).
        """
        if username not in self.secrets:
            return False

        secret = self.secrets[username]
        totp = pyotp.TOTP(secret)

        # valid_window=1 allows codes from t-1 and t+1 periods
        return totp.verify(code, valid_window=1)

    def get_current_code(self, username: str) -> str:
        """Get the current TOTP code (for testing only)."""
        if username not in self.secrets:
            return None

        secret = self.secrets[username]
        totp = pyotp.TOTP(secret)
        return totp.now()


# Backup codes for account recovery
def generate_backup_codes(count: int = 10) -> list:
    """
    Generate one-time-use backup codes.
    Each code is 8 characters, alphanumeric.
    """
    import secrets
    codes = []
    for _ in range(count):
        code = secrets.token_hex(4).upper()  # 8 hex characters
        # Format as XXXX-XXXX for readability
        formatted = f"{code[:4]}-{code[4:]}"
        codes.append(formatted)
    return codes


if __name__ == "__main__":
    manager = TOTPManager()

    # Enroll user
    uri, secret = manager.enroll_user("alice@example.com")
    print(f"Secret: {secret}")
    print(f"URI: {uri}")

    # Generate current code
    current_code = manager.get_current_code("alice@example.com")
    print(f"\nCurrent TOTP code: {current_code}")

    # Verify
    print(f"Verification: {manager.verify_totp('alice@example.com', current_code)}")
    print(f"Wrong code:   {manager.verify_totp('alice@example.com', '000000')}")

    # Generate backup codes
    print("\nBackup Codes:")
    for code in generate_backup_codes():
        print(f"  {code}")
```

### 2.3 FIDO2 / WebAuthn

FIDO2 (Fast Identity Online) and its web component WebAuthn represent the strongest form of authentication available today. They use public-key cryptography and are phishing-resistant.

```
┌─────────────────────────────────────────────────────────────────┐
│                  WebAuthn Registration Flow                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Browser (Client)           Server (Relying Party)              │
│        │                          │                              │
│        │   1. Request challenge   │                              │
│        │ ──────────────────────▶  │                              │
│        │                          │                              │
│        │   2. Challenge +         │                              │
│        │      RP info + user info │                              │
│        │ ◀──────────────────────  │                              │
│        │                          │                              │
│   ┌────┴────┐                     │                              │
│   │ Browser │                     │                              │
│   │ prompts │                     │                              │
│   │ user to │                     │                              │
│   │ touch   │                     │                              │
│   │ key or  │                     │                              │
│   │ use     │                     │                              │
│   │ biometr.│                     │                              │
│   └────┬────┘                     │                              │
│        │                          │                              │
│   ┌────┴──────────────┐          │                              │
│   │ Authenticator     │          │                              │
│   │ generates new     │          │                              │
│   │ key pair:         │          │                              │
│   │ - Private key     │          │                              │
│   │   (stored in key) │          │                              │
│   │ - Public key      │          │                              │
│   │   (sent to server)│          │                              │
│   └────┬──────────────┘          │                              │
│        │                          │                              │
│        │   3. Public key +        │                              │
│        │      signed challenge    │                              │
│        │ ──────────────────────▶  │                              │
│        │                          │                              │
│        │                    ┌─────┴─────┐                       │
│        │                    │ Verify    │                        │
│        │                    │ signature │                        │
│        │                    │ Store     │                        │
│        │                    │ public key│                        │
│        │                    └─────┬─────┘                       │
│        │                          │                              │
│        │   4. Registration OK     │                              │
│        │ ◀──────────────────────  │                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key advantages of WebAuthn:**

| Feature | Passwords | TOTP | WebAuthn/FIDO2 |
|---------|-----------|------|----------------|
| Phishing resistant | No | No | Yes |
| No shared secrets on server | No | No | Yes (public key only) |
| User effort | High (memorize) | Medium (copy code) | Low (touch/biometric) |
| Replay attacks | Vulnerable | Time-limited | Not possible |
| Breach impact | High | Medium | Minimal |

---

## 3. OAuth 2.0 and OpenID Connect

### 3.1 OAuth 2.0 Overview

OAuth 2.0 is an **authorization** framework (not authentication). It allows a third-party application to access resources on behalf of a user without sharing the user's credentials.

```
┌─────────────────────────────────────────────────────────────────┐
│                    OAuth 2.0 Roles                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Resource Owner  = The user who owns the data                    │
│  Client          = The application requesting access             │
│  Authorization   = The server that authenticates the user        │
│    Server          and issues tokens (e.g., Google, GitHub)      │
│  Resource Server = The API server holding protected resources    │
│                                                                  │
│  Example:                                                        │
│  "MyApp wants to access your Google Calendar"                    │
│                                                                  │
│  Resource Owner    = You (the Google user)                       │
│  Client            = MyApp                                       │
│  Authorization     = accounts.google.com                         │
│    Server                                                        │
│  Resource Server   = calendar.googleapis.com                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Authorization Code Flow (Most Common)

This is the recommended flow for server-side web applications.

```
┌─────────────────────────────────────────────────────────────────┐
│          OAuth 2.0 Authorization Code Flow                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User        Client App       Auth Server      Resource Server   │
│   │              │                │                  │            │
│   │  1. Click    │                │                  │            │
│   │  "Login w/   │                │                  │            │
│   │   Google"    │                │                  │            │
│   │─────────────▶│                │                  │            │
│   │              │                │                  │            │
│   │              │  2. Redirect   │                  │            │
│   │              │  to auth URL   │                  │            │
│   │◀─────────────│                │                  │            │
│   │              │                │                  │            │
│   │  3. User logs in and         │                  │            │
│   │     consents to permissions  │                  │            │
│   │─────────────────────────────▶│                  │            │
│   │              │                │                  │            │
│   │  4. Redirect back with       │                  │            │
│   │     authorization code       │                  │            │
│   │◀─────────────────────────────│                  │            │
│   │              │                │                  │            │
│   │─────────────▶│                │                  │            │
│   │  (code)      │                │                  │            │
│   │              │  5. Exchange   │                  │            │
│   │              │  code for      │                  │            │
│   │              │  tokens        │                  │            │
│   │              │───────────────▶│                  │            │
│   │              │                │                  │            │
│   │              │  6. Access +   │                  │            │
│   │              │  Refresh tokens│                  │            │
│   │              │◀───────────────│                  │            │
│   │              │                │                  │            │
│   │              │  7. API request with access token │            │
│   │              │──────────────────────────────────▶│            │
│   │              │                │                  │            │
│   │              │  8. Protected resource            │            │
│   │              │◀──────────────────────────────────│            │
│   │              │                │                  │            │
│   │  9. Response │                │                  │            │
│   │◀─────────────│                │                  │            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Authorization Code Flow with PKCE

PKCE (Proof Key for Code Exchange) is **required** for public clients (SPAs, mobile apps) and recommended for all clients.

```
┌─────────────────────────────────────────────────────────────────┐
│                      PKCE Extension                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Before redirect (step 2):                                       │
│                                                                  │
│  1. Client generates random "code_verifier"                      │
│     code_verifier = random_string(43-128 chars)                  │
│                                                                  │
│  2. Client computes "code_challenge"                             │
│     code_challenge = BASE64URL(SHA256(code_verifier))            │
│                                                                  │
│  3. Client sends code_challenge in authorization request         │
│     GET /authorize?                                              │
│       response_type=code&                                        │
│       client_id=...&                                             │
│       code_challenge=...&                                        │
│       code_challenge_method=S256                                 │
│                                                                  │
│  At token exchange (step 5):                                     │
│                                                                  │
│  4. Client sends code_verifier with token request                │
│     POST /token                                                  │
│       grant_type=authorization_code&                             │
│       code=...&                                                  │
│       code_verifier=...                                          │
│                                                                  │
│  5. Server verifies:                                             │
│     BASE64URL(SHA256(code_verifier)) == stored code_challenge    │
│                                                                  │
│  Why? Prevents authorization code interception attacks.          │
│  An attacker who steals the code cannot exchange it              │
│  without the code_verifier.                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 OpenID Connect (OIDC)

OpenID Connect is an **authentication** layer built on top of OAuth 2.0. While OAuth 2.0 provides authorization ("this app can access your calendar"), OIDC provides authentication ("this user is alice@example.com").

```
┌─────────────────────────────────────────────────────────────────┐
│              OAuth 2.0 vs OpenID Connect                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  OAuth 2.0:                                                      │
│  - Purpose: Authorization (access delegation)                    │
│  - Token: Access Token (opaque or JWT)                           │
│  - Question answered: "What can this app do?"                    │
│                                                                  │
│  OpenID Connect (OIDC):                                          │
│  - Purpose: Authentication (identity verification)               │
│  - Token: ID Token (always JWT) + Access Token                   │
│  - Question answered: "Who is this user?"                        │
│  - Adds: UserInfo endpoint, standard claims (sub, email, name)   │
│  - Scope: "openid" (required), "profile", "email"               │
│                                                                  │
│  OIDC builds ON TOP of OAuth 2.0:                                │
│  ┌─────────────────────────────────┐                            │
│  │       OpenID Connect            │  ← Authentication          │
│  │  ┌──────────────────────────┐   │                            │
│  │  │      OAuth 2.0           │   │  ← Authorization           │
│  │  │  ┌───────────────────┐   │   │                            │
│  │  │  │     HTTP/TLS      │   │   │  ← Transport               │
│  │  │  └───────────────────┘   │   │                            │
│  │  └──────────────────────────┘   │                            │
│  └─────────────────────────────────┘                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.5 Python Example: OAuth 2.0 Client

```python
"""
oauth_client.py - OAuth 2.0 Authorization Code Flow with PKCE
Using the requests-oauthlib library
pip install requests-oauthlib
"""
import hashlib
import base64
import secrets
import os
from urllib.parse import urlencode, urlparse, parse_qs

from flask import Flask, redirect, request, session, jsonify
import requests


app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# OAuth 2.0 Configuration (example with GitHub)
OAUTH_CONFIG = {
    "client_id": os.environ.get("OAUTH_CLIENT_ID", "your-client-id"),
    "client_secret": os.environ.get("OAUTH_CLIENT_SECRET", "your-secret"),
    "authorize_url": "https://github.com/login/oauth/authorize",
    "token_url": "https://github.com/login/oauth/access_token",
    "userinfo_url": "https://api.github.com/user",
    "redirect_uri": "http://localhost:5000/callback",
    "scope": "read:user user:email",
}


def generate_pkce_pair():
    """Generate PKCE code_verifier and code_challenge."""
    # code_verifier: 43-128 chars, unreserved URI characters
    code_verifier = base64.urlsafe_b64encode(
        secrets.token_bytes(32)
    ).rstrip(b'=').decode('ascii')

    # code_challenge: BASE64URL(SHA256(code_verifier))
    digest = hashlib.sha256(code_verifier.encode('ascii')).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')

    return code_verifier, code_challenge


@app.route("/login")
def login():
    """Initiate OAuth 2.0 Authorization Code Flow."""
    # Generate PKCE pair
    code_verifier, code_challenge = generate_pkce_pair()

    # Generate state for CSRF protection
    state = secrets.token_urlsafe(32)

    # Store in session
    session["oauth_state"] = state
    session["code_verifier"] = code_verifier

    # Build authorization URL
    params = {
        "client_id": OAUTH_CONFIG["client_id"],
        "redirect_uri": OAUTH_CONFIG["redirect_uri"],
        "scope": OAUTH_CONFIG["scope"],
        "response_type": "code",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }

    auth_url = f"{OAUTH_CONFIG['authorize_url']}?{urlencode(params)}"
    return redirect(auth_url)


@app.route("/callback")
def callback():
    """Handle OAuth 2.0 callback with authorization code."""
    # Verify state to prevent CSRF
    if request.args.get("state") != session.get("oauth_state"):
        return "State mismatch - possible CSRF attack", 403

    # Check for errors
    if "error" in request.args:
        return f"OAuth error: {request.args['error']}", 400

    # Exchange authorization code for tokens
    code = request.args.get("code")
    token_response = requests.post(
        OAUTH_CONFIG["token_url"],
        data={
            "client_id": OAUTH_CONFIG["client_id"],
            "client_secret": OAUTH_CONFIG["client_secret"],
            "code": code,
            "redirect_uri": OAUTH_CONFIG["redirect_uri"],
            "grant_type": "authorization_code",
            "code_verifier": session.get("code_verifier"),
        },
        headers={"Accept": "application/json"},
    )

    tokens = token_response.json()
    access_token = tokens.get("access_token")

    if not access_token:
        return "Failed to obtain access token", 400

    # Fetch user info
    user_response = requests.get(
        OAUTH_CONFIG["userinfo_url"],
        headers={"Authorization": f"Bearer {access_token}"},
    )
    user_info = user_response.json()

    # Store user in session
    session["user"] = {
        "id": user_info.get("id"),
        "login": user_info.get("login"),
        "name": user_info.get("name"),
        "email": user_info.get("email"),
    }

    # Clean up OAuth state
    session.pop("oauth_state", None)
    session.pop("code_verifier", None)

    return redirect("/profile")


@app.route("/profile")
def profile():
    """Display user profile (protected route)."""
    user = session.get("user")
    if not user:
        return redirect("/login")
    return jsonify(user)


@app.route("/logout")
def logout():
    """Clear session and log out."""
    session.clear()
    return redirect("/")
```

---

## 4. Session Management

### 4.1 Server-Side Sessions (Cookie-Based)

```
┌─────────────────────────────────────────────────────────────────┐
│                Server-Side Session Flow                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Browser                          Server                         │
│     │                                │                           │
│     │  1. POST /login                │                           │
│     │  (username + password)         │                           │
│     │───────────────────────────────▶│                           │
│     │                                │                           │
│     │                          ┌─────┴─────┐                    │
│     │                          │ Validate  │                     │
│     │                          │ Create    │                     │
│     │                          │ session   │                     │
│     │                          │ ID=abc123 │                     │
│     │                          │ Store in  │                     │
│     │                          │ Redis/DB  │                     │
│     │                          └─────┬─────┘                    │
│     │                                │                           │
│     │  2. Set-Cookie:                │                           │
│     │  session_id=abc123;            │                           │
│     │  HttpOnly; Secure;             │                           │
│     │  SameSite=Lax                  │                           │
│     │◀───────────────────────────────│                           │
│     │                                │                           │
│     │  3. GET /dashboard             │                           │
│     │  Cookie: session_id=abc123     │                           │
│     │───────────────────────────────▶│                           │
│     │                          ┌─────┴─────┐                    │
│     │                          │ Look up   │                     │
│     │                          │ session   │                     │
│     │                          │ in store  │                     │
│     │                          └─────┬─────┘                    │
│     │  4. Response with user data    │                           │
│     │◀───────────────────────────────│                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Secure Cookie Attributes

```
Set-Cookie: session_id=abc123;
            HttpOnly;        ← Cannot be accessed by JavaScript (XSS protection)
            Secure;          ← Only sent over HTTPS
            SameSite=Lax;    ← CSRF protection (not sent on cross-site POST)
            Path=/;          ← Cookie scope
            Max-Age=3600;    ← Expires in 1 hour
            Domain=.app.com  ← Sent to subdomains too
```

| Attribute | Purpose | Recommended Value |
|-----------|---------|-------------------|
| `HttpOnly` | Prevent XSS from reading cookie | Always set |
| `Secure` | HTTPS only | Always set in production |
| `SameSite` | CSRF protection | `Lax` (or `Strict` for sensitive actions) |
| `Max-Age` | Session duration | 1-24 hours depending on sensitivity |
| `Path` | URL scope | `/` or specific path |

### 4.3 Session Security Best Practices

```python
"""
session_security.py - Secure session management with Flask
"""
from flask import Flask, session, request, redirect, url_for
from datetime import timedelta
import secrets
import time


app = Flask(__name__)

# Session configuration
app.config.update(
    SECRET_KEY=secrets.token_hex(32),
    SESSION_COOKIE_HTTPONLY=True,     # Prevent JavaScript access
    SESSION_COOKIE_SECURE=True,       # HTTPS only
    SESSION_COOKIE_SAMESITE='Lax',    # CSRF protection
    PERMANENT_SESSION_LIFETIME=timedelta(hours=1),  # Session timeout
    SESSION_COOKIE_NAME='__Host-session',  # __Host- prefix forces Secure+Path=/
)


@app.before_request
def check_session_security():
    """Middleware to enforce session security policies."""
    if 'user_id' not in session:
        return  # Not logged in, skip checks

    # 1. Session timeout (absolute)
    created_at = session.get('created_at', 0)
    if time.time() - created_at > 3600:  # 1 hour absolute timeout
        session.clear()
        return redirect(url_for('login'))

    # 2. Idle timeout
    last_active = session.get('last_active', 0)
    if time.time() - last_active > 900:  # 15 min idle timeout
        session.clear()
        return redirect(url_for('login'))

    # 3. Update last active time
    session['last_active'] = time.time()

    # 4. IP binding (optional - can cause issues with mobile users)
    if session.get('ip_address') != request.remote_addr:
        # Log suspicious activity
        app.logger.warning(
            f"IP change detected for user {session.get('user_id')}: "
            f"{session.get('ip_address')} -> {request.remote_addr}"
        )


def regenerate_session(user_id: int):
    """
    Regenerate session ID after authentication state change.
    Prevents session fixation attacks.
    """
    # Preserve necessary data
    old_data = dict(session)

    # Clear old session
    session.clear()

    # Create new session with fresh ID
    session['user_id'] = user_id
    session['created_at'] = time.time()
    session['last_active'] = time.time()
    session['ip_address'] = request.remote_addr
    session.permanent = True  # Use PERMANENT_SESSION_LIFETIME

    # Note: Flask automatically generates a new session ID
    # when the session is modified after being cleared


@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    # Validate credentials (simplified)
    user = authenticate(username, password)
    if user:
        # IMPORTANT: Regenerate session after login
        regenerate_session(user.id)
        return redirect(url_for('dashboard'))

    return "Invalid credentials", 401


@app.route('/logout')
def logout():
    """Properly destroy the session."""
    session.clear()
    response = redirect(url_for('login'))
    # Explicitly expire the cookie
    response.delete_cookie('__Host-session')
    return response
```

---

## 5. JSON Web Tokens (JWT)

### 5.1 JWT Structure

A JWT consists of three Base64URL-encoded parts separated by dots.

```
┌─────────────────────────────────────────────────────────────────┐
│                     JWT Structure                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.                        │
│  eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4iLCJpYXQiOjE2M.  │
│  SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c                  │
│                                                                  │
│  ┌──────────────────┐                                           │
│  │     HEADER       │  ← Algorithm + token type                 │
│  │  {               │                                            │
│  │    "alg": "HS256"│    HMAC SHA-256                           │
│  │    "typ": "JWT"  │    JSON Web Token                         │
│  │  }               │                                            │
│  └──────────────────┘                                           │
│           .                                                      │
│  ┌──────────────────┐                                           │
│  │     PAYLOAD      │  ← Claims (data)                          │
│  │  {               │                                            │
│  │    "sub": "1234" │    Subject (user ID)                      │
│  │    "name": "John"│    Custom claim                           │
│  │    "iat": 163... │    Issued at                              │
│  │    "exp": 163... │    Expiration                             │
│  │    "iss": "myapp"│    Issuer                                 │
│  │    "aud": "api"  │    Audience                               │
│  │  }               │                                            │
│  └──────────────────┘                                           │
│           .                                                      │
│  ┌──────────────────┐                                           │
│  │    SIGNATURE     │  ← Integrity verification                 │
│  │                  │                                            │
│  │  HMACSHA256(     │                                           │
│  │    base64url(    │                                            │
│  │      header) +   │                                            │
│  │    "." +         │                                            │
│  │    base64url(    │                                            │
│  │      payload),   │                                            │
│  │    secret        │                                            │
│  │  )               │                                            │
│  └──────────────────┘                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 JWT Signing Algorithms

| Algorithm | Type | Key | Use Case |
|-----------|------|-----|----------|
| HS256 | Symmetric | Shared secret | Single service (same key signs and verifies) |
| RS256 | Asymmetric | RSA key pair | Microservices (private key signs, public key verifies) |
| ES256 | Asymmetric | ECDSA key pair | Modern alternative to RS256 (smaller keys) |
| EdDSA | Asymmetric | Ed25519 pair | Best performance, smallest keys |
| **none** | **None** | **None** | **NEVER USE - critical vulnerability** |

### 5.3 Python JWT Implementation

```python
"""
jwt_auth.py - JWT creation, verification, and common patterns
pip install PyJWT cryptography
"""
import jwt
import time
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any


# ==============================================================
# Symmetric (HS256) - For single-service applications
# ==============================================================

class JWTManagerSymmetric:
    """JWT manager using HMAC-SHA256 (symmetric key)."""

    def __init__(self, secret_key: str = None):
        # In production, load from environment variable
        self.secret_key = secret_key or secrets.token_hex(32)
        self.algorithm = "HS256"

    def create_access_token(
        self,
        user_id: str,
        roles: list = None,
        expires_minutes: int = 15
    ) -> str:
        """Create a short-lived access token."""
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,           # Subject (user identifier)
            "iat": now,                # Issued at
            "exp": now + timedelta(minutes=expires_minutes),  # Expiration
            "iss": "myapp",            # Issuer
            "aud": "myapp-api",        # Audience
            "type": "access",          # Token type
            "roles": roles or [],      # User roles
            "jti": secrets.token_hex(16),  # Unique token ID (for revocation)
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(
        self,
        user_id: str,
        expires_days: int = 30
    ) -> str:
        """Create a long-lived refresh token."""
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": now + timedelta(days=expires_days),
            "iss": "myapp",
            "type": "refresh",
            "jti": secrets.token_hex(16),
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str, expected_type: str = "access") -> Dict:
        """
        Verify and decode a JWT token.
        Raises jwt.InvalidTokenError on failure.
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],  # IMPORTANT: always specify!
                issuer="myapp",
                audience="myapp-api",
                options={
                    "require": ["exp", "iat", "sub", "iss"],
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_iss": True,
                }
            )

            # Verify token type
            if payload.get("type") != expected_type:
                raise jwt.InvalidTokenError(
                    f"Expected {expected_type} token, got {payload.get('type')}"
                )

            return payload

        except jwt.ExpiredSignatureError:
            raise jwt.InvalidTokenError("Token has expired")
        except jwt.InvalidAudienceError:
            raise jwt.InvalidTokenError("Invalid audience")
        except jwt.InvalidIssuerError:
            raise jwt.InvalidTokenError("Invalid issuer")
        except jwt.DecodeError:
            raise jwt.InvalidTokenError("Token decode failed")


# ==============================================================
# Asymmetric (RS256) - For microservices
# ==============================================================

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization


class JWTManagerAsymmetric:
    """JWT manager using RSA-SHA256 (asymmetric keys)."""

    def __init__(self, private_key_pem: str = None, public_key_pem: str = None):
        if private_key_pem and public_key_pem:
            self.private_key = serialization.load_pem_private_key(
                private_key_pem.encode(), password=None
            )
            self.public_key = serialization.load_pem_public_key(
                public_key_pem.encode()
            )
        else:
            # Generate key pair for demo
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
            self.public_key = self.private_key.public_key()

        self.algorithm = "RS256"

    def create_token(self, payload: dict) -> str:
        """Sign a token with the private key."""
        return jwt.encode(payload, self.private_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> dict:
        """Verify a token with the public key."""
        return jwt.decode(
            token,
            self.public_key,
            algorithms=[self.algorithm],
        )

    def get_public_key_pem(self) -> str:
        """Export public key (share with other services)."""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()


# ==============================================================
# Token Refresh Pattern
# ==============================================================

class TokenService:
    """Complete token service with refresh flow."""

    def __init__(self):
        self.jwt_manager = JWTManagerSymmetric()
        # In production, use Redis or a database
        self.revoked_tokens = set()

    def login(self, user_id: str, roles: list) -> Dict[str, str]:
        """Issue access and refresh tokens upon login."""
        return {
            "access_token": self.jwt_manager.create_access_token(
                user_id, roles, expires_minutes=15
            ),
            "refresh_token": self.jwt_manager.create_refresh_token(
                user_id, expires_days=30
            ),
            "token_type": "Bearer",
            "expires_in": 900,  # 15 minutes in seconds
        }

    def refresh(self, refresh_token: str) -> Dict[str, str]:
        """Use refresh token to get a new access token."""
        # Verify refresh token
        payload = self.jwt_manager.verify_token(
            refresh_token, expected_type="refresh"
        )

        # Check if token has been revoked
        jti = payload.get("jti")
        if jti in self.revoked_tokens:
            raise jwt.InvalidTokenError("Token has been revoked")

        # Revoke old refresh token (rotation)
        self.revoked_tokens.add(jti)

        # Issue new token pair
        user_id = payload["sub"]
        return self.login(user_id, roles=[])  # Re-fetch roles from DB

    def revoke(self, token: str):
        """Revoke a token (logout)."""
        try:
            payload = jwt.decode(
                token,
                self.jwt_manager.secret_key,
                algorithms=["HS256"],
                options={"verify_exp": False}  # Allow revoking expired tokens
            )
            jti = payload.get("jti")
            if jti:
                self.revoked_tokens.add(jti)
        except jwt.DecodeError:
            pass  # Invalid token, nothing to revoke


# ==============================================================
# Demo
# ==============================================================

if __name__ == "__main__":
    print("=== Symmetric JWT (HS256) ===")
    manager = JWTManagerSymmetric()

    token = manager.create_access_token("user123", roles=["admin", "editor"])
    print(f"Token: {token[:50]}...")

    payload = manager.verify_token(token)
    print(f"Payload: {payload}")

    print("\n=== Token Service ===")
    service = TokenService()

    tokens = service.login("user123", ["admin"])
    print(f"Access:  {tokens['access_token'][:50]}...")
    print(f"Refresh: {tokens['refresh_token'][:50]}...")

    # Refresh
    new_tokens = service.refresh(tokens["refresh_token"])
    print(f"New access: {new_tokens['access_token'][:50]}...")

    print("\n=== Asymmetric JWT (RS256) ===")
    asym_manager = JWTManagerAsymmetric()

    token = asym_manager.create_token({
        "sub": "user123",
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
    })
    print(f"Token: {token[:50]}...")

    payload = asym_manager.verify_token(token)
    print(f"Payload: {payload}")
    print(f"\nPublic key (share with other services):")
    print(asym_manager.get_public_key_pem()[:100] + "...")
```

### 5.4 Common JWT Pitfalls

```
┌─────────────────────────────────────────────────────────────────┐
│                  JWT Security Pitfalls                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Algorithm "none" Attack                                      │
│     ─────────────────────                                        │
│     Attacker changes header to {"alg": "none"} and removes      │
│     signature. Server accepts unsigned token.                    │
│                                                                  │
│     FIX: Always specify allowed algorithms:                      │
│     jwt.decode(token, key, algorithms=["HS256"])                 │
│     NEVER use algorithms=["none"] or accept any algorithm        │
│                                                                  │
│  2. Algorithm Confusion (RS256 → HS256)                          │
│     ──────────────────────────────────                           │
│     Server uses RS256 (asymmetric). Attacker changes to HS256    │
│     and signs with the PUBLIC key. Server verifies using the     │
│     same public key as HMAC secret → valid!                      │
│                                                                  │
│     FIX: Explicitly specify expected algorithm, not just "any"   │
│     Use separate keys for symmetric/asymmetric                   │
│                                                                  │
│  3. No Expiration                                                │
│     ─────────────                                                │
│     Token without "exp" claim lives forever.                     │
│                                                                  │
│     FIX: Always set exp. Use short-lived access tokens (15 min)  │
│     with longer refresh tokens.                                  │
│                                                                  │
│  4. Sensitive Data in Payload                                    │
│     ─────────────────────────                                    │
│     JWT payload is Base64-encoded, NOT encrypted.                │
│     Anyone can decode and read it.                               │
│                                                                  │
│     FIX: Never put passwords, PII, or secrets in JWT payload    │
│     Use JWE (JSON Web Encryption) if payload must be private    │
│                                                                  │
│  5. Token Not Revocable                                          │
│     ──────────────────                                           │
│     JWTs are stateless - once issued, they remain valid          │
│     until expiration. Logout doesn't invalidate the token.       │
│                                                                  │
│     FIX: Use short expiry + token blocklist (Redis) for logout   │
│     Or use "jti" claim and track revoked token IDs               │
│                                                                  │
│  6. Storing JWT in localStorage                                  │
│     ─────────────────────────                                    │
│     localStorage is accessible by any JavaScript on the page,    │
│     making it vulnerable to XSS.                                 │
│                                                                  │
│     FIX: Store in HttpOnly cookie (immune to XSS)               │
│     Or use in-memory storage + refresh token in HttpOnly cookie  │
│                                                                  │
│  7. Weak Secret Key                                              │
│     ────────────────                                             │
│     Using short or guessable secret for HMAC signing.            │
│     Attackers can brute-force the key.                           │
│                                                                  │
│     FIX: Use at least 256 bits of entropy:                       │
│     secret = secrets.token_hex(32)  # 256 bits                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Password Reset Flows

### 6.1 Secure Password Reset Design

```
┌─────────────────────────────────────────────────────────────────┐
│              Secure Password Reset Flow                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User                      Server                     Email      │
│   │                          │                          │        │
│   │ 1. "Forgot password"     │                          │        │
│   │  (enter email)           │                          │        │
│   │─────────────────────────▶│                          │        │
│   │                          │                          │        │
│   │                    ┌─────┴─────┐                   │        │
│   │                    │ Generate  │                    │        │
│   │                    │ random    │                    │        │
│   │                    │ token     │                    │        │
│   │                    │ Store     │                    │        │
│   │                    │ hash(tkn) │                    │        │
│   │                    │ + expiry  │                    │        │
│   │                    └─────┬─────┘                   │        │
│   │                          │                          │        │
│   │ 2. "Check your email"    │  3. Send reset link     │        │
│   │  (SAME response whether  │  with token             │        │
│   │   email exists or not!)  │─────────────────────────▶│       │
│   │◀─────────────────────────│                          │        │
│   │                          │                          │        │
│   │ 4. Click link in email   │                          │        │
│   │  /reset?token=abc123     │                          │        │
│   │─────────────────────────▶│                          │        │
│   │                          │                          │        │
│   │ 5. New password form     │                          │        │
│   │◀─────────────────────────│                          │        │
│   │                          │                          │        │
│   │ 6. Submit new password   │                          │        │
│   │─────────────────────────▶│                          │        │
│   │                    ┌─────┴─────┐                   │        │
│   │                    │ Verify    │                    │        │
│   │                    │ token     │                    │        │
│   │                    │ Update    │                    │        │
│   │                    │ password  │                    │        │
│   │                    │ Invalidate│                    │        │
│   │                    │ token     │                    │        │
│   │                    │ Invalidate│                    │        │
│   │                    │ sessions  │                    │        │
│   │                    └─────┬─────┘                   │        │
│   │                          │                          │        │
│   │ 7. "Password updated"    │                          │        │
│   │◀─────────────────────────│                          │        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Implementation

```python
"""
password_reset.py - Secure password reset implementation
"""
import secrets
import hashlib
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class ResetToken:
    token_hash: str
    user_id: int
    created_at: float
    expires_at: float
    used: bool = False


class PasswordResetService:
    """Secure password reset token management."""

    TOKEN_EXPIRY_MINUTES = 30
    MAX_REQUESTS_PER_HOUR = 3

    def __init__(self):
        # In production, use a database
        self.tokens = {}  # token_hash -> ResetToken
        self.rate_limit = {}  # email -> [timestamps]

    def request_reset(self, email: str, user_id: Optional[int]) -> Optional[str]:
        """
        Generate a password reset token.
        Returns the token (to be emailed) or None if rate limited.

        SECURITY: Always return the same response to the user,
        regardless of whether the email exists.
        """
        # Rate limiting
        now = time.time()
        if email in self.rate_limit:
            recent = [t for t in self.rate_limit[email] if now - t < 3600]
            if len(recent) >= self.MAX_REQUESTS_PER_HOUR:
                return None  # Rate limited
            self.rate_limit[email] = recent
        else:
            self.rate_limit[email] = []

        self.rate_limit[email].append(now)

        # If user doesn't exist, return None silently
        # (caller should still show "check your email" message)
        if user_id is None:
            return None

        # Generate cryptographically secure token
        token = secrets.token_urlsafe(32)  # 256 bits of entropy

        # Store HASH of token (not the token itself!)
        # If database is breached, attacker can't use the hashes
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        # Invalidate any existing tokens for this user
        self.tokens = {
            h: t for h, t in self.tokens.items()
            if t.user_id != user_id
        }

        # Store new token
        self.tokens[token_hash] = ResetToken(
            token_hash=token_hash,
            user_id=user_id,
            created_at=now,
            expires_at=now + (self.TOKEN_EXPIRY_MINUTES * 60),
        )

        return token  # Send this in the email link

    def verify_and_consume_token(self, token: str) -> Optional[int]:
        """
        Verify a reset token and return the user_id.
        Token is consumed (single-use).
        Returns None if token is invalid/expired/used.
        """
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        reset_token = self.tokens.get(token_hash)
        if not reset_token:
            return None

        # Check expiration
        if time.time() > reset_token.expires_at:
            del self.tokens[token_hash]
            return None

        # Check if already used
        if reset_token.used:
            return None

        # Mark as used
        reset_token.used = True

        # Clean up
        del self.tokens[token_hash]

        return reset_token.user_id

    def cleanup_expired(self):
        """Remove expired tokens (run periodically)."""
        now = time.time()
        self.tokens = {
            h: t for h, t in self.tokens.items()
            if t.expires_at > now and not t.used
        }


# Flask route example
from flask import Flask, request, jsonify

app = Flask(__name__)
reset_service = PasswordResetService()


@app.route('/api/forgot-password', methods=['POST'])
def forgot_password():
    email = request.json.get('email', '').strip().lower()

    if not email:
        return jsonify({"error": "Email required"}), 400

    # Look up user (may return None if not found)
    user = find_user_by_email(email)  # Your DB lookup
    user_id = user.id if user else None

    token = reset_service.request_reset(email, user_id)

    if token and user:
        # Send email with reset link
        reset_link = f"https://myapp.com/reset-password?token={token}"
        send_reset_email(email, reset_link)  # Your email function

    # ALWAYS return the same response (prevent email enumeration)
    return jsonify({
        "message": "If that email is registered, you will receive a reset link."
    })


@app.route('/api/reset-password', methods=['POST'])
def reset_password():
    token = request.json.get('token')
    new_password = request.json.get('new_password')

    if not token or not new_password:
        return jsonify({"error": "Token and new password required"}), 400

    # Validate new password
    # (use password_policy.check_password_strength here)

    user_id = reset_service.verify_and_consume_token(token)
    if not user_id:
        return jsonify({"error": "Invalid or expired token"}), 400

    # Update password
    update_user_password(user_id, new_password)  # Hash and store

    # Invalidate all existing sessions for this user
    invalidate_all_sessions(user_id)

    return jsonify({"message": "Password updated successfully"})
```

**Key Security Properties:**

| Property | Implementation |
|----------|---------------|
| Token entropy | `secrets.token_urlsafe(32)` - 256 bits |
| Token storage | Store hash only, never plaintext |
| Single use | Token consumed on first use |
| Time-limited | 30-minute expiration |
| Rate limiting | Max 3 requests per hour per email |
| No enumeration | Same response whether email exists or not |
| Session invalidation | All sessions cleared after reset |

---

## 7. Biometric Authentication

### 7.1 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              Biometric Authentication Types                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Physiological:                                                  │
│  ├── Fingerprint   - Most common, mature technology              │
│  ├── Face          - Widespread on mobile (Face ID)              │
│  ├── Iris          - High accuracy, expensive                    │
│  ├── Retina        - Very high accuracy, intrusive               │
│  └── Palm/Vein     - Contactless, increasingly popular           │
│                                                                  │
│  Behavioral:                                                     │
│  ├── Voice         - Convenient for phone systems                │
│  ├── Typing rhythm - Continuous authentication                   │
│  ├── Gait          - Walking pattern recognition                 │
│  └── Signature     - Dynamic analysis (pressure, speed)          │
│                                                                  │
│  Key Metrics:                                                    │
│  ┌─────────────┬──────────────────────────────────────────┐     │
│  │ FAR         │ False Acceptance Rate                     │     │
│  │             │ (impostor accepted as genuine)            │     │
│  ├─────────────┼──────────────────────────────────────────┤     │
│  │ FRR         │ False Rejection Rate                      │     │
│  │             │ (genuine user rejected)                   │     │
│  ├─────────────┼──────────────────────────────────────────┤     │
│  │ EER         │ Equal Error Rate                          │     │
│  │             │ (where FAR = FRR; lower is better)       │     │
│  └─────────────┴──────────────────────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Biometric Template Security

Biometric data requires special handling because, unlike passwords, biometric traits **cannot be changed** if compromised.

```
┌─────────────────────────────────────────────────────────────────┐
│           Biometric Template Protection                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  WRONG: Store raw biometric data                                 │
│  ┌──────────┐    ┌──────────────┐                               │
│  │ Raw scan │───▶│ Store image  │  ← If breached, game over    │
│  └──────────┘    │ in database  │    (can't change fingerprint) │
│                  └──────────────┘                                │
│                                                                  │
│  CORRECT: Cancelable biometrics / template protection            │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐           │
│  │ Raw scan │───▶│ Extract      │───▶│ Transform   │           │
│  └──────────┘    │ features     │    │ (one-way,   │           │
│                  │ (minutiae)   │    │  cancelable)│            │
│                  └──────────────┘    └──────┬──────┘           │
│                                             │                    │
│                                      ┌──────▼──────┐           │
│                                      │ Store       │            │
│                                      │ template    │            │
│                                      │ (revocable) │            │
│                                      └─────────────┘           │
│                                                                  │
│  On-Device Processing (preferred):                               │
│  - Biometric matching happens on the device (Secure Enclave)    │
│  - Server never sees biometric data                              │
│  - Device releases a cryptographic key upon match                │
│  - This is how Apple Face ID and Touch ID work                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Biometric Tradeoffs

| Factor | Passwords | TOTP | Biometrics | FIDO2 |
|--------|-----------|------|------------|-------|
| Can be changed | Yes | Yes (re-enroll) | **No** | Yes (re-register) |
| Can be shared | Yes (bad) | Yes (bad) | Difficult | No |
| Can be forgotten | Yes | N/A | No | N/A |
| Spoofing risk | Phishing | Phishing | Presentation attack | Very low |
| Privacy concern | Low | Low | **High** | Low |
| Best used as | Primary factor | 2nd factor | 2nd factor (local) | 2nd factor or passwordless |

---

## 8. Authentication Architecture Patterns

### 8.1 Choosing the Right Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│          Authentication Pattern Decision Tree                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  What type of application?                                       │
│  │                                                               │
│  ├── Traditional web app (server-rendered)                       │
│  │   └── Use: Server-side sessions + HttpOnly cookies            │
│  │                                                               │
│  ├── SPA (React, Vue, etc.)                                      │
│  │   └── Use: OAuth 2.0 + PKCE                                  │
│  │       Store access token in memory                            │
│  │       Store refresh token in HttpOnly cookie                  │
│  │                                                               │
│  ├── Mobile app                                                  │
│  │   └── Use: OAuth 2.0 + PKCE + Secure storage                 │
│  │       (iOS Keychain, Android Keystore)                        │
│  │                                                               │
│  ├── API-to-API (service mesh)                                   │
│  │   └── Use: Client Credentials flow + mTLS                    │
│  │       Or: Service mesh (Istio) with automatic mTLS            │
│  │                                                               │
│  └── Microservices                                               │
│      └── Use: JWT (RS256) with centralized auth service          │
│          Auth service issues tokens                              │
│          Each service verifies with public key                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Centralized Authentication (Auth Service Pattern)

```
┌─────────────────────────────────────────────────────────────────┐
│         Microservices Authentication Architecture                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                    ┌──────────────┐                              │
│                    │   API        │                              │
│                    │   Gateway    │                              │
│                    └──────┬───────┘                              │
│                           │                                      │
│            ┌──────────────┼──────────────┐                      │
│            │              │              │                        │
│            ▼              ▼              ▼                        │
│     ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│     │ Service  │  │ Service  │  │ Service  │                    │
│     │    A     │  │    B     │  │    C     │                    │
│     └──────────┘  └──────────┘  └──────────┘                   │
│            │              │              │                        │
│            └──────────────┼──────────────┘                      │
│                           │                                      │
│                           ▼                                      │
│                    ┌──────────────┐                              │
│                    │  Auth        │                              │
│                    │  Service     │                              │
│                    │  ─────────── │                              │
│                    │  - Login     │                              │
│                    │  - Token     │                              │
│                    │    issuance  │                              │
│                    │  - User      │                              │
│                    │    management│                              │
│                    │  - JWKS      │                              │
│                    │    endpoint  │                              │
│                    └──────────────┘                              │
│                                                                  │
│  Flow:                                                           │
│  1. Client authenticates with Auth Service → gets JWT            │
│  2. Client sends JWT to API Gateway                              │
│  3. Gateway validates JWT signature (using public key from JWKS) │
│  4. Gateway forwards request + JWT claims to services            │
│  5. Services trust validated claims without re-validating        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Exercises

### Exercise 1: Implement Secure Password Storage

Build a complete user registration and login system:

```python
"""
Exercise: Implement the following UserService class.
Use argon2 for password hashing.
Include input validation and proper error handling.
"""

class UserService:
    def register(self, username: str, email: str, password: str) -> dict:
        """
        Register a new user.
        - Validate password strength (min 12 chars, not common)
        - Check username/email uniqueness
        - Hash password with argon2id
        - Return user info (without password hash)
        """
        pass

    def login(self, username: str, password: str) -> dict:
        """
        Authenticate a user.
        - Verify credentials
        - Implement account lockout after 5 failed attempts
        - Regenerate session on successful login
        - Return access + refresh tokens
        """
        pass

    def change_password(self, user_id: int, old_password: str,
                        new_password: str) -> bool:
        """
        Change user's password.
        - Verify old password
        - Validate new password
        - Invalidate all existing sessions
        """
        pass
```

### Exercise 2: TOTP Integration

Add TOTP-based 2FA to the UserService:

```python
"""
Exercise: Add these methods to UserService.
Use the pyotp library.
"""

class UserService:
    # ... (from Exercise 1)

    def enable_2fa(self, user_id: int) -> dict:
        """
        Enable TOTP 2FA for a user.
        Returns: {"secret": ..., "qr_uri": ..., "backup_codes": [...]}
        """
        pass

    def verify_2fa_setup(self, user_id: int, code: str) -> bool:
        """Verify initial TOTP code to confirm setup."""
        pass

    def login_2fa(self, username: str, password: str,
                  totp_code: str) -> dict:
        """
        Login with 2FA.
        - First verify password
        - Then verify TOTP code
        - Support backup codes as fallback
        """
        pass
```

### Exercise 3: JWT Security Audit

Identify and fix the security issues in this code:

```python
"""
Exercise: Find and fix ALL security issues in this JWT implementation.
"""
import jwt
import time

SECRET = "mysecret"  # Issue 1: ???

def create_token(user_id):
    payload = {
        "user_id": user_id,
        "password": get_user_password(user_id),  # Issue 2: ???
        "admin": False,
    }
    return jwt.encode(payload, SECRET)  # Issue 3: ???

def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET, algorithms=["HS256", "none"])  # Issue 4: ???
        return payload
    except:  # Issue 5: ???
        return None

def protected_route(token):
    payload = verify_token(token)
    if payload:
        if payload.get("admin"):  # Issue 6: ???
            return admin_dashboard()
        return user_dashboard(payload["user_id"])
    return "Unauthorized"
```

### Exercise 4: OAuth 2.0 Flow Implementation

Implement a complete OAuth 2.0 client with PKCE:

```python
"""
Exercise: Complete this OAuth 2.0 client implementation.
Include PKCE, state validation, and secure token storage.
"""

class OAuthClient:
    def __init__(self, client_id: str, auth_url: str,
                 token_url: str, redirect_uri: str):
        pass

    def start_auth_flow(self) -> str:
        """
        Generate the authorization URL.
        Include PKCE code_challenge and state parameter.
        Returns the URL to redirect the user to.
        """
        pass

    def handle_callback(self, callback_url: str) -> dict:
        """
        Handle the OAuth callback.
        - Verify state parameter
        - Exchange code for tokens using code_verifier
        - Return tokens
        """
        pass

    def refresh_access_token(self, refresh_token: str) -> dict:
        """Refresh an expired access token."""
        pass
```

### Exercise 5: Password Reset Security Review

Review this password reset flow and list all security issues:

```python
"""
Exercise: Identify ALL security vulnerabilities in this code.
Write a corrected version.
"""
from flask import Flask, request
import random
import string

app = Flask(__name__)
reset_codes = {}  # email -> code

@app.route('/forgot', methods=['POST'])
def forgot_password():
    email = request.form['email']
    user = db.find_user(email=email)

    if not user:
        return "Email not found", 404  # Issue: ???

    # Generate 4-digit reset code
    code = ''.join(random.choices(string.digits, k=4))  # Issue: ???
    reset_codes[email] = code  # Issue: ???

    send_email(email, f"Your reset code is: {code}")
    return "Code sent"

@app.route('/reset', methods=['POST'])
def reset_password():
    email = request.form['email']
    code = request.form['code']
    new_password = request.form['password']

    if reset_codes.get(email) == code:  # Issue: ???
        user = db.find_user(email=email)
        user.password = new_password  # Issue: ???
        db.save(user)
        return "Password updated"

    return "Invalid code", 400
```

---

## 10. Summary

```
┌─────────────────────────────────────────────────────────────────┐
│              Authentication Systems Summary                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Password Storage:                                               │
│  - Use Argon2id or bcrypt (NEVER MD5/SHA1/SHA256 alone)         │
│  - Automatic salting, key stretching                             │
│  - Follow NIST SP 800-63B guidelines                             │
│                                                                  │
│  Multi-Factor Auth:                                              │
│  - TOTP is the minimum recommended 2nd factor                   │
│  - FIDO2/WebAuthn is the gold standard (phishing-resistant)     │
│  - SMS is the weakest 2nd factor (SIM swapping)                 │
│  - Always provide backup codes for account recovery             │
│                                                                  │
│  OAuth 2.0 / OIDC:                                              │
│  - Use Authorization Code flow with PKCE                        │
│  - Always validate state parameter (CSRF)                       │
│  - OIDC adds identity layer (ID tokens) on top of OAuth         │
│                                                                  │
│  Sessions & JWT:                                                 │
│  - HttpOnly + Secure + SameSite cookies                         │
│  - Regenerate session ID after login                             │
│  - JWT: short-lived access + long-lived refresh                 │
│  - Always specify allowed algorithms in JWT verification         │
│  - Never store sensitive data in JWT payload                     │
│                                                                  │
│  Password Reset:                                                 │
│  - Cryptographically random tokens (256+ bits)                   │
│  - Store hash of token, not token itself                        │
│  - Single-use, time-limited (30 min)                            │
│  - Same response regardless of email existence                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

**Previous**: [04. TLS/SSL and Public Key Infrastructure](./04_TLS_and_PKI.md) | **Next**: [06. Authorization and Access Control](06_Authorization.md)
