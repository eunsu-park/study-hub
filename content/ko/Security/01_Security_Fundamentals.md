# Security 기초와 Threat Modeling

**다음**: [02. Cryptography 기초](./02_Cryptography_Basics.md)

---

보안 엔지니어링은 악의, 오류, 사고에 직면하여 신뢰할 수 있는 시스템을 구축하는 학문입니다. 암호학이나 TLS 같은 구체적인 기술에 들어가기 전에, 위협, 위험, 방어에 대해 논의하기 위한 공통된 용어가 필요합니다. 이 레슨은 모든 후속 레슨이 기반으로 하는 개념적 기초를 확립합니다.

**난이도**: ⭐⭐

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. CIA Triad와 확장된 보안 속성 이해하기
2. 실제 시스템에 Threat Modeling 방법론(STRIDE, DREAD) 적용하기
3. 소프트웨어 아키텍처에서 공격 표면과 위협 벡터 식별하기
4. Defense-in-depth 원칙을 사용하여 계층적 방어 설계하기
5. 시스템 설계에서 최소 권한 원칙 적용하기
6. Security by design과 security by obscurity 구별하기
7. 표준 프레임워크를 사용하여 기본 위험 평가 수행하기
8. Common Vulnerability Scoring System(CVSS) 점수 해석하기

---

## 목차

1. [CIA Triad](#1-cia-triad)
2. [확장된 보안 속성](#2-확장된-보안-속성)
3. [STRIDE를 사용한 Threat Modeling](#3-stride를-사용한-threat-modeling)
4. [DREAD를 사용한 위험 점수화](#4-dread를-사용한-위험-점수화)
5. [공격 표면과 위협 벡터](#5-공격-표면과-위협-벡터)
6. [Defense in Depth](#6-defense-in-depth)
7. [최소 권한 원칙](#7-최소-권한-원칙)
8. [Security by Design vs Security by Obscurity](#8-security-by-design-vs-security-by-obscurity)
9. [위험 평가 프레임워크](#9-위험-평가-프레임워크)
10. [Common Vulnerability Scoring System(CVSS)](#10-common-vulnerability-scoring-systemcvss)
11. [종합 예제](#11-종합-예제)
12. [연습 문제](#12-연습-문제)
13. [참고 자료](#13-참고-자료)

---

## 1. CIA Triad

CIA Triad는 정보 보안에서 가장 기본적인 모델입니다. 모든 보안 제어, 모든 공격, 모든 위험은 이 세 가지 속성을 통해 분석할 수 있습니다.

```
                         ┌───────────────────────┐
                         │   Confidentiality     │
                         │                       │
                         │   "누가 볼 수 있나?"    │
                         └───────────┬───────────┘
                                     │
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
          ┌─────────────────┐              ┌─────────────────┐
          │    Integrity    │              │  Availability   │
          │                 │◀────────────▶│                 │
          │ "신뢰할 수       │              │ "필요할 때      │
          │  있나?"         │              │  접근할 수      │
          └─────────────────┘              │  있나?"         │
                                           └─────────────────┘
```

### 1.1 Confidentiality (기밀성)

Confidentiality는 정보가 접근 권한이 있는 사람만 접근할 수 있도록 보장합니다. 기밀성 침해는 정보의 무단 공개를 의미합니다.

**기밀성에 대한 위협:**
- 네트워크 트래픽 도청(패킷 스니핑)
- 데이터베이스에 대한 무단 접근
- Social engineering(피싱)
- Shoulder surfing, 쓰레기통 뒤지기
- 내부자 위협

**기밀성을 위한 제어:**
- 암호화(저장 데이터 및 전송 데이터)
- Access control lists(ACL)
- 인증 메커니즘
- 데이터 분류 및 처리 정책
- 물리적 보안 제어

```python
# Example: Demonstrating confidentiality through encryption
from cryptography.fernet import Fernet

# Generate a secret key (only authorized parties should have this)
key = Fernet.generate_key()
cipher = Fernet(key)

# Sensitive data
secret_message = b"Patient diagnosis: confidential medical record"

# Encrypt - now only key holders can read this
encrypted = cipher.encrypt(secret_message)
print(f"Encrypted: {encrypted[:50]}...")

# Decrypt - requires the key
decrypted = cipher.decrypt(encrypted)
print(f"Decrypted: {decrypted.decode()}")

# Without the key, the data is meaningless
wrong_key = Fernet.generate_key()
wrong_cipher = Fernet(wrong_key)
try:
    wrong_cipher.decrypt(encrypted)
except Exception as e:
    print(f"Unauthorized access denied: {type(e).__name__}")
```

### 1.2 Integrity (무결성)

Integrity는 정보가 무단으로 변경되지 않았음을 보장합니다. 데이터는 정확하고 일관되며 신뢰할 수 있어야 합니다.

**무결성에 대한 위협:**
- Man-in-the-middle(MITM) 공격
- SQL injection으로 데이터베이스 레코드 수정
- 파일을 변경하는 악성 코드
- 무단 구성 변경
- Bit-rot 또는 저장소 손상

**무결성을 위한 제어:**
- 암호화 해시 함수(SHA-256)
- 디지털 서명
- Message Authentication Codes(MAC/HMAC)
- 버전 제어 시스템
- 데이터베이스 제약 조건 및 트랜잭션

```python
import hashlib
import json

def compute_integrity_hash(data: dict) -> str:
    """Compute a SHA-256 hash of structured data for integrity verification."""
    # Canonical JSON encoding ensures consistent hashing
    canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()

# Original financial record
record = {
    "account": "ACCT-001",
    "amount": 1500.00,
    "currency": "USD",
    "timestamp": "2026-01-15T10:30:00Z"
}

original_hash = compute_integrity_hash(record)
print(f"Original hash: {original_hash}")

# Verify integrity - unchanged data
verified_hash = compute_integrity_hash(record)
print(f"Verification:  {verified_hash}")
print(f"Integrity OK:  {original_hash == verified_hash}")

# Tampered record - even a tiny change completely changes the hash
tampered = record.copy()
tampered["amount"] = 1500.01  # Changed by $0.01
tampered_hash = compute_integrity_hash(tampered)
print(f"\nTampered hash: {tampered_hash}")
print(f"Integrity OK:  {original_hash == tampered_hash}")  # False
```

### 1.3 Availability (가용성)

Availability는 정보와 시스템이 권한이 있는 사용자가 필요할 때 접근 가능하도록 보장합니다.

**가용성에 대한 위협:**
- Distributed Denial of Service(DDoS) 공격
- 하드웨어 장애
- 소프트웨어 버그 및 크래시
- 자연 재해
- 랜섬웨어

**가용성을 위한 제어:**
- 이중화 및 장애 조치 시스템
- 로드 밸런싱
- 정기 백업 및 재해 복구 계획
- DDoS 완화 서비스
- 용량 계획 및 모니터링

```python
# Example: Simple health check and availability monitor
import time
import random
from dataclasses import dataclass, field
from typing import List

@dataclass
class ServiceHealthCheck:
    """Monitor service availability."""
    name: str
    endpoints: List[str]
    check_history: List[dict] = field(default_factory=list)

    def check_endpoint(self, url: str) -> dict:
        """Simulate checking an endpoint's availability."""
        # In production, this would make actual HTTP requests
        is_up = random.random() > 0.1  # 90% uptime simulation
        latency_ms = random.uniform(10, 500) if is_up else None
        return {
            "url": url,
            "status": "UP" if is_up else "DOWN",
            "latency_ms": round(latency_ms, 2) if latency_ms else None,
            "timestamp": time.time()
        }

    def check_all(self) -> dict:
        """Check all endpoints and compute availability."""
        results = [self.check_endpoint(ep) for ep in self.endpoints]
        up_count = sum(1 for r in results if r["status"] == "UP")
        availability = up_count / len(results) * 100

        summary = {
            "service": self.name,
            "total_endpoints": len(results),
            "up": up_count,
            "down": len(results) - up_count,
            "availability_pct": round(availability, 2),
            "results": results
        }
        self.check_history.append(summary)
        return summary

# Define service endpoints
service = ServiceHealthCheck(
    name="Payment API",
    endpoints=[
        "https://api.example.com/v1/health",
        "https://api-backup.example.com/v1/health",
        "https://api-eu.example.com/v1/health",
    ]
)

# Run health check
result = service.check_all()
print(f"Service: {result['service']}")
print(f"Availability: {result['availability_pct']}%")
for r in result["results"]:
    status_icon = "[OK]" if r["status"] == "UP" else "[!!]"
    latency = f"{r['latency_ms']}ms" if r["latency_ms"] else "N/A"
    print(f"  {status_icon} {r['url']} - {latency}")
```

### 1.4 CIA 트레이드오프

실제로는 세 가지 속성이 종종 서로 긴장 관계에 있습니다:

```
┌────────────────────────────────────────────────────────────────────┐
│                      CIA 트레이드오프 예시                          │
├──────────────────┬─────────────────────────────────────────────────┤
│ 시나리오         │ 트레이드오프                                     │
├──────────────────┼─────────────────────────────────────────────────┤
│ 전체 디스크      │ Confidentiality ↑  Availability ↓              │
│ 암호화           │ (복호화가 지연 추가; 키 분실 = 데이터 없음)      │
├──────────────────┼─────────────────────────────────────────────────┤
│ 데이터베이스     │ Integrity ↑  Availability ↓                    │
│ 복제             │ (동기 복제가 쓰기 속도 저하)                     │
├──────────────────┼─────────────────────────────────────────────────┤
│ 인증 없는        │ Availability ↑  Confidentiality ↓              │
│ 공개 API         │ (누구나 접근 가능; 최대 가용성)                  │
├──────────────────┼─────────────────────────────────────────────────┤
│ Air-gapped       │ Confidentiality ↑  Availability ↓              │
│ 네트워크         │ (매우 안전하지만 원격 접근 어려움)               │
└──────────────────┴─────────────────────────────────────────────────┘
```

---

## 2. 확장된 보안 속성

CIA Triad를 넘어서, 완전한 보안 모델을 위해 몇 가지 추가 속성이 필수적입니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     확장된 보안 속성                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Authentication        "당신은 당신이 주장하는 사람인가?"            │
│  ├── 알고 있는 것 (비밀번호, PIN)                                   │
│  ├── 가지고 있는 것 (토큰, 스마트 카드, 전화)                        │
│  └── 자신인 것 (지문, 얼굴, 홍채)                                   │
│                                                                     │
│  Authorization         "무엇을 할 수 있는가?"                        │
│  ├── Role-Based Access Control (RBAC)                               │
│  ├── Attribute-Based Access Control (ABAC)                          │
│  └── Mandatory Access Control (MAC)                                 │
│                                                                     │
│  Non-repudiation       "이 행위를 부인할 수 있는가?"                 │
│  ├── 디지털 서명                                                     │
│  ├── 변조 방지 저장소가 있는 감사 로그                               │
│  └── 블록체인 기반 기록                                              │
│                                                                     │
│  Accountability        "행위를 행위자까지 추적할 수 있는가?"         │
│  ├── 포괄적인 로깅                                                   │
│  ├── 세션 추적                                                       │
│  └── 포렌식 분석 기능                                                │
│                                                                     │
│  Privacy               "개인 데이터가 적절히 보호되는가?"            │
│  ├── 데이터 최소화                                                   │
│  ├── 목적 제한                                                       │
│  └── 동의 관리 (GDPR, CCPA)                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1 인증 요소

Multi-factor authentication(MFA)은 두 개 이상의 독립적인 요소를 결합합니다:

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import hashlib
import secrets
import time

class AuthFactor(Enum):
    KNOWLEDGE = "something_you_know"    # Password, PIN
    POSSESSION = "something_you_have"   # TOTP token, smart card
    INHERENCE = "something_you_are"     # Biometric

@dataclass
class AuthenticationAttempt:
    username: str
    factors_provided: list
    timestamp: float = 0.0
    ip_address: str = ""

    def __post_init__(self):
        self.timestamp = time.time()

class SimpleAuthenticator:
    """Demonstrates multi-factor authentication concepts."""

    def __init__(self):
        self.users = {}
        self.failed_attempts = {}
        self.max_attempts = 5
        self.lockout_duration = 300  # 5 minutes

    def register_user(self, username: str, password: str):
        """Register with salted password hash."""
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256(
            (salt + password).encode()
        ).hexdigest()
        self.users[username] = {
            "salt": salt,
            "password_hash": password_hash,
            "totp_secret": secrets.token_hex(20),  # For MFA
        }

    def verify_password(self, username: str, password: str) -> bool:
        """Verify knowledge factor (password)."""
        if username not in self.users:
            # Constant-time comparison to prevent user enumeration
            hashlib.sha256(b"dummy_computation").hexdigest()
            return False

        user = self.users[username]
        computed = hashlib.sha256(
            (user["salt"] + password).encode()
        ).hexdigest()
        return secrets.compare_digest(computed, user["password_hash"])

    def is_locked_out(self, username: str) -> bool:
        """Check if account is locked due to failed attempts."""
        if username not in self.failed_attempts:
            return False
        attempts = self.failed_attempts[username]
        if len(attempts) >= self.max_attempts:
            last_attempt = attempts[-1]
            if time.time() - last_attempt < self.lockout_duration:
                return True
            # Lockout expired, reset
            self.failed_attempts[username] = []
        return False

    def authenticate(self, attempt: AuthenticationAttempt) -> dict:
        """Process authentication attempt with multiple factors."""
        if self.is_locked_out(attempt.username):
            return {
                "success": False,
                "reason": "Account locked. Try again later.",
                "factors_verified": []
            }

        factors_verified = []

        for factor in attempt.factors_provided:
            if factor["type"] == AuthFactor.KNOWLEDGE:
                if self.verify_password(attempt.username, factor["value"]):
                    factors_verified.append(AuthFactor.KNOWLEDGE)
            elif factor["type"] == AuthFactor.POSSESSION:
                # In production: verify TOTP code
                factors_verified.append(AuthFactor.POSSESSION)
            elif factor["type"] == AuthFactor.INHERENCE:
                # In production: verify biometric
                factors_verified.append(AuthFactor.INHERENCE)

        success = len(factors_verified) >= 2  # Require at least 2 factors

        if not success:
            self.failed_attempts.setdefault(attempt.username, []).append(
                time.time()
            )

        return {
            "success": success,
            "reason": "MFA verified" if success else "Insufficient factors",
            "factors_verified": [f.value for f in factors_verified]
        }

# Usage
auth = SimpleAuthenticator()
auth.register_user("alice", "correct-horse-battery-staple")

# Single factor - insufficient
attempt_1fa = AuthenticationAttempt(
    username="alice",
    factors_provided=[
        {"type": AuthFactor.KNOWLEDGE, "value": "correct-horse-battery-staple"}
    ],
    ip_address="192.168.1.100"
)
result = auth.authenticate(attempt_1fa)
print(f"1FA Result: {result}")
# {'success': False, 'reason': 'Insufficient factors', ...}

# Two factors - success
attempt_2fa = AuthenticationAttempt(
    username="alice",
    factors_provided=[
        {"type": AuthFactor.KNOWLEDGE, "value": "correct-horse-battery-staple"},
        {"type": AuthFactor.POSSESSION, "value": "123456"}  # TOTP code
    ],
    ip_address="192.168.1.100"
)
result = auth.authenticate(attempt_2fa)
print(f"2FA Result: {result}")
# {'success': True, 'reason': 'MFA verified', ...}
```

---

## 3. STRIDE를 사용한 Threat Modeling

STRIDE는 Microsoft에서 개발한 위협 모델링 프레임워크입니다. 각 글자는 특정 보안 속성 위반과 매핑되는 위협 범주를 식별합니다.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        STRIDE Threat Model                           │
├────────────────┬──────────────────────┬──────────────────────────────┤
│ 위협           │ 위반 속성            │ 예시                         │
├────────────────┼──────────────────────┼──────────────────────────────┤
│ S - Spoofing   │ Authentication       │ 가짜 로그인 페이지,          │
│                │                      │ 위조된 JWT 토큰              │
├────────────────┼──────────────────────┼──────────────────────────────┤
│ T - Tampering  │ Integrity            │ 수정된 API 요청,             │
│                │                      │ SQL injection                │
├────────────────┼──────────────────────┼──────────────────────────────┤
│ R - Repudiation│ Non-repudiation      │ 트랜잭션 부인,               │
│                │                      │ 로그 삭제                    │
├────────────────┼──────────────────────┼──────────────────────────────┤
│ I - Information│ Confidentiality      │ 데이터 유출, 패킷            │
│   Disclosure   │                      │ 스니핑, 오류 메시지          │
├────────────────┼──────────────────────┼──────────────────────────────┤
│ D - Denial of  │ Availability         │ DDoS, 리소스                 │
│   Service      │                      │ 고갈, 크래시 버그            │
├────────────────┼──────────────────────┼──────────────────────────────┤
│ E - Elevation  │ Authorization        │ 권한 상승,                   │
│   of Privilege │                      │ path traversal               │
└────────────────┴──────────────────────┴──────────────────────────────┘
```

### 3.1 STRIDE 분석 프로세스

```
1단계: 시스템 분해
         │
         ▼
2단계: Data Flow Diagram(DFD) 생성
         │
         ▼
3단계: 각 요소에 STRIDE를 사용하여 위협 식별
         │
         ▼
4단계: 위협 평가 및 우선순위 지정
         │
         ▼
5단계: 완화 계획 수립
```

### 3.2 Data Flow Diagram 요소

```
┌─────────────────────────────────────────────────────────────────┐
│                    DFD 요소 타입                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐    External Entity (사용자, 외부 시스템)           │
│  │ Entity  │    - 신뢰 경계 외부                                │
│  └─────────┘    - 데이터의 출발지 또는 목적지                   │
│                                                                 │
│  ┌─────────┐    Process (데이터를 변환하는 코드)                │
│  │(Process)│    - 애플리케이션 로직                             │
│  └─────────┘    - API, 서비스, 함수                             │
│                                                                 │
│  ═══════════    Data Store (데이터베이스, 파일, 캐시)           │
│  ║ Store   ║    - 데이터가 지속되는 곳                          │
│  ═══════════    - 데이터베이스, 파일, 큐                        │
│                                                                 │
│  ──────────▶    Data Flow (요소 간 데이터 이동)                 │
│                 - HTTP 요청, API 호출, 파일 읽기                │
│                                                                 │
│  - - - - - -    Trust Boundary                                  │
│  |          |   - 서로 다른 신뢰 수준을 구분                     │
│  - - - - - -    - 네트워크 경계, 프로세스 경계                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 예시: 웹 애플리케이션 위협 모델

```
┌─────────────────────────────────────────────────────────────────────┐
│                  웹 애플리케이션 DFD                                 │
│                                                                      │
│   ┌──────────┐         HTTPS          ┌──────────────┐              │
│   │ Browser  │ ──────────────────────▶│  Web Server  │              │
│   │  (User)  │ ◀──────────────────────│  (Nginx)     │              │
│   └──────────┘                        └──────┬───────┘              │
│                                               │                      │
│   - - - - - - - - - - - - - - - - - - - - - -│- - - - - - - - - -  │
│   │    신뢰 경계 (DMZ → Internal)             │                  │  │
│   - - - - - - - - - - - - - - - - - - - - - -│- - - - - - - - - -  │
│                                               │                      │
│                                        ┌──────▼───────┐             │
│                                        │  App Server  │             │
│                                        │  (Flask/     │             │
│                                        │   Django)    │             │
│                                        └──────┬───────┘             │
│                                               │                      │
│                               ┌───────────────┼───────────────┐     │
│                               │               │               │     │
│                        ═══════▼═══════ ═══════▼═══════ ┌──────▼──┐  │
│                        ║  PostgreSQL ║ ║   Redis     ║ │ External│  │
│                        ║  (Users,   ║ ║   (Sessions,║ │ Payment │  │
│                        ║   Orders)  ║ ║   Cache)    ║ │ API     │  │
│                        ═══════════════ ═══════════════ └─────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

이제 각 요소에 STRIDE를 적용합니다:

```python
from dataclasses import dataclass
from typing import List

@dataclass
class Threat:
    category: str       # S, T, R, I, D, or E
    element: str        # Which DFD element
    description: str    # What could go wrong
    severity: str       # Critical, High, Medium, Low
    mitigation: str     # How to prevent it

def stride_analysis(system_name: str) -> List[Threat]:
    """Example STRIDE analysis for a web application."""
    threats = [
        # Spoofing threats
        Threat(
            category="Spoofing",
            element="Browser → Web Server",
            description="Attacker impersonates a legitimate user with "
                        "stolen credentials or forged session tokens",
            severity="High",
            mitigation="Implement MFA, use short-lived JWTs, "
                       "enforce strong password policies"
        ),
        Threat(
            category="Spoofing",
            element="App Server → Payment API",
            description="Attacker intercepts and replays API calls "
                        "to the payment processor",
            severity="Critical",
            mitigation="Use mutual TLS (mTLS), API key rotation, "
                       "request signing with timestamps"
        ),

        # Tampering threats
        Threat(
            category="Tampering",
            element="Browser → Web Server",
            description="Attacker modifies request parameters to change "
                        "order amounts or access other users' data",
            severity="High",
            mitigation="Server-side validation, HMAC on critical parameters, "
                       "parameterized queries"
        ),
        Threat(
            category="Tampering",
            element="PostgreSQL",
            description="SQL injection allows attacker to modify database "
                        "records directly",
            severity="Critical",
            mitigation="Prepared statements, ORM usage, input sanitization, "
                       "database-level access controls"
        ),

        # Repudiation threats
        Threat(
            category="Repudiation",
            element="App Server",
            description="User denies making a purchase; no audit trail exists",
            severity="Medium",
            mitigation="Comprehensive audit logging with timestamps, "
                       "digital signatures on transactions"
        ),

        # Information Disclosure threats
        Threat(
            category="Information Disclosure",
            element="Web Server",
            description="Verbose error messages reveal stack traces, "
                        "database schema, or internal IPs",
            severity="Medium",
            mitigation="Custom error pages, structured logging "
                       "(not exposed to clients), security headers"
        ),
        Threat(
            category="Information Disclosure",
            element="Redis",
            description="Session data stored in Redis without encryption "
                        "is readable if Redis is compromised",
            severity="High",
            mitigation="Encrypt session data, enable Redis AUTH, "
                       "network isolation for Redis"
        ),

        # Denial of Service threats
        Threat(
            category="Denial of Service",
            element="Web Server",
            description="Volumetric DDoS attack overwhelms the web server",
            severity="High",
            mitigation="CDN/WAF (e.g., Cloudflare), rate limiting, "
                       "auto-scaling infrastructure"
        ),

        # Elevation of Privilege threats
        Threat(
            category="Elevation of Privilege",
            element="App Server",
            description="Regular user accesses admin endpoints due to "
                        "missing authorization checks (IDOR/broken access control)",
            severity="Critical",
            mitigation="RBAC enforcement on every endpoint, "
                       "automated authorization testing"
        ),
    ]
    return threats

# Run analysis
threats = stride_analysis("E-Commerce Web App")
print(f"STRIDE Analysis: {len(threats)} threats identified\n")

# Summary by category
from collections import Counter
by_category = Counter(t.category for t in threats)
for cat, count in sorted(by_category.items()):
    print(f"  {cat}: {count} threats")

print()

# Critical threats
critical = [t for t in threats if t.severity == "Critical"]
print(f"Critical threats ({len(critical)}):")
for t in critical:
    print(f"  [{t.category}] {t.element}")
    print(f"    Risk: {t.description}")
    print(f"    Fix:  {t.mitigation}")
    print()
```

---

## 4. DREAD를 사용한 위험 점수화

DREAD는 위협의 우선순위를 지정하기 위한 수치 점수화 시스템을 제공합니다. 각 요소는 1-10으로 평가됩니다.

```
┌───────────────────────────────────────────────────────────────────┐
│                      DREAD 점수화 모델                            │
├──────────────────┬────────────────────────────────────────────────┤
│ 요소             │ 질문                                           │
├──────────────────┼────────────────────────────────────────────────┤
│ D - Damage       │ 공격이 얼마나 심각한가?                        │
│                  │ 1 = 사소함  10 = 완전한 시스템 침해            │
├──────────────────┼────────────────────────────────────────────────┤
│ R - Reproducibil.│ 재현하기 얼마나 쉬운가?                        │
│                  │ 1 = 매우 어려움  10 = 항상 재현 가능          │
├──────────────────┼────────────────────────────────────────────────┤
│ E - Exploitab.   │ 공격을 시작하는 데 얼마나 많은 노력이 필요한가?│
│                  │ 1 = 전문가 + 맞춤 도구  10 = 브라우저만 필요   │
├──────────────────┼────────────────────────────────────────────────┤
│ A - Affected     │ 얼마나 많은 사용자가 영향을 받는가?            │
│   Users          │ 1 = 단일 사용자  10 = 모든 사용자             │
├──────────────────┼────────────────────────────────────────────────┤
│ D - Discoverab.  │ 취약점을 찾기 얼마나 쉬운가?                   │
│                  │ 1 = 소스 코드 필요  10 = URL에서 명백         │
└──────────────────┴────────────────────────────────────────────────┘

DREAD 점수 = (D + R + E + A + D) / 5

평가:    1-3 = Low    4-6 = Medium    7-9 = High    10 = Critical
```

```python
from dataclasses import dataclass

@dataclass
class DREADScore:
    """DREAD risk scoring for a vulnerability."""
    vulnerability: str
    damage: int          # 1-10
    reproducibility: int # 1-10
    exploitability: int  # 1-10
    affected_users: int  # 1-10
    discoverability: int # 1-10

    @property
    def score(self) -> float:
        total = (self.damage + self.reproducibility +
                 self.exploitability + self.affected_users +
                 self.discoverability)
        return total / 5

    @property
    def rating(self) -> str:
        s = self.score
        if s >= 9:
            return "CRITICAL"
        elif s >= 7:
            return "HIGH"
        elif s >= 4:
            return "MEDIUM"
        else:
            return "LOW"

    def __str__(self):
        return (
            f"{self.vulnerability}\n"
            f"  D={self.damage} R={self.reproducibility} "
            f"E={self.exploitability} A={self.affected_users} "
            f"D={self.discoverability}\n"
            f"  Score: {self.score:.1f} ({self.rating})"
        )

# Score several vulnerabilities
vulns = [
    DREADScore(
        vulnerability="SQL Injection in login form",
        damage=9, reproducibility=8, exploitability=7,
        affected_users=10, discoverability=8
    ),
    DREADScore(
        vulnerability="Missing rate limiting on password reset",
        damage=5, reproducibility=10, exploitability=9,
        affected_users=6, discoverability=7
    ),
    DREADScore(
        vulnerability="XSS in user profile bio field",
        damage=6, reproducibility=9, exploitability=6,
        affected_users=4, discoverability=5
    ),
    DREADScore(
        vulnerability="Cleartext password in debug log",
        damage=8, reproducibility=3, exploitability=3,
        affected_users=2, discoverability=2
    ),
]

# Sort by score descending
vulns.sort(key=lambda v: v.score, reverse=True)

print("DREAD Vulnerability Ranking")
print("=" * 50)
for i, v in enumerate(vulns, 1):
    print(f"\n#{i}: {v}")
```

---

## 5. 공격 표면과 위협 벡터

### 5.1 공격 표면이란?

공격 표면은 공격자가 시스템에서 데이터를 입력하거나 추출하려고 시도할 수 있는 모든 지점의 합입니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    공격 표면 범주                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  네트워크 공격 표면                                                  │
│  ├── 열린 포트와 서비스 (SSH, HTTP, SMTP)                           │
│  ├── API 엔드포인트                                                  │
│  ├── DNS 구성                                                        │
│  └── 네트워크 프로토콜 (TCP, UDP, ICMP)                             │
│                                                                     │
│  소프트웨어 공격 표면                                                │
│  ├── 웹 애플리케이션 입력 (폼, URL, 헤더, 쿠키)                      │
│  ├── 파일 업로드 기능                                                │
│  ├── 서드파티 의존성 (npm, pip 패키지)                               │
│  ├── 데이터베이스 인터페이스                                         │
│  └── IPC 메커니즘 (파이프, 소켓, 공유 메모리)                        │
│                                                                     │
│  물리적 공격 표면                                                    │
│  ├── 서버의 USB 포트                                                 │
│  ├── 데이터 센터에 대한 물리적 접근                                  │
│  ├── 이동식 미디어                                                   │
│  └── 하드웨어 임플란트                                               │
│                                                                     │
│  인적 공격 표면                                                      │
│  ├── 피싱 대상 (접근 권한이 있는 직원)                               │
│  ├── Social engineering (헬프 데스크, 지원)                         │
│  ├── 내부자 위협                                                     │
│  └── 서드파티 벤더 및 계약자                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 공격 표면 열거

```python
import json
from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum

class SurfaceType(Enum):
    NETWORK = "network"
    SOFTWARE = "software"
    PHYSICAL = "physical"
    HUMAN = "human"

class RiskLevel(Enum):
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1

@dataclass
class AttackSurfaceEntry:
    name: str
    surface_type: SurfaceType
    description: str
    risk_level: RiskLevel
    exposed_to: str           # Who can reach this?
    current_controls: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class AttackSurfaceReport:
    system_name: str
    entries: List[AttackSurfaceEntry] = field(default_factory=list)

    def add(self, entry: AttackSurfaceEntry):
        self.entries.append(entry)

    def summary(self) -> Dict:
        by_type = {}
        by_risk = {}
        for entry in self.entries:
            by_type[entry.surface_type.value] = (
                by_type.get(entry.surface_type.value, 0) + 1
            )
            by_risk[entry.risk_level.name] = (
                by_risk.get(entry.risk_level.name, 0) + 1
            )
        return {"by_type": by_type, "by_risk": by_risk, "total": len(self.entries)}

    def print_report(self):
        print(f"Attack Surface Report: {self.system_name}")
        print("=" * 60)

        summary = self.summary()
        print(f"\nTotal attack surface entries: {summary['total']}")
        print(f"By type: {json.dumps(summary['by_type'], indent=2)}")
        print(f"By risk: {json.dumps(summary['by_risk'], indent=2)}")

        # Show critical/high items
        urgent = [e for e in self.entries
                  if e.risk_level.value >= RiskLevel.HIGH.value]
        if urgent:
            print(f"\nHigh/Critical Items ({len(urgent)}):")
            for e in sorted(urgent, key=lambda x: x.risk_level.value, reverse=True):
                print(f"  [{e.risk_level.name}] {e.name} ({e.surface_type.value})")
                print(f"    {e.description}")
                if e.recommendations:
                    print(f"    Recommendation: {e.recommendations[0]}")

# Build an attack surface report
report = AttackSurfaceReport("Corporate Web Portal")

report.add(AttackSurfaceEntry(
    name="Public API (port 443)",
    surface_type=SurfaceType.NETWORK,
    description="REST API serving 50+ endpoints to internet",
    risk_level=RiskLevel.HIGH,
    exposed_to="Internet",
    current_controls=["WAF", "Rate limiting", "JWT auth"],
    recommendations=["Add API gateway with stricter rate limits"]
))

report.add(AttackSurfaceEntry(
    name="Admin panel (/admin)",
    surface_type=SurfaceType.SOFTWARE,
    description="Django admin interface accessible on same domain",
    risk_level=RiskLevel.CRITICAL,
    exposed_to="Internet (by misconfiguration)",
    current_controls=["Username/password"],
    recommendations=["Restrict to VPN only", "Add MFA", "Separate domain"]
))

report.add(AttackSurfaceEntry(
    name="npm dependencies (347 packages)",
    surface_type=SurfaceType.SOFTWARE,
    description="Third-party JavaScript packages in frontend build",
    risk_level=RiskLevel.MEDIUM,
    exposed_to="Supply chain",
    current_controls=["npm audit in CI"],
    recommendations=["Pin dependency versions", "Use lockfile", "Add Snyk/Dependabot"]
))

report.add(AttackSurfaceEntry(
    name="Employee email accounts",
    surface_type=SurfaceType.HUMAN,
    description="150 employees with corporate email, potential phishing targets",
    risk_level=RiskLevel.HIGH,
    exposed_to="Internet (email)",
    current_controls=["Spam filter", "Annual security training"],
    recommendations=["Quarterly phishing simulations", "DMARC/DKIM enforcement"]
))

report.print_report()
```

### 5.3 일반적인 위협 벡터

```
┌─────────────────────────────────────────────────────────────────────┐
│                      일반적인 위협 벡터                              │
├──────────────────┬──────────────────────────────────────────────────┤
│ 벡터             │ 설명 및 예시                                     │
├──────────────────┼──────────────────────────────────────────────────┤
│ 네트워크 기반    │ • 포트 스캐닝 및 익스플로잇                      │
│                  │ • Man-in-the-middle 공격                        │
│                  │ • DNS 스푸핑/중독                                │
│                  │ • ARP 스푸핑                                     │
├──────────────────┼──────────────────────────────────────────────────┤
│ 웹 애플리케이션  │ • SQL injection (SQLi)                          │
│                  │ • Cross-site scripting (XSS)                     │
│                  │ • Cross-site request forgery (CSRF)              │
│                  │ • Server-side request forgery (SSRF)             │
│                  │ • Insecure direct object references (IDOR)       │
├──────────────────┼──────────────────────────────────────────────────┤
│ 공급망           │ • 손상된 의존성 (SolarWinds, log4j)             │
│                  │ • 패키지 레지스트리의 Typosquatting              │
│                  │ • 빌드 파이프라인의 악성 코드                    │
├──────────────────┼──────────────────────────────────────────────────┤
│ Social engineer. │ • 피싱 (이메일, SMS, 음성)                       │
│                  │ • Pretexting과 baiting                          │
│                  │ • Tailgating / piggybacking                     │
├──────────────────┼──────────────────────────────────────────────────┤
│ 물리적           │ • USB 드롭 공격                                  │
│                  │ • Evil maid 공격                                 │
│                  │ • 하드웨어 임플란트                              │
├──────────────────┼──────────────────────────────────────────────────┤
│ 내부자           │ • 불만을 품은 직원                               │
│                  │ • 우발적 데이터 노출                             │
│                  │ • 자격 증명 공유                                 │
└──────────────────┴──────────────────────────────────────────────────┘
```

---

## 6. Defense in Depth

Defense in depth는 정보 시스템 전체에 걸쳐 여러 계층의 보안 제어를 사용하는 전략입니다. 한 계층이 실패하면 후속 계층이 계속 보호를 제공합니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Defense in Depth 계층                             │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Layer 7: DATA                                               │    │
│  │  저장 데이터 암호화, DLP, 데이터 분류, 백업                  │    │
│  │  ┌─────────────────────────────────────────────────────┐    │    │
│  │  │  Layer 6: APPLICATION                                │    │    │
│  │  │  입력 검증, WAF, 안전한 코딩, SAST/DAST              │    │    │
│  │  │  ┌─────────────────────────────────────────────┐    │    │    │
│  │  │  │  Layer 5: HOST                               │    │    │    │
│  │  │  │  OS 강화, AV/EDR, 패칭, FIM                  │    │    │    │
│  │  │  │  ┌─────────────────────────────────────┐    │    │    │    │
│  │  │  │  │  Layer 4: INTERNAL NETWORK           │    │    │    │    │
│  │  │  │  │  세그먼테이션, VLAN, IDS/IPS, NTA    │    │    │    │    │
│  │  │  │  │  ┌─────────────────────────────┐    │    │    │    │    │
│  │  │  │  │  │  Layer 3: PERIMETER          │    │    │    │    │    │
│  │  │  │  │  │  방화벽, DMZ, VPN, 프록시    │    │    │    │    │    │
│  │  │  │  │  │  ┌─────────────────────┐    │    │    │    │    │    │
│  │  │  │  │  │  │  Layer 2: PHYSICAL   │    │    │    │    │    │    │
│  │  │  │  │  │  │  경비, 자물쇠, CCTV  │    │    │    │    │    │    │
│  │  │  │  │  │  │  ┌─────────────┐    │    │    │    │    │    │    │
│  │  │  │  │  │  │  │ Layer 1:    │    │    │    │    │    │    │    │
│  │  │  │  │  │  │  │ POLICIES    │    │    │    │    │    │    │    │
│  │  │  │  │  │  │  │ & TRAINING  │    │    │    │    │    │    │    │
│  │  │  │  │  │  │  └─────────────┘    │    │    │    │    │    │    │
│  │  │  │  │  │  └─────────────────────┘    │    │    │    │    │    │
│  │  │  │  │  └─────────────────────────────┘    │    │    │    │    │
│  │  │  │  └─────────────────────────────────────┘    │    │    │    │
│  │  │  └─────────────────────────────────────────────┘    │    │    │
│  │  └─────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.1 계층화된 보안 구현

```python
from dataclasses import dataclass, field
from typing import List, Callable
from enum import Enum
import re

class SecurityAction(Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    LOG = "LOG"
    ALERT = "ALERT"

@dataclass
class SecurityEvent:
    source_ip: str
    request_path: str
    method: str
    headers: dict
    body: str
    user: str = ""
    blocked_by: str = ""
    actions_taken: List[str] = field(default_factory=list)

@dataclass
class SecurityLayer:
    """A single layer in the defense-in-depth chain."""
    name: str
    order: int
    check: Callable  # Function that returns (SecurityAction, reason)

class DefenseInDepth:
    """Chain of security layers processed in order."""

    def __init__(self):
        self.layers: List[SecurityLayer] = []

    def add_layer(self, layer: SecurityLayer):
        self.layers.append(layer)
        self.layers.sort(key=lambda l: l.order)

    def process(self, event: SecurityEvent) -> SecurityEvent:
        """Process an event through all security layers."""
        for layer in self.layers:
            action, reason = layer.check(event)
            event.actions_taken.append(f"[{layer.name}] {action.value}: {reason}")

            if action == SecurityAction.BLOCK:
                event.blocked_by = layer.name
                return event

        return event

# Define security layers as check functions

def rate_limiter(event: SecurityEvent):
    """Layer 1: Rate limiting (perimeter)."""
    # Simplified - in production, track request counts per IP
    suspicious_ips = {"10.0.0.99", "192.168.1.200"}
    if event.source_ip in suspicious_ips:
        return SecurityAction.BLOCK, f"IP {event.source_ip} rate limited"
    return SecurityAction.ALLOW, "Rate check passed"

def waf_check(event: SecurityEvent):
    """Layer 2: Web Application Firewall."""
    # Check for common attack patterns
    sqli_patterns = [
        r"(\b(union|select|insert|update|delete|drop)\b.*\b(from|into|table)\b)",
        r"(--|#|/\*)",
        r"(\bor\b\s+\b\d+\b\s*=\s*\b\d+\b)",
    ]
    xss_patterns = [
        r"<script[^>]*>",
        r"javascript:",
        r"on(load|error|click)\s*=",
    ]

    full_input = f"{event.request_path} {event.body}".lower()

    for pattern in sqli_patterns:
        if re.search(pattern, full_input, re.IGNORECASE):
            return SecurityAction.BLOCK, f"SQLi pattern detected: {pattern}"

    for pattern in xss_patterns:
        if re.search(pattern, full_input, re.IGNORECASE):
            return SecurityAction.BLOCK, f"XSS pattern detected: {pattern}"

    return SecurityAction.ALLOW, "WAF check passed"

def auth_check(event: SecurityEvent):
    """Layer 3: Authentication verification."""
    auth_header = event.headers.get("Authorization", "")
    if not auth_header:
        if event.request_path.startswith("/api/"):
            return SecurityAction.BLOCK, "Missing authentication for API endpoint"
    return SecurityAction.ALLOW, "Auth check passed"

def input_validation(event: SecurityEvent):
    """Layer 4: Application-level input validation."""
    if len(event.body) > 10000:
        return SecurityAction.BLOCK, "Request body exceeds maximum size"
    if "../" in event.request_path:
        return SecurityAction.BLOCK, "Path traversal attempt detected"
    return SecurityAction.ALLOW, "Input validation passed"

# Assemble defense layers
defense = DefenseInDepth()
defense.add_layer(SecurityLayer("Rate Limiter", 1, rate_limiter))
defense.add_layer(SecurityLayer("WAF", 2, waf_check))
defense.add_layer(SecurityLayer("Auth Check", 3, auth_check))
defense.add_layer(SecurityLayer("Input Validation", 4, input_validation))

# Test with various events
events = [
    SecurityEvent(
        source_ip="203.0.113.50",
        request_path="/api/users/1",
        method="GET",
        headers={"Authorization": "Bearer eyJ..."},
        body=""
    ),
    SecurityEvent(
        source_ip="203.0.113.51",
        request_path="/api/users",
        method="POST",
        headers={"Authorization": "Bearer eyJ..."},
        body="name=test' OR 1=1 --"
    ),
    SecurityEvent(
        source_ip="203.0.113.52",
        request_path="/api/files/../../../etc/passwd",
        method="GET",
        headers={"Authorization": "Bearer eyJ..."},
        body=""
    ),
]

for i, event in enumerate(events, 1):
    result = defense.process(event)
    status = "BLOCKED" if result.blocked_by else "ALLOWED"
    print(f"\nEvent #{i}: {result.method} {result.request_path} -> {status}")
    for action in result.actions_taken:
        print(f"  {action}")
    if result.blocked_by:
        print(f"  >>> Blocked by: {result.blocked_by}")
```

---

## 7. 최소 권한 원칙

최소 권한 원칙(PoLP)은 모든 모듈, 사용자 또는 프로세스가 작업을 완료하는 데 필요한 최소한의 권한 집합만 사용하여 작동해야 한다고 명시합니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│              최소 권한 원칙 - 예시                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  나쁜 예 (과도한 권한):                                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  웹 앱 서비스 계정                                           │   │
│  │  ├── DB: root 접근 (DROP DATABASE 가능)                     │   │
│  │  ├── 파일: 전체 파일시스템 읽기/쓰기                         │   │
│  │  ├── 네트워크: 제한 없는 아웃바운드                          │   │
│  │  └── 프로세스: root로 무엇이든 실행 가능                     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  좋은 예 (최소 권한):                                                │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  웹 앱 서비스 계정                                           │   │
│  │  ├── DB: app_db.users, app_db.orders에만 SELECT, INSERT     │   │
│  │  ├── 파일: /app/static 읽기 전용, /app/uploads 쓰기 전용    │   │
│  │  ├── 네트워크: payment-api.internal:443으로만 아웃바운드    │   │
│  │  └── 프로세스: 비특권 사용자로 실행 (uid 1000)              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.1 데이터베이스 접근에서의 최소 권한

```sql
-- BAD: Application uses root/superuser
-- GRANT ALL PRIVILEGES ON *.* TO 'webapp'@'%';

-- GOOD: Granular permissions per function
-- Read-only user for reporting
CREATE USER 'webapp_reader'@'10.0.0.%' IDENTIFIED BY 'strong_random_pw';
GRANT SELECT ON app_db.users TO 'webapp_reader'@'10.0.0.%';
GRANT SELECT ON app_db.orders TO 'webapp_reader'@'10.0.0.%';
GRANT SELECT ON app_db.products TO 'webapp_reader'@'10.0.0.%';

-- Write user for the application (limited tables and operations)
CREATE USER 'webapp_writer'@'10.0.0.%' IDENTIFIED BY 'another_strong_pw';
GRANT SELECT, INSERT, UPDATE ON app_db.users TO 'webapp_writer'@'10.0.0.%';
GRANT SELECT, INSERT, UPDATE ON app_db.orders TO 'webapp_writer'@'10.0.0.%';
GRANT SELECT ON app_db.products TO 'webapp_writer'@'10.0.0.%';
-- Note: No DELETE, no DROP, no ALTER, no access to other databases

-- Admin user (separate account, not used by the application)
CREATE USER 'db_admin'@'10.0.0.1' IDENTIFIED BY 'very_strong_admin_pw';
GRANT ALL PRIVILEGES ON app_db.* TO 'db_admin'@'10.0.0.1';
-- Only accessible from a single bastion host IP
```

### 7.2 코드에서의 최소 권한

```python
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class SandboxedFileAccess:
    """Enforce least privilege for file system access."""
    allowed_read_dirs: list
    allowed_write_dirs: list
    max_file_size: int = 10 * 1024 * 1024  # 10 MB default

    def _is_within_allowed_dirs(self, path: Path, dirs: list) -> bool:
        """Check if a resolved path falls within allowed directories."""
        resolved = path.resolve()
        return any(
            str(resolved).startswith(str(Path(d).resolve()))
            for d in dirs
        )

    def read_file(self, filepath: str) -> Optional[str]:
        """Read a file only if it's within allowed read directories."""
        path = Path(filepath)

        # Prevent path traversal
        if ".." in path.parts:
            raise PermissionError(f"Path traversal not allowed: {filepath}")

        if not self._is_within_allowed_dirs(path, self.allowed_read_dirs):
            raise PermissionError(
                f"Read access denied: {filepath} is outside allowed directories"
            )

        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        return path.read_text()

    def write_file(self, filepath: str, content: str) -> bool:
        """Write a file only if it's within allowed write directories."""
        path = Path(filepath)

        if ".." in path.parts:
            raise PermissionError(f"Path traversal not allowed: {filepath}")

        if not self._is_within_allowed_dirs(path, self.allowed_write_dirs):
            raise PermissionError(
                f"Write access denied: {filepath} is outside allowed directories"
            )

        if len(content.encode()) > self.max_file_size:
            raise ValueError(
                f"File size {len(content.encode())} exceeds max {self.max_file_size}"
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return True

# Usage: application can only read from /app/config and write to /app/uploads
sandbox = SandboxedFileAccess(
    allowed_read_dirs=["/app/config", "/app/static"],
    allowed_write_dirs=["/app/uploads"],
    max_file_size=5 * 1024 * 1024  # 5 MB
)

# This would succeed (within allowed read directory)
# content = sandbox.read_file("/app/config/settings.json")

# This would raise PermissionError (outside allowed directories)
try:
    sandbox.read_file("/etc/passwd")
except PermissionError as e:
    print(f"Blocked: {e}")

# This would raise PermissionError (path traversal attempt)
try:
    sandbox.read_file("/app/config/../../etc/shadow")
except PermissionError as e:
    print(f"Blocked: {e}")
```

### 7.3 클라우드에서의 최소 권한 (AWS IAM 예시)

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowS3ReadSpecificBucket",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::my-app-uploads",
                "arn:aws:s3:::my-app-uploads/*"
            ]
        },
        {
            "Sid": "AllowDynamoDBReadWrite",
            "Effect": "Allow",
            "Action": [
                "dynamodb:GetItem",
                "dynamodb:PutItem",
                "dynamodb:Query"
            ],
            "Resource": "arn:aws:dynamodb:us-east-1:123456789:table/users"
        },
        {
            "Sid": "DenyAllElse",
            "Effect": "Deny",
            "Action": "*",
            "NotResource": [
                "arn:aws:s3:::my-app-uploads",
                "arn:aws:s3:::my-app-uploads/*",
                "arn:aws:dynamodb:us-east-1:123456789:table/users"
            ]
        }
    ]
}
```

---

## 8. Security by Design vs Security by Obscurity

### 8.1 Security by Design

Security by design은 나중에 추가하는 것이 아니라 처음부터 시스템에 보안을 구축하는 것을 의미합니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Security by Design 원칙                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 공격 표면 최소화                                                 │
│     └── 불필요한 기능, 포트, 서비스 제거                             │
│                                                                      │
│  2. 안전한 기본값 설정                                               │
│     └── 기본 구성이 가장 안전해야 함                                 │
│                                                                      │
│  3. 최소 권한 원칙                                                   │
│     └── 필요한 최소 접근 권한 부여                                   │
│                                                                      │
│  4. Defense in Depth                                                 │
│     └── 여러 독립적인 보안 계층                                      │
│                                                                      │
│  5. 안전하게 실패                                                    │
│     └── 오류가 보안 취약점을 만들어서는 안 됨                        │
│                                                                      │
│  6. 서비스를 신뢰하지 마라                                           │
│     └── 모든 외부 입력 및 응답 검증                                  │
│                                                                      │
│  7. 직무 분리                                                        │
│     └── 단일 사람/프로세스가 모든 것을 제어하지 않음                 │
│                                                                      │
│  8. 보안을 단순하게 유지                                             │
│     └── 복잡한 보안은 종종 우회되거나 잘못 구성됨                    │
│                                                                      │
│  9. 문제를 올바르게 수정                                             │
│     └── 증상이 아닌 근본 원인 해결                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Kerckhoffs' Principle (1883):** 암호화 시스템은 키를 제외한 시스템의 모든 것이 공개되어 있어도 안전해야 합니다. 이것이 security by design의 금본위제입니다.

```python
# Security by Design: Fail-secure examples

def get_user_role(user_id: int, db) -> str:
    """
    Fail-secure: if anything goes wrong, return the most restrictive role.
    Never fail into a privileged state.
    """
    try:
        user = db.get_user(user_id)
        if user and user.role:
            return user.role
    except Exception:
        pass  # Log this in production

    # Fail secure: default to no access
    return "none"


def is_authorized(user_role: str, required_role: str) -> bool:
    """
    Security by design: explicit allowlist, not blocklist.
    """
    role_hierarchy = {
        "admin": 3,
        "editor": 2,
        "viewer": 1,
        "none": 0,
    }

    # Unknown roles get no access (fail secure)
    user_level = role_hierarchy.get(user_role, 0)
    required_level = role_hierarchy.get(required_role, 999)

    return user_level >= required_level


# Secure defaults in configuration
SECURE_DEFAULTS = {
    "session_timeout_minutes": 15,       # Short timeout
    "max_login_attempts": 5,             # Lockout after failures
    "password_min_length": 12,           # Strong passwords
    "require_mfa": True,                 # MFA on by default
    "cors_origins": [],                  # No CORS by default (deny all)
    "debug_mode": False,                 # Never expose debug info
    "tls_min_version": "1.2",           # No legacy TLS
    "cookie_secure": True,               # HTTPS-only cookies
    "cookie_httponly": True,             # No JS access to cookies
    "cookie_samesite": "Strict",         # Prevent CSRF
    "content_security_policy": "default-src 'self'",  # Strict CSP
    "x_frame_options": "DENY",           # Prevent clickjacking
    "hsts_max_age": 31536000,           # 1 year HSTS
}
```

### 8.2 Security by Obscurity (그리고 실패하는 이유)

Security by obscurity는 설계의 건전성이 아닌 구현의 비밀성에 의존합니다. 단독으로는 유효한 보안 전략이 아닙니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│          Security by Obscurity - 일반적인 (나쁜) 예시               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  "아무도 /xK9mQ2p의 관리자 패널을 찾지 못할 것이다"                 │
│    → 디렉토리 브루트포싱, 로그 또는 브라우저 히스토리로             │
│      쉽게 발견됨                                                     │
│                                                                      │
│  "우리의 맞춤 암호화 알고리즘은 비밀이므로 안전하다"                │
│    → 리버스 엔지니어링되거나 유출됨; 피어 리뷰 없으면 버그           │
│                                                                      │
│  "서버 버전 헤더를 숨긴다"                                           │
│    → 핑거프린팅으로 여전히 서버 소프트웨어 식별 가능                │
│                                                                      │
│  "데이터베이스 포트를 5432 대신 54321로 한다"                       │
│    → 포트 스캐너가 65535개 포트를 몇 초 만에 검사                   │
│                                                                      │
│  "API 키가 난독화된 JavaScript에 포함되어 있다"                     │
│    → 브라우저 개발자 도구로 모든 것이 즉시 노출됨                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
│                                                                      │
│  참고: Obscurity는 보조 계층(defense in depth)이 될 수 있지만      │
│  주요 보안 메커니즘이 되어서는 안 됩니다.                           │
│                                                                      │
│  Obscurity가 한계적 가치를 더하는 예:                               │
│  - 기본 SSH 포트 변경 (노이즈 감소, 실제 공격은 아님)               │
│  - 버전 배너 제거 (자동화 스캐너를 약간 느리게 함)                  │
│  - 비표준 디렉토리 이름 사용 (일반적인 탐색 억제)                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. 위험 평가 프레임워크

### 9.1 정성적 위험 평가

```
위험 = 가능성 × 영향

┌─────────────────────────────────────────────────────────────────────┐
│                    위험 평가 매트릭스 (5×5)                          │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────────┤
│          │ 무시할 수│  경미한  │  중간    │  주요    │ 치명적       │
│          │ 있는 영향│  영향    │  영향    │  영향    │ 영향         │
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────────┤
│ 거의     │  Medium  │   High   │   High   │ Critical │  Critical    │
│ 확실함   │          │          │          │          │              │
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────────┤
│ 가능성   │   Low    │  Medium  │   High   │   High   │  Critical    │
│ 높음     │          │          │          │          │              │
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────────┤
│ 가능함   │   Low    │  Medium  │  Medium  │   High   │   High       │
│          │          │          │          │          │              │
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────────┤
│ 가능성   │   Low    │   Low    │  Medium  │  Medium  │   High       │
│ 낮음     │          │          │          │          │              │
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────────┤
│ 드묾     │   Low    │   Low    │   Low    │  Medium  │  Medium      │
│          │          │          │          │          │              │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────────┘
```

### 9.2 코드로 하는 위험 평가

```python
from dataclasses import dataclass
from enum import IntEnum
from typing import List

class Likelihood(IntEnum):
    RARE = 1
    UNLIKELY = 2
    POSSIBLE = 3
    LIKELY = 4
    ALMOST_CERTAIN = 5

class Impact(IntEnum):
    NEGLIGIBLE = 1
    MINOR = 2
    MODERATE = 3
    MAJOR = 4
    CATASTROPHIC = 5

@dataclass
class Risk:
    name: str
    description: str
    likelihood: Likelihood
    impact: Impact
    existing_controls: List[str]
    recommended_actions: List[str]

    @property
    def score(self) -> int:
        return self.likelihood.value * self.impact.value

    @property
    def rating(self) -> str:
        s = self.score
        if s >= 20:
            return "CRITICAL"
        elif s >= 12:
            return "HIGH"
        elif s >= 6:
            return "MEDIUM"
        else:
            return "LOW"

    @property
    def response_strategy(self) -> str:
        """Determine risk response based on rating."""
        strategies = {
            "CRITICAL": "Mitigate immediately - stop and fix",
            "HIGH": "Mitigate soon - plan remediation within sprint",
            "MEDIUM": "Accept with monitoring, or mitigate in backlog",
            "LOW": "Accept - document and monitor"
        }
        return strategies[self.rating]

def risk_assessment(risks: List[Risk]) -> None:
    """Perform and display a risk assessment."""
    print("=" * 70)
    print("RISK ASSESSMENT REPORT")
    print("=" * 70)

    # Sort by score (highest risk first)
    sorted_risks = sorted(risks, key=lambda r: r.score, reverse=True)

    for i, risk in enumerate(sorted_risks, 1):
        print(f"\n{'─' * 70}")
        print(f"Risk #{i}: {risk.name}")
        print(f"  Description:  {risk.description}")
        print(f"  Likelihood:   {risk.likelihood.name} ({risk.likelihood.value})")
        print(f"  Impact:       {risk.impact.name} ({risk.impact.value})")
        print(f"  Score:        {risk.score} / 25")
        print(f"  Rating:       {risk.rating}")
        print(f"  Response:     {risk.response_strategy}")
        print(f"  Controls:     {', '.join(risk.existing_controls)}")
        print(f"  Recommended:  {', '.join(risk.recommended_actions)}")

    # Summary statistics
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    from collections import Counter
    ratings = Counter(r.rating for r in risks)
    for rating in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        count = ratings.get(rating, 0)
        bar = "#" * (count * 3)
        print(f"  {rating:10s}: {count:2d} {bar}")

# Example assessment
risks = [
    Risk(
        name="Unpatched web server",
        description="Apache 2.4.49 with known path traversal CVE-2021-41773",
        likelihood=Likelihood.ALMOST_CERTAIN,
        impact=Impact.MAJOR,
        existing_controls=["WAF in front of server"],
        recommended_actions=["Patch immediately to 2.4.52+"]
    ),
    Risk(
        name="No encryption at rest",
        description="Customer PII stored in plaintext in PostgreSQL",
        likelihood=Likelihood.POSSIBLE,
        impact=Impact.CATASTROPHIC,
        existing_controls=["Database behind VPN", "Access logging"],
        recommended_actions=[
            "Enable TDE or application-level encryption",
            "Classify and encrypt PII columns"
        ]
    ),
    Risk(
        name="Weak password policy",
        description="Minimum 6 characters, no complexity requirements",
        likelihood=Likelihood.LIKELY,
        impact=Impact.MODERATE,
        existing_controls=["Account lockout after 10 attempts"],
        recommended_actions=[
            "Require 12+ characters",
            "Check against breached password lists",
            "Implement MFA"
        ]
    ),
    Risk(
        name="Missing CSRF protection",
        description="State-changing forms lack CSRF tokens",
        likelihood=Likelihood.POSSIBLE,
        impact=Impact.MODERATE,
        existing_controls=["SameSite=Lax cookies"],
        recommended_actions=["Add CSRF tokens to all forms", "Use SameSite=Strict"]
    ),
]

risk_assessment(risks)
```

### 9.3 주요 위험 프레임워크

```
┌─────────────────────────────────────────────────────────────────────┐
│                    위험 평가 프레임워크                              │
├─────────────────┬───────────────────────────────────────────────────┤
│ 프레임워크      │ 설명                                              │
├─────────────────┼───────────────────────────────────────────────────┤
│ NIST RMF        │ Risk Management Framework (SP 800-37)             │
│ (SP 800-37)     │ - 분류 → 선택 → 구현 → 평가 →                    │
│                 │   승인 → 모니터링                                 │
│                 │ - 미국 정부 표준, 널리 채택됨                     │
├─────────────────┼───────────────────────────────────────────────────┤
│ ISO 27005       │ 정보 보안 위험 관리                               │
│                 │ - 컨텍스트 → 평가 → 처리 → 모니터링              │
│                 │ - 국제 표준, 인증 가능                            │
├─────────────────┼───────────────────────────────────────────────────┤
│ FAIR            │ Factor Analysis of Information Risk               │
│                 │ - 정량적: 위험을 달러로 추정                      │
│                 │ - Loss Event Frequency × Loss Magnitude           │
├─────────────────┼───────────────────────────────────────────────────┤
│ OCTAVE          │ Operationally Critical Threat, Asset, and        │
│                 │ Vulnerability Evaluation                          │
│                 │ - 자기 주도적, 조직 중심                          │
│                 │ - 소규모 팀에 적합                                │
├─────────────────┼───────────────────────────────────────────────────┤
│ OWASP Risk      │ 웹 애플리케이션 위험에 초점                       │
│ Rating          │ - 가능성 × 영향 with 상세 요소                   │
│                 │ - OWASP Top 10과 정렬                            │
└─────────────────┴───────────────────────────────────────────────────┘
```

---

## 10. Common Vulnerability Scoring System(CVSS)

CVSS는 취약점의 주요 특성을 포착하고 심각도를 반영하는 수치 점수(0-10)를 생성하는 표준화된 방법을 제공합니다.

### 10.1 CVSS v3.1 / v4.0 메트릭

```
┌─────────────────────────────────────────────────────────────────────┐
│                   CVSS v3.1 메트릭 그룹                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  BASE METRICS (본질적인 취약점 특성)                                 │
│  ├── Attack Vector (AV): Network / Adjacent / Local / Physical      │
│  ├── Attack Complexity (AC): Low / High                              │
│  ├── Privileges Required (PR): None / Low / High                    │
│  ├── User Interaction (UI): None / Required                          │
│  ├── Scope (S): Unchanged / Changed                                  │
│  ├── Confidentiality Impact (C): None / Low / High                  │
│  ├── Integrity Impact (I): None / Low / High                        │
│  └── Availability Impact (A): None / Low / High                     │
│                                                                      │
│  TEMPORAL METRICS (시간에 따라 변하는 것)                            │
│  ├── Exploit Code Maturity: Not Defined / Unproven / PoC /          │
│  │                          Functional / High                        │
│  ├── Remediation Level: Not Defined / Official Fix / Temp Fix /     │
│  │                      Workaround / Unavailable                    │
│  └── Report Confidence: Not Defined / Unknown / Reasonable /        │
│                         Confirmed                                   │
│                                                                      │
│  ENVIRONMENTAL METRICS (조직별 특성)                                 │
│  ├── 수정된 Base 메트릭 (환경에 맞게 조정)                          │
│  └── Security Requirements (CR, IR, AR): Low / Medium / High       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

점수 범위:
  0.0        = 없음
  0.1 - 3.9  = Low
  4.0 - 6.9  = Medium
  7.0 - 8.9  = High
  9.0 - 10.0 = Critical
```

### 10.2 CVSS 점수 계산기

```python
from dataclasses import dataclass
from typing import Optional
import math

@dataclass
class CVSSv31:
    """
    Simplified CVSS v3.1 Base Score calculator.
    Reference: https://www.first.org/cvss/v3.1/specification-document
    """
    # Attack Vector
    attack_vector: str       # N(etwork), A(djacent), L(ocal), P(hysical)
    # Attack Complexity
    attack_complexity: str   # L(ow), H(igh)
    # Privileges Required
    privileges_required: str # N(one), L(ow), H(igh)
    # User Interaction
    user_interaction: str    # N(one), R(equired)
    # Scope
    scope: str               # U(nchanged), C(hanged)
    # Impact
    confidentiality: str     # N(one), L(ow), H(igh)
    integrity: str           # N(one), L(ow), H(igh)
    availability: str        # N(one), L(ow), H(igh)

    # Metric value lookups
    AV_VALUES = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.20}
    AC_VALUES = {"L": 0.77, "H": 0.44}
    PR_VALUES_UNCHANGED = {"N": 0.85, "L": 0.62, "H": 0.27}
    PR_VALUES_CHANGED = {"N": 0.85, "L": 0.68, "H": 0.50}
    UI_VALUES = {"N": 0.85, "R": 0.62}
    IMPACT_VALUES = {"N": 0.00, "L": 0.22, "H": 0.56}

    @property
    def vector_string(self) -> str:
        return (f"CVSS:3.1/AV:{self.attack_vector}/AC:{self.attack_complexity}/"
                f"PR:{self.privileges_required}/UI:{self.user_interaction}/"
                f"S:{self.scope}/C:{self.confidentiality}/"
                f"I:{self.integrity}/A:{self.availability}")

    def _exploitability_score(self) -> float:
        av = self.AV_VALUES[self.attack_vector]
        ac = self.AC_VALUES[self.attack_complexity]

        if self.scope == "C":
            pr = self.PR_VALUES_CHANGED[self.privileges_required]
        else:
            pr = self.PR_VALUES_UNCHANGED[self.privileges_required]

        ui = self.UI_VALUES[self.user_interaction]
        return 8.22 * av * ac * pr * ui

    def _impact_score(self) -> float:
        c = self.IMPACT_VALUES[self.confidentiality]
        i = self.IMPACT_VALUES[self.integrity]
        a = self.IMPACT_VALUES[self.availability]

        iss = 1 - ((1 - c) * (1 - i) * (1 - a))

        if self.scope == "U":
            return 6.42 * iss
        else:
            return 7.52 * (iss - 0.029) - 3.25 * ((iss - 0.02) ** 15)

    def base_score(self) -> float:
        impact = self._impact_score()
        exploitability = self._exploitability_score()

        if impact <= 0:
            return 0.0

        if self.scope == "U":
            score = min(impact + exploitability, 10)
        else:
            score = min(1.08 * (impact + exploitability), 10)

        # Round up to nearest 0.1
        return math.ceil(score * 10) / 10

    @property
    def severity(self) -> str:
        score = self.base_score()
        if score == 0.0:
            return "NONE"
        elif score <= 3.9:
            return "LOW"
        elif score <= 6.9:
            return "MEDIUM"
        elif score <= 8.9:
            return "HIGH"
        else:
            return "CRITICAL"

# Example: Log4Shell (CVE-2021-44228)
log4shell = CVSSv31(
    attack_vector="N",      # Network - remotely exploitable
    attack_complexity="L",   # Low - trivial to exploit
    privileges_required="N", # None - no auth needed
    user_interaction="N",    # None - no user action needed
    scope="C",               # Changed - can affect other components
    confidentiality="H",     # High - full read access
    integrity="H",           # High - full write access
    availability="H",        # High - can crash/DoS
)
print(f"Log4Shell (CVE-2021-44228)")
print(f"  Vector: {log4shell.vector_string}")
print(f"  Score:  {log4shell.base_score()}")
print(f"  Rating: {log4shell.severity}")

print()

# Example: A less severe vulnerability (local privilege escalation)
local_vuln = CVSSv31(
    attack_vector="L",      # Local - requires local access
    attack_complexity="H",   # High - specific conditions needed
    privileges_required="L", # Low - needs basic user account
    user_interaction="N",    # None
    scope="U",               # Unchanged
    confidentiality="H",     # High
    integrity="H",           # High
    availability="N",        # None
)
print(f"Local Privilege Escalation")
print(f"  Vector: {local_vuln.vector_string}")
print(f"  Score:  {local_vuln.base_score()}")
print(f"  Rating: {local_vuln.severity}")

print()

# Example: Low severity (reflected XSS requiring user interaction)
xss_vuln = CVSSv31(
    attack_vector="N",
    attack_complexity="L",
    privileges_required="N",
    user_interaction="R",    # Requires user to click a link
    scope="C",               # Can affect other origins
    confidentiality="L",     # Low - can steal some data
    integrity="L",           # Low - can modify some page content
    availability="N",        # None
)
print(f"Reflected XSS")
print(f"  Vector: {xss_vuln.vector_string}")
print(f"  Score:  {xss_vuln.base_score()}")
print(f"  Rating: {xss_vuln.severity}")
```

---

## 11. 종합 예제

가상의 온라인 뱅킹 애플리케이션에 대한 완전한 보안 분석을 진행해 보겠습니다.

### 11.1 시스템 설명

```
┌─────────────────────────────────────────────────────────────────────┐
│                    온라인 뱅킹 시스템                                │
│                                                                      │
│   ┌──────────┐     HTTPS     ┌───────────┐     gRPC     ┌────────┐ │
│   │ Mobile   │ ────────────▶│   API     │ ──────────▶│ Account│ │
│   │ App      │               │  Gateway  │             │ Service│ │
│   └──────────┘               └─────┬─────┘             └────────┘ │
│                                    │                               │
│   ┌──────────┐     HTTPS     ┌─────▼─────┐     gRPC     ┌────────┐ │
│   │ Web      │ ────────────▶│   Auth    │ ──────────▶│Transfer│ │
│   │ Browser  │               │  Service  │             │Service │ │
│   └──────────┘               └─────┬─────┘             └────────┘ │
│                                    │                               │
│   ┌──────────┐                ┌────▼──────┐              ┌────────┐ │
│   │ Admin    │ ── VPN ──────▶│  Admin    │ ── SQL ────▶│  DB    │ │
│   │ Console  │               │  Portal   │              │(PgSQL) │ │
│   └──────────┘               └───────────┘              └────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 11.2 STRIDE 위협 요약

| # | 위협 | 범주 | CIA 영향 | DREAD 점수 | 완화 |
|---|--------|----------|------------|-------------|------------|
| T1 | 도난당한 세션 토큰 재사용 | Spoofing | C | 7.2 | 짧은 수명의 JWT, 토큰 바인딩 |
| T2 | 수정된 이체 금액 | Tampering | I | 8.0 | 요청 서명, 서버 검증 |
| T3 | 사용자가 이체 부인 | Repudiation | Non-rep | 5.4 | 서명된 감사 로그, 디지털 영수증 |
| T4 | 계좌 번호 유출 | Info Disc. | C | 7.8 | 저장 데이터 암호화, 필드 마스킹 |
| T5 | API 게이트웨이에 대한 DDoS | DoS | A | 6.2 | CDN, rate limiting, 자동 스케일링 |
| T6 | 관리자가 슈퍼 관리자로 상승 | Elev. Priv | Auth | 8.6 | RBAC, 승인 워크플로, MFA |

### 11.3 적용된 Defense in Depth

```
Layer 1 - 정책:     보안 교육, 사고 대응 계획
Layer 2 - 물리적:    생체 인증 접근이 있는 데이터 센터
Layer 3 - 경계:     CDN/WAF, DDoS 보호, 관리자용 VPN
Layer 4 - 네트워크:  VLAN 세그먼테이션, 서비스 간 mTLS
Layer 5 - 호스트:    강화된 OS, 자동 패칭, EDR 에이전트
Layer 6 - 애플리케이션:  입력 검증, CSRF 토큰, CSP 헤더
Layer 7 - 데이터:    저장 데이터 AES-256 암호화, 전송 중 TLS 1.3
```

### 11.4 적용된 최소 권한

```
Mobile App:     API Gateway만 호출 가능 (자신의 계정 읽기, 이체 시작)
API Gateway:    Auth 및 Account 서비스로 라우팅; 직접 DB 접근 없음
Auth Service:   users 테이블 읽기 전용; sessions 테이블 쓰기 전용
Account Svc:    accounts 테이블 읽기/쓰기; users 접근 없음
Transfer Svc:   transfers 테이블 삽입 전용; accounts 읽기
Admin Portal:   VPN을 통해서만 접근 가능; MFA + 관리자 승인 필요
Database:       인터넷 접근 없음; 서비스 VPC로부터만 연결
```

---

## 12. 연습 문제

### 연습 1: CIA 분석 (초급)

아래 각 시나리오에 대해 주로 위반되는 CIA 속성을 식별하고 이를 방지할 수 있는 제어를 제시하세요:

1. 병원의 전자 건강 기록 시스템이 랜섬웨어 공격으로 오프라인 상태입니다.
2. 해커가 전자상거래 데이터베이스의 항목 가격을 $99에서 $0.01로 변경합니다.
3. 직원이 고객 사회보장번호가 담긴 스프레드시트를 개인 이메일로 보냅니다.
4. DNS 레코드가 수정되어 은행 고객이 피싱 사이트로 리디렉션됩니다.
5. 불만을 품은 직원이 프로덕션 데이터베이스 백업을 삭제합니다.

### 연습 2: STRIDE Threat Model (중급)

다음 구성 요소를 가진 음식 배달 애플리케이션에 대한 Data Flow Diagram을 그리세요:
- 고객 모바일 앱
- 레스토랑 대시보드(웹)
- API 서버
- 결제 처리기(Stripe)
- 배달 추적 서비스
- PostgreSQL 데이터베이스

STRIDE를 적용하여 최소 8개의 위협을 식별하세요. 각 위협에 대해 다음을 명시하세요:
- STRIDE 범주
- 영향받는 DFD 요소
- 심각도(DREAD 점수 사용)
- 최소 한 가지 완화

### 연습 3: CVSS 점수화 (중급)

다음 취약점에 대한 CVSS v3.1 기본 점수를 계산하세요:

1. 웹 애플리케이션이 인증되지 않은 사용자가 URL 파라미터를 조작하여 서버의 모든 파일을 다운로드할 수 있도록 허용합니다(path traversal). 사용자 상호작용은 필요하지 않습니다.

2. 데스크톱 애플리케이션에 특별히 제작된 파일을 열면 트리거될 수 있는 버퍼 오버플로우가 있습니다. 공격자는 피해자가 파일을 열어야 하며, 성공적인 익스플로잇은 공격자에게 피해자와 동일한 권한을 부여합니다.

3. API 엔드포인트가 인증을 요구하지만 인증된 사용자가 요청된 리소스에 접근할 권한이 있는지 확인하지 않습니다(IDOR). 동일한 애플리케이션 내의 데이터만 영향을 받습니다.

### 연습 4: Defense in Depth 설계 (고급)

신용카드 결제를 처리하는 스타트업의 보안 아키텍처를 설계하고 있습니다. 현재 그들은:
- 애플리케이션과 데이터베이스를 모두 실행하는 단일 웹 서버
- 저장 데이터 암호화 없음
- 비밀번호 인증으로 SSH를 통한 관리자 접근
- 로깅 또는 모니터링 없음
- 모든 직원이 하나의 데이터베이스 관리자 비밀번호 공유

7계층 defense-in-depth 전략을 설계하세요. 각 계층에 대해 다음을 명시하세요:
- 최소 2개의 구체적인 제어
- 각 제어가 완화하는 위협
- 우선순위(첫 번째/두 번째/세 번째 구현)

### 연습 5: 최소 권한 감사 (고급)

다음 AWS IAM 정책에서 최소 권한의 모든 위반 사항을 식별하고 적절하게 범위가 지정되도록 다시 작성하세요:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "s3:*",
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "ec2:*",
                "rds:*"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": "iam:*",
            "Resource": "*"
        }
    ]
}
```

컨텍스트: 이 정책은 다음이 필요한 웹 애플리케이션 백엔드를 위한 것입니다:
- 단일 S3 버킷(`my-app-uploads`)에서 파일 읽기 및 쓰기
- RDS PostgreSQL 인스턴스(`my-app-db`)에서 읽기
- EC2 관리 필요 없음(ECS에서 실행)
- IAM 관리 필요 없음

### 연습 6: Security by Design 검토 (고급)

다음 Flask 코드를 검토하고 모든 보안 문제를 식별하세요. security-by-design 원칙에 따라 다시 작성하세요:

```python
from flask import Flask, request, jsonify
import sqlite3
import hashlib

app = Flask(__name__)

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    password_hash = hashlib.md5(password.encode()).hexdigest()
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password_hash}'"
    cursor.execute(query)
    user = cursor.fetchone()

    if user:
        return jsonify({"status": "success", "user_id": user[0], "role": user[3]})
    else:
        return jsonify({"status": "failed", "error": f"No user found with username {username}"})

@app.route('/admin/delete_user/<user_id>')
def delete_user(user_id):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM users WHERE id={user_id}")
    conn.commit()
    return jsonify({"status": "deleted"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
```

최소 10개의 보안 문제를 식별하고 수정된 코드를 제공하세요.

---

## 13. 참고 자료

- Anderson, R. (2020). *Security Engineering*, 3rd Edition. Wiley.
- Shostack, A. (2014). *Threat Modeling: Designing for Security*. Wiley.
- OWASP Threat Modeling - https://owasp.org/www-community/Threat_Modeling
- NIST SP 800-30: Guide for Conducting Risk Assessments
- NIST SP 800-37: Risk Management Framework
- FIRST CVSS v3.1 Specification - https://www.first.org/cvss/v3.1/specification-document
- Microsoft STRIDE - https://learn.microsoft.com/en-us/azure/security/develop/threat-modeling-tool-threats

---

## 다음 레슨

[02. Cryptography 기초](./02_Cryptography_Basics.md)에서는 대칭 및 비대칭 암호화, 키 교환, 디지털 서명을 실용적인 Python 예제와 함께 다룹니다.
