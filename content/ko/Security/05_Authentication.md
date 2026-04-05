# 05. 인증 시스템

**이전**: [04. TLS/SSL 및 공개키 기반 구조](./04_TLS_and_PKI.md) | **다음**: [06. 접근 제어 및 권한 부여](06_Authorization.md)

---

인증(Authentication)은 사용자나 시스템이 주장하는 신원을 확인하는 프로세스입니다. "당신은 누구입니까?"라는 질문에 답하며, 모든 접근 제어 결정의 기반이 됩니다. 잘못 구현된 인증 시스템은 가장 정교한 권한 부여 및 암호화조차 무용지물로 만들 수 있습니다. 이 강의는 비밀번호 기반 인증, 다중 인증, 토큰 기반 시스템, OAuth 2.0/OIDC, 세션 관리, 생체 인증 방식을 실용적인 Python 예제와 함께 다룹니다.

## 학습 목표

- 솔팅과 키 스트레칭을 사용한 안전한 비밀번호 저장 구현
- 다중 인증(TOTP, FIDO2/WebAuthn) 이해 및 구현
- OAuth 2.0 및 OpenID Connect 인증/권한 부여 플로우 설명
- 쿠키, 토큰, JWT를 사용한 안전한 세션 관리
- 일반적인 JWT 함정 식별 및 회피
- 안전한 비밀번호 재설정 플로우 설계
- 생체 인증 개념 및 트레이드오프 이해

---

## 1. 비밀번호 기반 인증

### 1.1 비밀번호 문제

비밀번호는 잘 알려진 약점에도 불구하고 가장 일반적인 인증 방법으로 남아있습니다.

```
┌─────────────────────────────────────────────────────────────────┐
│               비밀번호 인증 플로우                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   사용자                  서버                                     │
│   ┌──────┐               ┌──────────┐                           │
│   │ 폼   │───────────────▶│ 수신     │                           │
│   │ 사용자│  username +    │ username │                           │
│   │ pass │  password      │ + pass   │                          │
│   └──────┘               └────┬─────┘                           │
│                                │                                 │
│                                ▼                                 │
│                         ┌──────────────┐                        │
│                         │  제공된      │                         │
│                         │  비밀번호    │                         │
│                         │  해시화      │                         │
│                         └──────┬───────┘                        │
│                                │                                 │
│                                ▼                                 │
│                    ┌────────────────────┐                       │
│                    │  DB에 저장된       │                        │
│                    │  해시와 비교       │                        │
│                    └────────┬───────────┘                       │
│                             │                                    │
│                      ┌──────┴──────┐                            │
│                      │             │                             │
│                   일치?         불일치?                           │
│                      │             │                             │
│                      ▼             ▼                             │
│                  ┌────────┐   ┌────────┐                        │
│                  │ 접근   │   │ 접근   │                         │
│                  │ 허용   │   │ 거부   │                         │
│                  └────────┘   └────────┘                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 평문 비밀번호를 저장하지 말아야 하는 이유

공격자가 데이터베이스에 접근하게 되면(SQL 인젝션, 백업 도난, 내부자 위협 등을 통해) 평문 비밀번호는 즉시 노출됩니다. **심층 방어(defense in depth)** 원칙은 데이터베이스 침해가 발생하더라도 사용자 자격 증명이 직접 노출되지 않도록 요구합니다.

```
┌──────────────────────────────────────────────────────────────┐
│  절대 이렇게 하지 마세요:                                        │
│                                                               │
│  users 테이블:                                                 │
│  ┌──────────┬────────────────┐                               │
│  │ username │ password       │                                │
│  ├──────────┼────────────────┤                                │
│  │ alice    │ MyP@ssw0rd!    │  ← 평문 = 재앙                 │
│  │ bob      │ hunter2        │                                │
│  └──────────┴────────────────┘                               │
│                                                               │
│  대신 이렇게 하세요:                                            │
│                                                               │
│  users 테이블:                                                 │
│  ┌──────────┬──────────────────────────────────────────────┐ │
│  │ username │ password_hash                                │  │
│  ├──────────┼──────────────────────────────────────────────┤ │
│  │ alice    │ $2b$12$LJ3m4ys3Lk0aB...  (bcrypt 해시)     │  │
│  │ bob      │ $argon2id$v=19$m=65536... (argon2 해시)     │  │
│  └──────────┴──────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 1.3 해싱, 솔팅, 키 스트레칭

**해싱(Hashing)**은 비밀번호를 고정 길이 문자열로 변환합니다. 하지만 단순 해싱(MD5, SHA-256)은 레인보우 테이블과 무차별 대입 공격에 취약합니다.

**솔팅(Salting)**은 해싱 전에 각 비밀번호에 고유한 랜덤 값을 추가하여 레인보우 테이블을 무력화합니다.

**키 스트레칭(Key Stretching)**은 해시 함수를 수천 또는 수백만 번 적용하여 무차별 대입 공격을 계산적으로 비용이 많이 들게 만듭니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                  비밀번호 해싱 파이프라인                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   "MyP@ssw0rd!"                                                  │
│        │                                                         │
│        ▼                                                         │
│   ┌──────────────┐                                              │
│   │  랜덤 솔트   │ ──▶  salt = "x9Kp2mQ..."  (랜덤, 고유)      │
│   │  생성        │                                               │
│   └──────┬───────┘                                              │
│          │                                                       │
│          ▼                                                       │
│   ┌──────────────────┐                                          │
│   │  연결            │ ──▶  "x9Kp2mQ..." + "MyP@ssw0rd!"      │
│   │  salt + password │                                           │
│   └──────┬───────────┘                                          │
│          │                                                       │
│          ▼                                                       │
│   ┌──────────────────┐                                          │
│   │  키 스트레칭     │ ──▶  100,000+ 회 반복 해시               │
│   │  (bcrypt/argon2) │      또는 메모리 하드 함수                │
│   └──────┬───────────┘                                          │
│          │                                                       │
│          ▼                                                       │
│   "$2b$12$x9Kp2mQ.../hashed_output"                            │
│   (솔트 + 해시가 함께 저장됨)                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**권장 알고리즘 (우선순위 순서):**

| 알고리즘 | 유형 | 주요 특징 | 권장 파라미터 |
|---------|------|----------|-------------|
| Argon2id | 메모리 하드 | GPU/ASIC 저항 | m=65536, t=3, p=4 |
| bcrypt | CPU 하드 | 널리 지원됨 | 비용 계수 12+ |
| scrypt | 메모리 하드 | 좋은 대안 | N=2^15, r=8, p=1 |
| PBKDF2 | 반복 기반 | NIST 승인 | 600,000+ 반복 (SHA-256) |

### 1.4 Python 구현: 비밀번호 해싱

```python
"""
password_hashing.py - bcrypt와 argon2를 사용한 안전한 비밀번호 저장
"""
import bcrypt
import hashlib
import os
import secrets


# ==============================================================
# 방법 1: bcrypt (가장 널리 사용됨)
# ==============================================================

def hash_password_bcrypt(password: str) -> str:
    """자동 솔팅을 사용하여 bcrypt로 비밀번호 해시"""
    # bcrypt는 자동으로 솔트를 생성하고 출력에 포함시킴
    # 비용 계수(라운드)는 계산 시간을 제어: 2^라운드 반복
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt(rounds=12)  # 2^12 = 4096 반복
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password_bcrypt(password: str, stored_hash: str) -> bool:
    """bcrypt 해시에 대해 비밀번호 검증"""
    password_bytes = password.encode('utf-8')
    stored_bytes = stored_hash.encode('utf-8')
    return bcrypt.checkpw(password_bytes, stored_bytes)


# ==============================================================
# 방법 2: Argon2 (OWASP 권장)
# ==============================================================

# pip install argon2-cffi
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

def hash_password_argon2(password: str) -> str:
    """Argon2id를 사용하여 비밀번호 해시"""
    ph = PasswordHasher(
        time_cost=3,        # 반복 횟수
        memory_cost=65536,   # 64 MB 메모리
        parallelism=4,       # 병렬 스레드 수
        hash_len=32,         # 해시 출력 길이
        salt_len=16          # 랜덤 솔트 길이
    )
    return ph.hash(password)


def verify_password_argon2(password: str, stored_hash: str) -> bool:
    """Argon2 해시에 대해 비밀번호 검증"""
    ph = PasswordHasher()
    try:
        return ph.verify(stored_hash, password)
    except VerifyMismatchError:
        return False


# ==============================================================
# 방법 3: PBKDF2 (Python 내장, 외부 의존성 없음)
# ==============================================================

def hash_password_pbkdf2(password: str) -> str:
    """PBKDF2-HMAC-SHA256을 사용하여 비밀번호 해시"""
    salt = os.urandom(32)  # 32바이트 랜덤 솔트
    iterations = 600_000   # SHA-256에 대한 OWASP 권장 최소값

    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        iterations,
        dklen=32
    )

    # 솔트 + 반복 횟수 + 해시를 함께 저장
    # 형식: iterations$salt_hex$hash_hex
    return f"{iterations}${salt.hex()}${key.hex()}"


def verify_password_pbkdf2(password: str, stored: str) -> bool:
    """PBKDF2 해시에 대해 비밀번호 검증"""
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

    # 타이밍 공격 방지를 위한 상수 시간 비교 사용
    return secrets.compare_digest(key, stored_key)


# ==============================================================
# 데모
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

### 1.5 비밀번호 정책

강력한 비밀번호만으로는 충분하지 않습니다. 포괄적인 비밀번호 정책은 다음을 포함합니다:

```
┌─────────────────────────────────────────────────────────────────┐
│               현대 비밀번호 정책 (NIST SP 800-63B)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  해야 할 것:                                                      │
│  ✓ 최소 8자 (12자 이상 권장)                                      │
│  ✓ 최대 64자 이상 허용                                            │
│  ✓ 모든 출력 가능한 ASCII + Unicode 문자 허용                      │
│  ✓ 침해된 비밀번호 목록 확인 (haveibeenpwned.com)                  │
│  ✓ 일반적인 비밀번호 확인 (password, 123456 등)                    │
│  ✓ 비밀번호 필드에 붙여넣기 허용 (비밀번호 관리자용)                │
│  ✓ 비밀번호 강도 측정기 표시                                       │
│                                                                  │
│  하지 말아야 할 것:                                                │
│  ✗ 임의의 복잡성 규칙 강제 (대문자 + 숫자 + ...)                   │
│  ✗ 정기적인 비밀번호 변경 강제 (침해 의심 시 제외)                  │
│  ✗ 비밀번호 힌트 또는 지식 기반 질문 사용                           │
│  ✗ 비밀번호 자동 자르기                                            │
│  ✗ 비밀번호 복구에 SMS 사용 (SIM 스와핑 공격)                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
"""
password_policy.py - NIST 지침에 따른 현대적 비밀번호 검증
"""
import re
import hashlib
import requests
from typing import Tuple, List


# 일반 비밀번호 목록 (상위 20개 - 실제로는 훨씬 큰 목록 사용)
COMMON_PASSWORDS = {
    "password", "123456", "123456789", "12345678", "12345",
    "1234567", "qwerty", "abc123", "password1", "111111",
    "iloveyou", "1234567890", "123123", "admin", "letmein",
    "welcome", "monkey", "dragon", "master", "000000",
}


def check_password_strength(password: str) -> Tuple[bool, List[str]]:
    """
    현대 보안 지침에 따라 비밀번호 검증.
    (is_valid, 문제점_목록) 반환.
    """
    issues = []

    # 길이 확인 (NIST 최소: 8, 권장: 12+)
    if len(password) < 8:
        issues.append("비밀번호는 최소 8자 이상이어야 합니다")
    elif len(password) < 12:
        issues.append("경고: 더 나은 보안을 위해 12자 이상 권장됩니다")

    # 일반 비밀번호 확인
    if password.lower() in COMMON_PASSWORDS:
        issues.append("일반적으로 사용되는 비밀번호입니다")

    # 반복 패턴 확인
    if re.match(r'^(.)\1+$', password):
        issues.append("비밀번호는 단일 반복 문자일 수 없습니다")

    # 순차 패턴 확인
    if re.search(r'(012|123|234|345|456|567|678|789|890)', password):
        issues.append("경고: 순차적 숫자 패턴 포함")

    # 컨텍스트별 확인 (사용자 이름, 이메일 등 포함)
    # 실제 환경에서는 사용자 개인 정보도 확인

    is_valid = not any(
        not issue.startswith("경고") for issue in issues
    )

    return is_valid, issues


def check_breached_password(password: str) -> bool:
    """
    Have I Been Pwned API를 사용하여 알려진 침해에서
    비밀번호가 나타나는지 확인 (k-익명성 모델 - SHA-1 해시의
    처음 5자만 전송).
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

        # 응답은 "SUFFIX:COUNT" 형식의 라인들을 포함
        for line in response.text.splitlines():
            hash_suffix, count = line.split(':')
            if hash_suffix == suffix:
                return True  # 비밀번호가 침해됨

        return False  # 침해에서 발견되지 않음
    except requests.RequestException:
        # API를 사용할 수 없으면 실패를 허용 (하지만 오류는 기록)
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
        status = "통과" if is_valid else "실패"
        print(f"\n[{status}] '{pwd}'")
        for issue in issues:
            print(f"  - {issue}")
```

---

## 2. 다중 인증 (MFA)

### 2.1 인증 요소

```
┌─────────────────────────────────────────────────────────────────┐
│                  인증 요소                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  요소 1: 아는 것 (Something You KNOW)                            │
│  ├── 비밀번호                                                     │
│  ├── PIN                                                         │
│  └── 보안 질문 (권장하지 않음)                                     │
│                                                                  │
│  요소 2: 가진 것 (Something You HAVE)                            │
│  ├── 스마트폰 (TOTP 앱)                                           │
│  ├── 하드웨어 보안 키 (YubiKey, Titan)                            │
│  ├── 스마트 카드                                                  │
│  └── SMS (약함 - SIM 스와핑 위험)                                 │
│                                                                  │
│  요소 3: 본인 자체 (Something You ARE)                            │
│  ├── 지문                                                        │
│  ├── 얼굴 인식                                                    │
│  ├── 홍채 스캔                                                    │
│  └── 음성 인식                                                    │
│                                                                  │
│  MFA = 2개 이상의 다른 요소 결합                                  │
│  (비밀번호 + TOTP = 2FA, 하지만 두 개의 비밀번호 ≠ 2FA)         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 TOTP (시간 기반 일회용 비밀번호)

TOTP는 공유 비밀과 현재 시간을 기반으로 짧은 수명의 코드를 생성합니다. RFC 6238에 정의되어 있습니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                     TOTP 알고리즘                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  설정 (일회성):                                                   │
│  1. 서버가 랜덤 비밀 키 생성 (base32 인코딩)                       │
│  2. 서버가 QR 코드를 통해 사용자와 비밀 공유                        │
│  3. 사용자가 인증 앱으로 QR 코드 스캔                              │
│                                                                  │
│  검증 (각 로그인):                                                │
│                                                                  │
│       공유 비밀              현재 시간                              │
│            │                          │                          │
│            ▼                          ▼                          │
│       ┌─────────┐            ┌──────────────┐                   │
│       │ 비밀    │            │  T = floor   │                    │
│       │  키     │            │  (time / 30) │                    │
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
│              │ 6자리로     │                                    │
│              │  축약       │                                    │
│              └──────┬──────┘                                    │
│                     │                                            │
│                     ▼                                            │
│                  "482916"   (30초 동안 유효)                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
"""
totp_example.py - pyotp를 사용한 TOTP 구현
pip install pyotp qrcode[pil]
"""
import pyotp
import qrcode
import time
import io


class TOTPManager:
    """TOTP 기반 2단계 인증 관리"""

    def __init__(self):
        self.secrets = {}  # 실제 환경에서는 암호화된 데이터베이스에 저장

    def enroll_user(self, username: str, issuer: str = "MyApp") -> str:
        """
        새 사용자를 위한 TOTP 비밀 생성.
        QR 코드 생성을 위한 프로비저닝 URI 반환.
        """
        # 랜덤 base32 비밀 생성 (160 비트)
        secret = pyotp.random_base32()
        self.secrets[username] = secret

        # 인증 앱을 위한 프로비저닝 URI 생성
        totp = pyotp.TOTP(secret)
        uri = totp.provisioning_uri(
            name=username,
            issuer_name=issuer
        )

        return uri, secret

    def generate_qr_code(self, uri: str, filename: str = "totp_qr.png"):
        """프로비저닝 URI로부터 QR 코드 이미지 생성"""
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(uri)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(filename)
        print(f"QR code saved to {filename}")

    def verify_totp(self, username: str, code: str) -> bool:
        """
        주어진 사용자에 대한 TOTP 코드 검증.
        1 기간의 클럭 드리프트 허용 (±30초).
        """
        if username not in self.secrets:
            return False

        secret = self.secrets[username]
        totp = pyotp.TOTP(secret)

        # valid_window=1은 t-1 및 t+1 기간의 코드 허용
        return totp.verify(code, valid_window=1)

    def get_current_code(self, username: str) -> str:
        """현재 TOTP 코드 가져오기 (테스트 전용)"""
        if username not in self.secrets:
            return None

        secret = self.secrets[username]
        totp = pyotp.TOTP(secret)
        return totp.now()


# 계정 복구를 위한 백업 코드
def generate_backup_codes(count: int = 10) -> list:
    """
    일회용 백업 코드 생성.
    각 코드는 8자, 영숫자.
    """
    import secrets
    codes = []
    for _ in range(count):
        code = secrets.token_hex(4).upper()  # 8개의 16진수 문자
        # 가독성을 위해 XXXX-XXXX 형식으로 포맷
        formatted = f"{code[:4]}-{code[4:]}"
        codes.append(formatted)
    return codes


if __name__ == "__main__":
    manager = TOTPManager()

    # 사용자 등록
    uri, secret = manager.enroll_user("alice@example.com")
    print(f"Secret: {secret}")
    print(f"URI: {uri}")

    # 현재 코드 생성
    current_code = manager.get_current_code("alice@example.com")
    print(f"\nCurrent TOTP code: {current_code}")

    # 검증
    print(f"Verification: {manager.verify_totp('alice@example.com', current_code)}")
    print(f"Wrong code:   {manager.verify_totp('alice@example.com', '000000')}")

    # 백업 코드 생성
    print("\nBackup Codes:")
    for code in generate_backup_codes():
        print(f"  {code}")
```

### 2.3 FIDO2 / WebAuthn

FIDO2(Fast Identity Online)와 그 웹 컴포넌트인 WebAuthn은 오늘날 사용 가능한 가장 강력한 인증 형태를 나타냅니다. 공개키 암호화를 사용하며 피싱에 저항력이 있습니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                  WebAuthn 등록 플로우                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   브라우저 (클라이언트)     서버 (신뢰 당사자)                       │
│        │                          │                              │
│        │   1. 챌린지 요청         │                              │
│        │ ──────────────────────▶  │                              │
│        │                          │                              │
│        │   2. 챌린지 +            │                              │
│        │      RP 정보 + 사용자 정보│                              │
│        │ ◀──────────────────────  │                              │
│        │                          │                              │
│   ┌────┴────┐                     │                              │
│   │ 브라우저│                     │                              │
│   │ 사용자에게│                    │                              │
│   │ 키 터치 │                     │                              │
│   │ 또는    │                     │                              │
│   │ 생체인증│                     │                              │
│   │ 사용    │                     │                              │
│   │ 프롬프트│                     │                              │
│   └────┬────┘                     │                              │
│        │                          │                              │
│   ┌────┴──────────────┐          │                              │
│   │ 인증기가          │          │                              │
│   │ 새 키 쌍 생성:     │          │                              │
│   │ - 개인 키         │          │                              │
│   │   (키에 저장)      │          │                              │
│   │ - 공개 키         │          │                              │
│   │   (서버로 전송)    │          │                              │
│   └────┬──────────────┘          │                              │
│        │                          │                              │
│        │   3. 공개 키 +           │                              │
│        │      서명된 챌린지       │                              │
│        │ ──────────────────────▶  │                              │
│        │                          │                              │
│        │                    ┌─────┴─────┐                       │
│        │                    │ 서명      │                        │
│        │                    │ 검증      │                        │
│        │                    │ 공개 키   │                        │
│        │                    │ 저장      │                        │
│        │                    └─────┬─────┘                       │
│        │                          │                              │
│        │   4. 등록 완료           │                              │
│        │ ◀──────────────────────  │                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**WebAuthn의 주요 장점:**

| 특징 | 비밀번호 | TOTP | WebAuthn/FIDO2 |
|-----|---------|------|---------------|
| 피싱 저항 | 아니오 | 아니오 | 예 |
| 서버에 공유 비밀 없음 | 아니오 | 아니오 | 예 (공개 키만) |
| 사용자 노력 | 높음 (암기) | 중간 (코드 복사) | 낮음 (터치/생체인증) |
| 재생 공격 | 취약 | 시간 제한 | 불가능 |
| 침해 영향 | 높음 | 중간 | 최소 |

---

## 3. OAuth 2.0 및 OpenID Connect

### 3.1 OAuth 2.0 개요

OAuth 2.0은 **권한 부여(authorization)** 프레임워크입니다(인증이 아님). 사용자의 자격 증명을 공유하지 않고 제3자 애플리케이션이 사용자를 대신하여 리소스에 접근할 수 있도록 합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    OAuth 2.0 역할                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  리소스 소유자   = 데이터를 소유한 사용자                          │
│  클라이언트      = 접근을 요청하는 애플리케이션                     │
│  권한 부여       = 사용자를 인증하고                               │
│    서버            토큰을 발행하는 서버 (예: Google, GitHub)       │
│  리소스 서버     = 보호된 리소스를 보유한 API 서버                 │
│                                                                  │
│  예시:                                                           │
│  "MyApp이 Google 캘린더에 접근하려고 합니다"                       │
│                                                                  │
│  리소스 소유자    = 당신 (Google 사용자)                          │
│  클라이언트       = MyApp                                         │
│  권한 부여        = accounts.google.com                          │
│    서버                                                          │
│  리소스 서버      = calendar.googleapis.com                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 권한 부여 코드 플로우 (가장 일반적)

서버 측 웹 애플리케이션에 권장되는 플로우입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│          OAuth 2.0 권한 부여 코드 플로우                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  사용자      클라이언트 앱     인증 서버      리소스 서버          │
│   │              │                │                  │            │
│   │  1. 클릭     │                │                  │            │
│   │  "Google로   │                │                  │            │
│   │   로그인"    │                │                  │            │
│   │─────────────▶│                │                  │            │
│   │              │                │                  │            │
│   │              │  2. 인증 URL로 │                  │            │
│   │              │  리디렉션      │                  │            │
│   │◀─────────────│                │                  │            │
│   │              │                │                  │            │
│   │  3. 사용자가 로그인하고       │                  │            │
│   │     권한 동의                 │                  │            │
│   │─────────────────────────────▶│                  │            │
│   │              │                │                  │            │
│   │  4. 권한 부여 코드와          │                  │            │
│   │     함께 리디렉션             │                  │            │
│   │◀─────────────────────────────│                  │            │
│   │              │                │                  │            │
│   │─────────────▶│                │                  │            │
│   │  (code)      │                │                  │            │
│   │              │  5. 코드를     │                  │            │
│   │              │  토큰으로      │                  │            │
│   │              │  교환          │                  │            │
│   │              │───────────────▶│                  │            │
│   │              │                │                  │            │
│   │              │  6. 접근 +     │                  │            │
│   │              │  갱신 토큰     │                  │            │
│   │              │◀───────────────│                  │            │
│   │              │                │                  │            │
│   │              │  7. 접근 토큰과 함께 API 요청     │            │
│   │              │──────────────────────────────────▶│            │
│   │              │                │                  │            │
│   │              │  8. 보호된 리소스                 │            │
│   │              │◀──────────────────────────────────│            │
│   │              │                │                  │            │
│   │  9. 응답     │                │                  │            │
│   │◀─────────────│                │                  │            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 PKCE를 사용한 권한 부여 코드 플로우

PKCE(Proof Key for Code Exchange)는 공개 클라이언트(SPA, 모바일 앱)에 **필수**이며 모든 클라이언트에 권장됩니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                      PKCE 확장                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  리디렉션 전 (2단계):                                             │
│                                                                  │
│  1. 클라이언트가 랜덤 "code_verifier" 생성                        │
│     code_verifier = random_string(43-128 chars)                  │
│                                                                  │
│  2. 클라이언트가 "code_challenge" 계산                           │
│     code_challenge = BASE64URL(SHA256(code_verifier))            │
│                                                                  │
│  3. 클라이언트가 권한 부여 요청에 code_challenge 전송             │
│     GET /authorize?                                              │
│       response_type=code&                                        │
│       client_id=...&                                             │
│       code_challenge=...&                                        │
│       code_challenge_method=S256                                 │
│                                                                  │
│  토큰 교환 시 (5단계):                                            │
│                                                                  │
│  4. 클라이언트가 토큰 요청과 함께 code_verifier 전송              │
│     POST /token                                                  │
│       grant_type=authorization_code&                             │
│       code=...&                                                  │
│       code_verifier=...                                          │
│                                                                  │
│  5. 서버가 검증:                                                  │
│     BASE64URL(SHA256(code_verifier)) == stored code_challenge    │
│                                                                  │
│  이유? 권한 부여 코드 가로채기 공격 방지.                          │
│  코드를 훔친 공격자는 code_verifier 없이는 교환할 수 없음.         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 OpenID Connect (OIDC)

OpenID Connect는 OAuth 2.0 위에 구축된 **인증(authentication)** 레이어입니다. OAuth 2.0이 권한 부여를 제공하는 반면("이 앱은 캘린더에 접근할 수 있음"), OIDC는 인증을 제공합니다("이 사용자는 alice@example.com입니다").

```
┌─────────────────────────────────────────────────────────────────┐
│              OAuth 2.0 vs OpenID Connect                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  OAuth 2.0:                                                      │
│  - 목적: 권한 부여 (접근 위임)                                    │
│  - 토큰: 접근 토큰 (불투명 또는 JWT)                              │
│  - 답하는 질문: "이 앱은 무엇을 할 수 있는가?"                     │
│                                                                  │
│  OpenID Connect (OIDC):                                          │
│  - 목적: 인증 (신원 검증)                                         │
│  - 토큰: ID 토큰 (항상 JWT) + 접근 토큰                           │
│  - 답하는 질문: "이 사용자는 누구인가?"                            │
│  - 추가: UserInfo 엔드포인트, 표준 클레임 (sub, email, name)      │
│  - 스코프: "openid" (필수), "profile", "email"                   │
│                                                                  │
│  OIDC는 OAuth 2.0 위에 구축됨:                                   │
│  ┌─────────────────────────────────┐                            │
│  │       OpenID Connect            │  ← 인증                    │
│  │  ┌──────────────────────────┐   │                            │
│  │  │      OAuth 2.0           │   │  ← 권한 부여               │
│  │  │  ┌───────────────────┐   │   │                            │
│  │  │  │     HTTP/TLS      │   │   │  ← 전송                    │
│  │  │  └───────────────────┘   │   │                            │
│  │  └──────────────────────────┘   │                            │
│  └─────────────────────────────────┘                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.5 Python 예제: OAuth 2.0 클라이언트

```python
"""
oauth_client.py - PKCE를 사용한 OAuth 2.0 권한 부여 코드 플로우
requests-oauthlib 라이브러리 사용
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

# OAuth 2.0 설정 (GitHub 예시)
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
    """PKCE code_verifier 및 code_challenge 생성"""
    # code_verifier: 43-128자, 예약되지 않은 URI 문자
    code_verifier = base64.urlsafe_b64encode(
        secrets.token_bytes(32)
    ).rstrip(b'=').decode('ascii')

    # code_challenge: BASE64URL(SHA256(code_verifier))
    digest = hashlib.sha256(code_verifier.encode('ascii')).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')

    return code_verifier, code_challenge


@app.route("/login")
def login():
    """OAuth 2.0 권한 부여 코드 플로우 시작"""
    # PKCE 쌍 생성
    code_verifier, code_challenge = generate_pkce_pair()

    # CSRF 방지를 위한 state 생성
    state = secrets.token_urlsafe(32)

    # 세션에 저장
    session["oauth_state"] = state
    session["code_verifier"] = code_verifier

    # 권한 부여 URL 구축
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
    """권한 부여 코드와 함께 OAuth 2.0 콜백 처리"""
    # CSRF 방지를 위한 state 검증
    if request.args.get("state") != session.get("oauth_state"):
        return "State mismatch - possible CSRF attack", 403

    # 오류 확인
    if "error" in request.args:
        return f"OAuth error: {request.args['error']}", 400

    # 권한 부여 코드를 토큰으로 교환
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

    # 사용자 정보 가져오기
    user_response = requests.get(
        OAUTH_CONFIG["userinfo_url"],
        headers={"Authorization": f"Bearer {access_token}"},
    )
    user_info = user_response.json()

    # 세션에 사용자 저장
    session["user"] = {
        "id": user_info.get("id"),
        "login": user_info.get("login"),
        "name": user_info.get("name"),
        "email": user_info.get("email"),
    }

    # OAuth state 정리
    session.pop("oauth_state", None)
    session.pop("code_verifier", None)

    return redirect("/profile")


@app.route("/profile")
def profile():
    """사용자 프로필 표시 (보호된 경로)"""
    user = session.get("user")
    if not user:
        return redirect("/login")
    return jsonify(user)


@app.route("/logout")
def logout():
    """세션 지우고 로그아웃"""
    session.clear()
    return redirect("/")
```

---

## 4. 세션 관리

### 4.1 서버 측 세션 (쿠키 기반)

```
┌─────────────────────────────────────────────────────────────────┐
│                서버 측 세션 플로우                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  브라우저                          서버                           │
│     │                                │                           │
│     │  1. POST /login                │                           │
│     │  (username + password)         │                           │
│     │───────────────────────────────▶│                           │
│     │                                │                           │
│     │                          ┌─────┴─────┐                    │
│     │                          │ 검증      │                     │
│     │                          │ 세션      │                     │
│     │                          │ 생성      │                     │
│     │                          │ ID=abc123 │                     │
│     │                          │ Redis/DB에│                     │
│     │                          │ 저장      │                     │
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
│     │                          │ 저장소에서│                     │
│     │                          │ 세션      │                     │
│     │                          │ 조회      │                     │
│     │                          └─────┬─────┘                    │
│     │  4. 사용자 데이터와 함께 응답  │                           │
│     │◀───────────────────────────────│                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 안전한 쿠키 속성

```
Set-Cookie: session_id=abc123;
            HttpOnly;        ← JavaScript로 접근 불가 (XSS 방지)
            Secure;          ← HTTPS를 통해서만 전송
            SameSite=Lax;    ← CSRF 방지 (교차 사이트 POST에서 전송 안 됨)
            Path=/;          ← 쿠키 범위
            Max-Age=3600;    ← 1시간 후 만료
            Domain=.app.com  ← 서브도메인에도 전송
```

| 속성 | 목적 | 권장 값 |
|-----|------|--------|
| `HttpOnly` | XSS로부터 쿠키 읽기 방지 | 항상 설정 |
| `Secure` | HTTPS 전용 | 프로덕션에서 항상 설정 |
| `SameSite` | CSRF 방지 | `Lax` (또는 민감한 작업에 `Strict`) |
| `Max-Age` | 세션 지속 시간 | 민감도에 따라 1-24시간 |
| `Path` | URL 범위 | `/` 또는 특정 경로 |

### 4.3 세션 보안 모범 사례

```python
"""
session_security.py - Flask를 사용한 안전한 세션 관리
"""
from flask import Flask, session, request, redirect, url_for
from datetime import timedelta
import secrets
import time


app = Flask(__name__)

# 세션 설정
app.config.update(
    SECRET_KEY=secrets.token_hex(32),
    SESSION_COOKIE_HTTPONLY=True,     # JavaScript 접근 방지
    SESSION_COOKIE_SECURE=True,       # HTTPS 전용
    SESSION_COOKIE_SAMESITE='Lax',    # CSRF 방지
    PERMANENT_SESSION_LIFETIME=timedelta(hours=1),  # 세션 타임아웃
    SESSION_COOKIE_NAME='__Host-session',  # __Host- 접두사는 Secure+Path=/를 강제
)


@app.before_request
def check_session_security():
    """세션 보안 정책을 시행하는 미들웨어"""
    if 'user_id' not in session:
        return  # 로그인하지 않음, 확인 건너뛰기

    # 1. 세션 타임아웃 (절대)
    created_at = session.get('created_at', 0)
    if time.time() - created_at > 3600:  # 1시간 절대 타임아웃
        session.clear()
        return redirect(url_for('login'))

    # 2. 유휴 타임아웃
    last_active = session.get('last_active', 0)
    if time.time() - last_active > 900:  # 15분 유휴 타임아웃
        session.clear()
        return redirect(url_for('login'))

    # 3. 마지막 활동 시간 업데이트
    session['last_active'] = time.time()

    # 4. IP 바인딩 (선택사항 - 모바일 사용자에게 문제 발생 가능)
    if session.get('ip_address') != request.remote_addr:
        # 의심스러운 활동 기록
        app.logger.warning(
            f"IP change detected for user {session.get('user_id')}: "
            f"{session.get('ip_address')} -> {request.remote_addr}"
        )


def regenerate_session(user_id: int):
    """
    인증 상태 변경 후 세션 ID 재생성.
    세션 고정 공격 방지.
    """
    # 필요한 데이터 보존
    old_data = dict(session)

    # 이전 세션 지우기
    session.clear()

    # 새 ID로 새 세션 생성
    session['user_id'] = user_id
    session['created_at'] = time.time()
    session['last_active'] = time.time()
    session['ip_address'] = request.remote_addr
    session.permanent = True  # PERMANENT_SESSION_LIFETIME 사용

    # 참고: Flask는 세션이 지워진 후 수정될 때
    # 자동으로 새 세션 ID를 생성함


@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    # 자격 증명 검증 (단순화)
    user = authenticate(username, password)
    if user:
        # 중요: 로그인 후 세션 재생성
        regenerate_session(user.id)
        return redirect(url_for('dashboard'))

    return "Invalid credentials", 401


@app.route('/logout')
def logout():
    """세션을 적절히 파괴"""
    session.clear()
    response = redirect(url_for('login'))
    # 명시적으로 쿠키 만료
    response.delete_cookie('__Host-session')
    return response
```

---

## 5. JSON 웹 토큰 (JWT)

### 5.1 JWT 구조

JWT는 점으로 구분된 세 개의 Base64URL 인코딩 부분으로 구성됩니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                     JWT 구조                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.                        │
│  eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4iLCJpYXQiOjE2M.  │
│  SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c                  │
│                                                                  │
│  ┌──────────────────┐                                           │
│  │     HEADER       │  ← 알고리즘 + 토큰 타입                    │
│  │  {               │                                            │
│  │    "alg": "HS256"│    HMAC SHA-256                           │
│  │    "typ": "JWT"  │    JSON Web Token                         │
│  │  }               │                                            │
│  └──────────────────┘                                           │
│           .                                                      │
│  ┌──────────────────┐                                           │
│  │     PAYLOAD      │  ← 클레임 (데이터)                         │
│  │  {               │                                            │
│  │    "sub": "1234" │    주체 (사용자 ID)                        │
│  │    "name": "John"│    커스텀 클레임                           │
│  │    "iat": 163... │    발행 시간                               │
│  │    "exp": 163... │    만료                                    │
│  │    "iss": "myapp"│    발행자                                  │
│  │    "aud": "api"  │    대상                                    │
│  │  }               │                                            │
│  └──────────────────┘                                           │
│           .                                                      │
│  ┌──────────────────┐                                           │
│  │    SIGNATURE     │  ← 무결성 검증                             │
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

### 5.2 JWT 서명 알고리즘

| 알고리즘 | 유형 | 키 | 사용 사례 |
|---------|------|---|---------|
| HS256 | 대칭 | 공유 비밀 | 단일 서비스 (동일한 키로 서명 및 검증) |
| RS256 | 비대칭 | RSA 키 쌍 | 마이크로서비스 (개인 키로 서명, 공개 키로 검증) |
| ES256 | 비대칭 | ECDSA 키 쌍 | RS256의 현대적 대안 (더 작은 키) |
| EdDSA | 비대칭 | Ed25519 쌍 | 최고 성능, 가장 작은 키 |
| **none** | **없음** | **없음** | **절대 사용 금지 - 치명적 취약점** |

### 5.3 Python JWT 구현

```python
"""
jwt_auth.py - JWT 생성, 검증 및 일반 패턴
pip install PyJWT cryptography
"""
import jwt
import time
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any


# ==============================================================
# 대칭 (HS256) - 단일 서비스 애플리케이션용
# ==============================================================

class JWTManagerSymmetric:
    """HMAC-SHA256 (대칭 키)을 사용하는 JWT 관리자"""

    def __init__(self, secret_key: str = None):
        # 프로덕션에서는 환경 변수에서 로드
        self.secret_key = secret_key or secrets.token_hex(32)
        self.algorithm = "HS256"

    def create_access_token(
        self,
        user_id: str,
        roles: list = None,
        expires_minutes: int = 15
    ) -> str:
        """짧은 수명의 접근 토큰 생성"""
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,           # 주체 (사용자 식별자)
            "iat": now,                # 발행 시간
            "exp": now + timedelta(minutes=expires_minutes),  # 만료
            "iss": "myapp",            # 발행자
            "aud": "myapp-api",        # 대상
            "type": "access",          # 토큰 타입
            "roles": roles or [],      # 사용자 역할
            "jti": secrets.token_hex(16),  # 고유 토큰 ID (폐기용)
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(
        self,
        user_id: str,
        expires_days: int = 30
    ) -> str:
        """긴 수명의 갱신 토큰 생성"""
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
        JWT 토큰 검증 및 디코딩.
        실패 시 jwt.InvalidTokenError 발생.
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],  # 중요: 항상 지정!
                issuer="myapp",
                audience="myapp-api",
                options={
                    "require": ["exp", "iat", "sub", "iss"],
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_iss": True,
                }
            )

            # 토큰 타입 검증
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
# 비대칭 (RS256) - 마이크로서비스용
# ==============================================================

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization


class JWTManagerAsymmetric:
    """RSA-SHA256 (비대칭 키)을 사용하는 JWT 관리자"""

    def __init__(self, private_key_pem: str = None, public_key_pem: str = None):
        if private_key_pem and public_key_pem:
            self.private_key = serialization.load_pem_private_key(
                private_key_pem.encode(), password=None
            )
            self.public_key = serialization.load_pem_public_key(
                public_key_pem.encode()
            )
        else:
            # 데모용 키 쌍 생성
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
            self.public_key = self.private_key.public_key()

        self.algorithm = "RS256"

    def create_token(self, payload: dict) -> str:
        """개인 키로 토큰 서명"""
        return jwt.encode(payload, self.private_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> dict:
        """공개 키로 토큰 검증"""
        return jwt.decode(
            token,
            self.public_key,
            algorithms=[self.algorithm],
        )

    def get_public_key_pem(self) -> str:
        """공개 키 내보내기 (다른 서비스와 공유)"""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()


# ==============================================================
# 토큰 갱신 패턴
# ==============================================================

class TokenService:
    """갱신 플로우를 포함한 완전한 토큰 서비스"""

    def __init__(self):
        self.jwt_manager = JWTManagerSymmetric()
        # 프로덕션에서는 Redis 또는 데이터베이스 사용
        self.revoked_tokens = set()

    def login(self, user_id: str, roles: list) -> Dict[str, str]:
        """로그인 시 접근 및 갱신 토큰 발행"""
        return {
            "access_token": self.jwt_manager.create_access_token(
                user_id, roles, expires_minutes=15
            ),
            "refresh_token": self.jwt_manager.create_refresh_token(
                user_id, expires_days=30
            ),
            "token_type": "Bearer",
            "expires_in": 900,  # 초 단위 15분
        }

    def refresh(self, refresh_token: str) -> Dict[str, str]:
        """갱신 토큰을 사용하여 새 접근 토큰 가져오기"""
        # 갱신 토큰 검증
        payload = self.jwt_manager.verify_token(
            refresh_token, expected_type="refresh"
        )

        # 토큰이 폐기되었는지 확인
        jti = payload.get("jti")
        if jti in self.revoked_tokens:
            raise jwt.InvalidTokenError("Token has been revoked")

        # 이전 갱신 토큰 폐기 (순환)
        self.revoked_tokens.add(jti)

        # 새 토큰 쌍 발행
        user_id = payload["sub"]
        return self.login(user_id, roles=[])  # DB에서 역할 다시 가져오기

    def revoke(self, token: str):
        """토큰 폐기 (로그아웃)"""
        try:
            payload = jwt.decode(
                token,
                self.jwt_manager.secret_key,
                algorithms=["HS256"],
                options={"verify_exp": False}  # 만료된 토큰 폐기 허용
            )
            jti = payload.get("jti")
            if jti:
                self.revoked_tokens.add(jti)
        except jwt.DecodeError:
            pass  # 잘못된 토큰, 폐기할 것 없음


# ==============================================================
# 데모
# ==============================================================

if __name__ == "__main__":
    print("=== 대칭 JWT (HS256) ===")
    manager = JWTManagerSymmetric()

    token = manager.create_access_token("user123", roles=["admin", "editor"])
    print(f"Token: {token[:50]}...")

    payload = manager.verify_token(token)
    print(f"Payload: {payload}")

    print("\n=== 토큰 서비스 ===")
    service = TokenService()

    tokens = service.login("user123", ["admin"])
    print(f"Access:  {tokens['access_token'][:50]}...")
    print(f"Refresh: {tokens['refresh_token'][:50]}...")

    # 갱신
    new_tokens = service.refresh(tokens["refresh_token"])
    print(f"New access: {new_tokens['access_token'][:50]}...")

    print("\n=== 비대칭 JWT (RS256) ===")
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

### 5.4 일반적인 JWT 함정

```
┌─────────────────────────────────────────────────────────────────┐
│                  JWT 보안 함정                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 알고리즘 "none" 공격                                          │
│     ─────────────────────                                        │
│     공격자가 헤더를 {"alg": "none"}으로 변경하고 서명 제거.       │
│     서버가 서명되지 않은 토큰 수락.                                │
│                                                                  │
│     수정: 허용된 알고리즘을 항상 지정:                             │
│     jwt.decode(token, key, algorithms=["HS256"])                 │
│     절대 algorithms=["none"]을 사용하거나 모든 알고리즘 허용 금지   │
│                                                                  │
│  2. 알고리즘 혼동 (RS256 → HS256)                                │
│     ──────────────────────────────────                           │
│     서버가 RS256 사용 (비대칭). 공격자가 HS256으로 변경하고       │
│     공개 키로 서명. 서버가 동일한 공개 키를 HMAC 비밀로 사용하여   │
│     검증 → 유효!                                                  │
│                                                                  │
│     수정: 예상 알고리즘 명시적 지정, "모든" 것이 아님             │
│     대칭/비대칭에 별도 키 사용                                    │
│                                                                  │
│  3. 만료 없음                                                     │
│     ─────────────                                                │
│     "exp" 클레임 없는 토큰은 영원히 유지됨.                       │
│                                                                  │
│     수정: 항상 exp 설정. 짧은 수명의 접근 토큰 사용 (15분)       │
│     긴 갱신 토큰과 함께.                                          │
│                                                                  │
│  4. 페이로드의 민감한 데이터                                       │
│     ─────────────────────────                                    │
│     JWT 페이로드는 Base64 인코딩됨, 암호화되지 않음.              │
│     누구나 디코딩하고 읽을 수 있음.                                │
│                                                                  │
│     수정: 비밀번호, PII 또는 비밀을 JWT 페이로드에 넣지 말 것     │
│     페이로드가 비공개여야 하면 JWE (JSON Web Encryption) 사용     │
│                                                                  │
│  5. 토큰을 폐기할 수 없음                                          │
│     ──────────────────                                           │
│     JWT는 상태 비저장 - 일단 발행되면 만료까지 유효함.             │
│     로그아웃이 토큰을 무효화하지 않음.                             │
│                                                                  │
│     수정: 짧은 만료 + 로그아웃용 토큰 블록리스트 (Redis) 사용     │
│     또는 "jti" 클레임 사용 및 폐기된 토큰 ID 추적                 │
│                                                                  │
│  6. JWT를 localStorage에 저장                                    │
│     ─────────────────────────                                    │
│     localStorage는 페이지의 모든 JavaScript로 접근 가능하여       │
│     XSS에 취약함.                                                 │
│                                                                  │
│     수정: HttpOnly 쿠키에 저장 (XSS 면역)                        │
│     또는 메모리 내 저장 + HttpOnly 쿠키의 갱신 토큰 사용         │
│                                                                  │
│  7. 약한 비밀 키                                                  │
│     ────────────────                                             │
│     HMAC 서명에 짧거나 추측 가능한 비밀 사용.                      │
│     공격자가 키를 무차별 대입할 수 있음.                           │
│                                                                  │
│     수정: 최소 256비트 엔트로피 사용:                             │
│     secret = secrets.token_hex(32)  # 256 비트                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. 비밀번호 재설정 플로우

### 6.1 안전한 비밀번호 재설정 설계

```
┌─────────────────────────────────────────────────────────────────┐
│              안전한 비밀번호 재설정 플로우                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  사용자                    서버                     이메일        │
│   │                          │                          │        │
│   │ 1. "비밀번호 찾기"       │                          │        │
│   │  (이메일 입력)           │                          │        │
│   │─────────────────────────▶│                          │        │
│   │                          │                          │        │
│   │                    ┌─────┴─────┐                   │        │
│   │                    │ 랜덤      │                    │        │
│   │                    │ 토큰      │                    │        │
│   │                    │ 생성      │                    │        │
│   │                    │ hash(tkn) │                    │        │
│   │                    │ 저장      │                    │        │
│   │                    │ + 만료    │                    │        │
│   │                    └─────┬─────┘                   │        │
│   │                          │                          │        │
│   │ 2. "이메일 확인"         │  3. 토큰과 함께         │        │
│   │  (이메일 존재 여부와     │  재설정 링크 전송       │        │
│   │   관계없이 동일한 응답!) │─────────────────────────▶│       │
│   │◀─────────────────────────│                          │        │
│   │                          │                          │        │
│   │ 4. 이메일에서 링크 클릭  │                          │        │
│   │  /reset?token=abc123     │                          │        │
│   │─────────────────────────▶│                          │        │
│   │                          │                          │        │
│   │ 5. 새 비밀번호 폼        │                          │        │
│   │◀─────────────────────────│                          │        │
│   │                          │                          │        │
│   │ 6. 새 비밀번호 제출      │                          │        │
│   │─────────────────────────▶│                          │        │
│   │                    ┌─────┴─────┐                   │        │
│   │                    │ 토큰      │                    │        │
│   │                    │ 검증      │                    │        │
│   │                    │ 비밀번호  │                    │        │
│   │                    │ 업데이트  │                    │        │
│   │                    │ 토큰      │                    │        │
│   │                    │ 무효화    │                    │        │
│   │                    │ 세션      │                    │        │
│   │                    │ 무효화    │                    │        │
│   │                    └─────┬─────┘                   │        │
│   │                          │                          │        │
│   │ 7. "비밀번호 업데이트됨" │                          │        │
│   │◀─────────────────────────│                          │        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 구현

```python
"""
password_reset.py - 안전한 비밀번호 재설정 구현
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
    """안전한 비밀번호 재설정 토큰 관리"""

    TOKEN_EXPIRY_MINUTES = 30
    MAX_REQUESTS_PER_HOUR = 3

    def __init__(self):
        # 프로덕션에서는 데이터베이스 사용
        self.tokens = {}  # token_hash -> ResetToken
        self.rate_limit = {}  # email -> [timestamps]

    def request_reset(self, email: str, user_id: Optional[int]) -> Optional[str]:
        """
        비밀번호 재설정 토큰 생성.
        토큰(이메일로 전송)을 반환하거나 속도 제한 시 None 반환.

        보안: 이메일 존재 여부와 관계없이 사용자에게
        항상 동일한 응답 반환.
        """
        # 속도 제한
        now = time.time()
        if email in self.rate_limit:
            recent = [t for t in self.rate_limit[email] if now - t < 3600]
            if len(recent) >= self.MAX_REQUESTS_PER_HOUR:
                return None  # 속도 제한됨
            self.rate_limit[email] = recent
        else:
            self.rate_limit[email] = []

        self.rate_limit[email].append(now)

        # 사용자가 존재하지 않으면 조용히 None 반환
        # (호출자는 여전히 "이메일 확인" 메시지 표시)
        if user_id is None:
            return None

        # 암호학적으로 안전한 토큰 생성
        token = secrets.token_urlsafe(32)  # 256비트 엔트로피

        # 토큰의 해시 저장 (토큰 자체가 아님!)
        # 데이터베이스가 침해되어도 공격자가 해시를 사용할 수 없음
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        # 이 사용자의 기존 토큰 무효화
        self.tokens = {
            h: t for h, t in self.tokens.items()
            if t.user_id != user_id
        }

        # 새 토큰 저장
        self.tokens[token_hash] = ResetToken(
            token_hash=token_hash,
            user_id=user_id,
            created_at=now,
            expires_at=now + (self.TOKEN_EXPIRY_MINUTES * 60),
        )

        return token  # 이메일 링크에 이것을 보냄

    def verify_and_consume_token(self, token: str) -> Optional[int]:
        """
        재설정 토큰 검증 및 user_id 반환.
        토큰 사용됨 (일회용).
        토큰이 유효하지 않거나/만료되었거나/사용된 경우 None 반환.
        """
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        reset_token = self.tokens.get(token_hash)
        if not reset_token:
            return None

        # 만료 확인
        if time.time() > reset_token.expires_at:
            del self.tokens[token_hash]
            return None

        # 이미 사용되었는지 확인
        if reset_token.used:
            return None

        # 사용됨으로 표시
        reset_token.used = True

        # 정리
        del self.tokens[token_hash]

        return reset_token.user_id

    def cleanup_expired(self):
        """만료된 토큰 제거 (주기적으로 실행)"""
        now = time.time()
        self.tokens = {
            h: t for h, t in self.tokens.items()
            if t.expires_at > now and not t.used
        }


# Flask 라우트 예제
from flask import Flask, request, jsonify

app = Flask(__name__)
reset_service = PasswordResetService()


@app.route('/api/forgot-password', methods=['POST'])
def forgot_password():
    email = request.json.get('email', '').strip().lower()

    if not email:
        return jsonify({"error": "Email required"}), 400

    # 사용자 조회 (찾지 못하면 None 반환 가능)
    user = find_user_by_email(email)  # DB 조회
    user_id = user.id if user else None

    token = reset_service.request_reset(email, user_id)

    if token and user:
        # 재설정 링크가 포함된 이메일 전송
        reset_link = f"https://myapp.com/reset-password?token={token}"
        send_reset_email(email, reset_link)  # 이메일 함수

    # 항상 동일한 응답 반환 (이메일 열거 방지)
    return jsonify({
        "message": "If that email is registered, you will receive a reset link."
    })


@app.route('/api/reset-password', methods=['POST'])
def reset_password():
    token = request.json.get('token')
    new_password = request.json.get('new_password')

    if not token or not new_password:
        return jsonify({"error": "Token and new password required"}), 400

    # 새 비밀번호 검증
    # (password_policy.check_password_strength 사용)

    user_id = reset_service.verify_and_consume_token(token)
    if not user_id:
        return jsonify({"error": "Invalid or expired token"}), 400

    # 비밀번호 업데이트
    update_user_password(user_id, new_password)  # 해시 및 저장

    # 이 사용자의 모든 기존 세션 무효화
    invalidate_all_sessions(user_id)

    return jsonify({"message": "Password updated successfully"})
```

**주요 보안 속성:**

| 속성 | 구현 |
|-----|------|
| 토큰 엔트로피 | `secrets.token_urlsafe(32)` - 256비트 |
| 토큰 저장 | 해시만 저장, 평문 절대 안 됨 |
| 일회용 | 첫 사용 시 토큰 소비됨 |
| 시간 제한 | 30분 만료 |
| 속도 제한 | 이메일당 시간당 최대 3회 요청 |
| 열거 방지 | 이메일 존재 여부와 관계없이 동일한 응답 |
| 세션 무효화 | 재설정 후 모든 세션 지움 |

---

## 7. 생체 인증

### 7.1 개요

```
┌─────────────────────────────────────────────────────────────────┐
│              생체 인증 유형                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  생리적:                                                         │
│  ├── 지문       - 가장 일반적, 성숙한 기술                        │
│  ├── 얼굴       - 모바일에서 널리 보급 (Face ID)                  │
│  ├── 홍채       - 높은 정확도, 비싸                              │
│  ├── 망막       - 매우 높은 정확도, 침습적                        │
│  └── 손바닥/정맥 - 비접촉, 점점 인기                             │
│                                                                  │
│  행동:                                                           │
│  ├── 음성       - 전화 시스템에 편리                              │
│  ├── 타이핑 리듬 - 지속적 인증                                   │
│  ├── 걸음걸이    - 걷는 패턴 인식                                │
│  └── 서명       - 동적 분석 (압력, 속도)                         │
│                                                                  │
│  주요 지표:                                                      │
│  ┌─────────────┬──────────────────────────────────────────┐     │
│  │ FAR         │ 거짓 수락률                               │     │
│  │             │ (사칭자를 진짜로 수락)                     │     │
│  ├─────────────┼──────────────────────────────────────────┤     │
│  │ FRR         │ 거짓 거부율                               │     │
│  │             │ (진짜 사용자 거부)                         │     │
│  ├─────────────┼──────────────────────────────────────────┤     │
│  │ EER         │ 동일 오류율                               │     │
│  │             │ (FAR = FRR인 지점; 낮을수록 좋음)         │     │
│  └─────────────┴──────────────────────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 생체 템플릿 보안

생체 데이터는 **변경할 수 없기** 때문에 침해되면 비밀번호와 달리 특별한 처리가 필요합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│           생체 템플릿 보호                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  잘못됨: 원시 생체 데이터 저장                                    │
│  ┌──────────┐    ┌──────────────┐                               │
│  │ 원시 스캔│───▶│ 데이터베이스에│  ← 침해 시, 게임 오버        │
│  └──────────┘    │ 이미지 저장  │    (지문 변경 불가)           │
│                  └──────────────┘                                │
│                                                                  │
│  올바름: 취소 가능한 생체 / 템플릿 보호                           │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐           │
│  │ 원시 스캔│───▶│ 특징         │───▶│ 변환        │           │
│  └──────────┘    │ 추출         │    │ (일방향,    │           │
│                  │ (미뉴샤)     │    │  취소 가능) │            │
│                  └──────────────┘    └──────┬──────┘           │
│                                             │                    │
│                                      ┌──────▼──────┐           │
│                                      │ 템플릿      │            │
│                                      │ 저장        │            │
│                                      │ (폐기 가능) │            │
│                                      └─────────────┘           │
│                                                                  │
│  장치 내 처리 (선호):                                            │
│  - 생체 매칭이 장치에서 발생 (Secure Enclave)                   │
│  - 서버는 생체 데이터를 보지 못함                                │
│  - 매칭 시 장치가 암호화 키 릴리스                               │
│  - Apple Face ID 및 Touch ID가 작동하는 방식                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 생체 트레이드오프

| 요소 | 비밀번호 | TOTP | 생체 인증 | FIDO2 |
|-----|---------|------|---------|-------|
| 변경 가능 | 예 | 예 (재등록) | **아니오** | 예 (재등록) |
| 공유 가능 | 예 (나쁨) | 예 (나쁨) | 어려움 | 아니오 |
| 잊어버릴 수 있음 | 예 | 해당 없음 | 아니오 | 해당 없음 |
| 스푸핑 위험 | 피싱 | 피싱 | 프레젠테이션 공격 | 매우 낮음 |
| 개인정보 우려 | 낮음 | 낮음 | **높음** | 낮음 |
| 가장 좋은 용도 | 주요 요소 | 2차 요소 | 2차 요소 (로컬) | 2차 요소 또는 비밀번호 없음 |

---

## 8. 인증 아키텍처 패턴

### 8.1 올바른 패턴 선택

```
┌─────────────────────────────────────────────────────────────────┐
│          인증 패턴 결정 트리                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  어떤 유형의 애플리케이션?                                        │
│  │                                                               │
│  ├── 전통적인 웹 앱 (서버 렌더링)                                 │
│  │   └── 사용: 서버 측 세션 + HttpOnly 쿠키                      │
│  │                                                               │
│  ├── SPA (React, Vue 등)                                         │
│  │   └── 사용: OAuth 2.0 + PKCE                                 │
│  │       메모리에 접근 토큰 저장                                  │
│  │       HttpOnly 쿠키에 갱신 토큰 저장                          │
│  │                                                               │
│  ├── 모바일 앱                                                    │
│  │   └── 사용: OAuth 2.0 + PKCE + 보안 저장소                    │
│  │       (iOS Keychain, Android Keystore)                        │
│  │                                                               │
│  ├── API 간 (서비스 메시)                                         │
│  │   └── 사용: Client Credentials 플로우 + mTLS                 │
│  │       또는: 서비스 메시 (Istio) 자동 mTLS                      │
│  │                                                               │
│  └── 마이크로서비스                                               │
│      └── 사용: JWT (RS256) 중앙 인증 서비스                       │
│          인증 서비스가 토큰 발행                                   │
│          각 서비스가 공개 키로 검증                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 중앙 인증 (인증 서비스 패턴)

```
┌─────────────────────────────────────────────────────────────────┐
│         마이크로서비스 인증 아키텍처                                │
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
│     │ 서비스   │  │ 서비스   │  │ 서비스   │                    │
│     │    A     │  │    B     │  │    C     │                    │
│     └──────────┘  └──────────┘  └──────────┘                   │
│            │              │              │                        │
│            └──────────────┼──────────────┘                      │
│                           │                                      │
│                           ▼                                      │
│                    ┌──────────────┐                              │
│                    │  인증        │                              │
│                    │  서비스      │                              │
│                    │  ─────────── │                              │
│                    │  - 로그인    │                              │
│                    │  - 토큰      │                              │
│                    │    발행      │                              │
│                    │  - 사용자    │                              │
│                    │    관리      │                              │
│                    │  - JWKS      │                              │
│                    │    엔드포인트│                              │
│                    └──────────────┘                              │
│                                                                  │
│  플로우:                                                         │
│  1. 클라이언트가 인증 서비스로 인증 → JWT 받음                    │
│  2. 클라이언트가 API Gateway에 JWT 전송                          │
│  3. Gateway가 JWT 서명 검증 (JWKS의 공개 키 사용)               │
│  4. Gateway가 요청 + JWT 클레임을 서비스로 전달                  │
│  5. 서비스들이 재검증 없이 검증된 클레임 신뢰                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. 연습 문제

### 연습문제 1: 안전한 비밀번호 저장 구현

완전한 사용자 등록 및 로그인 시스템 구축:

```python
"""
연습문제: 다음 UserService 클래스를 구현하세요.
비밀번호 해싱에 argon2를 사용하세요.
입력 검증과 적절한 오류 처리를 포함하세요.
"""

class UserService:
    def register(self, username: str, email: str, password: str) -> dict:
        """
        새 사용자 등록.
        - 비밀번호 강도 검증 (최소 12자, 일반적이지 않음)
        - 사용자 이름/이메일 고유성 확인
        - argon2id로 비밀번호 해시
        - 사용자 정보 반환 (비밀번호 해시 제외)
        """
        pass

    def login(self, username: str, password: str) -> dict:
        """
        사용자 인증.
        - 자격 증명 검증
        - 5회 실패 후 계정 잠금 구현
        - 성공적인 로그인 시 세션 재생성
        - 접근 + 갱신 토큰 반환
        """
        pass

    def change_password(self, user_id: int, old_password: str,
                        new_password: str) -> bool:
        """
        사용자 비밀번호 변경.
        - 이전 비밀번호 검증
        - 새 비밀번호 검증
        - 모든 기존 세션 무효화
        """
        pass
```

### 연습문제 2: TOTP 통합

UserService에 TOTP 기반 2FA 추가:

```python
"""
연습문제: UserService에 이 메서드들을 추가하세요.
pyotp 라이브러리를 사용하세요.
"""

class UserService:
    # ... (연습문제 1에서)

    def enable_2fa(self, user_id: int) -> dict:
        """
        사용자에 대해 TOTP 2FA 활성화.
        반환: {"secret": ..., "qr_uri": ..., "backup_codes": [...]}
        """
        pass

    def verify_2fa_setup(self, user_id: int, code: str) -> bool:
        """설정 확인을 위한 초기 TOTP 코드 검증"""
        pass

    def login_2fa(self, username: str, password: str,
                  totp_code: str) -> dict:
        """
        2FA로 로그인.
        - 먼저 비밀번호 검증
        - 그 다음 TOTP 코드 검증
        - 대체 수단으로 백업 코드 지원
        """
        pass
```

### 연습문제 3: JWT 보안 감사

이 코드의 보안 문제를 식별하고 수정:

```python
"""
연습문제: 이 JWT 구현의 모든 보안 문제를 찾아 수정하세요.
"""
import jwt
import time

SECRET = "mysecret"  # 문제 1: ???

def create_token(user_id):
    payload = {
        "user_id": user_id,
        "password": get_user_password(user_id),  # 문제 2: ???
        "admin": False,
    }
    return jwt.encode(payload, SECRET)  # 문제 3: ???

def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET, algorithms=["HS256", "none"])  # 문제 4: ???
        return payload
    except:  # 문제 5: ???
        return None

def protected_route(token):
    payload = verify_token(token)
    if payload:
        if payload.get("admin"):  # 문제 6: ???
            return admin_dashboard()
        return user_dashboard(payload["user_id"])
    return "Unauthorized"
```

### 연습문제 4: OAuth 2.0 플로우 구현

PKCE를 사용한 완전한 OAuth 2.0 클라이언트 구현:

```python
"""
연습문제: 이 OAuth 2.0 클라이언트 구현을 완성하세요.
PKCE, state 검증, 안전한 토큰 저장을 포함하세요.
"""

class OAuthClient:
    def __init__(self, client_id: str, auth_url: str,
                 token_url: str, redirect_uri: str):
        pass

    def start_auth_flow(self) -> str:
        """
        권한 부여 URL 생성.
        PKCE code_challenge 및 state 파라미터 포함.
        사용자를 리디렉션할 URL 반환.
        """
        pass

    def handle_callback(self, callback_url: str) -> dict:
        """
        OAuth 콜백 처리.
        - state 파라미터 검증
        - code_verifier를 사용하여 코드를 토큰으로 교환
        - 토큰 반환
        """
        pass

    def refresh_access_token(self, refresh_token: str) -> dict:
        """만료된 접근 토큰 갱신"""
        pass
```

### 연습문제 5: 비밀번호 재설정 보안 검토

이 비밀번호 재설정 플로우를 검토하고 모든 보안 문제 나열:

```python
"""
연습문제: 이 코드의 모든 보안 취약점을 식별하세요.
수정된 버전을 작성하세요.
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
        return "Email not found", 404  # 문제: ???

    # 4자리 재설정 코드 생성
    code = ''.join(random.choices(string.digits, k=4))  # 문제: ???
    reset_codes[email] = code  # 문제: ???

    send_email(email, f"Your reset code is: {code}")
    return "Code sent"

@app.route('/reset', methods=['POST'])
def reset_password():
    email = request.form['email']
    code = request.form['code']
    new_password = request.form['password']

    if reset_codes.get(email) == code:  # 문제: ???
        user = db.find_user(email=email)
        user.password = new_password  # 문제: ???
        db.save(user)
        return "Password updated"

    return "Invalid code", 400
```

---

## 10. 요약

```
┌─────────────────────────────────────────────────────────────────┐
│              인증 시스템 요약                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  비밀번호 저장:                                                   │
│  - Argon2id 또는 bcrypt 사용 (MD5/SHA1/SHA256 단독 절대 안 됨)  │
│  - 자동 솔팅, 키 스트레칭                                         │
│  - NIST SP 800-63B 지침 따르기                                   │
│                                                                  │
│  다중 인증:                                                       │
│  - TOTP가 최소 권장 2차 요소                                      │
│  - FIDO2/WebAuthn이 황금 표준 (피싱 저항)                        │
│  - SMS는 가장 약한 2차 요소 (SIM 스와핑)                          │
│  - 계정 복구를 위해 항상 백업 코드 제공                            │
│                                                                  │
│  OAuth 2.0 / OIDC:                                              │
│  - PKCE와 함께 권한 부여 코드 플로우 사용                         │
│  - 항상 state 파라미터 검증 (CSRF)                               │
│  - OIDC가 OAuth 위에 신원 레이어 추가 (ID 토큰)                   │
│                                                                  │
│  세션 & JWT:                                                     │
│  - HttpOnly + Secure + SameSite 쿠키                            │
│  - 로그인 후 세션 ID 재생성                                       │
│  - JWT: 짧은 수명 접근 + 긴 수명 갱신                             │
│  - JWT 검증에서 항상 허용 알고리즘 지정                           │
│  - JWT 페이로드에 민감한 데이터 절대 저장 금지                     │
│                                                                  │
│  비밀번호 재설정:                                                 │
│  - 암호학적으로 랜덤한 토큰 (256+ 비트)                           │
│  - 토큰 자체가 아닌 토큰의 해시 저장                              │
│  - 일회용, 시간 제한 (30분)                                       │
│  - 이메일 존재 여부와 관계없이 동일한 응답                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

**이전**: [04. TLS/SSL 및 공개키 기반 구조](./04_TLS_and_PKI.md) | **다음**: [06. 접근 제어 및 권한 부여](06_Authorization.md)
