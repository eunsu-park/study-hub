# Secrets Management와 환경 설정

**이전**: [10_API_Security.md](./10_API_Security.md) | **다음**: [12_Container_Security.md](./12_Container_Security.md)

---

Secrets — API 키, 데이터베이스 비밀번호, 암호화 키, OAuth 클라이언트 비밀 — 는 모든 애플리케이션의 가장 중요한 자산입니다. 하나의 유출된 secret만으로도 전체 시스템이 손상될 수 있습니다. 그럼에도 불구하고 secrets 관리는 소프트웨어 개발에서 가장 흔히 잘못 다루어지는 영역 중 하나로 남아있습니다. 이 레슨에서는 secrets의 전체 생명주기를 다룹니다: 저장, 교체, 런타임 주입, 우발적 유출 스캐닝, CI/CD 파이프라인과 클라우드 환경에서의 관리 방법.

## 학습 목표

- 12-factor app의 구성 및 secrets 접근 방식 이해
- python-dotenv를 사용하여 환경 변수와 .env 파일을 안전하게 사용
- 다운타임 없는 secret 교체 전략 구현
- 중앙 집중식 secrets 관리를 위한 HashiCorp Vault 구성
- 클라우드 네이티브 secret 저장소(AWS Secrets Manager, GCP Secret Manager) 사용
- git 히스토리에서 유출된 secrets를 스캐닝 도구로 감지
- CI/CD 파이프라인에서 secrets 관리(GitHub Actions, GitLab CI)
- 구성 파일의 암호화
- secret 노출로 이어지는 일반적인 실수 방지

---

## 1. Secrets 기초

### 1.1 무엇이 Secret인가?

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Secrets의 유형                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  자격 증명                                                   │    │
│  │  ├── 데이터베이스 비밀번호 (PostgreSQL, MySQL, Redis)        │    │
│  │  ├── 서비스 계정 비밀번호                                    │    │
│  │  ├── SMTP/이메일 자격 증명                                   │    │
│  │  └── SSH 비밀번호                                            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  API 키와 토큰                                               │    │
│  │  ├── 서드파티 API 키 (Stripe, Twilio, AWS)                  │    │
│  │  ├── OAuth 클라이언트 비밀                                   │    │
│  │  ├── JWT 서명 키                                             │    │
│  │  └── 개인 액세스 토큰                                        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  암호화 자료                                                 │    │
│  │  ├── TLS 개인 키                                             │    │
│  │  ├── 암호화 키 (AES, RSA 개인 키)                            │    │
│  │  ├── SSH 개인 키                                             │    │
│  │  └── 코드 서명 키                                            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  인프라 Secrets                                              │    │
│  │  ├── 클라우드 제공자 자격 증명 (AWS 액세스 키)               │    │
│  │  ├── 컨테이너 레지스트리 자격 증명                           │    │
│  │  ├── Kubernetes secrets                                     │    │
│  │  └── Terraform 상태 암호화 키                                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  경험 법칙: 노출되면 피해를 입힐 수 있다면, 그것은 secret입니다.      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Secret 생명주기

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Secret 생명주기                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 생성 ──▶ 2. 저장 ──▶ 3. 배포 ──▶ 4. 사용                        │
│       │                │               │                  │          │
│       │                │               │                  │          │
│       ▼                ▼               ▼                  ▼          │
│  강력한 무작위       저장 시          보안 채널         최소 노출      │
│  생성                암호화           (TLS, IAM)                      │
│                                                                      │
│  5. 교체 ──▶ 6. 폐기 ──▶ 7. 감사                                    │
│       │                │                │                            │
│       ▼                ▼                ▼                            │
│  자동화된           손상 시          액세스 및                        │
│  주기적             즉시             변경 기록                        │
│                                                                      │
│  주요 원칙:                                                           │
│  • 최소 권한: 필요한 사람에게만 액세스 부여                            │
│  • 단기간: 영구 자격 증명보다 임시 자격 증명 선호                      │
│  • 저장 시 암호화: 절대 평문으로 secrets 저장 금지                    │
│  • 감사 가능: 누가 언제 어떤 secret에 액세스했는지 기록                │
│  • 교체 가능: secret 교체를 처리할 수 있도록 시스템 설계               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 환경 변수와 .env 파일

### 2.1 12-Factor App 접근 방식

```
┌─────────────────────────────────────────────────────────────────────┐
│                    12-Factor App: 구성                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Factor III: 환경에 구성 저장                                         │
│                                                                      │
│  ┌─────────────────┐                                                │
│  │  애플리케이션     │                                               │
│  │  코드            │ ← 동일한 코드가 모든 곳에 배포됨                 │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │  개발 환경       │  │  스테이징        │  │  프로덕션        │    │
│  │  DB=localhost    │  │  DB=staging.db   │  │  DB=prod.db      │    │
│  │  DEBUG=true      │  │  DEBUG=false     │  │  DEBUG=false     │    │
│  │  KEY=dev_key     │  │  KEY=stage_key   │  │  KEY=prod_key    │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
│                                                                      │
│  배포 간 변경되는 구성:                                                │
│  ✓ 데이터베이스 URL, API 키, 기능 플래그                              │
│                                                                      │
│  변경되지 않는 구성:                                                   │
│  ✗ 프레임워크 설정, 로깅 포맷, 라우트                                  │
│  (이것들은 코드/구성 파일에 속하며, 환경 변수가 아님)                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 python-dotenv 사용

```python
"""
python-dotenv를 사용하여 .env 파일에서 구성 로드.
pip install python-dotenv
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── 기본 사용법 ──────────────────────────────────────────────────
# 현재 디렉토리에서 .env 파일 로드
load_dotenv()

# 환경 변수 액세스
database_url = os.getenv('DATABASE_URL')
secret_key = os.getenv('SECRET_KEY')
debug = os.getenv('DEBUG', 'false').lower() == 'true'

# ── 특정 경로에서 로드 ─────────────────────────────────────────
env_path = Path(__file__).parent / '.env.production'
load_dotenv(dotenv_path=env_path)

# ── 기존 환경 변수 덮어쓰기 ──────────────────────────────────────
# 기본적으로 dotenv는 기존 환경 변수를 덮어쓰지 않음
# 이는 안전함: 시스템 환경 변수가 우선순위를 가짐
load_dotenv(override=False)  # 기본 동작

# 덮어쓰기 강제(거의 필요하지 않음):
load_dotenv(override=True)


# ── 구조화된 구성 클래스 ──────────────────────────────────────
from dataclasses import dataclass

@dataclass
class AppConfig:
    """환경에서 로드된 애플리케이션 구성."""

    # 데이터베이스
    database_url: str
    database_pool_size: int

    # 보안
    secret_key: str
    jwt_secret: str
    jwt_expiry_minutes: int

    # 외부 서비스
    stripe_api_key: str
    sendgrid_api_key: str

    # 애플리케이션
    debug: bool
    log_level: str

    @classmethod
    def from_env(cls) -> 'AppConfig':
        """환경 변수에서 구성 로드."""
        load_dotenv()

        def require_env(key: str) -> str:
            """필수 환경 변수를 가져오거나 오류 발생."""
            value = os.getenv(key)
            if value is None:
                raise EnvironmentError(
                    f"Required environment variable '{key}' is not set. "
                    f"Check your .env file or environment configuration."
                )
            return value

        return cls(
            database_url=require_env('DATABASE_URL'),
            database_pool_size=int(os.getenv('DATABASE_POOL_SIZE', '5')),
            secret_key=require_env('SECRET_KEY'),
            jwt_secret=require_env('JWT_SECRET'),
            jwt_expiry_minutes=int(os.getenv('JWT_EXPIRY_MINUTES', '15')),
            stripe_api_key=require_env('STRIPE_API_KEY'),
            sendgrid_api_key=require_env('SENDGRID_API_KEY'),
            debug=os.getenv('DEBUG', 'false').lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
        )


# ── 사용법 ────────────────────────────────────────────────────────
config = AppConfig.from_env()
print(f"Debug mode: {config.debug}")
print(f"Database: {config.database_url[:20]}...")  # 전체 URL 로깅하지 않기
```

### 2.3 .env 파일

```bash
# ── .env 파일 (절대 git에 커밋하지 말 것) ─────────────────────────

# 데이터베이스
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
DATABASE_POOL_SIZE=5

# 보안
SECRET_KEY=your-256-bit-secret-key-here-change-me
JWT_SECRET=another-different-secret-key-for-jwt

# 외부 API
STRIPE_API_KEY=sk_test_EXAMPLE_KEY_REPLACE_ME
SENDGRID_API_KEY=SG.xxxxxxxxxxxxx

# 애플리케이션
DEBUG=false
LOG_LEVEL=INFO
```

```bash
# ── .env.example 파일 (git에 커밋할 것) ───────────────────────────
# 이 파일을 .env로 복사하고 값을 채우세요
# cp .env.example .env

# 데이터베이스
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
DATABASE_POOL_SIZE=5

# 보안 (생성: python -c "import secrets; print(secrets.token_hex(32))")
SECRET_KEY=change-me-generate-a-real-secret
JWT_SECRET=change-me-use-a-different-secret

# 외부 API
STRIPE_API_KEY=sk_test_your_test_key_here
SENDGRID_API_KEY=SG.your_api_key_here

# 애플리케이션
DEBUG=true
LOG_LEVEL=DEBUG
```

### 2.4 .gitignore 구성

```gitignore
# ── Secrets 및 환경 파일 ────────────────────────────────────────
.env
.env.local
.env.production
.env.staging
.env.*.local

# 예제 파일 유지
!.env.example
!.env.template

# 개인 키
*.pem
*.key
*.p12
*.pfx

# 클라우드 자격 증명
credentials.json
service-account*.json
.gcloud/
.aws/credentials

# IDE secrets
.idea/dataSources/
.vscode/settings.json
```

### 2.5 Pydantic Settings (타입 안전 구성)

```python
"""
Pydantic Settings를 사용한 타입 안전 구성.
pip install pydantic-settings
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr, PostgresDsn
from typing import Optional


class Settings(BaseSettings):
    """검증 및 타입 강제가 있는 애플리케이션 설정."""

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        # 충돌을 피하기 위해 모든 환경 변수 앞에 APP_ 접두사 추가
        env_prefix='APP_',
    )

    # 데이터베이스
    database_url: PostgresDsn
    database_pool_size: int = Field(default=5, ge=1, le=50)

    # 보안 — SecretStr은 로그와 repr에서 값을 숨김
    secret_key: SecretStr
    jwt_secret: SecretStr
    jwt_expiry_minutes: int = Field(default=15, ge=1, le=1440)

    # 외부 API
    stripe_api_key: SecretStr
    sendgrid_api_key: Optional[SecretStr] = None

    # 애플리케이션
    debug: bool = False
    log_level: str = Field(default="INFO", pattern=r'^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$')

    # 서버
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)


# ── 사용법 ────────────────────────────────────────────────────────
settings = Settings()

# SecretStr은 우발적 로깅 방지
print(settings.secret_key)
# 출력: SecretStr('**********')

# 필요할 때 실제 값 액세스
actual_key = settings.secret_key.get_secret_value()

# 비밀이 아닌 설정은 안전하게 출력
print(f"Debug: {settings.debug}")
print(f"Port: {settings.port}")

# 이것은 secret 값을 드러내지 않음
print(settings.model_dump())
# {'database_url': ..., 'secret_key': SecretStr('**********'), ...}
```

---

## 3. Secret 교체 전략

### 3.1 왜 Secrets를 교체하는가?

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Secret 교체 이유                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 손상의 폭발 반경 제한                                              │
│     이전 secret 손상됨 ──▶ 다음 교체까지만 유효                        │
│                                                                      │
│  2. 규정 준수 요구사항                                                 │
│     PCI DSS, SOC 2, HIPAA는 주기적 교체 요구                           │
│                                                                      │
│  3. 인사 변경                                                         │
│     직원 퇴사 ──▶ 그들이 액세스한 모든 secrets 교체                     │
│                                                                      │
│  4. 도난당한 secrets의 가치 감소                                       │
│     단기간 secrets는 공격자에게 덜 유용                                 │
│                                                                      │
│  교체 빈도 권장사항:                                                   │
│  ├── API 키:         90일마다                                         │
│  ├── 데이터베이스 비밀번호: 30-90일마다                                │
│  ├── TLS 인증서:     만료 전 (ACME/Let's Encrypt로 자동화)             │
│  ├── JWT 서명 키:    30일마다                                         │
│  └── 암호화 키:      365일마다 (재암호화 계획 포함)                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 다운타임 없는 교체 패턴

```
┌─────────────────────────────────────────────────────────────────────┐
│               다운타임 없는 Secret 교체                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  단계 1: 새 secret 생성 (이전 것은 활성 유지)                          │
│  ┌──────────┐                                                       │
│  │ Secret A │ ← 현재 (활성)                                         │
│  │ Secret B │ ← 새것 (활성)                                         │
│  └──────────┘                                                       │
│  두 secrets가 동시에 유효함                                           │
│                                                                      │
│  단계 2: 모든 소비자를 새 secret 사용으로 업데이트                      │
│  ┌──────────┐                                                       │
│  │ Secret A │ ← 이전 것 (유예 기간 동안 여전히 활성)                  │
│  │ Secret B │ ← 모든 서비스가 이제 이것을 사용                        │
│  └──────────┘                                                       │
│                                                                      │
│  단계 3: 유예 기간 후 이전 secret 폐기                                 │
│  ┌──────────┐                                                       │
│  │ Secret A │ ← 폐기됨                                               │
│  │ Secret B │ ← 유일한 활성 secret                                   │
│  └──────────┘                                                       │
│                                                                      │
│  이 이중 secret 기간은 교체 중 다운타임을 방지합니다.                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
"""
JWT 서명 키에 대한 다운타임 없는 secret 교체 구현.
"""
import jwt
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional

@dataclass
class SigningKey:
    """메타데이터를 가진 JWT 서명 키."""
    key_id: str
    secret: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    revoked: bool = False


class KeyRotationManager:
    """다운타임 없이 JWT 서명 키 교체를 관리."""

    def __init__(self, rotation_interval_days: int = 30,
                 grace_period_days: int = 7):
        self.rotation_interval = timedelta(days=rotation_interval_days)
        self.grace_period = timedelta(days=grace_period_days)
        self.keys: list[SigningKey] = []
        self._generate_new_key()

    def _generate_new_key(self) -> SigningKey:
        """새 서명 키 생성."""
        import secrets
        now = datetime.now(timezone.utc)
        key = SigningKey(
            key_id=f"key_{secrets.token_hex(8)}",
            secret=secrets.token_hex(32),
            created_at=now,
            expires_at=now + self.rotation_interval + self.grace_period,
        )
        self.keys.append(key)
        return key

    @property
    def current_key(self) -> SigningKey:
        """현재(가장 새로운) 폐기되지 않은 서명 키 가져오기."""
        active_keys = [k for k in self.keys if not k.revoked]
        if not active_keys:
            return self._generate_new_key()
        return active_keys[-1]  # 가장 최근에 생성됨

    @property
    def valid_keys(self) -> list[SigningKey]:
        """모든 유효한(폐기되지 않고 만료되지 않은) 키 가져오기."""
        now = datetime.now(timezone.utc)
        return [
            k for k in self.keys
            if not k.revoked and (k.expires_at is None or k.expires_at > now)
        ]

    def rotate(self) -> SigningKey:
        """새 서명 키로 교체."""
        # 새 키 생성
        new_key = self._generate_new_key()

        # 이전 키는 만료될 때까지 유효 유지(유예 기간)
        # 이를 통해 이전 키로 서명된 기존 토큰이 유효하게 유지됨

        # 만료되고 폐기된 키 정리
        now = datetime.now(timezone.utc)
        self.keys = [
            k for k in self.keys
            if not k.revoked and (k.expires_at is None or k.expires_at > now)
        ]

        return new_key

    def sign_token(self, payload: dict) -> str:
        """현재 키로 JWT 서명."""
        key = self.current_key
        headers = {"kid": key.key_id}
        return jwt.encode(payload, key.secret, algorithm="HS256",
                         headers=headers)

    def verify_token(self, token: str) -> dict:
        """모든 유효한 키를 시도하여 JWT 검증."""
        # 먼저 헤더에서 kid 가져오기 시도
        try:
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")
        except jwt.DecodeError:
            raise ValueError("Invalid token format")

        # kid가 있으면 특정 키 찾기
        if kid:
            for key in self.valid_keys:
                if key.key_id == kid:
                    return jwt.decode(token, key.secret,
                                     algorithms=["HS256"])
            raise ValueError(f"Unknown key ID: {kid}")

        # 폴백: 모든 유효한 키 시도(kid가 없는 토큰용)
        errors = []
        for key in self.valid_keys:
            try:
                return jwt.decode(token, key.secret,
                                 algorithms=["HS256"])
            except jwt.InvalidSignatureError:
                errors.append(f"Key {key.key_id}: signature mismatch")
                continue

        raise ValueError(f"Token verification failed with all keys")

    def should_rotate(self) -> bool:
        """현재 키를 교체해야 하는지 확인."""
        key = self.current_key
        age = datetime.now(timezone.utc) - key.created_at
        return age >= self.rotation_interval


# ── 사용법 ────────────────────────────────────────────────────────
manager = KeyRotationManager(
    rotation_interval_days=30,
    grace_period_days=7,
)

# 토큰 서명
token = manager.sign_token({"sub": "user_123", "role": "admin"})

# 나중에 토큰 검증 (교체 후에도 작동)
payload = manager.verify_token(token)

# 주기적 교체 (예약된 작업에서 호출)
if manager.should_rotate():
    new_key = manager.rotate()
    print(f"Rotated to new key: {new_key.key_id}")
```

### 3.3 데이터베이스 비밀번호 교체

```python
"""
데이터베이스 비밀번호 교체 전략.
"""
import psycopg2
import secrets
import logging

logger = logging.getLogger('secret_rotation')


class DatabasePasswordRotator:
    """다운타임 없이 데이터베이스 비밀번호 교체."""

    def __init__(self, admin_conn_string: str):
        self.admin_conn_string = admin_conn_string

    def rotate_password(self, username: str) -> str:
        """데이터베이스 사용자의 비밀번호 교체.

        전략:
        1. 새 비밀번호 생성
        2. 데이터베이스에서 비밀번호 업데이트
        3. 애플리케이션 구성 업데이트
        4. 새 비밀번호로 연결 확인
        5. 검증 실패 시 롤백
        """
        new_password = secrets.token_urlsafe(32)

        conn = psycopg2.connect(self.admin_conn_string)
        conn.autocommit = True

        try:
            with conn.cursor() as cur:
                # 단계 1: 비밀번호 변경
                # format()을 주의해서 사용 — 이것은 DDL 명령이므로
                # 매개변수화된 쿼리를 사용할 수 없음
                cur.execute(
                    f"ALTER USER {username} WITH PASSWORD %s",
                    (new_password,)
                )
                logger.info(f"Password rotated for user: {username}")

            # 단계 2: 새 비밀번호가 작동하는지 확인
            test_conn_string = (
                f"postgresql://{username}:{new_password}"
                f"@localhost:5432/mydb"
            )
            test_conn = psycopg2.connect(test_conn_string)
            test_conn.close()
            logger.info(f"New password verified for user: {username}")

            return new_password

        except Exception as e:
            logger.error(f"Password rotation failed: {e}")
            raise
        finally:
            conn.close()


# ── 스케줄러를 사용한 자동 교체 ───────────────────────────────
"""
# APScheduler 사용
from apscheduler.schedulers.blocking import BlockingScheduler

scheduler = BlockingScheduler()

@scheduler.scheduled_job('cron', day='1', hour='3')  # 매월 1일 오전 3시
def rotate_db_passwords():
    rotator = DatabasePasswordRotator(admin_conn_string=ADMIN_DB_URL)
    new_password = rotator.rotate_password('app_user')

    # secret 저장소 업데이트 (Vault, AWS Secrets Manager 등)
    update_secret_store('db_password', new_password)

    # 애플리케이션에 구성 재로드 알림
    notify_config_reload()

scheduler.start()
"""
```

---

## 4. HashiCorp Vault

### 4.1 Vault 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HashiCorp Vault 아키텍처                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                     ┌──────────────────┐                             │
│                     │   Vault Server   │                             │
│                     │                  │                             │
│  ┌──────────┐       │  ┌────────────┐  │       ┌──────────┐        │
│  │  Client   │──────│──│  Auth      │  │───────│  Backend │        │
│  │  (App)    │      │  │  Methods   │  │       │  Storage │        │
│  └──────────┘       │  ├────────────┤  │       │  (Consul,│        │
│                     │  │  Token     │  │       │   Raft,  │        │
│  ┌──────────┐       │  │  AppRole   │  │       │   File)  │        │
│  │  Client   │──────│──│  LDAP      │  │       └──────────┘        │
│  │  (CI/CD)  │      │  │  K8s       │  │                           │
│  └──────────┘       │  │  AWS IAM   │  │       ┌──────────┐        │
│                     │  └────────────┘  │       │  Audit   │        │
│  ┌──────────┐       │                  │───────│  Log     │        │
│  │  Client   │──────│──┌────────────┐  │       └──────────┘        │
│  │  (Admin)  │      │  │  Secret    │  │                           │
│  └──────────┘       │  │  Engines   │  │                           │
│                     │  ├────────────┤  │                           │
│                     │  │  KV v2     │  │  (정적 키-값)              │
│                     │  │  Database  │  │  (동적 자격 증명)          │
│                     │  │  PKI       │  │  (인증 기관)               │
│                     │  │  Transit   │  │  (서비스로서의 암호화)     │
│                     │  │  AWS       │  │  (동적 IAM 자격 증명)      │
│                     │  └────────────┘  │                           │
│                     └──────────────────┘                             │
│                                                                      │
│  주요 기능:                                                           │
│  • 동적 secrets: 필요 시 자격 증명 생성                                │
│  • 리스: secrets에 TTL 설정, 자동 만료                                 │
│  • 폐기: 즉시 모든 secret 또는 secret 트리 폐기                        │
│  • 서비스로서의 암호화: 키를 보지 않고 암호화/복호화                    │
│  • 감사 로깅: 모든 액세스가 기록됨                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Vault 빠른 시작

```bash
# ── Vault 설치 ────────────────────────────────────────────────
# macOS
brew install vault

# Linux
wget https://releases.hashicorp.com/vault/1.15.0/vault_1.15.0_linux_amd64.zip
unzip vault_1.15.0_linux_amd64.zip
sudo mv vault /usr/local/bin/

# ── 개발 서버 시작 (프로덕션용 아님) ───────────────────────────
vault server -dev
# Root 토큰이 출력됨 — 저장하세요
# export VAULT_ADDR='http://127.0.0.1:8200'
# export VAULT_TOKEN='hvs.xxxxxxxxxxxxx'

# ── KV secrets 엔진 활성화 ────────────────────────────────────
vault secrets enable -path=secret kv-v2

# ── secret 저장 ──────────────────────────────────────────────
vault kv put secret/myapp/database \
    username="dbuser" \
    password="supersecretpassword" \
    host="db.example.com" \
    port="5432"

# ── secret 읽기 ───────────────────────────────────────────────
vault kv get secret/myapp/database
vault kv get -format=json secret/myapp/database

# ── 특정 필드 읽기 ─────────────────────────────────────────────
vault kv get -field=password secret/myapp/database

# ── secrets 목록 ────────────────────────────────────────────────
vault kv list secret/myapp/

# ── secret 삭제 ─────────────────────────────────────────────────
vault kv delete secret/myapp/database

# ── 버전 히스토리 (KV v2) ────────────────────────────────────
vault kv get -version=1 secret/myapp/database
```

### 4.3 Python과 Vault (hvac)

```python
"""
Python용 HashiCorp Vault 클라이언트.
pip install hvac
"""
import hvac
import os
from typing import Optional


class VaultClient:
    """HashiCorp Vault 작업을 위한 래퍼."""

    def __init__(self, url: str = None, token: str = None):
        self.client = hvac.Client(
            url=url or os.getenv('VAULT_ADDR', 'http://127.0.0.1:8200'),
            token=token or os.getenv('VAULT_TOKEN'),
        )
        if not self.client.is_authenticated():
            raise RuntimeError("Vault authentication failed")

    # ── KV Secrets ───────────────────────────────────────────────
    def get_secret(self, path: str, mount_point: str = 'secret') -> dict:
        """KV v2에서 secret 읽기."""
        response = self.client.secrets.kv.v2.read_secret_version(
            path=path,
            mount_point=mount_point,
        )
        return response['data']['data']

    def set_secret(self, path: str, data: dict,
                   mount_point: str = 'secret') -> None:
        """KV v2에 secret 쓰기."""
        self.client.secrets.kv.v2.create_or_update_secret(
            path=path,
            secret=data,
            mount_point=mount_point,
        )

    def delete_secret(self, path: str,
                      mount_point: str = 'secret') -> None:
        """secret 삭제."""
        self.client.secrets.kv.v2.delete_metadata_and_all_versions(
            path=path,
            mount_point=mount_point,
        )

    # ── 동적 데이터베이스 자격 증명 ────────────────────────────────
    def get_database_creds(self, role: str) -> dict:
        """동적 데이터베이스 자격 증명 가져오기."""
        response = self.client.secrets.database.generate_credentials(
            name=role,
        )
        return {
            'username': response['data']['username'],
            'password': response['data']['password'],
            'lease_id': response['lease_id'],
            'lease_duration': response['lease_duration'],
        }

    def revoke_lease(self, lease_id: str) -> None:
        """동적 secret 리스 폐기."""
        self.client.sys.revoke_lease(lease_id)

    # ── Transit 암호화 ──────────────────────────────────────────
    def encrypt(self, key_name: str, plaintext: str) -> str:
        """Vault Transit 엔진을 사용하여 데이터 암호화."""
        import base64
        b64 = base64.b64encode(plaintext.encode()).decode()
        response = self.client.secrets.transit.encrypt_data(
            name=key_name,
            plaintext=b64,
        )
        return response['data']['ciphertext']

    def decrypt(self, key_name: str, ciphertext: str) -> str:
        """Vault Transit 엔진을 사용하여 데이터 복호화."""
        import base64
        response = self.client.secrets.transit.decrypt_data(
            name=key_name,
            ciphertext=ciphertext,
        )
        return base64.b64decode(response['data']['plaintext']).decode()


# ── 사용법 ────────────────────────────────────────────────────────
vault = VaultClient()

# secret 저장
vault.set_secret('myapp/database', {
    'username': 'dbuser',
    'password': 'supersecret',
    'host': 'db.example.com',
})

# secret 읽기
db_config = vault.get_secret('myapp/database')
print(f"Connecting to {db_config['host']} as {db_config['username']}")

# 동적 데이터베이스 자격 증명 가져오기 (자동 만료)
creds = vault.get_database_creds('readonly')
print(f"Temporary user: {creds['username']}")
print(f"Expires in: {creds['lease_duration']} seconds")

# 민감한 데이터 암호화
ciphertext = vault.encrypt('my-key', 'Social Security: 123-45-6789')
# ciphertext: vault:v1:8SDd3WHDOjf7mq69CyCqYjBXAiQQAVZRkFM13ok481zVCKqkLQ==

# 복호화
plaintext = vault.decrypt('my-key', ciphertext)
# plaintext: Social Security: 123-45-6789
```

### 4.4 Vault AppRole 인증

```python
"""
애플리케이션용(사람이 아닌) Vault AppRole 인증.
"""
import hvac
import os


def authenticate_with_approle(vault_addr: str, role_id: str,
                               secret_id: str) -> hvac.Client:
    """AppRole 방법을 사용하여 Vault에 인증.

    AppRole은 머신 간 인증을 위해 설계됨.
    role_id = 사용자 이름과 같음 (안정적, Vault에서 구성됨)
    secret_id = 비밀번호와 같음 (교체 가능, 단기간)
    """
    client = hvac.Client(url=vault_addr)

    # AppRole로 로그인
    response = client.auth.approle.login(
        role_id=role_id,
        secret_id=secret_id,
    )

    # 클라이언트가 이제 인증됨
    client.token = response['auth']['client_token']
    print(f"Authenticated. Token TTL: {response['auth']['lease_duration']}s")

    return client


# ── 프로덕션에서는 secret_id가 오케스트레이터에 의해 주입됨 ────
# Kubernetes: 파일로 마운트됨
# Docker: 환경 변수로 전달됨
# CI/CD: 파이프라인 secrets에 저장됨

vault = authenticate_with_approle(
    vault_addr=os.getenv('VAULT_ADDR'),
    role_id=os.getenv('VAULT_ROLE_ID'),
    secret_id=os.getenv('VAULT_SECRET_ID'),  # 단기간
)

# 이제 vault를 사용하여 애플리케이션 secrets 가져오기
secrets = vault.secrets.kv.v2.read_secret_version(path='myapp/config')
```

---

## 5. 클라우드 Secret 관리자

### 5.1 AWS Secrets Manager

```python
"""
AWS Secrets Manager 클라이언트.
pip install boto3
"""
import boto3
import json
from botocore.exceptions import ClientError


class AWSSecretsManager:
    """AWS Secrets Manager 인터페이스."""

    def __init__(self, region: str = 'us-east-1'):
        self.client = boto3.client(
            'secretsmanager',
            region_name=region,
        )

    def get_secret(self, secret_name: str) -> dict:
        """secret 값 가져오기."""
        try:
            response = self.client.get_secret_value(
                SecretId=secret_name,
            )
            # Secrets는 문자열 또는 바이너리일 수 있음
            if 'SecretString' in response:
                return json.loads(response['SecretString'])
            else:
                import base64
                return json.loads(
                    base64.b64decode(response['SecretBinary'])
                )
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                raise KeyError(f"Secret not found: {secret_name}")
            elif error_code == 'AccessDeniedException':
                raise PermissionError(f"Access denied to: {secret_name}")
            raise

    def create_secret(self, name: str, value: dict,
                      description: str = '') -> str:
        """새 secret 생성."""
        response = self.client.create_secret(
            Name=name,
            Description=description,
            SecretString=json.dumps(value),
        )
        return response['ARN']

    def update_secret(self, name: str, value: dict) -> None:
        """기존 secret 업데이트."""
        self.client.update_secret(
            SecretId=name,
            SecretString=json.dumps(value),
        )

    def rotate_secret(self, name: str, rotation_lambda_arn: str,
                      rotation_days: int = 30) -> None:
        """secret의 자동 교체 활성화."""
        self.client.rotate_secret(
            SecretId=name,
            RotationLambdaARN=rotation_lambda_arn,
            RotationRules={
                'AutomaticallyAfterDays': rotation_days,
            },
        )

    def delete_secret(self, name: str,
                      recovery_days: int = 30) -> None:
        """복구 기간이 있는 secret 삭제."""
        self.client.delete_secret(
            SecretId=name,
            RecoveryWindowInDays=recovery_days,
        )


# ── 사용법 ────────────────────────────────────────────────────────
sm = AWSSecretsManager(region='us-east-1')

# 데이터베이스 자격 증명 저장
sm.create_secret(
    name='prod/myapp/database',
    value={
        'engine': 'postgresql',
        'host': 'db.example.com',
        'port': 5432,
        'username': 'app_user',
        'password': 'strong_password_here',
        'dbname': 'production',
    },
    description='Production database credentials',
)

# 가져오고 사용
db_config = sm.get_secret('prod/myapp/database')
connection_string = (
    f"postgresql://{db_config['username']}:{db_config['password']}"
    f"@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
)
```

### 5.2 GCP Secret Manager

```python
"""
Google Cloud Secret Manager 클라이언트.
pip install google-cloud-secret-manager
"""
from google.cloud import secretmanager


class GCPSecretManager:
    """GCP Secret Manager 인터페이스."""

    def __init__(self, project_id: str):
        self.client = secretmanager.SecretManagerServiceClient()
        self.project_id = project_id

    def _secret_path(self, secret_id: str, version: str = 'latest') -> str:
        """전체 리소스 경로 구축."""
        return (
            f"projects/{self.project_id}/secrets/{secret_id}"
            f"/versions/{version}"
        )

    def get_secret(self, secret_id: str,
                   version: str = 'latest') -> str:
        """secret 값 가져오기."""
        name = self._secret_path(secret_id, version)
        response = self.client.access_secret_version(
            request={"name": name}
        )
        return response.payload.data.decode('UTF-8')

    def create_secret(self, secret_id: str, value: str) -> str:
        """초기 버전으로 새 secret 생성."""
        parent = f"projects/{self.project_id}"

        # secret 생성
        secret = self.client.create_secret(
            request={
                "parent": parent,
                "secret_id": secret_id,
                "secret": {
                    "replication": {
                        "automatic": {},
                    },
                },
            }
        )

        # 실제 값으로 버전 추가
        version = self.client.add_secret_version(
            request={
                "parent": secret.name,
                "payload": {
                    "data": value.encode('UTF-8'),
                },
            }
        )

        return version.name

    def add_version(self, secret_id: str, value: str) -> str:
        """기존 secret에 새 버전 추가."""
        parent = f"projects/{self.project_id}/secrets/{secret_id}"
        version = self.client.add_secret_version(
            request={
                "parent": parent,
                "payload": {
                    "data": value.encode('UTF-8'),
                },
            }
        )
        return version.name

    def disable_version(self, secret_id: str, version: str) -> None:
        """secret 버전 비활성화(소프트 삭제)."""
        name = self._secret_path(secret_id, version)
        self.client.disable_secret_version(
            request={"name": name}
        )

    def delete_secret(self, secret_id: str) -> None:
        """secret와 모든 버전 삭제."""
        name = f"projects/{self.project_id}/secrets/{secret_id}"
        self.client.delete_secret(request={"name": name})


# ── 사용법 ────────────────────────────────────────────────────────
sm = GCPSecretManager(project_id='my-project-123')

# secret 생성
sm.create_secret('database-password', 'my_secret_password')

# 최신 버전 읽기
password = sm.get_secret('database-password')

# 교체: 새 버전 추가
sm.add_version('database-password', 'new_rotated_password')

# 이전 버전은 버전 번호로 여전히 액세스 가능
old_password = sm.get_secret('database-password', version='1')
```

---

## 6. Git Secrets 스캐닝

### 6.1 문제점

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Git의 Secrets — 문제점                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  secret 유출의 일반적인 시나리오:                                      │
│                                                                      │
│  1. .env 파일의 우발적 커밋                                           │
│     $ git add .                                                      │
│     $ git commit -m "initial commit"                                 │
│     # 실제 비밀번호가 있는 .env가 이제 git 히스토리에 영원히 남음       │
│                                                                      │
│  2. 소스 코드에 하드코딩된 API 키                                      │
│     api_key = "sk_live_EXAMPLE_KEY_REPLACE_ME"                   │
│     # 나중에 삭제해도 git 히스토리에 존재함                            │
│                                                                      │
│  3. 자격 증명이 있는 구성 파일                                         │
│     database:                                                        │
│       password: "production_password_123"                            │
│                                                                      │
│  4. 실제 자격 증명이 있는 테스트 픽스처                                 │
│     STRIPE_KEY = "sk_live_..." # "test" 키가 실제로는 라이브          │
│                                                                      │
│  중요: git rm은 히스토리에서 제거하지 않음!                             │
│  secret은 다음을 통해 여전히 액세스 가능: git log --all --full-history │
│  제거하려면: git filter-branch 또는 BFG Repo-Cleaner가 필요            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Pre-Commit 훅

```bash
# ── git-secrets (AWS) ────────────────────────────────────────────
# 설치
brew install git-secrets  # macOS
# 또는: git clone https://github.com/awslabs/git-secrets.git && make install

# 저장소에 설정
cd /path/to/repo
git secrets --install        # 훅 설치
git secrets --register-aws   # AWS 패턴 등록

# 커스텀 패턴 추가
git secrets --add 'PRIVATE_KEY'
git secrets --add 'password\s*=\s*.+'
git secrets --add --allowed 'password\s*=\s*os\.getenv'  # 환경 조회 허용

# 스캐닝 테스트
git secrets --scan           # 스테이징된 변경사항 스캔
git secrets --scan-history   # 전체 히스토리 스캔

# ── pre-commit 프레임워크 ────────────────────────────────────────
# pip install pre-commit
```

```yaml
# .pre-commit-config.yaml
repos:
  # 커밋 전에 secrets 감지
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']

  # gitleaks — 포괄적인 secret 스캐너
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks

  # 개인 키 확인
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: detect-private-key
      - id: check-added-large-files
        args: ['--maxkb=100']
```

```bash
# pre-commit 훅 설치
pre-commit install

# 모든 파일에 수동 실행
pre-commit run --all-files

# 특정 훅 실행
pre-commit run detect-secrets --all-files
```

### 6.3 스캐닝 도구

```bash
# ── gitleaks — 빠르고 포괄적인 스캐너 ──────────────────────────
# 설치
brew install gitleaks  # macOS
# 또는: go install github.com/gitleaks/gitleaks/v8@latest

# 현재 저장소 스캔
gitleaks detect --source . --verbose

# 특정 커밋 범위 스캔
gitleaks detect --source . --log-opts="HEAD~10..HEAD"

# 전체 히스토리 스캔
gitleaks detect --source . --log-opts="--all"

# 보고서 생성
gitleaks detect --source . --report-format json --report-path report.json

# 커스텀 규칙 사용
gitleaks detect --source . --config gitleaks.toml

# ── trufflehog — 높은 정확도 스캐너 ──────────────────────────
# 설치
pip install trufflehog

# 또는 Docker 사용
docker run --rm -v "$PWD:/repo" trufflesecurity/trufflehog:latest \
    git file:///repo --only-verified

# 저장소 스캔
trufflehog git file:///path/to/repo --only-verified

# GitHub 저장소 직접 스캔
trufflehog github --org your-org --only-verified

# ── detect-secrets (Yelp) ───────────────────────────────────────
# 설치
pip install detect-secrets

# 베이스라인 생성 (기존 결과를 허용됨으로 표시)
detect-secrets scan > .secrets.baseline

# 베이스라인 대화식으로 감사
detect-secrets audit .secrets.baseline

# 새 secrets 스캔 (베이스라인과 비교)
detect-secrets scan --baseline .secrets.baseline
```

### 6.4 커스텀 gitleaks 구성

```toml
# gitleaks.toml — secret 감지를 위한 커스텀 규칙
title = "Custom Gitleaks Config"

# 기본 규칙 확장
[extend]
useDefault = true

# 커스텀 규칙
[[rules]]
id = "custom-api-key"
description = "Custom API Key Pattern"
regex = '''(?i)(api[_-]?key|apikey)\s*[=:]\s*['"]?([a-zA-Z0-9_\-]{20,})['"]?'''
secretGroup = 2

[[rules]]
id = "slack-webhook"
description = "Slack Webhook URL"
regex = '''https://hooks\.slack\.com/services/T[a-zA-Z0-9_]{8,}/B[a-zA-Z0-9_]{8,}/[a-zA-Z0-9_]{24,}'''

[[rules]]
id = "internal-password"
description = "Hardcoded Password"
regex = '''(?i)(password|passwd|pwd)\s*[=:]\s*['"]([^'"]{8,})['"]'''
secretGroup = 2

# 허용 목록 (오탐지 억제)
[allowlist]
paths = [
    '''\.env\.example''',
    '''\.env\.template''',
    '''test_.*\.py''',        # 주의해서 사용
    '''docs/.*\.md''',
]
regexes = [
    '''password\s*=\s*os\.getenv''',    # 환경에서 읽는 것은 허용
    '''password\s*=\s*["']changeme["']''',  # 플레이스홀더 값
    '''EXAMPLE_KEY_DO_NOT_USE''',
]
```

### 6.5 Git 히스토리에서 Secrets 제거

```bash
# ── BFG Repo-Cleaner (권장) ─────────────────────────────────────
# 설치: brew install bfg

# 모든 히스토리에서 파일 제거
bfg --delete-files '.env' my-repo.git

# 모든 파일에서 텍스트 교체
echo "sk_live_EXAMPLE_KEY_REPLACE_ME" > passwords.txt
bfg --replace-text passwords.txt my-repo.git

# BFG 후 정리
cd my-repo.git
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 강제 푸시 (경고: 히스토리 재작성)
git push --force

# ── git filter-repo (대안) ──────────────────────────────────────
# pip install git-filter-repo

# 히스토리에서 파일 제거
git filter-repo --path .env --invert-paths

# 모든 파일에서 문자열 교체
git filter-repo --replace-text <(echo 'literal:sk_live_EXAMPLE_KEY_REPLACE_ME==>REDACTED')

# ── 중요 ───────────────────────────────────────────────────────
# 히스토리에서 secrets 제거 후:
# 1. 원격에 강제 푸시 (모든 협업자는 재복제해야 함)
# 2. 즉시 손상된 secret 교체
# 3. secret은 이미 노출됨 — git에서 제거는 피해 통제임
# 4. GitHub 캐시는 일시적으로 여전히 데이터를 가질 수 있음
# 5. 모든 포크는 여전히 secret을 포함할 수 있음
```

---

## 7. CI/CD Secrets

### 7.1 GitHub Actions Secrets

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

# ── GitHub Actions에서 secrets 액세스 ──────────────────────────
jobs:
  deploy:
    runs-on: ubuntu-latest

    # 다른 단계를 위한 환경 수준 secrets 사용
    environment: production

    steps:
      - uses: actions/checkout@v4

      # Secrets는 환경 변수로 사용 가능
      - name: Configure AWS
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_REGION: us-east-1
        run: |
          aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
          aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
          aws configure set region $AWS_REGION

      # secrets를 직접 사용 (로그에서 마스킹됨)
      - name: Deploy
        run: |
          echo "Deploying to production..."
          # ${{ secrets.DEPLOY_KEY }}는 로그에서 ***로 마스킹됨
          ./deploy.sh --key "${{ secrets.DEPLOY_KEY }}"

      # secrets로 Docker 로그인
      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      # OIDC 인증 (정적 자격 증명보다 선호됨)
      - name: Configure AWS with OIDC
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789:role/github-actions
          aws-region: us-east-1
          # 정적 자격 증명 불필요!
```

```yaml
# ── GitHub Actions 보안 모범 사례 ──────────────────────────────
# 1. 필수 검토자가 있는 환경 수준 secrets 사용
# 2. 가능한 경우 정적 자격 증명 대신 OIDC 사용
# 3. actions를 태그가 아닌 특정 SHA에 고정
# 4. secret 액세스를 특정 환경으로 제한
# 5. PAT가 아닌 GITHUB_TOKEN의 자동 권한 사용

# .github/workflows/secure.yml
name: Secure Workflow

on:
  push:
    branches: [main]

permissions:
  contents: read  # 최소 권한

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      # actions를 SHA에 고정 (태그 아님) — 공급망 공격 방지
      - uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11  # v4.1.1

      # 저장소 액세스를 위해 GITHUB_TOKEN 사용 (자동 범위 지정)
      - name: Create Release
        env:
          GH_TOKEN: ${{ github.token }}
        run: gh release create v1.0 --generate-notes

      # secrets를 절대 echo하지 말 것 (실수로라도)
      - name: Safe Logging
        env:
          API_KEY: ${{ secrets.API_KEY }}
        run: |
          # 나쁨: echo "Key is: $API_KEY"
          # 좋음: 출력하지 않고 키가 작동하는지 테스트
          curl -sf -H "Authorization: Bearer $API_KEY" \
               https://api.example.com/health || exit 1
```

### 7.2 GitLab CI 변수

```yaml
# .gitlab-ci.yml
# 변수는 GitLab UI에서 설정됨:
# Settings → CI/CD → Variables

stages:
  - test
  - deploy

test:
  stage: test
  variables:
    # 민감하지 않은 것은 여기에 설정 가능
    NODE_ENV: test
  script:
    # Protected 변수는 protected 브랜치에서만 사용 가능
    - echo "Running tests..."
    - pytest --cov
    # Masked 변수는 작업 로그에서 숨겨짐
    - echo "DB URL is $DATABASE_URL"  # [MASKED]로 표시됨

deploy:
  stage: deploy
  # protected 브랜치로만 제한
  only:
    - main
  variables:
    # 여러 줄 secrets(인증서 같은)를 위한 파일 타입 변수 사용
    # GitLab은 값을 파일에 쓰고 변수를 경로로 설정
    KUBE_CONFIG: $KUBE_CONFIG_FILE  # 이것은 파일 경로임
  script:
    - kubectl --kubeconfig="$KUBE_CONFIG" apply -f deployment.yaml
  environment:
    name: production

# ── GitLab CI Secret 모범 사례 ─────────────────────────────────
# 1. 민감한 변수를 "Masked"로 표시 (로그에서 숨김)
# 2. 배포 secrets를 "Protected"로 표시 (protected 브랜치에서만)
# 3. 여러 줄 secrets(인증서, 키)를 위해 "File" 타입 사용
# 4. 변수를 특정 환경으로 범위 지정
# 5. GitLab의 외부 secrets 통합 사용 (Vault, AWS, GCP)
```

---

## 8. 저장 시 암호화

### 8.1 구성 파일 암호화

```python
"""
Fernet(대칭 암호화)를 사용하여 구성 파일 암호화.
pip install cryptography
"""
from cryptography.fernet import Fernet
import json
import os
from pathlib import Path


class ConfigEncryptor:
    """구성 파일 암호화 및 복호화."""

    def __init__(self, key: bytes = None):
        if key is None:
            # 환경에서 키 로드
            key_str = os.getenv('CONFIG_ENCRYPTION_KEY')
            if key_str is None:
                raise ValueError(
                    "CONFIG_ENCRYPTION_KEY environment variable not set"
                )
            key = key_str.encode()
        self.cipher = Fernet(key)

    @staticmethod
    def generate_key() -> str:
        """새 암호화 키 생성."""
        return Fernet.generate_key().decode()

    def encrypt_config(self, config: dict, output_path: str) -> None:
        """구성 딕셔너리를 암호화하고 파일에 쓰기."""
        json_data = json.dumps(config, indent=2).encode()
        encrypted = self.cipher.encrypt(json_data)

        with open(output_path, 'wb') as f:
            f.write(encrypted)

    def decrypt_config(self, input_path: str) -> dict:
        """구성 파일 읽고 복호화."""
        with open(input_path, 'rb') as f:
            encrypted = f.read()

        decrypted = self.cipher.decrypt(encrypted)
        return json.loads(decrypted)

    def encrypt_value(self, value: str) -> str:
        """단일 값 암호화."""
        return self.cipher.encrypt(value.encode()).decode()

    def decrypt_value(self, encrypted_value: str) -> str:
        """단일 값 복호화."""
        return self.cipher.decrypt(encrypted_value.encode()).decode()


# ── 사용법 ────────────────────────────────────────────────────────
# 처음: 키 생성
key = ConfigEncryptor.generate_key()
print(f"Store this key securely: {key}")
# 저장 위치: 환경 변수, Vault, 클라우드 KMS

# 구성 암호화
encryptor = ConfigEncryptor(key.encode())
encryptor.encrypt_config(
    config={
        'database_url': 'postgresql://user:pass@host/db',
        'api_key': 'sk_live_xxxxx',
        'jwt_secret': 'supersecret',
    },
    output_path='config.enc',
)

# 구성 복호화 (애플리케이션 시작 시)
config = encryptor.decrypt_config('config.enc')
print(config['database_url'])
```

### 8.2 SOPS (Secrets OPerationS)

```bash
# ── SOPS: 암호화된 파일 관리 ─────────────────────────────────────
# SOPS는 키를 보이게 유지하면서 YAML/JSON/ENV 파일의 값을 암호화
# 이는 diff, 검토, 암호화된 파일 커밋이 가능함을 의미

# 설치
brew install sops

# ── age 암호화와 SOPS 사용 ──────────────────────────────────────
# age 키 생성
age-keygen -o keys.txt
# 공개 키: age1xxxxxxx...
# 개인 키를 안전하게 저장

# .sops.yaml 구성 생성
cat > .sops.yaml << 'EOF'
creation_rules:
  - path_regex: \.enc\.yaml$
    age: age1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  - path_regex: \.enc\.json$
    age: age1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
EOF

# 파일 암호화
sops --encrypt secrets.yaml > secrets.enc.yaml

# 파일 복호화
sops --decrypt secrets.enc.yaml > secrets.yaml

# 제자리에서 암호화된 파일 편집 (복호화, 에디터 열기, 재암호화)
sops secrets.enc.yaml
```

```yaml
# secrets.enc.yaml — 값은 암호화되고, 키는 읽을 수 있음
database:
    host: ENC[AES256_GCM,data:kDf5...,iv:abc...,tag:def...,type:str]
    port: ENC[AES256_GCM,data:NTQzMg==,iv:ghi...,tag:jkl...,type:int]
    username: ENC[AES256_GCM,data:mno...,iv:pqr...,tag:stu...,type:str]
    password: ENC[AES256_GCM,data:vwx...,iv:yza...,tag:bcd...,type:str]
api_keys:
    stripe: ENC[AES256_GCM,data:efg...,iv:hij...,tag:klm...,type:str]
    sendgrid: ENC[AES256_GCM,data:nop...,iv:qrs...,tag:tuv...,type:str]
sops:
    kms: []
    age:
        - recipient: age1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
          enc: |
            -----BEGIN AGE ENCRYPTED FILE-----
            ...
            -----END AGE ENCRYPTED FILE-----
    lastmodified: "2025-01-15T10:30:00Z"
    version: 3.8.0
```

```python
"""
Python에서 SOPS로 암호화된 구성 로드.
"""
import subprocess
import json
import yaml


def load_sops_config(path: str) -> dict:
    """SOPS 파일을 복호화하고 딕셔너리로 반환."""
    result = subprocess.run(
        ['sops', '--decrypt', path],
        capture_output=True,
        text=True,
        check=True,
    )

    if path.endswith('.json'):
        return json.loads(result.stdout)
    elif path.endswith('.yaml') or path.endswith('.yml'):
        return yaml.safe_load(result.stdout)
    else:
        raise ValueError(f"Unsupported file format: {path}")


# 사용법
config = load_sops_config('secrets.enc.yaml')
db_password = config['database']['password']
```

---

## 9. 일반적인 실수

### 9.1 안티패턴

```python
"""
일반적인 secrets 관리 실수 — 이것들을 하지 마세요.
"""

# ── 실수 1: 하드코딩된 secrets ────────────────────────────────
# 나쁨
API_KEY = "sk_live_EXAMPLE_KEY_REPLACE_ME"
DB_PASSWORD = "production_password_123"

# 좋음
import os
API_KEY = os.getenv("API_KEY")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# ── 실수 2: 로그의 Secrets ──────────────────────────────────────
# 나쁨
import logging
logger = logging.getLogger(__name__)
logger.info(f"Connecting to database with password: {password}")
logger.debug(f"API response: {response.headers}")  # 인증 헤더를 포함할 수 있음

# 좋음
logger.info("Connecting to database as user: %s", username)
logger.debug("API response status: %s", response.status_code)


# ── 실수 3: URL의 Secrets ──────────────────────────────────────
# 나쁨 — 비밀번호가 서버 로그, 브라우저 히스토리, Referer 헤더에 나타남
DATABASE_URL = "postgresql://admin:P@ssw0rd@db.example.com/prod"

# 좋음 — 별도의 변수 사용
DATABASE_HOST = os.getenv("DB_HOST")
DATABASE_USER = os.getenv("DB_USER")
DATABASE_PASS = os.getenv("DB_PASS")


# ── 실수 4: 에러 메시지의 Secrets ────────────────────────────────
# 나쁨
try:
    conn = connect(password=secret_password)
except ConnectionError as e:
    raise RuntimeError(f"Failed to connect with password {secret_password}: {e}")

# 좋음
try:
    conn = connect(password=secret_password)
except ConnectionError as e:
    raise RuntimeError(f"Database connection failed: {e}")


# ── 실수 5: .env를 git에 커밋 ───────────────────────────────────
# 나쁨: .gitignore에서 .env 잊음
# 더 나쁨: .env를 추가한 후 제거 (히스토리에 여전히 남음)

# 좋음: 파일 생성 전에 .gitignore에 추가
# 그리고 우발적 커밋 방지를 위해 pre-commit 훅 사용


# ── 실수 6: 모든 환경에서 동일한 secret 사용 ─────────────────────
# 나쁨: 개발 환경에서 프로덕션 키 사용
# 이는 개발 머신 손상의 폭발 반경을 증가시킴

# 좋음: dev, staging, prod를 위한 다른 secrets
# 개발은 테스트/샌드박스 API 키 사용
# 프로덕션 키는 프로덕션 서버에만 존재


# ── 실수 7: secrets를 절대 교체하지 않음 ───────────────────────
# 나쁨: 2019년에 생성된 API 키, 절대 변경하지 않음
# 손상되면 공격자가 수년간 액세스 권한을 가졌음

# 좋음: 다운타임 없는 전략으로 자동 교체


# ── 실수 8: 팀원 간 secrets 공유 ────────────────────────────────
# 나쁨: "프로덕션 DB 비밀번호를 Slack으로 보내줄 수 있나요?"

# 좋음:
# - secret 관리자 사용 (Vault, AWS SM, 1Password)
# - 각 사람이 자신의 자격 증명을 가짐
# - 가능한 곳에서 SSO/OIDC 사용 (공유 secrets 없음)
# - Secrets는 사람이 아닌 프로그래밍 방식으로 액세스됨


# ── 실수 9: Docker 이미지의 Secrets ─────────────────────────────
# 나쁨
# Dockerfile:
# ENV API_KEY=sk_live_xxxxx
# COPY .env /app/.env

# 좋음: 런타임에 주입
# docker run -e API_KEY=$API_KEY myimage
# 또는 Docker secrets / Kubernetes secrets 사용


# ── 실수 10: 클라이언트 측 secrets ─────────────────────────────
# 나쁨: JavaScript/모바일 앱에 API 키 임베딩
# <script>
#   const API_KEY = "sk_live_xxxxx";  // 누구나 볼 수 있음
# </script>

# 좋음: 백엔드 프록시 사용
# 클라이언트 → 당신의 백엔드 (secret 보유) → 서드파티 API
```

### 9.2 Secret 감지 체크리스트

```python
"""
코드에서 일반적인 secret 패턴에 대한 자동 확인.
"""
import re
from pathlib import Path
from dataclasses import dataclass

@dataclass
class SecretFinding:
    file: str
    line_number: int
    pattern_name: str
    matched_text: str

# 잠재적 secrets를 나타내는 패턴
SECRET_PATTERNS = {
    "aws_access_key": re.compile(
        r'(?:A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)'
        r'[A-Z0-9]{16}'
    ),
    "aws_secret_key": re.compile(
        r'(?i)aws_secret_access_key\s*[=:]\s*["\']?[A-Za-z0-9/+=]{40}'
    ),
    "generic_password": re.compile(
        r'(?i)(password|passwd|pwd)\s*[=:]\s*["\'][^"\']{8,}["\']'
    ),
    "generic_api_key": re.compile(
        r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\'][A-Za-z0-9_\-]{20,}["\']'
    ),
    "private_key": re.compile(
        r'-----BEGIN (?:RSA|EC|DSA|OPENSSH)? ?PRIVATE KEY-----'
    ),
    "jwt_token": re.compile(
        r'eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+'
    ),
    "github_token": re.compile(
        r'gh[pousr]_[A-Za-z0-9_]{36,}'
    ),
    "slack_webhook": re.compile(
        r'https://hooks\.slack\.com/services/T[a-zA-Z0-9_]+/B[a-zA-Z0-9_]+/[a-zA-Z0-9_]+'
    ),
    "stripe_key": re.compile(
        r'[sr]k_(live|test)_[A-Za-z0-9]{20,}'
    ),
}

# 오탐지를 나타내는 패턴
FALSE_POSITIVE_PATTERNS = [
    re.compile(r'(?i)password\s*=\s*os\.getenv'),
    re.compile(r'(?i)password\s*=\s*["\']changeme["\']'),
    re.compile(r'(?i)password\s*=\s*["\']<.*>["\']'),
    re.compile(r'(?i)password\s*=\s*["\']your_password_here["\']'),
    re.compile(r'(?i)password\s*=\s*["\']placeholder["\']'),
    re.compile(r'#.*password'),  # 주석
]

SKIP_DIRS = {'.git', 'node_modules', '__pycache__', '.venv', 'venv'}
SKIP_EXTENSIONS = {'.pyc', '.pyo', '.so', '.dylib', '.png', '.jpg', '.gif'}


def scan_directory(root: str) -> list[SecretFinding]:
    """디렉토리에서 잠재적 secrets 스캔."""
    findings = []

    for path in Path(root).rglob('*'):
        # 디렉토리 건너뛰기
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix in SKIP_EXTENSIONS:
            continue

        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
        except (PermissionError, UnicodeDecodeError):
            continue

        for line_num, line in enumerate(content.splitlines(), 1):
            for pattern_name, pattern in SECRET_PATTERNS.items():
                if pattern.search(line):
                    # 오탐지 확인
                    is_false_positive = any(
                        fp.search(line) for fp in FALSE_POSITIVE_PATTERNS
                    )
                    if not is_false_positive:
                        findings.append(SecretFinding(
                            file=str(path),
                            line_number=line_num,
                            pattern_name=pattern_name,
                            matched_text=line.strip()[:100],
                        ))

    return findings


if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    findings = scan_directory(root)

    if findings:
        print(f"\nFound {len(findings)} potential secrets:\n")
        for f in findings:
            print(f"  [{f.pattern_name}] {f.file}:{f.line_number}")
            print(f"    {f.matched_text}")
            print()
        sys.exit(1)
    else:
        print("No secrets detected.")
        sys.exit(0)
```

---

## 10. 연습 문제

### 연습 문제 1: 환경 구성 시스템

다음을 수행하는 구성 관리 시스템 구축:

1. 우선순위 순서로 여러 소스에서 설정 로드:
   - 환경 변수(최우선)
   - `.env.local` 파일
   - `.env` 파일
   - 기본값(최하위)
2. 시작 시 모든 필수 설정 검증(빠른 실패)
3. Pydantic `SecretStr`을 사용하여 우발적 secrets 로깅 방지
4. secrets를 마스킹하여 모든 설정을 보여주는 `config.dump_safe()` 메서드 제공
5. 타입 강제 지원(문자열을 int, bool, list로)
6. 우선순위 순서와 검증을 확인하는 테스트 작성

### 연습 문제 2: Secret 교체 서비스

다음을 수행하는 자동 secret 교체 서비스 구현:

1. 구성 가능한 교체 간격으로 secrets 세트 관리
2. 다운타임 없는 교체 지원(이중 secret 기간)
3. secrets가 교체될 때 등록된 소비자에게 알림
4. 감사를 위한 모든 교체 이벤트 기록
5. 새 secret 검증 실패 시 롤백 지원
6. 스케줄러와 통합 가능(매시간 교체 확인 실행)
7. 규정 준수를 위한 교체 히스토리 저장

### 연습 문제 3: Git Secret 스캐너

다음을 수행하는 포괄적인 git secret 스캐너 구축:

1. 스테이징된 파일 스캔(pre-commit 훅)
2. 과거 유출을 위한 전체 git 히스토리 스캔
3. 구성 가능한 패턴 지원(정규식 기반)
4. 알려진 오탐지를 위한 베이스라인/허용 목록
5. JSON 및 사람이 읽을 수 있는 형식으로 보고서 생성
6. pre-commit 훅 또는 CI/CD 단계로 실행 가능
7. 증분 스캐닝 지원(새 커밋만)

### 연습 문제 4: Vault 통합 라이브러리

다음을 수행하는 Python 라이브러리 생성:

1. AppRole 또는 Token을 사용하여 Vault에 인증
2. TTL로 secrets를 로컬에 캐싱(반복되는 Vault 호출 방지)
3. 만료 전에 secrets 자동 갱신
4. Vault를 사용할 수 없을 때 환경 변수로 폴백
5. Django/Flask 설정 통합 제공
6. Vault 토큰 갱신을 투명하게 처리
7. secret 값을 드러내지 않고 액세스 패턴 기록

### 연습 문제 5: SOPS 워크플로 자동화

다음을 수행하는 명령줄 도구 구축:

1. 여러 환경(dev, staging, prod)을 위한 암호화된 구성 파일 생성
2. 암호화된 파일에 필요한 모든 키가 포함되어 있는지 검증
3. 복호화하지 않고 두 암호화된 파일 diff(구조 비교)
4. 암호화 키 교체(새 키로 모든 파일 재암호화)
5. 암호화되지 않은 파일의 커밋을 방지하기 위해 git 훅과 통합
6. 로컬 개발을 위해 암호화된 구성에서 `.env` 파일 생성

### 연습 문제 6: CI/CD Secrets 감사

다음을 수행하는 감사 도구 작성:

1. secret 사용 패턴을 위한 GitHub Actions 워크플로 파일 스캔
2. 안전하지 않게 전달된 secrets 식별(예: 로그에 보이는 명령 인자)
3. actions가 SHA에 고정되어 있는지 확인(변경 가능한 태그 아님)
4. secrets가 올바른 환경으로 범위 지정되었는지 확인
5. `GITHUB_TOKEN` 권한이 과도하게 넓은지 감지
6. 규정 준수 보고서 생성

---

## 요약

### Secrets 관리 성숙도 모델

| 레벨 | 설명 | 실천사항 |
|------|------|----------|
| 레벨 0 | 관리 없음 | 하드코딩된 secrets, git에 커밋됨 |
| 레벨 1 | 기본 | .env 파일, .gitignore, 수동 교체 |
| 레벨 2 | 중급 | Secret 관리자(Vault/클라우드), pre-commit 훅 |
| 레벨 3 | 고급 | 자동 교체, 동적 secrets, 감사 로깅 |
| 레벨 4 | 최적 | 제로 트러스트, 모든 곳에 OIDC, 연속 스캐닝 |

### 핵심 요점

1. **절대 secrets를 하드코딩하지 말 것** — 환경 변수 또는 secret 관리자 사용
2. **지속적으로 스캔** — pre-commit 훅과 CI/CD 스캐닝 사용
3. **정기적으로 교체** — 다운타임 없는 패턴으로 교체 자동화
4. **저장 시 암호화** — SOPS, Vault Transit 또는 클라우드 KMS 사용
5. **액세스 감사** — 누가 언제 어떤 secret에 액세스했는지 기록
6. **노출 최소화** — 단기간 자격 증명과 최소 권한 사용
7. **손상 계획** — secret 유출 시(언제가 아니라)를 위한 런북 준비

---

**이전**: [10_API_Security.md](./10_API_Security.md) | **다음**: [12_Container_Security.md](./12_Container_Security.md)
