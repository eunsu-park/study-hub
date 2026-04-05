# Secrets Management and Environment Configuration

**Previous**: [10_API_Security.md](./10_API_Security.md) | **Next**: [12_Container_Security.md](./12_Container_Security.md)

---

Secrets — API keys, database passwords, encryption keys, OAuth client secrets — are the crown jewels of any application. A single leaked secret can compromise an entire system. Despite this, secrets management remains one of the most commonly mishandled areas in software development. This lesson covers the full lifecycle of secrets: how to store them, rotate them, inject them at runtime, scan for accidental leaks, and manage them across CI/CD pipelines and cloud environments.

## Learning Objectives

- Understand the 12-factor app approach to configuration and secrets
- Use environment variables and .env files safely with python-dotenv
- Implement secret rotation strategies without downtime
- Configure HashiCorp Vault for centralized secrets management
- Use cloud-native secret stores (AWS Secrets Manager, GCP Secret Manager)
- Detect leaked secrets in git history using scanning tools
- Manage secrets in CI/CD pipelines (GitHub Actions, GitLab CI)
- Encrypt configuration at rest
- Avoid common mistakes that lead to secret exposure

---

## 1. Secrets Fundamentals

### 1.1 What Counts as a Secret?

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Types of Secrets                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Credentials                                                │    │
│  │  ├── Database passwords (PostgreSQL, MySQL, Redis)          │    │
│  │  ├── Service account passwords                              │    │
│  │  ├── SMTP/email credentials                                 │    │
│  │  └── SSH passwords                                          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  API Keys and Tokens                                        │    │
│  │  ├── Third-party API keys (Stripe, Twilio, AWS)             │    │
│  │  ├── OAuth client secrets                                   │    │
│  │  ├── JWT signing keys                                       │    │
│  │  └── Personal access tokens                                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Cryptographic Material                                     │    │
│  │  ├── TLS private keys                                       │    │
│  │  ├── Encryption keys (AES, RSA private keys)                │    │
│  │  ├── SSH private keys                                       │    │
│  │  └── Code signing keys                                      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Infrastructure Secrets                                     │    │
│  │  ├── Cloud provider credentials (AWS access keys)           │    │
│  │  ├── Container registry credentials                         │    │
│  │  ├── Kubernetes secrets                                     │    │
│  │  └── Terraform state encryption keys                        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Rule of thumb: If exposing it would cause harm, it is a secret.     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 The Secret Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Secret Lifecycle                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Generation ──▶ 2. Storage ──▶ 3. Distribution ──▶ 4. Usage      │
│       │                │               │                  │          │
│       │                │               │                  │          │
│       ▼                ▼               ▼                  ▼          │
│  Strong random     Encrypted       Secure channel    Minimal        │
│  generation        at rest         (TLS, IAM)       exposure       │
│                                                                      │
│  5. Rotation ──▶ 6. Revocation ──▶ 7. Audit                        │
│       │                │                │                            │
│       ▼                ▼                ▼                            │
│  Automated         Immediate        Log access                      │
│  periodic          on compromise    and changes                     │
│                                                                      │
│  Key principles:                                                     │
│  • Least privilege: only give access to who needs it                │
│  • Short-lived: prefer temporary credentials over permanent ones    │
│  • Encrypted at rest: never store secrets in plaintext              │
│  • Auditable: log who accessed what secret and when                 │
│  • Rotatable: design systems to handle secret rotation              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Environment Variables and .env Files

### 2.1 The 12-Factor App Approach

```
┌─────────────────────────────────────────────────────────────────────┐
│                    12-Factor App: Config                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Factor III: Store config in the environment                         │
│                                                                      │
│  ┌─────────────────┐                                                │
│  │  Application     │                                               │
│  │  Code            │ ← Same code deploys everywhere                │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │  Development     │  │  Staging         │  │  Production      │    │
│  │  DB=localhost    │  │  DB=staging.db   │  │  DB=prod.db      │    │
│  │  DEBUG=true      │  │  DEBUG=false     │  │  DEBUG=false     │    │
│  │  KEY=dev_key     │  │  KEY=stage_key   │  │  KEY=prod_key    │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
│                                                                      │
│  Config that varies between deploys:                                 │
│  ✓ Database URLs, API keys, feature flags                           │
│                                                                      │
│  Config that does NOT vary:                                          │
│  ✗ Framework settings, logging format, routes                       │
│  (These belong in code/config files, not environment)                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Using python-dotenv

```python
"""
Loading configuration from .env files with python-dotenv.
pip install python-dotenv
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Basic usage ──────────────────────────────────────────────────
# Load .env file from the current directory
load_dotenv()

# Access environment variables
database_url = os.getenv('DATABASE_URL')
secret_key = os.getenv('SECRET_KEY')
debug = os.getenv('DEBUG', 'false').lower() == 'true'

# ── Load from specific path ─────────────────────────────────────
env_path = Path(__file__).parent / '.env.production'
load_dotenv(dotenv_path=env_path)

# ── Override existing environment variables ──────────────────────
# By default, dotenv does NOT override existing env vars
# This is safe: system env vars take precedence
load_dotenv(override=False)  # Default behavior

# To force override (rarely needed):
load_dotenv(override=True)


# ── Structured configuration class ──────────────────────────────
from dataclasses import dataclass

@dataclass
class AppConfig:
    """Application configuration loaded from environment."""

    # Database
    database_url: str
    database_pool_size: int

    # Security
    secret_key: str
    jwt_secret: str
    jwt_expiry_minutes: int

    # External services
    stripe_api_key: str
    sendgrid_api_key: str

    # Application
    debug: bool
    log_level: str

    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Load configuration from environment variables."""
        load_dotenv()

        def require_env(key: str) -> str:
            """Get a required environment variable or raise error."""
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


# ── Usage ────────────────────────────────────────────────────────
config = AppConfig.from_env()
print(f"Debug mode: {config.debug}")
print(f"Database: {config.database_url[:20]}...")  # Don't log full URL
```

### 2.3 The .env File

```bash
# ── .env file (NEVER commit this to git) ─────────────────────────

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
DATABASE_POOL_SIZE=5

# Security
SECRET_KEY=your-256-bit-secret-key-here-change-me
JWT_SECRET=another-different-secret-key-for-jwt

# External APIs
STRIPE_API_KEY=sk_test_EXAMPLE_KEY_REPLACE_ME
SENDGRID_API_KEY=SG.xxxxxxxxxxxxx

# Application
DEBUG=false
LOG_LEVEL=INFO
```

```bash
# ── .env.example file (DO commit this to git) ───────────────────
# Copy this file to .env and fill in the values
# cp .env.example .env

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
DATABASE_POOL_SIZE=5

# Security (generate with: python -c "import secrets; print(secrets.token_hex(32))")
SECRET_KEY=change-me-generate-a-real-secret
JWT_SECRET=change-me-use-a-different-secret

# External APIs
STRIPE_API_KEY=sk_test_your_test_key_here
SENDGRID_API_KEY=SG.your_api_key_here

# Application
DEBUG=true
LOG_LEVEL=DEBUG
```

### 2.4 .gitignore Configuration

```gitignore
# ── Secrets and environment files ────────────────────────────────
.env
.env.local
.env.production
.env.staging
.env.*.local

# Keep the example file
!.env.example
!.env.template

# Private keys
*.pem
*.key
*.p12
*.pfx

# Cloud credentials
credentials.json
service-account*.json
.gcloud/
.aws/credentials

# IDE secrets
.idea/dataSources/
.vscode/settings.json
```

### 2.5 Pydantic Settings (Type-Safe Configuration)

```python
"""
Type-safe configuration with Pydantic Settings.
pip install pydantic-settings
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr, PostgresDsn
from typing import Optional


class Settings(BaseSettings):
    """Application settings with validation and type coercion."""

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        # Prefix all env vars with APP_ to avoid collisions
        env_prefix='APP_',
    )

    # Database
    database_url: PostgresDsn
    database_pool_size: int = Field(default=5, ge=1, le=50)

    # Security — SecretStr hides values in logs and repr
    secret_key: SecretStr
    jwt_secret: SecretStr
    jwt_expiry_minutes: int = Field(default=15, ge=1, le=1440)

    # External APIs
    stripe_api_key: SecretStr
    sendgrid_api_key: Optional[SecretStr] = None

    # Application
    debug: bool = False
    log_level: str = Field(default="INFO", pattern=r'^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$')

    # Server
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)


# ── Usage ────────────────────────────────────────────────────────
settings = Settings()

# SecretStr prevents accidental logging
print(settings.secret_key)
# Output: SecretStr('**********')

# Access the actual value when needed
actual_key = settings.secret_key.get_secret_value()

# Safe to print non-secret settings
print(f"Debug: {settings.debug}")
print(f"Port: {settings.port}")

# This will NOT reveal secret values
print(settings.model_dump())
# {'database_url': ..., 'secret_key': SecretStr('**********'), ...}
```

---

## 3. Secret Rotation Strategies

### 3.1 Why Rotate Secrets?

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Secret Rotation Reasons                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Limit blast radius of compromise                                 │
│     Old secret compromised ──▶ Only valid until next rotation        │
│                                                                      │
│  2. Compliance requirements                                          │
│     PCI DSS, SOC 2, HIPAA require periodic rotation                  │
│                                                                      │
│  3. Personnel changes                                                │
│     Employee leaves ──▶ Rotate all secrets they had access to        │
│                                                                      │
│  4. Reduce value of stolen secrets                                   │
│     Short-lived secrets are less useful to attackers                  │
│                                                                      │
│  Rotation frequency recommendations:                                 │
│  ├── API keys:         Every 90 days                                │
│  ├── Database passwords: Every 30-90 days                           │
│  ├── TLS certificates:  Before expiry (automated with ACME/Let's Encrypt) │
│  ├── JWT signing keys:  Every 30 days                               │
│  └── Encryption keys:   Every 365 days (with re-encryption plan)    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Zero-Downtime Rotation Pattern

```
┌─────────────────────────────────────────────────────────────────────┐
│               Zero-Downtime Secret Rotation                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Step 1: Generate new secret (keep old one active)                   │
│  ┌──────────┐                                                       │
│  │ Secret A │ ← current (active)                                    │
│  │ Secret B │ ← new (active)                                        │
│  └──────────┘                                                       │
│  Both secrets are valid simultaneously                               │
│                                                                      │
│  Step 2: Update all consumers to use new secret                      │
│  ┌──────────┐                                                       │
│  │ Secret A │ ← old (still active for grace period)                 │
│  │ Secret B │ ← all services now using this                         │
│  └──────────┘                                                       │
│                                                                      │
│  Step 3: Revoke old secret after grace period                        │
│  ┌──────────┐                                                       │
│  │ Secret A │ ← revoked                                             │
│  │ Secret B │ ← sole active secret                                  │
│  └──────────┘                                                       │
│                                                                      │
│  This dual-secret window prevents downtime during rotation.          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
"""
Implementing zero-downtime secret rotation for JWT signing keys.
"""
import jwt
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional

@dataclass
class SigningKey:
    """A JWT signing key with metadata."""
    key_id: str
    secret: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    revoked: bool = False


class KeyRotationManager:
    """Manages JWT signing key rotation with zero downtime."""

    def __init__(self, rotation_interval_days: int = 30,
                 grace_period_days: int = 7):
        self.rotation_interval = timedelta(days=rotation_interval_days)
        self.grace_period = timedelta(days=grace_period_days)
        self.keys: list[SigningKey] = []
        self._generate_new_key()

    def _generate_new_key(self) -> SigningKey:
        """Generate a new signing key."""
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
        """Get the current (newest) non-revoked signing key."""
        active_keys = [k for k in self.keys if not k.revoked]
        if not active_keys:
            return self._generate_new_key()
        return active_keys[-1]  # Most recently created

    @property
    def valid_keys(self) -> list[SigningKey]:
        """Get all valid (non-revoked, non-expired) keys."""
        now = datetime.now(timezone.utc)
        return [
            k for k in self.keys
            if not k.revoked and (k.expires_at is None or k.expires_at > now)
        ]

    def rotate(self) -> SigningKey:
        """Rotate to a new signing key."""
        # Generate new key
        new_key = self._generate_new_key()

        # Old keys remain valid until they expire (grace period)
        # This allows existing tokens signed with old key to remain valid

        # Clean up expired and revoked keys
        now = datetime.now(timezone.utc)
        self.keys = [
            k for k in self.keys
            if not k.revoked and (k.expires_at is None or k.expires_at > now)
        ]

        return new_key

    def sign_token(self, payload: dict) -> str:
        """Sign a JWT with the current key."""
        key = self.current_key
        headers = {"kid": key.key_id}
        return jwt.encode(payload, key.secret, algorithm="HS256",
                         headers=headers)

    def verify_token(self, token: str) -> dict:
        """Verify a JWT, trying all valid keys."""
        # First, try to get kid from header
        try:
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")
        except jwt.DecodeError:
            raise ValueError("Invalid token format")

        # If kid is present, find the specific key
        if kid:
            for key in self.valid_keys:
                if key.key_id == kid:
                    return jwt.decode(token, key.secret,
                                     algorithms=["HS256"])
            raise ValueError(f"Unknown key ID: {kid}")

        # Fallback: try all valid keys (for tokens without kid)
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
        """Check if the current key should be rotated."""
        key = self.current_key
        age = datetime.now(timezone.utc) - key.created_at
        return age >= self.rotation_interval


# ── Usage ────────────────────────────────────────────────────────
manager = KeyRotationManager(
    rotation_interval_days=30,
    grace_period_days=7,
)

# Sign a token
token = manager.sign_token({"sub": "user_123", "role": "admin"})

# Later, verify the token (works even after rotation)
payload = manager.verify_token(token)

# Periodic rotation (call from a scheduled task)
if manager.should_rotate():
    new_key = manager.rotate()
    print(f"Rotated to new key: {new_key.key_id}")
```

### 3.3 Database Password Rotation

```python
"""
Database password rotation strategy.
"""
import psycopg2
import secrets
import logging

logger = logging.getLogger('secret_rotation')


class DatabasePasswordRotator:
    """Rotate database passwords with zero downtime."""

    def __init__(self, admin_conn_string: str):
        self.admin_conn_string = admin_conn_string

    def rotate_password(self, username: str) -> str:
        """Rotate the password for a database user.

        Strategy:
        1. Generate new password
        2. Update password in database
        3. Update application config
        4. Verify connectivity with new password
        5. If verification fails, roll back
        """
        new_password = secrets.token_urlsafe(32)

        conn = psycopg2.connect(self.admin_conn_string)
        conn.autocommit = True

        try:
            with conn.cursor() as cur:
                # Step 1: Change password
                # Use format() carefully — this is a DDL command
                # that cannot use parameterized queries
                cur.execute(
                    f"ALTER USER {username} WITH PASSWORD %s",
                    (new_password,)
                )
                logger.info(f"Password rotated for user: {username}")

            # Step 2: Verify the new password works
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


# ── Automated rotation with scheduler ───────────────────────────
"""
# Using APScheduler
from apscheduler.schedulers.blocking import BlockingScheduler

scheduler = BlockingScheduler()

@scheduler.scheduled_job('cron', day='1', hour='3')  # 1st of each month, 3 AM
def rotate_db_passwords():
    rotator = DatabasePasswordRotator(admin_conn_string=ADMIN_DB_URL)
    new_password = rotator.rotate_password('app_user')

    # Update secret store (Vault, AWS Secrets Manager, etc.)
    update_secret_store('db_password', new_password)

    # Notify application to reload config
    notify_config_reload()

scheduler.start()
"""
```

---

## 4. HashiCorp Vault

### 4.1 Vault Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HashiCorp Vault Architecture                       │
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
│                     │  │  KV v2     │  │  (static key-value)       │
│                     │  │  Database  │  │  (dynamic credentials)    │
│                     │  │  PKI       │  │  (certificate authority)  │
│                     │  │  Transit   │  │  (encryption as service)  │
│                     │  │  AWS       │  │  (dynamic IAM creds)      │
│                     │  └────────────┘  │                           │
│                     └──────────────────┘                             │
│                                                                      │
│  Key features:                                                       │
│  • Dynamic secrets: generate credentials on-demand                   │
│  • Leasing: secrets have TTL, auto-expire                           │
│  • Revocation: revoke any secret or tree of secrets instantly       │
│  • Encryption as a service: encrypt/decrypt without seeing keys     │
│  • Audit logging: every access is logged                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Vault Quick Start

```bash
# ── Install Vault ────────────────────────────────────────────────
# macOS
brew install vault

# Linux
wget https://releases.hashicorp.com/vault/1.15.0/vault_1.15.0_linux_amd64.zip
unzip vault_1.15.0_linux_amd64.zip
sudo mv vault /usr/local/bin/

# ── Start development server (NOT for production) ───────────────
vault server -dev
# Root token will be printed — save it
# export VAULT_ADDR='http://127.0.0.1:8200'
# export VAULT_TOKEN='hvs.xxxxxxxxxxxxx'

# ── Enable KV secrets engine ────────────────────────────────────
vault secrets enable -path=secret kv-v2

# ── Store a secret ──────────────────────────────────────────────
vault kv put secret/myapp/database \
    username="dbuser" \
    password="supersecretpassword" \
    host="db.example.com" \
    port="5432"

# ── Read a secret ───────────────────────────────────────────────
vault kv get secret/myapp/database
vault kv get -format=json secret/myapp/database

# ── Read specific field ─────────────────────────────────────────
vault kv get -field=password secret/myapp/database

# ── List secrets ────────────────────────────────────────────────
vault kv list secret/myapp/

# ── Delete a secret ─────────────────────────────────────────────
vault kv delete secret/myapp/database

# ── Version history (KV v2) ────────────────────────────────────
vault kv get -version=1 secret/myapp/database
```

### 4.3 Vault with Python (hvac)

```python
"""
HashiCorp Vault client for Python.
pip install hvac
"""
import hvac
import os
from typing import Optional


class VaultClient:
    """Wrapper for HashiCorp Vault operations."""

    def __init__(self, url: str = None, token: str = None):
        self.client = hvac.Client(
            url=url or os.getenv('VAULT_ADDR', 'http://127.0.0.1:8200'),
            token=token or os.getenv('VAULT_TOKEN'),
        )
        if not self.client.is_authenticated():
            raise RuntimeError("Vault authentication failed")

    # ── KV Secrets ───────────────────────────────────────────────
    def get_secret(self, path: str, mount_point: str = 'secret') -> dict:
        """Read a secret from KV v2."""
        response = self.client.secrets.kv.v2.read_secret_version(
            path=path,
            mount_point=mount_point,
        )
        return response['data']['data']

    def set_secret(self, path: str, data: dict,
                   mount_point: str = 'secret') -> None:
        """Write a secret to KV v2."""
        self.client.secrets.kv.v2.create_or_update_secret(
            path=path,
            secret=data,
            mount_point=mount_point,
        )

    def delete_secret(self, path: str,
                      mount_point: str = 'secret') -> None:
        """Delete a secret."""
        self.client.secrets.kv.v2.delete_metadata_and_all_versions(
            path=path,
            mount_point=mount_point,
        )

    # ── Dynamic Database Credentials ────────────────────────────
    def get_database_creds(self, role: str) -> dict:
        """Get dynamic database credentials."""
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
        """Revoke a dynamic secret lease."""
        self.client.sys.revoke_lease(lease_id)

    # ── Transit Encryption ──────────────────────────────────────
    def encrypt(self, key_name: str, plaintext: str) -> str:
        """Encrypt data using Vault Transit engine."""
        import base64
        b64 = base64.b64encode(plaintext.encode()).decode()
        response = self.client.secrets.transit.encrypt_data(
            name=key_name,
            plaintext=b64,
        )
        return response['data']['ciphertext']

    def decrypt(self, key_name: str, ciphertext: str) -> str:
        """Decrypt data using Vault Transit engine."""
        import base64
        response = self.client.secrets.transit.decrypt_data(
            name=key_name,
            ciphertext=ciphertext,
        )
        return base64.b64decode(response['data']['plaintext']).decode()


# ── Usage ────────────────────────────────────────────────────────
vault = VaultClient()

# Store a secret
vault.set_secret('myapp/database', {
    'username': 'dbuser',
    'password': 'supersecret',
    'host': 'db.example.com',
})

# Read a secret
db_config = vault.get_secret('myapp/database')
print(f"Connecting to {db_config['host']} as {db_config['username']}")

# Get dynamic database credentials (auto-expire)
creds = vault.get_database_creds('readonly')
print(f"Temporary user: {creds['username']}")
print(f"Expires in: {creds['lease_duration']} seconds")

# Encrypt sensitive data
ciphertext = vault.encrypt('my-key', 'Social Security: 123-45-6789')
# ciphertext: vault:v1:8SDd3WHDOjf7mq69CyCqYjBXAiQQAVZRkFM13ok481zVCKqkLQ==

# Decrypt
plaintext = vault.decrypt('my-key', ciphertext)
# plaintext: Social Security: 123-45-6789
```

### 4.4 Vault AppRole Authentication

```python
"""
Vault AppRole authentication for applications (not humans).
"""
import hvac
import os


def authenticate_with_approle(vault_addr: str, role_id: str,
                               secret_id: str) -> hvac.Client:
    """Authenticate to Vault using AppRole method.

    AppRole is designed for machine-to-machine authentication.
    role_id = like a username (stable, configured in Vault)
    secret_id = like a password (rotatable, short-lived)
    """
    client = hvac.Client(url=vault_addr)

    # Login with AppRole
    response = client.auth.approle.login(
        role_id=role_id,
        secret_id=secret_id,
    )

    # Client is now authenticated
    client.token = response['auth']['client_token']
    print(f"Authenticated. Token TTL: {response['auth']['lease_duration']}s")

    return client


# ── In production, secret_id is injected by the orchestrator ────
# Kubernetes: mounted as a file
# Docker: passed as environment variable
# CI/CD: stored in pipeline secrets

vault = authenticate_with_approle(
    vault_addr=os.getenv('VAULT_ADDR'),
    role_id=os.getenv('VAULT_ROLE_ID'),
    secret_id=os.getenv('VAULT_SECRET_ID'),  # Short-lived
)

# Now use vault to fetch application secrets
secrets = vault.secrets.kv.v2.read_secret_version(path='myapp/config')
```

---

## 5. Cloud Secret Managers

### 5.1 AWS Secrets Manager

```python
"""
AWS Secrets Manager client.
pip install boto3
"""
import boto3
import json
from botocore.exceptions import ClientError


class AWSSecretsManager:
    """Interface to AWS Secrets Manager."""

    def __init__(self, region: str = 'us-east-1'):
        self.client = boto3.client(
            'secretsmanager',
            region_name=region,
        )

    def get_secret(self, secret_name: str) -> dict:
        """Retrieve a secret value."""
        try:
            response = self.client.get_secret_value(
                SecretId=secret_name,
            )
            # Secrets can be string or binary
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
        """Create a new secret."""
        response = self.client.create_secret(
            Name=name,
            Description=description,
            SecretString=json.dumps(value),
        )
        return response['ARN']

    def update_secret(self, name: str, value: dict) -> None:
        """Update an existing secret."""
        self.client.update_secret(
            SecretId=name,
            SecretString=json.dumps(value),
        )

    def rotate_secret(self, name: str, rotation_lambda_arn: str,
                      rotation_days: int = 30) -> None:
        """Enable automatic rotation for a secret."""
        self.client.rotate_secret(
            SecretId=name,
            RotationLambdaARN=rotation_lambda_arn,
            RotationRules={
                'AutomaticallyAfterDays': rotation_days,
            },
        )

    def delete_secret(self, name: str,
                      recovery_days: int = 30) -> None:
        """Delete a secret with a recovery window."""
        self.client.delete_secret(
            SecretId=name,
            RecoveryWindowInDays=recovery_days,
        )


# ── Usage ────────────────────────────────────────────────────────
sm = AWSSecretsManager(region='us-east-1')

# Store database credentials
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

# Retrieve and use
db_config = sm.get_secret('prod/myapp/database')
connection_string = (
    f"postgresql://{db_config['username']}:{db_config['password']}"
    f"@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
)
```

### 5.2 GCP Secret Manager

```python
"""
Google Cloud Secret Manager client.
pip install google-cloud-secret-manager
"""
from google.cloud import secretmanager


class GCPSecretManager:
    """Interface to GCP Secret Manager."""

    def __init__(self, project_id: str):
        self.client = secretmanager.SecretManagerServiceClient()
        self.project_id = project_id

    def _secret_path(self, secret_id: str, version: str = 'latest') -> str:
        """Build the full resource path."""
        return (
            f"projects/{self.project_id}/secrets/{secret_id}"
            f"/versions/{version}"
        )

    def get_secret(self, secret_id: str,
                   version: str = 'latest') -> str:
        """Retrieve a secret value."""
        name = self._secret_path(secret_id, version)
        response = self.client.access_secret_version(
            request={"name": name}
        )
        return response.payload.data.decode('UTF-8')

    def create_secret(self, secret_id: str, value: str) -> str:
        """Create a new secret with an initial version."""
        parent = f"projects/{self.project_id}"

        # Create the secret
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

        # Add a version with the actual value
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
        """Add a new version to an existing secret."""
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
        """Disable a secret version (soft delete)."""
        name = self._secret_path(secret_id, version)
        self.client.disable_secret_version(
            request={"name": name}
        )

    def delete_secret(self, secret_id: str) -> None:
        """Delete a secret and all its versions."""
        name = f"projects/{self.project_id}/secrets/{secret_id}"
        self.client.delete_secret(request={"name": name})


# ── Usage ────────────────────────────────────────────────────────
sm = GCPSecretManager(project_id='my-project-123')

# Create a secret
sm.create_secret('database-password', 'my_secret_password')

# Read the latest version
password = sm.get_secret('database-password')

# Rotate: add a new version
sm.add_version('database-password', 'new_rotated_password')

# The old version is still accessible by version number
old_password = sm.get_secret('database-password', version='1')
```

---

## 6. Git Secrets Scanning

### 6.1 The Problem

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Secrets in Git — The Problem                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Common scenarios for secret leakage:                                │
│                                                                      │
│  1. Accidental commit of .env file                                   │
│     $ git add .                                                      │
│     $ git commit -m "initial commit"                                 │
│     # .env with real passwords is now in git history FOREVER         │
│                                                                      │
│  2. Hardcoded API key in source code                                 │
│     api_key = "sk_live_EXAMPLE_KEY_REPLACE_ME"                   │
│     # Even if deleted later, it exists in git history                │
│                                                                      │
│  3. Configuration file with credentials                              │
│     database:                                                        │
│       password: "production_password_123"                            │
│                                                                      │
│  4. Test fixtures with real credentials                              │
│     STRIPE_KEY = "sk_live_..." # "test" key is actually live         │
│                                                                      │
│  IMPORTANT: git rm does NOT remove from history!                     │
│  The secret remains accessible via: git log --all --full-history     │
│  Removing requires: git filter-branch or BFG Repo-Cleaner           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Pre-Commit Hooks

```bash
# ── git-secrets (AWS) ────────────────────────────────────────────
# Install
brew install git-secrets  # macOS
# or: git clone https://github.com/awslabs/git-secrets.git && make install

# Set up for a repository
cd /path/to/repo
git secrets --install        # Install hooks
git secrets --register-aws   # Register AWS patterns

# Add custom patterns
git secrets --add 'PRIVATE_KEY'
git secrets --add 'password\s*=\s*.+'
git secrets --add --allowed 'password\s*=\s*os\.getenv'  # Allow env lookups

# Test scanning
git secrets --scan           # Scan staged changes
git secrets --scan-history   # Scan entire history

# ── pre-commit framework ────────────────────────────────────────
# pip install pre-commit
```

```yaml
# .pre-commit-config.yaml
repos:
  # Detect secrets before they are committed
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']

  # gitleaks — comprehensive secret scanner
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks

  # Check for private keys
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: detect-private-key
      - id: check-added-large-files
        args: ['--maxkb=100']
```

```bash
# Install the pre-commit hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Run specific hook
pre-commit run detect-secrets --all-files
```

### 6.3 Scanning Tools

```bash
# ── gitleaks — Fast, comprehensive scanner ──────────────────────
# Install
brew install gitleaks  # macOS
# or: go install github.com/gitleaks/gitleaks/v8@latest

# Scan current repository
gitleaks detect --source . --verbose

# Scan specific commit range
gitleaks detect --source . --log-opts="HEAD~10..HEAD"

# Scan entire history
gitleaks detect --source . --log-opts="--all"

# Generate report
gitleaks detect --source . --report-format json --report-path report.json

# Use custom rules
gitleaks detect --source . --config gitleaks.toml

# ── trufflehog — High-accuracy scanner ──────────────────────────
# Install
pip install trufflehog

# or use Docker
docker run --rm -v "$PWD:/repo" trufflesecurity/trufflehog:latest \
    git file:///repo --only-verified

# Scan a repo
trufflehog git file:///path/to/repo --only-verified

# Scan a GitHub repo directly
trufflehog github --org your-org --only-verified

# ── detect-secrets (Yelp) ───────────────────────────────────────
# Install
pip install detect-secrets

# Create a baseline (marks existing findings as accepted)
detect-secrets scan > .secrets.baseline

# Audit the baseline interactively
detect-secrets audit .secrets.baseline

# Scan for new secrets (compared to baseline)
detect-secrets scan --baseline .secrets.baseline
```

### 6.4 Custom gitleaks Configuration

```toml
# gitleaks.toml — Custom rules for secret detection
title = "Custom Gitleaks Config"

# Extend the default rules
[extend]
useDefault = true

# Custom rules
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

# Allowlist (false positive suppression)
[allowlist]
paths = [
    '''\.env\.example''',
    '''\.env\.template''',
    '''test_.*\.py''',        # Be careful with this
    '''docs/.*\.md''',
]
regexes = [
    '''password\s*=\s*os\.getenv''',    # Reading from env is OK
    '''password\s*=\s*["']changeme["']''',  # Placeholder values
    '''EXAMPLE_KEY_DO_NOT_USE''',
]
```

### 6.5 Removing Secrets from Git History

```bash
# ── BFG Repo-Cleaner (recommended) ─────────────────────────────
# Install: brew install bfg

# Remove a file from all history
bfg --delete-files '.env' my-repo.git

# Replace text in all files across history
echo "sk_live_EXAMPLE_KEY_REPLACE_ME" > passwords.txt
bfg --replace-text passwords.txt my-repo.git

# After BFG, clean up
cd my-repo.git
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push (WARNING: rewriting history)
git push --force

# ── git filter-repo (alternative) ──────────────────────────────
# pip install git-filter-repo

# Remove a file from history
git filter-repo --path .env --invert-paths

# Replace a string in all files
git filter-repo --replace-text <(echo 'literal:sk_live_EXAMPLE_KEY_REPLACE_ME==>REDACTED')

# ── IMPORTANT ───────────────────────────────────────────────────
# After removing secrets from history:
# 1. Force push to remote (all collaborators must re-clone)
# 2. IMMEDIATELY rotate the compromised secret
# 3. The secret was already exposed — removing from git is damage control
# 4. GitHub caches may still have the data temporarily
# 5. Any fork may still contain the secret
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

# ── Accessing secrets in GitHub Actions ──────────────────────────
jobs:
  deploy:
    runs-on: ubuntu-latest

    # Use environment-level secrets for different stages
    environment: production

    steps:
      - uses: actions/checkout@v4

      # Secrets are available as environment variables
      - name: Configure AWS
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_REGION: us-east-1
        run: |
          aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
          aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
          aws configure set region $AWS_REGION

      # Use secrets directly (masked in logs)
      - name: Deploy
        run: |
          echo "Deploying to production..."
          # ${{ secrets.DEPLOY_KEY }} is masked as *** in logs
          ./deploy.sh --key "${{ secrets.DEPLOY_KEY }}"

      # Docker login with secrets
      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      # OIDC authentication (preferred over static credentials)
      - name: Configure AWS with OIDC
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789:role/github-actions
          aws-region: us-east-1
          # No static credentials needed!
```

```yaml
# ── Security best practices for GitHub Actions ──────────────────
# 1. Use environment-level secrets with required reviewers
# 2. Use OIDC instead of static credentials where possible
# 3. Pin actions to specific SHA, not tags
# 4. Limit secret access to specific environments
# 5. Use GITHUB_TOKEN's automatic permissions (not PATs)

# .github/workflows/secure.yml
name: Secure Workflow

on:
  push:
    branches: [main]

permissions:
  contents: read  # Minimal permissions

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      # Pin actions to SHA (not tag) to prevent supply chain attacks
      - uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11  # v4.1.1

      # Use GITHUB_TOKEN for repository access (auto-scoped)
      - name: Create Release
        env:
          GH_TOKEN: ${{ github.token }}
        run: gh release create v1.0 --generate-notes

      # Never echo secrets (even accidentally)
      - name: Safe Logging
        env:
          API_KEY: ${{ secrets.API_KEY }}
        run: |
          # BAD: echo "Key is: $API_KEY"
          # GOOD: Test the key works without printing it
          curl -sf -H "Authorization: Bearer $API_KEY" \
               https://api.example.com/health || exit 1
```

### 7.2 GitLab CI Variables

```yaml
# .gitlab-ci.yml
# Variables are set in GitLab UI:
# Settings → CI/CD → Variables

stages:
  - test
  - deploy

test:
  stage: test
  variables:
    # Non-sensitive can be set here
    NODE_ENV: test
  script:
    # Protected variables only available on protected branches
    - echo "Running tests..."
    - pytest --cov
    # Masked variables are hidden in job logs
    - echo "DB URL is $DATABASE_URL"  # Shows as [MASKED]

deploy:
  stage: deploy
  # Restrict to protected branches only
  only:
    - main
  variables:
    # Use file-type variables for multi-line secrets (like certificates)
    # GitLab writes the value to a file and sets the variable to the path
    KUBE_CONFIG: $KUBE_CONFIG_FILE  # This is a file path
  script:
    - kubectl --kubeconfig="$KUBE_CONFIG" apply -f deployment.yaml
  environment:
    name: production

# ── GitLab CI Secret Best Practices ─────────────────────────────
# 1. Mark sensitive variables as "Masked" (hidden in logs)
# 2. Mark deployment secrets as "Protected" (only on protected branches)
# 3. Use "File" type for multi-line secrets (certs, keys)
# 4. Scope variables to specific environments
# 5. Use GitLab's external secrets integration (Vault, AWS, GCP)
```

---

## 8. Encryption at Rest

### 8.1 Encrypting Configuration Files

```python
"""
Encrypting configuration files with Fernet (symmetric encryption).
pip install cryptography
"""
from cryptography.fernet import Fernet
import json
import os
from pathlib import Path


class ConfigEncryptor:
    """Encrypt and decrypt configuration files."""

    def __init__(self, key: bytes = None):
        if key is None:
            # Load key from environment
            key_str = os.getenv('CONFIG_ENCRYPTION_KEY')
            if key_str is None:
                raise ValueError(
                    "CONFIG_ENCRYPTION_KEY environment variable not set"
                )
            key = key_str.encode()
        self.cipher = Fernet(key)

    @staticmethod
    def generate_key() -> str:
        """Generate a new encryption key."""
        return Fernet.generate_key().decode()

    def encrypt_config(self, config: dict, output_path: str) -> None:
        """Encrypt a configuration dictionary and write to file."""
        json_data = json.dumps(config, indent=2).encode()
        encrypted = self.cipher.encrypt(json_data)

        with open(output_path, 'wb') as f:
            f.write(encrypted)

    def decrypt_config(self, input_path: str) -> dict:
        """Read and decrypt a configuration file."""
        with open(input_path, 'rb') as f:
            encrypted = f.read()

        decrypted = self.cipher.decrypt(encrypted)
        return json.loads(decrypted)

    def encrypt_value(self, value: str) -> str:
        """Encrypt a single value."""
        return self.cipher.encrypt(value.encode()).decode()

    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a single value."""
        return self.cipher.decrypt(encrypted_value.encode()).decode()


# ── Usage ────────────────────────────────────────────────────────
# First time: generate a key
key = ConfigEncryptor.generate_key()
print(f"Store this key securely: {key}")
# Store in: environment variable, Vault, cloud KMS

# Encrypt config
encryptor = ConfigEncryptor(key.encode())
encryptor.encrypt_config(
    config={
        'database_url': 'postgresql://user:pass@host/db',
        'api_key': 'sk_live_xxxxx',
        'jwt_secret': 'supersecret',
    },
    output_path='config.enc',
)

# Decrypt config (at application startup)
config = encryptor.decrypt_config('config.enc')
print(config['database_url'])
```

### 8.2 SOPS (Secrets OPerationS)

```bash
# ── SOPS: Encrypted file management ─────────────────────────────
# SOPS encrypts values in YAML/JSON/ENV files while keeping keys visible
# This means you can diff, review, and commit encrypted files

# Install
brew install sops

# ── Using SOPS with age encryption ──────────────────────────────
# Generate an age key
age-keygen -o keys.txt
# Public key: age1xxxxxxx...
# Store private key securely

# Create .sops.yaml configuration
cat > .sops.yaml << 'EOF'
creation_rules:
  - path_regex: \.enc\.yaml$
    age: age1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  - path_regex: \.enc\.json$
    age: age1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
EOF

# Encrypt a file
sops --encrypt secrets.yaml > secrets.enc.yaml

# Decrypt a file
sops --decrypt secrets.enc.yaml > secrets.yaml

# Edit encrypted file in place (decrypts, opens editor, re-encrypts)
sops secrets.enc.yaml
```

```yaml
# secrets.enc.yaml — Values are encrypted, keys are readable
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
Loading SOPS-encrypted config in Python.
"""
import subprocess
import json
import yaml


def load_sops_config(path: str) -> dict:
    """Decrypt a SOPS file and return as dictionary."""
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


# Usage
config = load_sops_config('secrets.enc.yaml')
db_password = config['database']['password']
```

---

## 9. Common Mistakes

### 9.1 Anti-Patterns

```python
"""
Common secrets management mistakes — DO NOT DO THESE.
"""

# ── MISTAKE 1: Hardcoded secrets ────────────────────────────────
# BAD
API_KEY = "sk_live_EXAMPLE_KEY_REPLACE_ME"
DB_PASSWORD = "production_password_123"

# GOOD
import os
API_KEY = os.getenv("API_KEY")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# ── MISTAKE 2: Secrets in logs ──────────────────────────────────
# BAD
import logging
logger = logging.getLogger(__name__)
logger.info(f"Connecting to database with password: {password}")
logger.debug(f"API response: {response.headers}")  # May contain auth headers

# GOOD
logger.info("Connecting to database as user: %s", username)
logger.debug("API response status: %s", response.status_code)


# ── MISTAKE 3: Secrets in URLs ──────────────────────────────────
# BAD — Password appears in server logs, browser history, Referer headers
DATABASE_URL = "postgresql://admin:P@ssw0rd@db.example.com/prod"

# GOOD — Use separate variables
DATABASE_HOST = os.getenv("DB_HOST")
DATABASE_USER = os.getenv("DB_USER")
DATABASE_PASS = os.getenv("DB_PASS")


# ── MISTAKE 4: Secrets in error messages ────────────────────────
# BAD
try:
    conn = connect(password=secret_password)
except ConnectionError as e:
    raise RuntimeError(f"Failed to connect with password {secret_password}: {e}")

# GOOD
try:
    conn = connect(password=secret_password)
except ConnectionError as e:
    raise RuntimeError(f"Database connection failed: {e}")


# ── MISTAKE 5: Committing .env to git ───────────────────────────
# BAD: Forgetting .env in .gitignore
# Even worse: Adding .env then removing it (still in history)

# GOOD: Add to .gitignore BEFORE creating the file
# And use pre-commit hooks to prevent accidental commits


# ── MISTAKE 6: Same secret for all environments ─────────────────
# BAD: Using production keys in development
# This increases the blast radius of a development machine compromise

# GOOD: Different secrets for dev, staging, production
# Development uses test/sandbox API keys
# Production keys are only on production servers


# ── MISTAKE 7: Never rotating secrets ───────────────────────────
# BAD: API key created in 2019, never changed
# If compromised, attacker has had access for years

# GOOD: Automated rotation with zero-downtime strategy


# ── MISTAKE 8: Shared secrets among team members ────────────────
# BAD: "Hey, can you Slack me the production DB password?"

# GOOD:
# - Use a secret manager (Vault, AWS SM, 1Password)
# - Each person gets their own credentials
# - Use SSO/OIDC where possible (no shared secrets)
# - Secrets are accessed programmatically, not by humans


# ── MISTAKE 9: Secrets in Docker images ─────────────────────────
# BAD
# Dockerfile:
# ENV API_KEY=sk_live_xxxxx
# COPY .env /app/.env

# GOOD: Inject at runtime
# docker run -e API_KEY=$API_KEY myimage
# Or use Docker secrets / Kubernetes secrets


# ── MISTAKE 10: Client-side secrets ─────────────────────────────
# BAD: Embedding API keys in JavaScript/mobile apps
# <script>
#   const API_KEY = "sk_live_xxxxx";  // Visible to anyone
# </script>

# GOOD: Use a backend proxy
# Client → Your Backend (has the secret) → Third-party API
```

### 9.2 Secret Detection Checklist

```python
"""
Automated check for common secret patterns in code.
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

# Patterns that indicate potential secrets
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

# Patterns that indicate false positives
FALSE_POSITIVE_PATTERNS = [
    re.compile(r'(?i)password\s*=\s*os\.getenv'),
    re.compile(r'(?i)password\s*=\s*["\']changeme["\']'),
    re.compile(r'(?i)password\s*=\s*["\']<.*>["\']'),
    re.compile(r'(?i)password\s*=\s*["\']your_password_here["\']'),
    re.compile(r'(?i)password\s*=\s*["\']placeholder["\']'),
    re.compile(r'#.*password'),  # Comments
]

SKIP_DIRS = {'.git', 'node_modules', '__pycache__', '.venv', 'venv'}
SKIP_EXTENSIONS = {'.pyc', '.pyo', '.so', '.dylib', '.png', '.jpg', '.gif'}


def scan_directory(root: str) -> list[SecretFinding]:
    """Scan a directory for potential secrets."""
    findings = []

    for path in Path(root).rglob('*'):
        # Skip directories
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
                    # Check for false positives
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

## 10. Exercises

### Exercise 1: Environment Configuration System

Build a configuration management system that:

1. Loads settings from multiple sources with priority order:
   - Environment variables (highest priority)
   - `.env.local` file
   - `.env` file
   - Default values (lowest priority)
2. Validates all required settings at startup (fail fast)
3. Uses Pydantic `SecretStr` to prevent accidental logging of secrets
4. Provides a `config.dump_safe()` method that shows all settings with secrets masked
5. Supports type coercion (string to int, bool, list)
6. Write tests to verify the priority order and validation

### Exercise 2: Secret Rotation Service

Implement an automated secret rotation service that:

1. Manages a set of secrets with configurable rotation intervals
2. Supports zero-downtime rotation (dual-secret window)
3. Notifies registered consumers when secrets are rotated
4. Logs all rotation events for audit
5. Supports rollback if the new secret fails verification
6. Can be integrated with a scheduler (run rotation checks every hour)
7. Stores rotation history for compliance

### Exercise 3: Git Secret Scanner

Build a comprehensive git secret scanner that:

1. Scans staged files (pre-commit hook)
2. Scans entire git history for past leaks
3. Supports configurable patterns (regex-based)
4. Has a baseline/allowlist for known false positives
5. Generates reports in JSON and human-readable format
6. Can be run as a pre-commit hook or CI/CD step
7. Supports incremental scanning (only new commits)

### Exercise 4: Vault Integration Library

Create a Python library that:

1. Authenticates to Vault using AppRole or Token
2. Caches secrets locally with TTL (avoid repeated Vault calls)
3. Automatically refreshes secrets before they expire
4. Falls back to environment variables if Vault is unavailable
5. Provides a Django/Flask settings integration
6. Handles Vault token renewal transparently
7. Logs access patterns without revealing secret values

### Exercise 5: SOPS Workflow Automation

Build a command-line tool that:

1. Creates encrypted config files for multiple environments (dev, staging, prod)
2. Validates that encrypted files contain all required keys
3. Diffs two encrypted files without decrypting (compare structure)
4. Rotates encryption keys (re-encrypt all files with new key)
5. Integrates with git hooks to prevent committing unencrypted files
6. Generates a `.env` file from an encrypted config for local development

### Exercise 6: CI/CD Secrets Audit

Write an audit tool that:

1. Scans GitHub Actions workflow files for secret usage patterns
2. Identifies secrets that are passed insecurely (e.g., in command arguments visible in logs)
3. Checks if actions are pinned to SHA (not mutable tags)
4. Verifies that secrets are scoped to the correct environment
5. Detects if `GITHUB_TOKEN` permissions are overly broad
6. Generates a compliance report

---

## Summary

### Secrets Management Maturity Model

| Level | Description | Practices |
|-------|-------------|-----------|
| Level 0 | No management | Hardcoded secrets, committed to git |
| Level 1 | Basic | .env files, .gitignore, manual rotation |
| Level 2 | Intermediate | Secret manager (Vault/cloud), pre-commit hooks |
| Level 3 | Advanced | Automated rotation, dynamic secrets, audit logging |
| Level 4 | Optimal | Zero-trust, OIDC everywhere, continuous scanning |

### Key Takeaways

1. **Never hardcode secrets** — use environment variables or a secret manager
2. **Scan continuously** — use pre-commit hooks and CI/CD scanning
3. **Rotate regularly** — automate rotation with zero-downtime patterns
4. **Encrypt at rest** — use SOPS, Vault Transit, or cloud KMS
5. **Audit access** — log who accessed what secret and when
6. **Minimize exposure** — use short-lived credentials and least privilege
7. **Plan for compromise** — have a runbook for when (not if) a secret leaks

---

**Previous**: [10_API_Security.md](./10_API_Security.md) | **Next**: [12_Container_Security.md](./12_Container_Security.md)
