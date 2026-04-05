# Secrets Management

**이전**: [GitOps](./14_GitOps.md) | **다음**: [Chaos Engineering](./16_Chaos_Engineering.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 시크릿이 전문적인 관리가 필요한 이유와 평문 또는 Git에 저장할 때의 위험 설명하기
2. 시크릿 엔진, 정책, 인증 방법을 사용하여 HashiCorp Vault 배포 및 구성하기
3. 관리형 시크릿 라이프사이클을 위한 AWS Secrets Manager 및 기타 클라우드 네이티브 시크릿 저장소 사용하기
4. GitOps 호환성을 위해 SOPS와 Sealed Secrets를 사용하여 Git 저장용 시크릿 암호화하기
5. 요청 시 생성되고 자동으로 만료되는 동적 시크릿 구현하기
6. 장기 자격 증명을 제거하는 시크릿 로테이션 전략 설계하기

---

시크릿 -- API 키, 데이터베이스 비밀번호, TLS 인증서, 암호화 키, 서비스 계정 토큰 -- 은 모든 시스템에서 가장 민감한 데이터입니다. 유출된 시크릿은 데이터 유출, 금전적 손실, 규제 처벌로 이어질 수 있습니다. 그럼에도 불구하고 시크릿은 일상적으로 소스 코드에 하드코딩되고, Git에 커밋되고, 환경 변수에 전달되고, Slack 메시지로 공유됩니다. 이 레슨은 시크릿을 일급 인프라로 취급하는 도구와 관행을 다룹니다: 중앙 집중식 저장, 접근 제어, 감사 로깅, 동적 생성, 자동 로테이션.

> **비유 -- 은행 금고 vs 매트리스 아래**: 환경 변수나 설정 파일에 시크릿을 저장하는 것은 현금을 매트리스 아래에 두는 것과 같습니다 -- 집에 들어오는 누구나 가져갈 수 있고, 누가 접근했는지 기록이 없으며, 잠금을 변경할 방법이 없습니다. 시크릿 관리 시스템(Vault, AWS Secrets Manager)은 은행 금고와 같습니다: 접근하려면 인증이 필요하고, 모든 접근을 기록하며, 누가 무엇에 접근할 수 있는지 제한하고, 운영을 중단하지 않으면서 조합(로테이션)을 변경할 수 있습니다.

## 1. Secrets Management의 필요성

### 1.1 일반적인 안티패턴

| 안티패턴 | 위험 |
|---------|------|
| **소스 코드에 하드코딩** | 삭제 후에도 시크릿이 Git 이력에 영구 존재 |
| **환경 변수** | 프로세스 목록, 컨테이너 inspect, crash dump에서 볼 수 있음 |
| **공유 `.env` 파일** | 접근 제어 없음, 감사 추적 없음, 팀 간 복사-붙여넣기 |
| **Slack/이메일 공유** | 서드파티 시스템에 저장, 검색 가능, 만료 없음 |
| **모든 곳에 같은 비밀번호** | 하나의 침해가 모든 시스템을 노출 |
| **로테이션 안 함** | 유출된 자격 증명이 무한정 유효 |

### 1.2 Secrets Management 요구사항

| 요구사항 | 설명 |
|---------|------|
| **저장 시 암호화** | 저장 시 시크릿 암호화 |
| **전송 시 암호화** | 모든 시크릿 접근에 TLS |
| **접근 제어** | 세밀한 정책 (누가 어떤 시크릿에 접근할 수 있는지) |
| **감사 로깅** | 모든 접근과 수정 기록 |
| **로테이션** | 다운타임 없이 시크릿 변경 가능 |
| **동적 시크릿** | 요청 시 생성되는 단기 자격 증명 |
| **폐기** | 침해된 시크릿을 즉시 무효화하는 기능 |
| **버전 관리** | 롤백을 위한 이전 버전 보존 |

---

## 2. HashiCorp Vault

### 2.1 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                       Vault Architecture                         │
│                                                                  │
│  ┌──────────┐   ┌──────────────────────────────────────────┐   │
│  │  Client   │──→│              Vault Server                │   │
│  │(CLI/API/  │   │  ┌──────────────┐  ┌──────────────┐    │   │
│  │ SDK)      │   │  │   Auth       │  │   Secret     │    │   │
│  └──────────┘   │  │   Methods    │  │   Engines    │    │   │
│                  │  │  ┌────────┐  │  │  ┌────────┐  │    │   │
│                  │  │  │Token   │  │  │  │KV v2   │  │    │   │
│                  │  │  │AppRole │  │  │  │Database│  │    │   │
│                  │  │  │K8s    │  │  │  │PKI     │  │    │   │
│                  │  │  │OIDC   │  │  │  │AWS     │  │    │   │
│                  │  │  │LDAP   │  │  │  │Transit │  │    │   │
│                  │  │  └────────┘  │  │  └────────┘  │    │   │
│                  │  └──────────────┘  └──────────────┘    │   │
│                  │                                          │   │
│                  │  ┌──────────────┐  ┌──────────────┐    │   │
│                  │  │   Audit      │  │   Storage    │    │   │
│                  │  │   Devices    │  │   Backend    │    │   │
│                  │  │  (file/      │  │  (Consul/    │    │   │
│                  │  │   syslog)    │  │   Raft/      │    │   │
│                  │  │              │  │   S3)        │    │   │
│                  │  └──────────────┘  └──────────────┘    │   │
│                  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 설치 및 초기화

```bash
# Start Vault in dev mode (for learning only -- data is in-memory)
vault server -dev -dev-root-token-id="root"

# Production: start with config file
vault server -config=/etc/vault/config.hcl

# Initialize Vault (first time only)
vault operator init -key-shares=5 -key-threshold=3
# Produces 5 unseal keys and a root token
# Store keys in SEPARATE secure locations (never together)

# Unseal Vault (requires 3 of 5 keys)
vault operator unseal <key-1>
vault operator unseal <key-2>
vault operator unseal <key-3>

# Login
export VAULT_ADDR='http://127.0.0.1:8200'
vault login <root-token>
```

### 2.3 KV (Key-Value) Secret Engine

```bash
# Enable KV v2 engine
vault secrets enable -path=secret kv-v2

# Store a secret
vault kv put secret/myapp/database \
  username="dbadmin" \
  password="s3cur3P@ssw0rd" \
  host="db.example.com" \
  port="5432"

# Read a secret
vault kv get secret/myapp/database

# Read specific field
vault kv get -field=password secret/myapp/database

# Read as JSON
vault kv get -format=json secret/myapp/database

# List secrets
vault kv list secret/myapp/

# Update a secret (creates new version)
vault kv put secret/myapp/database \
  username="dbadmin" \
  password="n3wP@ssw0rd" \
  host="db.example.com" \
  port="5432"

# Read a previous version
vault kv get -version=1 secret/myapp/database

# Delete current version (soft delete)
vault kv delete secret/myapp/database

# Undelete
vault kv undelete -versions=2 secret/myapp/database

# Permanently destroy a version
vault kv destroy -versions=1 secret/myapp/database
```

### 2.4 정책

정책은 토큰이 접근할 수 있는 것을 정의합니다:

```hcl
# policy: myapp-read.hcl
# Allow read access to myapp secrets
path "secret/data/myapp/*" {
  capabilities = ["read", "list"]
}

# Deny access to production secrets
path "secret/data/production/*" {
  capabilities = ["deny"]
}

# Allow the app to renew its own token
path "auth/token/renew-self" {
  capabilities = ["update"]
}
```

```bash
# Create a policy
vault policy write myapp-read myapp-read.hcl

# List policies
vault policy list

# Read a policy
vault policy read myapp-read
```

### 2.5 인증 방법

```bash
# --- AppRole (for applications/CI systems) ---
vault auth enable approle

# Create a role
vault write auth/approle/role/myapp \
  token_policies="myapp-read" \
  secret_id_ttl=24h \
  token_ttl=1h \
  token_max_ttl=4h

# Get role ID (embed in app config -- not secret)
vault read auth/approle/role/myapp/role-id

# Generate secret ID (deliver securely to the app)
vault write -f auth/approle/role/myapp/secret-id

# Application authenticates
vault write auth/approle/login \
  role_id="<role-id>" \
  secret_id="<secret-id>"

# --- Kubernetes Auth (for pods) ---
vault auth enable kubernetes

vault write auth/kubernetes/config \
  kubernetes_host="https://kubernetes.default.svc:443"

vault write auth/kubernetes/role/myapp \
  bound_service_account_names=myapp \
  bound_service_account_namespaces=production \
  policies=myapp-read \
  ttl=1h
```

### 2.6 동적 시크릿 (데이터베이스)

동적 시크릿은 요청 시 생성되고 TTL 후 자동으로 폐기됩니다:

```bash
# Enable database secret engine
vault secrets enable database

# Configure PostgreSQL connection
vault write database/config/mydb \
  plugin_name=postgresql-database-plugin \
  allowed_roles="myapp-role" \
  connection_url="postgresql://{{username}}:{{password}}@db.example.com:5432/myapp" \
  username="vault_admin" \
  password="vault_admin_password"

# Create a role with a creation SQL statement
vault write database/roles/myapp-role \
  db_name=mydb \
  creation_statements="CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}'; \
    GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO \"{{name}}\";" \
  revocation_statements="REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM \"{{name}}\"; \
    DROP ROLE IF EXISTS \"{{name}}\";" \
  default_ttl="1h" \
  max_ttl="24h"

# Generate dynamic credentials
vault read database/creds/myapp-role
# Returns:
#   username: v-approle-myapp-role-abc123
#   password: A1B2C3D4E5F6G7H8
#   lease_id: database/creds/myapp-role/xyz789
#   lease_duration: 1h

# Renew a lease
vault lease renew database/creds/myapp-role/xyz789

# Revoke immediately (incident response)
vault lease revoke database/creds/myapp-role/xyz789

# Revoke ALL leases for a role
vault lease revoke -prefix database/creds/myapp-role
```

**동적 시크릿의 이점:**
- 탈취할 장기 자격 증명 없음
- 각 애플리케이션 인스턴스가 고유한 자격 증명을 받음 (접근 추적 가능)
- TTL 후 자동 폐기 (오래된 자격 증명 없음)
- 인시던트 중 즉시 폐기

---

## 3. AWS Secrets Manager

### 3.1 핵심 기능

| 기능 | 설명 |
|------|------|
| **저장** | 암호화된 저장 (AWS KMS를 통한 AES-256) |
| **로테이션** | RDS, Redshift, DocumentDB를 위한 내장 Lambda 기반 로테이션 |
| **접근 제어** | IAM 정책 및 리소스 정책 |
| **감사** | 모든 API 호출에 대한 CloudTrail 로깅 |
| **버전 관리** | `AWSCURRENT`와 `AWSPREVIOUS` 스테이징 라벨 |
| **교차 계정** | 리소스 정책을 통한 AWS 계정 간 시크릿 공유 |

### 3.2 CLI 작업

```bash
# Create a secret
aws secretsmanager create-secret \
  --name myapp/database \
  --description "Database credentials for myapp" \
  --secret-string '{"username":"dbadmin","password":"s3cur3P@ssw0rd","host":"db.example.com","port":"5432"}'

# Retrieve a secret
aws secretsmanager get-secret-value \
  --secret-id myapp/database \
  --query 'SecretString' --output text

# Update a secret
aws secretsmanager update-secret \
  --secret-id myapp/database \
  --secret-string '{"username":"dbadmin","password":"n3wP@ssw0rd","host":"db.example.com","port":"5432"}'

# Enable automatic rotation (every 30 days)
aws secretsmanager rotate-secret \
  --secret-id myapp/database \
  --rotation-lambda-arn arn:aws:lambda:us-east-1:123456789012:function:SecretsManagerRotation \
  --rotation-rules AutomaticallyAfterDays=30

# Trigger immediate rotation
aws secretsmanager rotate-secret \
  --secret-id myapp/database

# Delete a secret (with recovery window)
aws secretsmanager delete-secret \
  --secret-id myapp/database \
  --recovery-window-in-days 7
```

### 3.3 애플리케이션 통합 (Python)

```python
import json
import boto3
from botocore.exceptions import ClientError

def get_secret(secret_name: str, region: str = "us-east-1") -> dict:
    """Retrieve a secret from AWS Secrets Manager."""
    client = boto3.client("secretsmanager", region_name=region)

    try:
        response = client.get_secret_value(SecretId=secret_name)
        secret = json.loads(response["SecretString"])
        return secret
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            raise ValueError(f"Secret {secret_name} not found")
        raise

# Usage
db_creds = get_secret("myapp/database")
connection_string = (
    f"postgresql://{db_creds['username']}:{db_creds['password']}"
    f"@{db_creds['host']}:{db_creds['port']}/myapp"
)
```

### 3.4 Kubernetes 통합 (External Secrets Operator)

```yaml
# Install External Secrets Operator (ESO)
# ESO syncs secrets from AWS/Vault/GCP into Kubernetes Secrets

# SecretStore: defines the provider connection
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
  namespace: production
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa

---
# ExternalSecret: syncs a specific secret
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: myapp-database
  namespace: production
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: myapp-database-secret     # Kubernetes Secret name
    creationPolicy: Owner
  data:
    - secretKey: username
      remoteRef:
        key: myapp/database
        property: username
    - secretKey: password
      remoteRef:
        key: myapp/database
        property: password
    - secretKey: host
      remoteRef:
        key: myapp/database
        property: host
```

---

## 4. SOPS (Secrets OPerationS)

### 4.1 SOPS를 사용하는 이유

SOPS는 YAML/JSON 파일에서 키와 구조를 평문으로 유지하면서 시크릿 값을 암호화합니다. 이를 통해 암호화된 시크릿이 Git 및 코드 리뷰와 호환됩니다 -- 리뷰어가 값을 보지 않고도 어떤 필드가 변경되었는지 확인할 수 있습니다.

```yaml
# Before SOPS encryption
database:
  host: db.example.com
  username: dbadmin
  password: s3cur3P@ssw0rd

# After SOPS encryption (keys visible, values encrypted)
database:
  host: ENC[AES256_GCM,data:abc123...,iv:xyz...,tag:def...]
  username: ENC[AES256_GCM,data:ghi789...,iv:uvw...,tag:jkl...]
  password: ENC[AES256_GCM,data:mno456...,iv:rst...,tag:pqr...]
sops:
  kms:
    - arn: arn:aws:kms:us-east-1:123456789012:key/abc-def-ghi
  version: 3.8.1
```

### 4.2 설정 및 사용법

```bash
# Install SOPS
brew install sops    # macOS

# Create SOPS configuration
cat > .sops.yaml << 'EOF'
creation_rules:
  # Production: encrypted with AWS KMS
  - path_regex: production/.*
    kms: arn:aws:kms:us-east-1:123456789012:key/abc-def-ghi

  # Staging: encrypted with GCP KMS
  - path_regex: staging/.*
    gcp_kms: projects/myproject/locations/global/keyRings/sops/cryptoKeys/sops-key

  # Dev: encrypted with age key (local)
  - path_regex: dev/.*
    age: age1xyz...
EOF

# Encrypt a file
sops --encrypt secrets.yaml > secrets.enc.yaml

# Decrypt a file
sops --decrypt secrets.enc.yaml > secrets.yaml

# Edit encrypted file in place (decrypts → opens editor → re-encrypts)
sops secrets.enc.yaml

# Encrypt specific keys only (leave non-secret values plain)
sops --encrypt --encrypted-regex '^(password|token|key)$' secrets.yaml > secrets.enc.yaml
```

### 4.3 Kustomize와 SOPS (GitOps)

```yaml
# Use ksops (Kustomize SOPS plugin) to decrypt secrets during ArgoCD sync

# kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
generators:
  - secret-generator.yaml

# secret-generator.yaml
apiVersion: viaduct.ai/v1
kind: ksops
metadata:
  name: myapp-secrets
files:
  - secrets.enc.yaml

# secrets.enc.yaml (encrypted with SOPS)
apiVersion: v1
kind: Secret
metadata:
  name: myapp-database
type: Opaque
stringData:
  username: ENC[AES256_GCM,data:abc123...]
  password: ENC[AES256_GCM,data:def456...]
```

---

## 5. Sealed Secrets

### 5.1 Sealed Secrets 작동 원리

Sealed Secrets는 Kubernetes Secret을 공개 키로 암호화합니다. 클러스터 내의 Sealed Secrets 컨트롤러(개인 키를 보유)만 복호화할 수 있습니다. 암호화된 시크릿은 Git에 안전하게 커밋할 수 있습니다.

```
Developer Workstation                    Kubernetes Cluster
┌──────────────────┐                     ┌──────────────────────┐
│                  │                     │  Sealed Secrets      │
│  kubeseal CLI    │                     │  Controller          │
│  (encrypts with  │                     │  (decrypts with      │
│   public key)    │                     │   private key)       │
│                  │                     │                      │
│  Secret ──→      │   Git commit ───→   │  SealedSecret ──→   │
│  SealedSecret    │   (safe to store)   │  Kubernetes Secret   │
│                  │                     │                      │
└──────────────────┘                     └──────────────────────┘
```

### 5.2 사용법

```bash
# Install Sealed Secrets controller
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.25.0/controller.yaml

# Create a regular Kubernetes Secret (not committed to Git)
kubectl create secret generic myapp-database \
  --from-literal=username=dbadmin \
  --from-literal=password=s3cur3P@ssw0rd \
  --dry-run=client -o yaml > secret.yaml

# Encrypt with kubeseal
kubeseal --format yaml < secret.yaml > sealed-secret.yaml

# The sealed-secret.yaml is safe to commit to Git
cat sealed-secret.yaml
```

```yaml
# sealed-secret.yaml (safe for Git)
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: myapp-database
  namespace: production
spec:
  encryptedData:
    username: AgBy3i4OJSWK+P...  # Encrypted
    password: AgCtr8h5QkBz+Y...  # Encrypted
  template:
    metadata:
      name: myapp-database
    type: Opaque
```

```bash
# Apply the SealedSecret to the cluster
kubectl apply -f sealed-secret.yaml

# The controller decrypts it into a regular Secret
kubectl get secret myapp-database -o yaml
```

### 5.3 SOPS vs Sealed Secrets

| 측면 | SOPS | Sealed Secrets |
|------|------|---------------|
| **암호화 범위** | 모든 파일 (YAML, JSON, .env) | Kubernetes Secret만 |
| **키 관리** | 외부 KMS (AWS, GCP, age) | 컨트롤러 관리 키 쌍 |
| **가독성** | 키 보임, 값 암호화 | 전체 데이터 블록 암호화 |
| **멀티 클러스터** | 동일 KMS 키가 어디서든 작동 | 클러스터당 하나의 키 쌍 |
| **GitOps 호환** | 예 (ksops 플러그인 사용) | 예 (네이티브 Kubernetes 리소스) |
| **적합한 용도** | 멀티 클라우드, Kubernetes 이외의 시크릿 | Kubernetes 전용 환경 |

---

## 6. Secret 로테이션

### 6.1 시크릿 로테이션이 필요한 이유

| 이유 | 설명 |
|------|------|
| **노출 창 제한** | 시크릿이 유출되면 로테이션으로 유효 기간 제한 |
| **컴플라이언스** | PCI-DSS, SOC 2, HIPAA가 주기적 로테이션 요구 |
| **인사 변경** | 직원 퇴사; 공유 시크릿을 로테이션해야 함 |
| **인시던트 대응** | 침해 후 모든 시크릿 로테이션 |

### 6.2 로테이션 전략

```
Strategy 1: Single Secret Rotation
──────────────────────────────────
1. Generate new secret
2. Update secret in Vault/Secrets Manager
3. Restart application to pick up new secret
4. Old secret is immediately invalid
Problem: Downtime during step 3

Strategy 2: Dual Secret (Overlapping) Rotation
──────────────────────────────────────────────
1. Generate new secret (secret-B)
2. Configure target system to accept BOTH old (secret-A) AND new (secret-B)
3. Update application to use secret-B
4. Verify application works with secret-B
5. Revoke secret-A
Benefit: Zero downtime

Strategy 3: Dynamic Secrets (Vault)
────────────────────────────────────
1. Application requests credentials from Vault
2. Vault generates unique, short-lived credentials (TTL: 1 hour)
3. Application uses credentials until they expire
4. Application requests new credentials before expiry
5. Vault automatically revokes expired credentials
Benefit: No rotation needed — credentials are always fresh
```

### 6.3 AWS Secrets Manager 로테이션 Lambda

```python
# rotation_lambda.py — Multi-user rotation for RDS PostgreSQL

import boto3
import json
import psycopg2

def lambda_handler(event, context):
    secret_arn = event["SecretId"]
    step = event["Step"]
    token = event["ClientRequestToken"]
    client = boto3.client("secretsmanager")

    if step == "createSecret":
        # Generate new password
        new_password = client.get_random_password(
            PasswordLength=32,
            ExcludeCharacters="/@\"'\\"
        )["RandomPassword"]

        # Stage the new secret
        current = json.loads(
            client.get_secret_value(SecretId=secret_arn)["SecretString"]
        )
        current["password"] = new_password
        client.put_secret_value(
            SecretId=secret_arn,
            ClientRequestToken=token,
            SecretString=json.dumps(current),
            VersionStages=["AWSPENDING"]
        )

    elif step == "setSecret":
        # Update the password in the database
        pending = json.loads(
            client.get_secret_value(
                SecretId=secret_arn, VersionStage="AWSPENDING"
            )["SecretString"]
        )
        conn = psycopg2.connect(
            host=pending["host"],
            dbname=pending["dbname"],
            user="admin_user",  # Admin user for ALTER ROLE
            password=get_admin_password()
        )
        with conn.cursor() as cur:
            cur.execute(
                f"ALTER ROLE {pending['username']} WITH PASSWORD %s",
                (pending["password"],)
            )
        conn.commit()
        conn.close()

    elif step == "testSecret":
        # Verify the new credentials work
        pending = json.loads(
            client.get_secret_value(
                SecretId=secret_arn, VersionStage="AWSPENDING"
            )["SecretString"]
        )
        conn = psycopg2.connect(
            host=pending["host"],
            dbname=pending["dbname"],
            user=pending["username"],
            password=pending["password"]
        )
        conn.close()

    elif step == "finishSecret":
        # Promote AWSPENDING to AWSCURRENT
        client.update_secret_version_stage(
            SecretId=secret_arn,
            VersionStage="AWSCURRENT",
            MoveToVersionId=token,
            RemoveFromVersionId=get_current_version(client, secret_arn)
        )
```

---

## 7. 모범 사례

### 7.1 Secret Management 체크리스트

| 사례 | 설명 |
|------|------|
| **시크릿을 Git에 커밋하지 않기** | `.gitignore`, pre-commit hook, `git-secrets`를 사용하여 우발적 커밋 방지 |
| **가능하면 동적 시크릿 사용** | 단기 자격 증명으로 로테이션 부담 제거 |
| **일정에 따른 로테이션** | 비밀번호 90일, API 키 365일, 고위험 시크릿 30일 |
| **모든 접근 감사** | Vault의 감사 로깅, AWS의 CloudTrail, GCP의 Cloud Audit Logs 활성화 |
| **최소 권한 원칙** | 각 애플리케이션이 자체 시크릿에만 접근 |
| **환경 분리** | dev, staging, production은 다른 시크릿을 가져야 함 |
| **저장 및 전송 시 암호화** | 통신에 TLS, 저장에 AES-256 |
| **침해 대응 계획 보유** | 비상시 모든 시크릿을 로테이션하는 방법 문서화 |

### 7.2 Git Secret 방지

```bash
# Install git-secrets
brew install git-secrets

# Add AWS secret patterns
git secrets --register-aws

# Add custom patterns
git secrets --add 'password\s*=\s*.+'
git secrets --add 'PRIVATE KEY'

# Install pre-commit hook (prevents pushing secrets)
git secrets --install

# Scan existing history
git secrets --scan-history
```

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
```

---

## 8. 도구 비교

| 기능 | Vault | AWS Secrets Manager | SOPS | Sealed Secrets |
|------|-------|--------------------|----- |----------------|
| **유형** | 자체 호스팅 또는 클라우드 | 관리형 서비스 | 암호화 도구 | K8s 컨트롤러 |
| **동적 시크릿** | 예 | 제한적 (RDS 로테이션) | 아니오 | 아니오 |
| **PKI/인증서** | 예 (PKI 엔진) | ACM (별도 서비스) | 아니오 | 아니오 |
| **Transit 암호화** | 예 (encrypt-as-a-service) | KMS (별도 서비스) | 아니오 | 아니오 |
| **멀티 클라우드** | 예 | AWS만 | 예 | Kubernetes만 |
| **GitOps 호환** | ESO/CSI driver를 통해 | ESO를 통해 | 예 (ksops) | 예 (네이티브) |
| **복잡도** | 높음 (클러스터 운영) | 낮음 (관리형) | 낮음 (CLI 도구) | 중간 |
| **비용** | 자체 호스팅: 인프라; HCP Vault: $$ | $0.40/시크릿/월 | 무료 | 무료 |

---

## 9. 다음 단계

- [16_Chaos_Engineering.md](./16_Chaos_Engineering.md) - 장애 조건에서 시크릿 관리 테스트
- [14_GitOps.md](./14_GitOps.md) - GitOps 워크플로우에 시크릿 통합

---

## 연습 문제

### 연습 문제 1: Secret 아키텍처 설계

다음 요구사항을 가진 회사를 위한 시크릿 관리 아키텍처를 설계하십시오:
- Kubernetes에서 실행되는 20개 마이크로서비스
- AWS가 기본 클라우드 제공자
- GitOps를 지원해야 함 (모든 구성이 Git에)
- 데이터베이스 자격 증명은 30일마다 로테이션되어야 함
- 모든 시크릿 접근에 대한 감사 추적이 있어야 함

<details>
<summary>정답 보기</summary>

**권장 아키텍처: Vault + External Secrets Operator + SOPS**

```
┌─────────────────────────────────────────────────────────────┐
│                 Secrets Architecture                         │
│                                                              │
│  GitOps Repo (SOPS-encrypted)                               │
│  ├── Non-sensitive config: plain YAML                       │
│  └── Sensitive config: SOPS-encrypted (AWS KMS)             │
│       └── Decrypted by ksops during ArgoCD sync             │
│                                                              │
│  HashiCorp Vault (HA cluster)                                │
│  ├── Dynamic database credentials (PostgreSQL, Redis)        │
│  ├── PKI engine for mTLS certificates                       │
│  ├── KV engine for static secrets (API keys)                │
│  └── Audit logging → CloudWatch Logs                        │
│                                                              │
│  External Secrets Operator (ESO)                             │
│  ├── Syncs Vault secrets → Kubernetes Secrets               │
│  ├── RefreshInterval: 5m (for static), 30m (for dynamic)    │
│  └── Creates K8s Secrets consumed by pods as env vars       │
│                                                              │
│  Application Pods                                            │
│  ├── Auth: Kubernetes auth method (service account → Vault)  │
│  ├── Static secrets: from ESO-synced K8s Secrets            │
│  └── Dynamic DB creds: direct Vault API call via SDK        │
└─────────────────────────────────────────────────────────────┘
```

**이 조합을 선택하는 이유:**
1. **Vault**는 동적 시크릿(1시간 TTL의 DB 자격 증명, 자동 생성)을 처리합니다. 수동 30일 로테이션의 필요성을 제거합니다 -- 자격 증명은 항상 새 것입니다.
2. **SOPS**는 Git의 구성 시크릿(API 키, webhook URL)을 암호화합니다. 리뷰어가 값을 보지 않고도 어떤 키가 변경되었는지 확인합니다.
3. **ESO**는 Vault와 Kubernetes를 연결합니다. 애플리케이션이 Vault SDK 의존성 없이 표준 Kubernetes Secret으로 시크릿을 소비합니다.
4. **감사**: Vault 감사 로그가 모든 시크릿 읽기/쓰기를 캡처합니다. SOPS 변경은 Git 커밋을 통해 추적됩니다. ESO 동기화 이벤트는 Kubernetes 이벤트에 있습니다.

**데이터베이스 자격 증명 흐름:**
1. 파드 시작 -> 서비스 계정이 Kubernetes auth를 통해 Vault에 인증
2. Vault가 고유한 PostgreSQL 자격 증명 생성 (TTL: 1시간)
3. 파드가 데이터베이스 접근에 자격 증명 사용
4. TTL 만료 전, 파드가 새 자격 증명 요청 (또는 Vault agent sidecar가 갱신 처리)
5. Vault가 만료된 자격 증명을 자동으로 폐기

</details>

### 연습 문제 2: Vault 정책 설계

결제 처리 시스템에서 다음 세 가지 역할에 대한 Vault 정책을 작성하십시오:

1. **결제 서비스**: 데이터베이스 자격 증명과 암호화 키 읽기
2. **CI 파이프라인**: KV 저장소의 애플리케이션 시크릿 업데이트
3. **보안 팀**: 감사 로그 읽기 및 모든 시크릿 로테이션

<details>
<summary>정답 보기</summary>

**1. 결제 서비스 정책:**
```hcl
# payment-service.hcl
# Read dynamic database credentials
path "database/creds/payment-db-role" {
  capabilities = ["read"]
}

# Read encryption keys (transit engine)
path "transit/encrypt/payment-key" {
  capabilities = ["update"]
}
path "transit/decrypt/payment-key" {
  capabilities = ["update"]
}

# Read static secrets (API keys for payment gateways)
path "secret/data/payment-service/*" {
  capabilities = ["read", "list"]
}

# Renew own token
path "auth/token/renew-self" {
  capabilities = ["update"]
}

# Deny access to other services' secrets
path "secret/data/auth-service/*" {
  capabilities = ["deny"]
}
path "secret/data/order-service/*" {
  capabilities = ["deny"]
}
```

**2. CI 파이프라인 정책:**
```hcl
# ci-pipeline.hcl
# Read and update application secrets in KV
path "secret/data/*/config" {
  capabilities = ["create", "read", "update"]
}

# Cannot delete secrets
path "secret/data/*" {
  capabilities = ["read", "list"]
}

# Cannot access database credentials
path "database/*" {
  capabilities = ["deny"]
}

# Cannot access transit encryption
path "transit/*" {
  capabilities = ["deny"]
}
```

**3. 보안 팀 정책:**
```hcl
# security-team.hcl
# Read all secrets (for audit)
path "secret/data/*" {
  capabilities = ["read", "list"]
}

# Manage secret metadata
path "secret/metadata/*" {
  capabilities = ["read", "list", "delete"]
}

# Read and manage database roles (for rotation)
path "database/roles/*" {
  capabilities = ["read", "list", "create", "update"]
}

# Rotate database root credentials
path "database/rotate-root/*" {
  capabilities = ["update"]
}

# Read audit device configuration
path "sys/audit" {
  capabilities = ["read", "list"]
}

# Manage policies
path "sys/policies/acl/*" {
  capabilities = ["read", "list"]
}

# Revoke leases (incident response)
path "sys/leases/revoke-prefix/*" {
  capabilities = ["update"]
}
```

**핵심 원칙:** 각 정책은 최소 권한을 따릅니다 -- 결제 서비스는 시크릿을 수정할 수 없고, CI 파이프라인은 데이터베이스 자격 증명에 접근할 수 없으며, 보안 팀도 정책을 수정할 수 없습니다 (그것은 admin/root 토큰이 필요합니다).

</details>

### 연습 문제 3: 인시던트 대응

개발자가 실수로 AWS 접근 키를 공개 GitHub 레포지토리에 커밋했습니다. 단계별 인시던트 대응 프로세스를 설명하십시오.

<details>
<summary>정답 보기</summary>

**즉각적인 조치 (수 분 이내):**

1. **노출된 키를 즉시 폐기**:
   ```bash
   aws iam delete-access-key --user-name <username> --access-key-id <exposed-key-id>
   ```
   조사를 기다리지 마십시오 -- 먼저 폐기하고, 나중에 조사합니다.

2. **영향받는 서비스를 위한 새 키 생성**:
   ```bash
   aws iam create-access-key --user-name <username>
   ```
   시크릿 관리 시스템(Vault/Secrets Manager)에서 새 키를 업데이트합니다.

3. **Git에서 시크릿 제거**:
   ```bash
   # Remove from current branch
   git rm --cached <file-with-secret>
   git commit -m "security: remove exposed credentials"
   git push

   # Remove from Git history using BFG Repo-Cleaner
   bfg --replace-text passwords.txt repo.git
   git reflog expire --expire=now --all
   git gc --prune=now --aggressive
   git push --force
   ```
   참고: 시크릿은 이력에서 제거된 후에도 캐시되었거나, 포크되었거나, 스크래핑되었을 수 있으므로 여전히 침해된 것으로 간주해야 합니다.

**조사 (수 시간 이내):**

4. **CloudTrail에서 무단 사용 확인**:
   ```bash
   aws cloudtrail lookup-events \
     --lookup-attributes AttributeKey=AccessKeyId,AttributeValue=<exposed-key-id> \
     --start-time 2024-01-01T00:00:00Z
   ```
   예상치 못한 API 호출(IAM 변경, 데이터 접근, 리소스 생성)을 확인합니다.

5. **영향 범위 평가**:
   - 키가 어떤 권한을 가졌는지? (IAM 정책 확인)
   - 어떤 리소스에 접근할 수 있었는지?
   - 리소스가 생성되었는지 (크립토 마이너, 데이터 유출 엔드포인트)?

6. **이해관계자에게 통지**: 보안 팀, 영향받는 서비스 소유자, 경영진 (데이터 유출인 경우).

**복구 (수일 이내):**

7. 침해된 키가 접근할 수 있었던 **모든 시크릿을 로테이션** (키 자체만이 아님).

8. **예방 제어 설치**:
   ```bash
   # Install git-secrets hook
   git secrets --install
   git secrets --register-aws

   # Add to CI pipeline
   # GitHub: Enable secret scanning alerts
   # Pre-commit: detect-secrets hook
   ```

9. **인시던트 후 검토**: 타임라인, 근본 원인, 예방 조치를 문서화합니다.

**예방 체크리스트:**
- [ ] 모든 개발자 머신에 `git-secrets` pre-commit hook 설치
- [ ] 모든 레포지토리에 GitHub secret scanning 활성화
- [ ] CI 파이프라인에 시크릿 스캐닝 단계 포함
- [ ] 모든 AWS 자격 증명이 단기 STS 토큰 사용 (정적 IAM 키가 아닌)
- [ ] 애플리케이션이 접근 키 대신 IAM 역할(EC2 인스턴스 프로필, EKS IRSA) 사용

</details>

### 연습 문제 4: GitOps Secrets 전략

팀이 GitOps에 ArgoCD를 사용합니다. 다음 시나리오에 대해 SOPS와 Sealed Secrets를 비교하고 하나를 추천하십시오:
- 서로 다른 AWS 리전의 3개 Kubernetes 클러스터 (dev, staging, production)
- 20개 마이크로서비스에 걸쳐 50개의 시크릿
- 플랫폼 팀이 모든 클러스터를 관리
- 감사 가능해야 함 (누가 어떤 시크릿을 언제 변경했는지)

<details>
<summary>정답 보기</summary>

**추천: AWS KMS를 사용한 SOPS**

**이 시나리오에서의 비교:**

| 기준 | SOPS + AWS KMS | Sealed Secrets |
|------|---------------|---------------|
| **멀티 클러스터** | 단일 KMS 키가 모든 클러스터에서 암호화 | 각 클러스터가 자체 키 쌍; 클러스터별로 시크릿 재암호화 필요 |
| **감사 추적** | Git 커밋이 누가 무엇을 변경했는지 표시; KMS CloudTrail이 복호화 이벤트 표시 | 변경에 대한 Git 커밋; 복호화 감사 없음 (컨트롤러가 내부적으로 복호화) |
| **키 로테이션** | KMS가 키 로테이션을 자동으로 처리 | 컨트롤러 키 쌍을 수동으로 백업 및 로테이션해야 함 |
| **재해 복구** | 클러스터 재생성 후 ArgoCD를 Git에 연결; KMS 키는 AWS가 관리 | 컨트롤러의 개인 키를 복원해야 함, 또는 모든 시크릿 손실 |
| **50개 시크릿 x 3개 클러스터** | 동일 암호화 파일이 모든 클러스터에서 작동 (동일 KMS 키인 경우) | 각 시크릿에 대해 `kubeseal`을 3번 실행해야 함 (클러스터 공개 키당 1번) |
| **개발자 경험** | `sops secrets.yaml`로 편집 (편집기에서 복호화 -> 저장 시 재암호화) | `kubeseal < secret.yaml > sealed.yaml` (in-place 편집 불가) |

**이 시나리오에서 SOPS가 우수한 이유:**

1. **멀티 클러스터 단순성**: 하나의 AWS KMS 키(또는 리전당 하나)로 동일한 SOPS 암호화 파일이 세 클러스터 모두에서 작동합니다. Sealed Secrets는 다른 키로 암호화된 각 시크릿의 3개 복사본을 유지해야 합니다.

2. **감사**: KMS CloudTrail이 호출자의 IAM ID, 타임스탬프, 소스 IP로 모든 복호화 이벤트를 기록합니다. Git 이력과 결합하면 완전한 감사 추적이 됩니다. Sealed Secrets에는 동등한 복호화 감사가 없습니다.

3. **재해 복구**: 클러스터가 파괴되면 Git의 SOPS 암호화 시크릿을 KMS 접근 권한이 있는 모든 클러스터가 복호화할 수 있습니다. Sealed Secrets는 컨트롤러의 개인 키를 백업하고 복원해야 하며 -- 분실하면 모든 시크릿을 잃습니다.

4. **규모**: 3개 클러스터의 20개 서비스에 걸친 50개 시크릿 = 관리할 150개 Sealed Secret 리소스 vs 어디서든 작동하는 50개 SOPS 파일.

**구현:**
```yaml
# .sops.yaml in gitops-repo root
creation_rules:
  - path_regex: overlays/production/.*
    kms: arn:aws:kms:us-east-1:123456789012:key/prod-key
  - path_regex: overlays/staging/.*
    kms: arn:aws:kms:us-west-2:123456789012:key/staging-key
  - path_regex: overlays/dev/.*
    kms: arn:aws:kms:us-west-2:123456789012:key/dev-key
```

ArgoCD는 동기화 중 SOPS 파일을 복호화하기 위해 `ksops` 플러그인으로 구성됩니다.

</details>

---

## 참고 자료

- [HashiCorp Vault Documentation](https://developer.hashicorp.com/vault/docs)
- [AWS Secrets Manager Documentation](https://docs.aws.amazon.com/secretsmanager/)
- [SOPS Documentation](https://github.com/getsops/sops)
- [Sealed Secrets Documentation](https://github.com/bitnami-labs/sealed-secrets)
- [External Secrets Operator](https://external-secrets.io/)
- [git-secrets](https://github.com/awslabs/git-secrets)
