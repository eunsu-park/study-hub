# Secrets Management

**Previous**: [GitOps](./14_GitOps.md) | **Next**: [Chaos Engineering](./16_Chaos_Engineering.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why secrets require specialized management and the risks of storing them in plain text or Git
2. Deploy and configure HashiCorp Vault with secret engines, policies, and authentication methods
3. Use AWS Secrets Manager and other cloud-native secret stores for managed secret lifecycle
4. Encrypt secrets for Git storage using SOPS and Sealed Secrets for GitOps compatibility
5. Implement dynamic secrets that are generated on demand and automatically expired
6. Design a secret rotation strategy that eliminates long-lived credentials

---

Secrets -- API keys, database passwords, TLS certificates, encryption keys, service account tokens -- are the most sensitive data in any system. A leaked secret can lead to data breaches, financial loss, and regulatory penalties. Despite this, secrets are routinely hardcoded in source code, committed to Git, passed in environment variables, and shared in Slack messages. This lesson covers the tools and practices that treat secrets as first-class infrastructure: centralized storage, access control, audit logging, dynamic generation, and automatic rotation.

> **Analogy -- Bank Vault vs Under the Mattress**: Storing secrets in environment variables or config files is like keeping cash under your mattress -- anyone who enters your house can take it, there is no record of who accessed it, and you have no way to change the locks. A secrets management system (Vault, AWS Secrets Manager) is like a bank vault: it requires authentication to access, logs every access, limits who can access what, and can change the combination (rotate) without disrupting operations.

## 1. Why Secrets Management

### 1.1 Common Anti-Patterns

| Anti-Pattern | Risk |
|-------------|------|
| **Hardcoded in source code** | Secrets in Git history forever, even after deletion |
| **Environment variables** | Visible in process listings, container inspect, crash dumps |
| **Shared `.env` files** | No access control, no audit trail, copy-pasted across teams |
| **Slack/Email sharing** | Stored in third-party systems, searchable, no expiration |
| **Same password everywhere** | One compromise exposes all systems |
| **Never rotated** | Leaked credentials remain valid indefinitely |

### 1.2 Secrets Management Requirements

| Requirement | Description |
|-------------|-------------|
| **Encryption at rest** | Secrets encrypted when stored |
| **Encryption in transit** | TLS for all secret access |
| **Access control** | Fine-grained policies (who can access which secrets) |
| **Audit logging** | Every access and modification is logged |
| **Rotation** | Secrets can be changed without downtime |
| **Dynamic secrets** | Short-lived credentials generated on demand |
| **Revocation** | Ability to immediately invalidate compromised secrets |
| **Versioning** | Previous versions retained for rollback |

---

## 2. HashiCorp Vault

### 2.1 Architecture

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

### 2.2 Installation and Initialization

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

### 2.4 Policies

Policies define what a token can access:

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

### 2.5 Authentication Methods

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

### 2.6 Dynamic Secrets (Database)

Dynamic secrets are generated on demand and automatically revoked after a TTL:

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

**Benefits of dynamic secrets:**
- No long-lived credentials to steal
- Each application instance gets unique credentials (attributable access)
- Automatic revocation after TTL (no stale credentials)
- Instant revocation during incidents

---

## 3. AWS Secrets Manager

### 3.1 Core Features

| Feature | Description |
|---------|-------------|
| **Storage** | Encrypted storage (AES-256 via AWS KMS) |
| **Rotation** | Built-in Lambda-based rotation for RDS, Redshift, DocumentDB |
| **Access control** | IAM policies and resource policies |
| **Audit** | CloudTrail logging for all API calls |
| **Versioning** | `AWSCURRENT` and `AWSPREVIOUS` staging labels |
| **Cross-account** | Share secrets across AWS accounts via resource policies |

### 3.2 CLI Operations

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

### 3.3 Application Integration (Python)

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

### 3.4 Kubernetes Integration (External Secrets Operator)

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

### 4.1 Why SOPS

SOPS encrypts secret values in YAML/JSON files while keeping keys and structure in plain text. This makes encrypted secrets compatible with Git and code review -- reviewers can see which fields changed without seeing the values.

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

### 4.2 Setup and Usage

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

### 4.3 SOPS with Kustomize (GitOps)

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

### 5.1 How Sealed Secrets Work

Sealed Secrets encrypts Kubernetes Secrets with a public key. Only the Sealed Secrets controller in the cluster (which holds the private key) can decrypt them. Encrypted secrets can be safely committed to Git.

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

### 5.2 Usage

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

| Aspect | SOPS | Sealed Secrets |
|--------|------|---------------|
| **Encryption scope** | Any file (YAML, JSON, .env) | Kubernetes Secrets only |
| **Key management** | External KMS (AWS, GCP, age) | Controller-managed key pair |
| **Readability** | Keys visible, values encrypted | Entire data block encrypted |
| **Multi-cluster** | Same KMS key works anywhere | One key pair per cluster |
| **GitOps compatible** | Yes (with ksops plugin) | Yes (native Kubernetes resource) |
| **Best for** | Multi-cloud, non-Kubernetes secrets | Kubernetes-only environments |

---

## 6. Secret Rotation

### 6.1 Why Rotate Secrets

| Reason | Explanation |
|--------|-------------|
| **Limit exposure window** | If a secret is leaked, rotation limits how long it is valid |
| **Compliance** | PCI-DSS, SOC 2, HIPAA require periodic rotation |
| **Personnel changes** | Employees leave; shared secrets must be rotated |
| **Incident response** | Rotate all secrets after a breach |

### 6.2 Rotation Strategies

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

### 6.3 AWS Secrets Manager Rotation Lambda

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

## 7. Best Practices

### 7.1 Secret Management Checklist

| Practice | Description |
|----------|-------------|
| **Never commit secrets to Git** | Use `.gitignore`, pre-commit hooks, and `git-secrets` to prevent accidental commits |
| **Use dynamic secrets when possible** | Short-lived credentials eliminate rotation burden |
| **Rotate on a schedule** | 90 days for passwords, 365 days for API keys, 30 days for high-risk secrets |
| **Audit all access** | Enable audit logging in Vault, CloudTrail for AWS, Cloud Audit Logs for GCP |
| **Principle of least privilege** | Each application gets access only to its own secrets |
| **Separate environments** | Dev, staging, and production must have different secrets |
| **Encrypt at rest and in transit** | TLS for communication, AES-256 for storage |
| **Have a breach response plan** | Document how to rotate all secrets in an emergency |

### 7.2 Git Secret Prevention

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

## 8. Tool Comparison

| Feature | Vault | AWS Secrets Manager | SOPS | Sealed Secrets |
|---------|-------|--------------------|----- |----------------|
| **Type** | Self-hosted or cloud | Managed service | Encryption tool | K8s controller |
| **Dynamic secrets** | Yes | Limited (RDS rotation) | No | No |
| **PKI/certificates** | Yes (PKI engine) | ACM (separate service) | No | No |
| **Transit encryption** | Yes (encrypt-as-a-service) | KMS (separate service) | No | No |
| **Multi-cloud** | Yes | AWS only | Yes | Kubernetes only |
| **GitOps compatible** | Via ESO/CSI driver | Via ESO | Yes (ksops) | Yes (native) |
| **Complexity** | High (operate cluster) | Low (managed) | Low (CLI tool) | Medium |
| **Cost** | Self-hosted: infrastructure; HCP Vault: $$ | $0.40/secret/month | Free | Free |

---

## 9. Next Steps

- [16_Chaos_Engineering.md](./16_Chaos_Engineering.md) - Testing secret management under failure conditions
- [14_GitOps.md](./14_GitOps.md) - Integrating secrets into GitOps workflows

---

## Exercises

### Exercise 1: Secret Architecture Design

Design a secrets management architecture for a company with the following requirements:
- 20 microservices running on Kubernetes
- AWS as the primary cloud provider
- Must support GitOps (all configuration in Git)
- Database credentials must be rotated every 30 days
- Must have an audit trail for all secret access

<details>
<summary>Show Answer</summary>

**Recommended architecture: Vault + External Secrets Operator + SOPS**

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

**Why this combination:**
1. **Vault** handles dynamic secrets (DB credentials with 1-hour TTL, auto-generated). Eliminates the need for manual 30-day rotation -- credentials are always fresh.
2. **SOPS** encrypts configuration secrets in Git (API keys, webhook URLs). Reviewers see which keys changed without seeing values.
3. **ESO** bridges Vault and Kubernetes. Applications consume secrets as standard Kubernetes Secrets without Vault SDK dependency.
4. **Audit**: Vault audit log captures every secret read/write. SOPS changes are tracked via Git commits. ESO sync events are in Kubernetes events.

**Database credential flow:**
1. Pod starts → service account authenticates to Vault via Kubernetes auth
2. Vault generates unique PostgreSQL credentials (TTL: 1 hour)
3. Pod uses credentials for database access
4. Before TTL expires, pod requests new credentials (or Vault agent sidecar handles renewal)
5. Vault automatically revokes expired credentials

</details>

### Exercise 2: Vault Policy Design

Write Vault policies for the following three roles in a payment processing system:

1. **Payment service**: Read database credentials and encryption keys
2. **CI pipeline**: Update application secrets in the KV store
3. **Security team**: Read audit logs and rotate all secrets

<details>
<summary>Show Answer</summary>

**1. Payment service policy:**
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

**2. CI pipeline policy:**
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

**3. Security team policy:**
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

**Key principle:** Each policy follows least privilege -- the payment service cannot modify secrets, the CI pipeline cannot access database credentials, and even the security team cannot modify policies (that requires the admin/root token).

</details>

### Exercise 3: Incident Response

A developer accidentally commits an AWS access key to a public GitHub repository. Describe the step-by-step incident response process.

<details>
<summary>Show Answer</summary>

**Immediate actions (within minutes):**

1. **Revoke the exposed key immediately**:
   ```bash
   aws iam delete-access-key --user-name <username> --access-key-id <exposed-key-id>
   ```
   Do not wait for investigation -- revoke first, investigate later.

2. **Generate a new key for the affected service**:
   ```bash
   aws iam create-access-key --user-name <username>
   ```
   Update the new key in the secrets management system (Vault/Secrets Manager).

3. **Remove the secret from Git**:
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
   Note: The secret must still be considered compromised even after removal from history, because it may have been cached, forked, or scraped.

**Investigation (within hours):**

4. **Check CloudTrail for unauthorized usage**:
   ```bash
   aws cloudtrail lookup-events \
     --lookup-attributes AttributeKey=AccessKeyId,AttributeValue=<exposed-key-id> \
     --start-time 2024-01-01T00:00:00Z
   ```
   Look for unexpected API calls (IAM changes, data access, resource creation).

5. **Assess blast radius**:
   - What permissions did the key have? (Check IAM policy)
   - What resources could have been accessed?
   - Were any resources created (crypto miners, data exfiltration endpoints)?

6. **Notify stakeholders**: Security team, affected service owners, management (if data breach).

**Remediation (within days):**

7. **Rotate ALL secrets** that the compromised key could access (not just the key itself).

8. **Install preventive controls**:
   ```bash
   # Install git-secrets hook
   git secrets --install
   git secrets --register-aws

   # Add to CI pipeline
   # GitHub: Enable secret scanning alerts
   # Pre-commit: detect-secrets hook
   ```

9. **Post-incident review**: Document timeline, root cause, and preventive measures.

**Prevention checklist:**
- [ ] `git-secrets` pre-commit hook installed on all developer machines
- [ ] GitHub secret scanning enabled on all repositories
- [ ] CI pipeline includes secret scanning step
- [ ] All AWS credentials use short-lived STS tokens (not static IAM keys)
- [ ] Applications use IAM roles (EC2 instance profile, EKS IRSA) instead of access keys

</details>

### Exercise 4: GitOps Secrets Strategy

Your team uses ArgoCD for GitOps. Compare SOPS and Sealed Secrets for the following scenario and recommend one:
- 3 Kubernetes clusters (dev, staging, production) in different AWS regions
- 50 secrets across 20 microservices
- Platform team manages all clusters
- Must be auditable (who changed which secret, when)

<details>
<summary>Show Answer</summary>

**Recommendation: SOPS with AWS KMS**

**Comparison for this scenario:**

| Criterion | SOPS + AWS KMS | Sealed Secrets |
|-----------|---------------|---------------|
| **Multi-cluster** | Single KMS key encrypts for all clusters | Each cluster has its own key pair; secrets must be re-encrypted per cluster |
| **Audit trail** | Git commits show who changed what; KMS CloudTrail shows decryption events | Git commits for changes; no decryption audit (controller decrypts internally) |
| **Key rotation** | KMS handles key rotation automatically | Must back up and rotate controller key pair manually |
| **Disaster recovery** | Re-create cluster, point ArgoCD at Git; KMS key is managed by AWS | Must restore the controller's private key, or all secrets are lost |
| **50 secrets x 3 clusters** | Same encrypted file works in all clusters (if same KMS key) | Must run `kubeseal` 3 times (once per cluster's public key) for each secret |
| **Developer experience** | `sops secrets.yaml` to edit (decrypts in editor, re-encrypts on save) | `kubeseal < secret.yaml > sealed.yaml` (no in-place editing) |

**Why SOPS wins for this scenario:**

1. **Multi-cluster simplicity**: With one AWS KMS key (or one per region), the same SOPS-encrypted file works across all three clusters. Sealed Secrets would require maintaining 3 copies of each secret, encrypted with different keys.

2. **Audit**: KMS CloudTrail logs every decryption event with the caller's IAM identity, timestamp, and source IP. Combined with Git history, you have a complete audit trail. Sealed Secrets has no equivalent decryption audit.

3. **Disaster recovery**: If a cluster is destroyed, SOPS-encrypted secrets in Git can be decrypted by any cluster with KMS access. Sealed Secrets requires the controller's private key to be backed up and restored -- losing it means losing all secrets.

4. **Scale**: 50 secrets across 20 services in 3 clusters = 150 Sealed Secret resources to manage vs 50 SOPS files that work everywhere.

**Implementation:**
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

ArgoCD is configured with the `ksops` plugin to decrypt SOPS files during sync.

</details>

---

## References

- [HashiCorp Vault Documentation](https://developer.hashicorp.com/vault/docs)
- [AWS Secrets Manager Documentation](https://docs.aws.amazon.com/secretsmanager/)
- [SOPS Documentation](https://github.com/getsops/sops)
- [Sealed Secrets Documentation](https://github.com/bitnami-labs/sealed-secrets)
- [External Secrets Operator](https://external-secrets.io/)
- [git-secrets](https://github.com/awslabs/git-secrets)
