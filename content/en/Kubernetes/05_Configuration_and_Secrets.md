# 05. Configuration and Secrets

**Previous**: [Storage and Persistence](./04_Storage_and_Persistence.md) | **Next**: [RBAC and Security](./06_RBAC_and_Security.md)

## Learning Objectives
- Create and manage ConfigMaps using multiple approaches (literal, file, directory)
- Understand Secret types, encoding, and secure mounting patterns
- Implement immutable ConfigMaps and Secrets for performance and safety
- Integrate external secret management systems (External Secrets Operator, Sealed Secrets, Vault)
- Apply configuration best practices for multi-environment deployments

---

Configuration management is a critical operational concern in Kubernetes.
Applications need database URLs, feature flags, API keys, and TLS certificates—all
of which must be managed separately from container images. Kubernetes provides
ConfigMaps for non-sensitive data and Secrets for sensitive data, but production
environments often require external secret management systems for proper security.

## Table of Contents
1. [ConfigMaps](#1-configmaps)
2. [Secrets](#2-secrets)
3. [Immutable ConfigMaps and Secrets](#3-immutable-configmaps-and-secrets)
4. [External Secrets Operator](#4-external-secrets-operator)
5. [Sealed Secrets](#5-sealed-secrets)
6. [HashiCorp Vault Integration](#6-hashicorp-vault-integration)
7. [Secret Rotation Patterns](#7-secret-rotation-patterns)
8. [Configuration Best Practices](#8-configuration-best-practices)
9. [Environment-Specific Configuration](#9-environment-specific-configuration)
10. [Encrypting Secrets at Rest (EncryptionConfiguration)](#10-encrypting-secrets-at-rest-encryptionconfiguration)
11. [Exercises](#exercises)

---

## 1. ConfigMaps

A ConfigMap stores non-confidential configuration data as key-value pairs.
ConfigMaps decouple configuration from container images, making applications
portable across environments.

### 1.1 Creating ConfigMaps

**From literal values:**

```bash
kubectl create configmap app-config \
  --from-literal=DATABASE_HOST=postgres.default.svc \
  --from-literal=DATABASE_PORT=5432 \
  --from-literal=LOG_LEVEL=info \
  --from-literal=MAX_CONNECTIONS=100
```

**From a file:**

```bash
# Create a config file
cat > /tmp/app.properties <<EOF
database.host=postgres.default.svc
database.port=5432
log.level=info
max.connections=100
EOF

kubectl create configmap app-config --from-file=/tmp/app.properties
# Key = filename (app.properties), Value = file contents
```

**From a directory:**

```bash
# All files in the directory become key-value pairs
mkdir -p /tmp/config
echo "postgres.default.svc" > /tmp/config/database_host
echo "5432" > /tmp/config/database_port

kubectl create configmap app-config --from-file=/tmp/config/
# Keys: database_host, database_port
```

**From YAML manifest:**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: default
data:
  # Simple key-value pairs
  DATABASE_HOST: "postgres.default.svc"
  DATABASE_PORT: "5432"
  LOG_LEVEL: "info"
  MAX_CONNECTIONS: "100"

  # Multi-line configuration file
  nginx.conf: |
    server {
        listen 80;
        server_name localhost;
        location / {
            proxy_pass http://backend:8080;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        location /health {
            return 200 'OK';
            add_header Content-Type text/plain;
        }
    }

  # Application configuration in YAML format
  config.yaml: |
    server:
      port: 8080
      host: 0.0.0.0
    database:
      host: postgres.default.svc
      port: 5432
      pool_size: 20
    logging:
      level: info
      format: json
```

### 1.2 Using ConfigMaps as Environment Variables

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-env-demo
spec:
  containers:
    - name: app
      image: my-app:v1.0

      # Individual keys as env vars
      env:
        - name: DB_HOST
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: DATABASE_HOST
        - name: DB_PORT
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: DATABASE_PORT
              optional: true          # Pod starts even if key is missing

      # All keys as env vars (prefix optional)
      envFrom:
        - configMapRef:
            name: app-config
          prefix: CFG_              # CFG_DATABASE_HOST, CFG_LOG_LEVEL, etc.
```

### 1.3 Mounting ConfigMaps as Volumes

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-volume-demo
spec:
  containers:
    - name: nginx
      image: nginx:1.25
      volumeMounts:
        # Mount entire ConfigMap as a directory
        - name: nginx-config
          mountPath: /etc/nginx/conf.d
          readOnly: true

        # Mount specific keys
        - name: app-config
          mountPath: /etc/app/config.yaml
          subPath: config.yaml       # Mount single key as file (not directory)
          readOnly: true

  volumes:
    - name: nginx-config
      configMap:
        name: app-config
        items:
          - key: nginx.conf
            path: default.conf       # Rename the file
            mode: 0644               # File permissions
    - name: app-config
      configMap:
        name: app-config
        defaultMode: 0644
```

### 1.4 ConfigMap Auto-Updates

When a ConfigMap is mounted as a volume (not `subPath`), updates propagate
automatically to the pod. The kubelet checks for changes periodically.

```bash
# Update a ConfigMap
kubectl edit configmap app-config
# Or:
kubectl create configmap app-config --from-literal=LOG_LEVEL=debug --dry-run=client -o yaml | kubectl apply -f -

# Changes are reflected in mounted volumes within ~1 minute
# (configurable via kubelet's --sync-frequency flag, default: 1m)

# NOTE: Environment variables do NOT update. The pod must be restarted.
# NOTE: subPath mounts do NOT update. Only full volume mounts auto-update.
```

### 1.5 ConfigMap Size Limits

- Maximum size: **1 MiB** (1,048,576 bytes) per ConfigMap
- For larger configuration, consider mounting from a PersistentVolume or
  using init containers to fetch configuration

---

## 2. Secrets

Secrets store sensitive data such as passwords, tokens, and TLS certificates.
They are similar to ConfigMaps but with additional security considerations.

### 2.1 Secret Types

| Type | Description | Example |
|------|-------------|---------|
| `Opaque` | Arbitrary user-defined data (default) | API keys, passwords |
| `kubernetes.io/tls` | TLS certificate and key | HTTPS certificates |
| `kubernetes.io/dockerconfigjson` | Docker registry credentials | Image pull secrets |
| `kubernetes.io/basic-auth` | Basic authentication | Username/password |
| `kubernetes.io/ssh-auth` | SSH private key | Git SSH keys |
| `kubernetes.io/service-account-token` | ServiceAccount token | Auto-generated |

### 2.2 Creating Secrets

**Opaque secret:**

```bash
kubectl create secret generic db-credentials \
  --from-literal=username=admin \
  --from-literal=password='S3cur3P@ss!'
```

**TLS secret:**

```bash
kubectl create secret tls my-tls \
  --cert=./tls.crt \
  --key=./tls.key
```

**Docker registry secret:**

```bash
kubectl create secret docker-registry regcred \
  --docker-server=ghcr.io \
  --docker-username=myuser \
  --docker-password=ghp_xxxxxxxxxxxx \
  --docker-email=user@example.com
```

**From YAML manifest:**

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
type: Opaque
data:
  # Values MUST be base64-encoded
  username: YWRtaW4=              # echo -n "admin" | base64
  password: UzNjdXIzUEBzcyE=    # echo -n "S3cur3P@ss!" | base64
---
# Alternative: use stringData for plain text (auto-encoded)
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials-plain
type: Opaque
stringData:
  username: admin
  password: "S3cur3P@ss!"
  connection-string: "postgresql://admin:S3cur3P@ss!@postgres:5432/mydb"
```

> **Important**: base64 is encoding, NOT encryption. Anyone with access to the
> Secret object can decode the values. Secrets are only marginally more secure
> than ConfigMaps by default.

### 2.3 Using Secrets

**As environment variables:**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-with-secrets
spec:
  containers:
    - name: app
      image: my-app:v1.0
      env:
        - name: DB_USERNAME
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: username
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
      # Or load all keys
      envFrom:
        - secretRef:
            name: db-credentials
            optional: false
```

**As volume mounts:**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-with-tls
spec:
  containers:
    - name: nginx
      image: nginx:1.25
      volumeMounts:
        - name: tls-certs
          mountPath: /etc/nginx/ssl
          readOnly: true
        - name: db-creds
          mountPath: /etc/secrets
          readOnly: true
  volumes:
    - name: tls-certs
      secret:
        secretName: my-tls
        defaultMode: 0400          # Restrictive permissions
    - name: db-creds
      secret:
        secretName: db-credentials
        items:
          - key: username
            path: db-user          # /etc/secrets/db-user
          - key: password
            path: db-pass          # /etc/secrets/db-pass
            mode: 0400
```

### 2.4 Image Pull Secrets

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: private-registry-pod
spec:
  imagePullSecrets:
    - name: regcred              # Docker registry secret
  containers:
    - name: app
      image: ghcr.io/myorg/my-app:v1.0
```

Attach image pull secrets to a ServiceAccount so all pods use them automatically:

```bash
kubectl patch serviceaccount default \
  -p '{"imagePullSecrets": [{"name": "regcred"}]}'
```

### 2.5 Secret Security Considerations

```
Default Kubernetes Secret security:
├── Stored in etcd (base64-encoded, NOT encrypted by default)
├── Transmitted over TLS (API server ↔ kubelet)
├── Accessible to anyone with RBAC access to the Secret
├── Visible in pod spec (kubectl get pod -o yaml shows secretKeyRef names)
└── Mounted as tmpfs (not written to node disk)
```

**Enable etcd encryption at rest:**

```yaml
# /etc/kubernetes/encryption-config.yaml
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
      - secrets
    providers:
      - aescbc:
          keys:
            - name: key1
              secret: <base64-encoded-32-byte-key>
      - identity: {}    # Fallback for reading unencrypted secrets
```

```bash
# Apply encryption config to API server
# Add to kube-apiserver flags:
# --encryption-provider-config=/etc/kubernetes/encryption-config.yaml

# Re-encrypt existing secrets
kubectl get secrets --all-namespaces -o json | kubectl replace -f -
```

---

## 3. Immutable ConfigMaps and Secrets

Immutable ConfigMaps and Secrets cannot be updated after creation. This provides:
- **Performance**: kubelet skips continuous watch polling for immutable objects
- **Safety**: Prevents accidental configuration changes in production
- **Scale**: Reduces API server load (no watches on these objects)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config-v2
immutable: true                  # Cannot be modified after creation
data:
  DATABASE_HOST: "postgres.default.svc"
  LOG_LEVEL: "info"
---
apiVersion: v1
kind: Secret
metadata:
  name: api-key-v3
immutable: true
type: Opaque
stringData:
  api-key: "sk-xxxxxxxxxxxxxxxxxxxx"
```

```bash
# Attempting to update an immutable ConfigMap fails
kubectl edit configmap app-config-v2
# Error: "app-config-v2" is immutable

# To "update", create a new version and update pod references
kubectl create configmap app-config-v3 \
  --from-literal=DATABASE_HOST=postgres.default.svc \
  --from-literal=LOG_LEVEL=debug

# Update the deployment to use the new ConfigMap
kubectl set env deployment/my-app --from=configmap/app-config-v3

# Delete the old ConfigMap
kubectl delete configmap app-config-v2
```

### 3.1 Versioning Pattern

```yaml
# Use version suffixes for immutable configs
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config-20240115-001
  labels:
    app: my-app
    config-version: "20240115-001"
immutable: true
data:
  config.yaml: |
    server:
      port: 8080
    database:
      host: postgres.production.svc
      pool_size: 50
```

---

## 4. External Secrets Operator

The External Secrets Operator (ESO) synchronizes secrets from external secret
management systems (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault,
HashiCorp Vault) into Kubernetes Secrets.

### 4.1 Installation

```bash
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets \
  external-secrets/external-secrets \
  -n external-secrets \
  --create-namespace
```

### 4.2 SecretStore Configuration

```yaml
# ClusterSecretStore: available across all namespaces
apiVersion: external-secrets.io/v1beta1
kind: ClusterSecretStore
metadata:
  name: aws-secrets-manager
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa
            namespace: external-secrets
```

### 4.3 ExternalSecret Definition

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: database-credentials
  namespace: production
spec:
  refreshInterval: 5m              # Sync every 5 minutes
  secretStoreRef:
    name: aws-secrets-manager
    kind: ClusterSecretStore

  target:
    name: db-credentials           # Name of the K8s Secret to create
    creationPolicy: Owner          # ESO owns the Secret lifecycle
    deletionPolicy: Retain         # Keep Secret if ExternalSecret is deleted
    template:
      type: Opaque
      data:
        # Template the secret data
        connection-string: "postgresql://{{ .username }}:{{ .password }}@{{ .host }}:5432/{{ .database }}"

  data:
    - secretKey: username
      remoteRef:
        key: production/database    # AWS Secrets Manager secret name
        property: username          # JSON key within the secret

    - secretKey: password
      remoteRef:
        key: production/database
        property: password

    - secretKey: host
      remoteRef:
        key: production/database
        property: host

    - secretKey: database
      remoteRef:
        key: production/database
        property: dbname
```

```bash
# Check sync status
kubectl get externalsecret database-credentials -n production

# Output:
# NAME                    STORE                  REFRESH INTERVAL   STATUS
# database-credentials    aws-secrets-manager    5m                 SecretSynced

# The created Kubernetes Secret
kubectl get secret db-credentials -n production -o yaml
```

### 4.4 Push Secrets (Bidirectional)

```yaml
# Push a K8s Secret to the external store
apiVersion: external-secrets.io/v1alpha1
kind: PushSecret
metadata:
  name: push-to-aws
spec:
  secretStoreRefs:
    - name: aws-secrets-manager
      kind: ClusterSecretStore
  selector:
    secret:
      name: generated-credentials
  data:
    - match:
        secretKey: api-key
        remoteRef:
          remoteKey: production/api-credentials
          property: key
```

---

## 5. Sealed Secrets

Sealed Secrets by Bitnami enables encrypting secrets that can be safely stored
in Git repositories. Only the controller in the cluster can decrypt them.

### 5.1 Installation

```bash
# Install the controller
helm repo add sealed-secrets https://bitnami-labs.github.io/sealed-secrets
helm install sealed-secrets sealed-secrets/sealed-secrets \
  -n kube-system

# Install the client tool
brew install kubeseal   # macOS
# or download from GitHub releases
```

### 5.2 Creating Sealed Secrets

```bash
# Create a regular secret (dry-run)
kubectl create secret generic db-credentials \
  --from-literal=username=admin \
  --from-literal=password='S3cur3P@ss!' \
  --dry-run=client -o yaml > /tmp/secret.yaml

# Encrypt it with kubeseal
kubeseal \
  --controller-name=sealed-secrets \
  --controller-namespace=kube-system \
  --format=yaml \
  < /tmp/secret.yaml \
  > sealed-secret.yaml

# The sealed secret can be safely committed to Git
cat sealed-secret.yaml
```

The sealed secret looks like:

```yaml
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: db-credentials
  namespace: default
spec:
  encryptedData:
    username: AgBxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...
    password: AgCyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy...
  template:
    metadata:
      name: db-credentials
    type: Opaque
```

```bash
# Apply the sealed secret (controller decrypts it into a regular Secret)
kubectl apply -f sealed-secret.yaml

# Verify the regular Secret was created
kubectl get secret db-credentials
kubectl get secret db-credentials -o jsonpath='{.data.username}' | base64 -d
# admin
```

### 5.3 Scope Modes

| Scope | Description | Key bound to |
|-------|-------------|-------------|
| strict (default) | Secret name + namespace | Exact name and namespace |
| namespace-wide | Any name in namespace | Namespace only |
| cluster-wide | Any name, any namespace | Nothing |

```bash
# Namespace-wide scope
kubeseal --scope namespace-wide < secret.yaml > sealed.yaml

# Cluster-wide scope (least restrictive)
kubeseal --scope cluster-wide < secret.yaml > sealed.yaml
```

### 5.4 Key Rotation

```bash
# The controller automatically rotates its sealing key every 30 days
# Old keys are kept so existing SealedSecrets can still be decrypted

# Re-encrypt all SealedSecrets with the new key
kubeseal --re-encrypt < sealed-secret.yaml > sealed-secret-new.yaml
```

---

## 6. HashiCorp Vault Integration

HashiCorp Vault provides centralized secret management with access control,
audit logging, and dynamic secrets.

### 6.1 Vault Agent Injector

The Vault Agent Injector is a mutating webhook that injects a Vault Agent sidecar
into pods to fetch secrets.

```bash
# Install Vault with the injector
helm repo add hashicorp https://helm.releases.hashicorp.com
helm install vault hashicorp/vault \
  --set "injector.enabled=true" \
  --set "server.dev.enabled=true"    # Dev mode for testing
```

### 6.2 Configuring Vault

```bash
# Enable Kubernetes auth method
kubectl exec -it vault-0 -- vault auth enable kubernetes

# Configure Kubernetes auth
kubectl exec -it vault-0 -- vault write auth/kubernetes/config \
  kubernetes_host="https://kubernetes.default.svc:443"

# Create a policy
kubectl exec -it vault-0 -- vault policy write app-policy - <<EOF
path "secret/data/production/*" {
  capabilities = ["read"]
}
path "database/creds/readonly" {
  capabilities = ["read"]
}
EOF

# Create a role for the application
kubectl exec -it vault-0 -- vault write auth/kubernetes/role/app-role \
  bound_service_account_names=app-sa \
  bound_service_account_namespaces=production \
  policies=app-policy \
  ttl=1h

# Store a secret
kubectl exec -it vault-0 -- vault kv put secret/production/database \
  username=admin \
  password="S3cur3P@ss!" \
  host="postgres.production.svc"
```

### 6.3 Injecting Secrets into Pods

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: app-sa
  namespace: production
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
      annotations:
        # Vault Agent Injector annotations
        vault.hashicorp.com/agent-inject: "true"
        vault.hashicorp.com/role: "app-role"

        # Inject database credentials
        vault.hashicorp.com/agent-inject-secret-db-creds: "secret/data/production/database"
        vault.hashicorp.com/agent-inject-template-db-creds: |
          {{- with secret "secret/data/production/database" -}}
          export DB_HOST="{{ .Data.data.host }}"
          export DB_USER="{{ .Data.data.username }}"
          export DB_PASS="{{ .Data.data.password }}"
          {{- end }}

        # Auto-rotate: re-fetch secrets periodically
        vault.hashicorp.com/agent-inject-command-db-creds: "sh -c 'kill -HUP $(pidof my-app) || true'"
    spec:
      serviceAccountName: app-sa
      containers:
        - name: app
          image: my-app:v1.0
          command:
            - sh
            - -c
            - |
              source /vault/secrets/db-creds
              exec ./my-app
```

### 6.4 Vault CSI Provider

An alternative to the injector—mount secrets as files via CSI:

```yaml
apiVersion: secrets-store.csi.x-k8s.io/v1
kind: SecretProviderClass
metadata:
  name: vault-db-creds
spec:
  provider: vault
  parameters:
    vaultAddress: "http://vault.default:8200"
    roleName: "app-role"
    objects: |
      - objectName: "db-username"
        secretPath: "secret/data/production/database"
        secretKey: "username"
      - objectName: "db-password"
        secretPath: "secret/data/production/database"
        secretKey: "password"
  # Optionally sync to a Kubernetes Secret
  secretObjects:
    - secretName: vault-db-secret
      type: Opaque
      data:
        - objectName: db-username
          key: username
        - objectName: db-password
          key: password
---
apiVersion: v1
kind: Pod
metadata:
  name: app-csi-vault
spec:
  serviceAccountName: app-sa
  containers:
    - name: app
      image: my-app:v1.0
      volumeMounts:
        - name: secrets
          mountPath: /mnt/secrets
          readOnly: true
      env:
        - name: DB_USER
          valueFrom:
            secretKeyRef:
              name: vault-db-secret
              key: username
  volumes:
    - name: secrets
      csi:
        driver: secrets-store.csi.k8s.io
        readOnly: true
        volumeAttributes:
          secretProviderClass: vault-db-creds
```

---

## 7. Secret Rotation Patterns

### 7.1 Dual-Secret Rotation

Support both old and new credentials during rotation:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: api-credentials
  labels:
    rotation-phase: dual       # Track rotation state
type: Opaque
stringData:
  current-key: "new-api-key-v2"
  previous-key: "old-api-key-v1"
```

```go
package main

import (
	"fmt"
	"os"
)

// Application code that supports dual-key rotation
func getAPIKey() string {
	// Try current key first
	currentKey := os.Getenv("CURRENT_API_KEY")
	if currentKey != "" {
		return currentKey
	}
	// Fall back to previous key during rotation
	previousKey := os.Getenv("PREVIOUS_API_KEY")
	if previousKey != "" {
		fmt.Println("WARNING: Using previous API key during rotation")
		return previousKey
	}
	panic("No API key configured")
}
```

### 7.2 Rolling Restart on Secret Change

ConfigMaps and Secrets mounted as volumes auto-update, but environment variables
require a pod restart. Use annotations to trigger rolling restarts:

```bash
# Trigger a rolling restart by updating an annotation
kubectl patch deployment my-app -p \
  "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":{\"secret-version\":\"$(date +%s)\"}}}}}"
```

Or use a hash-based approach:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  template:
    metadata:
      annotations:
        # Compute hash of the secret and store as annotation
        # When secret changes, hash changes, triggering a rollout
        checksum/secret: "sha256:abc123..."
    spec:
      containers:
        - name: app
          image: my-app:v1.0
          envFrom:
            - secretRef:
                name: db-credentials
```

### 7.3 Reloader

Use Stakater Reloader to automatically restart deployments when ConfigMaps
or Secrets change:

```bash
# Install Reloader
helm repo add stakater https://stakater.github.io/stakater-charts
helm install reloader stakater/reloader -n kube-system
```

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  annotations:
    # Watch specific resources
    reloader.stakater.com/auto: "true"
    # Or watch specific ConfigMaps/Secrets:
    # configmap.reloader.stakater.com/reload: "app-config"
    # secret.reloader.stakater.com/reload: "db-credentials"
spec:
  template:
    spec:
      containers:
        - name: app
          image: my-app:v1.0
          envFrom:
            - configMapRef:
                name: app-config
            - secretRef:
                name: db-credentials
```

---

## 8. Configuration Best Practices

### 8.1 Separation of Concerns

```
Configuration hierarchy:
├── Container image: Code + default config
├── ConfigMap: Environment-specific overrides
├── Secret: Sensitive values (credentials, keys)
├── Environment variables: Simple key-value overrides
└── Command-line args: Runtime flags
```

### 8.2 Naming Conventions

```yaml
# Use descriptive, versioned names
apiVersion: v1
kind: ConfigMap
metadata:
  name: frontend-config           # app-component + "config"
  labels:
    app.kubernetes.io/name: frontend
    app.kubernetes.io/component: web
    app.kubernetes.io/part-of: my-platform
    config.kubernetes.io/version: "3"
```

### 8.3 Do Not Store Secrets in ConfigMaps

```yaml
# BAD: Sensitive data in ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  DATABASE_URL: "postgresql://admin:password123@postgres:5432/mydb"
  # password is visible in plain text!

# GOOD: Split configuration and credentials
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  DATABASE_HOST: "postgres.default.svc"
  DATABASE_PORT: "5432"
  DATABASE_NAME: "mydb"
---
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
type: Opaque
stringData:
  username: admin
  password: "password123"
```

### 8.4 Use Volume Mounts Over Environment Variables

```yaml
# Prefer volume mounts for configuration files
# - Auto-updates when ConfigMap changes (no restart needed)
# - Better for multi-line configs
# - Does not leak into child processes or crash dumps

spec:
  containers:
    - name: app
      volumeMounts:
        - name: config
          mountPath: /etc/app/config.yaml
          subPath: config.yaml
  volumes:
    - name: config
      configMap:
        name: app-config
```

### 8.5 RBAC for Secrets

```yaml
# Restrict secret access to only the service accounts that need it
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: secret-reader
  namespace: production
rules:
  - apiGroups: [""]
    resources: ["secrets"]
    resourceNames: ["db-credentials", "api-keys"]   # Specific secrets only
    verbs: ["get"]
    # Never grant "list" or "watch" on all secrets
```

### 8.6 Avoid Hardcoding

```yaml
# BAD: Hardcoded values
spec:
  containers:
    - name: app
      env:
        - name: API_URL
          value: "https://api.production.example.com"

# GOOD: ConfigMap reference
spec:
  containers:
    - name: app
      env:
        - name: API_URL
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: API_URL
```

---

## 9. Environment-Specific Configuration

### 9.1 Kustomize Overlays

```
config/
├── base/
│   ├── kustomization.yaml
│   ├── deployment.yaml
│   ├── configmap.yaml
│   └── secret.yaml
├── overlays/
│   ├── dev/
│   │   ├── kustomization.yaml
│   │   └── configmap-patch.yaml
│   ├── staging/
│   │   ├── kustomization.yaml
│   │   └── configmap-patch.yaml
│   └── production/
│       ├── kustomization.yaml
│       └── configmap-patch.yaml
```

**Base ConfigMap:**

```yaml
# config/base/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  LOG_LEVEL: "info"
  CACHE_TTL: "300"
  FEATURE_NEW_UI: "false"
```

**Production overlay:**

```yaml
# config/overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
patches:
  - path: configmap-patch.yaml
configMapGenerator:
  - name: app-config
    behavior: merge
    literals:
      - LOG_LEVEL=warn
      - CACHE_TTL=3600
      - FEATURE_NEW_UI=true
      - REPLICAS=10
```

```bash
# Preview the production configuration
kubectl kustomize config/overlays/production/

# Apply
kubectl apply -k config/overlays/production/
```

### 9.2 Helm Values

```yaml
# values.yaml (default)
config:
  logLevel: info
  cacheTTL: 300
  featureFlags:
    newUI: false

# values-production.yaml
config:
  logLevel: warn
  cacheTTL: 3600
  featureFlags:
    newUI: true
```

```yaml
# templates/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ .Release.Name }}-config
data:
  LOG_LEVEL: {{ .Values.config.logLevel | quote }}
  CACHE_TTL: {{ .Values.config.cacheTTL | quote }}
  FEATURE_NEW_UI: {{ .Values.config.featureFlags.newUI | quote }}
```

```bash
# Deploy with environment-specific values
helm install my-app ./chart -f values-production.yaml
```

### 9.3 ConfigMap per Environment

```yaml
# dev-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: dev
data:
  config.yaml: |
    server:
      port: 8080
      debug: true
    database:
      host: postgres.dev.svc
      pool_size: 5
    logging:
      level: debug
      format: text
---
# production-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: production
data:
  config.yaml: |
    server:
      port: 8080
      debug: false
    database:
      host: postgres.production.svc
      pool_size: 50
    logging:
      level: warn
      format: json
```

---

## 10. Encrypting Secrets at Rest (EncryptionConfiguration)

By default, Kubernetes stores Secrets in etcd as base64-encoded plain text.
`EncryptionConfiguration` instructs the API server to encrypt Secret data
before writing to etcd, protecting against unauthorized etcd access.

### 10.1 EncryptionConfiguration Manifest

Place this file on the control-plane node (e.g., `/etc/kubernetes/encryption-config.yaml`)
and reference it with `--encryption-provider-config` in the kube-apiserver manifest.

```yaml
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
      - secrets
    providers:
      # aesgcm: authenticated encryption, recommended for new clusters
      - aesgcm:
          keys:
            - name: key1
              secret: <base64-encoded-32-byte-key>   # openssl rand -base64 32

      # aescbc: older AES-CBC mode (HMAC-SHA256 authenticated)
      - aescbc:
          keys:
            - name: key1
              secret: <base64-encoded-32-byte-key>

      # kms: use an external KMS provider (AWS KMS, GCP KMS, HashiCorp Vault)
      # - kms:
      #     name: myKmsPlugin
      #     endpoint: unix:///tmp/socketfile.sock
      #     cachesize: 100
      #     timeout: 3s

      # identity: no encryption (always place last as fallback for reading old data)
      - identity: {}
```

Provider precedence: the **first** provider is used for writes; all providers
are tried in order for reads. Keep `identity` last so pre-existing unencrypted
Secrets can still be read after enabling encryption.

### 10.2 Provider Comparison

| Provider | Algorithm | Notes |
|----------|-----------|-------|
| `aesgcm` | AES-GCM | Authenticated encryption; recommended |
| `aescbc` | AES-CBC + HMAC-SHA256 | Widely supported; older clusters |
| `kms` (v1) | Envelope encryption | Delegates key management to external KMS |
| `kms` (v2) | Envelope encryption | KMS v2 API (GA in 1.29); improved performance |
| `identity` | None | No encryption; used as read-fallback |
| `secretbox` | XSalsa20+Poly1305 | Fast; less common |

### 10.3 Migrating Existing Secrets

After enabling `EncryptionConfiguration`, existing Secrets in etcd remain
unencrypted. Force re-encryption by rewriting all Secrets:

```bash
# Re-encrypt all existing Secrets (reads with identity, writes with new provider)
kubectl get secrets --all-namespaces -o json | kubectl replace -f -

# Verify a Secret is encrypted in etcd (should show gibberish, not plain base64)
# Run on the control-plane node:
ETCDCTL_API=3 etcdctl \
  --cacert /etc/kubernetes/pki/etcd/ca.crt \
  --cert   /etc/kubernetes/pki/etcd/server.crt \
  --key    /etc/kubernetes/pki/etcd/server.key \
  get /registry/secrets/default/my-secret | hexdump -C | head
```

After migration, you can remove the `identity` provider from the config and
restart the API server. New Secrets will then be unreadable without the key.

### 10.4 Key Rotation

1. Add a new key as the **first** entry (new writes use this key).
2. Restart the API server.
3. Re-encrypt all Secrets (step above) so they use the new key.
4. Remove the old key entry and restart the API server again.

---

## Exercises

### Exercise 1: ConfigMap Management

Create a ConfigMap with a multi-line nginx configuration file. Mount it into an
nginx pod and verify the custom configuration is active.

<details>
<summary>Show Answer</summary>

```yaml
# Save as /tmp/configmap-exercise.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-custom-config
data:
  default.conf: |
    server {
        listen 80;
        server_name localhost;

        location / {
            root /usr/share/nginx/html;
            index index.html;
        }

        location /health {
            return 200 '{"status":"healthy"}';
            add_header Content-Type application/json;
        }

        location /info {
            return 200 'Server: nginx\nConfig: custom\n';
            add_header Content-Type text/plain;
        }
    }
---
apiVersion: v1
kind: Pod
metadata:
  name: nginx-custom
spec:
  containers:
    - name: nginx
      image: nginx:1.25
      ports:
        - containerPort: 80
      volumeMounts:
        - name: nginx-config
          mountPath: /etc/nginx/conf.d
          readOnly: true
      resources:
        requests:
          cpu: "50m"
          memory: "64Mi"
  volumes:
    - name: nginx-config
      configMap:
        name: nginx-custom-config
```

```bash
kubectl apply -f /tmp/configmap-exercise.yaml
kubectl wait --for=condition=Ready pod/nginx-custom --timeout=60s

# Verify the config is mounted
kubectl exec nginx-custom -- cat /etc/nginx/conf.d/default.conf

# Test the custom endpoints
kubectl exec nginx-custom -- curl -s localhost/health
# {"status":"healthy"}

kubectl exec nginx-custom -- curl -s localhost/info
# Server: nginx
# Config: custom

# Verify the ConfigMap
kubectl get configmap nginx-custom-config -o yaml

# Clean up
kubectl delete pod nginx-custom
kubectl delete configmap nginx-custom-config
```

</details>

### Exercise 2: Secrets with TLS

Create a self-signed TLS certificate, store it as a Kubernetes Secret, and
configure an nginx pod to serve HTTPS using the certificate.

<details>
<summary>Show Answer</summary>

```bash
# Generate self-signed certificate
openssl req -x509 -nodes -days 365 \
  -newkey rsa:2048 \
  -keyout /tmp/tls.key \
  -out /tmp/tls.crt \
  -subj "/CN=nginx-tls.default.svc/O=exercise"

# Create the TLS secret
kubectl create secret tls nginx-tls-secret \
  --cert=/tmp/tls.crt \
  --key=/tmp/tls.key
```

```yaml
# Save as /tmp/tls-exercise.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-tls-config
data:
  default.conf: |
    server {
        listen 443 ssl;
        server_name localhost;

        ssl_certificate /etc/nginx/ssl/tls.crt;
        ssl_certificate_key /etc/nginx/ssl/tls.key;

        location / {
            return 200 'Hello from HTTPS!\n';
            add_header Content-Type text/plain;
        }
    }
---
apiVersion: v1
kind: Pod
metadata:
  name: nginx-tls
spec:
  containers:
    - name: nginx
      image: nginx:1.25
      ports:
        - containerPort: 443
      volumeMounts:
        - name: tls-certs
          mountPath: /etc/nginx/ssl
          readOnly: true
        - name: nginx-config
          mountPath: /etc/nginx/conf.d
          readOnly: true
      resources:
        requests:
          cpu: "50m"
          memory: "64Mi"
  volumes:
    - name: tls-certs
      secret:
        secretName: nginx-tls-secret
        defaultMode: 0400
    - name: nginx-config
      configMap:
        name: nginx-tls-config
```

```bash
kubectl apply -f /tmp/tls-exercise.yaml
kubectl wait --for=condition=Ready pod/nginx-tls --timeout=60s

# Verify HTTPS is working
kubectl exec nginx-tls -- curl -sk https://localhost/
# Hello from HTTPS!

# Verify the certificate
kubectl exec nginx-tls -- openssl s_client -connect localhost:443 -servername localhost </dev/null 2>/dev/null | openssl x509 -noout -subject
# subject=CN = nginx-tls.default.svc, O = exercise

# Check secret type
kubectl get secret nginx-tls-secret
# TYPE: kubernetes.io/tls

# Clean up
kubectl delete pod nginx-tls
kubectl delete configmap nginx-tls-config
kubectl delete secret nginx-tls-secret
rm /tmp/tls.key /tmp/tls.crt
```

</details>

### Exercise 3: Immutable ConfigMap Versioning

Create an immutable ConfigMap, deploy an application using it, then "update"
the configuration by creating a new version and performing a rolling update.

<details>
<summary>Show Answer</summary>

```yaml
# Save as /tmp/immutable-v1.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config-v1
  labels:
    app: config-demo
    version: v1
immutable: true
data:
  LOG_LEVEL: "info"
  FEATURE_FLAG: "false"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: config-demo
spec:
  replicas: 3
  selector:
    matchLabels:
      app: config-demo
  template:
    metadata:
      labels:
        app: config-demo
    spec:
      containers:
        - name: app
          image: busybox:1.36
          command:
            - sh
            - -c
            - |
              echo "Config: LOG_LEVEL=$LOG_LEVEL, FEATURE_FLAG=$FEATURE_FLAG"
              sleep 3600
          envFrom:
            - configMapRef:
                name: app-config-v1
          resources:
            requests:
              cpu: "50m"
              memory: "32Mi"
```

```bash
# Deploy v1
kubectl apply -f /tmp/immutable-v1.yaml
kubectl rollout status deployment/config-demo

# Verify v1 config
kubectl exec deploy/config-demo -- sh -c 'echo "LOG_LEVEL=$LOG_LEVEL FEATURE_FLAG=$FEATURE_FLAG"'
# LOG_LEVEL=info FEATURE_FLAG=false

# Try to edit the immutable ConfigMap (should fail)
kubectl patch configmap app-config-v1 -p '{"data":{"LOG_LEVEL":"debug"}}'
# Error: ConfigMap is immutable

# Create v2
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config-v2
  labels:
    app: config-demo
    version: v2
immutable: true
data:
  LOG_LEVEL: "debug"
  FEATURE_FLAG: "true"
EOF

# Update deployment to use v2
kubectl set env deployment/config-demo --from=configmap/app-config-v2
kubectl rollout status deployment/config-demo

# Verify v2 config
kubectl exec deploy/config-demo -- sh -c 'echo "LOG_LEVEL=$LOG_LEVEL FEATURE_FLAG=$FEATURE_FLAG"'
# LOG_LEVEL=debug FEATURE_FLAG=true

# Delete old ConfigMap
kubectl delete configmap app-config-v1

# Clean up
kubectl delete deployment config-demo
kubectl delete configmap app-config-v2
```

</details>

### Exercise 4: Multi-Source Configuration

Create a pod that combines configuration from a ConfigMap, a Secret, and the
Downward API into a single projected volume.

<details>
<summary>Show Answer</summary>

```yaml
# Save as /tmp/projected-exercise.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-settings
data:
  app.conf: |
    log_level=info
    max_retries=3
    timeout=30s
---
apiVersion: v1
kind: Secret
metadata:
  name: app-credentials
type: Opaque
stringData:
  api-key: "sk-abc123def456"
  db-password: "SuperSecret123!"
---
apiVersion: v1
kind: Pod
metadata:
  name: projected-demo
  labels:
    app: projected-demo
    version: v1.0.0
  annotations:
    environment: staging
spec:
  containers:
    - name: app
      image: busybox:1.36
      command:
        - sh
        - -c
        - |
          echo "=== Configuration (ConfigMap) ==="
          cat /config/app.conf
          echo ""
          echo "=== Credentials (Secret) ==="
          echo "API Key: $(cat /config/credentials/api-key)"
          echo "DB Password: $(cat /config/credentials/db-password)"
          echo ""
          echo "=== Pod Info (Downward API) ==="
          echo "Pod Name: $(cat /config/podinfo/name)"
          echo "Namespace: $(cat /config/podinfo/namespace)"
          echo "Labels:"
          cat /config/podinfo/labels
          echo ""
          echo "=== Service Account Token ==="
          echo "Token (first 50 chars): $(head -c 50 /config/token)"
          echo "..."
          sleep 3600
      volumeMounts:
        - name: all-config
          mountPath: /config
          readOnly: true
      resources:
        requests:
          cpu: "50m"
          memory: "32Mi"
  volumes:
    - name: all-config
      projected:
        sources:
          - configMap:
              name: app-settings
              items:
                - key: app.conf
                  path: app.conf
          - secret:
              name: app-credentials
              items:
                - key: api-key
                  path: credentials/api-key
                  mode: 0400
                - key: db-password
                  path: credentials/db-password
                  mode: 0400
          - downwardAPI:
              items:
                - path: podinfo/name
                  fieldRef:
                    fieldPath: metadata.name
                - path: podinfo/namespace
                  fieldRef:
                    fieldPath: metadata.namespace
                - path: podinfo/labels
                  fieldRef:
                    fieldPath: metadata.labels
          - serviceAccountToken:
              path: token
              expirationSeconds: 3600
              audience: "kubernetes.default.svc"
```

```bash
kubectl apply -f /tmp/projected-exercise.yaml
kubectl wait --for=condition=Ready pod/projected-demo --timeout=60s

# View the output
kubectl logs projected-demo

# Verify the directory structure
kubectl exec projected-demo -- find /config -type f
# /config/app.conf
# /config/credentials/api-key
# /config/credentials/db-password
# /config/podinfo/name
# /config/podinfo/namespace
# /config/podinfo/labels
# /config/token

# Check file permissions on secrets
kubectl exec projected-demo -- ls -la /config/credentials/
# Should show 0400 permissions

# Clean up
kubectl delete pod projected-demo
kubectl delete configmap app-settings
kubectl delete secret app-credentials
```

</details>

### Exercise 5: Environment-Specific Configuration with Kustomize

Create a base deployment with a ConfigMap and two overlays (dev and production)
with different configuration values. Deploy both and verify the differences.

<details>
<summary>Show Answer</summary>

```bash
# Create directory structure
mkdir -p /tmp/kustomize-exercise/{base,overlays/dev,overlays/production}
```

```yaml
# /tmp/kustomize-exercise/base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
  - configmap.yaml
```

```yaml
# /tmp/kustomize-exercise/base/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: env-config
data:
  LOG_LEVEL: "info"
  CACHE_ENABLED: "true"
  CACHE_TTL: "300"
```

```yaml
# /tmp/kustomize-exercise/base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: env-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: env-app
  template:
    metadata:
      labels:
        app: env-app
    spec:
      containers:
        - name: app
          image: busybox:1.36
          command:
            - sh
            - -c
            - |
              echo "Environment: $ENVIRONMENT"
              echo "Log Level: $LOG_LEVEL"
              echo "Cache: $CACHE_ENABLED (TTL: $CACHE_TTL)"
              sleep 3600
          envFrom:
            - configMapRef:
                name: env-config
          resources:
            requests:
              cpu: "50m"
              memory: "32Mi"
```

```yaml
# /tmp/kustomize-exercise/overlays/dev/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
namespace: dev-env
patches:
  - target:
      kind: ConfigMap
      name: env-config
    patch: |
      - op: replace
        path: /data/LOG_LEVEL
        value: "debug"
      - op: add
        path: /data/ENVIRONMENT
        value: "development"
```

```yaml
# /tmp/kustomize-exercise/overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
namespace: prod-env
patches:
  - target:
      kind: Deployment
      name: env-app
    patch: |
      - op: replace
        path: /spec/replicas
        value: 3
  - target:
      kind: ConfigMap
      name: env-config
    patch: |
      - op: replace
        path: /data/LOG_LEVEL
        value: "warn"
      - op: replace
        path: /data/CACHE_TTL
        value: "3600"
      - op: add
        path: /data/ENVIRONMENT
        value: "production"
```

```bash
# Create namespaces
kubectl create namespace dev-env
kubectl create namespace prod-env

# Preview dev configuration
kubectl kustomize /tmp/kustomize-exercise/overlays/dev/

# Preview production configuration
kubectl kustomize /tmp/kustomize-exercise/overlays/production/

# Deploy both environments
kubectl apply -k /tmp/kustomize-exercise/overlays/dev/
kubectl apply -k /tmp/kustomize-exercise/overlays/production/

# Wait for deployments
kubectl -n dev-env rollout status deployment/env-app
kubectl -n prod-env rollout status deployment/env-app

# Compare configurations
echo "=== Dev Environment ==="
kubectl -n dev-env exec deploy/env-app -- env | grep -E "LOG_LEVEL|CACHE|ENVIRONMENT" | sort

echo "=== Production Environment ==="
kubectl -n prod-env exec deploy/env-app -- env | grep -E "LOG_LEVEL|CACHE|ENVIRONMENT" | sort

# Dev:     LOG_LEVEL=debug, CACHE_TTL=300, ENVIRONMENT=development
# Prod:    LOG_LEVEL=warn,  CACHE_TTL=3600, ENVIRONMENT=production

# Compare replica counts
echo "Dev replicas: $(kubectl -n dev-env get deploy env-app -o jsonpath='{.spec.replicas}')"
echo "Prod replicas: $(kubectl -n prod-env get deploy env-app -o jsonpath='{.spec.replicas}')"
# Dev: 1, Prod: 3

# Clean up
kubectl delete namespace dev-env prod-env
rm -rf /tmp/kustomize-exercise
```

</details>

---

**Previous**: [Storage and Persistence](./04_Storage_and_Persistence.md) | **Next**: [RBAC and Security](./06_RBAC_and_Security.md)
