# Container and Cloud Security

**Previous**: [11_Secrets_Management.md](./11_Secrets_Management.md) | **Next**: [13. Security Testing](./13_Security_Testing.md)

---

Containers revolutionized software deployment, but they also introduced a new attack surface. A vulnerable base image, an overly permissive Dockerfile, or a misconfigured Kubernetes cluster can expose your entire infrastructure. Cloud environments add further complexity with IAM policies, network configurations, and shared responsibility models. This lesson covers security best practices across the container and cloud stack — from building minimal, hardened images to implementing Kubernetes security policies and cloud IAM governance.

## Learning Objectives

- Write secure Dockerfiles that minimize attack surface
- Choose and audit minimal base images (distroless, Alpine, scratch)
- Scan container images for vulnerabilities using Trivy, Snyk, and Hadolint
- Sign and verify container images with cosign and Sigstore
- Implement Kubernetes security controls (RBAC, NetworkPolicy, Pod Security Standards)
- Configure service mesh security with mTLS
- Apply cloud IAM best practices across AWS, GCP, and Azure
- Secure Infrastructure as Code (Terraform, CloudFormation)
- Monitor container runtime security
- Understand supply chain security with SBOM and SLSA

---

## 1. Docker Security Fundamentals

### 1.1 Container Threat Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Container Threat Model                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────┐           │
│  │                    Host OS                            │           │
│  │                                                       │           │
│  │  ┌──────────────────────────────────────────────┐    │           │
│  │  │              Container Runtime                │    │           │
│  │  │                                               │    │           │
│  │  │  ┌────────┐  ┌────────┐  ┌────────┐         │    │           │
│  │  │  │Container│  │Container│  │Container│        │    │           │
│  │  │  │   A     │  │   B     │  │   C     │        │    │           │
│  │  │  │ ┌────┐  │  │ ┌────┐  │  │ ┌────┐  │       │    │           │
│  │  │  │ │App │  │  │ │App │  │  │ │App │  │       │    │           │
│  │  │  │ └────┘  │  │ └────┘  │  │ └────┘  │       │    │           │
│  │  │  └────────┘  └────────┘  └────────┘         │    │           │
│  │  └──────────────────────────────────────────────┘    │           │
│  └──────────────────────────────────────────────────────┘           │
│                                                                      │
│  Attack vectors:                                                     │
│  1. Vulnerable base images (CVEs in OS packages)                    │
│  2. Application vulnerabilities (dependency CVEs)                   │
│  3. Misconfigured containers (running as root, excessive caps)      │
│  4. Container escape (kernel exploits, mount escapes)               │
│  5. Image tampering (supply chain attacks)                          │
│  6. Secrets in images (embedded credentials)                        │
│  7. Excessive network access (no network isolation)                 │
│  8. Resource exhaustion (no limits = noisy neighbor / DoS)          │
│                                                                      │
│  Containers share the host kernel — they are NOT VMs.               │
│  A kernel exploit can compromise ALL containers on the host.         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Docker Security Principles

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Container Security Principles                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Minimal Attack Surface                                           │
│     Use the smallest possible base image                             │
│     Install only required packages                                   │
│     Remove build tools from final image                              │
│                                                                      │
│  2. Least Privilege                                                  │
│     Run as non-root user                                             │
│     Drop all capabilities, add back only what is needed              │
│     Use read-only filesystem where possible                          │
│                                                                      │
│  3. Immutability                                                     │
│     Images are built once, deployed many times                       │
│     Never patch running containers — rebuild and redeploy            │
│     Tag images with digests, not just version tags                   │
│                                                                      │
│  4. Defense in Depth                                                 │
│     Scan images at build, push, and deploy time                      │
│     Network policies between containers                              │
│     Runtime monitoring for anomalous behavior                        │
│                                                                      │
│  5. Verifiable                                                       │
│     Sign images and verify signatures before deployment              │
│     Generate SBOM for every image                                    │
│     Pin dependencies to exact versions                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Secure Dockerfiles

### 2.1 Insecure vs Secure Dockerfile

```dockerfile
# ══════════════════════════════════════════════════════════════════
# BAD: Insecure Dockerfile (common mistakes)
# ══════════════════════════════════════════════════════════════════
FROM python:3.12                    # Full image (900MB+), many packages
WORKDIR /app
COPY . .                             # Copies everything including .env, .git
RUN pip install -r requirements.txt  # Runs as root
ENV API_KEY=sk_live_xxxxx            # Secret baked into image
EXPOSE 8000
CMD ["python", "app.py"]             # Runs as root
# Problems:
# - Runs as root
# - Large attack surface (full OS)
# - Secrets in environment/image layers
# - Copies unnecessary files
# - No health check
# - No resource limits defined


# ══════════════════════════════════════════════════════════════════
# GOOD: Secure Dockerfile (best practices)
# ══════════════════════════════════════════════════════════════════
# Stage 1: Build
FROM python:3.12-slim AS builder

WORKDIR /build

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application code
COPY src/ ./src/

# Stage 2: Production
FROM python:3.12-slim AS production

# Security: Create non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Install only runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what we need from builder
COPY --from=builder /root/.local /home/appuser/.local
COPY --from=builder /build/src ./src/

# Set PATH for user-installed packages
ENV PATH=/home/appuser/.local/bin:$PATH

# Security: Set ownership and switch to non-root
RUN chown -R appuser:appuser /app
USER appuser

# Security: Read-only filesystem friendly
VOLUME ["/tmp"]

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

# Use tini as init process (proper signal handling)
ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2.2 .dockerignore

```
# .dockerignore — Exclude files from Docker build context
# This is critical for security AND build performance

# Version control
.git
.gitignore

# Environment and secrets
.env
.env.*
*.pem
*.key
credentials.json

# Python
__pycache__
*.pyc
*.pyo
.venv
venv
.pytest_cache
.mypy_cache
.coverage
htmlcov

# IDE
.vscode
.idea
*.swp
*.swo

# Docker
Dockerfile*
docker-compose*.yml

# Documentation
README.md
docs/
*.md

# CI/CD
.github
.gitlab-ci.yml

# OS files
.DS_Store
Thumbs.db
```

### 2.3 Multi-Stage Build Patterns

```dockerfile
# ══════════════════════════════════════════════════════════════════
# Pattern 1: Go application — build from scratch
# ══════════════════════════════════════════════════════════════════
FROM golang:1.22 AS builder

WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download

COPY . .
# Build static binary (no external dependencies)
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-w -s" -o /app/server ./cmd/server

# Final image: scratch (literally empty — ~0 MB base)
FROM scratch

# Import CA certificates for HTTPS
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
# Import timezone data
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

# Copy the binary
COPY --from=builder /app/server /server

# Non-root user (numeric UID since scratch has no /etc/passwd)
USER 65534

EXPOSE 8080
ENTRYPOINT ["/server"]


# ══════════════════════════════════════════════════════════════════
# Pattern 2: Node.js — distroless image
# ══════════════════════════════════════════════════════════════════
FROM node:20-slim AS builder

WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci --only=production

COPY src/ ./src/

# Final image: distroless (no shell, no package manager)
FROM gcr.io/distroless/nodejs20-debian12

WORKDIR /app
COPY --from=builder /build/node_modules ./node_modules
COPY --from=builder /build/src ./src

# Distroless runs as nonroot by default
USER nonroot

EXPOSE 3000
CMD ["src/index.js"]


# ══════════════════════════════════════════════════════════════════
# Pattern 3: Python — Alpine (small but has shell)
# ══════════════════════════════════════════════════════════════════
FROM python:3.12-alpine AS builder

RUN apk add --no-cache build-base

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-alpine

# Security hardening
RUN addgroup -S appuser && adduser -S -G appuser appuser && \
    apk add --no-cache tini

WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ ./src/

RUN chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "src.app"]
```

### 2.4 Base Image Comparison

| Base Image | Size | Shell | Package Mgr | CVEs | Use Case |
|-----------|------|-------|-------------|------|----------|
| `ubuntu:24.04` | ~75 MB | Yes | apt | Medium | Full OS needed |
| `debian:bookworm-slim` | ~80 MB | Yes | apt | Medium | General purpose |
| `alpine:3.19` | ~7 MB | Yes | apk | Low | Small with shell |
| `python:3.12-slim` | ~150 MB | Yes | apt + pip | Medium | Python apps |
| `python:3.12-alpine` | ~55 MB | Yes | apk + pip | Low | Python (small) |
| `gcr.io/distroless/python3` | ~50 MB | No | None | Very Low | Python (hardened) |
| `gcr.io/distroless/static` | ~2 MB | No | None | Minimal | Static binaries |
| `scratch` | 0 MB | No | None | None | Go/Rust static |

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Base Image Decision Tree                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Need shell for debugging?                                           │
│  ├── Yes: Alpine (small) or Debian-slim (compatible)                │
│  └── No:                                                             │
│       └── Static binary (Go, Rust)?                                  │
│            ├── Yes: scratch or distroless/static                     │
│            └── No:  distroless/<language>                            │
│                                                                      │
│  General recommendations:                                            │
│  • Development: language-slim (e.g., python:3.12-slim)              │
│  • Production:  distroless or Alpine                                │
│  • Maximum security: scratch (requires static binary)               │
│                                                                      │
│  Alpine caveats:                                                     │
│  • Uses musl libc (not glibc) — some packages may not work         │
│  • Python wheels may need compilation (slower builds)               │
│  • DNS resolution differs from glibc-based images                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Image Scanning

### 3.1 Trivy (Comprehensive Scanner)

```bash
# ── Install Trivy ────────────────────────────────────────────────
brew install trivy          # macOS
apt-get install trivy       # Debian/Ubuntu (add aquasecurity repo first)

# ── Scan a Docker image ─────────────────────────────────────────
trivy image python:3.12-slim

# ── Scan with severity filter ───────────────────────────────────
trivy image --severity CRITICAL,HIGH python:3.12-slim

# ── Fail if vulnerabilities found (for CI/CD) ───────────────────
trivy image --exit-code 1 --severity CRITICAL myapp:latest

# ── Scan a Dockerfile (misconfiguration) ────────────────────────
trivy config Dockerfile

# ── Scan a filesystem (application dependencies) ────────────────
trivy fs --scanners vuln,secret,misconfig /path/to/project

# ── Generate SBOM ───────────────────────────────────────────────
trivy image --format spdx-json --output sbom.json myapp:latest

# ── Scan Kubernetes manifests ────────────────────────────────────
trivy config k8s-manifests/

# ── JSON output for programmatic processing ─────────────────────
trivy image --format json --output results.json myapp:latest

# ── Ignore unfixed vulnerabilities ──────────────────────────────
trivy image --ignore-unfixed myapp:latest
```

```yaml
# ── Trivy in GitHub Actions ─────────────────────────────────────
# .github/workflows/security.yml
name: Security Scan

on:
  push:
    branches: [main]
  pull_request:

jobs:
  trivy-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build image
        run: docker build -t myapp:${{ github.sha }} .

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'myapp:${{ github.sha }}'
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'
          exit-code: '1'

      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'
```

### 3.2 Hadolint (Dockerfile Linter)

```bash
# ── Install Hadolint ─────────────────────────────────────────────
brew install hadolint        # macOS
# Or use Docker:
docker run --rm -i hadolint/hadolint < Dockerfile

# ── Lint a Dockerfile ────────────────────────────────────────────
hadolint Dockerfile

# ── Example output ───────────────────────────────────────────────
# Dockerfile:3 DL3006 warning: Always tag the version of an image explicitly
# Dockerfile:7 DL3008 warning: Pin versions in apt-get install
# Dockerfile:7 DL3009 info: Delete apt-get lists after installing
# Dockerfile:12 DL3025 warning: Use arguments JSON notation for CMD
# Dockerfile:5 DL3045 warning: COPY to a relative destination without WORKDIR

# ── Ignore specific rules ───────────────────────────────────────
hadolint --ignore DL3008 --ignore DL3013 Dockerfile

# ── Configuration file ──────────────────────────────────────────
# .hadolint.yaml
```

```yaml
# .hadolint.yaml
ignored:
  - DL3008   # Pin versions in apt-get (sometimes impractical)

trustedRegistries:
  - docker.io
  - gcr.io
  - ghcr.io

override:
  error:
    - DL3001  # Pipe to install with no version
    - DL3002  # Last user should not be root
  warning:
    - DL3003  # Use WORKDIR
    - DL3006  # Always tag image version
  info:
    - DL3007  # Using latest tag
```

### 3.3 Snyk Container Scanning

```bash
# ── Install Snyk CLI ─────────────────────────────────────────────
npm install -g snyk
snyk auth  # Authenticate

# ── Scan a Docker image ─────────────────────────────────────────
snyk container test myapp:latest

# ── Scan with Dockerfile for better remediation advice ──────────
snyk container test myapp:latest --file=Dockerfile

# ── Monitor (continuous scanning) ───────────────────────────────
snyk container monitor myapp:latest

# ── Generate SBOM ───────────────────────────────────────────────
snyk container sbom myapp:latest --format=spdx-json > sbom.json
```

---

## 4. Image Signing and Verification

### 4.1 Why Sign Images?

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Image Supply Chain Attack                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Without signing:                                                    │
│                                                                      │
│  Developer ──push──▶ Registry ──pull──▶ Production                  │
│                          ↑                                           │
│                      Attacker replaces                                │
│                      image with malicious                             │
│                      version (same tag)                               │
│                                                                      │
│  With signing (cosign):                                              │
│                                                                      │
│  Developer ──push──▶ Registry ──pull──▶ Verify Signature ──▶ Deploy │
│       │                  │                     ↑                     │
│       └──sign────────────┘                     │                     │
│                                           Reject if                  │
│                                           signature                  │
│                                           invalid                    │
│                                                                      │
│  Signing guarantees:                                                 │
│  1. Integrity: image has not been modified                           │
│  2. Authenticity: image was built by a trusted entity                │
│  3. Non-repudiation: signer cannot deny signing                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Cosign (Sigstore)

```bash
# ── Install cosign ──────────────────────────────────────────────
brew install cosign

# ── Keyless signing (recommended — uses OIDC identity) ──────────
# Sign an image (opens browser for OIDC auth)
cosign sign myregistry.io/myapp:v1.0.0

# Verify the signature
cosign verify myregistry.io/myapp:v1.0.0 \
    --certificate-identity user@example.com \
    --certificate-oidc-issuer https://accounts.google.com

# ── Key-based signing ───────────────────────────────────────────
# Generate a key pair
cosign generate-key-pair

# Sign with the private key
cosign sign --key cosign.key myregistry.io/myapp:v1.0.0

# Verify with the public key
cosign verify --key cosign.pub myregistry.io/myapp:v1.0.0

# ── Sign with digest (more secure than tag) ─────────────────────
# Tags are mutable; digests are immutable
IMAGE_DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' myapp:v1.0.0)
cosign sign --key cosign.key "$IMAGE_DIGEST"

# ── Attach SBOM to image ────────────────────────────────────────
cosign attach sbom --sbom sbom.json myregistry.io/myapp:v1.0.0

# ── Verify in CI/CD before deployment ───────────────────────────
cosign verify --key cosign.pub myregistry.io/myapp@sha256:abc123... || exit 1
```

```yaml
# ── Cosign in GitHub Actions ────────────────────────────────────
# .github/workflows/build-sign.yml
name: Build and Sign

on:
  push:
    tags: ['v*']

jobs:
  build-and-sign:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write   # Required for keyless signing

    steps:
      - uses: actions/checkout@v4

      - uses: sigstore/cosign-installer@v3

      - name: Login to registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        id: build
        uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.ref_name }}

      - name: Sign the image (keyless)
        run: cosign sign --yes ghcr.io/${{ github.repository }}@${{ steps.build.outputs.digest }}
```

---

## 5. Kubernetes Security

### 5.1 Kubernetes Security Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Kubernetes Security Layers                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Layer 1: Cluster Infrastructure                                     │
│  ├── API server authentication and encryption                        │
│  ├── etcd encryption at rest                                         │
│  ├── Node security (OS hardening, kubelet config)                    │
│  └── Network encryption (TLS everywhere)                             │
│                                                                      │
│  Layer 2: Cluster Configuration                                      │
│  ├── RBAC (Role-Based Access Control)                                │
│  ├── Admission controllers (OPA/Gatekeeper, Kyverno)                │
│  ├── Pod Security Standards (Restricted/Baseline/Privileged)         │
│  └── Network Policies                                                │
│                                                                      │
│  Layer 3: Application Security                                       │
│  ├── Image scanning and signing                                      │
│  ├── Secret management (External Secrets Operator)                   │
│  ├── Service mesh (mTLS with Istio/Linkerd)                         │
│  └── Runtime security (Falco, Tetragon)                              │
│                                                                      │
│  Layer 4: Data Security                                              │
│  ├── Encryption at rest (StorageClass encryption)                    │
│  ├── Encryption in transit (mTLS)                                    │
│  └── Backup and disaster recovery                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 RBAC (Role-Based Access Control)

```yaml
# ── Principle: Grant minimum required permissions ────────────────

# Role: defines permissions within a namespace
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: production
  name: app-reader
rules:
  # Can read pods and their logs
  - apiGroups: [""]
    resources: ["pods", "pods/log"]
    verbs: ["get", "list", "watch"]
  # Can read configmaps
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list"]
  # Cannot read secrets (intentionally excluded)

---
# RoleBinding: assigns role to a user/group/service account
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  namespace: production
  name: developer-read-access
subjects:
  - kind: Group
    name: developers
    apiGroup: rbac.authorization.k8s.io
  - kind: ServiceAccount
    name: monitoring-sa
    namespace: production
roleRef:
  kind: Role
  name: app-reader
  apiGroup: rbac.authorization.k8s.io

---
# ClusterRole: cluster-wide permissions
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: namespace-admin
rules:
  - apiGroups: [""]
    resources: ["namespaces"]
    verbs: ["get", "list"]
  - apiGroups: ["apps"]
    resources: ["deployments", "replicasets"]
    verbs: ["get", "list", "watch", "create", "update", "patch"]
  # Explicitly deny delete on namespaces
  # (by not including "delete" in verbs)

---
# ClusterRoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: team-leads-namespace-admin
subjects:
  - kind: Group
    name: team-leads
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: namespace-admin
  apiGroup: rbac.authorization.k8s.io
```

```bash
# ── RBAC Audit Commands ──────────────────────────────────────────

# List all roles and bindings
kubectl get roles,rolebindings -A
kubectl get clusterroles,clusterrolebindings

# Check what a user can do
kubectl auth can-i --list --as developer@example.com

# Check specific permission
kubectl auth can-i create pods --namespace production --as developer@example.com

# Find overly permissive bindings
kubectl get clusterrolebindings -o json | \
    jq '.items[] | select(.roleRef.name == "cluster-admin") | .subjects'
```

### 5.3 Network Policies

```yaml
# ── Default: Deny all ingress and egress ─────────────────────────
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production
spec:
  podSelector: {}  # Applies to ALL pods in namespace
  policyTypes:
    - Ingress
    - Egress
  # No ingress or egress rules = deny all

---
# ── Allow specific traffic patterns ──────────────────────────────
# Allow web pods to receive traffic from ingress controller
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-web-ingress
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: web
  policyTypes:
    - Ingress
  ingress:
    - from:
        # Allow from ingress controller namespace
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
          podSelector:
            matchLabels:
              app: ingress-nginx
      ports:
        - protocol: TCP
          port: 8080

---
# Allow web pods to connect to API pods
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-web-to-api
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: api
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: web
      ports:
        - protocol: TCP
          port: 3000

---
# Allow API pods to connect to database
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-api-to-db
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: database
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: api
      ports:
        - protocol: TCP
          port: 5432

---
# Allow DNS resolution for all pods
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Egress
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: kube-system
      ports:
        - protocol: UDP
          port: 53
        - protocol: TCP
          port: 53
```

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Network Policy Visualization                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Internet                                                            │
│      │                                                               │
│      ▼                                                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                       │
│  │ Ingress  │───▶│   Web    │───▶│   API    │                       │
│  │Controller│    │  :8080   │    │  :3000   │                       │
│  └──────────┘    └──────────┘    └──────────┘                       │
│                                       │                              │
│                                       ▼                              │
│                                  ┌──────────┐                        │
│                                  │ Database │                        │
│                                  │  :5432   │                        │
│                                  └──────────┘                        │
│                                                                      │
│  Blocked paths (by NetworkPolicy):                                   │
│  ✗ Internet → API directly                                          │
│  ✗ Internet → Database directly                                     │
│  ✗ Web → Database directly                                          │
│  ✗ Any pod → Internet (except DNS)                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.4 Pod Security Standards

```yaml
# ── Pod Security Standards (PSS) — Restricted Level ──────────────
# Applied via namespace labels (Kubernetes 1.25+)

# Label the namespace to enforce restricted security
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    # Enforce: reject pods that violate
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: latest
    # Warn: log a warning (but allow)
    pod-security.kubernetes.io/warn: restricted
    pod-security.kubernetes.io/warn-version: latest
    # Audit: record in audit log
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/audit-version: latest

---
# Pod that meets restricted security standard
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
  namespace: production
spec:
  # Security: Don't use host namespaces
  hostNetwork: false
  hostPID: false
  hostIPC: false

  # Security: Use non-root
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault

  containers:
    - name: app
      image: myregistry.io/myapp@sha256:abc123...  # Pin to digest
      ports:
        - containerPort: 8080

      # Security context (container level)
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        runAsNonRoot: true
        runAsUser: 1000
        capabilities:
          drop:
            - ALL
          # Only add capabilities that are truly needed
          # add:
          #   - NET_BIND_SERVICE  # If binding to port < 1024

      # Resource limits (prevent DoS)
      resources:
        requests:
          memory: "128Mi"
          cpu: "100m"
        limits:
          memory: "512Mi"
          cpu: "500m"

      # Writable directories via emptyDir
      volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /app/cache

  volumes:
    - name: tmp
      emptyDir:
        sizeLimit: 100Mi
    - name: cache
      emptyDir:
        sizeLimit: 200Mi

  # Security: Service account
  serviceAccountName: app-sa
  automountServiceAccountToken: false  # Disable unless needed
```

### 5.5 Kubernetes Security Checklist

```python
"""
Kubernetes security audit script.
Checks common misconfigurations.
"""
import subprocess
import json

def run_kubectl(args: str) -> dict:
    """Run kubectl command and return JSON output."""
    result = subprocess.run(
        f"kubectl {args} -o json",
        shell=True, capture_output=True, text=True
    )
    return json.loads(result.stdout) if result.stdout else {}


def audit_cluster_security():
    """Audit Kubernetes cluster for security issues."""
    findings = []

    # ── Check 1: Pods running as root ────────────────────────────
    pods = run_kubectl("get pods -A")
    for pod in pods.get("items", []):
        for container in pod["spec"].get("containers", []):
            sc = container.get("securityContext", {})
            if not sc.get("runAsNonRoot", False):
                pod_name = pod["metadata"]["name"]
                ns = pod["metadata"]["namespace"]
                findings.append(
                    f"[HIGH] Pod {ns}/{pod_name} container "
                    f"'{container['name']}' may run as root"
                )

    # ── Check 2: Privileged containers ───────────────────────────
    for pod in pods.get("items", []):
        for container in pod["spec"].get("containers", []):
            sc = container.get("securityContext", {})
            if sc.get("privileged", False):
                pod_name = pod["metadata"]["name"]
                ns = pod["metadata"]["namespace"]
                findings.append(
                    f"[CRITICAL] Pod {ns}/{pod_name} container "
                    f"'{container['name']}' is PRIVILEGED"
                )

    # ── Check 3: No resource limits ──────────────────────────────
    for pod in pods.get("items", []):
        for container in pod["spec"].get("containers", []):
            resources = container.get("resources", {})
            if not resources.get("limits"):
                pod_name = pod["metadata"]["name"]
                ns = pod["metadata"]["namespace"]
                findings.append(
                    f"[MEDIUM] Pod {ns}/{pod_name} container "
                    f"'{container['name']}' has no resource limits"
                )

    # ── Check 4: Default service account usage ───────────────────
    for pod in pods.get("items", []):
        sa = pod["spec"].get("serviceAccountName", "default")
        if sa == "default":
            pod_name = pod["metadata"]["name"]
            ns = pod["metadata"]["namespace"]
            auto_mount = pod["spec"].get(
                "automountServiceAccountToken", True
            )
            if auto_mount:
                findings.append(
                    f"[HIGH] Pod {ns}/{pod_name} uses default SA "
                    f"with token auto-mounted"
                )

    # ── Check 5: Images without digest ───────────────────────────
    for pod in pods.get("items", []):
        for container in pod["spec"].get("containers", []):
            image = container.get("image", "")
            if "@sha256:" not in image and ":latest" in image:
                pod_name = pod["metadata"]["name"]
                ns = pod["metadata"]["namespace"]
                findings.append(
                    f"[MEDIUM] Pod {ns}/{pod_name} uses :latest "
                    f"tag for '{container['name']}'"
                )

    # ── Check 6: Namespaces without network policies ────────────
    namespaces = run_kubectl("get namespaces")
    for ns in namespaces.get("items", []):
        ns_name = ns["metadata"]["name"]
        if ns_name in ("kube-system", "kube-public", "kube-node-lease"):
            continue
        netpols = run_kubectl(f"get networkpolicies -n {ns_name}")
        if not netpols.get("items"):
            findings.append(
                f"[HIGH] Namespace '{ns_name}' has no NetworkPolicies"
            )

    return findings


if __name__ == "__main__":
    print("Kubernetes Security Audit")
    print("=" * 60)
    findings = audit_cluster_security()

    if findings:
        for f in findings:
            print(f)
        print(f"\nTotal findings: {len(findings)}")
    else:
        print("No security issues found.")
```

---

## 6. Service Mesh Security

### 6.1 mTLS with Istio

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Mutual TLS (mTLS) with Service Mesh               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Without mTLS:                                                       │
│  Pod A ────── plaintext ──────▶ Pod B                                │
│  (anyone on the network can intercept)                               │
│                                                                      │
│  With mTLS (Istio sidecar proxy):                                    │
│  ┌────────────────────┐         ┌────────────────────┐              │
│  │  Pod A             │         │  Pod B             │              │
│  │  ┌─────┐ ┌──────┐ │ TLS     │ ┌──────┐ ┌─────┐  │              │
│  │  │ App │→│Envoy │─┼─────────┼─│Envoy │→│ App │  │              │
│  │  └─────┘ └──────┘ │ mTLS    │ └──────┘ └─────┘  │              │
│  └────────────────────┘ mutual  └────────────────────┘              │
│                         auth                                         │
│                                                                      │
│  mTLS provides:                                                      │
│  1. Encryption:      All traffic encrypted in transit                │
│  2. Authentication:  Both sides verify identity (certificates)       │
│  3. Authorization:   Only allowed services can communicate           │
│  4. Automatic:       Certificate rotation handled by mesh            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```yaml
# ── Istio PeerAuthentication: Enforce mTLS ──────────────────────
apiVersion: security.istio.io/v1
kind: PeerAuthentication
metadata:
  name: default
  namespace: production
spec:
  mtls:
    mode: STRICT  # Reject non-mTLS connections

---
# ── Istio AuthorizationPolicy ───────────────────────────────────
# Only allow specific service-to-service communication
apiVersion: security.istio.io/v1
kind: AuthorizationPolicy
metadata:
  name: api-access
  namespace: production
spec:
  selector:
    matchLabels:
      app: api-service
  action: ALLOW
  rules:
    # Allow web-frontend to call api-service
    - from:
        - source:
            principals:
              - "cluster.local/ns/production/sa/web-frontend-sa"
      to:
        - operation:
            methods: ["GET", "POST"]
            paths: ["/api/*"]

    # Allow monitoring to call health endpoint
    - from:
        - source:
            principals:
              - "cluster.local/ns/monitoring/sa/prometheus-sa"
      to:
        - operation:
            methods: ["GET"]
            paths: ["/health", "/metrics"]

---
# Deny all other traffic to api-service
apiVersion: security.istio.io/v1
kind: AuthorizationPolicy
metadata:
  name: deny-all
  namespace: production
spec:
  selector:
    matchLabels:
      app: api-service
  action: DENY
  rules:
    - {}
```

---

## 7. Cloud IAM Best Practices

### 7.1 IAM Principles

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Cloud IAM Security Principles                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Least Privilege                                                  │
│     ├── Grant only the permissions needed for the task               │
│     ├── Use fine-grained policies (not AdministratorAccess)          │
│     └── Review and reduce permissions regularly                      │
│                                                                      │
│  2. No Long-Lived Credentials                                        │
│     ├── Use IAM roles instead of access keys                        │
│     ├── Use OIDC federation for CI/CD                               │
│     └── If keys are necessary, rotate every 90 days                 │
│                                                                      │
│  3. MFA Everywhere                                                   │
│     ├── Require MFA for console access                              │
│     ├── Require MFA for sensitive API calls                         │
│     └── Use hardware security keys for privileged accounts          │
│                                                                      │
│  4. Separation of Duties                                             │
│     ├── Different accounts for dev/staging/prod                     │
│     ├── Break glass procedures for emergency access                  │
│     └── No single person should deploy + approve                    │
│                                                                      │
│  5. Audit Everything                                                 │
│     ├── Enable CloudTrail/Cloud Audit Logs                          │
│     ├── Alert on unusual API patterns                               │
│     └── Regular access reviews                                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 AWS IAM Policies

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowS3ReadOnly",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-app-bucket",
        "arn:aws:s3:::my-app-bucket/*"
      ],
      "Condition": {
        "StringEquals": {
          "aws:RequestedRegion": "us-east-1"
        },
        "Bool": {
          "aws:SecureTransport": "true"
        }
      }
    },
    {
      "Sid": "DenyDeleteBucket",
      "Effect": "Deny",
      "Action": [
        "s3:DeleteBucket",
        "s3:DeleteObject"
      ],
      "Resource": "*"
    }
  ]
}
```

```python
"""
AWS IAM security audit script.
pip install boto3
"""
import boto3
from datetime import datetime, timezone, timedelta


def audit_iam_security():
    """Audit AWS IAM configuration for security issues."""
    iam = boto3.client('iam')
    findings = []

    # ── Check 1: Root account access keys ────────────────────────
    summary = iam.get_account_summary()['SummaryMap']
    if summary.get('AccountAccessKeysPresent', 0) > 0:
        findings.append(
            "[CRITICAL] Root account has active access keys. "
            "Delete them immediately."
        )

    # ── Check 2: MFA not enabled ────────────────────────────────
    if summary.get('AccountMFAEnabled', 0) == 0:
        findings.append(
            "[CRITICAL] Root account does not have MFA enabled."
        )

    # ── Check 3: Users without MFA ──────────────────────────────
    users = iam.list_users()['Users']
    for user in users:
        mfa_devices = iam.list_mfa_devices(
            UserName=user['UserName']
        )['MFADevices']
        if not mfa_devices:
            findings.append(
                f"[HIGH] User '{user['UserName']}' does not have MFA enabled."
            )

    # ── Check 4: Old access keys ────────────────────────────────
    for user in users:
        keys = iam.list_access_keys(
            UserName=user['UserName']
        )['AccessKeyMetadata']
        for key in keys:
            age = datetime.now(timezone.utc) - key['CreateDate']
            if age > timedelta(days=90):
                findings.append(
                    f"[HIGH] User '{user['UserName']}' has access key "
                    f"'{key['AccessKeyId'][:8]}...' that is {age.days} days old. "
                    f"Rotate immediately."
                )
            if key['Status'] == 'Inactive':
                findings.append(
                    f"[MEDIUM] User '{user['UserName']}' has inactive "
                    f"access key '{key['AccessKeyId'][:8]}...'. Delete it."
                )

    # ── Check 5: Unused users ───────────────────────────────────
    for user in users:
        last_used = user.get('PasswordLastUsed')
        if last_used:
            age = datetime.now(timezone.utc) - last_used
            if age > timedelta(days=90):
                findings.append(
                    f"[MEDIUM] User '{user['UserName']}' has not logged "
                    f"in for {age.days} days. Consider disabling."
                )

    # ── Check 6: Overly permissive policies ─────────────────────
    for user in users:
        attached = iam.list_attached_user_policies(
            UserName=user['UserName']
        )['AttachedPolicies']
        for policy in attached:
            if policy['PolicyName'] == 'AdministratorAccess':
                findings.append(
                    f"[HIGH] User '{user['UserName']}' has "
                    f"AdministratorAccess attached directly. "
                    f"Use groups and least-privilege policies."
                )

    return findings


if __name__ == "__main__":
    print("AWS IAM Security Audit")
    print("=" * 60)
    for finding in audit_iam_security():
        print(finding)
```

---

## 8. Infrastructure as Code Security

### 8.1 Terraform Security

```hcl
# ── Secure Terraform Configuration ──────────────────────────────

# Provider configuration — never hardcode credentials
provider "aws" {
  region = var.region
  # Credentials from environment or IAM role
  # NEVER: access_key = "AKIA..."
  # NEVER: secret_key = "wJal..."
}

# Remote state with encryption
terraform {
  backend "s3" {
    bucket         = "terraform-state-mycompany"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
    # State may contain secrets — always encrypt
  }
}

# Security group — restrictive by default
resource "aws_security_group" "web" {
  name_prefix = "web-sg-"
  vpc_id      = var.vpc_id

  # Only allow HTTPS from anywhere
  ingress {
    description = "HTTPS from internet"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # BAD: Don't do this
  # ingress {
  #   from_port   = 0
  #   to_port     = 0
  #   protocol    = "-1"
  #   cidr_blocks = ["0.0.0.0/0"]
  # }

  # Restrict egress too
  egress {
    description = "HTTPS outbound"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# S3 bucket with security controls
resource "aws_s3_bucket" "data" {
  bucket = "mycompany-data-${var.environment}"
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.data.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
```

### 8.2 Scanning IaC with tfsec/Checkov

```bash
# ── tfsec — Terraform security scanner ──────────────────────────
brew install tfsec

# Scan Terraform files
tfsec /path/to/terraform

# Scan with minimum severity
tfsec --minimum-severity HIGH /path/to/terraform

# Generate SARIF output for CI
tfsec --format sarif --out results.sarif /path/to/terraform

# ── Checkov — Multi-framework IaC scanner ───────────────────────
pip install checkov

# Scan Terraform
checkov -d /path/to/terraform

# Scan Kubernetes manifests
checkov -d /path/to/k8s-manifests

# Scan Dockerfiles
checkov --framework dockerfile -f Dockerfile

# Scan CloudFormation
checkov -d /path/to/cfn-templates

# Compact output with only failures
checkov -d /path/to/terraform --compact --quiet
```

```yaml
# ── Checkov in CI/CD ────────────────────────────────────────────
# .github/workflows/iac-security.yml
name: IaC Security

on:
  pull_request:
    paths:
      - 'terraform/**'
      - 'k8s/**'
      - 'Dockerfile*'

jobs:
  checkov:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Checkov
        uses: bridgecrewio/checkov-action@v12
        with:
          directory: terraform/
          framework: terraform
          output_format: sarif
          output_file_path: results.sarif
          soft_fail: false  # Fail the build on issues

      - name: Upload results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: results.sarif
```

---

## 9. Runtime Security Monitoring

### 9.1 Falco (Runtime Threat Detection)

```yaml
# ── Falco rules for container runtime security ──────────────────
# Falco monitors system calls and detects anomalous behavior

# Detect shell spawned in container
- rule: Terminal shell in container
  desc: A shell was started inside a container
  condition: >
    spawned_process and container and
    proc.name in (bash, sh, zsh, ksh, csh, dash) and
    not proc.pname in (cron, crond, supervisord)
  output: >
    Shell spawned in container
    (user=%user.name container=%container.name
     shell=%proc.name parent=%proc.pname
     cmdline=%proc.cmdline image=%container.image.repository)
  priority: WARNING

# Detect sensitive file access
- rule: Read sensitive file in container
  desc: Sensitive file was read inside a container
  condition: >
    open_read and container and
    fd.name in (/etc/shadow, /etc/passwd, /etc/sudoers) and
    not proc.name in (sshd, login)
  output: >
    Sensitive file read in container
    (user=%user.name file=%fd.name container=%container.name
     image=%container.image.repository)
  priority: WARNING

# Detect outbound connection to unexpected port
- rule: Unexpected outbound connection
  desc: Container making outbound connection on unexpected port
  condition: >
    outbound and container and
    not fd.sport in (80, 443, 53, 8080, 8443) and
    not proc.name in (curl, wget, python, node)
  output: >
    Unexpected outbound connection
    (command=%proc.cmdline connection=%fd.name
     container=%container.name image=%container.image.repository)
  priority: NOTICE

# Detect privilege escalation attempt
- rule: Privilege escalation in container
  desc: Detected privilege escalation inside a container
  condition: >
    spawned_process and container and
    (proc.name in (sudo, su) or
     proc.name = setuid or
     proc.args contains "--privileged")
  output: >
    Privilege escalation attempt in container
    (user=%user.name command=%proc.cmdline
     container=%container.name image=%container.image.repository)
  priority: CRITICAL
```

```bash
# ── Install Falco on Kubernetes ─────────────────────────────────
helm repo add falcosecurity https://falcosecurity.github.io/charts
helm install falco falcosecurity/falco \
    --namespace falco --create-namespace \
    --set tty=true \
    --set falcosidekick.enabled=true \
    --set falcosidekick.config.slack.webhookurl="https://hooks.slack.com/..."

# ── View Falco alerts ───────────────────────────────────────────
kubectl logs -l app.kubernetes.io/name=falco -n falco -f
```

---

## 10. Supply Chain Security

### 10.1 Software Bill of Materials (SBOM)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Software Supply Chain                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Source Code ──▶ Dependencies ──▶ Build ──▶ Image ──▶ Deploy        │
│       │               │            │          │          │           │
│       ▼               ▼            ▼          ▼          ▼           │
│  Code Review     Lock files    Hermetic    Sign &     Verify        │
│  SAST scanning   Audit deps    builds      SBOM      signatures    │
│  Secret scan     Vuln scan     Reproduce   Store      Admit only   │
│                                            digest     signed       │
│                                                                      │
│  SBOM (Software Bill of Materials):                                  │
│  A complete inventory of all components in your software             │
│  ├── Direct dependencies (your requirements.txt)                     │
│  ├── Transitive dependencies (deps of deps)                         │
│  ├── OS packages (from base image)                                   │
│  └── Licenses for each component                                    │
│                                                                      │
│  Formats:                                                            │
│  ├── SPDX (ISO standard)                                            │
│  ├── CycloneDX (OWASP)                                              │
│  └── Syft (native)                                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```bash
# ── Generate SBOM with Syft ─────────────────────────────────────
# Install
brew install syft

# Generate SBOM for a Docker image
syft myapp:latest -o spdx-json > sbom.spdx.json
syft myapp:latest -o cyclonedx-json > sbom.cdx.json

# Generate SBOM for a directory (source code)
syft dir:/path/to/project -o spdx-json

# ── Scan SBOM for vulnerabilities with Grype ───────────────────
brew install grype

# Scan from SBOM
grype sbom:sbom.spdx.json

# Scan image directly
grype myapp:latest

# Fail on high/critical
grype myapp:latest --fail-on high
```

### 10.2 SLSA (Supply Chain Levels for Software Artifacts)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SLSA Levels                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Level 0: No guarantees                                              │
│  └── Anyone can build anything, no provenance                        │
│                                                                      │
│  Level 1: Documentation of the build process                         │
│  ├── Provenance exists (who built it, how)                          │
│  └── Build process is documented                                     │
│                                                                      │
│  Level 2: Tamper resistance of the build service                     │
│  ├── Provenance is signed                                            │
│  ├── Build service generates provenance automatically                │
│  └── Build service is version-controlled                             │
│                                                                      │
│  Level 3: Extra resistance to specific threats                       │
│  ├── Provenance is non-forgeable                                     │
│  ├── Isolated, ephemeral build environments                          │
│  └── Source is version-controlled with verified history              │
│                                                                      │
│  For most projects, aim for SLSA Level 2 minimum.                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```yaml
# ── GitHub Actions SLSA provenance generator ────────────────────
# .github/workflows/slsa-build.yml
name: SLSA Build

on:
  push:
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      digest: ${{ steps.build.outputs.digest }}

    steps:
      - uses: actions/checkout@v4

      - name: Build image
        id: build
        run: |
          docker build -t myapp:${{ github.ref_name }} .
          DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' myapp:${{ github.ref_name }} | cut -d@ -f2)
          echo "digest=$DIGEST" >> "$GITHUB_OUTPUT"

  provenance:
    needs: build
    permissions:
      actions: read
      id-token: write
      packages: write
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_container_slsa3.yml@v2.0.0
    with:
      image: ghcr.io/${{ github.repository }}
      digest: ${{ needs.build.outputs.digest }}
```

---

## 11. Exercises

### Exercise 1: Secure Dockerfile Challenge

Take the following insecure Dockerfile and fix all security issues:

```dockerfile
FROM ubuntu:latest
RUN apt-get update && apt-get install -y python3 python3-pip
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
ENV DATABASE_URL=postgresql://admin:password123@db.prod.example.com/mydb
ENV API_SECRET=sk_live_supersecret
EXPOSE 22 80 443 5432 8080
CMD python3 app.py
```

Requirements:
1. Use a minimal base image
2. Multi-stage build
3. Non-root user
4. No secrets in the image
5. Proper .dockerignore
6. Health check
7. Read-only filesystem
8. Pinned versions

### Exercise 2: Kubernetes Security Hardening

Write complete Kubernetes manifests for a web application with:

1. A Deployment with security-hardened pod spec
2. A Service with appropriate type
3. A NetworkPolicy that allows only ingress from the ingress controller and egress to the API service and DNS
4. A ServiceAccount with no default token mounting
5. A Role and RoleBinding for a "developer" group with read-only access
6. Pod Security Standards enforcement on the namespace

### Exercise 3: Container Image Scanner

Build a Python tool that:

1. Pulls a Docker image
2. Runs Trivy scan (or parses Trivy JSON output)
3. Checks for critical/high vulnerabilities
4. Verifies the image is signed (cosign)
5. Generates a compliance report (pass/fail with reasons)
6. Can be used as a CI/CD gate (exit code 1 on failure)

### Exercise 4: Cloud IAM Audit Tool

Create an AWS IAM audit tool that:

1. Lists all IAM users and their attached policies
2. Identifies users without MFA
3. Finds access keys older than 90 days
4. Detects users with AdministratorAccess
5. Identifies unused roles (not assumed in 90 days)
6. Checks for overly permissive S3 bucket policies
7. Generates a CSV report with findings and remediation steps

### Exercise 5: Terraform Security Scanner

Write a tool that scans Terraform configuration for:

1. Security groups with 0.0.0.0/0 on sensitive ports (22, 3389, 5432)
2. S3 buckets without encryption
3. S3 buckets without public access block
4. RDS instances without encryption
5. Resources without required tags (Environment, Owner, ManagedBy)
6. Hardcoded credentials in .tf files
7. Output results in a format suitable for CI/CD gating

### Exercise 6: Supply Chain Security Pipeline

Design and implement a CI/CD pipeline that:

1. Scans source code for vulnerabilities (SAST)
2. Scans dependencies for known CVEs
3. Builds the container image with multi-stage Dockerfile
4. Scans the built image with Trivy
5. Signs the image with cosign
6. Generates SBOM (SPDX format)
7. Attaches SBOM to the image
8. Only deploys if all checks pass

---

## Summary

### Container Security Checklist

| Category | Control | Priority |
|----------|---------|----------|
| Image | Use minimal base images (distroless/Alpine) | Critical |
| Image | Scan for vulnerabilities (Trivy/Snyk) | Critical |
| Image | No secrets in images | Critical |
| Image | Pin base image digests | High |
| Image | Sign images (cosign) | High |
| Dockerfile | Run as non-root | Critical |
| Dockerfile | Multi-stage builds | High |
| Dockerfile | Drop all capabilities | High |
| Dockerfile | Read-only filesystem | Medium |
| Kubernetes | RBAC with least privilege | Critical |
| Kubernetes | Network Policies | Critical |
| Kubernetes | Pod Security Standards | High |
| Kubernetes | No default service account tokens | High |
| Cloud | Least-privilege IAM | Critical |
| Cloud | No long-lived credentials | High |
| Cloud | MFA on all accounts | Critical |
| Cloud | Encrypt state files | High |
| Supply Chain | Generate SBOM | High |
| Supply Chain | Dependency scanning | High |
| Runtime | Monitor with Falco/Tetragon | Medium |

### Key Takeaways

1. **Minimal images reduce attack surface** — every package is a potential vulnerability; use distroless or scratch when possible
2. **Never run as root** — create a dedicated user in every Dockerfile
3. **Sign and verify** — use cosign to sign images and verify before deployment
4. **Network isolation is critical** — default-deny NetworkPolicies prevent lateral movement
5. **Shift left** — scan images, IaC, and dependencies in CI/CD, not just in production
6. **Cloud IAM is the new perimeter** — treat IAM policies with the same rigor as firewall rules
7. **Automate everything** — manual security checks do not scale; integrate into pipelines

---

**Previous**: [11_Secrets_Management.md](./11_Secrets_Management.md) | **Next**: [13. Security Testing](./13_Security_Testing.md)
