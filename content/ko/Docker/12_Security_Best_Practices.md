# 보안 모범 사례(Security Best Practices)

**이전**: [컨테이너 네트워킹](./11_Container_Networking.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 컨테이너 보안 위협 모델을 설명하고 일반적인 공격 벡터(Attack Vector)를 파악한다
2. 최소 베이스 이미지와 취약점 스캐닝을 포함한 이미지 보안 모범 사례를 적용한다
3. 최소 권한 원칙(Principle of Least Privilege)에 따른 보안 Dockerfile을 작성한다
4. 읽기 전용 파일시스템(Read-only Filesystem)과 권한 제한(Capability Restriction)으로 런타임 보안 제어를 구현한다
5. Docker Secrets, Kubernetes Secrets, 외부 볼트(Vault)를 사용해 시크릿을 안전하게 관리한다
6. 격리, 암호화, 인그레스/이그레스(Ingress/Egress) 제어로 네트워크 보안을 구성한다
7. 이미지 서명과 Docker Content Trust로 컨테이너 레지스트리를 보호한다
8. Kubernetes SecurityContext와 Pod Security Standards를 적용해 워크로드를 강화한다

## 목차
1. [컨테이너 보안 개요](#1-컨테이너-보안-개요)
2. [이미지 보안](#2-이미지-보안)
3. [Dockerfile 모범 사례](#3-dockerfile-모범-사례)
4. [런타임 보안](#4-런타임-보안)
5. [시크릿 관리](#5-시크릿-관리)
6. [네트워크 보안](#6-네트워크-보안)
7. [컨테이너 레지스트리 보안](#7-컨테이너-레지스트리-보안)
8. [Kubernetes 보안 컨텍스트](#8-kubernetes-보안-컨텍스트)
9. [모니터링 및 감사](#9-모니터링-및-감사)
10. [연습 문제](#10-연습-문제)

**난이도**: ⭐⭐⭐⭐

---

컨테이너는 호스트 커널을 공유하기 때문에, 한 컨테이너의 취약점이 시스템 전체를 위협할 수 있습니다. 보안은 컨테이너 라이프사이클의 모든 계층에 내재되어야 합니다. 최소화되고 스캔된 이미지 빌드부터 최소 권한으로 실행, 네트워크 트래픽 암호화, 런타임 동작의 지속적인 모니터링까지 모두 포함됩니다. 이 레슨은 Docker와 Kubernetes를 아우르는 포괄적인 보안 프레임워크를 제공하여, "동작하는 것"에서 "프로덕션에서 안전하게 동작하는 것"으로 나아갈 수 있도록 합니다.

---

## 1. 컨테이너 보안 개요

### 컨테이너 위협 모델(Container Threat Model)

```
┌─────────────────────────────────────────────────────────────┐
│                 Container Attack Surface                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  1. Supply Chain Attacks                           │    │
│  │     - Malicious base images                        │    │
│  │     - Compromised dependencies                     │    │
│  │     - Vulnerable packages                          │    │
│  └────────────────────────────────────────────────────┘    │
│                         │                                   │
│                         ▼                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │  2. Image Vulnerabilities                          │    │
│  │     - Known CVEs in OS/libraries                   │    │
│  │     - Outdated software                            │    │
│  │     - Exposed secrets in layers                    │    │
│  └────────────────────────────────────────────────────┘    │
│                         │                                   │
│                         ▼                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │  3. Runtime Threats                                │    │
│  │     - Container breakout                           │    │
│  │     - Privilege escalation                         │    │
│  │     - Resource abuse                               │    │
│  └────────────────────────────────────────────────────┘    │
│                         │                                   │
│                         ▼                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │  4. Host/Orchestrator Attacks                      │    │
│  │     - Compromised Docker daemon                    │    │
│  │     - Kubernetes API abuse                         │    │
│  │     - Node compromise                              │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 심층 방어 전략(Defense in Depth Strategy)

```
┌─────────────────────────────────────────────────────────────┐
│               Container Security Layers                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Layer 7: Monitoring & Incident Response                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Runtime detection, Audit logs, Alerting           │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  Layer 6: Network Security                                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Network policies, TLS, Firewalls                  │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  Layer 5: Secrets Management                                │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Vault, Docker secrets, Encrypted storage          │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  Layer 4: Runtime Security                                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Capabilities, Seccomp, AppArmor, SELinux          │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  Layer 3: Image Security                                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Scanning, Signing, Minimal base, No root          │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  Layer 2: Host Security                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │  OS hardening, CIS benchmarks, Updates             │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  Layer 1: Infrastructure Security                           │
│  ┌────────────────────────────────────────────────────┐    │
│  │  IAM, VPC isolation, Physical security             │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 보안 원칙

1. **최소 권한(Least Privilege)**: 필요한 최소 권한으로 실행
2. **심층 방어(Defense in Depth)**: 여러 계층의 보안 제어
3. **불변성(Immutability)**: 컨테이너를 불변 아티팩트로 취급
4. **최소 공격 표면(Minimal Attack Surface)**: 노출된 구성 요소 감소
5. **시프트 레프트(Shift Left)**: 개발 라이프사이클 초기에 보안 적용
6. **제로 트러스트(Zero Trust)**: 모든 것을 검증하고, 아무것도 신뢰하지 않음

---

## 2. 이미지 보안

### 안전한 베이스 이미지 선택

```dockerfile
# ❌ BAD: Large attack surface, many vulnerabilities
FROM ubuntu:latest

# ❌ BAD: "latest" tag is not reproducible
FROM node:latest

# ✅ GOOD: Minimal base image
FROM alpine:3.19

# ✅ GOOD: Distroless (no shell, package manager) — eliminates an entire class of attacks since there are no tools to exploit
FROM gcr.io/distroless/base-debian12

# ✅ GOOD: Specific version tag for reproducibility — ensures every build uses the exact same base
FROM node:18.19-alpine3.19

# ✅ BEST: Digest pinning for immutability — even if a tag is re-pushed with different content, you get the exact image you audited
FROM node:18.19-alpine3.19@sha256:abc123...
```

### 최소 프로덕션 이미지를 위한 멀티 스테이지 빌드

```dockerfile
# ❌ BAD: Build tools in production image
FROM golang:1.21
WORKDIR /app
COPY . .
RUN go build -o myapp
CMD ["./myapp"]

# ✅ GOOD: Multi-stage build
FROM golang:1.21 AS builder
WORKDIR /app
COPY . .
RUN go build -o myapp

FROM alpine:3.19
RUN apk add --no-cache ca-certificates  # Only install what the binary needs — fewer packages means fewer CVEs
COPY --from=builder /app/myapp /myapp
USER 1000  # Non-root user — limits damage if the container is compromised
CMD ["/myapp"]

# ✅ BEST: Distroless final image
FROM golang:1.21 AS builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -o myapp  # Static binary — no libc dependency, so it runs on scratch/distroless without shared libraries

FROM gcr.io/distroless/static-debian12  # No shell, no package manager — an attacker has no tools to work with
COPY --from=builder /app/myapp /myapp
USER nonroot:nonroot  # Distroless ships with a built-in nonroot user for this purpose
CMD ["/myapp"]
```

### Trivy를 사용한 이미지 스캐닝

```bash
# Install Trivy
# macOS
brew install trivy

# Linux
wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee /etc/apt/sources.list.d/trivy.list
sudo apt-get update
sudo apt-get install trivy

# Scan Docker image
trivy image nginx:latest

# Output (truncated):
# nginx:latest (debian 12.4)
# Total: 85 (CRITICAL: 12, HIGH: 23, MEDIUM: 35, LOW: 15)
#
# ├── libssl3 (CVE-2023-12345) CRITICAL
# │   Installed Version: 3.0.11-1~deb12u1
# │   Fixed Version: 3.0.11-1~deb12u2
# └── ...

# Scan with severity filter
trivy image --severity CRITICAL,HIGH nginx:latest

# Scan and exit with error if vulnerabilities found — use this in CI to block deployment of vulnerable images
trivy image --exit-code 1 --severity CRITICAL myapp:latest

# Scan for secrets in image — catches accidentally baked-in API keys, passwords, and private keys in any layer
trivy image --scanners secret nginx:latest

# Generate JSON report
trivy image -f json -o report.json nginx:latest
```

### Snyk를 사용한 스캐닝

```bash
# Install Snyk
npm install -g snyk

# Authenticate
snyk auth

# Scan image
snyk container test nginx:latest

# Scan and monitor
snyk container monitor nginx:latest

# Scan Dockerfile
snyk container test nginx:latest --file=Dockerfile
```

### CI/CD에서 자동화된 스캐닝

```yaml
# .github/workflows/scan.yml
name: Security Scan

on: [push, pull_request]

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build image
        run: docker build -t myapp:${{ github.sha }} .

      - name: Run Trivy scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: myapp:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'
          exit-code: '1'

      - name: Upload Trivy results to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'
```

---

## 3. Dockerfile 모범 사례

### Non-Root 사용자로 실행

```dockerfile
# ❌ BAD: Running as root (default)
FROM nginx:alpine
COPY app /usr/share/nginx/html

# ✅ GOOD: Create and use non-root user
FROM nginx:alpine
RUN addgroup -g 1000 appgroup && \
    adduser -D -u 1000 -G appgroup appuser
USER appuser
COPY app /usr/share/nginx/html

# ✅ GOOD: Use numeric UID (works better in Kubernetes) — K8s runAsUser validates UIDs, not usernames
FROM node:18-alpine
RUN addgroup -g 1001 nodegroup && \
    adduser -D -u 1001 -G nodegroup nodeuser
USER 1001  # Numeric UID avoids issues if /etc/passwd is missing or different in the runtime image
COPY --chown=1001:1001 . /app
WORKDIR /app
CMD ["node", "server.js"]

# ✅ BEST: Distroless with non-root
FROM node:18 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM gcr.io/distroless/nodejs18-debian12
COPY --from=builder /app/node_modules /app/node_modules
COPY --chown=nonroot:nonroot . /app
WORKDIR /app
USER nonroot
CMD ["server.js"]
```

### COPY vs ADD

```dockerfile
# ❌ BAD: ADD has unexpected behavior (auto-extraction)
ADD archive.tar.gz /app/
ADD https://example.com/file.txt /app/

# ✅ GOOD: Use COPY for local files
COPY archive.tar.gz /app/
COPY src/ /app/src/

# ✅ GOOD: Explicit tar extraction if needed
RUN wget https://example.com/archive.tar.gz && \
    tar xzf archive.tar.gz && \
    rm archive.tar.gz
```

### 시크릿 처리

```dockerfile
# ❌ BAD: Secrets in environment variables
FROM alpine
ENV API_KEY=sk-1234567890abcdef
CMD ["./app"]

# ❌ BAD: Secrets in build args (exposed in history)
FROM alpine
ARG API_KEY
RUN curl -H "Authorization: Bearer $API_KEY" https://api.example.com
CMD ["./app"]

# ✅ GOOD: Use Docker BuildKit secrets — the secret is mounted only during this RUN step and never persisted in any layer
# syntax=docker/dockerfile:1.4
FROM alpine
RUN --mount=type=secret,id=api_key \
    API_KEY=$(cat /run/secrets/api_key) && \
    curl -H "Authorization: Bearer $API_KEY" https://api.example.com
CMD ["./app"]

# Build with:
# docker buildx build --secret id=api_key,src=./api_key.txt -t myapp .

# ✅ GOOD: Runtime secrets with Docker Swarm
# Secrets mounted at /run/secrets/<secret_name>
FROM alpine
CMD ["sh", "-c", "API_KEY=$(cat /run/secrets/api_key) ./app"]
```

### .dockerignore 사용

```bash
# .dockerignore
# Prevent sensitive files from being copied into image

# Git
.git
.gitignore

# Secrets
.env
.env.*
*.key
*.pem
secrets/
credentials.json

# Build artifacts
node_modules/
dist/
build/
*.log

# Documentation
README.md
docs/
*.md

# CI/CD
.github/
.gitlab-ci.yml
Jenkinsfile

# IDE
.vscode/
.idea/
*.swp
```

### 레이어 최적화

```dockerfile
# ❌ BAD: Many layers, cache invalidation
FROM node:18-alpine
COPY package.json .
RUN npm install
COPY src/ ./src/
RUN npm run build
RUN npm prune --production

# ✅ GOOD: Optimized layer caching
FROM node:18-alpine AS builder
WORKDIR /app
# Cache dependencies separately — this layer is only rebuilt when package.json changes, saving minutes on every build
COPY package*.json ./
RUN npm ci
# Copy source and build
COPY . .
RUN npm run build && npm prune --production

FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
USER node
CMD ["node", "dist/server.js"]
```

### 강화된 Dockerfile 예제

```dockerfile
# syntax=docker/dockerfile:1.4
FROM golang:1.21-alpine3.19 AS builder

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

WORKDIR /build

# Cache dependencies
COPY go.mod go.sum ./
RUN go mod download

# Build with security flags
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build \
    -ldflags="-w -s -extldflags '-static'" \
    -a \
    -o app \
    ./cmd/server
# -w -s strips debug info and symbol tables — smaller binary with less information for reverse engineering

# Production image — scratch has zero OS packages, zero CVEs, and the smallest possible attack surface
FROM scratch

# Copy necessary files from builder
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo
COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /build/app /app

# Use non-root user (UID 65534 = nobody) — minimal identity with no login shell, home directory, or extra privileges
USER 65534:65534

# Health check — Docker restarts the container automatically if the app becomes unresponsive
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/app", "healthcheck"]

# Run application
ENTRYPOINT ["/app"]
```

---

## 4. 런타임 보안

### Capabilities 제거

Linux capabilities는 세분화된 권한 제어를 제공합니다.

```bash
# ❌ BAD: Running with all capabilities
docker run --privileged myapp

# ✅ GOOD: Drop all capabilities, add only needed ones — even if the container is exploited, the attacker has no kernel-level powers
docker run \
  --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \
  myapp

# List of common capabilities:
# NET_BIND_SERVICE - Bind to ports < 1024
# CHOWN - Change file ownership
# DAC_OVERRIDE - Bypass file permission checks
# SETUID/SETGID - Change UID/GID
# NET_ADMIN - Network configuration
# SYS_TIME - Set system clock
```

### Capabilities를 사용한 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    image: nginx:alpine
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
      - CHOWN
      - SETUID
      - SETGID
    security_opt:
      - no-new-privileges:true  # Prevents setuid/setgid binaries from escalating — blocks common privilege-escalation exploits
    read_only: true  # Immutable filesystem: an attacker cannot install tools or drop malware
    tmpfs:
      - /var/run  # Writable tmpfs for PID files — nginx needs this but the rest of the filesystem stays immutable
      - /var/cache/nginx
      - /tmp
```

### 읽기 전용 파일시스템(Read-Only Filesystem)

```bash
# ❌ BAD: Writable filesystem (default)
docker run myapp

# ✅ GOOD: Read-only root filesystem
docker run --read-only myapp

# ✅ GOOD: Read-only with tmpfs for writable dirs
docker run \
  --read-only \
  --tmpfs /tmp \
  --tmpfs /var/run \
  myapp
```

```yaml
# docker-compose.yml
services:
  app:
    image: myapp:latest
    read_only: true  # Immutable filesystem: prevents persistent malware even if the container is compromised
    tmpfs:
      - /tmp:noexec,nosuid,size=64m  # noexec prevents executing binaries from /tmp — blocks a common attack vector
      - /var/run:noexec,nosuid,size=64m
```

### Seccomp 프로파일

Seccomp(Secure Computing Mode)은 시스템 호출을 제한합니다.

```json
// seccomp-profile.json
{
  "defaultAction": "SCMP_ACT_ERRNO",  // Default-deny: any syscall not explicitly allowed returns an error
  "architectures": [
    "SCMP_ARCH_X86_64",
    "SCMP_ARCH_X86",
    "SCMP_ARCH_AARCH64"
  ],
  "syscalls": [
    {
      "names": [
        "accept",
        "accept4",
        "bind",
        "connect",
        "socket",
        "read",
        "write",
        "open",
        "close",
        "stat",
        "fstat",
        "mmap",
        "mprotect",
        "rt_sigaction",
        "rt_sigreturn",
        "futex",
        "exit_group"
      ],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
```

```bash
# Run with custom seccomp profile
docker run \
  --security-opt seccomp=seccomp-profile.json \
  myapp

# Disable seccomp (not recommended in production)
docker run --security-opt seccomp=unconfined myapp
```

### AppArmor 프로파일

```bash
# Check AppArmor status
sudo aa-status

# Docker's default AppArmor profile: docker-default
# Located at: /etc/apparmor.d/docker

# Run with custom AppArmor profile
docker run \
  --security-opt apparmor=docker-custom \
  myapp

# Disable AppArmor (not recommended)
docker run --security-opt apparmor=unconfined myapp
```

### 보안 옵션 예제

```yaml
# docker-compose.yml
version: '3.8'

services:
  secure-app:
    image: myapp:latest

    # Drop all capabilities
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE

    # Read-only filesystem
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,nodev,size=64m

    # Security options
    security_opt:
      - no-new-privileges:true  # Blocks setuid/setgid escalation — defense-in-depth alongside capability drops
      - apparmor:docker-default  # Mandatory access control — restricts file/network access even for root
      - seccomp:seccomp-profile.json  # Drop dangerous syscalls — limits kernel attack surface

    # User
    user: "1000:1000"  # Non-root — even if code has a vulnerability, the attacker cannot modify system files

    # Resource limits — prevent a runaway container from consuming all host resources (CPU/memory bomb)
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M  # Reservations guarantee scheduling — the container always gets at least this much

    # Prevent privilege escalation
    privileged: false  # Never use privileged mode — it gives the container full host kernel access
```

---

## 5. 시크릿 관리

### Docker Secrets (Swarm Mode)

```bash
# Create secret from file
echo "my-db-password" | docker secret create db_password -

# Create secret from stdin
docker secret create api_key api_key.txt

# List secrets
docker secret ls

# Inspect secret (content not shown)
docker secret inspect db_password

# Use secret in service
docker service create \
  --name web \
  --secret db_password \
  --secret api_key \
  myapp:latest

# Secret available at /run/secrets/<secret_name>
```

```yaml
# docker-compose.yml (Swarm stack)
version: '3.8'

services:
  app:
    image: myapp:latest
    secrets:
      - db_password
      - api_key
    environment:
      # Reference secret file in env var
      - DB_PASSWORD_FILE=/run/secrets/db_password
    deploy:
      replicas: 3

secrets:
  db_password:
    external: true
  api_key:
    file: ./api_key.txt
```

```python
# app.py - Reading secrets
import os

def get_secret(secret_name):
    """Read secret from file."""
    secret_path = f'/run/secrets/{secret_name}'
    try:
        with open(secret_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fallback to environment variable (dev only)
        return os.getenv(secret_name.upper())

db_password = get_secret('db_password')
api_key = get_secret('api_key')
```

### 환경 변수의 함정

```bash
# ❌ BAD: Secrets in environment variables (visible in docker inspect)
docker run -e DB_PASSWORD=secret123 myapp

# ❌ BAD: Secrets in Dockerfile
ENV API_KEY=sk-1234567890

# ❌ BAD: Secrets visible in process list
docker run myapp --api-key=sk-1234567890

# ✅ GOOD: Secrets in files — file-based secrets don't appear in `docker inspect` or process environment
docker run -v /path/to/secrets:/secrets:ro myapp

# ✅ GOOD: Docker secrets (Swarm)
docker service create --secret db_password myapp

# ✅ BETTER: External secret manager
docker run -e VAULT_ADDR=https://vault.example.com myapp
```

### 외부 시크릿 매니저

#### HashiCorp Vault

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    image: myapp:latest
    environment:
      - VAULT_ADDR=https://vault.example.com
      - VAULT_TOKEN_FILE=/run/secrets/vault-token
    secrets:
      - vault-token
    command: sh -c "
      export VAULT_TOKEN=$(cat /run/secrets/vault-token) &&
      export DB_PASSWORD=$(vault kv get -field=password secret/db) &&
      ./app
      "

secrets:
  vault-token:
    file: ./vault-token.txt
```

#### AWS Secrets Manager

```python
# app.py
import boto3
import json

def get_secret(secret_name):
    """Retrieve secret from AWS Secrets Manager."""
    client = boto3.client('secretsmanager', region_name='us-east-1')
    response = client.get_secret_value(SecretId=secret_name)

    if 'SecretString' in response:
        return json.loads(response['SecretString'])
    else:
        return base64.b64decode(response['SecretBinary'])

# Retrieve database credentials
db_creds = get_secret('prod/db/credentials')
db_password = db_creds['password']
```

```dockerfile
# Dockerfile
FROM python:3.11-slim
RUN pip install boto3
COPY app.py /app/
WORKDIR /app

# IAM role attached to ECS task or EC2 instance
# No hardcoded credentials needed
CMD ["python", "app.py"]
```

### Kubernetes Secrets

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque
data:
  # Base64 encoded
  password: cGFzc3dvcmQxMjM=

---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  template:
    spec:
      containers:
      - name: app
        image: myapp:latest
        env:
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: password
        # Or mount as file
        volumeMounts:
        - name: secrets
          mountPath: /secrets
          readOnly: true
      volumes:
      - name: secrets
        secret:
          secretName: db-secret
```

---

## 6. 네트워크 보안

### 네트워크 격리

```yaml
# docker-compose.yml
version: '3.8'

services:
  frontend:
    image: nginx
    networks:
      - public
    ports:
      - "80:80"

  backend:
    image: api:latest
    networks:
      - public
      - private

  database:
    image: postgres
    networks:
      - private
    # Database not exposed to public network — even if the frontend is compromised, the DB is unreachable

networks:
  public:
    driver: bridge
  private:
    driver: bridge
    internal: true  # No external access — containers on this network cannot reach the internet, blocking data exfiltration
```

### TLS 암호화

```yaml
# docker-compose.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    networks:
      - frontend

  app:
    image: myapp:latest
    environment:
      - TLS_CERT=/certs/server.crt
      - TLS_KEY=/certs/server.key
    volumes:
      - ./certs:/certs:ro
    networks:
      - frontend
```

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name example.com;

    # TLS configuration
    ssl_certificate /etc/nginx/certs/server.crt;
    ssl_certificate_key /etc/nginx/certs/server.key;
    ssl_protocols TLSv1.2 TLSv1.3;  # Disable older TLS versions — TLS 1.0/1.1 have known vulnerabilities
    ssl_ciphers HIGH:!aNULL:!MD5;  # Exclude weak ciphers — prevents downgrade attacks
    ssl_prefer_server_ciphers on;

    # Security headers — each header defends against a specific class of web attacks
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;  # HSTS: forces browsers to use HTTPS for a year
    add_header X-Frame-Options "SAMEORIGIN" always;  # Prevents clickjacking by disallowing framing from other origins
    add_header X-Content-Type-Options "nosniff" always;  # Prevents MIME-type sniffing — browser trusts declared Content-Type
    add_header X-XSS-Protection "1; mode=block" always;

    location / {
        proxy_pass http://app:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Egress 제어

```bash
# Block all egress except specific destinations
docker network create \
  --internal \
  isolated-net

docker run -d \
  --name proxy \
  --network isolated-net \
  squid

# Configure Squid to allow only specific domains
# /etc/squid/squid.conf
# acl allowed_sites dstdomain .example.com .api.trusted.com
# http_access allow allowed_sites
# http_access deny all
```

### Kubernetes 네트워크 정책

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all-ingress
  namespace: production
spec:
  podSelector: {}  # Default-deny + explicit allow — limits blast radius of a compromised pod
  policyTypes:
  - Ingress
  - Egress  # Denying both directions forces every service to declare exactly who it talks to

---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-to-backend
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8080

---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-backend-to-db
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Egress
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: database
    ports:
    - protocol: TCP
      port: 5432
  # Allow DNS
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53
```

---

## 7. 컨테이너 레지스트리 보안

### Docker Content Trust를 사용한 이미지 서명

```bash
# Enable Docker Content Trust — ensures only cryptographically signed images can be pulled and run
export DOCKER_CONTENT_TRUST=1

# Generate root and repository keys
docker trust key generate mykey

# Add signer to repository
docker trust signer add --key mykey.pub myuser myrepo/myimage

# Push signed image
docker push myrepo/myimage:v1.0
# Signing and pushing trust metadata...

# Pull signed image (verification automatic with DCT enabled)
docker pull myrepo/myimage:v1.0
# Pull (1 of 1): myrepo/myimage:v1.0@sha256:abc...
# Tagging myrepo/myimage:v1.0@sha256:abc... as myrepo/myimage:v1.0
```

### 이미지 서명을 위한 Notary

```bash
# Install Notary
go install github.com/notaryproject/notation/cmd/notation@latest

# Initialize repository
notary init myrepo/myimage

# Sign image
notary addhash myrepo/myimage v1.0 sha256:abc123...

# List trusted tags
notary list myrepo/myimage

# Verify signature
notary verify myrepo/myimage v1.0
```

### 인증을 사용한 프라이빗 레지스트리

```yaml
# docker-compose.yml
version: '3.8'

services:
  registry:
    image: registry:2
    ports:
      - "5000:5000"
    environment:
      REGISTRY_AUTH: htpasswd
      REGISTRY_AUTH_HTPASSWD_PATH: /auth/htpasswd
      REGISTRY_AUTH_HTPASSWD_REALM: Registry Realm
      REGISTRY_HTTP_TLS_CERTIFICATE: /certs/domain.crt
      REGISTRY_HTTP_TLS_KEY: /certs/domain.key
    volumes:
      - ./auth:/auth
      - ./certs:/certs
      - registry-data:/var/lib/registry

volumes:
  registry-data:
```

```bash
# Create htpasswd file
docker run --rm --entrypoint htpasswd \
  httpd:2 -Bbn myuser mypassword > auth/htpasswd

# Login to private registry
docker login localhost:5000
# Username: myuser
# Password: mypassword

# Tag and push image
docker tag myapp:latest localhost:5000/myapp:latest
docker push localhost:5000/myapp:latest
```

### Harbor 레지스트리

Harbor는 보안 기능이 있는 엔터프라이즈급 레지스트리를 제공합니다:

```yaml
# docker-compose.yml (simplified Harbor setup)
version: '3.8'

services:
  harbor-core:
    image: goharbor/harbor-core:v2.9.0
    environment:
      - CORE_SECRET=not-a-secure-secret
      - JOBSERVICE_SECRET=not-a-secure-secret
    depends_on:
      - harbor-db
      - redis

  harbor-db:
    image: goharbor/harbor-db:v2.9.0
    environment:
      - POSTGRES_PASSWORD=root123

  redis:
    image: goharbor/redis-photon:v2.9.0

  harbor-portal:
    image: goharbor/harbor-portal:v2.9.0
    ports:
      - "80:8080"
```

**Harbor 기능**:
- 취약점 스캐닝 (Trivy, Clair)
- 이미지 서명 (Notary)
- RBAC 및 멀티 테넌시
- 레지스트리 간 복제
- 감사 로깅
- 쿼터 관리

---

## 8. Kubernetes 보안 컨텍스트

### Pod 보안 컨텍스트

```yaml
# secure-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
spec:
  securityContext:
    # Pod-level security context
    runAsNonRoot: true  # Prevents container from running as UID 0 even if the image defaults to root
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000  # Volumes are owned by this GID — ensures the non-root user can read/write mounted data
    seccompProfile:
      type: RuntimeDefault  # Drop dangerous syscalls — defense-in-depth even if container runtime has a bug

  containers:
  - name: app
    image: myapp:latest
    securityContext:
      # Container-level security context (overrides pod-level)
      allowPrivilegeEscalation: false  # Blocks setuid/setgid binaries from gaining elevated privileges
      readOnlyRootFilesystem: true  # Immutable filesystem: an attacker cannot install tools or drop malware
      runAsNonRoot: true
      runAsUser: 1000
      capabilities:
        drop:
        - ALL  # Drop all Linux capabilities first — start from zero privilege
        add:
        - NET_BIND_SERVICE  # Add back only what the app truly needs (binding to ports < 1024)

    volumeMounts:
    - name: tmp
      mountPath: /tmp
    - name: cache
      mountPath: /app/cache

  volumes:
  - name: tmp
    emptyDir: {}
  - name: cache
    emptyDir: {}
```

### Pod 보안 표준(Pod Security Standards)

Kubernetes는 세 가지 보안 수준을 정의합니다:

1. **Privileged**: 제한 없음 (권장하지 않음)
2. **Baseline**: 최소한의 제한
3. **Restricted**: 엄격한 제한 (모범 사례)

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

### 제한된 Pod 예제

```yaml
# restricted-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: secure-app
  namespace: production
spec:
  replicas: 3  # Multiple replicas for high availability — if one pod crashes, others continue serving
  selector:
    matchLabels:
      app: secure-app
  template:
    metadata:
      labels:
        app: secure-app
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault

      containers:
      - name: app
        image: myapp:latest
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          runAsUser: 1000
          capabilities:
            drop:
            - ALL

        resources:
          limits:
            cpu: "1"
            memory: "512Mi"  # limits prevent one pod from starving others
          requests:
            cpu: "100m"
            memory: "128Mi"  # requests guarantee scheduling; the scheduler reserves this much

        volumeMounts:
        - name: tmp
          mountPath: /tmp

        livenessProbe:  # liveness restarts the pod; separate from readiness to avoid cascading restarts
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10

        readinessProbe:  # readiness gates traffic; a failing probe removes the pod from the Service
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

      volumes:
      - name: tmp
        emptyDir:
          sizeLimit: 100Mi  # Prevents a misbehaving process from filling node disk — enforces a hard cap on tmp usage
```

### Pod를 위한 RBAC

```yaml
# service-account.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: myapp-sa
  namespace: production

---
# role.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: myapp-role
  namespace: production
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]

---
# role-binding.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: myapp-rolebinding
  namespace: production
subjects:
- kind: ServiceAccount
  name: myapp-sa
  namespace: production
roleRef:
  kind: Role
  name: myapp-role
  apiGroup: rbac.authorization.k8s.io

---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  namespace: production
spec:
  template:
    spec:
      serviceAccountName: myapp-sa  # Dedicated SA per app — avoids sharing the default SA's broad permissions
      automountServiceAccountToken: false  # Disable if not needed — reduces attack surface if the container is compromised
      containers:
      - name: app
        image: myapp:latest
```

---

## 9. 모니터링 및 감사

### 런타임 이상 탐지를 위한 Falco

```yaml
# falco-daemonset.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: falco
  namespace: falco
spec:
  selector:
    matchLabels:
      app: falco
  template:
    metadata:
      labels:
        app: falco
    spec:
      serviceAccountName: falco
      hostNetwork: true  # Falco needs host-level visibility to detect anomalous network syscalls
      hostPID: true  # Required to see all host processes and correlate events to containers
      containers:
      - name: falco
        image: falcosecurity/falco:0.36.0
        securityContext:
          privileged: true  # Falco needs kernel-level access to intercept syscalls — this is the exception that proves the least-privilege rule
        volumeMounts:
        - name: docker-socket
          mountPath: /var/run/docker.sock
        - name: dev
          mountPath: /host/dev
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: boot
          mountPath: /host/boot
          readOnly: true
        - name: lib-modules
          mountPath: /host/lib/modules
          readOnly: true
        - name: usr
          mountPath: /host/usr
          readOnly: true
        - name: etc
          mountPath: /host/etc
          readOnly: true
      volumes:
      - name: docker-socket
        hostPath:
          path: /var/run/docker.sock
      - name: dev
        hostPath:
          path: /dev
      - name: proc
        hostPath:
          path: /proc
      - name: boot
        hostPath:
          path: /boot
      - name: lib-modules
        hostPath:
          path: /lib/modules
      - name: usr
        hostPath:
          path: /usr
      - name: etc
        hostPath:
          path: /etc
```

**Falco 규칙** (`/etc/falco/falco_rules.yaml`):

```yaml
# Detect shell in container
- rule: Terminal shell in container
  desc: A shell was spawned in a container
  condition: >
    spawned_process and
    container and
    shell_procs and
    proc.tty != 0
  output: >
    Shell spawned in container
    (user=%user.name container=%container.name
    shell=%proc.name parent=%proc.pname
    cmdline=%proc.cmdline)
  priority: WARNING

# Detect file modification in /etc
- rule: Write to /etc directory
  desc: File was written to /etc directory
  condition: >
    evt.type in (write, open) and
    evt.dir = < and
    fd.name startswith /etc
  output: >
    File written in /etc
    (user=%user.name file=%fd.name
    command=%proc.cmdline container=%container.name)
  priority: ERROR

# Detect privilege escalation
- rule: Set Setuid or Setgid bit
  desc: An attempt to set setuid or setgid bit
  condition: >
    evt.type = chmod or evt.type = fchmod and
    ((evt.arg.mode contains S_ISUID) or
     (evt.arg.mode contains S_ISGID))
  output: >
    Setuid or setgid bit set
    (user=%user.name file=%evt.arg.filename
    mode=%evt.arg.mode container=%container.name)
  priority: CRITICAL
```

### Docker Bench Security

```bash
# Run Docker Bench Security
docker run -it --rm \
  --net host \
  --pid host \
  --userns host \
  --cap-add audit_control \
  -e DOCKER_CONTENT_TRUST=$DOCKER_CONTENT_TRUST \
  -v /etc:/etc:ro \
  -v /usr/bin/containerd:/usr/bin/containerd:ro \
  -v /usr/bin/runc:/usr/bin/runc:ro \
  -v /usr/lib/systemd:/usr/lib/systemd:ro \
  -v /var/lib:/var/lib:ro \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  --label docker_bench_security \
  docker/docker-bench-security

# Output:
# [INFO] 1 - Host Configuration
# [PASS] 1.1.1 - Ensure a separate partition for containers...
# [WARN] 1.1.2 - Ensure only trusted users are allowed...
# ...
# [INFO] 4 - Container Images and Build File
# [WARN] 4.1 - Ensure a user for the container has been created
# [PASS] 4.5 - Ensure Content trust for Docker is Enabled
```

### 감사 로깅

#### Docker Daemon 감사

```json
// /etc/docker/daemon.json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "audit-log-enabled": true,
  "audit-log-path": "/var/log/docker-audit.log",
  "audit-log-max-size": "100m",
  "audit-log-max-backups": "5"
}
```

#### Kubernetes 감사 정책

```yaml
# audit-policy.yaml
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
# Log pod changes at RequestResponse level
- level: RequestResponse
  resources:
  - group: ""
    resources: ["pods"]

# Log secret access at Metadata level
- level: Metadata
  resources:
  - group: ""
    resources: ["secrets"]

# Don't log read-only requests
- level: None
  verbs: ["get", "list", "watch"]
```

### Prometheus를 사용한 모니터링

```yaml
# prometheus-config.yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'docker'
    static_configs:
      - targets: ['localhost:9323']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

# Alerting rules
rule_files:
  - '/etc/prometheus/alerts.yml'
```

```yaml
# alerts.yml
groups:
- name: container_security
  rules:
  - alert: ContainerRunningAsRoot
    expr: container_running_as_root == 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Container running as root user"

  - alert: HighCPUUsage
    expr: rate(container_cpu_usage_seconds_total[5m]) > 0.8
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Container CPU usage above 80%"
```

---

## 10. 연습 문제

### 연습 1: 안전한 Dockerfile

안전하지 않은 Dockerfile을 안전한 것으로 변환합니다.

**안전하지 않은 Dockerfile**:
```dockerfile
FROM ubuntu:latest
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . /app
WORKDIR /app
ENV SECRET_KEY=mysecretkey123
CMD ["python3", "app.py"]
```

**작업**:
1. 최소 베이스 이미지 사용
2. 멀티 스테이지 빌드 구현
3. Non-root 사용자로 실행
4. 하드코딩된 시크릿 제거
5. 보안 스캐닝 추가
6. 읽기 전용 파일시스템 구현

### 연습 2: 런타임 보안 구성

안전한 Docker Compose 구성을 생성합니다.

```yaml
# TODO: Harden this configuration
version: '3.8'

services:
  web:
    image: nginx:latest
    ports:
      - "80:80"

  app:
    build: .
    environment:
      - DB_PASSWORD=password123

  db:
    image: postgres:latest
```

**작업**:
1. Capability 제거 추가
2. 읽기 전용 파일시스템 구현
3. 적절한 사용자 구성
4. 환경 변수 시크릿 제거
5. 네트워크 격리 추가
6. 리소스 제한 구현

### 연습 3: Kubernetes 보안 컨텍스트

제한된 Pod 보안 표준으로 배포를 생성합니다.

```yaml
# TODO: Add security context
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 2
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: app
        image: myapp:latest
        ports:
        - containerPort: 8080
```

**작업**:
1. Pod 레벨 보안 컨텍스트 추가
2. 컨테이너 레벨 보안 컨텍스트 추가
3. 모든 capabilities 제거
4. 읽기 전용 루트 파일시스템 구현
5. 리소스 제한 추가
6. 적절한 프로브 구성

### 연습 4: 이미지 스캐닝 파이프라인

취약점을 위해 이미지를 스캔하는 CI/CD 파이프라인을 생성합니다.

**GitHub Actions 워크플로우**:
```yaml
# TODO: Complete this workflow
name: Security Scan

on: [push]

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build image
        run: docker build -t myapp:${{ github.sha }} .

      # TODO: Add Trivy scanning
      # TODO: Fail on CRITICAL vulnerabilities
      # TODO: Upload results to GitHub Security
```

### 연습 5: 시크릿 관리

다중 컨테이너 애플리케이션을 위한 적절한 시크릿 관리를 구현합니다.

**시나리오**: 웹 애플리케이션에 필요한 것:
- 데이터베이스 비밀번호
- 외부 서비스를 위한 API 키
- TLS 인증서

**작업**:
1. Docker Swarm secrets 구현
2. 파일에서 시크릿을 읽는 애플리케이션 코드 생성
3. 적절한 파일 권한 구성
4. 시크릿 로테이션 전략 구현
5. 시크릿 라이프사이클 문서화

### 연습 6: 네트워크 보안

3계층 애플리케이션을 위한 네트워크 격리를 구현합니다.

```yaml
# TODO: Add network security
version: '3.8'

services:
  frontend:
    image: nginx
    ports:
      - "443:443"

  backend:
    image: api:latest

  database:
    image: postgres

  cache:
    image: redis
```

**작업**:
1. 격리된 네트워크 생성
2. 프론트엔드가 데이터베이스에 직접 접근할 수 없도록 보장
3. 백엔드 서비스를 위한 내부 네트워크 구현
4. 모든 통신에 TLS 구성
5. 네트워크 정책 추가 (Kubernetes 사용 시)

---

## 요약

이 레슨에서 배운 내용:

- 컨테이너 보안 위협 모델과 심층 방어 전략
- 이미지 보안: 최소 베이스 이미지, 멀티 스테이지 빌드, 취약점 스캐닝
- Dockerfile 모범 사례: non-root 사용자, 시크릿 처리, 레이어 최적화
- 런타임 보안: capabilities, seccomp, AppArmor, 읽기 전용 파일시스템
- 시크릿 관리: Docker secrets, 외부 vault, Kubernetes secrets
- 네트워크 보안: 격리, TLS, egress 제어, 네트워크 정책
- 레지스트리 보안: 이미지 서명, 콘텐츠 신뢰, 프라이빗 레지스트리
- Kubernetes 보안 컨텍스트: Pod 보안 표준, RBAC, 제한된 pods
- 모니터링 및 감사: Falco, Docker Bench, 감사 로깅, Prometheus

**핵심 요점**:
- 보안은 일회성 구성이 아닌 지속적인 프로세스입니다
- 심층 방어 적용: 여러 계층의 보안 제어
- 시프트 레프트: 개발 라이프사이클 초기에 보안 통합
- 최소 권한: 필요한 최소 권한으로 실행
- 모니터링 및 감사: 보안 사고 탐지 및 대응
- CI/CD 파이프라인에서 보안 스캐닝 자동화

**프로덕션 체크리스트**:
- [ ] 최소 베이스 이미지 사용 (Alpine, distroless)
- [ ] 취약점을 위해 이미지 스캔 (Trivy, Snyk)
- [ ] 컨테이너를 non-root 사용자로 실행
- [ ] 불필요한 capabilities 제거
- [ ] 읽기 전용 파일시스템 구현
- [ ] 시크릿 관리 사용 (Vault, Docker secrets)
- [ ] 네트워크 격리 및 암호화 활성화
- [ ] 이미지 서명 및 검증 (Content Trust)
- [ ] Pod 보안 표준 적용 (Kubernetes)
- [ ] 런타임 동작 모니터링 (Falco)
- [ ] 감사 로깅 구현
- [ ] 정기적인 보안 검토 및 업데이트

**추가 읽기 자료**:
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [CIS Kubernetes Benchmark](https://www.cisecurity.org/benchmark/kubernetes)
- [NIST Application Container Security Guide](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-190.pdf)
- [OWASP Docker Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)

---

## 연습 문제

### 연습 1: 이미지 취약점(Vulnerability) 스캔

Trivy를 사용하여 컨테이너 이미지의 알려진 CVE(공통 취약점 및 노출)를 식별하고 수정합니다.

1. Trivy를 설치합니다: `brew install trivy` (macOS) 또는 [공식 문서](https://aquasecurity.github.io/trivy/) 참조
2. 알려진 취약점이 있는 구버전 이미지를 스캔합니다: `trivy image python:3.8`
3. CRITICAL 및 HIGH 심각도 CVE의 수를 기록합니다
4. 동일 이미지의 최신 버전을 스캔합니다: `trivy image python:3.12-alpine`
5. 결과를 비교합니다 — 더 최신의 alpine 변형이 훨씬 적은 취약점을 가져야 합니다
6. `python:3.12-alpine`을 기본 이미지로 사용하고, 비루트(non-root) 사용자로 실행하며, 간단한 `app.py`를 복사하는 `Dockerfile`을 작성합니다
7. 커스텀 이미지를 빌드하고 스캔합니다: `trivy image myapp:latest`

### 연습 2: 보안을 강화한 Dockerfile 작성

Dockerfile 보안 모범 사례를 적용하여 애플리케이션 이미지를 강화합니다.

1. 다음 안전하지 않은 `Dockerfile`로 시작합니다:
   ```dockerfile
   FROM ubuntu:latest
   RUN apt-get update && apt-get install -y curl wget vim python3
   COPY . /app
   RUN chmod 777 /app
   CMD ["python3", "/app/main.py"]
   ```
2. 위 Dockerfile에서 최소 5가지 보안 문제를 식별합니다
3. 최소한의 기본 이미지, 고정된 버전, 비루트 사용자, 최소 권한 파일 권한, 해당하는 경우 멀티 스테이지 빌드(multi-stage build), 불필요한 도구 제거를 적용하여 재작성합니다
4. 두 버전을 모두 빌드하고 이미지 크기를 비교합니다: `docker images`
5. 두 이미지를 Trivy로 스캔하고 취약점 수를 비교합니다

### 연습 3: 최소 권한(Least Privilege)으로 컨테이너 실행

컨테이너 시작 시 런타임(runtime) 보안 제어를 적용합니다.

1. 비루트 사용자로 컨테이너를 실행합니다: `docker run --rm --user 1000:1000 alpine whoami`
2. 읽기 전용 루트 파일시스템으로 컨테이너를 실행합니다: `docker run --rm --read-only alpine sh -c "echo test > /test.txt"` — 실패를 확인합니다
3. 쓰기 가능한 `/tmp` tmpfs를 추가하여 동일한 컨테이너를 실행합니다: `docker run --rm --read-only --tmpfs /tmp alpine sh -c "echo test > /tmp/test.txt && cat /tmp/test.txt"`
4. 모든 Linux capabilities를 삭제합니다: `docker run --rm --cap-drop ALL alpine ping -c 1 8.8.8.8` — 실패를 확인합니다 (ping은 `CAP_NET_RAW`가 필요)
5. 필요한 capability만 다시 추가합니다: `docker run --rm --cap-drop ALL --cap-add NET_RAW alpine ping -c 1 8.8.8.8`
6. 세 가지 제약을 모두 결합하여 컨테이너를 실행합니다: 비루트 사용자, 읽기 전용 파일시스템, 모든 capabilities 삭제

### 연습 4: 시크릿(Secret)을 이미지에 포함하지 않고 관리하기

이미지나 환경 변수에 시크릿을 저장하지 않는 시크릿 주입(secret injection) 패턴을 실습합니다.

1. 시크릿 파일을 생성합니다: `echo "supersecret_db_password" > /tmp/db_password.txt`
2. 런타임에 바인드 마운트(bind mount)로 시크릿 파일을 마운트합니다: `docker run --rm -v /tmp/db_password.txt:/run/secrets/db_password:ro alpine cat /run/secrets/db_password`
3. 빌드한 이미지에서 `docker history`를 실행하여 시크릿이 이미지에 포함되지 않았음을 확인합니다
4. Docker Compose에서 최상위 `secrets` 블록을 정의하고 서비스에서 참조합니다:
   ```yaml
   secrets:
     db_password:
       file: ./db_password.txt
   services:
     app:
       image: alpine
       secrets:
         - db_password
       command: cat /run/secrets/db_password
   ```
5. `docker compose up`을 실행하고 컨테이너 내부에서 시크릿에 접근할 수 있는지 확인합니다
6. `docker inspect` 환경 변수에 시크릿이 나타나지 않는지 확인합니다

### 연습 5: Docker Content Trust(콘텐츠 신뢰)로 이미지 서명

Docker Content Trust를 사용하여 컨테이너 이미지를 서명하고 검증합니다.

1. Content Trust를 활성화합니다: `export DOCKER_CONTENT_TRUST=1`
2. 신뢰할 수 있는 이미지를 풀(pull)하고 서명 검증을 확인합니다: `docker pull nginx:alpine`
3. 로컬 이미지에 태그를 지정합니다: `docker tag nginx:alpine yourusername/signed-nginx:latest`
4. 서명된 이미지를 Docker Hub에 푸시합니다: `docker push yourusername/signed-nginx:latest` (Docker가 서명 키 생성을 요청함)
5. Content Trust가 활성화된 상태에서 서명된 이미지를 풀합니다: `docker pull yourusername/signed-nginx:latest`
6. Content Trust를 비활성화하고 서명되지 않은 이미지를 풀합니다: `DOCKER_CONTENT_TRUST=0 docker pull <서명되지-않은-이미지>`
7. Content Trust가 보호하는 것과 그 한계에 대해 설명합니다

---

[이전: 11_Container_Networking](./11_Container_Networking.md) | [다음: 00_Overview](./00_Overview.md)
