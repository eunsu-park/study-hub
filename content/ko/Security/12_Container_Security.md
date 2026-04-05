# 컨테이너 및 클라우드 보안

**이전**: [11_Secrets_Management.md](./11_Secrets_Management.md) | **다음**: [13. Security Testing](./13_Security_Testing.md)

---

컨테이너는 소프트웨어 배포에 혁명을 일으켰지만, 새로운 공격 표면도 도입했습니다. 취약한 베이스 이미지, 지나치게 관대한 Dockerfile, 또는 잘못 구성된 Kubernetes 클러스터는 전체 인프라를 노출시킬 수 있습니다. 클라우드 환경은 IAM 정책, 네트워크 구성, 공유 책임 모델로 더욱 복잡성을 추가합니다. 이 레슨은 컨테이너 및 클라우드 스택 전반의 보안 모범 사례를 다룹니다 — 최소화되고 강화된 이미지 구축부터 Kubernetes 보안 정책 구현 및 클라우드 IAM 거버넌스까지.

## 학습 목표

- 공격 표면을 최소화하는 안전한 Dockerfile 작성
- 최소 베이스 이미지 선택 및 감사(distroless, Alpine, scratch)
- Trivy, Snyk, Hadolint를 사용하여 컨테이너 이미지의 취약점 스캔
- cosign과 Sigstore로 컨테이너 이미지 서명 및 검증
- Kubernetes 보안 제어 구현(RBAC, NetworkPolicy, Pod Security Standards)
- mTLS로 서비스 메시 보안 구성
- AWS, GCP, Azure 전반에 클라우드 IAM 모범 사례 적용
- Infrastructure as Code(Terraform, CloudFormation) 보안
- 컨테이너 런타임 보안 모니터링
- SBOM 및 SLSA로 공급망 보안 이해

---

## 1. Docker 보안 기초

### 1.1 컨테이너 위협 모델

```
┌─────────────────────────────────────────────────────────────────────┐
│                    컨테이너 위협 모델                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────┐           │
│  │                    호스트 OS                          │           │
│  │                                                       │           │
│  │  ┌──────────────────────────────────────────────┐    │           │
│  │  │              컨테이너 런타임                  │    │           │
│  │  │                                               │    │           │
│  │  │  ┌────────┐  ┌────────┐  ┌────────┐         │    │           │
│  │  │  │컨테이너│  │컨테이너│  │컨테이너│        │    │           │
│  │  │  │   A     │  │   B     │  │   C     │        │    │           │
│  │  │  │ ┌────┐  │  │ ┌────┐  │  │ ┌────┐  │       │    │           │
│  │  │  │ │App │  │  │ │App │  │  │ │App │  │       │    │           │
│  │  │  │ └────┘  │  │ └────┘  │  │ └────┘  │       │    │           │
│  │  │  └────────┘  └────────┘  └────────┘         │    │           │
│  │  └──────────────────────────────────────────────┘    │           │
│  └──────────────────────────────────────────────────────┘           │
│                                                                      │
│  공격 벡터:                                                            │
│  1. 취약한 베이스 이미지 (OS 패키지의 CVE)                            │
│  2. 애플리케이션 취약점 (의존성 CVE)                                   │
│  3. 잘못 구성된 컨테이너 (root로 실행, 과도한 권한)                    │
│  4. 컨테이너 탈출 (커널 익스플로잇, 마운트 탈출)                       │
│  5. 이미지 변조 (공급망 공격)                                          │
│  6. 이미지 내 secrets (임베디드 자격 증명)                            │
│  7. 과도한 네트워크 액세스 (네트워크 격리 없음)                        │
│  8. 리소스 고갈 (제한 없음 = 시끄러운 이웃 / DoS)                     │
│                                                                      │
│  컨테이너는 호스트 커널을 공유함 — VM이 아닙니다.                      │
│  커널 익스플로잇은 호스트의 모든 컨테이너를 손상시킬 수 있습니다.       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Docker 보안 원칙

```
┌─────────────────────────────────────────────────────────────────────┐
│                    컨테이너 보안 원칙                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 최소 공격 표면                                                     │
│     가능한 가장 작은 베이스 이미지 사용                                 │
│     필요한 패키지만 설치                                                │
│     최종 이미지에서 빌드 도구 제거                                       │
│                                                                      │
│  2. 최소 권한                                                          │
│     비root 사용자로 실행                                                │
│     모든 권한 삭제, 필요한 것만 다시 추가                                │
│     가능한 곳에서 읽기 전용 파일시스템 사용                              │
│                                                                      │
│  3. 불변성                                                             │
│     이미지는 한 번 빌드되고, 여러 번 배포됨                              │
│     실행 중인 컨테이너를 절대 패치하지 말 것 — 재빌드 및 재배포          │
│     버전 태그뿐만 아니라 digest로 이미지 태그                           │
│                                                                      │
│  4. 심층 방어                                                          │
│     빌드, 푸시, 배포 시 이미지 스캔                                     │
│     컨테이너 간 네트워크 정책                                           │
│     비정상 동작에 대한 런타임 모니터링                                   │
│                                                                      │
│  5. 검증 가능성                                                        │
│     배포 전 이미지 서명 및 서명 검증                                     │
│     모든 이미지에 대한 SBOM 생성                                        │
│     의존성을 정확한 버전으로 고정                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 안전한 Dockerfiles

### 2.1 안전하지 않은 vs 안전한 Dockerfile

```dockerfile
# ══════════════════════════════════════════════════════════════════
# 나쁨: 안전하지 않은 Dockerfile (일반적인 실수)
# ══════════════════════════════════════════════════════════════════
FROM python:3.12                    # 전체 이미지 (900MB+), 많은 패키지
WORKDIR /app
COPY . .                             # .env, .git 포함 모든 것을 복사
RUN pip install -r requirements.txt  # root로 실행
ENV API_KEY=sk_live_xxxxx            # Secret이 이미지에 구워짐
EXPOSE 8000
CMD ["python", "app.py"]             # root로 실행
# 문제점:
# - root로 실행
# - 큰 공격 표면 (전체 OS)
# - 환경/이미지 레이어에 secrets
# - 불필요한 파일 복사
# - 헬스 체크 없음
# - 정의된 리소스 제한 없음


# ══════════════════════════════════════════════════════════════════
# 좋음: 안전한 Dockerfile (모범 사례)
# ══════════════════════════════════════════════════════════════════
# 단계 1: 빌드
FROM python:3.12-slim AS builder

WORKDIR /build

# 먼저 의존성 설치 (레이어 캐싱)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 애플리케이션 코드 복사
COPY src/ ./src/

# 단계 2: 프로덕션
FROM python:3.12-slim AS production

# 보안: 비root 사용자 생성
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin appuser

# 런타임 의존성만 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 빌더에서 필요한 것만 복사
COPY --from=builder /root/.local /home/appuser/.local
COPY --from=builder /build/src ./src/

# 사용자가 설치한 패키지를 위한 PATH 설정
ENV PATH=/home/appuser/.local/bin:$PATH

# 보안: 소유권 설정 및 비root로 전환
RUN chown -R appuser:appuser /app
USER appuser

# 보안: 읽기 전용 파일시스템 친화적
VOLUME ["/tmp"]

# 헬스 체크
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

# tini를 init 프로세스로 사용 (적절한 신호 처리)
ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2.2 .dockerignore

```
# .dockerignore — Docker 빌드 컨텍스트에서 파일 제외
# 이것은 보안과 빌드 성능 모두에 중요

# 버전 관리
.git
.gitignore

# 환경 및 secrets
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

# 문서
README.md
docs/
*.md

# CI/CD
.github
.gitlab-ci.yml

# OS 파일
.DS_Store
Thumbs.db
```

### 2.3 다단계 빌드 패턴

```dockerfile
# ══════════════════════════════════════════════════════════════════
# 패턴 1: Go 애플리케이션 — scratch에서 빌드
# ══════════════════════════════════════════════════════════════════
FROM golang:1.22 AS builder

WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download

COPY . .
# 정적 바이너리 빌드 (외부 의존성 없음)
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-w -s" -o /app/server ./cmd/server

# 최종 이미지: scratch (말 그대로 비어있음 — ~0 MB 베이스)
FROM scratch

# HTTPS를 위한 CA 인증서 가져오기
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
# 타임존 데이터 가져오기
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

# 바이너리 복사
COPY --from=builder /app/server /server

# 비root 사용자 (scratch에는 /etc/passwd가 없으므로 숫자 UID)
USER 65534

EXPOSE 8080
ENTRYPOINT ["/server"]


# ══════════════════════════════════════════════════════════════════
# 패턴 2: Node.js — distroless 이미지
# ══════════════════════════════════════════════════════════════════
FROM node:20-slim AS builder

WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci --only=production

COPY src/ ./src/

# 최종 이미지: distroless (셸 없음, 패키지 관리자 없음)
FROM gcr.io/distroless/nodejs20-debian12

WORKDIR /app
COPY --from=builder /build/node_modules ./node_modules
COPY --from=builder /build/src ./src

# Distroless는 기본적으로 nonroot로 실행
USER nonroot

EXPOSE 3000
CMD ["src/index.js"]


# ══════════════════════════════════════════════════════════════════
# 패턴 3: Python — Alpine (작지만 셸 있음)
# ══════════════════════════════════════════════════════════════════
FROM python:3.12-alpine AS builder

RUN apk add --no-cache build-base

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-alpine

# 보안 강화
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

### 2.4 베이스 이미지 비교

| 베이스 이미지 | 크기 | 셸 | 패키지 관리자 | CVE | 사용 사례 |
|-------------|------|-----|--------------|------|----------|
| `ubuntu:24.04` | ~75 MB | Yes | apt | Medium | 전체 OS 필요 |
| `debian:bookworm-slim` | ~80 MB | Yes | apt | Medium | 범용 |
| `alpine:3.19` | ~7 MB | Yes | apk | Low | 셸이 있는 작은 크기 |
| `python:3.12-slim` | ~150 MB | Yes | apt + pip | Medium | Python 앱 |
| `python:3.12-alpine` | ~55 MB | Yes | apk + pip | Low | Python (작음) |
| `gcr.io/distroless/python3` | ~50 MB | No | None | Very Low | Python (강화됨) |
| `gcr.io/distroless/static` | ~2 MB | No | None | Minimal | 정적 바이너리 |
| `scratch` | 0 MB | No | None | None | Go/Rust 정적 |

```
┌─────────────────────────────────────────────────────────────────────┐
│                    베이스 이미지 결정 트리                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  디버깅을 위한 셸이 필요한가요?                                        │
│  ├── Yes: Alpine (작음) 또는 Debian-slim (호환성)                    │
│  └── No:                                                             │
│       └── 정적 바이너리 (Go, Rust)?                                  │
│            ├── Yes: scratch 또는 distroless/static                   │
│            └── No:  distroless/<language>                            │
│                                                                      │
│  일반 권장사항:                                                        │
│  • 개발: language-slim (예: python:3.12-slim)                        │
│  • 프로덕션: distroless 또는 Alpine                                  │
│  • 최대 보안: scratch (정적 바이너리 필요)                            │
│                                                                      │
│  Alpine 주의사항:                                                     │
│  • glibc가 아닌 musl libc 사용 — 일부 패키지가 작동하지 않을 수 있음  │
│  • Python 휠이 컴파일 필요할 수 있음 (빌드 느림)                      │
│  • DNS 해석이 glibc 기반 이미지와 다름                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 이미지 스캐닝

### 3.1 Trivy (포괄적 스캐너)

```bash
# ── Trivy 설치 ────────────────────────────────────────────────
brew install trivy          # macOS
apt-get install trivy       # Debian/Ubuntu (먼저 aquasecurity 저장소 추가)

# ── Docker 이미지 스캔 ─────────────────────────────────────────
trivy image python:3.12-slim

# ── 심각도 필터로 스캔 ───────────────────────────────────────────
trivy image --severity CRITICAL,HIGH python:3.12-slim

# ── 취약점 발견 시 실패 (CI/CD용) ───────────────────────────────
trivy image --exit-code 1 --severity CRITICAL myapp:latest

# ── Dockerfile 스캔 (잘못된 구성) ────────────────────────────────
trivy config Dockerfile

# ── 파일시스템 스캔 (애플리케이션 의존성) ────────────────────────
trivy fs --scanners vuln,secret,misconfig /path/to/project

# ── SBOM 생성 ───────────────────────────────────────────────────
trivy image --format spdx-json --output sbom.json myapp:latest

# ── Kubernetes 매니페스트 스캔 ────────────────────────────────────
trivy config k8s-manifests/

# ── 프로그래밍 방식 처리를 위한 JSON 출력 ─────────────────────────
trivy image --format json --output results.json myapp:latest

# ── 수정되지 않은 취약점 무시 ──────────────────────────────────
trivy image --ignore-unfixed myapp:latest
```

```yaml
# ── GitHub Actions의 Trivy ─────────────────────────────────────
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

### 3.2 Hadolint (Dockerfile 린터)

```bash
# ── Hadolint 설치 ─────────────────────────────────────────────
brew install hadolint        # macOS
# 또는 Docker 사용:
docker run --rm -i hadolint/hadolint < Dockerfile

# ── Dockerfile 린트 ────────────────────────────────────────────
hadolint Dockerfile

# ── 예제 출력 ───────────────────────────────────────────────────
# Dockerfile:3 DL3006 warning: Always tag the version of an image explicitly
# Dockerfile:7 DL3008 warning: Pin versions in apt-get install
# Dockerfile:7 DL3009 info: Delete apt-get lists after installing
# Dockerfile:12 DL3025 warning: Use arguments JSON notation for CMD
# Dockerfile:5 DL3045 warning: COPY to a relative destination without WORKDIR

# ── 특정 규칙 무시 ───────────────────────────────────────────────
hadolint --ignore DL3008 --ignore DL3013 Dockerfile

# ── 구성 파일 ──────────────────────────────────────────────────
# .hadolint.yaml
```

```yaml
# .hadolint.yaml
ignored:
  - DL3008   # apt-get에서 버전 고정 (때때로 비현실적)

trustedRegistries:
  - docker.io
  - gcr.io
  - ghcr.io

override:
  error:
    - DL3001  # 버전 없이 설치로 파이프
    - DL3002  # 마지막 사용자가 root가 아니어야 함
  warning:
    - DL3003  # WORKDIR 사용
    - DL3006  # 항상 이미지 버전 태그
  info:
    - DL3007  # latest 태그 사용
```

### 3.3 Snyk 컨테이너 스캐닝

```bash
# ── Snyk CLI 설치 ─────────────────────────────────────────────
npm install -g snyk
snyk auth  # 인증

# ── Docker 이미지 스캔 ─────────────────────────────────────────
snyk container test myapp:latest

# ── 더 나은 개선 조언을 위해 Dockerfile과 함께 스캔 ──────────
snyk container test myapp:latest --file=Dockerfile

# ── 모니터 (지속적 스캐닝) ───────────────────────────────────────
snyk container monitor myapp:latest

# ── SBOM 생성 ───────────────────────────────────────────────────
snyk container sbom myapp:latest --format=spdx-json > sbom.json
```

---

## 4. 이미지 서명 및 검증

### 4.1 왜 이미지를 서명하는가?

```
┌─────────────────────────────────────────────────────────────────────┐
│                    이미지 공급망 공격                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  서명 없이:                                                            │
│                                                                      │
│  개발자 ──push──▶ 레지스트리 ──pull──▶ 프로덕션                       │
│                          ↑                                           │
│                      공격자가 이미지를                                 │
│                      악의적인 버전으로                                 │
│                      교체 (동일한 태그)                                │
│                                                                      │
│  서명과 함께 (cosign):                                                │
│                                                                      │
│  개발자 ──push──▶ 레지스트리 ──pull──▶ 서명 검증 ──▶ 배포             │
│       │                  │                     ↑                     │
│       └──서명────────────┘                     │                     │
│                                           서명이 유효하지             │
│                                           않으면 거부                 │
│                                                                      │
│  서명은 보장함:                                                        │
│  1. 무결성: 이미지가 수정되지 않았음                                   │
│  2. 진정성: 이미지가 신뢰할 수 있는 엔티티에 의해 빌드됨                │
│  3. 부인 방지: 서명자가 서명을 부인할 수 없음                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Cosign (Sigstore)

```bash
# ── cosign 설치 ──────────────────────────────────────────────
brew install cosign

# ── 키 없는 서명 (권장 — OIDC ID 사용) ──────────────────────
# 이미지 서명 (OIDC 인증을 위해 브라우저 열림)
cosign sign myregistry.io/myapp:v1.0.0

# 서명 검증
cosign verify myregistry.io/myapp:v1.0.0 \
    --certificate-identity user@example.com \
    --certificate-oidc-issuer https://accounts.google.com

# ── 키 기반 서명 ───────────────────────────────────────────────
# 키 쌍 생성
cosign generate-key-pair

# 개인 키로 서명
cosign sign --key cosign.key myregistry.io/myapp:v1.0.0

# 공개 키로 검증
cosign verify --key cosign.pub myregistry.io/myapp:v1.0.0

# ── digest로 서명 (태그보다 더 안전) ─────────────────────────
# 태그는 변경 가능; digest는 불변
IMAGE_DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' myapp:v1.0.0)
cosign sign --key cosign.key "$IMAGE_DIGEST"

# ── 이미지에 SBOM 첨부 ────────────────────────────────────────
cosign attach sbom --sbom sbom.json myregistry.io/myapp:v1.0.0

# ── 배포 전 CI/CD에서 검증 ───────────────────────────────────
cosign verify --key cosign.pub myregistry.io/myapp@sha256:abc123... || exit 1
```

```yaml
# ── GitHub Actions의 Cosign ────────────────────────────────────
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
      id-token: write   # 키 없는 서명에 필요

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

## 5. Kubernetes 보안

### 5.1 Kubernetes 보안 계층

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Kubernetes 보안 계층                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  레이어 1: 클러스터 인프라                                             │
│  ├── API 서버 인증 및 암호화                                          │
│  ├── etcd 저장 시 암호화                                              │
│  ├── 노드 보안 (OS 강화, kubelet 구성)                                │
│  └── 네트워크 암호화 (모든 곳에 TLS)                                  │
│                                                                      │
│  레이어 2: 클러스터 구성                                               │
│  ├── RBAC (역할 기반 액세스 제어)                                     │
│  ├── 승인 컨트롤러 (OPA/Gatekeeper, Kyverno)                        │
│  ├── Pod Security Standards (Restricted/Baseline/Privileged)       │
│  └── Network Policies                                              │
│                                                                      │
│  레이어 3: 애플리케이션 보안                                           │
│  ├── 이미지 스캐닝 및 서명                                            │
│  ├── Secret 관리 (External Secrets Operator)                       │
│  ├── 서비스 메시 (Istio/Linkerd와 mTLS)                             │
│  └── 런타임 보안 (Falco, Tetragon)                                  │
│                                                                      │
│  레이어 4: 데이터 보안                                                 │
│  ├── 저장 시 암호화 (StorageClass 암호화)                            │
│  ├── 전송 중 암호화 (mTLS)                                           │
│  └── 백업 및 재해 복구                                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 RBAC (역할 기반 액세스 제어)

```yaml
# ── 원칙: 필요한 최소 권한 부여 ────────────────────────────────

# Role: 네임스페이스 내에서 권한 정의
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: production
  name: app-reader
rules:
  # pod와 로그 읽기 가능
  - apiGroups: [""]
    resources: ["pods", "pods/log"]
    verbs: ["get", "list", "watch"]
  # configmaps 읽기 가능
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list"]
  # secrets 읽기 불가 (의도적으로 제외됨)

---
# RoleBinding: 사용자/그룹/서비스 계정에 역할 할당
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
# ClusterRole: 클러스터 전체 권한
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
  # 네임스페이스에서 delete 명시적으로 거부
  # (verbs에 "delete"를 포함하지 않음)

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
# ── RBAC 감사 명령 ──────────────────────────────────────────────

# 모든 역할 및 바인딩 목록
kubectl get roles,rolebindings -A
kubectl get clusterroles,clusterrolebindings

# 사용자가 할 수 있는 작업 확인
kubectl auth can-i --list --as developer@example.com

# 특정 권한 확인
kubectl auth can-i create pods --namespace production --as developer@example.com

# 지나치게 관대한 바인딩 찾기
kubectl get clusterrolebindings -o json | \
    jq '.items[] | select(.roleRef.name == "cluster-admin") | .subjects'
```

### 5.3 Network Policies

```yaml
# ── 기본: 모든 인그레스 및 이그레스 거부 ─────────────────────────
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production
spec:
  podSelector: {}  # 네임스페이스의 모든 pod에 적용
  policyTypes:
    - Ingress
    - Egress
  # 인그레스 또는 이그레스 규칙 없음 = 모두 거부

---
# ── 특정 트래픽 패턴 허용 ──────────────────────────────────────
# 웹 pod가 인그레스 컨트롤러로부터 트래픽 받기 허용
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
        # 인그레스 컨트롤러 네임스페이스에서 허용
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
# 웹 pod가 API pod에 연결 허용
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
# API pod가 데이터베이스에 연결 허용
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
# 모든 pod에 대해 DNS 해석 허용
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
│                    Network Policy 시각화                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  인터넷                                                               │
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
│  NetworkPolicy로 차단된 경로:                                         │
│  ✗ 인터넷 → API 직접                                                 │
│  ✗ 인터넷 → 데이터베이스 직접                                        │
│  ✗ Web → 데이터베이스 직접                                           │
│  ✗ 모든 pod → 인터넷 (DNS 제외)                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.4 Pod Security Standards

```yaml
# ── Pod Security Standards (PSS) — Restricted 레벨 ──────────────
# 네임스페이스 레이블을 통해 적용 (Kubernetes 1.25+)

# 제한된 보안을 강제하기 위해 네임스페이스 레이블 지정
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    # Enforce: 위반하는 pod 거부
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: latest
    # Warn: 경고 로그 (하지만 허용)
    pod-security.kubernetes.io/warn: restricted
    pod-security.kubernetes.io/warn-version: latest
    # Audit: 감사 로그에 기록
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/audit-version: latest

---
# 제한된 보안 표준을 충족하는 Pod
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
  namespace: production
spec:
  # 보안: 호스트 네임스페이스 사용하지 않기
  hostNetwork: false
  hostPID: false
  hostIPC: false

  # 보안: 비root 사용
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault

  containers:
    - name: app
      image: myregistry.io/myapp@sha256:abc123...  # digest에 고정
      ports:
        - containerPort: 8080

      # 보안 컨텍스트 (컨테이너 레벨)
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        runAsNonRoot: true
        runAsUser: 1000
        capabilities:
          drop:
            - ALL
          # 정말 필요한 권한만 추가
          # add:
          #   - NET_BIND_SERVICE  # 1024 미만 포트에 바인딩하는 경우

      # 리소스 제한 (DoS 방지)
      resources:
        requests:
          memory: "128Mi"
          cpu: "100m"
        limits:
          memory: "512Mi"
          cpu: "500m"

      # emptyDir을 통한 쓰기 가능 디렉토리
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

  # 보안: 서비스 계정
  serviceAccountName: app-sa
  automountServiceAccountToken: false  # 필요하지 않으면 비활성화
```

### 5.5 Kubernetes 보안 체크리스트

```python
"""
Kubernetes 보안 감사 스크립트.
일반적인 잘못된 구성 확인.
"""
import subprocess
import json

def run_kubectl(args: str) -> dict:
    """kubectl 명령 실행 및 JSON 출력 반환."""
    result = subprocess.run(
        f"kubectl {args} -o json",
        shell=True, capture_output=True, text=True
    )
    return json.loads(result.stdout) if result.stdout else {}


def audit_cluster_security():
    """보안 문제를 위한 Kubernetes 클러스터 감사."""
    findings = []

    # ── 확인 1: root로 실행되는 Pod ────────────────────────────
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

    # ── 확인 2: 특권 컨테이너 ───────────────────────────────────
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

    # ── 확인 3: 리소스 제한 없음 ──────────────────────────────────
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

    # ── 확인 4: 기본 서비스 계정 사용 ───────────────────────────────
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

    # ── 확인 5: digest 없는 이미지 ───────────────────────────────
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

    # ── 확인 6: 네트워크 정책 없는 네임스페이스 ────────────────────
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

## 6. 서비스 메시 보안

### 6.1 Istio와 mTLS

```
┌─────────────────────────────────────────────────────────────────────┐
│                    서비스 메시와 Mutual TLS (mTLS)                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  mTLS 없이:                                                           │
│  Pod A ────── 평문 ──────▶ Pod B                                     │
│  (네트워크의 누구나 가로챌 수 있음)                                    │
│                                                                      │
│  mTLS와 함께 (Istio 사이드카 프록시):                                 │
│  ┌────────────────────┐         ┌────────────────────┐              │
│  │  Pod A             │         │  Pod B             │              │
│  │  ┌─────┐ ┌──────┐ │ TLS     │ ┌──────┐ ┌─────┐  │              │
│  │  │ App │→│Envoy │─┼─────────┼─│Envoy │→│ App │  │              │
│  │  └─────┘ └──────┘ │ mTLS    │ └──────┘ └─────┘  │              │
│  └────────────────────┘ 상호    └────────────────────┘              │
│                         인증                                         │
│                                                                      │
│  mTLS 제공:                                                           │
│  1. 암호화:      모든 트래픽이 전송 중 암호화됨                        │
│  2. 인증:        양쪽 모두 신원 확인 (인증서)                          │
│  3. 권한 부여:   허용된 서비스만 통신 가능                             │
│  4. 자동:        인증서 교체가 메시에 의해 처리됨                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```yaml
# ── Istio PeerAuthentication: mTLS 강제 ──────────────────────
apiVersion: security.istio.io/v1
kind: PeerAuthentication
metadata:
  name: default
  namespace: production
spec:
  mtls:
    mode: STRICT  # mTLS가 아닌 연결 거부

---
# ── Istio AuthorizationPolicy ───────────────────────────────────
# 특정 서비스 간 통신만 허용
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
    # web-frontend가 api-service 호출 허용
    - from:
        - source:
            principals:
              - "cluster.local/ns/production/sa/web-frontend-sa"
      to:
        - operation:
            methods: ["GET", "POST"]
            paths: ["/api/*"]

    # 모니터링이 헬스 엔드포인트 호출 허용
    - from:
        - source:
            principals:
              - "cluster.local/ns/monitoring/sa/prometheus-sa"
      to:
        - operation:
            methods: ["GET"]
            paths: ["/health", "/metrics"]

---
# api-service로의 다른 모든 트래픽 거부
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

## 7. 클라우드 IAM 모범 사례

### 7.1 IAM 원칙

```
┌─────────────────────────────────────────────────────────────────────┐
│                    클라우드 IAM 보안 원칙                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 최소 권한                                                          │
│     ├── 작업에 필요한 권한만 부여                                      │
│     ├── 세분화된 정책 사용 (AdministratorAccess 아님)                 │
│     └── 정기적으로 권한 검토 및 축소                                   │
│                                                                      │
│  2. 장기 자격 증명 없음                                                │
│     ├── 액세스 키 대신 IAM 역할 사용                                  │
│     ├── CI/CD를 위한 OIDC 페더레이션 사용                              │
│     └── 키가 필요한 경우 90일마다 교체                                 │
│                                                                      │
│  3. 모든 곳에 MFA                                                      │
│     ├── 콘솔 액세스에 MFA 필요                                        │
│     ├── 민감한 API 호출에 MFA 필요                                    │
│     └── 특권 계정에 하드웨어 보안 키 사용                              │
│                                                                      │
│  4. 직무 분리                                                          │
│     ├── dev/staging/prod를 위한 다른 계정                             │
│     ├── 긴급 액세스를 위한 비상 절차                                   │
│     └── 단일 사람이 배포 + 승인하면 안 됨                              │
│                                                                      │
│  5. 모든 것 감사                                                       │
│     ├── CloudTrail/Cloud Audit Logs 활성화                          │
│     ├── 비정상적인 API 패턴에 경고                                     │
│     └── 정기적인 액세스 검토                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 AWS IAM 정책

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
AWS IAM 보안 감사 스크립트.
pip install boto3
"""
import boto3
from datetime import datetime, timezone, timedelta


def audit_iam_security():
    """보안 문제를 위한 AWS IAM 구성 감사."""
    iam = boto3.client('iam')
    findings = []

    # ── 확인 1: Root 계정 액세스 키 ────────────────────────────
    summary = iam.get_account_summary()['SummaryMap']
    if summary.get('AccountAccessKeysPresent', 0) > 0:
        findings.append(
            "[CRITICAL] Root account has active access keys. "
            "Delete them immediately."
        )

    # ── 확인 2: MFA 활성화되지 않음 ────────────────────────────
    if summary.get('AccountMFAEnabled', 0) == 0:
        findings.append(
            "[CRITICAL] Root account does not have MFA enabled."
        )

    # ── 확인 3: MFA 없는 사용자 ──────────────────────────────────
    users = iam.list_users()['Users']
    for user in users:
        mfa_devices = iam.list_mfa_devices(
            UserName=user['UserName']
        )['MFADevices']
        if not mfa_devices:
            findings.append(
                f"[HIGH] User '{user['UserName']}' does not have MFA enabled."
            )

    # ── 확인 4: 오래된 액세스 키 ────────────────────────────────
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

    # ── 확인 5: 사용하지 않는 사용자 ───────────────────────────────
    for user in users:
        last_used = user.get('PasswordLastUsed')
        if last_used:
            age = datetime.now(timezone.utc) - last_used
            if age > timedelta(days=90):
                findings.append(
                    f"[MEDIUM] User '{user['UserName']}' has not logged "
                    f"in for {age.days} days. Consider disabling."
                )

    # ── 확인 6: 지나치게 관대한 정책 ─────────────────────────────
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

## 8. Infrastructure as Code 보안

### 8.1 Terraform 보안

```hcl
# ── 안전한 Terraform 구성 ──────────────────────────────────────

# Provider 구성 — 절대 자격 증명 하드코딩하지 말 것
provider "aws" {
  region = var.region
  # 환경 또는 IAM 역할에서 자격 증명
  # 절대 안 됨: access_key = "AKIA..."
  # 절대 안 됨: secret_key = "wJal..."
}

# 암호화가 있는 원격 상태
terraform {
  backend "s3" {
    bucket         = "terraform-state-mycompany"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
    # 상태에 secrets가 포함될 수 있음 — 항상 암호화
  }
}

# 보안 그룹 — 기본적으로 제한적
resource "aws_security_group" "web" {
  name_prefix = "web-sg-"
  vpc_id      = var.vpc_id

  # 어디서나 HTTPS만 허용
  ingress {
    description = "HTTPS from internet"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # 나쁨: 이것을 하지 마세요
  # ingress {
  #   from_port   = 0
  #   to_port     = 0
  #   protocol    = "-1"
  #   cidr_blocks = ["0.0.0.0/0"]
  # }

  # 이그레스도 제한
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

# 보안 제어가 있는 S3 버킷
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

### 8.2 tfsec/Checkov로 IaC 스캔

```bash
# ── tfsec — Terraform 보안 스캐너 ──────────────────────────────
brew install tfsec

# Terraform 파일 스캔
tfsec /path/to/terraform

# 최소 심각도로 스캔
tfsec --minimum-severity HIGH /path/to/terraform

# CI를 위한 SARIF 출력 생성
tfsec --format sarif --out results.sarif /path/to/terraform

# ── Checkov — 다중 프레임워크 IaC 스캐너 ───────────────────────
pip install checkov

# Terraform 스캔
checkov -d /path/to/terraform

# Kubernetes 매니페스트 스캔
checkov -d /path/to/k8s-manifests

# Dockerfiles 스캔
checkov --framework dockerfile -f Dockerfile

# CloudFormation 스캔
checkov -d /path/to/cfn-templates

# 실패만 포함한 간결한 출력
checkov -d /path/to/terraform --compact --quiet
```

```yaml
# ── CI/CD의 Checkov ────────────────────────────────────────────
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
          soft_fail: false  # 문제 발생 시 빌드 실패

      - name: Upload results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: results.sarif
```

---

## 9. 런타임 보안 모니터링

### 9.1 Falco (런타임 위협 감지)

```yaml
# ── 컨테이너 런타임 보안을 위한 Falco 규칙 ──────────────────────
# Falco는 시스템 콜을 모니터링하고 비정상 동작 감지

# 컨테이너에서 셸이 생성됨 감지
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

# 민감한 파일 액세스 감지
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

# 예상치 못한 포트로의 아웃바운드 연결 감지
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

# 권한 상승 시도 감지
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
# ── Kubernetes에 Falco 설치 ─────────────────────────────────────
helm repo add falcosecurity https://falcosecurity.github.io/charts
helm install falco falcosecurity/falco \
    --namespace falco --create-namespace \
    --set tty=true \
    --set falcosidekick.enabled=true \
    --set falcosidekick.config.slack.webhookurl="https://hooks.slack.com/..."

# ── Falco 알림 보기 ───────────────────────────────────────────────
kubectl logs -l app.kubernetes.io/name=falco -n falco -f
```

---

## 10. 공급망 보안

### 10.1 Software Bill of Materials (SBOM)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    소프트웨어 공급망                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  소스 코드 ──▶ 의존성 ──▶ 빌드 ──▶ 이미지 ──▶ 배포                  │
│       │               │            │          │          │           │
│       ▼               ▼            ▼          ▼          ▼           │
│  코드 리뷰       Lock 파일     Hermetic    Sign &     Verify        │
│  SAST 스캔      의존성 감사     빌드        SBOM      서명          │
│  Secret 스캔    취약점 스캔    재현 가능    digest   서명된 것만   │
│                                            저장      승인          │
│                                                                      │
│  SBOM (Software Bill of Materials):                                 │
│  소프트웨어의 모든 구성 요소의 완전한 목록                             │
│  ├── 직접 의존성 (requirements.txt)                                 │
│  ├── 전이 의존성 (의존성의 의존성)                                   │
│  ├── OS 패키지 (베이스 이미지에서)                                   │
│  └── 각 구성 요소의 라이선스                                         │
│                                                                      │
│  형식:                                                                │
│  ├── SPDX (ISO 표준)                                                │
│  ├── CycloneDX (OWASP)                                             │
│  └── Syft (네이티브)                                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```bash
# ── Syft로 SBOM 생성 ─────────────────────────────────────────────
# 설치
brew install syft

# Docker 이미지에 대한 SBOM 생성
syft myapp:latest -o spdx-json > sbom.spdx.json
syft myapp:latest -o cyclonedx-json > sbom.cdx.json

# 디렉토리에 대한 SBOM 생성 (소스 코드)
syft dir:/path/to/project -o spdx-json

# ── Grype로 SBOM 취약점 스캔 ───────────────────────────────────
brew install grype

# SBOM에서 스캔
grype sbom:sbom.spdx.json

# 이미지 직접 스캔
grype myapp:latest

# high/critical에서 실패
grype myapp:latest --fail-on high
```

### 10.2 SLSA (Supply Chain Levels for Software Artifacts)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SLSA 레벨                                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  레벨 0: 보장 없음                                                    │
│  └── 누구나 무엇이든 빌드 가능, 출처 없음                              │
│                                                                      │
│  레벨 1: 빌드 프로세스 문서화                                          │
│  ├── 출처 존재 (누가, 어떻게 빌드했는지)                              │
│  └── 빌드 프로세스가 문서화됨                                          │
│                                                                      │
│  레벨 2: 빌드 서비스의 변조 저항                                       │
│  ├── 출처가 서명됨                                                     │
│  ├── 빌드 서비스가 출처를 자동으로 생성                                │
│  └── 빌드 서비스가 버전 관리됨                                         │
│                                                                      │
│  레벨 3: 특정 위협에 대한 추가 저항                                    │
│  ├── 출처가 위조 불가능                                                │
│  ├── 격리되고 일시적인 빌드 환경                                       │
│  └── 소스가 검증된 히스토리로 버전 관리됨                              │
│                                                                      │
│  대부분의 프로젝트는 최소 SLSA 레벨 2를 목표로 해야 합니다.             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```yaml
# ── GitHub Actions SLSA 출처 생성기 ────────────────────────────
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

## 11. 연습 문제

### 연습 문제 1: 안전한 Dockerfile 챌린지

다음의 안전하지 않은 Dockerfile을 가져와 모든 보안 문제를 수정하세요:

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

요구사항:
1. 최소 베이스 이미지 사용
2. 다단계 빌드
3. 비root 사용자
4. 이미지에 secrets 없음
5. 적절한 .dockerignore
6. 헬스 체크
7. 읽기 전용 파일시스템
8. 고정된 버전

### 연습 문제 2: Kubernetes 보안 강화

다음을 포함한 웹 애플리케이션을 위한 완전한 Kubernetes 매니페스트 작성:

1. 보안 강화된 pod 스펙이 있는 Deployment
2. 적절한 타입의 Service
3. 인그레스 컨트롤러로부터의 인그레스와 API 서비스 및 DNS로의 이그레스만 허용하는 NetworkPolicy
4. 기본 토큰 마운팅 없는 ServiceAccount
5. "developer" 그룹을 위한 읽기 전용 액세스 Role 및 RoleBinding
6. 네임스페이스에 Pod Security Standards 강제

### 연습 문제 3: 컨테이너 이미지 스캐너

다음을 수행하는 Python 도구 구축:

1. Docker 이미지 풀
2. Trivy 스캔 실행 (또는 Trivy JSON 출력 파싱)
3. critical/high 취약점 확인
4. 이미지가 서명되었는지 검증 (cosign)
5. 준수 보고서 생성 (통과/실패 및 이유)
6. CI/CD 게이트로 사용 가능 (실패 시 종료 코드 1)

### 연습 문제 4: 클라우드 IAM 감사 도구

다음을 수행하는 AWS IAM 감사 도구 생성:

1. 모든 IAM 사용자와 첨부된 정책 목록
2. MFA 없는 사용자 식별
3. 90일 이상 오래된 액세스 키 찾기
4. AdministratorAccess가 있는 사용자 감지
5. 사용하지 않는 역할 식별 (90일간 가정되지 않음)
6. 지나치게 관대한 S3 버킷 정책 확인
7. 결과 및 개선 단계가 있는 CSV 보고서 생성

### 연습 문제 5: Terraform 보안 스캐너

다음을 위한 Terraform 구성 스캔 도구 작성:

1. 민감한 포트(22, 3389, 5432)에서 0.0.0.0/0가 있는 보안 그룹
2. 암호화 없는 S3 버킷
3. 공개 액세스 차단 없는 S3 버킷
4. 암호화 없는 RDS 인스턴스
5. 필수 태그 없는 리소스 (Environment, Owner, ManagedBy)
6. .tf 파일의 하드코딩된 자격 증명
7. CI/CD 게이팅에 적합한 형식으로 결과 출력

### 연습 문제 6: 공급망 보안 파이프라인

다음을 수행하는 CI/CD 파이프라인 설계 및 구현:

1. 취약점에 대한 소스 코드 스캔 (SAST)
2. 알려진 CVE에 대한 의존성 스캔
3. 다단계 Dockerfile로 컨테이너 이미지 빌드
4. Trivy로 빌드된 이미지 스캔
5. cosign으로 이미지 서명
6. SBOM 생성 (SPDX 형식)
7. 이미지에 SBOM 첨부
8. 모든 확인이 통과할 때만 배포

---

## 요약

### 컨테이너 보안 체크리스트

| 카테고리 | 제어 | 우선순위 |
|---------|------|----------|
| 이미지 | 최소 베이스 이미지 사용 (distroless/Alpine) | Critical |
| 이미지 | 취약점 스캔 (Trivy/Snyk) | Critical |
| 이미지 | 이미지에 secrets 없음 | Critical |
| 이미지 | 베이스 이미지 digest 고정 | High |
| 이미지 | 이미지 서명 (cosign) | High |
| Dockerfile | 비root로 실행 | Critical |
| Dockerfile | 다단계 빌드 | High |
| Dockerfile | 모든 권한 삭제 | High |
| Dockerfile | 읽기 전용 파일시스템 | Medium |
| Kubernetes | 최소 권한 RBAC | Critical |
| Kubernetes | Network Policies | Critical |
| Kubernetes | Pod Security Standards | High |
| Kubernetes | 기본 서비스 계정 토큰 없음 | High |
| 클라우드 | 최소 권한 IAM | Critical |
| 클라우드 | 장기 자격 증명 없음 | High |
| 클라우드 | 모든 계정에 MFA | Critical |
| 클라우드 | 상태 파일 암호화 | High |
| 공급망 | SBOM 생성 | High |
| 공급망 | 의존성 스캐닝 | High |
| 런타임 | Falco/Tetragon으로 모니터링 | Medium |

### 핵심 요점

1. **최소 이미지가 공격 표면 감소** — 모든 패키지는 잠재적 취약점; 가능하면 distroless 또는 scratch 사용
2. **절대 root로 실행하지 말 것** — 모든 Dockerfile에서 전용 사용자 생성
3. **서명 및 검증** — cosign을 사용하여 이미지를 서명하고 배포 전 검증
4. **네트워크 격리가 중요** — 기본 거부 NetworkPolicies는 측면 이동 방지
5. **왼쪽으로 이동** — 프로덕션뿐만 아니라 CI/CD에서 이미지, IaC, 의존성 스캔
6. **클라우드 IAM이 새로운 경계** — 방화벽 규칙과 동일한 엄격함으로 IAM 정책 다루기
7. **모든 것 자동화** — 수동 보안 검사는 확장되지 않음; 파이프라인에 통합

---

**이전**: [11_Secrets_Management.md](./11_Secrets_Management.md) | **다음**: [13. Security Testing](./13_Security_Testing.md)
