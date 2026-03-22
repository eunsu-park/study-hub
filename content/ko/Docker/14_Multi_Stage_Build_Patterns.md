# 14. 멀티 스테이지 빌드 패턴(Multi-Stage Build Patterns)

**이전**: [영구 볼륨](./13_Persistent_Volumes.md) | **다음**: [Podman과 OCI](./15_Podman_and_OCI.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 멀티 스테이지 빌드의 기본 원리와 더 작고 안전한 이미지를 만드는 이유를 설명한다
2. Go, Rust, C++ 등 컴파일 언어에 빌더 패턴(Builder Pattern)을 적용한다
3. 선택적 COPY를 사용하여 Docker 이미지 크기를 최적화한다
4. BuildKit 캐시 마운트 및 기타 성능 향상 기능을 활용한다
5. 최소 프로덕션 이미지를 위한 distroless 및 scratch 베이스 이미지를 사용한다
6. 빌드 인수(Build Arguments), 타겟 스테이지, 조건부 빌드를 관리한다
7. Node.js, Python, Java 애플리케이션을 위한 실제 Dockerfile 패턴을 구현한다

## 목차
1. [멀티 스테이지 빌드 기초](#1-멀티-스테이지-빌드-기초)
2. [컴파일 언어를 위한 빌더 패턴](#2-컴파일-언어를-위한-빌더-패턴)
3. [이미지 크기 최적화](#3-이미지-크기-최적화)
4. [BuildKit 캐시 전략](#4-buildkit-캐시-전략)
5. [BuildKit 기능 및 향상](#5-buildkit-기능-및-향상)
6. [Distroless 및 Scratch 이미지](#6-distroless-및-scratch-이미지)
7. [빌드 인수와 타겟 스테이지](#7-빌드-인수와-타겟-스테이지)
8. [실제 Dockerfile 패턴](#8-실제-dockerfile-패턴)
9. [고급 패턴](#9-고급-패턴)
10. [연습 문제](#10-연습-문제)

**난이도**: ⭐⭐⭐

---

멀티 스테이지 빌드 이전에 개발자들은 딜레마에 직면했습니다: 빌드 도구, 컴파일러, 소스 코드가 포함된 큰 이미지를 배포하거나, Docker 외부에서 아티팩트를 빌드하고 복사하는 복잡한 스크립트를 유지해야 했습니다. 멀티 스테이지 빌드는 단일 Dockerfile에서 여러 `FROM` 지시어를 허용하여 이 문제를 우아하게 해결합니다. 각 지시어는 새로운 빌드 스테이지를 시작합니다. 최종 스테이지만 배포되는 이미지가 되고, 중간 스테이지는 빌드 도구와 아티팩트를 제공한 후 폐기됩니다.

---

## 1. 멀티 스테이지 빌드 기초

### 문제점: 비대한 이미지(Fat Images)

Go 애플리케이션을 위한 전통적인 단일 스테이지 Dockerfile:

```dockerfile
# 단일 스테이지: 800MB+ 이미지
FROM golang:1.22
WORKDIR /app
COPY . .
RUN go build -o server .
CMD ["./server"]
```

이 이미지는 전체 Go 툴체인, 소스 코드, 모든 빌드 종속성을 포함합니다. 이 중 어느 것도 런타임에 필요하지 않습니다.

### 해결책: 멀티 스테이지 빌드

```dockerfile
# Stage 1: 빌드
FROM golang:1.22 AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o server .

# Stage 2: 런타임 (최종 이미지)
FROM alpine:3.19
RUN apk --no-cache add ca-certificates
WORKDIR /app
COPY --from=builder /app/server .
CMD ["./server"]
```

```
┌─────────────────────────────────────────────────────────────┐
│                멀티 스테이지 빌드 흐름                         │
│                                                              │
│  Stage 1 (builder)              Stage 2 (final)             │
│  ┌─────────────────────┐       ┌─────────────────────┐     │
│  │ golang:1.22          │       │ alpine:3.19          │     │
│  │ ┌─────────────────┐ │       │ ┌─────────────────┐ │     │
│  │ │ Go 툴체인        │ │       │ │ ca-certificates  │ │     │
│  │ │ 소스 코드        │ │       │ │                  │ │     │
│  │ │ 종속성           │ │  COPY │ │ server 바이너리   │ │     │
│  │ │ ─────────────── │ │──────►│ │ (builder에서)    │ │     │
│  │ │ server 바이너리 ★│ │       │ └─────────────────┘ │     │
│  │ └─────────────────┘ │       │                      │     │
│  │ 크기: ~800MB         │       │ 크기: ~15MB          │     │
│  └─────────────────────┘       └─────────────────────┘     │
│       ↓ 폐기                        ↓ 배포                  │
└─────────────────────────────────────────────────────────────┘
```

### 핵심 개념

- 각 `FROM` 지시어는 새로운 스테이지를 시작한다
- 스테이지에 `AS <이름>`으로 이름을 지정할 수 있다
- `COPY --from=<스테이지>`로 이전 스테이지에서 파일을 복사한다
- 최종 스테이지만 빌드된 이미지에 포함된다
- 중간 스테이지는 캐시되지만 배포되지 않는다

---

## 2. 컴파일 언어를 위한 빌더 패턴

### Go

```dockerfile
FROM golang:1.22-alpine AS builder
RUN apk add --no-cache git
WORKDIR /app

# 종속성 캐시
COPY go.mod go.sum ./
RUN go mod download

# 빌드
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-w -s" -o /app/server ./cmd/server

# 최종 이미지
FROM scratch
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /app/server /server
ENTRYPOINT ["/server"]
```

결과: 정적 바이너리와 CA 인증서만 포함하는 ~5-10MB 이미지.

### Rust

```dockerfile
FROM rust:1.77 AS builder
WORKDIR /app

# 더미 빌드로 종속성 캐시
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src

# 실제 애플리케이션 빌드
COPY src ./src
RUN touch src/main.rs  # 재빌드 보장
RUN cargo build --release

# 최종 이미지
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/myapp /usr/local/bin/
CMD ["myapp"]
```

### C++

```dockerfile
FROM gcc:13 AS builder
WORKDIR /app
COPY . .
RUN mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc)

# 최종 이미지
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/build/myapp /usr/local/bin/
CMD ["myapp"]
```

---

## 3. 이미지 크기 최적화

### 레이어 최적화(Layer Optimization)

```dockerfile
# 나쁜 예: 각 RUN이 별도의 레이어를 생성
RUN apt-get update
RUN apt-get install -y curl wget git
RUN apt-get clean

# 좋은 예: 정리를 포함한 단일 레이어
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl wget git \
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/*
```

### 선택적 COPY(Selective COPY)

```dockerfile
FROM node:20 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# 프로덕션에 필요한 것만 복사
FROM node:20-slim
WORKDIR /app
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
CMD ["node", "dist/index.js"]
```

### 이미지 크기 비교

```
┌─────────────────────────────────────────────────────────────┐
│            이미지 크기 비교 (Go 앱)                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  golang:1.22         ███████████████████████████████  ~820MB │
│  golang:1.22-alpine  █████████████                   ~260MB │
│  alpine + binary     █                               ~15MB  │
│  distroless          ▌                               ~8MB   │
│  scratch + binary    ▏                               ~5MB   │
│                                                              │
│  ──────────────────────────────────────────────────────────  │
│  핵심: 전체 이미지에서 scratch까지 99% 감소                    │
└─────────────────────────────────────────────────────────────┘
```

### .dockerignore

빌드 컨텍스트에서 불필요한 파일을 제외하려면 항상 `.dockerignore`를 사용하세요:

```
# .dockerignore
.git
.gitignore
node_modules
*.md
LICENSE
.env
.env.*
docker-compose*.yml
Dockerfile
.dockerignore
__pycache__
*.pyc
.pytest_cache
.coverage
dist
build
```

---

## 4. BuildKit 캐시 전략

### 종속성 캐시 마운트(Dependency Cache Mounts)

캐시 마운트는 빌드 간에 패키지 관리자 캐시를 유지하여 재빌드 속도를 크게 향상시킵니다:

```dockerfile
# syntax=docker/dockerfile:1

# pip 캐시가 있는 Python
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt

FROM python:3.12-slim
COPY --from=builder /install /usr/local
COPY . /app
CMD ["python", "/app/main.py"]
```

```dockerfile
# npm 캐시가 있는 Node.js
FROM node:20 AS builder
WORKDIR /app
COPY package*.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci
COPY . .
RUN npm run build
```

```dockerfile
# 모듈 캐시가 있는 Go
FROM golang:1.22 AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download
COPY . .
RUN --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go build -o /app/server .
```

### 시크릿 마운트(Secret Mounts)

레이어에 포함하지 않고 빌드 중에 시크릿을 안전하게 전달합니다:

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim
RUN --mount=type=secret,id=pip_conf,target=/etc/pip.conf \
    pip install private-package
```

```bash
# 시크릿으로 빌드
docker build --secret id=pip_conf,src=./pip.conf .
```

---

## 5. BuildKit 기능 및 향상

### BuildKit 활성화

```bash
# 환경 변수
export DOCKER_BUILDKIT=1
docker build .

# 또는 docker buildx 사용
docker buildx build .

# Docker Compose
COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose build
```

### 병렬 스테이지 빌드(Parallel Stage Building)

BuildKit은 독립적인 스테이지를 자동으로 병렬로 빌드합니다:

```dockerfile
# 이 스테이지들은 병렬로 빌드됨
FROM golang:1.22 AS backend-builder
WORKDIR /app
COPY backend/ .
RUN go build -o /backend ./cmd/server

FROM node:20 AS frontend-builder
WORKDIR /app
COPY frontend/ .
RUN npm ci && npm run build

# 최종 스테이지는 두 스테이지의 아티팩트를 사용
FROM nginx:alpine
COPY --from=backend-builder /backend /usr/local/bin/
COPY --from=frontend-builder /app/dist /usr/share/nginx/html
```

```
┌─────────────────────────────────────────────────────────────┐
│              BuildKit 병렬 빌드                               │
│                                                              │
│  시간 ──────────────────────────────────────────────►        │
│                                                              │
│  BuildKit 없이 (순차):                                       │
│  [backend-builder ████████████][frontend-builder ████████]   │
│  총: ~8분                                                    │
│                                                              │
│  BuildKit 사용 (병렬):                                       │
│  [backend-builder  ████████████]                             │
│  [frontend-builder ████████    ]                             │
│  총: ~5분                                                    │
└─────────────────────────────────────────────────────────────┘
```

### 빌드 출력 모드

```bash
# 일반 텍스트 출력 (CI 로그용)
docker buildx build --progress=plain .

# TTY 출력 (기본, 대화형)
docker buildx build --progress=auto .

# OCI tarball로 빌드 내보내기
docker buildx build --output type=oci,dest=image.tar .

# 이미지 생성 없이 파일시스템 내보내기
docker buildx build --output type=local,dest=./output .
```

---

## 6. Distroless 및 Scratch 이미지

### scratch

`scratch`는 특별한 빈 이미지입니다 -- 절대적으로 최소한의 베이스:

```dockerfile
FROM golang:1.22 AS builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -o /server .

# scratch = 빈 이미지, 아무것도 없음
FROM scratch
COPY --from=builder /server /server
ENTRYPOINT ["/server"]
```

scratch의 제한사항:
- 셸 없음 (`docker exec sh` 불가)
- 패키지 관리자 없음
- 타임존 데이터, CA 인증서 없음 (수동으로 복사해야 함)
- 사용자 관리 없음 (`/etc/passwd` 없음)

### Distroless (Google)

Distroless 이미지는 애플리케이션 런타임만 포함합니다 -- 셸, 패키지 관리자, OS 유틸리티가 없습니다:

```dockerfile
FROM golang:1.22 AS builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -o /server .

# Distroless: CA 인증서, 타임존 데이터가 포함된 최소 런타임
FROM gcr.io/distroless/static-debian12
COPY --from=builder /server /server
USER nonroot:nonroot
ENTRYPOINT ["/server"]
```

사용 가능한 distroless 이미지:
- `gcr.io/distroless/static-debian12` -- 정적 링크 바이너리 (Go)
- `gcr.io/distroless/base-debian12` -- 동적 링크 바이너리 (Rust, C++)
- `gcr.io/distroless/cc-debian12` -- libstdc++ 필요
- `gcr.io/distroless/java21-debian12` -- Java 애플리케이션
- `gcr.io/distroless/python3-debian12` -- Python 애플리케이션
- `gcr.io/distroless/nodejs22-debian12` -- Node.js 애플리케이션

### scratch에서 비루트(Non-Root) 사용자 생성

```dockerfile
FROM golang:1.22 AS builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -o /server .

# 최소 passwd 파일 생성
RUN echo "appuser:x:10001:10001::/nonexistent:/sbin/nologin" > /etc/passwd.app

FROM scratch
COPY --from=builder /etc/passwd.app /etc/passwd
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /server /server
USER appuser
ENTRYPOINT ["/server"]
```

---

## 7. 빌드 인수와 타겟 스테이지

### 빌드 인수(Build Arguments)

```dockerfile
# 베이스 이미지를 매개변수화
ARG GO_VERSION=1.22
FROM golang:${GO_VERSION} AS builder
ARG APP_VERSION=dev
WORKDIR /app
COPY . .
RUN go build -ldflags="-X main.version=${APP_VERSION}" -o /server .

FROM alpine:3.19
COPY --from=builder /server /server
CMD ["/server"]
```

```bash
# 기본값 재정의
docker build \
  --build-arg GO_VERSION=1.21 \
  --build-arg APP_VERSION=1.2.3 \
  -t myapp:1.2.3 .
```

### 타겟 스테이지(Target Stages)

특정 스테이지까지만 빌드:

```dockerfile
FROM node:20 AS base
WORKDIR /app
COPY package*.json ./

FROM base AS dependencies
RUN npm ci

FROM dependencies AS test
COPY . .
RUN npm test

FROM dependencies AS build
COPY . .
RUN npm run build

FROM node:20-slim AS production
WORKDIR /app
COPY --from=build /app/dist ./dist
COPY --from=dependencies /app/node_modules ./node_modules
CMD ["node", "dist/index.js"]
```

```bash
# test 스테이지만 빌드
docker build --target test -t myapp:test .

# production 스테이지만 빌드 (기본: 마지막 스테이지)
docker build --target production -t myapp:prod .

# 개발용 빌드
docker build --target dependencies -t myapp:dev .
```

### 조건부 스테이지(Conditional Stages)

```dockerfile
ARG BUILD_ENV=production

FROM node:20 AS base
WORKDIR /app
COPY package*.json ./
RUN npm ci

FROM base AS development
COPY . .
CMD ["npm", "run", "dev"]

FROM base AS production-build
COPY . .
RUN npm run build

FROM node:20-slim AS production
WORKDIR /app
COPY --from=production-build /app/dist ./dist
COPY --from=production-build /app/node_modules ./node_modules
CMD ["node", "dist/index.js"]
```

---

## 8. 실제 Dockerfile 패턴

### Node.js (풀스택 애플리케이션)

```dockerfile
# syntax=docker/dockerfile:1
FROM node:20-alpine AS base
RUN apk add --no-cache libc6-compat
WORKDIR /app

# 종속성 설치
FROM base AS deps
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --only=production && \
    cp -R node_modules /prod_modules && \
    npm ci

# 빌드
FROM base AS build
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# 프로덕션
FROM base AS production
ENV NODE_ENV=production
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 appuser
COPY --from=build /app/dist ./dist
COPY --from=deps /prod_modules ./node_modules
USER appuser
EXPOSE 3000
CMD ["node", "dist/index.js"]
```

### Python (FastAPI 애플리케이션)

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

# 빌드 스테이지: 종속성 컴파일
FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install --no-warn-script-location \
    -r requirements.txt

# 프로덕션
FROM base AS production
COPY --from=builder /install /usr/local
RUN useradd --create-home --shell /bin/bash appuser
COPY --chown=appuser:appuser . .
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Java (Spring Boot 애플리케이션)

```dockerfile
# syntax=docker/dockerfile:1
FROM eclipse-temurin:21-jdk AS builder
WORKDIR /app
COPY gradle/ gradle/
COPY gradlew build.gradle.kts settings.gradle.kts ./
RUN --mount=type=cache,target=/root/.gradle \
    ./gradlew dependencies --no-daemon
COPY src ./src
RUN --mount=type=cache,target=/root/.gradle \
    ./gradlew bootJar --no-daemon

# 더 나은 캐싱을 위한 Spring Boot 레이어 추출
FROM eclipse-temurin:21-jdk AS extractor
WORKDIR /app
COPY --from=builder /app/build/libs/*.jar app.jar
RUN java -Djarmode=layertools -jar app.jar extract

# 프로덕션
FROM eclipse-temurin:21-jre-alpine
RUN addgroup --system spring && adduser --system --ingroup spring spring
WORKDIR /app
COPY --from=extractor /app/dependencies/ ./
COPY --from=extractor /app/spring-boot-loader/ ./
COPY --from=extractor /app/snapshot-dependencies/ ./
COPY --from=extractor /app/application/ ./
USER spring
EXPOSE 8080
ENTRYPOINT ["java", "org.springframework.boot.loader.launch.JarLauncher"]
```

---

## 9. 고급 패턴

### 크로스 컴파일(Cross-Compilation)

```dockerfile
# 여러 아키텍처용 빌드
FROM --platform=$BUILDPLATFORM golang:1.22 AS builder
ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
    go build -o /server .

FROM alpine:3.19
COPY --from=builder /server /server
CMD ["/server"]
```

```bash
# 여러 플랫폼용 빌드
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t myapp:latest \
  --push .
```

### 외부 이미지에서 복사(Copying from External Images)

```dockerfile
FROM nginx:alpine

# 완전히 별도의 이미지에서 바이너리 복사
COPY --from=busybox:latest /bin/wget /usr/local/bin/wget

# 커스텀 이미지에서 설정 복사
COPY --from=mycompany/nginx-config:latest /etc/nginx/nginx.conf /etc/nginx/
```

### 멀티 스테이지 빌드에서의 테스트

```dockerfile
FROM python:3.12-slim AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# 테스트 스테이지
FROM base AS test
RUN pip install pytest pytest-cov
RUN pytest --cov=app tests/

# 린트 스테이지
FROM base AS lint
RUN pip install ruff
RUN ruff check .

# 프로덕션 (--target으로 test와 lint가 통과한 경우에만 빌드)
FROM base AS production
USER nobody
CMD ["python", "main.py"]
```

---

## 10. 연습 문제

### 연습 1: 기본 멀티 스테이지 빌드 (초급)

이 단일 스테이지 Dockerfile을 멀티 스테이지 빌드로 변환하세요:

```dockerfile
FROM python:3.12
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python -m pytest tests/
CMD ["python", "app.py"]
```

<details>
<summary>풀이</summary>

```dockerfile
# 테스트 스테이지
FROM python:3.12-slim AS test
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python -m pytest tests/

# 프로덕션 스테이지
FROM python:3.12-slim AS production
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY src/ ./src/
USER nobody
CMD ["python", "app.py"]
```

</details>

### 연습 2: Scratch를 사용한 Go 애플리케이션 (중급)

다음을 만족하는 Go 웹 서버용 멀티 스테이지 Dockerfile을 작성하세요:
- 정적 링크 바이너리를 빌드
- scratch를 최종 이미지로 사용
- HTTPS를 위한 CA 인증서 포함
- 비루트(non-root) 사용자로 실행

<details>
<summary>풀이</summary>

```dockerfile
FROM golang:1.22-alpine AS builder
RUN apk add --no-cache ca-certificates
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-w -s" -o /server .
RUN echo "appuser:x:10001:10001::/nonexistent:/sbin/nologin" > /etc/passwd.min

FROM scratch
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /etc/passwd.min /etc/passwd
COPY --from=builder /server /server
USER appuser
EXPOSE 8080
ENTRYPOINT ["/server"]
```

</details>

### 연습 3: 풀스택 빌드 (고급)

다음을 포함하는 애플리케이션용 멀티 스테이지 Dockerfile을 만드세요:
- React 프론트엔드 (npm build)
- Go 백엔드 API
- 프론트엔드를 서빙하고 API 요청을 프록시하는 Nginx
- 모든 것이 병렬 스테이지로 빌드

<details>
<summary>풀이</summary>

```dockerfile
# syntax=docker/dockerfile:1

# Stage 1: 프론트엔드 빌드
FROM node:20-alpine AS frontend
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# Stage 2: 백엔드 빌드 (Stage 1과 병렬 실행)
FROM golang:1.22-alpine AS backend
WORKDIR /app
COPY backend/go.mod backend/go.sum ./
RUN go mod download
COPY backend/ .
RUN CGO_ENABLED=0 go build -o /api-server .

# Stage 3: 최종 이미지
FROM nginx:alpine
# 프론트엔드 빌드 복사
COPY --from=frontend /app/dist /usr/share/nginx/html
# 백엔드 바이너리 복사
COPY --from=backend /api-server /usr/local/bin/
# API 프록시가 포함된 nginx 설정 복사
COPY nginx.conf /etc/nginx/nginx.conf
# 시작 스크립트
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 80
ENTRYPOINT ["/entrypoint.sh"]
```

</details>

### 연습 4: 캐시 최적화 (고급)

이 Dockerfile을 BuildKit 캐시 마운트, 적절한 레이어 순서, 멀티 스테이지 빌드를 사용하여 가능한 가장 빠른 재빌드 시간을 달성하도록 최적화하세요:

```dockerfile
FROM python:3.12
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN python -m pytest
RUN python setup.py bdist_wheel
CMD ["python", "-m", "myapp"]
```

<details>
<summary>풀이</summary>

```dockerfile
# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base
WORKDIR /app

# 종속성 (requirements가 변경되지 않으면 캐시됨)
FROM base AS deps
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt

# 테스트 스테이지
FROM base AS test
COPY --from=deps /install /usr/local
COPY . .
RUN python -m pytest

# 빌드 wheel
FROM base AS build
COPY --from=deps /install /usr/local
COPY . .
RUN python setup.py bdist_wheel

# 프로덕션
FROM python:3.12-slim AS production
COPY --from=build /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl
USER nobody
CMD ["python", "-m", "myapp"]
```

</details>

---

**이전**: [영구 볼륨](./13_Persistent_Volumes.md) | **다음**: [Podman과 OCI](./15_Podman_and_OCI.md)
