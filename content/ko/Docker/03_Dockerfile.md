# Dockerfile

**이전**: [이미지와 컨테이너](./02_Images_and_Containers.md) | **다음**: [Docker Compose](./04_Docker_Compose.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Dockerfile이 무엇인지, 그리고 재현 가능하고 버전 관리 가능한 이미지 빌드를 어떻게 제공하는지 설명할 수 있다
2. FROM, WORKDIR, COPY, RUN, CMD, EXPOSE, ENV와 같은 핵심 명령어를 사용하여 Dockerfile을 작성할 수 있다
3. CMD와 ENTRYPOINT의 차이, 그리고 COPY와 ADD의 차이를 구분할 수 있다
4. 멀티 스테이지 빌드(multi-stage build)를 적용하여 빌드 환경과 런타임 환경을 분리하고 이미지 크기를 줄일 수 있다
5. .dockerignore, 레이어 캐싱(layer caching), 작은 베이스 이미지, 비루트 사용자(non-root user) 사용 등의 베스트 프랙티스(best practice)를 적용할 수 있다
6. 태그(tag), 빌드 인자(build argument), 캐시 제어를 사용하여 `docker build`로 Docker 이미지를 빌드할 수 있다

---

Docker Hub에서 사전 빌드된 이미지를 받아 사용하는 것은 편리하지만, 실제 프로젝트에서는 특정 애플리케이션과 의존성에 맞춤화된 커스텀 이미지가 필요합니다. Dockerfile은 이러한 커스텀 이미지를 코드로 정의하는 표준 메커니즘(mechanism)입니다. Dockerfile 문법과 멀티 스테이지 빌드 및 레이어 최적화와 같은 베스트 프랙티스를 익히면 애플리케이션 패키징을 완전히 제어하고, 일관되고 안전하며 효율적인 컨테이너 이미지를 보장할 수 있습니다.

## 1. Dockerfile이란?

Dockerfile은 Docker 이미지를 만들기 위한 **설정 파일**입니다. 텍스트 파일에 명령어를 작성하면 Docker가 순서대로 실행하여 이미지를 생성합니다.

```
Dockerfile → docker build → Docker Image → docker run → Container
(Blueprint)    (Build)       (Template)      (Run)      (Instance)
```

### 왜 Dockerfile을 사용할까요?

| 장점 | 설명 |
|------|------|
| **재현성** | 동일한 이미지를 반복 생성 |
| **자동화** | 수동 설정 불필요 |
| **버전 관리** | Git으로 이력 추적 |
| **문서화** | 환경 설정이 코드로 기록 |

---

## 2. Dockerfile 기본 문법

### 기본 구조

```dockerfile
# Comment
INSTRUCTION argument
```

### 주요 명령어

| 명령어 | 설명 | 예시 |
|--------|------|------|
| `FROM` | 베이스 이미지 | `FROM node:18` |
| `WORKDIR` | 작업 디렉토리 | `WORKDIR /app` |
| `COPY` | 파일 복사 | `COPY . .` |
| `RUN` | 빌드 시 명령 실행 | `RUN npm install` |
| `CMD` | 컨테이너 시작 명령 | `CMD ["npm", "start"]` |
| `EXPOSE` | 포트 노출 | `EXPOSE 3000` |
| `ENV` | 환경 변수 | `ENV NODE_ENV=production` |

---

## 3. 명령어 상세 설명

### FROM - 베이스 이미지

모든 Dockerfile은 `FROM`으로 시작합니다.

```dockerfile
# Basic
FROM ubuntu:22.04

# Node.js image
FROM node:18

# Alpine: ~175 MB vs ~1 GB full image — smaller attack surface and faster CI pulls
FROM node:18-alpine

# Multi-stage build — build tools stay in 'builder', excluded from final image
FROM node:18 AS builder
FROM nginx:alpine AS production
```

### WORKDIR - 작업 디렉토리

이후 명령어가 실행될 디렉토리를 설정합니다.

```dockerfile
WORKDIR /app

# Subsequent commands execute in /app
COPY . .          # Copy to /app
RUN npm install   # Execute in /app
```

### COPY - 파일 복사

호스트의 파일을 이미지로 복사합니다.

```dockerfile
# Copy file
COPY package.json .

# Copy directory
COPY src/ ./src/

# Copy all files
COPY . .

# Copy multiple files
COPY package.json package-lock.json ./
```

### ADD vs COPY

```dockerfile
# COPY: Simple copy (recommended)
COPY local-file.txt /app/

# ADD: URL download, archive extraction
ADD https://example.com/file.tar.gz /app/
ADD archive.tar.gz /app/  # Auto-extracts
```

### RUN - 빌드 시 명령 실행

이미지 빌드 중에 실행됩니다.

```dockerfile
# Basic
RUN npm install

# Combine in one RUN so the apt cache never persists in a committed layer
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*  # Remove apt cache; must be in same RUN to avoid bloating the image

# Layer caching: copy dependency manifest first (changes rarely), then install, then copy source (changes often)
COPY package*.json ./   # Dependency manifest only — changes less often than source code
RUN npm install         # Cached as long as package*.json is unchanged
COPY . .                # Source changes every build; placed last to preserve npm install cache
```

### CMD - 컨테이너 시작 명령

컨테이너가 시작될 때 실행됩니다.

```dockerfile
# exec form (recommended) — no shell wrapper, so the process receives
# OS signals (e.g., SIGTERM) directly for graceful shutdown
CMD ["npm", "start"]
CMD ["node", "app.js"]

# shell form — runs via /bin/sh -c; process won't receive signals directly
CMD npm start
```

### ENTRYPOINT vs CMD

```dockerfile
# ENTRYPOINT = fixed command, CMD = overridable default argument
ENTRYPOINT ["node"]
CMD ["app.js"]           # Default arg; override with: docker run myimage other.js
# Executes: node app.js

# docker run myimage other.js
# Executes: node other.js (ENTRYPOINT stays, only CMD is replaced)
```

### ENV - 환경 변수

```dockerfile
# Single variable
ENV NODE_ENV=production

# Multiple variables
ENV NODE_ENV=production \
    PORT=3000 \
    DB_HOST=localhost
```

### EXPOSE - 포트 문서화

```dockerfile
# EXPOSE is documentation only — does not actually publish the port (use -p at runtime for that)
EXPOSE 3000
EXPOSE 80 443
```

### ARG - 빌드 시 변수

```dockerfile
# ARG: available only at build time — use for values that should not persist in the running container
ARG NODE_VERSION=18
FROM node:${NODE_VERSION}

# Promote ARG to ENV so the value is available at runtime too (e.g., for version endpoints)
ARG APP_VERSION=1.0.0
ENV APP_VERSION=${APP_VERSION}
```

```bash
# Pass value during build
docker build --build-arg NODE_VERSION=20 .
```

---

## 4. 실습 예제

### 예제 1: Node.js 애플리케이션

**프로젝트 구조:**
```
my-node-app/
├── Dockerfile
├── package.json
└── app.js
```

**package.json:**
```json
{
  "name": "my-node-app",
  "version": "1.0.0",
  "main": "app.js",
  "scripts": {
    "start": "node app.js"
  },
  "dependencies": {
    "express": "^4.18.2"
  }
}
```

**app.js:**
```javascript
const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;

app.get('/', (req, res) => {
  res.json({ message: 'Hello from Docker!', version: '1.0.0' });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

**Dockerfile:**
```dockerfile
# Base image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy dependency manifest first — changes less often than source code
COPY package*.json ./

# Install deps — this layer is cached as long as package*.json hasn't changed
RUN npm install

# Copy source last — changes every build, so it doesn't invalidate npm install cache
COPY . .

# Document the port this app listens on (actual mapping done with -p at runtime)
EXPOSE 3000

# exec form: process receives OS signals directly (needed for graceful shutdown)
CMD ["npm", "start"]
```

**빌드 및 실행:**
```bash
# Build image
docker build -t my-node-app .

# Run container
docker run -d -p 3000:3000 --name node-app my-node-app

# Test
curl http://localhost:3000

# Cleanup
docker rm -f node-app
```

### 예제 2: Python Flask 애플리케이션

**프로젝트 구조:**
```
my-flask-app/
├── Dockerfile
├── requirements.txt
└── app.py
```

**requirements.txt:**
```
flask==3.0.0
gunicorn==21.2.0
```

**app.py:**
```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    return jsonify(message='Hello from Flask in Docker!')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt  # --no-cache-dir: skip storing downloaded packages in the layer

# Copy source
COPY . .

EXPOSE 5000

# Run with Gunicorn (production)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

**빌드 및 실행:**
```bash
docker build -t my-flask-app .
docker run -d -p 5000:5000 my-flask-app
curl http://localhost:5000
```

### 예제 3: 정적 웹사이트 (Nginx)

**프로젝트 구조:**
```
my-website/
├── Dockerfile
├── nginx.conf
└── public/
    └── index.html
```

**public/index.html:**
```html
<!DOCTYPE html>
<html>
<head>
    <title>My Docker Website</title>
</head>
<body>
    <h1>Hello from Nginx in Docker!</h1>
</body>
</html>
```

**Dockerfile:**
```dockerfile
# Alpine: ~5 MB base — ideal for serving static files with minimal overhead
FROM nginx:alpine

# Copy custom config (optional)
# COPY nginx.conf /etc/nginx/nginx.conf

# Copy static files
COPY public/ /usr/share/nginx/html/

EXPOSE 80

# "daemon off;" keeps nginx in the foreground so Docker can track the process as PID 1
CMD ["nginx", "-g", "daemon off;"]
```

---

## 5. 멀티 스테이지 빌드

빌드 환경과 실행 환경을 분리하여 이미지 크기를 줄입니다.

### React 앱 예시

```dockerfile
# Stage 1: Build — node_modules + build toolchain (~300 MB) are discarded after this stage
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Stage 2: Runtime — only the static build output is copied; final image is ~25 MB
FROM nginx:alpine

# --from=builder: pull artifacts from the build stage without carrying over node_modules
COPY --from=builder /app/build /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Go 앱 예시

```dockerfile
# Stage 1: Build — Go compiler + stdlib needed only at compile time
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY . .
# Static binary: no external C library deps, so the runtime stage needs almost nothing
RUN go build -o main .

# Stage 2: Runtime — alpine:latest is ~5 MB; use 'scratch' for even smaller (~0 MB base)
FROM alpine:latest

WORKDIR /app
COPY --from=builder /app/main .

EXPOSE 8080
CMD ["./main"]
```

**크기 비교:**
```
golang:1.21-alpine  →  ~300MB (build environment)
Final image         →  ~15MB (runtime environment)
```

---

## 6. 베스트 프랙티스

### .dockerignore 파일

불필요한 파일을 빌드에서 제외합니다.

```
# .dockerignore — reduces build context size and prevents secrets/large dirs from leaking into the image
node_modules
npm-debug.log
.git
.gitignore
.env
*.md
Dockerfile
.dockerignore
```

### 레이어 최적화

```dockerfile
# Bad: Copying everything first means ANY source change invalidates the npm install cache
COPY . .
RUN npm install

# Good: Copy manifest first — npm install is cached until package.json changes
COPY package*.json ./
RUN npm install
COPY . .   # Source changes don't trigger a reinstall
```

### 작은 이미지 사용

```dockerfile
# Large — full Debian with build tools; only needed if you compile native addons
FROM node:18           # ~1GB

# Recommended — Alpine Linux: ~5 MB base, minimal packages, smaller attack surface
FROM node:18-alpine    # ~175MB

# Minimal — Debian slim: smaller than full but includes glibc (better native addon compat than Alpine)
FROM node:18-slim      # ~200MB
```

### 보안

```dockerfile
# Run as non-root user (limits damage if container is compromised)
FROM node:18-alpine

# -S = system account (no home dir, no login shell) — appropriate for service processes
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
USER appuser

WORKDIR /app
COPY --chown=appuser:appgroup . .  # --chown ensures the non-root user can read the copied files
```

---

## 7. 이미지 빌드 명령어

```bash
# Basic build
docker build -t imagename .

# Specify tag
docker build -t myapp:1.0 .

# Use different Dockerfile
docker build -f Dockerfile.prod -t myapp:prod .

# Pass build arguments
docker build --build-arg NODE_ENV=production -t myapp .

# --no-cache: force rebuild all layers — useful when a base image or remote dep changed
docker build --no-cache -t myapp .

# --progress=plain: show full build output — easier to debug failed RUN steps
docker build --progress=plain -t myapp .
```

---

## 명령어 요약

| Dockerfile 명령어 | 설명 |
|------------------|------|
| `FROM` | 베이스 이미지 지정 |
| `WORKDIR` | 작업 디렉토리 설정 |
| `COPY` | 파일/디렉토리 복사 |
| `RUN` | 빌드 시 명령 실행 |
| `CMD` | 컨테이너 시작 명령 |
| `EXPOSE` | 포트 문서화 |
| `ENV` | 환경 변수 설정 |
| `ARG` | 빌드 시 변수 |
| `ENTRYPOINT` | 고정 실행 명령 |

---

## 연습 문제

### 연습 1: 첫 번째 Dockerfile 작성

간단한 Python Flask 애플리케이션을 위한 Dockerfile을 작성합니다.

1. 프로젝트 디렉토리를 만들고 다음 파일을 추가합니다:
   - `flask==3.0.0`을 포함하는 `requirements.txt`
   - 루트 경로(route)에서 `{"message": "Hello, Docker!"}`를 반환하는 Flask 앱 `app.py`
2. `python:3.11-slim`을 베이스 이미지(base image)로 사용하고, 비루트 사용자(non-root user)를 추가하며, 올바른 레이어(layer) 캐싱(caching)을 적용하는 `Dockerfile`을 작성합니다 (`app.py` 전에 `requirements.txt`를 복사)
3. 이미지 빌드: `docker build -t flask-hello:1.0 .`
4. 포트 5000에서 컨테이너 실행: `docker run -d -p 5000:5000 flask-hello:1.0`
5. `curl http://localhost:5000`으로 테스트하고 응답을 확인합니다

### 연습 2: 레이어(Layer) 캐싱(Caching) 실험

레이어 캐싱이 빌드 시간에 미치는 영향을 관찰합니다.

1. 먼저 모든 파일을 복사한 후 `npm install`을 실행하는 Node.js Dockerfile로 시작합니다:
   ```dockerfile
   FROM node:18-alpine
   WORKDIR /app
   COPY . .
   RUN npm install
   CMD ["node", "app.js"]
   ```
2. 빌드하고 (`docker build -t cache-test:bad .`) 빌드 시간을 기록합니다
3. Dockerfile을 수정하여 `package*.json`을 먼저 복사하고 `npm install`을 실행한 다음 나머지 파일을 복사하도록 합니다
4. 다시 빌드하고 (`docker build -t cache-test:good .`) 빌드 시간을 기록합니다
5. `app.js`만 수정한 후 두 버전을 모두 다시 빌드하여 각각 얼마나 캐시가 활용되는지 비교합니다

### 연습 3: 멀티 스테이지 빌드(Multi-Stage Build)

멀티 스테이지 빌드를 사용하여 이미지 크기를 줄입니다.

1. "Hello from Go!"를 출력하는 간단한 Go 프로그램(`main.go`)을 작성합니다
2. `golang:1.21-alpine`을 사용하는 단일 스테이지(single-stage) Dockerfile을 작성하고 빌드한 후 이미지 크기를 기록합니다
3. 멀티 스테이지 빌드로 재작성합니다: `golang:1.21-alpine`에서 컴파일하고 바이너리(binary)만 `FROM scratch` 또는 `alpine:latest`로 복사합니다
4. `docker images`로 단일 스테이지와 멀티 스테이지 이미지의 크기를 비교합니다
5. 멀티 스테이지 이미지가 올바르게 실행되는지 확인합니다

### 연습 4: CMD와 ENTRYPOINT 차이

실험을 통해 `CMD`와 `ENTRYPOINT`의 차이를 이해합니다.

1. `ENTRYPOINT ["echo"]`와 `CMD ["Hello, World!"]`를 가진 Dockerfile을 작성합니다
2. 빌드하고 실행하여 기본 출력을 확인합니다
3. 런타임에 CMD를 오버라이드(override)합니다: `docker run <이미지> "Goodbye, World!"` — 어떤 결과가 나타나나요?
4. ENTRYPOINT를 오버라이드해봅니다: `docker run --entrypoint /bin/sh <이미지>` — 어떤 차이가 있나요?
5. Dockerfile을 수정하여 `CMD ["echo", "Hello, World!"]`만 사용하도록 변경하고 (ENTRYPOINT 없이), 동일한 오버라이드를 시도합니다. 차이점을 문서화합니다.

### 연습 5: .dockerignore와 빌드 컨텍스트(Build Context)

`.dockerignore`를 사용하여 빌드 컨텍스트(build context)를 최적화합니다.

1. `node_modules/`, `.git/`, `.env`, 소스 파일을 포함하는 프로젝트를 만듭니다
2. `.dockerignore` 없이 빌드하고 `docker build --no-cache --progress=plain -t context-test .`를 실행하여 출력에서 빌드 컨텍스트 크기를 관찰합니다
3. `node_modules`, `.git`, `.env`, `*.log`를 제외하는 `.dockerignore` 파일을 작성합니다
4. 다시 빌드하여 빌드 컨텍스트 크기를 비교합니다
5. `docker build --no-cache --progress=plain -t context-test:optimized .`를 실행하여 컨텍스트가 더 작아졌는지 확인합니다

---

## 다음 단계

[Docker Compose](./04_Docker_Compose.md)에서 여러 컨테이너를 함께 관리하는 방법을 배워봅시다!
