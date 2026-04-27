# 16. 프로덕션 배포(Production Deployment)

**이전**: [인증 패턴](./15_Authentication_Patterns.md) | **다음**: [관찰 가능성](./17_Observability.md)

**난이도**: ⭐⭐⭐⭐

## 학습 목표

- 프로덕션 환경에 적합한 워커(worker) 설정으로 ASGI 및 WSGI 애플리케이션 서버(uvicorn, gunicorn)를 구성한다
- SSL/TLS 종료(termination), 로드 밸런싱(load balancing), 보안 헤더를 갖춘 역방향 프록시(reverse proxy)로 nginx를 설정한다
- 다단계 빌드(multi-stage build)와 다중 서비스 오케스트레이션(multi-service orchestration)을 위한 Docker Compose를 활용하여 백엔드 애플리케이션을 컨테이너화한다
- 무중단 배포(zero-downtime deployment)를 위한 헬스 체크(health check)와 그레이스풀 셧다운(graceful shutdown) 패턴을 구현한다
- 12-팩터 앱(12-Factor App) 방법론을 적용하여 설정과 환경 변수를 관리한다

## 목차

1. [ASGI 서버: Uvicorn과 Hypercorn](#1-asgi-서버-uvicorn과-hypercorn)
2. [WSGI 서버: Gunicorn](#2-wsgi-서버-gunicorn)
3. [Node.js를 위한 PM2](#3-nodejs를-위한-pm2)
4. [nginx 역방향 프록시](#4-nginx-역방향-프록시)
5. [Docker 컨테이너화](#5-docker-컨테이너화)
6. [다중 서비스 설정을 위한 Docker Compose](#6-다중-서비스-설정을-위한-docker-compose)
7. [헬스 체크와 그레이스풀 셧다운](#7-헬스-체크와-그레이스풀-셧다운)
8. [환경 설정 (12-팩터 앱)](#8-환경-설정-12-팩터-앱)
9. [SSL/TLS 종료](#9-ssltls-종료)
10. [연습 문제](#10-연습-문제)

---

## 1. ASGI 서버: Uvicorn과 Hypercorn

FastAPI와 다른 비동기 Python 프레임워크는 ASGI(Asynchronous Server Gateway Interface) 프로토콜을 사용한다. 개발 환경에서는 `uvicorn`이 단일 프로세스로 실행되지만, 프로덕션 환경에서는 모든 CPU 코어를 활용하기 위해 여러 워커가 필요하다.

### Gunicorn 워커로 Uvicorn 실행

권장되는 프로덕션 설정은 gunicorn이 uvicorn 워커 프로세스를 관리하는 방식이다. Gunicorn은 프로세스 수명 주기(생성, 충돌한 워커 재시작)를 담당하고, uvicorn은 비동기 이벤트 루프(event loop)를 처리한다.

```bash
# Production command
gunicorn app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --graceful-timeout 30 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --access-logfile - \
    --error-logfile -
```

**주요 설정 설명:**

| 플래그                    | 목적                                                        |
|---------------------------|-------------------------------------------------------------|
| `--workers 4`             | 워커 프로세스 수 (경험 법칙: 2 * CPU + 1)                   |
| `--worker-class`          | gunicorn의 동기 워커 대신 uvicorn의 비동기 워커를 사용한다   |
| `--timeout 120`           | 120초 동안 응답이 없는 워커를 종료한다                       |
| `--graceful-timeout 30`   | 셧다운 시 진행 중인 요청을 완료할 수 있는 시간               |
| `--max-requests 1000`     | N개 요청 후 워커를 재시작한다 (메모리 누수 방지)             |
| `--max-requests-jitter`   | 모든 워커가 동시에 재시작하지 않도록 무작위화한다            |

### Hypercorn

HTTP/2와 HTTP/3(QUIC)를 지원하는 대안 ASGI 서버다.

```bash
hypercorn app.main:app \
    --workers 4 \
    --bind 0.0.0.0:8000 \
    --access-logfile - \
    --error-logfile -
```

### 워커 수 가이드라인

```python
import multiprocessing

# CPU 집약적 워크로드: 코어 수에 맞춤
workers = multiprocessing.cpu_count()

# I/O 집약적 워크로드 (웹 API에 일반적): 코어의 2~4배
workers = multiprocessing.cpu_count() * 2 + 1

# 메모리 제약 환경: 가용 RAM을 기반으로 계산
# 각 워커는 애플리케이션에 따라 약 50~150MB를 사용한다
max_workers = available_memory_mb // worker_memory_mb
```

---

## 2. WSGI 서버: Gunicorn

Django와 Flask는 WSGI(Web Server Gateway Interface) 프로토콜을 사용한다. Gunicorn은 표준 프로덕션 WSGI 서버다.

### Django를 위한 Gunicorn

```bash
# gunicorn.conf.py
import multiprocessing

bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "gthread"       # Threaded workers for I/O-bound Django apps
threads = 4                     # Threads per worker
timeout = 120
graceful_timeout = 30
max_requests = 1000
max_requests_jitter = 50
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Preload the application for faster worker startup
# Trade-off: cannot do hot code reload per-worker
preload_app = True
```

```bash
# Run with config file
gunicorn myproject.wsgi:application --config gunicorn.conf.py

# Or inline
gunicorn myproject.wsgi:application \
    --workers 4 \
    --threads 4 \
    --worker-class gthread \
    --bind 0.0.0.0:8000
```

### 워커 클래스 비교

| 워커 클래스   | 동시성 모델              | 적합한 용도                       |
|--------------|-------------------------|---------------------------------|
| `sync`       | 워커당 요청 하나         | CPU 집약적, 단순 앱              |
| `gthread`    | 워커 내 스레드           | I/O 집약적 Django/Flask 앱       |
| `gevent`     | 그린 스레드 (코루틴)     | 높은 동시성 I/O 워크로드          |
| `uvicorn`    | asyncio 이벤트 루프      | ASGI 앱 (FastAPI, Starlette)     |

---

## 3. Node.js를 위한 PM2

PM2는 Node.js용 프로세스 관리자로, 클러스터링(clustering), 모니터링, 로그 관리, 무중단 리로드(zero-downtime reload)를 처리한다.

### ecosystem.config.js

```javascript
// ecosystem.config.js
module.exports = {
  apps: [{
    name: "api-server",
    script: "./dist/server.js",
    instances: "max",          // Use all available CPU cores
    exec_mode: "cluster",      // Enable cluster mode
    max_memory_restart: "500M", // Restart if memory exceeds 500MB
    env: {
      NODE_ENV: "production",
      PORT: 3000,
    },
    // Log configuration
    log_date_format: "YYYY-MM-DD HH:mm:ss Z",
    error_file: "./logs/error.log",
    out_file: "./logs/output.log",
    merge_logs: true,
    // Graceful shutdown
    kill_timeout: 5000,        // ms to wait before SIGKILL
    listen_timeout: 10000,     // ms to wait for app to listen
    // Zero-downtime reload
    wait_ready: true,          // Wait for process.send('ready')
    max_restarts: 10,
    restart_delay: 1000,
  }],
};
```

### 주요 PM2 명령어

```bash
# Start application
pm2 start ecosystem.config.js

# Zero-downtime reload (cluster mode required)
pm2 reload api-server

# Monitor processes
pm2 monit

# View logs
pm2 logs api-server --lines 100

# Save process list for auto-restart on reboot
pm2 save
pm2 startup    # Generate OS startup script
```

### Express에서의 그레이스풀 셧다운

```javascript
// server.js
const app = require("./app");
const http = require("http");

const server = http.createServer(app);

server.listen(process.env.PORT, () => {
  console.log(`Server listening on port ${process.env.PORT}`);
  // Signal PM2 that the app is ready
  if (process.send) {
    process.send("ready");
  }
});

// Handle graceful shutdown
process.on("SIGINT", gracefulShutdown);
process.on("SIGTERM", gracefulShutdown);

function gracefulShutdown() {
  console.log("Received shutdown signal, closing server...");
  server.close(() => {
    console.log("Server closed, cleaning up...");
    // Close database connections, flush logs, etc.
    process.exit(0);
  });

  // Force exit if cleanup takes too long
  setTimeout(() => {
    console.error("Forced shutdown after timeout");
    process.exit(1);
  }, 10000);
}
```

---

## 4. nginx 역방향 프록시

프로덕션 환경에서 애플리케이션 서버는 nginx 뒤에 위치하며, nginx는 SSL 종료(SSL termination), 정적 파일 서빙, 로드 밸런싱(load balancing), 속도 제한(rate limiting), 요청 버퍼링(request buffering)을 처리한다.

### 프로덕션 nginx 설정

```nginx
# /etc/nginx/conf.d/api.conf

upstream backend {
    # Load balancing across application server instances
    server 127.0.0.1:8000 weight=3;
    server 127.0.0.1:8001 weight=3;
    server 127.0.0.1:8002 weight=1 backup;

    # Keep-alive connections to upstream
    keepalive 32;
}

server {
    listen 80;
    server_name api.example.com;
    # Redirect all HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    # SSL configuration (see Section 9)
    ssl_certificate     /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Request size limit
    client_max_body_size 10M;

    # Timeouts
    proxy_connect_timeout 30s;
    proxy_send_timeout 60s;
    proxy_read_timeout 60s;

    # Proxy to application server
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Serve static files directly (Django collectstatic output)
    location /static/ {
        alias /var/www/app/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # Health check endpoint (no logging to reduce noise)
    location /health {
        proxy_pass http://backend;
        access_log off;
    }
}
```

---

## 5. Docker 컨테이너화

### 이론: 배포 전략

새 코드를 프로덕션에 푸시할 때 사용하는 전략이 결정합니다.

- 변경이 모든 사용자에게 얼마나 빨리 도달하는가.
- 변경이 망가졌을 때 무슨 일이 일어나는가.
- 얼마나 빨리 롤백할 수 있는가.

세 가지 전략이 지배합니다.

#### B.1 Rolling 배포 (기본값)

Replica를 한 번에 하나씩 교체:

```
시간: ──────────►
v1: [v1] [v1] [v1] [v1] [v1]
       ↓ 한 번에 하나씩 교체
v2: [v2] [v1] [v1] [v1] [v1]
v2: [v2] [v2] [v1] [v1] [v1]
...
v2: [v2] [v2] [v2] [v2] [v2]
```

장점: 단순, 추가 용량 불필요, 부드러운 트래픽 전환.

단점: rolling 중 두 버전이 트래픽을 처리합니다 — 스키마 마이그레이션과 API 계약 변경은 하위 호환되어야 합니다. 롤백은 "v1을 다시 배포"이며, 원래 배포와 같은 시간이 걸립니다.

이것이 Kubernetes의 기본 `Deployment`가 하는 것입니다.

#### B.2 Blue/green 배포

새 버전을 실행하는 완전한 두 번째 환경(green)을 세우고, 현재 환경(blue)이 트래픽을 계속 처리하게 합니다. 준비가 되면 로드 밸런서를 green으로 전환합니다.

```
Blue (v1): [v1] [v1] [v1]  ← 모든 트래픽
Green (v2): [v2] [v2] [v2]  ← idle, healthy
                ↓ 로드 밸런서 flip
Blue (v1): [v1] [v1] [v1]  ← idle (롤백을 위해 보존)
Green (v2): [v2] [v2] [v2]  ← 모든 트래픽
```

장점: 즉각적 롤백(로드 밸런서를 다시 flip), 전환 중 혼합 버전 상태 없음.

단점: 2배 용량 필요, 조정해야 할 stateful backing service가 있으면 복잡.

#### B.3 Canary 배포

먼저 작은 트래픽 비율을 새 버전으로 라우팅합니다. 메트릭이 좋아 보이면 비율을 늘리고, 그렇지 않으면 롤백합니다.

```
시간: ──────────►
v1: 100%  → 95%  → 50%  → 0%
v2:   0%  →  5%  → 50%  → 100%
              ↑ 각 단계에서 메트릭 검사
```

장점: 나쁜 배포의 폭발 반경이 canary 비율로 제한됩니다. 모두에게 영향을 주기 전에 실세계 실패를 잡습니다.

단점: 트래픽 분할 인프라(service mesh, 똑똑한 로드 밸런서, ingress controller)가 필요합니다. 롤아웃 결정을 위한 신뢰할 만한 버전별 메트릭이 필요합니다.

#### B.4 전략 고르기

| 위험 허용도 | 전략 |
|----------------|----------|
| 낮음(소비자 대상, 큰 사용자 기반) | Canary |
| 중간(내부 앱) | Blue/green |
| 낮은 비용 / 빠른 반복 | Rolling |
| 하드 컷오버가 있는 stateful 시스템 | 유지보수 창과 함께 Blue/green |

매력 없는 진실: 대부분의 팀이 rolling을 사용합니다. 플랫폼 기본값이고 동작하기 때문입니다. 결정은 위험한 배포의 상위 1%에서 가장 중요합니다.

### Python(FastAPI)을 위한 다단계 빌드

다단계 빌드(multi-stage build)는 빌드 환경과 런타임 환경을 분리하여 더 작고 안전한 이미지를 생성한다.

```dockerfile
# ---- Build stage ----
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a virtual environment
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# ---- Runtime stage ----
FROM python:3.12-slim AS runtime

# Create non-root user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash appuser

WORKDIR /app

# Copy only the virtual environment from the build stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser . .

# Runtime dependencies only (no gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["gunicorn", "app.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--access-logfile", "-"]
```

### Node.js(Express)를 위한 다단계 빌드

```dockerfile
# ---- Build stage ----
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build    # TypeScript compilation

# ---- Runtime stage ----
FROM node:20-alpine AS runtime

RUN addgroup -g 1001 -S appgroup \
    && adduser -S appuser -u 1001 -G appgroup

WORKDIR /app

COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package.json .

USER appuser

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

CMD ["node", "dist/server.js"]
```

### .dockerignore

```
node_modules
__pycache__
*.pyc
.git
.env
.env.local
docker-compose*.yml
Dockerfile*
*.md
.mypy_cache
.pytest_cache
```

---

## 6. 다중 서비스 설정을 위한 Docker Compose

Docker Compose는 다중 컨테이너 애플리케이션을 정의하고 실행한다. 일반적인 백엔드 배포에는 애플리케이션 서버, 데이터베이스, 캐시, 역방향 프록시가 포함된다.

```yaml
# docker-compose.yml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://app:secret@db:5432/myapp
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=${SECRET_KEY}
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: "1.0"

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: app
      POSTGRES_PASSWORD: secret
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app -d myapp"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 128mb --maxmemory-policy allkeys-lru
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### 유용한 명령어

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f app

# Scale application workers
docker compose up -d --scale app=3

# Execute command in running container
docker compose exec app alembic upgrade head

# Rebuild after code changes
docker compose up -d --build app

# Stop and remove everything (preserves volumes)
docker compose down

# Stop and remove everything including volumes
docker compose down -v
```

---

## 7. 헬스 체크와 그레이스풀 셧다운

### 이론: 헬스 체크와 graceful shutdown

무중단 배포는 앱과 오케스트레이터 사이의 두 계약에 의존합니다. 둘 다 단순한 HTTP/시그널 프로토콜이지만, 이를 제대로 하는 것이 "배포는 무섭다"와 "배포는 일상이다"의 차이입니다.

#### C.1 Liveness vs readiness

종종 혼동되는 두 가지 다른 헬스 체크:

- **Liveness probe** — "살아 있는가?" 실패하면 오케스트레이터가 컨테이너를 재시작합니다. 프로세스가 회복 불가능한 상태(데드락, OOM, 내부 손상)에 있을 때 실패시키세요.
- **Readiness probe** — "트래픽을 처리할 준비가 되었는가?" 실패하면 오케스트레이터가 트래픽 송신을 멈추지만 재시작은 하지 않습니다. 시작 동안 의존성이 연결되기 전, SIGTERM 후 종료 동안, 또는 의존성(DB, 캐시)이 일시적으로 도달 불가능할 때 실패시키세요.

피해야 할 실수: 둘에 같은 probe 사용. 데이터베이스가 다운됐을 때 실패하는 liveness 검사는 오케스트레이터가 컨테이너를 재시작하게 만듭니다 — 데이터베이스를 고치지는 못하지만, 재시작의 떼거리(thundering herd)는 시작합니다.

```python
@app.get("/healthz/live")  # liveness — 최소
async def liveness():
    return {"status": "ok"}

@app.get("/healthz/ready")  # readiness — 진짜 의존성에 의존
async def readiness():
    if not db_pool.is_healthy(): raise HTTPException(503)
    return {"status": "ok"}
```

#### C.2 SIGTERM의 graceful shutdown

오케스트레이터가 컨테이너를 멈추고 싶으면 SIGTERM을 보냅니다. 앱은

1. 자신을 "not ready"로 표시(readiness probe가 실패하기 시작).
2. 새 연결 수락 중단(리스닝 소켓 닫기).
3. 진행 중 요청이 끝나기를 대기(타임아웃과 함께).
4. Backing-service 연결 해제(DB 풀 drain, 캐시 disconnect).
5. 깨끗하게 종료.

앱이 `terminationGracePeriodSeconds`(Kubernetes에서 기본 30s) 안에 종료하지 않으면 오케스트레이터가 SIGKILL을 보냅니다 — 진행 중 모든 요청이 도중에 죽습니다.

이를 위한 프레임워크 hook:

- **FastAPI / Starlette**: lifespan context manager(`@asynccontextmanager`) — `yield` 이후 코드가 종료 단계.
- **Express**: `process.on('SIGTERM', ...)`과 `server`, `pool` 닫기.
- **Django**: WSGI/ASGI 핸들러 수준의 시그널 핸들러.

#### C.3 Readiness flip이 먼저 중요한 이유

1단계(readiness flip)가 결정적이며 종종 건너뜁니다. 이것 없이는 로드 밸런서가 종료 *동안* 새 요청을 계속 보냅니다. 그 요청들은 절반쯤 처리되다가 SIGKILL됩니다. 자신을 "not ready"로 표시하는 것은 로드 밸런서에 라우팅을 멈추라고 말합니다 — 2단계에서 연결 거부를 시작할 때쯤이면 로드 밸런서가 이미 회전(rotation)에서 빼냈습니다.

전체 시퀀스는 dropped-request 경쟁 조건을 막습니다. 그것과 함께라면 배포가 사용자에게 보이지 않고, 그것 없이는 모든 배포가 몇 개의 운 나쁜 요청을 깨뜨립니다.

### 헬스 체크 엔드포인트

프로덕션 배포에는 로드 밸런서, 컨테이너 오케스트레이터, 모니터링 시스템을 위한 헬스 체크가 필요하다.

```python
# FastAPI health checks
from fastapi import FastAPI
from datetime import datetime, timezone
import asyncpg

app = FastAPI()

@app.get("/health")
async def health_check():
    """라이브니스 프로브(Liveness probe): 프로세스가 실행 중인가?"""
    return {"status": "ok"}

@app.get("/health/ready")
async def readiness_check():
    """레디니스 프로브(Readiness probe): 서비스가 요청을 처리할 수 있는가?
    데이터베이스와 캐시 연결을 확인한다.
    """
    checks = {}

    # Check database
    try:
        await db.execute("SELECT 1")
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {str(e)}"

    # Check Redis
    try:
        await redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {str(e)}"

    all_ok = all(v == "ok" for v in checks.values())
    return {
        "status": "ok" if all_ok else "degraded",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
```

### FastAPI에서의 그레이스풀 셧다운

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize resources
    print("Starting up...")
    app.state.db_pool = await create_db_pool()
    app.state.redis = await create_redis_connection()

    yield  # Application runs here

    # Shutdown: clean up resources
    print("Shutting down gracefully...")
    await app.state.db_pool.close()
    await app.state.redis.close()
    print("Cleanup complete")

app = FastAPI(lifespan=lifespan)
```

### Kubernetes 프로브 설정 (참고)

```yaml
# For context: how orchestrators use these endpoints
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 15
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
  failureThreshold: 2
```

---

## 8. 환경 설정 (12-팩터 앱)

[12-팩터 앱(12-Factor App)](https://12factor.net/) 방법론은 클라우드 네이티브 애플리케이션 구축을 위한 모범 사례를 정의한다. 세 번째 팩터인 **설정(Config)**은 설정을 코드가 아닌 환경에 저장해야 한다고 명시한다.

### 이론: 12-Factor App 방법론

원래 Heroku가 2011년에 작성한 12 factors는 여전히 "배포 가능한 백엔드 서비스"의 사실상 표준입니다. Heroku 트릭이 아닙니다 — Kubernetes, Cloud Run, ECS, 모든 현대 PaaS가 앱이 이를 따르기를 기대합니다.

#### A.1 12개 factor 한눈에

| # | Factor | 구체적 의미 |
|---|--------|------------------|
| 1 | Codebase | 앱당 코드베이스 1개, 버전 관리에서 추적. |
| 2 | Dependencies | 명시적 선언(`requirements.txt`, `package.json`), 시스템 패키지 사용 안 함. |
| 3 | Config | 환경 변수, 코드 안에 절대 두지 않음. |
| 4 | Backing services | 데이터베이스, 캐시, 큐 모두 env의 URL로 연결. |
| 5 | Build, release, run | 세 개의 엄격한 단계, 절대 결합하지 않음. |
| 6 | Processes | 앱은 하나 이상의 *무상태* 프로세스; 상태는 backing service로 공유. |
| 7 | Port binding | 앱이 포트에 바인딩해 HTTP를 직접 export(Apache 모듈 없음). |
| 8 | Concurrency | 한 프로세스를 더 크게 만들지 말고 프로세스를 추가해 확장. |
| 9 | Disposability | 빠른 시작, SIGTERM 시 graceful shutdown. |
| 10 | Dev/prod parity | dev와 prod에서 같은 OS, 같은 의존성, 같은 backing service. |
| 11 | Logs | stdout으로 스트림; 플랫폼이 집계 처리. |
| 12 | Admin processes | 일회성 스크립트도 같은 코드/config로 실행. |

#### A.2 가장 큰 짐을 지는 세 가지 factor

Factor 3, 6, 11이 현대 배포에서 가장 많은 일을 합니다.

- **3 (env의 config).** 코드에 `if production: ...` 분기 없음. 같은 Docker 이미지가 dev, staging, prod에서 실행됩니다 — env 변수만 바뀝니다. 컨테이너 기반 배포가 작동하게 만드는 것.
- **6 (무상태 프로세스).** 앱은 재시작에서 살아남아야 하는 메모리 내 상태를 보관하지 않습니다. 상태는 데이터베이스, 캐시, 객체 저장소에 삽니다. 수평 확장과 무중단 배포가 작동하게 만드는 것.
- **11 (stdout으로 로그).** 앱은 로그를 이벤트 스트림으로 stdout에 씁니다. 플랫폼(Docker, Kubernetes, systemd-journald)이 수집·라우팅·인덱싱합니다. 앱은 로그가 stdout-only로 끝날지 Elasticsearch로 갈지 모릅니다.

#### A.3 대부분의 앱이 여전히 위반하는 factor

Factor 6(무상태 프로세스)이 가장 흔하게 위반됩니다. 스티키 세션, "한 인스턴스만 돌리니 괜찮다"는 메모리 내 캐시, 로컬 디스크에 저장된 파일 업로드 — 모두 두 replica로 확장하는 순간 깨집니다. 규율: **현재 1개를 돌릴지라도 앱이 N개 replica로 돌아간다고 가정하라**. 그 가정이 올바른 설계를 강제합니다.

### Pydantic Settings를 이용한 설정 관리

```python
# config.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """환경 변수에서 로드되는 애플리케이션 설정.

    Pydantic은 환경 변수를 자동으로 읽고
    타입을 검증한다. .env 파일이 폴백(fallback)으로 사용된다.
    """
    # Application
    app_name: str = "MyAPI"
    debug: bool = False
    environment: str = "production"

    # Database
    database_url: str
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Authentication
    secret_key: str
    access_token_expire_minutes: int = 15

    # External services
    smtp_host: str = ""
    smtp_port: int = 587
    sentry_dsn: str = ""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

@lru_cache
def get_settings() -> Settings:
    """캐시된 설정 인스턴스. 한 번 생성되어 모든 곳에서 재사용된다."""
    return Settings()

# Usage in FastAPI
from fastapi import Depends

@app.get("/info")
async def app_info(settings: Settings = Depends(get_settings)):
    return {
        "app": settings.app_name,
        "environment": settings.environment,
    }
```

### 환경 파일 (.env)

```bash
# .env (NEVER commit this file)
DATABASE_URL=postgresql://user:pass@localhost:5432/myapp
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-256-bit-secret-key-here
ENVIRONMENT=development
DEBUG=true
SENTRY_DSN=https://examplePublicKey@o0.ingest.sentry.io/0
```

### 백엔드 개발자를 위한 주요 12-팩터 원칙

| 팩터  | 원칙                        | 실천 방법                                       |
|-------|-----------------------------|-------------------------------------------------|
| I. 코드베이스    | 하나의 코드베이스, 다중 배포 | Git 저장소, dev/staging/prod에 배포           |
| III. 설정        | 설정은 환경에               | `.env` 파일 사용, 시크릿을 절대 하드코딩하지 않음 |
| IV. 백킹 서비스  | 연결된 리소스로 취급         | 데이터베이스 URL을 설정으로, 교체 가능하게 유지  |
| VI. 프로세스     | 스테이트리스 프로세스        | 인메모리 세션 사용 금지 (Redis 사용)            |
| VII. 포트 바인딩 | 포트를 통해 서비스 내보내기  | `--bind 0.0.0.0:8000`                          |
| VIII. 동시성     | 프로세스 모델로 확장         | Gunicorn 워커, 컨테이너 레플리카               |
| XI. 로그         | 이벤트 스트림으로 취급       | stdout에 기록, 플랫폼이 수집하도록 설정         |

### 시크릿 관리(Secrets Management)

버전 관리에 커밋된 환경 파일이나 Docker 이미지에 시크릿을 저장하지 말 것. 대신 전용 시크릿 관리자를 사용한다.

**HashiCorp Vault** — 자체 호스팅, 동적 시크릿(요청 시 생성되는 단기 DB 자격증명) 지원:

```bash
# 런타임에 시크릿 조회
vault kv get -field=password secret/myapp/database
```

**AWS Secrets Manager** — 관리형 서비스, EC2/ECS/Lambda에서 자격증명 없는 접근을 위해 IAM 역할과 통합:

```python
import boto3, json

def get_secret(secret_name: str) -> dict:
    client = boto3.client("secretsmanager", region_name="us-east-1")
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])

db_creds = get_secret("prod/myapp/database")
DATABASE_URL = f"postgresql://{db_creds['username']}:{db_creds['password']}@{db_creds['host']}/myapp"
```

모범 사례: 시크릿을 자동으로 교체(rotate)하고, 서비스별로 최소 권한 IAM/Vault 정책을 부여하며, 시크릿 값을 절대 로그에 기록하지 않는다.

### 블루-그린 배포(Blue-Green Deployment)

블루-그린 배포는 두 개의 동일한 환경(blue = 현재, green = 신규)을 운영하여 다운타임을 제거한다. 로드 밸런서를 통해 트래픽이 원자적으로 전환된다.

```
                   ┌─────────────────────────────────┐
                   │        Load Balancer / nginx      │
                   └───────────────┬─────────────────┘
                   Traffic: 100%   │   Traffic: 0%
                   ┌───────────────▼──┐  ┌────────────────┐
                   │  Blue (v1.2 live) │  │ Green (v1.3 new)│
                   └──────────────────┘  └────────────────┘
```

**배포 단계:**
1. blue가 트래픽을 처리하는 동안 유휴 환경(green)에 새 버전 배포
2. 내부 포트에서 green에 대해 스모크 테스트(smoke test) 및 헬스 체크 실행
3. 로드 밸런서를 전환하여 100% 트래픽을 green으로 즉시 전달
4. 관찰 시간(10~15분) 동안 blue를 유지; 로드 밸런서를 다시 가리켜 롤백 가능
5. 관찰 시간이 지나면 blue 제거

Docker Compose에서는 `--scale`과 nginx `upstream` 재설정을 사용한다. Kubernetes에서는 `Service` 셀렉터(selector)를 새 Deployment로 업데이트한다. 롤링 배포(rolling deploy) 대비 핵심 장점은 새 버전에 확신이 생길 때까지 이전 버전이 완전히 온전한 상태로 유지된다는 것이다.

---

## 9. SSL/TLS 종료

역방향 프록시 계층에서의 SSL/TLS 종료(SSL/TLS termination)는 프록시가 암호화/복호화를 처리하고, 애플리케이션 서버는 내부적으로 평문 HTTP로 통신함을 의미한다. 이는 애플리케이션을 단순화하고 인증서 관리를 중앙화한다.

### Let's Encrypt와 Certbot

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate (nginx plugin handles configuration)
sudo certbot --nginx -d api.example.com

# Auto-renewal (certbot installs a cron job / systemd timer)
sudo certbot renew --dry-run
```

### nginx SSL 모범 사례

```nginx
# /etc/nginx/conf.d/ssl.conf

# Modern TLS configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;

# OCSP stapling
ssl_stapling on;
ssl_stapling_verify on;
resolver 1.1.1.1 8.8.8.8 valid=300s;
resolver_timeout 5s;

# Session resumption for performance
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 1d;
ssl_session_tickets off;

# HSTS (HTTP Strict Transport Security)
add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
```

### 아키텍처 다이어그램

```
Client (HTTPS)
    |
    v
[nginx] --- SSL termination, static files, rate limiting
    |
    | (HTTP, internal network)
    v
[gunicorn/uvicorn] --- Application logic
    |
    v
[PostgreSQL] [Redis] --- Backing services
```

nginx와 애플리케이션 서버 간의 내부 HTTP 트래픽은 양쪽이 같은 호스트 또는 신뢰할 수 있는 네트워크 내에서 실행될 때 (예: Docker 브리지 네트워크, Kubernetes 파드) 허용된다. 신뢰할 수 없는 네트워크를 통한 통신에는 상호 TLS(mutual TLS, mTLS)를 사용한다.

---

## 10. 연습 문제

### 문제 1: Dockerfile 최적화

다음 Dockerfile에서 최소 5개의 문제점을 찾아 모범 사례로 다시 작성하라:

```dockerfile
FROM python:3.12
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y postgresql-client
EXPOSE 8000
CMD python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 문제 2: Gunicorn 설정

FastAPI 애플리케이션이 초당 500개의 요청을 수신하며, 평균 응답 시간은 200ms다. 각 워커는 약 100MB의 RAM을 사용한다. 서버에는 8개의 CPU 코어와 16GB의 RAM이 있다. 최적의 gunicorn 설정을 계산하고 각 설정에 대한 근거와 함께 완전한 `gunicorn.conf.py` 파일을 작성하라.

### 문제 3: Docker Compose 개발 환경

6절의 프로덕션 compose 파일을 확장하는 `docker-compose.dev.yml`을 다음 조건으로 작성하라:
- 애플리케이션 핫 리로드(hot-reload) (소스 코드 볼륨 마운트)
- 데이터베이스 관리를 위한 pgAdmin
- 캐시 검사를 위한 Redis Commander
- 이메일 테스트를 위한 Mailhog
- 호스트에서 접근 가능한 모든 개발 도구

### 문제 4: 무중단 배포

Docker Compose 기반 애플리케이션에서 무중단 배포(zero-downtime deployment)를 달성하는 배포 스크립트(`deploy.sh`)를 설계하라. 스크립트는 다음을 수행해야 한다:
1. 새 이미지 빌드
2. 기존 컨테이너 옆에 새 컨테이너 시작
3. 헬스 체크 통과 대기
4. nginx 업스트림을 새 컨테이너로 전환
5. 기존 컨테이너 드레인(drain) 및 중지
6. 헬스 체크 실패 시 롤백

### 문제 5: 완전한 nginx 설정

다음 조건을 갖춘 API를 위한 프로덕션 nginx 설정을 작성하라:
- Let's Encrypt 인증서로 SSL 종료
- 3개의 애플리케이션 인스턴스에 대한 로드 밸런싱
- IP당 분당 100개 요청으로 속도 제한
- 공격적인 캐싱으로 정적 파일 서빙
- 일반적인 공격 패턴 차단 (경로 탐색, URL의 SQL 인젝션 시도)
- 특정 프론트엔드 도메인에 대한 적절한 CORS 헤더 반환

---

## 참고 자료

- [Gunicorn Documentation](https://docs.gunicorn.org/)
- [Uvicorn Deployment](https://www.uvicorn.org/deployment/)
- [Docker Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [Docker Compose Specification](https://docs.docker.com/compose/compose-file/)
- [nginx Reverse Proxy Guide](https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy/)
- [12-Factor App](https://12factor.net/)
- [PM2 Documentation](https://pm2.keymetrics.io/docs/)
- [Let's Encrypt / Certbot](https://certbot.eff.org/)
- [Mozilla SSL Configuration Generator](https://ssl-config.mozilla.org/)

---

**이전**: [인증 패턴](./15_Authentication_Patterns.md) | **다음**: [관찰 가능성](./17_Observability.md)
