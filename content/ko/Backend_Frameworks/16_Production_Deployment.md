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
