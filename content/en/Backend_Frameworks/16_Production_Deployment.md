# 16. Production Deployment

**Previous**: [Authentication Patterns](./15_Authentication_Patterns.md) | **Next**: [Observability](./17_Observability.md)

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

- Configure ASGI and WSGI application servers (uvicorn, gunicorn) with appropriate worker settings for production
- Set up nginx as a reverse proxy with SSL/TLS termination, load balancing, and security headers
- Containerize backend applications using Docker with multi-stage builds and Docker Compose for multi-service orchestration
- Implement health checks and graceful shutdown patterns for zero-downtime deployments
- Apply the 12-Factor App methodology to manage configuration and environment variables

## Table of Contents

1. [ASGI Servers: Uvicorn and Hypercorn](#1-asgi-servers-uvicorn-and-hypercorn)
2. [WSGI Servers: Gunicorn](#2-wsgi-servers-gunicorn)
3. [PM2 for Node.js](#3-pm2-for-nodejs)
4. [Reverse Proxy with nginx](#4-reverse-proxy-with-nginx)
5. [Docker Containerization](#5-docker-containerization)
6. [Docker Compose for Multi-Service Setups](#6-docker-compose-for-multi-service-setups)
7. [Health Checks and Graceful Shutdown](#7-health-checks-and-graceful-shutdown)
8. [Environment Configuration (12-Factor App)](#8-environment-configuration-12-factor-app)
9. [SSL/TLS Termination](#9-ssltls-termination)
10. [Practice Problems](#10-practice-problems)

---

## 1. ASGI Servers: Uvicorn and Hypercorn

FastAPI and other async Python frameworks use the ASGI (Asynchronous Server Gateway Interface) protocol. In development, `uvicorn` runs a single process. In production, you need multiple workers to utilize all CPU cores.

### Uvicorn with Gunicorn Workers

The recommended production setup is gunicorn managing uvicorn worker processes. Gunicorn handles process lifecycle (spawning, restarting crashed workers), while uvicorn handles the async event loop.

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

**Key settings explained:**

| Flag                    | Purpose                                                    |
|-------------------------|------------------------------------------------------------|
| `--workers 4`           | Number of worker processes (rule of thumb: 2 * CPU + 1)    |
| `--worker-class`        | Use uvicorn's async worker instead of gunicorn's sync worker |
| `--timeout 120`         | Kill workers that are silent for 120 seconds               |
| `--graceful-timeout 30` | Time to finish in-flight requests during shutdown          |
| `--max-requests 1000`   | Restart workers after N requests (prevents memory leaks)   |
| `--max-requests-jitter` | Randomize restart to avoid all workers restarting at once  |

### Hypercorn

An alternative ASGI server that supports HTTP/2 and HTTP/3 (QUIC).

```bash
hypercorn app.main:app \
    --workers 4 \
    --bind 0.0.0.0:8000 \
    --access-logfile - \
    --error-logfile -
```

### Worker Count Guidelines

```python
import multiprocessing

# CPU-bound workloads: match core count
workers = multiprocessing.cpu_count()

# I/O-bound workloads (typical for web APIs): 2x-4x cores
workers = multiprocessing.cpu_count() * 2 + 1

# Memory-constrained environments: calculate based on available RAM
# Each worker uses ~50-150MB depending on your application
max_workers = available_memory_mb // worker_memory_mb
```

---

## 2. WSGI Servers: Gunicorn

Django and Flask use the WSGI (Web Server Gateway Interface) protocol. Gunicorn is the standard production WSGI server.

### Gunicorn for Django

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

### Worker Class Comparison

| Worker Class | Concurrency Model       | Best For                        |
|--------------|------------------------|---------------------------------|
| `sync`       | One request per worker  | CPU-bound, simple apps          |
| `gthread`    | Threads within workers  | I/O-bound Django/Flask apps     |
| `gevent`     | Green threads (coroutines) | High-concurrency I/O workloads |
| `uvicorn`    | asyncio event loop      | ASGI apps (FastAPI, Starlette)  |

---

## 3. PM2 for Node.js

PM2 is a process manager for Node.js that handles clustering, monitoring, log management, and zero-downtime reloads.

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

### Common PM2 Commands

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

### Graceful Shutdown in Express

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

## 4. Reverse Proxy with nginx

In production, application servers sit behind nginx, which handles SSL termination, static files, load balancing, rate limiting, and request buffering.

### Production nginx Configuration

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

## 5. Docker Containerization

### Multi-Stage Build for Python (FastAPI)

Multi-stage builds separate the build environment from the runtime environment, producing smaller and more secure images.

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

### Multi-Stage Build for Node.js (Express)

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

## 6. Docker Compose for Multi-Service Setups

Docker Compose defines and runs multi-container applications. A typical backend deployment includes the application server, database, cache, and reverse proxy.

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

### Useful Commands

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

## 7. Health Checks and Graceful Shutdown

### Health Check Endpoints

Production deployments need health checks for load balancers, container orchestrators, and monitoring systems.

```python
# FastAPI health checks
from fastapi import FastAPI
from datetime import datetime, timezone
import asyncpg

app = FastAPI()

@app.get("/health")
async def health_check():
    """Liveness probe: is the process running?"""
    return {"status": "ok"}

@app.get("/health/ready")
async def readiness_check():
    """Readiness probe: can the service handle requests?
    Checks database and cache connectivity.
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

### Graceful Shutdown in FastAPI

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

### Kubernetes Probe Configuration (Reference)

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

## 8. Environment Configuration (12-Factor App)

The [12-Factor App](https://12factor.net/) methodology defines best practices for building cloud-native applications. Factor III --- **Config** --- states that configuration should be stored in the environment, not in code.

### Configuration Management with Pydantic Settings

```python
# config.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Pydantic automatically reads from environment variables
    and validates types. A .env file is used as a fallback.
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
    """Cached settings instance. Created once, reused everywhere."""
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

### Environment File (.env)

```bash
# .env (NEVER commit this file)
DATABASE_URL=postgresql://user:pass@localhost:5432/myapp
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-256-bit-secret-key-here
ENVIRONMENT=development
DEBUG=true
SENTRY_DSN=https://examplePublicKey@o0.ingest.sentry.io/0
```

### Key 12-Factor Principles for Backend Developers

| Factor | Principle | Practice |
|--------|-----------|----------|
| I. Codebase | One codebase, many deploys | Git repo, deploy to dev/staging/prod |
| III. Config | Config in environment | `.env` files, never hardcode secrets |
| IV. Backing services | Treat as attached resources | Database URL as config, swappable |
| VI. Processes | Stateless processes | No in-memory sessions (use Redis) |
| VII. Port binding | Export services via port | `--bind 0.0.0.0:8000` |
| VIII. Concurrency | Scale via process model | Gunicorn workers, container replicas |
| XI. Logs | Treat as event streams | Write to stdout, let platform collect |

---

## 9. SSL/TLS Termination

SSL/TLS termination at the reverse proxy layer means the proxy handles encryption/decryption, and the application server communicates in plain HTTP internally. This simplifies the application and centralizes certificate management.

### Let's Encrypt with Certbot

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate (nginx plugin handles configuration)
sudo certbot --nginx -d api.example.com

# Auto-renewal (certbot installs a cron job / systemd timer)
sudo certbot renew --dry-run
```

### nginx SSL Best Practices

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

### Architecture Diagram

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

The internal HTTP traffic between nginx and the application server is acceptable when both run on the same host or within a trusted network (e.g., Docker bridge network, Kubernetes pod). For communication across untrusted networks, use mutual TLS (mTLS).

---

## 10. Practice Problems

### Problem 1: Dockerfile Optimization

Given this Dockerfile, identify at least 5 problems and rewrite it with best practices:

```dockerfile
FROM python:3.12
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y postgresql-client
EXPOSE 8000
CMD python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Problem 2: Gunicorn Configuration

A FastAPI application receives 500 requests/second, with an average response time of 200ms. Each worker uses approximately 100MB of RAM. The server has 8 CPU cores and 16GB of RAM. Calculate the optimal gunicorn configuration and write the complete `gunicorn.conf.py` file with justification for each setting.

### Problem 3: Docker Compose Development Environment

Create a `docker-compose.dev.yml` that extends the production compose file from Section 6 with:
- Hot-reload for the application (volume mount source code)
- pgAdmin for database management
- Redis Commander for cache inspection
- Mailhog for email testing
- All development tools accessible from the host

### Problem 4: Zero-Downtime Deployment

Design a deployment script (`deploy.sh`) that achieves zero-downtime deployment for a Docker Compose-based application. The script should:
1. Build the new image
2. Start new containers alongside old ones
3. Wait for health checks to pass
4. Switch nginx upstream to new containers
5. Drain and stop old containers
6. Roll back if health checks fail

### Problem 5: Complete nginx Configuration

Write a production nginx configuration for an API that:
- Terminates SSL with Let's Encrypt certificates
- Load-balances across 3 application instances
- Rate-limits to 100 requests/minute per IP
- Serves static files with aggressive caching
- Blocks common attack patterns (path traversal, SQL injection attempts in URLs)
- Returns proper CORS headers for a specific frontend domain

---

## References

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

**Previous**: [Authentication Patterns](./15_Authentication_Patterns.md) | **Next**: [Observability](./17_Observability.md)
