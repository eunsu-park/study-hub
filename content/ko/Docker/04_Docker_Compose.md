# Docker Compose

**이전**: [Dockerfile](./03_Dockerfile.md) | **다음**: [실전 예제](./05_Practical_Examples.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Docker Compose가 무엇인지, 그리고 왜 멀티 컨테이너 애플리케이션 관리를 단순화하는지 설명할 수 있다
2. 서비스(services), 포트(ports), 환경 변수(environment variables), 볼륨(volumes), 네트워크(networks)를 포함한 docker-compose.yml 파일을 작성할 수 있다
3. `depends_on`, `healthcheck`, 재시작 정책(restart policies)을 사용하여 서비스 의존성과 안정성을 관리할 수 있다
4. Docker Compose CLI 명령어를 활용하여 서비스를 시작, 중지, 스케일링, 모니터링할 수 있다
5. 여러 Compose 파일을 이용한 환경별 설정 오버라이드(override)를 구성할 수 있다
6. 헬스체크(health check)와 조건부 시작을 통한 서비스 준비 패턴을 구현할 수 있다

---

대부분의 실제 애플리케이션은 웹 서버, 데이터베이스, 캐시, 메시지 큐 등 여러 서비스로 구성됩니다. 이 각각을 별도의 `docker run` 명령으로 관리하면 금세 다루기 어렵고 오류가 발생하기 쉬워집니다. Docker Compose는 전체 애플리케이션 스택을 하나의 YAML 파일에 정의하고 단 하나의 명령으로 제어할 수 있게 해줍니다. 로컬 개발 환경과 간단한 프로덕션 배포를 위한 표준 도구입니다.

## 1. Docker Compose란?

Docker Compose는 **여러 컨테이너를 정의하고 실행**하는 도구입니다. YAML 파일 하나로 전체 애플리케이션 스택을 관리합니다.

### 왜 Docker Compose를 사용할까요?

**일반 Docker 명령어:**
```bash
# Create network — needed so containers can reach each other by name
docker network create myapp-network

# Run database
docker run -d \
  --name db \
  --network myapp-network \
  -e POSTGRES_PASSWORD=secret \
  -v pgdata:/var/lib/postgresql/data \
  postgres:15

# Run backend — must remember the exact network, env vars, volume for every service
docker run -d \
  --name backend \
  --network myapp-network \
  -e DATABASE_URL=postgres://... \
  -p 3000:3000 \
  my-backend

# Run frontend — three separate commands to manage; error-prone and hard to reproduce
docker run -d \
  --name frontend \
  --network myapp-network \
  -p 80:80 \
  my-frontend
```

**Docker Compose:**
```bash
docker compose up -d
```

| 장점 | 설명 |
|------|------|
| **간편함** | 한 명령으로 전체 실행 |
| **선언적** | YAML로 명확하게 정의 |
| **버전 관리** | 설정 파일을 Git으로 관리 |
| **재현성** | 동일한 환경 재현 가능 |

---

## 2. 설치 확인

Docker Desktop에는 Docker Compose가 포함되어 있습니다.

```bash
# Check version
docker compose version
# Docker Compose version v2.23.0

# Or (old version)
docker-compose --version
```

> **참고:** `docker-compose` (하이픈)은 구버전, `docker compose` (공백)은 신버전입니다.

---

## 3. docker-compose.yml 기본 구조

```yaml
# docker-compose.yml

services:
  service-name1:
    image: image-name
    ports:
      - "host:container"
    environment:
      - variable=value
    volumes:
      - volume:path
    depends_on:
      - other-service

  service-name2:
    build: ./path
    ...

volumes:
  volume-name:

networks:
  network-name:
```

---

## 4. 주요 설정 옵션

### services - 서비스 정의

```yaml
services:
  web:
    image: nginx:alpine
```

### image - 이미지 지정

```yaml
services:
  db:
    image: postgres:15

  redis:
    image: redis:7-alpine
```

### build - Dockerfile로 빌드

```yaml
services:
  app:
    build: .                    # Dockerfile in current directory

  api:
    build:
      context: ./backend        # Build context
      dockerfile: Dockerfile    # Dockerfile path
      args:                     # Build arguments
        - NODE_ENV=production
```

### ports - 포트 매핑

```yaml
services:
  web:
    ports:
      - "8080:80"              # host:container
      - "443:443"

  api:
    ports:
      - "3000:3000"
```

### environment - 환경 변수

```yaml
services:
  db:
    environment:
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=secret
      - POSTGRES_DB=myapp

  # Or key: value format
  api:
    environment:
      NODE_ENV: production
      DB_HOST: db
```

### env_file - 환경 변수 파일

```yaml
services:
  api:
    env_file:
      - .env
      - .env.local
```

**.env 파일:**
```
DB_HOST=localhost
DB_PASSWORD=secret
API_KEY=abc123
```

### volumes - 볼륨 마운트

```yaml
services:
  db:
    volumes:
      - pgdata:/var/lib/postgresql/data    # Named volume — data survives container removal
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql  # Bind mount — auto-runs SQL on first start

  app:
    volumes:
      - ./src:/app/src                      # Source code mount — enables live-reload during dev
      - /app/node_modules                   # Anonymous volume — prevents host's node_modules from overwriting container's

volumes:
  pgdata:                                   # Declare here so Compose manages the volume lifecycle
```

### depends_on - 의존성

```yaml
services:
  api:
    depends_on:
      - db
      - redis

  db:
    image: postgres:15

  redis:
    image: redis:7
```

> **주의:** `depends_on`은 시작 순서만 보장합니다. 서비스가 "준비"될 때까지 기다리지 않습니다.

### networks - 네트워크

```yaml
services:
  frontend:
    networks:
      - frontend-net      # frontend can only talk to backend, not directly to db

  backend:
    networks:
      - frontend-net      # reachable by frontend
      - backend-net       # can reach db — acts as a gateway between the two networks

  db:
    networks:
      - backend-net       # isolated from frontend — reduces attack surface

networks:
  frontend-net:           # Separate networks enforce least-privilege network access
  backend-net:
```

### restart - 재시작 정책

```yaml
services:
  web:
    restart: always              # Always restart — even after daemon reboot (production use)

  api:
    restart: unless-stopped      # Auto-restart on crash, but respect manual docker stop

  worker:
    restart: on-failure          # Restart only on non-zero exit — avoids infinite loops from intentional shutdowns
```

### healthcheck - 헬스체크

```yaml
services:
  api:
    healthcheck:
      # Orchestrators use health checks to restart unhealthy containers automatically
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s            # How often to probe
      timeout: 10s             # Max wait per probe before marking as failure
      retries: 3               # Consecutive failures before marking "unhealthy"
      start_period: 40s        # Grace period for slow-starting apps (failures don't count here)
```

---

## 5. Docker Compose 명령어

### 실행

```bash
# Run (foreground)
docker compose up

# Run in background
docker compose up -d

# Rebuild images then run
docker compose up --build

# Run specific services only
docker compose up -d web api
```

### 중지/삭제

```bash
# Stop
docker compose stop

# Stop and remove containers
docker compose down

# Also remove volumes — destroys persistent data; use only when you want a clean slate
docker compose down -v

# Also remove images — forces a fresh pull/build on next 'up'; useful after major changes
docker compose down --rmi all
```

### 상태 확인

```bash
# List services
docker compose ps

# View logs
docker compose logs

# View specific service logs
docker compose logs api

# Real-time logs
docker compose logs -f
```

### 서비스 관리

```bash
# Restart
docker compose restart

# Restart specific service
docker compose restart api

# Scale services
docker compose up -d --scale api=3

# Execute command in service
docker compose exec api bash
docker compose exec db psql -U postgres
```

---

## 6. 실습 예제

### 예제 1: 웹 + 데이터베이스

**프로젝트 구조:**
```
my-webapp/
├── docker-compose.yml
├── .env
└── app/
    ├── Dockerfile
    └── index.js
```

**docker-compose.yml:**
```yaml
services:
  app:
    build: ./app
    ports:
      - "3000:3000"
    environment:
      # 'db' hostname works because Compose creates a shared network with DNS for each service
      - DATABASE_URL=postgres://user:pass@db:5432/mydb
    depends_on:
      - db                       # Ensures db container starts first (but not necessarily "ready")

  db:
    image: postgres:15-alpine    # Alpine variant: smaller image, faster pulls
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=mydb
    volumes:
      - pgdata:/var/lib/postgresql/data   # Named volume — data persists across restarts
    ports:
      - "5432:5432"              # Expose to host for local DB tools (pgAdmin, DBeaver, etc.)

volumes:
  pgdata:
```

**app/Dockerfile:**
```dockerfile
FROM node:18-alpine
WORKDIR /app
# Copy dependency manifest first — changes less often, so Docker caches the install layer
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
# Exec form: process runs as PID 1, receives SIGTERM for graceful shutdown
CMD ["node", "index.js"]
```

**app/index.js:**
```javascript
const express = require('express');
const app = express();

app.get('/', (req, res) => {
  res.json({
    message: 'Hello from Docker Compose!',
    db_url: process.env.DATABASE_URL ? 'Connected' : 'Not set'
  });
});

app.listen(3000, () => console.log('Server on port 3000'));
```

**실행:**
```bash
cd my-webapp
docker compose up -d
curl http://localhost:3000
docker compose logs -f
docker compose down
```

### 예제 2: 풀스택 애플리케이션

```yaml
# docker-compose.yml

services:
  # Frontend (React)
  frontend:
    build: ./frontend
    ports:
      - "80:80"              # Standard HTTP port — no port prefix needed in browser URL
    depends_on:
      - backend

  # Backend (Node.js)
  backend:
    build: ./backend
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DB_HOST=db           # Compose DNS resolves 'db' to the database container's IP
      - DB_NAME=myapp
      - REDIS_HOST=redis     # Same DNS-based discovery for the cache service
    depends_on:
      - db
      - redis

  # Database (PostgreSQL)
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=myapp
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=${DB_PASSWORD}   # Read from .env file — keeps secrets out of YAML
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql  # Auto-runs on first container start only

  # Cache (Redis)
  redis:
    image: redis:7-alpine                  # Alpine: ~30 MB vs ~130 MB full Redis image
    volumes:
      - redisdata:/data                    # Persist cache across restarts (useful for sessions)

  # Admin tool (pgAdmin)
  pgadmin:
    image: dpage/pgadmin4
    environment:
      - PGADMIN_DEFAULT_EMAIL=admin@example.com
      - PGADMIN_DEFAULT_PASSWORD=admin
    ports:
      - "5050:80"            # Non-standard host port to avoid conflicts with other services on :80
    depends_on:
      - db

volumes:
  pgdata:
  redisdata:
```

**.env:**
```
DB_PASSWORD=supersecret123
```

### 예제 3: 개발 환경

```yaml
# docker-compose.dev.yml

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.dev     # Separate Dockerfile — may include dev tools (nodemon, debugger)
    ports:
      - "3000:3000"
    volumes:
      - .:/app                    # Bind mount — edit on host, changes appear instantly in container
      - /app/node_modules         # Anonymous volume: prevents host bind mount from hiding container's installed modules
    environment:
      - NODE_ENV=development
    command: npm run dev          # Override CMD — use a file-watching dev server instead of production start

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_PASSWORD=devpass
    ports:
      - "5432:5432"              # Expose to host so local DB tools (pgAdmin, psql) can connect directly
```

**실행:**
```bash
# Development environment
docker compose -f docker-compose.dev.yml up

# Production environment
docker compose -f docker-compose.yml up -d
```

---

## 7. 유용한 패턴

### 환경별 설정 분리

```yaml
# docker-compose.yml (base)
services:
  app:
    image: myapp

# docker-compose.override.yml (dev, auto-merged)
services:
  app:
    build: .
    volumes:
      - .:/app

# docker-compose.prod.yml (production)
services:
  app:
    restart: always
```

```bash
# Development: auto-merges docker-compose.yml + docker-compose.override.yml
docker compose up

# Production
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 서비스 대기 (wait-for-it)

```yaml
services:
  app:
    depends_on:
      db:
        condition: service_healthy   # Wait until db is actually ready, not just started

  db:
    image: postgres:15
    healthcheck:
      # pg_isready checks if Postgres is accepting connections — better than just checking if the process is alive
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
```

---

## 명령어 요약

| 명령어 | 설명 |
|--------|------|
| `docker compose up` | 서비스 시작 |
| `docker compose up -d` | 백그라운드 시작 |
| `docker compose up --build` | 재빌드 후 시작 |
| `docker compose down` | 서비스 중지 및 삭제 |
| `docker compose down -v` | 볼륨도 삭제 |
| `docker compose ps` | 서비스 상태 |
| `docker compose logs` | 로그 확인 |
| `docker compose logs -f` | 실시간 로그 |
| `docker compose exec 서비스 명령` | 명령 실행 |
| `docker compose restart` | 재시작 |

---

## 연습 문제

### 연습 1: 두 서비스 스택(Two-Service Stack)

간단한 웹 앱과 Redis 카운터로 Docker Compose 스택(stack)을 만듭니다.

1. 두 개의 서비스(service)를 포함하는 `docker-compose.yml`을 작성합니다:
   - `redis`: `redis:7-alpine` 이미지 사용
   - `web`: `python:3.11-slim` 사용, 포트(port) 5000 게시, `DATABASE_URL=redis://redis:6379`를 환경 변수(environment variable)로 설정
2. `web`이 `redis` 이후에 시작되도록 `depends_on` 규칙을 추가합니다
3. `docker compose up -d`를 실행하고 `docker compose ps`로 두 서비스가 실행 중인지 확인합니다
4. `redis` 서비스의 로그를 확인합니다: `docker compose logs redis`
5. `redis` 컨테이너 내부에서 Redis CLI 명령을 실행합니다: `docker compose exec redis redis-cli ping`
6. `docker compose down`으로 종료하고 모든 컨테이너가 삭제되었는지 확인합니다

### 연습 2: 헬스 체크(Health Check)가 있는 영속적 데이터베이스

헬스 체크(health check)와 의존성 기반 앱 시작을 포함하여 PostgreSQL 서비스를 구성합니다.

1. 다음을 포함하는 `docker-compose.yml`을 작성합니다:
   - `db`: `postgres:15-alpine`, 네임드 볼륨(named volume) `pgdata:/var/lib/postgresql/data`, `pg_isready`를 사용하는 `healthcheck`
   - `app`: 임의의 이미지, `depends_on.db.condition: service_healthy`
2. 스택을 시작하고 `docker compose ps`를 실행합니다 — `db`가 건강한 상태(healthy)가 된 후에만 `app`이 시작되는 것을 관찰합니다
3. 스택을 중지하고 재시작하여 네임드 볼륨 덕분에 `db`의 데이터가 유지되는지 확인합니다
4. `docker compose down -v`를 실행하여 네임드 볼륨도 삭제되는지 확인합니다

### 연습 3: 개발과 프로덕션(Production) 환경

여러 Compose 파일을 사용하여 환경별 설정을 관리합니다.

1. 빌드된 이미지(`build: .`)를 사용하는 `web` 서비스를 포함하는 기본 `docker-compose.yml`을 작성합니다
2. 개발 환경을 위한 `docker-compose.override.yml`을 작성합니다:
   - 소스 코드를 볼륨으로 마운트: `.:/app`
   - `NODE_ENV=development` 설정
   - 포트 `3001:3000` 매핑
3. 프로덕션을 위한 `docker-compose.prod.yml`을 작성합니다:
   - `restart: always` 추가
   - `NODE_ENV=production` 설정
   - 포트 `80:3000` 매핑
4. 개발 모드로 시작: `docker compose up` (override 자동 병합)
5. 프로덕션 모드로 시작: `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d`
6. 두 모드 간의 구성 차이를 확인합니다

### 연습 4: 서비스 스케일링(Service Scaling)

서비스를 스케일(scale)하고 부하 분산을 관찰합니다.

1. 호스트명(hostname)으로 응답하는 `api` 서비스가 포함된 `docker-compose.yml`을 작성합니다 (`hashicorp/http-echo -text="$(hostname)"` 또는 유사한 이미지 사용)
2. `docker compose up -d`로 시작합니다
3. `api` 서비스를 3개의 레플리카(replica)로 스케일합니다: `docker compose up -d --scale api=3`
4. `docker compose ps`로 세 개의 컨테이너가 실행 중인지 확인합니다
5. `docker compose logs api`로 모든 레플리카의 로그를 확인합니다
6. 1개의 레플리카로 다시 스케일 다운(scale down)하고 확인합니다

### 연습 5: 풀스택(Full-Stack) 애플리케이션 Compose

프론트엔드(frontend), 백엔드(backend), 데이터베이스 3개 서비스로 compose 파일을 만듭니다.

1. `docker-compose.yml`에 세 개의 서비스를 정의합니다:
   - `db`: `postgres:15-alpine`, 환경 변수와 네임드 볼륨
   - `backend`: 로컬 Dockerfile로 빌드, `db`에 의존, 데이터베이스 연결 환경 변수 포함
   - `frontend`: 다른 Dockerfile로 빌드, `backend`에 의존, 포트 80 게시
2. 두 개의 네트워크(network)를 정의합니다: `frontend-net` (frontend + backend)과 `backend-net` (backend + db)
3. `frontend`가 `db`에 직접 접근할 수 없도록 각 서비스를 적절한 네트워크에 할당합니다
4. 스택을 시작하고 `docker compose exec db psql`을 사용하여 `backend`에서는 데이터베이스에 접근 가능하지만 `frontend`에서는 불가능한지 확인합니다
5. `docker inspect`를 사용하여 네트워크 할당을 확인합니다

---

**이전**: [Dockerfile](./03_Dockerfile.md) | **다음**: [실전 예제](./05_Practical_Examples.md)
