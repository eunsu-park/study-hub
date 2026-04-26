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

YAML 레퍼런스 전에 [**이론과 원리**](#이론과-원리) 섹션을 읽으세요. Compose의 선언적 모델, 서비스 시작 순서를 결정하는 의존성 DAG, 그리고 서비스가 이름으로 통신하도록 자동으로 만들어지는 프로젝트 스코프 네트워크를 다룹니다.

대부분의 실제 애플리케이션은 웹 서버, 데이터베이스, 캐시, 메시지 큐 등 여러 서비스로 구성됩니다. 이 각각을 별도의 `docker run` 명령으로 관리하면 금세 다루기 어렵고 오류가 발생하기 쉬워집니다. Docker Compose는 전체 애플리케이션 스택을 하나의 YAML 파일에 정의하고 단 하나의 명령으로 제어할 수 있게 해줍니다. 로컬 개발 환경과 간단한 프로덕션 배포를 위한 표준 도구입니다.

---

## 이론과 원리

Docker Compose는 `docker run`을 받쳐 주는 같은 엔진 위에 얹은 얇은 선언적 계층입니다. 흥미로운 부분은 *모델*(선언적 상태, 프로젝트 스코프, 의존성 그래프)과 *Compose가 조용히 깔아 주는 기본값들*(프로젝트 네트워크, 명명 볼륨, 컨테이너 이름 규칙, 환경 변수 보간)입니다. 이를 보고 나면 YAML이 마법처럼 보이는 대신 API를 호출하는 구조화된 방식으로 보이게 됩니다.

### A. 선언적 모델: 단계가 아닌 상태

`docker run` 명령은 *명령형*입니다 — "지금 이 컨테이너를 이 플래그로 실행해라". `docker-compose.yml`은 *선언형*입니다 — "이게 존재해야 할 서비스, 네트워크, 볼륨의 집합이다". Compose 엔진은 원하는 상태를 읽고, 엔진의 실제 상태와 비교하고, 차이를 계산합니다.

`docker compose up`이 다음과 같이 조정(reconcile)합니다.

1. YAML을 읽고 원하는 서비스/네트워크/볼륨 집합을 계산합니다.
2. 같은 프로젝트 이름 아래 이미 존재하는 것을 데몬에 질의합니다.
3. 빠진 것을 만듭니다(네트워크 먼저, 그다음 볼륨, 마지막 컨테이너).
4. YAML과 설정이 어긋난 컨테이너를 재생성합니다(다른 이미지 태그, 다른 env, 다른 포트 — Compose가 config를 해시해 라벨로 저장하므로 해시가 바뀌면 컨테이너가 교체됨).
5. 이미 일치하는 것은 그대로 둡니다.

반대 동사가 `docker compose down`이며, 프로젝트의 컨테이너를 제거합니다(선택적으로 볼륨/네트워크도). `up`과 `down` 사이에 `docker compose up -d`(분리 모드), `docker compose restart`, `docker compose stop`(컨테이너 유지), `docker compose rm`(컨테이너 제거하되 네트워크/볼륨 유지)이 있습니다.

모델이 선언적이라 멱등성(idempotency)이 무료입니다 — 변경 없는 YAML에 `up`을 두 번 돌려도 두 번째는 아무 일도 안 합니다. 이 성질이 Compose를 CI와 로컬 개발 루프에서 안전하게 만듭니다.

### B. 프로젝트 스코프: 명명과 격리

모든 `docker compose` 호출은 **프로젝트 이름** 하에 동작합니다 — 기본은 디렉터리 이름(`my-app`), `-p projectname` 또는 `COMPOSE_PROJECT_NAME` 환경 변수로 덮어쓸 수 있습니다. 프로젝트 이름은 Compose가 만드는 모든 것의 네임스페이스입니다.

- 컨테이너: `<project>-<service>-<replica>`(예: `my-app-web-1`).
- 네트워크: `<project>_<network>`(기본 네트워크는 `<project>_default`).
- 볼륨: `<project>_<volume>`.

이름이 다른 두 프로젝트는 같은 엔진에서 충돌 없이 공존합니다 — 같은 Compose 파일의 `dev`, `staging`, `feature-branch-x` 사본을 세 디렉터리에서 동시에 돌릴 수 있습니다.

프로젝트 이름은 또한 조정 시 어떤 컨테이너가 자기 것인지 Compose가 아는 방법이기도 합니다. 이미지나 명령으로 검색하지 않고, 자기가 만드는 모든 자원에 붙이는 `com.docker.compose.project=<project>` 라벨로 필터링합니다.

### C. 의존성 DAG와 서비스 시작 순서

`depends_on`은 서비스 B가 서비스 A에 의존한다고 선언하게 해 줍니다. Compose는 이 선언을 방향 비순환 그래프의 간선으로 다루고, 두 곳에서 사용합니다.

- **시작 순서.** 위상 정렬 순서로 시작. 의존성이 먼저 시작.
- **종료 순서.** 역위상 정렬 순서로 종료. 의존자가 먼저 종료.

단순 형태인 `depends_on: [db]`는 "나보다 먼저 `db`를 시작해라"입니다. 그것은 *"`db`가 연결을 받을 준비가 될 때까지 기다려라"가 아닙니다*. Postgres 컨테이너는 postgres 프로세스가 존재하는 순간 "시작"된 것으로 간주되며, 데이터베이스 파일 초기화와 리스닝 시작은 그보다 한참 뒤입니다.

풍부한 형태가 `condition`으로 이를 해결합니다.

```yaml
depends_on:
  db:
    condition: service_healthy   # db의 헬스체크가 healthy를 보고할 때까지 대기
  cache:
    condition: service_started   # 기본 — 컨테이너가 시작되기만 하면 됨
```

`service_healthy`는 `db`가 `healthcheck`(컨테이너 안에서 Docker가 주기적으로 실행하는 명령. `start_period + retries × interval` 동안 명령이 성공하면 컨테이너가 "healthy")를 정의해야 합니다. "Postgres가 실제로 쿼리를 받기 시작한 뒤에 API 서버를 시작해라"를 표현하는 방법입니다.

호스트 측 대기에 의존할 수 없는 경우(예: Compose 안과 밖에서 모두 동작해야 하는 앱), 표준 패턴은 엔트리포인트의 작은 wait-for-it 스크립트로 백오프를 두고 연결을 재시도하는 것입니다. Compose의 `depends_on`은 베스트 에포트(best-effort) 오케스트레이션이며, 정확성은 여전히 앱의 책임입니다.

### D. 기본 네트워크와 서비스 디스커버리

각 프로젝트마다 Compose는 `<project>_default` 이름의 사용자 정의 브리지 네트워크 한 개를 만듭니다. 명시적으로 지시하지 않는 한 모든 서비스가 여기 합류합니다. 사용자 정의 브리지에서는 Docker DNS 리졸버가 컨테이너의 IP를 **서비스 이름**으로 반환합니다.

```yaml
services:
  web:
    image: myapp
    environment:
      - DATABASE_URL=postgres://user:pass@db:5432/mydb   # 'db'가 리졸브됨
  db:
    image: postgres:16
```

`web`이 `db`라는 호스트네임으로 `db`에 연결합니다. IP 디스커버리도, 호스트 이름의 환경 변수 주입도 필요 없습니다 — 두 컨테이너가 같은 Compose 관리 네트워크에 있고, Docker 데몬이 이름을 아는 임베디드 DNS 서버(컨테이너 내부 127.0.0.11)를 돌리기 때문입니다.

여러 네트워크를 정의하고 서비스를 특정 네트워크에 배치할 수도 있습니다 — 예를 들어 `frontend` 네트워크와 `backend` 네트워크를 격리할 때. 두 네트워크에 합류한 서비스가 그 사이의 다리 역할을 합니다.

이 기본 네트워크 동작이 Compose에서 가장 과소평가된 기능입니다. 같은 구성을 raw `docker run`으로 하려면 `docker network create`, 모든 컨테이너에 `--network`, 그리고 IP를 알거나 `--network-alias`를 써야 합니다.

### E. 환경 변수 보간과 멀티 파일 오버라이드

YAML은 `${VAR}`와 `${VAR:-default}` 보간을 지원하며, 파일이 파싱되기 *전에* 평가됩니다. 출처는 다음 순서입니다.

1. 프로젝트 디렉터리의 `.env` 파일(있다면).
2. `docker compose`를 실행한 사용자의 셸 환경.

같은 `docker-compose.yml`을 환경별로 설정 가능하게 만드는 방법입니다 — `image: myapp:${TAG:-latest}`이라 쓰고 CI는 `TAG=v1.2.3`을 설정, 개발자는 비워 두면 됩니다.

더 큰 차이(개발/프로덕션 포트 다름, `mailhog` 같은 개발 전용 서비스 추가, 다른 볼륨 매핑)를 위해 Compose는 **멀티 파일 오버라이드**를 지원합니다.

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

Compose가 파일을 순서대로 머지합니다 — 뒤 파일이 스칼라 값을 덮어쓰고, 리스트에는 추가하고, 맵은 키별로 머지합니다. 관습 — 베이스 파일과 환경별 오버라이드 하나씩, 또는 베이스 + 자동 로드되는 `docker-compose.override.yml`(로컬 개발용).

### F. Compose Spec, 단순한 CLI 포맷이 아니다

Compose는 한때 자체 포맷의 별도 파이썬 도구였습니다. 오늘날 **Compose Specification**은 compose-spec.io의 Compose 커뮤니티가 유지하는 오픈 표준입니다. 같은 YAML을 다음이 소비합니다.

- `docker compose` — 공식 Go 기반 플러그인, 가장 흔한 소비자.
- `docker-compose`(V1) — 옛 파이썬 도구, 현재 폐기.
- Podman Compose, BuildKit, 그리고 다양한 PaaS 배포 시스템 같은 다른 도구들.

포맷은 더 이상 Docker 전용이 아닙니다. "멀티 서비스 로컬 또는 소규모 프로덕션 스택을 선언한다"의 *사실상* 표준 포맷입니다.

### 이론에서 아래의 YAML로

- `version: "3.x"`(현대 Compose에선 선택) — 옛날엔 엄격한 스펙 버전이었지만, 지금은 정보성 필드이며 항상 최신 스키마가 가정됩니다.
- `services:` — 원하는 컨테이너 집합. `depends_on`에서 의존성 그래프가 빌드됩니다.
- `networks:` — 원하는 프로젝트 스코프 브리지 집합. 명시되지 않은 기본은 `<project>_default`.
- `volumes:` — 원하는 명명 볼륨 집합. 한 번 만들고 명시적으로 제거하지 않는 한 `up`/`down` 사이클 사이에 공유됩니다.
- `condition: service_healthy`의 `depends_on` — §C에서 설명한 시작 순서 DAG와 헬스체크 상태 기계를 연결.
- `environment`와 `${VAR}` — §E의 보간 파이프라인. `.env`와 셸이 머지된 집합에 대해 평가.
- `docker compose up -d` — 데몬의 현재 상태에서 YAML 상태로 분리 모드 조정.
- `docker compose ps` — `project=<project>` 라벨 필터로 프로젝트 스코프 컨테이너 나열.
- `docker compose logs -f <service>` — 그 서비스에 속하는 모든 컨테이너의 JSON 로그 스트림.

남은 본문은 YAML 레퍼런스입니다. Compose가 컨테이너를 재생성하거나 하지 않은 이유가 궁금할 때마다, 그 라벨의 config 해시와 편집한 YAML 사이의 diff를 보세요.

---

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
