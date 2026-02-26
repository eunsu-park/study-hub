# Docker 실전 예제

**이전**: [Docker Compose](./04_Docker_Compose.md) | **다음**: [Kubernetes 입문](./06_Kubernetes_Intro.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Docker Compose를 사용하여 완전한 Node.js + Express + PostgreSQL 애플리케이션을 구축할 수 있다
2. Nginx로 서빙하는 React 애플리케이션을 위한 멀티 스테이지 Docker 빌드(multi-stage Docker build)를 구현할 수 있다
3. 프론트엔드, 백엔드, 데이터베이스, 캐시 서비스로 구성된 풀스택(full-stack) 애플리케이션을 조합할 수 있다
4. Docker Compose를 사용하여 WordPress와 MySQL을 구성해 빠르게 CMS를 배포할 수 있다
5. 로그 모니터링, 컨테이너 접속, 네트워크 검사 등의 디버깅 기법을 적용할 수 있다
6. 영구 데이터를 위한 볼륨 백업(backup) 및 복원(restore) 전략을 구현할 수 있다

---

Docker 명령어와 Compose 문법을 아는 것은 절반에 불과합니다 — 진짜 실력은 실제 프로젝트에 적용할 때 드러납니다. 이 레슨에서는 간단한 API와 데이터베이스 구성부터 React, Node.js, PostgreSQL, Redis를 갖춘 풀스택 애플리케이션까지, 점차 복잡해지는 네 가지 실전 시나리오를 단계별로 살펴봅니다. 이 예제들을 직접 따라 하면서 자신의 프로젝트를 Docker화(Dockerize)하는 데 필요한 실전 감각과 문제 해결 능력을 키울 수 있습니다.

---

## 예제 1: Node.js + Express + PostgreSQL

### 프로젝트 구조

```
nodejs-postgres-app/
├── docker-compose.yml
├── .env
├── .dockerignore
├── backend/
│   ├── Dockerfile
│   ├── package.json
│   └── src/
│       └── index.js
└── db/
    └── init.sql
```

### 파일 생성

**backend/package.json:**
```json
{
  "name": "express-postgres-app",
  "version": "1.0.0",
  "main": "src/index.js",
  "scripts": {
    "start": "node src/index.js",
    "dev": "node --watch src/index.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "pg": "^8.11.3"
  }
}
```

**backend/src/index.js:**
```javascript
const express = require('express');
const { Pool } = require('pg');

const app = express();
app.use(express.json());

// PostgreSQL connection
const pool = new Pool({
  host: process.env.DB_HOST || 'localhost',
  port: process.env.DB_PORT || 5432,
  database: process.env.DB_NAME || 'myapp',
  user: process.env.DB_USER || 'postgres',
  password: process.env.DB_PASSWORD || 'password'
});

// Routes
app.get('/', (req, res) => {
  res.json({ message: 'Hello Docker!', status: 'running' });
});

app.get('/health', async (req, res) => {
  try {
    await pool.query('SELECT 1');
    res.json({ status: 'healthy', database: 'connected' });
  } catch (error) {
    res.status(500).json({ status: 'unhealthy', error: error.message });
  }
});

app.get('/users', async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM users ORDER BY id');
    res.json(result.rows);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/users', async (req, res) => {
  const { name, email } = req.body;
  try {
    const result = await pool.query(
      'INSERT INTO users (name, email) VALUES ($1, $2) RETURNING *',
      [name, email]
    );
    res.status(201).json(result.rows[0]);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

**backend/Dockerfile:**
```dockerfile
# Alpine: ~175 MB vs ~1 GB full image — smaller attack surface and faster CI pulls
FROM node:18-alpine

WORKDIR /app

# Copy dependency manifest first — changes less often, so Docker caches the install layer
COPY package*.json ./
# --production: skip devDependencies — smaller image and fewer potential vulnerabilities
RUN npm install --production

# Copy source code last — source changes don't invalidate the npm install cache
COPY . .

# Non-root user: limits damage if an attacker escapes the container
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
USER appuser

# Documentation only — does not actually publish the port (use -p for that)
EXPOSE 3000

# Exec form: process runs as PID 1, receives SIGTERM for graceful shutdown
CMD ["npm", "start"]
```

**backend/.dockerignore:**
```
node_modules
npm-debug.log
.git
.env
*.md
```

**db/init.sql:**
```sql
-- Create initial table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sample data
INSERT INTO users (name, email) VALUES
    ('John Doe', 'john@example.com'),
    ('Jane Smith', 'jane@example.com'),
    ('Bob Johnson', 'bob@example.com');
```

**.env:**
```
DB_PASSWORD=secretpassword123
DB_USER=appuser
DB_NAME=myapp
```

**docker-compose.yml:**
```yaml
services:
  backend:
    build: ./backend
    ports:
      - "3000:3000"
    environment:
      - DB_HOST=db             # Compose DNS resolves 'db' to the database container's IP
      - DB_PORT=5432
      - DB_NAME=${DB_NAME}
      - DB_USER=${DB_USER}
      - DB_PASSWORD=${DB_PASSWORD}   # Read from .env file — keeps secrets out of version control
    depends_on:
      db:
        condition: service_healthy   # Wait until db passes health check, not just until it starts
    restart: unless-stopped          # Auto-restart on crash, but respect manual docker stop

  db:
    image: postgres:15-alpine        # Alpine variant: smaller image, faster pulls
    environment:
      - POSTGRES_DB=${DB_NAME}
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data          # Named volume — data survives container removal
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql  # Auto-runs on first start only
    healthcheck:
      # pg_isready verifies Postgres is accepting connections — not just that the process exists
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER} -d ${DB_NAME}"]
      interval: 5s
      timeout: 5s
      retries: 5
    ports:
      - "5432:5432"            # Expose to host for local DB tools (pgAdmin, DBeaver, etc.)

volumes:
  pgdata:
```

### 실행 및 테스트

```bash
# Create directories and navigate
mkdir -p nodejs-postgres-app/backend/src nodejs-postgres-app/db
cd nodejs-postgres-app

# (After creating above files)

# Run
docker compose up -d

# Check status
docker compose ps

# Check logs
docker compose logs -f backend

# API tests
curl http://localhost:3000/
curl http://localhost:3000/health
curl http://localhost:3000/users

# Add user
curl -X POST http://localhost:3000/users \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice Park", "email": "alice@example.com"}'

# Cleanup
docker compose down -v
```

---

## 예제 2: React + Nginx (프로덕션 빌드)

### 프로젝트 구조

```
react-nginx-app/
├── docker-compose.yml
├── Dockerfile
├── nginx.conf
├── package.json
├── public/
│   └── index.html
└── src/
    ├── App.js
    └── index.js
```

### 파일 생성

**package.json:**
```json
{
  "name": "react-docker-app",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build"
  },
  "browserslist": {
    "production": [">0.2%", "not dead", "not op_mini all"],
    "development": ["last 1 chrome version"]
  }
}
```

**public/index.html:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>React Docker App</title>
</head>
<body>
  <div id="root"></div>
</body>
</html>
```

**src/index.js:**
```javascript
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
```

**src/App.js:**
```javascript
import React, { useState, useEffect } from 'react';

function App() {
  const [message, setMessage] = useState('Loading...');

  useEffect(() => {
    setMessage('Hello from React in Docker!');
  }, []);

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '100vh',
      fontFamily: 'Arial, sans-serif'
    }}>
      <div style={{ textAlign: 'center' }}>
        <h1>{message}</h1>
        <p>This app is deployed with Docker.</p>
        <p>Build time: {new Date().toLocaleString()}</p>
      </div>
    </div>
  );
}

export default App;
```

**Dockerfile (멀티 스테이지 빌드):**
```dockerfile
# Stage 1: Build — node_modules + build toolchain (~300 MB) are discarded after this stage
FROM node:18-alpine AS builder

WORKDIR /app

# Copy dependency manifest first — changes less often, so Docker caches the install layer
COPY package*.json ./
RUN npm install

# Copy source and build
COPY . .
RUN npm run build

# Stage 2: Serve with Nginx — final image contains only static files (~25 MB)
FROM nginx:alpine

# --from=builder: pull artifacts from the build stage without carrying over node_modules
COPY --from=builder /app/build /usr/share/nginx/html

# Custom config for SPA routing, caching, and compression
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

# "daemon off;" keeps nginx in the foreground so Docker can track the process as PID 1
CMD ["nginx", "-g", "daemon off;"]
```

**nginx.conf:**
```nginx
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    # React Router support (SPA) — all routes fall back to index.html so client-side routing works
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Static file caching — hashed filenames allow aggressive caching; "immutable" prevents revalidation
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # gzip compression — reduces transfer size by 60-80% for text-based assets
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml;
}
```

**docker-compose.yml:**
```yaml
services:
  frontend:
    build: .
    ports:
      - "80:80"
    restart: unless-stopped
```

### 실행

```bash
# Build and run
docker compose up -d --build

# Check in browser
# http://localhost

# Cleanup
docker compose down
```

---

## 예제 3: 전체 스택 (React + Node.js + PostgreSQL + Redis)

### 프로젝트 구조

```
fullstack-app/
├── docker-compose.yml
├── docker-compose.dev.yml
├── .env
├── frontend/
│   ├── Dockerfile
│   ├── nginx.conf
│   └── (React project)
├── backend/
│   ├── Dockerfile
│   └── (Express project)
└── db/
    └── init.sql
```

**docker-compose.yml:**
```yaml
services:
  # Frontend
  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    restart: unless-stopped      # Auto-restart on crash, but respect manual docker stop

  # Backend API
  backend:
    build: ./backend
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DB_HOST=db               # Compose DNS resolves service names to container IPs
      - DB_PORT=5432
      - DB_NAME=${DB_NAME}
      - DB_USER=${DB_USER}
      - DB_PASSWORD=${DB_PASSWORD}
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      db:
        condition: service_healthy    # Wait for Postgres to accept connections before starting
      redis:
        condition: service_started    # Redis starts fast — no health check needed
    restart: unless-stopped

  # PostgreSQL database
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=${DB_NAME}
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data       # Named volume — data survives container removal
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      # pg_isready verifies Postgres is accepting connections — not just that the process exists
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # Redis cache
  redis:
    image: redis:7-alpine
    # --appendonly yes: persist writes to disk — prevents data loss on restart (at slight perf cost)
    command: redis-server --appendonly yes
    volumes:
      - redisdata:/data
    restart: unless-stopped

volumes:
  pgdata:
  redisdata:
```

**docker-compose.dev.yml (개발용 오버라이드):**
```yaml
services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev     # Dev Dockerfile may include hot-reload tooling
    ports:
      - "3001:3000"                  # Different host port avoids conflict with backend's :3000
    volumes:
      - ./frontend/src:/app/src      # Bind mount — edit on host, see changes instantly via hot-reload
    environment:
      - REACT_APP_API_URL=http://localhost:3000

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    volumes:
      - ./backend/src:/app/src       # Bind mount — enables live-reload for server code too
    environment:
      - NODE_ENV=development
    command: npm run dev             # Override CMD — use file-watching dev server instead of production start

  db:
    ports:
      - "5432:5432"                  # Expose to host so local DB tools can connect directly

  redis:
    ports:
      - "6379:6379"                  # Expose to host for redis-cli and debugging
```

### 실행 명령어

```bash
# Production
docker compose up -d

# Development
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Specific service logs
docker compose logs -f backend

# Database access
docker compose exec db psql -U ${DB_USER} -d ${DB_NAME}

# Redis CLI
docker compose exec redis redis-cli

# Full cleanup
docker compose down -v
```

---

## 예제 4: WordPress + MySQL

### docker-compose.yml

```yaml
services:
  wordpress:
    image: wordpress:latest
    ports:
      - "8080:80"              # Non-standard host port to avoid conflicts if another service uses :80
    environment:
      - WORDPRESS_DB_HOST=db   # Compose DNS resolves 'db' to the MySQL container
      - WORDPRESS_DB_USER=wordpress
      - WORDPRESS_DB_PASSWORD=${DB_PASSWORD}
      - WORDPRESS_DB_NAME=wordpress
    volumes:
      - wordpress_data:/var/www/html   # Persist themes, plugins, and uploads across restarts
    depends_on:
      - db
    restart: unless-stopped    # Auto-restart on crash, but respect manual docker stop

  db:
    image: mysql:8
    environment:
      - MYSQL_DATABASE=wordpress
      - MYSQL_USER=wordpress
      - MYSQL_PASSWORD=${DB_PASSWORD}
      - MYSQL_ROOT_PASSWORD=${DB_ROOT_PASSWORD}   # Keep root password separate from app password
    volumes:
      - db_data:/var/lib/mysql         # Named volume — database files survive container removal
    restart: unless-stopped

  # phpMyAdmin (optional) — web-based DB admin for quick debugging; remove in production
  phpmyadmin:
    image: phpmyadmin:latest
    ports:
      - "8081:80"
    environment:
      - PMA_HOST=db
      - PMA_USER=wordpress
      - PMA_PASSWORD=${DB_PASSWORD}
    depends_on:
      - db

volumes:
  wordpress_data:
  db_data:
```

**.env:**
```
DB_PASSWORD=wordpresspass123
DB_ROOT_PASSWORD=rootpass123
```

### 실행

```bash
docker compose up -d

# WordPress: http://localhost:8080
# phpMyAdmin: http://localhost:8081
```

---

## 유용한 명령어 모음

### 디버깅

```bash
# Access container
docker compose exec backend sh

# Real-time log monitoring
docker compose logs -f

# Check resource usage
docker stats

# Check network
docker network ls
docker network inspect <network_name>
```

### 정리

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove unused volumes
docker volume prune

# Full cleanup — removes ALL unused images, containers, networks, AND volumes (caution!)
docker system prune -a --volumes
```

### 백업

```bash
# Backup volume — uses a throwaway Alpine container to tar the volume contents
docker run --rm \
  -v pgdata:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/pgdata-backup.tar.gz -C /data .
# --rm: container auto-removes after the backup completes (no leftover containers)

# Restore volume — extracts the tarball into the named volume
docker run --rm \
  -v pgdata:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/pgdata-backup.tar.gz -C /data
```

---

## 연습 문제

### 연습 1: Node.js + PostgreSQL 예제 확장

예제 1을 기반으로 새로운 API 엔드포인트(endpoint)를 추가하고 데이터 영속성(data persistence)을 검증합니다.

1. 예제 1을 따라 Node.js + PostgreSQL 스택을 실행합니다
2. `backend/src/index.js`에 ID로 사용자를 삭제하는 `DELETE /users/:id` 엔드포인트를 추가합니다
3. 백엔드 이미지만 재빌드합니다: `docker compose build backend`
4. 백엔드 서비스만 재시작합니다: `docker compose up -d backend`
5. `curl -X DELETE http://localhost:3000/users/1`로 사용자를 삭제합니다
6. 사용자가 삭제되었는지 확인합니다: `curl http://localhost:3000/users`
7. `-v` 없이 `docker compose down`을 실행하고 재시작 후 users 테이블에 데이터가 남아있는지 확인합니다

### 연습 2: React + Nginx 멀티 스테이지 빌드 분석

예제 2의 React + Nginx 멀티 스테이지 빌드를 분석하고 최적화합니다.

1. 예제 2를 따라 React + Nginx 이미지를 빌드합니다
2. `docker history <이미지명>`으로 모든 레이어(layer)와 크기를 확인합니다
3. `docker images`로 최종 이미지 크기를 일반 `node:18-alpine` 이미지와 비교합니다
4. `node_modules`, `.git`, 테스트 파일을 제외하는 `.dockerignore` 파일을 추가하고, 재빌드하여 크기를 비교합니다
5. `nginx.conf`를 수정하여 `/index.html`에 `Cache-Control: no-store` 헤더를 추가하고 JS/CSS 파일에는 1년 캐시를 설정합니다
6. 재빌드하고 `curl -I http://localhost`로 헤더를 확인합니다

### 연습 3: 풀스택(Full-Stack) 디버깅

React + Node.js + PostgreSQL + Redis 풀스택 예제를 사용하여 디버깅을 실습합니다.

1. 예제 3의 풀스택을 시작합니다
2. `docker compose ps`와 `docker compose logs`를 사용하여 실패한 컨테이너(있다면)를 파악합니다
3. PostgreSQL 데이터베이스에 직접 접속합니다: `docker compose exec db psql -U $DB_USER -d $DB_NAME`
4. Redis CLI에 접속합니다: `docker compose exec redis redis-cli`
5. 네트워크(network)를 검사합니다: `docker network inspect <프로젝트>_default`에서 어떤 컨테이너가 연결되어 있는지 문서화합니다
6. `docker stats`를 사용하여 네 가지 서비스 간의 CPU와 메모리 사용량을 비교합니다
7. Redis 서비스만 중지하고 백엔드가 캐시 없이 어떻게 동작하는지 관찰합니다

### 연습 4: WordPress 볼륨(Volume) 백업

예제 4에서 WordPress를 설정하고 데이터 백업 및 복원을 실습합니다.

1. 예제 4의 WordPress + MySQL 스택을 시작합니다
2. 브라우저에서 `http://localhost:8080`으로 WordPress 설치를 완료합니다
3. 테스트 블로그 글을 작성합니다
4. `db_data` 볼륨을 백업합니다:
   ```bash
   docker run --rm \
     -v <project>_db_data:/data \
     -v $(pwd):/backup \
     alpine tar czf /backup/db-backup.tar.gz -C /data .
   ```
5. `docker compose down -v`로 모든 데이터를 삭제합니다
6. 볼륨을 복원하고 스택을 재시작하여 WordPress 글이 남아있는지 확인합니다

### 연습 5: 커스텀 풀스택 프로젝트

이 레슨의 패턴을 사용하여 자신만의 프로젝트를 Docker화합니다.

1. 최소 두 가지 컴포넌트(앱 + 데이터베이스)가 있는 간단한 애플리케이션을 선택합니다 (예: 블로그, 태스크 매니저, REST API)
2. 각 서비스에 대한 `Dockerfile`을 모범 사례에 따라 작성합니다: 비루트 사용자(non-root user), 레이어 캐싱(layer caching), 해당되는 경우 멀티 스테이지(multi-stage)
3. 적절한 `depends_on`, 헬스 체크(health check), 네임드 볼륨(named volume), 시크릿(secret)을 위한 `.env` 파일이 포함된 `docker-compose.yml`을 작성합니다
4. 소스 코드 볼륨 마운트가 있는 로컬 개발용 `docker-compose.dev.yml`을 추가합니다
5. 개발 및 프로덕션(production) 명령을 모두 포함하여 스택 시작 방법을 `README.md`에 문서화합니다
6. `docker compose down`과 `docker compose up` 사이클에서 데이터가 유지되는지 확인합니다

---

## 학습 완료!

Docker 학습을 완료했습니다. 다음 단계로:

1. **실습**: 자신의 프로젝트를 Docker화 해보기
2. **CI/CD**: GitHub Actions와 Docker 연동
3. **오케스트레이션**: Kubernetes 기초 학습
4. **보안**: Docker 보안 베스트 프랙티스

### 추가 학습 자료

- [Docker 공식 문서](https://docs.docker.com/)
- [Docker Hub](https://hub.docker.com/)
- [Play with Docker](https://labs.play-with-docker.com/) - 브라우저에서 Docker 실습

---

**이전**: [Docker Compose](./04_Docker_Compose.md) | **다음**: [Kubernetes 입문](./06_Kubernetes_Intro.md)
