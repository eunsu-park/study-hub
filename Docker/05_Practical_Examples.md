# Docker 실전 예제

이 문서에서는 실제 프로젝트에 Docker를 적용하는 방법을 단계별로 실습합니다.

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

// PostgreSQL 연결
const pool = new Pool({
  host: process.env.DB_HOST || 'localhost',
  port: process.env.DB_PORT || 5432,
  database: process.env.DB_NAME || 'myapp',
  user: process.env.DB_USER || 'postgres',
  password: process.env.DB_PASSWORD || 'password'
});

// 라우트
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
FROM node:18-alpine

WORKDIR /app

# 의존성 설치 (캐시 활용)
COPY package*.json ./
RUN npm install --production

# 소스 코드 복사
COPY . .

# 비루트 사용자로 실행
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
USER appuser

EXPOSE 3000

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
-- 초기 테이블 생성
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 샘플 데이터
INSERT INTO users (name, email) VALUES
    ('홍길동', 'hong@example.com'),
    ('김철수', 'kim@example.com'),
    ('이영희', 'lee@example.com');
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
      - DB_HOST=db
      - DB_PORT=5432
      - DB_NAME=${DB_NAME}
      - DB_USER=${DB_USER}
      - DB_PASSWORD=${DB_PASSWORD}
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=${DB_NAME}
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER} -d ${DB_NAME}"]
      interval: 5s
      timeout: 5s
      retries: 5
    ports:
      - "5432:5432"

volumes:
  pgdata:
```

### 실행 및 테스트

```bash
# 디렉토리 생성 및 이동
mkdir -p nodejs-postgres-app/backend/src nodejs-postgres-app/db
cd nodejs-postgres-app

# (위 파일들 생성 후)

# 실행
docker compose up -d

# 상태 확인
docker compose ps

# 로그 확인
docker compose logs -f backend

# API 테스트
curl http://localhost:3000/
curl http://localhost:3000/health
curl http://localhost:3000/users

# 사용자 추가
curl -X POST http://localhost:3000/users \
  -H "Content-Type: application/json" \
  -d '{"name": "박민수", "email": "park@example.com"}'

# 정리
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
<html lang="ko">
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
        <p>이 앱은 Docker로 배포되었습니다.</p>
        <p>빌드 시간: {new Date().toLocaleString()}</p>
      </div>
    </div>
  );
}

export default App;
```

**Dockerfile (멀티 스테이지 빌드):**
```dockerfile
# Stage 1: 빌드
FROM node:18-alpine AS builder

WORKDIR /app

# 의존성 설치
COPY package*.json ./
RUN npm install

# 소스 복사 및 빌드
COPY . .
RUN npm run build

# Stage 2: Nginx로 서빙
FROM nginx:alpine

# 빌드 결과물 복사
COPY --from=builder /app/build /usr/share/nginx/html

# Nginx 설정 복사
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

**nginx.conf:**
```nginx
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    # React Router 지원 (SPA)
    location / {
        try_files $uri $uri/ /index.html;
    }

    # 정적 파일 캐싱
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # gzip 압축
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
# 빌드 및 실행
docker compose up -d --build

# 브라우저에서 확인
# http://localhost

# 정리
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
│   └── (React 프로젝트)
├── backend/
│   ├── Dockerfile
│   └── (Express 프로젝트)
└── db/
    └── init.sql
```

**docker-compose.yml:**
```yaml
services:
  # 프론트엔드
  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    restart: unless-stopped

  # 백엔드 API
  backend:
    build: ./backend
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DB_HOST=db
      - DB_PORT=5432
      - DB_NAME=${DB_NAME}
      - DB_USER=${DB_USER}
      - DB_PASSWORD=${DB_PASSWORD}
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    restart: unless-stopped

  # PostgreSQL 데이터베이스
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=${DB_NAME}
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # Redis 캐시
  redis:
    image: redis:7-alpine
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
      dockerfile: Dockerfile.dev
    ports:
      - "3001:3000"
    volumes:
      - ./frontend/src:/app/src
    environment:
      - REACT_APP_API_URL=http://localhost:3000

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    volumes:
      - ./backend/src:/app/src
    environment:
      - NODE_ENV=development
    command: npm run dev

  db:
    ports:
      - "5432:5432"

  redis:
    ports:
      - "6379:6379"
```

### 실행 명령어

```bash
# 프로덕션
docker compose up -d

# 개발
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# 특정 서비스 로그
docker compose logs -f backend

# 데이터베이스 접속
docker compose exec db psql -U ${DB_USER} -d ${DB_NAME}

# Redis CLI
docker compose exec redis redis-cli

# 전체 정리
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
      - "8080:80"
    environment:
      - WORDPRESS_DB_HOST=db
      - WORDPRESS_DB_USER=wordpress
      - WORDPRESS_DB_PASSWORD=${DB_PASSWORD}
      - WORDPRESS_DB_NAME=wordpress
    volumes:
      - wordpress_data:/var/www/html
    depends_on:
      - db
    restart: unless-stopped

  db:
    image: mysql:8
    environment:
      - MYSQL_DATABASE=wordpress
      - MYSQL_USER=wordpress
      - MYSQL_PASSWORD=${DB_PASSWORD}
      - MYSQL_ROOT_PASSWORD=${DB_ROOT_PASSWORD}
    volumes:
      - db_data:/var/lib/mysql
    restart: unless-stopped

  # phpMyAdmin (선택사항)
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
# 컨테이너 내부 접속
docker compose exec backend sh

# 실시간 로그 모니터링
docker compose logs -f

# 리소스 사용량 확인
docker stats

# 네트워크 확인
docker network ls
docker network inspect <network_name>
```

### 정리

```bash
# 중지된 컨테이너 삭제
docker container prune

# 사용하지 않는 이미지 삭제
docker image prune

# 사용하지 않는 볼륨 삭제
docker volume prune

# 전체 정리 (주의!)
docker system prune -a --volumes
```

### 백업

```bash
# 볼륨 백업
docker run --rm \
  -v pgdata:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/pgdata-backup.tar.gz -C /data .

# 볼륨 복원
docker run --rm \
  -v pgdata:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/pgdata-backup.tar.gz -C /data
```

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
