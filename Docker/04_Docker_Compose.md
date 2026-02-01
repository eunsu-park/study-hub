# Docker Compose

## 1. Docker Compose란?

Docker Compose는 **여러 컨테이너를 정의하고 실행**하는 도구입니다. YAML 파일 하나로 전체 애플리케이션 스택을 관리합니다.

### 왜 Docker Compose를 사용할까요?

**일반 Docker 명령어:**
```bash
# 네트워크 생성
docker network create myapp-network

# 데이터베이스 실행
docker run -d \
  --name db \
  --network myapp-network \
  -e POSTGRES_PASSWORD=secret \
  -v pgdata:/var/lib/postgresql/data \
  postgres:15

# 백엔드 실행
docker run -d \
  --name backend \
  --network myapp-network \
  -e DATABASE_URL=postgres://... \
  -p 3000:3000 \
  my-backend

# 프론트엔드 실행
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
# 버전 확인
docker compose version
# Docker Compose version v2.23.0

# 또는 (구버전)
docker-compose --version
```

> **참고:** `docker-compose` (하이픈)은 구버전, `docker compose` (공백)은 신버전입니다.

---

## 3. docker-compose.yml 기본 구조

```yaml
# docker-compose.yml

services:
  서비스명1:
    image: 이미지명
    ports:
      - "호스트:컨테이너"
    environment:
      - 변수=값
    volumes:
      - 볼륨:경로
    depends_on:
      - 다른서비스

  서비스명2:
    build: ./경로
    ...

volumes:
  볼륨명:

networks:
  네트워크명:
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
    build: .                    # 현재 디렉토리의 Dockerfile

  api:
    build:
      context: ./backend        # 빌드 컨텍스트
      dockerfile: Dockerfile    # Dockerfile 경로
      args:                     # 빌드 인자
        - NODE_ENV=production
```

### ports - 포트 매핑

```yaml
services:
  web:
    ports:
      - "8080:80"              # 호스트:컨테이너
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

  # 또는 key: value 형식
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
      - pgdata:/var/lib/postgresql/data    # Named volume
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql  # 바인드 마운트

  app:
    volumes:
      - ./src:/app/src                      # 소스 코드 마운트 (개발용)
      - /app/node_modules                   # 익명 볼륨 (덮어쓰기 방지)

volumes:
  pgdata:                                   # Named volume 선언
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
      - frontend-net

  backend:
    networks:
      - frontend-net
      - backend-net

  db:
    networks:
      - backend-net

networks:
  frontend-net:
  backend-net:
```

### restart - 재시작 정책

```yaml
services:
  web:
    restart: always              # 항상 재시작

  api:
    restart: unless-stopped      # 수동 중지 전까지 재시작

  worker:
    restart: on-failure          # 실패 시 재시작
```

### healthcheck - 헬스체크

```yaml
services:
  api:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

---

## 5. Docker Compose 명령어

### 실행

```bash
# 실행 (포그라운드)
docker compose up

# 백그라운드 실행
docker compose up -d

# 이미지 재빌드 후 실행
docker compose up --build

# 특정 서비스만 실행
docker compose up -d web api
```

### 중지/삭제

```bash
# 중지
docker compose stop

# 중지 및 컨테이너 삭제
docker compose down

# 볼륨도 함께 삭제
docker compose down -v

# 이미지도 함께 삭제
docker compose down --rmi all
```

### 상태 확인

```bash
# 서비스 목록
docker compose ps

# 로그 확인
docker compose logs

# 특정 서비스 로그
docker compose logs api

# 실시간 로그
docker compose logs -f
```

### 서비스 관리

```bash
# 재시작
docker compose restart

# 특정 서비스 재시작
docker compose restart api

# 서비스 스케일링
docker compose up -d --scale api=3

# 서비스 내 명령 실행
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
      - DATABASE_URL=postgres://user:pass@db:5432/mydb
    depends_on:
      - db

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=mydb
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  pgdata:
```

**app/Dockerfile:**
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
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
  # 프론트엔드 (React)
  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend

  # 백엔드 (Node.js)
  backend:
    build: ./backend
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DB_HOST=db
      - DB_NAME=myapp
      - REDIS_HOST=redis
    depends_on:
      - db
      - redis

  # 데이터베이스 (PostgreSQL)
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=myapp
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql

  # 캐시 (Redis)
  redis:
    image: redis:7-alpine
    volumes:
      - redisdata:/data

  # 관리 도구 (pgAdmin)
  pgadmin:
    image: dpage/pgadmin4
    environment:
      - PGADMIN_DEFAULT_EMAIL=admin@example.com
      - PGADMIN_DEFAULT_PASSWORD=admin
    ports:
      - "5050:80"
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
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
    volumes:
      - .:/app                    # 소스 코드 마운트
      - /app/node_modules         # node_modules 보호
    environment:
      - NODE_ENV=development
    command: npm run dev          # 개발 서버 실행

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_PASSWORD=devpass
    ports:
      - "5432:5432"              # 로컬 접속 허용
```

**실행:**
```bash
# 개발 환경
docker compose -f docker-compose.dev.yml up

# 프로덕션 환경
docker compose -f docker-compose.yml up -d
```

---

## 7. 유용한 패턴

### 환경별 설정 분리

```yaml
# docker-compose.yml (기본)
services:
  app:
    image: myapp

# docker-compose.override.yml (개발용, 자동 병합)
services:
  app:
    build: .
    volumes:
      - .:/app

# docker-compose.prod.yml (프로덕션용)
services:
  app:
    restart: always
```

```bash
# 개발: docker-compose.yml + docker-compose.override.yml 자동 병합
docker compose up

# 프로덕션
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 서비스 대기 (wait-for-it)

```yaml
services:
  app:
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:15
    healthcheck:
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

## 다음 단계

[05_Practical_Examples.md](./05_Practical_Examples.md)에서 실제 프로젝트에 Docker를 적용해봅시다!
