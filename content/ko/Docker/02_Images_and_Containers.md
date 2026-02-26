# Docker 이미지와 컨테이너

**이전**: [Docker 기초](./01_Docker_Basics.md) | **다음**: [Dockerfile](./03_Dockerfile.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Docker 이미지의 레이어(layer) 구조와 저장 방식을 설명할 수 있다
2. 레지스트리(registry), 저장소(repository), 태그(tag)를 포함한 이미지 명명 규칙을 설명할 수 있다
3. Docker CLI 명령어를 사용하여 이미지를 검색, 다운로드, 목록 조회, 상세 검사, 삭제할 수 있다
4. 포트 매핑(port mapping), 환경 변수(environment variable), 볼륨(volume), 인터랙티브 모드(interactive mode) 옵션을 사용하여 컨테이너를 실행할 수 있다
5. 컨테이너 생명주기(lifecycle) 작업인 시작, 중지, 재시작, 삭제를 관리할 수 있다
6. 실행 중인 컨테이너에 접속하고, 로그를 확인하며, 리소스 사용량을 모니터링할 수 있다
7. 개발 및 데이터 영속성(data persistence)을 위한 일반적인 옵션 조합을 적용할 수 있다

---

이미지(Image)와 컨테이너(Container)는 Docker에서 가장 핵심적인 두 개념입니다. 이미지는 애플리케이션이 실행에 필요한 모든 것을 담은 읽기 전용 청사진이고, 컨테이너는 그 이미지를 실제로 실행한 살아있는 인스턴스(instance)입니다. Docker Hub에서 사전 빌드된 이미지를 받아 실행하고, 검사하고, 정리하는 것부터 시작하여, Docker CLI를 통해 이미지와 컨테이너를 관리하는 방법을 익히는 것은 일상적인 개발 작업에 필수적입니다.

## 1. Docker 이미지

### 이미지란?

- 컨테이너를 만들기 위한 **읽기 전용 템플릿**
- 애플리케이션 + 실행 환경 포함
- 레이어 구조로 효율적 저장

### 이미지 이름 구조

```
[registry/]repository:tag

Examples:
nginx                    → nginx:latest (default)
nginx:1.25              → specific version
node:18-alpine          → Node 18, Alpine Linux based
myname/myapp:v1.0       → user image
gcr.io/project/app:tag  → Google Container Registry
```

| 구성요소 | 설명 | 예시 |
|----------|------|------|
| 레지스트리 | 이미지 저장소 | docker.io, gcr.io |
| 저장소 | 이미지 이름 | nginx, node |
| 태그 | 버전 | latest, 1.25, alpine |

---

## 2. 이미지 관리 명령어

### 이미지 검색

```bash
# Search on Docker Hub
docker search nginx

# Output example:
# NAME          DESCRIPTION                 STARS   OFFICIAL
# nginx         Official build of Nginx     18000   [OK]
# bitnami/nginx Bitnami nginx Docker Image  150
```

### 이미지 다운로드 (Pull)

```bash
# Download latest version
docker pull nginx

# Download specific version — pin versions in production to avoid surprise breakages
docker pull nginx:1.25

# Alpine variant: ~175 MB vs ~1 GB full image — smaller attack surface, faster pulls
docker pull node:18-alpine
```

### 이미지 목록 확인

```bash
# List local images
docker images

# Output example:
# REPOSITORY   TAG       IMAGE ID       CREATED        SIZE
# nginx        latest    a6bd71f48f68   2 days ago     187MB
# node         18-alpine 5d5f5d5f5d5f   1 week ago     175MB
```

### 이미지 삭제

```bash
# Delete image
docker rmi nginx

# Delete by image ID
docker rmi a6bd71f48f68

# Force delete (image in use)
docker rmi -f nginx

# Delete all unused images — reclaims disk space from dangling (untagged) layers
docker image prune

# Delete all images (caution!)
docker rmi $(docker images -q)
```

### 이미지 상세 정보

```bash
# Image detailed information
docker inspect nginx

# Image history (check layers)
docker history nginx
```

---

## 3. 컨테이너 실행

### 기본 실행

```bash
# Basic run
docker run nginx

# -d: Detached mode — container runs in background, freeing the terminal
docker run -d nginx

# --name: Assign a human-readable name for easier management (logs, stop, exec)
docker run -d --name my-nginx nginx

# --rm: Auto-remove on exit — prevents accumulation of stopped containers
docker run --rm nginx
```

### 포트 매핑 (-p)

```bash
# -p host:container — forwards traffic from host port to the container's internal port
docker run -d -p 8080:80 nginx

# Multiple port mappings — e.g., HTTP and HTTPS on separate host ports
docker run -d -p 8080:80 -p 8443:443 nginx

# -P: Map all EXPOSEd ports to random high host ports (useful for quick testing)
docker run -d -P nginx
```

```
┌─────────────────────────────────────────────────────┐
│  Host (my computer)                                  │
│                                                     │
│  localhost:8080 ──────────────┐                     │
│                               │                     │
│  ┌────────────────────────────▼────────────────┐   │
│  │           Container (nginx)                  │   │
│  │                                             │   │
│  │           :80 (nginx default port)          │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 환경 변수 (-e)

```bash
# -e passes config at runtime — keeps images generic and reusable across environments
docker run -d -e MYSQL_ROOT_PASSWORD=secret mysql

# Multiple environment variables
docker run -d \
  -e MYSQL_ROOT_PASSWORD=secret \
  -e MYSQL_DATABASE=mydb \
  mysql
```

### 볼륨 마운트 (-v)

```bash
# Bind mount — syncs host files into the container (useful for development)
docker run -d -v /host/path:/container/path nginx

# Mount current directory
docker run -d -v $(pwd):/app node

# :ro = read-only — container can read but not modify host files (security best practice)
docker run -d -v /host/path:/container/path:ro nginx

# Named volume — Docker manages the storage; data survives container removal
docker run -d -v mydata:/var/lib/mysql mysql
```

### 인터랙티브 모드 (-it)

```bash
# Access container shell
docker run -it ubuntu bash

# Inside container:
# root@container:/# ls
# root@container:/# exit
```

---

## 4. 컨테이너 관리

### 컨테이너 목록

```bash
# Running containers
docker ps

# All containers (including stopped)
docker ps -a

# Container IDs only
docker ps -q

# Output example:
# CONTAINER ID   IMAGE   COMMAND                  STATUS          PORTS                  NAMES
# abc123def456   nginx   "/docker-entrypoint.…"   Up 2 hours      0.0.0.0:8080->80/tcp   my-nginx
```

### 컨테이너 시작/중지/재시작

```bash
# Stop
docker stop my-nginx

# Start (stopped container)
docker start my-nginx

# Restart
docker restart my-nginx

# Force kill
docker kill my-nginx
```

### 컨테이너 삭제

```bash
# Delete container (stopped only)
docker rm my-nginx

# Force delete (even if running)
docker rm -f my-nginx

# Delete all stopped containers
docker container prune

# Delete all containers (caution!)
docker rm -f $(docker ps -aq)
```

### 컨테이너 로그

```bash
# View logs
docker logs my-nginx

# Real-time logs (-f: follow)
docker logs -f my-nginx

# Last 100 lines
docker logs --tail 100 my-nginx

# Include timestamps
docker logs -t my-nginx
```

### 실행 중인 컨테이너 접속

```bash
# Access container shell
docker exec -it my-nginx bash

# Execute specific command
docker exec my-nginx cat /etc/nginx/nginx.conf

# Access with root privileges
docker exec -it -u root my-nginx bash
```

### 컨테이너 정보

```bash
# Detailed information
docker inspect my-nginx

# Resource usage
docker stats

# Real-time resource monitoring
docker stats my-nginx
```

---

## 5. 실습 예제

### 예제 1: Nginx 웹서버

```bash
# 1. Run Nginx container
docker run -d --name web -p 8080:80 nginx

# 2. Check in browser
# http://localhost:8080

# 3. Check logs
docker logs web

# 4. Access container
docker exec -it web bash

# 5. Check Nginx configuration
cat /etc/nginx/nginx.conf

# 6. Cleanup
exit
docker stop web
docker rm web
```

### 예제 2: 커스텀 HTML 서빙

```bash
# 1. Create HTML file
mkdir -p ~/docker-test
echo "<h1>Hello Docker!</h1>" > ~/docker-test/index.html

# 2. Run with volume mount
docker run -d \
  --name my-web \
  -p 8080:80 \
  -v ~/docker-test:/usr/share/nginx/html:ro \
  nginx
# :ro — container serves files read-only; edits happen on the host only

# 3. Check in browser
# http://localhost:8080

# 4. Edit HTML (reflected in real-time)
echo "<h1>Updated!</h1>" > ~/docker-test/index.html

# 5. Cleanup
docker rm -f my-web
```

### 예제 3: MySQL 데이터베이스

```bash
# 1. Run MySQL container
docker run -d \
  --name mydb \
  -e MYSQL_ROOT_PASSWORD=secret \
  -e MYSQL_DATABASE=testdb \
  -p 3306:3306 \
  mysql:8
# No named volume here — data is lost when the container is removed (fine for quick tests)

# 2. Check startup with logs — MySQL takes a few seconds to initialize; watch for "ready for connections"
docker logs -f mydb

# 3. Connect to MySQL client
docker exec -it mydb mysql -uroot -psecret

# 4. Inside MySQL:
# mysql> SHOW DATABASES;
# mysql> USE testdb;
# mysql> CREATE TABLE users (id INT, name VARCHAR(50));
# mysql> exit

# 5. Cleanup
docker rm -f mydb
```

### 예제 4: Node.js 애플리케이션

```bash
# 1. Create project directory
mkdir -p ~/node-docker
cd ~/node-docker

# 2. Create package.json
cat > package.json << 'EOF'
{
  "name": "docker-test",
  "version": "1.0.0",
  "main": "app.js",
  "scripts": {
    "start": "node app.js"
  }
}
EOF

# 3. Create app.js
cat > app.js << 'EOF'
const http = require('http');
const server = http.createServer((req, res) => {
  res.writeHead(200, {'Content-Type': 'text/plain'});
  res.end('Hello from Node.js in Docker!\n');
});
server.listen(3000, () => {
  console.log('Server running on port 3000');
});
EOF

# 4. Run container
docker run -d \
  --name node-app \
  -p 3000:3000 \
  -v $(pwd):/app \
  -w /app \
  node:18-alpine \
  node app.js
# -w /app: sets the working directory inside the container so 'node app.js' resolves correctly

# 5. Test
curl http://localhost:3000

# 6. Cleanup
docker rm -f node-app
```

---

## 6. 유용한 옵션 조합

### 개발 환경

```bash
docker run -d \
  --name dev-server \
  -p 3000:3000 \
  -v $(pwd):/app \
  -w /app \
  --restart unless-stopped \
  node:18-alpine \
  npm run dev
# --restart unless-stopped: auto-restart on crash, but respect manual docker stop
# -v $(pwd):/app: bind mount enables live-reload — edit on host, see changes instantly
```

### 데이터 영속성

```bash
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=secret \
  -v pgdata:/var/lib/postgresql/data \
  -p 5432:5432 \
  postgres:15
# Named volume 'pgdata' — data survives container removal and can be backed up independently
```

---

## 명령어 요약

### 이미지 명령어

| 명령어 | 설명 |
|--------|------|
| `docker pull 이미지` | 이미지 다운로드 |
| `docker images` | 이미지 목록 |
| `docker rmi 이미지` | 이미지 삭제 |
| `docker image prune` | 미사용 이미지 삭제 |

### 컨테이너 명령어

| 명령어 | 설명 |
|--------|------|
| `docker run` | 컨테이너 생성 및 실행 |
| `docker ps` | 실행 중인 컨테이너 |
| `docker ps -a` | 모든 컨테이너 |
| `docker stop` | 컨테이너 중지 |
| `docker start` | 컨테이너 시작 |
| `docker rm` | 컨테이너 삭제 |
| `docker logs` | 로그 확인 |
| `docker exec -it` | 컨테이너 접속 |

### 주요 옵션

| 옵션 | 설명 |
|------|------|
| `-d` | 백그라운드 실행 |
| `-p 호스트:컨테이너` | 포트 매핑 |
| `-v 호스트:컨테이너` | 볼륨 마운트 |
| `-e KEY=VALUE` | 환경 변수 |
| `--name` | 컨테이너 이름 |
| `--rm` | 종료 시 자동 삭제 |
| `-it` | 인터랙티브 모드 |

---

## 연습 문제

### 연습 1: 이미지 탐색

`python:3.11-slim` 이미지를 받아 구조를 살펴봅니다.

1. 이미지 받기: `docker pull python:3.11-slim`
2. 로컬 이미지를 모두 나열하고 `python:3.11-slim`의 크기를 확인합니다
3. `docker history python:3.11-slim`을 실행하여 레이어(layer) 수를 세어봅니다
4. `docker inspect python:3.11-slim`을 실행하여 노출된 포트(port)와 기본 명령어(default command)를 찾습니다
5. `python:3.11-slim`과 `python:3.11`(전체 이미지)의 크기를 비교합니다. 어느 쪽이 더 크고 얼마나 차이가 나나요?

### 연습 2: 컨테이너(Container) 생명주기 관리

Nginx 컨테이너를 사용하여 전체 생명주기(lifecycle)를 실습합니다.

1. `lifecycle-test`라는 이름으로 포트(port) 9090에서 Nginx 컨테이너를 백그라운드(detached) 모드로 실행합니다: `docker run -d --name lifecycle-test -p 9090:80 nginx`
2. `docker ps`로 컨테이너가 실행 중인지 확인합니다
3. 컨테이너를 중지한 후 `docker ps -a`로 중지된 상태를 확인합니다
4. 컨테이너를 다시 시작하고 실행 중인지 확인합니다
5. `docker logs --tail 20 lifecycle-test`로 마지막 20줄의 로그를 봅니다
6. 컨테이너에 접속하여 Nginx 버전을 확인합니다: `docker exec -it lifecycle-test nginx -v`
7. `docker rm -f lifecycle-test`로 컨테이너를 강제 삭제합니다

### 연습 3: 볼륨(Volume) 마운트와 환경 변수(Environment Variable)

영속적 데이터와 커스텀 설정으로 PostgreSQL 컨테이너를 실행합니다.

1. 네임드 볼륨(named volume) 생성: `docker volume create pgdata`
2. PostgreSQL 컨테이너 실행:
   ```bash
   docker run -d \
     --name my-postgres \
     -e POSTGRES_USER=devuser \
     -e POSTGRES_PASSWORD=devpass \
     -e POSTGRES_DB=devdb \
     -v pgdata:/var/lib/postgresql/data \
     -p 5432:5432 \
     postgres:15-alpine
   ```
3. `docker logs my-postgres`로 컨테이너 상태를 확인합니다
4. 컨테이너 내부에서 PostgreSQL에 접속합니다: `docker exec -it my-postgres psql -U devuser -d devdb`
5. psql 내에서 `\l`로 데이터베이스 목록을 확인한 후 `\q`로 종료합니다
6. 컨테이너를 중지하고 삭제한 뒤, 동일한 `pgdata` 볼륨을 사용하는 새 컨테이너를 시작하여 `devdb` 데이터베이스가 유지되는지 확인합니다

### 연습 4: 리소스 모니터링과 정리

Docker 리소스 모니터링과 정리를 실습합니다.

1. 컨테이너 두 개 시작: `docker run -d --name web1 nginx` 및 `docker run -d --name web2 nginx`
2. `docker stats --no-stream`을 사용하여 두 컨테이너의 현재 리소스 사용량을 봅니다
3. `docker inspect web1`을 사용하여 Docker 네트워크 내 IP 주소를 찾습니다
4. 두 컨테이너를 삭제하지 않고 중지합니다
5. `docker ps -a`를 실행하여 두 컨테이너가 중지 상태인지 확인합니다
6. `docker container prune`으로 중지된 컨테이너를 모두 정리하고 확인합니다
7. `docker rmi nginx`로 Nginx 이미지 삭제를 시도하고, 중지된 컨테이너에서 이미지를 참조하는 경우 어떤 일이 발생하는지 관찰합니다. 오류를 해결하세요.

### 연습 5: 멀티 컨테이너(Multi-Container) 시나리오

Redis 캐시에 연결하는 Node.js 애플리케이션 컨테이너를 실행합니다.

1. Docker 네트워크(network) 생성: `docker network create app-net`
2. 커스텀 네트워크에서 Redis 실행:
   ```bash
   docker run -d --name redis-cache --network app-net redis:7-alpine
   ```
3. 동일한 네트워크에서 소스 코드를 마운트하여 Node.js 컨테이너 실행:
   ```bash
   docker run -it --rm \
     --name node-app \
     --network app-net \
     -v $(pwd):/app \
     -w /app \
     node:18-alpine \
     sh
   ```
4. 컨테이너 내부에서 DNS 해석(DNS resolution)을 확인합니다: `ping -c 2 redis-cache`
5. Redis 클라이언트(client)를 설치하고 연결을 테스트합니다: `npm install redis && node -e "const r=require('redis').createClient({url:'redis://redis-cache:6379'});r.connect().then(()=>{console.log('Connected!');r.quit()})"`
6. 정리: `redis-cache`를 중지하고 `app-net` 네트워크를 삭제합니다

---

## 다음 단계

[Dockerfile](./03_Dockerfile.md)에서 나만의 Docker 이미지를 만들어봅시다!
