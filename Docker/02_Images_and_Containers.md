# Docker 이미지와 컨테이너

## 1. Docker 이미지

### 이미지란?

- 컨테이너를 만들기 위한 **읽기 전용 템플릿**
- 애플리케이션 + 실행 환경 포함
- 레이어 구조로 효율적 저장

### 이미지 이름 구조

```
[레지스트리/]저장소:태그

예시:
nginx                    → nginx:latest (기본)
nginx:1.25              → 특정 버전
node:18-alpine          → Node 18, Alpine Linux 기반
myname/myapp:v1.0       → 사용자 이미지
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
# Docker Hub에서 검색
docker search nginx

# 출력 예시:
# NAME          DESCRIPTION                 STARS   OFFICIAL
# nginx         Official build of Nginx     18000   [OK]
# bitnami/nginx Bitnami nginx Docker Image  150
```

### 이미지 다운로드 (Pull)

```bash
# 최신 버전 다운로드
docker pull nginx

# 특정 버전 다운로드
docker pull nginx:1.25

# 특정 태그 다운로드
docker pull node:18-alpine
```

### 이미지 목록 확인

```bash
# 로컬 이미지 목록
docker images

# 출력 예시:
# REPOSITORY   TAG       IMAGE ID       CREATED        SIZE
# nginx        latest    a6bd71f48f68   2 days ago     187MB
# node         18-alpine 5d5f5d5f5d5f   1 week ago     175MB
```

### 이미지 삭제

```bash
# 이미지 삭제
docker rmi nginx

# 이미지 ID로 삭제
docker rmi a6bd71f48f68

# 강제 삭제 (사용 중인 이미지)
docker rmi -f nginx

# 사용하지 않는 이미지 모두 삭제
docker image prune

# 모든 이미지 삭제 (주의!)
docker rmi $(docker images -q)
```

### 이미지 상세 정보

```bash
# 이미지 상세 정보
docker inspect nginx

# 이미지 히스토리 (레이어 확인)
docker history nginx
```

---

## 3. 컨테이너 실행

### 기본 실행

```bash
# 기본 실행
docker run nginx

# 백그라운드 실행 (-d: detached)
docker run -d nginx

# 이름 지정
docker run -d --name my-nginx nginx

# 실행 후 자동 삭제 (--rm)
docker run --rm nginx
```

### 포트 매핑 (-p)

```bash
# 호스트:컨테이너 포트 매핑
docker run -d -p 8080:80 nginx

# 여러 포트 매핑
docker run -d -p 8080:80 -p 8443:443 nginx

# 랜덤 포트 매핑
docker run -d -P nginx
```

```
┌─────────────────────────────────────────────────────┐
│  호스트 (내 컴퓨터)                                   │
│                                                     │
│  localhost:8080 ──────────────┐                     │
│                               │                     │
│  ┌────────────────────────────▼────────────────┐   │
│  │           컨테이너 (nginx)                   │   │
│  │                                             │   │
│  │           :80 (nginx 기본 포트)              │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 환경 변수 (-e)

```bash
# 환경 변수 설정
docker run -d -e MYSQL_ROOT_PASSWORD=secret mysql

# 여러 환경 변수
docker run -d \
  -e MYSQL_ROOT_PASSWORD=secret \
  -e MYSQL_DATABASE=mydb \
  mysql
```

### 볼륨 마운트 (-v)

```bash
# 호스트 디렉토리 마운트
docker run -d -v /host/path:/container/path nginx

# 현재 디렉토리 마운트
docker run -d -v $(pwd):/app node

# 읽기 전용 마운트
docker run -d -v /host/path:/container/path:ro nginx

# Named Volume
docker run -d -v mydata:/var/lib/mysql mysql
```

### 인터랙티브 모드 (-it)

```bash
# 컨테이너 내부 쉘 접속
docker run -it ubuntu bash

# 컨테이너 내부에서:
# root@container:/# ls
# root@container:/# exit
```

---

## 4. 컨테이너 관리

### 컨테이너 목록

```bash
# 실행 중인 컨테이너
docker ps

# 모든 컨테이너 (종료 포함)
docker ps -a

# 컨테이너 ID만
docker ps -q

# 출력 예시:
# CONTAINER ID   IMAGE   COMMAND                  STATUS          PORTS                  NAMES
# abc123def456   nginx   "/docker-entrypoint.…"   Up 2 hours      0.0.0.0:8080->80/tcp   my-nginx
```

### 컨테이너 시작/중지/재시작

```bash
# 중지
docker stop my-nginx

# 시작 (중지된 컨테이너)
docker start my-nginx

# 재시작
docker restart my-nginx

# 강제 종료
docker kill my-nginx
```

### 컨테이너 삭제

```bash
# 컨테이너 삭제 (중지된 것만)
docker rm my-nginx

# 강제 삭제 (실행 중이어도)
docker rm -f my-nginx

# 중지된 컨테이너 모두 삭제
docker container prune

# 모든 컨테이너 삭제 (주의!)
docker rm -f $(docker ps -aq)
```

### 컨테이너 로그

```bash
# 로그 확인
docker logs my-nginx

# 실시간 로그 (-f: follow)
docker logs -f my-nginx

# 최근 100줄
docker logs --tail 100 my-nginx

# 타임스탬프 포함
docker logs -t my-nginx
```

### 실행 중인 컨테이너 접속

```bash
# 컨테이너 내부 쉘 접속
docker exec -it my-nginx bash

# 특정 명령어 실행
docker exec my-nginx cat /etc/nginx/nginx.conf

# 루트 권한으로 접속
docker exec -it -u root my-nginx bash
```

### 컨테이너 정보

```bash
# 상세 정보
docker inspect my-nginx

# 리소스 사용량
docker stats

# 실시간 리소스 모니터링
docker stats my-nginx
```

---

## 5. 실습 예제

### 예제 1: Nginx 웹서버

```bash
# 1. Nginx 컨테이너 실행
docker run -d --name web -p 8080:80 nginx

# 2. 브라우저에서 확인
# http://localhost:8080

# 3. 로그 확인
docker logs web

# 4. 컨테이너 내부 접속
docker exec -it web bash

# 5. Nginx 설정 확인
cat /etc/nginx/nginx.conf

# 6. 종료
exit
docker stop web
docker rm web
```

### 예제 2: 커스텀 HTML 서빙

```bash
# 1. HTML 파일 생성
mkdir -p ~/docker-test
echo "<h1>Hello Docker!</h1>" > ~/docker-test/index.html

# 2. 볼륨 마운트로 실행
docker run -d \
  --name my-web \
  -p 8080:80 \
  -v ~/docker-test:/usr/share/nginx/html:ro \
  nginx

# 3. 브라우저에서 확인
# http://localhost:8080

# 4. HTML 수정 (실시간 반영)
echo "<h1>Updated!</h1>" > ~/docker-test/index.html

# 5. 정리
docker rm -f my-web
```

### 예제 3: MySQL 데이터베이스

```bash
# 1. MySQL 컨테이너 실행
docker run -d \
  --name mydb \
  -e MYSQL_ROOT_PASSWORD=secret \
  -e MYSQL_DATABASE=testdb \
  -p 3306:3306 \
  mysql:8

# 2. 로그로 시작 확인
docker logs -f mydb

# 3. MySQL 클라이언트 접속
docker exec -it mydb mysql -uroot -psecret

# 4. MySQL 내부에서:
# mysql> SHOW DATABASES;
# mysql> USE testdb;
# mysql> CREATE TABLE users (id INT, name VARCHAR(50));
# mysql> exit

# 5. 정리
docker rm -f mydb
```

### 예제 4: Node.js 애플리케이션

```bash
# 1. 프로젝트 디렉토리 생성
mkdir -p ~/node-docker
cd ~/node-docker

# 2. package.json 생성
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

# 3. app.js 생성
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

# 4. 컨테이너 실행
docker run -d \
  --name node-app \
  -p 3000:3000 \
  -v $(pwd):/app \
  -w /app \
  node:18-alpine \
  node app.js

# 5. 테스트
curl http://localhost:3000

# 6. 정리
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
```

### 데이터 영속성

```bash
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=secret \
  -v pgdata:/var/lib/postgresql/data \
  -p 5432:5432 \
  postgres:15
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

## 다음 단계

[03_Dockerfile.md](./03_Dockerfile.md)에서 나만의 Docker 이미지를 만들어봅시다!
