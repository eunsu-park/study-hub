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

---

## 1. Docker 이미지

### 이미지란?

- 컨테이너를 만들기 위한 **읽기 전용 템플릿**
- 애플리케이션 + 실행 환경 포함
- 레이어 구조로 효율적 저장

#### 이론: 레이어드 스토리지와 OverlayFS

Docker 이미지는 *하나의 파일이 아닙니다*. 작은 JSON 문서들과 tar 아카이브로 이루어진 방향 비순환 그래프이며, 각 노드는 자기 자신 바이트의 암호학적 해시로 주소화됩니다. **매니페스트(manifest)**, **config**, **레이어(layer)** 세 조각과 "바이트의 SHA256이 곧 그 바이트의 이름이다"라는 규칙만 이해하면, 캐싱·중복 제거·멀티 아키텍처·콘텐츠 신뢰는 모두 그 결과로 자연스럽게 따라옵니다.

이미지는 읽기 전용 **레이어**가 쌓인 더미입니다. 각 레이어는 그 아래 레이어 대비 파일시스템 변경(추가/수정/삭제)을 담은 tarball입니다. 빌드 시 파일시스템을 바꾸는 Dockerfile 명령마다 새 레이어 하나가 만들어집니다.

런타임에는 컨테이너 엔진이 OverlayFS에게 이 레이어들을 통합 읽기 전용 뷰로 마운트하도록 요청한 뒤, 컨테이너마다 단 하나의 **쓰기 가능 레이어**를 위에 얹습니다.

```
   컨테이너 쓰기 레이어  (upperdir)         <-- 컨테이너별, 휘발성
   ┌────────────────────────────────┐
   │ Layer N  (예: CMD 메타데이터)   │
   │ Layer 3  (pip install ...)     │
   │ Layer 2  (apt-get install ...) │   <-- 공유되는 이미지 레이어 (lowerdir)
   │ Layer 1  (베이스 OS 파일)       │
   └────────────────────────────────┘
```

읽기는 위에서 아래로 훑어 첫 번째 매치를 반환합니다. 하부 레이어 파일에 대한 쓰기는 **copy-on-write**를 트리거합니다 — 파일을 쓰기 레이어로 복사한 뒤 그 사본을 수정합니다. 하부 레이어 파일에 대한 삭제는 상부 레이어에 "whiteout" 항목을 써서, 원본은 건드리지 않은 채 머지 뷰에서만 가립니다.

결과: 같은 이미지에서 만든 컨테이너 열 개는 디스크의 한 벌짜리 레이어 파일을 공유하고, 컨테이너별 쓰기 레이어만 새 바이트를 잡아먹습니다. 베이스를 공유하는 두 이미지도 레이어를 공유합니다 — `python:3.11-slim`과 `node:20-slim`이 모두 `debian:slim` 위에 빌드되어 있다면 데비안 레이어는 정확히 한 번만 다운로드됩니다.

이 공유는 스토리지 드라이버가 강제합니다. `overlay2`가 현대의 기본값이며, `aufs`, `devicemapper`, `btrfs`, `zfs`는 옛 대안들입니다. 최신 containerd는 "스토리지 드라이버" 대신 **스냅샷터(snapshotter)** 플러그인을 쓰지만, 레이어 머지 모델은 동일합니다.

#### 이론: 콘텐츠 주소화 스토리지와 SHA256 식별자

Docker가 저장하는 모든 블롭(blob) — 모든 레이어 tarball, 모든 config JSON, 모든 매니페스트 — 은 자신의 바이트에 대한 SHA256 해시로 이름이 정해집니다. `docker pull` 시 보이는 이름은 다음과 같습니다.

```
sha256:5a3df9a8b2c1...e6f0
```

이를 **콘텐츠 주소화 스토리지(Content-Addressable Storage, CAS)**라 부르며, 네 가지 직접적 결과를 가져옵니다.

1. **이름이 곧 무결성 검사다.** 바이트가 1비트만 바뀌어도 다이제스트(digest)가 완전히 달라집니다. 데몬은 수신 시 해시를 다시 계산하고, 실제 다이제스트가 요청한 다이제스트와 다르면 거부합니다. 변조 탐지를 위한 별도의 서명 단계가 필요 없습니다.
2. **중복 제거가 무료다.** 두 레이어의 바이트가 동일하면 다이제스트가 같으므로, 한 번만 저장됩니다. 출처가 다른 레포든, 다른 벤더든, 다른 빌드든 무관합니다.
3. **태그(tag)는 가변, 다이제스트는 불변.** `nginx:latest`는 어떤 다이제스트를 *가리키는 포인터*에 불과하며, 레지스트리는 내일 다른 다이제스트로 재지정할 수 있습니다. `nginx@sha256:5a3df9a8...`는 *약속*입니다 — 영원히 같은 바이트(또는 레지스트리가 404를 반환). 재현성을 위해 프로덕션 배포는 다이제스트로 핀(pin)해야 합니다.
4. **다이제스트가 곧 캐시 키다.** 로컬 데몬은 그 다이제스트를 가지고 있으면 그 레이어를 가지고 있는 것입니다. `docker pull`은 없는 다이제스트만 받아 옵니다. 아무것도 변하지 않은 재(再)풀에서는 매니페스트만 다운로드됩니다.

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

### 이론: 매니페스트, Config, 그리고 멀티 아키텍처 인덱스

`docker pull nginx:1.27`을 실행하면, 데몬은 "이미지"를 통째로 받지 않습니다. 작은 그래프를 따라 순회합니다.

1. `nginx:1.27`의 **매니페스트**를 가져옵니다. 매니페스트는 다음을 나열한 JSON 문서입니다.
   - **config 블롭**의 다이제스트.
   - **레이어 다이제스트**의 순서 있는 목록(크기와 미디어 타입 포함).
2. **config 블롭**을 가져옵니다. 이 JSON은 이미지 메타데이터를 기술합니다 — `CMD`, `ENTRYPOINT`, `ENV`, `WORKDIR`, 노출 포트, 라벨, 빌드 단계 히스토리, 그리고 `rootfs.diff_ids`(각 레이어의 비압축 다이제스트, 순서대로).
3. 로컬 데몬에 없는 **레이어 블롭**을 압축 다이제스트로 받아 옵니다.
4. 조립: 레이어들을 순서대로 스냅샷터 아래에 풀고, config가 `docker run`에게 프로그램을 실제로 어떻게 호출할지 알려 줍니다.

멀티 아키텍처 이미지의 경우, 레지스트리는 태그에 **매니페스트 리스트**(또는 OCI **이미지 인덱스**)를 서빙합니다. 인덱스는 `(os, architecture, variant)` 조합마다 한 매니페스트 항목을 가집니다. Docker 클라이언트는 호스트 플랫폼에 맞는 매니페스트를 자동 선택합니다. 그래서 `docker pull alpine:3.19`이 `linux/amd64`, `linux/arm64/v8`, `linux/arm/v7`에서 동일하게 동작합니다 — 태그가 인덱스를 가리키고, 인덱스가 아키텍처별 매니페스트를 가리키고, 그 매니페스트가 아키텍처별 레이어를 가리키기 때문입니다.

OCI image-spec이 이 모든 것을 표준화합니다 — JSON 스키마, 미디어 타입, 다이제스트 알고리즘. Docker 자체 포맷(Docker Image Manifest V2, Schema 2)은 OCI 명세와 거의 동일하며, 레지스트리는 둘을 호환되게 서빙합니다. 구체적으로 `docker images`는 로컬 매니페스트와 각 태그가 가리키는 다이제스트를 나열하고, `docker image inspect`는 config JSON을 덤프하며, `docker history`는 config의 `history` 배열을 읽어 어떤 Dockerfile 명령이 어떤 레이어를 만들었는지 재구성합니다.

### 이미지 검색

```bash
# Search on Docker Hub
docker search nginx

# Output example:
# NAME          DESCRIPTION                 STARS   OFFICIAL
# nginx         Official build of Nginx     18000   [OK]
# bitnami/nginx Bitnami nginx Docker Image  150
```

### 이론: 레이어 캐시 — `docker pull`과 `docker build`이 자주 싸게 끝나는 이유

CAS를 활용하는 두 가지 캐시가 있습니다.

**Pull 캐시.** `docker pull`은 다운로드 전에 각 레이어 다이제스트를 로컬 스토어와 대조합니다. `python:3.11-slim`을 받은 뒤 나중에 `python:3.12-slim`을 받으면 — 데비안 베이스 레이어, apt 캐시 레이어, 기타 공유 바이트가 모두 스킵됩니다. 대역폭 절약.

**Build 캐시.** `docker build`는 Dockerfile 명령을 순서대로 따라갑니다. 각 명령마다 다음으로 캐시 키를 만듭니다.

- 명령 텍스트 자체(`RUN apt-get install -y curl`은 한 키, `git`을 추가하면 다른 키).
- 부모 레이어의 다이제스트(앞쪽이 바뀌면 뒤쪽 캐시가 모두 무효).
- `COPY`/`ADD`의 경우, 복사되는 파일 내용의 해시.

캐시 키가 기존 로컬 레이어와 일치하면 명령은 스킵되고 캐시된 레이어가 재사용됩니다. 일치하지 않으면 명령이 실행되어 새 다이제스트의 새 레이어가 생성됩니다. 그래서 Dockerfile에서 "순서가 중요"합니다 — OS 패키지와 언어 의존성을 먼저 설치하고, 소스 코드는 마지막에 복사하세요. `app.py`를 편집한다고 apt-get 캐시가 무효화되면 곤란합니다.

BuildKit(현대 빌드 엔진)은 레이어 시스템 *바깥에* 빌드 사이에 살아남는 마운트 기반 캐시(`RUN --mount=type=cache,target=/root/.cache/pip`)와, 독립 스테이지의 병렬 실행으로 이를 확장합니다. `docker rmi`는 매니페스트 참조를 제거하지만, 어떤 매니페스트도 참조하지 *않는* 레이어만 디스크에서 사라집니다. `docker system prune`은 참조 없는 레이어를 가비지 컬렉트합니다 — CAS 모델은 그렇지 않으면 자체적으로 회수하지 않는 스토리지를 회수합니다.

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

### 이론: 컨테이너 라이프사이클 — CLI가 숨기는 다섯 가지 상태

`docker run` 한 번에 두 작업이 순차로 일어납니다.

1. `docker create` — 엔진이 런타임에게 쓰기 레이어 할당, 이미지 config 기반 네임스페이스/cgroup 셋업을 요청하고, `created` 상태의 컨테이너 레코드를 만듭니다. 아직 프로세스는 실행되지 않습니다.
2. `docker start` — 런타임이 엔트리포인트에 대해 `clone() + execve()`를 호출합니다. 컨테이너는 `running` 상태로 전이합니다.

`running`에서 다음으로 전이할 수 있습니다.

- `paused` — `docker pause`는 freezer cgroup으로 컨테이너 안 모든 PID의 스케줄링을 중단시킵니다. 메모리는 상주, 소켓도 열린 채. 스냅샷 용도로 유용하나 실무에서는 잘 안 씁니다.
- `restarting` — 엔진이 `--restart` 정책에 따라 재기동을 대기 중인 상태.
- `exited` — 엔트리포인트가 종료(또는 킬)되었습니다. 쓰기 레이어는 살아남습니다. `docker start <name>`은 쓰기 레이어의 현재 상태에서 재개합니다. `docker rm`이 마침내 쓰기 레이어를 삭제합니다.
- `dead` — 종단 실패 상태. 엔진이 정리(cleanup)조차 못 한 경우. 드물지만 `docker rm -f`로 강제 정리해야 합니다.

`docker run --rm`은 `start`에 종료 시 자동 `rm`을 체이닝합니다. `docker run -d`는 stdio를 붙이지 않습니다(컨테이너가 백그라운드로 돌아가지만 그 외 경로는 동일). `run = create + start`임을 알면, 이미지 빌드, 실패한 시작 디버깅(`create`와 `start`를 분리), `--rm`이 정확히 무엇을 지우는지에 대한 추론이 모두 명확해집니다. `docker ps`는 `running` 상태만 표시하고, `docker ps -a`는 `exited`, `paused` 등도 포함합니다.

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
