# Docker 기초

**다음**: [이미지와 컨테이너](./02_Images_and_Containers.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Docker가 무엇인지, 그리고 "내 컴퓨터에서는 되는데요" 문제를 어떻게 해결하는지 설명할 수 있다
2. 아키텍처와 리소스 사용 측면에서 컨테이너(Container)와 가상 머신(Virtual Machine)의 차이를 구분할 수 있다
3. Docker의 핵심 개념인 이미지(Image), 컨테이너(Container), Docker Hub를 설명할 수 있다
4. macOS, Windows, Linux에 Docker를 설치할 수 있다
5. 테스트 컨테이너를 실행하여 Docker 설치를 검증할 수 있다
6. CLI 명령어에서 실행 중인 컨테이너까지 Docker 워크플로우(workflow)를 설명할 수 있다
7. 포트 매핑(port mapping)과 일반적인 옵션을 사용하여 기본 컨테이너를 실행할 수 있다

---

Docker 이전에는 소프트웨어 배포가 악명 높을 정도로 불안정했습니다. 한 머신에서 완벽하게 동작하던 애플리케이션이 다른 머신에서는 다른 라이브러리 버전, OS 설정, 또는 누락된 의존성으로 인해 원인을 알 수 없는 오류를 일으켰습니다. Docker는 애플리케이션을 완전한 런타임 환경과 함께 가볍고 이식 가능한 컨테이너(Container)로 패키징함으로써 이러한 문제를 원천 차단합니다. Docker를 이해하는 것은 이제 개발자, DevOps 엔지니어, 그리고 현대적인 소프트웨어 배포에 관여하는 모든 사람에게 필수적인 기초 역량입니다.

> **비유 -- 선적 컨테이너:** 표준화된 선적 컨테이너(Shipping Container)가 등장하기 전에는 각 항구마다 다양한 화물 형태를 처리하기 위한 서로 다른 장비가 필요했습니다. Docker는 소프트웨어에 동일한 원리를 적용합니다. 애플리케이션을 모든 의존성과 함께 표준화된 컨테이너로 패키징하여 어떤 머신에서든 -- 노트북, 테스트 서버, 또는 프로덕션 클러스터 -- 동일하게 실행될 수 있게 합니다.

## 1. Docker란?

Docker는 **컨테이너 기반 가상화 플랫폼**입니다. 애플리케이션과 그 실행 환경을 패키징하여 어디서든 동일하게 실행할 수 있게 해줍니다.

### 왜 Docker를 사용할까요?

**문제 상황:**
```
Developer A: "It works on my computer?"
Developer B: "I have Node 18 but the server has Node 16..."
Operations team: "Different library versions cause errors"
```

**Docker 해결책:**
```
Package entire environment in a container → Runs identically everywhere
```

### Docker의 장점

| 장점 | 설명 |
|------|------|
| **일관성** | 개발/테스트/운영 환경 동일 |
| **격리** | 애플리케이션 간 독립 실행 |
| **이식성** | 어디서든 동일하게 실행 |
| **경량** | VM보다 빠르고 가벼움 |
| **버전 관리** | 이미지로 환경 버전 관리 |

---

## 2. 컨테이너 vs 가상머신 (VM)

```
┌────────────────────────────────────────────────────────────┐
│         Virtual Machine (VM)            Container           │
├────────────────────────────────────────────────────────────┤
│  ┌─────┐ ┌─────┐ ┌─────┐     ┌─────┐ ┌─────┐ ┌─────┐     │
│  │App A│ │App B│ │App C│     │App A│ │App B│ │App C│     │
│  ├─────┤ ├─────┤ ├─────┤     ├─────┴─┴─────┴─┴─────┤     │
│  │Guest│ │Guest│ │Guest│     │     Docker Engine    │     │
│  │ OS  │ │ OS  │ │ OS  │     ├──────────────────────┤     │
│  ├─────┴─┴─────┴─┴─────┤     │       Host OS        │     │
│  │     Hypervisor      │     ├──────────────────────┤     │
│  ├──────────────────────┤     │      Hardware        │     │
│  │       Host OS        │     └──────────────────────┘     │
│  ├──────────────────────┤                                  │
│  │      Hardware        │     ✓ Shares OS → Light & fast  │
│  └──────────────────────┘     ✓ Starts in seconds         │
│  ✗ Each VM needs OS          ✓ Low resource usage         │
│  ✗ Starts in minutes                                       │
│  ✗ High resource usage                                     │
└────────────────────────────────────────────────────────────┘
```

---

## 3. Docker 핵심 개념

### 이미지 (Image)

- 컨테이너를 만들기 위한 **템플릿**
- 읽기 전용
- 레이어 구조로 구성

```
┌─────────────────────┐
│   Application       │  ← My application
├─────────────────────┤
│   Node.js 18        │  ← Runtime
├─────────────────────┤
│   Ubuntu 22.04      │  ← Base OS
└─────────────────────┘
       Image layers
```

### 컨테이너 (Container)

- 이미지를 실행한 **인스턴스**
- 읽기/쓰기 가능
- 격리된 환경에서 실행

```
Image ────▶ Container
(Blueprint)  (Actual building)

One image → Can create multiple containers
```

### Docker Hub

- Docker 이미지 저장소 (GitHub 같은 역할)
- 공식 이미지 제공: nginx, node, python, mysql 등
- https://hub.docker.com

---

## 4. Docker 설치

### macOS

**Docker Desktop 설치 (권장):**
1. [Docker Desktop](https://www.docker.com/products/docker-desktop/) 다운로드
2. DMG 파일 실행
3. Applications 폴더로 드래그
4. Docker Desktop 실행

**Homebrew로 설치:**
```bash
brew install --cask docker
```

### Windows

1. [Docker Desktop](https://www.docker.com/products/docker-desktop/) 다운로드
2. 설치 프로그램 실행
3. WSL 2 백엔드 활성화 (권장)
4. 재시작 후 Docker Desktop 실행

### Linux (Ubuntu)

```bash
# 1. Remove old versions — prevents conflicts with the official Docker packages
sudo apt remove docker docker-engine docker.io containerd runc

# 2. Install required packages
sudo apt update
sudo apt install ca-certificates curl gnupg lsb-release

# 3. Add Docker GPG key — verifies package integrity; prevents tampered downloads
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# 4. Add Docker repository — uses Docker's own repo for latest stable releases
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 5. Install Docker
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 6. Add user to docker group — avoids typing sudo for every docker command
sudo usermod -aG docker $USER
# Log out and log back in
```

---

## 5. 설치 확인

```bash
# Check Docker version
docker --version
# Output example: Docker version 24.0.7, build afdd53b

# Docker detailed information
docker info

# Run test container
docker run hello-world
```

### hello-world 실행 결과

```
Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
 3. The Docker daemon created a new container from that image.
 4. The Docker daemon streamed that output to the Docker client.
...
```

---

## 6. Docker 작동 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  docker run nginx                                               │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Docker    │───▶│   Docker    │───▶│  Docker     │         │
│  │   Client    │    │   Daemon    │    │  Hub        │         │
│  │  (CLI)      │    │  (Server)   │    │ (Image repo)│         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                            │                  │                 │
│                            │   Download image │                 │
│                            │◀─────────────────┘                 │
│                            │                                    │
│                            ▼                                    │
│                     ┌─────────────┐                             │
│                     │  Container  │                             │
│                     │   (nginx)   │                             │
│                     └─────────────┘                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

1. **docker run** 명령 실행
2. Docker Client가 Docker Daemon에 요청
3. 로컬에 이미지 없으면 Docker Hub에서 다운로드
4. 이미지로 컨테이너 생성 및 실행

---

## 실습 예제

### 예제 1: 첫 번째 컨테이너 실행

```bash
# Run hello-world image
docker run hello-world

# Check running containers
docker ps

# Check all containers (including stopped)
docker ps -a
```

### 예제 2: Nginx 웹서버 실행

```bash
# -d: Detached mode — container runs in background, freeing the terminal
# -p 8080:80: Port mapping — host port 8080 → container port 80
docker run -d -p 8080:80 nginx

# Access in browser at http://localhost:8080

# Check running containers
docker ps

# Stop container — sends SIGTERM for graceful shutdown; SIGKILL after 10s timeout
docker stop <container-ID>
```

---

## 명령어 요약

| 명령어 | 설명 |
|--------|------|
| `docker --version` | 버전 확인 |
| `docker info` | Docker 상세 정보 |
| `docker run 이미지` | 컨테이너 실행 |
| `docker ps` | 실행 중인 컨테이너 목록 |
| `docker ps -a` | 모든 컨테이너 목록 |

---

## 다음 단계

[Docker 이미지와 컨테이너](./02_Images_and_Containers.md)에서 이미지와 컨테이너를 자세히 다뤄봅시다!
