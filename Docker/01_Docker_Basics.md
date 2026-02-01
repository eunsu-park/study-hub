# Docker 기초

## 1. Docker란?

Docker는 **컨테이너 기반 가상화 플랫폼**입니다. 애플리케이션과 그 실행 환경을 패키징하여 어디서든 동일하게 실행할 수 있게 해줍니다.

### 왜 Docker를 사용할까요?

**문제 상황:**
```
개발자 A: "내 컴퓨터에서는 되는데요?"
개발자 B: "저는 Node 18인데 서버는 Node 16이네요..."
운영팀: "라이브러리 버전이 달라서 에러가 나요"
```

**Docker 해결책:**
```
모든 환경을 컨테이너로 패키징 → 어디서든 동일하게 실행
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
│            가상머신 (VM)              컨테이너               │
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
│  │      Hardware        │     ✓ OS 공유 → 가볍고 빠름       │
│  └──────────────────────┘     ✓ 초 단위 시작               │
│  ✗ 각 VM마다 OS 필요          ✓ 적은 리소스 사용            │
│  ✗ 분 단위 시작                                            │
│  ✗ 많은 리소스 사용                                         │
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
│   Application       │  ← 내 애플리케이션
├─────────────────────┤
│   Node.js 18        │  ← 런타임
├─────────────────────┤
│   Ubuntu 22.04      │  ← 기본 OS
└─────────────────────┘
       이미지 레이어
```

### 컨테이너 (Container)

- 이미지를 실행한 **인스턴스**
- 읽기/쓰기 가능
- 격리된 환경에서 실행

```
이미지 ────▶ 컨테이너
(설계도)     (실제 건물)

하나의 이미지 → 여러 컨테이너 생성 가능
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
# 1. 이전 버전 제거
sudo apt remove docker docker-engine docker.io containerd runc

# 2. 필요한 패키지 설치
sudo apt update
sudo apt install ca-certificates curl gnupg lsb-release

# 3. Docker GPG 키 추가
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# 4. Docker 저장소 추가
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 5. Docker 설치
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 6. 사용자를 docker 그룹에 추가 (sudo 없이 사용)
sudo usermod -aG docker $USER
# 로그아웃 후 다시 로그인
```

---

## 5. 설치 확인

```bash
# Docker 버전 확인
docker --version
# 출력 예: Docker version 24.0.7, build afdd53b

# Docker 상세 정보
docker info

# 테스트 컨테이너 실행
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
│  │  (CLI)      │    │  (서버)     │    │ (이미지저장소)│         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                            │                  │                 │
│                            │   이미지 다운로드  │                 │
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
# hello-world 이미지 실행
docker run hello-world

# 실행 중인 컨테이너 확인
docker ps

# 모든 컨테이너 확인 (종료된 것 포함)
docker ps -a
```

### 예제 2: Nginx 웹서버 실행

```bash
# Nginx 컨테이너 실행 (백그라운드)
docker run -d -p 8080:80 nginx

# 브라우저에서 http://localhost:8080 접속

# 실행 중인 컨테이너 확인
docker ps

# 컨테이너 중지
docker stop <컨테이너ID>
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

[02_Images_and_Containers.md](./02_Images_and_Containers.md)에서 이미지와 컨테이너를 자세히 다뤄봅시다!
