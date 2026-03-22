# 15. Podman과 OCI(Podman and OCI)

**이전**: [멀티 스테이지 빌드 패턴](./14_Multi_Stage_Build_Patterns.md) | **다음**: [컨테이너 디버깅](./16_Container_Debugging.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. OCI(Open Container Initiative) 표준과 컨테이너 생태계에서의 역할을 설명한다
2. Podman의 데몬리스(Daemonless), 루트리스(Rootless) 아키텍처가 Docker와 어떻게 다른지 설명한다
3. Podman CLI 명령어를 Docker 명령어의 대체품으로 사용한다
4. Buildah로 컨테이너 이미지를 빌드하고 Skopeo로 관리한다
5. 경량 Kubernetes 추상화로서 Podman 파드(Pod)를 생성하고 관리한다
6. Podman 컨테이너를 systemd와 통합하여 프로덕션 서비스를 관리한다
7. Docker에서 Podman으로의 마이그레이션을 계획하고 실행한다

## 목차
1. [OCI 표준](#1-oci-표준)
2. [Podman 아키텍처](#2-podman-아키텍처)
3. [Podman CLI 호환성](#3-podman-cli-호환성)
4. [Buildah를 사용한 이미지 빌드](#4-buildah를-사용한-이미지-빌드)
5. [Skopeo를 사용한 이미지 관리](#5-skopeo를-사용한-이미지-관리)
6. [Podman 파드](#6-podman-파드)
7. [Systemd 통합](#7-systemd-통합)
8. [Docker에서 Podman으로 마이그레이션](#8-docker에서-podman으로-마이그레이션)
9. [Podman Compose와 Kubernetes](#9-podman-compose와-kubernetes)
10. [연습 문제](#10-연습-문제)

**난이도**: ⭐⭐⭐

---

Docker가 컨테이너를 대중화했지만, 생태계는 단일 도구를 넘어 발전했습니다. OCI(Open Container Initiative)는 컨테이너 형식과 런타임에 대한 개방형 표준을 수립하여 Podman, Buildah, Skopeo와 같은 대안을 가능하게 했습니다. Podman의 데몬리스(Daemonless), 루트리스(Rootless) 설계는 Docker의 권한 있는 데몬 모델에 대한 근본적인 보안 우려를 해결하여, 기업 환경과 보안에 민감한 환경에서 특히 매력적입니다.

---

## 1. OCI 표준

### OCI란 무엇인가?

OCI(Open Container Initiative)는 2015년 Linux Foundation 산하에 설립되었으며, 세 가지 핵심 사양을 정의합니다:

```
┌──────────────────────────────────────────────────────────────┐
│                    OCI 사양(Specifications)                    │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. 이미지 사양 (image-spec)                                  │
│     ┌─────────────────────────────────────────────┐          │
│     │ 컨테이너 이미지 구조 정의                     │          │
│     │ • 이미지 매니페스트(Image manifest)           │          │
│     │ • 이미지 인덱스(다중 아키텍처)                │          │
│     │ • 파일시스템 레이어 (tar+gzip)                │          │
│     │ • 이미지 설정 (env, cmd 등)                   │          │
│     └─────────────────────────────────────────────┘          │
│                                                               │
│  2. 런타임 사양 (runtime-spec)                                │
│     ┌─────────────────────────────────────────────┐          │
│     │ 컨테이너 실행 방법 정의                       │          │
│     │ • 컨테이너 라이프사이클 (create/start/stop)   │          │
│     │ • 설정 형식 (config.json)                     │          │
│     │ • Linux 관련: 네임스페이스, cgroups, 권한     │          │
│     └─────────────────────────────────────────────┘          │
│                                                               │
│  3. 배포 사양 (distribution-spec)                             │
│     ┌─────────────────────────────────────────────┐          │
│     │ 컨테이너 이미지 배포 방법 정의                │          │
│     │ • Push/Pull 작업                              │          │
│     │ • 레지스트리 API (HTTP 기반)                   │          │
│     │ • 콘텐츠 검색 및 해석                         │          │
│     └─────────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────────┘
```

### OCI 호환 도구

| 도구 | 목적 | OCI 호환 |
|---|---|---|
| Docker | 빌드, 실행, 배포 | 예 |
| Podman | 빌드, 실행 | 예 |
| Buildah | 이미지 빌드 | 예 |
| Skopeo | 이미지 복사/검사 | 예 |
| containerd | 컨테이너 런타임 | 예 |
| CRI-O | Kubernetes 런타임 | 예 |
| runc | 저수준 런타임 | 참조 구현 |

### OCI가 중요한 이유

```bash
# Docker로 빌드한 이미지는 Podman에서 작동 (그 반대도 마찬가지)
docker build -t myapp .
docker save myapp -o myapp.tar

# Podman에 로드
podman load -i myapp.tar
podman run myapp

# 모든 OCI 호환 레지스트리에 푸시
podman push myapp docker.io/myuser/myapp:latest
```

---

## 2. Podman 아키텍처

### Docker vs Podman 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│  Docker 아키텍처                                              │
│                                                               │
│  User ──► docker CLI ──► Docker Daemon (dockerd) ──► containerd
│                              │ (root로 실행)          │       │
│                              │                     ┌──┴──┐   │
│                              │                     │runc │   │
│                              │                     └──┬──┘   │
│                              │                        │      │
│                              ▼                        ▼      │
│                         Container A              Container B  │
│                                                               │
│  ⚠ 단일 장애점: 데몬 충돌 시 모든 컨테이너가 중단됨            │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  Podman 아키텍처 (데몬리스)                                    │
│                                                               │
│  User ──► podman CLI ──► conmon ──► runc ──► Container        │
│              │              │                                  │
│              │    (각 컨테이너가 자체 conmon 프로세스를 가짐)    │
│              │                                                 │
│  User ──► podman CLI ──► conmon ──► runc ──► Container        │
│                                                               │
│  ✓ 데몬 없음: 컨테이너가 직접 자식 프로세스로 실행              │
│  ✓ 단일 장애점 없음                                            │
│  ✓ 기본적으로 루트리스(Rootless)                                │
└──────────────────────────────────────────────────────────────┘
```

### 주요 차이점

| 기능 | Docker | Podman |
|---|---|---|
| 데몬 | 필수 (dockerd) | 데몬 없음 |
| 루트 필요 | 예 (데몬이 root로 실행) | 아니오 (기본 루트리스) |
| 소켓 | `/var/run/docker.sock` | 사용자별 소켓 |
| 컨테이너 프로세스 부모 | dockerd | conmon (컨테이너별) |
| 파드 (Kubernetes 유사) | 네이티브 미지원 | 일급(First-class) 파드 |
| Systemd 통합 | 별도 유닛 파일 | `podman generate systemd` |
| Docker Compose | 네이티브 | podman-compose 사용 |
| 재부팅 시 컨테이너 재시작 | 데몬 자동 시작 | systemd 유닛 사용 |

### 루트리스 컨테이너(Rootless Containers)

Podman은 사용자 네임스페이스(User Namespaces)를 활용하여 루트 권한 없이 컨테이너를 실행합니다:

```bash
# 루트리스 설정 확인
podman info --format '{{.Host.Security.Rootless}}'
# true

# 사용자 네임스페이스 매핑
podman unshare cat /proc/self/uid_map
#     0    1000       1
#     1  100000   65536

# 루트리스 컨테이너는 기본적으로 1024 미만의 포트에 바인딩 불가
podman run -p 8080:80 nginx   # 작동
podman run -p 80:80 nginx     # 실패 (설정하지 않는 한)

# 루트리스에서 낮은 포트 허용
sudo sysctl net.ipv4.ip_unprivileged_port_start=80
```

---

## 3. Podman CLI 호환성

### 드롭인 대체(Drop-In Replacement)

Podman은 Docker의 CLI 호환 대체품으로 설계되었습니다:

```bash
# 이 명령어들은 Docker와 Podman에서 동일하게 작동
podman pull nginx:alpine
podman run -d --name web -p 8080:80 nginx:alpine
podman ps
podman logs web
podman exec -it web sh
podman stop web
podman rm web

# 마이그레이션을 위한 일반적인 별칭
alias docker=podman
```

### 컨테이너 관리

```bash
# 컨테이너 실행
podman run -d --name myapp \
  -p 8080:8080 \
  -v mydata:/data \
  -e DB_HOST=localhost \
  myapp:latest

# 컨테이너 목록 (실행 중 및 중지됨)
podman ps -a

# 컨테이너 리소스 통계
podman stats --no-stream

# 컨테이너 프로세스 목록
podman top myapp

# 파일 복사
podman cp myapp:/app/config.json ./config.json
podman cp ./newconfig.json myapp:/app/config.json
```

### 이미지 관리

```bash
# 이미지 빌드
podman build -t myapp:latest .

# 이미지 목록
podman images

# 태그 및 푸시
podman tag myapp:latest docker.io/myuser/myapp:latest
podman push docker.io/myuser/myapp:latest

# 이미지 히스토리
podman history myapp:latest

# 미사용 이미지 제거
podman image prune -a
```

### 주의할 차이점

```bash
# Podman에 없는 Docker 전용 기능:
# 1. Docker Swarm (대신 Kubernetes 사용)
# 2. docker-compose (podman-compose 또는 podman play kube 사용)

# Docker에 없는 Podman 전용 기능:
# 1. 파드 (podman pod create)
# 2. systemd 유닛 생성 (podman generate systemd)
# 3. Kubernetes YAML 생성 (podman generate kube)
# 4. 기본 루트리스

# 레지스트리 처리 차이
# Docker는 기본적으로 docker.io를 사용; Podman은 unqualified-search-registries 사용
# /etc/containers/registries.conf에서 설정
```

```ini
# /etc/containers/registries.conf
unqualified-search-registries = ["docker.io", "quay.io", "ghcr.io"]
```

---

## 4. Buildah를 사용한 이미지 빌드

### Buildah를 사용하는 이유

Buildah는 OCI 이미지를 빌드하기 위한 전문 도구입니다. 데몬이 필요 없으며 Dockerfile 없이 이미지를 빌드할 수 있습니다.

```
┌──────────────────────────────────────────────────────────────┐
│             Buildah vs Docker Build                           │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Docker Build:                                                │
│  • 데몬 필요                                                  │
│  • Dockerfile만 사용                                          │
│  • 완전한 이미지 빌드                                         │
│                                                               │
│  Buildah:                                                     │
│  • 데몬리스                                                   │
│  • Dockerfile 또는 스크립트 빌드                              │
│  • 세밀한 레이어 제어                                         │
│  • 호스트 파일시스템을 빌드에 마운트 가능                      │
│  • 모든 컨테이너를 이미지로 커밋 가능                         │
└──────────────────────────────────────────────────────────────┘
```

### Dockerfile로 빌드

```bash
# Buildah는 표준 Dockerfile을 지원
buildah bud -t myapp:latest .

# 동일:
buildah build -t myapp:latest .

# 빌드 인수 사용
buildah build --build-arg VERSION=1.0 -t myapp:1.0 .
```

### 스크립트 빌드 (Dockerfile 없이)

```bash
#!/bin/bash
# build.sh -- Dockerfile 없이 이미지 빌드

# 베이스 이미지에서 새 컨테이너 생성
container=$(buildah from python:3.12-slim)

# 컨테이너 안에서 명령어 실행
buildah run $container pip install flask gunicorn

# 파일을 컨테이너로 복사
buildah copy $container ./app /app

# 설정
buildah config --workingdir /app $container
buildah config --port 8000 $container
buildah config --cmd '["gunicorn", "app:app", "-b", "0.0.0.0:8000"]' $container
buildah config --label maintainer="dev@example.com" $container

# 컨테이너를 이미지로 커밋
buildah commit $container myapp:latest

# 정리
buildah rm $container
```

### Buildah 마운트 (호스트 통합)

```bash
# 컨테이너의 파일시스템을 호스트에 마운트
container=$(buildah from fedora)
mountpoint=$(buildah mount $container)

# 이제 호스트 도구로 컨테이너의 파일시스템을 조작 가능
dnf install --installroot $mountpoint --releasever 39 python3 -y

# 언마운트 및 커밋
buildah unmount $container
buildah commit $container my-fedora-python
```

---

## 5. Skopeo를 사용한 이미지 관리

### 이미지 검사(Image Inspection)

```bash
# 풀 없이 원격 이미지 검사
skopeo inspect docker://docker.io/library/nginx:alpine

# 이미지 다이제스트 가져오기
skopeo inspect --format '{{.Digest}}' docker://nginx:alpine

# 레포지토리의 태그 나열
skopeo list-tags docker://docker.io/library/python

# 로컬 이미지 검사
skopeo inspect containers-storage:localhost/myapp:latest
```

### 이미지 복사(Image Copying)

```bash
# 레지스트리 간 복사 (로컬 스토리지 불필요)
skopeo copy \
  docker://docker.io/library/nginx:alpine \
  docker://myregistry.example.com/nginx:alpine

# 로컬 디렉토리로 복사 (OCI 레이아웃)
skopeo copy \
  docker://nginx:alpine \
  oci:/tmp/nginx-oci:alpine

# Docker 아카이브로 복사 (tar 파일)
skopeo copy \
  docker://nginx:alpine \
  docker-archive:/tmp/nginx.tar:nginx:alpine

# 로컬 Podman 이미지에서 레지스트리로 복사
skopeo copy \
  containers-storage:localhost/myapp:latest \
  docker://myregistry.example.com/myapp:latest
```

### 이미지 동기화(Image Synchronization)

```bash
# 이미지의 모든 태그를 로컬 디렉토리에 동기화
skopeo sync --src docker --dest dir \
  docker.io/library/python /tmp/python-mirror

# 디렉토리에서 프라이빗 레지스트리로 동기화
skopeo sync --src dir --dest docker \
  /tmp/python-mirror myregistry.example.com/mirror

# 에어갭(air-gapped) 환경에 유용
```

### 이미지 삭제

```bash
# 레지스트리에서 이미지 삭제
skopeo delete docker://myregistry.example.com/myapp:old-tag
```

---

## 6. Podman 파드

### 파드란 무엇인가?

파드(Pod)는 네트워크, PID, IPC 네임스페이스를 공유하는 컨테이너 그룹입니다 -- Kubernetes 파드와 동일한 개념입니다.

```
┌──────────────────────────────────────────────────────────────┐
│                        Podman 파드                            │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  공유 네임스페이스: network, IPC, (선택) PID           │   │
│  │  공유 localhost (127.0.0.1)                            │   │
│  │                                                        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │ infra    │  │ app      │  │ sidecar  │            │   │
│  │  │ (pause)  │  │ (nginx)  │  │ (logging)│            │   │
│  │  │          │  │ :80      │  │ :9090    │            │   │
│  │  └──────────┘  └──────────┘  └──────────┘            │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
│  포트 매핑은 파드 수준에서 (infra 컨테이너를 통해)             │
│  컨테이너 간 통신은 localhost를 통해                           │
└──────────────────────────────────────────────────────────────┘
```

### 파드 생성 및 관리

```bash
# 게시된 포트가 있는 파드 생성
podman pod create --name webapp \
  -p 8080:80 \
  -p 5432:5432

# 파드에 컨테이너 추가
podman run -d --pod webapp \
  --name web \
  nginx:alpine

podman run -d --pod webapp \
  --name db \
  -e POSTGRES_PASSWORD=secret \
  postgres:16-alpine

# web 컨테이너는 localhost:5432로 postgres에 접근 가능
# 외부 접근은 host:8080 (nginx) 및 host:5432 (postgres)

# 파드 목록
podman pod ls

# 파드 상세 정보
podman pod inspect webapp

# 전체 파드 중지/시작/재시작
podman pod stop webapp
podman pod start webapp
podman pod restart webapp

# 파드와 모든 컨테이너 제거
podman pod rm -f webapp
```

### 파드 vs Docker Compose

```
┌──────────────────────────────────────────────────────────┐
│          Podman 파드 vs Docker Compose                     │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Docker Compose:                                          │
│  • 컨테이너가 브릿지 네트워크를 공유                       │
│  • DNS 이름으로 서비스 검색                                │
│  • 각 컨테이너에 고유 IP                                   │
│  • 컨테이너별 포트 매핑                                    │
│                                                           │
│  Podman 파드:                                             │
│  • 컨테이너가 localhost (127.0.0.1)를 공유                 │
│  • localhost:port로 통신                                   │
│  • 파드에 단일 IP                                          │
│  • 파드에서 포트 매핑 (infra 컨테이너를 통해)              │
│  • Kubernetes 파드 모델에 더 가까움                        │
└──────────────────────────────────────────────────────────┘
```

---

## 7. Systemd 통합

### Systemd 유닛 생성

Podman은 컨테이너와 파드에 대한 systemd 서비스 파일을 생성할 수 있습니다:

```bash
# 컨테이너에 대한 systemd 유닛 생성
podman generate systemd --new --name webapp > webapp.service

# 추가 옵션으로 생성
podman generate systemd --new --name webapp \
  --restart-policy=always \
  --time 30 \
  > webapp.service

# 전체 파드에 대해 생성
podman generate systemd --new --name mypod --files
# 생성됨: pod-mypod.service, container-web.service, container-db.service
```

### 사용자 수준 서비스 설치 (루트리스)

```bash
# systemd 사용자 디렉토리 생성
mkdir -p ~/.config/systemd/user

# 서비스 생성 및 설치
podman generate systemd --new --name webapp \
  > ~/.config/systemd/user/webapp.service

# 활성화 및 시작
systemctl --user daemon-reload
systemctl --user enable --now webapp.service

# 상태 확인
systemctl --user status webapp.service

# 링거링(Lingering) 활성화 (로그아웃 후에도 계속 실행)
loginctl enable-linger $USER
```

### 시스템 수준 서비스 설치 (루트)

```bash
# root로 생성
sudo podman generate systemd --new --name webapp \
  > /etc/systemd/system/webapp.service

# 활성화 및 시작
sudo systemctl daemon-reload
sudo systemctl enable --now webapp.service
```

### Quadlet (Podman 4.4+)

Quadlet은 Podman 컨테이너를 systemd 유닛으로 선언적으로 정의하는 방법을 제공합니다:

```ini
# ~/.config/containers/systemd/webapp.container
[Container]
Image=docker.io/library/nginx:alpine
PublishPort=8080:80
Volume=webdata.volume:/usr/share/nginx/html:ro

[Service]
Restart=always

[Install]
WantedBy=default.target
```

```ini
# ~/.config/containers/systemd/webdata.volume
[Volume]
Label=app=webapp
```

```bash
# 리로드 및 시작
systemctl --user daemon-reload
systemctl --user start webapp.service
```

---

## 8. Docker에서 Podman으로 마이그레이션

### 단계별 마이그레이션

```
┌──────────────────────────────────────────────────────────────┐
│              Docker에서 Podman으로 마이그레이션 경로            │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1단계: 평가                                                  │
│  ├─ Docker 사용 현황 파악 (이미지, 볼륨, 네트워크)             │
│  ├─ Docker 전용 기능 식별 (Swarm 등)                          │
│  └─ 기존 Dockerfile로 Podman 테스트                           │
│                                                               │
│  2단계: 공존                                                  │
│  ├─ Docker와 함께 Podman 설치                                 │
│  ├─ 테스트를 위해 alias docker=podman                         │
│  └─ 개발/테스트 워크로드를 Podman에서 실행                    │
│                                                               │
│  3단계: 마이그레이션                                          │
│  ├─ Docker 이미지 내보내기 → Podman으로 가져오기               │
│  ├─ docker-compose.yml → 파드 YAML로 변환                     │
│  ├─ Docker systemd 유닛 → Podman systemd 유닛으로 교체        │
│  └─ CI/CD 파이프라인 업데이트                                 │
│                                                               │
│  4단계: 정리                                                  │
│  ├─ Docker 데몬 제거                                          │
│  ├─ docker.sock 종속성 제거                                   │
│  └─ Podman 전용 워크플로우 문서화                             │
└──────────────────────────────────────────────────────────────┘
```

### 이미지 마이그레이션

```bash
# Docker에서 내보내기
docker save myapp:latest -o myapp.tar

# Podman으로 가져오기
podman load -i myapp.tar

# 또는 Skopeo를 사용하여 직접 복사
skopeo copy \
  docker-daemon:myapp:latest \
  containers-storage:myapp:latest
```

### 볼륨 마이그레이션

```bash
# Docker 볼륨 데이터 내보내기
docker run --rm -v mydata:/source:ro -v $(pwd):/backup \
  alpine tar czf /backup/mydata.tar.gz -C /source .

# Podman 볼륨 생성 및 복원
podman volume create mydata
podman run --rm -v mydata:/target -v $(pwd):/backup:ro \
  alpine sh -c "cd /target && tar xzf /backup/mydata.tar.gz"
```

### Docker Compose 마이그레이션

```bash
# 옵션 1: podman-compose (Python, 드롭인 대체)
pip install podman-compose
podman-compose up -d

# 옵션 2: Podman의 내장 compose 지원 사용 (Podman 3.0+)
podman compose up -d

# 옵션 3: Kubernetes YAML로 변환
podman generate kube mypod > pod.yaml
podman play kube pod.yaml
```

---

## 9. Podman Compose와 Kubernetes

### Podman Generate Kube

실행 중인 파드/컨테이너를 Kubernetes 호환 YAML로 변환:

```bash
# 파드에서 Kubernetes YAML 생성
podman generate kube webapp > webapp-pod.yaml

# 서비스 정의와 함께 생성
podman generate kube webapp -s > webapp-with-service.yaml
```

```yaml
# 생성된 webapp-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    app: webapp
  name: webapp
spec:
  containers:
    - name: web
      image: docker.io/library/nginx:alpine
      ports:
        - containerPort: 80
          hostPort: 8080
    - name: db
      image: docker.io/library/postgres:16-alpine
      env:
        - name: POSTGRES_PASSWORD
          value: secret
```

### Podman Play Kube

Kubernetes YAML 파일을 Podman으로 직접 배포:

```bash
# Kubernetes YAML 배포
podman play kube webapp-pod.yaml

# 볼륨 생성과 함께
podman play kube --build webapp-pod.yaml

# 해체
podman play kube --down webapp-pod.yaml

# 업데이트 (삭제 후 재생성)
podman play kube --replace webapp-pod.yaml
```

이를 통해 Podman으로 로컬에서 개발하고 동일한 YAML 정의로 Kubernetes에 배포하는 워크플로우가 가능합니다.

---

## 10. 연습 문제

### 연습 1: Podman 기초 (초급)

Podman으로 nginx 컨테이너를 실행하고, 작동을 확인한 후, 정리하세요.

```bash
# 1. Podman으로 nginx:alpine 이미지 풀
# 2. 포트 8080을 80에 매핑하여 실행
# 3. curl로 확인
# 4. 컨테이너 중지 및 제거
```

<details>
<summary>풀이</summary>

```bash
podman pull nginx:alpine
podman run -d --name web -p 8080:80 nginx:alpine
curl http://localhost:8080
podman stop web
podman rm web
```

</details>

### 연습 2: 파드 생성 (중급)

Flask 앱과 Redis가 있는 Podman 파드를 생성하고, Flask 앱이 localhost를 통해 Redis에 연결하도록 하세요.

<details>
<summary>풀이</summary>

```bash
# 파드 생성
podman pod create --name flask-redis -p 5000:5000

# Redis 추가
podman run -d --pod flask-redis --name redis redis:alpine

# 간단한 Flask 앱 생성
mkdir /tmp/flask-app
cat > /tmp/flask-app/app.py << 'PYEOF'
from flask import Flask
import redis

app = Flask(__name__)
r = redis.Redis(host='localhost', port=6379)

@app.route('/')
def hello():
    count = r.incr('hits')
    return f'Hello! This page has been visited {count} times.\n'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
PYEOF

cat > /tmp/flask-app/requirements.txt << 'EOF'
flask
redis
EOF

cat > /tmp/flask-app/Dockerfile << 'EOF'
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
CMD ["python", "app.py"]
EOF

# 빌드 및 실행
podman build -t flask-app /tmp/flask-app
podman run -d --pod flask-redis --name app flask-app

# 테스트
curl http://localhost:5000

# 정리
podman pod rm -f flask-redis
```

</details>

### 연습 3: Buildah 스크립트 빌드 (중급)

Buildah를 사용하여 (Dockerfile 없이) nginx로 정적 HTML 페이지를 서빙하는 이미지를 만드세요.

<details>
<summary>풀이</summary>

```bash
#!/bin/bash
# HTML 콘텐츠 생성
mkdir -p /tmp/mysite
echo "<h1>Built with Buildah!</h1>" > /tmp/mysite/index.html

# Buildah 스크립트 빌드
ctr=$(buildah from nginx:alpine)
buildah copy $ctr /tmp/mysite/index.html /usr/share/nginx/html/index.html
buildah config --port 80 $ctr
buildah config --label maintainer="student@example.com" $ctr
buildah commit $ctr mysite:latest
buildah rm $ctr

# Podman으로 실행
podman run -d --name mysite -p 8080:80 mysite:latest
curl http://localhost:8080
podman rm -f mysite
```

</details>

### 연습 4: Skopeo와 마이그레이션 (고급)

Skopeo를 사용하여 원격 이미지를 검사하고, 로컬 OCI 디렉토리로 복사한 다음 Podman에 로드하세요.

<details>
<summary>풀이</summary>

```bash
# 원격 이미지 검사
skopeo inspect docker://docker.io/library/alpine:3.19

# 로컬 OCI 디렉토리로 복사
skopeo copy docker://alpine:3.19 oci:/tmp/alpine-oci:3.19

# OCI 레이아웃 검사
ls -la /tmp/alpine-oci/

# OCI 디렉토리에서 Podman 스토리지로 복사
skopeo copy oci:/tmp/alpine-oci:3.19 containers-storage:alpine-local:3.19

# 확인
podman images alpine-local
podman run --rm alpine-local:3.19 cat /etc/os-release

# 정리
podman rmi alpine-local:3.19
rm -rf /tmp/alpine-oci
```

</details>

### 연습 5: Systemd 서비스 (고급)

웹 애플리케이션을 위한 Podman 컨테이너를 만들고, 부팅 시 시작되고 실패 시 재시작되는 루트리스 systemd 서비스로 설정하세요.

<details>
<summary>풀이</summary>

```bash
# 컨테이너 실행
podman run -d --name webapp -p 8080:80 nginx:alpine

# systemd 유닛 생성
mkdir -p ~/.config/systemd/user
podman generate systemd --new --name webapp \
  --restart-policy=always \
  > ~/.config/systemd/user/webapp.service

# 수동으로 생성된 컨테이너 중지
podman stop webapp
podman rm webapp

# systemd 서비스 활성화
systemctl --user daemon-reload
systemctl --user enable --now webapp.service

# 확인
systemctl --user status webapp.service
curl http://localhost:8080

# 부팅 시작을 위해 링거링 활성화
loginctl enable-linger $USER

# 정리
systemctl --user disable --now webapp.service
rm ~/.config/systemd/user/webapp.service
systemctl --user daemon-reload
```

</details>

---

**이전**: [멀티 스테이지 빌드 패턴](./14_Multi_Stage_Build_Patterns.md) | **다음**: [컨테이너 디버깅](./16_Container_Debugging.md)
