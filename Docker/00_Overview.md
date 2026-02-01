# Docker & Kubernetes 학습 가이드

## 소개

이 폴더는 Docker와 Kubernetes를 학습하기 위한 자료를 담고 있습니다. 컨테이너의 기본 개념부터 오케스트레이션까지 단계별로 학습할 수 있습니다.

**대상 독자**: 개발자, DevOps 입문자

---

## 학습 로드맵

```
[Docker 기초]              [Docker 심화]           [오케스트레이션]
     │                          │                       │
     ▼                          ▼                       ▼
Docker 기초 ──────▶ Dockerfile ──────▶ Kubernetes 입문
     │                   │
     ▼                   ▼
이미지/컨테이너 ──▶ Docker Compose
                         │
                         ▼
                    실전 예제
```

---

## 선수 지식

- 리눅스 기본 명령어
- 터미널/쉘 사용 경험
- 웹 애플리케이션 기본 이해 (권장)

---

## 파일 목록

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [01_Docker_Basics.md](./01_Docker_Basics.md) | ⭐ | Docker 개념, 설치, 기본 명령어 |
| [02_Images_and_Containers.md](./02_Images_and_Containers.md) | ⭐ | 이미지 관리, 컨테이너 실행/관리 |
| [03_Dockerfile.md](./03_Dockerfile.md) | ⭐⭐ | Dockerfile 작성, 이미지 빌드 |
| [04_Docker_Compose.md](./04_Docker_Compose.md) | ⭐⭐ | 다중 컨테이너, docker-compose.yml |
| [05_Practical_Examples.md](./05_Practical_Examples.md) | ⭐⭐⭐ | 웹앱 컨테이너화, DB 연동 |
| [06_Kubernetes_Intro.md](./06_Kubernetes_Intro.md) | ⭐⭐⭐ | K8s 개념, Pod, Deployment, Service |
| [07_Kubernetes_Security.md](./07_Kubernetes_Security.md) | ⭐⭐⭐⭐ | RBAC, NetworkPolicy, Secrets, PodSecurity |
| [08_Kubernetes_Advanced.md](./08_Kubernetes_Advanced.md) | ⭐⭐⭐⭐ | Ingress, StatefulSet, DaemonSet, PV/PVC |
| [09_Helm_Package_Management.md](./09_Helm_Package_Management.md) | ⭐⭐⭐ | Helm 차트, values.yaml, 릴리스 관리 |
| [10_CI_CD_Pipelines.md](./10_CI_CD_Pipelines.md) | ⭐⭐⭐⭐ | GitHub Actions, 이미지 빌드, K8s 배포 |

---

## 추천 학습 순서

### 1단계: Docker 기초
1. Docker 기초 → 이미지와 컨테이너

### 2단계: Docker 활용
2. Dockerfile → Docker Compose → 실전 예제

### 3단계: 오케스트레이션
3. Kubernetes 입문 → Kubernetes 보안 → Kubernetes 심화

### 4단계: 배포 자동화
4. Helm 패키지관리 → CI/CD 파이프라인

---

## 실습 환경

### Docker 설치

```bash
# macOS
brew install --cask docker

# Ubuntu
sudo apt-get install docker.io

# 설치 확인
docker --version
```

### 실습 예제

```bash
# Hello World
docker run hello-world

# Nginx 웹서버
docker run -d -p 8080:80 nginx
```

---

## 관련 자료

- [Git 학습](../Git/00_Overview.md) - 코드 버전 관리
- [PostgreSQL 학습](../PostgreSQL/00_Overview.md) - 데이터베이스 (Docker와 함께 사용)
