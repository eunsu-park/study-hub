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

Podman / Buildah / Skopeo 레퍼런스 전에 [**이론과 원리**](#이론과-원리) 섹션을 읽으세요. OCI 이미지 / 런타임 / 배포 명세, Podman의 데몬리스 fork-exec 모델과 그것이 보안 자세를 어떻게 바꾸는지, 그리고 컨테이너별 모니터 프로세스인 conmon의 역할을 다룹니다.

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

## 이론과 원리

Podman은 "데몬 없는 Docker"가 아니라 같은 설계 공간의 다른 점으로 이해하는 게 가장 좋습니다. 둘 다 OCI 표준 위에 빌드되고, 둘 다 그 아래 runc(또는 crun)를 호출합니다. 흥미로운 차이 — 데몬리스 아키텍처, 기본 루트리스, Kubernetes에서 가져온 Pod 추상화, conmon 모니터 프로세스, 네이티브 systemd 통합 — 은 모두 한 설계 결정의 결과 — *컨테이너를 장수하는 권한 데몬 아래서 돌리지 말라.* 이 교체를 가능하게 만드는 OCI 명세와 데몬을 대체하는 fork-exec 모델을 이해하면, Podman 생태계의 나머지(Buildah, Skopeo, Podman Compose)는 같은 원칙에서 따라 나옵니다.

### A. 세 가지 OCI 명세

Open Container Initiative는 2015년에 "컨테이너"가 무엇인지 표준화하기 위해 결성되었습니다. 표준은 문제를 세 조각으로 쪼갭니다 — 함께 사용하면 어떤 호환 도구가 만든 이미지든 다른 호환 도구가 생산/배포/실행할 수 있습니다.

**OCI 이미지 명세(image-spec).** 이미지가 디스크와 와이어에서 *무엇인지*를 정의 —

- **매니페스트** — config 블롭 다이제스트와 레이어 블롭 다이제스트의 순서 있는 목록(크기, 미디어 타입 포함)을 나열한 JSON.
- **이미지 인덱스(매니페스트 리스트)** — 멀티 플랫폼 이미지용 JSON. `(os, architecture, variant)`마다 한 매니페스트를 가리킴.
- **Config** — `Cmd`, `Entrypoint`, `Env`, `WorkingDir`, 노출 포트, 빌드 명령 히스토리, `rootfs.diff_ids`를 가진 JSON.
- **레이어** — 파일시스템 변경(추가, 수정, whiteout)의 gzip 압축 tarball.
- **미디어 타입** — `application/vnd.oci.image.manifest.v1+json` 같은 엄격한 콘텐츠 타입. HTTP와 레지스트리 계층이 조각을 올바르게 식별.

**OCI 런타임 명세(runtime-spec).** 저수준 런타임이 받아들이고 해야 할 일을 정의 —

- **OCI 번들**은 `rootfs/`(언팩된 레이어들) + `config.json`(런타임 명세)을 가진 디렉터리.
- `config.json`이 선언 — 실행할 프로세스, 만들 네임스페이스, cgroup 한도, 마운트, 유지할 capability, seccomp 프로필, 훅(pre-create, post-start, pre-stop), 사용자 네임스페이스 매핑.
- 런타임(`runc`, `crun`, `youki`, `kata-runtime`)이 `config.json`을 읽고, 모든 것을 셋업하고, 프로세스를 exec.

**OCI 배포 명세(distribution-spec).** 레지스트리 HTTP API를 정의 —

- `GET /v2/<name>/manifests/<reference>` — 태그나 다이제스트로 매니페스트 가져오기.
- `GET /v2/<name>/blobs/<digest>` — 다이제스트로 블롭 가져오기.
- `PUT /v2/<name>/blobs/uploads/...` — 블롭 푸시.
- `PUT /v2/<name>/manifests/<reference>` — 매니페스트 푸시, 이미지 업로드 완료.
- CI/CD 레슨에서 설명한 OAuth2 토큰 춤으로 인증.

이 3계층 분리가 Podman이 Docker Hub와 통신, Buildah가 Docker가 읽는 이미지 생산, Skopeo가 레지스트리 사이에 이미지 복사, Kubernetes가 그들 중 어떤 것이든 런타임으로 사용하게 합니다. 표준이 보편적 용매.

### B. 데몬리스: Fork-Exec vs 장수 데몬

Docker 아키텍처는 모든 컨테이너를 소유한 권한 데몬(`dockerd`)을 가짐. CLI는 그저 HTTP 클라이언트. 함의 —

- 데몬이 root로 실행. 데몬 침해 = 호스트 root.
- 데몬이 모든 컨테이너의 PID 1 소유. 데몬 재시작 = 모든 컨테이너 재시작(또는 live-restore로 계속 돌지만 잠시 고아).
- `docker` 그룹의 누구나 데몬 소켓을 통해 사실상 root.

Podman 아키텍처는 데몬이 없음. `podman run`은 다음을 하는 평범한 프로세스 —

1. 자신을 fork.
2. 자식이 libcontainer 등가 라이브러리로 네임스페이스, cgroup, 마운트를 직접 셋업.
3. 자식이 `runc`(또는 `crun`)을 exec.
4. `runc`가 컨테이너의 엔트리포인트를 exec.
5. 원래 `podman` 프로세스는 종료. 컨테이너는 `conmon`의 자식으로 남음(§C 참조).

중앙 서버 없음. 침해할 데몬 없음. 재시작할 데몬 없음. 각 `podman` 호출이 단명 — 셋업하고 비킴.

트레이드오프 — 상태를 조정할 중앙 컴포넌트가 없음. 상태는 사용자별 파일(`~/.local/share/containers/`)에 저장. "내 모든 컨테이너에게 말하기"는 모든 상태 파일을 나열하는 것을 의미. Podman은 소켓 활성화될 수 있고 Docker API를 에뮬레이트하는 시스템 서비스(`podman.service`)를 제공하지만, 선택 사항이고 요청당 단명.

### C. conmon: 컨테이너별 모니터

`runc`가 컨테이너의 엔트리포인트를 exec할 때 누군가는 다음을 해야 함 —

- 컨테이너의 stdout/stderr 파일 디스크립터를 잡고 영속할 수 있는 곳으로 라우팅.
- 컨테이너의 종료를 기다리고 종료 코드 기록.
- `-it`이 요청되었으면 TTY 가상 터미널을 살아 있게 유지.

Docker 데몬은 모든 컨테이너에 대해 이를 함. 데몬이 없는 Podman은 `runc` exec 전에 컨테이너당 하나의 **`conmon`**(container monitor) 프로세스를 spawn. `conmon`이 —

1. 로깅 셋업 — 컨테이너 stdout/stderr를 로그 파일로 파이프(기본 위치는 로그 드라이버에 의존).
2. 요청되었으면 TTY 프록시 셋업.
3. `runc`를 fork해 컨테이너 시작.
4. `runc`가 반환된 후(컨테이너 실행 중) `conmon`이 wait.
5. 컨테이너 종료 시 `conmon`이 종료 코드를 상태 파일에 쓰고 자기도 종료.

`conmon`은 컨테이너당 작고 무상태. 호스트에 수백 개가 있을 수 있음(실행 중 컨테이너 하나당 하나). 조정하지 않음. 각각이 자기 컨테이너만 앎.

같은 아키텍처를 CRI-O(Kubernetes 런타임)도 사용하므로 `conmon`이 성숙하고 생태계에서 공유.

### D. 기본 루트리스: 사용자 네임스페이스의 실전

Podman은 권한 없는 사용자로 실행. 전통적으로 root가 필요했던 두 문제를 풀어야 함 —

1. **여러 UID로 사용자 네임스페이스 만들기.** 평범한 권한 없는 프로세스가 사용자 네임스페이스를 만들 수 있지만, 단일 UID/GID 매핑(자기 자신)으로만. 여러 사용자(예: uid 0 root + uid 1000 app)를 가진 컨테이너를 돌리려면 매핑할 UID *범위*가 필요. Podman은 **`/etc/subuid`**와 **`/etc/subgid`** — `useradd`가 유지하는 파일로 각 사용자에게 `100000-165535` 같은 범위의 "sub-UID"를 부여 — 를 사용. Podman이 컨테이너의 UID 0..65535를 사용자의 sub-UID 범위로 매핑.
2. **root 없는 네트워킹.** veth를 브리징하고 iptables 규칙을 쓰는 것은 root 필요. 루트리스 Podman은 **slirp4netns** — 컨테이너와 같은 네임스페이스에서 도는 사용자 공간 TCP/IP 스택. 사용자의 기존 네트워크 capability로 트래픽 포워드. root 불필요, 그러나 약간의 성능 오버헤드와 몇 가지 기능 격차(기본적으로 인바운드 연결 없음, ICMP는 capability 필요).

결과 — 평범한 사용자 계정에서 `podman run -d nginx`가 호스트에서 보면 `runc`와 `conmon`이 부모인 당신의 사용자 소유 프로세스로 보이는 컨테이너를 spawn. 컨테이너가 익스플로잇되어도 공격자는 당신의 사용자 권한과 Podman이 매핑한 sub-UID 범위로 제한.

### E. Buildah와 Skopeo: Docker CLI 분해

Docker는 한 CLI에 너무 많은 것을 묶음 — 이미지 빌드, 컨테이너 실행, 레지스트리 상호작용, 이미지 검사. Podman 생태계가 분해 —

- **`podman`** — 컨테이너 실행, 컨테이너 나열, 컨테이너 검사. 런타임 측.
- **`buildah`** — 이미지 빌드. 빌드는 사실 런타임이 필요 없으므로 분리 — 그저 파일시스템 레이어를 깔고 매니페스트를 쓰면 됨. Buildah는 Dockerfile(`buildah bud`)을 쓰거나, Dockerfile이 허용하는 것보다 더 많은 통제가 필요한 경우를 위한 스크립트 주도 명령형 API(`buildah from`, `buildah copy`, `buildah commit`)를 쓸 수 있음.
- **`skopeo`** — 데몬이나 런타임을 끌어들이지 않고 레지스트리의 이미지를 다룸. 레지스트리 사이에 이미지 복사(`skopeo copy docker://src docker://dst`), 풀하지 않고 레지스트리 이미지 검사(`skopeo inspect docker://nginx:1.27`), 이미지 서명과 검증.

분해는 CI/CD와 에어갭 시나리오에서 보상 받음. CI는 컨테이너를 돌릴 필요 없음 — 빌드(Buildah)와 푸시(Skopeo). 에어갭 환경은 인터넷 측 미러에서 내부 레지스트리로 데몬 측 이미지를 부팅하지 않고 복사하는 데 Skopeo 사용.

세 도구 모두 같은 이미지 라이브러리(`containers/image`)와 스토리지 라이브러리(`containers/storage`)를 공유 — 같은 로컬 레이어 캐시와 같은 레지스트리 자격 증명을 봄.

### F. Pod: Kubernetes 추상화를 가져오다

Podman은 이름이 문자 그대로 "Pod manager"에서 옴 — Kubernetes의 Pod 개념을 단일 호스트 컨테이너 관리에 가져옴. Podman pod는 네임스페이스(네트워크, IPC, 가끔 PID)와 라이프사이클을 공유하는 컨테이너 그룹 —

```bash
podman pod create --name webpod -p 8080:80
podman run -d --pod webpod nginx
podman run -d --pod webpod fluentd
```

두 컨테이너 모두 pod의 네트워크 네임스페이스 공유. nginx가 80 포트 게시, pod이 8080:80을 호스트에 게시, fluentd는 localhost 공유. Kubernetes Pod 의미와 정확히 일치.

더 좋은 점 — `podman generate kube`가 실행 중인 Podman pod에서 Kubernetes Pod 매니페스트를 출력, `podman play kube`가 Kubernetes 매니페스트를 받아 로컬에서 실행. Podman pod로 로컬 개발하고, K8s YAML 생성하고, 클러스터로 배포 — 임피던스 불일치 없음.

### G. Systemd 통합: 시스템 서비스로서의 컨테이너

Podman은 `podman generate systemd`(레거시)나 **Quadlet**(`~/.config/containers/systemd/`의 `*.container`, `*.pod`, `*.kube` 파일, 현대 방식)으로 systemd 유닛 파일 생성. 그 후 systemd가 컨테이너 라이프사이클 관리 — 부팅 시 자동 시작, 실패 시 재시작, 순서 있는 의존성, journal 로그 통합.

이게 Podman을 Kubernetes 없이 "프로덕션 단일 호스트 서비스"의 실행 가능한 선택지로 만드는 것. Quadlet `nginx.container` 파일이 systemd에 대해 가지는 관계는 Compose 서비스가 Compose 엔진에 대해 가지는 관계와 같음 — 단 systemd는 모든 리눅스 서버에서 이미 돌고 있음. 추가 오케스트레이터 없음, 추가 데몬 없음, 그저 systemd가 이미 하던 일을 함.

Docker의 등가물은 `docker run --restart=always`인데 덜 유연함(다른 유닛 후 시작 표현 못 함, journald 네이티브 로깅으로 폴백 못 함, `systemctl`로 제어 못 함).

### 이론에서 아래의 도구로

- **OCI image-spec / runtime-spec / distribution-spec**(§A) — Podman + Docker + Buildah + Kubernetes 상호 운용을 가능하게 하는 공용어.
- **Podman 데몬리스 모델 + `conmon`**(§B, §C) — `podman run`은 fork-exec, 중앙 데몬 없음, 컨테이너당 모니터 프로세스 하나.
- **`/etc/subuid`, `/etc/subgid`, `slirp4netns`**(§D) — 루트리스 배관.
- **`buildah bud` / `buildah from` / `buildah commit`**(§E) — 데몬리스 이미지 빌드. Dockerfile 없이도 가능.
- **`skopeo copy` / `skopeo inspect`**(§E) — 런타임 없는 레지스트리 간 이미지 작업.
- **`podman pod create`, `podman play kube`, `podman generate kube`**(§F) — Kubernetes와 공유하는 Pod 추상화.
- **Quadlet `*.container` 파일 + `systemctl --user`**(§G) — systemd 서비스로서의 컨테이너.

남은 본문은 이 도구들을 둘러봅니다. Podman이 어떤 엣지 케이스에서 Docker와 다르게 동작하는 이유가 궁금할 때, 답은 보통 이 설계 결정 중 하나에 뿌리를 둡니다 — 데몬 없음, 기본 루트리스, 계약으로서의 OCI 표준.

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
