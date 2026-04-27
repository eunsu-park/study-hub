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

---

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

### 이론: 두 가지 격리 모델

가상 머신(Virtual Machine, VM)은 하드웨어를 에뮬레이트합니다. 하이퍼바이저(KVM, Xen, VMware ESXi, Hyper-V)가 가상 CPU·가상 메모리·가상 디바이스를 *게스트 커널*에 제공하고, 게스트 커널이 자체 OS를 부팅해 자체 프로세스를 실행합니다. 격리는 강하지만, VM마다 완전한 커널과 사용자 공간을 짊어지므로 RAM 수백 MB와 부팅 시간 수십 초의 비용이 듭니다 — 애플리케이션이 시작되기도 전에.

컨테이너는 하드웨어를 에뮬레이트하지 않습니다. *호스트 커널* 위에서 동작하며, 평범한 프로세스가 자기 혼자 머신을 점유한 것처럼 믿게 만드는 커널 기능을 사용합니다. 게스트 OS도, 가상 하드웨어도, 두 번째 커널도 없습니다. 시작 비용은 `fork() + exec()`에 약간의 네임스페이스 셋업 시스템 콜이 더해진 정도, 즉 밀리초 단위입니다. 메모리 오버헤드는 프로세스 자체의 RSS에 수 MB 부기(bookkeeping) 정도에 불과합니다.

대가는 신뢰 경계(trust boundary)입니다. VM 탈출(escape)은 하이퍼바이저를 깨야 하지만, 컨테이너 탈출은 호스트 커널을 깨면 됩니다. 그래서 컨테이너는 같은 신뢰 도메인의 워크로드(한 팀, 한 애플리케이션 스택)를 같이 두는 것이 전통적이며, 멀티 테넌트 클라우드는 신뢰 경계가 밀리초 시작 시간보다 더 중요할 때 컨테이너를 다시 경량 VM(Firecracker, Kata Containers)으로 감쌉니다.

컨테이너는 구체적으로는 호스트 커널 위에서 동작하는 프로세스(또는 작은 프로세스 그룹)에 세 가지 리눅스 기능을 얹은 것입니다 — **네임스페이스(namespaces)**(프로세스가 *볼 수 있는 것* 격리), **cgroups**(프로세스가 *소비할 수 있는 것* 제한), 그리고 **유니온 파일시스템(union filesystem)**(컨테이너마다 읽기 전용 이미지 레이어와 쓰기 가능 최상단 레이어를 결합한 독립 루트 파일시스템 제공). 아래 다이어그램이 아키텍처 차이를 보여주며, 다음 섹션에서 각 메커니즘을 풀어냅니다.

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

#### 이론: 유니온 파일시스템 — 중복 없는 레이어 스토리지

컨테이너는 루트 파일시스템(`/bin`, `/etc`, `/lib`, ...)이 필요하지만, 컨테이너마다 트리 전체를 복사하면 경량성이 무너집니다. 유니온 파일시스템은 디렉터리들을 *쌓아서(stack)* 이를 해결합니다. 아래쪽에 여러 읽기 전용 레이어, 그 위에 쓰기 가능한 레이어 하나, 그리고 이를 합친 단일 뷰가 컨테이너에게 제공됩니다.

현대 Docker는 **OverlayFS**(`overlay2` 스토리지 드라이버)를 사용합니다. 디렉터리 입력 세 개와 출력 하나를 받습니다.

- `lowerdir` — 한 개 이상의 읽기 전용 레이어(이미지 레이어들이 쌓여 있음).
- `upperdir` — 새 파일과 수정된 파일이 들어가는 단일 쓰기 레이어.
- `workdir` — 커널이 원자적 연산용으로 쓰는 내부 스크래치 공간.
- `merged` — 컨테이너가 `/`로 보는 통합 뷰.

컨테이너가 파일을 읽으면 커널이 레이어를 위에서 아래로 훑어 첫 번째 매치를 반환합니다. 컨테이너가 *하부 레이어에만 존재하는* 파일을 *수정*하면 OverlayFS는 **copy-on-write**를 수행합니다 — 파일을 `upperdir`로 복사한 뒤 그 사본을 수정합니다. 컨테이너가 하부 레이어의 파일을 *삭제*하면, OverlayFS는 `upperdir`에 특수한 "whiteout" 항목을 만들어 머지된 뷰에서만 파일을 가립니다. 하부에서는 실제로 아무것도 지워지지 않습니다.

결과적으로, 같은 이미지에서 만든 컨테이너 열 개는 디스크의 한 벌짜리 레이어 파일을 공유합니다. 새로 차지하는 공간은 컨테이너별 `upperdir`뿐이며, 워크로드가 많이 쓰지 않는 한 보통 수 MB에 그칩니다.

### 컨테이너 (Container)

- 이미지를 실행한 **인스턴스**
- 읽기/쓰기 가능
- 격리된 환경에서 실행

```
Image ────▶ Container
(Blueprint)  (Actual building)

One image → Can create multiple containers
```

#### 이론: 리눅스 네임스페이스 — 프로세스가 볼 수 있는 것을 격리

네임스페이스는 한 종류의 시스템 자원에 대한 커널 수준의 "뷰(view)"입니다. 같은 종류의 서로 다른 네임스페이스에 있는 두 프로세스는, 같은 커널을 공유하면서도 서로 다른 자원을 봅니다. 컨테이너와 관련된 일곱 가지 네임스페이스 종류는 다음과 같습니다.

| 네임스페이스 | 격리하는 대상 | 시스템 콜 플래그 |
|--------------|---------------|------------------|
| `PID` | 프로세스 ID(컨테이너 안의 PID 1, 호스트 프로세스는 보이지 않음) | `CLONE_NEWPID` |
| `NET` | 네트워크 인터페이스, 라우팅 테이블, iptables 규칙, 포트 | `CLONE_NEWNET` |
| `MNT` | 마운트 포인트(컨테이너 루트 파일시스템 뷰) | `CLONE_NEWNS` |
| `UTS` | 호스트네임과 도메인네임 | `CLONE_NEWUTS` |
| `IPC` | System V IPC, POSIX 메시지 큐 | `CLONE_NEWIPC` |
| `USER` | 사용자/그룹 ID(안의 root와 바깥의 root는 다름) | `CLONE_NEWUSER` |
| `CGROUP` | cgroup 계층 구조 뷰 | `CLONE_NEWCGROUP` |

네임스페이스는 세 가지 시스템 콜로 만들고 다룹니다.

- `clone(flags, ...)` — 자식 프로세스를 fork하면서 동시에 새 네임스페이스에 배치합니다. 컨테이너 생성의 핵심.
- `unshare(flags)` — *현재* 프로세스를 공유 네임스페이스에서 떼어냅니다. 같은 이름의 CLI 도구로 셸에서 직접 실험할 수 있습니다.
- `setns(fd, ...)` — 파일 디스크립터로 기존 네임스페이스에 합류합니다. `docker exec`이 실행 중인 컨테이너의 네임스페이스에 들어가는 방식입니다.

PID 네임스페이스 안에서는 첫 프로세스가 PID 1을 받으며(일반 시스템의 `init`과 같은 번호), 커널이 네임스페이스 바깥의 모든 PID를 숨깁니다. 컨테이너는 호스트 프로세스를 정말로 볼 수 없습니다 — 격리는 출력을 필터링하는 것이 아니라 시스템 콜 계층에서 강제됩니다.

USER 네임스페이스가 가장 늦게 들어왔으면서도 가장 강력합니다. 네임스페이스 안의 UID/GID 범위를 바깥의 다른 범위로 매핑해 줍니다. 그래서 컨테이너 내부에서 UID 0(root)인 프로세스가 호스트에서는 UID 100000(권한이 없는 사용자)일 수 있습니다. *루트리스(rootless) 컨테이너*의 기반이 바로 이것입니다.

#### 이론: cgroups — 프로세스가 쓸 수 있는 양을 제한

네임스페이스가 자원을 *숨긴다면*, cgroups는 자원을 *계량*합니다. cgroup은 계층 트리의 노드이며, 각 노드는 부착된 프로세스 집합과 자원 한도를 가지고, 커널의 "컨트롤러(controller)"가 그 한도를 강제합니다.

- `cpu` — CPU 점유율(비례 가중치)과 쿼터(하드 캡, 예: "최대 1.5코어").
- `memory` — RSS 한도(초과 시 OOM-kill), 스왑 한도, 커널 메모리.
- `io`(예전 `blkio`) — 블록 디바이스 읽기/쓰기 대역폭과 IOPS.
- `pids` — 최대 프로세스 수(fork bomb 방어).
- `cpuset` — 특정 CPU와 NUMA 메모리 노드에 프로세스를 핀(pin).

실제 환경에는 두 버전이 공존합니다.

- **cgroup v1** — 컨트롤러마다 별도의 계층 트리를 가집니다. 한 프로세스가 `cpu` 트리와 `memory` 트리에서 서로 다른 위치에 있을 수 있습니다. 유연하지만 운영상 혼란스럽고, 잘못 섞으면 망가지기 쉽습니다.
- **cgroup v2** — 모든 컨트롤러를 단일 계층으로 통합합니다. 모델이 단순하고, 자원 회계(특히 메모리 압박 메트릭)가 더 정확하며, 루트리스 cgroup 위임 같은 최신 기능에 필수입니다. 최근 커널(5.x+)과 배포판은 v2를 기본으로 합니다.

`docker run --memory=512m --cpus=1.5 myimage`을 실행하면, Docker가 cgroup을 만들고 `memory.max` 파일에 `512M`을, CPU 관련 파일에 적절한 값을 쓴 뒤 컨테이너 프로세스를 그 cgroup에 `clone()`합니다. 나머지는 커널이 처리합니다 — 프로세스가 좋든 싫든 512 MB에서 OOM-kill됩니다.

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

### 이론: Docker 엔진 스택 — dockerd, containerd, runc

`docker run`을 입력하면 네 컴포넌트가 작업을 차례로 넘깁니다.

1. **`docker`(CLI)** — 명령을 파싱해 로컬 데몬 소켓(`/var/run/docker.sock`)으로 HTTP 요청을 보냅니다.
2. **`dockerd`(데몬)** — 상위 관심사를 담당합니다. 이미지 풀링, 네트워크 구성, 볼륨 관리, 빌드 오케스트레이션. 컨테이너를 직접 실행하지는 *않습니다*.
3. **`containerd`** — 더 낮은 수준의 데몬. 컨테이너 라이프사이클을 책임집니다. 이미지 저장, 스냅샷 관리, OCI 런타임 호출. dockerd가 containerd에게 "이 OCI 번들을 실행해라"라고 요청합니다.
4. **`runc`** — OCI 호환 런타임. 작은 정적 바이너리이며, OCI 번들(루트 파일시스템과 네임스페이스/cgroup/capability 명세를 담은 `config.json` 디렉터리)을 읽고, 적절한 플래그로 `clone()`을 호출하고, cgroup을 설정하고, capability를 떨어뜨리고, 컨테이너의 엔트리포인트를 `exec()`합니다.

이렇게 분리한 이유는 각 계층의 인터페이스가 안정적으로 다르기 때문입니다. **CRI-O**는 일부 쿠버네티스 배포판이 쓰는 containerd의 대안으로, Kubernetes Container Runtime Interface와 직접 통신하면서 결국 runc를 호출합니다 — 그래서 커널 수준의 컨테이너는 Docker가 만들었을 때와 동일합니다. **crun**은 C로 다시 쓴 더 빠른 runc 구현으로, 위쪽을 건드리지 않고도 교체할 수 있습니다.

OCI(Open Container Initiative)는 containerd급과 runc급 컴포넌트 사이의 경계를 표준화합니다. **image-spec**은 이미지 레이아웃을, **runtime-spec**은 `config.json`이 담아야 할 내용을, **distribution-spec**은 레지스트리가 이미지를 서빙하는 방법을 정의합니다. 그래서 Docker 이미지를 Podman으로 실행할 수 있고, Buildah가 만든 이미지를 Docker가 그대로 풀(pull)할 수 있으며, containerd를 다른 구현으로 바꾸는 것도 가능합니다.

구체적으로 이 레슨의 모든 명령은 위 메커니즘에 친근한 동사를 붙인 것에 불과합니다.

- `docker run -it ubuntu bash` — `dockerd`가 `containerd`에게, `containerd`가 `runc`에게 요청하고, `runc`가 `clone(CLONE_NEWPID|CLONE_NEWNET|CLONE_NEWNS|...)`과 `execve("bash")`를 호출합니다. TTY 플래그(`-it`)가 stdin/stdout을 터미널에 연결합니다.
- `docker run -p 8080:80 nginx` — Docker가 NET 네임스페이스를 만들고 `veth` 페어의 한쪽을 컨테이너에, 다른 쪽을 `docker0` 브리지에 붙인 뒤, 호스트 8080 포트로 들어오는 트래픽을 컨테이너의 80 포트로 재작성하는 iptables DNAT 규칙을 씁니다.
- `docker run --memory=512m --cpus=1` — Docker가 `memory.max=512M`과 한 코어에 해당하는 CPU 쿼터를 가진 cgroup을 만들고 컨테이너 PID를 그 안에 넣습니다.
- `docker ps` / `docker exec` — 모두 containerd 상태를 들여다보며(`exec`은 추가로 `setns()`을 사용) 이미 실행 중인 컨테이너의 네임스페이스를 찾거나 합류합니다.
- `docker version` — 위에서 설명한 4계층 스택(client → daemon → containerd → runc)을 그대로 보여줍니다.

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
