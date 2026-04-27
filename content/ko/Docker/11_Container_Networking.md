# 컨테이너 네트워킹(Container Networking)

**이전**: [CI/CD 파이프라인](./10_CI_CD_Pipelines.md) | **다음**: [보안 베스트 프랙티스](./12_Security_Best_Practices.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Docker 네트워크 드라이버(bridge, host, overlay, macvlan)와 각각의 적합한 사용 사례를 설명한다
2. 서브넷(Subnet), 게이트웨이(Gateway), IP 할당을 포함한 커스텀 브리지 네트워크를 구성한다
3. Swarm 클러스터에서 멀티 호스트 컨테이너 통신을 위한 오버레이 네트워크(Overlay Network)를 구현한다
4. 컨테이너 간 통신을 위한 DNS 기반 서비스 디스커버리(Service Discovery)를 적용한다
5. 호스트 바인딩과 프로토콜 선택을 포함한 고급 포트 매핑 전략을 구성한다
6. 격리, 암호화, 접근 제어를 통한 네트워크 보안을 구현한다
7. 진단 도구를 활용해 컨테이너 네트워크 연결 문제를 해결한다

## 목차


1. [Docker 네트워크 드라이버](#1-docker-네트워크-드라이버)
2. [브리지 네트워크 심화](#2-브리지-네트워크-심화)
3. [호스트 및 None 네트워크](#3-호스트-및-none-네트워크)
4. [오버레이 네트워크](#4-오버레이-네트워크)
5. [네트워크 구성](#5-네트워크-구성)
6. [DNS와 서비스 디스커버리](#6-dns와-서비스-디스커버리)
7. [고급 포트 매핑](#7-고급-포트-매핑)
8. [네트워크 보안](#8-네트워크-보안)
9. [문제 해결](#9-문제-해결)
10. [연습 문제](#10-연습-문제)

**난이도**: ⭐⭐⭐

---

컨테이너 네트워킹은 Docker를 프로덕션에서 운용할 때 가장 복잡하면서도 핵심적인 영역 중 하나입니다. 모든 컨테이너는 다른 컨테이너, 호스트, 그리고 외부 세계와 통신해야 하며, 선택하는 네트워킹 모델은 성능, 보안, 안정성에 직접적인 영향을 미칩니다. 이 레슨에서는 Docker의 네트워크 드라이버, DNS 기반 서비스 디스커버리(Service Discovery), 보안 구성을 심층적으로 다루어, 자신감을 갖고 컨테이너 네트워크를 설계하고 문제를 해결할 수 있도록 합니다.

---

## 1. Docker 네트워크 드라이버

Docker는 다양한 사용 사례를 위한 여러 네트워크 드라이버를 제공합니다.

### 이론: 네트워크 네임스페이스 — 컨테이너별 네트워크 스택

컨테이너 네트워킹은 각 네트워크 드라이버가 리눅스 커널 기능들의 작은 합성임을 깨닫기 전까지는 마법처럼 보입니다. `docker network create -d bridge`의 "네트워킹"은 **네트워크 네임스페이스** + **veth 페어** + **리눅스 브리지** + **iptables NAT 규칙**입니다. 오버레이 네트워킹이 **VXLAN 캡슐화**를 추가합니다. Macvlan이 **가상 MAC 서브 인터페이스**를 추가합니다.

커널의 **네트워크 네임스페이스**는 네트워크 관련 모든 것을 격리합니다 — 인터페이스, 라우팅 테이블, iptables 규칙, 소켓, 포트 할당. 네트워크 네임스페이스 안의 프로세스는 그 네임스페이스가 담은 인터페이스와 라우트만 봅니다.

Docker가 컨테이너를 시작하면 `runc`에 `clone(CLONE_NEWNET)` 호출을 요청해 새 프로세스를 신선한 빈 네트워크 네임스페이스에 둡니다. 그 네임스페이스에는 `lo`(loopback) 인터페이스만 있고, 라우트 없고, 어디에도 닿을 수 없습니다. 컨테이너가 결국 갖는 어떤 연결성이든 Docker(또는 `runc`의 셋업 훅)가 컨테이너의 메인 프로세스가 `exec`되기 전에 그 네임스페이스에 들어가 구성한 결과입니다.

`lsns -t net`으로 네임스페이스 목록, `nsenter -t <pid> -n <command>`로 진입, `ip netns add foo`로 수동 생성. Docker가 네트워크적으로 하는 모든 것은 `ip` 명령으로 손수 재현 가능 — Docker는 그것을 자동화할 뿐.

### 이론: macvlan, ipvlan, CNI

**Macvlan**은 부모 물리 인터페이스를 공유하면서 자체 MAC 주소를 가진 가상 서브 인터페이스를 생성합니다. 컨테이너가 호스트의 L2 네트워크에 직접 IP를 가짐 — NAT 없음, 브리지 없음, 포트 매핑 없음. 네트워크의 다른 시각에서 각 컨테이너가 자체 MAC과 IP를 가진 1급 호스트. 사용 사례 — 자체 IP로 회사 LAN에 있어야 하는 레거시 애플리케이션, 또는 MAC 기반 ACL을 모니터링하는 애플리케이션. 트레이드오프 — 대부분의 클라우드 공급자가 macvlan에 필요한 "promiscuous" 모드를 차단(AWS / GCP / Azure VM에서 기본적으로 사용 불가), 일부 스위치가 포트당 MAC을 제한.

**Ipvlan**은 비슷하지만 서브 인터페이스가 부모의 MAC을 공유하고 IP만 다릅니다. 클라우드 호환성 더 좋지만 L3 전용(서브 인터페이스 사이 broadcast/multicast 없음).

Kubernetes는 내장 네트워크 드라이버가 없습니다. **CNI(Container Network Interface)** 명세를 정의 — "이 네트워크 네임스페이스와 이 파라미터로 컨테이너의 네트워킹을 셋업"의 JSON 스키마. kubelet이 stdin의 JSON 설정으로 `/opt/cni/bin/<plugin> ADD/DEL`을 호출, 플러그인이 veth, IP 할당, 라우트 등을 셋업하고 한 일을 기술하는 JSON을 반환. 표준 CNI 플러그인 — `bridge`(Docker 기본 브리지와 같은 아이디어), `host-local`(정적 범위에서 할당하는 IPAM), `flannel`(단순 컨트롤 플레인의 VXLAN 오버레이), `calico`(BGP 라우팅, NetworkPolicy 구현), `cilium`(eBPF 데이터플레인, iptables 대신 eBPF의 NetworkPolicy + Service 로드 밸런싱), `weave`(암호화된 VXLAN류 오버레이).

CNI 명세 덕분에 오케스트레이터가 특정 네트워킹 구현에 의존하지 않습니다. CNI 플러그인을 교체해 클러스터 네트워킹 동작을 바꿉니다. Docker 엔진은 CNI를 앞서는 자체 libnetwork 플러그인 시스템을 가집니다. Kubernetes에서는 kubelet이 찾는 CNI 바이너리만 중요합니다.

### 네트워크 드라이버 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network Drivers                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  bridge  │  │   host   │  │ overlay  │  │ macvlan  │   │
│  │          │  │          │  │          │  │          │   │
│  │ Default  │  │  Native  │  │  Swarm   │  │  Legacy  │   │
│  │ Isolated │  │  Network │  │Multi-host│  │  Bridge  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                                                              │
│              ┌──────────┐                                    │
│              │   none   │                                    │
│              │          │                                    │
│              │ Disabled │                                    │
│              └──────────┘                                    │
└─────────────────────────────────────────────────────────────┘
```

### 네트워크 나열

```bash
# List all networks
docker network ls

# Output:
# NETWORK ID     NAME      DRIVER    SCOPE
# 3c7f2a8b4d91   bridge    bridge    local
# 9f8e3d2c1a45   host      host      local
# 1b5a6c9d3e72   none      null      local
```

### 네트워크 검사

```bash
# Detailed network information
docker network inspect bridge

# Output (truncated):
# [
#     {
#         "Name": "bridge",
#         "Driver": "bridge",
#         "IPAM": {
#             "Config": [
#                 {
#                     "Subnet": "172.17.0.0/16",
#                     "Gateway": "172.17.0.1"
#                 }
#             ]
#         },
#         "Containers": {...}
#     }
# ]
```

### 네트워크 드라이버 사용 사례

| 드라이버(Driver) | 사용 사례 | 범위(Scope) | DNS |
|--------|----------|-------|-----|
| **bridge** | 단일 호스트, 격리된 컨테이너 | Local | Yes (사용자 정의) |
| **host** | 높은 성능, 격리 없음 | Local | Host DNS |
| **overlay** | 다중 호스트, Swarm 서비스 | Swarm | Yes |
| **macvlan** | MAC 주소가 필요한 레거시 앱 | Local | No |
| **none** | 완전한 격리, 커스텀 네트워킹 | Local | No |

---

## 2. 브리지 네트워크 심화

### 이론: veth 페어 — 네임스페이스 사이의 케이블

**veth(virtual Ethernet) 페어**는 서로 연결된 두 가상 인터페이스. 한 끝으로 보낸 것은 다른 끝으로 정확히 물리 케이블처럼 나옵니다. 인터페이스가 다른 네임스페이스에 살 수 있어, veth 페어가 호스트 네트워크 네임스페이스와 컨테이너 네트워크 네임스페이스를 *연결*하는 표준 방법.

기본 브리지의 컨테이너에 대해 Docker는 —

1. 호스트에 `vethXXXX` 생성(한 끝).
2. 컨테이너 netns 안에 `eth0` 생성(다른 끝).
3. `vethXXXX`를 호스트 네트워크 네임스페이스로 이동(거기에 부착된 채로).
4. `eth0`을 컨테이너 netns로 이동하고 `docker0`의 서브넷에서 IP(예: `172.17.0.2/16`)로 구성.
5. `vethXXXX`를 `docker0` 리눅스 브리지의 멤버로 추가.

이제 컨테이너의 `eth0` 안에서 보낸 패킷이 호스트의 `vethXXXX`로 나와 `docker0` 브리지로 들어가고, 거기서 브리지에 부착된 다른 어떤 veth(즉, 같은 네트워크의 다른 컨테이너)로든 포워딩될 수 있습니다.

### 이론: 리눅스 브리지 — veth 엔드포인트용 소프트웨어 스위치

`docker0`은 **리눅스 브리지** — 커널에 구현된 소프트웨어 L2 스위치. 하드웨어 스위치처럼 MAC 주소 테이블을 갖고, 목적지 MAC으로 부착된 인터페이스 사이에 이더넷 프레임을 포워딩.

사용자 정의 브리지 네트워크(`docker network create my-net`)를 만들면 Docker가 새 리눅스 브리지(`br-XXXXXXXXXXXX`)를 만들고 `docker0` 대신 거기에 컨테이너 veth를 부착. `docker0` 대비 두 핵심 차이 —

- **DNS 기반 서비스 디스커버리 활성화.** Docker가 사용자 정의 브리지의 모든 컨테이너 이름을 아는 임베디드 DNS 서버(각 컨테이너에서 닿는 `127.0.0.11`)를 돌립니다. 컨테이너가 컨테이너 이름으로 서로 리졸브 가능. 기본 `docker0` 브리지에는 이게 *없음* — 같은 효과를 위해 `--link`(폐기됨) 필요.
- **컨테이너 간 트래픽이 기본 허용.** 둘 다 같지만, 사용자 정의 브리지는 `--internal`(외부 연결성 전혀 없음)와 `--icc=false`(컨테이너 간 트래픽 차단) 플래그 가능.

사용자 정의 브리지는 사소하지 않은 어떤 셋업에도 권장 원시.

브리지 네트워크(Bridge Network)는 컨테이너에 가장 일반적인 네트워크 유형입니다.

### 기본 브리지 vs 사용자 정의 브리지

```
Default Bridge Network              User-Defined Bridge Network
┌──────────────────────┐            ┌──────────────────────┐
│   172.17.0.0/16      │            │   172.20.0.0/16      │
│                      │            │                      │
│  ┌────────────┐      │            │  ┌────────────┐      │
│  │ Container1 │      │            │  │ Container1 │      │
│  │ 172.17.0.2 │      │            │  │ 172.20.0.2 │      │
│  │            │      │            │  │ web        │      │
│  └────────────┘      │            │  └────────────┘      │
│                      │            │         │            │
│  ┌────────────┐      │            │         │ DNS        │
│  │ Container2 │      │            │         ▼            │
│  │ 172.17.0.3 │      │            │  ┌────────────┐      │
│  │            │      │            │  │ Container2 │      │
│  └────────────┘      │            │  │ 172.20.0.3 │      │
│                      │            │  │ db         │      │
│  No automatic DNS    │            │  └────────────┘      │
│  Link by IP only     │            │                      │
└──────────────────────┘            │  Automatic DNS       │
                                    │  Link by name        │
                                    └──────────────────────┘
```

### 사용자 정의 브리지 네트워크 생성

```bash
# Create custom bridge network
docker network create my-app-network

# Create with custom subnet
docker network create \
  --driver bridge \
  --subnet 172.25.0.0/16 \
  --gateway 172.25.0.1 \
  my-custom-network

# Create with IP range reservation — carves out a smaller pool for dynamic allocation, leaving room for static IPs outside the range
docker network create \
  --subnet 172.26.0.0/16 \
  --ip-range 172.26.5.0/24 \
  my-reserved-network
```

### 브리지 네트워크에 컨테이너 연결

```bash
# Run container on custom network
docker run -d \
  --name web \
  --network my-app-network \
  nginx

# Run another container on same network
docker run -d \
  --name db \
  --network my-app-network \
  postgres

# Test DNS resolution
docker exec web ping db
# PING db (172.25.0.3): 56 data bytes
# 64 bytes from 172.25.0.3: seq=0 ttl=64 time=0.123 ms
```

### 런타임에 네트워크 연결/연결 해제

```bash
# Connect running container to additional network
docker network connect my-app-network my-container

# Disconnect from network
docker network disconnect my-app-network my-container

# Connect with static IP
docker network connect --ip 172.25.0.100 my-app-network my-container
```

### Docker Compose 브리지 네트워크

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    image: nginx
    networks:
      - frontend
      - backend  # web bridges both networks — it reverse-proxies public traffic to the backend tier

  app:
    image: myapp:latest
    networks:
      backend:
        ipv4_address: 172.28.0.100  # Static IP — useful when external config or firewalls reference a fixed address

  db:
    image: postgres
    networks:
      - backend  # db is only on backend — unreachable from the frontend network, reducing attack surface

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge  # Separate bridge isolates backend traffic — containers on frontend cannot sniff DB queries
    ipam:
      config:
        - subnet: 172.28.0.0/16
          gateway: 172.28.0.1
```

---

## 3. 호스트 및 None 네트워크

### 호스트 네트워크(Host Network)

컨테이너가 호스트의 네트워크 스택을 직접 공유합니다.

```
┌─────────────────────────────────────────┐
│            Host Network                  │
│  ┌───────────────────────────────────┐  │
│  │         Host OS Network           │  │
│  │                                   │  │
│  │  ┌──────────┐    ┌──────────┐    │  │
│  │  │Container1│    │Container2│    │  │
│  │  │  :80     │    │  :443    │    │  │
│  │  └──────────┘    └──────────┘    │  │
│  │                                   │  │
│  │  No network isolation             │  │
│  │  Direct host network access       │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

**사용 사례**:
- 높은 네트워크 성능 (NAT 오버헤드 없음)
- 네트워크 모니터링 도구
- 호스트 네트워크 기능이 필요한 서비스

**제한 사항**:
- 네트워크 격리 없음
- 호스트와의 포트 충돌
- 동일한 포트에서 여러 컨테이너를 실행할 수 없음

```bash
# Run container with host network
docker run -d \
  --name nginx-host \
  --network host \
  nginx

# Container listens on host's port 80 directly
# No -p flag needed (ignored if specified)

# Check listening ports
netstat -tlnp | grep 80
# tcp  0  0  0.0.0.0:80  0.0.0.0:*  LISTEN  12345/nginx
```

### 성능 비교

```bash
# Bridge network (with NAT)
docker run --rm --network bridge alpine ping -c 4 8.8.8.8
# avg RTT: ~0.5ms overhead

# Host network (no NAT)
docker run --rm --network host alpine ping -c 4 8.8.8.8
# avg RTT: native host performance
```

### None 네트워크

완전한 네트워크 격리.

```
┌─────────────────────────────────────────┐
│           None Network                   │
│                                          │
│        ┌──────────────┐                  │
│        │  Container   │                  │
│        │              │                  │
│        │  No network  │                  │
│        │  interface   │                  │
│        │              │                  │
│        │  Only: lo    │                  │
│        └──────────────┘                  │
│                                          │
└─────────────────────────────────────────┘
```

**사용 사례**:
- 완전한 네트워크 격리
- 커스텀 네트워크 스택 구현
- 테스트 시나리오

```bash
# Run container with no network
docker run -d \
  --name isolated \
  --network none \
  alpine sleep 3600

# Verify no network interfaces (except loopback)
docker exec isolated ip addr
# 1: lo: <LOOPBACK,UP,LOWER_UP>
#     inet 127.0.0.1/8 scope host lo
```

---

## 4. 오버레이 네트워크

### 이론: 오버레이 네트워크와 VXLAN 캡슐화

다중 호스트 네트워킹(Swarm, 오버레이 CNI 있는 Kubernetes)에서 다른 호스트의 컨테이너들이 같은 L2 세그먼트에 있는 것처럼 통신해야 합니다. 표준 메커니즘은 **VXLAN(Virtual eXtensible LAN) 캡슐화**입니다.

흐름 —

1. Host 1의 Container A가 Host 2의 Container B로 이더넷 프레임 전송.
2. 프레임이 Host 1의 오버레이 브리지에 도착, 브리지가 Container B는 Host 2 "뒤"에 있다고 앎.
3. Host 1의 VXLAN 드라이버가 전체 이더넷 프레임을 Host 2 주소의 UDP 데이터그램(VXLAN 헤더 + 외부 IP/UDP)으로 감쌈.
4. UDP 패킷이 underlay 네트워크(실제 LAN/VPC)를 가로질러 이동.
5. Host 2가 UDP를 받아 내부 이더넷 프레임을 디캡슐화하고 Container B의 veth로 전달.

"L2 세그먼트"는 가상이고, 물리적으로는 그저 4789 포트의 UDP 트래픽. VXLAN이 24비트 **VNI(Virtual Network Identifier)**를 할당해 여러 오버레이 네트워크가 같은 underlay를 공유 가능. 컨트롤 플레인(어느 호스트에 어느 컨테이너가 있고, 어느 VNI에)은 오케스트레이터가 유지 — Swarm은 gossip 사용, Kubernetes 오버레이 CNI(Flannel, Weave, Calico VXLAN)는 다양한 메커니즘 사용.

비용은 패킷당 오버헤드(~50바이트 헤더), MTU 튜닝 두통(내부 MTU가 외부보다 작아야 공간 확보), 더 어려운 디버깅(`tcpdump`가 VXLAN 감싼 트래픽을 보여줌. `tcpdump -v vxlan`이나 컨테이너 netns 내부에서 캡처 필요).

오버레이 네트워크(Overlay Network)는 Docker Swarm에서 다중 호스트 컨테이너 통신을 가능하게 합니다.

### 오버레이 네트워크 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    Overlay Network                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────┐      ┌─────────────────────┐      │
│  │    Host 1           │      │    Host 2           │      │
│  │  ┌──────────────┐   │      │  ┌──────────────┐   │      │
│  │  │ Container A  │   │◄────►│  │ Container B  │   │      │
│  │  │ 10.0.0.2     │   │VXLAN │  │ 10.0.0.3     │   │      │
│  │  └──────────────┘   │Tunnel│  └──────────────┘   │      │
│  │                     │      │                     │      │
│  │  Physical: 192.168.1.10    │  Physical: 192.168.1.20  │
│  └─────────────────────┘      └─────────────────────┘      │
│                                                              │
│  Overlay subnet: 10.0.0.0/24                                │
│  Underlay network: 192.168.1.0/24                           │
└─────────────────────────────────────────────────────────────┘
```

### 오버레이 네트워크 생성

```bash
# Initialize Swarm (required for overlay networks)
docker swarm init --advertise-addr 192.168.1.10

# Create overlay network
docker network create \
  --driver overlay \
  --subnet 10.0.9.0/24 \
  my-overlay

# Create with encryption — IPsec encrypts VXLAN traffic so inter-node communication is confidential even on untrusted networks
docker network create \
  --driver overlay \
  --opt encrypted \
  --subnet 10.0.10.0/24 \
  secure-overlay

# Create attachable overlay (for standalone containers) — without --attachable, only Swarm services can join
docker network create \
  --driver overlay \
  --attachable \
  --subnet 10.0.11.0/24 \
  attachable-overlay
```

### 오버레이 네트워크에 서비스 배포

```bash
# Create service on overlay network
docker service create \
  --name web \
  --network my-overlay \
  --replicas 3 \
  nginx

# Create backend service
docker service create \
  --name api \
  --network my-overlay \
  --replicas 2 \
  myapi:latest

# Services can communicate by name across hosts
docker exec <web-container> curl http://api:8080
```

### Docker Compose와 오버레이 (Swarm Stack)

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    image: nginx
    deploy:
      replicas: 3
    networks:
      - frontend

  api:
    image: myapi:latest
    deploy:
      replicas: 2
    networks:
      - frontend
      - backend

  db:
    image: postgres
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.role == manager
    networks:
      - backend
    volumes:
      - db-data:/var/lib/postgresql/data

networks:
  frontend:
    driver: overlay
  backend:
    driver: overlay
    driver_opts:
      encrypted: "true"  # Encrypt DB traffic between nodes — prevents eavesdropping on the physical network

volumes:
  db-data:
```

```bash
# Deploy stack
docker stack deploy -c docker-compose.yml myapp

# List networks
docker network ls
# NETWORK ID     NAME              DRIVER    SCOPE
# abc123def456   myapp_frontend    overlay   swarm
# def789ghi012   myapp_backend     overlay   swarm
```

### 오버레이 네트워크 암호화

```bash
# Create encrypted overlay
docker network create \
  --driver overlay \
  --opt encrypted=true \
  --subnet 10.0.20.0/24 \
  encrypted-net

# IPsec encrypts VXLAN traffic between nodes
# Performance impact: ~10-20% overhead
```

---

## 5. 네트워크 구성

### 커스텀 서브넷과 게이트웨이

```bash
# Create network with custom IPAM
docker network create \
  --driver bridge \
  --subnet 172.30.0.0/16 \
  --gateway 172.30.0.1 \
  --ip-range 172.30.5.0/24 \
  --aux-address "my-router=172.30.1.1" \
  custom-net
```

### MTU 구성

```bash
# Set MTU (Maximum Transmission Unit) — match the underlying network's MTU to avoid packet fragmentation and throughput loss
docker network create \
  --driver bridge \
  --opt com.docker.network.driver.mtu=1450 \
  low-mtu-net

# Useful for:
# - VPN/overlay networks (avoid fragmentation)
# - Cloud environments (GCP: 1460, AWS: 9001 for jumbo frames)
```

### IPv6 지원

```bash
# Enable IPv6 in daemon.json
# /etc/docker/daemon.json
{
  "ipv6": true,
  "fixed-cidr-v6": "2001:db8:1::/64"
}

# Restart Docker
sudo systemctl restart docker

# Create network with IPv6
docker network create \
  --ipv6 \
  --subnet 172.31.0.0/16 \
  --subnet 2001:db8:2::/64 \
  ipv6-net
```

### 네트워크 드라이버 옵션

```bash
# Bridge options
docker network create \
  --driver bridge \
  --opt com.docker.network.bridge.name=my-bridge \
  --opt com.docker.network.bridge.enable_icc=true \
  --opt com.docker.network.bridge.enable_ip_masquerade=true \
  my-configured-net
```

**일반 옵션**:
- `com.docker.network.bridge.name`: 커스텀 브리지 인터페이스 이름
- `com.docker.network.bridge.enable_icc`: 컨테이너 간 통신(Inter-Container Communication) (기본값: true)
- `com.docker.network.bridge.enable_ip_masquerade`: 외부 트래픽을 위한 NAT (기본값: true)
- `com.docker.network.driver.mtu`: MTU 크기 (기본값: 1500)

---

## 6. DNS와 서비스 디스커버리

### 이론: DNS 기반 서비스 디스커버리

Docker 데몬은 각 사용자 정의 브리지 네트워크 안에서 `127.0.0.11`에 임베디드 DNS 서버를 돌립니다. 컨테이너가 `web`을 질의하면 리졸버가 —

1. `/etc/hosts` 확인(Docker가 `127.0.0.11 ndots:0` 리졸버 힌트와 컨테이너 자체 이름을 씀).
2. 질의를 `127.0.0.11`로 포워드.
3. Docker DNS가 로컬 네트워크의 이름 테이블에서 `web` 조회 — 모든 컨테이너의 `--name`(과 모든 alias)이 등록됨.
4. 컨테이너의 IP 반환.

외부 이름(`google.com`)에 대해서는 Docker DNS가 호스트의 상류 리졸버(보통 호스트의 `/etc/resolv.conf`)로 포워드. `docker run`의 `--dns`나 compose의 `dns:`로 덮어쓰기 가능.

Kubernetes의 동등물은 Pod으로 도는 **CoreDNS** + 모든 Pod의 `/etc/resolv.conf`에 `nameserver`로 주입된 Service IP. `my-svc.my-ns.svc.cluster.local` 같은 서비스 이름이 ClusterIP로 리졸브.

### 임베디드 DNS 서버

Docker는 컨테이너 이름에 대한 자동 DNS 해석을 제공합니다.

```
┌─────────────────────────────────────────┐
│         User-Defined Network            │
│                                         │
│  ┌──────────┐         ┌──────────┐     │
│  │   web    │         │   db     │     │
│  │          │         │          │     │
│  │          │──DNS───►│          │     │
│  │          │  query  │          │     │
│  │          │  "db"   │          │     │
│  └──────────┘         └──────────┘     │
│       │                                 │
│       │ DNS query                       │
│       ▼                                 │
│  ┌─────────────────────────┐           │
│  │  Embedded DNS Server    │           │
│  │  (127.0.0.11:53)        │           │
│  │                         │           │
│  │  web → 172.20.0.2       │           │
│  │  db  → 172.20.0.3       │           │
│  └─────────────────────────┘           │
└─────────────────────────────────────────┘
```

### DNS 해석 예제

```bash
# Create network and containers
docker network create my-net
docker run -d --name web --network my-net nginx
docker run -d --name db --network my-net postgres

# Test DNS resolution
docker exec web nslookup db
# Server:    127.0.0.11
# Address:   127.0.0.11:53
#
# Name:      db
# Address:   172.20.0.3

# Ping by container name
docker exec web ping -c 2 db
# PING db (172.20.0.3): 56 data bytes
# 64 bytes from 172.20.0.3: seq=0 ttl=64 time=0.123 ms
```

### 커스텀 DNS 구성

```bash
# Run container with custom DNS servers
docker run -d \
  --name custom-dns \
  --dns 8.8.8.8 \
  --dns 8.8.4.4 \
  --dns-search example.com \
  nginx

# Verify DNS configuration
docker exec custom-dns cat /etc/resolv.conf
# nameserver 8.8.8.8
# nameserver 8.8.4.4
# search example.com
```

### Docker Compose에서 서비스 디스커버리

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    image: nginx
    networks:
      - app-net

  api:
    image: myapi:latest
    environment:
      # Resolve by service name
      - DATABASE_URL=postgresql://db:5432/mydb
    networks:
      - app-net

  db:
    image: postgres
    networks:
      - app-net

networks:
  app-net:
    driver: bridge
```

### DNS 라운드 로빈 (여러 컨테이너)

```bash
# Create multiple containers with same name (using --network-alias)
# DNS round-robin provides basic load balancing without an external LB — good enough for internal service-to-service calls
docker run -d --name api1 --network my-net --network-alias api myapi:latest
docker run -d --name api2 --network my-net --network-alias api myapi:latest
docker run -d --name api3 --network my-net --network-alias api myapi:latest

# DNS query returns all IPs (round-robin)
docker run --rm --network my-net alpine nslookup api
# Name:      api
# Address:   172.20.0.2
# Address:   172.20.0.3
# Address:   172.20.0.4
```

---

## 7. 고급 포트 매핑

### 이론: iptables NAT — 컨테이너가 인터넷에 닿는 법, 호스트가 컨테이너에 닿는 법

`docker0`의 컨테이너는 호스트 외부 네트워크에서 라우팅 안 되는 사설 IP(`172.17.0.X`)를 갖습니다. outbound 트래픽을 위해 Docker가 **MASQUERADE** 규칙 설치 —

```
iptables -t nat -A POSTROUTING -s 172.17.0.0/16 ! -o docker0 -j MASQUERADE
```

번역 — 브리지 서브넷 출처의 어떤 패킷이든 `docker0` 자체가 아닌 어떤 인터페이스로 나가면 출처 IP를 호스트의 outbound 인터페이스 IP로 재작성. 커널이 연결을 추적하고 응답이 돌아올 때 재작성. 컨테이너가 인터넷 접근을 갖고, 외부 세계는 호스트 IP를 봄.

`docker run -p 8080:80`을 통한 inbound 트래픽을 위해 Docker가 **DNAT** 규칙 설치 —

```
iptables -t nat -A DOCKER -p tcp --dport 8080 -j DNAT --to-destination 172.17.0.2:80
```

추가로 헤어핀과 특정 엣지 케이스용 폴백으로 `docker-proxy` 유저 공간 프로세스. DNAT가 들어오는 패킷의 목적지 IP/포트를 재작성해 컨테이너로 포워드, 컨테이너가 응답하면 커널이 응답을 역재작성.

호스트의 `iptables -t nat -L -n -v`가 Docker가 설치한 모든 게시 포트와 outbound 규칙을 보여 줍니다. `docker run -p`가 동작 안 할 때 여기를 봐야 합니다.

### 포트 퍼블리싱 모드

```bash
# Publish to random host port
docker run -d -P nginx
# Maps all EXPOSE ports to random high ports (32768+)

# Publish to specific host port
docker run -d -p 8080:80 nginx

# Publish to specific interface — binds only to loopback, preventing external network access to this port
docker run -d -p 127.0.0.1:8080:80 nginx
# Only accessible from localhost

# Publish UDP port
docker run -d -p 53:53/udp dns-server

# Publish port range
docker run -d -p 5000-5010:5000-5010 multi-port-app
```

### 포트 매핑 다이어그램

```
┌──────────────────────────────────────────────────────────┐
│                    Host (192.168.1.100)                   │
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │              iptables NAT                        │    │
│  │                                                  │    │
│  │  8080 ──► DNAT ──► 172.17.0.2:80               │    │
│  │  8443 ──► DNAT ──► 172.17.0.2:443              │    │
│  └─────────────────────────────────────────────────┘    │
│                          │                               │
│                          ▼                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │         Docker Bridge (docker0)                 │    │
│  │                                                  │    │
│  │         ┌──────────────────┐                    │    │
│  │         │   nginx          │                    │    │
│  │         │   172.17.0.2     │                    │    │
│  │         │   :80, :443      │                    │    │
│  │         └──────────────────┘                    │    │
│  └─────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘

External request: http://192.168.1.100:8080
    ↓
NAT translation: 172.17.0.2:80
    ↓
Container receives request on port 80
```

### Docker Compose 포트 매핑

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    image: nginx
    ports:
      # Short syntax
      - "8080:80"
      - "8443:443"

      # Long syntax
      - target: 80
        published: 8080
        protocol: tcp
        mode: host

      # Localhost only
      - "127.0.0.1:9090:9090"

      # Port range
      - "5000-5010:5000-5010"
```

### 포트 매핑 검사

```bash
# List port mappings
docker port nginx
# 80/tcp -> 0.0.0.0:8080
# 443/tcp -> 0.0.0.0:8443

# Inspect with docker ps
docker ps --format "table {{.Names}}\t{{.Ports}}"
# NAMES    PORTS
# nginx    0.0.0.0:8080->80/tcp, 0.0.0.0:8443->443/tcp
```

---

## 8. 네트워크 보안

### 네트워크 격리

```
┌─────────────────────────────────────────────────────────┐
│                  Network Isolation                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐        ┌──────────────────┐      │
│  │  frontend-net    │        │  backend-net     │      │
│  │                  │        │                  │      │
│  │  ┌────────┐      │        │  ┌────────┐     │      │
│  │  │  web   │      │        │  │  api   │     │      │
│  │  └────────┘      │        │  └────────┘     │      │
│  │                  │        │        │         │      │
│  └──────────────────┘        │        │         │      │
│                               │  ┌────────┐     │      │
│                               │  │  db    │     │      │
│                               │  └────────┘     │      │
│                               └──────────────────┘      │
│                                                          │
│  web CANNOT communicate with db directly                │
│  api bridges both networks                              │
└─────────────────────────────────────────────────────────┘
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    image: nginx
    networks:
      - frontend

  api:
    image: myapi:latest
    networks:
      - frontend
      - backend

  db:
    image: postgres
    networks:
      - backend
    # db is NOT exposed to frontend network — even if the web tier is compromised, the DB is unreachable

networks:
  frontend:
  backend:
    internal: true  # No external access — containers on this network cannot reach the internet, reducing data exfiltration risk
```

### 내부 네트워크(Internal Networks)

```bash
# Create internal network (no external access)
docker network create \
  --internal \
  --subnet 172.40.0.0/16 \
  internal-net

# Containers on this network cannot reach internet
docker run -d --name isolated-db --network internal-net postgres
```

### 컨테이너 간 통신(ICC)

```bash
# Disable ICC (containers can't talk to each other by default) — forces explicit port publishing for communication, tightening isolation
docker network create \
  --driver bridge \
  --opt com.docker.network.bridge.enable_icc=false \
  restricted-net

# Containers must use port publishing to communicate
```

### iptables를 사용한 네트워크 정책

```bash
# View Docker iptables rules
sudo iptables -t nat -L -n -v | grep DOCKER

# Block traffic between specific containers (manual iptables)
sudo iptables -I DOCKER-USER -s 172.17.0.2 -d 172.17.0.3 -j DROP

# Allow only specific ports
sudo iptables -I DOCKER-USER -p tcp --dport 5432 -s 172.17.0.2 -j ACCEPT
sudo iptables -I DOCKER-USER -s 172.17.0.2 -j DROP
```

### 암호화된 오버레이 네트워크

```bash
# All traffic between nodes is encrypted with IPsec
docker network create \
  --driver overlay \
  --opt encrypted=true \
  secure-overlay
```

---

## 9. 문제 해결

### 네트워크 검사 명령어

```bash
# Inspect network details
docker network inspect my-net

# Show containers on network
docker network inspect my-net --format '{{range .Containers}}{{.Name}} {{end}}'

# Show network configuration of container
docker inspect my-container --format '{{json .NetworkSettings.Networks}}'
```

### 연결성 테스트

```bash
# Ping between containers
docker exec container1 ping container2

# Test DNS resolution
docker exec container1 nslookup container2

# Test port connectivity
docker exec container1 nc -zv container2 80

# Test with curl
docker exec container1 curl http://container2:8080
```

### 패킷 캡처

```bash
# Capture traffic on docker0 bridge
sudo tcpdump -i docker0 -n

# Capture traffic for specific container
# Find container's veth interface
docker inspect my-container --format '{{.NetworkSettings.SandboxKey}}'
# /var/run/docker/netns/abc123def456

# Enter network namespace and capture
sudo nsenter --net=/var/run/docker/netns/abc123def456 tcpdump -i eth0 -n

# Or use docker exec with tcpdump
docker exec my-container tcpdump -i eth0 -n
```

### 일반적인 네트워크 문제

#### 문제 1: 컨테이너가 인터넷에 접근할 수 없음

```bash
# Check DNS configuration
docker exec my-container cat /etc/resolv.conf

# Test DNS resolution
docker exec my-container nslookup google.com

# Test connectivity
docker exec my-container ping 8.8.8.8

# Solution: Check IP masquerading
sudo iptables -t nat -L -n | grep MASQUERADE
# Should see: MASQUERADE  all  --  172.17.0.0/16  0.0.0.0/0
```

#### 문제 2: 컨테이너가 이름으로 통신할 수 없음

```bash
# Only works on user-defined networks, NOT default bridge
# Solution: Create custom network
docker network create my-net
docker network connect my-net container1
docker network connect my-net container2
```

#### 문제 3: 포트가 이미 사용 중

```bash
# Find process using port
sudo lsof -i :8080
# or
sudo netstat -tlnp | grep 8080

# Solution: Stop conflicting process or use different port
docker run -d -p 8081:80 nginx
```

#### 문제 4: DNS 해석 느림

```bash
# Check embedded DNS server
docker exec my-container cat /etc/resolv.conf
# Should see: nameserver 127.0.0.11

# Test DNS performance
docker exec my-container time nslookup container2

# Solution: Add custom DNS servers if needed
docker run --dns 8.8.8.8 --dns 8.8.4.4 my-image
```

### 네트워크 디버깅 도구

```bash
# Run debugging container with network tools
docker run -it --rm --network my-net nicolaka/netshoot

# Available tools in netshoot:
# - ping, traceroute, mtr
# - nslookup, dig, host
# - curl, wget, httpie
# - netcat, socat
# - tcpdump, tshark
# - iftop, nethogs
# - ip, ss, netstat, iptables
```

### 네트워크 로그 보기

```bash
# Enable debug logging in Docker daemon
# /etc/docker/daemon.json
{
  "debug": true,
  "log-level": "debug"
}

# Restart Docker
sudo systemctl restart docker

# View logs
sudo journalctl -u docker -f
```

---

## 10. 연습 문제

### 연습 1: 다계층 애플리케이션 네트워크

격리된 네트워크로 3계층 애플리케이션을 생성합니다.

```yaml
# docker-compose.yml
version: '3.8'

services:
  nginx:
    image: nginx
    ports:
      - "80:80"
    networks:
      - frontend
    depends_on:
      - app

  app:
    image: node:18
    command: node server.js
    networks:
      - frontend
      - backend
    depends_on:
      - db

  db:
    image: postgres
    environment:
      POSTGRES_PASSWORD: secret
    networks:
      - backend

networks:
  frontend:
  backend:
    internal: true
```

**작업**:
1. 스택 배포
2. nginx가 app에 접근할 수 있는지 확인
3. app이 db에 접근할 수 있는지 확인
4. nginx가 db에 직접 접근할 수 없는지 확인
5. db가 인터넷 접근이 없는지 확인

### 연습 2: 정적 IP를 사용한 커스텀 브리지 네트워크

```bash
# Create network with specific subnet
docker network create \
  --driver bridge \
  --subnet 172.50.0.0/24 \
  --gateway 172.50.0.1 \
  --ip-range 172.50.0.128/25 \
  static-net

# Run containers with static IPs
docker run -d \
  --name web \
  --network static-net \
  --ip 172.50.0.10 \
  nginx

docker run -d \
  --name api \
  --network static-net \
  --ip 172.50.0.20 \
  myapi:latest

docker run -d \
  --name db \
  --network static-net \
  --ip 172.50.0.30 \
  postgres
```

**작업**:
1. 컨테이너가 할당된 IP를 가지는지 확인
2. 컨테이너 간 연결성 테스트
3. IP 할당 체계 문서화

### 연습 3: DNS 라운드 로빈 로드 밸런싱

```bash
# Create network
docker network create lb-net

# Create multiple backend containers with same alias
for i in 1 2 3; do
  docker run -d \
    --name backend-$i \
    --network lb-net \
    --network-alias backend \
    hashicorp/http-echo -text="Backend $i"
done

# Create client container
docker run -it --rm \
  --network lb-net \
  alpine sh

# Test DNS round-robin
for i in {1..6}; do
  wget -qO- http://backend:5678
done
```

**예상 출력**:
```
Backend 1
Backend 2
Backend 3
Backend 1
Backend 2
Backend 3
```

### 연습 4: 네트워크 문제 해결

손상된 네트워크 설정을 식별하고 수정합니다.

```yaml
# broken-compose.yml
version: '3.8'

services:
  web:
    image: nginx
    ports:
      - "80:80"
    networks:
      - frontend

  api:
    image: myapi:latest
    environment:
      - DB_HOST=db
    networks:
      - frontend  # BUG: Should be on backend too

  db:
    image: postgres
    networks:
      - backend

networks:
  frontend:
  backend:
```

**작업**:
1. 배포하고 api가 db에 접근할 수 없는 이유 식별
2. 네트워크 구성 수정
3. 모든 서비스가 올바르게 통신할 수 있는지 확인
4. 문제와 해결책 문서화

### 연습 5: 안전한 다중 호스트 네트워크

```bash
# On host1 (manager)
docker swarm init --advertise-addr 192.168.1.10

# Create encrypted overlay network
docker network create \
  --driver overlay \
  --opt encrypted=true \
  --attachable \
  secure-overlay

# Deploy service
docker service create \
  --name web \
  --network secure-overlay \
  --replicas 3 \
  nginx

# On host2 (worker)
docker swarm join --token <token> 192.168.1.10:2377

# Verify service spans both hosts
docker service ps web
```

**작업**:
1. 2노드 Swarm 클러스터 설정
2. 암호화된 오버레이 네트워크 생성
3. 노드 간 서비스 배포
4. 트래픽을 캡처하고 암호화 확인
5. 교차 호스트 컨테이너 통신 테스트

### 연습 6: 네트워크 성능 테스트

```bash
# Create test network
docker network create perf-net

# Run iperf3 server
docker run -d \
  --name iperf-server \
  --network perf-net \
  networkstatic/iperf3 -s

# Run iperf3 client (bridge network)
docker run --rm \
  --network perf-net \
  networkstatic/iperf3 -c iperf-server -t 30

# Run iperf3 client (host network)
docker run --rm \
  --network host \
  networkstatic/iperf3 -c <host-ip> -t 30
```

**작업**:
1. 브리지 네트워크에서 대역폭 측정
2. 호스트 네트워크에서 대역폭 측정
3. 결과 비교 및 오버헤드 문서화
4. 다양한 MTU 크기로 테스트

---

## 요약

이 레슨에서 배운 내용:

- Docker 네트워크 드라이버: bridge, host, overlay, macvlan, none
- 자동 DNS 해석을 지원하는 사용자 정의 브리지 네트워크
- 성능을 위한 호스트 네트워크와 격리를 위한 none 네트워크
- Swarm에서 다중 호스트 통신을 위한 오버레이 네트워크
- 커스텀 네트워크 구성: 서브넷, 게이트웨이, IP 범위, MTU
- 임베디드 DNS 서버를 사용한 DNS 및 서비스 디스커버리
- 고급 포트 매핑 및 퍼블리싱 옵션
- 네트워크 보안: 격리, 내부 네트워크, 암호화
- 네트워크 디버깅을 위한 문제 해결 도구 및 기법

**핵심 요점**:
- 자동 DNS 해석을 위해 항상 사용자 정의 네트워크 사용
- 보안을 위해 여러 네트워크로 서비스 격리
- 다중 호스트 배포를 위해 암호화된 오버레이 네트워크 사용
- 서비스 디스커버리를 위해 임베디드 DNS 활용
- 적절한 도구로 모니터링 및 문제 해결

**다음 단계**:
- 프로덕션 환경을 위한 네트워크 정책 구현
- 고급 네트워킹을 위한 서비스 메시 솔루션 (Istio, Linkerd) 탐색
- Kubernetes용 CNI 플러그인 학습
- 네트워크 성능 최적화 기법 연구

---

## 연습 문제

### 연습 1: Docker 네트워크 드라이버(Network Driver) 탐색

bridge, host, none 네트워크 드라이버의 동작 방식 차이를 관찰합니다.

1. 기본 브리지(bridge) 네트워크에서 컨테이너를 실행하고 IP를 확인합니다: `docker run --rm alpine ip addr`
2. 호스트(host) 네트워킹으로 컨테이너를 실행하고 인터페이스를 비교합니다: `docker run --rm --network host alpine ip addr`
3. 네트워킹 없이 컨테이너를 실행하고 외부 연결이 없음을 확인합니다: `docker run --rm --network none alpine ping -c 1 8.8.8.8`
4. 모든 네트워크를 나열합니다: `docker network ls`
5. 기본 브리지 네트워크를 조사하여 연결된 컨테이너와 서브넷(subnet)을 확인합니다: `docker network inspect bridge`
6. `bridge`, `host`, `none` 드라이버 간의 격리 차이를 설명합니다

### 연습 2: DNS(도메인 네임 시스템) 해석이 가능한 사용자 정의 브리지 네트워크 생성

사용자 정의 네트워크를 사용하여 컨테이너 간 자동 서비스 디스커버리(service discovery)를 활성화합니다.

1. 사용자 정의 브리지 네트워크를 생성합니다: `docker network create --subnet 192.168.100.0/24 mynet`
2. 네트워크에 `server`라는 이름의 컨테이너를 시작합니다: `docker run -d --name server --network mynet nginx:alpine`
3. 동일한 네트워크에서 두 번째 컨테이너를 시작하고 DNS 해석을 테스트합니다: `docker run --rm --network mynet alpine ping -c 3 server`
4. 기본 브리지 네트워크에서 동일한 ping을 시도합니다 — 이름으로 실패해야 합니다: `docker run --rm alpine ping -c 3 server`
5. `server` 컨테이너를 두 번째 네트워크에 연결합니다: `docker network create mynet2 && docker network connect mynet2 server`
6. 컨테이너가 이제 두 네트워크에 인터페이스를 가지는지 확인합니다: `docker inspect server | grep -A 20 Networks`

### 연습 3: Docker Compose 네트워크로 멀티 컨테이너 통신 구현

Compose의 내장 네트워킹을 사용하여 격리된 프론트엔드/백엔드/데이터베이스 계층을 구현합니다.

1. `frontend` (nginx), `backend` (HTTP 서버), `db` (postgres) 세 개의 서비스를 가진 `docker-compose.yml`을 작성합니다
2. `web-tier` (frontend + backend)와 `data-tier` (backend + db) 두 개의 네트워크를 정의합니다
3. 각 서비스를 적절한 네트워크에 할당합니다
4. 스택을 시작합니다: `docker compose up -d`
5. `frontend`에 exec로 접속하여 `backend`에 접근할 수 있는지 확인합니다: `docker compose exec frontend wget -qO- http://backend`
6. `frontend`에서 `db`에 호스트명으로 접근할 수 없는지 확인합니다: `docker compose exec frontend ping db`
7. `backend`에서 `frontend`와 `db` 모두에 접근할 수 있는지 확인합니다

### 연습 4: 네트워크 연결 검사 및 디버깅

진단 도구를 사용하여 끊어진 컨테이너 네트워크를 문제 해결합니다.

1. 별도의 사용자 정의 네트워크에서 두 개의 컨테이너를 시작합니다 (공유 네트워크 없음):
   ```bash
   docker network create net-a
   docker network create net-b
   docker run -d --name container-a --network net-a nginx:alpine
   docker run -d --name container-b --network net-b nginx:alpine
   ```
2. `container-b`에서 `container-a`로 ping을 시도합니다 — 실패하는지 확인합니다
3. `docker inspect`를 사용하여 두 컨테이너의 IP 주소를 찾습니다
4. `docker exec container-b ping <container-a의 IP>`를 시도합니다 — 다른 서브넷이므로 역시 실패하는지 확인합니다
5. `container-b`를 `net-a`에 연결합니다: `docker network connect net-a container-b`
6. 이름과 IP로 ping을 재시도합니다 — 이제 두 방법 모두 성공하는지 확인합니다
7. `docker network inspect net-a`로 두 컨테이너가 네트워크에 나타나는지 확인합니다

### 연습 5: 포트 매핑(Port Mapping)과 호스트 바인딩(Host Binding)

프로토콜 선택과 인터페이스 바인딩을 포함한 세부적인 포트 매핑을 실습합니다.

1. 컨테이너를 실행하고 포트 80을 호스트 포트 8080에 매핑합니다: `docker run -d -p 8080:80 nginx:alpine`
2. 접근 가능한지 확인합니다: `curl http://localhost:8080`
3. 루프백(loopback)에만 바인딩하는 컨테이너를 실행합니다: `docker run -d -p 127.0.0.1:8081:80 nginx:alpine`
4. `127.0.0.1:8081`에서는 접근 가능하지만 `0.0.0.0:8081`에서는 접근 불가한지 확인합니다
5. UDP 서비스를 실행하고 명시적 프로토콜로 포트를 매핑합니다: `docker run -d -p 5353:53/udp some-dns-image` (또는 UDP를 리슨하는 컨테이너 사용)
6. 모든 포트 매핑을 나열합니다: `docker ps --format "table {{.Names}}\t{{.Ports}}"`
7. 모든 테스트 컨테이너를 정리합니다: `docker rm -f $(docker ps -aq)`

---

[이전: 10_CI_CD_Pipelines](./10_CI_CD_Pipelines.md) | [다음: 12_Security_Best_Practices](./12_Security_Best_Practices.md)
