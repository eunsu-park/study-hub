# 컨테이너 네트워킹

**이전**: [IPv6](./18_IPv6.md) | **다음**: [소프트웨어 정의 네트워킹](./20_Software_Defined_Networking.md)

## 학습 목표(Learning Objectives)

이 레슨을 마치면 다음을 할 수 있습니다:

1. Linux 네트워크 네임스페이스(network namespace)와 가상 네트워킹(virtual networking)의 동작 원리 이해하기
2. Docker 네트워킹 아키텍처(networking architecture)와 네트워크 드라이버(network driver) 마스터하기
3. Kubernetes 네트워킹 모델(networking model)과 CNI 플러그인(CNI plugin) 학습하기
4. 컨테이너의 서비스 디스커버리(service discovery)와 로드 밸런싱(load balancing) 이해하기
5. 보안을 위한 네트워크 정책(NetworkPolicy) 구현하기
6. 서비스 메시(service mesh) 개념과 구현 학습하기
7. 인그레스(Ingress)와 외부 로드 밸런싱(external load balancing) 구성하기
8. 컨테이너 네트워킹 문제 해결하기

---

컨테이너(container)는 소프트웨어 배포 방식을 근본적으로 바꿨지만, 동시에 완전히 새로운 네트워킹 복잡성(networking complexity) 계층을 도입했습니다. 각 컨테이너는 자체 네트워크 네임스페이스(network namespace), 가상 인터페이스(virtual interface), 라우팅 테이블(routing table) 뷰를 가지면서도 서로, 그리고 외부 세계와 원활하게 통신해야 합니다. 이 레슨은 기존 네트워킹 지식을 컨테이너 네이티브 세계와 연결하며, Docker 브리지 네트워크(bridge network)부터 Kubernetes 서비스 메시(service mesh)까지 모든 것을 다룹니다.

## 목차
1. [컨테이너 네트워킹 기초](#1-컨테이너-네트워킹-기초)
2. [Docker 네트워킹 모델](#2-docker-네트워킹-모델)
3. [Docker 네트워크 드라이버](#3-docker-네트워크-드라이버)
4. [Kubernetes 네트워킹 모델](#4-kubernetes-네트워킹-모델)
5. [CNI 플러그인 비교](#5-cni-플러그인-비교)
6. [서비스 디스커버리와 로드 밸런싱](#6-서비스-디스커버리와-로드-밸런싱)
7. [네트워크 정책](#7-네트워크-정책)
8. [서비스 메시](#8-서비스-메시)
9. [인그레스와 로드 밸런싱](#9-인그레스와-로드-밸런싱)
10. [컨테이너 네트워크 문제 해결](#10-컨테이너-네트워크-문제-해결)
11. [연습 문제](#11-연습-문제)

---

## 1. 컨테이너 네트워킹 기초

### 네트워크 네임스페이스(Network Namespaces)

네트워크 네임스페이스는 Linux에서 네트워크 격리를 제공합니다:

```
Default Namespace                  Container Namespace
┌────────────────────┐            ┌────────────────────┐
│  eth0: 10.0.1.10   │            │  eth0: 172.17.0.2  │
│                    │            │  (inside container)│
│  Routing Table     │            │  Routing Table     │
│  Firewall Rules    │            │  Firewall Rules    │
└────────────────────┘            └────────────────────┘
```

**네트워크 네임스페이스 생성:**
```bash
# Create namespace
sudo ip netns add my_namespace

# List namespaces
sudo ip netns list

# Execute command in namespace
sudo ip netns exec my_namespace ip addr show

# Enter namespace shell
sudo ip netns exec my_namespace bash
```

### 가상 이더넷(Virtual Ethernet, veth) 쌍

veth 쌍은 가상 케이블 연결입니다:

```
┌─────────────────┐               ┌─────────────────┐
│   Namespace A   │               │   Namespace B   │
│                 │               │                 │
│   ┌──────────┐  │    veth0 ←───┼───→ veth1       │
│   │  veth0   │──┼───────────────┼────────────┐    │
│   │10.0.0.1  │  │               │   │10.0.0.2│    │
│   └──────────┘  │               │   └────────┘    │
└─────────────────┘               └─────────────────┘
```

**veth 쌍 생성:**
```bash
# Create veth pair
sudo ip link add veth0 type veth peer name veth1

# Move veth1 to namespace
sudo ip link set veth1 netns my_namespace

# Configure veth0 (host side)
sudo ip addr add 10.0.0.1/24 dev veth0
sudo ip link set veth0 up

# Configure veth1 (namespace side)
sudo ip netns exec my_namespace ip addr add 10.0.0.2/24 dev veth1
sudo ip netns exec my_namespace ip link set veth1 up
sudo ip netns exec my_namespace ip link set lo up

# Test connectivity
ping 10.0.0.2
```

### Linux 브리지(Linux Bridge)

브리지는 여러 네트워크 인터페이스를 연결합니다:

```
                   Linux Bridge (br0)
         ┌──────────────┬──────────────┬──────────────┐
         │              │              │              │
      veth0          veth2          veth4          eth0 (host)
         │              │              │              │
         │              │              │              │
    Container 1    Container 2    Container 3     Physical Net
  172.17.0.2/16  172.17.0.3/16  172.17.0.4/16
```

**브리지 생성:**
```bash
# Create bridge
sudo ip link add br0 type bridge
sudo ip link set br0 up
sudo ip addr add 172.17.0.1/16 dev br0

# Create container namespace and veth pair
sudo ip netns add container1
sudo ip link add veth0 type veth peer name veth1

# Connect veth1 to container
sudo ip link set veth1 netns container1

# Connect veth0 to bridge
sudo ip link set veth0 master br0
sudo ip link set veth0 up

# Configure container interface
sudo ip netns exec container1 ip addr add 172.17.0.2/16 dev veth1
sudo ip netns exec container1 ip link set veth1 up
sudo ip netns exec container1 ip link set lo up
sudo ip netns exec container1 ip route add default via 172.17.0.1

# Enable NAT for internet access
sudo iptables -t nat -A POSTROUTING -s 172.17.0.0/16 ! -o br0 -j MASQUERADE
```

### 컨테이너 네트워킹 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                        Host OS                              │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │              Docker Bridge (docker0)                  │ │
│  │                  172.17.0.1/16                        │ │
│  └─────┬──────────────┬──────────────┬──────────────┬────┘ │
│        │              │              │              │      │
│   ┌────▼────┐    ┌───▼─────┐   ┌───▼─────┐   ┌────▼────┐ │
│   │  veth0  │    │  veth2  │   │  veth4  │   │  veth6  │ │
│   └────┬────┘    └────┬────┘   └────┬────┘   └────┬────┘ │
│        │              │              │              │      │
│  ┌─────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐ ┌────▼─────┐
│  │ Container1 │ │ Container2 │ │ Container3 │ │Container4│
│  │ 172.17.0.2 │ │ 172.17.0.3 │ │ 172.17.0.4 │ │172.17.0.5│
│  └────────────┘ └────────────┘ └────────────┘ └──────────┘
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Docker 네트워킹 모델

### 컨테이너 네트워크 모델(Container Network Model, CNM)

Docker는 세 가지 구성요소를 가진 CNM을 사용합니다:

```
┌────────────────────────────────────────────────────────┐
│                   Docker Engine                        │
│                                                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐          │
│  │ Sandbox  │   │ Endpoint │   │ Network  │          │
│  │          │   │          │   │          │          │
│  │ (netns)  │──▶│  (veth)  │──▶│ (bridge) │          │
│  └──────────┘   └──────────┘   └──────────┘          │
│                                                        │
└────────────────────────────────────────────────────────┘

- Sandbox: Container's network stack (namespace)
- Endpoint: Virtual network interface (veth)
- Network: Virtual switch (bridge, overlay, etc.)
```

### libnetwork

Docker의 네트워킹 라이브러리:

```
┌─────────────────────────────────────────────────────┐
│               Docker Engine                         │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│              libnetwork                             │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │  Bridge  │  │  Overlay │  │  Macvlan │  ...    │
│  │  Driver  │  │  Driver  │  │  Driver  │         │
│  └──────────┘  └──────────┘  └──────────┘         │
└─────────────────────────────────────────────────────┘
```

### Docker 네트워크 명령어

```bash
# List networks
docker network ls

# Inspect network
docker network inspect bridge

# Create network
docker network create my_network

# Connect container to network
docker network connect my_network my_container

# Disconnect container
docker network disconnect my_network my_container

# Remove network
docker network rm my_network

# Run container on specific network
docker run --network my_network nginx
```

### 기본 Docker 네트워크

```bash
# List default networks
docker network ls

NETWORK ID     NAME      DRIVER    SCOPE
abcdef123456   bridge    bridge    local
1234567890ab   host      host      local
fedcba098765   none      null      local
```

---

## 3. Docker 네트워크 드라이버

### 브리지 네트워크(Bridge Network)

기본 네트워크 드라이버:

```
┌──────────────────────────────────────────────────────┐
│                    Host                              │
│                                                      │
│  eth0 (10.0.1.10)                                    │
│       │                                              │
│       │  ┌────────────────────────────────────┐     │
│       └──│   docker0 (172.17.0.1/16)          │     │
│          │   (Linux Bridge)                   │     │
│          └────┬──────────────┬─────────────┬──┘     │
│               │              │             │        │
│          ┌────▼────┐    ┌───▼─────┐  ┌────▼────┐   │
│          │ nginx   │    │  redis  │  │  mysql  │   │
│          │.17.0.2  │    │ .17.0.3 │  │ .17.0.4 │   │
│          └─────────┘    └─────────┘  └─────────┘   │
└──────────────────────────────────────────────────────┘
```

**사용자 정의 브리지 생성:**
```bash
# Create custom bridge network
docker network create \
  --driver bridge \
  --subnet 192.168.1.0/24 \
  --gateway 192.168.1.1 \
  my_bridge

# Run containers
docker run -d --name web --network my_bridge nginx
docker run -d --name db --network my_bridge mysql

# Containers can communicate by name
docker exec web ping db
```

**포트 퍼블리싱(Port publishing):**
```bash
# Publish port 80 to host port 8080
docker run -d -p 8080:80 --name web nginx

# iptables NAT rule created:
# DNAT: Host:8080 → Container:80
```

### 호스트 네트워크(Host Network)

컨테이너가 호스트의 네트워크 스택을 공유:

```
┌──────────────────────────────────────────────────────┐
│                    Host                              │
│                                                      │
│  eth0 (10.0.1.10)                                    │
│       │                                              │
│       │  (Same network namespace)                    │
│       │                                              │
│  ┌────▼──────────────────────────────────┐          │
│  │  Container (--network host)           │          │
│  │  Listens on 10.0.1.10:80              │          │
│  └───────────────────────────────────────┘          │
└──────────────────────────────────────────────────────┘
```

**사용법:**
```bash
# Run with host network
docker run --network host nginx

# No port publishing needed
# Container binds directly to host's ports
```

**장점:**
- 최고 성능 (NAT 없음)
- 간단한 구성

**단점:**
- 네트워크 격리 없음
- 포트 충돌 가능

### 오버레이 네트워크(Overlay Network)

멀티 호스트 네트워킹:

```
┌────────────────────────────┐      ┌────────────────────────────┐
│       Host 1 (10.0.1.10)   │      │       Host 2 (10.0.1.11)   │
│                            │      │                            │
│  ┌──────────────────────┐  │      │  ┌──────────────────────┐  │
│  │ Container A          │  │      │  │ Container B          │  │
│  │ 10.0.9.2 (overlay)   │  │      │  │ 10.0.9.3 (overlay)   │  │
│  └──────────┬───────────┘  │      │  └──────────┬───────────┘  │
│             │ VXLAN        │      │             │ VXLAN        │
│  ┌──────────▼───────────┐  │      │  ┌──────────▼───────────┐  │
│  │ br0 (overlay bridge) │  │      │  │ br0 (overlay bridge) │  │
│  └──────────┬───────────┘  │      │  └──────────┬───────────┘  │
│             │              │      │             │              │
│          eth0 ─────────────┼──────┼─────────── eth0            │
│        10.0.1.10           │      │        10.0.1.11           │
└────────────────────────────┘      └────────────────────────────┘
         Encapsulated traffic (VXLAN over UDP 4789)
```

**오버레이 네트워크 생성 (Docker Swarm):**
```bash
# Initialize swarm
docker swarm init

# Create overlay network
docker network create \
  --driver overlay \
  --subnet 10.0.9.0/24 \
  my_overlay

# Deploy service on overlay
docker service create \
  --name web \
  --network my_overlay \
  --replicas 3 \
  nginx
```

### Macvlan 네트워크(Macvlan Network)

컨테이너에 MAC 주소 할당:

```
┌──────────────────────────────────────────────────────┐
│                    Host                              │
│                                                      │
│  eth0 (10.0.1.10) ─── Physical Network              │
│       │                                              │
│       │  (Macvlan in bridge mode)                    │
│       │                                              │
│  ┌────┼──────────────┬─────────────┬──────────────┐ │
│  │    │              │             │              │ │
│  │ ┌──▼────┐    ┌───▼─────┐  ┌────▼────┐         │ │
│  │ │nginx  │    │  redis  │  │  mysql  │         │ │
│  │ │.0.1.20│    │ .0.1.21 │  │ .0.1.22 │         │ │
│  │ └───────┘    └─────────┘  └─────────┘         │ │
│  │ (Appears on physical network with own MAC)    │ │
│  └───────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**macvlan 네트워크 생성:**
```bash
# Create macvlan network
docker network create -d macvlan \
  --subnet=10.0.1.0/24 \
  --gateway=10.0.1.1 \
  -o parent=eth0 \
  my_macvlan

# Run container
docker run -d \
  --network my_macvlan \
  --ip 10.0.1.20 \
  nginx
```

**사용 사례:**
- L2 연결성이 필요한 레거시 애플리케이션
- 물리 네트워크를 모니터링하는 컨테이너
- NAT 없이 직접 네트워크 접근

### 네트워크 드라이버 비교

| 드라이버 | 격리 | 멀티 호스트 | 성능 | 사용 사례 |
|--------|-----------|------------|-------------|----------|
| **Bridge** | 예 | 아니오 | 좋음 | 단일 호스트, 개발 |
| **Host** | 아니오 | 아니오 | 뛰어남 | 성능 중요 |
| **Overlay** | 예 | 예 | 좋음 | 멀티 호스트, Swarm/K8s |
| **Macvlan** | 예 | 아니오 | 뛰어남 | L2 연결성 필요 |
| **None** | 완전 | N/A | N/A | 네트워킹 불필요 |

---

## 4. Kubernetes 네트워킹 모델

### Kubernetes 네트워크 요구사항

Kubernetes는 다음 요구사항을 부과합니다:

1. **모든 Pod는 NAT 없이 서로 통신** 가능
2. **모든 노드는 NAT 없이 모든 Pod와 통신** 가능
3. **Pod는 다른 Pod가 보는 것과 동일한 자신의 IP** 확인 (NAT 없음)

```
┌────────────────────────────────────────────────────────────┐
│                      Cluster Network                       │
│                                                            │
│  ┌──────────────────┐              ┌──────────────────┐   │
│  │    Node 1        │              │    Node 2        │   │
│  │  IP: 10.0.1.10   │              │  IP: 10.0.1.11   │   │
│  │                  │              │                  │   │
│  │  ┌────────────┐  │              │  ┌────────────┐  │   │
│  │  │  Pod A     │  │              │  │  Pod C     │  │   │
│  │  │ 10.244.1.2 │──┼──────────────┼──│ 10.244.2.2 │  │   │
│  │  └────────────┘  │              │  └────────────┘  │   │
│  │                  │              │                  │   │
│  │  ┌────────────┐  │              │  ┌────────────┐  │   │
│  │  │  Pod B     │  │              │  │  Pod D     │  │   │
│  │  │ 10.244.1.3 │  │              │  │ 10.244.2.3 │  │   │
│  │  └────────────┘  │              │  └────────────┘  │   │
│  └──────────────────┘              └──────────────────┘   │
│                                                            │
│  Pod A can directly communicate with Pod C (10.244.2.2)   │
└────────────────────────────────────────────────────────────┘
```

### Pod 네트워킹

각 Pod는 자체 IP를 받습니다:

```
┌─────────────────────────────────────────────────┐
│                Pod (10.244.1.5)                 │
│                                                 │
│  ┌──────────────┐  ┌──────────────┐            │
│  │ Container A  │  │ Container B  │            │
│  │ localhost:80 │  │localhost:3306│            │
│  └──────┬───────┘  └──────┬───────┘            │
│         │                 │                    │
│         └────────┬────────┘                    │
│                  │                             │
│          ┌───────▼────────┐                    │
│          │  Network NS    │                    │
│          │  eth0          │                    │
│          │  10.244.1.5    │                    │
│          └────────────────┘                    │
└─────────────────────────────────────────────────┘
```

### 컨테이너 네트워크 인터페이스(Container Network Interface, CNI)

CNI는 네트워크 플러그인을 위한 표준 인터페이스입니다:

```
┌──────────────────────────────────────────────────┐
│              Kubernetes (kubelet)                │
└──────────────────┬───────────────────────────────┘
                   │
                   │ CNI Specification
                   │
┌──────────────────▼───────────────────────────────┐
│                CNI Plugin                        │
│  (Calico, Cilium, Flannel, Weave, etc.)         │
│                                                  │
│  Responsibilities:                               │
│  - Allocate IP to pod                            │
│  - Setup network interface                       │
│  - Configure routing                             │
└──────────────────────────────────────────────────┘
```

**CNI 구성 예제:**
```json
{
  "cniVersion": "0.4.0",
  "name": "k8s-pod-network",
  "type": "calico",
  "log_level": "info",
  "datastore_type": "kubernetes",
  "ipam": {
    "type": "calico-ipam"
  },
  "policy": {
    "type": "k8s"
  },
  "kubernetes": {
    "kubeconfig": "/etc/cni/net.d/calico-kubeconfig"
  }
}
```

---

## 5. CNI 플러그인 비교

### Calico

**아키텍처:**
```
┌────────────────────────────────────────────────────┐
│              Calico Components                     │
│                                                    │
│  ┌──────────────┐   ┌──────────────┐             │
│  │   Felix      │   │    BIRD      │             │
│  │ (Agent)      │   │ (BGP daemon) │             │
│  │ - Routes     │   │ - Route      │             │
│  │ - ACLs       │   │   exchange   │             │
│  └──────────────┘   └──────────────┘             │
│                                                    │
│  ┌──────────────────────────────────┐             │
│  │       etcd / Kubernetes API      │             │
│  │       (Datastore)                │             │
│  └──────────────────────────────────┘             │
└────────────────────────────────────────────────────┘
```

**기능:**
- 순수 L3 네트워킹 (오버레이 없음)
- BGP 경로 배포
- 확장 가능 (1000+ 노드 테스트)
- 풍부한 네트워크 정책

**네트워크 모드:**
- IP-in-IP (캡슐화)
- VXLAN
- Direct/Native (캡슐화 없음)

### Cilium

**아키텍처:**
```
┌────────────────────────────────────────────────────┐
│              Cilium Components                     │
│                                                    │
│  ┌──────────────────────────────────┐             │
│  │     eBPF Programs (kernel)       │             │
│  │  - Packet filtering              │             │
│  │  - Load balancing                │             │
│  │  - Network policy                │             │
│  └────────────┬─────────────────────┘             │
│               │                                    │
│  ┌────────────▼─────────────────────┐             │
│  │      Cilium Agent                │             │
│  │  - Identity management           │             │
│  │  - Policy enforcement            │             │
│  └──────────────────────────────────┘             │
└────────────────────────────────────────────────────┘
```

**기능:**
- eBPF 기반 (Linux 커널 기술)
- L7 프로토콜 가시성 (HTTP, gRPC, Kafka)
- ID 기반 보안
- Hubble 관찰성

**사용 사례:**
- 고급 보안 정책
- 사이드카 없는 서비스 메시
- API 인식 필터링

### Flannel

**아키텍처:**
```
┌────────────────────────────────────────────────────┐
│              Flannel Components                    │
│                                                    │
│  ┌──────────────────────────────────┐             │
│  │     flanneld (agent)             │             │
│  │  - Allocate subnet               │             │
│  │  - Configure VXLAN/host-gw       │             │
│  └────────────┬─────────────────────┘             │
│               │                                    │
│  ┌────────────▼─────────────────────┐             │
│  │      etcd / Kubernetes API       │             │
│  │      (Subnet allocation)         │             │
│  └──────────────────────────────────┘             │
└────────────────────────────────────────────────────┘
```

**기능:**
- 간단한 오버레이 네트워크
- 여러 백엔드 (VXLAN, host-gw, UDP)
- 배포 용이
- 네트워크 정책 지원 없음

**백엔드 비교:**
- VXLAN: L3를 넘어 작동, 약간의 오버헤드
- host-gw: L2 필요, 더 나은 성능
- UDP: 레거시, 성능 저하

### Weave

**아키텍처:**
```
┌────────────────────────────────────────────────────┐
│              Weave Components                      │
│                                                    │
│  ┌──────────────────────────────────┐             │
│  │     Weave Router (per node)      │             │
│  │  - Overlay network               │             │
│  │  - Mesh topology                 │             │
│  │  - Encryption (optional)         │             │
│  └──────────────────────────────────┘             │
│                                                    │
│  Automatic mesh formation between nodes            │
└────────────────────────────────────────────────────┘
```

**기능:**
- 메시 네트워크 토폴로지
- 내장 암호화
- 멀티캐스트 지원
- 네트워크 정책 지원

### CNI 플러그인 비교

| 플러그인 | 기술 | 성능 | 기능 | 복잡도 |
|--------|-----------|-------------|----------|------------|
| **Calico** | BGP/eBPF | 뛰어남 | 풍부한 정책, 확장 가능 | 중간 |
| **Cilium** | eBPF | 뛰어남 | L7 정책, 관찰성 | 높음 |
| **Flannel** | VXLAN/host-gw | 좋음 | 간단, 신뢰성 | 낮음 |
| **Weave** | Mesh/VXLAN | 좋음 | 암호화, 멀티캐스트 | 중간 |

**선택 기준:**
- **간단한 오버레이**: Flannel
- **네트워크 정책 + 확장성**: Calico
- **L7 가시성**: Cilium
- **암호화**: Weave
- **온프레미스, L2**: Calico (BGP)

---

## 6. 서비스 디스커버리와 로드 밸런싱

### Kubernetes 서비스(Kubernetes Services)

서비스는 Pod에 안정적인 엔드포인트를 제공합니다:

```
┌────────────────────────────────────────────────────┐
│              Service (ClusterIP)                   │
│              my-service: 10.96.0.10:80             │
└─────────────────────┬──────────────────────────────┘
                      │ (Load balances to endpoints)
         ┌────────────┼────────────┐
         │            │            │
    ┌────▼────┐  ┌───▼─────┐  ┌──▼──────┐
    │  Pod A  │  │  Pod B  │  │  Pod C  │
    │10.244.1.2│ │10.244.1.3│ │10.244.2.2│
    └─────────┘  └─────────┘  └─────────┘
```

**서비스 유형:**

1. **ClusterIP (기본)**: 내부 클러스터 IP
2. **NodePort**: 각 노드 IP의 정적 포트에 노출
3. **LoadBalancer**: 외부 로드 밸런서 (클라우드 제공자)
4. **ExternalName**: 외부 서비스에 대한 CNAME

**서비스 정의:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  type: ClusterIP
  selector:
    app: nginx
  ports:
    - protocol: TCP
      port: 80        # Service port
      targetPort: 80  # Container port
```

### kube-proxy

kube-proxy는 서비스 로드 밸런싱을 구현합니다:

```
┌────────────────────────────────────────────────────┐
│                  Node                              │
│                                                    │
│  ┌─────────────────────────────────┐              │
│  │        kube-proxy                │              │
│  │  Watches Service/Endpoint API   │              │
│  └────────────┬────────────────────┘              │
│               │                                    │
│  ┌────────────▼────────────────────┐              │
│  │   Packet forwarding rules       │              │
│  │   (iptables / IPVS / eBPF)      │              │
│  └─────────────────────────────────┘              │
└────────────────────────────────────────────────────┘
```

### iptables 모드

iptables NAT를 사용하는 기본 모드:

```bash
# Example iptables rules for service 10.96.0.10:80
# (Load balance to 3 backends)

# KUBE-SERVICES chain
-A KUBE-SERVICES -d 10.96.0.10/32 -p tcp -m tcp --dport 80 \
   -j KUBE-SVC-XYZ

# Service chain (probabilistic load balancing)
-A KUBE-SVC-XYZ -m statistic --mode random --probability 0.33333 \
   -j KUBE-SEP-AAA
-A KUBE-SVC-XYZ -m statistic --mode random --probability 0.50000 \
   -j KUBE-SEP-BBB
-A KUBE-SVC-XYZ -j KUBE-SEP-CCC

# Endpoint chains (DNAT to pod IPs)
-A KUBE-SEP-AAA -p tcp -m tcp \
   -j DNAT --to-destination 10.244.1.2:80
-A KUBE-SEP-BBB -p tcp -m tcp \
   -j DNAT --to-destination 10.244.1.3:80
-A KUBE-SEP-CCC -p tcp -m tcp \
   -j DNAT --to-destination 10.244.2.2:80
```

**장점:**
- 성숙하고 테스트됨
- 커널 수준 성능

**단점:**
- O(n) 규칙 처리
- 1000+ 서비스에서 성능 저하

### IPVS 모드

더 나은 성능을 위한 IP Virtual Server:

```
┌────────────────────────────────────────────────────┐
│              IPVS Virtual Server                   │
│        10.96.0.10:80 (rr scheduling)               │
└─────────────────────┬──────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         │            │            │
    10.244.1.2    10.244.1.3   10.244.2.2
    (weight 1)    (weight 1)   (weight 1)
```

**장점:**
- O(1) 룩업 복잡도
- 대규모에서 더 나은 성능
- 여러 스케줄링 알고리즘 (rr, lc, dh, sh 등)

**IPVS 모드 활성화:**
```yaml
# kube-proxy config
apiVersion: kubeproxy.config.k8s.io/v1alpha1
kind: KubeProxyConfiguration
mode: "ipvs"
ipvs:
  scheduler: "rr"  # round-robin
```

### CoreDNS

DNS 기반 서비스 디스커버리:

```
┌────────────────────────────────────────────────────┐
│              CoreDNS                               │
│  Watches Services, creates DNS records             │
└────────────────────────────────────────────────────┘

DNS Records:
- my-service.default.svc.cluster.local → 10.96.0.10
- my-service.default.svc               → 10.96.0.10
- my-service.default                   → 10.96.0.10
- my-service                           → 10.96.0.10
  (from default namespace)

# From pod:
nslookup my-service
# Returns: 10.96.0.10
```

**CoreDNS 구성:**
```
# Corefile
.:53 {
    errors
    health
    kubernetes cluster.local in-addr.arpa ip6.arpa {
      pods insecure
      fallthrough in-addr.arpa ip6.arpa
    }
    prometheus :9153
    forward . /etc/resolv.conf
    cache 30
    loop
    reload
    loadbalance
}
```

---

## 7. 네트워크 정책

### Kubernetes NetworkPolicy

Pod 간 트래픽 제어:

```
Default: All traffic allowed
┌─────────┐     ┌─────────┐     ┌─────────┐
│  Pod A  │────▶│  Pod B  │────▶│  Pod C  │
└─────────┘     └─────────┘     └─────────┘

With NetworkPolicy:
┌─────────┐     ┌─────────┐     ┌─────────┐
│  Pod A  │  ✓  │  Pod B  │  ✗  │  Pod C  │
└─────────┘────▶└─────────┘- - -▶└─────────┘
```

**기본 NetworkPolicy:**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-from-frontend
  namespace: default
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: frontend
      ports:
        - protocol: TCP
          port: 8080
```

**효과:**
- `app=frontend` 레이블을 가진 Pod만 포트 8080에서 백엔드 접근 가능
- 백엔드로의 다른 모든 트래픽 거부

### 인그레스와 이그레스 규칙(Ingress and Egress Rules)

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: db-network-policy
spec:
  podSelector:
    matchLabels:
      app: database
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow from backend pods on port 3306
    - from:
        - podSelector:
            matchLabels:
              app: backend
      ports:
        - protocol: TCP
          port: 3306
  egress:
    # Allow DNS
    - to:
        - namespaceSelector:
            matchLabels:
              name: kube-system
      ports:
        - protocol: UDP
          port: 53
    # Allow external backup server
    - to:
        - ipBlock:
            cidr: 10.0.5.0/24
      ports:
        - protocol: TCP
          port: 22
```

### 네임스페이스 격리(Namespace Isolation)

```yaml
# Deny all traffic to pods in namespace
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
# No ingress/egress rules = deny all
```

```yaml
# Allow only from same namespace
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-same-namespace
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector: {}  # Same namespace
```

### Calico NetworkPolicy

Kubernetes NetworkPolicy보다 더 고급:

```yaml
apiVersion: projectcalico.org/v3
kind: GlobalNetworkPolicy
metadata:
  name: deny-egress-to-metadata-server
spec:
  selector: all()
  types:
    - Egress
  egress:
    # Deny access to cloud metadata service
    - action: Deny
      destination:
        nets:
          - 169.254.169.254/32
    # Allow all other egress
    - action: Allow
```

**Calico 기능:**
- 글로벌 정책
- 정책 순서
- Layer 7 규칙 (Istio 사용)
- 로깅/모니터링

---

## 8. 서비스 메시

### 서비스 메시 아키텍처

```
┌────────────────────────────────────────────────────┐
│              Control Plane                         │
│  (Istio Pilot, Citadel, Galley, Telemetry)        │
└──────────────────┬─────────────────────────────────┘
                   │ Configuration
         ┌─────────┼─────────┐
         │         │         │
    ┌────▼────┐ ┌─▼──────┐ ┌▼─────────┐
    │ Envoy   │ │ Envoy  │ │ Envoy    │
    │ Proxy   │ │ Proxy  │ │ Proxy    │
    ├─────────┤ ├────────┤ ├──────────┤
    │ App A   │ │ App B  │ │ App C    │
    └─────────┘ └────────┘ └──────────┘
```

### Istio 사이드카 패턴(Istio Sidecar Pattern)

```
┌───────────────────────────────────────────────────┐
│                 Pod                               │
│                                                   │
│  ┌──────────────────┐   ┌──────────────────┐     │
│  │  Application     │   │  Envoy Proxy     │     │
│  │  Container       │◀─▶│  (Sidecar)       │     │
│  │  localhost:8080  │   │  - mTLS          │     │
│  └──────────────────┘   │  - Metrics       │     │
│                         │  - Tracing       │     │
│                         │  - Retries       │     │
│                         └─────────┬────────┘     │
└───────────────────────────────────┼──────────────┘
                                    │
                              Encrypted traffic
```

### 상호 TLS(Mutual TLS, mTLS)

서비스 간 암호화:

```
Service A                          Service B
┌─────────────┐                   ┌─────────────┐
│  App        │                   │  App        │
│  │          │                   │  │          │
│  ▼          │                   │  ▼          │
│ Envoy       │    mTLS tunnel    │ Envoy       │
│  │          │◀─────────────────▶│  │          │
│  │          │  (cert exchange)  │  │          │
└──┼──────────┘                   └──┼──────────┘
   │                                 │
   └─────────────────────────────────┘
     Encrypted, authenticated traffic
```

**Istio PeerAuthentication:**
```yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: default
spec:
  mtls:
    mode: STRICT  # STRICT, PERMISSIVE, or DISABLE
```

### 트래픽 관리(Traffic Management)

**Virtual Service (라우팅 규칙):**
```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: reviews
spec:
  hosts:
    - reviews
  http:
    # 90% to v1, 10% to v2 (canary deployment)
    - match:
        - headers:
            end-user:
              exact: jason
      route:
        - destination:
            host: reviews
            subset: v2
    - route:
        - destination:
            host: reviews
            subset: v1
          weight: 90
        - destination:
            host: reviews
            subset: v2
          weight: 10
```

**Destination Rule (로드 밸런싱, 서킷 브레이킹):**
```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: reviews
spec:
  host: reviews
  trafficPolicy:
    loadBalancer:
      simple: LEAST_REQUEST
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        maxRequestsPerConnection: 2
    outlierDetection:
      consecutiveErrors: 5
      interval: 30s
      baseEjectionTime: 30s
  subsets:
    - name: v1
      labels:
        version: v1
    - name: v2
      labels:
        version: v2
```

### 서비스 메시 비교

| 기능 | Istio | Linkerd | Envoy |
|---------|-------|---------|-------|
| **언어** | Go/C++ | Rust | C++ |
| **복잡도** | 높음 | 낮음 | 중간 |
| **리소스 사용** | 무거움 | 가벼움 | 중간 |
| **기능** | 광범위 | 보통 | 프록시만 |
| **mTLS** | 예 | 예 | 예 |
| **관찰성** | 광범위 | 좋음 | 기본 |

---

## 9. 인그레스와 로드 밸런싱

### 인그레스 컨트롤러(Ingress Controller)

서비스로의 HTTP/HTTPS 라우팅:

```
                    Internet
                       │
                 ┌─────▼─────┐
                 │  LoadBalancer │
                 │  (Cloud LB)   │
                 └─────┬─────┘
                       │
         ┌─────────────┼─────────────┐
         │   Ingress Controller      │
         │   (nginx/traefik/haproxy) │
         └─────────────┬─────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    ┌────▼────┐   ┌───▼─────┐  ┌───▼─────┐
    │Service A│   │Service B│  │Service C│
    └─────────┘   └─────────┘  └─────────┘
```

**인그레스 리소스:**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
    - host: example.com
      http:
        paths:
          - path: /app1
            pathType: Prefix
            backend:
              service:
                name: app1-service
                port:
                  number: 80
          - path: /app2
            pathType: Prefix
            backend:
              service:
                name: app2-service
                port:
                  number: 80
  tls:
    - hosts:
        - example.com
      secretName: tls-secret
```

### Gateway API

차세대 인그레스:

```yaml
apiVersion: gateway.networking.k8s.io/v1beta1
kind: Gateway
metadata:
  name: my-gateway
spec:
  gatewayClassName: nginx
  listeners:
    - name: http
      protocol: HTTP
      port: 80
    - name: https
      protocol: HTTPS
      port: 443
      tls:
        certificateRefs:
          - name: tls-cert
---
apiVersion: gateway.networking.k8s.io/v1beta1
kind: HTTPRoute
metadata:
  name: my-route
spec:
  parentRefs:
    - name: my-gateway
  hostnames:
    - example.com
  rules:
    - matches:
        - path:
            type: PathPrefix
            value: /app1
      backendRefs:
        - name: app1-service
          port: 80
```

### 외부 로드 밸런서(External Load Balancer)

클라우드 제공자 통합:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  type: LoadBalancer
  selector:
    app: nginx
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
  # Cloud-specific annotations
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
```

**결과:**
- 클라우드 제공자가 로드 밸런서 프로비저닝
- 서비스에 외부 IP 할당
- NodePort로 트래픽 라우팅

---

## 10. 컨테이너 네트워크 문제 해결

### Docker 네트워킹 문제 해결

**컨테이너 네트워크 확인:**
```bash
# Inspect container network
docker inspect <container> | jq '.[0].NetworkSettings'

# Check IP address
docker inspect -f '{{.NetworkSettings.IPAddress}}' <container>

# List connected networks
docker inspect -f '{{range $net, $conf := .NetworkSettings.Networks}}{{$net}} {{end}}' <container>

# Enter container network namespace
docker exec -it <container> bash
ip addr show
ip route show
```

**연결성 테스트:**
```bash
# Ping from one container to another
docker exec container1 ping container2

# Check DNS resolution
docker exec container1 nslookup container2

# Check port connectivity
docker exec container1 nc -zv container2 80
```

**일반적인 문제:**

1. **컨테이너 간 통신 불가:**
   - 동일 네트워크에 있는지 확인
   - 방화벽 규칙 확인
   - DNS 해석 확인

2. **컨테이너가 인터넷에 접근 불가:**
   - NAT/masquerade 규칙 확인
   - 기본 경로 확인
   - DNS 구성 확인

3. **포트 퍼블리싱 작동 안 함:**
   - iptables DNAT 규칙 확인
   - 호스트 방화벽 확인
   - 포트가 이미 사용 중이 아닌지 확인

### Kubernetes 네트워킹 문제 해결

**Pod 연결성:**
```bash
# Check pod IP and network
kubectl get pod <pod> -o wide

# Describe pod (check events)
kubectl describe pod <pod>

# Check pod network interfaces
kubectl exec <pod> -- ip addr show

# Test pod-to-pod connectivity
kubectl exec <pod1> -- ping <pod2-ip>

# Test service connectivity
kubectl exec <pod> -- curl http://my-service
```

**서비스 문제 해결:**
```bash
# Check service endpoints
kubectl get endpoints my-service

# Verify service DNS
kubectl exec <pod> -- nslookup my-service

# Check kube-proxy logs
kubectl logs -n kube-system -l k8s-app=kube-proxy

# Check iptables rules (iptables mode)
kubectl exec -n kube-system <kube-proxy-pod> -- iptables-save | grep <service-name>
```

**CNI 문제 해결:**
```bash
# Check CNI plugin pods
kubectl get pods -n kube-system | grep calico
kubectl get pods -n kube-system | grep cilium

# Check CNI logs
kubectl logs -n kube-system <cni-pod>

# Verify CNI configuration
cat /etc/cni/net.d/*.conf
```

**NetworkPolicy 디버깅:**
```bash
# Check if policy applied
kubectl get networkpolicy

# Describe policy
kubectl describe networkpolicy <policy>

# Test connectivity before/after policy
kubectl exec <pod> -- curl <target>

# Check CNI plugin logs (policy enforcement)
kubectl logs -n kube-system <calico-node-pod>
```

### 네트워크 진단 도구

**컨테이너 디버깅 이미지:**
```bash
# Run debug container with network tools
kubectl run debug --rm -it --image=nicolaka/netshoot -- bash

# Tools included:
# - tcpdump, wireshark
# - curl, wget, httpie
# - nslookup, dig, host
# - netcat, socat
# - iperf3, mtr, traceroute
```

**임시 디버그 컨테이너 (K8s 1.23+):**
```bash
# Attach debug container to existing pod
kubectl debug -it <pod> --image=nicolaka/netshoot --target=<container>
```

**패킷 캡처:**
```bash
# Capture traffic on pod interface
kubectl exec <pod> -- tcpdump -i eth0 -w /tmp/capture.pcap

# Copy to local machine
kubectl cp <pod>:/tmp/capture.pcap ./capture.pcap

# Analyze with Wireshark
wireshark capture.pcap
```

---

## 11. 연습 문제

### 문제 1: Docker 사용자 정의 네트워크
다음 요구사항으로 사용자 정의 Docker 네트워크를 생성하세요:
- 서브넷: 172.20.0.0/16
- 게이트웨이: 172.20.0.1
- 3개 컨테이너 실행: web, api, db
- web은 api와, api는 db와 연결
- 연결성 테스트

**정답:**
```bash
# Create network
docker network create \
  --driver bridge \
  --subnet 172.20.0.0/16 \
  --gateway 172.20.0.1 \
  mynetwork

# Run containers
docker run -d --name db --network mynetwork postgres
docker run -d --name api --network mynetwork my-api
docker run -d --name web --network mynetwork nginx

# Test connectivity
docker exec web ping api
docker exec api ping db
docker exec web curl http://api:8080/health
```

### 문제 2: Kubernetes 서비스
다음을 배포하세요:
- 3개의 nginx Pod
- ClusterIP 서비스
- 로드 밸런싱 확인

**정답:**
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
        - name: nginx
          image: nginx
          ports:
            - containerPort: 80
---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: web-service
spec:
  selector:
    app: web
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
```

```bash
# Apply
kubectl apply -f deployment.yaml

# Check endpoints
kubectl get endpoints web-service

# Test load balancing
kubectl run test --rm -it --image=busybox -- sh
# In pod:
for i in $(seq 1 10); do
  wget -qO- http://web-service | grep 'Server:'
done
# Should see different pod IPs
```

### 문제 3: NetworkPolicy
보안 정책 구현:
- 프론트엔드가 포트 8080에서 백엔드 접근 허용
- 백엔드가 포트 5432에서 데이터베이스 접근 허용
- 다른 모든 트래픽 거부

**정답:**
```yaml
# backend-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-policy
spec:
  podSelector:
    matchLabels:
      tier: backend
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              tier: frontend
      ports:
        - protocol: TCP
          port: 8080
  egress:
    # Allow DNS
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: UDP
          port: 53
    # Allow database
    - to:
        - podSelector:
            matchLabels:
              tier: database
      ports:
        - protocol: TCP
          port: 5432
---
# database-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: database-policy
spec:
  podSelector:
    matchLabels:
      tier: database
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              tier: backend
      ports:
        - protocol: TCP
          port: 5432
```

### 문제 4: 네트워크 문제 디버깅
시나리오: Pod가 외부 서비스(example.com)에 접근 불가

**문제 해결 단계:**
```bash
# 1. Check pod IP and interface
kubectl exec <pod> -- ip addr show
kubectl exec <pod> -- ip route show

# 2. Check DNS resolution
kubectl exec <pod> -- nslookup example.com
# If fails, check DNS pods
kubectl get pods -n kube-system -l k8s-app=kube-dns

# 3. Test external connectivity
kubectl exec <pod> -- ping 8.8.8.8
# If fails, check network policy
kubectl get networkpolicy

# 4. Check egress policy
kubectl describe networkpolicy <policy>
# Ensure egress to 0.0.0.0/0 is allowed

# 5. Check NAT/masquerade
# On node:
sudo iptables -t nat -L POSTROUTING -n -v
# Should see MASQUERADE rule for pod CIDR

# 6. Verify CNI configuration
cat /etc/cni/net.d/*.conf
# Check if CNI supports egress

# 7. Check node routing
ip route show
# Should have route for pod CIDR
```

### 문제 5: 서비스 메시 트래픽 분할
카나리 배포 구현:
- v1로 90% 트래픽
- v2로 10% 트래픽
- Istio 사용

**정답:**
```yaml
# virtual-service.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: my-app
spec:
  hosts:
    - my-app
  http:
    - route:
        - destination:
            host: my-app
            subset: v1
          weight: 90
        - destination:
            host: my-app
            subset: v2
          weight: 10
---
# destination-rule.yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: my-app
spec:
  host: my-app
  subsets:
    - name: v1
      labels:
        version: v1
    - name: v2
      labels:
        version: v2
```

```bash
# Apply configuration
kubectl apply -f virtual-service.yaml

# Generate traffic and observe distribution
for i in $(seq 1 100); do
  curl http://my-app/version
done | sort | uniq -c
# Should see approximately 90 v1, 10 v2 responses
```

---

## 요약

컨테이너 네트워킹은 복잡하지만 일관된 원칙을 따릅니다:

**핵심 개념:**
1. **네트워크 네임스페이스**는 격리 제공
2. **veth 쌍과 브리지**는 컨테이너 연결
3. **Docker CNM**은 표준화된 네트워킹 정의
4. **Kubernetes 네트워킹 모델**은 플랫 네트워크 필요
5. **CNI 플러그인**은 다양한 접근 방식 구현 (오버레이, BGP, eBPF)
6. **서비스**는 안정적인 엔드포인트와 로드 밸런싱 제공
7. **NetworkPolicy**는 트래픽 흐름 제어
8. **서비스 메시**는 L7 기능 추가 (mTLS, 관찰성)
9. **인그레스**는 HTTP/HTTPS 라우팅 제공

**모범 사례:**
- 요구사항에 따라 CNI 선택 (정책, 성능, 확장성)
- 보안을 위해 NetworkPolicy 사용
- 네트워크 성능 모니터링
- 복잡한 마이크로서비스를 위한 서비스 메시 구현
- IP 주소 할당 신중하게 계획

컨테이너 네트워킹은 eBPF(Cilium) 및 Gateway API와 같은 기술과 함께 계속 발전하여 더 강력하고 관리하기 쉬워지고 있습니다.

---

**난이도:** ⭐⭐⭐⭐

**추가 읽을거리:**
- Kubernetes Network Model: https://kubernetes.io/docs/concepts/cluster-administration/networking/
- CNI Specification: https://github.com/containernetworking/cni
- Istio Documentation: https://istio.io/latest/docs/
- Calico Documentation: https://docs.projectcalico.org/

---

**이전**: [IPv6](./18_IPv6.md) | **다음**: [소프트웨어 정의 네트워킹](./20_Software_Defined_Networking.md)
