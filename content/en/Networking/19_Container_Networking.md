# Container Networking

**Previous**: [IPv6](./18_IPv6.md)

## Learning Objectives
- Understand Linux network namespaces and virtual networking
- Master Docker networking architecture and network drivers
- Learn Kubernetes networking model and CNI plugins
- Understand service discovery and load balancing in containers
- Implement network policies for security
- Learn service mesh concepts and implementations
- Configure ingress and external load balancing
- Troubleshoot container networking issues

---

Containers have fundamentally changed how we deploy software, but they have also introduced an entirely new layer of networking complexity. Each container gets its own network namespace, its own virtual interfaces, and its own view of the routing table -- yet all must communicate seamlessly with each other and the outside world. This lesson bridges traditional networking knowledge with the container-native world, covering everything from Docker bridge networks to Kubernetes service meshes.

## Table of Contents
1. [Container Networking Fundamentals](#1-container-networking-fundamentals)
2. [Docker Networking Model](#2-docker-networking-model)
3. [Docker Network Drivers](#3-docker-network-drivers)
4. [Kubernetes Networking Model](#4-kubernetes-networking-model)
5. [CNI Plugins Comparison](#5-cni-plugins-comparison)
6. [Service Discovery and Load Balancing](#6-service-discovery-and-load-balancing)
7. [Network Policies](#7-network-policies)
8. [Service Mesh](#8-service-mesh)
9. [Ingress and Load Balancing](#9-ingress-and-load-balancing)
10. [Troubleshooting Container Networks](#10-troubleshooting-container-networks)
11. [Practice Problems](#11-practice-problems)

---

## 1. Container Networking Fundamentals

### Network Namespaces

Network namespaces provide network isolation in Linux:

```
Default Namespace                  Container Namespace
┌────────────────────┐            ┌────────────────────┐
│  eth0: 10.0.1.10   │            │  eth0: 172.17.0.2  │
│                    │            │  (inside container)│
│  Routing Table     │            │  Routing Table     │
│  Firewall Rules    │            │  Firewall Rules    │
└────────────────────┘            └────────────────────┘
```

**Creating a network namespace:**
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

### Virtual Ethernet (veth) Pairs

veth pairs are virtual cable connections:

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

**Creating veth pair:**
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

### Linux Bridge

Bridge connects multiple network interfaces:

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

**Creating a bridge:**
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

### Container Networking Architecture

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

## 2. Docker Networking Model

### Container Network Model (CNM)

Docker uses CNM with three components:

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

Docker's networking library:

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

### Docker Network Commands

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

### Default Docker Networks

```bash
# List default networks
docker network ls

NETWORK ID     NAME      DRIVER    SCOPE
abcdef123456   bridge    bridge    local
1234567890ab   host      host      local
fedcba098765   none      null      local
```

---

## 3. Docker Network Drivers

### Bridge Network

Default network driver:

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

**Creating a custom bridge:**
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

**Port publishing:**
```bash
# Publish port 80 to host port 8080
docker run -d -p 8080:80 --name web nginx

# iptables NAT rule created:
# DNAT: Host:8080 → Container:80
```

### Host Network

Container shares host's network stack:

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

**Usage:**
```bash
# Run with host network
docker run --network host nginx

# No port publishing needed
# Container binds directly to host's ports
```

**Pros:**
- Best performance (no NAT)
- Simple configuration

**Cons:**
- No network isolation
- Port conflicts possible

### Overlay Network

Multi-host networking:

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

**Creating overlay network (Docker Swarm):**
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

### Macvlan Network

Assign MAC addresses to containers:

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

**Creating macvlan network:**
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

**Use cases:**
- Legacy applications requiring L2 connectivity
- Container monitoring physical network
- Direct network access without NAT

### Network Driver Comparison

| Driver | Isolation | Multi-host | Performance | Use Case |
|--------|-----------|------------|-------------|----------|
| **Bridge** | Yes | No | Good | Single host, development |
| **Host** | No | No | Excellent | Performance-critical |
| **Overlay** | Yes | Yes | Good | Multi-host, Swarm/K8s |
| **Macvlan** | Yes | No | Excellent | L2 connectivity needed |
| **None** | Complete | N/A | N/A | No networking required |

---

## 4. Kubernetes Networking Model

### Kubernetes Network Requirements

Kubernetes imposes these requirements:

1. **All pods can communicate with each other** without NAT
2. **All nodes can communicate with all pods** without NAT
3. **Pod sees its own IP** as others see it (no NAT)

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

### Pod Networking

Each pod gets its own IP:

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

### Container Network Interface (CNI)

CNI is the standard interface for network plugins:

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

**CNI configuration example:**
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

## 5. CNI Plugins Comparison

### Calico

**Architecture:**
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

**Features:**
- Pure L3 networking (no overlay)
- BGP route distribution
- Scalable (tested with 1000+ nodes)
- Rich network policy

**Network modes:**
- IP-in-IP (encapsulation)
- VXLAN
- Direct/Native (no encapsulation)

### Cilium

**Architecture:**
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

**Features:**
- eBPF-based (Linux kernel technology)
- L7 protocol visibility (HTTP, gRPC, Kafka)
- Identity-based security
- Hubble observability

**Use cases:**
- Advanced security policies
- Service mesh without sidecars
- API-aware filtering

### Flannel

**Architecture:**
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

**Features:**
- Simple overlay network
- Multiple backends (VXLAN, host-gw, UDP)
- Easy to deploy
- No network policy support

**Backend comparison:**
- VXLAN: Works across L3, some overhead
- host-gw: L2 required, better performance
- UDP: Legacy, poor performance

### Weave

**Architecture:**
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

**Features:**
- Mesh network topology
- Built-in encryption
- Multicast support
- Network policy support

### CNI Plugin Comparison

| Plugin | Technology | Performance | Features | Complexity |
|--------|-----------|-------------|----------|------------|
| **Calico** | BGP/eBPF | Excellent | Rich policy, scalable | Medium |
| **Cilium** | eBPF | Excellent | L7 policy, observability | High |
| **Flannel** | VXLAN/host-gw | Good | Simple, reliable | Low |
| **Weave** | Mesh/VXLAN | Good | Encryption, multicast | Medium |

**Selection criteria:**
- **Simple overlay**: Flannel
- **Network policy + scale**: Calico
- **L7 visibility**: Cilium
- **Encryption**: Weave
- **On-premises, L2**: Calico (BGP)

---

## 6. Service Discovery and Load Balancing

### Kubernetes Services

Services provide stable endpoints for pods:

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

**Service types:**

1. **ClusterIP (default)**: Internal cluster IP
2. **NodePort**: Exposes on each node's IP at static port
3. **LoadBalancer**: External load balancer (cloud provider)
4. **ExternalName**: CNAME to external service

**Service definition:**
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

kube-proxy implements service load balancing:

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

### iptables Mode

Default mode using iptables NAT:

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

**Pros:**
- Mature, well-tested
- Kernel-level performance

**Cons:**
- O(n) rule processing
- Poor performance with 1000+ services

### IPVS Mode

IP Virtual Server for better performance:

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

**Advantages:**
- O(1) lookup complexity
- Better performance at scale
- Multiple scheduling algorithms (rr, lc, dh, sh, etc.)

**Enable IPVS mode:**
```yaml
# kube-proxy config
apiVersion: kubeproxy.config.k8s.io/v1alpha1
kind: KubeProxyConfiguration
mode: "ipvs"
ipvs:
  scheduler: "rr"  # round-robin
```

### CoreDNS

DNS-based service discovery:

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

**CoreDNS configuration:**
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

## 7. Network Policies

### Kubernetes NetworkPolicy

Control traffic between pods:

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

**Basic NetworkPolicy:**
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

**Effect:**
- Only pods with label `app=frontend` can access backend on port 8080
- All other traffic to backend is denied

### Ingress and Egress Rules

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

### Namespace Isolation

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

More advanced than Kubernetes NetworkPolicy:

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

**Calico features:**
- Global policies
- Policy ordering
- Layer 7 rules (with Istio)
- Logging/monitoring

---

## 8. Service Mesh

### Service Mesh Architecture

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

### Istio Sidecar Pattern

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

### Mutual TLS (mTLS)

Service-to-service encryption:

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

### Traffic Management

**Virtual Service (routing rules):**
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

**Destination Rule (load balancing, circuit breaking):**
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

### Service Mesh Comparison

| Feature | Istio | Linkerd | Envoy |
|---------|-------|---------|-------|
| **Language** | Go/C++ | Rust | C++ |
| **Complexity** | High | Low | Medium |
| **Resource usage** | Heavy | Light | Medium |
| **Features** | Extensive | Moderate | Proxy only |
| **mTLS** | Yes | Yes | Yes |
| **Observability** | Extensive | Good | Basic |

---

## 9. Ingress and Load Balancing

### Ingress Controller

HTTP/HTTPS routing to services:

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

**Ingress resource:**
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

Next-generation Ingress:

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

### External Load Balancer

Cloud provider integration:

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

**Result:**
- Cloud provider provisions load balancer
- External IP assigned to service
- Traffic routed to NodePorts

---

## 10. Troubleshooting Container Networks

### Docker Networking Troubleshooting

**Check container network:**
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

**Test connectivity:**
```bash
# Ping from one container to another
docker exec container1 ping container2

# Check DNS resolution
docker exec container1 nslookup container2

# Check port connectivity
docker exec container1 nc -zv container2 80
```

**Common issues:**

1. **Containers can't reach each other:**
   - Check if on same network
   - Check firewall rules
   - Verify DNS resolution

2. **Container can't reach internet:**
   - Check NAT/masquerade rules
   - Verify default route
   - Check DNS configuration

3. **Port publishing not working:**
   - Verify iptables DNAT rules
   - Check host firewall
   - Confirm port not already in use

### Kubernetes Networking Troubleshooting

**Pod connectivity:**
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

**Service troubleshooting:**
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

**CNI troubleshooting:**
```bash
# Check CNI plugin pods
kubectl get pods -n kube-system | grep calico
kubectl get pods -n kube-system | grep cilium

# Check CNI logs
kubectl logs -n kube-system <cni-pod>

# Verify CNI configuration
cat /etc/cni/net.d/*.conf
```

**NetworkPolicy debugging:**
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

### Network Diagnostic Tools

**Container debugging image:**
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

**Ephemeral debug container (K8s 1.23+):**
```bash
# Attach debug container to existing pod
kubectl debug -it <pod> --image=nicolaka/netshoot --target=<container>
```

**Packet capture:**
```bash
# Capture traffic on pod interface
kubectl exec <pod> -- tcpdump -i eth0 -w /tmp/capture.pcap

# Copy to local machine
kubectl cp <pod>:/tmp/capture.pcap ./capture.pcap

# Analyze with Wireshark
wireshark capture.pcap
```

---

## 11. Practice Problems

### Problem 1: Docker Custom Network
Create a custom Docker network with:
- Subnet: 172.20.0.0/16
- Gateway: 172.20.0.1
- Run 3 containers: web, api, db
- web should connect to api, api to db
- Test connectivity

**Solution:**
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

### Problem 2: Kubernetes Service
Deploy a web application with:
- 3 nginx pods
- ClusterIP service
- Verify load balancing

**Solution:**
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

### Problem 3: NetworkPolicy
Implement security policy:
- Allow frontend to access backend on port 8080
- Allow backend to access database on port 5432
- Deny all other traffic

**Solution:**
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

### Problem 4: Debugging Network Issue
Scenario: Pod can't reach external service (example.com)

**Troubleshooting steps:**
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

### Problem 5: Service Mesh Traffic Splitting
Implement canary deployment:
- 90% traffic to v1
- 10% traffic to v2
- Use Istio

**Solution:**
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

## Summary

Container networking is complex but follows consistent principles:

**Key concepts:**
1. **Network namespaces** provide isolation
2. **veth pairs and bridges** connect containers
3. **Docker CNM** defines standardized networking
4. **Kubernetes networking model** requires flat network
5. **CNI plugins** implement different approaches (overlay, BGP, eBPF)
6. **Services** provide stable endpoints and load balancing
7. **NetworkPolicies** control traffic flow
8. **Service meshes** add L7 features (mTLS, observability)
9. **Ingress** provides HTTP/HTTPS routing

**Best practices:**
- Choose CNI based on requirements (policy, performance, scale)
- Use NetworkPolicies for security
- Monitor network performance
- Implement service mesh for complex microservices
- Plan IP address allocation carefully

Container networking continues to evolve with technologies like eBPF (Cilium) and Gateway API, making it more powerful and easier to manage.

---

**Difficulty:** ⭐⭐⭐⭐

**Further Reading:**
- Kubernetes Network Model: https://kubernetes.io/docs/concepts/cluster-administration/networking/
- CNI Specification: https://github.com/containernetworking/cni
- Istio Documentation: https://istio.io/latest/docs/
- Calico Documentation: https://docs.projectcalico.org/

---

**Previous**: [IPv6](./18_IPv6.md)
