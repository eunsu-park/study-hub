# Container Networking

**Previous**: [CI/CD Pipelines](./10_CI_CD_Pipelines.md) | **Next**: [Security Best Practices](./12_Security_Best_Practices.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain Docker network drivers (bridge, host, overlay, macvlan) and their appropriate use cases
2. Configure custom bridge networks with subnets, gateways, and IP allocation
3. Implement overlay networks for multi-host container communication in Swarm clusters
4. Apply DNS-based service discovery for inter-container communication
5. Configure advanced port mapping strategies including host binding and protocol selection
6. Implement network security with isolation, encryption, and access controls
7. Troubleshoot container network connectivity issues using diagnostic tools

## Table of Contents
1. [Docker Network Drivers](#1-docker-network-drivers)
2. [Bridge Network Deep Dive](#2-bridge-network-deep-dive)
3. [Host and None Networks](#3-host-and-none-networks)
4. [Overlay Networks](#4-overlay-networks)
5. [Network Configuration](#5-network-configuration)
6. [DNS and Service Discovery](#6-dns-and-service-discovery)
7. [Port Mapping Advanced](#7-port-mapping-advanced)
8. [Network Security](#8-network-security)
9. [Troubleshooting](#9-troubleshooting)
10. [Practice Exercises](#10-practice-exercises)

**Difficulty**: ⭐⭐⭐

---

Container networking is one of the most complex yet essential aspects of running Docker in production. Every container needs to communicate -- with other containers, with the host, and with the outside world -- and the networking model you choose directly impacts performance, security, and reliability. This lesson takes a deep dive into Docker's network drivers, DNS-based service discovery, and security configurations, equipping you to design and troubleshoot container networks with confidence.

---

## 1. Docker Network Drivers

Docker provides multiple network drivers for different use cases.

### Network Driver Overview

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

### Listing Networks

```bash
# List all networks
docker network ls

# Output:
# NETWORK ID     NAME      DRIVER    SCOPE
# 3c7f2a8b4d91   bridge    bridge    local
# 9f8e3d2c1a45   host      host      local
# 1b5a6c9d3e72   none      null      local
```

### Inspecting Networks

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

### Network Driver Use Cases

| Driver | Use Case | Scope | DNS |
|--------|----------|-------|-----|
| **bridge** | Single-host, isolated containers | Local | Yes (user-defined) |
| **host** | High performance, no isolation | Local | Host DNS |
| **overlay** | Multi-host, Swarm services | Swarm | Yes |
| **macvlan** | Legacy apps needing MAC addresses | Local | No |
| **none** | Complete isolation, custom networking | Local | No |

---

## 2. Bridge Network Deep Dive

Bridge networks are the most common network type for containers.

### Default Bridge vs User-Defined Bridge

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

### Creating a User-Defined Bridge Network

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

### Connecting Containers to Bridge Networks

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

### Connecting/Disconnecting Networks at Runtime

```bash
# Connect running container to additional network
docker network connect my-app-network my-container

# Disconnect from network
docker network disconnect my-app-network my-container

# Connect with static IP
docker network connect --ip 172.25.0.100 my-app-network my-container
```

### Docker Compose Bridge Network

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

## 3. Host and None Networks

### Host Network

Containers share the host's network stack directly.

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

**Use Cases**:
- High network performance (no NAT overhead)
- Network monitoring tools
- Services needing host network features

**Limitations**:
- No network isolation
- Port conflicts with host
- Can't run multiple containers on same port

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

### Performance Comparison

```bash
# Bridge network (with NAT)
docker run --rm --network bridge alpine ping -c 4 8.8.8.8
# avg RTT: ~0.5ms overhead

# Host network (no NAT)
docker run --rm --network host alpine ping -c 4 8.8.8.8
# avg RTT: native host performance
```

### None Network

Complete network isolation.

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

**Use Cases**:
- Complete network isolation
- Custom network stack implementation
- Testing scenarios

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

## 4. Overlay Networks

Overlay networks enable multi-host container communication in Docker Swarm.

### Overlay Network Architecture

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

### Creating Overlay Networks

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

### Deploying Services on Overlay Network

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

### Docker Compose with Overlay (Swarm Stack)

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

### Overlay Network Encryption

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

## 5. Network Configuration

### Custom Subnet and Gateway

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

### MTU Configuration

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

### IPv6 Support

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

### Network Driver Options

```bash
# Bridge options
docker network create \
  --driver bridge \
  --opt com.docker.network.bridge.name=my-bridge \
  --opt com.docker.network.bridge.enable_icc=true \
  --opt com.docker.network.bridge.enable_ip_masquerade=true \
  my-configured-net
```

**Common Options**:
- `com.docker.network.bridge.name`: Custom bridge interface name
- `com.docker.network.bridge.enable_icc`: Inter-container communication (default: true)
- `com.docker.network.bridge.enable_ip_masquerade`: NAT for external traffic (default: true)
- `com.docker.network.driver.mtu`: MTU size (default: 1500)

---

## 6. DNS and Service Discovery

### Embedded DNS Server

Docker provides automatic DNS resolution for container names.

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

### DNS Resolution Examples

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

### Custom DNS Configuration

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

### Service Discovery in Docker Compose

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

### DNS Round-Robin (Multiple Containers)

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

## 7. Port Mapping Advanced

### Port Publishing Modes

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

### Port Mapping Diagram

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

### Docker Compose Port Mapping

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

### Inspecting Port Mappings

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

## 8. Network Security

### Network Isolation

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

### Internal Networks

```bash
# Create internal network (no external access)
docker network create \
  --internal \
  --subnet 172.40.0.0/16 \
  internal-net

# Containers on this network cannot reach internet
docker run -d --name isolated-db --network internal-net postgres
```

### Inter-Container Communication (ICC)

```bash
# Disable ICC (containers can't talk to each other by default) — forces explicit port publishing for communication, tightening isolation
docker network create \
  --driver bridge \
  --opt com.docker.network.bridge.enable_icc=false \
  restricted-net

# Containers must use port publishing to communicate
```

### Network Policy with iptables

```bash
# View Docker iptables rules
sudo iptables -t nat -L -n -v | grep DOCKER

# Block traffic between specific containers (manual iptables)
sudo iptables -I DOCKER-USER -s 172.17.0.2 -d 172.17.0.3 -j DROP

# Allow only specific ports
sudo iptables -I DOCKER-USER -p tcp --dport 5432 -s 172.17.0.2 -j ACCEPT
sudo iptables -I DOCKER-USER -s 172.17.0.2 -j DROP
```

### Encrypted Overlay Networks

```bash
# All traffic between nodes is encrypted with IPsec
docker network create \
  --driver overlay \
  --opt encrypted=true \
  secure-overlay
```

---

## 9. Troubleshooting

### Network Inspection Commands

```bash
# Inspect network details
docker network inspect my-net

# Show containers on network
docker network inspect my-net --format '{{range .Containers}}{{.Name}} {{end}}'

# Show network configuration of container
docker inspect my-container --format '{{json .NetworkSettings.Networks}}'
```

### Testing Connectivity

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

### Packet Capture

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

### Common Network Issues

#### Issue 1: Container Cannot Reach Internet

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

#### Issue 2: Containers Cannot Communicate by Name

```bash
# Only works on user-defined networks, NOT default bridge
# Solution: Create custom network
docker network create my-net
docker network connect my-net container1
docker network connect my-net container2
```

#### Issue 3: Port Already in Use

```bash
# Find process using port
sudo lsof -i :8080
# or
sudo netstat -tlnp | grep 8080

# Solution: Stop conflicting process or use different port
docker run -d -p 8081:80 nginx
```

#### Issue 4: DNS Resolution Slow

```bash
# Check embedded DNS server
docker exec my-container cat /etc/resolv.conf
# Should see: nameserver 127.0.0.11

# Test DNS performance
docker exec my-container time nslookup container2

# Solution: Add custom DNS servers if needed
docker run --dns 8.8.8.8 --dns 8.8.4.4 my-image
```

### Network Debugging Tools

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

### Viewing Network Logs

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

## 10. Practice Exercises

### Exercise 1: Multi-Tier Application Network

Create a three-tier application with isolated networks.

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

**Tasks**:
1. Deploy the stack
2. Verify nginx can reach app
3. Verify app can reach db
4. Confirm nginx CANNOT reach db directly
5. Confirm db has no internet access

### Exercise 2: Custom Bridge Network with Static IPs

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

**Tasks**:
1. Verify containers have assigned IPs
2. Test connectivity between containers
3. Document the IP allocation scheme

### Exercise 3: DNS Round-Robin Load Balancing

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

**Expected Output**:
```
Backend 1
Backend 2
Backend 3
Backend 1
Backend 2
Backend 3
```

### Exercise 4: Network Troubleshooting

Given a broken network setup, identify and fix issues.

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

**Tasks**:
1. Deploy and identify why api cannot reach db
2. Fix network configuration
3. Verify all services can communicate correctly
4. Document the issue and solution

### Exercise 5: Secure Multi-Host Network

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

**Tasks**:
1. Set up 2-node Swarm cluster
2. Create encrypted overlay network
3. Deploy service across nodes
4. Capture traffic and verify encryption
5. Test cross-host container communication

### Exercise 6: Network Performance Testing

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

**Tasks**:
1. Measure bandwidth on bridge network
2. Measure bandwidth on host network
3. Compare results and document overhead
4. Test with different MTU sizes

---

## Summary

In this lesson, you learned:

- Docker network drivers: bridge, host, overlay, macvlan, none
- User-defined bridge networks with automatic DNS resolution
- Host network for performance and none network for isolation
- Overlay networks for multi-host communication in Swarm
- Custom network configuration: subnets, gateways, IP ranges, MTU
- DNS and service discovery with embedded DNS server
- Advanced port mapping and publishing options
- Network security: isolation, internal networks, encryption
- Troubleshooting tools and techniques for network debugging

**Key Takeaways**:
- Always use user-defined networks for automatic DNS resolution
- Isolate services with multiple networks for security
- Use overlay networks with encryption for multi-host deployments
- Leverage embedded DNS for service discovery
- Monitor and troubleshoot with proper tooling

---

## Exercises

### Exercise 1: Explore Docker Network Drivers

Observe how bridge, host, and none network drivers behave differently.

1. Run a container on the default bridge network and inspect its IP: `docker run --rm alpine ip addr`
2. Run a container with host networking and compare the interfaces: `docker run --rm --network host alpine ip addr`
3. Run a container with no networking and confirm there is no external connectivity: `docker run --rm --network none alpine ping -c 1 8.8.8.8`
4. List all networks: `docker network ls`
5. Inspect the default bridge network to see connected containers and subnet: `docker network inspect bridge`
6. Explain in writing the difference in isolation between `bridge`, `host`, and `none` drivers

### Exercise 2: Create a Custom Bridge Network with DNS Resolution

Use a user-defined network to enable automatic service discovery between containers.

1. Create a custom bridge network: `docker network create --subnet 192.168.100.0/24 mynet`
2. Start a container named `server` on the network: `docker run -d --name server --network mynet nginx:alpine`
3. Start a second container on the same network and test DNS resolution: `docker run --rm --network mynet alpine ping -c 3 server`
4. Now try the same ping using the default bridge network — it should fail by name: `docker run --rm alpine ping -c 3 server`
5. Connect the `server` container to a second network: `docker network create mynet2 && docker network connect mynet2 server`
6. Verify the container now has interfaces on both networks: `docker inspect server | grep -A 20 Networks`

### Exercise 3: Multi-Container Communication with Docker Compose Networks

Use Compose's built-in networking to implement a frontend/backend/database tier with isolation.

1. Write a `docker-compose.yml` with three services: `frontend` (nginx), `backend` (any HTTP server), `db` (postgres)
2. Define two networks: `web-tier` (frontend + backend) and `data-tier` (backend + db)
3. Assign each service to the appropriate networks
4. Start the stack: `docker compose up -d`
5. Exec into `frontend` and confirm it can reach `backend`: `docker compose exec frontend wget -qO- http://backend`
6. Exec into `frontend` and confirm it cannot reach `db` by hostname: `docker compose exec frontend ping db`
7. Exec into `backend` and confirm it can reach both `frontend` and `db`

### Exercise 4: Inspect and Debug Network Connectivity

Use diagnostic tools to troubleshoot a broken container network.

1. Start two containers on separate custom networks (no shared network):
   ```bash
   docker network create net-a
   docker network create net-b
   docker run -d --name container-a --network net-a nginx:alpine
   docker run -d --name container-b --network net-b nginx:alpine
   ```
2. Attempt to ping `container-a` from `container-b` — confirm it fails
3. Use `docker inspect` to find both containers' IP addresses
4. Try `docker exec container-b ping <IP-of-container-a>` — confirm it also fails (different subnets, no routing)
5. Connect `container-b` to `net-a`: `docker network connect net-a container-b`
6. Retry the ping by name and by IP — confirm both now succeed
7. Use `docker network inspect net-a` to confirm both containers appear in the network

### Exercise 5: Port Mapping and Host Binding

Practice fine-grained port mapping including protocol selection and interface binding.

1. Run a container and map port 80 to host port 8080: `docker run -d -p 8080:80 nginx:alpine`
2. Verify it is accessible: `curl http://localhost:8080`
3. Run another container that only binds on loopback: `docker run -d -p 127.0.0.1:8081:80 nginx:alpine`
4. Confirm it is accessible from `127.0.0.1:8081` but not from `0.0.0.0:8081`
5. Run a UDP service and map the port with explicit protocol: `docker run -d -p 5353:53/udp some-dns-image` (or use a container that listens on UDP)
6. List all port mappings: `docker ps --format "table {{.Names}}\t{{.Ports}}"`
7. Clean up all test containers: `docker rm -f $(docker ps -aq)`

---

**Previous**: [CI/CD Pipelines](./10_CI_CD_Pipelines.md) | **Next**: [Security Best Practices](./12_Security_Best_Practices.md)
