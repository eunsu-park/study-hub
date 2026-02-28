#!/bin/bash
# Exercises for Lesson 11: Container Networking
# Topic: Docker
# Solutions to practice problems from the lesson.

# === Exercise 1: Explore Docker Network Drivers ===
# Problem: Observe how bridge, host, and none drivers behave differently.
exercise_1() {
    echo "=== Exercise 1: Explore Docker Network Drivers ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Run a container on the default bridge network
docker run --rm alpine ip addr
# Output includes:
# 1: lo: <LOOPBACK> ... inet 127.0.0.1/8
# 2: eth0@if7: ... inet 172.17.0.2/16
# The container gets its own network namespace with:
# - A loopback interface (lo)
# - A virtual ethernet interface (eth0) on the 172.17.0.0/16 subnet
# - Traffic is NATed through the docker0 bridge on the host

# Step 2: Run with host networking
docker run --rm --network host alpine ip addr
# Output shows ALL host interfaces:
# 1: lo: ...
# 2: eth0: ... inet 192.168.1.100/24
# 3: docker0: ... inet 172.17.0.1/16
# 4: wlan0: ...
# The container shares the host's network stack — no isolation.
# Ports bind directly on the host (no -p mapping needed).

# Step 3: Run with no networking
docker run --rm --network none alpine ping -c 1 8.8.8.8 2>&1
# ping: sendto: Network unreachable
# Only the loopback interface exists — complete network isolation.

docker run --rm --network none alpine ip addr
# 1: lo: <LOOPBACK> ... inet 127.0.0.1/8
# No eth0 — the container has no external network connectivity.

# Step 4: List all networks
docker network ls
# NETWORK ID     NAME      DRIVER    SCOPE
# abc123...      bridge    bridge    local
# def456...      host      host      local
# ghi789...      none      null      local

# Step 5: Inspect the default bridge network
docker network inspect bridge
# Shows subnet (172.17.0.0/16), gateway (172.17.0.1),
# and any connected containers.

# Step 6: Explanation of isolation differences
#
# bridge: MODERATE isolation
#   - Own IP address and network namespace
#   - Traffic goes through NAT (docker0 bridge)
#   - Containers on the same bridge can communicate
#   - External access requires port mapping (-p)
#
# host: NO isolation
#   - Shares host's network stack entirely
#   - No NAT overhead (best performance)
#   - Port conflicts with host processes
#   - Use case: monitoring tools, high-performance services
#
# none: COMPLETE isolation
#   - Only loopback interface (127.0.0.1)
#   - No external connectivity at all
#   - Use case: security-sensitive computation, custom networking
SOLUTION
}

# === Exercise 2: Custom Bridge Network with DNS Resolution ===
# Problem: Enable automatic service discovery between containers.
exercise_2() {
    echo "=== Exercise 2: Custom Bridge Network with DNS Resolution ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Create a custom bridge network with a specific subnet
docker network create --subnet 192.168.100.0/24 mynet
# Custom subnet: 192.168.100.0/24 (254 usable addresses)
# Docker's embedded DNS server is automatically enabled for user-defined networks

# Step 2: Start a container named 'server' on the network
docker run -d --name server --network mynet nginx:alpine

# Step 3: Test DNS resolution from another container on the same network
docker run --rm --network mynet alpine ping -c 3 server
# PING server (192.168.100.2): 56 data bytes
# 64 bytes from 192.168.100.2: seq=0 ttl=64 time=0.105 ms
# DNS resolves 'server' to its IP address automatically!
# This is the embedded DNS server at 127.0.0.11 inside each container.

# Step 4: Try the same ping on the DEFAULT bridge — it fails
docker run --rm alpine ping -c 1 server 2>&1
# ping: bad address 'server'
# The default bridge does NOT provide DNS resolution.
# Containers can only communicate by IP address on the default bridge.
# This is the #1 reason to ALWAYS use user-defined networks.

# Step 5: Connect the server to a second network
docker network create mynet2
docker network connect mynet2 server
# A container can be on multiple networks simultaneously.
# It gets a separate IP on each network.

# Step 6: Verify dual-network connectivity
docker inspect server --format '{{range $net, $config := .NetworkSettings.Networks}}{{$net}}: {{$config.IPAddress}}
{{end}}'
# mynet: 192.168.100.2
# mynet2: 172.19.0.2
# The container has two interfaces, one per network.
# It can communicate with containers on either network.

# Cleanup
docker rm -f server
docker network rm mynet mynet2
SOLUTION
}

# === Exercise 3: Multi-Container Communication with Docker Compose Networks ===
# Problem: Implement frontend/backend/database tier isolation.
exercise_3() {
    echo "=== Exercise 3: Multi-Container Communication with Compose Networks ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- docker-compose.yml ---"
    cat << 'SOLUTION'
services:
  frontend:
    image: nginx:alpine
    networks:
      - web-tier
    # frontend is ONLY on web-tier
    # It can reach backend (also on web-tier) but NOT db

  backend:
    image: nginx:alpine
    networks:
      - web-tier
      - data-tier
    # backend bridges both networks:
    # - Accepts requests from frontend via web-tier
    # - Queries the database via data-tier
    # This is the gateway pattern for network isolation

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_PASSWORD: secret
    networks:
      - data-tier
    # db is ONLY on data-tier
    # It can only be reached by backend, not by frontend

networks:
  web-tier:
    driver: bridge
  data-tier:
    driver: bridge
    # Two separate bridge networks create an L2 boundary
    # Containers on different networks cannot communicate
    # unless they share a common network
SOLUTION
    echo ""
    echo "--- Verification ---"
    cat << 'SOLUTION'
# Start the stack
docker compose up -d

# Step 5: frontend CAN reach backend
docker compose exec frontend wget -qO- --timeout=2 http://backend
# Returns nginx welcome page — success

# Step 6: frontend CANNOT reach db
docker compose exec frontend ping -c 1 -W 2 db 2>&1
# ping: bad address 'db'
# DNS doesn't even resolve 'db' from the frontend container
# because they are on different networks with no overlap

# Step 7: backend CAN reach both frontend and db
docker compose exec backend ping -c 1 frontend
# PING frontend (172.x.x.x): 56 data bytes — success

docker compose exec backend ping -c 1 db
# PING db (172.y.y.y): 56 data bytes — success

# Network isolation summary:
# ┌──────────┐              ┌──────────┐
# │ frontend │──web-tier──│ backend  │
# └──────────┘              └────┬─────┘
#                                │
#                           data-tier
#                                │
#                           ┌────┴─────┐
#                           │    db    │
#                           └──────────┘
# frontend → backend: OK (web-tier)
# backend → db: OK (data-tier)
# frontend → db: BLOCKED (no shared network)

docker compose down
SOLUTION
}

# === Exercise 4: Inspect and Debug Network Connectivity ===
# Problem: Troubleshoot a broken container network using diagnostic tools.
exercise_4() {
    echo "=== Exercise 4: Inspect and Debug Network Connectivity ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Start two containers on SEPARATE networks
docker network create net-a
docker network create net-b
docker run -d --name container-a --network net-a nginx:alpine
docker run -d --name container-b --network net-b nginx:alpine

# Step 2: Attempt to ping container-a from container-b
docker exec container-b ping -c 1 -W 2 container-a 2>&1
# ping: bad address 'container-a'
# FAILS — they are on different networks, DNS doesn't cross network boundaries

# Step 3: Find both containers' IP addresses
docker inspect container-a --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'
# 172.18.0.2 (on net-a)

docker inspect container-b --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'
# 172.19.0.2 (on net-b)

# Step 4: Try pinging by IP
docker exec container-b ping -c 1 -W 2 172.18.0.2 2>&1
# 1 packets transmitted, 0 packets received, 100% packet loss
# FAILS — different subnets (172.18 vs 172.19), no routing between them
# Docker bridge networks are isolated L2 domains

# Step 5: Fix by connecting container-b to net-a
docker network connect net-a container-b
# container-b now has interfaces on BOTH net-a and net-b

# Step 6: Retry — both name and IP should work now
docker exec container-b ping -c 1 container-a
# PING container-a (172.18.0.2): 56 data bytes
# 64 bytes from 172.18.0.2: seq=0 ttl=64 time=0.123 ms — SUCCESS!

docker exec container-b ping -c 1 172.18.0.2
# 64 bytes from 172.18.0.2: seq=0 ttl=64 time=0.098 ms — SUCCESS!

# Step 7: Confirm both containers appear on net-a
docker network inspect net-a --format '{{range .Containers}}{{.Name}} {{end}}'
# container-a container-b

# Debugging toolkit:
# - docker network inspect: see subnet, gateway, connected containers
# - docker inspect <container>: see IP, MAC, network memberships
# - docker exec ping/nslookup/curl: test connectivity from inside
# - nicolaka/netshoot: Swiss army knife container with every network tool

# Cleanup
docker rm -f container-a container-b
docker network rm net-a net-b
SOLUTION
}

# === Exercise 5: Port Mapping and Host Binding ===
# Problem: Practice fine-grained port mapping including interface binding.
exercise_5() {
    echo "=== Exercise 5: Port Mapping and Host Binding ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Standard port mapping (all interfaces)
docker run -d --name web1 -p 8080:80 nginx:alpine
# Binds on 0.0.0.0:8080 — accessible from any network interface
# External clients can reach this port

# Step 2: Verify accessibility
curl http://localhost:8080
# Returns nginx welcome page

# Step 3: Loopback-only binding (security)
docker run -d --name web2 -p 127.0.0.1:8081:80 nginx:alpine
# Binds ONLY on 127.0.0.1:8081 — accessible only from localhost
# External clients CANNOT reach this port

# Step 4: Verify loopback restriction
curl http://127.0.0.1:8081
# Returns nginx welcome page — works from localhost

# From another machine (or using the host's external IP):
# curl http://<host-ip>:8081
# Connection refused — the port is not bound on external interfaces

# Step 5: UDP port mapping
docker run -d --name dns1 -p 5353:53/udp alpine sleep 3600
# /udp suffix explicitly maps a UDP port
# Default is TCP if not specified
# Use case: DNS servers, game servers, streaming protocols

# Step 6: List all port mappings
docker ps --format "table {{.Names}}\t{{.Ports}}"
# NAMES    PORTS
# web1     0.0.0.0:8080->80/tcp
# web2     127.0.0.1:8081->80/tcp
# dns1     0.0.0.0:5353->53/udp
#
# Notice the difference:
# web1: 0.0.0.0 (all interfaces)
# web2: 127.0.0.1 (loopback only)
# dns1: UDP protocol

# Port mapping best practices:
# 1. Use 127.0.0.1 for admin/debug ports (DB, monitoring)
# 2. Use 0.0.0.0 only for services that need external access
# 3. Document port mappings in docker-compose.yml
# 4. Avoid port conflicts: check 'docker ps' or 'lsof -i :PORT' first
# 5. Use random ports (-P) for testing to avoid conflicts

# Step 7: Cleanup
docker rm -f web1 web2 dns1
SOLUTION
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
echo ""
exercise_5
echo ""
echo "All exercises completed!"
