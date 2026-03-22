# Container Debugging

**Previous**: [Podman and OCI](./15_Podman_and_OCI.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Use `docker exec` for interactive debugging inside running containers
2. Analyze container logs with `docker logs` and configure log drivers for production
3. Extract container metadata and configuration using `docker inspect`
4. Debug container networking issues with network inspection and diagnostic tools
5. Use `nsenter` to explore container namespaces from the host
6. Apply `strace` and `ltrace` for system call and library call tracing in containers
7. Configure health checks and restart policies for self-healing containers
8. Debug multi-container applications and resolve common container issues

## Table of Contents
1. [Interactive Debugging with docker exec](#1-interactive-debugging-with-docker-exec)
2. [Container Logs and Log Drivers](#2-container-logs-and-log-drivers)
3. [docker inspect for Metadata](#3-docker-inspect-for-metadata)
4. [Debugging Networking Issues](#4-debugging-networking-issues)
5. [Namespace Exploration with nsenter](#5-namespace-exploration-with-nsenter)
6. [System Call Tracing](#6-system-call-tracing)
7. [Health Checks and Restart Policies](#7-health-checks-and-restart-policies)
8. [Debugging Multi-Container Applications](#8-debugging-multi-container-applications)
9. [Common Issues and Solutions](#9-common-issues-and-solutions)
10. [Practice Exercises](#10-practice-exercises)

**Difficulty**: ⭐⭐⭐⭐

---

When containers misbehave, traditional debugging approaches often fall short. You cannot SSH into a container, many images lack debugging tools, and the ephemeral nature of containers means evidence can vanish. This lesson covers a comprehensive toolkit for container debugging -- from simple log inspection to advanced namespace exploration and system call tracing. Mastering these techniques is essential for anyone operating containers in production.

---

## 1. Interactive Debugging with docker exec

### Basic Usage

`docker exec` runs a command inside a running container:

```bash
# Start a shell inside a running container
docker exec -it myapp /bin/bash

# If bash is not available (minimal images)
docker exec -it myapp /bin/sh

# Run a specific command
docker exec myapp cat /etc/hosts

# Run as a specific user
docker exec -u root myapp whoami

# Set environment variables for the exec session
docker exec -e DEBUG=1 myapp python debug_script.py

# Set working directory
docker exec -w /app/logs myapp ls -la
```

### Debugging a Running Application

```bash
# Check running processes
docker exec myapp ps aux

# Check resource usage from inside
docker exec myapp top -bn1

# Inspect file system
docker exec myapp ls -la /app/
docker exec myapp df -h
docker exec myapp du -sh /app/*

# Check network connectivity from inside
docker exec myapp ping -c 3 database
docker exec myapp curl -s http://localhost:8080/health

# Check environment variables
docker exec myapp env

# Read configuration files
docker exec myapp cat /app/config.yaml
```

### Debugging Containers Without a Shell

Some minimal images (scratch, distroless) have no shell. Use a debug container:

```bash
# Method 1: Docker debug (Docker Desktop 4.27+)
docker debug myapp

# Method 2: Copy a static binary into the container
docker cp /usr/bin/busybox myapp:/busybox
docker exec myapp /busybox sh

# Method 3: Use nsenter from the host (see Section 5)

# Method 4: Use ephemeral debug container (Kubernetes)
kubectl debug -it myapp --image=busybox --target=myapp
```

### Attach vs Exec

```bash
# docker attach: Connect to the container's MAIN process (PID 1)
# Warning: Ctrl+C may kill the container
docker attach myapp

# docker exec: Start a NEW process inside the container
# Safe: Exiting the exec session does not affect the container
docker exec -it myapp sh

# Use attach for interactive applications (e.g., REPL)
# Use exec for debugging (safer, independent process)
```

---

## 2. Container Logs and Log Drivers

### Basic Log Access

```bash
# View all logs
docker logs myapp

# Follow logs (like tail -f)
docker logs -f myapp

# Show last N lines
docker logs --tail 100 myapp

# Show logs since a timestamp
docker logs --since 2025-01-15T10:00:00 myapp

# Show logs in the last 30 minutes
docker logs --since 30m myapp

# Show timestamps
docker logs -t myapp

# Combine options
docker logs -f --tail 50 -t myapp
```

### Log Drivers

Docker supports multiple log drivers for production log management:

```
┌──────────────────────────────────────────────────────────────┐
│                     Docker Log Drivers                        │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  json-file (default)                                          │
│  ├─ Logs stored as JSON in /var/lib/docker/containers/        │
│  ├─ Supports docker logs command                              │
│  └─ Configure max size and rotation                           │
│                                                               │
│  journald                                                     │
│  ├─ Logs to systemd journal                                   │
│  ├─ Supports docker logs command                              │
│  └─ Rich metadata and filtering                               │
│                                                               │
│  syslog                                                       │
│  ├─ Logs to syslog daemon                                     │
│  ├─ Does NOT support docker logs                              │
│  └─ Standard Unix logging                                     │
│                                                               │
│  fluentd                                                      │
│  ├─ Logs to Fluentd/Fluent Bit collector                     │
│  ├─ Does NOT support docker logs                              │
│  └─ Best for centralized logging pipelines                    │
│                                                               │
│  awslogs / gcplogs / splunk                                   │
│  ├─ Cloud-native log shipping                                 │
│  └─ Direct integration with cloud platforms                   │
│                                                               │
│  none                                                         │
│  └─ No logging (use for performance-critical containers)      │
└──────────────────────────────────────────────────────────────┘
```

### Configuring Log Drivers

```bash
# Set log driver per container
docker run -d --name myapp \
  --log-driver json-file \
  --log-opt max-size=10m \
  --log-opt max-file=5 \
  myapp:latest

# Set default log driver in daemon.json
```

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3",
    "labels": "production_status",
    "env": "os,customer"
  }
}
```

```yaml
# docker-compose.yml log configuration
version: "3.9"
services:
  app:
    image: myapp
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "5"
```

### Structured Logging Best Practices

```bash
# Application should log to stdout/stderr (12-factor app)
# Docker captures stdout/stderr and routes to the log driver

# Check where logs are stored
docker inspect --format='{{.LogPath}}' myapp

# Check log file size
ls -lh $(docker inspect --format='{{.LogPath}}' myapp)
```

---

## 3. docker inspect for Metadata

### Container Inspection

```bash
# Full JSON output
docker inspect myapp

# Specific fields using Go template
docker inspect --format='{{.State.Status}}' myapp
docker inspect --format='{{.State.StartedAt}}' myapp
docker inspect --format='{{.State.Pid}}' myapp

# Network information
docker inspect --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' myapp

# Port mappings
docker inspect --format='{{json .NetworkSettings.Ports}}' myapp | jq

# Volume mounts
docker inspect --format='{{json .Mounts}}' myapp | jq

# Environment variables
docker inspect --format='{{json .Config.Env}}' myapp | jq

# Restart count
docker inspect --format='{{.RestartCount}}' myapp

# OOM killed status
docker inspect --format='{{.State.OOMKilled}}' myapp
```

### Useful Inspection Patterns

```bash
# Check if container exited with an error
docker inspect --format='{{.State.ExitCode}}' myapp

# Get the container's main command
docker inspect --format='{{json .Config.Cmd}}' myapp

# Check resource limits
docker inspect --format='{{.HostConfig.Memory}}' myapp
docker inspect --format='{{.HostConfig.NanoCpus}}' myapp

# Find the container's log file
docker inspect --format='{{.LogPath}}' myapp

# Get the image used
docker inspect --format='{{.Config.Image}}' myapp

# Check health status
docker inspect --format='{{json .State.Health}}' myapp | jq
```

### Image Inspection

```bash
# Inspect an image
docker inspect nginx:alpine

# Get image layers
docker inspect --format='{{json .RootFS.Layers}}' nginx:alpine | jq

# Get image size
docker inspect --format='{{.Size}}' nginx:alpine

# View image history (how it was built)
docker history nginx:alpine
docker history --no-trunc nginx:alpine
```

### Comparing Containers

```bash
# Compare two container configurations
diff <(docker inspect container1) <(docker inspect container2)

# Compare specific fields
diff \
  <(docker inspect --format='{{json .Config.Env}}' prod | jq -S) \
  <(docker inspect --format='{{json .Config.Env}}' staging | jq -S)
```

---

## 4. Debugging Networking Issues

### Network Inspection

```bash
# List all networks
docker network ls

# Inspect a network
docker network inspect bridge

# Find which containers are on a network
docker network inspect mynet --format='{{range .Containers}}{{.Name}} {{.IPv4Address}}{{end}}'

# Check container's DNS configuration
docker exec myapp cat /etc/resolv.conf

# Check container's host entries
docker exec myapp cat /etc/hosts
```

### Common Network Debugging

```bash
# Test connectivity between containers
docker exec app1 ping -c 3 app2

# Test DNS resolution
docker exec myapp nslookup database
docker exec myapp getent hosts database

# Test port connectivity
docker exec myapp nc -zv database 5432

# Check listening ports inside a container
docker exec myapp ss -tlnp
docker exec myapp netstat -tlnp

# Test HTTP endpoint
docker exec myapp curl -v http://api:8080/health

# Trace network path
docker exec myapp traceroute database
```

### Network Debugging Container

When your application container lacks network tools, use a dedicated debug container on the same network:

```bash
# Run a debug container on the same network
docker run --rm -it --network myapp_default \
  nicolaka/netshoot \
  bash

# Inside netshoot, you have: curl, dig, nslookup, tcpdump,
# iperf, nmap, netstat, ss, ip, and more

# Capture packets
docker run --rm --net container:myapp \
  nicolaka/netshoot \
  tcpdump -i eth0 -w /tmp/capture.pcap
```

### Debugging Port Issues

```bash
# Check published ports
docker port myapp

# Check if port is in use on the host
ss -tlnp | grep 8080

# Verify port mapping
docker inspect --format='{{json .NetworkSettings.Ports}}' myapp | jq

# Common issue: container listens on 127.0.0.1, not 0.0.0.0
docker exec myapp ss -tlnp
# If app binds to 127.0.0.1 inside container, external access fails
# Fix: configure app to bind to 0.0.0.0
```

```
┌──────────────────────────────────────────────────────────────┐
│           Network Debugging Decision Tree                     │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Can't reach container from host?                             │
│  ├─ Check: docker port <container>                            │
│  ├─ Check: ss -tlnp | grep <port> (host)                     │
│  └─ Check: app binds to 0.0.0.0 (not 127.0.0.1)             │
│                                                               │
│  Can't reach container from another container?                │
│  ├─ Check: both on same network?                              │
│  ├─ Check: DNS resolution works?                              │
│  ├─ Check: ping between containers                            │
│  └─ Check: target port is listening                           │
│                                                               │
│  Intermittent connectivity?                                   │
│  ├─ Check: resource limits (OOM?)                             │
│  ├─ Check: health check failures                              │
│  ├─ Check: DNS caching issues                                 │
│  └─ Check: container restarts (docker events)                 │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. Namespace Exploration with nsenter

### Understanding Container Namespaces

Containers use Linux namespaces for isolation. `nsenter` lets you enter a container's namespace from the host:

```
┌──────────────────────────────────────────────────────────────┐
│                  Linux Namespaces                             │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  PID  ─── Process isolation (container sees its own PIDs)     │
│  NET  ─── Network isolation (own interfaces, IPs, routes)     │
│  MNT  ─── Mount isolation (own filesystem view)               │
│  UTS  ─── Hostname isolation                                  │
│  IPC  ─── Inter-process communication isolation               │
│  USER ─── User/group ID isolation (rootless containers)       │
│  CGROUP ─ Cgroup view isolation                               │
└──────────────────────────────────────────────────────────────┘
```

### Using nsenter

```bash
# Get the container's PID on the host
PID=$(docker inspect --format='{{.State.Pid}}' myapp)

# Enter all namespaces of the container
sudo nsenter -t $PID -m -u -i -n -p -- /bin/sh

# Enter only the network namespace
sudo nsenter -t $PID -n -- ip addr

# Enter only the PID namespace
sudo nsenter -t $PID -p -- ps aux

# Enter the mount namespace to see container's filesystem
sudo nsenter -t $PID -m -- ls /app

# Check iptables rules in container's network namespace
sudo nsenter -t $PID -n -- iptables -L -n
```

### Practical nsenter Examples

```bash
# Debug a distroless container (no shell inside)
PID=$(docker inspect --format='{{.State.Pid}}' distroless-app)

# Use host tools in the container's namespace
sudo nsenter -t $PID -n -- ss -tlnp        # Network sockets
sudo nsenter -t $PID -n -- ip route         # Routing table
sudo nsenter -t $PID -m -- cat /app/config  # Read config
sudo nsenter -t $PID -p -- kill -SIGUSR1 1  # Send signal to PID 1

# Compare container and host namespace
echo "Host network:"
ip addr show
echo "Container network:"
sudo nsenter -t $PID -n -- ip addr show
```

### /proc Exploration

```bash
# Container's process info from the host
PID=$(docker inspect --format='{{.State.Pid}}' myapp)

# View container's environment variables
sudo cat /proc/$PID/environ | tr '\0' '\n'

# View container's file descriptors
sudo ls -la /proc/$PID/fd/

# View container's memory maps
sudo cat /proc/$PID/maps

# View container's cgroup limits
sudo cat /proc/$PID/cgroup

# Check container's resource limits
sudo cat /proc/$PID/limits
```

---

## 6. System Call Tracing

### strace in Containers

`strace` traces system calls made by a process, invaluable for debugging permission errors, file access issues, and hanging processes:

```bash
# Install strace in a running container
docker exec myapp apt-get update && docker exec myapp apt-get install -y strace

# Trace all system calls of a process
docker exec myapp strace -p 1

# Trace with timestamps
docker exec myapp strace -tt -p 1

# Trace specific system calls
docker exec myapp strace -e trace=open,read,write -p 1

# Trace network-related calls
docker exec myapp strace -e trace=network -p 1

# Trace file-related calls
docker exec myapp strace -e trace=file -p 1

# Save trace to file
docker exec myapp strace -o /tmp/trace.log -p 1
docker cp myapp:/tmp/trace.log ./trace.log
```

### strace from the Host

```bash
# Trace using nsenter (no strace needed inside container)
PID=$(docker inspect --format='{{.State.Pid}}' myapp)

# Requires SYS_PTRACE capability
docker run --cap-add=SYS_PTRACE ...

# Or trace from host
sudo strace -p $PID -e trace=network

# Trace all processes in the container
sudo strace -p $PID -f -e trace=file
```

### ltrace for Library Calls

```bash
# ltrace traces library calls (e.g., malloc, printf)
docker exec myapp ltrace -p 1

# Trace specific libraries
docker exec myapp ltrace -e malloc+free -p 1
```

### Practical Debugging with strace

```bash
# Debugging "Permission denied" errors
docker exec myapp strace -e trace=open,openat,access -f -p 1
# Look for EACCES or EPERM in the output

# Debugging "Connection refused" errors
docker exec myapp strace -e trace=connect -f -p 1
# Look for ECONNREFUSED

# Debugging slow startup
docker exec myapp strace -T -e trace=file -f -p 1
# -T shows time spent in each syscall

# Debugging "No such file or directory"
docker exec myapp strace -e trace=openat,stat -f -p 1
# Look for ENOENT
```

---

## 7. Health Checks and Restart Policies

### Dockerfile HEALTHCHECK

```dockerfile
FROM nginx:alpine

# Basic health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10s \
  CMD curl -f http://localhost/ || exit 1
```

```dockerfile
FROM python:3.12-slim

# Health check for a Python API
HEALTHCHECK --interval=15s --timeout=5s --retries=3 --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

COPY . /app
CMD ["python", "/app/main.py"]
```

### Health Check Parameters

```
┌──────────────────────────────────────────────────────────────┐
│                Health Check Parameters                        │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  --interval=30s     How often to run the check                │
│  --timeout=5s       Max time for a single check               │
│  --retries=3        Consecutive failures before "unhealthy"   │
│  --start-period=0s  Grace period for container startup        │
│                                                               │
│  Exit codes:                                                  │
│  0 = healthy                                                  │
│  1 = unhealthy                                                │
│  2 = reserved (do not use)                                    │
│                                                               │
│  State transitions:                                           │
│  starting ──(pass)──► healthy ──(fail x retries)──► unhealthy │
│           ──(fail)──► starting (within start-period)          │
└──────────────────────────────────────────────────────────────┘
```

### Monitoring Health Status

```bash
# Check health status
docker inspect --format='{{json .State.Health}}' myapp | jq

# Watch health events
docker events --filter event=health_status

# List containers by health status
docker ps --filter health=unhealthy
docker ps --filter health=healthy

# Health check log
docker inspect --format='{{json .State.Health.Log}}' myapp | jq
```

### Restart Policies

```bash
# No restart (default)
docker run --restart=no myapp

# Always restart
docker run --restart=always myapp

# Restart on failure (with max retry count)
docker run --restart=on-failure:5 myapp

# Restart unless manually stopped
docker run --restart=unless-stopped myapp
```

```
┌──────────────────────────────────────────────────────────────┐
│                   Restart Policies                            │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  no              Never restart automatically                  │
│  always          Always restart (even on manual stop + reboot)│
│  on-failure[:N]  Restart only on non-zero exit code           │
│                  Optional: max N retries                      │
│  unless-stopped  Like always, but not after manual stop       │
│                                                               │
│  Recommendation:                                              │
│  ├─ Development: no (default)                                 │
│  ├─ Production: unless-stopped or on-failure:10               │
│  └─ System services: always                                   │
└──────────────────────────────────────────────────────────────┘
```

### Docker Compose Health Checks

```yaml
# docker-compose.yml
version: "3.9"
services:
  app:
    image: myapp
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 15s
      timeout: 5s
      retries: 3
      start_period: 30s
    restart: unless-stopped
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: secret
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
```

---

## 8. Debugging Multi-Container Applications

### docker compose logs

```bash
# View logs from all services
docker compose logs

# Follow logs from specific services
docker compose logs -f app db

# Show last N lines per service
docker compose logs --tail 50

# Show timestamps
docker compose logs -t
```

### Docker Events

```bash
# Monitor all Docker events in real time
docker events

# Filter by container
docker events --filter container=myapp

# Filter by event type
docker events --filter event=start --filter event=stop --filter event=die

# Filter by time range
docker events --since "2025-01-15T10:00:00" --until "2025-01-15T11:00:00"

# JSON format for parsing
docker events --format '{{json .}}' | jq
```

### Resource Monitoring

```bash
# Live resource usage for all containers
docker stats

# One-shot resource snapshot
docker stats --no-stream

# Format output
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# Check specific containers
docker stats app db redis

# Find containers using excessive resources
docker stats --no-stream --format '{{.Name}}\t{{.CPUPerc}}' | sort -t$'\t' -k2 -rn
```

### Debugging Compose Startup Order

```bash
# Check which services are unhealthy
docker compose ps

# Test depends_on health conditions
docker compose up -d db
docker compose ps db  # Wait for healthy
docker compose up -d app

# Watch the startup sequence
docker compose up 2>&1 | grep -E "(Creating|Started|healthy|unhealthy)"
```

### Inspecting Inter-Service Communication

```bash
# Check Compose network
docker network inspect $(docker compose config --format json | jq -r '.networks | keys[0]')

# DNS resolution within Compose
docker compose exec app nslookup db
docker compose exec app getent hosts db

# Trace inter-service requests
docker compose exec app curl -v http://db:5432
```

---

## 9. Common Issues and Solutions

### Issue 1: Container Exits Immediately

```bash
# Check exit code
docker inspect --format='{{.State.ExitCode}}' myapp
# 0 = normal exit, 137 = killed (OOM or SIGKILL), 1 = application error

# Check logs for errors
docker logs myapp

# Run interactively to see what happens
docker run -it myapp /bin/sh

# Common causes:
# - CMD/ENTRYPOINT runs in background (use exec form)
# - Missing dependencies or config files
# - Permission errors
```

### Issue 2: Container OOM Killed

```bash
# Check if OOM killed
docker inspect --format='{{.State.OOMKilled}}' myapp

# Check memory limit
docker inspect --format='{{.HostConfig.Memory}}' myapp

# Check actual memory usage
docker stats --no-stream myapp

# Solutions:
# - Increase memory limit: docker run -m 2g myapp
# - Tune application memory (JVM heap, Python, etc.)
# - Monitor with docker events --filter event=oom
```

### Issue 3: Permission Denied

```bash
# Check running user
docker exec myapp whoami
docker exec myapp id

# Check file permissions
docker exec myapp ls -la /app/data/

# Check volume mount permissions
docker inspect --format='{{json .Mounts}}' myapp | jq

# Solutions:
# - Match container UID with volume owner
# - Use --user flag: docker run --user 1000:1000 myapp
# - Fix in Dockerfile: RUN chown -R appuser:appuser /app
```

### Issue 4: DNS Resolution Failure

```bash
# Check DNS configuration
docker exec myapp cat /etc/resolv.conf

# Test resolution
docker exec myapp nslookup google.com

# Check Docker DNS settings
docker inspect --format='{{json .HostConfig.Dns}}' myapp

# Solutions:
# - Custom DNS: docker run --dns 8.8.8.8 myapp
# - Check Docker daemon DNS in /etc/docker/daemon.json
# - Ensure containers are on the same network
```

### Issue 5: Slow Container Startup

```bash
# Time the startup
time docker run --rm myapp echo "started"

# Profile with strace
docker run --cap-add=SYS_PTRACE myapp strace -c -f /app/entrypoint.sh

# Common causes and solutions:
# - Large image pull: Use smaller base images
# - Dependency downloads: Cache in image layers
# - Database migration: Use init containers or health checks
# - DNS timeout: Configure DNS servers explicitly
```

### Debugging Checklist

```
┌──────────────────────────────────────────────────────────────┐
│              Container Debugging Checklist                     │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  □ 1. Check container status:     docker ps -a                │
│  □ 2. Check exit code:            docker inspect (ExitCode)   │
│  □ 3. Check logs:                 docker logs --tail 100      │
│  □ 4. Check events:               docker events               │
│  □ 5. Check resource usage:       docker stats                │
│  □ 6. Check OOM:                  docker inspect (OOMKilled)  │
│  □ 7. Check mounts:               docker inspect (Mounts)     │
│  □ 8. Check network:              docker network inspect      │
│  □ 9. Check health:               docker inspect (Health)     │
│  □ 10. Interactive debug:         docker exec -it ... sh      │
│  □ 11. Compare with working:      diff <(inspect A) <(B)     │
│  □ 12. System calls:              strace -p PID               │
└──────────────────────────────────────────────────────────────┘
```

---

## 10. Practice Exercises

### Exercise 1: Log Investigation (Beginner)

Start an nginx container, generate some traffic, and use `docker logs` to analyze the access patterns.

```bash
# 1. Start nginx with JSON file logging and size limits
# 2. Generate 20 HTTP requests using curl
# 3. Use docker logs with --since to find recent entries
# 4. Find the log file path using docker inspect
```

<details>
<summary>Solution</summary>

```bash
# Start nginx
docker run -d --name web \
  --log-opt max-size=5m --log-opt max-file=3 \
  -p 8080:80 nginx:alpine

# Generate traffic
for i in $(seq 1 20); do
  curl -s http://localhost:8080 > /dev/null
  curl -s http://localhost:8080/nonexistent > /dev/null 2>&1
done

# View recent logs
docker logs --since 5m web

# Filter for 404 errors
docker logs web 2>&1 | grep "404"

# Find log file
docker inspect --format='{{.LogPath}}' web

# Cleanup
docker rm -f web
```

</details>

### Exercise 2: Container Inspection (Beginner)

Run a container with specific resource limits and environment variables, then use `docker inspect` to extract all configuration details.

<details>
<summary>Solution</summary>

```bash
# Run with configuration
docker run -d --name inspect-test \
  -m 256m --cpus=0.5 \
  -e APP_ENV=production \
  -e APP_DEBUG=false \
  -p 9090:80 \
  -v testdata:/data \
  --restart=on-failure:3 \
  nginx:alpine

# Inspect various fields
echo "Status: $(docker inspect --format='{{.State.Status}}' inspect-test)"
echo "PID: $(docker inspect --format='{{.State.Pid}}' inspect-test)"
echo "Memory Limit: $(docker inspect --format='{{.HostConfig.Memory}}' inspect-test)"
echo "CPU: $(docker inspect --format='{{.HostConfig.NanoCpus}}' inspect-test)"
echo "Restart Policy: $(docker inspect --format='{{.HostConfig.RestartPolicy.Name}}' inspect-test)"
echo "IP: $(docker inspect --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' inspect-test)"

# Environment variables
docker inspect --format='{{json .Config.Env}}' inspect-test | jq

# Mounts
docker inspect --format='{{json .Mounts}}' inspect-test | jq

# Cleanup
docker rm -f inspect-test
docker volume rm testdata
```

</details>

### Exercise 3: Network Debugging (Intermediate)

Set up a multi-container environment with an app that cannot connect to its database, and debug the issue.

<details>
<summary>Solution</summary>

```bash
# Create two separate networks (simulate misconfiguration)
docker network create frontend
docker network create backend

# Start database on backend network
docker run -d --name db \
  --network backend \
  -e POSTGRES_PASSWORD=secret \
  postgres:16-alpine

# Start app on frontend network (WRONG -- can't reach db)
docker run -d --name app \
  --network frontend \
  alpine sleep 3600

# Debug: Try to reach db from app
docker exec app ping -c 1 db
# ping: bad address 'db' -- DNS resolution fails

# Check networks
docker network inspect frontend --format='{{range .Containers}}{{.Name}} {{end}}'
docker network inspect backend --format='{{range .Containers}}{{.Name}} {{end}}'

# Fix: Connect app to backend network
docker network connect backend app

# Verify fix
docker exec app ping -c 1 db
# Should work now

# Cleanup
docker rm -f app db
docker network rm frontend backend
```

</details>

### Exercise 4: Health Check Debugging (Intermediate)

Create a container with a health check that initially fails, debug why, fix it, and verify the health status transitions.

<details>
<summary>Solution</summary>

```bash
# Create a container with a health check that will fail
docker run -d --name health-test \
  --health-cmd="curl -f http://localhost:80/" \
  --health-interval=5s \
  --health-retries=3 \
  --health-start-period=5s \
  alpine sleep 3600

# Watch health status changes
docker events --filter container=health-test --filter event=health_status &
EVENT_PID=$!

# Wait and check status
sleep 20
docker inspect --format='{{.State.Health.Status}}' health-test
# unhealthy -- because alpine has no curl and no web server

# Check health log for details
docker inspect --format='{{json .State.Health.Log}}' health-test | jq

# Clean up event watcher
kill $EVENT_PID 2>/dev/null

# Fix: Create a proper container with health check
docker rm -f health-test
docker run -d --name health-test \
  --health-cmd="wget -qO- http://localhost:80/ || exit 1" \
  --health-interval=5s \
  --health-retries=3 \
  --health-start-period=10s \
  nginx:alpine

# Wait and verify
sleep 15
docker inspect --format='{{.State.Health.Status}}' health-test
# healthy

# Cleanup
docker rm -f health-test
```

</details>

### Exercise 5: Full Debugging Workflow (Advanced)

A multi-container application (web + API + database) is having issues. One container is crashing, another has connectivity problems. Use all the debugging tools from this lesson to diagnose and fix the issues.

<details>
<summary>Solution</summary>

```yaml
# docker-compose.yml -- Intentionally broken setup
version: "3.9"
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: myapp
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      retries: 5

  api:
    image: python:3.12-slim
    command: >
      sh -c "pip install flask psycopg2-binary &&
             python -c \"
      from flask import Flask
      app = Flask(__name__)
      @app.route('/health')
      def health(): return 'ok'
      app.run(host='0.0.0.0', port=5000)
      \""
    depends_on:
      db:
        condition: service_healthy

  web:
    image: nginx:alpine
    ports:
      - "8080:80"
```

```bash
# Step 1: Start the stack
docker compose up -d

# Step 2: Check status
docker compose ps

# Step 3: Check logs for errors
docker compose logs api
docker compose logs db

# Step 4: Check health
docker inspect --format='{{json .State.Health.Status}}' $(docker compose ps -q db)

# Step 5: Check network connectivity
docker compose exec web ping -c 1 api
docker compose exec api ping -c 1 db

# Step 6: Verify API is listening
docker compose exec api ss -tlnp

# Step 7: Check resource usage
docker stats --no-stream

# Step 8: Check container events
docker events --since 5m --filter label=com.docker.compose.project

# Step 9: Fix any issues found and redeploy
docker compose down
# Edit docker-compose.yml to fix issues
docker compose up -d

# Step 10: Verify everything works
curl http://localhost:8080
docker compose ps
docker compose logs
```

</details>

---

**Previous**: [Podman and OCI](./15_Podman_and_OCI.md)
