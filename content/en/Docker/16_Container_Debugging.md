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

Before the tool reference, read [**Theory & Principles**](#theory--principles) — what `setns()` does under `docker exec`, what `/proc` exposes about a container, the ephemeral debug container pattern in Kubernetes, and how `strace`/`lsof`/`tcpdump` work *inside* a container's namespace.

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

## Theory & Principles

Container debugging is the same Linux process debugging you have always done — `strace`, `lsof`, `tcpdump`, `/proc` inspection — performed *inside* the right namespace. The novelty is the namespace plumbing. The kernel's `setns()` syscall, the `/proc/<pid>/ns/` symlinks, the ephemeral debug container pattern in Kubernetes — these are the bridges between the host's debugging tools and the container's view of the world. Once you can name what each tool does at the namespace level, container debugging stops being mysterious and becomes "the same thing as before, with one extra step."

### A. `docker exec` and the `setns()` Syscall

`docker exec -it <container> sh` does *not* fork a new container. It enters the existing container's namespaces. The mechanism is the kernel's **`setns()`** syscall: given a file descriptor pointing to `/proc/<pid>/ns/<type>`, `setns(fd, 0)` moves the calling process into that namespace.

Each running process exposes its namespaces under `/proc/<pid>/ns/`:

```
$ ls -l /proc/1234/ns/
cgroup -> cgroup:[4026531835]
ipc    -> ipc:[4026531839]
mnt    -> mnt:[4026531840]
net    -> net:[4026532008]
pid    -> pid:[4026532009]
user   -> user:[4026531837]
uts    -> uts:[4026531838]
```

The number in brackets is the namespace inode. Two processes are in the same namespace iff they have the same inode. This is how you check "are these two processes really in the same network namespace?" — `readlink /proc/<pid>/ns/net` for both and compare.

`docker exec` (and `kubectl exec`) work by:

1. Looking up the container's PID 1.
2. Opening each `/proc/<pid1>/ns/<type>` file descriptor.
3. Calling `setns(fd, 0)` for each.
4. Now in the container's namespace world, `execve()`-ing the requested command.

`nsenter` is the standalone CLI version of the same operation: `nsenter -t <pid> -p -m -u -n -i <command>`. Useful when you want to enter only some namespaces (e.g., `-n` for network namespace only — useful for using host's `tcpdump` to sniff a container's traffic).

### B. `/proc/<pid>/` as the Container Inspector

Every Linux process has a directory under `/proc/<pid>/` that exposes its state via files. From the *host* (which sees real PIDs), these tell you almost everything about a container's process without entering it:

| File | What it tells you |
|------|-------------------|
| `/proc/<pid>/status` | UID, GID, capabilities, namespace inodes, parent PID |
| `/proc/<pid>/cmdline` | The argv that was exec'd |
| `/proc/<pid>/environ` | The environment variables (only if you have permission) |
| `/proc/<pid>/cgroup` | Which cgroup the process belongs to (and so its resource limits) |
| `/proc/<pid>/maps` | Memory map: every loaded library, every heap segment |
| `/proc/<pid>/fd/` | File descriptors the process has open (sockets, files, pipes) |
| `/proc/<pid>/root/` | Magic symlink to the container's root filesystem (you can `cat /proc/<pid>/root/etc/passwd` from the host) |
| `/proc/<pid>/cwd` | Current working directory |
| `/proc/<pid>/net/tcp` | TCP connections (in the process's network namespace) |
| `/proc/<pid>/limits` | RLIMIT settings |

The most powerful one is `/proc/<pid>/root/`. From the host, with no shell inside the container required, you can read any file in the container's filesystem. For distroless containers where there is no shell, this is the primary inspection mechanism: `ls /proc/<pid>/root/app/`, `cat /proc/<pid>/root/etc/config`.

`/proc/<pid>/net/tcp` and `/proc/<pid>/net/udp` give you connection state from the *container's* network namespace, even though you are reading from the host.

### C. Logging: Where Stdout/Stderr Actually Go

A container's process writes to file descriptors 1 (stdout) and 2 (stderr). Those FDs are owned not by the process itself but by **the container monitor** (`containerd-shim`, or `conmon` for Podman/CRI-O). The monitor reads them and routes them through a **log driver**:

| Log driver | Where logs go |
|------------|---------------|
| `json-file` (Docker default) | `/var/lib/docker/containers/<id>/<id>-json.log` — JSON lines on disk |
| `journald` | systemd-journald (queryable with `journalctl`) |
| `syslog` | local syslog daemon |
| `fluentd` / `gelf` / `awslogs` / `gcplogs` | streamed to a remote aggregator |
| `none` | discarded |

`docker logs <container>` reads the json-file (or queries journald, or whichever driver) and prints. `kubectl logs` does the equivalent via the kubelet.

Implications:

- **`docker logs` only works if the driver supports it.** json-file and journald do; remote drivers do not.
- **A logging volume that fills up will hang the container.** json-file with no rotation is the classic outage. Set `--log-opt max-size=10m --log-opt max-file=3` (or use journald, which manages its own).
- **Multi-line stack traces are one log entry per line by default.** Use a log shipper that knows how to merge consecutive lines belonging to one event, or have your app emit single-line JSON logs.

### D. Network Debugging: Inside and Outside the Namespace

Networking issues are the most common and most confusing container debugging task. The split:

- **Tools that exist on the host and operate on the container's view via `nsenter`:** `tcpdump`, `iptables`, `ip route`, `ss`, `netstat`. The host has these even if the container doesn't.
- **Tools that need to run inside the container:** `curl` to test what the application can reach, `dig` to test DNS the way the app sees it, `nslookup` for the same.

The standard incantation to capture traffic from a container without installing tcpdump in it:

```bash
# Get the container's PID
PID=$(docker inspect -f '{{.State.Pid}}' mycontainer)
# Run tcpdump in the container's network namespace, from the host
sudo nsenter -t $PID -n tcpdump -i any -w /tmp/cap.pcap
```

For DNS issues, check three places:

1. `/etc/resolv.conf` *inside the container* — what nameservers the resolver will query.
2. The Docker DNS server at `127.0.0.11` (in user-defined bridges) or CoreDNS in K8s — does it know the name?
3. Upstream DNS — can the host even resolve the name?

For "container can't reach the outside world," check iptables MASQUERADE and route tables. For "host can't reach the container," check iptables DNAT and the bridge's `forwarding` setting.

### E. `strace`, `lsof`, and Inspection from Inside

These are the workhorses of Linux process debugging, and they all work inside containers — they just need to be installed (or run via `nsenter` from the host).

- **`strace -p <pid>`** — show every system call the process makes, with arguments and return values. Slow (intercept overhead is significant) but the most precise way to answer "what is the process actually trying to do?". Filter with `-e trace=network` or `-e trace=file` to reduce noise. `strace -f` follows forks for multi-process apps.
- **`lsof -p <pid>`** — list open file descriptors with their paths/sockets/pipes. Find file leaks, find which port the process actually bound to, find what config file is open.
- **`pgrep`, `pkill`, `ps -ef`** — basic process inspection. Inside the container's PID namespace, PIDs start at 1 (the entrypoint).
- **`top`, `htop`** — resource usage from the container's view.
- **`/proc/self/status` and `/proc/self/cgroup`** — confirm what UID, what capabilities, what cgroup the *currently-running shell* has, which is what your debug commands inherit.

When the image is distroless (no shell, no `strace`, no `lsof`), the workflow shifts to debugging from the host using `/proc/<pid>/`, or attaching an **ephemeral debug container** (next section).

### F. Ephemeral Debug Containers: The Modern Workflow

Distroless and scratch images are great for production, terrible for debugging. Modern Kubernetes (1.23+) solves this with **ephemeral containers** (`kubectl debug`):

```bash
kubectl debug -it mypod --image=busybox --target=mycontainer -- sh
```

What happens:

1. The kubelet asks containerd to create a *new* container in the *existing* Pod, sharing the Pod's network and PID namespaces, and the target container's process namespace via `--target`.
2. The new container has its own filesystem (busybox) but can see and signal the target container's processes.
3. You get a shell with `ps`, `cat`, `curl`, etc., even though the target image has none of those.

Docker has the equivalent via `docker run --network=container:other --pid=container:other --volumes-from=other busybox sh` — manually attach a debug container's namespaces to an existing one. The K8s `kubectl debug` is the same idea with Pod awareness.

This pattern is what makes "harden the production image to bare minimum" practical. Distroless or scratch is the *production* image; the *debug* image is busybox or an alpine with whatever tools you need, attached on demand.

### G. Health Checks and the Restart Loop

A container's restart policy turns "the process exited" into "the orchestrator restarts it." For this to work as self-healing rather than as a denial-of-service against your own dependencies:

- **Health check defines "alive."** `HEALTHCHECK CMD curl -f http://localhost/ || exit 1` in the Dockerfile, or `livenessProbe` in K8s. Runs periodically; `start_period` skips the first N seconds (for slow startup); `retries` count consecutive failures before declaring unhealthy.
- **Liveness vs Readiness in K8s.** Liveness failure → restart the container. Readiness failure → remove from Service load balancing but don't restart. A common bug — using a single probe that does too much (e.g., DB query) and tearing down a healthy app because the DB is briefly slow.
- **Backoff matters.** A container that crashes immediately on every start with no backoff would consume the host. Both Docker and K8s implement exponential backoff (`CrashLoopBackOff` in K8s) — restart immediately, then after 10s, 20s, 40s, ... up to 5 minutes. This is why a misconfigured Pod ends up "stuck in CrashLoopBackOff" — not stuck, just being restarted slowly so it doesn't burn the cluster.

When debugging "container keeps restarting," your first questions:

1. What was the *exit code*? `docker inspect -f '{{.State.ExitCode}}' <id>` or `kubectl describe pod`. 0 = clean exit (which the orchestrator restarted because policy is `always`); non-zero = crash; 137 = SIGKILL (likely OOM); 143 = SIGTERM (usually shutdown).
2. What did the *last* container log? `docker logs --previous` or `kubectl logs --previous`. The current container is fresh; the *previous* one's log has the actual death message.
3. What does `dmesg` on the host say? OOM kills appear there with the killed PID and memory totals.

### From Theory to the Tool Reference Below

- **`docker exec`** — `setns()` into the container's namespaces (§A); shell + tools must be in the image.
- **`docker logs`** — read whatever the log driver is recording (§C).
- **`docker inspect`** — dump the container's metadata, including the State.Pid you'll need for `nsenter` (§A, §B).
- **`nsenter -t <pid> -n <cmd>`** — run host tools in the container's network namespace (§D).
- **`strace`, `lsof`, `tcpdump`** — universal Linux process debuggers, used inside the container (§E) or via nsenter from the host (§D).
- **`/proc/<pid>/root/`, `/proc/<pid>/net/tcp`, `/proc/<pid>/fd/`** — host-side inspection of container internals (§B).
- **`HEALTHCHECK`, `--restart`, K8s liveness/readiness probes** — the self-healing loop (§G).
- **`kubectl debug --image=...`** — ephemeral debug containers for distroless production images (§F).

The remaining sections walk these tools with concrete examples. Whenever you are stuck, ask: which namespace is the symptom in (network? mount? PID?), and which tool gives me a view *into* that namespace from where I currently am?

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
