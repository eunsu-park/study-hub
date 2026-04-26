# Docker Basics

**Next**: [Docker Images and Containers](./02_Images_and_Containers.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what Docker is and why it solves the "works on my machine" problem
2. Distinguish between containers and virtual machines in terms of architecture and resource usage
3. Describe the core concepts of Docker: images, containers, and Docker Hub
4. Install Docker on macOS, Windows, or Linux
5. Verify a Docker installation by running test containers
6. Describe the Docker workflow from CLI command to running container
7. Run basic containers with port mapping and common options

---

Before the CLI walkthrough, read [**Theory & Principles**](#theory--principles) — the Linux kernel mechanisms (namespaces, cgroups, union filesystems) that make a container a container, and the dockerd / containerd / runc split that turns a `docker run` into a running process.

Before Docker, deploying software was notoriously fragile: an application that worked perfectly on one machine would fail mysteriously on another due to different library versions, OS configurations, or missing dependencies. Docker eliminates this entire class of problems by packaging applications together with their complete runtime environment into lightweight, portable containers. Understanding Docker is now a foundational skill for developers, DevOps engineers, and anyone involved in modern software delivery.

> **Analogy -- Shipping Container:** Before standardized shipping containers, every port needed different equipment to handle different cargo shapes. Docker does the same for software: it packages an application with all its dependencies into a standardized container that runs identically on any machine -- your laptop, a test server, or a production cluster.

---

## Theory & Principles

A container is not a new kind of virtual machine. It is just a process — or a small group of processes — running on the host kernel with three Linux features layered on top:

- **(A) Namespaces** isolate what the process can *see*: PIDs, mounts, network interfaces, hostnames, IPC objects, users.
- **(B) cgroups (control groups)** limit what the process can *consume*: CPU time, memory, block I/O, PIDs.
- **(C) A union filesystem** gives every container its own root filesystem assembled from read-only image layers plus a writable top layer, without copying gigabytes of files.

Once you internalize these three pieces, the rest of Docker is a packaging format (images), a daemon (dockerd / containerd) that wires them together, and a CLI that hides the system calls behind friendly verbs.

### A. Containers vs Virtual Machines: Two Different Isolation Models

A virtual machine emulates hardware. The hypervisor (KVM, Xen, VMware ESXi, Hyper-V) presents a virtual CPU, virtual memory, and virtual devices to a *guest kernel*, which boots its own OS and runs its own processes. Strong isolation, but each VM carries the cost of a full kernel and userspace — typically hundreds of megabytes of RAM and tens of seconds to boot, even before the application starts.

A container does not emulate hardware. It runs on the *host kernel* and uses kernel features to make a normal process believe it is alone on the machine. There is no guest OS, no virtual hardware, no second kernel. Startup is the cost of `fork() + exec()` plus a few namespace-setup syscalls — milliseconds. Memory overhead is the resident set of the process itself plus a few MB of bookkeeping.

The trade-off is the trust boundary. A VM escape requires breaking the hypervisor; a container escape requires breaking the host kernel. So containers traditionally co-locate workloads from the same trust domain (one team, one application stack), while multi-tenant clouds still wrap containers in lightweight VMs (Firecracker, Kata Containers) when the trust boundary matters more than the millisecond startup.

### B. Linux Namespaces: Isolating What a Process Sees

A namespace is a kernel-level "view" of one type of system resource. Two processes in different namespaces of the same type see different resources, even though they share the same kernel. The seven namespace types relevant to containers:

| Namespace | What it isolates | Key syscall flag |
|-----------|------------------|------------------|
| `PID` | Process IDs (containerized PID 1, no view of host processes) | `CLONE_NEWPID` |
| `NET` | Network interfaces, routing tables, iptables rules, ports | `CLONE_NEWNET` |
| `MNT` | Mount points (containerized root filesystem view) | `CLONE_NEWNS` |
| `UTS` | Hostname and domain name | `CLONE_NEWUTS` |
| `IPC` | System V IPC, POSIX message queues | `CLONE_NEWIPC` |
| `USER` | User and group IDs (root inside ≠ root outside) | `CLONE_NEWUSER` |
| `CGROUP` | View of the cgroup hierarchy | `CLONE_NEWCGROUP` |

Namespaces are created with three system calls:

- `clone(flags, ...)` — fork a child and place it in new namespaces in one step. This is how containers are created.
- `unshare(flags)` — detach the *current* process from one or more shared namespaces. The CLI tool of the same name lets you experiment from a shell.
- `setns(fd, ...)` — join an existing namespace by file descriptor. This is how `docker exec` enters a running container's namespaces.

Inside a PID namespace, the first process gets PID 1 — the same number as `init` on a normal system — and the kernel hides every PID outside the namespace. The container truly cannot see the host's processes; the isolation is enforced at the syscall layer, not by filtering output.

The USER namespace is the youngest and most powerful. It maps a range of UIDs/GIDs in the namespace to a different range outside, so a process can be UID 0 (root) inside the container but UID 100000 (an unprivileged user) on the host. This is the foundation of *rootless containers*.

### C. cgroups: Limiting What a Process Can Consume

Namespaces hide resources. cgroups *meter* them. A cgroup is a node in a hierarchical tree where each node has a set of attached processes and a set of resource limits, enforced by kernel "controllers":

- `cpu` — CPU shares (proportional weight) and quotas (hard cap, e.g. "use at most 1.5 cores").
- `memory` — RSS limit (OOM-kill on overrun), swap limit, kernel memory.
- `io` (formerly `blkio`) — block-device read/write bandwidth and IOPS.
- `pids` — maximum number of processes (defends against fork bombs).
- `cpuset` — pin processes to specific CPUs and NUMA memory nodes.

Two versions coexist in the wild:

- **cgroup v1** has separate hierarchies per controller — a process can sit in different positions in the `cpu` tree and the `memory` tree. Flexible but operationally confusing; mixed configurations are easy to break.
- **cgroup v2** unifies all controllers into a single hierarchy. Simpler model, better resource accounting (especially for memory pressure), and required by modern features like rootless cgroup delegation. All recent kernels (5.x+) and distros default to v2.

When you run `docker run --memory=512m --cpus=1.5 myimage`, Docker creates a cgroup, writes `512M` to its `memory.max` file and the appropriate values to its CPU files, then `clone()`s the container process into that cgroup. The kernel handles the rest — your process gets OOM-killed at 512 MB whether it likes it or not.

### D. Union Filesystems: Layered Storage Without Duplication

A container needs a root filesystem (`/bin`, `/etc`, `/lib`, ...) but copying the whole tree per container would defeat the lightweight model. Union filesystems solve this by *stacking* directories: multiple read-only layers below, one writable layer on top, presented as a single merged view.

Modern Docker uses **OverlayFS** (`overlay2` storage driver). It takes three directory inputs and one output:

- `lowerdir` — one or more read-only layers (the image layers, stacked).
- `upperdir` — the single writable layer where new and modified files live.
- `workdir` — internal scratch space the kernel uses for atomic operations.
- `merged` — the unified view the container sees as `/`.

When the container reads a file, the kernel walks layers top-down and returns the first match. When the container *writes* a file that exists only in a lower layer, OverlayFS performs **copy-on-write**: it copies the file up into the `upperdir` and then modifies the copy. When the container deletes a file that lives in a lower layer, OverlayFS creates a special "whiteout" entry in the `upperdir` that hides the lower file from the merged view without actually deleting anything below.

The consequence is that ten containers from the same image share one set of layer files on disk. Only their per-container `upperdir` consumes new space — typically a few MB until the workload writes a lot.

### E. The Docker Engine Stack: dockerd, containerd, runc

When you type `docker run`, four components hand work down a chain:

1. **`docker` (CLI)** — parses your command and sends an HTTP request to the local daemon socket (`/var/run/docker.sock`).
2. **`dockerd` (daemon)** — handles the high-level concerns: image pulling, network setup, volume management, build orchestration. It does *not* run containers itself.
3. **`containerd`** — a lower-level daemon that owns the container lifecycle: image storage, snapshot management, calling the OCI runtime. dockerd asks containerd to "run this OCI bundle".
4. **`runc`** — the OCI-compliant runtime. It is a small static binary that reads an OCI bundle (a directory containing the rootfs and a `config.json` with namespace/cgroup/capability spec), calls `clone()` with the right flags, sets up cgroups, drops capabilities, and `exec()`s the container's entrypoint.

This split exists because each layer has a different stable interface. **CRI-O** is an alternative to containerd used by some Kubernetes distros — it talks the Kubernetes Container Runtime Interface directly and still calls runc underneath, so the kernel-level container is identical to one Docker would make. Likewise **crun** is a faster runc rewrite in C; you can swap it in without changing anything above.

The Open Container Initiative (OCI) standardizes the boundary between containerd-class and runc-class components: the **image-spec** says how an image is laid out, the **runtime-spec** says what `config.json` must contain, and the **distribution-spec** says how registries serve images. That standardization is why a Docker image can run under Podman, why Buildah can produce images Docker pulls without modification, and why containerd swap-outs are even possible.

### From Theory to the Commands Below

Each section that follows is one of these mechanisms with a friendly verb in front:

- `docker run -it ubuntu bash` — `dockerd` asks `containerd`, which asks `runc`, which calls `clone(CLONE_NEWPID|CLONE_NEWNET|CLONE_NEWNS|...)` and `execve("bash")`. The TTY flags (`-it`) wire stdin/stdout to your terminal.
- `docker run -p 8080:80 nginx` — Docker creates a NET namespace, attaches one end of a `veth` pair to the container and the other to the `docker0` bridge, then writes an iptables DNAT rule that rewrites incoming traffic on host port 8080 to the container's port 80.
- `docker run --memory=512m --cpus=1` — Docker creates a cgroup with `memory.max=512M` and CPU quota equivalent to one core, then places the container PID into it.
- `docker ps` / `docker exec` — both reach into containerd's state (and use `setns()` for `exec`) to find or join an already-running container's namespaces.
- `docker version` shows the four-layer stack: client → daemon → containerd → runc.

The remainder of the lesson is the practical side. Keep this picture in mind: every command below ultimately turns into the namespace, cgroup, and union-filesystem operations described here.

---

## 1. What is Docker?

Docker is a **container-based virtualization platform**. It packages applications and their execution environments so they can run identically anywhere.

### Why use Docker?

**Problem scenario:**
```
Developer A: "It works on my computer?"
Developer B: "I have Node 18 but the server has Node 16..."
Operations team: "Different library versions cause errors"
```

**Docker solution:**
```
Package entire environment in a container → Runs identically everywhere
```

### Advantages of Docker

| Advantage | Description |
|-----------|-------------|
| **Consistency** | Identical dev/test/production environments |
| **Isolation** | Applications run independently |
| **Portability** | Runs identically anywhere |
| **Lightweight** | Faster and lighter than VMs |
| **Version control** | Manage environment versions with images |

---

## 2. Containers vs Virtual Machines (VM)

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

## 3. Docker Core Concepts

### Image

- **Template** for creating containers
- Read-only
- Composed of layers

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

### Container

- Running **instance** of an image
- Read/write capable
- Runs in isolated environment

```
Image ────▶ Container
(Blueprint)  (Actual building)

One image → Can create multiple containers
```

### Docker Hub

- Docker image repository (like GitHub)
- Provides official images: nginx, node, python, mysql, etc.
- https://hub.docker.com

---

## 4. Installing Docker

### macOS

**Docker Desktop installation (recommended):**
1. Download [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Run DMG file
3. Drag to Applications folder
4. Run Docker Desktop

**Install via Homebrew:**
```bash
brew install --cask docker
```

### Windows

1. Download [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Run installer
3. Enable WSL 2 backend (recommended)
4. Run Docker Desktop after restart

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

## 5. Verify Installation

```bash
# Check Docker version
docker --version
# Output example: Docker version 24.0.7, build afdd53b

# Docker detailed information
docker info

# Run test container
docker run hello-world
```

### hello-world execution result

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

## 6. Docker Workflow

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

1. Execute **docker run** command
2. Docker Client requests Docker Daemon
3. If image doesn't exist locally, download from Docker Hub
4. Create and run container from image

---

## Practice Examples

### Example 1: Run First Container

```bash
# Run hello-world image
docker run hello-world

# Check running containers
docker ps

# Check all containers (including stopped)
docker ps -a
```

### Example 2: Run Nginx Web Server

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

## Command Summary

| Command | Description |
|---------|-------------|
| `docker --version` | Check version |
| `docker info` | Docker detailed information |
| `docker run image` | Run container |
| `docker ps` | List running containers |
| `docker ps -a` | List all containers |

---

**Next**: [Docker Images and Containers](./02_Images_and_Containers.md)
