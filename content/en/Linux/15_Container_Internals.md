# 15. Container Internals

**Previous**: [Performance Tuning](./14_Performance_Tuning.md) | **Next**: [Storage Management](./16_Storage_Management.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the fundamental differences between containers and virtual machines
2. Create and manage Linux namespaces to isolate processes, networks, and filesystems
3. Configure cgroups v2 to limit CPU, memory, and I/O resources for process groups
4. Describe how OverlayFS provides layered, copy-on-write storage for container images
5. Build a minimal container manually using namespaces, cgroups, and chroot
6. Apply security mechanisms including capabilities, seccomp, and AppArmor to harden containers

## Table of Contents
1. [Container Fundamentals](#1-container-fundamentals)
2. [Linux Namespaces](#2-linux-namespaces)
3. [Control Groups (cgroups)](#3-control-groups-cgroups)
4. [Union Filesystem](#4-union-filesystem)
5. [Container Runtime](#5-container-runtime)
6. [Security](#6-security)
7. [eBPF Basics](#7-ebpf-basics)
8. [Container Networking Deep Dive](#8-container-networking-deep-dive)
9. [Practice Exercises](#9-practice-exercises)

---

Containers have revolutionized how software is built, shipped, and run. But behind tools like Docker and Podman lies a set of Linux kernel features -- namespaces, cgroups, and union filesystems -- that have existed for years. Understanding these internals is essential for debugging container issues, writing secure container configurations, and knowing what containers actually guarantee (and what they do not). This lesson peels back the abstraction layer so you can work with containers at the kernel level.

## 1. Container Fundamentals

### 1.1 Containers vs Virtual Machines

```
┌─────────────────────────────────────────────────────────────┐
│               Virtual Machines vs Containers                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Virtual Machine                                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                       │
│  │  App A  │ │  App B  │ │  App C  │                       │
│  ├─────────┤ ├─────────┤ ├─────────┤                       │
│  │ Guest OS│ │ Guest OS│ │ Guest OS│                       │
│  └─────────┴─────────┴─────────┘                           │
│  ┌─────────────────────────────────────┐                   │
│  │           Hypervisor                │                   │
│  └─────────────────────────────────────┘                   │
│  ┌─────────────────────────────────────┐                   │
│  │            Host OS                  │                   │
│  └─────────────────────────────────────┘                   │
│                                                             │
│  Container                                                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                       │
│  │  App A  │ │  App B  │ │  App C  │                       │
│  ├─────────┤ ├─────────┤ ├─────────┤                       │
│  │ Bins/   │ │ Bins/   │ │ Bins/   │                       │
│  │ Libs    │ │ Libs    │ │ Libs    │                       │
│  └─────────┴─────────┴─────────┘                           │
│  ┌─────────────────────────────────────┐                   │
│  │        Container Runtime            │                   │
│  └─────────────────────────────────────┘                   │
│  ┌─────────────────────────────────────┐                   │
│  │      Host OS (Shared Kernel)        │                   │
│  └─────────────────────────────────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Core Container Technologies

```
┌─────────────────────────────────────────────────────────────┐
│                  Core Container Technologies                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Namespaces - Isolation                                  │
│     • PID namespace    - Process ID isolation               │
│     • Network namespace - Network stack isolation           │
│     • Mount namespace  - Filesystem isolation               │
│     • UTS namespace    - Hostname isolation                 │
│     • IPC namespace    - Inter-process communication        │
│     • User namespace   - User/Group ID isolation            │
│     • Cgroup namespace - cgroup root isolation              │
│                                                             │
│  2. Cgroups - Resource Limiting                             │
│     • CPU, memory, I/O, network bandwidth limits            │
│     • Process group management                              │
│                                                             │
│  3. Union Filesystem - Layered images                       │
│     • OverlayFS, AUFS                                      │
│     • Copy-on-Write                                        │
│                                                             │
│  4. Capabilities - Privilege separation                     │
│     • Fine-grained root privileges                          │
│                                                             │
│  5. Seccomp - System call filtering                         │
│     • Only allowed syscalls can execute                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

> **Analogy: The Apartment Building.** Linux containers are like apartments in a building. Namespaces give each tenant their own view -- their own mailbox (PID namespace), their own door number (network namespace), and their own utility meter (mount namespace) -- even though they share the same building structure (kernel). Cgroups are the building manager who ensures no single tenant uses more than their fair share of water and electricity.

## 2. Linux Namespaces

### 2.1 Namespace Types

```bash
# Check current process namespaces
ls -la /proc/$$/ns/
# cgroup -> 'cgroup:[4026531835]'
# ipc -> 'ipc:[4026531839]'
# mnt -> 'mnt:[4026531840]'
# net -> 'net:[4026531992]'
# pid -> 'pid:[4026531836]'
# user -> 'user:[4026531837]'
# uts -> 'uts:[4026531838]'

# All system namespaces
lsns

# Namespaces of specific process
lsns -p <PID>
```

### 2.2 Creating Namespaces with unshare

```bash
# UTS namespace (hostname isolation)
unshare --uts /bin/bash
hostname container-test
hostname  # container-test
exit
hostname  # Original hostname

# PID namespace (process isolation)
unshare --pid --fork --mount-proc /bin/bash
ps aux  # Only isolated processes visible
echo $$  # PID 1
exit

# Mount namespace (filesystem isolation)
unshare --mount /bin/bash
mount --bind /tmp /mnt
ls /mnt  # No effect on host's /mnt
exit

# Network namespace (network isolation)
unshare --net /bin/bash
ip a  # Only lo exists
exit

# User namespace (user isolation)
unshare --user --map-root-user /bin/bash
id  # uid=0(root) gid=0(root)
# Actually running as normal user
exit

# Combined (container-like)
unshare --mount --uts --ipc --net --pid --fork --user --map-root-user /bin/bash
```

### 2.3 Entering Namespaces with nsenter

```bash
# Enter another process's namespace
nsenter -t <PID> --all /bin/bash

# Specific namespaces only
nsenter -t <PID> --net /bin/bash
nsenter -t <PID> --pid --mount /bin/bash

# Enter Docker container namespace
docker inspect --format '{{.State.Pid}}' <container_id>
nsenter -t <PID> --all /bin/bash
```

### 2.4 Namespace C Example

```c
// simple_container.c
#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#define STACK_SIZE (1024 * 1024)

static char child_stack[STACK_SIZE];

int child_fn(void *arg) {
    // Change hostname
    sethostname("container", 9);

    // chroot to new root filesystem (if prepared)
    // chroot("/path/to/rootfs");
    // chdir("/");

    // Execute shell
    char *argv[] = {"/bin/bash", NULL};
    execv(argv[0], argv);
    return 0;
}

int main() {
    // Create child process with new namespaces
    int flags = CLONE_NEWUTS |     // UTS namespace
                CLONE_NEWPID |     // PID namespace
                CLONE_NEWNS |      // Mount namespace
                CLONE_NEWNET |     // Network namespace
                SIGCHLD;

    pid_t pid = clone(child_fn, child_stack + STACK_SIZE, flags, NULL);

    if (pid == -1) {
        perror("clone");
        exit(1);
    }

    waitpid(pid, NULL, 0);
    return 0;
}
```

```bash
# Compile and run
gcc -o simple_container simple_container.c
sudo ./simple_container
```

---

## 3. Control Groups (cgroups)

### 3.1 cgroups v2 Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    cgroups v2 Hierarchy                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  /sys/fs/cgroup/ (cgroup2 root)                            │
│  ├── cgroup.controllers      # Available controllers        │
│  ├── cgroup.subtree_control  # Controllers delegated        │
│  ├── cgroup.procs            # Processes in this cgroup     │
│  │                                                          │
│  ├── system.slice/           # systemd system services      │
│  │   ├── cgroup.procs                                      │
│  │   ├── cpu.max                                           │
│  │   └── memory.max                                        │
│  │                                                          │
│  ├── user.slice/             # User sessions                │
│  │   └── user-1000.slice/                                  │
│  │                                                          │
│  └── mygroup/                # Custom group                 │
│      ├── cgroup.procs                                      │
│      ├── cpu.max             # CPU limit                    │
│      ├── cpu.stat            # CPU statistics               │
│      ├── memory.max          # Memory limit                 │
│      ├── memory.current      # Current memory usage         │
│      └── io.max              # I/O limit                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Basic cgroups Commands

```bash
# Check cgroups v2
mount | grep cgroup2
cat /sys/fs/cgroup/cgroup.controllers
# cpuset cpu io memory hugetlb pids rdma misc

# Create new cgroup
mkdir /sys/fs/cgroup/mygroup

# Enable controllers
echo "+cpu +memory +io" > /sys/fs/cgroup/cgroup.subtree_control

# Add process
echo $$ > /sys/fs/cgroup/mygroup/cgroup.procs

# Check processes
cat /sys/fs/cgroup/mygroup/cgroup.procs

# Delete cgroup (must be empty)
rmdir /sys/fs/cgroup/mygroup
```

### 3.3 CPU Limiting

```bash
# CPU limit (quota / period)
# cpu.max: "quota period" (microseconds)
# 50% CPU limit
echo "50000 100000" > /sys/fs/cgroup/mygroup/cpu.max

# Specific CPUs only
# cpuset.cpus: CPUs to use
echo "0-1" > /sys/fs/cgroup/mygroup/cpuset.cpus

# CPU weight (1-10000, default 100)
echo "50" > /sys/fs/cgroup/mygroup/cpu.weight

# Check statistics
cat /sys/fs/cgroup/mygroup/cpu.stat
# usage_usec 12345
# user_usec 10000
# system_usec 2345
```

### 3.4 Memory Limiting

```bash
# Memory limit
echo "512M" > /sys/fs/cgroup/mygroup/memory.max
# Or in bytes
echo "536870912" > /sys/fs/cgroup/mygroup/memory.max

# Memory + swap limit
echo "1G" > /sys/fs/cgroup/mygroup/memory.swap.max

# OOM settings
# memory.oom.group: 1 kills entire group
echo 1 > /sys/fs/cgroup/mygroup/memory.oom.group

# Current usage
cat /sys/fs/cgroup/mygroup/memory.current
cat /sys/fs/cgroup/mygroup/memory.stat
```

### 3.5 I/O Limiting

```bash
# Check devices
lsblk
# Example: sda -> 8:0

# I/O bandwidth limit (bytes/sec)
echo "8:0 rbps=10485760 wbps=10485760" > /sys/fs/cgroup/mygroup/io.max
# 10MB/s read/write limit

# IOPS limit
echo "8:0 riops=1000 wiops=1000" > /sys/fs/cgroup/mygroup/io.max

# I/O weight
echo "8:0 100" > /sys/fs/cgroup/mygroup/io.weight

# Statistics
cat /sys/fs/cgroup/mygroup/io.stat
```

### 3.6 systemd and cgroups

```bash
# systemd-cgls - view cgroup tree
systemd-cgls

# Specific slice
systemd-cgls /system.slice

# systemd-cgtop - real-time monitoring
systemd-cgtop

# Service resource limits (unit file)
# [Service]
# CPUQuota=50%
# MemoryMax=512M
# IOWriteBandwidthMax=/dev/sda 10M

# Change limits at runtime
systemctl set-property nginx.service CPUQuota=50%
systemctl set-property nginx.service MemoryMax=512M
```

---

## 4. Union Filesystem

### 4.1 OverlayFS Concept

```
┌─────────────────────────────────────────────────────────────┐
│                    OverlayFS Structure                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Merged (Unified view) - Filesystem seen by container       │
│  /merged                                                    │
│     │                                                       │
│     ├── [From Upper]    │                                   │
│     ├── [From Lower]    │                                   │
│     └── [From Lower]    │                                   │
│                                                             │
│  ┌─────────────────────┐                                   │
│  │    Upper Layer      │  ← Writable (container changes)    │
│  │    /upper           │                                    │
│  └─────────────────────┘                                   │
│           ↑                                                 │
│  ┌─────────────────────┐                                   │
│  │   Lower Layer(s)    │  ← Read-only (image layers)        │
│  │   /lower1           │                                    │
│  │   /lower2           │                                    │
│  │   /lower3           │                                    │
│  └─────────────────────┘                                   │
│                                                             │
│  Work Directory                                             │
│  /work - Internal OverlayFS working directory               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Using OverlayFS

```bash
# Mount OverlayFS
mkdir -p /lower /upper /work /merged

# Create files in lower
echo "from lower" > /lower/file1.txt
echo "will be overwritten" > /lower/file2.txt

# Create files in upper
echo "from upper" > /upper/file2.txt
echo "only in upper" > /upper/file3.txt

# Mount OverlayFS
mount -t overlay overlay \
  -o lowerdir=/lower,upperdir=/upper,workdir=/work \
  /merged

# Check result
ls /merged/
# file1.txt  file2.txt  file3.txt

cat /merged/file1.txt  # from lower
cat /merged/file2.txt  # from upper (overwritten)
cat /merged/file3.txt  # only in upper

# Write new file
echo "new file" > /merged/file4.txt
ls /upper/  # file4.txt created in upper

# Delete file (whiteout)
rm /merged/file1.txt
ls -la /upper/
# c--------- ... file1.txt  (whiteout file)

# Unmount
umount /merged
```

### 4.3 Docker Image Layers

```bash
# Check Docker image layers
docker image inspect ubuntu:22.04 --format '{{.RootFS.Layers}}'

# Layer storage location
ls /var/lib/docker/overlay2/

# Mount information for specific container
docker inspect <container_id> --format '{{.GraphDriver.Data}}'
# Check LowerDir, UpperDir, MergedDir, WorkDir
```

---

## 5. Container Runtime

### 5.1 Runtime Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Container Runtime Layers                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  High-Level Runtime (Container Engine)                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Docker Engine / Podman / containerd                │   │
│  │  • Image management (pull, push, build)             │   │
│  │  • Networking                                        │   │
│  │  • Volume management                                 │   │
│  │  • API provision                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  Low-Level Runtime (OCI Runtime)                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  runc / crun / kata-containers                      │   │
│  │  • Actual container creation                         │   │
│  │  • namespace, cgroups setup                          │   │
│  │  • OCI spec compliance                               │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  Linux Kernel                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  namespaces, cgroups, seccomp, capabilities        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Creating Container Manually

```bash
#!/bin/bash
# manual_container.sh

# 1. Prepare rootfs
mkdir -p /tmp/mycontainer/{rootfs,upper,work,merged}

# Download base rootfs (Alpine)
curl -o /tmp/alpine.tar.gz https://dl-cdn.alpinelinux.org/alpine/v3.18/releases/x86_64/alpine-minirootfs-3.18.0-x86_64.tar.gz
tar -xzf /tmp/alpine.tar.gz -C /tmp/mycontainer/rootfs

# 2. Mount OverlayFS
mount -t overlay overlay \
  -o lowerdir=/tmp/mycontainer/rootfs,upperdir=/tmp/mycontainer/upper,workdir=/tmp/mycontainer/work \
  /tmp/mycontainer/merged

# 3. Essential mounts
mount -t proc proc /tmp/mycontainer/merged/proc
mount -t sysfs sysfs /tmp/mycontainer/merged/sys
mount -o bind /dev /tmp/mycontainer/merged/dev

# 4. chroot with new namespaces
unshare --mount --uts --ipc --net --pid --fork \
  chroot /tmp/mycontainer/merged /bin/sh -c '
    hostname mycontainer
    mount -t proc proc /proc
    exec /bin/sh
  '

# Cleanup
umount /tmp/mycontainer/merged/{proc,sys,dev}
umount /tmp/mycontainer/merged
```

### 5.3 Using runc

```bash
# Install runc
apt install runc

# OCI bundle structure
mkdir -p bundle/rootfs
cd bundle

# Prepare rootfs (extract from Docker)
docker export $(docker create alpine) | tar -C rootfs -xf -

# Generate config.json
runc spec

# Edit config.json (change terminal: true)

# Run container
runc run mycontainer

# From another terminal
runc list
runc state mycontainer
runc kill mycontainer
runc delete mycontainer
```

### 5.4 Rootless Containers

```bash
# Podman rootless
podman run --rm -it alpine sh

# Check user namespace mapping
cat /proc/self/uid_map
cat /proc/self/gid_map

# subuid/subgid configuration
# /etc/subuid
# username:100000:65536
# /etc/subgid
# username:100000:65536

# Rootless Docker
dockerd-rootless-setuptool.sh install
export PATH=/home/$USER/bin:$PATH
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
docker run --rm hello-world
```

---

## 6. Security

### 6.1 Capabilities

```bash
# Check process capabilities
cat /proc/$$/status | grep Cap
getpcaps $$

# Capabilities list
capsh --print

# Grant specific capabilities only
docker run --cap-drop ALL --cap-add NET_BIND_SERVICE nginx

# Key capabilities:
# CAP_NET_ADMIN - Network configuration
# CAP_NET_BIND_SERVICE - Bind to ports < 1024
# CAP_SYS_ADMIN - System administration (dangerous)
# CAP_SYS_PTRACE - Process tracing
# CAP_MKNOD - Create special files
```

### 6.2 Seccomp

```bash
# Check default seccomp profile
docker info --format '{{.SecurityOptions}}'

# Custom seccomp profile
cat > seccomp.json << 'EOF'
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": ["SCMP_ARCH_X86_64"],
  "syscalls": [
    {
      "names": ["read", "write", "exit", "exit_group"],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
EOF

# Apply profile
docker run --security-opt seccomp=seccomp.json alpine sh

# Run without seccomp (not recommended)
docker run --security-opt seccomp=unconfined alpine sh
```

### 6.3 AppArmor/SELinux

```bash
# AppArmor status
aa-status

# Docker AppArmor profile
cat /etc/apparmor.d/docker

# Apply custom profile
docker run --security-opt apparmor=my-profile alpine

# SELinux (RHEL/CentOS)
getenforce
# docker run --security-opt label=type:my_container_t alpine
```

### 6.4 Read-only Root

```bash
# Read-only root filesystem
docker run --read-only alpine sh

# Allow temporary directory
docker run --read-only --tmpfs /tmp alpine sh

# Allow write via volume
docker run --read-only -v /data alpine sh
```

## 7. eBPF Basics

eBPF (extended Berkeley Packet Filter) is a revolutionary technology that allows sandboxed programs to run inside the Linux kernel without changing kernel source code or loading kernel modules. Originally designed for packet filtering, eBPF has evolved into a general-purpose in-kernel virtual machine that powers modern observability, networking, and security tools.

### 7.1 What is eBPF?

```
┌─────────────────────────────────────────────────────────────┐
│                       eBPF Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Space                                                 │
│  ┌───────────────────────────────────────────────────┐     │
│  │  bcc / bpftrace / libbpf                          │     │
│  │  (eBPF program loader & frontend)                 │     │
│  └───────────────────┬───────────────────────────────┘     │
│                      │ load program                         │
│                      ▼                                      │
│  ─────────────────────────────────────────────────────      │
│  Kernel Space                                               │
│  ┌───────────────┐   ┌───────────────┐                     │
│  │   Verifier    │──>│  JIT Compiler │                     │
│  │ (safety check)│   │ (native code) │                     │
│  └───────────────┘   └───────┬───────┘                     │
│                              │ attach                       │
│                              ▼                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ kprobes  │ │tracepoint│ │   XDP    │ │  cgroup  │      │
│  │          │ │          │ │(network) │ │(resource)│      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
│                                                             │
│  eBPF Maps (shared data between kernel & user space)       │
│  ┌─────────────────────────────────────────────────┐       │
│  │  Hash maps, Arrays, Ring buffers, Per-CPU maps  │       │
│  └─────────────────────────────────────────────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

> **Classic BPF vs eBPF.** Classic BPF (cBPF) was limited to packet filtering with a simple 32-bit instruction set and two registers. eBPF extends this to a 64-bit instruction set with 11 registers, supports maps for state storage, allows function calls, and can attach to nearly any kernel event -- not just network packets. Think of cBPF as a pocket calculator and eBPF as a full programming environment.

### 7.2 eBPF Program Types

| Program Type | Attach Point | Use Case |
|-------------|-------------|----------|
| `kprobe` / `kretprobe` | Any kernel function entry/return | Tracing kernel internals |
| `tracepoint` | Static kernel tracepoints | Stable performance monitoring |
| `XDP` (eXpress Data Path) | Network driver (before sk_buff) | Ultra-fast packet processing |
| `tc` (traffic control) | Network traffic control layer | Packet mangling, policing |
| `cgroup` | cgroup events | Per-container resource control |
| `socket_filter` | Socket layer | Per-socket packet filtering |
| `perf_event` | CPU performance counters | Hardware-level profiling |
| `LSM` (Linux Security Module) | Security hooks | Fine-grained security policy |

### 7.3 eBPF Verifier and JIT Compilation

The eBPF verifier is the safety mechanism that ensures no eBPF program can crash the kernel or access unauthorized memory:

```bash
# The verifier checks:
# 1. No unbounded loops (programs must terminate)
# 2. All memory accesses are bounds-checked
# 3. No null pointer dereferences
# 4. Stack size limited to 512 bytes
# 5. Program size limited (1M instructions as of kernel 5.2+)

# JIT compilation converts verified eBPF bytecode to native machine code
# Check JIT status
cat /proc/sys/net/core/bpf_jit_enable
# 0 = disabled, 1 = enabled, 2 = enabled with debug

# Enable JIT compilation
echo 1 | sudo tee /proc/sys/net/core/bpf_jit_enable
```

### 7.4 bpftool and BCC Tools

```bash
# bpftool: inspect and manage eBPF programs/maps
# List loaded eBPF programs
bpftool prog list

# Show details of a specific program
bpftool prog show id 42

# List eBPF maps
bpftool map list

# Dump map contents
bpftool map dump id 10

# --- BCC (BPF Compiler Collection) Tools ---
# Install BCC tools
apt install bpfcc-tools  # Debian/Ubuntu
# or
yum install bcc-tools    # RHEL/CentOS

# tcptop: monitor TCP traffic by process (like top for TCP)
tcptop

# execsnoop: trace new process execution
execsnoop
# Shows: PCOMM PID PPID RET ARGS
# bash   1234 1000 0   /usr/bin/ls -la

# biolatency: block I/O latency histogram
biolatency

# opensnoop: trace file opens system-wide
opensnoop

# tcpconnect: trace active TCP connections
tcpconnect

# funccount: count kernel function calls
funccount 'tcp_send*'
```

### 7.5 eBPF Use Cases

```
┌─────────────────────────────────────────────────────────────┐
│                    eBPF Use Cases                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Observability                                           │
│     • System call tracing (strace replacement)              │
│     • Application latency profiling                         │
│     • Continuous production profiling (Parca, Pyroscope)    │
│     • Custom metrics without code changes                   │
│                                                             │
│  2. Networking                                              │
│     • XDP-based DDoS mitigation (Cloudflare, Facebook)     │
│     • Load balancing (Cilium, Katran)                       │
│     • Network policy enforcement in K8s                     │
│     • DNS request monitoring                                │
│                                                             │
│  3. Security                                                │
│     • Runtime threat detection (Falco, Tetragon)            │
│     • File integrity monitoring                             │
│     • Network policy enforcement                            │
│     • Syscall auditing (replacing auditd)                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Container Networking Deep Dive

Understanding how containers communicate requires knowledge of Linux networking primitives. Every container network -- whether Docker bridge, Kubernetes pod networking, or service mesh -- builds on these kernel-level building blocks.

### 8.1 Network Namespaces

```bash
# Create two network namespaces (simulating two containers)
ip netns add container1
ip netns add container2

# Verify
ip netns list
# container1
# container2

# Each namespace has its own isolated network stack
ip netns exec container1 ip a
# Only lo (loopback) interface exists, and it's DOWN

# Bring up loopback in each namespace
ip netns exec container1 ip link set lo up
ip netns exec container2 ip link set lo up
```

### 8.2 Veth Pairs and Bridge Networking

```
┌─────────────────────────────────────────────────────────────┐
│              Container Bridge Networking                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Container 1                Container 2                     │
│  (netns: container1)        (netns: container2)             │
│  ┌───────────────┐          ┌───────────────┐              │
│  │ eth0          │          │ eth0          │              │
│  │ 172.17.0.2/24 │          │ 172.17.0.3/24 │              │
│  └───────┬───────┘          └───────┬───────┘              │
│          │ veth pair                │ veth pair              │
│          │                          │                        │
│  ┌───────┴──────────────────────────┴───────┐              │
│  │              docker0 bridge               │              │
│  │              172.17.0.1/24                │              │
│  └──────────────────┬───────────────────────┘              │
│                     │                                       │
│                     │ NAT (iptables MASQUERADE)             │
│                     ▼                                       │
│  ┌─────────────────────────────────────────┐               │
│  │          Host eth0 / ens33              │               │
│  │          192.168.1.100                  │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```bash
# Step-by-step: build container networking manually

# 1. Create a bridge (equivalent to docker0)
ip link add br0 type bridge
ip addr add 172.18.0.1/24 dev br0
ip link set br0 up

# 2. Create veth pairs for container1
ip link add veth1 type veth peer name veth1-br

# 3. Move one end into the container namespace
ip link set veth1 netns container1

# 4. Attach the other end to the bridge
ip link set veth1-br master br0
ip link set veth1-br up

# 5. Configure IP inside the container
ip netns exec container1 ip addr add 172.18.0.2/24 dev veth1
ip netns exec container1 ip link set veth1 up
ip netns exec container1 ip route add default via 172.18.0.1

# 6. Repeat for container2
ip link add veth2 type veth peer name veth2-br
ip link set veth2 netns container2
ip link set veth2-br master br0
ip link set veth2-br up
ip netns exec container2 ip addr add 172.18.0.3/24 dev veth2
ip netns exec container2 ip link set veth2 up
ip netns exec container2 ip route add default via 172.18.0.1

# 7. Test container-to-container communication
ip netns exec container1 ping -c 3 172.18.0.3
# PING 172.18.0.3: 3 packets transmitted, 3 received, 0% packet loss

# 8. Enable NAT for external access
sysctl -w net.ipv4.ip_forward=1
iptables -t nat -A POSTROUTING -s 172.18.0.0/24 -j MASQUERADE
```

### 8.3 Docker Network Modes

| Mode | Command | Isolation | Performance | Use Case |
|------|---------|-----------|-------------|----------|
| **bridge** (default) | `--network bridge` | Container-level | Moderate (NAT overhead) | General purpose |
| **host** | `--network host` | None (shares host stack) | Native | Performance-critical apps |
| **none** | `--network none` | Complete | N/A | Security-sensitive workloads |
| **overlay** | `--network my-overlay` | Cross-host | Lower (VXLAN encap) | Swarm / multi-host |
| **macvlan** | `--network my-macvlan` | VLAN-level | Near-native | Direct LAN integration |

```bash
# Inspect Docker bridge network
docker network inspect bridge
# Shows: subnet, gateway, connected containers, IPAM config

# Create custom bridge with specific subnet
docker network create --driver bridge --subnet 10.10.0.0/16 mynet

# Run containers on custom network (automatic DNS by container name)
docker run -d --name web --network mynet nginx
docker run -it --network mynet alpine ping web
# Container name "web" resolves automatically
```

### 8.4 Kubernetes CNI (Container Network Interface)

```
┌─────────────────────────────────────────────────────────────┐
│                Kubernetes CNI Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  kubelet                                                    │
│    │                                                        │
│    │ "Create pod network"                                   │
│    ▼                                                        │
│  CRI (containerd/CRI-O)                                    │
│    │                                                        │
│    │ CNI ADD / DEL                                          │
│    ▼                                                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    CNI Plugin                         │  │
│  │                                                      │  │
│  │  Calico     │  Cilium      │  Flannel               │  │
│  │  • BGP      │  • eBPF      │  • VXLAN overlay       │  │
│  │  • IP-in-IP │  • No kube-  │  • Simple setup        │  │
│  │  • Policy   │    proxy     │  • Limited policy      │  │
│  │    engine   │  • L7 policy │                         │  │
│  │  • IPAM     │  • Hubble    │                         │  │
│  │             │    (observ.) │                         │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  Pod-to-Pod: Every pod gets a routable IP address          │
│  No NAT between pods (flat network model)                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.5 CNI Plugin Comparison: Calico vs Cilium vs Flannel

| Feature | Calico | Cilium | Flannel |
|---------|--------|--------|---------|
| **Data plane** | iptables or eBPF | eBPF | VXLAN / host-gw |
| **Network policy** | Full (L3/L4) | Full (L3/L4/L7) | None (requires add-on) |
| **Encryption** | WireGuard | WireGuard / IPsec | None |
| **Observability** | Limited | Hubble (deep visibility) | None |
| **Performance** | High (BGP mode) | Highest (eBPF bypass) | Moderate |
| **Complexity** | Medium | Medium-High | Low |
| **Best for** | Large clusters, BGP | Security + observability | Simple/small clusters |

```bash
# Calico: Install via manifest
kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.27.0/manifests/calico.yaml

# Cilium: Install via Helm
helm repo add cilium https://helm.cilium.io/
helm install cilium cilium/cilium --namespace kube-system

# Flannel: Install via manifest
kubectl apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml

# Verify CNI is working
kubectl get pods -n kube-system | grep -E 'calico|cilium|flannel'
```

> **Why Cilium is gaining adoption.** Cilium replaces kube-proxy entirely with eBPF programs, eliminating iptables rules that scale poorly with thousands of services. It provides L7 (HTTP, gRPC, Kafka) network policies, built-in observability via Hubble, and transparent encryption -- all without sidecar proxies. For clusters that need both performance and security visibility, Cilium has become the leading choice.

---

## 9. Practice Exercises

### Exercise 1: Namespace Practice
```bash
# Requirements:
# 1. Create isolated environment using all namespace types
# 2. Verify isolation from host (hostname, PID, network)
# 3. Enter namespace with nsenter

# Write commands:
```

### Exercise 2: cgroups Resource Limiting
```bash
# Requirements:
# 1. Limit CPU to 25%
# 2. Limit memory to 256MB
# 3. Limit I/O to 1MB/s
# 4. Test with stress tool

# Write configuration and commands:
```

### Exercise 3: Manual Container Creation
```bash
# Requirements:
# 1. Prepare rootfs (Alpine)
# 2. Configure OverlayFS
# 3. Namespace isolation
# 4. cgroups limits
# 5. Execute shell

# Write script:
```

### Exercise 4: Hardened Container
```bash
# Requirements:
# 1. Grant minimal capabilities only
# 2. Apply seccomp profile
# 3. Read-only root
# 4. non-root user

# Write docker run command:
```

---

---

## References

- [Linux Namespaces](https://man7.org/linux/man-pages/man7/namespaces.7.html)
- [cgroups v2](https://docs.kernel.org/admin-guide/cgroup-v2.html)
- [OverlayFS](https://docs.kernel.org/filesystems/overlayfs.html)
- [runc](https://github.com/opencontainers/runc)
- [Container Security](https://docs.docker.com/engine/security/)
- [eBPF Documentation](https://ebpf.io/what-is-ebpf/)
- [BCC Tools](https://github.com/iovisor/bcc)
- [Cilium Documentation](https://docs.cilium.io/)
- [Kubernetes CNI Specification](https://github.com/containernetworking/cni)

---

**Previous**: [Performance Tuning](./14_Performance_Tuning.md) | **Next**: [Storage Management](./16_Storage_Management.md)
