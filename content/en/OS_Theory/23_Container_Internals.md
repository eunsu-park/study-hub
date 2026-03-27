[Previous: Real-Time OS](./22_Real_Time_OS.md)

---

# 23. Container Internals

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain Linux namespaces and how they provide process isolation
2. Implement cgroups v2 for resource limiting and accounting
3. Describe overlay filesystems and container image layering
4. Build a minimal container runtime from scratch using Linux primitives
5. Analyze the OCI runtime specification and container lifecycle

---

## Table of Contents

1. [Containers vs VMs](#1-containers-vs-vms)
2. [Linux Namespaces](#2-linux-namespaces)
3. [Control Groups (cgroups v2)](#3-control-groups-cgroups-v2)
4. [Overlay Filesystems](#4-overlay-filesystems)
5. [Building a Minimal Container](#5-building-a-minimal-container)
6. [OCI Runtime Specification](#6-oci-runtime-specification)
7. [Container Networking](#7-container-networking)
8. [Exercises](#8-exercises)

---

## 1. Containers vs VMs

### 1.1 Architecture Comparison

```
Virtual Machines:
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │  App A   │ │  App B   │ │  App C   │
  ├──────────┤ ├──────────┤ ├──────────┤
  │ Bins/Libs│ │ Bins/Libs│ │ Bins/Libs│
  ├──────────┤ ├──────────┤ ├──────────┤
  │ Guest OS │ │ Guest OS │ │ Guest OS │  ← Full OS per VM
  └────┬─────┘ └────┬─────┘ └────┬─────┘
  ┌────┴──────────────┴──────────────┴────┐
  │            Hypervisor                  │
  ├───────────────────────────────────────┤
  │            Host OS                     │
  └───────────────────────────────────────┘

Containers:
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │  App A   │ │  App B   │ │  App C   │
  ├──────────┤ ├──────────┤ ├──────────┤
  │ Bins/Libs│ │ Bins/Libs│ │ Bins/Libs│
  └────┬─────┘ └────┬─────┘ └────┬─────┘
  ┌────┴──────────────┴──────────────┴────┐
  │     Container Runtime (runc)           │
  ├───────────────────────────────────────┤
  │            Host OS (shared kernel)     │  ← ONE kernel
  └───────────────────────────────────────┘
```

---

## 2. Linux Namespaces

### 2.1 Namespace Types

```c
#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mount.h>

/*
 * Linux Namespaces isolate different system resources:
 *
 * CLONE_NEWPID:   Process IDs (container sees PID 1)
 * CLONE_NEWNS:    Mount points (own filesystem view)
 * CLONE_NEWNET:   Network stack (own interfaces, IPs)
 * CLONE_NEWUTS:   Hostname (own hostname)
 * CLONE_NEWIPC:   IPC (own semaphores, shared memory)
 * CLONE_NEWUSER:  User IDs (root inside, unprivileged outside)
 * CLONE_NEWCGROUP: Cgroup root
 */

#define STACK_SIZE (1024 * 1024)

int child_fn(void *arg) {
    /* Inside the container! */

    /* Set hostname */
    sethostname("container", 9);

    char hostname[64];
    gethostname(hostname, sizeof(hostname));
    printf("[Container] Hostname: %s\n", hostname);
    printf("[Container] PID: %d\n", getpid());
    printf("[Container] UID: %d\n", getuid());

    /* Mount proc for the new PID namespace */
    mount("proc", "/proc", "proc", 0, NULL);

    /* Execute shell inside container */
    char *args[] = {"/bin/sh", NULL};
    execvp(args[0], args);
    perror("execvp");
    return 1;
}

int main(void) {
    char *stack = malloc(STACK_SIZE);
    if (!stack) { perror("malloc"); return 1; }

    printf("[Host] PID: %d\n", getpid());

    /* Create child process in new namespaces */
    int flags = CLONE_NEWPID | CLONE_NEWNS | CLONE_NEWUTS |
                CLONE_NEWNET | CLONE_NEWIPC | SIGCHLD;

    pid_t child = clone(child_fn, stack + STACK_SIZE, flags, NULL);
    if (child == -1) {
        perror("clone");
        free(stack);
        return 1;
    }

    printf("[Host] Container PID (from host view): %d\n", child);
    waitpid(child, NULL, 0);

    free(stack);
    return 0;
}
```

---

## 3. Control Groups (cgroups v2)

### 3.1 cgroups v2 Architecture

```
cgroups v2: Unified hierarchy for resource management.

Resources controlled:
  cpu:     CPU time allocation
  memory:  Memory limits, accounting
  io:      Block I/O bandwidth
  pids:    Process count limit
  cpuset:  CPU/memory node assignment
  rdma:    RDMA resources

cgroup hierarchy:
  /sys/fs/cgroup/
  ├── cgroup.controllers    # Available controllers
  ├── cgroup.subtree_control # Enabled for children
  ├── system.slice/         # System services
  ├── user.slice/           # User sessions
  └── my_container/         # Our container cgroup
      ├── cgroup.procs      # PIDs in this cgroup
      ├── cpu.max           # CPU limit
      ├── memory.max        # Memory limit
      ├── memory.current    # Current memory usage
      ├── io.max            # I/O bandwidth limit
      └── pids.max          # Process count limit
```

### 3.2 cgroups Implementation

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

#define CGROUP_ROOT "/sys/fs/cgroup"

void write_file(const char *path, const char *content) {
    FILE *fp = fopen(path, "w");
    if (!fp) { perror(path); return; }
    fputs(content, fp);
    fclose(fp);
}

void setup_cgroup(const char *name, int cpu_percent,
                  long memory_bytes, int max_pids) {
    char path[512];

    /* Create cgroup directory */
    snprintf(path, sizeof(path), "%s/%s", CGROUP_ROOT, name);
    mkdir(path, 0755);

    /* Enable controllers */
    snprintf(path, sizeof(path), "%s/cgroup.subtree_control", CGROUP_ROOT);
    write_file(path, "+cpu +memory +pids +io");

    /* CPU limit: cpu.max = "quota period" (microseconds) */
    /* 50% CPU = "50000 100000" (50ms out of 100ms) */
    snprintf(path, sizeof(path), "%s/%s/cpu.max", CGROUP_ROOT, name);
    char cpu_max[64];
    snprintf(cpu_max, sizeof(cpu_max), "%d 100000",
             cpu_percent * 1000);
    write_file(path, cpu_max);
    printf("CPU limit: %d%%\n", cpu_percent);

    /* Memory limit */
    snprintf(path, sizeof(path), "%s/%s/memory.max", CGROUP_ROOT, name);
    char mem_max[64];
    snprintf(mem_max, sizeof(mem_max), "%ld", memory_bytes);
    write_file(path, mem_max);
    printf("Memory limit: %ld bytes\n", memory_bytes);

    /* PID limit */
    snprintf(path, sizeof(path), "%s/%s/pids.max", CGROUP_ROOT, name);
    char pids_max[32];
    snprintf(pids_max, sizeof(pids_max), "%d", max_pids);
    write_file(path, pids_max);
    printf("PID limit: %d\n", max_pids);

    /* Move current process into cgroup */
    snprintf(path, sizeof(path), "%s/%s/cgroup.procs", CGROUP_ROOT, name);
    char pid_str[32];
    snprintf(pid_str, sizeof(pid_str), "%d", getpid());
    write_file(path, pid_str);
    printf("Process %s moved to cgroup %s\n", pid_str, name);
}
```

---

## 4. Overlay Filesystems

### 4.1 How Container Images Work

```
OverlayFS: Union mount filesystem for container images.

Docker image layers:
  Layer 3 (top): Application code      (read-write)
  Layer 2:       pip install packages   (read-only)
  Layer 1:       apt-get update         (read-only)
  Layer 0:       Ubuntu base image      (read-only)

OverlayFS merge view:
  ┌─────────────────────────────┐
  │     Merged View (union)      │  ← Container sees this
  ├─────────────────────────────┤
  │  upperdir (read-write)       │  ← Container writes here
  ├─────────────────────────────┤
  │  lowerdir (read-only layers) │  ← Image layers
  └─────────────────────────────┘

  Read: Check upper first, then lower layers
  Write: Always goes to upper layer (copy-up if needed)
  Delete: Whiteout file in upper layer
```

### 4.2 Setting Up OverlayFS

```c
#include <stdio.h>
#include <sys/mount.h>
#include <sys/stat.h>

void setup_overlay(const char *lower, const char *upper,
                   const char *work, const char *merged) {
    /* Create directories */
    mkdir(upper, 0755);
    mkdir(work, 0755);
    mkdir(merged, 0755);

    /* Mount overlayfs */
    char options[1024];
    snprintf(options, sizeof(options),
             "lowerdir=%s,upperdir=%s,workdir=%s",
             lower, upper, work);

    int ret = mount("overlay", merged, "overlay", 0, options);
    if (ret != 0) {
        perror("mount overlay");
        return;
    }

    printf("OverlayFS mounted at %s\n", merged);
    printf("  Lower (read-only): %s\n", lower);
    printf("  Upper (read-write): %s\n", upper);
}
```

---

## 5. Building a Minimal Container

### 5.1 Mini-Container Runtime

```c
#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <string.h>

#define STACK_SIZE (1024 * 1024)

typedef struct {
    char *rootfs;
    char *hostname;
    char **command;
    int cpu_percent;
    long memory_limit;
} container_config_t;

int container_main(void *arg) {
    container_config_t *config = (container_config_t *)arg;

    /* 1. Set hostname */
    sethostname(config->hostname, strlen(config->hostname));

    /* 2. Set up mount namespace */
    /* Pivot root to new rootfs */
    if (chroot(config->rootfs) != 0) {
        perror("chroot");
        return 1;
    }
    chdir("/");

    /* 3. Mount essential filesystems */
    mount("proc", "/proc", "proc", 0, NULL);
    mount("sysfs", "/sys", "sysfs", 0, NULL);
    mount("tmpfs", "/tmp", "tmpfs", 0, NULL);

    /* 4. Print container info */
    printf("=== Container Started ===\n");
    printf("Hostname: %s\n", config->hostname);
    printf("PID: %d (appears as 1 inside container)\n", getpid());
    printf("Root: %s\n", config->rootfs);

    /* 5. Execute command */
    execvp(config->command[0], config->command);
    perror("execvp");
    return 1;
}

void run_container(container_config_t *config) {
    char *stack = malloc(STACK_SIZE);

    int flags = CLONE_NEWPID    /* New PID namespace */
              | CLONE_NEWNS     /* New mount namespace */
              | CLONE_NEWUTS    /* New UTS namespace (hostname) */
              | CLONE_NEWNET    /* New network namespace */
              | CLONE_NEWIPC    /* New IPC namespace */
              | SIGCHLD;

    pid_t child = clone(container_main, stack + STACK_SIZE,
                        flags, config);
    if (child == -1) {
        perror("clone");
        free(stack);
        return;
    }

    /* Set up cgroups for the container process */
    printf("[Host] Container process: %d\n", child);

    int status;
    waitpid(child, &status, 0);
    printf("[Host] Container exited with status %d\n",
           WEXITSTATUS(status));

    free(stack);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <rootfs> <command> [args...]\n", argv[0]);
        return 1;
    }

    char *cmd[] = {argv[2], NULL};  /* Simplified */

    container_config_t config = {
        .rootfs = argv[1],
        .hostname = "mini-container",
        .command = cmd,
        .cpu_percent = 50,
        .memory_limit = 256 * 1024 * 1024,
    };

    run_container(&config);
    return 0;
}
```

---

## 6. OCI Runtime Specification

### 6.1 OCI Lifecycle

```
OCI (Open Container Initiative) defines the container lifecycle:

1. create:  Set up namespaces, cgroups, rootfs
2. start:   Execute the container's entrypoint
3. running: Container is executing
4. stop:    Send SIGTERM, then SIGKILL after timeout
5. delete:  Clean up all resources

Container State:
  {
    "ociVersion": "1.0.0",
    "id": "container-abc123",
    "status": "running",
    "pid": 12345,
    "bundle": "/containers/myapp",
    "annotations": {}
  }
```

---

## 7. Container Networking

### 7.1 Network Namespaces and veth Pairs

```
Container networking uses virtual ethernet (veth) pairs:

  Host namespace              Container namespace
  ┌──────────────┐           ┌──────────────┐
  │   eth0       │           │   eth0       │
  │   (real NIC) │           │   (veth peer)│
  │   bridge0    │◄─ veth ──▶│   172.17.0.2 │
  │   172.17.0.1 │           │              │
  └──────────────┘           └──────────────┘

  NAT/iptables rules handle:
  - Container -> Internet: Masquerade (SNAT)
  - Internet -> Container: Port mapping (DNAT)
```

---

## 8. Exercises

### Exercise 1: Namespace Exploration

Explore Linux namespaces hands-on:
1. Create a PID namespace: verify PID 1 inside, real PID outside
2. Create a UTS namespace: change hostname without affecting host
3. Create a mount namespace: mount tmpfs invisible to host
4. Create a network namespace: show no interfaces initially
5. Combine all namespaces in one program for a mini-container

### Exercise 2: cgroups Resource Limiting

Implement cgroups-based resource limits:
1. Create a cgroup with 50% CPU limit and verify with stress test
2. Set memory limit to 100 MB and test with memory-hungry program
3. Set PID limit to 10 and try to fork beyond it
4. Monitor resource usage via cgroup statistics files
5. Implement a simple cgroup manager that cleans up on exit

### Exercise 3: Build Your Own Container

Create a minimal container runtime in C:
1. Use clone() with namespace flags for isolation
2. Set up chroot with a minimal rootfs (busybox)
3. Mount /proc and /sys inside the container
4. Apply cgroup limits for CPU and memory
5. Execute /bin/sh and verify isolation

### Exercise 4: OverlayFS Image Layers

Work with overlay filesystems:
1. Create a 3-layer overlay filesystem (base, packages, app)
2. Demonstrate copy-up: modify a file from lower layer
3. Demonstrate whiteout: delete a file from lower layer
4. Measure: read performance from lower vs upper layers
5. Implement simple "docker commit": snapshot upper layer as new lower

### Exercise 5: Container Networking

Set up container networking manually:
1. Create a network namespace
2. Create a veth pair connecting host and container namespace
3. Assign IP addresses and set up routing
4. Enable internet access with NAT (iptables masquerade)
5. Set up port forwarding to expose a container service

---

*End of Lesson 23*
