[Previous: Container Internals](./23_Container_Internals.md)

---

# 24. eBPF and Kernel Tracing

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain eBPF architecture and its role in modern kernel observability
2. Write BPF programs using BCC (BPF Compiler Collection) and bpftrace
3. Implement XDP programs for high-performance packet processing
4. Use eBPF for system performance analysis and security monitoring
5. Describe the eBPF verifier and safety guarantees

---

## Table of Contents

1. [What Is eBPF?](#1-what-is-ebpf)
2. [eBPF Architecture](#2-ebpf-architecture)
3. [BCC: BPF Compiler Collection](#3-bcc-bpf-compiler-collection)
4. [bpftrace: High-Level Tracing](#4-bpftrace-high-level-tracing)
5. [XDP: eXpress Data Path](#5-xdp-express-data-path)
6. [eBPF for Security](#6-ebpf-for-security)
7. [Performance Analysis with eBPF](#7-performance-analysis-with-ebpf)
8. [Exercises](#8-exercises)

---

## 1. What Is eBPF?

### 1.1 eBPF Overview

```
eBPF (extended Berkeley Packet Filter):
  Run sandboxed programs in the Linux kernel WITHOUT modifying kernel code.

Traditional approach to kernel extension:
  Write kernel module -> compile -> load -> risk crashing kernel

eBPF approach:
  Write BPF program -> verifier checks safety -> JIT compile -> run safely

Use cases:
  - Networking: packet filtering, load balancing (Cilium)
  - Observability: tracing, profiling (bpftrace, BCC)
  - Security: syscall filtering, runtime enforcement (Falco)

  "eBPF is to the kernel what JavaScript is to the web browser"
  - Run custom code safely in a controlled environment
```

### 1.2 eBPF Program Types

```
BPF_PROG_TYPE_KPROBE:        Trace kernel functions
BPF_PROG_TYPE_TRACEPOINT:    Trace predefined kernel events
BPF_PROG_TYPE_PERF_EVENT:    Performance monitoring
BPF_PROG_TYPE_XDP:           Network packet processing
BPF_PROG_TYPE_SOCKET_FILTER: Socket-level packet filtering
BPF_PROG_TYPE_SCHED_CLS:     Traffic control
BPF_PROG_TYPE_CGROUP_SKB:    Per-cgroup network filtering
BPF_PROG_TYPE_LSM:           Linux Security Module hooks
```

---

## 2. eBPF Architecture

### 2.1 Execution Flow

```
User Space                    Kernel Space
┌──────────┐                 ┌──────────────────┐
│ BPF      │   bpf()        │  eBPF Verifier    │
│ Program  │──syscall──────▶│  (safety check)   │
│ (C code) │                │        │           │
└──────────┘                │        ▼           │
                            │  JIT Compiler      │
                            │  (to native code)  │
                            │        │           │
                            │        ▼           │
                            │  Attach to hook:   │
                            │  - kprobe          │
                            │  - tracepoint      │
                            │  - XDP             │
                            │        │           │
                            │        ▼           │
                            │  eBPF Maps         │
                            │  (shared data)     │
                            └────────┬───────────┘
                                     │
┌──────────┐                         │
│ User App │◄──read maps─────────────┘
│ (Python/ │
│  Go/C)   │
└──────────┘
```

### 2.2 eBPF Maps

```
eBPF Maps: Key-value data structures shared between
kernel BPF programs and user-space applications.

Map types:
  BPF_MAP_TYPE_HASH:          Hash table
  BPF_MAP_TYPE_ARRAY:         Array (integer keys)
  BPF_MAP_TYPE_PERF_EVENT_ARRAY: Per-CPU event buffers
  BPF_MAP_TYPE_RINGBUF:       Ring buffer (efficient)
  BPF_MAP_TYPE_LRU_HASH:      LRU hash table
  BPF_MAP_TYPE_STACK_TRACE:   Stack traces
```

---

## 3. BCC: BPF Compiler Collection

### 3.1 BCC Python Examples

```python
#!/usr/bin/env python3
"""
Trace all open() system calls with file paths.
Requires: pip install bcc
Run as root.
"""

from bcc import BPF

# BPF program (C code compiled to eBPF bytecode)
bpf_program = """
#include <uapi/linux/ptrace.h>
#include <linux/fs.h>

struct event_t {
    u32 pid;
    char comm[16];
    char filename[256];
};

BPF_PERF_OUTPUT(events);

int trace_open(struct pt_regs *ctx, const char __user *filename, int flags) {
    struct event_t event = {};

    event.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&event.comm, sizeof(event.comm));
    bpf_probe_read_user_str(&event.filename, sizeof(event.filename), filename);

    events.perf_submit(ctx, &event, sizeof(event));
    return 0;
}
"""

# Load and attach
b = BPF(text=bpf_program)
b.attach_kprobe(event=b.get_syscall_fnname("open"), fn_name="trace_open")

# Process events
def print_event(cpu, data, size):
    event = b["events"].event(data)
    print(f"PID {event.pid:6d} ({event.comm.decode():16s}): {event.filename.decode()}")

b["events"].open_perf_buffer(print_event)

print("Tracing open() calls... Ctrl+C to stop")
while True:
    try:
        b.perf_buffer_poll()
    except KeyboardInterrupt:
        break
```

### 3.2 Counting System Calls

```python
#!/usr/bin/env python3
"""Count system calls per process."""

from bcc import BPF
from time import sleep

bpf_program = """
BPF_HASH(syscall_count, u32, u64);

TRACEPOINT_PROBE(raw_syscalls, sys_enter) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 *count = syscall_count.lookup(&pid);
    if (count) {
        (*count)++;
    } else {
        u64 one = 1;
        syscall_count.update(&pid, &one);
    }
    return 0;
}
"""

b = BPF(text=bpf_program)

print("Counting syscalls for 5 seconds...")
sleep(5)

print(f"{'PID':>8s} {'COUNT':>12s}")
for k, v in sorted(b["syscall_count"].items(), key=lambda x: -x[1].value):
    print(f"{k.value:>8d} {v.value:>12d}")
```

---

## 4. bpftrace: High-Level Tracing

### 4.1 bpftrace One-Liners

```bash
# Count syscalls by process
bpftrace -e 'tracepoint:raw_syscalls:sys_enter { @[comm] = count(); }'

# Trace file opens
bpftrace -e 'tracepoint:syscalls:sys_enter_openat { printf("%s %s\n", comm, str(args->filename)); }'

# Histogram of read() sizes
bpftrace -e 'tracepoint:syscalls:sys_exit_read /args->ret > 0/ { @bytes = hist(args->ret); }'

# Trace process execution
bpftrace -e 'tracepoint:sched:sched_process_exec { printf("%d %s\n", pid, comm); }'

# Count context switches per second
bpftrace -e 'tracepoint:sched:sched_switch { @[comm] = count(); } interval:s:1 { print(@); clear(@); }'

# Block I/O latency histogram
bpftrace -e 'tracepoint:block:block_rq_issue { @start[args->dev, args->sector] = nsecs; }
             tracepoint:block:block_rq_complete /@start[args->dev, args->sector]/ {
               @usecs = hist((nsecs - @start[args->dev, args->sector]) / 1000);
               delete(@start[args->dev, args->sector]); }'

# TCP connection latency
bpftrace -e 'kprobe:tcp_v4_connect { @start[tid] = nsecs; }
             kretprobe:tcp_v4_connect /@start[tid]/ {
               @us = hist((nsecs - @start[tid]) / 1000);
               delete(@start[tid]); }'
```

---

## 5. XDP: eXpress Data Path

### 5.1 XDP Overview

```
XDP: Process packets at the earliest possible point in the network stack.

Traditional packet path:
  NIC → Driver → sk_buff allocation → Netfilter → TCP/IP → Application
  (Many allocations and copies)

XDP packet path:
  NIC → Driver → XDP program → Action
  (Before sk_buff allocation! Minimal overhead)

XDP Actions:
  XDP_PASS:     Pass to normal network stack
  XDP_DROP:     Drop packet (DDoS mitigation!)
  XDP_TX:       Bounce back out same NIC
  XDP_REDIRECT: Send to another NIC or CPU
  XDP_ABORTED:  Error, drop with trace

Performance: 24M packets/second per core (vs ~1M for iptables)
```

### 5.2 XDP Firewall Example

```c
/*
 * Simple XDP firewall: drop packets from specific IP.
 * This is the BPF C code loaded into the kernel.
 */

#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <bpf/bpf_helpers.h>

/* Map to store blocked IPs */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, __u32);      /* IPv4 address */
    __type(value, __u64);    /* Packet count */
    __uint(max_entries, 1024);
} blocked_ips SEC(".maps");

SEC("xdp")
int xdp_firewall(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    /* Parse Ethernet header */
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return XDP_PASS;

    if (eth->h_proto != __constant_htons(ETH_P_IP))
        return XDP_PASS;

    /* Parse IP header */
    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end)
        return XDP_PASS;

    /* Check if source IP is blocked */
    __u32 src_ip = ip->saddr;
    __u64 *count = bpf_map_lookup_elem(&blocked_ips, &src_ip);

    if (count) {
        /* IP is blocked - increment counter and drop */
        __sync_fetch_and_add(count, 1);
        return XDP_DROP;
    }

    return XDP_PASS;
}

char _license[] SEC("license") = "GPL";
```

---

## 6. eBPF for Security

### 6.1 Syscall Monitoring

```python
#!/usr/bin/env python3
"""Monitor suspicious system calls for security."""

from bcc import BPF

bpf_program = """
#include <uapi/linux/ptrace.h>

struct security_event_t {
    u32 pid;
    u32 uid;
    char comm[16];
    int syscall_nr;
};

BPF_PERF_OUTPUT(security_events);

TRACEPOINT_PROBE(raw_syscalls, sys_enter) {
    struct security_event_t event = {};

    /* Monitor sensitive syscalls */
    int nr = args->id;

    /* ptrace (debugging/injection), execve (execution),
     * init_module (kernel module loading) */
    if (nr == 101 || nr == 59 || nr == 175) {
        event.pid = bpf_get_current_pid_tgid() >> 32;
        event.uid = bpf_get_current_uid_gid();
        event.syscall_nr = nr;
        bpf_get_current_comm(&event.comm, sizeof(event.comm));

        security_events.perf_submit(args, &event, sizeof(event));
    }

    return 0;
}
"""

b = BPF(text=bpf_program)

syscall_names = {101: "ptrace", 59: "execve", 175: "init_module"}

def print_event(cpu, data, size):
    event = b["security_events"].event(data)
    name = syscall_names.get(event.syscall_nr, str(event.syscall_nr))
    print(f"[SECURITY] PID={event.pid} UID={event.uid} "
          f"comm={event.comm.decode()} syscall={name}")

b["security_events"].open_perf_buffer(print_event)
print("Monitoring sensitive syscalls... Ctrl+C to stop")
while True:
    try:
        b.perf_buffer_poll()
    except KeyboardInterrupt:
        break
```

---

## 7. Performance Analysis with eBPF

### 7.1 CPU Analysis

```bash
# Profile CPU usage by stack trace
# Shows where CPU time is spent
profile -F 99 -p $(pgrep myapp) 10

# Off-CPU analysis: where is the process waiting?
offcputime -p $(pgrep myapp) 5

# Scheduler latency: how long tasks wait in run queue
runqlat 1

# Per-process scheduler latency
runqslower 10000  # Show tasks waiting > 10ms
```

### 7.2 Memory Analysis

```bash
# Track memory allocations by stack trace
memleak -p $(pgrep myapp)

# Page fault tracing
bpftrace -e 'software:page-faults:1 { @[comm, ustack] = count(); }'

# OOM killer monitoring
bpftrace -e 'kprobe:oom_kill_process { printf("OOM: %s pid=%d\n", comm, pid); }'
```

---

## 8. Exercises

### Exercise 1: System Call Tracer

Build a syscall tracer with BCC:
1. Trace all syscalls for a specific PID
2. Count syscalls by type (open, read, write, etc.)
3. Measure latency for each syscall type
4. Generate a report: top 10 syscalls by count and by total time
5. Compare with strace output to verify correctness

### Exercise 2: bpftrace Scripts

Write bpftrace scripts for common analysis:
1. Track file opens by process (with full path)
2. Histogram of TCP connection durations
3. Count DNS queries by process
4. Measure disk I/O latency per device
5. Detect processes with excessive context switches

### Exercise 3: XDP Packet Counter

Build an XDP-based packet counter:
1. Write XDP program that counts packets by protocol (TCP/UDP/ICMP)
2. Store counts in BPF maps
3. Write userspace program to read and display statistics
4. Measure: packets per second your counter can handle
5. Compare overhead: XDP counter vs iptables counter

### Exercise 4: eBPF Security Monitor

Create a security monitoring tool:
1. Monitor process execution (execve calls) with full command lines
2. Track file modifications in sensitive directories (/etc, /root)
3. Detect network connections to unusual ports
4. Alert on privilege escalation (setuid calls)
5. Log all events with timestamps for forensic analysis

### Exercise 5: Performance Profiler

Build a comprehensive profiler:
1. CPU profiling: flame graph from eBPF stack traces
2. Memory profiling: allocation tracking and leak detection
3. I/O profiling: latency histogram per file
4. Network profiling: connection latency and throughput
5. Generate a single-page HTML report with all metrics

---

*End of Lesson 24*
