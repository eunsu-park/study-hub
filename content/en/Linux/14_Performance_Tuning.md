# 14. Linux Performance Tuning

**Previous**: [Advanced systemd](./13_Systemd_Advanced.md) | **Next**: [Container Internals](./15_Container_Internals.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply the USE methodology to systematically identify performance bottlenecks
2. Monitor system performance using tools like top, vmstat, mpstat, and iostat
3. Tune CPU scheduling, governor settings, and process affinity
4. Optimize memory behavior through sysctl parameters including swappiness and dirty page ratios
5. Select and configure appropriate I/O schedulers for different workloads
6. Tune TCP/IP stack parameters for high-performance networking
7. Profile applications with perf and generate flamegraphs for bottleneck analysis

## Table of Contents
1. [Performance Analysis Fundamentals](#1-performance-analysis-fundamentals)
2. [CPU Tuning](#2-cpu-tuning)
3. [Memory Tuning](#3-memory-tuning)
4. [I/O Tuning](#4-io-tuning)
5. [Network Tuning](#5-network-tuning)
6. [Profiling Tools](#6-profiling-tools)
7. [Practice Exercises](#7-practice-exercises)

---

A slow server does not just frustrate users -- it costs revenue, violates SLAs, and can cascade into outages. Performance tuning is the discipline of measuring first, then making targeted adjustments to CPU scheduling, memory management, I/O paths, and network stacks. Mastering these techniques transforms you from someone who reboots and hopes, into an engineer who pinpoints the exact bottleneck and resolves it with confidence.

## 1. Performance Analysis Fundamentals

### 1.1 USE Methodology

```
┌─────────────────────────────────────────────────────────────┐
│                USE Methodology (Brendan Gregg)               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Check for each resource:                                   │
│                                                             │
│  U - Utilization                                            │
│      How much is the resource being used?                   │
│      Example: CPU at 80% usage                              │
│                                                             │
│  S - Saturation                                             │
│      Are tasks waiting?                                     │
│      Example: 10 processes in run queue                     │
│                                                             │
│  E - Errors                                                 │
│      Are errors occurring?                                  │
│      Example: Network packet drops                          │
│                                                             │
│  Key resources:                                             │
│  • CPU: mpstat, vmstat, top                                │
│  • Memory: free, vmstat, /proc/meminfo                     │
│  • Disk I/O: iostat, iotop                                 │
│  • Network: netstat, ss, sar                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Basic Monitoring Tools

```bash
# top - real-time process monitoring
top
# Shortcuts: 1=per CPU, M=sort by memory, P=sort by CPU, k=kill

# htop - enhanced top
htop

# vmstat - virtual memory statistics
vmstat 1 5  # 1 second interval, 5 times
#  r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st
#  2  0      0 1234567 12345 234567    0    0     1     2  100  200  5  2 93  0  0
# r: processes waiting to run
# b: processes waiting for I/O
# si/so: swap in/out
# bi/bo: block in/out
# us/sy/id/wa: user/system/idle/wait

# mpstat - CPU statistics
mpstat -P ALL 1  # All CPUs, 1 second interval

# iostat - I/O statistics
iostat -x 1      # Extended info, 1 second interval

# sar - system activity report
sar -u 1 5       # CPU
sar -r 1 5       # Memory
sar -d 1 5       # Disk
sar -n DEV 1 5   # Network

# free - memory usage
free -h

# uptime - load average
uptime
# load average: 1.50, 1.20, 0.80  (1min, 5min, 15min)
```

### 1.3 sysctl Basics

```bash
# View current settings
sysctl -a                    # All settings
sysctl vm.swappiness         # Specific setting
cat /proc/sys/vm/swappiness  # Direct read

# Temporary change
sysctl -w vm.swappiness=10
# Or
echo 10 > /proc/sys/vm/swappiness

# Persistent configuration
# /etc/sysctl.conf or /etc/sysctl.d/*.conf
echo "vm.swappiness = 10" >> /etc/sysctl.d/99-custom.conf
sysctl -p /etc/sysctl.d/99-custom.conf  # Apply
sysctl --system  # Load all configuration files
```

---

## 2. CPU Tuning

### 2.1 CPU Information

```bash
# CPU information
lscpu
cat /proc/cpuinfo

# CPU frequency
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
cpupower frequency-info

# NUMA information
numactl --hardware
lscpu | grep NUMA
```

### 2.2 CPU Governor

```bash
# Check current governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Available governors
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors
# performance, powersave, userspace, ondemand, conservative, schedutil

# Change governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Using cpupower
cpupower frequency-set -g performance

# Persistent configuration (Ubuntu)
# /etc/default/cpufrequtils
GOVERNOR="performance"
```

### 2.3 Process Priority

```bash
# nice value (-20 to 19, lower is higher priority)
nice -n -10 ./high-priority-task
renice -n -10 -p <PID>

# Real-time scheduling
chrt -f 50 ./realtime-task  # FIFO, priority 50
chrt -r 50 ./realtime-task  # Round Robin

# CPU affinity
taskset -c 0,1 ./my-program  # Run on CPU 0, 1 only
taskset -cp 0-3 <PID>        # Change running process

# CPU limit with cgroups
# /sys/fs/cgroup/cpu/mygroup/
mkdir /sys/fs/cgroup/cpu/mygroup
echo 50000 > /sys/fs/cgroup/cpu/mygroup/cpu.cfs_quota_us  # 50% limit
echo <PID> > /sys/fs/cgroup/cpu/mygroup/cgroup.procs
```

### 2.4 CPU-related sysctl

```bash
# /etc/sysctl.d/99-cpu.conf

# Scheduler tuning
kernel.sched_min_granularity_ns = 10000000
kernel.sched_wakeup_granularity_ns = 15000000
kernel.sched_migration_cost_ns = 5000000

# Workload-specific optimization
# Server workload (throughput-oriented)
kernel.sched_autogroup_enabled = 0

# Desktop workload (responsiveness-oriented)
kernel.sched_autogroup_enabled = 1
```

---

## 3. Memory Tuning

### 3.1 Memory Information

```bash
# Memory usage
free -h
cat /proc/meminfo

# Per-process memory
ps aux --sort=-%mem | head
pmap -x <PID>

# Page cache status
cat /proc/meminfo | grep -E "Cached|Buffers|Dirty"

# NUMA memory
numastat
```

### 3.2 Swap Tuning

```bash
# swappiness (0-100, lower uses less swap)
sysctl -w vm.swappiness=10  # Server: 10, Desktop: 60

# Create swap file
dd if=/dev/zero of=/swapfile bs=1G count=4
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile

# Add to /etc/fstab
# /swapfile none swap sw 0 0

# Swap status
swapon --show
cat /proc/swaps
```

### 3.3 Memory-related sysctl

```bash
# /etc/sysctl.d/99-memory.conf

# Reduce swap usage
vm.swappiness = 10

# Dirty page ratio (write delay)
vm.dirty_ratio = 20              # Allow up to 20% of total memory dirty
vm.dirty_background_ratio = 5    # Start background flush at 5%

# Or absolute values
vm.dirty_bytes = 1073741824      # 1GB
vm.dirty_background_bytes = 268435456  # 256MB

# Cache pressure
vm.vfs_cache_pressure = 50       # Default 100, lower keeps cache

# OOM Killer tuning
vm.overcommit_memory = 0         # 0=heuristic, 1=always allow, 2=limit
vm.overcommit_ratio = 50         # Used when overcommit_memory=2

# Memory compaction
vm.compaction_proactiveness = 20

# Transparent Huge Pages
# /sys/kernel/mm/transparent_hugepage/enabled
# [always] madvise never
```

### 3.4 Cache Management

```bash
# Clear page cache (use with caution in production!)
sync
echo 1 > /proc/sys/vm/drop_caches  # Page cache
echo 2 > /proc/sys/vm/drop_caches  # dentries, inodes
echo 3 > /proc/sys/vm/drop_caches  # All

# Check file cache
vmtouch -v /path/to/file
fincore /path/to/file

# Per-process cache usage
cat /proc/<PID>/smaps | grep -E "^(Rss|Shared|Private)"
```

---

## 4. I/O Tuning

### 4.1 I/O Scheduler

```bash
# Check current scheduler
cat /sys/block/sda/queue/scheduler
# [mq-deadline] kyber bfq none

# Scheduler types
# - none: For NVMe SSD (NOOP)
# - mq-deadline: Deadline-based, server default
# - bfq: Budget Fair Queueing, desktop
# - kyber: For fast devices

# Change scheduler
echo mq-deadline > /sys/block/sda/queue/scheduler

# Persistent configuration (GRUB)
# /etc/default/grub
# GRUB_CMDLINE_LINUX="elevator=mq-deadline"
# update-grub

# Set via udev rules
# /etc/udev/rules.d/60-scheduler.rules
# ACTION=="add|change", KERNEL=="sd[a-z]", ATTR{queue/scheduler}="mq-deadline"
# ACTION=="add|change", KERNEL=="nvme[0-9]*", ATTR{queue/scheduler}="none"
```

### 4.2 Disk I/O Tuning

```bash
# Readahead
cat /sys/block/sda/queue/read_ahead_kb  # Default 128
echo 256 > /sys/block/sda/queue/read_ahead_kb

# Queue depth
cat /sys/block/sda/queue/nr_requests
echo 256 > /sys/block/sda/queue/nr_requests

# Maximum sectors
cat /sys/block/sda/queue/max_sectors_kb

# Enable SSD TRIM
fstrim -v /
# Or automatic TRIM (mount option: discard)
# /dev/sda1 / ext4 defaults,discard 0 1

# Periodic TRIM (recommended)
systemctl enable fstrim.timer
```

### 4.3 Filesystem Tuning

```bash
# ext4 mount options
# /etc/fstab
# noatime    - Don't update access time (performance gain)
# nodiratime - Don't update directory access time
# data=writeback - Journaling mode (risky but fast)
# barrier=0  - Disable write barrier (risky)
# commit=60  - Commit interval (seconds)

# XFS tuning
# logbufs=8 - Number of log buffers
# logbsize=256k - Log buffer size

# Filesystem information
tune2fs -l /dev/sda1  # ext4
xfs_info /dev/sda1    # XFS
```

### 4.4 I/O Priority

```bash
# ionice - I/O priority
ionice -c 3 command        # Idle
ionice -c 2 -n 0 command   # Best-effort, high priority
ionice -c 1 command        # Realtime (root only)

# Change running process
ionice -c 2 -n 7 -p <PID>  # Lower priority

# Check current I/O priority
ionice -p <PID>
```

---

## 5. Network Tuning

### 5.1 Network Information

```bash
# Interface information
ip link show
ethtool eth0

# Network statistics
ss -s
netstat -s
cat /proc/net/netstat

# Connection status
ss -tuln   # Listening ports
ss -tupn   # All connections
conntrack -L  # Connection tracking table
```

### 5.2 TCP Tuning

```bash
# /etc/sysctl.d/99-network.conf

# TCP buffer sizes
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.core.rmem_default = 1048576
net.core.wmem_default = 1048576

# TCP socket buffer (min, default, max)
net.ipv4.tcp_rmem = 4096 1048576 16777216
net.ipv4.tcp_wmem = 4096 1048576 16777216

# TCP backlog
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.core.netdev_max_backlog = 65535

# TIME_WAIT optimization
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_tw_reuse = 1

# TCP Keepalive
net.ipv4.tcp_keepalive_time = 600
net.ipv4.tcp_keepalive_intvl = 60
net.ipv4.tcp_keepalive_probes = 3

# TCP congestion control
net.ipv4.tcp_congestion_control = bbr  # Or cubic
net.core.default_qdisc = fq

# Port range
net.ipv4.ip_local_port_range = 1024 65535

# SYN cookies (SYN flood defense)
net.ipv4.tcp_syncookies = 1
```

### 5.3 High-Performance Web Server Configuration

```bash
# /etc/sysctl.d/99-webserver.conf

# File handle limits
fs.file-max = 2097152
fs.nr_open = 2097152

# Network stack
net.core.somaxconn = 65535
net.ipv4.tcp_max_tw_buckets = 2000000
net.ipv4.tcp_max_syn_backlog = 65535
net.core.netdev_max_backlog = 65535

# Buffers
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 12582912 16777216
net.ipv4.tcp_wmem = 4096 12582912 16777216

# TCP optimization
net.ipv4.tcp_slow_start_after_idle = 0
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_mtu_probing = 1

# BBR
net.ipv4.tcp_congestion_control = bbr
net.core.default_qdisc = fq
```

### 5.4 Connection Limits

```bash
# System limits
ulimit -n        # Current limit
ulimit -n 65535  # Change

# /etc/security/limits.conf
# * soft nofile 65535
# * hard nofile 65535

# systemd service limits
# [Service]
# LimitNOFILE=65535
```

---

## 6. Profiling Tools

### 6.1 perf Basics

```bash
# Install perf
apt install linux-tools-common linux-tools-$(uname -r)

# CPU profiling
perf stat ./my-program
perf stat -d ./my-program  # Detailed

# Sampling
perf record -g ./my-program
perf record -g -p <PID> -- sleep 30

# Analyze results
perf report
perf report --stdio

# Real-time monitoring
perf top
perf top -p <PID>

# System-wide
perf record -a -g -- sleep 10
```

### 6.2 Flamegraph

```bash
# Install FlameGraph tools
git clone https://github.com/brendangregg/FlameGraph

# Collect data with perf
perf record -g -p <PID> -- sleep 60

# Generate flamegraph
perf script | ./FlameGraph/stackcollapse-perf.pl | ./FlameGraph/flamegraph.pl > flame.svg

# Or all at once
perf record -F 99 -a -g -- sleep 60
perf script | \
  ./FlameGraph/stackcollapse-perf.pl | \
  ./FlameGraph/flamegraph.pl > flame.svg
```

### 6.3 strace/ltrace

```bash
# System call tracing
strace ./my-program
strace -p <PID>

# Specific system calls only
strace -e open,read,write ./my-program

# Time measurement
strace -T ./my-program    # Time per syscall
strace -c ./my-program    # Summary statistics

# Library call tracing
ltrace ./my-program
```

### 6.4 Other Tools

```bash
# bpftrace - eBPF-based tracing
bpftrace -e 'tracepoint:syscalls:sys_enter_open { printf("%s %s\n", comm, str(args->filename)); }'

# Memory profiling (Valgrind)
valgrind --tool=massif ./my-program
ms_print massif.out.*

# CPU profiling (Valgrind)
valgrind --tool=callgrind ./my-program
kcachegrind callgrind.out.*

# Benchmarking
stress-ng --cpu 4 --timeout 60s
fio --name=random-write --ioengine=libaio --iodepth=32 --rw=randwrite --bs=4k --direct=1 --size=1G --numjobs=4 --runtime=60
```

### 6.5 Performance Checklist

```bash
#!/bin/bash
# performance-check.sh

echo "=== System Information ==="
uname -a
uptime

echo -e "\n=== CPU ==="
lscpu | grep -E "^(CPU\(s\)|Thread|Core|Model name)"
mpstat 1 1

echo -e "\n=== Memory ==="
free -h
cat /proc/meminfo | grep -E "^(MemTotal|MemFree|Buffers|Cached|SwapTotal|SwapFree)"

echo -e "\n=== Disk I/O ==="
iostat -x 1 1

echo -e "\n=== Network ==="
ss -s
cat /proc/net/netstat | grep -E "^(Tcp|Udp)"

echo -e "\n=== Load Average ==="
cat /proc/loadavg

echo -e "\n=== Top Processes (CPU) ==="
ps aux --sort=-%cpu | head -5

echo -e "\n=== Top Processes (Memory) ==="
ps aux --sort=-%mem | head -5

echo -e "\n=== Open Files ==="
cat /proc/sys/fs/file-nr

echo -e "\n=== Network Connections ==="
ss -s
```

---

## 6.6 Deep Dive: perf and Flamegraph Profiling

The `perf` tool and flamegraphs form the gold standard for CPU profiling on Linux. While the earlier sections introduced the basic commands, this section dives into the methodology, event types, and interpretation skills needed to turn raw profiling data into actionable performance insights.

### Understanding perf Events

`perf` works by sampling or counting hardware and software events. Knowing which events to use is half the battle.

```bash
# List all available events on the current system
# Why: different kernels/CPUs expose different counters
perf list

# Hardware counters (PMU-based, very low overhead)
# - cycles             : CPU clock cycles consumed
# - instructions       : instructions retired (completed)
# - cache-references   : L3 cache lookups
# - cache-misses       : L3 cache misses (data not in cache → RAM fetch)
# - branch-misses      : branch prediction failures (pipeline stalls)

# Software events (kernel-level)
# - page-faults        : memory pages brought in from disk/swap
# - context-switches   : process/thread context switches
# - cpu-migrations     : process moved between CPUs

# Tracepoints (detailed kernel function tracing)
# - sched:sched_switch : scheduler context switch details
# - block:block_rq_issue : block device I/O request
```

### perf stat: Quick Performance Summary

```bash
# Get a high-level performance profile of a command
# Why: perf stat gives you the "executive summary" -- is the workload
# CPU-bound, memory-bound, or suffering from branch mispredictions?
perf stat ./my-program

# Example output and how to read it:
#  1,234,567,890  cycles              # Total CPU cycles
#  2,345,678,901  instructions        # 1.90 IPC (instructions per cycle)
#     12,345,678  cache-misses        # 5.2% of cache-references
#      1,234,567  branch-misses       # 0.8% of branches

# Key metrics to watch:
# - IPC < 1.0 → likely memory-bound (CPU waiting for data)
# - IPC > 2.0 → CPU-efficient, look elsewhere for bottlenecks
# - cache-miss ratio > 10% → poor data locality, consider restructuring
# - branch-miss ratio > 5% → consider branchless algorithms

# Detailed mode: adds L1/L2 cache, TLB stats
# Why: -d gives more granular insight into cache hierarchy
perf stat -d ./my-program

# Repeat measurement for statistical confidence
# Why: a single run may be noisy; 5 runs gives mean + stddev
perf stat -r 5 ./my-program
```

### perf record + perf report: Sampling Workflow

```bash
# Record CPU call stacks at 99 Hz for 30 seconds
# Why -F 99: avoids aliasing with 100Hz timer; common best practice
# Why -g: captures call graph (stack traces) for meaningful analysis
# Why -a: system-wide (all CPUs, all processes)
perf record -F 99 -a -g -- sleep 30

# For a specific process
# Why --: separates perf args from the command/PID
perf record -F 99 -g -p $(pgrep my-program) -- sleep 30

# Analyze the recording interactively
# Why: perf report opens a TUI where you can drill into hot functions
perf report

# Text-based output (useful for scripting or remote sessions)
# Why --stdio: no TUI, prints directly to stdout
perf report --stdio --sort=dso,symbol

# Show callers of a specific function
# Why: helps trace who is calling the expensive function
perf report --call-graph=callee --symbol-filter=malloc
```

### perf top: Live Monitoring

```bash
# Real-time view of hottest functions (system-wide)
# Why: perf top is like "top" but for functions -- shows where
# CPU cycles are being spent right now
perf top

# Monitor a specific process
perf top -p $(pgrep nginx)

# Show call graphs in live view
# Why -g: see not just which function is hot, but who called it
perf top -g

# Filter by specific event (e.g., cache misses)
# Why: helps find functions with poor cache behavior
perf top -e cache-misses
```

### Flamegraph Generation: Full Pipeline

Brendan Gregg's flamegraphs transform stack traces into an interactive SVG visualization. The workflow is a three-stage pipeline:

```
perf record → perf script → stackcollapse-perf.pl → flamegraph.pl → SVG
```

```bash
# Step 1: Clone the FlameGraph toolkit (one-time setup)
git clone https://github.com/brendangregg/FlameGraph /opt/FlameGraph

# Step 2: Record profiling data
# Why -F 99: sample at 99 Hz (not 100, to avoid lockstep with timers)
# Why -a: all CPUs for a complete system picture
# Why -g: call graph is essential -- without it, flamegraph has no stacks
perf record -F 99 -a -g -- sleep 60

# Step 3: Convert binary perf data to readable stack traces
# Why: perf script outputs human-readable text that the collapse tool parses
perf script > /tmp/perf.out

# Step 4: Collapse stacks into a single-line-per-stack format
# Why: stackcollapse counts identical stacks, producing "stack;stack;func count"
/opt/FlameGraph/stackcollapse-perf.pl /tmp/perf.out > /tmp/perf.folded

# Step 5: Generate the SVG flamegraph
# Why: flamegraph.pl creates an interactive SVG you can open in a browser
/opt/FlameGraph/flamegraph.pl /tmp/perf.folded > /tmp/flamegraph.svg

# Or combine steps 3-5 in one pipeline
perf script | /opt/FlameGraph/stackcollapse-perf.pl | \
  /opt/FlameGraph/flamegraph.pl > /tmp/flamegraph.svg

# Open in browser
xdg-open /tmp/flamegraph.svg  # or: open /tmp/flamegraph.svg (macOS)
```

### Reading Flamegraphs

A flamegraph encodes a wealth of information in a compact visual form:

```
┌──────────────────────────────────────────────────────────────────────┐
│                     How to Read a Flamegraph                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Y-axis (vertical): Call stack depth                                 │
│    - Bottom: entry point (e.g., main, _start)                        │
│    - Top: leaf function where CPU time is actually spent             │
│                                                                      │
│  X-axis (horizontal): NOT time! It is sorted alphabetically          │
│    - Width of a box = proportion of CPU time in that function        │
│      (including all its children)                                    │
│    - Wide box at the top = CPU hotspot (most actionable)             │
│                                                                      │
│  Colors: random warm palette (no meaning by default)                 │
│    - Some tools use color to distinguish: user vs kernel,            │
│      language runtime vs application code                            │
│                                                                      │
│  Interactivity (SVG in browser):                                     │
│    - Click a box to zoom into that subtree                           │
│    - Ctrl+F to search for a function name (matches highlight)        │
│    - Reset zoom by clicking "Reset Zoom" at the bottom               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Common Performance Patterns to Look For

| Pattern | What You See | Likely Cause | Action |
|---------|-------------|--------------|--------|
| **Tall narrow tower** | Deep call stack, thin width | Recursive algorithm | Consider iterative approach |
| **Wide plateau at top** | Single function consuming most CPU | Hot loop or expensive computation | Optimize the algorithm or data structure |
| **Wide `malloc`/`free`** | Memory allocation dominates | Excessive heap allocation | Use object pools or arena allocators |
| **Wide `__GI___libc_read`** | I/O system calls consuming CPU | I/O-bound workload | Add buffering, use async I/O |
| **Wide `spin_lock`** | Kernel lock contention | Lock contention in multi-threaded code | Reduce critical section size, use lock-free structures |
| **Sawtooth pattern** | Periodic spikes in GC/runtime functions | Garbage collection pauses | Tune GC parameters, reduce allocation rate |

### Off-CPU Flamegraphs

Standard flamegraphs show on-CPU time. Off-CPU flamegraphs show where threads are **waiting** (blocked on I/O, locks, sleep). Together they provide the complete picture.

```bash
# Record scheduler events to capture off-CPU time
# Why -e sched:sched_switch: captures every context switch
# This reveals what threads are blocked on
perf record -e sched:sched_switch -a -g -- sleep 30

# Generate off-CPU flamegraph (requires different collapse script)
perf script | /opt/FlameGraph/stackcollapse-perf.pl | \
  /opt/FlameGraph/flamegraph.pl --color=io --title="Off-CPU Flamegraph" \
  > /tmp/offcpu-flamegraph.svg
```

### Brendan Gregg's Performance Analysis Methodology

Gregg recommends a systematic approach rather than random tool usage:

```
1. USE Method (per resource: CPU, memory, disk, network)
   - Utilization → saturation → errors

2. Workload Characterization
   - Who is causing the load? (perf top, pidstat)
   - What type of work? (CPU, I/O, network)

3. Drill-Down Analysis
   - Start broad (perf stat), narrow down (perf record → flamegraph)
   - On-CPU flamegraph → find hot code paths
   - Off-CPU flamegraph → find blocking/waiting

4. Latency Analysis
   - perf trace (like strace but lower overhead)
   - bpftrace for custom latency histograms
```

---

## 7. Practice Exercises

### Exercise 1: Web Server Tuning
```bash
# Requirements:
# 1. Support 100,000 concurrent connections
# 2. TCP optimization (BBR, keepalive)
# 3. Increase file handle limits
# 4. Choose appropriate I/O scheduler

# Write sysctl configuration:
```

### Exercise 2: Database Server Tuning
```bash
# Requirements:
# 1. Memory optimization (low swappiness)
# 2. Disk I/O optimization
# 3. Dirty page management
# 4. CPU affinity configuration

# Write configuration and commands:
```

### Exercise 3: Performance Problem Diagnosis
```bash
# Scenario:
# List items to check sequentially when server becomes slow

# Diagnostic command list:
```

### Exercise 4: Flamegraph Analysis
```bash
# Requirements:
# 1. Write or select CPU-intensive program
# 2. Profile with perf
# 3. Generate flamegraph
# 4. Analyze bottlenecks

# Commands and analysis approach:
```

---

---

## References

- [Linux Performance](https://www.brendangregg.com/linuxperf.html)
- [Red Hat Performance Tuning Guide](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/8/html/monitoring_and_managing_system_status_and_performance/index)
- [kernel.org sysctl Documentation](https://www.kernel.org/doc/Documentation/sysctl/)
- [perf Examples](https://www.brendangregg.com/perf.html)

---

**Previous**: [Advanced systemd](./13_Systemd_Advanced.md) | **Next**: [Container Internals](./15_Container_Internals.md)
