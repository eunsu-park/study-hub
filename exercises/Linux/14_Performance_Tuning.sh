#!/bin/bash
# Exercises for Lesson 14: Performance Tuning
# Topic: Linux
# Solutions to practice problems from the lesson.

# === Exercise 1: sysctl Kernel Parameters Tuning ===
# Problem: Tune kernel parameters for a high-performance database server
#          using sysctl and understand the impact of each setting.
exercise_1() {
    echo "=== Exercise 1: sysctl Kernel Parameters Tuning ==="
    echo ""
    echo "Scenario: A PostgreSQL database server handling thousands of concurrent"
    echo "connections needs kernel-level tuning for memory, networking, and I/O."
    echo ""

    echo "--- Part A: Virtual Memory Tuning ---"
    echo "Solution: /etc/sysctl.d/99-database.conf"
    cat << 'SYSCTL'
  # --- Virtual Memory ---
  # Swappiness: 0-100, lower = prefer keeping data in RAM
  # Default 60 is too aggressive for database servers
  # 10 = only swap under heavy memory pressure
  vm.swappiness = 10

  # Dirty page ratio: % of RAM that can hold dirty (unwritten) pages
  # Higher = more write caching, risk of data loss on crash
  # Lower = more frequent flushes, better durability
  vm.dirty_ratio = 15
  vm.dirty_background_ratio = 5

  # Overcommit: 2 = strict (never overcommit), good for databases
  # Prevents OOM killer from killing the database process
  vm.overcommit_memory = 2
  vm.overcommit_ratio = 80    # Allow 80% of RAM + swap to be committed

  # Shared memory for PostgreSQL (max shared buffer size)
  # Set to at least shared_buffers + a margin
  kernel.shmmax = 8589934592    # 8 GB
  kernel.shmall = 2097152       # 8 GB / 4096 (page size)
SYSCTL
    echo ""
    echo "  Explanation:"
    echo "    vm.swappiness=10 keeps database pages in RAM as long as possible"
    echo "    vm.dirty_ratio=15 limits dirty page cache to 15% of RAM"
    echo "    vm.overcommit_memory=2 prevents the kernel from promising more memory than exists"
    echo "    This prevents the OOM killer from randomly killing processes"
    echo "    shmmax/shmall control System V shared memory (used by PostgreSQL)"
    echo ""

    echo "--- Part B: Network Stack Tuning ---"
    echo "Solution: Add to /etc/sysctl.d/99-database.conf"
    cat << 'NETWORK'
  # --- Network ---
  # Connection queue backlog
  net.core.somaxconn = 4096
  net.ipv4.tcp_max_syn_backlog = 4096

  # TCP keepalive (detect dead client connections faster)
  net.ipv4.tcp_keepalive_time = 120
  net.ipv4.tcp_keepalive_intvl = 30
  net.ipv4.tcp_keepalive_probes = 3

  # Disable SYN cookies in controlled environments (datacenter)
  # SYN cookies sacrifice TCP options for SYN flood protection
  net.ipv4.tcp_syncookies = 0

  # Local port range for outbound connections
  net.ipv4.ip_local_port_range = 10000 65535
NETWORK
    echo ""
    echo "  Explanation:"
    echo "    somaxconn=4096 handles burst connection storms to the database"
    echo "    Faster keepalive detects dead connections sooner (frees connection slots)"
    echo "    tcp_syncookies=0 is safe in controlled datacenter environments"
    echo "    Wider port range prevents 'cannot assign address' under heavy load"
    echo ""

    echo "--- Part C: Applying and Verifying sysctl Settings ---"
    echo "Solution:"
    echo "  # Apply all custom sysctl files"
    echo "  sudo sysctl --system"
    echo ""
    echo "  # Apply a specific file"
    echo "  sudo sysctl -p /etc/sysctl.d/99-database.conf"
    echo ""
    echo "  # Verify individual settings"
    echo "  sysctl vm.swappiness"
    echo "  sysctl net.core.somaxconn"
    echo ""
    echo "  # Show all current settings"
    echo "  sysctl -a 2>/dev/null | grep vm.dirty"
    echo ""
    echo "  # Temporary change (lost on reboot, useful for testing)"
    echo "  sudo sysctl -w vm.swappiness=10"
    echo ""

    # Safe read-only check
    echo "--- Current Values on This System ---"
    for param in vm.swappiness vm.dirty_ratio net.core.somaxconn; do
        value=$(sysctl -n "$param" 2>/dev/null)
        if [ -n "$value" ]; then
            echo "  $param = $value"
        fi
    done
    echo ""

    echo "  Explanation:"
    echo "    sysctl --system loads all files in /etc/sysctl.d/ in order"
    echo "    sysctl -w makes temporary changes (good for testing before persisting)"
    echo "    Files are loaded in lexicographic order; 99- prefix ensures loading last"
}

# === Exercise 2: I/O Scheduler and Disk Performance ===
# Problem: Configure I/O schedulers and optimize disk performance for
#          different workload types.
exercise_2() {
    echo "=== Exercise 2: I/O Scheduler and Disk Performance ==="
    echo ""
    echo "Scenario: Optimize disk I/O for a mixed-workload server: SSD for database"
    echo "storage and HDD for log/archive storage."
    echo ""

    echo "--- Part A: I/O Schedulers ---"
    echo "Solution:"
    echo "  # Check current scheduler for a device"
    echo "  cat /sys/block/sda/queue/scheduler"
    echo "  # Output example: [mq-deadline] kyber bfq none"
    echo "  # The one in brackets is the active scheduler"
    echo ""
    echo "  # Change scheduler temporarily"
    echo "  echo 'none' | sudo tee /sys/block/nvme0n1/queue/scheduler    # SSD: none (noop)"
    echo "  echo 'mq-deadline' | sudo tee /sys/block/sda/queue/scheduler # HDD: mq-deadline"
    echo ""
    echo "  # Persistent via udev rule: /etc/udev/rules.d/60-ioscheduler.rules"
    echo "  ACTION==\"add|change\", KERNEL==\"sd[a-z]\", ATTR{queue/rotational}==\"1\", ATTR{queue/scheduler}=\"mq-deadline\""
    echo "  ACTION==\"add|change\", KERNEL==\"nvme*\", ATTR{queue/rotational}==\"0\", ATTR{queue/scheduler}=\"none\""
    echo ""
    echo "  Scheduler choices:"
    echo "    none/noop  - No reordering, best for SSDs/NVMe (hardware handles optimization)"
    echo "    mq-deadline - Deadline-based, prevents starvation, good for HDDs and databases"
    echo "    bfq        - Budget Fair Queuing, best for interactive/desktop workloads"
    echo "    kyber      - Token-based, low overhead, good for fast devices"
    echo ""
    echo "  Explanation:"
    echo "    SSDs have no seek time, so reordering I/O requests provides no benefit"
    echo "    HDDs benefit from deadline scheduling to prevent request starvation"
    echo "    rotational==1 indicates HDD; rotational==0 indicates SSD"
    echo ""

    echo "--- Part B: Block Device Tuning ---"
    echo "Solution:"
    echo "  # Read-ahead: prefetch data for sequential reads"
    echo "  blockdev --getra /dev/sda                     # Show current read-ahead (sectors)"
    echo "  sudo blockdev --setra 2048 /dev/sda           # Set to 1MB (2048 * 512 bytes)"
    echo ""
    echo "  # Queue depth: max outstanding I/O requests"
    echo "  cat /sys/block/sda/queue/nr_requests"
    echo "  echo 512 | sudo tee /sys/block/sda/queue/nr_requests"
    echo ""
    echo "  # Filesystem mount options for performance"
    echo "  # /etc/fstab entry:"
    echo "  /dev/nvme0n1p1  /data  ext4  defaults,noatime,nodiratime,discard  0 2"
    echo ""
    echo "  Explanation:"
    echo "    Read-ahead: higher = better for sequential reads (streaming, backups)"
    echo "    Read-ahead: lower = better for random reads (databases)"
    echo "    noatime skips updating access timestamps (significant write reduction)"
    echo "    discard enables TRIM for SSDs (maintains performance over time)"
    echo "    nr_requests: higher allows more parallel I/O (good for NVMe/RAID)"
    echo ""

    echo "--- Part C: Benchmarking Disk Performance ---"
    echo "Solution:"
    echo "  # Sequential write test with dd"
    echo "  dd if=/dev/zero of=/tmp/testfile bs=1M count=1024 conv=fdatasync"
    echo ""
    echo "  # Random read/write test with fio"
    echo "  fio --name=randread --ioengine=libaio --iodepth=32 --rw=randread \\"
    echo "      --bs=4k --size=1G --numjobs=4 --runtime=60 --group_reporting"
    echo ""
    echo "  # Quick latency test"
    echo "  ioping -c 20 /data"
    echo ""
    echo "  Explanation:"
    echo "    dd gives a rough sequential throughput number (not for random I/O)"
    echo "    conv=fdatasync ensures data is flushed to disk before reporting speed"
    echo "    fio is the standard Linux I/O benchmark tool (install: apt install fio)"
    echo "    ioping measures per-request latency (like ping for storage)"
    echo "    Always test on the actual workload pattern (random vs sequential, read vs write)"
}

# === Exercise 3: CPU and Memory Profiling ===
# Problem: Profile application performance using perf, strace, and
#          related tools to identify bottlenecks.
exercise_3() {
    echo "=== Exercise 3: CPU and Memory Profiling ==="
    echo ""
    echo "Scenario: An application is consuming excessive CPU and memory."
    echo "Use profiling tools to identify the bottleneck functions and system calls."
    echo ""

    echo "--- Part A: CPU Profiling with perf ---"
    echo "Solution:"
    echo "  # Record CPU samples for a running process (30 seconds)"
    echo "  sudo perf record -g -p \$(pgrep myapp) -- sleep 30"
    echo ""
    echo "  # View the profiling report"
    echo "  sudo perf report                               # Interactive TUI"
    echo "  sudo perf report --stdio                       # Text output"
    echo ""
    echo "  # One-liner: top-like view of functions consuming CPU"
    echo "  sudo perf top -p \$(pgrep myapp)"
    echo ""
    echo "  # Count specific hardware events"
    echo "  sudo perf stat -e cache-misses,cache-references,instructions,cycles \\"
    echo "      -p \$(pgrep myapp) -- sleep 10"
    echo ""
    echo "  Explanation:"
    echo "    perf record samples the call stack at regular intervals (-g = call graphs)"
    echo "    perf report shows which functions consumed the most CPU samples"
    echo "    perf stat gives hardware counter summaries (cache misses, branch mispredictions)"
    echo "    High cache-miss ratio indicates poor data locality (memory access pattern issue)"
    echo "    Install: apt install linux-tools-\$(uname -r)"
    echo ""

    echo "--- Part B: System Call Tracing with strace ---"
    echo "Solution:"
    echo "  # Trace a running process (attach)"
    echo "  sudo strace -p \$(pgrep myapp) -c              # Summary of syscall counts and times"
    echo ""
    echo "  # Trace with timing information"
    echo "  sudo strace -p \$(pgrep myapp) -T -e trace=file  # File operations with duration"
    echo ""
    echo "  # Trace a command from start"
    echo "  strace -f -o /tmp/trace.log -e trace=network myapp   # Network calls with children (-f)"
    echo ""
    echo "  # Common trace filters:"
    echo "  echo \"  -e trace=file      # open, read, write, close, stat\""
    echo "  echo \"  -e trace=network   # socket, connect, send, recv\""
    echo "  echo \"  -e trace=process   # fork, exec, wait\""
    echo "  echo \"  -e trace=memory    # mmap, brk, mprotect\""
    echo ""
    echo "  Explanation:"
    echo "    strace -c is the first tool to reach for (minimal overhead, quick summary)"
    echo "    -T shows time spent in each syscall (identifies slow I/O or blocking calls)"
    echo "    -f follows child processes (essential for multi-process applications)"
    echo "    High counts of futex() calls often indicate lock contention"
    echo "    Many short read()/write() calls suggest buffering issues"
    echo ""

    echo "--- Part C: Memory Analysis ---"
    echo "Solution:"
    echo "  # Process memory map"
    echo "  pmap -x \$(pgrep myapp)                        # Detailed memory map"
    echo "  cat /proc/\$(pgrep myapp)/smaps_rollup          # Summarized memory usage"
    echo ""
    echo "  # System-wide memory consumers"
    echo "  ps aux --sort=-%mem | head -20                  # Top 20 by memory"
    echo "  smem -tk                                        # Per-process PSS (proportional set size)"
    echo ""
    echo "  # Memory leak detection patterns with valgrind"
    echo "  valgrind --leak-check=full --show-leak-kinds=all ./myapp"
    echo ""
    echo "  # /proc/meminfo key fields"
    echo "  echo \"  MemTotal:     Total physical RAM\""
    echo "  echo \"  MemAvailable:  Available for new allocations (best 'free' metric)\""
    echo "  echo \"  Buffers:       Filesystem metadata cache\""
    echo "  echo \"  Cached:        File content page cache\""
    echo "  echo \"  SwapTotal:     Total swap space\""
    echo "  echo \"  SwapFree:      Unused swap\""
    echo "  echo \"  Dirty:         Waiting to be written to disk\""
    echo ""
    echo "  Explanation:"
    echo "    pmap shows virtual vs resident memory per mapping (identify large allocations)"
    echo "    PSS (Proportional Set Size) splits shared memory proportionally among users"
    echo "    RSS (Resident Set Size) counts shared memory fully for each process"
    echo "    valgrind is heavy (~20x slowdown) — use in development, not production"
    echo "    For production, use: /proc/PID/smaps for non-intrusive analysis"
    echo ""

    echo "--- Verification ---"
    echo "  # Quick performance overview (one command)"
    echo "  sudo perf stat -a -- sleep 5    # System-wide hardware counters for 5 seconds"
    echo "  dmesg | grep -i 'oom\\|out of memory'   # Check for OOM killer activity"
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
echo "All exercises completed!"
