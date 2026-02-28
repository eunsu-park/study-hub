#!/usr/bin/env bash
# =============================================================================
# 08_process_cgroup_namespace.sh - Processes, Cgroups v2, and Namespaces
#
# PURPOSE: Demonstrates the Linux process model, cgroups v2 resource control,
#          and namespace isolation. These are the building blocks of containers.
#          Cgroup/namespace operations require root and are dry-run by default.
#
# USAGE:
#   ./08_process_cgroup_namespace.sh [--process|--cgroup|--namespace|--limits|--all]
#
# MODES:
#   --process    Process tree, signals, states
#   --cgroup     Cgroups v2 resource accounting and limits
#   --namespace  Linux namespaces (mount, PID, net, user, UTS, IPC)
#   --limits     ulimit, prlimit, and resource constraints
#   --all        Run all sections (default)
#
# CONCEPTS COVERED:
#   - Process lifecycle: fork, exec, wait, signals
#   - /proc filesystem and process inspection
#   - Cgroups v2: cpu, memory, io controllers
#   - Namespaces: the 7 namespace types and their purposes
#   - ulimit vs cgroup: per-process vs per-group resource limits
#   - How containers combine cgroups + namespaces + overlay fs
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

DRY_RUN="${DRY_RUN:-true}"

section() { echo -e "\n${BOLD}${CYAN}=== $1 ===${RESET}\n"; }
explain() { echo -e "${GREEN}[INFO]${RESET} $1"; }
show_cmd() { echo -e "${YELLOW}[CMD]${RESET} $1"; }
run_safe() { show_cmd "$1"; eval "$1" 2>/dev/null || echo "  (unavailable)"; }
run_or_dry() {
    show_cmd "$1"
    if [[ "${DRY_RUN}" == "false" ]]; then eval "$1"; else echo -e "  ${RED}(dry-run)${RESET}"; fi
}

# ---------------------------------------------------------------------------
# 1. Process Management
# ---------------------------------------------------------------------------
demo_process() {
    section "1. Process Management"

    explain "The process tree (init/systemd is PID 1):"
    # Why: Understanding the process hierarchy is fundamental. Every process
    # (except PID 1) has a parent. Orphaned children are re-parented to PID 1.
    if command -v pstree &>/dev/null; then
        run_safe "pstree -p | head -20"
    elif command -v ps &>/dev/null; then
        run_safe "ps axjf 2>/dev/null | head -20 || ps -ej | head -20"
    fi
    echo ""

    explain "Process states:"
    echo "  R — Running or runnable (on CPU or in run queue)"
    echo "  S — Sleeping (waiting for an event, e.g., I/O)"
    echo "  D — Uninterruptible sleep (usually disk I/O; cannot be killed)"
    echo "  T — Stopped (by signal or debugger)"
    echo "  Z — Zombie (exited but parent hasn't called wait())"
    echo ""
    # Why: D-state processes are a common cause of system hangs. They
    # cannot be killed (not even with SIGKILL) until the I/O completes.

    explain "Finding zombie processes:"
    show_cmd "ps aux | awk '\$8 ~ /Z/ {print \$0}'"
    echo "  Fix: Kill the parent process (zombies have already exited)"
    echo ""

    explain "The /proc filesystem (process introspection):"
    # Why: /proc is a virtual filesystem that exposes kernel data structures
    # as files. It's the source of truth for ps, top, and htop.
    local pid=$$
    echo "  Current shell PID: ${pid}"
    if [[ -d "/proc/${pid}" ]]; then
        run_safe "cat /proc/${pid}/status | grep -E '^(Name|State|Pid|PPid|Threads|VmRSS)'"
        echo ""
        run_safe "ls /proc/${pid}/fd | wc -l"
        explain "Above: number of open file descriptors for this process"
    else
        explain "(/proc not available — this is likely macOS)"
    fi
    echo ""

    explain "Signal handling:"
    echo "  SIGTERM (15) — Graceful shutdown (app cleans up, flushes buffers)"
    echo "  SIGKILL (9)  — Forced kill (kernel terminates immediately, no cleanup)"
    echo "  SIGHUP  (1)  — Reload config (by convention; e.g., nginx, sshd)"
    echo "  SIGUSR1 (10) — User-defined (apps use for log rotation, status dumps)"
    echo ""
    # Why: Always send SIGTERM first and wait. SIGKILL should be a last
    # resort because the process cannot clean up (temp files, locks, etc.).
    show_cmd "kill -SIGTERM <pid>     # Graceful stop"
    show_cmd "kill -SIGKILL <pid>     # Only if SIGTERM was ignored for >10s"
}

# ---------------------------------------------------------------------------
# 2. Cgroups v2
# ---------------------------------------------------------------------------
demo_cgroup() {
    section "2. Cgroups v2 — Resource Control"

    explain "Cgroups (control groups) limit and account for resource usage."
    explain "Cgroups v2 uses a unified hierarchy (unlike v1's per-controller trees)."
    echo ""

    explain "Cgroups v2 hierarchy:"
    echo "  /sys/fs/cgroup/              # Root cgroup"
    echo "  ├── system.slice/            # System services"
    echo "  │   └── nginx.service/       # Per-service cgroup"
    echo "  ├── user.slice/              # User sessions"
    echo "  │   └── user-1000.slice/     # Per-user cgroup"
    echo "  └── myapp.slice/             # Custom slice"
    echo ""

    # Why: Checking for cgroups v2 mount lets us show real data when possible.
    if [[ -d /sys/fs/cgroup/cgroup.controllers ]]; then
        explain "Cgroups v2 detected. Available controllers:"
        run_safe "cat /sys/fs/cgroup/cgroup.controllers"
    else
        explain "(Cgroups v2 unified hierarchy not detected on this system)"
    fi
    echo ""

    explain "Creating a cgroup and setting limits:"
    # Why: Each controller manages one resource type. Enabling a controller
    # in the parent's cgroup.subtree_control propagates it to children.
    run_or_dry "mkdir -p /sys/fs/cgroup/myapp.slice"
    run_or_dry "echo '+cpu +memory +io' > /sys/fs/cgroup/myapp.slice/cgroup.subtree_control"
    echo ""

    explain "Memory limits:"
    # Why: memory.max is a hard limit — if exceeded, the OOM killer
    # activates. memory.high is a soft limit — the kernel throttles
    # allocations (slows down rather than kills).
    run_or_dry "echo 512M > /sys/fs/cgroup/myapp.slice/memory.max    # Hard limit"
    run_or_dry "echo 256M > /sys/fs/cgroup/myapp.slice/memory.high   # Soft limit (throttle)"
    echo ""

    explain "CPU limits:"
    # Why: cpu.max uses a bandwidth model. '200000 1000000' means
    # 200ms out of every 1000ms period = 20% of one CPU core.
    run_or_dry "echo '200000 1000000' > /sys/fs/cgroup/myapp.slice/cpu.max  # 20% of 1 core"
    echo "  Format: \$MAX \$PERIOD (microseconds)"
    echo "  200000/1000000 = 20% of one CPU core"
    echo ""

    explain "I/O limits (per-device):"
    # Why: Without I/O limits, a log-heavy process can saturate the disk
    # and cause latency spikes for other services.
    run_or_dry "echo '8:0 rbps=10485760 wbps=5242880' > /sys/fs/cgroup/myapp.slice/io.max"
    echo "  8:0 = major:minor of the block device (check with lsblk)"
    echo "  rbps=10MB/s read, wbps=5MB/s write"
    echo ""

    explain "Adding a process to a cgroup:"
    run_or_dry "echo \$\$ > /sys/fs/cgroup/myapp.slice/cgroup.procs"
    echo ""

    explain "Monitoring cgroup resource usage:"
    show_cmd "cat /sys/fs/cgroup/myapp.slice/memory.current   # Current usage"
    show_cmd "cat /sys/fs/cgroup/myapp.slice/cpu.stat          # CPU time stats"
    show_cmd "cat /sys/fs/cgroup/myapp.slice/io.stat           # I/O bytes/ops"
}

# ---------------------------------------------------------------------------
# 3. Linux Namespaces
# ---------------------------------------------------------------------------
demo_namespace() {
    section "3. Linux Namespaces"

    explain "Namespaces provide process isolation — each namespace type"
    explain "virtualizes a different global resource."
    echo ""

    echo -e "${BOLD}  Namespace  │ Isolates                         │ Flag${RESET}"
    echo "  ──────────┼──────────────────────────────────┼──────────"
    echo "  Mount     │ Filesystem mount points          │ CLONE_NEWNS"
    echo "  PID       │ Process IDs (PID 1 per namespace)│ CLONE_NEWPID"
    echo "  Network   │ Network stack (interfaces, IPs)  │ CLONE_NEWNET"
    echo "  UTS       │ Hostname and domain name         │ CLONE_NEWUTS"
    echo "  IPC       │ System V IPC, POSIX message queue│ CLONE_NEWIPC"
    echo "  User      │ UIDs/GIDs (rootless containers)  │ CLONE_NEWUSER"
    echo "  Cgroup    │ Cgroup root directory             │ CLONE_NEWCGROUP"
    echo ""

    # Why: Containers = namespaces (isolation) + cgroups (limits) + overlay
    # filesystem (image layers). Understanding namespaces demystifies Docker.

    explain "Viewing current process namespaces:"
    if [[ -d /proc/self/ns ]]; then
        run_safe "ls -la /proc/self/ns/"
        echo ""
        explain "Each symlink points to a namespace inode. Processes sharing"
        explain "the same inode are in the same namespace."
    else
        explain "(/proc/self/ns not available on this platform)"
    fi
    echo ""

    explain "Creating an isolated environment with unshare:"
    # Why: unshare creates new namespaces without the complexity of
    # container runtimes. It's the building block Docker uses internally.
    show_cmd "unshare --mount --pid --fork --mount-proc bash"
    echo "  This gives you:"
    echo "    - Isolated PID namespace (your bash is PID 1)"
    echo "    - Isolated mount namespace (mounts don't affect host)"
    echo "    - /proc shows only processes in the new PID namespace"
    echo ""

    explain "Network namespace example (isolated network stack):"
    run_or_dry "ip netns add test_ns                           # Create namespace"
    run_or_dry "ip netns exec test_ns ip link                  # List interfaces inside"
    run_or_dry "ip link add veth0 type veth peer name veth1    # Create veth pair"
    run_or_dry "ip link set veth1 netns test_ns                # Move one end into ns"
    echo "  # Why: veth pairs are virtual ethernet cables connecting namespaces."
    echo "  # Docker uses this exact mechanism for container networking."
    run_or_dry "ip netns del test_ns                            # Clean up"
}

# ---------------------------------------------------------------------------
# 4. Resource Limits (ulimit / prlimit)
# ---------------------------------------------------------------------------
demo_limits() {
    section "4. Resource Limits — ulimit & prlimit"

    explain "ulimit: per-process resource limits (inherited by children)"
    echo ""

    explain "Current limits for this shell:"
    # Why: Soft limits can be raised by the user up to the hard limit.
    # Hard limits can only be raised by root. This two-tier model lets
    # admins set ceilings while users adjust within them.
    run_safe "ulimit -a"
    echo ""

    explain "Key limits explained:"
    echo "  -n (open files)    — Default 1024 is too low for servers"
    echo "                       Web servers need 10k-100k for concurrent connections"
    echo "  -u (max processes) — Prevents fork bombs"
    echo "  -v (virtual memory)— Caps address space per process"
    echo "  -c (core file size)— 0 disables core dumps (security vs debugging)"
    echo ""

    explain "Setting limits:"
    show_cmd "ulimit -n 65536                          # Raise open files (soft)"
    show_cmd "ulimit -Hn 65536                         # Raise hard limit (root only)"
    echo ""

    explain "Persistent limits via /etc/security/limits.conf:"
    echo "  # Format: <domain> <type> <item> <value>"
    echo "  appuser  soft  nofile  65536"
    echo "  appuser  hard  nofile  131072"
    echo "  @webteam soft  nproc   4096"
    echo "  *        hard  core    0          # Disable core dumps for all"
    echo ""

    explain "prlimit: inspect/set limits for a running process:"
    # Why: prlimit can modify limits of an already-running process,
    # unlike ulimit which only affects the current shell and children.
    show_cmd "prlimit --pid \$(pgrep nginx | head -1)"
    show_cmd "prlimit --pid 1234 --nofile=65536:131072  # Set soft:hard"
    echo ""

    explain "ulimit vs cgroups — when to use which:"
    echo "  ulimit   — Per-process; inherited by fork(); old but universal"
    echo "  cgroups  — Per-group of processes; hierarchical; more granular"
    echo "  Example: ulimit -m limits RSS per process, but a service with"
    echo "           10 workers could still use 10x that. Cgroup memory.max"
    echo "           limits the entire service collectively."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo -e "${BOLD}=================================================${RESET}"
    echo -e "${BOLD} Processes, Cgroups v2 & Namespaces Demo${RESET}"
    echo -e "${BOLD}=================================================${RESET}"

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo -e "${RED}Privileged commands run in DRY-RUN mode.${RESET}\n"
    fi

    local mode="${1:---all}"
    case "${mode}" in
        --process)   demo_process ;;
        --cgroup)    demo_cgroup ;;
        --namespace) demo_namespace ;;
        --limits)    demo_limits ;;
        --all)
            demo_process
            demo_cgroup
            demo_namespace
            demo_limits
            ;;
        *) echo "Usage: $0 [--process|--cgroup|--namespace|--limits|--all]"; exit 1 ;;
    esac

    echo -e "\n${GREEN}${BOLD}Done.${RESET}"
}

main "$@"
