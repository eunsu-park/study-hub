#!/usr/bin/env bash
# =============================================================================
# 11_system_monitoring.sh — Observing CPU, Memory, I/O, and Processes
#
# PURPOSE: A tour of the classic observability tools. Every command here is
#          read-only — runs safely on any system, including production.
#          Where a tool is unavailable (e.g., vmstat on macOS), the script
#          prints a fallback or explanation instead of failing.
#
# USAGE:
#   ./11_system_monitoring.sh [--load|--cpu-mem|--processes|--io|--all]
# =============================================================================

set -euo pipefail

section() { printf "\n=== %s ===\n\n" "$1"; }
explain() { printf "[INFO] %s\n" "$1"; }
show()    { printf "[CMD]  %s\n" "$1"; }

have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# 1. System load and uptime
# ---------------------------------------------------------------------------
demo_load() {
    section "1. System Load and Uptime"

    explain "uptime — boot time, logged-in users, 1/5/15-minute load averages."
    explain "Load average is the average number of runnable + uninterruptible"
    explain "tasks in the run queue. Compare to CPU count for a meaningful read."
    show "uptime"
    uptime

    explain "nproc (or sysctl on macOS) — number of CPUs for context:"
    if have nproc; then
        show "nproc"
        nproc
    else
        show "sysctl -n hw.ncpu"
        sysctl -n hw.ncpu 2>/dev/null || echo "  (CPU count unavailable)"
    fi

    explain "Rule of thumb: load ≈ CPU count → saturated. Load >> CPU count → overloaded."
}

# ---------------------------------------------------------------------------
# 2. CPU and memory
# ---------------------------------------------------------------------------
demo_cpu_mem() {
    section "2. CPU and Memory"

    if have free; then
        explain "free -h — memory used/free/buffered/cached (Linux):"
        show "free -h"
        free -h
    else
        explain "On macOS, use vm_stat for memory page counts:"
        show "vm_stat | head -10"
        vm_stat 2>/dev/null | head -10 || echo "  (vm_stat unavailable)"
    fi

    if have vmstat; then
        explain "vmstat 1 3 — system activity snapshot, 1 sec interval, 3 samples:"
        show "vmstat 1 3"
        vmstat 1 3
    else
        explain "vmstat unavailable on this platform — skipping."
    fi

    if have mpstat; then
        explain "mpstat 1 2 — per-CPU utilization (user/sys/idle):"
        show "mpstat 1 2"
        mpstat 1 2
    fi
}

# ---------------------------------------------------------------------------
# 3. Processes — what is running
# ---------------------------------------------------------------------------
demo_processes() {
    section "3. Processes"

    explain "ps — snapshot of processes. Many flag styles; BSD (aux) is most common:"
    show "ps aux | head -6"
    ps aux 2>/dev/null | head -6 || ps -ef | head -6

    explain "Top N processes by CPU (portable pipeline):"
    show "ps aux --sort=-%cpu | head -6"
    ps aux --sort=-%cpu 2>/dev/null | head -6 || {
        # Fallback for BSD ps (macOS) that does not accept --sort
        ps -Ao pid,pcpu,comm | sort -k2 -rn | head -6
    }

    explain "Top N processes by resident memory:"
    show "ps aux --sort=-rss | head -6"
    ps aux --sort=-rss 2>/dev/null | head -6 || {
        ps -Ao pid,rss,comm | sort -k2 -rn | head -6
    }

    # Why 'top' / 'htop' in scripts is awkward: both are interactive. For
    # script-friendly snapshots, use 'ps' or 'top -b -n 1' (batch, one shot).
    if have top; then
        explain "top -b -n 1 — batch-mode single snapshot (Linux; macOS top differs):"
        show "top -b -n 1 2>/dev/null | head -12"
        top -b -n 1 2>/dev/null | head -12 || echo "  (top in batch mode unavailable — interactive only on this OS)"
    fi
}

# ---------------------------------------------------------------------------
# 4. Disk and network I/O
# ---------------------------------------------------------------------------
demo_io() {
    section "4. Disk and Network I/O"

    explain "df -h — free space per mounted filesystem:"
    show "df -h"
    df -h | head -8

    explain "du -sh — total size of a directory (read-only; follow with --max-depth on Linux):"
    show "du -sh . 2>/dev/null"
    du -sh . 2>/dev/null | head -1 || true

    if have iostat; then
        explain "iostat 1 2 — disk utilization, transfers/sec, await times:"
        show "iostat 1 2"
        iostat 1 2 2>/dev/null | head -15
    else
        explain "iostat unavailable — on Linux try 'apt install sysstat'."
    fi

    if have ss; then
        explain "ss -tuln — listening TCP/UDP sockets (replaces netstat on modern Linux):"
        show "ss -tuln | head -10"
        ss -tuln | head -10
    elif have netstat; then
        show "netstat -tuln | head -10"
        netstat -tuln 2>/dev/null | head -10 || true
    fi
}

# ---------------------------------------------------------------------------
main() {
    local mode="${1:---all}"
    case "$mode" in
        --load)      demo_load ;;
        --cpu-mem)   demo_cpu_mem ;;
        --processes) demo_processes ;;
        --io)        demo_io ;;
        --all|*)
            demo_load
            demo_cpu_mem
            demo_processes
            demo_io
            ;;
    esac
}

main "$@"
