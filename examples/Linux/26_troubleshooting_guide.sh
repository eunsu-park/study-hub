#!/usr/bin/env bash
# =============================================================================
# 26_troubleshooting_guide.sh — A Systematic Diagnosis Checklist
#
# PURPOSE: Walks through the first-hour triage steps for "something is
#          broken." Each section asks a question, shows the command that
#          answers it, and explains what the output means. Read-only on
#          every system — safe to run anywhere.
#
# USAGE:
#   ./26_troubleshooting_guide.sh [--process|--logs|--network|--disk|--kernel|--all]
# =============================================================================

set -euo pipefail

section() { printf "\n=== %s ===\n\n" "$1"; }
question() { printf "\n[?] %s\n" "$1"; }
explain() { printf "[INFO] %s\n" "$1"; }
show()    { printf "[CMD]  %s\n" "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# 1. Process: is it even running?
# ---------------------------------------------------------------------------
demo_process() {
    section "1. Is the Process Running?"

    question "Is the expected process alive?"
    explain "Swap in the real name you are looking for (e.g., nginx, postgres)."
    show "pgrep -laf bash | head -5"
    pgrep -laf bash 2>/dev/null | head -5 || ps aux | grep -v grep | grep bash | head -5

    question "How much CPU/memory is it using?"
    show "ps -o pid,pcpu,pmem,rss,comm -p \$(pgrep -f bash | head -1)"
    local pid
    pid="$(pgrep -f bash | head -1 || echo $$)"
    ps -o pid,pcpu,pmem,rss,comm -p "$pid" 2>/dev/null || true

    question "Is it stuck on a syscall?"
    explain "strace on Linux (or dtruss on macOS) attaches to a live process and"
    explain "shows the syscalls it is issuing. Requires sudo in most cases."
    if have strace; then
        show "sudo strace -p <PID> -c -e trace=read,write,poll   # summarize syscalls"
    elif have dtruss; then
        show "sudo dtruss -p <PID>   # macOS equivalent"
    else
        explain "Neither strace nor dtruss is available here — install for deep tracing."
    fi
}

# ---------------------------------------------------------------------------
# 2. Logs: what does the system say?
# ---------------------------------------------------------------------------
demo_logs() {
    section "2. What Do the Logs Say?"

    question "Recent systemd-journal entries?"
    if have journalctl; then
        show "journalctl -xe --no-pager | tail -10"
        journalctl -xe --no-pager 2>/dev/null | tail -10 || echo "  (journalctl access restricted — try sudo)"
    else
        explain "journalctl unavailable — likely a non-systemd or non-Linux host."
    fi

    question "Legacy syslog tail?"
    local logfile=""
    for candidate in /var/log/syslog /var/log/messages /var/log/system.log; do
        if [[ -r "$candidate" ]]; then
            logfile="$candidate"
            break
        fi
    done
    if [[ -n "$logfile" ]]; then
        show "tail -5 $logfile"
        tail -5 "$logfile" 2>/dev/null || true
    else
        explain "No readable /var/log file at typical paths (may need sudo)."
    fi

    question "What processes are writing most to disk?"
    explain "Look for the PID whose /proc/<pid>/io shows growing write_bytes"
    explain "(Linux-only; on macOS use fs_usage)."
    if [[ -d /proc ]]; then
        show "cat /proc/\$\$/io   # io counters for this shell"
        cat "/proc/$$/io" 2>/dev/null || echo "  (io counters restricted)"
    fi
}

# ---------------------------------------------------------------------------
# 3. Network: who is listening, who is connecting?
# ---------------------------------------------------------------------------
demo_network() {
    section "3. Network Reachability"

    question "Which ports are listening?"
    if have ss; then
        show "ss -tlnp 2>/dev/null | head -10"
        ss -tlnp 2>/dev/null | head -10 || ss -tln | head -10
    elif have netstat; then
        show "netstat -tlnp 2>/dev/null | head -10"
        netstat -tlnp 2>/dev/null | head -10 || netstat -tln | head -10
    fi

    question "Can we reach the outside?"
    explain "Prefer getent / dig over ping for DNS sanity — some firewalls block ICMP."
    if have getent; then
        show "getent hosts example.com"
        getent hosts example.com 2>/dev/null || echo "  (DNS lookup failed)"
    elif have dig; then
        show "dig +short example.com"
        dig +short example.com 2>/dev/null || echo "  (DNS lookup failed)"
    fi

    question "What does curl -v see?"
    show "curl -sS -o /dev/null -w '  http_code=%{http_code} time=%{time_total}s\\n' https://example.com"
    curl -sS -o /dev/null -w '  http_code=%{http_code} time=%{time_total}s\n' https://example.com 2>/dev/null || echo "  (curl unavailable or no internet)"
}

# ---------------------------------------------------------------------------
# 4. Disk: is something out of space?
# ---------------------------------------------------------------------------
demo_disk() {
    section "4. Disk Space and Inodes"

    question "Any filesystem near full?"
    show "df -h"
    df -h | head -10

    # Inode exhaustion is sneaky — df -h shows plenty of space, but df -i
    # reveals you've run out of inodes. Classic with many small files.
    question "Inode exhaustion (many-small-file symptom)?"
    show "df -i | head -10"
    df -i 2>/dev/null | head -10 || echo "  (-i may not be supported here)"

    question "Largest directories under home (top 5)?"
    show "du -sh ~/* 2>/dev/null | sort -h | tail -5"
    du -sh ~/* 2>/dev/null | sort -h | tail -5 || true
}

# ---------------------------------------------------------------------------
# 5. Kernel messages: what did the kernel notice?
# ---------------------------------------------------------------------------
demo_kernel() {
    section "5. Kernel Messages"

    question "Anything recent from dmesg?"
    # Why: OOM-killer, I/O errors, USB disconnects, driver panics — all here.
    if have dmesg; then
        show "dmesg --color=never 2>/dev/null | tail -10"
        dmesg --color=never 2>/dev/null | tail -10 || echo "  (dmesg access restricted — try sudo)"
    else
        explain "dmesg unavailable on this platform."
    fi
}

# ---------------------------------------------------------------------------
main() {
    local mode="${1:---all}"
    case "$mode" in
        --process) demo_process ;;
        --logs)    demo_logs ;;
        --network) demo_network ;;
        --disk)    demo_disk ;;
        --kernel)  demo_kernel ;;
        --all|*)
            demo_process
            demo_logs
            demo_network
            demo_disk
            demo_kernel
            ;;
    esac
}

main "$@"
