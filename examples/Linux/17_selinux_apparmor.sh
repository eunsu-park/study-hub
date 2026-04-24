#!/usr/bin/env bash
# =============================================================================
# 17_selinux_apparmor.sh — Mandatory Access Control (SELinux / AppArmor)
#
# PURPOSE: Mandatory Access Control (MAC) systems add a kernel-enforced layer
#          on top of ordinary Unix permissions (DAC). This script inspects
#          whatever MAC stack is running and explains what each state means.
#          Read-only — no policy files are modified.
#
# USAGE:
#   ./17_selinux_apparmor.sh [--detect|--selinux|--apparmor|--all]
# =============================================================================

set -euo pipefail

section() { printf "\n=== %s ===\n\n" "$1"; }
explain() { printf "[INFO] %s\n" "$1"; }
show()    { printf "[CMD]  %s\n" "$1"; }
have()    { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# 1. Detect which MAC stack is present
# ---------------------------------------------------------------------------
demo_detect() {
    section "1. Which MAC Stack is Active?"

    explain "Linux distros ship one of two mainstream MAC systems:"
    explain "  SELinux    — RHEL/Fedora/CentOS Stream default; label-based"
    explain "  AppArmor   — Ubuntu/SUSE default; path-based"
    explain "Both can be installed on any distro, but only one is typically active."

    if have getenforce; then
        echo "  SELinux detected — 'getenforce' is present"
    fi
    if have aa-status; then
        echo "  AppArmor detected — 'aa-status' is present"
    fi
    if [[ ! -d /sys/fs/selinux ]] && [[ ! -d /sys/module/apparmor ]]; then
        explain "Neither selinuxfs nor apparmor module found — this system has no MAC active."
        explain "(macOS, WSL without the LSM module, or a minimal container may show this.)"
    fi
}

# ---------------------------------------------------------------------------
# 2. SELinux inspection
# ---------------------------------------------------------------------------
demo_selinux() {
    section "2. SELinux Status (Read-Only)"

    if ! have getenforce; then
        explain "SELinux tools not installed — skipping."
        return 0
    fi

    explain "Three SELinux modes:"
    explain "  Enforcing  — policy violations are DENIED and logged"
    explain "  Permissive — violations are LOGGED but allowed (useful while authoring policy)"
    explain "  Disabled   — SELinux is off entirely"
    show "getenforce"
    getenforce

    show "sestatus"
    sestatus 2>/dev/null || echo "  (sestatus failed — may need install: sudo apt install policycoreutils or dnf install ...)"

    explain "Files have a SECURITY CONTEXT: user:role:type:level."
    explain "The 'type' is what most policies actually check (Type Enforcement)."
    show "ls -Z /etc/passwd"
    ls -Z /etc/passwd 2>/dev/null || echo "  (ls -Z unsupported here)"

    # Why the audit log matters: SELinux denials do not always show up in
    # application error messages — applications see "Permission denied" and
    # may attribute it to a file-mode issue. audit.log is the ground truth.
    explain "Denials are logged to /var/log/audit/audit.log (needs root):"
    show "sudo ausearch -m AVC --start recent 2>/dev/null | head"
    sudo -n ausearch -m AVC --start recent 2>/dev/null | head || echo "  (needs sudo or audit not installed)"
}

# ---------------------------------------------------------------------------
# 3. AppArmor inspection
# ---------------------------------------------------------------------------
demo_apparmor() {
    section "3. AppArmor Status (Read-Only)"

    if ! have aa-status; then
        explain "AppArmor tools not installed — skipping."
        return 0
    fi

    explain "AppArmor profiles can be in three modes:"
    explain "  enforce    — violations are denied"
    explain "  complain   — violations are logged but allowed"
    explain "  unconfined — program runs without a profile"
    show "sudo aa-status --summary 2>/dev/null"
    sudo -n aa-status --summary 2>/dev/null || aa-status --summary 2>/dev/null || echo "  (needs sudo)"

    explain "AppArmor profiles are PATH-based (unlike SELinux's label-based model):"
    show "ls /etc/apparmor.d/ | head"
    ls /etc/apparmor.d/ 2>/dev/null | head || true

    explain "Denials log to /var/log/kern.log or dmesg:"
    show "dmesg 2>/dev/null | grep -i apparmor | tail -5"
    dmesg 2>/dev/null | grep -i apparmor | tail -5 || echo "  (dmesg restricted or no apparmor events)"
}

# ---------------------------------------------------------------------------
main() {
    local mode="${1:---all}"

    # Early reminder that this is a read-only tour.
    explain "Read-only. No modes are changed; no policies are loaded or unloaded."

    case "$mode" in
        --detect)   demo_detect ;;
        --selinux)  demo_selinux ;;
        --apparmor) demo_apparmor ;;
        --all|*)
            demo_detect
            demo_selinux
            demo_apparmor
            ;;
    esac
}

main "$@"
