#!/usr/bin/env bash
# =============================================================================
# 20_kernel_management.sh — Kernel Modules, sysctl, and Runtime Info
#
# PURPOSE: Inspect the running kernel, its loaded modules, and its runtime
#          parameters (sysctl). All operations are read-only — no modules
#          are loaded or unloaded, no sysctl values are changed.
#
# USAGE:
#   ./20_kernel_management.sh [--version|--modules|--sysctl|--cmdline|--all]
# =============================================================================

set -euo pipefail

section() { printf "\n=== %s ===\n\n" "$1"; }
explain() { printf "[INFO] %s\n" "$1"; }
show()    { printf "[CMD]  %s\n" "$1"; }

# ---------------------------------------------------------------------------
# 1. Kernel version and build
# ---------------------------------------------------------------------------
demo_version() {
    section "1. Kernel Version and Build"

    explain "uname — kernel name, release, version, machine:"
    show "uname -a"
    uname -a

    explain "Full version string and build config metadata:"
    if [[ -r /proc/version ]]; then
        show "cat /proc/version"
        cat /proc/version
    fi

    explain "Kernel release identifies the source tree (useful for bug reports):"
    show "uname -r"
    uname -r
}

# ---------------------------------------------------------------------------
# 2. Loaded kernel modules
# ---------------------------------------------------------------------------
demo_modules() {
    section "2. Kernel Modules"

    explain "Modules are dynamically loadable pieces of the kernel — drivers,"
    explain "filesystems, network protocols. /proc/modules is the authoritative list."

    if [[ -r /proc/modules ]]; then
        show "lsmod | head -10"
        lsmod 2>/dev/null | head -10 || head -10 /proc/modules
    else
        explain "/proc/modules not available — likely macOS or a minimal container."
        return 0
    fi

    # modinfo tells you the license, description, and parameters of a module.
    # Useful when tracking down which kernel option controls a driver behavior.
    local first_mod
    first_mod="$(lsmod 2>/dev/null | awk 'NR==2 {print $1; exit}')"
    if [[ -n "${first_mod:-}" ]]; then
        show "modinfo $first_mod 2>/dev/null | head -8"
        modinfo "$first_mod" 2>/dev/null | head -8 || echo "  (modinfo requires sudo on some distros)"
    fi

    explain "modules.dep — the load-order dependency graph (resolved by modprobe):"
    local depfile="/lib/modules/$(uname -r)/modules.dep"
    if [[ -r "$depfile" ]]; then
        show "head -3 $depfile"
        head -3 "$depfile"
    fi
}

# ---------------------------------------------------------------------------
# 3. sysctl — runtime kernel parameters
# ---------------------------------------------------------------------------
demo_sysctl() {
    section "3. sysctl Runtime Parameters"

    if ! command -v sysctl >/dev/null 2>&1; then
        explain "sysctl unavailable — skipping."
        return 0
    fi

    explain "sysctl exposes kernel knobs under /proc/sys. A few of the most-asked-about:"

    # Selection of parameters that show up in tuning discussions
    for key in \
        kernel.hostname \
        kernel.osrelease \
        net.ipv4.ip_forward \
        net.ipv4.tcp_syncookies \
        net.core.somaxconn \
        vm.swappiness \
        fs.file-max; do
        show "sysctl $key 2>/dev/null"
        sysctl "$key" 2>/dev/null || echo "  $key: unavailable on this platform"
    done

    explain ""
    explain "Changes via 'sudo sysctl -w net.ipv4.ip_forward=1' are runtime-only."
    explain "To persist across reboot, add to /etc/sysctl.conf or /etc/sysctl.d/*.conf."
}

# ---------------------------------------------------------------------------
# 4. Boot-time kernel command line
# ---------------------------------------------------------------------------
demo_cmdline() {
    section "4. Boot-Time Kernel Command Line"

    if [[ -r /proc/cmdline ]]; then
        explain "/proc/cmdline — the arguments passed to the kernel by the bootloader."
        show "cat /proc/cmdline"
        cat /proc/cmdline

        explain ""
        explain "Common entries: 'root=UUID=...' (root FS), 'ro' (mount read-only first),"
        explain "'quiet' (silence boot messages), 'splash', 'nomodeset' (debug GPU),"
        explain "'mitigations=off' (disable CPU vulnerability mitigations — NOT recommended)."
    else
        explain "/proc/cmdline not available — not a Linux-like kernel."
    fi
}

# ---------------------------------------------------------------------------
main() {
    local mode="${1:---all}"

    explain "Read-only: no modules are loaded/unloaded, no sysctl values changed."

    case "$mode" in
        --version) demo_version ;;
        --modules) demo_modules ;;
        --sysctl)  demo_sysctl ;;
        --cmdline) demo_cmdline ;;
        --all|*)
            demo_version
            demo_modules
            demo_sysctl
            demo_cmdline
            ;;
    esac
}

main "$@"
