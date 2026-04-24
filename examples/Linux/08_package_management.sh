#!/usr/bin/env bash
# =============================================================================
# 08_package_management.sh — apt, dnf, pacman, zypper, brew (read-only tour)
#
# PURPOSE: Compares the major Linux package managers plus macOS Homebrew.
#          The script detects which tools are available and prints the
#          equivalent commands for "install / remove / search / list" on
#          each. All operations default to read-only query commands; install
#          / remove commands are ONLY shown as text, never executed, unless
#          ALLOW_WRITES=1 is set (still not recommended in this script).
#
# USAGE:
#   ./08_package_management.sh [--detect|--query|--cheatsheet|--all]
# =============================================================================

set -euo pipefail

section() { printf "\n=== %s ===\n\n" "$1"; }
explain() { printf "[INFO] %s\n" "$1"; }
show()    { printf "[CMD]  %s\n" "$1"; }

ALLOW_WRITES="${ALLOW_WRITES:-0}"   # never set to 1 here; shown only for didactic clarity

# ---------------------------------------------------------------------------
# 1. Detect package manager
# ---------------------------------------------------------------------------
detect_pm() {
    # Why: the same distro family can ship different tools (RHEL 8+ uses dnf,
    # older used yum; Alpine uses apk; macOS uses brew). Detecting lets a
    # script stay portable across environments.
    if   command -v apt-get >/dev/null 2>&1; then echo "apt"
    elif command -v dnf     >/dev/null 2>&1; then echo "dnf"
    elif command -v yum     >/dev/null 2>&1; then echo "yum"
    elif command -v pacman  >/dev/null 2>&1; then echo "pacman"
    elif command -v zypper  >/dev/null 2>&1; then echo "zypper"
    elif command -v apk     >/dev/null 2>&1; then echo "apk"
    elif command -v brew    >/dev/null 2>&1; then echo "brew"
    else echo "unknown"
    fi
}

demo_detect() {
    section "1. Detecting the Package Manager"

    local pm
    pm="$(detect_pm)"
    explain "This machine appears to use: ${pm}"

    explain "Detected by probing command -v for the usual binaries. A portable"
    explain "script can branch on the detection result to run the right tool."
}

# ---------------------------------------------------------------------------
# 2. Read-only queries — safe on any system
# ---------------------------------------------------------------------------
demo_query() {
    section "2. Read-only Queries (safe to run)"

    local pm
    pm="$(detect_pm)"

    case "$pm" in
        apt)
            show "dpkg -l | head -5            # installed packages (Debian/Ubuntu)"
            dpkg -l 2>/dev/null | head -5 || true
            show "apt list --installed 2>/dev/null | head -5"
            apt list --installed 2>/dev/null | head -5 || true
            ;;
        dnf|yum)
            show "rpm -qa | head -5            # installed packages (RHEL/Fedora)"
            rpm -qa 2>/dev/null | head -5 || true
            ;;
        pacman)
            show "pacman -Q | head -5          # installed packages (Arch)"
            pacman -Q 2>/dev/null | head -5 || true
            ;;
        zypper)
            show "rpm -qa | head -5            # installed packages (openSUSE)"
            rpm -qa 2>/dev/null | head -5 || true
            ;;
        apk)
            show "apk info | head -5           # installed packages (Alpine)"
            apk info 2>/dev/null | head -5 || true
            ;;
        brew)
            show "brew list | head -5          # installed formulae (macOS)"
            brew list 2>/dev/null | head -5 || true
            ;;
        *)
            explain "No known package manager detected — skipping query demo."
            ;;
    esac
}

# ---------------------------------------------------------------------------
# 3. Cheatsheet — side-by-side command equivalents
# ---------------------------------------------------------------------------
demo_cheatsheet() {
    section "3. Cheatsheet — Common Operations Across Managers"

    # A compact side-by-side reference. printf for alignment.
    printf "%-14s %-28s %-28s %-28s\n" "TASK" "apt (Debian/Ubuntu)" "dnf (RHEL/Fedora)" "pacman (Arch)"
    printf "%-14s %-28s %-28s %-28s\n" "-----" "-------------------" "------------------" "---------------"
    printf "%-14s %-28s %-28s %-28s\n" "refresh"   "apt update"              "dnf check-update"           "pacman -Sy"
    printf "%-14s %-28s %-28s %-28s\n" "upgrade"   "apt upgrade"             "dnf upgrade"                "pacman -Su"
    printf "%-14s %-28s %-28s %-28s\n" "install"   "apt install <pkg>"       "dnf install <pkg>"          "pacman -S <pkg>"
    printf "%-14s %-28s %-28s %-28s\n" "remove"    "apt remove <pkg>"        "dnf remove <pkg>"           "pacman -R <pkg>"
    printf "%-14s %-28s %-28s %-28s\n" "search"    "apt search <term>"       "dnf search <term>"          "pacman -Ss <term>"
    printf "%-14s %-28s %-28s %-28s\n" "info"      "apt show <pkg>"          "dnf info <pkg>"             "pacman -Si <pkg>"
    printf "%-14s %-28s %-28s %-28s\n" "list"      "dpkg -l | grep <pkg>"    "rpm -q <pkg>"               "pacman -Q <pkg>"
    printf "%-14s %-28s %-28s %-28s\n" "files"     "dpkg -L <pkg>"           "rpm -ql <pkg>"              "pacman -Ql <pkg>"
    printf "%-14s %-28s %-28s %-28s\n" "which-owns" "dpkg -S /path"          "rpm -qf /path"              "pacman -Qo /path"

    explain ""
    explain "macOS Homebrew uses its own vocabulary:"
    explain "  brew update / upgrade / install / uninstall / search / info / list"
    explain ""
    explain "Repo config locations (know where to look when a repo misbehaves):"
    explain "  apt:    /etc/apt/sources.list, /etc/apt/sources.list.d/"
    explain "  dnf:    /etc/yum.repos.d/*.repo"
    explain "  pacman: /etc/pacman.conf, /etc/pacman.d/mirrorlist"
}

# ---------------------------------------------------------------------------
main() {
    local mode="${1:---all}"

    # Safety note upfront — ensures the reader knows this script does not
    # mutate state, regardless of which path runs.
    explain "This script only runs READ-ONLY commands."
    explain "Install/remove commands are shown as text, never executed."

    case "$mode" in
        --detect)     demo_detect ;;
        --query)      demo_query ;;
        --cheatsheet) demo_cheatsheet ;;
        --all|*)
            demo_detect
            demo_query
            demo_cheatsheet
            ;;
    esac
}

main "$@"
