#!/usr/bin/env bash
# =============================================================================
# 12_security_and_firewall.sh — Firewall Inspection and SSH Hardening (read-only)
#
# PURPOSE: Inspects firewall rules (iptables / nftables / ufw / pf) and
#          surveys SSH hardening settings. Read-only: no rules are added,
#          removed, or changed. Demonstrates how to see what IS configured
#          before making changes.
#
# USAGE:
#   ./12_security_and_firewall.sh [--detect|--rules|--ssh|--all]
# =============================================================================

set -euo pipefail

section() { printf "\n=== %s ===\n\n" "$1"; }
explain() { printf "[INFO] %s\n" "$1"; }
show()    { printf "[CMD]  %s\n" "$1"; }
have()    { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# 1. Detect which firewall stack is in use
# ---------------------------------------------------------------------------
demo_detect() {
    section "1. Which Firewall Stack is Active?"

    explain "A system may have multiple tools installed, but typically ONE active:"
    explain "  nftables   — modern Linux kernel firewall (successor to iptables)"
    explain "  iptables   — classic Linux; often a shim over nftables on newer kernels"
    explain "  ufw        — Ubuntu/Debian friendly front-end to iptables/nftables"
    explain "  firewalld  — RHEL/Fedora; zone-based abstraction"
    explain "  pf         — BSD / macOS packet filter"

    for tool in nft iptables ufw firewall-cmd pfctl; do
        if have "$tool"; then
            echo "  found: $tool — $(command -v $tool)"
        fi
    done
}

# ---------------------------------------------------------------------------
# 2. Read current rules (requires sudo for most; we just show commands)
# ---------------------------------------------------------------------------
demo_rules() {
    section "2. Reading Current Rules"

    explain "Every check below is read-only. Many need sudo; errors are shown"
    explain "as informational rather than fatal."

    if have ufw; then
        show "sudo ufw status verbose"
        sudo -n ufw status verbose 2>/dev/null || echo "  (needs sudo; run interactively)"
    fi

    if have iptables; then
        show "sudo iptables -L -n -v --line-numbers   # dump chains"
        sudo -n iptables -L -n -v --line-numbers 2>/dev/null | head -15 || echo "  (needs sudo)"
    fi

    if have nft; then
        show "sudo nft list ruleset"
        sudo -n nft list ruleset 2>/dev/null | head -15 || echo "  (needs sudo)"
    fi

    if have firewall-cmd; then
        show "firewall-cmd --list-all"
        firewall-cmd --list-all 2>/dev/null | head -15 || echo "  (firewalld inactive or needs sudo)"
    fi

    if have pfctl; then
        show "sudo pfctl -s rules   # macOS / BSD"
        sudo -n pfctl -s rules 2>/dev/null | head -15 || echo "  (needs sudo)"
    fi

    explain "Principle: default DROP on INPUT, ALLOW on OUTPUT, plus explicit allows"
    explain "for ssh (22), http (80), https (443) as needed. 'Allow all' is a red flag."
}

# ---------------------------------------------------------------------------
# 3. SSH hardening — look at config without changing it
# ---------------------------------------------------------------------------
demo_ssh() {
    section "3. SSH Hardening — Reading sshd_config"

    local cfg="/etc/ssh/sshd_config"
    if [[ ! -r "$cfg" ]]; then
        explain "sshd_config not readable here (normal on macOS end-user accounts,"
        explain "or without sudo on a server). Showing recommended lines instead."
    else
        explain "Current settings for the hardening-relevant directives:"
        show "grep -Ei '^(PermitRootLogin|PasswordAuthentication|PubkeyAuthentication|Port|AllowUsers|MaxAuthTries|X11Forwarding|ClientAliveInterval)' $cfg"
        grep -Ei '^(PermitRootLogin|PasswordAuthentication|PubkeyAuthentication|Port|AllowUsers|MaxAuthTries|X11Forwarding|ClientAliveInterval)' "$cfg" 2>/dev/null || echo "  (none of those directives explicitly set — defaults apply)"
    fi

    explain ""
    explain "Recommended defaults for internet-facing servers:"
    cat <<'EOF'
  PermitRootLogin          no
  PasswordAuthentication   no       # keys only
  PubkeyAuthentication     yes
  MaxAuthTries             3
  ClientAliveInterval      300      # disconnect idle sessions after 5 min
  AllowUsers               alice bob   # explicit allowlist
  X11Forwarding            no       # unless desktop forwarding is actually used
EOF

    # Why keys > passwords: password brute-forcing is cheap at internet scale.
    # Keys (Ed25519) are effectively immune to guessing, and fail2ban /
    # rate-limiting further protects the login port.
    explain ""
    explain "Additional layers: fail2ban (temp-ban brute forcers) and"
    explain "firewall rules limiting source IPs for port 22."
}

# ---------------------------------------------------------------------------
main() {
    local mode="${1:---all}"
    case "$mode" in
        --detect) demo_detect ;;
        --rules)  demo_rules ;;
        --ssh)    demo_ssh ;;
        --all|*)
            demo_detect
            demo_rules
            demo_ssh
            ;;
    esac
}

main "$@"
