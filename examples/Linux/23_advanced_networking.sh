#!/usr/bin/env bash
# =============================================================================
# 23_advanced_networking.sh — Interfaces, Routing, Namespaces (read-only)
#
# PURPOSE: Tour of the Linux networking stack beyond ping and ifconfig —
#          `ip` commands, routing tables, network namespaces, and the few
#          tools that every sysadmin reaches for when the usual pings fail.
#          Read-only: no interfaces are brought up/down, no routes changed.
#
# USAGE:
#   ./23_advanced_networking.sh [--iface|--route|--ns|--diag|--all]
# =============================================================================

set -euo pipefail

section() { printf "\n=== %s ===\n\n" "$1"; }
explain() { printf "[INFO] %s\n" "$1"; }
show()    { printf "[CMD]  %s\n" "$1"; }
have()    { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# 1. Interfaces
# ---------------------------------------------------------------------------
demo_iface() {
    section "1. Network Interfaces"

    explain "'ip addr' is the modern replacement for 'ifconfig'. Output per interface:"
    explain "  state (UP/DOWN), MTU, link-layer address, L3 addresses (IPv4/IPv6)."

    if have ip; then
        show "ip -br addr"
        ip -br addr 2>/dev/null || ip addr | head -20
    elif have ifconfig; then
        show "ifconfig -a"
        ifconfig -a 2>/dev/null | head -20
    else
        explain "Neither 'ip' nor 'ifconfig' found — unusual for a Linux or macOS system."
        return 0
    fi

    explain ""
    explain "Link statistics (errors, drops, overruns) for troubleshooting:"
    if have ip; then
        show "ip -s link show | head -15"
        ip -s link show 2>/dev/null | head -15
    fi
}

# ---------------------------------------------------------------------------
# 2. Routing
# ---------------------------------------------------------------------------
demo_route() {
    section "2. Routing Table"

    explain "The route table is consulted for every outgoing packet to choose:"
    explain "  - destination network (longest-prefix match)"
    explain "  - outgoing interface"
    explain "  - next-hop gateway (if not on a directly-attached link)"

    if have ip; then
        show "ip route"
        ip route 2>/dev/null || true
        echo
        explain "To trace which interface/gateway a particular destination would use:"
        show "ip route get 8.8.8.8 2>/dev/null"
        ip route get 8.8.8.8 2>/dev/null || true
    elif have netstat; then
        show "netstat -rn"
        netstat -rn 2>/dev/null | head -10
    fi

    explain ""
    explain "Linux supports MULTIPLE routing tables for policy routing."
    explain "List the policy rules (most systems just have 'main'):"
    if have ip; then
        show "ip rule"
        ip rule 2>/dev/null || true
    fi
}

# ---------------------------------------------------------------------------
# 3. Network namespaces
# ---------------------------------------------------------------------------
demo_ns() {
    section "3. Network Namespaces"

    explain "A network namespace is a separate copy of the entire networking stack —"
    explain "its own interfaces, routing table, and firewall rules. Containers (Docker,"
    explain "Kubernetes pods) use one namespace per container to isolate networking."

    if ! have ip; then
        explain "'ip' tool unavailable — network namespaces are a Linux feature."
        return 0
    fi

    show "ip netns list 2>/dev/null"
    ip netns list 2>/dev/null | head -10 || echo "  (may need sudo; an empty list is normal on non-container hosts)"

    explain ""
    explain "Create & enter a namespace (needs sudo) — shown only as commands:"
    show "  sudo ip netns add demo"
    show "  sudo ip netns exec demo ip link set lo up"
    show "  sudo ip netns exec demo ip addr"
    show "  sudo ip netns delete demo"
    explain "The container runtimes wrap these calls with additional veth-pair setup."
}

# ---------------------------------------------------------------------------
# 4. Diagnostics — in order of when to reach for each
# ---------------------------------------------------------------------------
demo_diag() {
    section "4. Diagnostic Tools (escalating)"

    explain "Standard playbook when 'ping fails': layer by layer from link to application."

    explain "  L2 link up?       — 'ip link show <iface>' or 'mii-tool'"
    explain "  L3 have IP/gw?    — 'ip -br addr' and 'ip route'"
    explain "  L3 reachable?     — 'ping', 'ping6' (enable via sysctl if ICMP blocked)"
    explain "  L3+ path?         — 'traceroute' / 'mtr' (combined traceroute+ping)"
    explain "  L4 port open?     — 'nc -zv <host> <port>' or 'nmap -p <port>'"
    explain "  L7 TLS/HTTP ok?   — 'curl -v', 'openssl s_client -connect host:443'"

    if have mtr; then
        show "mtr --report --report-cycles 1 8.8.8.8 2>/dev/null | head -10"
        mtr --report --report-cycles 1 8.8.8.8 2>/dev/null | head -10 || echo "  (mtr failed or no ICMP)"
    fi

    if have traceroute; then
        show "traceroute -n -q 1 -w 2 8.8.8.8 2>/dev/null | head -5"
        traceroute -n -q 1 -w 2 8.8.8.8 2>/dev/null | head -5 || echo "  (traceroute requires raw sockets or sudo)"
    fi

    if have nc; then
        show "nc -zv -w 2 google.com 443 2>&1"
        nc -zv -w 2 google.com 443 2>&1 || true
    fi
}

# ---------------------------------------------------------------------------
main() {
    local mode="${1:---all}"

    explain "Read-only: interfaces are not reconfigured, namespaces are not created."

    case "$mode" in
        --iface) demo_iface ;;
        --route) demo_route ;;
        --ns)    demo_ns ;;
        --diag)  demo_diag ;;
        --all|*)
            demo_iface
            demo_route
            demo_ns
            demo_diag
            ;;
    esac
}

main "$@"
