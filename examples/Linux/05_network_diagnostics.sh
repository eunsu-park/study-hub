#!/usr/bin/env bash
# =============================================================================
# 05_network_diagnostics.sh - Network Diagnostics and Firewall Management
#
# PURPOSE: Demonstrates essential network troubleshooting commands (ss, ip,
#          traceroute, dig, nmap) and firewall rule management with iptables
#          and nftables. Safe to run — only read-only commands execute live;
#          firewall modifications are dry-run.
#
# USAGE:
#   ./05_network_diagnostics.sh [--connectivity|--dns|--ports|--firewall|--all]
#
# MODES:
#   --connectivity  Interface config, routing, traceroute
#   --dns           DNS resolution and debugging
#   --ports         Port scanning and connection analysis
#   --firewall      iptables/nftables rule examples
#   --all           Run all sections (default)
#
# CONCEPTS COVERED:
#   - L2/L3 diagnostics: ip link, ip addr, ip route
#   - TCP connection states and socket statistics (ss)
#   - DNS resolution chain: dig, nslookup, host
#   - Port scanning fundamentals (nmap)
#   - Stateful firewall with iptables and nftables
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

run_safe() {
    # Why: Read-only diagnostic commands are safe to execute live.
    # This helper runs the command and shows output directly.
    show_cmd "$1"
    eval "$1" 2>/dev/null || echo "  (command unavailable or insufficient permissions)"
}

run_or_dry() {
    show_cmd "$1"
    if [[ "${DRY_RUN}" == "false" ]]; then
        eval "$1"
    else
        echo -e "  ${RED}(dry-run — skipped)${RESET}"
    fi
}

# ---------------------------------------------------------------------------
# 1. Connectivity and Routing
# ---------------------------------------------------------------------------
demo_connectivity() {
    section "1. Connectivity & Routing"

    explain "Listing network interfaces with addresses:"
    # Why: 'ip addr' (or 'ip a') replaces the deprecated ifconfig.
    # It shows L2 (MAC) and L3 (IP) info together with scope and flags.
    if command -v ip &>/dev/null; then
        run_safe "ip -brief addr show"
    else
        # macOS fallback
        run_safe "ifconfig | grep -E 'flags|inet '"
    fi
    echo ""

    explain "Viewing the routing table:"
    # Why: The default route determines where packets go when no specific
    # route matches. 'ip route' shows all routes with metrics and devices.
    if command -v ip &>/dev/null; then
        run_safe "ip route show"
    else
        run_safe "netstat -rn"
    fi
    echo ""

    explain "Checking gateway reachability (ping 3 packets):"
    # Why: -c 3 limits to 3 packets (don't flood); -W 2 sets a 2-second
    # timeout per packet so the script doesn't hang on unreachable hosts.
    local gateway
    if command -v ip &>/dev/null; then
        gateway=$(ip route | grep default | awk '{print $3}' | head -1)
    else
        gateway=$(netstat -rn | grep default | awk '{print $2}' | head -1)
    fi
    if [[ -n "${gateway:-}" ]]; then
        run_safe "ping -c 3 -W 2 ${gateway}"
    else
        explain "(No default gateway found)"
    fi
    echo ""

    explain "Traceroute to a public host:"
    # Why: traceroute reveals each hop between source and destination,
    # helping identify where packet loss or latency occurs.
    if command -v traceroute &>/dev/null; then
        show_cmd "traceroute -m 10 8.8.8.8"
        explain "(Skipping live traceroute — can take 30+ seconds)"
    else
        show_cmd "traceroute -m 10 8.8.8.8  # Install: apt install traceroute"
    fi

    explain "MTU discovery:"
    # Why: Mismatched MTU causes mysterious packet loss. This finds the
    # largest packet that passes without fragmentation.
    show_cmd "ping -c 1 -M do -s 1472 8.8.8.8  # Test 1500-byte frames"
}

# ---------------------------------------------------------------------------
# 2. DNS Diagnostics
# ---------------------------------------------------------------------------
demo_dns() {
    section "2. DNS Diagnostics"

    explain "Basic DNS resolution:"
    if command -v dig &>/dev/null; then
        run_safe "dig +short example.com A"
        echo ""

        explain "Full query with timing and authority info:"
        # Why: +noall +answer shows just the answer section, which is
        # the resolved record. +stats adds query time for latency analysis.
        run_safe "dig example.com A +noall +answer +stats"
        echo ""

        explain "Querying a specific DNS server:"
        run_safe "dig @8.8.8.8 example.com A +short"
        echo ""

        explain "Reverse DNS lookup:"
        run_safe "dig -x 8.8.8.8 +short"
        echo ""

        explain "Tracing the DNS delegation chain:"
        # Why: +trace follows the resolution from root servers down,
        # showing exactly which nameserver answers at each level.
        show_cmd "dig +trace example.com"
        explain "(Skipping live trace — outputs many lines)"
    elif command -v host &>/dev/null; then
        run_safe "host example.com"
    else
        explain "(dig/host not installed — apt install dnsutils)"
    fi

    echo ""
    explain "Checking /etc/resolv.conf (system DNS config):"
    if [[ -f /etc/resolv.conf ]]; then
        # Why: resolv.conf determines which nameservers the system uses.
        # On modern systemd systems, this is often a symlink to resolved.
        run_safe "cat /etc/resolv.conf | grep -v '^#'"
    else
        explain "(/etc/resolv.conf not found)"
    fi
}

# ---------------------------------------------------------------------------
# 3. Port and Connection Analysis
# ---------------------------------------------------------------------------
demo_ports() {
    section "3. Port & Connection Analysis"

    explain "Listing all listening TCP ports:"
    # Why: ss (socket statistics) replaces the deprecated netstat.
    # -t=TCP, -l=listening, -n=numeric (skip DNS), -p=show process.
    if command -v ss &>/dev/null; then
        run_safe "ss -tlnp"
    else
        run_safe "netstat -tlnp 2>/dev/null || lsof -iTCP -sTCP:LISTEN -P -n"
    fi
    echo ""

    explain "TCP connection states (established, time-wait, etc.):"
    # Why: A high number of TIME_WAIT connections may indicate a connection
    # leak. CLOSE_WAIT means the remote closed but the app hasn't.
    if command -v ss &>/dev/null; then
        run_safe "ss -s"
    else
        run_safe "netstat -an | awk '/^tcp/ {print \$6}' | sort | uniq -c | sort -rn"
    fi
    echo ""

    explain "Connections to a specific port (e.g., 443):"
    if command -v ss &>/dev/null; then
        run_safe "ss -tn state established dport = :443"
    fi
    echo ""

    explain "Port scanning with nmap (if available):"
    # Why: nmap is the standard tool for network discovery. Common scans:
    #   -sT: TCP connect scan (safe, no root needed)
    #   -sS: SYN scan (stealthier, needs root)
    #   -sV: Version detection (identify service software)
    if command -v nmap &>/dev/null; then
        explain "Scanning localhost for open ports (TCP connect scan):"
        run_safe "nmap -sT -p 1-1024 127.0.0.1 --open"
    else
        show_cmd "nmap -sT -p 1-1024 127.0.0.1  # Install: apt install nmap"
    fi

    echo ""
    explain "Testing a specific port with nc (netcat):"
    show_cmd "nc -zv example.com 443 -w 3   # -z=scan only, -w=timeout"
}

# ---------------------------------------------------------------------------
# 4. Firewall Rules
# ---------------------------------------------------------------------------
demo_firewall() {
    section "4. Firewall Rules (iptables / nftables)"

    explain "iptables: Traditional Linux packet filter (still widely used)"
    echo ""

    explain "Viewing current rules:"
    show_cmd "iptables -L -n -v --line-numbers"
    echo ""

    explain "Common iptables rules (dry-run):"
    # Why: Rules are processed in order. The first matching rule wins.
    # This is why the order of -A (append) vs -I (insert) matters.

    echo -e "  ${BOLD}Allow established connections:${RESET}"
    run_or_dry "iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT"

    echo -e "  ${BOLD}Allow SSH:${RESET}"
    run_or_dry "iptables -A INPUT -p tcp --dport 22 -j ACCEPT"

    echo -e "  ${BOLD}Allow HTTP/HTTPS:${RESET}"
    run_or_dry "iptables -A INPUT -p tcp -m multiport --dports 80,443 -j ACCEPT"

    echo -e "  ${BOLD}Rate-limit SSH (brute-force protection):${RESET}"
    # Why: --seconds 60 --hitcount 4 means if 4+ connection attempts
    # arrive within 60 seconds, further packets are dropped.
    run_or_dry "iptables -A INPUT -p tcp --dport 22 -m recent --set --name SSH"
    run_or_dry "iptables -A INPUT -p tcp --dport 22 -m recent --update --seconds 60 --hitcount 4 --name SSH -j DROP"

    echo -e "  ${BOLD}Drop all other incoming traffic:${RESET}"
    run_or_dry "iptables -P INPUT DROP"
    echo ""

    explain "nftables: Modern replacement for iptables (since kernel 3.13)"
    echo ""
    explain "Example nftables ruleset:"
    cat <<'NFT'
  table inet filter {
      chain input {
          type filter hook input priority 0; policy drop;

          # Why: ct state tracks connection state in the kernel's conntrack
          # table. Accepting established/related packets is the foundation
          # of a stateful firewall.
          ct state established,related accept
          ct state invalid drop

          iif lo accept              # Loopback is always trusted
          tcp dport 22 accept        # SSH
          tcp dport { 80, 443 } accept  # Web

          # Rate limiting with nftables meter
          tcp dport 22 meter ssh-limit { ip saddr limit rate 3/minute } accept
      }
  }
NFT

    echo ""
    explain "Saving and restoring rules:"
    show_cmd "iptables-save > /etc/iptables/rules.v4    # Persist across reboots"
    show_cmd "nft list ruleset > /etc/nftables.conf      # nftables equivalent"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo -e "${BOLD}========================================${RESET}"
    echo -e "${BOLD} Network Diagnostics & Firewall Demo${RESET}"
    echo -e "${BOLD}========================================${RESET}"

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo -e "${RED}Firewall commands run in DRY-RUN mode.${RESET}"
        echo -e "Diagnostic commands (read-only) execute live.\n"
    fi

    local mode="${1:---all}"
    case "${mode}" in
        --connectivity) demo_connectivity ;;
        --dns)          demo_dns ;;
        --ports)        demo_ports ;;
        --firewall)     demo_firewall ;;
        --all)
            demo_connectivity
            demo_dns
            demo_ports
            demo_firewall
            ;;
        *) echo "Usage: $0 [--connectivity|--dns|--ports|--firewall|--all]"; exit 1 ;;
    esac

    echo -e "\n${GREEN}${BOLD}Done.${RESET}"
}

main "$@"
