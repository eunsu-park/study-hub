#!/bin/bash
# Exercises for Lesson 23: Advanced Networking
# Topic: Linux
# Solutions to practice problems from the lesson.

# === Exercise 1: VLAN Configuration ===
# Problem: Add VLAN 100 to eth0 and assign 192.168.100.1/24.
exercise_1() {
    echo "=== Exercise 1: VLAN Configuration ==="
    echo ""
    echo "Scenario: Create a VLAN interface to segment network traffic."
    echo ""

    echo "Solution (using iproute2 commands):"
    echo ""
    echo "--- Step 1: Create the VLAN interface ---"
    echo "  sudo ip link add link eth0 name eth0.100 type vlan id 100"
    echo ""
    echo "  Breakdown:"
    echo "    link eth0       # Parent physical interface"
    echo "    name eth0.100   # Name for the VLAN interface (convention: parent.vlanid)"
    echo "    type vlan       # Interface type is 802.1Q VLAN"
    echo "    id 100          # VLAN ID (1-4094)"
    echo ""

    echo "--- Step 2: Assign IP address ---"
    echo "  sudo ip addr add 192.168.100.1/24 dev eth0.100"
    echo ""

    echo "--- Step 3: Bring the interface up ---"
    echo "  sudo ip link set eth0.100 up"
    echo ""

    echo "--- Step 4: Verify ---"
    echo "  ip addr show eth0.100"
    echo "  ip -d link show eth0.100    # Show VLAN details (id, protocol)"
    echo ""

    echo "What is a VLAN?"
    echo "  VLANs (Virtual LANs) logically segment a physical network."
    echo "  Frames on VLAN 100 get an 802.1Q tag (4-byte header with VLAN ID)."
    echo "  The switch only forwards tagged frames to ports in the same VLAN."
    echo "  This provides isolation without separate physical networks."
    echo ""

    echo "Making it persistent (Netplan - Ubuntu):"
    cat << 'NETPLAN'
# /etc/netplan/01-vlans.yaml
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: no
  vlans:
    eth0.100:
      id: 100
      link: eth0
      addresses:
        - 192.168.100.1/24
NETPLAN
    echo ""
    echo "  sudo netplan apply   # Apply the configuration"
    echo ""

    echo "Note: The physical switch port connected to this host must be configured"
    echo "as a trunk port carrying VLAN 100 for this to work."
}

# === Exercise 2: nftables Firewall ===
# Problem: Write nftables rules allowing SSH/HTTP/HTTPS, 192.168.1.0/24, logging drops.
exercise_2() {
    echo "=== Exercise 2: nftables Firewall Rules ==="
    echo ""
    echo "Scenario: Configure a stateful firewall using nftables (successor to iptables)."
    echo ""

    echo "Solution: /etc/nftables.conf"
    echo ""
    cat << 'NFTABLES'
#!/usr/sbin/nft -f
# /etc/nftables.conf
# nftables firewall configuration

# Flush existing rules (clean start)
flush ruleset

table inet filter {
    # 'inet' = applies to both IPv4 and IPv6

    chain input {
        # Default policy: DROP everything not explicitly allowed
        type filter hook input priority 0; policy drop;

        # --- Stateful tracking ---
        # Allow packets belonging to established/related connections
        # This means once a connection is allowed, response packets flow freely
        ct state established,related accept

        # Drop invalid packets (malformed, out-of-state)
        ct state invalid drop

        # Allow loopback interface (local processes talking to each other)
        iif lo accept

        # --- Trusted subnet ---
        # Allow ALL traffic from the local network
        ip saddr 192.168.1.0/24 accept

        # --- Service ports ---
        # Allow SSH(22), HTTP(80), HTTPS(443) from anywhere
        tcp dport { 22, 80, 443 } accept

        # --- Default deny with logging ---
        # Log and drop everything else
        # Logs appear in: journalctl -k -g "INPUT DROP"
        log prefix "INPUT DROP: " counter drop
    }

    chain forward {
        # Drop all forwarded traffic (this host is not a router)
        type filter hook forward priority 0; policy drop;
    }

    chain output {
        # Allow all outbound traffic (trust local processes)
        type filter hook output priority 0; policy accept;
    }
}
NFTABLES
    echo ""

    echo "To apply and verify:"
    echo "  sudo nft -f /etc/nftables.conf        # Load the ruleset"
    echo "  sudo nft list ruleset                  # View active rules"
    echo "  sudo nft list chain inet filter input  # View specific chain"
    echo "  sudo systemctl enable nftables         # Persist across reboots"
    echo ""

    echo "nftables vs iptables:"
    echo "  nftables                         iptables"
    echo "  -------                          --------"
    echo "  Single tool (nft)                Multiple tools (iptables, ip6tables, ebtables)"
    echo "  Atomic rule loading              Rule-by-rule loading"
    echo "  Sets for grouping ({22,80,443})  Requires separate rules per port"
    echo "  inet family (IPv4+IPv6)          Separate tables per protocol"
    echo "  Better performance               More overhead with large rulesets"
    echo ""

    echo "Key nftables concepts:"
    echo "  table   = container for chains (families: ip, ip6, inet, arp, bridge)"
    echo "  chain   = list of rules with a hook point (input, output, forward)"
    echo "  rule    = match + action (accept, drop, reject, log, counter)"
    echo "  set     = reusable collection of values ({ 22, 80, 443 })"
    echo "  ct      = connection tracking (stateful inspection)"
}

# === Exercise 3: Traffic Control ===
# Problem: Limit eth0 output bandwidth to 10Mbit/s.
exercise_3() {
    echo "=== Exercise 3: Traffic Control (Bandwidth Limiting) ==="
    echo ""
    echo "Scenario: Limit outbound bandwidth on an interface using tc (traffic control)."
    echo ""

    echo "Solution (Token Bucket Filter):"
    echo "  sudo tc qdisc add dev eth0 root tbf rate 10mbit burst 32kbit latency 400ms"
    echo ""
    echo "  Parameter breakdown:"
    echo "    qdisc          = Queueing Discipline (packet scheduling algorithm)"
    echo "    add             = Add a new qdisc (use 'change' to modify, 'del' to remove)"
    echo "    dev eth0        = Apply to this network interface"
    echo "    root            = Attach at the root (egress/outbound)"
    echo "    tbf             = Token Bucket Filter algorithm"
    echo ""
    echo "    rate 10mbit     = Average bandwidth limit (sustained rate)"
    echo "    burst 32kbit    = Maximum burst size allowed above the rate"
    echo "                      Packets queue in a 'bucket' of tokens"
    echo "                      Larger burst = more tolerance for bursty traffic"
    echo "    latency 400ms   = Maximum time a packet can wait in the queue"
    echo "                      Packets waiting longer are dropped"
    echo ""

    echo "How TBF works:"
    echo "  Think of it as a bucket that fills with tokens at 'rate' speed."
    echo "  Each packet needs tokens to pass through."
    echo "  If the bucket is full (burst), packets pass immediately."
    echo "  If the bucket is empty, packets wait until tokens accumulate."
    echo "  If they wait too long (latency), they're dropped."
    echo ""

    echo "Verify and manage:"
    echo "  tc qdisc show dev eth0                # Show current qdiscs"
    echo "  tc -s qdisc show dev eth0             # Show with statistics"
    echo "  sudo tc qdisc del dev eth0 root       # Remove all qdiscs (restore default)"
    echo ""

    echo "Alternative: HTB (Hierarchical Token Bucket) for more complex scenarios:"
    echo ""
    cat << 'HTB'
# HTB allows multiple classes with different rates
# Example: Web traffic gets 8Mbit, everything else gets 2Mbit

# Add root qdisc
sudo tc qdisc add dev eth0 root handle 1: htb default 20

# Parent class (total bandwidth)
sudo tc class add dev eth0 parent 1: classid 1:1 htb rate 10mbit

# Web traffic class (80% of bandwidth)
sudo tc class add dev eth0 parent 1:1 classid 1:10 htb rate 8mbit ceil 10mbit

# Default class (20% of bandwidth)
sudo tc class add dev eth0 parent 1:1 classid 1:20 htb rate 2mbit ceil 10mbit

# Filter: send HTTP/HTTPS to the web class
sudo tc filter add dev eth0 parent 1: protocol ip u32 \
    match ip dport 80 0xffff flowid 1:10
sudo tc filter add dev eth0 parent 1: protocol ip u32 \
    match ip dport 443 0xffff flowid 1:10
HTB
    echo ""
    echo "  HTB 'ceil' = maximum rate a class can borrow when other classes are idle."
    echo "  This allows web traffic to burst to 10Mbit when default class isn't busy."
    echo ""

    echo "Note: tc only controls EGRESS (outbound) traffic."
    echo "For ingress (inbound) limiting, use 'tc qdisc add dev eth0 ingress' with"
    echo "policing, or shape at the upstream router."
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
echo "All exercises completed!"
