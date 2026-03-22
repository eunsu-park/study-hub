#!/bin/bash
# Exercises for Lesson 10: Network Basics
# Topic: Linux
# Solutions to practice problems from the lesson.

# === Exercise 1: IP Configuration and Interface Management ===
# Problem: Configure network interfaces, manage IP addresses, and set up routing
#          using the modern 'ip' command suite.
exercise_1() {
    echo "=== Exercise 1: IP Configuration and Interface Management ==="
    echo ""
    echo "Scenario: You are provisioning a multi-homed server with two network"
    echo "interfaces: eth0 for public traffic and eth1 for a private backend network."
    echo ""

    echo "--- Part A: View and Configure IP Addresses ---"
    echo "Solution:"
    echo "  ip addr show                          # Show all interfaces and their addresses"
    echo "  ip -4 addr show eth0                  # Show only IPv4 addresses on eth0"
    echo "  ip addr add 10.0.1.50/24 dev eth1     # Add static IP to private interface"
    echo "  ip addr del 10.0.1.50/24 dev eth1     # Remove an IP address"
    echo "  ip link set eth1 up                   # Bring interface up"
    echo "  ip link set eth1 down                 # Bring interface down"
    echo ""
    echo "  Explanation:"
    echo "    'ip addr' replaces the deprecated 'ifconfig' command"
    echo "    /24 is CIDR notation = 255.255.255.0 subnet mask (254 usable hosts)"
    echo "    'ip link' manages the interface state (up/down, MTU, etc.)"
    echo "    These changes are temporary; they do not survive a reboot"
    echo ""

    echo "--- Part B: Persistent Network Configuration ---"
    echo "Solution (Netplan - Ubuntu 18.04+):"
    cat << 'NETPLAN'
  # /etc/netplan/01-static.yaml
  network:
    version: 2
    renderer: networkd
    ethernets:
      eth0:
        dhcp4: true                  # Public interface uses DHCP
      eth1:
        addresses:
          - 10.0.1.50/24             # Static IP on private network
        routes:
          - to: 10.0.0.0/16          # Route private subnet via gateway
            via: 10.0.1.1
        nameservers:
          addresses: [8.8.8.8, 8.8.4.4]
NETPLAN
    echo ""
    echo "  Apply with: sudo netplan apply"
    echo ""
    echo "  Explanation:"
    echo "    Netplan is the modern Ubuntu network configuration layer"
    echo "    'renderer: networkd' uses systemd-networkd (server); use NetworkManager for desktop"
    echo "    YAML format requires exact indentation (2 spaces)"
    echo ""

    echo "--- Part C: Routing Table Management ---"
    echo "Solution:"
    echo "  ip route show                                    # Display routing table"
    echo "  ip route add 192.168.0.0/16 via 10.0.1.1         # Add route to private subnets"
    echo "  ip route add default via 203.0.113.1 dev eth0    # Set default gateway"
    echo "  ip route del 192.168.0.0/16                      # Remove a route"
    echo "  ip route get 8.8.8.8                             # Show which route a packet would take"
    echo ""
    echo "  Explanation:"
    echo "    'default' route = 0.0.0.0/0, matches everything not covered by more specific routes"
    echo "    Most specific route wins (longest prefix match)"
    echo "    'ip route get' is invaluable for debugging routing decisions"
    echo ""

    # Safe read-only check
    echo "--- Current Interface Status ---"
    if command -v ip &>/dev/null; then
        ip -br addr show 2>/dev/null || echo "  (ip command not available)"
    else
        echo "  (ip command not available on this system)"
    fi
}

# === Exercise 2: SSH Key Management and Tunneling ===
# Problem: Set up SSH key-based authentication, configure SSH client,
#          and create SSH tunnels for secure service access.
exercise_2() {
    echo "=== Exercise 2: SSH Key Management and Tunneling ==="
    echo ""
    echo "Scenario: Set up secure, passwordless SSH access to production servers"
    echo "and create tunnels to access internal services."
    echo ""

    echo "--- Part A: SSH Key Generation and Deployment ---"
    echo "Solution:"
    echo "  # Generate Ed25519 key (recommended over RSA for modern systems)"
    echo "  ssh-keygen -t ed25519 -C \"admin@company.com\" -f ~/.ssh/id_ed25519_prod"
    echo ""
    echo "  # Copy public key to remote server"
    echo "  ssh-copy-id -i ~/.ssh/id_ed25519_prod.pub user@server.example.com"
    echo ""
    echo "  # Manually (if ssh-copy-id is unavailable):"
    echo "  cat ~/.ssh/id_ed25519_prod.pub | ssh user@server 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'"
    echo ""
    echo "  Explanation:"
    echo "    Ed25519 keys are shorter, faster, and more secure than RSA"
    echo "    -C adds a comment (usually email) for identification"
    echo "    -f specifies a custom key file (useful for multiple keys)"
    echo "    Permissions matter: ~/.ssh (700), authorized_keys (600), private key (600)"
    echo ""

    echo "--- Part B: SSH Client Configuration ---"
    echo "Solution:"
    cat << 'SSHCONFIG'
  # ~/.ssh/config
  Host prod-web
      HostName 203.0.113.10
      User deploy
      IdentityFile ~/.ssh/id_ed25519_prod
      Port 2222
      ForwardAgent no

  Host staging-*
      HostName %h.staging.internal
      User admin
      ProxyJump bastion

  Host bastion
      HostName bastion.example.com
      User jumpuser
      IdentityFile ~/.ssh/id_ed25519_bastion
      ControlMaster auto
      ControlPath ~/.ssh/sockets/%r@%h-%p
      ControlPersist 600
SSHCONFIG
    echo ""
    echo "  Usage: ssh prod-web    # Connects using all configured settings"
    echo ""
    echo "  Explanation:"
    echo "    Host aliases let you type 'ssh prod-web' instead of full connection details"
    echo "    ProxyJump chains through a bastion/jump host automatically"
    echo "    ControlMaster reuses TCP connections for faster subsequent SSH sessions"
    echo "    ControlPersist keeps the master connection alive for 600 seconds"
    echo ""

    echo "--- Part C: SSH Tunneling ---"
    echo "Solution:"
    echo "  # Local port forward: access remote database on localhost:5432"
    echo "  ssh -L 5432:db.internal:5432 bastion -N"
    echo ""
    echo "  # Remote port forward: expose local dev server to remote"
    echo "  ssh -R 8080:localhost:3000 remote-server -N"
    echo ""
    echo "  # Dynamic SOCKS proxy: route all traffic through SSH"
    echo "  ssh -D 1080 bastion -N"
    echo ""
    echo "  Explanation:"
    echo "    -L local_port:target_host:target_port  (forward local port to remote service)"
    echo "    -R remote_port:local_host:local_port   (expose local service on remote port)"
    echo "    -D port  (SOCKS5 proxy for all traffic, useful with browser proxy settings)"
    echo "    -N  (no remote command; just hold the tunnel open)"
    echo "    -f  (run in background after authentication)"
    echo ""

    echo "--- Verification ---"
    echo "  ssh -T git@github.com                   # Test SSH auth without shell"
    echo "  ssh -vvv user@server                    # Verbose debug output (3 levels)"
    echo "  ss -tlnp | grep 5432                    # Verify local tunnel is listening"
}

# === Exercise 3: Network Diagnostics ===
# Problem: Diagnose connectivity issues using standard networking tools.
exercise_3() {
    echo "=== Exercise 3: Network Diagnostics ==="
    echo ""
    echo "Scenario: Users report intermittent connectivity to an application server."
    echo "Systematically diagnose the issue from Layer 3 up."
    echo ""

    echo "--- Part A: Connectivity Testing with ping and traceroute ---"
    echo "Solution:"
    echo "  # Basic reachability test"
    echo "  ping -c 4 server.example.com          # Send 4 ICMP echo requests"
    echo ""
    echo "  # Path analysis"
    echo "  traceroute server.example.com          # Show route to destination (UDP-based)"
    echo "  traceroute -T server.example.com       # Use TCP SYN (works when ICMP is blocked)"
    echo "  mtr server.example.com                 # Continuous traceroute with statistics"
    echo ""
    echo "  Explanation:"
    echo "    ping tests basic reachability and round-trip time"
    echo "    traceroute shows each hop with latency (high latency or * indicates problem)"
    echo "    mtr combines ping + traceroute and runs continuously for pattern detection"
    echo "    -c 4 limits to 4 packets (prevents infinite ping)"
    echo ""

    echo "--- Part B: Port and Connection Analysis with ss and netstat ---"
    echo "Solution:"
    echo "  # Show all listening TCP ports with process names"
    echo "  ss -tlnp"
    echo ""
    echo "  # Show all established connections"
    echo "  ss -tn state established"
    echo ""
    echo "  # Count connections by state"
    echo "  ss -tan | awk '{print \$1}' | sort | uniq -c | sort -rn"
    echo ""
    echo "  # Check specific port"
    echo "  ss -tlnp | grep :8080"
    echo ""
    echo "  Explanation:"
    echo "    ss replaces the older netstat command (faster, more info)"
    echo "    -t = TCP, -u = UDP, -l = listening, -n = numeric (no DNS), -p = show process"
    echo "    Many TIME-WAIT connections may indicate connection reuse issues"
    echo "    Many SYN-RECV may indicate a SYN flood attack"
    echo ""

    echo "--- Part C: DNS Diagnostics with dig and nslookup ---"
    echo "Solution:"
    echo "  # Full DNS query"
    echo "  dig server.example.com"
    echo ""
    echo "  # Query specific record types"
    echo "  dig MX example.com                     # Mail exchange records"
    echo "  dig +short A server.example.com        # Just the IP address"
    echo "  dig @8.8.8.8 server.example.com        # Query specific DNS server"
    echo ""
    echo "  # Reverse DNS lookup"
    echo "  dig -x 203.0.113.10                    # PTR record lookup"
    echo ""
    echo "  # Trace delegation chain"
    echo "  dig +trace server.example.com          # Follow from root servers down"
    echo ""
    echo "  Explanation:"
    echo "    dig is the standard DNS debugging tool (replaces nslookup)"
    echo "    +short gives concise output; +trace shows full resolution chain"
    echo "    @8.8.8.8 bypasses local resolver (useful to confirm local DNS issues)"
    echo "    Check: ANSWER SECTION for results, AUTHORITY for nameservers, status: NXDOMAIN for not found"
    echo ""

    echo "--- Part D: Packet Capture for Deep Analysis ---"
    echo "Solution:"
    echo "  # Capture HTTP traffic on eth0"
    echo "  sudo tcpdump -i eth0 port 80 -w /tmp/capture.pcap -c 100"
    echo ""
    echo "  # Read capture file"
    echo "  tcpdump -r /tmp/capture.pcap -A       # ASCII output"
    echo ""
    echo "  Explanation:"
    echo "    -i specifies the interface; -w writes to file; -c limits packet count"
    echo "    -A shows packet content in ASCII (useful for HTTP debugging)"
    echo "    Use Wireshark for GUI analysis of .pcap files"
    echo "    Always use -c or -w to avoid filling terminal/disk"
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
echo "All exercises completed!"
