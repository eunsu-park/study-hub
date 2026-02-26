# Advanced Networking

**Previous**: [Ansible Basics](./22_Ansible_Basics.md) | **Next**: [Cloud Integration](./24_Cloud_Integration.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Create and configure 802.1Q VLAN interfaces to segment network traffic
2. Set up NIC bonding with modes like active-backup and 802.3ad (LACP) for redundancy
3. Configure Linux bridge interfaces for connecting VMs and containers to the network
4. Write advanced iptables rules including stateful filtering, rate limiting, and NAT
5. Translate iptables configurations to the modern nftables syntax
6. Implement policy-based and multipath routing for multi-ISP environments
7. Shape traffic with the tc command using HTB queuing disciplines

---

## Table of Contents

1. [VLAN Configuration](#1-vlan-configuration)
2. [NIC Bonding](#2-nic-bonding)
3. [Bridge Networking](#3-bridge-networking)
4. [Advanced iptables](#4-advanced-iptables)
5. [nftables](#5-nftables)
6. [Advanced Routing](#6-advanced-routing)
7. [Traffic Control (tc)](#7-traffic-control-tc)

---

Modern data centers and cloud environments demand far more from networking than basic IP configuration. VLANs isolate broadcast domains, bonding provides link redundancy, bridges connect virtual machines to physical networks, and firewalls enforce security boundaries. Mastering these advanced networking concepts allows you to design infrastructure that is resilient, secure, and performant under real-world conditions.

## 1. VLAN Configuration

### VLAN Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Physical Switch                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Port 1    Port 2    Port 3    Port 4    Port 5     │   │
│  │  VLAN 10   VLAN 10   VLAN 20   VLAN 20   Trunk     │   │
│  └─────────────────────────────────────────────────────┘   │
│        │         │         │         │         │           │
└────────┼─────────┼─────────┼─────────┼─────────┼───────────┘
         │         │         │         │         │
     ┌───┴───┐ ┌───┴───┐ ┌───┴───┐ ┌───┴───┐ ┌───┴───┐
     │Host A │ │Host B │ │Host C │ │Host D │ │Server │
     │192.168│ │192.168│ │192.168│ │192.168│ │eth0.10│
     │.10.1  │ │.10.2  │ │.20.1  │ │.20.2  │ │eth0.20│
     └───────┘ └───────┘ └───────┘ └───────┘ └───────┘
```

### Creating VLAN Interfaces

```bash
# Check module
lsmod | grep 8021q

# Load module
sudo modprobe 8021q
echo "8021q" | sudo tee /etc/modules-load.d/8021q.conf

# Create VLAN interface (ip command)
sudo ip link add link eth0 name eth0.10 type vlan id 10
sudo ip link add link eth0 name eth0.20 type vlan id 20

# Assign IP addresses
sudo ip addr add 192.168.10.1/24 dev eth0.10
sudo ip addr add 192.168.20.1/24 dev eth0.20

# Bring interfaces up
sudo ip link set eth0.10 up
sudo ip link set eth0.20 up

# Verify VLAN
cat /proc/net/vlan/config
ip -d link show eth0.10
```

### Ubuntu Netplan Configuration

```yaml
# /etc/netplan/01-vlans.yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    eth0:
      dhcp4: no

  vlans:
    eth0.10:
      id: 10
      link: eth0
      addresses:
        - 192.168.10.1/24
      routes:
        - to: 10.0.0.0/8
          via: 192.168.10.254

    eth0.20:
      id: 20
      link: eth0
      addresses:
        - 192.168.20.1/24
```

```bash
sudo netplan apply
```

### RHEL/CentOS NetworkManager Configuration

```bash
# Create VLAN connection
sudo nmcli connection add type vlan con-name eth0.10 dev eth0 id 10
sudo nmcli connection modify eth0.10 ipv4.addresses 192.168.10.1/24
sudo nmcli connection modify eth0.10 ipv4.method manual
sudo nmcli connection up eth0.10

# Or use ifcfg file
# /etc/sysconfig/network-scripts/ifcfg-eth0.10
```

```ini
# /etc/sysconfig/network-scripts/ifcfg-eth0.10
DEVICE=eth0.10
BOOTPROTO=none
ONBOOT=yes
VLAN=yes
IPADDR=192.168.10.1
NETMASK=255.255.255.0
```

---

## 2. NIC Bonding

### Bonding Modes

| Mode | Name | Description | Switch Config |
|------|------|-------------|---------------|
| 0 | balance-rr | Round Robin | Not required |
| 1 | active-backup | Active-Backup | Not required |
| 2 | balance-xor | XOR Hash | Not required |
| 3 | broadcast | Broadcast | Not required |
| 4 | 802.3ad | LACP | LACP required |
| 5 | balance-tlb | Transmit Load Balancing | Not required |
| 6 | balance-alb | Adaptive Load Balancing | Not required |

### Ubuntu Netplan Bonding

```yaml
# /etc/netplan/01-bonding.yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    enp3s0:
      dhcp4: no
    enp4s0:
      dhcp4: no

  bonds:
    bond0:
      interfaces:
        - enp3s0
        - enp4s0
      addresses:
        - 192.168.1.100/24
      routes:
        - to: default
          via: 192.168.1.1
      parameters:
        mode: 802.3ad
        lacp-rate: fast
        mii-monitor-interval: 100
        transmit-hash-policy: layer3+4
```

### RHEL/CentOS NetworkManager Bonding

```bash
# Create bonding connection
sudo nmcli connection add type bond con-name bond0 ifname bond0 \
    bond.options "mode=802.3ad,miimon=100,lacp_rate=fast"

# Add slaves
sudo nmcli connection add type ethernet con-name bond0-slave1 \
    ifname enp3s0 master bond0
sudo nmcli connection add type ethernet con-name bond0-slave2 \
    ifname enp4s0 master bond0

# Configure IP
sudo nmcli connection modify bond0 ipv4.addresses 192.168.1.100/24
sudo nmcli connection modify bond0 ipv4.gateway 192.168.1.1
sudo nmcli connection modify bond0 ipv4.method manual

# Activate
sudo nmcli connection up bond0
```

### Checking Bonding Status

```bash
# Bonding status
cat /proc/net/bonding/bond0

# Interface information
ip -d link show bond0
ip link show master bond0

# Slave status
cat /sys/class/net/bond0/bonding/slaves
cat /sys/class/net/bond0/bonding/mode
```

---

## 3. Bridge Networking

### Creating a Bridge

```bash
# Create bridge
sudo ip link add name br0 type bridge
sudo ip link set br0 up

# Add interfaces
sudo ip link set eth0 master br0
sudo ip link set eth1 master br0

# Assign IP to bridge
sudo ip addr add 192.168.1.100/24 dev br0

# Check status
bridge link show
brctl show  # bridge-utils package
```

### Ubuntu Netplan Bridge

```yaml
# /etc/netplan/01-bridge.yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    enp3s0:
      dhcp4: no
    enp4s0:
      dhcp4: no

  bridges:
    br0:
      interfaces:
        - enp3s0
        - enp4s0
      addresses:
        - 192.168.1.100/24
      routes:
        - to: default
          via: 192.168.1.1
      parameters:
        stp: true
        forward-delay: 4
```

### Bridge + VLAN Combination

```yaml
# /etc/netplan/01-bridge-vlan.yaml
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: no

  vlans:
    eth0.10:
      id: 10
      link: eth0

  bridges:
    br-vlan10:
      interfaces:
        - eth0.10
      addresses:
        - 192.168.10.1/24
```

---

## 4. Advanced iptables

### iptables Chains and Tables

```
┌─────────────────────────────────────────────────────────────┐
│                    Packet Flow                              │
│                                                             │
│  PREROUTING ──▶ INPUT ──────▶ Local Process                │
│       │                              │                      │
│       │                              │                      │
│       ▼                              ▼                      │
│   FORWARD ◀───────────────────── OUTPUT                    │
│       │                              │                      │
│       └───────▶ POSTROUTING ◀───────┘                      │
│                      │                                      │
│                      ▼                                      │
│                   Network                                   │
└─────────────────────────────────────────────────────────────┘

Tables:
- filter: Packet filtering (INPUT, FORWARD, OUTPUT)
- nat: Address translation (PREROUTING, OUTPUT, POSTROUTING)
- mangle: Packet modification (all chains)
- raw: Connection tracking exclusion (PREROUTING, OUTPUT)
```

### Advanced Rule Examples

```bash
# Stateful filtering
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A INPUT -m state --state NEW -p tcp --dport 22 -j ACCEPT
iptables -A INPUT -m state --state INVALID -j DROP

# Connection limiting (connections per second)
iptables -A INPUT -p tcp --dport 80 -m connlimit --connlimit-above 50 -j REJECT

# Rate limiting (requests per minute)
iptables -A INPUT -p tcp --dport 22 -m recent --set --name SSH
iptables -A INPUT -p tcp --dport 22 -m recent --update --seconds 60 --hitcount 4 --name SSH -j DROP

# Port range
iptables -A INPUT -p tcp -m multiport --dports 80,443,8080:8090 -j ACCEPT

# IP range
iptables -A INPUT -s 192.168.1.0/24 -j ACCEPT
iptables -A INPUT -m iprange --src-range 10.0.0.1-10.0.0.100 -j ACCEPT

# Time-based rules
iptables -A INPUT -p tcp --dport 22 -m time --timestart 09:00 --timestop 18:00 --weekdays Mon,Tue,Wed,Thu,Fri -j ACCEPT
```

### NAT Configuration

```bash
# Enable IP forwarding
echo 1 > /proc/sys/net/ipv4/ip_forward
echo "net.ipv4.ip_forward = 1" >> /etc/sysctl.conf

# SNAT (Source NAT) - Internal → External
iptables -t nat -A POSTROUTING -s 192.168.1.0/24 -o eth0 -j MASQUERADE
# Or with static IP
iptables -t nat -A POSTROUTING -s 192.168.1.0/24 -o eth0 -j SNAT --to-source 203.0.113.1

# DNAT (Destination NAT) - Port forwarding
iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 80 -j DNAT --to-destination 192.168.1.10:8080
iptables -A FORWARD -p tcp -d 192.168.1.10 --dport 8080 -j ACCEPT
```

### Saving/Restoring iptables

```bash
# Save current rules
iptables-save > /etc/iptables/rules.v4
ip6tables-save > /etc/iptables/rules.v6

# Restore rules
iptables-restore < /etc/iptables/rules.v4

# Ubuntu: iptables-persistent package
sudo apt install iptables-persistent
sudo netfilter-persistent save

# RHEL/CentOS
sudo service iptables save
```

---

## 5. nftables

### nftables Basics

```bash
# Check nftables status
sudo nft list ruleset

# Create table
sudo nft add table inet filter

# Create chains
sudo nft add chain inet filter input { type filter hook input priority 0 \; policy drop \; }
sudo nft add chain inet filter forward { type filter hook forward priority 0 \; policy drop \; }
sudo nft add chain inet filter output { type filter hook output priority 0 \; policy accept \; }
```

### nftables Rules

```bash
# Basic rules
nft add rule inet filter input ct state established,related accept
nft add rule inet filter input ct state invalid drop
nft add rule inet filter input iif lo accept

# Allow SSH
nft add rule inet filter input tcp dport 22 accept

# Multiple ports
nft add rule inet filter input tcp dport { 80, 443, 8080 } accept

# IP range
nft add rule inet filter input ip saddr 192.168.1.0/24 accept

# Rate limiting
nft add rule inet filter input tcp dport 22 meter ssh-meter { ip saddr limit rate 3/minute } accept

# Logging
nft add rule inet filter input log prefix "INPUT DROP: " counter drop
```

### nftables Configuration File

```bash
# /etc/nftables.conf
#!/usr/sbin/nft -f

flush ruleset

table inet filter {
    chain input {
        type filter hook input priority 0; policy drop;

        # Allow established connections
        ct state established,related accept
        ct state invalid drop

        # Local interface
        iif lo accept

        # ICMP
        ip protocol icmp accept
        ip6 nexthdr icmpv6 accept

        # SSH
        tcp dport 22 accept

        # HTTP/HTTPS
        tcp dport { 80, 443 } accept

        # Log and drop
        log prefix "INPUT DROP: " counter drop
    }

    chain forward {
        type filter hook forward priority 0; policy drop;
        ct state established,related accept
    }

    chain output {
        type filter hook output priority 0; policy accept;
    }
}

table ip nat {
    chain prerouting {
        type nat hook prerouting priority -100;

        # Port forwarding
        tcp dport 80 dnat to 192.168.1.10:8080
    }

    chain postrouting {
        type nat hook postrouting priority 100;

        # Masquerade
        oifname "eth0" masquerade
    }
}
```

```bash
# Apply configuration
sudo nft -f /etc/nftables.conf

# Enable service
sudo systemctl enable nftables
```

### iptables vs nftables Comparison

| iptables | nftables |
|----------|----------|
| `iptables -A INPUT` | `nft add rule inet filter input` |
| `iptables -t nat` | `nft add table ip nat` |
| `--dport 80` | `tcp dport 80` |
| `-m multiport --dports 80,443` | `tcp dport { 80, 443 }` |
| `-m state --state ESTABLISHED` | `ct state established` |
| `-j ACCEPT` | `accept` |
| `-j MASQUERADE` | `masquerade` |

---

## 6. Advanced Routing

### Policy Routing

```bash
# Add routing tables
echo "100 isp1" >> /etc/iproute2/rt_tables
echo "200 isp2" >> /etc/iproute2/rt_tables

# Configure ISP1 routing table
ip route add default via 203.0.113.1 table isp1
ip route add 192.168.1.0/24 dev eth0 table isp1

# Configure ISP2 routing table
ip route add default via 198.51.100.1 table isp2
ip route add 192.168.1.0/24 dev eth0 table isp2

# Add rules (source-based)
ip rule add from 192.168.1.0/24 lookup isp1
ip rule add from 10.0.0.0/8 lookup isp2

# Verify rules
ip rule show
ip route show table isp1
```

### Multipath Routing

```bash
# ECMP (Equal Cost Multi-Path)
ip route add default \
    nexthop via 203.0.113.1 weight 1 \
    nexthop via 198.51.100.1 weight 1

# Weighted load balancing
ip route add default \
    nexthop via 203.0.113.1 weight 3 \
    nexthop via 198.51.100.1 weight 1
```

### Network Namespaces

```bash
# Create namespaces
ip netns add ns1
ip netns add ns2

# List namespaces
ip netns list

# Create veth pair
ip link add veth0 type veth peer name veth1

# Assign to namespaces
ip link set veth0 netns ns1
ip link set veth1 netns ns2

# Configure within namespaces
ip netns exec ns1 ip addr add 10.0.0.1/24 dev veth0
ip netns exec ns1 ip link set veth0 up
ip netns exec ns1 ip link set lo up

ip netns exec ns2 ip addr add 10.0.0.2/24 dev veth1
ip netns exec ns2 ip link set veth1 up
ip netns exec ns2 ip link set lo up

# Test connectivity
ip netns exec ns1 ping 10.0.0.2
```

---

## 7. Traffic Control (tc)

### Bandwidth Limiting

```bash
# Limit output bandwidth (1Mbit/s)
tc qdisc add dev eth0 root tbf rate 1mbit burst 32kbit latency 400ms

# Check configuration
tc qdisc show dev eth0

# Remove
tc qdisc del dev eth0 root
```

### HTB (Hierarchical Token Bucket)

```bash
# Create HTB qdisc
tc qdisc add dev eth0 root handle 1: htb default 30

# Total bandwidth class
tc class add dev eth0 parent 1: classid 1:1 htb rate 100mbit ceil 100mbit

# Subclasses (per service)
tc class add dev eth0 parent 1:1 classid 1:10 htb rate 50mbit ceil 100mbit  # web
tc class add dev eth0 parent 1:1 classid 1:20 htb rate 30mbit ceil 100mbit  # mail
tc class add dev eth0 parent 1:1 classid 1:30 htb rate 20mbit ceil 100mbit  # default

# Filters to classify traffic
tc filter add dev eth0 protocol ip parent 1:0 prio 1 u32 \
    match ip dport 80 0xffff flowid 1:10
tc filter add dev eth0 protocol ip parent 1:0 prio 1 u32 \
    match ip dport 443 0xffff flowid 1:10
tc filter add dev eth0 protocol ip parent 1:0 prio 2 u32 \
    match ip dport 25 0xffff flowid 1:20
```

### Delay and Packet Loss Simulation

```bash
# Add delay (100ms)
tc qdisc add dev eth0 root netem delay 100ms

# Delay + variation
tc qdisc add dev eth0 root netem delay 100ms 20ms distribution normal

# Packet loss (1%)
tc qdisc add dev eth0 root netem loss 1%

# Combined configuration
tc qdisc add dev eth0 root netem delay 100ms 20ms loss 1% duplicate 0.1%
```

---

## Practice Problems

### Problem 1: VLAN Configuration

Add VLAN 100 to eth0 interface and assign 192.168.100.1/24 address.

### Problem 2: nftables Firewall

Write nftables rules for the following conditions:
- Allow SSH(22), HTTP(80), HTTPS(443)
- Allow all traffic from 192.168.1.0/24
- Block and log remaining input traffic

### Problem 3: Traffic Control

Limit eth0 interface output bandwidth to 10Mbit/s.

---

## Answers

### Problem 1 Answer

```bash
# Create VLAN interface
sudo ip link add link eth0 name eth0.100 type vlan id 100

# Assign IP
sudo ip addr add 192.168.100.1/24 dev eth0.100

# Bring up
sudo ip link set eth0.100 up

# Verify
ip addr show eth0.100
```

### Problem 2 Answer

```bash
# /etc/nftables.conf
table inet filter {
    chain input {
        type filter hook input priority 0; policy drop;

        ct state established,related accept
        ct state invalid drop
        iif lo accept

        ip saddr 192.168.1.0/24 accept

        tcp dport { 22, 80, 443 } accept

        log prefix "INPUT DROP: " counter drop
    }

    chain forward {
        type filter hook forward priority 0; policy drop;
    }

    chain output {
        type filter hook output priority 0; policy accept;
    }
}
```

```bash
sudo nft -f /etc/nftables.conf
```

### Problem 3 Answer

```bash
# Use TBF (Token Bucket Filter)
sudo tc qdisc add dev eth0 root tbf rate 10mbit burst 32kbit latency 400ms

# Verify
tc qdisc show dev eth0
```

---

---

## References

- [Linux Advanced Routing & Traffic Control](https://lartc.org/)
- [nftables Wiki](https://wiki.nftables.org/)
- [Netplan Documentation](https://netplan.io/)
- `man ip`, `man nft`, `man tc`

---

**Previous**: [Ansible Basics](./22_Ansible_Basics.md) | **Next**: [Cloud Integration](./24_Cloud_Integration.md)
