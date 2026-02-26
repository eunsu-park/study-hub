# Network Basics

**Previous**: [Shell Scripting](./09_Shell_Scripting.md) | **Next**: [System Monitoring](./11_System_Monitoring.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain IP addressing, subnetting, and common port assignments
2. Check and configure network interfaces using ip addr, ip link, and ip route
3. Test network connectivity with ping, traceroute, and mtr
4. Perform DNS lookups using dig, nslookup, and host
5. Inspect open ports and socket connections with ss, netstat, and lsof
6. Connect to remote systems securely using SSH with key-based authentication
7. Transfer files between systems using scp, rsync, and sftp
8. Configure persistent network settings with Netplan (Ubuntu) and NetworkManager (CentOS)

---

Networking is what makes Linux servers useful -- without it, they are just expensive space heaters. Whether you are diagnosing why a web application is unreachable, setting up SSH access for remote administration, or configuring a static IP for a production server, networking commands are among the most frequently used tools in a system administrator's toolkit.

## 1. Basic Network Concepts

### IP Address

```
IPv4: 192.168.1.100
      └─┬─┘ └─┬─┘
     Network  Host

IPv6: 2001:0db8:85a3:0000:0000:8a2e:0370:7334
```

### Private IP Ranges

| Class | Range | Usage |
|-------|-------|-------|
| A | 10.0.0.0/8 | Large networks |
| B | 172.16.0.0/12 | Medium networks |
| C | 192.168.0.0/16 | Home/small networks |

### Subnet Mask

```
IP:      192.168.1.100
Subnet:  255.255.255.0   (/24)
         └──Network──┘ └Host┘

CIDR notation: 192.168.1.0/24
→ 256 addresses (192.168.1.0 ~ 192.168.1.255)
```

### Common Ports

| Port | Service |
|------|---------|
| 22 | SSH |
| 80 | HTTP |
| 443 | HTTPS |
| 3306 | MySQL |
| 5432 | PostgreSQL |
| 6379 | Redis |

---

## 2. Network Configuration Check

### ip addr - IP Address Check

```bash
# All interfaces
ip addr
ip a

# Specific interface
ip addr show eth0
```

Output:
```
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc fq_codel state UP
    link/ether 00:11:22:33:44:55 brd ff:ff:ff:ff:ff:ff
    inet 192.168.1.100/24 brd 192.168.1.255 scope global eth0
       valid_lft forever preferred_lft forever
    inet6 fe80::1/64 scope link
       valid_lft forever preferred_lft forever
```

### ip link - Interface Status

```bash
# List interfaces
ip link

# Enable interface
sudo ip link set eth0 up

# Disable interface
sudo ip link set eth0 down
```

### ip route - Routing Table

```bash
# Check routing
ip route
ip r
```

Output:
```
default via 192.168.1.1 dev eth0 proto dhcp metric 100
192.168.1.0/24 dev eth0 proto kernel scope link src 192.168.1.100
```

### ifconfig (Legacy)

```bash
# May need installation
sudo apt install net-tools    # Ubuntu
sudo dnf install net-tools    # CentOS

# Usage
ifconfig
ifconfig eth0
```

### hostname - Hostname

```bash
# Check hostname
hostname
hostname -I    # IP addresses only

# Change hostname (temporary)
sudo hostname new-hostname

# Change hostname (permanent)
sudo hostnamectl set-hostname new-hostname
```

---

## 3. Connection Testing

### ping - Connection Check

```bash
# Basic ping
ping google.com

# Specify count
ping -c 4 google.com

# Specify interval
ping -i 0.5 192.168.1.1
```

### traceroute - Route Tracing

```bash
# Install
sudo apt install traceroute    # Ubuntu
sudo dnf install traceroute    # CentOS

# Usage
traceroute google.com
traceroute -n 8.8.8.8    # No name resolution
```

### mtr - Combined Tool

```bash
# Install
sudo apt install mtr    # Ubuntu
sudo dnf install mtr    # CentOS

# Usage (ping + traceroute)
mtr google.com
mtr -r -c 10 google.com    # Report mode
```

---

## 4. DNS Lookup

### nslookup

```bash
# Basic lookup
nslookup google.com

# Specific DNS server
nslookup google.com 8.8.8.8

# Specify record type
nslookup -type=MX google.com
```

### dig

```bash
# Basic lookup
dig google.com

# Short output
dig +short google.com

# Specific record
dig google.com MX
dig google.com TXT

# Specific DNS server
dig @8.8.8.8 google.com

# Reverse lookup
dig -x 8.8.8.8
```

### host

```bash
# Simple lookup
host google.com
host -t MX google.com
```

### /etc/hosts

Local DNS configuration.

```bash
cat /etc/hosts
```

```
127.0.0.1   localhost
192.168.1.50   myserver.local myserver
```

### /etc/resolv.conf

DNS server configuration.

```bash
cat /etc/resolv.conf
```

```
nameserver 8.8.8.8
nameserver 8.8.4.4
search example.com
```

---

## 5. Port Check

### ss - Socket Statistics

```bash
# Check open ports
ss -tuln

# TCP connections
ss -t

# Listening ports
ss -l

# Include process information
ss -tulnp

# Specific port
ss -tuln | grep :80
```

Options:
| Option | Description |
|--------|-------------|
| `-t` | TCP |
| `-u` | UDP |
| `-l` | LISTEN state only |
| `-n` | Numeric display |
| `-p` | Process information |

### netstat (Legacy)

```bash
# Open ports
netstat -tuln

# All connections
netstat -an

# Include process
sudo netstat -tulnp
```

### lsof - Open Files/Ports

```bash
# Process using specific port
sudo lsof -i :80

# Network connections of specific process
sudo lsof -i -a -p 1234

# All network connections
sudo lsof -i
```

---

## 6. SSH - Remote Access

### Basic Connection

```bash
# Basic connection
ssh user@hostname
ssh user@192.168.1.100

# Specify port
ssh -p 2222 user@hostname

# Verbose output
ssh -v user@hostname
```

### SSH Key Authentication

```bash
# 1. Generate key
ssh-keygen -t rsa -b 4096

# Or ed25519 (recommended)
ssh-keygen -t ed25519

# 2. Copy public key
ssh-copy-id user@hostname

# 3. Connect with key
ssh user@hostname
```

### SSH Key Files

```
~/.ssh/
├── id_rsa           # Private key (never share)
├── id_rsa.pub       # Public key (register on server)
├── authorized_keys  # List of allowed public keys
├── known_hosts      # List of connected servers
└── config           # SSH configuration
```

### SSH config Configuration

```bash
# ~/.ssh/config
Host myserver
    HostName 192.168.1.100
    User ubuntu
    Port 22
    IdentityFile ~/.ssh/id_rsa

Host production
    HostName prod.example.com
    User deploy
    Port 2222
```

Usage:
```bash
ssh myserver
ssh production
```

---

## 7. File Transfer

### scp - File Copy

```bash
# Local → Remote
scp file.txt user@host:/path/to/destination/

# Remote → Local
scp user@host:/path/to/file.txt ./

# Copy directory
scp -r directory/ user@host:/path/to/

# Specify port
scp -P 2222 file.txt user@host:/path/
```

### rsync - Synchronization

```bash
# Basic sync
rsync -av source/ destination/

# Remote sync
rsync -av local_dir/ user@host:/remote_dir/
rsync -av user@host:/remote_dir/ local_dir/

# Sync deletions
rsync -av --delete source/ destination/

# Show progress
rsync -av --progress source/ destination/

# Compressed transfer
rsync -avz source/ user@host:/destination/

# Specify SSH port
rsync -av -e "ssh -p 2222" source/ user@host:/dest/
```

### sftp - Interactive Transfer

```bash
sftp user@hostname

# sftp commands
sftp> ls              # Remote list
sftp> lls             # Local list
sftp> cd /path        # Remote change directory
sftp> lcd /path       # Local change directory
sftp> get file.txt    # Download
sftp> put file.txt    # Upload
sftp> quit            # Exit
```

---

## 8. Network Configuration (Permanent)

### Ubuntu/Debian (Netplan)

```bash
# Configuration file
sudo vi /etc/netplan/01-netcfg.yaml
```

```yaml
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: no
      addresses:
        - 192.168.1.100/24
      gateway4: 192.168.1.1
      nameservers:
        addresses:
          - 8.8.8.8
          - 8.8.4.4
```

```bash
# Apply
sudo netplan apply
```

### CentOS/RHEL (NetworkManager)

```bash
# Configuration file
sudo vi /etc/sysconfig/network-scripts/ifcfg-eth0
```

```ini
TYPE=Ethernet
BOOTPROTO=static
NAME=eth0
DEVICE=eth0
ONBOOT=yes
IPADDR=192.168.1.100
NETMASK=255.255.255.0
GATEWAY=192.168.1.1
DNS1=8.8.8.8
DNS2=8.8.4.4
```

```bash
# Apply
sudo systemctl restart NetworkManager
# Or
sudo nmcli connection reload
sudo nmcli connection up eth0
```

### nmcli (NetworkManager CLI)

```bash
# List connections
nmcli connection show

# Device status
nmcli device status

# Set static IP
sudo nmcli connection modify eth0 ipv4.addresses 192.168.1.100/24
sudo nmcli connection modify eth0 ipv4.gateway 192.168.1.1
sudo nmcli connection modify eth0 ipv4.dns "8.8.8.8 8.8.4.4"
sudo nmcli connection modify eth0 ipv4.method manual
sudo nmcli connection up eth0

# Change to DHCP
sudo nmcli connection modify eth0 ipv4.method auto
sudo nmcli connection up eth0
```

---

## 9. Practice Exercises

### Exercise 1: Check Network Information

```bash
# Check IP address
ip addr

# Routing table
ip route

# DNS configuration
cat /etc/resolv.conf

# Hostname
hostname -I
```

### Exercise 2: Connection Testing

```bash
# Ping test
ping -c 4 google.com

# DNS lookup
dig +short google.com
nslookup google.com

# Route trace
traceroute -n google.com
```

### Exercise 3: Port Check

```bash
# Check open ports
ss -tuln

# Process using specific port
sudo lsof -i :22

# Test port from outside
nc -zv localhost 22
```

### Exercise 4: SSH Key Setup

```bash
# Generate key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Check key
ls -la ~/.ssh/

# View public key content
cat ~/.ssh/id_ed25519.pub
```

### Exercise 5: File Transfer

```bash
# Test local file copy
mkdir -p ~/test_sync/source ~/test_sync/dest
echo "test content" > ~/test_sync/source/test.txt

# rsync sync
rsync -av ~/test_sync/source/ ~/test_sync/dest/

# Check result
ls -la ~/test_sync/dest/
```

---

**Previous**: [Shell Scripting](./09_Shell_Scripting.md) | **Next**: [System Monitoring](./11_System_Monitoring.md)
