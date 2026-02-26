# Practical Network Tools

**Previous**: [Security Threats and Response](./16_Security_Threats_Response.md) | **Next**: [IPv6](./18_IPv6.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use `ping` to test connectivity and interpret RTT, TTL, and packet loss statistics
2. Trace the path of packets with `traceroute`/`mtr` and identify problematic hops
3. Inspect open ports and active connections using `ss` and `netstat`
4. Perform DNS lookups with `dig` and `nslookup` and interpret the output sections
5. Capture and filter network packets with `tcpdump` using BPF syntax
6. Navigate the Wireshark interface, apply display filters, and follow TCP streams
7. Use `curl` to test HTTP endpoints with custom methods, headers, and authentication
8. Apply a systematic, layer-by-layer methodology to diagnose network problems

---

**Difficulty**: ⭐⭐⭐

Knowing how networks work in theory is only half the battle -- you also need to be able to observe, diagnose, and fix them in practice. This lesson puts concrete tools in your hands: the same tools that network engineers and system administrators reach for every day when something goes wrong. From a simple `ping` to deep packet inspection with Wireshark, these utilities transform abstract protocol knowledge into actionable troubleshooting skills.

## Table of Contents

1. [ping](#1-ping)
2. [traceroute / tracert](#2-traceroute--tracert)
3. [netstat / ss](#3-netstat--ss)
4. [nslookup / dig](#4-nslookup--dig)
5. [tcpdump](#5-tcpdump)
6. [Wireshark Basics](#6-wireshark-basics)
7. [curl](#7-curl)
8. [Network Troubleshooting Methodology](#8-network-troubleshooting-methodology)
9. [Practice Problems](#9-practice-problems)
10. [References](#10-references)

---

## 1. ping

### ping Overview

ping is a basic tool that uses ICMP (Internet Control Message Protocol) to test network connectivity.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ping Operation Principle                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Local Host]                              [Target Host]        │
│       │                                        │                │
│       │──── ICMP Echo Request ───────────────▶│                │
│       │     Type: 8, Code: 0                   │                │
│       │     Sequence: 1                        │                │
│       │                                        │                │
│       │◀─── ICMP Echo Reply ──────────────────│                │
│       │     Type: 0, Code: 0                   │                │
│       │     Sequence: 1                        │                │
│       │                                        │                │
│       └─── RTT (Round-Trip Time) Calculation ──────────┘        │
│                                                                 │
│  Measurements:                                                  │
│  - Connectivity status                                          │
│  - Round-trip time (RTT)                                        │
│  - Packet loss rate                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### ping Basic Usage

```bash
# Basic ping (Linux/macOS: continuous, Windows: 4 times)
ping google.com

# Specify count
ping -c 4 google.com          # Linux/macOS
ping -n 4 google.com          # Windows

# Specify interval (seconds)
ping -i 0.5 google.com        # 0.5 second interval

# Specify packet size
ping -s 1000 google.com       # 1000 byte packet

# Specify TTL
ping -t 64 google.com         # Linux/macOS
ping -i 64 google.com         # Windows (TTL)

# Specify timeout
ping -W 2 google.com          # Linux: 2 second timeout
ping -w 2000 google.com       # Windows: 2000ms

# Send without response (flood ping - requires root)
sudo ping -f google.com       # Warning: network load
```

### ping Output Analysis

```bash
$ ping -c 4 google.com

PING google.com (142.250.196.110): 56 data bytes
64 bytes from 142.250.196.110: icmp_seq=0 ttl=116 time=31.2 ms
64 bytes from 142.250.196.110: icmp_seq=1 ttl=116 time=29.8 ms
64 bytes from 142.250.196.110: icmp_seq=2 ttl=116 time=30.5 ms
64 bytes from 142.250.196.110: icmp_seq=3 ttl=116 time=32.1 ms

--- google.com ping statistics ---
4 packets transmitted, 4 packets received, 0.0% packet loss
round-trip min/avg/max/stddev = 29.8/30.9/32.1/0.8 ms
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    ping Output Interpretation                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  64 bytes from 142.250.196.110: icmp_seq=0 ttl=116 time=31.2 ms │
│  │            │                  │         │       │            │
│  │            │                  │         │       └─ RTT       │
│  │            │                  │         └─ TTL (routers      │
│  │            │                  │            128-116=12)       │
│  │            │                  └─ Sequence number             │
│  │            └─ Response IP address                            │
│  └─ Response packet size                                        │
│                                                                 │
│  Statistics:                                                    │
│  - packets transmitted: Sent packet count                       │
│  - packets received: Received packet count                      │
│  - packet loss: Loss rate (0% is normal)                        │
│  - min/avg/max/stddev: RTT minimum/average/maximum/std dev     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### ping Failure Reasons

| Message | Cause |
|---------|-------|
| `Destination Host Unreachable` | No route to destination |
| `Request timed out` | No response (possibly firewall blocked) |
| `Unknown host` | DNS resolution failed |
| `TTL expired in transit` | TTL exceeded (possible loop) |
| `Network is unreachable` | Network not connected |

---

## 2. traceroute / tracert

### traceroute Overview

traceroute is a tool that traces the path packets take to reach their destination.

```
┌─────────────────────────────────────────────────────────────────┐
│                    traceroute Operation Principle                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Trace path by incrementing TTL (Time To Live)                  │
│                                                                 │
│  [Source]      [Router1]     [Router2]     [Destination]        │
│      │            │            │            │                   │
│      │            │            │            │                   │
│      │ TTL=1 ────▶│            │            │                   │
│      │◀─ ICMP ────│            │            │                   │
│      │  Time      │ TTL 0     │            │                   │
│      │  Exceeded  │ (expired) │            │                   │
│      │            │            │            │                   │
│      │ TTL=2 ────────────────▶│            │                   │
│      │◀─ ICMP ─────────────────│            │                   │
│      │  Time Exceeded         │ TTL 0      │                   │
│      │                        │            │                   │
│      │ TTL=3 ────────────────────────────▶│                   │
│      │◀─ ICMP Echo Reply ─────────────────│                   │
│      │            │            │  Destination reached           │
│                                                                 │
│  Measure router IP and RTT at each hop                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### traceroute Basic Usage

```bash
# Basic usage (Linux/macOS)
traceroute google.com

# Windows
tracert google.com

# Without name resolution (faster)
traceroute -n google.com
tracert -d google.com        # Windows

# Specify maximum hops
traceroute -m 20 google.com
tracert -h 20 google.com     # Windows

# Use ICMP (some routers block UDP)
traceroute -I google.com     # Linux

# Use TCP (bypass firewall)
traceroute -T -p 443 google.com

# Specify probe count
traceroute -q 1 google.com   # 1 probe per hop
```

### traceroute Output Analysis

```bash
$ traceroute google.com

traceroute to google.com (142.250.196.110), 30 hops max, 60 byte packets
 1  192.168.1.1 (192.168.1.1)  1.234 ms  1.123 ms  1.098 ms
 2  10.0.0.1 (10.0.0.1)  5.432 ms  5.321 ms  5.234 ms
 3  isp-router.net (203.0.113.1)  12.345 ms  12.234 ms  12.123 ms
 4  * * *
 5  peer-link.google.com (72.14.232.84)  25.432 ms  25.321 ms  25.234 ms
 6  142.250.196.110  30.123 ms  29.987 ms  30.234 ms
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    traceroute Output Interpretation              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1  192.168.1.1 (192.168.1.1)  1.234 ms  1.123 ms  1.098 ms     │
│  │        │            │         │          │         │         │
│  │        │            │         └──────────┴─────────┘         │
│  │        │            │               3 probe RTTs             │
│  │        │            └─ Reverse DNS (hostname)                │
│  │        └─ Router IP address                                  │
│  └─ Hop number                                                  │
│                                                                 │
│  Special indicators:                                            │
│  * * *  - No response (timeout, ICMP blocked)                   │
│  !H     - Host unreachable                                     │
│  !N     - Network unreachable                                  │
│  !P     - Protocol unreachable                                 │
│  !F     - Fragmentation needed                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### mtr (traceroute + ping combined)

```bash
# Installation
sudo apt install mtr    # Ubuntu/Debian
brew install mtr        # macOS

# Basic usage (interactive mode)
mtr google.com

# Report mode
mtr -r -c 10 google.com

# Output example
HOST: myhost           Loss%   Snt   Last   Avg  Best  Wrst StDev
  1. 192.168.1.1        0.0%    10    1.2   1.3   1.1   1.5   0.1
  2. 10.0.0.1           0.0%    10    5.4   5.5   5.2   6.1   0.3
  3. isp-router.net     0.0%    10   12.3  12.4  12.1  13.0   0.3
  4. ???               100.0%    10    0.0   0.0   0.0   0.0   0.0
  5. google-peer        0.0%    10   25.4  25.6  25.1  26.2   0.4
  6. google.com         0.0%    10   30.1  30.2  29.8  30.5   0.2
```

---

## 3. netstat / ss

### netstat Overview

netstat displays network connections, routing tables, interface statistics, etc. Modern Linux systems recommend using ss.

```bash
# Installation (if needed)
sudo apt install net-tools  # Ubuntu/Debian
```

### netstat Key Options

```bash
# Show all connections
netstat -a

# TCP connections only
netstat -t

# UDP connections only
netstat -u

# Listening ports only
netstat -l

# Display as numbers (no name resolution)
netstat -n

# Show process information
netstat -p

# Common combinations
netstat -tuln      # TCP/UDP listening ports
netstat -tulnp     # + process info (requires root)
netstat -an        # All connections (numeric)

# Routing table
netstat -r
netstat -rn        # Numeric

# Interface statistics
netstat -i
```

### ss (Socket Statistics)

ss is faster than netstat and provides more information.

```bash
# Basic usage (similar options to netstat)
ss -tuln           # TCP/UDP listening ports
ss -tulnp          # + process info

# Filter by TCP state
ss -t state established
ss -t state time-wait
ss -t state listening

# Port filtering
ss -tuln 'sport = :80'
ss -tuln 'dport = :443'

# Filter by process name
ss -tulnp | grep nginx

# Detailed information
ss -tulnpe

# Timer information
ss -to
```

### Connection Status Check Example

```bash
$ ss -tuln

Netid  State   Recv-Q  Send-Q   Local Address:Port   Peer Address:Port
tcp    LISTEN  0       128      0.0.0.0:22           0.0.0.0:*
tcp    LISTEN  0       511      0.0.0.0:80           0.0.0.0:*
tcp    LISTEN  0       511      0.0.0.0:443          0.0.0.0:*
udp    UNCONN  0       0        0.0.0.0:68           0.0.0.0:*
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    ss Output Interpretation                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Netid  : Protocol (tcp, udp, unix, etc.)                       │
│  State  : Connection state                                      │
│           - LISTEN: Waiting for connections                     │
│           - ESTAB: Established                                  │
│           - TIME-WAIT: Waiting for connection close             │
│           - CLOSE-WAIT: Waiting for peer to close               │
│  Recv-Q : Receive queue                                         │
│  Send-Q : Send queue                                            │
│  Local Address:Port : Local address:port                        │
│  Peer Address:Port  : Remote address:port                       │
│                                                                 │
│  0.0.0.0:* = Listening on all interfaces                        │
│  127.0.0.1:* = Listening on localhost only                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Check Specific Port Usage

```bash
# Find process using port 80
ss -tulnp | grep :80
lsof -i :80
fuser 80/tcp

# Network connections for specific process
ss -tulnp | grep nginx
lsof -i -a -p $(pgrep nginx)
```

---

## 4. nslookup / dig

### nslookup

nslookup is a basic tool for performing DNS lookups.

```bash
# Basic lookup (A record)
nslookup google.com

# Use specific DNS server
nslookup google.com 8.8.8.8

# Specify record type
nslookup -type=MX google.com      # Mail server
nslookup -type=NS google.com      # Name server
nslookup -type=TXT google.com     # TXT record
nslookup -type=CNAME www.google.com
nslookup -type=AAAA google.com    # IPv6

# Reverse lookup (IP → domain)
nslookup 8.8.8.8
```

### nslookup Output Example

```bash
$ nslookup google.com

Server:         192.168.1.1
Address:        192.168.1.1#53

Non-authoritative answer:
Name:   google.com
Address: 142.250.196.110
```

### dig (Recommended)

dig is a powerful tool that provides more detailed DNS information.

```bash
# Basic lookup
dig google.com

# Short output
dig +short google.com

# Specific records
dig google.com MX
dig google.com NS
dig google.com TXT
dig google.com AAAA

# Specific DNS server
dig @8.8.8.8 google.com

# Reverse lookup
dig -x 8.8.8.8

# Trace (DNS path trace)
dig +trace google.com

# All records
dig google.com ANY

# Show answer only
dig +noall +answer google.com

# Check TTL
dig +ttlunits google.com
```

### dig Output Analysis

```bash
$ dig google.com

; <<>> DiG 9.16.1 <<>> google.com
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 12345
;; flags: qr rd ra; QUERY: 1, ANSWER: 1, AUTHORITY: 0, ADDITIONAL: 1

;; QUESTION SECTION:
;google.com.                    IN      A

;; ANSWER SECTION:
google.com.             300     IN      A       142.250.196.110

;; Query time: 25 msec
;; SERVER: 192.168.1.1#53(192.168.1.1)
;; WHEN: Mon Jan 27 10:30:00 KST 2026
;; MSG SIZE  rcvd: 55
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    dig Output Interpretation                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  HEADER:                                                        │
│  - status: NOERROR = success                                    │
│  - flags: qr (response), rd (recursion desired), ra (recursion  │
│           available)                                            │
│                                                                 │
│  QUESTION SECTION:                                              │
│  - The query requested                                          │
│                                                                 │
│  ANSWER SECTION:                                                │
│  google.com.   300   IN   A   142.250.196.110                   │
│      │          │    │   │          │                           │
│      │          │    │   │          └─ IP address               │
│      │          │    │   └─ Record type (A)                     │
│      │          │    └─ Class (IN = Internet)                   │
│      │          └─ TTL (seconds)                                │
│      └─ Domain                                                  │
│                                                                 │
│  Query time: DNS server response time                           │
│  SERVER: DNS server used                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### host Command

Useful for simple DNS lookups.

```bash
# Basic lookup
host google.com

# Detailed information
host -a google.com

# Specific type
host -t MX google.com

# Reverse
host 8.8.8.8
```

---

## 5. tcpdump

### tcpdump Overview

tcpdump is a command-line tool for capturing and analyzing network packets.

```
┌─────────────────────────────────────────────────────────────────┐
│                    tcpdump Basic Structure                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  sudo tcpdump [options] [filter expression]                     │
│                                                                 │
│  Key options:                                                   │
│  -i <interface>  : Specify capture interface                    │
│  -n              : No name resolution                           │
│  -nn             : Ports as numbers too                         │
│  -v/-vv/-vvv     : Verbose output levels                        │
│  -c <count>      : Number of packets to capture                 │
│  -w <file>       : Save to file                                 │
│  -r <file>       : Read from file                               │
│  -A              : ASCII output                                 │
│  -X              : HEX + ASCII output                           │
│  -s <size>       : Capture size (0 = full)                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### tcpdump Basic Usage

```bash
# Basic capture (requires root)
sudo tcpdump

# Specify interface
sudo tcpdump -i eth0
sudo tcpdump -i any        # All interfaces

# Display as numbers (faster)
sudo tcpdump -n
sudo tcpdump -nn           # Ports as numbers too

# Limit packet count
sudo tcpdump -c 10         # Capture 10 only

# Verbose output
sudo tcpdump -v
sudo tcpdump -vv
sudo tcpdump -vvv

# Show packet content
sudo tcpdump -A            # ASCII
sudo tcpdump -X            # HEX + ASCII
```

### tcpdump Filters

```bash
# Host filters
sudo tcpdump host 192.168.1.100
sudo tcpdump src host 192.168.1.100
sudo tcpdump dst host 192.168.1.100

# Network filter
sudo tcpdump net 192.168.1.0/24

# Port filters
sudo tcpdump port 80
sudo tcpdump port 80 or port 443
sudo tcpdump src port 80
sudo tcpdump dst port 80
sudo tcpdump portrange 8000-9000

# Protocol filters
sudo tcpdump tcp
sudo tcpdump udp
sudo tcpdump icmp
sudo tcpdump arp

# Combinations
sudo tcpdump 'tcp port 80 and host 192.168.1.100'
sudo tcpdump 'tcp port 80 and not host 192.168.1.1'
sudo tcpdump 'icmp or arp'

# TCP flags
sudo tcpdump 'tcp[tcpflags] & tcp-syn != 0'
sudo tcpdump 'tcp[tcpflags] & tcp-rst != 0'
```

### tcpdump File Save/Read

```bash
# Save to file
sudo tcpdump -w capture.pcap
sudo tcpdump -w capture.pcap -c 1000

# Read from file
tcpdump -r capture.pcap
tcpdump -r capture.pcap 'port 80'

# Rotation save
sudo tcpdump -w log-%H%M%S.pcap -G 3600  # Hourly
sudo tcpdump -w log.pcap -C 100          # Per 100MB
```

### tcpdump Output Analysis

```bash
$ sudo tcpdump -i eth0 -nn port 80 -c 3

tcpdump: verbose output suppressed, use -v or -vv for full protocol decode
listening on eth0, link-type EN10MB (Ethernet), capture size 262144 bytes
10:30:00.123456 IP 192.168.1.10.45678 > 93.184.216.34.80: Flags [S], seq 123456789, win 65535, options [mss 1460,nop,wscale 6], length 0
10:30:00.234567 IP 93.184.216.34.80 > 192.168.1.10.45678: Flags [S.], seq 987654321, ack 123456790, win 65535, options [mss 1460,nop,wscale 6], length 0
10:30:00.234789 IP 192.168.1.10.45678 > 93.184.216.34.80: Flags [.], ack 1, win 1024, length 0
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    tcpdump Output Interpretation                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  10:30:00.123456 IP 192.168.1.10.45678 > 93.184.216.34.80:      │
│       │              │          │           │         │         │
│       │              │          │           │         └─ Dest port │
│       │              │          │           └─ Dest IP          │
│       │              │          └─ Source port                  │
│       │              └─ Source IP                                │
│       └─ Timestamp                                               │
│                                                                 │
│  Flags [S], seq 123456789                                       │
│       │         │                                               │
│       │         └─ TCP sequence number                          │
│       └─ TCP flags                                              │
│          [S]  = SYN                                             │
│          [S.] = SYN-ACK                                         │
│          [.]  = ACK                                             │
│          [P.] = PSH-ACK (data)                                  │
│          [F.] = FIN-ACK                                         │
│          [R]  = RST                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Useful tcpdump Examples

```bash
# Capture HTTP GET requests
sudo tcpdump -i eth0 -A 'tcp port 80 and (((ip[2:2] - ((ip[0]&0xf)<<2)) - ((tcp[12]&0xf0)>>2)) != 0)' | grep -i 'GET\|Host'

# Capture DNS queries
sudo tcpdump -i eth0 -nn port 53

# SYN packets only (connection attempts)
sudo tcpdump 'tcp[tcpflags] == tcp-syn'

# Capture HTTPS connections (content encrypted)
sudo tcpdump -i eth0 port 443

# ARP traffic
sudo tcpdump -i eth0 arp

# Communication between specific hosts
sudo tcpdump -i eth0 host 192.168.1.10 and host 192.168.1.20
```

---

## 6. Wireshark Basics

### Wireshark Overview

Wireshark is a powerful graphical packet analysis tool.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Wireshark Interface                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     Filter Bar                           │   │
│  │  [ http.request.method == "GET"                     ]   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Packet List Pane                       │   │
│  │  No.   Time    Source       Destination   Protocol Info │   │
│  │  1     0.000   192.168.1.10 93.184.216.34 TCP     SYN   │   │
│  │  2     0.030   93.184.216.34 192.168.1.10 TCP     SYN-ACK│  │
│  │  3     0.031   192.168.1.10 93.184.216.34 TCP     ACK   │   │
│  │  4     0.032   192.168.1.10 93.184.216.34 HTTP    GET / │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Packet Details Pane                    │   │
│  │  ▶ Frame                                               │   │
│  │  ▶ Ethernet II                                         │   │
│  │  ▶ Internet Protocol Version 4                         │   │
│  │  ▼ Transmission Control Protocol                       │   │
│  │      Source Port: 45678                                │   │
│  │      Destination Port: 80                              │   │
│  │  ▶ Hypertext Transfer Protocol                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Bytes Pane                             │   │
│  │  0000  00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Wireshark Installation

```bash
# Ubuntu/Debian
sudo apt install wireshark

# macOS
brew install --cask wireshark

# Windows
# Download from https://www.wireshark.org/download.html
```

### Wireshark Capture Filters

Applied during capture (same BPF syntax as tcpdump):

```
# Specific host
host 192.168.1.100

# Specific port
port 80
port 80 or port 443

# Specific network
net 192.168.1.0/24

# Protocol
tcp
udp
icmp

# Combinations
tcp port 80 and host 192.168.1.100
```

### Wireshark Display Filters

Filter packets to display after capture:

```
# IP filters
ip.addr == 192.168.1.100
ip.src == 192.168.1.100
ip.dst == 192.168.1.100

# Port filters
tcp.port == 80
tcp.srcport == 80
tcp.dstport == 443

# HTTP filters
http
http.request
http.response
http.request.method == "GET"
http.request.method == "POST"
http.response.code == 200
http.response.code >= 400
http.host contains "google"

# DNS filters
dns
dns.qry.name contains "google"

# TCP filters
tcp.flags.syn == 1
tcp.flags.reset == 1
tcp.analysis.retransmission

# TLS/SSL filters
tls
tls.handshake
ssl.handshake.type == 1  # Client Hello

# Combinations
http and ip.src == 192.168.1.100
tcp.port == 443 and ip.addr == 192.168.1.100

# Negation
not arp
not broadcast
!(ip.addr == 192.168.1.1)
```

### Wireshark Key Features

```
┌─────────────────────────────────────────────────────────────────┐
│                    Wireshark Key Features                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. TCP Stream Following (Follow TCP Stream)                    │
│     - Right-click → Follow → TCP Stream                        │
│     - View entire HTTP conversation                             │
│                                                                 │
│  2. Statistics                                                  │
│     - Protocol Hierarchy: Protocol statistics                   │
│     - Conversations: Host-to-host communication                 │
│     - Endpoints: Communication endpoints                        │
│     - I/O Graphs: Traffic graphs                                │
│                                                                 │
│  3. Expert Information (Analyze → Expert Information)           │
│     - Shows errors, warnings, notes                             │
│     - Identifies issues like retransmissions, duplicate ACKs    │
│                                                                 │
│  4. File Extraction (File → Export Objects)                     │
│     - Extract HTTP objects (images, files, etc.)                │
│                                                                 │
│  5. Time Analysis                                               │
│     - View → Time Display Format                                │
│     - RTT, delay analysis                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### tshark (CLI Wireshark)

```bash
# Basic capture
sudo tshark -i eth0

# Apply filter
sudo tshark -i eth0 -f "port 80"
sudo tshark -i eth0 -Y "http.request"

# Save to file
sudo tshark -i eth0 -w capture.pcap

# Read file
tshark -r capture.pcap

# Extract specific fields
tshark -r capture.pcap -T fields -e ip.src -e ip.dst -e http.host

# JSON output
tshark -r capture.pcap -T json
```

---

## 7. curl

### curl Overview

curl is a command-line tool for transferring data using various protocols.

```bash
# Basic request
curl http://example.com

# Save output
curl -o file.html http://example.com
curl -O http://example.com/file.zip  # Use original filename

# Follow redirects
curl -L http://example.com

# Include headers in output
curl -i http://example.com

# Headers only
curl -I http://example.com

# Verbose output (debug)
curl -v http://example.com
curl -vvv http://example.com  # More verbose

# Silent mode
curl -s http://example.com
curl -sS http://example.com   # Show errors only
```

### HTTP Methods

```bash
# GET (default)
curl http://api.example.com/users

# POST
curl -X POST http://api.example.com/users \
  -H "Content-Type: application/json" \
  -d '{"name": "John", "email": "john@example.com"}'

# POST (form data)
curl -X POST http://example.com/form \
  -d "name=John&email=john@example.com"

# PUT
curl -X PUT http://api.example.com/users/1 \
  -H "Content-Type: application/json" \
  -d '{"name": "John Updated"}'

# PATCH
curl -X PATCH http://api.example.com/users/1 \
  -H "Content-Type: application/json" \
  -d '{"status": "active"}'

# DELETE
curl -X DELETE http://api.example.com/users/1
```

### Header Settings

```bash
# Add headers
curl -H "Authorization: Bearer token123" http://api.example.com
curl -H "Content-Type: application/json" \
     -H "Accept: application/json" \
     http://api.example.com

# Set User-Agent
curl -A "MyApp/1.0" http://example.com

# Send cookies
curl -b "session=abc123" http://example.com
curl -b cookies.txt http://example.com

# Save cookies
curl -c cookies.txt http://example.com
```

### Authentication

```bash
# Basic authentication
curl -u username:password http://example.com
curl -u username http://example.com  # Prompt for password

# Bearer token
curl -H "Authorization: Bearer token123" http://api.example.com

# API key
curl -H "X-API-Key: myapikey" http://api.example.com
```

### HTTPS and Certificates

```bash
# HTTPS request
curl https://example.com

# Ignore certificate verification (testing only)
curl -k https://self-signed.example.com

# Specify certificate
curl --cacert ca.crt https://example.com

# Client certificate
curl --cert client.crt --key client.key https://example.com
```

### Useful Options

```bash
# Timeouts
curl --connect-timeout 5 http://example.com  # Connection timeout
curl --max-time 30 http://example.com        # Total timeout

# Retry
curl --retry 3 http://example.com

# Proxy
curl -x http://proxy:8080 http://example.com

# Compression support
curl --compressed http://example.com

# Show progress
curl -# -O http://example.com/large-file.zip

# File upload
curl -F "file=@/path/to/file.pdf" http://example.com/upload

# Send JSON file
curl -X POST http://api.example.com \
  -H "Content-Type: application/json" \
  -d @data.json
```

### curl Response Analysis

```bash
# Check response code only
curl -s -o /dev/null -w "%{http_code}" http://example.com

# Detailed timing information
curl -s -o /dev/null -w "\
DNS lookup: %{time_namelookup}s\n\
Connect: %{time_connect}s\n\
TLS handshake: %{time_appconnect}s\n\
Start transfer: %{time_starttransfer}s\n\
Total: %{time_total}s\n\
" http://example.com

# Multiple information output
curl -s -o /dev/null -w "\
Response code: %{http_code}\n\
Size: %{size_download} bytes\n\
Time: %{time_total}s\n\
" http://example.com
```

---

## 8. Network Troubleshooting Methodology

### Systematic Approach

```
┌─────────────────────────────────────────────────────────────────┐
│               Network Troubleshooting Flow                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Define Problem                                              │
│     │                                                           │
│     ▼                                                           │
│  2. Gather Information                                          │
│     │  - What are the symptoms?                                 │
│     │  - When did it start?                                     │
│     │  - Which systems affected?                                │
│     │  - Any recent changes?                                    │
│     ▼                                                           │
│  3. Formulate Hypothesis                                        │
│     │  - Physical issue?                                        │
│     │  - Network configuration?                                 │
│     │  - Service problem?                                       │
│     │  - Firewall?                                              │
│     ▼                                                           │
│  4. Verify and Test                                             │
│     │  - Step-by-step testing                                   │
│     │  - Isolate variables                                      │
│     ▼                                                           │
│  5. Resolve and Document                                        │
│        - Record actions taken                                    │
│        - Prevent recurrence                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Layer-by-Layer Diagnosis

```
┌─────────────────────────────────────────────────────────────────┐
│               OSI Layer-based Diagnosis                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  L1 Physical Layer                                              │
│  ─────────────                                                  │
│  Check: Cable connection, LED status, link status               │
│  Tools: ip link, ethtool                                        │
│                                                                 │
│  L2 Data Link Layer                                             │
│  ───────────────                                                │
│  Check: MAC addresses, ARP table, switch settings               │
│  Tools: arp, ip neigh, arping                                   │
│                                                                 │
│  L3 Network Layer                                               │
│  ──────────────                                                 │
│  Check: IP settings, routing, firewall                          │
│  Tools: ip addr, ip route, ping, traceroute                     │
│                                                                 │
│  L4 Transport Layer                                             │
│  ───────────                                                    │
│  Check: Port openness, connection status                        │
│  Tools: ss, netstat, nc, telnet                                 │
│                                                                 │
│  L7 Application Layer                                           │
│  ─────────────────                                              │
│  Check: Service status, logs                                    │
│  Tools: curl, dig, service logs                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Diagnosis Example

```bash
# 1. Check physical/link
ip link show
ethtool eth0

# 2. Check IP settings
ip addr show
ip route show

# 3. Ping local gateway
ping -c 4 192.168.1.1

# 4. Ping external IP
ping -c 4 8.8.8.8

# 5. Check DNS
nslookup google.com
dig google.com

# 6. Ping external domain
ping -c 4 google.com

# 7. Test port connection
nc -zv google.com 443
curl -v https://google.com

# 8. Trace route
traceroute google.com
```

### Common Problems and Solutions

| Symptom | Diagnosis | Solution |
|---------|-----------|----------|
| No connections | ip link, check cables | Repair cable/interface |
| Local only | ip route, ping gateway | Fix routing/gateway settings |
| IP OK, domain fails | nslookup | Fix DNS settings |
| Specific port fails | ss, iptables | Fix firewall/service settings |
| Intermittent connection | mtr, log analysis | Check network quality |
| Slow connection | traceroute, iperf | Identify bottleneck |

### Troubleshooting Script

```bash
#!/bin/bash
# Basic network diagnostics script

echo "=== Network Diagnostics Started ==="
echo ""

echo "1. Interface Status:"
ip link show | grep -E "^[0-9]|state"
echo ""

echo "2. IP Addresses:"
ip addr show | grep -E "inet |inet6 "
echo ""

echo "3. Routing Table:"
ip route show
echo ""

echo "4. DNS Settings:"
cat /etc/resolv.conf | grep nameserver
echo ""

echo "5. Gateway Connection Test:"
GATEWAY=$(ip route | grep default | awk '{print $3}')
ping -c 2 $GATEWAY
echo ""

echo "6. External Connection Test (8.8.8.8):"
ping -c 2 8.8.8.8
echo ""

echo "7. DNS Resolution Test:"
nslookup google.com
echo ""

echo "8. Listening Ports:"
ss -tuln
echo ""

echo "=== Diagnostics Complete ==="
```

---

## 9. Practice Problems

### Basic Problems

1. **ping**
   - What 3 pieces of information can you get from ping?
   - If TTL is 116 and the original TTL was 128, how many routers were traversed?

2. **traceroute**
   - Explain the principle of how traceroute uses TTL.
   - What does `* * *` output mean?

3. **netstat/ss**
   - What's the difference between LISTEN and ESTABLISHED states?
   - What command finds processes using port 80?

### Intermediate Problems

4. **DNS Tools**
   - What are the differences between nslookup and dig?
   - What's the difference in purpose between MX and A records?

5. **tcpdump**
   - Explain the meaning of this filter:
     ```bash
     sudo tcpdump -i eth0 'tcp port 80 and host 192.168.1.100'
     ```

6. **Practical Problem**
   Write the tools and commands you would use in these situations:

   a) Web server (192.168.1.100) not responding
   b) Domain access works but specific site doesn't
   c) Intermittent packet loss occurring

### Advanced Problems

7. **Wireshark**
   - What display filter filters TCP 3-way handshake?
   - How do you filter HTTP response codes 500 and above?

8. **Comprehensive Troubleshooting**
   - Explain step-by-step approach to identify the cause when web service is slow.

---

## 10. References

### Official Tool Documentation

- [Wireshark User Guide](https://www.wireshark.org/docs/wsug_html/)
- [tcpdump Manual](https://www.tcpdump.org/manpages/tcpdump.1.html)
- [curl Manual](https://curl.se/docs/manual.html)

### Cheat Sheet

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frequently Used Commands Summary              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Connection Testing:                                            │
│    ping -c 4 google.com                                        │
│    traceroute google.com                                       │
│    mtr -r google.com                                           │
│                                                                 │
│  DNS Lookup:                                                    │
│    dig +short google.com                                       │
│    nslookup google.com                                         │
│    host google.com                                             │
│                                                                 │
│  Port Check:                                                    │
│    ss -tuln                                                    │
│    ss -tulnp | grep :80                                        │
│    lsof -i :80                                                 │
│                                                                 │
│  Port Testing:                                                  │
│    nc -zv google.com 443                                       │
│    telnet google.com 80                                        │
│                                                                 │
│  Packet Capture:                                                │
│    sudo tcpdump -i eth0 -nn port 80                            │
│    sudo tcpdump -i eth0 -w capture.pcap                        │
│                                                                 │
│  HTTP Testing:                                                  │
│    curl -I http://example.com                                  │
│    curl -v https://example.com                                 │
│    curl -o /dev/null -w "%{http_code}" http://example.com      │
│                                                                 │
│  Network Information:                                           │
│    ip addr show                                                │
│    ip route show                                               │
│    ip neigh show                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Related Folders

| Folder | Related Content |
|--------|-----------------|
| [Linux/](../Linux/00_Overview.md) | Linux network commands |
| [Docker/](../Docker/00_Overview.md) | Container networking |
| [Web_Development/](../Web_Development/00_Overview.md) | HTTP/HTTPS, APIs |

---

---

**Previous**: [Security Threats and Response](./16_Security_Threats_Response.md) | **Next**: [IPv6](./18_IPv6.md)
