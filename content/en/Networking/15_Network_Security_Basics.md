# Network Security Basics

**Previous**: [Other Application Protocols](./14_Other_Application_Protocols.md) | **Next**: [Security Threats and Response](./16_Security_Threats_Response.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define the CIA triad (Confidentiality, Integrity, Availability) and explain the defense-in-depth strategy
2. Compare packet filtering, stateful inspection, application layer, and next-generation firewalls
3. Write basic `iptables` rules to allow or deny traffic by port, protocol, and source address
4. Distinguish between Static NAT, Dynamic NAT, and PAT/NAPT and explain how PAT enables many-to-one address sharing
5. Compare Site-to-Site and Remote Access VPN architectures and evaluate protocols such as IPsec, OpenVPN, and WireGuard
6. Differentiate symmetric and asymmetric encryption and explain why TLS uses a hybrid approach
7. Describe how hash functions and digital signatures provide data integrity and authentication

---

**Difficulty**: ⭐⭐⭐

Networks are under constant threat -- from eavesdroppers intercepting traffic, to attackers exploiting open ports, to insiders exfiltrating data. Security is not a single product you install; it is a layered discipline woven into every level of the network stack. This lesson introduces the fundamental building blocks -- firewalls, NAT, VPNs, and cryptographic primitives -- that together form a robust defense posture.

## Table of Contents

1. [Network Security Overview](#1-network-security-overview)
2. [Firewalls](#2-firewalls)
3. [NAT](#3-nat)
4. [VPN](#4-vpn)
5. [Encryption Basics](#5-encryption-basics)
6. [Practice Problems](#6-practice-problems)
7. [References](#7-references)

---

## 1. Network Security Overview

### CIA Triad

```
┌─────────────────────────────────────────────────────────────────┐
│                    CIA Triad (Security Elements)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                   ┌─────────────────┐                           │
│                   │ Confidentiality │                           │
│                   │                 │                           │
│                   └────────┬────────┘                           │
│                            │                                    │
│                    Only authorized users                        │
│                    can access information                       │
│                            │                                    │
│          ┌─────────────────┼─────────────────┐                  │
│          │                 │                 │                  │
│  ┌───────▼───────┐         │         ┌───────▼───────┐          │
│  │   Integrity   │         │         │ Availability  │          │
│  │               │◀───────┼────────▶│               │          │
│  └───────────────┘         │         └───────────────┘          │
│                            │                                    │
│  Information accurate      │         Information accessible     │
│  and unmodified            │         when needed                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Security Elements Details

| Element | Description | Threat Examples | Countermeasures |
|------|------|----------|----------|
| Confidentiality | Authorized access only | Eavesdropping, sniffing | Encryption, access control |
| Integrity | Prevent data modification | MITM attack, tampering | Hashing, digital signatures |
| Availability | Continuous service provision | DoS/DDoS attacks | Redundancy, load balancing |

### Additional Security Elements

```
┌─────────────────────────────────────────────────────────────────┐
│                    Additional Security Elements                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Authentication                                                 │
│  └─ Verify user/system identity                                 │
│     Examples: Passwords, certificates, biometrics               │
│                                                                 │
│  Authorization                                                  │
│  └─ Grant access permissions                                    │
│     Example: Role-Based Access Control (RBAC)                   │
│                                                                 │
│  Non-repudiation                                                │
│  └─ Cannot deny actions performed                               │
│     Examples: Digital signatures, audit logs                    │
│                                                                 │
│  Accountability                                                 │
│  └─ Actions can be traced to actors                             │
│     Examples: Logging, monitoring                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Defense in Depth

```
┌─────────────────────────────────────────────────────────────────┐
│                    Defense in Depth Strategy                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     Physical Security                    │   │
│  │                  (Server room, access control)           │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │                   Perimeter Security             │    │   │
│  │  │               (Firewall, IDS/IPS)                │    │   │
│  │  │  ┌─────────────────────────────────────────┐    │    │   │
│  │  │  │                Network Security          │    │    │   │
│  │  │  │            (VLAN, network segregation)   │    │    │   │
│  │  │  │  ┌─────────────────────────────────┐    │    │    │   │
│  │  │  │  │            Host Security         │    │    │    │   │
│  │  │  │  │      (OS security, antivirus)    │    │    │    │   │
│  │  │  │  │  ┌─────────────────────────┐    │    │    │    │   │
│  │  │  │  │  │  Application Security   │    │    │    │    │   │
│  │  │  │  │  │  (Input validation, auth)│    │    │    │    │   │
│  │  │  │  │  │  ┌─────────────────┐    │    │    │    │    │   │
│  │  │  │  │  │  │   Data Security │    │    │    │    │    │   │
│  │  │  │  │  │  │ (Encryption,backup)│    │    │    │    │   │
│  │  │  │  │  │  └─────────────────┘    │    │    │    │    │   │
│  │  │  │  │  └─────────────────────────┘    │    │    │    │   │
│  │  │  │  └─────────────────────────────────┘    │    │    │   │
│  │  │  └─────────────────────────────────────────┘    │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Firewalls

### Firewall Overview

A firewall is a network security device that monitors network traffic and allows or blocks traffic according to security rules.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Firewall Position                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Internet]                                                     │
│     │                                                           │
│     ▼                                                           │
│  ┌──────────────────┐                                           │
│  │  Border Router   │                                           │
│  └────────┬─────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────┐                                           │
│  │    Firewall      │◀─ Traffic filtering                      │
│  │ (External FW)    │                                           │
│  └────────┬─────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────┐                                           │
│  │      DMZ         │◀─ Web server, mail server                │
│  └────────┬─────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────┐                                           │
│  │    Firewall      │◀─ Internal protection                    │
│  │ (Internal FW)    │                                           │
│  └────────┬─────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────┐                                           │
│  │ Internal Network │◀─ Employee PCs, internal servers         │
│  └──────────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Firewall Types

#### 1. Packet Filtering Firewall

```
┌─────────────────────────────────────────────────────────────────┐
│                  Packet Filtering Firewall                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Operating Layer: L3 (Network), L4 (Transport)                  │
│                                                                 │
│  Inspection criteria:                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │Source IP│Dest IP│Protocol│Source Port│Dest Port│        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Rule example:                                                  │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ # Rule  Source IP      Dest IP      Port   Protocol Action ││
│  │ 1    192.168.1.0/24   any           80     TCP     ALLOW  ││
│  │ 2    any              192.168.1.10  22     TCP     ALLOW  ││
│  │ 3    10.0.0.0/8       any           any    any     DENY   ││
│  │ 4    any              any           any    any     DENY   ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                 │
│  Pros: Fast speed, simple implementation                        │
│  Cons: Cannot inspect packet content, no state tracking         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 2. Stateful Inspection Firewall

```
┌─────────────────────────────────────────────────────────────────┐
│                  Stateful Inspection Firewall                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Operating Layer: L3, L4 + connection state tracking            │
│                                                                 │
│  State Table:                                                   │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Src IP:Port   Dest IP:Port   Protocol  State    Timeout    ││
│  │ 192.168.1.10:45000  93.184.216.34:80  TCP ESTABLISHED 3600 ││
│  │ 192.168.1.10:45001  8.8.8.8:53       UDP  ACTIVE      60   ││
│  │ 192.168.1.20:52000  10.0.0.5:22      TCP ESTABLISHED 7200 ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                 │
│  TCP state tracking:                                            │
│                                                                 │
│  [Client]          [Firewall]          [Server]                 │
│       │                  │                 │                    │
│       │──── SYN ────────▶│──── SYN ───────▶│                    │
│       │    (NEW)         │                 │                    │
│       │                  │◀─── SYN-ACK ────│                    │
│       │◀─── SYN-ACK ─────│                 │                    │
│       │──── ACK ────────▶│──── ACK ───────▶│                    │
│       │  (ESTABLISHED)   │                 │                    │
│       │                  │                 │                    │
│                                                                 │
│  Pros: Connection state tracking, auto-allow return traffic     │
│  Cons: Cannot inspect packet content                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 3. Application Layer Firewall

```
┌─────────────────────────────────────────────────────────────────┐
│                Application Layer Firewall                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Operating Layer: L7 (Application)                              │
│                                                                 │
│  Inspection criteria:                                           │
│  - HTTP method, URL, headers, body                              │
│  - DNS query content                                            │
│  - FTP commands                                                 │
│  - SQL query patterns                                           │
│                                                                 │
│  Features:                                                      │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ - Web Application Firewall (WAF)                            ││
│  │ - Block SQL Injection                                       ││
│  │ - Block XSS attacks                                         ││
│  │ - Block malicious file uploads                              ││
│  │ - API request validation                                    ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                 │
│  Pros: Detailed traffic analysis, application-level protection  │
│  Cons: High processing load, complex configuration              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Next-Generation Firewall (NGFW)

```
┌─────────────────────────────────────────────────────────────────┐
│              Next-Generation Firewall (NGFW)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     NGFW Features                        │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐│   │
│  │  │ Packet Filter │  │ Stateful Insp │  │ Application   ││   │
│  │  │               │  │               │  │ Recognition   ││   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘│   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐│   │
│  │  │ IPS Integrated│  │ SSL Decryption│  │ User Identity ││   │
│  │  │               │  │               │  │               ││   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘│   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐│   │
│  │  │Threat Intel   │  │ Sandboxing    │  │ URL Filtering ││   │
│  │  │               │  │               │  │               ││   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘│   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Major vendors: Palo Alto, Fortinet, Check Point, Cisco        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Firewall Rule Example (iptables)

```bash
# Set default policy
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow localhost
iptables -A INPUT -i lo -j ACCEPT

# Allow established connections (stateful)
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (port 22)
iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTP/HTTPS
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow SSH from specific IP only
iptables -A INPUT -p tcp -s 192.168.1.0/24 --dport 22 -j ACCEPT

# Allow ICMP (ping)
iptables -A INPUT -p icmp --icmp-type echo-request -j ACCEPT

# View rules
iptables -L -n -v

# Save rules
iptables-save > /etc/iptables.rules
```

---

## 3. NAT

### NAT Overview

NAT (Network Address Translation) is a technology that translates IP addresses to different IP addresses.

```
┌─────────────────────────────────────────────────────────────────┐
│                    NAT Basic Concept                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐│
│  │                      Private Network                        ││
│  │   192.168.1.0/24                                           ││
│  │                                                            ││
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐                    ││
│  │  │ PC-1    │  │ PC-2    │  │ PC-3    │                    ││
│  │  │.10      │  │.20      │  │.30      │                    ││
│  │  └────┬────┘  └────┬────┘  └────┬────┘                    ││
│  │       │            │            │                          ││
│  │       └─────────┬──┴────────────┘                          ││
│  │                 │                                          ││
│  │           ┌─────┴─────┐                                    ││
│  │           │  Router   │                                    ││
│  │           │   NAT     │                                    ││
│  │           │ 192.168.1.1 (internal)                         ││
│  │           │ 203.0.113.1 (external)                         ││
│  │           └─────┬─────┘                                    ││
│  └─────────────────┼──────────────────────────────────────────┘│
│                    │                                           │
│                    ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐
│  │                      Internet                                │
│  │                                                             │
│  │  All PCs appear as 203.0.113.1                              │
│  └─────────────────────────────────────────────────────────────┘
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### NAT Types

#### 1. Static NAT (1:1)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Static NAT                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Fixed mapping: Private IP ↔ Public IP (1:1)                    │
│                                                                 │
│  Mapping table:                                                 │
│  ┌────────────────────┬────────────────────┐                   │
│  │   Private IP       │    Public IP       │                   │
│  ├────────────────────┼────────────────────┤                   │
│  │  192.168.1.10      │  203.0.113.10      │                   │
│  │  192.168.1.20      │  203.0.113.20      │                   │
│  │  192.168.1.30      │  203.0.113.30      │                   │
│  └────────────────────┴────────────────────┘                   │
│                                                                 │
│  Use case: When external access to internal server is needed    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 2. Dynamic NAT (N:N)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dynamic NAT                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Dynamic mapping: Private IP pool → Public IP pool              │
│                                                                 │
│  Private IP Pool           Public IP Pool                       │
│  192.168.1.10           203.0.113.10                            │
│  192.168.1.20   ───▶    203.0.113.11                            │
│  192.168.1.30           203.0.113.12                            │
│  192.168.1.40           (available IP assigned)                 │
│                                                                 │
│  Features:                                                      │
│  - First-come-first-served allocation                           │
│  - Concurrent connections limited to number of public IPs       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 3. PAT/NAPT (N:1) - Most Common

```
┌─────────────────────────────────────────────────────────────────┐
│              PAT (Port Address Translation)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Multiple private IPs share one public IP (distinguished by port)│
│                                                                 │
│  [Private Network]                    [NAT Router]              │
│                                                                 │
│  192.168.1.10:45000 ──────▶  203.0.113.1:10001  ──────▶ Internet│
│  192.168.1.20:45001 ──────▶  203.0.113.1:10002  ──────▶ Internet│
│  192.168.1.30:45002 ──────▶  203.0.113.1:10003  ──────▶ Internet│
│                                                                 │
│  NAT Table:                                                     │
│  ┌────────────────────────┬───────────────────────┐            │
│  │ Internal Address:Port  │ External Address:Port │            │
│  ├────────────────────────┼───────────────────────┤            │
│  │ 192.168.1.10:45000     │ 203.0.113.1:10001     │            │
│  │ 192.168.1.20:45001     │ 203.0.113.1:10002     │            │
│  │ 192.168.1.30:45002     │ 203.0.113.1:10003     │            │
│  └────────────────────────┴───────────────────────┘            │
│                                                                 │
│  Use case: Home, small business (routers)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### NAT Operation Process

```
┌─────────────────────────────────────────────────────────────────┐
│                    NAT Packet Translation                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Internal → External (Outbound)                              │
│  ─────────────────────────                                      │
│                                                                 │
│  [PC]                    [NAT Router]               [Web Server]│
│  192.168.1.10:45000      203.0.113.1              93.184.216.34 │
│       │                       │                        │        │
│       │──────────────────────▶│                        │        │
│       │ Source: 192.168.1.10:45000                     │        │
│       │ Dest: 93.184.216.34:80                         │        │
│       │                       │                        │        │
│       │        [NAT Translation]│───────────────────────▶│        │
│       │                       │ Source: 203.0.113.1:10001│        │
│       │                       │ Dest: 93.184.216.34:80 │        │
│       │                       │                        │        │
│                                                                 │
│  2. External → Internal (Inbound - response)                    │
│  ──────────────────────────────                                 │
│       │                       │                        │        │
│       │                       │◀───────────────────────│        │
│       │                       │ Source: 93.184.216.34:80│        │
│       │                       │ Dest: 203.0.113.1:10001│        │
│       │                       │                        │        │
│       │◀──────────────────────│        [NAT Reverse]   │        │
│       │ Source: 93.184.216.34:80│                       │        │
│       │ Dest: 192.168.1.10:45000                       │        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### NAT's Security Role

```
┌─────────────────────────────────────────────────────────────────┐
│                    NAT Security Characteristics                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Pros (Security Perspective):                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. Hide internal IP addresses                            │   │
│  │    - Difficult for outsiders to learn internal network   │   │
│  │                                                          │   │
│  │ 2. Natural firewall effect                               │   │
│  │    - Direct external access to internal not possible     │   │
│  │    - Only internally-initiated connections allowed       │   │
│  │                                                          │   │
│  │ 3. Session-based filtering                               │   │
│  │    - Packets not in NAT table are blocked                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Caution:                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ⚠ NAT is address translation, not a security function    │   │
│  │ ⚠ Should be used with firewall                           │   │
│  │ ⚠ Internal exposure with port forwarding                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Port Forwarding

```
┌─────────────────────────────────────────────────────────────────┐
│                    Port Forwarding                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  NAT configuration for external access to internal server       │
│                                                                 │
│  [Internet]                [NAT Router]              [Internal Server]│
│                         203.0.113.1              192.168.1.100  │
│     │                       │                        │          │
│     │───────────────────────▶│                        │          │
│     │ Dest: 203.0.113.1:80   │                        │          │
│     │                       │                        │          │
│     │      [Port Forwarding] │───────────────────────▶│          │
│     │                       │ Dest: 192.168.1.100:80 │          │
│     │                       │                        │          │
│                                                                 │
│  Configuration example:                                         │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ External Port  Internal IP      Internal Port  Protocol    ││
│  │ 80         192.168.1.100     80          TCP              ││
│  │ 443        192.168.1.100     443         TCP              ││
│  │ 22         192.168.1.200     22          TCP              ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                 │
│  Linux iptables example:                                        │
│  iptables -t nat -A PREROUTING -p tcp --dport 80 \             │
│    -j DNAT --to-destination 192.168.1.100:80                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. VPN

### VPN Overview

VPN (Virtual Private Network) provides secure private network connections through public networks.

```
┌─────────────────────────────────────────────────────────────────┐
│                    VPN Basic Concept                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Without VPN (Public Internet)                                  │
│  ──────────────────────                                         │
│  [PC] ───── Plaintext data ───────▶ [Internet] ───────▶ [Server]│
│             Eavesdropping possible                              │
│                                                                 │
│  With VPN                                                       │
│  ────────────                                                   │
│  [PC] ═══ Encrypted tunnel ═══▶ [Internet] ═══▶ [VPN Server] ──▶ [Server]│
│           Secure connection                                      │
│                                                                 │
│  VPN Features:                                                  │
│  - Data encryption (confidentiality)                            │
│  - Data integrity verification                                  │
│  - User authentication                                          │
│  - IP address hiding                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### VPN Types

#### 1. Site-to-Site VPN

```
┌─────────────────────────────────────────────────────────────────┐
│                    Site-to-Site VPN                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────┐                     ┌────────────────┐      │
│  │ HQ Network     │                     │ Branch Network │      │
│  │  10.1.0.0/16   │                     │  10.2.0.0/16   │      │
│  │                │                     │                │      │
│  │ ┌────┐ ┌────┐ │                     │ ┌────┐ ┌────┐ │      │
│  │ │PC  │ │Srv │ │                     │ │PC  │ │Srv │ │      │
│  │ └──┬─┘ └──┬─┘ │                     │ └──┬─┘ └──┬─┘ │      │
│  │    └──┬───┘   │                     │    └──┬───┘   │      │
│  │       │       │                     │       │       │      │
│  │ ┌─────┴─────┐ │                     │ ┌─────┴─────┐ │      │
│  │ │VPN Gateway│ │                     │ │VPN Gateway│ │      │
│  │ └─────┬─────┘ │                     │ └─────┬─────┘ │      │
│  └───────┼───────┘                     └───────┼───────┘      │
│          │                                     │              │
│          │      ┌──────────────────┐          │              │
│          └──────│    Internet      │──────────┘              │
│                 │                  │                         │
│                 │ ═══ VPN Tunnel ══│                         │
│                 └──────────────────┘                         │
│                                                                 │
│  Use case: HQ-branch connection, data center connection         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 2. Remote Access VPN

```
┌─────────────────────────────────────────────────────────────────┐
│                  Remote Access VPN                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│      [Remote Users]                                             │
│                                                                 │
│  ┌─────┐  ┌─────┐  ┌─────┐                                     │
│  │ PC1 │  │ PC2 │  │ PC3 │                                     │
│  │ VPN │  │ VPN │  │ VPN │                                     │
│  │Client│ │Client│ │Client│                                    │
│  └──┬──┘  └──┬──┘  └──┬──┘                                     │
│     │        │        │                                         │
│     │   ┌────┴────────┴────┐                                   │
│     └───│     Internet     │                                   │
│         │                  │                                   │
│         │ ═══ VPN Tunnel ══│                                   │
│         └────────┬─────────┘                                   │
│                  │                                             │
│           ┌──────┴──────┐                                      │
│           │  VPN Server │                                      │
│           └──────┬──────┘                                      │
│                  │                                             │
│  ┌───────────────┼───────────────────┐                         │
│  │           Company Network          │                         │
│  │    ┌────┐  ┌────┐  ┌────┐        │                         │
│  │    │Srv │  │ DB │  │File│        │                         │
│  │    └────┘  └────┘  └────┘        │                         │
│  └───────────────────────────────────┘                         │
│                                                                 │
│  Use case: Remote work, business trip access to company network │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### VPN Protocols

| Protocol | Layer | Features | Security |
|----------|------|------|------|
| PPTP | L2 | Old, fast | Weak (not recommended) |
| L2TP/IPsec | L2+L3 | Common | Strong |
| IPsec | L3 | Standard, compatible | Strong |
| OpenVPN | L3/L4 | Open source, flexible | Strong |
| WireGuard | L3 | Latest, fast, simple | Strong |
| SSL/TLS VPN | L4-L7 | Browser-based | Strong |

### IPsec VPN

```
┌─────────────────────────────────────────────────────────────────┐
│                    IPsec Protocol                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  IPsec Components:                                              │
│                                                                 │
│  1. IKE (Internet Key Exchange)                                 │
│     - Key exchange and SA(Security Association) establishment   │
│     - Phase 1: IKE SA establishment (auth, encryption negotiation)│
│     - Phase 2: IPsec SA establishment (actual tunnel setup)     │
│                                                                 │
│  2. AH (Authentication Header)                                  │
│     - Data integrity, source authentication                     │
│     - No encryption                                             │
│                                                                 │
│  3. ESP (Encapsulating Security Payload)                        │
│     - Data encryption + integrity + authentication              │
│     - Most commonly used                                        │
│                                                                 │
│  IPsec Modes:                                                   │
│  ┌────────────────────────────────────────────────────────────┐│
│  │  Tunnel Mode                                                ││
│  │  - Encrypts entire IP packet                                ││
│  │  - Used in Site-to-Site VPN                                 ││
│  │                                                             ││
│  │  Original: [IP Header][Data]                                ││
│  │  Result: [New IP Header][ESP Header][Encrypted Original Packet][ESP Trailer]││
│  ├────────────────────────────────────────────────────────────┤│
│  │  Transport Mode                                             ││
│  │  - Encrypts data payload only                               ││
│  │  - Used in host-to-host communication                       ││
│  │                                                             ││
│  │  Original: [IP Header][Data]                                ││
│  │  Result: [IP Header][ESP Header][Encrypted Data][ESP Trailer]││
│  └────────────────────────────────────────────────────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### WireGuard

```
┌─────────────────────────────────────────────────────────────────┐
│                    WireGuard VPN                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Features:                                                      │
│  - Modern encryption (ChaCha20, Poly1305, Curve25519)          │
│  - ~4,000 lines of code (100x less than IPsec)                 │
│  - Fast connection, low latency                                 │
│  - Built into Linux kernel (5.6+)                               │
│                                                                 │
│  Configuration example (/etc/wireguard/wg0.conf):              │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ [Interface]                                                ││
│  │ PrivateKey = <server private key>                          ││
│  │ Address = 10.0.0.1/24                                      ││
│  │ ListenPort = 51820                                         ││
│  │                                                            ││
│  │ [Peer]                                                     ││
│  │ PublicKey = <client public key>                            ││
│  │ AllowedIPs = 10.0.0.2/32                                   ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                 │
│  Commands:                                                      │
│  wg-quick up wg0      # Start VPN                               │
│  wg-quick down wg0    # Stop VPN                                │
│  wg show              # Show status                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Encryption Basics

### Encryption Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Basic Encryption Concepts                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Plaintext              Encryption              Ciphertext      │
│                                                                 │
│  "Hello World"  ──────────────────▶  "Xj2#kL9@mP"              │
│                        │                                        │
│                      [Key]                                      │
│                        │                                        │
│  "Hello World"  ◀──────────────────  "Xj2#kL9@mP"              │
│                                                                 │
│                   Decryption                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Symmetric Encryption

```
┌─────────────────────────────────────────────────────────────────┐
│                    Symmetric Encryption                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Encrypt/decrypt with same key                                  │
│                                                                 │
│  [Sender]                              [Receiver]               │
│     │                                     │                     │
│     │  Plaintext: "Hello"                 │                     │
│     │     │                               │                     │
│     │     ▼                               │                     │
│     │  ┌──────┐                           │                     │
│     │  │Encrypt│◀──── Secret Key ────────▶│                     │
│     │  └──┬───┘      "secretkey"         │                     │
│     │     │                               │                     │
│     │     ▼                               ▼                     │
│     │  Ciphertext: "Xj2#k" ─────────▶  ┌──────┐                │
│     │                              │Decrypt│                    │
│     │                              └──┬───┘                    │
│     │                                 │                        │
│     │                                 ▼                        │
│     │                              Plaintext: "Hello"          │
│                                                                 │
│  Main algorithms:                                               │
│  - AES (Advanced Encryption Standard) - Current standard       │
│  - ChaCha20 - Optimized for mobile                             │
│  - 3DES - Legacy (not recommended)                              │
│                                                                 │
│  Pros: Fast speed                                               │
│  Cons: Key distribution problem (how to share key securely?)    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Asymmetric Encryption

```
┌─────────────────────────────────────────────────────────────────┐
│                    Asymmetric Encryption                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Uses public/private key pair                                   │
│                                                                 │
│  [Receiver Key Pair]                                            │
│  ┌─────────────┐     ┌─────────────┐                           │
│  │ Public Key  │     │Private Key  │                           │
│  │(Public access)│    │(Owner only) │                           │
│  └─────────────┘     └─────────────┘                           │
│                                                                 │
│  Encryption scenario:                                           │
│                                                                 │
│  [Sender Alice]        Public channel       [Receiver Bob]      │
│       │                                     │                   │
│       │          Request Bob's public key   │                   │
│       │◀────────────────────────────────────│                   │
│       │                                     │                   │
│       │  Plaintext: "Hello"                 │                   │
│       │     │                               │                   │
│       │     ▼                               │                   │
│       │  ┌────────────┐                     │                   │
│       │  │Encrypt with│                     │                   │
│       │  │Bob's public│                     │                   │
│       │  │    key     │                     │                   │
│       │  └─────┬──────┘                     │                   │
│       │        │                            │                   │
│       │        ▼                            ▼                   │
│       │  Ciphertext ──────────────────▶  ┌────────────┐        │
│       │                             │Decrypt with │            │
│       │                             │Bob's private│            │
│       │                             │    key      │            │
│       │                             └─────┬──────┘            │
│       │                                   │                    │
│       │                                   ▼                    │
│       │                              Plaintext: "Hello"        │
│                                                                 │
│  Main algorithms:                                               │
│  - RSA (2048+ bit)                                              │
│  - ECC (Elliptic Curve Cryptography)                            │
│  - Ed25519 (specialized for digital signatures)                 │
│                                                                 │
│  Pros: Solves key distribution problem                          │
│  Cons: Slower than symmetric (100~1000x)                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Hybrid Encryption

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hybrid Encryption                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Exchange symmetric key securely with asymmetric key, then      │
│  communicate with symmetric key                                 │
│                                                                 │
│  [Client]                              [Server]                 │
│       │                                     │                   │
│       │    ───(1) Request server public key ─▶                  │
│       │                                     │                   │
│       │    ◀─(2) Send server public key ────│                   │
│       │                                     │                   │
│       │  ┌───────────────────────┐         │                   │
│       │  │ 1. Generate symmetric │         │                   │
│       │  │    key (session key)  │         │                   │
│       │  │    (e.g., AES-256)    │         │                   │
│       │  │                       │         │                   │
│       │  │ 2. Encrypt session    │         │                   │
│       │  │    key with server    │         │                   │
│       │  │    public key         │         │                   │
│       │  └───────────────────────┘         │                   │
│       │                                     │                   │
│       │    ───(3) Send encrypted session key─▶                  │
│       │                                     │                   │
│       │                          ┌───────────────────────┐     │
│       │                          │ Decrypt session key   │     │
│       │                          │ with private key      │     │
│       │                          └───────────────────────┘     │
│       │                                     │                   │
│       │    ═══(4) Communication encrypted with session key═══▶│  │
│       │    ◀══════════════════════════════│                   │
│                                                                 │
│  ※ TLS/SSL uses this method                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Hash Functions

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hash Functions                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Convert input data to fixed-length hash value (one-way)        │
│                                                                 │
│  Input (variable length)           Hash (fixed length)          │
│  ─────────────────────────────────────────────────              │
│  "Hello"             ──▶     a591a6d40bf...  (SHA-256)         │
│  "Hello World"       ──▶     b94d27b9934...                    │
│  "hello"             ──▶     2cf24dba5fb...                    │
│                              (Small change → completely different result)│
│                                                                 │
│  Properties:                                                    │
│  1. One-way: Cannot recover original from hash                  │
│  2. Deterministic: Same input → always same hash                │
│  3. Collision resistance: Difficult for different inputs to have same hash│
│  4. Avalanche effect: Small input change → large hash change    │
│                                                                 │
│  Main algorithms:                                               │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Algorithm │ Output   │ Status                               ││
│  ├────────────────────────────────────────────────────────────┤│
│  │ MD5       │ 128 bit  │ Weak (prohibited)                    ││
│  │ SHA-1     │ 160 bit  │ Weak (prohibited)                    ││
│  │ SHA-256   │ 256 bit  │ Safe (recommended)                   ││
│  │ SHA-3     │ Variable │ Safe (latest)                        ││
│  │ BLAKE2    │ Variable │ Safe (fast)                          ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                 │
│  Use cases:                                                     │
│  - Password storage (hash + salt)                               │
│  - Data integrity verification                                  │
│  - Digital signatures                                           │
│  - File checksums                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Digital Signatures

```
┌─────────────────────────────────────────────────────────────────┐
│                    Digital Signatures                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Signature creation (Sender)                                    │
│  ─────────────────                                              │
│                                                                 │
│  Original document                                              │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────┐                                                   │
│  │  Hash   │──▶ Document hash value                            │
│  └─────────┘        │                                          │
│                     ▼                                          │
│               ┌───────────┐                                    │
│               │Sender's   │                                    │
│               │private key│──▶ Digital signature               │
│               │encryption │                                    │
│               └───────────┘                                    │
│                                                                 │
│  Transmission: [Original document] + [Digital signature]        │
│                                                                 │
│  Signature verification (Receiver)                              │
│  ─────────────────                                              │
│                                                                 │
│  [Original document]      [Digital signature]                   │
│      │                        │                                │
│      ▼                        ▼                                │
│  ┌─────────┐           ┌───────────┐                           │
│  │  Hash   │           │Sender's   │                           │
│  └────┬────┘           │public key │                           │
│       │                │decryption │                           │
│       │                └─────┬─────┘                           │
│       │                      │                                  │
│       ▼                      ▼                                  │
│  Calculated hash      =?  Decrypted hash                        │
│                                                                 │
│  Match → Integrity verified + Sender authenticated              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Practice Problems

### Basic Problems

1. **Security Basics**
   - Explain the three elements of the CIA Triad.
   - What is Defense in Depth?

2. **Firewalls**
   - What's the difference between packet filtering and stateful firewalls?
   - What does this iptables rule do?
     ```bash
     iptables -A INPUT -p tcp --dport 22 -s 192.168.1.0/24 -j ACCEPT
     ```

3. **NAT**
   - What are the two main purposes of NAT?
   - Explain the operating principle of PAT (Port Address Translation).

### Intermediate Problems

4. **VPN**
   - What's the difference between Site-to-Site VPN and Remote Access VPN?
   - Explain the difference between IPsec tunnel mode and transport mode.

5. **Encryption**
   - Compare the pros and cons of symmetric and asymmetric encryption.
   - Why does TLS use hybrid encryption?

6. **Practical Problems**

```bash
# Suggest appropriate security solutions for these scenarios

# 1. Remote worker needs to access company internal network
#    Answer:

# 2. Need to block SQL Injection attacks on web server
#    Answer:

# 3. Secure communication needed between HQ and branch office
#    Answer:
```

### Advanced Problems

7. **Comprehensive Analysis**
   - Find security vulnerabilities in this network:
     ```
     Internet ─── Router ─── Internal Network
                  │
               Web Server
     ```

8. **Encryption Application**
   - Explain how to use hashing to verify file integrity.
   - Why is using only hash insufficient for password storage?

---

## 7. References

### Standards and RFC

- [RFC 4301](https://tools.ietf.org/html/rfc4301) - IPsec Architecture
- [RFC 5246](https://tools.ietf.org/html/rfc5246) - TLS 1.2
- [RFC 8446](https://tools.ietf.org/html/rfc8446) - TLS 1.3

### Tools

- iptables/nftables - Linux firewalls
- OpenVPN - Open source VPN
- WireGuard - Modern VPN
- OpenSSL - Encryption tools

### Learning Resources

- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [OWASP](https://owasp.org/)
- [Cloudflare Learning Center](https://www.cloudflare.com/learning/)

---

**Previous**: [Other Application Protocols](./14_Other_Application_Protocols.md) | **Next**: [Security Threats and Response](./16_Security_Threats_Response.md)
