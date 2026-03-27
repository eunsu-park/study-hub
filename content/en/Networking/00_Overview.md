# Networking Learning Guide

## Introduction

This folder contains materials for systematically learning computer networking. You can understand the principles of network communication from the OSI 7-layer model to TCP/IP, routing, and security.

**Target Audience**: Developers, system administrators, anyone learning networking fundamentals

---

## Learning Roadmap

```
[Basics]                  [Intermediate]            [Advanced]
  │                         │                         │
  ▼                         ▼                         ▼
Network Overview ───▶ IP Addressing ──────▶ Routing Protocols
  │                         │                         │
  ▼                         ▼                         ▼
OSI/TCP-IP ─────────▶ TCP/UDP ────────────▶ Network Security
  │                         │                         │
  ▼                         ▼                         ▼
Physical/Data Link ──▶ Application Layer ──▶ Practical Tools
```

---

## Prerequisites

- Computer fundamentals (operating system concepts)
- Understanding of binary and hexadecimal numbers
- Basic command-line usage

---

## File List

### Network Fundamentals (01-04)

| File | Difficulty | Key Topics |
|------|------------|------------|
| [01_Network_Fundamentals.md](./01_Network_Fundamentals.md) | ⭐ | Network definition, LAN/WAN, topology |
| [02_OSI_7_Layer_Model.md](./02_OSI_7_Layer_Model.md) | ⭐⭐ | Layer roles, protocols, PDU |
| [03_TCP_IP_Model.md](./03_TCP_IP_Model.md) | ⭐⭐ | TCP/IP 4-layer, comparison with OSI |
| [04_Physical_Layer.md](./04_Physical_Layer.md) | ⭐ | Transmission media, signals, Ethernet cables |

### Data Link and Network Layer (05-09)

| File | Difficulty | Key Topics |
|------|------------|------------|
| [05_Data_Link_Layer.md](./05_Data_Link_Layer.md) | ⭐⭐ | MAC address, frames, switches, ARP |
| [06_IP_Address_Subnetting.md](./06_IP_Address_Subnetting.md) | ⭐⭐ | IPv4, subnet masks, CIDR |
| [07_Subnetting_Practice.md](./07_Subnetting_Practice.md) | ⭐⭐⭐ | Subnet calculations, VLSM |
| [08_Routing_Basics.md](./08_Routing_Basics.md) | ⭐⭐⭐ | Routing tables, static/dynamic routing |
| [09_Routing_Protocols.md](./09_Routing_Protocols.md) | ⭐⭐⭐ | RIP, OSPF, BGP |

### Transport Layer (10-11)

| File | Difficulty | Key Topics |
|------|------------|------------|
| [10_TCP_Protocol.md](./10_TCP_Protocol.md) | ⭐⭐⭐ | 3-way handshake, flow/congestion control |
| [11_UDP_and_Ports.md](./11_UDP_and_Ports.md) | ⭐⭐ | UDP characteristics, port numbers, TCP vs UDP |

### Application Layer (12-14)

| File | Difficulty | Key Topics |
|------|------------|------------|
| [12_DNS.md](./12_DNS.md) | ⭐⭐ | Domain structure, DNS lookup, records |
| [13_HTTP_and_HTTPS.md](./13_HTTP_and_HTTPS.md) | ⭐⭐⭐ | HTTP methods, status codes, TLS |
| [14_Other_Application_Protocols.md](./14_Other_Application_Protocols.md) | ⭐⭐ | DHCP, FTP, SMTP, SSH |

### Network Security and Practice (15-17)

| File | Difficulty | Key Topics |
|------|------------|------------|
| [15_Network_Security_Basics.md](./15_Network_Security_Basics.md) | ⭐⭐⭐ | Firewalls, NAT, VPN |
| [16_Security_Threats_Response.md](./16_Security_Threats_Response.md) | ⭐⭐⭐⭐ | Sniffing, spoofing, DDoS |
| [17_Practical_Network_Tools.md](./17_Practical_Network_Tools.md) | ⭐⭐⭐ | ping, netstat, tcpdump, Wireshark |

### Modern Networking (18-22)

| File | Difficulty | Key Topics |
|------|------------|------------|
| [18_IPv6.md](./18_IPv6.md) | ⭐⭐⭐ | IPv6 addressing, SLAAC, DHCPv6, transition mechanisms |
| [19_Container_Networking.md](./19_Container_Networking.md) | ⭐⭐⭐⭐ | Docker CNM, K8s networking, CNI plugins, service mesh |
| [20_Software_Defined_Networking.md](./20_Software_Defined_Networking.md) | ⭐⭐⭐⭐ | SDN architecture, OpenFlow, control/data plane separation, network virtualization |
| [21_Quality_of_Service.md](./21_Quality_of_Service.md) | ⭐⭐⭐ | QoS mechanisms, traffic shaping, DiffServ, DSCP, bandwidth management |
| [22_Multicast.md](./22_Multicast.md) | ⭐⭐⭐ | Multicast addressing, IGMP, PIM, multicast routing protocols |

---

## Recommended Learning Path

### Phase 1: Network Fundamentals (1 week)
```
01_Network_Fundamentals → 02_OSI_7_Layer_Model → 03_TCP_IP_Model
```

### Phase 2: Lower Layers (1 week)
```
04_Physical_Layer → 05_Data_Link_Layer
```

### Phase 3: IP and Routing (1-2 weeks)
```
06_IP_Address_Subnetting → 07_Subnetting_Practice → 08_Routing_Basics → 09_Routing_Protocols
```

### Phase 4: Transport Layer (1 week)
```
10_TCP_Protocol → 11_UDP_and_Ports
```

### Phase 5: Application Layer (1 week)
```
12_DNS → 13_HTTP_and_HTTPS → 14_Other_Application_Protocols
```

### Phase 6: Security and Practice (1-2 weeks)
```
15_Network_Security_Basics → 16_Security_Threats_Response → 17_Practical_Network_Tools
```

### Phase 7: Modern Networking (1-2 weeks)
```
18_IPv6 → 19_Container_Networking → 20_Software_Defined_Networking → 21_Quality_of_Service → 22_Multicast
```

---

## Practice Environment

### Command-Line Tools

```bash
# Test network connectivity
ping google.com
traceroute google.com

# Check network information
ip addr                    # Linux
ifconfig                   # macOS
ipconfig                   # Windows

# Check connection status
netstat -an
ss -tuln                   # Linux

# DNS lookup
nslookup google.com
dig google.com
```

### Packet Capture

```bash
# tcpdump (Linux/macOS)
sudo tcpdump -i eth0 -n

# Wireshark (GUI)
# https://www.wireshark.org/

# tshark (CLI)
tshark -i eth0
```

### Simulators

- **Cisco Packet Tracer**: Network simulation
- **GNS3**: Advanced network emulation
- **EVE-NG**: Virtual network lab

---

## Common Port Numbers

| Port | Protocol | Description |
|------|----------|-------------|
| 20, 21 | FTP | File transfer |
| 22 | SSH | Secure shell |
| 23 | Telnet | Remote access (unencrypted) |
| 25 | SMTP | Email transmission |
| 53 | DNS | Domain name service |
| 67, 68 | DHCP | Automatic IP assignment |
| 80 | HTTP | Web |
| 443 | HTTPS | Secure web |
| 3306 | MySQL | Database |
| 5432 | PostgreSQL | Database |

---

## Related Resources

### Links to Other Folders

| Folder | Related Content |
|--------|-----------------|
| [Linux/](../Linux/00_Overview.md) | Network configuration, firewalls |
| [Docker/](../Docker/00_Overview.md) | Container networking |
| [Web_Development/](../Web_Development/00_Overview.md) | HTTP, REST API |

### External Resources

- [Computer Networking: A Top-Down Approach](https://gaia.cs.umass.edu/kurose_ross/)
- [RFC Documents](https://www.rfc-editor.org/)
- [Cloudflare Learning Center](https://www.cloudflare.com/learning/)
- [Network+ Certification](https://www.comptia.org/certifications/network)

---

## Learning Tips

1. **Layer-by-Layer Understanding**: Thoroughly understand OSI/TCP-IP layers
2. **Practice-Oriented**: Directly verify with ping, traceroute, Wireshark
3. **Packet Analysis**: Learn actual packet structures with Wireshark
4. **Subnetting Practice**: Solve many subnet calculation problems
5. **Protocol Headers**: Memorize header structures of each protocol
