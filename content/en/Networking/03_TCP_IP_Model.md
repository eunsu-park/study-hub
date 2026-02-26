# TCP/IP Model

**Previous**: [OSI 7-Layer Model](./02_OSI_7_Layer_Model.md) | **Next**: [Physical Layer](./04_Physical_Layer.md)

**Difficulty**: ⭐⭐ (Beginner-Intermediate)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the four layers of the TCP/IP model and the role of each layer
2. Compare the TCP/IP model with the OSI 7-layer model and explain why TCP/IP became the Internet standard
3. Trace the complete communication flow when a web page is loaded (DNS, TCP handshake, HTTP, encapsulation)
4. Distinguish between TCP and UDP and identify appropriate use cases for each
5. Explain the function of key supporting protocols (ICMP, ARP, DHCP)
6. Identify major application-layer protocols and their port numbers

---

While the OSI model provides the theoretical vocabulary, the TCP/IP model is the protocol stack that actually powers the Internet. Every packet you send -- whether a web request, a video stream, or a DNS lookup -- is processed by this four-layer architecture. Understanding TCP/IP is not just academic; it is the practical foundation for configuring networks, debugging connectivity issues, and building networked applications.

## Table of Contents

1. [TCP/IP Model Overview](#1-tcpip-model-overview)
2. [History of the Internet](#2-history-of-the-internet)
3. [TCP/IP 4 Layers](#3-tcpip-4-layers)
4. [OSI vs TCP/IP Comparison](#4-osi-vs-tcpip-comparison)
5. [Protocols at Each Layer](#5-protocols-at-each-layer)
6. [Actual Communication Flow](#6-actual-communication-flow)
7. [TCP vs UDP](#7-tcp-vs-udp)
8. [Practice Problems](#8-practice-problems)

---

## 1. TCP/IP Model Overview

### What is TCP/IP?

```
┌─────────────────────────────────────────────────────────────────┐
│                    TCP/IP Protocol Stack                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   TCP/IP (Transmission Control Protocol / Internet Protocol)    │
│                                                                  │
│   "A collection of communication protocols used for exchanging  │
│    information between computers on the Internet"               │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │  TCP (Transmission Control Protocol)                    │   │
│   │  └── Ensures reliable data transmission                 │   │
│   │  └── Connection-oriented, flow control, error recovery  │   │
│   │                                                          │   │
│   │  IP (Internet Protocol)                                 │   │
│   │  └── Packet addressing and routing                      │   │
│   │  └── Connectionless, best-effort delivery               │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Characteristics:                                              │
│   - Practical implementation-focused model                      │
│   - Flexible boundaries between layers                          │
│   - De facto standard for the Internet                          │
│   - Open standard (published as RFC documents)                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### TCP/IP 4-Layer Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                     TCP/IP 4-Layer Model                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Layer #     Layer Name              Main Role                 │
│   ────────────────────────────────────────────────────────────  │
│                                                                  │
│     ┌─────────────────────────────────────────────────────┐     │
│   4 │           Application Layer                          │     │
│     │      HTTP, FTP, SMTP, DNS, SSH, Telnet              │     │
│     │      User applications and network interface        │     │
│     ├─────────────────────────────────────────────────────┤     │
│   3 │           Transport Layer                            │     │
│     │      TCP, UDP                                        │     │
│     │      End-to-end communication, reliable/unreliable  │     │
│     ├─────────────────────────────────────────────────────┤     │
│   2 │          Internet Layer                              │     │
│     │      IP, ICMP, ARP, RARP                             │     │
│     │      Logical addressing, packet routing             │     │
│     ├─────────────────────────────────────────────────────┤     │
│   1 │      Network Access Layer                            │     │
│     │      Ethernet, Wi-Fi, PPP                            │     │
│     │      Physical transmission, frame delivery           │     │
│     └─────────────────────────────────────────────────────┘     │
│                                                                  │
│   * Some documents divide into 5 layers                         │
│     (separating Network Access into Data Link + Physical)       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### TCP/IP Design Principles

```
┌─────────────────────────────────────────────────────────────────┐
│                   TCP/IP Design Principles                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. End-to-End Principle                                       │
│      ┌───────────────────────────────────────────────────┐      │
│      │  Intelligence (complex functions) at endpoints     │      │
│      │  Keep network core (routers) simple                │      │
│      │                                                    │      │
│      │  Host A ─────[Router]─────[Router]───── Host B     │      │
│      │  (Complex)   (Simple)     (Simple)     (Complex)   │      │
│      └───────────────────────────────────────────────────┘      │
│                                                                  │
│   2. Robustness Principle                                       │
│      ┌───────────────────────────────────────────────────┐      │
│      │  "Be conservative in what you send,                │      │
│      │   be liberal in what you accept"                   │      │
│      └───────────────────────────────────────────────────┘      │
│                                                                  │
│   3. Layering Principle                                         │
│      ┌───────────────────────────────────────────────────┐      │
│      │  Each layer operates independently                 │      │
│      │  Lower layers provide services to upper layers     │      │
│      │  Standardized interfaces between layers            │      │
│      └───────────────────────────────────────────────────┘      │
│                                                                  │
│   4. Packet Switching Principle                                 │
│      ┌───────────────────────────────────────────────────┐      │
│      │  Divide data into small packets for transmission   │      │
│      │  Each packet routed independently                  │      │
│      │  Efficient use of network resources                │      │
│      └───────────────────────────────────────────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. History of the Internet

### Internet Development Timeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     Internet Timeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1960s: Conceptual Beginning                                    │
│  ─────────────────────────────────────────────────────────────  │
│  1962  J.C.R. Licklider's "Intergalactic Network" concept       │
│  1965  First WAN connection (MIT-UCLA)                          │
│  1969  ARPANET begins (4 nodes)                                 │
│        - UCLA, SRI, UCSB, Utah                                  │
│                                                                  │
│  1970s: Protocol Development                                    │
│  ─────────────────────────────────────────────────────────────  │
│  1971  Email invented (Ray Tomlinson)                           │
│  1973  TCP/IP concept proposed (Vint Cerf, Bob Kahn)            │
│  1974  TCP specification published                              │
│  1976  Ethernet developed (Xerox PARC)                          │
│                                                                  │
│  1980s: Standardization and Expansion                           │
│  ─────────────────────────────────────────────────────────────  │
│  1981  IPv4 standardized (RFC 791)                              │
│  1983  TCP/IP adopted (ARPANET), DNS introduced                 │
│        ★ Official birth of the Internet                         │
│  1986  NSFNET launched (56 Kbps backbone)                       │
│  1989  WWW invented (Tim Berners-Lee, CERN)                     │
│                                                                  │
│  1990s: Commercialization and Popularization                    │
│  ─────────────────────────────────────────────────────────────  │
│  1991  WWW released publicly, Gopher appears                    │
│  1993  Mosaic web browser released                              │
│  1994  Netscape Navigator released                              │
│  1995  Commercial Internet spreads, Amazon, eBay founded        │
│  1998  Google founded, IPv6 standardized                        │
│                                                                  │
│  2000s: Mobile and Social                                       │
│  ─────────────────────────────────────────────────────────────  │
│  2004  Facebook founded, Web 2.0                                │
│  2005  YouTube founded                                          │
│  2007  iPhone released, mobile Internet popularized             │
│                                                                  │
│  2010s-Present: Cloud and IoT                                   │
│  ─────────────────────────────────────────────────────────────  │
│  2010  Cloud computing spreads                                  │
│  2016  IPv4 exhaustion, IPv6 transition accelerates             │
│  2020  5G commercialization, hyperconnected era                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### From ARPANET to Internet

```
┌─────────────────────────────────────────────────────────────────┐
│                 ARPANET → Internet Evolution                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1969: ARPANET begins (4 nodes)                                │
│                                                                  │
│                    ┌─────┐                                      │
│                    │UCLA │                                      │
│                    └──┬──┘                                      │
│                       │                                          │
│        ┌──────────────┼──────────────┐                          │
│        │              │              │                          │
│     ┌──┴──┐       ┌───┴───┐      ┌───┴───┐                     │
│     │ SRI │       │ UCSB  │      │ Utah  │                     │
│     └─────┘       └───────┘      └───────┘                     │
│                                                                  │
│   1983: TCP/IP transition                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  NCP (Network Control Protocol) → TCP/IP                │   │
│   │                                                          │   │
│   │  - This transition is considered the "birth of Internet"│   │
│   │  - Enabled heterogeneous network connectivity           │   │
│   │  - Achieved scalability                                 │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   1990s: Commercial Internet                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │       ARPANET     NSFNET      Commercial ISPs           │   │
│   │       (Military/ (Academic) → (General Public)          │   │
│   │        Academic)                                         │   │
│   │   ┌─────────┐    ┌─────────┐    ┌─────────┐            │   │
│   │   │Universities│  │Research │    │ Home/   │            │   │
│   │   │ Research │    │Institutes│   │Business │            │   │
│   │   └─────────┘    └─────────┘    └─────────┘            │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Present: Global Internet                                      │
│   - Billions of devices connected                               │
│   - Continents linked by submarine cables                       │
│   - IoT, Cloud, 5G/6G                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Figures in TCP/IP Development

| Person | Contribution | Major Achievement |
|--------|--------------|-------------------|
| Vint Cerf | TCP/IP co-inventor | "Father of the Internet" |
| Bob Kahn | TCP/IP co-inventor | ARPANET design |
| Tim Berners-Lee | WWW inventor | HTTP, HTML |
| Ray Tomlinson | Email inventor | Introduced @ symbol |
| Bob Metcalfe | Ethernet inventor | LAN standard |
| Jon Postel | Protocol standardization | RFC editor |

---

## 3. TCP/IP 4 Layers

### Layer 4: Application Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                4th Layer: Application Layer                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Role: Interface between users and network                     │
│         Application protocol implementation                     │
│                                                                  │
│   Relationship with OSI model:                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   OSI 7 layers: Application + Presentation + Session    │   │
│   │        ↓                                                 │   │
│   │   TCP/IP: Application Layer (combined into one)         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Main Protocols:                                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   ┌──────────┐  ┌──────────┐  ┌──────────┐             │   │
│   │   │   HTTP   │  │   FTP    │  │   SMTP   │             │   │
│   │   │Web comm. │  │File xfer │  │Mail send │             │   │
│   │   │ TCP/80   │  │ TCP/21   │  │ TCP/25   │             │   │
│   │   └──────────┘  └──────────┘  └──────────┘             │   │
│   │                                                          │   │
│   │   ┌──────────┐  ┌──────────┐  ┌──────────┐             │   │
│   │   │   DNS    │  │   SSH    │  │   DHCP   │             │   │
│   │   │Name res. │  │Sec.access│  │IP assign │             │   │
│   │   │ UDP/53   │  │ TCP/22   │  │UDP/67,68 │             │   │
│   │   └──────────┘  └──────────┘  └──────────┘             │   │
│   │                                                          │   │
│   │   ┌──────────┐  ┌──────────┐  ┌──────────┐             │   │
│   │   │   POP3   │  │   IMAP   │  │  Telnet  │             │   │
│   │   │Mail recv.│  │Mail acc. │  │Remote acc│             │   │
│   │   │ TCP/110  │  │ TCP/143  │  │ TCP/23   │             │   │
│   │   └──────────┘  └──────────┘  └──────────┘             │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Relationship with Transport Layer:                            │
│   - Applications access transport layer through sockets         │
│   - Port numbers identify processes                             │
│   - Choose between TCP or UDP                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 3: Transport Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                3rd Layer: Transport Layer                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Role: End-to-End data transmission service                    │
│                                                                  │
│   Two Main Protocols:                                           │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │           TCP                        UDP                 │   │
│   │   ┌─────────────────┐       ┌─────────────────┐         │   │
│   │   │ Connection-     │       │ Connectionless  │         │   │
│   │   │  oriented       │       │                 │         │   │
│   │   │ Reliable        │       │ Unreliable      │         │   │
│   │   │ Ordered         │       │ Unordered       │         │   │
│   │   │ Flow/congestion │       │ No control      │         │   │
│   │   │  control        │       │                 │         │   │
│   │   │ Higher overhead │       │ Lower overhead  │         │   │
│   │   └─────────────────┘       └─────────────────┘         │   │
│   │                                                          │   │
│   │   Used for: Web,            Used for: DNS, streaming,   │   │
│   │   email, file transfer      gaming, VoIP                │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Port Numbers:                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   ┌─────────────────────────────────────────────────┐   │   │
│   │   │        Range         │          Usage            │   │   │
│   │   ├─────────────────────────────────────────────────┤   │   │
│   │   │  0 ~ 1023            │  Well-known (System)     │   │   │
│   │   │  1024 ~ 49151        │  Registered              │   │   │
│   │   │  49152 ~ 65535       │  Dynamic (Ephemeral)     │   │   │
│   │   └─────────────────────────────────────────────────┘   │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Socket:                                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   IP Address + Port Number = Socket                     │   │
│   │   Example: 192.168.1.100:8080                           │   │
│   │                                                          │   │
│   │   Socket pair identifies connection:                    │   │
│   │   (Source IP:Source Port, Dest IP:Dest Port)            │   │
│   │   Example: (192.168.1.100:50000, 93.184.216.34:443)     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 2: Internet Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                2nd Layer: Internet Layer                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Role: Logical addressing (IP), packet routing                 │
│         Data delivery across heterogeneous networks             │
│                                                                  │
│   Core: IP (Internet Protocol)                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Characteristics:                                       │   │
│   │   - Connectionless                                       │   │
│   │   - Best-effort Delivery                                 │   │
│   │   - No reliability guarantee (handled by TCP)           │   │
│   │                                                          │   │
│   │   Functions:                                             │   │
│   │   - Addressing (IP addresses)                           │   │
│   │   - Routing (path determination)                        │   │
│   │   - Fragmentation                                       │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   IP Packet Structure (IPv4):                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │    0                   1                   2             3   │
│   │    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7  │
│   │   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+  │
│   │   |Ver| IHL |  TOS  |         Total Length              |  │
│   │   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+  │
│   │   |      Identification      |Flg|   Fragment Offset    |  │
│   │   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+  │
│   │   |   TTL   | Protocol |       Header Checksum           |  │
│   │   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+  │
│   │   |                  Source IP Address                   |  │
│   │   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+  │
│   │   |               Destination IP Address                 |  │
│   │   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+  │
│   │   |                     Options + Data                   |  │
│   │   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+  │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Main Protocols:                                               │
│   ┌──────────┬───────────────────────────────────────────────┐  │
│   │   IP     │ Addressing, routing (IPv4, IPv6)             │  │
│   │   ICMP   │ Error reporting, diagnostics (ping, tracert) │  │
│   │   ARP    │ IP address → MAC address translation         │  │
│   │   RARP   │ MAC address → IP address translation         │  │
│   │   IGMP   │ Multicast group management                   │  │
│   └──────────┴───────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 1: Network Access Layer

```
┌─────────────────────────────────────────────────────────────────┐
│           1st Layer: Network Access Layer                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Role: Physical network access, frame transmission             │
│         Corresponds to OSI Data Link + Physical layers          │
│                                                                  │
│   Relationship with OSI model:                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   OSI Layer 2: Data Link                                │   │
│   │   OSI Layer 1: Physical                                 │   │
│   │        ↓                                                 │   │
│   │   TCP/IP: Network Access Layer (combined)               │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Functions:                                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   1. Physical Addressing (MAC Address)                  │   │
│   │      - 48-bit hardware address                          │   │
│   │      - Example: 00:1A:2B:3C:4D:5E                       │   │
│   │                                                          │   │
│   │   2. Framing                                             │   │
│   │      - Encapsulate IP packets into frames               │   │
│   │      - Add header and trailer                           │   │
│   │                                                          │   │
│   │   3. Media Access Control                                │   │
│   │      - CSMA/CD (Ethernet)                               │   │
│   │      - CSMA/CA (Wi-Fi)                                  │   │
│   │                                                          │   │
│   │   4. Bit Transmission                                    │   │
│   │      - Convert to electrical/optical/radio signals      │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Technologies/Protocols:                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Wired:                                                 │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│   │   │  Ethernet    │  │    PPP       │  │  FDDI        │  │   │
│   │   │  (802.3)     │  │(Point-to-Pt) │  │ (Fiber)      │  │   │
│   │   └──────────────┘  └──────────────┘  └──────────────┘  │   │
│   │                                                          │   │
│   │   Wireless:                                              │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│   │   │   Wi-Fi      │  │  Bluetooth   │  │     5G       │  │   │
│   │   │  (802.11)    │  │  (802.15)    │  │   (NR)       │  │   │
│   │   └──────────────┘  └──────────────┘  └──────────────┘  │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. OSI vs TCP/IP Comparison

### Layer Structure Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    OSI vs TCP/IP Layers                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│        OSI 7 Layers                   TCP/IP 4 Layers           │
│                                                                  │
│   ┌─────────────────┐                                           │
│   │ 7. Application  │                                           │
│   ├─────────────────┤            ┌─────────────────┐            │
│   │ 6. Presentation │  ───────►  │ 4. Application  │            │
│   ├─────────────────┤            └─────────────────┘            │
│   │ 5. Session      │                                           │
│   ├─────────────────┤            ┌─────────────────┐            │
│   │ 4. Transport    │  ───────►  │ 3. Transport    │            │
│   ├─────────────────┤            └─────────────────┘            │
│   │ 3. Network      │  ───────►  ┌─────────────────┐            │
│   ├─────────────────┤            │ 2. Internet     │            │
│   │ 2. Data Link    │            └─────────────────┘            │
│   ├─────────────────┤  ───────►  ┌─────────────────┐            │
│   │ 1. Physical     │            │ 1. Network      │            │
│   └─────────────────┘            │    Access       │            │
│                                  └─────────────────┘            │
│                                                                  │
│   OSI: 7 layers (Theoretical)    TCP/IP: 4 layers (Practical)  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed Comparison Table

```
┌─────────────────────────────────────────────────────────────────┐
│                   OSI vs TCP/IP Detailed Comparison              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┬─────────────────┬────────────────────────┐   │
│   │     Aspect   │      OSI        │       TCP/IP           │   │
│   ├──────────────┼─────────────────┼────────────────────────┤   │
│   │ # of Layers  │      7          │        4               │   │
│   ├──────────────┼─────────────────┼────────────────────────┤   │
│   │ Developer    │      ISO        │   DARPA (US DoD)       │   │
│   ├──────────────┼─────────────────┼────────────────────────┤   │
│   │ Timeline     │    1984         │      1970s             │   │
│   ├──────────────┼─────────────────┼────────────────────────┤   │
│   │ Purpose      │ Reference model │   Actual implementation│   │
│   ├──────────────┼─────────────────┼────────────────────────┤   │
│   │ Nature       │Theoretical/     │   Practical/Standard   │   │
│   │              │Educational      │                        │   │
│   ├──────────────┼─────────────────┼────────────────────────┤   │
│   │ Protocols    │Defined separate │   Defined together     │   │
│   ├──────────────┼─────────────────┼────────────────────────┤   │
│   │ Layer bounds │   Clear         │      Flexible          │   │
│   ├──────────────┼─────────────────┼────────────────────────┤   │
│   │ Reliability  │Transport layer  │   Transport layer only │   │
│   │              │     only        │                        │   │
│   ├──────────────┼─────────────────┼────────────────────────┤   │
│   │ Current use  │Reference/       │   Internet standard    │   │
│   │              │Educational      │                        │   │
│   └──────────────┴─────────────────┴────────────────────────┘   │
│                                                                  │
│   Key Differences:                                              │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 1. OSI: Model before protocols (Top-down)               │   │
│   │    TCP/IP: Protocols before model (Bottom-up)           │   │
│   │                                                          │   │
│   │ 2. OSI: Each layer defined independently                │   │
│   │    TCP/IP: Considers inter-layer interaction            │   │
│   │                                                          │   │
│   │ 3. OSI Session/Presentation layers integrated           │   │
│   │    into TCP/IP Application layer                        │   │
│   │                                                          │   │
│   │ 4. TCP/IP is de facto Internet standard                 │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why TCP/IP Prevailed

```
┌─────────────────────────────────────────────────────────────────┐
│                  TCP/IP Success Factors                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Practicality                                               │
│      ┌─────────────────────────────────────────────────────┐    │
│      │ - Implementation first, then model                   │    │
│      │ - Evolved with working code                          │    │
│      │ - "Rough consensus and running code"                 │    │
│      └─────────────────────────────────────────────────────┘    │
│                                                                  │
│   2. Openness                                                   │
│      ┌─────────────────────────────────────────────────────┐    │
│      │ - Free to use for anyone                             │    │
│      │ - Standards published as RFCs                        │    │
│      │ - No proprietary technology                          │    │
│      └─────────────────────────────────────────────────────┘    │
│                                                                  │
│   3. Flexibility                                                │
│      ┌─────────────────────────────────────────────────────┐    │
│      │ - Accommodates various network technologies          │    │
│      │ - Easy to add new applications                       │    │
│      │ - Connects heterogeneous systems                     │    │
│      └─────────────────────────────────────────────────────┘    │
│                                                                  │
│   4. First-mover Advantage                                      │
│      ┌─────────────────────────────────────────────────────┐    │
│      │ - Already in use on ARPANET                          │    │
│      │ - Early adoption by universities and research labs   │    │
│      │ - Included in BSD Unix by default                    │    │
│      └─────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Protocols at Each Layer

### Complete Protocol Stack Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                   TCP/IP Protocol Stack                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                     Application Layer                    │   │
│   │                                                          │   │
│   │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │   │
│   │  │HTTP │ │ FTP │ │SMTP │ │ DNS │ │ SSH │ │DHCP │       │   │
│   │  │HTTPS│ │SFTP │ │POP3 │ │     │ │     │ │     │       │   │
│   │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘       │   │
│   │     │       │       │       │       │       │            │   │
│   └─────┼───────┼───────┼───────┼───────┼───────┼────────────┘   │
│         │       │       │       │       │       │                │
│   ┌─────┼───────┼───────┼───────┼───────┼───────┼────────────┐   │
│   │     ▼       ▼       ▼       │       ▼       │ Transport  │   │
│   │  ┌─────────────────────────┐│   ┌───────────┐            │   │
│   │  │          TCP            ││   │    UDP    │            │   │
│   │  │ (Connection-oriented,   ││   │(Connless) │            │   │
│   │  │  Reliable)               ││   │           │            │   │
│   │  └───────────┬─────────────┘│   └─────┬─────┘            │   │
│   │              │              │         │                   │   │
│   └──────────────┼──────────────┼─────────┼───────────────────┘   │
│                  │              │         │                       │
│   ┌──────────────┼──────────────┼─────────┼───────────────────┐   │
│   │              ▼              ▼         ▼      Internet     │   │
│   │            ┌─────────────────────────────┐                │   │
│   │            │            IP               │                │   │
│   │            │    (Addressing, Routing)    │                │   │
│   │            └──────────────┬──────────────┘                │   │
│   │                           │                               │   │
│   │    ┌──────────┬───────────┼───────────┬──────────┐       │   │
│   │    │   ICMP   │    ARP    │   RARP    │   IGMP   │       │   │
│   │    └──────────┴───────────┴───────────┴──────────┘       │   │
│   │                           │                               │   │
│   └───────────────────────────┼───────────────────────────────┘   │
│                               │                                   │
│   ┌───────────────────────────┼───────────────────────────────┐   │
│   │                           ▼         Network Access Layer  │   │
│   │    ┌───────────┐   ┌───────────┐   ┌───────────┐         │   │
│   │    │  Ethernet │   │   Wi-Fi   │   │    PPP    │         │   │
│   │    │  (802.3)  │   │  (802.11) │   │           │         │   │
│   │    └───────────┘   └───────────┘   └───────────┘         │   │
│   │                                                           │   │
│   └───────────────────────────────────────────────────────────┘   │
│                               │                                   │
│                               ▼                                   │
│                      Physical Transmission Media                 │
│                (Cables, Fiber optics, Radio waves)               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Major Protocol Details

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer Protocols                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Protocol    Port     Transport  Description                   │
│   ─────────────────────────────────────────────────────────────│
│   HTTP        80       TCP        Hypertext transfer            │
│   HTTPS       443      TCP        Secure HTTP (TLS)             │
│   FTP         20/21    TCP        File transfer (data/control)  │
│   SSH         22       TCP        Secure remote access          │
│   Telnet      23       TCP        Remote access (insecure)      │
│   SMTP        25       TCP        Mail sending                  │
│   DNS         53       UDP/TCP    Domain name resolution        │
│   DHCP        67/68    UDP        Automatic IP assignment       │
│   TFTP        69       UDP        Trivial file transfer         │
│   HTTP/3      443      QUIC       Next-gen HTTP                 │
│   POP3        110      TCP        Mail retrieval                │
│   IMAP        143      TCP        Mail access                   │
│   SNMP        161/162  UDP        Network management            │
│   LDAP        389      TCP        Directory service             │
│   SMTPS       465      TCP        Secure mail sending           │
│   IMAPS       993      TCP        Secure mail access            │
│   POP3S       995      TCP        Secure mail retrieval         │
│   MySQL       3306     TCP        Database                      │
│   RDP         3389     TCP        Remote desktop                │
│   PostgreSQL  5432     TCP        Database                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### ICMP (Internet Control Message Protocol)

> **Why does ICMP exist as a separate protocol?** IP itself has no error reporting — when a packet is dropped (TTL expired, destination unreachable), the sender would never know without ICMP. ICMP rides inside IP packets but serves as the network's *feedback channel*: `ping` uses Echo Request/Reply to test reachability, while `traceroute` deliberately triggers TTL expiry at each hop to map the path. Without ICMP, network troubleshooting would be blind guesswork.

```
┌─────────────────────────────────────────────────────────────────┐
│                          ICMP                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Role: Network error reporting and diagnostics                 │
│                                                                  │
│   Main Message Types:                                           │
│   ┌──────────┬───────────────────────────────────────────────┐  │
│   │ Type 0   │ Echo Reply (ping response)                    │  │
│   │ Type 3   │ Destination Unreachable                       │  │
│   │ Type 5   │ Redirect (route change advice)                │  │
│   │ Type 8   │ Echo Request (ping request)                   │  │
│   │ Type 11  │ Time Exceeded (TTL expired)                   │  │
│   └──────────┴───────────────────────────────────────────────┘  │
│                                                                  │
│   ping command:                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   $ ping 8.8.8.8                                         │   │
│   │   PING 8.8.8.8 (8.8.8.8): 56 data bytes                 │   │
│   │   64 bytes from 8.8.8.8: icmp_seq=0 ttl=117 time=9.2 ms │   │
│   │   64 bytes from 8.8.8.8: icmp_seq=1 ttl=117 time=8.9 ms │   │
│   │                                                          │   │
│   │   Type 8 (request) → destination → Type 0 (response)    │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   traceroute command:                                           │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   Send packet with TTL=1 → expires at first router      │   │
│   │   Send packet with TTL=2 → expires at second router     │   │
│   │   ...repeat to discover all routers in path             │   │
│   │                                                          │   │
│   │   $ traceroute google.com                                │   │
│   │   1  192.168.1.1    1.234 ms                             │   │
│   │   2  10.0.0.1       5.678 ms                             │   │
│   │   3  ...                                                 │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### ARP (Address Resolution Protocol)

```
┌─────────────────────────────────────────────────────────────────┐
│                           ARP                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Role: IP address → MAC address translation                    │
│                                                                  │
│   Operation Process:                                            │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   1. ARP Request (Broadcast)                            │   │
│   │                                                          │   │
│   │   Host A                        All hosts on network    │   │
│   │   ┌─────────┐   "What's the     ┌─────────┐           │   │
│   │   │ IP:     │    MAC address    │ IP:     │           │   │
│   │   │192.168  │    of 192.168     │192.168  │           │   │
│   │   │ .1.1    │    .1.2?"         │ .1.2    │           │   │
│   │   │ MAC:    │ ═══════════════════►│ MAC:    │           │   │
│   │   │ AA:BB:  │                     │ CC:DD:  │           │   │
│   │   │ CC:DD:  │                     │ EE:FF:  │           │   │
│   │   │ EE:FF   │                     │ 00:11   │           │   │
│   │   └─────────┘                     └─────────┘           │   │
│   │                                                          │   │
│   │   2. ARP Reply (Unicast)                                │   │
│   │                                                          │   │
│   │   Host A                        Host B                  │   │
│   │   ┌─────────┐   "My MAC addr is ┌─────────┐           │   │
│   │   │         │◄═══════════════════ │ CC:DD:  │           │   │
│   │   │         │    CC:DD:EE:FF:     │ EE:FF:  │           │   │
│   │   │         │    00:11"           │ 00:11   │           │   │
│   │   └─────────┘                     └─────────┘           │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   ARP Table:                                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   $ arp -a                                               │   │
│   │   Internet Address    Physical Address    Type           │   │
│   │   192.168.1.1         aa-bb-cc-dd-ee-ff   dynamic       │   │
│   │   192.168.1.2         cc-dd-ee-ff-00-11   dynamic       │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   * ARP cache: Stores results for efficiency                    │
│   * ARP only works within same network (subnet)                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Actual Communication Flow

### Complete Web Page Request Process

```
┌─────────────────────────────────────────────────────────────────┐
│            www.example.com Web Page Request Process              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   User: Enter www.example.com in browser                        │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Step 1: DNS lookup (domain → IP)                       │   │
│   │                                                          │   │
│   │  Browser → DNS server                                   │   │
│   │  "What's the IP of www.example.com?"                    │   │
│   │                                                          │   │
│   │  DNS server → Browser                                   │   │
│   │  "93.184.216.34"                                        │   │
│   └─────────────────────────────────────────────────────────┘   │
│                           ↓                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Step 2: TCP connection (3-way handshake)               │   │
│   │                                                          │   │
│   │  Client                        Server                   │   │
│   │      │ ──────── SYN ─────────► │                        │   │
│   │      │ ◄─────── SYN-ACK ─────── │                        │   │
│   │      │ ──────── ACK ─────────► │                        │   │
│   │                                                          │   │
│   │  Connection established (session started)               │   │
│   └─────────────────────────────────────────────────────────┘   │
│                           ↓                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Step 3: HTTP request                                   │   │
│   │                                                          │   │
│   │  Client → Server                                        │   │
│   │  ┌─────────────────────────────────────────────────┐    │   │
│   │  │ GET / HTTP/1.1                                   │    │   │
│   │  │ Host: www.example.com                            │    │   │
│   │  │ User-Agent: Mozilla/5.0...                       │    │   │
│   │  │ Accept: text/html...                             │    │   │
│   │  └─────────────────────────────────────────────────┘    │   │
│   └─────────────────────────────────────────────────────────┘   │
│                           ↓                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Step 4: HTTP response                                  │   │
│   │                                                          │   │
│   │  Server → Client                                        │   │
│   │  ┌─────────────────────────────────────────────────┐    │   │
│   │  │ HTTP/1.1 200 OK                                  │    │   │
│   │  │ Content-Type: text/html                          │    │   │
│   │  │ Content-Length: 1256                             │    │   │
│   │  │                                                  │    │   │
│   │  │ <!DOCTYPE html>                                  │    │   │
│   │  │ <html>...                                        │    │   │
│   │  └─────────────────────────────────────────────────┘    │   │
│   └─────────────────────────────────────────────────────────┘   │
│                           ↓                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Step 5: TCP connection termination (4-way handshake)   │   │
│   │                                                          │   │
│   │  Client                        Server                   │   │
│   │      │ ──────── FIN ─────────► │                        │   │
│   │      │ ◄─────── ACK ─────────── │                        │   │
│   │      │ ◄─────── FIN ─────────── │                        │   │
│   │      │ ──────── ACK ─────────► │                        │   │
│   │                                                          │   │
│   │  Connection closed                                      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed Encapsulation Process

```
┌─────────────────────────────────────────────────────────────────┐
│                      Detailed Encapsulation                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Application Layer (HTTP request)                              │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              HTTP DATA (GET / HTTP/1.1...)               │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│   Transport Layer (TCP)                                         │
│   ┌────────────┬────────────────────────────────────────────┐   │
│   │ TCP Header │              HTTP DATA                      │   │
│   │ Src: 50000 │                                             │   │
│   │ Dst: 80    │                                             │   │
│   │ Seq: 1000  │                                             │   │
│   └────────────┴────────────────────────────────────────────┘   │
│                              ↓                                   │
│   Internet Layer (IP)                                           │
│   ┌────────────┬────────────┬────────────────────────────────┐  │
│   │ IP Header  │ TCP Header │          HTTP DATA             │  │
│   │ Src: 192.  │            │                                │  │
│   │ 168.1.100  │            │                                │  │
│   │ Dst: 93.   │            │                                │  │
│   │ 184.216.34 │            │                                │  │
│   │ TTL: 64    │            │                                │  │
│   └────────────┴────────────┴────────────────────────────────┘  │
│                              ↓                                   │
│   Network Access Layer (Ethernet)                               │
│   ┌──────┬────────────┬────────────┬──────────────────┬─────┐   │
│   │Pream │Eth Header  │ IP Header  │ TCP + HTTP DATA  │ FCS │   │
│   │ble   │Src MAC:    │            │                  │     │   │
│   │      │aa:bb:cc... │            │                  │     │   │
│   │      │Dst MAC:    │            │                  │     │   │
│   │      │11:22:33... │            │                  │     │   │
│   └──────┴────────────┴────────────┴──────────────────┴─────┘   │
│                              ↓                                   │
│   Physical Layer (Bit stream)                                   │
│   10110100 01101011 11010010 10101100 01011001 ...              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Routing Process

```
┌─────────────────────────────────────────────────────────────────┐
│                       Routing Process                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Source: 192.168.1.100                                         │
│   Destination: 93.184.216.34                                    │
│                                                                  │
│   ┌─────────────┐                                               │
│   │   Client    │  192.168.1.100                                │
│   └──────┬──────┘                                               │
│          │                                                       │
│          │ ① Not same network, send to default gateway         │
│          │    (Get gateway MAC via ARP)                         │
│          ▼                                                       │
│   ┌─────────────┐                                               │
│   │  Router A   │  192.168.1.1 (internal) / 10.0.0.1 (external)│
│   └──────┬──────┘                                               │
│          │                                                       │
│          │ ② Check routing table, determine next hop           │
│          │    Decrement TTL in IP header (64 → 63)             │
│          │    Create new Ethernet frame                         │
│          ▼                                                       │
│   ┌─────────────┐                                               │
│   │  Router B   │  ISP router                                   │
│   └──────┬──────┘                                               │
│          │                                                       │
│          │ ③ Continue routing...                                │
│          │    (TTL decrement, frame recreation)                 │
│          ▼                                                       │
│        ......                                                   │
│          │                                                       │
│          │ ④ Reach destination network                         │
│          ▼                                                       │
│   ┌─────────────┐                                               │
│   │   Server    │  93.184.216.34                                │
│   └─────────────┘                                               │
│                                                                  │
│   At each hop:                                                  │
│   - IP packet unchanged (except TTL, checksum)                  │
│   - Ethernet frame recreated (MAC addresses changed)            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. TCP vs UDP

### TCP Details

```
┌─────────────────────────────────────────────────────────────────┐
│              TCP (Transmission Control Protocol)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Characteristics: Connection-oriented, reliability, ordering   │
│                                                                  │
│   3-Way Handshake (Connection establishment):                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Client                           Server               │   │
│   │       │                                  │               │   │
│   │       │ ─────── SYN (seq=x) ──────────► │               │   │
│   │       │                                  │               │   │
│   │       │ ◄────── SYN-ACK ──────────────  │               │   │
│   │       │         (seq=y, ack=x+1)         │               │   │
│   │       │                                  │               │   │
│   │       │ ─────── ACK (ack=y+1) ────────► │               │   │
│   │       │                                  │               │   │
│   │       │      Connection established       │               │   │
│   │       │                                  │               │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Data Transmission (Sliding window):                           │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Send window (Window Size = 4)                         │   │
│   │                                                          │   │
│   │   ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐            │   │
│   │   │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │            │   │
│   │   └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘            │   │
│   │     ▲                   ▲                               │   │
│   │     └─────┬─────────────┘                               │   │
│   │           │                                              │   │
│   │   Can send without ACK                                  │   │
│   │                                                          │   │
│   │   Window "slides" forward when ACK received             │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   4-Way Handshake (Connection termination):                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Client                           Server               │   │
│   │       │                                  │               │   │
│   │       │ ─────── FIN ────────────────► │               │   │
│   │       │                                  │               │   │
│   │       │ ◄────── ACK ────────────────  │               │   │
│   │       │                                  │               │   │
│   │       │ ◄────── FIN ────────────────  │               │   │
│   │       │                                  │               │   │
│   │       │ ─────── ACK ────────────────► │               │   │
│   │       │                                  │               │   │
│   │       │      Connection closed           │               │   │
│   │       │                                  │               │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Congestion Control:                                           │
│   - Slow Start: Small initial window, exponential growth        │
│   - Congestion Avoidance: Linear growth after threshold         │
│   - Fast Retransmit: Immediate retransmit on 3 duplicate ACKs   │
│   - Fast Recovery: Quick recovery after congestion              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### UDP Details

```
┌─────────────────────────────────────────────────────────────────┐
│                UDP (User Datagram Protocol)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Characteristics: Connectionless, unreliable, fast             │
│                                                                  │
│   Communication Method:                                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   Sender                               Receiver          │   │
│   │       │                                  │               │   │
│   │       │ ═══════ Datagram 1 ═══════► │               │   │
│   │       │ ═══════ Datagram 2 ═══════► │               │   │
│   │       │ ═══════ Datagram 3 ═══════► │               │   │
│   │       │                                  │               │   │
│   │       │      (No ACK, no connection)      │               │   │
│   │       │                                  │               │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   UDP Header (8 bytes):                                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   0       8      16      24      32                      │   │
│   │   ┌───────────────┬───────────────┐                     │   │
│   │   │   Source Port │    Dest Port   │                     │   │
│   │   ├───────────────┼───────────────┤                     │   │
│   │   │     Length    │    Checksum    │                     │   │
│   │   └───────────────┴───────────────┘                     │   │
│   │                                                          │   │
│   │   Much simpler than TCP header (20+ bytes)              │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Use Cases:                                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │   1. DNS queries                                        │   │
│   │      - Small data, fast response needed                 │   │
│   │      - Retry on failure (application level)             │   │
│   │                                                          │   │
│   │   2. Video/Audio streaming                              │   │
│   │      - Real-time important                              │   │
│   │      - Some loss acceptable                             │   │
│   │                                                          │   │
│   │   3. Online gaming                                      │   │
│   │      - Low latency required                             │   │
│   │      - Latest data more important than old              │   │
│   │                                                          │   │
│   │   4. VoIP                                               │   │
│   │      - Real-time voice calls                            │   │
│   │      - Better to skip than retransmit                   │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### TCP vs UDP Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                      TCP vs UDP Comparison                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────┬─────────────────┬─────────────────────┐   │
│   │    Feature      │      TCP        │        UDP          │   │
│   ├─────────────────┼─────────────────┼─────────────────────┤   │
│   │ Connection      │ Connection-     │ Connectionless      │   │
│   │                 │  oriented       │                     │   │
│   │ Reliability     │ Guaranteed      │ Not guaranteed      │   │
│   │ Ordering        │ Guaranteed      │ Not guaranteed      │   │
│   │ Flow control    │ Yes             │ No                  │   │
│   │ Congestion ctrl │ Yes             │ No                  │   │
│   │ Overhead        │ High (20+ bytes)│ Low (8 bytes)       │   │
│   │ Speed           │ Relatively slow │ Fast                │   │
│   │ Retransmission  │ Automatic       │ None                │   │
│   │ Broadcast       │ No              │ Yes                 │   │
│   │ Multicast       │ No              │ Yes                 │   │
│   ├─────────────────┼─────────────────┼─────────────────────┤   │
│   │                 │ Web (HTTP)      │ DNS                 │   │
│   │ Usage examples  │ Email (SMTP)    │ Streaming           │   │
│   │                 │ File xfer (FTP) │ Gaming              │   │
│   │                 │ SSH             │ VoIP                │   │
│   └─────────────────┴─────────────────┴─────────────────────┘   │
│                                                                  │
│   Selection Criteria:                                           │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ Choose TCP when:                                         │   │
│   │ - Data integrity is critical                            │   │
│   │ - All data must arrive                                  │   │
│   │ - Order matters                                         │   │
│   │                                                          │   │
│   │ Choose UDP when:                                         │   │
│   │ - Real-time is critical                                 │   │
│   │ - Some loss is acceptable                               │   │
│   │ - Broadcast/multicast needed                            │   │
│   │ - Fast response required                                │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Practice Problems

### Basic Problems

**1. List the 4 layers of the TCP/IP model from bottom to top.**

**2. Classify the following protocols as TCP or UDP:**
   - HTTP, DNS, FTP, VoIP, SMTP, online gaming

**3. Choose the protocol that matches each description:**
   - Translates IP address to MAC address: (  )
   - Reports network errors and provides ping: (  )
   - Translates domain name to IP: (  )

   Options: DNS, ARP, ICMP, DHCP

**4. Explain the sequence of the TCP 3-way handshake.**

### Application Problems

**5. Describe the network communication process that occurs when accessing www.google.com in sequence.**

**6. Explain at least 5 differences between TCP and UDP.**

**7. For the following situations, explain which protocol (TCP or UDP) would be appropriate and why:**
   - (a) Online banking service
   - (b) Real-time video conference
   - (c) Large file download

### Advanced Problems

**8. Compare the OSI 7-layer and TCP/IP 4-layer models and explain why TCP/IP became the Internet standard.**

**9. Analyze the following scenario:**
```
When running traceroute, "* * *" appears at the 5th hop.
What are possible causes?
```

**10. Explain why TCP's congestion control mechanisms (Slow Start, Congestion Avoidance) are necessary and how they work.**

---

<details>
<summary>Answers</summary>

**1.** Network Access Layer → Internet Layer → Transport Layer → Application Layer

**2.**
- TCP: HTTP, FTP, SMTP
- UDP: DNS (mostly), VoIP, online gaming

**3.**
- IP → MAC: ARP
- Error reporting/ping: ICMP
- Domain → IP: DNS

**4.**
1. Client → Server: SYN (seq=x)
2. Server → Client: SYN-ACK (seq=y, ack=x+1)
3. Client → Server: ACK (ack=y+1)

**5.**
1. DNS lookup to translate domain to IP
2. TCP 3-way handshake to establish connection
3. TLS handshake (for HTTPS)
4. Send HTTP request
5. Receive HTTP response from server
6. Render web page
7. TCP connection termination (4-way handshake)

**6.**
1. Connection: TCP is connection-oriented, UDP is connectionless
2. Reliability: TCP guarantees, UDP doesn't
3. Ordering: TCP guarantees, UDP doesn't
4. Speed: TCP slower, UDP faster
5. Overhead: TCP 20+ bytes, UDP 8 bytes
6. Flow/congestion control: TCP only
7. Broadcast: UDP only

**7.**
- (a) TCP - Financial data requires integrity and reliability
- (b) UDP - Real-time is critical, some loss acceptable
- (c) TCP - All data must arrive correctly

**8.**
- OSI is theoretical 7 layers, TCP/IP is practical 4 layers
- TCP/IP became standard due to: practicality, openness, flexibility, first-mover advantage
- Evolved on ARPANET with working code

**9.** Possible causes:
- Router blocking ICMP responses
- Firewall filtering ICMP
- Router CPU load preventing ICMP processing
- Network congestion

**10.**
- Necessity: Prevent network congestion, fair bandwidth distribution
- Slow Start: Start window at 1, double every RTT
- Congestion Avoidance: Linear growth after threshold (ssthresh)
- When packet loss detected, reduce window size to alleviate congestion

</details>

---

**Previous**: [OSI 7-Layer Model](./02_OSI_7_Layer_Model.md) | **Next**: [Physical Layer](./04_Physical_Layer.md)

---

## References

- TCP/IP Illustrated (W. Richard Stevens)
- Computer Networking: A Top-Down Approach (Kurose & Ross)
- [RFC 793: TCP](https://tools.ietf.org/html/rfc793)
- [RFC 768: UDP](https://tools.ietf.org/html/rfc768)
- [RFC 791: IP](https://tools.ietf.org/html/rfc791)
