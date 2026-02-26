# Network Fundamentals

**Next**: [OSI 7-Layer Model](./02_OSI_7_Layer_Model.md)

**Difficulty**: ⭐ (Beginner)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what a network is and describe its core components (nodes, links, protocols)
2. Distinguish between PAN, LAN, MAN, and WAN by range and use case
3. Compare network topologies (bus, star, ring, mesh, hybrid) and identify their trade-offs
4. Distinguish between packet switching and circuit switching
5. Compare the client-server and peer-to-peer communication models
6. Identify common network devices and the OSI layer at which each operates

---

Every time you load a web page, send a message, or stream a video, data travels across a network -- often passing through dozens of devices and thousands of kilometers in milliseconds. Understanding how networks are structured, how devices connect, and how data finds its way from sender to receiver is the foundation of every other topic in networking and modern IT infrastructure.

## Table of Contents

1. [What is a Network?](#1-what-is-a-network)
2. [History of Networking](#2-history-of-networking)
3. [Network Types](#3-network-types)
4. [Network Topologies](#4-network-topologies)
5. [Packet Switching vs Circuit Switching](#5-packet-switching-vs-circuit-switching)
6. [Client-Server vs P2P](#6-client-server-vs-p2p)
7. [Network Devices](#7-network-devices)
8. [Practice Problems](#8-practice-problems)

---

## 1. What is a Network?

### Definition of a Network

```
┌─────────────────────────────────────────────────────────────────┐
│                     Network                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   "A system where two or more computers or devices are          │
│    connected through communication media to share data and      │
│    resources"                                                    │
│                                                                  │
│   ┌─────────┐      Communication Media      ┌─────────┐        │
│   │Computer │ ◄──────────────────────────► │Computer │        │
│   │    A    │    (Cable, Wireless, etc.)    │    B    │        │
│   └─────────┘                                └─────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Purpose of Networks

```
┌─────────────────────────────────────────────────────────────────┐
│                    Main Purposes of Networks                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Resource Sharing                                            │
│     └── Share printers, files, software                        │
│                                                                  │
│  2. Communication                                               │
│     └── Email, messaging, video conferencing                   │
│                                                                  │
│  3. Data Sharing                                                │
│     └── File transfer, database access                         │
│                                                                  │
│  4. Centralized Management                                      │
│     └── Security policies, backup, update management           │
│                                                                  │
│  5. Cost Reduction                                              │
│     └── Cost efficiency through equipment sharing              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Network Components

```
┌─────────────────────────────────────────────────────────────────┐
│                   Network Components                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│   │   Sender    │     │    Media    │     │  Receiver   │      │
│   │             │────►│             │────►│             │      │
│   └─────────────┘     └─────────────┘     └─────────────┘      │
│                              │                                   │
│                        ┌─────┴─────┐                            │
│                        │ Protocol  │                            │
│                        │           │                            │
│                        └───────────┘                            │
│                                                                  │
│   Component Descriptions:                                       │
│   ┌──────────────┬──────────────────────────────────────────┐  │
│   │ Node         │ Device connected to network (PC, printer)│  │
│   ├──────────────┼──────────────────────────────────────────┤  │
│   │ Link         │ Physical connection between nodes        │  │
│   ├──────────────┼──────────────────────────────────────────┤  │
│   │ Protocol     │ Communication rules and standards        │  │
│   │              │ (TCP/IP, HTTP)                           │  │
│   ├──────────────┼──────────────────────────────────────────┤  │
│   │ Network      │ Switch, router, hub                      │  │
│   │ Equipment    │                                          │  │
│   └──────────────┴──────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. History of Networking

### Major Timeline

| Year | Event | Description |
|------|-------|-------------|
| 1969 | ARPANET | First packet-switched network, precursor to the Internet |
| 1973 | Ethernet Invented | Xerox PARC, Bob Metcalfe |
| 1974 | TCP/IP Proposed | Vint Cerf, Bob Kahn |
| 1983 | DNS Introduced | Domain Name System |
| 1989 | WWW Invented | Tim Berners-Lee, CERN |
| 1991 | World Wide Web Public | Released to general public |
| 1995 | Commercial Internet | ISPs begin full-scale service |
| 2007 | iPhone Released | Mobile internet popularization |
| 2020s | 5G, IoT | Hyper-connected era |

### From ARPANET to the Internet

```
1969: ARPANET begins (4 nodes)
     ┌─────────┐
     │  UCLA   │
     └────┬────┘
          │
    ┌─────┴─────┬───────────────┐
    │           │               │
┌───┴───┐ ┌─────┴─────┐ ┌───────┴───────┐
│  SRI  │ │   UCSB    │ │ Utah (1969.12)│
└───────┘ └───────────┘ └───────────────┘

1983: TCP/IP adopted → ARPANET transforms into Internet

Present: Global network connecting billions of devices worldwide
```

---

## 3. Network Types

### Classification by Size

```
┌─────────────────────────────────────────────────────────────────┐
│                Network Types (by Scale)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                         WAN                              │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │                       MAN                        │    │    │
│  │  │  ┌─────────────────────────────────────────┐    │    │    │
│  │  │  │                   LAN                    │    │    │    │
│  │  │  │  ┌─────────────────────────────────┐    │    │    │    │
│  │  │  │  │              PAN                 │    │    │    │    │
│  │  │  │  │      (Personal Area)             │    │    │    │    │
│  │  │  │  └─────────────────────────────────┘    │    │    │    │
│  │  │  │        (Building/Campus)                 │    │    │    │
│  │  │  └─────────────────────────────────────────┘    │    │    │
│  │  │                    (City)                        │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  │                    (Country/Continent)                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### PAN (Personal Area Network)

```
┌─────────────────────────────────────────────────────────────────┐
│              PAN (Personal Area Network)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Range: Within about 10m of personal space                     │
│   Purpose: Connection between personal devices                  │
│                                                                  │
│              ┌─────────────┐                                    │
│              │ Smartwatch  │                                    │
│              └──────┬──────┘                                    │
│                     │ Bluetooth                                 │
│   ┌─────────┐   ┌───┴───┐   ┌─────────────┐                    │
│   │ Earbuds │───│Smartphone│──│   Laptop    │                   │
│   └─────────┘   └───┬───┘   └─────────────┘                    │
│                     │                                           │
│              ┌──────┴──────┐                                    │
│              │ Wireless KB │                                    │
│              └─────────────┘                                    │
│                                                                  │
│   Technology: Bluetooth, USB, NFC, Zigbee                       │
│   Examples: Smartphone-earbuds, PC-mouse connection            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### LAN (Local Area Network)

```
┌─────────────────────────────────────────────────────────────────┐
│              LAN (Local Area Network)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Range: Building, campus (meters to kilometers)               │
│   Ownership: Single organization                                │
│   Speed: High-speed (100Mbps ~ 10Gbps)                         │
│                                                                  │
│   ┌─────────────────────────────────────────────────┐           │
│   │                  Office LAN                       │           │
│   │                                                   │           │
│   │   ┌────┐   ┌────┐   ┌────┐   ┌────┐            │           │
│   │   │ PC │   │ PC │   │ PC │   │ PC │            │           │
│   │   └──┬─┘   └──┬─┘   └──┬─┘   └──┬─┘            │           │
│   │      │        │        │        │               │           │
│   │      └────────┴────┬───┴────────┘               │           │
│   │                    │                            │           │
│   │               ┌────┴────┐                       │           │
│   │               │ Switch  │                       │           │
│   │               └────┬────┘                       │           │
│   │                    │                            │           │
│   │   ┌────────────────┼────────────────┐          │           │
│   │   │                │                │          │           │
│   │ ┌─┴──┐         ┌───┴───┐       ┌────┴────┐    │           │
│   │ │Servr│        │Printer│       │Router   │    │           │
│   │ └────┘         └───────┘       └─────────┘    │           │
│   │                                     │          │           │
│   └─────────────────────────────────────┼──────────┘           │
│                                         │                       │
│                                  Internet Connection            │
│                                                                  │
│   Technology: Ethernet (IEEE 802.3), Wi-Fi (IEEE 802.11)       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### MAN (Metropolitan Area Network)

```
┌─────────────────────────────────────────────────────────────────┐
│              MAN (Metropolitan Area Network)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Range: City or metropolitan area (kilometers to tens of km)  │
│   Ownership: ISP or large organizations                         │
│                                                                  │
│           ┌──────────┐                                          │
│           │ Main HQ  │                                          │
│           │   LAN    │                                          │
│           └─────┬────┘                                          │
│                 │                                               │
│         Fiber Optic Backbone                                    │
│     ┌───────────┼───────────┐                                   │
│     │           │           │                                   │
│  ┌──┴───┐   ┌───┴───┐   ┌───┴───┐                              │
│  │Branch│   │Branch │   │  Data │                              │
│  │  A   │   │  B    │   │Center │                              │
│  │ LAN  │   │ LAN   │   │       │                              │
│  └──────┘   └───────┘   └───────┘                              │
│                                                                  │
│   Examples:                                                     │
│   - University campus network                                   │
│   - City cable TV network                                       │
│   - Corporate multi-site connectivity                           │
│                                                                  │
│   Technology: FDDI, Metro Ethernet, WiMAX                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### WAN (Wide Area Network)

```
┌─────────────────────────────────────────────────────────────────┐
│              WAN (Wide Area Network)                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Range: Country, continent, worldwide                          │
│   Ownership: Telecommunications carriers                        │
│   Example: The Internet                                         │
│                                                                  │
│                        ┌─────────────┐                          │
│                        │  Seoul LAN  │                          │
│                        └──────┬──────┘                          │
│                               │                                  │
│     ┌───────────┐       ┌─────┴─────┐       ┌───────────┐      │
│     │ Busan LAN │───────│ ISP Backbn│───────│Daejeon LAN│      │
│     └───────────┘       └─────┬─────┘       └───────────┘      │
│                               │                                  │
│                    ┌──────────┼──────────┐                      │
│                    │          │          │                      │
│              ┌─────┴─────┐ ┌──┴──┐ ┌─────┴─────┐               │
│              │ Tokyo ISP │ │Under│ │ NYC ISP   │               │
│              └───────────┘ │ Sea │ └───────────┘               │
│                            │Cable│                              │
│                            └─────┘                              │
│                                                                  │
│   Characteristics:                                              │
│   - Uses various transmission media (fiber, satellite, subsea) │
│   - Relatively lower bandwidth, higher latency                 │
│   - Connected through ISPs                                      │
│                                                                  │
│   Technology: MPLS, VPN, Leased Lines, SD-WAN                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Network Type Comparison

| Type | Range | Speed | Examples |
|------|-------|-------|----------|
| PAN | ~10m | Varies | Bluetooth earbuds, USB devices |
| LAN | ~1km | High (1-10 Gbps) | Office, home network |
| MAN | ~50km | Medium | City cable network |
| WAN | Unlimited | Low-High | Internet, enterprise WAN |

---

## 4. Network Topologies

### What is Topology?

```
Topology: Physical/logical connection structure of a network

Physical topology: Actual placement of cables/devices
Logical topology: Path that data flows
```

### Bus Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                       Bus Topology                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ═══════════════════════════════════════════════════════       │
│       │           │           │           │           │         │
│   ┌───┴───┐   ┌───┴───┐   ┌───┴───┐   ┌───┴───┐   ┌───┴───┐   │
│   │  PC1  │   │  PC2  │   │  PC3  │   │  PC4  │   │  PC5  │   │
│   └───────┘   └───────┘   └───────┘   └───────┘   └───────┘   │
│                                                                  │
│   Terminator                                      Terminator    │
│       ◄═══════════════ Backbone Cable ═══════════════►         │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│   Advantages                    │   Disadvantages                │
│   - Simple and cheap           │   - Cable failure affects all  │
│   - Less cable required        │   - Collision possible         │
│   - Easy to add nodes          │   - Difficult troubleshooting  │
│                                 │   - Performance degrades       │
└─────────────────────────────────────────────────────────────────┘
```

### Star Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                       Star Topology                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                        ┌───────┐                                │
│                        │  PC1  │                                │
│                        └───┬───┘                                │
│                            │                                    │
│   ┌───────┐            ┌───┴───┐            ┌───────┐          │
│   │  PC5  │────────────│ Switch│────────────│  PC2  │          │
│   └───────┘            │ / Hub │            └───────┘          │
│                        └───┬───┘                                │
│                       ╱    │    ╲                               │
│                     ╱      │      ╲                             │
│                   ╱        │        ╲                           │
│            ┌───────┐   ┌───┴───┐   ┌───────┐                   │
│            │  PC4  │   │  PC3  │   │Server │                   │
│            └───────┘   └───────┘   └───────┘                   │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│   Advantages                    │   Disadvantages                │
│   - Node failure doesn't affect │   - Central device failure     │
│     entire network              │     affects entire network     │
│   - Easy to identify & fix      │   - More cable cost            │
│   - Easy to add/remove nodes    │   - Central device bottleneck  │
│   - Most widely used today      │                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Ring Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                       Ring Topology                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                       ┌───────┐                                 │
│                       │  PC1  │                                 │
│                       └───┬───┘                                 │
│                      ↙    │    ↘                                │
│                    ╱      │      ╲                              │
│              ┌───────┐    │    ┌───────┐                       │
│              │  PC5  │    │    │  PC2  │                       │
│              └───┬───┘    │    └───┬───┘                       │
│                  │        │        │                            │
│                  ↓   Data Flow     ↓                            │
│                  │  (Unidirectional)│                           │
│              ┌───┴───┐         ┌───┴───┐                       │
│              │  PC4  │─────────│  PC3  │                       │
│              └───────┘    ←    └───────┘                       │
│                                                                  │
│   Token Ring:                                                   │
│   - Only token holder can transmit                              │
│   - No collisions                                               │
│   - FDDI (fiber optic ring)                                     │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│   Advantages                    │   Disadvantages                │
│   - No collisions (with token) │   - Single node failure        │
│   - Equal access for all nodes │     affects entire network     │
│   - Performance maintained     │   - Disruption when adding/    │
│     under high load            │     removing nodes             │
│                                 │   - Hard to locate problems   │
└─────────────────────────────────────────────────────────────────┘
```

### Mesh Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                       Mesh Topology                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Full Mesh                        Partial Mesh                 │
│                                                                  │
│       ┌───────┐                        ┌───────┐                │
│       │  PC1  │                        │  PC1  │                │
│       └─┬─┬─┬─┘                        └─┬───┬─┘                │
│        ╱  │  ╲                          ╱     ╲                 │
│       ╱   │   ╲                        ╱       ╲                │
│  ┌───┴─┐  │  ┌─┴───┐              ┌───┴─┐   ┌─┴───┐            │
│  │ PC4 │──┼──│ PC2 │              │ PC4 │───│ PC2 │            │
│  └──┬──┘  │  └──┬──┘              └──┬──┘   └──┬──┘            │
│     │╲    │    ╱│                    │         │                │
│     │ ╲   │   ╱ │                    │         │                │
│     │  ╲  │  ╱  │                    └────┬────┘                │
│     │   ┌─┴─┐   │                         │                     │
│     └───│PC3│───┘                    ┌────┴────┐                │
│         └───┘                        │   PC3   │                │
│                                      └─────────┘                │
│   All nodes connected               Only some nodes connected   │
│   n(n-1)/2 links                                                │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│   Advantages                    │   Disadvantages                │
│   - High reliability, redundancy│   - Complex installation, high │
│   - Multiple paths for failover│     cost                       │
│   - Fast data transmission     │   - Requires many cables/ports │
│   - Used in Internet backbone  │   - Complex management         │
└─────────────────────────────────────────────────────────────────┘
```

### Hybrid Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                     Hybrid Topology                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Real networks combine multiple topologies.                    │
│                                                                  │
│                     ┌─────────────┐                             │
│                     │ Core Switch │                             │
│                     └──────┬──────┘                             │
│                  ┌─────────┼─────────┐                          │
│                  │         │         │                          │
│            ┌─────┴─────┐ ┌─┴─┐ ┌─────┴─────┐                   │
│            │ Switch A  │ │...│ │ Switch B  │ ← Star Topology   │
│            └─────┬─────┘ └───┘ └─────┬─────┘                   │
│           ╱╲     │     ╲       ╱     │     ╲                   │
│          ╱  ╲    │      ╲     ╱      │      ╲                  │
│       ┌──┐┌──┐┌──┐   ┌──┐ ┌──┐┌──┐┌──┐   ┌──┐                 │
│       │PC││PC││PC│   │PC│ │PC││PC││PC│   │PC│                 │
│       └──┘└──┘└──┘   └──┘ └──┘└──┘└──┘   └──┘                 │
│        Dept A             │        Dept B                       │
│                           │                                     │
│   Star-Bus Hybrid: Each department uses star, inter-dept bus   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Topology Comparison Table

| Topology | Reliability | Cost | Scalability | Use Case |
|----------|-------------|------|-------------|----------|
| Bus | Low | Low | Low | Small-scale, legacy |
| Star | Medium | Medium | High | Office, home |
| Ring | Medium | Medium | Medium | FDDI, some SANs |
| Mesh | High | High | High | Backbone, WAN |
| Hybrid | High | Medium | High | Large networks |

---

## 5. Packet Switching vs Circuit Switching

### Circuit Switching

```
┌─────────────────────────────────────────────────────────────────┐
│                       Circuit Switching                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Establishes and maintains dedicated path before communication │
│   Example: Traditional telephone network (PSTN)                 │
│                                                                  │
│   1. Connection Setup                                           │
│   ┌────┐     ┌────┐     ┌────┐     ┌────┐     ┌────┐          │
│   │ A  │════►│ SW │════►│ SW │════►│ SW │════►│ B  │          │
│   └────┘     └────┘     └────┘     └────┘     └────┘          │
│              ══════════════════════════════                     │
│                    Dedicated circuit established                │
│                                                                  │
│   2. Data Transmission                                          │
│   ┌────┐     ┌────┐     ┌────┐     ┌────┐     ┌────┐          │
│   │ A  │────►│ SW │────►│ SW │────►│ SW │────►│ B  │          │
│   └────┘     └────┘     └────┘     └────┘     └────┘          │
│              Continuous data stream                             │
│                                                                  │
│   3. Connection Teardown                                        │
│   ┌────┐     ┌────┐     ┌────┐     ┌────┐     ┌────┐          │
│   │ A  │     │ SW │     │ SW │     │ SW │     │ B  │          │
│   └────┘     └────┘     └────┘     └────┘     └────┘          │
│              Circuit released, resources freed                  │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│   Advantages                    │   Disadvantages                │
│   - Guaranteed constant        │   - Inefficient resource usage │
│     bandwidth                  │   - Circuit setup time needed  │
│   - Consistent latency         │   - Circuit occupied even when │
│   - Suitable for real-time     │     not transmitting           │
│     communication              │   - Limited scalability        │
└─────────────────────────────────────────────────────────────────┘
```

### Packet Switching

```
┌─────────────────────────────────────────────────────────────────┐
│                       Packet Switching                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Data divided into small packets and transmitted independently │
│   Example: The Internet                                         │
│                                                                  │
│   Original data:                                                │
│   ┌─────────────────────────────────────────────────────┐       │
│   │          "Hello, World! This is a message."          │       │
│   └─────────────────────────────────────────────────────┘       │
│                              ↓                                   │
│   Split into packets:                                           │
│   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                  │
│   │Pkt 1   │ │Pkt 2   │ │Pkt 3   │ │Pkt 4   │                  │
│   │"Hello, │ │"World! │ │"This is│ │"message│                  │
│   │"       │ │"       │ │" a "   │ │"."     │                  │
│   └────────┘ └────────┘ └────────┘ └────────┘                  │
│                                                                  │
│   Transmitted via different paths:                              │
│                                                                  │
│   ┌────┐     ┌────┐─────────────────────┌────┐                 │
│   │ A  │─────│ R1 │────────────────────►│ B  │                 │
│   └────┘     └─┬──┘     ┌────┐          └────┘                 │
│               │        │ R2 │              ↑                    │
│               └───────►└────┘──────────────┘                    │
│                                                                  │
│   Pkt 1, 3: A → R1 → B                                          │
│   Pkt 2, 4: A → R1 → R2 → B                                     │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│   Advantages                    │   Disadvantages                │
│   - Efficient resource usage   │   - Variable latency           │
│   - Multiple communications    │   - Packet loss possible       │
│     simultaneously             │   - Packet order not guaranteed│
│   - Excellent scalability      │   - Header overhead            │
│   - Alternative routes on      │                                 │
│     failure                    │                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Packet Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                        Packet Structure                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────────────────────────────────────────┐      │
│   │ Header             │ Payload           │ Trailer     │      │
│   │                    │  (Actual Data)    │ (Optional)  │      │
│   └──────────────────────────────────────────────────────┘      │
│                                                                  │
│   Information in header:                                        │
│   ┌─────────────────────────────────────────────────────┐       │
│   │ Source    │ Destination │ Packet  │ Protocol        │       │
│   │ Address   │ Address     │ Number  │ Information     │       │
│   └─────────────────────────────────────────────────────┘       │
│                                                                  │
│   Example: IP Packet                                            │
│   ┌───────┬───────┬───────┬───────┬─────────────────────┐       │
│   │Version│ IHL   │  TOS  │ Total │  Identification      │       │
│   │(4bit) │(4bit) │(8bit) │Length │       (16bit)        │       │
│   ├───────┴───────┴───────┴───────┼─────────────────────┤       │
│   │        Flags  │ Fragment Offset│      TTL │Protocol │       │
│   ├────────────────────────────────┼──────────┴─────────┤       │
│   │           Header Checksum      │                     │       │
│   ├────────────────────────────────┴─────────────────────┤       │
│   │              Source IP Address (32bit)               │       │
│   ├──────────────────────────────────────────────────────┤       │
│   │           Destination IP Address (32bit)             │       │
│   ├──────────────────────────────────────────────────────┤       │
│   │                     Data...                           │       │
│   └──────────────────────────────────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Comparison Summary

| Characteristic | Circuit Switching | Packet Switching |
|----------------|-------------------|------------------|
| Connection Setup | Required | Not required |
| Bandwidth | Fixed allocation | Dynamic allocation |
| Resource Efficiency | Low | High |
| Latency | Constant | Variable |
| Reliability | High | Protocol dependent |
| Usage Examples | Telephone, ISDN | Internet, VoIP |

---

## 6. Client-Server vs P2P

### Client-Server Model

```
┌─────────────────────────────────────────────────────────────────┐
│                    Client-Server Model                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                      ┌─────────────┐                            │
│                      │   Server    │                            │
│                      │             │                            │
│                      │ - Provides  │                            │
│                      │   resources │                            │
│                      │ - Handles   │                            │
│                      │   requests  │                            │
│                      │ - Centralized│                           │
│                      └──────┬──────┘                            │
│                             │                                   │
│           ┌─────────────────┼─────────────────┐                 │
│           │                 │                 │                 │
│           ▼                 ▼                 ▼                 │
│     ┌───────────┐    ┌───────────┐    ┌───────────┐            │
│     │  Client   │    │  Client   │    │  Client   │            │
│     │           │    │           │    │           │            │
│     │ - Requests│    │ - Requests│    │ - Requests│            │
│     │ - User    │    │ - User    │    │ - User    │            │
│     └───────────┘    └───────────┘    └───────────┘            │
│                                                                  │
│   Examples: Web browser ←→ Web server                           │
│            Email client ←→ Mail server                          │
│            Mobile app ←→ API server                             │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│   Advantages                    │   Disadvantages                │
│   - Centralized management     │   - Server failure affects all │
│   - Easy security management   │   - Server bottleneck          │
│   - Data consistency           │   - High scaling cost          │
│   - Easy backup & recovery     │   - Single Point of Failure    │
└─────────────────────────────────────────────────────────────────┘
```

### P2P (Peer-to-Peer) Model

```
┌─────────────────────────────────────────────────────────────────┐
│                      P2P (Peer-to-Peer) Model                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   All nodes have equal status, acting as both client & server   │
│                                                                  │
│        ┌────────┐                ┌────────┐                     │
│        │ Peer A │◄──────────────►│ Peer B │                     │
│        │        │                │        │                     │
│        └───┬────┘                └────┬───┘                     │
│            │ ╲                    ╱ │                           │
│            │  ╲                  ╱  │                           │
│            │   ╲                ╱   │                           │
│            │    ╲              ╱    │                           │
│            │     ╲            ╱     │                           │
│            ▼      ▼          ▼      ▼                           │
│        ┌────────┐                ┌────────┐                     │
│        │ Peer C │◄──────────────►│ Peer D │                     │
│        │        │                │        │                     │
│        └────────┘                └────────┘                     │
│                                                                  │
│   File sharing example (BitTorrent):                            │
│   ┌─────────────────────────────────────────────────┐           │
│   │  File: movie.mp4 (1GB)                          │           │
│   │                                                  │           │
│   │  Peer A: [######....] 60% - pieces 1,2,3,4,5,6  │           │
│   │  Peer B: [....######] 60% - pieces 5,6,7,8,9,10 │           │
│   │  Peer C: [##........] 20% - pieces 1,2          │           │
│   │  Peer D: [..####....] 40% - pieces 3,4,5,6      │           │
│   │                                                  │           │
│   │  Each peer shares pieces they have             │           │
│   └─────────────────────────────────────────────────┘           │
│                                                                  │
│   Examples: BitTorrent, Bitcoin, Skype (early), IPFS           │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│   Advantages                    │   Disadvantages                │
│   - No single point of failure │   - Difficult security mgmt    │
│   - Excellent scalability      │   - Hard to maintain data      │
│   - Cost effective             │     consistency                │
│   - More users = more resources│   - Malicious nodes possible   │
│                                 │   - Performance depends on     │
│                                 │     participants               │
└─────────────────────────────────────────────────────────────────┘
```

### Hybrid Model

```
┌─────────────────────────────────────────────────────────────────┐
│                      Hybrid Model                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Combines central server and P2P                               │
│                                                                  │
│                  ┌─────────────────┐                            │
│                  │  Index Server   │                            │
│                  │ (Manages peer   │                            │
│                  │      list)      │                            │
│                  └────────┬────────┘                            │
│                           │                                     │
│              ┌────────────┼────────────┐                        │
│              │            │            │                        │
│         ┌────┴────┐  ┌────┴────┐  ┌────┴────┐                  │
│         │ Peer A  │  │ Peer B  │  │ Peer C  │                  │
│         └────┬────┘  └────┬────┘  └────┬────┘                  │
│              │            │            │                        │
│              └────────────┼────────────┘                        │
│                    Direct P2P communication                     │
│                                                                  │
│   Examples:                                                     │
│   - Spotify: Authentication via server, streaming via P2P (some)│
│   - Skype (old): Login via server, calls via P2P               │
│   - Online games: Matchmaking via server, gameplay via P2P     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Model Comparison

| Characteristic | Client-Server | P2P | Hybrid |
|----------------|---------------|-----|--------|
| Management | Centralized | Distributed | Mixed |
| Scalability | Limited | Excellent | Excellent |
| Reliability | Server dependent | Distributed | Mixed |
| Security | Easy management | Difficult | Medium |
| Cost | High server cost | Low | Medium |

---

## 7. Network Devices

### Network Devices by Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                      Network Device Layers                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   OSI Layer             Device                                  │
│   ─────────────────────────────────────                         │
│   7. Application   ─→  Firewall (L7), Proxy, Load Balancer     │
│   6. Presentation  ─→                                          │
│   5. Session       ─→                                          │
│   4. Transport     ─→  Firewall (L4)                           │
│   3. Network       ─→  Router, L3 Switch                       │
│   2. Data Link     ─→  Switch (L2), Bridge                     │
│   1. Physical      ─→  Hub, Repeater, Cable, NIC              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Major Network Devices

```
┌─────────────────────────────────────────────────────────────────┐
│  Device        │  Layer │  Function                             │
├────────────────┼────────┼────────────────────────────────────────┤
│  Repeater      │  L1    │  Signal amplification, extend distance│
│                │        │                                       │
├────────────────┼────────┼────────────────────────────────────────┤
│  Hub           │  L1    │  Multi-port repeater, broadcasts to   │
│                │        │  all ports                            │
├────────────────┼────────┼────────────────────────────────────────┤
│  Bridge        │  L2    │  Connects two networks, learns MAC    │
│                │        │  addresses                            │
├────────────────┼────────┼────────────────────────────────────────┤
│  Switch        │  L2    │  Multi-port bridge, MAC-based         │
│                │ (L3)   │  forwarding; L3 switch includes       │
│                │        │  routing                              │
├────────────────┼────────┼────────────────────────────────────────┤
│  Router        │  L3    │  Forwards packets between networks,   │
│                │        │  IP-based routing                     │
├────────────────┼────────┼────────────────────────────────────────┤
│  Gateway       │ L3-L7  │  Protocol conversion between          │
│                │        │  different protocols                  │
├────────────────┼────────┼────────────────────────────────────────┤
│  Firewall      │ L3-L7  │  Traffic filtering, enforces security │
│                │        │  policies                             │
└────────────────┴────────┴────────────────────────────────────────┘
```

### Hub vs Switch

```
┌─────────────────────────────────────────────────────────────────┐
│                     Hub - L1 Device                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   PC1 sending data to PC3:                                      │
│                                                                  │
│   ┌────┐ ┌────┐ ┌────┐ ┌────┐                                  │
│   │PC1 │ │PC2 │ │PC3 │ │PC4 │                                  │
│   └──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘                                  │
│      ▼      ▼      ▼      ▼                                    │
│   ═══╪══════╪══════╪══════╪═══                                  │
│      │      │      │      │                                     │
│      └──────┴──────┴──────┘                                     │
│           ┌────────┐                                            │
│           │  Hub   │                                            │
│           └────────┘                                            │
│                                                                  │
│   Broadcasts to all ports                                       │
│   Collisions possible (collision domain = entire network)       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   Switch - L2 Device                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   PC1 sending data to PC3:                                      │
│                                                                  │
│   ┌────┐ ┌────┐ ┌────┐ ┌────┐                                  │
│   │PC1 │ │PC2 │ │PC3 │ │PC4 │                                  │
│   └──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘                                  │
│      │      │      ▲      │                                     │
│      ▼      │      │      │                                     │
│   ═══╪══════╪══════╪══════╪═══                                  │
│      │             │                                            │
│      └─────────────┘                                            │
│           ┌────────┐                                            │
│           │ Switch │   Uses MAC table                           │
│           └────────┘   Forwards only to destination             │
│                                                                  │
│   MAC address-based forwarding (unicast)                        │
│   Separate collision domain per port                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Router

```
┌─────────────────────────────────────────────────────────────────┐
│                     Router - L3 Device                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Forwards packets between different networks                   │
│                                                                  │
│   ┌───────────────────┐       ┌───────────────────┐            │
│   │  Network A        │       │  Network B        │            │
│   │  192.168.1.0/24   │       │  192.168.2.0/24   │            │
│   │                   │       │                   │            │
│   │  ┌────┐ ┌────┐   │       │  ┌────┐ ┌────┐   │            │
│   │  │PC1 │ │PC2 │   │       │  │PC3 │ │PC4 │   │            │
│   │  └──┬─┘ └──┬─┘   │       │  └──┬─┘ └──┬─┘   │            │
│   │     └──┬───┘     │       │     └──┬───┘     │            │
│   │        │         │       │        │         │            │
│   └────────┼─────────┘       └────────┼─────────┘            │
│            │                          │                        │
│            │     ┌──────────────┐     │                        │
│            └────►│   Router     │◄────┘                        │
│                  │              │                              │
│                  │ Routing Table│                              │
│                  │ Path Decision│                              │
│                  └───────┬──────┘                              │
│                          │                                     │
│                          ▼                                     │
│                     ┌─────────┐                                │
│                     │Internet │                                │
│                     └─────────┘                                │
│                                                                  │
│   Functions:                                                    │
│   - IP address-based packet forwarding                         │
│   - Routing protocols (RIP, OSPF, BGP)                         │
│   - NAT (Network Address Translation)                          │
│   - Firewall capabilities                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Practice Problems

### Basic Problems

**1. Arrange the following network types in order from smallest to largest range:**
   - WAN, LAN, PAN, MAN

**2. What happens when the central device in a star topology fails?**

**3. Which of the following is an L2 device?**
   - (a) Hub
   - (b) Switch
   - (c) Router
   - (d) Repeater

**4. Explain the difference between packet switching and circuit switching.**

### Applied Problems

**5. Choose the appropriate topology for the following scenarios and explain why:**
   - A small office with 10 people
   - A bank's ATM network
   - Intercontinental connection via submarine cable

**6. Describe two situations each where client-server model is appropriate and where P2P model is appropriate.**

**7. Look at the following network diagram and answer the questions:**

```
[PC1] ──┐
        │
[PC2] ──┼──[SwitchA]──[Router]──[SwitchB]──┼──[PC5]
        │                                  │
[PC3] ──┘                                  └──[PC6]
```

   - (a) What devices does PC1 go through to communicate with PC3?
   - (b) What devices does PC1 go through to communicate with PC5?
   - (c) If SwitchA fails, which PCs are affected?

### Advanced Problems

**8. Explain the role of TCP/IP in the evolution from ARPANET to the Internet.**

**9. Explain at least 3 reasons why mesh topology is suitable for the Internet backbone.**

**10. Explain why bus topology is rarely used in modern networks.**

---

<details>
<summary>Answers</summary>

**1.** PAN < LAN < MAN < WAN

**2.** The entire network goes down (single point of failure)

**3.** (b) Switch

**4.**
- Circuit switching: Dedicated path established before communication, fixed bandwidth allocation during connection, low resource efficiency
- Packet switching: Data divided into packets, transmitted via independent paths, high resource efficiency, variable latency

**5.**
- Small office: Star topology (easy management, fault isolation)
- ATM network: Mesh or star (reliability critical)
- Intercontinental connection: Mesh topology (multiple paths, high reliability needed)

**6.**
- Client-Server: Online banking (security/consistency), corporate email (centralized management)
- P2P: File sharing (BitTorrent), cryptocurrency (distributed ledger)

**7.**
- (a) SwitchA only
- (b) SwitchA → Router → SwitchB
- (c) PC1, PC2, PC3

**8.** TCP/IP is a standard protocol for connecting different networks. When ARPANET adopted TCP/IP instead of NCP in 1983, it enabled communication between heterogeneous networks, which became the foundation of today's Internet.

**9.**
- Multiple paths allow rerouting on failure
- Provides high bandwidth and reliability
- Distributes traffic to alleviate bottlenecks
- Excellent scalability
- Enhances network resilience

**10.**
- Cable failure causes entire network failure
- Performance degrades with more nodes (increased collisions)
- Difficult troubleshooting
- Unsuitable for modern high-speed network requirements

</details>

---

**Next**: [OSI 7-Layer Model](./02_OSI_7_Layer_Model.md)

---

## References

- Computer Networking: A Top-Down Approach (Kurose & Ross)
- [Cisco Networking Basics](https://www.cisco.com/c/en/us/solutions/small-business/resource-center/networking/networking-basics.html)
- [Khan Academy: Internet 101](https://www.khanacademy.org/computing/computers-and-internet)
- [NetworkChuck YouTube Channel](https://www.youtube.com/c/NetworkChuck)
