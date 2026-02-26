# Routing Protocols

**Previous**: [Routing Basics](./08_Routing_Basics.md) | **Next**: [TCP Protocol](./10_TCP_Protocol.md)

**Difficulty**: ⭐⭐⭐
**Estimated Learning Time**: 3-4 hours
**Prerequisites**: [08_Routing_Basics.md](./08_Routing_Basics.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Classify routing protocols as IGP vs. EGP and distance vector vs. link state
2. Explain how RIP works, including its hop count metric and limitations
3. Describe OSPF operation, including areas, LSAs, SPF calculation, and DR/BDR election
4. Explain the role of BGP as the Internet's inter-domain routing protocol and how it uses path attributes
5. Define an Autonomous System (AS) and explain its significance in Internet routing
6. Compare RIP, OSPF, and BGP and recommend the appropriate protocol for a given network scenario

---

Dynamic routing protocols are what make the Internet self-healing and scalable. When a link goes down or a new network comes online, routing protocols automatically detect the change and recalculate optimal paths -- no manual intervention needed. From RIP in small labs to BGP governing the global Internet routing table of over 900,000 prefixes, these protocols are the invisible intelligence behind every packet's journey.

## Table of Contents

1. [Classification of Routing Protocols](#1-classification-of-routing-protocols)
2. [Distance Vector vs Link State](#2-distance-vector-vs-link-state)
3. [RIP (Routing Information Protocol)](#3-rip-routing-information-protocol)
4. [OSPF (Open Shortest Path First)](#4-ospf-open-shortest-path-first)
5. [BGP (Border Gateway Protocol)](#5-bgp-border-gateway-protocol)
6. [AS (Autonomous System)](#6-as-autonomous-system)
7. [Practice Problems](#7-practice-problems)
8. [References](#8-references)

---

## 1. Classification of Routing Protocols

### 1.1 IGP vs EGP

Routing protocols are classified into IGP and EGP based on their scope of use.

```
                    Routing Protocol Classification

┌─────────────────────────────────────────────────────────────────┐
│                          Internet                                │
│                                                                  │
│  ┌──────────────────┐          ┌──────────────────┐            │
│  │      AS 100      │   EGP    │      AS 200      │            │
│  │                  │◄────────►│                  │            │
│  │    (e.g., KT)    │   BGP    │   (e.g., SKT)    │            │
│  │                  │          │                  │            │
│  │  ┌────┐ ┌────┐  │          │  ┌────┐ ┌────┐  │            │
│  │  │ R1 │─│ R2 │  │          │  │ R3 │─│ R4 │  │            │
│  │  └────┘ └────┘  │          │  └────┘ └────┘  │            │
│  │       IGP       │          │       IGP       │            │
│  │   (OSPF/RIP)    │          │   (OSPF/RIP)    │            │
│  └──────────────────┘          └──────────────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

| Aspect | IGP (Interior Gateway Protocol) | EGP (Exterior Gateway Protocol) |
|------|--------------------------------|--------------------------------|
| Scope | Within AS | Between ASes |
| Purpose | Optimal path in internal network | External network connection policy |
| Protocols | RIP, OSPF, EIGRP, IS-IS | BGP |
| Metrics | Hop count, bandwidth, delay, etc. | Path Attributes |

### 1.2 Classification by Algorithm

```
┌─────────────────────────────────────────────────────────────────┐
│                   Routing Algorithm Classification               │
├───────────────────────────┬─────────────────────────────────────┤
│      Distance Vector      │           Link State                │
│                           │                                     │
├───────────────────────────┼─────────────────────────────────────┤
│ • RIP                     │ • OSPF                              │
│ • EIGRP (Hybrid)          │ • IS-IS                             │
│                           │                                     │
│ Features:                 │ Features:                           │
│ - Exchange info with      │ - Understand entire topology        │
│   neighbors               │ - Dijkstra algorithm                │
│ - Bellman-Ford algorithm  │ - Complex, more resources           │
│ - Simple, fewer resources │ - Fast convergence                  │
│ - Slow convergence        │                                     │
└───────────────────────────┴─────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Path Vector                                 │
├─────────────────────────────────────────────────────────────────┤
│ • BGP                                                           │
│                                                                  │
│ Features:                                                       │
│ - Transmits AS path information                                 │
│ - Policy-based routing                                          │
│ - Used in Internet backbone                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Classful vs Classless

| Aspect | Classful | Classless |
|------|---------|-----------|
| Subnet Mask | Not transmitted | Transmitted together |
| VLSM Support | X | O |
| CIDR Support | X | O |
| Protocols | RIPv1, IGRP | RIPv2, OSPF, EIGRP, BGP |

---

## 2. Distance Vector vs Link State

### 2.1 Distance Vector

"Trust the information my neighbors tell me"

```
                    Distance Vector Operation

Initial State:
    R1 ─── R2 ─── R3 ─── R4
   [A]    [ ]    [ ]    [D]

Step 1: Advertise routing table to neighbors
    R1: "Distance to A is 0"  ──►  R2
    R4: "Distance to D is 0"  ──►  R3

Step 2: Received info + 1 (hop count)
    R2: "Distance to A is 1" (via R1)
    R3: "Distance to D is 1" (via R4)

Step 3: Advertise again to neighbors
    R2: "Distance to A is 1"  ──►  R3
    R3: "Distance to D is 1"  ──►  R2

Step 4: Final Convergence
    R1: A(0), D(3)
    R2: A(1), D(2)
    R3: A(2), D(1)
    R4: A(3), D(0)
```

**Advantages**:
- Simple implementation
- Low CPU/memory usage
- Suitable for small networks

**Disadvantages**:
- Slow convergence
- Potential for routing loops
- Hop count limitation (RIP: 15)

### 2.2 Link State

"I understand the entire network myself"

```
                    Link State Operation

Network Topology:
         10          5           15
    R1 ────── R2 ────── R3 ────── R4
     │                            │
     └────────────────────────────┘
                  20

Step 1: Each router generates LSA (Link State Advertisement)
    R1's LSA: "R1 connected: R2(cost 10), R4(cost 20)"
    R2's LSA: "R2 connected: R1(cost 10), R3(cost 5)"
    ...

Step 2: Flood LSAs to all routers
    All routers maintain identical LSDB (Link State Database)

Step 3: Calculate shortest path with Dijkstra algorithm
    R1 → R4 paths:
    - R1 → R4 direct: cost 20
    - R1 → R2 → R3 → R4: cost 10+5+15 = 30
    Selected: Direct path (cost 20)
```

**SPF (Shortest Path First) Tree Example**:

```
R1's SPF Tree:

                R1 (root)
               /   \
         (10) /     \ (20)
             /       \
           R2        R4
           |
      (5)  |
           |
          R3
           |
     (15)  |
           |
          R4 (duplicate - higher cost, ignored)
```

**Advantages**:
- Fast convergence
- Accurate topology information
- No routing loops
- VLSM/CIDR support

**Disadvantages**:
- High CPU/memory requirements
- Complex configuration
- Initial flooding consumes bandwidth

### 2.3 Comparison Summary

| Feature | Distance Vector | Link State |
|------|----------|----------|
| Information Shared | Routing table | Link state (topology) |
| Algorithm | Bellman-Ford | Dijkstra |
| Updates | Periodic (complete) | On change (changes only) |
| Convergence Speed | Slow | Fast |
| Resources | Low | High |
| Scalability | Low | High |
| Representative Protocols | RIP, EIGRP | OSPF, IS-IS |

---

## 3. RIP (Routing Information Protocol)

### 3.1 RIP Overview

RIP is the oldest distance vector routing protocol, simple but limited.

```
┌─────────────────────────────────────────────────────────────────┐
│                       RIP Features                               │
├─────────────────────────────────────────────────────────────────┤
│  • Metric: Hop Count                                            │
│  • Maximum Hops: 15 (16 = unreachable)                          │
│  • Update Interval: 30 seconds                                  │
│  • Administrative Distance (AD): 120                            │
│  • Port: UDP 520                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 RIP Version Comparison

| Feature | RIPv1 | RIPv2 |
|------|-------|-------|
| Class | Classful | Classless |
| Subnet Mask | Not transmitted | Transmitted |
| VLSM | Not supported | Supported |
| CIDR | Not supported | Supported |
| Authentication | None | Supported (MD5) |
| Transmission | Broadcast | Multicast (224.0.0.9) |

### 3.3 RIP Operation Process

```
RIP Routing Update Example

Network Configuration:
    Network A          Network B          Network C
   10.0.0.0/24        10.1.0.0/24        10.2.0.0/24
       │                  │                  │
    ┌──┴──┐            ┌──┴──┐            ┌──┴──┐
    │ R1  │────────────│ R2  │────────────│ R3  │
    └─────┘            └─────┘            └─────┘

Initial Routing Table:

R1:
┌────────────────┬──────┬───────────┐
│ Network        │ Hops │ Next Hop  │
├────────────────┼──────┼───────────┤
│ 10.0.0.0/24    │ 0    │ Direct    │
└────────────────┴──────┴───────────┘

After 30 seconds (R2 receives updates from R1 and R3):

R2:
┌────────────────┬──────┬───────────┐
│ Network        │ Hops │ Next Hop  │
├────────────────┼──────┼───────────┤
│ 10.0.0.0/24    │ 1    │ R1        │
│ 10.1.0.0/24    │ 0    │ Direct    │
│ 10.2.0.0/24    │ 1    │ R3        │
└────────────────┴──────┴───────────┘

After 60 seconds (convergence complete):

R1:
┌────────────────┬──────┬───────────┐
│ Network        │ Hops │ Next Hop  │
├────────────────┼──────┼───────────┤
│ 10.0.0.0/24    │ 0    │ Direct    │
│ 10.1.0.0/24    │ 1    │ R2        │
│ 10.2.0.0/24    │ 2    │ R2        │
└────────────────┴──────┴───────────┘
```

### 3.4 RIP Timers

| Timer | Value | Description |
|--------|------|------|
| Update | 30s | Routing update transmission interval |
| Invalid | 180s | Time until route marked invalid |
| Holddown | 180s | Time to prevent route changes |
| Flush | 240s | Time to delete from routing table |

### 3.5 RIP Loop Prevention Mechanisms

```
1. Split Horizon
   - Don't advertise route back through interface it was learned

       R1 ──────── R2
       │
   "Learned 10.0.0.0 from R1,
    won't advertise 10.0.0.0 back to R1"

2. Route Poisoning
   - Advertise down routes with metric 16

   R1: "10.0.0.0 is down"
   R1 → R2: "10.0.0.0 metric=16 (unreachable)"

3. Holddown Timer
   - After route goes down, refuse new routes for period
   - Prevents propagation of incorrect information

4. Triggered Update
   - Send update immediately when change occurs
   - Fast convergence without waiting 30 seconds
```

### 3.6 RIP Configuration Examples

**Cisco Router**:
```
Router(config)# router rip
Router(config-router)# version 2
Router(config-router)# network 10.0.0.0
Router(config-router)# network 192.168.1.0
Router(config-router)# no auto-summary
```

**Linux (Quagga/FRR)**:
```
router rip
 version 2
 network 10.0.0.0/8
 network 192.168.1.0/24
 no auto-summary
```

---

## 4. OSPF (Open Shortest Path First)

### 4.1 OSPF Overview

OSPF is the most widely used link state routing protocol.

```
┌─────────────────────────────────────────────────────────────────┐
│                       OSPF Features                              │
├─────────────────────────────────────────────────────────────────┤
│  • Metric: Cost = Reference Bandwidth / Interface Bandwidth     │
│  • Reference Bandwidth: Default 100 Mbps                        │
│  • Administrative Distance (AD): 110                            │
│  • Protocol: IP Protocol 89                                     │
│  • Multicast: 224.0.0.5 (AllSPFRouters)                        │
│               224.0.0.6 (AllDRouters)                           │
│  • Area-based hierarchical structure                            │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 OSPF Cost Calculation

```
OSPF Cost = Reference Bandwidth (100 Mbps) / Interface Bandwidth

┌─────────────────────┬───────────────┬────────────┐
│ Interface Type      │ Bandwidth     │ OSPF Cost  │
├─────────────────────┼───────────────┼────────────┤
│ Serial (T1)         │ 1.544 Mbps    │ 64         │
│ Ethernet            │ 10 Mbps       │ 10         │
│ Fast Ethernet       │ 100 Mbps      │ 1          │
│ Gigabit Ethernet    │ 1000 Mbps     │ 1 (default)│
│ 10 Gigabit Ethernet │ 10000 Mbps    │ 1 (default)│
└─────────────────────┴───────────────┴────────────┘

※ For Gig+, reference bandwidth adjustment needed (e.g., 10000 Mbps)
```

### 4.3 OSPF Areas

```
                    OSPF Area Structure

┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│                         Area 0                                   │
│                      (Backbone)                                  │
│                                                                  │
│            ┌─────────────────────────────┐                      │
│            │                             │                      │
│    ┌───────┴──┐    ┌─────────┐    ┌──────┴───────┐             │
│    │   ABR    │    │   ABR   │    │     ABR      │             │
│    └────┬─────┘    └────┬────┘    └──────┬───────┘             │
│         │               │                 │                      │
│    ┌────┴────┐    ┌────┴────┐    ┌──────┴──────┐              │
│    │ Area 1  │    │ Area 2  │    │   Area 3    │              │
│    │         │    │         │    │             │              │
│    │ R1──R2  │    │ R3──R4  │    │  R5──R6     │              │
│    │         │    │         │    │             │              │
│    └─────────┘    └─────────┘    └─────────────┘              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

ABR: Area Border Router
```

**Purpose of Areas**:
- Reduce LSDB size
- Limit SPF calculation scope
- Reduce routing updates
- Improve network stability

**Area Types**:

| Area Type | Description |
|----------|------|
| Backbone (Area 0) | Central area, connects all areas |
| Standard Area | Regular area |
| Stub Area | Blocks external routes, uses default route |
| Totally Stubby | Blocks external + other area routes |
| NSSA | Allows limited external routes |

### 4.4 OSPF Router Types

```
┌─────────────────────────────────────────────────────────────────┐
│                     OSPF Router Roles                            │
├──────────────────┬──────────────────────────────────────────────┤
│ Internal Router  │ Operates within single area                  │
│                  │                                              │
│ ABR              │ Area Border Router                           │
│                  │ Connected to multiple areas, exchanges       │
│                  │ routing info between areas                   │
│                  │                                              │
│ ASBR             │ AS Boundary Router                           │
│                  │ Connected to external routing domain         │
│                  │                                              │
│ Backbone Router  │ Router in Area 0                            │
│                  │                                              │
│ DR               │ Designated Router                            │
│                  │ Representative router on multi-access network│
│                  │                                              │
│ BDR              │ Backup Designated Router                     │
│                  │ DR backup                                    │
└──────────────────┴──────────────────────────────────────────────┘
```

### 4.5 OSPF Packet Types

| Type | Name | Description |
|------|------|------|
| 1 | Hello | Neighbor discovery and relationship maintenance |
| 2 | DBD (Database Description) | Exchange LSDB summary information |
| 3 | LSR (Link State Request) | Request specific LSA |
| 4 | LSU (Link State Update) | Transmit LSA |
| 5 | LSAck | LSA receipt acknowledgment |

### 4.6 OSPF Neighbor States

```
OSPF Neighbor Relationship Establishment

Down → Init → 2-Way → ExStart → Exchange → Loading → Full

┌─────────────────────────────────────────────────────────────────┐
│  Down      : Initial state, no Hello received                   │
│  Init      : Hello received, bidirectional not confirmed        │
│  2-Way     : Bidirectional communication confirmed (DR/BDR elect)│
│  ExStart   : Master/Slave decision, sequence number exchange    │
│  Exchange  : Exchange LSDB summary via DBD packets              │
│  Loading   : Request and receive missing LSAs via LSR/LSU       │
│  Full      : LSDB synchronized, neighbor relationship established│
└─────────────────────────────────────────────────────────────────┘
```

### 4.7 OSPF Configuration Examples

**Cisco Router**:
```
! Process ID 1 is locally significant — routers with different IDs can still form adjacencies
Router(config)# router ospf 1
! area 0 = backbone; ALL inter-area traffic must transit area 0 (hierarchical design reduces LSDB flooding)
! Wildcard mask 0.255.255.255 = inverse of subnet mask; 0 bits must match, 255 bits are "don't care"
Router(config-router)# network 10.0.0.0 0.255.255.255 area 0
! Separate area 1 limits SPF recalculation scope — topology changes in area 1 don't trigger re-computation in area 0
Router(config-router)# network 192.168.1.0 0.0.0.255 area 1
```

**Linux (FRR)**:
```
router ospf
 # FRR uses CIDR notation instead of wildcard masks — same semantics, more intuitive syntax
 network 10.0.0.0/8 area 0.0.0.0
 # Dotted-decimal area IDs (0.0.0.1 = area 1) — required format in FRR/Quagga
 network 192.168.1.0/24 area 0.0.0.1
```

---

## 5. BGP (Border Gateway Protocol)

### 5.1 BGP Overview

BGP is the path vector protocol used for inter-AS routing on the Internet backbone.

```
┌─────────────────────────────────────────────────────────────────┐
│                       BGP Features                               │
├─────────────────────────────────────────────────────────────────┤
│  • Type: Path Vector protocol                                   │
│  • Purpose: Inter-AS routing (EGP)                              │
│  • Port: TCP 179                                                │
│  • Administrative Distance: eBGP=20, iBGP=200                   │
│  • Path Selection: Policy-based (Path Attributes)               │
│  • Current Version: BGP-4                                       │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 eBGP vs iBGP

```
                    eBGP and iBGP

┌──────────────────────┐        ┌──────────────────────┐
│       AS 100         │        │       AS 200         │
│                      │        │                      │
│   R1 ◄──── iBGP ────► R2 ◄── eBGP ──► R3 ◄── iBGP ──► R4
│                      │        │                      │
│  (Within same AS)    │        │  (Within same AS)    │
└──────────────────────┘        └──────────────────────┘
        │                                    │
        └──────────── eBGP ──────────────────┘
                 (Between different ASes)

eBGP (External BGP):
- Connection between different ASes
- AD: 20 (high trust)
- Typically directly connected

iBGP (Internal BGP):
- Connection between BGP routers in same AS
- AD: 200 (low trust)
- Requires full mesh or Route Reflector
```

### 5.3 BGP Path Attributes

| Attribute | Type | Description |
|------|------|------|
| AS_PATH | Well-known Mandatory | List of ASes path traversed |
| NEXT_HOP | Well-known Mandatory | Next hop IP address |
| ORIGIN | Well-known Mandatory | Route origin (IGP/EGP/Incomplete) |
| LOCAL_PREF | Well-known Discretionary | Local preference (iBGP use) |
| MED | Optional Non-transitive | Multi-Exit Discriminator |
| COMMUNITY | Optional Transitive | Route grouping tag |

### 5.4 BGP Path Selection Process

```
BGP Best Path Selection Algorithm

1. Weight (higher preferred) - Cisco proprietary
2. LOCAL_PREF (higher preferred)
3. Locally Originated (prefer self-generated routes)
4. AS_PATH (shorter preferred)
5. ORIGIN (IGP > EGP > Incomplete)
6. MED (lower preferred)
7. eBGP over iBGP
8. IGP metric (cost to next hop)
9. Oldest route
10. Router ID (lower preferred)
11. Neighbor IP (lower preferred)

Example:
┌────────────────────────────────────────────────────┐
│ Route A: AS_PATH = 100 200 300, LOCAL_PREF = 100  │
│ Route B: AS_PATH = 400 500, LOCAL_PREF = 200      │
│                                                     │
│ Selected: Route B (higher LOCAL_PREF)              │
└────────────────────────────────────────────────────┘
```

### 5.5 BGP Message Types

| Message | Description |
|--------|------|
| OPEN | BGP session establishment, parameter exchange |
| UPDATE | Advertise/withdraw route information |
| KEEPALIVE | Connection maintenance check (60 second interval) |
| NOTIFICATION | Error notification, session termination |

### 5.6 BGP States

```
BGP State Transition

Idle → Connect → OpenSent → OpenConfirm → Established

┌─────────────────────────────────────────────────────────────────┐
│  Idle        : Initial state, start TCP connection              │
│  Connect     : Waiting for TCP connection                       │
│  Active      : TCP connection retry (if Connect fails)          │
│  OpenSent    : OPEN message sent, waiting for response          │
│  OpenConfirm : OPEN message received, waiting for KEEPALIVE     │
│  Established : BGP session established, route exchange starts   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.7 BGP Configuration Examples

**Cisco Router (eBGP)**:
```
! BGP uses TCP port 179 — reliable transport prevents route flapping from lost packets
Router(config)# router bgp 100
! remote-as 200 ≠ local AS 100, so this is an eBGP session (different trust level: AD=20 vs iBGP's AD=200)
Router(config-router)# neighbor 203.0.113.1 remote-as 200
! Explicit mask required — BGP does NOT auto-summarize; you advertise exactly the prefix you specify
Router(config-router)# network 10.0.0.0 mask 255.0.0.0
```

**Linux (FRR)**:
```
router bgp 100
 neighbor 203.0.113.1 remote-as 200
 # address-family separates IPv4/IPv6/VPN route tables — one BGP process handles multiple protocols
 address-family ipv4 unicast
  network 10.0.0.0/8
 exit-address-family
```

---

## 6. AS (Autonomous System)

### 6.1 What is AS?

An AS (Autonomous System) is a collection of networks under a single administrative domain with common routing policy.

```
┌─────────────────────────────────────────────────────────────────┐
│                    AS (Autonomous System)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Definition:                                                    │
│  - Collection of IP networks and routers managed by single org  │
│  - Follows common routing policy                                │
│  - Identified by unique AS number (ASN)                         │
│                                                                  │
│  Examples:                                                      │
│  - ISPs (KT, SKT, LGU+)                                        │
│  - Large enterprises                                            │
│  - Cloud providers (AWS, GCP, Azure)                            │
│  - Content providers (Netflix, Google)                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 AS Numbers (ASN)

| Range | Type | Description |
|------|------|------|
| 1 - 64,495 | Public 2-byte | Usable on Internet |
| 64,496 - 64,511 | Documentation | For RFC document examples |
| 64,512 - 65,534 | Private 2-byte | Internal use |
| 65,535 | Reserved | Not usable |
| 1 - 4,199,999,999 | Public 4-byte | Extended AS numbers |
| 4,200,000,000 - 4,294,967,294 | Private 4-byte | Internal use |

**Famous AS Number Examples**:

| ASN | Organization |
|-----|------|
| AS7018 | AT&T |
| AS15169 | Google |
| AS16509 | Amazon |
| AS32934 | Facebook |
| AS4766 | KT |
| AS9318 | SKT |

### 6.3 Inter-AS Relationships

```
                    AS Peering Relationships

┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  1. Transit:                                                    │
│     - Large ISP provides Internet connectivity to smaller ISP   │
│     - Paid relationship                                         │
│                                                                  │
│     ┌─────────┐                                                 │
│     │ Tier 1  │  ← Payment                                      │
│     │   ISP   │                                                 │
│     └────┬────┘                                                 │
│          │ Transit                                               │
│     ┌────┴────┐                                                 │
│     │ Tier 2  │                                                 │
│     │   ISP   │                                                 │
│     └─────────┘                                                 │
│                                                                  │
│  2. Peering:                                                    │
│     - Free traffic exchange between peer ISPs                   │
│     - Connection at IXP (Internet Exchange Point)               │
│                                                                  │
│     ┌─────────┐   Free Exchange   ┌─────────┐                  │
│     │  AS A   │◄─────────────────►│  AS B   │                  │
│     └─────────┘     (Peering)     └─────────┘                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.4 Internet Hierarchy

```
                    Internet AS Hierarchy

                     ┌─────────────┐
                     │   Tier 1    │  ← Can reach entire Internet
                     │ (Global ISP)│    No need to buy transit
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
         ┌────┴────┐  ┌────┴────┐  ┌────┴────┐
         │ Tier 2  │  │ Tier 2  │  │ Tier 2  │  ← Regional ISP
         │         │  │         │  │         │    Buy transit
         └────┬────┘  └────┬────┘  └────┬────┘    from Tier 1
              │             │             │
         ┌────┴────┐  ┌────┴────┐  ┌────┴────┐
         │ Tier 3  │  │ Tier 3  │  │ Tier 3  │  ← Local ISP
         │         │  │         │  │         │    End-user
         └─────────┘  └─────────┘  └─────────┘    services

Tier 1 Examples: AT&T, NTT, Cogent, Lumen
Tier 2 Examples: Large regional ISPs
Tier 3 Examples: Small regional ISPs, cable operators
```

---

## 7. Practice Problems

### Problem 1: Protocol Feature Matching

Match the following features with their corresponding protocol.

```
Features:
a) Maximum hop count 15
b) Uses Dijkstra algorithm
c) Used for inter-AS routing
d) Sends complete routing table every 30 seconds
e) Area-based hierarchical structure
f) Uses TCP port 179

Protocols: RIP, OSPF, BGP
```

### Problem 2: OSPF Cost Calculation

Calculate the total OSPF cost for the following path when reference bandwidth is 100 Mbps.

```
R1 ──(FastEthernet)── R2 ──(Serial T1)── R3 ──(GigabitEthernet)── R4
        100 Mbps           1.544 Mbps          1000 Mbps
```

### Problem 3: BGP Path Selection

Which of the following two BGP routes will be selected?

```
Route A:
- AS_PATH: 100 200 300
- LOCAL_PREF: 150
- MED: 100

Route B:
- AS_PATH: 400 500
- LOCAL_PREF: 150
- MED: 50
```

### Problem 4: Routing Protocol Selection

Select the appropriate routing protocol for each scenario and explain why.

a) Small office with 10 routers
b) Large enterprise network with 500 routers
c) Connection between two ISPs
d) Branch office network with single path

---

## Answers

### Problem 1 Answers

- a) Maximum hop count 15 → **RIP**
- b) Dijkstra algorithm → **OSPF**
- c) Inter-AS routing → **BGP**
- d) 30 second table broadcast → **RIP**
- e) Area-based structure → **OSPF**
- f) TCP 179 → **BGP**

### Problem 2 Answer

```
FastEthernet: 100 / 100 = 1
Serial T1:    100 / 1.544 = 64 (rounded)
GigabitEthernet: 100 / 1000 = 1 (minimum 1)

Total cost = 1 + 64 + 1 = 66
```

### Problem 3 Answer

**Route B selected**

Analysis:
1. LOCAL_PREF: Both 150 (tie)
2. AS_PATH length: Route A = 3, Route B = 2
   → **Route B selected for shorter AS_PATH**

(MED only used for comparing routes from same neighbor AS)

### Problem 4 Answers

a) Small office (10 routers): **RIP** or **Static Routing**
   - Suitable for simple networks
   - Easy configuration and management

b) Large enterprise (500 routers): **OSPF**
   - Fast convergence
   - Scalability via area division
   - VLSM/CIDR support

c) ISP interconnection: **BGP**
   - Standard for inter-AS routing
   - Policy-based path control

d) Single path branch: **Static Routing** (+ default route)
   - Dynamic routing unnecessary
   - Resource saving

---

**Previous**: [Routing Basics](./08_Routing_Basics.md) | **Next**: [TCP Protocol](./10_TCP_Protocol.md)

---

## 8. References

### RFC Documents

- RFC 2453 - RIP Version 2
- RFC 2328 - OSPF Version 2
- RFC 4271 - BGP-4
- RFC 1930 - AS Operation Guidelines

### Useful Tools

```bash
# BGP route lookup — Looking Glass servers let you see the Internet from another AS's perspective
# BGP Looking Glass: https://lg.he.net/

# AS information lookup — RADB (Routing Assets Database) contains IRR records with route objects and policies
whois -h whois.radb.net AS15169

# Path tracing — -A shows AS numbers at each hop, revealing which organizations carry your traffic
traceroute -A google.com
# mtr combines traceroute + ping for real-time latency monitoring per hop
mtr google.com
```

### Learning Resources

- [BGP Table Statistics](https://bgp.potaroo.net/)
- [Hurricane Electric BGP Toolkit](https://bgp.he.net/)
- [PeeringDB](https://www.peeringdb.com/)
- Cisco Networking Academy

### Simulators

- GNS3 - Uses actual router images
- Cisco Packet Tracer - Free learning simulator
- EVE-NG - Virtual network lab

---

**Document Information**
- Last Updated: 2024
- Difficulty: ⭐⭐⭐
- Estimated Learning Time: 3-4 hours
