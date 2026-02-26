# Routing Basics

**Previous**: [Subnetting Practice](./07_Subnetting_Practice.md) | **Next**: [Routing Protocols](./09_Routing_Protocols.md)

**Difficulty**: ⭐⭐⭐
**Estimated Learning Time**: 2-3 hours
**Prerequisites**: [07_Subnetting_Practice.md](./07_Subnetting_Practice.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what routing is and describe the role of a router in inter-network communication
2. Read and interpret a routing table, identifying destination, next hop, metric, and interface fields
3. Configure static routes and explain when static routing is appropriate
4. Describe how dynamic routing differs from static routing and list its advantages
5. Explain the concept of a default gateway and its role in host communication
6. Apply the longest prefix match rule to determine how a router selects the best route for a packet

---

Routing is how the Internet works at its core. When you send a packet to a server on the other side of the world, it must traverse dozens of routers, each independently deciding where to forward the packet next. Understanding routing tables, static vs. dynamic routing, and prefix matching is essential for designing networks, diagnosing "destination unreachable" errors, and configuring anything from a home router to an enterprise backbone.

## Table of Contents

1. [What is Routing?](#1-what-is-routing)
2. [The Role of Routers](#2-the-role-of-routers)
3. [Routing Table Structure](#3-routing-table-structure)
4. [Static Routing vs Dynamic Routing](#4-static-routing-vs-dynamic-routing)
5. [Default Gateway](#5-default-gateway)
6. [Longest Prefix Match](#6-longest-prefix-match)
7. [Practice Problems](#7-practice-problems)
8. [Next Steps](#8-next-steps)
9. [References](#9-references)

---

## 1. What is Routing?

### 1.1 Definition of Routing

Routing is the process of determining the optimal path to forward packets from source to destination across a network.

```
                        Basic Concept of Routing

   Source                                              Destination
[192.168.1.10] ─────?─────?─────?─────?───────► [10.0.0.50]

                    Router determines path

              Router A ──── Router B ──── Router C
                  │             │             │
   [192.168.1.10] │             │             │ [10.0.0.50]
                  └─────────────┴─────────────┘
                        Selected optimal path
```

### 1.2 Routing vs Switching

| Aspect | Switching (L2) | Routing (L3) |
|------|------------|------------|
| Operating Layer | Data Link Layer | Network Layer |
| Address Used | MAC Address | IP Address |
| Scope | Within same network | Between different networks |
| Device | Switch | Router |
| Table | MAC Address Table | Routing Table |

```
Switching (within same network)          Routing (between networks)

  ┌─────────────────┐                  Network A        Network B
  │  192.168.1.0/24 │                 ┌────────┐      ┌────────┐
  │                 │                 │        │      │        │
  │ PC1 ─── Switch ─── PC2           │ PC1    │      │   PC2  │
  │                 │                 │   │    │      │    │   │
  │ MAC-based       │                 │   └────┼──────┼────┘   │
  │   forwarding    │                 │ Router │      │ Router │
  └─────────────────┘                 └────────┘      └────────┘
                                          IP-based forwarding
```

### 1.3 Purpose of Routing

1. **Network Connectivity**: Connect different networks
2. **Path Determination**: Select optimal path to destination
3. **Traffic Distribution**: Load balancing across network
4. **Fault Recovery**: Maintain connectivity through alternate paths

---

## 2. The Role of Routers

### 2.1 Basic Functions of Routers

```
                        Core Functions of Routers

┌─────────────────────────────────────────────────────────────┐
│                        Router                               │
├─────────────────────────────────────────────────────────────┤
│  1. Receive packet      ─────►  Receive packet at interface │
│  2. Check destination   ─────►  Check destination IP in header│
│  3. Routing table lookup ────►  Search for optimal path     │
│  4. Forward packet      ─────►  Send to next hop            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Router Components

```
┌────────────────────────────────────────────────────────┐
│                     Router Internal Structure           │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐ │
│  │   CPU    │  │   RAM    │  │    Routing Table     │ │
│  │(Processing)│ │(Temp Storage)│ │   (Path Info)    │ │
│  └──────────┘  └──────────┘  └──────────────────────┘ │
│                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐ │
│  │  NVRAM   │  │  Flash   │  │    ARP Cache         │ │
│  │ (Config) │  │  (OS)    │  │   (IP-MAC Mapping)   │ │
│  └──────────┘  └──────────┘  └──────────────────────┘ │
│                                                        │
│  ┌─────────────────────────────────────────────────┐  │
│  │              Interfaces (Ports)                  │  │
│  │  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐           │  │
│  │  │Eth0│ │Eth1│ │Eth2│ │Ser0│ │Ser1│           │  │
│  │  └────┘ └────┘ └────┘ └────┘ └────┘           │  │
│  └─────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

### 2.3 Packet Processing Flow

```
[PC A: 192.168.1.10] ──────► [Router] ──────► [PC B: 10.0.0.50]

Step 1: Receive Packet
┌──────────────────────────────────────────────┐
│ Ethernet Header │ IP Header │ Data           │
│ Dst MAC: Router │ Dst IP: 10.0.0.50          │
│ Src MAC: PC A   │ Src IP: 192.168.1.10       │
└──────────────────────────────────────────────┘

Step 2: Remove L2 Header, Check IP Header
┌──────────────────────────────────────────────┐
│ IP Header │ Data                             │
│ Dst IP: 10.0.0.50                            │
│ Src IP: 192.168.1.10                         │
│ TTL: 64 → 63 (decrement)                     │
└──────────────────────────────────────────────┘

Step 3: Routing Table Lookup
┌──────────────────────────────────────────────┐
│ Destination: 10.0.0.50                       │
│ Match: 10.0.0.0/24 via 192.168.2.1 Eth1     │
│ Next Hop: 192.168.2.1                        │
│ Output Interface: Eth1                       │
└──────────────────────────────────────────────┘

Step 4: Add New L2 Header, Forward Packet
┌──────────────────────────────────────────────┐
│ Ethernet Header │ IP Header │ Data           │
│ Dst MAC: Next Hop│ Dst IP: 10.0.0.50         │
│ Src MAC: Router  │ Src IP: 192.168.1.10      │
│ (Eth1 MAC)       │ TTL: 63                   │
└──────────────────────────────────────────────┘
```

### 2.4 Router vs L3 Switch

| Feature | Router | L3 Switch |
|------|--------|----------|
| Primary Use | WAN connectivity, complex routing | VLAN inter-routing in LAN |
| Processing Method | Software-based | Hardware (ASIC)-based |
| Port Count | Few (2-8) | Many (24-48) |
| Features | Rich (NAT, VPN, etc.) | Limited |
| Price | Expensive | Relatively cheaper |

---

## 3. Routing Table Structure

### 3.1 Components of Routing Table

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Routing Table                                │
├──────────────┬────────────┬──────────┬───────────┬─────────────────┤
│ Destination  │ Subnet Mask│ Next Hop │ Interface │ Metric/Distance │
│   Network    │            │          │           │                 │
├──────────────┼────────────┼──────────┼───────────┼─────────────────┤
│ 10.0.0.0     │ /24        │ 192.168.1.1│ Eth0    │ 10              │
│ 172.16.0.0   │ /16        │ 192.168.2.1│ Eth1    │ 20              │
│ 192.168.0.0  │ /24        │ directly  │ Eth0     │ 0 (direct)      │
│ 0.0.0.0      │ /0         │ 203.0.113.1│ Eth2    │ 1 (default)     │
└──────────────┴────────────┴──────────┴───────────┴─────────────────┘
```

### 3.2 Routing Table Entry Types

| Type | Description | Example |
|------|------|------|
| Directly Connected (C) | Network directly connected to router | C 192.168.1.0/24 |
| Static Route (S) | Manually configured by admin | S 10.0.0.0/8 via 192.168.1.1 |
| Dynamic Route | Learned via routing protocol | R, O, B, etc. |
| Default Route (S*) | 0.0.0.0/0 (default route) | S* 0.0.0.0/0 via 203.0.113.1 |

### 3.3 Real Routing Table Examples

**Linux (ip route)**
```bash
$ ip route show
default via 192.168.1.1 dev eth0 proto dhcp metric 100
10.0.0.0/8 via 192.168.1.254 dev eth0
172.16.0.0/16 via 192.168.1.253 dev eth0
192.168.1.0/24 dev eth0 proto kernel scope link src 192.168.1.10
192.168.2.0/24 dev eth1 proto kernel scope link src 192.168.2.1
```

**Cisco Router**
```
Router# show ip route
Codes: C - connected, S - static, R - RIP, O - OSPF, B - BGP

Gateway of last resort is 203.0.113.1 to network 0.0.0.0

C    192.168.1.0/24 is directly connected, GigabitEthernet0/0
C    192.168.2.0/24 is directly connected, GigabitEthernet0/1
S    10.0.0.0/8 [1/0] via 192.168.1.254
O    172.16.0.0/16 [110/20] via 192.168.2.254, 00:05:32, Gi0/1
S*   0.0.0.0/0 [1/0] via 203.0.113.1
```

### 3.4 Metric and Administrative Distance

**Metric**: Route preference within same protocol
```
Destination: 10.0.0.0/24
Route 1: via 192.168.1.1 (metric: 10) ← Selected
Route 2: via 192.168.2.1 (metric: 20)
```

**Administrative Distance (AD)**: Trustworthiness between different protocols

| Route Source | Administrative Distance |
|----------|----------|
| Directly Connected | 0 |
| Static Route | 1 |
| EIGRP (internal) | 90 |
| OSPF | 110 |
| RIP | 120 |
| External BGP | 20 |
| Internal BGP | 200 |

```
Route Selection Example for Same Destination

Destination: 10.0.0.0/24
OSPF route:   AD=110, metric=20
RIP route:    AD=120, metric=2
Static route: AD=1,   metric=0

Selected: Static route (lowest AD)
```

---

## 4. Static Routing vs Dynamic Routing

### 4.1 Static Routing

Administrator manually configures routes.

```
                      Static Routing Configuration

       ┌─────────┐              ┌─────────┐
       │  R1     │              │  R2     │
       │         │ 10.0.0.0/30  │         │
       │ .1────────────────────.2         │
       │         │              │         │
  ┌────┴─────┐   │              │   ┌─────┴────┐
  │192.168.1.0│  │              │   │172.16.0.0│
  │   /24     │  │              │   │  /16     │
  └──────────┘   │              │   └──────────┘

R1 Configuration:
ip route 172.16.0.0 255.255.0.0 10.0.0.2

R2 Configuration:
ip route 192.168.1.0 255.255.255.0 10.0.0.1
```

**Advantages**:
- Simple configuration
- No CPU/memory overhead
- Security (no routing information exchange)
- Bandwidth saving

**Disadvantages**:
- Manual update required for network changes
- Difficult to manage in large networks
- No automatic failover

**Use Cases**:
- Small networks
- Stub networks with single path
- Default gateway configuration
- Security-critical segments

### 4.2 Dynamic Routing

Routing protocols automatically learn and update routes.

```
                      Dynamic Routing Operation

       ┌─────────┐              ┌─────────┐
       │  R1     │◄────────────►│  R2     │
       │ OSPF    │ Routing Info │ OSPF    │
       │         │   Exchange   │         │
       └────┬────┘              └────┬────┘
            │                        │
     ┌──────┴──────┐          ┌──────┴──────┐
     │192.168.1.0/24│         │172.16.0.0/16│
     │   Advertise │         │  Advertise  │
     └─────────────┘          └─────────────┘

R1 and R2 automatically learn each other's networks
```

**Advantages**:
- Automatic route learning and updates
- Automatic failover
- Easy management of large networks
- Adapts to network changes

**Disadvantages**:
- CPU/memory usage
- Bandwidth consumption (routing updates)
- Complex configuration
- Security considerations needed

### 4.3 Comparison Summary

| Feature | Static Routing | Dynamic Routing |
|------|-----------|------------|
| Configuration | Manual | Automatic |
| Scalability | Low | High |
| Fault Response | Manual | Automatic |
| Resource Usage | Low | High |
| Security | High | Requires configuration |
| Management Complexity | Small: Low / Large: High | Initial: High / Operation: Low |

### 4.4 Hybrid Approach

Real networks often use both approaches together.

```
                     Enterprise Network Example

        ISP                                    ISP
         │                                      │
    Static route                           Static route
    (default gateway)                      (default gateway)
         │                                      │
    ┌────┴────┐                           ┌────┴────┐
    │ HQ      │◄─────── OSPF ────────────►│ Branch  │
    │ Router  │      (Dynamic Routing)     │ Router  │
    └────┬────┘                           └────┬────┘
         │                                      │
    Internal Network                      Internal Network
```

---

## 5. Default Gateway

### 5.1 What is Default Gateway?

Default Gateway is the IP address of the router used by hosts to send packets to other networks.

```
┌─────────────────────────────────────────────────────────────┐
│                     PC Network Configuration                 │
├─────────────────────────────────────────────────────────────┤
│  IP Address:       192.168.1.100                            │
│  Subnet Mask:      255.255.255.0                            │
│  Default Gateway:  192.168.1.1  ◄── Router's LAN Interface │
│  DNS Server:       8.8.8.8                                  │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Default Gateway Operation

```
Destination Determination Process

┌──────────────────────────────────────────────────────────┐
│                 PC (192.168.1.100/24)                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Packet transmission request: destination 10.0.0.50      │
│                                                          │
│  1. Is destination in same network?                      │
│     └─ Destination IP AND my subnet mask                 │
│        10.0.0.50 AND 255.255.255.0 = 10.0.0.0           │
│     └─ My network: 192.168.1.0                          │
│     └─ 10.0.0.0 ≠ 192.168.1.0 → Different network!     │
│                                                          │
│  2. Send to default gateway                              │
│     └─ Destination IP: 10.0.0.50 (unchanged)            │
│     └─ Destination MAC: Gateway MAC (via ARP)           │
│                                                          │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
                    ┌───────────┐
                    │  Router   │
                    │192.168.1.1│
                    └───────────┘
                          │
                    Forward to other network
```

### 5.3 Default Route

Router's default route is represented as 0.0.0.0/0.

```
Default Route in Routing Table

┌─────────────────────────────────────────────────────────┐
│ Destination     │ Next Hop      │ Interface           │
├─────────────────┼───────────────┼────────────────────┤
│ 192.168.1.0/24  │ directly      │ Eth0               │
│ 10.0.0.0/24     │ 192.168.2.1   │ Eth1               │
│ 172.16.0.0/16   │ 192.168.2.2   │ Eth1               │
│ 0.0.0.0/0       │ 203.0.113.1   │ Eth2  ← Default route│
└─────────────────┴───────────────┴────────────────────┘

Packet Processing:
- 192.168.1.50 → Forward directly to Eth0
- 10.0.0.100   → Forward to 192.168.2.1
- 8.8.8.8      → No matching route → Use default route
                 Forward to 203.0.113.1
```

### 5.4 Multiple Gateways

```
               Load Balancing / Fault Tolerance

    ┌───────────┐         ┌───────────┐
    │ Router 1  │         │ Router 2  │
    │192.168.1.1│         │192.168.1.2│
    │ (primary) │         │ (backup)  │
    └─────┬─────┘         └─────┬─────┘
          │                     │
          └──────────┬──────────┘
                     │
              ┌──────┴──────┐
              │   Switch    │
              └──────┬──────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
 ┌──┴──┐          ┌──┴──┐          ┌──┴──┐
 │ PC1 │          │ PC2 │          │ PC3 │
 │GW:.1│          │GW:.1│          │GW:.2│
 └─────┘          └─────┘          └─────┘
```

**VRRP/HSRP**: Gateway redundancy via virtual IP

```
Virtual IP: 192.168.1.254 (VRRP)
Router 1: 192.168.1.1 (Master)
Router 2: 192.168.1.2 (Backup)

All PCs' gateway: 192.168.1.254
→ Automatically switches to Router 2 if Router 1 fails
```

---

## 6. Longest Prefix Match

### 6.1 Concept

Select the route that most specifically matches the destination IP from the routing table.

```
Longest Prefix Match Principle

Routing Table:
┌─────────────────┬───────────────┐
│ Destination     │ Next Hop      │
│   Network       │               │
├─────────────────┼───────────────┤
│ 10.0.0.0/8      │ Router A      │
│ 10.1.0.0/16     │ Router B      │
│ 10.1.1.0/24     │ Router C      │
│ 0.0.0.0/0       │ Router D      │
└─────────────────┴───────────────┘

Destination: 10.1.1.100

Match Analysis:
- 10.0.0.0/8     match (8 bits)   ✓
- 10.1.0.0/16    match (16 bits)  ✓✓
- 10.1.1.0/24    match (24 bits)  ✓✓✓ ← Selected (longest match)
- 0.0.0.0/0      match (0 bits)   ✓

Result: Forward to Router C
```

### 6.2 Understanding in Binary

```
Destination IP: 10.1.1.100 = 00001010.00000001.00000001.01100100

10.0.0.0/8:
  00001010.xxxxxxxx.xxxxxxxx.xxxxxxxx
  ^^^^^^^^ (8 bits match)

10.1.0.0/16:
  00001010.00000001.xxxxxxxx.xxxxxxxx
  ^^^^^^^^.^^^^^^^^ (16 bits match)

10.1.1.0/24:
  00001010.00000001.00000001.xxxxxxxx
  ^^^^^^^^.^^^^^^^^.^^^^^^^^ (24 bits match) ← Longest match
```

### 6.3 Importance of Longest Prefix Match

```
Example: Traffic Engineering

Internet Traffic Path Control

                        ┌──────────┐
                        │ Internet │
                        └────┬─────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
         │ ISP A   │   │ ISP B   │   │ ISP C   │
         └────┬────┘   └────┬────┘   └────┬────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                      ┌──────┴──────┐
                      │Enterprise   │
                      │   Router    │
                      │              │
                      │Routing Policy:│
                      │ 0.0.0.0/0    │
                      │   → ISP A    │
                      │ 8.8.8.0/24   │
                      │   → ISP B    │
                      └──────────────┘

Accessing Google DNS (8.8.8.8):
- 0.0.0.0/0 (ISP A) matches
- 8.8.8.0/24 (ISP B) matches ← Selected by longest match
```

### 6.4 Route Summarization

Using longest prefix match for route summarization:

```
Before Summarization              After Summarization

192.168.0.0/24 → R1            192.168.0.0/22 → R1
192.168.1.0/24 → R1               (4 networks into 1)
192.168.2.0/24 → R1
192.168.3.0/24 → R1

Advantages:
- Reduced routing table size
- Reduced routing updates
- Improved network stability
```

---

## 7. Practice Problems

### Problem 1: Routing Table Analysis

Analyze the following routing table and answer the questions.

```
Router# show ip route
C    192.168.1.0/24 is directly connected, Eth0
C    192.168.2.0/24 is directly connected, Eth1
S    10.0.0.0/8 [1/0] via 192.168.1.254
S    172.16.0.0/16 [1/0] via 192.168.2.254
O    172.16.10.0/24 [110/20] via 192.168.2.253
S*   0.0.0.0/0 [1/0] via 192.168.1.1
```

a) What is the next hop for packets to destination 10.10.10.10?
b) What is the next hop for packets to destination 172.16.10.50?
c) What is the next hop for packets to destination 8.8.8.8?
d) What do C, S, and O mean?

### Problem 2: Static Routing Configuration

Write the necessary static routing commands for R1 and R2 in the following network.

```
Network A          R1           R2          Network B
192.168.1.0/24 ──[.1]──[.2]──[.1]──[.2]── 10.0.0.0/24
                   192.168.100.0/30
```

### Problem 3: Longest Prefix Match

Determine the next hop for each destination IP using the following routing table.

```
Routing Table:
- 0.0.0.0/0      → 10.0.0.1
- 10.0.0.0/8     → 10.0.0.2
- 10.10.0.0/16   → 10.0.0.3
- 10.10.10.0/24  → 10.0.0.4
- 10.10.10.128/25→ 10.0.0.5
```

a) 10.10.10.200
b) 10.10.10.50
c) 10.10.20.100
d) 10.20.30.40
e) 8.8.8.8

### Problem 4: Network Design

Design routing for a company with 3 branch offices.

```
         HQ
    192.168.0.0/24
           │
      ┌────┴────┐
      │ Core R  │
      └────┬────┘
     ┌─────┼─────┐
     │     │     │
  Branch A Branch B Branch C
 10.1.0/24 10.2.0/24 10.3.0/24
```

Write the necessary static routes for each router.

---

## Answers

### Problem 1 Answers

a) **192.168.1.254** (matches 10.0.0.0/8 route)
b) **192.168.2.253** (172.16.10.0/24 is more specific than 172.16.0.0/16, OSPF route)
c) **192.168.1.1** (uses default route 0.0.0.0/0)
d)
   - C: Connected (directly connected)
   - S: Static (static route)
   - O: OSPF (dynamic routing protocol)

### Problem 2 Answers

```
R1:
ip route 10.0.0.0 255.255.255.0 192.168.100.2

R2:
ip route 192.168.1.0 255.255.255.0 192.168.100.1
```

### Problem 3 Answers

a) 10.10.10.200 → **10.0.0.5** (matches 10.10.10.128/25, 200 is in 128-255 range)
b) 10.10.10.50 → **10.0.0.4** (matches 10.10.10.0/24, 50 is in 0-127 range)
c) 10.10.20.100 → **10.0.0.3** (matches 10.10.0.0/16)
d) 10.20.30.40 → **10.0.0.2** (matches 10.0.0.0/8)
e) 8.8.8.8 → **10.0.0.1** (matches default route)

### Problem 4 Answers

```
Core Router:
# Directly connected networks are automatically added
# Default route (to Internet)
ip route 0.0.0.0 0.0.0.0 [ISP-Gateway]

Branch A Router:
ip route 192.168.0.0 255.255.255.0 [CoreRouterIP]
ip route 10.2.0.0 255.255.255.0 [CoreRouterIP]
ip route 10.3.0.0 255.255.255.0 [CoreRouterIP]
ip route 0.0.0.0 0.0.0.0 [CoreRouterIP]

# Or simply:
ip route 0.0.0.0 0.0.0.0 [CoreRouterIP]

Branch B and C follow the same pattern
```

---

**Previous**: [Subnetting Practice](./07_Subnetting_Practice.md) | **Next**: [Routing Protocols](./09_Routing_Protocols.md)

---

## 8. References

### Command Reference

```bash
# Linux - Check routing table
ip route show
route -n
netstat -rn

# macOS
netstat -rn

# Windows
route print
netstat -rn

# Path tracing
traceroute google.com      # Linux/macOS
tracert google.com         # Windows

# Add static route (Linux)
sudo ip route add 10.0.0.0/8 via 192.168.1.1
sudo ip route del 10.0.0.0/8

# Add static route (Windows)
route add 10.0.0.0 mask 255.0.0.0 192.168.1.1
route delete 10.0.0.0
```

### Learning Resources

- RFC 1812 - Requirements for IP Version 4 Routers
- Cisco Networking Academy - Routing Concepts
- [Juniper Networks - Understanding Routing](https://www.juniper.net/documentation/)

### Cisco IOS Commands

```
# Check routing table
show ip route
show ip route summary

# Configure static route
ip route 10.0.0.0 255.0.0.0 192.168.1.1
ip route 0.0.0.0 0.0.0.0 203.0.113.1

# Interface information
show ip interface brief
```

---

**Document Information**
- Last Updated: 2024
- Difficulty: ⭐⭐⭐
- Estimated Learning Time: 2-3 hours
