[Previous: Quality of Service](./21_Quality_of_Service.md)

---

# 22. Multicast

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the difference between unicast, broadcast, and multicast addressing
2. Describe IGMP (v1/v2/v3) operation for group membership management
3. Implement multicast routing with PIM Sparse Mode and Dense Mode
4. Map multicast IP addresses to Ethernet MAC addresses
5. Apply multicast in real-world scenarios (live streaming, financial data, IPTV)

---

## Table of Contents

1. [Multicast Fundamentals](#1-multicast-fundamentals)
2. [Multicast Addressing](#2-multicast-addressing)
3. [IGMP: Group Membership](#3-igmp-group-membership)
4. [Multicast Routing Protocols](#4-multicast-routing-protocols)
5. [Source-Specific Multicast (SSM)](#5-source-specific-multicast-ssm)
6. [Applications](#6-applications)
7. [Exercises](#7-exercises)

---

## 1. Multicast Fundamentals

### 1.1 Unicast vs Broadcast vs Multicast

```
Unicast (one-to-one):
  Source ──────► Receiver A
  Source ──────► Receiver B    3 copies on the network
  Source ──────► Receiver C

Broadcast (one-to-all):
  Source ──────► ALL devices   Even those not interested
                               Wastes bandwidth

Multicast (one-to-many):
  Source ──────┬──► Receiver A (interested)
              ├──► Receiver C (interested)
              │
              ✗    Receiver B (not interested, not sent)

  Only one copy sent per link.
  Network replicates at branch points.
```

### 1.2 Why Multicast?

| Scenario | Unicast | Multicast | Savings |
|----------|---------|-----------|---------|
| Live video to 1000 viewers | 1000 streams × 5 Mbps = 5 Gbps | 1 stream × 5 Mbps = 5 Mbps | 1000× |
| Stock market data to 500 traders | 500 × 100 Kbps = 50 Mbps | 1 × 100 Kbps = 100 Kbps | 500× |
| OS update to 10,000 machines | 10,000 × 2 GB = 20 TB transfer | 1 × 2 GB = 2 GB | 10,000× |

### 1.3 Multicast Delivery Model

```
Multicast tree:

        Source
          │
          ▼
      ┌───────┐
      │Router │ (root)
      │   A   │
      └───┬───┘
          │
     ┌────┴────┐
     ▼         ▼
  ┌──────┐  ┌──────┐
  │Router│  │Router│
  │  B   │  │  C   │
  └──┬───┘  └──┬───┘
     │         │
  ┌──┴──┐   ┌─┴──┐
  ▼     ▼   ▼    ▼
 R1    R2  R3   R4     Receivers (group members)

One packet from source → replicated at routers B and C.
R1-R4 all receive the same stream.
```

---

## 2. Multicast Addressing

### 2.1 IPv4 Multicast Addresses

Class D addresses: 224.0.0.0 – 239.255.255.255

```
Reserved ranges:
  224.0.0.0/24     — Local network control (TTL=1, never forwarded)
    224.0.0.1      — All hosts on this subnet
    224.0.0.2      — All routers on this subnet
    224.0.0.5      — OSPF routers
    224.0.0.6      — OSPF designated routers
    224.0.0.13     — PIM routers

  224.0.1.0/24     — Internetwork control
  232.0.0.0/8      — Source-Specific Multicast (SSM)
  239.0.0.0/8      — Administratively scoped (private, like RFC 1918)
```

### 2.2 IP to MAC Mapping

```python
def multicast_ip_to_mac(ip_address):
    """Convert multicast IP to Ethernet MAC address.

    Rule: MAC = 01:00:5E + lower 23 bits of IP address.
    Note: 5 bits of the IP are lost → 32 IPs share 1 MAC.

    Example: 239.1.2.3
      IP binary: 11101111.00000001.00000010.00000011
      Lower 23:           0.00000001.00000010.00000011
      MAC: 01:00:5E:01:02:03
    """
    octets = [int(x) for x in ip_address.split('.')]
    # Lower 23 bits: clear top bit of second octet
    mac = f"01:00:5E:{octets[1] & 0x7F:02X}:{octets[2]:02X}:{octets[3]:02X}"
    return mac
```

### 2.3 IPv6 Multicast

```
IPv6 multicast: FF00::/8

  FF02::1    — All nodes (link-local)
  FF02::2    — All routers (link-local)
  FF02::5    — OSPF routers
  FF02::9    — RIP routers
  FF02::FB   — mDNS
  FF05::2    — All routers (site-local)

IPv6 MAC mapping:
  MAC = 33:33 + lower 32 bits of IPv6 multicast address
  Example: FF02::1 → 33:33:00:00:00:01
```

---

## 3. IGMP: Group Membership

### 3.1 IGMP Overview

IGMP (Internet Group Management Protocol) manages multicast group membership between hosts and their local router.

```
IGMP conversation:

  Host A                Router              Host B
    │                     │                    │
    │── IGMP Report ────►│                    │
    │   "Join 239.1.1.1" │                    │
    │                     │                    │
    │                     │◄── IGMP Report ───│
    │                     │  "Join 239.1.1.1"  │
    │                     │                    │
    │                     │  Router knows:     │
    │                     │  239.1.1.1 has     │
    │                     │  members on this   │
    │                     │  interface         │
    │                     │                    │
    │◄─ Multicast data ──│──────────────────►│
    │   for 239.1.1.1    │   for 239.1.1.1    │
    │                     │                    │
    │── IGMP Leave ─────►│                    │
    │   "Leave 239.1.1.1" │                    │
    │                     │                    │
    │                     │── IGMP Query ────►│
    │                     │  "Anyone still in  │
    │                     │   239.1.1.1?"       │
    │                     │                    │
    │                     │◄── IGMP Report ───│
    │                     │  "Still here"       │
```

### 3.2 IGMP Versions

| Feature | IGMPv1 | IGMPv2 | IGMPv3 |
|---------|--------|--------|--------|
| Leave mechanism | Timeout only | Explicit leave | Explicit leave |
| Group-specific query | No | Yes | Yes |
| Source filtering | No | No | Yes (include/exclude) |
| Querier election | No | Yes | Yes |
| SSM support | No | No | Yes |

### 3.3 IGMP Snooping

Switches learn which ports have multicast group members and only forward multicast to those ports (instead of flooding to all ports):

```python
class IGMPSnoopingSwitch:
    """IGMP snooping: optimize multicast forwarding at Layer 2.

    Without snooping: multicast flooded to all ports (wasteful).
    With snooping: multicast sent only to interested ports.
    """

    def __init__(self, num_ports):
        self.num_ports = num_ports
        # group_address → set of ports with members
        self.group_table = {}
        self.router_ports = set()  # ports connected to routers

    def handle_igmp_report(self, port, group):
        """Host on this port joined a multicast group."""
        if group not in self.group_table:
            self.group_table[group] = set()
        self.group_table[group].add(port)

    def handle_igmp_leave(self, port, group):
        """Host on this port left a multicast group."""
        if group in self.group_table:
            self.group_table[group].discard(port)
            if not self.group_table[group]:
                del self.group_table[group]

    def forward_multicast(self, ingress_port, group):
        """Determine which ports to forward multicast to."""
        ports = set()

        # Always forward to router ports
        ports.update(self.router_ports)

        # Forward to ports with group members
        if group in self.group_table:
            ports.update(self.group_table[group])

        # Don't send back on ingress port
        ports.discard(ingress_port)
        return ports
```

---

## 4. Multicast Routing Protocols

### 4.1 PIM Dense Mode (PIM-DM)

Flood-and-prune: send multicast everywhere, then prune branches with no receivers.

```
PIM Dense Mode:

Step 1: Flood everywhere
  Source → All routers receive traffic

Step 2: Prune branches with no members
  Routers with no downstream receivers send Prune messages upstream

Step 3: Result = source-based tree (shortest path tree)
  Only branches with receivers remain

Problem: Initial flood wastes bandwidth.
Use case: Dense networks where most subnets have receivers.
```

### 4.2 PIM Sparse Mode (PIM-SM)

Receivers explicitly join; traffic only flows where requested.

```
PIM Sparse Mode:

  1. Receiver sends IGMP Join
  2. Router sends PIM Join toward Rendezvous Point (RP)
  3. Source registers with RP
  4. RP forwards traffic down the shared tree

      Source                    RP
        │                       │
        │── Register ─────────►│
        │                       │
        │   Shared tree:        │
        │                    ┌──┴──┐
        │                    │     │
        │                   R1    R2
        │                    │     │
        │                  Rcv1  Rcv2

  5. Optional: Switch to Shortest Path Tree (SPT)
     for high-bandwidth sources
```

### 4.3 PIM-SM Key Concepts

| Concept | Description |
|---------|-------------|
| Rendezvous Point (RP) | Shared root for all multicast trees in a group |
| Shared Tree (*,G) | Tree rooted at RP, shared by all sources for group G |
| Shortest Path Tree (S,G) | Tree rooted at source S, optimized path to receivers |
| SPT switchover | When receiver traffic exceeds threshold, switch from shared to SPT |
| RP election | Static config, Auto-RP (Cisco), or BSR (standard) |

---

## 5. Source-Specific Multicast (SSM)

### 5.1 Motivation

Standard multicast (ASM) allows any source to send to a group. This creates security concerns (anyone can inject traffic) and requires RP infrastructure.

SSM: receiver specifies both the group AND the source.

```
ASM (Any-Source Multicast):
  Join(G)         — "I want traffic for group G from ANY source"
  Security risk: anyone can send to group G

SSM (Source-Specific Multicast):
  Join(S, G)      — "I want traffic for group G from source S ONLY"
  No RP needed, no unwanted sources

  SSM range: 232.0.0.0/8 (IPv4)
```

### 5.2 SSM Benefits

- No Rendezvous Point needed (simpler infrastructure)
- Receiver controls which sources it receives
- No source registration required
- Better security (source validation)
- Works well with IGMPv3

---

## 6. Applications

### 6.1 Live Video Streaming (IPTV)

```
IPTV architecture:

  Content ─────► Encoder ─────► Multicast ─────► DSLAM/OLT ─────► STB
  Source                        Network                            (TV)

  Each TV channel = one multicast group
  Channel change = IGMP Leave (old) + IGMP Join (new)
  Zapping time: < 2 seconds (leave + join + video decode)
```

### 6.2 Financial Market Data

```
Stock exchange multicast:

  Exchange ──► Multicast feed ──► Trading firms

  Benefits:
    • Microsecond latency (no TCP handshake)
    • Fair: all firms receive data simultaneously
    • Efficient: one stream serves thousands of clients
    • Reliable multicast (PGM/NORM) for gap detection

  Market data groups:
    239.1.1.1  — Equities Level 1 (best bid/ask)
    239.1.1.2  — Equities Level 2 (full order book)
    239.1.2.1  — Options chain
    239.1.3.1  — Futures
```

### 6.3 Multicast DNS (mDNS)

```
mDNS for local service discovery:

  Query:  "printer._ipp._tcp.local" → 224.0.0.251 (mDNS group)
  All devices on LAN hear the query.
  The printer responds with its IP address.

  Used by: Bonjour (Apple), Avahi (Linux)
  Also: Chromecast discovery, IoT device discovery
```

---

## 7. Exercises

### Exercise 1: IP-to-MAC Mapping

Implement multicast IP to MAC address conversion:
1. Convert these IPs: 239.1.2.3, 224.0.0.5, 232.10.20.30, 239.128.1.1
2. Find two multicast IPs that map to the same MAC address
3. Explain why 32 IP addresses share 1 MAC and how switches handle this
4. Implement the inverse: given a MAC, what IP range could it represent?

### Exercise 2: IGMP Snooping Simulation

Build an IGMP snooping switch:
1. 8-port switch, port 8 connected to router
2. Simulate: hosts on ports 1-4 join/leave groups over time
3. For each multicast packet, show which ports receive it
4. Compare bandwidth used: with vs without snooping

### Exercise 3: PIM-SM Shared Tree

Simulate PIM Sparse Mode tree building:
1. Create a 6-router topology with one RP
2. Receivers join group 239.1.1.1 at different times
3. Build the shared tree (*, G) as joins propagate toward RP
4. Source registers with RP, traffic flows down shared tree
5. Visualize the tree at each step

### Exercise 4: Multicast Traffic Generator

Build a multicast sender and receiver:
1. Sender: generates simulated video frames (1 KB, 30 fps)
2. Receiver: joins group, receives frames, calculates jitter and loss
3. Simulate network with configurable delay and drop rate
4. Plot received frame rate vs network loss rate

### Exercise 5: SSM vs ASM Comparison

Compare Any-Source and Source-Specific Multicast:
1. ASM scenario: 3 sources, 5 receivers, RP-based tree
2. SSM scenario: receivers specify source, no RP
3. Simulate: unwanted source tries to inject traffic
4. Measure tree setup time and bandwidth efficiency for both
5. Discuss: when is ASM still preferred over SSM?

---

*End of Lesson 22*
