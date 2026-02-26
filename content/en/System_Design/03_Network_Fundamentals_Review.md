# Network Fundamentals Review

**Previous**: [Scalability Basics](./02_Scalability_Basics.md) | **Next**: [Load Balancing](./04_Load_Balancing.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how DNS resolution works and how DNS-based load balancing distributes traffic across servers
2. Compare CDN push and pull models and determine which suits a given content delivery scenario
3. Describe the key improvements in HTTP/2 (multiplexing, server push, header compression) over HTTP/1.1
4. Explain how HTTP/3 and QUIC address head-of-line blocking at the transport layer
5. Compare REST and gRPC, identifying when each protocol is the better choice for inter-service communication
6. Evaluate the networking trade-offs involved in designing a globally distributed system

**Difficulty**: ⭐⭐
**Estimated Study Time**: 2-3 hours
**Prerequisites**: [Networking folder](../Networking/00_Overview.md) basics

---

Networking is the invisible plumbing of every distributed system. A single web page load can involve dozens of DNS lookups, CDN edge hits, and HTTP round trips -- and each of those steps introduces latency, failure modes, and design choices. Before you can reason about load balancers, caches, or databases at scale, you need a solid grasp of the network layer that connects them all.

## Table of Contents

1. [DNS Operation and Load Balancing](#1-dns-operation-and-load-balancing)
2. [CDN (Content Delivery Network)](#2-cdn-content-delivery-network)
3. [HTTP/2 and HTTP/3](#3-http2-and-http3)
4. [REST vs gRPC](#4-rest-vs-grpc)
5. [Practice Problems](#5-practice-problems)
6. [References](#6-references)

---

## 1. DNS Operation and Load Balancing

### 1.1 DNS Operation Review

```
┌─────────────────────────────────────────────────────────────────┐
│                    DNS Query Process                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Client                                                         │
│    │                                                            │
│    │ 1. www.example.com?                                        │
│    ▼                                                            │
│  ┌───────────────────┐                                          │
│  │ Local DNS Resolver│ ← ISP or 8.8.8.8 (Google)               │
│  │ (Recursive)       │                                          │
│  └─────────┬─────────┘                                          │
│            │ 2. Where is .com?                                  │
│            ▼                                                    │
│  ┌───────────────────┐                                          │
│  │ Root DNS Server   │ (a.root-servers.net, etc. 13 total)     │
│  └─────────┬─────────┘                                          │
│            │ 3. Returns .com TLD server address                 │
│            ▼                                                    │
│  ┌───────────────────┐                                          │
│  │ TLD DNS Server    │ (.com, .org, .kr, etc.)                 │
│  └─────────┬─────────┘                                          │
│            │ 4. Returns example.com NS                          │
│            ▼                                                    │
│  ┌───────────────────┐                                          │
│  │ Authoritative DNS │ (responsible for example.com)            │
│  └─────────┬─────────┘                                          │
│            │ 5. A record: 93.184.216.34                         │
│            ▼                                                    │
│  Client ◄───── Returns IP address                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 DNS-Based Load Balancing

```
┌─────────────────────────────────────────────────────────────────┐
│                  DNS Load Balancing Methods                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Round Robin DNS                                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  www.example.com → 192.168.1.1                             │ │
│  │  www.example.com → 192.168.1.2                             │ │
│  │  www.example.com → 192.168.1.3                             │ │
│  │                                                            │ │
│  │  1st request → 192.168.1.1                                 │ │
│  │  2nd request → 192.168.1.2                                 │ │
│  │  3rd request → 192.168.1.3                                 │ │
│  │  4th request → 192.168.1.1 (cycles back)                   │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  2. Weighted Round Robin                                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Server A (Weight: 5) → 50% traffic                        │ │
│  │  Server B (Weight: 3) → 30% traffic                        │ │
│  │  Server C (Weight: 2) → 20% traffic                        │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  3. Geolocation DNS                                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Korean users    → Seoul server (ap-northeast-2)           │ │
│  │  US users        → Virginia server (us-east-1)             │ │
│  │  European users  → Frankfurt server (eu-central-1)         │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  4. Latency-based DNS                                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Routes to server with lowest latency from user location   │ │
│  │  Provided by AWS Route 53, Cloudflare, etc.                │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 DNS Load Balancing Limitations

```
┌─────────────────────────────────────────────────────────────────┐
│              DNS Load Balancing Limitations                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. TTL-induced delay                                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Traffic continues to failed server until TTL expires     │ │
│  │ • Low TTL = fast failure response but high DNS query load  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  2. Client caching                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Browsers/OS cache DNS results                            │ │
│  │ • Server-side changes don't reflect immediately            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  3. Uneven distribution                                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Large ISP's DNS Resolver represents many users           │ │
│  │ • All users from that ISP connect to same server           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  4. Health check limitations                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Difficult to check real-time server status               │ │
│  │ • Less sophisticated health checks than dedicated LB       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Solution: Use DNS + L4/L7 load balancer combination           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. CDN (Content Delivery Network)

### 2.1 What is CDN?

```
┌─────────────────────────────────────────────────────────────────┐
│                       CDN Concept                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CDN = Geographically distributed server network                │
│        Serving static content close to users                    │
│                                                                  │
│                    ┌─────────────┐                              │
│                    │ Origin      │                              │
│                    │ Server      │                              │
│                    └──────┬──────┘                              │
│                           │                                      │
│              ┌────────────┼────────────┐                        │
│              │            │            │                        │
│              ▼            ▼            ▼                        │
│        ┌─────────┐  ┌─────────┐  ┌─────────┐                   │
│        │ Edge    │  │ Edge    │  │ Edge    │                   │
│        │ Server  │  │ Server  │  │ Server  │                   │
│        │ (Seoul) │  │ (Tokyo) │  │ (NY)    │                   │
│        └────┬────┘  └────┬────┘  └────┬────┘                   │
│             │            │            │                         │
│             ▼            ▼            ▼                         │
│        Korean users  Japanese users  US users                   │
│                                                                  │
│  Advantages:                                                    │
│  • Reduced latency (respond from nearby server)                 │
│  • Reduced Origin server load                                   │
│  • DDoS defense                                                 │
│  • High availability                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Push vs Pull CDN

```
┌─────────────────────────────────────────────────────────────────┐
│                    Push CDN vs Pull CDN                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Push CDN (Pre-distribution)                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Origin Server                                             │ │
│  │       │                                                    │ │
│  │       │ Admin manually uploads                             │ │
│  │       ▼                                                    │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐                    │ │
│  │  │ Edge 1  │  │ Edge 2  │  │ Edge 3  │                    │ │
│  │  │ (Seoul) │  │ (Tokyo) │  │ (NY)    │                    │ │
│  │  └─────────┘  └─────────┘  └─────────┘                    │ │
│  │                                                            │ │
│  │  Features:                                                 │ │
│  │  • Upload directly on content change                       │ │
│  │  • Pre-distributed to all edges                            │ │
│  │  • Fast response from first request                        │ │
│  │                                                            │ │
│  │  Suitable: Large files that change infrequently           │ │
│  │            (videos, game patches)                          │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Pull CDN (Cache on request)                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  User Request                                              │ │
│  │       │                                                    │ │
│  │       ▼                                                    │ │
│  │  ┌─────────┐     Cache Miss?     ┌─────────────┐          │ │
│  │  │ Edge    │─────────────────────│ Origin      │          │ │
│  │  │ Server  │ ◄─────────────────── │ Server     │          │ │
│  │  └─────────┘     Fetch & Cache   └─────────────┘          │ │
│  │       │                                                    │ │
│  │       │ Cached response                                    │ │
│  │       ▼                                                    │ │
│  │     User                                                   │ │
│  │                                                            │ │
│  │  Features:                                                 │ │
│  │  • First request fetches from Origin                       │ │
│  │  • TTL-based cache management                              │ │
│  │  • Simple operation (automatic caching)                    │ │
│  │                                                            │ │
│  │  Suitable: Website static assets (images, CSS, JS)         │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Comparison Summary

| Item | Push CDN | Pull CDN |
|------|----------|----------|
| Distribution | Admin manually uploads | Automatic cache on request |
| First request | Immediate response | Origin access needed |
| Cache management | Manual | Automatic (TTL) |
| Storage cost | High (full distribution) | Low (as needed) |
| Suitable for | Large, infrequently changed files | Frequently changing content |
| Examples | Game updates, movies | Websites, API responses |

### 2.4 CDN Cache Invalidation

```
┌─────────────────────────────────────────────────────────────────┐
│                   CDN Cache Invalidation Strategies              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. TTL (Time To Live) setting                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Cache-Control: max-age=3600  (expire after 1 hour)         │ │
│  │ Cache-Control: s-maxage=86400  (CDN only, 1 day)           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  2. Purge (immediate deletion)                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Force delete cache for specific URL                      │ │
│  │ • Use for urgent fixes                                     │ │
│  │ • API call: POST /purge?url=...                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  3. Versioning (include version in filename)                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ /static/app.v1.js → /static/app.v2.js                      │ │
│  │ /static/app.abc123.js (use hash)                           │ │
│  │                                                            │ │
│  │ Advantage: Deploy new version without cache invalidation   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  4. Soft Purge (Stale-While-Revalidate)                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Serve stale cache while refreshing in background         │ │
│  │ Cache-Control: stale-while-revalidate=60                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. HTTP/2 and HTTP/3

### 3.1 HTTP Version Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    HTTP Version Evolution                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  HTTP/1.1 (1997)                                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Request 1 ──────▶ Response 1                              │ │
│  │  Request 2 ──────▶ Response 2  (sequential)                │ │
│  │  Request 3 ──────▶ Response 3                              │ │
│  │                                                            │ │
│  │  Problems:                                                 │ │
│  │  • Head-of-Line Blocking (HOL)                             │ │
│  │  • One request per connection                              │ │
│  │  • Workarounds: domain sharding, sprite images             │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                           │                                      │
│                           ▼                                      │
│  HTTP/2 (2015)                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  ┌──────────────── Single TCP Connection ─────────────┐   │ │
│  │  │                                                    │   │ │
│  │  │  Stream 1 ═══════▶ ═══════▶ ═══════▶              │   │ │
│  │  │  Stream 2 ═══════▶ ═══════▶ ═══════▶  (parallel)  │   │ │
│  │  │  Stream 3 ═══════▶ ═══════▶ ═══════▶              │   │ │
│  │  │                                                    │   │ │
│  │  └────────────────────────────────────────────────────┘   │ │
│  │                                                            │ │
│  │  Improvements:                                             │ │
│  │  • Multiplexing (multiple streams in one connection)       │ │
│  │  • Header Compression (HPACK)                              │ │
│  │  • Server Push                                             │ │
│  │  • Binary Protocol                                         │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                           │                                      │
│                           ▼                                      │
│  HTTP/3 (2022)                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  ┌──────────────── QUIC (UDP-based) ───────────────────┐  │ │
│  │  │                                                    │   │ │
│  │  │  Stream 1 ═══════▶ (independent, no packet loss)   │   │ │
│  │  │  Stream 2 ═══════▶                                 │   │ │
│  │  │  Stream 3 ═══════▶                                 │   │ │
│  │  │                                                    │   │ │
│  │  └────────────────────────────────────────────────────┘   │ │
│  │                                                            │ │
│  │  Improvements:                                             │ │
│  │  • 0-RTT connection (on reconnection)                      │ │
│  │  • Independent packet loss handling per stream             │ │
│  │  • Connection Migration (connection survives IP change)    │ │
│  │  • Built-in encryption (TLS 1.3)                           │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 HTTP/2 Key Features

```
┌─────────────────────────────────────────────────────────────────┐
│                     HTTP/2 Core Features                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Multiplexing                                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  HTTP/1.1:  ───────────────────────────────────────────    │ │
│  │             │ Req1 │ Res1 │ Req2 │ Res2 │ Req3 │ Res3 │    │ │
│  │             ───────────────────────────────────────────    │ │
│  │                                                            │ │
│  │  HTTP/2:    ───────────────────────────────────────────    │ │
│  │             │ R1│R2│R3│S1│S2│S1│R3│S3│S2│S3│             │ │
│  │             ───────────────────────────────────────────    │ │
│  │             (frame-level interleaving)                     │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  2. Header Compression (HPACK)                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  HTTP/1.1:                                                 │ │
│  │  GET /page HTTP/1.1                                        │ │
│  │  Host: example.com                                         │ │
│  │  User-Agent: Mozilla/5.0...  (repeated, ~800 bytes)        │ │
│  │                                                            │ │
│  │  HTTP/2:                                                   │ │
│  │  • Static table (pre-defined common headers)               │ │
│  │  • Dynamic table (store repeated headers during session)   │ │
│  │  • Huffman encoding                                        │ │
│  │  → 85-88% header size reduction                            │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  3. Server Push                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Client: GET /index.html                                   │ │
│  │  Server:                                                   │ │
│  │    → Response: /index.html                                 │ │
│  │    → Push: /style.css (send before request)                │ │
│  │    → Push: /app.js                                         │ │
│  │                                                            │ │
│  │  Caution: Can conflict with client cache, use carefully    │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 HTTP/3 (QUIC) Advantages

```
┌─────────────────────────────────────────────────────────────────┐
│                    HTTP/3 (QUIC) Advantages                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Independent stream handling                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  HTTP/2 (TCP): All streams wait on packet loss             │ │
│  │                                                            │ │
│  │  Stream 1 ═══╳══════════════════  (loss)                   │ │
│  │  Stream 2 ════════════════════    (waiting)                │ │
│  │  Stream 3 ════════════════════    (waiting)                │ │
│  │                                                            │ │
│  │  HTTP/3 (QUIC): Only lost stream affected                  │ │
│  │                                                            │ │
│  │  Stream 1 ═══╳══════════════════  (retransmit)             │ │
│  │  Stream 2 ═══════════════════════ (continue)               │ │
│  │  Stream 3 ═══════════════════════ (continue)               │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  2. Fast connection establishment (0-RTT)                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  TCP + TLS 1.3:  2-3 RTT needed                            │ │
│  │  ┌────┐        ┌────┐                                      │ │
│  │  │ C  │──SYN──▶│ S  │                                      │ │
│  │  │    │◀SYN+ACK│    │   1 RTT (TCP)                        │ │
│  │  │    │──ACK──▶│    │                                      │ │
│  │  │    │◀──────▶│    │   1 RTT (TLS)                        │ │
│  │  └────┘        └────┘                                      │ │
│  │                                                            │ │
│  │  QUIC (reconnection):  0 RTT                               │ │
│  │  ┌────┐        ┌────┐                                      │ │
│  │  │ C  │──DATA─▶│ S  │  Immediate data transfer             │ │
│  │  │    │◀──────▶│    │  (reuse previous session key)        │ │
│  │  └────┘        └────┘                                      │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  3. Connection Migration                                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Wi-Fi → LTE switch:                                       │ │
│  │                                                            │ │
│  │  TCP: Connection drops, need reconnection                  │ │
│  │  QUIC: Connection ID-based, connection survives            │ │
│  │                                                            │ │
│  │  Big advantage in mobile environments!                     │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 Version Comparison Table

| Feature | HTTP/1.1 | HTTP/2 | HTTP/3 |
|------|----------|--------|--------|
| Transport Layer | TCP | TCP | QUIC (UDP) |
| Multiplexing | X | O | O |
| HOL Blocking | O | Partial (TCP) | X |
| Header Compression | X | HPACK | QPACK |
| Server Push | X | O | O |
| Connection Setup | 1-2 RTT | 1-2 RTT | 0-1 RTT |
| Connection Migration | X | X | O |

---

## 4. REST vs gRPC

### 4.1 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    REST vs gRPC Overview                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  REST (Representational State Transfer)                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  • HTTP-based architectural style                          │ │
│  │  • JSON/XML format                                         │ │
│  │  • Resource-centric (URL + HTTP Method)                    │ │
│  │  • Stateless                                               │ │
│  │                                                            │ │
│  │  GET /users/123                                            │ │
│  │  POST /users                                               │ │
│  │  PUT /users/123                                            │ │
│  │  DELETE /users/123                                         │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  gRPC (Google Remote Procedure Call)                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  • HTTP/2-based                                            │ │
│  │  • Protocol Buffers (Protobuf) serialization               │ │
│  │  • Function call style                                     │ │
│  │  • Bidirectional streaming support                         │ │
│  │                                                            │ │
│  │  service UserService {                                     │ │
│  │    rpc GetUser(GetUserRequest) returns (User);             │ │
│  │    rpc ListUsers(ListRequest) returns (stream User);       │ │
│  │  }                                                         │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Detailed Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    REST vs gRPC Detailed Comparison              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Data Format                                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  REST (JSON):                                              │ │
│  │  {                                                         │ │
│  │    "id": 123,                                              │ │
│  │    "name": "John",                                         │ │
│  │    "email": "john@example.com"                             │ │
│  │  }                                                         │ │
│  │  → Human-readable, larger size                             │ │
│  │                                                            │ │
│  │  gRPC (Protobuf):                                          │ │
│  │  message User {                                            │ │
│  │    int32 id = 1;                                           │ │
│  │    string name = 2;                                        │ │
│  │    string email = 3;                                       │ │
│  │  }                                                         │ │
│  │  → Binary format, smaller size (10x difference)            │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Performance                                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  REST:                                                     │ │
│  │  • Text parsing overhead                                   │ │
│  │  • One request per connection in HTTP/1.1                  │ │
│  │                                                            │ │
│  │  gRPC:                                                     │ │
│  │  • Fast parsing with binary serialization                  │ │
│  │  • HTTP/2 Multiplexing                                     │ │
│  │  • 7-10x faster serialization                              │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Streaming                                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  REST:                                                     │ │
│  │  • Basically request-response pattern                      │ │
│  │  • Need SSE, WebSocket for streaming                       │ │
│  │                                                            │ │
│  │  gRPC:                                                     │ │
│  │  • Unary: single request-response                          │ │
│  │  • Server Streaming: server sends multiple responses       │ │
│  │  • Client Streaming: client sends multiple requests        │ │
│  │  • Bidirectional: both directions streaming                │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Selection Criteria

```
┌─────────────────────────────────────────────────────────────────┐
│                     REST vs gRPC Selection Criteria              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Choose REST when:                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  ✓ Public API (for external developers)                    │ │
│  │  ✓ Direct browser calls                                    │ │
│  │  ✓ Simple CRUD operations                                  │ │
│  │  ✓ Debugging convenience important                         │ │
│  │  ✓ Team familiar with REST                                 │ │
│  │  ✓ Caching important (HTTP cache)                          │ │
│  │                                                            │ │
│  │  Examples:                                                 │ │
│  │  • GitHub API, Stripe API                                  │ │
│  │  • General web services                                    │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Choose gRPC when:                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  ✓ Internal communication between microservices            │ │
│  │  ✓ High performance needed (low latency, high throughput)  │ │
│  │  ✓ Real-time bidirectional communication                   │ │
│  │  ✓ Strong-typed API contract                               │ │
│  │  ✓ Multi-language environment (code generation)            │ │
│  │  ✓ Network bandwidth constraints                           │ │
│  │                                                            │ │
│  │  Examples:                                                 │ │
│  │  • Netflix, Google internal                                │ │
│  │  • Real-time games, IoT                                    │ │
│  │  • Mobile backends                                         │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Hybrid Approach:                                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  External API: REST (developer-friendly)                   │ │
│  │  Internal comms: gRPC (high performance)                   │ │
│  │                                                            │ │
│  │  ┌──────────┐     REST     ┌───────────┐                   │ │
│  │  │ Client   │─────────────▶│ API       │                   │ │
│  │  │ (Web/App)│              │ Gateway   │                   │ │
│  │  └──────────┘              └─────┬─────┘                   │ │
│  │                                  │ gRPC                    │ │
│  │                    ┌─────────────┼─────────────┐           │ │
│  │                    ▼             ▼             ▼           │ │
│  │               ┌────────┐   ┌────────┐   ┌────────┐         │ │
│  │               │Service │   │Service │   │Service │         │ │
│  │               │   A    │   │   B    │   │   C    │         │ │
│  │               └────────┘   └────────┘   └────────┘         │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Comparison Summary Table

| Item | REST | gRPC |
|------|------|------|
| Protocol | HTTP/1.1, HTTP/2 | HTTP/2 |
| Data Format | JSON, XML | Protocol Buffers |
| Streaming | Limited | Native support |
| Code Generation | OpenAPI (optional) | Required |
| Browser Support | Perfect | grpc-web needed |
| Learning Curve | Low | Medium |
| Performance | Good | Very Good |
| Debugging | Easy | Tools needed |

---

## 5. Practice Problems

### Problem 1: DNS Load Balancing Design

Assume you're running a global service. Design a DNS-based load balancing strategy.

Conditions:
- 3 regions: Seoul, Tokyo, Virginia
- Normal: Route to closest region
- Failures: Automatic failover to other region

### Problem 2: CDN Strategy

Develop CDN strategy for online video streaming service.

a) Decide between Push and Pull CDN and explain why.
b) Design cache strategy for new content uploads.

### Problem 3: HTTP Version Selection

Choose appropriate HTTP version for these scenarios and explain why.

a) Mobile app (frequent network switches)
b) Legacy system integration
c) News site with many images

### Problem 4: REST vs gRPC

Must choose communication method for microservices architecture.

Situation:
- 10 internal services
- Services written in Go, Java, Python
- Need real-time data synchronization

---

## Answers

### Problem 1 Answer

```
DNS Configuration:
1. Use Geolocation DNS
   - Korea → Seoul region IP
   - Japan → Tokyo region IP
   - US → Virginia region IP

2. Health Check Setup
   - Monitor health check endpoint for each region
   - Remove from DNS on failure

3. Failover Configuration
   - Primary: Closest region
   - Secondary: Second closest region
   - Example: Seoul → Tokyo → Virginia

4. TTL Configuration
   - Low TTL (60-300 seconds) for fast failure response

Recommend AWS Route 53 or Cloudflare
```

### Problem 2 Answer

```
a) Choose Push CDN

Reasons:
- Videos are large and change infrequently
- Need fast response on first view
- Can pre-distribute popular content

b) Cache Strategy:
1. New Content:
   - Pre-distribute to major edge servers after upload
   - Prioritize distribution for expected popular content

2. Live Content:
   - Pull method + short TTL
   - Streaming optimization settings

3. Version Management:
   - Include version in content ID
   - Issue new URL on update
```

### Problem 3 Answer

```
a) Mobile app: HTTP/3
   - Connection Migration survives network switches
   - 0-RTT for fast reconnection
   - Improved battery efficiency

b) Legacy system: HTTP/1.1
   - Wide compatibility
   - No conflicts with existing infrastructure
   - Gradual upgrade if needed

c) Image-heavy site: HTTP/2
   - Multiplexing loads many images simultaneously
   - Header compression for efficiency
   - Server Push for pre-sending CSS/JS
```

### Problem 4 Answer

```
Choose gRPC

Reasons:
1. Internal service communication
   - Not public, browser support not needed

2. Multi-language environment
   - Go, Java, Python all support gRPC
   - Protobuf auto-generates code → consistency

3. Real-time synchronization
   - Bidirectional streaming for real-time data exchange
   - Server Streaming for event push

4. Performance
   - Efficient for high-frequency inter-service communication
   - Protobuf serialization for network efficiency

Implementation:
- Manage common .proto files
- Generate gRPC client/server code per service
```

---

## Hands-On Exercises

### Exercise 1: DNS Resolution Tracer

Build a simple DNS resolution simulator that demonstrates recursive vs. iterative queries.

**Tasks:**
1. Create a hierarchy of DNS servers: Root → TLD (.com) → Authoritative (example.com)
2. Implement `recursive_resolve(domain)` where each server queries the next level and returns the final answer
3. Implement `iterative_resolve(domain)` where the resolver follows referrals itself
4. Track and compare the number of messages exchanged in each mode
5. Add a caching layer and show how subsequent lookups are faster

### Exercise 2: HTTP/1.1 vs HTTP/2 Comparison

Simulate the performance difference between HTTP/1.1 sequential requests and HTTP/2 multiplexing.

**Tasks:**
1. Model a page that needs 10 resources (CSS, JS, images) with varying sizes
2. Simulate HTTP/1.1: max 6 parallel connections, each resource blocks its connection
3. Simulate HTTP/2: all 10 requests multiplexed on 1 connection, interleaved frames
4. Calculate total page load time for each protocol
5. Show how head-of-line blocking affects HTTP/1.1 when one resource is slow

### Exercise 3: REST vs gRPC Payload Comparison

Compare message sizes and serialization overhead between JSON (REST) and Protocol Buffers (gRPC).

**Tasks:**
1. Define a sample data structure (e.g., user profile with name, age, email, addresses list)
2. Serialize it as JSON (`json.dumps`) and measure byte size
3. Simulate protobuf encoding (use a simplified binary encoding or actual `struct.pack`)
4. Compare payload sizes for 1, 10, and 100 records
5. Discuss when the size difference matters (high-throughput microservices vs. simple APIs)

---

## 6. References

### RFC Documents
- RFC 7540 - HTTP/2
- RFC 9000 - QUIC
- RFC 9114 - HTTP/3

### Online Resources
- [HTTP/2 Explained](https://http2-explained.haxx.se/)
- [HTTP/3 Explained](https://http3-explained.haxx.se/)
- [gRPC Documentation](https://grpc.io/docs/)
- [Cloudflare CDN](https://www.cloudflare.com/learning/cdn/)

### Tools
- [WebPageTest](https://www.webpagetest.org/) - HTTP version testing
- [Postman](https://www.postman.com/) - REST API testing
- [grpcurl](https://github.com/fullstorydev/grpcurl) - gRPC testing

---

**Previous**: [Scalability Basics](./02_Scalability_Basics.md) | **Next**: [Load Balancing](./04_Load_Balancing.md)

---

**Document Information**
- Last Updated: 2024
- Difficulty: ⭐⭐
- Estimated Study Time: 2-3 hours
