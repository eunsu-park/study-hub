# Scalability Basics

**Previous**: [System Design Overview](./01_System_Design_Overview.md) | **Next**: [Network Fundamentals Review](./03_Network_Fundamentals_Review.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define scalability and explain why it is a critical concern for modern systems
2. Compare vertical scaling and horizontal scaling, identifying the trade-offs of each approach
3. Distinguish between stateless and stateful architectures and explain their impact on scaling
4. Describe common session management strategies (sticky sessions, centralized store, JWT)
5. Explain the CAP theorem and classify real-world systems as CP or AP
6. Describe the PACELC extension and how it captures latency-consistency trade-offs under normal operation
7. Estimate whether a given workload is better served by scaling up or scaling out

**Difficulty**: ⭐⭐
**Estimated Study Time**: 2-3 hours
**Prerequisites**: [01_System_Design_Overview.md](./01_System_Design_Overview.md)

---

A single server can only grow so large before hitting physical limits -- there is no CPU with infinite cores or a hard drive with infinite capacity. Scalability is the art of designing systems that grow gracefully as demand increases, whether that means serving ten users or ten million. Understanding the fundamental trade-offs between scaling strategies and distributed-system guarantees will shape every architecture decision you make from this point forward.

## Table of Contents

1. [What is Scalability?](#1-what-is-scalability)
2. [Vertical Scaling vs Horizontal Scaling](#2-vertical-scaling-vs-horizontal-scaling)
3. [Stateless vs Stateful](#3-stateless-vs-stateful)
4. [Session Management Methods](#4-session-management-methods)
5. [CAP Theorem](#5-cap-theorem)
6. [PACELC Theory](#6-pacelc-theory)
7. [Practice Problems](#7-practice-problems)
8. [References](#8-references)

---

## 1. What is Scalability?

### 1.1 Definition

Scalability is a system's ability to add resources to handle increasing load.

```
┌─────────────────────────────────────────────────────────────────┐
│                    What is Scalability?                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "The ability for a system to grow"                             │
│                                                                  │
│  Traffic Growth                                                 │
│  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐                        │
│  │1000│  │10K │  │100K│  │1M  │  │10M │   User Count           │
│  └──┬─┘  └──┬─┘  └──┬─┘  └──┬─┘  └──┬─┘                        │
│     │       │       │       │       │                           │
│     ▼       ▼       ▼       ▼       ▼                           │
│  ┌────────────────────────────────────────┐                     │
│  │      Can the system handle this?       │                     │
│  └────────────────────────────────────────┘                     │
│                                                                  │
│  Good Scalability = Linear performance improvement with         │
│                     resource addition                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Measuring Load

```
┌─────────────────────────────────────────────────────────────────┐
│                    Load Measurement Metrics                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Requests Per Second                                         │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ • Web server: HTTP requests/sec                        │  │
│     │ • Database: Queries/sec (QPS)                          │  │
│     │ • API: API calls/sec                                   │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
│  2. Concurrent Users                                            │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ • Number of simultaneously connected users             │  │
│     │ • Important for WebSocket, streaming services          │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
│  3. Data Volume                                                 │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ • Amount of data to store                              │  │
│     │ • Read/write ratio                                     │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
│  4. Complexity                                                  │
│     ┌────────────────────────────────────────────────────────┐  │
│     │ • Processing time per request                          │  │
│     │ • Number of dependencies                               │  │
│     └────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Vertical Scaling vs Horizontal Scaling

### 2.1 Vertical Scaling (Scale Up)

```
┌─────────────────────────────────────────────────────────────────┐
│                  Vertical Scaling (Scale Up)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Upgrade to more powerful hardware"                            │
│                                                                  │
│  Before                    After                                │
│  ┌────────────────┐        ┌────────────────┐                   │
│  │    Server      │        │    Server      │                   │
│  │  ┌──────────┐  │        │  ┌──────────┐  │                   │
│  │  │ CPU: 4cores│ │   ──▶  │  │ CPU: 32cores│                  │
│  │  │ RAM: 16GB │  │        │  │ RAM: 256GB│  │                   │
│  │  │ SSD: 500GB│  │        │  │ SSD: 4TB  │  │                   │
│  │  └──────────┘  │        │  └──────────┘  │                   │
│  └────────────────┘        └────────────────┘                   │
│                                                                  │
│  Pros:                                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Simple implementation (no code changes)                  │ │
│  │ • Easy to maintain data consistency                        │ │
│  │ • Simple management                                        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Cons:                                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Hardware limits exist (can't scale infinitely)           │ │
│  │ • Costs increase rapidly (2x performance ≠ 2x cost)        │ │
│  │ • Single point of failure (SPOF)                           │ │
│  │ • Downtime during upgrades                                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Horizontal Scaling (Scale Out)

```
┌─────────────────────────────────────────────────────────────────┐
│                 Horizontal Scaling (Scale Out)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Add more servers"                                             │
│                                                                  │
│  Before                    After                                │
│  ┌──────┐                  ┌──────┐ ┌──────┐ ┌──────┐          │
│  │Server│                  │Server│ │Server│ │Server│          │
│  │  1   │           ──▶    │  1   │ │  2   │ │  3   │          │
│  └──────┘                  └──────┘ └──────┘ └──────┘          │
│                                   │     │     │                 │
│                                   └─────┼─────┘                 │
│                                         │                       │
│                                  ┌──────────────┐               │
│                                  │Load Balancer │               │
│                                  └──────────────┘               │
│                                                                  │
│  Pros:                                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Theoretically infinite scaling                           │ │
│  │ • Cost effective (use commodity hardware)                  │ │
│  │ • High availability (failure → other servers handle)       │ │
│  │ • Zero-downtime scaling possible                           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Cons:                                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Increased architecture complexity                        │ │
│  │ • Difficult data consistency management                    │ │
│  │ • Session/state management required                        │ │
│  │ • Complex operations/monitoring                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Comparison Summary

| Item | Vertical Scaling | Horizontal Scaling |
|------|----------|----------|
| Method | More powerful server | More servers |
| Scaling Limit | Hardware limits | Theoretically infinite |
| Cost | Exponential increase | Linear increase |
| Complexity | Low | High |
| Availability | SPOF risk | High availability |
| Downtime | Required for upgrades | Zero-downtime possible |
| Suitable For | Early stage, small scale | Large scale, distributed |

### 2.4 Practical Application Examples

```
┌─────────────────────────────────────────────────────────────────┐
│                  Choosing Scaling Strategy                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Suitable for Vertical Scaling:                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Startup early stage                                      │ │
│  │ • Single database (RDB)                                    │ │
│  │ • Legacy systems with complex state management             │ │
│  │ • When architecture changes are difficult due to           │ │
│  │   cost/time constraints                                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Suitable for Horizontal Scaling:                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Large-scale traffic handling                             │ │
│  │ • Stateless web servers                                    │ │
│  │ • Microservices architecture                               │ │
│  │ • Cloud environments (Auto Scaling)                        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Hybrid Approach:                                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Web servers: Horizontal scaling                          │ │
│  │ • Database: Vertical scaling + Replication                 │ │
│  │ • Cache: Horizontal scaling (Redis Cluster)                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Stateless vs Stateful

### 3.1 Stateful Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Stateful Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Server stores client state"                                   │
│                                                                  │
│  User A ───▶ Server 1 (Stores session A)                        │
│  User B ───▶ Server 2 (Stores session B)                        │
│  User C ───▶ Server 1 (Stores session C)                        │
│                                                                  │
│  Problem scenarios:                                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  1. What if User A is routed to Server 2?                 │ │
│  │                                                            │ │
│  │  User A ───▶ Server 2  ──▶  "No session!" (Logged out)    │ │
│  │                                                            │ │
│  │  2. What if Server 1 goes down?                           │ │
│  │                                                            │ │
│  │  User A, C ───▶ ???  ──▶  "Session lost!" (Logged out)    │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Sticky Session needed:                                         │
│  ┌─────────────────┐                                            │
│  │  Load Balancer  │                                            │
│  │  (Sticky)       │                                            │
│  │  User A → S1    │                                            │
│  │  User B → S2    │                                            │
│  └─────────────────┘                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Stateless Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Stateless Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Server doesn't store state"                                   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  All state in external storage!                           │ │
│  │                                                            │ │
│  │              ┌─────────────┐                               │ │
│  │   Users ───▶ │ Load        │ ───▶ ┌───────┐               │ │
│  │              │ Balancer    │      │Server1│               │ │
│  │              └─────────────┘ ───▶ │Server2│               │ │
│  │                                ───▶│Server3│               │ │
│  │                                    └───┬───┘               │ │
│  │                                        │                   │ │
│  │                                        ▼                   │ │
│  │                              ┌────────────────┐            │ │
│  │                              │ Session Store  │            │ │
│  │                              │ (Redis/DB)     │            │ │
│  │                              └────────────────┘            │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Pros:                                                          │
│  • Same handling regardless of which server                     │
│  • Free to add/remove servers                                   │
│  • Server failures don't affect other users                     │
│  • Suitable for Auto Scaling                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Comparison

| Item | Stateful | Stateless |
|------|----------|-----------|
| State Storage | Server memory | External storage |
| Scalability | Limited | Excellent |
| Complexity | Low | External storage management needed |
| Failure Recovery | Session loss | Quick recovery |
| Load Balancing | Sticky Session needed | Free distribution |
| Suitable For | Simple systems | Large-scale distributed systems |

---

## 4. Session Management Methods

### 4.1 Session Management Options

```
┌─────────────────────────────────────────────────────────────────┐
│                   Session Management Methods                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Sticky Session (Session Affinity)                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   ┌─────────────┐                          │ │
│  │  User A ─────────▶│ LB: A→S1   │──────────▶ Server 1      │ │
│  │  (Cookie: S1)     │     B→S2   │                          │ │
│  │  User B ─────────▶│     C→S1   │──────────▶ Server 2      │ │
│  │  (Cookie: S2)     └─────────────┘                          │ │
│  │                                                            │ │
│  │  Pros: Simple implementation                               │ │
│  │  Cons: Load imbalance, session loss on server failure      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  2. Session Replication                                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  ┌──────────┐     ┌──────────┐     ┌──────────┐           │ │
│  │  │ Server 1 │◀───▶│ Server 2 │◀───▶│ Server 3 │           │ │
│  │  │ Ses A,B,C│     │ Ses A,B,C│     │ Ses A,B,C│           │ │
│  │  └──────────┘     └──────────┘     └──────────┘           │ │
│  │         ▲              ▲              ▲                    │ │
│  │         └──────────────┴──────────────┘                    │ │
│  │                  (Synchronization)                         │ │
│  │                                                            │ │
│  │  Pros: Easy failure recovery                               │ │
│  │  Cons: Sync overhead, limited scalability                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  3. Centralized Session Store                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  ┌──────────┐                                              │ │
│  │  │ Server 1 │──┐                                           │ │
│  │  └──────────┘  │      ┌───────────────┐                    │ │
│  │  ┌──────────┐  ├─────▶│ Redis/        │                    │ │
│  │  │ Server 2 │──┤      │ Memcached     │                    │ │
│  │  └──────────┘  │      │ (Sessions)    │                    │ │
│  │  ┌──────────┐  │      └───────────────┘                    │ │
│  │  │ Server 3 │──┘                                           │ │
│  │  └──────────┘                                              │ │
│  │                                                            │ │
│  │  Pros: Excellent scalability, stateless servers            │ │
│  │  Cons: Additional infrastructure, network latency          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  4. Client-Side Session (JWT)                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Client ────────▶ JWT Token (contains session info)        │ │
│  │    │                                                       │ │
│  │    └──▶ Send token with each request ──▶ Server verifies  │ │
│  │                                                            │ │
│  │  Pros: No server storage, best scalability                 │ │
│  │  Cons: Token size limit, difficult token invalidation      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Session Management Method Comparison

| Method | Scalability | Complexity | Failure Recovery | Recommended Situation |
|------|--------|--------|----------|----------|
| Sticky Session | Low | Low | Difficult | Small scale |
| Session Replication | Medium | High | Good | Medium scale |
| Centralized Store | High | Medium | Good | Large scale |
| JWT | Highest | Medium | N/A | API servers |

---

## 5. CAP Theorem

### 5.1 What is CAP Theorem?

```
┌─────────────────────────────────────────────────────────────────┐
│                       CAP Theorem                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Distributed systems cannot simultaneously satisfy all 3:       │
│                                                                  │
│                       Consistency                                │
│                           C                                      │
│                          /\                                      │
│                         /  \                                     │
│                        /    \                                    │
│                       /Cannot \                                  │
│                      /  have 3 \                                 │
│                     /____________\                               │
│                    A              P                              │
│              Availability    Partition                           │
│                             Tolerance                            │
│                                                                  │
│  Since network partitions (P) are unavoidable:                  │
│  In reality, must choose between C and A!                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Meaning of Each Property

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAP Properties Detail                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Consistency                                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ "All nodes see the same data"                              │ │
│  │                                                            │ │
│  │  Client ──▶ Write X=5 ──▶ Node A                           │ │
│  │                               │                             │ │
│  │                         (Synchronize)                       │ │
│  │                               ▼                             │ │
│  │  Client ──▶ Read X ───▶ Node B ──▶ X=5 (always)            │ │
│  │                                                            │ │
│  │  Example: Bank account balance                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Availability                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ "All requests receive a response" (success/failure)        │ │
│  │                                                            │ │
│  │  Client ──▶ Request ──▶ System ──▶ Response (always)       │ │
│  │                                                            │ │
│  │  Must respond even during failures                         │ │
│  │  Example: Social media feed                                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Partition Tolerance                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ "System operates despite network partitions"              │ │
│  │                                                            │ │
│  │  ┌──────┐          X          ┌──────┐                     │ │
│  │  │Node A│──────────X──────────│Node B│                     │ │
│  │  └──────┘   (Network split)    └──────┘                     │ │
│  │                                                            │ │
│  │  Service continues even when partitioned                   │ │
│  │  Can't give up P in distributed systems!                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 CP vs AP Choice

```
┌─────────────────────────────────────────────────────────────────┐
│                     CP vs AP Systems                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CP (Consistency + Partition Tolerance)                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  During network partition:                                 │ │
│  │  • Make some nodes unavailable to ensure consistency       │ │
│  │  • Sacrifice availability                                  │ │
│  │                                                            │ │
│  │  Examples:                                                 │ │
│  │  • Banking systems (balance accuracy critical)            │ │
│  │  • MongoDB, HBase, Redis (single)                          │ │
│  │  • Zookeeper, etcd                                         │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  AP (Availability + Partition Tolerance)                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  During network partition:                                 │ │
│  │  • All nodes can respond                                   │ │
│  │  • Some data may temporarily be inconsistent               │ │
│  │  • Eventual Consistency                                    │ │
│  │                                                            │ │
│  │  Examples:                                                 │ │
│  │  • Social media (like counts)                              │ │
│  │  • DNS                                                     │ │
│  │  • Cassandra, DynamoDB, CouchDB                            │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 Real System Classification

```
┌─────────────────────────────────────────────────────────────────┐
│                  CAP Classification by System                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│              Consistency                                        │
│                  ▲                                              │
│                  │                                              │
│     MongoDB ─────┼───── MySQL                                   │
│     HBase        │      PostgreSQL                              │
│     Redis        │      (Single node: CA)                       │
│     Zookeeper    │                                              │
│     etcd         │                                              │
│                  │                                              │
│                  │                                              │
│  ─────────────── ┼ ─────────────────────▶ Availability          │
│                  │                                              │
│                  │      Cassandra                               │
│                  │      DynamoDB                                │
│                  │      CouchDB                                 │
│                  │      Riak                                    │
│                  │                                              │
│                  │                                              │
│               Partition Tolerance                               │
│               (Required for distributed systems)                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. PACELC Theory

### 6.1 What is PACELC?

```
┌─────────────────────────────────────────────────────────────────┐
│                       PACELC Theory                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CAP extension: Also considers trade-offs during normal state   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  IF Partition:                                             │ │
│  │      Choose between Availability and Consistency           │ │
│  │      (P → A or C)                                          │ │
│  │                                                            │ │
│  │  ELSE (normal state):                                      │ │
│  │      Choose between Latency and Consistency                │ │
│  │      (E → L or C)                                          │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  PACELC = (P)artition → (A)vailability vs (C)onsistency        │
│           (E)lse → (L)atency vs (C)onsistency                   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Network partition?                                        │ │
│  │        │                                                   │ │
│  │    ┌───┴───┐                                               │ │
│  │    │       │                                               │ │
│  │   Yes      No                                              │ │
│  │    │       │                                               │ │
│  │    ▼       ▼                                               │ │
│  │  A vs C   L vs C                                           │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 PACELC System Classification

```
┌─────────────────────────────────────────────────────────────────┐
│                   PACELC System Classification                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PA/EL (Availability + Low latency preference)                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Cassandra (without Quorum)                               │ │
│  │ • DynamoDB                                                 │ │
│  │ Partition: maintain availability, Normal: fast response    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  PC/EC (Consistency priority)                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • MongoDB (default)                                        │ │
│  │ • HBase                                                    │ │
│  │ • Zookeeper                                                │ │
│  │ Partition: maintain consistency, Normal: maintain          │ │
│  │ consistency (accept latency)                               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  PA/EC (Availability + Consistency)                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • PNUTS (Yahoo)                                            │ │
│  │ Partition: maintain availability, Normal: maintain         │ │
│  │ consistency                                                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  PC/EL (Consistency + Low latency)                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Cassandra (with Quorum)                                  │ │
│  │ Partition: maintain consistency, Normal: fast response     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Selection Guide

| Requirement | Recommended | Example Systems |
|----------|------|------------|
| Data accuracy critical | PC/EC | Finance, inventory management |
| Fast response critical | PA/EL | Social media, games |
| Balanced | PA/EC or PC/EL | General web services |

---

## 7. Practice Problems

### Problem 1: Choose Scaling Method

Choose appropriate scaling method for these scenarios.

a) Early startup, 1,000 daily users
b) Large e-commerce, 10x traffic on Black Friday
c) Database handling complex analytics queries
d) CDN edge servers

### Problem 2: Stateless Design

Design shopping cart feature for stateless architecture. Where to store sessions and why?

### Problem 3: CAP Choice

Should you choose CP or AP for these services? Explain why.

a) Bank transfer system
b) Facebook like count
c) Online booking system
d) DNS server

### Problem 4: PACELC Analysis

Which PACELC combination would you choose for a system with these requirements?

Conditions:
- Global service (multiple regions)
- Read latency < 100ms required
- Data loss unacceptable

---

## Answers

### Problem 1 Answer

a) **Vertical scaling** - Small scale, so simple approach is efficient
b) **Horizontal scaling** - Elastic response with Auto Scaling
c) **Vertical scaling** - Analytics queries are efficient on single node
d) **Horizontal scaling** - Geographic distribution needed

### Problem 2 Answer

```
Recommendation: Redis (Centralized Session Store)

Reasons:
1. Stateless web servers - same handling regardless of server
2. Fast read/write - frequent cart lookups/updates
3. TTL support - automatic cart expiration
4. Easy failure recovery - Redis Sentinel/Cluster

Alternatives:
- JWT: Cart size limit, need new token per update
- DB: Possible but slower than Redis
```

### Problem 3 Answer

a) **CP** - Balance accuracy more important than availability
b) **AP** - Service must continue even if like counts differ temporarily
c) **CP** - Prevent double booking (strong consistency)
d) **AP** - Responding with stale info better than not responding

### Problem 4 Answer

```
Recommendation: PC/EL or PA/EC

Analysis:
- Global service → P required
- Read latency < 100ms → L (low latency) important
- Data loss unacceptable → C (consistency) important

PC/EL choice:
- Partition: maintain consistency (some requests may fail)
- Normal: fast reads

PA/EC choice:
- Partition: maintain availability + sync later
- Normal: strong consistency

Practical implementation:
- Reads: Local replicas (L)
- Writes: Synchronous replication (C)
- Cassandra + LOCAL_QUORUM
```

---

## Hands-On Exercises

### Exercise 1: Stateless Service Simulation

Design a simulation that demonstrates the difference between stateful and stateless architectures.

**Tasks:**
1. Implement a `StatefulServer` class that stores session data in local memory, and a `StatelessServer` class that reads from an external store (simulated dict)
2. Create a load balancer that routes requests randomly across 3 server instances
3. Demonstrate that stateful servers break when requests hit different instances, while stateless servers work correctly regardless of routing
4. Measure the "session loss rate" for stateful architecture under random routing

### Exercise 2: CAP Theorem Demonstration

Build a simple 3-node key-value store that lets you toggle between CP and AP behavior.

**Tasks:**
1. Create a `DistributedKV` class with 3 replica nodes
2. Implement a `partition(node)` method that isolates a node (simulates network partition)
3. In CP mode: writes/reads fail if a partition prevents quorum (2/3 nodes)
4. In AP mode: writes/reads succeed on available nodes, but reads may return stale data
5. Show a scenario where CP rejects a write and AP allows it but creates inconsistency

### Exercise 3: Scaling Cost Calculator

Build a tool that compares vertical vs. horizontal scaling costs for growing traffic.

**Tasks:**
1. Model vertical scaling: single server with cost doubling per capacity tier (e.g., 1x→2x→4x→8x CPU, costs $100→$250→$600→$1400)
2. Model horizontal scaling: N identical servers at $100 each + $50 load balancer overhead
3. Plot or print cost vs. capacity for both strategies from 1x to 16x traffic
4. Find the crossover point where horizontal becomes cheaper than vertical

---

## 8. References

### Books
- Designing Data-Intensive Applications - Martin Kleppmann (Chapter 5, 9)

### Papers
- "Brewer's Conjecture and the Feasibility of Consistent, Available, Partition-Tolerant Web Services" (2002)
- "Consistency Tradeoffs in Modern Distributed Database System Design" (PACELC)

### Online Resources
- [CAP Theorem Revisited](https://www.infoq.com/articles/cap-twelve-years-later-how-the-rules-have-changed/)
- [PACELC Wikipedia](https://en.wikipedia.org/wiki/PACELC_theorem)

---

**Previous**: [System Design Overview](./01_System_Design_Overview.md) | **Next**: [Network Fundamentals Review](./03_Network_Fundamentals_Review.md)

---

**Document Information**
- Last Updated: 2024
- Difficulty: ⭐⭐
- Estimated Study Time: 2-3 hours
