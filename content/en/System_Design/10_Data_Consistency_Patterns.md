# Data Consistency Patterns

**Previous**: [Database Replication](./09_Database_Replication.md) | **Next**: [Message Queue Basics](./11_Message_Queue_Basics.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define the spectrum of consistency models from strong consistency to eventual consistency and explain the trade-offs involved
2. Compare linearizability, sequential consistency, and causal consistency with concrete examples
3. Describe read consistency patterns (read-your-writes, monotonic reads, consistent prefix) and explain when each is needed
4. Explain why distributed transactions using two-phase commit (2PC) are problematic at scale
5. Design a Saga pattern implementation using both choreography and orchestration approaches, including compensating transactions
6. Select the appropriate consistency model for a given business requirement, balancing correctness against availability and latency

**Difficulty**: ⭐⭐⭐⭐

---

In a single-database world, consistency is straightforward -- every read sees the latest write. But the moment data lives on multiple nodes, you must make hard choices: do you block writes until all replicas agree, or do you let replicas diverge temporarily for the sake of speed and availability? These trade-offs shape every distributed system, from banking platforms that demand strict correctness to social media feeds where a few seconds of staleness is perfectly acceptable.

## Table of Contents

1. [Consistency Models Overview](#1-consistency-models-overview)
2. [Strong vs Eventual Consistency](#2-strong-vs-eventual-consistency)
3. [Read Consistency Patterns](#3-read-consistency-patterns)
4. [Distributed Transactions and 2PC](#4-distributed-transactions-and-2pc)
5. [Saga Pattern](#5-saga-pattern)
6. [Practical Application Guide](#6-practical-application-guide)
7. [Practice Problems](#7-practice-problems)

---

## 1. Consistency Models Overview

### CAP Theorem Review

```
┌─────────────────────────────────────────────────────────────┐
│                      CAP Theorem                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    Consistency (C)                          │
│                         /\                                  │
│                        /  \                                 │
│                       /    \                                │
│                      / CP   \                               │
│                     /        \                              │
│                    /──────────\                             │
│      Availability (A)        Partition Tolerance (P)        │
│                    \    AP    /                             │
│                     \        /                              │
│                      \──────/                               │
│                                                             │
│   When Network Partition Occurs:                           │
│   - Choose CP: Maintain consistency, sacrifice availability │
│   - Choose AP: Maintain availability, sacrifice consistency │
│                (Eventual)                                   │
└─────────────────────────────────────────────────────────────┘
```

### Consistency Spectrum

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Consistency Spectrum                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Strong                                              Eventual           │
│  Consistency          ◄────────────────────────►    Consistency         │
│                                                                         │
│  ├──────────┬──────────┬──────────┬──────────┬──────────┤             │
│  │          │          │          │          │          │             │
│  Lineariz-  Sequential Causal    Monotonic   Eventual                  │
│  ability    Consistency          Reads       Consistency               │
│                                                                         │
│  ◄───────── Strong Consistency ──────►◄──── Weak Consistency ────►     │
│  ◄─────── Low Availability/Performance ──►◄── High Availability/Perf ──►│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Strong vs Eventual Consistency

### Strong Consistency

All reads return the result of the most recent write.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Strong Consistency                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Time ──────────────────────────────────────────────────────►           │
│                                                                         │
│  Client A:  ──── Write(X=5) ─────────────────────────────►              │
│                      │                                                  │
│                      ▼                                                  │
│  Primary:   ════════[X=5]════════════════════════════════►              │
│                      │                                                  │
│                      │ Synchronous replication                          │
│                      │ (replication completes before write completes)   │
│                      ▼                                                  │
│  Replica:   ════════[X=5]════════════════════════════════►              │
│                      │                                                  │
│                      ▼                                                  │
│  Client B:  ──────── Read() ─────── returns X=5 ────────►               │
│                                                                         │
│  Characteristics:                                                       │
│  - All nodes see the same data                                          │
│  - Increased write latency                                              │
│  - Decreased availability (when replica fails)                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Eventual Consistency

Given sufficient time, all reads will return the same value.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Eventual Consistency                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Time ──────────────────────────────────────────────────────►           │
│                                                                         │
│  Client A:  ──── Write(X=5) ─────────────────────────────►              │
│                      │                                                  │
│                      ▼                                                  │
│  Primary:   ════════[X=5]════════════════════════════════►              │
│                      │                                                  │
│                      │ Asynchronous replication                         │
│                      │ (write completes immediately, replication later) │
│                      ▼                                                  │
│  Replica:   ═══[X=0]════════[X=5]════════════════════════►              │
│                 │              │                                        │
│                 │   replication lag │                                   │
│                 ▼              ▼                                        │
│  Client B:  ─── Read()=0 ──── Read()=5 ─────────────────►               │
│                 (stale)       (current)                                 │
│                                                                         │
│  Characteristics:                                                       │
│  - Temporary inconsistency allowed                                      │
│  - Reduced write latency                                                │
│  - High availability                                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Trade-off Comparison

| Characteristic | Strong Consistency | Eventual Consistency |
|------|-------------------|---------------------|
| Read consistency | Always latest | Temporary lag |
| Write latency | High | Low |
| Availability | Low | High |
| Implementation complexity | High | Low |
| Suitable use cases | Finance, inventory | SNS, like counts |

### Selection by Use Case

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Consistency Selection by Use Case                   │
├───────────────────────────────┬─────────────────────────────────────────┤
│  Strong Consistency Required  │  Eventual Consistency Suitable          │
├───────────────────────────────┼─────────────────────────────────────────┤
│  - Bank account balance       │  - SNS timeline                         │
│  - Inventory quantity (prevent│  - Like/follower counts                 │
│    overselling)               │  - Search index                          │
│  - Reservation systems        │  - Log collection                        │
│  - Payment processing         │  - View counters                         │
│  - User authentication state  │  - Recommendation systems                │
│  - Distributed locks          │                                          │
└───────────────────────────────┴─────────────────────────────────────────┘
```

---

## 3. Read Consistency Patterns

### Read-Your-Writes

Guarantees that users can always read the data they wrote.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Read-Your-Writes                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Problem Scenario:                                                      │
│  ┌──────────┐         ┌──────────┐         ┌──────────┐                │
│  │  User A  │         │  Primary │         │ Replica  │                │
│  └────┬─────┘         └────┬─────┘         └────┬─────┘                │
│       │                    │                    │                       │
│       │──Write(name=Bob)──►│                    │                       │
│       │◄── Success ────────│                    │                       │
│       │                    │── async replicate─►│                       │
│       │──Read(name)────────────────────────────►│ (not replicated yet)  │
│       │◄─────────────────── name=Alice ─────────│                       │
│       │                                         │                       │
│  "I just changed it to Bob, why is it Alice?"                           │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Solution 1: Read from Primary                                          │
│  ┌──────────┐         ┌──────────┐                                     │
│  │  User A  │         │  Primary │                                     │
│  └────┬─────┘         └────┬─────┘                                     │
│       │                    │                                            │
│       │──Write(name=Bob)──►│                                            │
│       │◄── Success ────────│                                            │
│       │──Read(name)───────►│ ← After own write, read from Primary      │
│       │◄─── name=Bob ──────│                                            │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Solution 2: Timestamp-based                                            │
│                                                                         │
│  - Record timestamp T on write                                          │
│  - On read, only read from replicas with data after T                   │
│  - Use only if replica's replication_timestamp >= T                     │
│                                                                         │
│  Write Response: { success: true, timestamp: T1 }                       │
│  Read Request:   { key: "name", min_timestamp: T1 }                     │
│  → Read only from replicas updated after T1                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

```python
# Read-Your-Writes implementation example
class ReadYourWritesClient:
    def __init__(self, primary, replicas):
        self.primary = primary
        self.replicas = replicas
        self.last_write_timestamp = {}  # key -> timestamp

    def write(self, key, value):
        timestamp = self.primary.write(key, value)
        self.last_write_timestamp[key] = timestamp
        return timestamp

    def read(self, key):
        if key in self.last_write_timestamp:
            # Read recently written keys from Primary
            return self.primary.read(key)
        else:
            # Read unwritten keys from Replica
            return self.select_replica().read(key)
```

### Monotonic Reads

Guarantees that once a value is read, older values will not be read subsequently.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Monotonic Reads                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Problem Scenario (Monotonic Reads Violation):                         │
│                                                                         │
│  Time ──────────────────────────────────────────────────────►           │
│                                                                         │
│  Replica 1: [v1]───────[v2]───────[v3]────────────────────►             │
│  Replica 2: [v1]───────────────────────[v2]───────────────►             │
│                                                                         │
│  User:      Read@R1=v2  Read@R2=v1  Read@R1=v3                          │
│                    │         │                                          │
│                    └─────────┘                                          │
│                    "Feels like time went backwards"                     │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Solution: Sticky Session                                               │
│                                                                         │
│  ┌──────────┐         ┌──────────┐                                     │
│  │  User A  │─────────│ Replica 1│                                     │
│  └──────────┘  fixed  └──────────┘                                     │
│                                                                         │
│  ┌──────────┐         ┌──────────┐                                     │
│  │  User B  │─────────│ Replica 2│                                     │
│  └──────────┘  fixed  └──────────┘                                     │
│                                                                         │
│  - Same user always reads from same replica                             │
│  - Switch to different replica if current one goes down                 │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Solution: Version-based                                                │
│                                                                         │
│  Read Response: { value: "v2", version: 42 }                            │
│  Next Read Request: { key: "data", min_version: 42 }                    │
│  → Read only from replicas with version >= 42                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Monotonic Writes

Guarantees that writes from the same session are applied in order.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Monotonic Writes                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Problem Scenario:                                                      │
│                                                                         │
│  User: Write(X=1) → Write(X=2) → Write(X=3)                            │
│                                                                         │
│  Due to network latency:                                                │
│  Replica receive order: X=2 → X=3 → X=1                                │
│  Final result: X=1 (not as intended!)                                   │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Solution: Version/Timestamp-based                                      │
│                                                                         │
│  Write(X=1, ts=100) → Write(X=2, ts=101) → Write(X=3, ts=102)          │
│                                                                         │
│  Receive order: ts=101 → ts=102 → ts=100                               │
│  Apply: X=2 applied → X=3 applied → X=1 ignored (ts=100 < current=102) │
│  Final result: X=3 (correct!)                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Distributed Transactions and 2PC

### Two-Phase Commit (2PC)

A protocol that guarantees atomicity of distributed transactions.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Two-Phase Commit (2PC)                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Phase 1: Prepare (Voting phase)                                        │
│  ┌─────────────┐                                                       │
│  │ Coordinator │                                                       │
│  └──────┬──────┘                                                       │
│         │                                                               │
│         │─── PREPARE ───►┌────────────┐                                │
│         │                │Participant1│───► Prepare transaction         │
│         │◄── VOTE YES ───└────────────┘     (acquire lock, write log)  │
│         │                                                               │
│         │─── PREPARE ───►┌────────────┐                                │
│         │                │Participant2│───► Prepare transaction         │
│         │◄── VOTE YES ───└────────────┘                                │
│         │                                                               │
│  Phase 2: Commit (Decision phase)                                       │
│         │                                                               │
│         │─── COMMIT ────►┌────────────┐                                │
│         │                │Participant1│───► Execute commit              │
│         │◄─── ACK ───────└────────────┘                                │
│         │                                                               │
│         │─── COMMIT ────►┌────────────┐                                │
│         │                │Participant2│───► Execute commit              │
│         │◄─── ACK ───────└────────────┘                                │
│         │                                                               │
│         ▼                                                               │
│  Transaction Complete                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2PC Failure Scenarios

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     2PC Failure Scenarios                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Scenario 1: Participant votes NO                                       │
│  ┌─────────────┐                                                       │
│  │ Coordinator │                                                       │
│  └──────┬──────┘                                                       │
│         │─── PREPARE ───►│Participant1│◄── VOTE YES                    │
│         │─── PREPARE ───►│Participant2│◄── VOTE NO (failed)            │
│         │                                                               │
│         │─── ROLLBACK ──►│Participant1│                                │
│         │─── ROLLBACK ──►│Participant2│                                │
│         ▼                                                               │
│  Transaction Aborted                                                    │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Scenario 2: Coordinator Failure (Blocking problem)                     │
│                                                                         │
│  ┌─────────────┐                                                       │
│  │ Coordinator │──X (failure)                                          │
│  └──────┬──────┘                                                       │
│         │─── PREPARE ───►│Participant1│◄── VOTE YES (waiting...)       │
│         │─── PREPARE ───►│Participant2│◄── VOTE YES (waiting...)       │
│         │                                                               │
│         X  Coordinator down!                                            │
│                                                                         │
│         │Participant1│: "Commit? Rollback? Cannot decide..."           │
│         │Participant2│: "Resources remain locked..."                   │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────    │
│  This is the biggest problem with 2PC: BLOCKING                         │
│  Participants wait indefinitely until Coordinator recovers              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Limitations of 2PC

| Limitation | Description |
|------|------|
| Blocking | Entire system blocks when Coordinator fails |
| Performance | 2 network round trips, all participants wait |
| Availability | Entire transaction rolls back if one participant fails |
| Scalability | Performance degrades rapidly as participants increase |

### 3PC (Three-Phase Commit)

An attempt to mitigate the blocking problem of 2PC:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     3PC vs 2PC                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  2PC:                          3PC:                                     │
│  1. Prepare                    1. CanCommit (voting)                    │
│  2. Commit/Rollback            2. PreCommit (prepare confirmation)      │
│                                3. DoCommit (actual commit)               │
│                                                                         │
│  3PC Advantages:                                                        │
│  - Participants share state in PreCommit phase                          │
│  - Consensus among participants possible even when Coordinator fails    │
│                                                                         │
│  3PC Limitations:                                                       │
│  - Still possible inconsistency during network partition                │
│  - Increased complexity, rarely used in practice                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Saga Pattern

A distributed transaction pattern to overcome the limitations of 2PC.

### Saga Basic Concept

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Saga Pattern Overview                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  2PC:  One large atomic transaction                                     │
│        [─────────── Entire Transaction ───────────]                     │
│        Rollback entire transaction on failure (locks held)              │
│                                                                         │
│  Saga: Sequence of local transactions                                   │
│        [T1] → [T2] → [T3] → [T4]                                       │
│        Execute compensating transactions on failure                     │
│        [T1] → [T2] → [T3-fail] → [C2] → [C1]                           │
│                                     Compensating transactions           │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────      │
│                                                                         │
│  Example: Travel Booking Saga                                           │
│                                                                         │
│  Normal flow:                                                           │
│  [Book Flight] → [Book Hotel] → [Book Car] → [Payment]                 │
│       T1             T2            T3          T4                       │
│                                                                         │
│  T3 fails:                                                              │
│  [Book Flight] → [Book Hotel] → [Car-fail] → [Cancel Hotel] → [Cancel Flight]│
│       T1             T2            T3            C2             C1      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Choreography

Each service autonomously operates by publishing and subscribing to events.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Choreography Saga                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│  │  Order  │    │ Payment │    │  Stock  │    │Shipping │             │
│  │ Service │    │ Service │    │ Service │    │ Service │             │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘             │
│       │              │              │              │                    │
│  ═════╪══════════════╪══════════════╪══════════════╪═══  Event Bus     │
│       │              │              │              │                    │
│  1.OrderCreated      │              │              │                    │
│  ─────┼─────────────►│              │              │                    │
│       │         2.PaymentCompleted  │              │                    │
│       │         ─────┼─────────────►│              │                    │
│       │              │       3.StockReserved       │                    │
│       │              │       ───────┼─────────────►│                    │
│       │              │              │    4.ShippingScheduled            │
│       │              │              │    ──────────┼────►               │
│       │◄─────────────┼──────────────┼──────────────┼──── 5.OrderComplete│
│       │              │              │              │                    │
│  ─────────────────────────────────────────────────────────────────────  │
│  Compensating events on failure:                                        │
│                                                                         │
│  1.OrderCreated ──► 2.PaymentCompleted ──► 3.StockFailed!              │
│                                                 │                       │
│                     PaymentRefunded ◄──────────┘                       │
│                          │                                              │
│       OrderCancelled ◄───┘                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Choreography Pros and Cons

| Pros | Cons |
|------|------|
| Loose coupling | Difficult to understand overall flow |
| No single point of failure | Risk of circular dependencies |
| Services scale independently | Complex debugging |
| Simple implementation | Difficult testing |

### Orchestration

A central Orchestrator controls the entire flow.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Orchestration Saga                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                      ┌─────────────────┐                               │
│                      │   Orchestrator  │                               │
│                      │ (Saga Manager)  │                               │
│                      └────────┬────────┘                               │
│                               │                                         │
│         ┌─────────────────────┼─────────────────────┐                  │
│         │                     │                     │                  │
│         ▼                     ▼                     ▼                  │
│  ┌────────────┐       ┌────────────┐       ┌────────────┐             │
│  │   Order    │       │  Payment   │       │   Stock    │             │
│  │  Service   │       │  Service   │       │  Service   │             │
│  └────────────┘       └────────────┘       └────────────┘             │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Normal flow:                                                           │
│                                                                         │
│  Orchestrator                                                          │
│       │                                                                 │
│       │──── 1. CreateOrder ────►│Order│                                │
│       │◄─── OrderCreated ───────│     │                                │
│       │                                                                 │
│       │──── 2. ProcessPayment ─►│Payment│                              │
│       │◄─── PaymentDone ────────│      │                               │
│       │                                                                 │
│       │──── 3. ReserveStock ───►│Stock│                                │
│       │◄─── StockReserved ──────│     │                                │
│       │                                                                 │
│       │──── 4. CompleteOrder ──►│Order│                                │
│       ▼                                                                 │
│  Saga Complete                                                         │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Compensation on failure:                                               │
│                                                                         │
│  Orchestrator                                                          │
│       │                                                                 │
│       │──── 1. CreateOrder ────►│Order│ ✓                              │
│       │──── 2. ProcessPayment ─►│Payment│ ✓                            │
│       │──── 3. ReserveStock ───►│Stock│ ✗ (failed!)                    │
│       │                                                                 │
│       │──── 4. RefundPayment ──►│Payment│ (compensation)               │
│       │──── 5. CancelOrder ────►│Order│ (compensation)                 │
│       ▼                                                                 │
│  Saga Rolled Back                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Orchestration Pros and Cons

| Pros | Cons |
|------|------|
| Centralized flow management | Single point of failure |
| Clear workflow | Increased orchestrator complexity |
| Easy debugging/monitoring | Increased coupling |
| Easy testing | Orchestrator scalability |

### Saga Pattern Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Choreography vs Orchestration Selection Criteria           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Choose Choreography:                                                   │
│  - When few services (2-4)                                              │
│  - Simple workflows                                                     │
│  - When service independence is important                               │
│  - Want to avoid strong coupling between teams                          │
│                                                                         │
│  Choose Orchestration:                                                  │
│  - Complex business logic                                               │
│  - When many services (5+)                                              │
│  - When clear error handling is needed                                  │
│  - When workflow visibility is important                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Saga Implementation Example (Orchestration)

```python
class OrderSaga:
    """Order Saga Orchestrator"""

    def __init__(self, order_id):
        self.order_id = order_id
        self.state = "STARTED"
        self.compensations = []  # Compensation transaction stack

    def execute(self):
        try:
            # Step 1: Create order
            self.create_order()
            self.compensations.append(self.cancel_order)

            # Step 2: Process payment
            self.process_payment()
            self.compensations.append(self.refund_payment)

            # Step 3: Reserve stock
            self.reserve_stock()
            self.compensations.append(self.release_stock)

            # Step 4: Schedule shipping
            self.schedule_shipping()

            self.state = "COMPLETED"

        except SagaException as e:
            self.compensate()
            self.state = "COMPENSATED"
            raise

    def compensate(self):
        """Execute compensating transactions in reverse order"""
        while self.compensations:
            compensation = self.compensations.pop()
            try:
                compensation()
            except Exception as e:
                # Log compensation failure, add to retry queue
                log_compensation_failure(self.order_id, compensation, e)

    def create_order(self):
        # Call Order Service
        pass

    def cancel_order(self):
        # Order Service - cancel order
        pass

    # ... other methods
```

---

## 6. Practical Application Guide

### Consistency Pattern Selection Checklist

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Consistency Pattern Selection Guide                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Question 1: What is the impact of data inconsistency on business?      │
│  ├── Critical (financial loss, legal issues) ──► Strong Consistency    │
│  └── Acceptable (temporarily stale info) ──► Eventual Consistency      │
│                                                                         │
│  Question 2: What are the main system requirements?                     │
│  ├── Low latency ──► Eventual Consistency                              │
│  └── Data accuracy ──► Strong Consistency                              │
│                                                                         │
│  Question 3: What should system do on failure?                          │
│  ├── Allow partial functionality outage ──► Strong Consistency (CP)    │
│  └── Always need to respond ──► Eventual Consistency (AP)              │
│                                                                         │
│  Question 4: What is the read/write ratio?                              │
│  ├── Read-heavy (90%+) ──► Eventual + Read Replica                     │
│  └── Write-heavy ──► Careful! Consider replication lag                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Hybrid Approach

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Hybrid Consistency Strategy                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Apply different consistency levels per data type within same system    │
│                                                                         │
│  Example: E-commerce system                                             │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │  Data Type           │  Consistency Level │  Reason           │     │
│  ├──────────────────────┼────────────────────┼──────────────────┤     │
│  │  Inventory quantity  │  Strong            │  Prevent oversell │     │
│  │  Payment status      │  Strong            │  Money accuracy   │     │
│  │  Product reviews     │  Eventual          │  No immediate need│     │
│  │  View counts         │  Eventual          │  Less important   │     │
│  │  Shopping cart       │  Session           │  Per-user consist │     │
│  │  Recommended products│  Eventual          │  Not real-time    │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Practice Problems

### Exercise 1: Analyze Consistency Requirements

Choose an appropriate consistency model for the following scenarios and explain why:

1. Bank account balance
2. Twitter follower count
3. Online game ranking
4. Airline seat reservation
5. News article view count

### Exercise 2: Saga Design

Design an order process for an online shopping mall using the Saga pattern:
- Include steps: order creation, inventory check, payment, shipping reservation
- Define compensating transactions for each step
- Design both Choreography and Orchestration versions

### Exercise 3: Read Consistency Implementation

Design a client library that guarantees Read-Your-Writes:
- Guarantee latest data on read after write
- Use timestamp-based approach
- Include caching strategy

---

## References

- "Designing Data-Intensive Applications" - Martin Kleppmann
- "Microservices Patterns" - Chris Richardson
- Google Cloud Spanner: TrueTime and External Consistency
- AWS: Building Distributed Locks with DynamoDB
- "Sagas" - Hector Garcia-Molina, Kenneth Salem (1987)

---

**Previous**: [Database Replication](./09_Database_Replication.md) | **Next**: [Message Queue Basics](./11_Message_Queue_Basics.md)
