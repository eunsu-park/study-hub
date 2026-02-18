# Distributed Databases

**Previous**: [13. NoSQL Data Models](./13_NoSQL_Data_Models.md) | **Next**: [15. NewSQL and Modern Trends](./15_NewSQL_and_Modern_Trends.md)

---

Modern applications serve users across the globe, generate massive volumes of data, and must remain available even when hardware fails. A single database server, no matter how powerful, cannot meet all three demands simultaneously. Distributed databases solve this by spreading data and computation across multiple machines. This lesson examines the architectures, algorithms, and tradeoffs that underpin every distributed database system, from data fragmentation and replication to consensus protocols and distributed transactions.

**Difficulty**: ⭐⭐⭐⭐

**Learning Objectives**:
- Compare shared-nothing, shared-disk, and shared-memory architectures
- Design horizontal, vertical, and hybrid fragmentation strategies
- Analyze synchronous vs asynchronous replication and quorum-based protocols
- Explain distributed query processing and optimization
- Trace through the Two-Phase Commit (2PC) and Three-Phase Commit (3PC) protocols
- Describe the Paxos and Raft consensus algorithms at a conceptual level
- Apply distributed concurrency control techniques
- Select appropriate partitioning strategies (range, hash, consistent hashing)
- Reason about CAP theorem implications in distributed system design

---

## Table of Contents

1. [Distributed Database Architecture](#1-distributed-database-architecture)
2. [Data Fragmentation](#2-data-fragmentation)
3. [Data Replication](#3-data-replication)
4. [Distributed Query Processing](#4-distributed-query-processing)
5. [Distributed Transactions: Two-Phase Commit (2PC)](#5-distributed-transactions-two-phase-commit-2pc)
6. [Three-Phase Commit (3PC)](#6-three-phase-commit-3pc)
7. [Consensus Algorithms: Paxos and Raft](#7-consensus-algorithms-paxos-and-raft)
8. [Distributed Concurrency Control](#8-distributed-concurrency-control)
9. [CAP Theorem Implications for Distributed Design](#9-cap-theorem-implications-for-distributed-design)
10. [Partitioning Strategies](#10-partitioning-strategies)
11. [Exercises](#11-exercises)
12. [References](#12-references)

---

## 1. Distributed Database Architecture

A distributed database is a collection of logically interrelated databases distributed over a computer network. There are three fundamental architectures, distinguished by how they share resources.

### 1.1 Shared-Nothing Architecture

In a shared-nothing architecture, each node has its own CPU, memory, and storage. Nodes communicate only through the network.

```
┌─────────────────────────────────────────────────────────────┐
│  Shared-Nothing Architecture                                │
│                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                │
│  │  Node 1 │    │  Node 2 │    │  Node 3 │                │
│  │ ┌─────┐ │    │ ┌─────┐ │    │ ┌─────┐ │                │
│  │ │ CPU │ │    │ │ CPU │ │    │ │ CPU │ │                │
│  │ ├─────┤ │    │ ├─────┤ │    │ ├─────┤ │                │
│  │ │ RAM │ │    │ │ RAM │ │    │ │ RAM │ │                │
│  │ ├─────┤ │    │ ├─────┤ │    │ ├─────┤ │                │
│  │ │Disk │ │    │ │Disk │ │    │ │Disk │ │                │
│  │ └─────┘ │    │ └─────┘ │    │ └─────┘ │                │
│  └────┬────┘    └────┬────┘    └────┬────┘                │
│       │              │              │                      │
│       └──────────────┼──────────────┘                      │
│                      │                                      │
│              ┌───────┴───────┐                              │
│              │   Network     │                              │
│              └───────────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

**Characteristics**:
- Each node operates independently
- Data is partitioned across nodes
- Scales linearly by adding nodes
- No single point of failure (if designed correctly)
- Network is the bottleneck

**Examples**: Cassandra, CockroachDB, Citus (PostgreSQL extension), Google Spanner, TiDB

**Advantages**:
- Near-linear horizontal scalability
- No resource contention between nodes
- Cost-effective (commodity hardware)

**Disadvantages**:
- Cross-node queries require network communication
- Distributed transactions are expensive
- Rebalancing data when adding/removing nodes is complex

### 1.2 Shared-Disk Architecture

In a shared-disk architecture, each node has its own CPU and memory, but all nodes access a shared storage layer (typically a SAN or distributed filesystem).

```
┌─────────────────────────────────────────────────────────────┐
│  Shared-Disk Architecture                                   │
│                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                │
│  │  Node 1 │    │  Node 2 │    │  Node 3 │                │
│  │ ┌─────┐ │    │ ┌─────┐ │    │ ┌─────┐ │                │
│  │ │ CPU │ │    │ │ CPU │ │    │ │ CPU │ │                │
│  │ ├─────┤ │    │ ├─────┤ │    │ ├─────┤ │                │
│  │ │ RAM │ │    │ │ RAM │ │    │ │ RAM │ │                │
│  │ └─────┘ │    │ └─────┘ │    │ └─────┘ │                │
│  └────┬────┘    └────┬────┘    └────┬────┘                │
│       │              │              │                      │
│       └──────────────┼──────────────┘                      │
│                      │                                      │
│              ┌───────┴───────┐                              │
│              │  Shared Disk  │                              │
│              │   (SAN/NAS)   │                              │
│              │  ┌─────────┐  │                              │
│              │  │  Disk   │  │                              │
│              │  │  Array  │  │                              │
│              │  └─────────┘  │                              │
│              └───────────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

**Characteristics**:
- All nodes see the same data (no partitioning needed for storage)
- Requires distributed lock management (DLM) to coordinate writes
- Storage layer must be highly available and performant
- Adding compute nodes is easy; storage is the scaling bottleneck

**Examples**: Oracle RAC, Amazon Aurora, Neon

**Advantages**:
- Simpler data management (no partitioning logic)
- Easy to add compute nodes
- Storage layer handles durability and replication

**Disadvantages**:
- Shared storage can become a bottleneck
- Distributed lock management adds complexity
- Storage layer is a potential single point of failure (mitigated by redundant SANs)

### 1.3 Shared-Memory Architecture (SMP)

In a shared-memory (symmetric multiprocessing) architecture, all processors share the same memory and disk.

```
┌─────────────────────────────────────────────────────────────┐
│  Shared-Memory Architecture (SMP)                           │
│                                                             │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                   │
│  │ CPU1 │  │ CPU2 │  │ CPU3 │  │ CPU4 │                   │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘                   │
│     │         │         │         │                        │
│     └─────────┴────┬────┴─────────┘                        │
│                    │                                        │
│            ┌───────┴───────┐                                │
│            │  Shared RAM   │                                │
│            │  (Shared Bus) │                                │
│            └───────┬───────┘                                │
│                    │                                        │
│            ┌───────┴───────┐                                │
│            │  Shared Disk  │                                │
│            └───────────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

**Characteristics**:
- Simplest programming model (all data accessible to all CPUs)
- Limited by memory bus bandwidth
- Cannot scale beyond a single machine's limits
- Not truly "distributed" -- a single large server

**Examples**: Traditional enterprise database servers (single-node PostgreSQL, Oracle, SQL Server)

**Advantages**:
- Fastest inter-processor communication (shared memory)
- No network overhead for data access
- Simple transaction management

**Disadvantages**:
- Vertical scaling only
- Memory bus becomes a bottleneck
- Single point of failure

### 1.4 Architecture Comparison

| Feature | Shared-Nothing | Shared-Disk | Shared-Memory |
|---------|---------------|-------------|---------------|
| **Scalability** | Excellent (horizontal) | Good (compute) / Limited (storage) | Limited (vertical) |
| **Complexity** | High (partitioning, distributed txns) | Medium (DLM) | Low |
| **Fault tolerance** | High | Medium (depends on storage) | Low |
| **Cost** | Low (commodity HW) | Medium-High (SAN) | High (big servers) |
| **Network dependency** | High | High (for storage) | None |
| **Data locality** | Yes (co-located) | No (network to storage) | Yes (shared memory) |

---

## 2. Data Fragmentation

Data fragmentation (also called partitioning or sharding) is the process of dividing a relation (table) into smaller pieces and distributing them across nodes.

### 2.1 Horizontal Fragmentation

Horizontal fragmentation divides a table by rows. Each fragment contains a subset of tuples that satisfy some predicate.

```
Original Table: employees
┌─────┬────────┬────────────┬────────┐
│ id  │ name   │ department │ salary │
├─────┼────────┼────────────┼────────┤
│ 1   │ Alice  │ Engineering│ 95000  │
│ 2   │ Bob    │ Marketing  │ 72000  │
│ 3   │ Charlie│ Engineering│ 88000  │
│ 4   │ Diana  │ Marketing  │ 68000  │
│ 5   │ Eve    │ Engineering│ 92000  │
│ 6   │ Frank  │ Marketing  │ 75000  │
└─────┴────────┴────────────┴────────┘

Horizontal Fragmentation by department:

Fragment 1 (Node A):                Fragment 2 (Node B):
σ(department='Engineering')         σ(department='Marketing')
┌─────┬────────┬────────────┬──────┐ ┌─────┬───────┬────────────┬──────┐
│ 1   │ Alice  │ Engineering│95000 │ │ 2   │ Bob   │ Marketing  │72000 │
│ 3   │ Charlie│ Engineering│88000 │ │ 4   │ Diana │ Marketing  │68000 │
│ 5   │ Eve    │ Engineering│92000 │ │ 6   │ Frank │ Marketing  │75000 │
└─────┴────────┴────────────┴──────┘ └─────┴───────┴────────────┴──────┘
```

**Correctness conditions**:

1. **Completeness**: Every tuple in the original relation must appear in at least one fragment.
   - `R = F1 ∪ F2 ∪ ... ∪ Fn`

2. **Reconstruction**: The original relation must be recoverable from the fragments.
   - `R = F1 ∪ F2 ∪ ... ∪ Fn` (union for horizontal fragmentation)

3. **Disjointness** (optional but desirable): Each tuple appears in exactly one fragment. This avoids redundancy.
   - `Fi ∩ Fj = ∅` for all `i ≠ j`

**Types of horizontal fragmentation**:

- **Primary horizontal fragmentation**: Based on predicates on the relation's own attributes.
  - Example: `σ(salary > 80000)(employees)` and `σ(salary ≤ 80000)(employees)`

- **Derived horizontal fragmentation**: Based on predicates on a related relation.
  - Example: Fragment `projects` based on which department owns them, aligning with the `employees` fragmentation.

### 2.2 Vertical Fragmentation

Vertical fragmentation divides a table by columns. Each fragment contains a subset of attributes, plus the primary key (to enable reconstruction).

```
Original Table: employees
┌─────┬────────┬────────────┬────────┬─────────────┬───────┐
│ id  │ name   │ department │ salary │ ssn         │ email │
└─────┴────────┴────────────┴────────┴─────────────┴───────┘

Vertical Fragmentation:

Fragment 1 (Node A):              Fragment 2 (Node B):
Public info                       Sensitive info
┌─────┬────────┬────────────┐     ┌─────┬────────┬─────────────┐
│ id  │ name   │ department │     │ id  │ salary │ ssn         │
├─────┼────────┼────────────┤     ├─────┼────────┼─────────────┤
│ 1   │ Alice  │ Engineering│     │ 1   │ 95000  │ 123-45-6789 │
│ 2   │ Bob    │ Marketing  │     │ 2   │ 72000  │ 234-56-7890 │
│ ... │ ...    │ ...        │     │ ... │ ...    │ ...         │
└─────┴────────┴────────────┘     └─────┴────────┴─────────────┘

Reconstruction: π_{id,name,department}(F1) ⋈ π_{id,salary,ssn}(F2) = R
```

**Correctness conditions**:
1. **Completeness**: Every attribute appears in at least one fragment.
2. **Reconstruction**: `R = F1 ⋈ F2 ⋈ ... ⋈ Fn` (natural join on primary key).
3. **Disjointness**: Attributes (other than the primary key) do not overlap.

**Use cases**:
- **Security**: Sensitive columns (salary, SSN) on a separate, more secure node
- **Performance**: Frequently accessed columns together; rarely accessed columns separate
- **Mixed workloads**: OLTP queries access a few columns; OLAP queries access many columns

### 2.3 Hybrid Fragmentation

Hybrid fragmentation combines horizontal and vertical fragmentation. First fragment vertically, then horizontally (or vice versa).

```
Step 1: Vertical fragmentation
   R → V1(id, name, dept) + V2(id, salary, ssn)

Step 2: Horizontal fragmentation of V1
   V1 → H1(dept='Eng') + H2(dept='Mktg')

Result: 3 fragments
   F1 = H1 (id, name, dept) where dept='Eng'     → Node A
   F2 = H2 (id, name, dept) where dept='Mktg'    → Node B
   F3 = V2 (id, salary, ssn)                      → Node C (secure)
```

### 2.4 Fragmentation Transparency

Ideally, the distributed database provides **fragmentation transparency**: users write queries against the global schema, and the system automatically routes subqueries to the appropriate fragments.

```
Transparency Levels:

Level 1: Fragmentation Transparency
  User sees: "SELECT * FROM employees WHERE dept = 'Engineering'"
  System handles: routing to the correct fragment(s)

Level 2: Location Transparency
  User knows fragments exist but not where they are stored
  User writes: "SELECT * FROM employees_eng"
  System handles: finding the node that stores employees_eng

Level 3: No Transparency
  User must specify both fragment and location
  "SELECT * FROM node_a.employees_eng"
```

---

## 3. Data Replication

Replication maintains copies of data at multiple nodes to improve availability, fault tolerance, and read performance.

### 3.1 Replication Topologies

```
Single-Leader (Master-Slave):
┌──────────┐     ┌──────────┐
│  Leader   │────▶│ Follower │  (writes go to leader,
│  (R/W)   │     │  (Read)  │   reads from followers)
└──────────┘     └──────────┘
     │
     └──────────▶┌──────────┐
                 │ Follower │
                 │  (Read)  │
                 └──────────┘

Multi-Leader:
┌──────────┐◀───▶┌──────────┐  (writes accepted at
│ Leader 1 │     │ Leader 2 │   any leader, replicated
│  (R/W)   │     │  (R/W)   │   bidirectionally)
└──────────┘     └──────────┘

Leaderless:
┌──────────┐     ┌──────────┐  (writes sent to all
│  Node 1  │◀───▶│  Node 2  │   replicas; reads from
│  (R/W)   │     │  (R/W)   │   multiple replicas)
└──────────┘     └──────────┘
     ▲                ▲
     │                │
     └───▶┌──────────┐│
          │  Node 3  │┘
          │  (R/W)   │
          └──────────┘
```

### 3.2 Synchronous vs Asynchronous Replication

**Synchronous replication**: The leader waits for all (or a defined subset of) followers to acknowledge the write before confirming it to the client.

```
Client        Leader        Follower 1     Follower 2
  │             │               │              │
  │──WRITE──▶  │               │              │
  │             │──replicate──▶│              │
  │             │──replicate──────────────────▶│
  │             │               │              │
  │             │◀──ACK────────│              │
  │             │◀──ACK────────────────────────│
  │◀──OK───────│               │              │
  │             │               │              │

Timeline: Client waits until ALL followers acknowledge
Guarantee: After OK, data is durable on all replicas
Risk: One slow follower blocks the entire write
```

**Asynchronous replication**: The leader confirms the write to the client immediately, then replicates in the background.

```
Client        Leader        Follower 1     Follower 2
  │             │               │              │
  │──WRITE──▶  │               │              │
  │◀──OK───────│               │              │
  │             │               │              │
  │             │──replicate──▶│              │  (background)
  │             │──replicate──────────────────▶│  (background)
  │             │               │              │
  │             │◀──ACK────────│              │
  │             │◀──ACK────────────────────────│

Timeline: Client gets OK immediately
Guarantee: Data is durable on leader only at OK time
Risk: Leader crash before replication → DATA LOSS
```

**Semi-synchronous replication**: The leader waits for at least one follower to acknowledge (used by MySQL semi-sync, PostgreSQL synchronous_standby_names).

```
Client        Leader        Follower 1     Follower 2
  │             │               │              │
  │──WRITE──▶  │               │              │
  │             │──replicate──▶│              │
  │             │──replicate──────────────────▶│
  │             │◀──ACK────────│              │
  │◀──OK───────│               │              │  (F2 ACK arrives later)
  │             │◀──ACK────────────────────────│

Timeline: Client waits until at least ONE follower acknowledges
Guarantee: Data survives leader + one follower failure
Balance: Faster than full-sync, safer than async
```

### 3.3 Quorum-Based Replication

Quorum protocols generalize the synchronous/asynchronous tradeoff.

Given N replicas, define:
- **W** = number of replicas that must acknowledge a write (write quorum)
- **R** = number of replicas that must respond to a read (read quorum)

**Strong consistency condition**: `W + R > N`

This ensures that every read quorum overlaps with every write quorum, so at least one replica in the read set has the latest value.

```
Example: N=3, W=2, R=2

Write quorum (W=2):     Read quorum (R=2):
Must write to 2 of 3    Must read from 2 of 3

   N1 ✓                    N1 ✓
   N2 ✓                    N2 ✓
   N3 (optional)           N3 (optional)

Overlap: At least one node (N1 or N2) has the latest write.
The read picks the value with the highest timestamp/version.
```

**Common configurations**:

| N | W | R | W+R > N? | Behavior |
|---|---|---|----------|----------|
| 3 | 3 | 1 | 4 > 3 ✓ | Strong reads, slow writes |
| 3 | 1 | 3 | 4 > 3 ✓ | Fast writes, slow reads |
| 3 | 2 | 2 | 4 > 3 ✓ | Balanced (most common) |
| 3 | 1 | 1 | 2 > 3 ✗ | Eventual consistency (fast but potentially stale) |
| 5 | 3 | 3 | 6 > 5 ✓ | Higher availability (tolerates 2 failures) |

**Sloppy quorums and hinted handoff**:

When some replicas are unreachable, a sloppy quorum allows writes to be accepted by other nodes (not the designated replicas) and stored as "hints." When the original replicas recover, the hints are forwarded to them. This improves availability at the cost of potential consistency violations.

### 3.4 Conflict Resolution

In multi-leader and leaderless systems, concurrent writes to the same key can create conflicts.

**Conflict resolution strategies**:

| Strategy | Description | Used By |
|----------|-------------|---------|
| **Last-Writer-Wins (LWW)** | The write with the highest timestamp wins; others are discarded | Cassandra, DynamoDB |
| **Version Vectors** | Track causal history; detect true conflicts (concurrent writes) | Riak, Dynamo |
| **CRDTs** | Data structures that merge automatically without conflicts (Conflict-free Replicated Data Types) | Riak, Redis (CRDT module) |
| **Application-level** | Return all conflicting versions to the application; let it decide | CouchDB, DynamoDB (optional) |
| **Operational Transform** | Transform concurrent operations to maintain consistency (used in collaborative editing) | Google Docs |

**Example: Version Vectors**

```
Node A and Node B both hold key "cart" with value ["item1"]

1. User writes to Node A: cart = ["item1", "item2"]
   Version: {A:1}

2. User writes to Node B: cart = ["item1", "item3"]
   Version: {B:1}

3. Node A and B sync:
   A has {A:1}, B has {B:1}
   Neither dominates → CONFLICT detected

   Resolution options:
   a. Merge: cart = ["item1", "item2", "item3"]  (union)
   b. Present both to user: "Which cart do you want?"
   c. LWW: pick the one with the higher timestamp (loses one write)
```

---

## 4. Distributed Query Processing

### 4.1 Overview

When data is fragmented across nodes, a query may need to access data from multiple nodes. The distributed query processor must:

1. **Decompose** the global query into subqueries for each fragment
2. **Optimize** the execution plan to minimize data transfer
3. **Execute** subqueries in parallel where possible
4. **Assemble** the final result

### 4.2 Query Decomposition

```
Global query:
  SELECT e.name, d.dept_name
  FROM employees e JOIN departments d ON e.dept_id = d.id
  WHERE e.salary > 80000;

Assume:
  - employees is horizontally fragmented by dept_id across Node A and Node B
  - departments is replicated on all nodes

Decomposition:

Node A (dept_id = 1..50):
  SELECT e.name, d.dept_name
  FROM employees_frag_A e JOIN departments d ON e.dept_id = d.id
  WHERE e.salary > 80000;

Node B (dept_id = 51..100):
  SELECT e.name, d.dept_name
  FROM employees_frag_B e JOIN departments d ON e.dept_id = d.id
  WHERE e.salary > 80000;

Coordinator:
  UNION ALL results from Node A and Node B
```

### 4.3 Cost Factors

The dominant cost in distributed query processing is **data transfer** over the network. Local I/O and CPU costs are secondary.

```
Cost model:
  Total Cost = Σ(Local I/O cost at each node)
             + Σ(CPU cost at each node)
             + Σ(Network transfer cost)
               ↑
               This dominates!
```

**Optimization strategies** to minimize data transfer:

1. **Push selections and projections down**: Filter and project at each node before sending results, reducing the volume of data transferred.

2. **Semi-join reduction**: Instead of shipping an entire relation for a join, ship only the join column values, then request matching tuples.

```
Without semi-join:
  Node A sends entire employees table to Node B for join
  Transfer: |employees| rows × row_size bytes

With semi-join:
  Step 1: Node B sends π_{dept_id}(departments) to Node A    (small!)
  Step 2: Node A computes employees ⋉ dept_ids               (local filter)
  Step 3: Node A sends only matching employees to Node B      (much smaller)
```

3. **Bloom filter optimization**: Instead of sending the actual join keys, send a compact Bloom filter. False positives send a few extra rows, but the filter size is dramatically smaller.

4. **Parallel execution**: Execute subqueries at different nodes simultaneously.

### 4.4 Distributed Join Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Ship-whole** | Send one entire relation to the other's node | One relation is very small |
| **Semi-join** | Exchange join keys, then ship matching tuples | Selective join (few matches) |
| **Hash partitioned join** | Both relations hash-partitioned on join key, join locally | Large-large equi-join |
| **Broadcast join** | Replicate small table to all nodes | Small-large join |
| **Collocated join** | Both tables partitioned on join key → join locally with no transfer | Tables co-partitioned |

**Collocated join example**:

```
If orders and order_items are both partitioned by order_id:

Node 1: orders (order_id 1-1000)     + order_items (order_id 1-1000)
Node 2: orders (order_id 1001-2000)  + order_items (order_id 1001-2000)
Node 3: orders (order_id 2001-3000)  + order_items (order_id 2001-3000)

JOIN orders o ON o.order_id = order_items.order_id
→ Each node can execute the join LOCALLY (no network transfer!)
```

This is why choosing the right partition key is critical: it determines which joins can be collocated.

---

## 5. Distributed Transactions: Two-Phase Commit (2PC)

### 5.1 The Problem

When a transaction spans multiple nodes, we need all nodes to agree on whether to commit or abort. Partial commits (some nodes commit, others abort) violate atomicity.

```
Transaction T: Transfer $100 from Account A (Node 1) to Account B (Node 2)

Node 1: UPDATE accounts SET balance = balance - 100 WHERE id = 'A';
Node 2: UPDATE accounts SET balance = balance + 100 WHERE id = 'B';

What if Node 1 commits but Node 2 crashes?
→ $100 disappears! Atomicity violated.
```

### 5.2 Two-Phase Commit Protocol

2PC uses a designated **coordinator** and the participating **nodes** (participants).

**Phase 1: Prepare (Voting)**

```
Coordinator                  Node 1                  Node 2
    │                          │                       │
    │───── PREPARE ──────────▶│                       │
    │───── PREPARE ─────────────────────────────────▶ │
    │                          │                       │
    │                    (execute txn                   │
    │                     locally,                      │
    │                     acquire locks,                │
    │                     write to WAL)                 │
    │                          │                       │
    │◀──── VOTE YES ──────────│                       │
    │◀──── VOTE YES ────────────────────────────────── │
    │                          │                       │
```

Each participant:
1. Executes the transaction locally (but does NOT commit)
2. Writes a prepare record to its WAL (Write-Ahead Log)
3. Responds with VOTE YES (if it can commit) or VOTE NO (if it cannot)

**Phase 2: Commit/Abort (Decision)**

If all participants voted YES:

```
Coordinator                  Node 1                  Node 2
    │                          │                       │
    │  (write COMMIT to        │                       │
    │   coordinator WAL)       │                       │
    │                          │                       │
    │───── COMMIT ───────────▶│                       │
    │───── COMMIT ──────────────────────────────────▶ │
    │                          │                       │
    │                    (commit locally,               │
    │                     release locks,                │
    │                     write commit to WAL)          │
    │                          │                       │
    │◀──── ACK ───────────────│                       │
    │◀──── ACK ─────────────────────────────────────── │
    │                          │                       │
```

If any participant voted NO:

```
Coordinator                  Node 1                  Node 2
    │                          │                       │
    │  (write ABORT to         │                       │
    │   coordinator WAL)       │                       │
    │                          │                       │
    │───── ABORT ────────────▶│                       │
    │───── ABORT ───────────────────────────────────▶ │
    │                          │                       │
    │                    (rollback locally,             │
    │                     release locks)                │
    │                          │                       │
    │◀──── ACK ───────────────│                       │
    │◀──── ACK ─────────────────────────────────────── │
```

### 5.3 State Machine

```
                    Coordinator                          Participant
              ┌─────────────────┐                 ┌─────────────────┐
              │     INITIAL     │                 │     INITIAL     │
              └────────┬────────┘                 └────────┬────────┘
                       │ send PREPARE                      │ receive PREPARE
                       ▼                                   ▼
              ┌─────────────────┐                 ┌─────────────────┐
              │     WAITING     │                 │     READY       │
              │  (for votes)    │                 │  (voted YES)    │
              └────────┬────────┘                 └────────┬────────┘
                       │                                   │
              ┌────────┴────────┐                 ┌────────┴────────┐
              │ all YES  │ any NO                 │ COMMIT   │ ABORT
              ▼          ▼                        ▼          ▼
        ┌──────────┐ ┌──────────┐         ┌──────────┐ ┌──────────┐
        │ COMMITTED│ │ ABORTED  │         │ COMMITTED│ │ ABORTED  │
        └──────────┘ └──────────┘         └──────────┘ └──────────┘
```

### 5.4 Failure Analysis

2PC must handle various failure scenarios:

**Case 1: Participant crashes before voting**

```
Coordinator times out waiting for vote → ABORT transaction
Participant recovers → consults WAL → no prepare record → knows to abort
```

**Case 2: Participant crashes after voting YES**

```
Coordinator received YES → proceeds normally
Participant recovers → consults WAL → sees prepare record → contacts coordinator for decision
This is the "in-doubt" state: participant has voted YES but does not know the outcome
```

**Case 3: Coordinator crashes after collecting votes**

```
THIS IS THE CRITICAL PROBLEM WITH 2PC!

Participants are in the READY state (voted YES) and cannot proceed:
- Cannot COMMIT: the coordinator might have decided ABORT
- Cannot ABORT: the coordinator might have decided COMMIT
- Cannot ask other participants: they might also be in READY state

RESULT: Participants must BLOCK until the coordinator recovers.
```

This **blocking problem** is the fundamental weakness of 2PC. In the worst case, participants hold locks indefinitely, blocking other transactions.

### 5.5 Optimizations

**Presumed Abort**: If the coordinator crashes without writing a decision to its log, it presumes ABORT upon recovery. This eliminates the need to write ABORT decisions to the log.

**Read-Only Optimization**: If a participant determines that the transaction did not modify any data at its site, it votes READ-ONLY and can immediately release its resources without waiting for the decision.

**Transfer of Coordination**: A participant can take over the coordinator role if the original coordinator fails (requires agreement among remaining participants).

---

## 6. Three-Phase Commit (3PC)

### 6.1 Motivation

3PC was proposed by Dale Skeen in 1981 to solve the blocking problem of 2PC by adding an extra phase.

### 6.2 Protocol

**Phase 1: CanCommit (Voting)** -- Same as 2PC Phase 1.

**Phase 2: PreCommit** -- A new intermediate phase.

**Phase 3: DoCommit** -- Same as 2PC Phase 2.

```
Coordinator              Node 1                 Node 2
    │                      │                      │
    │── CAN_COMMIT? ──────▶│                      │
    │── CAN_COMMIT? ─────────────────────────────▶│
    │                      │                      │
    │◀── YES ─────────────│                      │
    │◀── YES ──────────────────────────────────── │
    │                      │                      │
    │── PRE_COMMIT ───────▶│                      │   ← NEW PHASE
    │── PRE_COMMIT ──────────────────────────────▶│
    │                      │                      │
    │◀── ACK ─────────────│                      │
    │◀── ACK ──────────────────────────────────── │
    │                      │                      │
    │── DO_COMMIT ────────▶│                      │
    │── DO_COMMIT ─────────────────────────────▶ │
    │                      │                      │
    │◀── DONE ────────────│                      │
    │◀── DONE ─────────────────────────────────── │
```

### 6.3 How 3PC Avoids Blocking

The key insight: in 3PC, if a participant is in the PreCommit state, it knows that ALL participants voted YES. Therefore:

- If the coordinator crashes during PreCommit, any surviving participant that received PRE_COMMIT knows the coordinator intended to commit. They can elect a new coordinator and proceed with COMMIT.
- If a participant has NOT received PRE_COMMIT, it can safely ABORT (because the coordinator might not have reached the commit decision).

**The non-blocking property**: No single node failure can cause the remaining nodes to block indefinitely.

### 6.4 Limitations of 3PC

Despite solving the blocking problem in theory, 3PC has significant practical limitations:

1. **Network partitions**: 3PC assumes a fail-stop model (nodes crash but do not partition). In a real network partition, 3PC can still violate consistency:

```
Scenario:
  Partition splits nodes into {Coordinator, Node1} and {Node2}

  Coordinator sends PRE_COMMIT to Node1 (received)
  Coordinator sends PRE_COMMIT to Node2 (LOST due to partition)

  Coordinator crashes.

  Node1 (received PRE_COMMIT): "Coordinator wanted to commit" → COMMIT
  Node2 (did not receive PRE_COMMIT): "Timeout, no PRE_COMMIT" → ABORT

  INCONSISTENCY! Node1 committed, Node2 aborted.
```

2. **Extra round-trip**: 3PC requires one more round-trip than 2PC, increasing latency.

3. **Rarely used in practice**: Due to the network partition vulnerability, most modern systems use 2PC with timeouts or Paxos/Raft-based consensus instead.

### 6.5 Comparison: 2PC vs 3PC

| Property | 2PC | 3PC |
|----------|-----|-----|
| **Phases** | 2 | 3 |
| **Blocking** | Yes (coordinator crash) | No (in fail-stop model) |
| **Network partitions** | Blocks | Can be inconsistent |
| **Latency** | 2 round-trips | 3 round-trips |
| **Complexity** | Moderate | High |
| **Practical use** | Widespread | Rare |
| **Message count** | 4N (prepare + vote + decision + ack) | 6N (more messages) |

---

## 7. Consensus Algorithms: Paxos and Raft

### 7.1 The Consensus Problem

In a distributed system, consensus is the problem of getting multiple nodes to agree on a single value. This is fundamental to:

- **Leader election**: Which node is the current leader?
- **Distributed transactions**: Should we commit or abort?
- **Replicated state machines**: What is the next operation to apply?
- **Configuration management**: What is the current cluster membership?

**Formal requirements**:
1. **Agreement**: All correct nodes decide the same value.
2. **Validity**: The decided value was proposed by some node.
3. **Termination**: Every correct node eventually decides.

The **FLP impossibility theorem** (Fischer, Lynch, Paterson, 1985) proves that in an asynchronous system with even one faulty node, it is impossible to guarantee consensus. Practical algorithms like Paxos and Raft overcome this by using timeouts and randomization (technically, they may not terminate in adversarial scenarios, but they work well in practice).

### 7.2 Paxos

Paxos, invented by Leslie Lamport in 1989, is the foundational consensus algorithm. It is notoriously difficult to understand (Lamport originally described it as a parliamentary protocol on the fictional Greek island of Paxos).

**Roles**:
- **Proposer**: Proposes a value
- **Acceptor**: Votes on proposals
- **Learner**: Learns the decided value

(A single node can play multiple roles.)

**Single-decree Paxos** (agreeing on one value):

**Phase 1: Prepare**

```
Proposer                       Acceptors (majority needed)
   │                             A1        A2        A3
   │                              │         │         │
   │──PREPARE(n=1)──────────────▶│         │         │
   │──PREPARE(n=1)──────────────────────▶ │         │
   │──PREPARE(n=1)────────────────────────────────▶ │
   │                              │         │         │
   │◀─PROMISE(n=1, null)────────│         │         │
   │◀─PROMISE(n=1, null)──────────────── │         │
   │◀─PROMISE(n=1, null)─────────────────────────── │
   │                              │         │         │

Each acceptor promises:
- "I will not accept any proposal with number < 1"
- If acceptor has previously accepted a value, it includes that value
```

**Phase 2: Accept**

```
Proposer                       Acceptors
   │                             A1        A2        A3
   │                              │         │         │
   │──ACCEPT(n=1, v="X")────────▶│         │         │
   │──ACCEPT(n=1, v="X")────────────────▶ │         │
   │──ACCEPT(n=1, v="X")──────────────────────────▶ │
   │                              │         │         │
   │◀─ACCEPTED(n=1)─────────────│         │         │
   │◀─ACCEPTED(n=1)───────────────────── │         │
   │◀─ACCEPTED(n=1)──────────────────────────────── │
   │                              │         │         │

Majority accepted → value "X" is CHOSEN
```

**Key insight**: If a proposer receives a promise with a previously accepted value, it MUST propose that value (not its own). This ensures that once a value is chosen, it cannot be changed.

**Multi-Paxos**: Extends single-decree Paxos by electing a stable leader that skips Phase 1 for subsequent proposals. This reduces the protocol to a single phase (Phase 2 only) in the common case, making it practical for replicated logs.

### 7.3 Raft

Raft was designed by Diego Ongaro and John Ousterhout in 2014 as an understandable alternative to Paxos. It provides the same guarantees but with a clearer structure.

**Key idea**: Raft decomposes consensus into three sub-problems:
1. **Leader election**: Choose one node to be the leader.
2. **Log replication**: The leader accepts entries and replicates them to followers.
3. **Safety**: Ensure that committed entries are not lost.

**Raft state machine**:

```
                        ┌──────────┐
               timeout, │          │ receives vote
               start    │ Follower │ from candidate
               election │          │ with higher term
                   ┌────│          │◀────┐
                   │    └──────────┘     │
                   │         ▲           │
                   │         │           │
                   ▼         │           │
              ┌──────────┐   │      ┌──────────┐
              │Candidate │   │      │  Leader   │
              │          │───┘      │          │
              │          │─────────▶│          │
              └──────────┘  wins    └──────────┘
                             election
```

**Leader Election**:

```
Time ──────────────────────────────────────────────────────▶

Term 1                    Term 2                    Term 3
┌─────────────────────┐  ┌─────────────────────┐  ┌────────
│ Leader: Node A      │  │ Leader: Node B      │  │ ...
│ Normal operation    │  │ Normal operation    │  │
└─────────────────────┘  └─────────────────────┘  └────────
                      ↑
                Node A crashes,
                Node B wins election

Election process:
1. Follower times out (no heartbeat from leader)
2. Becomes candidate, increments term, votes for itself
3. Sends RequestVote RPCs to all other nodes
4. If majority votes YES → becomes leader
5. If another leader discovered → reverts to follower
6. If timeout → increment term, restart election
```

**Log Replication**:

```
Leader                          Followers
  │                         F1         F2         F3
  │                          │          │          │
  │  Client: SET x = 5       │          │          │
  │  Append to local log     │          │          │
  │                          │          │          │
  │──AppendEntries(x=5)─────▶│          │          │
  │──AppendEntries(x=5)──────────────▶ │          │
  │──AppendEntries(x=5)───────────────────────── ▶│
  │                          │          │          │
  │◀──ACK──────────────────│          │          │
  │◀──ACK───────────────────────────── │          │
  │                          │          │          │
  │  Majority (2 of 3) ACKed           │          │
  │  → COMMIT entry                    │          │
  │  → Respond to client               │          │
  │                          │          │          │
  │──AppendEntries(commit)──▶│          │          │
  │──AppendEntries(commit)───────────▶ │          │
  │──AppendEntries(commit)────────────────────── ▶│
```

**Safety property**: Only a node with all committed entries can become leader. Raft ensures this through the election mechanism: a candidate must have a log that is at least as up-to-date as the voter's log to receive a vote.

### 7.4 Paxos vs Raft

| Aspect | Paxos | Raft |
|--------|-------|------|
| **Understandability** | Notoriously difficult | Designed for clarity |
| **Leader** | Optional (Multi-Paxos has leader) | Required |
| **Log structure** | Can have gaps | No gaps (sequential) |
| **Safety proof** | Complex | Straightforward |
| **Reconfiguration** | Requires separate protocol | Joint consensus built-in |
| **Real-world use** | Google Chubby, Spanner | etcd, CockroachDB, Consul, TiKV |
| **Theoretical foundations** | Stronger (more general) | Equivalent in practice |

### 7.5 Consensus in Databases

| Database | Consensus Algorithm | Purpose |
|----------|---------------------|---------|
| Google Spanner | Multi-Paxos | Log replication within each Paxos group |
| CockroachDB | Raft | Range replication across nodes |
| TiDB (TiKV) | Raft | Region replication |
| etcd | Raft | Key-value store for Kubernetes |
| ZooKeeper | Zab (Paxos variant) | Coordination service |
| Cassandra | Paxos (lightweight txns) | Compare-and-set operations |

---

## 8. Distributed Concurrency Control

### 8.1 Challenges

Concurrency control in a distributed setting faces additional challenges compared to single-node systems:

- **No global clock**: Nodes cannot agree on the exact time, making timestamp ordering difficult.
- **Network latency**: Lock requests and releases incur network round-trips.
- **Distributed deadlocks**: Deadlock cycles may span multiple nodes.
- **Partial failures**: A node holding locks may crash, leaving locks held indefinitely.

### 8.2 Distributed Two-Phase Locking (D2PL)

Extend 2PL to distributed settings: each node has a local lock manager, and a transaction acquires locks at each node it accesses.

```
Transaction T accesses Node 1 and Node 2:

Node 1 Lock Manager               Node 2 Lock Manager
┌─────────────────┐               ┌─────────────────┐
│ Lock table:      │               │ Lock table:      │
│ row_A → T (X)   │               │ row_B → T (X)   │
│ row_C → T (S)   │               │ row_D → T (S)   │
└─────────────────┘               └─────────────────┘

Growing phase: T acquires locks at both nodes
Commit: 2PC ensures atomic commit
Shrinking phase: After 2PC decision, all nodes release T's locks
```

### 8.3 Distributed Deadlock Detection

**Wait-for graph**: Each node maintains a local wait-for graph. A global deadlock occurs when the union of all local graphs has a cycle.

```
Node 1 local WFG:       Node 2 local WFG:
  T1 → T2                 T2 → T3

Node 3 local WFG:
  T3 → T1

Global WFG: T1 → T2 → T3 → T1   ← CYCLE = DEADLOCK
```

**Detection approaches**:

| Approach | Description | Tradeoffs |
|----------|-------------|-----------|
| **Centralized** | One node collects all local WFGs and checks for global cycles | Simple but single point of failure |
| **Distributed** | Nodes send WFG edges to each other; each node detects cycles it can see | No single point of failure but may detect phantom deadlocks |
| **Timeout-based** | If a transaction waits longer than a threshold, assume deadlock and abort | Simple but may abort non-deadlocked transactions |

**Phantom deadlock**: A false deadlock detected due to stale information.

```
Time 1: T1 waits for T2 at Node 1 (WFG edge: T1 → T2)
Time 2: T2 finishes at Node 1, but the WFG edge hasn't been removed yet
Time 3: T2 starts waiting for T1 at Node 2 (WFG edge: T2 → T1)

If the deadlock detector sees both edges simultaneously:
  T1 → T2 → T1 → PHANTOM DEADLOCK (T2 already released the lock)
```

### 8.4 Distributed Timestamp Ordering

Assign globally unique timestamps to transactions. The challenge is generating timestamps without a global clock.

**Lamport timestamps**: A logical clock that ensures partial ordering.

```
Each node maintains a counter C:
- Before each event: C = C + 1
- When sending a message: attach C to the message
- When receiving a message with timestamp T: C = max(C, T) + 1

Timestamp = (C, node_id) for total ordering
```

**TrueTime (Google Spanner)**: Uses GPS and atomic clocks to provide a bounded uncertainty interval [earliest, latest] for the real time. Spanner waits out the uncertainty before committing, achieving external consistency (real-time ordering).

```
TrueTime API:
  TT.now() → [earliest, latest]
  TT.after(t) → true if t is definitely in the past
  TT.before(t) → true if t is definitely in the future

Commit wait:
  1. Transaction T gets commit timestamp s = TT.now().latest
  2. Wait until TT.after(s) is true
  3. Commit with timestamp s

  This ensures that if T1 commits before T2 starts (real time),
  then s1 < s2 (timestamp order matches real-time order).
```

### 8.5 Distributed MVCC

Many distributed databases use Multi-Version Concurrency Control:

```
Distributed MVCC for key "account_balance":

Node 1 (primary):
  Version 1: value=1000, ts=100, committed
  Version 2: value=900,  ts=150, committed
  Version 3: value=850,  ts=200, committed

Node 2 (replica):
  Version 1: value=1000, ts=100, committed
  Version 2: value=900,  ts=150, committed
  (Version 3 not yet replicated)

Read at Node 2 with snapshot ts=175:
  → Returns Version 2 (value=900) because ts=150 ≤ 175

Read at Node 1 with snapshot ts=175:
  → Returns Version 2 (value=900) because ts=200 > 175

Both reads are consistent! (snapshot isolation)
```

---

## 9. CAP Theorem Implications for Distributed Design

### 9.1 Revisiting CAP in Design Context

We introduced the CAP theorem in [Lesson 13](./13_NoSQL_Data_Models.md). Now we apply it to specific design decisions in distributed databases.

### 9.2 CP Design Patterns

**Pattern: Linearizable reads via leader**

```
All reads and writes go through the leader:

Client ──▶ Leader ──▶ Followers (replication)
                  ◀── (reads served by leader only)

Tradeoff: If the leader is unreachable (partition), reads fail (reduced availability)
Used by: etcd, ZooKeeper, HBase
```

**Pattern: Majority quorum**

```
Read from majority, write to majority:

Client writes to 2 of 3 replicas (W=2)
Client reads from 2 of 3 replicas (R=2)
W + R = 4 > 3 = N → guaranteed overlap

Tradeoff: During partition, if majority is unreachable, operations fail
Used by: MongoDB (majority read/write concern)
```

### 9.3 AP Design Patterns

**Pattern: Read from any replica**

```
Client reads from any available replica:

Client ──▶ Any Replica (may return stale data)

Tradeoff: Always available, but may read stale data after a write
Used by: Cassandra (with consistency level ONE), DynamoDB (eventually consistent reads)
```

**Pattern: Hinted handoff**

```
During partition, write to available nodes:

Normal:    Client ──▶ Replica A, Replica B, Replica C
Partition: Client ──▶ Replica A, Replica D* (hint), Replica E* (hint)
Recovery:  Replica D ──▶ Replica B (forward hinted write)
           Replica E ──▶ Replica C (forward hinted write)

* D and E are not the designated replicas but accept the write as a "hint"
Tradeoff: Write succeeds (available) but consistency is deferred
Used by: Dynamo, Cassandra, Riak
```

### 9.4 Consistency Level Tuning

Many distributed databases allow per-operation consistency tuning:

```
Cassandra consistency levels:

ONE:    Write/Read to 1 replica      (fastest, least consistent)
QUORUM: Write/Read to majority       (balanced)
ALL:    Write/Read to all replicas   (slowest, most consistent)
LOCAL_QUORUM: Majority in local DC   (multi-datacenter)

DynamoDB consistency:
  Eventually Consistent Read: 0.5x cost, may return stale data
  Strongly Consistent Read:   1x cost, always returns latest

MongoDB read concern:
  "local":    Return local data (fast, may be stale)
  "majority": Return data acknowledged by majority (consistent)
  "linearizable": Strongest, single-document reads only
```

---

## 10. Partitioning Strategies

### 10.1 Range Partitioning

Assign contiguous ranges of the key space to each partition.

```
Key space: [0, 1000]

Partition 1 (Node A): keys [0, 333]
Partition 2 (Node B): keys [334, 666]
Partition 3 (Node C): keys [667, 1000]

Example: User IDs
  Node A: users 1-333
  Node B: users 334-666
  Node C: users 667-1000
```

**Advantages**:
- Range queries are efficient (scan a single partition or consecutive partitions)
- Easy to understand and implement

**Disadvantages**:
- **Hot spots**: If the key distribution is skewed (e.g., recent timestamps concentrate on one partition), one node gets disproportionate load
- **Rebalancing**: When splitting a hot partition, data must be moved

**Used by**: HBase, Spanner, CockroachDB, TiDB

### 10.2 Hash Partitioning

Apply a hash function to the key and assign hash ranges to partitions.

```
Hash function: h(key) → [0, 2^32)

Partition 1 (Node A): h(key) ∈ [0, 2^32/3)
Partition 2 (Node B): h(key) ∈ [2^32/3, 2*2^32/3)
Partition 3 (Node C): h(key) ∈ [2*2^32/3, 2^32)

Example: h("user_42") = 178294 → Partition 1 (Node A)
         h("user_43") = 891023 → Partition 2 (Node B)

Adjacent keys are scattered across different partitions.
```

**Advantages**:
- Even distribution of keys across partitions (no hot spots for random keys)
- Works well with any key distribution

**Disadvantages**:
- **Range queries impossible**: Keys that are close together in the original space are scattered across partitions. `SELECT * FROM users WHERE id BETWEEN 100 AND 200` must query ALL partitions.
- **Rebalancing**: Adding a node requires rehashing and moving ~1/N of all data.

**Used by**: DynamoDB, Cassandra (default partitioner)

### 10.3 Consistent Hashing

Consistent hashing solves the rebalancing problem of hash partitioning. It was introduced by Karger et al. in 1997.

**Concept**: Both keys and nodes are mapped onto a circular hash ring.

```
           0 (= 2^32)
            │
            │
   N3 ──── │ ──── N1
  (hash=    │     (hash=
   300)     │      50)
            │
            │
            │
            │
   N2 ──── │
  (hash=    │
   180)     │

Keys are assigned to the first node clockwise:
  h("k1") = 30  → N1 (next clockwise from 30 is N1 at 50)
  h("k2") = 70  → N2 (next clockwise from 70 is N2 at 180)
  h("k3") = 200 → N3 (next clockwise from 200 is N3 at 300)
  h("k4") = 310 → N1 (wraps around to N1 at 50)
```

**Adding a node**: Only keys between the new node and its predecessor need to move.

```
Before: N1(50), N2(180), N3(300)
Add N4 at position 120:

   Before:
   h("k5") = 100 → N2 (next after 100 was N2 at 180)

   After:
   h("k5") = 100 → N4 (next after 100 is now N4 at 120)

   Only keys in range (50, 120] move from N2 to N4.
   All other keys stay on their current nodes!
```

**Virtual nodes**: To ensure even distribution, each physical node is assigned multiple "virtual" positions on the ring.

```
Physical Node A → Virtual nodes at positions 50, 120, 240
Physical Node B → Virtual nodes at positions 80, 190, 310
Physical Node C → Virtual nodes at positions 30, 160, 280

With more virtual nodes per physical node:
- More even distribution of keys
- Smoother rebalancing when nodes join/leave
- Better fault tolerance (a failed node's load is spread across many nodes)
```

**Used by**: Dynamo, Cassandra, Riak, Memcached, CDN load balancing

### 10.4 Partitioning Comparison

| Feature | Range | Hash | Consistent Hashing |
|---------|-------|------|-------------------|
| **Range queries** | Efficient | Impossible | Impossible |
| **Load distribution** | May be skewed | Even | Even (with vnodes) |
| **Rebalancing cost** | Split/merge ranges | Rehash ~1/N | Move ~1/N |
| **Hot spots** | Possible | Unlikely | Unlikely |
| **Implementation** | Simple | Simple | Moderate |
| **Ordering** | Preserved | Lost | Lost |

### 10.5 Compound Partitioning

Some databases use a combination: hash the partition key for distribution, then range-order within each partition using a sort key.

```
Cassandra: PRIMARY KEY ((partition_key), clustering_key)
DynamoDB:  Partition Key + Sort Key

Example: Messages table
  Partition Key: conversation_id (hashed → even distribution)
  Sort Key: timestamp (range-ordered within partition)

Physical layout:
  Node A: conversation_123 → [msg at t1, msg at t2, msg at t3, ...]  (sorted)
  Node B: conversation_456 → [msg at t1, msg at t2, ...]  (sorted)

Query: "Get messages in conversation_123 from last hour"
  → Hash conversation_123 → Node A
  → Range scan on timestamp within the partition (efficient!)
```

---

## 11. Exercises

### Exercise 1: Architecture Selection

For each of the following scenarios, recommend a distributed database architecture (shared-nothing, shared-disk, or shared-memory) and justify your choice.

1. A global e-commerce company with datacenters on 4 continents, serving 500 million users.
2. A medium-sized company running analytics on 10 TB of data with 20 concurrent analysts.
3. A startup with a single database server reaching its limits and needing to scale compute independently of storage.

### Exercise 2: Fragmentation Design

Given the following schema:

```sql
CREATE TABLE employees (
  id INT PRIMARY KEY,
  name VARCHAR(100),
  dept_id INT,
  salary DECIMAL,
  ssn VARCHAR(11),
  hire_date DATE
);

CREATE TABLE departments (
  id INT PRIMARY KEY,
  name VARCHAR(100),
  location VARCHAR(50)
);
```

Design a fragmentation strategy for a distributed database with 3 nodes (one in New York, one in London, one in Tokyo). The company has departments in all three cities. Consider:
1. How would you horizontally fragment `employees`?
2. How would you vertically fragment `employees` to protect sensitive data?
3. Should `departments` be replicated or fragmented? Why?
4. Verify the completeness and reconstruction conditions for your fragments.

### Exercise 3: Quorum Calculations

Given a cluster with N = 5 replicas:

1. What values of W and R guarantee strong consistency?
2. If W = 3 and R = 3, how many node failures can the system tolerate for reads? For writes?
3. If you want writes to be fast (W = 1), what must R be for strong consistency? What is the read latency implication?
4. A system has N = 7, W = 4, R = 4. Is this strongly consistent? How many failures can it tolerate?

### Exercise 4: 2PC Trace

Trace through the 2PC protocol for the following scenario:

A transaction T transfers $500 from Account A (Node 1) to Account B (Node 2).

1. Show the message sequence when both nodes vote YES.
2. Show the message sequence when Node 2 votes NO.
3. Show what happens if the coordinator crashes AFTER receiving all YES votes but BEFORE sending COMMIT messages. What state are the participants in? How long might they wait?

### Exercise 5: Consensus Scenario

Consider a 5-node Raft cluster (nodes A, B, C, D, E) where node A is the current leader in term 3.

1. Node A crashes. Walk through the leader election process. Which node becomes the new leader and why?
2. Before crashing, node A had replicated log entry [term 3, SET x=5] to nodes A, B, and C (but not D and E). Is this entry committed? Why or why not?
3. If node B becomes the new leader in term 4, will it include entry [term 3, SET x=5] in its log? Explain.

### Exercise 6: Distributed Deadlock

Three transactions run across two nodes:

```
Node 1:
  T1 holds lock on row A, waiting for lock on row B (held by T2)
  T2 holds lock on row B

Node 2:
  T2 is waiting for lock on row C (held by T3)
  T3 holds lock on row C, waiting for lock on row A (held by T1 on Node 1)
```

1. Draw the global wait-for graph.
2. Is there a deadlock? If so, which transaction should be aborted to break the cycle?
3. How would a centralized deadlock detector discover this deadlock?
4. How might a timeout-based approach handle this? What are the risks?

### Exercise 7: Partitioning Strategy

You are designing a distributed database for a social media platform. The `posts` table has the following schema:

```sql
CREATE TABLE posts (
  post_id BIGINT,
  user_id BIGINT,
  content TEXT,
  created_at TIMESTAMP,
  like_count INT
);
```

Common queries:
1. Get all posts by a specific user, ordered by creation time (most recent first).
2. Get a specific post by ID.
3. Get the 100 most recent posts across all users (global timeline).

For each partitioning strategy (range on user_id, hash on post_id, consistent hashing on user_id):
1. Explain how each query would be executed.
2. Identify potential hot spots.
3. Recommend the best strategy and justify your choice.

### Exercise 8: Consistent Hashing

Given a consistent hashing ring with positions [0, 360):

Nodes: A at 45, B at 120, C at 200, D at 310

Keys and their hash values:
- k1 = 30, k2 = 90, k3 = 150, k4 = 210, k5 = 330, k6 = 10

1. Which node is responsible for each key?
2. Node E is added at position 170. Which keys are reassigned and to which node?
3. Node B fails. Which keys are reassigned and to which node?
4. If each physical node has 3 virtual nodes (e.g., A at 45, 165, 285), redo the key assignments.

### Exercise 9: Replication Conflict Resolution

Two users concurrently update a shared document in a leaderless replicated system (3 replicas):

```
User X (connects to Replica 1): SET title = "Draft v2"    at time T=100
User Y (connects to Replica 2): SET title = "Final Draft"  at time T=101
```

Due to a network partition, Replica 3 is unreachable.

1. Under Last-Writer-Wins (LWW), what is the final value? Is any data lost?
2. Under version vectors, how would the conflict be detected and represented?
3. Propose an application-level resolution strategy for this scenario.
4. How would a CRDT (specifically, a Last-Writer-Wins Register) handle this?

### Exercise 10: Distributed Query Optimization

Consider two tables distributed across two nodes:

```
Node 1: orders (10 million rows, 500 bytes/row)
  Columns: order_id, customer_id, total, order_date

Node 2: customers (100,000 rows, 200 bytes/row)
  Columns: customer_id, name, city, email
```

Query: `SELECT c.name, o.total FROM orders o JOIN customers c ON o.customer_id = c.customer_id WHERE c.city = 'Tokyo';`

Assume Tokyo customers are 5% of all customers (5,000 customers).
Assume each customer has ~100 orders on average.
Network transfer cost: 1 ms per MB.

Compare the cost of these strategies:
1. **Ship-whole**: Send entire `customers` table to Node 1, join there.
2. **Ship-whole (reverse)**: Send entire `orders` table to Node 2, join there.
3. **Semi-join**: Send Tokyo customer_ids to Node 1, fetch matching orders, join at Node 2.
4. **Bloom filter**: Send a Bloom filter of Tokyo customer_ids to Node 1.

Calculate the approximate data transfer for each strategy in MB.

---

## 12. References

1. Ozsu, M. T. & Valduriez, P. (2020). *Principles of Distributed Database Systems*, 4th Edition. Springer.
2. Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly Media. Chapters 5-9.
3. Lamport, L. (1998). "The Part-Time Parliament" (Paxos). ACM TOCS.
4. Ongaro, D. & Ousterhout, J. (2014). "In Search of an Understandable Consensus Algorithm" (Raft). USENIX ATC.
5. Fischer, M., Lynch, N., Paterson, M. (1985). "Impossibility of Distributed Consensus with One Faulty Process" (FLP). JACM.
6. Corbett, J. et al. (2013). "Spanner: Google's Globally-Distributed Database." ACM TODS.
7. DeCandia, G. et al. (2007). "Dynamo: Amazon's Highly Available Key-Value Store." SOSP.
8. Karger, D. et al. (1997). "Consistent Hashing and Random Trees." ACM STOC.
9. Skeen, D. (1981). "Nonblocking Commit Protocols." ACM SIGMOD.
10. Gray, J. & Lamport, L. (2006). "Consensus on Transaction Commit." ACM TODS.

---

**Previous**: [13. NoSQL Data Models](./13_NoSQL_Data_Models.md) | **Next**: [15. NewSQL and Modern Trends](./15_NewSQL_and_Modern_Trends.md)
