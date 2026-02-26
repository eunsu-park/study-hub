# Database Replication

**Previous**: [Database Scaling](./08_Database_Scaling.md) | **Next**: [Data Consistency Patterns](./10_Data_Consistency_Patterns.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the purpose of database replication and the benefits it provides (availability, fault tolerance, read scaling)
2. Compare single-leader, multi-leader, and leaderless replication topologies and their respective trade-offs
3. Distinguish between synchronous and asynchronous replication and explain semi-synchronous as a practical compromise
4. Identify replication lag issues and describe consistency guarantees such as read-your-writes and monotonic reads
5. Describe the failover process for leader election and explain how split-brain scenarios are prevented
6. Apply quorum formulas (R + W > N) to configure consistency levels for a distributed database

**Difficulty**: ⭐⭐⭐ (Intermediate)

---

Every production database needs copies of its data on multiple machines -- if your only database server crashes, you lose everything. Replication is the mechanism that keeps these copies in sync, but the devil is in the details: how quickly do replicas converge, what happens during a network partition, and who resolves conflicting writes? These questions sit at the heart of distributed systems design and come up in virtually every system design interview.

## Table of Contents

1. [Concept and Purpose of Replication](#1-concept-and-purpose-of-replication)
2. [Single-Leader Replication](#2-single-leader-replication)
3. [Multi-Leader Replication](#3-multi-leader-replication)
4. [Leaderless Replication](#4-leaderless-replication)
5. [Synchronous/Asynchronous Replication](#5-synchronousasynchronous-replication)
6. [Replication Lag and Consistency Issues](#6-replication-lag-and-consistency-issues)
7. [Failure Recovery and Leader Election](#7-failure-recovery-and-leader-election)
8. [Quorum and Consistency Levels](#8-quorum-and-consistency-levels)
9. [Practice Problems](#9-practice-problems)
10. [References](#10-references)

---

## 1. Concept and Purpose of Replication

### Why Replication is Needed

```
┌─────────────────────────────────────────────────────────────────┐
│                    Main Purposes of Replication                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. High Availability                                           │
│     ┌─────────┐         ┌─────────┐                            │
│     │ Primary │ ──X──>  │ Replica │  ← When Primary fails,     │
│     │  (Down) │         │ (Active)│    Replica maintains       │
│     └─────────┘         └─────────┘    service                  │
│                                                                 │
│  2. Read Scalability                                            │
│     ┌─────────┐                                                │
│     │ Primary │────┬────> Replica 1 ←─── Read                  │
│     │ (Write) │    ├────> Replica 2 ←─── Read                  │
│     └─────────┘    └────> Replica 3 ←─── Read                  │
│                                                                 │
│  3. Geographical Distribution                                   │
│                                                                 │
│     Seoul ──────────────> Tokyo ──────────────> US-West        │
│     [Primary]            [Replica]             [Replica]       │
│     Latency: 0ms         Latency: ~30ms        Latency: ~100ms │
│                                                                 │
│  4. Data Protection                                             │
│     - Hardware failure protection                               │
│     - Data center failure protection                            │
│     - Disaster Recovery                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Types of Replication Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                Replication Architecture Comparison               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Single-Leader                                               │
│     ┌────────┐                                                 │
│     │ Leader │ ──Write──>                                      │
│     └───┬────┘                                                 │
│         │ Replicate                                            │
│     ┌───┴───────────┐                                          │
│     ▼       ▼       ▼                                          │
│  Follower Follower Follower                                    │
│                                                                 │
│  2. Multi-Leader                                                │
│     ┌────────┐     ┌────────┐     ┌────────┐                   │
│     │Leader 1│<───>│Leader 2│<───>│Leader 3│                   │
│     └───┬────┘     └───┬────┘     └───┬────┘                   │
│         │              │              │                        │
│      Follower       Follower       Follower                    │
│                                                                 │
│  3. Leaderless                                                  │
│       ┌──────┐   ┌──────┐   ┌──────┐                           │
│       │Node 1│   │Node 2│   │Node 3│                           │
│       └──────┘   └──────┘   └──────┘                           │
│           ▲          ▲          ▲                              │
│           └──────────┴──────────┘                              │
│              All nodes are equal                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Single-Leader Replication

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                  Single-Leader Replication                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Client                                                         │
│    │                                                            │
│    │ Write Request                                              │
│    ▼                                                            │
│  ┌─────────────────────────────────────────┐                    │
│  │              Leader (Primary)            │                    │
│  │  ┌─────────────────────────────────┐    │                    │
│  │  │         Transaction Log          │    │                    │
│  │  │  [1] INSERT INTO users...        │    │                    │
│  │  │  [2] UPDATE orders...            │    │                    │
│  │  │  [3] DELETE from cart...         │    │                    │
│  │  └─────────────────────────────────┘    │                    │
│  └─────────────────┬───────────────────────┘                    │
│                    │                                            │
│          ┌─────────┼─────────┐                                  │
│          │  Replication Log  │                                  │
│          ▼         ▼         ▼                                  │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐                      │
│  │ Follower1 │ │ Follower2 │ │ Follower3 │                      │
│  │ (Replica) │ │ (Replica) │ │ (Replica) │                      │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘                      │
│        │             │             │                            │
│        └─────────────┴─────────────┘                            │
│                      │                                          │
│                Read Requests                                    │
│                      ▲                                          │
│                   Clients                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Replication Methods

```
┌─────────────────────────────────────────────────────────────────┐
│                    Replication Methods Comparison                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Statement-Based Replication                                 │
│     ┌─────────────────────────────────────────┐                │
│     │  Leader: INSERT INTO users VALUES (...)  │                │
│     │            │                             │                │
│     │            ▼                             │                │
│     │  Follower: INSERT INTO users VALUES (...) │ ← SQL replay  │
│     └─────────────────────────────────────────┘                │
│     Problem: Non-deterministic functions like NOW(), RAND()     │
│                                                                 │
│  2. Write-Ahead Log (WAL) Shipping                              │
│     ┌─────────────────────────────────────────┐                │
│     │  Leader WAL:                             │                │
│     │  [Page 5, Offset 120, Data: 0x45AB...]   │                │
│     │            │                             │                │
│     │            ▼                             │                │
│     │  Follower: Apply identically byte by byte│ ← Low-level   │
│     └─────────────────────────────────────────┘   replication  │
│     Problem: Version compatibility (downtime on upgrade)        │
│                                                                 │
│  3. Row-Based (Logical) Replication                             │
│     ┌─────────────────────────────────────────┐                │
│     │  Change Log:                             │                │
│     │  {table: "users",                        │                │
│     │   op: "INSERT",                          │                │
│     │   new_row: {id:1, name:"Kim"}}           │ ← Logical     │
│     │            │                             │    changes     │
│     │            ▼                             │                │
│     │  Follower: Apply row data                │                │
│     └─────────────────────────────────────────┘                │
│     Advantages: Version compatible, flexible replication        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### PostgreSQL Streaming Replication Example

```sql
-- Primary configuration (postgresql.conf)
wal_level = replica
max_wal_senders = 10
wal_keep_size = 1GB

-- Primary: Create replication slot
SELECT * FROM pg_create_physical_replication_slot('replica1_slot');

-- Standby configuration (postgresql.conf)
primary_conninfo = 'host=primary-host port=5432 user=replicator'
primary_slot_name = 'replica1_slot'

-- Check replication status
SELECT
    client_addr,
    state,
    sent_lsn,
    write_lsn,
    flush_lsn,
    replay_lsn,
    pg_wal_lsn_diff(sent_lsn, replay_lsn) AS replication_lag
FROM pg_stat_replication;
```

### MySQL Replication Configuration Example

```sql
-- Master configuration (my.cnf)
-- server-id=1
-- log_bin=mysql-bin
-- binlog_format=ROW

-- Master: Create replication user
CREATE USER 'repl'@'%' IDENTIFIED BY 'password';
GRANT REPLICATION SLAVE ON *.* TO 'repl'@'%';
SHOW MASTER STATUS;

-- Slave configuration
CHANGE MASTER TO
    MASTER_HOST='master-host',
    MASTER_USER='repl',
    MASTER_PASSWORD='password',
    MASTER_LOG_FILE='mysql-bin.000001',
    MASTER_LOG_POS=154;

START SLAVE;
SHOW SLAVE STATUS\G
```

---

## 3. Multi-Leader Replication

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│               Multi-Leader Replication                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    Data Center A              Data Center B                     │
│  ┌─────────────────┐        ┌─────────────────┐                 │
│  │    Leader A     │ <────> │    Leader B     │                 │
│  │  ┌───────────┐  │        │  ┌───────────┐  │                 │
│  │  │ users: 1  │  │  Sync  │  │ users: 1  │  │                 │
│  │  │ orders: 5 │  │ <────> │  │ orders: 5 │  │                 │
│  │  └───────────┘  │        │  └───────────┘  │                 │
│  │       │         │        │       │         │                 │
│  │   ┌───┴───┐     │        │   ┌───┴───┐     │                 │
│  │   ▼       ▼     │        │   ▼       ▼     │                 │
│  │ Follower Follower│        │ Follower Follower│                │
│  └─────────────────┘        └─────────────────┘                 │
│         ▲                          ▲                            │
│         │                          │                            │
│    [Clients A]                [Clients B]                       │
│    Latency: ~1ms              Latency: ~1ms                     │
│                                                                 │
│  Advantages:                                                    │
│  - Low write latency at each data center                        │
│  - Can continue writing at other location on DC failure         │
│                                                                 │
│  Disadvantages:                                                 │
│  - Conflict resolution needed                                   │
│  - Increased complexity                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Write Conflict Problem

```
┌─────────────────────────────────────────────────────────────────┐
│                    Write Conflict Scenario                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Time ────────────────────────────────────────────────>         │
│                                                                 │
│  Leader A:                                                      │
│    T1: UPDATE users SET name='Kim' WHERE id=1                   │
│         │                                                       │
│         ▼ (replicate)                                           │
│                                                                 │
│  Leader B:                                                      │
│    T1: UPDATE users SET name='Lee' WHERE id=1                   │
│         │                                                       │
│         ▼ (replicate)                                           │
│                                                                 │
│  Result:                                                        │
│  ┌─────────────┐         ┌─────────────┐                       │
│  │  Leader A   │         │  Leader B   │                       │
│  │ name='Lee'? │   ≠     │ name='Kim'? │   Conflict!           │
│  └─────────────┘         └─────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Conflict Resolution Strategies

```
┌─────────────────────────────────────────────────────────────────┐
│                    Conflict Resolution Strategies                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Last Write Wins (LWW)                                       │
│     ┌─────────────────────────────────────────────┐            │
│     │  Write A: {value: "Kim", timestamp: 100}    │            │
│     │  Write B: {value: "Lee", timestamp: 105}    │            │
│     │                                             │            │
│     │  Result: "Lee" (timestamp 105 > 100)        │            │
│     └─────────────────────────────────────────────┘            │
│     Problem: Possible data loss                                 │
│                                                                 │
│  2. Version Vector (Vector Clock)                               │
│     ┌─────────────────────────────────────────────┐            │
│     │  Node A: {value: "Kim", version: [A:1, B:0]} │            │
│     │  Node B: {value: "Lee", version: [A:0, B:1]} │            │
│     │                                             │            │
│     │  Detect concurrent writes → Merge needed     │            │
│     └─────────────────────────────────────────────┘            │
│                                                                 │
│  3. CRDT (Conflict-free Replicated Data Type)                   │
│     ┌─────────────────────────────────────────────┐            │
│     │  G-Counter: Maintain counter per node        │            │
│     │  Node A: +3                                 │            │
│     │  Node B: +2                                 │            │
│     │  Total: 3 + 2 = 5 (automatic merge)         │            │
│     └─────────────────────────────────────────────┘            │
│                                                                 │
│  4. Custom Resolution Logic                                     │
│     ┌─────────────────────────────────────────────┐            │
│     │  Example: Shopping cart conflict             │            │
│     │  Cart A: [Item1, Item2]                     │            │
│     │  Cart B: [Item1, Item3]                     │            │
│     │  Merge: [Item1, Item2, Item3] (Union)       │            │
│     └─────────────────────────────────────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Replication Topologies

```
┌─────────────────────────────────────────────────────────────────┐
│                    Replication Topology Types                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Circular                                                    │
│       ┌───┐                                                    │
│       │ A │ ──────────┐                                        │
│       └─┬─┘           │                                        │
│         │           ┌─▼─┐                                      │
│         │           │ B │                                      │
│         │           └─┬─┘                                      │
│       ┌─▼─┐           │                                        │
│       │ D │ <─────────┘                                        │
│       └─┬─┘           │                                        │
│         └──> ┌───┐ <──┘                                        │
│              │ C │                                              │
│              └───┘                                              │
│     Risk of entire replication stopping on failure              │
│                                                                 │
│  2. Star                                                        │
│           ┌───┐                                                │
│           │ B │                                                │
│           └─┬─┘                                                │
│       ┌───┐ │ ┌───┐                                            │
│       │ A │─┼─│ C │                                            │
│       └───┘ │ └───┘                                            │
│           ┌─▼─┐                                                │
│           │Hub│  ← Vulnerable when central node fails          │
│           └───┘                                                │
│                                                                 │
│  3. All-to-All                                                  │
│       ┌───┐     ┌───┐                                          │
│       │ A │ ←──→│ B │                                          │
│       └─┬─┘     └─┬─┘                                          │
│         │    ╲ ╱  │                                            │
│         │     ╳   │                                            │
│         │    ╱ ╲  │                                            │
│       ┌─▼─┐     ┌─▼─┐                                          │
│       │ D │ ←──→│ C │                                          │
│       └───┘     └───┘                                          │
│     High fault tolerance, complex management                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Leaderless Replication

### Dynamo-Style Replication

```
┌─────────────────────────────────────────────────────────────────┐
│              Leaderless Replication                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Characteristics:                                               │
│  - All nodes can read/write                                     │
│  - Client sends requests to multiple nodes simultaneously       │
│  - Used in Amazon Dynamo, Cassandra, Riak, etc.                 │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                        Client                            │   │
│  │                          │                               │   │
│  │           Write: key=X, value=V                          │   │
│  │           ┌──────────────┼──────────────┐                │   │
│  │           ▼              ▼              ▼                │   │
│  │       ┌──────┐      ┌──────┐      ┌──────┐              │   │
│  │       │Node 1│      │Node 2│      │Node 3│              │   │
│  │       │ X=V  │      │ X=V  │      │(Down)│              │   │
│  │       │ ✓    │      │ ✓    │      │  ✗   │              │   │
│  │       └──────┘      └──────┘      └──────┘              │   │
│  │                                                          │   │
│  │   N=3 (total replicas), W=2 (required write successes)   │   │
│  │   2 nodes succeeded → Write successful!                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Read Repair and Anti-Entropy

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Repair Mechanisms                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Read Repair (Repair on Read)                                │
│                                                                 │
│     Client                                                      │
│        │ Read key=X                                             │
│        ├────────────────────────────────────┐                   │
│        ▼              ▼                     ▼                   │
│    ┌──────┐      ┌──────┐              ┌──────┐                │
│    │Node 1│      │Node 2│              │Node 3│                │
│    │X=V2  │      │X=V2  │              │X=V1  │ ← Stale version │
│    │ver:2 │      │ver:2 │              │ver:1 │                │
│    └──────┘      └──────┘              └──────┘                │
│        │              │                     │                   │
│        └──────────────┴─────────────────────┘                   │
│                       │                                         │
│                       ▼                                         │
│              Client: Compare versions                           │
│              Return latest version V2                           │
│              Request V2 update to Node 3 ←── Read Repair        │
│                                                                 │
│  2. Anti-Entropy Process (Background Sync)                      │
│                                                                 │
│    ┌──────────────────────────────────────────────────────┐    │
│    │           Background Anti-Entropy Process            │    │
│    │                                                      │    │
│    │  Node 1          Node 2          Node 3              │    │
│    │  ┌────┐         ┌────┐          ┌────┐              │    │
│    │  │ A  │         │ A  │          │ A  │              │    │
│    │  │ B  │  <────> │ B  │  <────>  │ B  │              │    │
│    │  │ C  │ Merkle  │ C  │  Merkle  │ C* │ ← Mismatch   │    │
│    │  │ D  │  Tree   │ D  │   Tree   │ D  │              │    │
│    │  └────┘ Compare └────┘  Compare └────┘              │    │
│    │                                                      │    │
│    │  Efficient difference detection and sync via Merkle  │    │
│    │  Tree                                                │    │
│    └──────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Sloppy Quorum and Hinted Handoff

```
┌─────────────────────────────────────────────────────────────────┐
│              Sloppy Quorum & Hinted Handoff                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Strict Quorum: Read/write only from designated nodes           │
│                                                                 │
│  Sloppy Quorum: Other nodes substitute when some fail           │
│                                                                 │
│  Normal situation:                                              │
│  ┌──────┐   ┌──────┐   ┌──────┐                                │
│  │Node 1│   │Node 2│   │Node 3│   ← Home nodes for key X       │
│  └──────┘   └──────┘   └──────┘                                │
│                                                                 │
│  When Node 3 fails:                                             │
│  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐                     │
│  │Node 1│   │Node 2│   │(Down)│   │Node 4│                     │
│  │ X=V  │   │ X=V  │   │      │   │ X=V* │ ← Store with Hint   │
│  └──────┘   └──────┘   └──────┘   └──────┘                     │
│                                                                 │
│  *Hinted Handoff:                                               │
│  ┌──────────────────────────────────────────┐                  │
│  │  Data stored in Node 4:                   │                  │
│  │  {                                        │                  │
│  │    key: "X",                              │                  │
│  │    value: "V",                            │                  │
│  │    hint: "Node 3"  ← Original target node │                  │
│  │  }                                        │                  │
│  │                                           │                  │
│  │  When Node 3 recovers → Transfer then     │                  │
│  │  delete                                   │                  │
│  └──────────────────────────────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Synchronous/Asynchronous Replication

### Synchronous Replication

```
┌─────────────────────────────────────────────────────────────────┐
│                    Synchronous Replication                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Client          Leader           Follower 1      Follower 2    │
│    │               │                  │               │         │
│    │  Write X=V    │                  │               │         │
│    │──────────────>│                  │               │         │
│    │               │  Replicate       │               │         │
│    │               │─────────────────>│               │         │
│    │               │─────────────────────────────────>│         │
│    │               │                  │               │         │
│    │               │   ACK            │               │         │
│    │               │<─────────────────│               │         │
│    │               │<─────────────────────────────────│         │
│    │               │                  │               │         │
│    │   ACK         │  (After all replicated)         │         │
│    │<──────────────│                  │               │         │
│    │               │                  │               │         │
│                                                                 │
│  Advantages:                                                    │
│  - Strong consistency guaranteed                                │
│  - No data loss                                                 │
│                                                                 │
│  Disadvantages:                                                 │
│  - Slow write latency (wait for all replicas)                   │
│  - Cannot write when replica fails                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Asynchronous Replication

```
┌─────────────────────────────────────────────────────────────────┐
│                   Asynchronous Replication                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Client          Leader           Follower 1      Follower 2    │
│    │               │                  │               │         │
│    │  Write X=V    │                  │               │         │
│    │──────────────>│                  │               │         │
│    │               │                  │               │         │
│    │   ACK         │  (Immediate response)           │         │
│    │<──────────────│                  │               │         │
│    │               │                  │               │         │
│    │               │  Replicate (background)         │         │
│    │               │─────────────────>│               │         │
│    │               │─────────────────────────────────>│         │
│    │               │                  │               │         │
│                                                                 │
│  Advantages:                                                    │
│  - Fast write response                                          │
│  - Replica failure doesn't affect writes                        │
│                                                                 │
│  Disadvantages:                                                 │
│  - Possible data loss on leader failure                         │
│  - Read consistency issues (stale read)                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Semi-Synchronous Replication

```
┌─────────────────────────────────────────────────────────────────┐
│                  Semi-Synchronous Replication                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Replicate synchronously to at least 1 replica, async for rest  │
│                                                                 │
│  Client          Leader        Sync Follower   Async Followers  │
│    │               │                │               │           │
│    │  Write X=V    │                │               │           │
│    │──────────────>│                │               │           │
│    │               │  Replicate     │               │           │
│    │               │───────────────>│               │           │
│    │               │  ACK           │               │           │
│    │               │<───────────────│               │           │
│    │   ACK         │                │               │           │
│    │<──────────────│                │               │           │
│    │               │  Replicate (async)             │           │
│    │               │───────────────────────────────>│           │
│    │               │                │               │           │
│                                                                 │
│  MySQL Semi-Sync Configuration:                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  -- Master                                                 │ │
│  │  SET GLOBAL rpl_semi_sync_master_enabled = 1;              │ │
│  │  SET GLOBAL rpl_semi_sync_master_timeout = 10000; -- 10s   │ │
│  │                                                            │ │
│  │  -- Slave                                                  │ │
│  │  SET GLOBAL rpl_semi_sync_slave_enabled = 1;               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Replication Lag and Consistency Issues

### Replication Lag

```
┌─────────────────────────────────────────────────────────────────┐
│                    Replication Lag Scenario                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Time ────────────────────────────────────────────────>         │
│                                                                 │
│  T0      T1       T2       T3       T4       T5                 │
│  │       │        │        │        │        │                  │
│  Leader: Write    │        │        │        │                  │
│          X=1      │        │        │        │                  │
│           │       │        │        │        │                  │
│           │       ▼        │        │        │                  │
│  Follower:        X=1      │        │        │                  │
│                   (delay)  │        │        │                  │
│                            │        │        │                  │
│  Client A:        Read from Leader  │        │                  │
│                   X=1 ✓    │        │        │                  │
│                            │        │        │                  │
│  Client B:                 Read from Follower│                  │
│                            X=??? ← At T2,                       │
│                                    Follower not yet replicated  │
│                                    → Stale Read!                │
│                                                                 │
│  Replication Lag = T1(Leader Write) ~ T2(Follower Apply) time   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Consistency Guarantee Levels

```
┌─────────────────────────────────────────────────────────────────┐
│                    Consistency Guarantee Levels                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Read-Your-Writes                                            │
│     ┌─────────────────────────────────────────────────────┐    │
│     │  User A: Write profile photo                        │    │
│     │          │                                          │    │
│     │          ▼                                          │    │
│     │  User A: Read profile → Should see new photo        │    │
│     │                                                     │    │
│     │  Solutions:                                          │    │
│     │  - Always read own changes from Leader              │    │
│     │  - Or timestamp-based read position                 │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                 │
│  2. Monotonic Reads                                             │
│     ┌─────────────────────────────────────────────────────┐    │
│     │  T1: Read from Follower A → X=2                     │    │
│     │  T2: Read from Follower B → X=1  ← Problem!         │    │
│     │                                                     │    │
│     │  Appears as if time went backwards                  │    │
│     │                                                     │    │
│     │  Solutions:                                          │    │
│     │  - Read from same replica per user                  │    │
│     │  - Version-based reads                              │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                 │
│  3. Consistent Prefix Reads (Causality Preservation)            │
│     ┌─────────────────────────────────────────────────────┐    │
│     │  Chat:                                              │    │
│     │  A: "How's the weather in Seoul?"  (T1)             │    │
│     │  B: "It's sunny!"                  (T2)             │    │
│     │                                                     │    │
│     │  With replication lag:                              │    │
│     │  What C sees:                                       │    │
│     │  B: "It's sunny!"  ← T2 arrives first               │    │
│     │  A: "How's the weather in Seoul?"  ← T1 arrives     │    │
│     │                                        later        │    │
│     │                                                     │    │
│     │  Solutions:                                          │    │
│     │  - Store causally related data in same partition    │    │
│     │  - Causality tracking                               │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Replication Lag Monitoring

```sql
-- PostgreSQL replication lag check
SELECT
    client_addr,
    state,
    pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) AS lag_bytes,
    EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) AS lag_seconds
FROM pg_stat_replication;

-- MySQL replication lag check
SHOW SLAVE STATUS\G
-- Check Seconds_Behind_Master value

-- Lag-based routing logic (application)
/*
def get_connection(query_type, max_lag_seconds=5):
    if query_type == 'write':
        return master_connection

    for replica in replicas:
        if replica.lag_seconds < max_lag_seconds:
            return replica.connection

    # When all replicas have high lag, read from master
    return master_connection
*/
```

---

## 7. Failure Recovery and Leader Election

### Follower Failure Recovery

```
┌─────────────────────────────────────────────────────────────────┐
│                    Follower Failure Recovery                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Catch-up Recovery                                           │
│                                                                 │
│     Normal state:                                               │
│     Leader:    [1] [2] [3] [4] [5] [6]                          │
│     Follower:  [1] [2] [3]                                      │
│                         ▲                                       │
│                         └── Remembers last applied position     │
│                                                                 │
│     After Follower recovers:                                    │
│     Leader:    [1] [2] [3] [4] [5] [6] [7] [8]                  │
│     Follower:  [1] [2] [3] → [4] [5] [6] [7] [8]               │
│                             ↑                                   │
│                             Catch-up                            │
│                                                                 │
│  2. Recovery Process                                            │
│     ┌────────────────────────────────────────────────────┐     │
│     │  1. Follower restarts                              │     │
│     │  2. Check last applied LSN/binlog position         │     │
│     │  3. Request logs after that position from Leader   │     │
│     │  4. Catch up while applying logs                   │     │
│     │  5. Resume real-time replication                   │     │
│     └────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Leader Failure and Failover

```
┌─────────────────────────────────────────────────────────────────┐
│                    Leader Failure Recovery (Failover)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Failover Process:                                              │
│                                                                 │
│  Step 1: Failure Detection                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │   ┌────────┐        Heartbeat failure                    │   │
│  │   │ Leader │ ────X──── (3 consecutive)                   │   │
│  │   │ (Down) │                                             │   │
│  │   └────────┘        Timeout: 30 seconds                  │   │
│  │                                                          │   │
│  │   Followers: "No Leader response!" → Start Failover      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Step 2: New Leader Election                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │   Follower A: LSN = 1000                                 │   │
│  │   Follower B: LSN = 1050  ← Most up-to-date → New Leader │   │
│  │   Follower C: LSN = 980                                  │   │
│  │                                                          │   │
│  │   Election criteria:                                     │   │
│  │   1. Node with most up-to-date data                      │   │
│  │   2. Node priority                                       │   │
│  │   3. Consensus algorithm (Raft, Paxos)                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Step 3: Client Reconnection                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │   Clients ───── (old) Leader: 10.0.0.1                   │   │
│  │           ╲                                              │   │
│  │            ╲─── (new) Leader: 10.0.0.2                   │   │
│  │                                                          │   │
│  │   Methods:                                               │   │
│  │   - DNS update                                           │   │
│  │   - Virtual IP (VIP) transfer                            │   │
│  │   - Proxy/Load Balancer configuration change             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Failover Problems

```
┌─────────────────────────────────────────────────────────────────┐
│                    Failover Problems                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Data Loss (with async replication)                          │
│     ┌────────────────────────────────────────────────────┐     │
│     │  Old Leader: [1][2][3][4][5]                       │     │
│     │                          ↑ Not replicated          │     │
│     │  New Leader:  [1][2][3][4]                         │     │
│     │                                                    │     │
│     │  Write [5] is lost!                                │     │
│     │                                                    │     │
│     │  When Old Leader recovers: What about [5]?         │     │
│     │  → Typically discarded (conflict prevention)       │     │
│     └────────────────────────────────────────────────────┘     │
│                                                                 │
│  2. Split Brain (Two Leaders problem)                           │
│     ┌────────────────────────────────────────────────────┐     │
│     │        Network Partition                           │     │
│     │                                                    │     │
│     │  ┌────────┐       ╳        ┌────────┐             │     │
│     │  │Old     │                │New     │             │     │
│     │  │Leader  │  Network split │Leader  │             │     │
│     │  └────────┘                └────────┘             │     │
│     │       ▲                          ▲                │     │
│     │  Clients A                  Clients B             │     │
│     │  (continue writing)         (continue writing)    │     │
│     │                                                    │     │
│     │  → Data inconsistency occurs!                      │     │
│     │                                                    │     │
│     │  Solutions:                                        │     │
│     │  - Fencing (STONITH: Shoot The Other Node In Head)│     │
│     │  - Force shutdown Old Leader                       │     │
│     └────────────────────────────────────────────────────┘     │
│                                                                 │
│  3. Timeout Configuration Dilemma                               │
│     ┌────────────────────────────────────────────────────┐     │
│     │  Timeout too short:                                │     │
│     │  - Unnecessary failover on network delay/load      │     │
│     │  - System instability                              │     │
│     │                                                    │     │
│     │  Timeout too long:                                 │     │
│     │  - Delayed recovery on actual failure              │     │
│     │  - Increased service downtime                      │     │
│     └────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Leader Election Algorithm

```
┌─────────────────────────────────────────────────────────────────┐
│                    Raft Leader Election                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  States: Follower, Candidate, Leader                            │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                                                           │  │
│  │    ┌──────────┐     Election      ┌───────────┐          │  │
│  │    │          │     Timeout       │           │          │  │
│  │    │ Follower │ ─────────────────>│ Candidate │          │  │
│  │    │          │                   │           │          │  │
│  │    └──────────┘                   └─────┬─────┘          │  │
│  │         ▲                               │                │  │
│  │         │ Discovers                     │ Receives       │  │
│  │         │ current                       │ majority       │  │
│  │         │ leader                        │ votes          │  │
│  │         │                               ▼                │  │
│  │         │                         ┌──────────┐           │  │
│  │         └─────────────────────────│  Leader  │           │  │
│  │                                   │          │           │  │
│  │                                   └──────────┘           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Election Process:                                              │
│  1. Follower becomes Candidate if no heartbeat received         │
│  2. Increment Term number                                       │
│  3. Vote for self + RequestVote to other nodes                  │
│  4. Leader elected when majority votes received                 │
│  5. Leader sends periodic heartbeats                            │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Example: 5-node cluster                                  │  │
│  │                                                           │  │
│  │  Node A (Candidate, Term 5)                               │  │
│  │    │                                                      │  │
│  │    ├── RequestVote ──> Node B: Vote YES                   │  │
│  │    ├── RequestVote ──> Node C: Vote YES                   │  │
│  │    ├── RequestVote ──> Node D: Vote NO (already voted)    │  │
│  │    └── RequestVote ──> Node E: Vote YES                   │  │
│  │                                                           │  │
│  │  3/5 = Majority → Node A is Leader!                       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Quorum and Consistency Levels

### Quorum Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                    Quorum Concept                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  N = Number of replicas (Total Replicas)                        │
│  W = Number of responses needed for write success (Write Quorum)│
│  R = Number of responses needed for read success (Read Quorum)  │
│                                                                 │
│  Strong consistency guarantee condition: R + W > N              │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                                                        │    │
│  │   When N=3, W=2, R=2:                                  │    │
│  │                                                        │    │
│  │   On write:                                            │    │
│  │   [Node1: X=V]  [Node2: X=V]  [Node3: ?]              │    │
│  │        ✓             ✓           (pending)             │    │
│  │                                                        │    │
│  │   On read:                                             │    │
│  │   [Node1: X=V]  [Node2: ?]  [Node3: X=V]              │    │
│  │        ✓          (skip)         ✓                    │    │
│  │                                                        │    │
│  │   R(2) + W(2) = 4 > N(3)                              │    │
│  │   → At least 1 node has the latest write!             │    │
│  │                                                        │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│                 ┌─────────────────┐                            │
│    Write Set    │                 │    Read Set                │
│    (W=2)        │   ┌─────────┐   │    (R=2)                   │
│  ┌─────────┐    │   │ Node 1  │   │    ┌─────────┐            │
│  │ Node 1  │────┼───│ (overlap)│───┼────│ Node 1  │            │
│  │ Node 2  │    │   └─────────┘   │    │ Node 3  │            │
│  └─────────┘    │   (at least 1)  │    └─────────┘            │
│                 └─────────────────┘                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Quorum Configuration Characteristics

```
┌─────────────────────────────────────────────────────────────────┐
│                    Quorum Configuration Strategies               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Based on N = 3 replicas                                        │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Config 1: W=1, R=3 (Write priority)                       │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │  Write: Only 1 node needs to succeed → Fast write   │  │ │
│  │  │  Read: Must read all 3 → Slow read                  │  │ │
│  │  │                                                     │  │ │
│  │  │  Use case: Log collection, IoT data                 │  │ │
│  │  └─────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Config 2: W=3, R=1 (Read priority)                        │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │  Write: All 3 must succeed → Slow write             │  │ │
│  │  │  Read: Only 1 needed → Fast read                    │  │ │
│  │  │                                                     │  │ │
│  │  │  Use case: Catalogs, reference data                 │  │ │
│  │  └─────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Config 3: W=2, R=2 (Balanced)                             │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │  Write: 2 nodes must succeed                        │  │ │
│  │  │  Read: Read from 2 nodes                            │  │ │
│  │  │                                                     │  │ │
│  │  │  Common balanced configuration                      │  │ │
│  │  │  Tolerates 1 node failure                           │  │ │
│  │  └─────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Fault tolerance: min(N-W, N-R) node failures tolerated         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Cassandra Consistency Levels

```
┌─────────────────────────────────────────────────────────────────┐
│                Cassandra Consistency Levels                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Write Consistency Levels:                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  ANY         : 1 node (including hints) → Lowest           │ │
│  │                consistency                                 │ │
│  │  ONE         : 1 replica                                   │ │
│  │  TWO         : 2 replicas                                  │ │
│  │  THREE       : 3 replicas                                  │ │
│  │  QUORUM      : (RF/2)+1 replicas                           │ │
│  │  LOCAL_QUORUM: (RF/2)+1 in local DC                        │ │
│  │  EACH_QUORUM : (RF/2)+1 in each DC                         │ │
│  │  ALL         : All replicas → Highest consistency          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Example: RF=3 cluster                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  QUORUM calculation: (3/2)+1 = 2                           │ │
│  │                                                            │ │
│  │  Write CL=QUORUM:   2 node ACKs required                   │ │
│  │  Read CL=QUORUM:    Read from 2 nodes                      │ │
│  │                                                            │ │
│  │  R(2) + W(2) = 4 > N(3) → Strong consistency!              │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Code Example:                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  -- CQL                                                    │ │
│  │  CONSISTENCY QUORUM;                                       │ │
│  │  INSERT INTO users (id, name) VALUES (1, 'Kim');           │ │
│  │                                                            │ │
│  │  -- Or per-query configuration                             │ │
│  │  SELECT * FROM users WHERE id = 1                          │ │
│  │  USING CONSISTENCY LOCAL_QUORUM;                           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Consistency vs Availability Trade-off

```
┌─────────────────────────────────────────────────────────────────┐
│              Consistency vs Availability Trade-off               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                  Consistency                                    │
│                        ▲                                        │
│                        │                                        │
│        ALL ●──────────┤                                        │
│                       │                                        │
│     QUORUM ●─────────┤     Balance point                       │
│                      │     (Recommended)                       │
│        TWO ●────────┤                                          │
│                     │                                          │
│        ONE ●───────┤                                           │
│                    │                                           │
│        ANY ●──────┼─────────────────────────────────>          │
│                              Availability                      │
│                                                                 │
│  Recommended settings by scenario:                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Financial transactions: W=ALL, R=ALL  (Consistency first) │ │
│  │  Social feeds:          W=ONE, R=ONE   (Availability first)│ │
│  │  E-commerce:            W=QUORUM, R=QUORUM (Balanced)      │ │
│  │  Analytics data:        W=ONE, R=ONE   (Eventual           │ │
│  │                                         consistency OK)    │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Practice Problems

### Practice 1: Choosing Replication Strategy

Choose the appropriate replication strategy for the following scenarios and explain your reasoning.

```
Scenarios:
1. Social media service with global users
2. Bank system processing financial transactions
3. Real-time chat application
4. Log collection and analysis system

Options:
A. Single-leader + Synchronous replication
B. Single-leader + Asynchronous replication
C. Multi-leader replication
D. Leaderless replication
```

**Sample Answer:**

```
1. Social Media: C (Multi-leader)
   - Geographically distributed users
   - Low latency needed in each region
   - Slight consistency delay acceptable

2. Bank System: A (Single-leader + Sync)
   - Strong consistency required
   - Data loss unacceptable
   - Write latency tolerable

3. Real-time Chat: B (Single-leader + Async)
   - Fast response needed
   - Message order important
   - Slight data loss acceptable

4. Log Collection: D (Leaderless)
   - High write throughput needed
   - Eventual consistency sufficient
   - High availability important
```

### Practice 2: Quorum Calculation

```
Problem:
In an N=5 replica cluster:

1. What W, R values for strong consistency with max availability?
2. With W=3, R=2, how many node failures tolerated?
3. How to maximize write availability while maintaining strong consistency?
```

**Sample Answer:**

```
1. W=3, R=3 (or W=2, R=4, etc.)
   - R + W > N (3+3=6 > 5) ✓
   - Write: Tolerates 2 node failures
   - Read: Tolerates 2 node failures

2. W=3, R=2 analysis:
   - R + W = 5 = N (Not strong consistency!)
   - Strictly speaking R + W > N needed for strong consistency
   - Write availability: 5-3 = 2 node failures tolerated
   - Read availability: 5-2 = 3 node failures tolerated

3. Maximize write availability:
   - W=1, R=5 (Write: 4 failures tolerated)
   - However, read requires all nodes (low availability)
```

### Practice 3: Failover Scenario

```
Problem:
In a MySQL master-slave configuration using asynchronous replication,
when master fails:

1. Slave A: binlog position 1000
2. Slave B: binlog position 950
3. Master last binlog position: 1020

Answer the following:
1. Which slave should be promoted to new master?
2. Maximum how many transactions could be lost?
3. How should the old master be handled when recovered?
```

**Sample Answer:**

```
1. Promote Slave A to new master
   - Position 1000 > 950
   - Has most up-to-date data

2. Maximum lost transactions:
   - 1020 - 1000 = 20 transactions could be lost

3. When old master recovers:
   - Reconfigure as slave
   - Discard position 1000 ~ 1020 data (conflict prevention)
   - Start replication from new master
```

### Practice 4: Conflict Resolution Design

```
Problem:
In a multi-leader e-commerce system, when updating
the same product's inventory simultaneously:

Leader A (Seoul): inventory = 100 - 5 = 95
Leader B (Tokyo): inventory = 100 - 3 = 97

How can this conflict be resolved?
Propose several methods and explain pros/cons.
```

**Sample Answer:**

```
Method 1: LWW (Last Write Wins)
- Apply write with later timestamp
- Pros: Simple implementation
- Cons: Data loss (5 or 3 sales missing)

Method 2: CRDT (Counter)
- Record sales per leader: A: -5, B: -3
- Final inventory = 100 - 5 - 3 = 92
- Pros: No data loss
- Cons: Inventory could go negative

Method 3: Distributed Lock (Not recommended)
- Acquire distributed lock before write
- Cons: Increased latency, complexity

Method 4: Route to Single Leader
- Inventory writes only to specific leader
- Pros: Prevents conflicts at source
- Cons: Potential overload on that leader

Recommended: CRDT + Inventory threshold alerts
- Alert/adjust when negative inventory occurs
```

### Practice 5: Interview Question

```
Interviewer: "How would you implement Read-Your-Writes consistency?"

Requirements:
- User must always be able to read their own writes
- Asynchronous replication environment
- Multiple replicas present

Propose a design approach.
```

**Sample Answer:**

```
Method 1: Read from Leader
- User reads recently written data from Leader
- Implementation: Use Leader for certain time (e.g., 1 min) after write

Method 2: Timestamp-based
- Client remembers last write timestamp
- Include timestamp in read request
- Wait until replica is caught up to that timestamp

Method 3: Replication Position-based
- Return LSN (Log Sequence Number) after write
- Read from replica that has applied at least that LSN

Implementation Example (Method 2):
```python
class Client:
    def __init__(self):
        self.last_write_timestamp = 0

    def write(self, data):
        result = leader.write(data)
        self.last_write_timestamp = result.timestamp

    def read(self, key):
        return db.read(
            key,
            min_timestamp=self.last_write_timestamp
        )
```

---

## 10. References

### Essential Books

1. **Designing Data-Intensive Applications** - Martin Kleppmann
   - Chapter 5: Replication
   - Chapter 9: Consistency and Consensus

2. **Database Internals** - Alex Petrov
   - Part II: Distributed Systems

### Database-Specific Documentation

1. **PostgreSQL Streaming Replication**
   - https://www.postgresql.org/docs/current/warm-standby.html

2. **MySQL Replication**
   - https://dev.mysql.com/doc/refman/8.0/en/replication.html

3. **Cassandra Documentation**
   - https://cassandra.apache.org/doc/latest/cassandra/architecture/dynamo.html

### Papers

1. **Dynamo: Amazon's Highly Available Key-value Store** (2007)
2. **In Search of an Understandable Consensus Algorithm (Raft)** (2014)
3. **Chain Replication for Supporting High Throughput and Availability** (2004)

---

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    Key Concepts Summary                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Replication Types:                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Single-leader: Write only to leader, read from         │   │
│  │                 followers too                            │   │
│  │  Multi-leader: Write to multiple nodes, conflict         │   │
│  │                resolution needed                         │   │
│  │  Leaderless: All nodes equal, quorum for consistency     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Sync/Async:                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Sync: Strong consistency, slow writes, can't write on  │   │
│  │        failure                                           │   │
│  │  Async: Fast writes, possible data loss                  │   │
│  │  Semi-sync: At least 1 sync + rest async (balanced)      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Quorum:                                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  R + W > N guarantees strong consistency                 │   │
│  │  Fault tolerance = min(N-W, N-R)                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Interview Key Points:                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. Trade-offs by replication type                       │   │
│  │  2. Failover process and problems                        │   │
│  │  3. Quorum calculation and consistency levels            │   │
│  │  4. Split Brain prevention strategies                    │   │
│  │  5. Read-Your-Writes implementation methods              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

**Previous**: [Database Scaling](./08_Database_Scaling.md) | **Next**: [Data Consistency Patterns](./10_Data_Consistency_Patterns.md)
