# Lesson 9: Replication Strategies

[Overview](./00_Overview.md) | [Previous: Distributed Transactions](./08_Distributed_Transactions.md) | [Next: CRDTs and Eventual Consistency](./10_CRDTs_and_Eventual_Consistency.md)

---

## Learning Objectives

- Understand the fundamental motivations for data replication: fault tolerance, latency reduction, and throughput scaling
- Compare and contrast single-leader, multi-leader, and leaderless replication architectures with their trade-offs
- Analyze quorum-based consistency guarantees and identify when they break down
- Implement conflict detection and resolution strategies for multi-leader and leaderless systems
- Evaluate chain replication as an alternative to quorum-based approaches

---

## 1. Why Replicate Data?

Replication — maintaining copies of the same data on multiple machines — is one of the most fundamental techniques in distributed systems. Before diving into specific strategies, we must clearly understand *why* we replicate.

### 1.1 Fault Tolerance

Hardware fails. Disks corrupt, servers crash, network links go down, entire data centers lose power. Without replication, a single disk failure can destroy data permanently.

**Mean Time Between Failures (MTBF)** for a commodity disk is roughly 3–5 years. In a cluster of 10,000 disks, that means multiple disk failures *per day*.

```
Single copy:
  P(data loss in 1 year) = P(disk failure) ≈ 20-33%

Three replicas on independent disks:
  P(data loss in 1 year) = P(all three fail) ≈ (0.25)^3 ≈ 1.5%
  (Much lower with prompt replacement: ~0.0001%)
```

The arithmetic is stark: without replication, data loss is inevitable at scale.

### 1.2 Latency Reduction

Physics imposes hard limits. The speed of light in fiber is roughly 200,000 km/s, meaning a round trip from New York to Tokyo (~10,800 km one way) takes at minimum ~108 ms. Placing replicas in multiple geographic regions lets users read from a nearby copy.

```
┌──────────┐         ┌──────────┐         ┌──────────┐
│ US-East  │◄───────►│ EU-West  │◄───────►│ AP-Tokyo │
│ Replica  │  ~70ms  │ Replica  │  ~120ms │ Replica  │
└──────────┘         └──────────┘         └──────────┘
     ▲                                         ▲
     │ <5ms                              <5ms  │
  US User                              JP User
```

### 1.3 Throughput Scaling (Read Scaling)

A single server can handle a finite number of read queries per second. By directing reads to replicas, we can horizontally scale read throughput nearly linearly.

| Configuration | Read Throughput | Write Throughput |
|---|---|---|
| Single node | 10,000 QPS | 10,000 QPS |
| 1 leader + 4 replicas | ~50,000 QPS (reads) | 10,000 QPS (writes) |
| 1 leader + 9 replicas | ~100,000 QPS (reads) | 10,000 QPS (writes) |

**Key trade-off**: replication improves read throughput but does *not* improve write throughput (every write must be applied to all replicas). Write scaling requires *partitioning* (Lesson 11).

### 1.4 The Fundamental Trade-Off

Replication creates the problem of **keeping replicas in sync**. Every replication strategy must answer:

1. **How are writes propagated?** Synchronously or asynchronously?
2. **How are conflicts handled?** What if two replicas accept conflicting writes?
3. **What consistency guarantees do clients observe?** Can a client read stale data?

The remainder of this lesson explores the major strategies for answering these questions.

---

## 2. Single-Leader Replication

Single-leader (also called primary-backup or master-slave) replication is the most common strategy. One designated node — the **leader** — accepts all writes. The leader sends a replication stream (write-ahead log, logical log, or statement-based log) to **followers** (replicas).

```
  Writes                 Reads
    │                   ┌──► Follower 1
    ▼                   │
┌────────┐   Replication│   ┌──────────┐
│ Leader │──────────────┼──►│Follower 2│
└────────┘              │   └──────────┘
    │                   │
    │                   └──► Follower 3
    ▼
  Reads
```

### 2.1 Synchronous vs Asynchronous Replication

The critical design choice is whether the leader waits for followers to confirm before acknowledging the write to the client.

#### Synchronous Replication

```
Client        Leader        Follower
  │───write───►│               │
  │            │───replicate──►│
  │            │◄──────ack─────│
  │◄───ack─────│               │
```

**Guarantee**: If the leader confirms a write, the follower has a durable copy.

**Problem**: If the follower is slow or unreachable, the leader is *blocked*. Write latency becomes the latency of the *slowest* follower.

```python
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReplicaState:
    """Simulates a follower replica with variable latency."""
    name: str
    is_available: bool = True
    latency_ms: float = 5.0

    def apply_write(self, key: str, value: str) -> bool:
        """Simulate applying a write. Returns success/failure."""
        if not self.is_available:
            return False
        time.sleep(self.latency_ms / 1000.0)
        return True


def synchronous_write(leader_data: dict, replicas: list[ReplicaState],
                      key: str, value: str, timeout_s: float = 5.0) -> bool:
    """
    Synchronous replication: wait for ALL followers before acknowledging.
    Blocked if any follower is slow or down.
    """
    # Step 1: Apply to leader
    leader_data[key] = value

    # Step 2: Wait for ALL followers
    for replica in replicas:
        start = time.time()
        success = replica.apply_write(key, value)
        elapsed = time.time() - start

        if not success or elapsed > timeout_s:
            # Synchronous replication fails if any follower is unreachable
            print(f"  [SYNC] Follower {replica.name} failed — write BLOCKED")
            return False
        print(f"  [SYNC] Follower {replica.name} acked in {elapsed*1000:.1f}ms")

    return True
```

#### Asynchronous Replication

```
Client        Leader        Follower
  │───write───►│               │
  │◄───ack─────│               │
  │            │───replicate──►│  (happens later)
  │            │◄──────ack─────│
```

**Guarantee**: The leader acknowledges immediately. Writes are fast.

**Problem**: If the leader crashes before replication completes, acknowledged writes are **lost**. Followers may serve stale data.

```python
import threading
import queue
from typing import Any


class AsyncReplicationLog:
    """Asynchronous replication via a background thread."""

    def __init__(self, replicas: list[ReplicaState]):
        self.replicas = replicas
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._worker = threading.Thread(target=self._replication_loop, daemon=True)
        self._worker.start()

    def enqueue_write(self, key: str, value: str) -> None:
        """Enqueue a write for asynchronous replication. Returns immediately."""
        self._queue.put((key, value))

    def _replication_loop(self) -> None:
        """Background loop: drain the queue and replicate to followers."""
        while True:
            key, value = self._queue.get()
            for replica in self.replicas:
                try:
                    replica.apply_write(key, value)
                except Exception as e:
                    # In practice: retry with exponential backoff
                    print(f"  [ASYNC] Replication to {replica.name} failed: {e}")
            self._queue.task_done()


def async_write(leader_data: dict, repl_log: AsyncReplicationLog,
                key: str, value: str) -> bool:
    """
    Asynchronous replication: leader acks immediately.
    Replication happens in background — fast but risks data loss.
    """
    leader_data[key] = value
    repl_log.enqueue_write(key, value)  # non-blocking
    return True  # ack to client before followers confirm
```

### 2.2 Semi-Synchronous Replication

A practical compromise used by MySQL, PostgreSQL, and others: one follower is synchronous, the rest are asynchronous. This guarantees the write is on at least two nodes before acknowledging.

```
Client        Leader       Sync Follower    Async Followers
  │───write───►│               │                  │
  │            │───replicate──►│                  │
  │            │◄──────ack─────│                  │
  │◄───ack─────│               │                  │
  │            │───replicate──────────────────────►│ (background)
```

If the synchronous follower becomes unavailable, another follower is promoted to the synchronous role. This is sometimes called **semi-synchronous** replication.

```python
def semi_sync_write(leader_data: dict, sync_replica: ReplicaState,
                    async_replicas: list[ReplicaState],
                    repl_log: AsyncReplicationLog,
                    key: str, value: str) -> bool:
    """
    Semi-synchronous: wait for ONE sync follower, then ack.
    Remaining followers replicate asynchronously.
    """
    # Step 1: Apply to leader
    leader_data[key] = value

    # Step 2: Synchronously replicate to the designated sync follower
    success = sync_replica.apply_write(key, value)
    if not success:
        # Failover: promote an async follower to sync role
        for candidate in async_replicas:
            if candidate.apply_write(key, value):
                print(f"  [SEMI] Promoted {candidate.name} to sync role")
                break
        else:
            return False  # all followers unreachable

    # Step 3: Asynchronously replicate to remaining followers
    for replica in async_replicas:
        repl_log.enqueue_write(key, value)

    return True
```

### 2.3 Replication Log Formats

The leader must communicate *what changed* to followers. Three approaches:

| Format | Description | Pros | Cons |
|---|---|---|---|
| **Statement-based** | Send the SQL/command (e.g., `UPDATE users SET name='X' WHERE id=1`) | Simple, compact | Non-deterministic functions (NOW(), RAND()), auto-increment, triggers |
| **Write-ahead log (WAL) shipping** | Send the physical WAL bytes | Exact byte-level replication | Coupled to storage engine version; no cross-version replication |
| **Logical (row-based) log** | Send logical changes: "row X changed from A to B" | Version-independent, can be consumed by external systems | Larger than WAL for bulk operations |

Modern systems (PostgreSQL logical replication, MySQL binlog row format, MongoDB oplog) overwhelmingly use logical logs.

### 2.4 Replication Lag and Its Effects

With asynchronous replication, followers may lag behind the leader. This creates several consistency anomalies visible to clients.

#### Read-Your-Writes Violation

A user writes data, then reads it back from a follower that hasn't received the update yet.

```
Time ──────────────────────────────────────►

Client:    WRITE x=42 ──────── READ x ──► sees x=OLD!
                │                   │
Leader:    x=42 applied             │
                │                   │
Follower:  ─────────── lag ────── x=OLD still
```

**Fix**: Read-your-writes consistency. Route the user's own reads to the leader, or track the user's latest write timestamp and only read from followers that are caught up.

#### Monotonic Read Violation

A user reads from follower A (caught up), then reads from follower B (lagging), and sees data go *backward in time*.

```
Time ──────────────────────────────────────►

Client:    READ x ──► x=42     READ x ──► x=17 (older!)
                │                   │
Follower A: x=42 (caught up)       │
Follower B: ──────────────────── x=17 (lagging)
```

**Fix**: Monotonic reads. Pin each user's reads to a single follower (session stickiness), or track the user's latest observed version.

#### Causal Ordering Violation

User A writes a question, User B writes an answer. A follower receives the answer before the question, making no sense.

```
Leader:    Q posted at t=1    A posted at t=2
           │                  │
Follower:  A posted at t=2   Q posted at t=1  ← wrong order!
```

**Fix**: Causal consistency. Ensure that if event B depends on event A, all replicas see A before B. This requires tracking causal dependencies (vector clocks, version vectors).

### 2.5 Leader Failure: Failover

When the leader crashes, a follower must be promoted. This process — **failover** — is one of the most dangerous operations in distributed systems.

#### Failover Steps

```
1. Detect leader failure (timeout-based heartbeat)
2. Choose a new leader (most up-to-date follower, or by election)
3. Reconfigure clients to send writes to the new leader
4. Old leader must recognize it is no longer leader when it recovers
```

#### Split-Brain Problem

If the old leader comes back online without realizing it has been replaced, *two nodes accept writes simultaneously*. This is called **split-brain** and causes data divergence.

```
                  Network partition
                       │
┌───────────┐     ┌────┼────────┐
│ Old Leader│     │    │        │
│ (thinks   │     │ New Leader  │
│  it's     │     │ (elected)   │
│  leader)  │     │             │
└───────────┘     └─────────────┘
  Writes ▲              ▲ Writes
         │              │
   Client A         Client B
         (DATA DIVERGES!)
```

**Prevention mechanisms**:

| Mechanism | How It Works |
|---|---|
| **Fencing tokens** | New leader gets a monotonically increasing token. Writes with old tokens are rejected. |
| **STONITH** (Shoot The Other Node In The Head) | Physically power off the old leader before promoting the new one. |
| **Epoch numbers** | Each leader election increments an epoch. Followers reject writes from stale epochs. |
| **Lease-based leadership** | Leader must periodically renew a time-limited lease. If it fails, another node acquires the lease. |

```python
import time
from dataclasses import dataclass


@dataclass
class FencingToken:
    """Monotonically increasing token to prevent split-brain."""
    epoch: int
    leader_id: str


class FencedLeaderNode:
    """A leader node that uses fencing tokens to prevent split-brain."""

    def __init__(self, node_id: str, epoch: int):
        self.node_id = node_id
        self.epoch = epoch
        self.data: dict[str, str] = {}
        self.is_leader = True

    def write(self, key: str, value: str, token: FencingToken) -> bool:
        """Accept write only if the fencing token matches current epoch."""
        if token.epoch < self.epoch:
            print(f"  [FENCE] Rejected write from epoch {token.epoch} "
                  f"(current: {self.epoch})")
            return False
        if token.leader_id != self.node_id:
            print(f"  [FENCE] Rejected write from {token.leader_id} "
                  f"(current leader: {self.node_id})")
            return False
        self.data[key] = value
        return True

    def step_down(self) -> None:
        """Recognize that this node is no longer the leader."""
        self.is_leader = False
        print(f"  [FENCE] Node {self.node_id} stepped down from epoch {self.epoch}")


def simulate_failover():
    """Demonstrate fencing tokens preventing split-brain."""
    old_leader = FencedLeaderNode("node-1", epoch=1)
    old_token = FencingToken(epoch=1, leader_id="node-1")

    # Old leader accepts writes in epoch 1
    assert old_leader.write("x", "100", old_token)

    # Failover: new leader elected with epoch 2
    new_leader = FencedLeaderNode("node-2", epoch=2)
    new_token = FencingToken(epoch=2, leader_id="node-2")

    # New leader accepts writes
    assert new_leader.write("x", "200", new_token)

    # Old leader comes back — tries to write with old token
    # A properly fenced follower/storage rejects it
    assert not new_leader.write("x", "300", old_token)  # REJECTED

    old_leader.step_down()
```

---

## 3. Multi-Leader Replication

In multi-leader (also called multi-master or active-active) replication, more than one node accepts writes. Each leader replicates its writes to all other leaders.

### 3.1 Use Cases

Multi-leader replication makes sense when:

| Use Case | Why Multi-Leader? |
|---|---|
| **Multi-datacenter** | Each DC has a local leader. Writes are fast (local), then replicated across DCs asynchronously. |
| **Offline-capable clients** | Each device is a "leader" while offline. Syncs when connectivity resumes (CouchDB, PouchDB). |
| **Real-time collaborative editing** | Each user's client acts as a leader. Changes merge asynchronously (though CRDTs from Lesson 10 are often preferred). |

### 3.2 Replication Topologies

How do leaders propagate writes to each other?

```
All-to-All              Circular              Star (Hub-and-Spoke)

  A ◄──► B              A ──► B              A ──► H ◄── B
  ▲  ╲╱  ▲              ▲     │                    │
  │  ╱╲  │              │     ▼                    ▼
  C ◄──► D              D ◄── C              C ◄── H ──► D
```

| Topology | Fault Tolerance | Latency | Complexity |
|---|---|---|---|
| **All-to-all** | High (any node can fail) | Lowest (direct paths) | Ordering issues (overtaking) |
| **Circular** | Low (single failure breaks ring) | Higher (hop-by-hop) | Simple, but fragile |
| **Star** | Medium (hub is SPOF) | Medium | Simple routing |

All-to-all is most common in practice but requires version vectors or Lamport timestamps to detect and resolve ordering issues.

### 3.3 Conflict Detection

With multiple leaders accepting writes concurrently, conflicts are inevitable. A conflict occurs when two leaders modify the same data independently.

```
Time ──────────────────────────────────────►

Leader A:  SET x = "foo"   ──── replicate ──► Leader B
Leader B:  SET x = "bar"   ──── replicate ──► Leader A
                                                  │
           x = ??? (conflict!)                    │
```

**When is the conflict detected?**

| Timing | Mechanism | Implication |
|---|---|---|
| **Synchronous** (at write time) | Wait for all leaders to agree | Defeats purpose of multi-leader (blocks on WAN latency) |
| **Asynchronous** (at replication time) | Detect when replicated write arrives | Must resolve after the fact; both writes already acknowledged |

In practice, asynchronous detection is almost always used. The key question becomes: how do we *resolve* conflicts?

### 3.4 Conflict Resolution Strategies

#### Last-Writer-Wins (LWW)

Assign each write a timestamp. The write with the highest timestamp wins; the other is silently discarded.

```python
from dataclasses import dataclass
from typing import Any


@dataclass
class TimestampedValue:
    value: Any
    timestamp: float  # wall clock or logical clock
    writer_id: str


def lww_resolve(existing: TimestampedValue, incoming: TimestampedValue) -> TimestampedValue:
    """Last-Writer-Wins: higher timestamp wins. Tie-break on writer_id."""
    if incoming.timestamp > existing.timestamp:
        return incoming
    if incoming.timestamp == existing.timestamp:
        # Deterministic tie-breaking: higher writer_id wins
        return incoming if incoming.writer_id > existing.writer_id else existing
    return existing
```

**Problem**: LWW *silently drops writes*. If two users concurrently update different fields of the same record, one update is lost. This is the source of many production bugs.

#### Merge Functions

Instead of discarding one write, merge both. This is application-specific.

```python
def merge_shopping_carts(cart_a: set[str], cart_b: set[str]) -> set[str]:
    """Merge two shopping carts by taking the union of items."""
    return cart_a | cart_b


def merge_counters(count_a: int, origin_a: int,
                   count_b: int, origin_b: int) -> int:
    """
    Merge two counters that diverged from a common origin.
    Each counter records its starting point (origin).
    Merged value = origin + (delta_a) + (delta_b).
    """
    delta_a = count_a - origin_a
    delta_b = count_b - origin_b
    return origin_a + delta_a + delta_b
```

#### Custom Application-Level Resolution

Preserve all conflicting versions and let the application (or user) resolve them. Amazon's shopping cart famously used this approach.

```python
@dataclass
class ConflictRecord:
    key: str
    conflicting_values: list[TimestampedValue]
    resolved: bool = False
    resolution: Any = None

    def resolve(self, strategy: str = "manual") -> Any:
        if strategy == "union":
            # For set-like values: merge via union
            merged = set()
            for v in self.conflicting_values:
                if isinstance(v.value, (set, list)):
                    merged.update(v.value)
            self.resolution = merged
        elif strategy == "lww":
            self.resolution = max(
                self.conflicting_values, key=lambda v: v.timestamp
            ).value
        elif strategy == "manual":
            # Present all versions to user
            print(f"Conflict on key '{self.key}':")
            for i, v in enumerate(self.conflicting_values):
                print(f"  [{i}] {v.value} (from {v.writer_id} at {v.timestamp})")
            # In production: return all versions to client for resolution
            self.resolution = self.conflicting_values
        self.resolved = True
        return self.resolution
```

### 3.5 Multi-Leader Replication Implementation

```python
import time
import threading
from collections import defaultdict
from typing import Optional


class MultiLeaderNode:
    """A node in a multi-leader replication system."""

    def __init__(self, node_id: str, peers: Optional[list] = None):
        self.node_id = node_id
        self.data: dict[str, TimestampedValue] = {}
        self.peers: list['MultiLeaderNode'] = peers or []
        self.replication_log: list[tuple[str, TimestampedValue]] = []
        self._lock = threading.Lock()

    def local_write(self, key: str, value: Any) -> TimestampedValue:
        """Accept a write locally and enqueue for replication."""
        ts_value = TimestampedValue(
            value=value,
            timestamp=time.time(),
            writer_id=self.node_id,
        )
        with self._lock:
            self.data[key] = ts_value
            self.replication_log.append((key, ts_value))
        return ts_value

    def receive_replication(self, key: str, incoming: TimestampedValue) -> None:
        """Receive a replicated write from a peer. Resolve conflicts via LWW."""
        with self._lock:
            existing = self.data.get(key)
            if existing is None:
                self.data[key] = incoming
            else:
                self.data[key] = lww_resolve(existing, incoming)

    def replicate_to_peers(self) -> None:
        """Send all pending replication log entries to all peers."""
        with self._lock:
            pending = list(self.replication_log)
            self.replication_log.clear()
        for key, ts_value in pending:
            for peer in self.peers:
                peer.receive_replication(key, ts_value)

    def read(self, key: str) -> Optional[Any]:
        with self._lock:
            tv = self.data.get(key)
            return tv.value if tv else None


def demo_multi_leader():
    """Demonstrate multi-leader replication with conflict resolution."""
    node_a = MultiLeaderNode("A")
    node_b = MultiLeaderNode("B")
    node_a.peers = [node_b]
    node_b.peers = [node_a]

    # Concurrent writes to the same key
    node_a.local_write("user:1:name", "Alice")
    node_b.local_write("user:1:name", "Alicia")

    # Before replication: each node has its own version
    print(f"Before sync — A: {node_a.read('user:1:name')}, "
          f"B: {node_b.read('user:1:name')}")

    # Replicate (LWW resolution)
    node_a.replicate_to_peers()
    node_b.replicate_to_peers()

    # After replication: both converge (LWW picks higher timestamp)
    print(f"After sync  — A: {node_a.read('user:1:name')}, "
          f"B: {node_b.read('user:1:name')}")
```

---

## 4. Leaderless Replication (Dynamo-Style)

In leaderless replication, there is no designated leader. Any node can accept reads and writes. Consistency is achieved through **quorum** protocols.

This approach was popularized by Amazon's Dynamo paper (2007) and is used by Cassandra, Riak, and Voldemort.

### 4.1 Read and Write Quorums

Given **N** replicas, define:
- **W**: the number of replicas that must acknowledge a write
- **R**: the number of replicas that must respond to a read

**The quorum condition**: if **W + R > N**, then every read is guaranteed to see at least one replica that has the latest write.

```
    N = 3 replicas

    Write quorum W = 2:          Read quorum R = 2:
    ┌─────────────┐              ┌─────────────┐
    │ Replica 1 ✓ │              │ Replica 1 ✓ │
    │ Replica 2 ✓ │              │ Replica 2   │
    │ Replica 3   │              │ Replica 3 ✓ │
    └─────────────┘              └─────────────┘

    W + R = 2 + 2 = 4 > 3 = N  ✓
    Overlap guaranteed: at least one replica has the latest write
```

#### Mathematical Basis

The quorum condition ensures **set intersection** between the write set and read set is non-empty:

$$|W \cap R| \geq W + R - N > 0 \quad \text{when} \quad W + R > N$$

Common configurations:

| N | W | R | Properties |
|---|---|---|---|
| 3 | 2 | 2 | Standard quorum. Tolerates 1 failure for both reads and writes. |
| 3 | 3 | 1 | Write-heavy: fast reads, but any failure blocks writes. |
| 3 | 1 | 3 | Read-heavy: fast writes, but any failure blocks reads. |
| 5 | 3 | 3 | Tolerates 2 failures. Higher availability at cost of latency. |

### 4.2 Quorum Reads and Writes: Implementation

```python
import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class VersionedValue:
    """A value tagged with a version (timestamp) for comparison."""
    value: Any
    version: int  # logical clock / version number
    timestamp: float = field(default_factory=time.time)

    def __lt__(self, other: 'VersionedValue') -> bool:
        return self.version < other.version


class DynamoStyleNode:
    """Simulates a node in a leaderless (Dynamo-style) system."""

    def __init__(self, node_id: str, is_available: bool = True):
        self.node_id = node_id
        self.store: dict[str, VersionedValue] = {}
        self.is_available = is_available
        self.hinted_handoff: list[tuple[str, str, VersionedValue]] = []

    def put(self, key: str, value: VersionedValue) -> bool:
        """Write a value to local storage."""
        if not self.is_available:
            return False
        existing = self.store.get(key)
        if existing is None or value.version > existing.version:
            self.store[key] = value
        return True

    def get(self, key: str) -> Optional[VersionedValue]:
        """Read a value from local storage."""
        if not self.is_available:
            return None
        return self.store.get(key)


class QuorumCoordinator:
    """
    Coordinator for quorum-based reads and writes.
    Implements W + R > N consistency with read repair.
    """

    def __init__(self, nodes: list[DynamoStyleNode], n: int, w: int, r: int):
        self.nodes = nodes
        self.n = n
        self.w = w
        self.r = r
        self._version_counter = 0

        assert w + r > n, f"Quorum condition violated: W({w}) + R({r}) <= N({n})"
        assert len(nodes) >= n, f"Need at least N={n} nodes"

    def _select_replicas(self, key: str) -> list[DynamoStyleNode]:
        """
        Select N replica nodes for a key using consistent hashing.
        (Simplified: use hash to pick starting position, then take N consecutive.)
        """
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        start = h % len(self.nodes)
        replicas = []
        for i in range(self.n):
            replicas.append(self.nodes[(start + i) % len(self.nodes)])
        return replicas

    def write(self, key: str, value: Any) -> bool:
        """
        Quorum write: send to N replicas, wait for W acknowledgments.
        Returns True if quorum is met.
        """
        self._version_counter += 1
        versioned = VersionedValue(value=value, version=self._version_counter)

        replicas = self._select_replicas(key)
        acks = 0
        for replica in replicas:
            if replica.put(key, versioned):
                acks += 1

        if acks >= self.w:
            print(f"  [WRITE] key={key}, value={value}, version={versioned.version}, "
                  f"acks={acks}/{self.n} (need {self.w})")
            return True
        else:
            print(f"  [WRITE FAILED] key={key}, acks={acks}/{self.n} "
                  f"(need {self.w})")
            return False

    def read(self, key: str) -> Optional[Any]:
        """
        Quorum read: read from N replicas, wait for R responses.
        Return the value with the highest version. Trigger read repair.
        """
        replicas = self._select_replicas(key)
        responses: list[tuple[DynamoStyleNode, Optional[VersionedValue]]] = []

        for replica in replicas:
            val = replica.get(key)
            responses.append((replica, val))

        # Filter successful responses
        successful = [(node, val) for node, val in responses if val is not None]

        if len(successful) < self.r:
            print(f"  [READ FAILED] key={key}, responses={len(successful)}/{self.n} "
                  f"(need {self.r})")
            return None

        # Find the most recent value
        best_node, best_val = max(successful, key=lambda x: x[1].version)

        # Read repair: send the latest value to nodes with stale data
        for node, val in responses:
            if val is None or val.version < best_val.version:
                node.put(key, best_val)  # read repair
                print(f"  [READ REPAIR] Updating {node.node_id} "
                      f"from v{val.version if val else 0} to v{best_val.version}")

        print(f"  [READ] key={key}, value={best_val.value}, "
              f"version={best_val.version}")
        return best_val.value


def demo_quorum():
    """Demonstrate quorum reads and writes with read repair."""
    nodes = [DynamoStyleNode(f"node-{i}") for i in range(5)]
    coord = QuorumCoordinator(nodes, n=3, w=2, r=2)

    # Normal write and read
    coord.write("user:42", {"name": "Alice", "email": "alice@example.com"})
    coord.read("user:42")

    # Simulate a node failure during write, then read repair
    print("\n--- Simulating node failure ---")
    nodes[1].is_available = False
    coord.write("user:42", {"name": "Alice Updated", "email": "alice@new.com"})

    # Node comes back — read repair will update it
    nodes[1].is_available = True
    coord.read("user:42")  # triggers read repair on node-1
```

### 4.3 Sloppy Quorums and Hinted Handoff

A **strict quorum** requires W and R acknowledgments from the N *designated* replicas for a key. But what if several of those N nodes are down? Writes fail even though other healthy nodes exist in the cluster.

A **sloppy quorum** allows writes to be accepted by *any* W nodes in the cluster, not just the designated N. When the designated node recovers, the temporary holder sends the data to it — this is **hinted handoff**.

```
Normal:
  Key "x" → Replicas {A, B, C}
  Write to A, B, C — W=2 → OK if 2 ack

Sloppy (C is down):
  Key "x" → Replicas {A, B, C}  but C is down
  Write to A, B, D — W=2 → OK (D holds hint for C)

  When C recovers:
    D → C: "Here's the write I held for you"  (hinted handoff)
    D deletes its temporary copy
```

```python
@dataclass
class HintedWrite:
    """A write held temporarily for an unavailable node."""
    target_node_id: str
    key: str
    value: VersionedValue
    hint_holder_id: str


class SloppyQuorumCoordinator(QuorumCoordinator):
    """Extends quorum coordinator with sloppy quorum and hinted handoff."""

    def __init__(self, nodes: list[DynamoStyleNode], n: int, w: int, r: int):
        super().__init__(nodes, n, w, r)
        self.hints: list[HintedWrite] = []

    def write_sloppy(self, key: str, value: Any) -> bool:
        """
        Sloppy quorum write: if designated replicas are down,
        write to other nodes and create hints for handoff.
        """
        self._version_counter += 1
        versioned = VersionedValue(value=value, version=self._version_counter)

        designated = self._select_replicas(key)
        all_nodes = list(self.nodes)
        random.shuffle(all_nodes)

        acks = 0
        for replica in designated:
            if replica.put(key, versioned):
                acks += 1

        # If we haven't reached quorum from designated nodes,
        # try other nodes (sloppy quorum)
        if acks < self.w:
            for node in all_nodes:
                if node not in designated and node.is_available:
                    node.put(key, versioned)
                    acks += 1
                    # Record hint for the unavailable designated node
                    down_nodes = [n for n in designated if not n.is_available]
                    if down_nodes:
                        self.hints.append(HintedWrite(
                            target_node_id=down_nodes[0].node_id,
                            key=key,
                            value=versioned,
                            hint_holder_id=node.node_id,
                        ))
                    if acks >= self.w:
                        break

        return acks >= self.w

    def process_hinted_handoff(self) -> int:
        """Deliver hinted writes to nodes that have recovered."""
        delivered = 0
        remaining = []
        for hint in self.hints:
            target = next(
                (n for n in self.nodes if n.node_id == hint.target_node_id), None
            )
            if target and target.is_available:
                target.put(hint.key, hint.value)
                delivered += 1
                print(f"  [HANDOFF] Delivered {hint.key} v{hint.value.version} "
                      f"from {hint.hint_holder_id} to {hint.target_node_id}")
            else:
                remaining.append(hint)
        self.hints = remaining
        return delivered
```

**Important caveat**: Sloppy quorums do **not** guarantee the W + R > N overlap property. A sloppy quorum write might go to nodes outside the designated set, so a subsequent strict quorum read might miss it. Sloppy quorums prioritize availability over consistency.

### 4.4 Anti-Entropy with Merkle Trees

Read repair only fixes stale data when a key is read. For data that is rarely read, a background process — **anti-entropy** — proactively synchronizes replicas.

The naive approach (compare every key-value pair) is O(total data), which is impractical for large datasets. **Merkle trees** (hash trees) enable efficient comparison.

```
Level 0 (root):      H(H12 + H34)
                      ╱         ╲
Level 1:          H12            H34
                 ╱   ╲          ╱   ╲
Level 2:       H1     H2     H3     H4
               │      │      │      │
Data:        key1   key2   key3   key4
```

Two nodes compare their Merkle trees top-down:
1. If the root hashes match → all data is identical → done
2. If roots differ → descend into children
3. Repeat until the differing leaf ranges are identified
4. Synchronize only the differing ranges

**Complexity**: O(log N) comparisons to identify differences, vs O(N) for brute force.

```python
import hashlib
from typing import Optional


class MerkleNode:
    """Node in a Merkle tree for anti-entropy synchronization."""

    def __init__(self, range_start: str, range_end: str):
        self.range_start = range_start
        self.range_end = range_end
        self.hash_value: str = ""
        self.left: Optional['MerkleNode'] = None
        self.right: Optional['MerkleNode'] = None
        self.keys: list[tuple[str, str]] = []  # leaf nodes store (key, value) pairs

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def compute_hash(self) -> str:
        """Compute hash bottom-up."""
        if self.is_leaf():
            data = "".join(f"{k}:{v}" for k, v in sorted(self.keys))
            self.hash_value = hashlib.sha256(data.encode()).hexdigest()[:16]
        else:
            left_hash = self.left.compute_hash() if self.left else ""
            right_hash = self.right.compute_hash() if self.right else ""
            combined = left_hash + right_hash
            self.hash_value = hashlib.sha256(combined.encode()).hexdigest()[:16]
        return self.hash_value


def find_differences(node_a: MerkleNode, node_b: MerkleNode) -> list[str]:
    """
    Compare two Merkle trees and return the key ranges that differ.
    Only descends into subtrees where hashes don't match.
    """
    if node_a.hash_value == node_b.hash_value:
        return []  # subtrees are identical

    if node_a.is_leaf() or node_b.is_leaf():
        return [f"[{node_a.range_start}, {node_a.range_end}]"]

    diffs = []
    if node_a.left and node_b.left:
        diffs.extend(find_differences(node_a.left, node_b.left))
    if node_a.right and node_b.right:
        diffs.extend(find_differences(node_a.right, node_b.right))
    return diffs
```

### 4.5 When Quorum Consistency Fails

Even with W + R > N, quorum consistency can be violated in practice:

| Scenario | Why It Fails |
|---|---|
| **Sloppy quorum** | Write goes to non-designated nodes; read doesn't see it |
| **Concurrent writes** | Two writes to the same key at similar timestamps; different replicas may disagree on order |
| **Concurrent read and write** | Read may see old or new value depending on which replicas respond |
| **Failed write rollback** | Write reaches W-1 replicas, then coordinator fails; partial write persists on some replicas |
| **Clock skew with LWW** | If using wall-clock timestamps for versioning, clock skew can cause newer writes to be overwritten |

**Key insight**: Dynamo-style quorums provide **probabilistic** consistency, not strong consistency. For strong consistency, you need consensus protocols (Lessons 5–6).

---

## 5. Chain Replication

Chain replication is an alternative to quorum-based approaches that provides **strong consistency** with high throughput for read-heavy workloads.

### 5.1 Basic Chain Replication

Nodes are organized in a chain. Writes enter at the **head** and propagate down the chain. Reads are served by the **tail**.

```
            Write                                         Read
              │                                            ▲
              ▼                                            │
┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
│ Head │───►│Node 2│───►│Node 3│───►│Node 4│───►│ Tail │
└──────┘    └──────┘    └──────┘    └──────┘    └──────┘
              │           │           │           │
         Write is confirmed only when it reaches the tail
```

**Properties**:
- **Strong consistency**: The tail has a total ordering of all committed writes
- **Write latency**: Sum of all inter-node latencies (not parallel like quorum)
- **Read latency**: Single node (the tail)
- **Read throughput**: Limited to single node (the tail)

```python
from collections import OrderedDict
from typing import Optional, Any


class ChainNode:
    """A node in a chain replication system."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.data: dict[str, Any] = {}
        self.successor: Optional['ChainNode'] = None
        self.pending_writes: OrderedDict[int, tuple[str, Any]] = OrderedDict()
        self._write_seq = 0

    @property
    def is_head(self) -> bool:
        return True  # set by chain manager

    @property
    def is_tail(self) -> bool:
        return self.successor is None

    def handle_write(self, key: str, value: Any, seq: Optional[int] = None) -> bool:
        """
        Handle a write request.
        Head: create sequence number, forward to successor.
        Middle: apply and forward.
        Tail: apply and acknowledge.
        """
        if seq is None:
            # Head node: assign sequence number
            self._write_seq += 1
            seq = self._write_seq

        # Apply locally
        self.data[key] = value

        if self.is_tail:
            # Tail: write is committed
            return True
        else:
            # Forward to successor
            return self.successor.handle_write(key, value, seq)

    def handle_read(self, key: str) -> Optional[Any]:
        """
        Only the tail serves reads (guarantees strong consistency).
        """
        if not self.is_tail:
            raise RuntimeError("Reads must be served by the tail node")
        return self.data.get(key)


def build_chain(node_ids: list[str]) -> tuple[ChainNode, ChainNode]:
    """Build a chain of nodes. Returns (head, tail)."""
    nodes = [ChainNode(nid) for nid in node_ids]
    for i in range(len(nodes) - 1):
        nodes[i].successor = nodes[i + 1]
    return nodes[0], nodes[-1]


def demo_chain_replication():
    """Demonstrate basic chain replication."""
    head, tail = build_chain(["H", "M1", "M2", "T"])

    # Writes enter at head, propagate through chain
    head.handle_write("account:1", {"balance": 1000})
    head.handle_write("account:2", {"balance": 500})

    # Reads served only by tail (strong consistency)
    print(f"Read account:1 = {tail.handle_read('account:1')}")
    print(f"Read account:2 = {tail.handle_read('account:2')}")
```

### 5.2 CRAQ: Chain Replication with Apportioned Queries

The bottleneck of basic chain replication is that the tail serves all reads. **CRAQ** (Chain Replication with Apportioned Queries) allows reads from *any* node in the chain, improving read throughput linearly with chain length.

**Key idea**: Each node stores multiple versions of each key. A key is **clean** if the node's latest version equals the committed (tail) version. Clean keys can be served by any node. Dirty keys require checking with the tail.

```
CRAQ Node State for key "x":

  Node has versions:  [v1: "old", v2: "new"]
  Tail committed:     v1

  → Key is DIRTY (v2 not yet committed)
  → Read must check with tail

  After tail commits v2:
  → Key is CLEAN
  → Any node can serve the read
```

```python
class CRAQNode:
    """A node in a CRAQ (Chain Replication with Apportioned Queries) system."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.versions: dict[str, list[tuple[int, Any]]] = {}  # key -> [(version, value)]
        self.committed_version: dict[str, int] = {}  # key -> latest committed version
        self.successor: Optional['CRAQNode'] = None
        self.predecessor: Optional['CRAQNode'] = None
        self._seq = 0

    @property
    def is_tail(self) -> bool:
        return self.successor is None

    @property
    def is_head(self) -> bool:
        return self.predecessor is None

    def handle_write(self, key: str, value: Any, version: Optional[int] = None) -> bool:
        """Write propagates head to tail. Tail sends commit back up the chain."""
        if version is None:
            self._seq += 1
            version = self._seq

        # Store new version
        if key not in self.versions:
            self.versions[key] = []
        self.versions[key].append((version, value))

        if self.is_tail:
            # Tail: commit and propagate commit backward
            self.committed_version[key] = version
            self._clean_old_versions(key, version)
            self._propagate_commit_backward(key, version)
            return True
        else:
            return self.successor.handle_write(key, value, version)

    def _propagate_commit_backward(self, key: str, version: int) -> None:
        """Send commit notification up the chain."""
        if self.predecessor:
            self.predecessor._receive_commit(key, version)

    def _receive_commit(self, key: str, version: int) -> None:
        """Process commit notification: clean old versions, propagate backward."""
        self.committed_version[key] = version
        self._clean_old_versions(key, version)
        if self.predecessor:
            self.predecessor._receive_commit(key, version)

    def _clean_old_versions(self, key: str, committed: int) -> None:
        """Remove versions older than the committed version."""
        if key in self.versions:
            self.versions[key] = [
                (v, val) for v, val in self.versions[key] if v >= committed
            ]

    def read(self, key: str) -> Optional[Any]:
        """
        CRAQ read: serve locally if key is clean, otherwise check tail.
        A key is clean if only one version exists (the committed one).
        """
        if key not in self.versions:
            return None

        versions = self.versions[key]
        if len(versions) == 1:
            # Clean: only committed version exists — serve locally
            return versions[0][1]
        else:
            # Dirty: multiple versions — must check tail for committed version
            return self._check_tail_version(key)

    def _check_tail_version(self, key: str) -> Optional[Any]:
        """Ask the tail for the committed version of a key."""
        node = self
        while not node.is_tail:
            node = node.successor
        committed_ver = node.committed_version.get(key)
        if committed_ver is None:
            return None
        # Return the committed version from local store
        for v, val in self.versions.get(key, []):
            if v == committed_ver:
                return val
        return None
```

### 5.3 Comparison of Replication Approaches

| Property | Single-Leader | Multi-Leader | Leaderless (Quorum) | Chain Replication |
|---|---|---|---|---|
| **Consistency** | Strong (sync) or eventual (async) | Eventual (conflicts) | Probabilistic | Strong |
| **Write latency** | Low (leader only) | Low (local leader) | W replicas in parallel | Sum of chain hops |
| **Read latency** | Leader or follower | Any leader | R replicas in parallel | Tail only (or CRAQ) |
| **Read throughput** | Scales with followers | Scales with leaders | Scales with N | CRAQ: scales with chain |
| **Write throughput** | Single leader bottleneck | Multiple leaders | Any node | Head bottleneck |
| **Fault tolerance** | Leader failure needs failover | Tolerates individual leader failure | Tolerates N-W write or N-R read failures | Head/tail failure needs reconfiguration |
| **Conflict handling** | No conflicts (single writer) | Must resolve conflicts | Version conflicts possible | No conflicts (total order at tail) |
| **Complexity** | Low | High (conflicts) | Medium | Low-Medium |

---

## 6. Putting It All Together: Comprehensive Quorum Simulator

The following implementation ties together many concepts from this lesson into a configurable simulator.

```python
"""
Comprehensive quorum-based replication simulator.
Demonstrates configurable W, R, N with failure injection,
read repair, and consistency measurement.
"""

import hashlib
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ConsistencyLevel(Enum):
    ONE = 1
    QUORUM = 2
    ALL = 3


@dataclass
class WriteResult:
    success: bool
    acks: int
    required: int
    version: int
    key: str
    value: Any


@dataclass
class ReadResult:
    success: bool
    value: Any
    version: int
    responses: int
    required: int
    stale_replicas_repaired: int


@dataclass
class Replica:
    """A storage replica with failure simulation."""
    replica_id: str
    store: dict[str, tuple[Any, int]] = field(default_factory=dict)
    is_available: bool = True
    latency_ms: float = 1.0
    failure_rate: float = 0.0  # probability of transient failure per operation

    def put(self, key: str, value: Any, version: int) -> bool:
        if not self.is_available:
            return False
        if random.random() < self.failure_rate:
            return False
        existing = self.store.get(key)
        if existing is None or version > existing[1]:
            self.store[key] = (value, version)
        return True

    def get(self, key: str) -> Optional[tuple[Any, int]]:
        if not self.is_available:
            return None
        if random.random() < self.failure_rate:
            return None
        return self.store.get(key)


class ReplicationSimulator:
    """
    Configurable quorum-based replication simulator.
    Supports strict and sloppy quorums, read repair, and consistency measurement.
    """

    def __init__(self, num_replicas: int = 5, n: int = 3, w: int = 2, r: int = 2,
                 sloppy_quorum: bool = False):
        self.replicas = [Replica(f"R{i}") for i in range(num_replicas)]
        self.n = n
        self.w = w
        self.r = r
        self.sloppy_quorum = sloppy_quorum
        self._version = 0
        self._write_log: list[WriteResult] = []
        self._read_log: list[ReadResult] = []

    def _get_preference_list(self, key: str) -> list[Replica]:
        """Determine the N preferred replicas for a key."""
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        start = h % len(self.replicas)
        result = []
        for i in range(len(self.replicas)):
            idx = (start + i) % len(self.replicas)
            result.append(self.replicas[idx])
            if len(result) == self.n:
                break
        return result

    def write(self, key: str, value: Any,
              consistency: ConsistencyLevel = ConsistencyLevel.QUORUM) -> WriteResult:
        """Write with configurable consistency level."""
        self._version += 1
        version = self._version

        required = {
            ConsistencyLevel.ONE: 1,
            ConsistencyLevel.QUORUM: self.w,
            ConsistencyLevel.ALL: self.n,
        }[consistency]

        preferred = self._get_preference_list(key)
        acks = 0

        # Try preferred replicas first
        for replica in preferred:
            if replica.put(key, value, version):
                acks += 1

        # Sloppy quorum: try non-preferred replicas if needed
        if self.sloppy_quorum and acks < required:
            others = [r for r in self.replicas if r not in preferred]
            for replica in others:
                if replica.put(key, value, version):
                    acks += 1
                    if acks >= required:
                        break

        result = WriteResult(
            success=acks >= required,
            acks=acks,
            required=required,
            version=version,
            key=key,
            value=value,
        )
        self._write_log.append(result)
        return result

    def read(self, key: str,
             consistency: ConsistencyLevel = ConsistencyLevel.QUORUM) -> ReadResult:
        """Read with configurable consistency level and read repair."""
        required = {
            ConsistencyLevel.ONE: 1,
            ConsistencyLevel.QUORUM: self.r,
            ConsistencyLevel.ALL: self.n,
        }[consistency]

        preferred = self._get_preference_list(key)
        responses: list[tuple[Replica, Any, int]] = []

        for replica in preferred:
            result = replica.get(key)
            if result is not None:
                responses.append((replica, result[0], result[1]))

        if len(responses) < required:
            return ReadResult(False, None, -1, len(responses), required, 0)

        # Find the latest version
        best_replica, best_value, best_version = max(responses, key=lambda x: x[2])

        # Read repair: update stale replicas
        repaired = 0
        for replica, value, version in responses:
            if version < best_version:
                replica.put(key, best_value, best_version)
                repaired += 1

        result = ReadResult(
            success=True,
            value=best_value,
            version=best_version,
            responses=len(responses),
            required=required,
            stale_replicas_repaired=repaired,
        )
        self._read_log.append(result)
        return result

    def measure_consistency(self, num_operations: int = 1000) -> dict:
        """
        Measure consistency by performing concurrent writes and reads.
        Returns statistics on stale reads and read repair frequency.
        """
        stale_reads = 0
        total_reads = 0
        repairs = 0

        for i in range(num_operations):
            key = f"key-{random.randint(0, 99)}"

            if random.random() < 0.5:
                # Write
                self.write(key, f"value-{i}")
            else:
                # Read
                result = self.read(key)
                if result.success:
                    total_reads += 1
                    repairs += result.stale_replicas_repaired
                    # Check if we got the latest version
                    latest = self._get_latest_version(key)
                    if result.version < latest:
                        stale_reads += 1

        return {
            "total_reads": total_reads,
            "stale_reads": stale_reads,
            "stale_read_rate": stale_reads / max(total_reads, 1),
            "total_repairs": repairs,
            "repair_rate": repairs / max(total_reads, 1),
        }

    def _get_latest_version(self, key: str) -> int:
        """Get the maximum version of a key across all replicas."""
        max_version = 0
        for replica in self.replicas:
            result = replica.store.get(key)
            if result and result[1] > max_version:
                max_version = result[1]
        return max_version


def run_consistency_experiment():
    """Compare consistency under different configurations."""
    configs = [
        {"n": 3, "w": 2, "r": 2, "sloppy": False, "label": "Strict W=2 R=2"},
        {"n": 3, "w": 1, "r": 1, "sloppy": False, "label": "Strict W=1 R=1"},
        {"n": 3, "w": 3, "r": 1, "sloppy": False, "label": "Strict W=3 R=1"},
        {"n": 3, "w": 2, "r": 2, "sloppy": True,  "label": "Sloppy W=2 R=2"},
    ]

    print(f"{'Configuration':<25} {'Stale Rate':>12} {'Repair Rate':>12}")
    print("-" * 51)

    for cfg in configs:
        sim = ReplicationSimulator(
            num_replicas=5, n=cfg["n"], w=cfg["w"], r=cfg["r"],
            sloppy_quorum=cfg["sloppy"],
        )
        # Inject some failures
        sim.replicas[2].failure_rate = 0.1
        stats = sim.measure_consistency(2000)
        print(f"{cfg['label']:<25} {stats['stale_read_rate']:>11.2%} "
              f"{stats['repair_rate']:>11.2%}")


if __name__ == "__main__":
    print("=== Quorum Demo ===")
    demo_quorum()

    print("\n=== Chain Replication Demo ===")
    demo_chain_replication()

    print("\n=== Consistency Experiment ===")
    run_consistency_experiment()
```

---

## 7. Summary

| Concept | Key Takeaway |
|---|---|
| **Single-leader** | Simple, strong consistency possible, but leader is bottleneck and failover is dangerous |
| **Multi-leader** | Low write latency across regions, but conflicts are inevitable and hard to resolve |
| **Leaderless (quorum)** | High availability via W + R > N, but only probabilistic consistency |
| **Chain replication** | Strong consistency with high read throughput (CRAQ), but write latency is sum of chain |
| **Replication lag** | Causes read-your-writes, monotonic reads, and causal ordering violations |
| **Conflict resolution** | LWW is simple but loses data; merge functions and application-level resolution are safer |
| **Anti-entropy** | Merkle trees efficiently identify divergent data for background synchronization |

### Key Questions to Ask When Choosing a Replication Strategy

1. **What consistency does the application need?** Strong → single-leader or chain. Eventual → multi-leader or leaderless.
2. **What is the read:write ratio?** Read-heavy → chain or leaderless. Write-heavy → multi-leader.
3. **Is geographic distribution required?** Yes → multi-leader or leaderless.
4. **What is the failure tolerance requirement?** High → leaderless (tunable W, R, N).
5. **Can the application resolve conflicts?** Yes → multi-leader or leaderless with custom merge. No → single-leader.

---

[Next: CRDTs and Eventual Consistency](./10_CRDTs_and_Eventual_Consistency.md)
