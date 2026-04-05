# Lesson 4: Consistency Models Deep Dive

[Overview](./00_Overview.md) | [Previous](./03_FLP_Impossibility_and_Bounds.md) | [Next](./05_Paxos_Family.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define linearizability formally using histories and real-time ordering, and identify linearization points in concurrent executions
2. Distinguish between linearizability, sequential consistency, causal consistency, and eventual consistency with precise formal criteria
3. State and prove the CAP theorem and explain common misconceptions about its implications
4. Apply the PACELC framework to reason about consistency-latency trade-offs beyond partitions
5. Implement a linearizability checker and simulate different consistency levels in code

---

## Table of Contents

1. [Why Consistency Matters](#1-why-consistency-matters)
2. [Linearizability](#2-linearizability)
3. [Sequential Consistency](#3-sequential-consistency)
4. [Causal Consistency](#4-causal-consistency)
5. [Eventual Consistency](#5-eventual-consistency)
6. [Session Guarantees](#6-session-guarantees)
7. [The Consistency Hierarchy](#7-the-consistency-hierarchy)
8. [CAP Theorem Revisited](#8-cap-theorem-revisited)
9. [PACELC: Beyond CAP](#9-pacelc-beyond-cap)
10. [Jepsen-Style Consistency Checking](#10-jepsen-style-consistency-checking)
11. [Code: Linearizability Checker](#11-code-linearizability-checker)
12. [Code: Simulating Consistency Levels](#12-code-simulating-consistency-levels)
13. [Summary](#13-summary)
14. [Practice Problems](#14-practice-problems)
15. [References](#15-references)

---

## 1. Why Consistency Matters

### 1.1 The Replication Problem

When data is replicated across multiple nodes, a fundamental question arises: **what guarantees do clients get about the data they read?**

Consider a simple scenario:

```
Client A writes x = 1 to Node 1 at time t₁
Client B reads x from Node 2 at time t₂ > t₁

Questions:
  - Must B see x = 1? (strong consistency)
  - Could B see x = null (the old value)? (weak consistency)
  - Is there a middle ground? (yes, many)
```

The answer depends on the **consistency model** -- the contract between the distributed system and its clients about what behaviors are possible.

### 1.2 The Consistency Spectrum

```
Strong ◄─────────────────────────────────────────────────► Weak

Linearizable  Sequential  Causal   PRAM   Eventual   No
                                                   Guarantee
     │            │          │       │        │         │
     │            │          │       │        │         │
  Strongest    Program    Cause-    Per-   Eventually  Chaos
  (real-time   order      effect   client  converge
   ordering)   preserved  tracked  order
```

Each point on this spectrum represents a different trade-off between:

- **Consistency**: How "up-to-date" reads are
- **Availability**: Whether the system can respond during failures
- **Performance**: Latency and throughput
- **Complexity**: How hard it is to implement and reason about

### 1.3 The Cost of Consistency

| Consistency Level | Coordination Required | Typical Latency | Availability During Partitions |
|-------------------|----------------------|-----------------|-------------------------------|
| Linearizable | Full (quorum reads + writes) | High | Low (unavailable in minority) |
| Sequential | Per-session ordering | Medium | Medium |
| Causal | Metadata tracking | Medium | High (can serve from any replica) |
| Eventual | None (async replication) | Low | High (always available) |

---

## 2. Linearizability

### 2.1 Informal Definition

Linearizability (Herlihy & Wing, 1990) is the strongest single-object consistency model. Informally:

> Every operation appears to take effect **atomically** at some instant between its invocation and response, and all operations are consistent with a single sequential order that respects real-time ordering.

In other words, a linearizable system "acts like" a single copy of the data, even though it is replicated.

### 2.2 Formal Definition

A **history** `H` is a sequence of invocation and response events for operations on a shared object. Each operation `op` consists of:

- `inv(op)`: the invocation event (when the client sends the request)
- `res(op)`: the response event (when the client receives the reply)
- The **real-time interval** of `op` is `[inv(op), res(op)]`

**Real-time ordering**: Operation `op₁` **precedes** operation `op₂` (written `op₁ <_H op₂`) if `res(op₁)` occurs before `inv(op₂)` in `H`.

**Concurrent operations**: `op₁` and `op₂` are concurrent if neither precedes the other (their real-time intervals overlap).

**Linearizability**: A history `H` is linearizable if there exists a **linearization** -- a total order `S` of the operations in `H` such that:

1. **Real-time constraint**: If `op₁ <_H op₂`, then `op₁ <_S op₂` (the linearization respects the real-time order).
2. **Sequential specification**: The total order `S`, applied to the object, produces results consistent with the object's sequential specification (e.g., a register returns the last value written).

### 2.3 Linearization Points

A **linearization point** is the specific instant within an operation's real-time interval at which the operation "takes effect." If we can assign a linearization point to each operation such that executing operations in linearization point order satisfies the sequential specification, then the history is linearizable.

```
Example: Two writes and one read on a register (initial value = 0)

Time ────────────────────────────────────────────────────────►

Client A:  |──── write(x, 1) ────|
                      ↑ linearization point

Client B:       |──── write(x, 2) ────|
                            ↑ linearization point

Client C:                |──── read(x) → 2 ────|
                                  ↑ linearization point

Linearization order: write(x,1), write(x,2), read(x)→2  ✓
  read returns 2, which is the last written value.

Alternative linearization: write(x,2), write(x,1), read(x)→2  ✗
  This would require read to return 1 (last written), not 2.
  But the real-time constraint is also violated:
  write(x,1) starts before write(x,2), so their linearization
  points must be compatible with overlap (both are ok here).
  However, read must return 1 if write(x,1) comes last → contradiction.
```

### 2.4 Linearizability Examples

**Example 1: Linearizable history**

```
Time ──────────────────────────────────────►

Client A:  |── w(x,1) ──|     |── r(x)→2 ──|
Client B:        |── w(x,2) ──|

Linearization: w(x,1), w(x,2), r(x)→2  ✓
  w(x,1) completes before r(x) starts → w(x,1) <_S r(x)  ✓
  w(x,2) completes before r(x) starts → w(x,2) <_S r(x)  ✓
  r(x) returns 2, which is the value of the last write   ✓
```

**Example 2: Non-linearizable history**

```
Time ──────────────────────────────────────►

Client A:  |── w(x,1) ──|
Client B:        |── w(x,2) ──|
Client C:                         |── r(x)→1 ──|

Is this linearizable?
  r(x) starts after both writes complete.
  Any linearization must have both writes before the read.
  Possible orders:
    w(x,1), w(x,2), r(x) → read should return 2, but returns 1 ✗
    w(x,2), w(x,1), r(x) → read should return 1 ✓
      But does this violate real-time? w(x,1) starts before w(x,2),
      but since they overlap, either order is allowed.
  So this IS linearizable with order: w(x,2), w(x,1), r(x)→1  ✓
```

**Example 3: Definitely non-linearizable**

```
Time ──────────────────────────────────────────────────────►

Client A:  |── w(x,1) ──|
Client B:                     |── w(x,2) ──|
Client C:                                       |── r(x)→1 ──|

Is this linearizable?
  w(x,1) completes before w(x,2) starts → w(x,1) <_H w(x,2)
  w(x,2) completes before r(x) starts → w(x,2) <_H r(x)
  Therefore: w(x,1) <_S w(x,2) <_S r(x)
  Last write before r(x) is w(x,2), so read must return 2.
  But read returns 1. CONTRADICTION.
  This history is NOT linearizable.  ✗
```

### 2.5 Checking Linearizability

**Theorem** (Gibbons & Korach, 1997): Checking whether a history is linearizable is NP-complete in general.

However, for single-object histories with a small number of concurrent operations, practical algorithms exist:

1. **Wing & Gong algorithm**: Enumerate all possible linearizations (exponential in worst case, but pruning makes it practical for small histories).
2. **Knossos** (used by Jepsen): Efficient linearizability checker for register histories.
3. **Porcupine** (Go): Fast linearizability checker using the P-compositionality optimization.

### 2.6 Locality and Composability

**Theorem** (Herlihy & Wing, 1990): Linearizability is a **local** property: a history `H` is linearizable if and only if, for each object `x`, the sub-history `H|x` (restricted to operations on `x`) is linearizable.

This is a powerful property: you can check linearizability object by object. Sequential consistency does NOT have this property.

### 2.7 Cost of Linearizability

Linearizability requires coordination, which has inherent costs:

```
To guarantee linearizable reads:
  Option 1: Read from leader (bottleneck, not fault-tolerant for reads)
  Option 2: Read from quorum (majority must agree)
  Option 3: LeaseRead (leader holds a lease, serves reads locally during lease)
  Option 4: ReadIndex (leader confirms it is still leader before serving read)

All options add latency compared to reading from the nearest replica.
```

**Theorem** (Attiya & Welch, 1994): In a message-passing system, any implementation of a linearizable read-write register requires at least one message round-trip (cannot be purely local).

---

## 3. Sequential Consistency

### 3.1 Definition

**Sequential consistency** (Lamport, 1979): A history is sequentially consistent if there exists a total order of all operations such that:

1. **Program order**: Operations from each process appear in the order the process issued them.
2. **Sequential specification**: The total order is consistent with the object's sequential specification.

**Key difference from linearizability**: Sequential consistency does NOT require respecting real-time ordering. Two operations from different processes may be reordered in the total order, even if one completed before the other started in real time.

### 3.2 Comparison with Linearizability

```
Linearizability:
  Respects: program order + real-time order
  Guarantee: "behaves like a single copy in real time"

Sequential consistency:
  Respects: program order ONLY (not real-time)
  Guarantee: "behaves like a single copy, but may be delayed"
```

### 3.3 Example: Sequentially Consistent but Not Linearizable

```
Time ──────────────────────────────────────────────────────►

Process P1:  |── w(x,1) ──|                |── r(y)→0 ──|
Process P2:                  |── w(y,1) ──|   |── r(x)→0 ──|

Linearizable?
  w(x,1) completes before w(y,1) starts → w(x,1) <_H w(y,1)
  w(y,1) completes before r(x) starts → w(y,1) <_H r(x)
  So: w(x,1) <_S w(y,1) <_S r(x)
  r(x) must return 1 (x was written). But r(x)→0. NOT linearizable. ✗

Sequentially consistent?
  Try order: r(y)→0, r(x)→0, w(x,1), w(y,1)
  Check program order:
    P1: w(x,1) before r(y)→0? In our total order, r(y) is first. VIOLATES P1's order. ✗

  Try order: r(x)→0, w(x,1), r(y)→0, w(y,1)
  Check program order:
    P1: w(x,1) before r(y)→0? In total order, w(x,1) is second, r(y) is third. ✓
    P2: w(y,1) before r(x)→0? In total order, w(y,1) is fourth, r(x) is first. VIOLATES P2's order. ✗

  Try order: w(x,1), r(x)→0 ... no, r(x) returns 0 but x was written to 1. ✗

  Actually, consider: r(x)→0, r(y)→0, w(x,1), w(y,1)
  P1 order: w(x,1) then r(y)→0? In total order, w(x,1) is 3rd, r(y) is 2nd. VIOLATES. ✗

  Hmm, let us try: w(y,1), r(y)→0 ... r(y) should return 1, not 0. ✗

  Actually this history IS sequentially consistent:
  Order: r(y)→0, w(y,1), r(x)→0, w(x,1)
  P1 program order: P1 does w(x,1) then r(y)→0.
    In total order: r(y)→0 is 1st, w(x,1) is 4th. VIOLATES P1's order. ✗

  Let me reconsider. The history r(x)→0, r(y)→0 means both reads return initial value 0.
  For this to be sequentially consistent, both reads must occur before the corresponding writes.

  Total order: r(y)→0, r(x)→0, w(x,1), w(y,1)
  P1 program order: w(x,1) is before r(y). In total order: r(y) is 1st, w(x,1) is 3rd. VIOLATES.

  This history is NEITHER linearizable NOR sequentially consistent.
  Let me fix the example.
```

**Corrected example**:

```
Time ──────────────────────────────────────────────────────►

Process P1:  |── w(x,1) ──|
Process P2:                               |── r(x)→0 ──|

Linearizable? No.
  w(x,1) completes before r(x) starts → r(x) must return 1. But returns 0. ✗

Sequentially consistent? Yes!
  Total order: r(x)→0, w(x,1)
  P1 program order: only w(x,1) → satisfied trivially.
  P2 program order: only r(x)→0 → satisfied trivially.
  Sequential spec: r(x) happens before w(x,1), so x has initial value 0. ✓

The key: sequential consistency allows reordering across processes,
even if one operation completed before another started in real time.
```

### 3.4 Properties

| Property | Linearizability | Sequential Consistency |
|----------|----------------|----------------------|
| Respects real-time | Yes | No |
| Respects program order | Yes | Yes |
| Composable (local) | Yes | **No** |
| NP-complete to check | Yes | Yes |
| Used in practice | Databases, linearizable KV stores | CPU memory models, Zookeeper |

**Non-composability of sequential consistency**: Two objects may each be sequentially consistent in isolation, but the combined history may not be. This makes it much harder to build complex systems from sequentially consistent components.

---

## 4. Causal Consistency

### 4.1 Definition

**Causal consistency** (Ahamad et al., 1995): A history is causally consistent if all processes observe causally related operations in the same order. Concurrent operations (not causally related) may be observed in different orders by different processes.

Formally, a replicated store is causally consistent if, for any two operations `a` and `b`:

```
If a →(causally precedes) b, then every process sees a before b.
If a ‖ b (concurrent), different processes may see them in different orders.
```

### 4.2 Causal Precedence

Two operations are causally related (`a → b`) if:

1. **Same session**: `a` and `b` are issued by the same client session, and `a` precedes `b`.
2. **Reads-from**: `b` reads a value written by `a`.
3. **Transitivity**: `a → c` and `c → b` implies `a → b`.

### 4.3 Example

```
Process P1: w(x, 1)
Process P2: r(x) → 1; w(y, 2)     [reads x=1, then writes y=2]
Process P3: r(y) → 2; r(x) → ?

Causal chain: w(x,1) → r(x)→1 → w(y,2) → r(y)→2

Since r(y)→2 causally depends on w(y,2), which causally depends on
r(x)→1, which causally depends on w(x,1):

  r(y)→2 causally depends on w(x,1)

Therefore, P3 MUST see w(x,1) before or by the time it sees w(y,2).
So r(x) at P3 must return 1 (or a later value).

Under causal consistency:  r(x) = 1  ✓
Under eventual consistency: r(x) could be 0 (stale)  — depends on timing
```

### 4.4 Implementing Causal Consistency

Causal consistency can be implemented using:

1. **Vector clocks / version vectors**: Track causal dependencies and delay delivery until dependencies are satisfied (see Lesson 02).
2. **Causal broadcast**: Use the causal broadcast protocol to ensure messages are delivered in causal order.
3. **Dependency tracking**: Each write carries the set of operations it depends on.

```python
class CausalStore:
    """
    A causally consistent key-value store using dependency tracking.
    Each write carries a vector clock summarizing its causal past.
    Reads are blocked until the causal dependencies are satisfied.
    """

    def __init__(self, replica_id: str, all_replicas: list[str]):
        self.replica_id = replica_id
        self.all_replicas = sorted(all_replicas)
        self.idx = self.all_replicas.index(replica_id)
        self.n = len(all_replicas)

        # Local state
        self.store: dict[str, tuple[any, list[int]]] = {}  # key -> (value, vector_clock)
        self.vc = [0] * self.n  # local vector clock

        # Pending writes from other replicas (waiting for causal deps)
        self.pending: list[tuple[str, any, list[int]]] = []

    def write(self, key: str, value) -> list[int]:
        """Write a value (locally). Returns the write's vector clock."""
        self.vc[self.idx] += 1
        write_vc = list(self.vc)
        self.store[key] = (value, write_vc)
        return write_vc

    def read(self, key: str) -> tuple[any, list[int]]:
        """Read a value. Returns (value, vector_clock) or (None, [0]*n)."""
        if key in self.store:
            return self.store[key]
        return (None, [0] * self.n)

    def receive_write(self, key: str, value, write_vc: list[int]):
        """Receive a replicated write from another replica."""
        self.pending.append((key, value, write_vc))
        self._try_apply_pending()

    def _can_apply(self, write_vc: list[int]) -> bool:
        """Check if all causal dependencies are satisfied."""
        for i in range(self.n):
            if i == self.idx:
                continue  # skip self
            if write_vc[i] > self.vc[i]:
                return False  # missing a causally preceding write
        return True

    def _try_apply_pending(self):
        """Apply pending writes whose dependencies are satisfied."""
        applied = True
        while applied:
            applied = False
            remaining = []
            for key, value, write_vc in self.pending:
                if self._can_apply(write_vc):
                    # Apply the write
                    existing_vc = self.store.get(key, (None, [0] * self.n))[1]
                    # Only apply if this write is newer
                    if not all(write_vc[i] <= existing_vc[i] for i in range(self.n)):
                        self.store[key] = (value, write_vc)
                    # Update local vector clock
                    for i in range(self.n):
                        self.vc[i] = max(self.vc[i], write_vc[i])
                    applied = True
                else:
                    remaining.append((key, value, write_vc))
            self.pending = remaining
```

### 4.5 Causal Consistency in Practice

| System | Causal Consistency Implementation |
|--------|----------------------------------|
| COPS (Lloyd et al., 2011) | Explicit dependency tracking with nearest-replica reads |
| Eiger | Multi-key causal transactions with dependency metadata |
| MongoDB (causal sessions) | Hybrid logical clocks for causal ordering |
| Riak (with CRDTs) | Vector clocks + CRDTs for conflict-free causal updates |

---

## 5. Eventual Consistency

### 5.1 Definition

**Eventual consistency**: If no new updates are made to a data item, all replicas will **eventually** converge to the same value.

Formally:

```
For any key k, if the last write to k occurs at time t:
  ∃ time T > t such that ∀ replicas r, at time T:
    read(r, k) returns the value of the last write

The convergence time T is unbounded (no guarantee on how long it takes).
```

### 5.2 What Eventual Consistency Does NOT Guarantee

| Non-guarantee | Consequence |
|--------------|-------------|
| Read-your-writes | You write x=5, immediately read x, get old value 3 |
| Monotonic reads | You read x=5, read again, get x=3 (went backward) |
| Causal ordering | You see a reply to a message without seeing the message |
| Write ordering | Writes by the same client may be reordered |
| Convergence time | No bound on how long it takes to converge |
| Conflict resolution | When replicas diverge, who wins? (implementation-specific) |

### 5.3 Anti-Entropy Protocols

Eventual consistency relies on **anti-entropy** mechanisms to propagate updates:

```python
class AntiEntropyStore:
    """
    Eventually consistent key-value store using anti-entropy.
    Periodically synchronizes with random peers.
    """

    def __init__(self, node_id: str, peers: list[str]):
        self.node_id = node_id
        self.peers = peers
        self.store: dict[str, tuple[any, float]] = {}  # key -> (value, timestamp)

    def write(self, key: str, value):
        """Write with last-writer-wins (LWW) timestamp."""
        self.store[key] = (value, time.time())

    def read(self, key: str):
        """Read local value (may be stale)."""
        if key in self.store:
            return self.store[key][0]
        return None

    def anti_entropy_push(self, peer_store: 'AntiEntropyStore'):
        """Push local state to a peer (epidemic/gossip protocol)."""
        for key, (value, ts) in self.store.items():
            peer_val = peer_store.store.get(key)
            if peer_val is None or ts > peer_val[1]:
                peer_store.store[key] = (value, ts)

    def anti_entropy_pull(self, peer_store: 'AntiEntropyStore'):
        """Pull state from a peer."""
        for key, (value, ts) in peer_store.store.items():
            local_val = self.store.get(key)
            if local_val is None or ts > local_val[1]:
                self.store[key] = (value, ts)

    def anti_entropy_round(self, all_stores: dict[str, 'AntiEntropyStore']):
        """One round of anti-entropy: pick a random peer and sync."""
        peer_id = random.choice(self.peers)
        peer = all_stores[peer_id]
        # Push-pull: bidirectional sync
        self.anti_entropy_push(peer)
        self.anti_entropy_pull(peer)
```

### 5.4 Conflict Resolution Strategies

When concurrent writes create conflicts, eventual consistency systems use one of these strategies:

| Strategy | How it works | Pros | Cons |
|----------|-------------|------|------|
| Last-Writer-Wins (LWW) | Higher timestamp wins | Simple, automatic | Data loss (loser is discarded) |
| Multi-value (siblings) | Return all concurrent values to client | No data loss | Client must resolve |
| CRDTs | Use algebraic merge function | Automatic, no conflicts | Limited data types |
| Application merge | Custom merge logic | Flexible | Complex application code |
| Operational transform | Transform concurrent operations | Good for text editing | Complex, tricky to implement |

---

## 6. Session Guarantees

Session guarantees (Terry et al., 1994) provide per-client consistency properties that are stronger than eventual consistency but weaker than sequential consistency.

### 6.1 The Four Session Guarantees

**Read Your Writes**: If a client writes value `v`, any subsequent read by the same client returns `v` or a later value.

```
Client A: write(x, 5) → OK
Client A: read(x) → must return 5 (or later)
           NOT allowed: read(x) → 3 (stale)
```

**Monotonic Reads**: If a client reads value `v` for key `x`, any subsequent read of `x` by the same client returns `v` or a later value.

```
Client A: read(x) → 5
Client A: read(x) → must return 5 or later
           NOT allowed: read(x) → 3 (went backward)
```

**Monotonic Writes**: Writes from the same client are applied in order at all replicas.

```
Client A: write(x, 1) then write(x, 2)
All replicas must apply write(x,1) before write(x,2)
NOT allowed: some replica applies write(x,2) first
```

**Writes Follow Reads**: If a client reads `x`, then writes `y`, the write to `y` is ordered after the read of `x` at all replicas. (Captures the "reply follows message" pattern.)

```
Client A: read(x) → 5, then write(y, "response to x=5")
All replicas see the causal dependency:
  the state that produced x=5 is visible before write(y) is applied
```

### 6.2 Implementation

Session guarantees can be implemented using **version vectors** or **session tokens**:

```python
class SessionToken:
    """
    Track session state for implementing session guarantees.
    The token is sent with each request and updated with each response.
    """

    def __init__(self):
        self.read_vector: dict[str, int] = {}   # key -> version last read
        self.write_vector: dict[str, int] = {}   # key -> version last written

    def after_read(self, key: str, version: int):
        """Update token after a read."""
        self.read_vector[key] = max(self.read_vector.get(key, 0), version)

    def after_write(self, key: str, version: int):
        """Update token after a write."""
        self.write_vector[key] = max(self.write_vector.get(key, 0), version)

    def min_read_version(self, key: str) -> int:
        """Minimum version that a read of key must return (monotonic reads + RYW)."""
        return max(
            self.read_vector.get(key, 0),    # monotonic reads
            self.write_vector.get(key, 0),    # read your writes
        )
```

### 6.3 Session Guarantees in Practice

| System | Session Guarantees Provided |
|--------|---------------------------|
| Azure Cosmos DB | Configurable (session consistency level provides all four) |
| MongoDB | Causal sessions (all four + causal ordering) |
| DynamoDB | Read-your-writes (with consistent reads) |
| Cassandra | Read-your-writes (with LOCAL_QUORUM) |

---

## 7. The Consistency Hierarchy

### 7.1 Hierarchy Diagram

```
                    Strict Serializability
                    (linearizable + serializable)
                           │
                    Linearizability
                    (single-object, real-time)
                           │
                    Sequential Consistency
                    (single-object, program-order)
                           │
                    Causal Consistency
                    (respects causality)
                      /          \
              PRAM                Writes Follow Reads
          (per-process order)     (read-write causality)
              |        \           /
        Monotonic      Read Your    Monotonic
         Reads         Writes        Writes
              \          |          /
               Eventual Consistency
               (convergence only)
```

### 7.2 Comparison Table

| Model | Real-time | Program Order | Causal | Convergence | Operations |
|-------|-----------|---------------|--------|-------------|------------|
| Linearizable | Yes | Yes | Yes | Yes | Single-object |
| Sequential | No | Yes | Partial | Yes | Single-object |
| Causal | No | Yes | Yes | Yes | Multi-object possible |
| PRAM | No | Yes (per-proc) | No | Yes | Single-object |
| Eventual | No | No | No | Yes | Single-object |

### 7.3 Multi-Object Consistency

Single-object consistency models (linearizability, sequential consistency) apply to individual data items. For transactions spanning multiple objects, we need:

| Model | Description | Single-object analog |
|-------|-------------|---------------------|
| Strict serializability | Transactions appear serialized in real-time order | Linearizability |
| Serializability | Transactions appear serialized (any order) | Sequential consistency |
| Snapshot isolation | Transactions read from a consistent snapshot | N/A (weaker than serializable) |
| Read committed | Each query reads committed data (but may see different snapshots) | N/A |
| Read uncommitted | May read uncommitted (dirty) data | No guarantee |

**Strict serializability** = Linearizability + Serializability. This is the gold standard for databases but the most expensive to implement.

---

## 8. CAP Theorem Revisited

### 8.1 Informal Statement

The CAP theorem (Brewer, 2000; Gilbert & Lynch, 2002) states that a distributed data store can provide at most **two out of three** guarantees simultaneously:

- **Consistency** (C): Every read receives the most recent write (linearizability)
- **Availability** (A): Every request receives a non-error response (no timeouts)
- **Partition tolerance** (P): The system continues to operate despite network partitions

### 8.2 Formal Statement

**Theorem** (Gilbert & Lynch, 2002): It is impossible to implement a read/write data object that guarantees all of the following in an asynchronous network:

1. **Consistency (Linearizability)**: All operations act as if executing on a single copy
2. **Availability**: Every request to a non-failing node receives a response
3. **Partition tolerance**: The system functions correctly despite arbitrary message loss between nodes

### 8.3 Proof Sketch

The proof is by contradiction. Consider a system with two nodes, `G₁` and `G₂`, connected by a network that can partition.

```
Setup:
  Node G₁        Node G₂
    │                │
    │    PARTITION    │
    │     ✕✕✕✕✕✕     │
    │                │

Execution:
  1. Client writes x = v₁ to G₁
  2. Network partitions (G₁ and G₂ cannot communicate)
  3. Client reads x from G₂

If Available + Partition-tolerant:
  G₂ must respond to the read (availability).
  G₂ has not received the write (partition).
  G₂ returns stale value → NOT consistent (linearizable). ✗

If Consistent + Partition-tolerant:
  G₂ must return v₁ (consistency).
  G₂ cannot learn v₁ (partition).
  G₂ must not respond → NOT available. ✗

If Consistent + Available:
  Must respond correctly to all requests.
  But if a partition occurs, this is impossible.
  So the system is NOT partition-tolerant. ✗
```

### 8.4 Common Misconceptions

**Misconception 1: "Pick two out of three"**

This framing suggests CA, CP, and AP are equally valid choices. In reality, network partitions **will happen** in any distributed system (cables get cut, switches fail, cloud AZs lose connectivity). So partition tolerance is not optional -- you must handle partitions. The real choice is:

> **During a partition, do you sacrifice consistency (AP) or availability (CP)?**

**Misconception 2: "CAP applies to entire systems"**

CAP applies to **individual operations** on **individual data items**. A system can make different trade-offs for different data:

```
Example: E-commerce system
  - Product catalog: AP (eventual consistency, always available)
  - Shopping cart: AP (merge conflicts with CRDTs)
  - Inventory count: CP (linearizable, may be unavailable during partition)
  - Payment processing: CP (must be consistent, may timeout)
```

**Misconception 3: "C means any consistency"**

In CAP, "C" specifically means **linearizability** -- the strongest single-object consistency model. Weaker models (causal, eventual) can be achieved alongside availability and partition tolerance.

**Misconception 4: "You lose C or A permanently"**

The trade-off applies only **during partitions**. When the network is healthy:

```
No partition:  C + A are both achievable (e.g., Raft with healthy leader)
Partition:     Must choose C or A for each request
After healing: C + A resume
```

### 8.5 CAP in Practice

| System | During Partition | Normal Operation | Classification |
|--------|-----------------|-----------------|----------------|
| Spanner | CP (unavailable in minority) | CA (with TrueTime) | CP |
| DynamoDB | AP (sloppy quorums) | Tunable (can be CP) | AP default |
| ZooKeeper | CP (minority unavailable) | CP | CP |
| Cassandra | AP (tunable) | AP or CP (tunable) | AP default |
| CockroachDB | CP (unavailable in minority) | CA | CP |
| Riak | AP (siblings) | AP | AP |
| etcd | CP (Raft majority) | CP | CP |

---

## 9. PACELC: Beyond CAP

### 9.1 Motivation

The CAP theorem only tells us what happens **during a partition** (P). But partitions are rare. What trade-offs exist during **normal operation** (no partition)?

### 9.2 The PACELC Framework

Daniel Abadi (2012) proposed PACELC:

```
If Partition (P):
  Choose Availability (A) or Consistency (C)
Else (E):
  Choose Latency (L) or Consistency (C)
```

This captures the fundamental insight that even without partitions, there is a trade-off between **consistency** and **latency** (because consistency requires coordination, which adds latency).

### 9.3 PACELC Classification

| System | If P: A or C? | Else: L or C? | Full Classification |
|--------|--------------|---------------|---------------------|
| DynamoDB | PA | EL | PA/EL |
| Cassandra | PA | EL | PA/EL (default) |
| Riak | PA | EL | PA/EL |
| MongoDB | PA | EC | PA/EC (with w:majority) |
| Spanner | PC | EC | PC/EC |
| CockroachDB | PC | EC | PC/EC |
| ZooKeeper | PC | EC | PC/EC |
| etcd | PC | EC | PC/EC |
| PNUTS (Yahoo) | PC | EL | PC/EL |
| Megastore (Google) | PC | EC | PC/EC |

### 9.4 The Latency-Consistency Trade-off

Even without partitions, synchronous replication (for consistency) adds latency:

```
Eventual consistency (EL):
  Client → Nearest Replica → ACK
  Latency: ~1 RTT (local)

Strong consistency (EC):
  Client → Leader → Replicate to Majority → ACK
  Latency: ~2-3 RTTs (may cross data centers)

Numbers (US East to US West):
  Local replica: ~1 ms
  Cross-region replication: ~60-80 ms
  Quorum write (majority across 3 regions): ~120-160 ms
```

### 9.5 Tunable Consistency

Some systems let you choose consistency per-operation:

```python
# Cassandra-style tunable consistency
def write(key, value, consistency_level):
    """
    consistency_level determines how many replicas must acknowledge.
    """
    if consistency_level == "ONE":
        # Write to 1 replica, acknowledge immediately
        # Fast but weak consistency (PA/EL)
        replicate_async(key, value)
        return "OK"

    elif consistency_level == "QUORUM":
        # Write to majority of replicas before acknowledging
        # Slower but strong consistency (PC/EC)
        acks = replicate_sync(key, value, count=quorum_size())
        if acks >= quorum_size():
            return "OK"
        else:
            raise TimeoutError("Could not reach quorum")

    elif consistency_level == "ALL":
        # Write to ALL replicas before acknowledging
        # Slowest, strongest, but least available
        acks = replicate_sync(key, value, count=total_replicas())
        if acks == total_replicas():
            return "OK"
        else:
            raise TimeoutError("Could not reach all replicas")


def quorum_size():
    """Quorum = floor(n/2) + 1 for n replicas."""
    return total_replicas() // 2 + 1
```

**Quorum intersection guarantee**: If `W + R > N` (write quorum + read quorum > total replicas), then every read quorum overlaps with every write quorum, ensuring the read sees the latest write. This is how systems like Cassandra provide tunable strong consistency.

---

## 10. Jepsen-Style Consistency Checking

### 10.1 What is Jepsen?

Jepsen (by Kyle Kingsbury) is a framework for testing distributed systems' consistency claims. It:

1. Sets up a distributed system (e.g., a database cluster)
2. Runs a workload (concurrent reads and writes)
3. Injects failures (network partitions, process crashes, clock skew)
4. Records a history of all operations (invocations and responses)
5. Checks whether the history satisfies the claimed consistency model

### 10.2 How Consistency Checking Works

```
Step 1: Run workload and record history

  Client 1: write(x, 1) [t=100, ok at t=110]
  Client 2: read(x) [t=105, returned 0 at t=108]
  Client 3: write(x, 2) [t=112, ok at t=120]
  Client 2: read(x) [t=115, returned 2 at t=118]
  ...

Step 2: Model the history

  Operation 1: write(x, 1), invoked=100, completed=110
  Operation 2: read(x)→0,   invoked=105, completed=108
  Operation 3: write(x, 2), invoked=112, completed=120
  Operation 4: read(x)→2,   invoked=115, completed=118

Step 3: Check linearizability

  Try to find a linearization (total order respecting real-time):
  Op2 must come before Op1 completes (Op2 finishes at 108, Op1 finishes at 110).
  But Op2 overlaps with Op1 (invoked 105 < 110).
  Both orders are possible for Op1 and Op2.

  If Op2 is linearized before Op1: read(x)→0, then write(x,1). OK (x starts at 0). ✓
  If Op1 is linearized before Op2: write(x,1), then read(x)→0. NOT OK (should return 1). ✗

  So only the first linearization works for Op1 and Op2.
  Continue with Op3 and Op4...
```

### 10.3 Notable Jepsen Findings

| System | Claimed | Found | Issue |
|--------|---------|-------|-------|
| MongoDB (2013) | Strong consistency | Lost writes during partition | Rollback of committed data |
| Elasticsearch (2014) | Sequential consistency | Split brain | No proper quorum |
| RabbitMQ (2014) | Durable queues | Lost messages | Partition handling bug |
| etcd (2020) | Linearizable | Stale reads | Lease-based reads incorrect |
| Redis Cluster (2013) | Consistent | Lost writes | No consensus for writes |
| CockroachDB (2020) | Serializable | Correct | Well-implemented Raft |

---

## 11. Code: Linearizability Checker

### 11.1 Data Structures

```python
"""
Linearizability checker for a read-write register.
Uses the Wing & Gong algorithm with pruning.
"""

from dataclasses import dataclass
from itertools import permutations
from typing import Optional


@dataclass
class Operation:
    """An operation on a read-write register."""
    op_id: int
    op_type: str      # "write" or "read"
    key: str
    value: Optional[int]        # value written (write) or value read (read)
    invoke_time: float
    complete_time: float

    def __repr__(self):
        if self.op_type == "write":
            return f"w({self.key},{self.value})[{self.invoke_time}-{self.complete_time}]"
        else:
            return f"r({self.key})→{self.value}[{self.invoke_time}-{self.complete_time}]"


def real_time_precedes(op1: Operation, op2: Operation) -> bool:
    """Check if op1 completes before op2 starts (real-time precedence)."""
    return op1.complete_time < op2.invoke_time


def is_valid_sequential_history(ops: list[Operation], initial_value: int = 0) -> bool:
    """
    Check if a sequence of operations is a valid sequential execution
    of a read-write register.
    """
    store: dict[str, int] = {}

    for op in ops:
        if op.op_type == "write":
            store[op.key] = op.value
        elif op.op_type == "read":
            current = store.get(op.key, initial_value)
            if op.value != current:
                return False
    return True


def check_linearizability_brute(
    history: list[Operation],
    initial_value: int = 0,
) -> tuple[bool, Optional[list[Operation]]]:
    """
    Brute-force linearizability checker.
    Try all permutations and check if any is a valid linearization.

    WARNING: O(n!) complexity. Only suitable for small histories (n ≤ 10).
    """
    n = len(history)

    for perm in permutations(range(n)):
        ordered = [history[i] for i in perm]

        # Check real-time constraint
        valid_rt = True
        for i in range(n):
            for j in range(i + 1, n):
                original_i = perm[i]
                original_j = perm[j]
                # If op at position j precedes op at position i in real time,
                # then this ordering violates real-time constraint
                if real_time_precedes(history[original_j], history[original_i]):
                    valid_rt = False
                    break
            if not valid_rt:
                break

        if not valid_rt:
            continue

        # Check sequential specification
        if is_valid_sequential_history(ordered, initial_value):
            return (True, ordered)

    return (False, None)
```

### 11.2 Efficient Linearizability Checker

```python
def check_linearizability_wg(
    history: list[Operation],
    initial_value: int = 0,
) -> tuple[bool, Optional[list[Operation]]]:
    """
    Wing & Gong style linearizability checker with pruning.
    Uses DFS with backtracking, pruning invalid prefixes early.

    Much faster than brute force for practical histories.
    """
    n = len(history)
    if n == 0:
        return (True, [])

    # Build precedence graph
    # must_precede[i] = set of op indices that must come before op i
    must_precede: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and real_time_precedes(history[i], history[j]):
                must_precede[j].add(i)

    # DFS with backtracking
    linearization: list[int] = []
    used: set[int] = set()
    store: dict[str, int] = {}  # current register state

    def dfs() -> bool:
        if len(linearization) == n:
            return True

        # Find candidate operations: those whose predecessors are all used
        candidates = []
        for i in range(n):
            if i not in used and must_precede[i].issubset(used):
                candidates.append(i)

        for i in candidates:
            op = history[i]

            # Check if adding this operation is consistent
            if op.op_type == "read":
                current = store.get(op.key, initial_value)
                if op.value != current:
                    continue  # prune: read returns wrong value

            # Apply operation
            old_value = store.get(op.key)
            if op.op_type == "write":
                store[op.key] = op.value

            linearization.append(i)
            used.add(i)

            if dfs():
                return True

            # Backtrack
            linearization.pop()
            used.discard(i)
            if op.op_type == "write":
                if old_value is not None:
                    store[op.key] = old_value
                else:
                    del store[op.key]

        return False

    if dfs():
        return (True, [history[i] for i in linearization])
    return (False, None)
```

### 11.3 Testing the Checker

```python
def test_linearizability_checker():
    """Test the linearizability checker with known examples."""

    print("="*60)
    print("LINEARIZABILITY CHECKER TESTS")
    print("="*60)

    # Test 1: Simple linearizable history
    h1 = [
        Operation(0, "write", "x", 1, 0, 10),
        Operation(1, "read", "x", 1, 15, 20),
    ]
    result, order = check_linearizability_wg(h1)
    print(f"\nTest 1 (simple write-read): {'PASS' if result else 'FAIL'}")
    if order:
        print(f"  Linearization: {order}")

    # Test 2: Non-linearizable (stale read after write completes)
    h2 = [
        Operation(0, "write", "x", 1, 0, 10),
        Operation(1, "write", "x", 2, 15, 25),
        Operation(2, "read", "x", 1, 30, 35),  # reads old value after both writes
    ]
    result, order = check_linearizability_wg(h2)
    print(f"\nTest 2 (stale read): {'PASS' if not result else 'FAIL'}")
    print(f"  Linearizable: {result} (expected False)")

    # Test 3: Concurrent operations (linearizable)
    h3 = [
        Operation(0, "write", "x", 1, 0, 15),
        Operation(1, "write", "x", 2, 5, 20),   # overlaps with write(x,1)
        Operation(2, "read", "x", 2, 25, 30),
    ]
    result, order = check_linearizability_wg(h3)
    print(f"\nTest 3 (concurrent writes, read 2): {'PASS' if result else 'FAIL'}")
    if order:
        print(f"  Linearization: {order}")

    # Test 4: Concurrent reads returning different values (linearizable if writes overlap)
    h4 = [
        Operation(0, "write", "x", 1, 0, 20),
        Operation(1, "write", "x", 2, 10, 30),  # overlaps
        Operation(2, "read", "x", 1, 12, 18),   # during overlap, reads 1
        Operation(3, "read", "x", 2, 15, 25),   # during overlap, reads 2
    ]
    result, order = check_linearizability_wg(h4)
    print(f"\nTest 4 (concurrent everything): {'PASS' if result else 'FAIL'}")
    if order:
        print(f"  Linearization: {order}")

    # Test 5: Clearly non-linearizable (time travel)
    h5 = [
        Operation(0, "write", "x", 1, 0, 10),
        Operation(1, "write", "x", 2, 20, 30),
        Operation(2, "read", "x", 2, 35, 40),
        Operation(3, "read", "x", 1, 45, 50),  # reads 1 after reading 2!
    ]
    result, order = check_linearizability_wg(h5)
    print(f"\nTest 5 (time-travel read): {'PASS' if not result else 'FAIL'}")
    print(f"  Linearizable: {result} (expected False)")

    # Test 6: Multi-key history
    h6 = [
        Operation(0, "write", "x", 1, 0, 10),
        Operation(1, "write", "y", 2, 5, 15),
        Operation(2, "read", "x", 1, 20, 25),
        Operation(3, "read", "y", 2, 22, 28),
    ]
    result, order = check_linearizability_wg(h6)
    print(f"\nTest 6 (multi-key): {'PASS' if result else 'FAIL'}")
    if order:
        print(f"  Linearization: {order}")


test_linearizability_checker()
```

---

## 12. Code: Simulating Consistency Levels

### 12.1 Multi-Level Consistency Store

```python
"""
Simulate a replicated key-value store operating at different
consistency levels. Shows the observable differences between
linearizable, sequentially consistent, causally consistent,
and eventually consistent stores.
"""

import random
import time
import threading
from collections import defaultdict
from enum import Enum


class ConsistencyLevel(Enum):
    LINEARIZABLE = "linearizable"
    SEQUENTIAL = "sequential"
    CAUSAL = "causal"
    EVENTUAL = "eventual"


class ReplicatedStore:
    """
    A simulated replicated store with configurable consistency.
    """

    def __init__(
        self,
        num_replicas: int = 3,
        consistency: ConsistencyLevel = ConsistencyLevel.LINEARIZABLE,
        replication_delay_ms: float = 50,
    ):
        self.num_replicas = num_replicas
        self.consistency = consistency
        self.replication_delay = replication_delay_ms / 1000

        # Each replica has its own copy of the data
        self.replicas: list[dict[str, tuple[int, float]]] = [
            {} for _ in range(num_replicas)
        ]  # key -> (value, write_timestamp)

        # Replication queue (for async replication)
        self.replication_queue: list[tuple[str, int, float, int]] = []
        # (key, value, timestamp, source_replica)

        # Operation history for checking
        self.history: list[dict] = []
        self.lock = threading.Lock()
        self.global_order_counter = 0

    def write(self, key: str, value: int, client_id: str = "default") -> dict:
        """Write a value to the store."""
        with self.lock:
            self.global_order_counter += 1
            ts = time.time()
            op = {
                "type": "write",
                "key": key,
                "value": value,
                "client": client_id,
                "invoke_time": ts,
                "global_order": self.global_order_counter,
            }

            if self.consistency == ConsistencyLevel.LINEARIZABLE:
                # Synchronous replication to majority before acknowledging
                quorum = self.num_replicas // 2 + 1
                for i in range(quorum):
                    self.replicas[i][key] = (value, ts)
                # Async replication to remaining
                for i in range(quorum, self.num_replicas):
                    self.replication_queue.append((key, value, ts, 0))

            elif self.consistency == ConsistencyLevel.SEQUENTIAL:
                # Write to primary (replica 0) only, async to others
                self.replicas[0][key] = (value, ts)
                for i in range(1, self.num_replicas):
                    self.replication_queue.append((key, value, ts, 0))

            elif self.consistency == ConsistencyLevel.CAUSAL:
                # Write to local replica, track causal dependencies
                replica_id = hash(client_id) % self.num_replicas
                self.replicas[replica_id][key] = (value, ts)
                for i in range(self.num_replicas):
                    if i != replica_id:
                        self.replication_queue.append((key, value, ts, replica_id))

            elif self.consistency == ConsistencyLevel.EVENTUAL:
                # Write to random replica, lazy propagation
                replica_id = random.randint(0, self.num_replicas - 1)
                self.replicas[replica_id][key] = (value, ts)
                for i in range(self.num_replicas):
                    if i != replica_id:
                        self.replication_queue.append((key, value, ts, replica_id))

            op["complete_time"] = time.time()
            self.history.append(op)
            return op

    def read(self, key: str, client_id: str = "default") -> dict:
        """Read a value from the store."""
        with self.lock:
            self.global_order_counter += 1
            ts = time.time()
            op = {
                "type": "read",
                "key": key,
                "client": client_id,
                "invoke_time": ts,
                "global_order": self.global_order_counter,
            }

            if self.consistency == ConsistencyLevel.LINEARIZABLE:
                # Read from quorum (majority)
                quorum = self.num_replicas // 2 + 1
                values = []
                for i in range(quorum):
                    if key in self.replicas[i]:
                        values.append(self.replicas[i][key])
                # Return the latest value
                if values:
                    latest = max(values, key=lambda x: x[1])
                    op["value"] = latest[0]
                else:
                    op["value"] = None

            elif self.consistency == ConsistencyLevel.SEQUENTIAL:
                # Read from primary (ensures program order)
                if key in self.replicas[0]:
                    op["value"] = self.replicas[0][key][0]
                else:
                    op["value"] = None

            elif self.consistency == ConsistencyLevel.CAUSAL:
                # Read from local replica (may be stale for non-causal)
                replica_id = hash(client_id) % self.num_replicas
                if key in self.replicas[replica_id]:
                    op["value"] = self.replicas[replica_id][key][0]
                else:
                    op["value"] = None

            elif self.consistency == ConsistencyLevel.EVENTUAL:
                # Read from random replica (may be very stale)
                replica_id = random.randint(0, self.num_replicas - 1)
                if key in self.replicas[replica_id]:
                    op["value"] = self.replicas[replica_id][key][0]
                else:
                    op["value"] = None

            op["complete_time"] = time.time()
            self.history.append(op)
            return op

    def apply_pending_replications(self, fraction: float = 1.0):
        """
        Apply a fraction of pending replications.
        Simulates async replication delay.
        """
        with self.lock:
            n = int(len(self.replication_queue) * fraction)
            to_apply = self.replication_queue[:n]
            self.replication_queue = self.replication_queue[n:]

            for key, value, ts, source in to_apply:
                for i in range(self.num_replicas):
                    if i != source:
                        existing = self.replicas[i].get(key)
                        if existing is None or ts > existing[1]:
                            self.replicas[i][key] = (value, ts)


def demonstrate_consistency_differences():
    """
    Run the same workload under different consistency levels
    and observe the behavioral differences.
    """
    print("="*70)
    print("CONSISTENCY LEVEL COMPARISON")
    print("="*70)

    for level in ConsistencyLevel:
        print(f"\n{'─'*70}")
        print(f"Consistency Level: {level.value.upper()}")
        print(f"{'─'*70}")

        store = ReplicatedStore(
            num_replicas=3,
            consistency=level,
            replication_delay_ms=50,
        )

        # Workload: Client A writes, Client B reads
        store.write("x", 1, client_id="A")
        store.write("x", 2, client_id="A")

        # Apply some replications (simulate partial convergence)
        store.apply_pending_replications(fraction=0.5)

        # Client B reads immediately
        read1 = store.read("x", client_id="B")
        print(f"  Client A writes x=1, then x=2")
        print(f"  Client B reads x → {read1['value']}")

        # Apply remaining replications
        store.apply_pending_replications(fraction=1.0)

        # Client B reads again after convergence
        read2 = store.read("x", client_id="B")
        print(f"  After convergence, Client B reads x → {read2['value']}")

        # Stale read check
        if read1['value'] != 2:
            print(f"  ⚠ Stale read detected! (expected 2, got {read1['value']})")
        else:
            print(f"  ✓ Read returned latest value")

        # Check: do all replicas agree?
        values = set()
        for r in store.replicas:
            if "x" in r:
                values.add(r["x"][0])
        if len(values) <= 1:
            print(f"  ✓ All replicas converged to same value")
        else:
            print(f"  ⚠ Replicas diverged: {values}")


demonstrate_consistency_differences()
```

### 12.2 Anomaly Detection

```python
def detect_anomalies(history: list[dict]) -> list[str]:
    """
    Detect common consistency anomalies in a recorded history.
    """
    anomalies = []

    # Group operations by client
    by_client: dict[str, list[dict]] = defaultdict(list)
    for op in history:
        by_client[op["client"]].append(op)

    # Check read-your-writes
    for client, ops in by_client.items():
        last_write: dict[str, int] = {}
        for op in sorted(ops, key=lambda x: x["invoke_time"]):
            if op["type"] == "write":
                last_write[op["key"]] = op["value"]
            elif op["type"] == "read":
                key = op["key"]
                if key in last_write and op["value"] != last_write[key]:
                    anomalies.append(
                        f"READ-YOUR-WRITES violation: {client} wrote "
                        f"{key}={last_write[key]} but read {key}={op['value']}"
                    )

    # Check monotonic reads
    for client, ops in by_client.items():
        last_read: dict[str, int] = {}
        for op in sorted(ops, key=lambda x: x["invoke_time"]):
            if op["type"] == "read" and op["value"] is not None:
                key = op["key"]
                if key in last_read:
                    # Check if value went "backward" (crude check using value ordering)
                    if op["value"] < last_read[key]:
                        anomalies.append(
                            f"MONOTONIC-READS violation: {client} read "
                            f"{key}={last_read[key]} then {key}={op['value']}"
                        )
                last_read[key] = op["value"]

    # Check for stale reads (read returning old value after a write completed)
    writes_by_key: dict[str, list[dict]] = defaultdict(list)
    reads_by_key: dict[str, list[dict]] = defaultdict(list)
    for op in history:
        if op["type"] == "write":
            writes_by_key[op["key"]].append(op)
        else:
            reads_by_key[op["key"]].append(op)

    for key in reads_by_key:
        for read_op in reads_by_key[key]:
            for write_op in writes_by_key.get(key, []):
                # If write completed before read started, read should see it (linearizable)
                if (write_op["complete_time"] < read_op["invoke_time"] and
                        read_op["value"] is not None and
                        read_op["value"] != write_op["value"]):
                    # Check if there is a later write that the read might have seen
                    later_writes = [
                        w for w in writes_by_key[key]
                        if w["complete_time"] <= read_op["invoke_time"]
                    ]
                    if later_writes:
                        latest = max(later_writes, key=lambda w: w["complete_time"])
                        if read_op["value"] != latest["value"]:
                            anomalies.append(
                                f"STALE-READ: read({key})→{read_op['value']} "
                                f"but latest completed write was {key}={latest['value']}"
                            )

    return anomalies
```

---

## 13. Summary

### Consistency Model Decision Guide

```
                     Need real-time guarantees?
                    /                            \
                 Yes                              No
                  |                                |
           Linearizable                    Need program-order?
           (Spanner, etcd)                /                  \
                                        Yes                   No
                                         |                     |
                                  Sequential            Need causality?
                                  Consistency           /              \
                                  (ZooKeeper)         Yes              No
                                                       |                |
                                                 Causal              Eventual
                                                 Consistency         Consistency
                                                 (MongoDB)           (DynamoDB)
```

### Key Takeaways

1. **Linearizability** is the gold standard: operations appear atomic and respect real-time ordering. But it requires coordination (quorum reads/writes) and sacrifices availability during partitions.

2. **Sequential consistency** relaxes real-time ordering but preserves program order. Cheaper than linearizability but not composable (non-local).

3. **Causal consistency** respects cause-and-effect but allows concurrent operations to diverge. Achievable with high availability (no quorum needed for reads).

4. **Eventual consistency** guarantees only convergence. Cheapest and most available, but requires application-level conflict handling.

5. **CAP theorem** is really about the partition trade-off: during partitions, choose consistency (CP) or availability (AP). Most real systems choose CP for critical data and AP for non-critical data.

6. **PACELC** extends CAP to include the latency-consistency trade-off during normal operation, which is often more impactful than the partition trade-off.

7. **Test your claims**: Use Jepsen-style testing to verify that your system actually provides the consistency it claims. Many production systems have been found to violate their stated guarantees.

---

## 14. Practice Problems

### Problem 1: Linearizability Check

Determine whether each history is linearizable (initial value of all registers is 0):

```
History A:
  Client 1: w(x,1) [t=0, t=10]
  Client 2: r(x)→1 [t=5, t=15]
  Client 3: w(x,2) [t=12, t=20]
  Client 3: r(x)→2 [t=25, t=30]

History B:
  Client 1: w(x,1) [t=0, t=10]
  Client 2: w(x,2) [t=5, t=15]
  Client 3: r(x)→1 [t=20, t=25]
  Client 4: r(x)→2 [t=22, t=28]

History C:
  Client 1: w(x,1) [t=0, t=10]
  Client 1: w(x,2) [t=15, t=25]
  Client 2: r(x)→2 [t=20, t=30]
  Client 2: r(x)→1 [t=35, t=40]
```

### Problem 2: CAP Classification

For each scenario, explain whether the system is choosing CP or AP:

1. During a network partition, a database returns errors for all write operations but continues serving read operations from the local replica.
2. During a partition, a key-value store continues accepting both reads and writes, but writes may be lost when the partition heals.
3. A system uses quorum reads and writes (W + R > N). During a partition where fewer than a quorum of nodes are reachable, all operations fail.

### Problem 3: PACELC Analysis

Classify each system using PACELC and justify:

1. A cache that serves stale data during partitions and always returns data from the nearest replica.
2. A financial database that blocks during partitions and waits for majority acknowledgment even during normal operation.
3. A social media feed that serves stale data during partitions but synchronously replicates likes counts during normal operation.

### Problem 4: Session Guarantee Violations

A client performs the following operations against a replicated store:

```
write(x, 5) → OK (to Replica A)
read(x) → 3         (from Replica B, stale)
read(x) → 5         (from Replica A)
read(x) → 3         (from Replica B, still stale)
```

Which session guarantees are violated? How would you fix each violation?

### Problem 5: Implementation Challenge

Extend the linearizability checker to:

1. Support compare-and-swap (CAS) operations in addition to reads and writes
2. Handle incomplete operations (operations that were invoked but never received a response -- either the operation took effect or it did not)
3. Generate a visualization of the linearization (a timeline showing where each operation's linearization point falls)

---

## 15. References

1. Herlihy, M. P., & Wing, J. M. (1990). "Linearizability: A Correctness Condition for Concurrent Objects." *ACM Transactions on Programming Languages and Systems*, 12(3), 463-492.
2. Lamport, L. (1979). "How to Make a Multiprocessor Computer That Correctly Executes Multiprocess Programs." *IEEE Transactions on Computers*, C-28(9), 690-691.
3. Gilbert, S., & Lynch, N. (2002). "Brewer's Conjecture and the Feasibility of Consistent, Available, Partition-Tolerant Web Services." *ACM SIGACT News*, 33(2), 51-59.
4. Brewer, E. (2012). "CAP Twelve Years Later: How the 'Rules' Have Changed." *IEEE Computer*, 45(2), 23-29.
5. Abadi, D. (2012). "Consistency Tradeoffs in Modern Distributed Database System Design." *IEEE Computer*, 45(2), 37-42.
6. Ahamad, M., et al. (1995). "Causal Memory: Definitions, Implementation, and Programming." *Distributed Computing*, 9(1), 37-49.
7. Terry, D. B., et al. (1994). "Session Guarantees for Weakly Consistent Replicated Data." *PDIS 1994*.
8. Attiya, H., & Welch, J. (1994). "Sequential Consistency versus Linearizability." *ACM Transactions on Computer Systems*, 12(2), 91-122.
9. Gibbons, P. B., & Korach, E. (1997). "Testing Shared Memories." *SIAM Journal on Computing*, 26(4), 1208-1244.
10. Kingsbury, K. "Jepsen: Analyses of Distributed Systems." https://jepsen.io/analyses.
11. Lloyd, W., et al. (2011). "Don't Settle for Eventual: Scalable Causal Consistency for Wide-Area Storage with COPS." *SOSP 2011*.
12. Viotti, P., & Vukolic, M. (2016). "Consistency in Non-Transactional Distributed Storage Systems." *ACM Computing Surveys*, 49(1), 1-34.

---

[Next: Lesson 05 - Paxos Family](./05_Paxos_Family.md)
