# Lesson 14: Distributed Coordination Primitives

[Overview](./00_Overview.md) | [Previous: Failure Detection and Membership](./13_Failure_Detection_and_Membership.md) | [Next: Formal Verification with TLA+](./15_Formal_Verification_TLAplus.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Design and evaluate distributed locking mechanisms with fencing tokens for correctness
2. Analyze the Chubby, ZooKeeper, and Redlock approaches with their respective trade-offs
3. Implement leader election using multiple strategies (bully, ring, consensus-based)
4. Generate globally unique, time-sortable identifiers using the Snowflake algorithm
5. Compare service discovery architectures and select the appropriate pattern for a given system

---

## Table of Contents

1. [Why Coordination Primitives Matter](#1-why-coordination-primitives-matter)
2. [Distributed Locks](#2-distributed-locks)
3. [Fencing Tokens](#3-fencing-tokens)
4. [Distributed Barriers](#4-distributed-barriers)
5. [Leader Election Patterns](#5-leader-election-patterns)
6. [Sequence Numbers and Ordering](#6-sequence-numbers-and-ordering)
7. [Service Discovery](#7-service-discovery)
8. [Implementation: Distributed Lock with Fencing](#8-implementation-distributed-lock-with-fencing)
9. [Implementation: Snowflake ID Generator](#9-implementation-snowflake-id-generator)
10. [Summary and Further Reading](#10-summary-and-further-reading)

---

## 1. Why Coordination Primitives Matter

Distributed systems consist of independent processes that must cooperate. Without coordination primitives, achieving mutual exclusion, ordered operations, or consistent configuration is either impossible or requires ad-hoc solutions that inevitably fail under edge cases.

### 1.1 The Coordination Landscape

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Coordination Primitives                          │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Mutual       │  │ Ordering     │  │ Discovery    │              │
│  │ Exclusion    │  │              │  │              │              │
│  │              │  │              │  │              │              │
│  │ • Locks      │  │ • Sequence   │  │ • Service    │              │
│  │ • Barriers   │  │   numbers    │  │   registry   │              │
│  │ • Leader     │  │ • Barriers   │  │ • Config     │              │
│  │   election   │  │ • Total      │  │   management │              │
│  │              │  │   ordering   │  │ • DNS        │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                     │
│  Built on: Consensus (Paxos/Raft/ZAB) or Probabilistic Guarantees  │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Requirements for Correct Coordination

Any coordination primitive must address:

| Requirement | Description |
|------------|-------------|
| **Safety** | Bad things never happen (e.g., two clients hold the same lock) |
| **Liveness** | Good things eventually happen (e.g., a lock is eventually granted) |
| **Fault tolerance** | The primitive works despite node/network failures |
| **Performance** | Overhead is acceptable for the use case |

**Observation**: By FLP impossibility (Lesson 3), you cannot have all of these perfectly in an asynchronous system. Every coordination primitive makes trade-offs.

---

## 2. Distributed Locks

### 2.1 Requirements

A distributed lock must satisfy:

1. **Mutual exclusion**: At most one client holds the lock at any time
2. **Deadlock freedom**: If a client crashes while holding a lock, the lock is eventually released
3. **Fault tolerance**: The lock service continues to function despite failures

### 2.2 Chubby (Google)

Chubby is Google's distributed lock service, built on top of Paxos consensus. Published by Mike Burrows in 2006.

**Architecture**:

```
┌──────────────────────────────────────────────┐
│                Chubby Cell                    │
│                                              │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
│  │Server│  │Server│  │Server│  │Server│  │Server│
│  │  1   │  │  2   │  │  3   │  │  4   │  │  5   │
│  │      │  │(MASTER)│ │      │  │      │  │      │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘
│     │         │         │         │         │     │
│     └─────────┴────┬────┴─────────┴─────────┘     │
│                    │ Paxos                         │
│                    │ Consensus                     │
└────────────────────┼──────────────────────────────┘
                     │
            ┌────────┴────────┐
            │  Client Library │
            │  (Caching,      │
            │   KeepAlive)    │
            └─────────────────┘
```

**Key design decisions**:

| Feature | Design Choice | Rationale |
|---------|--------------|-----------|
| Lock granularity | Coarse-grained | Locks held for hours/days (e.g., master election) |
| Lock delay | 1 minute delay after lock release | Allows crashed client's operations to drain |
| Sequencer | Opaque byte-string | Clients pass sequencer to resource servers for fencing |
| Caching | Client library caches file data | Invalidated by Chubby master via KeepAlive |
| KeepAlive | Periodic heartbeat (default 12s) | Maintains session, detects client failures |
| Sessions | Ephemeral leases | Session timeout releases all locks |

**Lock delay mechanism**:

```
Timeline:
  Client A acquires lock ─────────── Client A crashes ──── Lock delay (60s) ──── Client B acquires
       │                                    │                                          │
       └── Operations using lock ───────────┘                                          │
                                                                                       │
  During lock delay:                                                                   │
  • No new client can acquire                                                          │
  • Client A's in-flight operations complete                                           │
  • Resource servers reject stale sequencers                                            │
```

### 2.3 ZooKeeper Recipes

ZooKeeper provides lower-level primitives that can be composed into locks, barriers, and elections.

**Ephemeral Sequential Znodes for Locks**:

```
Lock path: /locks/my-resource

Client A creates: /locks/my-resource/lock-0000000001 (ephemeral, sequential)
Client B creates: /locks/my-resource/lock-0000000002 (ephemeral, sequential)
Client C creates: /locks/my-resource/lock-0000000003 (ephemeral, sequential)

Rule: Client with the lowest sequence number holds the lock.
      Others watch the znode just before them.

Client A holds lock (lowest: 0001)
Client B watches 0001 (will be notified if A's znode deleted)
Client C watches 0002 (will be notified if B's znode deleted)
```

**Why watch the predecessor (not the lock znode)?**

This avoids the **herd effect**: if N clients all watch the lock znode, when the lock is released, all N clients wake up simultaneously, but only one can acquire. With predecessor watches, only one client wakes up — the next in line.

```python
# Pseudocode for ZooKeeper distributed lock
class ZooKeeperLock:
    def __init__(self, zk_client, lock_path):
        self.zk = zk_client
        self.lock_path = lock_path
        self.my_znode = None

    def acquire(self):
        # Create ephemeral sequential znode
        self.my_znode = self.zk.create(
            f"{self.lock_path}/lock-",
            ephemeral=True,
            sequential=True,
        )

        while True:
            # Get all children sorted by sequence number
            children = sorted(self.zk.get_children(self.lock_path))

            my_seq = self.my_znode.split("-")[-1]
            my_index = next(
                i for i, c in enumerate(children)
                if c.split("-")[-1] == my_seq
            )

            if my_index == 0:
                # We have the lowest sequence number: lock acquired!
                return True

            # Watch the predecessor
            predecessor = children[my_index - 1]
            predecessor_path = f"{self.lock_path}/{predecessor}"

            # Block until predecessor is deleted
            event = self.zk.exists(predecessor_path, watch=True)
            if event is not None:
                # Predecessor still exists: wait for watch notification
                self._wait_for_event()

    def release(self):
        if self.my_znode:
            self.zk.delete(self.my_znode)
            self.my_znode = None
```

**Read-Write Locks with ZooKeeper**:

```
Write lock: same as exclusive lock above
Read lock:
  1. Create ephemeral sequential znode: /locks/resource/read-NNNN
  2. Get all children
  3. If no WRITE znode has a lower sequence number → read lock acquired
  4. Otherwise, watch the highest WRITE znode with lower sequence number

This allows:
  - Multiple concurrent readers
  - Writers block until all prior readers complete
  - Readers block until prior writer completes
```

### 2.4 Redlock (Redis)

Redlock is an algorithm proposed by Salvatore Sanfilippo (antirez) for distributed locking using multiple independent Redis instances.

**The Algorithm**:

```
Setup: N Redis masters (typically N=5), no replication

Step 1: Client gets current time T1

Step 2: Client tries to acquire lock on all N instances:
   SET resource_name my_random_value NX PX 30000
   (set if not exists, with 30 second expiry)

Step 3: Client gets current time T2

Step 4: Lock is acquired if and only if:
   a) Lock was acquired on at least N/2 + 1 instances (majority)
   b) Total elapsed time (T2 - T1) < lock validity time
   c) Remaining validity = lock_ttl - (T2 - T1)

Step 5: If lock not acquired, release on all instances

┌──────────┐
│ Client   │
│          │─── SET NX ──▶ Redis 1  ✓
│          │─── SET NX ──▶ Redis 2  ✓
│          │─── SET NX ──▶ Redis 3  ✓  ← majority (3/5)
│          │─── SET NX ──▶ Redis 4  ✗  (already locked)
│          │─── SET NX ──▶ Redis 5  ✓
│          │
│ Lock acquired (4/5 > 3 = N/2+1)
└──────────┘
```

### 2.5 The Redlock Debate

**Martin Kleppmann's Critique** ("How to do distributed locking", 2016):

```
Problem 1: GC Pauses

Client A                    Redis            Resource Server
   │                          │                      │
   │── acquire lock ──────▶  │                      │
   │◀── lock granted ──────  │                      │
   │                          │                      │
   │  ┌── GC PAUSE ──────┐  │                      │
   │  │  (30+ seconds)    │  │── lock expires ────▶│
   │  │                    │  │                      │
   │  │  Client B          │  │◀── acquires lock ──│
   │  │  acquires lock     │  │── writes data ─────▶│
   │  └───────────────────┘  │                      │
   │                          │                      │
   │── writes data ─────────────────────────────────▶│  ← UNSAFE!
   │  (believes it still holds the lock)             │
   │                                                  │

Two clients write concurrently → mutual exclusion violated!
```

```
Problem 2: Clock Skew

Redis 1:  time = 10:00:00 ──── lock expires at 10:00:30
Redis 2:  time = 10:00:00 ──── lock expires at 10:00:30
Redis 3:  time = 10:00:05 ──── lock expires at 10:00:35  ← clock ahead
Redis 4:  time = 09:59:55 ──── lock expires at 10:00:25  ← clock behind
Redis 5:  time = 10:00:00 ──── lock expires at 10:00:30

Locks expire at different real times due to clock skew.
A second client could acquire a majority before the first client's
locks have all expired.
```

**Kleppmann's solution**: Use fencing tokens (see Section 3).

**Antirez's Response**:

| Kleppmann's Point | Antirez's Counter |
|-------------------|-------------------|
| GC pauses break safety | Any lock algorithm (including ZooKeeper) has this problem. Redlock's TTL bounds the issue. |
| Clock skew is unbounded | Bounded clock drift is a reasonable assumption (NTP + CLOCK_MONOTONIC) |
| Fencing tokens solve everything | If you have fencing tokens, you don't even need locks |

**Lessons from the debate**:

1. **Distributed locks alone cannot guarantee safety** — you always need fencing or idempotent operations
2. **Clock assumptions matter** — explicitly state what your system assumes about clocks
3. **The cost of being wrong** — if a lock violation causes data corruption, use a consensus-backed lock (ZooKeeper, etcd), not Redlock
4. **Efficiency vs correctness** — Redlock is fine for efficiency (preventing duplicate work), not for correctness (preventing data corruption)

---

## 3. Fencing Tokens

### 3.1 The Problem: Stale Lock Holders

Even with a perfect lock service, a client that experiences a long pause (GC, network delay, page fault) might continue operating after its lock has expired and been granted to another client.

### 3.2 The Solution: Monotonically Increasing Tokens

```
Lock Service              Client A                Resource Server
     │                       │                           │
     │◀── acquire lock ─────│                           │
     │── token=33 ──────────▶│                           │
     │                       │── write(data, token=33) ─▶│
     │                       │                           │ Accepts: 33 > last_token(0)
     │                       │                           │ last_token = 33
     │                       │                           │
     │   (A pauses, lock expires)                        │
     │                       │                           │
     │◀── acquire lock ──── Client B                     │
     │── token=34 ──────────▶│                           │
     │                       │── write(data, token=34) ─▶│
     │                       │                           │ Accepts: 34 > 33
     │                       │                           │ last_token = 34
     │                       │                           │
     │            (A resumes from pause)                  │
     │                       │                           │
     │                 Client A                          │
     │                       │── write(data, token=33) ─▶│
     │                       │                           │ REJECTS: 33 < 34
     │                       │◀── error: stale token ───│
```

### 3.3 Requirements for Fencing Tokens

1. **Monotonicity**: Each new token must be strictly greater than all previous tokens
2. **Uniqueness**: No two lock grants produce the same token
3. **Durability**: The token sequence survives lock service crashes

**How different systems provide fencing tokens**:

| System | Fencing Token Source | Monotonicity Guarantee |
|--------|---------------------|----------------------|
| ZooKeeper | `czxid` (create transaction ID) | Globally ordered via ZAB |
| etcd | Revision number | Globally ordered via Raft |
| Chubby | Sequencer (opaque string) | Ordered by Paxos log index |
| Redlock | Not built-in | Must add separately (weakness of the approach) |

### 3.4 Implementing Fencing in the Resource Server

The resource server must:

1. Accept a fencing token with every request
2. Reject requests with a token lower than or equal to the highest seen
3. Persist the highest seen token (to survive crashes)

```python
class FencedResourceServer:
    """Resource server that enforces fencing tokens."""

    def __init__(self):
        self.data = {}
        self.highest_token = 0
        self._token_lock = threading.Lock()

    def write(self, key: str, value: str, fencing_token: int) -> bool:
        """
        Write a value, but only if the fencing token is valid.

        Args:
            key: The key to write
            value: The value to write
            fencing_token: Must be > highest previously seen token

        Returns:
            True if write succeeded, False if rejected (stale token)
        """
        with self._token_lock:
            if fencing_token <= self.highest_token:
                return False  # Stale token — reject
            self.highest_token = fencing_token
            self.data[key] = value
            return True

    def read(self, key: str) -> Optional[str]:
        """Read a value (no fencing needed for reads in simple case)."""
        return self.data.get(key)
```

---

## 4. Distributed Barriers

### 4.1 Single Barrier

A barrier blocks all processes until a condition is met (e.g., all processes have arrived).

```
Process 1: ─────────── arrive ──── WAIT ──── proceed ─────▶
Process 2: ── arrive ──────────── WAIT ──── proceed ─────▶
Process 3: ────────────── arrive ─ WAIT ──── proceed ─────▶
                                    │
                            All 3 arrived:
                            barrier opens
```

### 4.2 Double Barrier

A double barrier has two synchronization points:
1. **Entry barrier**: Block until all N processes have entered
2. **Exit barrier**: Block until all N processes have completed

```
                 Entry Barrier              Exit Barrier
Process 1: ──── enter ─── WAIT ── work ── done ─── WAIT ── leave ──▶
Process 2: ── enter ───── WAIT ── work ── done ─── WAIT ── leave ──▶
Process 3: ────── enter ─ WAIT ── work ── done ─── WAIT ── leave ──▶
                            │                        │
                    All entered                All finished
```

### 4.3 ZooKeeper Double Barrier Implementation

```python
class ZooKeeperDoubleBarrier:
    """
    Double barrier using ZooKeeper.

    Pseudocode for the ZooKeeper recipe.
    """

    def __init__(self, zk_client, barrier_path: str, num_processes: int):
        self.zk = zk_client
        self.barrier_path = barrier_path
        self.num_processes = num_processes
        self.my_node = None

    def enter(self, process_id: str) -> None:
        """
        Enter the barrier. Blocks until all processes have entered.
        """
        # Create ephemeral child node
        self.my_node = self.zk.create(
            f"{self.barrier_path}/{process_id}",
            ephemeral=True,
        )

        while True:
            children = self.zk.get_children(self.barrier_path)
            if len(children) >= self.num_processes:
                # Create "ready" node to signal all have arrived
                return
            else:
                # Watch for new children
                self.zk.get_children(self.barrier_path, watch=True)
                self._wait_for_event()

    def leave(self, process_id: str) -> None:
        """
        Leave the barrier. Blocks until all processes have left.
        """
        while True:
            children = sorted(self.zk.get_children(self.barrier_path))

            if len(children) == 0:
                return  # All have left

            if len(children) == 1 and children[0] == process_id:
                # We are the last one: delete and exit
                self.zk.delete(self.my_node)
                return

            if children[-1] == process_id:
                # We have the highest sequence: delete self, watch lowest
                self.zk.delete(self.my_node)
                lowest = f"{self.barrier_path}/{children[0]}"
                self.zk.exists(lowest, watch=True)
                self._wait_for_event()
            else:
                # Watch the highest sequence node
                highest = f"{self.barrier_path}/{children[-1]}"
                self.zk.exists(highest, watch=True)
                self._wait_for_event()

    def _wait_for_event(self):
        """Block until a watch fires (simplified)."""
        pass  # In real code, use threading.Event or asyncio
```

---

## 5. Leader Election Patterns

### 5.1 Why Leader Election

Many distributed algorithms benefit from having a single leader:
- **Consensus protocols**: Leader drives proposal (Raft, Multi-Paxos)
- **Database replication**: Primary accepts writes
- **Coordination**: One process performs scheduled work
- **Load balancing**: One process distributes work

### 5.2 Bully Algorithm (Garcia-Molina, 1982)

```
Assumption: Each process has a unique numeric ID. Higher ID = higher priority.

Algorithm (process P detects coordinator failure):

1. P sends ELECTION to all processes with higher IDs
2. If no response within timeout → P declares itself coordinator
3. If P receives OK from any higher-ID process → P waits
4. The highest-ID process that responds eventually sends COORDINATOR to all

Example with processes {1, 2, 3, 4, 5}, process 5 is current coordinator:

Process 5 crashes.
Process 2 detects crash.

  2 ──ELECTION──▶ 3    ✓ OK
  2 ──ELECTION──▶ 4    ✓ OK
  2 ──ELECTION──▶ 5    ✗ no response

  3 ──ELECTION──▶ 4    ✓ OK
  3 ──ELECTION──▶ 5    ✗ no response

  4 ──ELECTION──▶ 5    ✗ no response

  4 sends COORDINATOR to {1, 2, 3}
  Process 4 is the new coordinator.
```

**Properties**:

| Property | Value |
|----------|-------|
| Message complexity | O(n²) worst case |
| Time complexity | O(n) timeouts worst case |
| Fault tolerance | Handles crash failures |
| Assumption | Process IDs are totally ordered |
| Weakness | Unstable if highest-ID process keeps crashing and recovering |

### 5.3 Ring-Based Election (Chang-Roberts, 1979)

```
Processes arranged in a logical ring.

Process P starts election:
1. P sends ELECTION(P.id) to its successor
2. Each process forwards ELECTION(id) if id > own_id
3. If id < own_id, replace with ELECTION(own_id)
4. If id == own_id, this process is the leader: send COORDINATOR

Example: ring [3] → [1] → [4] → [2] → [3]

  3 starts election:
  3 ──ELECT(3)──▶ 1 ──ELECT(3)──▶ 4    (3 > 1, so forward 3)
                                   4 ──ELECT(4)──▶ 2   (4 > 3, so replace with 4)
                                                    2 ──ELECT(4)──▶ 3  (4 > 2)
                                                                     3 ──ELECT(4)──▶ 1 (4 > 3)
                                                                                      1 ──ELECT(4)──▶ 4
  4 receives ELECT(4): ID matches → 4 is leader
  4 sends COORDINATOR(4) around ring
```

**Properties**:

| Property | Value |
|----------|-------|
| Message complexity | O(n) best case, O(n²) worst case |
| Assumption | Logical ring topology |
| Fault tolerance | Ring must be maintained; not robust to failures during election |

### 5.4 Consensus-Based Election

Modern systems use consensus protocols for leader election because they provide strong guarantees:

**Raft leader election** (from Lesson 6):

```
Terms:  ─── term 1 (leader: A) ───│─── term 2 (election) ───│─── term 3 (leader: C) ──

Node A: Leader ─── heartbeats ───── CRASH
Node B: Follower ─────────────────── timeout ── Candidate(term 2) ── loses ── Follower
Node C: Follower ─────────────────── timeout ── Candidate(term 2) ── WINS ─── Leader
Node D: Follower ─────────────────── votes for C ────────────────────────── Follower
Node E: Follower ─────────────────── votes for C ────────────────────────── Follower
```

**ZooKeeper election using ephemeral sequential znodes**:

```python
class ZooKeeperLeaderElection:
    """Leader election using ZooKeeper ephemeral sequential znodes."""

    def __init__(self, zk_client, election_path: str, on_elected, on_revoked):
        self.zk = zk_client
        self.election_path = election_path
        self.on_elected = on_elected      # Callback when elected
        self.on_revoked = on_revoked      # Callback when leadership lost
        self.my_znode = None
        self.is_leader = False

    def run_election(self) -> None:
        """Participate in leader election."""
        # Create ephemeral sequential znode
        self.my_znode = self.zk.create(
            f"{self.election_path}/candidate-",
            ephemeral=True,
            sequential=True,
        )
        self._check_leadership()

    def _check_leadership(self) -> None:
        """Check if this node is the leader."""
        children = sorted(self.zk.get_children(self.election_path))
        my_seq = self.my_znode.split("-")[-1]

        if children[0].split("-")[-1] == my_seq:
            # We have the lowest sequence number: we are leader
            if not self.is_leader:
                self.is_leader = True
                self.on_elected()
        else:
            # Watch predecessor (avoid herd effect)
            my_index = next(
                i for i, c in enumerate(children)
                if c.split("-")[-1] == my_seq
            )
            predecessor = children[my_index - 1]
            predecessor_path = f"{self.election_path}/{predecessor}"

            # Watch predecessor: when it goes away, re-check
            exists = self.zk.exists(predecessor_path, watch=True)
            if exists is None:
                # Predecessor already gone: re-check immediately
                self._check_leadership()
            # Otherwise, wait for watch notification, then re-check
```

### 5.5 Comparison of Election Algorithms

| Algorithm | Messages | Fault Model | Guarantees | Used In |
|-----------|----------|-------------|------------|---------|
| Bully | O(n²) | Crash-stop | Unique leader | Academic |
| Ring | O(n)-O(n²) | No failures during election | Unique leader | Token rings |
| Raft | O(n) | Crash-recovery | At most one leader per term | etcd, CockroachDB |
| ZAB | O(n) | Crash-recovery | Unique leader | ZooKeeper |
| ZK ephemeral | O(n) | Crash-stop | Unique leader | Application-level |

---

## 6. Sequence Numbers and Ordering

### 6.1 The Need for Global Ordering

Many systems need globally unique, roughly time-ordered identifiers:
- Database primary keys
- Event ordering in distributed logs
- Distributed tracing (trace IDs, span IDs)
- Transaction IDs

### 6.2 Approaches to ID Generation

| Approach | Ordering | Uniqueness | Coordination | Throughput |
|----------|----------|------------|--------------|-----------|
| Auto-increment (single DB) | Total | Guaranteed | Single point | Low (~10K/s) |
| UUID v4 | None | Probabilistic | None | Very high |
| UUID v7 | Rough time | Probabilistic | None | Very high |
| Snowflake | Time + partial | Guaranteed | Machine ID assignment | High (~4M/s/node) |
| ULID | Time + random | Probabilistic | None | Very high |
| Timestamp + counter | Time | Requires coordination | Per-node counter | High |

### 6.3 Twitter Snowflake IDs

Snowflake was designed at Twitter to generate roughly time-ordered, unique IDs at scale without coordination between generators.

**Bit layout** (64 bits total):

```
┌─────────────────────────────────────────────────────────────────┐
│ 0 │      41 bits: timestamp        │ 5 │ 5 │  12 bits:        │
│   │      (milliseconds since       │ DC│ W │  sequence         │
│   │       custom epoch)            │ ID│ ID│  number           │
└─────────────────────────────────────────────────────────────────┘

Bit 63:    Unused (sign bit, always 0)
Bits 62-22: Timestamp (41 bits → 2^41 ms ≈ 69.7 years)
Bits 21-17: Datacenter ID (5 bits → 32 datacenters)
Bits 16-12: Worker ID (5 bits → 32 workers per DC)
Bits 11-0:  Sequence number (12 bits → 4096 IDs per ms per worker)
```

**Properties**:
- **Time-sortable**: IDs generated later have higher values (within clock skew tolerance)
- **No coordination**: Each worker generates independently
- **High throughput**: 4096 IDs/ms/worker = ~4M IDs/s/worker
- **K-sortable**: IDs within a few milliseconds of each other are in time order

**Clock skew handling**: If the system clock moves backward, Snowflake should refuse to generate IDs until the clock catches up (to prevent duplicate IDs).

### 6.4 ULID (Universally Unique Lexicographically Sortable Identifier)

```
 01AN4Z07BY      79KA1307SR9X4MV3

|----------|    |----------------|
 Timestamp       Randomness
  48 bits         80 bits
  (ms)

Total: 128 bits, encoded as 26-character Crockford Base32

Properties:
  - Lexicographically sortable (string comparison works)
  - 1.21e+24 unique ULIDs per millisecond
  - No coordination needed
  - Compatible with UUID storage (128 bits)
  - Monotonic option: within same ms, increment LSB of random part
```

### 6.5 UUID v7 (RFC 9562, 2024)

The newest standard UUID format, designed for database-friendly time-ordering:

```
┌──────────────────────────────────────────────────┐
│ 48 bits: Unix timestamp (ms) │ 4 bits: version  │
│ 12 bits: rand_a              │ 2 bits: variant   │
│ 62 bits: rand_b                                  │
└──────────────────────────────────────────────────┘

Total: 128 bits
Monotonic within millisecond: optional (implementation-defined)
Database friendly: indexes well due to time-ordering
```

---

## 7. Service Discovery

### 7.1 The Service Discovery Problem

In a dynamic distributed system, services start, stop, move, and scale. How does a client find the current address of a service it needs to talk to?

```
Without service discovery:           With service discovery:

Client → hardcoded IP:port          Client → registry → dynamic IP:port
(breaks when service moves)          (resilient to changes)
```

### 7.2 DNS-Based Discovery

```
┌────────┐   DNS query: api.service.consul
│ Client │ ──────────────────────────────────▶ ┌────────────┐
│        │ ◀────────────────────────────────── │ DNS Server │
│        │   A records: 10.0.1.5, 10.0.1.6    │ (Consul)   │
└────────┘                                     └────────────┘

Advantages:
  - Universal: every language/platform supports DNS
  - No client library needed
  - Caching built into DNS protocol

Disadvantages:
  - TTL-based caching: stale results during transitions
  - No health checking at query time (relies on DNS updater)
  - Limited metadata (just IP + port)
  - DNS caching in OS/language runtimes hard to control
```

**Systems**: Consul DNS interface, AWS Route 53 (with health checks), Kubernetes CoreDNS

### 7.3 KV-Based Discovery

```
┌────────┐   GET /v1/kv/services/api
│ Client │ ──────────────────────────────────▶ ┌──────────┐
│        │ ◀────────────────────────────────── │ etcd /   │
│        │   {"host": "10.0.1.5", "port": 8080,│ Consul / │
│        │    "version": "2.1", "weight": 100}  │ ZooKeeper│
└────────┘                                     └──────────┘

Advantages:
  - Rich metadata (version, weight, tags, health status)
  - Watch/subscribe for real-time updates
  - Strong consistency (consensus-backed)
  - Health checking with TTL-based sessions

Disadvantages:
  - Requires client library
  - Additional infrastructure to operate
  - More complex than DNS
```

### 7.4 Client-Side vs Server-Side Discovery

```
Client-Side Discovery:
┌────────┐     query     ┌──────────┐
│ Client │──────────────▶│ Registry │
│        │◀──────────────│          │
│        │  service list  └──────────┘
│        │
│        │──────────────▶ Service Instance A
│        │  (direct call,  (10.0.1.5:8080)
│        │   client picks)
└────────┘

  + No extra hop for requests
  + Client controls load balancing strategy
  - Client must implement discovery logic
  - Different languages need different implementations
  Examples: Netflix Eureka (client-side), gRPC name resolution


Server-Side Discovery:
┌────────┐              ┌───────────┐              ┌──────────┐
│ Client │─── request ─▶│ Load      │─── forward ─▶│ Service  │
│        │◀── response ──│ Balancer  │◀── response ──│ Instance │
└────────┘              │ / Gateway │              └──────────┘
                        └─────┬─────┘
                              │ query
                        ┌─────▼─────┐
                        │ Registry  │
                        └───────────┘

  + Client is simple (just call the load balancer)
  + Language-agnostic
  - Extra network hop
  - Load balancer can become bottleneck
  Examples: AWS ALB, Kubernetes kube-proxy, Nginx
```

### 7.5 Service Mesh

A service mesh moves discovery and communication logic into a sidecar proxy:

```
┌────────────────────────────────┐   ┌────────────────────────────────┐
│ Pod A                          │   │ Pod B                          │
│ ┌──────────┐  ┌──────────────┐│   │┌──────────────┐  ┌──────────┐│
│ │ Service  │──│ Envoy Proxy  ││───▶││ Envoy Proxy  │──│ Service  ││
│ │ (app)    │  │ (sidecar)    ││   │ │ (sidecar)    │  │ (app)    ││
│ └──────────┘  └──────────────┘│   │└──────────────┘  └──────────┘│
└────────────────────────────────┘   └────────────────────────────────┘
                     │                          │
                     └──────────┬───────────────┘
                         ┌──────▼──────┐
                         │ Control     │
                         │ Plane       │
                         │ (Istio/     │
                         │  Linkerd)   │
                         └─────────────┘

The sidecar handles:
  - Service discovery
  - Load balancing
  - TLS termination
  - Retries, circuit breaking
  - Observability (metrics, tracing)
```

### 7.6 Comparison of Service Discovery Approaches

| Aspect | DNS | KV Store | Client-Side | Server-Side | Service Mesh |
|--------|-----|----------|-------------|-------------|-------------|
| Complexity | Low | Medium | Medium | Low (client) | High |
| Latency | Cache-dependent | Low | Lowest | +1 hop | +1 hop |
| Metadata | Limited | Rich | Rich | Limited | Rich |
| Health checking | External | TTL sessions | Client-managed | LB health checks | Sidecar health |
| Language support | Universal | Client lib | Per-language | Universal | Universal |
| Consistency | Eventual | Strong | Eventual | Depends on LB | Eventually |
| Examples | Consul DNS, Route 53 | etcd, Consul KV | Eureka, gRPC | AWS ALB, k8s | Istio, Linkerd |

---

## 8. Implementation: Distributed Lock with Fencing

```python
import time
import threading
import hashlib
import os
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field

@dataclass
class LockInfo:
    """Information about a held lock."""
    owner: str
    fencing_token: int
    acquired_at: float
    ttl: float  # Time-to-live in seconds

    @property
    def is_expired(self) -> bool:
        return time.monotonic() - self.acquired_at > self.ttl


class DistributedLockManager:
    """
    A distributed lock manager with fencing token support.

    This implementation simulates a consensus-backed lock service
    (similar to ZooKeeper or etcd). In production, the lock state
    would be replicated via Raft or Paxos.

    Features:
      - Mutual exclusion with TTL-based expiration
      - Monotonically increasing fencing tokens
      - Lock delay (Chubby-style) to drain in-flight operations
      - Deadlock freedom via TTL expiration
    """

    def __init__(self, lock_delay: float = 5.0):
        self._locks: Dict[str, LockInfo] = {}
        self._next_token: int = 1
        self._lock_delay_until: Dict[str, float] = {}
        self._lock_delay_duration = lock_delay
        self._mutex = threading.Lock()

    def acquire(
        self,
        resource: str,
        owner: str,
        ttl: float = 30.0,
        timeout: float = 10.0,
    ) -> Optional[int]:
        """
        Attempt to acquire a lock on a resource.

        Args:
            resource: The resource to lock
            owner: Unique identifier for the lock owner
            ttl: Lock time-to-live in seconds
            timeout: Maximum time to wait for lock acquisition

        Returns:
            Fencing token if acquired, None if timeout
        """
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            with self._mutex:
                # Check lock delay
                if resource in self._lock_delay_until:
                    if time.monotonic() < self._lock_delay_until[resource]:
                        # Lock delay active: cannot acquire yet
                        time.sleep(0.1)
                        continue
                    else:
                        del self._lock_delay_until[resource]

                # Check if lock is free or expired
                current = self._locks.get(resource)
                if current is None or current.is_expired:
                    # Clean up expired lock (with lock delay)
                    if current is not None and current.is_expired:
                        self._lock_delay_until[resource] = (
                            time.monotonic() + self._lock_delay_duration
                        )
                        del self._locks[resource]
                        continue  # Must wait for lock delay

                    # Grant the lock
                    token = self._next_token
                    self._next_token += 1
                    self._locks[resource] = LockInfo(
                        owner=owner,
                        fencing_token=token,
                        acquired_at=time.monotonic(),
                        ttl=ttl,
                    )
                    return token

                # Lock held by someone else
                if current.owner == owner:
                    # Re-entrant: return existing token
                    return current.fencing_token

            time.sleep(0.1)  # Backoff before retry

        return None  # Timeout

    def release(self, resource: str, owner: str) -> bool:
        """
        Release a lock.

        Args:
            resource: The resource to unlock
            owner: Must match the lock owner

        Returns:
            True if released, False if not the owner
        """
        with self._mutex:
            current = self._locks.get(resource)
            if current is None:
                return False
            if current.owner != owner:
                return False

            # Activate lock delay
            self._lock_delay_until[resource] = (
                time.monotonic() + self._lock_delay_duration
            )
            del self._locks[resource]
            return True

    def get_lock_info(self, resource: str) -> Optional[LockInfo]:
        """Get information about a lock."""
        with self._mutex:
            lock = self._locks.get(resource)
            if lock is not None and lock.is_expired:
                return None
            return lock


class FencedKeyValueStore:
    """
    A key-value store that enforces fencing tokens.

    Every write operation must include a fencing token.
    Writes with tokens lower than or equal to the highest
    seen token for that key are rejected.
    """

    def __init__(self):
        self._data: Dict[str, str] = {}
        self._highest_token: Dict[str, int] = {}
        self._lock = threading.Lock()

    def write(self, key: str, value: str, fencing_token: int) -> Tuple[bool, str]:
        """
        Write a value with fencing token enforcement.

        Returns:
            (success, message) tuple
        """
        with self._lock:
            highest = self._highest_token.get(key, 0)
            if fencing_token <= highest:
                return (
                    False,
                    f"Stale token: {fencing_token} <= {highest}. "
                    f"Write rejected."
                )
            self._highest_token[key] = fencing_token
            self._data[key] = value
            return (True, f"Write accepted with token {fencing_token}")

    def read(self, key: str) -> Optional[str]:
        """Read a value (no fencing needed)."""
        with self._lock:
            return self._data.get(key)


def demo_fenced_locking():
    """Demonstrate distributed locking with fencing tokens."""
    lock_mgr = DistributedLockManager(lock_delay=0.5)
    kv_store = FencedKeyValueStore()

    print("=== Distributed Lock with Fencing Tokens Demo ===\n")

    # Client A acquires lock
    token_a = lock_mgr.acquire("my-resource", "client-A", ttl=2.0)
    print(f"Client A acquired lock with token: {token_a}")

    # Client A writes with its token
    ok, msg = kv_store.write("config", "value-from-A", token_a)
    print(f"Client A writes: {msg}")

    # Client B tries to acquire (should wait/fail)
    token_b = lock_mgr.acquire("my-resource", "client-B", ttl=2.0, timeout=0.5)
    print(f"Client B acquire attempt: token={token_b}")  # None (timeout)

    # Client A releases lock
    lock_mgr.release("my-resource", "client-A")
    print("Client A released lock")

    # Wait for lock delay to expire
    time.sleep(0.6)

    # Client B acquires lock (gets higher token)
    token_b = lock_mgr.acquire("my-resource", "client-B", ttl=2.0, timeout=5.0)
    print(f"\nClient B acquired lock with token: {token_b}")

    # Client B writes with its token
    ok, msg = kv_store.write("config", "value-from-B", token_b)
    print(f"Client B writes: {msg}")

    # Simulate: Client A (stale) tries to write with old token
    ok, msg = kv_store.write("config", "stale-value-from-A", token_a)
    print(f"\nClient A (stale) tries to write: {msg}")  # REJECTED

    # Client B writes again (still valid)
    ok, msg = kv_store.write("config", "value-from-B-v2", token_b)
    print(f"Client B writes again: {msg}")

    print(f"\nFinal value of 'config': {kv_store.read('config')}")


if __name__ == "__main__":
    demo_fenced_locking()
```

---

## 9. Implementation: Snowflake ID Generator

```python
import time
import threading
from typing import Optional

class SnowflakeIDGenerator:
    """
    Twitter Snowflake ID generator.

    Generates 64-bit, roughly time-ordered, unique IDs
    without coordination between generators.

    Bit layout:
      - 1 bit:  unused (sign bit)
      - 41 bits: timestamp (ms since custom epoch)
      - 5 bits:  datacenter ID (0-31)
      - 5 bits:  worker ID (0-31)
      - 12 bits: sequence number (0-4095)

    Supports up to 4096 IDs per millisecond per worker.
    Timestamp space: ~69.7 years from epoch.
    """

    # Bit allocation
    TIMESTAMP_BITS = 41
    DATACENTER_BITS = 5
    WORKER_BITS = 5
    SEQUENCE_BITS = 12

    # Maximum values
    MAX_DATACENTER_ID = (1 << DATACENTER_BITS) - 1   # 31
    MAX_WORKER_ID = (1 << WORKER_BITS) - 1            # 31
    MAX_SEQUENCE = (1 << SEQUENCE_BITS) - 1            # 4095

    # Bit shifts
    WORKER_SHIFT = SEQUENCE_BITS                       # 12
    DATACENTER_SHIFT = SEQUENCE_BITS + WORKER_BITS     # 17
    TIMESTAMP_SHIFT = (
        SEQUENCE_BITS + WORKER_BITS + DATACENTER_BITS  # 22
    )

    # Custom epoch: 2020-01-01 00:00:00 UTC (in ms)
    CUSTOM_EPOCH = 1577836800000

    def __init__(self, datacenter_id: int, worker_id: int):
        """
        Initialize the Snowflake generator.

        Args:
            datacenter_id: Datacenter identifier (0-31)
            worker_id: Worker identifier within datacenter (0-31)

        Raises:
            ValueError: If IDs are out of range
        """
        if not 0 <= datacenter_id <= self.MAX_DATACENTER_ID:
            raise ValueError(
                f"datacenter_id must be 0-{self.MAX_DATACENTER_ID}, "
                f"got {datacenter_id}"
            )
        if not 0 <= worker_id <= self.MAX_WORKER_ID:
            raise ValueError(
                f"worker_id must be 0-{self.MAX_WORKER_ID}, "
                f"got {worker_id}"
            )

        self.datacenter_id = datacenter_id
        self.worker_id = worker_id
        self._sequence = 0
        self._last_timestamp = -1
        self._lock = threading.Lock()

    def _current_time_ms(self) -> int:
        """Get current time in milliseconds."""
        return int(time.time() * 1000)

    def _wait_next_ms(self, last_ts: int) -> int:
        """Block until the clock advances to the next millisecond."""
        ts = self._current_time_ms()
        while ts <= last_ts:
            ts = self._current_time_ms()
        return ts

    def generate(self) -> int:
        """
        Generate a unique Snowflake ID.

        Returns:
            64-bit integer ID

        Raises:
            RuntimeError: If clock moves backward
        """
        with self._lock:
            timestamp = self._current_time_ms()

            if timestamp < self._last_timestamp:
                # Clock moved backward: refuse to generate
                delta = self._last_timestamp - timestamp
                raise RuntimeError(
                    f"Clock moved backward by {delta}ms. "
                    f"Refusing to generate ID to prevent duplicates."
                )

            if timestamp == self._last_timestamp:
                # Same millisecond: increment sequence
                self._sequence = (self._sequence + 1) & self.MAX_SEQUENCE
                if self._sequence == 0:
                    # Sequence exhausted: wait for next millisecond
                    timestamp = self._wait_next_ms(self._last_timestamp)
            else:
                # New millisecond: reset sequence
                self._sequence = 0

            self._last_timestamp = timestamp

            # Compose the ID
            snowflake_id = (
                ((timestamp - self.CUSTOM_EPOCH) << self.TIMESTAMP_SHIFT)
                | (self.datacenter_id << self.DATACENTER_SHIFT)
                | (self.worker_id << self.WORKER_SHIFT)
                | self._sequence
            )

            return snowflake_id

    @classmethod
    def parse(cls, snowflake_id: int) -> dict:
        """
        Decompose a Snowflake ID into its components.

        Args:
            snowflake_id: The ID to parse

        Returns:
            Dict with timestamp_ms, datacenter_id, worker_id, sequence
        """
        sequence = snowflake_id & cls.MAX_SEQUENCE
        worker_id = (snowflake_id >> cls.WORKER_SHIFT) & cls.MAX_WORKER_ID
        datacenter_id = (
            (snowflake_id >> cls.DATACENTER_SHIFT) & cls.MAX_DATACENTER_ID
        )
        timestamp_ms = (
            (snowflake_id >> cls.TIMESTAMP_SHIFT) + cls.CUSTOM_EPOCH
        )

        return {
            "id": snowflake_id,
            "timestamp_ms": timestamp_ms,
            "timestamp_iso": time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.gmtime(timestamp_ms / 1000),
            ),
            "datacenter_id": datacenter_id,
            "worker_id": worker_id,
            "sequence": sequence,
            "binary": format(snowflake_id, "064b"),
        }

    @classmethod
    def print_bit_layout(cls, snowflake_id: int) -> None:
        """Print a visual breakdown of the ID's bit layout."""
        binary = format(snowflake_id, "064b")
        print(f"ID: {snowflake_id}")
        print(f"Binary: {binary}")
        print(f"  [0]  Unused:    {binary[0]}")
        print(f"  [1-41]  Timestamp:  {binary[1:42]} "
              f"({int(binary[1:42], 2)})")
        print(f"  [42-46] Datacenter: {binary[42:47]} "
              f"({int(binary[42:47], 2)})")
        print(f"  [47-51] Worker:     {binary[47:52]} "
              f"({int(binary[47:52], 2)})")
        print(f"  [52-63] Sequence:   {binary[52:64]} "
              f"({int(binary[52:64], 2)})")


def demo_snowflake():
    """Demonstrate Snowflake ID generation."""
    print("=== Snowflake ID Generator Demo ===\n")

    # Create generators for different workers
    gen_dc0_w0 = SnowflakeIDGenerator(datacenter_id=0, worker_id=0)
    gen_dc0_w1 = SnowflakeIDGenerator(datacenter_id=0, worker_id=1)
    gen_dc1_w0 = SnowflakeIDGenerator(datacenter_id=1, worker_id=0)

    # Generate IDs from different workers
    print("IDs from datacenter 0, worker 0:")
    for i in range(5):
        sid = gen_dc0_w0.generate()
        parsed = SnowflakeIDGenerator.parse(sid)
        print(f"  {sid:>20d}  seq={parsed['sequence']}  "
              f"time={parsed['timestamp_iso']}")

    print("\nIDs from datacenter 0, worker 1:")
    for i in range(3):
        sid = gen_dc0_w1.generate()
        parsed = SnowflakeIDGenerator.parse(sid)
        print(f"  {sid:>20d}  seq={parsed['sequence']}  "
              f"dc={parsed['datacenter_id']} w={parsed['worker_id']}")

    print("\nIDs from datacenter 1, worker 0:")
    for i in range(3):
        sid = gen_dc1_w0.generate()
        parsed = SnowflakeIDGenerator.parse(sid)
        print(f"  {sid:>20d}  seq={parsed['sequence']}  "
              f"dc={parsed['datacenter_id']} w={parsed['worker_id']}")

    # Show bit layout
    print("\n=== Bit Layout ===")
    sid = gen_dc0_w0.generate()
    SnowflakeIDGenerator.print_bit_layout(sid)

    # Demonstrate ordering
    print("\n=== Ordering Guarantee ===")
    ids = [gen_dc0_w0.generate() for _ in range(10)]
    print(f"All IDs strictly increasing: {all(ids[i] < ids[i+1] for i in range(len(ids)-1))}")
    print(f"First: {ids[0]}")
    print(f"Last:  {ids[-1]}")

    # Multi-threaded generation
    print("\n=== Multi-threaded Generation ===")
    all_ids = []
    lock = threading.Lock()

    def generate_ids(generator, count):
        local_ids = [generator.generate() for _ in range(count)]
        with lock:
            all_ids.extend(local_ids)

    threads = [
        threading.Thread(target=generate_ids, args=(gen_dc0_w0, 1000)),
        threading.Thread(target=generate_ids, args=(gen_dc0_w1, 1000)),
        threading.Thread(target=generate_ids, args=(gen_dc1_w0, 1000)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    unique_ids = set(all_ids)
    print(f"Generated {len(all_ids)} IDs across 3 workers")
    print(f"Unique IDs: {len(unique_ids)}")
    print(f"Duplicates: {len(all_ids) - len(unique_ids)}")


if __name__ == "__main__":
    demo_snowflake()
```

---

## 10. Summary and Further Reading

### Key Takeaways

| Primitive | Key Insight | Production Systems |
|-----------|-------------|-------------------|
| Distributed locks | Always use fencing tokens for correctness | ZooKeeper, etcd, Chubby |
| Redlock | OK for efficiency, not for correctness guarantees | Redis |
| Fencing tokens | Monotonic tokens let resource servers reject stale operations | All lock services |
| Barriers | Double barriers synchronize batch-parallel computation | ZooKeeper, MapReduce |
| Leader election | Consensus-based is the only production-grade approach | Raft, ZAB, Paxos |
| Snowflake IDs | Time + machine + sequence = unique without coordination | Twitter, Discord |
| Service discovery | Match approach to system requirements and team capabilities | Consul, etcd, k8s |

### Essential Reading

1. **Burrows (2006)** — "The Chubby Lock Service for Loosely-Coupled Distributed Systems"
2. **Hunt et al. (2010)** — "ZooKeeper: Wait-free Coordination for Internet-scale Systems"
3. **Kleppmann (2016)** — "How to do distributed locking" (blog post on Redlock critique)
4. **Sanfilippo (2016)** — "Is Redlock safe?" (response to Kleppmann)
5. **Garcia-Molina (1982)** — "Elections in a Distributed Computing System" (Bully algorithm)

### Connection to Other Lessons

- **Lesson 5 (Paxos)** and **Lesson 6 (Raft)**: Consensus underpins coordination primitives
- **Lesson 8 (Distributed Transactions)**: Locks are used within 2PC coordinators
- **Lesson 13 (Failure Detection)**: Failure detectors trigger leader re-election
- **Lesson 16 (Capstone)**: Uses leader election and fencing in the KV store

---

[Next: Formal Verification with TLA+](./15_Formal_Verification_TLAplus.md)
