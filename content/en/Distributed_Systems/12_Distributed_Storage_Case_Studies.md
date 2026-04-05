# Lesson 12: Distributed Storage Case Studies

[Overview](./00_Overview.md) | [Previous: Partitioning and Sharding](./11_Partitioning_and_Sharding.md) | [Next: Failure Detection and Membership](./13_Failure_Detection_and_Membership.md)

---

## Learning Objectives

- Analyze Google Spanner's architecture, TrueTime API, and how it achieves external consistency
- Understand Amazon Dynamo's design philosophy and its always-writable approach at the cost of consistency
- Evaluate Apache Kafka's distributed commit log architecture and exactly-once semantics
- Examine CockroachDB's serializable distributed SQL implementation using Raft and hybrid logical clocks
- Compare and contrast these systems across consistency, partitioning, replication, and transaction models

---

## 1. Google Spanner

Spanner is Google's globally distributed, strongly consistent database. It is the first system to provide **external consistency** (stronger than linearizability) at global scale, using a combination of Paxos replication and a novel time API called **TrueTime**.

### 1.1 Architecture Overview

Spanner's architecture has multiple layers of abstraction:

```
Universe (one Spanner deployment, e.g., "production")
  │
  ├── Zone 1 (≈ datacenter)
  │   ├── Zone Master (assigns data to spanservers)
  │   ├── Location Proxy (clients use to find data)
  │   ├── Spanserver 1
  │   │   ├── Tablet 1 (key range: [A..F))
  │   │   │   ├── Colossus (distributed file system) for storage
  │   │   │   └── Paxos state machine (replicates this tablet)
  │   │   ├── Tablet 2 (key range: [F..K))
  │   │   └── ...
  │   └── Spanserver 2
  │       └── ...
  │
  ├── Zone 2 (another datacenter, possibly another continent)
  │   └── (same structure)
  │
  └── Zone 3
      └── (same structure)

Each tablet is replicated via Paxos across zones.
One Paxos replica is the leader; it handles reads and writes.
```

| Component | Role |
|---|---|
| **Universe** | A single Spanner deployment (e.g., one for production, one for test) |
| **Zone** | Unit of physical isolation (≈ datacenter). Minimum 3 zones for Paxos. |
| **Spanserver** | Serves data. Each manages 100-1000 tablets. |
| **Tablet** | A contiguous range of rows, stored in Colossus. Basic unit of replication. |
| **Directory** | Logical grouping of rows with a common key prefix. Unit of data placement and movement. |

### 1.2 TrueTime

Spanner's most revolutionary contribution is the **TrueTime API**, which exposes clock uncertainty explicitly.

Most systems assume clocks are accurate (and are wrong). Spanner accepts that clocks have bounded uncertainty and **waits out the uncertainty** before committing.

```
Standard clock API:
  now() → timestamp        (no uncertainty information)

TrueTime API:
  TT.now() → TTinterval    (earliest, latest)
  TT.after(t) → bool       ("is it definitely after time t?")
  TT.before(t) → bool      ("is it definitely before time t?")

TTinterval example:
  TT.now() = [t - ε, t + ε]
  where ε is typically 1-7 ms (GPS + atomic clock synchronization)
```

```python
import time
from dataclasses import dataclass


@dataclass
class TTInterval:
    """TrueTime interval: [earliest, latest]."""
    earliest: float
    latest: float

    @property
    def uncertainty(self) -> float:
        return self.latest - self.earliest

    def definitely_after(self, t: float) -> bool:
        """Is the current time definitely after t?"""
        return self.earliest > t

    def definitely_before(self, t: float) -> bool:
        """Is the current time definitely before t?"""
        return self.latest < t


class TrueTime:
    """
    Simulated TrueTime API.
    In production, Google uses GPS receivers + atomic clocks
    in each datacenter to bound uncertainty to ~1-7ms.
    """

    def __init__(self, epsilon_ms: float = 4.0):
        self.epsilon_ms = epsilon_ms  # half-width of uncertainty

    def now(self) -> TTInterval:
        """Return a time interval guaranteed to contain the true time."""
        t = time.time()
        epsilon = self.epsilon_ms / 1000.0
        return TTInterval(earliest=t - epsilon, latest=t + epsilon)

    def after(self, t: float) -> bool:
        """Is the true time definitely after t?"""
        return self.now().earliest > t

    def before(self, t: float) -> bool:
        """Is the true time definitely before t?"""
        return self.now().latest < t

    def wait_until_after(self, t: float) -> None:
        """
        Block until the true time is definitely after t.
        This is the 'commit wait' that ensures external consistency.
        """
        while not self.after(t):
            time.sleep(0.001)  # poll every 1ms
```

### 1.3 External Consistency via Commit Wait

External consistency means: if transaction T1 commits before transaction T2 starts (in real wall-clock time), then T1's commit timestamp is less than T2's commit timestamp.

**Commit protocol**:

```
1. Leader picks commit timestamp s = TT.now().latest
   (guaranteed to be ≥ true time)

2. Leader WAITS until TT.after(s) is true
   (commit wait: ensures no future transaction can pick a timestamp ≤ s)

3. Leader applies the write and responds to client

Wait time ≈ 2ε (twice the clock uncertainty)
With ε ≈ 4ms, commit wait ≈ 8ms
```

```python
class SpannerTransaction:
    """
    Simplified Spanner read-write transaction with commit wait.
    Demonstrates how TrueTime ensures external consistency.
    """

    def __init__(self, tt: TrueTime, transaction_id: str):
        self.tt = tt
        self.transaction_id = transaction_id
        self.read_set: dict[str, tuple] = {}    # key -> (value, timestamp)
        self.write_set: dict[str, any] = {}      # key -> value
        self.commit_timestamp: float | None = None
        self.state = "active"

    def read(self, store: dict, key: str) -> any:
        """Read a key (acquires read lock in real Spanner)."""
        if key in self.write_set:
            return self.write_set[key]
        entry = store.get(key)
        if entry:
            value, ts = entry
            self.read_set[key] = (value, ts)
            return value
        return None

    def write(self, key: str, value: any) -> None:
        """Buffer a write (applied at commit time)."""
        self.write_set[key] = value

    def commit(self, store: dict) -> bool:
        """
        Commit with TrueTime-based external consistency.

        Steps:
        1. Acquire write locks (simplified)
        2. Pick commit timestamp = TT.now().latest
        3. COMMIT WAIT: wait until TT.after(commit_timestamp)
        4. Apply writes with commit timestamp
        5. Release locks
        """
        if self.state != "active":
            return False

        # Step 2: Pick commit timestamp (guaranteed ≥ true time)
        interval = self.tt.now()
        self.commit_timestamp = interval.latest

        # Step 3: Commit wait — ensure no future tx can get a lower timestamp
        self.tt.wait_until_after(self.commit_timestamp)

        # Step 4: Apply writes
        for key, value in self.write_set.items():
            store[key] = (value, self.commit_timestamp)

        self.state = "committed"
        return True


class SpannerReadOnlyTransaction:
    """
    Spanner read-only transaction.
    Lock-free: picks a timestamp and reads a consistent snapshot.
    Can execute on ANY replica (not just the leader).
    """

    def __init__(self, tt: TrueTime):
        # Pick a read timestamp that is guaranteed to be in the past
        # (all writes with timestamp ≤ this are committed)
        interval = tt.now()
        self.read_timestamp = interval.latest
        # Wait to ensure this timestamp is in the past
        tt.wait_until_after(self.read_timestamp)

    def read(self, store: dict, key: str) -> any:
        """
        Read the value of key at self.read_timestamp.
        Uses multi-version storage: reads the latest version ≤ read_timestamp.
        """
        entry = store.get(key)
        if entry:
            value, write_ts = entry
            if write_ts <= self.read_timestamp:
                return value
        return None
```

### 1.4 Schema Interleaving

Spanner supports **interleaved tables**: child table rows are physically co-located with their parent row. This is crucial for performance in a distributed database.

```sql
-- Parent table
CREATE TABLE Users (
  user_id INT64 NOT NULL,
  name    STRING(100),
) PRIMARY KEY (user_id);

-- Child table interleaved in parent
CREATE TABLE Orders (
  user_id  INT64 NOT NULL,
  order_id INT64 NOT NULL,
  total    FLOAT64,
) PRIMARY KEY (user_id, order_id),
  INTERLEAVE IN PARENT Users ON DELETE CASCADE;
```

```
Physical layout (interleaved):

Tablet for key range [user_id 1000..2000]:
  ┌─────────────────────────────────────────┐
  │ Users(1000, "Alice")                    │
  │   Orders(1000, 1, 59.99)               │
  │   Orders(1000, 2, 124.50)              │
  │ Users(1001, "Bob")                      │
  │   Orders(1001, 1, 29.99)               │
  │ Users(1002, "Carol")                    │
  │   (no orders)                           │
  └─────────────────────────────────────────┘

Query: "SELECT * FROM Orders WHERE user_id = 1000"
→ Hits exactly ONE tablet, reads co-located data
→ No distributed join needed!
```

### 1.5 Spanner Performance Characteristics

| Operation | Latency | Notes |
|---|---|---|
| Read-write transaction | ~10-15ms | Paxos + commit wait (~2ε ≈ 8ms) |
| Read-only transaction (single region) | ~1-5ms | Lock-free snapshot read from local replica |
| Read-only transaction (cross-region) | ~5-50ms | Must wait for safe snapshot timestamp |
| Strong read | ~5-10ms | Reads at latest timestamp |
| Stale read (bounded staleness) | ~1-2ms | Reads at slightly older timestamp, avoids wait |

### 1.6 Spanner SLA

Google Cloud Spanner offers:
- **99.999% availability** (multi-region) — ~5 minutes downtime per year
- **99.99% availability** (single-region) — ~52 minutes downtime per year
- **External consistency** — strongest consistency guarantee offered by any commercial database

---

## 2. Amazon Dynamo

Amazon Dynamo (2007 paper) is the foundational design for many modern distributed key-value stores. Its design philosophy is radically different from Spanner.

### 2.1 Design Philosophy

| Principle | Spanner | Dynamo |
|---|---|---|
| **Consistency** | External consistency (strongest) | Eventual consistency (weakest practical) |
| **Availability** | Sacrifices availability for consistency | **Always writable** — never rejects writes |
| **Design for** | Strong guarantees | SLA-driven latency (99.9th percentile) |
| **Conflict resolution** | Prevention (locks + Paxos) | Detection + resolution (vector clocks, LWW) |

Dynamo was designed for Amazon's shopping cart, where it is always better to accept a write (even if conflicting) than to reject it. A customer adding items to their cart should never see an error.

### 2.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Dynamo Cluster                               │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Node A   │  │ Node B   │  │ Node C   │  │ Node D   │  ...     │
│  │          │  │          │  │          │  │          │          │
│  │ Key range│  │ Key range│  │ Key range│  │ Key range│          │
│  │ [0, 90)  │  │ [60, 180)│  │[120, 270)│  │[210, 360)│          │
│  │          │  │          │  │          │  │          │          │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘          │
│       │              │              │              │               │
│       └──── Consistent Hashing Ring (with virtual nodes) ─────────│
│       └──── Gossip Protocol (membership + failure detection) ─────│
│       └──── Sloppy Quorums + Hinted Handoff ──────────────────────│
│       └──── Vector Clocks (conflict detection) ───────────────────│
│       └──── Merkle Trees (anti-entropy) ──────────────────────────│
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Key Technologies

Each of the following technologies was covered in detail in Lessons 9-11. Here we see how Dynamo combines them:

#### Consistent Hashing with Virtual Nodes

Dynamo uses consistent hashing (Lesson 11) with virtual nodes to distribute keys. Each physical node is responsible for multiple positions on the hash ring.

```python
class DynamoNode:
    """
    Simplified Dynamo node with vector clocks and quorum operations.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.store: dict[str, list[tuple[any, dict[str, int]]]] = {}
        # key -> list of (value, vector_clock) for conflict tracking
        self.hint_store: list[tuple[str, str, any, dict[str, int]]] = []
        # (target_node_id, key, value, vector_clock)

    def put(self, key: str, value: any,
            context: dict[str, int] | None = None) -> dict[str, int]:
        """
        Write a value with vector clock versioning.
        context: the vector clock from a previous read (None for new key).
        """
        # Increment this node's entry in the vector clock
        vc = dict(context) if context else {}
        vc[self.node_id] = vc.get(self.node_id, 0) + 1

        if key not in self.store:
            self.store[key] = []

        # Remove versions dominated by the new vector clock
        surviving = []
        for old_val, old_vc in self.store[key]:
            if not self._vc_dominates(vc, old_vc):
                surviving.append((old_val, old_vc))
        surviving.append((value, vc))
        self.store[key] = surviving

        return vc

    def get(self, key: str) -> list[tuple[any, dict[str, int]]]:
        """
        Read all versions of a key.
        Returns list of (value, vector_clock) — may have multiple
        concurrent versions (conflicts).
        """
        return self.store.get(key, [])

    @staticmethod
    def _vc_dominates(vc_a: dict[str, int], vc_b: dict[str, int]) -> bool:
        """Check if vector clock A dominates (causally after) B."""
        # A dominates B iff A[i] >= B[i] for all i and A[j] > B[j] for some j
        all_nodes = set(vc_a.keys()) | set(vc_b.keys())
        at_least_one_greater = False
        for node in all_nodes:
            a_val = vc_a.get(node, 0)
            b_val = vc_b.get(node, 0)
            if a_val < b_val:
                return False
            if a_val > b_val:
                at_least_one_greater = True
        return at_least_one_greater


class DynamoCoordinator:
    """
    Coordinator for Dynamo-style operations.
    Handles quorum reads/writes with conflict detection.
    """

    def __init__(self, nodes: list[DynamoNode], n: int = 3, w: int = 2, r: int = 2):
        self.nodes = nodes
        self.n = n
        self.w = w
        self.r = r

    def put(self, key: str, value: any,
            context: dict[str, int] | None = None) -> dict:
        """
        Quorum write with vector clock.
        Sends to N nodes, waits for W acknowledgments.
        """
        # Select N nodes (simplified: first N)
        target_nodes = self.nodes[:self.n]
        acks = 0
        final_vc = None

        for node in target_nodes:
            try:
                vc = node.put(key, value, context)
                acks += 1
                final_vc = vc
            except Exception:
                pass

        return {
            "success": acks >= self.w,
            "acks": acks,
            "vector_clock": final_vc,
        }

    def get(self, key: str) -> dict:
        """
        Quorum read.
        Reads from N nodes, waits for R responses.
        Returns all concurrent versions (for client-side resolution).
        """
        target_nodes = self.nodes[:self.n]
        all_versions: list[tuple[any, dict[str, int]]] = []
        responses = 0

        for node in target_nodes:
            versions = node.get(key)
            if versions:
                all_versions.extend(versions)
                responses += 1

        # Merge: keep only non-dominated versions
        merged = self._merge_versions(all_versions)

        return {
            "success": responses >= self.r,
            "responses": responses,
            "versions": merged,
            "has_conflict": len(merged) > 1,
        }

    @staticmethod
    def _merge_versions(
            versions: list[tuple[any, dict[str, int]]]
    ) -> list[tuple[any, dict[str, int]]]:
        """Keep only versions not dominated by any other version."""
        if not versions:
            return []
        result = []
        for val, vc in versions:
            dominated = any(
                DynamoNode._vc_dominates(other_vc, vc)
                for other_val, other_vc in versions
                if other_vc != vc
            )
            if not dominated and (val, vc) not in result:
                result.append((val, vc))
        return result


def demo_dynamo_conflict():
    """Demonstrate Dynamo-style conflict detection with vector clocks."""
    nodes = [DynamoNode(f"N{i}") for i in range(3)]
    coord = DynamoCoordinator(nodes, n=3, w=2, r=2)

    # Initial write
    result = coord.put("cart:user1", {"items": ["book"]})
    print(f"Initial write: {result}")

    # Simulate partition: two nodes independently update
    # Node 0 adds "pen"
    vc0 = nodes[0].put("cart:user1", {"items": ["book", "pen"]},
                        context={"N0": 1})

    # Node 1 (didn't see the pen addition) adds "notebook"
    vc1 = nodes[1].put("cart:user1", {"items": ["book", "notebook"]},
                        context={"N1": 1})

    # Read detects conflict: two concurrent versions
    read_result = coord.get("cart:user1")
    print(f"\nConflict detected: {read_result['has_conflict']}")
    print(f"Concurrent versions:")
    for val, vc in read_result["versions"]:
        print(f"  {val} (vc={vc})")

    # Client resolves by merging (union of items)
    if read_result["has_conflict"]:
        all_items = set()
        merged_vc: dict[str, int] = {}
        for val, vc in read_result["versions"]:
            all_items.update(val.get("items", []))
            for node, count in vc.items():
                merged_vc[node] = max(merged_vc.get(node, 0), count)

        resolved = {"items": sorted(all_items)}
        coord.put("cart:user1", resolved, context=merged_vc)
        print(f"\nResolved: {resolved}")

        # Verify: no more conflicts
        final = coord.get("cart:user1")
        print(f"After resolution, conflict: {final['has_conflict']}")
        print(f"Final value: {final['versions'][0][0]}")
```

### 2.4 Dynamo vs DynamoDB

Amazon Dynamo (the 2007 paper) is an *internal* system. **Amazon DynamoDB** (the managed service, launched 2012) is inspired by Dynamo but differs significantly:

| Aspect | Dynamo (2007 paper) | DynamoDB (managed service) |
|---|---|---|
| **Conflict resolution** | Vector clocks + client resolution | Last-writer-wins (simplified) |
| **Partitioning** | Consistent hashing | Hash + range (compound) |
| **Membership** | Gossip protocol | Centralized metadata service |
| **Storage** | In-memory + disk | SSD-backed with automatic tiering |
| **Transactions** | None | TransactWriteItems / TransactGetItems (2018) |
| **Strong consistency** | Not supported | Optional per-read (ConsistentRead=true) |
| **Global tables** | Manual multi-DC | Built-in multi-region replication |

---

## 3. Apache Kafka

Kafka is a distributed commit log platform used for event streaming, pub/sub messaging, and event sourcing. While not a database in the traditional sense, it is a distributed storage system with unique consistency and durability properties.

### 3.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Kafka Cluster                               │
│                                                                  │
│  Topic: "orders" (3 partitions, replication factor 2)           │
│                                                                  │
│  Broker 1           Broker 2           Broker 3                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ P0 (Leader)  │  │ P0 (Follower)│  │ P1 (Follower)│         │
│  │ P1 (Leader)  │  │ P2 (Leader)  │  │ P2 (Follower)│         │
│  │ P2 (Follower)│  │ P1 (Follower)│  │ P0 (spare)   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
│  Each partition is an append-only log:                           │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐                    │
│  │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │  ← offsets       │
│  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘                    │
│  │◄─── committed ────►│◄─── uncommitted ──►│                    │
│            (HW)                 (LEO)                            │
└─────────────────────────────────────────────────────────────────┘

HW  = High Watermark (all ISR replicas have this offset)
LEO = Log End Offset (latest offset on the leader)
```

### 3.2 ISR (In-Sync Replicas) and Leader Election

Kafka uses a concept called **ISR** — the set of replicas that are "in sync" with the leader. A replica is in the ISR if it has fetched all messages up to the leader's log end within a configurable time window.

```
Leader (Broker 1):  [0, 1, 2, 3, 4, 5, 6]  ← LEO = 7
ISR Replica (Br 2): [0, 1, 2, 3, 4, 5]      ← caught up within timeout → IN ISR
Lagging (Br 3):     [0, 1, 2, 3]             ← too far behind → OUT of ISR

High Watermark (HW) = min(LEO of all ISR replicas) = 6
Messages 0-5 are "committed" (safe to consume)
Message 6 is on Br 2 but not committed (Br 3 doesn't have it — but Br 3 is out of ISR)
```

**Leader election**: When the leader fails, a new leader is chosen from the ISR. Only ISR members are eligible (unless `unclean.leader.election.enable=true`, which risks data loss).

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class AckMode(Enum):
    FIRE_AND_FORGET = 0    # acks=0: don't wait
    LEADER_ACK = 1          # acks=1: leader confirms write to its log
    ALL_ISR_ACK = -1        # acks=all: all ISR replicas confirm


@dataclass
class LogEntry:
    """A single entry in the Kafka log."""
    offset: int
    key: Optional[str]
    value: str
    timestamp: float
    producer_id: Optional[int] = None
    sequence_num: Optional[int] = None  # for idempotent producers


@dataclass
class KafkaPartition:
    """Simulated Kafka partition with ISR tracking."""
    topic: str
    partition_id: int
    leader_id: int
    isr: list[int]  # broker IDs in ISR
    log: list[LogEntry] = field(default_factory=list)
    high_watermark: int = 0

    @property
    def leo(self) -> int:
        """Log End Offset."""
        return len(self.log)

    def append(self, key: Optional[str], value: str,
               ack_mode: AckMode = AckMode.ALL_ISR_ACK,
               producer_id: Optional[int] = None,
               sequence_num: Optional[int] = None) -> dict:
        """
        Append a message to the partition log.
        Acknowledgment depends on ack_mode.
        """
        # Idempotent producer: reject duplicate sequence numbers
        if producer_id is not None and sequence_num is not None:
            for entry in reversed(self.log):
                if (entry.producer_id == producer_id and
                        entry.sequence_num == sequence_num):
                    return {
                        "status": "duplicate",
                        "offset": entry.offset,
                    }

        import time
        entry = LogEntry(
            offset=self.leo,
            key=key,
            value=value,
            timestamp=time.time(),
            producer_id=producer_id,
            sequence_num=sequence_num,
        )
        self.log.append(entry)

        if ack_mode == AckMode.ALL_ISR_ACK:
            # In production: wait for all ISR replicas to fetch this offset
            self.high_watermark = self.leo  # simplified: assume instant replication
        elif ack_mode == AckMode.LEADER_ACK:
            pass  # leader-only ack, HW advances when replicas catch up
        elif ack_mode == AckMode.FIRE_AND_FORGET:
            pass  # no waiting at all

        return {
            "status": "ok",
            "offset": entry.offset,
            "partition": self.partition_id,
        }

    def read(self, offset: int, max_records: int = 10,
             read_committed: bool = True) -> list[LogEntry]:
        """
        Read records starting from offset.
        read_committed=True: only return up to high watermark.
        """
        end = self.high_watermark if read_committed else self.leo
        return self.log[offset:min(offset + max_records, end)]
```

### 3.3 Log Compaction

Kafka supports **log compaction**: instead of deleting old segments by time, keep only the latest value for each key. This turns Kafka into a changelog-style store.

```
Before compaction:
  offset: 0   key=A  value="v1"
  offset: 1   key=B  value="v1"
  offset: 2   key=A  value="v2"    ← supersedes offset 0
  offset: 3   key=C  value="v1"
  offset: 4   key=B  value="v2"    ← supersedes offset 1
  offset: 5   key=A  value=null    ← tombstone: delete key A

After compaction:
  offset: 3   key=C  value="v1"    (latest for C)
  offset: 4   key=B  value="v2"    (latest for B)
  (key A deleted: tombstone removes it entirely after grace period)
```

```python
def compact_log(log: list[LogEntry]) -> list[LogEntry]:
    """
    Simulate Kafka log compaction.
    Keep only the latest entry for each key.
    Tombstones (value=None) delete the key entirely.
    """
    # Track the latest entry for each key
    latest: dict[str, LogEntry] = {}
    for entry in log:
        if entry.key is not None:
            latest[entry.key] = entry

    # Build compacted log, preserving order
    compacted = []
    seen_keys: set[str] = set()
    for entry in reversed(log):
        if entry.key is None:
            continue
        if entry.key not in seen_keys:
            seen_keys.add(entry.key)
            if entry.value is not None:  # skip tombstones
                compacted.append(entry)

    compacted.reverse()
    return compacted
```

### 3.4 Exactly-Once Semantics (EOS)

Kafka achieves exactly-once processing through two mechanisms:

1. **Idempotent producers**: Each producer gets a unique ID and sequence numbers. The broker deduplicates messages.

2. **Transactions**: Atomic writes across multiple partitions. Either all messages are committed or none.

```python
class KafkaTransaction:
    """
    Simplified Kafka transaction.
    Provides atomic writes across multiple partitions.
    """

    def __init__(self, transaction_id: str, partitions: list[KafkaPartition]):
        self.transaction_id = transaction_id
        self.partitions = {p.partition_id: p for p in partitions}
        self.pending: list[tuple[int, str, str]] = []  # (partition_id, key, value)
        self.state = "active"  # active → preparing → committed / aborted

    def send(self, partition_id: int, key: str, value: str) -> None:
        """Buffer a message for transactional send."""
        if self.state != "active":
            raise RuntimeError(f"Transaction is {self.state}")
        self.pending.append((partition_id, key, value))

    def commit(self) -> bool:
        """
        Commit the transaction atomically.
        Uses a two-phase protocol with a transaction coordinator.
        """
        self.state = "preparing"

        # Phase 1: Write all messages to partition logs (not yet visible)
        offsets = []
        for pid, key, value in self.pending:
            partition = self.partitions[pid]
            result = partition.append(key, value)
            offsets.append(result)

        # Phase 2: Write commit marker to transaction log
        # (In real Kafka: __transaction_state internal topic)
        self.state = "committed"

        # Messages become visible to consumers with read_committed=True
        return True

    def abort(self) -> None:
        """Abort the transaction. Pending messages are discarded."""
        self.state = "aborted"
        self.pending.clear()
```

### 3.5 Consumer Groups and Partition Assignment

```
Topic "orders" with 4 partitions:

Consumer Group "order-processing":
  Consumer A: assigned P0, P1
  Consumer B: assigned P2, P3

  Each partition is consumed by exactly ONE consumer in the group.
  Adding Consumer C triggers rebalance:
    Consumer A: P0
    Consumer B: P2, P3
    Consumer C: P1

  Max parallelism = number of partitions
```

### 3.6 KRaft: Removing ZooKeeper

Since Kafka 3.3 (2022), Kafka uses **KRaft** (Kafka Raft) for metadata management, replacing ZooKeeper:

| Aspect | ZooKeeper-based | KRaft |
|---|---|---|
| **Metadata storage** | External ZooKeeper ensemble | Internal Raft-based metadata quorum |
| **Controller** | Single active controller (elected via ZK) | Active controller + standby controllers via Raft |
| **Scalability** | ZK bottleneck at ~200K partitions | Millions of partitions |
| **Operations** | Two systems to manage (Kafka + ZK) | Single system |
| **Startup** | Must start ZK first | Self-contained |

---

## 4. CockroachDB

CockroachDB is an open-source, distributed SQL database designed to provide the consistency guarantees of Spanner without requiring specialized hardware (like GPS/atomic clocks).

### 4.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CockroachDB Cluster                          │
│                                                                  │
│  SQL Layer:     Parse → Plan → Optimize → Distribute → Execute  │
│                            │                                     │
│  Distribution Layer:       │                                     │
│    ┌────────────────────────┼──────────────────────┐            │
│    │                        ▼                      │            │
│    │  Range 1            Range 2         Range N   │            │
│    │  [A..F)             [F..K)          [X..Z)    │            │
│    │  ┌────────────┐    ┌────────────┐   ┌──────┐  │            │
│    │  │Raft Group 1│    │Raft Group 2│   │Raft N│  │            │
│    │  │ L F F      │    │ F L F      │   │F F L │  │            │
│    │  └────────────┘    └────────────┘   └──────┘  │            │
│    │  (L=Leader, F=Follower)                       │            │
│    └───────────────────────────────────────────────┘            │
│                                                                  │
│  Storage Layer:  Pebble (LSM-tree key-value engine, Go)         │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Range-Based Partitioning with Automatic Splitting

CockroachDB partitions data into **ranges** (default 512 MB). Each range is a contiguous span of the key space. When a range grows beyond the threshold, it automatically splits.

```
Initial state:
  Range 1: ["", "") — one range covering everything

After data growth:
  Range 1: ["", "customer:5000")
  Range 2: ["customer:5000", "order:3000")
  Range 3: ["order:3000", "")

Each range is independently replicated via its own Raft group.
```

### 4.3 Transaction Model: Write Intents and Parallel Commits

CockroachDB uses a unique transaction protocol that avoids the need for TrueTime.

#### Write Intents

Instead of acquiring locks, transactions write **intents** — provisional values that are visible to other transactions as "uncommitted."

```
Transaction T1 writes key "x":

Before T1:
  key "x" → value "old" (committed)

During T1 (before commit):
  key "x" → INTENT(txn=T1, value="new", status=PENDING)
  (other transactions see the intent and check T1's status)

After T1 commits:
  key "x" → value "new" (committed, intent cleaned up)

If T1 aborts:
  key "x" → value "old" (intent cleaned up)
```

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import time


class TxnStatus(Enum):
    PENDING = "pending"
    COMMITTED = "committed"
    ABORTED = "aborted"


@dataclass
class WriteIntent:
    """A provisional write that may or may not be committed."""
    txn_id: str
    key: str
    value: Any
    timestamp: float
    status: TxnStatus = TxnStatus.PENDING


@dataclass
class CockroachTransaction:
    """
    Simplified CockroachDB transaction with write intents.
    """
    txn_id: str
    timestamp: float = field(default_factory=time.time)
    status: TxnStatus = TxnStatus.PENDING
    intents: list[WriteIntent] = field(default_factory=list)

    def write(self, store: dict, key: str, value: Any) -> WriteIntent:
        """
        Write an intent (provisional value).
        The intent is visible to other transactions as PENDING.
        """
        intent = WriteIntent(
            txn_id=self.txn_id,
            key=key,
            value=value,
            timestamp=self.timestamp,
        )
        store[key] = intent
        self.intents.append(intent)
        return intent

    def read(self, store: dict, key: str) -> Optional[Any]:
        """
        Read a key. If there's a pending intent from another txn,
        we must check that txn's status (push or wait).
        """
        entry = store.get(key)
        if entry is None:
            return None

        if isinstance(entry, WriteIntent):
            if entry.txn_id == self.txn_id:
                # Our own intent — read the provisional value
                return entry.value
            elif entry.status == TxnStatus.COMMITTED:
                return entry.value
            elif entry.status == TxnStatus.PENDING:
                # Conflict! In real CockroachDB:
                # 1. Push the other txn's timestamp forward, OR
                # 2. Wait for the other txn to complete, OR
                # 3. Abort the other txn if it's expired
                raise Exception(
                    f"Write-read conflict: {self.txn_id} blocked by "
                    f"{entry.txn_id} on key {key}"
                )
            else:
                # Aborted intent — ignore it
                return None
        else:
            return entry

    def commit(self, txn_record: dict) -> bool:
        """
        Commit the transaction.
        1. Write COMMITTED status to transaction record
        2. Asynchronously clean up intents
        """
        self.status = TxnStatus.COMMITTED
        txn_record[self.txn_id] = self.status

        # Mark all intents as committed
        for intent in self.intents:
            intent.status = TxnStatus.COMMITTED

        return True

    def abort(self, txn_record: dict) -> None:
        """Abort the transaction and clean up intents."""
        self.status = TxnStatus.ABORTED
        txn_record[self.txn_id] = self.status
        for intent in self.intents:
            intent.status = TxnStatus.ABORTED
```

#### Parallel Commits

CockroachDB's **parallel commits** optimization reduces commit latency by writing the transaction record and intents in parallel, rather than sequentially.

```
Standard 2-phase commit:
  1. Write intents to all ranges                    (1 round trip)
  2. Write COMMITTED to transaction record           (1 round trip)
  3. Clean up intents asynchronously
  Total: 2 round trips before acknowledging client

Parallel commits:
  1. Write intents + STAGING status in parallel      (1 round trip)
  2. Transaction is implicitly committed if all
     intents are present (verified lazily)
  Total: 1 round trip before acknowledging client
```

### 4.4 Clock Synchronization: Hybrid Logical Clocks

CockroachDB doesn't have TrueTime. Instead, it uses **Hybrid Logical Clocks (HLC)** — a combination of physical time and logical counters (from Lesson 2).

```python
@dataclass
class HybridLogicalClock:
    """
    Hybrid Logical Clock (HLC).
    Combines physical time with a logical counter.
    Guarantees causality without perfect clock synchronization.
    """
    physical: int = 0    # wall clock time in nanoseconds
    logical: int = 0     # logical counter for same-physical-time ordering
    node_id: str = ""

    def now(self) -> 'HybridLogicalClock':
        """Generate a new HLC timestamp."""
        import time
        wall = int(time.time() * 1e9)

        if wall > self.physical:
            self.physical = wall
            self.logical = 0
        else:
            self.logical += 1

        return HybridLogicalClock(self.physical, self.logical, self.node_id)

    def update(self, remote: 'HybridLogicalClock') -> 'HybridLogicalClock':
        """Update HLC upon receiving a message from another node."""
        import time
        wall = int(time.time() * 1e9)

        if wall > self.physical and wall > remote.physical:
            self.physical = wall
            self.logical = 0
        elif remote.physical > self.physical:
            self.physical = remote.physical
            self.logical = remote.logical + 1
        elif self.physical > remote.physical:
            self.logical += 1
        else:
            # Same physical time
            self.logical = max(self.logical, remote.logical) + 1

        return HybridLogicalClock(self.physical, self.logical, self.node_id)

    def __lt__(self, other: 'HybridLogicalClock') -> bool:
        if self.physical != other.physical:
            return self.physical < other.physical
        return self.logical < other.logical

    def __repr__(self) -> str:
        return f"HLC({self.physical}, {self.logical})"
```

**The uncertainty window problem**: Without TrueTime, CockroachDB cannot precisely bound clock skew. It uses a configurable `max_offset` (default 500ms). If a node's clock is detected to be off by more than `max_offset`, it is ejected from the cluster.

For reads, CockroachDB may need to perform an **uncertainty restart**: if a read encounters a value with a timestamp within the uncertainty window, it restarts the transaction at a higher timestamp.

### 4.5 Multi-Region Capabilities

CockroachDB supports several multi-region topologies:

```
1. REGIONAL BY ROW:
   Each row specifies its home region. Reads are fast from that region.
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ US-East │    │ EU-West │    │ AP-SE   │
   │ US rows │    │ EU rows │    │ AP rows │
   │ (leader)│    │ (leader)│    │ (leader)│
   └─────────┘    └─────────┘    └─────────┘

2. GLOBAL TABLES:
   All regions can read locally (stale follower reads).
   Writes go to the leaseholder in any region.

3. REGIONAL TABLES:
   Entire table is homed in one region.
   All reads and writes go to that region.
```

---

## 5. Comprehensive Comparison

### 5.1 Feature Comparison Table

| Feature | Spanner | Dynamo/DynamoDB | Kafka | CockroachDB |
|---|---|---|---|---|
| **Type** | Relational (SQL) | Key-Value | Distributed Log | Relational (SQL) |
| **Consistency** | External (strongest) | Eventual (configurable) | Per-partition ordering | Serializable |
| **Transactions** | Full ACID (2PC + Paxos) | Limited (DDB: single-item or transact API) | Producer transactions | Full ACID (parallel commits) |
| **Partitioning** | Range (directory-based) | Hash (consistent hashing) | Hash (by key or round-robin) | Range (automatic splitting) |
| **Replication** | Paxos per split | Sloppy quorum (N, W, R) | ISR (leader + in-sync replicas) | Raft per range |
| **Conflict resolution** | Prevention (locks) | Detection (vector clocks / LWW) | N/A (append-only) | Prevention (write intents) |
| **Clock mechanism** | TrueTime (GPS + atomic) | Vector clocks → LWW | N/A (offset-based ordering) | Hybrid Logical Clocks |
| **Schema** | SQL with interleaved tables | Key-value (DDB: document) | Schema Registry (Avro/Protobuf) | PostgreSQL-compatible SQL |
| **Open source** | No (Cloud Spanner) | No (DynamoDB) | Yes (Apache 2.0) | Yes (BSL → Apache 2.0) |

### 5.2 Consistency and Availability Trade-offs

```
                        Consistency
                            ▲
                            │
          Spanner ●         │
                            │
   CockroachDB ●           │
                            │
                            │
                            │
                ────────────┼────────────► Availability
                            │
              Kafka ●       │
     (per-partition)        │
                            │
               DynamoDB ●   │  (configurable)
                            │
          Dynamo ●          │
     (always writable)      │
```

### 5.3 When to Use Each System

| System | Best For | Not Ideal For |
|---|---|---|
| **Spanner** | Global transactions, financial systems, inventory management | Cost-sensitive workloads, simple key-value access |
| **DynamoDB** | High-throughput key-value, serverless backends, shopping carts | Complex queries, joins, strong consistency requirements |
| **Kafka** | Event streaming, log aggregation, event sourcing, CDC | Low-latency point queries, OLTP workloads |
| **CockroachDB** | Multi-region SQL, PostgreSQL migration, regulatory compliance | Ultra-low latency single-region, very high write throughput |

---

## 6. Lessons Learned: Common Patterns

Studying these four systems reveals recurring patterns in distributed storage design.

### 6.1 Pattern: Partition + Replicate

Every system combines partitioning (for scale) with replication (for fault tolerance).

```python
def summarize_partition_replicate():
    """Common pattern: how each system partitions and replicates."""
    systems = {
        "Spanner": {
            "partition": "Range-based (directories, key-order preserving)",
            "replicate": "Paxos per split (synchronous, cross-zone)",
            "unit": "Split (tablet)",
        },
        "Dynamo": {
            "partition": "Consistent hashing with virtual nodes",
            "replicate": "Sloppy quorum (N copies, W write acks, R read acks)",
            "unit": "Virtual node range",
        },
        "Kafka": {
            "partition": "Hash or key-based (producer determines partition)",
            "replicate": "ISR-based (leader + in-sync followers)",
            "unit": "Partition (append-only log segment)",
        },
        "CockroachDB": {
            "partition": "Range-based with automatic splitting at 512MB",
            "replicate": "Raft per range (typically 3 or 5 replicas)",
            "unit": "Range",
        },
    }

    for name, details in systems.items():
        print(f"\n{name}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
```

### 6.2 Pattern: Consensus per Partition

Spanner, CockroachDB, and Kafka all use per-partition consensus (Paxos, Raft, or ISR) rather than cluster-wide consensus. This is critical for scalability — consensus overhead is per-partition, not global.

### 6.3 Pattern: Timestamp-Based MVCC

All four systems use some form of timestamp ordering:

| System | Timestamp Source | MVCC? |
|---|---|---|
| Spanner | TrueTime (physical + uncertainty) | Yes |
| Dynamo | Vector clocks (logical) | Multiple versions per key |
| Kafka | Offsets (logical, per-partition) | Append-only (natural versioning) |
| CockroachDB | HLC (physical + logical) | Yes |

### 6.4 Pattern: Trade-Off Between Latency and Consistency

```
Higher Consistency ─────────────────────────► Lower Consistency
Higher Latency                                Lower Latency

Spanner          CockroachDB       Kafka ISR        Dynamo
(commit wait     (uncertainty       (acks=all)      (sloppy quorum
 ~2ε ≈ 8ms)      restart)                           W=1)
```

### 6.5 Anti-Pattern: Clock Dependency

| System | Clock Dependency | Risk |
|---|---|---|
| Spanner | High (TrueTime is critical) | Mitigated by GPS + atomic clocks |
| Dynamo | Low (vector clocks are logical) | No clock risk, but metadata overhead |
| CockroachDB | Medium (HLC + max_offset) | Clock skew beyond max_offset → node ejection |
| Kafka | None (offset-based) | No clock risk |

---

## 7. Architectural Decision Simulator

```python
"""
Decision framework for choosing a distributed storage system.
Scores each system based on workload requirements.
"""

from dataclasses import dataclass


@dataclass
class WorkloadRequirements:
    """Describe the workload to get system recommendations."""
    needs_sql: bool = False
    needs_transactions: bool = False
    consistency: str = "eventual"  # "strong", "serializable", "eventual"
    read_write_ratio: float = 1.0  # >1 = read-heavy, <1 = write-heavy
    latency_p99_ms: float = 100.0
    multi_region: bool = False
    throughput_qps: int = 10000
    data_model: str = "key_value"  # "key_value", "relational", "event_log"
    open_source_required: bool = False
    budget: str = "medium"  # "low", "medium", "high"


def recommend_system(req: WorkloadRequirements) -> list[tuple[str, float, str]]:
    """
    Score each system (0-100) based on workload fit.
    Returns sorted list of (system, score, rationale).
    """
    scores: dict[str, tuple[float, list[str]]] = {
        "Spanner": (50.0, []),
        "DynamoDB": (50.0, []),
        "Kafka": (50.0, []),
        "CockroachDB": (50.0, []),
    }

    # SQL support
    if req.needs_sql:
        scores["Spanner"] = (scores["Spanner"][0] + 15, scores["Spanner"][1] + ["SQL support"])
        scores["CockroachDB"] = (scores["CockroachDB"][0] + 15, scores["CockroachDB"][1] + ["PostgreSQL-compatible SQL"])
        scores["DynamoDB"] = (scores["DynamoDB"][0] - 10, scores["DynamoDB"][1] + ["No SQL (PartiQL is limited)"])
        scores["Kafka"] = (scores["Kafka"][0] - 20, scores["Kafka"][1] + ["Not a query engine"])

    # Transactions
    if req.needs_transactions:
        scores["Spanner"] = (scores["Spanner"][0] + 15, scores["Spanner"][1] + ["Full ACID transactions"])
        scores["CockroachDB"] = (scores["CockroachDB"][0] + 15, scores["CockroachDB"][1] + ["Serializable transactions"])
        scores["DynamoDB"] = (scores["DynamoDB"][0] - 5, scores["DynamoDB"][1] + ["Limited transactions"])
        scores["Kafka"] = (scores["Kafka"][0] + 5, scores["Kafka"][1] + ["Producer transactions (EOS)"])

    # Consistency
    if req.consistency == "strong":
        scores["Spanner"] = (scores["Spanner"][0] + 20, scores["Spanner"][1] + ["External consistency"])
        scores["CockroachDB"] = (scores["CockroachDB"][0] + 15, scores["CockroachDB"][1] + ["Serializable"])
        scores["DynamoDB"] = (scores["DynamoDB"][0] - 10, scores["DynamoDB"][1] + ["Eventual by default"])
    elif req.consistency == "eventual":
        scores["DynamoDB"] = (scores["DynamoDB"][0] + 10, scores["DynamoDB"][1] + ["Optimized for eventual"])
        scores["Kafka"] = (scores["Kafka"][0] + 10, scores["Kafka"][1] + ["Per-partition ordering"])

    # Data model
    if req.data_model == "event_log":
        scores["Kafka"] = (scores["Kafka"][0] + 25, scores["Kafka"][1] + ["Purpose-built for event logs"])
    elif req.data_model == "key_value":
        scores["DynamoDB"] = (scores["DynamoDB"][0] + 15, scores["DynamoDB"][1] + ["Purpose-built key-value"])

    # Multi-region
    if req.multi_region:
        scores["Spanner"] = (scores["Spanner"][0] + 15, scores["Spanner"][1] + ["Best-in-class multi-region"])
        scores["CockroachDB"] = (scores["CockroachDB"][0] + 10, scores["CockroachDB"][1] + ["Multi-region support"])

    # Open source
    if req.open_source_required:
        scores["Kafka"] = (scores["Kafka"][0] + 10, scores["Kafka"][1] + ["Apache 2.0"])
        scores["CockroachDB"] = (scores["CockroachDB"][0] + 10, scores["CockroachDB"][1] + ["Open source core"])
        scores["Spanner"] = (scores["Spanner"][0] - 15, scores["Spanner"][1] + ["Proprietary (Cloud only)"])
        scores["DynamoDB"] = (scores["DynamoDB"][0] - 15, scores["DynamoDB"][1] + ["Proprietary (AWS only)"])

    # Latency
    if req.latency_p99_ms < 10:
        scores["DynamoDB"] = (scores["DynamoDB"][0] + 10, scores["DynamoDB"][1] + ["Single-digit ms P99"])
        scores["Spanner"] = (scores["Spanner"][0] - 5, scores["Spanner"][1] + ["Commit wait adds latency"])

    # Budget
    if req.budget == "low":
        scores["CockroachDB"] = (scores["CockroachDB"][0] + 5, scores["CockroachDB"][1] + ["Self-hosted option"])
        scores["Kafka"] = (scores["Kafka"][0] + 5, scores["Kafka"][1] + ["Self-hosted option"])
        scores["Spanner"] = (scores["Spanner"][0] - 10, scores["Spanner"][1] + ["Expensive"])

    # Format results
    results = [
        (name, max(0, min(100, score)), "; ".join(reasons))
        for name, (score, reasons) in scores.items()
    ]
    results.sort(key=lambda x: -x[1])
    return results


def demo_decision_framework():
    """Demonstrate the decision framework with sample workloads."""

    workloads = [
        ("E-commerce order management", WorkloadRequirements(
            needs_sql=True, needs_transactions=True,
            consistency="strong", multi_region=True,
            data_model="relational", budget="high",
        )),
        ("Real-time event streaming", WorkloadRequirements(
            needs_sql=False, needs_transactions=False,
            consistency="eventual", throughput_qps=500000,
            data_model="event_log", open_source_required=True,
        )),
        ("User session store", WorkloadRequirements(
            needs_sql=False, needs_transactions=False,
            consistency="eventual", latency_p99_ms=5,
            data_model="key_value", budget="medium",
        )),
        ("Multi-region financial ledger", WorkloadRequirements(
            needs_sql=True, needs_transactions=True,
            consistency="strong", multi_region=True,
            data_model="relational", open_source_required=True,
        )),
    ]

    for name, req in workloads:
        print(f"\nWorkload: {name}")
        print("-" * 60)
        recommendations = recommend_system(req)
        for system, score, rationale in recommendations:
            print(f"  {system:<15} Score: {score:>5.0f}/100")
            if rationale:
                print(f"    Reasons: {rationale}")


if __name__ == "__main__":
    print("=== Dynamo Conflict Detection Demo ===")
    demo_dynamo_conflict()

    print("\n=== System Decision Framework ===")
    demo_decision_framework()
```

---

## 8. Summary

| System | Key Innovation | Core Trade-Off |
|---|---|---|
| **Spanner** | TrueTime enables external consistency at global scale | Requires specialized hardware (GPS + atomic clocks); higher write latency |
| **Dynamo** | Always-writable with sloppy quorums and conflict detection | Application must handle conflicts; weaker consistency |
| **Kafka** | Distributed append-only commit log with ISR replication | Not a general-purpose database; no point queries |
| **CockroachDB** | Serializable SQL without specialized hardware | Uncertainty restarts; higher latency than eventually consistent systems |

### Key Insights Across All Systems

1. **There is no free lunch**: Every system trades something. Spanner trades cost and latency for consistency. Dynamo trades consistency for availability and latency.

2. **Consensus per partition scales**: All systems that provide strong guarantees use per-partition consensus (not global consensus).

3. **Time is the hardest problem**: Each system's approach to time (TrueTime, vector clocks, HLC, offsets) defines its fundamental characteristics.

4. **Physical co-location matters**: Spanner's interleaved tables, CockroachDB's ranges, Kafka's partitions — all leverage data locality.

5. **Operational simplicity has value**: Kafka's move from ZooKeeper to KRaft, DynamoDB's serverless model — reducing operational complexity is a feature.

---

[Next: Failure Detection and Membership](./13_Failure_Detection_and_Membership.md)
