# Lesson 25: Vector Clocks and Causality Tracking

[Overview](./00_Overview.md) | [Previous: Event Sourcing and CQRS](./24_Event_Sourcing_CQRS.md) | [Next: Distributed Testing](./26_Distributed_Testing.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement Lamport clocks, vector clocks, and version vectors from scratch
2. Determine causal ordering and detect concurrent events using vector timestamps
3. Build conflict resolution strategies using vector clocks (last-writer-wins, merge, CRDTs)
4. Implement dotted version vectors for accurate causality tracking in key-value stores
5. Analyze the space-time trade-offs of different logical clock mechanisms

---

## Table of Contents

1. [Causality in Distributed Systems](#1-causality-in-distributed-systems)
2. [Lamport Clocks](#2-lamport-clocks)
3. [Vector Clocks](#3-vector-clocks)
4. [Comparing Events with Vector Clocks](#4-comparing-events-with-vector-clocks)
5. [Version Vectors](#5-version-vectors)
6. [Conflict Detection and Resolution](#6-conflict-detection-and-resolution)
7. [Dotted Version Vectors](#7-dotted-version-vectors)
8. [Hybrid Logical Clocks](#8-hybrid-logical-clocks)
9. [Real-World Applications](#9-real-world-applications)
10. [Summary and Key Takeaways](#10-summary-and-key-takeaways)
11. [Practice Problems](#11-practice-problems)
12. [References](#12-references)

---

## 1. Causality in Distributed Systems

### 1.1 The Happens-Before Relation

Lamport's happens-before relation (→) captures causal ordering:

```
Process P1:  a ─────── b ─────── c
                        \
                         msg
                          \
Process P2:  d ─────── e ─ f ─── g

a → b (same process, a before b)
b → f (b sends message, f receives it)
a → f (transitivity: a → b → f)
d ∥ a (concurrent: no causal path between them)
```

```python
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy


@dataclass
class CausalEvent:
    """An event with causal metadata."""
    process_id: str
    event_id: str
    event_type: str  # "local", "send", "receive"
    data: dict = field(default_factory=dict)
    timestamp: dict = field(default_factory=dict)  # Logical timestamp
```

---

## 2. Lamport Clocks

### 2.1 Implementation

```python
class LamportClock:
    """
    Lamport logical clock implementation.

    Rules:
    1. Before each local event: counter += 1
    2. Before sending: counter += 1; attach counter to message
    3. On receive: counter = max(local, received) + 1

    Property: If a → b, then L(a) < L(b)
    Limitation: L(a) < L(b) does NOT imply a → b
    """

    def __init__(self, process_id: str):
        self.process_id = process_id
        self.counter: int = 0
        self.history: list[Tuple[str, int]] = []

    def local_event(self, event_name: str = "") -> int:
        """Record a local event."""
        self.counter += 1
        self.history.append((event_name or f"local_{self.counter}", self.counter))
        return self.counter

    def send(self) -> int:
        """Prepare a message for sending. Returns timestamp to attach."""
        self.counter += 1
        self.history.append(("send", self.counter))
        return self.counter

    def receive(self, received_timestamp: int) -> int:
        """Process a received message."""
        self.counter = max(self.counter, received_timestamp) + 1
        self.history.append(("receive", self.counter))
        return self.counter

    def get_time(self) -> int:
        return self.counter


def demonstrate_lamport_clocks():
    """Demonstrate Lamport clocks and their limitations."""
    print("=== Lamport Clocks ===\n")

    p1 = LamportClock("P1")
    p2 = LamportClock("P2")
    p3 = LamportClock("P3")

    # P1: local event, then send to P2
    p1.local_event("write_x")          # L=1
    ts = p1.send()                      # L=2
    p2.receive(ts)                      # L=3

    # P2: local event, then send to P3
    p2.local_event("process_x")        # L=4
    ts = p2.send()                      # L=5
    p3.receive(ts)                      # L=6

    # P3: local event (concurrent with P1's write_x)
    p3_local = p3.local_event("write_y")  # L=7

    # P1: another local event (concurrent with P3)
    p1.local_event("read_x")           # L=3

    print("Event histories:")
    for proc in [p1, p2, p3]:
        print(f"  {proc.process_id}: {proc.history}")

    print(f"\nLimitation demo:")
    print(f"  P1's read_x has L=3, P3's write_y has L=7")
    print(f"  L(read_x) < L(write_y) but they are CONCURRENT!")
    print(f"  Lamport clocks cannot detect concurrency")


demonstrate_lamport_clocks()
```

---

## 3. Vector Clocks

### 3.1 Implementation

```python
class VectorClock:
    """
    Vector clock implementation.

    Each process maintains a vector of counters, one per process.
    Vector clocks can determine causal ordering AND detect concurrency.

    Rules:
    1. Before each local event: VC[self] += 1
    2. Before sending: VC[self] += 1; attach VC to message
    3. On receive: VC[i] = max(VC[i], received[i]) for all i; VC[self] += 1
    """

    def __init__(self, process_id: str, all_processes: list[str]):
        self.process_id = process_id
        self.clock: Dict[str, int] = {p: 0 for p in all_processes}

    def local_event(self) -> Dict[str, int]:
        """Record a local event."""
        self.clock[self.process_id] += 1
        return self.get_timestamp()

    def send(self) -> Dict[str, int]:
        """Prepare to send a message. Returns timestamp to attach."""
        self.clock[self.process_id] += 1
        return self.get_timestamp()

    def receive(self, received_vc: Dict[str, int]) -> Dict[str, int]:
        """Process a received message with its vector timestamp."""
        for proc_id in self.clock:
            self.clock[proc_id] = max(
                self.clock.get(proc_id, 0),
                received_vc.get(proc_id, 0),
            )
        self.clock[self.process_id] += 1
        return self.get_timestamp()

    def get_timestamp(self) -> Dict[str, int]:
        """Return a copy of the current vector timestamp."""
        return dict(self.clock)

    @staticmethod
    def happens_before(vc1: Dict[str, int], vc2: Dict[str, int]) -> bool:
        """Check if vc1 → vc2 (vc1 happens before vc2)."""
        all_keys = set(vc1.keys()) | set(vc2.keys())
        at_least_one_less = False
        for key in all_keys:
            v1 = vc1.get(key, 0)
            v2 = vc2.get(key, 0)
            if v1 > v2:
                return False
            if v1 < v2:
                at_least_one_less = True
        return at_least_one_less

    @staticmethod
    def concurrent(vc1: Dict[str, int], vc2: Dict[str, int]) -> bool:
        """Check if vc1 ∥ vc2 (concurrent events)."""
        return (not VectorClock.happens_before(vc1, vc2) and
                not VectorClock.happens_before(vc2, vc1) and
                vc1 != vc2)

    @staticmethod
    def merge(vc1: Dict[str, int], vc2: Dict[str, int]) -> Dict[str, int]:
        """Merge two vector clocks (component-wise max)."""
        all_keys = set(vc1.keys()) | set(vc2.keys())
        return {k: max(vc1.get(k, 0), vc2.get(k, 0)) for k in all_keys}

    @staticmethod
    def compare(vc1: Dict[str, int], vc2: Dict[str, int]) -> str:
        """Compare two vector clocks. Returns: 'before', 'after', 'concurrent', 'equal'."""
        if vc1 == vc2:
            return "equal"
        if VectorClock.happens_before(vc1, vc2):
            return "before"
        if VectorClock.happens_before(vc2, vc1):
            return "after"
        return "concurrent"


def demonstrate_vector_clocks():
    """Demonstrate vector clocks for causality tracking."""
    print("=== Vector Clocks ===\n")

    procs = ["P1", "P2", "P3"]
    vc1 = VectorClock("P1", procs)
    vc2 = VectorClock("P2", procs)
    vc3 = VectorClock("P3", procs)

    # P1 writes x, sends to P2
    ts_a = vc1.local_event()                    # {P1:1, P2:0, P3:0}
    ts_send1 = vc1.send()                        # {P1:2, P2:0, P3:0}
    ts_b = vc2.receive(ts_send1)                  # {P1:2, P2:1, P3:0}

    # P3 writes y independently (concurrent with P1's events)
    ts_c = vc3.local_event()                    # {P1:0, P2:0, P3:1}

    # P2 sends to P3
    ts_send2 = vc2.send()                        # {P1:2, P2:2, P3:0}
    ts_d = vc3.receive(ts_send2)                  # {P1:2, P2:2, P3:2}

    print("Events and their vector timestamps:")
    print(f"  a (P1 write x): {ts_a}")
    print(f"  b (P2 receive): {ts_b}")
    print(f"  c (P3 write y): {ts_c}")
    print(f"  d (P3 receive): {ts_d}")

    print(f"\nCausality analysis:")
    pairs = [
        ("a", ts_a, "b", ts_b),
        ("a", ts_a, "c", ts_c),
        ("b", ts_b, "c", ts_c),
        ("c", ts_c, "d", ts_d),
        ("a", ts_a, "d", ts_d),
    ]

    for name1, vc_1, name2, vc_2 in pairs:
        relation = VectorClock.compare(vc_1, vc_2)
        symbol = {"before": "→", "after": "←", "concurrent": "∥", "equal": "="}[relation]
        print(f"  {name1} {symbol} {name2} ({relation})")


demonstrate_vector_clocks()
```

---

## 4. Comparing Events with Vector Clocks

### 4.1 Partial Order Visualization

```python
def visualize_partial_order():
    """Visualize the partial order defined by vector clocks."""
    print("=== Partial Order Visualization ===\n")

    events = {
        "a": {"P1": 1, "P2": 0, "P3": 0},
        "b": {"P1": 2, "P2": 0, "P3": 0},
        "c": {"P1": 0, "P2": 1, "P3": 0},
        "d": {"P1": 2, "P2": 2, "P3": 0},
        "e": {"P1": 0, "P2": 0, "P3": 1},
        "f": {"P1": 2, "P2": 2, "P3": 2},
    }

    print("Events:")
    for name, vc in events.items():
        print(f"  {name}: {vc}")

    print(f"\nPartial order (→ means happens-before, ∥ means concurrent):")
    names = sorted(events.keys())
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            relation = VectorClock.compare(events[n1], events[n2])
            if relation != "equal":
                symbol = {"before": "→", "after": "←", "concurrent": "∥"}[relation]
                print(f"  {n1} {symbol} {n2}")

    # Build Hasse diagram (direct causality, no transitivity)
    print(f"\nHasse diagram (direct causal links only):")
    for n1 in names:
        for n2 in names:
            if n1 != n2 and VectorClock.happens_before(events[n1], events[n2]):
                # Check if direct (no intermediate event)
                is_direct = True
                for n3 in names:
                    if n3 != n1 and n3 != n2:
                        if (VectorClock.happens_before(events[n1], events[n3]) and
                            VectorClock.happens_before(events[n3], events[n2])):
                            is_direct = False
                            break
                if is_direct:
                    print(f"  {n1} → {n2}")


visualize_partial_order()
```

---

## 5. Version Vectors

### 5.1 Version Vectors for Replicated Data

```python
class VersionVector:
    """
    Version vector for tracking data versions in replicated systems.

    Unlike vector clocks (which track events), version vectors
    track the version of a specific data item across replicas.
    Each replica increments its own counter when it writes.
    """

    def __init__(self):
        self.vector: Dict[str, int] = {}

    def increment(self, replica_id: str) -> 'VersionVector':
        """Increment the version for a replica (on write)."""
        new = VersionVector()
        new.vector = dict(self.vector)
        new.vector[replica_id] = new.vector.get(replica_id, 0) + 1
        return new

    def merge(self, other: 'VersionVector') -> 'VersionVector':
        """Merge two version vectors (component-wise max)."""
        new = VersionVector()
        all_keys = set(self.vector.keys()) | set(other.vector.keys())
        new.vector = {
            k: max(self.vector.get(k, 0), other.vector.get(k, 0))
            for k in all_keys
        }
        return new

    def dominates(self, other: 'VersionVector') -> bool:
        """Check if this version vector dominates (is newer than) other."""
        for key in set(self.vector.keys()) | set(other.vector.keys()):
            if self.vector.get(key, 0) < other.vector.get(key, 0):
                return False
        return self.vector != other.vector

    def concurrent_with(self, other: 'VersionVector') -> bool:
        """Check if two version vectors are concurrent (conflict)."""
        return (not self.dominates(other) and
                not other.dominates(self) and
                self.vector != other.vector)

    def __repr__(self):
        return f"VV({self.vector})"


class ReplicatedKVStore:
    """
    Replicated key-value store with version vector conflict detection.
    """

    def __init__(self, replica_id: str):
        self.replica_id = replica_id
        self.data: Dict[str, list[Tuple[str, VersionVector]]] = {}

    def put(self, key: str, value: str,
            context: Optional[VersionVector] = None) -> VersionVector:
        """
        Write a value with optional causal context.

        If context is provided, it represents the version the client read.
        The write supersedes that version. If no context, it's a blind write
        that may create a sibling (concurrent version).
        """
        current_versions = self.data.get(key, [])

        if context is not None:
            # Remove versions dominated by the context
            remaining = [
                (v, vv) for v, vv in current_versions
                if not context.dominates(vv) and context.vector != vv.vector
            ]
            new_vv = context.increment(self.replica_id)
        else:
            remaining = list(current_versions)
            # Merge all existing versions and increment
            merged = VersionVector()
            for _, vv in current_versions:
                merged = merged.merge(vv)
            new_vv = merged.increment(self.replica_id)

        remaining.append((value, new_vv))
        self.data[key] = remaining
        return new_vv

    def get(self, key: str) -> list[Tuple[str, VersionVector]]:
        """Read a key. May return multiple concurrent versions (siblings)."""
        return self.data.get(key, [])

    def sync_from(self, other: 'ReplicatedKVStore', key: str):
        """Sync a key from another replica."""
        remote_versions = other.get(key)
        local_versions = self.get(key)

        # Merge: keep all versions that are not dominated
        all_versions = local_versions + remote_versions
        merged = []
        for val, vv in all_versions:
            dominated = False
            for other_val, other_vv in all_versions:
                if other_vv.dominates(vv):
                    dominated = True
                    break
            if not dominated:
                # Avoid duplicates
                if not any(vv.vector == existing_vv.vector for _, existing_vv in merged):
                    merged.append((val, vv))

        self.data[key] = merged


def demonstrate_version_vectors():
    """Demonstrate version vectors for conflict detection."""
    print("=== Version Vectors ===\n")

    r1 = ReplicatedKVStore("R1")
    r2 = ReplicatedKVStore("R2")

    # R1 writes "x" = "alice"
    vv1 = r1.put("x", "alice")
    print(f"R1 writes x='alice': {vv1}")

    # Sync R1 → R2
    r2.sync_from(r1, "x")
    print(f"R2 after sync: {r2.get('x')}")

    # R1 updates (with context)
    vv2 = r1.put("x", "alice_v2", context=vv1)
    print(f"\nR1 writes x='alice_v2': {vv2}")

    # R2 updates CONCURRENTLY (with old context!)
    vv3 = r2.put("x", "bob", context=vv1)
    print(f"R2 writes x='bob': {vv3}")

    # Sync — should detect conflict
    r1.sync_from(r2, "x")
    versions = r1.get("x")
    print(f"\nR1 after sync from R2:")
    for val, vv in versions:
        print(f"  value='{val}', version={vv}")

    if len(versions) > 1:
        print(f"\n  CONFLICT DETECTED! {len(versions)} concurrent versions")
        print(f"  Application must resolve: 'alice_v2' vs 'bob'")


demonstrate_version_vectors()
```

---

## 6. Conflict Detection and Resolution

### 6.1 Resolution Strategies

```python
class ConflictResolver:
    """
    Conflict resolution strategies for concurrent writes.
    """

    @staticmethod
    def last_writer_wins(versions: list[Tuple[str, VersionVector, float]]) -> str:
        """
        Last-Writer-Wins (LWW): use wall-clock timestamp to pick winner.

        Simple but can lose data. Used by Cassandra.
        """
        return max(versions, key=lambda v: v[2])[0]

    @staticmethod
    def merge_values(versions: list[Tuple[str, VersionVector]]) -> str:
        """
        Application-specific merge. Example: merge JSON objects.
        """
        merged = {}
        for val, _ in versions:
            try:
                obj = json.loads(val) if isinstance(val, str) else val
                if isinstance(obj, dict):
                    merged.update(obj)
            except (json.JSONDecodeError, TypeError):
                pass
        return json.dumps(merged) if merged else versions[0][0]

    @staticmethod
    def union_set(versions: list[Tuple[set, VersionVector]]) -> set:
        """
        Set union for concurrent additions.
        Used in OR-Set CRDTs.
        """
        result = set()
        for val, _ in versions:
            result |= val
        return result

    @staticmethod
    def client_resolve(versions: list[Tuple[str, VersionVector]]) -> str:
        """
        Return all versions to the client for manual resolution.
        Used by Riak (returns siblings).
        """
        # In practice, return all versions to the client
        # Client chooses or merges
        return versions  # Client must resolve


def demonstrate_conflict_resolution():
    """Demonstrate different conflict resolution strategies."""
    print("=== Conflict Resolution Strategies ===\n")

    import json

    # Scenario: Two concurrent writes
    versions = [
        ('{"name": "Alice", "age": 30}', VersionVector(), time.time()),
        ('{"name": "Alice", "email": "a@b.com"}', VersionVector(), time.time() + 0.001),
    ]

    print("Concurrent versions:")
    for val, _, ts in versions:
        print(f"  {val}")

    # LWW
    winner = ConflictResolver.last_writer_wins(versions)
    print(f"\nLWW winner: {winner}")
    print(f"  Problem: Lost 'age: 30'")

    # Merge
    merge_input = [(v, vv) for v, vv, _ in versions]
    merged = ConflictResolver.merge_values(merge_input)
    print(f"\nMerged: {merged}")
    print(f"  All fields preserved")

    # Set union for tags
    set_versions = [
        ({"python", "rust"}, VersionVector()),
        ({"python", "go"}, VersionVector()),
    ]
    union = ConflictResolver.union_set(set_versions)
    print(f"\nSet union: {union}")


demonstrate_conflict_resolution()
```

---

## 7. Dotted Version Vectors

### 7.1 Why Plain Version Vectors Are Not Enough

```python
class DottedVersionVector:
    """
    Dotted version vector for accurate causality in KV stores.

    A regular version vector cannot distinguish between:
    - "This client has seen version X" (causal context)
    - "This is version X" (the dot)

    A dotted VV = (dot, version_vector) where:
    - dot = (replica, counter) identifies this specific write
    - version_vector = causal context of the write
    """

    def __init__(self, dot: Optional[Tuple[str, int]] = None,
                 vv: Optional[Dict[str, int]] = None):
        self.dot = dot  # (replica_id, counter) — identifies this event
        self.vv = vv or {}  # Causal context

    def __repr__(self):
        return f"DVV(dot={self.dot}, vv={self.vv})"

    def dominates(self, other: 'DottedVersionVector') -> bool:
        """Check if this DVV causally dominates another."""
        if other.dot:
            replica, counter = other.dot
            if self.vv.get(replica, 0) >= counter:
                return True
        # Also check VV dominance
        for key, val in other.vv.items():
            if self.vv.get(key, 0) < val:
                return False
        return self.vv != other.vv or (self.dot is not None and other.dot is None)

    def merge(self, other: 'DottedVersionVector') -> 'DottedVersionVector':
        """Merge two DVVs."""
        merged_vv = {}
        all_keys = set(self.vv.keys()) | set(other.vv.keys())
        for key in all_keys:
            merged_vv[key] = max(self.vv.get(key, 0), other.vv.get(key, 0))

        # Absorb dots into VV
        if self.dot:
            r, c = self.dot
            merged_vv[r] = max(merged_vv.get(r, 0), c)
        if other.dot:
            r, c = other.dot
            merged_vv[r] = max(merged_vv.get(r, 0), c)

        return DottedVersionVector(dot=None, vv=merged_vv)


def demonstrate_dotted_vv():
    """Demonstrate dotted version vectors."""
    print("=== Dotted Version Vectors ===\n")

    # Write 1: Client writes "x=1" to replica R1
    dvv1 = DottedVersionVector(dot=("R1", 1), vv={})
    print(f"Write 1 (x=1 at R1): {dvv1}")

    # Write 2: Client reads version 1, writes "x=2" to replica R2
    dvv2 = DottedVersionVector(dot=("R2", 1), vv={"R1": 1})
    print(f"Write 2 (x=2 at R2, after reading v1): {dvv2}")

    # dvv2 dominates dvv1? Yes, because dvv2.vv[R1] >= dvv1.dot[1]
    print(f"\nWrite 2 dominates Write 1: {dvv2.dominates(dvv1)}")

    # Concurrent write: another client writes "x=3" at R3 without reading v1
    dvv3 = DottedVersionVector(dot=("R3", 1), vv={})
    print(f"Write 3 (x=3 at R3, blind): {dvv3}")
    print(f"Write 2 dominates Write 3: {dvv2.dominates(dvv3)}")
    print(f"Write 3 dominates Write 2: {dvv3.dominates(dvv2)}")
    print(f"Concurrent: both are valid (conflict)")


demonstrate_dotted_vv()
```

---

## 8. Hybrid Logical Clocks

### 8.1 HLC: Combining Physical and Logical Time

```python
class HybridLogicalClock:
    """
    Hybrid Logical Clock (HLC) — Kulkarni et al., 2014.

    Combines physical time with logical counters to provide:
    - Timestamps that are close to real time (within clock skew)
    - Causal ordering guarantee (like Lamport clocks)
    - Bounded size (unlike vector clocks which grow with N)

    HLC timestamp = (physical_time, logical_counter, process_id)
    """

    def __init__(self, process_id: str, max_clock_skew_ms: float = 500):
        self.process_id = process_id
        self.max_skew = max_clock_skew_ms / 1000.0
        self.l: float = 0.0  # Physical component
        self.c: int = 0       # Logical component

    def _physical_time(self) -> float:
        return time.time()

    def local_event(self) -> Tuple[float, int]:
        """Record a local or send event."""
        pt = self._physical_time()
        old_l = self.l

        self.l = max(old_l, pt)
        if self.l == old_l:
            self.c += 1
        else:
            self.c = 0

        return (self.l, self.c)

    def receive(self, remote_l: float, remote_c: int) -> Tuple[float, int]:
        """Process a receive event."""
        pt = self._physical_time()
        old_l = self.l

        self.l = max(old_l, remote_l, pt)

        if self.l == old_l == remote_l:
            self.c = max(self.c, remote_c) + 1
        elif self.l == old_l:
            self.c += 1
        elif self.l == remote_l:
            self.c = remote_c + 1
        else:
            self.c = 0

        return (self.l, self.c)

    def timestamp(self) -> Tuple[float, int, str]:
        """Get current HLC timestamp."""
        return (self.l, self.c, self.process_id)

    @staticmethod
    def compare(ts1: Tuple[float, int, str],
                ts2: Tuple[float, int, str]) -> int:
        """Compare two HLC timestamps. Returns -1, 0, or 1."""
        if ts1[0] != ts2[0]:
            return -1 if ts1[0] < ts2[0] else 1
        if ts1[1] != ts2[1]:
            return -1 if ts1[1] < ts2[1] else 1
        if ts1[2] != ts2[2]:
            return -1 if ts1[2] < ts2[2] else 1
        return 0


def demonstrate_hlc():
    """Demonstrate Hybrid Logical Clocks."""
    print("=== Hybrid Logical Clocks ===\n")

    hlc1 = HybridLogicalClock("P1")
    hlc2 = HybridLogicalClock("P2")

    # Local events
    ts1 = hlc1.local_event()
    ts2 = hlc1.local_event()  # Same physical time → c increments
    print(f"P1 event 1: l={ts1[0]:.6f}, c={ts1[1]}")
    print(f"P1 event 2: l={ts2[0]:.6f}, c={ts2[1]}")

    # Send to P2
    ts_send = hlc1.local_event()
    ts_recv = hlc2.receive(ts_send[0], ts_send[1])
    print(f"\nP1 send:    l={ts_send[0]:.6f}, c={ts_send[1]}")
    print(f"P2 receive: l={ts_recv[0]:.6f}, c={ts_recv[1]}")

    print(f"\nAdvantages over vector clocks:")
    print(f"  - Fixed size (not proportional to number of processes)")
    print(f"  - Close to physical time (useful for TTL, ordering)")
    print(f"  - Still provides causal ordering")
    print(f"\nLimitation:")
    print(f"  - Cannot detect concurrency (total order, like Lamport)")
    print(f"  - Depends on bounded clock skew ({hlc1.max_skew*1000:.0f}ms)")


demonstrate_hlc()
```

---

## 9. Real-World Applications

### 9.1 Comparison

```python
def compare_clock_mechanisms():
    """Compare logical clock mechanisms used in real systems."""
    print("=== Logical Clocks in Real Systems ===\n")

    systems = [
        {"system": "Amazon DynamoDB", "clock": "Vector clocks → LWW",
         "notes": "Originally VV, switched to LWW for simplicity"},
        {"system": "Riak", "clock": "Dotted version vectors",
         "notes": "Accurate sibling detection, client-side resolution"},
        {"system": "CockroachDB", "clock": "Hybrid Logical Clock",
         "notes": "Causal ordering + real-time bounds for serializable isolation"},
        {"system": "Spanner", "clock": "TrueTime (physical)",
         "notes": "GPS + atomic clocks, bounded uncertainty interval"},
        {"system": "Cassandra", "clock": "Last-Writer-Wins (wall clock)",
         "notes": "Simple but can lose writes; depends on NTP"},
        {"system": "Git", "clock": "DAG (not clocks)",
         "notes": "Content-addressable; merge commits resolve concurrent edits"},
    ]

    for s in systems:
        print(f"  {s['system']:20s} | {s['clock']:30s} | {s['notes']}")


compare_clock_mechanisms()
```

---

## 10. Summary and Key Takeaways

### Clock Mechanism Selection Guide

> **CHOOSING A LOGICAL CLOCK**
>
> Need to detect concurrency? → Vector Clocks or Dotted VVs
> Need total order + real time? → Hybrid Logical Clocks
> Need simplicity + total order? → Lamport Clocks
> Need exact physical time? → TrueTime (requires hardware)
> Can tolerate data loss? → Last-Writer-Wins

### Key Principles

1. **Lamport clocks provide total order but cannot detect concurrency**: If L(a) < L(b), a may or may not have caused b.
2. **Vector clocks capture causality exactly**: a → b iff VC(a) < VC(b); a ∥ b iff neither dominates.
3. **Vector clocks grow with N**: Impractical for very large systems; version vectors or HLC are alternatives.
4. **Conflict detection is separate from conflict resolution**: Clocks detect conflicts; the application decides how to resolve them.
5. **HLC is the practical sweet spot**: Bounded size, close to real time, causal ordering.

---

## 11. Practice Problems

### Problem 1: Vector Clock Computation

Three processes exchange messages. Compute vector timestamps for all events:
- P1: send to P2, local, send to P3
- P2: receive from P1, send to P3, local
- P3: local, receive from P1, receive from P2

### Problem 2: Conflict Detection

Given these version vectors for key "x": VV_A = {R1:3, R2:1}, VV_B = {R1:2, R2:2}. Are they concurrent? If so, design a merge function for a shopping cart (set of items).

### Problem 3: HLC Bounds

Prove that HLC timestamps are always within max_clock_skew of real time. What happens if a node's clock jumps backward by more than max_clock_skew?

### Problem 4: Implementation Challenge

Implement a replicated key-value store with:
- Dotted version vectors for causality tracking
- Configurable conflict resolution (LWW, client-resolve, CRDT merge)
- Anti-entropy protocol to sync replicas
- Read-repair on GET

### Problem 5: Scalability Analysis

Vector clocks have O(N) size where N is the number of processes. Design a scheme to prune vector clock entries for processes that have been permanently removed. What safety guarantees can you maintain?

---

## 12. References

1. Lamport, L. (1978). "Time, Clocks, and the Ordering of Events in a Distributed System." *Communications of the ACM*, 21(7).
2. Fidge, C. (1988). "Timestamps in Message-Passing Systems That Preserve the Partial Ordering." *Australian Computer Science Communications*.
3. Mattern, F. (1989). "Virtual Time and Global States of Distributed Systems." *Parallel and Distributed Algorithms*.
4. Preguica, N. et al. (2012). "Dotted Version Vectors: Logical Clocks for Optimistic Replication." arXiv:1011.5808.
5. Kulkarni, S. et al. (2014). "Logical Physical Clocks and Consistent Snapshots in Globally Distributed Databases." *OPODIS*.
6. Corbett, J. et al. (2013). "Spanner: Google's Globally-Distributed Database." *ACM TOCS*.
7. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Ch. 8. O'Reilly Media.

---

[Next: Lesson 26 — Distributed Testing](./26_Distributed_Testing.md)
