"""
Eventual Consistency

Demonstrates:
- Vector clocks for causality tracking
- Last-Writer-Wins (LWW) conflict resolution
- Read repair
- Anti-entropy (Merkle tree concept)

Theory:
- In distributed systems, CAP theorem forces a trade-off between
  consistency and availability during network partitions.
- Eventual consistency: replicas may diverge temporarily but
  converge when communication resumes.
- Vector clocks: each node maintains a vector of logical timestamps.
  Enables detecting concurrent (conflicting) writes.
- LWW: conflicts resolved by wall-clock timestamp. Simple but
  can lose writes.
- Read repair: on read, detect stale replicas and update them.

Adapted from System Design Lesson 10.
"""

from collections import defaultdict
from copy import deepcopy
from typing import Any


# ── Vector Clock ───────────────────────────────────────────────────────

# Why: Vector clocks capture causal ordering between events across nodes. Unlike
# wall-clock timestamps, they can distinguish "A happened before B" from "A and B
# are concurrent" — critical for detecting conflicting writes in distributed stores.
class VectorClock:
    """Vector clock for tracking causality."""

    def __init__(self):
        self.clock: dict[str, int] = defaultdict(int)

    def increment(self, node: str) -> None:
        self.clock[node] += 1

    def merge(self, other: "VectorClock") -> None:
        for node, ts in other.clock.items():
            self.clock[node] = max(self.clock[node], ts)

    def __le__(self, other: "VectorClock") -> bool:
        """self happens-before or equals other."""
        for node, ts in self.clock.items():
            if ts > other.clock.get(node, 0):
                return False
        return True

    def __lt__(self, other: "VectorClock") -> bool:
        return self <= other and self.clock != other.clock

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        all_nodes = set(self.clock) | set(other.clock)
        return all(self.clock.get(n, 0) == other.clock.get(n, 0)
                   for n in all_nodes)

    def concurrent(self, other: "VectorClock") -> bool:
        """Neither happens-before the other."""
        # Why: Concurrent events are the problematic case — they represent
        # conflicting writes that no ordering can resolve automatically.
        # The application must choose a resolution strategy (LWW, merge, ask user).
        return not (self <= other) and not (other <= self)

    def copy(self) -> "VectorClock":
        vc = VectorClock()
        vc.clock = dict(self.clock)
        return vc

    def __repr__(self) -> str:
        items = sorted(self.clock.items())
        return "{" + ", ".join(f"{k}:{v}" for k, v in items) + "}"


# ── Versioned Value ────────────────────────────────────────────────────

class VersionedValue:
    """Value with vector clock and wall-clock timestamp."""

    def __init__(self, value: Any, vc: VectorClock, timestamp: float = 0.0):
        self.value = value
        self.vc = vc
        self.timestamp = timestamp

    def __repr__(self) -> str:
        return f"({self.value}, vc={self.vc}, t={self.timestamp})"


# ── Replica Node ───────────────────────────────────────────────────────

# Why: Each replica stores a list of versioned values (siblings) per key, not
# just one. This preserves all concurrent writes until the application explicitly
# resolves the conflict — similar to how Amazon Dynamo handles shopping cart merges.
class ReplicaNode:
    """A single replica in an eventually consistent system."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.store: dict[str, list[VersionedValue]] = {}
        self.clock = VectorClock()

    def write(self, key: str, value: Any, timestamp: float = 0.0) -> VersionedValue:
        self.clock.increment(self.node_id)
        vv = VersionedValue(value, self.clock.copy(), timestamp)

        if key not in self.store:
            self.store[key] = [vv]
        else:
            # Remove versions that this write supersedes
            self.store[key] = [
                existing for existing in self.store[key]
                if not existing.vc <= vv.vc
            ]
            self.store[key].append(vv)
        return vv

    def read(self, key: str) -> list[VersionedValue]:
        return self.store.get(key, [])

    def receive_write(self, key: str, vv: VersionedValue) -> str:
        """Receive a replicated write. Returns resolution status."""
        # Why: Merging the incoming vector clock ensures this node's clock
        # reflects all events it has observed, maintaining the causal ordering
        # guarantee across the cluster.
        self.clock.merge(vv.vc)

        if key not in self.store:
            self.store[key] = [vv]
            return "NEW"

        existing = self.store[key]

        # Check if this write supersedes all existing
        if all(e.vc <= vv.vc for e in existing):
            self.store[key] = [vv]
            return "SUPERSEDE"

        # Check if any existing supersedes this write
        if any(vv.vc <= e.vc for e in existing):
            return "STALE"

        # Concurrent: keep both (sibling)
        self.store[key].append(vv)
        return "CONFLICT"


# ── LWW Register ───────────────────────────────────────────────────────

# Why: LWW is the simplest conflict resolution strategy — it always converges
# and needs no application logic. The fundamental weakness is silent data loss:
# concurrent writes are discarded based solely on timestamp ordering.
class LWWRegister:
    """Last-Writer-Wins register for conflict resolution."""

    def __init__(self):
        self.value: Any = None
        self.timestamp: float = 0.0

    def write(self, value: Any, timestamp: float) -> bool:
        if timestamp > self.timestamp:
            self.value = value
            self.timestamp = timestamp
            return True
        return False

    def merge(self, other: "LWWRegister") -> None:
        if other.timestamp > self.timestamp:
            self.value = other.value
            self.timestamp = other.timestamp


# ── Demos ──────────────────────────────────────────────────────────────

def demo_vector_clocks():
    print("=" * 60)
    print("VECTOR CLOCKS")
    print("=" * 60)

    # Three nodes: A, B, C
    vc_a = VectorClock()
    vc_b = VectorClock()

    # A does an event
    vc_a.increment("A")
    print(f"\n  A does event:    A={vc_a}")

    # A sends to B, B does event
    vc_b.merge(vc_a)
    vc_b.increment("B")
    print(f"  A→B, B event:    B={vc_b}")

    # A does another event
    vc_a.increment("A")
    print(f"  A does event:    A={vc_a}")

    # Causality check
    print(f"\n  Causality:")
    print(f"    A < B?  {vc_a < vc_b}  (A happens-before B? No)")
    print(f"    B < A?  {vc_b < vc_a}  (B happens-before A? No)")
    print(f"    Concurrent? {vc_a.concurrent(vc_b)}  (Yes, independent events)")

    # After merge
    vc_a.merge(vc_b)
    vc_a.increment("A")
    print(f"\n  A merges B, event: A={vc_a}")
    print(f"    B < A?  {vc_b < vc_a}  (Now B happens-before A)")


def demo_conflict_detection():
    print("\n" + "=" * 60)
    print("CONFLICT DETECTION WITH REPLICAS")
    print("=" * 60)

    node_a = ReplicaNode("A")
    node_b = ReplicaNode("B")

    # Both write same key concurrently
    print(f"\n  Concurrent writes to 'user:1':")
    vv_a = node_a.write("user:1", "Alice", timestamp=1.0)
    print(f"    Node A writes 'Alice': {vv_a}")

    vv_b = node_b.write("user:1", "Bob", timestamp=1.1)
    print(f"    Node B writes 'Bob':   {vv_b}")

    # Replicate: A receives B's write
    status_a = node_a.receive_write("user:1", vv_b)
    print(f"\n  A receives B's write: {status_a}")
    values_a = node_a.read("user:1")
    print(f"    A's values for user:1: {values_a}")
    print(f"    → {len(values_a)} siblings (conflict!)")

    # Replicate: B receives A's write
    status_b = node_b.receive_write("user:1", vv_a)
    print(f"\n  B receives A's write: {status_b}")
    values_b = node_b.read("user:1")
    print(f"    B's values for user:1: {values_b}")

    # Resolve conflict with application logic
    print(f"\n  Application resolves: pick latest timestamp")
    resolved = max(values_a, key=lambda v: v.timestamp)
    print(f"    Winner: {resolved.value} (t={resolved.timestamp})")


def demo_lww():
    print("\n" + "=" * 60)
    print("LAST-WRITER-WINS (LWW) REGISTER")
    print("=" * 60)

    reg_a = LWWRegister()
    reg_b = LWWRegister()

    # Simulate writes with timestamps
    writes = [
        ("A", "value-1", 1.0),
        ("B", "value-2", 1.5),
        ("A", "value-3", 1.3),
        ("B", "value-4", 2.0),
    ]

    print(f"\n  Writes (wall clock):")
    for node, value, ts in writes:
        reg = reg_a if node == "A" else reg_b
        accepted = reg.write(value, ts)
        print(f"    Node {node}: write '{value}' at t={ts} → "
              f"{'accepted' if accepted else 'rejected (older)'}")

    print(f"\n  Before merge:")
    print(f"    Reg A: {reg_a.value} (t={reg_a.timestamp})")
    print(f"    Reg B: {reg_b.value} (t={reg_b.timestamp})")

    reg_a.merge(reg_b)
    reg_b.merge(reg_a)
    print(f"\n  After merge:")
    print(f"    Reg A: {reg_a.value} (t={reg_a.timestamp})")
    print(f"    Reg B: {reg_b.value} (t={reg_b.timestamp})")
    print(f"    Converged: {reg_a.value == reg_b.value}")

    print(f"\n  ⚠ LWW silently drops 'value-1' and 'value-3'!")
    print(f"  Pros: Simple, always converges")
    print(f"  Cons: Lost writes, depends on clock synchronization")


def demo_read_repair():
    print("\n" + "=" * 60)
    print("READ REPAIR")
    print("=" * 60)

    nodes = [ReplicaNode(f"N{i}") for i in range(3)]

    # Write to N0 only (simulating partial replication)
    vv = nodes[0].write("key:1", "latest-value", timestamp=1.0)
    print(f"\n  Write 'latest-value' to N0 only: {vv}")
    # Stale value on N1
    stale_vv = nodes[1].write("key:1", "stale-value", timestamp=0.5)
    # N2 has nothing

    print(f"\n  State before read repair:")
    for n in nodes:
        vals = n.read("key:1")
        print(f"    {n.node_id}: {vals if vals else '(empty)'}")

    # Read from all replicas (quorum read)
    print(f"\n  Quorum read from all 3 replicas:")
    all_versions = []
    for n in nodes:
        versions = n.read("key:1")
        all_versions.extend([(n, v) for v in versions])

    if all_versions:
        # Find the latest version
        latest = max(all_versions, key=lambda x: x[1].timestamp)
        print(f"    Latest: {latest[1]} from {latest[0].node_id}")

        # Repair stale replicas
        print(f"\n  Read repair: push latest to all replicas")
        for n in nodes:
            if n.node_id != latest[0].node_id:
                status = n.receive_write("key:1", latest[1])
                print(f"    → {n.node_id}: {status}")

    print(f"\n  State after read repair:")
    for n in nodes:
        vals = n.read("key:1")
        print(f"    {n.node_id}: {[v.value for v in vals]}")


if __name__ == "__main__":
    demo_vector_clocks()
    demo_conflict_detection()
    demo_lww()
    demo_read_repair()
