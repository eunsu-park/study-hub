"""
Distributed Systems Concepts — CAP Theorem, Vector Clocks, Partitions

Demonstrates:
- CAP theorem trade-offs (CP vs AP systems)
- Vector clocks for causal ordering
- Network partition simulation
- Split-brain detection

Theory:
- CAP theorem: a distributed system can provide at most two of three
  guarantees: Consistency, Availability, Partition tolerance.
  Since network partitions are unavoidable, the real choice is CP vs AP.
- CP systems (e.g., ZooKeeper, HBase): refuse requests during partitions
  to maintain consistency.
- AP systems (e.g., Cassandra, DynamoDB): continue serving requests during
  partitions, accepting temporary inconsistency.
- Vector clocks: each node maintains a vector of logical timestamps.
  Enables detection of causal ordering and concurrent writes without
  a global clock.

Adapted from System Design Lesson 15.
"""

from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy


# ── Vector Clocks ─────────────────────────────────────────────────────

# Why: In a distributed system, wall clocks are unreliable (clock skew, NTP
# drift). Vector clocks provide a logical ordering that captures causality:
# if event A's vector clock is strictly less than B's, then A happened before B.
# If neither dominates, the events are concurrent (potential conflict).
class VectorClock:
    """Vector clock for causal ordering in distributed systems."""

    def __init__(self, node_id: str, initial: dict[str, int] | None = None):
        self.node_id = node_id
        self.clock: dict[str, int] = initial or {}

    def increment(self) -> "VectorClock":
        """Increment this node's counter (local event)."""
        self.clock[self.node_id] = self.clock.get(self.node_id, 0) + 1
        return self

    def merge(self, other: "VectorClock") -> "VectorClock":
        """Merge with another vector clock (receive event)."""
        all_nodes = set(self.clock) | set(other.clock)
        for node in all_nodes:
            self.clock[node] = max(
                self.clock.get(node, 0), other.clock.get(node, 0)
            )
        self.increment()
        return self

    def happens_before(self, other: "VectorClock") -> bool:
        """Return True if self causally precedes other."""
        at_least_one_less = False
        for node in set(self.clock) | set(other.clock):
            s = self.clock.get(node, 0)
            o = other.clock.get(node, 0)
            if s > o:
                return False
            if s < o:
                at_least_one_less = True
        return at_least_one_less

    def is_concurrent(self, other: "VectorClock") -> bool:
        """Return True if neither clock dominates (concurrent events)."""
        return not self.happens_before(other) and not other.happens_before(self)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}:{v}" for k, v in sorted(self.clock.items()))
        return f"VC({items})"


# ── CAP Theorem Simulation ───────────────────────────────────────────

class CAPMode(Enum):
    CP = "CP"  # Consistency + Partition tolerance
    AP = "AP"  # Availability + Partition tolerance


@dataclass
class DistributedNode:
    """A node in a distributed system."""
    node_id: str
    data: dict[str, str] = field(default_factory=dict)
    vclock: VectorClock = field(default=None)
    reachable: set[str] = field(default_factory=set)
    alive: bool = True

    def __post_init__(self):
        if self.vclock is None:
            self.vclock = VectorClock(self.node_id)


# Why: This simulation shows the CAP trade-off in action. During a network
# partition, a CP system returns errors (preserving consistency) while an AP
# system returns potentially stale data (preserving availability). Neither
# is universally better — the choice depends on business requirements.
class CAPCluster:
    """Distributed cluster demonstrating CAP trade-offs."""

    def __init__(self, node_ids: list[str], mode: CAPMode = CAPMode.CP):
        self.mode = mode
        self.nodes: dict[str, DistributedNode] = {}
        for nid in node_ids:
            node = DistributedNode(nid)
            node.reachable = set(node_ids) - {nid}
            self.nodes[nid] = node
        self.event_log: list[str] = []

    @property
    def majority(self) -> int:
        return len(self.nodes) // 2 + 1

    def write(self, node_id: str, key: str, value: str) -> dict:
        """Write to a node. Behavior depends on CAP mode."""
        node = self.nodes[node_id]
        if not node.alive:
            return {"status": "error", "reason": "node down"}

        node.vclock.increment()

        if self.mode == CAPMode.CP:
            # Must replicate to majority before acknowledging
            acks = 1  # self
            for peer_id in node.reachable:
                peer = self.nodes[peer_id]
                if peer.alive:
                    peer.data[key] = value
                    peer.vclock.merge(deepcopy(node.vclock))
                    acks += 1

            if acks >= self.majority:
                node.data[key] = value
                self.event_log.append(
                    f"  [{node_id}] CP WRITE {key}={value} "
                    f"({acks}/{len(self.nodes)} acks) — COMMITTED"
                )
                return {"status": "ok", "acks": acks}
            else:
                self.event_log.append(
                    f"  [{node_id}] CP WRITE {key}={value} "
                    f"({acks}/{len(self.nodes)} acks) — REJECTED (no majority)"
                )
                return {"status": "error", "reason": "no majority", "acks": acks}

        else:  # AP mode
            node.data[key] = value
            replicated = 0
            for peer_id in node.reachable:
                peer = self.nodes[peer_id]
                if peer.alive:
                    peer.data[key] = value
                    peer.vclock.merge(deepcopy(node.vclock))
                    replicated += 1

            self.event_log.append(
                f"  [{node_id}] AP WRITE {key}={value} "
                f"(replicated to {replicated} peers) — ACCEPTED"
            )
            return {"status": "ok", "replicated": replicated}

    def read(self, node_id: str, key: str) -> dict:
        """Read from a node."""
        node = self.nodes[node_id]
        if not node.alive:
            return {"status": "error", "reason": "node down"}

        if self.mode == CAPMode.CP:
            # Must confirm with majority for consistent read
            values: dict[str, int] = {}
            for nid, n in self.nodes.items():
                if n.alive and (nid == node_id or nid in node.reachable):
                    v = n.data.get(key, "NOT_FOUND")
                    values[v] = values.get(v, 0) + 1

            if sum(values.values()) < self.majority:
                return {"status": "error", "reason": "cannot reach majority"}

            # Return majority value
            majority_value = max(values, key=values.get)
            return {"status": "ok", "value": majority_value}

        else:  # AP
            value = node.data.get(key, "NOT_FOUND")
            return {"status": "ok", "value": value, "warning": "may be stale"}

    def partition(self, group_a: list[str], group_b: list[str]) -> None:
        """Create a network partition between two groups."""
        set_a, set_b = set(group_a), set(group_b)
        for nid in group_a:
            self.nodes[nid].reachable -= set_b
        for nid in group_b:
            self.nodes[nid].reachable -= set_a
        self.event_log.append(
            f"  PARTITION: {group_a} <-/-> {group_b}"
        )

    def heal_partition(self) -> None:
        """Heal network partition — all nodes can reach each other."""
        all_ids = set(self.nodes.keys())
        for node in self.nodes.values():
            node.reachable = all_ids - {node.node_id}
        self.event_log.append("  PARTITION HEALED: all nodes connected")

    def print_state(self) -> None:
        print(f"\n    {'Node':<8} {'Data':>30} {'Reachable':>25} {'VClock':>20}")
        print(f"    {'-'*8} {'-'*30} {'-'*25} {'-'*20}")
        for nid in sorted(self.nodes):
            n = self.nodes[nid]
            data_str = str(dict(n.data)) if n.data else "{}"
            reach = ",".join(sorted(n.reachable)) or "ISOLATED"
            print(f"    {nid:<8} {data_str:>30} {reach:>25} {n.vclock!r:>20}")


# ── Split-Brain Detector ─────────────────────────────────────────────

# Why: Split-brain occurs when a partition causes two groups to both believe
# they are the "active" side, accepting writes independently. Detecting this
# requires checking whether any group has a majority — if neither does,
# both groups should refuse writes (CP) or accept and reconcile later (AP).
def detect_split_brain(cluster: CAPCluster) -> list[set[str]]:
    """Detect connected components (potential split-brain groups)."""
    visited: set[str] = set()
    components: list[set[str]] = []

    for nid in cluster.nodes:
        if nid in visited:
            continue
        component: set[str] = set()
        stack = [nid]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            for peer in cluster.nodes[current].reachable:
                if peer not in visited:
                    stack.append(peer)
        components.append(component)

    return components


# ── Demos ─────────────────────────────────────────────────────────────

def demo_vector_clocks():
    print("=" * 60)
    print("VECTOR CLOCKS — CAUSAL ORDERING")
    print("=" * 60)

    vc_a = VectorClock("A")
    vc_b = VectorClock("B")
    vc_c = VectorClock("C")

    # A does local event
    vc_a.increment()
    print(f"\n  A local event:     A={vc_a}")

    # A sends to B
    vc_b.merge(deepcopy(vc_a))
    print(f"  A → B (message):   B={vc_b}")

    # B does local event
    vc_b.increment()
    print(f"  B local event:     B={vc_b}")

    # C does local event (concurrent with A and B)
    vc_c.increment()
    print(f"  C local event:     C={vc_c}")

    # Check ordering
    print(f"\n  Ordering analysis:")
    print(f"    A happens-before B? {vc_a.happens_before(vc_b)} "
          f"(A caused B via message)")
    print(f"    B happens-before A? {vc_b.happens_before(vc_a)} "
          f"(B did not cause A)")
    print(f"    A concurrent with C? {vc_a.is_concurrent(vc_c)} "
          f"(no communication)")
    print(f"    B concurrent with C? {vc_b.is_concurrent(vc_c)} "
          f"(no communication)")

    # Merge B and C (conflict detection)
    vc_b_copy = VectorClock("B", dict(vc_b.clock))
    vc_c_copy = VectorClock("C", dict(vc_c.clock))
    print(f"\n  Before merge: B={vc_b_copy}, C={vc_c_copy}")
    print(f"  Concurrent? {vc_b_copy.is_concurrent(vc_c_copy)} → CONFLICT!")
    vc_b.merge(deepcopy(vc_c))
    print(f"  After B merges C: B={vc_b}")


def demo_cap_cp():
    print("\n" + "=" * 60)
    print("CAP THEOREM — CP SYSTEM (Consistency + Partition tolerance)")
    print("=" * 60)

    cluster = CAPCluster(["N1", "N2", "N3", "N4", "N5"], CAPMode.CP)

    # Normal write
    print(f"\n  --- Normal operation (no partition) ---")
    cluster.write("N1", "x", "100")
    for msg in cluster.event_log:
        print(msg)
    cluster.event_log.clear()

    # Create partition: [N1, N2] vs [N3, N4, N5]
    print(f"\n  --- Network partition ---")
    cluster.partition(["N1", "N2"], ["N3", "N4", "N5"])
    for msg in cluster.event_log:
        print(msg)
    cluster.event_log.clear()

    # Write from minority side — should FAIL
    print(f"\n  Write from minority side (N1):")
    result = cluster.write("N1", "y", "200")
    for msg in cluster.event_log:
        print(msg)
    cluster.event_log.clear()
    print(f"  Result: {result}")

    # Write from majority side — should SUCCEED
    print(f"\n  Write from majority side (N3):")
    result = cluster.write("N3", "y", "300")
    for msg in cluster.event_log:
        print(msg)
    cluster.event_log.clear()
    print(f"  Result: {result}")

    cluster.print_state()

    print(f"\n  CP trade-off: minority partition becomes UNAVAILABLE")
    print(f"  to preserve CONSISTENCY.")


def demo_cap_ap():
    print("\n" + "=" * 60)
    print("CAP THEOREM — AP SYSTEM (Availability + Partition tolerance)")
    print("=" * 60)

    cluster = CAPCluster(["N1", "N2", "N3", "N4", "N5"], CAPMode.AP)

    # Normal write
    cluster.write("N1", "x", "100")
    cluster.event_log.clear()

    # Create partition
    print(f"\n  --- Network partition ---")
    cluster.partition(["N1", "N2"], ["N3", "N4", "N5"])
    for msg in cluster.event_log:
        print(msg)
    cluster.event_log.clear()

    # Both sides accept writes
    print(f"\n  Write from partition A (N1):")
    cluster.write("N1", "counter", "10")
    for msg in cluster.event_log:
        print(msg)
    cluster.event_log.clear()

    print(f"\n  Write from partition B (N3) — SAME KEY, different value:")
    cluster.write("N3", "counter", "20")
    for msg in cluster.event_log:
        print(msg)
    cluster.event_log.clear()

    cluster.print_state()

    # Read from different partitions
    print(f"\n  Reading 'counter' from different partitions:")
    for nid in ["N1", "N3"]:
        result = cluster.read(nid, "counter")
        print(f"    {nid}: {result}")

    print(f"\n  AP trade-off: both sides AVAILABLE, but data is INCONSISTENT.")
    print(f"  Requires conflict resolution after partition heals.")


def demo_split_brain():
    print("\n" + "=" * 60)
    print("SPLIT-BRAIN DETECTION")
    print("=" * 60)

    cluster = CAPCluster(["N1", "N2", "N3", "N4", "N5"], CAPMode.CP)

    print(f"\n  --- Before partition ---")
    components = detect_split_brain(cluster)
    print(f"  Connected components: {[sorted(c) for c in components]}")
    print(f"  Split brain? {'YES' if len(components) > 1 else 'No'}")

    # Create partition
    cluster.partition(["N1", "N2"], ["N3", "N4", "N5"])

    print(f"\n  --- After partition [N1,N2] vs [N3,N4,N5] ---")
    components = detect_split_brain(cluster)
    print(f"  Connected components: {[sorted(c) for c in components]}")
    print(f"  Split brain? {'YES' if len(components) > 1 else 'No'}")

    for comp in components:
        has_majority = len(comp) >= cluster.majority
        print(f"    {sorted(comp)}: {len(comp)} nodes "
              f"{'(HAS MAJORITY)' if has_majority else '(minority, should refuse writes)'}")

    # Heal
    cluster.heal_partition()
    for msg in cluster.event_log[-1:]:
        print(f"\n  {msg}")
    components = detect_split_brain(cluster)
    print(f"  Connected components: {[sorted(c) for c in components]}")
    print(f"  Split brain? {'YES' if len(components) > 1 else 'No'}")


if __name__ == "__main__":
    demo_vector_clocks()
    demo_cap_cp()
    demo_cap_ap()
    demo_split_brain()
