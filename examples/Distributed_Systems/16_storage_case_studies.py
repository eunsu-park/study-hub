"""
Distributed Storage Case Studies: Simplified Dynamo and Spanner Models

Simulates key architectural patterns from real-world distributed storage
systems: Dynamo's sloppy quorum with hinted handoff, and Spanner's
TrueTime-based external consistency. Compares design choices across
Dynamo, Spanner, Kafka, and CockroachDB.

Key concepts:
- Dynamo: sloppy quorum, hinted handoff, vector clocks, anti-entropy
- Spanner: TrueTime, external consistency, commit-wait
- Kafka: log-based storage, ISR (in-sync replicas), exactly-once
- CockroachDB: Raft per range, hybrid logical clocks

Usage:
    python 16_storage_case_studies.py
"""

from __future__ import annotations

import random
import time as time_mod
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Dynamo-style: Sloppy Quorum + Hinted Handoff
# ---------------------------------------------------------------------------

@dataclass
class DynamoNode:
    """A Dynamo-style storage node."""
    node_id: str
    store: dict[str, tuple[str, int]] = field(default_factory=dict)  # key -> (value, version)
    hints: list[tuple[str, str, str, int]] = field(default_factory=list)  # (target, key, val, ver)
    alive: bool = True


class DynamoCluster:
    """Simulates Dynamo's sloppy quorum and hinted handoff."""

    def __init__(self, n: int, seed: int = 42):
        self.nodes = [DynamoNode(f"node-{i}") for i in range(n)]
        self.n = n
        self._rng = random.Random(seed)
        self._version = 0

    def _preference_list(self, key: str) -> list[int]:
        """Hash key to a preference list of node indices."""
        h = hash(key) % self.n
        return [(h + i) % self.n for i in range(self.n)]

    def put(self, key: str, value: str, w: int = 3) -> dict:
        """Sloppy quorum write with hinted handoff."""
        self._version += 1
        pref = self._preference_list(key)
        result = {"written": [], "hinted": [], "success": False}

        written = 0
        for idx in pref:
            node = self.nodes[idx]
            if node.alive:
                node.store[key] = (value, self._version)
                result["written"].append(node.node_id)
                written += 1
            else:
                # Hinted handoff: write to next alive node with a hint
                for fallback_idx in pref:
                    fb = self.nodes[fallback_idx]
                    if fb.alive and fb.node_id not in result["written"]:
                        fb.hints.append((node.node_id, key, value, self._version))
                        result["hinted"].append(
                            f"{fb.node_id} (hint for {node.node_id})")
                        written += 1
                        break

            if written >= w:
                result["success"] = True
                break

        return result

    def handoff_hints(self) -> int:
        """Deliver hinted writes to recovered nodes."""
        delivered = 0
        for node in self.nodes:
            remaining = []
            for target_id, key, value, version in node.hints:
                target = next((n for n in self.nodes if n.node_id == target_id), None)
                if target and target.alive:
                    existing = target.store.get(key)
                    if existing is None or version > existing[1]:
                        target.store[key] = (value, version)
                    delivered += 1
                else:
                    remaining.append((target_id, key, value, version))
            node.hints = remaining
        return delivered


# ---------------------------------------------------------------------------
# Spanner-style: TrueTime and Commit-Wait
# ---------------------------------------------------------------------------

@dataclass
class TrueTimeInterval:
    """Represents a TrueTime interval [earliest, latest]."""
    earliest: float
    latest: float

    @property
    def uncertainty(self) -> float:
        return self.latest - self.earliest

    def __repr__(self) -> str:
        return f"[{self.earliest:.3f}, {self.latest:.3f}] (±{self.uncertainty:.3f})"


class TrueTimeClock:
    """Simulates Google's TrueTime API."""

    def __init__(self, epsilon_ms: float = 7.0, seed: int = 42):
        """
        Args:
            epsilon_ms: Maximum clock uncertainty in milliseconds.
        """
        self.epsilon_ms = epsilon_ms
        self._rng = random.Random(seed)
        self._base_time = 1000.0

    def now(self) -> TrueTimeInterval:
        """Return a TrueTime interval containing the true time."""
        self._base_time += self._rng.uniform(0.5, 2.0)
        epsilon = self._rng.uniform(1.0, self.epsilon_ms)
        return TrueTimeInterval(
            earliest=self._base_time - epsilon,
            latest=self._base_time + epsilon,
        )

    def after(self, t: float) -> bool:
        """Check if current time is definitely after t."""
        interval = self.now()
        return interval.earliest > t

    def before(self, t: float) -> bool:
        """Check if current time is definitely before t."""
        interval = self.now()
        return interval.latest < t


class SpannerTransaction:
    """Simulates Spanner's commit-wait protocol."""

    def __init__(self, clock: TrueTimeClock):
        self.clock = clock
        self.log: list[str] = []

    def commit(self, data: str) -> float:
        """
        Commit with external consistency using commit-wait.
        Returns the commit timestamp.
        """
        # Step 1: Acquire locks and choose commit timestamp
        tt = self.clock.now()
        commit_ts = tt.latest  # Choose the latest possible time
        self.log.append(f"TrueTime at commit: {tt}")
        self.log.append(f"Chosen commit timestamp: {commit_ts:.3f}")

        # Step 2: Commit-wait: wait until TrueTime.after(commit_ts)
        wait_iterations = 0
        while not self.clock.after(commit_ts):
            wait_iterations += 1

        self.log.append(f"Commit-wait: waited {wait_iterations} iterations")
        self.log.append(f"Now definitely after {commit_ts:.3f} => COMMIT '{data}'")

        return commit_ts


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_dynamo_sloppy_quorum() -> None:
    """Demonstrate Dynamo's sloppy quorum with hinted handoff."""
    print("=" * 70)
    print("Dynamo: Sloppy Quorum + Hinted Handoff")
    print("=" * 70)

    cluster = DynamoCluster(n=5, seed=42)

    # Normal write
    result = cluster.put("user:alice", "age=30", w=3)
    print(f"\n  Normal write (all nodes up):")
    print(f"    Written to: {result['written']}")
    print(f"    Hints: {result['hinted']}")
    print(f"    Success: {result['success']}")

    # Write with 2 nodes down
    cluster.nodes[0].alive = False
    cluster.nodes[1].alive = False
    result = cluster.put("user:bob", "age=25", w=3)
    print(f"\n  Write with 2 nodes down:")
    print(f"    Written to: {result['written']}")
    print(f"    Hints: {result['hinted']}")
    print(f"    Success: {result['success']}")

    # Show hints
    for node in cluster.nodes:
        if node.hints:
            print(f"    {node.node_id} holds hints: "
                  f"{[(h[0], h[1]) for h in node.hints]}")

    # Recover nodes and deliver hints
    cluster.nodes[0].alive = True
    cluster.nodes[1].alive = True
    delivered = cluster.handoff_hints()
    print(f"\n  Nodes recovered, delivered {delivered} hinted writes")

    # Verify all nodes have the data
    print(f"\n  Final state:")
    for node in cluster.nodes:
        print(f"    {node.node_id}: {dict(node.store)}")


def demo_spanner_truetime() -> None:
    """Demonstrate Spanner's TrueTime and commit-wait."""
    print("\n" + "=" * 70)
    print("Spanner: TrueTime and Commit-Wait")
    print("=" * 70)

    clock = TrueTimeClock(epsilon_ms=7.0, seed=42)

    # Show TrueTime intervals
    print(f"\n  TrueTime intervals (epsilon = ~7ms):")
    for i in range(5):
        tt = clock.now()
        print(f"    Sample {i}: {tt}")

    # Commit-wait demonstration
    print(f"\n  Commit-wait protocol:")
    txn1 = SpannerTransaction(clock)
    ts1 = txn1.commit("INSERT INTO users VALUES ('alice', 30)")
    for line in txn1.log:
        print(f"    {line}")

    txn2 = SpannerTransaction(clock)
    ts2 = txn2.commit("INSERT INTO users VALUES ('bob', 25)")
    print(f"\n  Second transaction:")
    for line in txn2.log:
        print(f"    {line}")

    print(f"\n  External consistency: ts1={ts1:.3f} < ts2={ts2:.3f}: {ts1 < ts2}")
    print(f"  Any observer seeing txn2 is guaranteed to also see txn1")


def demo_comparison() -> None:
    """Compare major distributed storage systems."""
    print("\n" + "=" * 70)
    print("Distributed Storage System Comparison")
    print("=" * 70)

    print("""
  ┌──────────────┬────────────────┬────────────────┬────────────────┐
  │ System       │ Spanner        │ Dynamo/Cass.   │ CockroachDB    │
  ├──────────────┼────────────────┼────────────────┼────────────────┤
  │ Consistency  │ Linearizable   │ Eventual       │ Serializable   │
  │ Replication  │ Paxos          │ Leaderless     │ Raft per range │
  │ Partitioning │ Range-based    │ Consistent hash│ Range-based    │
  │ Timestamps   │ TrueTime (GPS) │ Vector clocks  │ Hybrid logical │
  │ Transactions │ 2PC + Paxos    │ Per-key only   │ Percolator     │
  │ PACELC       │ PC/EC          │ PA/EL          │ PC/EC          │
  │ Use case     │ Global OLTP    │ High write vol │ Geo-distributed│
  └──────────────┴────────────────┴────────────────┴────────────────┘

  ┌──────────────┬────────────────┐
  │ Kafka        │ Notes          │
  ├──────────────┼────────────────┤
  │ Model        │ Log-based      │
  │ Replication  │ ISR (leader)   │
  │ Ordering     │ Per-partition  │
  │ Retention    │ Time/size based│
  │ Consistency  │ At-least-once* │
  │ Use case     │ Event streaming│
  └──────────────┴────────────────┘
  * Exactly-once with idempotent producer + transactions

  Key design decisions:
  1. Spanner: GPS clocks enable global consistency without coordination
  2. Dynamo: sacrifice consistency for availability (AP in CAP)
  3. CockroachDB: Spanner-like without GPS (uses HLC + clockskew bounds)
  4. Kafka: append-only log enables high throughput and replay
""")


if __name__ == "__main__":
    demo_dynamo_sloppy_quorum()
    demo_spanner_truetime()
    demo_comparison()
    print("Done.")
