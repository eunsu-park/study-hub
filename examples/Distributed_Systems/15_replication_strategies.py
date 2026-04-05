"""
Replication Strategies Simulator

Simulates single-leader, multi-leader, and leaderless replication
strategies. Demonstrates quorum reads/writes, read-repair, chain
replication, and the tradeoffs between latency, consistency, and
fault tolerance.

Key concepts:
- Single-leader: all writes through one node, strong consistency
- Multi-leader: writes at any leader, conflict resolution needed
- Leaderless (Dynamo-style): quorum R + W > N
- Chain replication: head writes, tail reads, strong consistency
- Read repair and anti-entropy

Usage:
    python 15_replication_strategies.py
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class VersionedValue:
    """A value with a version number for conflict detection."""
    value: str
    version: int
    timestamp: float
    origin: str       # Which node/leader wrote this

    def __repr__(self) -> str:
        return f"v{self.version}:'{self.value}'@{self.origin}"


class ReplicaNode:
    """A replica node that stores versioned key-value pairs."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.store: dict[str, VersionedValue] = {}
        self.alive = True
        self.log: list[str] = []

    def write(self, key: str, value: str, version: int,
              timestamp: float, origin: str) -> bool:
        """Write if version is higher than existing."""
        if not self.alive:
            return False
        existing = self.store.get(key)
        if existing is None or version > existing.version:
            self.store[key] = VersionedValue(value, version, timestamp, origin)
            self.log.append(f"WRITE {key}='{value}' v{version}")
            return True
        return False

    def read(self, key: str) -> VersionedValue | None:
        """Read the current value for a key."""
        if not self.alive:
            return None
        return self.store.get(key)


# ---------------------------------------------------------------------------
# Single-Leader Replication
# ---------------------------------------------------------------------------

class SingleLeaderCluster:
    """Single-leader replication: all writes through the leader."""

    def __init__(self, n_replicas: int):
        self.leader = ReplicaNode("leader")
        self.followers = [ReplicaNode(f"follower-{i}") for i in range(n_replicas - 1)]
        self.all_nodes = [self.leader] + self.followers
        self._version = 0
        self._time = 0.0

    def write(self, key: str, value: str) -> bool:
        """Write through the leader, replicate to followers."""
        self._version += 1
        self._time += 1.0
        self.leader.write(key, value, self._version, self._time, "leader")

        # Synchronous replication to followers
        replicated = 1
        for f in self.followers:
            if f.write(key, value, self._version, self._time, "leader"):
                replicated += 1
        return True

    def read(self, key: str, from_leader: bool = True) -> VersionedValue | None:
        """Read from leader (strong) or any follower (potentially stale)."""
        if from_leader:
            return self.leader.read(key)
        # Read from a random alive follower
        alive = [f for f in self.followers if f.alive]
        if alive:
            return random.choice(alive).read(key)
        return self.leader.read(key)


# ---------------------------------------------------------------------------
# Leaderless Replication (Dynamo-style)
# ---------------------------------------------------------------------------

class LeaderlessCluster:
    """Dynamo-style leaderless replication with quorum reads/writes."""

    def __init__(self, n: int, seed: int = 42):
        self.n = n
        self.nodes = [ReplicaNode(f"node-{i}") for i in range(n)]
        self._version = 0
        self._time = 0.0
        self._rng = random.Random(seed)

    def write(self, key: str, value: str, w: int) -> tuple[bool, int]:
        """
        Write to W replicas. Returns (success, count_written).
        Success requires at least W acknowledgments.
        """
        self._version += 1
        self._time += 1.0
        written = 0
        for node in self.nodes:
            if node.write(key, value, self._version, self._time, node.node_id):
                written += 1
                if written >= w:
                    break
        return written >= w, written

    def read(self, key: str, r: int) -> tuple[VersionedValue | None, int]:
        """
        Read from R replicas. Returns the most recent value and read count.
        """
        results: list[VersionedValue] = []
        for node in self.nodes:
            val = node.read(key)
            if val is not None:
                results.append(val)
                if len(results) >= r:
                    break

        if not results:
            return None, 0

        # Return the highest-versioned value
        best = max(results, key=lambda v: v.version)
        return best, len(results)

    def read_repair(self, key: str, expected: VersionedValue) -> int:
        """Push the correct value to stale replicas. Returns count repaired."""
        repaired = 0
        for node in self.nodes:
            if not node.alive:
                continue
            current = node.read(key)
            if current is None or current.version < expected.version:
                node.write(key, expected.value, expected.version,
                           expected.timestamp, expected.origin)
                repaired += 1
        return repaired


# ---------------------------------------------------------------------------
# Chain Replication
# ---------------------------------------------------------------------------

class ChainReplicationCluster:
    """
    Chain replication: writes go to head, propagate through chain,
    reads served from tail. Strong consistency with high throughput.
    """

    def __init__(self, chain_length: int):
        self.chain = [ReplicaNode(f"chain-{i}") for i in range(chain_length)]
        self._version = 0
        self._time = 0.0

    @property
    def head(self) -> ReplicaNode:
        return self.chain[0]

    @property
    def tail(self) -> ReplicaNode:
        return self.chain[-1]

    def write(self, key: str, value: str) -> bool:
        """Write to head, propagate through chain."""
        self._version += 1
        self._time += 1.0

        for node in self.chain:
            if not node.write(key, value, self._version, self._time, "head"):
                return False
        return True

    def read(self, key: str) -> VersionedValue | None:
        """Read from tail (always returns committed value)."""
        return self.tail.read(key)


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_single_leader() -> None:
    """Demonstrate single-leader replication."""
    print("=" * 70)
    print("Single-Leader Replication")
    print("=" * 70)

    cluster = SingleLeaderCluster(n_replicas=3)
    cluster.write("x", "1")
    cluster.write("y", "2")
    cluster.write("x", "3")

    print("\n  After 3 writes (x=1, y=2, x=3):")
    for node in cluster.all_nodes:
        snap = {k: str(v) for k, v in node.store.items()}
        print(f"    {node.node_id}: {snap}")

    # Simulate follower lag
    cluster.followers[1].alive = False
    cluster.write("z", "4")
    cluster.followers[1].alive = True

    print(f"\n  After writing z=4 with follower-1 down:")
    for node in cluster.all_nodes:
        snap = {k: str(v) for k, v in node.store.items()}
        print(f"    {node.node_id}: {snap}")

    print(f"\n  follower-1 missed z=4 — needs catch-up replication")


def demo_leaderless_quorum() -> None:
    """Demonstrate quorum reads and writes."""
    print("\n" + "=" * 70)
    print("Leaderless Replication: Quorum R + W > N")
    print("=" * 70)

    n = 5
    cluster = LeaderlessCluster(n=n, seed=42)

    # Demonstrate different quorum configurations
    configs = [
        (3, 3, "Strict: W=3, R=3 (strongly consistent)"),
        (3, 1, "Write-heavy: W=3, R=1 (fast reads, but R+W=4 < N+1=6? No: 3+1=4 < 6)"),
        (1, 5, "Read-heavy: W=1, R=5 (fast writes, all-read)"),
        (3, 3, "Balanced: W=3, R=3 (R+W=6 > N=5)"),
    ]

    for w, r, desc in configs:
        cluster2 = LeaderlessCluster(n=n, seed=42)
        ok, written = cluster2.write("x", "hello", w=w)
        val, read_count = cluster2.read("x", r=r)

        quorum_ok = r + w > n
        print(f"\n  {desc}")
        print(f"    Write: W={w}, wrote to {written} nodes, success={ok}")
        print(f"    Read:  R={r}, read from {read_count} nodes, value={val}")
        print(f"    R+W={r+w} {'>' if quorum_ok else '<='} N={n}: "
              f"{'CONSISTENT' if quorum_ok else 'MAY BE STALE'}")


def demo_read_repair() -> None:
    """Show read repair in leaderless replication."""
    print("\n" + "=" * 70)
    print("Read Repair: Fixing Stale Replicas")
    print("=" * 70)

    cluster = LeaderlessCluster(n=5, seed=42)

    # Write to all 5 nodes
    cluster.write("x", "v1", w=5)

    # Take 2 nodes down, write update
    cluster.nodes[3].alive = False
    cluster.nodes[4].alive = False
    cluster.write("x", "v2", w=3)

    # Bring nodes back
    cluster.nodes[3].alive = True
    cluster.nodes[4].alive = True

    print(f"\n  State after partial write (nodes 3,4 missed update):")
    for node in cluster.nodes:
        val = node.read("x")
        print(f"    {node.node_id}: {val}")

    # Read with quorum detects stale value
    val, _ = cluster.read("x", r=3)
    print(f"\n  Quorum read returns: {val}")

    # Trigger read repair
    if val:
        repaired = cluster.read_repair("x", val)
        print(f"  Read repair: updated {repaired} stale replicas")

    print(f"\n  State after read repair:")
    for node in cluster.nodes:
        v = node.read("x")
        print(f"    {node.node_id}: {v}")


def demo_chain_replication() -> None:
    """Demonstrate chain replication."""
    print("\n" + "=" * 70)
    print("Chain Replication")
    print("=" * 70)

    cluster = ChainReplicationCluster(chain_length=4)
    cluster.write("x", "100")
    cluster.write("y", "200")

    print(f"\n  Chain: head -> node-1 -> node-2 -> tail")
    print(f"  Writes go to HEAD, reads from TAIL\n")

    for node in cluster.chain:
        snap = {k: str(v) for k, v in node.store.items()}
        role = "HEAD" if node == cluster.head else ("TAIL" if node == cluster.tail else "MID")
        print(f"    {node.node_id} [{role}]: {snap}")

    val = cluster.read("x")
    print(f"\n  Read 'x' from tail: {val}")
    print(f"\n  Properties:")
    print(f"    - Strong consistency: tail only returns committed values")
    print(f"    - High write throughput: writes pipelined through chain")
    print(f"    - Read throughput: tail handles all reads (can be bottleneck)")


def demo_comparison() -> None:
    """Compare all replication strategies."""
    print("\n" + "=" * 70)
    print("Replication Strategy Comparison")
    print("=" * 70)

    print("""
  ┌───────────────────┬──────────────┬───────────────┬──────────────────┐
  │ Property          │ Single-Leader│ Leaderless    │ Chain Repl.      │
  ├───────────────────┼──────────────┼───────────────┼──────────────────┤
  │ Consistency       │ Strong       │ Tunable       │ Strong           │
  │ Write path        │ Leader only  │ Any W nodes   │ Head -> chain    │
  │ Read path         │ Leader/follow│ Any R nodes   │ Tail only        │
  │ Conflict handling │ No conflicts │ Last-write-win│ No conflicts     │
  │ Write latency     │ Low (1 hop)  │ W hops        │ Chain length     │
  │ Read latency      │ Low          │ R hops        │ Low (1 hop)      │
  │ Failover          │ Election     │ Automatic     │ Remove failed    │
  │ Examples          │ PostgreSQL   │ Cassandra     │ HDFS, Chain Rep. │
  └───────────────────┴──────────────┴───────────────┴──────────────────┘
""")


if __name__ == "__main__":
    demo_single_leader()
    demo_leaderless_quorum()
    demo_read_repair()
    demo_chain_replication()
    demo_comparison()
    print("Done.")
