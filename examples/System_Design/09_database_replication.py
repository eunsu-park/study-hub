"""
Database Replication — Leader-Follower, Multi-Leader, Conflict Resolution

Demonstrates:
- Single-leader (master-slave) replication
- Multi-leader replication with conflict detection
- Conflict resolution strategies (last-write-wins, merge, custom)
- Replication lag simulation

Theory:
- Replication copies data across multiple nodes for fault tolerance and
  read scalability.
- Single-leader: one node accepts writes, followers replicate asynchronously.
  Simple, but the leader is a bottleneck and SPOF.
- Multi-leader: multiple nodes accept writes. Higher write throughput and
  availability, but conflicts can occur when the same record is modified
  on different leaders concurrently.
- Conflict resolution: LWW (last-write-wins) is simple but loses data;
  merge strategies preserve both changes; application-level resolution
  gives the most control.

Adapted from System Design Lesson 09.
"""

import time
import random
from dataclasses import dataclass, field
from enum import Enum


# ── Data Model ────────────────────────────────────────────────────────

@dataclass
class WriteOp:
    """A single write operation."""
    key: str
    value: str
    timestamp: float
    origin: str  # node that accepted the write


@dataclass
class ReplicaNode:
    """A database replica node."""
    name: str
    is_leader: bool = False
    data: dict[str, str] = field(default_factory=dict)
    write_log: list[WriteOp] = field(default_factory=list)
    replication_lag_ms: float = 0.0
    alive: bool = True


# ── Single-Leader Replication ─────────────────────────────────────────

# Why: Single-leader replication is the default for most databases (PostgreSQL,
# MySQL). All writes go to one leader, then replicate to followers. This
# guarantees no write conflicts but creates a single point of failure and
# limits write throughput to one machine.
class SingleLeaderCluster:
    """Single-leader replication cluster."""

    def __init__(self, num_followers: int = 3,
                 replication_lag_range: tuple[float, float] = (5.0, 50.0)):
        self.leader = ReplicaNode("leader", is_leader=True)
        self.followers = [
            ReplicaNode(
                f"follower-{i}",
                replication_lag_ms=random.uniform(*replication_lag_range),
            )
            for i in range(num_followers)
        ]
        self.event_log: list[str] = []

    def write(self, key: str, value: str) -> bool:
        """Write to leader, replicate to followers."""
        if not self.leader.alive:
            self.event_log.append(f"  WRITE FAILED: leader is down")
            return False

        op = WriteOp(key, value, time.monotonic(), self.leader.name)
        self.leader.data[key] = value
        self.leader.write_log.append(op)
        self.event_log.append(
            f"  [{self.leader.name}] WRITE {key}={value}"
        )

        # Replicate to followers
        acks = 0
        for f in self.followers:
            if f.alive:
                f.data[key] = value
                f.write_log.append(op)
                acks += 1
                self.event_log.append(
                    f"  [{f.name}] replicated {key}={value} "
                    f"(lag ~{f.replication_lag_ms:.0f}ms)"
                )
        return True

    def read(self, key: str, from_node: str = "leader") -> str | None:
        """Read from a specific node."""
        if from_node == "leader":
            return self.leader.data.get(key)
        for f in self.followers:
            if f.name == from_node:
                return f.data.get(key)
        return None

    def failover(self) -> str | None:
        """Promote a follower to leader when leader fails."""
        # Why: During failover, we pick the follower with the most write log
        # entries (most up-to-date). This minimizes data loss but can still
        # lose writes that hadn't replicated yet — the fundamental trade-off
        # of asynchronous replication.
        if self.leader.alive:
            return None

        best = max(
            (f for f in self.followers if f.alive),
            key=lambda f: len(f.write_log),
            default=None,
        )
        if best:
            best.is_leader = True
            self.leader = best
            self.followers = [f for f in self.followers if f.name != best.name]
            self.event_log.append(f"  FAILOVER: {best.name} promoted to leader")
            return best.name
        return None


# ── Multi-Leader Replication ──────────────────────────────────────────

class ConflictStrategy(Enum):
    LAST_WRITE_WINS = "lww"
    MERGE = "merge"
    CUSTOM = "custom"


# Why: Multi-leader replication is used when you need writes in multiple
# datacenters (e.g., Google Docs, CRDTs). The price you pay is conflict
# handling — two leaders can accept conflicting writes to the same key
# before they learn about each other's changes.
class MultiLeaderCluster:
    """Multi-leader replication with conflict detection and resolution."""

    def __init__(self, leader_names: list[str],
                 strategy: ConflictStrategy = ConflictStrategy.LAST_WRITE_WINS):
        self.leaders = {
            name: ReplicaNode(name, is_leader=True)
            for name in leader_names
        }
        self.strategy = strategy
        self.conflicts: list[dict] = []
        self.event_log: list[str] = []
        self._clock = 0.0

    def write(self, leader_name: str, key: str, value: str) -> None:
        """Write to a specific leader."""
        self._clock += 1
        leader = self.leaders[leader_name]
        op = WriteOp(key, value, self._clock, leader_name)
        leader.data[key] = value
        leader.write_log.append(op)
        self.event_log.append(
            f"  [{leader_name}] WRITE {key}={value} (t={self._clock:.0f})"
        )

    def sync(self) -> list[dict]:
        """Synchronize all leaders, detecting and resolving conflicts."""
        detected = []
        all_ops: dict[str, list[WriteOp]] = {}

        # Collect all writes per key across leaders
        for leader in self.leaders.values():
            for op in leader.write_log:
                all_ops.setdefault(op.key, []).append(op)

        # Detect conflicts: same key written by different leaders
        for key, ops in all_ops.items():
            origins = set(op.origin for op in ops)
            if len(origins) > 1:
                conflict = {"key": key, "ops": ops}
                resolved_value = self._resolve(ops)
                conflict["resolved"] = resolved_value
                detected.append(conflict)

                # Apply resolution to all leaders
                for leader in self.leaders.values():
                    leader.data[key] = resolved_value

                self.event_log.append(
                    f"  CONFLICT on '{key}': {len(ops)} writes from "
                    f"{origins} → resolved to '{resolved_value}'"
                )

        self.conflicts.extend(detected)
        # Clear logs after sync
        for leader in self.leaders.values():
            leader.write_log.clear()
        return detected

    def _resolve(self, ops: list[WriteOp]) -> str:
        """Resolve conflicting writes."""
        if self.strategy == ConflictStrategy.LAST_WRITE_WINS:
            # Why: LWW is the simplest strategy — highest timestamp wins.
            # It guarantees convergence but silently drops concurrent writes.
            # CassandraDB uses this as its default conflict resolution.
            winner = max(ops, key=lambda o: o.timestamp)
            return winner.value

        elif self.strategy == ConflictStrategy.MERGE:
            # Concatenate all values (e.g., for collaborative editing)
            values = sorted(set(op.value for op in ops))
            return " | ".join(values)

        else:
            # Custom: keep the longest value
            return max(ops, key=lambda o: len(o.value)).value


# ── Replication Lag Simulator ─────────────────────────────────────────

def simulate_replication_lag(num_writes: int = 20,
                              lag_range: tuple[float, float] = (10, 200)):
    """Simulate stale reads due to replication lag."""
    results = []
    leader_data: dict[str, str] = {}
    follower_data: dict[str, str] = {}
    follower_lag_ms = random.uniform(*lag_range)
    pending_replications: list[tuple[float, str, str]] = []

    clock = 0.0
    for i in range(num_writes):
        clock += random.uniform(5, 50)  # time between writes
        key = f"key-{i % 5}"
        value = f"v{i}"

        # Write to leader
        leader_data[key] = value
        # Schedule replication
        pending_replications.append((clock + follower_lag_ms, key, value))

        # Apply any pending replications
        for repl_time, rkey, rval in pending_replications[:]:
            if clock >= repl_time:
                follower_data[rkey] = rval
                pending_replications.remove((repl_time, rkey, rval))

        # Check for stale read
        follower_val = follower_data.get(key, "MISSING")
        is_stale = follower_val != value
        results.append({
            "time": clock,
            "key": key,
            "leader": value,
            "follower": follower_val,
            "stale": is_stale,
        })

    return results, follower_lag_ms


# ── Demos ─────────────────────────────────────────────────────────────

def demo_single_leader():
    print("=" * 60)
    print("SINGLE-LEADER REPLICATION")
    print("=" * 60)

    cluster = SingleLeaderCluster(num_followers=3)

    print(f"\n  Cluster: 1 leader + {len(cluster.followers)} followers")

    # Normal writes
    for key, val in [("user:1", "Alice"), ("user:2", "Bob"), ("user:3", "Carol")]:
        cluster.write(key, val)

    for msg in cluster.event_log:
        print(msg)

    # Show data consistency
    print(f"\n  Data on each node:")
    print(f"    leader: {dict(cluster.leader.data)}")
    for f in cluster.followers:
        print(f"    {f.name}: {dict(f.data)}")


def demo_failover():
    print("\n" + "=" * 60)
    print("LEADER FAILOVER")
    print("=" * 60)

    cluster = SingleLeaderCluster(num_followers=3)
    cluster.write("x", "1")
    cluster.write("y", "2")
    cluster.event_log.clear()

    # Kill leader
    print(f"\n  Leader crashes after 2 writes!")
    cluster.leader.alive = False

    # Try write — should fail
    success = cluster.write("z", "3")
    print(f"  Write during outage: {'success' if success else 'FAILED'}")

    # Failover
    new_leader = cluster.failover()
    for msg in cluster.event_log:
        print(msg)
    cluster.event_log.clear()

    # Resume writes
    success = cluster.write("z", "3")
    print(f"  Write after failover: {'success' if success else 'FAILED'}")
    print(f"  New leader data: {dict(cluster.leader.data)}")


def demo_multi_leader():
    print("\n" + "=" * 60)
    print("MULTI-LEADER REPLICATION")
    print("=" * 60)

    for strategy in ConflictStrategy:
        cluster = MultiLeaderCluster(
            ["DC-US", "DC-EU", "DC-AP"], strategy=strategy
        )

        # Concurrent writes to same key from different leaders
        cluster.write("DC-US", "profile:1", "name=Alice")
        cluster.write("DC-EU", "profile:1", "name=Alicia")
        cluster.write("DC-AP", "config:theme", "dark")
        cluster.write("DC-US", "config:theme", "light")

        print(f"\n  Strategy: {strategy.value}")
        print(f"  Before sync:")
        for name, leader in cluster.leaders.items():
            print(f"    {name}: {dict(leader.data)}")

        conflicts = cluster.sync()
        for msg in cluster.event_log:
            print(msg)

        print(f"  After sync (all leaders converged):")
        for name, leader in cluster.leaders.items():
            print(f"    {name}: {dict(leader.data)}")
        print(f"  Conflicts detected: {len(conflicts)}")


def demo_replication_lag():
    print("\n" + "=" * 60)
    print("REPLICATION LAG — STALE READS")
    print("=" * 60)

    results, lag = simulate_replication_lag(num_writes=15)

    print(f"\n  Follower replication lag: {lag:.0f} ms")
    print(f"\n  {'Time':>8} {'Key':>8} {'Leader':>10} {'Follower':>10} {'Stale?':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

    stale_count = 0
    for r in results:
        stale_count += r["stale"]
        print(f"  {r['time']:>7.0f} {r['key']:>8} {r['leader']:>10} "
              f"{r['follower']:>10} {'YES' if r['stale'] else '':>8}")

    print(f"\n  Stale reads: {stale_count}/{len(results)} "
          f"({stale_count/len(results)*100:.0f}%)")
    print(f"  Mitigation: read-your-writes consistency, monotonic reads,")
    print(f"  or synchronous replication (at the cost of latency).")


if __name__ == "__main__":
    demo_single_leader()
    demo_failover()
    demo_multi_leader()
    demo_replication_lag()
