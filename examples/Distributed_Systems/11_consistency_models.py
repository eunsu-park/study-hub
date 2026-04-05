"""
Consistency Models Simulator

Simulates and compares different consistency models in a distributed
key-value store: linearizability, sequential consistency, causal
consistency, and eventual consistency. Each model is demonstrated with
concrete read/write sequences showing which orderings are valid.

Key concepts:
- Linearizability: real-time ordering, strongest guarantee
- Sequential consistency: global order respecting per-process order
- Causal consistency: respects happens-before, concurrent ops unordered
- Eventual consistency: all replicas converge eventually
- PACELC theorem: partition behavior AND else (latency vs consistency)

Usage:
    python 11_consistency_models.py
"""

from __future__ import annotations

import random
import time as time_mod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

class OpKind(Enum):
    WRITE = "W"
    READ = "R"


@dataclass
class Operation:
    """A read or write operation with timing information."""
    client: str
    kind: OpKind
    key: str
    value: str | None = None       # Write value or read result
    start_time: float = 0.0
    end_time: float = 0.0
    replica: int = 0

    def __repr__(self) -> str:
        if self.kind == OpKind.WRITE:
            return f"{self.client}:W({self.key}={self.value})@[{self.start_time:.1f},{self.end_time:.1f}]"
        return f"{self.client}:R({self.key})={self.value}@[{self.start_time:.1f},{self.end_time:.1f}]"


# ---------------------------------------------------------------------------
# Replica store with configurable propagation
# ---------------------------------------------------------------------------

class Replica:
    """A single replica with delayed propagation."""

    def __init__(self, replica_id: int):
        self.replica_id = replica_id
        self._store: dict[str, str] = {}
        self._pending_writes: list[tuple[float, str, str]] = []  # (apply_time, key, val)
        self._write_log: list[tuple[float, str, str]] = []

    def write(self, key: str, value: str, timestamp: float) -> None:
        """Local write: immediately visible on this replica."""
        self._store[key] = value
        self._write_log.append((timestamp, key, value))

    def read(self, key: str) -> str | None:
        """Read from local state."""
        return self._store.get(key)

    def queue_remote_write(self, key: str, value: str, apply_time: float) -> None:
        """Queue a write from another replica with propagation delay."""
        self._pending_writes.append((apply_time, key, value))

    def apply_pending(self, current_time: float) -> int:
        """Apply pending remote writes that have arrived. Returns count applied."""
        applied = 0
        remaining = []
        for apply_time, key, value in self._pending_writes:
            if current_time >= apply_time:
                self._store[key] = value
                applied += 1
            else:
                remaining.append((apply_time, key, value))
        self._pending_writes = remaining
        return applied

    def snapshot(self) -> dict[str, str]:
        return dict(self._store)


class DistributedStore:
    """Multi-replica store with configurable replication delay."""

    def __init__(self, n_replicas: int, propagation_delay: float = 2.0,
                 seed: int = 42):
        self.replicas = {i: Replica(i) for i in range(n_replicas)}
        self.propagation_delay = propagation_delay
        self._rng = random.Random(seed)
        self.history: list[Operation] = []

    def write(self, client: str, key: str, value: str,
              replica_id: int, timestamp: float) -> Operation:
        """Write to a specific replica and propagate asynchronously."""
        op = Operation(client, OpKind.WRITE, key, value,
                       start_time=timestamp, end_time=timestamp + 0.1,
                       replica=replica_id)

        # Local write
        self.replicas[replica_id].write(key, value, timestamp)

        # Queue propagation to other replicas
        for rid, replica in self.replicas.items():
            if rid != replica_id:
                delay = self.propagation_delay * self._rng.uniform(0.5, 1.5)
                replica.queue_remote_write(key, value, timestamp + delay)

        self.history.append(op)
        return op

    def read(self, client: str, key: str, replica_id: int,
             timestamp: float) -> Operation:
        """Read from a specific replica at the given time."""
        self.replicas[replica_id].apply_pending(timestamp)
        value = self.replicas[replica_id].read(key)
        op = Operation(client, OpKind.READ, key, value,
                       start_time=timestamp, end_time=timestamp + 0.1,
                       replica=replica_id)
        self.history.append(op)
        return op

    def tick(self, timestamp: float) -> None:
        """Advance time on all replicas."""
        for replica in self.replicas.values():
            replica.apply_pending(timestamp)


# ---------------------------------------------------------------------------
# Consistency checkers
# ---------------------------------------------------------------------------

def check_linearizability(ops: list[Operation]) -> tuple[bool, list[str]]:
    """
    Check if operations are linearizable.
    Each operation must appear to take effect at a single point between
    its start and end time. Reads must return the latest completed write.
    """
    violations = []
    writes = [op for op in ops if op.kind == OpKind.WRITE]
    reads = [op for op in ops if op.kind == OpKind.READ]

    for read_op in reads:
        # Find the most recent write to this key that completed before read started
        latest_write = None
        for w in writes:
            if w.key == read_op.key and w.end_time <= read_op.start_time:
                if latest_write is None or w.end_time > latest_write.end_time:
                    latest_write = w

        if latest_write and read_op.value != latest_write.value:
            # Check if any concurrent write could justify this read
            concurrent_writes = [
                w for w in writes
                if w.key == read_op.key
                and w.start_time <= read_op.end_time
                and w.end_time >= read_op.start_time
            ]
            concurrent_vals = {w.value for w in concurrent_writes}
            if read_op.value not in concurrent_vals and read_op.value != latest_write.value:
                violations.append(
                    f"  {read_op} should return '{latest_write.value}' "
                    f"(last completed write)")

    return len(violations) == 0, violations


def check_eventual_consistency(store: DistributedStore,
                               final_time: float) -> tuple[bool, list[str]]:
    """Check that all replicas converge after sufficient time."""
    store.tick(final_time)
    snapshots = {rid: r.snapshot() for rid, r in store.replicas.items()}
    first = list(snapshots.values())[0]
    violations = []
    for rid, snap in snapshots.items():
        if snap != first:
            violations.append(f"  Replica {rid}: {snap} != Replica 0: {first}")
    return len(violations) == 0, violations


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_linearizability() -> None:
    """Demonstrate linearizability with a single-replica store."""
    print("=" * 70)
    print("Linearizability (Single Replica — Always Linearizable)")
    print("=" * 70)

    store = DistributedStore(n_replicas=1, propagation_delay=0)

    # All operations go to replica 0
    ops = []
    ops.append(store.write("A", "x", "1", 0, timestamp=1.0))
    ops.append(store.write("B", "x", "2", 0, timestamp=2.0))
    ops.append(store.read("A", "x", 0, timestamp=3.0))
    ops.append(store.write("A", "x", "3", 0, timestamp=4.0))
    ops.append(store.read("B", "x", 0, timestamp=5.0))

    print("\n  Operations (single replica):")
    for op in ops:
        print(f"    {op}")

    ok, violations = check_linearizability(ops)
    print(f"\n  Linearizable: {ok}")
    if violations:
        for v in violations:
            print(v)


def demo_linearizability_violation() -> None:
    """Show how multi-replica reads can violate linearizability."""
    print("\n" + "=" * 70)
    print("Linearizability Violation (Multi-Replica with Stale Reads)")
    print("=" * 70)

    store = DistributedStore(n_replicas=2, propagation_delay=5.0)

    # Client A writes x=1 to replica 0 at t=1
    op1 = store.write("A", "x", "1", replica_id=0, timestamp=1.0)
    # Client A writes x=2 to replica 0 at t=2
    op2 = store.write("A", "x", "2", replica_id=0, timestamp=2.0)
    # Client B reads from replica 1 at t=3 — propagation hasn't arrived!
    op3 = store.read("B", "x", replica_id=1, timestamp=3.0)

    ops = [op1, op2, op3]

    print("\n  Operations (two replicas, 5s propagation delay):")
    for op in ops:
        print(f"    {op} [replica {op.replica}]")

    ok, violations = check_linearizability(ops)
    print(f"\n  Linearizable: {ok}")
    if not ok:
        for v in violations:
            print(v)
        print(f"  Client B read from replica 1 before the write propagated")

    # Show eventual convergence
    print(f"\n  After waiting for propagation (t=10):")
    store.tick(10.0)
    for rid, replica in store.replicas.items():
        print(f"    Replica {rid}: x = {replica.read('x')}")


def demo_causal_consistency() -> None:
    """Illustrate causal consistency."""
    print("\n" + "=" * 70)
    print("Causal Consistency")
    print("=" * 70)

    print("""
  Causal consistency ensures: if operation A "happened before" B,
  then every process sees A before B. But concurrent operations
  may be seen in different orders by different processes.

  Example scenario:
    Client A: W(x=1) then W(y=2)          [causally ordered]
    Client B: R(x)=1 then W(z=3)          [B saw A's write, causal dep]
    Client C: can see W(z=3) before W(y=2) [y and z are concurrent]
""")

    store = DistributedStore(n_replicas=3, propagation_delay=3.0)

    # A writes x=1 at t=1 to replica 0
    store.write("A", "x", "1", replica_id=0, timestamp=1.0)
    # A writes y=2 at t=2 to replica 0 (causally after x=1)
    store.write("A", "y", "2", replica_id=0, timestamp=2.0)

    # B reads x=1 from replica 0 at t=3 (sees A's write)
    store.read("B", "x", replica_id=0, timestamp=3.0)
    # B writes z=3 at t=4 to replica 1 (causally after seeing x=1)
    store.write("B", "z", "3", replica_id=1, timestamp=4.0)

    # C reads from replica 2 at t=4.5 — may see z but not y yet
    r_z = store.read("C", "z", replica_id=2, timestamp=4.5)
    r_y = store.read("C", "y", replica_id=2, timestamp=4.5)
    r_x = store.read("C", "x", replica_id=2, timestamp=4.5)

    print(f"  At t=4.5, Replica 2 (Client C's view):")
    print(f"    x = {r_x.value}")
    print(f"    y = {r_y.value}")
    print(f"    z = {r_z.value}")
    print(f"\n  C might see z=3 before y=2 arrives — this is allowed under")
    print(f"  causal consistency because y and z are CONCURRENT (no causal link)")


def demo_eventual_consistency() -> None:
    """Show eventual consistency convergence."""
    print("\n" + "=" * 70)
    print("Eventual Consistency: Convergence Over Time")
    print("=" * 70)

    store = DistributedStore(n_replicas=3, propagation_delay=2.0, seed=42)

    # Multiple clients write to different replicas
    store.write("A", "x", "1", replica_id=0, timestamp=1.0)
    store.write("B", "y", "2", replica_id=1, timestamp=1.5)
    store.write("C", "z", "3", replica_id=2, timestamp=2.0)

    # Check state at various times
    check_times = [1.0, 2.0, 3.0, 4.0, 5.0, 8.0]

    print(f"\n  3 replicas, 3 writes to different replicas")
    print(f"  Propagation delay: ~2s\n")
    print(f"  {'Time':>6}  {'R0':>12}  {'R1':>12}  {'R2':>12}  Converged?")
    print("  " + "-" * 58)

    for t in check_times:
        store.tick(t)
        snaps = []
        for rid in range(3):
            snap = store.replicas[rid].snapshot()
            snaps.append(snap)

        converged = all(s == snaps[0] for s in snaps)
        snap_strs = [str(s) for s in snaps]
        print(f"  {t:>5.1f}s  {snap_strs[0]:>12}  {snap_strs[1]:>12}  "
              f"{snap_strs[2]:>12}  {'YES' if converged else 'no'}")

    ok, violations = check_eventual_consistency(store, 10.0)
    print(f"\n  Eventually consistent at t=10: {ok}")


def demo_pacelc() -> None:
    """Illustrate PACELC theorem tradeoffs."""
    print("\n" + "=" * 70)
    print("PACELC Theorem: Partition + Else Tradeoffs")
    print("=" * 70)

    print("""
  PACELC extends CAP: during Partition, choose A(vailability) or C(onsistency).
  Else (no partition), choose L(atency) or C(onsistency).

  ┌──────────────────┬─────────────┬──────────────┬─────────────────┐
  │ System           │ Partition   │ Else         │ Classification  │
  ├──────────────────┼─────────────┼──────────────┼─────────────────┤
  │ Spanner          │ C           │ C            │ PC/EC           │
  │ Dynamo/Cassandra │ A           │ L            │ PA/EL           │
  │ MongoDB          │ C (default) │ L (default)  │ PC/EL           │
  │ PNUTS            │ A           │ C            │ PA/EC           │
  └──────────────────┴─────────────┴──────────────┴─────────────────┘

  Simulating latency vs consistency tradeoff:
""")

    # Simulate: strong consistency (wait for all replicas) vs eventual (local read)
    n_ops = 20
    rng = random.Random(42)

    # Strong consistency: wait for propagation
    strong_latencies = [rng.uniform(2.0, 5.0) for _ in range(n_ops)]
    # Eventual: local read
    eventual_latencies = [rng.uniform(0.1, 0.3) for _ in range(n_ops)]

    strong_avg = sum(strong_latencies) / len(strong_latencies)
    eventual_avg = sum(eventual_latencies) / len(eventual_latencies)

    print(f"  {n_ops} read operations:")
    print(f"    Strong consistency:  avg latency = {strong_avg:.2f}s  (wait for quorum)")
    print(f"    Eventual consistency: avg latency = {eventual_avg:.2f}s  (local read)")
    print(f"    Speedup: {strong_avg / eventual_avg:.1f}x faster with eventual consistency")
    print(f"\n  Tradeoff: eventual reads may return stale data, but are much faster")


def demo_comparison() -> None:
    """Side-by-side comparison of all consistency models."""
    print("\n" + "=" * 70)
    print("Consistency Models Comparison")
    print("=" * 70)

    print("""
  ┌─────────────────────┬───────────┬──────────┬───────────┬──────────┐
  │ Property            │ Linear.   │ Seq.     │ Causal    │ Eventual │
  ├─────────────────────┼───────────┼──────────┼───────────┼──────────┤
  │ Real-time ordering  │ Yes       │ No       │ No        │ No       │
  │ Per-process order   │ Yes       │ Yes      │ Yes       │ No*      │
  │ Causal ordering     │ Yes       │ Yes      │ Yes       │ No       │
  │ Convergence         │ Immediate │ Immediate│ Eventually│ Eventually│
  │ Stale reads         │ Never     │ Possible │ Possible  │ Likely   │
  │ Availability        │ Low       │ Medium   │ High      │ Highest  │
  │ Latency             │ High      │ Medium   │ Low       │ Lowest   │
  │ Implementation      │ Raft/Paxos│ Zab      │ COPS      │ Dynamo   │
  └─────────────────────┴───────────┴──────────┴───────────┴──────────┘

  * Eventual consistency may reorder operations from the same process
    across different replicas during convergence.

  Stronger consistency = more coordination = higher latency = lower availability
  Weaker consistency  = less coordination = lower latency  = higher availability
""")


if __name__ == "__main__":
    demo_linearizability()
    demo_linearizability_violation()
    demo_causal_consistency()
    demo_eventual_consistency()
    demo_pacelc()
    demo_comparison()
    print("Done.")
