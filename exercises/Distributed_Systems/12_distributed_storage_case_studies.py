"""
Exercises for Lesson 12: Distributed Storage Case Studies
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import time
import random
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field


# === Exercise 1: Dynamo-Style Sloppy Quorum with Hinted Handoff ===
# Problem: Simulate Dynamo's sloppy quorum where writes can go to
# non-preferred nodes when preferred nodes are down. Implement hinted
# handoff to transfer data back when nodes recover.

@dataclass
class HintedValue:
    """A value stored with a hint for its intended node."""
    key: str
    value: int
    version: int
    intended_node: str  # the node that should eventually own this


class DynamoNode:
    """A node in a Dynamo-style system."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.data: Dict[str, Tuple[int, int]] = {}  # key -> (value, version)
        self.hints: List[HintedValue] = []  # hinted handoff queue
        self.is_alive = True

    def write(self, key: str, value: int, version: int) -> bool:
        if not self.is_alive:
            return False
        current = self.data.get(key)
        if current is None or version > current[1]:
            self.data[key] = (value, version)
        return True

    def write_hint(self, hint: HintedValue) -> bool:
        """Accept a hinted write intended for another node."""
        if not self.is_alive:
            return False
        self.hints.append(hint)
        self.data[hint.key] = (hint.value, hint.version)
        return True

    def handoff_hints(self, target_node: "DynamoNode") -> int:
        """Transfer hinted data to its intended owner."""
        handed_off = 0
        remaining = []
        for hint in self.hints:
            if hint.intended_node == target_node.node_id:
                if target_node.write(hint.key, hint.value, hint.version):
                    handed_off += 1
                else:
                    remaining.append(hint)
            else:
                remaining.append(hint)
        self.hints = remaining
        return handed_off


class DynamoCluster:
    """Simplified Dynamo cluster with sloppy quorum."""

    def __init__(self, node_ids: List[str], n: int = 3, w: int = 2, r: int = 2):
        self.nodes = {nid: DynamoNode(nid) for nid in node_ids}
        self.preference_list = node_ids  # sorted ring order
        self.n = n  # replication factor
        self.w = w  # write quorum
        self.r = r
        self.version = 0

    def _get_preference_list(self, key: str) -> List[str]:
        """Get N preferred nodes for a key (simplified: first N nodes)."""
        h = hash(key) % len(self.preference_list)
        result = []
        for i in range(len(self.preference_list)):
            idx = (h + i) % len(self.preference_list)
            result.append(self.preference_list[idx])
            if len(result) == self.n:
                break
        return result

    def sloppy_write(self, key: str, value: int) -> Tuple[bool, List[str]]:
        """
        Write with sloppy quorum. If preferred nodes are down,
        write to the next available node with a hint.
        """
        self.version += 1
        version = self.version
        preferred = self._get_preference_list(key)
        written_to = []
        acks = 0

        # Try preferred nodes first
        for nid in preferred:
            node = self.nodes[nid]
            if node.write(key, value, version):
                written_to.append(nid)
                acks += 1

        # Sloppy quorum: try non-preferred nodes if needed
        if acks < self.w:
            for nid in self.preference_list:
                if nid not in preferred:
                    node = self.nodes[nid]
                    hint = HintedValue(key, value, version, preferred[0])
                    if node.write_hint(hint):
                        written_to.append(f"{nid}(hint)")
                        acks += 1
                        if acks >= self.w:
                            break

        return (acks >= self.w, written_to)


def exercise_1():
    """
    Demonstrate sloppy quorum with hinted handoff.
    """
    print("=== Exercise 1: Dynamo Sloppy Quorum + Hinted Handoff ===\n")

    cluster = DynamoCluster(["N1", "N2", "N3", "N4", "N5"], n=3, w=2, r=2)

    # Normal write
    success, targets = cluster.sloppy_write("user:1", 42)
    print(f"Normal write: success={success}, targets={targets}")

    # Take down N1 and N2
    cluster.nodes["N1"].is_alive = False
    cluster.nodes["N2"].is_alive = False
    print(f"\nN1 and N2 are DOWN")

    # Sloppy quorum write
    success, targets = cluster.sloppy_write("user:2", 99)
    print(f"Sloppy write: success={success}, targets={targets}")

    # Check hints stored
    for nid, node in cluster.nodes.items():
        if node.hints:
            print(f"  {nid} has {len(node.hints)} hints")

    # Recover N1
    cluster.nodes["N1"].is_alive = True
    print(f"\nN1 recovered!")

    # Handoff hints
    for nid, node in cluster.nodes.items():
        if node.hints:
            count = node.handoff_hints(cluster.nodes["N1"])
            if count:
                print(f"  {nid} handed off {count} hints to N1")

    print(f"  N1 data: {cluster.nodes['N1'].data}")
    print()


# === Exercise 2: Kafka-Style ISR Manager ===
# Problem: Implement a simplified In-Sync Replica (ISR) manager.
# Replicas fall out of ISR if they fall behind the leader's log.
# The ISR shrinks and expands dynamically based on replica lag.

@dataclass
class ReplicaState:
    """State of a Kafka-like replica."""
    replica_id: str
    log_end_offset: int = 0
    last_fetch_time: float = 0.0
    is_leader: bool = False


class ISRManager:
    """
    Manages the In-Sync Replica set for a Kafka-style partition.
    """

    def __init__(
        self,
        partition_id: str,
        replicas: List[str],
        leader: str,
        max_lag_offsets: int = 10,
        max_lag_time_ms: float = 10000,
    ):
        self.partition_id = partition_id
        self.leader = leader
        self.max_lag_offsets = max_lag_offsets
        self.max_lag_time_ms = max_lag_time_ms
        self.replicas: Dict[str, ReplicaState] = {}
        self.isr: Set[str] = set()
        self.leader_offset = 0
        self.log: List[str] = []

        for rid in replicas:
            self.replicas[rid] = ReplicaState(
                replica_id=rid, is_leader=(rid == leader)
            )
            self.isr.add(rid)

    def produce(self, message: str) -> int:
        """Leader appends a message. Returns the new offset."""
        self.leader_offset += 1
        self.log.append(message)
        self.replicas[self.leader].log_end_offset = self.leader_offset
        return self.leader_offset

    def fetch(self, replica_id: str, current_time: float) -> Optional[int]:
        """
        Replica fetches from leader. Updates its offset and fetch time.
        """
        if replica_id not in self.replicas or replica_id == self.leader:
            return None

        replica = self.replicas[replica_id]
        # Simulate fetching: replica catches up by 1 offset
        if replica.log_end_offset < self.leader_offset:
            replica.log_end_offset += 1
        replica.last_fetch_time = current_time
        return replica.log_end_offset

    def check_isr(self, current_time: float) -> Tuple[Set[str], Set[str]]:
        """
        Check ISR membership. Remove replicas that are too far behind.
        Returns (removed, added) sets.
        """
        removed = set()
        added = set()

        for rid, state in self.replicas.items():
            if rid == self.leader:
                continue

            lag_offsets = self.leader_offset - state.log_end_offset
            lag_time = current_time - state.last_fetch_time if state.last_fetch_time > 0 else 0

            if rid in self.isr:
                # Check if should be removed
                if lag_offsets > self.max_lag_offsets or lag_time > self.max_lag_time_ms:
                    self.isr.discard(rid)
                    removed.add(rid)
            else:
                # Check if should be re-added
                if lag_offsets <= self.max_lag_offsets and lag_time <= self.max_lag_time_ms:
                    self.isr.add(rid)
                    added.add(rid)

        return (removed, added)

    def min_isr_met(self, min_isr: int) -> bool:
        """Check if the minimum ISR requirement is met."""
        return len(self.isr) >= min_isr


def exercise_2():
    """
    Demonstrate ISR management with replica lag.
    """
    print("=== Exercise 2: Kafka-Style ISR Manager ===\n")

    mgr = ISRManager(
        "topic-0", ["R0", "R1", "R2"], leader="R0",
        max_lag_offsets=5, max_lag_time_ms=5000,
    )

    current_time = 1000.0

    # Produce messages, R1 fetches but R2 falls behind
    print("Producing messages and R1 fetching...")
    for i in range(10):
        mgr.produce(f"msg_{i}")
        if i < 8:  # R1 keeps up mostly
            mgr.fetch("R1", current_time + i * 100)

    # R2 never fetches -> falls out of ISR
    removed, added = mgr.check_isr(current_time + 1000)
    print(f"ISR: {sorted(mgr.isr)}")
    print(f"Removed from ISR: {removed}")
    print(f"Leader offset: {mgr.leader_offset}")
    print(f"R1 offset: {mgr.replicas['R1'].log_end_offset}")
    print(f"R2 offset: {mgr.replicas['R2'].log_end_offset}")
    print(f"Min ISR (2) met: {mgr.min_isr_met(2)}")

    # R2 catches up
    print(f"\nR2 starts catching up...")
    for i in range(15):
        mgr.fetch("R2", current_time + 2000)
    removed, added = mgr.check_isr(current_time + 2000)
    print(f"R2 offset: {mgr.replicas['R2'].log_end_offset}")
    print(f"Added back to ISR: {added}")
    print(f"ISR: {sorted(mgr.isr)}")
    print()


# === Exercise 3: Simplified TrueTime API ===
# Problem: Build a simplified TrueTime API that returns a time interval
# [earliest, latest] representing clock uncertainty. Simulate the effect
# of GPS synchronization and crystal oscillator drift.

class TrueTime:
    """
    Simplified Google TrueTime API.

    Returns a time interval [earliest, latest] where the actual time
    is guaranteed to fall within this interval.
    """

    def __init__(
        self,
        base_uncertainty_ms: float = 1.0,
        drift_rate_ppm: float = 200.0,
        sync_interval_ms: float = 30000.0,
    ):
        """
        Args:
            base_uncertainty_ms: Minimum uncertainty after GPS sync.
            drift_rate_ppm: Crystal oscillator drift in parts per million.
            sync_interval_ms: Time between GPS synchronizations.
        """
        self.base_uncertainty_ms = base_uncertainty_ms
        self.drift_rate_ppm = drift_rate_ppm
        self.sync_interval_ms = sync_interval_ms
        self.last_sync_time = 0.0
        self.actual_time = 0.0

    def sync(self):
        """Perform GPS synchronization (resets uncertainty)."""
        self.last_sync_time = self.actual_time

    def advance(self, ms: float):
        """Advance the actual time by ms milliseconds."""
        self.actual_time += ms

    def now(self) -> Tuple[float, float]:
        """
        TT.now() returns [earliest, latest] interval.

        Uncertainty grows with time since last sync:
        uncertainty = base + drift_rate * time_since_sync
        """
        time_since_sync = self.actual_time - self.last_sync_time
        drift_uncertainty = (self.drift_rate_ppm / 1e6) * time_since_sync
        total_uncertainty = self.base_uncertainty_ms + drift_uncertainty

        earliest = self.actual_time - total_uncertainty
        latest = self.actual_time + total_uncertainty
        return (earliest, latest)

    def after(self, timestamp: float) -> bool:
        """TT.after(t): True if t has definitely passed."""
        earliest, _ = self.now()
        return earliest > timestamp

    def before(self, timestamp: float) -> bool:
        """TT.before(t): True if t has definitely not arrived."""
        _, latest = self.now()
        return latest < timestamp


def exercise_3():
    """
    Demonstrate TrueTime API with clock uncertainty.
    """
    print("=== Exercise 3: Simplified TrueTime API ===\n")

    tt = TrueTime(
        base_uncertainty_ms=1.0,
        drift_rate_ppm=200.0,
        sync_interval_ms=30000.0,
    )

    # GPS sync
    tt.sync()

    print("Time progression and uncertainty growth:")
    print(f"{'Time (ms)':>12s} {'Earliest':>12s} {'Latest':>12s} {'Uncertainty':>14s}")
    print("-" * 55)

    for ms in [0, 1000, 5000, 10000, 20000, 30000]:
        tt.actual_time = float(ms)
        earliest, latest = tt.now()
        uncertainty = (latest - earliest) / 2
        print(f"{ms:12.0f} {earliest:12.2f} {latest:12.2f} {uncertainty:14.4f}")

    # Demonstrate commit-wait
    print(f"\nSpanner-style commit-wait:")
    tt.actual_time = 10000.0
    _, commit_ts = tt.now()
    print(f"  Transaction commits at latest={commit_ts:.2f}")

    # Wait until commit_ts has definitely passed
    wait_count = 0
    while not tt.after(commit_ts):
        tt.advance(0.5)
        wait_count += 1

    print(f"  Waited {wait_count * 0.5:.1f}ms until TT.after({commit_ts:.2f}) = True")
    print(f"  Now safe to release locks (linearizability guaranteed)")

    # After GPS re-sync, uncertainty resets
    tt.actual_time = 30000.0
    _, latest_before = tt.now()
    uncertainty_before = (latest_before - tt.actual_time)

    tt.sync()
    _, latest_after = tt.now()
    uncertainty_after = (latest_after - tt.actual_time)

    print(f"\n  Before GPS re-sync: uncertainty = {uncertainty_before:.4f}ms")
    print(f"  After GPS re-sync:  uncertainty = {uncertainty_after:.4f}ms")
    print()


# === Main ===

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
