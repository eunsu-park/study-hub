"""
Exercises for Lesson 19: Raft Implementation Part 2
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import time
import json
import random
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from itertools import combinations


# === Exercise 1: Membership Change Safety Proof ===
def exercise_1_membership_safety():
    """
    Prove that single-server membership changes cannot create
    two disjoint majorities.

    Key insight: Any two configurations that differ by at most 1 member
    have overlapping majorities.
    """
    print("=== Exercise 1: Membership Change Safety ===\n")

    def check_overlap(config_old: set, config_new: set) -> bool:
        """Verify all majority pairs overlap."""
        majority_old = len(config_old) // 2 + 1
        majority_new = len(config_new) // 2 + 1

        for q_old in combinations(config_old, majority_old):
            for q_new in combinations(config_new, majority_new):
                if not (set(q_old) & set(q_new)):
                    return False
        return True

    # Test: add one server
    old = {"A", "B", "C"}
    new = {"A", "B", "C", "D"}
    overlap = check_overlap(old, new)
    print(f"  {old} → {new}: overlapping={overlap}")

    # Test: remove one server
    old = {"A", "B", "C", "D", "E"}
    new = {"A", "B", "C", "D"}
    overlap = check_overlap(old, new)
    print(f"  {old} → {new}: overlapping={overlap}")

    # Test: add two servers (UNSAFE)
    old = {"A", "B", "C"}
    new = {"A", "B", "C", "D", "E"}
    overlap = check_overlap(old, new)
    print(f"  {old} → {new}: overlapping={overlap} (two-server change)")

    # Maximum safe sequential additions without waiting
    print(f"\n  Answer: You can safely add at most 1 server at a time.")
    print(f"  Each change must be committed before the next begins.")


exercise_1_membership_safety()


# === Exercise 2: Copy-on-Write Snapshot ===
class CopyOnWriteSnapshot:
    """
    Copy-on-write snapshot mechanism.

    Uses a versioned data structure that shares unchanged pages
    between the live state and the snapshot.
    """

    def __init__(self):
        self.data: Dict[str, Tuple[str, int]] = {}  # key → (value, version)
        self.version: int = 0
        self.snapshot_data: Optional[Dict[str, Tuple[str, int]]] = None
        self.snapshot_version: int = 0
        self.cow_pages: Dict[str, str] = {}  # Overwritten keys during snapshot

    def put(self, key: str, value: str):
        """Write a key-value pair with COW for active snapshot."""
        if self.snapshot_data is not None and key in self.data:
            # Copy-on-write: save old value for snapshot
            if key not in self.cow_pages:
                old_val, old_ver = self.data[key]
                if old_ver <= self.snapshot_version:
                    self.cow_pages[key] = old_val

        self.version += 1
        self.data[key] = (value, self.version)

    def get(self, key: str) -> Optional[str]:
        entry = self.data.get(key)
        return entry[0] if entry else None

    def start_snapshot(self) -> int:
        """Start a COW snapshot. Returns snapshot version."""
        self.snapshot_version = self.version
        self.snapshot_data = {}
        self.cow_pages = {}
        return self.snapshot_version

    def get_snapshot(self) -> Dict[str, str]:
        """Get the snapshot data (consistent point-in-time view)."""
        if self.snapshot_data is None:
            return {}

        result = {}
        for key, (value, ver) in self.data.items():
            if key in self.cow_pages:
                result[key] = self.cow_pages[key]
            elif ver <= self.snapshot_version:
                result[key] = value
        return result

    def finish_snapshot(self):
        """Complete the snapshot and release COW pages."""
        self.snapshot_data = None
        self.cow_pages = {}


def exercise_2():
    """Test copy-on-write snapshot."""
    print("\n=== Exercise 2: Copy-on-Write Snapshot ===\n")

    cow = CopyOnWriteSnapshot()
    cow.put("x", "1")
    cow.put("y", "2")
    cow.put("z", "3")

    # Start snapshot
    snap_ver = cow.start_snapshot()
    print(f"  Snapshot started at version {snap_ver}")

    # Modify data while snapshot is in progress
    cow.put("x", "10")  # Modified
    cow.put("w", "4")   # New key (not in snapshot)

    # Snapshot should reflect pre-modification state
    snapshot = cow.get_snapshot()
    print(f"  Snapshot: {snapshot}")
    print(f"  Live x={cow.get('x')}, snapshot x={snapshot.get('x')}")
    print(f"  COW pages: {cow.cow_pages}")

    cow.finish_snapshot()


exercise_2()


# === Exercise 3: ReadIndex Latency Calculation ===
def exercise_3():
    """
    Calculate read latency for different approaches.

    5-node cluster, 2ms network RTT.
    """
    print("\n=== Exercise 3: ReadIndex Latency ===\n")

    rtt = 2  # ms

    # Log Read: full consensus round
    # Propose → replicate to majority → commit → respond
    log_read = rtt  # One round trip for replication + local apply
    print(f"  Log Read: {log_read}ms (1 consensus RTT)")

    # ReadIndex: heartbeat confirmation
    # Send heartbeat to majority → receive acks → respond
    read_index = rtt  # One heartbeat round trip
    print(f"  ReadIndex: {read_index}ms (1 heartbeat RTT)")

    # LeaseRead with valid lease
    lease_read_valid = 0.01  # Local read, ~microseconds
    print(f"  LeaseRead (valid): ~{lease_read_valid}ms (local)")

    # LeaseRead with expired lease
    lease_read_expired = rtt  # Falls back to ReadIndex
    print(f"  LeaseRead (expired): {lease_read_expired}ms (fallback to ReadIndex)")


exercise_3()


# === Exercise 4: Batching Tradeoff ===
def exercise_4():
    """
    Calculate optimal batch size for 10,000 req/s workload.
    """
    print("\n=== Exercise 4: Batching Tradeoff ===\n")

    req_per_sec = 10000
    rtt = 1  # ms

    # Without batching
    no_batch_p99 = rtt  # Each request waits for 1 RTT
    no_batch_throughput = 1000 / rtt  # Max throughput = 1000/rtt per connection
    print(f"  Without batching:")
    print(f"    p99 latency: {no_batch_p99}ms")
    print(f"    RPCs/sec: {req_per_sec}")

    # With batching: batch_wait + RTT
    for batch_wait in [0.5, 1.0, 2.0, 5.0]:
        batch_size = req_per_sec * batch_wait / 1000
        rpcs_per_sec = req_per_sec / batch_size
        p99 = batch_wait + rtt
        print(f"  Batch wait={batch_wait}ms: "
              f"batch_size={batch_size:.0f}, "
              f"RPCs/sec={rpcs_per_sec:.0f}, "
              f"p99={p99:.1f}ms")

    print(f"\n  Optimal: batch_wait=1ms gives 10 entries/batch, "
          f"1000 RPCs/s (10x reduction), p99=2ms")


exercise_4()


# === Exercise 5: InstallSnapshot Implementation ===
def exercise_5():
    """Implement snapshot installation for a follower."""
    print("\n=== Exercise 5: InstallSnapshot ===\n")

    class RaftFollower:
        def __init__(self):
            self.log = [{"term": 1, "index": i, "cmd": f"old_{i}"}
                       for i in range(1, 11)]
            self.data = {f"key_{i}": f"val_{i}" for i in range(10)}
            self.commit_index = 8
            self.last_applied = 8

        def install_snapshot(self, last_included_index: int,
                           last_included_term: int,
                           snapshot_data: dict):
            """Install a snapshot from the leader."""
            # 1. Discard log entries covered by snapshot
            self.log = [
                e for e in self.log
                if e["index"] > last_included_index
            ]

            # 2. Restore state machine from snapshot
            self.data = dict(snapshot_data)

            # 3. Update indices
            self.commit_index = max(self.commit_index, last_included_index)
            self.last_applied = last_included_index

            return {
                "ok": True,
                "log_remaining": len(self.log),
                "data_keys": len(self.data),
            }

    follower = RaftFollower()
    print(f"  Before: log={len(follower.log)}, data={len(follower.data)}, "
          f"commit={follower.commit_index}")

    snapshot = {f"key_{i}": f"new_val_{i}" for i in range(20)}
    result = follower.install_snapshot(
        last_included_index=15,
        last_included_term=3,
        snapshot_data=snapshot,
    )

    print(f"  After: {result}")
    print(f"  commit_index={follower.commit_index}, "
          f"last_applied={follower.last_applied}")


exercise_5()


if __name__ == "__main__":
    print("\nAll exercises completed.")
