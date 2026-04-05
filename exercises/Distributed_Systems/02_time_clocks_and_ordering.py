"""
Exercises for Lesson 02: Time, Clocks, and Ordering
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import time
from typing import Dict, List, Optional, Set, Tuple


# === Exercise 1: Hybrid Logical Clock (HLC) ===
# Problem: Implement a Hybrid Logical Clock that combines a physical clock
# with a logical component. HLC must satisfy:
# 1. e hb f => hlc(e) < hlc(f)  (captures causality)
# 2. |hlc.pt - pt| is bounded     (stays close to physical time)

class HybridLogicalClock:
    """
    Hybrid Logical Clock (HLC) implementation.

    Each HLC timestamp is (physical_component, logical_component, node_id).
    - physical_component (pt): tracks physical time but never goes backward.
    - logical_component (l): disambiguates events at the same physical time.
    """

    def __init__(self, node_id: str, physical_clock=None):
        self.node_id = node_id
        self.pt = 0  # physical component
        self.l = 0   # logical component
        self._physical_clock = physical_clock or (lambda: int(time.time() * 1000))

    def now(self) -> Tuple[int, int, str]:
        """
        Generate a timestamp for a local event.
        """
        physical_now = self._physical_clock()
        if physical_now > self.pt:
            self.pt = physical_now
            self.l = 0
        else:
            self.l += 1
        return (self.pt, self.l, self.node_id)

    def receive(self, remote_pt: int, remote_l: int) -> Tuple[int, int, str]:
        """
        Generate a timestamp upon receiving a message with remote HLC.
        """
        physical_now = self._physical_clock()
        old_pt = self.pt

        if physical_now > old_pt and physical_now > remote_pt:
            self.pt = physical_now
            self.l = 0
        elif old_pt > remote_pt:
            # Local pt is ahead
            self.l += 1
        elif remote_pt > old_pt:
            self.pt = remote_pt
            self.l = remote_l + 1
        else:
            # old_pt == remote_pt
            self.l = max(self.l, remote_l) + 1

        return (self.pt, self.l, self.node_id)

    @staticmethod
    def compare(ts1: Tuple[int, int, str], ts2: Tuple[int, int, str]) -> int:
        """Compare two HLC timestamps. Returns -1, 0, or 1."""
        if ts1[0] != ts2[0]:
            return -1 if ts1[0] < ts2[0] else 1
        if ts1[1] != ts2[1]:
            return -1 if ts1[1] < ts2[1] else 1
        if ts1[2] != ts2[2]:
            return -1 if ts1[2] < ts2[2] else 1
        return 0


def exercise_1():
    """
    Demonstrate HLC maintaining causality while staying close to
    physical time.
    """
    print("=== Exercise 1: Hybrid Logical Clock ===\n")

    # Simulated physical clock that we can control
    clock_time = [100]

    def fake_clock():
        return clock_time[0]

    hlc_a = HybridLogicalClock("A", fake_clock)
    hlc_b = HybridLogicalClock("B", fake_clock)

    # Event 1: A sends at time 100
    ts1 = hlc_a.now()
    print(f"A local event:       pt={ts1[0]}, l={ts1[1]}")

    # Event 2: A sends again at same physical time
    ts2 = hlc_a.now()
    print(f"A local event:       pt={ts2[0]}, l={ts2[1]}")
    assert HybridLogicalClock.compare(ts1, ts2) == -1, "ts1 < ts2"

    # Event 3: B receives A's message (ts2), physical time still 100
    ts3 = hlc_b.receive(ts2[0], ts2[1])
    print(f"B receive from A:    pt={ts3[0]}, l={ts3[1]}")
    assert HybridLogicalClock.compare(ts2, ts3) == -1, "ts2 < ts3"

    # Event 4: Physical time advances to 200
    clock_time[0] = 200
    ts4 = hlc_b.now()
    print(f"B local (time=200):  pt={ts4[0]}, l={ts4[1]}")
    assert ts4[0] == 200 and ts4[1] == 0, "Should reset logical on new physical"

    print("\nAll causality checks passed.")
    print(f"HLC stayed within {max(ts1[0], ts2[0], ts3[0]) - 100}ms of physical time.")
    print()


# === Exercise 2: Detect Causal Violations with Vector Clocks ===
# Problem: Given a set of events with vector clocks, detect all pairs
# of events where one claims to causally follow the other but the
# vector clock does not support this claim.

class VectorClock:
    """Simple vector clock implementation."""

    def __init__(self, node_id: str, size: int):
        self.node_id = node_id
        self.clock: Dict[str, int] = {}

    def increment(self):
        self.clock[self.node_id] = self.clock.get(self.node_id, 0) + 1

    def merge(self, other_clock: Dict[str, int]):
        for node, ts in other_clock.items():
            self.clock[node] = max(self.clock.get(node, 0), ts)
        self.increment()

    def snapshot(self) -> Dict[str, int]:
        return dict(self.clock)

    @staticmethod
    def happens_before(vc1: Dict[str, int], vc2: Dict[str, int]) -> bool:
        """Returns True if vc1 -> vc2 (vc1 causally precedes vc2)."""
        all_keys = set(vc1.keys()) | set(vc2.keys())
        at_least_one_less = False
        for k in all_keys:
            v1 = vc1.get(k, 0)
            v2 = vc2.get(k, 0)
            if v1 > v2:
                return False
            if v1 < v2:
                at_least_one_less = True
        return at_least_one_less

    @staticmethod
    def concurrent(vc1: Dict[str, int], vc2: Dict[str, int]) -> bool:
        """Returns True if vc1 || vc2 (concurrent)."""
        return (
            not VectorClock.happens_before(vc1, vc2)
            and not VectorClock.happens_before(vc2, vc1)
            and vc1 != vc2
        )


def detect_causal_violations(
    events: List[Tuple[str, Dict[str, int]]],
    claimed_order: List[Tuple[int, int]],
) -> List[Tuple[int, int, str]]:
    """
    Detect causal violations.

    Args:
        events: List of (event_name, vector_clock_snapshot).
        claimed_order: List of (i, j) meaning event i is claimed to
            causally precede event j.

    Returns:
        List of (i, j, reason) for each violation found.
    """
    violations = []
    for i, j in claimed_order:
        vc_i = events[i][1]
        vc_j = events[j][1]
        if not VectorClock.happens_before(vc_i, vc_j):
            if VectorClock.concurrent(vc_i, vc_j):
                reason = "events are concurrent, not causally related"
            elif VectorClock.happens_before(vc_j, vc_i):
                reason = "causal order is reversed"
            else:
                reason = "same vector clock (same event?)"
            violations.append((i, j, reason))
    return violations


def exercise_2():
    """
    Detect causal violations from vector clock analysis.
    """
    print("=== Exercise 2: Detect Causal Violations ===\n")

    events = [
        ("e0", {"A": 1, "B": 0, "C": 0}),
        ("e1", {"A": 1, "B": 1, "C": 0}),
        ("e2", {"A": 0, "B": 0, "C": 1}),
        ("e3", {"A": 2, "B": 1, "C": 1}),
        ("e4", {"A": 1, "B": 2, "C": 0}),
    ]

    # Claims: (i, j) means "e_i happened before e_j"
    claimed_order = [
        (0, 1),  # e0 -> e1: valid (A:1 < A:1,B:1)
        (0, 3),  # e0 -> e3: valid
        (2, 1),  # e2 -> e1: VIOLATION (concurrent)
        (4, 0),  # e4 -> e0: VIOLATION (reversed)
        (1, 3),  # e1 -> e3: valid
    ]

    print("Events:")
    for i, (name, vc) in enumerate(events):
        print(f"  {i}: {name} = {vc}")

    print("\nClaimed causal ordering:")
    for i, j in claimed_order:
        print(f"  {events[i][0]} -> {events[j][0]}")

    violations = detect_causal_violations(events, claimed_order)

    print(f"\nViolations found: {len(violations)}")
    for i, j, reason in violations:
        print(f"  {events[i][0]} -> {events[j][0]}: {reason}")

    assert len(violations) == 2, "Should detect exactly 2 violations"
    print("\nAll checks passed.")
    print()


# === Exercise 3: Version Vector for Replicated KV Store ===
# Problem: Implement a version vector-based replicated key-value store.
# Each replica tracks a version vector per key. On write, the local
# entry increments. On sync, version vectors are merged and conflicts
# are detected.

class VersionedValue:
    """A value with its version vector."""

    def __init__(self, value: any, version: Dict[str, int]):
        self.value = value
        self.version = dict(version)

    def __repr__(self):
        return f"VV({self.value}, {self.version})"


class ReplicatedKVStore:
    """
    A replicated key-value store using version vectors for
    conflict detection.
    """

    def __init__(self, replica_id: str):
        self.replica_id = replica_id
        self.store: Dict[str, VersionedValue] = {}

    def put(self, key: str, value: any):
        """Write a key-value pair, incrementing local version."""
        if key in self.store:
            vv = dict(self.store[key].version)
        else:
            vv = {}
        vv[self.replica_id] = vv.get(self.replica_id, 0) + 1
        self.store[key] = VersionedValue(value, vv)

    def get(self, key: str) -> Optional[VersionedValue]:
        return self.store.get(key)

    def sync_key(
        self, key: str, remote_vv: VersionedValue
    ) -> Tuple[str, Optional[VersionedValue]]:
        """
        Sync a single key from a remote replica.

        Returns:
            ("applied", value) if remote is newer
            ("conflict", None) if concurrent writes detected
            ("ignored", None) if local is newer or equal
        """
        local = self.store.get(key)
        if local is None:
            self.store[key] = VersionedValue(remote_vv.value, dict(remote_vv.version))
            return ("applied", self.store[key])

        local_v = local.version
        remote_v = remote_vv.version

        if VectorClock.happens_before(local_v, remote_v):
            self.store[key] = VersionedValue(
                remote_vv.value, dict(remote_vv.version)
            )
            return ("applied", self.store[key])
        elif VectorClock.happens_before(remote_v, local_v):
            return ("ignored", None)
        elif local_v == remote_v:
            return ("ignored", None)
        else:
            return ("conflict", None)


def exercise_3():
    """
    Demonstrate version vectors detecting conflicts in a replicated
    key-value store.
    """
    print("=== Exercise 3: Version Vector KV Store ===\n")

    r1 = ReplicatedKVStore("R1")
    r2 = ReplicatedKVStore("R2")

    # R1 writes x=10
    r1.put("x", 10)
    print(f"R1 writes x=10: {r1.get('x')}")

    # Sync R1 -> R2 (should apply)
    result, _ = r2.sync_key("x", r1.get("x"))
    print(f"R2 syncs from R1: {result}, R2.x = {r2.get('x')}")

    # Both write concurrently (no sync in between)
    r1.put("x", 20)
    r2.put("x", 30)
    print(f"\nConcurrent writes:")
    print(f"  R1 writes x=20: {r1.get('x')}")
    print(f"  R2 writes x=30: {r2.get('x')}")

    # Try to sync - should detect conflict
    result, _ = r2.sync_key("x", r1.get("x"))
    print(f"\nR2 syncs R1's x=20: {result}")
    assert result == "conflict", "Should detect conflict"

    # R1 syncs R2's update - also conflict
    result, _ = r1.sync_key("x", r2.get("x"))
    print(f"R1 syncs R2's x=30: {result}")
    assert result == "conflict", "Should detect conflict"

    print("\nConflict detection working correctly.")
    print()


# === Main ===

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
