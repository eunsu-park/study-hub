"""
Exercises for Lesson 25: Vector Clocks and Causality Tracking
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

from typing import Dict, Optional, Tuple
from copy import deepcopy


# === Helper: Vector Clock Operations ===
def vc_inc(vc: dict, proc: str) -> dict:
    r = dict(vc)
    r[proc] = r.get(proc, 0) + 1
    return r

def vc_merge(vc1: dict, vc2: dict) -> dict:
    keys = set(vc1) | set(vc2)
    return {k: max(vc1.get(k, 0), vc2.get(k, 0)) for k in keys}

def vc_send(vc: dict, proc: str) -> dict:
    return vc_inc(vc, proc)

def vc_receive(vc: dict, received: dict, proc: str) -> dict:
    merged = vc_merge(vc, received)
    return vc_inc(merged, proc)

def vc_compare(a: dict, b: dict) -> str:
    keys = set(a) | set(b)
    a_less = any(a.get(k, 0) < b.get(k, 0) for k in keys)
    b_less = any(b.get(k, 0) < a.get(k, 0) for k in keys)
    if a == b: return "equal"
    if a_less and not b_less: return "before"
    if b_less and not a_less: return "after"
    return "concurrent"


# === Exercise 1: Vector Clock Computation ===
def exercise_1():
    """
    Compute vector timestamps for all events.

    P1: send to P2, local, send to P3
    P2: receive from P1, send to P3, local
    P3: local, receive from P1, receive from P2
    """
    print("=== Exercise 1: Vector Clock Computation ===\n")

    p1 = {"P1": 0, "P2": 0, "P3": 0}
    p2 = {"P1": 0, "P2": 0, "P3": 0}
    p3 = {"P1": 0, "P2": 0, "P3": 0}

    # P1: send to P2
    p1 = vc_send(p1, "P1")
    msg1 = dict(p1)
    print(f"  P1 send to P2:     {p1}")

    # P3: local (concurrent with P1's send)
    p3 = vc_inc(p3, "P3")
    print(f"  P3 local:          {p3}")

    # P2: receive from P1
    p2 = vc_receive(p2, msg1, "P2")
    print(f"  P2 recv from P1:   {p2}")

    # P1: local
    p1 = vc_inc(p1, "P1")
    print(f"  P1 local:          {p1}")

    # P2: send to P3
    p2 = vc_send(p2, "P2")
    msg2 = dict(p2)
    print(f"  P2 send to P3:     {p2}")

    # P1: send to P3
    p1 = vc_send(p1, "P1")
    msg3 = dict(p1)
    print(f"  P1 send to P3:     {p1}")

    # P3: receive from P1
    p3 = vc_receive(p3, msg3, "P3")
    print(f"  P3 recv from P1:   {p3}")

    # P3: receive from P2
    p3 = vc_receive(p3, msg2, "P3")
    print(f"  P3 recv from P2:   {p3}")

    # P2: local
    p2 = vc_inc(p2, "P2")
    print(f"  P2 local:          {p2}")


exercise_1()


# === Exercise 2: Conflict Detection ===
def exercise_2():
    """
    Detect conflict between VV_A = {R1:3, R2:1} and VV_B = {R1:2, R2:2}.
    """
    print("\n=== Exercise 2: Conflict Detection ===\n")

    vv_a = {"R1": 3, "R2": 1}
    vv_b = {"R1": 2, "R2": 2}

    result = vc_compare(vv_a, vv_b)
    print(f"  VV_A = {vv_a}")
    print(f"  VV_B = {vv_b}")
    print(f"  Relation: {result}")

    if result == "concurrent":
        print(f"\n  CONFLICT! Need application-level resolution.")
        print(f"  Shopping cart merge strategy: SET UNION")

        cart_a = {"widget", "gadget", "bolt"}
        cart_b = {"widget", "spring", "nut"}
        merged = cart_a | cart_b
        print(f"    Cart A: {cart_a}")
        print(f"    Cart B: {cart_b}")
        print(f"    Merged: {merged}")
        print(f"    (All items preserved — no data loss)")


exercise_2()


# === Exercise 3: HLC Bounds ===
def exercise_3():
    """
    Prove HLC timestamps are within max_clock_skew of real time.
    """
    print("\n=== Exercise 3: HLC Bounds ===\n")

    print("  Proof sketch:")
    print("    HLC.l = max(HLC.l, physical_time)")
    print("    Therefore: HLC.l >= physical_time (always)")
    print("    On receive: HLC.l = max(HLC.l, remote.l, physical_time)")
    print("    Remote.l <= remote_physical_time + epsilon (by induction)")
    print("    HLC.l <= max(physical_time, remote_physical_time + epsilon)")
    print("    <= physical_time + epsilon (within clock skew)")
    print("    Therefore: physical_time <= HLC.l <= physical_time + epsilon")
    print()
    print("  What if clock jumps backward by > epsilon?")
    print("    HLC.l stays at old (higher) value")
    print("    HLC.c increments on each event")
    print("    HLC.l - physical_time > epsilon → VIOLATION of bounds")
    print("    Mitigation: refuse to operate if |HLC.l - physical_time| > max_skew")
    print("    CockroachDB halts if clock offset exceeds threshold")


exercise_3()


# === Exercise 4: Replicated KV Store ===
def exercise_4():
    """Build replicated KV store with version vectors."""
    print("\n=== Exercise 4: Replicated KV with DVVs ===\n")

    class ReplicaKV:
        def __init__(self, rid):
            self.rid = rid
            self.data = {}  # key → [(value, vv)]
            self.counter = 0

        def put(self, key, value, context=None):
            self.counter += 1
            vv = dict(context or {})
            vv[self.rid] = self.counter
            if key not in self.data:
                self.data[key] = []
            # Remove dominated versions
            self.data[key] = [
                (v, old_vv) for v, old_vv in self.data[key]
                if vc_compare(old_vv, vv) != "before" and old_vv != vv
            ]
            self.data[key].append((value, vv))

        def get(self, key):
            return self.data.get(key, [])

        def sync(self, other, key):
            remote = other.get(key)
            local = self.get(key)
            all_versions = local + remote
            # Keep non-dominated
            result = []
            for v, vv in all_versions:
                dominated = any(
                    vc_compare(vv, other_vv) == "before"
                    for _, other_vv in all_versions if other_vv != vv
                )
                if not dominated and not any(vv == r_vv for _, r_vv in result):
                    result.append((v, vv))
            self.data[key] = result

    r1 = ReplicaKV("R1")
    r2 = ReplicaKV("R2")

    # R1 writes
    r1.put("x", "hello")
    print(f"  R1 writes x='hello': {r1.get('x')}")

    # Sync to R2
    r2.sync(r1, "x")
    print(f"  R2 after sync: {r2.get('x')}")

    # Concurrent writes
    r1.put("x", "world", context=r1.get("x")[0][1] if r1.get("x") else None)
    r2.put("x", "earth", context=r2.get("x")[0][1] if r2.get("x") else None)

    print(f"  R1 writes x='world': {[(v, vv) for v, vv in r1.get('x')]}")
    print(f"  R2 writes x='earth': {[(v, vv) for v, vv in r2.get('x')]}")

    # Sync — detect conflict
    r1.sync(r2, "x")
    versions = r1.get("x")
    print(f"  R1 after sync: {len(versions)} versions (conflict={'yes' if len(versions) > 1 else 'no'})")
    for v, vv in versions:
        print(f"    '{v}': {vv}")


exercise_4()


# === Exercise 5: Vector Clock Pruning ===
def exercise_5():
    """Design scheme to prune VV entries for removed processes."""
    print("\n=== Exercise 5: Vector Clock Pruning ===\n")

    print("  Problem: VV grows with N (number of processes ever seen)")
    print()
    print("  Pruning strategy:")
    print("    1. Maintain a 'retired' set of permanently removed processes")
    print("    2. When retiring process P:")
    print("       a. Wait until all entries causally after P's last event are committed")
    print("       b. Record max(VV[P]) across all replicas as 'base_version'")
    print("       c. Remove P from all VVs")
    print("    3. Safety guarantee maintained:")
    print("       - All events from P are already ordered and committed")
    print("       - No future events will reference P")
    print("       - Concurrent detection for P's events is no longer needed")
    print("    4. Cannot maintain: detecting concurrency with events")
    print("       that happened BEFORE P was retired (already resolved)")

    # Demonstration
    vv = {"P1": 5, "P2": 3, "P3": 8, "P4": 2}
    print(f"\n  Before pruning P4: {vv}")
    del vv["P4"]
    print(f"  After pruning P4:  {vv}")
    print(f"  Size reduced: 4 → {len(vv)} entries")


exercise_5()


if __name__ == "__main__":
    print("\nAll exercises completed.")
