"""
G-Counter and PN-Counter CRDT Implementations

Conflict-free Replicated Data Types (CRDTs) are data structures that can be
replicated across multiple nodes and always converge to a consistent state,
without requiring consensus. This module implements two counter CRDTs:

- G-Counter (Grow-only Counter): supports only increment, converges via max
- PN-Counter (Positive-Negative Counter): supports increment and decrement
  by combining two G-Counters

Key CRDT properties verified:
- Commutativity: merge(a, b) == merge(b, a)
- Associativity: merge(merge(a, b), c) == merge(a, merge(b, c))
- Idempotence:   merge(a, a) == a

Usage:
    python 05_crdt_counters.py
"""

from __future__ import annotations

from copy import deepcopy


class GCounter:
    """
    Grow-only Counter CRDT.

    Each replica maintains a vector of counts (one per replica ID).
    The counter value is the sum of all entries. Merge takes the
    element-wise maximum.
    """

    def __init__(self, replica_id: str, num_replicas: int,
                 replica_ids: list[str] | None = None):
        self.replica_id = replica_id
        if replica_ids is not None:
            self.counts: dict[str, int] = {rid: 0 for rid in replica_ids}
        else:
            self.counts = {replica_id: 0}

    def increment(self, amount: int = 1) -> None:
        """Increment this replica's counter by the given amount."""
        if amount < 0:
            raise ValueError("G-Counter only supports non-negative increments. "
                             "Use PN-Counter for decrements.")
        self.counts[self.replica_id] = self.counts.get(self.replica_id, 0) + amount

    def value(self) -> int:
        """Return the current counter value (sum of all replicas)."""
        return sum(self.counts.values())

    def merge(self, other: GCounter) -> GCounter:
        """
        Merge with another G-Counter. Takes element-wise max.
        Returns a new G-Counter (does not mutate self).
        """
        result = deepcopy(self)
        for rid, count in other.counts.items():
            result.counts[rid] = max(result.counts.get(rid, 0), count)
        return result

    def merge_in_place(self, other: GCounter) -> None:
        """Merge in-place (mutates self)."""
        for rid, count in other.counts.items():
            self.counts[rid] = max(self.counts.get(rid, 0), count)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GCounter):
            return NotImplemented
        return self.counts == other.counts

    def __repr__(self) -> str:
        entries = ", ".join(f"{k}:{v}" for k, v in sorted(self.counts.items()))
        return f"GCounter({self.replica_id}, value={self.value()}, [{entries}])"


class PNCounter:
    """
    Positive-Negative Counter CRDT.

    Uses two G-Counters: one for increments (P) and one for decrements (N).
    The value is P.value() - N.value().
    """

    def __init__(self, replica_id: str, replica_ids: list[str]):
        self.replica_id = replica_id
        self.p = GCounter(replica_id, len(replica_ids), replica_ids)
        self.n = GCounter(replica_id, len(replica_ids), replica_ids)

    def increment(self, amount: int = 1) -> None:
        """Increment the counter."""
        self.p.increment(amount)

    def decrement(self, amount: int = 1) -> None:
        """Decrement the counter."""
        self.n.increment(amount)

    def value(self) -> int:
        """Return the current counter value (P - N)."""
        return self.p.value() - self.n.value()

    def merge(self, other: PNCounter) -> PNCounter:
        """Merge with another PN-Counter. Returns a new PN-Counter."""
        result = PNCounter(self.replica_id,
                           list(self.p.counts.keys()))
        result.p = self.p.merge(other.p)
        result.n = self.n.merge(other.n)
        return result

    def merge_in_place(self, other: PNCounter) -> None:
        """Merge in-place."""
        self.p.merge_in_place(other.p)
        self.n.merge_in_place(other.n)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PNCounter):
            return NotImplemented
        return self.p == other.p and self.n == other.n

    def __repr__(self) -> str:
        return (f"PNCounter({self.replica_id}, value={self.value()}, "
                f"P={self.p.value()}, N={self.n.value()})")


def demo_g_counter() -> None:
    """Demonstrate G-Counter with 3 replicas."""
    print("=" * 65)
    print("G-Counter Demo: 3 Replicas")
    print("=" * 65)

    rids = ["A", "B", "C"]
    a = GCounter("A", 3, rids)
    b = GCounter("B", 3, rids)
    c = GCounter("C", 3, rids)

    # Each replica increments independently
    print("\nPhase 1: Independent increments")
    a.increment(3)
    b.increment(5)
    c.increment(2)
    print(f"  {a}")
    print(f"  {b}")
    print(f"  {c}")

    # More increments
    a.increment(1)
    b.increment(2)
    print(f"\nAfter more increments:")
    print(f"  {a}")
    print(f"  {b}")
    print(f"  {c}")

    # Simulate network partition: A and B can communicate, C is isolated
    print("\n--- Network partition: {A, B} | {C} ---")

    # A and B merge
    a.merge_in_place(b)
    b.merge_in_place(a)
    print(f"  After A <-> B merge:")
    print(f"  {a}  (value={a.value()})")
    print(f"  {b}  (value={b.value()})")
    print(f"  {c}  (value={c.value()})  [still isolated]")

    # C does more work while isolated
    c.increment(10)
    print(f"\n  C increments by 10 while isolated:")
    print(f"  {c}  (value={c.value()})")

    # Partition heals — all replicas merge
    print("\n--- Partition healed ---")
    a.merge_in_place(c)
    b.merge_in_place(a)
    c.merge_in_place(a)
    print(f"  After full merge:")
    print(f"  {a}  (value={a.value()})")
    print(f"  {b}  (value={b.value()})")
    print(f"  {c}  (value={c.value()})")

    assert a.value() == b.value() == c.value()
    print(f"\n  All replicas converged to value = {a.value()}")


def demo_pn_counter() -> None:
    """Demonstrate PN-Counter with increments and decrements."""
    print("\n" + "=" * 65)
    print("PN-Counter Demo: 3 Replicas with Increments and Decrements")
    print("=" * 65)

    rids = ["X", "Y", "Z"]
    x = PNCounter("X", rids)
    y = PNCounter("Y", rids)
    z = PNCounter("Z", rids)

    # Concurrent operations
    print("\nPhase 1: Concurrent operations")
    x.increment(10)
    y.increment(5)
    y.decrement(3)
    z.increment(8)
    z.decrement(2)

    print(f"  {x}")
    print(f"  {y}")
    print(f"  {z}")

    # Merge all
    print("\nPhase 2: Merge all replicas")
    x.merge_in_place(y)
    x.merge_in_place(z)
    y.merge_in_place(x)
    z.merge_in_place(x)

    print(f"  {x}")
    print(f"  {y}")
    print(f"  {z}")

    expected = (10 + 5 + 8) - (3 + 2)
    print(f"\n  Expected: (10+5+8) - (3+2) = {expected}")
    print(f"  All replicas: {x.value()}, {y.value()}, {z.value()}")
    assert x.value() == y.value() == z.value() == expected
    print("  Converged correctly!")


def verify_crdt_properties() -> None:
    """Verify the mathematical properties that make CRDTs work."""
    print("\n" + "=" * 65)
    print("Verifying CRDT Properties")
    print("=" * 65)

    rids = ["A", "B", "C"]

    # Create three distinct counters
    a = GCounter("A", 3, rids)
    b = GCounter("B", 3, rids)
    c = GCounter("C", 3, rids)
    a.increment(3)
    b.increment(5)
    c.increment(7)

    # 1. Commutativity: merge(a, b) == merge(b, a)
    ab = a.merge(b)
    ba = b.merge(a)
    commutative = (ab == ba)
    print(f"\n  1. Commutativity: merge(a,b) == merge(b,a)")
    print(f"     merge(a,b) = {ab}")
    print(f"     merge(b,a) = {ba}")
    print(f"     Result: {'PASS' if commutative else 'FAIL'}")

    # 2. Associativity: merge(merge(a, b), c) == merge(a, merge(b, c))
    ab_c = a.merge(b).merge(c)
    a_bc = a.merge(b.merge(c))
    associative = (ab_c == a_bc)
    print(f"\n  2. Associativity: merge(merge(a,b), c) == merge(a, merge(b,c))")
    print(f"     merge(merge(a,b), c) = {ab_c}")
    print(f"     merge(a, merge(b,c)) = {a_bc}")
    print(f"     Result: {'PASS' if associative else 'FAIL'}")

    # 3. Idempotence: merge(a, a) == a
    aa = a.merge(a)
    idempotent = (aa == a)
    print(f"\n  3. Idempotence: merge(a, a) == a")
    print(f"     a          = {a}")
    print(f"     merge(a,a) = {aa}")
    print(f"     Result: {'PASS' if idempotent else 'FAIL'}")

    # Also verify for PN-Counter
    print(f"\n  --- Same checks for PN-Counter ---")
    pa = PNCounter("A", rids)
    pb = PNCounter("B", rids)
    pc = PNCounter("C", rids)
    pa.increment(10); pa.decrement(2)
    pb.increment(5);  pb.decrement(1)
    pc.increment(8);  pc.decrement(3)

    pab = pa.merge(pb)
    pba = pb.merge(pa)
    print(f"  Commutativity: {'PASS' if pab == pba else 'FAIL'}")

    pab_c = pa.merge(pb).merge(pc)
    pa_bc = pa.merge(pb.merge(pc))
    print(f"  Associativity: {'PASS' if pab_c == pa_bc else 'FAIL'}")

    paa = pa.merge(pa)
    print(f"  Idempotence:   {'PASS' if paa == pa else 'FAIL'}")

    all_pass = commutative and associative and idempotent
    print(f"\n  All CRDT properties verified: {'PASS' if all_pass else 'FAIL'}")


def demo_convergence_under_partition() -> None:
    """Show that replicas converge even after extended partition."""
    print("\n" + "=" * 65)
    print("Convergence Demo: Extended Partition and Recovery")
    print("=" * 65)

    rids = ["R1", "R2", "R3"]
    r1 = PNCounter("R1", rids)
    r2 = PNCounter("R2", rids)
    r3 = PNCounter("R3", rids)

    # Phase 1: All connected, initial operations
    print("\nPhase 1: All connected")
    r1.increment(100)
    r2.increment(50)
    r3.increment(25)

    # Full sync
    for r in [r1, r2, r3]:
        for other in [r1, r2, r3]:
            if r is not other:
                r.merge_in_place(other)

    print(f"  All replicas: {r1.value()}, {r2.value()}, {r3.value()}")
    assert r1.value() == r2.value() == r3.value() == 175

    # Phase 2: Partition into {R1} and {R2, R3}
    print("\nPhase 2: Partition {R1} | {R2, R3}")
    # R1 does operations alone
    r1.increment(20)
    r1.decrement(5)

    # R2 and R3 do operations and sync with each other
    r2.increment(10)
    r3.decrement(30)
    r2.merge_in_place(r3)
    r3.merge_in_place(r2)

    print(f"  R1 (isolated): {r1.value()}")
    print(f"  R2 (synced with R3): {r2.value()}")
    print(f"  R3 (synced with R2): {r3.value()}")
    print(f"  Values differ: R1={r1.value()}, R2=R3={r2.value()}")

    # Phase 3: Heal partition
    print("\nPhase 3: Partition healed")
    for r in [r1, r2, r3]:
        for other in [r1, r2, r3]:
            if r is not other:
                r.merge_in_place(other)

    print(f"  R1: {r1.value()}")
    print(f"  R2: {r2.value()}")
    print(f"  R3: {r3.value()}")
    assert r1.value() == r2.value() == r3.value()
    print(f"  Converged! Final value = {r1.value()}")
    print(f"  Expected: 175 + 20 - 5 + 10 - 30 = {175 + 20 - 5 + 10 - 30}")


if __name__ == "__main__":
    demo_g_counter()
    demo_pn_counter()
    verify_crdt_properties()
    demo_convergence_under_partition()
    print("\nDone.")
