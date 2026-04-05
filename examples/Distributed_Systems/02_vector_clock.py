"""
Vector Clock Implementation with Causality Detection

Vector clocks extend Lamport clocks to capture true concurrency. Each process
maintains a vector of counters (one per process). By comparing vectors, we can
determine if two events are causally related or truly concurrent.

Key concepts:
- Vector clock comparison: <=, <, ||  (concurrent)
- Happens-before: VC(a) < VC(b) iff forall i: a[i] <= b[i] and exists j: a[j] < b[j]
- Concurrent:     VC(a) || VC(b) iff neither a < b nor b < a
- Merge on receive: max element-wise, then increment own position
- Advantage over Lamport: can detect concurrency (Lamport cannot)

Usage:
    python 02_vector_clock.py
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum


class Relation(Enum):
    """Causal relationship between two events."""
    HAPPENS_BEFORE = "happens-before (->)"
    HAPPENS_AFTER = "happens-after (<-)"
    CONCURRENT = "concurrent (||)"
    EQUAL = "equal (=)"


class VectorClock:
    """
    A vector clock for a system with a known number of processes.

    The clock is a list of integers, one per process. Index i holds the
    logical time for process i as known by the owner of this clock.
    """

    def __init__(self, num_processes: int, process_id: int):
        self.vec = [0] * num_processes
        self.pid = process_id

    def increment(self) -> None:
        """Increment own position (on any local event)."""
        self.vec[self.pid] += 1

    def send(self) -> list[int]:
        """
        Record a send event: increment own clock, return a copy of the
        vector to attach to the message.
        """
        self.increment()
        return list(self.vec)

    def receive(self, msg_vec: list[int]) -> None:
        """
        Record a receive event: merge with the incoming vector
        (element-wise max), then increment own position.
        """
        for i in range(len(self.vec)):
            self.vec[i] = max(self.vec[i], msg_vec[i])
        self.increment()

    def snapshot(self) -> list[int]:
        """Return a copy of the current vector."""
        return list(self.vec)

    @staticmethod
    def compare(a: list[int], b: list[int]) -> Relation:
        """
        Compare two vector timestamps.
        Returns the causal relation of a with respect to b.
        """
        leq = all(ai <= bi for ai, bi in zip(a, b))  # a <= b
        geq = all(ai >= bi for ai, bi in zip(a, b))  # a >= b

        if leq and geq:
            return Relation.EQUAL
        if leq:
            return Relation.HAPPENS_BEFORE
        if geq:
            return Relation.HAPPENS_AFTER
        return Relation.CONCURRENT

    def __repr__(self) -> str:
        return f"VC(P{self.pid}, {self.vec})"


@dataclass
class VCEvent:
    """An event recorded with its vector timestamp."""
    process_id: int
    event_type: str         # "internal", "send", "receive"
    description: str
    timestamp: list[int]    # Snapshot of vector clock at event time
    partner: int = -1


class VCProcess:
    """A process using a vector clock."""

    def __init__(self, pid: int, name: str, num_processes: int):
        self.pid = pid
        self.name = name
        self.vc = VectorClock(num_processes, pid)
        self.events: list[VCEvent] = []

    def internal_event(self, description: str) -> VCEvent:
        """Record an internal event."""
        self.vc.increment()
        event = VCEvent(self.pid, "internal", description, self.vc.snapshot())
        self.events.append(event)
        return event

    def send(self, description: str, receiver_id: int) -> tuple[VCEvent, list[int]]:
        """Send a message; returns event and vector to attach to message."""
        msg_vec = self.vc.send()
        event = VCEvent(self.pid, "send", description, self.vc.snapshot(),
                        partner=receiver_id)
        self.events.append(event)
        return event, msg_vec

    def receive(self, description: str, sender_id: int,
                msg_vec: list[int]) -> VCEvent:
        """Receive a message and merge clocks."""
        self.vc.receive(msg_vec)
        event = VCEvent(self.pid, "receive", description, self.vc.snapshot(),
                        partner=sender_id)
        self.events.append(event)
        return event


def fmt_vec(v: list[int]) -> str:
    """Format a vector timestamp for display."""
    return f"[{', '.join(str(x) for x in v)}]"


def print_event(proc: VCProcess, event: VCEvent) -> None:
    """Pretty-print an event with its vector timestamp."""
    arrow = {"internal": "  *", "send": " ->", "receive": " <-"}[event.event_type]
    partner_info = f" (peer=P{event.partner})" if event.partner >= 0 else ""
    print(f"  {fmt_vec(event.timestamp):>15} P{event.process_id}({proc.name}) "
          f"{arrow} {event.description}{partner_info}")


def simulate_scenario() -> tuple[list[VCProcess], list[VCEvent]]:
    """
    Simulate 3 processes with a mix of causal and concurrent events.

    Timeline (conceptual):
        P0: e0_1(internal) -> e0_2(send to P1) ---------> e0_3(internal)
        P1: ---- e1_1(internal) -> e1_2(recv from P0) -> e1_3(send to P2)
        P2: e2_1(internal) -> e2_2(internal) ---------> e2_3(recv from P1)

    This creates events that are concurrent (e.g., e0_1 || e2_1) and
    causally related (e.g., e0_2 -> e1_2 -> e1_3 -> e2_3).
    """
    n = 3
    p0 = VCProcess(0, "Alice", n)
    p1 = VCProcess(1, "Bob", n)
    p2 = VCProcess(2, "Carol", n)

    print("=" * 70)
    print("Scenario: 3 Processes with Causal and Concurrent Events")
    print("=" * 70)
    print(f"  Vector format: [P0, P1, P2]\n")

    all_events: list[VCEvent] = []

    # P0: internal event
    e = p0.internal_event("prepare request")
    print_event(p0, e)
    all_events.append(e)

    # P1: internal event (concurrent with P0's event)
    e = p1.internal_event("load config")
    print_event(p1, e)
    all_events.append(e)

    # P2: internal event (concurrent with P0 and P1)
    e = p2.internal_event("init storage")
    print_event(p2, e)
    all_events.append(e)

    # P0 sends to P1
    e, msg_vec = p0.send("send request to Bob", receiver_id=1)
    print_event(p0, e)
    all_events.append(e)

    # P2: another internal event (concurrent with P0's send)
    e = p2.internal_event("scan index")
    print_event(p2, e)
    all_events.append(e)

    # P1 receives from P0
    e = p1.receive("receive request from Alice", sender_id=0, msg_vec=msg_vec)
    print_event(p1, e)
    all_events.append(e)

    # P1 sends to P2
    e, msg_vec2 = p1.send("forward to Carol", receiver_id=2)
    print_event(p1, e)
    all_events.append(e)

    # P0: internal event (concurrent with P1->P2 message)
    e = p0.internal_event("local computation")
    print_event(p0, e)
    all_events.append(e)

    # P2 receives from P1
    e = p2.receive("receive from Bob", sender_id=1, msg_vec=msg_vec2)
    print_event(p2, e)
    all_events.append(e)

    # P2: internal event
    e = p2.internal_event("finalize")
    print_event(p2, e)
    all_events.append(e)

    return [p0, p1, p2], all_events


def analyze_causality(all_events: list[VCEvent]) -> None:
    """Compare all pairs of events and report causal relationships."""
    print("\n" + "=" * 70)
    print("Causality Analysis (all event pairs)")
    print("=" * 70)

    # Label events for readability
    labels = [f"e{e.process_id}_{i}" for i, e in enumerate(all_events)]

    concurrent_pairs = []
    causal_pairs = []

    for i in range(len(all_events)):
        for j in range(i + 1, len(all_events)):
            rel = VectorClock.compare(all_events[i].timestamp,
                                      all_events[j].timestamp)
            if rel == Relation.CONCURRENT:
                concurrent_pairs.append((i, j, rel))
            else:
                causal_pairs.append((i, j, rel))

    print(f"\nCausal relationships ({len(causal_pairs)} pairs):")
    for i, j, rel in causal_pairs:
        print(f"  {labels[i]} {fmt_vec(all_events[i].timestamp)} "
              f"{rel.value} "
              f"{labels[j]} {fmt_vec(all_events[j].timestamp)}")

    print(f"\nConcurrent events ({len(concurrent_pairs)} pairs):")
    for i, j, rel in concurrent_pairs:
        print(f"  {labels[i]} {fmt_vec(all_events[i].timestamp)} "
              f"|| "
              f"{labels[j]} {fmt_vec(all_events[j].timestamp)}")
        print(f"    ({all_events[i].description} vs {all_events[j].description})")


def compare_with_lamport(all_events: list[VCEvent]) -> None:
    """
    Show why vector clocks are superior to Lamport clocks.
    With Lamport clocks, some concurrent events might have ordered timestamps,
    misleading us into thinking they are causally related.
    """
    print("\n" + "=" * 70)
    print("Vector Clocks vs Lamport Clocks")
    print("=" * 70)

    # Simulate Lamport timestamps (sum of vector components as approximation)
    # In reality, Lamport timestamps would be computed differently, but
    # for concurrent events, the key point is that Lamport can't distinguish
    # concurrent from causally ordered.

    print("""
With Lamport clocks:
  - If C(a) < C(b), we CANNOT conclude a -> b
  - Two concurrent events might have different Lamport timestamps
  - We can only say: if a -> b then C(a) < C(b) (one direction)

With Vector clocks:
  - VC(a) < VC(b) IFF a -> b (both directions!)
  - We can definitively detect concurrent events
  - More space (O(n) per event vs O(1) for Lamport)

This is the fundamental tradeoff: vector clocks give us complete
causality information at the cost of O(n) space per timestamp,
where n is the number of processes.
""")


if __name__ == "__main__":
    processes, all_events = simulate_scenario()
    analyze_causality(all_events)
    compare_with_lamport(all_events)
    print("Done.")
