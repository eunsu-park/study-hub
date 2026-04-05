"""
Lamport Timestamp Implementation with Simulated Message Passing

Demonstrates the Lamport logical clock algorithm for ordering events in a
distributed system. Each process maintains a local counter that increments
on every event (internal, send, or receive). On message receipt, the clock
is updated to max(local, received) + 1, ensuring the happens-before relation
is respected.

Key concepts:
- Logical clocks vs physical clocks
- Happens-before relation (Lamport's "->")
- Concurrent events and arbitrary ordering
- Clock condition: if a -> b, then C(a) < C(b)
  (but NOT the converse: C(a) < C(b) does NOT imply a -> b)

Usage:
    python 01_lamport_clock.py
"""

from dataclasses import dataclass, field


@dataclass
class Event:
    """Represents a single event in the distributed system."""
    process_id: int
    clock: int
    event_type: str       # "internal", "send", or "receive"
    description: str
    partner: int = -1     # The other process involved in send/receive


class LamportProcess:
    """A process with a Lamport logical clock."""

    def __init__(self, pid: int, name: str):
        self.pid = pid
        self.name = name
        self.clock = 0
        self.events: list[Event] = []

    def internal_event(self, description: str) -> Event:
        """Record an internal event. Clock increments by 1."""
        self.clock += 1
        event = Event(self.pid, self.clock, "internal", description)
        self.events.append(event)
        return event

    def send(self, description: str, receiver_id: int) -> tuple[Event, int]:
        """
        Send a message to another process.
        Returns the event and the timestamp to include in the message.
        """
        self.clock += 1
        event = Event(self.pid, self.clock, "send", description, partner=receiver_id)
        self.events.append(event)
        # The message carries the sender's current timestamp
        return event, self.clock

    def receive(self, description: str, sender_id: int, msg_timestamp: int) -> Event:
        """
        Receive a message from another process.
        Update clock to max(local, received) + 1.
        """
        self.clock = max(self.clock, msg_timestamp) + 1
        event = Event(self.pid, self.clock, "receive", description, partner=sender_id)
        self.events.append(event)
        return event

    def __repr__(self) -> str:
        return f"Process({self.name}, clock={self.clock})"


def print_event(process: LamportProcess, event: Event) -> None:
    """Pretty-print an event with its Lamport timestamp."""
    arrow = {
        "internal": "  *",
        "send":     " ->",
        "receive":  " <-",
    }[event.event_type]
    partner_info = ""
    if event.partner >= 0:
        partner_info = f" (peer=P{event.partner})"
    print(f"  [T={event.clock:>3}] P{event.process_id}({process.name}) "
          f"{arrow} {event.description}{partner_info}")


def simulate_basic_scenario() -> list[LamportProcess]:
    """
    Simulate 3 processes exchanging messages.

    Timeline:
        P0: internal -> send(m1 to P1) -> internal
        P1: internal -> receive(m1) -> send(m2 to P2) -> internal
        P2: internal -> internal -> receive(m2) -> internal
    """
    p0 = LamportProcess(0, "Alice")
    p1 = LamportProcess(1, "Bob")
    p2 = LamportProcess(2, "Carol")

    print("=" * 65)
    print("Scenario: Basic Message Passing with 3 Processes")
    print("=" * 65)

    # P0: internal event
    e = p0.internal_event("compute hash")
    print_event(p0, e)

    # P1: internal event (concurrent with P0's first event)
    e = p1.internal_event("read config")
    print_event(p1, e)

    # P2: internal event (concurrent with both above)
    e = p2.internal_event("init storage")
    print_event(p2, e)

    # P0 sends message m1 to P1
    e, ts = p0.send("send request m1 to Bob", receiver_id=1)
    print_event(p0, e)

    # P2: another internal event (concurrent with P0's send)
    e = p2.internal_event("scan disk")
    print_event(p2, e)

    # P1 receives m1 from P0
    e = p1.receive("receive m1 from Alice", sender_id=0, msg_timestamp=ts)
    print_event(p1, e)

    # P1 sends m2 to P2
    e, ts2 = p1.send("send response m2 to Carol", receiver_id=2)
    print_event(p1, e)

    # P0: internal event
    e = p0.internal_event("log result")
    print_event(p0, e)

    # P1: internal event
    e = p1.internal_event("update cache")
    print_event(p1, e)

    # P2 receives m2 from P1
    e = p2.receive("receive m2 from Bob", sender_id=1, msg_timestamp=ts2)
    print_event(p2, e)

    # P2: internal event
    e = p2.internal_event("process data")
    print_event(p2, e)

    return [p0, p1, p2]


def analyze_ordering(processes: list[LamportProcess]) -> None:
    """
    Collect all events, sort by Lamport timestamp, and show
    how concurrent events get arbitrary ordering.
    """
    print("\n" + "=" * 65)
    print("Global Event Ordering by Lamport Timestamp")
    print("(ties broken by process ID — arbitrary but deterministic)")
    print("=" * 65)

    all_events: list[tuple[LamportProcess, Event]] = []
    for proc in processes:
        for event in proc.events:
            all_events.append((proc, event))

    # Sort by (clock, process_id) — the tiebreaker is arbitrary
    all_events.sort(key=lambda x: (x[1].clock, x[1].process_id))

    for proc, event in all_events:
        print_event(proc, event)

    # Identify concurrent events (same timestamp, different processes)
    print("\n" + "-" * 65)
    print("Concurrent events (same Lamport timestamp, different processes):")
    print("-" * 65)

    from collections import defaultdict
    by_clock: dict[int, list[tuple[LamportProcess, Event]]] = defaultdict(list)
    for proc, event in all_events:
        by_clock[event.clock].append((proc, event))

    found_concurrent = False
    for clock_val, group in sorted(by_clock.items()):
        if len(group) > 1:
            found_concurrent = True
            pids = [f"P{e.process_id}({p.name})" for p, e in group]
            print(f"  T={clock_val}: {', '.join(pids)} — "
                  f"these events are CONCURRENT (no causal relationship)")

    if not found_concurrent:
        print("  (none found in this scenario)")

    print("\nKey insight: Lamport clocks guarantee that if a -> b then C(a) < C(b),")
    print("but C(a) < C(b) does NOT imply a -> b. Concurrent events may have")
    print("different timestamps. To detect true concurrency, use Vector Clocks.")


def demonstrate_clock_condition() -> None:
    """Show that the clock condition holds for causally related events."""
    print("\n" + "=" * 65)
    print("Verifying Clock Condition: a -> b implies C(a) < C(b)")
    print("=" * 65)

    p0 = LamportProcess(0, "Sender")
    p1 = LamportProcess(1, "Receiver")

    # Chain of causal events
    e1 = p0.internal_event("prepare data")
    e2, ts = p0.send("send data", receiver_id=1)
    e3 = p1.receive("receive data", sender_id=0, msg_timestamp=ts)
    e4 = p1.internal_event("process received data")

    chain = [
        (p0, e1, "e1"),
        (p0, e2, "e2"),
        (p1, e3, "e3"),
        (p1, e4, "e4"),
    ]

    print("\nCausal chain: e1 -> e2 -> e3 -> e4")
    for proc, event, label in chain:
        print(f"  {label}: P{event.process_id}({proc.name}) "
              f"T={event.clock} — {event.description}")

    # Verify ordering
    for i in range(len(chain) - 1):
        _, ea, la = chain[i]
        _, eb, lb = chain[i + 1]
        ok = ea.clock < eb.clock
        symbol = "<" if ok else ">="
        status = "OK" if ok else "VIOLATION"
        print(f"  C({la})={ea.clock} {symbol} C({lb})={eb.clock} — {status}")


if __name__ == "__main__":
    processes = simulate_basic_scenario()
    analyze_ordering(processes)
    demonstrate_clock_condition()
    print("\nDone.")
