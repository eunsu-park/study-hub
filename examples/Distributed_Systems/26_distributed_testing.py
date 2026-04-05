"""
Distributed Testing: Fault Injection and Chaos Engineering

Implements a fault injection framework for testing distributed systems.
Simulates Jepsen-style consistency checks, network partition tests,
and deterministic simulation testing.

Key concepts:
- Fault injection: crashes, partitions, delays, clock skew
- Jepsen-style linearizability checking
- Chaos engineering: random failure injection in running systems
- Deterministic simulation testing (FoundationDB style)
- Property-based testing for distributed protocols

Usage:
    python 26_distributed_testing.py
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Fault Injection Framework
# ---------------------------------------------------------------------------

class FaultType(Enum):
    CRASH = "crash"
    NETWORK_PARTITION = "network_partition"
    NETWORK_DELAY = "network_delay"
    CLOCK_SKEW = "clock_skew"
    DISK_SLOW = "disk_slow"
    MESSAGE_DROP = "message_drop"


@dataclass
class Fault:
    """A scheduled fault injection."""
    fault_type: FaultType
    target: str           # Node or link identifier
    start_time: float
    duration: float
    params: dict = field(default_factory=dict)

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    def is_active(self, t: float) -> bool:
        return self.start_time <= t < self.end_time

    def __repr__(self) -> str:
        return (f"Fault({self.fault_type.value}, target={self.target}, "
                f"t=[{self.start_time:.1f},{self.end_time:.1f}))")


class FaultInjector:
    """Manages fault injection during a test."""

    def __init__(self, seed: int = 42):
        self.faults: list[Fault] = []
        self._rng = random.Random(seed)
        self.log: list[str] = []

    def schedule(self, fault: Fault) -> None:
        self.faults.append(fault)
        self.log.append(f"Scheduled: {fault}")

    def random_partition(self, nodes: list[str], t: float,
                         duration: float) -> Fault:
        """Create a random network partition."""
        mid = len(nodes) // 2
        self._rng.shuffle(nodes)
        group_a = nodes[:mid]
        group_b = nodes[mid:]
        fault = Fault(
            FaultType.NETWORK_PARTITION,
            f"{group_a}<-X->{group_b}",
            t, duration,
            {"group_a": group_a, "group_b": group_b},
        )
        self.schedule(fault)
        return fault

    def random_crash(self, nodes: list[str], t: float,
                     duration: float) -> Fault:
        """Crash a random node."""
        target = self._rng.choice(nodes)
        fault = Fault(FaultType.CRASH, target, t, duration)
        self.schedule(fault)
        return fault

    def active_faults(self, t: float) -> list[Fault]:
        return [f for f in self.faults if f.is_active(t)]

    def is_node_crashed(self, node: str, t: float) -> bool:
        return any(
            f.fault_type == FaultType.CRASH and f.target == node and f.is_active(t)
            for f in self.faults
        )

    def is_partitioned(self, a: str, b: str, t: float) -> bool:
        for f in self.faults:
            if f.fault_type == FaultType.NETWORK_PARTITION and f.is_active(t):
                ga = f.params.get("group_a", [])
                gb = f.params.get("group_b", [])
                if (a in ga and b in gb) or (a in gb and b in ga):
                    return True
        return False


# ---------------------------------------------------------------------------
# Jepsen-style Linearizability Checker
# ---------------------------------------------------------------------------

@dataclass
class Operation:
    """A client operation for linearizability checking."""
    op_type: str        # "write" or "read"
    key: str
    value: str | None
    start_time: float
    end_time: float
    result: str | None = None
    ok: bool = True

    def __repr__(self) -> str:
        if self.op_type == "write":
            return f"W({self.key}={self.value})@[{self.start_time:.1f},{self.end_time:.1f}]"
        return f"R({self.key})={self.result}@[{self.start_time:.1f},{self.end_time:.1f}]"


def check_linearizability(history: list[Operation]) -> tuple[bool, list[str]]:
    """
    Simple linearizability check: every read must return the value of
    the most recent completed write, or a concurrent write.
    """
    violations = []
    writes = [op for op in history if op.op_type == "write" and op.ok]
    reads = [op for op in history if op.op_type == "read" and op.ok]

    for read in reads:
        # Find writes to the same key
        key_writes = [w for w in writes if w.key == read.key]

        # Find the latest write that definitely completed before the read
        completed_before = [w for w in key_writes if w.end_time <= read.start_time]

        # Find writes concurrent with the read
        concurrent = [w for w in key_writes
                      if w.start_time <= read.end_time
                      and w.end_time >= read.start_time]

        valid_values = set()
        if completed_before:
            latest = max(completed_before, key=lambda w: w.end_time)
            valid_values.add(latest.value)
        for w in concurrent:
            valid_values.add(w.value)
        if not valid_values:
            valid_values.add(None)  # No writes yet

        if read.result not in valid_values:
            violations.append(
                f"VIOLATION: {read} returned '{read.result}' but valid "
                f"values are {valid_values}")

    return len(violations) == 0, violations


# ---------------------------------------------------------------------------
# Deterministic Simulation
# ---------------------------------------------------------------------------

class DeterministicSimulator:
    """
    Deterministic simulation testing (FoundationDB style).
    All randomness, timing, and I/O are controlled by the simulator.
    """

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.clock = 0.0
        self.events: list[tuple[float, str, dict]] = []  # (time, type, data)
        self.log: list[str] = []

    def schedule_event(self, delay: float, event_type: str,
                       data: dict = None) -> None:
        t = self.clock + delay
        self.events.append((t, event_type, data or {}))
        self.events.sort(key=lambda e: e[0])

    def advance(self) -> tuple[str, dict] | None:
        """Process the next event. Returns (event_type, data) or None."""
        if not self.events:
            return None
        t, event_type, data = self.events.pop(0)
        self.clock = t
        self.log.append(f"t={t:.3f}: {event_type} {data}")
        return event_type, data

    def run_until(self, end_time: float) -> int:
        """Run until end_time. Returns number of events processed."""
        count = 0
        while self.events and self.events[0][0] <= end_time:
            self.advance()
            count += 1
        return count


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_fault_injection() -> None:
    """Demonstrate fault injection framework."""
    print("=" * 70)
    print("Fault Injection Framework")
    print("=" * 70)

    injector = FaultInjector(seed=42)
    nodes = ["node-0", "node-1", "node-2", "node-3", "node-4"]

    # Schedule various faults
    injector.random_crash(nodes, t=5.0, duration=10.0)
    injector.random_partition(list(nodes), t=8.0, duration=5.0)
    injector.schedule(Fault(FaultType.NETWORK_DELAY, "node-2", 3.0, 7.0,
                            {"delay_ms": 500}))
    injector.schedule(Fault(FaultType.CLOCK_SKEW, "node-4", 1.0, 20.0,
                            {"skew_ms": 200}))

    print(f"\n  Scheduled faults:")
    for line in injector.log:
        print(f"    {line}")

    # Check faults at various times
    print(f"\n  Active faults over time:")
    for t in [0, 3, 5, 8, 10, 13, 15, 20]:
        active = injector.active_faults(float(t))
        active_types = [f.fault_type.value for f in active]
        print(f"    t={t:>2}: {active_types if active_types else '(none)'}")


def demo_jepsen_check() -> None:
    """Demonstrate Jepsen-style linearizability checking."""
    print("\n" + "=" * 70)
    print("Jepsen-Style Linearizability Check")
    print("=" * 70)

    # Scenario 1: Linearizable history
    print(f"\n  Scenario 1: Valid (linearizable) history")
    history1 = [
        Operation("write", "x", "1", 0.0, 1.0),
        Operation("write", "x", "2", 2.0, 3.0),
        Operation("read", "x", None, 4.0, 5.0, result="2"),
        Operation("write", "x", "3", 6.0, 7.0),
        Operation("read", "x", None, 8.0, 9.0, result="3"),
    ]
    ok, violations = check_linearizability(history1)
    print(f"    Linearizable: {ok}")

    # Scenario 2: Stale read (violation)
    print(f"\n  Scenario 2: Stale read (violation)")
    history2 = [
        Operation("write", "x", "1", 0.0, 1.0),
        Operation("write", "x", "2", 2.0, 3.0),
        Operation("read", "x", None, 4.0, 5.0, result="1"),  # Stale!
    ]
    ok, violations = check_linearizability(history2)
    print(f"    Linearizable: {ok}")
    for v in violations:
        print(f"    {v}")

    # Scenario 3: Concurrent operations (valid)
    print(f"\n  Scenario 3: Concurrent write and read (valid)")
    history3 = [
        Operation("write", "x", "1", 0.0, 1.0),
        Operation("write", "x", "2", 1.5, 3.0),  # Overlaps with read
        Operation("read", "x", None, 2.0, 2.5, result="1"),  # Valid: write is concurrent
    ]
    ok, violations = check_linearizability(history3)
    print(f"    Linearizable: {ok} (read can return '1' because write is concurrent)")


def demo_deterministic_simulation() -> None:
    """Demonstrate deterministic simulation testing."""
    print("\n" + "=" * 70)
    print("Deterministic Simulation Testing")
    print("=" * 70)

    print("""
  FoundationDB approach: run the entire distributed system in a single
  process with simulated time, network, and disk. This enables:
  - Reproducible bugs (same seed = same execution)
  - Fast testing (no real I/O delays)
  - Comprehensive fault injection
""")

    sim = DeterministicSimulator(seed=42)

    # Schedule events that model a simple Raft exchange
    sim.schedule_event(0.0, "client_request", {"cmd": "PUT x=1"})
    sim.schedule_event(0.1, "leader_append_log", {"entry": "PUT x=1"})
    sim.schedule_event(0.2, "replicate_to_follower", {"follower": "node-1"})
    sim.schedule_event(0.3, "replicate_to_follower", {"follower": "node-2"})
    sim.schedule_event(0.4, "commit", {"index": 1})
    sim.schedule_event(0.5, "apply_to_state_machine", {"cmd": "PUT x=1"})
    sim.schedule_event(0.6, "client_response", {"result": "OK"})

    # Inject a fault: delay replication to node-2
    sim.schedule_event(0.25, "FAULT_network_delay",
                       {"target": "node-2", "delay_ms": 100})

    count = sim.run_until(1.0)

    print(f"  Simulation (seed=42):")
    for line in sim.log:
        print(f"    {line}")

    print(f"\n  Processed {count} events in deterministic order")
    print(f"  Re-running with same seed produces identical execution")

    # Verify reproducibility
    sim2 = DeterministicSimulator(seed=42)
    sim2.schedule_event(0.0, "client_request", {"cmd": "PUT x=1"})
    sim2.schedule_event(0.1, "leader_append_log", {"entry": "PUT x=1"})
    sim2.schedule_event(0.2, "replicate_to_follower", {"follower": "node-1"})
    sim2.schedule_event(0.3, "replicate_to_follower", {"follower": "node-2"})
    sim2.schedule_event(0.4, "commit", {"index": 1})
    sim2.schedule_event(0.5, "apply_to_state_machine", {"cmd": "PUT x=1"})
    sim2.schedule_event(0.6, "client_response", {"result": "OK"})
    sim2.schedule_event(0.25, "FAULT_network_delay",
                        {"target": "node-2", "delay_ms": 100})
    sim2.run_until(1.0)

    identical = sim.log == sim2.log
    print(f"  Reproducible: {identical}")


def demo_chaos_engineering() -> None:
    """Describe chaos engineering practices."""
    print("\n" + "=" * 70)
    print("Chaos Engineering Practices")
    print("=" * 70)

    print("""
  Chaos engineering principles (Netflix):
  1. Define "steady state" (normal behavior metrics)
  2. Hypothesize that steady state continues during faults
  3. Inject real-world failures (instance crash, network loss, etc.)
  4. Observe whether steady state is maintained
  5. Automate experiments and run continuously

  Common fault types:
  ┌───────────────────────┬───────────────────────────────────────┐
  │ Fault                 │ Tools                                 │
  ├───────────────────────┼───────────────────────────────────────┤
  │ Process crash         │ kill -9, Chaos Monkey                 │
  │ Network partition     │ iptables, tc netem                    │
  │ Network delay/loss    │ tc netem, Toxiproxy                   │
  │ Clock skew            │ libfaketime                           │
  │ Disk full/slow        │ dd, dm-delay                          │
  │ CPU stress            │ stress-ng                             │
  │ DNS failure           │ iptables on port 53                   │
  └───────────────────────┴───────────────────────────────────────┘

  Testing frameworks:
  - Jepsen: Clojure-based, linearizability/serializability checking
  - Chaos Monkey: Random instance termination (Netflix)
  - Toxiproxy: Programmable network proxy for fault injection
  - LitmusChaos: Kubernetes-native chaos engineering
  - FoundationDB: Deterministic simulation (most thorough)
""")


if __name__ == "__main__":
    demo_fault_injection()
    demo_jepsen_check()
    demo_deterministic_simulation()
    demo_chaos_engineering()
    print("Done.")
