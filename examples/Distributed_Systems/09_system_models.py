"""
System Models and Failure Mode Simulator

Simulates different distributed system models (synchronous, asynchronous,
partially synchronous) and failure modes (crash-stop, crash-recovery,
Byzantine). Demonstrates how timing assumptions and failure types affect
protocol correctness and progress guarantees.

Key concepts:
- Synchronous model: bounded message delay and processing time
- Asynchronous model: no timing guarantees
- Partially synchronous model: eventually bounded (GST)
- Crash-stop vs crash-recovery vs Byzantine failures
- Safety vs liveness tradeoffs under different models

Usage:
    python 09_system_models.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


# ---------------------------------------------------------------------------
# System model definitions
# ---------------------------------------------------------------------------

class SystemModel(Enum):
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    PARTIALLY_SYNCHRONOUS = "partially_synchronous"


class FailureMode(Enum):
    CRASH_STOP = "crash_stop"
    CRASH_RECOVERY = "crash_recovery"
    BYZANTINE = "byzantine"


@dataclass
class Message:
    """A message in the distributed system."""
    sender: int
    receiver: int
    content: str
    send_time: float
    deliver_time: float | None = None


@dataclass
class ProcessState:
    """Represents a process in the system."""
    pid: int
    alive: bool = True
    recovered: bool = False
    byzantine: bool = False
    crash_time: float | None = None
    recovery_time: float | None = None
    events: list[str] = field(default_factory=list)


class NetworkSimulator:
    """
    Simulates message delivery under different system models.

    - Synchronous: messages delivered within a known bound delta.
    - Asynchronous: messages can be delayed arbitrarily (but eventually arrive).
    - Partially synchronous: messages unbounded until GST, bounded after.
    """

    def __init__(self, model: SystemModel, delta: float = 1.0,
                 gst: float = 10.0, seed: int = 42):
        """
        Args:
            model: System timing model.
            delta: Maximum message delay (for synchronous / post-GST).
            gst: Global Stabilisation Time (for partially synchronous).
            seed: Random seed for reproducibility.
        """
        self.model = model
        self.delta = delta
        self.gst = gst
        self._rng = random.Random(seed)
        self._messages: list[Message] = []

    def send(self, sender: int, receiver: int, content: str,
             send_time: float) -> Message:
        """Send a message; delivery time depends on the system model."""
        if self.model == SystemModel.SYNCHRONOUS:
            delay = self._rng.uniform(0.1, self.delta)
        elif self.model == SystemModel.ASYNCHRONOUS:
            # Arbitrary delay: could be very long
            delay = self._rng.expovariate(0.1)  # mean = 10
        else:
            # Partially synchronous: unbounded before GST, bounded after
            if send_time < self.gst:
                delay = self._rng.expovariate(0.1)
            else:
                delay = self._rng.uniform(0.1, self.delta)

        msg = Message(sender, receiver, content, send_time,
                      deliver_time=send_time + delay)
        self._messages.append(msg)
        return msg

    def get_delivered(self, by_time: float) -> list[Message]:
        """Return messages delivered by the given time."""
        return [m for m in self._messages
                if m.deliver_time is not None and m.deliver_time <= by_time]

    @property
    def all_messages(self) -> list[Message]:
        return list(self._messages)


# ---------------------------------------------------------------------------
# Failure simulators
# ---------------------------------------------------------------------------

class FailureInjector:
    """Simulates different failure modes on a set of processes."""

    def __init__(self, mode: FailureMode, seed: int = 42):
        self.mode = mode
        self._rng = random.Random(seed)

    def inject(self, processes: dict[int, ProcessState],
               time: float, max_failures: int) -> list[int]:
        """Inject failures into up to max_failures processes. Returns failed pids."""
        alive = [p for p in processes.values() if p.alive]
        if not alive:
            return []

        count = min(max_failures, len(alive))
        victims = self._rng.sample(alive, count)
        failed_pids = []

        for proc in victims:
            if self.mode == FailureMode.CRASH_STOP:
                proc.alive = False
                proc.crash_time = time
                proc.events.append(f"t={time:.1f}: CRASH-STOP (permanent)")
            elif self.mode == FailureMode.CRASH_RECOVERY:
                proc.alive = False
                proc.crash_time = time
                # Schedule recovery after some delay
                recovery_delay = self._rng.uniform(1.0, 5.0)
                proc.recovery_time = time + recovery_delay
                proc.events.append(
                    f"t={time:.1f}: CRASH (will recover at "
                    f"t={proc.recovery_time:.1f})")
            elif self.mode == FailureMode.BYZANTINE:
                proc.byzantine = True
                proc.events.append(
                    f"t={time:.1f}: BYZANTINE (sending arbitrary messages)")
            failed_pids.append(proc.pid)

        return failed_pids

    def tick_recovery(self, processes: dict[int, ProcessState],
                      time: float) -> list[int]:
        """Check for crash-recovery processes ready to recover."""
        recovered = []
        for proc in processes.values():
            if (not proc.alive and proc.recovery_time is not None
                    and time >= proc.recovery_time):
                proc.alive = True
                proc.recovered = True
                proc.events.append(f"t={time:.1f}: RECOVERED")
                proc.recovery_time = None
                recovered.append(proc.pid)
        return recovered


# ---------------------------------------------------------------------------
# Simple consensus simulation to show model effects
# ---------------------------------------------------------------------------

def simulate_consensus_round(
    n_processes: int,
    model: SystemModel,
    failure_mode: FailureMode,
    n_failures: int,
    proposal_value: str = "v1"
) -> dict:
    """
    Simulate a single round of consensus under the given model and failures.
    Uses a simplified coordinator-based protocol (like Phase 1 of Paxos).

    Returns a summary dict with results.
    """
    net = NetworkSimulator(model, delta=1.0, gst=5.0)
    injector = FailureInjector(failure_mode, seed=42)

    processes: dict[int, ProcessState] = {
        i: ProcessState(pid=i) for i in range(n_processes)
    }

    coordinator = 0
    time = 0.0
    results = {
        "model": model.value,
        "failure_mode": failure_mode.value,
        "n_processes": n_processes,
        "n_failures": n_failures,
        "safety_maintained": True,
        "progress_made": False,
        "rounds_needed": 0,
        "events": [],
    }

    # Phase 1: Coordinator sends proposal to all
    results["events"].append(f"t={time:.1f}: Coordinator P{coordinator} "
                             f"proposes '{proposal_value}'")

    messages_sent = []
    for pid in range(n_processes):
        if pid == coordinator:
            continue
        msg = net.send(coordinator, pid, f"PROPOSE:{proposal_value}", time)
        messages_sent.append(msg)

    # Inject failures at t=0.5 (mid-flight)
    time = 0.5
    failed = injector.inject(processes, time, n_failures)
    if failed:
        results["events"].append(
            f"t={time:.1f}: Failures injected at P{failed}")

    # Phase 2: Check responses
    time = 2.0  # Wait for delta

    # Crash-recovery: check for recoveries
    if failure_mode == FailureMode.CRASH_RECOVERY:
        recovered = injector.tick_recovery(processes, time)
        if recovered:
            results["events"].append(
                f"t={time:.1f}: Recovered: P{recovered}")

    delivered = net.get_delivered(time)
    acks = 0
    for msg in delivered:
        receiver = processes[msg.receiver]
        if receiver.alive and not receiver.byzantine:
            acks += 1
        elif receiver.byzantine:
            # Byzantine node might send conflicting ack
            results["events"].append(
                f"  P{receiver.pid} (byzantine) sends conflicting ACK")

    # Coordinator counts itself
    if processes[coordinator].alive:
        acks += 1

    majority = n_processes // 2 + 1
    results["progress_made"] = acks >= majority
    results["rounds_needed"] = 1

    results["events"].append(
        f"t={time:.1f}: Coordinator received {acks}/{n_processes} acks "
        f"(need {majority} for majority)")

    if results["progress_made"]:
        results["events"].append(f"  => CONSENSUS REACHED on '{proposal_value}'")
    else:
        results["events"].append(f"  => NO CONSENSUS (insufficient acks)")

    # Safety check: Byzantine nodes might cause disagreement
    if failure_mode == FailureMode.BYZANTINE and n_failures >= majority:
        results["safety_maintained"] = False
        results["events"].append(
            "  WARNING: Too many Byzantine failures — safety cannot be guaranteed")

    return results


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_system_models() -> None:
    """Compare message delivery under different system models."""
    print("=" * 70)
    print("Message Delivery Under Different System Models")
    print("=" * 70)

    models = [
        (SystemModel.SYNCHRONOUS, "Bounded delay (0, delta]"),
        (SystemModel.ASYNCHRONOUS, "Unbounded delay"),
        (SystemModel.PARTIALLY_SYNCHRONOUS, "Unbounded until GST, then bounded"),
    ]

    for model, desc in models:
        print(f"\n  {model.value}: {desc}")
        net = NetworkSimulator(model, delta=1.0, gst=5.0, seed=42)

        # Send 10 messages at various times
        delays = []
        for i in range(10):
            send_t = i * 1.0
            msg = net.send(0, 1, f"msg_{i}", send_t)
            delay = msg.deliver_time - msg.send_time
            delays.append(delay)

        min_d = min(delays)
        max_d = max(delays)
        avg_d = sum(delays) / len(delays)
        print(f"    10 messages: min_delay={min_d:.2f}s, "
              f"max_delay={max_d:.2f}s, avg_delay={avg_d:.2f}s")

        # Show histogram-like distribution
        buckets = [0] * 5
        bucket_labels = ["<0.5s", "0.5-1s", "1-2s", "2-5s", ">5s"]
        for d in delays:
            if d < 0.5:
                buckets[0] += 1
            elif d < 1.0:
                buckets[1] += 1
            elif d < 2.0:
                buckets[2] += 1
            elif d < 5.0:
                buckets[3] += 1
            else:
                buckets[4] += 1
        for label, count in zip(bucket_labels, buckets):
            bar = "#" * (count * 3)
            print(f"    {label:>6}: {bar} ({count})")


def demo_failure_modes() -> None:
    """Demonstrate the three failure modes."""
    print("\n" + "=" * 70)
    print("Failure Modes Demonstration")
    print("=" * 70)

    modes = [
        (FailureMode.CRASH_STOP, "Process halts permanently"),
        (FailureMode.CRASH_RECOVERY, "Process halts then recovers with stable storage"),
        (FailureMode.BYZANTINE, "Process behaves arbitrarily (may lie)"),
    ]

    n = 5
    for mode, desc in modes:
        print(f"\n  --- {mode.value}: {desc} ---")
        processes = {i: ProcessState(pid=i) for i in range(n)}
        injector = FailureInjector(mode, seed=42)

        # Inject 2 failures
        failed = injector.inject(processes, time=1.0, max_failures=2)

        # Show state at t=1
        for pid, proc in processes.items():
            status = "ALIVE" if proc.alive else "DEAD"
            if proc.byzantine:
                status = "BYZANTINE"
            print(f"    P{pid}: {status}")
            for evt in proc.events:
                print(f"      {evt}")

        # For crash-recovery, show recovery at t=5
        if mode == FailureMode.CRASH_RECOVERY:
            recovered = injector.tick_recovery(processes, time=5.0)
            if recovered:
                print(f"    At t=5.0: recovered P{recovered}")
                for pid in recovered:
                    for evt in processes[pid].events:
                        if "RECOVERED" in evt:
                            print(f"      {evt}")


def demo_fault_tolerance_bounds() -> None:
    """Show required redundancy for each failure mode."""
    print("\n" + "=" * 70)
    print("Fault Tolerance Bounds")
    print("=" * 70)

    print("""
  For N total processes and f faulty processes:

  ┌──────────────────────┬───────────────┬────────────────────────────┐
  │ Failure Mode         │ Tolerance     │ Explanation                │
  ├──────────────────────┼───────────────┼────────────────────────────┤
  │ Crash-stop           │ f < N/2       │ Need majority alive        │
  │ Crash-recovery       │ f < N/2       │ Same, but with stable log  │
  │ Byzantine            │ f < N/3       │ Need 2f+1 honest majority  │
  └──────────────────────┴───────────────┴────────────────────────────┘
""")

    # Concrete examples
    scenarios = [
        (5, 2, FailureMode.CRASH_STOP),
        (5, 2, FailureMode.CRASH_RECOVERY),
        (5, 1, FailureMode.BYZANTINE),
        (5, 2, FailureMode.BYZANTINE),
        (7, 2, FailureMode.BYZANTINE),
    ]

    print(f"  {'N':>3}  {'f':>3}  {'Mode':<20}  {'Tolerable?':<12}  Reason")
    print("  " + "-" * 65)

    for n, f, mode in scenarios:
        if mode == FailureMode.BYZANTINE:
            ok = f < n / 3
            reason = f"f={f} {'<' if ok else '>='} N/3={n/3:.1f}"
        else:
            ok = f < n / 2
            reason = f"f={f} {'<' if ok else '>='} N/2={n/2:.1f}"

        status = "YES" if ok else "NO"
        print(f"  {n:>3}  {f:>3}  {mode.value:<20}  {status:<12}  {reason}")


def demo_consensus_scenarios() -> None:
    """Run consensus under various model/failure combinations."""
    print("\n" + "=" * 70)
    print("Consensus Under Different Models and Failures")
    print("=" * 70)

    scenarios = [
        (5, SystemModel.SYNCHRONOUS, FailureMode.CRASH_STOP, 1,
         "Sync + 1 crash: should succeed"),
        (5, SystemModel.SYNCHRONOUS, FailureMode.CRASH_STOP, 3,
         "Sync + 3 crashes: too many failures"),
        (5, SystemModel.SYNCHRONOUS, FailureMode.BYZANTINE, 1,
         "Sync + 1 Byzantine: within f<N/3 bound"),
        (5, SystemModel.SYNCHRONOUS, FailureMode.BYZANTINE, 2,
         "Sync + 2 Byzantine: exceeds f<N/3 bound"),
        (5, SystemModel.PARTIALLY_SYNCHRONOUS, FailureMode.CRASH_STOP, 1,
         "Partial sync + 1 crash: may need retries"),
    ]

    for n, model, fmode, nf, desc in scenarios:
        print(f"\n  --- {desc} ---")
        result = simulate_consensus_round(n, model, fmode, nf)
        for evt in result["events"]:
            print(f"    {evt}")
        print(f"    Safety maintained: {result['safety_maintained']}")
        print(f"    Progress made: {result['progress_made']}")


def demo_safety_vs_liveness() -> None:
    """Illustrate the fundamental tension between safety and liveness."""
    print("\n" + "=" * 70)
    print("Safety vs Liveness Properties")
    print("=" * 70)

    print("""
  Safety:   "Nothing bad happens" — never decide on conflicting values.
  Liveness: "Something good eventually happens" — eventually decide.

  FLP Impossibility (Lesson 03) shows that in an asynchronous system
  with even one crash failure, no deterministic protocol can guarantee
  BOTH safety and liveness simultaneously.

  Practical protocols choose:
  ┌────────────────────┬──────────┬──────────┬──────────────────────┐
  │ Protocol           │ Safety   │ Liveness │ Notes                │
  ├────────────────────┼──────────┼──────────┼──────────────────────┤
  │ Paxos / Raft       │ Always   │ Mostly*  │ *May stall in async  │
  │ 2PC                │ Always   │ No       │ Blocks on coord fail │
  │ Ben-Or (random)    │ Always   │ Prob.    │ Expected O(2^n) rnds │
  │ FLP-violating      │ Maybe    │ Always   │ Unsafe!              │
  └────────────────────┴──────────┴──────────┴──────────────────────┘

  Real systems use partial synchrony (GST assumption) to achieve both
  safety (always) and liveness (after GST).
""")


if __name__ == "__main__":
    demo_system_models()
    demo_failure_modes()
    demo_fault_tolerance_bounds()
    demo_consensus_scenarios()
    demo_safety_vs_liveness()
    print("Done.")
