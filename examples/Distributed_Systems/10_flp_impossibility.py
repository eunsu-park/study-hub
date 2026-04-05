"""
FLP Impossibility Demonstration

Illustrates the Fischer-Lynch-Paterson impossibility result: no
deterministic consensus protocol can guarantee termination in an
asynchronous system if even a single process can crash. The simulation
shows how a carefully timed crash can keep a consensus protocol from
ever deciding, and then demonstrates circumvention strategies (randomness,
timeouts, failure detectors).

Key concepts:
- Bivalent vs univalent configurations
- Adversarial scheduler that prevents decision
- Circumvention: randomised consensus (Ben-Or style)
- Circumvention: failure detectors (Omega / eventually perfect)
- Partial synchrony as practical workaround

Usage:
    python 10_flp_impossibility.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Deterministic consensus (vulnerable to FLP)
# ---------------------------------------------------------------------------

class ProcessStatus(Enum):
    ACTIVE = "active"
    DECIDED = "decided"
    CRASHED = "crashed"


@dataclass
class ConsensusProcess:
    """A process in a simple round-based consensus protocol."""
    pid: int
    proposal: int
    status: ProcessStatus = ProcessStatus.ACTIVE
    decision: int | None = None
    round_num: int = 0
    received: list[int] = field(default_factory=list)


class DeterministicConsensus:
    """
    Simple round-based consensus: in each round, every active process
    broadcasts its current proposal and adopts the majority value.
    Decides when all received values agree for two consecutive rounds.

    This is intentionally simplified to demonstrate the FLP vulnerability.
    """

    def __init__(self, n: int, proposals: list[int], seed: int = 42):
        self.processes = {
            i: ConsensusProcess(pid=i, proposal=proposals[i])
            for i in range(n)
        }
        self.n = n
        self._rng = random.Random(seed)
        self.rounds_executed = 0
        self.decided = False
        self.history: list[str] = []

    def run_round(self, crash_pid: int | None = None,
                  delay_pid: int | None = None) -> bool:
        """
        Execute one round. Optionally crash or delay a process.
        Returns True if consensus was reached.
        """
        self.rounds_executed += 1
        self.history.append(f"--- Round {self.rounds_executed} ---")

        # Crash injection
        if crash_pid is not None and self.processes[crash_pid].status == ProcessStatus.ACTIVE:
            self.processes[crash_pid].status = ProcessStatus.CRASHED
            self.history.append(f"  P{crash_pid} CRASHED (FLP adversary)")

        # Collect proposals from active processes
        active_proposals: dict[int, int] = {}
        for pid, proc in self.processes.items():
            if proc.status == ProcessStatus.ACTIVE:
                if pid != delay_pid:
                    active_proposals[pid] = proc.proposal

        self.history.append(
            f"  Active proposals: {active_proposals}")

        if not active_proposals:
            self.history.append("  No active processes — protocol stalled")
            return False

        # Each active process receives proposals and updates
        for pid, proc in self.processes.items():
            if proc.status != ProcessStatus.ACTIVE:
                continue
            if pid == delay_pid:
                # Delayed process does not receive this round
                self.history.append(f"  P{pid} message DELAYED (FLP adversary)")
                continue

            received = list(active_proposals.values())
            proc.received = received

            # Adopt majority value
            count_0 = received.count(0)
            count_1 = received.count(1)
            if count_0 > count_1:
                proc.proposal = 0
            elif count_1 > count_0:
                proc.proposal = 1
            # Tie: keep current proposal (bivalent!)

        # Check if all active processes agree
        active = [p for p in self.processes.values()
                  if p.status == ProcessStatus.ACTIVE]
        if active:
            vals = set(p.proposal for p in active)
            if len(vals) == 1:
                for p in active:
                    p.decision = p.proposal
                    p.status = ProcessStatus.DECIDED
                self.decided = True
                self.history.append(
                    f"  DECIDED: all active processes agree on {vals.pop()}")
                return True
            else:
                self.history.append(
                    f"  Bivalent: proposals are {[p.proposal for p in active]}")

        return False

    def print_history(self) -> None:
        for line in self.history:
            print(f"    {line}")


def demo_flp_adversary() -> None:
    """Show how an adversary can prevent consensus forever."""
    print("=" * 70)
    print("FLP Adversary: Preventing Deterministic Consensus")
    print("=" * 70)

    # 3 processes, proposals [0, 1, 0]
    n = 3
    proposals = [0, 1, 0]
    print(f"\n  {n} processes, initial proposals: {proposals}")
    print(f"  Adversary will strategically crash/delay one process each round\n")

    proto = DeterministicConsensus(n, proposals)

    # Round 1: delay P2 to keep system bivalent
    proto.run_round(delay_pid=2)
    # Round 2: delay P0
    proto.run_round(delay_pid=0)
    # Round 3: crash P1 — now we may deadlock
    proto.run_round(crash_pid=1)
    # Round 4-5: remaining processes try but may not agree
    for _ in range(2):
        if proto.decided:
            break
        proto.run_round()

    proto.print_history()

    if not proto.decided:
        print(f"\n  Result: Protocol FAILED to decide after "
              f"{proto.rounds_executed} rounds")
        print(f"  This demonstrates FLP: a single crash can prevent termination")
    else:
        print(f"\n  Result: Protocol decided (adversary was not optimal)")

    # Now show normal operation without adversary
    print(f"\n  --- Without adversary (same initial proposals) ---")
    proto2 = DeterministicConsensus(n, proposals)
    for _ in range(5):
        if proto2.decided:
            break
        proto2.run_round()
    proto2.print_history()
    print(f"  Result: {'DECIDED' if proto2.decided else 'UNDECIDED'} "
          f"in {proto2.rounds_executed} rounds")


# ---------------------------------------------------------------------------
# Circumvention: Randomised consensus (Ben-Or style)
# ---------------------------------------------------------------------------

def demo_randomised_consensus() -> None:
    """Ben-Or style randomised consensus circumvents FLP."""
    print("\n" + "=" * 70)
    print("Circumvention 1: Randomised Consensus (Ben-Or Style)")
    print("=" * 70)

    print("""
  Ben-Or's protocol: each process flips a coin when no majority exists.
  FLP says deterministic protocols can be blocked, but randomised ones
  terminate with probability 1 (though expected rounds can be exponential).
""")

    n = 5
    rng = random.Random(42)

    # Run multiple trials
    trial_rounds = []

    for trial in range(10):
        proposals = [rng.choice([0, 1]) for _ in range(n)]
        # Simplified Ben-Or: each round, broadcast; if supermajority, decide;
        # else, adopt random value.
        values = list(proposals)
        decided = False
        max_rounds = 100

        for r in range(1, max_rounds + 1):
            # Count
            c0 = values.count(0)
            c1 = values.count(1)

            # Supermajority threshold: > (n+f)/2 where f=1
            threshold = (n + 1) // 2 + 1

            if c0 >= threshold:
                trial_rounds.append(r)
                decided = True
                break
            elif c1 >= threshold:
                trial_rounds.append(r)
                decided = True
                break
            else:
                # No supermajority: each process randomises
                for i in range(n):
                    values[i] = rng.choice([0, 1])

        if not decided:
            trial_rounds.append(max_rounds)

    print(f"  10 trials with {n} processes (1 crash tolerated):")
    for i, rounds in enumerate(trial_rounds):
        bar = "#" * min(rounds, 40)
        print(f"    Trial {i}: {rounds:>3} rounds {bar}")

    avg = sum(trial_rounds) / len(trial_rounds)
    print(f"\n  Average rounds to decide: {avg:.1f}")
    print(f"  Key: Randomness breaks the adversary's ability to maintain bivalence")


# ---------------------------------------------------------------------------
# Circumvention: Failure detectors
# ---------------------------------------------------------------------------

def demo_failure_detectors() -> None:
    """Show how failure detectors circumvent FLP."""
    print("\n" + "=" * 70)
    print("Circumvention 2: Failure Detectors")
    print("=" * 70)

    print("""
  Chandra-Toueg showed that consensus is solvable with an "eventually
  perfect" failure detector (◇P) or an "Omega" leader oracle (Ω).

  ◇P: After some unknown time, accurately detects all crashes (no false positives).
  Ω:  After some unknown time, all correct processes agree on one leader.

  Below we simulate Omega-based consensus:
""")

    n = 5
    rng = random.Random(42)
    proposals = [rng.choice([0, 1]) for _ in range(n)]
    crashed = {2}  # P2 crashes

    print(f"  Processes: {n}, proposals: {proposals}, crashed: {crashed}")

    # Phase 1: Omega oracle eventually elects a stable leader
    # Simulate: first few rounds, oracle is wrong (FLP-like delay)
    # Eventually it stabilises
    oracle_stabilises_at = 3

    for round_num in range(1, 8):
        if round_num < oracle_stabilises_at:
            # Oracle is unstable — might pick crashed process
            leader = rng.choice(list(range(n)))
            stable = False
        else:
            # Oracle stabilised — picks a correct process
            leader = 0
            stable = True

        leader_ok = leader not in crashed

        if leader_ok and stable:
            decision = proposals[leader]
            print(f"    Round {round_num}: Ω={leader} (stable={stable}, "
                  f"alive={leader_ok}) => DECIDE {decision}")
            break
        else:
            print(f"    Round {round_num}: Ω={leader} (stable={stable}, "
                  f"alive={leader_ok}) => no decision")

    print(f"\n  After Omega stabilises, consensus is reached in O(1) rounds")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def demo_summary() -> None:
    """Print summary of FLP and circumventions."""
    print("\n" + "=" * 70)
    print("Summary: FLP Impossibility and Circumventions")
    print("=" * 70)

    print("""
  FLP Theorem (1985):
    No deterministic protocol can solve consensus in an asynchronous
    system if even one process may crash.

  Why it matters:
    - Explains why Paxos/Raft need timeouts (partial synchrony)
    - Explains why 2PC can block forever
    - Drives the design of practical consensus protocols

  Circumventions:
  ┌─────────────────────────┬──────────────────────────────────────────┐
  │ Approach                │ How it breaks FLP                        │
  ├─────────────────────────┼──────────────────────────────────────────┤
  │ Randomisation           │ Adversary can't predict coin flips       │
  │ Failure detectors (◇P)  │ Eventually accurate crash detection      │
  │ Leader oracle (Ω)       │ Eventually stable leader election        │
  │ Partial synchrony       │ Bounded delays after GST                 │
  │ Timeouts + retries      │ Practical version of partial synchrony   │
  └─────────────────────────┴──────────────────────────────────────────┘

  Practical systems (Raft, etcd, ZooKeeper) assume partial synchrony:
  safety is ALWAYS maintained, liveness holds AFTER the network stabilises.
""")


if __name__ == "__main__":
    demo_flp_adversary()
    demo_randomised_consensus()
    demo_failure_detectors()
    demo_summary()
    print("Done.")
