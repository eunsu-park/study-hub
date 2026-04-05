"""
Exercises for Lesson 03: FLP Impossibility and Bounds
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import random
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


# === Exercise 1: Adversarial Consensus Prevention ===
# Problem: Implement a simple consensus protocol and an adversary
# scheduler that prevents the protocol from ever terminating,
# illustrating the FLP impossibility result.

class ConsensusProcess:
    """
    A process participating in binary consensus.
    Uses a simple round-based protocol where processes exchange
    their current preference and adopt the majority.
    """

    def __init__(self, pid: int, initial_value: int):
        self.pid = pid
        self.preference = initial_value
        self.decided = False
        self.decided_value = None
        self.inbox: List[Tuple[int, int]] = []  # (sender_pid, value)

    def send_preference(self) -> Tuple[int, int]:
        return (self.pid, self.preference)

    def receive(self, sender_pid: int, value: int):
        self.inbox.append((sender_pid, value))

    def process_round(self, n_total: int) -> bool:
        """
        Process received messages. Adopt majority preference.
        Decide if unanimous. Returns True if decided.
        """
        if self.decided:
            return True
        values = [v for _, v in self.inbox] + [self.preference]
        self.inbox.clear()
        ones = sum(v for v in values)
        zeros = len(values) - ones
        self.preference = 1 if ones > zeros else 0
        # Decide only if all received values agree
        if ones == len(values) or zeros == len(values):
            self.decided = True
            self.decided_value = self.preference
            return True
        return False


class FLPAdversary:
    """
    An adversary scheduler that exploits message delay to prevent
    consensus, illustrating FLP impossibility.
    """

    def __init__(self):
        self.round_count = 0

    def schedule_messages(
        self,
        messages: List[Tuple[int, int, int]],
        processes: List[ConsensusProcess],
    ) -> List[Tuple[int, int, int]]:
        """
        Given pending messages (sender, receiver, value), decide which
        to deliver. The adversary delays critical messages to prevent
        unanimity.

        Strategy: If delivering all messages would lead to a decision,
        delay messages from one side to maintain a split.
        """
        self.round_count += 1
        ones = sum(1 for _, _, v in messages if v == 1)
        zeros = sum(1 for _, _, v in messages if v == 0)

        if ones == len(messages) or zeros == len(messages):
            # All agree - delay one message to break unanimity
            # (In FLP proof, adversary can always find such a schedule)
            delayed_idx = 0
            return messages[1:]  # Drop one message
        return messages  # Deliver all if already split


def exercise_1():
    """
    Demonstrate how an adversary prevents consensus termination.
    """
    print("=== Exercise 1: Adversarial Consensus Prevention ===\n")

    n = 3
    max_rounds = 20

    # Bivalent initial configuration: not all same value
    processes = [
        ConsensusProcess(0, 0),
        ConsensusProcess(1, 1),
        ConsensusProcess(2, 0),
    ]
    adversary = FLPAdversary()

    decided = False
    for round_num in range(max_rounds):
        # Generate messages
        all_messages = []
        for p in processes:
            msg = p.send_preference()
            for q in processes:
                if q.pid != p.pid:
                    all_messages.append((msg[0], q.pid, msg[1]))

        # Adversary filters messages
        delivered = adversary.schedule_messages(all_messages, processes)

        # Deliver
        for sender, receiver, value in delivered:
            processes[receiver].receive(sender, value)

        # Process round
        any_decided = False
        for p in processes:
            if p.process_round(n):
                any_decided = True

        prefs = [p.preference for p in processes]
        decided_flags = [p.decided for p in processes]
        print(
            f"Round {round_num+1:2d}: preferences={prefs}, "
            f"decided={decided_flags}"
        )

        if all(p.decided for p in processes):
            decided = True
            break

    if not decided:
        print(
            f"\nAfter {max_rounds} rounds, consensus NOT reached."
        )
        print(
            "This demonstrates FLP: an adversary can delay messages "
            "to prevent termination in an asynchronous system."
        )
    print()


# === Exercise 2: Ben-Or's Randomized Consensus ===
# Problem: Implement Ben-Or's randomized binary consensus protocol.
# The protocol circumvents FLP by using randomization to eventually
# break symmetry. It works for f < n/2 crash failures.

class BenOrProcess:
    """
    Ben-Or's randomized binary consensus protocol.

    Phase 1: Broadcast preference, collect values.
             If > n/2 have same value v, set preference to v.
    Phase 2: Report preference or '?'. If > f+1 report same v, decide v.
             Otherwise, if any report v, adopt v; else flip a coin.
    """

    def __init__(self, pid: int, initial_value: int, n: int, f: int):
        self.pid = pid
        self.preference = initial_value
        self.n = n
        self.f = f
        self.decided = False
        self.decided_value = None
        self.phase1_msgs: List[int] = []
        self.phase2_msgs: List[Optional[int]] = []

    def phase1_propose(self) -> int:
        """Broadcast current preference."""
        return self.preference

    def phase1_collect(self, values: List[int]) -> Optional[int]:
        """
        Collect phase-1 values. If a majority exists, propose that
        value for phase 2. Otherwise, propose '?' (None).
        """
        count = [0, 0]
        for v in values:
            count[v] += 1
        threshold = (self.n // 2) + 1
        if count[0] >= threshold:
            return 0
        elif count[1] >= threshold:
            return 1
        return None

    def phase2_decide(self, reports: List[Optional[int]]) -> bool:
        """
        Phase 2 decision. If enough processes agree, decide.
        Otherwise, adopt a reported value or flip a coin.
        """
        if self.decided:
            return True

        count = {0: 0, 1: 0}
        for r in reports:
            if r is not None:
                count[r] += 1

        for v in [0, 1]:
            if count[v] >= self.f + 1:
                self.decided = True
                self.decided_value = v
                self.preference = v
                return True

        # Adopt any reported value, or flip coin
        for r in reports:
            if r is not None:
                self.preference = r
                return False

        self.preference = random.randint(0, 1)
        return False


def exercise_2():
    """
    Run Ben-Or's randomized consensus and show it terminates
    (with high probability).
    """
    print("=== Exercise 2: Ben-Or's Randomized Consensus ===\n")

    random.seed(42)
    n = 5
    f = 2  # tolerate 2 crash faults (f < n/2)
    max_rounds = 50

    processes = [BenOrProcess(i, random.randint(0, 1), n, f) for i in range(n)]
    initial = [p.preference for p in processes]
    print(f"Initial preferences: {initial}")

    for round_num in range(max_rounds):
        # Phase 1: everyone broadcasts preference
        phase1_values = [p.phase1_propose() for p in processes]

        # Phase 1 collect: everyone sees all values
        phase2_proposals = [p.phase1_collect(phase1_values) for p in processes]

        # Phase 2: everyone sees all proposals
        all_decided = True
        for p in processes:
            if not p.phase2_decide(phase2_proposals):
                all_decided = False

        prefs = [p.preference for p in processes]
        decs = [p.decided_value for p in processes]
        print(f"Round {round_num+1:2d}: prefs={prefs}, decided={decs}")

        if all(p.decided for p in processes):
            print(f"\nConsensus reached in round {round_num+1}!")
            print(f"Decided value: {processes[0].decided_value}")
            break
    else:
        print(f"\nDid not converge in {max_rounds} rounds (unlikely but possible).")
    print()


# === Exercise 3: Failure Detector Simulator ===
# Problem: Implement three types of failure detectors:
# - Perfect (P): No false positives, eventually detects all failures
# - Eventually Perfect (<>P): May have false positives initially,
#   eventually accurate
# - Eventually Strong (<>S): Eventually, one correct process is not
#   suspected by any correct process

class FailureDetector:
    """Base class for failure detectors."""

    def __init__(self, all_processes: Set[str], correct_processes: Set[str]):
        self.all_processes = all_processes
        self.correct_processes = correct_processes
        self.crashed_processes = all_processes - correct_processes
        self.time_step = 0

    def suspected(self, observer: str) -> Set[str]:
        raise NotImplementedError

    def advance_time(self):
        self.time_step += 1


class PerfectFailureDetector(FailureDetector):
    """
    Perfect failure detector (P):
    - Strong completeness: every crashed process is eventually suspected
    - Strong accuracy: no correct process is ever suspected
    """

    def __init__(self, all_procs: Set[str], correct: Set[str], detection_delay: int = 2):
        super().__init__(all_procs, correct)
        self.detection_delay = detection_delay

    def suspected(self, observer: str) -> Set[str]:
        if observer not in self.correct_processes:
            return set()
        if self.time_step >= self.detection_delay:
            return set(self.crashed_processes)
        return set()


class EventuallyPerfectFailureDetector(FailureDetector):
    """
    Eventually perfect failure detector (<>P):
    - Strong completeness
    - Eventual strong accuracy: after some time, no correct process
      is suspected
    """

    def __init__(
        self, all_procs: Set[str], correct: Set[str],
        stabilization_time: int = 5,
    ):
        super().__init__(all_procs, correct)
        self.stabilization_time = stabilization_time

    def suspected(self, observer: str) -> Set[str]:
        if observer not in self.correct_processes:
            return set()
        result = set(self.crashed_processes)
        if self.time_step < self.stabilization_time:
            # Before stabilization: may falsely suspect correct processes
            for p in self.correct_processes:
                if p != observer and random.random() < 0.3:
                    result.add(p)
        return result


class EventuallyStrongFailureDetector(FailureDetector):
    """
    Eventually strong failure detector (<>S):
    - Strong completeness
    - Eventual weak accuracy: eventually, some correct process is
      not suspected by any correct process (a 'trusted' leader)
    """

    def __init__(
        self, all_procs: Set[str], correct: Set[str],
        stabilization_time: int = 5,
    ):
        super().__init__(all_procs, correct)
        self.stabilization_time = stabilization_time
        self.trusted_leader = min(correct)  # deterministic choice

    def suspected(self, observer: str) -> Set[str]:
        if observer not in self.correct_processes:
            return set()
        result = set(self.crashed_processes)
        if self.time_step < self.stabilization_time:
            # May suspect anyone except self
            for p in self.correct_processes:
                if p != observer and random.random() < 0.4:
                    result.add(p)
        else:
            # After stabilization: never suspect the trusted leader
            result.discard(self.trusted_leader)
        return result


def exercise_3():
    """
    Compare the three failure detector classes.
    """
    print("=== Exercise 3: Failure Detector Simulator ===\n")

    random.seed(123)
    all_procs = {"P1", "P2", "P3", "P4", "P5"}
    correct = {"P1", "P2", "P3", "P4"}
    crashed = all_procs - correct

    print(f"Processes: {sorted(all_procs)}")
    print(f"Correct:   {sorted(correct)}")
    print(f"Crashed:   {sorted(crashed)}\n")

    detectors = {
        "Perfect (P)": PerfectFailureDetector(all_procs, correct, detection_delay=2),
        "Eventually Perfect (<>P)": EventuallyPerfectFailureDetector(
            all_procs, correct, stabilization_time=5
        ),
        "Eventually Strong (<>S)": EventuallyStrongFailureDetector(
            all_procs, correct, stabilization_time=5
        ),
    }

    for name, fd in detectors.items():
        print(f"--- {name} ---")
        for t in range(8):
            fd.time_step = t
            # Show suspicion from P1's perspective
            suspected = fd.suspected("P1")
            false_positives = suspected & correct
            missed_crashes = crashed - suspected
            print(
                f"  t={t}: suspected={sorted(suspected):20s} "
                f"FP={sorted(false_positives)} "
                f"missed={sorted(missed_crashes)}"
            )
        print()

    print("Perfect: no false positives, detects crash after delay.")
    print("<>P: false positives initially, stabilizes eventually.")
    print("<>S: eventually has a trusted correct process (leader).")
    print()


# === Main ===

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
