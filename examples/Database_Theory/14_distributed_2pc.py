"""
Two-Phase Commit (2PC) Protocol

Demonstrates:
- Coordinator-participant model
- Prepare/commit phases
- Participant failure handling
- Blocking nature of 2PC

Theory:
- 2PC ensures atomic commits across distributed participants.
- Phase 1 (Prepare): coordinator asks all participants to prepare.
  Each votes YES (can commit) or NO (must abort).
- Phase 2 (Commit/Abort): if all vote YES, coordinator sends COMMIT.
  If any votes NO, coordinator sends ABORT.
- Blocking: if coordinator crashes after prepare, participants
  holding locks are stuck until coordinator recovers.

Adapted from Database Theory Lesson 14.
"""

from enum import Enum
from dataclasses import dataclass, field
import random


class Vote(Enum):
    YES = "YES"
    NO = "NO"
    TIMEOUT = "TIMEOUT"


class Decision(Enum):
    COMMIT = "COMMIT"
    ABORT = "ABORT"
    UNKNOWN = "UNKNOWN"


@dataclass
class Participant:
    name: str
    data: dict[str, str] = field(default_factory=dict)
    prepared: bool = False
    committed: bool = False
    alive: bool = True
    failure_prob: float = 0.0  # Probability of voting NO

    def prepare(self, txn_data: dict) -> Vote:
        if not self.alive:
            return Vote.TIMEOUT
        if random.random() < self.failure_prob:
            return Vote.NO
        self.prepared = True
        self.pending_data = txn_data
        return Vote.YES

    def commit(self) -> bool:
        if not self.alive:
            return False
        if self.prepared:
            self.data.update(self.pending_data)
            self.committed = True
            self.prepared = False
            return True
        return False

    def abort(self) -> bool:
        self.prepared = False
        self.pending_data = {}
        return True


class Coordinator:
    """2PC Coordinator."""

    def __init__(self, participants: list[Participant]):
        self.participants = participants
        self.log: list[str] = []
        self.decision = Decision.UNKNOWN

    def execute(self, txn_data: dict) -> Decision:
        """Execute a distributed transaction using 2PC."""
        self.log.append("Phase 1: PREPARE")

        # Phase 1: Prepare
        votes: dict[str, Vote] = {}
        for p in self.participants:
            vote = p.prepare(txn_data)
            votes[p.name] = vote
            self.log.append(f"  {p.name} votes {vote.value}")

        # Decision
        all_yes = all(v == Vote.YES for v in votes.values())

        if all_yes:
            self.decision = Decision.COMMIT
            self.log.append(f"\nDecision: COMMIT (all voted YES)")
            self.log.append(f"\nPhase 2: COMMIT")
            for p in self.participants:
                success = p.commit()
                status = "OK" if success else "FAILED"
                self.log.append(f"  {p.name}: commit {status}")
        else:
            self.decision = Decision.ABORT
            no_voters = [n for n, v in votes.items() if v != Vote.YES]
            self.log.append(f"\nDecision: ABORT (NO from: {no_voters})")
            self.log.append(f"\nPhase 2: ABORT")
            for p in self.participants:
                p.abort()
                self.log.append(f"  {p.name}: abort OK")

        return self.decision


# ── Demos ──────────────────────────────────────────────────────────────

def demo_successful_2pc():
    print("=" * 60)
    print("2PC: SUCCESSFUL COMMIT")
    print("=" * 60)

    random.seed(42)
    participants = [
        Participant("DB-East", data={"balance": "1000"}),
        Participant("DB-West", data={"balance": "500"}),
        Participant("DB-Central", data={"balance": "2000"}),
    ]

    coord = Coordinator(participants)
    txn = {"balance": "1500"}

    print(f"\n  Transaction: Update balance to 1500 across 3 DBs")
    print(f"  Before: {[(p.name, p.data) for p in participants]}\n")

    result = coord.execute(txn)
    for msg in coord.log:
        print(f"  {msg}")

    print(f"\n  Result: {result.value}")
    print(f"  After: {[(p.name, p.data) for p in participants]}")
    print(f"  All consistent: {all(p.data == txn for p in participants)}")


def demo_participant_failure():
    print("\n" + "=" * 60)
    print("2PC: PARTICIPANT VOTES NO")
    print("=" * 60)

    random.seed(42)
    participants = [
        Participant("DB-East", data={"balance": "1000"}),
        Participant("DB-West", data={"balance": "500"}, failure_prob=1.0),
        Participant("DB-Central", data={"balance": "2000"}),
    ]

    coord = Coordinator(participants)
    txn = {"balance": "1500"}

    print(f"\n  Transaction: Update balance (DB-West will reject)")
    print(f"  Before: {[(p.name, p.data) for p in participants]}\n")

    result = coord.execute(txn)
    for msg in coord.log:
        print(f"  {msg}")

    print(f"\n  Result: {result.value}")
    print(f"  After: {[(p.name, p.data) for p in participants]}")
    print(f"  Data unchanged (atomic abort)")


def demo_network_partition():
    print("\n" + "=" * 60)
    print("2PC: PARTICIPANT UNREACHABLE")
    print("=" * 60)

    participants = [
        Participant("DB-East", data={"x": "1"}),
        Participant("DB-West", data={"x": "1"}, alive=False),
        Participant("DB-Central", data={"x": "1"}),
    ]

    coord = Coordinator(participants)

    print(f"\n  DB-West is unreachable (network partition)")
    result = coord.execute({"x": "2"})
    for msg in coord.log:
        print(f"  {msg}")

    print(f"\n  Result: {result.value}")
    print(f"  TIMEOUT treated as NO → entire transaction aborted")


def demo_blocking_problem():
    print("\n" + "=" * 60)
    print("2PC BLOCKING PROBLEM")
    print("=" * 60)

    print(f"""
  Scenario: Coordinator crashes after sending PREPARE

  Timeline:
    t=0  Coordinator → PREPARE → All participants
    t=1  All participants vote YES
    t=2  Coordinator receives votes, decides COMMIT
    t=3  *** Coordinator CRASHES before sending COMMIT ***

  Problem:
    - Participants have voted YES and hold locks
    - They don't know the decision (COMMIT or ABORT?)
    - They CANNOT release locks safely:
      → If they commit: other participants might have been told ABORT
      → If they abort: coordinator might have decided COMMIT
    - They must WAIT for coordinator recovery → BLOCKING

  Solution approaches:
    1. Coordinator WAL: writes decision to log before Phase 2
       → On recovery, reads log and completes Phase 2
    2. 3PC (Three-Phase Commit): adds pre-commit phase
       → Non-blocking but more complex, still fails with partitions
    3. Paxos/Raft-based commit: replicated coordinator
       → No single point of failure

  2PC vs 3PC vs Paxos:
    {'Protocol':<12} {'Blocking':>9} {'Messages':>9} {'Complexity':>11}
    {'-'*12} {'-'*9} {'-'*9} {'-'*11}
    {'2PC':<12} {'Yes':>9} {'3N':>9} {'Low':>11}
    {'3PC':<12} {'No*':>9} {'4N':>9} {'Medium':>11}
    {'Paxos':<12} {'No':>9} {'4N+':>9} {'High':>11}

    *3PC is non-blocking only without network partitions""")


if __name__ == "__main__":
    demo_successful_2pc()
    demo_participant_failure()
    demo_network_partition()
    demo_blocking_problem()
