"""
Distributed Transaction Protocols: 2PC and Saga

Simulates Two-Phase Commit (2PC) and the Saga pattern for coordinating
distributed transactions across multiple services. Demonstrates blocking
behavior of 2PC on coordinator failure and compensating transactions in
the Saga pattern.

Key concepts:
- 2PC: Prepare/Commit with coordinator, blocking on failure
- 3PC: Non-blocking but requires synchronous model
- Saga: sequence of local transactions with compensations
- Choreography vs orchestration Saga patterns

Usage:
    python 14_distributed_transactions.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Two-Phase Commit (2PC)
# ---------------------------------------------------------------------------

class TwoPhaseCommitState(Enum):
    INIT = "init"
    PREPARED = "prepared"
    COMMITTED = "committed"
    ABORTED = "aborted"
    BLOCKED = "blocked"     # Waiting for crashed coordinator


@dataclass
class Participant:
    """A participant in 2PC."""
    name: str
    state: TwoPhaseCommitState = TwoPhaseCommitState.INIT
    can_prepare: bool = True   # Simulates whether participant can commit
    crashed: bool = False
    log: list[str] = field(default_factory=list)

    def prepare(self) -> bool:
        """Phase 1: Vote to commit or abort."""
        if self.crashed:
            self.log.append("CRASHED — cannot vote")
            return False
        if self.can_prepare:
            self.state = TwoPhaseCommitState.PREPARED
            self.log.append("VOTE: YES (prepared)")
            return True
        else:
            self.state = TwoPhaseCommitState.ABORTED
            self.log.append("VOTE: NO (abort)")
            return False

    def commit(self) -> None:
        self.state = TwoPhaseCommitState.COMMITTED
        self.log.append("COMMITTED")

    def abort(self) -> None:
        self.state = TwoPhaseCommitState.ABORTED
        self.log.append("ABORTED")


class TwoPhaseCommitCoordinator:
    """Coordinator for 2PC."""

    def __init__(self, participants: list[Participant]):
        self.participants = participants
        self.crashed = False
        self.crash_after_phase = 0   # 0=no crash, 1=after prepare
        self.log: list[str] = []

    def execute(self) -> str:
        """Run the 2PC protocol. Returns 'committed', 'aborted', or 'blocked'."""
        self.log.append("=== Phase 1: PREPARE ===")

        votes = []
        for p in self.participants:
            vote = p.prepare()
            votes.append(vote)
            self.log.append(f"  {p.name}: {'YES' if vote else 'NO'}")

        # Simulate coordinator crash after Phase 1
        if self.crash_after_phase == 1:
            self.crashed = True
            self.log.append("\n*** COORDINATOR CRASHED after Phase 1 ***")
            self.log.append("Participants are BLOCKED — cannot decide alone!")
            for p in self.participants:
                if p.state == TwoPhaseCommitState.PREPARED:
                    p.state = TwoPhaseCommitState.BLOCKED
                    p.log.append("BLOCKED — waiting for coordinator")
            return "blocked"

        self.log.append("\n=== Phase 2: DECISION ===")

        if all(votes):
            self.log.append("  Decision: COMMIT (all voted YES)")
            for p in self.participants:
                if not p.crashed:
                    p.commit()
            return "committed"
        else:
            self.log.append("  Decision: ABORT (at least one voted NO)")
            for p in self.participants:
                if not p.crashed:
                    p.abort()
            return "aborted"


# ---------------------------------------------------------------------------
# Saga Pattern
# ---------------------------------------------------------------------------

@dataclass
class SagaStep:
    """A single step in a Saga with its compensating action."""
    name: str
    action: str
    compensation: str
    will_fail: bool = False    # Simulate failure
    executed: bool = False
    compensated: bool = False


class SagaOrchestrator:
    """Orchestrates a Saga: executes steps forward, compensates on failure."""

    def __init__(self, steps: list[SagaStep]):
        self.steps = steps
        self.log: list[str] = []
        self.completed_steps: list[SagaStep] = []

    def execute(self) -> bool:
        """Run the Saga. Returns True if all steps succeeded."""
        self.log.append("=== Saga Execution (Forward) ===")

        for step in self.steps:
            if step.will_fail:
                self.log.append(f"  [{step.name}] {step.action} => FAILED!")
                self.log.append(f"\n=== Saga Compensation (Backward) ===")
                self._compensate()
                return False
            else:
                step.executed = True
                self.completed_steps.append(step)
                self.log.append(f"  [{step.name}] {step.action} => OK")

        self.log.append("\n  Saga completed successfully!")
        return True

    def _compensate(self) -> None:
        """Execute compensating transactions in reverse order."""
        for step in reversed(self.completed_steps):
            step.compensated = True
            self.log.append(f"  [{step.name}] COMPENSATE: {step.compensation}")
        self.log.append("\n  All completed steps compensated.")


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_2pc_success() -> None:
    """Successful 2PC with all participants agreeing."""
    print("=" * 70)
    print("2PC: Successful Commit (All Participants Agree)")
    print("=" * 70)

    participants = [
        Participant("OrderService"),
        Participant("PaymentService"),
        Participant("InventoryService"),
    ]
    coord = TwoPhaseCommitCoordinator(participants)
    result = coord.execute()

    for line in coord.log:
        print(f"  {line}")
    print(f"\n  Final states:")
    for p in participants:
        print(f"    {p.name}: {p.state.value}")


def demo_2pc_abort() -> None:
    """2PC abort when one participant votes NO."""
    print("\n" + "=" * 70)
    print("2PC: Abort (Payment Service Cannot Commit)")
    print("=" * 70)

    participants = [
        Participant("OrderService"),
        Participant("PaymentService", can_prepare=False),
        Participant("InventoryService"),
    ]
    coord = TwoPhaseCommitCoordinator(participants)
    result = coord.execute()

    for line in coord.log:
        print(f"  {line}")
    print(f"\n  Final states:")
    for p in participants:
        print(f"    {p.name}: {p.state.value}")


def demo_2pc_coordinator_crash() -> None:
    """2PC blocking when coordinator crashes after Phase 1."""
    print("\n" + "=" * 70)
    print("2PC: Coordinator Crash (Blocking Problem)")
    print("=" * 70)

    participants = [
        Participant("OrderService"),
        Participant("PaymentService"),
        Participant("InventoryService"),
    ]
    coord = TwoPhaseCommitCoordinator(participants)
    coord.crash_after_phase = 1
    result = coord.execute()

    for line in coord.log:
        print(f"  {line}")
    print(f"\n  Final states:")
    for p in participants:
        print(f"    {p.name}: {p.state.value}")

    print(f"""
  Problem: Participants voted YES but coordinator crashed before
  sending COMMIT or ABORT. They cannot safely decide on their own:
  - Committing might violate atomicity (if coordinator chose ABORT)
  - Aborting might violate atomicity (if coordinator chose COMMIT)
  - They must WAIT for coordinator recovery => BLOCKING

  Solutions: 3PC (adds pre-commit phase) or Paxos-based commit.
""")


def demo_saga_success() -> None:
    """Successful Saga execution."""
    print("=" * 70)
    print("Saga: Successful Execution (All Steps Complete)")
    print("=" * 70)

    steps = [
        SagaStep("Order", "Create order #123", "Cancel order #123"),
        SagaStep("Payment", "Charge $50 to card", "Refund $50 to card"),
        SagaStep("Inventory", "Reserve 1 widget", "Release 1 widget"),
        SagaStep("Shipping", "Schedule delivery", "Cancel delivery"),
    ]
    saga = SagaOrchestrator(steps)
    saga.execute()

    for line in saga.log:
        print(f"  {line}")


def demo_saga_failure() -> None:
    """Saga with failure triggering compensation."""
    print("\n" + "=" * 70)
    print("Saga: Failure at Step 3 (Compensation Triggered)")
    print("=" * 70)

    steps = [
        SagaStep("Order", "Create order #456", "Cancel order #456"),
        SagaStep("Payment", "Charge $200 to card", "Refund $200 to card"),
        SagaStep("Inventory", "Reserve 5 widgets", "Release 5 widgets",
                 will_fail=True),
        SagaStep("Shipping", "Schedule delivery", "Cancel delivery"),
    ]
    saga = SagaOrchestrator(steps)
    saga.execute()

    for line in saga.log:
        print(f"  {line}")


def demo_comparison() -> None:
    """Compare 2PC vs Saga patterns."""
    print("\n" + "=" * 70)
    print("2PC vs Saga Comparison")
    print("=" * 70)

    print("""
  ┌────────────────────┬───────────────────────┬───────────────────────┐
  │ Property           │ 2PC                   │ Saga                  │
  ├────────────────────┼───────────────────────┼───────────────────────┤
  │ Consistency        │ Strong (ACID)         │ Eventual              │
  │ Isolation          │ Full (locks held)     │ None (intermediate    │
  │                    │                       │ states visible)       │
  │ Blocking           │ YES (coord crash)     │ NO                    │
  │ Latency            │ High (2 round-trips)  │ Lower (async steps)   │
  │ Compensation       │ Not needed            │ Must define per step  │
  │ Complexity         │ Low (protocol)        │ High (compensations)  │
  │ Use case           │ Database transactions │ Microservice workflows│
  │ Examples           │ Spanner, XA           │ Order processing,     │
  │                    │                       │ travel booking        │
  └────────────────────┴───────────────────────┴───────────────────────┘

  Other patterns:
  - Percolator (Google): Optimistic 2PC with BigTable timestamps
  - Spanner TrueTime: GPS-synchronized clocks for global consistency
  - Calvin: Deterministic database — pre-orders all transactions
""")


if __name__ == "__main__":
    demo_2pc_success()
    demo_2pc_abort()
    demo_2pc_coordinator_crash()
    demo_saga_success()
    demo_saga_failure()
    demo_comparison()
    print("Done.")
