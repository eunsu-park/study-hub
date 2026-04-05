"""
Exercises for Lesson 08: Distributed Transactions
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import time
import random
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# === Exercise 1: 2PC with WAL Logging and Timeout ===
# Problem: Implement the Two-Phase Commit protocol with:
# - Write-Ahead Logging (WAL) for crash recovery
# - Timeout-based abort when participants don't respond

class TxState(Enum):
    INIT = "INIT"
    PREPARED = "PREPARED"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"


@dataclass
class WALEntry:
    """Write-ahead log entry."""
    tx_id: str
    state: TxState
    timestamp: float


class TwoPhaseCoordinator:
    """
    2PC coordinator with WAL logging and timeout-based abort.
    """

    def __init__(self, coordinator_id: str, timeout_ms: float = 1000):
        self.coordinator_id = coordinator_id
        self.timeout_ms = timeout_ms
        self.wal: List[WALEntry] = []
        self.tx_states: Dict[str, TxState] = {}

    def _log(self, tx_id: str, state: TxState):
        """Write to WAL before changing state."""
        self.wal.append(WALEntry(tx_id, state, time.time()))
        self.tx_states[tx_id] = state

    def begin_transaction(self, tx_id: str, participants: List["TwoPhaseParticipant"]) -> bool:
        """
        Execute 2PC for a transaction.

        Phase 1: Prepare - ask all participants to vote.
        Phase 2: Commit/Abort based on votes.

        Returns True if committed, False if aborted.
        """
        self._log(tx_id, TxState.INIT)

        # Phase 1: Prepare
        votes = {}
        for p in participants:
            vote = p.prepare(tx_id)
            votes[p.participant_id] = vote
            if vote is None:
                # Timeout - treat as abort
                votes[p.participant_id] = False

        all_yes = all(votes.values())

        if all_yes:
            # Phase 2: Commit
            self._log(tx_id, TxState.COMMITTED)
            for p in participants:
                p.commit(tx_id)
            return True
        else:
            # Phase 2: Abort
            self._log(tx_id, TxState.ABORTED)
            for p in participants:
                p.abort(tx_id)
            return False

    def recover(self) -> Dict[str, TxState]:
        """
        Recover transaction states from WAL after crash.
        """
        recovered = {}
        for entry in self.wal:
            recovered[entry.tx_id] = entry.state
        return recovered


class TwoPhaseParticipant:
    """2PC participant with WAL logging."""

    def __init__(self, participant_id: str, fail_on_prepare: bool = False):
        self.participant_id = participant_id
        self.fail_on_prepare = fail_on_prepare
        self.wal: List[WALEntry] = []
        self.tx_states: Dict[str, TxState] = {}
        self.data: Dict[str, int] = {}

    def _log(self, tx_id: str, state: TxState):
        self.wal.append(WALEntry(tx_id, state, time.time()))
        self.tx_states[tx_id] = state

    def prepare(self, tx_id: str) -> Optional[bool]:
        """Vote YES or NO on prepare. Returns None for timeout."""
        if self.fail_on_prepare:
            return None  # Simulate timeout
        self._log(tx_id, TxState.PREPARED)
        return True

    def commit(self, tx_id: str):
        self._log(tx_id, TxState.COMMITTED)

    def abort(self, tx_id: str):
        self._log(tx_id, TxState.ABORTED)


def exercise_1():
    """
    Demonstrate 2PC with WAL and timeout-based abort.
    """
    print("=== Exercise 1: 2PC with WAL and Timeout ===\n")

    # Scenario 1: All participants vote YES
    coord = TwoPhaseCoordinator("C1")
    p1 = TwoPhaseParticipant("P1")
    p2 = TwoPhaseParticipant("P2")
    p3 = TwoPhaseParticipant("P3")

    result = coord.begin_transaction("TX001", [p1, p2, p3])
    print(f"TX001 (all vote YES): {'COMMITTED' if result else 'ABORTED'}")
    assert result is True

    # Scenario 2: One participant times out
    p4 = TwoPhaseParticipant("P4", fail_on_prepare=True)
    result2 = coord.begin_transaction("TX002", [p1, p2, p4])
    print(f"TX002 (P4 timeout):   {'COMMITTED' if result2 else 'ABORTED'}")
    assert result2 is False

    # Show WAL
    print(f"\nCoordinator WAL:")
    for entry in coord.wal:
        print(f"  {entry.tx_id}: {entry.state.value}")

    # Recovery
    recovered = coord.recover()
    print(f"\nRecovered states: {recovered}")
    print()


# === Exercise 2: Saga Pattern with Compensating Transactions ===
# Problem: Implement the Saga pattern where a long-running transaction
# is broken into a sequence of local transactions, each with a
# compensating transaction for rollback.

@dataclass
class SagaStep:
    """A single step in a saga with its compensation."""
    name: str
    action: callable
    compensation: callable
    executed: bool = False
    compensated: bool = False


class SagaOrchestrator:
    """
    Orchestrates a saga: execute steps in order, compensate in
    reverse order on failure.
    """

    def __init__(self, saga_id: str):
        self.saga_id = saga_id
        self.steps: List[SagaStep] = []
        self.completed = False
        self.compensated = False
        self.log: List[str] = []

    def add_step(self, name: str, action: callable, compensation: callable):
        self.steps.append(SagaStep(name, action, compensation))

    def execute(self) -> bool:
        """
        Execute the saga. On failure, run compensating transactions
        in reverse order.
        """
        executed_steps = []

        for step in self.steps:
            try:
                self.log.append(f"Executing: {step.name}")
                step.action()
                step.executed = True
                executed_steps.append(step)
                self.log.append(f"  Success: {step.name}")
            except Exception as e:
                self.log.append(f"  Failed: {step.name} ({e})")
                # Compensate in reverse order
                self._compensate(executed_steps)
                return False

        self.completed = True
        return True

    def _compensate(self, executed_steps: List[SagaStep]):
        """Run compensating transactions in reverse order."""
        self.log.append("Starting compensation...")
        for step in reversed(executed_steps):
            try:
                self.log.append(f"  Compensating: {step.name}")
                step.compensation()
                step.compensated = True
            except Exception as e:
                self.log.append(f"  Compensation FAILED: {step.name} ({e})")
        self.compensated = True


def exercise_2():
    """
    Demonstrate the Saga pattern with compensation on failure.
    """
    print("=== Exercise 2: Saga Pattern ===\n")

    # Simulate a travel booking saga
    booking_state = {
        "flight": None,
        "hotel": None,
        "car": None,
        "balance": 1000,
    }

    def book_flight():
        booking_state["flight"] = "FL123"
        booking_state["balance"] -= 300

    def cancel_flight():
        booking_state["flight"] = None
        booking_state["balance"] += 300

    def book_hotel():
        booking_state["hotel"] = "HT456"
        booking_state["balance"] -= 200

    def cancel_hotel():
        booking_state["hotel"] = None
        booking_state["balance"] += 200

    def book_car():
        # Simulate failure
        raise RuntimeError("Car rental service unavailable")

    def cancel_car():
        booking_state["car"] = None

    # Saga 1: Car booking fails, triggers compensation
    saga = SagaOrchestrator("SAGA-001")
    saga.add_step("Book Flight", book_flight, cancel_flight)
    saga.add_step("Book Hotel", book_hotel, cancel_hotel)
    saga.add_step("Book Car", book_car, cancel_car)

    print("Saga 1: Travel booking (car service fails)")
    result = saga.execute()
    print(f"Result: {'SUCCESS' if result else 'COMPENSATED'}")
    print(f"Booking state: {booking_state}")
    for entry in saga.log:
        print(f"  {entry}")

    assert booking_state["flight"] is None, "Flight should be cancelled"
    assert booking_state["hotel"] is None, "Hotel should be cancelled"
    assert booking_state["balance"] == 1000, "Balance should be restored"

    # Saga 2: All steps succeed
    def book_car_ok():
        booking_state["car"] = "CR789"
        booking_state["balance"] -= 100

    saga2 = SagaOrchestrator("SAGA-002")
    saga2.add_step("Book Flight", book_flight, cancel_flight)
    saga2.add_step("Book Hotel", book_hotel, cancel_hotel)
    saga2.add_step("Book Car", book_car_ok, cancel_car)

    print(f"\nSaga 2: Travel booking (all succeed)")
    result2 = saga2.execute()
    print(f"Result: {'SUCCESS' if result2 else 'COMPENSATED'}")
    print(f"Booking state: {booking_state}")
    assert result2 is True
    print()


# === Exercise 3: 2PC vs 3PC Under Coordinator Failure ===
# Problem: Compare the blocking behavior of 2PC and 3PC when the
# coordinator fails. Show that 2PC blocks (participants are uncertain)
# while 3PC can make progress with a timeout protocol.

class ThreePhaseCoordinator:
    """
    3PC coordinator that adds a pre-commit phase to reduce blocking.
    """

    def __init__(self, coordinator_id: str):
        self.coordinator_id = coordinator_id
        self.state = TxState.INIT
        self.crash_after_phase: Optional[int] = None

    def run_3pc(
        self, tx_id: str, participants: List["ThreePhaseParticipant"]
    ) -> Optional[bool]:
        """
        Run 3PC. Returns True (commit), False (abort), None (crashed).
        """
        # Phase 1: Can-Commit?
        votes = []
        for p in participants:
            votes.append(p.can_commit(tx_id))
        if self.crash_after_phase == 1:
            return None  # coordinator crash

        if not all(votes):
            for p in participants:
                p.do_abort(tx_id)
            return False

        # Phase 2: Pre-Commit
        for p in participants:
            p.pre_commit(tx_id)
        if self.crash_after_phase == 2:
            return None  # coordinator crash

        # Phase 3: Do-Commit
        for p in participants:
            p.do_commit(tx_id)
        return True


class ThreePhaseParticipant:
    """3PC participant with timeout-based recovery."""

    def __init__(self, pid: str):
        self.pid = pid
        self.state = TxState.INIT
        self.pre_committed = False

    def can_commit(self, tx_id: str) -> bool:
        return True

    def pre_commit(self, tx_id: str):
        self.pre_committed = True
        self.state = TxState.PREPARED

    def do_commit(self, tx_id: str):
        self.state = TxState.COMMITTED

    def do_abort(self, tx_id: str):
        self.state = TxState.ABORTED

    def timeout_recovery(self) -> str:
        """
        If coordinator crashes and participant times out:
        - In INIT state: safe to abort
        - In pre-committed state: can coordinate with others
        """
        if not self.pre_committed:
            self.state = TxState.ABORTED
            return "abort (safe: not pre-committed)"
        else:
            # In 3PC, pre-committed participants can elect a new coordinator
            return "can elect new coordinator (pre-committed)"


def exercise_3():
    """
    Compare 2PC vs 3PC under coordinator failure.
    """
    print("=== Exercise 3: 2PC vs 3PC Under Coordinator Failure ===\n")

    # 2PC: Coordinator crashes after Phase 1
    print("--- 2PC: Coordinator crashes after Phase 1 (Prepare) ---")
    coord_2pc = TwoPhaseCoordinator("2PC-C1")
    p1 = TwoPhaseParticipant("2PC-P1")
    p2 = TwoPhaseParticipant("2PC-P2")

    # Simulate: participants voted YES, coordinator crashes before Phase 2
    p1.prepare("TX-2PC")
    p2.prepare("TX-2PC")
    print(f"  P1 state: {p1.tx_states.get('TX-2PC', 'UNKNOWN')}")
    print(f"  P2 state: {p2.tx_states.get('TX-2PC', 'UNKNOWN')}")
    print(f"  Coordinator crashed! Participants are BLOCKED.")
    print(f"  They cannot decide to commit or abort without coordinator.")

    # 3PC: Coordinator crashes after Phase 2 (pre-commit)
    print("\n--- 3PC: Coordinator crashes after Phase 2 (Pre-Commit) ---")
    coord_3pc = ThreePhaseCoordinator("3PC-C1")
    coord_3pc.crash_after_phase = 2
    q1 = ThreePhaseParticipant("3PC-P1")
    q2 = ThreePhaseParticipant("3PC-P2")

    result = coord_3pc.run_3pc("TX-3PC", [q1, q2])
    print(f"  Coordinator result: {result} (crashed)")
    print(f"  P1 pre-committed: {q1.pre_committed}, state: {q1.state}")
    print(f"  P2 pre-committed: {q2.pre_committed}, state: {q2.state}")

    recovery1 = q1.timeout_recovery()
    recovery2 = q2.timeout_recovery()
    print(f"  P1 timeout recovery: {recovery1}")
    print(f"  P2 timeout recovery: {recovery2}")

    print("\nKey difference:")
    print("  2PC: Participants BLOCK when coordinator fails after prepare.")
    print("  3PC: Pre-commit phase allows participants to recover via timeout.")
    print("  Trade-off: 3PC adds latency (extra round-trip) in the normal case.")
    print()


# === Main ===

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
