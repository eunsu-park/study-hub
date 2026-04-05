"""
Single-Decree Paxos Implementation

Simulates the Paxos consensus algorithm with Proposers, Acceptors, and
Learners. Demonstrates the two-phase protocol (Prepare/Promise, Accept/Accepted),
duelling proposers, and how Paxos guarantees safety even under message loss
and concurrent proposals.

Key concepts:
- Phase 1: Prepare(n) / Promise(n, accepted_value)
- Phase 2: Accept(n, v) / Accepted(n, v)
- Proposal numbers must be globally unique and monotonically increasing
- Duelling proposers can cause livelock (not safety violation)
- Majority quorum guarantees at most one value is chosen

Usage:
    python 12_paxos.py
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Proposal:
    """A proposal with a unique number and value."""
    number: int
    value: str | None = None


@dataclass
class PromiseResponse:
    """Acceptor's response to a Prepare request."""
    ok: bool
    acceptor_id: int
    highest_proposal: int
    accepted_proposal: int | None = None
    accepted_value: str | None = None


@dataclass
class AcceptResponse:
    """Acceptor's response to an Accept request."""
    ok: bool
    acceptor_id: int
    proposal_number: int


class Acceptor:
    """
    A Paxos acceptor. Maintains:
    - highest_prepare: highest proposal number promised
    - accepted_proposal: highest proposal number accepted
    - accepted_value: value of the accepted proposal
    """

    def __init__(self, acceptor_id: int):
        self.acceptor_id = acceptor_id
        self.highest_prepare: int = 0
        self.accepted_proposal: int | None = None
        self.accepted_value: str | None = None
        self.log: list[str] = []

    def prepare(self, proposal_number: int) -> PromiseResponse:
        """Handle a Prepare(n) request."""
        if proposal_number > self.highest_prepare:
            self.highest_prepare = proposal_number
            self.log.append(f"PROMISE n={proposal_number}")
            return PromiseResponse(
                ok=True,
                acceptor_id=self.acceptor_id,
                highest_proposal=proposal_number,
                accepted_proposal=self.accepted_proposal,
                accepted_value=self.accepted_value,
            )
        else:
            self.log.append(
                f"REJECT Prepare n={proposal_number} "
                f"(already promised n={self.highest_prepare})")
            return PromiseResponse(
                ok=False,
                acceptor_id=self.acceptor_id,
                highest_proposal=self.highest_prepare,
            )

    def accept(self, proposal_number: int, value: str) -> AcceptResponse:
        """Handle an Accept(n, v) request."""
        if proposal_number >= self.highest_prepare:
            self.highest_prepare = proposal_number
            self.accepted_proposal = proposal_number
            self.accepted_value = value
            self.log.append(f"ACCEPTED n={proposal_number} v='{value}'")
            return AcceptResponse(ok=True, acceptor_id=self.acceptor_id,
                                  proposal_number=proposal_number)
        else:
            self.log.append(
                f"REJECT Accept n={proposal_number} "
                f"(promised n={self.highest_prepare})")
            return AcceptResponse(ok=False, acceptor_id=self.acceptor_id,
                                  proposal_number=proposal_number)


class Proposer:
    """
    A Paxos proposer. Drives the two-phase protocol.
    """

    def __init__(self, proposer_id: int, initial_value: str,
                 n_acceptors: int):
        self.proposer_id = proposer_id
        self.value = initial_value
        self.n_acceptors = n_acceptors
        self.majority = n_acceptors // 2 + 1
        self.proposal_counter = proposer_id  # Ensure unique proposals
        self.log: list[str] = []

    def next_proposal_number(self) -> int:
        """Generate a unique, increasing proposal number."""
        self.proposal_counter += self.n_acceptors + 1
        return self.proposal_counter

    def run_phase1(self, acceptors: list[Acceptor],
                   proposal_number: int) -> tuple[bool, str | None]:
        """
        Phase 1: Send Prepare(n) to all acceptors.
        Returns (success, value_to_propose).
        """
        self.log.append(f"Phase1: Prepare(n={proposal_number})")
        promises: list[PromiseResponse] = []

        for acceptor in acceptors:
            resp = acceptor.prepare(proposal_number)
            if resp.ok:
                promises.append(resp)

        if len(promises) < self.majority:
            self.log.append(
                f"  Phase1 FAILED: got {len(promises)}/{self.majority} promises")
            return False, None

        self.log.append(
            f"  Phase1 OK: got {len(promises)} promises from "
            f"{[p.acceptor_id for p in promises]}")

        # Must use the value from the highest previously accepted proposal
        highest_accepted = None
        for p in promises:
            if p.accepted_proposal is not None:
                if (highest_accepted is None or
                        p.accepted_proposal > highest_accepted[0]):
                    highest_accepted = (p.accepted_proposal, p.accepted_value)

        if highest_accepted is not None:
            value = highest_accepted[1]
            self.log.append(
                f"  Adopting previously accepted value '{value}' "
                f"from proposal {highest_accepted[0]}")
        else:
            value = self.value
            self.log.append(f"  No prior accepted value; proposing own: '{value}'")

        return True, value

    def run_phase2(self, acceptors: list[Acceptor],
                   proposal_number: int, value: str) -> bool:
        """
        Phase 2: Send Accept(n, v) to all acceptors.
        Returns True if majority accepted.
        """
        self.log.append(f"Phase2: Accept(n={proposal_number}, v='{value}')")
        accepted_count = 0

        for acceptor in acceptors:
            resp = acceptor.accept(proposal_number, value)
            if resp.ok:
                accepted_count += 1

        if accepted_count >= self.majority:
            self.log.append(
                f"  Phase2 OK: {accepted_count}/{self.majority} accepted "
                f"=> VALUE CHOSEN: '{value}'")
            return True
        else:
            self.log.append(
                f"  Phase2 FAILED: only {accepted_count}/{self.majority} accepted")
            return False


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_basic_paxos() -> None:
    """Single proposer, no contention."""
    print("=" * 70)
    print("Basic Paxos: Single Proposer (No Contention)")
    print("=" * 70)

    acceptors = [Acceptor(i) for i in range(3)]
    proposer = Proposer(0, "alpha", n_acceptors=3)

    n = proposer.next_proposal_number()
    ok, value = proposer.run_phase1(acceptors, n)
    if ok and value:
        proposer.run_phase2(acceptors, n, value)

    print("\n  Proposer log:")
    for line in proposer.log:
        print(f"    {line}")

    print("\n  Acceptor states:")
    for a in acceptors:
        print(f"    A{a.acceptor_id}: accepted=({a.accepted_proposal}, "
              f"'{a.accepted_value}'), promised={a.highest_prepare}")


def demo_duelling_proposers() -> None:
    """Two proposers competing — shows safety and potential livelock."""
    print("\n" + "=" * 70)
    print("Duelling Proposers: Two Proposers Compete")
    print("=" * 70)

    acceptors = [Acceptor(i) for i in range(5)]
    p1 = Proposer(0, "alpha", n_acceptors=5)
    p2 = Proposer(1, "beta", n_acceptors=5)

    print("\n  P1 wants 'alpha', P2 wants 'beta'\n")

    # P1 Phase 1
    n1 = p1.next_proposal_number()
    ok1, v1 = p1.run_phase1(acceptors, n1)

    # P2 Phase 1 with higher number (preempts P1)
    n2 = p2.next_proposal_number()
    ok2, v2 = p2.run_phase1(acceptors, n2)

    # P1 Phase 2 — will fail because acceptors promised higher number to P2
    if ok1 and v1:
        p1.run_phase2(acceptors, n1, v1)

    # P2 Phase 2 — should succeed
    if ok2 and v2:
        p2.run_phase2(acceptors, n2, v2)

    print("  Proposer 1 (wants 'alpha'):")
    for line in p1.log:
        print(f"    {line}")
    print("\n  Proposer 2 (wants 'beta'):")
    for line in p2.log:
        print(f"    {line}")

    # Safety check: all accepted values must be the same
    accepted_values = set()
    for a in acceptors:
        if a.accepted_value is not None:
            accepted_values.add(a.accepted_value)

    print(f"\n  Accepted values across acceptors: {accepted_values}")
    print(f"  Safety maintained: {len(accepted_values) <= 1}")


def demo_previous_value_adoption() -> None:
    """Show that a proposer must adopt a previously accepted value."""
    print("\n" + "=" * 70)
    print("Value Adoption: Proposer Must Respect Prior Accepted Values")
    print("=" * 70)

    acceptors = [Acceptor(i) for i in range(3)]
    p1 = Proposer(0, "alpha", n_acceptors=3)
    p2 = Proposer(1, "beta", n_acceptors=3)

    # P1 completes full Paxos round
    n1 = p1.next_proposal_number()
    ok1, v1 = p1.run_phase1(acceptors, n1)
    if ok1 and v1:
        p1.run_phase2(acceptors, n1, v1)

    print("\n  Phase 1: P1 proposes and gets 'alpha' accepted\n")

    # P2 tries to propose 'beta' but must discover 'alpha' in Phase 1
    n2 = p2.next_proposal_number()
    ok2, v2 = p2.run_phase1(acceptors, n2)
    if ok2 and v2:
        p2.run_phase2(acceptors, n2, v2)

    print("  Phase 2: P2 tries to propose 'beta':\n")
    for line in p2.log:
        print(f"    {line}")

    print(f"\n  P2 was forced to adopt 'alpha' — Paxos safety preserved!")


def demo_livelock() -> None:
    """Simulate proposer livelock scenario."""
    print("\n" + "=" * 70)
    print("Livelock: Proposers Keep Preempting Each Other")
    print("=" * 70)

    acceptors = [Acceptor(i) for i in range(3)]
    p1 = Proposer(0, "alpha", n_acceptors=3)
    p2 = Proposer(1, "beta", n_acceptors=3)

    decided = False
    rounds = 0
    max_rounds = 6

    print(f"\n  Simulating {max_rounds} rounds of duelling proposers:\n")

    while not decided and rounds < max_rounds:
        rounds += 1

        # P1 tries
        n1 = p1.next_proposal_number()
        ok1, v1 = p1.run_phase1(acceptors, n1)

        # P2 immediately preempts
        n2 = p2.next_proposal_number()
        ok2, v2 = p2.run_phase1(acceptors, n2)

        # P1 Phase 2 fails
        if ok1 and v1:
            result1 = p1.run_phase2(acceptors, n1, v1)
            if result1:
                decided = True

        # P2 also fails if P1 comes back
        if not decided and ok2 and v2:
            result2 = p2.run_phase2(acceptors, n2, v2)
            if result2:
                decided = True

        status = "DECIDED" if decided else "no decision"
        print(f"    Round {rounds}: P1 n={n1}, P2 n={n2} => {status}")

    if not decided:
        print(f"\n  Livelock after {rounds} rounds! Neither proposer succeeded.")
        print(f"  Solution: Use a distinguished proposer (Multi-Paxos leader)")
    else:
        print(f"\n  Eventually decided in round {rounds}")


def demo_summary() -> None:
    """Print Paxos summary."""
    print("\n" + "=" * 70)
    print("Paxos Summary")
    print("=" * 70)

    print("""
  Single-Decree Paxos guarantees:

  Safety:
  - Only a single value is chosen (even with duelling proposers)
  - A proposer MUST adopt a previously accepted value

  Liveness:
  - NOT guaranteed with multiple competing proposers (livelock)
  - Multi-Paxos uses a stable leader to avoid livelock

  Protocol flow:
    Proposer           Acceptors          Learner
       |---Prepare(n)--->|                   |
       |<--Promise(n,v)--|                   |
       |---Accept(n,v)-->|                   |
       |<--Accepted(n,v)-|---Accepted(n,v)-->|
       |                 |                   |

  Quorum: majority of acceptors (⌊N/2⌋ + 1)

  Variants:
  - Multi-Paxos: stable leader skips Phase 1 for subsequent slots
  - FPaxos: flexible quorums (Phase 1 and 2 quorums need not be identical)
  - EPaxos: leaderless; any replica can propose with fast path
""")


if __name__ == "__main__":
    demo_basic_paxos()
    demo_duelling_proposers()
    demo_previous_value_adoption()
    demo_livelock()
    demo_summary()
    print("Done.")
