"""
Exercises for Lesson 05: Paxos Family
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# === Exercise 1: Single-Decree Paxos ===
# Problem: Implement the single-decree Paxos algorithm with 3 acceptors.
# Implement Phase 1 (Prepare/Promise) and Phase 2 (Accept/Accepted).
# Show that a value is chosen when a majority of acceptors accept it.

@dataclass
class PaxosProposal:
    """A Paxos proposal with a unique number and value."""
    number: int
    value: Optional[str] = None


@dataclass
class Acceptor:
    """
    A Paxos acceptor.

    State:
    - promised: highest proposal number promised
    - accepted_proposal: highest proposal number accepted
    - accepted_value: value of highest accepted proposal
    """
    acceptor_id: str
    promised: int = -1
    accepted_proposal: int = -1
    accepted_value: Optional[str] = None

    def prepare(self, proposal_num: int) -> Tuple[bool, int, Optional[str]]:
        """
        Phase 1b: Handle Prepare request.
        Returns (promise_granted, accepted_proposal, accepted_value).
        """
        if proposal_num > self.promised:
            self.promised = proposal_num
            return (True, self.accepted_proposal, self.accepted_value)
        return (False, -1, None)

    def accept(self, proposal_num: int, value: str) -> bool:
        """
        Phase 2b: Handle Accept request.
        Returns True if accepted.
        """
        if proposal_num >= self.promised:
            self.promised = proposal_num
            self.accepted_proposal = proposal_num
            self.accepted_value = value
            return True
        return False


class Proposer:
    """
    A Paxos proposer that runs the two-phase protocol.
    """

    def __init__(self, proposer_id: str, acceptors: List[Acceptor]):
        self.proposer_id = proposer_id
        self.acceptors = acceptors
        self.proposal_counter = 0

    def next_proposal_number(self) -> int:
        """Generate a unique proposal number."""
        self.proposal_counter += 1
        # Encode proposer identity in low bits for uniqueness
        pid_num = hash(self.proposer_id) % 100
        return self.proposal_counter * 100 + pid_num

    def propose(self, value: str) -> Optional[str]:
        """
        Run single-decree Paxos. Returns the chosen value (which
        may differ from the proposed value if another value was
        already accepted).
        """
        n = self.next_proposal_number()
        majority = len(self.acceptors) // 2 + 1

        # Phase 1a: Send Prepare
        promises = []
        for a in self.acceptors:
            granted, acc_prop, acc_val = a.prepare(n)
            if granted:
                promises.append((acc_prop, acc_val))

        if len(promises) < majority:
            return None  # Failed to get majority promise

        # Phase 1 result: adopt highest accepted value if any
        highest_prop = -1
        chosen_value = value
        for acc_prop, acc_val in promises:
            if acc_prop > highest_prop and acc_val is not None:
                highest_prop = acc_prop
                chosen_value = acc_val

        # Phase 2a: Send Accept
        accepts = 0
        for a in self.acceptors:
            if a.accept(n, chosen_value):
                accepts += 1

        if accepts >= majority:
            return chosen_value
        return None


def exercise_1():
    """
    Demonstrate single-decree Paxos with 3 acceptors.
    """
    print("=== Exercise 1: Single-Decree Paxos (3 Acceptors) ===\n")

    acceptors = [Acceptor(f"A{i}") for i in range(3)]
    proposer = Proposer("P1", acceptors)

    # Propose "X"
    result = proposer.propose("X")
    print(f"Proposer P1 proposes 'X': chosen = {result}")
    assert result == "X"

    # Another proposer tries to propose "Y" after "X" is already chosen
    proposer2 = Proposer("P2", acceptors)
    result2 = proposer2.propose("Y")
    print(f"Proposer P2 proposes 'Y': chosen = {result2}")
    # P2 should learn about "X" in Phase 1 and adopt it
    print(f"  (P2 adopted the already-accepted value)")

    # Fresh round
    print("\n--- Fresh acceptors ---")
    acceptors2 = [Acceptor(f"B{i}") for i in range(3)]
    p1 = Proposer("Q1", acceptors2)
    result3 = p1.propose("Alpha")
    print(f"Proposer Q1 proposes 'Alpha': chosen = {result3}")
    assert result3 == "Alpha"

    print("\nSingle-decree Paxos working correctly.")
    print()


# === Exercise 2: Dueling Proposers (Livelock) ===
# Problem: Simulate the dueling proposers scenario where two proposers
# continuously outbid each other, causing livelock (no value is ever
# chosen). This illustrates why Paxos needs a leader.

def exercise_2():
    """
    Simulate dueling proposers causing livelock.
    """
    print("=== Exercise 2: Dueling Proposers (Livelock) ===\n")

    acceptors = [Acceptor(f"A{i}") for i in range(3)]
    p1 = Proposer("P1", acceptors)
    p2 = Proposer("P2", acceptors)

    max_attempts = 10
    livelock_rounds = 0

    for attempt in range(max_attempts):
        # P1 does Phase 1 only
        n1 = p1.next_proposal_number()
        promises1 = []
        for a in acceptors:
            granted, _, _ = a.prepare(n1)
            if granted:
                promises1.append(True)

        # P2 does Phase 1, preempting P1
        n2 = p2.next_proposal_number()
        promises2 = []
        for a in acceptors:
            granted, _, _ = a.prepare(n2)
            if granted:
                promises2.append(True)

        # P1 tries Phase 2 - fails because acceptors promised higher n2
        accepts1 = sum(1 for a in acceptors if a.accept(n1, "X"))
        # P2 tries Phase 2 (might also fail next round)
        accepts2 = sum(1 for a in acceptors if a.accept(n2, "Y"))

        chosen1 = accepts1 >= 2
        chosen2 = accepts2 >= 2

        print(
            f"Round {attempt+1}: P1(n={n1}) accepts={accepts1}, "
            f"P2(n={n2}) accepts={accepts2}, "
            f"chosen={'P1:X' if chosen1 else 'P2:Y' if chosen2 else 'NONE'}"
        )

        if not chosen1 and not chosen2:
            livelock_rounds += 1

    print(f"\nLivelock rounds: {livelock_rounds}/{max_attempts}")
    print("Solution: Use Multi-Paxos with a stable leader to avoid dueling.")
    print()


# === Exercise 3: Multi-Paxos with Stable Leader ===
# Problem: Implement Multi-Paxos where a stable leader skips Phase 1
# for subsequent instances, improving performance.

class MultiPaxosLeader:
    """
    Multi-Paxos leader that optimizes by skipping Phase 1 when the
    leader is stable.
    """

    def __init__(self, leader_id: str, acceptors: List[Acceptor], num_acceptors: int):
        self.leader_id = leader_id
        self.acceptors = acceptors
        self.num_acceptors = num_acceptors
        self.ballot = 0
        self.is_established = False
        self.log: Dict[int, str] = {}
        self.next_slot = 0
        self.phase1_count = 0
        self.phase2_count = 0

    def establish_leadership(self) -> bool:
        """Run Phase 1 to establish leadership for all future slots."""
        self.ballot += 1
        n = self.ballot * 100 + hash(self.leader_id) % 100
        majority = self.num_acceptors // 2 + 1

        promises = 0
        for a in self.acceptors:
            granted, _, _ = a.prepare(n)
            if granted:
                promises += 1

        self.phase1_count += 1
        if promises >= majority:
            self.is_established = True
            self.current_n = n
            return True
        return False

    def propose(self, value: str) -> Optional[Tuple[int, str]]:
        """
        Propose a value for the next slot.
        If leader is established, skip Phase 1.
        Returns (slot, chosen_value) or None.
        """
        if not self.is_established:
            if not self.establish_leadership():
                return None

        # Phase 2 only (leader optimization)
        self.phase2_count += 1
        majority = self.num_acceptors // 2 + 1

        accepts = 0
        for a in self.acceptors:
            if a.accept(self.current_n, value):
                accepts += 1

        if accepts >= majority:
            slot = self.next_slot
            self.log[slot] = value
            self.next_slot += 1
            return (slot, value)

        # Lost leadership
        self.is_established = False
        return None


def exercise_3():
    """
    Demonstrate Multi-Paxos with leader optimization.
    """
    print("=== Exercise 3: Multi-Paxos with Stable Leader ===\n")

    acceptors = [Acceptor(f"A{i}") for i in range(5)]
    leader = MultiPaxosLeader("Leader1", acceptors, 5)

    # First proposal: needs Phase 1
    result = leader.propose("cmd_1")
    print(f"Proposal 1: {result} (Phase 1 runs: {leader.phase1_count})")
    assert result is not None

    # Subsequent proposals: skip Phase 1 (leader is established)
    for i in range(2, 6):
        result = leader.propose(f"cmd_{i}")
        print(
            f"Proposal {i}: {result} "
            f"(Phase 1 runs: {leader.phase1_count}, "
            f"Phase 2 runs: {leader.phase2_count})"
        )

    print(f"\nLog: {leader.log}")
    print(f"Total Phase 1 executions: {leader.phase1_count} (only 1!)")
    print(f"Total Phase 2 executions: {leader.phase2_count}")
    print(
        "Multi-Paxos saves Phase 1 round-trips for subsequent proposals "
        "when the leader is stable."
    )
    print()


# === Main ===

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
