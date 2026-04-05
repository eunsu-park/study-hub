"""
TLA+ Formal Verification Concepts in Python

Demonstrates the core ideas behind TLA+ model checking by implementing
a simple state space explorer that verifies safety and liveness properties
of distributed protocols. This is NOT TLA+ itself, but illustrates the
verification methodology using Python.

Key concepts:
- State machines and state space exploration
- Safety properties: invariants that hold in every reachable state
- Liveness properties: something good eventually happens
- Model checking: exhaustive search of finite state spaces
- Deadlock detection
- How TLA+ finds protocol bugs

Usage:
    python 18_tlaplus_verification.py
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Simple Model Checker
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class State:
    """An immutable state in the system."""
    values: tuple  # Hashable representation of all state variables

    def __repr__(self) -> str:
        return f"State{self.values}"


class ModelChecker:
    """
    A simplified TLA+-style model checker.
    Explores the state space by applying all possible actions in every state
    and checking invariants.
    """

    def __init__(self, name: str):
        self.name = name
        self.initial_states: list[State] = []
        self.actions: list[tuple[str, Callable[[State], list[State]]]] = []
        self.invariants: list[tuple[str, Callable[[State], bool]]] = []
        self.liveness: list[tuple[str, Callable[[State], bool]]] = []
        self.log: list[str] = []

    def add_initial(self, state: State) -> None:
        self.initial_states.append(state)

    def add_action(self, name: str, fn: Callable[[State], list[State]]) -> None:
        self.actions.append((name, fn))

    def add_invariant(self, name: str, fn: Callable[[State], bool]) -> None:
        self.invariants.append((name, fn))

    def add_liveness(self, name: str, fn: Callable[[State], bool]) -> None:
        self.liveness.append((name, fn))

    def check(self, max_states: int = 10000) -> dict:
        """
        Run BFS model checking. Returns result dict.
        """
        visited: set[State] = set()
        queue: deque[tuple[State, list[str]]] = deque()
        deadlocks: list[tuple[State, list[str]]] = []
        violations: list[tuple[str, State, list[str]]] = []
        liveness_satisfied: dict[str, bool] = {name: False for name, _ in self.liveness}

        for s in self.initial_states:
            queue.append((s, ["INIT"]))
            visited.add(s)

        while queue and len(visited) < max_states:
            state, trace = queue.popleft()

            # Check invariants
            for inv_name, inv_fn in self.invariants:
                if not inv_fn(state):
                    violations.append((inv_name, state, trace))

            # Check liveness (has this property been satisfied in any state?)
            for live_name, live_fn in self.liveness:
                if live_fn(state):
                    liveness_satisfied[live_name] = True

            # Generate next states
            has_successor = False
            for act_name, act_fn in self.actions:
                next_states = act_fn(state)
                for ns in next_states:
                    has_successor = True
                    if ns not in visited:
                        visited.add(ns)
                        queue.append((ns, trace + [f"{act_name}->{ns}"]))

            if not has_successor:
                deadlocks.append((state, trace))

        return {
            "states_explored": len(visited),
            "violations": violations,
            "deadlocks": deadlocks,
            "liveness": liveness_satisfied,
        }


# ---------------------------------------------------------------------------
# Example 1: Mutex Protocol
# ---------------------------------------------------------------------------

def verify_mutex_correct() -> None:
    """Verify a correct mutual exclusion protocol."""
    print("=" * 70)
    print("Model Check: Correct Mutex Protocol")
    print("=" * 70)

    # State: (turn, p0_in_cs, p1_in_cs)
    # turn: whose turn it is (0 or 1)
    # p0_in_cs, p1_in_cs: whether process 0/1 is in critical section

    mc = ModelChecker("Mutex")
    mc.add_initial(State((0, False, False)))
    mc.add_initial(State((1, False, False)))

    def p0_enter(s: State) -> list[State]:
        turn, p0, p1 = s.values
        if turn == 0 and not p0:
            return [State((turn, True, p1))]
        return []

    def p0_exit(s: State) -> list[State]:
        turn, p0, p1 = s.values
        if p0:
            return [State((1, False, p1))]
        return []

    def p1_enter(s: State) -> list[State]:
        turn, p0, p1 = s.values
        if turn == 1 and not p1:
            return [State((turn, p0, True))]
        return []

    def p1_exit(s: State) -> list[State]:
        turn, p0, p1 = s.values
        if p1:
            return [State((0, p0, False))]
        return []

    mc.add_action("P0_enter", p0_enter)
    mc.add_action("P0_exit", p0_exit)
    mc.add_action("P1_enter", p1_enter)
    mc.add_action("P1_exit", p1_exit)

    # Safety: never both in CS
    mc.add_invariant("MutualExclusion",
                     lambda s: not (s.values[1] and s.values[2]))

    result = mc.check()
    print(f"\n  States explored: {result['states_explored']}")
    print(f"  Invariant violations: {len(result['violations'])}")
    print(f"  Deadlocks: {len(result['deadlocks'])}")

    if not result["violations"]:
        print(f"  Mutual exclusion: VERIFIED")
    else:
        for name, state, trace in result["violations"]:
            print(f"  VIOLATION of {name} at {state}")
            print(f"    Trace: {' -> '.join(trace[:5])}")


def verify_mutex_buggy() -> None:
    """Verify a BUGGY mutex protocol that allows both in CS."""
    print("\n" + "=" * 70)
    print("Model Check: BUGGY Mutex Protocol (No Turn Variable)")
    print("=" * 70)

    # Buggy: no turn variable, both can enter simultaneously
    # State: (p0_in_cs, p1_in_cs)
    mc = ModelChecker("BuggyMutex")
    mc.add_initial(State((False, False)))

    def p0_enter(s: State) -> list[State]:
        p0, p1 = s.values
        if not p0:
            return [State((True, p1))]
        return []

    def p0_exit(s: State) -> list[State]:
        p0, p1 = s.values
        if p0:
            return [State((False, p1))]
        return []

    def p1_enter(s: State) -> list[State]:
        p0, p1 = s.values
        if not p1:
            return [State((p0, True))]
        return []

    def p1_exit(s: State) -> list[State]:
        p0, p1 = s.values
        if p1:
            return [State((p0, False))]
        return []

    mc.add_action("P0_enter", p0_enter)
    mc.add_action("P0_exit", p0_exit)
    mc.add_action("P1_enter", p1_enter)
    mc.add_action("P1_exit", p1_exit)

    mc.add_invariant("MutualExclusion",
                     lambda s: not (s.values[0] and s.values[1]))

    result = mc.check()
    print(f"\n  States explored: {result['states_explored']}")
    print(f"  Invariant violations: {len(result['violations'])}")

    if result["violations"]:
        name, state, trace = result["violations"][0]
        print(f"\n  BUG FOUND! Violation of {name}")
        print(f"  State: {state} (both processes in critical section!)")
        print(f"  Trace: {' -> '.join(trace)}")


# ---------------------------------------------------------------------------
# Example 2: Simple Consensus
# ---------------------------------------------------------------------------

def verify_consensus() -> None:
    """Model check a simple consensus protocol."""
    print("\n" + "=" * 70)
    print("Model Check: Simple Consensus (2 Processes)")
    print("=" * 70)

    # State: (p0_proposal, p1_proposal, p0_decided, p1_decided)
    mc = ModelChecker("Consensus")

    # Both processes start with different proposals
    mc.add_initial(State((0, 1, None, None)))

    def p0_decides_own(s: State) -> list[State]:
        pp0, pp1, d0, d1 = s.values
        if d0 is None:
            return [State((pp0, pp1, pp0, d1))]
        return []

    def p0_decides_other(s: State) -> list[State]:
        pp0, pp1, d0, d1 = s.values
        if d0 is None:
            return [State((pp0, pp1, pp1, d1))]
        return []

    def p1_decides_own(s: State) -> list[State]:
        pp0, pp1, d0, d1 = s.values
        if d1 is None:
            return [State((pp0, pp1, d0, pp1))]
        return []

    def p1_decides_other(s: State) -> list[State]:
        pp0, pp1, d0, d1 = s.values
        if d1 is None:
            return [State((pp0, pp1, d0, pp0))]
        return []

    mc.add_action("P0_own", p0_decides_own)
    mc.add_action("P0_other", p0_decides_other)
    mc.add_action("P1_own", p1_decides_own)
    mc.add_action("P1_other", p1_decides_other)

    # Safety: Agreement — if both decided, they decided the same value
    def agreement(s: State) -> bool:
        _, _, d0, d1 = s.values
        if d0 is not None and d1 is not None:
            return d0 == d1
        return True

    # Safety: Validity — decision must be one of the original proposals
    def validity(s: State) -> bool:
        pp0, pp1, d0, d1 = s.values
        proposals = {pp0, pp1}
        if d0 is not None and d0 not in proposals:
            return False
        if d1 is not None and d1 not in proposals:
            return False
        return True

    mc.add_invariant("Agreement", agreement)
    mc.add_invariant("Validity", validity)

    result = mc.check()
    print(f"\n  States explored: {result['states_explored']}")
    print(f"  Agreement violations: ", end="")

    agreement_violations = [v for v in result["violations"] if v[0] == "Agreement"]
    print(f"{len(agreement_violations)}")

    if agreement_violations:
        name, state, trace = agreement_violations[0]
        print(f"\n  BUG: Both processes can decide independently => DISAGREEMENT")
        print(f"  State: {state}")
        print(f"  Trace: {' -> '.join(trace)}")
        print(f"\n  This is why consensus requires coordination (Paxos/Raft)!")


# ---------------------------------------------------------------------------
# Example 3: TLA+ comparison
# ---------------------------------------------------------------------------

def demo_tlaplus_comparison() -> None:
    """Show equivalent TLA+ specification."""
    print("\n" + "=" * 70)
    print("Equivalent TLA+ Specification (for reference)")
    print("=" * 70)

    print("""
  The mutex protocol above corresponds to this TLA+ spec:

  ---- MODULE Mutex ----
  VARIABLES turn, p0_in_cs, p1_in_cs

  Init == /\\ turn \\in {0, 1}
          /\\ p0_in_cs = FALSE
          /\\ p1_in_cs = FALSE

  P0_Enter == /\\ turn = 0
              /\\ p0_in_cs = FALSE
              /\\ p0_in_cs' = TRUE
              /\\ UNCHANGED <<turn, p1_in_cs>>

  P0_Exit == /\\ p0_in_cs = TRUE
             /\\ p0_in_cs' = FALSE
             /\\ turn' = 1
             /\\ UNCHANGED p1_in_cs

  P1_Enter == /\\ turn = 1
              /\\ p1_in_cs = FALSE
              /\\ p1_in_cs' = TRUE
              /\\ UNCHANGED <<turn, p0_in_cs>>

  P1_Exit == /\\ p1_in_cs = TRUE
             /\\ p1_in_cs' = FALSE
             /\\ turn' = 0
             /\\ UNCHANGED p0_in_cs

  Next == P0_Enter \\/ P0_Exit \\/ P1_Enter \\/ P1_Exit

  MutualExclusion == ~(p0_in_cs /\\ p1_in_cs)

  Spec == Init /\\ [][Next]_<<turn, p0_in_cs, p1_in_cs>>
  ====

  TLC (TLA+ model checker) would verify MutualExclusion as an invariant
  and report the same results as our Python model checker.

  Real-world TLA+ use:
  - Amazon: DynamoDB, S3, EBS protocol verification
  - Microsoft: Cosmos DB consistency protocols
  - MongoDB: replication protocol
  - CockroachDB: Raft implementation verification
""")


if __name__ == "__main__":
    verify_mutex_correct()
    verify_mutex_buggy()
    verify_consensus()
    demo_tlaplus_comparison()
    print("Done.")
