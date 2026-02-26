"""
DFA (Deterministic Finite Automaton) Simulator
================================================

Demonstrates:
- Formal DFA definition as a 5-tuple (Q, Σ, δ, q0, F)
- String acceptance via extended transition function
- DFA product construction for intersection/union
- DFA complement
- DFA minimization via Hopcroft's algorithm

Reference: Formal_Languages Lesson 2 — Deterministic Finite Automata
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Set, Tuple


# Why: A dataclass captures the formal 5-tuple (Q, Σ, δ, q0, F) directly,
# making the mathematical definition executable and easy to inspect.
@dataclass
class DFA:
    """Deterministic Finite Automaton."""
    states: Set[str]
    alphabet: Set[str]
    transitions: Dict[Tuple[str, str], str]  # (state, symbol) -> state
    start: str
    accept: Set[str]

    def process(self, input_str: str) -> Tuple[bool, list[str]]:
        """Process input string, return (accepted, state_trace)."""
        current = self.start
        trace = [current]
        for symbol in input_str:
            if symbol not in self.alphabet:
                raise ValueError(f"Symbol '{symbol}' not in alphabet {self.alphabet}")
            key = (current, symbol)
            if key not in self.transitions:
                # Why: A missing transition acts as an implicit dead/trap state.
                # This avoids bloating the transition table with explicit sink states.
                return False, trace
            current = self.transitions[key]
            trace.append(current)
        return current in self.accept, trace

    def accepts(self, input_str: str) -> bool:
        """Check if DFA accepts the input string."""
        return self.process(input_str)[0]

    def complement(self) -> DFA:
        """Return DFA for the complement language."""
        # Why: Complementing a DFA is trivial — just swap accept/non-accept.
        # This only works for DFAs (not NFAs) because every input has exactly
        # one computation path, so flipping acceptance is sound.
        return DFA(
            states=self.states.copy(),
            alphabet=self.alphabet.copy(),
            transitions=self.transitions.copy(),
            start=self.start,
            accept=self.states - self.accept,
        )

    def __repr__(self) -> str:
        return (f"DFA(states={self.states}, alphabet={self.alphabet}, "
                f"start='{self.start}', accept={self.accept})")


# Why: Product construction runs both DFAs in parallel on the same input.
# A state in the product is a pair (q1, q2). Acceptance depends on the
# mode: intersection requires both to accept, union requires at least one.
def product_construction(
    dfa1: DFA, dfa2: DFA, mode: str = "intersection"
) -> DFA:
    """
    Build the product DFA for intersection or union of two DFAs.

    Args:
        dfa1, dfa2: Input DFAs (must share the same alphabet)
        mode: 'intersection' or 'union'
    """
    assert dfa1.alphabet == dfa2.alphabet, "Alphabets must match"
    alphabet = dfa1.alphabet

    states: Set[str] = set()
    transitions: Dict[Tuple[str, str], str] = {}
    start = f"({dfa1.start},{dfa2.start})"
    accept: Set[str] = set()

    # Why: BFS from the start pair ensures we only generate reachable states,
    # avoiding the full |Q1|×|Q2| Cartesian product when many pairs are unreachable.
    queue = [(dfa1.start, dfa2.start)]
    visited: Set[Tuple[str, str]] = {(dfa1.start, dfa2.start)}

    while queue:
        q1, q2 = queue.pop(0)
        name = f"({q1},{q2})"
        states.add(name)

        # Determine acceptance
        in_f1 = q1 in dfa1.accept
        in_f2 = q2 in dfa2.accept
        if mode == "intersection" and in_f1 and in_f2:
            accept.add(name)
        elif mode == "union" and (in_f1 or in_f2):
            accept.add(name)

        for a in alphabet:
            next1 = dfa1.transitions.get((q1, a))
            next2 = dfa2.transitions.get((q2, a))
            if next1 is not None and next2 is not None:
                next_name = f"({next1},{next2})"
                transitions[(name, a)] = next_name
                if (next1, next2) not in visited:
                    visited.add((next1, next2))
                    queue.append((next1, next2))

    return DFA(states=states, alphabet=alphabet, transitions=transitions,
               start=start, accept=accept)


# Why: Hopcroft's algorithm merges states that are "equivalent" (indistinguishable
# by any input string). It starts with {accept, non-accept} and refines until
# no partition can be split further — guaranteed to produce the unique minimal DFA.
def minimize_dfa(dfa: DFA) -> DFA:
    """
    Minimize a DFA using Hopcroft's partition refinement algorithm.

    Returns a new DFA with the minimum number of states.
    """
    # Step 1: Remove unreachable states
    reachable: Set[str] = set()
    queue = [dfa.start]
    reachable.add(dfa.start)
    while queue:
        state = queue.pop(0)
        for symbol in dfa.alphabet:
            next_state = dfa.transitions.get((state, symbol))
            if next_state and next_state not in reachable:
                reachable.add(next_state)
                queue.append(next_state)

    # Step 2: Partition refinement
    accept_reachable = dfa.accept & reachable
    non_accept_reachable = reachable - dfa.accept

    if not accept_reachable or not non_accept_reachable:
        # All states are accept or all non-accept
        partitions = [reachable]
    else:
        partitions = [accept_reachable, non_accept_reachable]

    def find_partition(state: str) -> int:
        for i, p in enumerate(partitions):
            if state in p:
                return i
        return -1

    changed = True
    while changed:
        changed = False
        new_partitions = []
        for partition in partitions:
            # Try to split this partition
            splits: Dict[tuple, Set[str]] = {}
            for state in partition:
                # Why: The "signature" captures where each symbol leads (by partition index).
                # Two states with different signatures are distinguishable and must be split.
                signature = tuple(
                    find_partition(dfa.transitions.get((state, a), ""))
                    for a in sorted(dfa.alphabet)
                )
                splits.setdefault(signature, set()).add(state)

            if len(splits) > 1:
                changed = True
            new_partitions.extend(splits.values())
        partitions = new_partitions

    # Step 3: Build minimized DFA
    state_to_partition = {}
    for i, p in enumerate(partitions):
        for state in p:
            state_to_partition[state] = f"P{i}"

    min_states = {f"P{i}" for i in range(len(partitions))}
    min_transitions = {}
    min_accept = set()
    min_start = state_to_partition[dfa.start]

    for i, partition in enumerate(partitions):
        representative = next(iter(partition))
        p_name = f"P{i}"

        if representative in dfa.accept:
            min_accept.add(p_name)

        for symbol in dfa.alphabet:
            next_state = dfa.transitions.get((representative, symbol))
            if next_state and next_state in state_to_partition:
                min_transitions[(p_name, symbol)] = state_to_partition[next_state]

    return DFA(states=min_states, alphabet=dfa.alphabet,
               transitions=min_transitions, start=min_start, accept=min_accept)


# ─────────────── Demo ───────────────

def demo_even_ones():
    """DFA accepting strings with an even number of 1s."""
    print("=" * 60)
    print("Demo 1: Even number of 1s")
    print("=" * 60)

    dfa = DFA(
        states={"q_even", "q_odd"},
        alphabet={"0", "1"},
        transitions={
            ("q_even", "0"): "q_even", ("q_even", "1"): "q_odd",
            ("q_odd", "0"): "q_odd",   ("q_odd", "1"): "q_even",
        },
        start="q_even",
        accept={"q_even"},
    )

    tests = ["", "0", "1", "11", "101", "1001", "111", "1111"]
    for s in tests:
        accepted, trace = dfa.process(s)
        display = s if s else "ε"
        print(f"  '{display}': {' → '.join(trace)} → {'ACCEPT' if accepted else 'REJECT'}")


def demo_divisible_by_3():
    """DFA accepting binary numbers divisible by 3."""
    print("\n" + "=" * 60)
    print("Demo 2: Binary numbers divisible by 3")
    print("=" * 60)

    dfa = DFA(
        states={"r0", "r1", "r2"},
        alphabet={"0", "1"},
        transitions={
            ("r0", "0"): "r0", ("r0", "1"): "r1",
            ("r1", "0"): "r2", ("r1", "1"): "r0",
            ("r2", "0"): "r1", ("r2", "1"): "r2",
        },
        start="r0",
        accept={"r0"},
    )

    for n in range(16):
        binary = bin(n)[2:] if n > 0 else "0"
        accepted = dfa.accepts(binary)
        div3 = "✓" if n % 3 == 0 else " "
        dfa_result = "✓" if accepted else " "
        match = "OK" if (n % 3 == 0) == accepted else "MISMATCH"
        print(f"  {n:3d} = {binary:>5s}  div3={div3}  DFA={dfa_result}  {match}")


def demo_product_construction():
    """Product DFA for intersection: even 0s AND odd 1s."""
    print("\n" + "=" * 60)
    print("Demo 3: Product Construction (even 0s ∩ odd 1s)")
    print("=" * 60)

    # DFA 1: even number of 0s
    dfa_even_0s = DFA(
        states={"e0", "o0"},
        alphabet={"0", "1"},
        transitions={
            ("e0", "0"): "o0", ("e0", "1"): "e0",
            ("o0", "0"): "e0", ("o0", "1"): "o0",
        },
        start="e0",
        accept={"e0"},
    )

    # DFA 2: odd number of 1s
    dfa_odd_1s = DFA(
        states={"e1", "o1"},
        alphabet={"0", "1"},
        transitions={
            ("e1", "0"): "e1", ("e1", "1"): "o1",
            ("o1", "0"): "o1", ("o1", "1"): "e1",
        },
        start="e1",
        accept={"o1"},
    )

    product = product_construction(dfa_even_0s, dfa_odd_1s, "intersection")
    print(f"  Product DFA states: {product.states}")
    print(f"  Accept states: {product.accept}")
    print(f"  Number of states: {len(product.states)}")

    tests = ["1", "01", "001", "010", "0011", "00111"]
    for s in tests:
        count_0 = s.count("0")
        count_1 = s.count("1")
        accepted = product.accepts(s)
        expected = (count_0 % 2 == 0) and (count_1 % 2 == 1)
        status = "OK" if accepted == expected else "MISMATCH"
        print(f"  '{s}': 0s={count_0}(even={'Y' if count_0%2==0 else 'N'}) "
              f"1s={count_1}(odd={'Y' if count_1%2==1 else 'N'}) "
              f"→ {'ACCEPT' if accepted else 'REJECT'} {status}")


def demo_minimization():
    """DFA minimization: reducing redundant states."""
    print("\n" + "=" * 60)
    print("Demo 4: DFA Minimization")
    print("=" * 60)

    # Redundant DFA for strings ending in '1' (has unnecessary states)
    dfa = DFA(
        states={"A", "B", "C", "D"},
        alphabet={"0", "1"},
        transitions={
            ("A", "0"): "B", ("A", "1"): "C",
            ("B", "0"): "B", ("B", "1"): "D",
            ("C", "0"): "B", ("C", "1"): "C",
            ("D", "0"): "B", ("D", "1"): "D",
        },
        start="A",
        accept={"C", "D"},
    )

    print(f"  Original: {len(dfa.states)} states: {dfa.states}")

    minimized = minimize_dfa(dfa)
    print(f"  Minimized: {len(minimized.states)} states: {minimized.states}")

    # Verify both accept the same strings
    tests = ["", "0", "1", "00", "01", "10", "11", "010", "101"]
    all_match = True
    for s in tests:
        if dfa.accepts(s) != minimized.accepts(s):
            all_match = False
            print(f"  MISMATCH on '{s}'!")
    print(f"  Equivalence verified on {len(tests)} test strings: {'PASS' if all_match else 'FAIL'}")


def demo_complement():
    """DFA complement: swapping accept/reject."""
    print("\n" + "=" * 60)
    print("Demo 5: DFA Complement")
    print("=" * 60)

    # DFA accepting strings containing "01"
    dfa = DFA(
        states={"start", "saw0", "saw01"},
        alphabet={"0", "1"},
        transitions={
            ("start", "0"): "saw0", ("start", "1"): "start",
            ("saw0", "0"): "saw0",  ("saw0", "1"): "saw01",
            ("saw01", "0"): "saw01", ("saw01", "1"): "saw01",
        },
        start="start",
        accept={"saw01"},
    )

    comp = dfa.complement()
    print(f"  Original accepts: strings containing '01'")
    print(f"  Complement accepts: strings NOT containing '01'")

    tests = ["", "0", "1", "01", "10", "11", "001", "110", "111", "010"]
    for s in tests:
        orig = dfa.accepts(s)
        compl = comp.accepts(s)
        display = s if s else "ε"
        assert orig != compl, f"Complement error on '{s}'"
        print(f"  '{display}': original={'ACCEPT' if orig else 'REJECT'}, "
              f"complement={'ACCEPT' if compl else 'REJECT'}")


if __name__ == "__main__":
    demo_even_ones()
    demo_divisible_by_3()
    demo_product_construction()
    demo_minimization()
    demo_complement()
