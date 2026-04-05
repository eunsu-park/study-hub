"""
NFA Simulator and Subset Construction (NFA → DFA Conversion)
==============================================================

Demonstrates:
- NFA definition with epsilon-transitions
- Epsilon-closure computation
- NFA simulation (parallel state tracking)
- Subset construction algorithm (NFA → DFA)
- Exponential blowup example

Reference: Formal_Languages Lesson 3 — Nondeterministic Finite Automata
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Set, Tuple


# Why: Using a named constant instead of an empty string prevents confusion
# with "no input" and makes epsilon-transitions explicit in transition keys.
EPSILON = "ε"


@dataclass
class NFA:
    """Nondeterministic Finite Automaton with epsilon-transitions."""
    states: Set[str]
    alphabet: Set[str]
    transitions: Dict[Tuple[str, str], Set[str]]  # (state, symbol|ε) -> {states}
    start: str
    accept: Set[str]

    def epsilon_closure(self, states: Set[str]) -> Set[str]:
        """Compute epsilon-closure of a set of states."""
        # Why: Epsilon-closure finds all states reachable via zero or more
        # epsilon-transitions. This is the foundation of NFA simulation —
        # without it, we'd miss states the NFA can "silently" transition to.
        closure = set(states)
        stack = list(states)
        while stack:
            state = stack.pop()
            for next_state in self.transitions.get((state, EPSILON), set()):
                if next_state not in closure:
                    closure.add(next_state)
                    stack.append(next_state)
        return closure

    def move(self, states: Set[str], symbol: str) -> Set[str]:
        """Compute the set of states reachable from 'states' on 'symbol'."""
        result: Set[str] = set()
        for state in states:
            result |= self.transitions.get((state, symbol), set())
        return result

    def accepts(self, input_str: str) -> bool:
        """Simulate NFA on input string."""
        # Why: We track a SET of current states (not a single state) because
        # the NFA can be in multiple states simultaneously. This is equivalent
        # to on-the-fly subset construction without building the full DFA.
        current = self.epsilon_closure({self.start})
        for symbol in input_str:
            current = self.epsilon_closure(self.move(current, symbol))
        return bool(current & self.accept)

    def simulate_trace(self, input_str: str) -> list[Set[str]]:
        """Return trace of state sets during NFA simulation."""
        current = self.epsilon_closure({self.start})
        trace = [current]
        for symbol in input_str:
            current = self.epsilon_closure(self.move(current, symbol))
            trace.append(current)
        return trace


@dataclass
class DFA:
    """DFA from subset construction (states are frozensets)."""
    states: Set[FrozenSet[str]]
    alphabet: Set[str]
    transitions: Dict[Tuple[FrozenSet[str], str], FrozenSet[str]]
    start: FrozenSet[str]
    accept: Set[FrozenSet[str]]

    def accepts(self, input_str: str) -> bool:
        current = self.start
        for symbol in input_str:
            key = (current, symbol)
            if key not in self.transitions:
                return False
            current = self.transitions[key]
        return current in self.accept


# Why: Subset construction proves NFA == DFA in expressive power. Each DFA state
# is a frozenset of NFA states. Worst case is 2^n DFA states for n NFA states,
# but in practice most state-sets are unreachable, keeping the DFA manageable.
def subset_construction(nfa: NFA) -> DFA:
    """
    Convert NFA to equivalent DFA using subset construction.

    Each DFA state corresponds to a set of NFA states.
    """
    alphabet = nfa.alphabet - {EPSILON}

    start = frozenset(nfa.epsilon_closure({nfa.start}))
    dfa_states: Set[FrozenSet[str]] = set()
    dfa_transitions: Dict[Tuple[FrozenSet[str], str], FrozenSet[str]] = {}
    dfa_accept: Set[FrozenSet[str]] = set()

    queue = [start]
    dfa_states.add(start)

    while queue:
        current = queue.pop(0)

        if current & nfa.accept:
            dfa_accept.add(current)

        for symbol in alphabet:
            next_states = frozenset(
                nfa.epsilon_closure(nfa.move(set(current), symbol))
            )
            dfa_transitions[(current, symbol)] = next_states

            if next_states not in dfa_states:
                dfa_states.add(next_states)
                queue.append(next_states)

    return DFA(
        states=dfa_states,
        alphabet=alphabet,
        transitions=dfa_transitions,
        start=start,
        accept=dfa_accept,
    )


def format_state_set(s: frozenset | set) -> str:
    """Format a set of states for display."""
    if not s:
        return "∅"
    return "{" + ", ".join(sorted(s)) + "}"


# ─────────────── Demos ───────────────

def demo_nfa_ends_01():
    """NFA for strings ending in '01'."""
    print("=" * 60)
    print("Demo 1: NFA for strings ending in '01'")
    print("=" * 60)

    nfa = NFA(
        states={"q0", "q1", "q2"},
        alphabet={"0", "1"},
        transitions={
            ("q0", "0"): {"q0", "q1"},
            ("q0", "1"): {"q0"},
            ("q1", "1"): {"q2"},
        },
        start="q0",
        accept={"q2"},
    )

    tests = ["01", "001", "101", "0101", "10", "11", "00", ""]
    for s in tests:
        trace = nfa.simulate_trace(s)
        accepted = bool(trace[-1] & nfa.accept)
        display = s if s else "ε"
        trace_str = " → ".join(format_state_set(t) for t in trace)
        print(f"  '{display}': {trace_str} → {'ACCEPT' if accepted else 'REJECT'}")


def demo_subset_construction():
    """Convert NFA (ends in '01') to DFA via subset construction."""
    print("\n" + "=" * 60)
    print("Demo 2: Subset Construction (NFA → DFA)")
    print("=" * 60)

    nfa = NFA(
        states={"q0", "q1", "q2"},
        alphabet={"0", "1"},
        transitions={
            ("q0", "0"): {"q0", "q1"},
            ("q0", "1"): {"q0"},
            ("q1", "1"): {"q2"},
        },
        start="q0",
        accept={"q2"},
    )

    dfa = subset_construction(nfa)

    print(f"  NFA states: {len(nfa.states)}")
    print(f"  DFA states: {len(dfa.states)}")
    print(f"  DFA start: {format_state_set(dfa.start)}")
    print(f"  DFA accept: {[format_state_set(s) for s in dfa.accept]}")
    print()

    # Transition table
    print("  Transition table:")
    print(f"  {'State':<20} {'On 0':<20} {'On 1':<20}")
    print("  " + "-" * 60)
    for state in sorted(dfa.states, key=lambda s: sorted(s)):
        on_0 = format_state_set(dfa.transitions.get((state, "0"), frozenset()))
        on_1 = format_state_set(dfa.transitions.get((state, "1"), frozenset()))
        marker = " *" if state in dfa.accept else ""
        print(f"  {format_state_set(state):<20} {on_0:<20} {on_1:<20}{marker}")

    # Verify equivalence
    print("\n  Equivalence verification:")
    tests = ["01", "001", "101", "0101", "10", "11", "00", "", "0", "1"]
    for s in tests:
        nfa_result = nfa.accepts(s)
        dfa_result = dfa.accepts(s)
        display = s if s else "ε"
        match = "OK" if nfa_result == dfa_result else "MISMATCH"
        print(f"    '{display}': NFA={'ACCEPT' if nfa_result else 'REJECT'}, "
              f"DFA={'ACCEPT' if dfa_result else 'REJECT'} {match}")


def demo_epsilon_transitions():
    """NFA with epsilon-transitions: L = {a*} ∪ {b*}."""
    print("\n" + "=" * 60)
    print("Demo 3: NFA with ε-transitions (a* ∪ b*)")
    print("=" * 60)

    nfa = NFA(
        states={"q0", "q1", "q2"},
        alphabet={"a", "b", EPSILON},
        transitions={
            ("q0", EPSILON): {"q1", "q2"},
            ("q1", "a"): {"q1"},
            ("q2", "b"): {"q2"},
        },
        start="q0",
        accept={"q1", "q2"},
    )

    e_closure = nfa.epsilon_closure({"q0"})
    print(f"  ε-closure(q0) = {format_state_set(e_closure)}")

    tests = ["", "a", "aa", "b", "bb", "ab", "ba", "aaa", "bbb"]
    for s in tests:
        accepted = nfa.accepts(s)
        display = s if s else "ε"
        print(f"  '{display}': {'ACCEPT' if accepted else 'REJECT'}")

    dfa = subset_construction(nfa)
    print(f"\n  Converted DFA has {len(dfa.states)} states")


def demo_exponential_blowup():
    """
    Exponential blowup: NFA for 'n-th from last symbol is 1'.

    For n=3: an NFA with 4 states requires a DFA with up to 8 states.
    """
    print("\n" + "=" * 60)
    print("Demo 4: Exponential Blowup (3rd-from-last is '1')")
    print("=" * 60)

    n = 3
    # Why: This NFA nondeterministically "guesses" when the n-th-from-last symbol
    # arrives and starts counting. With n+2 NFA states, the equivalent DFA needs
    # up to 2^(n+1) states — demonstrating the exponential blowup is real.
    states = {f"q{i}" for i in range(n + 1)} | {"q_start"}
    nfa = NFA(
        states=states,
        alphabet={"0", "1"},
        transitions={
            ("q_start", "0"): {"q_start"},
            ("q_start", "1"): {"q_start", "q0"},
            ("q0", "0"): {"q1"},
            ("q0", "1"): {"q1"},
            ("q1", "0"): {"q2"},
            ("q1", "1"): {"q2"},
        },
        start="q_start",
        accept={f"q{n-1}"},
    )

    print(f"  NFA states: {len(nfa.states)} (for n={n})")

    dfa = subset_construction(nfa)
    print(f"  DFA states: {len(dfa.states)} (up to 2^{n+1}={2**(n+1)} possible)")

    tests = ["100", "1000", "0100", "1100", "111", "000", "010", "1010"]
    for s in tests:
        nfa_r = nfa.accepts(s)
        dfa_r = dfa.accepts(s)
        # Check: 3rd from last is '1'
        expected = len(s) >= n and s[-n] == "1"
        print(f"  '{s}': expected={expected}, NFA={nfa_r}, DFA={dfa_r}, "
              f"{'OK' if nfa_r == dfa_r == expected else 'ERROR'}")


def demo_nfa_closure_constructions():
    """Closure constructions: union (ε-transitions) and concatenation."""
    print("\n" + "=" * 60)
    print("Demo 5: NFA Closure Constructions")
    print("=" * 60)

    # L1 = {a^n | n >= 1}, L2 = {b^n | n >= 1}
    # Concatenation L1·L2 = {a^i b^j | i,j >= 1}
    nfa_concat = NFA(
        states={"s1", "a1", "bridge", "s2", "b1"},
        alphabet={"a", "b", EPSILON},
        transitions={
            ("s1", "a"): {"a1"},
            ("a1", "a"): {"a1"},
            ("a1", EPSILON): {"s2"},
            ("s2", "b"): {"b1"},
            ("b1", "b"): {"b1"},
        },
        start="s1",
        accept={"b1"},
    )

    print("  L1·L2 = {a^i b^j | i,j >= 1} (concatenation)")
    tests = ["ab", "aab", "abb", "aabb", "a", "b", "", "ba", "aba"]
    for s in tests:
        accepted = nfa_concat.accepts(s)
        display = s if s else "ε"
        print(f"    '{display}': {'ACCEPT' if accepted else 'REJECT'}")

    # Kleene star: L = {ab}*
    print("\n  {ab}* (Kleene star)")
    nfa_star = NFA(
        states={"q_new", "q0", "q1", "q_acc"},
        alphabet={"a", "b", EPSILON},
        transitions={
            ("q_new", EPSILON): {"q0", "q_acc"},  # accept ε; start matching
            ("q0", "a"): {"q1"},
            ("q1", "b"): {"q_acc"},
            ("q_acc", EPSILON): {"q0"},  # loop back for star
        },
        start="q_new",
        accept={"q_acc"},
    )

    tests = ["", "ab", "abab", "ababab", "a", "b", "ba", "aba"]
    for s in tests:
        accepted = nfa_star.accepts(s)
        display = s if s else "ε"
        print(f"    '{display}': {'ACCEPT' if accepted else 'REJECT'}")


if __name__ == "__main__":
    demo_nfa_ends_01()
    demo_subset_construction()
    demo_epsilon_transitions()
    demo_exponential_blowup()
    demo_nfa_closure_constructions()
