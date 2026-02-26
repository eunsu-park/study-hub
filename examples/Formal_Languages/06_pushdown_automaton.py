"""
Pushdown Automaton (PDA) Simulator
====================================

Demonstrates:
- PDA definition with stack operations
- Nondeterministic PDA simulation via BFS
- Acceptance by final state and by empty stack
- PDA design patterns for common CFLs

Reference: Formal_Languages Lesson 7 — Pushdown Automata
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


EPSILON = "ε"


@dataclass(frozen=True)
class PDATransition:
    """A PDA transition: (next_state, push_symbols)."""
    next_state: str
    push: str  # symbols to push (leftmost = top), "" for pop/no push


@dataclass
class PDA:
    """
    Nondeterministic Pushdown Automaton.

    Transitions: (state, input_symbol|ε, stack_top|ε) → [(next_state, push)]
    """
    states: Set[str]
    input_alphabet: Set[str]
    stack_alphabet: Set[str]
    transitions: Dict[Tuple[str, str, str], List[PDATransition]]
    start: str
    start_stack: str  # initial stack symbol
    accept_states: Set[str]
    accept_mode: str = "final_state"  # "final_state" or "empty_stack"

    def accepts(self, input_str: str, max_steps: int = 1000) -> bool:
        """
        Simulate PDA on input using BFS over configurations.

        A configuration is (state, remaining_input, stack).
        """
        # Why: PDAs are nondeterministic, so we use BFS to explore all possible
        # computation paths in parallel. A string is accepted if ANY path reaches
        # an accepting configuration — we cannot just follow one path.
        initial_config = (self.start, input_str, self.start_stack)
        queue = [initial_config]
        visited: Set[Tuple[str, str, str]] = set()
        steps = 0

        while queue and steps < max_steps:
            state, remaining, stack = queue.pop(0)
            steps += 1

            # Why: The stack can grow unboundedly, so we truncate to 50 chars for
            # the visited check. This trades correctness for termination — a
            # pragmatic choice since exact PDA simulation is undecidable in general.
            stack_key = stack[:50]
            config_key = (state, remaining, stack_key)
            if config_key in visited:
                continue
            visited.add(config_key)

            # Check acceptance
            if not remaining:
                if self.accept_mode == "final_state" and state in self.accept_states:
                    return True
                if self.accept_mode == "empty_stack" and not stack:
                    return True

            # Generate successor configurations
            stack_top = stack[0] if stack else ""

            # Why: Each step tries both reading a symbol and epsilon (no read),
            # and both popping the stack top and not popping. This covers all
            # four combinations of input/stack consumption per transition.
            for input_sym in ([remaining[0], EPSILON] if remaining else [EPSILON]):
                for stk_sym in ([stack_top, EPSILON] if stack_top else [EPSILON]):
                    key = (state, input_sym, stk_sym)
                    for trans in self.transitions.get(key, []):
                        # Compute new stack
                        new_stack = stack
                        if stk_sym and stk_sym != EPSILON:
                            new_stack = new_stack[1:]  # pop
                        if trans.push:
                            new_stack = trans.push + new_stack  # push

                        # Compute new remaining input
                        new_remaining = remaining
                        if input_sym != EPSILON:
                            new_remaining = remaining[1:]

                        queue.append((trans.next_state, new_remaining, new_stack))

        return False

    def trace(self, input_str: str, max_steps: int = 200) -> List[Tuple[str, str, str]]:
        """Return one accepting trace (BFS), or empty list if rejected."""
        initial = (self.start, input_str, self.start_stack, [(self.start, input_str, self.start_stack)])
        queue = [initial]
        visited: Set[Tuple[str, str, str]] = set()
        steps = 0

        while queue and steps < max_steps:
            state, remaining, stack, path = queue.pop(0)
            steps += 1

            stack_key = stack[:50]
            config_key = (state, remaining, stack_key)
            if config_key in visited:
                continue
            visited.add(config_key)

            if not remaining:
                if self.accept_mode == "final_state" and state in self.accept_states:
                    return path
                if self.accept_mode == "empty_stack" and not stack:
                    return path

            stack_top = stack[0] if stack else ""
            for input_sym in ([remaining[0], EPSILON] if remaining else [EPSILON]):
                for stk_sym in ([stack_top, EPSILON] if stack_top else [EPSILON]):
                    key = (state, input_sym, stk_sym)
                    for trans in self.transitions.get(key, []):
                        new_stack = stack
                        if stk_sym and stk_sym != EPSILON:
                            new_stack = new_stack[1:]
                        if trans.push:
                            new_stack = trans.push + new_stack
                        new_remaining = remaining if input_sym == EPSILON else remaining[1:]
                        new_path = path + [(trans.next_state, new_remaining, new_stack)]
                        queue.append((trans.next_state, new_remaining, new_stack, new_path))

        return []


def format_config(state: str, remaining: str, stack: str) -> str:
    """Format a PDA configuration for display."""
    r = remaining if remaining else "ε"
    s = stack if stack else "ε"
    return f"({state}, {r}, {s})"


# ─────────────── Demos ───────────────

def demo_anbn():
    """PDA for {a^n b^n | n >= 0}."""
    print("=" * 60)
    print("Demo 1: PDA for {a^n b^n | n >= 0}")
    print("=" * 60)

    pda = PDA(
        states={"q0", "q1", "q2"},
        input_alphabet={"a", "b"},
        stack_alphabet={"Z", "X"},
        transitions={
            ("q0", "a", "Z"): [PDATransition("q0", "XZ")],
            ("q0", "a", "X"): [PDATransition("q0", "XX")],
            ("q0", "b", "X"): [PDATransition("q1", "")],
            ("q1", "b", "X"): [PDATransition("q1", "")],
            ("q1", EPSILON, "Z"): [PDATransition("q2", "")],
            ("q0", EPSILON, "Z"): [PDATransition("q2", "")],  # accept ε
        },
        start="q0",
        start_stack="Z",
        accept_states={"q2"},
    )

    tests = ["", "ab", "aabb", "aaabbb", "a", "b", "aab", "abb"]
    for s in tests:
        accepted = pda.accepts(s)
        display = s if s else "ε"
        expected = len(s) % 2 == 0 and s[:len(s)//2] == 'a' * (len(s)//2) and s[len(s)//2:] == 'b' * (len(s)//2)
        status = "OK" if accepted == expected else "ERROR"
        print(f"  '{display}': {'ACCEPT' if accepted else 'REJECT'} {status}")

    # Show trace for "aabb"
    trace = pda.trace("aabb")
    if trace:
        print(f"\n  Trace for 'aabb':")
        for state, rem, stack in trace:
            print(f"    {format_config(state, rem, stack)}")


def demo_palindromes():
    """PDA for even-length palindromes {ww^R | w ∈ {a,b}*}."""
    print("\n" + "=" * 60)
    print("Demo 2: PDA for {ww^R | w ∈ {a,b}*} (nondeterministic)")
    print("=" * 60)

    pda = PDA(
        states={"push", "pop", "accept"},
        input_alphabet={"a", "b"},
        stack_alphabet={"Z", "a", "b"},
        transitions={
            # Push phase
            ("push", "a", "Z"): [PDATransition("push", "aZ")],
            ("push", "b", "Z"): [PDATransition("push", "bZ")],
            ("push", "a", "a"): [PDATransition("push", "aa")],
            ("push", "a", "b"): [PDATransition("push", "ab")],
            ("push", "b", "a"): [PDATransition("push", "ba")],
            ("push", "b", "b"): [PDATransition("push", "bb")],
            # Nondeterministic switch to pop phase
            ("push", EPSILON, "a"): [PDATransition("pop", "a")],
            ("push", EPSILON, "b"): [PDATransition("pop", "b")],
            ("push", EPSILON, "Z"): [PDATransition("accept", "")],
            # Pop phase: match input with stack
            ("pop", "a", "a"): [PDATransition("pop", "")],
            ("pop", "b", "b"): [PDATransition("pop", "")],
            # Accept when stack has only Z
            ("pop", EPSILON, "Z"): [PDATransition("accept", "")],
        },
        start="push",
        start_stack="Z",
        accept_states={"accept"},
    )

    tests = ["", "aa", "bb", "abba", "abba", "baab", "aabbaa", "ab", "aba", "abc"]
    for s in tests:
        accepted = pda.accepts(s)
        display = s if s else "ε"
        is_palindrome = s == s[::-1] and len(s) % 2 == 0
        print(f"  '{display}': {'ACCEPT' if accepted else 'REJECT'} "
              f"(even palindrome: {is_palindrome})")


def demo_balanced_parens():
    """PDA for balanced parentheses."""
    print("\n" + "=" * 60)
    print("Demo 3: PDA for Balanced Parentheses")
    print("=" * 60)

    pda = PDA(
        states={"q0", "q_acc"},
        input_alphabet={"(", ")"},
        stack_alphabet={"Z", "X"},
        transitions={
            ("q0", "(", "Z"): [PDATransition("q0", "XZ")],
            ("q0", "(", "X"): [PDATransition("q0", "XX")],
            ("q0", ")", "X"): [PDATransition("q0", "")],
            ("q0", EPSILON, "Z"): [PDATransition("q_acc", "")],
        },
        start="q0",
        start_stack="Z",
        accept_states={"q_acc"},
    )

    tests = ["", "()", "(())", "((()))", "()()", "(()())", "(", ")", "(()", "())", ")("]
    for s in tests:
        accepted = pda.accepts(s)
        display = s if s else "ε"

        # Verify
        count = 0
        expected = True
        for c in s:
            count += 1 if c == '(' else -1
            if count < 0:
                expected = False
                break
        expected = expected and count == 0

        status = "OK" if accepted == expected else "ERROR"
        print(f"  '{display}': {'ACCEPT' if accepted else 'REJECT'} {status}")


def demo_equal_ab():
    """PDA for {w ∈ {a,b}* | |w|_a = |w|_b}."""
    print("\n" + "=" * 60)
    print("Demo 4: PDA for equal a's and b's")
    print("=" * 60)

    # Strategy: Use stack to track difference between count(a) and count(b)
    pda = PDA(
        states={"q0", "q_acc"},
        input_alphabet={"a", "b"},
        stack_alphabet={"Z", "A", "B"},
        transitions={
            # Reading 'a' when stack shows excess b's: cancel one
            ("q0", "a", "B"): [PDATransition("q0", "")],
            # Reading 'a' when stack shows excess a's: add more
            ("q0", "a", "A"): [PDATransition("q0", "AA")],
            # Reading 'a' when stack is neutral (Z on top)
            ("q0", "a", "Z"): [PDATransition("q0", "AZ")],
            # Reading 'b' when stack shows excess a's: cancel one
            ("q0", "b", "A"): [PDATransition("q0", "")],
            # Reading 'b' when stack shows excess b's: add more
            ("q0", "b", "B"): [PDATransition("q0", "BB")],
            # Reading 'b' when stack is neutral
            ("q0", "b", "Z"): [PDATransition("q0", "BZ")],
            # Accept when done and stack has only Z
            ("q0", EPSILON, "Z"): [PDATransition("q_acc", "")],
        },
        start="q0",
        start_stack="Z",
        accept_states={"q_acc"},
    )

    tests = ["", "ab", "ba", "aabb", "abab", "abba", "aab", "bba", "ababab", "aabbb"]
    for s in tests:
        accepted = pda.accepts(s)
        display = s if s else "ε"
        expected = s.count('a') == s.count('b')
        status = "OK" if accepted == expected else "ERROR"
        print(f"  '{display}': a={s.count('a')}, b={s.count('b')} → "
              f"{'ACCEPT' if accepted else 'REJECT'} {status}")


def demo_dpda_vs_npda():
    """Show the difference between DPDA and NPDA capabilities."""
    print("\n" + "=" * 60)
    print("Demo 5: DPDA vs NPDA")
    print("=" * 60)

    print("  Deterministic CFL: {wcw^R | w ∈ {a,b}*}")
    print("  The center marker 'c' makes it deterministic.\n")

    # DPDA for wcw^R (deterministic: push until c, then pop)
    pda = PDA(
        states={"push", "pop", "accept"},
        input_alphabet={"a", "b", "c"},
        stack_alphabet={"Z", "a", "b"},
        transitions={
            # Push phase (before c)
            ("push", "a", "Z"): [PDATransition("push", "aZ")],
            ("push", "b", "Z"): [PDATransition("push", "bZ")],
            ("push", "a", "a"): [PDATransition("push", "aa")],
            ("push", "a", "b"): [PDATransition("push", "ab")],
            ("push", "b", "a"): [PDATransition("push", "ba")],
            ("push", "b", "b"): [PDATransition("push", "bb")],
            # See 'c': switch to pop (deterministic!)
            ("push", "c", "a"): [PDATransition("pop", "a")],
            ("push", "c", "b"): [PDATransition("pop", "b")],
            ("push", "c", "Z"): [PDATransition("accept", "")],
            # Pop phase: match
            ("pop", "a", "a"): [PDATransition("pop", "")],
            ("pop", "b", "b"): [PDATransition("pop", "")],
            ("pop", EPSILON, "Z"): [PDATransition("accept", "")],
        },
        start="push",
        start_stack="Z",
        accept_states={"accept"},
    )

    tests = ["c", "aca", "bcb", "abcba", "aabcbaa", "abcab", "abc", ""]
    for s in tests:
        accepted = pda.accepts(s)
        # Verify: s = wcw^R?
        if 'c' in s:
            idx = s.index('c')
            w = s[:idx]
            rest = s[idx+1:]
            expected = w == rest[::-1]
        else:
            expected = False
        status = "OK" if accepted == expected else "ERROR"
        print(f"  '{s}': {'ACCEPT' if accepted else 'REJECT'} {status}")

    # Why: This demo illustrates a key theoretical result — DPDAs recognize a
    # proper subset of CFLs. The center marker 'c' eliminates the need to guess,
    # proving that nondeterminism genuinely adds power for pushdown automata.
    print("\n  Key insight: {ww^R} needs nondeterminism (guess midpoint)")
    print("  But {wcw^R} is deterministic (c marks the midpoint)")
    print("  So DPDA ⊊ NPDA (proper subset)")


if __name__ == "__main__":
    demo_anbn()
    demo_palindromes()
    demo_balanced_parens()
    demo_equal_ab()
    demo_dpda_vs_npda()
