"""
Exercises for Lesson 02: Deterministic Finite Automata (DFA)
Topic: Formal_Languages

Solutions to practice problems from the lesson.
"""


class DFA:
    """A Deterministic Finite Automaton."""

    def __init__(self, states, alphabet, transitions, start_state, accept_states):
        """
        Initialize a DFA.

        Args:
            states: set of state names
            alphabet: set of input symbols
            transitions: dict mapping (state, symbol) -> next_state
            start_state: the initial state
            accept_states: set of accepting states
        """
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states

    def process(self, input_string):
        """Process an input string, returning (accepted, final_state, trace)."""
        current = self.start_state
        trace = [current]
        for symbol in input_string:
            if symbol not in self.alphabet:
                raise ValueError(f"Symbol '{symbol}' not in alphabet {self.alphabet}")
            current = self.transitions[(current, symbol)]
            trace.append(current)
        accepted = current in self.accept_states
        return accepted, current, trace

    def accepts(self, input_string):
        """Return True if the DFA accepts the input string."""
        accepted, _, _ = self.process(input_string)
        return accepted


def product_dfa(dfa1, dfa2, mode="intersection"):
    """
    Construct the product DFA of two DFAs.

    Args:
        dfa1, dfa2: DFA instances (must share the same alphabet)
        mode: 'intersection' or 'union'

    Returns:
        A new DFA recognizing L(dfa1) intersect L(dfa2) or L(dfa1) union L(dfa2).
    """
    assert dfa1.alphabet == dfa2.alphabet, "Alphabets must match"
    alphabet = dfa1.alphabet

    # States are pairs (q1, q2)
    states = set()
    transitions = {}
    start = (dfa1.start_state, dfa2.start_state)

    # Build transitions via BFS from start state
    queue = [start]
    visited = {start}
    while queue:
        (q1, q2) = queue.pop(0)
        states.add((q1, q2))
        for a in alphabet:
            next_q1 = dfa1.transitions[(q1, a)]
            next_q2 = dfa2.transitions[(q2, a)]
            next_state = (next_q1, next_q2)
            transitions[((q1, q2), a)] = next_state
            if next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)

    # Determine accept states
    if mode == "intersection":
        accept_states = {
            (q1, q2) for (q1, q2) in states
            if q1 in dfa1.accept_states and q2 in dfa2.accept_states
        }
    else:  # union
        accept_states = {
            (q1, q2) for (q1, q2) in states
            if q1 in dfa1.accept_states or q2 in dfa2.accept_states
        }

    return DFA(states, alphabet, transitions, start, accept_states)


# === Exercise 1: DFA Construction ===
# Problem: Design DFAs for each language over Sigma = {0, 1}:
# 1. L = {w | w contains the substring 110}
# 2. L = {w | |w| is divisible by 3}
# 3. L = {w | w does not contain two consecutive 1s}

def exercise_1():
    """DFA Construction for three languages."""
    # --- Part 1: L = {w | w contains the substring 110} ---
    # States track progress toward seeing "110":
    #   q0: haven't started matching
    #   q1: just saw "1"
    #   q2: just saw "11"
    #   q3: saw "110" (accepting trap state)
    dfa_110 = DFA(
        states={"q0", "q1", "q2", "q3"},
        alphabet={"0", "1"},
        transitions={
            ("q0", "0"): "q0", ("q0", "1"): "q1",
            ("q1", "0"): "q0", ("q1", "1"): "q2",
            ("q2", "0"): "q3", ("q2", "1"): "q2",
            ("q3", "0"): "q3", ("q3", "1"): "q3",
        },
        start_state="q0",
        accept_states={"q3"},
    )

    print("Part 1: L = {w | w contains substring '110'}")
    print("  States: {q0, q1, q2, q3}")
    print("  Start: q0, Accept: {q3}")
    test_cases_1 = ["", "0", "1", "110", "0110", "1101", "010", "111", "11001"]
    for w in test_cases_1:
        accepted, _, trace = dfa_110.process(w)
        label = "epsilon" if w == "" else w
        print(f"  '{label}': {' -> '.join(trace)} => {'ACCEPT' if accepted else 'reject'}")

    # --- Part 2: L = {w | |w| is divisible by 3} ---
    # States: r0, r1, r2 tracking |w| mod 3
    dfa_div3_len = DFA(
        states={"r0", "r1", "r2"},
        alphabet={"0", "1"},
        transitions={
            ("r0", "0"): "r1", ("r0", "1"): "r1",
            ("r1", "0"): "r2", ("r1", "1"): "r2",
            ("r2", "0"): "r0", ("r2", "1"): "r0",
        },
        start_state="r0",
        accept_states={"r0"},
    )

    print("\nPart 2: L = {w | |w| is divisible by 3}")
    print("  States: {r0, r1, r2}")
    print("  Start: r0, Accept: {r0}")
    test_cases_2 = ["", "0", "01", "010", "0101", "01010", "010101"]
    for w in test_cases_2:
        accepted, _, trace = dfa_div3_len.process(w)
        label = "epsilon" if w == "" else w
        print(f"  '{label}' (len={len(w)}): {' -> '.join(trace)} => {'ACCEPT' if accepted else 'reject'}")

    # --- Part 3: L = {w | w does not contain two consecutive 1s} ---
    # States:
    #   q0: start / last symbol was 0 (or nothing yet)
    #   q1: last symbol was 1
    #   dead: saw "11" -- reject trap
    dfa_no_11 = DFA(
        states={"q0", "q1", "dead"},
        alphabet={"0", "1"},
        transitions={
            ("q0", "0"): "q0", ("q0", "1"): "q1",
            ("q1", "0"): "q0", ("q1", "1"): "dead",
            ("dead", "0"): "dead", ("dead", "1"): "dead",
        },
        start_state="q0",
        accept_states={"q0", "q1"},
    )

    print("\nPart 3: L = {w | w does not contain two consecutive 1s}")
    print("  States: {q0, q1, dead}")
    print("  Start: q0, Accept: {q0, q1}")
    test_cases_3 = ["", "0", "1", "01", "10", "010", "101", "110", "0110", "10101"]
    for w in test_cases_3:
        accepted, _, trace = dfa_no_11.process(w)
        label = "epsilon" if w == "" else w
        print(f"  '{label}': {' -> '.join(trace)} => {'ACCEPT' if accepted else 'reject'}")


# === Exercise 2: Product Construction ===
# Problem: Given:
#   M1 accepts strings with an even number of 0s
#   M2 accepts strings with an odd number of 1s
# Construct the product DFA for L(M1) intersect L(M2).
# How many states does it have?

def exercise_2():
    """Product construction: even number of 0s AND odd number of 1s."""
    # M1: accepts strings with even number of 0s
    dfa_even_0s = DFA(
        states={"e0", "o0"},
        alphabet={"0", "1"},
        transitions={
            ("e0", "0"): "o0", ("e0", "1"): "e0",
            ("o0", "0"): "e0", ("o0", "1"): "o0",
        },
        start_state="e0",
        accept_states={"e0"},
    )

    # M2: accepts strings with odd number of 1s
    dfa_odd_1s = DFA(
        states={"e1", "o1"},
        alphabet={"0", "1"},
        transitions={
            ("e1", "0"): "e1", ("e1", "1"): "o1",
            ("o1", "0"): "o1", ("o1", "1"): "e1",
        },
        start_state="e1",
        accept_states={"o1"},
    )

    # Product DFA for intersection
    product = product_dfa(dfa_even_0s, dfa_odd_1s, mode="intersection")

    print("M1: even number of 0s (states: {e0, o0}, accept: {e0})")
    print("M2: odd number of 1s  (states: {e1, o1}, accept: {o1})")
    print(f"\nProduct DFA for L(M1) ∩ L(M2):")
    print(f"  States: {sorted(product.states)}")
    print(f"  Start: {product.start_state}")
    print(f"  Accept: {sorted(product.accept_states)}")
    print(f"  Number of states: {len(product.states)}")

    print("\n  Transitions:")
    for (state, symbol), next_state in sorted(product.transitions.items()):
        print(f"    delta({state}, {symbol}) = {next_state}")

    print("\n  Test strings:")
    test_cases = ["", "1", "0", "01", "10", "001", "1001", "0110", "010"]
    for w in test_cases:
        accepted = product.accepts(w)
        count_0 = w.count("0")
        count_1 = w.count("1")
        label = "epsilon" if w == "" else w
        print(
            f"    '{label}': #0s={count_0}({'even' if count_0 % 2 == 0 else 'odd'}), "
            f"#1s={count_1}({'even' if count_1 % 2 == 0 else 'odd'}) => "
            f"{'ACCEPT' if accepted else 'reject'}"
        )


# === Exercise 3: Correctness Proof ===
# Problem: Prove by induction that the "divisible by 3" DFA correctly
# recognizes binary numbers divisible by 3.
# (Implemented as a verification rather than a formal proof)

def exercise_3():
    """Verify the 'divisible by 3' DFA and demonstrate the inductive argument."""
    # DFA for binary numbers divisible by 3
    dfa_div3 = DFA(
        states={"r0", "r1", "r2"},
        alphabet={"0", "1"},
        transitions={
            ("r0", "0"): "r0", ("r0", "1"): "r1",
            ("r1", "0"): "r2", ("r1", "1"): "r0",
            ("r2", "0"): "r1", ("r2", "1"): "r2",
        },
        start_state="r0",
        accept_states={"r0"},
    )

    print("Correctness verification for 'divisible by 3' DFA")
    print("=" * 55)
    print()
    print("Claim: After reading binary string w, the DFA is in state r_i")
    print("       iff the numeric value of w is congruent to i (mod 3).")
    print()
    print("Proof sketch (by induction on |w|):")
    print("  Base: |w| = 0 (epsilon). Value = 0. State = r0. 0 mod 3 = 0. Correct.")
    print("  Inductive step: Suppose after reading w, DFA is in r_i where")
    print("    value(w) ≡ i (mod 3). Reading bit b gives value 2*value(w) + b.")
    print("    New remainder = (2i + b) mod 3.")
    print("    This matches the transition function delta(r_i, b).")
    print()

    # Verify the inductive step explicitly
    print("  Verification of transition function vs (2i + b) mod 3:")
    state_map = {"r0": 0, "r1": 1, "r2": 2}
    reverse_map = {0: "r0", 1: "r1", 2: "r2"}

    for state in ["r0", "r1", "r2"]:
        i = state_map[state]
        for bit in ["0", "1"]:
            b = int(bit)
            expected_remainder = (2 * i + b) % 3
            expected_state = reverse_map[expected_remainder]
            actual_state = dfa_div3.transitions[(state, bit)]
            match = "OK" if actual_state == expected_state else "MISMATCH"
            print(
                f"    delta({state}, {bit}): (2*{i}+{b}) mod 3 = {expected_remainder} "
                f"=> {expected_state}, actual: {actual_state} [{match}]"
            )

    # Exhaustive test for numbers 0-31
    print("\n  Exhaustive test for binary numbers 0-31:")
    all_correct = True
    for n in range(32):
        binary = bin(n)[2:] if n > 0 else "0"
        # Special case: empty string represents 0
        accepted = dfa_div3.accepts(binary)
        is_div3 = (n % 3 == 0)
        correct = accepted == is_div3
        if not correct:
            all_correct = False
        _, final_state, _ = dfa_div3.process(binary)
        print(
            f"    {n:2d} = '{binary:>5s}' => state {final_state}, "
            f"n mod 3 = {n % 3}, "
            f"{'ACCEPT' if accepted else 'reject'} "
            f"{'OK' if correct else 'WRONG'}"
        )

    print(f"\n  All correct: {all_correct}")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: DFA Construction ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Product Construction ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Correctness Proof ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
