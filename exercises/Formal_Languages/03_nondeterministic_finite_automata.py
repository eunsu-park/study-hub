"""
Exercises for Lesson 03: Nondeterministic Finite Automata (NFA)
Topic: Formal_Languages

Solutions to practice problems from the lesson.
"""


class NFA:
    """A Nondeterministic Finite Automaton (with epsilon-transitions)."""

    EPSILON = ""  # Represents epsilon-transition

    def __init__(self, states, alphabet, transitions, start_state, accept_states):
        """
        Initialize an NFA.

        Args:
            states: set of state names
            alphabet: set of input symbols (not including epsilon)
            transitions: dict mapping (state, symbol_or_epsilon) -> set of states
            start_state: the initial state
            accept_states: set of accepting states
        """
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states

    def epsilon_closure(self, states_set):
        """Compute the epsilon-closure of a set of states."""
        closure = set(states_set)
        stack = list(states_set)
        while stack:
            state = stack.pop()
            eps_targets = self.transitions.get((state, self.EPSILON), set())
            for target in eps_targets:
                if target not in closure:
                    closure.add(target)
                    stack.append(target)
        return frozenset(closure)

    def move(self, states_set, symbol):
        """Compute the set of states reachable from states_set on symbol."""
        result = set()
        for state in states_set:
            targets = self.transitions.get((state, symbol), set())
            result |= targets
        return result

    def accepts(self, input_string):
        """Return True if the NFA accepts the input string."""
        current_states = self.epsilon_closure({self.start_state})
        for symbol in input_string:
            next_states = self.move(current_states, symbol)
            current_states = self.epsilon_closure(next_states)
        return bool(current_states & self.accept_states)

    def to_dfa(self):
        """Convert NFA to DFA using subset construction."""
        dfa_start = self.epsilon_closure({self.start_state})
        dfa_states = {dfa_start}
        dfa_transitions = {}
        queue = [dfa_start]
        visited = {dfa_start}

        while queue:
            current = queue.pop(0)
            for symbol in self.alphabet:
                next_states = self.epsilon_closure(self.move(current, symbol))
                dfa_transitions[(current, symbol)] = next_states
                if next_states not in visited:
                    visited.add(next_states)
                    dfa_states.add(next_states)
                    queue.append(next_states)

        dfa_accept = {
            s for s in dfa_states if s & self.accept_states
        }

        return dfa_states, dfa_transitions, dfa_start, dfa_accept


# === Exercise 1: NFA Design ===
# Problem: Design NFAs for:
# 1. L = {w in {a,b}* | w contains "aab" as a substring}
# 2. L = {w in {0,1}* | the third symbol from the end is 0}
# 3. L = {w in {a,b}* | |w| ≡ 0 (mod 2) or w ends with "ab"}

def exercise_1():
    """NFA Design for three languages."""

    # --- Part 1: L = {w | w contains "aab" as a substring} ---
    # The NFA guesses when "aab" starts.
    nfa_aab = NFA(
        states={"q0", "q1", "q2", "q3"},
        alphabet={"a", "b"},
        transitions={
            ("q0", "a"): {"q0", "q1"},  # Stay in q0 or guess "aab" starts
            ("q0", "b"): {"q0"},
            ("q1", "a"): {"q2"},        # Saw first 'a', need second 'a'
            ("q2", "b"): {"q3"},        # Saw "aa", need 'b'
            ("q3", "a"): {"q3"},        # Trap: already found "aab"
            ("q3", "b"): {"q3"},
        },
        start_state="q0",
        accept_states={"q3"},
    )

    print("Part 1: L = {w | w contains 'aab' as a substring}")
    print("  NFA: states={q0,q1,q2,q3}, start=q0, accept={q3}")
    print("  delta(q0, a) = {q0, q1}  -- nondeterministically guess start of 'aab'")
    print("  delta(q0, b) = {q0}")
    print("  delta(q1, a) = {q2}")
    print("  delta(q2, b) = {q3}")
    print("  delta(q3, a) = {q3}, delta(q3, b) = {q3}  -- accepting trap")
    test_cases_1 = ["", "a", "ab", "aab", "baab", "aabb", "bbb", "aaab", "aba"]
    for w in test_cases_1:
        result = nfa_aab.accepts(w)
        label = "epsilon" if w == "" else w
        print(f"    '{label}': {'ACCEPT' if result else 'reject'}")

    # --- Part 2: L = {w in {0,1}* | the third symbol from the end is 0} ---
    # NFA guesses "we are 3 symbols from the end" and checks that symbol is 0.
    nfa_3rd_from_end = NFA(
        states={"q0", "q1", "q2", "q3"},
        alphabet={"0", "1"},
        transitions={
            ("q0", "0"): {"q0", "q1"},  # Guess: this 0 is 3rd from end
            ("q0", "1"): {"q0"},
            ("q1", "0"): {"q2"},        # 2nd from end
            ("q1", "1"): {"q2"},
            ("q2", "0"): {"q3"},        # Last symbol
            ("q2", "1"): {"q3"},
        },
        start_state="q0",
        accept_states={"q3"},
    )

    print("\nPart 2: L = {w | third-from-last symbol is 0}")
    print("  NFA: 4 states, guesses position of 0 that is 3rd from end")
    test_cases_2 = [
        "000", "001", "010", "011",   # 3rd from end is 0 -> all accept
        "100", "101", "110", "111",   # 3rd from end is 1 -> all reject
        "1000", "0111", "01",         # Various lengths
    ]
    for w in test_cases_2:
        result = nfa_3rd_from_end.accepts(w)
        third_from_end = w[-3] if len(w) >= 3 else "N/A"
        print(f"    '{w}': 3rd-from-end='{third_from_end}' => {'ACCEPT' if result else 'reject'}")

    # --- Part 3: L = {w | |w| even OR w ends with "ab"} ---
    # Use epsilon-transitions to nondeterministically choose which condition to check.
    nfa_even_or_ab = NFA(
        states={"q0", "e0", "e1", "a0", "a1", "a2"},
        alphabet={"a", "b"},
        transitions={
            # Start: branch to either check
            ("q0", ""): {"e0", "a0"},
            # Branch 1: even length
            ("e0", "a"): {"e1"}, ("e0", "b"): {"e1"},
            ("e1", "a"): {"e0"}, ("e1", "b"): {"e0"},
            # Branch 2: ends with "ab"
            ("a0", "a"): {"a0", "a1"}, ("a0", "b"): {"a0"},
            ("a1", "b"): {"a2"},
        },
        start_state="q0",
        accept_states={"e0", "a2"},
    )

    print("\nPart 3: L = {w | |w| is even OR w ends with 'ab'}")
    print("  NFA uses epsilon-transition from start to branch into two checks")
    test_cases_3 = [
        ("", True),         # even length (0)
        ("a", False),       # odd, doesn't end with ab
        ("ab", True),       # even AND ends with ab
        ("aa", True),       # even length
        ("b", False),       # odd, doesn't end with ab
        ("aab", True),      # odd but ends with ab
        ("bab", True),      # odd but ends with ab
        ("ba", True),       # even length
        ("aba", False),     # odd, doesn't end with ab
        ("abab", True),     # even AND ends with ab
    ]
    for w, expected in test_cases_3:
        result = nfa_even_or_ab.accepts(w)
        label = "epsilon" if w == "" else w
        status = "OK" if result == expected else "MISMATCH"
        print(f"    '{label}': {'ACCEPT' if result else 'reject'} [{status}]")


# === Exercise 2: Subset Construction ===
# Problem: Perform the subset construction on this NFA:
#   States: {q0, q1, q2}, Start: q0, Accept: {q2}
#   delta(q0, a) = {q0, q1}   delta(q0, b) = {q0}
#   delta(q1, a) = empty       delta(q1, b) = {q2}
#   delta(q2, a) = empty       delta(q2, b) = empty

def exercise_2():
    """Subset construction on a specific NFA."""
    nfa = NFA(
        states={"q0", "q1", "q2"},
        alphabet={"a", "b"},
        transitions={
            ("q0", "a"): {"q0", "q1"},
            ("q0", "b"): {"q0"},
            ("q1", "b"): {"q2"},
            # All other transitions implicitly go to empty set
        },
        start_state="q0",
        accept_states={"q2"},
    )

    print("NFA:")
    print("  States: {q0, q1, q2}, Start: q0, Accept: {q2}")
    print("  delta(q0, a) = {q0, q1}   delta(q0, b) = {q0}")
    print("  delta(q1, a) = {}          delta(q1, b) = {q2}")
    print("  delta(q2, a) = {}          delta(q2, b) = {}")

    dfa_states, dfa_transitions, dfa_start, dfa_accept = nfa.to_dfa()

    def fmt_state(fs):
        """Format a frozenset state nicely."""
        if not fs:
            return "{}"
        return "{" + ", ".join(sorted(fs)) + "}"

    print("\nSubset Construction Result (DFA):")
    print(f"  Start state: {fmt_state(dfa_start)}")
    print(f"  Number of reachable states: {len(dfa_states)}")
    print(f"  Accept states: {[fmt_state(s) for s in sorted(dfa_accept, key=lambda x: sorted(x))]}")

    print("\n  Transition table:")
    print(f"    {'DFA State':<20} {'On a':<20} {'On b':<20}")
    print(f"    {'-'*20} {'-'*20} {'-'*20}")
    for state in sorted(dfa_states, key=lambda x: (len(x), sorted(x))):
        on_a = dfa_transitions.get((state, "a"), frozenset())
        on_b = dfa_transitions.get((state, "b"), frozenset())
        accept_mark = " *" if state in dfa_accept else ""
        print(f"    {fmt_state(state):<20} {fmt_state(on_a):<20} {fmt_state(on_b):<20}{accept_mark}")

    # Verify NFA and DFA agree
    print("\n  Verification (NFA vs DFA agree):")
    test_strings = ["", "a", "b", "ab", "aa", "aab", "aba", "aabb", "abab", "bab"]
    for w in test_strings:
        nfa_result = nfa.accepts(w)
        # Simulate DFA manually
        current = dfa_start
        for symbol in w:
            current = dfa_transitions.get((current, symbol), frozenset())
        dfa_result = current in dfa_accept
        match = "OK" if nfa_result == dfa_result else "MISMATCH"
        label = "epsilon" if w == "" else w
        print(f"    '{label}': NFA={'ACCEPT' if nfa_result else 'reject'}, "
              f"DFA={'ACCEPT' if dfa_result else 'reject'} [{match}]")


# === Exercise 3: Concatenation Construction ===
# Problem: Given NFAs for L1 = {a^n | n >= 1} and L2 = {b^n | n >= 1},
# construct an NFA for L1 . L2 using the concatenation construction.

def exercise_3():
    """Concatenation construction for L1 = {a^n | n>=1} and L2 = {b^n | n>=1}."""

    # NFA for L1 = {a^n | n >= 1}: at least one 'a'
    # States: p0 (start) -> p1 (accept, loop on a)
    print("NFA for L1 = {a^n | n >= 1}:")
    print("  States: {p0, p1}, Start: p0, Accept: {p1}")
    print("  delta(p0, a) = {p1}")
    print("  delta(p1, a) = {p1}")

    # NFA for L2 = {b^n | n >= 1}: at least one 'b'
    # States: r0 (start) -> r1 (accept, loop on b)
    print("\nNFA for L2 = {b^n | n >= 1}:")
    print("  States: {r0, r1}, Start: r0, Accept: {r1}")
    print("  delta(r0, b) = {r1}")
    print("  delta(r1, b) = {r1}")

    # Concatenation construction:
    # 1. Add epsilon-transition from accept states of N1 (p1) to start of N2 (r0)
    # 2. Make p1 non-accepting
    # 3. Start = p0, Accept = {r1}
    nfa_concat = NFA(
        states={"p0", "p1", "r0", "r1"},
        alphabet={"a", "b"},
        transitions={
            ("p0", "a"): {"p1"},
            ("p1", "a"): {"p1"},
            ("p1", ""): {"r0"},      # Epsilon-transition: N1 accept -> N2 start
            ("r0", "b"): {"r1"},
            ("r1", "b"): {"r1"},
        },
        start_state="p0",
        accept_states={"r1"},  # Only N2's accept states
    )

    print("\nConcatenation NFA for L1 . L2:")
    print("  States: {p0, p1, r0, r1}")
    print("  Start: p0, Accept: {r1}")
    print("  delta(p0, a) = {p1}")
    print("  delta(p1, a) = {p1}")
    print("  delta(p1, epsilon) = {r0}  <-- concatenation link")
    print("  delta(r0, b) = {r1}")
    print("  delta(r1, b) = {r1}")
    print("  Note: p1 is NO LONGER an accept state")

    print("\n  Language: L1.L2 = {a^n b^m | n >= 1, m >= 1}")
    test_cases = [
        ("", False),
        ("a", False),
        ("b", False),
        ("ab", True),
        ("aab", True),
        ("abb", True),
        ("aabb", True),
        ("aaabbb", True),
        ("ba", False),
        ("aba", False),
    ]
    for w, expected in test_cases:
        result = nfa_concat.accepts(w)
        label = "epsilon" if w == "" else w
        status = "OK" if result == expected else "MISMATCH"
        print(f"    '{label}': {'ACCEPT' if result else 'reject'} (expected: {'ACCEPT' if expected else 'reject'}) [{status}]")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: NFA Design ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Subset Construction ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Concatenation Construction ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
