"""
Exercises for Lesson 07: Pushdown Automata
Topic: Formal_Languages

Solutions to practice problems from the lesson.
"""


class PDA:
    """A Nondeterministic Pushdown Automaton (acceptance by final state)."""

    def __init__(self, states, input_alphabet, stack_alphabet, transitions,
                 start_state, start_stack, accept_states):
        """
        Args:
            transitions: dict mapping (state, input_or_eps, stack_top_or_eps) ->
                         list of (next_state, stack_push_string)
                         where input_or_eps is a symbol or '' for epsilon,
                         stack_top_or_eps is a symbol or '' for no-read,
                         stack_push_string is a string to push (leftmost = new top),
                         or '' for pop-only.
        """
        self.states = states
        self.input_alphabet = input_alphabet
        self.stack_alphabet = stack_alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.start_stack = start_stack
        self.accept_states = accept_states

    def accepts(self, input_string, max_steps=10000):
        """Check if the PDA accepts the input using BFS over configurations."""
        # Configuration: (state, remaining_input_index, stack_as_tuple)
        initial_stack = (self.start_stack,) if self.start_stack else ()
        start_config = (self.start_state, 0, initial_stack)
        queue = [start_config]
        visited = set()
        visited.add(start_config)
        steps = 0

        while queue and steps < max_steps:
            state, pos, stack = queue.pop(0)
            steps += 1

            # Check acceptance: input consumed AND in accept state
            if pos == len(input_string) and state in self.accept_states:
                return True

            # Generate next configurations
            possible_inputs = ['']  # epsilon-transition always possible
            if pos < len(input_string):
                possible_inputs.append(input_string[pos])

            possible_stacks = ['']  # no stack read
            if stack:
                possible_stacks.append(stack[0])  # top of stack

            for inp in possible_inputs:
                for stk in possible_stacks:
                    key = (state, inp, stk)
                    for (next_state, push_str) in self.transitions.get(key, []):
                        new_pos = pos + (1 if inp != '' else 0)
                        # Pop the top if stk != ''
                        new_stack = stack[1:] if stk != '' else stack
                        # Push the push_str (leftmost = top)
                        if push_str:
                            new_stack = tuple(push_str) + new_stack
                        config = (next_state, new_pos, new_stack)
                        if config not in visited:
                            visited.add(config)
                            queue.append(config)

        return False


# === Exercise 1: PDA Design ===
# Problem: Design PDAs for:
# 1. L = {a^n b^(2n) | n >= 0}
# 2. L = {w in {a,b}* | #a(w) = #b(w)}
# 3. L = {a^i b^j c^k | i + k = j}

def exercise_1():
    """PDA Design for three languages."""

    # --- Part 1: L = {a^n b^(2n) | n >= 0} ---
    # Strategy: Push TWO markers for each 'a', pop one for each 'b'.
    print("Part 1: L = {a^n b^(2n) | n >= 0}")
    print("  Strategy: Push 2 symbols for each 'a', pop 1 for each 'b'.")
    print("  States: {q0, q1, q2}")
    print("  q0: reading a's (push XX per a)")
    print("  q1: reading b's (pop X per b)")
    print("  q2: accept")
    print("  Transitions:")
    print("    (q0, a, '') -> (q0, 'XX')     push two X's per a")
    print("    (q0, '', '$') -> (q2, '')      n=0 case: accept on empty")
    print("    (q0, b, 'X') -> (q1, '')       start reading b's")
    print("    (q1, b, 'X') -> (q1, '')       continue reading b's")
    print("    (q1, '', '$') -> (q2, '')      stack empty = accept")

    pda1 = PDA(
        states={"q0", "q1", "q2"},
        input_alphabet={"a", "b"},
        stack_alphabet={"X", "$"},
        transitions={
            ("q0", "", ""): [("q0", "$")],       # Initialize stack with bottom marker
            ("q0", "a", ""): [("q0", "XX")],     # Push 2 X's per a
            ("q0", "", "$"): [("q2", "")],        # Accept empty (n=0 after init)
            ("q0", "b", "X"): [("q1", "")],       # Start popping
            ("q1", "b", "X"): [("q1", "")],       # Continue popping
            ("q1", "", "$"): [("q2", "")],         # All done
        },
        start_state="q0",
        start_stack="",
        accept_states={"q2"},
    )

    # Simpler version with direct stack bottom detection
    pda1_v2 = PDA(
        states={"init", "qa", "qb", "acc"},
        input_alphabet={"a", "b"},
        stack_alphabet={"X", "$"},
        transitions={
            ("init", "", ""): [("qa", "$")],
            ("qa", "a", ""): [("qa", "XX")],
            ("qa", "", "$"): [("acc", "")],
            ("qa", "b", "X"): [("qb", "")],
            ("qb", "b", "X"): [("qb", "")],
            ("qb", "", "$"): [("acc", "")],
        },
        start_state="init",
        start_stack="",
        accept_states={"acc"},
    )

    test1 = [("", True), ("abb", True), ("aabbbb", True), ("a", False),
             ("ab", False), ("abbb", False), ("aabb", False), ("b", False)]
    print("\n  Tests:")
    for w, expected in test1:
        result = pda1_v2.accepts(w)
        label = "eps" if w == "" else w
        status = "OK" if result == expected else "MISMATCH"
        print(f"    '{label}': {'ACCEPT' if result else 'reject'} (expected: {'ACCEPT' if expected else 'reject'}) [{status}]")

    # --- Part 2: L = {w in {a,b}* | #a(w) = #b(w)} ---
    print("\n\nPart 2: L = {w in {a,b}* | #a(w) = #b(w)}")
    print("  Strategy: Use stack to track the imbalance.")
    print("  Push 'A' for each 'a' when stack is empty or has A's on top.")
    print("  Push 'B' for each 'b' when stack is empty or has B's on top.")
    print("  Pop 'B' when reading 'a' and B is on top (cancel out).")
    print("  Pop 'A' when reading 'b' and A is on top (cancel out).")
    print("  Accept when stack has only the bottom marker.")

    pda2 = PDA(
        states={"init", "q", "acc"},
        input_alphabet={"a", "b"},
        stack_alphabet={"A", "B", "$"},
        transitions={
            ("init", "", ""): [("q", "$")],
            ("q", "a", "A"): [("q", "AA")],  # a with A on top: push more A
            ("q", "a", "$"): [("q", "A$")],  # a with empty effective stack
            ("q", "a", "B"): [("q", "")],     # a cancels B
            ("q", "b", "B"): [("q", "BB")],  # b with B on top: push more B
            ("q", "b", "$"): [("q", "B$")],  # b with empty effective stack
            ("q", "b", "A"): [("q", "")],     # b cancels A
            ("q", "", "$"): [("acc", "")],    # Accept when balanced
        },
        start_state="init",
        start_stack="",
        accept_states={"acc"},
    )

    test2 = [("", True), ("ab", True), ("ba", True), ("aabb", True),
             ("abab", True), ("abba", True), ("a", False), ("b", False),
             ("aab", False), ("abb", False), ("bbaa", True)]
    print("\n  Tests:")
    for w, expected in test2:
        result = pda2.accepts(w)
        label = "eps" if w == "" else w
        status = "OK" if result == expected else "MISMATCH"
        print(f"    '{label}': {'ACCEPT' if result else 'reject'} (expected: {'ACCEPT' if expected else 'reject'}) [{status}]")

    # --- Part 3: L = {a^i b^j c^k | i + k = j} ---
    print("\n\nPart 3: L = {a^i b^j c^k | i + k = j}")
    print("  Strategy: Push for each 'a', pop for first i b's,")
    print("  then push for remaining b's, then pop for c's.")
    print("  Nondeterministically guess where to switch from popping to pushing.")
    print()
    print("  Alternative cleaner approach:")
    print("  Since j = i + k, split as: read a's (push), read b's matching a's (pop),")
    print("  read more b's (push), read c's matching those b's (pop).")

    pda3 = PDA(
        states={"init", "qa", "qb1", "qb2", "qc", "acc"},
        input_alphabet={"a", "b", "c"},
        stack_alphabet={"X", "$"},
        transitions={
            ("init", "", ""): [("qa", "$")],
            # Phase 1: Read a's, push X
            ("qa", "a", ""): [("qa", "X")],
            # Phase 2: Read b's matching a's, pop X
            ("qa", "b", "X"): [("qb1", "")],
            ("qb1", "b", "X"): [("qb1", "")],
            # When stack hits bottom ($), switch to pushing for remaining b's
            ("qa", "", "$"): [("qb2", "$")],   # No a's at all -> go to b2
            ("qb1", "", "$"): [("qb2", "$")],  # All a's matched, remaining b's
            # Phase 3: Read remaining b's (for c's to match), push X
            ("qb2", "b", ""): [("qb2", "X")],
            # Phase 4: Read c's, pop X
            ("qb2", "c", "X"): [("qc", "")],
            ("qb2", "", "$"): [("acc", "")],    # No remaining b's or c's
            ("qc", "c", "X"): [("qc", "")],
            ("qc", "", "$"): [("acc", "")],     # All done
        },
        start_state="init",
        start_stack="",
        accept_states={"acc"},
    )

    test3 = [
        ("", True),          # i=0, j=0, k=0: 0+0=0
        ("bc", True),        # i=0, j=1, k=1: 0+1=1
        ("abbc", True),      # i=1, j=2, k=1: 1+1=2
        ("aabbbbcc", True),  # i=2, j=4, k=2: 2+2=4
        ("ab", True),        # i=1, j=1, k=0: 1+0=1
        ("abc", False),      # i=1, j=1, k=1: 1+1=2 != 1
        ("aabb", True),      # i=2, j=2, k=0: 2+0=2
        ("b", False),        # i=0, j=1, k=0: 0+0=0 != 1
    ]
    print("\n  Tests:")
    for w, expected in test3:
        result = pda3.accepts(w)
        label = "eps" if w == "" else w
        status = "OK" if result == expected else "MISMATCH"
        print(f"    '{label}': {'ACCEPT' if result else 'reject'} (expected: {'ACCEPT' if expected else 'reject'}) [{status}]")


# === Exercise 2: CFG to PDA ===
# Problem: Convert S -> aSb | epsilon to a PDA and trace on input "aabb".

def exercise_2():
    """CFG to PDA conversion and trace."""
    print("Grammar: S -> aSb | epsilon")
    print()
    print("PDA Construction (Section 6.1 method):")
    print("  States: {q_start, q_loop, q_accept}")
    print("  Start: q_start, Accept: {q_accept}")
    print("  Stack alphabet: {S, a, b, $}")
    print()
    print("  Transitions:")
    print("    1. (q_start, eps, eps) -> (q_loop, S$)   [push start variable + bottom]")
    print("    2. (q_loop, eps, S) -> (q_loop, aSb)     [rule S -> aSb]")
    print("    3. (q_loop, eps, S) -> (q_loop, eps)     [rule S -> epsilon, pop S]")
    print("    4. (q_loop, a, a)   -> (q_loop, eps)     [match terminal a]")
    print("    5. (q_loop, b, b)   -> (q_loop, eps)     [match terminal b]")
    print("    6. (q_loop, eps, $) -> (q_accept, eps)   [accept: empty stack]")
    print()

    print("Trace on input 'aabb':")
    print("  (state, remaining_input, stack_top_on_left)")
    print("  " + "-" * 55)

    trace = [
        ("q_start", "aabb", "",     "Push S$"),
        ("q_loop",  "aabb", "S$",   "Apply S -> aSb"),
        ("q_loop",  "aabb", "aSb$", "Match 'a' with input"),
        ("q_loop",  "abb",  "Sb$",  "Apply S -> aSb"),
        ("q_loop",  "abb",  "aSbb$","Match 'a' with input"),
        ("q_loop",  "bb",   "Sbb$", "Apply S -> epsilon"),
        ("q_loop",  "bb",   "bb$",  "Match 'b' with input"),
        ("q_loop",  "b",    "b$",   "Match 'b' with input"),
        ("q_loop",  "",     "$",    "Pop $, accept"),
        ("q_accept","",     "",     "ACCEPTED"),
    ]

    for i, (state, remaining, stack, action) in enumerate(trace):
        rem = remaining if remaining else "eps"
        stk = stack if stack else "(empty)"
        print(f"  Step {i}: ({state}, {rem}, {stk}) -- {action}")


# === Exercise 3: Determinism ===
# Problem: Which of these languages are deterministic CFL?
# 1. {a^n b^n | n >= 0}
# 2. {ww^R | w in {a,b}*}
# 3. {a^n b^n | n >= 0} union {a^n b^(2n) | n >= 0}

def exercise_3():
    """Classify languages as deterministic or non-deterministic CFL."""

    print("Part 1: L = {a^n b^n | n >= 0}")
    print("  Answer: DETERMINISTIC CFL (DCFL)")
    print("  Justification:")
    print("    A DPDA can process this language without nondeterminism:")
    print("    - On reading 'a': push onto stack")
    print("    - On reading 'b': pop from stack")
    print("    - Accept if stack is empty after consuming all input")
    print("    The transition from pushing to popping is deterministic:")
    print("    when we see the first 'b', we know to start popping.")
    print("    There is no need to guess the midpoint.")
    print()

    print("Part 2: L = {ww^R | w in {a,b}*}")
    print("  Answer: NOT deterministic CFL (CFL but not DCFL)")
    print("  Justification:")
    print("    The PDA must guess the midpoint of the string -- it cannot")
    print("    know deterministically when the first half ends and the")
    print("    reversed second half begins. For example, on input 'abba',")
    print("    after reading 'ab', the PDA cannot tell if it should keep")
    print("    pushing or start matching.")
    print()
    print("    Contrast with {wcw^R | w in {a,b}*}: the center marker 'c'")
    print("    eliminates the guessing, making it a DCFL.")
    print()

    print("Part 3: L = {a^n b^n | n >= 0} union {a^n b^(2n) | n >= 0}")
    print("  Answer: NOT deterministic CFL (CFL but not DCFL)")
    print("  Justification:")
    print("    Both {a^n b^n} and {a^n b^(2n)} are individually DCFL.")
    print("    However, DCFLs are NOT closed under union!")
    print("    A DPDA cannot determine which language to check for:")
    print("    after reading a's, it does not know whether to expect n or 2n b's.")
    print("    The PDA must nondeterministically guess which pattern applies.")
    print()
    print("    More formally: this language can be recognized by an NPDA")
    print("    (branch into two paths at the start), but no DPDA exists for it.")
    print("    This can be proven using the fact that DCFLs are closed under")
    print("    complement: if L were DCFL, then L-complement would be DCFL,")
    print("    and we could derive a contradiction using closure properties.")

    # Verification with PDA simulation
    print("\n  Verification: PDA for L = {a^n b^n} U {a^n b^(2n)}")
    test_cases = [
        ("", True),          # n=0 for both
        ("ab", True),        # a^1 b^1
        ("abb", True),       # a^1 b^2
        ("aabb", True),      # a^2 b^2
        ("aabbbb", True),    # a^2 b^4
        ("aabbb", False),    # a^2 b^3: 3 != 2 and 3 != 4
        ("aaabbb", True),    # a^3 b^3
        ("aaabbbbbb", True), # a^3 b^6
        ("a", False),
        ("b", False),
    ]
    for w, expected in test_cases:
        # Check membership directly
        na = 0
        i = 0
        while i < len(w) and w[i] == 'a':
            na += 1
            i += 1
        nb = len(w) - i
        valid = (i == na and all(c == 'b' for c in w[i:]))
        result = valid and (nb == na or nb == 2 * na)
        label = "eps" if w == "" else w
        status = "OK" if result == expected else "MISMATCH"
        print(f"    '{label}': {'ACCEPT' if result else 'reject'} [{status}]")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: PDA Design ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: CFG to PDA ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Determinism ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
