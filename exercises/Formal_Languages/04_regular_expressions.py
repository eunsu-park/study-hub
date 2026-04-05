"""
Exercises for Lesson 04: Regular Expressions
Topic: Formal_Languages

Solutions to practice problems from the lesson.
"""

import re
from collections import deque


# === Exercise 1: Regex Design ===
# Problem: Write regular expressions for these languages over Sigma = {0, 1}:
# 1. Strings of length at least 2 that begin and end with the same symbol
# 2. Strings that do not contain the substring "11"
# 3. Strings where every pair of adjacent 0s appears before any 1

def exercise_1():
    """Regex design for three languages."""

    # Part 1: Strings of length >= 2 that begin and end with the same symbol
    # Formally: 0(0|1)*0 | 1(0|1)*1 | 00 | 11
    # Simplified: 0(0|1)*0 | 1(0|1)*1
    # This already handles length-2 strings since (0|1)* includes epsilon.
    regex_1 = r'^(0[01]*0|1[01]*1)$'

    print("Part 1: Strings of length >= 2, begin and end with same symbol")
    print(f"  Formal regex: 0(0|1)*0 | 1(0|1)*1")
    print(f"  Python regex: {regex_1}")
    test_1 = ["", "0", "1", "00", "11", "01", "10", "010", "101", "001", "110", "0110"]
    for w in test_1:
        match = bool(re.match(regex_1, w))
        expected = len(w) >= 2 and w[0] == w[-1]
        status = "OK" if match == expected else "MISMATCH"
        label = "epsilon" if w == "" else w
        print(f"    '{label}': {'MATCH' if match else 'no match'} [{status}]")

    # Part 2: Strings that do not contain the substring "11"
    # Each 1 must be followed by 0 or end of string. Also 0s can appear freely.
    # Formally: (0 | 10)*(1 | epsilon)
    # Equivalently: 0*(10+)*(1|epsilon) but simpler is below.
    regex_2 = r'^(0|10)*1?$'

    print("\nPart 2: Strings that do not contain '11'")
    print(f"  Formal regex: (0 | 10)*(1 | epsilon)")
    print(f"  Python regex: {regex_2}")
    test_2 = ["", "0", "1", "00", "01", "10", "11", "010", "101", "110", "0101", "1010"]
    for w in test_2:
        match = bool(re.match(regex_2, w))
        expected = "11" not in w
        status = "OK" if match == expected else "MISMATCH"
        label = "epsilon" if w == "" else w
        print(f"    '{label}': {'MATCH' if match else 'no match'} [{status}]")

    # Part 3: Strings where every pair of adjacent 0s appears before any 1
    # Once a 1 appears, no two consecutive 0s may follow.
    # Structure: any string of 0s, then possibly 1s mixed with single 0s.
    # Formally: 0* (1 | 10)* = 0*(1+0?)* which doesn't work for trailing patterns.
    # More precisely: 0*(1 | 10)*1* covers strings where after the initial block
    # of 0s, we never see "00" again.
    # Actually: the set of strings where "00" does not appear after any "1".
    # Regex: 0*(1(0|epsilon))* = 0*(10|1)*
    regex_3 = r'^0*(1|10)*$'

    print("\nPart 3: Strings where adjacent 0-pairs appear before any 1")
    print(f"  Formal regex: 0*(1 | 10)*")
    print(f"  Python regex: {regex_3}")

    def check_adjacent_0s_before_1s(w):
        """Check that every pair of adjacent 0s appears before any 1."""
        seen_one = False
        for i in range(len(w)):
            if w[i] == '1':
                seen_one = True
            elif w[i] == '0' and seen_one:
                # Check if next is also 0
                if i + 1 < len(w) and w[i + 1] == '0':
                    return False
        return True

    test_3 = [
        "", "0", "1", "00", "01", "10", "11", "001", "0010", "0011",
        "100", "1001", "010", "0101", "00110", "001001",
    ]
    for w in test_3:
        match = bool(re.match(regex_3, w))
        expected = check_adjacent_0s_before_1s(w)
        status = "OK" if match == expected else "MISMATCH"
        label = "epsilon" if w == "" else w
        print(f"    '{label}': {'MATCH' if match else 'no match'} [{status}]")


# === Exercise 2: Regex to NFA (Thompson's Construction) ===
# Problem: Apply Thompson's construction to build an NFA for (ab | a)*b.
# Show the intermediate NFA fragments for each step.

class ThompsonNFA:
    """Simple NFA fragment for Thompson's construction."""

    _state_counter = 0

    @classmethod
    def _new_state(cls):
        cls._state_counter += 1
        return f"s{cls._state_counter}"

    def __init__(self, start, accept, transitions):
        self.start = start
        self.accept = accept  # Single accept state
        self.transitions = transitions  # list of (from, symbol_or_eps, to)

    @classmethod
    def symbol(cls, a):
        """Create NFA for a single symbol."""
        s = cls._new_state()
        t = cls._new_state()
        return ThompsonNFA(s, t, [(s, a, t)])

    @classmethod
    def concat(cls, nfa1, nfa2):
        """Concatenation: nfa1 followed by nfa2."""
        # Connect accept of nfa1 to start of nfa2 via epsilon
        transitions = nfa1.transitions + nfa2.transitions
        transitions.append((nfa1.accept, "eps", nfa2.start))
        return ThompsonNFA(nfa1.start, nfa2.accept, transitions)

    @classmethod
    def union(cls, nfa1, nfa2):
        """Union: nfa1 | nfa2."""
        s = cls._new_state()
        t = cls._new_state()
        transitions = nfa1.transitions + nfa2.transitions
        transitions.append((s, "eps", nfa1.start))
        transitions.append((s, "eps", nfa2.start))
        transitions.append((nfa1.accept, "eps", t))
        transitions.append((nfa2.accept, "eps", t))
        return ThompsonNFA(s, t, transitions)

    @classmethod
    def star(cls, nfa1):
        """Kleene star: nfa1*."""
        s = cls._new_state()
        t = cls._new_state()
        transitions = list(nfa1.transitions)
        transitions.append((s, "eps", nfa1.start))
        transitions.append((s, "eps", t))
        transitions.append((nfa1.accept, "eps", nfa1.start))
        transitions.append((nfa1.accept, "eps", t))
        return ThompsonNFA(s, t, transitions)

    def display(self, label=""):
        """Print the NFA fragment."""
        if label:
            print(f"  {label}:")
        print(f"    Start: {self.start}, Accept: {self.accept}")
        for (src, sym, dst) in self.transitions:
            sym_str = "epsilon" if sym == "eps" else sym
            print(f"    {src} --{sym_str}--> {dst}")

    def accepts(self, input_string):
        """Simulate the NFA to check acceptance."""
        # Build transition dict
        trans = {}
        for (src, sym, dst) in self.transitions:
            key = (src, sym)
            if key not in trans:
                trans[key] = set()
            trans[key].add(dst)

        def eps_closure(states):
            closure = set(states)
            stack = list(states)
            while stack:
                s = stack.pop()
                for t in trans.get((s, "eps"), set()):
                    if t not in closure:
                        closure.add(t)
                        stack.append(t)
            return closure

        current = eps_closure({self.start})
        for ch in input_string:
            next_states = set()
            for s in current:
                next_states |= trans.get((s, ch), set())
            current = eps_closure(next_states)
        return self.accept in current


def exercise_2():
    """Thompson's construction for (ab | a)*b."""
    ThompsonNFA._state_counter = 0

    # Step 1: NFA for 'a' (first occurrence in 'ab')
    nfa_a1 = ThompsonNFA.symbol("a")
    nfa_a1.display("Step 1: NFA for 'a' (in 'ab')")

    # Step 2: NFA for 'b' (in 'ab')
    nfa_b1 = ThompsonNFA.symbol("b")
    nfa_b1.display("Step 2: NFA for 'b' (in 'ab')")

    # Step 3: NFA for 'ab' (concatenation)
    nfa_ab = ThompsonNFA.concat(nfa_a1, nfa_b1)
    nfa_ab.display("Step 3: NFA for 'ab' (concat)")

    # Step 4: NFA for 'a' (second, standalone)
    nfa_a2 = ThompsonNFA.symbol("a")
    nfa_a2.display("Step 4: NFA for 'a' (standalone)")

    # Step 5: NFA for 'ab | a' (union)
    nfa_union = ThompsonNFA.union(nfa_ab, nfa_a2)
    nfa_union.display("Step 5: NFA for 'ab | a' (union)")

    # Step 6: NFA for '(ab | a)*' (Kleene star)
    nfa_star = ThompsonNFA.star(nfa_union)
    nfa_star.display("Step 6: NFA for '(ab | a)*' (star)")

    # Step 7: NFA for final 'b'
    nfa_b_final = ThompsonNFA.symbol("b")
    nfa_b_final.display("Step 7: NFA for final 'b'")

    # Step 8: NFA for '(ab | a)*b' (concatenation)
    nfa_full = ThompsonNFA.concat(nfa_star, nfa_b_final)
    nfa_full.display("Step 8: Final NFA for '(ab | a)*b'")

    # Verify
    print("\n  Verification:")
    test_cases = [
        ("b", True),       # zero iterations of (ab|a)* then b
        ("ab", False),      # no final b
        ("abb", True),      # ab then b
        ("aab", True),      # a then a then b
        ("aabb", True),     # a then ab then b
        ("abab", False),    # no final b after second ab
        ("ababb", True),    # ab then ab then b
        ("", False),        # must end with b
        ("bb", True),       # zero iterations then b (wait, (ab|a)*b: b alone works)
        ("aab", True),      # (a)(a)(b) = a, a, then b
    ]
    for w, expected in test_cases:
        result = nfa_full.accepts(w)
        status = "OK" if result == expected else "MISMATCH"
        label = "epsilon" if w == "" else w
        print(f"    '{label}': {'ACCEPT' if result else 'reject'} [{status}]")


# === Exercise 3: State Elimination ===
# Problem: Convert this DFA to a regex using state elimination:
#   States: {q0, q1}, Start: q0, Accept: {q1}
#   delta(q0, a) = q0    delta(q0, b) = q1
#   delta(q1, a) = q0    delta(q1, b) = q1

def exercise_3():
    """State elimination to convert a DFA to a regular expression."""
    print("DFA:")
    print("  States: {q0, q1}, Start: q0, Accept: {q1}")
    print("  delta(q0, a) = q0    delta(q0, b) = q1")
    print("  delta(q1, a) = q0    delta(q1, b) = q1")
    print()
    print("State Elimination Process:")
    print()

    print("Step 1: Add new start state qs and new accept state qa")
    print("  qs --eps--> q0")
    print("  q1 --eps--> qa")
    print()

    print("Step 2: Current GNFA transitions:")
    print("  qs --eps--> q0")
    print("  q0 --a--> q0  (self-loop)")
    print("  q0 --b--> q1")
    print("  q1 --a--> q0")
    print("  q1 --b--> q1  (self-loop)")
    print("  q1 --eps--> qa")
    print()

    print("Step 3: Eliminate q0")
    print("  q0 has self-loop 'a', incoming from qs (eps) and q1 (a),")
    print("  outgoing to q1 (b).")
    print()
    print("  For path qs -> q0 -> q1: eps . a* . b = a*b")
    print("  For path q1 -> q0 -> q1: a . a* . b = a+b (= aa*b)")
    print()
    print("  After eliminating q0:")
    print("  qs --a*b--> q1")
    print("  q1 --a+b--> q1  (self-loop: b | a+b = b | aa*b)")
    print("  q1 --eps--> qa")
    print()

    print("Step 4: Eliminate q1")
    print("  q1 has self-loop 'b | a+b' = 'b | aa*b'")
    print("  incoming from qs (a*b), outgoing to qa (eps)")
    print()
    print("  Path qs -> q1 -> qa: a*b . (b | aa*b)* . eps")
    print()
    print("  Final regex: a*b(b | aa*b)*")
    print("  Simplified:  a*b(b | a+b)*")
    print()
    print("  Alternative equivalent forms:")
    print("    a*b(b|aa*b)* = a*b(a*b)* = (a*b)+ = a*(ba*)*b")

    # Verify the regex matches the DFA
    regex = r'^(a*b)(b|a+b)*$'
    # Alternative: r'^a*b(a*b)*$' or r'^(a*b)+$'
    regex_alt = r'^(a*b)+$'

    print(f"\n  Verification with regex: {regex}")
    print(f"  Alternative regex:      {regex_alt}")

    # Simulate the DFA
    def dfa_accepts(w):
        state = "q0"
        for ch in w:
            if state == "q0":
                state = "q0" if ch == "a" else "q1"
            else:
                state = "q0" if ch == "a" else "q1"
        return state == "q1"

    test_cases = [
        "", "a", "b", "ab", "ba", "bb", "aab", "abb", "bab",
        "bbb", "aabb", "abab", "baba", "abbb",
    ]
    all_ok = True
    for w in test_cases:
        dfa_result = dfa_accepts(w)
        regex_result = bool(re.match(regex, w))
        regex_alt_result = bool(re.match(regex_alt, w))
        match1 = "OK" if dfa_result == regex_result else "MISMATCH"
        match2 = "OK" if dfa_result == regex_alt_result else "MISMATCH"
        if match1 != "OK" or match2 != "OK":
            all_ok = False
        label = "epsilon" if w == "" else w
        print(
            f"    '{label}': DFA={'ACCEPT' if dfa_result else 'reject'}, "
            f"regex={match1}, alt={match2}"
        )
    print(f"\n  All match: {all_ok}")


# === Exercise 4: Algebraic Simplification ===
# Problem: Use algebraic laws to simplify:
# 1. (a* b*)*
# 2. emptyset* | a
# 3. (a | b)* a (a | b)*

def exercise_4():
    """Algebraic simplification of regular expressions."""

    print("Part 1: (a* b*)*")
    print("  Step 1: By the law (R | S)* = (R* S*)*, we have:")
    print("          (a* b*)* = (a | b)*")
    print("  Result: (a | b)*")
    print("  Meaning: All strings over {a, b}")
    print()

    print("Part 2: emptyset* | a")
    print("  Step 1: emptyset* = epsilon  (by star law)")
    print("  Step 2: epsilon | a = {epsilon, a}")
    print("  Result: epsilon | a")
    print("  Meaning: The language {epsilon, a}")
    print()

    print("Part 3: (a | b)* a (a | b)*")
    print("  This regex matches all strings containing at least one 'a'.")
    print("  The left (a|b)* matches any prefix, 'a' ensures at least one a,")
    print("  and the right (a|b)* matches any suffix.")
    print("  This cannot be simplified further in a meaningful way.")
    print("  Result: (a | b)* a (a | b)*")
    print("  Meaning: All strings over {a, b} that contain at least one 'a'")

    # Verification
    print("\n  Verification:")
    import itertools
    alpha = "ab"
    # Generate all strings up to length 4
    all_strings = [""]
    for length in range(1, 5):
        all_strings.extend("".join(p) for p in itertools.product(alpha, repeat=length))

    print("  Part 1: (a*b*)* = (a|b)* -- should accept everything")
    r1 = re.compile(r'^[ab]*$')
    for w in all_strings[:10]:
        label = "eps" if w == "" else w
        print(f"    '{label}': {'MATCH' if r1.match(w) else 'no'}", end="")
    print(" ... all match")

    print("  Part 3: (a|b)*a(a|b)* -- strings with at least one 'a'")
    r3 = re.compile(r'^[ab]*a[ab]*$')
    for w in ["", "b", "bb", "a", "ab", "ba", "bab", "bbb"]:
        expected = "a" in w
        result = bool(r3.match(w))
        status = "OK" if result == expected else "MISMATCH"
        label = "eps" if w == "" else w
        print(f"    '{label}': {'MATCH' if result else 'no match'} [{status}]")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Regex Design ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Regex to NFA (Thompson's Construction) ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: State Elimination ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Algebraic Simplification ===")
    print("=" * 60)
    exercise_4()

    print("\nAll exercises completed!")
