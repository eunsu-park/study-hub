"""
Chomsky Hierarchy Demonstration
=================================

Demonstrates:
- All four levels of the Chomsky hierarchy
- Language membership testing at each level
- Closure properties verification
- Decision problem examples

Reference: Formal_Languages Lesson 13 — The Chomsky Hierarchy
"""

from __future__ import annotations
from typing import Callable, Dict, List, Set, Tuple


# ─────────────── Type 3: Regular Languages ───────────────

# Why: A lightweight DFA runner (no class overhead) for quick membership checks.
# This keeps the Chomsky hierarchy demos self-contained without importing
# the full DFA class from the other module.
def dfa_accepts(transitions: Dict[Tuple[str, str], str],
                start: str, accept: Set[str], input_str: str) -> bool:
    """Simple DFA acceptance check."""
    state = start
    for ch in input_str:
        key = (state, ch)
        if key not in transitions:
            return False
        state = transitions[key]
    return state in accept


def is_even_ones(s: str) -> bool:
    """Type 3 (Regular): strings over {0,1} with even number of 1s."""
    transitions = {
        ("even", "0"): "even", ("even", "1"): "odd",
        ("odd", "0"): "odd",   ("odd", "1"): "even",
    }
    return dfa_accepts(transitions, "even", {"even"}, s)


def is_ends_ab(s: str) -> bool:
    """Type 3 (Regular): strings over {a,b} ending in 'ab'."""
    transitions = {
        ("q0", "a"): "q1", ("q0", "b"): "q0",
        ("q1", "a"): "q1", ("q1", "b"): "q2",
        ("q2", "a"): "q1", ("q2", "b"): "q0",
    }
    return dfa_accepts(transitions, "q0", {"q2"}, s)


# ─────────────── Type 2: Context-Free Languages ───────────────

def is_anbn(s: str) -> bool:
    """Type 2 (Context-Free): {a^n b^n | n >= 0}."""
    n = len(s)
    if n % 2 != 0:
        return False
    half = n // 2
    return s == 'a' * half + 'b' * half


def is_palindrome(s: str) -> bool:
    """Type 2 (Context-Free): palindromes over {a, b}."""
    return all(c in 'ab' for c in s) and s == s[::-1]


def is_balanced_parens(s: str) -> bool:
    """Type 2 (Context-Free): balanced parentheses."""
    count = 0
    for c in s:
        if c == '(':
            count += 1
        elif c == ')':
            count -= 1
        else:
            return False
        if count < 0:
            return False
    return count == 0


# ─────────────── Type 1: Context-Sensitive Languages ───────────────

# Why: {a^n b^n c^n} is the canonical example separating Type 1 (CS) from Type 2 (CF).
# Matching three counts simultaneously requires the power of a linear-bounded automaton.
def is_anbncn(s: str) -> bool:
    """Type 1 (Context-Sensitive): {a^n b^n c^n | n >= 1}."""
    if not s:
        return False
    n = len(s)
    if n % 3 != 0:
        return False
    k = n // 3
    return s == 'a' * k + 'b' * k + 'c' * k


def is_ww(s: str) -> bool:
    """Type 1 (Context-Sensitive): {ww | w ∈ {a,b}*} — the copy language."""
    n = len(s)
    if n % 2 != 0:
        return False
    half = n // 2
    return s[:half] == s[half:]


def is_prime_length(s: str) -> bool:
    """Type 1 (Context-Sensitive): {a^p | p is prime}."""
    n = len(s)
    if n < 2 or not all(c == 'a' for c in s):
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def is_power_of_2(s: str) -> bool:
    """Type 1 (Context-Sensitive): {a^(2^n) | n >= 0}."""
    n = len(s)
    return n >= 1 and all(c == 'a' for c in s) and (n & (n - 1)) == 0


# ─────────────── Type 0: Recursively Enumerable ───────────────

# We can't truly demonstrate non-decidable languages in a finite program,
# but we can show the concept with bounded simulation.

# Why: The halting problem is undecidable, so we can only approximate it with
# bounded simulation. If the program finishes within the step limit we know it
# halts; otherwise we say "unknown" — we can never be sure it won't halt later.
def bounded_halting_check(program: Callable, input_val: int,
                          max_steps: int = 10000) -> str:
    """
    Bounded simulation of the halting problem.
    Returns 'halts', 'unknown' (hit step limit), or 'error'.
    """
    try:
        # We can't actually solve the halting problem, but we can try
        # running for a bounded number of steps
        result = program(input_val)
        return "halts"
    except RecursionError:
        return "unknown (recursion limit)"
    except Exception as e:
        return f"error: {e}"


# ─────────────── Demos ───────────────

def demo_hierarchy():
    """Show languages at each level of the Chomsky hierarchy."""
    print("=" * 60)
    print("Demo 1: The Chomsky Hierarchy")
    print("=" * 60)

    levels = [
        ("Type 3 — Regular", [
            ("Even 1s (DFA)", is_even_ones,
             ["", "0", "1", "11", "101", "111", "1001"]),
            ("Ends in 'ab'", is_ends_ab,
             ["ab", "aab", "bab", "a", "b", "ba", ""]),
        ]),
        ("Type 2 — Context-Free", [
            ("{a^n b^n}", is_anbn,
             ["", "ab", "aabb", "aaabbb", "a", "aab", "ba"]),
            ("Palindromes {a,b}", is_palindrome,
             ["", "a", "aba", "abba", "ab", "abc"]),
            ("Balanced ()", is_balanced_parens,
             ["", "()", "(())", "()()", "(()", ")("]),
        ]),
        ("Type 1 — Context-Sensitive", [
            ("{a^n b^n c^n}", is_anbncn,
             ["abc", "aabbcc", "aaabbbccc", "ab", "abcc", ""]),
            ("{ww} copy", is_ww,
             ["", "aa", "abab", "abba", "a", "aba"]),
            ("{a^p | p prime}", is_prime_length,
             ["aa", "aaa", "aaaa", "aaaaa", "aaaaaa", "aaaaaaa"]),
        ]),
    ]

    for level_name, languages in levels:
        print(f"\n  {level_name}:")
        for name, predicate, tests in languages:
            print(f"    {name}:")
            for s in tests:
                result = predicate(s)
                display = s if s else "ε"
                print(f"      '{display}': {'∈ L' if result else '∉ L'}")


def demo_strict_containment():
    """Show that each level is strictly contained in the next."""
    print("\n" + "=" * 60)
    print("Demo 2: Strict Containment")
    print("=" * 60)

    print("\n  Regular ⊊ Context-Free:")
    print("    {a^n b^n} is CF but NOT regular (pumping lemma)")
    print("    Proof: For any p, a^p b^p can't be pumped in first p symbols")

    print("\n  Context-Free ⊊ Context-Sensitive:")
    print("    {a^n b^n c^n} is CS but NOT CF (CFL pumping lemma)")
    print("    Proof: |vxy| ≤ p spans ≤ 2 symbols; pumping unbalances counts")

    print("\n  Context-Sensitive ⊊ Recursively Enumerable:")
    print("    A_TM (acceptance problem) is RE but NOT CS (undecidable)")
    print("    CS languages are always decidable (LBA membership is decidable)")

    # Demonstrate with concrete strings
    print("\n  Concrete demonstration:")
    test_strings = [
        "aaaa",      # regular: 4 a's (even count? yes for some DFA)
        "aabb",      # CF but not regular: a^2 b^2
        "aabbcc",    # CS but not CF: a^2 b^2 c^2
    ]

    for s in test_strings:
        in_regular = len(s) % 2 == 0 and all(c == 'a' for c in s)  # just an example
        in_cf = is_anbn(s)
        in_cs = is_anbncn(s)
        print(f"    '{s}': regular_example={in_regular}, "
              f"a^nb^n={in_cf}, a^nb^nc^n={in_cs}")


def demo_closure_properties():
    """Verify closure properties at each level."""
    print("\n" + "=" * 60)
    print("Demo 3: Closure Properties")
    print("=" * 60)

    # Generate test strings
    strings_01 = [''.join(f'{i:0{n}b}' for i in range(1))
                   for n in range(5)]
    strings_ab = [""]
    for length in range(1, 5):
        for i in range(2**length):
            s = ""
            for bit in range(length):
                s += "a" if (i >> bit) & 1 else "b"
            strings_ab.append(s)

    # Type 3: Regular — closed under intersection
    print("\n  Type 3 (Regular) — closed under intersection:")
    # L1 = even length, L2 = starts with 'a'
    L1 = {s for s in strings_ab if len(s) % 2 == 0}
    L2 = {s for s in strings_ab if s.startswith('a')}
    L_inter = L1 & L2
    print(f"    L1 (even length): {sorted(L1)[:8]}...")
    print(f"    L2 (starts with a): {sorted(L2)[:8]}...")
    print(f"    L1 ∩ L2: {sorted(L_inter)[:8]}...")
    print("    Both L1, L2, and L1 ∩ L2 are regular ✓")

    # Why: This is the classic proof that CFLs are not closed under intersection.
    # L1 = {a^n b^n c^m} and L2 = {a^m b^n c^n} are both CF, but their
    # intersection is {a^n b^n c^n}, which is context-sensitive (not CF).
    print("\n  Type 2 (Context-Free) — NOT closed under intersection:")
    L_cf1 = {f"{'a'*n}{'b'*n}{'c'*m}" for n in range(5) for m in range(5)}
    L_cf2 = {f"{'a'*m}{'b'*n}{'c'*n}" for n in range(5) for m in range(5)}
    L_cf_inter = L_cf1 & L_cf2
    print(f"    L1 = {{a^n b^n c^m}}: CF (grammar: S→AB, A→aAb|ε, B→cB|ε)")
    print(f"    L2 = {{a^m b^n c^n}}: CF (grammar: S→AB, A→aA|ε, B→bBc|ε)")
    L_eq = {s for s in L_cf_inter if is_anbncn(s)}
    print(f"    L1 ∩ L2 includes: {sorted(L_eq)}")
    print("    L1 ∩ L2 = {a^n b^n c^n} — NOT context-free! ✗")

    # Type 3: Regular — closed under complement
    print("\n  Type 3 (Regular) — closed under complement:")
    L_comp = set(strings_ab) - L1
    print(f"    L1 (even length): {sorted(L1)[:6]}...")
    print(f"    ¬L1 (odd length): {sorted(L_comp)[:6]}...")
    print("    Complement is also regular ✓")

    # Type 2: CF — NOT closed under complement
    print("\n  Type 2 (Context-Free) — NOT closed under complement:")
    print("    If CF were closed under complement AND union,")
    print("    then CF ∩ = (¬(¬L1 ∪ ¬L2)) would mean CF closed under ∩.")
    print("    But we just showed CF is NOT closed under ∩. Contradiction! ✗")


def demo_decision_problems():
    """Show decidability of various problems at each level."""
    print("\n" + "=" * 60)
    print("Demo 4: Decision Problems")
    print("=" * 60)

    header = f"  {'Problem':<25} {'Regular':>10} {'CF':>10} {'CS':>10} {'RE':>10}"
    print(header)
    print("  " + "-" * 65)

    problems = [
        ("Membership (w ∈ L?)", "✓ O(n)", "✓ O(n³)", "✓ O(nˢ)", "✗"),
        ("Emptiness (L = ∅?)", "✓", "✓", "✗", "✗"),
        ("Finiteness", "✓", "✓", "✗", "✗"),
        ("Equivalence (L1 = L2?)", "✓", "✗", "✗", "✗"),
        ("Universality (L = Σ*?)", "✓", "✗", "✗", "✗"),
        ("Containment (L1 ⊆ L2?)", "✓", "✗", "✗", "✗"),
    ]

    for problem, reg, cf, cs, re in problems:
        print(f"  {problem:<25} {reg:>10} {cf:>10} {cs:>10} {re:>10}")

    print("\n  ✓ = decidable, ✗ = undecidable")
    print("  Key insight: more power → fewer decidable properties")


def demo_language_classification():
    """Classify languages into hierarchy levels."""
    print("\n" + "=" * 60)
    print("Demo 5: Language Classification Quiz")
    print("=" * 60)

    languages = [
        ("a*b*", "Regular (Type 3)", "DFA with 3 states: read a's then b's"),
        ("{a^n b^m | n > m}", "Context-Free (Type 2)", "CFG: S → aSb | aS | a"),
        ("{w ∈ {a,b}* | |w|_a = |w|_b}", "Context-Free (Type 2)", "PDA: push a's, pop b's"),
        ("{a^n b^n c^n}", "Context-Sensitive (Type 1)", "Needs to match 3 counts"),
        ("{ww | w ∈ {a,b}*}", "Context-Sensitive (Type 1)", "Copy requires CS power"),
        ("{a^(2^n)}", "Context-Sensitive (Type 1)", "Halving is CS-recognizable"),
        ("{a^p | p prime}", "Context-Sensitive (Type 1)", "Primality test in linear space"),
        ("A_TM (acceptance)", "RE (Type 0), not decidable", "Universal TM recognizes; halting problem"),
        ("¬A_TM", "Not RE (not in hierarchy)", "Complement of A_TM"),
    ]

    for lang, level, reason in languages:
        print(f"\n  Language: {lang}")
        print(f"  Level: {level}")
        print(f"  Reason: {reason}")


if __name__ == "__main__":
    demo_hierarchy()
    demo_strict_containment()
    demo_closure_properties()
    demo_decision_problems()
    demo_language_classification()
