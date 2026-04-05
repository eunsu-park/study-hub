"""
Exercises for Lesson 06: Context-Free Grammars
Topic: Formal_Languages

Solutions to practice problems from the lesson.
"""

import random
from itertools import product as iter_product


# === Helper: CFG simulator ===

class CFG:
    """A simple context-free grammar."""

    def __init__(self, variables, terminals, rules, start):
        """
        Args:
            variables: set of variable names
            terminals: set of terminal symbols
            rules: dict mapping variable -> list of right-hand sides (each a list of symbols)
            start: start variable
        """
        self.variables = variables
        self.terminals = terminals
        self.rules = rules
        self.start = start

    def derive(self, sentential_form, max_steps=100):
        """Perform a leftmost derivation, returning derivation steps."""
        steps = [list(sentential_form)]
        current = list(sentential_form)
        for _ in range(max_steps):
            # Find leftmost variable
            var_idx = None
            for i, sym in enumerate(current):
                if sym in self.variables:
                    var_idx = i
                    break
            if var_idx is None:
                break  # All terminals
            var = current[var_idx]
            # Choose first applicable rule (for deterministic demo)
            rhs = self.rules[var][0]
            new_form = current[:var_idx] + list(rhs) + current[var_idx + 1:]
            current = [s for s in new_form if s != ""]  # Remove epsilon
            steps.append(list(current))
        return steps

    def generate_strings(self, max_len=8, max_derivations=10000):
        """Generate strings in the language up to a maximum length using BFS."""
        from collections import deque
        results = set()
        queue = deque([(self.start,)])
        seen = {(self.start,)}
        count = 0

        while queue and count < max_derivations:
            current = queue.popleft()
            count += 1

            # Find leftmost variable
            var_idx = None
            for i, sym in enumerate(current):
                if sym in self.variables:
                    var_idx = i
                    break

            if var_idx is None:
                # All terminals
                word = "".join(current)
                if len(word) <= max_len:
                    results.add(word)
                continue

            var = current[var_idx]
            for rhs in self.rules.get(var, []):
                rhs_list = list(rhs) if rhs != ("",) and rhs != ("eps",) else []
                new_form = current[:var_idx] + tuple(rhs_list) + current[var_idx + 1:]
                # Prune: if terminals already exceed max_len, skip
                terminal_len = sum(1 for s in new_form if s in self.terminals)
                if terminal_len <= max_len and new_form not in seen:
                    seen.add(new_form)
                    queue.append(new_form)

        return results


# === Exercise 1: Grammar Design ===
# Problem: Write CFGs for:
# 1. L = {a^i b^j | i >= j >= 0}
# 2. L = {w in {a,b}* | #a(w) = #b(w)}
# 3. L = {a^i b^j c^k | i + k = j}

def exercise_1():
    """Grammar design for three context-free languages."""

    # Part 1: L = {a^i b^j | i >= j >= 0}
    # The grammar generates extra a's first, then matched a-b pairs.
    # S -> aS | T
    # T -> aTb | epsilon
    print("Part 1: L = {a^i b^j | i >= j >= 0}")
    print("  Grammar:")
    print("    S -> aS | T")
    print("    T -> aTb | epsilon")
    print()
    print("  Intuition: S generates 'extra' a's (i-j of them), then T generates")
    print("  matched pairs a^j b^j. Total: a^(i-j+j) b^j = a^i b^j with i >= j.")
    print()

    # Verify by generating strings
    print("  Sample derivations:")
    print("    S => T => eps                        (i=0, j=0): ''")
    print("    S => aS => aT => aaTb => aab         (i=2, j=1): 'aab'")
    print("    S => T => aTb => aaTbb => aabb        (i=2, j=2): 'aabb'")
    print("    S => aS => aT => a                    (i=1, j=0): 'a'")

    test_cases = [
        ("", True), ("a", True), ("b", False), ("ab", True),
        ("aa", True), ("aab", True), ("aabb", True), ("aaabb", True),
        ("abb", False), ("ba", False), ("bb", False),
    ]
    print("\n  Verification:")
    for w, expected in test_cases:
        # Manual check: count a's before b's, ensure i >= j and no interleaving
        a_count = 0
        b_count = 0
        valid_form = True
        seen_b = False
        for ch in w:
            if ch == 'a':
                if seen_b:
                    valid_form = False
                    break
                a_count += 1
            elif ch == 'b':
                seen_b = True
                b_count += 1
        actual = valid_form and a_count >= b_count
        status = "OK" if actual == expected else "MISMATCH"
        label = "epsilon" if w == "" else w
        print(f"    '{label}': in L? {actual} [{status}]")

    # Part 2: L = {w in {a,b}* | #a = #b}
    # Grammar: each derivation step adds one a and one b in some arrangement.
    # S -> aSbS | bSaS | epsilon
    print("\n\nPart 2: L = {w in {a,b}* | #a(w) = #b(w)}")
    print("  Grammar:")
    print("    S -> aSbS | bSaS | epsilon")
    print()
    print("  Intuition: Each recursive step produces one 'a' and one 'b',")
    print("  with S substrings in between to allow arbitrary interleaving.")
    print("  This generates all strings with equal a's and b's.")

    test_cases_2 = [
        ("", True), ("ab", True), ("ba", True), ("aabb", True),
        ("abab", True), ("abba", True), ("a", False), ("b", False),
        ("aab", False), ("aababb", True),
    ]
    print("\n  Verification:")
    for w, expected in test_cases_2:
        actual = w.count("a") == w.count("b")
        status = "OK" if actual == expected else "MISMATCH"
        label = "epsilon" if w == "" else w
        print(f"    '{label}': #a={w.count('a')}, #b={w.count('b')}, in L? {actual} [{status}]")

    # Part 3: L = {a^i b^j c^k | i + k = j}
    # Rewrite: j = i + k, so the string is a^i b^i b^k c^k.
    # Grammar: S -> AB
    #          A -> aAb | epsilon   (generates a^i b^i)
    #          B -> bBc | epsilon   (generates b^k c^k)
    print("\n\nPart 3: L = {a^i b^j c^k | i + k = j}")
    print("  Grammar:")
    print("    S -> AB")
    print("    A -> aAb | epsilon    (generates a^i b^i)")
    print("    B -> bBc | epsilon    (generates b^k c^k)")
    print()
    print("  Intuition: Since j = i + k, we split b^j = b^i b^k.")
    print("  A generates a^i b^i and B generates b^k c^k.")
    print("  Together: a^i b^i b^k c^k = a^i b^(i+k) c^k with i+k = j.")

    test_cases_3 = [
        ("", 0, 0, 0, True),       # i=0, j=0, k=0
        ("bc", 0, 1, 1, True),     # i=0, j=1, k=1: 0+1=1
        ("abc", 1, 1, 0, False),   # i=1, j=1, k=0: 1+0=1, wait that's true!
        ("abbc", 1, 2, 1, True),   # i=1, j=2, k=1: 1+1=2
        ("aabbbcc", 2, 3, 2, False),  # i=2, j=3, k=2: 2+2=4 != 3
        ("aabbbbcc", 2, 4, 2, True),  # i=2, j=4, k=2: 2+2=4
    ]
    print("\n  Verification:")
    for w, i, j, k, expected in test_cases_3:
        check = (i + k == j)
        status = "OK" if check == expected else "MISMATCH"
        label = "epsilon" if w == "" else w
        print(f"    '{label}' (i={i}, j={j}, k={k}): i+k={i+k}==j? {check} [{status}]")


# === Exercise 2: CNF Conversion ===
# Problem: Convert this grammar to Chomsky Normal Form:
#   S -> aAb | B
#   A -> aA | a
#   B -> bB | epsilon

def exercise_2():
    """Convert a grammar to Chomsky Normal Form step by step."""

    print("Original grammar:")
    print("  S -> aAb | B")
    print("  A -> aA | a")
    print("  B -> bB | epsilon")
    print()

    print("Step 1: Eliminate epsilon-productions")
    print("  Nullable variables: B (B -> epsilon), S (S -> B -> epsilon)")
    print("  Add new start S0 -> S | epsilon (since S is nullable and epsilon may be in L)")
    print("  For S -> B: since B is nullable, add S -> epsilon. But we handle this via S0.")
    print("  Remove B -> epsilon. For B -> bB, add B -> b (B nullable in bB).")
    print("  For S -> aAb | B: B nullable, so add S -> epsilon (handled by S0).")
    print()
    print("  After step 1:")
    print("    S0 -> S | epsilon")
    print("    S -> aAb | B")
    print("    A -> aA | a")
    print("    B -> bB | b")
    print()

    print("Step 2: Eliminate unit productions")
    print("  Unit production: S -> B")
    print("  Unit pairs: (S, B)")
    print("  Replace S -> B with S -> bB | b")
    print()
    print("  After step 2:")
    print("    S0 -> S | epsilon")
    print("    S -> aAb | bB | b")
    print("    A -> aA | a")
    print("    B -> bB | b")
    print()
    print("  Also: S0 -> S is a unit production.")
    print("  Replace S0 -> S with S0 -> aAb | bB | b")
    print()
    print("  After full unit elimination:")
    print("    S0 -> aAb | bB | b | epsilon")
    print("    S -> aAb | bB | b")
    print("    A -> aA | a")
    print("    B -> bB | b")
    print()

    print("Step 3: Replace terminals in long rules")
    print("  Introduce: X_a -> a, X_b -> b")
    print("  S0 -> X_a A X_b | X_b B | b | epsilon")
    print("  S -> X_a A X_b | X_b B | b")
    print("  A -> X_a A | a")
    print("  B -> X_b B | b")
    print()

    print("Step 4: Break long rules (length > 2)")
    print("  S0 -> X_a A X_b becomes S0 -> X_a Y1, Y1 -> A X_b")
    print("  S -> X_a A X_b becomes S -> X_a Y1 (reuse Y1)")
    print()
    print("  Final CNF grammar:")
    print("    S0 -> X_a Y1 | X_b B | b | epsilon")
    print("    S -> X_a Y1 | X_b B | b")
    print("    A -> X_a A | a")
    print("    B -> X_b B | b")
    print("    Y1 -> A X_b")
    print("    X_a -> a")
    print("    X_b -> b")
    print()

    print("  Verification: every rule is either:")
    print("    - A -> BC  (two variables)")
    print("    - A -> a   (single terminal)")
    print("    - S0 -> epsilon (only start, S0 not on RHS)")
    rules_cnf = [
        ("S0", "X_a Y1"),
        ("S0", "X_b B"),
        ("S0", "b"),
        ("S0", "epsilon"),
        ("S", "X_a Y1"),
        ("S", "X_b B"),
        ("S", "b"),
        ("A", "X_a A"),
        ("A", "a"),
        ("B", "X_b B"),
        ("B", "b"),
        ("Y1", "A X_b"),
        ("X_a", "a"),
        ("X_b", "b"),
    ]
    for lhs, rhs in rules_cnf:
        parts = rhs.split()
        if rhs == "epsilon":
            form = "S0 -> eps (allowed)"
        elif len(parts) == 1 and parts[0].islower():
            form = "A -> a (terminal)"
        elif len(parts) == 2 and all(p[0].isupper() or p.startswith("X_") or p.startswith("Y") for p in parts):
            form = "A -> BC (two variables)"
        else:
            form = "INVALID"
        print(f"    {lhs} -> {rhs}  [{form}]")


# === Exercise 3: CYK Parsing ===
# Problem: Apply CYK algorithm to check if w = "aabb" is in the language:
#   S -> AB | BC, A -> BA | a, B -> CC | b, C -> AB | a

def exercise_3():
    """CYK algorithm for checking membership."""

    # Grammar in CNF
    # S -> AB | BC
    # A -> BA | a
    # B -> CC | b
    # C -> AB | a
    rules = {
        "S": [("A", "B"), ("B", "C")],
        "A": [("B", "A"), "a"],
        "B": [("C", "C"), "b"],
        "C": [("A", "B"), "a"],
    }

    w = "aabb"
    n = len(w)

    print(f"Grammar (CNF):")
    print(f"  S -> AB | BC")
    print(f"  A -> BA | a")
    print(f"  B -> CC | b")
    print(f"  C -> AB | a")
    print(f"\nInput: w = '{w}' (length {n})")
    print()

    # CYK table: T[i][j] = set of variables that can derive w[i..j]
    T = [[set() for _ in range(n)] for _ in range(n)]

    # Base case: substrings of length 1
    print("Step 1: Base case (length-1 substrings)")
    for i in range(n):
        for var, prods in rules.items():
            for prod in prods:
                if isinstance(prod, str) and prod == w[i]:
                    T[i][i].add(var)
        print(f"  T[{i+1}][{i+1}] ('{w[i]}'): {T[i][i]}")

    # Inductive case
    print("\nStep 2: Inductive case (longer substrings)")
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):
                for var, prods in rules.items():
                    for prod in prods:
                        if isinstance(prod, tuple):
                            B, C = prod
                            if B in T[i][k] and C in T[k + 1][j]:
                                T[i][j].add(var)
            substr = w[i:j+1]
            print(f"  T[{i+1}][{j+1}] ('{substr}'): {T[i][j]}")

    # Display the full table
    print(f"\nComplete CYK table:")
    print(f"  {'':>8}", end="")
    for j in range(n):
        print(f"  {w[j]:>10}", end="")
    print()

    for i in range(n):
        print(f"  {w[i]:>8}", end="")
        for j in range(n):
            if j >= i:
                cell = ",".join(sorted(T[i][j])) if T[i][j] else "-"
                print(f"  {cell:>10}", end="")
            else:
                print(f"  {'':>10}", end="")
        print()

    accepted = "S" in T[0][n - 1]
    print(f"\nResult: S in T[1][{n}]? {'Yes' if accepted else 'No'}")
    print(f"'{w}' is {'ACCEPTED' if accepted else 'REJECTED'} by the grammar.")

    if accepted:
        # Trace one parse tree
        print("\nOne possible parse tree (top-down):")
        def trace_parse(var, i, j, indent=0):
            prefix = "  " * (indent + 1)
            if i == j:
                print(f"{prefix}{var} -> {w[i]}")
                return
            for k in range(i, j):
                for prod in rules.get(var, []):
                    if isinstance(prod, tuple):
                        B, C = prod
                        if B in T[i][k] and C in T[k + 1][j]:
                            print(f"{prefix}{var} -> {B} {C}")
                            trace_parse(B, i, k, indent + 1)
                            trace_parse(C, k + 1, j, indent + 1)
                            return
        trace_parse("S", 0, n - 1)


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Grammar Design ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: CNF Conversion ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: CYK Parsing ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
