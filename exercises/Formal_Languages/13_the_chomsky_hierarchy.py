"""
Exercises for Lesson 13: The Chomsky Hierarchy
Topic: Formal_Languages

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Classification ===
# Problem: Classify each language into the smallest Chomsky hierarchy level:
# 1. {a^n b^m | n > m}
# 2. {a^n b^n c^n d^n | n >= 0}
# 3. {w in {a,b}* | #a(w) = #b(w)}
# 4. {a^p | p is prime}
# 5. {<M> | M halts on epsilon}

def exercise_1():
    """Classify languages into the Chomsky hierarchy."""

    classifications = [
        {
            "language": "L1 = {a^n b^m | n > m >= 0}",
            "level": "Type 2 (Context-Free)",
            "justification": (
                "This is context-free. Grammar:\n"
                "      S -> aS | aT\n"
                "      T -> aTb | a\n"
                "    Explanation: S generates one or more extra a's then hands off to T.\n"
                "    T generates matched a-b pairs with at least one a.\n"
                "    Total a's > total b's.\n"
                "    It is NOT regular because a DFA cannot count unbounded n and m\n"
                "    to verify n > m (provable via pumping lemma)."
            ),
        },
        {
            "language": "L2 = {a^n b^n c^n d^n | n >= 0}",
            "level": "Type 1 (Context-Sensitive)",
            "justification": (
                "This is context-sensitive but NOT context-free.\n"
                "    Not CF: By the CFL pumping lemma, any pump region |vxy| <= p\n"
                "    can span at most 2 of the 4 symbol types. Pumping disrupts\n"
                "    the 4-way equality.\n"
                "    Is CS: An LBA can verify all four counts are equal using\n"
                "    bounded workspace. A context-sensitive grammar can be\n"
                "    constructed (extending the a^n b^n c^n grammar technique)."
            ),
        },
        {
            "language": "L3 = {w in {a,b}* | #a(w) = #b(w)}",
            "level": "Type 2 (Context-Free)",
            "justification": (
                "This is context-free. Grammar:\n"
                "      S -> aSbS | bSaS | epsilon\n"
                "    Each recursive step adds one a and one b.\n"
                "    It is NOT regular: L intersect a*b* = {a^n b^n},\n"
                "    which is not regular. Since regular languages are closed\n"
                "    under intersection, L cannot be regular."
            ),
        },
        {
            "language": "L4 = {a^p | p is prime}",
            "level": "Type 1 (Context-Sensitive)",
            "justification": (
                "This is context-sensitive (decidable in linear space) but NOT CF.\n"
                "    Not CF: CFL pumping lemma. Choose w = a^q for prime q >= p.\n"
                "    |uv^2xy^2z| = q + |vy|. For i = q+1:\n"
                "    |uv^{q+1}xy^{q+1}z| = q + q*|vy| = q(1+|vy|), which is\n"
                "    composite when |vy| >= 1 and q >= 2.\n"
                "    Is CS: Primality can be tested in O(n) space (trial division\n"
                "    of unary number n requires O(log n) binary workspace,\n"
                "    which fits in O(n) tape cells of an LBA)."
            ),
        },
        {
            "language": "L5 = {<M> | M halts on epsilon}",
            "level": "Type 0 (Recursively Enumerable)",
            "justification": (
                "This is RE (Turing-recognizable) but NOT decidable.\n"
                "    RE: A recognizer simulates M on epsilon. If M halts, accept.\n"
                "    If M loops, the recognizer also loops (never rejects).\n"
                "    Not decidable: This is a variant of the halting problem.\n"
                "    Reduction from A_TM: given <M, w>, construct M' that ignores\n"
                "    its input, simulates M on w, and halts iff M accepts w.\n"
                "    Then M' halts on epsilon iff M accepts w."
            ),
        },
    ]

    for i, item in enumerate(classifications, 1):
        print(f"Language {i}: {item['language']}")
        print(f"  Classification: {item['level']}")
        print(f"  Justification: {item['justification']}")
        print()

    # Verification for L1 (CFL)
    print("Verification: L1 = {a^n b^m | n > m}")
    print("-" * 40)
    for n in range(5):
        for m in range(5):
            w = "a" * n + "b" * m
            in_L = n > m
            label = "eps" if w == "" else w
            if in_L:
                print(f"  n={n}, m={m}: '{label}' IN L1")

    # Verification for L3 (CFL)
    print("\nVerification: L3 = {w | #a = #b} (sample strings)")
    print("-" * 40)
    samples = ["", "ab", "ba", "aabb", "abba", "abab", "baba", "bbaa"]
    for w in samples:
        in_L = w.count("a") == w.count("b")
        label = "eps" if w == "" else w
        print(f"  '{label}': #a={w.count('a')}, #b={w.count('b')}, "
              f"in L3? {in_L}")

    # Verification for L4 (primes)
    print("\nVerification: L4 = {a^p | p is prime} (first few)")
    print("-" * 40)
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    for n in range(20):
        if is_prime(n):
            print(f"  a^{n} = '{'a'*n}' IN L4")


# === Exercise 2: Context-Sensitive Grammar ===
# Problem: Write a CSG for {a^n b^n c^n | n >= 1}. Trace derivation of aabbcc.

def exercise_2():
    """Context-sensitive grammar for {a^n b^n c^n}."""

    print("Context-Sensitive Grammar for {a^n b^n c^n | n >= 1}")
    print("=" * 60)
    print()
    print("Grammar rules:")
    print("  1. S -> aSBC    (generate one a and one B, one C)")
    print("  2. S -> aBC     (base case: one a, one B, one C)")
    print("  3. CB -> BC     (context-sensitive: swap C and B to sort)")
    print("  4. aB -> ab     (convert B to b when preceded by a)")
    print("  5. bB -> bb     (convert B to b when preceded by b)")
    print("  6. bC -> bc     (convert C to c when preceded by b)")
    print("  7. cC -> cc     (convert C to c when preceded by c)")
    print()
    print("Note: Rule 3 (CB -> BC) is the context-sensitive rule.")
    print("It ensures B's appear before C's by 'bubbling' B left past C.")
    print()

    print("Derivation of 'aabbcc' (n=2):")
    print("-" * 40)
    steps = [
        ("S", "Apply rule 1: S -> aSBC"),
        ("aSBC", "Apply rule 2: S -> aBC"),
        ("aaBCBC", "Apply rule 3: CB -> BC (swap inner CB)"),
        ("aaBBCC", "Apply rule 4: aB -> ab (first B)"),
        ("aabBCC", "Apply rule 5: bB -> bb (second B)"),
        ("aabbCC", "Apply rule 6: bC -> bc (first C)"),
        ("aabbcC", "Apply rule 7: cC -> cc (second C)"),
        ("aabbcc", "DONE - all terminals"),
    ]

    for i, (form, rule) in enumerate(steps):
        print(f"  Step {i}: {form:20s} ({rule})")

    print()
    print("Derivation of 'aaabbbccc' (n=3):")
    print("-" * 40)
    steps_3 = [
        ("S", "Rule 1: S -> aSBC"),
        ("aSBC", "Rule 1: S -> aSBC"),
        ("aaSBCBC", "Rule 2: S -> aBC"),
        ("aaaBCBCBC", "Rule 3: CB -> BC (multiple swaps needed)"),
        ("aaaBBCCBC", "Rule 3: CB -> BC"),
        ("aaaBBCBCC", "Rule 3: CB -> BC"),
        ("aaaBBBCCC", "Now sorted! Apply rules 4-7"),
        ("aaabBBCCC", "Rule 4: aB -> ab"),
        ("aaabbBCCC", "Rule 5: bB -> bb"),
        ("aaabbbCCC", "Rule 5: bB -> bb"),
        ("aaabbbcCC", "Rule 6: bC -> bc"),
        ("aaabbbccC", "Rule 7: cC -> cc"),
        ("aaabbbccc", "DONE"),
    ]

    for i, (form, rule) in enumerate(steps_3):
        print(f"  Step {i:2d}: {form:20s} ({rule})")

    # Verification
    print()
    print("Verification: the grammar generates exactly {a^n b^n c^n | n >= 1}")
    print("  n=1: S -> aBC -> abC -> abc")
    print("  n=2: S -> aSBC -> aaBCBC -> aaBBCC -> aabbCC -> aabbcC -> aabbcc")
    print("  n=3: as traced above")
    print("  The grammar cannot generate strings where counts differ,")
    print("  because each S-expansion adds exactly one a, one B, and one C.")


# === Exercise 3: Closure Properties ===
# Problem:
# 1. Show CSLs closed under union by constructing a grammar.
# 2. Explain why CSL closure under complement is nontrivial.

def exercise_3():
    """Closure properties of context-sensitive languages."""

    print("Part 1: CSLs are closed under union")
    print("=" * 60)
    print()
    print("  Proof by grammar construction:")
    print()
    print("  Given CSGs G1 = (V1, Sigma, R1, S1) and G2 = (V2, Sigma, R2, S2)")
    print("  for languages L1 and L2, construct G for L1 union L2:")
    print()
    print("  Step 1: Rename variables so V1 and V2 are disjoint")
    print("    (and neither contains the new start variable S).")
    print()
    print("  Step 2: G = (V1 union V2 union {S}, Sigma, R1 union R2 union {S->S1, S->S2}, S)")
    print()
    print("  The rules S -> S1 and S -> S2 are technically not context-sensitive")
    print("  (they are unit productions). However:")
    print("    - If epsilon is not in L1 or L2, we can eliminate unit productions")
    print("      by replacing S -> S1 with all rules S -> alpha where S1 -> alpha")
    print("      is in R1 (and similarly for S2).")
    print("    - If epsilon is in L1 (S1 -> epsilon allowed), handle S -> epsilon")
    print("      specially (S -> epsilon is allowed for the start symbol in CSGs,")
    print("      provided S does not appear on the right side of any rule).")
    print()
    print("  All resulting rules maintain the non-contracting property")
    print("  (|left side| <= |right side|), so G is a valid CSG.")
    print("  L(G) = L1 union L2. QED.")

    # Demonstration
    print("\n  Demonstration:")
    print("    G1 for {a^n b^n | n >= 1}: S1 -> aS1b | ab")
    print("    G2 for {c^n d^n | n >= 1}: S2 -> cS2d | cd")
    print("    G for L1 union L2:")
    print("      S -> aS1b | ab | cS2d | cd")
    print("      S1 -> aS1b | ab")
    print("      S2 -> cS2d | cd")
    print("    (Unit productions S->S1, S->S2 eliminated by inlining)")

    print()
    print()
    print("Part 2: Why CSL closure under complement is nontrivial")
    print("=" * 60)
    print()
    print("  For regular languages, complementation is trivial:")
    print("    Given a DFA, swap accept and non-accept states. Done!")
    print("    This works because DFAs are deterministic and always halt.")
    print()
    print("  For context-sensitive languages, complementation is HARD because:")
    print()
    print("  1. No simple 'swap' operation:")
    print("     CSLs are defined by grammars (generative) or LBAs (nondeterministic).")
    print("     Grammars don't have 'accept/reject' states to swap.")
    print("     LBAs are nondeterministic -- swapping accept/reject states of an")
    print("     NTM does NOT give the complement language (it gives a different")
    print("     language due to the existential nature of NTM acceptance).")
    print()
    print("  2. Nondeterminism is the core difficulty:")
    print("     For DFAs/NFAs: DFA = NFA (subset construction), so we can always")
    print("     get a deterministic machine and complement it.")
    print("     For LBAs: it is NOT known whether deterministic LBAs = nondeterministic")
    print("     LBAs (this is an open problem!). So we cannot simply determinize.")
    print()
    print("  3. The Immerman-Szelepcsenyi Theorem (1987-88):")
    print("     Proved that NSPACE(s(n)) = coNSPACE(s(n)) for s(n) >= log n.")
    print("     Since CSLs = NSPACE(n) (languages decidable in linear space),")
    print("     this implies CSLs are closed under complement.")
    print("     The proof uses an inductive counting technique:")
    print("       - First count the number of reachable configurations")
    print("       - Then verify that a string is NOT accepted by checking")
    print("         that ALL reachable accepting configurations reject")
    print("     This is a deep result -- it took decades to resolve!")
    print()
    print("  4. Historical significance:")
    print("     Whether CSLs are closed under complement was a major open")
    print("     problem from the 1960s to 1988. Immerman and Szelepcsenyi")
    print("     independently proved it, earning Immerman a share of the")
    print("     Computational Complexity Conference best paper award.")
    print("     The result applies more broadly: any nondeterministic space")
    print("     class equals its complement, which was surprising.")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Classification ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Context-Sensitive Grammar ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Closure Properties ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
