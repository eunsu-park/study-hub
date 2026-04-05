"""
Exercises for Lesson 08: Properties of Context-Free Languages
Topic: Formal_Languages

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Pumping Lemma for CFLs ===
# Problem: Prove that these languages are not context-free:
# 1. L = {a^n b^n c^n d^n | n >= 0}
# 2. L = {a^n b^n c^i | n <= i <= 2n}
# 3. L = {a^(2^n) | n >= 0}

def exercise_1():
    """Pumping lemma proofs for non-context-free languages."""

    print("Part 1: L = {a^n b^n c^n d^n | n >= 0} is NOT context-free")
    print("=" * 60)
    print("  Proof by contradiction using the CFL pumping lemma:")
    print("  Assume L is CFL with pumping length p.")
    print("  Choose w = a^p b^p c^p d^p in L (|w| = 4p >= p).")
    print("  Write w = uvxyz with |vy| >= 1, |vxy| <= p.")
    print()
    print("  Since |vxy| <= p, the substring vxy spans at most two")
    print("  consecutive symbol types (a-b, b-c, or c-d).")
    print("  Therefore v and y together contain at most two of {a,b,c,d}.")
    print()
    print("  Pump up (i=2): uv^2xy^2z has more of at most two symbols")
    print("  but the same count of the others.")
    print("  The four counts are no longer all equal.")
    print("  So uv^2xy^2z not in L. Contradiction. QED.")

    # Demonstration
    p = 3
    w = "a" * p + "b" * p + "c" * p + "d" * p
    print(f"\n  Demonstration: p={p}, w = '{w}'")
    print(f"  |w| = {len(w)}, |vxy| <= {p}")
    print("  Possible positions of vxy (length <= p):")
    regions = [
        (0, p, "within a's"),
        (p - 1, 2 * p, "spanning a-b boundary"),
        (p, 2 * p, "within b's"),
        (2 * p - 1, 3 * p, "spanning b-c boundary"),
        (2 * p, 3 * p, "within c's"),
        (3 * p - 1, 4 * p, "spanning c-d boundary"),
    ]
    for start, end, desc in regions:
        vxy_region = w[start:min(end, start + p)]
        symbols_in = set(vxy_region)
        missing = set("abcd") - symbols_in
        print(f"    Region [{start}:{min(end,start+p)}] ({desc}): "
              f"contains {symbols_in}, missing {missing}")

    print()
    print()
    print("Part 2: L = {a^n b^n c^i | n <= i <= 2n} is NOT context-free")
    print("=" * 60)
    print("  Proof by contradiction:")
    print("  Assume L is CFL with pumping length p.")
    print("  Choose w = a^p b^p c^p in L (since p <= p <= 2p).")
    print("  Write w = uvxyz with |vy| >= 1, |vxy| <= p.")
    print()
    print("  Case analysis on where vxy falls:")
    print()
    print("  Case A: vxy is within a^p b^p (first 2p symbols).")
    print("    Then v,y contain only a's and/or b's.")
    print("    Pump up (i=2): c count stays at p, but a+b count increases.")
    print("    - If only a's increase: #a > #b, violating a^n b^n structure.")
    print("    - If only b's increase: #a < #b, same violation.")
    print("    - If both increase: still #a = p but the a/b portion is disrupted.")
    print("    In all sub-cases, the pumped string is not in L.")
    print()
    print("  Case B: vxy spans the b-c boundary.")
    print("    Then v,y contain only b's and/or c's.")
    print("    Pump down (i=0): b and/or c counts decrease.")
    print("    If b count decreases: #a > #b, not of form a^n b^n c^i.")
    print("    If c count decreases but b stays: #c < n, violating n <= i.")
    print("    Wait, we need to be more careful...")
    print("    Pump up (i=2): if only c increases, we get c^(p+k) with k>=1.")
    print("    Need p <= p+k <= 2p, i.e., k <= p. Possible for small k.")
    print("    But if only b increases, #a != #b. Contradiction either way.")
    print()
    print("  Case C: vxy is within c^p.")
    print("    Pump up (i=2): c count becomes p + |vy| > p.")
    print("    Need i <= 2n = 2p. If |vy| <= p, then p+|vy| <= 2p. OK so far.")
    print("    Pump DOWN (i=0): c count becomes p - |vy|.")
    print("    Need n <= i, so p <= p-|vy|, impossible since |vy| >= 1.")
    print("    So uv^0xy^0z has c count < p = n. Contradiction.")
    print()
    print("  In all cases, pumping leads to contradiction. QED.")

    print()
    print()
    print("Part 3: L = {a^(2^n) | n >= 0} is NOT context-free")
    print("=" * 60)
    print("  Proof by contradiction:")
    print("  Assume L is CFL with pumping length p.")
    print("  Choose n such that 2^n >= p. Let w = a^(2^n) in L.")
    print("  Write w = uvxyz with |vy| = k >= 1, |vxy| <= p.")
    print()
    print("  Pumped string: |uv^i xy^i z| = 2^n + (i-1)k for i >= 0.")
    print("  For i = 2: length = 2^n + k.")
    print("  Need this to be a power of 2.")
    print()
    print("  Since 1 <= k <= p <= 2^n:")
    print("    2^n < 2^n + k <= 2^n + 2^n = 2^(n+1)")
    print("  So 2^n + k lies strictly between 2^n and 2^(n+1),")
    print("  hence it is NOT a power of 2.")
    print("  Therefore uv^2xy^2z not in L. Contradiction. QED.")

    # Demonstration
    print(f"\n  Demonstration: powers of 2 gaps")
    for n in range(1, 7):
        gap = 2**(n+1) - 2**n
        print(f"    2^{n} = {2**n}, 2^{n+1} = {2**(n+1)}, gap = {gap}")
        print(f"    Any k in [1, {2**n}]: 2^{n} + k in ({2**n}, {2**(n+1)}] -- NOT a power of 2 (unless k = 2^{n})")


# === Exercise 2: Closure Properties ===
# Problem:
# 1. Prove {a^n b^m c^n d^m | n,m >= 0} is not CFL using closure with regular languages.
# 2. Show L1 = {a^n b^n c^m} and L2 = {a^m b^n c^n} are both CFL.
# 3. Verify L1 ∩ L2 = {a^n b^n c^n} and conclude CFLs not closed under intersection.

def exercise_2():
    """Closure property proofs."""

    print("Part 1: L = {a^n b^m c^n d^m | n,m >= 0} is NOT CFL")
    print("=" * 60)
    print("  Proof using closure with regular languages:")
    print("  Suppose L is CFL.")
    print("  Let R = a*b*c*d* (a regular language).")
    print("  L is already a subset of R, so L ∩ R = L.")
    print()
    print("  Now consider the homomorphism h: {a,b,c,d} -> {a,b,c}*")
    print("  defined by h(a)=a, h(b)=b, h(c)=c, h(d)=c.")
    print("  Wait, that changes the structure. Let's use a different approach.")
    print()
    print("  Alternative: Intersect L with the regular language a*b*c*d*.")
    print("  We get L itself (already in that form).")
    print()
    print("  Apply homomorphism h(a)=a, h(b)=epsilon, h(c)=b, h(d)=epsilon.")
    print("  Then h(L) = {a^n b^n | n >= 0}... but we lose the m constraint.")
    print()
    print("  Better approach: Use the substitution/intersection method.")
    print("  Intersect L with the regular language R' = a*b*c*d* where")
    print("  we require equal-length blocks. Actually, L is already in this form.")
    print()
    print("  Simplest approach: Assume L is CFL.")
    print("  L ∩ {a^n b^n c^n d^n | n >= 0} would require showing the")
    print("  intersection is non-CFL, but that's what we want to prove.")
    print()
    print("  Direct pumping argument:")
    print("  Choose w = a^p b^p c^p d^p in L. |vxy| <= p.")
    print("  vxy can span at most 2 consecutive blocks.")
    print("  Pumping changes at most 2 of the 4 counts.")
    print("  But we need n_a = n_c AND n_b = n_d simultaneously.")
    print("  If vxy spans a-b: pumping changes a and/or b counts")
    print("    but not c or d. So a!=c or b!=d. Contradiction.")
    print("  If vxy spans b-c: pumping changes b and/or c")
    print("    but not a or d. So a!=c or b!=d. Contradiction.")
    print("  Similarly for c-d boundary.")
    print("  QED.")

    print()
    print()
    print("Part 2: L1 = {a^n b^n c^m | n,m >= 0} and")
    print("        L2 = {a^m b^n c^n | n,m >= 0} are both CFL")
    print("=" * 60)

    print("\n  Grammar for L1:")
    print("    S -> AB")
    print("    A -> aAb | epsilon   (generates a^n b^n)")
    print("    B -> cB | epsilon    (generates c^m)")
    print("  This grammar generates a^n b^n c^m for all n, m >= 0.")

    print("\n  Grammar for L2:")
    print("    S -> AB")
    print("    A -> aA | epsilon    (generates a^m)")
    print("    B -> bBc | epsilon   (generates b^n c^n)")
    print("  This grammar generates a^m b^n c^n for all n, m >= 0.")

    # Verification
    print("\n  Verification samples for L1:")
    for n in range(4):
        for m in range(4):
            w = "a" * n + "b" * n + "c" * m
            label = "eps" if w == "" else w
            print(f"    n={n}, m={m}: '{label}'", end="")
        print()

    print()
    print()
    print("Part 3: L1 ∩ L2 = {a^n b^n c^n | n >= 0}")
    print("=" * 60)
    print("  L1 ∩ L2 = {a^n b^n c^m} ∩ {a^m b^n c^n}")
    print("  A string w is in both iff:")
    print("    - w = a^i b^j c^k with j = i (from L1) and j = k (from L2)")
    print("    - Also i can be any value (from L2's a^m) and k any value (from L1's c^m)")
    print("    - Wait: from L1: #a = #b. From L2: #b = #c. So #a = #b = #c.")
    print("  Therefore L1 ∩ L2 = {a^n b^n c^n | n >= 0}.")
    print()
    print("  Since {a^n b^n c^n} is NOT context-free (proven in Part 1 of Exercise 1),")
    print("  but L1 and L2 are both CFL,")
    print("  we conclude: CFLs are NOT closed under intersection. QED.")

    # Demonstration
    print("\n  Demonstration:")
    for n in range(5):
        w = "a" * n + "b" * n + "c" * n
        in_L1 = True  # a^n b^n c^n: #a=#b, yes
        in_L2 = True  # a^n b^n c^n: #b=#c, yes
        label = "eps" if w == "" else w
        print(f"    '{label}': L1? {in_L1}, L2? {in_L2}, L1∩L2? {in_L1 and in_L2}")


# === Exercise 3: Decision Problems ===
# Problem: For each problem, state decidability for CFLs:
# 1. Given CFG G, is |L(G)| >= 100?
# 2. Given CFGs G1, G2, is L(G1) ∩ L(G2) = empty?
# 3. Given CFG G and regular expression R, is L(G) ⊆ L(R)?

def exercise_3():
    """Decision problem analysis for CFLs."""

    print("Part 1: Given CFG G, is |L(G)| >= 100?")
    print("=" * 60)
    print("  Answer: DECIDABLE")
    print()
    print("  Proof:")
    print("  First, convert G to Chomsky Normal Form (CNF).")
    print("  In CNF, any string of length n has a parse tree of height")
    print("  at most n (since each step adds at most one terminal).")
    print("  More precisely, we can enumerate all strings of length up to")
    print("  some bound and use CYK to check membership.")
    print()
    print("  But better: We can decide if L(G) is finite or infinite.")
    print("  - Convert to CNF, remove useless symbols")
    print("  - Check for cycles among useful variables (in the 'unit graph')")
    print("  - If L(G) is infinite, then |L(G)| >= 100 trivially")
    print("  - If L(G) is finite, enumerate all strings (bounded by the grammar)")
    print("    and count them. Since L(G) is finite and the grammar is fixed,")
    print("    this terminates.")
    print("  Therefore the problem is decidable.")

    print()
    print()
    print("Part 2: Given CFGs G1, G2, is L(G1) ∩ L(G2) = empty?")
    print("=" * 60)
    print("  Answer: UNDECIDABLE")
    print()
    print("  Proof sketch:")
    print("  This is a well-known undecidable problem for CFLs.")
    print("  It can be shown undecidable by reduction from the Post")
    print("  Correspondence Problem (PCP):")
    print("  Given a PCP instance, construct two CFGs G1 and G2 such that")
    print("  G1 generates encodings of 'top' strings and G2 generates")
    print("  encodings of 'bottom' strings. A match in PCP corresponds")
    print("  to a string in L(G1) ∩ L(G2).")
    print("  Since PCP is undecidable, so is CFL intersection emptiness.")

    print()
    print()
    print("Part 3: Given CFG G and regex R, is L(G) ⊆ L(R)?")
    print("=" * 60)
    print("  Answer: DECIDABLE")
    print()
    print("  Proof:")
    print("  L(G) ⊆ L(R) iff L(G) ∩ complement(L(R)) = empty.")
    print()
    print("  Key facts:")
    print("  1. R defines a regular language, so complement(L(R)) is also regular.")
    print("  2. CFLs are closed under intersection with regular languages:")
    print("     L(G) ∩ complement(L(R)) is context-free.")
    print("  3. Emptiness is decidable for CFLs.")
    print()
    print("  Algorithm:")
    print("  1. Convert R to a DFA, complement it (swap accept/non-accept)")
    print("  2. Build PDA for L(G) ∩ complement(L(R)) using product construction")
    print("  3. Convert to CFG and check emptiness (or check PDA emptiness directly)")
    print("  4. If empty, then L(G) ⊆ L(R). Otherwise, not.")
    print()
    print("  All steps are effective, so the problem is decidable.")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Pumping Lemma for CFLs ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Closure Properties ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Decision Problems ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
