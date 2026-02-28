"""
Exercises for Lesson 14: Advanced Topics and Applications
Topic: Formal_Languages

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Lambda Calculus ===
# Problem:
# 1. Evaluate (lambda x. lambda y. x) a b using beta-reduction.
# 2. Define the Church encoding of PLUS (addition on Church numerals).
# 3. Show that the Y combinator satisfies Y g = g (Y g) by expanding one step.

def exercise_1():
    """Lambda calculus exercises."""

    print("Part 1: Evaluate (lambda x. lambda y. x) a b")
    print("=" * 60)
    print()
    print("  Expression: (lambda x. lambda y. x) a b")
    print()
    print("  Step 1: Apply (lambda x. lambda y. x) to a")
    print("    (lambda x. lambda y. x) a")
    print("    = (lambda y. x)[x := a]     -- substitute a for x")
    print("    = lambda y. a")
    print()
    print("  Step 2: Apply (lambda y. a) to b")
    print("    (lambda y. a) b")
    print("    = a[y := b]                  -- substitute b for y")
    print("    = a                           -- y doesn't appear in body, so no change")
    print()
    print("  Result: a")
    print()
    print("  This is the K combinator (also called TRUE in Church encoding):")
    print("  K = lambda x. lambda y. x")
    print("  It always returns its first argument and discards the second.")

    # Python verification
    K = lambda x: lambda y: x
    result = K("a")("b")
    print(f"\n  Python verification: K('a')('b') = '{result}'")

    print()
    print()
    print("Part 2: Church encoding of PLUS (addition)")
    print("=" * 60)
    print()
    print("  Church numerals:")
    print("    0 = lambda f. lambda x. x")
    print("    1 = lambda f. lambda x. f x")
    print("    2 = lambda f. lambda x. f (f x)")
    print("    n = lambda f. lambda x. f^n x")
    print()
    print("  PLUS = lambda m. lambda n. lambda f. lambda x. m f (n f x)")
    print()
    print("  Intuition:")
    print("    n f x applies f n times to x: f^n(x)")
    print("    m f (n f x) applies f m more times to f^n(x): f^(m+n)(x)")
    print("    This is the Church numeral for m + n.")
    print()
    print("  Derivation example: PLUS 2 3")
    print("    PLUS 2 3")
    print("    = (lambda m. lambda n. lambda f. lambda x. m f (n f x)) 2 3")
    print("    = lambda f. lambda x. 2 f (3 f x)")
    print("    = lambda f. lambda x. 2 f (f (f (f x)))    -- 3 f x = f^3(x)")
    print("    = lambda f. lambda x. f (f (f (f (f x))))  -- 2 f applies f twice more")
    print("    = 5  (Church numeral for 5)")

    # Python verification
    zero = lambda f: lambda x: x
    one = lambda f: lambda x: f(x)
    two = lambda f: lambda x: f(f(x))
    three = lambda f: lambda x: f(f(f(x)))

    PLUS = lambda m: lambda n: lambda f: lambda x: m(f)(n(f)(x))

    # Convert Church numeral to int by applying (lambda x: x + 1) and starting at 0
    def church_to_int(n):
        return n(lambda x: x + 1)(0)

    print(f"\n  Python verification:")
    print(f"    church_to_int(0) = {church_to_int(zero)}")
    print(f"    church_to_int(1) = {church_to_int(one)}")
    print(f"    church_to_int(2) = {church_to_int(two)}")
    print(f"    church_to_int(3) = {church_to_int(three)}")
    print(f"    church_to_int(PLUS 2 3) = {church_to_int(PLUS(two)(three))}")
    print(f"    church_to_int(PLUS 0 3) = {church_to_int(PLUS(zero)(three))}")
    print(f"    church_to_int(PLUS 1 1) = {church_to_int(PLUS(one)(one))}")

    # Also define SUCC and MULT for completeness
    SUCC = lambda n: lambda f: lambda x: f(n(f)(x))
    MULT = lambda m: lambda n: lambda f: m(n(f))

    print(f"    church_to_int(SUCC 2) = {church_to_int(SUCC(two))}")
    print(f"    church_to_int(MULT 2 3) = {church_to_int(MULT(two)(three))}")

    print()
    print()
    print("Part 3: Y combinator satisfies Y g = g (Y g)")
    print("=" * 60)
    print()
    print("  Y = lambda f. (lambda x. f (x x)) (lambda x. f (x x))")
    print()
    print("  Expand Y g:")
    print("    Y g")
    print("    = (lambda f. (lambda x. f (x x)) (lambda x. f (x x))) g")
    print("    = (lambda x. g (x x)) (lambda x. g (x x))")
    print("      [substituted g for f]")
    print()
    print("  Let A = (lambda x. g (x x)). Then Y g = A A.")
    print()
    print("  Expand A A:")
    print("    A A = (lambda x. g (x x)) A")
    print("         = g (A A)")
    print("         = g (Y g)")
    print("      [since A A = Y g from above]")
    print()
    print("  Therefore: Y g = A A = g (A A) = g (Y g). QED.")
    print()
    print("  This means Y g is a fixed point of g: Y g = g (Y g).")
    print("  This enables recursion without named self-reference!")
    print()
    print("  Example: Factorial using Y combinator")
    print("    fact_step = lambda f. lambda n. 1 if n == 0 else n * f(n-1)")
    print("    factorial = Y(fact_step)  -- in a lazy language")

    # Python demonstration (using a strict-language Y combinator variant)
    # The standard Y combinator causes infinite recursion in strict languages.
    # Use the Z combinator (strict Y) instead:
    Z = lambda f: (lambda x: f(lambda v: x(x)(v)))(lambda x: f(lambda v: x(x)(v)))

    fact_step = lambda f: lambda n: 1 if n == 0 else n * f(n - 1)
    factorial = Z(fact_step)

    print(f"\n  Python verification (using Z combinator for strict evaluation):")
    for n in range(8):
        print(f"    factorial({n}) = {factorial(n)}")


# === Exercise 2: Applications ===
# Problem:
# 1. Explain why RNA secondary structure prediction is naturally modeled with CFGs.
# 2. A regex engine uses backtracking on pattern (a+)+b. Explain why input "aaa...a"
#    (no 'b') causes exponential time.
# 3. Describe how model checking uses the emptiness problem for automata.

def exercise_2():
    """Applications of formal language theory."""

    print("Part 1: RNA secondary structure and CFGs")
    print("=" * 60)
    print()
    print("  RNA is a single-stranded molecule made of nucleotides (A, U, G, C).")
    print("  It folds into a secondary structure by forming base pairs:")
    print("    A-U (adenine-uracil)")
    print("    G-C (guanine-cytosine)")
    print("    G-U (wobble pair)")
    print()
    print("  Base pairs create NESTED structures, similar to matched parentheses:")
    print("    ...(((...)))...  -- a stem-loop")
    print("    ...((..((..))..))..  -- nested stems")
    print()
    print("  Why CFGs are the right model:")
    print("  1. NESTING: Base pairs are nested (no crossing in secondary structure).")
    print("     Position i pairs with position j, and any pair (k, l) inside must")
    print("     satisfy i < k < l < j. This is exactly the structure of matched")
    print("     parentheses, which is context-free: S -> aSb | SS | epsilon.")
    print()
    print("  2. Grammar for RNA secondary structure:")
    print("     S -> aS'u | uS'a | gS'c | cS'g | gS'u | uS'g  (base pairs)")
    print("     S -> SS         (concatenation of structures)")
    print("     S -> xS         (unpaired base, x in {a,u,g,c})")
    print("     S -> epsilon    (empty)")
    print("     S' represents the content between paired bases.")
    print()
    print("  3. Stochastic CFGs (SCFGs):")
    print("     Assign probabilities to each rule based on biological data.")
    print("     Use CYK-like dynamic programming (Nussinov/Zuker algorithms)")
    print("     to find the most probable structure in O(n^3) time.")
    print()
    print("  4. Why not regular?")
    print("     Regular languages cannot express nesting. A finite automaton")
    print("     cannot track matching base pairs at arbitrary nesting depth.")

    print()
    print()
    print("Part 2: Exponential backtracking in regex (a+)+b")
    print("=" * 60)
    print()
    print("  Pattern: (a+)+b")
    print("  Input: 'aaaa...a' (n a's, no b)")
    print()
    print("  How backtracking regex engines work:")
    print("  The engine tries to match the pattern against the input.")
    print("  (a+) matches one or more a's, and the outer + means one or more")
    print("  groups of a's. The engine must try ALL ways to partition the")
    print("  a's into groups.")
    print()
    print("  For input 'aaaa' (4 a's):")
    print("    Try: (aaaa) then need b -> fail")
    print("    Try: (aaa)(a) then need b -> fail")
    print("    Try: (aa)(aa) then need b -> fail")
    print("    Try: (aa)(a)(a) then need b -> fail")
    print("    Try: (a)(aaa) then need b -> fail")
    print("    Try: (a)(aa)(a) then need b -> fail")
    print("    Try: (a)(a)(aa) then need b -> fail")
    print("    Try: (a)(a)(a)(a) then need b -> fail")
    print()
    print("  The number of ways to partition n items into ordered groups")
    print("  is 2^(n-1) (each of the n-1 gaps between a's is either a")
    print("  group boundary or not). The engine tries ALL partitions.")
    print()
    print("  Time complexity: O(2^n) -- exponential!")
    print()
    print("  This is called 'catastrophic backtracking' or 'ReDoS'")
    print("  (Regular Expression Denial of Service).")
    print()
    print("  The fix: Use a regex engine based on NFA simulation (Thompson's")
    print("  algorithm), which runs in O(nm) time for pattern size m and")
    print("  input size n. Alternatively, rewrite the pattern: 'a+b' suffices.")

    # Demonstration of the exponential blowup
    print("\n  Partition count demonstration:")
    for n in range(1, 16):
        partitions = 2 ** (n - 1)
        print(f"    n={n:2d} a's: {partitions:6d} partitions to try")

    print()
    print()
    print("Part 3: Model checking and automata emptiness")
    print("=" * 60)
    print()
    print("  Model checking verifies that a system satisfies a specification.")
    print()
    print("  1. System Model:")
    print("     The system is modeled as a finite automaton (Kripke structure)")
    print("     where states represent system configurations and transitions")
    print("     represent possible state changes. The language L(System) is the")
    print("     set of all possible execution traces (behaviors).")
    print()
    print("  2. Specification:")
    print("     The desired property is expressed in temporal logic (LTL, CTL).")
    print("     For LTL: the specification phi is converted to a Buchi automaton")
    print("     A_phi that accepts exactly the traces satisfying phi.")
    print("     The NEGATION of the specification, not-phi, gives automaton A_{not-phi}")
    print("     accepting traces that VIOLATE the specification.")
    print()
    print("  3. Verification via Emptiness:")
    print("     The system violates the spec iff there exists a trace that is")
    print("     both a valid system behavior AND violates the specification:")
    print("       L(System) ∩ L(A_{not-phi}) != empty")
    print()
    print("     Algorithm:")
    print("     a. Construct the product automaton: System x A_{not-phi}")
    print("     b. Check if L(product) is empty (reachability of accepting states)")
    print("     c. If EMPTY: system satisfies spec (no violating trace exists)")
    print("     d. If NON-EMPTY: system violates spec, and the accepting trace")
    print("        is a COUNTEREXAMPLE showing the violation")
    print()
    print("  4. Why automata theory is essential:")
    print("     - Closure under intersection enables product construction")
    print("     - Complementation enables negating the specification")
    print("     - Emptiness checking is decidable in polynomial time")
    print("     - The counterexample trace aids debugging")
    print()
    print("  5. Buchi automata (omega-automata):")
    print("     For non-terminating systems (OS, protocols), we need automata")
    print("     that accept infinite strings. Buchi automata accept an infinite")
    print("     word if some accepting state is visited infinitely often.")
    print("     All the above operations extend to Buchi automata.")


# === Exercise 3: Comprehensive Review ===
# Problem: For each claim, state true or false with justification:
# 1. Every context-free language is decidable.
# 2. If L is recognizable and complement(L) is recognizable, then L is decidable.
# 3. There exists a language that is decidable but not context-sensitive.
# 4. P = NP would imply that factoring integers is in P.
# 5. The halting problem is NP-hard.

def exercise_3():
    """Comprehensive review of formal language theory."""

    claims = [
        {
            "claim": "Every context-free language is decidable.",
            "answer": "TRUE",
            "justification": (
                "The CYK algorithm decides membership for any CFL in O(n^3) time.\n"
                "  Given CFG G and string w:\n"
                "    1. Convert G to Chomsky Normal Form (always possible).\n"
                "    2. Run CYK dynamic programming algorithm.\n"
                "    3. CYK always halts and correctly decides w in L(G).\n"
                "  Since every CFL has a CFG, every CFL is decidable.\n"
                "  In the Chomsky hierarchy: Regular subset CF subset Decidable subset RE."
            ),
        },
        {
            "claim": "If L is recognizable and complement(L) is recognizable, then L is decidable.",
            "answer": "TRUE",
            "justification": (
                "This is a fundamental theorem of computability theory.\n"
                "  Proof: Let M1 recognize L and M2 recognize complement(L).\n"
                "  Construct decider D:\n"
                "    D on input w:\n"
                "      Run M1 and M2 in parallel (alternating steps).\n"
                "      If M1 accepts, accept (w in L).\n"
                "      If M2 accepts, reject (w in complement(L), so w not in L).\n"
                "  D always halts because for every w, either w in L (M1 accepts)\n"
                "  or w in complement(L) (M2 accepts). One must accept.\n"
                "  Therefore L is decidable."
            ),
        },
        {
            "claim": "There exists a language that is decidable but not context-sensitive.",
            "answer": "TRUE",
            "justification": (
                "Context-sensitive languages = NSPACE(n) = languages decided by\n"
                "  linear-bounded automata using O(n) tape cells.\n"
                "  There exist decidable languages requiring more than linear space.\n"
                "  Example: By the space hierarchy theorem, NSPACE(n^2) strictly\n"
                "  contains NSPACE(n). Any language in NSPACE(n^2) - NSPACE(n) is\n"
                "  decidable (it has a decider using n^2 space) but not context-sensitive.\n"
                "  Concrete example: The language of TM encodings <M> such that M\n"
                "  accepts within n^2 space is decidable but not in NSPACE(n)."
            ),
        },
        {
            "claim": "P = NP would imply that factoring integers is in P.",
            "answer": "TRUE",
            "justification": (
                "The factoring DECISION problem is: given integers n and k, does n\n"
                "  have a factor d with 1 < d < k?\n"
                "  This problem is in NP: the certificate is the factor d, and\n"
                "  verification is just checking d divides n and 1 < d < k.\n"
                "  If P = NP, then every NP problem is in P, including factoring.\n"
                "  Note: Factoring is not known to be NP-complete. If P != NP,\n"
                "  factoring might be in P, in NP-intermediate (Ladner's theorem\n"
                "  guarantees such problems exist if P != NP), or nowhere obvious.\n"
                "  But IF P = NP, factoring is definitely in P."
            ),
        },
        {
            "claim": "The halting problem is NP-hard.",
            "answer": "TRUE",
            "justification": (
                "The halting problem HALT_TM is NP-hard (and much harder).\n"
                "  Proof: Any NP problem L has a polynomial-time NTM M_L deciding it.\n"
                "  Given input w for L:\n"
                "    1. Convert to a deterministic simulation M'_L that systematically\n"
                "       tries all nondeterministic branches of M_L on w.\n"
                "    2. M'_L halts iff M_L has at least one accepting branch iff w in L.\n"
                "  So L <=_P HALT_TM via the computable function w -> <M'_L, w>.\n"
                "  Since every NP problem reduces to HALT_TM, it is NP-hard.\n"
                "  In fact, HALT_TM is undecidable, so it is 'harder' than any\n"
                "  decidable problem, including all NP-complete problems."
            ),
        },
    ]

    for i, item in enumerate(claims, 1):
        print(f"Claim {i}: {item['claim']}")
        print(f"  Answer: {item['answer']}")
        print(f"  Justification: {item['justification']}")
        print()


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Lambda Calculus ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Applications ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Comprehensive Review ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
