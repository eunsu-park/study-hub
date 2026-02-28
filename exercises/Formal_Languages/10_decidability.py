"""
Exercises for Lesson 10: Decidability
Topic: Formal_Languages

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Decidability Proofs ===
# Problem: Prove that the following languages are decidable:
# 1. {<M> | M is a DFA that accepts some string of length <= 5}
# 2. {<G> | G is a CFG and L(G) is infinite}
# 3. {<M> | M is a DFA that accepts at least one palindrome}

def exercise_1():
    """Decidability proofs for three languages."""

    print("Part 1: {<M> | M is a DFA that accepts some string of length <= 5}")
    print("=" * 60)
    print("  Claim: This language is DECIDABLE.")
    print()
    print("  Proof: Construct a decider D:")
    print("  D on input <M> (encoding of a DFA):")
    print("    1. Verify <M> is a valid DFA encoding. If not, reject.")
    print("    2. Enumerate all strings of length 0, 1, 2, 3, 4, 5")
    print("       over M's input alphabet Sigma.")
    print("       There are at most |Sigma|^0 + |Sigma|^1 + ... + |Sigma|^5")
    print("       = O(|Sigma|^5) such strings (finite).")
    print("    3. For each string w, simulate M on w.")
    print("       DFA simulation always halts in |w| steps.")
    print("    4. If M accepts any such w, accept. Otherwise, reject.")
    print()
    print("  D always halts because:")
    print("    - There are finitely many strings to test")
    print("    - DFA simulation always terminates")
    print("  Therefore the language is decidable. QED.")

    # Demonstration
    print("\n  Demonstration:")
    # Simple DFA over {0,1} that accepts strings ending in '1'
    from itertools import product as iproduct
    alphabet = ['0', '1']
    dfa_transitions = {
        ('q0', '0'): 'q0', ('q0', '1'): 'q1',
        ('q1', '0'): 'q0', ('q1', '1'): 'q1',
    }
    dfa_start = 'q0'
    dfa_accept = {'q1'}

    def simulate_dfa(w):
        state = dfa_start
        for ch in w:
            state = dfa_transitions.get((state, ch), None)
            if state is None:
                return False
        return state in dfa_accept

    found = False
    for length in range(6):
        for combo in iproduct(alphabet, repeat=length):
            w = "".join(combo)
            if simulate_dfa(w):
                label = "eps" if w == "" else w
                print(f"    DFA accepts '{label}' (length {length}) -> ACCEPT")
                found = True
                break
        if found:
            break
    if not found:
        print("    No string of length <= 5 accepted -> REJECT")

    print()
    print()
    print("Part 2: {<G> | G is a CFG and L(G) is infinite}")
    print("=" * 60)
    print("  Claim: This language is DECIDABLE.")
    print()
    print("  Proof: Construct a decider D:")
    print("  D on input <G> (encoding of a CFG):")
    print("    1. Verify <G> is a valid CFG encoding. If not, reject.")
    print("    2. Convert G to Chomsky Normal Form (CNF).")
    print("    3. Eliminate useless symbols.")
    print("    4. Build the 'dependency graph' of useful variables:")
    print("       Add edge A -> B if there is a rule A -> BC or A -> CB.")
    print("    5. Check if this graph contains a cycle.")
    print("       - If YES: L(G) is infinite. Accept.")
    print("       - If NO: L(G) is finite. Reject.")
    print()
    print("  Why this works:")
    print("    In CNF, each derivation step either produces two variables")
    print("    (A -> BC) or a terminal (A -> a).")
    print("    If a useful variable A can derive a string containing A")
    print("    (cycle in the graph), then A can be expanded indefinitely,")
    print("    producing arbitrarily long strings -> L(G) is infinite.")
    print("    If no such cycle exists, every derivation tree has bounded")
    print("    depth -> L(G) is finite.")
    print("  All steps are effective, so D is a decider. QED.")

    # Demonstration
    print("\n  Demonstration:")
    print("    G1: S -> AB, A -> a, B -> b")
    print("      No cycles among {S, A, B} -> L(G1) = {ab} is FINITE -> reject")
    print("    G2: S -> AB, A -> aA | a, B -> b")
    print("      Cycle: A -> aA (A depends on A) -> L(G2) is INFINITE -> accept")

    print()
    print()
    print("Part 3: {<M> | M is a DFA that accepts at least one palindrome}")
    print("=" * 60)
    print("  Claim: This language is DECIDABLE.")
    print()
    print("  Proof: Construct a decider D:")
    print("  D on input <M> (encoding of a DFA):")
    print("    1. Verify <M> is a valid DFA encoding. If not, reject.")
    print("    2. Construct a DFA P that accepts all palindromes over Sigma.")
    print("       Wait -- the set of all palindromes is NOT regular!")
    print()
    print("  Revised approach:")
    print("    Key insight: The set of palindromes is not regular, but we can")
    print("    still decide the problem because:")
    print()
    print("    If M has n states, then if M accepts ANY palindrome, it accepts")
    print("    one of length at most 2n (by a pumping-style argument on the")
    print("    palindrome structure).")
    print()
    print("    Actually, a cleaner argument:")
    print("    The language of palindromes P is context-free (S -> aSa | bSb | a | b | eps).")
    print("    L(M) is regular. L(M) intersect P is context-free")
    print("    (CFL intersect regular = CFL). Emptiness of CFLs is decidable.")
    print()
    print("  Algorithm:")
    print("    1. Construct CFG G_P for palindromes over M's alphabet.")
    print("    2. Construct the PDA for L(M) ∩ L(G_P) using the product")
    print("       construction (PDA x DFA).")
    print("    3. Convert the product PDA to a CFG.")
    print("    4. Check if L(CFG) is empty.")
    print("    5. If non-empty: accept. If empty: reject.")
    print()
    print("  All steps are effective, so D is a decider. QED.")

    # Demonstration
    print("\n  Demonstration:")
    print("    DFA M accepts {w | w ends in '11'} over {0,1}")
    print("    Palindromes ending in 11: '11', '0110', '10110101', ...")
    print("    '11' is a palindrome accepted by M -> ACCEPT")

    dfa_11 = {('q0', '0'): 'q0', ('q0', '1'): 'q1',
              ('q1', '0'): 'q0', ('q1', '1'): 'q2',
              ('q2', '0'): 'q0', ('q2', '1'): 'q2'}
    dfa_11_accept = {'q2'}
    dfa_11_start = 'q0'

    def sim_dfa_11(w):
        state = dfa_11_start
        for ch in w:
            state = dfa_11.get((state, ch))
        return state in dfa_11_accept

    def is_palindrome(w):
        return w == w[::-1]

    # Check palindromes up to length 6
    found_pal = False
    for length in range(7):
        for combo in iproduct(['0', '1'], repeat=length):
            w = "".join(combo)
            if is_palindrome(w) and sim_dfa_11(w):
                print(f"    Found palindrome accepted by M: '{w}'")
                found_pal = True
                break
        if found_pal:
            break


# === Exercise 2: Undecidability ===
# Problem: Show that L = {<M> | M is a TM that accepts epsilon} is undecidable.
# (Hint: reduce from A_TM.)

def exercise_2():
    """Undecidability proof for {<M> | M accepts epsilon}."""

    print("L = {<M> | M is a TM that accepts epsilon} is UNDECIDABLE")
    print("=" * 60)
    print()
    print("Proof by reduction from A_TM:")
    print()
    print("  Recall: A_TM = {<M, w> | M is a TM that accepts w}")
    print("  A_TM is known to be undecidable.")
    print()
    print("  We show: A_TM <=_m L (mapping reduction)")
    print()
    print("  Reduction function f:")
    print("  Given <M, w>, construct TM M' as follows:")
    print()
    print("    M' on input x:")
    print("      1. Ignore x (don't even read it)")
    print("      2. Simulate M on w")
    print("      3. If M accepts w, accept")
    print("      4. If M rejects w, reject")
    print()
    print("  Key observations:")
    print("    - M' ignores its input entirely")
    print("    - M' accepts epsilon iff M accepts w")
    print("    - M' actually accepts EVERYTHING or NOTHING:")
    print("      * If M accepts w: M' accepts all inputs, including epsilon")
    print("      * If M rejects w: M' rejects all inputs, including epsilon")
    print("      * If M loops on w: M' loops on all inputs")
    print()
    print("  Correctness of reduction:")
    print("    <M, w> in A_TM")
    print("      iff M accepts w")
    print("      iff M' accepts epsilon")
    print("      iff <M'> in L")
    print()
    print("    <M, w> not in A_TM")
    print("      iff M does not accept w (rejects or loops)")
    print("      iff M' does not accept epsilon")
    print("      iff <M'> not in L")
    print()
    print("  The function f(<M,w>) = <M'> is computable because:")
    print("    - We can mechanically construct the description of M'")
    print("      from the descriptions of M and w")
    print("    - This is a simple transformation of TM descriptions")
    print()
    print("  Therefore A_TM <=_m L, and since A_TM is undecidable,")
    print("  L must also be undecidable. QED.")
    print()
    print("  Note: L IS Turing-recognizable, because a universal TM can")
    print("  simulate M on epsilon and accept if M accepts. But it may")
    print("  loop if M loops on epsilon.")


# === Exercise 3: Recognizability ===
# Problem: Classify each language as decidable, recognizable (not decidable),
# or not recognizable:
# 1. {<M, w> | M halts on w within 1000 steps}
# 2. {<M> | M accepts some string}
# 3. {<M> | L(M) = Sigma*}

def exercise_3():
    """Classify languages by decidability/recognizability."""

    print("Part 1: {<M, w> | M halts on w within 1000 steps}")
    print("=" * 60)
    print("  Classification: DECIDABLE")
    print()
    print("  Proof: Construct a decider D:")
    print("  D on input <M, w>:")
    print("    1. Simulate M on w for exactly 1000 steps.")
    print("    2. If M halts within 1000 steps, accept.")
    print("    3. If M has not halted after 1000 steps, reject.")
    print()
    print("  D always halts (runs for at most 1000 simulation steps).")
    print("  D correctly decides the language.")
    print("  Therefore the language is decidable.")
    print()
    print("  Key insight: Bounding the number of steps makes the problem")
    print("  decidable. The halting problem is undecidable only when we")
    print("  ask about unbounded computation.")

    # Demonstration
    print("\n  Demonstration:")
    print("    Simulating a simple TM (counter) for up to 1000 steps:")
    steps = 0
    x = 1
    while steps < 1000:
        x = (x * 3 + 1) % 997
        steps += 1
        if x == 0:
            print(f"    Halted at step {steps} -> ACCEPT")
            break
    else:
        print(f"    Did not halt within 1000 steps -> REJECT")

    print()
    print()
    print("Part 2: {<M> | M accepts some string} = complement of E_TM")
    print("=" * 60)
    print("  Classification: RECOGNIZABLE but NOT DECIDABLE")
    print()
    print("  Recognizability:")
    print("    Construct a recognizer R:")
    print("    R on input <M>:")
    print("      1. For i = 1, 2, 3, ...:")
    print("         a. Enumerate all strings of length <= i over Sigma.")
    print("         b. For each string w, simulate M on w for i steps.")
    print("         c. If M accepts any w within i steps, accept.")
    print("    This is called 'dovetailing' -- it interleaves simulations")
    print("    of M on all possible inputs, giving each more time as i grows.")
    print("    If M accepts some string w in k steps, R will find it when")
    print("    i >= max(|w|, k).")
    print("    If M accepts nothing, R runs forever (never accepts).")
    print()
    print("  Not decidable:")
    print("    This is the complement of E_TM = {<M> | L(M) = empty}.")
    print("    E_TM is known to be undecidable (proven via reduction from A_TM).")
    print("    If this language were decidable, flipping the answer would")
    print("    decide E_TM. Contradiction.")

    print()
    print()
    print("Part 3: {<M> | L(M) = Sigma*} = ALL_TM")
    print("=" * 60)
    print("  Classification: NOT RECOGNIZABLE (and therefore not decidable)")
    print()
    print("  Not decidable:")
    print("    By Rice's theorem. 'L(M) = Sigma*' is a nontrivial semantic")
    print("    property of TMs (some TMs have this property, some don't).")
    print("    Therefore it is undecidable.")
    print()
    print("  Not recognizable:")
    print("    We show that the complement, {<M> | L(M) != Sigma*}, IS recognizable.")
    print("    A recognizer for the complement:")
    print("      On input <M>: dovetail simulation of M on all strings.")
    print("      If M rejects some string w (in finite time), accept.")
    print("    This recognizes the complement.")
    print()
    print("    Now, if ALL_TM = {<M> | L(M)=Sigma*} were also recognizable,")
    print("    then both it and its complement would be recognizable,")
    print("    which would make it decidable (by the theorem: L is decidable")
    print("    iff both L and complement(L) are recognizable).")
    print("    But we just showed it's undecidable. Contradiction.")
    print("    Therefore ALL_TM is not recognizable.")
    print()
    print("  Summary of the decidability landscape:")
    print("    +------------------------------------------+")
    print("    |          Not recognizable                 |")
    print("    |   (e.g., ALL_TM, complement of A_TM)    |")
    print("    +------------------------------------------+")
    print("    |  Recognizable but not decidable           |")
    print("    |   (e.g., A_TM, HALT_TM, complement E_TM)|")
    print("    +------------------------------------------+")
    print("    |          Decidable                        |")
    print("    |   (e.g., A_DFA, E_CFG, 1000-step halt)  |")
    print("    +------------------------------------------+")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Decidability Proofs ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Undecidability ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Recognizability ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
