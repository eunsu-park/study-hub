"""
Exercises for Lesson 05: Properties of Regular Languages
Topic: Formal_Languages

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Pumping Lemma Proofs ===
# Problem: Prove that the following languages are not regular:
# 1. L = {a^n b^{2n} | n >= 0}
# 2. L = {w in {0,1}* | w has equal numbers of 0s and 1s}
# 3. L = {a^p | p is prime}

def exercise_1():
    """Pumping lemma proofs for three non-regular languages."""

    print("Part 1: L = {a^n b^(2n) | n >= 0} is NOT regular")
    print("=" * 55)
    print("  Proof by contradiction using the pumping lemma:")
    print("  Assume L is regular with pumping length p.")
    print("  Choose w = a^p b^(2p) in L (|w| = 3p >= p).")
    print("  By the pumping lemma, w = xyz with |y| >= 1, |xy| <= p.")
    print("  Since |xy| <= p, both x and y consist entirely of a's.")
    print("  Let y = a^k where k >= 1.")
    print("  Pump down (i=0): xy^0z = xz = a^(p-k) b^(2p).")
    print("  For this to be in L, we need 2(p-k) = 2p, i.e., k = 0.")
    print("  But k >= 1. Contradiction. QED.")
    print()

    # Demonstration: Show the pumping failure computationally
    def check_L1(s):
        """Check if s is in {a^n b^(2n)}."""
        n_a = 0
        i = 0
        while i < len(s) and s[i] == 'a':
            n_a += 1
            i += 1
        n_b = 0
        while i < len(s) and s[i] == 'b':
            n_b += 1
            i += 1
        return i == len(s) and n_b == 2 * n_a

    p = 4  # Example pumping length
    w = "a" * p + "b" * (2 * p)
    print(f"  Demonstration with p={p}, w = a^{p}b^{2*p} = '{w}'")
    # Try all valid splits
    for j in range(1, p + 1):  # |xy| <= p, |y| >= 1
        for i in range(j):  # x = w[:i], y = w[i:j]
            x = w[:i]
            y = w[i:j]
            z = w[j:]
            if len(y) >= 1 and len(x) + len(y) <= p:
                pumped = x + z  # xy^0z
                in_L = check_L1(pumped)
                if not in_L:
                    print(f"    x='{x}', y='{y}', z='{z[:10]}...': xy^0z NOT in L (len={len(pumped)})")
                    break
        else:
            continue
        break

    print()
    print("Part 2: L = {w in {0,1}* | #0s = #1s} is NOT regular")
    print("=" * 55)
    print("  Proof by contradiction:")
    print("  Assume L is regular with pumping length p.")
    print("  Choose w = 0^p 1^p in L (equal 0s and 1s, |w| = 2p >= p).")
    print("  By the pumping lemma, w = xyz with |y| >= 1, |xy| <= p.")
    print("  Since |xy| <= p, y consists entirely of 0s: y = 0^k, k >= 1.")
    print("  Pump up (i=2): xy^2z = 0^(p+k) 1^p.")
    print("  This has p+k zeros but only p ones. Since k >= 1, #0s != #1s.")
    print("  So xy^2z not in L. Contradiction. QED.")

    print()
    print("Part 3: L = {a^p | p is prime} is NOT regular")
    print("=" * 55)
    print("  Proof by contradiction:")
    print("  Assume L is regular with pumping length p (the PL constant).")
    print("  Let q be a prime >= p. Choose w = a^q in L.")
    print("  By the pumping lemma, w = xyz with |y| = k >= 1, |xy| <= p.")
    print("  Consider xy^{q+1}z. Its length is q + q*k = q(1+k).")
    print("  Since k >= 1, we have 1+k >= 2 and q >= 2,")
    print("  so q(1+k) is composite (product of two integers >= 2).")
    print("  Therefore |xy^{q+1}z| is not prime, so xy^{q+1}z not in L.")
    print("  Contradiction. QED.")
    print()

    # Demonstration
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    p_const = 5
    q_prime = 7  # A prime >= p_const
    print(f"  Demonstration: p={p_const}, chosen prime q={q_prime}")
    print(f"  w = a^{q_prime}, |w| = {q_prime}")
    for k in range(1, p_const + 1):
        pumped_len = q_prime + q_prime * k  # |xy^{q+1}z| = q + q*k = q(1+k)
        print(f"    |y|={k}: |xy^{{q+1}}z| = {q_prime}(1+{k}) = {pumped_len}, prime? {is_prime(pumped_len)}")


# === Exercise 2: Closure Properties ===
# Problem:
# 1. L = {w in {a,b}* | |w|_a = |w|_b} is not regular,
#    using the fact that L intersect a*b* = {a^n b^n}.
# 2. If L is regular, then HALF(L) = {x | exists y, |x|=|y|, xy in L} is regular.

def exercise_2():
    """Closure property proofs."""

    print("Part 1: L = {w | #a = #b} is NOT regular (closure argument)")
    print("=" * 55)
    print("  Proof:")
    print("  Assume L = {w in {a,b}* | #a(w) = #b(w)} is regular.")
    print("  The language R = a*b* is regular (recognized by a simple DFA).")
    print("  Since regular languages are closed under intersection:")
    print("    L ∩ R = {w in a*b* | #a(w) = #b(w)} = {a^n b^n | n >= 0}")
    print("  would be regular.")
    print("  But {a^n b^n} is known to be non-regular (pumping lemma).")
    print("  Contradiction. So L is not regular. QED.")

    # Demonstration
    import re
    print("\n  Demonstration: L ∩ a*b* = {a^n b^n}")
    for n in range(6):
        w = "a" * n + "b" * n
        in_equal = w.count("a") == w.count("b")
        in_astarb = bool(re.match(r'^a*b*$', w))
        in_anbn = in_equal and in_astarb
        print(f"    '{w}': #a=#b? {in_equal}, in a*b*? {in_astarb}, in a^nb^n? {in_anbn}")

    print()
    print("Part 2: If L is regular, then HALF(L) is regular")
    print("=" * 55)
    print("  Proof sketch:")
    print("  Let M = (Q, Sigma, delta, q0, F) be a DFA for L.")
    print("  Construct a DFA M' for HALF(L):")
    print("  Key idea: x is in HALF(L) iff there exists y with |y|=|x| and xy in L.")
    print("  After reading x, we need to know which states are reachable")
    print("  by reading |x| more symbols from any starting state.")
    print()
    print("  Define: S_k = set of states q such that delta_hat(q, w) in F")
    print("  for some w of length k. (States from which an accepting state")
    print("  is reachable in exactly k steps.)")
    print()
    print("  M' tracks pairs (current_state, states_reachable_in_remaining_steps).")
    print("  This is a standard construction -- the new DFA has states Q x P(Q),")
    print("  tracking both the state reached by x and the set of states from which")
    print("  an accept state is reachable in exactly |x| more steps.")
    print("  After reading x: accept iff current state is in the tracked set.")
    print()
    print("  Since Q is finite, P(Q) is finite, so M' is a DFA. QED.")

    # Concrete example
    print("\n  Concrete example: L = {w in {0,1}* | |w| is even}")
    print("  HALF(L) = {x | exists y, |x|=|y|, xy in L}")
    print("  Since |xy| = 2|x| is always even, HALF(L) = Sigma*")
    print("  (Every string x has a matching y of the same length making |xy| even)")


# === Exercise 3: Myhill-Nerode ===
# Problem: Determine the Myhill-Nerode equivalence classes for
# L = {w in {a,b}* | #a(w) is even}. What is the minimum DFA size?

def exercise_3():
    """Myhill-Nerode equivalence classes for L = {w | #a is even}."""

    print("L = {w in {a,b}* | number of a's in w is even}")
    print("=" * 55)
    print()
    print("Myhill-Nerode equivalence relation:")
    print("  x ≡_L y iff for all z: xz in L <=> yz in L")
    print()
    print("Analysis:")
    print("  The membership of xz in L depends only on the parity of #a(xz).")
    print("  #a(xz) = #a(x) + #a(z)")
    print("  So xz in L iff #a(x) + #a(z) is even.")
    print()
    print("  Two strings x, y are L-equivalent iff:")
    print("    for all z: #a(x)+#a(z) is even <=> #a(y)+#a(z) is even")
    print("  This holds iff #a(x) and #a(y) have the same parity.")
    print()
    print("Equivalence classes:")
    print("  [epsilon] = {w | #a(w) is even} = {epsilon, b, bb, aa, ab, ba, ...}")
    print("  [a]       = {w | #a(w) is odd}  = {a, aab, aba, bab, ...}")
    print()
    print("Number of equivalence classes: 2")
    print("Minimum DFA size: 2 states")
    print()

    # Build and verify minimum DFA
    print("Minimum DFA:")
    print("  States: {q_even, q_odd}")
    print("  Start: q_even")
    print("  Accept: {q_even}")
    print("  Transitions:")
    print("    delta(q_even, a) = q_odd")
    print("    delta(q_even, b) = q_even")
    print("    delta(q_odd, a) = q_even")
    print("    delta(q_odd, b) = q_odd")

    # Verify
    print("\n  Verification:")
    test_cases = [
        "",       # 0 a's (even) -> accept
        "a",      # 1 a (odd) -> reject
        "b",      # 0 a's (even) -> accept
        "aa",     # 2 a's (even) -> accept
        "ab",     # 1 a (odd) -> reject
        "ba",     # 1 a (odd) -> reject
        "bb",     # 0 a's (even) -> accept
        "aab",    # 2 a's (even) -> accept
        "aba",    # 2 a's (even) -> accept
        "bab",    # 1 a (odd) -> reject
        "abba",   # 2 a's (even) -> accept
    ]

    for w in test_cases:
        count_a = w.count("a")
        in_L = count_a % 2 == 0
        eq_class = "[epsilon]" if count_a % 2 == 0 else "[a]"
        label = "epsilon" if w == "" else w
        print(
            f"    '{label}': #a={count_a}, class={eq_class}, "
            f"{'ACCEPT' if in_L else 'reject'}"
        )

    # Show distinguishability
    print("\n  Distinguishability proof:")
    print("    Take x = epsilon (#a=0, even) and y = a (#a=1, odd)")
    print("    Distinguishing suffix z = epsilon:")
    print(f"      x.z = '' in L? {0 % 2 == 0} (0 a's)")
    print(f"      y.z = 'a' in L? {1 % 2 == 0} (1 a)")
    print("    Since exactly one is in L, epsilon and 'a' are L-distinguishable.")
    print("    These are the only two classes, so 2 = minimum DFA size.")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Pumping Lemma Proofs ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Closure Properties ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Myhill-Nerode ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
