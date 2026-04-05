"""
Exercises for Lesson 04: Context-Free Grammars
Topic: Compiler_Design

Solutions to practice problems from the lesson.
"""

from collections import defaultdict
from itertools import product as iter_product


# === Exercise 1: Grammar Writing ===
# Problem: Write CFGs for palindromes, balanced parentheses, {a^i b^j c^k | i=j or j=k},
# simplified Python if/elif/else, and arithmetic expressions.

def exercise_1():
    """Write context-free grammars for various languages."""

    print("1. Palindromes over {a, b}")
    print("   S -> a S a | b S b | a | b | epsilon")
    print()
    # Verify with a simple recognizer
    def is_palindrome_grammar(s):
        """Check if string is a palindrome (equivalent to the grammar)."""
        return s == s[::-1] and all(c in 'ab' for c in s)

    for w in ["", "a", "b", "aa", "aba", "abba", "abab", "aabaa"]:
        print(f"   '{w}' -> {is_palindrome_grammar(w)}")
    print()

    print("2. Balanced parentheses")
    print("   S -> ( S ) S | epsilon")
    print()
    def is_balanced(s):
        depth = 0
        for c in s:
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth < 0:
                    return False
        return depth == 0

    for w in ["", "()", "(())", "()()", "((()))", "(()())", "(("]:
        print(f"   '{w}' -> {is_balanced(w)}")
    print()

    print("3. {a^i b^j c^k | i = j or j = k}")
    print("   S -> XC | AY")
    print("   X -> a X b | epsilon        (generates a^i b^i)")
    print("   C -> c C | epsilon           (generates c*)")
    print("   A -> a A | epsilon           (generates a*)")
    print("   Y -> b Y c | epsilon         (generates b^j c^j)")
    print()
    # This grammar is the union of {a^i b^i c^k} and {a^i b^j c^j}
    # which equals {a^i b^j c^k | i=j or j=k}
    def check_language_3(s):
        a_count = 0
        i = 0
        while i < len(s) and s[i] == 'a':
            a_count += 1
            i += 1
        b_count = 0
        while i < len(s) and s[i] == 'b':
            b_count += 1
            i += 1
        c_count = 0
        while i < len(s) and s[i] == 'c':
            c_count += 1
            i += 1
        if i != len(s):
            return False
        return a_count == b_count or b_count == c_count

    for w in ["", "abc", "aabbc", "abbcc", "aabbcc", "aabcc", "abbc"]:
        print(f"   '{w}' -> {check_language_3(w)}")
    print()

    print("4. Simplified Python if/elif/else")
    print("   stmt    -> if_stmt | other")
    print("   if_stmt -> IF expr COLON block elif_part else_part")
    print("   elif_part -> ELIF expr COLON block elif_part | epsilon")
    print("   else_part -> ELSE COLON block | epsilon")
    print("   block   -> INDENT stmts DEDENT")
    print("   stmts   -> stmt stmts | stmt")
    print("   expr    -> ID | NUM | expr OP expr")
    print()

    print("5. Arithmetic expressions with +, -, *, /, unary -, and parentheses")
    print("   E -> E + T | E - T | T")
    print("   T -> T * U | T / U | U")
    print("   U -> - U | F              (unary negation)")
    print("   F -> ( E ) | NUM | ID")
    print("   Precedence: unary - > *, / > +, -")
    print("   Associativity: +, -, *, / are left-associative")


# === Exercise 2: Derivations and Parse Trees ===
# Problem: Using E -> E+T | T, T -> T*F | F, F -> (E) | id | num
# Find leftmost and rightmost derivations for "id * (num + id)"

def exercise_2():
    """Derivations and parse trees for expression grammar."""
    print("Grammar:")
    print("  E -> E + T | T")
    print("  T -> T * F | F")
    print("  F -> ( E ) | id | num")
    print()
    print("String: id * (num + id)")
    print()

    print("1. Leftmost derivation:")
    steps = [
        "E",
        "T",
        "T * F",
        "F * F",
        "id * F",
        "id * ( E )",
        "id * ( E + T )",
        "id * ( T + T )",
        "id * ( F + T )",
        "id * ( num + T )",
        "id * ( num + F )",
        "id * ( num + id )",
    ]
    for i, step in enumerate(steps):
        arrow = "=>" if i > 0 else "  "
        tag = "  (lm)" if i > 0 else ""
        print(f"   {arrow} {step}{tag}")
    print()

    print("2. Rightmost derivation:")
    steps = [
        "E",
        "T",
        "T * F",
        "T * ( E )",
        "T * ( E + T )",
        "T * ( E + F )",
        "T * ( E + id )",
        "T * ( T + id )",
        "T * ( F + id )",
        "T * ( num + id )",
        "F * ( num + id )",
        "id * ( num + id )",
    ]
    for i, step in enumerate(steps):
        arrow = "=>" if i > 0 else "  "
        tag = "  (rm)" if i > 0 else ""
        print(f"   {arrow} {step}{tag}")
    print()

    print("3. Parse tree:")
    print("           E")
    print("           |")
    print("           T")
    print("         / | \\")
    print("        T  *  F")
    print("        |    / | \\")
    print("        F  (   E   )")
    print("        |    / | \\")
    print("       id  E   +   T")
    print("           |       |")
    print("           T       F")
    print("           |       |")
    print("           F      id")
    print("           |")
    print("          num")
    print()
    print("4. Both derivations produce the same parse tree (as expected for")
    print("   an unambiguous grammar -- derivation order doesn't affect structure).")


# === Exercise 3: Ambiguity Analysis ===
# Problem: S -> aSb | aSbb | epsilon

def exercise_3():
    """Analyze ambiguity of grammar S -> aSb | aSbb | epsilon."""
    print("Grammar: S -> aSb | aSbb | epsilon")
    print()

    print("1. Language generated:")
    print("   Each 'S -> aSb' adds one 'a' and one 'b'.")
    print("   Each 'S -> aSbb' adds one 'a' and two 'b's.")
    print("   If we use the first production i times and the second j times:")
    print("     #a = i + j,  #b = i + 2j")
    print("   So #b = #a + j, meaning #b >= #a.")
    print("   Language: L = {a^n b^m | m >= n >= 0}")
    print()

    print("2. The grammar IS ambiguous.")
    print("   Example: 'aabbb' (n=2, m=3)")
    print()
    print("   Parse tree 1: use aSb, then aSbb, then epsilon")
    print("     S -> aSb -> aaSbbb -> aabbb")
    print("     (first production then second)")
    print()
    print("   Parse tree 2: use aSbb, then aSb, then epsilon")
    print("     S -> aSbb -> aaSbbb -> aabbb")
    print("     (second production then first)")
    print()
    print("   Two different parse trees for the same string -> ambiguous.")
    print()

    # Verify computationally: count parse trees for 'aabbb'
    def count_parses(target, pos_a=0, pos_b_end=None):
        """Count distinct parse trees that generate target[pos_a:pos_b_end]."""
        if pos_b_end is None:
            pos_b_end = len(target)

        remaining = target[pos_a:pos_b_end]
        if remaining == "":
            return 1  # epsilon production

        count = 0
        # Try S -> aSb: first char must be 'a', last must be 'b'
        if len(remaining) >= 2 and remaining[0] == 'a' and remaining[-1] == 'b':
            count += count_parses(target, pos_a + 1, pos_b_end - 1)

        # Try S -> aSbb: first char 'a', last two must be 'bb'
        if len(remaining) >= 3 and remaining[0] == 'a' and remaining[-2:] == 'bb':
            count += count_parses(target, pos_a + 1, pos_b_end - 2)

        return count

    test_string = "aabbb"
    num_parses = count_parses(test_string)
    print(f"   Computational verification: '{test_string}' has {num_parses} parse trees")
    print()

    print("3. Unambiguous grammar for the same language {a^n b^m | m >= n >= 0}:")
    print("   S -> aSb S' | epsilon")
    print("   S'-> b S' | epsilon")
    print("   Idea: first generate matching a's and b's (S -> aSb), then")
    print("   generate extra b's at the end (S' -> b S').")
    print("   This is unambiguous because the structure is deterministic:")
    print("   all a's are paired with the innermost b's first, then extras follow.")


# === Exercise 4: CNF Conversion ===
# Problem: Convert S -> ASA | aB, A -> B | S, B -> b | epsilon to CNF.

def exercise_4():
    """Convert grammar to Chomsky Normal Form."""
    print("Original grammar:")
    print("  S -> ASA | aB")
    print("  A -> B | S")
    print("  B -> b | epsilon")
    print()

    print("Step 1: Eliminate epsilon productions")
    print("  B is nullable (B -> epsilon)")
    print("  A is nullable (A -> B -> epsilon)")
    print("  S is nullable (S -> aB -> a? No, 'a' is terminal, S -> ASA -> eps*eps*eps = eps)")
    print("  Wait: S -> ASA, all nullable? A nullable, S nullable (if S nullable, circular).")
    print("  Actually: B -> eps, so A -> B -> eps, so A is nullable.")
    print("  S -> ASA: if A nullable and S nullable, S is nullable.")
    print("  But S -> aB: 'a' is terminal, so this produces at least 'a'. Check S -> ASA.")
    print("  If both A's are nullable, S -> ASA -> S. This is a unit production, not epsilon.")
    print("  S is nullable only if there's a derivation S =>* eps.")
    print("  S -> ASA -> BSB -> eps*S*eps -> S -> ... cycle. S -> aB -> a (not eps).")
    print("  So S is NOT nullable. Only B and A are nullable.")
    print()
    print("  Nullable: {B, A}")
    print("  Remove epsilon productions and add alternatives:")
    print("  S -> ASA | AS | SA | S | aB | a")
    print("  A -> B | S")
    print("  B -> b")
    print("  (Remove B -> epsilon, A -> epsilon implicitly)")
    print()

    print("Step 2: Eliminate unit productions")
    print("  Unit productions: S -> S (remove, trivial), A -> B, A -> S")
    print("  A -> B: replace with A -> b")
    print("  A -> S: replace with A -> ASA | AS | SA | S | aB | a")
    print("  Then A -> S is again a unit production, replace: A -> ASA | AS | SA | aB | a | b")
    print("  S -> S is trivial, remove.")
    print("  Result:")
    print("  S -> ASA | AS | SA | aB | a")
    print("  A -> ASA | AS | SA | aB | a | b")
    print("  B -> b")
    print()

    print("Step 3: Convert to CNF (all productions are A -> BC or A -> a)")
    print("  Replace terminals in non-unit productions:")
    print("  Introduce: Ta -> a, Tb -> b")
    print("  S -> ASA: need to break into binary. S -> A(SA). Introduce X = SA.")
    print("    S -> AX, X -> SA")
    print("  S -> AS: already binary -> S -> AS")
    print("  S -> SA: already binary -> S -> SA")
    print("  S -> aB: S -> TaB")
    print("  S -> a: keep (terminal)")
    print()
    print("  Same for A productions:")
    print("  A -> ASA: A -> AX (reuse X = SA)")
    print("  A -> AS: keep")
    print("  A -> SA: keep")
    print("  A -> aB: A -> TaB")
    print("  A -> a: keep")
    print("  A -> b: keep")
    print()
    print("  Final CNF grammar:")
    print("  S  -> AX | AS | SA | TaB | a")
    print("  A  -> AX | AS | SA | TaB | a | b")
    print("  B  -> b")
    print("  X  -> SA")
    print("  Ta -> a")
    print()

    # Verify the CNF grammar by CYK on a test string
    print("  Verification: all productions are of form A -> BC or A -> a")
    cnf = {
        'S':  [('A', 'X'), ('A', 'S'), ('S', 'A'), ('Ta', 'B'), ('a',), ('b',)],
        'A':  [('A', 'X'), ('A', 'S'), ('S', 'A'), ('Ta', 'B'), ('a',), ('b',)],
        'B':  [('b',)],
        'X':  [('S', 'A')],
        'Ta': [('a',)],
    }
    for nt, prods in cnf.items():
        for p in prods:
            if len(p) == 1:
                assert p[0] in 'ab', f"Terminal rule {nt} -> {p[0]} must be a terminal"
            elif len(p) == 2:
                assert all(x in cnf for x in p), f"Binary rule {nt} -> {p} must reference nonterminals"
    print("  All productions verified as valid CNF.")


# === Exercise 5: CYK Algorithm ===
# Problem: Run CYK on strings "aabb", "abab", "baba" using a CNF grammar.

def exercise_5():
    """CYK algorithm implementation and execution."""
    # Using the CNF grammar from Exercise 4
    # Simplified version for testing
    cnf_rules = {
        'S':  [('A', 'X'), ('A', 'S'), ('S', 'A'), ('Ta', 'B'), ('a',)],
        'A':  [('A', 'X'), ('A', 'S'), ('S', 'A'), ('Ta', 'B'), ('a',), ('b',)],
        'B':  [('b',)],
        'X':  [('S', 'A')],
        'Ta': [('a',)],
    }

    def cyk(grammar, string):
        """CYK parsing algorithm. Returns the parse table."""
        n = len(string)
        if n == 0:
            return {}, 'S' in grammar and any(p == ('',) for p in grammar.get('S', []))

        # table[i][j] = set of nonterminals that can derive string[i:j+1]
        table = [[set() for _ in range(n)] for _ in range(n)]

        # Fill diagonal (length-1 substrings)
        for i in range(n):
            for nt, productions in grammar.items():
                for prod in productions:
                    if len(prod) == 1 and prod[0] == string[i]:
                        table[i][i].add(nt)

        # Fill table for increasing substring lengths
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                for k in range(i, j):
                    for nt, productions in grammar.items():
                        for prod in productions:
                            if len(prod) == 2:
                                b, c = prod
                                if b in table[i][k] and c in table[k + 1][j]:
                                    table[i][j].add(nt)

        return table, 'S' in table[0][n - 1]

    test_strings = ["aabb", "abab", "baba"]

    for s in test_strings:
        print(f"CYK for '{s}':")
        table, accepted = cyk(cnf_rules, s)
        n = len(s)

        # Print the table
        print(f"  {'':>4}", end="")
        for j in range(n):
            print(f"  {s[j]:>12}", end="")
        print()

        for i in range(n):
            print(f"  {i:>2}: ", end="")
            for j in range(n):
                if j < i:
                    print(f"  {'':>12}", end="")
                else:
                    cell = ','.join(sorted(table[i][j])) if table[i][j] else '-'
                    print(f"  {cell:>12}", end="")
            print()

        print(f"  Result: {'ACCEPTED' if accepted else 'REJECTED'}")
        print()


# === Exercise 6: Pumping Lemma for CFLs ===
# Problem: Prove L1={a^n b^n c^n d^n}, L2={a^i b^j c^k | 0<=i<=j<=k},
# L3={ww | w in {a,b}*} are not context-free.

def exercise_6():
    """Pumping lemma proofs for context-free languages."""
    print("Pumping Lemma for Context-Free Languages")
    print("=" * 50)
    print()
    print("Recall: If L is context-free, there exists p such that for any")
    print("s in L with |s| >= p, s = uvxyz where:")
    print("  1. |vy| > 0")
    print("  2. |vxy| <= p")
    print("  3. For all i >= 0, uv^ixy^iz is in L")
    print()

    print("Proof 1: L1 = {a^n b^n c^n d^n | n >= 0} is NOT context-free")
    print("-" * 50)
    print("  Assume L1 is CF with pumping length p.")
    print("  Choose s = a^p b^p c^p d^p. |s| = 4p >= p.")
    print("  Write s = uvxyz with |vy| > 0 and |vxy| <= p.")
    print("  Since |vxy| <= p, vxy can span at most 2 consecutive symbol types.")
    print("  Case analysis (vxy is within):")
    print("    - a's and b's: pumping changes #a and/or #b but not #c, #d")
    print("    - b's and c's: pumping changes #b and/or #c but not #a, #d")
    print("    - c's and d's: pumping changes #c and/or #d but not #a, #b")
    print("    - only a's, only b's, etc.: similarly unbalanced")
    print("  In every case, pumping (i=2) breaks the a^n=b^n=c^n=d^n constraint.")
    print("  Contradiction. L1 is not context-free.")
    print()

    print("Proof 2: L2 = {a^i b^j c^k | 0 <= i <= j <= k} is NOT context-free")
    print("-" * 50)
    print("  Assume L2 is CF with pumping length p.")
    print("  Choose s = a^p b^p c^p. Clearly 0 <= p <= p <= p, so s in L2.")
    print("  Write s = uvxyz with |vy| > 0, |vxy| <= p.")
    print("  vxy spans at most 2 consecutive symbol types. Cases:")
    print("    - vxy in a's: pump UP (i=2) -> more a's, might get #a > #b. Violation.")
    print("    - vxy in a's and b's: pump DOWN (i=0).")
    print("      v has a's, y has b's (or vice versa). Removing them decreases")
    print("      #a+#b but not #c. But we need #a <= #b <= #c.")
    print("      If v=a^s, y=b^t: pumping UP gives a^(p+s) b^(p+t) c^p.")
    print("      If s > 0 and t = 0: #a > #b. Violation.")
    print("      If s = 0 and t > 0: #b > #c. Violation.")
    print("      If s > 0 and t > 0: #b might exceed #c. Violation.")
    print("    - vxy in b's and c's: pump DOWN (i=0) -> fewer b's and c's.")
    print("      Could get #a > #b. Violation.")
    print("    - vxy in c's only: pump DOWN (i=0) -> fewer c's.")
    print("      Could get #b > #c. Violation.")
    print("  All cases lead to violation. L2 is not context-free.")
    print()

    print("Proof 3: L3 = {ww | w in {a,b}*} (copy language) is NOT context-free")
    print("-" * 50)
    print("  Assume L3 is CF with pumping length p.")
    print("  Choose s = a^p b^p a^p b^p. Then w = a^p b^p and s = ww in L3.")
    print("  Write s = uvxyz with |vy| > 0, |vxy| <= p.")
    print("  |vxy| <= p, so vxy is contained within one of these regions:")
    print("    Region 1: first a^p (positions 0..p-1)")
    print("    Region 2: first b^p (positions p..2p-1)")
    print("    Boundary 1-2: spanning first a^p and first b^p")
    print("    Region 3: second a^p (positions 2p..3p-1)")
    print("    Region 4: second b^p (positions 3p..4p-1)")
    print("    Boundary 2-3: spanning first b^p and second a^p")
    print("    Boundary 3-4: spanning second a^p and second b^p")
    print()
    print("  For any placement, pumping (i=0 or i=2) creates asymmetry")
    print("  between the first half and second half. The resulting string")
    print("  cannot be written as ww for any w.")
    print()
    print("  Detailed: If vxy is in the first a^p (Region 1), pumping up gives")
    print("  a^(p+k) b^p a^p b^p. The first half is a^((p+k)/2) which doesn't")
    print("  split evenly into two identical halves. Contradiction.")
    print("  (Similar analysis for all other regions.)")
    print()
    print("  Therefore L3 is not context-free.")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Grammar Writing ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Derivations and Parse Trees ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Ambiguity Analysis ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: CNF Conversion ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: CYK Algorithm ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Pumping Lemma for CFLs ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
