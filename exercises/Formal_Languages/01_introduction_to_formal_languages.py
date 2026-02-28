"""
Exercises for Lesson 01: Introduction to Formal Languages
Topic: Formal_Languages

Solutions to practice problems from the lesson.
"""


# === Exercise 1: String Operations ===
# Problem: Let Sigma = {a, b}, x = abba, y = bab.
# 1. Compute xy, yx, x^R, y^R.
# 2. List all prefixes of x.
# 3. List all substrings of y (no duplicates).

def exercise_1():
    """String operations on x = 'abba' and y = 'bab'."""
    x = "abba"
    y = "bab"

    # Part 1: Concatenation and reversal
    xy = x + y
    yx = y + x
    x_rev = x[::-1]
    y_rev = y[::-1]

    print("Part 1: Concatenation and Reversal")
    print(f"  x = {x}")
    print(f"  y = {y}")
    print(f"  xy = {xy}")
    print(f"  yx = {yx}")
    print(f"  x^R = {x_rev}")
    print(f"  y^R = {y_rev}")

    # Part 2: All prefixes of x (including epsilon and x itself)
    prefixes = [x[:i] for i in range(len(x) + 1)]
    print("\nPart 2: All prefixes of x")
    for p in prefixes:
        label = "epsilon" if p == "" else p
        print(f"  '{label}' (length {len(p)})")

    # Part 3: All substrings of y (no duplicates)
    substrings = set()
    for i in range(len(y)):
        for j in range(i + 1, len(y) + 1):
            substrings.add(y[i:j])
    substrings.add("")  # epsilon is a substring of every string

    print("\nPart 3: All substrings of y (no duplicates)")
    for s in sorted(substrings, key=lambda x: (len(x), x)):
        label = "epsilon" if s == "" else s
        print(f"  '{label}'")


# === Exercise 2: Language Operations ===
# Problem: Let L1 = {a, bb} and L2 = {b, ab} over Sigma = {a, b}.
# 1. Compute L1 * L2 (concatenation).
# 2. Compute L1^2 = L1 * L1.
# 3. List strings in L1* of length <= 3.

def exercise_2():
    """Language operations on L1 = {a, bb} and L2 = {b, ab}."""
    L1 = {"a", "bb"}
    L2 = {"b", "ab"}

    # Part 1: L1 . L2 (concatenation)
    L1_L2 = {x + y for x in L1 for y in L2}
    print("Part 1: L1 . L2 (concatenation)")
    print(f"  L1 = {L1}")
    print(f"  L2 = {L2}")
    print(f"  L1 . L2 = {sorted(L1_L2)}")

    # Part 2: L1^2 = L1 . L1
    L1_squared = {x + y for x in L1 for y in L1}
    print("\nPart 2: L1^2 = L1 . L1")
    print(f"  L1^2 = {sorted(L1_squared)}")

    # Part 3: Strings in L1* of length <= 3
    # L1* = L1^0 union L1^1 union L1^2 union ...
    # L1^0 = {epsilon}
    # L1^1 = {a, bb}
    # L1^2 = {aa, abb, bba, bbbb} -- bbbb has length 4, skip
    # L1^3 would produce strings of length >= 3, check each

    def kleene_star_up_to_length(lang, max_len):
        """Generate all strings in lang* with length <= max_len."""
        result = {""}  # L^0 = {epsilon}
        current = {""}
        while True:
            new_gen = set()
            for s in current:
                for w in lang:
                    concatenated = s + w
                    if len(concatenated) <= max_len:
                        new_gen.add(concatenated)
            if not new_gen - result:
                break
            result |= new_gen
            current = new_gen
        return result

    L1_star_le3 = kleene_star_up_to_length(L1, 3)
    print("\nPart 3: Strings in L1* of length <= 3")
    for s in sorted(L1_star_le3, key=lambda x: (len(x), x)):
        label = "epsilon" if s == "" else s
        print(f"  '{label}' (length {len(s)})")


# === Exercise 3: Language Classification ===
# Problem: Classify each language and justify informally.
# 1. L = {w in {0,1}* | w has an even number of 0s}
# 2. L = {0^n 1^n | n >= 0}
# 3. L = {0^n 1^n 2^n | n >= 0}
# 4. The set of all valid C programs

def exercise_3():
    """Language classification with justification."""
    classifications = [
        {
            "language": "L = {w in {0,1}* | w has an even number of 0s}",
            "type": "Regular (Type 3)",
            "justification": (
                "A DFA with 2 states can track parity of 0s: one state for "
                "'even count' (also the start and accept state) and one for "
                "'odd count'. Reading a 0 toggles between states; reading a 1 "
                "stays in the current state."
            ),
        },
        {
            "language": "L = {0^n 1^n | n >= 0}",
            "type": "Context-Free (Type 2), but NOT Regular",
            "justification": (
                "A PDA can push a symbol for each 0, then pop for each 1. "
                "The grammar S -> 0S1 | epsilon generates this language. "
                "It is not regular because a DFA has finite memory and cannot "
                "count an unbounded number of 0s to match with 1s "
                "(provable via the pumping lemma)."
            ),
        },
        {
            "language": "L = {0^n 1^n 2^n | n >= 0}",
            "type": "Context-Sensitive (Type 1), but NOT Context-Free",
            "justification": (
                "A PDA's single stack cannot enforce three-way equality. "
                "The CFL pumping lemma proves this is not context-free: "
                "any pump region spans at most two of the three symbol types, "
                "so pumping breaks the n=n=n constraint. "
                "An LBA (or a context-sensitive grammar) can verify all three "
                "counts are equal using bounded workspace."
            ),
        },
        {
            "language": "The set of all valid C programs",
            "type": "Decidable (Recursive), approximately Context-Free for syntax",
            "justification": (
                "The syntax of C is specified by a context-free grammar "
                "(the C grammar in the standard). However, semantic validity "
                "(type checking, declaration before use, etc.) goes beyond "
                "context-free. The full set of valid C programs is decidable: "
                "a compiler can always determine in finite time whether a "
                "program is valid."
            ),
        },
    ]

    print("Language Classification:")
    for i, item in enumerate(classifications, 1):
        print(f"\n  {i}. {item['language']}")
        print(f"     Classification: {item['type']}")
        print(f"     Justification: {item['justification']}")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: String Operations ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Language Operations ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Language Classification ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
