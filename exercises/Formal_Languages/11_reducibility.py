"""
Exercises for Lesson 11: Reducibility
Topic: Formal_Languages

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Mapping Reductions ===
# Problem: Construct mapping reductions to prove:
# 1. A_TM <=_m HALT_TM (give the explicit construction of M')
# 2. A_TM <=_m {<M> | L(M) is infinite}

def exercise_1():
    """Mapping reduction constructions."""

    print("Part 1: A_TM <=_m HALT_TM")
    print("=" * 60)
    print()
    print("  Recall:")
    print("    A_TM = {<M, w> | M is a TM that accepts w}")
    print("    HALT_TM = {<M, w> | M is a TM that halts on w}")
    print()
    print("  Reduction function f:")
    print("  Given <M, w>, construct TM M' as follows:")
    print()
    print("    M' on input x:")
    print("      1. Simulate M on x.")
    print("      2. If M accepts x, ACCEPT (halt in accept state).")
    print("      3. If M rejects x, enter an INFINITE LOOP.")
    print()
    print("  Then define: f(<M, w>) = <M', w>")
    print()
    print("  Correctness:")
    print("    Case 1: <M, w> in A_TM (M accepts w)")
    print("      => M' on w: simulates M on w, M accepts, so M' accepts.")
    print("      => M' halts on w.")
    print("      => <M', w> in HALT_TM. Correct.")
    print()
    print("    Case 2: <M, w> not in A_TM (M rejects or loops on w)")
    print("      Sub-case 2a: M rejects w")
    print("        => M' on w: simulates M, M rejects, so M' enters infinite loop.")
    print("        => M' does NOT halt on w.")
    print("        => <M', w> not in HALT_TM. Correct.")
    print("      Sub-case 2b: M loops on w")
    print("        => M' on w: simulates M, M loops, so M' also loops.")
    print("        => M' does NOT halt on w.")
    print("        => <M', w> not in HALT_TM. Correct.")
    print()
    print("  In both directions: <M,w> in A_TM iff f(<M,w>) in HALT_TM.")
    print("  f is computable (simple TM transformation).")
    print("  Therefore A_TM <=_m HALT_TM, and HALT_TM is undecidable. QED.")

    print()
    print()
    print("Part 2: A_TM <=_m {<M> | L(M) is infinite}")
    print("=" * 60)
    print()
    print("  Let INF_TM = {<M> | L(M) is infinite}.")
    print()
    print("  Reduction function f:")
    print("  Given <M, w>, construct TM M' as follows:")
    print()
    print("    M' on input x:")
    print("      1. Simulate M on w (ignoring x for the decision).")
    print("      2. If M accepts w, ACCEPT x.")
    print("      3. If M rejects w, REJECT x.")
    print("      4. (If M loops on w, M' also loops on x.)")
    print()
    print("  Then define: f(<M, w>) = <M'>")
    print()
    print("  Correctness:")
    print("    Case 1: <M, w> in A_TM (M accepts w)")
    print("      => M' accepts ALL inputs (for every x, M' simulates M on w,")
    print("         M accepts, so M' accepts x).")
    print("      => L(M') = Sigma*, which is infinite.")
    print("      => <M'> in INF_TM. Correct.")
    print()
    print("    Case 2: <M, w> not in A_TM")
    print("      Sub-case 2a: M rejects w")
    print("        => M' rejects all inputs => L(M') = empty, not infinite.")
    print("        => <M'> not in INF_TM. Correct.")
    print("      Sub-case 2b: M loops on w")
    print("        => M' loops on all inputs => L(M') = empty, not infinite.")
    print("        => <M'> not in INF_TM. Correct.")
    print()
    print("  Therefore A_TM <=_m INF_TM, and INF_TM is undecidable. QED.")
    print()
    print("  Note: This also follows directly from Rice's theorem,")
    print("  since 'L(M) is infinite' is a nontrivial property of")
    print("  Turing-recognizable languages.")


# === Exercise 2: Rice's Theorem ===
# Problem: For each language, determine if Rice's theorem applies:
# 1. {<M> | M has at most 5 states}
# 2. {<M> | L(M) contains only even-length strings}
# 3. {<M> | M halts on all inputs within 100 steps}

def exercise_2():
    """Rice's theorem applications."""

    print("Part 1: {<M> | M has at most 5 states}")
    print("=" * 60)
    print("  Does Rice's theorem apply? NO.")
    print()
    print("  Reason: 'Having at most 5 states' is a property of the TM itself")
    print("  (its description/structure), NOT a property of the language it")
    print("  recognizes. Two different TMs can recognize the same language")
    print("  while having different numbers of states.")
    print()
    print("  Rice's theorem only applies to semantic properties -- properties")
    print("  of L(M), not syntactic properties of M's description.")
    print()
    print("  Is this language decidable? YES.")
    print("  A decider simply examines the encoding <M> and counts the number")
    print("  of states. This is a finite syntactic check on the input string.")

    print()
    print()
    print("Part 2: {<M> | L(M) contains only even-length strings}")
    print("=" * 60)
    print("  Does Rice's theorem apply? YES.")
    print()
    print("  Check nontriviality:")
    print("  - Some TMs have the property: e.g., a TM that accepts {aa, bb, ...}")
    print("    (a regular language of even-length strings)")
    print("  - Some TMs don't: e.g., a TM that accepts {a} (odd-length string)")
    print()
    print("  This is a nontrivial property of Turing-recognizable languages.")
    print("  By Rice's theorem, the language is UNDECIDABLE.")
    print()
    print("  Conclusion: We cannot write an algorithm that, given an arbitrary")
    print("  TM M, determines whether L(M) consists entirely of even-length strings.")

    print()
    print()
    print("Part 3: {<M> | M halts on all inputs within 100 steps}")
    print("=" * 60)
    print("  Does Rice's theorem apply? NO (at first glance), but careful analysis needed.")
    print()
    print("  Analysis: 'M halts on all inputs within 100 steps' is about the")
    print("  BEHAVIOR of M, not directly about L(M). Two TMs can recognize")
    print("  the same language but one might halt within 100 steps on all inputs")
    print("  while the other takes longer.")
    print()
    print("  Example: L = {0, 1}. TM M1 checks if input is '0' or '1' in 5 steps.")
    print("  TM M2 first wastes 1000 steps doing nothing, then checks. Same L(M),")
    print("  different halting behavior.")
    print()
    print("  Since this property depends on M's implementation, not just L(M),")
    print("  Rice's theorem does NOT apply.")
    print()
    print("  Is this language decidable? YES!")
    print()
    print("  Proof: Given <M>, we need to check if M halts within 100 steps")
    print("  on ALL inputs. But there are infinitely many inputs!")
    print()
    print("  Key insight: If M has n states and tape alphabet of size k,")
    print("  then after 100 steps, M can only have visited at most 101 tape cells.")
    print("  The number of distinct configurations after at most 100 steps is")
    print("  bounded by n * k^101 * 101 (state, tape contents, head position).")
    print("  This is finite, so we can determine M's behavior on ALL inputs of")
    print("  length <= 100 by direct simulation.")
    print()
    print("  For inputs longer than 100: after 100 steps, M can only have read")
    print("  the first 101 symbols. So M's behavior depends only on the first")
    print("  101 symbols, not the full input. We can check all O(k^101) prefixes.")
    print()
    print("  This is a finite computation, so the language is decidable.")


# === Exercise 3: PCP ===
# Problem: Find a match for this PCP instance (or argue no match exists):
# Dominos: (ab, a), (b, ab), (a, b)

def exercise_3():
    """Solve a Post Correspondence Problem instance."""

    print("PCP Instance:")
    print("  Domino 1: (ab, a)   -- top: 'ab', bottom: 'a'")
    print("  Domino 2: (b, ab)   -- top: 'b',  bottom: 'ab'")
    print("  Domino 3: (a, b)    -- top: 'a',  bottom: 'b'")
    print()

    dominos = [("ab", "a"), ("b", "ab"), ("a", "b")]

    # BFS search for a match
    from collections import deque

    # State: (top_string, bottom_string) or just the suffix difference
    # We search for sequences where top == bottom
    queue = deque()
    # Start with each domino
    for i, (t, b) in enumerate(dominos):
        queue.append(([i + 1], t, b))  # (sequence, top, bottom)

    visited = set()
    max_search_len = 12  # Limit search depth
    found = False

    while queue:
        seq, top, bottom = queue.popleft()

        if len(seq) > max_search_len:
            continue

        if top == bottom:
            # Found a match!
            print(f"  MATCH FOUND!")
            print(f"  Sequence: {seq}")
            print(f"  Top:    ", end="")
            for idx in seq:
                print(f"{dominos[idx-1][0]}", end=" ")
            print(f"= '{top}'")
            print(f"  Bottom: ", end="")
            for idx in seq:
                print(f"{dominos[idx-1][1]}", end=" ")
            print(f"= '{bottom}'")
            found = True
            break

        # Pruning: only continue if one is a prefix of the other
        if not (top.startswith(bottom) or bottom.startswith(top)):
            continue

        # Compute a state key for cycle detection
        if len(top) > len(bottom):
            state_key = ("T", top[len(bottom):])
        else:
            state_key = ("B", bottom[len(top):])

        if state_key in visited:
            continue
        visited.add(state_key)

        # Try appending each domino
        for i, (t, b) in enumerate(dominos):
            new_seq = seq + [i + 1]
            new_top = top + t
            new_bottom = bottom + b
            queue.append((new_seq, new_top, new_bottom))

    if not found:
        print(f"  No match found within sequence length {max_search_len}.")
        print()
        print("  Analysis of why no match exists:")
        print("  Let's track the length difference: |top| - |bottom|")
        print("    Domino 1: top='ab' (len 2), bottom='a' (len 1) -> diff += 1")
        print("    Domino 2: top='b' (len 1), bottom='ab' (len 2) -> diff -= 1")
        print("    Domino 3: top='a' (len 1), bottom='b' (len 1) -> diff += 0")
        print()
        print("  For a match, we need the top and bottom strings to be equal.")
        print("  Let's search more carefully by tracking the content...")
        print()

        # Deeper search
        queue2 = deque()
        for i, (t, b) in enumerate(dominos):
            queue2.append(([i + 1], t, b))

        visited2 = set()
        max_depth = 20
        found2 = False

        while queue2:
            seq, top, bottom = queue2.popleft()
            if len(seq) > max_depth:
                continue

            if top == bottom and len(top) > 0:
                print(f"  MATCH FOUND (deeper search)!")
                print(f"  Sequence: {seq}")
                print(f"  String: '{top}'")
                found2 = True
                break

            if not (top.startswith(bottom) or bottom.startswith(top)):
                continue

            if len(top) > len(bottom):
                diff = top[len(bottom):]
            else:
                diff = bottom[len(top):]
            state = (len(top) > len(bottom), diff, len(seq) % 5)

            for i, (t, b) in enumerate(dominos):
                new_seq = seq + [i + 1]
                queue2.append((new_seq, top + t, bottom + b))

        if not found2:
            print(f"  Exhaustive search up to length {max_depth}: no match found.")
            print("  This instance likely has NO solution.")

    # Verify the solution if found
    if found:
        print("\n  Verification:")
        top_str = "".join(dominos[i-1][0] for i in seq)
        bot_str = "".join(dominos[i-1][1] for i in seq)
        print(f"    Top string:    '{top_str}'")
        print(f"    Bottom string: '{bot_str}'")
        print(f"    Match: {top_str == bot_str}")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Mapping Reductions ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Rice's Theorem ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: PCP ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
