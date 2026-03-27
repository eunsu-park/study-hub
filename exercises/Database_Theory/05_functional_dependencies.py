"""
Exercises for Lesson 05: Functional Dependencies
Topic: Database_Theory

Solutions to practice problems from the lesson.
Implements attribute closure, candidate key finding, minimal cover,
Armstrong's axioms proofs, and FD implication checking.
"""

from itertools import combinations


# ============================================================
# Core FD Algorithms
# ============================================================

def attribute_closure(attrs, fds):
    """Compute the closure of a set of attributes under a set of FDs.

    Args:
        attrs: frozenset of attribute names
        fds: list of (lhs: frozenset, rhs: frozenset) tuples

    Returns:
        frozenset: the closure attrs+
    """
    closure = set(attrs)
    changed = True
    while changed:
        changed = False
        for lhs, rhs in fds:
            if lhs.issubset(closure) and not rhs.issubset(closure):
                closure.update(rhs)
                changed = True
    return frozenset(closure)


def is_superkey(attrs, all_attrs, fds):
    """Check if attrs is a superkey (its closure is all attributes)."""
    return attribute_closure(attrs, fds) == all_attrs


def find_candidate_keys(all_attrs, fds):
    """Find all candidate keys by classifying attributes and searching."""
    all_attrs = frozenset(all_attrs)

    # Classify attributes
    lhs_attrs = set()
    rhs_attrs = set()
    for lhs, rhs in fds:
        lhs_attrs.update(lhs)
        rhs_attrs.update(rhs)

    l_only = lhs_attrs - rhs_attrs    # Appear only on LHS
    r_only = rhs_attrs - lhs_attrs    # Appear only on RHS
    both = lhs_attrs & rhs_attrs      # Appear on both sides
    neither = all_attrs - lhs_attrs - rhs_attrs  # Appear in neither

    # Core must be in every key: L-only + Neither
    core = frozenset(l_only | neither)

    # Check if core alone is a superkey
    if is_superkey(core, all_attrs, fds):
        return [core]

    # Try adding combinations of 'both' attributes
    candidate_keys = []
    both_list = sorted(both)

    for size in range(1, len(both_list) + 1):
        for combo in combinations(both_list, size):
            candidate = frozenset(core | set(combo))
            if is_superkey(candidate, all_attrs, fds):
                # Check minimality: no proper subset should be a superkey
                is_minimal = True
                for ck in candidate_keys:
                    if ck.issubset(candidate):
                        is_minimal = False
                        break
                if is_minimal:
                    # Also check that removing any single attr breaks superkey property
                    truly_minimal = True
                    for attr in candidate:
                        subset = candidate - {attr}
                        if subset and is_superkey(subset, all_attrs, fds):
                            truly_minimal = False
                            break
                    if truly_minimal:
                        candidate_keys.append(candidate)

    return candidate_keys


def minimal_cover(fds):
    """Compute the minimal cover of a set of FDs.

    Steps:
    1. Decompose RHS to single attributes
    2. Remove extraneous LHS attributes
    3. Remove redundant FDs
    """
    # Step 1: Decompose RHS
    decomposed = []
    for lhs, rhs in fds:
        for attr in rhs:
            decomposed.append((frozenset(lhs), frozenset({attr})))

    # Step 2: Remove extraneous LHS attributes
    result = []
    for lhs, rhs in decomposed:
        new_lhs = set(lhs)
        for attr in list(lhs):
            if len(new_lhs) > 1:
                test_lhs = frozenset(new_lhs - {attr})
                # Check if test_lhs -> rhs still holds
                test_fds = [(frozenset(new_lhs - {attr}) if (frozenset(new_lhs), r) == (frozenset(new_lhs), rhs) else l, r)
                            for l, r in decomposed]
                # Actually, we check closure of test_lhs under remaining FDs
                remaining = [(l, r) for l, r in decomposed if (l, r) != (frozenset(new_lhs), rhs)]
                remaining.append((test_lhs, rhs))
                closure = attribute_closure(test_lhs, remaining)
                if rhs.issubset(closure):
                    new_lhs.discard(attr)
        result.append((frozenset(new_lhs), rhs))

    # Step 3: Remove redundant FDs
    final = list(result)
    i = 0
    while i < len(final):
        # Try removing FD at index i
        without = final[:i] + final[i+1:]
        lhs, rhs = final[i]
        closure = attribute_closure(lhs, without)
        if rhs.issubset(closure):
            final = without
        else:
            i += 1

    return final


def format_fd(lhs, rhs):
    """Format an FD for display."""
    return f"{''.join(sorted(lhs))} -> {''.join(sorted(rhs))}"


# ============================================================
# Exercise Solutions
# ============================================================

# === Exercise 1: Attribute Closure ===
# Problem: Compute closures for R(A,B,C,D,E) with F = {AB->C, C->D, BD->E, A->B}

def exercise_1():
    """Compute attribute closures."""
    fds = [
        (frozenset("AB"), frozenset("C")),
        (frozenset("C"),  frozenset("D")),
        (frozenset("BD"), frozenset("E")),
        (frozenset("A"),  frozenset("B")),
    ]

    closures_to_compute = [
        frozenset("A"),
        frozenset("BC"),
        frozenset("AD"),
        frozenset("CD"),
    ]

    print("R(A, B, C, D, E)")
    print("F = { AB->C, C->D, BD->E, A->B }\n")

    for attrs in closures_to_compute:
        closure = attribute_closure(attrs, fds)
        attr_str = "{" + ",".join(sorted(attrs)) + "}"
        closure_str = "{" + ",".join(sorted(closure)) + "}"
        print(f"  {attr_str}+ = {closure_str}")

        # Show derivation steps
        current = set(attrs)
        steps = [f"Start: {{{','.join(sorted(current))}}}"]
        changed = True
        while changed:
            changed = False
            for lhs, rhs in fds:
                if lhs.issubset(current) and not rhs.issubset(current):
                    fd_str = format_fd(lhs, rhs)
                    current.update(rhs)
                    steps.append(f"Apply {fd_str}: {{{','.join(sorted(current))}}}")
                    changed = True
        for step in steps:
            print(f"    {step}")
        print()


# === Exercise 2: Finding Candidate Keys ===
# Problem: Find all candidate keys for Exercise 1's relation.

def exercise_2():
    """Find all candidate keys."""
    all_attrs = frozenset("ABCDE")
    fds = [
        (frozenset("AB"), frozenset("C")),
        (frozenset("C"),  frozenset("D")),
        (frozenset("BD"), frozenset("E")),
        (frozenset("A"),  frozenset("B")),
    ]

    print("Classifying attributes:")
    lhs_attrs = set()
    rhs_attrs = set()
    for lhs, rhs in fds:
        lhs_attrs.update(lhs)
        rhs_attrs.update(rhs)

    l_only = lhs_attrs - rhs_attrs
    r_only = rhs_attrs - lhs_attrs
    both = lhs_attrs & rhs_attrs

    print(f"  L-only (must be in every key): {sorted(l_only)}")
    print(f"  R-only (never in any key): {sorted(r_only)}")
    print(f"  Both sides: {sorted(both)}")
    print()

    core = frozenset(l_only)
    print(f"  CORE = {{{','.join(sorted(core))}}}")
    closure = attribute_closure(core, fds)
    print(f"  {{A}}+ = {{{','.join(sorted(closure))}}}")

    if closure == all_attrs:
        print(f"  {{A}}+ = all attributes => {{A}} is the only candidate key!")
    print()

    keys = find_candidate_keys(all_attrs, fds)
    print(f"  Candidate keys: {['{' + ','.join(sorted(k)) + '}' for k in keys]}")


# === Exercise 3: Minimal Cover ===
# Problem: Find minimal cover for F = { A->BC, B->C, AB->D, D->BC }

def exercise_3():
    """Compute minimal cover step by step."""
    fds = [
        (frozenset("A"),  frozenset("BC")),
        (frozenset("B"),  frozenset("C")),
        (frozenset("AB"), frozenset("D")),
        (frozenset("D"),  frozenset("BC")),
    ]

    print("F = { A->BC, B->C, AB->D, D->BC }")
    print()

    # Step 1: Decompose RHS
    step1 = []
    for lhs, rhs in fds:
        for attr in sorted(rhs):
            step1.append((frozenset(lhs), frozenset(attr)))
    print("Step 1: Decompose RHS")
    print(f"  F = {{ {', '.join(format_fd(l, r) for l, r in step1)} }}")
    print()

    # Step 2: Check extraneous LHS attributes
    print("Step 2: Remove extraneous LHS attributes")
    print("  Check AB->D:")
    # Try removing A: {B}+ = {B, C}. D not in {B}+. Keep A.
    closure_b = attribute_closure(frozenset("B"), step1)
    print(f"    Remove A: {{B}}+ = {{{','.join(sorted(closure_b))}}}. D in closure? {'D' in closure_b}")
    # Try removing B: {A}+ under current FDs
    closure_a = attribute_closure(frozenset("A"), step1)
    print(f"    Remove B: {{A}}+ = {{{','.join(sorted(closure_a))}}}. D in closure? {'D' in closure_a}")
    if 'D' in closure_a:
        print("    B is extraneous! Replace AB->D with A->D")
    print()

    # Step 3: Remove redundant FDs
    print("Step 3: Remove redundant FDs")
    result = minimal_cover(fds)
    print(f"  Minimal cover: {{ {', '.join(format_fd(l, r) for l, r in result)} }}")


# === Exercise 4: Proving FD with Armstrong's Axioms ===
# Problem: Prove A->D from F = {A->B, B->C, C->D}

def exercise_4():
    """Prove A->D using Armstrong's axioms."""
    print("Given: F = { A->B, B->C, C->D }")
    print("Prove: A->D")
    print()
    proof = [
        ("1. A -> B",     "Given"),
        ("2. B -> C",     "Given"),
        ("3. A -> C",     "Transitivity on (1) and (2)"),
        ("4. C -> D",     "Given"),
        ("5. A -> D",     "Transitivity on (3) and (4)  QED"),
    ]
    for step, reason in proof:
        print(f"  {step:<20} [{reason}]")

    # Verify computationally
    fds = [
        (frozenset("A"), frozenset("B")),
        (frozenset("B"), frozenset("C")),
        (frozenset("C"), frozenset("D")),
    ]
    closure = attribute_closure(frozenset("A"), fds)
    print(f"\n  Verification: {{A}}+ = {{{','.join(sorted(closure))}}} -> D in {{A}}+: {'D' in closure}")


# === Exercise 5: FD Implication ===
# Problem: Check if AE->D, BE->D, A->D are implied by F = {A->B, BC->D, E->C}

def exercise_5():
    """Check FD implication using attribute closure."""
    fds = [
        (frozenset("A"),  frozenset("B")),
        (frozenset("BC"), frozenset("D")),
        (frozenset("E"),  frozenset("C")),
    ]

    print("F = { A->B, BC->D, E->C }")
    print()

    checks = [
        (frozenset("AE"), frozenset("D"), "AE -> D"),
        (frozenset("BE"), frozenset("D"), "BE -> D"),
        (frozenset("A"),  frozenset("D"), "A -> D"),
    ]

    for lhs, rhs, label in checks:
        closure = attribute_closure(lhs, fds)
        implied = rhs.issubset(closure)
        lhs_str = "{" + ",".join(sorted(lhs)) + "}"
        closure_str = "{" + ",".join(sorted(closure)) + "}"
        rhs_str = ",".join(sorted(rhs))
        status = "YES, implied" if implied else "NO, not implied"

        print(f"  {label}: {lhs_str}+ = {closure_str}")
        print(f"    {rhs_str} in closure? {status}")
        print()


# === Exercise 6: Equivalence of FD Sets ===
# Problem: Are F = {A->B, B->C, A->C} and G = {A->B, B->C} equivalent?

def exercise_6():
    """Check equivalence of two FD sets."""
    F = [
        (frozenset("A"), frozenset("B")),
        (frozenset("B"), frozenset("C")),
        (frozenset("A"), frozenset("C")),
    ]
    G = [
        (frozenset("A"), frozenset("B")),
        (frozenset("B"), frozenset("C")),
    ]

    print("F = { A->B, B->C, A->C }")
    print("G = { A->B, B->C }")
    print()

    # Check F covers G: every FD in G is implied by F
    print("Check: Every FD in G implied by F?")
    f_covers_g = True
    for lhs, rhs in G:
        closure = attribute_closure(lhs, F)
        implied = rhs.issubset(closure)
        print(f"  {format_fd(lhs, rhs)}: {{{','.join(sorted(lhs))}}}+_F = {{{','.join(sorted(closure))}}} -> {implied}")
        f_covers_g = f_covers_g and implied

    print()

    # Check G covers F: every FD in F is implied by G
    print("Check: Every FD in F implied by G?")
    g_covers_f = True
    for lhs, rhs in F:
        closure = attribute_closure(lhs, G)
        implied = rhs.issubset(closure)
        print(f"  {format_fd(lhs, rhs)}: {{{','.join(sorted(lhs))}}}+_G = {{{','.join(sorted(closure))}}} -> {implied}")
        g_covers_f = g_covers_f and implied

    print()
    equivalent = f_covers_g and g_covers_f
    print(f"F covers G: {f_covers_g}")
    print(f"G covers F: {g_covers_f}")
    print(f"F equiv G: {equivalent}")
    print("A->C in F is redundant (follows from A->B and B->C by transitivity).")


# === Exercise 7: Multiple Candidate Keys ===
# Problem: Find all candidate keys for R(A,B,C,D,E) with F = {AB->CDE, C->A, D->B}

def exercise_7():
    """Find all candidate keys with multiple keys."""
    all_attrs = frozenset("ABCDE")
    fds = [
        (frozenset("AB"), frozenset("CDE")),
        (frozenset("C"),  frozenset("A")),
        (frozenset("D"),  frozenset("B")),
    ]

    print("R(A, B, C, D, E)")
    print("F = { AB->CDE, C->A, D->B }")
    print()

    # Classify
    lhs_attrs = set()
    rhs_attrs = set()
    for lhs, rhs in fds:
        lhs_attrs.update(lhs)
        rhs_attrs.update(rhs)

    l_only = lhs_attrs - rhs_attrs
    r_only = rhs_attrs - lhs_attrs
    both = lhs_attrs & rhs_attrs
    neither = all_attrs - lhs_attrs - rhs_attrs

    print(f"  L-only: {sorted(l_only)}")
    print(f"  R-only: {sorted(r_only)}")
    print(f"  Both: {sorted(both)}")
    print(f"  Neither: {sorted(neither)}")
    print(f"  CORE = L-only ∪ Neither = {sorted(l_only | neither)}")
    print()

    # Try all pairs
    print("Checking pairs:")
    pairs = list(combinations(sorted(all_attrs), 2))
    for pair in pairs:
        attrs = frozenset(pair)
        closure = attribute_closure(attrs, fds)
        is_sk = closure == all_attrs
        if is_sk:
            print(f"  {{{','.join(sorted(attrs))}}}+ = {{{','.join(sorted(closure))}}} -- SUPERKEY")

    print()

    keys = find_candidate_keys(all_attrs, fds)
    key_strs = ["{" + ",".join(sorted(k)) + "}" for k in keys]
    print(f"Candidate keys: {key_strs}")


# === Exercise 8: Union Rule Proof ===
# Problem: Prove Union Rule from Armstrong's axioms.

def exercise_8():
    """Prove Union Rule: X->Y, X->Z => X->YZ."""
    print("Prove: Union Rule (X->Y, X->Z => X->YZ)")
    print("Using only Armstrong's three axioms:")
    print()
    proof = [
        ("1. X -> Y",       "Given"),
        ("2. X -> XY",      "Augment (1) with X: XX -> XY, and XX = X"),
        ("3. X -> Z",       "Given"),
        ("4. XY -> YZ",     "Augment (3) with Y: XY -> ZY"),
        ("5. X -> YZ",      "Transitivity on (2) and (4)  QED"),
    ]
    for step, reason in proof:
        print(f"  {step:<20} [{reason}]")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Attribute Closure ===")
    print("=" * 70)
    exercise_1()

    print("=" * 70)
    print("=== Exercise 2: Finding Candidate Keys ===")
    print("=" * 70)
    exercise_2()

    print("=" * 70)
    print("=== Exercise 3: Minimal Cover ===")
    print("=" * 70)
    exercise_3()

    print("=" * 70)
    print("=== Exercise 4: Armstrong's Axioms Proof ===")
    print("=" * 70)
    exercise_4()

    print("=" * 70)
    print("=== Exercise 5: FD Implication ===")
    print("=" * 70)
    exercise_5()

    print("=" * 70)
    print("=== Exercise 6: Equivalence of FD Sets ===")
    print("=" * 70)
    exercise_6()

    print("=" * 70)
    print("=== Exercise 7: Multiple Candidate Keys ===")
    print("=" * 70)
    exercise_7()

    print("=" * 70)
    print("=== Exercise 8: Union Rule Proof ===")
    print("=" * 70)
    exercise_8()

    print("\nAll exercises completed!")
