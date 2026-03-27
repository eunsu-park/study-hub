"""
Exercises for Lesson 06: Normalization
Topic: Database_Theory

Solutions to practice problems from the lesson.
Covers normal form identification, 3NF synthesis, BCNF decomposition,
lossless-join verification, and anomaly identification.
"""

from itertools import combinations


# ============================================================
# Reusable FD Algorithms (from Lesson 05)
# ============================================================

def attribute_closure(attrs, fds):
    """Compute closure of attrs under fds."""
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
    """Check if attrs is a superkey."""
    return attribute_closure(attrs, fds) == all_attrs


def find_candidate_keys(all_attrs, fds):
    """Find all candidate keys."""
    all_attrs = frozenset(all_attrs)
    lhs_attrs = set()
    rhs_attrs = set()
    for lhs, rhs in fds:
        lhs_attrs.update(lhs)
        rhs_attrs.update(rhs)

    l_only = lhs_attrs - rhs_attrs
    neither = all_attrs - lhs_attrs - rhs_attrs
    core = frozenset(l_only | neither)

    if is_superkey(core, all_attrs, fds):
        return [core]

    both = (lhs_attrs & rhs_attrs) | (rhs_attrs - lhs_attrs)
    remaining = sorted(all_attrs - core)

    candidate_keys = []
    for size in range(len(remaining) + 1):
        for combo in combinations(remaining, size):
            candidate = frozenset(core | set(combo))
            if is_superkey(candidate, all_attrs, fds):
                is_minimal = all(
                    not ck.issubset(candidate) or ck == candidate
                    for ck in candidate_keys
                )
                if is_minimal:
                    truly_minimal = all(
                        not is_superkey(candidate - {a}, all_attrs, fds)
                        for a in candidate
                    )
                    if truly_minimal:
                        candidate_keys.append(candidate)

    return candidate_keys


def format_attrs(attrs):
    """Format attribute set for display."""
    return "{" + ",".join(sorted(attrs)) + "}"


def format_fd(lhs, rhs):
    """Format an FD for display."""
    return f"{''.join(sorted(lhs))} -> {''.join(sorted(rhs))}"


# ============================================================
# Normal Form Checking
# ============================================================

def check_normal_form(all_attrs, fds, keys):
    """Determine the highest normal form of a relation.

    Returns: '1NF', '2NF', '3NF', or 'BCNF'
    """
    all_attrs = frozenset(all_attrs)
    prime_attrs = set()
    for k in keys:
        prime_attrs.update(k)
    non_prime = all_attrs - prime_attrs

    # Check BCNF: every non-trivial FD X->Y, X must be a superkey
    bcnf = True
    for lhs, rhs in fds:
        if not rhs.issubset(lhs):  # non-trivial
            if not is_superkey(lhs, all_attrs, fds):
                bcnf = False
                break

    if bcnf:
        return "BCNF"

    # Check 3NF: every non-trivial FD X->A, either X is superkey or A is prime
    three_nf = True
    for lhs, rhs in fds:
        for a in rhs:
            if a not in lhs:  # non-trivial
                if not is_superkey(lhs, all_attrs, fds) and a not in prime_attrs:
                    three_nf = False
                    break
        if not three_nf:
            break

    if three_nf:
        return "3NF"

    # Check 2NF: no partial dependencies (non-prime attr depends on part of a key)
    two_nf = True
    for lhs, rhs in fds:
        for a in rhs:
            if a not in lhs and a in non_prime:
                # Check if lhs is a proper subset of some key
                for k in keys:
                    if lhs.issubset(k) and lhs != k:
                        two_nf = False
                        break
            if not two_nf:
                break
        if not two_nf:
            break

    if two_nf:
        return "2NF"

    return "1NF"


# ============================================================
# Exercise Solutions
# ============================================================

# === Exercise 1: Identifying Normal Forms ===
# Problem: For each relation, identify the highest normal form.

def exercise_1():
    """Identify highest normal form for each relation."""
    relations = [
        {
            "label": "(a) R(A,B,C,D), Key: {A,B}, FDs: A->C, AB->D",
            "attrs": "ABCD",
            "fds": [(frozenset("A"), frozenset("C")),
                    (frozenset("AB"), frozenset("D"))],
            "keys": [frozenset("AB")]
        },
        {
            "label": "(b) R(A,B,C), Key: {A}, FDs: A->B, B->C",
            "attrs": "ABC",
            "fds": [(frozenset("A"), frozenset("B")),
                    (frozenset("B"), frozenset("C"))],
            "keys": [frozenset("A")]
        },
        {
            "label": "(c) R(A,B,C,D), Key: {A}, FDs: A->BCD",
            "attrs": "ABCD",
            "fds": [(frozenset("A"), frozenset("BCD"))],
            "keys": [frozenset("A")]
        },
        {
            "label": "(d) R(A,B,C), Keys: {A,B} and {A,C}, FDs: AB->C, AC->B, B->C, C->B",
            "attrs": "ABC",
            "fds": [(frozenset("AB"), frozenset("C")),
                    (frozenset("AC"), frozenset("B")),
                    (frozenset("B"),  frozenset("C")),
                    (frozenset("C"),  frozenset("B"))],
            "keys": [frozenset("AB"), frozenset("AC")]
        }
    ]

    for rel in relations:
        nf = check_normal_form(rel["attrs"], rel["fds"], rel["keys"])
        print(f"{rel['label']}")
        print(f"  Highest Normal Form: {nf}")

        # Explain why
        if nf == "1NF":
            print("  Reason: Partial dependency exists (non-prime attr depends on part of key).")
            print("  Example: A->C where A is part of key {A,B} and C is non-prime.")
        elif nf == "2NF":
            print("  Reason: Transitive dependency exists (non-prime depends on non-prime).")
            print("  Example: B->C where B is non-prime and C is non-prime (A->B->C).")
        elif nf == "3NF":
            print("  Reason: Non-trivial FD where determinant is not superkey, but RHS is prime.")
            print("  Example: B->C where B is not superkey but C is prime (part of key {A,C}).")
        elif nf == "BCNF":
            print("  Reason: Every non-trivial FD has a superkey as determinant.")
        print()


# === Exercise 2: 3NF Synthesis ===
# Problem: Apply 3NF synthesis to R(A,B,C,D,E) with F = {A->B, BC->D, D->E, E->C}

def exercise_2():
    """3NF synthesis algorithm step by step."""
    all_attrs = frozenset("ABCDE")
    fds = [
        (frozenset("A"),  frozenset("B")),
        (frozenset("BC"), frozenset("D")),
        (frozenset("D"),  frozenset("E")),
        (frozenset("E"),  frozenset("C")),
    ]

    print("R(A, B, C, D, E)")
    print("F = { A->B, BC->D, D->E, E->C }")
    print()

    # Step 1: Compute minimal cover
    print("Step 1: Minimal Cover")
    print("  Already in minimal form (single RHS, no extraneous LHS, no redundant FDs).")
    print("  F_min = { A->B, BC->D, D->E, E->C }")
    print()

    # Step 2: Group by LHS -> create schemas
    print("Step 2: Create relation schemas (group by LHS)")
    schemas = [
        ("R1", frozenset("AB"),  "A->B",   frozenset("A")),
        ("R2", frozenset("BCD"), "BC->D",  frozenset("BC")),
        ("R3", frozenset("DE"),  "D->E",   frozenset("D")),
        ("R4", frozenset("EC"),  "E->C",   frozenset("E")),
    ]
    for name, attrs, fd, key in schemas:
        print(f"  {name}({','.join(sorted(attrs))}) from {fd}, key: {format_attrs(key)}")
    print()

    # Step 3: Check if any schema contains a candidate key
    print("Step 3: Check for candidate key")
    keys = find_candidate_keys(all_attrs, fds)
    key_strs = [format_attrs(k) for k in keys]
    print(f"  Candidate keys of R: {key_strs}")

    has_key = False
    for name, attrs, _, _ in schemas:
        for k in keys:
            if k.issubset(attrs):
                print(f"  {name} contains key {format_attrs(k)}? Yes")
                has_key = True
                break

    if not has_key:
        print("  No schema contains a candidate key. Adding R5 with a candidate key.")
        schemas.append(("R5", keys[0], "candidate key", keys[0]))
    print()

    # Step 4: Remove subsets
    print("Step 4: Remove subset schemas")
    print("  No schema is a subset of another. All retained.")
    print()

    # Final result
    print("Final 3NF Decomposition:")
    for name, attrs, fd, key in schemas:
        print(f"  {name}({','.join(sorted(attrs))}) -- key: {format_attrs(key)}")
    print()
    print("  Properties: Lossless-join? YES, Dependency-preserving? YES")


# === Exercise 3: BCNF Decomposition ===
# Problem: Decompose R(A,B,C,D) with F = {AB->C, C->A, C->D} into BCNF.

def exercise_3():
    """BCNF decomposition algorithm."""
    print("R(A, B, C, D)")
    print("F = { AB->C, C->A, C->D }")
    print()

    all_attrs = frozenset("ABCD")
    fds = [
        (frozenset("AB"), frozenset("C")),
        (frozenset("C"),  frozenset("A")),
        (frozenset("C"),  frozenset("D")),
    ]

    # Find candidate keys
    keys = find_candidate_keys(all_attrs, fds)
    print(f"Candidate keys: {[format_attrs(k) for k in keys]}")

    # Verify
    for k in keys:
        cl = attribute_closure(k, fds)
        print(f"  {format_attrs(k)}+ = {format_attrs(cl)}")
    print()

    # Check BCNF violations
    print("Check BCNF violations:")
    for lhs, rhs in fds:
        is_sk = is_superkey(lhs, all_attrs, fds)
        status = "OK (superkey)" if is_sk else "VIOLATION!"
        print(f"  {format_fd(lhs, rhs)}: {format_attrs(lhs)} superkey? {is_sk} -> {status}")
    print()

    # Decompose on C->A (or C->AD)
    print("Decompose on C->A (C->AD since {C}+ includes A,C,D):")
    c_closure = attribute_closure(frozenset("C"), fds)
    print(f"  {{C}}+ = {format_attrs(c_closure)}")
    print(f"  R1 = {{C}}+ = (A, C, D) with key {{C}}")
    print(f"  R2 = {{C}} union (R - {{C}}+) ∪ {{C}} = (B, C) with key {{B,C}}")
    print()

    # Check R1 BCNF
    print("Check R1(A,C,D) with FDs projected: C->A, C->D")
    print("  C is the key. All FDs have C on LHS. BCNF holds.")
    print()

    # Check R2 BCNF
    print("Check R2(B,C) -- no non-trivial FDs with determinant that's not a superkey")
    print("  BCNF holds.")
    print()

    print("BCNF Decomposition: R1(A, C, D) and R2(B, C)")
    print("Note: AB->C is NOT preserved (requires joining R1 and R2 to check).")
    print("This is the classic BCNF trade-off: guaranteed lossless-join, but may lose dependency preservation.")


# === Exercise 4: Lossless-Join Verification (Chase Test) ===
# Problem: Verify if R1(A,B), R2(A,C), R3(B,D) is a lossless decomposition of R(A,B,C,D).

def exercise_4():
    """Chase test for lossless-join verification."""
    print("R(A, B, C, D) with F = { A->B, B->C }")
    print("Decomposition: R1(A,B), R2(A,C), R3(B,D)")
    print()

    # Initial chase matrix
    # Rows = relations, Columns = attributes
    # Distinguished symbol: 'a_j', Subscripted: 'b_ij'
    attrs = ['A', 'B', 'C', 'D']
    relations = {
        'R1': {'A': 'a', 'B': 'a', 'C': 'b13', 'D': 'b14'},
        'R2': {'A': 'a', 'B': 'b22', 'C': 'a', 'D': 'b24'},
        'R3': {'A': 'b31', 'B': 'a', 'C': 'b33', 'D': 'a'},
    }

    def print_matrix(matrix, step_name):
        print(f"  {step_name}:")
        header = "  " + " | ".join(f"{a:>6}" for a in [''] + attrs)
        print(header)
        print("  " + "-" * len(header))
        for rname in ['R1', 'R2', 'R3']:
            row = " | ".join(f"{matrix[rname][a]:>6}" for a in attrs)
            print(f"  {rname:>6} | {row}")
        print()

    print_matrix(relations, "Initial matrix")

    # Apply A -> B: R1 and R2 agree on A (both 'a')
    print("  Apply A->B: R1.A = R2.A = 'a'. Set R2.B = 'a' (distinguished).")
    relations['R2']['B'] = 'a'
    print_matrix(relations, "After A->B")

    # Apply B -> C: R1, R2, R3 all have B = 'a'
    print("  Apply B->C: R1.B = R2.B = R3.B = 'a'. All agree on B.")
    print("  C values: R1=b13, R2=a, R3=b33. Has distinguished 'a'. Set all to 'a'.")
    relations['R1']['C'] = 'a'
    relations['R3']['C'] = 'a'
    print_matrix(relations, "After B->C")

    # Check for complete row
    print("  Check for a row with all distinguished symbols:")
    for rname in ['R1', 'R2', 'R3']:
        all_dist = all(v == 'a' for v in relations[rname].values())
        non_dist = [a for a, v in relations[rname].items() if v != 'a']
        print(f"    {rname}: {'ALL distinguished -> LOSSLESS' if all_dist else f'Missing: {non_dist}'}")

    print()
    print("  Result: NO row has all distinguished symbols. Decomposition is NOT lossless-join.")
    print()
    print("  Correct decomposition: R1(A,B,C) and R2(A,D)")
    print("  Verification: R1 ∩ R2 = {A}. {A}+ = {A,B,C}. A is key of R1. Lossless!")


# === Exercise 5: Full Normalization (Library) ===
# Problem: Normalize the Library relation to 3NF.

def exercise_5():
    """Full normalization of Library relation using 3NF synthesis."""
    print("Library(isbn, title, author_id, author_name, publisher_id,")
    print("        publisher_name, publisher_city, branch_id, branch_name, copies)")
    print()
    print("FDs:")
    fds_desc = [
        "isbn -> title, author_id, publisher_id",
        "author_id -> author_name",
        "publisher_id -> publisher_name, publisher_city",
        "{isbn, branch_id} -> copies",
        "branch_id -> branch_name"
    ]
    for fd in fds_desc:
        print(f"  {fd}")
    print()

    # Verify with algorithm
    all_attrs = frozenset(["isbn", "title", "author_id", "author_name",
                           "publisher_id", "publisher_name", "publisher_city",
                           "branch_id", "branch_name", "copies"])
    fds = [
        (frozenset(["isbn"]), frozenset(["title", "author_id", "publisher_id"])),
        (frozenset(["author_id"]), frozenset(["author_name"])),
        (frozenset(["publisher_id"]), frozenset(["publisher_name", "publisher_city"])),
        (frozenset(["isbn", "branch_id"]), frozenset(["copies"])),
        (frozenset(["branch_id"]), frozenset(["branch_name"])),
    ]

    keys = find_candidate_keys(all_attrs, fds)
    print(f"Candidate key: {[format_attrs(k) for k in keys]}")
    print()

    # 3NF decomposition
    decomposition = [
        ("Book", ["isbn", "title", "author_id", "publisher_id"], "{isbn}"),
        ("Author", ["author_id", "author_name"], "{author_id}"),
        ("Publisher", ["publisher_id", "publisher_name", "publisher_city"], "{publisher_id}"),
        ("BranchStock", ["isbn", "branch_id", "copies"], "{isbn, branch_id}"),
        ("Branch", ["branch_id", "branch_name"], "{branch_id}"),
    ]

    print("3NF Decomposition:")
    for name, attrs, key in decomposition:
        print(f"  {name}({', '.join(attrs)}) -- key: {key}")

    print()
    print("Properties:")
    print("  - Lossless-join: YES (BranchStock contains the candidate key)")
    print("  - Dependency-preserving: YES (all FDs preserved within single tables)")
    print("  - Also in BCNF (every determinant is a key)")


# === Exercise 6: Anomaly Identification ===
# Problem: Identify update, insertion, and deletion anomalies.

def exercise_6():
    """Identify anomalies in a denormalized CourseSection table."""
    print("CourseSection(course_id, section, semester, instructor, building, room, capacity)")
    print("FDs: {course_id, section, semester} -> instructor, building, room")
    print("     {building, room} -> capacity")
    print()

    # Sample data
    data = [
        ("CS101", 1, "Fall24", "Dr. Smith", "Watson", 101, 50),
        ("CS101", 2, "Fall24", "Dr. Jones", "Watson", 101, 50),
        ("CS201", 1, "Fall24", "Dr. Smith", "Watson", 201, 30),
        ("CS201", 1, "Spr25",  "Dr. Smith", "Taylor", 105, 40),
    ]

    header = f"{'course_id':>10} {'section':>7} {'semester':>8} {'instructor':>12} {'building':>8} {'room':>4} {'capacity':>8}"
    print(header)
    print("-" * len(header))
    for row in data:
        print(f"{row[0]:>10} {row[1]:>7} {row[2]:>8} {row[3]:>12} {row[4]:>8} {row[5]:>4} {row[6]:>8}")
    print()

    anomalies = [
        {
            "type": "UPDATE ANOMALY",
            "example": "If Watson 101 capacity changes (renovation), rows 1 and 2 must both be updated.",
            "risk": "Updating only row 1 creates inconsistency: Watson 101 appears to have two capacities."
        },
        {
            "type": "INSERTION ANOMALY",
            "example": "Cannot record Taylor 302 capacity=60 without a course section scheduled there.",
            "risk": "Room information cannot exist independently of course assignments."
        },
        {
            "type": "DELETION ANOMALY",
            "example": "Deleting CS201 Section 1 Spring 2025 (row 4) loses Taylor 105 capacity=40.",
            "risk": "Room information is lost when the only course section using it is removed."
        }
    ]

    for a in anomalies:
        print(f"{a['type']}:")
        print(f"  Example: {a['example']}")
        print(f"  Risk: {a['risk']}")
        print()

    print("Root cause: Transitive dependency")
    print("  {course_id, section, semester} -> {building, room} -> capacity")
    print()
    print("Fix: Decompose into:")
    print("  CourseSection(course_id, section, semester, instructor, building, room) -- key: {course_id, section, semester}")
    print("  Room(building, room, capacity) -- key: {building, room}")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Identifying Normal Forms ===")
    print("=" * 70)
    exercise_1()

    print("=" * 70)
    print("=== Exercise 2: 3NF Synthesis ===")
    print("=" * 70)
    exercise_2()

    print("=" * 70)
    print("=== Exercise 3: BCNF Decomposition ===")
    print("=" * 70)
    exercise_3()

    print("=" * 70)
    print("=== Exercise 4: Lossless-Join Verification (Chase Test) ===")
    print("=" * 70)
    exercise_4()

    print("=" * 70)
    print("=== Exercise 5: Full Normalization (Library) ===")
    print("=" * 70)
    exercise_5()

    print("=" * 70)
    print("=== Exercise 6: Anomaly Identification ===")
    print("=" * 70)
    exercise_6()

    print("\nAll exercises completed!")
