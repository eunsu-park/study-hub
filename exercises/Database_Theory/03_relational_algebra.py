"""
Exercises for Lesson 03: Relational Algebra
Topic: Database_Theory

Solutions to practice problems from the lesson.
Implements relational algebra operations as Python functions on in-memory relations.
"""

from itertools import product as cartesian_product
from collections import defaultdict


# ============================================================
# Sample Database (matching the lesson's example)
# ============================================================

STUDENT = [
    {"sid": "S001", "name": "Alice", "dept": "CS", "year": 3, "gpa": 3.8},
    {"sid": "S002", "name": "Bob",   "dept": "CS", "year": 2, "gpa": 3.2},
    {"sid": "S003", "name": "Carol", "dept": "EE", "year": 4, "gpa": 3.9},
    {"sid": "S004", "name": "Dave",  "dept": "ME", "year": 1, "gpa": 2.8},
    {"sid": "S005", "name": "Eve",   "dept": "CS", "year": 3, "gpa": 3.5},
]

COURSE = [
    {"cid": "CS101", "title": "Intro to CS",        "dept": "CS", "credits": 3},
    {"cid": "CS301", "title": "Database Theory",     "dept": "CS", "credits": 4},
    {"cid": "EE201", "title": "Circuits",            "dept": "EE", "credits": 3},
    {"cid": "ME101", "title": "Statics",             "dept": "ME", "credits": 3},
    {"cid": "CS201", "title": "Data Structures",     "dept": "CS", "credits": 4},
]

ENROLLMENT = [
    {"sid": "S001", "cid": "CS101", "grade": "A"},
    {"sid": "S001", "cid": "CS301", "grade": "A"},
    {"sid": "S001", "cid": "EE201", "grade": "B"},
    {"sid": "S002", "cid": "CS101", "grade": "B"},
    {"sid": "S002", "cid": "EE201", "grade": "A"},
    {"sid": "S003", "cid": "EE201", "grade": "A"},
    {"sid": "S003", "cid": "CS301", "grade": "B"},
    {"sid": "S004", "cid": "ME101", "grade": "C"},
    {"sid": "S005", "cid": "CS101", "grade": "A"},
    {"sid": "S005", "cid": "CS301", "grade": "B"},
]


# ============================================================
# Relational Algebra Primitives
# ============================================================

def select(relation, predicate):
    """Selection: sigma_{predicate}(relation)"""
    return [t for t in relation if predicate(t)]


def project(relation, attrs):
    """Projection: pi_{attrs}(relation) with duplicate removal."""
    result = []
    seen = set()
    for t in relation:
        projected = tuple(t[a] for a in attrs)
        if projected not in seen:
            seen.add(projected)
            result.append({a: t[a] for a in attrs})
    return result


def natural_join(R, S):
    """Natural join: R |><| S on common attributes."""
    common = set(R[0].keys()) & set(S[0].keys()) if R and S else set()
    result = []
    for r in R:
        for s in S:
            if all(r[c] == s[c] for c in common):
                merged = {**r, **s}
                result.append(merged)
    return result


def union(R, S):
    """Union: R ∪ S (bag semantics with dedup)."""
    result = []
    seen = set()
    for t in R + S:
        key = tuple(sorted(t.items()))
        if key not in seen:
            seen.add(key)
            result.append(t)
    return result


def difference(R, S):
    """Difference: R - S."""
    s_keys = {tuple(sorted(t.items())) for t in S}
    return [t for t in R if tuple(sorted(t.items())) not in s_keys]


def intersection(R, S):
    """Intersection: R ∩ S."""
    s_keys = {tuple(sorted(t.items())) for t in S}
    return [t for t in R if tuple(sorted(t.items())) in s_keys]


def division(R, S):
    """Division: R ÷ S.
    Given R(X, Y) and S(Y), returns tuples in pi_X(R) such that
    for every tuple in S, the combination exists in R.
    """
    if not R or not S:
        return []
    r_attrs = set(R[0].keys())
    s_attrs = set(S[0].keys())
    x_attrs = r_attrs - s_attrs
    y_attrs = s_attrs

    # Build a mapping: x_tuple -> set of y_tuples
    x_to_ys = defaultdict(set)
    for t in R:
        x_key = tuple(t[a] for a in sorted(x_attrs))
        y_key = tuple(t[a] for a in sorted(y_attrs))
        x_to_ys[x_key].add(y_key)

    # All y-tuples in S
    all_s = {tuple(t[a] for a in sorted(y_attrs)) for t in S}

    result = []
    for x_key, y_set in x_to_ys.items():
        if all_s.issubset(y_set):
            result.append(dict(zip(sorted(x_attrs), x_key)))
    return result


def print_relation(name, relation, max_rows=20):
    """Pretty-print a relation."""
    if not relation:
        print(f"  {name}: (empty)")
        return
    attrs = list(relation[0].keys())
    header = " | ".join(f"{a:>10}" for a in attrs)
    print(f"  {name}:")
    print(f"  {header}")
    print(f"  {'-' * len(header)}")
    for t in relation[:max_rows]:
        row = " | ".join(f"{str(t[a]):>10}" for a in attrs)
        print(f"  {row}")
    if len(relation) > max_rows:
        print(f"  ... ({len(relation)} rows total)")
    print()


# ============================================================
# Exercise Solutions
# ============================================================

# === Exercise 3.1: Basic Operations ===
# Problem: Write relational algebra expressions using the sample database.

def exercise_3_1():
    """Basic relational algebra operations."""

    # (a) All students in year 2 or year 3
    print("(a) sigma_{year=2 OR year=3}(STUDENT)")
    result = select(STUDENT, lambda t: t["year"] in (2, 3))
    print_relation("Result", result)

    # (b) Course titles with 4 or more credits
    print("(b) pi_{title}(sigma_{credits >= 4}(COURSE))")
    result = project(select(COURSE, lambda t: t["credits"] >= 4), ["title"])
    print_relation("Result", result)

    # (c) Student IDs enrolled in EE201
    print("(c) pi_{sid}(sigma_{cid='EE201'}(ENROLLMENT))")
    result = project(select(ENROLLMENT, lambda t: t["cid"] == "EE201"), ["sid"])
    print_relation("Result", result)

    # (d) Names of students NOT in the CS department
    print("(d) pi_{name}(sigma_{dept != 'CS'}(STUDENT))")
    result = project(select(STUDENT, lambda t: t["dept"] != "CS"), ["name"])
    print_relation("Result", result)


# === Exercise 3.2: Step-by-Step Evaluation ===
# Problem: Evaluate expressions showing intermediate results.

def exercise_3_2():
    """Evaluate relational algebra expressions step by step."""

    # (a) pi_{name}(sigma_{year > 2}(STUDENT))
    print("(a) pi_{name}(sigma_{year > 2}(STUDENT))")
    step1 = select(STUDENT, lambda t: t["year"] > 2)
    print("  Step 1: sigma_{year > 2}(STUDENT)")
    print_relation("Intermediate", step1)
    step2 = project(step1, ["name"])
    print("  Step 2: pi_{name}(result)")
    print_relation("Final", step2)

    # (b) pi_{sid}(sigma_{grade='A'}(ENROLLMENT)) ∩ pi_{sid}(sigma_{dept='CS'}(STUDENT))
    print("(b) pi_{sid}(sigma_{grade='A'}(ENROLLMENT)) ∩ pi_{sid}(sigma_{dept='CS'}(STUDENT))")
    left = project(select(ENROLLMENT, lambda t: t["grade"] == "A"), ["sid"])
    print("  Left: Students with at least one A grade")
    print_relation("Left", left)
    right = project(select(STUDENT, lambda t: t["dept"] == "CS"), ["sid"])
    print("  Right: CS department students")
    print_relation("Right", right)
    result = intersection(left, right)
    print("  Intersection: CS students who earned an A")
    print_relation("Final", result)

    # (c) STUDENT |><| (sigma_{cid='CS301'}(ENROLLMENT))
    print("(c) STUDENT |><| sigma_{cid='CS301'}(ENROLLMENT)")
    filtered_enroll = select(ENROLLMENT, lambda t: t["cid"] == "CS301")
    print("  Step 1: sigma_{cid='CS301'}(ENROLLMENT)")
    print_relation("Filtered", filtered_enroll)
    result = natural_join(STUDENT, filtered_enroll)
    print("  Step 2: Natural join with STUDENT")
    print_relation("Final", result)


# === Exercise 3.3: Join with SQL Equivalents ===
# Problem: Relational algebra expressions AND equivalent SQL.

def exercise_3_3():
    """Join operations with SQL equivalents."""

    # (a) Names of students enrolled in "Database Theory"
    print('(a) Names of students enrolled in "Database Theory"')
    print("  RA: pi_{name}(STUDENT |><| ENROLLMENT |><| sigma_{title='Database Theory'}(COURSE))")
    print("  SQL: SELECT s.name FROM student s")
    print("       JOIN enrollment e ON s.sid = e.sid")
    print("       JOIN course c ON e.cid = c.cid")
    print("       WHERE c.title = 'Database Theory';")
    db_theory = select(COURSE, lambda t: t["title"] == "Database Theory")
    joined = natural_join(natural_join(STUDENT, ENROLLMENT), db_theory)
    result = project(joined, ["name"])
    print_relation("Result", result)

    # (b) Course titles with at least one A grade
    print("(b) Course titles with at least one student with grade A")
    print("  RA: pi_{title}(COURSE |><| sigma_{grade='A'}(ENROLLMENT))")
    print("  SQL: SELECT DISTINCT c.title FROM course c")
    print("       JOIN enrollment e ON c.cid = e.cid WHERE e.grade = 'A';")
    a_enroll = select(ENROLLMENT, lambda t: t["grade"] == "A")
    joined = natural_join(COURSE, a_enroll)
    result = project(joined, ["title"])
    print_relation("Result", result)

    # (c) Students enrolled in courses outside their department
    print("(c) Students enrolled in courses outside their department")
    print("  RA: pi_{name}(sigma_{STUDENT.dept != COURSE.dept}(STUDENT |><| ENROLLMENT |><| COURSE))")
    print("  SQL: SELECT DISTINCT s.name FROM student s")
    print("       JOIN enrollment e ON s.sid = e.sid")
    print("       JOIN course c ON e.cid = c.cid WHERE s.dept != c.dept;")
    all_joined = natural_join(natural_join(STUDENT, ENROLLMENT), COURSE)
    # Rename dept columns to avoid conflict in natural join
    # In our implementation, natural join merges on 'dept', so we need a workaround
    # Let's manually do this join
    result_rows = []
    for s in STUDENT:
        for e in ENROLLMENT:
            if s["sid"] == e["sid"]:
                for c in COURSE:
                    if e["cid"] == c["cid"] and s["dept"] != c["dept"]:
                        result_rows.append({"name": s["name"]})
    result = project(result_rows, ["name"]) if result_rows else []
    print_relation("Result", result)


# === Exercise 3.5: Division ===
# Problem: Use division to find students enrolled in specific sets of courses.

def exercise_3_5():
    """Division operations."""

    # (a) Students enrolled in every 3-credit course
    print("(a) Students enrolled in every 3-credit course")
    three_credit_courses = project(select(COURSE, lambda t: t["credits"] == 3), ["cid"])
    print("  3-credit courses:")
    print_relation("S (divisor)", three_credit_courses)

    student_course = project(ENROLLMENT, ["sid", "cid"])
    print("  Student-Course pairs (R):")
    print_relation("R (dividend)", student_course)

    result = division(student_course, three_credit_courses)
    print("  R / S = Students enrolled in ALL 3-credit courses:")
    print_relation("Result", result)
    print("  Verification: Each result student must be in CS101, EE201, and ME101.")

    # (b) Students who have taken ALL courses that Bob (S002) has taken
    print("(b) Students who have taken ALL courses that Bob (S002) has taken")
    bob_courses = project(select(ENROLLMENT, lambda t: t["sid"] == "S002"), ["cid"])
    print("  Bob's courses:")
    print_relation("S (divisor)", bob_courses)

    result = division(student_course, bob_courses)
    print("  Students who took all of Bob's courses:")
    print_relation("Result", result)


# === Exercise 3.8: Prove/Disprove Equivalences ===
# Problem: Verify algebraic equivalences with concrete examples.

def exercise_3_8():
    """Prove or disprove relational algebra equivalences."""
    R = [{"a": 1, "b": 10}, {"a": 2, "b": 20}, {"a": 3, "b": 30}]
    S = [{"a": 2, "b": 20}, {"a": 3, "b": 30}, {"a": 4, "b": 40}]

    # (a) sigma_{c1}(R ∪ S) = sigma_{c1}(R) ∪ sigma_{c1}(S)
    print("(a) sigma_{a>2}(R ∪ S) ?= sigma_{a>2}(R) ∪ sigma_{a>2}(S)")
    cond = lambda t: t["a"] > 2
    lhs = select(union(R, S), cond)
    rhs = union(select(R, cond), select(S, cond))
    lhs_set = {tuple(sorted(t.items())) for t in lhs}
    rhs_set = {tuple(sorted(t.items())) for t in rhs}
    print(f"  LHS: {[dict(t) for t in sorted(lhs_set)]}")
    print(f"  RHS: {[dict(t) for t in sorted(rhs_set)]}")
    print(f"  Equal: {lhs_set == rhs_set} -- TRUE: Selection distributes over union.\n")

    # (b) sigma_{c1}(R - S) = sigma_{c1}(R) - sigma_{c1}(S)
    print("(b) sigma_{a>2}(R - S) ?= sigma_{a>2}(R) - sigma_{a>2}(S)")
    lhs = select(difference(R, S), cond)
    rhs = difference(select(R, cond), select(S, cond))
    lhs_set = {tuple(sorted(t.items())) for t in lhs}
    rhs_set = {tuple(sorted(t.items())) for t in rhs}
    print(f"  LHS: {[dict(t) for t in sorted(lhs_set)]}")
    print(f"  RHS: {[dict(t) for t in sorted(rhs_set)]}")
    print(f"  Equal: {lhs_set == rhs_set} -- TRUE: Selection distributes over difference.\n")

    # (c) pi_A(R ∪ S) = pi_A(R) ∪ pi_A(S)
    print("(c) pi_{a}(R ∪ S) ?= pi_{a}(R) ∪ pi_{a}(S)")
    lhs = project(union(R, S), ["a"])
    rhs = union(project(R, ["a"]), project(S, ["a"]))
    lhs_set = {tuple(sorted(t.items())) for t in lhs}
    rhs_set = {tuple(sorted(t.items())) for t in rhs}
    print(f"  LHS: {sorted([t['a'] for t in lhs])}")
    print(f"  RHS: {sorted([t['a'] for t in rhs])}")
    print(f"  Equal: {lhs_set == rhs_set} -- TRUE: Projection distributes over union.\n")

    # (d) sigma_{c1}(R × S) = sigma_{c1}(R) × S  (where c1 involves only R's attributes)
    print("(d) sigma_{c1}(R x S) ?= sigma_{c1}(R) x S  where c1 uses only R's attrs")
    # Use different schema to avoid name conflicts
    R2 = [{"x": 1}, {"x": 2}, {"x": 3}]
    S2 = [{"y": 10}, {"y": 20}]
    cond2 = lambda t: t["x"] > 1

    # R x S (cartesian product)
    cross_rs = [{"x": r["x"], "y": s["y"]} for r in R2 for s in S2]
    lhs = select(cross_rs, cond2)

    filtered_r = select(R2, cond2)
    cross_filtered = [{"x": r["x"], "y": s["y"]} for r in filtered_r for s in S2]
    rhs = cross_filtered

    lhs_set = {tuple(sorted(t.items())) for t in lhs}
    rhs_set = {tuple(sorted(t.items())) for t in rhs}
    print(f"  LHS: {sorted(lhs, key=lambda t: (t['x'], t['y']))}")
    print(f"  RHS: {sorted(rhs, key=lambda t: (t['x'], t['y']))}")
    print(f"  Equal: {lhs_set == rhs_set} -- TRUE: Selection on R's attrs pushes through cross product.\n")


# === Exercise 3.11: Division Computation ===
# Problem: Compute R / S step by step.

def exercise_3_11():
    """Step-by-step division computation."""
    R = [
        {"A": "a1", "B": "b1", "C": "c1"},
        {"A": "a1", "B": "b2", "C": "c1"},
        {"A": "a1", "B": "b2", "C": "c2"},
        {"A": "a2", "B": "b1", "C": "c1"},
        {"A": "a2", "B": "b1", "C": "c2"},
    ]
    S = [
        {"B": "b1", "C": "c1"},
        {"B": "b1", "C": "c2"},
    ]

    print("R(A, B, C):")
    print_relation("R", R)
    print("S(B, C):")
    print_relation("S", S)

    # Step 1: pi_A(R) = {a1, a2}
    pi_a = project(R, ["A"])
    print("Step 1: pi_A(R) = all distinct A values")
    print_relation("pi_A(R)", pi_a)

    # Step 2: For each A value, find the (B,C) pairs
    print("Step 2: For each A value, find associated (B,C) pairs:")
    from collections import defaultdict
    a_to_bc = defaultdict(set)
    for t in R:
        a_to_bc[t["A"]].add((t["B"], t["C"]))

    s_set = {(t["B"], t["C"]) for t in S}
    print(f"  S as set of (B,C): {s_set}")

    for a_val in sorted(a_to_bc.keys()):
        bc_set = a_to_bc[a_val]
        contains_all = s_set.issubset(bc_set)
        print(f"  A={a_val}: (B,C) pairs = {bc_set}")
        print(f"    Contains all of S? {contains_all}")

    # Step 3: Result
    result = division(R, S)
    print("\nR / S result:")
    print_relation("Result", result)

    # Verification
    print("Verification:")
    for t in result:
        a_val = t["A"]
        for s_tuple in S:
            check = any(
                r["A"] == a_val and r["B"] == s_tuple["B"] and r["C"] == s_tuple["C"]
                for r in R
            )
            print(f"  ({a_val}, {s_tuple['B']}, {s_tuple['C']}) in R? {check}")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 3.1: Basic Operations ===")
    print("=" * 70)
    exercise_3_1()

    print("=" * 70)
    print("=== Exercise 3.2: Step-by-Step Evaluation ===")
    print("=" * 70)
    exercise_3_2()

    print("=" * 70)
    print("=== Exercise 3.3: Join with SQL Equivalents ===")
    print("=" * 70)
    exercise_3_3()

    print("=" * 70)
    print("=== Exercise 3.5: Division ===")
    print("=" * 70)
    exercise_3_5()

    print("=" * 70)
    print("=== Exercise 3.8: Prove/Disprove Equivalences ===")
    print("=" * 70)
    exercise_3_8()

    print("=" * 70)
    print("=== Exercise 3.11: Division Computation ===")
    print("=" * 70)
    exercise_3_11()

    print("\nAll exercises completed!")
