"""
Exercises for Lesson 07: Advanced Normalization
Topic: Database_Theory

Solutions to practice problems from the lesson.
Covers MVDs, 4NF decomposition, 5NF/PJNF, star schema design,
and denormalization decisions.
"""


# === Exercise 1: Identifying MVDs ===
# Problem: Given R(student, course, hobby), identify MVDs, normal form, decompose to 4NF.

def exercise_1():
    """Identify MVDs and decompose to 4NF."""
    print("R(student, course, hobby)")
    print("Constraint: A student's courses are independent of their hobbies.")
    print()

    # Demonstrate the MVD with sample data
    data = [
        ("Alice", "Math",    "Chess"),
        ("Alice", "Math",    "Piano"),
        ("Alice", "Physics", "Chess"),
        ("Alice", "Physics", "Piano"),
        ("Bob",   "CS",      "Hiking"),
    ]

    print("Sample data (satisfying the MVDs):")
    print(f"  {'student':>8} {'course':>8} {'hobby':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8}")
    for row in data:
        print(f"  {row[0]:>8} {row[1]:>8} {row[2]:>8}")
    print()

    print("1. MVDs that hold:")
    print("   student ->> course")
    print("   student ->> hobby  (complementation rule)")
    print("   For Alice: courses={Math,Physics} and hobbies={Chess,Piano}")
    print("   are independent -- all 2x2=4 combinations must exist.")
    print()

    print("2. Highest normal form:")
    print("   Key: {student, course, hobby} (all three attributes)")
    print("   No non-trivial FDs exist, so BCNF holds.")
    print("   But student ->> course has {student} as determinant, which is NOT a superkey.")
    print("   Therefore: BCNF but NOT 4NF.")
    print()

    print("3. 4NF Decomposition:")
    print("   Decompose on student ->> course:")
    r1 = set()
    r2 = set()
    for s, c, h in data:
        r1.add((s, c))
        r2.add((s, h))

    print("   R1(student, course):")
    for row in sorted(r1):
        print(f"     {row}")

    print("   R2(student, hobby):")
    for row in sorted(r2):
        print(f"     {row}")

    print()
    print("   Both R1 and R2 are in 4NF (only trivial MVDs remain).")

    # Verify lossless-join: R1 |><| R2 should reconstruct R
    reconstructed = set()
    for s1, c in r1:
        for s2, h in r2:
            if s1 == s2:
                reconstructed.add((s1, c, h))
    original = set(data)
    print(f"   Lossless verification: R1 |><| R2 = original? {reconstructed == original}")


# === Exercise 2: MVD vs FD ===
# Problem: Distinguish between MVD and FD.

def exercise_2():
    """Distinguish MVD from FD with concrete example."""
    print("R(A, B, C)")
    print("Constraint: For each value of A, the set of B-values is fixed regardless of C-values.")
    print()

    # Demonstrate with data
    data = [
        ("a1", "b1", "c1"),
        ("a1", "b1", "c2"),
        ("a1", "b2", "c1"),
        ("a1", "b2", "c2"),
    ]

    print("Example data:")
    print(f"  {'A':>4} {'B':>4} {'C':>4}")
    for row in data:
        print(f"  {row[0]:>4} {row[1]:>4} {row[2]:>4}")
    print()

    print("1. Is this an FD or an MVD?")
    print("   This is an MVD. The constraint says B-values and C-values are")
    print("   independent given A -- the hallmark of a multivalued dependency.")
    print()

    print("2. Does A -> B hold?")
    # Check: is there a single B for each A?
    a_to_bs = {}
    for a, b, c in data:
        a_to_bs.setdefault(a, set()).add(b)
    has_fd = all(len(bs) == 1 for bs in a_to_bs.values())
    print(f"   A -> B holds? {has_fd}")
    print(f"   A maps to B-values: {dict(a_to_bs)}")
    print("   A -> B does NOT hold: a1 maps to both b1 AND b2.")
    print()

    print("3. Does A ->> B hold?")
    print("   YES. For A=a1, the B-values {b1,b2} appear with EVERY C-value {c1,c2}.")
    print("   This is exactly the definition of A ->> B.")
    print()

    print("Key insight: Every FD is an MVD (A->B implies A->>B),")
    print("but NOT every MVD is an FD (A->>B does NOT imply A->B).")


# === Exercise 3: Star Schema Design ===
# Problem: Design a star schema for a library lending system.

def exercise_3():
    """Design a star schema for library checkouts."""
    print("=== Star Schema: Library Lending System ===\n")

    print("Fact Table: fact_checkout")
    fact_cols = [
        ("checkout_key", "SERIAL PRIMARY KEY", "Surrogate key"),
        ("date_key", "INT FK -> dim_date", "When the checkout happened"),
        ("book_key", "INT FK -> dim_book", "Which book was checked out"),
        ("patron_key", "INT FK -> dim_patron", "Who checked it out"),
        ("branch_key", "INT FK -> dim_branch", "Which branch"),
        ("days_borrowed", "INT", "Measure: duration of loan"),
        ("is_returned", "BOOLEAN", "Measure: return status"),
        ("late_fee", "DECIMAL(6,2)", "Measure: late fee amount"),
    ]
    for col, dtype, desc in fact_cols:
        print(f"  {col:<20} {dtype:<25} -- {desc}")
    print()

    # Dimension tables
    dimensions = {
        "dim_date": [
            ("date_key", "INT PK"),
            ("full_date", "DATE"),
            ("year", "INT"),
            ("quarter", "INT"),
            ("month", "INT"),
            ("day_of_week", "VARCHAR(10)"),
            ("is_weekend", "BOOLEAN"),
            ("is_holiday", "BOOLEAN"),
        ],
        "dim_book": [
            ("book_key", "INT PK"),
            ("isbn", "VARCHAR(20)"),
            ("title", "VARCHAR(200)"),
            ("author", "VARCHAR(100)"),
            ("genre", "VARCHAR(50)"),
            ("publisher", "VARCHAR(100)"),
            ("pub_year", "INT"),
        ],
        "dim_patron": [
            ("patron_key", "INT PK"),
            ("patron_id", "VARCHAR(20)"),
            ("name", "VARCHAR(100)"),
            ("membership_type", "VARCHAR(20)"),
            ("city", "VARCHAR(50)"),
            ("join_date", "DATE"),
        ],
        "dim_branch": [
            ("branch_key", "INT PK"),
            ("branch_name", "VARCHAR(100)"),
            ("city", "VARCHAR(50)"),
            ("state", "VARCHAR(2)"),
            ("manager_name", "VARCHAR(100)"),
        ]
    }

    for dim_name, cols in dimensions.items():
        print(f"Dimension: {dim_name}")
        for col, dtype in cols:
            print(f"  {col:<20} {dtype}")
        print()

    print("Sample SQL Query: Most popular genres by month in 2025")
    print("""
  SELECT dd.year, dd.month, db.genre, COUNT(*) AS checkouts
  FROM fact_checkout f
  JOIN dim_date dd ON f.date_key = dd.date_key
  JOIN dim_book db ON f.book_key = db.book_key
  WHERE dd.year = 2025
  GROUP BY dd.year, dd.month, db.genre
  ORDER BY dd.month, checkouts DESC;
""")


# === Exercise 4: 4NF Decomposition ===
# Problem: Decompose R(A,B,C,D) with A->>B, A->C into 4NF.

def exercise_4():
    """4NF decomposition with both MVD and FD."""
    print("R(A, B, C, D)")
    print("Dependencies: A ->> B, A -> C")
    print()

    print("Step 1: Identify all dependencies")
    print("  A ->> B implies A ->> CD (complementation: R - A - B = {C,D})")
    print("  A -> C is an FD (which implies A ->> C)")
    print()

    print("Step 2: Find key")
    print("  A determines C (via FD). B and D are not functionally determined.")
    print("  Key must include A, B, D: candidate key = {A, B, D}")
    print()

    print("Step 3: Check 4NF")
    print("  A ->> B is non-trivial, and {A} is NOT a superkey.")
    print("  VIOLATION! Decompose on A ->> B:")
    print()

    print("  R1(A, B) -- key: {A, B}")
    print("  R2(A, C, D) -- key: {A, D} (since A -> C)")
    print()

    print("Step 4: Check R2 for BCNF/4NF")
    print("  A -> C in R2: {A}+ = {A, C}. A is NOT a superkey of R2 (key is {A,D}).")
    print("  BCNF violation! Decompose R2:")
    print("  R3(A, C) -- key: {A}")
    print("  R4(A, D) -- key: {A, D}")
    print()

    print("Final 4NF decomposition:")
    final = [
        ("R1", "A, B", "{A, B}"),
        ("R3", "A, C", "{A}"),
        ("R4", "A, D", "{A, D}"),
    ]
    for name, attrs, key in final:
        print(f"  {name}({attrs}) -- key: {key}")
    print()
    print("  All in 4NF (only trivial MVDs in each).")


# === Exercise 5: Denormalization Decision ===
# Problem: For each scenario, normalize or denormalize?

def exercise_5():
    """Denormalization decision analysis."""
    scenarios = [
        {
            "scenario": "1. E-commerce product catalog: 10M products, read:write = 1000:1",
            "decision": "DENORMALIZE",
            "reasoning": [
                "Overwhelming read dominance (1000:1) justifies redundancy.",
                "Embed category name, brand name directly in product table.",
                "Use materialized views for search/filter pages.",
                "Writes are infrequent; batch process updates to denormalized columns.",
                "Key metric: 99th percentile read latency matters more than write simplicity."
            ]
        },
        {
            "scenario": "2. Banking transaction system: wire transfers",
            "decision": "NORMALIZE (BCNF)",
            "reasoning": [
                "Financial data demands perfect consistency -- every cent must balance.",
                "ACID transactions are essential; correctness > performance.",
                "The cost of joins is acceptable for financial accuracy.",
                "Regulatory requirements (SOX, PCI-DSS) favor normalized, auditable schemas.",
                "Anomalies (update/insert/delete) could cause financial errors."
            ]
        },
        {
            "scenario": "3. Social media news feed: posts with author names/avatars",
            "decision": "DENORMALIZE",
            "reasoning": [
                "News feeds are extremely read-heavy and latency-sensitive.",
                "Store author name and avatar URL directly in feed items.",
                "Accept eventual consistency: old posts can show stale profile data.",
                "Use Redis or similar cache layer for hot data.",
                "Fan-out-on-write pattern: denormalize at write time for read performance."
            ]
        },
        {
            "scenario": "4. Scientific sensor data collection: readings every second",
            "decision": "HYBRID",
            "reasoning": [
                "Sensor metadata (ID, location, type) -> NORMALIZE (rarely changes).",
                "Time-series readings -> DENORMALIZE or use specialized storage.",
                "Wide table partitioned by time (or use TimescaleDB/InfluxDB).",
                "Key constraint is write throughput, not join complexity.",
                "Aggregation queries benefit from columnar storage or pre-computed rollups."
            ]
        }
    ]

    for s in scenarios:
        print(f"{s['scenario']}")
        print(f"  Decision: {s['decision']}")
        for r in s["reasoning"]:
            print(f"    - {r}")
        print()


# === Exercise 6: Join Dependency ===
# Problem: R(A,B,C) with JD |><|{(A,B), (B,C), (A,C)}, check 4NF and 5NF.

def exercise_6():
    """Join dependency analysis and 5NF decomposition."""
    print("R(A, B, C) with join dependency JD = |><|{(A,B), (B,C), (A,C)}")
    print()

    # Demonstrate with example data
    data = [
        ("a1", "b1", "c1"),
        ("a1", "b2", "c2"),
        ("a2", "b1", "c2"),
    ]

    print("Sample data satisfying the JD:")
    print(f"  {'A':>4} {'B':>4} {'C':>4}")
    for row in data:
        print(f"  {row[0]:>4} {row[1]:>4} {row[2]:>4}")
    print()

    # Project into three components
    r1 = list(set((a, b) for a, b, c in data))
    r2 = list(set((b, c) for a, b, c in data))
    r3 = list(set((a, c) for a, b, c in data))

    print("Projections:")
    print(f"  R1(A,B): {sorted(r1)}")
    print(f"  R2(B,C): {sorted(r2)}")
    print(f"  R3(A,C): {sorted(r3)}")

    # Verify: natural join of R1, R2, R3 should equal original
    reconstructed = set()
    for a1, b1 in r1:
        for b2, c2 in r2:
            if b1 == b2:
                for a3, c3 in r3:
                    if a1 == a3 and c2 == c3:
                        reconstructed.add((a1, b1, c2))

    original = set(data)
    print(f"  R1 |><| R2 |><| R3 = original? {reconstructed == original}")
    print()

    print("1. Is R in 4NF?")
    print("   Possibly yes. The JD is a ternary join dependency, not implied by any MVD.")
    print("   If no non-trivial MVDs exist, the relation is in 4NF.")
    print()

    print("2. Is R in 5NF?")
    print("   No. The JD is non-trivial (not implied by candidate keys).")
    print("   The only key is {A,B,C} (all attributes).")
    print("   {A,B}, {B,C}, and {A,C} are not superkeys.")
    print("   Therefore: NOT in 5NF.")
    print()

    print("3. 5NF Decomposition:")
    print("   R1(A, B) -- key: {A, B}")
    print("   R2(B, C) -- key: {B, C}")
    print("   R3(A, C) -- key: {A, C}")
    print()
    print("   The JD guarantees lossless decomposition: R = R1 |><| R2 |><| R3")
    print("   Each component has only trivial JDs. All in 5NF.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Identifying MVDs ===")
    print("=" * 70)
    exercise_1()

    print("=" * 70)
    print("=== Exercise 2: MVD vs FD ===")
    print("=" * 70)
    exercise_2()

    print("=" * 70)
    print("=== Exercise 3: Star Schema Design ===")
    print("=" * 70)
    exercise_3()

    print("=" * 70)
    print("=== Exercise 4: 4NF Decomposition ===")
    print("=" * 70)
    exercise_4()

    print("=" * 70)
    print("=== Exercise 5: Denormalization Decision ===")
    print("=" * 70)
    exercise_5()

    print("=" * 70)
    print("=== Exercise 6: Join Dependency ===")
    print("=" * 70)
    exercise_6()

    print("\nAll exercises completed!")
