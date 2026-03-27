"""
Exercises for Lesson 02: Relational Model
Topic: Database_Theory

Solutions to practice problems from the lesson.
Covers keys, integrity constraints, NULL handling, and schema design.
"""


# === Exercise 2.2: Key Terminology ===
# Problem: Explain the difference between various types of keys and integrity constraints.

def exercise_2_2():
    """Explain key terminology differences."""
    comparisons = [
        {
            "pair": "(a) Superkey vs Candidate Key",
            "explanation": (
                "A SUPERKEY is any set of attributes that uniquely identifies a tuple. "
                "A CANDIDATE KEY is a minimal superkey — no proper subset of it is also a superkey. "
                "Example: In STUDENT(sid, name, email), {sid}, {email}, {sid, name}, {sid, email} "
                "are all superkeys. But only {sid} and {email} are candidate keys (minimal)."
            )
        },
        {
            "pair": "(b) Primary Key vs Alternate Key",
            "explanation": (
                "When a relation has multiple candidate keys, one is chosen as the PRIMARY KEY "
                "(used for foreign key references, indexing). The remaining candidate keys become "
                "ALTERNATE KEYS. Example: If {sid} is the primary key, {email} is an alternate key."
            )
        },
        {
            "pair": "(c) Natural Key vs Surrogate Key",
            "explanation": (
                "A NATURAL KEY has real-world meaning (e.g., SSN, ISBN, email). "
                "A SURROGATE KEY is system-generated with no business meaning (e.g., auto-increment id, UUID). "
                "Natural keys can change (name change, email change); surrogates are immutable."
            )
        },
        {
            "pair": "(d) Entity Integrity vs Referential Integrity",
            "explanation": (
                "ENTITY INTEGRITY: Primary key attributes cannot be NULL (every tuple must be "
                "uniquely identifiable). "
                "REFERENTIAL INTEGRITY: Every foreign key value must either be NULL or match an "
                "existing primary key value in the referenced relation."
            )
        }
    ]

    for comp in comparisons:
        print(f"{comp['pair']}")
        print(f"  {comp['explanation']}")
        print()


# === Exercise 2.6: Identifying Candidate Keys ===
# Problem: Identify all candidate keys for the FLIGHT relation.

def exercise_2_6():
    """Identify candidate keys for FLIGHT relation."""
    print("FLIGHT(flight_number, airline, departure_city, arrival_city,")
    print("       departure_time, arrival_time, gate, aircraft_id)")
    print()
    print("Constraints:")
    print("  - A flight number uniquely identifies a flight")
    print("  - An aircraft can only be at one gate at a time")
    print("  - A gate can only have one aircraft at a time")
    print()

    candidate_keys = [
        {
            "key": "{flight_number}",
            "reason": "Directly stated: flight_number uniquely identifies a flight."
        },
        {
            "key": "{aircraft_id, departure_time}",
            "reason": "An aircraft can only be at one gate at a time, so aircraft_id + time "
                      "uniquely identifies the flight (an aircraft does one flight at a time)."
        },
        {
            "key": "{gate, departure_time}",
            "reason": "A gate has one aircraft at a time, so gate + time uniquely identifies "
                      "the flight departing from that gate at that time."
        }
    ]

    print("Candidate Keys:")
    for ck in candidate_keys:
        print(f"  {ck['key']}")
        print(f"    Reason: {ck['reason']}")
    print()
    print("Note: {flight_number} is the simplest candidate key (single attribute).")
    print("It would typically be chosen as the primary key.")


# === Exercise 2.7: Integrity Constraint Violations ===
# Problem: Identify and fix integrity constraint violations in SQL inserts.

def exercise_2_7():
    """Identify integrity constraint violations in INSERT statements."""
    inserts = [
        {
            "sql": "INSERT INTO employee VALUES (1, 'Alice', 'CS01', 75000, NULL);",
            "violation": "REFERENTIAL INTEGRITY: dept_id 'CS01' references department(dept_id), "
                        "but department 'CS01' does not exist yet (inserted in next statement).",
            "fix": "Insert department CS01 BEFORE inserting employee 1, or defer constraint checking."
        },
        {
            "sql": "INSERT INTO department VALUES ('CS01', 'Computer Science');",
            "violation": "No violation (assuming executed after reordering).",
            "fix": "Move this INSERT before the first employee insert."
        },
        {
            "sql": "INSERT INTO employee VALUES (2, NULL, 'CS01', 60000, 1);",
            "violation": "NOT NULL CONSTRAINT: 'name' column has NOT NULL constraint, but NULL is provided.",
            "fix": "Provide a valid name: INSERT INTO employee VALUES (2, 'Bob', 'CS01', 60000, 1);"
        },
        {
            "sql": "INSERT INTO employee VALUES (3, 'Carol', 'EE01', 65000, 1);",
            "violation": "REFERENTIAL INTEGRITY: dept_id 'EE01' does not exist in department table.",
            "fix": "First insert department EE01, or change to an existing department."
        },
        {
            "sql": "INSERT INTO employee VALUES (NULL, 'Dave', 'CS01', 55000, 1);",
            "violation": "ENTITY INTEGRITY: emp_id is PRIMARY KEY and cannot be NULL.",
            "fix": "Provide a valid emp_id: INSERT INTO employee VALUES (4, 'Dave', 'CS01', 55000, 1);"
        }
    ]

    print("Analyzing INSERT statements for integrity violations:\n")
    for i, ins in enumerate(inserts, 1):
        print(f"Statement {i}: {ins['sql']}")
        print(f"  Violation: {ins['violation']}")
        print(f"  Fix: {ins['fix']}")
        print()

    print("Corrected order of inserts:")
    corrected = [
        "INSERT INTO department VALUES ('CS01', 'Computer Science');",
        "INSERT INTO department VALUES ('EE01', 'Electrical Engineering');",
        "INSERT INTO employee VALUES (1, 'Alice', 'CS01', 75000, NULL);",
        "INSERT INTO employee VALUES (2, 'Bob', 'CS01', 60000, 1);",
        "INSERT INTO employee VALUES (3, 'Carol', 'EE01', 65000, 1);",
        "INSERT INTO employee VALUES (4, 'Dave', 'CS01', 55000, 1);",
    ]
    for stmt in corrected:
        print(f"  {stmt}")


# === Exercise 2.8: NULL Query Behavior ===
# Problem: Predict the output of queries involving NULLs.

def exercise_2_8():
    """Simulate SQL NULL behavior in Python."""

    # Table t: (a, b)
    t = [(1, 10), (2, None), (3, 30), (None, 40)]

    print("Table t:")
    print("  a   | b")
    print("  ----+-----")
    for a, b in t:
        print(f"  {str(a):4s}| {str(b)}")
    print()

    # (a) SELECT * FROM t WHERE b > 20;
    # NULL > 20 => UNKNOWN => filtered out
    result_a = [(a, b) for a, b in t if b is not None and b > 20]
    print("(a) SELECT * FROM t WHERE b > 20;")
    print(f"    Result: {result_a}")
    print("    Explanation: (3, 30) and (None, 40). Row (2, NULL) excluded because NULL > 20 = UNKNOWN.")
    print()

    # (b) SELECT * FROM t WHERE b > 20 OR b <= 20;
    # For NULL: UNKNOWN OR UNKNOWN = UNKNOWN => filtered out
    result_b = [(a, b) for a, b in t if b is not None]
    print("(b) SELECT * FROM t WHERE b > 20 OR b <= 20;")
    print(f"    Result: {result_b}")
    print("    Explanation: Row (2, NULL) still excluded! NULL > 20 OR NULL <= 20 = UNKNOWN OR UNKNOWN = UNKNOWN.")
    print("    This is NOT the same as 'all rows' -- NULLs break the law of excluded middle.")
    print()

    # (c) SELECT * FROM t WHERE a IN (1, 2, NULL);
    # a IN (1, 2, NULL) is a=1 OR a=2 OR a=NULL
    # For (None, 40): NULL=1 => UNKNOWN, NULL=2 => UNKNOWN, NULL=NULL => UNKNOWN => UNKNOWN
    result_c = [(a, b) for a, b in t if a is not None and a in (1, 2)]
    print("(c) SELECT * FROM t WHERE a IN (1, 2, NULL);")
    print(f"    Result: {result_c}")
    print("    Explanation: Only rows where a=1 or a=2. The NULL in the IN list doesn't match anything,")
    print("    and (NULL, 40) is excluded because NULL=1, NULL=2, NULL=NULL are all UNKNOWN.")
    print()

    # (d) SELECT COUNT(*), COUNT(a), COUNT(b), SUM(b), AVG(b) FROM t;
    count_star = len(t)
    count_a = sum(1 for a, _ in t if a is not None)
    count_b = sum(1 for _, b in t if b is not None)
    sum_b = sum(b for _, b in t if b is not None)
    avg_b = sum_b / count_b if count_b > 0 else None
    print("(d) SELECT COUNT(*), COUNT(a), COUNT(b), SUM(b), AVG(b) FROM t;")
    print(f"    COUNT(*) = {count_star}  (counts all rows including NULLs)")
    print(f"    COUNT(a) = {count_a}  (skips NULL values in column a)")
    print(f"    COUNT(b) = {count_b}  (skips NULL values in column b)")
    print(f"    SUM(b)   = {sum_b}  (10 + 30 + 40 = 80, NULLs ignored)")
    print(f"    AVG(b)   = {avg_b:.4f}  (80 / 3 = 26.67, NULLs ignored in both sum and count)")
    print()

    # (e) SELECT * FROM t WHERE b NOT IN (10, NULL);
    # b NOT IN (10, NULL) = b != 10 AND b != NULL
    # For any b: b != NULL => UNKNOWN, so AND with UNKNOWN => UNKNOWN or FALSE
    # Result: EMPTY!
    print("(e) SELECT * FROM t WHERE b NOT IN (10, NULL);")
    print("    Result: [] (EMPTY!)")
    print("    Explanation: NOT IN with NULL always returns empty set!")
    print("    b NOT IN (10, NULL) = (b != 10) AND (b != NULL)")
    print("    The (b != NULL) part is always UNKNOWN, making the AND result UNKNOWN for every row.")
    print()

    # (f) SELECT COALESCE(a, 0) + COALESCE(b, 0) AS total FROM t;
    result_f = []
    for a, b in t:
        ca = a if a is not None else 0
        cb = b if b is not None else 0
        result_f.append(ca + cb)
    print("(f) SELECT COALESCE(a, 0) + COALESCE(b, 0) AS total FROM t;")
    for i, ((a, b), total) in enumerate(zip(t, result_f)):
        print(f"    ({a}, {b}) -> COALESCE({a}, 0) + COALESCE({b}, 0) = {total}")


# === Exercise 2.9: NULL-Safe SQL Queries ===
# Problem: Write SQL queries that correctly handle NULLs.

def exercise_2_9():
    """SQL queries with proper NULL handling."""
    print("Table: EMPLOYEE(emp_id, name, dept, salary, bonus)")
    print("Note: bonus can be NULL\n")

    queries = [
        {
            "label": "(a) Employees with total compensation > 100,000",
            "sql": (
                "SELECT * FROM employee\n"
                "WHERE salary + COALESCE(bonus, 0) > 100000;"
            ),
            "explanation": (
                "COALESCE(bonus, 0) treats NULL bonus as 0. Without COALESCE, "
                "salary + NULL = NULL, and NULL > 100000 = UNKNOWN, so employees "
                "with NULL bonus and high salary would be incorrectly excluded."
            )
        },
        {
            "label": "(b) Average bonus, treating NULL as 0",
            "sql": (
                "SELECT AVG(COALESCE(bonus, 0)) AS avg_bonus\n"
                "FROM employee;"
            ),
            "explanation": (
                "AVG(bonus) ignores NULLs entirely (divides only by non-NULL count). "
                "COALESCE(bonus, 0) converts NULLs to 0, so they contribute to both "
                "the sum and count, giving the true average across ALL employees."
            )
        },
        {
            "label": "(c) Departments where ALL employees have non-NULL bonus",
            "sql": (
                "SELECT dept FROM employee\n"
                "GROUP BY dept\n"
                "HAVING COUNT(*) = COUNT(bonus);"
            ),
            "explanation": (
                "COUNT(*) counts all rows; COUNT(bonus) counts only non-NULL bonus values. "
                "If they are equal, no employee in that department has a NULL bonus."
            )
        },
        {
            "label": "(d) Employees with different bonus from employee 'E001' (including NULLs)",
            "sql": (
                "SELECT * FROM employee\n"
                "WHERE NOT (bonus IS NOT DISTINCT FROM\n"
                "           (SELECT bonus FROM employee WHERE emp_id = 'E001'));"
            ),
            "explanation": (
                "IS NOT DISTINCT FROM treats NULLs as equal (NULL IS NOT DISTINCT FROM NULL = TRUE). "
                "The negation NOT(...) finds employees whose bonus differs, including the case "
                "where one is NULL and the other is not. Standard != would exclude NULL comparisons."
            )
        }
    ]

    for q in queries:
        print(f"{q['label']}")
        print(f"  SQL:\n    {q['sql']}")
        print(f"  Explanation: {q['explanation']}")
        print()


# === Exercise 2.10: Online Bookstore Schema ===
# Problem: Design a relational schema for an online bookstore.

def exercise_2_10():
    """Design relational schema for online bookstore."""
    print("=== Online Bookstore Relational Schema ===\n")

    schema = {
        "books": {
            "columns": "isbn VARCHAR(13) PK, title VARCHAR(255) NOT NULL, "
                       "price DECIMAL(10,2) NOT NULL CHECK(price > 0), "
                       "pub_date DATE, page_count INT CHECK(page_count > 0)",
            "pk": "isbn"
        },
        "authors": {
            "columns": "author_id SERIAL PK, name VARCHAR(100) NOT NULL, biography TEXT",
            "pk": "author_id"
        },
        "book_authors": {
            "columns": "isbn VARCHAR(13) FK->books, author_id INT FK->authors, "
                       "author_order INT NOT NULL",
            "pk": "(isbn, author_id)"
        },
        "customers": {
            "columns": "customer_id SERIAL PK, name VARCHAR(100) NOT NULL, "
                       "email VARCHAR(255) UNIQUE NOT NULL, shipping_address TEXT",
            "pk": "customer_id"
        },
        "orders": {
            "columns": "order_id SERIAL PK, customer_id INT FK->customers NOT NULL, "
                       "order_date TIMESTAMP DEFAULT NOW(), "
                       "status VARCHAR(20) CHECK(status IN ('pending','confirmed','shipped','delivered','cancelled'))",
            "pk": "order_id"
        },
        "order_items": {
            "columns": "order_id INT FK->orders, isbn VARCHAR(13) FK->books, "
                       "quantity INT NOT NULL CHECK(quantity > 0), "
                       "unit_price DECIMAL(10,2) NOT NULL",
            "pk": "(order_id, isbn)"
        },
        "reviews": {
            "columns": "review_id SERIAL PK, customer_id INT FK->customers, "
                       "isbn VARCHAR(13) FK->books, "
                       "rating INT NOT NULL CHECK(rating BETWEEN 1 AND 5), "
                       "review_text TEXT, review_date DATE DEFAULT CURRENT_DATE",
            "pk": "review_id",
            "unique": "(customer_id, isbn)  -- one review per customer per book"
        }
    }

    for table_name, info in schema.items():
        print(f"  {table_name}")
        print(f"    Columns: {info['columns']}")
        print(f"    PK: {info['pk']}")
        if 'unique' in info:
            print(f"    UNIQUE: {info['unique']}")
        print()

    print("Additional Integrity Constraints:")
    print("  - order_items.unit_price should match books.price at time of order (stored for history)")
    print("  - A customer can only review a book they have purchased (application-level or trigger)")
    print("  - DELETE on books should CASCADE to book_authors, RESTRICT if order_items exist")
    print()

    print("Example Valid Tuples:")
    print("  books: ('978-0132350884', 'Clean Code', 39.99, '2008-08-01', 464)")
    print("  reviews: (1, 42, '978-0132350884', 5, 'Excellent book!', '2025-01-15')")
    print()
    print("Example Invalid Tuples:")
    print("  books: (NULL, 'No ISBN', 29.99, ...) -- PK cannot be NULL")
    print("  reviews: (..., rating=6, ...) -- CHECK(rating BETWEEN 1 AND 5) violated")
    print("  orders: (..., status='unknown', ...) -- CHECK constraint violated")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 2.2: Key Terminology ===")
    print("=" * 70)
    exercise_2_2()

    print("=" * 70)
    print("=== Exercise 2.6: Candidate Keys for FLIGHT ===")
    print("=" * 70)
    exercise_2_6()

    print("=" * 70)
    print("=== Exercise 2.7: Integrity Constraint Violations ===")
    print("=" * 70)
    exercise_2_7()

    print("=" * 70)
    print("=== Exercise 2.8: NULL Query Behavior ===")
    print("=" * 70)
    exercise_2_8()

    print("=" * 70)
    print("=== Exercise 2.9: NULL-Safe SQL Queries ===")
    print("=" * 70)
    exercise_2_9()

    print("=" * 70)
    print("=== Exercise 2.10: Online Bookstore Schema ===")
    print("=" * 70)
    exercise_2_10()

    print("\nAll exercises completed!")
