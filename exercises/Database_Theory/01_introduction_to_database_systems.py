"""
Exercises for Lesson 01: Introduction to Database Systems
Topic: Database_Theory

Solutions to practice problems from the lesson.
These exercises are primarily conceptual, so solutions are presented
as structured data and explanatory text outputs.
"""


# === Exercise 1.1: File-Based Disadvantages ===
# Problem: List five disadvantages of the file-based approach to data management
# and explain how a DBMS addresses each one.

def exercise_1_1():
    """Five disadvantages of file-based systems and DBMS solutions."""
    disadvantages = {
        "Data Redundancy": {
            "file_problem": "Same data stored in multiple files (e.g., customer name in orders.csv and customers.csv)",
            "dbms_solution": "Normalization and single source of truth via relational schema with foreign keys"
        },
        "Data Inconsistency": {
            "file_problem": "Redundant copies can become inconsistent when one is updated but not others",
            "dbms_solution": "Referential integrity constraints and transactions ensure consistency"
        },
        "Difficulty Accessing Data": {
            "file_problem": "Each new query requires writing a custom program to parse files",
            "dbms_solution": "SQL provides a declarative query language; the DBMS optimizes execution"
        },
        "Concurrent Access Anomalies": {
            "file_problem": "Multiple users editing the same file can corrupt data (lost updates, dirty reads)",
            "dbms_solution": "DBMS provides concurrency control (locking, MVCC) to isolate transactions"
        },
        "Security Problems": {
            "file_problem": "File-level permissions are coarse-grained; cannot restrict access to specific records/fields",
            "dbms_solution": "DBMS offers fine-grained access control: views, row-level security, column permissions"
        }
    }

    for i, (name, details) in enumerate(disadvantages.items(), 1):
        print(f"{i}. {name}")
        print(f"   File Problem: {details['file_problem']}")
        print(f"   DBMS Solution: {details['dbms_solution']}")
        print()


# === Exercise 1.5: Schema Change Classification ===
# Problem: Classify each change as affecting external, conceptual, or internal schema.
# Explain whether data independence is preserved.

def exercise_1_5():
    """Classify schema changes and check data independence."""
    changes = [
        {
            "change": "(a) A new index is added to the STUDENT table",
            "schema_level": "Internal",
            "independence": "Physical data independence preserved — external/conceptual schemas unchanged. "
                          "Applications see no difference; only performance improves."
        },
        {
            "change": "(b) A new column 'email' is added to the STUDENT table",
            "schema_level": "Conceptual",
            "independence": "Logical data independence tested — existing external schemas (views) that don't "
                          "reference 'email' remain unchanged. New views can include it."
        },
        {
            "change": "(c) STUDENT table is split into STUDENT_PERSONAL and STUDENT_ACADEMIC",
            "schema_level": "Conceptual",
            "independence": "Logical data independence tested — if a view 'student_summary' is redefined as a "
                          "JOIN of the two tables, external schema is preserved for applications."
        },
        {
            "change": "(d) The database file is moved from HDD to SSD",
            "schema_level": "Internal",
            "independence": "Physical data independence preserved — only storage medium changes. "
                          "No schema modifications at any level; just faster I/O."
        },
        {
            "change": "(e) A new view is created for the financial aid office",
            "schema_level": "External",
            "independence": "No independence concern — adding a new external schema (view) does not "
                          "affect existing views or the conceptual/internal schemas."
        }
    ]

    for item in changes:
        print(f"{item['change']}")
        print(f"   Schema Level: {item['schema_level']}")
        print(f"   Independence: {item['independence']}")
        print()


# === Exercise 1.7: User Classification ===
# Problem: Classify hospital database users by their role.

def exercise_1_7():
    """Classify hospital database system users by role."""
    users = [
        ("(a) Person designing the ER diagram for patient records",
         "Database Designer",
         "Designs the conceptual and logical schema"),
        ("(b) Nurse entering patient vital signs through a tablet app",
         "End User (Parametric/Naive)",
         "Uses pre-built forms; does not write queries directly"),
        ("(c) IT staff performing nightly backups and monitoring queries",
         "Database Administrator (DBA)",
         "Responsible for maintenance, performance tuning, security, backups"),
        ("(d) Doctor querying the database to find patients with a specific diagnosis",
         "End User (Casual/Sophisticated)",
         "Writes ad-hoc queries (possibly SQL or via a query builder)"),
        ("(e) Programmer building the patient portal web application",
         "Application Programmer",
         "Writes application code that accesses the database via APIs/SQL")
    ]

    for description, role, explanation in users:
        print(f"{description}")
        print(f"   Role: {role}")
        print(f"   Reason: {explanation}")
        print()


# === Exercise 1.9: E-Commerce Migration Plan ===
# Problem: Design a migration plan from flat files to a relational database.

def exercise_1_9():
    """E-commerce flat file to relational database migration plan."""
    print("=== Migration Plan: Flat Files -> Relational Database ===\n")

    # Table design
    tables = {
        "customers": {
            "columns": ["customer_id (PK)", "name", "email (UNIQUE)", "phone", "address"],
            "source": "customers.csv"
        },
        "products": {
            "columns": ["product_id (PK)", "name", "description", "price", "stock_quantity", "category"],
            "source": "product_catalog.csv"
        },
        "orders": {
            "columns": ["order_id (PK)", "customer_id (FK->customers)", "order_date", "status", "total_amount"],
            "source": "orders.csv (header info)"
        },
        "order_items": {
            "columns": ["order_id (FK->orders)", "product_id (FK->products)", "quantity", "unit_price"],
            "source": "orders.csv (line items)"
        }
    }

    print("Tables:")
    for table_name, info in tables.items():
        print(f"\n  {table_name}:")
        print(f"    Source: {info['source']}")
        print(f"    Columns: {', '.join(info['columns'])}")

    print("\nRelationships:")
    print("  - customers 1:N orders (a customer places many orders)")
    print("  - orders 1:N order_items (an order contains many items)")
    print("  - products 1:N order_items (a product appears in many orders)")

    print("\nProblems Solved:")
    problems = [
        "Data redundancy: Customer name stored once, not repeated in every order",
        "Update anomalies: Changing a product price updates one row, not every order line",
        "Insertion anomalies: Can add a product without requiring an order",
        "Concurrent access: Multiple staff can process orders simultaneously (ACID)",
        "Data integrity: Foreign keys prevent orphaned orders or invalid product references"
    ]
    for p in problems:
        print(f"  - {p}")

    print("\nAdditional DBMS Features:")
    features = [
        "Indexing for fast lookups (e.g., by customer email, product name)",
        "SQL queries for reports (daily sales, popular products)",
        "Backup and recovery",
        "Access control (restrict who can modify prices vs. view orders)",
        "Scalability beyond flat files (~50K products, ~1K orders/day easily handled)"
    ]
    for f in features:
        print(f"  - {f}")


# === Exercise 1.10: ANSI/SPARC Query Trace ===
# Problem: Trace a query through the three-schema architecture.

def exercise_1_10():
    """Trace a SQL query through the ANSI/SPARC architecture levels."""
    print("Query: SELECT name, gpa FROM student_summary WHERE gpa > 3.5;\n")
    print("View definition: CREATE VIEW student_summary AS")
    print("  SELECT student_id, first_name || ' ' || last_name AS name, gpa FROM students;\n")

    levels = [
        ("EXTERNAL LEVEL (View Resolution)", [
            "User submits query referencing the view 'student_summary'.",
            "DBMS resolves the view: replaces 'student_summary' with its definition.",
            "Rewritten query: SELECT first_name || ' ' || last_name AS name, gpa",
            "  FROM students WHERE gpa > 3.5;",
            "External schema maps 'student_summary' columns to the conceptual schema."
        ]),
        ("CONCEPTUAL LEVEL (Logical Plan)", [
            "Query references the logical table 'students' with columns: student_id, first_name, last_name, gpa.",
            "Logical plan: Selection(gpa > 3.5) -> Projection(name, gpa) on 'students'.",
            "Integrity constraints checked (e.g., gpa domain is NUMERIC(3,2)).",
            "Access control: verify user has SELECT privilege on 'students' or 'student_summary'."
        ]),
        ("INTERNAL LEVEL (Physical Access)", [
            "Map logical 'students' table to physical storage (e.g., heap file on disk).",
            "Optimizer decides access path: if index on 'gpa' exists, use index scan;",
            "  otherwise, sequential scan of data blocks.",
            "Physical plan: IndexScan(gpa > 3.5) or SeqScan with filter.",
            "Buffer manager loads required data blocks into memory.",
            "Results flow back up through the levels to the user."
        ])
    ]

    for level_name, steps in levels:
        print(f"--- {level_name} ---")
        for step in steps:
            print(f"  {step}")
        print()


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1.1: File-Based Disadvantages ===")
    print("=" * 70)
    exercise_1_1()

    print("=" * 70)
    print("=== Exercise 1.5: Schema Change Classification ===")
    print("=" * 70)
    exercise_1_5()

    print("=" * 70)
    print("=== Exercise 1.7: User Classification ===")
    print("=" * 70)
    exercise_1_7()

    print("=" * 70)
    print("=== Exercise 1.9: E-Commerce Migration Plan ===")
    print("=" * 70)
    exercise_1_9()

    print("=" * 70)
    print("=== Exercise 1.10: ANSI/SPARC Query Trace ===")
    print("=" * 70)
    exercise_1_10()

    print("\nAll exercises completed!")
