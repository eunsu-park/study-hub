"""
Exercises for Lesson 04: ER Modeling
Topic: Database_Theory

Solutions to practice problems from the lesson.
Covers ER concepts, attribute classification, cardinality constraints,
weak entities, and ER-to-relational mapping.
"""


# === Exercise 4.1: Entity vs Relationship vs Attribute ===
# Problem: Identify whether each item should be modeled as entity, relationship, or attribute.

def exercise_4_1():
    """Classify items as entity type, relationship type, or attribute."""
    classifications = [
        {
            "item": "(a) Employee name",
            "type": "Attribute",
            "reason": "A property of the Employee entity. Single value per employee."
        },
        {
            "item": "(b) Department",
            "type": "Entity Type",
            "reason": "Has its own identity (dept_id), attributes (name, budget), and "
                      "participates in relationships (employees work in departments)."
        },
        {
            "item": "(c) Marriage (between two persons)",
            "type": "Relationship Type",
            "reason": "Connects two Person entities. May have attributes (marriage_date, location)."
        },
        {
            "item": "(d) Student GPA",
            "type": "Attribute (derived)",
            "reason": "Computed from grades. Could be stored or derived. Still an attribute of Student."
        },
        {
            "item": "(e) Book ISBN",
            "type": "Attribute (key)",
            "reason": "Uniquely identifies a Book entity. It is a property, not an entity itself."
        },
        {
            "item": "(f) Course enrollment",
            "type": "Relationship Type",
            "reason": "M:N relationship between Student and Course. Has attributes (grade, semester)."
        },
        {
            "item": "(g) Employee skill (assuming many skills per employee)",
            "type": "Entity Type (or multivalued attribute)",
            "reason": "If skills have attributes (proficiency level, certification date), model as entity. "
                      "If just a list of skill names, can be a multivalued attribute."
        },
        {
            "item": "(h) Project deadline",
            "type": "Attribute",
            "reason": "A single date/timestamp property of the Project entity."
        }
    ]

    for c in classifications:
        print(f"{c['item']}")
        print(f"  Type: {c['type']}")
        print(f"  Reason: {c['reason']}")
        print()


# === Exercise 4.2: Attribute Classification ===
# Problem: Classify attributes by multiple dimensions.

def exercise_4_2():
    """Classify attributes: simple/composite, single/multi, stored/derived, key."""
    attributes = [
        {
            "attribute": "SSN",
            "simple_composite": "Simple",
            "single_multi": "Single-valued",
            "stored_derived": "Stored",
            "key": "Yes (candidate key)"
        },
        {
            "attribute": "Full name (first + middle + last)",
            "simple_composite": "Composite",
            "single_multi": "Single-valued",
            "stored_derived": "Stored",
            "key": "No"
        },
        {
            "attribute": "Phone numbers (multiple)",
            "simple_composite": "Simple",
            "single_multi": "Multivalued",
            "stored_derived": "Stored",
            "key": "No"
        },
        {
            "attribute": "Age (given date of birth)",
            "simple_composite": "Simple",
            "single_multi": "Single-valued",
            "stored_derived": "Derived (from DOB and current date)",
            "key": "No"
        },
        {
            "attribute": "Email address",
            "simple_composite": "Simple (or composite: local@domain)",
            "single_multi": "Single-valued",
            "stored_derived": "Stored",
            "key": "Possibly (alternate key if UNIQUE)"
        },
        {
            "attribute": "Address (street, city, state, zip)",
            "simple_composite": "Composite",
            "single_multi": "Single-valued (or multi if multiple addresses)",
            "stored_derived": "Stored",
            "key": "No"
        }
    ]

    # Print as table
    header = f"{'Attribute':<35} {'Simple/Comp':<15} {'Single/Multi':<15} {'Stored/Derived':<30} {'Key?':<20}"
    print(header)
    print("-" * len(header))
    for a in attributes:
        print(f"{a['attribute']:<35} {a['simple_composite']:<15} {a['single_multi']:<15} "
              f"{a['stored_derived']:<30} {a['key']:<20}")


# === Exercise 4.5: Cardinality and Participation ===
# Problem: Determine cardinality ratios and participation constraints.

def exercise_4_5():
    """Determine cardinality and participation constraints."""
    scenarios = [
        {
            "scenario": "(a) Country - Capital City",
            "cardinality": "1:1",
            "participation": "Country: Total, Capital: Total",
            "reason": "Every country has exactly one capital; every capital belongs to exactly one country."
        },
        {
            "scenario": "(b) Student - Dormitory Room",
            "cardinality": "N:1 (many students to one room)",
            "participation": "Student: Partial (not all students live in dorms), Room: Partial (some empty)",
            "reason": "Multiple students can share a room; some students live off-campus."
        },
        {
            "scenario": "(c) Author - Book",
            "cardinality": "M:N",
            "participation": "Author: Partial (may have unpublished), Book: Total (every book has >= 1 author)",
            "reason": "A book can have multiple co-authors; an author writes many books."
        },
        {
            "scenario": "(d) Employee - Project",
            "cardinality": "M:N",
            "participation": "Employee: Partial (some may not be on projects), Project: Total (needs members)",
            "reason": "Employees work on multiple projects; projects have multiple members."
        },
        {
            "scenario": "(e) Person - Passport",
            "cardinality": "1:1",
            "participation": "Person: Partial (not everyone has a passport), Passport: Total (every passport has an owner)",
            "reason": "Each person has at most one passport; a passport belongs to exactly one person."
        }
    ]

    for s in scenarios:
        print(f"{s['scenario']}")
        print(f"  Cardinality: {s['cardinality']}")
        print(f"  Participation: {s['participation']}")
        print(f"  Reason: {s['reason']}")
        print()


# === Exercise 4.6: Weak Entities ===
# Problem: Identify weak entities and their partial keys.

def exercise_4_6():
    """Identify weak entities with partial keys and identifying relationships."""
    pairs = [
        {
            "pair": "(a) Building and Room",
            "weak_entity": "Room",
            "partial_key": "room_number (unique only within a building)",
            "identifying_rel": "LOCATED_IN (Room is identified by Building + room_number)",
            "reason": "Room 101 in Building A is different from Room 101 in Building B."
        },
        {
            "pair": "(b) Invoice and LineItem",
            "weak_entity": "LineItem",
            "partial_key": "line_number (unique within an invoice)",
            "identifying_rel": "BELONGS_TO (LineItem is identified by Invoice + line_number)",
            "reason": "Line item #1 on Invoice 1001 is different from line item #1 on Invoice 1002."
        },
        {
            "pair": "(c) Student and Course",
            "weak_entity": "Neither",
            "partial_key": "N/A",
            "identifying_rel": "N/A (both are strong entities with own keys: student_id, course_id)",
            "reason": "Both have globally unique identifiers. They relate via ENROLLMENT."
        },
        {
            "pair": "(d) Bank and Branch",
            "weak_entity": "Branch",
            "partial_key": "branch_name or branch_number (unique within a bank)",
            "identifying_rel": "PART_OF (Branch is identified by Bank + branch_name)",
            "reason": "'Downtown Branch' of Bank A differs from 'Downtown Branch' of Bank B."
        },
        {
            "pair": "(e) Order and OrderItem",
            "weak_entity": "OrderItem",
            "partial_key": "item_number (sequential within an order)",
            "identifying_rel": "CONTAINS (OrderItem identified by Order + item_number)",
            "reason": "Item #1 on Order 1000 is a different thing from Item #1 on Order 2000."
        }
    ]

    for p in pairs:
        print(f"{p['pair']}")
        print(f"  Weak Entity: {p['weak_entity']}")
        print(f"  Partial Key: {p['partial_key']}")
        print(f"  Identifying Relationship: {p['identifying_rel']}")
        print(f"  Reason: {p['reason']}")
        print()


# === Exercise 4.8: ER-to-Relational Mapping ===
# Problem: Apply 7-step mapping algorithm to produce SQL DDL.

def exercise_4_8():
    """ER-to-Relational mapping: complete SQL DDL."""
    print("ER-to-Relational Mapping (7-step algorithm):\n")

    steps = [
        ("Step 1: Map strong entity types",
         "DEPARTMENT and EMPLOYEE are strong entities."),
        ("Step 2: Map weak entity types",
         "DEPENDENT is a weak entity, identified by EMPLOYEE."),
        ("Step 3: Map 1:N relationships",
         "WORKS_IN (EMPLOYEE N:1 DEPARTMENT) -> add FK in EMPLOYEE."),
        ("Step 4: Map M:N relationships",
         "WORKS_ON (EMPLOYEE M:N PROJECT) -> create junction table."),
        ("Step 5: Map multivalued attributes",
         "skill (multivalued attribute of EMPLOYEE) -> separate table."),
        ("Step 6: Map specialization/generalization",
         "Not applicable in this ER diagram."),
        ("Step 7: Map higher-degree relationships",
         "Not applicable in this ER diagram.")
    ]

    for step_name, description in steps:
        print(f"  {step_name}: {description}")
    print()

    ddl = """-- Step 1: Strong entities
CREATE TABLE department (
    dept_id     VARCHAR(10) PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    budget      DECIMAL(12,2)
);

CREATE TABLE employee (
    emp_id      INT PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    salary      DECIMAL(10,2),
    birth_date  DATE,
    -- Step 3: FK from 1:N WORKS_IN relationship
    dept_id     VARCHAR(10) NOT NULL
        REFERENCES department(dept_id)
        ON DELETE RESTRICT  -- cannot delete dept if employees exist
        ON UPDATE CASCADE
);

CREATE TABLE project (
    proj_id     INT PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    budget      DECIMAL(12,2),
    location    VARCHAR(100)
);

-- Step 2: Weak entity DEPENDENT
CREATE TABLE dependent (
    emp_id      INT NOT NULL
        REFERENCES employee(emp_id)
        ON DELETE CASCADE,  -- delete dependents when employee is deleted
    dep_name    VARCHAR(100) NOT NULL,
    birth_date  DATE,
    relationship VARCHAR(20),
    PRIMARY KEY (emp_id, dep_name)  -- composite key: owner + partial key
);

-- Step 4: M:N relationship WORKS_ON
CREATE TABLE works_on (
    emp_id      INT NOT NULL
        REFERENCES employee(emp_id)
        ON DELETE CASCADE,
    proj_id     INT NOT NULL
        REFERENCES project(proj_id)
        ON DELETE CASCADE,
    hours       DECIMAL(5,1),  -- relationship attribute
    PRIMARY KEY (emp_id, proj_id)
);

-- Step 5: Multivalued attribute 'skill'
CREATE TABLE employee_skill (
    emp_id      INT NOT NULL
        REFERENCES employee(emp_id)
        ON DELETE CASCADE,
    skill       VARCHAR(50) NOT NULL,
    PRIMARY KEY (emp_id, skill)
);"""

    print("SQL DDL:")
    print(ddl)


# === Exercise 4.9: Specialization Mapping Options ===
# Problem: Map VEHICLE specialization hierarchy using three approaches.

def exercise_4_9():
    """Map specialization hierarchy using three approaches with trade-off analysis."""
    print("VEHICLE(vin, make, model, year, color)")
    print("  CAR(num_doors, trunk_vol)")
    print("  TRUCK(payload_cap, num_axles, cab_type)")
    print("  Constraint: disjoint, total\n")

    options = {
        "Option A: Single Table": {
            "ddl": """CREATE TABLE vehicle (
    vin         VARCHAR(17) PRIMARY KEY,
    make        VARCHAR(50) NOT NULL,
    model       VARCHAR(50) NOT NULL,
    year        INT NOT NULL,
    color       VARCHAR(20),
    vehicle_type VARCHAR(5) NOT NULL CHECK (vehicle_type IN ('CAR', 'TRUCK')),
    -- CAR attributes (NULL for trucks)
    num_doors   INT,
    trunk_vol   DECIMAL(5,1),
    -- TRUCK attributes (NULL for cars)
    payload_cap DECIMAL(10,2),
    num_axles   INT,
    cab_type    VARCHAR(20)
);""",
            "pros": [
                "No joins needed for any query",
                "Simple schema (one table)",
                "Good for queries that span all vehicle types"
            ],
            "cons": [
                "Many NULLs (car attrs NULL for trucks and vice versa)",
                "Cannot enforce NOT NULL on subclass attributes",
                "Wastes storage for sparse columns"
            ]
        },
        "Option B: Superclass + Subclass Tables": {
            "ddl": """CREATE TABLE vehicle (
    vin         VARCHAR(17) PRIMARY KEY,
    make        VARCHAR(50) NOT NULL,
    model       VARCHAR(50) NOT NULL,
    year        INT NOT NULL,
    color       VARCHAR(20),
    vehicle_type VARCHAR(5) NOT NULL
);

CREATE TABLE car (
    vin         VARCHAR(17) PRIMARY KEY REFERENCES vehicle(vin) ON DELETE CASCADE,
    num_doors   INT NOT NULL,
    trunk_vol   DECIMAL(5,1)
);

CREATE TABLE truck (
    vin         VARCHAR(17) PRIMARY KEY REFERENCES vehicle(vin) ON DELETE CASCADE,
    payload_cap DECIMAL(10,2) NOT NULL,
    num_axles   INT NOT NULL,
    cab_type    VARCHAR(20)
);""",
            "pros": [
                "No NULLs -- subclass attrs have proper NOT NULL constraints",
                "Clean separation of concerns",
                "Can query all vehicles without joins (superclass table)"
            ],
            "cons": [
                "Need JOIN for full car/truck details",
                "Cannot enforce total participation at DB level (vehicle without car or truck row)",
                "Two inserts needed to create one car"
            ]
        },
        "Option C: Separate Tables (no superclass table)": {
            "ddl": """CREATE TABLE car (
    vin         VARCHAR(17) PRIMARY KEY,
    make        VARCHAR(50) NOT NULL,
    model       VARCHAR(50) NOT NULL,
    year        INT NOT NULL,
    color       VARCHAR(20),
    num_doors   INT NOT NULL,
    trunk_vol   DECIMAL(5,1)
);

CREATE TABLE truck (
    vin         VARCHAR(17) PRIMARY KEY,
    make        VARCHAR(50) NOT NULL,
    model       VARCHAR(50) NOT NULL,
    year        INT NOT NULL,
    color       VARCHAR(20),
    payload_cap DECIMAL(10,2) NOT NULL,
    num_axles   INT NOT NULL,
    cab_type    VARCHAR(20)
);""",
            "pros": [
                "No NULLs, no joins for subclass queries",
                "Total participation enforced (every row is a car or truck)",
                "Best for queries targeting a specific subclass"
            ],
            "cons": [
                "Common attributes duplicated in schema definition",
                "Cross-type queries require UNION",
                "Cannot easily reference 'any vehicle' via foreign key",
                "Disjointness must be enforced at application level (same VIN in both tables)"
            ]
        }
    }

    for option_name, details in options.items():
        print(f"--- {option_name} ---")
        print(details["ddl"])
        print("\n  Pros:")
        for p in details["pros"]:
            print(f"    + {p}")
        print("  Cons:")
        for c in details["cons"]:
            print(f"    - {c}")
        print()

    print("Recommendation:")
    print("  For disjoint+total specialization: Option C is cleanest (no NULLs, enforces total).")
    print("  For overlapping or partial: Option B is safest (superclass always exists).")
    print("  For simple OLTP with few subclass queries: Option A is simplest.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 4.1: Entity vs Relationship vs Attribute ===")
    print("=" * 70)
    exercise_4_1()

    print("=" * 70)
    print("=== Exercise 4.2: Attribute Classification ===")
    print("=" * 70)
    exercise_4_2()

    print("=" * 70)
    print("=== Exercise 4.5: Cardinality and Participation ===")
    print("=" * 70)
    exercise_4_5()

    print("=" * 70)
    print("=== Exercise 4.6: Weak Entities ===")
    print("=" * 70)
    exercise_4_6()

    print("=" * 70)
    print("=== Exercise 4.8: ER-to-Relational Mapping ===")
    print("=" * 70)
    exercise_4_8()

    print("=" * 70)
    print("=== Exercise 4.9: Specialization Mapping Options ===")
    print("=" * 70)
    exercise_4_9()

    print("\nAll exercises completed!")
