# The Relational Model

**Previous**: [Introduction to Database Systems](./01_Introduction_to_Database_Systems.md) | **Next**: [Relational Algebra](./03_Relational_Algebra.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the historical context of Codd's relational model and describe the problems it solved over earlier data models.
2. Define the formal components of a relation — domain, tuple, attribute, and schema — using mathematical notation.
3. Identify and distinguish among superkeys, candidate keys, primary keys, and foreign keys, and explain how each enforces integrity.
4. Apply entity integrity and referential integrity constraints to validate relational schemas.
5. Describe the semantics of NULL values and evaluate expressions using three-valued logic (TRUE, FALSE, UNKNOWN).
6. Translate an informal data requirement into a formal relational schema with appropriate constraints and notation.

---

The relational model, introduced by Edgar F. Codd in 1970, is the most widely used data model and the theoretical foundation of SQL databases. It represents data as mathematical relations (tables) and provides a rigorous framework for defining structure, enforcing integrity, and manipulating data. This lesson covers the formal definitions, key concepts, integrity constraints, and the subtle semantics of NULLs that every database practitioner must understand.

## Table of Contents

1. [Historical Context: Codd's Vision](#1-historical-context-codds-vision)
2. [Codd's 12 Rules](#2-codds-12-rules)
3. [Formal Definitions](#3-formal-definitions)
4. [Keys](#4-keys)
5. [Integrity Constraints](#5-integrity-constraints)
6. [Relational Schema Notation](#6-relational-schema-notation)
7. [NULL Semantics and Three-Valued Logic](#7-null-semantics-and-three-valued-logic)
8. [Relational Model in Practice](#8-relational-model-in-practice)
9. [Exercises](#9-exercises)

---

## 1. Historical Context: Codd's Vision

In 1970, Edgar F. Codd, a researcher at IBM's San Jose Research Laboratory, published "A Relational Model of Data for Large Shared Data Banks" in Communications of the ACM. This paper fundamentally changed how we think about data management.

### The Problem Codd Addressed

Before the relational model, the two dominant approaches were:

```
Hierarchical Model (IMS):        Network Model (CODASYL):

     ROOT                          STUDENT ──── COURSE
    /    \                            │    \  /     │
  CHILD1  CHILD2                      │     \/      │
    |                                 │     /\      │
  CHILD3                              │    /  \     │
                                      ▼   /    \    ▼
                                   ADVISOR    ENROLLMENT

Both required:
  - Programs to navigate through pointer chains
  - Knowledge of physical data layout
  - Code rewriting when structure changed
```

### Codd's Key Insight

Codd proposed representing data as **relations** (mathematical concept), which correspond to tables:

```
Instead of:                         Use:
  "Follow pointer from              "SELECT student_name
   STUDENT record through            FROM students
   enrollment chain to               JOIN courses
   find course name"                 ON students.id = enrollments.student_id
                                     WHERE course_name = 'Database Theory'"

  Navigate HOW                      Declare WHAT
  (procedural)                      (declarative)
```

This shift from **procedural navigation** to **declarative specification** was revolutionary.

---

## 2. Codd's 12 Rules

In 1985, Codd published 12 rules (actually 13, numbered 0-12) that define what it means for a database management system to be truly "relational." These rules serve as a benchmark for evaluating RDBMS implementations.

### Rule 0: The Foundation Rule

> A relational DBMS must manage its stored data using only its relational capabilities.

### The 12 Rules

| # | Rule Name | Description |
|---|-----------|-------------|
| 1 | **Information Rule** | All data is represented as values in tables (relations). This includes metadata (the system catalog). |
| 2 | **Guaranteed Access Rule** | Every datum is accessible by specifying the table name, column name, and primary key value. No pointers needed. |
| 3 | **Systematic Treatment of NULL** | NULL represents missing or inapplicable data, distinct from empty string or zero. Supported for all data types. |
| 4 | **Dynamic Online Catalog** | The database description (metadata) is stored in tables and queryable using the same relational language as user data. |
| 5 | **Comprehensive Data Sublanguage** | At least one language must support data definition, manipulation, integrity constraints, authorization, and transactions. (SQL fulfills this.) |
| 6 | **View Updating Rule** | All views that are theoretically updatable must be updatable by the system. |
| 7 | **High-Level Insert, Update, Delete** | Set-at-a-time operations (not just row-at-a-time). You can insert, update, or delete multiple rows in a single statement. |
| 8 | **Physical Data Independence** | Applications are not affected by changes to physical storage or access methods. |
| 9 | **Logical Data Independence** | Applications are not affected by information-preserving changes to the conceptual schema. |
| 10 | **Integrity Independence** | Integrity constraints are defined in the catalog (not in application programs) and can be changed without affecting applications. |
| 11 | **Distribution Independence** | Applications work the same whether data is centralized or distributed. |
| 12 | **Nonsubversion Rule** | If the system provides a low-level (record-at-a-time) interface, it cannot be used to bypass relational integrity constraints. |

### Practical Assessment

No commercial RDBMS fully satisfies all 12 rules. Here is a rough assessment:

```
Rule compliance (approximate):

                    PostgreSQL  MySQL   Oracle  SQLite
Rule 1 (Info)          ✓         ✓       ✓       ✓
Rule 2 (Access)        ✓         ✓       ✓       ✓
Rule 3 (NULL)          ✓         ✓       ✓       ~
Rule 4 (Catalog)       ✓         ✓       ✓       ~
Rule 5 (Language)      ✓         ✓       ✓       ✓
Rule 6 (View Update)   ~         ~       ~       ~
Rule 7 (Set Ops)       ✓         ✓       ✓       ✓
Rule 8 (Physical DI)   ✓         ✓       ✓       ~
Rule 9 (Logical DI)    ~         ~       ~       ~
Rule 10 (Integrity)    ✓         ~       ✓       ~
Rule 11 (Distribution) ~         ~       ~       ✗
Rule 12 (Nonsub.)      ✓         ~       ✓       ~

✓ = Mostly compliant  ~ = Partially  ✗ = Not supported
```

---

## 3. Formal Definitions

The relational model is grounded in set theory and first-order predicate logic. Understanding the formal definitions is essential for database design and query formulation.

### Domain

A **domain** D is a set of atomic (indivisible) values. Each domain has a logical definition (what values mean) and a data type.

```
Examples of domains:

D_StudentID  = {S001, S002, S003, ..., S999}
D_Name       = set of all character strings of length <= 50
D_GPA        = {x in R | 0.0 <= x <= 4.0}
D_Grade      = {A+, A, A-, B+, B, B-, C+, C, C-, D+, D, F}
D_Credits    = {1, 2, 3, 4, 5}
D_Boolean    = {TRUE, FALSE}
D_Date       = set of all valid calendar dates
```

### Relation Schema

A **relation schema** R(A1, A2, ..., An) consists of:
- A relation name R
- A list of attributes A1, A2, ..., An
- Each attribute Ai has a domain dom(Ai)

```
Formal notation:

  R(A1: D1, A2: D2, ..., An: Dn)

Example:

  STUDENT(student_id: D_StudentID,
          name: D_Name,
          gpa: D_GPA)
```

The **degree** (or arity) of a relation is the number of attributes n.

### Relation (Instance)

A **relation** r of the relation schema R(A1, A2, ..., An) is a set of n-tuples:

```
r(R) ⊆ dom(A1) × dom(A2) × ... × dom(An)
```

Each n-tuple t is an ordered list of values:

```
t = <v1, v2, ..., vn>   where vi ∈ dom(Ai) ∪ {NULL}
```

### Formal Example

```
Schema:
  STUDENT(student_id: D_StudentID, name: D_Name, year: D_Year, gpa: D_GPA)

  where D_StudentID = {S001..S999}
        D_Name = strings of length <= 50
        D_Year = {1, 2, 3, 4}
        D_GPA = {x ∈ R | 0.0 ≤ x ≤ 4.0}

Instance (at time t):
  r(STUDENT) = {
    <S001, "Alice Kim",  3, 3.85>,
    <S002, "Bob Park",   2, 3.42>,
    <S003, "Carol Lee",  4, 3.91>,
    <S004, "Dave Choi",  1, NULL>
  }

  Degree = 4
  Cardinality (number of tuples) = 4
```

### Properties of Relations

| Property | Description |
|----------|-------------|
| **No duplicate tuples** | A relation is a set, so no two tuples can be identical |
| **Tuples are unordered** | There is no inherent ordering of rows |
| **Attributes are unordered** | The order of columns does not matter (though SQL implementations typically preserve declaration order) |
| **Attribute values are atomic** | Each cell contains a single, indivisible value (first normal form) |
| **Each attribute has a distinct name** | No two attributes in the same relation share a name |

### Relation vs. Table

While often used interchangeably, there are subtle differences:

```
┌─────────────────────────────┬────────────────────────────────┐
│     Relation (Theory)       │        Table (SQL)             │
├─────────────────────────────┼────────────────────────────────┤
│ Set of tuples (no dups)     │ Multiset of rows (dups OK)     │
│ Tuples are unordered        │ Rows may have physical order   │
│ Attributes are unordered    │ Columns have declared order    │
│ All values are atomic       │ Arrays, JSON allowed in some   │
│ Named perspective only      │ Positional access possible     │
│ Domain-based typing         │ SQL data types                 │
└─────────────────────────────┴────────────────────────────────┘

SQL tables are NOT strictly relations because:
  1. SQL allows duplicate rows (unless constrained)
  2. SQL preserves column order
  3. SQL has additional features (auto-increment, etc.)
```

### Attribute Types and Domains in SQL

```sql
-- Mapping mathematical domains to SQL types

CREATE DOMAIN grade_domain AS VARCHAR(2)
    CHECK (VALUE IN ('A+','A','A-','B+','B','B-','C+','C','C-','D+','D','F'));

CREATE DOMAIN gpa_domain AS NUMERIC(3,2)
    CHECK (VALUE >= 0.0 AND VALUE <= 4.0);

CREATE TABLE student (
    student_id  CHAR(4)        NOT NULL,  -- D_StudentID
    name        VARCHAR(50)    NOT NULL,  -- D_Name
    year        SMALLINT       NOT NULL,  -- D_Year
    gpa         gpa_domain,               -- D_GPA (with domain)
    CONSTRAINT pk_student PRIMARY KEY (student_id),
    CONSTRAINT ck_year CHECK (year BETWEEN 1 AND 4)
);
```

Note: The `CREATE DOMAIN` syntax is supported by PostgreSQL but not all RDBMS.

---

## 4. Keys

Keys are fundamental to the relational model. They provide the mechanism for uniquely identifying tuples and establishing relationships between relations.

### Superkey

A **superkey** of a relation schema R is a set of attributes SK ⊆ R such that no two tuples in any valid relation instance r(R) have the same values for SK.

Formally: For any two distinct tuples t1, t2 in r(R):

```
t1[SK] ≠ t2[SK]
```

```
STUDENT(student_id, name, year, gpa)

Superkeys of STUDENT:
  {student_id}                          ← minimal
  {student_id, name}                    ← not minimal (student_id alone suffices)
  {student_id, name, year}              ← not minimal
  {student_id, name, year, gpa}         ← trivial superkey (all attributes)
  {name, year}                          ← might be superkey IF unique

Note: The set of ALL attributes is always a superkey (trivial superkey).
```

### Candidate Key

A **candidate key** is a **minimal superkey** -- a superkey from which no attribute can be removed without losing the uniqueness property.

```
Formal definition:
  A superkey K of R is a candidate key if for every proper subset K' ⊂ K,
  K' is NOT a superkey of R.

Example:
  ENROLLMENT(student_id, course_id, semester, grade)

  Superkeys:
    {student_id, course_id, semester}             ← minimal → CANDIDATE KEY
    {student_id, course_id, semester, grade}      ← not minimal

  If we assume (name, year) is unique for STUDENT:
    {name, year} is another candidate key

  A relation can have MULTIPLE candidate keys.
```

### Primary Key

The **primary key** is the candidate key chosen by the database designer to be the main identifier for tuples. It is marked with underline in schema notation.

```
Conventions:
  - Underline the primary key attributes
  - Primary key values cannot be NULL
  - Each relation has exactly ONE primary key
  - Other candidate keys become ALTERNATE KEYS

  STUDENT(student_id, name, year, gpa)
         ^^^^^^^^^^
         Primary Key

  If both {student_id} and {name, year} are candidate keys:
    Primary key: {student_id}       (chosen by designer)
    Alternate key: {name, year}     (the other candidate key)
```

### Foreign Key

A **foreign key** is a set of attributes in one relation that refers to the primary key of another (or the same) relation.

```
Formal definition:
  A set of attributes FK in relation R1 is a foreign key referencing
  relation R2 if:
    1. The attributes in FK have the same domain(s) as the primary key
       PK of R2
    2. A value of FK in a tuple t1 of r(R1) either:
       (a) occurs as a value of PK for some tuple t2 in r(R2), or
       (b) is NULL (if allowed)

Example:
  STUDENT(student_id, name, year, gpa)
  COURSE(course_id, title, credits)
  ENROLLMENT(student_id, course_id, semester, grade)
             ^^^^^^^^^^  ^^^^^^^^^
             FK → STUDENT  FK → COURSE

Diagrammatically:

  STUDENT                    ENROLLMENT                    COURSE
  ┌───────────┐             ┌───────────────┐             ┌───────────┐
  │student_id │◄────────────│student_id (FK)│             │course_id  │
  │name       │             │course_id  (FK)│────────────►│title      │
  │year       │             │semester       │             │credits    │
  │gpa        │             │grade          │             └───────────┘
  └───────────┘             └───────────────┘
```

### Composite Key

A key consisting of multiple attributes:

```
ENROLLMENT(student_id, course_id, semester, grade)

Primary key: {student_id, course_id, semester}
  - A composite key of THREE attributes
  - student_id alone is NOT sufficient (student takes many courses)
  - (student_id, course_id) alone is NOT sufficient
    (student may retake a course in a different semester)
```

### Surrogate Key vs. Natural Key

```
Natural Key:                        Surrogate Key:
  Uses real-world data                Uses system-generated value

  STUDENT(ssn, name, ...)            STUDENT(student_id, ssn, name, ...)
          ^^^                                 ^^^^^^^^^^
          Natural PK                          Surrogate PK (auto-increment)

Advantages of Natural Key:           Advantages of Surrogate Key:
  - Meaningful                         - Compact (integer)
  - No extra column                    - Immutable
  - Already unique                     - No business meaning changes
  - Self-documenting                   - Simple joins

Disadvantages:                       Disadvantages:
  - May change (name change)           - Extra column
  - May be large (SSN, ISBN)           - No semantic meaning
  - Privacy concerns (SSN)             - Requires lookup for meaning
```

### Summary of Key Types

```
┌─────────────────────────────────────────────────────────────┐
│                      Key Hierarchy                          │
│                                                             │
│  Superkey                                                   │
│    │                                                        │
│    ├── Candidate Key (minimal superkey)                     │
│    │     │                                                  │
│    │     ├── Primary Key (chosen candidate key)             │
│    │     │                                                  │
│    │     └── Alternate Key (non-chosen candidate key)       │
│    │                                                        │
│    └── Non-minimal superkey                                 │
│                                                             │
│  Foreign Key (references another relation's primary key)    │
│                                                             │
│  Composite Key (key with multiple attributes)               │
│                                                             │
│  Surrogate Key (system-generated, no business meaning)      │
│  Natural Key (derived from real-world data)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Integrity Constraints

Integrity constraints are rules that restrict the values in a database to ensure data correctness and consistency. The relational model defines several types of constraints.

### Domain Constraints

A **domain constraint** specifies that each attribute value must be from the attribute's domain (or NULL if allowed).

```sql
-- Domain constraints in SQL

CREATE TABLE student (
    student_id  CHAR(4),
    name        VARCHAR(50)   NOT NULL,       -- domain + NOT NULL
    year        SMALLINT      CHECK (year BETWEEN 1 AND 4),
    gpa         NUMERIC(3,2)  CHECK (gpa >= 0.0 AND gpa <= 4.0),
    email       VARCHAR(100)  CHECK (email LIKE '%@%.%')
);

-- Domain constraint violation examples:
INSERT INTO student VALUES ('S001', 'Alice', 5, 3.85, 'alice@univ.edu');
-- ERROR: year=5 violates CHECK constraint (year BETWEEN 1 AND 4)

INSERT INTO student VALUES ('S001', 'Alice', 3, 4.50, 'alice@univ.edu');
-- ERROR: gpa=4.50 violates CHECK constraint (gpa <= 4.0)

INSERT INTO student VALUES ('S001', NULL, 3, 3.85, 'alice@univ.edu');
-- ERROR: name is NOT NULL
```

### Entity Integrity Constraint

The **entity integrity constraint** states that no primary key attribute can be NULL.

```
Rule: If PK = {A1, A2, ..., Ak} is the primary key of R,
      then for every tuple t in r(R):
        t[Ai] ≠ NULL  for all i = 1, 2, ..., k

Rationale:
  - The primary key uniquely identifies each tuple
  - If PK were NULL, we could not distinguish that tuple from others
  - NULL ≠ NULL in SQL (NULL compared to anything yields UNKNOWN)
  - Therefore, a NULL PK would make identification impossible

Example:
  ENROLLMENT(student_id, course_id, semester, grade)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
             Composite PK - NONE of these can be NULL

  Valid:   ('S001', 'CS301', 'Fall2025', 'A')
  Invalid: ('S001', NULL, 'Fall2025', 'A')     -- course_id is part of PK
  Valid:   ('S001', 'CS301', 'Fall2025', NULL)  -- grade is NOT part of PK
```

### Referential Integrity Constraint

The **referential integrity constraint** ensures that foreign key values match existing primary key values in the referenced relation (or are NULL).

```
Rule: For every tuple t1 in r(R1) with foreign key FK referencing R2:
      Either t1[FK] is NULL
      Or there exists a tuple t2 in r(R2) such that t1[FK] = t2[PK]

Example:

  STUDENT:                           COURSE:
  ┌──────┬──────┐                    ┌──────┬───────────┐
  │ S001 │Alice │                    │CS101 │Intro CS   │
  │ S002 │Bob   │                    │CS301 │DB Theory  │
  │ S003 │Carol │                    │MA201 │Lin Alg    │
  └──────┴──────┘                    └──────┴───────────┘

  ENROLLMENT:
  ┌────────┬─────────┬───────┐
  │stu_id  │course_id│ grade │
  ├────────┼─────────┼───────┤
  │ S001   │ CS101   │  A    │  ✓ S001 exists in STUDENT, CS101 in COURSE
  │ S002   │ CS301   │  B+   │  ✓ S002 exists in STUDENT, CS301 in COURSE
  │ S004   │ CS101   │  A-   │  ✗ S004 does NOT exist in STUDENT!
  │ S003   │ CS999   │  B    │  ✗ CS999 does NOT exist in COURSE!
  └────────┴─────────┴───────┘
```

### Referential Integrity Actions

When a referenced tuple is deleted or updated, the DBMS must handle the orphaned foreign keys:

```sql
CREATE TABLE enrollment (
    student_id  CHAR(4),
    course_id   CHAR(5),
    semester    VARCHAR(10),
    grade       VARCHAR(2),

    PRIMARY KEY (student_id, course_id, semester),

    FOREIGN KEY (student_id) REFERENCES student(student_id)
        ON DELETE CASCADE           -- Delete enrollment if student deleted
        ON UPDATE CASCADE,          -- Update FK if student PK changes

    FOREIGN KEY (course_id) REFERENCES course(course_id)
        ON DELETE RESTRICT          -- Prevent course deletion if enrolled
        ON UPDATE CASCADE           -- Update FK if course PK changes
);
```

| Action | Behavior |
|--------|----------|
| **CASCADE** | Propagate the delete/update to referencing tuples |
| **RESTRICT** (or NO ACTION) | Reject the delete/update if referencing tuples exist |
| **SET NULL** | Set foreign key to NULL in referencing tuples |
| **SET DEFAULT** | Set foreign key to its default value |

### Referential Integrity Action Examples

```
Scenario: DELETE FROM student WHERE student_id = 'S001';

ON DELETE CASCADE:
  → All enrollments for S001 are automatically deleted

ON DELETE RESTRICT:
  → DELETE is rejected because S001 has enrollments

ON DELETE SET NULL:
  → enrollment.student_id is set to NULL for S001's rows
  → (Only if student_id in enrollment allows NULL — but it is part
     of the PK, so SET NULL would violate entity integrity!)

ON DELETE SET DEFAULT:
  → enrollment.student_id is set to its default value
```

### Key Constraints

Additional constraints enforced through key declarations:

```sql
-- UNIQUE constraint: alternate key
CREATE TABLE student (
    student_id  CHAR(4)     PRIMARY KEY,
    email       VARCHAR(100) UNIQUE NOT NULL,  -- alternate key
    ssn         CHAR(11)     UNIQUE            -- another alternate key (nullable)
);

-- Composite UNIQUE constraint
CREATE TABLE course_offering (
    offering_id   SERIAL PRIMARY KEY,          -- surrogate key
    course_id     CHAR(5) NOT NULL,
    semester      VARCHAR(10) NOT NULL,
    instructor_id INT NOT NULL,
    UNIQUE (course_id, semester)                -- natural key constraint
);
```

### General Constraints (Semantic Constraints)

Constraints that cannot be expressed by domain, key, or referential constraints alone:

```sql
-- CHECK constraint (tuple-level)
CREATE TABLE course (
    course_id  CHAR(5) PRIMARY KEY,
    title      VARCHAR(100) NOT NULL,
    credits    SMALLINT CHECK (credits BETWEEN 1 AND 5),
    max_seats  INT CHECK (max_seats > 0),
    min_seats  INT CHECK (min_seats > 0),
    CONSTRAINT seats_check CHECK (max_seats >= min_seats)
);

-- ASSERTION (cross-table constraint, SQL standard but rarely implemented)
-- "No student can be enrolled in more than 7 courses per semester"
CREATE ASSERTION max_courses_per_semester
    CHECK (NOT EXISTS (
        SELECT student_id, semester
        FROM enrollment
        GROUP BY student_id, semester
        HAVING COUNT(*) > 7
    ));

-- In practice, use TRIGGERS for cross-table constraints:
CREATE OR REPLACE FUNCTION check_max_courses()
RETURNS TRIGGER AS $$
BEGIN
    IF (SELECT COUNT(*) FROM enrollment
        WHERE student_id = NEW.student_id
        AND semester = NEW.semester) >= 7 THEN
        RAISE EXCEPTION 'Student cannot enroll in more than 7 courses per semester';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_max_courses
    BEFORE INSERT ON enrollment
    FOR EACH ROW EXECUTE FUNCTION check_max_courses();
```

### Constraint Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    Integrity Constraints                         │
│                                                                 │
│  Inherent Model Constraints (built into relational model):      │
│    - No duplicate tuples                                        │
│    - Atomic attribute values (1NF)                              │
│                                                                 │
│  Schema-Based Constraints (DDL):                                │
│    - Domain constraints (data type, CHECK)                      │
│    - Key constraints (PRIMARY KEY, UNIQUE)                      │
│    - Entity integrity (PK NOT NULL)                             │
│    - Referential integrity (FOREIGN KEY)                        │
│    - NOT NULL constraints                                       │
│                                                                 │
│  Application-Based Constraints (business rules):                │
│    - Triggers                                                   │
│    - Application logic                                          │
│    - Assertions (rarely supported in SQL)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Relational Schema Notation

There are several standard ways to represent relational schemas in documentation and textbooks.

### Textual Notation

```
STUDENT(student_id, name, year, gpa)
  PK: student_id
  FK: (none)

COURSE(course_id, title, credits, dept_id)
  PK: course_id
  FK: dept_id → DEPARTMENT(dept_id)

ENROLLMENT(student_id, course_id, semester, grade)
  PK: (student_id, course_id, semester)
  FK: student_id → STUDENT(student_id)
      course_id → COURSE(course_id)

DEPARTMENT(dept_id, dept_name, building, budget)
  PK: dept_id

INSTRUCTOR(instructor_id, name, dept_id, salary)
  PK: instructor_id
  FK: dept_id → DEPARTMENT(dept_id)
```

### Underline Convention

Primary key attributes are underlined (shown with underscores here):

```
STUDENT(_student_id_, name, year, gpa)
COURSE(_course_id_, title, credits, dept_id*)
ENROLLMENT(_student_id*_, _course_id*_, _semester_, grade)
DEPARTMENT(_dept_id_, dept_name, building, budget)
INSTRUCTOR(_instructor_id_, name, dept_id*, salary)

Convention:
  _underline_ = primary key attribute
  * = foreign key attribute
```

### Diagrammatic Notation

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│    DEPARTMENT     │     │    INSTRUCTOR     │     │      COURSE      │
├──────────────────┤     ├──────────────────┤     ├──────────────────┤
│ PK dept_id       │◄────│ PK instructor_id │     │ PK course_id     │
│    dept_name     │     │    name          │     │    title         │
│    building      │     │ FK dept_id    ───┘     │    credits       │
│    budget        │     │    salary       │     │ FK dept_id    ───┤
└──────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                           │
                                                           │
┌──────────────────┐     ┌──────────────────────┐         │
│     STUDENT      │     │     ENROLLMENT       │         │
├──────────────────┤     ├──────────────────────┤         │
│ PK student_id    │◄────│ PK,FK student_id  ───┘         │
│    name          │     │ PK,FK course_id   ─────────────┘
│    year          │     │ PK    semester     │
│    gpa           │     │       grade        │
└──────────────────┘     └──────────────────────┘
```

### SQL DDL as Schema Definition

```sql
-- Complete schema definition in SQL

CREATE TABLE department (
    dept_id     CHAR(4)      PRIMARY KEY,
    dept_name   VARCHAR(50)  NOT NULL UNIQUE,
    building    VARCHAR(30),
    budget      NUMERIC(12,2) CHECK (budget >= 0)
);

CREATE TABLE instructor (
    instructor_id  SERIAL       PRIMARY KEY,
    name           VARCHAR(50)  NOT NULL,
    dept_id        CHAR(4)      NOT NULL,
    salary         NUMERIC(10,2) CHECK (salary > 0),
    FOREIGN KEY (dept_id) REFERENCES department(dept_id)
        ON DELETE RESTRICT ON UPDATE CASCADE
);

CREATE TABLE student (
    student_id  CHAR(4)      PRIMARY KEY,
    name        VARCHAR(50)  NOT NULL,
    year        SMALLINT     NOT NULL CHECK (year BETWEEN 1 AND 4),
    gpa         NUMERIC(3,2) CHECK (gpa >= 0.0 AND gpa <= 4.0)
);

CREATE TABLE course (
    course_id   CHAR(5)      PRIMARY KEY,
    title       VARCHAR(100) NOT NULL,
    credits     SMALLINT     NOT NULL CHECK (credits BETWEEN 1 AND 5),
    dept_id     CHAR(4)      NOT NULL,
    FOREIGN KEY (dept_id) REFERENCES department(dept_id)
        ON DELETE RESTRICT ON UPDATE CASCADE
);

CREATE TABLE enrollment (
    student_id  CHAR(4)      NOT NULL,
    course_id   CHAR(5)      NOT NULL,
    semester    VARCHAR(10)  NOT NULL,
    grade       VARCHAR(2),
    PRIMARY KEY (student_id, course_id, semester),
    FOREIGN KEY (student_id) REFERENCES student(student_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (course_id) REFERENCES course(course_id)
        ON DELETE RESTRICT ON UPDATE CASCADE
);
```

---

## 7. NULL Semantics and Three-Valued Logic

NULL is one of the most subtle and frequently misunderstood concepts in the relational model. Understanding NULL semantics is critical for writing correct queries.

### What NULL Represents

NULL is **not a value**. It is a **marker** indicating:

| Meaning | Example |
|---------|---------|
| **Missing** (value exists but is unknown) | A student's phone number not yet recorded |
| **Inapplicable** (no value makes sense) | Apartment number for a student living in a house |
| **Withheld** (value exists but is not disclosed) | Salary of a colleague |

```
IMPORTANT DISTINCTION:
  NULL ≠ 0       (zero is a known value)
  NULL ≠ ''      (empty string is a known value)
  NULL ≠ FALSE   (false is a known boolean value)
  NULL ≠ NULL    (NULL compared to NULL is UNKNOWN, not TRUE!)
```

### Three-Valued Logic (3VL)

Standard Boolean logic has two values: TRUE and FALSE. SQL uses **three-valued logic** with TRUE, FALSE, and **UNKNOWN**.

Any comparison involving NULL produces UNKNOWN:

```
5 > 3        → TRUE
5 > 7        → FALSE
5 > NULL     → UNKNOWN
NULL > NULL  → UNKNOWN
NULL = NULL  → UNKNOWN
NULL = 5     → UNKNOWN
NULL <> 5    → UNKNOWN
NULL <> NULL → UNKNOWN
```

### Truth Tables for 3VL

**AND:**

```
         │  TRUE     FALSE    UNKNOWN
─────────┼──────────────────────────────
TRUE     │  TRUE     FALSE    UNKNOWN
FALSE    │  FALSE    FALSE    FALSE
UNKNOWN  │  UNKNOWN  FALSE    UNKNOWN
```

**OR:**

```
         │  TRUE     FALSE    UNKNOWN
─────────┼──────────────────────────────
TRUE     │  TRUE     TRUE     TRUE
FALSE    │  TRUE     FALSE    UNKNOWN
UNKNOWN  │  TRUE     UNKNOWN  UNKNOWN
```

**NOT:**

```
NOT TRUE    → FALSE
NOT FALSE   → TRUE
NOT UNKNOWN → UNKNOWN
```

### Implications for SQL Queries

```sql
-- Setup
CREATE TABLE test (id INT, value INT);
INSERT INTO test VALUES (1, 10), (2, 20), (3, NULL);

-- Query 1: WHERE value = 10
-- Returns: {(1, 10)}
-- Row (3, NULL): NULL = 10 → UNKNOWN → row excluded

-- Query 2: WHERE value <> 10
-- Returns: {(2, 20)}
-- Row (3, NULL): NULL <> 10 → UNKNOWN → row EXCLUDED!
-- SURPRISE: Neither "value = 10" nor "value <> 10" returns the NULL row!

-- Query 3: WHERE value = NULL
-- Returns: EMPTY SET!
-- NULL = NULL → UNKNOWN → all rows with NULL excluded
-- THIS IS A COMMON BUG. Use IS NULL instead.

-- Query 4: WHERE value IS NULL
-- Returns: {(3, NULL)}
-- IS NULL is a special operator that checks for NULL correctly

-- Query 5: WHERE value IS NOT NULL
-- Returns: {(1, 10), (2, 20)}
```

### NULL in Aggregates

```sql
-- NULL behavior in aggregate functions

CREATE TABLE scores (student_id CHAR(4), score INT);
INSERT INTO scores VALUES
    ('S001', 90), ('S002', 80), ('S003', NULL), ('S004', 70);

SELECT COUNT(*)        FROM scores;  -- 4 (counts all rows)
SELECT COUNT(score)    FROM scores;  -- 3 (NULLs excluded)
SELECT SUM(score)      FROM scores;  -- 240 (NULLs excluded)
SELECT AVG(score)      FROM scores;  -- 80 (240/3, NULLs excluded)
SELECT MIN(score)      FROM scores;  -- 70 (NULLs excluded)
SELECT MAX(score)      FROM scores;  -- 90 (NULLs excluded)

-- IMPORTANT: AVG(score) = 80, NOT 60!
-- AVG ignores NULLs, so it computes 240/3 = 80, not 240/4 = 60
-- If you want NULLs treated as 0:
SELECT AVG(COALESCE(score, 0)) FROM scores;  -- 60 (240/4)
```

### NULL in Boolean Expressions

```sql
-- The WHERE clause only keeps rows where the condition is TRUE
-- Rows where the condition is FALSE or UNKNOWN are filtered out

SELECT * FROM student WHERE gpa > 3.5;
-- If gpa IS NULL: NULL > 3.5 → UNKNOWN → row excluded

SELECT * FROM student WHERE gpa > 3.5 OR gpa <= 3.5;
-- If gpa IS NULL:
--   NULL > 3.5 → UNKNOWN
--   NULL <= 3.5 → UNKNOWN
--   UNKNOWN OR UNKNOWN → UNKNOWN
--   Row EXCLUDED even though logically gpa > 3.5 OR gpa <= 3.5
--   should cover all cases!

-- To include NULLs:
SELECT * FROM student WHERE gpa > 3.5 OR gpa IS NULL;
```

### NULL in DISTINCT and GROUP BY

```sql
-- In DISTINCT and GROUP BY, NULLs are considered equal
-- (This is an exception to "NULL ≠ NULL")

SELECT DISTINCT dept FROM instructor;
-- If multiple instructors have dept = NULL,
-- only ONE NULL appears in the result

SELECT dept, COUNT(*) FROM instructor GROUP BY dept;
-- All rows with dept = NULL are grouped together
```

### NULL in Joins

```sql
-- NULL values NEVER match in joins

-- STUDENT: (S001, 'Alice', 'CS'), (S002, 'Bob', NULL)
-- DEPARTMENT: ('CS', 'Computer Science'), ('EE', 'Electrical Engineering')

SELECT s.name, d.dept_name
FROM student s
JOIN department d ON s.dept = d.dept_id;

-- Result: Only ('Alice', 'Computer Science')
-- Bob is excluded because NULL = 'CS' → UNKNOWN, NULL = 'EE' → UNKNOWN
```

### COALESCE and NULLIF

```sql
-- COALESCE: Return first non-NULL argument
SELECT COALESCE(phone, email, 'No contact info') AS contact
FROM student;
-- If phone is NULL and email is 'alice@univ.edu':
--   returns 'alice@univ.edu'
-- If both are NULL:
--   returns 'No contact info'

-- NULLIF: Return NULL if two values are equal
SELECT NULLIF(actual_score, 0) AS adjusted_score
FROM test_results;
-- If actual_score = 0: returns NULL (treat 0 as unknown)
-- If actual_score = 85: returns 85
```

### Best Practices for NULLs

```
DO:
  ✓ Use IS NULL / IS NOT NULL for NULL checks
  ✓ Use COALESCE to provide default values
  ✓ Consider NULL behavior when writing WHERE clauses
  ✓ Be explicit about NULL handling in aggregate queries
  ✓ Document which columns allow NULLs and why

DON'T:
  ✗ Use = NULL or <> NULL (always returns UNKNOWN)
  ✗ Assume AVG includes NULLs as zeros
  ✗ Forget that OR with UNKNOWN may exclude rows
  ✗ Allow NULL in primary key columns
  ✗ Use NULL as a meaningful business value (use a flag column instead)
```

---

## 8. Relational Model in Practice

### Python Implementation of Relational Concepts

```python
"""
Simplified implementation of relational model concepts in Python.
For educational purposes only.
"""

from typing import Any, Optional
from dataclasses import dataclass


class Domain:
    """Represents a domain (set of allowed values)."""

    def __init__(self, name: str, check_fn=None):
        self.name = name
        self.check_fn = check_fn or (lambda x: True)

    def validate(self, value: Any) -> bool:
        if value is None:
            return True  # NULL is allowed unless explicitly constrained
        return self.check_fn(value)


class Relation:
    """Simplified relation (table) with constraint checking."""

    def __init__(self, name: str, attributes: list[str],
                 primary_key: list[str],
                 domains: Optional[dict] = None):
        self.name = name
        self.attributes = attributes
        self.primary_key = primary_key
        self.domains = domains or {}
        self.tuples: list[dict] = []

    def insert(self, values: dict) -> bool:
        """Insert a tuple with constraint checking."""
        # Check: all attributes present
        for attr in self.attributes:
            if attr not in values:
                raise ValueError(f"Missing attribute: {attr}")

        # Entity integrity: PK cannot be NULL
        for pk_attr in self.primary_key:
            if values[pk_attr] is None:
                raise ValueError(
                    f"Entity integrity violation: "
                    f"PK attribute '{pk_attr}' cannot be NULL"
                )

        # Domain constraints
        for attr, domain in self.domains.items():
            if not domain.validate(values.get(attr)):
                raise ValueError(
                    f"Domain violation: {values.get(attr)} "
                    f"not in domain {domain.name}"
                )

        # Key constraint: no duplicate PK
        pk_values = tuple(values[k] for k in self.primary_key)
        for existing in self.tuples:
            existing_pk = tuple(existing[k] for k in self.primary_key)
            if pk_values == existing_pk:
                raise ValueError(
                    f"Key violation: duplicate PK {pk_values}"
                )

        self.tuples.append(values)
        return True

    def select(self, predicate=None) -> list[dict]:
        """Select tuples matching a predicate (sigma operation)."""
        if predicate is None:
            return self.tuples.copy()
        return [t for t in self.tuples if predicate(t)]

    def project(self, attrs: list[str]) -> list[tuple]:
        """Project onto given attributes (pi operation)."""
        result = set()
        for t in self.tuples:
            projected = tuple(t[a] for a in attrs)
            result.add(projected)
        return [dict(zip(attrs, p)) for p in result]

    def __repr__(self):
        header = " | ".join(f"{a:>12}" for a in self.attributes)
        separator = "-" * len(header)
        rows = []
        for t in self.tuples:
            row = " | ".join(
                f"{str(t.get(a, 'NULL')):>12}" for a in self.attributes
            )
            rows.append(row)
        return f"\n{self.name}\n{separator}\n{header}\n{separator}\n" + \
               "\n".join(rows) + f"\n{separator}\n"


# --- Demonstration ---

# Define domains
gpa_domain = Domain("GPA", lambda x: 0.0 <= x <= 4.0)
year_domain = Domain("Year", lambda x: x in {1, 2, 3, 4})

# Create relation
student = Relation(
    name="STUDENT",
    attributes=["student_id", "name", "year", "gpa"],
    primary_key=["student_id"],
    domains={"gpa": gpa_domain, "year": year_domain}
)

# Insert valid tuples
student.insert({"student_id": "S001", "name": "Alice", "year": 3, "gpa": 3.85})
student.insert({"student_id": "S002", "name": "Bob", "year": 2, "gpa": 3.42})
student.insert({"student_id": "S003", "name": "Carol", "year": 4, "gpa": None})

print(student)

# Entity integrity violation
try:
    student.insert({"student_id": None, "name": "Dave", "year": 1, "gpa": 3.0})
except ValueError as e:
    print(f"Caught: {e}")

# Domain constraint violation
try:
    student.insert({"student_id": "S004", "name": "Eve", "year": 5, "gpa": 3.0})
except ValueError as e:
    print(f"Caught: {e}")

# Key constraint violation
try:
    student.insert({"student_id": "S001", "name": "Frank", "year": 1, "gpa": 2.5})
except ValueError as e:
    print(f"Caught: {e}")

# Selection
honors = student.select(lambda t: t["gpa"] is not None and t["gpa"] > 3.5)
print("Honors students:", honors)

# Projection
names = student.project(["name", "year"])
print("Names and years:", names)
```

### Common Relational Model Misconceptions

```
Misconception                          Reality
─────────────────────────────          ─────────────────────────────────
"Tables are relations"                 SQL tables are multisets, not sets
                                       (they allow duplicates)

"Column order matters"                 In the formal model, it does not
                                       (SQL preserves declaration order)

"NULL means empty or zero"             NULL means unknown, inapplicable,
                                       or missing — NOT empty/zero

"Every table must have an              The model requires a primary key,
 auto-increment ID"                    but it can be any candidate key

"Foreign keys must reference           FKs reference PRIMARY keys
 any column"                           (or UNIQUE keys in practice)

"Relational = SQL"                     SQL deviates from the relational
                                       model in several ways (duplicates,
                                       NULL handling, bag semantics)
```

---

## 9. Exercises

### Conceptual Questions

**Exercise 2.1**: State Codd's 12 rules in your own words. For each rule, give an example of a DBMS feature that satisfies it.

**Exercise 2.2**: Explain the difference between:
- (a) Superkey and candidate key
- (b) Primary key and alternate key
- (c) Natural key and surrogate key
- (d) Entity integrity and referential integrity

**Exercise 2.3**: Why does the relational model prohibit NULL values in primary keys? What problems would arise if this restriction were relaxed?

**Exercise 2.4**: Explain why NULL = NULL evaluates to UNKNOWN in SQL, not TRUE. Describe a scenario where treating NULL = NULL as TRUE would produce incorrect query results.

### Schema Design Questions

**Exercise 2.5**: Given the following requirements for a library system, identify all relations, their attributes, primary keys, foreign keys, and domain constraints:

- The library has books, each with an ISBN, title, author(s), publisher, year, and number of copies
- Members have a member ID, name, address, and phone number
- Members can borrow books. Each loan records the member, the book, the loan date, due date, and return date
- Each book belongs to one or more categories (e.g., Fiction, Science, History)

**Exercise 2.6**: For the following relation schema, identify all candidate keys:

```
FLIGHT(flight_number, airline, departure_city, arrival_city,
       departure_time, arrival_time, gate, aircraft_id)

Constraints:
  - A flight number uniquely identifies a flight
  - An aircraft can only be at one gate at a time
  - A gate can only have one aircraft at a time
```

**Exercise 2.7**: Given this schema, identify and fix all integrity constraint violations:

```sql
CREATE TABLE department (
    dept_id   CHAR(4) PRIMARY KEY,
    dept_name VARCHAR(50)
);

CREATE TABLE employee (
    emp_id    INT PRIMARY KEY,
    name      VARCHAR(50) NOT NULL,
    dept_id   CHAR(4) REFERENCES department(dept_id),
    salary    NUMERIC(10,2),
    mgr_id    INT REFERENCES employee(emp_id)
);

-- Attempted inserts:
INSERT INTO employee VALUES (1, 'Alice', 'CS01', 75000, NULL);
INSERT INTO department VALUES ('CS01', 'Computer Science');
INSERT INTO employee VALUES (2, NULL, 'CS01', 60000, 1);
INSERT INTO employee VALUES (3, 'Carol', 'EE01', 65000, 1);
INSERT INTO employee VALUES (NULL, 'Dave', 'CS01', 55000, 1);
```

### SQL and NULL Questions

**Exercise 2.8**: Predict the output of each query. Explain your reasoning.

```sql
CREATE TABLE t (a INT, b INT);
INSERT INTO t VALUES (1, 10), (2, NULL), (3, 30), (NULL, 40);

-- (a)
SELECT * FROM t WHERE b > 20;

-- (b)
SELECT * FROM t WHERE b > 20 OR b <= 20;

-- (c)
SELECT * FROM t WHERE a IN (1, 2, NULL);

-- (d)
SELECT COUNT(*), COUNT(a), COUNT(b), SUM(b), AVG(b) FROM t;

-- (e)
SELECT * FROM t WHERE b NOT IN (10, NULL);

-- (f)
SELECT COALESCE(a, 0) + COALESCE(b, 0) AS total FROM t;
```

**Exercise 2.9**: Write SQL queries that correctly handle NULLs for the following:

Given `EMPLOYEE(emp_id, name, dept, salary, bonus)` where `bonus` can be NULL:

- (a) Find employees whose total compensation (salary + bonus) exceeds 100,000
- (b) Find the average bonus, treating NULL bonuses as 0
- (c) Find departments where ALL employees have a non-NULL bonus
- (d) Find employees who have a different bonus from employee 'E001' (including NULLs)

### Design Exercise

**Exercise 2.10**: Design a relational schema for an online bookstore with the following requirements:

- Books have ISBN, title, price, publication date, and page count
- Authors have an ID, name, and biography. A book can have multiple authors, and an author can write multiple books
- Customers have an ID, name, email, and shipping address
- Customers can place orders. Each order has an order ID, order date, and status
- Each order contains one or more books with quantities
- Customers can write reviews for books (rating 1-5, text, date)

For your schema, provide:
1. All relation schemas with primary and foreign keys
2. Domain constraints for each attribute
3. Any additional integrity constraints (as CHECK or written in English)
4. Example valid and invalid tuples for at least two relations

---

**Previous**: [Introduction to Database Systems](./01_Introduction_to_Database_Systems.md) | **Next**: [Relational Algebra](./03_Relational_Algebra.md)
