# ER Modeling

**Previous**: [Relational Algebra](./03_Relational_Algebra.md) | **Next**: [Functional Dependencies](./05_Functional_Dependencies.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the purpose of conceptual database design and describe where ER modeling fits in the overall database design process.
2. Identify and model entity types, attributes (simple, composite, multivalued, derived), and relationship types using standard ER notation.
3. Specify cardinality ratios (1:1, 1:N, M:N) and participation constraints (total, partial) for relationship types.
4. Model weak entities and their identifying relationships, distinguishing them from regular entity types.
5. Apply Enhanced ER (EER) constructs — specialization, generalization, and aggregation — to represent complex data structures.
6. Execute the ER-to-relational mapping algorithm to transform an ER diagram into a complete relational schema.

---

The Entity-Relationship (ER) model, introduced by Peter Chen in 1976, is the most widely used approach for conceptual database design. It provides a graphical notation for representing the structure of data at a high level of abstraction, independent of any particular DBMS. This lesson covers the ER model, its Enhanced version (EER), and the systematic algorithm for converting an ER diagram into a relational schema.

## Table of Contents

1. [Conceptual Design Overview](#1-conceptual-design-overview)
2. [Entity Types and Entity Sets](#2-entity-types-and-entity-sets)
3. [Attributes](#3-attributes)
4. [Relationship Types](#4-relationship-types)
5. [Cardinality Constraints](#5-cardinality-constraints)
6. [Participation Constraints](#6-participation-constraints)
7. [Weak Entities](#7-weak-entities)
8. [Enhanced ER (EER) Model](#8-enhanced-er-eer-model)
9. [ER-to-Relational Mapping Algorithm](#9-er-to-relational-mapping-algorithm)
10. [Design Case Study: University Database](#10-design-case-study-university-database)
11. [Common Pitfalls and Best Practices](#11-common-pitfalls-and-best-practices)
12. [Exercises](#12-exercises)

---

## 1. Conceptual Design Overview

Database design follows a structured process from requirements to implementation:

```
┌──────────────────┐
│  Requirements    │  "What data do we need? What queries?"
│  Analysis        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Conceptual      │  ER Diagram (DBMS-independent)
│  Design          │  ← THIS LESSON
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Logical         │  Relational schema (tables, keys, constraints)
│  Design          │  ← ER-to-Relational Mapping
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Physical        │  Indexes, storage, partitioning, SQL DDL
│  Design          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Implementation  │  CREATE TABLE, INSERT, stored procedures
│  & Tuning        │
└──────────────────┘
```

### Why Conceptual Design?

- **Communication**: ER diagrams are understandable by non-technical stakeholders
- **Abstraction**: Focus on data structure without worrying about implementation
- **Correctness**: Catch design errors early before writing SQL
- **Documentation**: Serves as a living blueprint of the data model

### ER Diagram Notation

This lesson uses the original **Chen notation** (most common in textbooks):

```
┌──────────────────────────────────────────────────────────────┐
│  Symbol Legend                                                │
│                                                              │
│  ┌─────────┐       Entity type (strong)                     │
│  │  NAME   │                                                │
│  └─────────┘                                                │
│                                                              │
│  ┌═════════┐       Entity type (weak)                       │
│  ║  NAME   ║                                                │
│  └═════════┘                                                │
│                                                              │
│  ◇  or  ◇───       Relationship type                       │
│  <WORKS_FOR>                                                │
│                                                              │
│  (attribute)        Attribute (oval)                         │
│  ((derived))        Derived attribute (dashed oval)          │
│  {multivalued}      Multivalued attribute (double oval)      │
│                                                              │
│  ─── single line    Partial participation                   │
│  ═══ double line    Total participation                     │
│                                                              │
│  1, N, M            Cardinality markers                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Entity Types and Entity Sets

### Entity

An **entity** is a "thing" or object in the real world that is distinguishable from other objects. It can be physical (a person, a book) or conceptual (a course, a bank account).

### Entity Type

An **entity type** defines a collection of entities that have the same attributes. It is like a class or template.

### Entity Set

An **entity set** (or entity instance set) is the collection of all entities of a particular type at a given point in time. It is like the set of all objects of a class.

```
Entity Type:     STUDENT
                 (defines the structure: sid, name, year, dept)

Entity Set:      The current collection of student entities:
                 {(S001, Alice, 3, CS), (S002, Bob, 2, CS), ...}

Entity (Instance): A single student, e.g., (S001, Alice, 3, CS)
```

### Notation in ER Diagrams

```
              ┌───────────┐
              │  STUDENT  │
              └───────────┘
             /   |    |    \
          (sid) (name)(year)(dept)
           [PK]
```

---

## 3. Attributes

Attributes describe properties of an entity type. There are several kinds:

### Simple (Atomic) Attribute

A simple attribute cannot be divided into smaller components.

```
Examples:
  - student_id: "S001"
  - year: 3
  - gpa: 3.85
```

### Composite Attribute

A composite attribute can be divided into smaller sub-attributes.

```
              (name)
             /      \
       (first_name) (last_name)

              (address)
            /    |      \
      (street) (city) (zip_code)
                        |
                   (state) (country)
```

### Multivalued Attribute

A multivalued attribute can have multiple values for a single entity.

```
  {phone_numbers}    A student may have 0, 1, or many phone numbers.
  {skills}           An employee may have multiple skills.
  {email_addresses}  A person may have multiple email addresses.

Notation: Double oval or curly braces {attribute}
```

### Derived Attribute

A derived attribute's value can be computed from other attributes.

```
  ((age))            Derived from date_of_birth and current date
  ((total_credits))  Derived from summing credits of enrolled courses
  ((employee_count)) Derived from counting employees in a department

Notation: Dashed oval or double parentheses ((attribute))
```

### Key Attribute

A key attribute uniquely identifies each entity in the entity set.

```
  For STUDENT: student_id (underlined in diagrams)
  For COURSE: course_id
  For EMPLOYEE: employee_id or ssn

Notation: Underlined attribute name
```

### Composite Key

When no single attribute uniquely identifies an entity, a combination of attributes forms the key.

```
Example: ENROLLMENT might be identified by (student_id, course_id, semester)
```

### NULL Values

Attributes may have NULL values when:
- The value is **not applicable** (apartment number for a house)
- The value is **unknown** (phone number not provided)

### Attribute Summary

```
Attribute Types:
                                              ┌──────────────┐
                            ┌────────────────►│   Simple      │
                            │                 │  (atomic)     │
         ┌──────────┐      │                 └──────────────┘
         │ Structure ├──────┤
         └──────────┘      │                 ┌──────────────┐
                            └────────────────►│  Composite   │
                                              │  (divisible) │
                                              └──────────────┘

                                              ┌──────────────┐
                            ┌────────────────►│ Single-valued│
                            │                 └──────────────┘
         ┌──────────┐      │
         │ Cardinality├────┤
         └──────────┘      │                 ┌──────────────┐
                            └────────────────►│ Multivalued  │
                                              │ {attr}       │
                                              └──────────────┘

                                              ┌──────────────┐
                            ┌────────────────►│   Stored     │
                            │                 └──────────────┘
         ┌──────────┐      │
         │  Source   ├──────┤
         └──────────┘      │                 ┌──────────────┐
                            └────────────────►│  Derived     │
                                              │ ((attr))     │
                                              └──────────────┘
```

### ER Diagram with All Attribute Types

```
                        ┌───────────┐
                        │ EMPLOYEE  │
                        └───────────┘
                       / |  |  |   \  \
                     /   |  |  |    \   \
                   /     |  |  |     \    \
            (emp_id)  (name) | (hire_date) {phone}  ((age))
             [PK]    / \     |               Multi-   Derived
                   /     \ (salary)          valued
            (first) (last)
            Composite

  emp_id:    Simple, Key
  name:      Composite (first + last)
  salary:    Simple, Single-valued
  hire_date: Simple, Stored
  phone:     Simple, Multivalued
  age:       Simple, Derived (from birth_date)
```

---

## 4. Relationship Types

A **relationship type** defines an association between entity types. A **relationship instance** is an association between specific entity instances.

### Binary Relationships

A **binary relationship** involves two entity types (the most common case).

```
  ┌──────────┐          ┌──────────┐
  │ STUDENT  │──<ENROLLS>──│  COURSE  │
  └──────────┘          └──────────┘

  Relationship instances:
    (Alice, CS101), (Alice, CS301), (Bob, CS101), ...
```

### Ternary Relationships

A **ternary relationship** involves three entity types.

```
  ┌──────────┐
  │ SUPPLIER │
  └──────────┘
       │
       │
  ◇ SUPPLIES ◇
  /           \
  │             │
  ┌──────────┐  ┌──────────┐
  │  PART    │  │ PROJECT  │
  └──────────┘  └──────────┘

  Relationship instance: (Supplier1, PartA, ProjectX)
  Meaning: Supplier1 supplies PartA to ProjectX

  NOTE: A ternary relationship CANNOT always be decomposed into
  three binary relationships without information loss!
```

### Recursive (Unary) Relationships

A **recursive relationship** relates an entity type to itself.

```
  ┌──────────┐
  │ EMPLOYEE │
  └────┬─────┘
       │    │
       │    │
    (supervisor)
       │    │
       ├────┘
    <SUPERVISES>

  Relationship instance: (Manager_Alice, Employee_Bob)
  Meaning: Alice supervises Bob

  Role names are important:
    EMPLOYEE (as supervisor) ──<SUPERVISES>── EMPLOYEE (as supervisee)
```

### Relationship Attributes

Relationships can have their own attributes:

```
  ┌──────────┐                              ┌──────────┐
  │ STUDENT  │────<ENROLLS_IN>────│  COURSE  │
  └──────────┘      │                       └──────────┘
                  (grade)
                  (semester)

  The grade and semester belong to the RELATIONSHIP, not to either entity.
  A student has a grade for a specific course, not in general.
```

### Degree of a Relationship

The **degree** of a relationship type is the number of participating entity types.

```
Degree 1: Unary (recursive)     EMPLOYEE supervises EMPLOYEE
Degree 2: Binary                STUDENT enrolls in COURSE
Degree 3: Ternary               SUPPLIER supplies PART to PROJECT
Degree n: n-ary (rare)          Typically decomposed into binaries
```

---

## 5. Cardinality Constraints

Cardinality constraints specify the number of relationship instances an entity can participate in. For binary relationships, the three fundamental ratios are 1:1, 1:N, and M:N.

### One-to-One (1:1)

Each entity in A is associated with at most one entity in B, and vice versa.

```
  ┌──────────┐    1         1    ┌──────────┐
  │ EMPLOYEE │────<MANAGES>────│ DEPARTMENT│
  └──────────┘                   └──────────┘

  Each employee manages at most one department.
  Each department is managed by at most one employee.

  Instance:
    Alice  ────  CS Department
    Bob    ────  EE Department
    Carol  ────  (no department managed)
    Dave   ────  ME Department

  Mapping:
    A:  Alice ───► CS
    B:  Bob   ───► EE
    D:  Dave  ───► ME
```

### One-to-Many (1:N)

Each entity in A can be associated with many entities in B, but each entity in B is associated with at most one entity in A.

```
  ┌──────────┐    1         N    ┌──────────┐
  │DEPARTMENT│────<HAS>────│ EMPLOYEE │
  └──────────┘                   └──────────┘

  A department has many employees.
  An employee belongs to at most one department.

  Instance:
    CS ────┬──── Alice
           ├──── Bob
           └──── Eve
    EE ─────── Carol
    ME ─────── Dave
```

### Many-to-Many (M:N)

Each entity in A can be associated with many entities in B, and each entity in B can be associated with many entities in A.

```
  ┌──────────┐    M         N    ┌──────────┐
  │ STUDENT  │────<ENROLLS>────│  COURSE  │
  └──────────┘                   └──────────┘

  A student can enroll in many courses.
  A course can have many students.

  Instance:
    Alice ──┬── CS101
            ├── CS301
            └── MA101
    Bob   ──┬── CS101
            └── CS301
    Carol ──┬── EE201
            └── CS101
```

### Cardinality in ER Diagrams

There are two main conventions:

**Convention 1: Chen's notation (labels on lines)**

```
  ┌──────────┐    1    ┌──────────┐    N    ┌──────────┐
  │DEPARTMENT│────────<WORKS_IN>────────│ EMPLOYEE │
  └──────────┘                               └──────────┘
```

**Convention 2: (min,max) notation (more precise)**

```
  ┌──────────┐  (1,1)   ┌────────────┐  (1,N)  ┌──────────┐
  │ EMPLOYEE │──────────<WORKS_IN>──────────│DEPARTMENT│
  └──────────┘                                  └──────────┘

  Reading:
    An employee works in (1,1) department = exactly one department
    A department has (1,N) employees = one or more employees
```

### Cardinality Constraint Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cardinality Ratios                            │
│                                                                 │
│  1:1  ──  Each A maps to at most 1 B; each B to at most 1 A   │
│           Example: Employee MANAGES Department                  │
│                                                                 │
│  1:N  ──  Each A maps to many B; each B to at most 1 A        │
│           Example: Department HAS Employees                     │
│                                                                 │
│  M:N  ──  Each A maps to many B; each B maps to many A        │
│           Example: Student ENROLLS IN Course                    │
│                                                                 │
│  (min,max) notation:                                            │
│    (0,1)  ──  optional, at most one                            │
│    (1,1)  ──  mandatory, exactly one                           │
│    (0,N)  ──  optional, unbounded many                         │
│    (1,N)  ──  mandatory, at least one                          │
│    (3,5)  ──  minimum 3, maximum 5 (specific bounds)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Participation Constraints

Participation constraints specify whether every entity must participate in a relationship or if participation is optional.

### Total Participation (Mandatory)

Every entity in the entity set must participate in at least one relationship instance. Shown with a **double line** (===).

```
  ┌──────────┐           ┌──────────────┐          ┌──────────┐
  │ EMPLOYEE │═══════════<WORKS_IN>──────────│DEPARTMENT│
  └──────────┘                                      └──────────┘

  Every employee MUST work in some department.
  (No employee can exist without a department.)
```

### Partial Participation (Optional)

An entity may or may not participate in the relationship. Shown with a **single line** (---).

```
  ┌──────────┐           ┌──────────────┐          ┌──────────┐
  │ EMPLOYEE │───────────<MANAGES>═══════════│DEPARTMENT│
  └──────────┘                                      └──────────┘

  Not every employee manages a department (partial on EMPLOYEE side).
  Every department MUST be managed by someone (total on DEPARTMENT side).
```

### Combining Cardinality and Participation

```
Example: University ER Diagram (fragment)

  ┌──────────┐  (1,1)  ┌──────────────┐  (1,N)  ┌──────────┐
  │ EMPLOYEE │═════════<WORKS_IN>════════════│DEPARTMENT│
  └──────────┘                                    └──────────┘

  Reading the (min,max):
    Employee side: (1,1) → total participation, exactly one department
    Department side: (1,N) → total participation, at least one employee

  ┌──────────┐  (0,N)  ┌──────────────┐  (0,N)  ┌──────────┐
  │ STUDENT  │─────────<ENROLLS_IN>──────────│  COURSE  │
  └──────────┘                                    └──────────┘

  Reading the (min,max):
    Student side: (0,N) → partial (student may not be enrolled), many courses
    Course side: (0,N) → partial (course may have no students), many students
```

### Existence Dependency

When an entity's existence depends on its relationship with another entity, it has **total participation** in that relationship.

```
Example:
  A DEPENDENT (family member) cannot exist without an EMPLOYEE.
  Therefore, DEPENDENT has total participation in the
  HAS_DEPENDENT relationship.

  ┌──────────┐           ┌─────────────────┐          ┌═══════════┐
  │ EMPLOYEE │───────────<HAS_DEPENDENT>════════════║ DEPENDENT ║
  └──────────┘                                        └═══════════┘

  DEPENDENT is also a weak entity (discussed next).
```

---

## 7. Weak Entities

A **weak entity type** is an entity type that cannot be uniquely identified by its own attributes alone. It depends on a related **owner** (or **identifying**) entity type.

### Characteristics of Weak Entities

```
1. No primary key of its own
2. Has a PARTIAL KEY (discriminator) that distinguishes
   weak entities related to the same owner entity
3. Always has TOTAL PARTICIPATION in the identifying relationship
4. Existence-dependent on the owner entity
```

### Notation

```
  ┌──────────┐              ┌═══════════════┐
  │  OWNER   │══<IDENTIFIES>══║ WEAK ENTITY ║
  │ (strong) │                ║             ║
  └──────────┘                └═══════════════┘
                                   |
                             (partial_key)
                              [dashed underline]

  Double rectangle: weak entity type
  Double diamond: identifying relationship type
  Dashed underline: partial key (discriminator)
```

### Example: Employee and Dependent

```
  ┌──────────┐                ┌═════════════┐
  │ EMPLOYEE │══<HAS_DEPENDENT>══║ DEPENDENT  ║
  └──────────┘     1:N          └═════════════┘
       |                          |    |    |
    (emp_id)              (dep_name) (birth) (relationship)
     [PK]                 [partial key]

  EMPLOYEE has a primary key: emp_id
  DEPENDENT has a partial key: dep_name

  Full identification of a DEPENDENT:
    (owner's emp_id, dep_name)

  Example:
    Employee E001 (Alice) has dependents:
      (E001, "Tom") → Alice's son Tom
      (E001, "Sue") → Alice's daughter Sue

    Employee E002 (Bob) has dependents:
      (E002, "Tom") → Bob's son Tom (different person from E001's Tom!)

  Without the owner's key, "Tom" alone is ambiguous.
```

### Weak Entity vs. Strong Entity

```
┌──────────────────────────────┬──────────────────────────────────┐
│       Strong Entity          │         Weak Entity              │
├──────────────────────────────┼──────────────────────────────────┤
│ Has its own primary key      │ Has only a partial key           │
│ Can exist independently      │ Existence depends on owner       │
│ Single rectangle             │ Double rectangle                 │
│ Partial participation OK     │ Total participation required     │
│ Example: EMPLOYEE, COURSE    │ Example: DEPENDENT, ROOM         │
└──────────────────────────────┴──────────────────────────────────┘

More examples of weak entities:
  BUILDING (strong) → ROOM (weak): room_number is partial key
  INVOICE (strong) → LINE_ITEM (weak): line_number is partial key
  COURSE (strong) → SECTION (weak): section_number is partial key
```

---

## 8. Enhanced ER (EER) Model

The **Enhanced ER** (EER) model extends the basic ER model with additional concepts borrowed from object-oriented modeling: specialization, generalization, and inheritance.

### Specialization

**Specialization** is a top-down process of defining subclasses of an entity type based on distinguishing characteristics.

```
                    ┌──────────┐
                    │  PERSON  │
                    └────┬─────┘
                         │
                        / \
                       / d \        d = disjoint
                      /     \       o = overlapping
                     /       \
              ┌──────────┐  ┌──────────┐
              │ STUDENT  │  │ EMPLOYEE │
              └──────────┘  └──────────┘

  PERSON is the SUPERCLASS
  STUDENT and EMPLOYEE are SUBCLASSES
  The circle with d/o specifies the constraint
```

### Generalization

**Generalization** is a bottom-up process of abstracting common features from multiple entity types into a higher-level (general) entity type.

```
  Generalization example:

  We observe that CAR and TRUCK both have:
    - vehicle_id, make, model, year, color

  So we generalize:

                    ┌──────────┐
                    │ VEHICLE  │  ← generalized superclass
                    └────┬─────┘
                         │
                        / \
                       / d \
                      /     \
              ┌──────────┐  ┌──────────┐
              │   CAR    │  │  TRUCK   │
              └──────────┘  └──────────┘
              (num_doors)   (payload_capacity)
              (trunk_size)  (num_axles)
```

### Specialization/Generalization Constraints

Two orthogonal constraints govern specialization:

**Constraint 1: Disjointness**

```
Disjoint (d):     An entity can belong to AT MOST ONE subclass
                  Example: A vehicle is either a CAR or TRUCK, not both

Overlapping (o):  An entity can belong to MULTIPLE subclasses
                  Example: A person can be both a STUDENT and an EMPLOYEE
```

**Constraint 2: Completeness**

```
Total:    Every superclass entity MUST belong to at least one subclass
          Double line from superclass to specialization circle
          Example: Every VEHICLE must be either a CAR or a TRUCK

Partial:  A superclass entity MAY not belong to any subclass
          Single line from superclass to specialization circle
          Example: A PERSON may be neither a STUDENT nor an EMPLOYEE
```

### Four Combinations

```
┌──────────────────────────────────────────────────────────────────┐
│              Specialization Constraint Combinations               │
│                                                                  │
│  {disjoint, total}:     Every entity in exactly one subclass    │
│                          Example: VEHICLE → CAR xor TRUCK        │
│                                                                  │
│  {disjoint, partial}:   Entity in at most one subclass          │
│                          Example: ACCOUNT → SAVINGS xor CHECKING │
│                          (some accounts may be neither)          │
│                                                                  │
│  {overlapping, total}:  Entity in one or more subclasses        │
│                          Example: PERSON → STUDENT and/or        │
│                          EMPLOYEE (but must be at least one)     │
│                                                                  │
│  {overlapping, partial}: Entity in zero or more subclasses      │
│                          Example: PERSON → STUDENT and/or        │
│                          EMPLOYEE (can be neither)               │
└──────────────────────────────────────────────────────────────────┘
```

### Attribute Inheritance

Subclasses **inherit** all attributes of their superclass and can have additional attributes specific to the subclass.

```
            ┌──────────────────┐
            │      PERSON      │
            │──────────────────│
            │ person_id (PK)   │
            │ name             │
            │ date_of_birth    │
            │ email            │
            └────────┬─────────┘
                     │
                    / \
                   / o \     (overlapping, partial)
                  /     \
    ┌────────────────┐  ┌────────────────┐
    │    STUDENT     │  │   EMPLOYEE     │
    │────────────────│  │────────────────│
    │ + student_id   │  │ + emp_id       │
    │ + year         │  │ + salary       │
    │ + gpa          │  │ + hire_date    │
    │ + major        │  │ + department   │
    └────────────────┘  └────────────────┘

    A STUDENT inherits: person_id, name, date_of_birth, email
    and adds: student_id, year, gpa, major

    A PERSON who is both STUDENT and EMPLOYEE has ALL attributes.
```

### Multiple Inheritance and Category (Union Type)

A **category** (or union type) is a subclass with multiple possible superclasses:

```
            ┌──────────┐      ┌──────────┐      ┌──────────┐
            │  PERSON  │      │ COMPANY  │      │   BANK   │
            └────┬─────┘      └────┬─────┘      └────┬─────┘
                 │                 │                  │
                 └─────────────────┼──────────────────┘
                                   │
                                  (U)    ← Union / Category
                                   │
                            ┌──────────────┐
                            │   OWNER      │  (of a vehicle)
                            └──────────────┘

    An OWNER can be EITHER a PERSON, a COMPANY, OR a BANK.
    (As opposed to specialization, where subclasses share ONE superclass.)
```

---

## 9. ER-to-Relational Mapping Algorithm

This section presents the systematic **7-step algorithm** for converting an ER/EER diagram into a relational schema.

### Step 1: Map Strong Entity Types

For each strong (regular) entity type E, create a relation R that includes all simple attributes of E. Choose a primary key for R.

```
ER:
  ┌──────────┐
  │ EMPLOYEE │
  └──────────┘
  (emp_id), (name), (salary), (hire_date)

Relational:
  EMPLOYEE(emp_id, first_name, last_name, salary, hire_date)
  PK: emp_id

Rules:
  - Composite attributes: include only leaf components
    (name → first_name, last_name)
  - Derived attributes: omit (computed at query time)
  - Multivalued attributes: handled in Step 6
```

### Step 2: Map Weak Entity Types

For each weak entity type W with owner entity E, create a relation R that includes:
- All simple attributes of W
- The primary key of E as a foreign key
- Primary key of R = PK of E + partial key of W

```
ER:
  ┌──────────┐    1:N    ┌═══════════┐
  │ EMPLOYEE │══════════════║ DEPENDENT ║
  └──────────┘              └═══════════┘
  (emp_id)             (dep_name), (birth_date), (relationship)

Relational:
  DEPENDENT(emp_id, dep_name, birth_date, relationship)
  PK: (emp_id, dep_name)
  FK: emp_id → EMPLOYEE(emp_id) ON DELETE CASCADE
```

### Step 3: Map Binary 1:1 Relationship Types

Three approaches (choose based on participation constraints):

```
Approach A: Foreign Key Approach (preferred)
  Add the PK of one entity as a FK in the other.
  Prefer to add FK on the side with TOTAL participation.

  ER: EMPLOYEE (0,1) ──<MANAGES>── (1,1) DEPARTMENT

  Relational:
    EMPLOYEE(emp_id, name, salary)
    DEPARTMENT(dept_id, dept_name, mgr_emp_id, mgr_start_date)
                                   ^^^^^^^^^
                                   FK → EMPLOYEE(emp_id)
    (FK on DEPARTMENT because it has total participation:
     every department must have a manager)


Approach B: Merged Relation
  Merge both entity types into one relation.
  Only feasible when BOTH sides have total participation.


Approach C: Cross-Reference (Relationship Relation)
  Create a separate relation for the relationship.
  Useful when the relationship has many attributes.
```

### Step 4: Map Binary 1:N Relationship Types

Add the PK of the "1-side" entity as a FK in the "N-side" entity.

```
ER: DEPARTMENT (1) ──<HAS>── (N) EMPLOYEE

Relational:
  DEPARTMENT(dept_id, dept_name, budget)
  EMPLOYEE(emp_id, name, salary, dept_id)
                                 ^^^^^^^
                                 FK → DEPARTMENT(dept_id)

  Any relationship attributes go with the N-side entity:
  If HAS has (start_date), add it to EMPLOYEE.
```

### Step 5: Map Binary M:N Relationship Types

Create a new **relationship relation** R. Include:
- PKs of both participating entity types as FKs
- Any attributes of the relationship
- PK of R = combination of both FKs

```
ER: STUDENT (M) ──<ENROLLS>── (N) COURSE
    With attributes: grade, semester

Relational:
  STUDENT(student_id, name, year)
  COURSE(course_id, title, credits)
  ENROLLMENT(student_id, course_id, semester, grade)
  PK: (student_id, course_id, semester)
  FK: student_id → STUDENT(student_id)
      course_id → COURSE(course_id)
```

### Step 6: Map Multivalued Attributes

Create a new relation for each multivalued attribute. Include:
- The multivalued attribute
- The PK of the entity as a FK
- PK = FK + multivalued attribute

```
ER: EMPLOYEE has multivalued attribute {phone_numbers}

Relational:
  EMPLOYEE(emp_id, name, salary)
  EMPLOYEE_PHONE(emp_id, phone_number)
  PK: (emp_id, phone_number)
  FK: emp_id → EMPLOYEE(emp_id) ON DELETE CASCADE
```

### Step 7: Map Specialization/Generalization

Four options exist. The best choice depends on the constraints:

**Option A: Single Table with Type Discriminator**

```
PERSON(person_id, name, dob, email, person_type,
       -- STUDENT attributes (NULL if not a student)
       student_id, year, gpa, major,
       -- EMPLOYEE attributes (NULL if not an employee)
       emp_id, salary, hire_date, department)

person_type IN ('S', 'E', 'SE', 'N')  -- Student, Employee, Both, Neither

Pros: Simple, no joins needed
Cons: Many NULLs, harder to enforce subclass constraints
Best for: Few subclasses, many queries across all types
```

**Option B: Separate Tables for Each Subclass (Superclass PKs Inherited)**

```
PERSON(person_id, name, dob, email)
STUDENT(person_id, student_id, year, gpa, major)
  FK: person_id → PERSON(person_id)
EMPLOYEE(person_id, emp_id, salary, hire_date, department)
  FK: person_id → PERSON(person_id)

Pros: No NULLs, clean separation
Cons: Requires joins to get full data
Best for: Many subclass-specific attributes, overlapping allowed
```

**Option C: Separate Tables (Full Attributes in Each)**

```
STUDENT(person_id, name, dob, email, student_id, year, gpa, major)
EMPLOYEE(person_id, name, dob, email, emp_id, salary, hire_date, dept)

Pros: No joins needed for subclass queries
Cons: Redundancy (shared attributes duplicated), hard to query
      across all persons, overlapping requires data duplication
Best for: Disjoint, total specialization
```

**Option D: Hybrid (superclass + specialized tables)**

```
Choose based on usage patterns:
  - Frequently queried together → Option A
  - Mostly queried separately → Option C
  - Need flexibility → Option B
```

### Mapping Decision Table

```
┌──────────────────────────┬───────────────────────────────────┐
│ ER Construct             │ Relational Mapping                │
├──────────────────────────┼───────────────────────────────────┤
│ Strong entity            │ New relation, own PK              │
│ Weak entity              │ New relation, composite PK        │
│ 1:1 relationship         │ FK in one entity (total side)     │
│ 1:N relationship         │ FK in N-side entity               │
│ M:N relationship         │ New relation (bridge table)       │
│ Multivalued attribute    │ New relation                      │
│ Composite attribute      │ Flatten to components             │
│ Derived attribute        │ Omit (compute at query time)      │
│ Specialization/general.  │ Options A, B, C, or D             │
│ Ternary relationship     │ New relation with 3 FKs           │
│ Recursive relationship   │ FK to own table (or bridge table) │
└──────────────────────────┴───────────────────────────────────┘
```

---

## 10. Design Case Study: University Database

Let us design a complete ER diagram for a university database and map it to a relational schema.

### Requirements

```
1. The university has DEPARTMENTS, each with a name, building, and budget.
2. Each department has one CHAIRPERSON (a faculty member).
3. FACULTY members have an ID, name, rank, and salary. Each belongs to
   one department.
4. STUDENTS have an ID, name, year, and GPA. Each has a major department.
5. COURSES have an ID, title, credits, and belong to a department.
6. Faculty members TEACH courses. Each offering is in a specific semester.
   A course can have multiple sections taught by different faculty.
7. Students ENROLL in course sections and receive grades.
8. Students may have multiple PHONE NUMBERS and EMAIL ADDRESSES.
9. Faculty can ADVISE students (a student has one advisor).
```

### ER Diagram (ASCII)

```
 {phone}  {email}
    \      /
     \    /
  ┌──────────┐ (0,1)   (1,1) ┌──────────┐
  │ STUDENT  │═════<ADVISES>═════│ FACULTY  │
  └──────────┘                   └──────────┘
  (sid)(name)                    (fid)(name)
  (year)(gpa)                    (rank)(salary)
       |                              |
    (1,N)|                         (1,N)|
       |                              |
  <ENROLLED_IN>                   <TEACHES>
       |                              |
    (1,N)|                         (1,N)|
       |                              |
  ┌══════════════┐               ┌══════════════┐
  ║   SECTION    ║               ║   SECTION    ║
  ║ (sec_number) ║               ║              ║
  └══════════════┘               └══════════════┘
       |                              |
    (1,1)|                            |
       |                              |
  ┌──────────┐     (1,N)    (1,1)  ┌──────────┐  (1,1)  ┌──────────┐
  │  COURSE  │═════<OFFERED_BY>══════│DEPARTMENT│════<CHAIRS>═════│ FACULTY  │
  └──────────┘                       └──────────┘                  (already
  (cid)(title)                       (did)(name)                   shown)
  (credits)                          (building)
                                     (budget)
```

Note: The above is simplified. A more accurate representation would model SECTION as a weak entity of COURSE, with TEACHES linking FACULTY to SECTION and ENROLLED_IN linking STUDENT to SECTION.

### Refined ER Design

```
Entities:
  DEPARTMENT(dept_id, dept_name, building, budget)              Strong
  FACULTY(fac_id, name, rank, salary)                           Strong
  STUDENT(stu_id, name, year, gpa, {phone}, {email})           Strong + MV
  COURSE(course_id, title, credits)                             Strong
  SECTION(sec_number, semester, year)                           Weak (owner: COURSE)

Relationships:
  WORKS_IN:      FACULTY (N,1) --- DEPARTMENT     (1:N)
  MAJOR_IN:      STUDENT (N,1) --- DEPARTMENT     (1:N)
  CHAIRS:        FACULTY (1,0..1) --- DEPARTMENT  (1:1)
  OFFERS:        DEPARTMENT (1,N) --- COURSE      (1:N)
  HAS_SECTION:   COURSE (1,N) === SECTION         (identifying, 1:N)
  TEACHES:       FACULTY (1,N) --- SECTION        (1:1 per section)
  ENROLLED_IN:   STUDENT (M) --- SECTION (N)      (M:N, attrs: grade)
  ADVISES:       FACULTY (1) --- STUDENT (N)      (1:N)
```

### Relational Schema (Mapped)

```sql
-- Step 1: Strong entities
CREATE TABLE department (
    dept_id     CHAR(4)      PRIMARY KEY,
    dept_name   VARCHAR(50)  NOT NULL UNIQUE,
    building    VARCHAR(30),
    budget      NUMERIC(12,2) CHECK (budget >= 0)
);

CREATE TABLE faculty (
    fac_id      CHAR(5)      PRIMARY KEY,
    name        VARCHAR(50)  NOT NULL,
    rank        VARCHAR(20)  CHECK (rank IN
                  ('Lecturer','Assistant','Associate','Full')),
    salary      NUMERIC(10,2) CHECK (salary > 0),
    dept_id     CHAR(4)      NOT NULL,  -- Step 4: 1:N WORKS_IN
    FOREIGN KEY (dept_id) REFERENCES department(dept_id)
);

CREATE TABLE student (
    stu_id      CHAR(5)      PRIMARY KEY,
    name        VARCHAR(50)  NOT NULL,
    year        SMALLINT     CHECK (year BETWEEN 1 AND 4),
    gpa         NUMERIC(3,2) CHECK (gpa >= 0.0 AND gpa <= 4.0),
    major_id    CHAR(4),                -- Step 4: 1:N MAJOR_IN
    advisor_id  CHAR(5),                -- Step 4: 1:N ADVISES
    FOREIGN KEY (major_id) REFERENCES department(dept_id),
    FOREIGN KEY (advisor_id) REFERENCES faculty(fac_id)
);

CREATE TABLE course (
    course_id   CHAR(6)      PRIMARY KEY,
    title       VARCHAR(100) NOT NULL,
    credits     SMALLINT     NOT NULL CHECK (credits BETWEEN 1 AND 5),
    dept_id     CHAR(4)      NOT NULL,  -- Step 4: 1:N OFFERS
    FOREIGN KEY (dept_id) REFERENCES department(dept_id)
);

-- Step 2: Weak entity (SECTION identified by COURSE)
CREATE TABLE section (
    course_id   CHAR(6)      NOT NULL,
    sec_number  SMALLINT     NOT NULL,
    semester    VARCHAR(10)  NOT NULL,
    sec_year    SMALLINT     NOT NULL,
    fac_id      CHAR(5),                -- Step 4: 1:N TEACHES
    PRIMARY KEY (course_id, sec_number, semester, sec_year),
    FOREIGN KEY (course_id) REFERENCES course(course_id)
        ON DELETE CASCADE,
    FOREIGN KEY (fac_id) REFERENCES faculty(fac_id)
);

-- Step 3: 1:1 CHAIRS (FK on department side, total participation)
ALTER TABLE department
    ADD COLUMN chair_fac_id CHAR(5),
    ADD CONSTRAINT fk_chair
        FOREIGN KEY (chair_fac_id) REFERENCES faculty(fac_id);

-- Step 5: M:N ENROLLED_IN
CREATE TABLE enrollment (
    stu_id      CHAR(5)      NOT NULL,
    course_id   CHAR(6)      NOT NULL,
    sec_number  SMALLINT     NOT NULL,
    semester    VARCHAR(10)  NOT NULL,
    sec_year    SMALLINT     NOT NULL,
    grade       VARCHAR(2),
    PRIMARY KEY (stu_id, course_id, sec_number, semester, sec_year),
    FOREIGN KEY (stu_id) REFERENCES student(stu_id),
    FOREIGN KEY (course_id, sec_number, semester, sec_year)
        REFERENCES section(course_id, sec_number, semester, sec_year)
);

-- Step 6: Multivalued attributes
CREATE TABLE student_phone (
    stu_id      CHAR(5)      NOT NULL,
    phone       VARCHAR(20)  NOT NULL,
    PRIMARY KEY (stu_id, phone),
    FOREIGN KEY (stu_id) REFERENCES student(stu_id) ON DELETE CASCADE
);

CREATE TABLE student_email (
    stu_id      CHAR(5)      NOT NULL,
    email       VARCHAR(100) NOT NULL,
    PRIMARY KEY (stu_id, email),
    FOREIGN KEY (stu_id) REFERENCES student(stu_id) ON DELETE CASCADE
);
```

### Schema Diagram Summary

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  DEPARTMENT  │     │   FACULTY    │     │   STUDENT    │
├──────────────┤     ├──────────────┤     ├──────────────┤
│PK dept_id    │◄────│PK fac_id     │◄────│PK stu_id     │
│   dept_name  │  FK │   name       │  FK │   name       │
│   building   │dept │   rank       │adv  │   year       │
│   budget     │     │   salary     │     │   gpa        │
│FK chair_fac  │─────│FK dept_id ───┘     │FK major_id──►│ DEPARTMENT
└──────────────┘     └──────────────┘     │FK advisor_id─│►FACULTY
       │                    │             └──────────────┘
       │                    │                    │
       │              ┌─────┘                    │
       │              │                          │
  ┌──────────┐  ┌─────┴──────┐  ┌───────────────┴──┐
  │  COURSE  │  │  SECTION   │  │   ENROLLMENT     │
  ├──────────┤  ├────────────┤  ├──────────────────┤
  │PK crs_id │◄─│PK crs_id   │◄─│PK stu_id         │
  │   title  │  │PK sec_num  │  │PK crs_id         │
  │   credits│  │PK semester │  │PK sec_num        │
  │FK dept_id│  │PK sec_year │  │PK semester       │
  └──────────┘  │FK fac_id   │  │PK sec_year       │
                └────────────┘  │   grade           │
                                └──────────────────┘

  ┌─────────────────┐  ┌─────────────────┐
  │ STUDENT_PHONE   │  │ STUDENT_EMAIL   │
  ├─────────────────┤  ├─────────────────┤
  │PK,FK stu_id     │  │PK,FK stu_id     │
  │PK    phone      │  │PK    email      │
  └─────────────────┘  └─────────────────┘
```

---

## 11. Common Pitfalls and Best Practices

### Pitfall 1: Fan Trap

A **fan trap** occurs when a path through multiple 1:N relationships fans out, causing ambiguous associations.

```
Problem:
  DEPARTMENT ─1:N─ EMPLOYEE
  DEPARTMENT ─1:N─ PROJECT

  Query: "Which employees work on which projects?"
  The path goes DEPARTMENT → EMPLOYEE and DEPARTMENT → PROJECT
  but there is NO direct link between EMPLOYEE and PROJECT!

Solution: Add a direct WORKS_ON relationship between EMPLOYEE and PROJECT.
```

### Pitfall 2: Chasm Trap

A **chasm trap** occurs when a path does not exist between entity types that should be related.

```
Problem:
  DEPARTMENT ─1:N─ EMPLOYEE (partial)
  EMPLOYEE ─1:N─ PROJECT

  If some departments have no employees, there is a "chasm"
  (gap) in the path from DEPARTMENT to PROJECT.

Solution: Add a direct HAS_PROJECT relationship between DEPARTMENT and PROJECT.
```

### Pitfall 3: Overuse of Multivalued Attributes

```
Bad:
  PERSON with {address}   ← if addresses need their own attributes
                             (street, city, state, zip), this is wrong

Better:
  PERSON ─1:N─ ADDRESS
  ADDRESS(address_id, street, city, state, zip_code)
```

### Pitfall 4: Missing Relationship Attributes

```
Bad:
  STUDENT has grade attribute    ← grade for which course?

Better:
  STUDENT ─M:N─ COURSE with relationship attribute: grade
```

### Best Practices

```
1. Start with entities, then relationships, then attributes
2. Every entity must have a key attribute
3. Avoid redundant relationships (derivable from others)
4. Use weak entities only when truly necessary
5. Prefer binary relationships over ternary when possible
6. Document all assumptions and constraints
7. Validate with stakeholders using concrete examples
8. Name entities as singular nouns (STUDENT, not STUDENTS)
9. Name relationships as verbs (ENROLLS_IN, TEACHES)
10. Use (min,max) notation for precise constraints
```

---

## 12. Exercises

### Basic Concepts

**Exercise 4.1**: For each of the following, identify whether it should be modeled as an entity type, a relationship type, or an attribute. Justify your answer.

- (a) Employee name
- (b) Department
- (c) Marriage (between two persons)
- (d) Student GPA
- (e) Book ISBN
- (f) Course enrollment
- (g) Employee skill (assuming an employee can have many skills)
- (h) Project deadline

**Exercise 4.2**: Classify each of the following attributes:

| Attribute | Simple/Composite | Single/Multi | Stored/Derived | Key? |
|-----------|-------------------|--------------|----------------|------|
| SSN | | | | |
| Full name (first + middle + last) | | | | |
| Phone numbers (multiple) | | | | |
| Age (given date of birth) | | | | |
| Email address | | | | |
| Address (street, city, state, zip) | | | | |

### ER Design

**Exercise 4.3**: Draw an ER diagram for a hospital system with the following requirements:
- Patients have an ID, name, date of birth, and blood type
- Doctors have an ID, name, specialty, and phone
- Each patient is assigned to a primary doctor
- Doctors can prescribe medications to patients (record date and dosage)
- Medications have a code, name, and manufacturer
- Patients can have multiple allergies

Specify cardinality and participation constraints.

**Exercise 4.4**: Draw an ER diagram for an online learning platform:
- Instructors create courses. A course has a title, description, and price.
- Courses contain multiple lessons in a specific order.
- Students can enroll in courses and track their progress per lesson (completion status, time spent).
- Students can rate and review courses (1-5 stars, text).
- There are quizzes at the end of each lesson. Students attempt quizzes and receive scores.

Identify all entity types, relationship types, attributes, keys, and constraints.

### Cardinality and Participation

**Exercise 4.5**: For each of the following scenarios, determine the cardinality ratio (1:1, 1:N, M:N) and participation constraints (total/partial):

- (a) A country has one capital city; a capital city belongs to one country
- (b) A student lives in one dormitory room; a room can house multiple students
- (c) An author can write many books; a book can have many authors
- (d) An employee works on multiple projects; a project has multiple employees
- (e) A person has one passport; a passport belongs to one person (not everyone has a passport)

### Weak Entities

**Exercise 4.6**: For each pair, determine which (if any) should be a weak entity:

- (a) Building and Room
- (b) Invoice and LineItem
- (c) Student and Course
- (d) Bank and Branch
- (e) Order and OrderItem

For each weak entity, identify the partial key and the identifying relationship.

### EER

**Exercise 4.7**: Design an EER diagram for the following:

A company has **employees**. Employees are specialized into **managers**, **engineers**, and **secretaries**. An engineer can be further specialized into **software engineers** and **hardware engineers**.

- Determine if the specializations should be disjoint or overlapping
- Determine if they should be total or partial
- Add at least two specific attributes for each subclass
- Draw the complete EER diagram

### ER-to-Relational Mapping

**Exercise 4.8**: Given the following ER diagram, apply the 7-step mapping algorithm to produce a complete relational schema with SQL DDL:

```
                     {skill}
                       |
  ┌──────────┐  1:N  ┌──────────────┐  M:N   ┌──────────┐
  │DEPARTMENT│═══════<WORKS_IN>═══════│ EMPLOYEE │════<WORKS_ON>════│ PROJECT  │
  └──────────┘        └──────────────┘  |hours|  └──────────┘
  (dept_id)          (emp_id)(name)              (proj_id)(name)
  (name)             (salary)                    (budget)
  (budget)           (birth_date)                (location)
                          |
                     1:N  |
                          |
                ┌═══════════════┐
                ║  DEPENDENT   ║
                ║(dep_name)    ║
                ║(birth_date)  ║
                ║(relationship)║
                └═══════════════┘
```

Include: all tables, PKs, FKs, domain constraints, and ON DELETE actions.

**Exercise 4.9**: Map the following specialization hierarchy to a relational schema using each of the three approaches (Option A: single table, Option B: superclass + subclass tables, Option C: separate tables). Discuss the trade-offs.

```
            VEHICLE
           (vin, make, model, year, color)
              |
            /   \
          d,t
          /       \
       CAR          TRUCK
    (num_doors,   (payload_cap,
     trunk_vol)    num_axles,
                   cab_type)
```

### Design Challenge

**Exercise 4.10**: A local library needs a database system. Design the complete conceptual schema (ER/EER diagram) and map it to a relational schema. The requirements are:

1. The library has multiple **branches**, each with a name, address, and phone
2. **Books** are identified by ISBN and have a title, publication year, and edition
3. Each book has one or more **authors**
4. Books belong to one or more **categories** (Fiction, Science, History, etc.)
5. Each branch maintains multiple **copies** of each book. Each copy has a copy number and condition status
6. **Members** have a card number, name, address, and phone. Members register at a specific branch
7. Members can **borrow** copies. Each borrowing records the borrow date, due date, and return date
8. Members can **reserve** books at a branch (not a specific copy)
9. **Employees** work at branches. There are **librarians** and **assistants** (disjoint specialization)
10. Each branch has one librarian who serves as the **branch manager**

Your deliverables:
- Complete ER/EER diagram with all constraints
- Relational schema (apply all 7 steps)
- SQL DDL for at least 5 key tables
- Three sample queries that demonstrate the design supports the library's needs

---

**Previous**: [Relational Algebra](./03_Relational_Algebra.md) | **Next**: [Functional Dependencies](./05_Functional_Dependencies.md)
