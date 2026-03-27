# Relational Algebra

**Previous**: [Relational Model](./02_Relational_Model.md) | **Next**: [ER Modeling](./04_ER_Modeling.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the role of relational algebra as the formal foundation of SQL and describe how query optimizers use algebraic expressions.
2. Apply unary operations — selection (σ) and projection (π) — to filter and reshape relations.
3. Use binary operations — union, intersection, difference, Cartesian product, and join variants — to combine relations and retrieve related data.
4. Construct relational algebra expressions for multi-table queries and translate between SQL and their algebraic equivalents.
5. Build and simplify query trees, applying algebraic equivalence rules to optimize query plans.
6. Distinguish between relational algebra (procedural) and relational calculus (declarative) and explain their expressive equivalence.

---

Relational algebra is the formal query language of the relational model. It provides a collection of operators that take one or two relations as input and produce a new relation as output. Understanding relational algebra is essential because it underlies SQL query processing: every SQL query is internally translated into a relational algebra expression that the query optimizer can then transform and improve.

## Table of Contents

1. [Overview of Relational Algebra](#1-overview-of-relational-algebra)
2. [Unary Operations](#2-unary-operations)
3. [Set Operations](#3-set-operations)
4. [Binary Operations: Cartesian Product and Joins](#4-binary-operations-cartesian-product-and-joins)
5. [Division](#5-division)
6. [Additional Operations](#6-additional-operations)
7. [Query Trees and Algebraic Optimization](#7-query-trees-and-algebraic-optimization)
8. [Relational Calculus (Brief Introduction)](#8-relational-calculus-brief-introduction)
9. [Equivalence with SQL](#9-equivalence-with-sql)
10. [Complete Worked Examples](#10-complete-worked-examples)
11. [Exercises](#11-exercises)

---

## 1. Overview of Relational Algebra

### What Is Relational Algebra?

Relational algebra is a **procedural** query language: it describes a sequence of operations to compute the desired result. Each operation takes one or more relations and produces a new relation.

```
Properties:
  - Closure: The result of every operation is a relation
             (enabling composition of operations)
  - Set semantics: Relations are sets (no duplicate tuples)
  - Foundation: Based on set theory and first-order logic
```

### Categories of Operations

```
┌─────────────────────────────────────────────────────────────┐
│              Relational Algebra Operations                    │
│                                                             │
│  Fundamental (cannot be expressed using other operations):   │
│    σ  Selection          (filter rows)                      │
│    π  Projection         (choose columns)                   │
│    ρ  Rename             (rename relation/attributes)       │
│    ∪  Union              (combine two relations)            │
│    −  Set Difference     (tuples in R but not in S)         │
│    ×  Cartesian Product  (all combinations)                 │
│                                                             │
│  Derived (can be expressed using fundamental operations):    │
│    ⋈  Join (various types)                                  │
│    ∩  Intersection                                          │
│    ÷  Division                                              │
│    ←  Assignment                                            │
│    δ  Duplicate elimination (for bag semantics)             │
│    γ  Grouping/Aggregation                                  │
│    τ  Sorting                                               │
└─────────────────────────────────────────────────────────────┘
```

### Running Example Database

Throughout this lesson, we use the following sample database:

```
STUDENT:
┌──────┬───────────┬──────┬──────┐
│ sid  │ name      │ year │ dept │
├──────┼───────────┼──────┼──────┤
│ S001 │ Alice     │  3   │ CS   │
│ S002 │ Bob       │  2   │ CS   │
│ S003 │ Carol     │  4   │ EE   │
│ S004 │ Dave      │  3   │ ME   │
│ S005 │ Eve       │  1   │ CS   │
└──────┴───────────┴──────┴──────┘

COURSE:
┌───────┬──────────────────────┬─────────┬──────┐
│ cid   │ title                │ credits │ dept │
├───────┼──────────────────────┼─────────┼──────┤
│ CS101 │ Intro to CS          │ 3       │ CS   │
│ CS301 │ Database Theory      │ 3       │ CS   │
│ CS401 │ Machine Learning     │ 4       │ CS   │
│ EE201 │ Circuit Analysis     │ 3       │ EE   │
│ MA101 │ Calculus I           │ 4       │ MA   │
└───────┴──────────────────────┴─────────┴──────┘

ENROLLMENT:
┌──────┬───────┬───────┐
│ sid  │ cid   │ grade │
├──────┼───────┼───────┤
│ S001 │ CS101 │ A     │
│ S001 │ CS301 │ A+    │
│ S001 │ MA101 │ B+    │
│ S002 │ CS101 │ B     │
│ S002 │ CS301 │ A-    │
│ S003 │ EE201 │ A     │
│ S003 │ CS101 │ B+    │
│ S004 │ CS101 │ C     │
│ S005 │ CS101 │ A     │
│ S005 │ MA101 │ A-    │
└──────┴───────┴───────┘

INSTRUCTOR:
┌──────┬──────────┬──────┬────────┐
│ iid  │ name     │ dept │ salary │
├──────┼──────────┼──────┼────────┤
│ I001 │ Prof. Kim│ CS   │ 95000  │
│ I002 │ Prof. Lee│ CS   │ 88000  │
│ I003 │ Prof. Park│ EE  │ 92000  │
│ I004 │ Prof. Choi│ MA  │ 85000  │
└──────┴──────────┴──────┴────────┘
```

---

## 2. Unary Operations

Unary operations take a single relation as input.

### Selection (sigma)

The **selection** operation filters rows based on a condition (predicate).

```
Notation:   σ_condition(R)

Output:     A relation containing only the tuples from R
            that satisfy the condition.

Schema:     Same as R (all attributes preserved)
```

**Formal definition:**

```
σ_condition(R) = { t | t ∈ R  AND  condition(t) is TRUE }
```

**Examples:**

```
1. Select CS students:

   σ_{dept='CS'}(STUDENT)

   Result:
   ┌──────┬───────┬──────┬──────┐
   │ sid  │ name  │ year │ dept │
   ├──────┼───────┼──────┼──────┤
   │ S001 │ Alice │  3   │ CS   │
   │ S002 │ Bob   │  2   │ CS   │
   │ S005 │ Eve   │  1   │ CS   │
   └──────┴───────┴──────┴──────┘


2. Select 3rd-year CS students:

   σ_{dept='CS' AND year=3}(STUDENT)

   Result:
   ┌──────┬───────┬──────┬──────┐
   │ sid  │ name  │ year │ dept │
   ├──────┼───────┼──────┼──────┤
   │ S001 │ Alice │  3   │ CS   │
   └──────┴───────┴──────┴──────┘


3. Select courses with 4 credits:

   σ_{credits=4}(COURSE)

   Result:
   ┌───────┬──────────────────┬─────────┬──────┐
   │ cid   │ title            │ credits │ dept │
   ├───────┼──────────────────┼─────────┼──────┤
   │ CS401 │ Machine Learning │ 4       │ CS   │
   │ MA101 │ Calculus I       │ 4       │ MA   │
   └───────┴──────────────────┴─────────┴──────┘
```

**Selection conditions can use:**
- Comparison operators: =, ≠, <, >, ≤, ≥
- Logical connectives: AND (∧), OR (∨), NOT (¬)
- Attribute names and constants

### Projection (pi)

The **projection** operation selects specific columns (attributes) and removes duplicates.

```
Notation:   π_{attr_list}(R)

Output:     A relation containing only the specified attributes,
            with duplicate tuples removed.

Schema:     Only the attributes in attr_list
```

**Formal definition:**

```
π_{A1, A2, ..., Ak}(R) = { <t[A1], t[A2], ..., t[Ak]> | t ∈ R }
```

**Examples:**

```
1. Project student names and departments:

   π_{name, dept}(STUDENT)

   Result:
   ┌───────┬──────┐
   │ name  │ dept │
   ├───────┼──────┤
   │ Alice │ CS   │
   │ Bob   │ CS   │
   │ Carol │ EE   │
   │ Dave  │ ME   │
   │ Eve   │ CS   │
   └───────┴──────┘


2. Project distinct departments from STUDENT:

   π_{dept}(STUDENT)

   Result:
   ┌──────┐
   │ dept │
   ├──────┤
   │ CS   │
   │ EE   │
   │ ME   │
   └──────┘
   (Note: CS appears only ONCE because duplicates are removed)


3. Compose selection and projection:

   "Find names of CS students"
   π_{name}(σ_{dept='CS'}(STUDENT))

   Step 1: σ_{dept='CS'}(STUDENT) → {Alice/CS, Bob/CS, Eve/CS}
   Step 2: π_{name}(...) → {Alice, Bob, Eve}

   Result:
   ┌───────┐
   │ name  │
   ├───────┤
   │ Alice │
   │ Bob   │
   │ Eve   │
   └───────┘
```

### Rename (rho)

The **rename** operation changes the name of a relation and/or its attributes.

```
Notation:   ρ_{S(B1, B2, ..., Bn)}(R)

            Renames relation R to S and attributes to B1, B2, ..., Bn

Shorthand:  ρ_S(R)          -- rename relation only
            ρ_{(B1,...,Bn)}(R) -- rename attributes only
```

**Examples:**

```
1. Rename STUDENT to S:

   ρ_S(STUDENT)
   → Same tuples, but relation is now called S

2. Rename for self-join preparation:

   ρ_{S1(sid1, name1, year1, dept1)}(STUDENT)
   ρ_{S2(sid2, name2, year2, dept2)}(STUDENT)

   Now we can join S1 and S2 without attribute name conflicts.

3. Practical use - find pairs of students in the same department:

   σ_{S1.dept = S2.dept AND S1.sid < S2.sid}(
       ρ_{S1(sid1, name1, year1, dept1)}(STUDENT) ×
       ρ_{S2(sid2, name2, year2, dept2)}(STUDENT)
   )
```

---

## 3. Set Operations

Set operations require **union-compatible** (or type-compatible) relations: they must have the same number of attributes, and corresponding attributes must have compatible domains.

```
Union compatibility:
  R(A1: D1, A2: D2, ..., An: Dn)
  S(B1: D1, B2: D2, ..., Bn: Dn)

  Same number of attributes (n) and compatible domains.
  Attribute names may differ (result uses R's names by convention).
```

### Union

```
Notation:   R ∪ S

Output:     All tuples that are in R, in S, or in both.
            Duplicates are eliminated.

R ∪ S = { t | t ∈ R  OR  t ∈ S }
```

**Example:**

```
CS_STUDENTS = π_{sid, name}(σ_{dept='CS'}(STUDENT))
┌──────┬───────┐
│ sid  │ name  │
├──────┼───────┤
│ S001 │ Alice │
│ S002 │ Bob   │
│ S005 │ Eve   │
└──────┴───────┘

YEAR3_STUDENTS = π_{sid, name}(σ_{year=3}(STUDENT))
┌──────┬───────┐
│ sid  │ name  │
├──────┼───────┤
│ S001 │ Alice │
│ S004 │ Dave  │
└──────┴───────┘

CS_STUDENTS ∪ YEAR3_STUDENTS =
┌──────┬───────┐
│ sid  │ name  │
├──────┼───────┤
│ S001 │ Alice │  ← appears in both, but listed only once
│ S002 │ Bob   │
│ S005 │ Eve   │
│ S004 │ Dave  │
└──────┴───────┘
```

### Set Difference

```
Notation:   R − S  (or R \ S)

Output:     Tuples in R that are NOT in S.

R − S = { t | t ∈ R  AND  t ∉ S }
```

**Example:**

```
"CS students who are NOT in year 3"

CS_STUDENTS − YEAR3_STUDENTS =
┌──────┬──────┐
│ sid  │ name │
├──────┼──────┤
│ S002 │ Bob  │
│ S005 │ Eve  │
└──────┴──────┘

Note: Alice (S001) is removed because she is in YEAR3_STUDENTS.
Note: R − S ≠ S − R  (set difference is NOT commutative)

YEAR3_STUDENTS − CS_STUDENTS =
┌──────┬──────┐
│ sid  │ name │
├──────┼──────┤
│ S004 │ Dave │
└──────┴──────┘
```

### Intersection

```
Notation:   R ∩ S

Output:     Tuples that are in BOTH R and S.

R ∩ S = { t | t ∈ R  AND  t ∈ S }

Note: R ∩ S = R − (R − S)  (intersection is derivable)
```

**Example:**

```
"Students who are in CS AND in year 3"

CS_STUDENTS ∩ YEAR3_STUDENTS =
┌──────┬───────┐
│ sid  │ name  │
├──────┼───────┤
│ S001 │ Alice │
└──────┴───────┘
```

### Set Operations Summary

```
     R           S                R ∪ S         R − S         R ∩ S

  ┌──────┐   ┌──────┐         ┌──────┐      ┌──────┐     ┌──────┐
  │ a    │   │ a    │         │ a    │      │ b    │     │ a    │
  │ b    │   │ a    │         │ b    │      │ c    │     │      │
  │ c    │   │ d    │         │ c    │      └──────┘     └──────┘
  └──────┘   └──────┘         │ d    │
                              └──────┘

  Venn diagram:
       ┌─────────────┐
       │  R          │
       │      ┌──────┼──────┐
       │      │ R∩S  │      │
       │      │      │   S  │
       └──────┼──────┘      │
              │             │
              └─────────────┘

  R ∪ S = entire shaded area
  R − S = left only (not overlap)
  R ∩ S = overlap only
  S − R = right only (not overlap)
```

---

## 4. Binary Operations: Cartesian Product and Joins

### Cartesian Product (Cross Product)

```
Notation:   R × S

Output:     Every tuple in R combined with every tuple in S.
            If R has n tuples and S has m tuples, R × S has n × m tuples.
            If R has p attributes and S has q attributes, R × S has p + q attributes.
```

**Formal definition:**

```
R × S = { <t_r, t_s> | t_r ∈ R  AND  t_s ∈ S }
```

**Example:**

```
A = {(a1, b1), (a2, b2)}     B = {(c1, d1), (c2, d2), (c3, d3)}

A × B =
┌────┬────┬────┬────┐
│ A  │ B  │ C  │ D  │
├────┼────┼────┼────┤
│ a1 │ b1 │ c1 │ d1 │
│ a1 │ b1 │ c2 │ d2 │
│ a1 │ b1 │ c3 │ d3 │
│ a2 │ b2 │ c1 │ d1 │
│ a2 │ b2 │ c2 │ d2 │
│ a2 │ b2 │ c3 │ d3 │
└────┴────┴────┴────┘

|A| = 2, |B| = 3, |A × B| = 2 × 3 = 6
```

The Cartesian product by itself is rarely useful because it produces many meaningless combinations. Its power comes when combined with selection (which gives us joins).

### Theta Join

A **theta join** combines Cartesian product with selection:

```
Notation:   R ⋈_θ S   (where θ is a condition)

Definition: R ⋈_θ S = σ_θ(R × S)

The condition θ can use any comparison: =, ≠, <, >, ≤, ≥
```

**Example:**

```
"Find students and courses in the same department"

STUDENT ⋈_{STUDENT.dept = COURSE.dept} COURSE

Equivalent to: σ_{STUDENT.dept = COURSE.dept}(STUDENT × COURSE)

Result (partial):
┌──────┬───────┬──────┬──────┬───────┬──────────────────┬─────────┬───────┐
│ sid  │ name  │ year │ s.dept│ cid  │ title            │ credits │c.dept │
├──────┼───────┼──────┼──────┼───────┼──────────────────┼─────────┼───────┤
│ S001 │ Alice │  3   │ CS   │ CS101 │ Intro to CS      │ 3       │ CS    │
│ S001 │ Alice │  3   │ CS   │ CS301 │ Database Theory  │ 3       │ CS    │
│ S001 │ Alice │  3   │ CS   │ CS401 │ Machine Learning │ 4       │ CS    │
│ S002 │ Bob   │  2   │ CS   │ CS101 │ Intro to CS      │ 3       │ CS    │
│ S002 │ Bob   │  2   │ CS   │ CS301 │ Database Theory  │ 3       │ CS    │
│ S002 │ Bob   │  2   │ CS   │ CS401 │ Machine Learning │ 4       │ CS    │
│ S003 │ Carol │  4   │ EE   │ EE201 │ Circuit Analysis │ 3       │ EE    │
│ S005 │ Eve   │  1   │ CS   │ CS101 │ Intro to CS      │ 3       │ CS    │
│ S005 │ Eve   │  1   │ CS   │ CS301 │ Database Theory  │ 3       │ CS    │
│ S005 │ Eve   │  1   │ CS   │ CS401 │ Machine Learning │ 4       │ CS    │
└──────┴───────┴──────┴──────┴───────┴──────────────────┴─────────┴───────┘
(Dave/ME has no matching course, so not in result)
```

### Equi-Join

An **equi-join** is a theta join where the condition uses only equality (=):

```
R ⋈_{R.A = S.B} S

All equi-joins are theta joins, but not all theta joins are equi-joins.
A theta join with R.A > S.B is NOT an equi-join.
```

### Natural Join

A **natural join** is a special equi-join that:
1. Joins on ALL common attribute names
2. Removes duplicate columns from the result

```
Notation:   R ⋈ S   (no subscript)

Definition: Join on all attributes with the same name in R and S,
            then project out the duplicate attribute columns.
```

**Example:**

```
STUDENT ⋈ ENROLLMENT

Common attribute: sid

Step 1: Equi-join on STUDENT.sid = ENROLLMENT.sid
Step 2: Remove duplicate sid column

Result:
┌──────┬───────┬──────┬──────┬───────┬───────┐
│ sid  │ name  │ year │ dept │ cid   │ grade │
├──────┼───────┼──────┼──────┼───────┼───────┤
│ S001 │ Alice │  3   │ CS   │ CS101 │ A     │
│ S001 │ Alice │  3   │ CS   │ CS301 │ A+    │
│ S001 │ Alice │  3   │ CS   │ MA101 │ B+    │
│ S002 │ Bob   │  2   │ CS   │ CS101 │ B     │
│ S002 │ Bob   │  2   │ CS   │ CS301 │ A-    │
│ S003 │ Carol │  4   │ EE   │ EE201 │ A     │
│ S003 │ Carol │  4   │ EE   │ CS101 │ B+    │
│ S004 │ Dave  │  3   │ ME   │ CS101 │ C     │
│ S005 │ Eve   │  1   │ CS   │ CS101 │ A     │
│ S005 │ Eve   │  1   │ CS   │ MA101 │ A-    │
└──────┴───────┴──────┴──────┴───────┴───────┘

WARNING: Be careful with natural join when relations share
attribute names unintentionally!

STUDENT(sid, name, year, dept) ⋈ COURSE(cid, title, credits, dept)
                          ^^^^                                ^^^^
This joins on 'dept', which may not be what you want!
It matches students with courses in THEIR department.
```

### Outer Joins

Standard joins discard tuples that do not match. **Outer joins** preserve unmatched tuples by padding them with NULLs.

```
Three types:

1. LEFT OUTER JOIN (⟕):
   Keep all tuples from the LEFT relation.
   Pad with NULLs if no match on the right.

2. RIGHT OUTER JOIN (⟖):
   Keep all tuples from the RIGHT relation.
   Pad with NULLs if no match on the left.

3. FULL OUTER JOIN (⟗):
   Keep all tuples from BOTH relations.
   Pad with NULLs on either side as needed.
```

**Example: Left Outer Join**

```
STUDENT ⟕_{STUDENT.sid = ENROLLMENT.sid} ENROLLMENT

"Find all students and their enrollments, INCLUDING students
 with no enrollments"

(Suppose S004/Dave had no enrollments in our data.)

If Dave had NO enrollments:

Result:
┌──────┬───────┬──────┬──────┬───────┬───────┐
│ sid  │ name  │ year │ dept │ cid   │ grade │
├──────┼───────┼──────┼──────┼───────┼───────┤
│ S001 │ Alice │  3   │ CS   │ CS101 │ A     │
│ S001 │ Alice │  3   │ CS   │ CS301 │ A+    │
│ S001 │ Alice │  3   │ CS   │ MA101 │ B+    │
│ S002 │ Bob   │  2   │ CS   │ CS101 │ B     │
│ S002 │ Bob   │  2   │ CS   │ CS301 │ A-    │
│ S003 │ Carol │  4   │ EE   │ EE201 │ A     │
│ S003 │ Carol │  4   │ EE   │ CS101 │ B+    │
│ S004 │ Dave  │  3   │ ME   │ NULL  │ NULL  │  ← preserved with NULLs
│ S005 │ Eve   │  1   │ CS   │ CS101 │ A     │
│ S005 │ Eve   │  1   │ CS   │ MA101 │ A-    │
└──────┴───────┴──────┴──────┴───────┴───────┘
```

### Join Comparison Summary

```
┌──────────────────────────────────────────────────────────────────┐
│                        Join Types                                │
│                                                                  │
│  Cartesian Product  R × S      All combinations (n × m tuples)  │
│  Theta Join         R ⋈_θ S    Cartesian + selection on θ       │
│  Equi-Join          R ⋈_{=} S  Theta join with equality only    │
│  Natural Join       R ⋈ S      Equi-join on common attrs,       │
│                                 remove duplicates                │
│  Left Outer Join    R ⟕ S      Natural + keep unmatched R       │
│  Right Outer Join   R ⟖ S      Natural + keep unmatched S       │
│  Full Outer Join    R ⟗ S      Natural + keep all unmatched     │
│  Semi-Join          R ⋉ S      Tuples in R with a match in S   │
│  Anti-Join          R ▷ S      Tuples in R with NO match in S  │
└──────────────────────────────────────────────────────────────────┘
```

### Semi-Join and Anti-Join

```
Semi-Join:  R ⋉ S = π_{attrs(R)}(R ⋈ S)

"Return tuples from R that have a matching tuple in S"
(Only R's attributes appear in the result)

Anti-Join:  R ▷ S = R − π_{attrs(R)}(R ⋈ S)

"Return tuples from R that have NO matching tuple in S"
```

**Example:**

```
"Students who are enrolled in at least one course" (Semi-Join):
STUDENT ⋉ ENROLLMENT
→ All students from STUDENT who appear in ENROLLMENT

"Students who are NOT enrolled in any course" (Anti-Join):
STUDENT ▷ ENROLLMENT
→ Students who have no matching enrollment
```

---

## 5. Division

The **division** operation answers "for all" queries. It is one of the most powerful and least intuitive operations in relational algebra.

```
Notation:   R ÷ S

Given:      R(A1, A2, ..., An, B1, B2, ..., Bm)
            S(B1, B2, ..., Bm)

Result:     Tuples t over {A1, ..., An} such that for EVERY tuple s in S,
            the tuple <t, s> is in R.

Formally:
  R ÷ S = { t | t ∈ π_{A}(R) AND ∀s ∈ S : <t,s> ∈ R }

  where A = {A1, ..., An} (attributes of R not in S)
```

### Division Expressed Using Fundamental Operations

```
R ÷ S = π_A(R) − π_A( (π_A(R) × S) − R )

Explanation:
  1. π_A(R)              = all possible A-values in R
  2. π_A(R) × S          = every A-value paired with every S-tuple
  3. (π_A(R) × S) − R    = combinations that are MISSING from R
  4. π_A(...)             = A-values that are missing at least one S-tuple
  5. π_A(R) − ...         = A-values that are NOT missing any S-tuple
                          = A-values associated with ALL S-tuples
```

### Division Example

```
"Find students enrolled in ALL CS courses"

Step 1: Define the dividend (student-course pairs)
  R = π_{sid, cid}(ENROLLMENT)
  ┌──────┬───────┐
  │ sid  │ cid   │
  ├──────┼───────┤
  │ S001 │ CS101 │
  │ S001 │ CS301 │
  │ S001 │ MA101 │
  │ S002 │ CS101 │
  │ S002 │ CS301 │
  │ S003 │ EE201 │
  │ S003 │ CS101 │
  │ S004 │ CS101 │
  │ S005 │ CS101 │
  │ S005 │ MA101 │
  └──────┴───────┘

Step 2: Define the divisor (all CS courses)
  S = π_{cid}(σ_{dept='CS'}(COURSE))
  ┌───────┐
  │ cid   │
  ├───────┤
  │ CS101 │
  │ CS301 │
  │ CS401 │
  └───────┘

Step 3: R ÷ S = students associated with ALL of {CS101, CS301, CS401}

  Check each student:
    S001: has {CS101, CS301} but NOT CS401 → ✗
    S002: has {CS101, CS301} but NOT CS401 → ✗
    S003: has {CS101} only                 → ✗
    S004: has {CS101} only                 → ✗
    S005: has {CS101} only                 → ✗

  Result: EMPTY SET (no student is enrolled in ALL three CS courses)

If CS401 were not in the course list (only CS101 and CS301):
  S = {CS101, CS301}
  S001: has {CS101, CS301} → ✓
  S002: has {CS101, CS301} → ✓
  Others: ✗

  Result:
  ┌──────┐
  │ sid  │
  ├──────┤
  │ S001 │
  │ S002 │
  └──────┘
```

---

## 6. Additional Operations

### Aggregation and Grouping

The **grouping/aggregation** operator extends relational algebra to support aggregate functions.

```
Notation:   _{G1, G2, ..., Gn} G _{F1(A1), F2(A2), ..., Fk(Ak)} (R)

            or more commonly written as:

            γ_{G; F1(A1) AS name1, ...}(R)

Where:
  G1, ..., Gn = grouping attributes
  F1, ..., Fk = aggregate functions (COUNT, SUM, AVG, MIN, MAX)
  A1, ..., Ak = attributes to aggregate
```

**Example:**

```
"Count students per department"

γ_{dept; COUNT(sid) AS count}(STUDENT)

Result:
┌──────┬───────┐
│ dept │ count │
├──────┼───────┤
│ CS   │ 3     │
│ EE   │ 1     │
│ ME   │ 1     │
└──────┴───────┘


"Average salary per department (for instructors)"

γ_{dept; AVG(salary) AS avg_sal, COUNT(*) AS num}(INSTRUCTOR)

Result:
┌──────┬─────────┬─────┐
│ dept │ avg_sal │ num │
├──────┼─────────┼─────┤
│ CS   │ 91500   │ 2   │
│ EE   │ 92000   │ 1   │
│ MA   │ 85000   │ 1   │
└──────┴─────────┴─────┘
```

### Assignment

The **assignment** operator stores intermediate results:

```
Notation:   temp ← expression

Example:
  CS_STUDENTS ← σ_{dept='CS'}(STUDENT)
  CS_NAMES ← π_{name}(CS_STUDENTS)
```

### Sorting

```
Notation:   τ_{A1 ASC, A2 DESC}(R)

Note: Sorting produces a LIST, not a SET. Strictly speaking, it
      goes beyond pure relational algebra (which produces only sets).
```

---

## 7. Query Trees and Algebraic Optimization

### Query Trees

A **query tree** (or operator tree) represents a relational algebra expression as a tree where:
- Leaf nodes are base relations
- Internal nodes are relational algebra operations
- The root produces the final result

**Example:** "Find names of CS students enrolled in CS301"

```
Relational algebra:
  π_{name}(σ_{dept='CS' AND cid='CS301'}(STUDENT ⋈ ENROLLMENT))

Query tree:

              π_{name}
                 │
          σ_{dept='CS' AND cid='CS301'}
                 │
                ⋈_{sid=sid}
               / \
          STUDENT  ENROLLMENT
```

### Algebraic Optimization

The query optimizer transforms query trees using **equivalence rules** to find more efficient execution plans.

### Key Equivalence Rules

**Rule 1: Cascade of Selection**

```
σ_{c1 AND c2}(R) = σ_{c1}(σ_{c2}(R))

A conjunctive selection can be broken into a sequence of selections.
```

**Rule 2: Commutativity of Selection**

```
σ_{c1}(σ_{c2}(R)) = σ_{c2}(σ_{c1}(R))

The order of selections does not matter.
```

**Rule 3: Cascade of Projection**

```
π_{L1}(π_{L2}(...π_{Ln}(R)...)) = π_{L1}(R)

Only the outermost projection matters (if L1 ⊆ L2 ⊆ ... ⊆ Ln).
```

**Rule 4: Commuting Selection with Projection**

```
If the selection condition c only involves attributes in L:
  π_L(σ_c(R)) = σ_c(π_L(R))
```

**Rule 5: Commutativity of Join**

```
R ⋈ S = S ⋈ R
R × S = S × R
```

**Rule 6: Associativity of Join**

```
(R ⋈ S) ⋈ T = R ⋈ (S ⋈ T)
```

**Rule 7: Pushing Selection Through Join**

```
If condition c involves only attributes of R:
  σ_c(R ⋈ S) = σ_c(R) ⋈ S

This is the MOST IMPORTANT optimization rule!
It reduces the size of intermediate results.
```

**Rule 8: Commutativity of Set Operations**

```
R ∪ S = S ∪ R
R ∩ S = S ∩ R
(But R − S ≠ S − R)
```

### Optimization Example

```
Original query tree (unoptimized):

              π_{name}
                 │
          σ_{dept='CS' AND cid='CS301'}
                 │
                ⋈_{sid=sid}
               / \
          STUDENT  ENROLLMENT
          (5 rows)  (10 rows)
          Cartesian: 50 rows before filter

Optimized query tree (push selections down):

              π_{name}
                 │
                ⋈_{sid=sid}
               / \
   σ_{dept='CS'}  σ_{cid='CS301'}
        |              |
     STUDENT       ENROLLMENT
     (→3 rows)     (→2 rows)
     Join: 6 combinations, ~2 matches

The optimized tree:
  1. Filters STUDENT to 3 CS students FIRST
  2. Filters ENROLLMENT to 2 CS301 enrollments FIRST
  3. Joins only 3 × 2 = 6 combinations instead of 5 × 10 = 50
  4. Significant reduction in intermediate result size
```

### Heuristic Optimization Rules

```
1. Push selections down as far as possible
   (Reduce tuple count early)

2. Push projections down as far as possible
   (Reduce attribute count early, but keep join attributes)

3. Choose the most restrictive selection first
   (A selection that eliminates the most tuples)

4. Avoid Cartesian products
   (Always prefer joins over Cartesian + selection)

5. Choose join order to minimize intermediate result size
   (This is the hardest optimization problem — often NP-hard)
```

---

## 8. Relational Calculus (Brief Introduction)

While relational algebra is **procedural** (specifies how to compute the result), **relational calculus** is **declarative** (specifies what the result should be, not how to compute it).

### Tuple Relational Calculus (TRC)

In TRC, queries are expressed using tuple variables that range over relations.

```
General form:
  { t | P(t) }

  "The set of all tuples t such that predicate P(t) is true"

Example: "Find all CS students"

  { t | t ∈ STUDENT ∧ t.dept = 'CS' }

  "The set of tuples t from STUDENT where dept is CS"

Example: "Find names and departments of students in year 3 or 4"

  { t.name, t.dept | t ∈ STUDENT ∧ (t.year = 3 ∨ t.year = 4) }

Example: "Find students enrolled in CS301"

  { t | t ∈ STUDENT ∧ ∃e ∈ ENROLLMENT(e.sid = t.sid ∧ e.cid = 'CS301') }

  "Tuples t from STUDENT such that there EXISTS an enrollment e
   with the same sid and cid = CS301"
```

### Domain Relational Calculus (DRC)

In DRC, variables range over individual values (domains) rather than tuples.

```
General form:
  { <x1, x2, ..., xn> | P(x1, x2, ..., xn) }

Example: "Find names of CS students"

  { <n> | ∃s, y, d (STUDENT(s, n, y, d) ∧ d = 'CS') }

  "The set of name values n such that there exist values s, y, d
   where (s, n, y, d) is a tuple in STUDENT and d is CS"
```

### Equivalence of Algebra and Calculus

**Codd's Theorem** (1972): Relational algebra and (safe) relational calculus have the same expressive power. Any query expressible in one can be expressed in the other.

```
┌──────────────────────────────────────────────────────┐
│                                                      │
│  Relational Algebra  ≡  Safe Tuple Relational Calc.  │
│                      ≡  Safe Domain Relational Calc.  │
│                      ≡  SQL (core)                   │
│                                                      │
│  A query language is "relationally complete" if it   │
│  can express everything that relational algebra can. │
│                                                      │
│  SQL is relationally complete (and more — it has     │
│  aggregation, ordering, recursion, etc.)             │
└──────────────────────────────────────────────────────┘
```

### Safety of Expressions

A relational calculus expression is **safe** if it guarantees a finite result. Unsafe expressions can produce infinite results:

```
Unsafe:  { t | ¬(t ∈ STUDENT) }
         "All tuples NOT in STUDENT" — this is infinite!

Safe:    { t | t ∈ STUDENT ∧ ¬(∃e ∈ ENROLLMENT(e.sid = t.sid)) }
         "Students NOT enrolled in any course" — finite result
```

---

## 9. Equivalence with SQL

Every relational algebra expression has an SQL equivalent. Understanding this correspondence helps in writing and optimizing SQL queries.

### Operation-by-Operation Mapping

```
┌────────────────────┬──────────────────────────────────────────┐
│ Relational Algebra │ SQL Equivalent                           │
├────────────────────┼──────────────────────────────────────────┤
│ σ_{c}(R)           │ SELECT * FROM R WHERE c                  │
│ π_{A,B}(R)         │ SELECT DISTINCT A, B FROM R              │
│ ρ_{S}(R)           │ R AS S  (in FROM clause)                 │
│ R ∪ S              │ SELECT * FROM R UNION SELECT * FROM S    │
│ R ∩ S              │ SELECT * FROM R INTERSECT SELECT * FROM S│
│ R − S              │ SELECT * FROM R EXCEPT SELECT * FROM S   │
│ R × S              │ SELECT * FROM R, S  (or CROSS JOIN)      │
│ R ⋈_{c} S          │ SELECT * FROM R JOIN S ON c              │
│ R ⋈ S              │ SELECT * FROM R NATURAL JOIN S           │
│ R ⟕ S              │ SELECT * FROM R LEFT OUTER JOIN S ON ... │
│ R ÷ S              │ (requires NOT EXISTS + correlated sub)   │
│ γ_{G;F(A)}(R)      │ SELECT G, F(A) FROM R GROUP BY G        │
│ τ_{A}(R)           │ SELECT * FROM R ORDER BY A               │
└────────────────────┴──────────────────────────────────────────┘
```

### Detailed SQL Equivalences

**Selection:**

```
σ_{dept='CS' AND year>=3}(STUDENT)

SELECT *
FROM student
WHERE dept = 'CS' AND year >= 3;
```

**Projection:**

```
π_{name, dept}(STUDENT)

SELECT DISTINCT name, dept
FROM student;

Note: SQL does NOT eliminate duplicates by default.
      You must use DISTINCT to match relational algebra's set semantics.
```

**Natural Join:**

```
STUDENT ⋈ ENROLLMENT

-- Method 1: NATURAL JOIN (matches on all common columns)
SELECT * FROM student NATURAL JOIN enrollment;

-- Method 2: Explicit JOIN (safer — you control which columns match)
SELECT s.sid, s.name, s.year, s.dept, e.cid, e.grade
FROM student s
JOIN enrollment e ON s.sid = e.sid;
```

**Division (the hardest to express in SQL):**

```
R ÷ S  =  "Find tuples in R associated with ALL tuples in S"

-- "Find students enrolled in ALL CS courses"
SELECT DISTINCT e.sid
FROM enrollment e
WHERE NOT EXISTS (
    SELECT c.cid
    FROM course c
    WHERE c.dept = 'CS'
    AND NOT EXISTS (
        SELECT 1
        FROM enrollment e2
        WHERE e2.sid = e.sid
        AND e2.cid = c.cid
    )
);

-- Alternative using COUNT:
SELECT e.sid
FROM enrollment e
JOIN course c ON e.cid = c.cid
WHERE c.dept = 'CS'
GROUP BY e.sid
HAVING COUNT(DISTINCT e.cid) = (
    SELECT COUNT(*) FROM course WHERE dept = 'CS'
);
```

**Composition Example:**

```
"Find names of students who received an A in any course"

Relational Algebra:
  π_{name}(σ_{grade='A'}(STUDENT ⋈ ENROLLMENT))

SQL:
  SELECT DISTINCT s.name
  FROM student s
  JOIN enrollment e ON s.sid = e.sid
  WHERE e.grade = 'A';
```

---

## 10. Complete Worked Examples

### Example 1: Multi-Step Query

**Query:** "Find the names and departments of students who are enrolled in a course taught by the CS department but are not CS majors themselves."

```
Relational Algebra:

  CS_COURSES ← σ_{dept='CS'}(COURSE)
  CS_ENROLLED ← ENROLLMENT ⋈_{cid} CS_COURSES
  NON_CS_STUDENTS ← σ_{dept≠'CS'}(STUDENT)
  RESULT ← π_{name, dept}(NON_CS_STUDENTS ⋈_{sid} CS_ENROLLED)

Step-by-step:

1. CS_COURSES = σ_{dept='CS'}(COURSE)
   → {CS101, CS301, CS401}

2. CS_ENROLLED = π_{sid}(ENROLLMENT ⋈ CS_COURSES)
   → {S001, S002, S003, S004, S005} (all who took CS101/CS301/CS401)

3. NON_CS_STUDENTS = σ_{dept≠'CS'}(STUDENT)
   → {S003/Carol/EE, S004/Dave/ME}

4. RESULT = π_{name, dept}(NON_CS_STUDENTS ⋈ CS_ENROLLED)
   → {(Carol, EE), (Dave, ME)}

SQL:
  SELECT DISTINCT s.name, s.dept
  FROM student s
  JOIN enrollment e ON s.sid = e.sid
  JOIN course c ON e.cid = c.cid
  WHERE c.dept = 'CS' AND s.dept <> 'CS';
```

### Example 2: Division Query

**Query:** "Find students who have taken ALL courses that student S001 has taken."

```
Relational Algebra:

  S001_COURSES ← π_{cid}(σ_{sid='S001'}(ENROLLMENT))
  ALL_ENROLLMENTS ← π_{sid, cid}(ENROLLMENT)
  RESULT ← ALL_ENROLLMENTS ÷ S001_COURSES

Step-by-step:

1. S001_COURSES = {CS101, CS301, MA101}

2. Check each student:
   S001: {CS101, CS301, MA101} ⊇ {CS101, CS301, MA101} → ✓
   S002: {CS101, CS301}        ⊇ {CS101, CS301, MA101} → ✗ (missing MA101)
   S003: {EE201, CS101}        ⊇ {CS101, CS301, MA101} → ✗
   S004: {CS101}               ⊇ {CS101, CS301, MA101} → ✗
   S005: {CS101, MA101}        ⊇ {CS101, CS301, MA101} → ✗ (missing CS301)

3. RESULT = {S001}

SQL:
  SELECT e.sid
  FROM enrollment e
  WHERE e.cid IN (
      SELECT cid FROM enrollment WHERE sid = 'S001'
  )
  GROUP BY e.sid
  HAVING COUNT(DISTINCT e.cid) = (
      SELECT COUNT(DISTINCT cid) FROM enrollment WHERE sid = 'S001'
  );
```

### Example 3: Outer Join

**Query:** "List all students with their enrollment counts, including students with no enrollments."

```
Relational Algebra:

  JOINED ← STUDENT ⟕ ENROLLMENT       (left outer join on sid)
  RESULT ← γ_{sid, name; COUNT(cid) AS num_courses}(JOINED)

If Dave (S004) had no enrollments:

SQL:
  SELECT s.sid, s.name, COUNT(e.cid) AS num_courses
  FROM student s
  LEFT OUTER JOIN enrollment e ON s.sid = e.sid
  GROUP BY s.sid, s.name
  ORDER BY num_courses DESC;

Result:
  ┌──────┬───────┬─────────────┐
  │ sid  │ name  │ num_courses │
  ├──────┼───────┼─────────────┤
  │ S001 │ Alice │ 3           │
  │ S002 │ Bob   │ 2           │
  │ S003 │ Carol │ 2           │
  │ S005 │ Eve   │ 2           │
  │ S004 │ Dave  │ 0           │  ← preserved by LEFT OUTER JOIN
  └──────┴───────┴─────────────┘
```

---

## 11. Exercises

### Basic Operations

**Exercise 3.1**: Using the sample database, write relational algebra expressions for:

- (a) All students in year 2 or year 3
- (b) Course titles with 4 or more credits
- (c) Student IDs enrolled in EE201
- (d) Names of students NOT in the CS department

**Exercise 3.2**: Evaluate the following expressions step by step, showing intermediate results:

- (a) `π_{name}(σ_{year > 2}(STUDENT))`
- (b) `π_{sid}(σ_{grade='A'}(ENROLLMENT)) ∩ π_{sid}(σ_{dept='CS'}(STUDENT))`
- (c) `STUDENT ⋈ (σ_{cid='CS301'}(ENROLLMENT))`

### Join Operations

**Exercise 3.3**: For each of the following, write the relational algebra expression AND the equivalent SQL:

- (a) Find names of students enrolled in "Database Theory"
- (b) Find course titles that have at least one student with grade A
- (c) Find students who are enrolled in courses outside their department
- (d) Find pairs of students who share at least one course

**Exercise 3.4**: Explain the difference between the results of these three queries on our sample data:

```
Q1: STUDENT ⋈ ENROLLMENT
Q2: STUDENT ⟕ ENROLLMENT
Q3: STUDENT × ENROLLMENT
```

How many tuples does each produce? Which tuples appear in Q2 but not Q1?

### Division

**Exercise 3.5**: Write relational algebra expressions using division for:

- (a) Students enrolled in every 3-credit course
- (b) Students who have taken ALL courses that Bob (S002) has taken

Show the step-by-step evaluation.

**Exercise 3.6**: Express each division query from Exercise 3.5 in SQL using:
- (i) Double NOT EXISTS
- (ii) GROUP BY with HAVING COUNT

### Optimization

**Exercise 3.7**: Given the query:

```
π_{name}(σ_{credits=3 AND grade='A'}(STUDENT ⋈ ENROLLMENT ⋈ COURSE))
```

- (a) Draw the initial (unoptimized) query tree
- (b) Apply algebraic optimization rules to produce an optimized query tree
- (c) Explain which rules you applied and why the optimized tree is better
- (d) Estimate the reduction in intermediate result sizes

**Exercise 3.8**: Prove or disprove the following equivalences:

- (a) `σ_{c1}(R ∪ S) = σ_{c1}(R) ∪ σ_{c1}(S)`
- (b) `σ_{c1}(R − S) = σ_{c1}(R) − σ_{c1}(S)`
- (c) `π_A(R ∪ S) = π_A(R) ∪ π_A(S)`
- (d) `σ_{c1}(R × S) = σ_{c1}(R) × S`  (where c1 involves only R's attributes)

### Relational Calculus

**Exercise 3.9**: Express the following in Tuple Relational Calculus (TRC):

- (a) Names of students in the CS department
- (b) Students enrolled in at least two courses
- (c) Courses with no enrolled students
- (d) Students enrolled in ALL courses offered by the CS department

**Exercise 3.10**: Determine which of the following TRC expressions are safe. If unsafe, explain why and provide a safe equivalent:

- (a) `{ t | ¬(t ∈ STUDENT) }`
- (b) `{ t.name | t ∈ STUDENT ∧ t.gpa > 3.5 }`
- (c) `{ <x, y> | ∃t ∈ STUDENT(t.sid = x ∧ t.name = y) }`
- (d) `{ t | t.salary > 100000 }`

### Challenge Problems

**Exercise 3.11**: A relation R(A, B, C) contains the following tuples:

```
{(a1, b1, c1), (a1, b2, c1), (a1, b2, c2), (a2, b1, c1), (a2, b1, c2)}
```

A relation S(B, C) contains: `{(b1, c1), (b1, c2)}`

Compute R ÷ S step by step. Verify your answer by checking that each resulting tuple is associated with ALL tuples in S.

**Exercise 3.12**: Write a single relational algebra expression (no assignment) for:

"Find the department with the highest average instructor salary."

Hint: This requires aggregation and a comparison pattern.

---

**Previous**: [Relational Model](./02_Relational_Model.md) | **Next**: [ER Modeling](./04_ER_Modeling.md)
