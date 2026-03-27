# Lesson 05: Functional Dependencies

**Previous**: [04_ER_Modeling.md](./04_ER_Modeling.md) | **Next**: [06_Normalization.md](./06_Normalization.md)

---

> **Topic**: Database Theory
> **Lesson**: 5 of 16
> **Prerequisites**: Relational model concepts (relations, tuples, attributes, keys), basic set theory
> **Objective**: Understand functional dependencies as the formal foundation for database normalization, master Armstrong's axioms, compute attribute closures, derive candidate keys, and compute minimal covers

## Learning Objectives

After completing this lesson, you will be able to:

1. Define functional dependencies (FDs) and explain their role as the formal foundation for relational schema design and normalization.
2. Apply Armstrong's axioms (reflexivity, augmentation, transitivity) and derived rules to infer new functional dependencies from a given set.
3. Compute the attribute closure of a set of attributes under a given set of FDs to determine key membership.
4. Identify all candidate keys of a relation by systematically analyzing functional dependencies.
5. Compute a minimal cover (canonical cover) for a set of functional dependencies by eliminating redundancies.
6. Explain how anomalies (insertion, deletion, update) arise from poor schema design and how FDs formalize the criteria for avoiding them.

---

## 1. Introduction

Functional dependencies (FDs) are the most important concept in relational database design theory. They formalize the notion of "one attribute uniquely determines another" and provide the mathematical foundation for normalization — the process of organizing a database to reduce redundancy and prevent anomalies.

Before we had functional dependencies, database designers relied on intuition and experience. FDs gave us a rigorous framework to reason about what constitutes a "good" schema, and algorithms to systematically achieve it.

### 1.1 Why Functional Dependencies Matter

Consider a university database with a single relation:

```
StudentCourse(student_id, student_name, dept_name, course_id, course_title, grade)
```

Intuitively, we know:
- A student_id uniquely identifies a student_name and dept_name
- A course_id uniquely identifies a course_title
- The combination (student_id, course_id) uniquely identifies the grade

These intuitive observations are precisely functional dependencies. They allow us to:

1. **Identify redundancy**: If student_name appears in every row for the same student_id, we have redundancy
2. **Detect anomalies**: Updating a student's department in one row but not others creates inconsistency
3. **Guide decomposition**: FDs tell us exactly how to split a relation into smaller, well-structured pieces
4. **Verify key constraints**: FDs formally define what it means for an attribute set to be a key

---

## 2. Formal Definition

### 2.1 Functional Dependency

Let R be a relation schema with attribute set U, and let X, Y ⊆ U.

> **Definition**: A **functional dependency** X → Y holds on R if and only if for every legal instance r of R, whenever two tuples t₁ and t₂ agree on the attributes in X, they must also agree on the attributes in Y. Formally:
>
> X → Y ⟺ ∀ t₁, t₂ ∈ r : t₁[X] = t₂[X] ⟹ t₁[Y] = t₂[Y]

Here:
- **X** is called the **determinant** (or left-hand side, LHS)
- **Y** is called the **dependent** (or right-hand side, RHS)
- We read X → Y as "X functionally determines Y" or "Y is functionally dependent on X"

### 2.2 Trivial and Non-Trivial FDs

> **Definition**: A functional dependency X → Y is **trivial** if Y ⊆ X.

Examples:
- {student_id, course_id} → {student_id} is trivial
- {student_id} → {student_id} is trivial
- {student_id} → {student_name} is **non-trivial** (student_name ∉ {student_id})

> **Definition**: A functional dependency X → Y is **completely non-trivial** if X ∩ Y = ∅.

### 2.3 Examples

Consider the relation schema:

```
Employee(emp_id, emp_name, dept_id, dept_name, salary, manager_id)
```

Typical functional dependencies:
- emp_id → emp_name, dept_id, salary, manager_id (employee ID determines all employee attributes)
- dept_id → dept_name (department ID determines department name)
- emp_id → dept_name (transitively, through dept_id)

**Important**: FDs are **semantic** constraints. They are determined by the meaning of the data in the real world, not by examining a particular instance. Looking at a snapshot of data can only **disprove** an FD (by finding a counterexample), never **prove** one holds in general.

### 2.4 FDs vs. Keys

There is a deep connection between functional dependencies and keys:

> **Definition**: A set of attributes K is a **superkey** of relation R if K → U, where U is the set of all attributes of R.

> **Definition**: A set of attributes K is a **candidate key** of R if:
> 1. K → U (K is a superkey), and
> 2. No proper subset of K functionally determines U (K is minimal)

Thus, keys are simply functional dependencies whose right-hand side is the entire attribute set, with a minimality condition.

---

## 3. Armstrong's Axioms

In 1974, William W. Armstrong proposed a set of inference rules for functional dependencies. These axioms are **sound** (they only derive correct FDs) and **complete** (they can derive all FDs that logically follow from a given set).

### 3.1 The Three Axioms

Let F be a set of functional dependencies on relation schema R, and let X, Y, Z ⊆ attributes(R).

#### Axiom 1: Reflexivity

> If Y ⊆ X, then X → Y.

This axiom generates all trivial FDs. For example:
- {A, B, C} → {A, B}
- {A, B, C} → {A}
- {A} → {A}

**Proof of soundness**: If Y ⊆ X and t₁[X] = t₂[X], then since Y ⊆ X, we have t₁[Y] = t₂[Y]. ∎

#### Axiom 2: Augmentation

> If X → Y, then XZ → YZ for any Z.

We can "augment" both sides of an FD with any set of attributes. For example:
- If A → B, then AC → BC
- If AB → C, then ABD → CD

**Proof of soundness**: Suppose X → Y and t₁[XZ] = t₂[XZ]. Then t₁[X] = t₂[X] and t₁[Z] = t₂[Z]. Since X → Y, we have t₁[Y] = t₂[Y]. Combining: t₁[YZ] = t₂[YZ]. ∎

#### Axiom 3: Transitivity

> If X → Y and Y → Z, then X → Z.

This allows chaining of functional dependencies. For example:
- If student_id → dept_id and dept_id → dept_name, then student_id → dept_name

**Proof of soundness**: Suppose X → Y and Y → Z. Let t₁[X] = t₂[X]. By X → Y, t₁[Y] = t₂[Y]. By Y → Z, t₁[Z] = t₂[Z]. Therefore X → Z. ∎

### 3.2 Soundness and Completeness

> **Theorem (Armstrong, 1974)**: Armstrong's axioms are **sound and complete**.
> - **Soundness**: Every FD derivable from F using these axioms is logically implied by F.
> - **Completeness**: Every FD logically implied by F can be derived using these axioms.

The completeness proof is non-trivial. The key idea is to show that if X → Y cannot be derived from F using Armstrong's axioms, then there exists a two-tuple relation that satisfies F but violates X → Y.

---

## 4. Derived Inference Rules

From Armstrong's three axioms, we can derive several useful additional rules.

### 4.1 Union Rule

> If X → Y and X → Z, then X → YZ.

**Proof**:
1. X → Y (given)
2. X → Z (given)
3. X → XY (augment step 1 with X; since XX = X, we get X → XY)
4. XY → YZ (augment step 2 with Y)
5. X → YZ (transitivity on steps 3 and 4) ∎

### 4.2 Decomposition Rule

> If X → YZ, then X → Y and X → Z.

**Proof**:
1. X → YZ (given)
2. YZ → Y (reflexivity, since Y ⊆ YZ)
3. X → Y (transitivity on steps 1 and 2)
4. YZ → Z (reflexivity, since Z ⊆ YZ)
5. X → Z (transitivity on steps 1 and 4) ∎

### 4.3 Pseudotransitivity

> If X → Y and WY → Z, then WX → Z.

**Proof**:
1. X → Y (given)
2. WX → WY (augment step 1 with W)
3. WY → Z (given)
4. WX → Z (transitivity on steps 2 and 3) ∎

### 4.4 Self-Determination

> X → X for any X.

This follows directly from reflexivity (X ⊆ X).

### 4.5 Accumulation

> If X → YZ and Z → BW, then X → YBW.

**Proof**:
1. X → YZ (given)
2. X → Z (decomposition of step 1)
3. Z → BW (given)
4. X → BW (transitivity on steps 2 and 3)
5. X → Y (decomposition of step 1)
6. X → YBW (union of steps 4 and 5) ∎

### 4.6 Summary of Rules

| Rule | Statement | Derived From |
|------|-----------|-------------|
| Reflexivity | Y ⊆ X ⟹ X → Y | Axiom |
| Augmentation | X → Y ⟹ XZ → YZ | Axiom |
| Transitivity | X → Y, Y → Z ⟹ X → Z | Axiom |
| Union | X → Y, X → Z ⟹ X → YZ | Aug + Trans |
| Decomposition | X → YZ ⟹ X → Y, X → Z | Refl + Trans |
| Pseudotransitivity | X → Y, WY → Z ⟹ WX → Z | Aug + Trans |

---

## 5. Closure of an Attribute Set

Computing the closure of an attribute set is the most fundamental algorithm in FD theory. It answers the question: "Given a set of FDs F, what attributes are functionally determined by a given set X?"

### 5.1 Definition

> **Definition**: The **closure of an attribute set X** under a set of functional dependencies F, denoted X⁺ (or X⁺_F), is the set of all attributes A such that X → A can be inferred from F using Armstrong's axioms.
>
> X⁺ = { A ∈ U | F ⊨ X → A }

### 5.2 Algorithm

The following algorithm computes X⁺ efficiently:

```
ALGORITHM: ComputeClosure(X, F)
INPUT:  X = a set of attributes
        F = a set of functional dependencies
OUTPUT: X⁺ = the closure of X under F

1.  result ← X
2.  REPEAT
3.      old_result ← result
4.      FOR EACH dependency (V → W) IN F DO
5.          IF V ⊆ result THEN
6.              result ← result ∪ W
7.          END IF
8.      END FOR
9.  UNTIL result = old_result
10. RETURN result
```

**Time complexity**: O(|F| × |U|) in the worst case, where |F| is the number of FDs and |U| is the number of attributes.

### 5.3 Worked Example 1

Let R(A, B, C, D, E, F) with the following FDs:

```
F = { A → BC,  CD → E,  B → D,  E → A }
```

**Compute {A}⁺:**

| Iteration | result | Applied FD | New Attributes |
|-----------|--------|-----------|----------------|
| Init | {A} | — | — |
| 1 | {A, B, C} | A → BC | B, C |
| 2 | {A, B, C, D} | B → D | D |
| 3 | {A, B, C, D, E} | CD → E | E |
| 4 | {A, B, C, D, E} | E → A (A already in result) | — |

Since no new attributes were added in iteration 4, we stop.

**{A}⁺ = {A, B, C, D, E}**

Note: F is not in the closure, so A is not a superkey of R (it doesn't determine F).

**Compute {A, F}⁺:**

Starting with {A, F}, after the same iterations plus F already present:

**{A, F}⁺ = {A, B, C, D, E, F}** = all attributes of R.

Therefore, {A, F} is a superkey.

### 5.4 Worked Example 2

Let R(A, B, C, D, E) with:

```
F = { AB → C,  C → D,  D → E,  E → A }
```

**Compute {B, C}⁺:**

| Iteration | result | Applied FD | New Attributes |
|-----------|--------|-----------|----------------|
| Init | {B, C} | — | — |
| 1 | {B, C, D} | C → D | D |
| 2 | {B, C, D, E} | D → E | E |
| 3 | {A, B, C, D, E} | E → A | A |
| 4 | {A, B, C, D, E} | AB → C (C already present) | — |

**{B, C}⁺ = {A, B, C, D, E}** — so {B, C} is a superkey.

### 5.5 Uses of Attribute Closure

The closure algorithm has three main applications:

1. **Testing if X → Y holds**: Compute X⁺. If Y ⊆ X⁺, then X → Y holds.

2. **Testing if X is a superkey**: Compute X⁺. If X⁺ = U (all attributes), then X is a superkey.

3. **Computing F⁺ (the closure of F)**: For each subset X ⊆ U, compute X⁺ and output X → Y for each Y ⊆ X⁺. (This is exponential and rarely practical, but it's theoretically important.)

---

## 6. Closure of a Set of FDs

### 6.1 Definition

> **Definition**: The **closure of a set of functional dependencies F**, denoted F⁺, is the set of all functional dependencies that can be logically inferred from F.
>
> F⁺ = { X → Y | F ⊨ X → Y }

F⁺ can be extremely large. For a relation with n attributes, there are 2ⁿ possible subsets for X and 2ⁿ for Y, giving up to 2²ⁿ possible FDs. In practice, we almost never compute F⁺ directly; we use the attribute closure algorithm instead.

### 6.2 Equivalence of FD Sets

> **Definition**: Two sets of FDs F and G are **equivalent** (denoted F ≡ G) if F⁺ = G⁺.

To test whether F ≡ G:
1. Check if every FD in G can be derived from F: for each X → Y in G, verify Y ⊆ X⁺_F
2. Check if every FD in F can be derived from G: for each X → Y in F, verify Y ⊆ X⁺_G

If both checks pass, F ≡ G.

### 6.3 Example

```
F = { A → B, B → C }
G = { A → B, B → C, A → C }
```

F ≡ G because A → C is derivable from F by transitivity, and all FDs in F are trivially in G.

---

## 7. Minimal Cover (Canonical Cover)

A minimal cover is a "simplified" version of a set of FDs — it removes redundancy while preserving the same logical content. This is essential for normalization algorithms.

### 7.1 Definition

> **Definition**: A set of FDs F_min is a **minimal cover** (or **canonical cover**) of F if:
> 1. **Equivalence**: F_min ≡ F (same closure)
> 2. **Single attribute on RHS**: Every FD in F_min has the form X → A where A is a single attribute
> 3. **No redundant FDs**: Removing any FD from F_min changes the closure
> 4. **No extraneous attributes on LHS**: For each FD X → A in F_min, no proper subset of X functionally determines A under F_min

### 7.2 Algorithm

```
ALGORITHM: MinimalCover(F)
INPUT:  F = a set of functional dependencies
OUTPUT: F_min = a minimal cover of F

Step 1: DECOMPOSE right-hand sides
    Replace each FD X → {A₁, A₂, ..., Aₙ} with
    X → A₁, X → A₂, ..., X → Aₙ

Step 2: REMOVE extraneous attributes from left-hand sides
    FOR EACH FD (X → A) IN F where |X| > 1 DO
        FOR EACH attribute B IN X DO
            IF A ∈ closure(X - {B}, F) THEN
                Replace (X → A) with ((X - {B}) → A)
            END IF
        END FOR
    END FOR

Step 3: REMOVE redundant FDs
    FOR EACH FD (X → A) IN F DO
        IF A ∈ closure(X, F - {X → A}) THEN
            Remove (X → A) from F
        END IF
    END FOR

RETURN F
```

**Important**: Step 2 must come before Step 3. If we remove redundant FDs first, some attributes that were extraneous might no longer appear to be extraneous.

### 7.3 Worked Example

Let R(A, B, C, D) with:

```
F = { A → BC,  B → C,  AB → D,  D → A }
```

**Step 1: Decompose RHS**

```
F = { A → B,  A → C,  B → C,  AB → D,  D → A }
```

**Step 2: Remove extraneous LHS attributes**

Check AB → D: Can we remove A or B?
- Check if B alone determines D: Compute {B}⁺ under current F:
  - {B} → {B, C} (via B → C) — no more. D ∉ {B}⁺. So B alone doesn't work; keep A.
- Check if A alone determines D: Compute {A}⁺ under current F:
  - {A} → {A, B, C} (via A → B, A → C) → {A, B, C, D} (via AB → D, since A and B are both present)
  - D ∈ {A}⁺. So A is extraneous in AB → D!

Replace AB → D with A → D:

```
F = { A → B,  A → C,  B → C,  A → D,  D → A }
```

**Step 3: Remove redundant FDs**

Check A → B: Remove it. Compute {A}⁺ under F - {A → B}:
- F - {A → B} = { A → C,  B → C,  A → D,  D → A }
- {A}⁺ = {A, C, D} (via A → C, A → D, D → A). B ∉ {A}⁺. So A → B is NOT redundant. Keep it.

Check A → C: Remove it. Compute {A}⁺ under F - {A → C}:
- {A}⁺: A → B gives B, B → C gives C. C ∈ {A}⁺. So A → C IS redundant. Remove it.

```
F = { A → B,  B → C,  A → D,  D → A }
```

Check B → C: Remove it. Compute {B}⁺ under F - {B → C}:
- {B}⁺ = {B}. C ∉ {B}⁺. Not redundant. Keep it.

Check A → D: Remove it. Compute {A}⁺ under F - {A → D}:
- {A}⁺ = {A, B, C}. D ∉ {A}⁺. Not redundant. Keep it.

Check D → A: Remove it. Compute {D}⁺ under F - {D → A}:
- {D}⁺ = {D}. A ∉ {D}⁺. Not redundant. Keep it.

**Minimal cover:**

```
F_min = { A → B,  B → C,  A → D,  D → A }
```

### 7.4 Non-Uniqueness of Minimal Covers

A minimal cover is **not unique**. Different orderings in Step 2 or Step 3 can produce different (but equivalent) minimal covers. For example, in Step 2, checking attributes in different orders can lead to different simplifications.

---

## 8. Finding Candidate Keys Using FDs

### 8.1 Attribute Classification

To find candidate keys efficiently, we first classify attributes:

| Category | Definition | Role in Keys |
|----------|-----------|-------------|
| **L-only** | Appears only on the LHS of FDs (never on RHS) | Must be in every key |
| **R-only** | Appears only on the RHS (never on LHS) | Never in any key |
| **Both** | Appears on both LHS and RHS | May or may not be in a key |
| **Neither** | Appears in no FD at all | Must be in every key |

### 8.2 Algorithm for Finding Candidate Keys

```
ALGORITHM: FindCandidateKeys(R, F)
INPUT:  R = relation schema, F = set of FDs
OUTPUT: Set of all candidate keys

Step 1: Classify attributes into L-only, R-only, Both, Neither.
Step 2: Let CORE = L-only ∪ Neither.
        (CORE must be part of every candidate key.)
Step 3: Compute CORE⁺.
        If CORE⁺ = all attributes, then CORE is the only candidate key. DONE.
Step 4: Otherwise, try adding subsets of "Both" attributes to CORE.
        Start with single attributes, then pairs, etc.
        For each subset S of "Both":
            Compute (CORE ∪ S)⁺
            If it equals all attributes and no proper subset of (CORE ∪ S)
            containing CORE is also a superkey, then CORE ∪ S is a candidate key.
```

### 8.3 Worked Example

Let R(A, B, C, D, E, F) with:

```
F = { AB → C,  C → D,  D → E,  CF → B }
```

**Step 1: Classify attributes**

| Attribute | LHS? | RHS? | Category |
|-----------|------|------|----------|
| A | Yes (AB→C) | No | L-only |
| B | Yes (AB→C) | Yes (CF→B) | Both |
| C | Yes (C→D, CF→B) | Yes (AB→C) | Both |
| D | Yes (D→E) | Yes (C→D) | Both |
| E | No | Yes (D→E) | R-only |
| F | Yes (CF→B) | No | L-only |

**Step 2: CORE = {A, F}** (L-only attributes; no "Neither" attributes)

**Step 3: Compute {A, F}⁺**

- Start: {A, F}
- No FD has LHS ⊆ {A, F} (AB needs B, C→D needs C, etc.)
- {A, F}⁺ = {A, F} ≠ all attributes

**Step 4: Try adding "Both" attributes (B, C, D)**

Try adding B: {A, B, F}⁺
- AB → C: {A, B, F, C}
- C → D: {A, B, C, D, F}
- D → E: {A, B, C, D, E, F} = all attributes!
- {A, B, F} is a superkey. Check minimality: CORE = {A, F} alone doesn't work. So {A, B, F} is a candidate key.

Try adding C: {A, C, F}⁺
- CF → B: {A, B, C, F}
- AB → C: already have C
- C → D: {A, B, C, D, F}
- D → E: {A, B, C, D, E, F} = all attributes!
- {A, C, F} is a superkey. Check minimality: {A, F} alone doesn't work. So {A, C, F} is a candidate key.

Try adding D: {A, D, F}⁺
- D → E: {A, D, E, F}
- No more applicable FDs. {A, D, E, F} ≠ all attributes. Not a superkey.

**Candidate keys: {A, B, F} and {A, C, F}**

We don't need to check pairs (like {B, C}) since we already found candidate keys with single additions and they are minimal.

---

## 9. Entailment and Implication

### 9.1 Logical Implication

> **Definition**: A set of FDs F **logically implies** an FD X → Y (written F ⊨ X → Y) if every relation instance that satisfies all FDs in F also satisfies X → Y.

### 9.2 Testing Implication

To test if F ⊨ X → Y:
1. Compute X⁺ under F
2. If Y ⊆ X⁺, then F ⊨ X → Y

This is the practical workhorse: instead of reasoning through chains of axiom applications, just run the closure algorithm.

### 9.3 Example

Given F = { A → B, B → C, CD → E }:

Does F ⊨ AD → E?

Compute {A, D}⁺:
- A → B: {A, B, D}
- B → C: {A, B, C, D}
- CD → E: {A, B, C, D, E}

Since E ∈ {A, D}⁺, yes, F ⊨ AD → E. ✓

Does F ⊨ A → E?

Compute {A}⁺:
- A → B: {A, B}
- B → C: {A, B, C}
- No more applicable FDs.

E ∉ {A}⁺ = {A, B, C}. So F ⊭ A → E. ✗

---

## 10. FDs in Practice

### 10.1 Identifying FDs from Requirements

Real-world FDs come from business rules and domain knowledge:

| Business Rule | Functional Dependency |
|--------------|----------------------|
| "Each employee has exactly one department" | emp_id → dept_id |
| "Each department has exactly one name" | dept_id → dept_name |
| "Each student gets one grade per course" | {student_id, course_id} → grade |
| "Each ISBN identifies one book title" | isbn → title |
| "A flight on a given date has one pilot" | {flight_num, date} → pilot_id |

### 10.2 FDs and NULL Values

Standard FD theory assumes no NULL values. In practice:
- NULLs complicate FD reasoning (NULL ≠ NULL in SQL)
- SQL's UNIQUE constraint allows multiple NULLs (except for PRIMARY KEY)
- Some database systems offer "NULLS NOT DISTINCT" to treat NULLs as equal for uniqueness checks

### 10.3 Discovering FDs from Data

While FDs are semantic constraints (determined by domain knowledge, not data), there are algorithms for **discovering approximate FDs** from data:

- **TANE** algorithm: Discovers all FDs holding in a dataset
- **FUN** algorithm: Uses lattice-based search
- **FDTool**: Practical tool for FD discovery

These are useful for reverse-engineering poorly documented databases, but the discovered FDs should always be validated against domain knowledge.

### 10.4 FDs in SQL

SQL doesn't have a direct `FUNCTIONAL DEPENDENCY` constraint, but FDs are enforced through:

```sql
-- Primary key enforces: emp_id → all other attributes
CREATE TABLE Employee (
    emp_id    INT PRIMARY KEY,
    emp_name  VARCHAR(100),
    dept_id   INT,
    salary    DECIMAL(10,2)
);

-- UNIQUE constraint enforces: email → (implicitly all attributes, if it's a key)
ALTER TABLE Employee ADD CONSTRAINT uq_email UNIQUE(email);

-- The FD dept_id → dept_name is enforced by having a separate Departments table
-- with dept_id as its primary key
CREATE TABLE Department (
    dept_id   INT PRIMARY KEY,
    dept_name VARCHAR(100)
);
```

---

## 11. Common Pitfalls

### 11.1 Confusing FDs with Data Patterns

A common mistake is looking at data and concluding an FD exists:

```
| city        | state |
|-------------|-------|
| Springfield | IL    |
| Portland    | OR    |
| Austin      | TX    |
```

This data is consistent with city → state, but in reality, many cities share names across states (Springfield exists in over 30 states). The FD city → state does **not** hold.

### 11.2 Direction Matters

X → Y does **not** imply Y → X.

- dept_id → dept_name (a department has one name) ✓
- dept_name → dept_id (a name identifies one department) — depends on whether names are unique!

### 11.3 FD on a Single Attribute

An FD like {} → A (the empty set determines A) means A has the same value in every tuple. This is a rare but valid FD (e.g., a table where all employees are in the same company: {} → company_name).

### 11.4 Order of Operations in Minimal Cover

Step 2 (remove extraneous LHS attributes) must precede Step 3 (remove redundant FDs). Reversing the order can produce incorrect results.

---

## 12. Proofs and Theory

### 12.1 Proving Completeness of Armstrong's Axioms (Sketch)

**Claim**: If F ⊭ X → Y using Armstrong's axioms, then there exists a relation instance satisfying F but violating X → Y.

**Proof sketch**: Construct a two-tuple relation r = {t₁, t₂} where:
- t₁[A] = t₂[A] = 1 for all A ∈ X⁺
- t₁[A] = 1, t₂[A] = 0 for all A ∉ X⁺

We need to verify:
1. r satisfies every FD in F: For any V → W in F, if t₁[V] = t₂[V], then V ⊆ X⁺, so W ⊆ X⁺ (by the closure algorithm), so t₁[W] = t₂[W]. ✓
2. r violates X → Y (assuming Y ⊄ X⁺): Since t₁[X] = t₂[X] (all 1's) but there exists some A ∈ Y with A ∉ X⁺, so t₁[A] ≠ t₂[A]. ✓

This contradicts the assumption that X → Y is logically implied by F, proving completeness. ∎

### 12.2 Complexity Results

| Problem | Complexity |
|---------|-----------|
| Computing X⁺ | O(\|F\| × \|U\|) — polynomial |
| Testing if F ⊨ X → Y | O(\|F\| × \|U\|) — polynomial |
| Computing F⁺ | Exponential (can be 2^(2n)) |
| Finding all candidate keys | NP-complete in general |
| Computing minimal cover | Polynomial |
| Testing if X is a superkey | O(\|F\| × \|U\|) — polynomial |

---

## 13. Exercises

### Exercise 1: Attribute Closure

Let R(A, B, C, D, E) with F = { AB → C, C → D, BD → E, A → B }.

Compute the following closures:
1. {A}⁺
2. {B, C}⁺
3. {A, D}⁺
4. {C, D}⁺

<details>
<summary>Solution</summary>

1. **{A}⁺**: A → B gives {A, B}; AB → C gives {A, B, C}; C → D gives {A, B, C, D}; BD → E gives {A, B, C, D, E}. **{A}⁺ = {A, B, C, D, E}**

2. **{B, C}⁺**: C → D gives {B, C, D}; BD → E gives {B, C, D, E}. No more. **{B, C}⁺ = {B, C, D, E}**

3. **{A, D}⁺**: A → B gives {A, B, D}; AB → C gives {A, B, C, D}; BD → E gives {A, B, C, D, E}. **{A, D}⁺ = {A, B, C, D, E}**

4. **{C, D}⁺**: C → D (already have D). No other FD has LHS ⊆ {C, D}. **{C, D}⁺ = {C, D}**
</details>

### Exercise 2: Finding Candidate Keys

For the relation and FDs in Exercise 1, find all candidate keys.

<details>
<summary>Solution</summary>

Classify attributes:
- A: LHS only (in AB→C, A→B) → L-only
- B: Both (LHS in AB→C, BD→E; RHS in A→B)
- C: Both (LHS in C→D; RHS in AB→C)
- D: Both (LHS in BD→E; RHS in C→D)
- E: RHS only (in BD→E) → R-only

CORE = {A} (L-only; no "Neither" attributes).
{A}⁺ = {A, B, C, D, E} = all attributes.

**{A} is the only candidate key.**
</details>

### Exercise 3: Minimal Cover

Find a minimal cover for:

```
F = { A → BC,  B → C,  AB → D,  D → BC }
```

<details>
<summary>Solution</summary>

**Step 1: Decompose RHS**
```
F = { A → B, A → C, B → C, AB → D, D → B, D → C }
```

**Step 2: Remove extraneous LHS attributes**

Check AB → D:
- Try removing A: {B}⁺ = {B, C}. D ∉ {B}⁺. Keep A.
- Try removing B: {A}⁺ = {A, B, C, D} (via A→B, A→C, then AB→D since B now included, then D→B, D→C). D ∈ {A}⁺. Remove B!

Replace AB → D with A → D.

```
F = { A → B, A → C, B → C, A → D, D → B, D → C }
```

**Step 3: Remove redundant FDs**

- A → B: Remove. {A}⁺ under F - {A→B} = {A, C, D, B, C} (A→C, A→D, D→B, D→C). B ∈ {A}⁺. **Redundant! Remove.**
- A → C: Remove. {A}⁺ under F - {A→C} = {A, D, B, C} (A→D, D→B, D→C). C ∈ {A}⁺. **Redundant! Remove.**
- B → C: Remove. {B}⁺ under F - {B→C} = {B}. C ∉ {B}⁺. **Not redundant. Keep.**
- A → D: Remove. {A}⁺ under F - {A→D} = {A}. D ∉ {A}⁺. **Not redundant. Keep.**
- D → B: Remove. {D}⁺ under F - {D→B} = {D, C}. B ∉ {D}⁺. **Not redundant. Keep.**
- D → C: Remove. {D}⁺ under F - {D→C} = {D, B, C}. C ∈ {D}⁺ (via D→B, B→C). **Redundant! Remove.**

**Minimal cover: F_min = { B → C, A → D, D → B }**
</details>

### Exercise 4: Proving an FD

Given F = { A → B, B → C, C → D }, prove that A → D using Armstrong's axioms. Write out each step explicitly.

<details>
<summary>Solution</summary>

1. A → B (given)
2. B → C (given)
3. A → C (transitivity on steps 1 and 2)
4. C → D (given)
5. A → D (transitivity on steps 3 and 4) ∎
</details>

### Exercise 5: FD Implication

Given F = { A → B, BC → D, E → C }:

Determine whether the following FDs are implied by F:
1. AE → D
2. BE → D
3. A → D

<details>
<summary>Solution</summary>

1. **AE → D**: Compute {A, E}⁺ = {A, E} → {A, B, E} (A→B) → {A, B, C, E} (E→C) → {A, B, C, D, E} (BC→D). D ∈ {A,E}⁺. **Yes, F ⊨ AE → D.** ✓

2. **BE → D**: Compute {B, E}⁺ = {B, E} → {B, C, E} (E→C) → {B, C, D, E} (BC→D). D ∈ {B,E}⁺. **Yes, F ⊨ BE → D.** ✓

3. **A → D**: Compute {A}⁺ = {A} → {A, B} (A→B). No more applicable. D ∉ {A}⁺. **No, F ⊭ A → D.** ✗
</details>

### Exercise 6: Equivalence of FD Sets

Are the following two sets of FDs equivalent?

```
F = { A → B, B → C, A → C }
G = { A → B, B → C }
```

<details>
<summary>Solution</summary>

Check if every FD in F is implied by G:
- A → B: In G directly. ✓
- B → C: In G directly. ✓
- A → C: {A}⁺_G = {A, B, C}. C ∈ {A}⁺. ✓

Check if every FD in G is implied by F:
- A → B: In F directly. ✓
- B → C: In F directly. ✓

**F ≡ G.** The FD A → C in F is redundant; it follows from A → B and B → C by transitivity.
</details>

### Exercise 7: Multiple Candidate Keys

Let R(A, B, C, D, E) with F = { AB → CDE, C → A, D → B }.

Find all candidate keys.

<details>
<summary>Solution</summary>

Classify attributes:
- A: Both (LHS: AB→CDE; RHS: C→A)
- B: Both (LHS: AB→CDE; RHS: D→B)
- C: Both (LHS: C→A; RHS: AB→CDE)
- D: Both (LHS: D→B; RHS: AB→CDE)
- E: RHS only → R-only

No L-only or Neither attributes, so CORE = {}.

Try single attributes:
- {A}⁺ = {A}. Not superkey.
- {B}⁺ = {B}. Not superkey.
- {C}⁺ = {C, A} = {A, C}. Not superkey.
- {D}⁺ = {D, B} = {B, D}. Not superkey.

Try pairs:
- {A, B}⁺ = {A, B, C, D, E}. **Superkey!** Minimal (neither {A} nor {B} alone works). **Candidate key: {A, B}**
- {A, D}⁺ = {A, B, D} → {A, B, C, D, E}. **Superkey!** Check: {A}⁺={A}, {D}⁺={B,D}. Minimal. **Candidate key: {A, D}**
- {B, C}⁺ = {A, B, C} → {A, B, C, D, E}. **Superkey!** Check: {B}⁺={B}, {C}⁺={A,C}. Minimal. **Candidate key: {B, C}**
- {C, D}⁺ = {A, B, C, D} → {A, B, C, D, E}. **Superkey!** Check: {C}⁺={A,C}, {D}⁺={B,D}. Minimal. **Candidate key: {C, D}**

**All candidate keys: {A,B}, {A,D}, {B,C}, {C,D}**
</details>

### Exercise 8: Closure Proof

Prove that the Union Rule (X → Y, X → Z ⟹ X → YZ) follows from Armstrong's axioms.

<details>
<summary>Solution</summary>

1. X → Y (given)
2. X → XY (augment step 1 with X: XX → XY, and XX = X)
3. X → Z (given)
4. XY → YZ (augment step 3 with Y: XY → ZY)
5. X → YZ (transitivity on steps 2 and 4) ∎
</details>

---

## 14. Summary

| Concept | Key Point |
|---------|-----------|
| **Functional Dependency** | X → Y means X uniquely determines Y |
| **Armstrong's Axioms** | Reflexivity, Augmentation, Transitivity — sound and complete |
| **Derived Rules** | Union, Decomposition, Pseudotransitivity |
| **Attribute Closure X⁺** | All attributes determined by X — the key algorithm |
| **FD Set Closure F⁺** | All FDs implied by F — usually computed via X⁺ |
| **Minimal Cover** | Simplified equivalent FD set — needed for normalization |
| **Candidate Keys** | Found by classifying attributes and computing closures |

Functional dependencies are the theoretical bedrock of relational database design. The algorithms in this lesson — attribute closure and minimal cover — will be used extensively in the next lesson on normalization, where we apply FDs to decompose relations into well-structured schemas.

---

**Previous**: [04_ER_Modeling.md](./04_ER_Modeling.md) | **Next**: [06_Normalization.md](./06_Normalization.md)
