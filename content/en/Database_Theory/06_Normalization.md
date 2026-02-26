# Lesson 06: Normalization (1NF through BCNF)

**Previous**: [05_Functional_Dependencies.md](./05_Functional_Dependencies.md) | **Next**: [07_Advanced_Normalization.md](./07_Advanced_Normalization.md)

---

> **Topic**: Database Theory
> **Lesson**: 6 of 16
> **Prerequisites**: Functional dependencies, attribute closure, minimal cover (Lesson 05)
> **Objective**: Understand normalization from 1NF through BCNF, master decomposition algorithms, verify lossless-join and dependency-preservation properties, and apply normalization to real-world schemas

## Learning Objectives

After completing this lesson, you will be able to:

1. Identify the three types of update anomalies (insertion, deletion, update) in unnormalized schemas and explain how normalization resolves them.
2. Determine whether a relation satisfies 1NF, 2NF, 3NF, or BCNF by checking for atomic values, partial dependencies, and transitive dependencies.
3. Apply the BCNF decomposition algorithm to produce a lossless-join decomposition of a schema.
4. Apply the 3NF synthesis algorithm to produce a lossless-join, dependency-preserving decomposition.
5. Verify lossless-join and dependency-preservation properties for a given decomposition using formal tests.
6. Evaluate the trade-offs between BCNF and 3NF decomposition strategies and choose an appropriate normalization target for a given schema.

---

## 1. Introduction

Normalization is the process of organizing a relational database schema to reduce redundancy and eliminate certain types of data anomalies. Introduced by Edgar F. Codd in 1970, it provides a systematic, theory-driven approach to schema design.

### 1.1 The Problem: Poor Schema Design

Consider this single-relation design for a university:

```
UniversityCourse(
    student_id, student_name, student_addr,
    course_id, course_title, dept_name, dept_building,
    instructor_id, instructor_name,
    grade, semester
)
```

This "universal relation" stores everything in one table. While it works for simple queries, it suffers from serious problems.

### 1.2 Data Anomalies

#### Update Anomaly

If the Computer Science department moves to a new building, we must update `dept_building` in **every row** where `dept_name = 'Computer Science'`. If we miss even one row, the data becomes inconsistent.

```
Before:
| course_id | dept_name | dept_building |
|-----------|-----------|---------------|
| CS101     | CS        | Watson Hall   |
| CS201     | CS        | Watson Hall   |    ← must update ALL rows
| CS301     | CS        | Watson Hall   |

If we update only the first row:
| CS101     | CS        | Taylor Hall   |    ← updated
| CS201     | CS        | Watson Hall   |    ← inconsistent!
| CS301     | CS        | Watson Hall   |    ← inconsistent!
```

#### Insertion Anomaly

We cannot record a new department (e.g., its name and building) unless a student enrolls in one of its courses, because `student_id` is part of the primary key.

#### Deletion Anomaly

If the last student enrolled in a course drops it, we lose not just the enrollment data but also the course title, instructor assignment, and department information.

### 1.3 Root Cause: Redundancy from FD Violations

All three anomalies stem from the same root cause: **attributes that depend on only part of the key, or on non-key attributes, are stored redundantly**. Normalization eliminates this redundancy through systematic decomposition guided by functional dependencies.

### 1.4 Goals of Normalization

1. **Eliminate redundancy**: Each fact is stored exactly once
2. **Prevent anomalies**: Updates, insertions, and deletions are clean
3. **Preserve information**: No data is lost during decomposition (lossless join)
4. **Preserve constraints**: FDs are still enforceable (dependency preservation)

---

## 2. First Normal Form (1NF)

### 2.1 Definition

> **Definition**: A relation is in **First Normal Form (1NF)** if:
> 1. All attributes contain only **atomic** (indivisible) values
> 2. There are no **repeating groups** or arrays
> 3. Each row is uniquely identifiable (has a primary key)

1NF is the baseline requirement for being a valid relation in the relational model.

### 2.2 Violations and Fixes

**Violation 1: Non-atomic values**

```
| student_id | name       | phone_numbers          |
|------------|------------|------------------------|
| 101        | Alice      | 555-1234, 555-5678     |    ← multi-valued!
| 102        | Bob        | 555-9999               |
```

**Fix**: Create a separate row for each phone number, or a separate table:

```
Student(student_id, name)
StudentPhone(student_id, phone_number)

| student_id | phone_number |
|------------|--------------|
| 101        | 555-1234     |
| 101        | 555-5678     |
| 102        | 555-9999     |
```

**Violation 2: Repeating groups**

```
| order_id | item1  | qty1 | item2  | qty2 | item3  | qty3 |
|----------|--------|------|--------|------|--------|------|
| 1001     | Pen    | 5    | Paper  | 10   | NULL   | NULL |
```

**Fix**: Normalize into two tables:

```
Order(order_id, order_date, customer_id)
OrderItem(order_id, item_name, quantity)
```

### 2.3 1NF and the Relational Model

In strict relational theory, a relation is by definition in 1NF — the relational model doesn't allow non-atomic domains. However, in practice, many systems allow arrays (PostgreSQL `int[]`), JSON columns, or comma-separated values. While sometimes useful for performance, these violate the spirit of 1NF and make FD-based reasoning difficult.

---

## 3. Second Normal Form (2NF)

### 3.1 Partial Dependency

> **Definition**: A **partial dependency** exists when a non-prime attribute (an attribute not part of any candidate key) is functionally dependent on a **proper subset** of a candidate key.

In other words, some attribute depends on only *part* of the key, not the whole key.

### 3.2 Definition

> **Definition**: A relation is in **Second Normal Form (2NF)** if:
> 1. It is in 1NF, and
> 2. Every non-prime attribute is **fully functionally dependent** on every candidate key (no partial dependencies)

Note: 2NF is only relevant when a candidate key is composite (has more than one attribute). If all candidate keys are single attributes, the relation is automatically in 2NF.

### 3.3 Example

Consider:

```
StudentCourse(student_id, course_id, student_name, grade)
```

Candidate key: {student_id, course_id}

FDs:
- {student_id, course_id} → grade (full dependency)
- student_id → student_name (partial dependency! student_name depends on only part of the key)

This violates 2NF.

**Decomposition to achieve 2NF:**

```
Student(student_id, student_name)
    Key: {student_id}
    FD: student_id → student_name

Enrollment(student_id, course_id, grade)
    Key: {student_id, course_id}
    FD: {student_id, course_id} → grade
```

### 3.4 Formal Test for 2NF

For each candidate key K and each non-prime attribute A:
1. Check if there exists a proper subset X ⊂ K such that X → A
2. If any such partial dependency exists, the relation is not in 2NF

---

## 4. Third Normal Form (3NF)

### 4.1 Transitive Dependency

> **Definition**: A **transitive dependency** exists when a non-prime attribute A depends on another non-prime attribute B, which in turn depends on a candidate key K:
>
> K → B → A, where B is not a superkey and A is not part of any candidate key.

### 4.2 Definition

> **Definition**: A relation schema R is in **Third Normal Form (3NF)** if for every non-trivial functional dependency X → A where A is a single attribute:
> 1. X is a superkey of R, **OR**
> 2. A is a **prime attribute** (member of some candidate key)

Equivalently: no non-prime attribute is transitively dependent on any candidate key.

### 4.3 Example

Consider:

```
Employee(emp_id, dept_id, dept_name, dept_location)
```

Candidate key: {emp_id}

FDs:
- emp_id → dept_id, dept_name, dept_location
- dept_id → dept_name, dept_location

The dependency emp_id → dept_name is transitive through dept_id:
- emp_id → dept_id → dept_name

This violates 3NF because dept_name depends on dept_id (a non-superkey) and dept_name is not a prime attribute.

**Decomposition to achieve 3NF:**

```
Employee(emp_id, dept_id)
    Key: {emp_id}
    FD: emp_id → dept_id

Department(dept_id, dept_name, dept_location)
    Key: {dept_id}
    FD: dept_id → dept_name, dept_location
```

### 4.4 The Importance of "Prime Attribute" Exception

The condition "A is a prime attribute" in 3NF is what distinguishes it from BCNF. This exception allows certain FDs where the dependent is part of a candidate key, even if the determinant is not a superkey. This exception is what makes 3NF decomposition always dependency-preserving (unlike BCNF).

---

## 5. Boyce-Codd Normal Form (BCNF)

### 5.1 Definition

> **Definition**: A relation schema R is in **Boyce-Codd Normal Form (BCNF)** if for every non-trivial functional dependency X → Y:
>
> X is a superkey of R.

BCNF is strictly stronger than 3NF. It eliminates the "prime attribute" exception: **every** determinant must be a superkey, period.

### 5.2 Relationship: 3NF vs BCNF

```
BCNF ⊂ 3NF ⊂ 2NF ⊂ 1NF

Every BCNF relation is in 3NF.
Every 3NF relation is in 2NF.
Every 2NF relation is in 1NF.

But not vice versa.
```

### 5.3 Example: 3NF but Not BCNF

Consider:

```
TeachingAssignment(student_id, course, instructor)
```

Business rules:
- Each instructor teaches exactly one course: instructor → course
- Each student has exactly one instructor per course: {student_id, course} → instructor
- A student can have only one instructor for a given course

Candidate keys: {student_id, course} and {student_id, instructor}

FDs:
- {student_id, course} → instructor
- {student_id, instructor} → course
- instructor → course

Check 3NF for instructor → course:
- instructor is not a superkey ✗
- course is a prime attribute (part of candidate key {student_id, course}) ✓

So the relation is in **3NF** (condition 2 of the 3NF definition is satisfied).

Check BCNF for instructor → course:
- instructor is not a superkey ✗

So the relation is **not in BCNF**.

### 5.4 When 3NF and BCNF Differ

3NF and BCNF differ only when:
1. There are multiple overlapping candidate keys, AND
2. A non-superkey attribute determines part of a candidate key

In practice, this situation is relatively rare. Most relations that are in 3NF are also in BCNF.

---

## 6. Decomposition Properties

When we decompose a relation to achieve a higher normal form, we must ensure the decomposition is "good." Two critical properties define what "good" means.

### 6.1 Lossless-Join Property

> **Definition**: A decomposition of R into R₁, R₂, ..., Rₙ has the **lossless-join property** if for every legal instance r of R:
>
> r = π_{R₁}(r) ⋈ π_{R₂}(r) ⋈ ... ⋈ π_{Rₙ}(r)

In other words, we can reconstruct the original relation by naturally joining the decomposed relations. No information is lost.

#### Binary Decomposition Test

For a decomposition of R into R₁ and R₂:

> **Theorem**: The decomposition is lossless-join if and only if:
>
> (R₁ ∩ R₂) → (R₁ - R₂) ∈ F⁺, or
> (R₁ ∩ R₂) → (R₂ - R₁) ∈ F⁺

The common attributes must functionally determine all of one side. Equivalently, the common attributes must be a superkey of at least one of the decomposed relations.

#### Example

Decompose Employee(emp_id, dept_id, dept_name) into:
- R₁(emp_id, dept_id) and R₂(dept_id, dept_name)

Common attributes: {dept_id}
R₂ - R₁ = {dept_name}

Is dept_id → dept_name in F⁺? Yes!

So this decomposition is lossless-join. ✓

#### Why Lossless-Join Matters

Without the lossless-join property, joining the decomposed tables produces **spurious tuples** — rows that didn't exist in the original relation:

```
Original:
| A | B | C |
|---|---|---|
| 1 | 2 | 3 |
| 4 | 2 | 5 |

Decompose into R1(A,B) and R2(B,C):
R1:          R2:
| A | B |    | B | C |
|---|---|    |---|---|
| 1 | 2 |    | 2 | 3 |
| 4 | 2 |    | 2 | 5 |

R1 ⋈ R2:
| A | B | C |
|---|---|---|
| 1 | 2 | 3 |    ← original ✓
| 1 | 2 | 5 |    ← SPURIOUS! ✗
| 4 | 2 | 3 |    ← SPURIOUS! ✗
| 4 | 2 | 5 |    ← original ✓
```

Here, B does not determine A or C, so the decomposition is lossy (not lossless).

### 6.2 Dependency Preservation

> **Definition**: A decomposition of R into R₁, R₂, ..., Rₙ is **dependency-preserving** if:
>
> (F₁ ∪ F₂ ∪ ... ∪ Fₙ)⁺ = F⁺
>
> where Fᵢ is the set of FDs in F⁺ that involve only attributes of Rᵢ.

In simpler terms: every FD in the original F can be verified by checking constraints within individual decomposed relations, without needing to join tables.

#### Why Dependency Preservation Matters

If a decomposition is not dependency-preserving, some FDs can only be enforced by joining multiple tables — this is expensive and often impractical. Without dependency preservation, the DBMS cannot efficiently maintain data integrity.

#### Example

R(A, B, C) with F = { A → B, B → C }

Decompose into R₁(A, C) and R₂(B, C).

FDs on R₁: Can we enforce A → B? No — B is not in R₁.
FDs on R₂: Can we enforce A → B? No — A is not in R₂.

The FD A → B is **not preserved**. To check it, we'd need to join R₁ and R₂.

A better decomposition: R₁(A, B) and R₂(B, C) — this preserves both A → B and B → C.

### 6.3 Can We Always Have Both?

| Normal Form | Lossless Join | Dependency Preservation |
|-------------|:------------:|:----------------------:|
| 3NF | Always achievable ✓ | Always achievable ✓ |
| BCNF | Always achievable ✓ | **Not always** ✗ |

This is the key tradeoff: BCNF is a stricter normal form, but achieving it may sacrifice dependency preservation. 3NF guarantees both properties.

---

## 7. The Lossless-Join Test Algorithm (For n-ary Decomposition)

For decompositions into more than two relations, we use a tabular algorithm.

### 7.1 Algorithm (Chase Test)

```
ALGORITHM: LosslessJoinTest(R, F, {R₁, R₂, ..., Rₙ})
INPUT:  R = {A₁, A₂, ..., Aₘ} (m attributes)
        F = set of FDs
        {R₁, R₂, ..., Rₙ} = decomposition
OUTPUT: TRUE if lossless-join, FALSE otherwise

Step 1: Create an n × m matrix.
        Row i corresponds to Rᵢ, column j to attribute Aⱼ.

Step 2: Initialize the matrix:
        If Aⱼ ∈ Rᵢ, set entry[i][j] = aⱼ (distinguished symbol)
        Otherwise, set entry[i][j] = bᵢⱼ (subscripted symbol)

Step 3: REPEAT
            FOR EACH FD (X → Y) IN F DO
                Find all rows that agree on all columns in X
                For those rows, make their Y-columns equal:
                    If any row has aⱼ for some column in Y, set all matching rows to aⱼ
                    Otherwise, pick one bᵢⱼ and set all matching rows to that value
            END FOR
        UNTIL no change occurs

Step 4: If any row has all distinguished symbols (a₁, a₂, ..., aₘ), RETURN TRUE.
        Otherwise, RETURN FALSE.
```

### 7.2 Worked Example

R(A, B, C, D, E), F = { A → C, B → C, C → D, DE → C, CE → A }

Decomposition: R₁(A, D), R₂(A, B), R₃(B, E), R₄(C, D, E), R₅(A, E)

**Step 1-2: Initial matrix**

|      | A   | B   | C   | D   | E   |
|------|-----|-----|-----|-----|-----|
| R₁   | a₁  | b₁₂ | b₁₃ | a₄  | b₁₅ |
| R₂   | a₁  | a₂  | b₂₃ | b₂₄ | b₂₅ |
| R₃   | b₃₁ | a₂  | b₃₃ | b₃₄ | a₅  |
| R₄   | b₄₁ | b₄₂ | a₃  | a₄  | a₅  |
| R₅   | a₁  | b₅₂ | b₅₃ | b₅₄ | a₅  |

**Step 3: Apply FDs iteratively**

Apply A → C: Rows with same A value.
- Rows R₁, R₂, R₅ have A = a₁. Make their C columns equal.
  - R₁: b₁₃, R₂: b₂₃, R₅: b₅₃. No distinguished symbol. Pick b₁₃ for all.
  - R₁: b₁₃, R₂: b₁₃, R₅: b₁₃

Apply B → C: Rows R₂, R₃ have B = a₂.
- R₂: b₁₃, R₃: b₃₃. Pick b₁₃.
- R₃: b₁₃

Apply C → D: Rows with same C value.
- Rows R₁, R₂, R₃, R₅ all have C = b₁₃. R₄ has C = a₃ (different).
  - R₁: a₄, R₂: b₂₄, R₃: b₃₄, R₅: b₅₄. R₁ has distinguished a₄. Set all to a₄.
  - R₂: a₄, R₃: a₄, R₅: a₄

Apply DE → C: Rows with same D and E values.
- R₃: D=a₄, E=a₅; R₄: D=a₄, E=a₅; R₅: D=a₄, E=a₅.
  - These rows agree on DE. Make C equal: R₃: b₁₃, R₄: a₃, R₅: b₁₃. R₄ has a₃. Set all to a₃.
  - R₃: a₃, R₅: a₃.

Apply CE → A: Rows with same C and E.
- R₃: C=a₃, E=a₅; R₄: C=a₃, E=a₅; R₅: C=a₃, E=a₅.
  - R₅ has A=a₁ (distinguished). Set R₃: a₁, R₄: a₁.

Now re-apply A → C: Rows R₁,R₂,R₃,R₄,R₅ have A=a₁.
- C values: R₁=b₁₃, R₂=b₁₃, R₃=a₃, R₄=a₃, R₅=a₃. Has a₃. Set all to a₃.
- R₁=a₃, R₂=a₃.

Re-apply C → D on all rows (now all C=a₃): already all D=a₄. No change.

Check row R₅: A=a₁, B=b₅₂, C=a₃, D=a₄, E=a₅. Still has b₅₂ for B.

Apply B → C: R₂ has B=a₂, R₃ has B=a₂. Both already C=a₃. No change.

Check: Row R₃ = (a₁, a₂, a₃, a₄, a₅) — **all distinguished symbols!**

**Result: The decomposition is lossless-join. ✓**

---

## 8. 3NF Synthesis Algorithm

The 3NF synthesis algorithm produces a decomposition that is both **lossless-join** and **dependency-preserving**.

### 8.1 Algorithm

```
ALGORITHM: 3NF_Synthesis(R, F)
INPUT:  R = relation schema, F = set of FDs
OUTPUT: Decomposition {R₁, R₂, ..., Rₙ} in 3NF with lossless join
        and dependency preservation

Step 1: Compute the minimal cover F_min of F.

Step 2: For each FD X → A in F_min:
            Create a relation schema Rᵢ = X ∪ {A}
        Group FDs with the same LHS:
            If X → A₁, X → A₂, ..., X → Aₖ all have the same X,
            create one schema Rᵢ = X ∪ {A₁, A₂, ..., Aₖ}

Step 3: If none of the schemas contains a candidate key of R,
        add a schema Rₖ = any candidate key of R.

Step 4: Remove any schema Rᵢ that is a subset of another schema Rⱼ.

RETURN {R₁, R₂, ..., Rₙ}
```

### 8.2 Why Each Step Matters

- **Step 1** (minimal cover): Ensures no redundant FDs generate extra tables
- **Step 2** (one table per LHS group): Directly preserves each FD
- **Step 3** (add key if needed): Guarantees lossless-join property
- **Step 4** (remove subsets): Eliminates redundant tables

### 8.3 Worked Example

R(A, B, C, D, E, H) with:

```
F = { A → BC,  E → HA,  B → D }
```

**Step 1: Compute minimal cover**

Decompose RHS:
```
F = { A → B, A → C, E → H, E → A, B → D }
```

Check extraneous LHS: All LHS are single attributes. Nothing to simplify.

Check redundant FDs:
- A → B: Remove. {A}⁺ under rest = {A, C} ∪ ... = {A, C}. B ∉ {A}⁺. Keep.
- A → C: Remove. {A}⁺ under rest = {A, B, D} (via A→B, B→D). C ∉ {A}⁺. Keep.
- E → H: Remove. {E}⁺ under rest = {E, A, B, C, D} (E→A, A→B, A→C, B→D). H ∉ {E}⁺. Keep.
- E → A: Remove. {E}⁺ under rest = {E, H}. A ∉ {E}⁺. Keep.
- B → D: Remove. {B}⁺ under rest = {B}. D ∉ {B}⁺. Keep.

Minimal cover:
```
F_min = { A → B, A → C, E → H, E → A, B → D }
```

**Step 2: Group by LHS and create schemas**

| LHS | FDs | Schema |
|-----|-----|--------|
| A | A → B, A → C | R₁(A, B, C) |
| E | E → H, E → A | R₂(E, H, A) |
| B | B → D | R₃(B, D) |

**Step 3: Check for candidate key**

Compute {E}⁺ = {E, H, A, B, C, D} = all attributes. So {E} is a candidate key.

Is E in any schema? R₂ contains E. R₂'s key is E. ✓ (A candidate key is present.)

**Step 4: Remove subset schemas**

None is a subset of another.

**Final decomposition:**
```
R₁(A, B, C)    — key: {A}
R₂(E, H, A)    — key: {E}
R₃(B, D)       — key: {B}
```

All in 3NF ✓, lossless-join ✓, dependency-preserving ✓.

---

## 9. BCNF Decomposition Algorithm

### 9.1 Algorithm

```
ALGORITHM: BCNF_Decomposition(R, F)
INPUT:  R = relation schema, F = set of FDs
OUTPUT: Decomposition into BCNF relations with lossless join

result ← {R}

WHILE there exists Rᵢ in result that is not in BCNF DO
    Find an FD X → Y that violates BCNF in Rᵢ
    (i.e., X is not a superkey of Rᵢ, and Y ⊄ X)

    Compute X⁺ with respect to F

    Replace Rᵢ with:
        R₁ = X⁺ ∩ attributes(Rᵢ)    (everything X determines within Rᵢ)
        R₂ = X ∪ (attributes(Rᵢ) - X⁺)   (X plus what it doesn't determine)
END WHILE

RETURN result
```

The key decomposition step splits Rᵢ into:
- R₁: all attributes that X determines (within Rᵢ) — X is a key of R₁
- R₂: X plus the remaining attributes — X is a foreign key back to R₁

This guarantees lossless join (R₁ ∩ R₂ = X, and X → R₁).

### 9.2 Worked Example

R(A, B, C, D) with F = { AB → C, C → B, AB → D }

**Step 1: Is R in BCNF?**

Candidate key: {A, B} ({A,B}⁺ = {A,B,C,D}).
Also {A, C} is a candidate key ({A,C}⁺: C→B gives {A,B,C}, AB→D gives {A,B,C,D}).

Check FDs:
- AB → C: {A,B} is a superkey. ✓
- C → B: {C}⁺ = {C, B}. C is not a superkey. **BCNF violation!**
- AB → D: {A,B} is a superkey. ✓

**Step 2: Decompose on C → B**

Compute {C}⁺ ∩ {A,B,C,D} = {B, C} ∩ {A,B,C,D} = {B, C}

- R₁ = {B, C} with FD: C → B (key: C)
- R₂ = {C} ∪ ({A,B,C,D} - {B,C}) = {A, C, D}

Project FDs onto R₂(A, C, D):
- AB → C becomes irrelevant (B ∉ R₂)
- C → B becomes irrelevant (B ∉ R₂)
- AB → D becomes... need to check: under projected FDs, {A,C}⁺ on R₂. C→B is not in R₂. But from original: AC → D? {A,C}⁺ = {A,C,B,D} (C→B, AB→D). So AC → D holds. Key of R₂ is {A, C}.
- Check: is AC a superkey of R₂? {A,C}⁺ restricted to R₂ = {A,C,D}. Yes. ✓

**Step 3: Check R₁ and R₂**

- R₁(B, C), FD: C → B. Key: {C}. C is a superkey. BCNF ✓
- R₂(A, C, D), FD: AC → D. Key: {A, C}. AC is a superkey. BCNF ✓

**Final BCNF decomposition:**
```
R₁(B, C)      — key: {C}
R₂(A, C, D)   — key: {A, C}
```

**Check dependency preservation:**
- AB → C: cannot be checked from R₁ or R₂ alone (A and B are never in the same table). **Not preserved!**
- C → B: in R₁. ✓
- AB → D: not directly preserved, but AC → D is in R₂.

This illustrates the BCNF tradeoff: we achieved BCNF but lost the FD AB → C.

### 9.3 BCNF vs 3NF: The Tradeoff

| Property | 3NF Synthesis | BCNF Decomposition |
|----------|:------------:|:------------------:|
| Lossless Join | ✓ Always | ✓ Always |
| Dependency Preservation | ✓ Always | ✗ Not always |
| Redundancy Elimination | Good (minimal) | Best (none from FDs) |
| When to prefer | When dependency preservation is critical | When minimal redundancy is critical |

**Practical guidance**:
- Start with BCNF decomposition
- If dependency preservation fails, fall back to 3NF
- In practice, most schemas achieve BCNF without losing dependencies

---

## 10. Complete Worked Example: Unnormalized to BCNF

### 10.1 Scenario

A company tracks project assignments:

```
ProjectAssignment(
    emp_id, emp_name, emp_phone,
    dept_id, dept_name, dept_budget,
    proj_id, proj_name, proj_budget,
    hours_worked, role
)
```

**Business rules (FDs):**
```
F = {
    emp_id → emp_name, emp_phone, dept_id,
    dept_id → dept_name, dept_budget,
    proj_id → proj_name, proj_budget,
    {emp_id, proj_id} → hours_worked, role
}
```

**Candidate key**: {emp_id, proj_id}

### 10.2 Check 1NF

All values are atomic (single values, no arrays). ✓ In 1NF.

### 10.3 Check 2NF

Non-prime attributes: emp_name, emp_phone, dept_id, dept_name, dept_budget, proj_name, proj_budget, hours_worked, role.

Partial dependencies (attribute depends on proper subset of key {emp_id, proj_id}):
- emp_id → emp_name, emp_phone, dept_id (partial: depends on emp_id alone)
- proj_id → proj_name, proj_budget (partial: depends on proj_id alone)

**Not in 2NF.** Decompose:

```
Employee(emp_id, emp_name, emp_phone, dept_id)
    FDs: emp_id → emp_name, emp_phone, dept_id

Project(proj_id, proj_name, proj_budget)
    FDs: proj_id → proj_name, proj_budget

Assignment(emp_id, proj_id, hours_worked, role)
    FDs: {emp_id, proj_id} → hours_worked, role
```

Now in 2NF. ✓

### 10.4 Check 3NF

**Employee(emp_id, emp_name, emp_phone, dept_id)**

Key: {emp_id}

Are there transitive dependencies?
- emp_id → dept_id (direct) ✓
- But where are dept_name, dept_budget? They were removed — but wait, we also have dept_id → dept_name, dept_budget from the original FDs. Since dept_name and dept_budget are not in this relation anymore, no transitive dependency exists within this relation.

Actually, let's reconsider. The original FD dept_id → dept_name, dept_budget means Employee should not contain dept_name or dept_budget. And it doesn't — they went with the decomposition in step 2. But we need a Department table:

```
Employee(emp_id, emp_name, emp_phone, dept_id)
```

Is there any transitive dependency among the remaining attributes? emp_id → emp_name, emp_phone, dept_id. All are direct dependencies from the key. No non-prime attribute determines another non-prime attribute within this table (emp_phone doesn't determine dept_id, etc.).

In 3NF ✓.

But the FD dept_id → dept_name, dept_budget from the original schema is "orphaned." We need a Department table:

```
Department(dept_id, dept_name, dept_budget)
    Key: {dept_id}
```

**All relations now in 3NF:**
```
Employee(emp_id, emp_name, emp_phone, dept_id)    — key: {emp_id}
Department(dept_id, dept_name, dept_budget)         — key: {dept_id}
Project(proj_id, proj_name, proj_budget)           — key: {proj_id}
Assignment(emp_id, proj_id, hours_worked, role)     — key: {emp_id, proj_id}
```

### 10.5 Check BCNF

For each relation, check: is every determinant a superkey?

- **Employee**: emp_id → (emp_name, emp_phone, dept_id). emp_id is the key. ✓
- **Department**: dept_id → (dept_name, dept_budget). dept_id is the key. ✓
- **Project**: proj_id → (proj_name, proj_budget). proj_id is the key. ✓
- **Assignment**: {emp_id, proj_id} → (hours_worked, role). {emp_id, proj_id} is the key. ✓

**All in BCNF!** ✓

### 10.6 Summary of Decomposition

```
ORIGINAL (unnormalized):
    ProjectAssignment(emp_id, emp_name, emp_phone, dept_id, dept_name,
                      dept_budget, proj_id, proj_name, proj_budget,
                      hours_worked, role)

FINAL (BCNF):
    Employee(emp_id, emp_name, emp_phone, dept_id)
    Department(dept_id, dept_name, dept_budget)
    Project(proj_id, proj_name, proj_budget)
    Assignment(emp_id, proj_id, hours_worked, role)
```

Anomalies eliminated:
- **Update**: Changing a department name requires updating only one row in Department
- **Insert**: Can add a department without any employees
- **Delete**: Removing all assignments for a project doesn't lose project information

---

## 11. Normalization in SQL

### 11.1 Implementing the Normalized Schema

```sql
CREATE TABLE Department (
    dept_id     INT PRIMARY KEY,
    dept_name   VARCHAR(100) NOT NULL,
    dept_budget DECIMAL(12, 2) NOT NULL
);

CREATE TABLE Employee (
    emp_id    INT PRIMARY KEY,
    emp_name  VARCHAR(100) NOT NULL,
    emp_phone VARCHAR(20),
    dept_id   INT NOT NULL,
    FOREIGN KEY (dept_id) REFERENCES Department(dept_id)
);

CREATE TABLE Project (
    proj_id     INT PRIMARY KEY,
    proj_name   VARCHAR(100) NOT NULL,
    proj_budget DECIMAL(12, 2) NOT NULL
);

CREATE TABLE Assignment (
    emp_id       INT NOT NULL,
    proj_id      INT NOT NULL,
    hours_worked DECIMAL(6, 2) NOT NULL DEFAULT 0,
    role         VARCHAR(50) NOT NULL,
    PRIMARY KEY (emp_id, proj_id),
    FOREIGN KEY (emp_id) REFERENCES Employee(emp_id),
    FOREIGN KEY (proj_id) REFERENCES Project(proj_id)
);
```

### 11.2 Verifying Normalization via Queries

To check for potential normalization issues in an existing database:

```sql
-- Check for potential 2NF violations: partial dependencies
-- If a non-key column has duplicate values correlated with part of a composite key
SELECT emp_id, COUNT(DISTINCT emp_name) AS name_count
FROM project_assignment_denormalized
GROUP BY emp_id
HAVING COUNT(DISTINCT emp_name) > 1;
-- If this returns rows, emp_name is inconsistently stored (update anomaly)

-- Check for potential 3NF violations: transitive dependencies
-- Columns that move together might indicate a missing entity
SELECT dept_id, COUNT(DISTINCT dept_name) AS names
FROM employee_denormalized
GROUP BY dept_id
HAVING COUNT(DISTINCT dept_name) > 1;
-- If this returns rows, dept_name is inconsistent for a given dept_id
```

---

## 12. Summary of Normal Forms

| Normal Form | Condition | Eliminates |
|-------------|-----------|-----------|
| **1NF** | Atomic values, no repeating groups | Non-relational structures |
| **2NF** | 1NF + no partial dependencies | Redundancy from partial key dependencies |
| **3NF** | 2NF + no transitive dependencies | Redundancy from transitive dependencies |
| **BCNF** | Every determinant is a superkey | All FD-based redundancy |

### Decision Flowchart

```
Start: Relation R with FDs F

Is R in 1NF?
├── No → Remove non-atomic values and repeating groups
└── Yes ↓

Is R in 2NF?
├── No → Remove partial dependencies (separate out attributes
│         that depend on part of the key)
└── Yes ↓

Is R in 3NF?
├── No → Remove transitive dependencies (separate out attributes
│         that depend on non-key attributes)
└── Yes ↓

Is R in BCNF?
├── No → Check: is dependency preservation acceptable to lose?
│   ├── Yes → Decompose using BCNF algorithm
│   └── No → Stay at 3NF
└── Yes → Done!
```

---

## 13. Exercises

### Exercise 1: Identifying Normal Forms

For each relation, identify the highest normal form (1NF, 2NF, 3NF, or BCNF):

**a)** R(A, B, C, D), Key: {A, B}, FDs: A → C, AB → D

**b)** R(A, B, C), Key: {A}, FDs: A → B, B → C

**c)** R(A, B, C, D), Key: {A}, FDs: A → BCD

**d)** R(A, B, C), Keys: {A, B} and {A, C}, FDs: AB → C, AC → B, B → C, C → B

<details>
<summary>Solution</summary>

**a)** A → C is a partial dependency (C depends on part of key {A,B}). **1NF** (not 2NF).

**b)** A → B (direct from key, OK), B → C (transitive: A → B → C). Not 3NF. But no partial dependency (single-attribute key), so 2NF. **2NF** (not 3NF).

**c)** Only FD is from the key. Every determinant (A) is a superkey. **BCNF**.

**d)** B → C: B is not a superkey, but C is a prime attribute (part of key {A,C}). So 3NF holds. B is not a superkey, so BCNF fails. **3NF** (not BCNF).
</details>

### Exercise 2: 3NF Synthesis

Apply the 3NF synthesis algorithm to:

R(A, B, C, D, E) with F = { A → B, BC → D, D → E, E → C }

<details>
<summary>Solution</summary>

**Step 1: Minimal cover**

Decompose RHS: already single attributes.

Check extraneous LHS in BC → D:
- Remove B: {C}⁺ = {C}. D ∉ {C}⁺. Keep B.
- Remove C: {B}⁺ = {B}. D ∉ {B}⁺. Keep C.

Check redundant FDs:
- A → B: {A}⁺ without A→B = {A}. B ∉ {A}⁺. Keep.
- BC → D: {B,C}⁺ without BC→D = {B,C}. D ∉ {B,C}⁺. Keep.
- D → E: {D}⁺ without D→E = {D}. E ∉ {D}⁺. Keep.
- E → C: {E}⁺ without E→C = {E}. C ∉ {E}⁺. Keep.

F_min = { A → B, BC → D, D → E, E → C }

**Step 2: Create schemas (group by LHS)**

- R₁(A, B) from A → B
- R₂(B, C, D) from BC → D
- R₃(D, E) from D → E
- R₄(E, C) from E → C

**Step 3: Check for candidate key**

{A}⁺ = {A, B}. Not all.
{A, C}⁺ = {A, B, C, D, E}. All! Candidate key: {A, C}.
{A, E}⁺ = {A, B, C, D, E} (E→C, A→B, BC→D, D→E). All! Candidate key: {A, E}.
{A, D}⁺ = {A, B, D, E, C}. All! Candidate key: {A, D}.

None of R₁-R₄ contains {A,C}, {A,E}, or {A,D} entirely.
- R₁ = {A,B}: no
- R₂ = {B,C,D}: no A
- R₃ = {D,E}: no A
- R₄ = {E,C}: no A

Add R₅ = {A, C} (or {A, D} or {A, E}).

**Step 4: Remove subsets**

R₄(E, C) ⊆ R₂(B, C, D)? No (E not in R₂). No subsets to remove.

**Final decomposition:**
```
R₁(A, B)       — key: {A}
R₂(B, C, D)    — key: {B, C}
R₃(D, E)       — key: {D}
R₄(E, C)       — key: {E}
R₅(A, C)       — key: {A, C} (candidate key of R)
```

All in 3NF ✓, lossless-join ✓, dependency-preserving ✓.
</details>

### Exercise 3: BCNF Decomposition

Decompose into BCNF:

R(A, B, C, D) with F = { AB → C, C → A, C → D }

<details>
<summary>Solution</summary>

Candidate keys: {A,B} and {B,C} (verify: {A,B}⁺ = {A,B,C,D}, {B,C}⁺ = {A,B,C,D}).

Check BCNF:
- AB → C: {A,B} is a superkey. ✓
- C → A: {C}⁺ = {A,C,D}. C is not a superkey. **BCNF violation!**
- C → D: Same issue. **BCNF violation!**

Decompose on C → A (or C → AD):
- {C}⁺ ∩ {A,B,C,D} = {A,C,D}
- R₁ = (A, C, D) with key {C}
- R₂ = {C} ∪ ({A,B,C,D} - {A,C,D}) = (B, C) with key {B,C}

Check R₁(A, C, D):
- C → A: C is a key of R₁. ✓
- C → D: C is a key of R₁. ✓
- BCNF ✓

Check R₂(B, C):
- No non-trivial FDs with determinant that's not a superkey.
- BCNF ✓

**BCNF decomposition: R₁(A, C, D), R₂(B, C)**

Dependency preservation: AB → C requires joining R₁ and R₂. **Not preserved.**
</details>

### Exercise 4: Lossless-Join Verification

Verify whether the following decomposition has the lossless-join property:

R(A, B, C, D) with F = { A → B, B → C }

Decomposition: R₁(A, B), R₂(A, C), R₃(B, D)

<details>
<summary>Solution</summary>

Use the chase test:

Initial matrix:
|    | A  | B   | C   | D   |
|----|----|-----|-----|-----|
| R₁ | a₁ | a₂  | b₁₃ | b₁₄ |
| R₂ | a₁ | b₂₂ | a₃  | b₂₄ |
| R₃ | b₃₁| a₂  | b₃₃ | a₄  |

Apply A → B: R₁ and R₂ agree on A (= a₁).
- R₁.B = a₂, R₂.B = b₂₂. R₁ has distinguished. Set R₂.B = a₂.

|    | A  | B  | C   | D   |
|----|----|----|-----|-----|
| R₁ | a₁ | a₂ | b₁₃ | b₁₄ |
| R₂ | a₁ | a₂ | a₃  | b₂₄ |
| R₃ | b₃₁| a₂ | b₃₃ | a₄  |

Apply B → C: R₁, R₂, R₃ agree on B (= a₂).
- C values: b₁₃, a₃, b₃₃. Has a₃. Set all to a₃.

|    | A  | B  | C  | D   |
|----|----|----|----| ----|
| R₁ | a₁ | a₂ | a₃ | b₁₄ |
| R₂ | a₁ | a₂ | a₃ | b₂₄ |
| R₃ | b₃₁| a₂ | a₃ | a₄  |

No more changes from further iterations.

Check rows: No row has all distinguished symbols. Row R₁ has b₁₄, Row R₂ has b₂₄, Row R₃ has b₃₁.

**The decomposition is NOT lossless-join.** ✗

The problem: R₃(B, D) shares only B with the others, and B is not a key of any relation containing D's determining attributes.

A correct decomposition: R₁(A, B), R₂(B, C, D) — this is lossless since B → C holds and {B} is a key of R₂ restricted to {B,C}.

Actually, wait: B → D is not given. The FDs are only A → B and B → C. So D has no determining FD. Let's reconsider: {A}⁺ = {A,B,C}. The key must include D somehow: key = {A, D}.

Better decomposition: R₁(A, B, C) and R₂(A, D). Common = {A}. A → {B,C}. {A} is a key of R₁. Lossless ✓.
</details>

### Exercise 5: Full Normalization

Normalize the following to 3NF using the synthesis algorithm:

```
Library(isbn, title, author_id, author_name, publisher_id,
        publisher_name, publisher_city, branch_id, branch_name, copies)
```

FDs:
```
isbn → title, author_id, publisher_id
author_id → author_name
publisher_id → publisher_name, publisher_city
{isbn, branch_id} → copies
branch_id → branch_name
```

<details>
<summary>Solution</summary>

**Step 1: Minimal cover**

Decompose RHS:
```
isbn → title, isbn → author_id, isbn → publisher_id,
author_id → author_name,
publisher_id → publisher_name, publisher_id → publisher_city,
{isbn, branch_id} → copies,
branch_id → branch_name
```

Already minimal (all single attributes on RHS, no extraneous LHS, no redundant FDs).

**Step 2: Group by LHS**

- R₁(isbn, title, author_id, publisher_id) — from isbn → title, author_id, publisher_id
- R₂(author_id, author_name) — from author_id → author_name
- R₃(publisher_id, publisher_name, publisher_city) — from publisher_id → publisher_name, publisher_city
- R₄(isbn, branch_id, copies) — from {isbn, branch_id} → copies
- R₅(branch_id, branch_name) — from branch_id → branch_name

**Step 3: Candidate key = {isbn, branch_id}**

R₄ contains {isbn, branch_id}. ✓

**Step 4: No subsets to remove.**

**Final 3NF decomposition:**
```
Book(isbn, title, author_id, publisher_id)         — key: {isbn}
Author(author_id, author_name)                      — key: {author_id}
Publisher(publisher_id, publisher_name, publisher_city) — key: {publisher_id}
BranchStock(isbn, branch_id, copies)                — key: {isbn, branch_id}
Branch(branch_id, branch_name)                      — key: {branch_id}
```

This is also in BCNF since every determinant is a single-attribute key (or composite key in BranchStock).
</details>

### Exercise 6: Anomaly Identification

Given the following relation and sample data, identify specific update, insertion, and deletion anomalies:

```
CourseSection(course_id, section, semester, instructor, building, room)

FDs: {course_id, section, semester} → instructor, building, room
     building, room → capacity   (assume capacity is also an attribute)
```

```
| course_id | section | semester | instructor | building | room | capacity |
|-----------|---------|----------|------------|----------|------|----------|
| CS101     | 1       | Fall24   | Dr. Smith  | Watson   | 101  | 50       |
| CS101     | 2       | Fall24   | Dr. Jones  | Watson   | 101  | 50       |
| CS201     | 1       | Fall24   | Dr. Smith  | Watson   | 201  | 30       |
| CS201     | 1       | Spr25    | Dr. Smith  | Taylor   | 105  | 40       |
```

<details>
<summary>Solution</summary>

**Update anomaly**: If the capacity of Watson 101 changes (e.g., renovation adds seats), we must update multiple rows (rows 1 and 2). If we update only row 1, rows 1 and 2 become inconsistent.

**Insertion anomaly**: We cannot record that Taylor 302 has capacity 60 unless there is a course section scheduled in that room. There's no way to store room information independently.

**Deletion anomaly**: If CS201 Section 1 Spring 2025 is cancelled (delete row 4), we lose the information that Taylor 105 has capacity 40 (assuming no other section uses that room).

**Root cause**: The transitive dependency {course_id, section, semester} → {building, room} → capacity creates redundancy.

**Fix**: Decompose into:
```
CourseSection(course_id, section, semester, instructor, building, room)
Room(building, room, capacity)
```
</details>

---

## 14. Summary

| Concept | Key Idea |
|---------|----------|
| **1NF** | Atomic values only — foundation of relational model |
| **2NF** | No partial dependencies — every non-prime attribute depends on the full key |
| **3NF** | No transitive dependencies — non-prime attributes depend only on keys |
| **BCNF** | Every determinant is a superkey — strictest FD-based form |
| **Lossless Join** | Natural join recovers original data — mandatory |
| **Dependency Preservation** | All FDs checkable without joins — desirable but sometimes sacrificed for BCNF |
| **3NF Synthesis** | Guarantees both lossless join and dependency preservation |
| **BCNF Decomposition** | Guarantees lossless join; may lose dependency preservation |

Normalization through BCNF handles all redundancy caused by functional dependencies. However, there are other types of dependencies — multivalued dependencies and join dependencies — that require higher normal forms. We explore these in the next lesson.

---

**Previous**: [05_Functional_Dependencies.md](./05_Functional_Dependencies.md) | **Next**: [07_Advanced_Normalization.md](./07_Advanced_Normalization.md)
