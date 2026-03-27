# Lesson 07: Advanced Normalization

**Previous**: [06_Normalization.md](./06_Normalization.md) | **Next**: [08_Query_Processing.md](./08_Query_Processing.md)

---

> **Topic**: Database Theory
> **Lesson**: 7 of 16
> **Prerequisites**: Functional dependencies (Lesson 05), normalization through BCNF (Lesson 06)
> **Objective**: Understand multivalued dependencies and 4NF, join dependencies and 5NF, the theoretical ideal of DKNF, and practical denormalization strategies for real-world systems

## Learning Objectives

After completing this lesson, you will be able to:

1. Define multivalued dependencies (MVDs) and explain why a BCNF relation can still contain redundancy due to MVDs.
2. Determine whether a relation satisfies Fourth Normal Form (4NF) by detecting non-trivial multivalued dependencies.
3. Define join dependencies and explain the conditions under which a relation must be decomposed to achieve Fifth Normal Form (5NF).
4. Describe the theoretical ideal of Domain-Key Normal Form (DKNF) and explain its practical limitations.
5. Identify scenarios where controlled denormalization improves performance, and apply common denormalization patterns (derived columns, summary tables, pre-joined tables).
6. Evaluate the trade-offs between higher normal forms and real-world performance requirements when designing database schemas.

---

## 1. Introduction

In Lesson 06, we saw how functional dependencies drive normalization through BCNF. However, BCNF does not eliminate **all** forms of redundancy. There are situations where a relation in BCNF still contains redundant data caused by different kinds of constraints: **multivalued dependencies** and **join dependencies**.

This lesson covers the higher normal forms (4NF, 5NF, DKNF) that address these issues, and then pivots to the practical reality that sometimes **less normalization is better** — the art of denormalization.

### 1.1 The Hierarchy of Normal Forms

```
DKNF  (Domain-Key Normal Form — theoretical ideal)
  ↑
5NF / PJNF  (Project-Join Normal Form)
  ↑
4NF  (Fourth Normal Form)
  ↑
BCNF  (Boyce-Codd Normal Form)
  ↑
3NF  (Third Normal Form)
  ↑
2NF  (Second Normal Form)
  ↑
1NF  (First Normal Form)
```

Each level eliminates an additional class of redundancy. In practice, BCNF is sufficient for most applications. 4NF is occasionally needed. 5NF and DKNF are primarily of theoretical interest.

---

## 2. Multivalued Dependencies (MVDs)

### 2.1 Motivation

Consider a relation that tracks employees, their skills, and their assigned projects:

```
EmpSkillProject(emp_id, skill, project)
```

Suppose:
- An employee's skills are **independent** of their project assignments
- Employee E1 has skills {Java, Python} and works on projects {Alpha, Beta}

To correctly represent this, we need **all combinations**:

```
| emp_id | skill  | project |
|--------|--------|---------|
| E1     | Java   | Alpha   |
| E1     | Java   | Beta    |
| E1     | Python | Alpha   |
| E1     | Python | Beta    |
```

If we forget one row (e.g., E1/Python/Beta), the data incorrectly implies that E1's Python skill is not associated with the Beta project. This "all combinations" requirement is the hallmark of a **multivalued dependency**.

Note: This relation is in BCNF (the only candidate key is the entire set {emp_id, skill, project}, and there are no non-trivial FDs). Yet it has obvious redundancy — each skill is listed once per project, and vice versa.

### 2.2 Definition

> **Definition**: A **multivalued dependency (MVD)** X →→ Y holds on relation R if, for every pair of tuples t₁ and t₂ in R with t₁[X] = t₂[X], there exist tuples t₃ and t₄ in R such that:
>
> - t₃[X] = t₄[X] = t₁[X] = t₂[X]
> - t₃[Y] = t₁[Y] and t₃[Z] = t₂[Z]
> - t₄[Y] = t₂[Y] and t₄[Z] = t₁[Z]
>
> where Z = R - X - Y (all remaining attributes).

In simpler terms: if we fix X, the set of Y-values is independent of the set of Z-values. Every combination must appear.

### 2.3 Intuitive Understanding

X →→ Y means:

> "For a given value of X, the set of values Y takes is **independent** of the values that the remaining attributes (R - X - Y) take."

In our example:
- emp_id →→ skill (for a given employee, skills are independent of projects)
- emp_id →→ project (for a given employee, projects are independent of skills)

### 2.4 Properties of MVDs

**Complementation rule**: If X →→ Y, then X →→ Z where Z = R - X - Y.

This follows directly from the definition. In our example, emp_id →→ skill implies emp_id →→ project.

**Every FD is an MVD**: If X → Y, then X →→ Y. (But not vice versa.)

Proof: If X → Y and t₁[X] = t₂[X], then t₁[Y] = t₂[Y]. The "swap" tuples t₃ and t₄ are just t₁ and t₂ themselves.

**MVD inference rules** (analogous to Armstrong's axioms):

| Rule | Statement |
|------|-----------|
| Complementation | X →→ Y ⟹ X →→ (R - X - Y) |
| Augmentation | X →→ Y and W ⊇ Z ⟹ XW →→ YZ |
| Transitivity | X →→ Y and Y →→ Z ⟹ X →→ (Z - Y) |
| Replication | X → Y ⟹ X →→ Y |
| Coalescence | X →→ Y, Z ⊆ Y, W ∩ Y = ∅, W → Z ⟹ X → Z |

### 2.5 Trivial MVDs

> **Definition**: An MVD X →→ Y is **trivial** if:
> - Y ⊆ X, or
> - X ∪ Y = R (all attributes of the relation)

Trivial MVDs always hold and do not cause redundancy.

---

## 3. Fourth Normal Form (4NF)

### 3.1 Definition

> **Definition**: A relation schema R is in **Fourth Normal Form (4NF)** if, for every non-trivial multivalued dependency X →→ Y that holds on R:
>
> X is a superkey of R.

4NF is strictly stronger than BCNF. It eliminates redundancy caused by multivalued dependencies.

### 3.2 Relationship to BCNF

Since every FD is an MVD, if every non-trivial MVD has a superkey determinant, then every non-trivial FD also has a superkey determinant. Therefore:

```
4NF ⊂ BCNF ⊂ 3NF ⊂ 2NF ⊂ 1NF
```

A relation in 4NF is always in BCNF, but a relation in BCNF may not be in 4NF (as our EmpSkillProject example showed).

### 3.3 4NF Decomposition Algorithm

```
ALGORITHM: 4NF_Decomposition(R, D)
INPUT:  R = relation schema
        D = set of FDs and MVDs
OUTPUT: Decomposition into 4NF relations with lossless join

result ← {R}

WHILE there exists Rᵢ in result that is not in 4NF DO
    Find a non-trivial MVD X →→ Y that violates 4NF in Rᵢ
    (i.e., X is not a superkey of Rᵢ)

    Replace Rᵢ with:
        R₁ = X ∪ Y
        R₂ = Rᵢ - Y    (equivalently, X ∪ Z where Z = Rᵢ - X - Y)
END WHILE

RETURN result
```

This is analogous to the BCNF decomposition algorithm but uses MVDs instead of FDs.

### 3.4 Worked Example

**EmpSkillProject(emp_id, skill, project)**

MVDs:
- emp_id →→ skill
- emp_id →→ project

Key: {emp_id, skill, project} (the whole relation, since no FDs exist)

Check 4NF: emp_id →→ skill is non-trivial, and {emp_id} is not a superkey. **Violation!**

Decompose on emp_id →→ skill:
- R₁ = {emp_id, skill}
- R₂ = {emp_id, project}

Check R₁(emp_id, skill):
- Key: {emp_id, skill}
- Only trivial MVDs. 4NF ✓

Check R₂(emp_id, project):
- Key: {emp_id, project}
- Only trivial MVDs. 4NF ✓

**Final 4NF decomposition:**
```
EmpSkill(emp_id, skill)       — key: {emp_id, skill}
EmpProject(emp_id, project)   — key: {emp_id, project}
```

### 3.5 A More Complex Example

Consider:

```
CourseBook(course, teacher, book)
```

Suppose:
- Each course can be taught by multiple teachers
- Each course uses multiple books
- The books used for a course are the same regardless of who teaches it

MVDs: course →→ teacher, course →→ book

But suppose there's also: teacher → book (each teacher uses a specific book, which may differ by teacher). This is an FD, not an MVD. In this case, the MVD course →→ book may NOT hold, and the analysis changes.

This illustrates why identifying MVDs requires careful analysis of the real-world semantics.

---

## 4. Fifth Normal Form (5NF / PJNF)

### 4.1 Join Dependencies

> **Definition**: A **join dependency (JD)** ⋈{R₁, R₂, ..., Rₙ} holds on relation R if:
>
> R = π_{R₁}(R) ⋈ π_{R₂}(R) ⋈ ... ⋈ π_{Rₙ}(R)
>
> for every legal instance of R.

A join dependency says that R can always be losslessly decomposed into R₁, R₂, ..., Rₙ.

### 4.2 MVDs as Special Case of JDs

Every MVD X →→ Y on R is equivalent to the join dependency ⋈{XY, XZ} where Z = R - X - Y.

So MVDs are binary join dependencies (decomposition into exactly two components). General join dependencies may require decomposition into three or more components.

### 4.3 Definition of 5NF

> **Definition**: A relation schema R is in **Fifth Normal Form (5NF)**, also called **Project-Join Normal Form (PJNF)**, if for every non-trivial join dependency ⋈{R₁, R₂, ..., Rₙ} that holds on R:
>
> Every Rᵢ is a superkey of R.

5NF is the strongest normal form defined in terms of join dependencies. It eliminates all redundancy that can be detected through projection and join.

### 4.4 Example: Need for 5NF

Consider a relation about suppliers, parts, and projects:

```
SPJ(supplier, part, project)
```

Suppose the following constraint holds: "If supplier S supplies part P, and part P is used in project J, and supplier S supplies some part to project J, then supplier S supplies part P to project J."

This is a **cyclic join dependency**:

⋈{(supplier, part), (part, project), (supplier, project)}

This means SPJ = π_{supplier,part}(SPJ) ⋈ π_{part,project}(SPJ) ⋈ π_{supplier,project}(SPJ)

This JD is not implied by any MVD (it requires a three-way decomposition). If this JD holds and the entire attribute set is the only key:
- The relation may be in 4NF (no non-trivial MVDs)
- But NOT in 5NF (non-trivial JD whose components are not superkeys)

**Decomposition:**
```
SP(supplier, part)         — which suppliers supply which parts
PJ(part, project)          — which parts are used in which projects
SJ(supplier, project)      — which suppliers supply to which projects
```

### 4.5 Detecting Join Dependencies

Join dependencies are very difficult to detect in practice because:

1. They are subtle — the cyclic constraint in the SPJ example is not obvious
2. There is no simple test analogous to computing attribute closure
3. They must be identified from domain knowledge, not data inspection

This is why 5NF is primarily of theoretical interest.

---

## 5. Domain-Key Normal Form (DKNF)

### 5.1 Definition

> **Definition (Fagin, 1981)**: A relation schema R is in **Domain-Key Normal Form (DKNF)** if every constraint on R is a logical consequence of the domain constraints and key constraints of R.
>
> - **Domain constraint**: A restriction on the allowed values of an attribute (e.g., age > 0, status ∈ {'active', 'inactive'})
> - **Key constraint**: A uniqueness constraint on a set of attributes

### 5.2 Significance

DKNF is the "ultimate" normal form. If a relation is in DKNF, it has **no** redundancy that can be characterized by any known type of dependency (FD, MVD, JD, or otherwise).

However, DKNF has a significant limitation:

> **There is no general algorithm to transform a relation into DKNF.**

This makes DKNF a theoretical ideal — it tells us what perfect normalization looks like, but doesn't tell us how to get there in all cases.

### 5.3 Example

```
Employee(emp_id, emp_name, dept, salary)
```

Domain constraints:
- emp_id: positive integer
- salary: positive decimal
- dept ∈ {'Engineering', 'Marketing', 'Sales', 'HR'}

Key constraint: emp_id is the primary key.

If the only constraint is that emp_id uniquely determines all other attributes, and the domain constraints are as above, then this relation is in DKNF — every constraint follows from the key and domain constraints alone.

### 5.4 When DKNF Fails

A relation is NOT in DKNF if there is a constraint that cannot be expressed as a domain or key constraint. For example:

```
Employee(emp_id, emp_name, dept, dept_budget)
```

Constraint: dept → dept_budget (all employees in the same department see the same budget).

This is an FD that is NOT a key constraint (dept is not a key). So this relation is not in DKNF. The solution: decompose into Employee(emp_id, emp_name, dept) and Department(dept, dept_budget), where dept is a key in the Department table.

### 5.5 Practical Relevance

| Normal Form | Practical Use |
|-------------|---------------|
| 1NF - BCNF | Very common. Every database designer should know these. |
| 4NF | Occasionally needed. Independent many-to-many relationships. |
| 5NF | Rare. Cyclic constraints in specific domains. |
| DKNF | Theoretical. No practical algorithm exists. |

---

## 6. Summary of All Normal Forms

| NF | Condition | Eliminates |
|----|-----------|-----------|
| **1NF** | Atomic values | Non-relational structures |
| **2NF** | No partial FDs | Partial key dependencies |
| **3NF** | No transitive FDs (with prime attribute exception) | Transitive dependencies |
| **BCNF** | Every FD determinant is a superkey | All FD-based redundancy |
| **4NF** | Every non-trivial MVD determinant is a superkey | MVD-based redundancy |
| **5NF** | Every non-trivial JD component is a superkey | JD-based redundancy |
| **DKNF** | All constraints follow from domains + keys | All possible redundancy |

---

## 7. Denormalization

### 7.1 When to Denormalize

Normalization optimizes for **data integrity** and **storage efficiency**. But real-world systems also need **read performance**. Denormalization is the deliberate introduction of redundancy to improve query performance.

Common reasons to denormalize:

1. **Expensive joins**: Frequently executed queries that join many tables
2. **Aggregation performance**: Pre-computed counts, sums, averages
3. **Read-heavy workloads**: Systems where reads vastly outnumber writes (e.g., 100:1 ratio)
4. **Reporting and analytics**: Complex analytical queries over historical data
5. **Latency requirements**: Sub-millisecond response times

### 7.2 Denormalization Techniques

#### Technique 1: Prejoined Tables

Store data from multiple normalized tables in a single table:

```sql
-- Normalized: requires join
SELECT o.order_id, o.order_date, c.customer_name, c.email
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id;

-- Denormalized: no join needed
SELECT order_id, order_date, customer_name, customer_email
FROM orders_denormalized;
```

**Trade-off**: Faster reads, but customer_name and customer_email are duplicated across all orders.

#### Technique 2: Derived/Computed Columns

Store pre-calculated values:

```sql
-- Instead of computing total each time:
SELECT order_id, SUM(quantity * price) AS total
FROM order_items
GROUP BY order_id;

-- Store the total directly:
ALTER TABLE orders ADD COLUMN total_amount DECIMAL(10,2);

-- Update on each item change:
UPDATE orders SET total_amount = (
    SELECT SUM(quantity * price)
    FROM order_items WHERE order_items.order_id = orders.order_id
) WHERE order_id = ?;
```

#### Technique 3: Redundant Columns

Add frequently accessed foreign-table columns to avoid joins:

```sql
-- Normalized
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    order_date DATE
);

-- Denormalized: add customer_name for display purposes
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    customer_name VARCHAR(100),  -- redundant, but avoids join
    order_date DATE
);
```

#### Technique 4: Summary/Aggregate Tables

Create separate tables for pre-computed aggregates:

```sql
CREATE TABLE daily_sales_summary (
    sale_date   DATE PRIMARY KEY,
    total_orders INT,
    total_revenue DECIMAL(12,2),
    avg_order_value DECIMAL(10,2)
);

-- Populated by a nightly batch job or trigger
```

#### Technique 5: Materialized Views

Some databases support materialized views — precomputed query results stored as tables:

```sql
-- PostgreSQL
CREATE MATERIALIZED VIEW product_sales_summary AS
SELECT
    p.product_id,
    p.product_name,
    p.category,
    COUNT(oi.order_id) AS times_ordered,
    SUM(oi.quantity) AS total_quantity,
    SUM(oi.quantity * oi.price) AS total_revenue
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.product_name, p.category;

-- Refresh periodically
REFRESH MATERIALIZED VIEW product_sales_summary;
```

### 7.3 Managing Denormalized Data

Denormalization introduces the very anomalies that normalization was designed to prevent. Strategies to manage this:

**1. Application-level enforcement**

The application ensures consistency when writing:

```python
def update_customer_name(customer_id, new_name):
    # Update the normalized source
    db.execute("UPDATE customers SET name = ? WHERE id = ?",
               new_name, customer_id)

    # Update all denormalized copies
    db.execute("UPDATE orders SET customer_name = ? WHERE customer_id = ?",
               new_name, customer_id)
    db.commit()
```

**2. Database triggers**

Let the database maintain consistency automatically:

```sql
CREATE OR REPLACE FUNCTION sync_customer_name()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.name <> OLD.name THEN
        UPDATE orders
        SET customer_name = NEW.name
        WHERE customer_id = NEW.customer_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_customer_name_sync
AFTER UPDATE OF name ON customers
FOR EACH ROW EXECUTE FUNCTION sync_customer_name();
```

**3. Eventual consistency**

Accept that denormalized data may be temporarily stale:

```sql
-- Use a last_synced timestamp to track freshness
ALTER TABLE orders_denormalized ADD COLUMN last_synced TIMESTAMP;

-- Background job refreshes periodically
UPDATE orders_denormalized od
SET customer_name = c.name, last_synced = NOW()
FROM customers c
WHERE od.customer_id = c.customer_id
AND od.customer_name <> c.name;
```

### 7.4 The Normalization-Denormalization Spectrum

```
Fully Normalized (BCNF/4NF)                    Fully Denormalized
├──────────────────────────────────────────────────────┤
│                                                      │
│  ✓ No redundancy          ✗ Maximum redundancy       │
│  ✓ No anomalies           ✗ All anomalies possible   │
│  ✗ Many joins needed       ✓ No joins needed          │
│  ✗ Complex queries         ✓ Simple queries           │
│  ✓ Write-optimized         ✓ Read-optimized           │
│                                                      │
│         OLTP systems ←──────→ OLAP/Analytics         │
```

### 7.5 Decision Framework

Ask these questions before denormalizing:

1. **Is the join truly a bottleneck?** Profile first. Often, proper indexing eliminates the need for denormalization.
2. **What is the read:write ratio?** Denormalization helps most when reads dominate (>10:1).
3. **How critical is consistency?** Financial systems demand perfect consistency; recommendation engines can tolerate staleness.
4. **Can materialized views suffice?** They provide denormalization benefits without modifying the base schema.
5. **Is the team prepared to maintain it?** Denormalized schemas require discipline — every write path must update all copies.

---

## 8. Star Schema and Snowflake Schema

Data warehousing uses specific denormalized patterns optimized for analytical queries.

### 8.1 Star Schema

The star schema is the simplest data warehouse pattern. It consists of:
- One **fact table** at the center (measurements/events)
- Multiple **dimension tables** radiating outward (descriptive attributes)

```
                    ┌──────────────┐
                    │   dim_date   │
                    │──────────────│
                    │ date_key (PK)│
                    │ full_date    │
                    │ year         │
                    │ quarter      │
                    │ month        │
                    │ day_of_week  │
                    └──────┬───────┘
                           │
┌──────────────┐    ┌──────┴───────┐    ┌──────────────┐
│ dim_product  │    │  fact_sales  │    │ dim_customer │
│──────────────│    │──────────────│    │──────────────│
│ product_key  │◄───│ product_key  │    │ customer_key │
│ product_name │    │ customer_key │───►│ cust_name    │
│ category     │    │ date_key     │    │ city         │
│ subcategory  │    │ store_key    │    │ state        │
│ brand        │    │──────────────│    │ segment      │
└──────────────┘    │ quantity     │    └──────────────┘
                    │ unit_price   │
                    │ total_amount │    ┌──────────────┐
                    │ discount     │    │  dim_store   │
                    └──────┬───────┘    │──────────────│
                           │            │ store_key    │
                           └───────────►│ store_name   │
                                        │ city         │
                                        │ state        │
                                        │ region       │
                                        └──────────────┘
```

**Key characteristics:**
- Fact table has foreign keys to all dimension tables
- Dimension tables are **denormalized** (e.g., dim_product has category AND subcategory in one flat table)
- Queries typically filter on dimensions and aggregate facts

```sql
-- Typical star schema query: total sales by product category and quarter
SELECT
    d.category,
    dd.year,
    dd.quarter,
    SUM(f.total_amount) AS total_sales
FROM fact_sales f
JOIN dim_product d ON f.product_key = d.product_key
JOIN dim_date dd ON f.date_key = dd.date_key
WHERE dd.year = 2025
GROUP BY d.category, dd.year, dd.quarter
ORDER BY dd.quarter, total_sales DESC;
```

### 8.2 Snowflake Schema

The snowflake schema normalizes dimension tables into sub-dimensions:

```
                         ┌──────────────┐
                         │  dim_brand   │
                         │──────────────│
                         │ brand_key    │
                         │ brand_name   │
                         │ manufacturer │
                         └──────┬───────┘
                                │
┌──────────────┐    ┌───────────┴──┐    ┌──────────────┐
│dim_subcategory│   │ dim_product  │    │  fact_sales  │
│──────────────│    │──────────────│    │──────────────│
│ subcat_key   │◄───│ product_key  │◄───│ product_key  │
│ subcat_name  │    │ product_name │    │ customer_key │
│ category_key │    │ subcat_key   │    │ date_key     │
└──────┬───────┘    │ brand_key    │    │ quantity     │
       │            └──────────────┘    │ total_amount │
┌──────┴───────┐                        └──────────────┘
│dim_category  │
│──────────────│
│ category_key │
│ category_name│
└──────────────┘
```

**Key characteristics:**
- Dimension tables are **normalized** (3NF or BCNF)
- Reduces storage for large dimension tables
- Requires more joins for queries

### 8.3 Star vs Snowflake

| Aspect | Star Schema | Snowflake Schema |
|--------|-------------|------------------|
| **Dimension structure** | Flat (denormalized) | Normalized (2NF-BCNF) |
| **Query complexity** | Simpler (fewer joins) | More complex (more joins) |
| **Query performance** | Faster (fewer joins) | Slower (more joins) |
| **Storage** | More (redundancy in dimensions) | Less (no redundancy) |
| **ETL complexity** | Simpler | More complex |
| **Maintenance** | Updates touch more rows | Updates are isolated |
| **Industry preference** | Most common | Used when dimensions are very large |

### 8.4 Fact Table Types

| Type | Description | Example |
|------|-------------|---------|
| **Transaction** | One row per event | Each sale, click, login |
| **Periodic snapshot** | Regular intervals | Daily account balance, monthly inventory |
| **Accumulating snapshot** | Lifecycle tracking | Order: placed → shipped → delivered dates |
| **Factless fact** | Events with no measures | Student attendance (just keys) |

### 8.5 Slowly Changing Dimensions (SCD)

When dimension attributes change over time:

**Type 1: Overwrite** — Simply update the value. Loses history.

```sql
UPDATE dim_customer SET city = 'New York' WHERE customer_key = 42;
```

**Type 2: Add new row** — Create a new dimension row with effective dates.

```sql
-- Before: customer lived in Boston
| customer_key | cust_id | city    | valid_from | valid_to   | current |
|-------------|---------|---------|------------|------------|---------|
| 42          | C100    | Boston  | 2020-01-01 | 9999-12-31 | Y       |

-- After: customer moved to New York
| customer_key | cust_id | city     | valid_from | valid_to   | current |
|-------------|---------|----------|------------|------------|---------|
| 42          | C100    | Boston   | 2020-01-01 | 2025-06-30 | N       |
| 99          | C100    | New York | 2025-07-01 | 9999-12-31 | Y       |
```

**Type 3: Add new column** — Track the previous value.

```sql
| customer_key | city     | prev_city | city_change_date |
|-------------|----------|-----------|-----------------|
| 42          | New York | Boston    | 2025-07-01      |
```

---

## 9. Practical Normalization Guidelines

### 9.1 Rules of Thumb

1. **Start normalized, denormalize when proven necessary**. Premature denormalization is a common mistake.

2. **OLTP systems**: Normalize to at least 3NF, preferably BCNF. Write-heavy workloads demand consistency.

3. **OLAP/Analytics**: Use star or snowflake schemas. Read-heavy analytical workloads benefit from fewer joins.

4. **Microservices**: Each service owns its data. Normalize within a service; accept redundancy across services.

5. **NoSQL databases**: Document stores (MongoDB) often use embedded/denormalized structures. Graph databases have their own modeling patterns. The relational normalization theory applies primarily to relational databases.

### 9.2 Common Patterns

**Pattern 1: Lookup tables** — Small, rarely changing reference data.

```sql
-- Always normalize: countries, currencies, status codes
CREATE TABLE country (
    country_code CHAR(2) PRIMARY KEY,
    country_name VARCHAR(100) NOT NULL
);
```

**Pattern 2: Audit/history tables** — Denormalize intentionally.

```sql
-- Store a snapshot of the data at the time of the event
CREATE TABLE order_audit (
    audit_id       SERIAL PRIMARY KEY,
    order_id       INT,
    customer_name  VARCHAR(100),  -- snapshot, not FK
    product_name   VARCHAR(100),  -- snapshot, not FK
    total_amount   DECIMAL(10,2),
    recorded_at    TIMESTAMP DEFAULT NOW()
);
```

**Pattern 3: Cache tables** — Pre-computed for performance.

```sql
-- Normalized source: compute from orders + order_items
-- Cache: updated by triggers or batch jobs
CREATE TABLE customer_stats (
    customer_id     INT PRIMARY KEY REFERENCES customers(customer_id),
    total_orders    INT DEFAULT 0,
    total_spent     DECIMAL(12,2) DEFAULT 0,
    last_order_date DATE,
    updated_at      TIMESTAMP DEFAULT NOW()
);
```

### 9.3 Anti-Patterns to Avoid

**1. Entity-Attribute-Value (EAV)**

```sql
-- AVOID: pseudo-flexible but terrible for queries and integrity
CREATE TABLE attributes (
    entity_id   INT,
    attr_name   VARCHAR(100),
    attr_value  VARCHAR(500)
);
```

Problems: no type safety, no referential integrity, horrific query performance.

**2. Over-normalization**

Splitting a table like `Address(street, city, state, zip)` into `Address(street, zip_id)` + `ZipCode(zip_id, city, state)` is technically correct (zip → city, state) but rarely beneficial — zip code data changes extremely rarely.

**3. One True Lookup Table (OTLT)**

```sql
-- AVOID: putting all reference data in one table
CREATE TABLE lookup (
    lookup_type  VARCHAR(50),
    lookup_code  VARCHAR(50),
    lookup_value VARCHAR(200)
);
```

Problems: no foreign key integrity, no type-specific validation, confusing semantics.

---

## 10. Exercises

### Exercise 1: Identifying MVDs

Given R(student, course, hobby), where a student's courses are independent of their hobbies:

1. What MVDs hold?
2. What is the highest normal form?
3. Decompose into 4NF.

<details>
<summary>Solution</summary>

1. **MVDs**: student →→ course and student →→ hobby (by complementation)

2. **Highest NF**: The only key is {student, course, hobby}. No non-trivial FDs exist, so BCNF holds. But the MVD student →→ course has determinant {student} which is not a superkey. **BCNF but not 4NF.**

3. **4NF decomposition**:
   - R₁(student, course) — key: {student, course}
   - R₂(student, hobby) — key: {student, hobby}

   Both are in 4NF (only trivial MVDs remain). ✓
</details>

### Exercise 2: MVD vs FD

Given R(A, B, C) with the constraint: for each value of A, the set of B-values is fixed regardless of C-values.

1. Is this an FD or an MVD?
2. Does A → B hold?
3. Does A →→ B hold?

<details>
<summary>Solution</summary>

1. This is an **MVD**. The constraint says the B-values and C-values are independent given A.

2. **A → B does NOT necessarily hold.** An FD A → B would mean each A-value is associated with exactly one B-value. The constraint says A is associated with a **set** of B-values, which may have more than one element.

3. **A →→ B holds.** This is exactly the definition of a multivalued dependency: for a given A, the set of B-values is independent of C-values.

Example:
```
| A  | B  | C  |
|----|----|----|
| a1 | b1 | c1 |
| a1 | b1 | c2 |
| a1 | b2 | c1 |
| a1 | b2 | c2 |
```
A →→ B holds (every A-B combo appears with every C). A → B does not hold (a1 maps to both b1 and b2).
</details>

### Exercise 3: Star Schema Design

Design a star schema for a library lending system. The fact is a "book checkout event." Identify:
1. The fact table and its measures
2. At least 3 dimension tables
3. Sample SQL query using the schema

<details>
<summary>Solution</summary>

**Fact table: fact_checkout**
```sql
CREATE TABLE fact_checkout (
    checkout_key    SERIAL PRIMARY KEY,
    date_key        INT REFERENCES dim_date(date_key),
    book_key        INT REFERENCES dim_book(book_key),
    patron_key      INT REFERENCES dim_patron(patron_key),
    branch_key      INT REFERENCES dim_branch(branch_key),
    -- Measures
    days_borrowed   INT,
    is_returned     BOOLEAN,
    late_fee        DECIMAL(6,2)
);
```

**Dimension tables:**
```sql
CREATE TABLE dim_date (
    date_key    INT PRIMARY KEY,
    full_date   DATE,
    year        INT,
    month       INT,
    day_of_week VARCHAR(10),
    is_weekend  BOOLEAN
);

CREATE TABLE dim_book (
    book_key    INT PRIMARY KEY,
    isbn        VARCHAR(20),
    title       VARCHAR(200),
    author      VARCHAR(100),
    genre       VARCHAR(50),
    publisher   VARCHAR(100),
    pub_year    INT
);

CREATE TABLE dim_patron (
    patron_key  INT PRIMARY KEY,
    patron_id   VARCHAR(20),
    name        VARCHAR(100),
    membership  VARCHAR(20),  -- 'adult', 'student', 'senior'
    city        VARCHAR(50)
);

CREATE TABLE dim_branch (
    branch_key  INT PRIMARY KEY,
    branch_name VARCHAR(100),
    city        VARCHAR(50),
    state       VARCHAR(2)
);
```

**Sample query: most popular genres by month:**
```sql
SELECT
    dd.year,
    dd.month,
    db.genre,
    COUNT(*) AS checkouts
FROM fact_checkout f
JOIN dim_date dd ON f.date_key = dd.date_key
JOIN dim_book db ON f.book_key = db.book_key
WHERE dd.year = 2025
GROUP BY dd.year, dd.month, db.genre
ORDER BY dd.month, checkouts DESC;
```
</details>

### Exercise 4: 4NF Decomposition

Given R(A, B, C, D) with:
- A →→ B
- A → C

Decompose into 4NF.

<details>
<summary>Solution</summary>

First, identify all dependencies:
- A →→ B implies A →→ CD (complementation, since R - A - B = {C, D})
- A → C (FD, which implies A →→ C)

Key: Consider {A, B, D}⁺ or find the actual key. With A → C, we get C from A. So we need A plus enough to determine B and D. If A →→ B, then B is multi-valued, so B must be in the key. Similarly D might be in the key.

Key candidates: {A, B, D} (since A determines only C functionally, and B and D are independent of each other given A).

Check 4NF: A →→ B is non-trivial, and {A} is not a superkey. Violation!

**Decompose on A →→ B:**
- R₁(A, B) — key: {A, B}
- R₂(A, C, D) — key: {A, D} (since A → C)

Check R₁(A, B): Only trivial MVDs. 4NF ✓
Check R₂(A, C, D): A → C. Is {A} a superkey? {A}⁺ = {A, C}. No, D is not determined. Key is {A, D}. A → C is an FD where A is not a superkey — this violates BCNF!

**Decompose R₂ on A → C:**
- R₃(A, C) — key: {A}
- R₄(A, D) — key: {A, D}

**Final 4NF decomposition:**
```
R₁(A, B)    — key: {A, B}
R₃(A, C)    — key: {A}
R₄(A, D)    — key: {A, D}
```

All in 4NF ✓.
</details>

### Exercise 5: Denormalization Decision

For each scenario, would you normalize or denormalize? Explain your reasoning.

1. An e-commerce product catalog with 10M products, read:write ratio of 1000:1
2. A banking transaction system processing wire transfers
3. A social media news feed showing posts with author names and profile pictures
4. A scientific data collection system recording sensor readings every second

<details>
<summary>Solution</summary>

1. **Product catalog (10M products, 1000:1 read:write)**: **Denormalize.** The overwhelming read dominance justifies redundancy. Use a denormalized product table with category names, brand names, etc. embedded directly. Consider a materialized view for search/filter pages. Updates are infrequent and can be batch-processed.

2. **Banking wire transfers**: **Normalize (BCNF).** Financial data demands perfect consistency. Every cent must be accounted for exactly once. The performance cost of joins is acceptable — correctness is paramount. Use normalized tables with proper foreign keys and constraints.

3. **Social media news feed**: **Denormalize.** News feeds are read-heavy and latency-sensitive. Store author name and profile picture URL directly in the feed item (or use a cache layer like Redis). Accept eventual consistency — if a user changes their profile picture, it's acceptable for old posts to show the old picture temporarily.

4. **Sensor data collection**: **Hybrid.** The sensor metadata (sensor_id, location, type) should be normalized (rarely changes). The time-series readings can use a specialized pattern — either a time-series database (InfluxDB, TimescaleDB) or a denormalized wide table partitioned by time. The key constraint is write throughput, not join complexity.
</details>

### Exercise 6: Join Dependency

Given R(A, B, C) with the join dependency ⋈{(A,B), (B,C), (A,C)}:

1. Is this relation necessarily in 4NF?
2. Is it in 5NF?
3. Decompose into 5NF.

<details>
<summary>Solution</summary>

1. **4NF?** Possibly yes. The JD ⋈{(A,B), (B,C), (A,C)} is a ternary join dependency, not implied by any MVD. If there are no non-trivial MVDs, the relation is in 4NF.

2. **5NF?** No. The JD ⋈{(A,B), (B,C), (A,C)} is non-trivial (it's not implied by candidate keys). If the only key is {A, B, C} (all attributes), then {A,B}, {B,C}, and {A,C} are not superkeys. **Not in 5NF.**

3. **5NF decomposition:**
   ```
   R₁(A, B)   — key: {A, B}
   R₂(B, C)   — key: {B, C}
   R₃(A, C)   — key: {A, C}
   ```

   The JD guarantees this decomposition is lossless: R = R₁ ⋈ R₂ ⋈ R₃. Each component has only trivial JDs. All in 5NF ✓.
</details>

---

## 11. Summary

| Concept | Key Point |
|---------|-----------|
| **MVD (X →→ Y)** | For a given X, Y-values are independent of other attributes |
| **4NF** | Every non-trivial MVD determinant is a superkey |
| **Join Dependency** | R can be losslessly decomposed into multiple projections |
| **5NF (PJNF)** | Every non-trivial JD component is a superkey |
| **DKNF** | All constraints follow from domains and keys (theoretical ideal) |
| **Denormalization** | Intentional redundancy for read performance |
| **Star Schema** | Fact table + flat dimension tables (data warehousing) |
| **Snowflake Schema** | Fact table + normalized dimension tables |

The key practical takeaway: **normalize by default (BCNF), and denormalize deliberately and with discipline when performance demands it**. Document every denormalization decision and its maintenance strategy.

The next lesson covers query processing — how the database engine actually executes the queries against these schemas, and how the optimizer chooses efficient execution plans.

---

**Previous**: [06_Normalization.md](./06_Normalization.md) | **Next**: [08_Query_Processing.md](./08_Query_Processing.md)
