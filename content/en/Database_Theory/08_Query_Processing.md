# Lesson 08: Query Processing

**Previous**: [07_Advanced_Normalization.md](./07_Advanced_Normalization.md) | **Next**: [09_Indexing.md](./09_Indexing.md)

---

> **Topic**: Database Theory
> **Lesson**: 8 of 16
> **Prerequisites**: Relational algebra (Lesson 03), SQL basics, understanding of disk I/O
> **Objective**: Understand how a DBMS transforms a SQL query into an efficient execution plan, master the cost models for selection and join algorithms, and grasp query optimization techniques

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the query processing pipeline — parsing, translation, optimization, and execution — and explain the role of each stage.
2. Calculate the I/O cost of selection algorithms (linear scan, primary index, secondary index) using disk block cost models.
3. Compare the cost characteristics of join algorithms (nested loop, block nested loop, sort-merge, hash join) and select the appropriate algorithm for a given scenario.
4. Explain how a query optimizer uses statistics and cost estimation to enumerate and evaluate alternative query plans.
5. Apply algebraic equivalence rules to transform query trees and produce more efficient execution plans.
6. Interpret a query execution plan (e.g., EXPLAIN output) to diagnose performance bottlenecks in SQL queries.

---

## 1. Introduction

When you write a SQL query, the database does not simply execute it as written. Between your SQL statement and the actual disk accesses lies a sophisticated pipeline of **parsing**, **optimization**, and **execution**. Understanding this pipeline is crucial for writing efficient queries and diagnosing performance problems.

### 1.1 The Query Processing Pipeline

```
SQL Query
    │
    ▼
┌─────────────────┐
│    Parser        │ → Syntax check, parse tree
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Translator      │ → Relational algebra expression
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Optimizer       │ → Choose best execution plan
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Execution       │ → Run the plan, return results
│  Engine          │
└─────────────────┘
```

### 1.2 Example: A Simple Query's Journey

```sql
SELECT e.name, d.dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
WHERE e.salary > 80000;
```

1. **Parser**: Checks syntax, resolves table/column names, produces a parse tree
2. **Translator**: Converts to relational algebra: π_{name, dept_name}(σ_{salary > 80000}(employees ⋈_{dept_id} departments))
3. **Optimizer**: Considers many equivalent plans:
   - Filter first, then join? Or join first, then filter?
   - Use an index on salary? On dept_id?
   - Nested loop join? Hash join? Sort-merge join?
4. **Execution engine**: Executes the chosen plan using the iterator model

---

## 2. Parsing and Translation

### 2.1 Parsing

The parser performs:

1. **Lexical analysis**: Breaks the query into tokens (keywords, identifiers, operators, literals)
2. **Syntax analysis**: Verifies the query follows SQL grammar rules, builds a parse tree
3. **Semantic analysis**: Checks that tables and columns exist, types are compatible, user has permissions

**Parse tree** for our example:

```
         SELECT
        /      \
   ProjectList  FROM
   /      \      |
 e.name  d.dept_name  JoinClause
                        /    \
                  employees  departments
                       |
                  ON e.dept_id = d.dept_id
                       |
                  WHERE e.salary > 80000
```

### 2.2 Translation to Relational Algebra

The parser output is translated into an initial relational algebra expression (or an equivalent internal representation called a **query tree**):

```
π_{name, dept_name}
    │
    σ_{salary > 80000}
    │
    ⋈_{dept_id}
   / \
  e   d
```

This initial expression is **logically correct** but not necessarily **efficient**. The optimizer's job is to find an equivalent but faster plan.

---

## 3. Query Evaluation Plans and the Iterator Model

### 3.1 Query Evaluation Plan

A **query evaluation plan** (or execution plan) specifies:
- The relational algebra operations to perform
- The **algorithm** to use for each operation
- The **order** in which operations are executed
- How data flows between operations

### 3.2 The Iterator (Volcano/Pipeline) Model

Most modern databases use the **iterator model** (also called the Volcano model, after the Volcano query processing system by Goetz Graefe):

Every operator implements three methods:

```
open()   → Initialize the operator. Open child iterators, allocate buffers.
next()   → Return the next tuple in the result. Call children's next() as needed.
close()  → Clean up. Release buffers, close child iterators.
```

**Key insight**: Operators are composed into a tree. The root calls `next()`, which cascades down to the leaves (table scans). Tuples flow **upward** one at a time.

```
         π_{name, dept_name}     ← root calls next()
              │
         σ_{salary > 80000}     ← filters, passes matching tuples up
              │
         ⋈_{dept_id}            ← produces joined tuples
            / \
      Scan(e)  Scan(d)          ← read tuples from disk
```

### 3.3 Materialization vs Pipelining

**Materialization**: Each operator produces its **entire** result, stores it in a temporary relation, then passes it to the parent. Simple but requires lots of temporary storage.

**Pipelining**: Tuples flow through operators without being fully materialized. As soon as one tuple is produced, it's passed to the next operator. Much more memory-efficient.

```
Materialization:
  Scan(e) → [full temp table] → σ → [full temp table] → ⋈ → [full temp table] → π

Pipelining:
  Scan(e) → σ → ⋈ → π
  (tuple by tuple, no full temp tables)
```

Pipelining is preferred but not always possible. Some operations are **blocking** — they must consume all input before producing any output:
- **Sorting** (must see all tuples to sort)
- **Hash join build phase** (must build the entire hash table)
- **Aggregation** (must process all groups)

### 3.4 Pull vs Push Model

The iterator model described above is a **pull** model (or demand-driven): the parent pulls tuples from children by calling `next()`.

Modern systems increasingly use a **push** model (or data-driven): children push tuples to parents. This can be more cache-friendly and amenable to compilation.

```
Pull (Volcano):                    Push:
  Parent calls child.next()          Child calls parent.consume(tuple)
  Child returns one tuple            Parent processes immediately
  Parent processes                   More cache-friendly
```

Some systems (e.g., HyPer, Umbra) compile queries into tight loops that push data through operators, achieving near-hand-coded performance.

---

## 4. Cost Estimation

### 4.1 Cost Metrics

The primary costs in query processing:

| Cost Component | Symbol | Description |
|---------------|--------|-------------|
| **Disk I/O** | tT, tS | Transfer time (sequential read) and seek time |
| **CPU** | — | Comparison, hashing, computation |
| **Memory** | M | Available buffer pages |
| **Network** | — | For distributed queries |

**Disk I/O dominates** in traditional systems. For a disk with:
- Seek time (tS) ≈ 4 ms
- Transfer time per block (tT) ≈ 0.1 ms

A single random I/O costs ~4.1 ms, while a sequential read costs ~0.1 ms per block. This 40:1 ratio explains why sequential access patterns are so important.

### 4.2 Notation

| Symbol | Meaning |
|--------|---------|
| n_r | Number of tuples in relation r |
| b_r | Number of disk blocks containing tuples of r |
| l_r | Size of a tuple of r in bytes |
| f_r | Blocking factor: tuples per block = ⌊B / l_r⌋ |
| B | Block (page) size in bytes |
| V(A, r) | Number of distinct values of attribute A in r |
| M | Number of available buffer pages in memory |

Relationship: b_r = ⌈n_r / f_r⌉

### 4.3 Example Catalog Statistics

```
employees (e):
    n_e = 10,000 tuples
    l_e = 200 bytes
    B   = 4,096 bytes (4 KB pages)
    f_e = ⌊4096 / 200⌋ = 20 tuples/block
    b_e = ⌈10000 / 20⌉ = 500 blocks
    V(dept_id, e) = 50 distinct departments
    V(salary, e) = 2,000 distinct salary values

departments (d):
    n_d = 50 tuples
    l_d = 100 bytes
    f_d = ⌊4096 / 100⌋ = 40 tuples/block
    b_d = ⌈50 / 40⌉ = 2 blocks
```

---

## 5. Selection Implementation

Selection (σ) filters tuples that satisfy a predicate. The implementation strategy depends heavily on available indexes.

### 5.1 Algorithm A1: Linear Scan (Full Table Scan)

Scan every block of the relation, test each tuple against the predicate.

```
ALGORITHM: LinearScan(r, predicate)
FOR EACH block b in r DO
    FOR EACH tuple t in b DO
        IF t satisfies predicate THEN
            output t
        END IF
    END FOR
END FOR
```

**Cost**: b_r block transfers + 1 seek

For our example: 500 transfers + 1 seek = 500 × 0.1ms + 4ms = **54 ms**

**When used**: Always applicable. Used when no index exists or when selectivity is very low (most tuples qualify).

### 5.2 Algorithm A2: Binary Search

If the file is sorted on the selection attribute and the predicate is an equality:

```
ALGORITHM: BinarySearch(r, A = v)
Use binary search to find the first block containing A = v
Scan forward to find all matching tuples
```

**Cost**: ⌈log₂(b_r)⌉ seeks and transfers for the search + additional blocks for duplicate values

For equality on a key: ⌈log₂(500)⌉ = 9 block accesses = 9 × (4ms + 0.1ms) = **37 ms**

### 5.3 Algorithm A3: Primary Index, Equality on Key

If a primary B⁺-tree index exists on the selection attribute (which is a key):

```
Cost = (h_i + 1) × (tS + tT)
```

where h_i is the height of the B⁺-tree (typically 2-4).

For h_i = 3: 4 × 4.1ms = **16.4 ms** (3 index levels + 1 data block)

### 5.4 Algorithm A4: Primary Index, Equality on Non-Key

Multiple tuples may match. They are contiguous (since the file is sorted on this attribute):

```
Cost = h_i × (tS + tT) + tS + tT × b
```

where b is the number of blocks containing matching tuples.

### 5.5 Algorithm A5: Secondary Index, Equality

**On a candidate key** (at most one match):
```
Cost = (h_i + 1) × (tS + tT)
```

Same as primary index for a key attribute.

**On a non-key attribute** (multiple matches):
```
Cost = (h_i + n) × (tS + tT)
```

where n is the number of matching tuples. Each matching tuple may be in a **different block** (unlike primary index where they're contiguous), so each requires a separate seek.

This can be **very expensive** for low-selectivity predicates. If n = 500, the cost is (3 + 500) × 4.1ms = **2,062 ms** — much worse than a full table scan (54 ms)!

### 5.6 Selection with Range Predicates

For predicates like `salary > 80000`:

| Method | Cost |
|--------|------|
| Linear scan | b_r (always works) |
| Primary index (B⁺-tree) | h_i + b/2 (scan half the leaf level on average) |
| Secondary index (B⁺-tree) | h_i + leaf pages in range + matching record pointers |

### 5.7 Selection with Complex Predicates

**Conjunctive selection** (σ_{θ₁ ∧ θ₂ ∧ ... ∧ θₙ}):

1. If an index exists on one condition, use it and apply remaining conditions as filters
2. If indexes exist on multiple conditions, use **index intersection**: fetch record pointers from each index, intersect them, then retrieve matching records
3. Composite index on multiple attributes (ideal if available)

**Disjunctive selection** (σ_{θ₁ ∨ θ₂ ∨ ... ∨ θₙ}):

1. If indexes exist on ALL conditions, use **index union**: fetch pointers from each index, union them
2. If any condition lacks an index, must use linear scan (one missing index invalidates the whole approach)

### 5.8 Comparison Summary

| Algorithm | Condition | Cost (blocks) |
|-----------|-----------|---------------|
| Linear scan | Always | b_r |
| Binary search | Sorted file, equality | ⌈log₂(b_r)⌉ |
| Primary B⁺-tree, key | Index on key | h_i + 1 |
| Primary B⁺-tree, non-key | Index on non-key | h_i + matching blocks |
| Secondary B⁺-tree, key | Index on key | h_i + 1 |
| Secondary B⁺-tree, non-key | Index on non-key | h_i + n (each match = 1 seek!) |

---

## 6. Join Algorithms

Join is typically the most expensive operation in query processing. The choice of join algorithm dramatically affects performance.

### 6.1 Notation

We join relations r (outer) and s (inner):
- b_r, b_s = number of blocks
- n_r, n_s = number of tuples
- M = available memory pages

### 6.2 Algorithm J1: Nested Loop Join (NLJ)

The simplest join algorithm. For each tuple in r, scan all of s looking for matches.

```
ALGORITHM: NestedLoopJoin(r, s, θ)
FOR EACH tuple t_r IN r DO
    FOR EACH tuple t_s IN s DO
        IF (t_r, t_s) satisfies θ THEN
            output (t_r ⋈ t_s)
        END IF
    END FOR
END FOR
```

**Cost (worst case — single buffer page for each relation)**:

```
Cost = n_r × b_s + b_r   block transfers
     = n_r + b_r          seeks
```

For each of the n_r tuples in r, we scan all b_s blocks of s. Plus b_r block reads for r itself.

**Example**: Join employees (outer) with departments (inner):
- n_r = 10,000, b_s = 2, b_r = 500
- Transfers: 10,000 × 2 + 500 = 20,500
- Seeks: 10,000 + 500 = 10,500
- Time: 20,500 × 0.1ms + 10,500 × 4ms = **44,050 ms ≈ 44 seconds**

**Optimization**: Always put the **smaller** relation as the inner (s). If we swap:
- n_r = 50, b_s = 500, b_r = 2
- Transfers: 50 × 500 + 2 = 25,002
- This is worse in transfers but better in seeks.

In practice, tuple-level nested loop is rarely used. Block-level is much better.

### 6.3 Algorithm J2: Block Nested Loop Join (BNLJ)

Instead of iterating tuple-by-tuple, iterate block-by-block.

```
ALGORITHM: BlockNestedLoopJoin(r, s, θ)
FOR EACH block B_r OF r DO
    FOR EACH block B_s OF s DO
        FOR EACH tuple t_r IN B_r DO
            FOR EACH tuple t_s IN B_s DO
                IF (t_r, t_s) satisfies θ THEN
                    output (t_r ⋈ t_s)
                END IF
            END FOR
        END FOR
    END FOR
END FOR
```

**Cost**:

```
Block transfers = b_r × b_s + b_r
Seeks           = 2 × b_r
```

Each block of r is read once. For each block of r, all of s is scanned (b_s blocks). s is read b_r times.

**Example**: Same tables:
- Transfers: 500 × 2 + 500 = 1,500
- Seeks: 2 × 500 = 1,000
- Time: 1,500 × 0.1ms + 1,000 × 4ms = **4,150 ms ≈ 4.2 seconds**

A 10x improvement over tuple-level NLJ!

**Further optimization with M buffer pages**:

Use (M - 2) pages for the outer relation, 1 page for the inner, 1 page for output:

```
Block transfers = ⌈b_r / (M-2)⌉ × b_s + b_r
Seeks           = 2 × ⌈b_r / (M-2)⌉
```

With M = 52 (50 pages for outer, 1 for inner, 1 for output):
- Outer chunks: ⌈500 / 50⌉ = 10
- Transfers: 10 × 2 + 500 = 520
- Seeks: 2 × 10 = 20
- Time: 520 × 0.1ms + 20 × 4ms = **132 ms**

If the entire outer fits in memory (b_r ≤ M - 2), the cost is just **b_r + b_s** transfers and **2** seeks — a single pass!

### 6.4 Algorithm J3: Indexed Nested Loop Join

If an index exists on the join attribute of the inner relation, use it instead of scanning.

```
ALGORITHM: IndexedNestedLoopJoin(r, s, θ)
FOR EACH tuple t_r IN r DO
    Use index on s to find tuples matching t_r
    FOR EACH matching t_s DO
        output (t_r ⋈ t_s)
    END FOR
END FOR
```

**Cost**:

```
Cost = b_r + n_r × c
```

where c is the cost of a single index lookup on s (typically h_i + 1 for an equality on a key with B⁺-tree).

**Example**: Index on departments.dept_id (h_i = 2):
- c = 2 + 1 = 3 (index traversal + 1 data block)
- Cost: 500 + 10,000 × 3 = 30,500 block accesses
- But with seeks: much better than BNLJ if index is in memory

If the index is in the buffer cache (common for small indexes):
- c ≈ 1 (just the data block)
- Cost: 500 + 10,000 × 1 = 10,500 transfers

### 6.5 Algorithm J4: Sort-Merge Join

Sort both relations on the join attribute, then merge them.

```
ALGORITHM: SortMergeJoin(r, s, join_attr)
Phase 1: Sort r on join_attr (external merge sort)
Phase 2: Sort s on join_attr (external merge sort)
Phase 3: Merge
    p_r ← first tuple of sorted r
    p_s ← first tuple of sorted s
    WHILE neither relation is exhausted DO
        IF p_r[join_attr] = p_s[join_attr] THEN
            Output all matching combinations
            Advance both pointers past the equal group
        ELSE IF p_r[join_attr] < p_s[join_attr] THEN
            Advance p_r
        ELSE
            Advance p_s
        END IF
    END WHILE
```

**Cost**:

```
Sorting cost = O(b × log_M(b)) for each relation (external merge sort)
Merge cost   = b_r + b_s (single pass through both sorted relations)

Total = sort(r) + sort(s) + b_r + b_s
```

External merge sort cost for relation with b blocks and M memory pages:
- Number of runs after initial sort: ⌈b / M⌉
- Number of merge passes: ⌈log_{M-1}(⌈b/M⌉)⌉
- Each pass reads and writes all blocks: 2 × b per pass
- Total sort cost: 2 × b × (1 + ⌈log_{M-1}(⌈b/M⌉)⌉) block transfers

**Example** (M = 52):
- Sort employees: ⌈500/52⌉ = 10 runs, ⌈log₅₁(10)⌉ = 1 merge pass
  - Cost: 2 × 500 × (1 + 1) = 2,000 transfers
- Sort departments: Already fits in memory (2 blocks < 52)
  - Cost: 2 × 2 = 4 transfers
- Merge: 500 + 2 = 502 transfers
- **Total: 2,506 transfers**

**When sort-merge excels**:
- Both relations are already sorted (skip the sort phase!)
- Large relations where hash join runs out of memory
- Non-equality joins (sort-merge can handle θ-joins, while hash join cannot)

### 6.6 Algorithm J5: Hash Join

Build a hash table on the smaller relation, then probe with the larger.

```
ALGORITHM: HashJoin(r, s, join_attr)
Phase 1 (Build): Hash the smaller relation (say s) into memory
    hash_table ← {}
    FOR EACH tuple t_s IN s DO
        bucket ← hash(t_s[join_attr])
        Insert t_s into hash_table[bucket]
    END FOR

Phase 2 (Probe): Scan the larger relation, probe the hash table
    FOR EACH tuple t_r IN r DO
        bucket ← hash(t_r[join_attr])
        FOR EACH t_s IN hash_table[bucket] DO
            IF t_r[join_attr] = t_s[join_attr] THEN
                output (t_r ⋈ t_s)
            END IF
        END FOR
    END FOR
```

**Cost (if build relation fits in memory)**:

```
Cost = b_s + b_r  block transfers (read both relations once)
     = 2          seeks
```

This is optimal! We read each relation exactly once.

**Example**: departments (2 blocks) fits in memory:
- Cost: 2 + 500 = 502 transfers, 2 seeks
- Time: 502 × 0.1ms + 2 × 4ms = **58.2 ms**

**Grace Hash Join (when build doesn't fit in memory)**:

If the smaller relation doesn't fit in memory, use partitioning:

```
Phase 1 (Partition): Hash both r and s into M-1 partitions
    Each partition is written to disk

Phase 2 (Build & Probe): For each partition i:
    Load partition i of s into a hash table
    Scan partition i of r, probe the hash table
```

**Cost**:

```
Partitioning: 2 × (b_r + b_s)    transfers (read + write both)
Build & Probe: b_r + b_s          transfers (read both partitions)
Total: 3 × (b_r + b_s)           transfers
```

**Requirement**: Each partition of the smaller relation must fit in memory:
```
b_s / (M - 1) ≤ M - 2
⟹ b_s ≤ (M - 1)(M - 2) ≈ M²
```

So hash join works if the smaller relation has at most about M² blocks.

### 6.7 Cost Comparison

| Algorithm | Block Transfers | Seeks | Best When |
|-----------|:-:|:-:|-----------|
| Nested Loop | n_r × b_s + b_r | n_r + b_r | Never (worst case) |
| Block Nested Loop | ⌈b_r/(M-2)⌉ × b_s + b_r | 2⌈b_r/(M-2)⌉ | No index, small M |
| Indexed NL | b_r + n_r × c | b_r + n_r | Index on inner join attr |
| Sort-Merge | Sort cost + b_r + b_s | Many seeks | Already sorted, or θ-joins |
| Hash Join (in-mem) | b_r + b_s | 2 | Smaller relation fits in memory |
| Grace Hash Join | 3(b_r + b_s) | Moderate | Large relations, M² sufficient |

**Practical comparison for our example** (employees ⋈ departments, M = 52):

| Algorithm | Transfers | Time (approx) |
|-----------|-----------|---------------|
| Tuple NLJ | 20,500 | 44 sec |
| Block NLJ (M=52) | 520 | 132 ms |
| Sort-Merge | 2,506 | ~260 ms |
| Hash Join (in-mem) | 502 | 58 ms |

Hash join wins decisively when the smaller relation fits in memory.

---

## 7. Query Optimization

### 7.1 Overview

The optimizer transforms an initial query plan into an equivalent but more efficient one. Two main approaches:

1. **Heuristic (rule-based) optimization**: Apply transformation rules that are "almost always" beneficial
2. **Cost-based optimization**: Enumerate alternative plans, estimate cost of each, pick the cheapest

Most real systems use a combination of both.

### 7.2 Equivalence Rules for Relational Algebra

These rules allow the optimizer to transform one expression into an equivalent one:

#### Rule 1: Cascade of Selections

```
σ_{θ₁ ∧ θ₂}(r) = σ_{θ₁}(σ_{θ₂}(r))
```

A conjunction can be split into sequential selections.

#### Rule 2: Commutativity of Selection

```
σ_{θ₁}(σ_{θ₂}(r)) = σ_{θ₂}(σ_{θ₁}(r))
```

Order of selections doesn't matter.

#### Rule 3: Cascade of Projections

```
π_{L₁}(π_{L₂}(...(π_{Lₙ}(r)))) = π_{L₁}(r)
```

Only the outermost projection matters (as long as L₁ ⊆ L₂ ⊆ ... ⊆ Lₙ).

#### Rule 4: Commutativity of Join

```
r ⋈ s = s ⋈ r
```

#### Rule 5: Associativity of Join

```
(r ⋈ s) ⋈ t = r ⋈ (s ⋈ t)
```

This is critical for multi-way joins. For n tables, there are (2(n-1))! / (n-1)! different join orderings (Catalan number). For 5 tables, that's 14 orderings. For 10 tables: 4,862.

#### Rule 6: Push Selection Through Join

```
σ_{θ}(r ⋈ s) = σ_{θ}(r) ⋈ s     (if θ involves only attributes of r)
```

This is the single most important optimization: **filter early to reduce intermediate result sizes**.

#### Rule 7: Push Selection Through Set Operations

```
σ_{θ}(r ∪ s) = σ_{θ}(r) ∪ σ_{θ}(s)
σ_{θ}(r ∩ s) = σ_{θ}(r) ∩ s     (or r ∩ σ_{θ}(s))
σ_{θ}(r - s) = σ_{θ}(r) - s
```

#### Rule 8: Push Projection Through Join

```
π_{L}(r ⋈_{θ} s) = π_{L}(π_{L₁}(r) ⋈_{θ} π_{L₂}(s))
```

where L₁ = attributes of r needed in L or θ, L₂ = attributes of s needed in L or θ.

### 7.3 Heuristic Optimization

The general strategy:

1. **Decompose** conjunctive selections (Rule 1)
2. **Push selections down** as far as possible (Rule 6, 7)
3. **Push projections down** as far as possible (Rule 8)
4. **Choose join order**: put the most selective joins first
5. **Identify subtrees** that can be executed as a pipeline

#### Example: Heuristic Optimization

Original:

```
π_{e.name, d.dept_name}(σ_{e.salary > 80000 ∧ d.building = 'Watson'}(employees ⋈ departments))
```

**Step 1**: Decompose selection
```
π_{e.name, d.dept_name}(σ_{e.salary > 80000}(σ_{d.building = 'Watson'}(employees ⋈ departments)))
```

**Step 2**: Push selections down
```
π_{e.name, d.dept_name}(σ_{e.salary > 80000}(employees) ⋈ σ_{d.building = 'Watson'}(departments))
```

**Step 3**: Push projections down
```
π_{e.name, d.dept_name}(
    π_{e.name, e.dept_id}(σ_{e.salary > 80000}(employees))
    ⋈
    π_{d.dept_id, d.dept_name}(σ_{d.building = 'Watson'}(departments))
)
```

**Before and after comparison:**

```
BEFORE: Join ALL employees with ALL departments, THEN filter.
  Cost: 10,000 × 50 = 500,000 intermediate tuples

AFTER: Filter employees (say 1,000 remain) and departments (say 5 remain), THEN join.
  Cost: 1,000 × 5 = 5,000 intermediate tuples — 100x reduction!
```

### 7.4 Cost-Based Optimization

Heuristic optimization is good but not sufficient. The optimizer must estimate the **actual cost** of each plan to choose the best one.

#### Selectivity Estimation

The **selectivity** of a predicate estimates the fraction of tuples that satisfy it:

| Predicate | Estimated Selectivity |
|-----------|----------------------|
| A = v (equality) | 1 / V(A, r) |
| A > v (range, uniform distribution) | (max(A) - v) / (max(A) - min(A)) |
| A ≥ v₁ AND A ≤ v₂ | (v₂ - v₁) / (max(A) - min(A)) |
| θ₁ ∧ θ₂ (conjunction, independent) | sel(θ₁) × sel(θ₂) |
| θ₁ ∨ θ₂ (disjunction, independent) | sel(θ₁) + sel(θ₂) - sel(θ₁) × sel(θ₂) |
| NOT θ | 1 - sel(θ) |

**Example**: Estimate size of σ_{salary > 80000}(employees)

If salary ranges from 30,000 to 150,000 (uniform distribution):
```
sel = (150,000 - 80,000) / (150,000 - 30,000) = 70,000 / 120,000 ≈ 0.583
Estimated tuples = 10,000 × 0.583 ≈ 5,833
```

#### Join Size Estimation

For a natural join r ⋈ s on attribute A:

```
Estimated size = (n_r × n_s) / max(V(A, r), V(A, s))
```

**Example**: employees ⋈ departments on dept_id:
```
Size = (10,000 × 50) / max(50, 50) = 500,000 / 50 = 10,000
```

This makes sense: each employee is in one department, so the join produces one tuple per employee.

#### Histograms

The uniform distribution assumption is often inaccurate. Real databases maintain **histograms** — statistics about the distribution of values:

**Equi-width histogram**: Divide the value range into equal-width buckets, count tuples per bucket.

```
salary histogram (5 buckets):
  [30K-54K):  2,500 employees
  [54K-78K):  3,000 employees
  [78K-102K): 2,500 employees
  [102K-126K): 1,500 employees
  [126K-150K]: 500 employees
```

With this histogram, σ_{salary > 80000} would estimate:
```
(102K-80K)/(102K-78K) × 2,500 + 1,500 + 500 = (22/24) × 2,500 + 2,000 ≈ 4,292
```

Much more accurate than the uniform estimate of 5,833!

**Equi-depth (equi-height) histogram**: Each bucket has approximately the same number of tuples. Better for skewed distributions.

### 7.5 Join Ordering Optimization

For multi-way joins, the order matters enormously. Consider:

```sql
SELECT *
FROM r1 JOIN r2 ON ... JOIN r3 ON ... JOIN r4 ON ...
```

Possible orderings (for 4 tables):
1. ((r1 ⋈ r2) ⋈ r3) ⋈ r4
2. (r1 ⋈ (r2 ⋈ r3)) ⋈ r4
3. (r1 ⋈ r2) ⋈ (r3 ⋈ r4)
4. ... (many more)

The optimizer uses **dynamic programming** to find the best order:

```
ALGORITHM: FindBestJoinOrder({R₁, R₂, ..., Rₙ})

FOR each single relation Rᵢ DO
    bestPlan({Rᵢ}) ← access path for Rᵢ
END FOR

FOR size = 2 TO n DO
    FOR each subset S of size 'size' DO
        bestPlan(S) ← MIN over all ways to split S into
                       S₁ ∪ S₂ where S₁, S₂ non-empty:
                       cost(bestPlan(S₁) ⋈ bestPlan(S₂))
    END FOR
END FOR

RETURN bestPlan({R₁, R₂, ..., Rₙ})
```

This considers all possible join trees (including bushy trees, not just left-deep trees). Complexity: O(3ⁿ) — exponential, but practical for queries with up to ~15-20 tables.

For larger queries, heuristics or greedy algorithms are used instead.

### 7.6 Left-Deep vs Bushy Join Trees

```
Left-deep tree:              Bushy tree:

        ⋈                        ⋈
       / \                      / \
      ⋈   R₄                  ⋈   ⋈
     / \                      / \ / \
    ⋈   R₃                  R₁ R₂ R₃ R₄
   / \
  R₁  R₂
```

**Left-deep trees** are preferred by many optimizers because:
1. The inner relation at each join step can use pipelining (no materialization)
2. Indexed nested loop join works naturally (inner = indexed table)
3. Search space is smaller: n! orderings vs. exponentially more for bushy trees

---

## 8. Statistics and Catalog Information

### 8.1 What the Catalog Stores

The system catalog (metadata) maintains statistics for cost estimation:

```sql
-- PostgreSQL catalog tables:
pg_class     -- table/index statistics (n_r, b_r, etc.)
pg_statistic -- column-level statistics (histograms, distinct values, correlation)
pg_stats     -- human-readable view of statistics
```

Key statistics:
- **n_r** (reltuples): Number of rows in the table
- **b_r** (relpages): Number of disk pages
- **V(A, r)** (n_distinct): Number of distinct values per column
- **Histograms**: Value distribution per column
- **Correlation**: How well the physical order matches the logical order (important for range scans)
- **Most common values (MCV)**: List of the most frequent values and their frequencies
- **NULL fraction**: Fraction of NULL values per column

### 8.2 Updating Statistics

Statistics become stale as data changes. Databases provide commands to refresh them:

```sql
-- PostgreSQL
ANALYZE employees;              -- Update stats for one table
ANALYZE;                         -- Update stats for all tables
ALTER TABLE employees SET (autovacuum_analyze_threshold = 50);

-- MySQL
ANALYZE TABLE employees;

-- SQL Server
UPDATE STATISTICS employees;
```

PostgreSQL's **autovacuum** process automatically updates statistics when enough rows have changed (default: 10% of the table).

### 8.3 Impact of Stale Statistics

Stale statistics lead to **bad plans**:

```
Scenario: Table had 1,000 rows when stats were collected.
          Now has 1,000,000 rows.

Optimizer thinks: "Small table, nested loop join is fine."
Reality: "Huge table, hash join would be 1000x faster."
```

This is one of the most common causes of sudden query performance degradation in production systems.

---

## 9. Query Execution Engine Architecture

### 9.1 Components

```
┌──────────────────────────────────────────────────────────┐
│                    Query Executor                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
│  │ Plan Cache  │  │ Iterator   │  │ Expression         │ │
│  │ (prepared   │  │ Operators  │  │ Evaluator          │ │
│  │  statements)│  │            │  │ (predicates,       │ │
│  │             │  │ - SeqScan  │  │  projections,      │ │
│  │             │  │ - IdxScan  │  │  aggregations)     │ │
│  │             │  │ - NestLoop │  │                    │ │
│  │             │  │ - HashJoin │  │                    │ │
│  │             │  │ - SortMrg  │  │                    │ │
│  │             │  │ - HashAgg  │  │                    │ │
│  │             │  │ - Sort     │  │                    │ │
│  └────────────┘  └────────────┘  └────────────────────┘ │
│                         │                                 │
│              ┌──────────┴──────────┐                     │
│              │  Buffer Manager     │                     │
│              │  (page cache)       │                     │
│              └──────────┬──────────┘                     │
│                         │                                 │
│              ┌──────────┴──────────┐                     │
│              │  Storage Engine     │                     │
│              │  (disk I/O)         │                     │
│              └─────────────────────┘                     │
└──────────────────────────────────────────────────────────┘
```

### 9.2 Plan Caching

Parsing and optimization are expensive. Databases cache execution plans to avoid repeating this work:

```sql
-- PostgreSQL: prepared statements cache plans
PREPARE find_emp(int) AS
    SELECT * FROM employees WHERE emp_id = $1;

EXECUTE find_emp(42);   -- First execution: parse + optimize + execute
EXECUTE find_emp(99);   -- Subsequent: reuse cached plan
```

**Plan invalidation**: Cached plans become invalid when:
- Table structure changes (ALTER TABLE)
- Statistics are updated (ANALYZE)
- Indexes are created or dropped

### 9.3 Reading Execution Plans

Most databases provide a command to view the execution plan:

```sql
-- PostgreSQL
EXPLAIN ANALYZE
SELECT e.name, d.dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
WHERE e.salary > 80000;
```

Output (example):

```
Hash Join  (cost=1.12..25.47 rows=167 width=64) (actual time=0.05..0.31 rows=150 loops=1)
  Hash Cond: (e.dept_id = d.dept_id)
  ->  Seq Scan on employees e  (cost=0.00..22.50 rows=167 width=40) (actual time=0.01..0.15 rows=150 loops=1)
        Filter: (salary > 80000)
        Rows Removed by Filter: 850
  ->  Hash  (cost=1.05..1.05 rows=50 width=28) (actual time=0.02..0.02 rows=50 loops=1)
        Buckets: 1024  Batches: 1  Memory Usage: 12kB
        ->  Seq Scan on departments d  (cost=0.00..1.05 rows=50 width=28) (actual time=0.00..0.01 rows=50 loops=1)
Planning Time: 0.15 ms
Execution Time: 0.38 ms
```

**Reading this plan (bottom-up)**:
1. Sequential scan on departments (50 rows) → build hash table (12 KB)
2. Sequential scan on employees with filter salary > 80000 (150 of 1000 rows pass)
3. Hash join using dept_id
4. Total: 0.38 ms execution time

### 9.4 Adaptive Query Execution

Modern databases can adjust execution plans during runtime:

- **PostgreSQL**: Uses generic vs. custom plans for prepared statements. After 5 executions, it compares and may switch.
- **Oracle**: Adaptive cursor sharing — detects when a cached plan performs poorly for certain parameter values
- **Spark SQL**: Adaptive Query Execution (AQE) — re-optimizes mid-query based on actual partition sizes

---

## 10. Advanced Topics

### 10.1 Parallel Query Execution

Modern databases parallelize query execution across multiple CPU cores:

```
              Gather
             /  |  \
     Worker1  Worker2  Worker3
        |        |        |
     Scan(p1) Scan(p2) Scan(p3)  ← parallel sequential scan
```

Parallelizable operations:
- Scan (divide table into ranges)
- Filter (each worker filters its partition)
- Hash join (parallel build + parallel probe)
- Aggregation (partial aggregation per worker, then merge)
- Sort (parallel sort, then merge)

### 10.2 Columnar Execution

Traditional row-store: read entire rows, even if only a few columns are needed.

Column-store: store each column separately, read only needed columns.

```
Row store:                     Column store:
[id=1, name=Alice, sal=80K]   id:   [1, 2, 3, ...]
[id=2, name=Bob,   sal=90K]   name: [Alice, Bob, ...]
[id=3, name=Carol, sal=75K]   sal:  [80K, 90K, 75K, ...]
```

Advantages of column stores for analytics:
- Read only needed columns (less I/O)
- Better compression (similar values together)
- CPU-friendly (SIMD operations on column arrays)
- Used by: DuckDB, ClickHouse, Redshift, BigQuery

### 10.3 Just-In-Time (JIT) Compilation

Instead of interpreting the query plan (calling virtual functions for each tuple), compile it into native machine code:

```
Traditional (interpreted):
  for each tuple:
    call virtual function: evaluate predicate
    call virtual function: project columns
    call virtual function: hash for join

JIT-compiled:
  for each tuple:
    if tuple.salary > 80000:    // inlined, no virtual dispatch
      hash = tuple.dept_id % N   // inlined
      emit(tuple.name, ...)      // inlined
```

JIT compilation removes interpretation overhead, especially beneficial for complex expressions and large datasets.

PostgreSQL supports JIT compilation (using LLVM) for expression evaluation and tuple deforming.

---

## 11. Exercises

### Exercise 1: Cost Calculation

Given:
- employees: n = 10,000, b = 500, index on emp_id (B⁺-tree, height 3)
- departments: n = 200, b = 10
- Memory: M = 12 pages

Calculate the cost (in block transfers) for joining employees and departments on dept_id using:

1. Block nested loop join (employees as outer)
2. Block nested loop join (departments as outer)
3. Hash join (departments as build)

<details>
<summary>Solution</summary>

1. **BNLJ (employees outer)**:
   - Outer chunks: ⌈500 / (12-2)⌉ = ⌈500/10⌉ = 50
   - Cost: 50 × 10 + 500 = 1,000 transfers

2. **BNLJ (departments outer)**:
   - Outer chunks: ⌈10 / (12-2)⌉ = ⌈10/10⌉ = 1
   - Cost: 1 × 500 + 10 = 510 transfers

3. **Hash join (departments as build)**:
   - departments (10 blocks) fits in 12-page memory
   - Cost: 10 + 500 = 510 transfers

The hash join and departments-outer BNLJ are comparable. Hash join has fewer seeks (2 vs. 2). In practice, hash join is preferred due to better cache behavior.
</details>

### Exercise 2: Selectivity Estimation

Given: employees table with 10,000 rows.
- salary: min=30,000, max=150,000, V(salary) = 2,000
- dept_id: V(dept_id) = 50
- city: V(city) = 100

Estimate the number of tuples returned by:

1. σ_{salary = 75000}(employees)
2. σ_{salary > 100000}(employees)
3. σ_{dept_id = 5 ∧ city = 'Boston'}(employees)

<details>
<summary>Solution</summary>

1. **salary = 75000**: sel = 1/V(salary) = 1/2000. Result: 10,000/2,000 = **5 tuples**

2. **salary > 100000**: sel = (150,000 - 100,000)/(150,000 - 30,000) = 50,000/120,000 ≈ 0.417. Result: 10,000 × 0.417 ≈ **4,167 tuples**

3. **dept_id = 5 AND city = 'Boston'** (assuming independence):
   - sel(dept_id = 5) = 1/50
   - sel(city = 'Boston') = 1/100
   - Combined: (1/50) × (1/100) = 1/5,000
   - Result: 10,000 / 5,000 = **2 tuples**
</details>

### Exercise 3: Heuristic Optimization

Optimize the following query tree using heuristic rules:

```sql
SELECT p.product_name, c.category_name
FROM products p, categories c, order_items oi
WHERE p.category_id = c.category_id
  AND p.product_id = oi.product_id
  AND oi.quantity > 10
  AND c.category_name = 'Electronics';
```

Draw the initial and optimized query trees.

<details>
<summary>Solution</summary>

**Initial (unoptimized) tree:**

```
π_{product_name, category_name}
    │
σ_{p.cat_id=c.cat_id ∧ p.prod_id=oi.prod_id ∧ oi.qty>10 ∧ c.cat_name='Electronics'}
    │
    ×  (Cartesian product)
   / \
  ×   oi
 / \
p   c
```

**Optimized tree (push selections down, use joins instead of Cartesian product):**

```
π_{product_name, category_name}
    │
    ⋈_{p.cat_id = c.cat_id}
   / \
  ⋈_{p.prod_id = oi.prod_id}    σ_{cat_name='Electronics'}(c)
 / \
p   σ_{qty > 10}(oi)
```

**Optimizations applied:**
1. Decomposed conjunctive selection
2. Pushed σ_{qty > 10} to order_items (before join)
3. Pushed σ_{cat_name = 'Electronics'} to categories (before join)
4. Replaced Cartesian products with targeted joins
5. Projected early (not shown for clarity, but only needed columns pass through)

The key gain: categories filtered to ~1 row ('Electronics'), order_items filtered to a subset (qty > 10), before any joins occur.
</details>

### Exercise 4: Join Algorithm Selection

For each scenario, which join algorithm would the optimizer likely choose?

1. Joining a 100-row lookup table with a 10M-row fact table. Index exists on the fact table's join column.
2. Joining two 1M-row tables, neither sorted, plenty of memory (1GB buffer pool).
3. Joining two 1M-row tables on a range condition (r.date BETWEEN s.start_date AND s.end_date).
4. Joining two tables where both are already sorted on the join column.

<details>
<summary>Solution</summary>

1. **Indexed nested loop join.** The lookup table (100 rows) is the outer; for each row, use the index on the fact table to find matches. Cost: 100 index lookups, each O(log n). Much faster than scanning 10M rows.

2. **Hash join.** With plenty of memory, one table's hash table fits entirely in memory. Cost: read both tables once (optimal). No index needed, no sorting needed.

3. **Sort-merge join** or **Block nested loop join.** Hash join doesn't work for range conditions (can't hash ranges). Sort-merge on the date columns allows efficient range matching. Block NLJ is a fallback if sorting is too expensive.

4. **Sort-merge join (skip the sort phase).** Both tables are already sorted, so the merge phase costs just b_r + b_s — a single pass through each table. This is optimal.
</details>

### Exercise 5: Reading Execution Plans

Given this PostgreSQL EXPLAIN output, answer the questions below:

```
Nested Loop  (cost=0.29..8.33 rows=1 width=64)
  ->  Index Scan using idx_emp_id on employees  (cost=0.29..4.30 rows=1 width=40)
        Index Cond: (emp_id = 42)
  ->  Seq Scan on departments  (cost=0.00..1.62 rows=1 width=24)
        Filter: (dept_id = employees.dept_id)
        Rows Removed by Filter: 49
```

1. What join algorithm is used?
2. Which table is the outer (driving) table?
3. Why does the optimizer use an index scan on employees?
4. Why is a sequential scan used on departments?
5. What is the estimated total cost?

<details>
<summary>Solution</summary>

1. **Nested Loop Join.**

2. **employees is the outer table** (listed first under Nested Loop). It drives the loop.

3. **Because emp_id = 42 is a highly selective equality predicate.** The index on emp_id finds exactly 1 row (rows=1). Reading the entire employees table would be wasteful.

4. **departments is small (50 rows, ~2 pages).** For each outer row (just 1 in this case), the entire departments table is scanned. Since there's only 1 outer row, the sequential scan runs only once. An index lookup might not be faster for a single probe on a tiny table.

5. **Total estimated cost: 8.33** (in PostgreSQL's cost units, where 1.0 ≈ a sequential page read). This is very cheap — essentially 1 index lookup + 1 small table scan.
</details>

### Exercise 6: Equivalence Rules

Using equivalence rules, show that these two expressions produce the same result:

**Expression A:**
```
σ_{dept='CS'}(employees ⋈ departments)
```

**Expression B:**
```
employees ⋈ σ_{dept='CS'}(departments)
```

Which is more efficient and why?

<details>
<summary>Solution</summary>

**Proof of equivalence:**

By Rule 6 (Push selection through join), if the predicate dept='CS' involves only attributes of departments:

```
σ_{dept='CS'}(employees ⋈ departments) = employees ⋈ σ_{dept='CS'}(departments)
```

This is valid because:
1. The join produces all matching (employee, department) pairs
2. The selection then filters to dept='CS'
3. Equivalently, we can filter departments first to get only the CS department, then join

**Expression B is more efficient** because:
- Expression A: Join ALL employees with ALL departments (10,000 × 50 combinations to evaluate), then filter. The join produces 10,000 rows, then the filter keeps only ~200 (if 1/50 are in CS).
- Expression B: Filter departments first (50 → 1 row), then join. The join only needs to match employees against 1 department row. Much less work.

The size of intermediate results:
- A: 10,000 intermediate rows → filter → 200 final rows
- B: 1 intermediate row × employees → 200 final rows directly
</details>

### Exercise 7: Cost-Based Optimization

Given three tables and their statistics:

```
orders (o):     n = 100,000,  b = 5,000
customers (c):  n = 10,000,   b = 500
products (p):   n = 1,000,    b = 50
```

Join predicates: o.cust_id = c.cust_id AND o.prod_id = p.prod_id

Assume hash join and M = 100 pages. Compare these two join orderings:

**Plan A**: (orders ⋈ customers) ⋈ products
**Plan B**: (orders ⋈ products) ⋈ customers

<details>
<summary>Solution</summary>

**Plan A: (orders ⋈ customers) ⋈ products**

Step 1: orders ⋈ customers (hash join, customers as build)
- Build: 500 blocks (customers fits in 100 pages? No, 500 > 100. Need Grace hash join.)
- Grace hash join cost: 3 × (5,000 + 500) = 16,500 transfers
- Result size: 100,000 rows (each order has one customer)
- Result blocks: ~5,000 (similar to orders)

Step 2: result ⋈ products (hash join, products as build)
- Build: 50 blocks (products fits in 100 pages. In-memory hash join.)
- Cost: 5,000 + 50 = 5,050 transfers
- Total for Plan A: 16,500 + 5,050 = **21,550 transfers**

**Plan B: (orders ⋈ products) ⋈ customers**

Step 1: orders ⋈ products (hash join, products as build)
- Build: 50 blocks (products fits in memory!)
- In-memory hash join cost: 5,000 + 50 = 5,050 transfers
- Result size: 100,000 rows (each order has one product)
- Result blocks: ~5,000

Step 2: result ⋈ customers (hash join, customers as build)
- Build: 500 blocks (doesn't fit in 100 pages. Grace hash join.)
- Cost: 3 × (5,000 + 500) = 16,500 transfers
- Total for Plan B: 5,050 + 16,500 = **21,550 transfers**

Interestingly, the total transfer count is the same! But Plan B is slightly better because:
1. Step 1 uses in-memory hash join (fewer seeks, better cache)
2. The intermediate result of Step 1 might be pipelined into Step 2

A smarter approach: Build hash tables on BOTH small tables (products: 50, customers: 500), then scan orders once:

**Plan C**: Scan orders once, probe both hash tables
- Cost: 5,000 + 500 + 50 = 5,550 transfers (if both hash tables fit in memory — they need 550 pages, which exceeds M=100)

With M=600, Plan C would be optimal.
</details>

---

## 12. Summary

| Concept | Key Point |
|---------|-----------|
| **Query Processing Pipeline** | Parse → Optimize → Execute |
| **Iterator Model** | open/next/close interface; tuples flow upward through operator tree |
| **Pipelining** | Avoids materializing intermediate results |
| **Selection algorithms** | Linear scan, binary search, index scan; choice depends on selectivity |
| **Join algorithms** | NLJ, Block NLJ, Indexed NLJ, Sort-Merge, Hash Join |
| **Hash join** | Optimal when build relation fits in memory: cost = b_r + b_s |
| **Sort-merge join** | Best for pre-sorted data and range joins |
| **Heuristic optimization** | Push selections down, push projections down, reorder joins |
| **Cost-based optimization** | Use statistics to estimate cost; dynamic programming for join ordering |
| **Statistics** | Histograms, distinct values, correlation — essential for good plans |
| **Plan caching** | Avoid repeated parsing/optimization for prepared statements |

Query processing is where database theory meets systems engineering. Understanding these concepts helps you write better queries, create appropriate indexes (covered in the next lesson), and diagnose performance problems by reading execution plans.

---

**Previous**: [07_Advanced_Normalization.md](./07_Advanced_Normalization.md) | **Next**: [09_Indexing.md](./09_Indexing.md)
