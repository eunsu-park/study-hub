# Conditions and Sorting

**Previous**: [CRUD Basics](./04_CRUD_Basics.md) | **Next**: [JOIN](./06_JOIN.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Write WHERE clauses using comparison operators (`=`, `<>`, `<`, `>`, `<=`, `>=`)
2. Combine conditions with logical operators (AND, OR, NOT) and apply correct precedence
3. Use BETWEEN, IN, and LIKE/ILIKE for range checks, set membership, and pattern matching
4. Handle NULL values correctly with IS NULL, IS NOT NULL, COALESCE, and NULLIF
5. Sort query results with ORDER BY including multi-column and expression-based ordering
6. Implement pagination using LIMIT, OFFSET, and the SQL-standard FETCH syntax
7. Apply DISTINCT and DISTINCT ON to eliminate duplicate rows from result sets

---

Raw data in a table is only useful when you can filter, sort, and page through it efficiently. In practice, almost every query you write will include a WHERE clause to narrow down results and an ORDER BY to present them in a meaningful sequence. These filtering and sorting skills are the bridge between storing data and extracting actionable information from it.

---

## 1. WHERE Clause Basics

The WHERE clause selects only rows that match a condition.

```sql
SELECT * FROM users WHERE condition;
UPDATE users SET ... WHERE condition;
DELETE FROM users WHERE condition;
```

---

### Theory: Sargability — Predicates an Index Can Use

A predicate is **sargable** if it can be rewritten as a key range on an indexed expression. Sargable forms:

```sql
WHERE x = 10                  -- = on indexed column
WHERE x > 10                  -- range
WHERE x BETWEEN 10 AND 20     -- compound range
WHERE x IN (1, 2, 3)          -- multiple equality probes
WHERE name LIKE 'abc%'        -- prefix LIKE — sargable!
```

Non-sargable forms — they prevent index use:

```sql
WHERE LOWER(name) = 'alice'   -- function on the indexed column
WHERE x + 1 = 10              -- arithmetic on the indexed column
WHERE name LIKE '%abc'        -- leading wildcard — no prefix to navigate to
WHERE x::text = '10'          -- cast on the indexed column
WHERE date_part('year', d) = 2026  -- function call hides the indexable form
```

#### B.1 Why functions kill index use

A B-tree on `name` stores `name` values, not `LOWER(name)`. The planner has no way to descend a B-tree on `name` to find rows where `LOWER(name) = 'alice'` — it would have to read every row to evaluate `LOWER`. The fix is either to rewrite the query or to build a **functional index**: `CREATE INDEX ON t (LOWER(name));`. Now the index stores the `LOWER` values and the predicate is sargable against this *new* index.

#### B.2 Why prefix LIKE is sargable but suffix LIKE is not

`WHERE name LIKE 'abc%'` is equivalent to `WHERE name >= 'abc' AND name < 'abd'` — a key range that B-tree handles natively. `WHERE name LIKE '%abc'` cannot be reduced to any range — the matching keys are scattered through the index. For suffix or substring matches, you need a different index type (`pg_trgm` GIN/GiST — covered in lesson 19).

## 2. Comparison Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `=` | Equal | `age = 30` |
| `<>` or `!=` | Not equal | `city <> 'Seoul'` |
| `<` | Less than | `age < 30` |
| `>` | Greater than | `age > 30` |
| `<=` | Less than or equal | `age <= 30` |
| `>=` | Greater than or equal | `age >= 30` |

```sql
-- Users with age 30
SELECT * FROM users WHERE age = 30;

-- Users not age 30
SELECT * FROM users WHERE age <> 30;
SELECT * FROM users WHERE age != 30;

-- Age between 25 and 35
SELECT * FROM users WHERE age >= 25 AND age <= 35;
```

---

## 3. Logical Operators

### AND

All conditions must be true.

```sql
-- People in Seoul in their 30s
SELECT * FROM users
WHERE city = 'Seoul' AND age >= 30 AND age < 40;
```

### OR

At least one condition must be true.

```sql
-- Users in Seoul or Busan
SELECT * FROM users
WHERE city = 'Seoul' OR city = 'Busan';
```

### NOT

Negates a condition.

```sql
-- Users not in Seoul
SELECT * FROM users WHERE NOT city = 'Seoul';
SELECT * FROM users WHERE city <> 'Seoul';  -- Same

-- Users not 30 or older
SELECT * FROM users WHERE NOT age >= 30;
SELECT * FROM users WHERE age < 30;  -- Same
```

### Operator Precedence

Processed in order: `NOT` > `AND` > `OR`. Use parentheses for clarity.

```sql
-- May not work as intended
SELECT * FROM users WHERE city = 'Seoul' OR city = 'Busan' AND age >= 30;
-- Actually: All of Seoul OR (Busan AND 30+)

-- Clear with parentheses
SELECT * FROM users WHERE (city = 'Seoul' OR city = 'Busan') AND age >= 30;
```

---

## 4. BETWEEN

Simplifies range conditions.

```sql
-- Age between 25 and 35
SELECT * FROM users WHERE age BETWEEN 25 AND 35;
-- Same as: WHERE age >= 25 AND age <= 35

-- NOT BETWEEN
SELECT * FROM users WHERE age NOT BETWEEN 25 AND 35;

-- Date range
SELECT * FROM orders
WHERE created_at BETWEEN '2024-01-01' AND '2024-01-31';
```

---

## 5. IN

Checks if value matches any in a list.

```sql
-- One of Seoul, Busan, Daejeon
SELECT * FROM users WHERE city IN ('Seoul', 'Busan', 'Daejeon');
-- Same as: WHERE city = 'Seoul' OR city = 'Busan' OR city = 'Daejeon'

-- NOT IN
SELECT * FROM users WHERE city NOT IN ('Seoul', 'Busan');

-- Can use with numbers too
SELECT * FROM users WHERE age IN (25, 30, 35);

-- With subquery
SELECT * FROM users WHERE id IN (SELECT user_id FROM orders);
```

---

## 6. LIKE - Pattern Matching

### Wildcards

| Symbol | Meaning |
|--------|---------|
| `%` | Zero or more characters |
| `_` | Exactly one character |

```sql
-- Names starting with 'Kim'
SELECT * FROM users WHERE name LIKE 'Kim%';

-- Names ending with 'su'
SELECT * FROM users WHERE name LIKE '%su';

-- Names containing 'young'
SELECT * FROM users WHERE name LIKE '%young%';

-- Exactly 3 character names
SELECT * FROM users WHERE name LIKE '___';

-- 2 character names starting with 'Kim'
SELECT * FROM users WHERE name LIKE 'Kim_';
```

### ILIKE - Case Insensitive

```sql
-- Case insensitive search (PostgreSQL specific)
SELECT * FROM users WHERE email ILIKE '%KIM%';
SELECT * FROM users WHERE email ILIKE 'kim@%';
```

### NOT LIKE

```sql
SELECT * FROM users WHERE name NOT LIKE 'Kim%';
```

### Escape

```sql
-- When searching for actual % or _
SELECT * FROM products WHERE name LIKE '%50\%%' ESCAPE '\';  -- Contains 50%
```

---

## 7. NULL Handling

NULL is an "unknown value" and cannot be compared with regular comparison operators.

### Theory: Three-Valued Logic and NULL

SQL is *three-valued*: every boolean expression evaluates to TRUE, FALSE, or NULL ("UNKNOWN"). NULL means "value not known" — and any operation involving an unknown value is itself unknown.

| Expression | Result |
|------------|--------|
| `5 = NULL` | NULL (not FALSE!) |
| `5 <> NULL` | NULL |
| `NULL = NULL` | NULL |
| `NULL AND TRUE` | NULL |
| `NULL AND FALSE` | FALSE (false absorbs) |
| `NULL OR TRUE` | TRUE (true absorbs) |
| `NULL OR FALSE` | NULL |
| `NOT NULL` | NULL |

`WHERE` keeps a row only if the predicate evaluates to TRUE — so NULL results are filtered out. This produces the surprise:

```sql
SELECT count(*) FROM users WHERE age <> 30;
-- Does NOT include users with age IS NULL!
```

To handle NULL explicitly, use `IS NULL` and `IS NOT NULL` (which always return TRUE or FALSE — never NULL), or `IS DISTINCT FROM` (a NULL-safe `<>`).

#### C.1 NULL and IN

`WHERE x IN (1, 2, NULL)` is exactly `WHERE x = 1 OR x = 2 OR x = NULL`. The last term is always NULL. So the expression is TRUE if `x = 1` or `x = 2`; otherwise NULL — and rows with NULL are filtered out. This is fine.

But `WHERE x NOT IN (1, 2, NULL)` is `WHERE x <> 1 AND x <> 2 AND x <> NULL` — that last AND can never be TRUE, so the *entire* clause is NULL or FALSE, and *no rows match*. This is the most famous SQL gotcha — always exclude NULLs from `NOT IN` lists, or use `NOT EXISTS` instead.

### IS NULL / IS NOT NULL

```sql
-- Users with NULL city
SELECT * FROM users WHERE city IS NULL;

-- Users with non-NULL city
SELECT * FROM users WHERE city IS NOT NULL;

-- Wrong example (always false)
SELECT * FROM users WHERE city = NULL;  -- Doesn't work!
```

### COALESCE - NULL Replacement

```sql
-- COALESCE returns the first non-NULL argument — essential for user-facing output
-- where NULL would display as blank or cause downstream errors in application code
SELECT name, COALESCE(city, 'Unspecified') AS city FROM users;

-- Chain multiple fallbacks: try phone first, then email, then a literal default
SELECT COALESCE(phone, email, 'No contact') AS contact FROM users;
```

### NULLIF

```sql
-- Return NULL if two values are equal
SELECT NULLIF(age, 0) FROM users;  -- NULL if age is 0

-- Prevent division by zero
SELECT total / NULLIF(count, 0) FROM stats;
```

---

## 8. ORDER BY - Sorting

### Theory: B-tree — The Default Index Type

`CREATE INDEX idx ON t(col);` (with no `USING` clause) builds a **B-tree** — specifically, a Lehman-Yao concurrent B-tree. The on-disk shape is:

```
            ┌────────────────┐
            │  Root page     │   (1 page, points to internal pages)
            └───┬─────┬──────┘
                │     │
        ┌───────┘     └────────┐
   ┌────┴──────┐         ┌─────┴──────┐
   │ Internal  │   ...   │  Internal  │     (depth = log₂(N) / log₂(branch_factor))
   └─┬───┬─────┘         └─────┬──────┘
     │   │                     │
   ┌─┴┐ ┌┴───┐              ┌──┴───┐
   │L │ │ L  │   ...        │  L   │              (Leaf pages — sorted)
   │  │ │    │              │      │
   └──┘ └────┘              └──────┘
   ←───────── linked list of leaves ─────────→
```

The leaves are in **sorted key order** and are linked left-to-right. So given a key (or range of keys), PostgreSQL navigates from root to the right leaf, then walks the leaf list scanning forward (or backward — also linked).

#### A.1 Why B-tree handles `=`, `<`, `>`, `BETWEEN`, and `ORDER BY` for free

All four operations reduce to "navigate to a leaf, then read sequentially":

- `WHERE x = 5` — descend to the first leaf containing key `5`, read until key changes.
- `WHERE x > 5` — descend to the first leaf with `x > 5`, read forward to end.
- `WHERE x BETWEEN 5 AND 10` — descend to first leaf `x ≥ 5`, stop when `x > 10`.
- `ORDER BY x` — descend to leftmost leaf, walk forward — no separate sort step.
- `ORDER BY x DESC` — descend to rightmost leaf, walk backward.

This is why one B-tree handles a wide variety of queries. Hash, GIN, GiST, BRIN — covered later — exist for cases B-tree cannot serve.

#### A.2 Multi-column B-tree and the leftmost-prefix rule

`CREATE INDEX idx ON t(a, b, c);` orders rows by `a`, then by `b` within equal `a`, then by `c` within equal `(a, b)`. This means the index is usable for:

- `WHERE a = ?`
- `WHERE a = ? AND b = ?`
- `WHERE a = ? AND b = ? AND c = ?`
- `WHERE a = ? AND b > ?`
- `WHERE a > ?` (range on first column)

But *not* (or, only inefficiently) usable for:

- `WHERE b = ?` alone — the index is sorted by `a` first, so all `a` values must be scanned.
- `WHERE c = ?` alone — same reason, deeper.

This is the **leftmost-prefix rule**. Index column order is a design choice that locks in which queries are fast.

### Basic Sorting

```sql
-- Ascending (default)
SELECT * FROM users ORDER BY age;
SELECT * FROM users ORDER BY age ASC;

-- Descending
SELECT * FROM users ORDER BY age DESC;

-- String sorting
SELECT * FROM users ORDER BY name;  -- Alphabetical
SELECT * FROM users ORDER BY name DESC;
```

### Multiple Column Sorting

```sql
-- Sort by city first, then by age
SELECT * FROM users ORDER BY city, age;

-- City ascending, age descending
SELECT * FROM users ORDER BY city ASC, age DESC;
```

### NULL Sorting Order

```sql
-- NULL last (default: NULL last in ASC)
SELECT * FROM users ORDER BY city NULLS LAST;

-- NULL first
SELECT * FROM users ORDER BY city NULLS FIRST;

-- NULL handling in DESC
SELECT * FROM users ORDER BY city DESC NULLS LAST;
```

### Sort by Expression

```sql
-- Sort by name length
SELECT * FROM users ORDER BY LENGTH(name);

-- Sort by calculated result
SELECT name, age, age * 12 AS months FROM users ORDER BY months DESC;

-- Sort by column position (1-based)
SELECT name, email, age FROM users ORDER BY 3 DESC;  -- Sort by age
```

---

## 9. LIMIT / OFFSET - Result Limiting

### LIMIT

```sql
-- Top 5 only
SELECT * FROM users LIMIT 5;

-- Top 3 oldest users
SELECT * FROM users ORDER BY age DESC LIMIT 3;
```

### OFFSET

```sql
-- Skip first 5, then continue
SELECT * FROM users ORDER BY id OFFSET 5;

-- Pagination: 5 rows starting from 6th
SELECT * FROM users ORDER BY id LIMIT 5 OFFSET 5;
```

### Pagination Calculation

```sql
-- Page 1 (rows 1-10)
SELECT * FROM users ORDER BY id LIMIT 10 OFFSET 0;

-- Page 2 (rows 11-20)
SELECT * FROM users ORDER BY id LIMIT 10 OFFSET 10;

-- Page N (calculation: OFFSET = (N-1) * page_size)
SELECT * FROM users ORDER BY id LIMIT 10 OFFSET 20;  -- Page 3
```

### FETCH (SQL Standard)

```sql
-- Same as LIMIT
SELECT * FROM users
ORDER BY age DESC
FETCH FIRST 5 ROWS ONLY;

-- With OFFSET
SELECT * FROM users
ORDER BY id
OFFSET 10 ROWS
FETCH NEXT 5 ROWS ONLY;
```

---

## 10. DISTINCT - Remove Duplicates

```sql
-- Remove duplicate cities
SELECT DISTINCT city FROM users;

-- Remove duplicates of column combinations
SELECT DISTINCT city, age FROM users;

-- With COUNT
SELECT COUNT(DISTINCT city) FROM users;
```

### DISTINCT ON (PostgreSQL Specific)

```sql
-- First user per city
SELECT DISTINCT ON (city) * FROM users ORDER BY city, created_at;

-- Oldest user per city
SELECT DISTINCT ON (city) * FROM users ORDER BY city, age DESC;
```

---

## 11. Practice Examples

### Sample Data

```sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    category VARCHAR(100),
    price NUMERIC(10, 2),
    stock INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO products (name, category, price, stock) VALUES
('MacBook Pro 14', 'Laptop', 2490000, 50),
('MacBook Air M2', 'Laptop', 1590000, 100),
('Galaxy Book Pro', 'Laptop', 1790000, 30),
('iPad Pro', 'Tablet', 1290000, 80),
('Galaxy Tab S9', 'Tablet', 1190000, 60),
('AirPods Pro', 'Earbuds', 329000, 200),
('Galaxy Buds2', 'Earbuds', 179000, 150),
('Apple Watch 9', 'Smartwatch', 599000, 70),
('Galaxy Watch6', 'Smartwatch', 399000, 90),
('iPhone 15', 'Smartphone', 1250000, 120),
('Galaxy S24', 'Smartphone', 1150000, NULL);
```

### Practice 1: Basic Conditional Searches

```sql
-- 1. Laptop category products
SELECT * FROM products WHERE category = 'Laptop';

-- 2. Products priced 1,000,000 or more
SELECT * FROM products WHERE price >= 1000000;

-- 3. Products with stock 100+
SELECT * FROM products WHERE stock >= 100;

-- 4. Laptops priced 2,000,000 or less
SELECT * FROM products
WHERE category = 'Laptop' AND price <= 2000000;
```

### Practice 2: Complex Conditions

```sql
-- 1. Laptops or tablets
SELECT * FROM products
WHERE category IN ('Laptop', 'Tablet')
ORDER BY price DESC;

-- 2. Price between 500,000-1,500,000
SELECT * FROM products
WHERE price BETWEEN 500000 AND 1500000
ORDER BY price;

-- 3. Products with 'Pro' in name
SELECT * FROM products WHERE name LIKE '%Pro%';

-- 4. Products with NULL or 0 stock
SELECT * FROM products
WHERE stock IS NULL OR stock = 0;
```

### Practice 3: Sorting and Pagination

```sql
-- 1. Top 5 most expensive products
SELECT * FROM products ORDER BY price DESC LIMIT 5;

-- 2. By category, then price (ascending)
SELECT * FROM products ORDER BY category, price;

-- 3. Page 2 (6th-10th products)
SELECT * FROM products ORDER BY id LIMIT 5 OFFSET 5;

-- 4. Most expensive product per category
SELECT DISTINCT ON (category) *
FROM products
ORDER BY category, price DESC;
```

### Practice 4: NULL Handling

```sql
-- 1. Products with no stock or NULL
SELECT name, COALESCE(stock, 0) AS stock FROM products
WHERE stock IS NULL OR stock = 0;

-- 2. Display NULL as 'Checking stock'
SELECT name, COALESCE(stock::TEXT, 'Checking stock') AS stock_status
FROM products;

-- 3. Sort with NULL last
SELECT * FROM products ORDER BY stock NULLS LAST;
```

---

## 12. Performance Tips

### Theory: Sequential, Index, and Bitmap Scans

Given a sargable predicate, the planner picks one of three access methods based on **selectivity** — the fraction of rows the predicate matches.

#### D.1 Sequential scan

Read every page of the heap from start to end, evaluate the predicate per row. Cost is proportional to table size. Wins when selectivity is high (returning > ~10% of rows) because no random I/O is needed and pages can be read in large prefetched chunks.

#### D.2 Index scan

Walk the B-tree to find matching keys, then for each key follow the heap pointer to read the row. Cost is `(matching_rows × random_page_cost) + log B-tree depth`. Wins when selectivity is low (returning < ~1% of rows). Loses badly at high selectivity because random heap reads are 4× the cost of sequential reads (`random_page_cost` defaults to 4.0, `seq_page_cost` to 1.0).

#### D.3 Bitmap scan

For middle selectivities (1%–10%), the planner uses a hybrid:

1. **Index scan** to build a **bitmap** of matching heap page numbers.
2. **Sort** the bitmap by page number.
3. **Sequential read** of the heap pages in order, applying the predicate per row.

This converts random reads to sequential reads at the cost of materializing the bitmap. For large result sets, it is dramatically faster than a pure index scan.

#### D.4 Index-only scan

If every column the query needs is in the index (a "covering index"), PostgreSQL can answer from the index alone without ever touching the heap. Requires the **visibility map** to confirm the row is visible without checking heap MVCC headers. Build with `CREATE INDEX ... INCLUDE (col1, col2)` to add non-key columns purely for this purpose.

### Use Indexes

```sql
-- Indexes turn O(n) sequential scans into O(log n) B-tree lookups.
-- Only create indexes on columns that appear in WHERE, JOIN, or ORDER BY clauses;
-- each index adds write overhead (INSERT/UPDATE must maintain the index).
CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_products_price ON products(price);

-- Composite index
CREATE INDEX idx_products_cat_price ON products(category, price);
```

### LIKE Pattern Optimization

```sql
-- Prefix patterns anchor to the start — B-tree can binary-search the sorted values
WHERE name LIKE 'MacBook%'

-- Leading wildcard requires scanning every row — consider pg_trgm GIN index
-- or full-text search if this pattern is common in your workload
WHERE name LIKE '%MacBook%'
```

### Apply LIMIT First

```sql
-- LIMIT after sorting (may be inefficient)
SELECT * FROM products ORDER BY price DESC LIMIT 10;

-- Efficient with index
CREATE INDEX idx_products_price_desc ON products(price DESC);
```

---

**Previous**: [CRUD Basics](./04_CRUD_Basics.md) | **Next**: [JOIN](./06_JOIN.md)
