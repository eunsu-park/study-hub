# JOIN

**Previous**: [Conditions and Sorting](./05_Conditions_and_Sorting.md) | **Next**: [Aggregation and Grouping](./07_Aggregation_and_Grouping.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the purpose of JOIN and how it connects rows from two or more tables
2. Distinguish among INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL JOIN, and CROSS JOIN
3. Write multi-table JOIN queries using table aliases and explicit ON conditions
4. Apply SELF JOIN to model hierarchical relationships (e.g., employee-manager)
5. Identify the difference between filtering in the ON clause versus the WHERE clause for outer joins
6. Use the USING clause and understand when NATURAL JOIN is appropriate (and when it is not)
7. Create indexes on foreign key columns to optimize JOIN performance

---

Real-world data is rarely stored in a single table. Customers, orders, and products each live in their own table, and the power of a relational database comes from connecting them on the fly. JOIN is the SQL mechanism that reassembles related data from multiple tables into a single, meaningful result set -- making it one of the most important operations you will use every day.

---

## 1. JOIN Concept

JOIN is a method to connect two or more tables to query data.

```
┌─────────────────┐     ┌─────────────────┐
│     users       │     │     orders      │
├─────────────────┤     ├─────────────────┤
│ id │ name       │     │ id │ user_id    │
├────┼────────────┤     ├────┼────────────┤
│ 1  │ John Kim   │◄────│ 1  │ 1          │
│ 2  │ Jane Lee   │◄────│ 2  │ 1          │
│ 3  │ Mike Park  │     │ 3  │ 2          │
└────┴────────────┘     └────┴────────────┘
         ↑ users.id = orders.user_id
```

---

## 2. Practice Table Setup

```sql
-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255)
);

-- Orders table
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    product_name VARCHAR(200),
    amount NUMERIC(10, 2),
    order_date DATE DEFAULT CURRENT_DATE
);

-- Sample data
INSERT INTO users (name, email) VALUES
('John Kim', 'kim@email.com'),
('Jane Lee', 'lee@email.com'),
('Mike Park', 'park@email.com'),
('Sarah Choi', 'choi@email.com');  -- User with no orders

INSERT INTO orders (user_id, product_name, amount) VALUES
(1, 'Laptop', 1500000),
(1, 'Mouse', 50000),
(2, 'Keyboard', 100000),
(2, 'Monitor', 300000),
(3, 'Headset', 150000),
(NULL, 'Gift Set', 80000);  -- Order without user
```

---

> **Analogy -- Venn Diagrams for JOINs**: Think of each table as a circle in a Venn diagram. An INNER JOIN returns the overlapping region -- rows that match in both tables. A LEFT JOIN returns the entire left circle plus the overlap, filling in NULL where the right circle has no match. Understanding JOINs as set operations makes it easy to predict which rows appear in your result.

## 3. INNER JOIN

Returns only data that matches in both tables.

```sql
-- Basic syntax
SELECT columns
FROM table1
INNER JOIN table2 ON table1.column = table2.column;

-- Query user and order information
SELECT
    users.name,
    users.email,
    orders.product_name,
    orders.amount
FROM users
INNER JOIN orders ON users.id = orders.user_id;
```

Result:
```
  name    │      email       │ product_name │  amount
──────────┼──────────────────┼──────────────┼──────────
 John Kim │ kim@email.com    │ Laptop       │ 1500000
 John Kim │ kim@email.com    │ Mouse        │   50000
 Jane Lee │ lee@email.com    │ Keyboard     │  100000
 Jane Lee │ lee@email.com    │ Monitor      │  300000
 Mike Park│ park@email.com   │ Headset      │  150000
```

### Theory: Nested Loop Join

The simplest join algorithm. For each row in the **outer** table, scan the **inner** table for matches:

```
for each row r1 in outer:
    for each row r2 in inner:
        if r1.key = r2.key:
            emit (r1, r2)
```

Cost: O(N · M) where N and M are the row counts. Looks terrible — and is, if both sides are big. But two factors save it.

#### A.1 The indexed inner side

If the inner table has an index on the join key, the inner loop is *not* a full scan — it is an O(log M) index probe. Total cost becomes O(N · log M), which beats hashing for small N.

```sql
SELECT * FROM small_table s JOIN big_table b ON s.id = b.s_id;
-- Plan: Nested Loop
--         Outer: Seq Scan on small_table  (10 rows)
--         Inner: Index Scan on big_table.s_id  (1 probe per outer row)
```

This is the typical "small table joined to indexed big table" pattern, and it is often the fastest join in PostgreSQL despite the bad-sounding name.

#### A.2 When Nested Loop loses

When the outer side has many rows and there is no usable index on the inner side. The planner switches to Hash Join in that case.

### Use Table Aliases

```sql
SELECT u.name, o.product_name, o.amount
FROM users u
INNER JOIN orders o ON u.id = o.user_id;
```

### JOIN Implies INNER JOIN

```sql
-- INNER can be omitted
SELECT u.name, o.product_name
FROM users u
JOIN orders o ON u.id = o.user_id;
```

---

## 4. LEFT (OUTER) JOIN

Returns all rows from left table + matching rows from right table.
Unmatched rows are filled with NULL.

```sql
SELECT
    u.name,
    o.product_name,
    o.amount
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;
```

Result:
```
   name     │ product_name │  amount
────────────┼──────────────┼──────────
 John Kim   │ Laptop       │ 1500000
 John Kim   │ Mouse        │   50000
 Jane Lee   │ Keyboard     │  100000
 Jane Lee   │ Monitor      │  300000
 Mike Park  │ Headset      │  150000
 Sarah Choi │ NULL         │ NULL      ← User with no orders included
```

### Theory: Semi-join, Anti-join, and OUTER joins

The three algorithms above are about *how* to match rows. They are independent of *what* to do with matches and non-matches, which is determined by the join *type*.

#### D.1 INNER JOIN

Emit only rows that match on both sides. The default.

#### D.2 LEFT OUTER JOIN

Emit all rows from the left side; for left rows with no matching right row, emit `NULL`s for the right columns. Implementation: same as INNER, but track which left rows have produced output. At the end, emit any unmatched left rows with NULL right columns.

#### D.3 Semi-join — `EXISTS` and `IN`

`SELECT * FROM A WHERE EXISTS (SELECT 1 FROM B WHERE B.x = A.x);` is a **semi-join**: emit each row from A *at most once*, regardless of how many B rows match. Crucially, the right side is *probed*, not joined — duplicates on the right side do not duplicate the output.

The planner can implement semi-join as a hash semi-join (build hash on B, probe with A, stop on first match per outer row) or as a nested loop with a `LIMIT 1` on the inner. Much cheaper than INNER JOIN + DISTINCT.

#### D.4 Anti-join — `NOT EXISTS`

`SELECT * FROM A WHERE NOT EXISTS (SELECT 1 FROM B WHERE B.x = A.x);` is an **anti-join**: emit each row from A *only if no* B row matches. Same engine, opposite emission rule. This is what you should use instead of `NOT IN (subquery)` — anti-join handles NULLs correctly while `NOT IN` does not (lesson 5 §C.1).

#### D.5 FULL OUTER JOIN

The symmetric LEFT — emit unmatched rows from both sides. Implemented by running the chosen algorithm twice (once each direction) or by carefully tracking matched-on-both-sides during a single hash/merge pass.

### Find Users Without Orders

```sql
SELECT u.name, u.email
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.id IS NULL;
```

---

## 5. RIGHT (OUTER) JOIN

Returns all rows from right table + matching rows from left table.

```sql
SELECT
    u.name,
    o.product_name,
    o.amount
FROM users u
RIGHT JOIN orders o ON u.id = o.user_id;
```

Result:
```
   name    │ product_name │  amount
───────────┼──────────────┼──────────
 John Kim  │ Laptop       │ 1500000
 John Kim  │ Mouse        │   50000
 Jane Lee  │ Keyboard     │  100000
 Jane Lee  │ Monitor      │  300000
 Mike Park │ Headset      │  150000
 NULL      │ Gift Set     │   80000   ← Order without user included
```

---

## 6. FULL (OUTER) JOIN

Returns all rows from both tables. Unmatched rows are filled with NULL.

```sql
SELECT
    u.name,
    o.product_name,
    o.amount
FROM users u
FULL JOIN orders o ON u.id = o.user_id;
```

Result:
```
   name     │ product_name │  amount
────────────┼──────────────┼──────────
 John Kim   │ Laptop       │ 1500000
 John Kim   │ Mouse        │   50000
 Jane Lee   │ Keyboard     │  100000
 Jane Lee   │ Monitor      │  300000
 Mike Park  │ Headset      │  150000
 Sarah Choi │ NULL         │ NULL      ← User without orders
 NULL       │ Gift Set     │   80000   ← Order without user
```

---

## 7. CROSS JOIN

Returns all possible combinations (Cartesian product).

```sql
-- Color and size tables
CREATE TABLE colors (name VARCHAR(20));
CREATE TABLE sizes (name VARCHAR(10));

INSERT INTO colors VALUES ('Red'), ('Blue'), ('Black');
INSERT INTO sizes VALUES ('S'), ('M'), ('L');

-- All combinations
SELECT c.name AS color, s.name AS size
FROM colors c
CROSS JOIN sizes s;
```

Result:
```
 color │ size
───────┼──────
 Red   │ S
 Red   │ M
 Red   │ L
 Blue  │ S
 Blue  │ M
 Blue  │ L
 Black │ S
 Black │ M
 Black │ L
```

---

## 8. SELF JOIN

Joins a table with itself.

```sql
-- Employee-Manager relationship
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    manager_id INTEGER REFERENCES employees(id)
);

INSERT INTO employees (name, manager_id) VALUES
('CEO', NULL),
('VP', 1),
('Manager A', 2),
('Manager B', 2),
('Employee', 3);

-- Query employee and manager names
SELECT
    e.name AS employee,
    m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;
```

Result:
```
  employee  │ manager
────────────┼─────────
 CEO        │ NULL
 VP         │ CEO
 Manager A  │ VP
 Manager B  │ VP
 Employee   │ Manager A
```

---

## 9. Multiple Table JOIN

Connect 3 or more tables.

```sql
-- Add category table
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100)
);

-- Products table
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    category_id INTEGER REFERENCES categories(id),
    name VARCHAR(200),
    price NUMERIC(10, 2)
);

-- Order items table
CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER
);

-- JOIN 3 tables
SELECT
    u.name AS user_name,
    p.name AS product_name,
    c.name AS category_name,
    oi.quantity
FROM users u
JOIN orders o ON u.id = o.user_id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id
JOIN categories c ON p.category_id = c.id;
```

---

### Theory: Hash Join

Build phase: read the smaller table once and insert every row into an in-memory hash table keyed by the join column. Probe phase: read the larger table once; for each row, hash the join column and look up matches in the hash table.

```
build:
    H = {}
    for each row r in smaller_table:
        H[hash(r.key)].append(r)

probe:
    for each row s in larger_table:
        for r in H[hash(s.key)]:
            if r.key == s.key:        # hash collision check
                emit (r, s)
```

Cost: O(N + M) — linear in both sides. Wins when both sides are large and you do not care about output order.

#### B.1 Memory and `work_mem`

The hash table must fit in `work_mem`. If it does not, PostgreSQL falls back to a **partitioned hash join**: it partitions both inputs by hash buckets to disk, then hash-joins each partition pair separately. This costs an extra read+write pass per side. Watching `EXPLAIN ANALYZE` for `Batches: N` greater than 1 means a partitioned hash — and bumping `work_mem` for that session might give a large speedup.

#### B.2 Why "smaller side as build"

The hash table has constant-size overhead per row plus the row itself. Building it on the smaller side keeps memory pressure down. The planner figures this out from row-count estimates (which is why bad statistics can lead to a build-on-the-wrong-side disaster).

#### B.3 Hash collisions

Two distinct keys can hash to the same bucket. The probe always rechecks the actual key after the bucket lookup, so correctness is preserved — but heavy collisions degrade the linear cost toward quadratic in the worst case. PostgreSQL uses a 32-bit hash with extendible hashing to keep buckets balanced.

### Theory: Merge Join

If both inputs are *sorted* on the join key, you can join them in a single linear pass like merging two sorted lists:

```
i, j = 0, 0
while i < |A| and j < |B|:
    if A[i].key < B[j].key: i += 1
    elif A[i].key > B[j].key: j += 1
    else:
        emit (A[i], B[j])
        # advance both, plus handle duplicates within either side
        ...
```

Cost: O(N + M) for the merge itself, plus O(N log N + M log M) if the inputs need to be sorted first.

#### C.1 When Merge Join wins

- Both sides are *already* sorted (e.g., reading from a B-tree index in key order). The sort cost vanishes; merge wins outright.
- The output needs to be sorted on the join key anyway (subsequent `ORDER BY`).
- Memory is tight: merge join only needs a buffer for one row from each side, while hash join may not fit in `work_mem`.

#### C.2 When Merge Join loses

- Inputs are not pre-sorted and the tables are large enough that the sort is expensive.
- The join key has very low cardinality — handling many equal keys requires a "rescan" of the inner side, which complicates the linear bound.

## 10. JOIN Conditions and WHERE

### ON vs WHERE

```sql
-- ON: Table join condition
-- WHERE: Result filtering

-- LEFT JOIN + WHERE
SELECT u.name, o.product_name, o.amount
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.amount > 100000;  -- NULL rows removed

-- LEFT JOIN + Additional condition in ON
SELECT u.name, o.product_name, o.amount
FROM users u
LEFT JOIN orders o ON u.id = o.user_id AND o.amount > 100000;
-- All users retained, only matching orders connected
```

### Composite JOIN Conditions

```sql
SELECT *
FROM table1 t1
JOIN table2 t2 ON t1.col1 = t2.col1 AND t1.col2 = t2.col2;
```

---

## 11. USING Clause

Simplifies joins when column names are the same.

```sql
-- Using ON
SELECT * FROM orders o
JOIN users u ON o.user_id = u.id;

-- Using USING (when column names match)
-- If orders.user_id and users.user_id are the same:
SELECT * FROM orders
JOIN users USING (user_id);
```

---

## 12. NATURAL JOIN

Automatically joins on all columns with the same name. (Not recommended)

```sql
-- Joins on all columns with same name
SELECT * FROM orders
NATURAL JOIN users;

-- May produce unintended results, explicit ON recommended
```

---

## 13. JOIN Visualization

```
INNER JOIN:         LEFT JOIN:          RIGHT JOIN:         FULL JOIN:
    ┌───┐              ┌───┐              ┌───┐              ┌───┐
   ┌┼───┼┐            ┌┼───┼┐            ┌┼───┼┐            ┌┼───┼┐
  ┌┼│███│┼┐          ┌┼│███│ │          │ │███│┼┐          ┌┼│███│┼┐
  │ │███│ │          ││████│ │          │ │████││          ││█████││
  └┼│███│┼┘          └┼│███│ │          │ │███│┼┘          └┼│███│┼┘
   └┼───┼┘            └┼───┘ │          │ └───┼┘            └─────┼┘
    └───┘              └─────┘          └─────┘              └─────┘
   A ∩ B               All A            All B              A ∪ B
```

---

## 14. Practice Examples

### Practice 1: Basic JOIN

```sql
-- 1. Users who have ordered and their order info
SELECT u.name, o.product_name, o.amount, o.order_date
FROM users u
INNER JOIN orders o ON u.id = o.user_id
ORDER BY o.order_date DESC;

-- 2. Total order amount per user
SELECT u.name, SUM(o.amount) AS total_amount
FROM users u
INNER JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name
ORDER BY total_amount DESC;
```

### Practice 2: OUTER JOIN

```sql
-- 1. All users (regardless of orders)
SELECT
    u.name,
    COALESCE(SUM(o.amount), 0) AS total_amount,
    COUNT(o.id) AS order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name
ORDER BY total_amount DESC;

-- 2. Find users who haven't ordered
SELECT u.name, u.email
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.id IS NULL;

-- 3. Find orders without users
SELECT o.id, o.product_name, o.amount
FROM users u
RIGHT JOIN orders o ON u.id = o.user_id
WHERE u.id IS NULL;
```

### Practice 3: Complex Condition JOIN

```sql
-- 1. Users who ordered 1,000,000 or more
SELECT DISTINCT u.name, u.email
FROM users u
INNER JOIN orders o ON u.id = o.user_id
WHERE o.amount >= 1000000;

-- 2. Users who ordered within last 30 days
SELECT DISTINCT u.name
FROM users u
INNER JOIN orders o ON u.id = o.user_id
WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days';
```

### Practice 4: Multiple Table JOIN

```sql
-- Connect categories → products → orders
SELECT
    c.name AS category,
    p.name AS product,
    u.name AS customer,
    oi.quantity,
    p.price * oi.quantity AS subtotal
FROM categories c
JOIN products p ON c.id = p.category_id
JOIN order_items oi ON p.id = oi.product_id
JOIN orders o ON oi.order_id = o.id
JOIN users u ON o.user_id = u.id
ORDER BY c.name, p.name;
```

---

## 15. Performance Considerations

### Use Indexes

```sql
-- Create indexes on foreign key columns
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);
```

### SELECT Only Needed Columns

```sql
-- Bad example
SELECT * FROM users u JOIN orders o ON u.id = o.user_id;

-- Good example
SELECT u.name, o.product_name, o.amount
FROM users u JOIN orders o ON u.id = o.user_id;
```

### Check Execution Plan with EXPLAIN

```sql
EXPLAIN SELECT u.name, o.product_name
FROM users u
JOIN orders o ON u.id = o.user_id;
```

---

**Previous**: [Conditions and Sorting](./05_Conditions_and_Sorting.md) | **Next**: [Aggregation and Grouping](./07_Aggregation_and_Grouping.md)
