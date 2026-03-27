# Views and Indexes

**Previous**: [Subqueries and CTE](./08_Subqueries_and_CTE.md) | **Next**: [Functions and Procedures](./10_Functions_and_Procedures.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what a view is and describe its advantages (query simplification, security, logical independence)
2. Create, replace, rename, and drop views using DDL statements
3. Distinguish between regular views and materialized views and explain when to use each
4. Identify whether a view is updatable and apply WITH CHECK OPTION for data integrity
5. Explain how B-tree indexes accelerate lookups and describe the trade-off with write performance
6. Create single-column, composite, partial, expression, and unique indexes
7. Compare index types (B-tree, Hash, GIN, GiST) and choose the appropriate one for a given use case
8. Read EXPLAIN / EXPLAIN ANALYZE output to verify that indexes are being used

---

As your database grows, two challenges emerge: complex queries become tedious to repeat, and full-table scans become unacceptably slow. Views solve the first problem by encapsulating a query behind a simple name, while indexes solve the second by giving PostgreSQL a fast lookup path to the rows you need. Together, they are essential tools for building maintainable and performant database applications.

## 1. VIEW Concept

A view is a stored query that can be used like a virtual table.

```
┌─────────────────────────────────────────────────────────┐
│                       VIEW                              │
│  ┌───────────────────────────────────────────────────┐ │
│  │  SELECT u.name, SUM(o.amount) AS total           │ │
│  │  FROM users u JOIN orders o ON u.id = o.user_id  │ │
│  │  GROUP BY u.id, u.name                           │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                          ↓
              SELECT * FROM user_sales;
                    (Use simply)
```

---

## 2. Create View

### Basic View Creation

```sql
-- View showing only active users
CREATE VIEW active_users AS
SELECT id, name, email
FROM users
WHERE is_active = true;

-- Use view
SELECT * FROM active_users;
SELECT * FROM active_users WHERE name LIKE 'Kim%';
```

### Complex Query as View

```sql
-- User order statistics view
CREATE VIEW user_order_stats AS
SELECT
    u.id AS user_id,
    u.name,
    u.email,
    COUNT(o.id) AS order_count,
    COALESCE(SUM(o.amount), 0) AS total_amount,
    MAX(o.order_date) AS last_order_date
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name, u.email;

-- Simple query
SELECT * FROM user_order_stats WHERE order_count > 0;
```

### OR REPLACE

```sql
-- Replace if exists, create if not
CREATE OR REPLACE VIEW active_users AS
SELECT id, name, email, created_at
FROM users
WHERE is_active = true;
```

---

## 3. Modify and Delete Views

### Delete View

```sql
DROP VIEW active_users;
DROP VIEW IF EXISTS active_users;

-- Delete with dependent objects
DROP VIEW active_users CASCADE;
```

### Rename View

```sql
ALTER VIEW active_users RENAME TO enabled_users;
```

---

## 4. Advantages of Views

```sql
-- 1. Simplify queries
-- After creating view with complex joins
SELECT * FROM user_order_stats WHERE total_amount > 1000000;

-- 2. Security (expose only specific columns)
CREATE VIEW public_users AS
SELECT id, name FROM users;  -- Exclude email, password

-- 3. Logical data independence
-- If table structure changes, only need to modify view
```

---

## 5. Updatable Views

Simple views allow INSERT, UPDATE, DELETE.

```sql
-- Simple view (updatable)
CREATE VIEW seoul_users AS
SELECT * FROM users WHERE city = 'Seoul';

-- Update through view
UPDATE seoul_users SET name = 'Kim Seoul' WHERE id = 1;

-- Insert through view
INSERT INTO seoul_users (name, email, city)
VALUES ('New User', 'new@email.com', 'Seoul');
```

### WITH CHECK OPTION

```sql
-- Prevent inserting/updating data outside view condition
CREATE VIEW seoul_users AS
SELECT * FROM users WHERE city = 'Seoul'
WITH CHECK OPTION;

-- Error (city is 'Busan')
INSERT INTO seoul_users (name, email, city)
VALUES ('Busan Person', 'busan@email.com', 'Busan');
```

---

## 6. Materialized View

A view that physically stores results.

### Create

```sql
CREATE MATERIALIZED VIEW monthly_sales AS
SELECT
    DATE_TRUNC('month', order_date) AS month,
    COUNT(*) AS order_count,
    SUM(amount) AS total_amount
FROM orders
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY month;
```

### Query

```sql
SELECT * FROM monthly_sales;
```

### Refresh (Update Data)

```sql
-- Full refresh (table lock)
REFRESH MATERIALIZED VIEW monthly_sales;

-- Concurrent refresh (allows concurrent access, needs UNIQUE index)
REFRESH MATERIALIZED VIEW CONCURRENTLY monthly_sales;
```

### Delete

```sql
DROP MATERIALIZED VIEW monthly_sales;
```

### Regular VIEW vs MATERIALIZED VIEW

| Feature | VIEW | MATERIALIZED VIEW |
|---------|------|-------------------|
| Data storage | No | Yes |
| Real-time updates | Yes | No (needs REFRESH) |
| Query speed | Slow (executes each time) | Fast (stored results) |
| Storage space | None | Required |

---

## 7. INDEX Concept

An index is a data structure that speeds up data retrieval.

```
Table (Sequential scan):
┌─────────────────────────────────────────────┐
│ 1, 2, 3, 4, 5, 6, ... 999998, 999999, 1000000
└─────────────────────────────────────────────┘
  → Worst case: 1,000,000 comparisons

Index (B-tree):
           ┌─── [500000] ───┐
           │                │
    ┌─[250000]─┐      ┌─[750000]─┐
    │          │      │          │
  [125K]    [375K]  [625K]    [875K]
  → Maximum ~20 comparisons to find
```

---

## 8. Create Index

### Basic Index

```sql
-- Single column index
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_orders_user_id ON orders(user_id);

-- Composite index (multiple columns)
CREATE INDEX idx_orders_user_date ON orders(user_id, order_date);
```

### Unique Index

```sql
CREATE UNIQUE INDEX idx_users_email_unique ON users(email);
```

### Partial Index (Conditional)

```sql
-- Index only active users
CREATE INDEX idx_active_users ON users(email) WHERE is_active = true;

-- Only non-NULL values
CREATE INDEX idx_orders_shipped ON orders(shipped_date) WHERE shipped_date IS NOT NULL;
```

### Expression Index

```sql
-- Index on lowercase conversion result
CREATE INDEX idx_users_lower_email ON users(LOWER(email));

-- Usage
SELECT * FROM users WHERE LOWER(email) = 'kim@email.com';
```

---

## 9. Index Types

### B-tree (Default)

```sql
-- Default index (B-tree)
CREATE INDEX idx_products_price ON products(price);

-- Effective for range searches, sorting, equality comparisons
SELECT * FROM products WHERE price BETWEEN 1000 AND 5000;
SELECT * FROM products ORDER BY price;
```

### Hash

```sql
-- Effective only for equality comparisons
CREATE INDEX idx_users_email_hash ON users USING hash(email);

-- Effective
SELECT * FROM users WHERE email = 'kim@email.com';

-- Hash index not used
SELECT * FROM users WHERE email LIKE 'kim%';
```

### GIN (Generalized Inverted Index)

```sql
-- For arrays, JSON, full-text search
CREATE INDEX idx_products_tags ON products USING gin(tags);
CREATE INDEX idx_products_attrs ON products USING gin(attributes);

-- Array search
SELECT * FROM products WHERE tags @> ARRAY['sale'];

-- JSON search
SELECT * FROM products WHERE attributes @> '{"color": "red"}';
```

### GiST (Generalized Search Tree)

```sql
-- For geometric data, full-text search
CREATE INDEX idx_locations_coords ON locations USING gist(coordinates);
```

---

## 10. Index Management

### List Indexes

```sql
-- psql command
\di

-- SQL query
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'users';
```

### Delete Index

```sql
DROP INDEX idx_users_email;
DROP INDEX IF EXISTS idx_users_email;
```

### Rebuild Index

```sql
-- Rebuild index
REINDEX INDEX idx_users_email;

-- Rebuild all indexes on table
REINDEX TABLE users;
```

---

## 11. EXPLAIN - Execution Plan Analysis

### Basic EXPLAIN

```sql
EXPLAIN SELECT * FROM users WHERE email = 'kim@email.com';
```

Output:
```
                        QUERY PLAN
----------------------------------------------------------
 Index Scan using idx_users_email on users  (cost=0.29..8.30 rows=1 width=100)
   Index Cond: (email = 'kim@email.com'::text)
```

### EXPLAIN ANALYZE (Actual Execution)

```sql
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'kim@email.com';
```

Output:
```
                        QUERY PLAN
----------------------------------------------------------
 Index Scan using idx_users_email on users  (cost=0.29..8.30 rows=1 width=100)
                                             (actual time=0.025..0.027 rows=1 loops=1)
   Index Cond: (email = 'kim@email.com'::text)
 Planning Time: 0.085 ms
 Execution Time: 0.045 ms
```

### Main Scan Types

| Scan Type | Description | Performance |
|-----------|-------------|-------------|
| Seq Scan | Full table sequential scan | Slow |
| Index Scan | Uses index | Fast |
| Index Only Scan | Returns results from index only | Very fast |
| Bitmap Index Scan | Combines multiple indexes | Medium |

### EXPLAIN Examples

```sql
-- Without index
EXPLAIN SELECT * FROM orders WHERE user_id = 1;
-- Seq Scan on orders  (inefficient)

-- After creating index
CREATE INDEX idx_orders_user_id ON orders(user_id);
EXPLAIN SELECT * FROM orders WHERE user_id = 1;
-- Index Scan using idx_orders_user_id  (efficient)
```

---

## 12. Index Design Guide

### When to Create Indexes

```sql
-- 1. Columns frequently used in WHERE clause
CREATE INDEX idx_users_city ON users(city);

-- 2. Columns used in JOIN conditions (foreign keys)
CREATE INDEX idx_orders_user_id ON orders(user_id);

-- 3. Columns used in ORDER BY
CREATE INDEX idx_products_price ON products(price);

-- 4. Columns needing unique constraint
CREATE UNIQUE INDEX idx_users_email ON users(email);
```

### When to Avoid Indexes

```sql
-- 1. Frequently modified columns (degrades INSERT/UPDATE performance)
-- 2. Low cardinality columns (e.g., gender, boolean)
-- 3. Small tables (full scan is faster)
-- 4. Rarely used columns
```

### Composite Index Column Order

```sql
-- Used from leftmost column
CREATE INDEX idx_orders_user_date ON orders(user_id, order_date);

-- Effective
SELECT * FROM orders WHERE user_id = 1;
SELECT * FROM orders WHERE user_id = 1 AND order_date > '2024-01-01';

-- Ineffective (no first column)
SELECT * FROM orders WHERE order_date > '2024-01-01';
```

---

## 13. Practice Examples

### Practice 1: Create Views

```sql
-- 1. Product details view
CREATE VIEW product_details AS
SELECT
    p.id,
    p.name,
    c.name AS category,
    p.price,
    p.stock,
    CASE
        WHEN p.stock = 0 THEN 'Out of stock'
        WHEN p.stock < 10 THEN 'Low stock'
        ELSE 'In stock'
    END AS status
FROM products p
JOIN categories c ON p.category_id = c.id;

-- Usage
SELECT * FROM product_details WHERE status = 'Out of stock';

-- 2. Monthly revenue view
CREATE VIEW monthly_revenue AS
SELECT
    DATE_TRUNC('month', order_date) AS month,
    COUNT(*) AS orders,
    SUM(amount) AS revenue
FROM orders
WHERE status = 'completed'
GROUP BY DATE_TRUNC('month', order_date);
```

### Practice 2: Materialized View

```sql
-- Category statistics (heavy query)
CREATE MATERIALIZED VIEW category_stats AS
SELECT
    c.name AS category,
    COUNT(p.id) AS product_count,
    AVG(p.price) AS avg_price,
    SUM(oi.quantity) AS total_sold
FROM categories c
LEFT JOIN products p ON c.id = p.category_id
LEFT JOIN order_items oi ON p.id = oi.product_id
GROUP BY c.id, c.name;

-- Create unique index (for CONCURRENTLY refresh)
CREATE UNIQUE INDEX idx_category_stats ON category_stats(category);

-- Refresh
REFRESH MATERIALIZED VIEW CONCURRENTLY category_stats;
```

### Practice 3: Index and Performance Comparison

```sql
-- Generate test data
CREATE TABLE test_orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    amount NUMERIC(10,2),
    order_date DATE
);

INSERT INTO test_orders (user_id, amount, order_date)
SELECT
    (random() * 1000)::INTEGER,
    (random() * 10000)::NUMERIC(10,2),
    '2024-01-01'::DATE + (random() * 365)::INTEGER
FROM generate_series(1, 100000);

-- Query without index
EXPLAIN ANALYZE SELECT * FROM test_orders WHERE user_id = 500;

-- Create index
CREATE INDEX idx_test_user_id ON test_orders(user_id);

-- Query with index
EXPLAIN ANALYZE SELECT * FROM test_orders WHERE user_id = 500;
```

---

**Previous**: [Subqueries and CTE](./08_Subqueries_and_CTE.md) | **Next**: [Functions and Procedures](./10_Functions_and_Procedures.md)
