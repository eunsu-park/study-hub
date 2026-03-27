-- ============================================================================
-- PostgreSQL Query Optimization
-- ============================================================================
-- Demonstrates:
--   - EXPLAIN and EXPLAIN ANALYZE
--   - Scan types (Seq Scan, Index Scan, Bitmap Scan)
--   - Join strategies (Nested Loop, Hash Join, Merge Join)
--   - Index optimization (composite, partial, covering)
--   - Common anti-patterns and fixes
--   - Query rewriting techniques
--
-- Prerequisites: PostgreSQL 12+
-- Usage: psql -U postgres -d your_database -f 15_optimization.sql
-- ============================================================================

-- Clean up
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
DROP TABLE IF EXISTS products CASCADE;

-- ============================================================================
-- Setup: Create tables with realistic data
-- ============================================================================

CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    region TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price NUMERIC(10, 2) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    status TEXT NOT NULL DEFAULT 'pending',
    total NUMERIC(10, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE order_items (
    item_id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id),
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price NUMERIC(10, 2) NOT NULL
);

-- Insert sample data
INSERT INTO customers (name, email, region)
SELECT
    'Customer ' || i,
    'customer' || i || '@example.com',
    (ARRAY['US-East', 'US-West', 'EU', 'Asia'])[1 + (i % 4)]
FROM generate_series(1, 10000) AS i;

INSERT INTO products (name, category, price, is_active)
SELECT
    'Product ' || i,
    (ARRAY['Electronics', 'Books', 'Clothing', 'Food'])[1 + (i % 4)],
    ROUND((RANDOM() * 500 + 10)::NUMERIC, 2),
    i % 10 != 0  -- 10% inactive
FROM generate_series(1, 500) AS i;

INSERT INTO orders (customer_id, status, total, created_at)
SELECT
    1 + (i % 10000),
    (ARRAY['pending', 'confirmed', 'shipped', 'delivered'])[1 + (i % 4)],
    ROUND((RANDOM() * 1000 + 10)::NUMERIC, 2),
    CURRENT_TIMESTAMP - (RANDOM() * 365 || ' days')::INTERVAL
FROM generate_series(1, 50000) AS i;

INSERT INTO order_items (order_id, product_id, quantity, unit_price)
SELECT
    1 + (i % 50000),
    1 + (i % 500),
    1 + (i % 5),
    ROUND((RANDOM() * 200 + 5)::NUMERIC, 2)
FROM generate_series(1, 100000) AS i;

-- Update statistics for the planner
ANALYZE customers;
ANALYZE products;
ANALYZE orders;
ANALYZE order_items;

-- ============================================================================
-- 1. EXPLAIN Basics
-- ============================================================================

-- EXPLAIN shows the query plan (estimated costs, no execution)
EXPLAIN
SELECT * FROM customers WHERE region = 'US-East';

-- EXPLAIN ANALYZE actually runs the query (shows actual times and rows)
EXPLAIN ANALYZE
SELECT * FROM customers WHERE region = 'US-East';

-- EXPLAIN with all options
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT * FROM orders WHERE status = 'pending';

-- ============================================================================
-- 2. Scan Types
-- ============================================================================

-- Sequential Scan: reads entire table (no index)
EXPLAIN ANALYZE
SELECT * FROM orders WHERE total > 500;

-- Create index and compare
CREATE INDEX idx_orders_total ON orders(total);

-- Index Scan: B-tree traversal (selective queries)
EXPLAIN ANALYZE
SELECT * FROM orders WHERE total > 999;

-- Bitmap Index Scan: for medium selectivity
EXPLAIN ANALYZE
SELECT * FROM orders WHERE total BETWEEN 400 AND 600;

-- Index Only Scan: all needed columns are in the index
CREATE INDEX idx_orders_status_total ON orders(status, total);

EXPLAIN ANALYZE
SELECT status, total FROM orders WHERE status = 'pending' AND total > 500;

-- ============================================================================
-- 3. Join Strategies
-- ============================================================================

-- Nested Loop Join: best for small result sets or indexed inner table
EXPLAIN ANALYZE
SELECT c.name, o.total
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE c.customer_id = 42;

-- Hash Join: best for large unindexed tables
EXPLAIN ANALYZE
SELECT c.region, COUNT(*), SUM(o.total)
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.region;

-- Merge Join: best when both sides are sorted
CREATE INDEX idx_orders_customer ON orders(customer_id);

EXPLAIN ANALYZE
SELECT c.name, o.total
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
ORDER BY c.customer_id
LIMIT 100;

-- ============================================================================
-- 4. Index Optimization
-- ============================================================================

-- 4a. Composite Index (multi-column)
-- Column order matters: most selective or most queried first
CREATE INDEX idx_orders_status_created
    ON orders(status, created_at);

-- Uses the index (matches prefix):
EXPLAIN ANALYZE
SELECT * FROM orders WHERE status = 'pending' AND created_at > '2025-06-01';

-- Also uses the index (just the first column):
EXPLAIN ANALYZE
SELECT * FROM orders WHERE status = 'shipped';

-- Does NOT use the index (skips first column):
EXPLAIN ANALYZE
SELECT * FROM orders WHERE created_at > '2025-06-01';

-- Why: A partial index only indexes rows matching the WHERE condition, making it
-- dramatically smaller than a full index. If 75% of orders are 'delivered' and
-- you only query 'pending' orders, the partial index is ~4x smaller and faster.
CREATE INDEX idx_orders_pending
    ON orders(created_at)
    WHERE status = 'pending';

-- Very efficient for common filtered queries
EXPLAIN ANALYZE
SELECT * FROM orders WHERE status = 'pending' AND created_at > '2025-06-01';

-- Why: A covering index (INCLUDE) stores additional columns IN the index leaf pages
-- so PostgreSQL can satisfy the query entirely from the index (Index Only Scan)
-- without visiting the heap table. The INCLUDEd columns are not part of the B-tree
-- sort key, so they don't affect index ordering or size as much as key columns.
CREATE INDEX idx_orders_covering
    ON orders(customer_id)
    INCLUDE (status, total);

-- Index Only Scan — no table access needed
EXPLAIN ANALYZE
SELECT customer_id, status, total
FROM orders
WHERE customer_id = 42;

-- ============================================================================
-- 5. Common Anti-Patterns
-- ============================================================================

-- Why: Wrapping an indexed column in a function (EXTRACT) prevents the planner
-- from using the B-tree index because the index stores raw timestamps, not
-- extracted years. Rewriting as a range condition allows direct index lookup.
-- Alternatively, you could create a functional index on EXTRACT(...).
EXPLAIN ANALYZE
SELECT * FROM orders WHERE EXTRACT(YEAR FROM created_at) = 2025;

-- GOOD: rewrite to range condition
EXPLAIN ANALYZE
SELECT * FROM orders
WHERE created_at >= '2025-01-01' AND created_at < '2026-01-01';

-- 5b. Implicit type casting
-- BAD (if customer_id is INTEGER):
-- SELECT * FROM orders WHERE customer_id = '42';
-- GOOD:
EXPLAIN ANALYZE
SELECT * FROM orders WHERE customer_id = 42;

-- Why: Leading wildcards ('%@example.com') force a full table scan because B-tree
-- indexes are sorted left-to-right and cannot skip unknown leading characters.
-- text_pattern_ops enables prefix matching ('customer42%') with B-tree, but for
-- suffix matching you would need a trigram index (pg_trgm) or a reverse() expression index.
EXPLAIN ANALYZE
SELECT * FROM customers WHERE email LIKE '%@example.com';

CREATE INDEX idx_customers_email_pattern
    ON customers(email text_pattern_ops);

-- This can use the index (prefix match):
EXPLAIN ANALYZE
SELECT * FROM customers WHERE email LIKE 'customer42%';

-- 5d. SELECT * when only few columns needed
-- BAD:
EXPLAIN ANALYZE
SELECT * FROM orders WHERE customer_id = 42;

-- GOOD: only needed columns, enables index-only scan
EXPLAIN ANALYZE
SELECT order_id, total FROM orders WHERE customer_id = 42;

-- ============================================================================
-- 6. Query Rewriting
-- ============================================================================

-- 6a. EXISTS vs IN for correlated subqueries
-- EXISTS (usually faster for large outer + small inner):
EXPLAIN ANALYZE
SELECT c.name
FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o
    WHERE o.customer_id = c.customer_id
    AND o.total > 900
);

-- IN (usually faster for small subquery result):
EXPLAIN ANALYZE
SELECT c.name
FROM customers c
WHERE c.customer_id IN (
    SELECT customer_id FROM orders WHERE total > 900
);

-- 6b. CTE vs Subquery (PostgreSQL 12+ inlines CTEs by default)
EXPLAIN ANALYZE
WITH high_value AS (
    SELECT customer_id, SUM(total) AS total_spent
    FROM orders
    GROUP BY customer_id
    HAVING SUM(total) > 5000
)
SELECT c.name, h.total_spent
FROM customers c
JOIN high_value h ON c.customer_id = h.customer_id;

-- 6c. LIMIT + ORDER optimization
CREATE INDEX idx_orders_created_desc ON orders(created_at DESC);

-- Efficient: uses index to avoid sorting
EXPLAIN ANALYZE
SELECT * FROM orders ORDER BY created_at DESC LIMIT 10;

-- ============================================================================
-- 7. Statistics and Planner Hints
-- ============================================================================

-- View column statistics
SELECT
    attname AS column,
    n_distinct,
    most_common_vals,
    most_common_freqs
FROM pg_stats
WHERE tablename = 'orders' AND attname = 'status';

-- Why: The default statistics target (100) may not capture the full distribution
-- of skewed columns. Increasing to 500 tells ANALYZE to sample more values for
-- its histogram, leading to more accurate row estimates and better query plans.
-- The trade-off is slightly slower ANALYZE and more planner memory usage.
ALTER TABLE orders ALTER COLUMN status SET STATISTICS 500;
ANALYZE orders;

-- Planner cost parameters (session-level tuning)
-- SET random_page_cost = 1.1;    -- SSD storage (default 4.0 for HDD)
-- SET effective_cache_size = '4GB';
-- SET work_mem = '256MB';        -- For sorts and hash joins
