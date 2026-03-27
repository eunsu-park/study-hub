-- Exercises for Lesson 15: Query Optimization
-- Topic: PostgreSQL
-- Solutions to practice problems from the lesson.


-- === Exercise 1: Analyze Execution Plan ===
-- Problem: Analyze and optimize a complex LEFT JOIN query with GROUP BY.

-- Solution:

-- Original query:
-- SELECT u.name, COUNT(o.id), SUM(o.total)
-- FROM users u
-- LEFT JOIN orders o ON u.id = o.user_id
-- WHERE u.country = 'US'
-- AND o.created_at > NOW() - INTERVAL '1 year'
-- GROUP BY u.name
-- HAVING COUNT(o.id) > 10
-- ORDER BY SUM(o.total) DESC
-- LIMIT 100;

-- Issue 1: LEFT JOIN + WHERE on right table converts it to INNER JOIN.
-- The WHERE o.created_at > ... filters out NULL rows from LEFT JOIN,
-- making it effectively an INNER JOIN. Use explicit INNER JOIN for clarity.

-- Issue 2: Missing indexes on filter and join columns.

-- Issue 3: GROUP BY u.name is ambiguous if names aren't unique.
-- Use u.id instead (PK guarantees uniqueness).

-- Step 1: Create supporting indexes
-- Composite index on users for the WHERE filter + JOIN
CREATE INDEX idx_users_country ON users (country);

-- Composite index on orders covering both the JOIN and date filter
-- Including total as a covering column avoids table lookups
CREATE INDEX idx_orders_user_date ON orders (user_id, created_at)
    INCLUDE (total);

-- Step 2: Optimized query
EXPLAIN ANALYZE
SELECT u.id, u.name, COUNT(o.id) AS order_count, SUM(o.total) AS total_spent
FROM users u
INNER JOIN orders o ON u.id = o.user_id
WHERE u.country = 'US'
  AND o.created_at > NOW() - INTERVAL '1 year'
GROUP BY u.id, u.name
HAVING COUNT(o.id) > 10
ORDER BY total_spent DESC
LIMIT 100;


-- === Exercise 2: Index Design ===
-- Problem: Design optimal indexes for three different query patterns.

-- Solution:

-- Query 1: SELECT * FROM orders WHERE user_id = ? AND status = 'pending'
--          ORDER BY created_at DESC
-- Composite index: equality columns first, then the sort column.
-- The planner can satisfy WHERE + ORDER BY in a single index scan.
CREATE INDEX idx_orders_user_status_date
    ON orders (user_id, status, created_at DESC);

-- Query 2: SELECT * FROM products WHERE category_id = ? AND price BETWEEN ? AND ?
-- Equality column first, then the range column.
-- B-tree efficiently handles equality + range on the trailing column.
CREATE INDEX idx_products_cat_price
    ON products (category_id, price);

-- Query 3: SELECT * FROM logs WHERE level = 'ERROR'
--          AND created_at > NOW() - INTERVAL '1 day'
-- Partial index: only indexes ERROR rows, making it much smaller.
-- Perfect when you only query a small subset of all rows.
CREATE INDEX idx_logs_error_recent
    ON logs (created_at DESC)
    WHERE level = 'ERROR';

-- Verify index usage with EXPLAIN
-- EXPLAIN ANALYZE SELECT * FROM orders
-- WHERE user_id = 42 AND status = 'pending'
-- ORDER BY created_at DESC;


-- === Exercise 3: Join Optimization ===
-- Problem: Optimize a 5-table join query.

-- Solution:

-- Original query uses SELECT * (fetches all columns from all 5 tables)
-- and has no supporting indexes on join/filter columns.

-- Step 1: Select only needed columns (reduces I/O and memory)
-- Step 2: Ensure foreign key columns are indexed
CREATE INDEX IF NOT EXISTS idx_orders_user_id ON orders (user_id);
CREATE INDEX IF NOT EXISTS idx_orders_product_id ON orders (product_id);
CREATE INDEX IF NOT EXISTS idx_products_category_id ON products (category_id);
CREATE INDEX IF NOT EXISTS idx_products_supplier_id ON products (supplier_id);

-- Step 3: Add a composite index for the WHERE clause
-- Filtering by category name + date range
CREATE INDEX IF NOT EXISTS idx_orders_created ON orders (created_at);

-- Step 4: Optimized query
-- Push the category filter into a CTE or subquery so fewer rows enter the join.
EXPLAIN ANALYZE
WITH electronics_products AS (
    -- Pre-filter to electronics category: reduces the join cardinality
    SELECT p.id, p.name AS product_name, p.supplier_id, c.name AS category_name
    FROM products p
    JOIN categories c ON p.category_id = c.id
    WHERE c.name = 'Electronics'
)
SELECT
    ep.category_name,
    ep.product_name,
    s.name AS supplier_name,
    u.name AS customer_name,
    o.created_at AS order_date
FROM orders o
JOIN users u ON o.user_id = u.id
JOIN electronics_products ep ON o.product_id = ep.id
JOIN suppliers s ON ep.supplier_id = s.id
WHERE o.created_at > '2024-01-01'
ORDER BY o.created_at DESC;


-- === Exercise 4: Partitioning Design ===
-- Problem: Design partitioning for a log table with 1M rows/day,
-- 3-month retention, frequently queried by level, created_at, user_id.

-- Solution:

-- Strategy: Range partition by month on created_at.
-- Monthly partitions keep each partition manageable (~30M rows)
-- and allow easy DROP for data older than 3 months.

CREATE TABLE app_logs (
    id BIGSERIAL,
    level VARCHAR(10) NOT NULL,
    user_id INTEGER,
    message TEXT,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    -- Primary key must include the partition key
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create monthly partitions (3 months + 1 month ahead)
DO $$
DECLARE
    start_date DATE := DATE_TRUNC('month', CURRENT_DATE - INTERVAL '2 months');
    partition_start DATE;
    partition_end DATE;
    partition_name TEXT;
BEGIN
    FOR i IN 0..3 LOOP
        partition_start := start_date + (i || ' months')::interval;
        partition_end := start_date + ((i + 1) || ' months')::interval;
        partition_name := 'app_logs_' || TO_CHAR(partition_start, 'YYYY_MM');

        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I PARTITION OF app_logs
             FOR VALUES FROM (%L) TO (%L)',
            partition_name, partition_start, partition_end
        );
    END LOOP;
END;
$$;

-- Create a default partition for data outside defined ranges
CREATE TABLE IF NOT EXISTS app_logs_default PARTITION OF app_logs DEFAULT;

-- Indexes: each partition gets its own copy automatically
-- B-tree on level + created_at for filtered time-range queries
CREATE INDEX idx_app_logs_level_time ON app_logs (level, created_at DESC);

-- B-tree on user_id for per-user log lookups
CREATE INDEX idx_app_logs_user ON app_logs (user_id, created_at DESC);

-- Maintenance: drop the oldest partition when it exceeds 3 months
-- This is O(1) — no row-by-row DELETE needed
-- DROP TABLE IF EXISTS app_logs_2024_01;

-- Verify partition pruning works (should only scan relevant partitions)
EXPLAIN ANALYZE
SELECT * FROM app_logs
WHERE level = 'ERROR'
  AND created_at > NOW() - INTERVAL '1 day'
ORDER BY created_at DESC
LIMIT 100;
