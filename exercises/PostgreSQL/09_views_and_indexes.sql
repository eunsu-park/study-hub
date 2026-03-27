-- Exercises for Lesson 09: Views and Indexes
-- Topic: PostgreSQL
-- Solutions to practice problems from the lesson.

-- Prerequisites: products, categories, orders, order_items tables
-- (created in earlier exercises)


-- === Exercise 1: Create Views ===
-- Problem: Create views that simplify common queries.

-- Solution:

-- 1. Product details view with stock status
-- A view encapsulates complex logic (here, a JOIN + CASE)
-- so consumers just SELECT * FROM product_details
CREATE OR REPLACE VIEW product_details AS
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

-- Usage: find out-of-stock products in one simple query
SELECT * FROM product_details WHERE status = 'Out of stock';

-- 2. Monthly revenue view
-- DATE_TRUNC groups timestamps into month buckets
CREATE OR REPLACE VIEW monthly_revenue AS
SELECT
    DATE_TRUNC('month', order_date) AS month,
    COUNT(*) AS orders,
    SUM(amount) AS revenue
FROM orders
WHERE status = 'completed'
GROUP BY DATE_TRUNC('month', order_date);

-- Usage
SELECT * FROM monthly_revenue ORDER BY month DESC;


-- === Exercise 2: Materialized View ===
-- Problem: Create a materialized view for expensive aggregation queries.

-- Solution:

-- Materialized views store the result physically, unlike regular views
-- which re-execute the query each time. Ideal for dashboard/reporting queries
-- that tolerate slightly stale data.
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

-- A unique index is required for CONCURRENTLY refresh (no table lock)
CREATE UNIQUE INDEX idx_category_stats ON category_stats(category);

-- Refresh data without blocking reads
REFRESH MATERIALIZED VIEW CONCURRENTLY category_stats;

-- Query the materialized view (instant, reads from stored result)
SELECT * FROM category_stats ORDER BY total_sold DESC NULLS LAST;


-- === Exercise 3: Index and Performance Comparison ===
-- Problem: Generate test data, compare query speed with and without indexes.

-- Solution:

-- 1. Create a test table and populate with 100,000 rows
CREATE TABLE test_orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    amount NUMERIC(10,2),
    order_date DATE
);

-- generate_series creates 100k rows; random() simulates realistic distribution
INSERT INTO test_orders (user_id, amount, order_date)
SELECT
    (random() * 1000)::INTEGER,
    (random() * 10000)::NUMERIC(10,2),
    '2024-01-01'::DATE + (random() * 365)::INTEGER
FROM generate_series(1, 100000);

-- 2. Query WITHOUT index — expect Seq Scan (full table scan)
EXPLAIN ANALYZE SELECT * FROM test_orders WHERE user_id = 500;

-- 3. Create B-tree index on user_id
CREATE INDEX idx_test_user_id ON test_orders(user_id);

-- 4. Query WITH index — expect Index Scan or Bitmap Index Scan
-- The planner switches from O(n) sequential scan to O(log n) index lookup
EXPLAIN ANALYZE SELECT * FROM test_orders WHERE user_id = 500;

-- 5. Additional: composite index for multi-column queries
CREATE INDEX idx_test_user_date ON test_orders(user_id, order_date);

-- The composite index supports this query efficiently because user_id
-- is the leading column
EXPLAIN ANALYZE
SELECT * FROM test_orders
WHERE user_id = 500 AND order_date > '2024-06-01';

-- Cleanup
DROP TABLE IF EXISTS test_orders;
