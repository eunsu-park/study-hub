-- ksqlDB Queries for E-Commerce Stream Processing
-- ================================================
-- Demonstrates: streams, tables, windowed aggregations, joins, connectors
-- Usage: Execute in ksqlDB CLI or REST API

-- ============================================================
-- 1. CREATE STREAMS (from existing Kafka topics)
-- ============================================================

-- Orders stream from JSON topic
CREATE STREAM orders_stream (
    order_id VARCHAR KEY,
    user_id VARCHAR,
    product_id VARCHAR,
    amount DOUBLE,
    currency VARCHAR,
    order_time TIMESTAMP
) WITH (
    KAFKA_TOPIC = 'orders',
    VALUE_FORMAT = 'JSON',
    TIMESTAMP = 'order_time'
);

-- Page views stream
CREATE STREAM page_views_stream (
    view_id VARCHAR KEY,
    user_id VARCHAR,
    page_url VARCHAR,
    referrer VARCHAR,
    view_time TIMESTAMP
) WITH (
    KAFKA_TOPIC = 'page_views',
    VALUE_FORMAT = 'JSON',
    TIMESTAMP = 'view_time'
);

-- ============================================================
-- 2. CREATE TABLES (materialized views)
-- ============================================================

-- User profiles table (latest state per user)
CREATE TABLE user_profiles (
    user_id VARCHAR PRIMARY KEY,
    name VARCHAR,
    email VARCHAR,
    tier VARCHAR,
    country VARCHAR
) WITH (
    KAFKA_TOPIC = 'user_profiles',
    VALUE_FORMAT = 'JSON'
);

-- Product catalog table
CREATE TABLE products (
    product_id VARCHAR PRIMARY KEY,
    name VARCHAR,
    category VARCHAR,
    price DOUBLE
) WITH (
    KAFKA_TOPIC = 'products',
    VALUE_FORMAT = 'JSON'
);

-- ============================================================
-- 3. DERIVED STREAMS (filtering and transformation)
-- ============================================================

-- High-value orders stream
CREATE STREAM high_value_orders AS
SELECT
    order_id,
    user_id,
    product_id,
    amount,
    currency,
    order_time
FROM orders_stream
WHERE amount > 1000
EMIT CHANGES;

-- USD-normalized orders
CREATE STREAM orders_usd AS
SELECT
    order_id,
    user_id,
    product_id,
    CASE
        WHEN currency = 'EUR' THEN amount * 1.08
        WHEN currency = 'GBP' THEN amount * 1.27
        ELSE amount
    END AS amount_usd,
    order_time
FROM orders_stream
EMIT CHANGES;

-- ============================================================
-- 4. WINDOWED AGGREGATIONS
-- ============================================================

-- Tumbling window: orders per user per hour
CREATE TABLE hourly_user_orders AS
SELECT
    user_id,
    COUNT(*) AS order_count,
    SUM(amount) AS total_amount,
    AVG(amount) AS avg_amount,
    MIN(amount) AS min_amount,
    MAX(amount) AS max_amount,
    WINDOWSTART AS window_start,
    WINDOWEND AS window_end
FROM orders_stream
WINDOW TUMBLING (SIZE 1 HOUR)
GROUP BY user_id
EMIT CHANGES;

-- Hopping window: revenue per product category (30-min window, 10-min hop)
CREATE TABLE category_revenue_30min AS
SELECT
    p.category,
    COUNT(*) AS order_count,
    SUM(o.amount) AS revenue,
    WINDOWSTART AS window_start,
    WINDOWEND AS window_end
FROM orders_stream o
INNER JOIN products p ON o.product_id = p.product_id
WINDOW HOPPING (SIZE 30 MINUTES, ADVANCE BY 10 MINUTES)
GROUP BY p.category
EMIT CHANGES;

-- Session window: user sessions from page views (30-min gap)
CREATE TABLE user_sessions AS
SELECT
    user_id,
    COUNT(*) AS page_count,
    COUNT_DISTINCT(page_url) AS unique_pages,
    WINDOWSTART AS session_start,
    WINDOWEND AS session_end
FROM page_views_stream
WINDOW SESSION (30 MINUTES)
GROUP BY user_id
EMIT CHANGES;

-- ============================================================
-- 5. STREAM-TABLE JOINS (enrichment)
-- ============================================================

-- Enrich orders with user profile and product info
CREATE STREAM enriched_orders AS
SELECT
    o.order_id,
    o.user_id,
    u.name AS user_name,
    u.tier AS user_tier,
    u.country AS user_country,
    o.product_id,
    p.name AS product_name,
    p.category AS product_category,
    o.amount,
    o.order_time
FROM orders_stream o
LEFT JOIN user_profiles u ON o.user_id = u.user_id
LEFT JOIN products p ON o.product_id = p.product_id
EMIT CHANGES;

-- ============================================================
-- 6. FRAUD DETECTION PATTERNS
-- ============================================================

-- Velocity check: users with > 5 orders in 10 minutes
CREATE TABLE suspicious_velocity AS
SELECT
    user_id,
    COUNT(*) AS order_count,
    SUM(amount) AS total_amount,
    WINDOWSTART AS window_start
FROM orders_stream
WINDOW TUMBLING (SIZE 10 MINUTES)
GROUP BY user_id
HAVING COUNT(*) > 5
EMIT CHANGES;

-- ============================================================
-- 7. PULL vs PUSH QUERIES
-- ============================================================

-- Pull query: point-in-time lookup (returns immediately)
-- SELECT * FROM hourly_user_orders WHERE user_id = 'user_123';

-- Push query: continuous updates (long-running, streaming results)
-- SELECT * FROM enriched_orders WHERE amount > 500 EMIT CHANGES;

-- ============================================================
-- 8. USEFUL FUNCTIONS
-- ============================================================

-- String functions
-- SELECT UCASE(user_name), LCASE(product_category) FROM enriched_orders EMIT CHANGES;

-- Date/Time functions
-- SELECT TIMESTAMPTOSTRING(order_time, 'yyyy-MM-dd HH:mm:ss') AS formatted_time
-- FROM orders_stream EMIT CHANGES;

-- Conditional
-- SELECT
--     order_id,
--     CASE
--         WHEN amount > 5000 THEN 'premium'
--         WHEN amount > 1000 THEN 'high'
--         WHEN amount > 100 THEN 'medium'
--         ELSE 'low'
--     END AS order_tier
-- FROM orders_stream EMIT CHANGES;
