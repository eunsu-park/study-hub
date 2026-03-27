-- Exercises for Lesson 18: Table Partitioning
-- Topic: PostgreSQL
-- Solutions to practice problems from the lesson.


-- === Exercise 1: Monthly Log Partitioning ===
-- Problem: Partition an access_logs table by month using range partitioning.

-- Solution:

-- The partition key (created_at) must be part of the primary key
-- because PostgreSQL enforces uniqueness per partition, not globally.
CREATE TABLE access_logs (
    id BIGSERIAL,
    user_id INT,
    action VARCHAR(50),
    ip_address INET,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Dynamically create 12 monthly partitions for 2024.
-- Using DO block with dynamic SQL avoids writing 12 nearly-identical statements.
DO $$
DECLARE
    start_date DATE := '2024-01-01';
BEGIN
    FOR i IN 0..11 LOOP
        EXECUTE format(
            'CREATE TABLE access_logs_%s PARTITION OF access_logs
             FOR VALUES FROM (%L) TO (%L)',
            -- Partition name: access_logs_2024_01, access_logs_2024_02, ...
            TO_CHAR(start_date + (i || ' months')::interval, 'YYYY_MM'),
            -- FROM is inclusive
            start_date + (i || ' months')::interval,
            -- TO is exclusive (standard range semantics)
            start_date + ((i+1) || ' months')::interval
        );
    END LOOP;
END;
$$;

-- Default partition catches anything outside the defined ranges
CREATE TABLE access_logs_default PARTITION OF access_logs DEFAULT;

-- Create indexes (automatically applied to all partitions)
CREATE INDEX idx_access_logs_user ON access_logs (user_id, created_at);
CREATE INDEX idx_access_logs_action ON access_logs (action, created_at);

-- Insert test data
INSERT INTO access_logs (user_id, action, ip_address, created_at)
SELECT
    (random() * 1000)::int,
    (ARRAY['login', 'logout', 'view', 'edit', 'delete'])[1 + (random()*4)::int],
    ('192.168.' || (random()*255)::int || '.' || (random()*255)::int)::inet,
    '2024-01-01'::timestamp + (random() * 365 || ' days')::interval
FROM generate_series(1, 10000);

-- Verify partition pruning: should only scan the January partition
EXPLAIN ANALYZE
SELECT * FROM access_logs
WHERE created_at BETWEEN '2024-01-01' AND '2024-01-31';


-- === Exercise 2: Regional Order Partitioning ===
-- Problem: Partition orders by country code using list partitioning.

-- Solution:

-- List partitioning maps discrete values to specific partitions.
-- Ideal when you have a known set of categories (countries, regions, types).
CREATE TABLE regional_orders (
    id BIGSERIAL,
    country_code CHAR(2) NOT NULL,
    customer_id INT,
    total NUMERIC(10,2),
    order_date TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (id, country_code)
) PARTITION BY LIST (country_code);

-- Dedicated partitions for high-volume countries
CREATE TABLE regional_orders_kr PARTITION OF regional_orders
    FOR VALUES IN ('KR');

CREATE TABLE regional_orders_us PARTITION OF regional_orders
    FOR VALUES IN ('US');

CREATE TABLE regional_orders_jp PARTITION OF regional_orders
    FOR VALUES IN ('JP');

-- Default partition for all other countries
-- Without this, INSERTs for unlisted country codes would fail
CREATE TABLE regional_orders_others PARTITION OF regional_orders DEFAULT;

-- Create indexes
CREATE INDEX idx_regional_orders_customer ON regional_orders (customer_id);
CREATE INDEX idx_regional_orders_date ON regional_orders (order_date);

-- Insert test data
INSERT INTO regional_orders (country_code, customer_id, total, order_date)
SELECT
    (ARRAY['KR', 'US', 'JP', 'DE', 'FR', 'GB'])[1 + (random()*5)::int],
    (random() * 500)::int,
    (random() * 10000)::numeric(10,2),
    NOW() - (random() * 365 || ' days')::interval
FROM generate_series(1, 5000);

-- Verify partition routing: should only scan the KR partition
EXPLAIN ANALYZE
SELECT * FROM regional_orders WHERE country_code = 'KR';

-- Count rows per partition
SELECT
    tableoid::regclass AS partition_name,
    COUNT(*) AS row_count
FROM regional_orders
GROUP BY tableoid
ORDER BY row_count DESC;


-- === Exercise 3: Partition Maintenance Query ===
-- Problem: Identify partitions with data older than 90 days for cleanup.

-- Solution:

-- This CTE queries the system catalog to extract partition boundaries.
-- pg_get_expr returns the human-readable partition bound expression,
-- and regex extracts the start date from the FROM clause.
WITH partition_info AS (
    SELECT
        child.relname AS partition_name,
        pg_get_expr(child.relpartbound, child.oid) AS bounds,
        -- Extract the start date from the partition bound expression
        -- Format: FOR VALUES FROM ('2024-01-01 00:00:00') TO ('2024-02-01 00:00:00')
        (regexp_match(
            pg_get_expr(child.relpartbound, child.oid),
            $$FROM \('([^']+)'\)$$
        ))[1]::date AS start_date
    FROM pg_inherits
    JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
    JOIN pg_class child ON pg_inherits.inhrelid = child.oid
    WHERE parent.relname = 'access_logs'
      AND child.relispartition = true
      -- Exclude the default partition (it has no FROM/TO bounds)
      AND pg_get_expr(child.relpartbound, child.oid) != 'DEFAULT'
)
SELECT
    partition_name,
    start_date,
    bounds,
    -- Calculate age of each partition
    CURRENT_DATE - start_date AS age_days,
    CASE
        WHEN start_date < CURRENT_DATE - INTERVAL '90 days' THEN 'ELIGIBLE FOR DROP'
        ELSE 'KEEP'
    END AS action
FROM partition_info
ORDER BY start_date;

-- To actually drop old partitions (run after review):
-- DO $$
-- DECLARE
--     part RECORD;
-- BEGIN
--     FOR part IN
--         SELECT child.relname
--         FROM pg_inherits
--         JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
--         JOIN pg_class child ON pg_inherits.inhrelid = child.oid
--         WHERE parent.relname = 'access_logs'
--           AND child.relispartition = true
--           AND pg_get_expr(child.relpartbound, child.oid) != 'DEFAULT'
--           AND (regexp_match(
--               pg_get_expr(child.relpartbound, child.oid),
--               $$FROM \('([^']+)'\)$$
--           ))[1]::date < CURRENT_DATE - INTERVAL '90 days'
--     LOOP
--         EXECUTE 'DROP TABLE IF EXISTS ' || part.relname;
--         RAISE NOTICE 'Dropped partition: %', part.relname;
--     END LOOP;
-- END;
-- $$;
