-- ============================================================================
-- PostgreSQL Table Partitioning
-- ============================================================================
-- Demonstrates:
--   - Declarative partitioning (PostgreSQL 10+)
--   - Range partitioning (by date, by value)
--   - List partitioning (by category)
--   - Hash partitioning (even distribution)
--   - Sub-partitioning (multi-level)
--   - Partition maintenance (add, detach, drop)
--   - Partition pruning verification
--
-- Prerequisites: PostgreSQL 12+ (14+ recommended for performance)
-- Usage: psql -U postgres -d your_database -f 18_partitioning.sql
-- ============================================================================

-- Clean up
DROP TABLE IF EXISTS events CASCADE;
DROP TABLE IF EXISTS regional_sales CASCADE;
DROP TABLE IF EXISTS sensor_data CASCADE;
DROP TABLE IF EXISTS orders_partitioned CASCADE;

-- ============================================================================
-- 1. Range Partitioning (by date)
-- ============================================================================
-- Most common pattern: partition time-series data by month/quarter/year

CREATE TABLE events (
    event_id BIGSERIAL,
    event_type TEXT NOT NULL,
    payload JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) PARTITION BY RANGE (created_at);

-- Create monthly partitions
CREATE TABLE events_2025_01 PARTITION OF events
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE events_2025_02 PARTITION OF events
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

CREATE TABLE events_2025_03 PARTITION OF events
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');

CREATE TABLE events_2025_q2 PARTITION OF events
    FOR VALUES FROM ('2025-04-01') TO ('2025-07-01');

-- Why: A DEFAULT partition prevents INSERT failures when data arrives outside
-- defined ranges (e.g., a timestamp in 2026). Without it, inserting a row
-- that matches no partition raises an error. However, it means you must DETACH
-- the default before adding a new partition that overlaps its data range.
CREATE TABLE events_default PARTITION OF events DEFAULT;

-- Why: Indexes defined on the parent table are automatically created on all
-- existing AND future partitions. Each partition gets its own independent
-- index, which is smaller and faster to scan than a single massive index
-- on a monolithic table.
CREATE INDEX idx_events_type ON events (event_type);
CREATE INDEX idx_events_created ON events (created_at);

-- Insert test data
INSERT INTO events (event_type, payload, created_at)
SELECT
    (ARRAY['click', 'view', 'purchase', 'login'])[1 + (i % 4)],
    jsonb_build_object('user_id', i % 100, 'value', i),
    '2025-01-01'::TIMESTAMP + (i * INTERVAL '1 hour')
FROM generate_series(1, 2000) AS i;

-- Verify distribution across partitions
SELECT
    tableoid::regclass AS partition,
    COUNT(*) AS rows,
    MIN(created_at) AS min_date,
    MAX(created_at) AS max_date
FROM events
GROUP BY tableoid
ORDER BY partition;

-- ============================================================================
-- 2. Partition Pruning
-- ============================================================================
-- PostgreSQL automatically skips irrelevant partitions

-- Only scans events_2025_01 partition
EXPLAIN ANALYZE
SELECT * FROM events
WHERE created_at >= '2025-01-15' AND created_at < '2025-02-01';

-- Scans only February partition
EXPLAIN ANALYZE
SELECT COUNT(*) FROM events
WHERE created_at >= '2025-02-01' AND created_at < '2025-03-01';

-- Verify pruning is enabled
SHOW enable_partition_pruning;

-- ============================================================================
-- 3. List Partitioning (by category)
-- ============================================================================

CREATE TABLE regional_sales (
    sale_id SERIAL,
    region TEXT NOT NULL,
    product TEXT NOT NULL,
    amount NUMERIC(10, 2) NOT NULL,
    sale_date DATE NOT NULL
) PARTITION BY LIST (region);

CREATE TABLE sales_us_east PARTITION OF regional_sales
    FOR VALUES IN ('US-East', 'US-Southeast');

CREATE TABLE sales_us_west PARTITION OF regional_sales
    FOR VALUES IN ('US-West', 'US-Northwest');

CREATE TABLE sales_eu PARTITION OF regional_sales
    FOR VALUES IN ('EU-West', 'EU-East', 'EU-North');

CREATE TABLE sales_asia PARTITION OF regional_sales
    FOR VALUES IN ('Asia-East', 'Asia-South');

CREATE TABLE sales_other PARTITION OF regional_sales DEFAULT;

-- Insert test data
INSERT INTO regional_sales (region, product, amount, sale_date) VALUES
    ('US-East', 'Widget', 100.00, '2025-01-15'),
    ('US-West', 'Gadget', 200.00, '2025-01-20'),
    ('EU-West', 'Widget', 150.00, '2025-02-10'),
    ('Asia-East', 'Gadget', 175.00, '2025-02-15'),
    ('US-Southeast', 'Widget', 125.00, '2025-03-01'),
    ('Africa', 'Gadget', 90.00, '2025-03-05');  -- Goes to default

-- Verify distribution
SELECT
    tableoid::regclass AS partition,
    COUNT(*) AS rows
FROM regional_sales
GROUP BY tableoid;

-- Only scans US-East partition
EXPLAIN ANALYZE
SELECT * FROM regional_sales WHERE region = 'US-East';

-- ============================================================================
-- 4. Hash Partitioning (even distribution)
-- ============================================================================
-- Why: Hash partitioning distributes rows evenly across partitions using a hash
-- function on the key. Use it when there is no natural range or category to
-- partition by, but you still want to split a large table for parallel query
-- execution and reduced per-partition index sizes.
CREATE TABLE sensor_data (
    sensor_id INTEGER NOT NULL,
    reading NUMERIC(10, 4) NOT NULL,
    recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) PARTITION BY HASH (sensor_id);

-- Create 4 hash partitions
CREATE TABLE sensor_data_p0 PARTITION OF sensor_data
    FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE sensor_data_p1 PARTITION OF sensor_data
    FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE sensor_data_p2 PARTITION OF sensor_data
    FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE sensor_data_p3 PARTITION OF sensor_data
    FOR VALUES WITH (MODULUS 4, REMAINDER 3);

-- Insert test data
INSERT INTO sensor_data (sensor_id, reading, recorded_at)
SELECT
    i % 100,
    RANDOM() * 100,
    CURRENT_TIMESTAMP - (i * INTERVAL '1 minute')
FROM generate_series(1, 10000) AS i;

-- Verify even distribution
SELECT
    tableoid::regclass AS partition,
    COUNT(*) AS rows
FROM sensor_data
GROUP BY tableoid
ORDER BY partition;

-- ============================================================================
-- 5. Sub-partitioning (Multi-level)
-- ============================================================================

CREATE TABLE orders_partitioned (
    order_id BIGSERIAL,
    region TEXT NOT NULL,
    amount NUMERIC(10, 2) NOT NULL,
    order_date DATE NOT NULL
) PARTITION BY LIST (region);

-- Why: Sub-partitioning (region -> date) enables pruning on BOTH dimensions.
-- A query with WHERE region = 'US' AND order_date = '2025-03-01' scans only
-- one sub-partition instead of the entire table. Use sub-partitioning when
-- queries consistently filter on two independent columns.
CREATE TABLE orders_us PARTITION OF orders_partitioned
    FOR VALUES IN ('US')
    PARTITION BY RANGE (order_date);  -- Second level: by date

CREATE TABLE orders_eu PARTITION OF orders_partitioned
    FOR VALUES IN ('EU')
    PARTITION BY RANGE (order_date);

-- Sub-partitions for US
CREATE TABLE orders_us_2025_q1 PARTITION OF orders_us
    FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');
CREATE TABLE orders_us_2025_q2 PARTITION OF orders_us
    FOR VALUES FROM ('2025-04-01') TO ('2025-07-01');

-- Sub-partitions for EU
CREATE TABLE orders_eu_2025_q1 PARTITION OF orders_eu
    FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');
CREATE TABLE orders_eu_2025_q2 PARTITION OF orders_eu
    FOR VALUES FROM ('2025-04-01') TO ('2025-07-01');

-- Insert data
INSERT INTO orders_partitioned (region, amount, order_date) VALUES
    ('US', 100.00, '2025-01-15'),
    ('US', 200.00, '2025-05-10'),
    ('EU', 150.00, '2025-02-20'),
    ('EU', 300.00, '2025-06-01');

-- Both region AND date pruning applied
EXPLAIN ANALYZE
SELECT * FROM orders_partitioned
WHERE region = 'US' AND order_date >= '2025-01-01' AND order_date < '2025-04-01';

-- ============================================================================
-- 6. Partition Maintenance
-- ============================================================================

-- Add a new partition
CREATE TABLE events_2025_07 PARTITION OF events
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');

-- Why: DETACH converts a partition back into a standalone table without deleting
-- any data. This is the standard pattern for data archival — detach old partitions,
-- optionally compress or move them to cheaper storage, and DROP when they expire.
-- Much faster than DELETE (which generates WAL and dead tuples).
ALTER TABLE events DETACH PARTITION events_2025_01;

-- The detached table is now a regular table
SELECT COUNT(*) FROM events_2025_01;  -- Still accessible

-- Re-attach
ALTER TABLE events ATTACH PARTITION events_2025_01
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- Drop a partition (deletes data)
-- DROP TABLE events_2025_07;

-- ============================================================================
-- 7. Partition Information Queries
-- ============================================================================

-- List all partitions of a table
SELECT
    parent.relname AS parent_table,
    child.relname AS partition,
    pg_get_expr(child.relpartbound, child.oid) AS partition_bounds
FROM pg_inherits
JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
JOIN pg_class child ON pg_inherits.inhrelid = child.oid
WHERE parent.relname = 'events'
ORDER BY child.relname;

-- Partition sizes
SELECT
    child.relname AS partition,
    pg_size_pretty(pg_relation_size(child.oid)) AS size,
    pg_stat_get_live_tuples(child.oid) AS live_rows
FROM pg_inherits
JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
JOIN pg_class child ON pg_inherits.inhrelid = child.oid
WHERE parent.relname = 'events'
ORDER BY child.relname;

-- ============================================================================
-- 8. Partitioning Best Practices
-- ============================================================================

-- When to partition:
--   - Table > 100GB or millions of rows
--   - Queries consistently filter on partition key
--   - Need fast data archival (drop/detach old partitions)
--   - Time-series data with rolling retention

-- Partition key selection:
--   - Choose columns used in WHERE clauses
--   - Avoid too many partitions (< 1000 recommended)
--   - Ensure partition key is part of PRIMARY KEY / UNIQUE constraints

-- Limitations:
--   - PRIMARY KEY must include partition key columns
--   - Foreign keys referencing partitioned tables not supported (PG < 12)
--   - Cross-partition unique constraints not enforced (use application logic)

-- Performance tips:
--   - Always include partition key in queries for pruning
--   - Use CONCURRENTLY for index creation on large partitions
--   - Monitor partition sizes for even distribution
--   - Automate partition creation with cron or pg_partman extension
