# 18. Table Partitioning

**Previous**: [Window Functions](./17_Window_Functions.md) | **Next**: [Full-Text Search](./19_Full_Text_Search.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the concept of table partitioning and identify when it provides meaningful performance benefits
2. Create range-partitioned tables for time-series data with proper primary key and index design
3. Implement list partitioning for categorical data such as regions or statuses
4. Set up hash partitioning for even data distribution when no natural range or category exists
5. Verify partition pruning behavior using EXPLAIN ANALYZE and avoid common pitfalls that defeat pruning
6. Manage partitions dynamically including adding, detaching, dropping, and automating creation with pg_cron
7. Convert an existing non-partitioned table to a partitioned table with minimal downtime

---

As tables grow from millions to billions of rows, even well-indexed queries can slow down because indexes themselves become large and maintenance operations like VACUUM and backup take longer. Partitioning splits a single logical table into smaller physical pieces, each containing a subset of the data. This lets PostgreSQL scan only the relevant partitions, maintain smaller indexes, and drop entire partitions instantly instead of running expensive DELETE statements. For any system dealing with time-series data, event logs, or high-volume transactional data, partitioning is an essential scaling strategy.

## Table of Contents
1. [Partitioning Overview](#1-partitioning-overview)
2. [Range Partitioning](#2-range-partitioning)
3. [List Partitioning](#3-list-partitioning)
4. [Hash Partitioning](#4-hash-partitioning)
5. [Partition Pruning](#5-partition-pruning)
6. [Partition Management](#6-partition-management)
7. [Practice Problems](#7-practice-problems)

---

## 1. Partitioning Overview

### 1.1 What is Partitioning?

```
┌─────────────────────────────────────────────────────────────────┐
│                    Table Partitioning Concept                    │
│                                                                 │
│   Regular Table                 Partitioned Table               │
│   ┌───────────────┐            ┌───────────────┐               │
│   │   orders      │            │ orders (parent)│               │
│   │   (100M rows) │            └───────┬───────┘               │
│   │               │                    │                       │
│   │   All data    │            ┌───────┼───────┐               │
│   │   one file    │            │       │       │               │
│   └───────────────┘        ┌───┴───┐ ┌─┴──┐ ┌──┴──┐           │
│                            │2024_Q1│ │Q2  │ │ Q3  │ ...       │
│                            │ 25M   │ │    │ │     │           │
│                            └───────┘ └────┘ └─────┘           │
│                            (split storage)                      │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Advantages of Partitioning

```
┌─────────────────┬───────────────────────────────────────────────┐
│ Advantage       │ Description                                    │
├─────────────────┼───────────────────────────────────────────────┤
│ Query Performance│ Reduce scan range with partition pruning     │
│ Easy Maintenance │ VACUUM, backup, delete by partition          │
│ Data Archiving   │ Move old partitions to separate tablespace   │
│ Bulk Delete      │ Fast deletion with DROP PARTITION (vs DELETE)│
│ Index Size       │ Smaller indexes per partition (memory efficient)│
│ Parallel Processing│ Parallel scan by partition                 │
└─────────────────┴───────────────────────────────────────────────┘
```

### 1.3 Partitioning Types

```
┌─────────────────────────────────────────────────────────────────┐
│                    Partitioning Types                            │
│                                                                 │
│   Range: based on continuous range                              │
│   ├── By date (monthly, quarterly, yearly)                      │
│   └── By number range (ID range, amount range)                  │
│                                                                 │
│   List: based on discrete value list                            │
│   ├── Region (country, city)                                    │
│   ├── Status (active, inactive, pending)                        │
│   └── Category                                                  │
│                                                                 │
│   Hash: based on hash value                                     │
│   └── When even distribution needed (no specific criteria)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Range Partitioning

### 2.1 Basic Structure

```sql
-- Partition key choice is critical: pick the column most frequently used in WHERE clauses.
-- order_date is ideal here because queries almost always filter by time range,
-- enabling partition pruning to skip irrelevant months entirely.
CREATE TABLE orders (
    id BIGSERIAL,
    customer_id INT NOT NULL,
    order_date DATE NOT NULL,
    amount NUMERIC(10,2),
    status VARCHAR(20)
) PARTITION BY RANGE (order_date);

-- Create partitions (monthly)
CREATE TABLE orders_2024_01 PARTITION OF orders
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE orders_2024_02 PARTITION OF orders
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

CREATE TABLE orders_2024_03 PARTITION OF orders
    FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');

-- Default partition (for data not matching ranges)
CREATE TABLE orders_default PARTITION OF orders DEFAULT;
```

### 2.2 Create Indexes

```sql
-- Create index on parent table (automatically applied to partitions)
CREATE INDEX idx_orders_customer ON orders (customer_id);
CREATE INDEX idx_orders_date ON orders (order_date);

-- Check individual partition indexes
SELECT
    schemaname,
    tablename,
    indexname
FROM pg_indexes
WHERE tablename LIKE 'orders%';
```

### 2.3 PRIMARY KEY and UNIQUE Constraints

```sql
-- PK/UNIQUE in partitioned tables must include partition key
CREATE TABLE orders (
    id BIGSERIAL,
    order_date DATE NOT NULL,
    customer_id INT NOT NULL,
    amount NUMERIC(10,2),
    PRIMARY KEY (id, order_date)  -- include partition key
) PARTITION BY RANGE (order_date);

-- Composite UNIQUE constraint
ALTER TABLE orders ADD CONSTRAINT orders_unique
    UNIQUE (id, order_date);
```

### 2.4 Quarterly Partitioning Example

```sql
-- Quarterly partitions
CREATE TABLE sales (
    id BIGSERIAL,
    sale_date DATE NOT NULL,
    product_id INT,
    amount NUMERIC(10,2),
    PRIMARY KEY (id, sale_date)
) PARTITION BY RANGE (sale_date);

-- 2024 quarterly partitions
CREATE TABLE sales_2024_q1 PARTITION OF sales
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');
CREATE TABLE sales_2024_q2 PARTITION OF sales
    FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');
CREATE TABLE sales_2024_q3 PARTITION OF sales
    FOR VALUES FROM ('2024-07-01') TO ('2024-10-01');
CREATE TABLE sales_2024_q4 PARTITION OF sales
    FOR VALUES FROM ('2024-10-01') TO ('2025-01-01');
```

---

## 3. List Partitioning

### 3.1 Regional Partitioning

```sql
-- Regional partitions
CREATE TABLE customers (
    id SERIAL,
    name VARCHAR(100),
    email VARCHAR(255),
    region VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (id, region)
) PARTITION BY LIST (region);

-- Continental partitions
CREATE TABLE customers_asia PARTITION OF customers
    FOR VALUES IN ('KR', 'JP', 'CN', 'SG', 'IN');

CREATE TABLE customers_europe PARTITION OF customers
    FOR VALUES IN ('UK', 'DE', 'FR', 'IT', 'ES');

CREATE TABLE customers_americas PARTITION OF customers
    FOR VALUES IN ('US', 'CA', 'MX', 'BR');

CREATE TABLE customers_others PARTITION OF customers DEFAULT;
```

### 3.2 Status-based Partitioning

```sql
-- Order status partitions
CREATE TABLE order_items (
    id BIGSERIAL,
    order_id BIGINT,
    status VARCHAR(20) NOT NULL,
    product_id INT,
    quantity INT,
    PRIMARY KEY (id, status)
) PARTITION BY LIST (status);

CREATE TABLE order_items_pending PARTITION OF order_items
    FOR VALUES IN ('pending', 'processing');

CREATE TABLE order_items_completed PARTITION OF order_items
    FOR VALUES IN ('shipped', 'delivered');

CREATE TABLE order_items_cancelled PARTITION OF order_items
    FOR VALUES IN ('cancelled', 'refunded');
```

### 3.3 Multi-column List Partitioning

```sql
-- PostgreSQL 11+ multi-column partition
CREATE TABLE events (
    id BIGSERIAL,
    event_type VARCHAR(20) NOT NULL,
    event_date DATE NOT NULL,
    data JSONB,
    PRIMARY KEY (id, event_type, event_date)
) PARTITION BY LIST (event_type);

-- Event type partition → Range subpartition inside
CREATE TABLE events_click PARTITION OF events
    FOR VALUES IN ('click')
    PARTITION BY RANGE (event_date);

CREATE TABLE events_click_2024_01 PARTITION OF events_click
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

---

## 4. Hash Partitioning

### 4.1 Basic Hash Partitioning

```sql
-- Hash partitioning (even distribution)
CREATE TABLE logs (
    id BIGSERIAL,
    user_id INT NOT NULL,
    action VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (id, user_id)
) PARTITION BY HASH (user_id);

-- Distribute into 4 partitions
CREATE TABLE logs_p0 PARTITION OF logs
    FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE logs_p1 PARTITION OF logs
    FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE logs_p2 PARTITION OF logs
    FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE logs_p3 PARTITION OF logs
    FOR VALUES WITH (MODULUS 4, REMAINDER 3);
```

### 4.2 Automate Hash Partition Creation

```sql
-- Dynamic partition creation function
CREATE OR REPLACE FUNCTION create_hash_partitions(
    parent_table TEXT,
    num_partitions INT
) RETURNS VOID AS $$
DECLARE
    i INT;
BEGIN
    FOR i IN 0..num_partitions-1 LOOP
        EXECUTE format(
            'CREATE TABLE %I PARTITION OF %I FOR VALUES WITH (MODULUS %s, REMAINDER %s)',
            parent_table || '_p' || i,
            parent_table,
            num_partitions,
            i
        );
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Usage
SELECT create_hash_partitions('logs', 8);
```

### 4.3 Hash vs Range/List Selection Criteria

```
┌─────────────────────────────────────────────────────────────────┐
│                    Partitioning Type Selection Guide             │
│                                                                 │
│   Choose Range:                                                 │
│   - Time-based data (logs, transactions)                        │
│   - Frequent range queries                                      │
│   - Need to archive/delete old data                             │
│                                                                 │
│   Choose List:                                                  │
│   - Clear categorical distinctions                              │
│   - Region, status, type and other discrete values              │
│   - Frequently query specific categories only                   │
│                                                                 │
│   Choose Hash:                                                  │
│   - No clear classification criteria                            │
│   - Goal is even data distribution                              │
│   - Range queries not needed                                    │
│   - Fixed number of partitions                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Partition Pruning

### 5.1 Verify Pruning Behavior

```sql
-- Partition pruning is the key performance win: the planner eliminates partitions that
-- cannot contain matching rows BEFORE execution begins, turning a 12-partition table
-- into a single-partition scan when the WHERE clause matches one month.
EXPLAIN (ANALYZE, COSTS OFF)
SELECT * FROM orders
WHERE order_date = '2024-02-15';

-- Example result:
-- Append
--   ->  Seq Scan on orders_2024_02  -- scan only February partition
--         Filter: (order_date = '2024-02-15'::date)
```

### 5.2 Pruning Configuration

```sql
-- Check pruning enabled
SHOW enable_partition_pruning;  -- on (default)

-- Runtime pruning (in joins, subqueries)
SET enable_partition_pruning = on;
```

### 5.3 Cases Where Pruning Fails

```sql
-- 1. Functions on the partition key defeat pruning — the planner cannot infer which
-- partitions to skip when the key is wrapped in EXTRACT() or other functions
SELECT * FROM orders
WHERE EXTRACT(YEAR FROM order_date) = 2024;

-- Rewrite as a range predicate so the planner can match partition boundaries directly
SELECT * FROM orders
WHERE order_date >= '2024-01-01' AND order_date < '2025-01-01';

-- 2. Implicit type conversion
-- Bad example (string comparison)
SELECT * FROM orders WHERE order_date = '2024-02-15';  -- string

-- Good example (explicit type)
SELECT * FROM orders WHERE order_date = DATE '2024-02-15';

-- 3. Partial pruning with OR conditions
SELECT * FROM orders
WHERE order_date = '2024-01-15' OR customer_id = 123;
-- customer_id condition causes scan of all partitions
```

### 5.4 Partition Exclusion Hints

```sql
-- Direct partition reference
SELECT * FROM orders_2024_02  -- direct partition reference
WHERE customer_id = 123;

-- constraint_exclusion setting
SET constraint_exclusion = partition;  -- default
```

---

## 6. Partition Management

### 6.1 Add Partition

```sql
-- Add new partition
CREATE TABLE orders_2024_04 PARTITION OF orders
    FOR VALUES FROM ('2024-04-01') TO ('2024-05-01');

-- Or attach existing table as partition
CREATE TABLE orders_2024_05 (LIKE orders INCLUDING ALL);
ALTER TABLE orders ATTACH PARTITION orders_2024_05
    FOR VALUES FROM ('2024-05-01') TO ('2024-06-01');
```

### 6.2 Detach and Drop Partition

```sql
-- Detach partition (preserve data, independent table)
ALTER TABLE orders DETACH PARTITION orders_2024_01;

-- Detached table exists independently
SELECT * FROM orders_2024_01;

-- Drop partition (delete data too)
DROP TABLE orders_2024_01;
```

### 6.3 Automatic Partition Creation

```sql
-- Monthly partition auto-creation function
CREATE OR REPLACE FUNCTION create_monthly_partition(
    parent_table TEXT,
    partition_date DATE
) RETURNS VOID AS $$
DECLARE
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    start_date := DATE_TRUNC('month', partition_date);
    end_date := start_date + INTERVAL '1 month';
    partition_name := parent_table || '_' || TO_CHAR(start_date, 'YYYY_MM');

    -- Skip if already exists
    IF NOT EXISTS (
        SELECT 1 FROM pg_tables WHERE tablename = partition_name
    ) THEN
        EXECUTE format(
            'CREATE TABLE %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
            partition_name,
            parent_table,
            start_date,
            end_date
        );
        RAISE NOTICE 'Created partition: %', partition_name;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Pre-create partitions for next 3 months
DO $$
BEGIN
    FOR i IN 0..2 LOOP
        PERFORM create_monthly_partition(
            'orders',
            CURRENT_DATE + (i || ' months')::interval
        );
    END LOOP;
END;
$$;
```

### 6.4 Automation with pg_cron

```sql
-- Install pg_cron extension (requires separate installation)
CREATE EXTENSION pg_cron;

-- Create new partition on 1st of each month
SELECT cron.schedule(
    'create-partition',
    '0 0 1 * *',  -- 1st of month at 00:00
    $$SELECT create_monthly_partition('orders', CURRENT_DATE + INTERVAL '2 months')$$
);

-- Auto-delete old partitions (12 months ago)
SELECT cron.schedule(
    'drop-old-partition',
    '0 1 1 * *',  -- 1st of month at 01:00
    $$DROP TABLE IF EXISTS orders_$$ || TO_CHAR(CURRENT_DATE - INTERVAL '12 months', 'YYYY_MM')
);
```

### 6.5 Query Partition Information

```sql
-- List partitions and ranges
SELECT
    parent.relname AS parent,
    child.relname AS partition,
    pg_get_expr(child.relpartbound, child.oid) AS bounds
FROM pg_inherits
JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
JOIN pg_class child ON pg_inherits.inhrelid = child.oid
WHERE parent.relname = 'orders';

-- Row count per partition
SELECT
    schemaname,
    relname AS partition_name,
    n_live_tup AS row_count
FROM pg_stat_user_tables
WHERE relname LIKE 'orders_%'
ORDER BY relname;

-- Size per partition
SELECT
    child.relname AS partition,
    pg_size_pretty(pg_relation_size(child.oid)) AS size
FROM pg_inherits
JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
JOIN pg_class child ON pg_inherits.inhrelid = child.oid
WHERE parent.relname = 'orders'
ORDER BY child.relname;
```

### 6.6 Convert Existing Table to Partitioned

```sql
-- 1. Create new partitioned table
CREATE TABLE orders_new (LIKE orders INCLUDING ALL)
    PARTITION BY RANGE (order_date);

-- 2. Create partitions
CREATE TABLE orders_new_2024_01 PARTITION OF orders_new
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
-- ... create needed partitions

-- 3. Migrate data
INSERT INTO orders_new SELECT * FROM orders;

-- 4. Swap tables (minimize downtime)
BEGIN;
ALTER TABLE orders RENAME TO orders_old;
ALTER TABLE orders_new RENAME TO orders;
COMMIT;

-- 5. Drop old table after verification
DROP TABLE orders_old;
```

---

## 7. Practice Problems

### Exercise 1: Monthly Log Partitioning
Partition access_logs table by month.

```sql
-- Example answer
CREATE TABLE access_logs (
    id BIGSERIAL,
    user_id INT,
    action VARCHAR(50),
    ip_address INET,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- 2024 monthly partitions
DO $$
DECLARE
    start_date DATE := '2024-01-01';
BEGIN
    FOR i IN 0..11 LOOP
        EXECUTE format(
            'CREATE TABLE access_logs_%s PARTITION OF access_logs
             FOR VALUES FROM (%L) TO (%L)',
            TO_CHAR(start_date + (i || ' months')::interval, 'YYYY_MM'),
            start_date + (i || ' months')::interval,
            start_date + ((i+1) || ' months')::interval
        );
    END LOOP;
END;
$$;
```

### Exercise 2: Regional Order Partitioning
Partition orders based on country code.

```sql
-- Example answer
CREATE TABLE regional_orders (
    id BIGSERIAL,
    country_code CHAR(2) NOT NULL,
    customer_id INT,
    total NUMERIC(10,2),
    order_date TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (id, country_code)
) PARTITION BY LIST (country_code);

CREATE TABLE regional_orders_kr PARTITION OF regional_orders
    FOR VALUES IN ('KR');
CREATE TABLE regional_orders_us PARTITION OF regional_orders
    FOR VALUES IN ('US');
CREATE TABLE regional_orders_others PARTITION OF regional_orders DEFAULT;
```

### Exercise 3: Partition Maintenance Query
Write a query to identify and handle partitions with data older than 90 days.

```sql
-- Example answer: identify old partitions
WITH partition_info AS (
    SELECT
        child.relname AS partition_name,
        pg_get_expr(child.relpartbound, child.oid) AS bounds,
        (regexp_match(
            pg_get_expr(child.relpartbound, child.oid),
            $$FROM \('([^']+)'\)$$
        ))[1]::date AS start_date
    FROM pg_inherits
    JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
    JOIN pg_class child ON pg_inherits.inhrelid = child.oid
    WHERE parent.relname = 'orders'
      AND child.relname != 'orders_default'
)
SELECT *
FROM partition_info
WHERE start_date < CURRENT_DATE - INTERVAL '90 days';
```

---

## References
- [PostgreSQL Table Partitioning](https://www.postgresql.org/docs/current/ddl-partitioning.html)
- [Partition Pruning](https://www.postgresql.org/docs/current/ddl-partitioning.html#DDL-PARTITION-PRUNING)
- [pg_partman Extension](https://github.com/pgpartman/pg_partman)
- [Best Practices for Partitioning](https://www.postgresql.org/docs/current/ddl-partitioning.html#DDL-PARTITIONING-OVERVIEW)

---

**Previous**: [Window Functions](./17_Window_Functions.md) | **Next**: [Full-Text Search](./19_Full_Text_Search.md)
