# 15. Advanced PostgreSQL Query Optimization

**Previous**: [JSON/JSONB Features](./14_JSON_JSONB.md) | **Next**: [Replication and High Availability](./16_Replication_HA.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Read and interpret EXPLAIN ANALYZE output including cost estimates, actual times, buffer statistics, and loop counts
2. Describe how the PostgreSQL query planner transforms SQL into an execution plan
3. Select the appropriate index type (B-tree, Hash, GIN, GiST, BRIN) for different query patterns
4. Design effective composite, partial, and covering indexes
5. Compare Nested Loop, Hash Join, and Merge Join algorithms and predict when each is chosen
6. Use table statistics and cost parameters to understand and influence planner decisions
7. Apply advanced optimization techniques including query refactoring, materialized views, and partitioning

---

A query that runs in 5 milliseconds versus 5 seconds can be the difference between a responsive application and a frustrated user. PostgreSQL's query optimizer is remarkably sophisticated, but it relies on accurate statistics, well-chosen indexes, and properly structured queries to do its best work. Understanding how the planner thinks -- and how to read the execution plans it produces -- gives you the power to diagnose and fix performance bottlenecks systematically rather than guessing.

## Table of Contents

1. [EXPLAIN ANALYZE Deep Dive](#1-explain-analyze-deep-dive)
2. [Query Planner](#2-query-planner)
3. [Index Strategies](#3-index-strategies)
4. [Join Optimization](#4-join-optimization)
5. [Statistics and Cost Estimation](#5-statistics-and-cost-estimation)
6. [Advanced Optimization Techniques](#6-advanced-optimization-techniques)
7. [Practice Problems](#7-practice-problems)

---

## 1. EXPLAIN ANALYZE Deep Dive

> **Analogy -- The Query Optimizer as GPS**: Just as a GPS evaluates multiple routes (highway vs. side streets vs. toll road) and picks the fastest based on current conditions, PostgreSQL's query planner evaluates multiple execution strategies (sequential scan, index scan, hash join, merge join) and chooses the one with the lowest estimated cost. Understanding EXPLAIN output is like reading the GPS's chosen route -- it tells you exactly which "roads" the database will take to reach your data.

### Theory: The Plan Tree and EXPLAIN

The planner's output is a tree of plan nodes. The leaves are scans (Seq Scan, Index Scan, …); interior nodes are operations (Nested Loop, Hash Join, Sort, Aggregate, …); the root produces the final result rows.

#### C.1 Reading EXPLAIN output

```
Sort  (cost=22.07..22.32 rows=100 width=64) (actual time=0.305..0.312 rows=100 loops=1)
  Sort Key: created_at DESC
  ->  Hash Join  (cost=10.00..18.50 rows=100 width=64) (actual time=0.150..0.270 rows=100 loops=1)
        Hash Cond: (orders.user_id = users.id)
        ->  Seq Scan on orders  (cost=0.00..7.00 rows=500 width=32) (actual time=0.005..0.080 rows=500 loops=1)
        ->  Hash  (cost=8.00..8.00 rows=200 width=32) (actual time=0.040..0.040 rows=200 loops=1)
              ->  Seq Scan on users  (cost=0.00..8.00 rows=200 width=32) (actual time=0.002..0.020 rows=200 loops=1)
```

Each line shows:

- **Node type** (Sort, Hash Join, Seq Scan, …).
- **`cost=startup..total`** — startup cost is "before first row", total cost is "for all rows".
- **`rows=N`** — estimated row count.
- **`width=W`** — average row width in bytes.
- **`actual time=...`** — only with `EXPLAIN ANALYZE` — actual milliseconds.
- **`actual rows=N`** — actual row count.
- **`loops=N`** — how many times this node was executed (in inner loops of nested joins).

#### C.2 What to look for

- **Estimated `rows` vs `actual rows`** — large discrepancies indicate stale or insufficient statistics.
- **Costs much higher than nearby alternatives** — sometimes a small index can drop the cost by 2-3 orders of magnitude.
- **`Rows Removed by Filter`** — a high number means the scan returned many rows that the filter discarded. Often fixable by an index covering the filter column.
- **`Heap Fetches`** in an index-only scan — non-zero means the visibility map is stale; a `VACUUM` may help.

### 1.1 EXPLAIN Options

```sql
-- Basic execution plan
EXPLAIN SELECT * FROM users WHERE id = 1;

-- Actual execution + timing
EXPLAIN ANALYZE SELECT * FROM users WHERE id = 1;

-- Include buffer information
EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM users WHERE id = 1;

-- Detailed output
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) SELECT ...;
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT ...;
EXPLAIN (ANALYZE, BUFFERS, FORMAT YAML) SELECT ...;

-- Plan only (without ANALYZE)
EXPLAIN (COSTS, VERBOSE) SELECT * FROM users;

-- Disable timing (reduce overhead)
EXPLAIN (ANALYZE, TIMING OFF) SELECT * FROM users;

-- Include settings
EXPLAIN (ANALYZE, SETTINGS) SELECT * FROM users;
```

### 1.2 Reading Execution Plans

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT u.name, COUNT(o.id)
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at > '2024-01-01'
GROUP BY u.name;

/*
HashAggregate  (cost=1234.56..1234.78 rows=100 width=40)
               (actual time=45.123..45.456 loops=1)
  Group Key: u.name
  Batches: 1  Memory Usage: 24kB
  Buffers: shared hit=500 read=100
  ->  Hash Right Join  (cost=100.00..1200.00 rows=5000 width=36)
                       (actual time=5.123..40.456 loops=1)
        Hash Cond: (o.user_id = u.id)
        Buffers: shared hit=400 read=80
        ->  Seq Scan on orders o  (cost=0.00..800.00 rows=30000 width=8)
                                  (actual time=0.015..15.123 loops=1)
              Buffers: shared hit=300 read=50
        ->  Hash  (cost=80.00..80.00 rows=1000 width=36)
                  (actual time=3.456..3.456 loops=1)
              Buckets: 1024  Batches: 1  Memory Usage: 72kB
              Buffers: shared hit=100 read=30
              ->  Index Scan using idx_users_created on users u
                  (cost=0.29..80.00 rows=1000 width=36)
                  (actual time=0.030..2.345 loops=1)
                    Index Cond: (created_at > '2024-01-01')
                    Buffers: shared hit=100 read=30
Planning Time: 0.456 ms
Execution Time: 46.789 ms
*/
```

### 1.3 Key Metrics Interpretation

```
┌─────────────────────────────────────────────────────────────┐
│              Execution Plan Metrics Interpretation           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  cost=startup_cost..total_cost                              │
│  • Startup cost: cost until first row                       │
│  • Total cost: cost until all rows                          │
│  • Unit: abstract cost units                                │
│                                                             │
│  rows=estimated_rows                                        │
│  • Planner's estimated row count                            │
│                                                             │
│  width=row_width                                            │
│  • Average bytes per row                                    │
│                                                             │
│  actual time=start..end                                     │
│  • Actual execution time (milliseconds)                     │
│                                                             │
│  loops=loop_count                                           │
│  • Number of times node was executed                        │
│  • Actual time = time × loops                               │
│                                                             │
│  Buffers:                                                   │
│  • shared hit: blocks read from cache                       │
│  • shared read: blocks read from disk                       │
│  • shared written: blocks written to disk                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.4 Problem Identification

```sql
-- Problem: Estimated vs actual row count difference
-- Expected: rows=100, Actual: rows=10000
-- Cause: Outdated statistics, ANALYZE needed

ANALYZE users;

-- Problem: High startup cost
-- Occurs in Sort, Hash operations
-- Solution: Add appropriate index

-- Problem: High loops in Nested Loop
-- Solution: Change JOIN method or add index

-- Problem: Seq Scan on large table
-- Solution: Add appropriate index
```

---

## 2. Query Planner

### Theory: The Cost Model

Every plan node has a cost expressed as a unit-less number. The planner picks the plan with the lowest **total cost**. The cost is a weighted sum of three things: pages read sequentially, pages read randomly, and CPU time per row processed.

#### A.1 The constants

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `seq_page_cost` | 1.0 | Cost to read one page sequentially |
| `random_page_cost` | 4.0 | Cost to read one page at a random offset |
| `cpu_tuple_cost` | 0.01 | Cost to process one tuple |
| `cpu_index_tuple_cost` | 0.005 | Cost to process one index entry |
| `cpu_operator_cost` | 0.0025 | Cost of an operator/function call |
| `parallel_tuple_cost` | 0.1 | Cost of transferring one tuple from worker to leader |
| `parallel_setup_cost` | 1000 | Cost of starting parallel workers |

These are *relative* costs. The defaults assume HDD-era random I/O is 4× more expensive than sequential I/O. **For SSD-backed storage, lower `random_page_cost` to ~1.1**. This single change can swing the planner from sequential scans (which were optimal on spinning disks) toward index scans (which are optimal when random I/O is cheap).

#### A.2 The formula for a sequential scan

```
seq_scan_cost = seq_page_cost × pages_in_table
              + cpu_tuple_cost × rows_in_table
              + cpu_operator_cost × rows_in_table × operators_per_row
```

For a 10,000-page table with 1,000,000 rows, that is `1.0 × 10000 + 0.01 × 1000000 + ... ≈ 20000`.

#### A.3 The formula for an index scan

```
index_scan_cost = index_pages_read × random_page_cost      (descend the B-tree)
                + cpu_index_tuple_cost × matching_index_entries
                + matching_rows × random_page_cost          (heap fetches)
                + cpu_tuple_cost × matching_rows
                + cpu_operator_cost × matching_rows
```

The `matching_rows × random_page_cost` term is what makes index scans expensive at high selectivity — at 10% of a 10,000-page table, that is `100,000 × 4.0 = 400,000` versus the sequential scan's `20,000`. The planner correctly picks the sequential scan in that range.

### 2.1 Planner Process

```
┌─────────────────────────────────────────────────────────────┐
│                    Query Planner Process                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  SQL Query                                                  │
│      │                                                      │
│      ▼                                                      │
│  ┌─────────┐                                               │
│  │ Parser  │ → Parse syntax → Parse Tree                   │
│  └─────────┘                                               │
│      │                                                      │
│      ▼                                                      │
│  ┌─────────┐                                               │
│  │Analyzer │ → Semantic analysis → Query Tree              │
│  └─────────┘                                               │
│      │                                                      │
│      ▼                                                      │
│  ┌─────────┐                                               │
│  │Rewriter │ → Apply rules (VIEW, etc)                     │
│  └─────────┘                                               │
│      │                                                      │
│      ▼                                                      │
│  ┌─────────┐    ┌──────────────┐                          │
│  │Planner  │◄───│  Statistics  │                          │
│  └─────────┘    └──────────────┘                          │
│      │                                                      │
│      ▼ Select optimal execution plan                       │
│  ┌─────────┐                                               │
│  │Executor │ → Execute → Result                            │
│  └─────────┘                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Planner Configuration

```sql
-- Check planner settings
SHOW seq_page_cost;      -- Sequential page read cost (default 1.0)
SHOW random_page_cost;   -- Random page read cost (default 4.0)
SHOW cpu_tuple_cost;     -- Tuple processing cost (default 0.01)
SHOW cpu_index_tuple_cost;
SHOW cpu_operator_cost;

-- Lower random_page_cost for SSD
SET random_page_cost = 1.1;

-- Disable specific plans (for testing)
SET enable_seqscan = off;
SET enable_indexscan = off;
SET enable_bitmapscan = off;
SET enable_hashjoin = off;
SET enable_mergejoin = off;
SET enable_nestloop = off;

-- Parallel query settings
SET max_parallel_workers_per_gather = 4;
SET parallel_tuple_cost = 0.01;
SET parallel_setup_cost = 1000;
```

### 2.3 Planner Hints (pg_hint_plan)

```sql
-- pg_hint_plan extension installation required
CREATE EXTENSION pg_hint_plan;

-- Index hint
/*+ IndexScan(users idx_users_email) */
SELECT * FROM users WHERE email = 'test@example.com';

-- Join order hint
/*+ Leading(orders users) */
SELECT * FROM users u JOIN orders o ON u.id = o.user_id;

-- Join method hint
/*+ HashJoin(users orders) */
SELECT * FROM users u JOIN orders o ON u.id = o.user_id;

/*+ NestLoop(users orders) */
SELECT * FROM users u JOIN orders o ON u.id = o.user_id;

-- Force Seq Scan
/*+ SeqScan(users) */
SELECT * FROM users WHERE id > 100;

-- Disable parallel query
/*+ Parallel(users 0) */
SELECT COUNT(*) FROM users;
```

---

## 3. Index Strategies

### 3.1 Index Type Selection

```sql
-- B-tree (default, most cases)
CREATE INDEX idx_users_email ON users(email);

-- Suitable for: =, <, >, <=, >=, BETWEEN, IN, IS NULL
-- LIKE 'abc%' (prefix matching)

-- Hash (equality only)
CREATE INDEX idx_users_email_hash ON users USING HASH (email);
-- Suitable for: = only
-- WAL support in PostgreSQL 10+

-- GiST (geometry, ranges, full-text search)
CREATE INDEX idx_locations_point ON locations USING GIST (point);
CREATE INDEX idx_events_range ON events USING GIST (time_range);

-- GIN (arrays, JSONB, full-text search)
CREATE INDEX idx_posts_tags ON posts USING GIN (tags);
CREATE INDEX idx_products_attrs ON products USING GIN (attributes);
CREATE INDEX idx_docs_search ON documents USING GIN (to_tsvector('english', content));

-- BRIN (large sequential data)
CREATE INDEX idx_logs_time ON logs USING BRIN (created_at);
-- Suitable for: physically ordered data (time series, etc)
-- Very small size, effective for large tables
```

### 3.2 Composite Indexes

```sql
-- Composite index order matters!
CREATE INDEX idx_orders_user_date ON orders(user_id, created_at);

-- These queries can use the index:
SELECT * FROM orders WHERE user_id = 1;
SELECT * FROM orders WHERE user_id = 1 AND created_at > '2024-01-01';

-- This query cannot use the index (no first column):
SELECT * FROM orders WHERE created_at > '2024-01-01';

-- Sort optimization
CREATE INDEX idx_orders_user_date_desc ON orders(user_id, created_at DESC);

-- INCLUDE (covering index, PostgreSQL 11+)
CREATE INDEX idx_orders_covering ON orders(user_id)
INCLUDE (status, total);
-- Query can use index only (Index Only Scan)
```

### 3.3 Partial Indexes

```sql
-- Index on specific condition
CREATE INDEX idx_orders_pending ON orders(created_at)
WHERE status = 'pending';

-- Exclude NULL
CREATE INDEX idx_users_email_notnull ON users(email)
WHERE email IS NOT NULL;

-- Recent data only
CREATE INDEX idx_logs_recent ON logs(level, message)
WHERE created_at > '2024-01-01';

-- Non-deleted rows only
CREATE INDEX idx_active_products ON products(category_id)
WHERE deleted_at IS NULL;
```

### 3.4 Index Management

```sql
-- Index usage statistics
SELECT
    schemaname,
    relname AS table_name,
    indexrelname AS index_name,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Find unused indexes
SELECT
    schemaname || '.' || relname AS table,
    indexrelname AS index,
    pg_size_pretty(pg_relation_size(i.indexrelid)) AS size,
    idx_scan
FROM pg_stat_user_indexes ui
JOIN pg_index i ON ui.indexrelid = i.indexrelid
WHERE idx_scan = 0
AND NOT indisunique
ORDER BY pg_relation_size(i.indexrelid) DESC;

-- Find duplicate indexes
SELECT
    a.indrelid::regclass AS table_name,
    a.indexrelid::regclass AS index1,
    b.indexrelid::regclass AS index2
FROM pg_index a
JOIN pg_index b ON a.indrelid = b.indrelid
AND a.indexrelid < b.indexrelid
AND (
    (a.indkey::text LIKE b.indkey::text || '%')
    OR (b.indkey::text LIKE a.indkey::text || '%')
);

-- Reindex
REINDEX INDEX idx_users_email;
REINDEX TABLE users;
REINDEX DATABASE mydb CONCURRENTLY;  -- PostgreSQL 12+

-- Create index concurrently (minimize locking)
CREATE INDEX CONCURRENTLY idx_users_email ON users(email);
```

---

## 4. Join Optimization

### Theory: Join-Order Search and GEQO

For N-way joins, the planner must consider many possible join orderings.

#### D.1 The search space

For `N` tables, the number of *trees* to consider (left-deep, bushy, etc.) grows exponentially: roughly `(2N)! / N!` for bushy trees. For 10 tables, that is millions; for 20 tables, astronomical.

#### D.2 Dynamic programming (default)

The planner uses **bottom-up dynamic programming**: for each pair of tables, find the cheapest join; for each triple, find the cheapest extension; and so on. The cost of intermediate joins is reused, so the algorithm is O(2^N · N^2) — manageable for ~12 tables but exponential beyond.

#### D.3 GEQO — fallback for big joins

When `from_collapse_limit + join_collapse_limit` worth of joinable tables exceeds `geqo_threshold` (default 12), PostgreSQL switches to the **Genetic Query Optimizer**: it represents join orders as chromosomes, uses crossover and mutation to evolve a population, and runs for a configurable number of generations. The result is *not guaranteed optimal* but is much faster to compute than exhaustive DP for ≥20 tables.

You can disable GEQO with `SET geqo = off;` if you want exhaustive search at the cost of planning time, or raise `geqo_threshold` to push the boundary.

### 4.1 Join Method Comparison

```
┌─────────────────────────────────────────────────────────────┐
│                    Join Method Comparison                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Nested Loop Join                                           │
│  ─────────────────                                          │
│  for each row in outer:                                     │
│      for each row in inner:                                 │
│          if match: emit                                     │
│                                                             │
│  • Suitable: small tables, with index                       │
│  • Cost: O(N × M), O(N × log M) with index                 │
│                                                             │
│  Hash Join                                                  │
│  ─────────────────                                          │
│  build hash table from inner                                │
│  for each row in outer:                                     │
│      probe hash table                                       │
│                                                             │
│  • Suitable: large tables, equijoin                         │
│  • Cost: O(N + M)                                          │
│  • Requires memory (work_mem)                               │
│                                                             │
│  Merge Join                                                 │
│  ─────────────────                                          │
│  sort both tables                                           │
│  merge sorted lists                                         │
│                                                             │
│  • Suitable: already sorted data, range join               │
│  • Cost: O(N log N + M log M + N + M)                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Join Order Optimization

```sql
-- Join order greatly affects performance
-- Planner auto-optimizes but limited with many tables

-- Check join limits
SHOW join_collapse_limit;  -- default 8
SHOW from_collapse_limit;  -- default 8

-- Many table joins: order matters
-- Small tables / heavily filtered tables first

-- Good example: filter first
SELECT *
FROM orders o
JOIN users u ON o.user_id = u.id
WHERE o.status = 'pending'  -- filtering
AND o.created_at > '2024-01-01';

-- Explicit join order (for testing)
SET join_collapse_limit = 1;
SELECT * FROM t1, t2, t3
WHERE t1.id = t2.t1_id AND t2.id = t3.t2_id;
RESET join_collapse_limit;
```

### 4.3 Join Performance Improvement

```sql
-- Appropriate indexes
CREATE INDEX idx_orders_user ON orders(user_id);

-- Match join column types
-- Bad: orders.user_id (int) JOIN users.id (bigint) → type conversion
-- Good: use same type

-- Remove unnecessary joins
-- Bad
SELECT o.* FROM orders o
JOIN users u ON o.user_id = u.id;  -- nothing from users

-- Good (remove join)
SELECT o.* FROM orders o
WHERE EXISTS (SELECT 1 FROM users u WHERE u.id = o.user_id);

-- Convert subquery → join
-- Bad (correlated subquery)
SELECT *,
    (SELECT name FROM users WHERE id = o.user_id) AS user_name
FROM orders o;

-- Good
SELECT o.*, u.name AS user_name
FROM orders o
JOIN users u ON o.user_id = u.id;
```

---

## 5. Statistics and Cost Estimation

### Theory: Statistics — Where the Row Counts Come From

The cost formula needs to know `matching_rows` and `pages_in_table`. The planner gets these from `pg_statistic`, populated by `ANALYZE`.

#### B.1 What `ANALYZE` collects

For each column, `ANALYZE` samples some rows (`default_statistics_target` × 300, default 30000) and computes:

- **Number of distinct values** (`n_distinct`)
- **Most common values (MCVs)** — typically 100 most frequent values and their per-value frequency.
- **Histogram** — the remaining values divided into N buckets of equal frequency (default 100 buckets).
- **Null fraction** (`null_frac`)
- **Correlation** between physical row order and logical column order (used to estimate the cost of an index scan that produces ordered output).
- **Average column width** (`avg_width`).

Stored in `pg_statistic` (or the `pg_stats` view for human readability).

#### B.2 Selectivity estimation

For `WHERE col = value`:

- If `value` is in the MCV list, selectivity = its frequency. Exact.
- Otherwise, selectivity = `(1 - sum_of_MCV_frequencies - null_frac) / (n_distinct - count_of_MCVs)`. Average of the long tail.

For `WHERE col BETWEEN a AND b`:

- Selectivity = (number of histogram buckets between a and b) / total buckets, with linear interpolation inside the boundary buckets.

For `WHERE col1 = ? AND col2 = ?`:

- By default, the planner assumes the columns are **independent** and multiplies the per-column selectivities. This is the most common source of bad estimates — when columns are correlated (e.g., `country = 'KR' AND city = 'Seoul'`), the assumption produces wildly wrong row counts.

#### B.3 Extended statistics — fixing correlation underestimation

`CREATE STATISTICS s_country_city (dependencies, ndistinct) ON country, city FROM addresses;` tells the planner to track joint statistics for those columns. After the next `ANALYZE`, selectivity estimates for combined predicates use the actual correlation instead of independence assumption.

#### B.4 Why bad statistics produce bad plans

If the planner estimates a join will return 10 rows when it actually returns 10,000, it picks Nested Loop (great for 10 rows, terrible for 10,000). The fix is rarely "tune `random_page_cost`" — it is `ANALYZE` (especially after large data changes) and, for correlated columns, `CREATE STATISTICS`.

### 5.1 Statistics Collection

```sql
-- Collect table statistics
ANALYZE users;
ANALYZE;  -- entire database

-- Auto ANALYZE settings
SHOW autovacuum_analyze_threshold;     -- default 50
SHOW autovacuum_analyze_scale_factor;  -- default 0.1

-- Column statistics detail level
ALTER TABLE users ALTER COLUMN email SET STATISTICS 1000;
-- default 100, max 10000
ANALYZE users;

-- Check statistics
SELECT
    attname,
    n_distinct,
    most_common_vals,
    most_common_freqs,
    histogram_bounds
FROM pg_stats
WHERE tablename = 'users';
```

### 5.2 Row Count Estimation

```sql
-- Estimate table row count
SELECT reltuples::bigint AS estimate
FROM pg_class
WHERE relname = 'users';

-- Exact row count (slow)
SELECT COUNT(*) FROM users;

-- Conditional row count estimate
EXPLAIN SELECT * FROM users WHERE status = 'active';
-- check rows=xxx

-- Improve estimation accuracy
-- 1. Run ANALYZE
-- 2. Increase statistics detail
-- 3. Extended statistics (PostgreSQL 10+)
CREATE STATISTICS stts_user_country_status (dependencies)
ON country, status FROM users;
ANALYZE users;
```

### 5.3 Cost Calculation

```sql
-- cost = (pages × page_cost) + (rows × row_cost)

-- Check page count
SELECT relpages FROM pg_class WHERE relname = 'users';

-- Cost parameters
SHOW seq_page_cost;        -- 1.0
SHOW random_page_cost;     -- 4.0
SHOW cpu_tuple_cost;       -- 0.01
SHOW cpu_index_tuple_cost; -- 0.005
SHOW cpu_operator_cost;    -- 0.0025

-- Seq Scan cost calculation example
-- cost = (relpages × seq_page_cost) + (reltuples × cpu_tuple_cost)
-- cost = (1000 × 1.0) + (100000 × 0.01) = 2000

-- Index Scan cost is more complex
-- depends on selectivity
```

---

## 6. Advanced Optimization Techniques

### 6.1 Query Refactoring

```sql
-- OR → UNION (use index)
-- Bad
SELECT * FROM products
WHERE category_id = 1 OR brand_id = 2;

-- Good
SELECT * FROM products WHERE category_id = 1
UNION
SELECT * FROM products WHERE brand_id = 2;

-- IN → EXISTS (large data)
-- Bad (when subquery returns many rows)
SELECT * FROM users
WHERE id IN (SELECT user_id FROM orders WHERE amount > 1000);

-- Good
SELECT * FROM users u
WHERE EXISTS (
    SELECT 1 FROM orders o
    WHERE o.user_id = u.id AND o.amount > 1000
);

-- NOT IN → NOT EXISTS (NULL handling)
-- NOT IN returns empty result if NULL exists
SELECT * FROM users
WHERE id NOT IN (SELECT user_id FROM orders);  -- problem if orders.user_id has NULL

-- Safe method
SELECT * FROM users u
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id);

-- DISTINCT → GROUP BY (use index)
SELECT DISTINCT user_id FROM orders;
-- →
SELECT user_id FROM orders GROUP BY user_id;
```

### 6.2 Materialized View

```sql
-- Store complex aggregation results
CREATE MATERIALIZED VIEW mv_daily_sales AS
SELECT
    date_trunc('day', created_at) AS day,
    COUNT(*) AS order_count,
    SUM(total) AS total_sales
FROM orders
GROUP BY date_trunc('day', created_at);

-- Add index
CREATE UNIQUE INDEX idx_mv_daily_sales_day ON mv_daily_sales(day);

-- Refresh
REFRESH MATERIALIZED VIEW mv_daily_sales;
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_sales;  -- requires UNIQUE index

-- Auto refresh (use pg_cron or trigger)
```

### 6.3 Partitioning

```sql
-- Range partitioning
CREATE TABLE orders (
    id BIGSERIAL,
    created_at TIMESTAMP NOT NULL,
    user_id INT,
    total DECIMAL(10,2)
) PARTITION BY RANGE (created_at);

CREATE TABLE orders_2024_q1 PARTITION OF orders
FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

CREATE TABLE orders_2024_q2 PARTITION OF orders
FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');

-- Check partition pruning
EXPLAIN SELECT * FROM orders WHERE created_at = '2024-02-15';
-- only orders_2024_q1 scanned

-- List partitioning
CREATE TABLE logs (
    id BIGSERIAL,
    level VARCHAR(10),
    message TEXT
) PARTITION BY LIST (level);

CREATE TABLE logs_error PARTITION OF logs FOR VALUES IN ('ERROR', 'FATAL');
CREATE TABLE logs_info PARTITION OF logs FOR VALUES IN ('INFO', 'DEBUG');

-- Hash partitioning
CREATE TABLE events (
    id BIGSERIAL,
    user_id INT
) PARTITION BY HASH (user_id);

CREATE TABLE events_p0 PARTITION OF events FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE events_p1 PARTITION OF events FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE events_p2 PARTITION OF events FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE events_p3 PARTITION OF events FOR VALUES WITH (MODULUS 4, REMAINDER 3);
```

### 6.4 Query Caching

```sql
-- Prepared Statement (cache query plan)
PREPARE get_user(int) AS
SELECT * FROM users WHERE id = $1;

EXECUTE get_user(1);
EXECUTE get_user(2);

DEALLOCATE get_user;

-- Caution with prepared statements in connection poolers like PgBouncer

-- Result caching (application level)
-- Redis, Memcached recommended
```

---

## 7. Practice Problems

### Exercise 1: Analyze Execution Plan
```sql
-- Analyze and optimize the following query execution plan:
SELECT u.name, COUNT(o.id), SUM(o.total)
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.country = 'US'
AND o.created_at > NOW() - INTERVAL '1 year'
GROUP BY u.name
HAVING COUNT(o.id) > 10
ORDER BY SUM(o.total) DESC
LIMIT 100;

-- Analyze and propose improvements:
```

### Exercise 2: Index Design
```sql
-- Design optimal indexes for the following queries:
-- 1. SELECT * FROM orders WHERE user_id = ? AND status = 'pending' ORDER BY created_at DESC
-- 2. SELECT * FROM products WHERE category_id = ? AND price BETWEEN ? AND ?
-- 3. SELECT * FROM logs WHERE level = 'ERROR' AND created_at > NOW() - INTERVAL '1 day'

-- Write index creation statements:
```

### Exercise 3: Join Optimization
```sql
-- Optimize 5-table join query:
SELECT *
FROM orders o
JOIN users u ON o.user_id = u.id
JOIN products p ON o.product_id = p.id
JOIN categories c ON p.category_id = c.id
JOIN suppliers s ON p.supplier_id = s.id
WHERE c.name = 'Electronics'
AND o.created_at > '2024-01-01';

-- Develop optimization strategy:
```

### Exercise 4: Partitioning Design
```sql
-- Design partitioning for large log table:
-- Requirements:
-- - Daily data: 1 million rows
-- - Retention: 3 months
-- - Frequently queried: level, created_at, user_id

-- Design partition:
```

---

## References

- [PostgreSQL EXPLAIN](https://www.postgresql.org/docs/current/using-explain.html)
- [Query Planning](https://www.postgresql.org/docs/current/planner-optimizer.html)
- [Index Types](https://www.postgresql.org/docs/current/indexes-types.html)
- [Use The Index, Luke](https://use-the-index-luke.com/)

---

**Previous**: [JSON/JSONB Features](./14_JSON_JSONB.md) | **Next**: [Replication and High Availability](./16_Replication_HA.md)
