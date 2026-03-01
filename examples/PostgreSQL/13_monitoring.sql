-- ============================================================================
-- PostgreSQL Monitoring and Performance Analysis
-- ============================================================================
-- Demonstrates:
--   - pg_stat_activity: active queries and connections
--   - pg_stat_user_tables: table I/O statistics
--   - pg_stat_user_indexes: index usage analysis
--   - pg_stat_statements: query performance (extension)
--   - pg_locks: lock monitoring
--   - Cache hit ratio
--   - Table bloat estimation
--   - Slow query identification
--
-- Prerequisites: PostgreSQL 12+, pg_stat_statements extension
-- Usage: psql -U postgres -d your_database -f 13_monitoring.sql
-- ============================================================================

-- ============================================================================
-- 1. Active Connections and Queries
-- ============================================================================

-- Who is connected right now?
SELECT
    pid,
    usename AS user,
    application_name AS app,
    client_addr AS ip,
    state,
    CASE
        WHEN state = 'active' THEN
            NOW() - query_start
        ELSE NULL
    END AS query_duration,
    LEFT(query, 80) AS current_query
FROM pg_stat_activity
WHERE backend_type = 'client backend'
ORDER BY query_start;

-- Connection count by state
SELECT
    state,
    COUNT(*) AS connections
FROM pg_stat_activity
WHERE backend_type = 'client backend'
GROUP BY state
ORDER BY connections DESC;

-- ============================================================================
-- 2. Database Size Overview
-- ============================================================================

-- Database sizes
SELECT
    datname AS database,
    pg_size_pretty(pg_database_size(datname)) AS size
FROM pg_database
WHERE datistemplate = FALSE
ORDER BY pg_database_size(datname) DESC;

-- Table sizes (including indexes and TOAST)
SELECT
    schemaname AS schema,
    relname AS table,
    pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
    pg_size_pretty(pg_relation_size(relid)) AS table_size,
    pg_size_pretty(pg_indexes_size(relid)) AS index_size,
    n_live_tup AS live_rows,
    n_dead_tup AS dead_rows
FROM pg_stat_user_tables
ORDER BY pg_total_relation_size(relid) DESC
LIMIT 20;

-- ============================================================================
-- 3. Cache Hit Ratio
-- ============================================================================

-- Why: The cache hit ratio is the single most important metric for PostgreSQL
-- performance. Below 99%, queries are frequently hitting disk instead of
-- shared_buffers + OS page cache, indicating either insufficient memory
-- or a working set larger than available RAM.
SELECT
    SUM(heap_blks_read) AS blocks_from_disk,
    SUM(heap_blks_hit) AS blocks_from_cache,
    CASE
        WHEN SUM(heap_blks_hit) + SUM(heap_blks_read) = 0 THEN 0
        ELSE ROUND(
            SUM(heap_blks_hit)::NUMERIC /
            (SUM(heap_blks_hit) + SUM(heap_blks_read)) * 100, 2
        )
    END AS cache_hit_ratio_pct
FROM pg_statio_user_tables;

-- Per-table cache hit ratio
SELECT
    relname AS table_name,
    heap_blks_read AS disk_reads,
    heap_blks_hit AS cache_hits,
    CASE
        WHEN heap_blks_hit + heap_blks_read = 0 THEN 0
        ELSE ROUND(
            heap_blks_hit::NUMERIC /
            (heap_blks_hit + heap_blks_read) * 100, 2
        )
    END AS cache_hit_pct
FROM pg_statio_user_tables
WHERE heap_blks_hit + heap_blks_read > 0
ORDER BY heap_blks_read DESC;

-- Index cache hit ratio
SELECT
    indexrelname AS index_name,
    idx_blks_read AS disk_reads,
    idx_blks_hit AS cache_hits,
    CASE
        WHEN idx_blks_hit + idx_blks_read = 0 THEN 0
        ELSE ROUND(
            idx_blks_hit::NUMERIC /
            (idx_blks_hit + idx_blks_read) * 100, 2
        )
    END AS cache_hit_pct
FROM pg_statio_user_indexes
WHERE idx_blks_hit + idx_blks_read > 0
ORDER BY idx_blks_read DESC;

-- ============================================================================
-- 4. Index Usage Analysis
-- ============================================================================

-- Index usage statistics — find unused indexes
SELECT
    schemaname AS schema,
    relname AS table,
    indexrelname AS index,
    idx_scan AS scans,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
ORDER BY idx_scan ASC;

-- Why: High sequential scan counts on large tables indicate missing indexes.
-- The seq_scan_pct metric shows what fraction of accesses skip indexes entirely.
-- Focus on tables with >1000 rows where seq_scan > idx_scan — these are the
-- strongest candidates for new indexes.
SELECT
    relname AS table,
    seq_scan AS sequential_scans,
    seq_tup_read AS seq_rows_read,
    idx_scan AS index_scans,
    CASE
        WHEN seq_scan + idx_scan = 0 THEN 0
        ELSE ROUND(
            seq_scan::NUMERIC / (seq_scan + idx_scan) * 100, 1
        )
    END AS seq_scan_pct,
    n_live_tup AS live_rows
FROM pg_stat_user_tables
WHERE n_live_tup > 1000
  AND seq_scan > idx_scan
ORDER BY seq_tup_read DESC;

-- ============================================================================
-- 5. Table Maintenance Status (VACUUM / ANALYZE)
-- ============================================================================

-- Why: Dead tuples accumulate from UPDATEs and DELETEs (MVCC creates new row
-- versions instead of modifying in-place). High dead_pct means table bloat and
-- slower sequential scans. If last_autovacuum is NULL or very old, autovacuum
-- may be misconfigured or blocked by long-running transactions.
SELECT
    relname AS table,
    n_live_tup AS live_rows,
    n_dead_tup AS dead_rows,
    CASE
        WHEN n_live_tup = 0 THEN 0
        ELSE ROUND(n_dead_tup::NUMERIC / n_live_tup * 100, 1)
    END AS dead_pct,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC;

-- ============================================================================
-- 6. Lock Monitoring
-- ============================================================================

-- Current locks
SELECT
    l.locktype,
    l.relation::regclass AS table_name,
    l.mode,
    l.granted,
    l.pid,
    a.usename,
    a.state,
    LEFT(a.query, 60) AS query
FROM pg_locks l
JOIN pg_stat_activity a ON l.pid = a.pid
WHERE l.relation IS NOT NULL
  AND a.pid != pg_backend_pid()
ORDER BY l.relation, l.mode;

-- Why: This query joins pg_locks with pg_stat_activity to trace which session is
-- blocking which. The key insight is matching NOT bl.granted (waiting lock) with
-- kl.granted (holding lock) on the same resource. Long blocked_duration values
-- indicate a stuck transaction that may need manual intervention (pg_terminate_backend).
SELECT
    blocked.pid AS blocked_pid,
    blocked.usename AS blocked_user,
    LEFT(blocked.query, 50) AS blocked_query,
    blocking.pid AS blocking_pid,
    blocking.usename AS blocking_user,
    LEFT(blocking.query, 50) AS blocking_query,
    NOW() - blocked.query_start AS blocked_duration
FROM pg_stat_activity blocked
JOIN pg_locks bl ON blocked.pid = bl.pid AND NOT bl.granted
JOIN pg_locks kl ON bl.locktype = kl.locktype
    AND bl.relation = kl.relation
    AND bl.page = kl.page
    AND bl.tuple = kl.tuple
    AND kl.granted
JOIN pg_stat_activity blocking ON kl.pid = blocking.pid
WHERE blocked.pid != blocking.pid;

-- ============================================================================
-- 7. pg_stat_statements — Query Performance
-- ============================================================================
-- Requires: CREATE EXTENSION pg_stat_statements;
-- Must be added to shared_preload_libraries in postgresql.conf

-- Top queries by total execution time
-- SELECT
--     LEFT(query, 80) AS query,
--     calls,
--     ROUND(total_exec_time::NUMERIC, 2) AS total_ms,
--     ROUND(mean_exec_time::NUMERIC, 2) AS avg_ms,
--     ROUND(stddev_exec_time::NUMERIC, 2) AS stddev_ms,
--     rows
-- FROM pg_stat_statements
-- ORDER BY total_exec_time DESC
-- LIMIT 10;

-- Top queries by average time (slow queries)
-- SELECT
--     LEFT(query, 80) AS query,
--     calls,
--     ROUND(mean_exec_time::NUMERIC, 2) AS avg_ms,
--     ROUND(max_exec_time::NUMERIC, 2) AS max_ms,
--     rows / GREATEST(calls, 1) AS avg_rows
-- FROM pg_stat_statements
-- WHERE calls > 10
-- ORDER BY mean_exec_time DESC
-- LIMIT 10;

-- ============================================================================
-- 8. Replication Status
-- ============================================================================

-- Why: Replication lag is measured by comparing the sent vs replayed WAL position.
-- sent_lsn > replay_lsn means the standby has received data but not yet applied
-- it. Large lag can cause stale reads on the standby and risk data loss during
-- failover. Monitor this metric continuously in production.
SELECT
    client_addr,
    state,
    sent_lsn,
    write_lsn,
    flush_lsn,
    replay_lsn,
    pg_wal_lsn_diff(sent_lsn, replay_lsn) AS replication_lag_bytes,
    pg_size_pretty(pg_wal_lsn_diff(sent_lsn, replay_lsn)) AS lag_pretty
FROM pg_stat_replication;

-- WAL statistics
SELECT
    pg_current_wal_lsn() AS current_lsn,
    pg_walfile_name(pg_current_wal_lsn()) AS current_wal_file,
    pg_size_pretty(pg_wal_lsn_diff(
        pg_current_wal_lsn(),
        '0/0'::pg_lsn
    )) AS total_wal_generated;

-- ============================================================================
-- 9. Configuration Check
-- ============================================================================

-- Key performance-related settings
SELECT name, setting, unit, short_desc
FROM pg_settings
WHERE name IN (
    'shared_buffers',
    'effective_cache_size',
    'work_mem',
    'maintenance_work_mem',
    'max_connections',
    'max_parallel_workers',
    'random_page_cost',
    'effective_io_concurrency',
    'wal_buffers',
    'checkpoint_completion_target'
)
ORDER BY name;

-- ============================================================================
-- 10. Health Check Summary Query
-- ============================================================================

-- All-in-one health dashboard
SELECT 'Database Size' AS metric,
       pg_size_pretty(pg_database_size(current_database())) AS value
UNION ALL
SELECT 'Active Connections',
       COUNT(*)::TEXT
FROM pg_stat_activity WHERE state = 'active'
UNION ALL
SELECT 'Cache Hit Ratio',
       ROUND(
           SUM(heap_blks_hit)::NUMERIC /
           GREATEST(SUM(heap_blks_hit) + SUM(heap_blks_read), 1) * 100, 2
       )::TEXT || '%'
FROM pg_statio_user_tables
UNION ALL
SELECT 'Dead Tuples (total)',
       SUM(n_dead_tup)::TEXT
FROM pg_stat_user_tables
UNION ALL
SELECT 'Longest Running Query',
       COALESCE(
           MAX(NOW() - query_start)::TEXT,
           'none'
       )
FROM pg_stat_activity
WHERE state = 'active' AND pid != pg_backend_pid();
