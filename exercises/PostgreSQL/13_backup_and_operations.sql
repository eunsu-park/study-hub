-- Exercises for Lesson 13: Backup and Operations
-- Topic: PostgreSQL
-- Solutions to practice problems from the lesson.


-- === Exercise 1: Backup and Restore ===
-- Problem: Perform a full database backup, restore to a new DB, and verify.

-- Solution:
-- (These are shell commands; run from bash, not psql)

-- 1. Backup using custom format (-Fc) for compression and selective restore
-- pg_dump -U postgres -Fc mydb > mydb_backup.dump

-- 2. Create a fresh target database
-- createdb -U postgres mydb_restored

-- 3. Restore the dump into the new database
-- pg_restore -U postgres -d mydb_restored mydb_backup.dump

-- 4. Verify: check that data was restored correctly
-- psql -U postgres -d mydb_restored -c "SELECT COUNT(*) FROM users;"

-- Alternative: SQL-format backup (human-readable, useful for diffing)
-- pg_dump -U postgres --format=plain mydb > mydb_backup.sql
-- psql -U postgres -d mydb_restored -f mydb_backup.sql

-- Backup a single table
-- pg_dump -U postgres -t users mydb > users_backup.sql

-- Backup with schema only (no data) — useful for version control
-- pg_dump -U postgres --schema-only mydb > mydb_schema.sql


-- === Exercise 2: Save Monitoring Queries as Views ===
-- Problem: Create reusable monitoring views for database health checks.

-- Solution:

-- View 1: Database sizes and connection counts
-- Excludes template databases which are system-internal
CREATE OR REPLACE VIEW v_db_stats AS
SELECT
    datname,
    pg_size_pretty(pg_database_size(datname)) AS size,
    numbackends AS connections
FROM pg_database
WHERE datistemplate = false
ORDER BY pg_database_size(datname) DESC;

-- View 2: Currently running slow queries (> 5 seconds)
-- Useful for identifying queries that need optimization
CREATE OR REPLACE VIEW v_slow_queries AS
SELECT
    pid,
    now() - query_start AS duration,
    state,
    LEFT(query, 200) AS query_preview
FROM pg_stat_activity
WHERE state = 'active'
  AND now() - query_start > interval '5 seconds'
ORDER BY duration DESC;

-- View 3: Table sizes (bonus — very useful for capacity planning)
CREATE OR REPLACE VIEW v_table_sizes AS
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) AS total_size,
    pg_size_pretty(pg_relation_size(schemaname || '.' || tablename)) AS data_size,
    pg_size_pretty(
        pg_total_relation_size(schemaname || '.' || tablename) -
        pg_relation_size(schemaname || '.' || tablename)
    ) AS index_size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname || '.' || tablename) DESC;

-- Usage
SELECT * FROM v_db_stats;
SELECT * FROM v_slow_queries;
SELECT * FROM v_table_sizes;


-- === Exercise 3: Maintenance Script ===
-- Problem: Create a stored procedure for routine database maintenance.

-- Solution:

-- This procedure runs ANALYZE (updates planner statistics) and VACUUM
-- (reclaims dead tuple space). In production, you would schedule this
-- via pg_cron or an external cron job.
CREATE OR REPLACE PROCEDURE run_maintenance()
AS $$
BEGIN
    -- Update statistics so the query planner makes better decisions
    ANALYZE;

    -- Clean up dead tuples left by UPDATE/DELETE operations
    -- Note: VACUUM cannot run inside a transaction block, so this
    -- works in a procedure (which has implicit transaction management)
    VACUUM;

    RAISE NOTICE 'Maintenance completed at %', NOW();
END;
$$ LANGUAGE plpgsql;

-- Execute maintenance
CALL run_maintenance();

-- Bonus: Check when tables were last vacuumed/analyzed
SELECT
    schemaname,
    relname AS table_name,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze,
    n_dead_tup AS dead_tuples
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY n_dead_tup DESC;
