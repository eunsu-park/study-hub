-- Exercises for Lesson 16: Replication and High Availability
-- Topic: PostgreSQL
-- Solutions to practice problems from the lesson.


-- === Exercise 1: Configure Streaming Replication ===
-- Problem: Set up Primary-Standby streaming replication using Docker.

-- Solution:

-- docker-compose.yml for primary + standby (save as docker-compose-repl.yml):
--
-- version: '3.8'
-- services:
--   primary:
--     image: postgres:16
--     environment:
--       POSTGRES_PASSWORD: postgres
--       POSTGRES_INITDB_ARGS: "--data-checksums"
--     command: |
--       postgres
--       -c wal_level=replica
--       -c max_wal_senders=3
--       -c max_replication_slots=3
--       -c hot_standby=on
--     ports:
--       - "5432:5432"
--     volumes:
--       - primary_data:/var/lib/postgresql/data
--
--   standby:
--     image: postgres:16
--     environment:
--       POSTGRES_PASSWORD: postgres
--       PGDATA: /var/lib/postgresql/data
--     depends_on:
--       - primary
--     ports:
--       - "5433:5432"
--     volumes:
--       - standby_data:/var/lib/postgresql/data
--
-- volumes:
--   primary_data:
--   standby_data:

-- On the primary: create a replication user and slot
CREATE ROLE replicator WITH REPLICATION LOGIN PASSWORD 'repl_password';

-- Create a replication slot (prevents WAL segments from being recycled
-- before the standby has consumed them)
SELECT pg_create_physical_replication_slot('standby_slot');

-- Verify WAL level is set correctly
SHOW wal_level;  -- should be 'replica'

-- On the standby: verify it's in recovery mode
-- SELECT pg_is_in_recovery();  -- should return true

-- Check replication status from the primary
SELECT
    client_addr,
    state,
    sent_lsn,
    write_lsn,
    flush_lsn,
    replay_lsn,
    pg_size_pretty(pg_wal_lsn_diff(sent_lsn, replay_lsn)) AS replication_lag
FROM pg_stat_replication;


-- === Exercise 2: Configure Logical Replication ===
-- Problem: Set up logical replication to replicate only specific tables.

-- Solution:

-- On the publisher (source database):

-- Create the table to be replicated
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    price NUMERIC(10,2),
    category VARCHAR(50)
);

INSERT INTO products (name, price, category) VALUES
    ('Laptop', 999.99, 'Electronics'),
    ('Book', 29.99, 'Books'),
    ('Headphones', 149.99, 'Electronics');

-- Create a publication for just this table.
-- Unlike streaming replication (which copies everything),
-- logical replication lets you choose specific tables.
CREATE PUBLICATION products_pub FOR TABLE products;

-- Verify publication
SELECT * FROM pg_publication;
SELECT * FROM pg_publication_tables;


-- On the subscriber (target database):

-- Create the same table structure (logical replication does NOT copy DDL)
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    price NUMERIC(10,2),
    category VARCHAR(50)
);

-- Subscribe to the publication
CREATE SUBSCRIPTION products_sub
    CONNECTION 'host=source_host port=5432 dbname=source_db user=replicator password=repl_password'
    PUBLICATION products_pub;

-- Verify subscription status
SELECT * FROM pg_subscription;
SELECT * FROM pg_stat_subscription;


-- === Exercise 3: Replication Monitoring Dashboard ===
-- Problem: Write a comprehensive query showing replication status.

-- Solution:

-- Single query that unions multiple replication health metrics
-- into a unified dashboard view
SELECT
    'Replication Lag' AS metric,
    COALESCE(
        (SELECT pg_size_pretty(pg_wal_lsn_diff(sent_lsn, replay_lsn))
         FROM pg_stat_replication
         LIMIT 1),
        'No standby connected'
    ) AS value

UNION ALL

SELECT
    'Standby Count',
    (SELECT COUNT(*)::text FROM pg_stat_replication)

UNION ALL

SELECT
    'Replication Slots',
    (SELECT COUNT(*)::text FROM pg_replication_slots)

UNION ALL

SELECT
    'Active Slots',
    (SELECT COUNT(*)::text FROM pg_replication_slots WHERE active)

UNION ALL

SELECT
    'WAL Level',
    (SELECT setting FROM pg_settings WHERE name = 'wal_level')

UNION ALL

SELECT
    'Max WAL Senders',
    (SELECT setting FROM pg_settings WHERE name = 'max_wal_senders')

UNION ALL

SELECT
    'Current WAL LSN',
    pg_current_wal_lsn()::text;

-- Bonus: Detailed per-standby view
CREATE OR REPLACE VIEW v_replication_status AS
SELECT
    pid,
    client_addr,
    state,
    sync_state,
    sent_lsn,
    write_lsn,
    flush_lsn,
    replay_lsn,
    pg_size_pretty(pg_wal_lsn_diff(sent_lsn, replay_lsn)) AS lag_bytes,
    now() - backend_start AS connection_duration,
    CASE
        WHEN pg_wal_lsn_diff(sent_lsn, replay_lsn) = 0 THEN 'Caught up'
        WHEN pg_wal_lsn_diff(sent_lsn, replay_lsn) < 1048576 THEN 'Slight lag (<1MB)'
        ELSE 'Significant lag'
    END AS health_status
FROM pg_stat_replication;
