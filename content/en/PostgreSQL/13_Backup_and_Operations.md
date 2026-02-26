# Backup and Operations

**Previous**: [Triggers](./12_Triggers.md) | **Next**: [JSON/JSONB Features](./14_JSON_JSONB.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the importance of backup strategies and distinguish between logical and physical backups
2. Use pg_dump and pg_dumpall to perform selective and full cluster backups
3. Restore databases using psql and pg_restore with various format options
4. Configure WAL archiving and perform physical backups with pg_basebackup
5. Write automated backup scripts with retention policies and cron scheduling
6. Monitor database health using pg_stat_activity, lock queries, and cache hit ratios
7. Perform routine maintenance tasks including VACUUM, ANALYZE, and REINDEX
8. Configure PostgreSQL logging for slow query detection and connection auditing

---

A database without a tested backup strategy is a disaster waiting to happen. Hardware failures, human errors, and software bugs can strike at any time, and when they do, your backup is the difference between a minor inconvenience and catastrophic data loss. Beyond backups, day-to-day operations -- monitoring performance, managing connections, and running maintenance tasks -- keep your PostgreSQL installation healthy and responsive. This lesson covers the essential DBA toolkit that every PostgreSQL practitioner needs.

## 1. Importance of Backup

Database backup is the most important task to prevent data loss.

```
┌──────────────────────────────────────────────────────────┐
│                    Backup Strategy                        │
├──────────────────────────────────────────────────────────┤
│  • Regular backups: Daily/weekly full backup              │
│  • Incremental backups: WAL archiving                     │
│  • Replication: Real-time replica servers                 │
└──────────────────────────────────────────────────────────┘
```

---

## 2. pg_dump - Logical Backup

### Basic Backup

```bash
# Single database backup
pg_dump dbname > backup.sql

# Specify user/host
pg_dump -U username -h localhost dbname > backup.sql

# Compressed backup
pg_dump dbname | gzip > backup.sql.gz
```

### Format Options

```bash
# Plain text SQL (-Fp, default)
pg_dump -Fp dbname > backup.sql

# Custom format (-Fc, compressed, selective restore)
pg_dump -Fc dbname > backup.dump

# Directory format (-Fd, parallel backup/restore support)
pg_dump -Fd dbname -f backup_dir

# Tar format (-Ft)
pg_dump -Ft dbname > backup.tar
```

### Selective Backup

```bash
# Specific tables only
pg_dump -t users -t orders dbname > tables.sql

# Exclude specific tables
pg_dump -T logs -T temp_* dbname > backup.sql

# Schema only (exclude data)
pg_dump -s dbname > schema.sql

# Data only (exclude schema)
pg_dump -a dbname > data.sql

# Specific schema only
pg_dump -n public dbname > public_schema.sql
```

### Backup from Docker

```bash
# Run pg_dump in Docker container
docker exec -t postgres-container pg_dump -U postgres dbname > backup.sql

# Compressed backup
docker exec -t postgres-container pg_dump -U postgres dbname | gzip > backup.sql.gz
```

---

## 3. pg_dumpall - Full Cluster Backup

Backs up all databases and global objects (users, permissions, etc.).

```bash
# Full cluster backup
pg_dumpall -U postgres > full_backup.sql

# Global objects only (users, roles, etc.)
pg_dumpall -U postgres --globals-only > globals.sql

# Roles only
pg_dumpall -U postgres --roles-only > roles.sql
```

---

## 4. pg_restore - Restore

### Restoring SQL Files

```bash
# Restore plain SQL
psql dbname < backup.sql

# Create new database and restore
createdb newdb
psql newdb < backup.sql
```

### Restoring Custom/Directory Format

```bash
# Restore custom format
pg_restore -d dbname backup.dump

# Restore to new database
createdb newdb
pg_restore -d newdb backup.dump

# Restore specific table only
pg_restore -d dbname -t users backup.dump

# Parallel restore (4 workers)
pg_restore -d dbname -j 4 backup_dir
```

### Restore Options

```bash
# Drop existing objects before restore
pg_restore -d dbname --clean backup.dump

# Ignore errors and continue
pg_restore -d dbname --if-exists backup.dump

# Data only restore
pg_restore -d dbname --data-only backup.dump

# Schema only restore
pg_restore -d dbname --schema-only backup.dump
```

---

## 5. Physical Backup (pg_basebackup)

Backs up the entire data directory.

```bash
# Basic backup
pg_basebackup -D /backup/path -U postgres -Fp -Xs -P

# Compressed backup
pg_basebackup -D /backup/path -U postgres -Ft -z -P

# Option descriptions:
# -D: Backup directory
# -Fp: Plain format
# -Ft: Tar format
# -Xs: WAL streaming
# -z: gzip compression
# -P: Show progress
```

### WAL Archiving Setup

`postgresql.conf`:
```
wal_level = replica
archive_mode = on
archive_command = 'cp %p /archive/%f'
```

---

## 6. Automated Backup Script

### Daily Backup Script

```bash
#!/bin/bash
# daily_backup.sh

# Configuration
DB_NAME="mydb"
DB_USER="postgres"
BACKUP_DIR="/backup/postgres"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=7

# Create backup directory
mkdir -p $BACKUP_DIR

# Execute backup
pg_dump -U $DB_USER -Fc $DB_NAME > $BACKUP_DIR/${DB_NAME}_${DATE}.dump

# Compress
gzip $BACKUP_DIR/${DB_NAME}_${DATE}.dump

# Delete old backups
find $BACKUP_DIR -name "*.dump.gz" -mtime +$RETENTION_DAYS -delete

echo "Backup completed: ${DB_NAME}_${DATE}.dump.gz"
```

### Cron Setup

```bash
# crontab -e
# Backup daily at 2 AM
0 2 * * * /scripts/daily_backup.sh >> /var/log/backup.log 2>&1
```

---

## 7. Monitoring

### Database Size

```sql
-- Database sizes
SELECT
    datname,
    pg_size_pretty(pg_database_size(datname)) AS size
FROM pg_database
ORDER BY pg_database_size(datname) DESC;

-- Table sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) AS total_size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname || '.' || tablename) DESC
LIMIT 10;
```

### Connection Status

```sql
-- Current connection count
SELECT COUNT(*) FROM pg_stat_activity;

-- Connections by state
SELECT state, COUNT(*)
FROM pg_stat_activity
GROUP BY state;

-- Active queries
SELECT
    pid,
    now() - query_start AS duration,
    query,
    state
FROM pg_stat_activity
WHERE state != 'idle'
  AND query NOT LIKE '%pg_stat_activity%'
ORDER BY duration DESC;
```

### Slow Queries

```sql
-- Queries running longer than 5 seconds
SELECT
    pid,
    now() - query_start AS duration,
    query
FROM pg_stat_activity
WHERE state = 'active'
  AND now() - query_start > interval '5 seconds';
```

### Lock Status

```sql
-- Queries waiting for locks
SELECT
    blocked.pid AS blocked_pid,
    blocked.query AS blocked_query,
    blocking.pid AS blocking_pid,
    blocking.query AS blocking_query
FROM pg_stat_activity blocked
JOIN pg_stat_activity blocking
    ON blocking.pid = ANY(pg_blocking_pids(blocked.pid));
```

---

## 8. Performance Statistics

### Table Statistics

```sql
-- Table access statistics
SELECT
    schemaname,
    relname,
    seq_scan,
    seq_tup_read,
    idx_scan,
    idx_tup_fetch,
    n_tup_ins,
    n_tup_upd,
    n_tup_del
FROM pg_stat_user_tables
ORDER BY seq_scan DESC
LIMIT 10;
```

### Index Usage

```sql
-- Unused indexes
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;
```

### Cache Hit Rate

```sql
-- Cache hit rate (99%+ is good)
SELECT
    sum(blks_hit) * 100.0 / sum(blks_hit + blks_read) AS cache_hit_ratio
FROM pg_stat_database;
```

---

## 9. Maintenance

### VACUUM

Cleans up unnecessary space.

```sql
-- Regular VACUUM
VACUUM;
VACUUM users;

-- VACUUM FULL (rebuilds table, locks table)
VACUUM FULL users;

-- VACUUM ANALYZE (includes statistics update)
VACUUM ANALYZE users;
```

### ANALYZE

Collects statistics for query optimization.

```sql
ANALYZE;
ANALYZE users;
```

### REINDEX

Rebuilds indexes.

```sql
REINDEX TABLE users;
REINDEX DATABASE mydb;
```

### Autovacuum Settings

`postgresql.conf`:
```
autovacuum = on
autovacuum_naptime = 1min
autovacuum_vacuum_threshold = 50
autovacuum_analyze_threshold = 50
```

---

## 10. Log Configuration

`postgresql.conf`:

```
# Log destination
logging_collector = on
log_directory = 'pg_log'
log_filename = 'postgresql-%Y-%m-%d.log'

# Log level
log_min_messages = warning
log_min_error_statement = error

# Query logging
log_statement = 'ddl'           # none, ddl, mod, all
log_duration = off
log_min_duration_statement = 1000  # Queries longer than 1 second

# Connection logging
log_connections = on
log_disconnections = on
```

---

## 11. Security Settings

### pg_hba.conf

```
# TYPE  DATABASE    USER        ADDRESS         METHOD

# Local connections
local   all         all                         peer

# IPv4 local connections
host    all         all         127.0.0.1/32    scram-sha-256

# Allow specific network
host    mydb        appuser     192.168.1.0/24  scram-sha-256

# Deny specific IP
host    all         all         192.168.1.100   reject
```

### SSL Configuration

```
# postgresql.conf
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file = 'server.key'
```

---

## 12. Practice Examples

### Practice 1: Backup and Restore

```bash
# 1. Backup
pg_dump -U postgres -Fc mydb > mydb_backup.dump

# 2. Create new database
createdb -U postgres mydb_restored

# 3. Restore
pg_restore -U postgres -d mydb_restored mydb_backup.dump

# 4. Verify
psql -U postgres -d mydb_restored -c "SELECT COUNT(*) FROM users;"
```

### Practice 2: Save Monitoring Queries

```sql
-- Create monitoring views
CREATE VIEW v_db_stats AS
SELECT
    datname,
    pg_size_pretty(pg_database_size(datname)) AS size,
    numbackends AS connections
FROM pg_database
WHERE datistemplate = false;

CREATE VIEW v_slow_queries AS
SELECT
    pid,
    now() - query_start AS duration,
    state,
    query
FROM pg_stat_activity
WHERE state = 'active'
  AND now() - query_start > interval '5 seconds';

-- Usage
SELECT * FROM v_db_stats;
SELECT * FROM v_slow_queries;
```

### Practice 3: Maintenance Script

```sql
-- Regular maintenance procedure
CREATE PROCEDURE run_maintenance()
AS $$
BEGIN
    -- Update statistics
    ANALYZE;

    -- Clean unnecessary space
    VACUUM;

    RAISE NOTICE 'Maintenance completed: %', NOW();
END;
$$ LANGUAGE plpgsql;

-- Execute
CALL run_maintenance();
```

---

## 13. Checklist

### Daily Checks

- [ ] Verify backup success
- [ ] Check disk usage
- [ ] Check connection count
- [ ] Review error logs

### Weekly Checks

- [ ] Check index usage
- [ ] Analyze slow queries
- [ ] Monitor table size trends

### Monthly Checks

- [ ] Test backup restore
- [ ] Clean up unnecessary data
- [ ] Analyze performance trends

---

**Previous**: [Triggers](./12_Triggers.md) | **Next**: [JSON/JSONB Features](./14_JSON_JSONB.md)
