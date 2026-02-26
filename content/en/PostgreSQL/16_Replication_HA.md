# 16. Replication & High Availability

**Previous**: [Query Optimization](./15_Query_Optimization.md) | **Next**: [Window Functions](./17_Window_Functions.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare physical (streaming) and logical replication and identify appropriate use cases for each
2. Configure a primary-standby streaming replication setup including WAL settings and replication slots
3. Set up logical replication with publications and subscriptions for selective table replication
4. Monitor replication lag, slot status, and WAL accumulation using system views
5. Perform manual failover and switchover operations and recover old primaries with pg_rewind
6. Design a high availability architecture using Patroni, etcd, and HAProxy
7. Integrate connection pooling (PgBouncer) with a replicated cluster

---

Downtime costs money and erodes user trust. For any application where availability matters -- which is nearly every production system -- a single PostgreSQL server is a single point of failure. Replication creates copies of your data on standby servers that can serve read traffic, provide disaster recovery, and take over within seconds when the primary fails. This lesson covers the full spectrum from basic streaming replication to production-grade high availability with automatic failover.

## Table of Contents
1. [Replication Overview](#1-replication-overview)
2. [Physical Replication (Streaming Replication)](#2-physical-replication-streaming-replication)
3. [Logical Replication](#3-logical-replication)
4. [Replication Monitoring](#4-replication-monitoring)
5. [Failover and Switchover](#5-failover-and-switchover)
6. [High Availability Solutions](#6-high-availability-solutions)
7. [Practice Problems](#7-practice-problems)

---

## 1. Replication Overview

### 1.1 Purpose of Replication

```
┌─────────────────────────────────────────────────────────────────┐
│                    Replication Purposes                          │
├─────────────────┬───────────────────────────────────────────────┤
│ High Availability (HA)  │ Minimize downtime with auto/manual failover │
│ Read Scaling            │ Distribute read queries to standby          │
│ Disaster Recovery (DR)  │ Geographically distributed replicas for DR  │
│ Backup                  │ Run backups on standby, reduce prod load    │
│ Data Analytics          │ Run heavy analytical queries on replica     │
└─────────────────┴───────────────────────────────────────────────┘
```

### 1.2 Replication Type Comparison

```
┌────────────────┬─────────────────────┬─────────────────────┐
│                │   Physical Repl     │   Logical Repl      │
├────────────────┼─────────────────────┼─────────────────────┤
│ Unit           │ Byte-level (WAL)    │ Row-level changes   │
│ Scope          │ Entire cluster      │ Selective (table)   │
│ Version Compat │ Same major version  │ Different versions  │
│ Standby Query  │ Read-only           │ Read/Write possible │
│ Config Complexity │ Simple           │ Medium              │
│ Use Case       │ HA, read scaling    │ Migration, integration │
└────────────────┴─────────────────────┴─────────────────────┘
```

### 1.3 WAL (Write-Ahead Logging) Basics

```sql
-- WAL (Write-Ahead Log) is already written for crash recovery — replication simply
-- ships these same WAL records to standbys, so it adds minimal overhead to the primary.
-- This dual-purpose design is why PostgreSQL replication is efficient by default.
SHOW wal_level;           -- replica or logical
SHOW max_wal_senders;     -- Number of WAL sender processes
SHOW max_replication_slots;
SHOW wal_keep_size;       -- WAL retention size

-- Check WAL position
SELECT pg_current_wal_lsn();           -- Current WAL position
SELECT pg_walfile_name(pg_current_wal_lsn());  -- WAL file name
```

---

## 2. Physical Replication (Streaming Replication)

### 2.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Streaming Replication Architecture             │
│                                                                 │
│   Primary                           Standby                    │
│   ┌─────────────┐                   ┌─────────────┐           │
│   │             │    WAL Stream     │             │           │
│   │  PostgreSQL │ ────────────────► │  PostgreSQL │           │
│   │   (R/W)     │                   │   (R/O)     │           │
│   │             │                   │             │           │
│   │ ┌─────────┐ │                   │ ┌─────────┐ │           │
│   │ │wal_sender│─┼───────────────────┼─│wal_recv │ │           │
│   │ └─────────┘ │                   │ └─────────┘ │           │
│   └─────────────┘                   └─────────────┘           │
│                                                                 │
│   [Synchronous/Asynchronous options available]                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Primary Server Configuration

```bash
# postgresql.conf (Primary)
listen_addresses = '*'
wal_level = replica          # minimum level for streaming replication
max_wal_senders = 5          # one sender process per standby + headroom for pg_basebackup
wal_keep_size = 1GB          # retain WAL segments so slow standbys can catch up without slot
max_replication_slots = 5    # slots prevent WAL recycling — always create one per standby

# Synchronous replication settings (optional)
synchronous_commit = on
synchronous_standby_names = 'standby1'

# pg_hba.conf (allow replication connections)
# TYPE  DATABASE        USER            ADDRESS                 METHOD
host    replication     replicator      192.168.1.0/24          scram-sha-256
```

```sql
-- Create replication user
CREATE ROLE replicator WITH REPLICATION LOGIN PASSWORD 'secure_password';

-- Create replication slot (recommended)
SELECT pg_create_physical_replication_slot('standby1_slot');

-- Check replication slots
SELECT * FROM pg_replication_slots;
```

### 2.3 Standby Server Configuration

```bash
# 1. Create base backup from Primary
pg_basebackup -h primary_host -U replicator -D /var/lib/postgresql/data \
    -Fp -Xs -P -R

# -R option: auto-create standby.signal file and primary_conninfo
```

```bash
# postgresql.conf (Standby)
hot_standby = on                  # Allow read queries while replaying WAL
hot_standby_feedback = on         # Tells primary about standby queries — prevents
                                  # primary from vacuuming rows the standby still needs
max_standby_streaming_delay = 30s # How long a standby query can block WAL replay
                                  # before being cancelled — balance freshness vs query stability
```

```bash
# postgresql.auto.conf (auto-generated by pg_basebackup -R)
primary_conninfo = 'host=primary_host port=5432 user=replicator password=secure_password'
primary_slot_name = 'standby1_slot'
```

### 2.4 Synchronous vs Asynchronous Replication

```sql
-- Asynchronous replication (default)
-- Primary commits immediately, standby may lag
synchronous_commit = on  -- guarantees local only

-- Synchronous replication
synchronous_commit = on
synchronous_standby_names = 'FIRST 1 (standby1, standby2)'

-- Synchronous replication options
-- remote_write: to remote OS buffer
-- remote_apply: to remote apply (safest, slowest)
synchronous_commit = remote_apply
```

```
Synchronous replication configuration example:
┌─────────────────────────────────────────────────────────────────┐
│ synchronous_standby_names = 'FIRST 2 (s1, s2, s3)'             │
│                                                                 │
│   - FIRST 2: requires confirmation from first 2 standbys       │
│   - ANY 2: requires confirmation from any 2 standbys           │
│   - s1, s2, s3: priority based on application_name             │
└─────────────────────────────────────────────────────────────────┘
```

### 2.5 Cascading Replication

```
┌──────────────────────────────────────────────────────────────┐
│              Cascading Replication Topology                   │
│                                                              │
│   Primary ──► Standby1 ──► Standby2 ──► Standby3           │
│              (relay)       (relay)       (final)            │
│                                                              │
│   Advantages:                                                │
│   - Reduce Primary load                                      │
│   - Efficient network bandwidth usage                        │
│   - Better for geographic distribution                       │
└──────────────────────────────────────────────────────────────┘
```

```bash
# Standby1 (relay server)
# postgresql.conf
hot_standby = on

# Standby2 (receive from Standby1)
# set Standby1 address in primary_conninfo
primary_conninfo = 'host=standby1_host ...'
```

---

## 3. Logical Replication

### 3.1 Logical Replication Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   Logical Replication Architecture               │
│                                                                 │
│   Publisher                         Subscriber                 │
│   ┌─────────────┐                   ┌─────────────┐           │
│   │ PostgreSQL  │   Publication     │ PostgreSQL  │           │
│   │             │ ────────────────► │             │           │
│   │  Table A    │   Subscription    │  Table A    │           │
│   │  Table B    │                   │  Table B    │           │
│   └─────────────┘                   └─────────────┘           │
│                                                                 │
│   Features:                                                     │
│   - Table-level selective replication                          │
│   - Replication between different PostgreSQL versions          │
│   - Subscriber can also write                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Publisher Configuration

```sql
-- postgresql.conf
-- wal_level = logical  (required)

-- Create publication
CREATE PUBLICATION my_pub FOR TABLE users, orders;

-- Publish all tables
CREATE PUBLICATION all_tables_pub FOR ALL TABLES;

-- Publish specific operations only
CREATE PUBLICATION insert_only_pub
FOR TABLE products
WITH (publish = 'insert');

-- Row filter (PostgreSQL 15+)
CREATE PUBLICATION active_users_pub
FOR TABLE users WHERE (status = 'active');

-- Column filter (PostgreSQL 15+)
CREATE PUBLICATION partial_pub
FOR TABLE users (id, name, email);

-- Check publications
SELECT * FROM pg_publication;
SELECT * FROM pg_publication_tables;
```

### 3.3 Subscriber Configuration

```sql
-- Create target tables (requires same schema)
CREATE TABLE users (LIKE source_db.users INCLUDING ALL);
CREATE TABLE orders (LIKE source_db.orders INCLUDING ALL);

-- Create subscription
CREATE SUBSCRIPTION my_sub
CONNECTION 'host=publisher_host dbname=source_db user=replicator password=xxx'
PUBLICATION my_pub;

-- Without initial data copy (if already synced)
CREATE SUBSCRIPTION my_sub
CONNECTION '...'
PUBLICATION my_pub
WITH (copy_data = false);

-- Manage subscription
ALTER SUBSCRIPTION my_sub DISABLE;
ALTER SUBSCRIPTION my_sub ENABLE;
ALTER SUBSCRIPTION my_sub REFRESH PUBLICATION;

-- Check subscription status
SELECT * FROM pg_subscription;
SELECT * FROM pg_stat_subscription;
```

### 3.4 Logical Replication Use Cases

```sql
-- 1. Version upgrade (minimal downtime)
-- Set up logical replication old version → new version, then switchover

-- 2. Selective data replication (data warehouse)
CREATE PUBLICATION analytics_pub
FOR TABLE sales, customers, products
WHERE (region = 'APAC');

-- 3. Data consolidation (multiple sources → single target)
-- Source DB 1
CREATE PUBLICATION region1_pub FOR TABLE orders;

-- Source DB 2
CREATE PUBLICATION region2_pub FOR TABLE orders;

-- Target DB
CREATE SUBSCRIPTION sub1 ... PUBLICATION region1_pub;
CREATE SUBSCRIPTION sub2 ... PUBLICATION region2_pub;

-- 4. Real-time reporting database
CREATE PUBLICATION reporting_pub
FOR TABLE transactions, accounts, audit_logs;
```

### 3.5 Conflict Handling

```sql
-- Conflicts can occur in logical replication
-- (since Subscriber allows writes)

-- Check conflicts
SELECT * FROM pg_stat_subscription;
-- srsubstate: 'e' = error

-- Conflict options:
-- 1. Manually resolve conflict row
-- 2. Skip transaction
SELECT pg_replication_origin_advance(
    'pg_' || subid::text,  -- origin name
    '0/XXXXXXX'::pg_lsn    -- LSN to skip
);

-- 3. Restart replication
ALTER SUBSCRIPTION my_sub DISABLE;
-- After resolving issue
ALTER SUBSCRIPTION my_sub ENABLE;
```

---

## 4. Replication Monitoring

### 4.1 Check Replication Status

```sql
-- Primary: WAL sender status
SELECT
    client_addr,
    state,
    sent_lsn,
    write_lsn,
    flush_lsn,
    replay_lsn,
    sync_state,
    pg_wal_lsn_diff(sent_lsn, replay_lsn) AS replay_lag_bytes
FROM pg_stat_replication;

-- Replication lag time (Primary)
SELECT
    client_addr,
    state,
    write_lag,
    flush_lag,
    replay_lag
FROM pg_stat_replication;

-- Standby: current replication status
SELECT
    pg_is_in_recovery() AS is_standby,
    pg_last_wal_receive_lsn() AS received_lsn,
    pg_last_wal_replay_lsn() AS replayed_lsn,
    pg_last_xact_replay_timestamp() AS last_replay_time,
    EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) AS lag_seconds;
```

### 4.2 Monitor Replication Slots

```sql
-- Replication slot status
SELECT
    slot_name,
    slot_type,
    active,
    restart_lsn,
    pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn) AS retained_bytes
FROM pg_replication_slots;

-- Check WAL accumulation from inactive slots
SELECT
    slot_name,
    pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn)) AS retained
FROM pg_replication_slots
WHERE NOT active;

-- Clean up inactive slots (caution!)
SELECT pg_drop_replication_slot('unused_slot');
```

### 4.3 Create Monitoring View

```sql
-- Comprehensive replication monitoring view
CREATE VIEW v_replication_status AS
SELECT
    'physical' AS repl_type,
    client_addr::text,
    application_name,
    state,
    sync_state,
    pg_size_pretty(pg_wal_lsn_diff(sent_lsn, replay_lsn)) AS lag_size,
    COALESCE(replay_lag::text, 'N/A') AS lag_time
FROM pg_stat_replication

UNION ALL

SELECT
    'logical' AS repl_type,
    subconninfo,
    subname,
    CASE WHEN subenabled THEN 'active' ELSE 'disabled' END,
    'async',
    'N/A',
    'N/A'
FROM pg_subscription;
```

---

## 5. Failover and Switchover

### 5.1 Concept Clarification

```
┌─────────────────────────────────────────────────────────────────┐
│ Switchover                                                       │
│ - Planned role transition                                       │
│ - Used for maintenance, upgrades                                │
│ - No data loss                                                  │
│                                                                 │
│ Failover                                                         │
│ - Unplanned role transition during failure                      │
│ - Standby promoted when Primary fails                           │
│ - Possible data loss with async replication                     │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Manual Failover

```bash
# Promote standby (using pg_ctl)
pg_ctl promote -D /var/lib/postgresql/data

# Or using SQL
SELECT pg_promote();

# Or using trigger file (legacy)
touch /var/lib/postgresql/data/promote
```

```sql
-- Verify promotion
SELECT pg_is_in_recovery();  -- false means Primary
```

### 5.3 Recover Old Primary with pg_rewind

```bash
# Convert old Primary to new Standby after failure
# (resolve timeline divergence)

# 1. Stop old Primary
pg_ctl stop -D /var/lib/postgresql/data

# 2. Run pg_rewind
pg_rewind --target-pgdata=/var/lib/postgresql/data \
          --source-server="host=new_primary port=5432 user=replicator"

# 3. Create standby.signal and configure
touch /var/lib/postgresql/data/standby.signal

# 4. Start
pg_ctl start -D /var/lib/postgresql/data
```

### 5.4 Automatic Failover Script Example

```bash
#!/bin/bash
# simple_failover.sh

PRIMARY_HOST="primary"
STANDBY_HOST="standby"
VIP="192.168.1.100"

check_primary() {
    pg_isready -h $PRIMARY_HOST -p 5432 -q
    return $?
}

promote_standby() {
    ssh $STANDBY_HOST "pg_ctl promote -D /var/lib/postgresql/data"
}

move_vip() {
    # Remove VIP from old Primary
    ssh $PRIMARY_HOST "ip addr del $VIP/24 dev eth0" 2>/dev/null
    # Assign VIP to new Primary
    ssh $STANDBY_HOST "ip addr add $VIP/24 dev eth0"
}

# Main logic
if ! check_primary; then
    echo "Primary failure detected, starting failover..."
    promote_standby
    sleep 5
    move_vip
    echo "Failover complete"
fi
```

---

## 6. High Availability Solutions

### 6.1 Patroni

```yaml
# patroni.yml
scope: postgres-cluster
name: node1

restapi:
  listen: 0.0.0.0:8008
  connect_address: node1:8008

etcd:
  hosts: etcd1:2379,etcd2:2379,etcd3:2379

bootstrap:
  dcs:
    ttl: 30
    loop_wait: 10
    retry_timeout: 10
    maximum_lag_on_failover: 1048576
    postgresql:
      use_pg_rewind: true
      parameters:
        wal_level: replica
        hot_standby: on
        max_wal_senders: 5
        max_replication_slots: 5
        wal_keep_size: 1GB

  initdb:
    - encoding: UTF8
    - data-checksums

postgresql:
  listen: 0.0.0.0:5432
  connect_address: node1:5432
  data_dir: /var/lib/postgresql/data
  authentication:
    replication:
      username: replicator
      password: rep_password
    superuser:
      username: postgres
      password: postgres_password
```

```bash
# Check Patroni cluster status
patronictl -c /etc/patroni/patroni.yml list

# Manual switchover
patronictl -c /etc/patroni/patroni.yml switchover

# Manual failover (forcibly remove Primary)
patronictl -c /etc/patroni/patroni.yml failover
```

### 6.2 High Availability Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 Patroni + HAProxy Architecture                  │
│                                                                 │
│   ┌───────────────┐                                            │
│   │   HAProxy     │ ◄── VIP                                    │
│   │  (Load Bal)   │                                            │
│   └───────┬───────┘                                            │
│           │                                                     │
│     ┌─────┴─────┐                                              │
│     │           │                                              │
│   ┌─┴─┐       ┌─┴─┐       ┌───┐                               │
│   │N1 │       │N2 │       │N3 │    PostgreSQL + Patroni       │
│   └─┬─┘       └─┬─┘       └─┬─┘                               │
│     │           │           │                                   │
│   ┌─┴───────────┴───────────┴─┐                               │
│   │      etcd Cluster          │   Distributed consensus store │
│   └───────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 HAProxy Configuration

```
# haproxy.cfg
global
    maxconn 1000

defaults
    mode tcp
    timeout connect 10s
    timeout client 30s
    timeout server 30s

listen postgres_write
    bind *:5432
    option httpchk GET /master
    http-check expect status 200
    default-server inter 3s fall 3 rise 2 on-marked-down shutdown-sessions
    server node1 node1:5432 check port 8008
    server node2 node2:5432 check port 8008
    server node3 node3:5432 check port 8008

listen postgres_read
    bind *:5433
    balance roundrobin
    option httpchk GET /replica
    http-check expect status 200
    default-server inter 3s fall 3 rise 2
    server node1 node1:5432 check port 8008
    server node2 node2:5432 check port 8008
    server node3 node3:5432 check port 8008
```

### 6.4 Integration with PgBouncer

```ini
# pgbouncer.ini
[databases]
mydb = host=haproxy_vip port=5432 dbname=mydb

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = scram-sha-256
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
```

### 6.5 Cloud Environment High Availability

```sql
-- AWS RDS: Multi-AZ automatic failover
-- Automatically configured when enabled

-- Azure Database for PostgreSQL: HA option
-- Select Zone-redundant HA

-- GCP Cloud SQL: Regional HA
-- Automatically configure failover replica

-- Application connection strings
-- Read/Write separation example
-- Primary: postgresql://primary.example.com:5432/mydb
-- Read: postgresql://read.example.com:5432/mydb
```

---

## 7. Practice Problems

### Exercise 1: Configure Streaming Replication
Set up Primary-Standby configuration using Docker.

```bash
# docker-compose.yml
version: '3.8'
services:
  primary:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_INITDB_ARGS: "--data-checksums"
    command: |
      postgres
      -c wal_level=replica
      -c max_wal_senders=3
      -c max_replication_slots=3
      -c hot_standby=on
    ports:
      - "5432:5432"
    volumes:
      - primary_data:/var/lib/postgresql/data

  standby:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: postgres
      PGDATA: /var/lib/postgresql/data
    depends_on:
      - primary
    # standby initialization script required
    ports:
      - "5433:5432"
    volumes:
      - standby_data:/var/lib/postgresql/data

volumes:
  primary_data:
  standby_data:
```

### Exercise 2: Configure Logical Replication
Set up logical replication to replicate only specific tables.

```sql
-- Publisher (source_db)
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    price NUMERIC(10,2),
    category VARCHAR(50)
);

INSERT INTO products (name, price, category) VALUES
    ('Laptop', 999.99, 'Electronics'),
    ('Book', 29.99, 'Books');

CREATE PUBLICATION products_pub FOR TABLE products;

-- Subscriber (target_db)
CREATE TABLE products (LIKE source_db.products);
CREATE SUBSCRIPTION products_sub
CONNECTION 'host=source_host dbname=source_db user=replicator'
PUBLICATION products_pub;
```

### Exercise 3: Replication Monitoring Dashboard
Write a query that comprehensively shows replication status.

```sql
-- Example answer
SELECT
    'Replication Lag' AS metric,
    COALESCE(
        (SELECT pg_size_pretty(pg_wal_lsn_diff(sent_lsn, replay_lsn))
         FROM pg_stat_replication
         LIMIT 1),
        'No standby'
    ) AS value
UNION ALL
SELECT
    'Standby Count',
    (SELECT COUNT(*)::text FROM pg_stat_replication)
UNION ALL
SELECT
    'Replication Slots',
    (SELECT COUNT(*)::text FROM pg_replication_slots);
```

---

## References
- [PostgreSQL Replication](https://www.postgresql.org/docs/current/high-availability.html)
- [Logical Replication](https://www.postgresql.org/docs/current/logical-replication.html)
- [Patroni Documentation](https://patroni.readthedocs.io/)
- [pg_basebackup](https://www.postgresql.org/docs/current/app-pgbasebackup.html)

---

**Previous**: [Query Optimization](./15_Query_Optimization.md) | **Next**: [Window Functions](./17_Window_Functions.md)
