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

Before the configuration commands, read [**Theory & Principles**](#theory--principles) — the WAL streaming protocol, the latency-vs-durability tradeoff in synchronous vs asynchronous replication, the publication/subscription model behind logical replication, and how etcd-Raft (used by Patroni) prevents split-brain.

1. [Replication Overview](#1-replication-overview)
2. [Physical Replication (Streaming Replication)](#2-physical-replication-streaming-replication)
3. [Logical Replication](#3-logical-replication)
4. [Replication Monitoring](#4-replication-monitoring)
5. [Failover and Switchover](#5-failover-and-switchover)
6. [High Availability Solutions](#6-high-availability-solutions)
7. [Practice Problems](#7-practice-problems)

---

## Theory & Principles

A "replicated PostgreSQL cluster" is really three coordinated systems running together: WAL streaming from primary to standby, a consensus layer (Patroni + etcd/Consul) to decide who is primary at any moment, and a connection router (HAProxy or pgBouncer) that sends clients to the right server. Each layer has its own failure modes and tuning knobs. Understanding the WAL streaming protocol, the synchronous-vs-asynchronous tradeoff, and the Raft consensus that prevents split-brain is what separates "we have a standby" from "we have automatic failover that works during the worst day".

This section covers:

- **(A)** WAL streaming: how the standby's `walreceiver` continuously fetches and replays WAL from the primary's `walsender`.
- **(B)** Synchronous vs asynchronous: the latency-vs-durability tradeoff, plus the `synchronous_commit` levels.
- **(C)** Logical replication: publication/subscription, decoding WAL into row-level change events.
- **(D)** Consensus and Raft: how Patroni + etcd elect a primary and prevent split-brain.

### A. WAL Streaming Replication

#### A.1 The protocol

The primary runs a process called **walsender** for each connected standby. The standby runs a **walreceiver** that connects via the special replication protocol (variant of the libpq frontend protocol).

```
┌──────────────────────┐                ┌──────────────────────┐
│ Primary              │                │ Standby              │
│                      │                │                      │
│ commit → WAL buffer  │                │  walreceiver         │
│       → pg_wal/      │  ──────────►   │   ↓                  │
│       → walsender    │  WAL records   │  startup process     │
│                      │                │   ↓                  │
│                      │                │  apply (redo)        │
│                      │  ◄────────     │                      │
│                      │  flush LSN     │                      │
└──────────────────────┘                └──────────────────────┘
```

The standby's `walreceiver` writes incoming WAL to its local `pg_wal/` and notifies the **startup process** (which is the same code path used during crash recovery) to redo the new records. The startup process applies them, advancing the standby's database state.

The standby periodically sends back its current LSN: how much WAL it has *received*, *flushed* to disk, and *applied*. The primary uses these LSNs to compute replication lag and to decide when synchronous commits can complete.

#### A.2 Replication slots

By default, the primary keeps WAL only as long as it might be needed by the next checkpoint. If a standby falls behind and the primary recycles WAL the standby still needs, the standby breaks and must be rebuilt from a fresh base backup.

A **replication slot** is a primary-side reservation that says "do not recycle WAL beyond LSN X until I (the slot owner) acknowledge". The primary refuses to recycle that WAL even if it fills the disk. This solves the falling-behind problem at the cost of needing disk monitoring — a stuck standby can fill `pg_wal/` and crash the primary.

### B. Synchronous vs Asynchronous

#### B.1 Asynchronous (default)

The primary returns COMMIT to the client as soon as the WAL is on the primary's disk. The standby gets the WAL "soon" (typically milliseconds later). If the primary crashes between commit and the standby receiving the WAL, the committed transaction is lost.

- **Pro**: low latency. Commit time is independent of standby health.
- **Con**: data loss possible during failover.

#### B.2 Synchronous

The primary waits for at least one (or N) standby to acknowledge before returning COMMIT. The exact behavior depends on `synchronous_commit`:

| `synchronous_commit` | Wait for standby until... |
|----------------------|---------------------------|
| `off` | Don't even wait for primary fsync (data loss on primary crash possible) |
| `local` | Primary fsync only (no wait for standby) |
| `remote_write` | WAL has reached standby's OS (not necessarily disk) |
| `on` (default) | WAL has been fsync'd on standby |
| `remote_apply` | WAL has been applied (visible on standby) |

`synchronous_standby_names` configures which standbys count.

#### B.3 The tradeoff

Synchronous commits guarantee zero data loss on failover but every commit waits for the network round-trip plus the standby's fsync. For a transaction-heavy workload, this can double commit latency. The pragmatic compromise: use `remote_write` (waits for OS-level receipt, not full fsync) on a low-latency network, or run synchronous only for critical transactions via `SET LOCAL synchronous_commit = 'on';`.

### C. Logical Replication

Physical replication ships *byte-level page changes*. Logical replication ships *row-level change events* — INSERT, UPDATE, DELETE statements that can be applied to a different schema, a different major version, or even a different database engine.

#### C.1 The publication/subscription model

```
Primary side:                         Subscriber side:
CREATE PUBLICATION pub                CREATE SUBSCRIPTION sub
  FOR TABLE orders, customers;          CONNECTION 'host=primary ...'
                                        PUBLICATION pub;
```

The subscriber connects via the replication protocol, the primary's **logical decoder** reads its WAL and converts each row change into a logical message (INSERT INTO orders VALUES (…), …), and the subscriber applies it as a normal SQL statement.

#### C.2 What you can do with logical that you cannot with physical

- **Replicate a subset of tables** (publication selects which tables).
- **Replicate to a different schema** (subscriber's table can have extra columns, different defaults).
- **Replicate across major versions** (PG 14 → PG 16).
- **Two-way / multi-master** with conflict resolution (with extensions like pglogical).
- **Capture changes for ETL** (debezium consumes the same logical decoding output to feed Kafka).

#### C.3 Limitations

- **No DDL replication**. Schema changes must be applied to both sides manually (or via tools).
- **Primary keys required** for UPDATE/DELETE replication (otherwise PostgreSQL doesn't know which row to update on the subscriber).
- **Higher overhead than physical** because of the per-row decoding step.

### D. Consensus and Split-Brain Prevention

The hardest problem in HA is not "swap to the standby when the primary fails" — it is "all nodes agree on who is currently primary, even when the network is partitioning". This is a **consensus problem**.

#### D.1 Why naive failover splits the brain

If node A (the primary) becomes unreachable, B (a standby) might decide A is dead and promote itself. But A might still be alive and accepting writes from a different network segment. Now both A and B are accepting writes — **split-brain**. Reconciling the two divergent timelines is hard or impossible.

#### D.2 Raft (used by etcd, Consul)

Raft is a consensus protocol where a quorum (majority, e.g., 2 of 3) of nodes must agree on every state change. Decisions cannot be made by a partitioned minority — if A is cut off from B and C, A cannot keep being primary because it can no longer get acknowledgments from a quorum.

The protocol has three roles: **leader**, **follower**, **candidate**. The leader serves all writes. Followers replicate from the leader. If a follower stops hearing from the leader (election timeout), it becomes a candidate, requests votes from others, and the one that gets a majority becomes the new leader.

#### D.3 Patroni

Patroni is a Python daemon that runs alongside each PostgreSQL instance. Each Patroni writes its node's state (primary/standby, last applied LSN, etc.) to a shared key in etcd or Consul. Patroni reads back the cluster state and decides: should I promote myself? should I demote? should I follow a different primary?

Because etcd is itself Raft-backed, the "who is primary" decision is consensus-protected — no Patroni can promote without holding the leader lease in etcd, and the lease is granted only by Raft majority. A partitioned Patroni cannot promote, period.

#### D.4 Fencing

The final layer: when Patroni promotes B, it must ensure A is not still serving writes. Mechanisms include STONITH ("Shoot The Other Node In The Head" via IPMI), virtual IP failover (so clients cannot reach A even if it is up), or VIP-based pool routing (HAProxy reads from etcd to know where the primary is).

### From Theory to the Configuration Below

Each of the following sections is one of these mechanisms made concrete:

- **`wal_level`, `max_wal_senders`, `wal_keep_size`** — primary-side configuration for WAL streaming (§A).
- **`primary_conninfo`, `restore_command`** — standby-side recovery configuration (§A).
- **`synchronous_standby_names`, `synchronous_commit`** — pick the durability/latency point (§B).
- **`CREATE PUBLICATION`, `CREATE SUBSCRIPTION`** — logical replication setup (§C).
- **`pg_create_physical_replication_slot`, `pg_create_logical_replication_slot`** — slot management (§A.2).
- **Patroni configuration** — DCS (etcd/Consul/ZooKeeper) connection, automatic failover policy (§D.3).
- **HAProxy / Pgpool / connection router** — direct clients to the current primary; depends on Patroni's DCS state (§D.4).

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
