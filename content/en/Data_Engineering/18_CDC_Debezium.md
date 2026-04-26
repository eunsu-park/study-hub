[← Previous: 17. Spark Structured Streaming](17_Spark_Structured_Streaming.md) | [Next: 19. Lakehouse Practical Patterns →](19_Lakehouse_Practical_Patterns.md)

# CDC with Debezium

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the limitations of traditional batch ETL and compare the three CDC approaches (timestamp-based, trigger-based, log-based), identifying why log-based CDC is preferred for production use.
2. Describe the Debezium architecture and how it reads database Write-Ahead Logs (WAL) to emit structured change events to Kafka topics.
3. Configure a Debezium connector for PostgreSQL or MySQL using Kafka Connect, including slot replication, connector properties, and topic naming conventions.
4. Interpret the Debezium event envelope format (before/after/op fields) to handle insert, update, and delete operations in downstream consumers.
5. Implement schema evolution strategies using the Confluent Schema Registry and Avro to safely propagate source schema changes without breaking consumers.
6. Design production CDC pipelines with considerations for initial snapshots, consumer lag monitoring, and handling connector failures and restarts.

---

## Theory & Principles

CDC's deeper claim is not "near-real-time" — it is **correctness**. Polling for "rows where updated_at > last_checkpoint" misses deletes, misses fast intra-second updates, and races with the writer. Reading the database's own write-ahead log captures *every* mutation in commit order. CDC is the only correct way to mirror an OLTP database's state to anywhere else.

- **(A) Polling vs log-based CDC** — what each approach captures and what it misses.
- **(B) The write-ahead log (WAL) as the source of truth** — PostgreSQL's logical replication, MySQL's binlog, MongoDB's oplog.
- **(C) Debezium's snapshot + stream model** — bootstrapping initial state, then catching up.
- **(D) Event format and schema evolution** — Kafka Connect, the Debezium event envelope, the schema registry.
- **(E) The transactional outbox pattern** — using CDC for application-level event sourcing without distributed transactions.

### A. Polling vs Log-based CDC

#### A.1 Polling: the naive approach

```sql
SELECT * FROM orders WHERE updated_at > '{last_checkpoint}';
-- update last_checkpoint to max(updated_at) in result set
```

Looks fine. What it misses:

- **Deletes.** A deleted row no longer exists; no `updated_at` to query. Polling literally cannot detect deletes.
- **In-flight transactions.** A row updated at 10:00:01.500 and again at 10:00:01.700 within one transaction shows only the latest. Updates in between are lost.
- **Sub-second changes.** If polling interval is 60 seconds, all changes within that window appear "simultaneous"; ordering is lost.
- **Row latency.** A 1-minute polling interval means downstream sees changes 1 minute late on average.
- **Race conditions.** What if a row is being updated at the moment your query reads it? You see either the old or new value, depending on isolation level — and there's no way to know which.

For most analytics use cases, polling is "good enough." For anything requiring correctness — replicating to a secondary database, feeding a streaming system, audit logs — polling is wrong.

#### A.2 Log-based CDC: the correct approach

Every relational database internally writes a *write-ahead log* (WAL): every insert/update/delete is appended to the log *before* the data files are touched. This log is what makes the database crash-safe (recover by replaying the WAL).

Log-based CDC:
1. Connects to the database as a *replication client* (the same way a read replica would).
2. Streams the WAL events: every insert, update, delete, in transaction commit order.
3. Translates to structured change events.
4. Publishes to Kafka (or another sink).

What this captures: **every mutation, in commit order, with full before/after state**. Deletes included. Sub-second updates included. Latency: typically <1 second from commit to event.

This is what Debezium does. Same fundamental approach as MySQL replication, PostgreSQL logical replication, Oracle GoldenGate.

### B. The Write-Ahead Log

The structure of the WAL varies by database:

#### B.1 PostgreSQL: WAL + logical replication

PostgreSQL writes physical WAL (binary delta of pages). For CDC, you need **logical replication**: a separate decoded stream from the WAL that produces row-level events with column values rather than page-level binary diffs.

Setup:
1. Set `wal_level = logical` (requires restart).
2. Create a *replication slot* — a server-side cursor remembering "where Debezium has read up to."
3. Create a *publication* — declares which tables to publish.
4. Debezium connects, reads the slot, decodes events.

The replication slot persists position across Debezium restarts. As long as Debezium is consuming, PostgreSQL retains WAL segments not yet read by the slot. **Critical operational constraint:** if Debezium stops for too long, WAL accumulates on the PostgreSQL side and can fill the disk.

#### B.2 MySQL: binlog

MySQL writes a *binary log* (binlog) of every committed transaction in row format (`binlog_format=ROW`). Debezium reads the binlog directly via the replication protocol — same way a read replica would.

Setup: enable binlog with row format, give Debezium a user with REPLICATION CLIENT and REPLICATION SLAVE privileges.

#### B.3 MongoDB: oplog

MongoDB's *operations log* (oplog) is a capped collection of every mutation. Replica set members read it to stay in sync. Debezium reads the oplog as if it were a replica set member.

#### B.4 The common pattern

All three: the database is already producing a log of every change for its own internal purposes (replication, recovery). CDC tools tap into that log. **No additional load on the primary**, no application changes needed, full mutation history captured.

### C. Debezium's Snapshot + Stream Model

A new CDC consumer needs both *historical state* and *ongoing changes*. Debezium handles this in three phases:

#### C.1 Snapshot phase

On first startup:
1. Acquire a consistent point-in-time snapshot (typically with `FLUSH TABLES WITH READ LOCK` on MySQL, or a snapshot transaction on PostgreSQL).
2. Read every row from every monitored table.
3. Emit each row as a `READ` event to Kafka.
4. Record the WAL/binlog position at snapshot start.

This gives downstream the complete current state.

#### C.2 Stream phase

After snapshot completes:
1. Resume reading the WAL/binlog from the recorded position.
2. Emit each subsequent change as `INSERT`, `UPDATE`, or `DELETE` event.

The handoff is exact — no events between snapshot start position and stream start position are missed.

#### C.3 The duplicate problem

A subtle issue: during snapshot, the WAL has *also been advancing* with new transactions. Debezium's snapshot at position P contains rows committed before P. The stream from position P forward will include those same transactions if their effects happened during the snapshot.

Modern Debezium (2.0+) uses *incremental snapshots* that handle this correctly: they snapshot in chunks while simultaneously consuming the stream, deduplicating with primary key.

### D. Event Format and Schema Evolution

#### D.1 The Debezium envelope

Each CDC event:

```json
{
  "before": { "id": 1, "name": "Alice", "email": "alice@old.com" },
  "after":  { "id": 1, "name": "Alice", "email": "alice@new.com" },
  "source": { "ts_ms": 1700000000000, "table": "users", "db": "prod", "lsn": 12345 },
  "op": "u",
  "ts_ms": 1700000000050
}
```

`op`: `c` (create/insert), `u` (update), `d` (delete), `r` (read/snapshot).
`before` and `after`: row state. `before` is null for inserts; `after` is null for deletes.
`source`: provenance — exactly which database, table, log position.
`ts_ms`: when the event was written.

Consumers can apply the change deterministically: for an update, "the row with PK X went from `before` to `after` at time `ts_ms`."

#### D.2 Schema registry and evolution

The event schemas (especially `before`/`after`) are formal — defined in Avro or JSON Schema, registered in Confluent Schema Registry. When the source table's schema changes (column added/removed), Debezium produces events with the new schema, the registry stores the new version, consumers using compatible deserializers handle the change.

The Schema Registry enforces compatibility: by default, only backward-compatible changes (adding nullable columns, never removing or retyping) are allowed. This catches breaking schema changes at registration time, before they break consumers.

#### D.3 Kafka Connect

Debezium runs as a *Kafka Connect connector* — Connect is the framework that handles the operational concerns: distributed runtime, restart on failure, parallelism (one task per partition), metric reporting. Debezium itself is the source-specific logic (talking to the WAL, decoding events). Kafka Connect provides everything else.

### E. The Transactional Outbox Pattern

A specific application of CDC that solves a notorious problem: how do you publish events to Kafka **atomically** with a database transaction?

The naive approach:

```python
def create_order(order):
    db.insert("orders", order)              # transaction 1
    kafka.publish("order_created", order)   # not in transaction
```

If the Kafka publish fails after the DB insert succeeds, you have an order with no event. If the DB commit fails after Kafka publish, you have an event with no order. Two-phase commit between DB and Kafka is impractical.

The outbox pattern:

```python
def create_order(order):
    with db.transaction():
        db.insert("orders", order)
        db.insert("outbox", {"event": "order_created", "payload": order})
```

Both inserts are in the same DB transaction; either both happen or neither does. The `outbox` table is monitored by Debezium; CDC events from the outbox become Kafka events. Atomicity is preserved.

This is the single most important pattern for event-driven architectures backed by relational databases.

### From Theory to the Code Below

Each section that follows operationalizes one piece of this framework:

- §1 (CDC Concepts) is §A — polling vs log-based.
- §2 (Debezium Architecture) is §B + §D.3 — connectors, Kafka Connect, source-specific implementations.
- §3 (PostgreSQL/MySQL Connectors) is §B.1 + §B.2 — concrete setup.
- §4 (Event Format and Schema Evolution) is §D — envelope, schema registry, compatibility.
- §5 (Snapshot Strategy) is §C — initial snapshot patterns and incremental snapshot.
- §6 (Production Patterns) brings together §E (outbox), monitoring, restart, lag.

---

## Overview

Change Data Capture (CDC) captures row-level changes in databases and streams them as events. Rather than periodic batch extracts, CDC provides near-real-time data synchronization between systems. This lesson covers CDC concepts, Debezium architecture, Kafka Connect integration, event formats, schema evolution, and production patterns.

---

## 1. Change Data Capture Fundamentals

### 1.1 Why CDC?

```python
"""
Traditional ETL vs CDC:

Traditional (Batch Extract):
  ┌──────────┐    Scheduled    ┌──────────┐
  │  Source   │ ──── every ──→ │  Target   │
  │ Database  │    N hours     │ Database  │
  └──────────┘                 └──────────┘
  Problems:
  - High latency (hours)
  - Full table scans → heavy load on source
  - Missing intermediate changes (update then delete)
  - Difficult to capture deletes

CDC (Streaming Changes):
  ┌──────────┐   Real-time    ┌──────────┐   Stream    ┌──────────┐
  │  Source   │ ──── WAL ──→  │  Kafka /  │ ────────→  │  Target  │
  │ Database  │   changes     │  Stream   │            │ Systems  │
  └──────────┘                └──────────┘            └──────────┘
  Benefits:
  - Low latency (seconds)
  - Minimal source load (reads transaction log)
  - Captures ALL changes (insert, update, delete)
  - Preserves change ordering
"""
```

### 1.2 CDC Approaches

```python
"""
Approach Comparison:

1. Timestamp-based (polling):
   SELECT * FROM orders WHERE updated_at > :last_check
   ✗ Misses deletes
   ✗ Requires timestamp column
   ✗ Clock skew issues
   ✓ Simple implementation

2. Trigger-based:
   CREATE TRIGGER capture_change AFTER INSERT ON orders ...
   ✗ Performance overhead on source
   ✗ Schema coupling
   ✓ Captures all operations

3. Log-based (Debezium):
   Read database transaction log (WAL/binlog/redo log)
   ✓ No source schema changes
   ✓ Minimal performance impact
   ✓ Captures all operations including schema changes
   ✓ Preserves transaction ordering
   ✗ Requires log access permissions
   ✗ Database-specific implementation
"""
```

---

## 2. Debezium Architecture

### 2.1 Components

```python
"""
Debezium Architecture:

┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  PostgreSQL  │     │  Kafka Connect   │     │    Kafka      │
│             │────→│  ┌────────────┐  │────→│              │
│  WAL (Write │     │  │ Debezium   │  │     │ Topic per    │
│  Ahead Log) │     │  │ Connector  │  │     │ table:       │
│             │     │  └────────────┘  │     │ dbserver.    │
└─────────────┘     │  ┌────────────┐  │     │ schema.table │
                    │  │ Transforms │  │     │              │
┌─────────────┐     │  └────────────┘  │     └──────┬───────┘
│   MySQL     │────→│  ┌────────────┐  │            │
│  binlog     │     │  │ Debezium   │  │     ┌──────▼───────┐
└─────────────┘     │  │ Connector  │  │     │  Consumers   │
                    │  └────────────┘  │     │ - Spark      │
┌─────────────┐     └──────────────────┘     │ - Flink      │
│ MongoDB     │                              │ - Python     │
│ oplog       │──── (same pattern) ─────────→│ - JDBC Sink  │
└─────────────┘                              └──────────────┘

Key Components:
1. Debezium Connector: Reads database transaction log
2. Kafka Connect: Framework for running connectors
3. Kafka Topics: One topic per captured table
4. Schema Registry: Manages Avro/JSON schemas
"""
```

### 2.2 Supported Databases

```python
"""
Debezium Connectors:

| Database    | Log Type        | Connector               |
|-------------|-----------------|--------------------------|
| PostgreSQL  | WAL (pgoutput)  | debezium-connector-postgres |
| MySQL       | binlog          | debezium-connector-mysql    |
| MongoDB     | oplog/changestr | debezium-connector-mongodb  |
| SQL Server  | CT tables       | debezium-connector-sqlserver|
| Oracle      | LogMiner/XStream| debezium-connector-oracle   |
| Cassandra   | commitlog       | debezium-connector-cassandra|
| Db2         | SQL replication | debezium-connector-db2      |

PostgreSQL Setup Requirements:
  - wal_level = logical (in postgresql.conf)
  - max_replication_slots >= 1
  - GRANT REPLICATION to Debezium user
  - Publication for tables to capture
"""
```

---

## 3. Debezium Configuration

### 3.1 PostgreSQL Connector

```json5
// NOTE: JSON does not support comments — this uses JSON5 notation for explanation.
// In production, submit pure JSON to the Kafka Connect REST API.
{
  "name": "inventory-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "postgres",
    "database.port": "5432",
    "database.user": "debezium",
    "database.password": "dbz_password",
    "database.dbname": "inventory",

    // topic.prefix becomes the first segment of every Kafka topic name
    // (e.g., dbserver1.public.orders). Choose a stable name — changing it
    // later creates new topics, breaking all existing consumers.
    "topic.prefix": "dbserver1",
    "schema.include.list": "public",
    // Explicitly listing tables prevents accidental capture of internal
    // or temporary tables. Omitting this captures ALL tables in the schema.
    "table.include.list": "public.orders,public.customers",

    // pgoutput is PostgreSQL's native logical decoding plugin (built-in since v10).
    // The older decoderbufs plugin requires a separate C extension install.
    "plugin.name": "pgoutput",
    // Each connector needs its own replication slot — sharing slots between
    // connectors causes missed events or duplicate delivery.
    "slot.name": "debezium_slot",
    "publication.name": "dbz_publication",

    "key.converter": "org.apache.kafka.connect.json.JsonConverter",
    // schemas.enable=false on keys keeps Kafka messages compact. Keys are
    // typically just primary key values, so the schema overhead is wasteful.
    "key.converter.schemas.enable": false,
    // schemas.enable=true on values embeds the schema in every message.
    // This is convenient for development but adds ~30-50% message size.
    // In production, use Avro + Schema Registry instead (see Section 7.2).
    "value.converter": "org.apache.kafka.connect.json.JsonConverter",
    "value.converter.schemas.enable": true,

    "snapshot.mode": "initial",
    // Tombstone records (null value after DELETE) are required for Kafka
    // log compaction to actually remove deleted keys from the topic.
    "tombstones.on.delete": true,
    // "string" mode avoids floating-point precision loss for DECIMAL/NUMERIC
    // columns — critical for financial data where $99.99 must not become $99.98999...
    "decimal.handling.mode": "string",
    "time.precision.mode": "connect",

    // Heartbeats prevent WAL accumulation during low-traffic periods.
    // Without them, a quiet table causes the replication slot to hold
    // WAL segments indefinitely, potentially filling the disk.
    "heartbeat.interval.ms": 10000,
    "max.batch.size": 2048,
    "poll.interval.ms": 500
  }
}
```

### 3.2 Snapshot Modes

```python
"""
Snapshot Modes (how Debezium handles initial data):

1. initial (default):
   - Takes full snapshot of existing data on first start
   - Then switches to streaming from transaction log
   - Use when: Starting fresh, need all historical data

2. initial_only:
   - Takes snapshot then stops (no streaming)
   - Use when: One-time data migration

3. no_data:
   - Skips snapshot, starts streaming from current log position
   - Use when: Only need changes from now on

4. when_needed:
   - Takes snapshot if offsets are missing or invalid
   - Use when: Want automatic recovery after data loss

5. never:
   - Never takes snapshot (deprecated, use no_data)

6. schema_only:
   - Captures schema but no data, then streams
   - Use when: Need schema info but only future changes
"""
```

---

## 4. Change Event Format

### 4.1 Event Structure

```python
"""
Debezium Change Event (Kafka message):

Key (identifies the row):
{
  "schema": {...},
  "payload": {
    "order_id": 1001
  }
}

Value (the change):
{
  "schema": {...},
  "payload": {
    "before": {                    ← Row state BEFORE change (null for INSERT)
      "order_id": 1001,
      "status": "pending",
      "amount": 99.99
    },
    "after": {                     ← Row state AFTER change (null for DELETE)
      "order_id": 1001,
      "status": "shipped",
      "amount": 99.99
    },
    "source": {                    ← Source metadata
      "version": "2.5.0.Final",
      "connector": "postgresql",
      "name": "dbserver1",
      "ts_ms": 1706000000000,     ← Database timestamp
      "db": "inventory",
      "schema": "public",
      "table": "orders",
      "txId": 42,                 ← Transaction ID
      "lsn": 33849472,           ← Log Sequence Number
      "sequence": "[...]"
    },
    "op": "u",                     ← Operation: c=create, u=update, d=delete, r=read(snapshot)
    "ts_ms": 1706000000123,       ← Debezium processing timestamp
    "transaction": {              ← Transaction metadata (optional)
      "id": "42",
      "total_order": 1,
      "data_collection_order": 1
    }
  }
}
"""
```

### 4.2 Operation Types

```python
"""
Operations (op field):

  op="c" (CREATE / INSERT):
    before = null
    after  = {new row data}

  op="u" (UPDATE):
    before = {old row data}    ← Requires REPLICA IDENTITY FULL for all columns
    after  = {new row data}

  op="d" (DELETE):
    before = {deleted row data}
    after  = null
    → Followed by a tombstone event (null value) if tombstones.on.delete=true

  op="r" (READ / snapshot):
    before = null
    after  = {row data from snapshot}

  op="t" (TRUNCATE):
    before = null
    after  = null

PostgreSQL REPLICA IDENTITY:
  ALTER TABLE orders REPLICA IDENTITY FULL;
  → Includes ALL columns in 'before' image for updates/deletes
  Without FULL: 'before' only contains primary key columns
"""
```

---

## 5. Consuming CDC Events in Python

### 5.1 Kafka Consumer

```python
import json
from kafka import KafkaConsumer


def create_cdc_consumer(bootstrap_servers, topic):
    """Create a Kafka consumer for Debezium CDC events."""
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        # "earliest" ensures we process the initial snapshot events too.
        # Use "latest" only if you want to skip historical data and start
        # from the current WAL position.
        auto_offset_reset="earliest",
        # Manual commits (enable_auto_commit=False) prevent offset advancement
        # before processing completes — this is essential for at-least-once
        # delivery. With auto-commit, a crash between poll and processing
        # would silently skip those events.
        enable_auto_commit=False,
        group_id="cdc-consumer-group",
        # The null check (if m else None) handles Kafka tombstone records —
        # these have a null value and signal that the row was deleted.
        value_deserializer=lambda m: json.loads(m.decode("utf-8")) if m else None,
        key_deserializer=lambda m: json.loads(m.decode("utf-8")) if m else None,
    )
    return consumer


def process_change_event(event):
    """Process a single Debezium change event."""
    if event is None:
        return  # Tombstone event (null value after a DELETE)

    # Debezium can emit events with or without the schema envelope depending
    # on value.converter.schemas.enable. This fallback handles both formats
    # so the consumer works regardless of connector configuration.
    payload = event.get("payload", event)

    op = payload.get("op")
    before = payload.get("before")
    after = payload.get("after")
    source = payload.get("source", {})

    # Reconstructing the fully-qualified table name from source metadata
    # enables a single consumer to handle events from multiple tables,
    # routing each to the correct target.
    table = f"{source.get('schema', '')}.{source.get('table', '')}"
    tx_id = source.get("txId")

    if op == "c":
        print(f"[INSERT] {table} tx={tx_id}: {after}")
        apply_insert(table, after)
    elif op == "u":
        print(f"[UPDATE] {table} tx={tx_id}: {before} → {after}")
        apply_update(table, before, after)
    elif op == "d":
        print(f"[DELETE] {table} tx={tx_id}: {before}")
        apply_delete(table, before)
    elif op == "r":
        print(f"[SNAPSHOT] {table}: {after}")
        apply_insert(table, after)


def apply_insert(table, row):
    """Apply INSERT to target system."""
    # Example: write to data warehouse, update cache, etc.
    pass


def apply_update(table, old_row, new_row):
    """Apply UPDATE to target system."""
    pass


def apply_delete(table, row):
    """Apply DELETE to target system."""
    pass


def run_consumer():
    """Main consumer loop with manual offset commits."""
    consumer = create_cdc_consumer(
        bootstrap_servers=["localhost:9092"],
        # Topic name follows Debezium convention: {topic.prefix}.{schema}.{table}
        topic="dbserver1.public.orders",
    )

    try:
        for message in consumer:
            process_change_event(message.value)
            # Committing per-message gives strongest at-least-once guarantee
            # but adds latency (~1 Kafka round-trip per event). For higher
            # throughput, batch commits every N messages or every T seconds,
            # accepting that a crash replays the uncommitted batch.
            consumer.commit()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # close() triggers a consumer group rebalance so other consumers
        # in the group immediately pick up this consumer's partitions.
        consumer.close()
```

### 5.2 Building a Materialized View

```python
import json
from collections import defaultdict
from kafka import KafkaConsumer


class MaterializedView:
    """Maintain a materialized view from CDC events.

    Applies CDC events to an in-memory dictionary,
    keeping a current snapshot of the source table.
    """

    def __init__(self, key_field="id"):
        self.key_field = key_field
        # Using a dict keyed by primary key gives O(1) lookup for updates
        # and deletes — essential for keeping up with high-throughput CDC streams.
        self.data = {}  # key → current row
        self.stats = defaultdict(int)

    def apply_event(self, event):
        """Apply a single CDC event to the view."""
        if event is None:
            return

        payload = event.get("payload", event)
        op = payload["op"]
        before = payload.get("before")
        after = payload.get("after")

        # Snapshot reads (op="r") are treated identically to inserts because
        # they represent the initial table state before streaming begins.
        if op in ("c", "r"):
            key = after[self.key_field]
            self.data[key] = after
            self.stats["inserts"] += 1

        elif op == "u":
            # Blindly overwriting with `after` is correct because Debezium
            # preserves transaction ordering — we always see the latest state.
            key = after[self.key_field]
            self.data[key] = after
            self.stats["updates"] += 1

        elif op == "d":
            # For deletes, key comes from `before` because `after` is null.
            # pop() with default None handles the rare case of a delete for
            # a row we never saw (e.g., consumer started mid-stream).
            key = before[self.key_field]
            self.data.pop(key, None)
            self.stats["deletes"] += 1

    def query(self, predicate=None):
        """Query the materialized view."""
        if predicate is None:
            return list(self.data.values())
        return [row for row in self.data.values() if predicate(row)]

    def __len__(self):
        return len(self.data)


# Usage
def run_materialized_view():
    """Build and query a materialized view from CDC events."""
    view = MaterializedView(key_field="order_id")

    # auto_offset_reset="earliest" is critical here: the materialized view
    # must replay ALL events from the beginning (including the initial snapshot)
    # to build a complete picture of the current table state.
    consumer = KafkaConsumer(
        "dbserver1.public.orders",
        bootstrap_servers=["localhost:9092"],
        auto_offset_reset="earliest",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")) if m else None,
    )

    for message in consumer:
        view.apply_event(message.value)

        # Log every 100th insert as a progress indicator. In production,
        # replace this with proper metrics (Prometheus counters) to avoid
        # flooding logs on high-throughput streams.
        if view.stats["inserts"] % 100 == 0:
            print(f"View size: {len(view)}, Stats: {dict(view.stats)}")

            # Query example: find high-value orders
            high_value = view.query(lambda r: r.get("amount", 0) > 1000)
            print(f"High-value orders: {len(high_value)}")
```

---

## 6. Single Message Transforms (SMTs)

### 6.1 Common Transforms

```python
"""
SMTs modify events in the Kafka Connect pipeline:

1. Route by Table:
   "transforms": "route",
   "transforms.route.type": "org.apache.kafka.connect.transforms.RegexRouter",
   "transforms.route.regex": "dbserver1.public.(.*)",
   "transforms.route.replacement": "cdc.$1"
   → Topic: dbserver1.public.orders → cdc.orders

2. Flatten Nested Structure:
   "transforms": "unwrap",
   "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState",
   "transforms.unwrap.drop.tombstones": false,
   "transforms.unwrap.delete.handling.mode": "rewrite",
   "transforms.unwrap.add.fields": "op,source.ts_ms"
   → Extracts 'after' payload, adds __op and __source_ts_ms fields
   → Simplifies downstream processing

3. Filter Events:
   "transforms": "filter",
   "transforms.filter.type": "io.debezium.transforms.Filter",
   "transforms.filter.language": "jsr223.groovy",
   "transforms.filter.condition": "value.op == 'u' && value.after.status == 'cancelled'"
   → Only emit cancelled order updates

4. Add Timestamp:
   "transforms": "timestamp",
   "transforms.timestamp.type": "org.apache.kafka.connect.transforms.InsertField$Value",
   "transforms.timestamp.timestamp.field": "cdc_ingested_at"
   → Adds ingestion timestamp to every event
"""
```

### 6.2 Outbox Pattern

```python
"""
Outbox Pattern: Reliable event publishing from a microservice.

Problem:
  Service needs to update DB AND publish event atomically.
  Two-phase commit is complex and fragile.

Solution:
  1. Service writes to 'outbox' table in same DB transaction
  2. Debezium captures outbox table changes
  3. Outbox SMT routes events to correct Kafka topics

  ┌──────────────────────────┐
  │  Service                 │
  │  BEGIN TRANSACTION       │
  │    INSERT INTO orders    │
  │    INSERT INTO outbox    │  ← Event payload + routing info
  │  COMMIT                  │
  └──────────┬───────────────┘
             │ WAL
             ▼
  ┌──────────────────────────┐
  │  Debezium + Outbox SMT   │
  │  Routes to topic from    │
  │  outbox.aggregate_type   │
  └──────────┬───────────────┘
             │
             ▼
  ┌──────────────────────────┐
  │  Kafka Topic:            │
  │  order.events            │
  └──────────────────────────┘

Outbox Table Schema:
  CREATE TABLE outbox (
    id            UUID PRIMARY KEY,
    aggregate_type VARCHAR(255) NOT NULL,  → Kafka topic
    aggregate_id   VARCHAR(255) NOT NULL,  → Kafka key
    type          VARCHAR(255) NOT NULL,   → Event type header
    payload       JSONB NOT NULL           → Event data
  );

Connector Config:
  "transforms": "outbox",
  "transforms.outbox.type": "io.debezium.transforms.outbox.EventRouter",
  "transforms.outbox.table.field.event.key": "aggregate_id",
  "transforms.outbox.table.field.event.payload": "payload",
  "transforms.outbox.route.topic.replacement": "${routedByValue}.events"
"""
```

---

## 7. Schema Evolution

### 7.1 Handling Schema Changes

```python
"""
Schema Evolution with Debezium:

Debezium captures DDL changes automatically:

1. ADD COLUMN:
   ALTER TABLE orders ADD COLUMN priority VARCHAR(10);
   → New events include the new field
   → Old events had the field as null (backward compatible)

2. DROP COLUMN:
   ALTER TABLE orders DROP COLUMN legacy_field;
   → New events no longer include the field
   → Consumers must handle missing fields

3. RENAME COLUMN:
   ALTER TABLE orders RENAME COLUMN status TO order_status;
   → Treated as DROP old + ADD new
   → Can break consumers expecting old name

4. TYPE CHANGE:
   ALTER TABLE orders ALTER COLUMN amount TYPE NUMERIC(12,4);
   → New events use new type
   → May require consumer updates

Best Practices:
  - Use Schema Registry (Confluent/Apicurio) with compatibility modes
  - BACKWARD compatibility: consumers using new schema can read old data
  - FORWARD compatibility: consumers using old schema can read new data
  - FULL compatibility: both directions (recommended)
  - Always add columns as nullable (backward compatible)
  - Never rename/drop columns in production — deprecate + add new instead
"""
```

### 7.2 Schema Registry Integration

```json5
// Avro + Schema Registry connector — recommended for production over JSON.
{
  "name": "inventory-connector-avro",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "postgres",
    "database.port": "5432",
    "database.user": "debezium",
    "database.password": "dbz_password",
    "database.dbname": "inventory",
    "topic.prefix": "dbserver1",

    // Avro serialization reduces message size by 50-80% compared to JSON
    // because the schema is stored once in the registry, not repeated per message.
    // The registry also enforces compatibility rules, blocking breaking schema
    // changes before they reach consumers.
    "key.converter": "io.confluent.connect.avro.AvroConverter",
    "key.converter.schema.registry.url": "http://schema-registry:8081",
    "value.converter": "io.confluent.connect.avro.AvroConverter",
    "value.converter.schema.registry.url": "http://schema-registry:8081",

    // This internal topic records DDL changes (CREATE, ALTER, DROP) from the
    // source database. Debezium needs this history to correctly interpret WAL
    // entries that reference schemas from before the connector started.
    "schema.history.internal.kafka.bootstrap.servers": "kafka:9092",
    "schema.history.internal.kafka.topic": "schema-changes.inventory"
  }
}
```

---

## 8. Production Patterns

### 8.1 Monitoring Debezium

```python
"""
Monitoring Checklist:

1. Connector Status (Kafka Connect REST API):
   GET /connectors/inventory-connector/status
   → Check state: RUNNING, PAUSED, FAILED

2. Key Metrics (JMX / Prometheus):
   - debezium.metrics.MilliSecondsBehindSource
     → CDC lag (how far behind real-time)
   - debezium.metrics.NumberOfEventsFiltered
   - debezium.metrics.TotalNumberOfEventsSeen
   - debezium.metrics.QueueTotalCapacity / QueueRemainingCapacity
   - debezium.metrics.SnapshotCompleted

3. Consumer Lag (Kafka):
   kafka-consumer-groups.sh --describe --group cdc-consumer-group
   → Monitor LAG column per partition

4. Replication Slot (PostgreSQL):
   SELECT slot_name, active, pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn)
   FROM pg_replication_slots;
   → If WAL retention grows unbounded, the slot is stuck

Alerts:
  - MilliSecondsBehindSource > 60000 → CDC lag > 1 minute
  - Connector state != RUNNING → Restart needed
  - WAL retention > 1GB → Slot may be stuck
"""
```

### 8.2 Failure Recovery

```python
"""
Recovery Scenarios:

1. Connector Crash:
   - Kafka Connect restarts the task automatically
   - Debezium resumes from last committed offset
   - No data loss (exactly-once within Kafka)

2. Database Failover:
   - Connector detects disconnection
   - Set: "database.hostname" to VIP or DNS
   - After reconnect, resumes from last WAL position
   - May need snapshot if WAL position is no longer available

3. Kafka Unavailable:
   - Connector pauses, buffers in memory (up to queue capacity)
   - Resumes when Kafka is back
   - If buffer overflows, connector fails and needs restart

4. Replication Slot Lost:
   - Must take a new snapshot
   - Set snapshot.mode = "when_needed"
   - Or manually: DELETE offset → restart connector

5. Schema Change Breaks Consumer:
   - Use Schema Registry with compatibility checks
   - Deploy consumer updates before producer changes
   - Use ExtractNewRecordState SMT to flatten events
"""
```

### 8.3 Docker Compose Setup

```yaml
# docker-compose.yml for Debezium + PostgreSQL + Kafka
version: '3.8'
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: inventory
    # wal_level=logical is the minimum WAL level that supports logical
    # replication — without it, Debezium cannot decode row-level changes.
    # max_replication_slots and max_wal_senders must be >= number of
    # Debezium connectors + any other replication consumers (e.g., read replicas).
    command:
      - "postgres"
      - "-c"
      - "wal_level=logical"
      - "-c"
      - "max_replication_slots=4"
      - "-c"
      - "max_wal_senders=4"
    ports:
      - "5432:5432"

  zookeeper:
    image: confluentinc/cp-zookeeper:7.6.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka:
    image: confluentinc/cp-kafka:7.6.0
    depends_on: [zookeeper]
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      # Replication factor of 1 is acceptable for local dev only.
      # In production, use 3 to survive broker failures without data loss.
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    ports:
      - "9092:9092"

  connect:
    image: debezium/connect:2.5
    # Both kafka and postgres must be healthy before Kafka Connect starts —
    # Connect will fail to register connectors if the broker is unreachable,
    # and connectors will fail if the source database isn't ready.
    depends_on: [kafka, postgres]
    environment:
      BOOTSTRAP_SERVERS: kafka:9092
      GROUP_ID: "debezium-connect"
      # These three internal topics store connector state. If you delete them,
      # all connectors lose their offset positions and must re-snapshot.
      CONFIG_STORAGE_TOPIC: connect_configs
      OFFSET_STORAGE_TOPIC: connect_offsets
      STATUS_STORAGE_TOPIC: connect_statuses
    ports:
      - "8083:8083"
```

---

## 9. Practice Problems

### Exercise 1: Build a CDC Pipeline

```python
"""
Set up a complete CDC pipeline:
1. Start PostgreSQL with logical replication enabled
2. Create an 'orders' table and insert sample data
3. Deploy Debezium connector via Kafka Connect REST API
4. Write a Python consumer that:
   a. Reads CDC events from Kafka
   b. Maintains an in-memory materialized view
   c. Logs all INSERT/UPDATE/DELETE operations
5. Verify: INSERT, UPDATE, DELETE in PostgreSQL → events appear in consumer
"""
```

### Exercise 2: Outbox Pattern Implementation

```python
"""
Implement the transactional outbox pattern:
1. Create an 'outbox' table in PostgreSQL
2. Write a service that:
   a. Inserts into 'orders' table
   b. Inserts event into 'outbox' table
   c. Both in the same transaction
3. Configure Debezium with EventRouter SMT
4. Verify: events appear on the correct Kafka topic
5. Bonus: add event deduplication in the consumer
"""
```

---

## Exercises

### Exercise 1: Deploy and Verify a Debezium Connector

Set up a minimal CDC pipeline from scratch using Docker Compose and verify all change types:

1. Start the Docker Compose stack from Section 8.3 (PostgreSQL, Zookeeper, Kafka, Kafka Connect)
2. Create an `inventory` database with an `orders` table:
   ```sql
   CREATE TABLE orders (
       order_id   SERIAL PRIMARY KEY,
       customer   VARCHAR(100),
       amount     NUMERIC(10,2),
       status     VARCHAR(20),
       created_at TIMESTAMP DEFAULT NOW()
   );
   ALTER TABLE orders REPLICA IDENTITY FULL;
   ```
3. Register the Debezium PostgreSQL connector via the Kafka Connect REST API (`POST /connectors`)
4. Perform INSERT, UPDATE, and DELETE operations on the `orders` table and consume the resulting Kafka messages using `kafka-console-consumer`
5. For each operation, record the `op` field value and describe the structure of the `before` and `after` fields
6. Explain in your notes: why does `REPLICA IDENTITY FULL` matter specifically for UPDATE and DELETE events?

### Exercise 2: Build a Materialized Cache with the Consumer

Extend the `MaterializedView` class from Section 5.2 to support multi-table CDC consumption:

1. Modify `MaterializedView` to accept a `table_name` parameter and store data in a dict keyed by `(table_name, primary_key)`
2. Write a `MultiTableConsumer` class that subscribes to multiple Debezium topics (e.g., `dbserver1.public.orders` and `dbserver1.public.customers`)
3. Route each CDC event to the correct `MaterializedView` instance based on the `source.table` field in the event payload
4. Add a `query_join(order_id)` method that looks up an order and then fetches the corresponding customer from the customers view — simulating a cross-table join in memory
5. Implement a `stats()` method that returns per-table counts of inserts, updates, and deletes
6. Test by inserting 10 orders and 5 customers, updating 3 orders, and deleting 1 customer — then verify the view sizes and stats match expectations

### Exercise 3: Apply Single Message Transforms

Configure three SMTs on your Debezium connector and observe their effects:

1. **RegexRouter**: Route all captured topics from `dbserver1.public.*` to `cdc.*` (e.g., `cdc.orders`, `cdc.customers`)
2. **ExtractNewRecordState**: Unwrap the Debezium envelope so consumers receive a flat record (just the `after` fields) with two extra fields: `__op` and `__source_ts_ms`
3. **InsertField**: Add a `cdc_processed_at` field with the current wall-clock time to every event
4. Chain all three transforms in the connector config (transforms property accepts a comma-separated list)
5. Consume messages from the transformed topics and compare the message structure before and after applying the SMTs
6. Explain in comments: when would you prefer to use `ExtractNewRecordState` vs consuming the full envelope with `before`/`after` fields?

### Exercise 4: Schema Evolution and Registry Integration

Test how Debezium handles schema changes and integrate Confluent Schema Registry:

1. With your connector running, add a new nullable column to the `orders` table:
   ```sql
   ALTER TABLE orders ADD COLUMN priority VARCHAR(10);
   ```
2. Insert new rows and observe that the Kafka messages now include the `priority` field; verify that old rows (without `priority`) had it as null
3. Configure the connector to use Avro serialization with the Schema Registry (update `key.converter` and `value.converter` to use `AvroConverter`)
4. Attempt a breaking schema change (drop a non-nullable column) and observe how the Schema Registry's compatibility check blocks or allows it depending on the compatibility mode
5. Set the compatibility mode to `FULL` and verify that only backward-and-forward compatible changes are accepted
6. Explain: what is the difference between `BACKWARD`, `FORWARD`, and `FULL` compatibility modes? Give one example DDL change that each mode would accept but the others might not

### Exercise 5: Implement the Transactional Outbox Pattern

Build a microservice that uses the outbox pattern to publish order events atomically:

1. Create an `orders` table and an `outbox` table in PostgreSQL:
   ```sql
   CREATE TABLE outbox (
       id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
       aggregate_type VARCHAR(255) NOT NULL,
       aggregate_id   VARCHAR(255) NOT NULL,
       type           VARCHAR(255) NOT NULL,
       payload        JSONB NOT NULL,
       created_at     TIMESTAMP DEFAULT NOW()
   );
   ```
2. Write a Python function `create_order(customer_id, amount)` that:
   - Inserts a row into `orders`
   - Inserts a corresponding event into `outbox` — both inside a single `BEGIN`/`COMMIT` transaction
3. Configure a Debezium connector to capture only the `outbox` table with the `EventRouter` SMT, routing events to the topic named by `aggregate_type` (e.g., `order.events`)
4. Write a Kafka consumer that reads from `order.events` and prints each event's `type`, `aggregate_id`, and `payload`
5. Simulate a failure: call `create_order` but crash the service before committing — verify no event appears in Kafka
6. Explain: how does the outbox pattern solve the dual-write problem? What would go wrong if you wrote to Kafka directly from the application instead of using an outbox table?

---

## 10. Summary

### Key Takeaways

| Concept | Description |
|---------|-------------|
| **Log-based CDC** | Read transaction log (WAL/binlog) for minimal impact |
| **Debezium** | Open-source CDC platform on Kafka Connect |
| **Event format** | before/after snapshots with operation type and source metadata |
| **Snapshot modes** | initial, no_data, schema_only for different starting points |
| **SMTs** | Transform events in-pipeline (flatten, route, filter) |
| **Outbox pattern** | Atomic DB write + event publish via outbox table |
| **Schema evolution** | Use Schema Registry with compatibility modes |

### Best Practices

1. **Set REPLICA IDENTITY FULL** on captured tables — ensures complete before/after images
2. **Monitor WAL retention** — unbounded growth means a stuck replication slot
3. **Use Schema Registry** — prevents breaking changes from propagating
4. **ExtractNewRecordState SMT** — simplifies downstream processing
5. **Heartbeat events** — prevent WAL accumulation during low-activity periods
6. **Test schema changes** in staging before production

### Next Steps

- **L19**: Lakehouse Practical Patterns — land CDC streams in Delta Lake / Iceberg
- Return to **L12** (Data Quality with Great Expectations) to validate CDC data downstream

---

[← Previous: 17. Spark Structured Streaming](17_Spark_Structured_Streaming.md) | [Next: 19. Lakehouse Practical Patterns →](19_Lakehouse_Practical_Patterns.md)
