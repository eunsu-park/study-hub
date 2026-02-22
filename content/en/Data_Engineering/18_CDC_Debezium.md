# CDC with Debezium

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

```json
{
  "name": "inventory-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "postgres",
    "database.port": "5432",
    "database.user": "debezium",
    "database.password": "dbz_password",
    "database.dbname": "inventory",

    "topic.prefix": "dbserver1",
    "schema.include.list": "public",
    "table.include.list": "public.orders,public.customers",

    "plugin.name": "pgoutput",
    "slot.name": "debezium_slot",
    "publication.name": "dbz_publication",

    "key.converter": "org.apache.kafka.connect.json.JsonConverter",
    "key.converter.schemas.enable": false,
    "value.converter": "org.apache.kafka.connect.json.JsonConverter",
    "value.converter.schemas.enable": true,

    "snapshot.mode": "initial",
    "tombstones.on.delete": true,
    "decimal.handling.mode": "string",
    "time.precision.mode": "connect",

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
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id="cdc-consumer-group",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")) if m else None,
        key_deserializer=lambda m: json.loads(m.decode("utf-8")) if m else None,
    )
    return consumer


def process_change_event(event):
    """Process a single Debezium change event."""
    if event is None:
        return  # Tombstone event

    payload = event.get("payload", event)  # Handle with/without envelope

    op = payload.get("op")
    before = payload.get("before")
    after = payload.get("after")
    source = payload.get("source", {})

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
        topic="dbserver1.public.orders",
    )

    try:
        for message in consumer:
            process_change_event(message.value)
            # Commit after processing each message
            consumer.commit()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
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

        if op in ("c", "r"):
            key = after[self.key_field]
            self.data[key] = after
            self.stats["inserts"] += 1

        elif op == "u":
            key = after[self.key_field]
            self.data[key] = after
            self.stats["updates"] += 1

        elif op == "d":
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

    consumer = KafkaConsumer(
        "dbserver1.public.orders",
        bootstrap_servers=["localhost:9092"],
        auto_offset_reset="earliest",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")) if m else None,
    )

    for message in consumer:
        view.apply_event(message.value)

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

```json
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

    "key.converter": "io.confluent.connect.avro.AvroConverter",
    "key.converter.schema.registry.url": "http://schema-registry:8081",
    "value.converter": "io.confluent.connect.avro.AvroConverter",
    "value.converter.schema.registry.url": "http://schema-registry:8081",

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
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    ports:
      - "9092:9092"

  connect:
    image: debezium/connect:2.5
    depends_on: [kafka, postgres]
    environment:
      BOOTSTRAP_SERVERS: kafka:9092
      GROUP_ID: "debezium-connect"
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
