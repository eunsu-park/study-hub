# Kafka Streams and ksqlDB

## Overview

Kafka Streams is a client library for building real-time stream processing applications on top of Apache Kafka. ksqlDB extends this with a SQL interface for stream processing. This lesson covers stream processing concepts, Faust (Python Kafka Streams), windowed aggregations, joins, and ksqlDB for interactive queries.

---

## 1. Stream Processing Concepts

### 1.1 Bounded vs Unbounded Data

```python
"""
Batch Processing (Bounded):
  - Fixed dataset with known start and end
  - Process once, produce result
  - Example: daily sales report from yesterday's data

Stream Processing (Unbounded):
  - Continuous flow of data with no end
  - Process incrementally as data arrives
  - Example: real-time fraud detection on live transactions

Key Concepts:
  - Event Time: When the event actually occurred (embedded in data)
  - Processing Time: When the system processes the event
  - Ingestion Time: When Kafka receives the event
  - Watermark: A threshold that declares "no more events before time T"
  - Late Data: Events arriving after the watermark has passed
"""
```

### 1.2 Stream Processing Architectures

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Event Sourcing** | Store all events, derive state | Financial ledger, audit trail |
| **CQRS** | Separate read/write models | High-read workloads |
| **Materialized Views** | Pre-computed query results from streams | Real-time dashboards |
| **CDC → Stream** | Database changes as events | Data replication, sync |

---

## 2. Kafka Streams Architecture

### 2.1 Core Concepts

```python
"""
Kafka Streams Topology:

  Source (Kafka Topic)
      │
      ▼
  ┌─────────────┐
  │  Processor   │ ← Stateless: filter, map, flatMap
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │  Processor   │ ← Stateful: aggregate, join, window
  │ + StateStore │
  └──────┬──────┘
         │
      ▼
  Sink (Kafka Topic)

Key Abstractions:
  - KStream: Unbounded sequence of events (insert semantics)
  - KTable: Changelog stream (upsert semantics, latest value per key)
  - GlobalKTable: Full copy of a table on every instance
  - State Store: Local key-value store for stateful operations (RocksDB)

KStream vs KTable:
  KStream: [key=A, val=1] [key=A, val=2] → both records exist
  KTable:  [key=A, val=1] [key=A, val=2] → only val=2 exists (upsert)
"""
```

### 2.2 Exactly-Once Semantics

```python
"""
Kafka Streams provides exactly-once processing:

1. Read from input topic
2. Process and update state store
3. Write to output topic
4. Commit consumer offset

All four steps happen atomically using Kafka transactions.

Configuration:
  processing.guarantee = exactly_once_v2  (Kafka 2.5+)

Note: Exactly-once is between Kafka input and Kafka output.
External systems (databases, APIs) need idempotent writes.
"""
```

---

## 3. Faust: Python Kafka Streams

### 3.1 Basic Faust Application

```python
"""
Faust is a Python stream processing library inspired by Kafka Streams.

# pip install faust-streaming  (maintained fork of original faust)
"""

import faust

# Create Faust app (connects to Kafka)
app = faust.App(
    'my_stream_app',
    broker='kafka://localhost:9092',
    store='rocksdb://',  # State store backend
    topic_replication_factor=1,
)

# Define data model
class Order(faust.Record):
    order_id: str
    user_id: str
    amount: float
    timestamp: float

# Define input topic
orders_topic = app.topic('orders', value_type=Order)

# Simple stream processor: filter high-value orders
@app.agent(orders_topic)
async def process_orders(orders):
    async for order in orders:
        if order.amount > 1000:
            print(f"High-value order: {order.order_id} = ${order.amount}")

# Run: faust -A myapp worker -l info
```

### 3.2 Stateful Processing: Tables and Aggregations

```python
import faust
from datetime import timedelta

app = faust.App('aggregation_demo', broker='kafka://localhost:9092')

class PageView(faust.Record):
    user_id: str
    page: str
    timestamp: float

page_views_topic = app.topic('page_views', value_type=PageView)

# Table: persistent key-value store (backed by Kafka changelog topic)
page_view_counts = app.Table(
    'page_view_counts',
    default=int,
    partitions=8,
)

# Count page views per page
@app.agent(page_views_topic)
async def count_views(views):
    async for view in views:
        page_view_counts[view.page] += 1

# Expose counts via HTTP endpoint
@app.page('/counts/')
async def get_counts(self, request):
    return self.json({k: v for k, v in page_view_counts.items()})

# Windowed table: count per page per hour
hourly_counts = app.Table(
    'hourly_page_views',
    default=int,
).hopping(
    size=timedelta(hours=1),
    step=timedelta(minutes=10),
    expires=timedelta(hours=24),
)

@app.agent(page_views_topic)
async def count_hourly(views):
    async for view in views:
        hourly_counts[view.page] += 1
```

### 3.3 Stream Operations

```python
import faust
from typing import AsyncIterable

app = faust.App('operations_demo', broker='kafka://localhost:9092')

class Event(faust.Record):
    event_type: str
    user_id: str
    value: float

events_topic = app.topic('events', value_type=Event)
filtered_topic = app.topic('high_value_events', value_type=Event)

# Filter
@app.agent(events_topic)
async def filter_events(events):
    async for event in events.filter(lambda e: e.value > 100):
        await filtered_topic.send(value=event)

# Group by
@app.agent(events_topic)
async def group_by_type(events):
    async for event in events.group_by(Event.event_type):
        # Events are now partitioned by event_type
        print(f"Type: {event.event_type}, Value: {event.value}")

# Map / Transform
@app.agent(events_topic)
async def enrich_events(events):
    async for event in events:
        enriched = {
            "original": event.to_representation(),
            "category": "premium" if event.value > 500 else "standard",
            "processed_at": faust.current_event().message.timestamp,
        }
        print(enriched)
```

---

## 4. Windowed Aggregations

### 4.1 Window Types

```python
"""
Window Types in Stream Processing:

1. Tumbling Window (fixed, non-overlapping)
   [0-5min] [5-10min] [10-15min]
   Each event belongs to exactly one window.

2. Hopping Window (fixed, overlapping)
   [0-10min] [5-15min] [10-20min]  (size=10min, step=5min)
   Each event may belong to multiple windows.

3. Session Window (activity-based, variable size)
   [--user active--] gap [--user active--]
   Window closes after inactivity timeout.

4. Sliding Window (event-triggered)
   Window is centered around each event within a time range.
   Used primarily for joins.
"""

import faust
from datetime import timedelta

app = faust.App('windowing_demo', broker='kafka://localhost:9092')

class Transaction(faust.Record):
    user_id: str
    amount: float
    timestamp: float

transactions_topic = app.topic('transactions', value_type=Transaction)

# Tumbling window: total spending per user per 5 minutes
tumbling_spending = app.Table(
    'tumbling_spending',
    default=float,
).tumbling(
    size=timedelta(minutes=5),
    expires=timedelta(hours=1),
)

@app.agent(transactions_topic)
async def aggregate_tumbling(transactions):
    async for txn in transactions:
        tumbling_spending[txn.user_id] += txn.amount
        current = tumbling_spending[txn.user_id].current()
        print(f"User {txn.user_id} 5-min spending: ${current:.2f}")

# Hopping window: total spending per user, 10-min window every 5 min
hopping_spending = app.Table(
    'hopping_spending',
    default=float,
).hopping(
    size=timedelta(minutes=10),
    step=timedelta(minutes=5),
    expires=timedelta(hours=1),
)

@app.agent(transactions_topic)
async def aggregate_hopping(transactions):
    async for txn in transactions:
        hopping_spending[txn.user_id] += txn.amount
```

---

## 5. Stream Joins

### 5.1 Join Patterns

```python
"""
Stream Join Types:

1. KStream-KStream Join (windowed)
   - Both sides are event streams
   - Must specify a time window for matching
   - Example: Match orders with shipments within 24 hours

2. KStream-KTable Join (enrichment)
   - Stream events enriched with table lookups
   - Table always has latest value per key
   - Example: Enrich orders with latest user profile

3. KTable-KTable Join
   - Both sides are changelog streams
   - Result is also a KTable
   - Example: Join user profiles with user preferences
"""

import faust

app = faust.App('join_demo', broker='kafka://localhost:9092')

class Order(faust.Record):
    order_id: str
    user_id: str
    amount: float

class UserProfile(faust.Record):
    user_id: str
    name: str
    tier: str

orders_topic = app.topic('orders', value_type=Order)
profiles_topic = app.topic('profiles', value_type=UserProfile)

# KTable for user profiles (latest profile per user)
user_profiles = app.Table('user_profiles', default=None)

@app.agent(profiles_topic)
async def update_profiles(profiles):
    async for profile in profiles:
        user_profiles[profile.user_id] = profile

# KStream-KTable join: enrich orders with user profiles
@app.agent(orders_topic)
async def enrich_orders(orders):
    async for order in orders:
        profile = user_profiles.get(order.user_id)
        if profile:
            print(f"Order {order.order_id}: {profile.name} ({profile.tier}) - ${order.amount}")
        else:
            print(f"Order {order.order_id}: Unknown user {order.user_id} - ${order.amount}")
```

---

## 6. ksqlDB Fundamentals

### 6.1 Architecture and Concepts

```python
"""
ksqlDB = SQL engine for Kafka Streams

Architecture:
  Kafka Topics → ksqlDB Server → SQL Interface → Kafka Topics
                      ↓
                 State Stores (RocksDB)

Key Concepts:
  - STREAM: Unbounded sequence of events (like KStream)
  - TABLE: Materialized view of latest values per key (like KTable)
  - Push Query: Continuous query that emits updates (SELECT ... EMIT CHANGES)
  - Pull Query: Point-in-time lookup (SELECT ... WHERE key = ?)
  - Persistent Query: Long-running query that writes results to a topic
"""
```

### 6.2 ksqlDB SQL Examples

```sql
-- Create a STREAM from a Kafka topic
CREATE STREAM orders_stream (
    order_id VARCHAR KEY,
    user_id VARCHAR,
    product_id VARCHAR,
    amount DOUBLE,
    order_time TIMESTAMP
) WITH (
    KAFKA_TOPIC = 'orders',
    VALUE_FORMAT = 'JSON',
    TIMESTAMP = 'order_time'
);

-- Create a TABLE (latest state per key)
CREATE TABLE user_profiles (
    user_id VARCHAR PRIMARY KEY,
    name VARCHAR,
    email VARCHAR,
    tier VARCHAR
) WITH (
    KAFKA_TOPIC = 'user_profiles',
    VALUE_FORMAT = 'JSON'
);

-- Filter stream: create a new stream of high-value orders
CREATE STREAM high_value_orders AS
SELECT *
FROM orders_stream
WHERE amount > 1000
EMIT CHANGES;

-- Windowed aggregation: orders per user per hour
CREATE TABLE hourly_user_orders AS
SELECT
    user_id,
    COUNT(*) AS order_count,
    SUM(amount) AS total_amount,
    WINDOWSTART AS window_start,
    WINDOWEND AS window_end
FROM orders_stream
WINDOW TUMBLING (SIZE 1 HOUR)
GROUP BY user_id
EMIT CHANGES;

-- Stream-Table join: enrich orders with user profiles
CREATE STREAM enriched_orders AS
SELECT
    o.order_id,
    o.user_id,
    u.name AS user_name,
    u.tier AS user_tier,
    o.amount,
    o.order_time
FROM orders_stream o
LEFT JOIN user_profiles u ON o.user_id = u.user_id
EMIT CHANGES;

-- Pull query: point-in-time lookup (REST API)
SELECT * FROM hourly_user_orders WHERE user_id = 'user_123';

-- Push query: continuous updates (Server-Sent Events)
SELECT * FROM enriched_orders WHERE amount > 500 EMIT CHANGES;
```

### 6.3 ksqlDB Connectors

```sql
-- Source connector: Import data from PostgreSQL into Kafka
CREATE SOURCE CONNECTOR pg_source WITH (
    'connector.class' = 'io.debezium.connector.postgresql.PostgresConnector',
    'database.hostname' = 'postgres',
    'database.port' = '5432',
    'database.user' = 'replicator',
    'database.password' = 'secret',
    'database.dbname' = 'mydb',
    'table.include.list' = 'public.orders,public.users',
    'topic.prefix' = 'pg'
);

-- Sink connector: Export processed data to Elasticsearch
CREATE SINK CONNECTOR es_sink WITH (
    'connector.class' = 'io.confluent.connect.elasticsearch.ElasticsearchSinkConnector',
    'connection.url' = 'http://elasticsearch:9200',
    'topics' = 'enriched_orders',
    'type.name' = '_doc',
    'key.ignore' = 'true'
);
```

---

## 7. Monitoring Stream Applications

### 7.1 Key Metrics

```python
"""
Critical metrics for stream processing:

1. Consumer Lag: How far behind the consumer is from the latest offset
   - Target: Near zero for real-time processing
   - Monitor: kafka-consumer-groups.sh --describe --group <group>

2. Throughput: Records processed per second
   - Input throughput vs output throughput
   - Should be stable (spikes = backpressure)

3. Processing Latency: Time from event to output
   - Event time to processing time gap
   - Includes Kafka produce/consume + processing

4. State Store Size: RocksDB memory and disk usage
   - Monitor for unbounded growth
   - Set TTL / window expiration

5. Rebalance Frequency: How often consumer group rebalances
   - Frequent rebalances = instability
   - Can be caused by processing timeouts
"""

# Faust monitoring with Prometheus
"""
from faust.sensors.prometheus import setup_prometheus_sensors

app = faust.App('monitored_app', broker='kafka://localhost:9092')

# Enable Prometheus metrics endpoint on port 6066
setup_prometheus_sensors(app, host='0.0.0.0', port=6066)

# Metrics exported:
# - faust_messages_received_total
# - faust_messages_sent_total
# - faust_events_runtime_seconds
# - faust_consumer_lag
# - faust_assignment_latency_seconds
"""
```

---

## 8. Practice Problems

### Exercise 1: Real-Time Fraud Detection

```python
"""
Build a Faust stream processor for fraud detection:
1. Input: transaction stream (user_id, amount, merchant, timestamp)
2. Rules:
   a) Single transaction > $5000 → flag immediately
   b) More than 3 transactions in 5 minutes → flag
   c) Transaction from a new country (maintain user location table)
3. Output: fraud alerts stream with reason and confidence score
4. Maintain a table of user spending patterns (rolling 24h total)
"""
```

### Exercise 2: Clickstream Sessionization (ksqlDB)

```sql
/*
Given a clickstream topic with: user_id, page_url, timestamp

1. Create a SESSION window with 30-minute gap
2. For each session, compute:
   - session_duration (first event to last)
   - page_count (distinct pages visited)
   - bounce (TRUE if only 1 page view)
3. Create a derived stream of completed sessions
4. Build a table of user engagement metrics (avg session duration, bounce rate)
*/
```

---

## 9. Summary

### Key Takeaways

| Concept | Description |
|---------|-------------|
| **KStream** | Unbounded event stream (insert semantics) |
| **KTable** | Changelog stream (upsert, latest value per key) |
| **Faust** | Python Kafka Streams library for stream processing |
| **Windowing** | Tumbling, hopping, session windows for time-bounded aggregation |
| **Stream Joins** | KStream-KStream (windowed), KStream-KTable (enrichment) |
| **ksqlDB** | SQL interface for Kafka Streams (STREAM + TABLE + queries) |
| **Push vs Pull** | Continuous updates vs point-in-time lookups |
| **Exactly-once** | Kafka transactions ensure no duplicates |

### Best Practices

1. **Use event time** (not processing time) for windowed operations
2. **Set window expiration** to prevent unbounded state growth
3. **Monitor consumer lag** — it's the primary health indicator
4. **Use KTable for enrichment** lookups (not KStream-KStream join)
5. **Start with ksqlDB** for prototyping, move to Faust/Java for production
6. **Handle late data** explicitly — configure grace periods for windows

### Next Steps

- **L17**: Spark Structured Streaming — DataFrame-based stream processing
- **L18**: CDC with Debezium — database changes as Kafka streams
