# Kafka Streaming

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the Apache Kafka architecture including Brokers, Topics, Partitions, Producers, Consumers, and Consumer Groups, and describe how replication ensures fault tolerance
2. Implement a Kafka Producer in Python to publish messages with serialization and partitioning strategies
3. Implement a Kafka Consumer with proper offset management and consumer group coordination
4. Configure key Kafka parameters for producers and consumers to tune throughput, latency, and delivery guarantees (at-most-once, at-least-once, exactly-once)
5. Integrate Kafka with Spark Structured Streaming to build end-to-end real-time data pipelines
6. Design a Kafka-based streaming architecture for a real-world use case, including topic design, partition count decisions, and consumer group configuration

---

## Overview

Apache Kafka is a distributed event streaming platform used for building real-time data pipelines and streaming applications. It provides high throughput and fault tolerance.

---

## 1. Kafka Overview

### 1.1 Kafka Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      Kafka Architecture                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Producers                         Consumers                    │
│   ┌─────────┐ ┌─────────┐          ┌─────────┐ ┌─────────┐      │
│   │Producer1│ │Producer2│          │Consumer1│ │Consumer2│      │
│   └────┬────┘ └────┬────┘          └────┬────┘ └────┬────┘      │
│        │           │                    │           │            │
│        └─────┬─────┘                    └─────┬─────┘            │
│              ↓                                ↑                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                    Kafka Cluster                          │  │
│   │  ┌──────────────────────────────────────────────────────┐│  │
│   │  │                    Topic: orders                      ││  │
│   │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐       ││  │
│   │  │  │Partition 0 │ │Partition 1 │ │Partition 2 │       ││  │
│   │  │  │ [0,1,2,3]  │ │ [0,1,2]    │ │ [0,1,2,3,4]│       ││  │
│   │  │  └────────────┘ └────────────┘ └────────────┘       ││  │
│   │  └──────────────────────────────────────────────────────┘│  │
│   │                                                          │  │
│   │  Broker 1         Broker 2         Broker 3              │  │
│   │  ┌──────────┐    ┌──────────┐    ┌──────────┐           │  │
│   │  │ P0(L)    │    │ P1(L)    │    │ P2(L)    │           │  │
│   │  │ P1(R)    │    │ P2(R)    │    │ P0(R)    │           │  │
│   │  └──────────┘    └──────────┘    └──────────┘           │  │
│   │                   L=Leader, R=Replica                    │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              ↑                                   │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                    ZooKeeper / KRaft                      │  │
│   │             (Cluster metadata management)                 │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Concepts

| Concept | Description |
|---------|-------------|
| **Broker** | Kafka server, stores/delivers messages |
| **Topic** | Message category (logical channel) |
| **Partition** | Physical division of topic, parallel processing |
| **Producer** | Message publisher |
| **Consumer** | Message consumer |
| **Consumer Group** | Group of consumers working cooperatively |
| **Offset** | Message position within partition |
| **Replication** | Partition replication for fault tolerance |

---

## 2. Installation and Configuration

### 2.1 Docker Compose Configuration

```yaml
# docker-compose.yaml — single-broker setup for local development.
# Production clusters use 3+ brokers for replication and fault tolerance.
version: '3'

services:
  # ZooKeeper manages broker metadata and leader election. Kafka 3.3+ supports
  # KRaft mode (no ZooKeeper) — consider using KRaft for new clusters to reduce
  # operational complexity.
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000     # Base unit for ZK session timeouts — 2s is the standard default
    ports:
      - "2181:2181"

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092  # What clients use to connect — must be reachable from outside Docker
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1   # Set to 1 for single-broker dev; use 3 in production for durability
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"     # Convenient for dev; disable in production to enforce topic governance

  # Web UI for monitoring topics, consumers, and lag — not needed for production
  # (use Confluent Control Center or Grafana + JMX metrics instead)
  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    ports:
      - "8080:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:9092  # Internal Docker network name, not localhost
```

```bash
# Run
docker-compose up -d

# Create topic (inside container)
docker exec -it kafka kafka-topics --create \
    --bootstrap-server localhost:9092 \
    --topic my-topic \
    --partitions 3 \
    --replication-factor 1
```

### 2.2 Python Client Installation

```bash
# confluent-kafka (recommended)
pip install confluent-kafka

# kafka-python (alternative)
pip install kafka-python
```

---

## 3. Topics and Partitions

### 3.1 Topic Management

```bash
# Create topic
kafka-topics --create \
    --bootstrap-server localhost:9092 \
    --topic orders \
    --partitions 6 \
    --replication-factor 3

# List topics
kafka-topics --list --bootstrap-server localhost:9092

# Describe topic
kafka-topics --describe \
    --bootstrap-server localhost:9092 \
    --topic orders

# Delete topic
kafka-topics --delete \
    --bootstrap-server localhost:9092 \
    --topic orders

# Increase partitions (cannot decrease)
kafka-topics --alter \
    --bootstrap-server localhost:9092 \
    --topic orders \
    --partitions 12
```

### 3.2 Partition Strategy

```python
"""
Partition selection strategy:
1. With key: hash(key) % partitions — guarantees same key always goes to
   the same partition, preserving per-key ordering (critical for event sourcing).
2. Without key: Round-robin — distributes evenly but no ordering guarantee.

Factors for determining partition count:
- Expected throughput / single partition throughput
- Number of consumers (partitions >= consumers — excess consumers sit idle)
- Disk I/O considerations
"""

# Recommended partition count
"""
- Assuming 100MB/s per partition (conservative estimate for modern SSDs)
- Need 1GB/s throughput → minimum 10 partitions
- Consider consumer scalability → 2-3x expected consumer count (room to scale)

Cautions:
- Too many partitions → more open file handles, slower leader elections during
  broker failure, and increased end-to-end latency (each partition adds overhead)
- Too few partitions → consumer parallelism capped, cannot scale beyond partition count
- You CAN increase partitions later but CANNOT decrease — plan for growth
"""
```

---

## 4. Producer

### 4.1 Basic Producer

```python
from confluent_kafka import Producer
import json

# Producer configuration
config = {
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'my-producer',
    'acks': 'all',  # Wait for ALL in-sync replicas to confirm — strongest durability guarantee.
                     # Prevents data loss even if the leader crashes right after writing.
                     # Trade-off: higher latency (~5-10ms vs ~1ms with acks=1).
}

producer = Producer(config)

# Async delivery callback — produce() is non-blocking; this callback fires when the
# broker acknowledges (or rejects) the message. Essential for detecting silent failures.
def delivery_callback(err, msg):
    if err:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}] @ {msg.offset()}')

# Message key determines partition placement — all messages with the same key
# go to the same partition, preserving ordering for that key (e.g., all events
# for order-123 arrive in sequence).
def send_message(topic: str, key: str, value: dict):
    producer.produce(
        topic=topic,
        key=key.encode('utf-8'),       # Keys must be bytes — use a stable encoding for consistent partitioning
        value=json.dumps(value).encode('utf-8'),
        callback=delivery_callback
    )
    # flush() blocks until all buffered messages are delivered — use in
    # single-message scripts. For high-throughput, call poll() periodically instead.
    producer.flush()

# Usage example
send_message(
    topic='orders',
    key='order-123',
    value={
        'order_id': 'order-123',
        'customer_id': 'cust-456',
        'amount': 99.99,
        'timestamp': '2024-01-15T10:30:00Z'
    }
)
```

### 4.2 High-Performance Producer

```python
from confluent_kafka import Producer
import json
import time

class HighThroughputProducer:
    """High-throughput Producer"""

    def __init__(self, bootstrap_servers: str):
        self.config = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': 'high-throughput-producer',

            # Performance settings — each choice trades off between throughput, latency, and durability
            'acks': '1',                    # Leader-only ack: faster but risks data loss if the leader crashes
                                            # before replicating. Use 'all' for critical data (financial, billing).
            'linger.ms': 5,                 # Producer waits up to 5ms to batch messages before sending — increases
                                            # throughput (larger batches = fewer network round-trips) at the cost of
                                            # 5ms additional latency. Set to 0 for minimum latency.
            'batch.size': 16384,            # Max batch size (16KB) — larger batches improve compression and throughput
                                            # but consume more memory per partition. Tune with linger.ms together.
            'buffer.memory': 33554432,      # 32MB total buffer for unsent messages — if the buffer fills (slow broker),
                                            # produce() blocks. Increase for bursty workloads.
            'compression.type': 'snappy',   # Snappy trades minimal CPU overhead for ~2x compression — reduces network
                                            # bandwidth and broker disk usage. Use 'lz4' for even faster compression,
                                            # 'zstd' for best ratio on cold data.

            # Retry settings
            'retries': 3,
            'retry.backoff.ms': 100,
        }
        self.producer = Producer(self.config)
        self.message_count = 0

    def send(self, topic: str, key: str, value: dict):
        """Async send"""
        self.producer.produce(
            topic=topic,
            key=key.encode('utf-8') if key else None,  # None key = round-robin partition assignment
            value=json.dumps(value).encode('utf-8'),
            callback=self._on_delivery
        )
        self.message_count += 1

        # poll(0) triggers callback processing without blocking — must be called
        # periodically to drain the delivery report queue. Without this, the internal
        # queue fills up and produce() eventually blocks or raises BufferError.
        if self.message_count % 1000 == 0:
            self.producer.poll(0)

    def _on_delivery(self, err, msg):
        if err:
            print(f'Delivery failed: {err}')

    def flush(self):
        """Wait for all message delivery"""
        self.producer.flush()

    def close(self):
        self.flush()


# Bulk send example
producer = HighThroughputProducer('localhost:9092')

start = time.time()
for i in range(100000):
    producer.send(
        topic='events',
        key=f'key-{i % 100}',
        value={'event_id': i, 'data': 'test'}
    )

producer.flush()
print(f'Sent 100,000 messages in {time.time() - start:.2f} seconds')
```

---

## 5. Consumer

### 5.1 Basic Consumer

```python
from confluent_kafka import Consumer
import json

# Consumer configuration
config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'my-consumer-group',   # All consumers with the same group.id share partition load
    'auto.offset.reset': 'earliest',   # On first join (no committed offset), start from the beginning.
                                        # Use 'latest' to skip historical data and process only new messages.
    'enable.auto.commit': True,        # Offsets committed every auto.commit.interval.ms — simple but
                                        # risks processing duplicates after crash (message processed but
                                        # offset not yet committed). Use manual commit for exactly-once.
    'auto.commit.interval.ms': 5000,   # 5-second interval — lower = less duplicate risk but more broker load
}

consumer = Consumer(config)

# subscribe() triggers a consumer group rebalance — partitions are distributed
# among all consumers in the group. Adding/removing consumers triggers rebalance.
consumer.subscribe(['orders'])

# poll() is the main loop — it fetches messages, sends heartbeats, and triggers
# rebalance callbacks. The timeout determines how long to wait for new messages
# before returning None (not an error condition).
try:
    while True:
        msg = consumer.poll(timeout=1.0)

        if msg is None:
            continue

        if msg.error():
            print(f'Consumer error: {msg.error()}')
            continue

        # Decode key/value — Kafka stores raw bytes; serialization format is
        # a contract between producer and consumer (JSON here, but Avro/Protobuf
        # are preferred in production for schema enforcement).
        key = msg.key().decode('utf-8') if msg.key() else None
        value = json.loads(msg.value().decode('utf-8'))

        print(f'Received: topic={msg.topic()}, partition={msg.partition()}, '
              f'offset={msg.offset()}, key={key}, value={value}')

except KeyboardInterrupt:
    pass
finally:
    # close() commits final offsets and leaves the consumer group cleanly —
    # without it, the broker waits for session.timeout.ms before rebalancing
    consumer.close()
```

### 5.2 Manual Commit

```python
from confluent_kafka import Consumer
import json

config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'manual-commit-group',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': False,  # Manual commit = at-least-once delivery: messages are only
                                   # committed AFTER successful processing. If the consumer crashes,
                                   # uncommitted messages are redelivered — your processing logic
                                   # must be idempotent to handle this safely.
}

consumer = Consumer(config)
consumer.subscribe(['orders'])

def process_message(value: dict) -> bool:
    """Message processing logic"""
    try:
        # Actual business logic — must be idempotent because the same message
        # may be delivered again if the consumer crashes before committing
        print(f"Processing: {value}")
        return True
    except Exception as e:
        print(f"Processing failed: {e}")
        return False

try:
    while True:
        msg = consumer.poll(timeout=1.0)

        if msg is None:
            continue
        if msg.error():
            continue

        value = json.loads(msg.value().decode('utf-8'))

        # Commit-after-process pattern: only advances the offset when processing
        # succeeds. On failure, the message remains uncommitted and will be
        # redelivered on next poll (or after rebalance).
        if process_message(value):
            consumer.commit(msg)  # Commits this specific offset — more precise than commit()
            # consumer.commit() would commit ALL consumed offsets up to this point
        else:
            # Not committing means this message will be reprocessed on restart.
            # For poison messages (always fail), add a dead-letter queue to avoid infinite loops.
            print("Message processing failed, not committing")

except KeyboardInterrupt:
    pass
finally:
    consumer.close()
```

---

## 6. Consumer Groups

### 6.1 Consumer Group Concept

```
┌────────────────────────────────────────────────────────────────┐
│                    Consumer Group Operation                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Topic: orders (6 partitions)                                 │
│   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐      │
│   │ P0   │ │ P1   │ │ P2   │ │ P3   │ │ P4   │ │ P5   │      │
│   └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘      │
│      │        │        │        │        │        │           │
│   Consumer Group A (3 consumers)                               │
│      │        │        │        │        │        │           │
│      ↓        ↓        ↓        ↓        ↓        ↓           │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│   │ Consumer 1  │  │ Consumer 2  │  │ Consumer 3  │          │
│   │  P0, P1     │  │  P2, P3     │  │  P4, P5     │          │
│   └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                                │
│   Each partition assigned to only one consumer in the group    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

This one-to-one constraint (each partition goes to exactly one consumer within a group) enables horizontal scaling -- add more consumers (up to the partition count) to increase throughput -- while guaranteeing that same-key messages always go to the same consumer, preserving per-key ordering. If you add more consumers than partitions, the extras sit idle; if a consumer fails, its partitions are reassigned to surviving consumers (rebalancing).

### 6.2 Rebalancing

```python
from confluent_kafka import Consumer

# Rebalance callbacks let you handle partition ownership changes gracefully —
# critical for stateful consumers that need to flush buffers or commit offsets
# before losing partition ownership.
def on_assign(consumer, partitions):
    """Partition assignment callback"""
    print(f"Partitions assigned: {[p.partition for p in partitions]}")

def on_revoke(consumer, partitions):
    """Partition revocation callback — fires BEFORE partitions are reassigned"""
    print(f"Partitions revoked: {[p.partition for p in partitions]}")
    # Commit in-progress work NOW — after revocation, another consumer owns
    # these partitions and any uncommitted messages would be processed twice
    consumer.commit()

config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'my-group',
    'auto.offset.reset': 'earliest',
    # cooperative-sticky minimizes partition movement during rebalance — only
    # reassigns partitions that need to move (unlike eager strategy which revokes
    # ALL partitions and redistributes). Reduces rebalance downtime from seconds to ~0.
    'partition.assignment.strategy': 'cooperative-sticky',
}

consumer = Consumer(config)
consumer.subscribe(
    ['orders'],
    on_assign=on_assign,
    on_revoke=on_revoke
)
```

### 6.3 Consumer Group Monitoring

```bash
# List consumer groups
kafka-consumer-groups --list --bootstrap-server localhost:9092

# Describe consumer group
kafka-consumer-groups --describe \
    --bootstrap-server localhost:9092 \
    --group my-consumer-group

# Example output:
# GROUP           TOPIC    PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
# my-group        orders   0          1500            1550            50
# my-group        orders   1          1200            1200            0

# Monitor lag (processing delay)
kafka-consumer-groups --describe \
    --bootstrap-server localhost:9092 \
    --group my-consumer-group \
    --members
```

---

## 7. Real-Time Data Processing Patterns

### 7.1 Event-Based Processing

```python
from confluent_kafka import Consumer, Producer
import json

class EventProcessor:
    """Event-based processing pipeline — implements the consume-transform-produce
    pattern, which is the fundamental building block of Kafka stream processing.
    Each stage reads from one topic and writes to another, forming a processing DAG."""

    def __init__(self, bootstrap_servers: str, group_id: str):
        # Manual commit (enable.auto.commit=False) ensures we only commit after
        # both the transform AND the downstream produce succeed — prevents data loss
        self.consumer = Consumer({
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False,
        })
        self.producer = Producer({
            'bootstrap.servers': bootstrap_servers,
        })

    def process_and_forward(
        self,
        source_topic: str,
        target_topic: str,
        transform_func
    ):
        """Process message and forward to another topic"""
        self.consumer.subscribe([source_topic])

        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    continue

                # Transform
                value = json.loads(msg.value().decode('utf-8'))
                transformed = transform_func(value)

                if transformed:
                    # Preserve the original key to maintain partition affinity —
                    # downstream consumers see the same key ordering as upstream
                    self.producer.produce(
                        topic=target_topic,
                        key=msg.key(),
                        value=json.dumps(transformed).encode('utf-8')
                    )
                    self.producer.poll(0)  # Trigger delivery callbacks without blocking

                # Commit AFTER produce — if we crash between produce and commit,
                # the message is reprocessed (at-least-once). For exactly-once,
                # use Kafka transactions (producer.init_transactions / send_offsets_to_transaction).
                self.consumer.commit(msg)

        except KeyboardInterrupt:
            pass
        finally:
            self.producer.flush()   # Ensure all produced messages are delivered before exit
            self.consumer.close()   # Leave consumer group cleanly to trigger immediate rebalance


# Usage example: Order → Shipment event transformation
def order_to_shipment(order: dict) -> dict:
    """Transform order event to shipment event"""
    return {
        'shipment_id': f"ship-{order['order_id']}",
        'order_id': order['order_id'],
        'customer_id': order['customer_id'],
        'status': 'pending',
        'created_at': order['timestamp']
    }

processor = EventProcessor('localhost:9092', 'order-processor')
processor.process_and_forward('orders', 'shipments', order_to_shipment)
```

### 7.2 Aggregation Processing (Windowing)

```python
from confluent_kafka import Consumer
from collections import defaultdict
from datetime import datetime, timedelta
import json
import threading
import time

class WindowedAggregator:
    """Time window-based aggregation — groups events into fixed time windows and
    computes aggregates per window. This is a simplified version of what Kafka Streams
    and Flink provide natively; use those frameworks for production windowed processing."""

    def __init__(self, window_size_seconds: int = 60):
        self.window_size = window_size_seconds
        # Nested defaultdict: windows[window_start][key] = aggregated_value
        # Using defaultdict avoids key-existence checks on every message
        self.windows = defaultdict(lambda: defaultdict(int))
        self.lock = threading.Lock()  # Thread safety for concurrent consumer + flusher threads

    def add(self, key: str, value: int, timestamp: datetime):
        """Add value — uses event time (from message) not processing time,
        ensuring consistent results regardless of consumer lag or replay speed."""
        window_start = self._get_window_start(timestamp)
        with self.lock:
            self.windows[window_start][key] += value

    def _get_window_start(self, timestamp: datetime) -> datetime:
        """Floor timestamp to window boundary — integer division truncates to
        the nearest window start, so all events within the same window map to
        the same key regardless of their exact timestamp."""
        seconds = int(timestamp.timestamp())
        window_start_seconds = (seconds // self.window_size) * self.window_size
        return datetime.fromtimestamp(window_start_seconds)

    def get_and_clear_completed_windows(self) -> dict:
        """Return and delete windows older than the current one. Windows are
        considered complete once the wall clock moves past them — this assumes
        events arrive roughly in order. Late-arriving events after window closure
        are lost (production systems use watermarks and allowed-lateness to handle this)."""
        current_window = self._get_window_start(datetime.now())
        completed = {}

        with self.lock:
            for window_start, data in list(self.windows.items()):
                if window_start < current_window:
                    completed[window_start] = dict(data)
                    del self.windows[window_start]

        return completed


# Usage example: Aggregate sales per category per minute
aggregator = WindowedAggregator(window_size_seconds=60)

def process_sales():
    consumer = Consumer({
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'sales-aggregator',
        'auto.offset.reset': 'earliest',
    })
    consumer.subscribe(['sales'])

    while True:
        msg = consumer.poll(timeout=1.0)
        if msg and not msg.error():
            value = json.loads(msg.value().decode('utf-8'))
            aggregator.add(
                key=value['category'],
                value=1,
                timestamp=datetime.fromisoformat(value['timestamp'])
            )

        # Output completed windows
        completed = aggregator.get_and_clear_completed_windows()
        for window, data in completed.items():
            print(f"Window {window}: {data}")
```

---

## 8. Kafka Streams and Alternatives

### 8.1 Faust (Python Kafka Streams)

```python
import faust

# Faust provides Kafka Streams-like stream processing in Python — uses asyncio
# for concurrent message handling without threads. Alternative to Java Kafka Streams
# when your team's expertise is Python.
app = faust.App(
    'myapp',
    broker='kafka://localhost:9092',
    value_serializer='json',  # Built-in JSON serialization — use Avro for schema evolution in production
)

# Typed topic declarations enforce a contract between producers and consumers —
# Faust validates message shape at runtime
orders_topic = app.topic('orders', value_type=dict)
processed_topic = app.topic('processed_orders', value_type=dict)

# @app.agent creates an async stream processor — each agent is a consumer that
# processes messages concurrently. Faust manages offset commits automatically.
@app.agent(orders_topic)
async def process_orders(orders):
    async for order in orders:
        processed = {
            **order,
            'processed': True,
            'processed_at': str(datetime.now())
        }
        # Sends to another topic as part of the processing pipeline —
        # Faust handles the produce + commit atomically within a single agent
        await processed_topic.send(value=processed)

# Tables are distributed key-value stores backed by Kafka changelog topics —
# state survives restarts because it's replayed from Kafka on recovery.
# This is equivalent to Kafka Streams KTable.
order_counts = app.Table('order_counts', default=int)

@app.agent(orders_topic)
async def count_orders(orders):
    async for order in orders:
        customer_id = order['customer_id']
        # Table updates are persisted to a changelog topic and a local RocksDB store —
        # local reads are fast, and the changelog enables state recovery on failure
        order_counts[customer_id] += 1

# Run: faust -A myapp worker
```

---

## Practice Problems

### Problem 1: Producer/Consumer
Write a Producer that generates order events and a Consumer that consumes them.

### Problem 2: Consumer Group
Create a Consumer Group with 3 consumers and verify partition assignment.

### Problem 3: Real-Time Aggregation
Write a streaming application that calculates total sales revenue per minute from real-time sales events.

---

## Summary

| Concept | Description |
|---------|-------------|
| **Topic** | Logical category of messages |
| **Partition** | Physical division of topic, unit of parallel processing |
| **Producer** | Message publisher |
| **Consumer** | Message consumer |
| **Consumer Group** | Set of consumers working cooperatively |
| **Offset** | Message position within partition |

---

## References

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Confluent Kafka Python](https://docs.confluent.io/kafka-clients/python/current/overview.html)
- [Kafka: The Definitive Guide](https://www.oreilly.com/library/view/kafka-the-definitive/9781491936153/)
