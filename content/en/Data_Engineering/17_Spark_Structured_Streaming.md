# Spark Structured Streaming

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the unbounded table model in Spark Structured Streaming and how the engine performs incremental computation over continuously arriving data.
2. Implement streaming queries using PySpark's DataFrame API, selecting appropriate output modes (append, complete, update) for different aggregation patterns.
3. Apply event-time windowing and watermarking to handle late-arriving data and compute accurate windowed aggregations.
4. Integrate Spark Structured Streaming with Apache Kafka as both a source and a sink, using appropriate serialization formats.
5. Design stateful stream processing using mapGroupsWithState or flatMapGroupsWithState for custom session and user-defined aggregation logic.
6. Configure checkpointing and fault-tolerance mechanisms to achieve end-to-end exactly-once processing guarantees.

---

## Overview

Spark Structured Streaming extends the DataFrame API to handle unbounded data streams. It treats a stream as a continuously growing table, enabling batch-like queries on streaming data. This lesson covers the programming model, sources/sinks, windowed aggregations, stateful processing, and Kafka integration.

---

## 1. The Structured Streaming Model

### 1.1 Unbounded Table Concept

```python
"""
Structured Streaming treats a live data stream as a table that grows:

  Time t0: | Alice | 10 |
  Time t1: | Alice | 10 |     ← New row appended
           | Bob   | 20 |
  Time t2: | Alice | 10 |
           | Bob   | 20 |     ← New row appended
           | Alice | 15 |

Queries on this "table" are incrementally updated as new data arrives.
The engine figures out what to compute incrementally — you just write
the query as if it were a static DataFrame.

Processing Modes:
  - Micro-batch (default): Process data in small batches (~100ms-seconds)
  - Continuous (experimental): Process row-by-row (~1ms latency)
"""
```

### 1.2 Basic Example

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split

spark = SparkSession.builder \
    .appName("StructuredStreaming") \
    .getOrCreate()

# Socket source is unbuffered and offers no replay — suitable only for demos.
# In production, use Kafka or file sources which support offset tracking for
# exactly-once semantics and fault-tolerant recovery.
lines = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# explode() turns each word into its own row — this is necessary because
# groupBy operates on rows, not on elements within a single column value.
words = lines.select(explode(split(lines.value, " ")).alias("word"))
word_counts = words.groupBy("word").count()

# "complete" mode re-emits the ENTIRE aggregation result on every trigger.
# This works here because the word count table is small enough to fit in memory.
# For high-cardinality aggregations, prefer "update" mode to emit only changed rows.
query = word_counts.writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

# awaitTermination() blocks the main thread — without it the driver exits
# immediately and the streaming query is killed before processing any data.
query.awaitTermination()
```

---

## 2. Sources and Sinks

### 2.1 Built-in Sources

```python
"""
Sources (readStream):

1. Kafka:
   spark.readStream.format("kafka")
     .option("kafka.bootstrap.servers", "broker:9092")
     .option("subscribe", "topic1,topic2")
     .option("startingOffsets", "latest")
     .load()

2. File (CSV, JSON, Parquet, ORC):
   spark.readStream.format("json")
     .schema(schema)  # Schema required for file sources
     .option("path", "/data/input/")
     .option("maxFilesPerTrigger", 10)
     .load()

3. Socket (testing only):
   spark.readStream.format("socket")
     .option("host", "localhost")
     .option("port", 9999)
     .load()

4. Rate (testing - generates rows per second):
   spark.readStream.format("rate")
     .option("rowsPerSecond", 100)
     .load()
"""
```

### 2.2 Built-in Sinks

```python
"""
Sinks (writeStream):

1. Kafka:
   df.writeStream.format("kafka")
     .option("kafka.bootstrap.servers", "broker:9092")
     .option("topic", "output_topic")
     .option("checkpointLocation", "/checkpoints/kafka_sink")
     .start()

2. File (Parquet, JSON, CSV):
   df.writeStream.format("parquet")
     .option("path", "/data/output/")
     .option("checkpointLocation", "/checkpoints/file_sink")
     .trigger(processingTime="1 minute")
     .start()

3. Console (debugging):
   df.writeStream.format("console")
     .outputMode("complete")
     .trigger(processingTime="10 seconds")
     .start()

4. Memory (testing):
   df.writeStream.format("memory")
     .queryName("my_table")
     .start()
   # Then: spark.sql("SELECT * FROM my_table")

5. Foreach / ForeachBatch (custom logic):
   df.writeStream.foreachBatch(process_batch_fn).start()
"""
```

---

## 3. Operations on Streams

### 3.1 Stateless Operations

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp, upper
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

spark = SparkSession.builder.appName("StreamOps").getOrCreate()

# Schema must be declared upfront for Kafka sources because Kafka values
# are opaque byte arrays — Spark cannot infer the schema by sampling.
# Getting the schema wrong here silently produces null columns, so keep
# this definition in sync with the producer's serialization format.
schema = StructType([
    StructField("user_id", StringType()),
    StructField("action", StringType()),
    StructField("amount", DoubleType()),
    StructField("timestamp", StringType()),
])

# Read from Kafka
raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "events") \
    .load()

# Kafka delivers key/value as binary — cast to string first, then parse
# the JSON envelope. The two-step select ("data" alias then "data.*")
# flattens nested struct columns into top-level columns for easier downstream use.
events = raw.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# Stateless transformations (filter, map, withColumn) run identically to
# batch DataFrames. They require no state tracking and support append mode,
# which is the most memory-efficient output mode.
filtered = events.filter(col("amount") > 0)
transformed = filtered.withColumn(
    "event_time", to_timestamp(col("timestamp"))
).withColumn(
    "action_upper", upper(col("action"))
)

# processingTime="5 seconds" sets the micro-batch interval. Shorter intervals
# reduce latency but increase scheduling overhead. For most use cases,
# 1-30 seconds balances freshness against throughput.
query = transformed.writeStream \
    .outputMode("append") \
    .format("console") \
    .trigger(processingTime="5 seconds") \
    .start()
```

### 3.2 Output Modes

```python
"""
Output Modes:

1. Append (default):
   - Only new rows added since last trigger
   - Works with: select, where, map, filter
   - Does NOT work with: aggregations without watermark

2. Complete:
   - Entire result table output every trigger
   - Works with: aggregations
   - Note: Entire result must fit in memory

3. Update:
   - Only rows updated since last trigger
   - Works with: aggregations (outputs only changed rows)
   - More efficient than Complete for large aggregations

Choosing output mode:
  - No aggregation → Append
  - Aggregation + dashboard → Complete
  - Aggregation + sink that handles upserts → Update
"""
```

---

## 4. Windowed Aggregations

### 4.1 Tumbling and Sliding Windows

```python
from pyspark.sql.functions import window, col, count, sum as spark_sum, avg

# Tumbling window: non-overlapping 10-minute windows
tumbling = events \
    .withWatermark("event_time", "5 minutes") \
    .groupBy(
        # Tumbling windows partition time into fixed, non-overlapping buckets.
        # Each event belongs to exactly one window — simpler and cheaper than
        # sliding windows, which duplicate events across overlapping windows.
        window(col("event_time"), "10 minutes"),
        col("action")
    ) \
    .agg(
        count("*").alias("event_count"),
        spark_sum("amount").alias("total_amount"),
        avg("amount").alias("avg_amount"),
    )

# Sliding window: 10-minute window, sliding every 5 minutes.
# The slide interval (5 min) < window duration (10 min) creates overlap,
# so each event appears in 2 windows. This doubles state size and processing
# cost compared to tumbling, but provides smoother trend detection.
sliding = events \
    .withWatermark("event_time", "5 minutes") \
    .groupBy(
        window(col("event_time"), "10 minutes", "5 minutes"),
        col("user_id")
    ) \
    .agg(
        count("*").alias("event_count"),
        spark_sum("amount").alias("total_amount"),
    )

# "update" mode emits only windows that changed in this micro-batch.
# This is far more efficient than "complete" for windowed aggregations
# because old, finalized windows are not re-emitted every trigger.
query = tumbling.writeStream \
    .outputMode("update") \
    .format("console") \
    .option("truncate", "false") \
    .trigger(processingTime="10 seconds") \
    .start()
```

### 4.2 Watermarking for Late Data

```python
"""
Watermark: "Allow data up to X late"

Without watermark:
  - Engine must keep ALL state forever → memory overflow
  - Late data accepted indefinitely

With watermark:
  - Engine drops data later than watermark threshold
  - Old state is cleaned up
  - Required for windowed aggregations in Append mode

Example:
  .withWatermark("event_time", "10 minutes")
  → Accept events up to 10 minutes late
  → Drop events that arrive > 10 minutes after the window closes
  → Clean up state for windows older than current time - 10 minutes
"""

# Example: watermark with window
windowed_with_watermark = events \
    .withWatermark("event_time", "10 minutes") \
    .groupBy(
        window(col("event_time"), "5 minutes"),
        col("user_id")
    ) \
    .count()

# In Append mode, results are emitted only when the watermark
# passes the end of the window + late threshold
query = windowed_with_watermark.writeStream \
    .outputMode("append") \
    .format("parquet") \
    .option("path", "/data/output/windowed/") \
    .option("checkpointLocation", "/checkpoints/windowed/") \
    .trigger(processingTime="1 minute") \
    .start()
```

---

## 5. Stream-Stream Joins

### 5.1 Inner Join with Time Constraint

```python
from pyspark.sql.functions import expr

# Stream 1: Orders
orders = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "orders") \
    .load() \
    .select(from_json(col("value").cast("string"), order_schema).alias("o")) \
    .select("o.*") \
    .withColumn("order_time", to_timestamp(col("timestamp")))

# Stream 2: Shipments
shipments = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "shipments") \
    .load() \
    .select(from_json(col("value").cast("string"), shipment_schema).alias("s")) \
    .select("s.*") \
    .withColumn("ship_time", to_timestamp(col("timestamp")))

# Both streams MUST have watermarks for stream-stream joins; without them
# the engine would buffer ALL events forever waiting for a potential match.
# The watermark values can differ per stream — set each based on how late
# that particular source's events typically arrive.
orders_wm = orders.withWatermark("order_time", "2 hours")
shipments_wm = shipments.withWatermark("ship_time", "3 hours")

# The time-range condition (ship_time within 24h of order_time) is critical:
# it bounds how long state is kept per order. Without it, every unmatched
# order would remain in state indefinitely. Tightening this window reduces
# memory usage but risks missing late-arriving shipment events.
joined = orders_wm.join(
    shipments_wm,
    expr("""
        orders.order_id = shipments.order_id AND
        ship_time >= order_time AND
        ship_time <= order_time + interval 24 hours
    """),
    "inner"
)
```

### 5.2 Stream-Static Join

```python
# Static DataFrame (loaded once, not streaming)
product_catalog = spark.read.parquet("/data/products/")

# Stream enrichment: join streaming orders with static product data
enriched = events.join(
    product_catalog,
    events.product_id == product_catalog.product_id,
    "left"
)
# No watermark needed for stream-static joins
```

---

## 6. Stateful Processing

### 6.1 foreachBatch for Custom Logic

```python
def process_batch(batch_df, batch_id):
    """Process each micro-batch with custom logic."""
    # Early return on empty batches avoids JDBC connection overhead and
    # prevents creating empty Parquet files that slow down later reads.
    if batch_df.isEmpty():
        return

    # foreachBatch is the only way to write to multiple sinks atomically
    # from a single streaming query. Without it, you'd need two separate
    # queries consuming the same source (doubling Kafka read traffic).
    batch_df.write.mode("append").parquet(f"/data/output/batch_{batch_id}/")

    # JDBC sink is not natively supported in Structured Streaming —
    # foreachBatch bridges this gap by giving you a standard DataFrame
    # that supports all batch write formats including JDBC.
    batch_df.write \
        .format("jdbc") \
        .option("url", "jdbc:postgresql://localhost:5432/warehouse") \
        .option("dbtable", "events") \
        .option("user", "etl_user") \
        .option("password", "secret") \
        .mode("append") \
        .save()

    print(f"Batch {batch_id}: {batch_df.count()} rows processed")

# foreachBatch inherits exactly-once guarantees from checkpointing: if the
# batch fails mid-write, it will be replayed from the checkpoint on restart.
# Your batch function must be idempotent (e.g., use MERGE or dedup by batch_id)
# to avoid duplicate writes during replay.
query = events.writeStream \
    .foreachBatch(process_batch) \
    .option("checkpointLocation", "/checkpoints/foreach/") \
    .trigger(processingTime="30 seconds") \
    .start()
```

### 6.2 Deduplication

```python
# Including event_time in dropDuplicates is required so the watermark
# can expire old entries from state. Using event_id alone would force
# the engine to retain every seen ID indefinitely (unbounded memory).
deduplicated = events \
    .withWatermark("event_time", "10 minutes") \
    .dropDuplicates(["event_id", "event_time"])

# The 10-minute watermark means duplicates arriving >10 minutes late will
# NOT be caught. Choose this threshold based on your producer's retry window —
# if retries happen within 5 minutes, a 10-minute watermark provides safe margin.
```

---

## 7. Monitoring and Checkpointing

### 7.1 Query Progress

```python
# Get query status
query = events.writeStream.format("console").start()

# Programmatic monitoring
print(query.status)        # Current status
print(query.lastProgress)  # Last micro-batch metrics
print(query.recentProgress)  # Recent progress history

"""
lastProgress example:
{
  "id": "abc-123",
  "runId": "def-456",
  "batchId": 42,
  "durationMs": {"triggerExecution": 1200, "getBatch": 50, "queryPlanning": 30},
  "numInputRows": 1000,
  "inputRowsPerSecond": 500.0,
  "processedRowsPerSecond": 833.3,
  "sources": [{"description": "KafkaV2[Subscribe[events]]", "startOffset": {...}, "endOffset": {...}}],
  "stateOperators": [{"numRowsTotal": 5000, "numRowsUpdated": 100}]
}
"""
```

### 7.2 Checkpoint and Recovery

```python
"""
Checkpointing ensures exactly-once fault tolerance:

1. Stores: offsets read, state of aggregations, committed offsets to sinks
2. Location: HDFS, S3, or local filesystem
3. On failure: Reads checkpoint, replays from last committed offset
4. Required for: production deployments, stateful operations

query = df.writeStream
    .option("checkpointLocation", "s3://my-bucket/checkpoints/query_1/")
    .start()

Rules:
- One checkpoint location per query (never share!)
- Checkpoint is tied to the query plan — changing the query may require a new checkpoint
- Use durable storage (HDFS/S3) in production, not local filesystem
"""
```

---

## 8. Kafka Integration Deep Dive

### 8.1 Reading from Kafka

```python
# Multiple brokers provide failover — if broker1 is down, Spark
# connects via broker2 without query interruption.
kafka_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "broker1:9092,broker2:9092") \
    .option("subscribe", "orders,shipments") \
    # "latest" skips historical data on first start — use "earliest" if you
    # need to backfill. After first run, offsets resume from checkpoint regardless.
    .option("startingOffsets", "latest") \
    # Caps how many records are read per micro-batch. Without this limit,
    # a large backlog (e.g., after downtime) can cause OOM by pulling
    # millions of records into a single batch.
    .option("maxOffsetsPerTrigger", 10000) \
    .option("kafka.group.id", "spark-streaming-group") \
    # failOnDataLoss=false prevents the query from crashing when Kafka
    # retention deletes offsets the checkpoint references. Set to true in
    # pipelines where data loss must halt processing for investigation.
    .option("failOnDataLoss", "false") \
    .load()

# Kafka DataFrame columns:
# key (binary), value (binary), topic, partition, offset, timestamp, timestampType

# Parse JSON value
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

schema = StructType([
    StructField("order_id", StringType()),
    StructField("amount", DoubleType()),
])

# Preserving kafka_key and kafka_timestamp alongside parsed fields enables
# downstream debugging (trace a bad record back to its Kafka partition/offset)
# and event-time processing (kafka_timestamp reflects when Kafka received the message).
parsed = kafka_stream.select(
    col("key").cast("string").alias("kafka_key"),
    from_json(col("value").cast("string"), schema).alias("data"),
    col("topic"),
    col("timestamp").alias("kafka_timestamp"),
).select("kafka_key", "data.*", "topic", "kafka_timestamp")
```

### 8.2 Writing to Kafka

```python
# Kafka sink requires exactly two columns: "key" and "value" (both strings
# or bytes). Using order_id as the key ensures all events for the same order
# land on the same Kafka partition, preserving per-order event ordering.
output = parsed.select(
    col("order_id").alias("key"),
    col("amount").cast("string").alias("value"),
)

# Each streaming query must have its OWN checkpoint directory — sharing
# checkpoints between queries causes offset corruption and data loss.
query = output.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "broker:9092") \
    .option("topic", "processed_orders") \
    .option("checkpointLocation", "/checkpoints/kafka_output/") \
    .trigger(processingTime="10 seconds") \
    .start()
```

---

## 9. Practice Problems

### Exercise 1: Real-Time Dashboard Metrics

```python
"""
Build a Structured Streaming pipeline:
1. Read clickstream events from Kafka (user_id, page, action, timestamp)
2. Compute 5-minute tumbling window aggregations:
   - Page views per page
   - Unique users per page
   - Most popular action type
3. Write results to both console (debugging) and Parquet (persistence)
4. Add 2-minute watermark for late events
5. Monitor processing rate and state size
"""
```

### Exercise 2: Stream-Stream Join

```python
"""
Match ad impressions with ad clicks:
1. Stream 1: impressions (ad_id, user_id, impression_time)
2. Stream 2: clicks (ad_id, user_id, click_time)
3. Join: match clicks within 30 minutes of impression
4. Compute click-through rate (CTR) per ad in 1-hour windows
5. Write high-CTR ads (>5%) to an alert topic
"""
```

---

## Exercises

### Exercise 1: Kafka-to-Parquet Streaming Pipeline

Build a complete Structured Streaming pipeline that reads from Kafka and writes results to Parquet:

1. Read from a Kafka topic `user_events` with schema `(event_id STRING, user_id STRING, action STRING, amount DOUBLE, event_time TIMESTAMP)`
2. Apply a 3-minute watermark on `event_time` to handle late arrivals
3. Compute tumbling 10-minute window aggregations grouped by `(action, window)`:
   - Event count
   - Sum of amount
   - Count of distinct users (hint: use `approx_count_distinct`)
4. Write results to Parquet at `/data/output/event_summary/` using Update output mode
5. Configure a checkpoint location and set `maxOffsetsPerTrigger=5000`
6. Monitor the query with `query.lastProgress` and print the `inputRowsPerSecond` after each micro-batch

### Exercise 2: Stateful Deduplication with Watermark

Implement an exactly-once deduplication pipeline for a high-volume event stream:

1. Read from Kafka topic `raw_events` where the same `event_id` may appear multiple times due to producer retries
2. Parse the JSON payload with schema `(event_id STRING, user_id STRING, amount DOUBLE, event_ts TIMESTAMP)`
3. Apply a 15-minute watermark on `event_ts` and use `dropDuplicates(["event_id", "event_ts"])` to remove duplicates
4. Write deduplicated events to a second Kafka topic `clean_events` using Append mode
5. Use `foreachBatch` to simultaneously write to both Kafka and a PostgreSQL table `clean_events_log`
6. Explain in comments: why must `event_ts` be included in `dropDuplicates` alongside `event_id`? What happens to state size without the watermark?

### Exercise 3: Stream-Stream Join for Order Matching

Build a pipeline that matches two event streams within a time window:

1. Stream 1: `orders` topic with schema `(order_id STRING, user_id STRING, amount DOUBLE, order_ts TIMESTAMP)`
2. Stream 2: `payments` topic with schema `(payment_id STRING, order_id STRING, payment_method STRING, pay_ts TIMESTAMP)`
3. Apply watermarks: 1 hour on `order_ts`, 2 hours on `pay_ts`
4. Inner join the streams: match orders with payments where `pay_ts` is between `order_ts` and `order_ts + 4 hours`
5. Write matched pairs to a `matched_orders` Kafka topic as JSON
6. For orders that have not been matched after 4 hours, use a separate `foreachBatch` query on the orders stream to write them to an `unmatched_orders` table in PostgreSQL
7. Explain in comments: why do both streams need watermarks for a stream-stream join? What determines the state retention period?

### Exercise 4: foreachBatch Multi-Sink Writer

Implement a `foreachBatch` function that writes each micro-batch to three destinations atomically:

1. Read streaming events from Kafka (same schema as Exercise 1)
2. Implement `process_batch(batch_df, batch_id)` that:
   - Skips empty batches (early return)
   - Writes raw events to Parquet partitioned by date
   - Aggregates per-user totals and upserts to a Delta Lake table using `DeltaTable.merge()` with `batch_id` in the condition to ensure idempotency
   - Writes a JSON summary `{batch_id, row_count, total_amount, timestamp}` to a `batch_log` PostgreSQL table
3. Configure `foreachBatch` with a 30-second trigger and a checkpoint location
4. Add error handling so that if the PostgreSQL write fails, the batch is logged to a local error file and the stream continues
5. Explain in comments why the `batch_id` parameter is critical for idempotent `foreachBatch` implementations

### Exercise 5: End-to-End Streaming Analytics System

Design and implement a complete streaming analytics system for an e-commerce platform:

1. **Ingestion**: Read from three Kafka topics: `clicks`, `cart_events`, and `purchases` — each with appropriate schemas and timestamps
2. **Watermarking**: Apply 5-minute watermarks to all three streams
3. **Funnel computation**: Using tumbling 15-minute windows, compute for each window:
   - `click_count`: total clicks
   - `cart_count`: total cart additions
   - `purchase_count`: total purchases
   - `conversion_rate`: `purchase_count / click_count` (handle division by zero)
4. **Static enrichment**: Join the purchases stream with a static `product_catalog` DataFrame (loaded from Parquet) to add `category` and `price` columns
5. **Output**: Write the funnel metrics to a Delta Lake table using Update mode; write enriched purchases to a separate Parquet sink
6. **Fault tolerance**: Use separate checkpoint locations for each query; explain in comments why checkpoint directories must never be shared between queries

---

## 10. Summary

### Key Takeaways

| Concept | Description |
|---------|-------------|
| **Unbounded table** | Stream = continuously growing DataFrame |
| **Watermark** | Controls late data acceptance and state cleanup |
| **Output modes** | Append (new only), Complete (all), Update (changed) |
| **Windows** | Tumbling (fixed), Sliding (overlapping) with time columns |
| **foreachBatch** | Custom logic per micro-batch (multi-sink, JDBC) |
| **Checkpointing** | Fault tolerance with exactly-once guarantees |
| **Kafka integration** | Native source/sink with offset management |

### Best Practices

1. **Always set watermarks** for windowed aggregations — prevents unbounded state
2. **Use checkpointing** in production — required for fault tolerance
3. **Prefer Update mode** over Complete for large aggregations
4. **Use foreachBatch** for writing to external systems (databases, APIs)
5. **Monitor state size** — large state slows processing and increases memory
6. **Set maxOffsetsPerTrigger** to control Kafka read rate

### Next Steps

- **L18**: CDC with Debezium — capture database changes as Kafka streams
- **L19**: Lakehouse Practical Patterns — land streaming data in Delta Lake
