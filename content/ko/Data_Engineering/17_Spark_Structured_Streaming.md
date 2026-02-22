# Spark Structured Streaming

## 개요

Spark Structured Streaming은 DataFrame API를 확장하여 무한한 데이터 스트림(unbounded data stream)을 처리합니다. 스트림을 지속적으로 증가하는 테이블로 간주하여, 스트리밍 데이터에 배치(batch)와 유사한 쿼리를 수행할 수 있습니다. 이 레슨에서는 프로그래밍 모델, 소스/싱크(source/sink), 윈도우 집계(windowed aggregation), 상태 기반 처리(stateful processing), 그리고 Kafka 통합을 다룹니다.

---

## 1. Structured Streaming 모델

### 1.1 무한 테이블(Unbounded Table) 개념

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

### 1.2 기본 예제

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split

spark = SparkSession.builder \
    .appName("StructuredStreaming") \
    .getOrCreate()

# Read from a socket (for demo; use Kafka in production)
lines = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# Split lines into words and count
words = lines.select(explode(split(lines.value, " ")).alias("word"))
word_counts = words.groupBy("word").count()

# Write results to console
query = word_counts.writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

query.awaitTermination()
```

---

## 2. 소스와 싱크(Sources and Sinks)

### 2.1 내장 소스(Built-in Sources)

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

### 2.2 내장 싱크(Built-in Sinks)

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

## 3. 스트림 연산(Operations on Streams)

### 3.1 비상태 연산(Stateless Operations)

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp, upper
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

spark = SparkSession.builder.appName("StreamOps").getOrCreate()

# Define schema for incoming JSON
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

# Parse JSON value
events = raw.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# Stateless operations (same as batch DataFrame)
filtered = events.filter(col("amount") > 0)
transformed = filtered.withColumn(
    "event_time", to_timestamp(col("timestamp"))
).withColumn(
    "action_upper", upper(col("action"))
)

# Write to console
query = transformed.writeStream \
    .outputMode("append") \
    .format("console") \
    .trigger(processingTime="5 seconds") \
    .start()
```

### 3.2 출력 모드(Output Modes)

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

## 4. 윈도우 집계(Windowed Aggregations)

### 4.1 텀블링(Tumbling)과 슬라이딩 윈도우(Sliding Windows)

```python
from pyspark.sql.functions import window, col, count, sum as spark_sum, avg

# Tumbling window: non-overlapping 10-minute windows
tumbling = events \
    .withWatermark("event_time", "5 minutes") \
    .groupBy(
        window(col("event_time"), "10 minutes"),  # Tumbling: windowDuration only
        col("action")
    ) \
    .agg(
        count("*").alias("event_count"),
        spark_sum("amount").alias("total_amount"),
        avg("amount").alias("avg_amount"),
    )

# Sliding window: 10-minute window, sliding every 5 minutes
sliding = events \
    .withWatermark("event_time", "5 minutes") \
    .groupBy(
        window(col("event_time"), "10 minutes", "5 minutes"),  # slideDuration added
        col("user_id")
    ) \
    .agg(
        count("*").alias("event_count"),
        spark_sum("amount").alias("total_amount"),
    )

# Write windowed aggregation
query = tumbling.writeStream \
    .outputMode("update") \
    .format("console") \
    .option("truncate", "false") \
    .trigger(processingTime="10 seconds") \
    .start()
```

### 4.2 지연 데이터를 위한 워터마크(Watermarking for Late Data)

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

## 5. 스트림-스트림 조인(Stream-Stream Joins)

### 5.1 시간 제약이 있는 내부 조인(Inner Join with Time Constraint)

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

# Join: match orders with shipments within 24 hours
# Both streams need watermarks for the join to work
orders_wm = orders.withWatermark("order_time", "2 hours")
shipments_wm = shipments.withWatermark("ship_time", "3 hours")

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

### 5.2 스트림-정적 조인(Stream-Static Join)

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

## 6. 상태 기반 처리(Stateful Processing)

### 6.1 사용자 정의 로직을 위한 foreachBatch

```python
def process_batch(batch_df, batch_id):
    """Process each micro-batch with custom logic."""
    if batch_df.isEmpty():
        return

    # Write to multiple sinks
    batch_df.write.mode("append").parquet(f"/data/output/batch_{batch_id}/")

    # Write to a database
    batch_df.write \
        .format("jdbc") \
        .option("url", "jdbc:postgresql://localhost:5432/warehouse") \
        .option("dbtable", "events") \
        .option("user", "etl_user") \
        .option("password", "secret") \
        .mode("append") \
        .save()

    print(f"Batch {batch_id}: {batch_df.count()} rows processed")

# Use foreachBatch
query = events.writeStream \
    .foreachBatch(process_batch) \
    .option("checkpointLocation", "/checkpoints/foreach/") \
    .trigger(processingTime="30 seconds") \
    .start()
```

### 6.2 중복 제거(Deduplication)

```python
# Deduplicate events by event_id within a watermark window
deduplicated = events \
    .withWatermark("event_time", "10 minutes") \
    .dropDuplicates(["event_id", "event_time"])

# Without watermark: keeps ALL event_ids in state (memory grows unbounded)
# With watermark: only keeps event_ids within the watermark window
```

---

## 7. 모니터링과 체크포인팅(Monitoring and Checkpointing)

### 7.1 쿼리 진행 상황(Query Progress)

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

### 7.2 체크포인트와 복구(Checkpoint and Recovery)

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

## 8. Kafka 통합 심화(Kafka Integration Deep Dive)

### 8.1 Kafka에서 읽기

```python
# Read from Kafka with full configuration
kafka_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "broker1:9092,broker2:9092") \
    .option("subscribe", "orders,shipments") \
    .option("startingOffsets", "latest") \
    .option("maxOffsetsPerTrigger", 10000) \
    .option("kafka.group.id", "spark-streaming-group") \
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

parsed = kafka_stream.select(
    col("key").cast("string").alias("kafka_key"),
    from_json(col("value").cast("string"), schema).alias("data"),
    col("topic"),
    col("timestamp").alias("kafka_timestamp"),
).select("kafka_key", "data.*", "topic", "kafka_timestamp")
```

### 8.2 Kafka에 쓰기

```python
# Write processed data back to Kafka
output = parsed.select(
    col("order_id").alias("key"),
    col("amount").cast("string").alias("value"),
)

query = output.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "broker:9092") \
    .option("topic", "processed_orders") \
    .option("checkpointLocation", "/checkpoints/kafka_output/") \
    .trigger(processingTime="10 seconds") \
    .start()
```

---

## 9. 연습 문제

### 연습 1: 실시간 대시보드 지표

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

### 연습 2: 스트림-스트림 조인

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

## 10. 요약

### 핵심 정리

| 개념 | 설명 |
|------|------|
| **무한 테이블(Unbounded table)** | 스트림 = 지속적으로 증가하는 DataFrame |
| **워터마크(Watermark)** | 지연 데이터 허용 범위 및 상태 정리 제어 |
| **출력 모드(Output modes)** | Append(신규만), Complete(전체), Update(변경된 것만) |
| **윈도우(Windows)** | Tumbling(고정), Sliding(중첩), 시간 컬럼 기반 |
| **foreachBatch** | 마이크로배치별 사용자 정의 로직(멀티 싱크, JDBC) |
| **체크포인팅(Checkpointing)** | 정확히 한 번(exactly-once) 보장하는 장애 내성 |
| **Kafka 통합** | 오프셋 관리를 포함한 네이티브 소스/싱크 |

### 모범 사례(Best Practices)

1. **윈도우 집계에는 항상 워터마크를 설정** — 무한한 상태(unbounded state) 방지
2. **프로덕션에서는 체크포인팅 사용** — 장애 허용(fault tolerance)에 필수
3. **대규모 집계에는 Complete보다 Update 모드 선호**
4. **외부 시스템(데이터베이스, API) 쓰기에는 foreachBatch 사용**
5. **상태 크기 모니터링** — 큰 상태는 처리 속도를 낮추고 메모리를 증가시킴
6. **maxOffsetsPerTrigger 설정**으로 Kafka 읽기 속도 제어

### 다음 단계

- **L18**: Debezium을 활용한 CDC(Change Data Capture) — 데이터베이스 변경 사항을 Kafka 스트림으로 캡처
- **L19**: Lakehouse 실전 패턴 — 스트리밍 데이터를 Delta Lake에 적재
