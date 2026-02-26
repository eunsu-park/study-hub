# Spark Structured Streaming

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Spark Structured Streaming의 무한 테이블(Unbounded Table) 모델과 지속적으로 도착하는 데이터에 대해 엔진이 증분 연산을 수행하는 방식을 설명할 수 있습니다.
2. PySpark DataFrame API를 사용하여 스트리밍 쿼리를 구현하고, 집계 패턴에 따라 적절한 출력 모드(추가, 전체, 업데이트)를 선택할 수 있습니다.
3. 이벤트 시간 윈도잉(Event-Time Windowing)과 워터마크(Watermark)를 적용하여 늦게 도착하는 데이터를 처리하고 정확한 윈도우 집계를 계산할 수 있습니다.
4. Spark Structured Streaming을 Apache Kafka와 소스 및 싱크로 통합하고, 적절한 직렬화 형식을 사용할 수 있습니다.
5. mapGroupsWithState 또는 flatMapGroupsWithState를 사용하여 커스텀 세션 및 사용자 정의 집계 로직을 위한 상태 기반 스트림 처리(Stateful Stream Processing)를 설계할 수 있습니다.
6. 체크포인팅(Checkpointing) 및 내결함성(Fault-Tolerance) 메커니즘을 구성하여 종단간(End-to-End) 정확히 한 번(Exactly-Once) 처리 보장을 달성할 수 있습니다.

---

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

# 소켓 소스는 버퍼링이 없고 재생(replay)을 지원하지 않아 데모 용도에만 적합하다.
# 프로덕션에서는 Kafka나 파일 소스를 사용할 것 — 이들은 오프셋 추적을 통해
# 정확히 한 번(exactly-once) 의미론과 장애 내성 복구를 지원한다.
lines = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# explode()는 각 단어를 별도의 행으로 분리한다 — groupBy가 하나의 컬럼 값 내의
# 요소가 아닌 행 단위로 동작하기 때문에 이 변환이 필요하다.
words = lines.select(explode(split(lines.value, " ")).alias("word"))
word_counts = words.groupBy("word").count()

# "complete" 모드는 매 트리거마다 집계 결과 전체를 재전송한다.
# 단어 빈도 테이블이 메모리에 충분히 작기 때문에 여기서는 이 방식이 적합하다.
# 높은 카디널리티(high-cardinality) 집계에서는 변경된 행만 출력하는 "update" 모드가 낫다.
query = word_counts.writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

# awaitTermination()은 메인 스레드를 블록한다 — 없으면 드라이버가 즉시 종료되어
# 데이터를 처리하기도 전에 스트리밍 쿼리가 종료된다.
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

# Kafka 소스는 값이 불투명한 바이트 배열이므로 스키마를 사전에 선언해야 한다 —
# Spark는 샘플링으로 스키마를 추론할 수 없다. 여기서 스키마가 틀리면
# 조용히 null 컬럼이 생성되므로, 이 정의를 프로듀서의 직렬화 형식과
# 항상 동기화 상태로 유지해야 한다.
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

# Kafka는 key/value를 바이너리로 전달한다 — 먼저 문자열로 캐스팅한 후 JSON
# 봉투(envelope)를 파싱한다. 두 단계 select("data" 별칭 후 "data.*")는
# 중첩된 struct 컬럼을 최상위 컬럼으로 평탄화(flatten)하여 하위 처리를 단순화한다.
events = raw.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# 비상태 변환(stateless transformation)(filter, map, withColumn)은
# 배치 DataFrame과 동일하게 동작한다. 상태 추적이 필요 없고
# 가장 메모리 효율적인 출력 모드인 append 모드를 지원한다.
filtered = events.filter(col("amount") > 0)
transformed = filtered.withColumn(
    "event_time", to_timestamp(col("timestamp"))
).withColumn(
    "action_upper", upper(col("action"))
)

# processingTime="5 seconds"는 마이크로배치 간격을 설정한다. 짧은 간격은
# 지연(latency)을 낮추지만 스케줄링 오버헤드를 증가시킨다. 대부분의 경우
# 1~30초가 데이터 신선도(freshness)와 처리량(throughput) 간의 균형점이다.
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

# 텀블링 윈도우(tumbling window): 겹치지 않는 10분 단위 윈도우
tumbling = events \
    .withWatermark("event_time", "5 minutes") \
    .groupBy(
        # 텀블링 윈도우는 시간을 고정된 비중복 버킷으로 분할한다.
        # 각 이벤트는 정확히 하나의 윈도우에만 속하므로, 이벤트를 중복 윈도우에
        # 복사하는 슬라이딩 윈도우보다 간단하고 비용이 적다.
        window(col("event_time"), "10 minutes"),
        col("action")
    ) \
    .agg(
        count("*").alias("event_count"),
        spark_sum("amount").alias("total_amount"),
        avg("amount").alias("avg_amount"),
    )

# 슬라이딩 윈도우(sliding window): 10분 윈도우, 5분마다 이동.
# 슬라이드 간격(5분) < 윈도우 길이(10분)이므로 중복이 발생하여
# 각 이벤트가 2개의 윈도우에 나타난다. 텀블링 대비 상태 크기와 처리 비용이
# 두 배가 되지만, 더 부드러운 트렌드 감지가 가능하다.
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

# "update" 모드는 이번 마이크로배치에서 변경된 윈도우만 내보낸다.
# 이미 완료된 오래된 윈도우를 매 트리거마다 재전송하지 않으므로
# 윈도우 집계에서 "complete" 모드보다 훨씬 효율적이다.
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

# 스트림-스트림 조인에서는 두 스트림 모두 반드시 워터마크가 있어야 한다.
# 없으면 엔진이 잠재적 매칭을 기다리며 모든 이벤트를 영원히 버퍼링한다.
# 워터마크 값은 스트림마다 다를 수 있다 — 각 소스의 이벤트가 얼마나
# 늦게 도착하는지에 따라 개별적으로 설정할 것.
orders_wm = orders.withWatermark("order_time", "2 hours")
shipments_wm = shipments.withWatermark("ship_time", "3 hours")

# 시간 범위 조건(ship_time이 order_time으로부터 24시간 이내)이 핵심이다:
# 이 조건이 주문당 상태 보유 시간을 제한한다. 없으면 매칭되지 않은 모든 주문이
# 상태에 무기한 남는다. 윈도우를 좁히면 메모리 사용이 줄지만
# 늦게 도착하는 배송 이벤트를 놓칠 위험이 있다.
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
    # 빈 배치에서 조기 반환하면 JDBC 연결 오버헤드를 피하고
    # 나중에 읽기를 느리게 만드는 빈 Parquet 파일 생성을 방지한다.
    if batch_df.isEmpty():
        return

    # foreachBatch는 단일 스트리밍 쿼리에서 여러 싱크에 원자적으로 쓸 수 있는
    # 유일한 방법이다. 없으면 같은 소스를 소비하는 두 개의 개별 쿼리가 필요하여
    # Kafka 읽기 트래픽이 두 배가 된다.
    batch_df.write.mode("append").parquet(f"/data/output/batch_{batch_id}/")

    # JDBC 싱크는 Structured Streaming에서 네이티브로 지원되지 않는다 —
    # foreachBatch는 JDBC를 포함한 모든 배치 쓰기 형식을 지원하는
    # 표준 DataFrame을 제공함으로써 이 격차를 해소한다.
    batch_df.write \
        .format("jdbc") \
        .option("url", "jdbc:postgresql://localhost:5432/warehouse") \
        .option("dbtable", "events") \
        .option("user", "etl_user") \
        .option("password", "secret") \
        .mode("append") \
        .save()

    print(f"Batch {batch_id}: {batch_df.count()} rows processed")

# foreachBatch는 체크포인팅으로부터 정확히 한 번(exactly-once) 보장을 상속한다:
# 배치가 쓰기 도중 실패하면 재시작 시 체크포인트에서 재실행된다.
# 재실행 시 중복 쓰기를 방지하려면 배치 함수가 멱등적(idempotent)이어야 한다
# (예: MERGE 사용 또는 batch_id로 중복 제거).
query = events.writeStream \
    .foreachBatch(process_batch) \
    .option("checkpointLocation", "/checkpoints/foreach/") \
    .trigger(processingTime="30 seconds") \
    .start()
```

### 6.2 중복 제거(Deduplication)

```python
# dropDuplicates에 event_time을 포함해야 워터마크가 상태에서 오래된
# 항목을 만료시킬 수 있다. event_id만 사용하면 엔진이 모든 확인된 ID를
# 무한정 보유해야 하여 메모리가 무제한으로 증가한다.
deduplicated = events \
    .withWatermark("event_time", "10 minutes") \
    .dropDuplicates(["event_id", "event_time"])

# 10분 워터마크는 10분을 초과하여 늦게 도착하는 중복이 탐지되지 않음을 의미한다.
# 이 임계값은 프로듀서의 재시도 윈도우에 따라 설정할 것 —
# 재시도가 5분 이내에 발생한다면 10분 워터마크가 안전한 여유를 제공한다.
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
# 여러 브로커를 지정하면 장애 조치(failover)가 가능하다 — broker1이 다운되어도
# Spark는 broker2를 통해 쿼리 중단 없이 연결을 유지한다.
kafka_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "broker1:9092,broker2:9092") \
    .option("subscribe", "orders,shipments") \
    # "latest"는 최초 시작 시 과거 데이터를 건너뛴다 — 백필(backfill)이 필요하면
    # "earliest"를 사용할 것. 첫 실행 이후에는 체크포인트에서 오프셋이 재개된다.
    .option("startingOffsets", "latest") \
    # 마이크로배치당 읽는 레코드 수를 제한한다. 이 제한이 없으면
    # 대규모 백로그(예: 다운타임 이후)가 수백만 건의 레코드를 단일 배치로
    # 끌어들여 OOM을 유발할 수 있다.
    .option("maxOffsetsPerTrigger", 10000) \
    .option("kafka.group.id", "spark-streaming-group") \
    # failOnDataLoss=false는 Kafka 보존 정책이 체크포인트가 참조하는 오프셋을
    # 삭제했을 때 쿼리가 충돌하지 않도록 방지한다. 데이터 손실 시
    # 처리를 중단하고 조사해야 하는 파이프라인에서는 true로 설정할 것.
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

# kafka_key와 kafka_timestamp를 파싱된 필드와 함께 보존하면 하위 디버깅이
# 가능하다(잘못된 레코드를 Kafka 파티션/오프셋으로 추적) 및 이벤트 시간
# 처리에 활용된다(kafka_timestamp는 Kafka가 메시지를 수신한 시점을 반영).
parsed = kafka_stream.select(
    col("key").cast("string").alias("kafka_key"),
    from_json(col("value").cast("string"), schema).alias("data"),
    col("topic"),
    col("timestamp").alias("kafka_timestamp"),
).select("kafka_key", "data.*", "topic", "kafka_timestamp")
```

### 8.2 Kafka에 쓰기

```python
# Kafka 싱크는 정확히 두 개의 컬럼이 필요하다: "key"와 "value"(문자열 또는 바이트).
# order_id를 키로 사용하면 같은 주문의 모든 이벤트가 동일한 Kafka 파티션에
# 저장되어 주문별 이벤트 순서가 보장된다.
output = parsed.select(
    col("order_id").alias("key"),
    col("amount").cast("string").alias("value"),
)

# 각 스트리밍 쿼리는 자체 체크포인트 디렉토리를 가져야 한다 —
# 쿼리 간에 체크포인트를 공유하면 오프셋 손상과 데이터 손실이 발생한다.
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

## 연습 문제

### 연습 1: Kafka에서 Parquet으로의 스트리밍 파이프라인

Kafka에서 읽어 결과를 Parquet에 쓰는 완전한 Structured Streaming 파이프라인을 구축하세요:

1. 스키마 `(event_id STRING, user_id STRING, action STRING, amount DOUBLE, event_time TIMESTAMP)`의 Kafka 토픽 `user_events`에서 읽으세요
2. 늦게 도착하는 데이터를 처리하기 위해 `event_time`에 3분 워터마크를 적용하세요
3. `(action, window)` 기준으로 그룹화한 텀블링(tumbling) 10분 윈도우 집계를 계산하세요:
   - 이벤트 건수
   - amount 합계
   - 고유 사용자 수 (`approx_count_distinct` 힌트)
4. Update 출력 모드를 사용하여 결과를 `/data/output/event_summary/`의 Parquet에 쓰세요
5. 체크포인트(checkpoint) 위치를 구성하고 `maxOffsetsPerTrigger=5000`을 설정하세요
6. `query.lastProgress`로 쿼리를 모니터링하고 각 마이크로배치 후 `inputRowsPerSecond`를 출력하세요

### 연습 2: 워터마크를 활용한 상태 저장 중복 제거

대용량 이벤트 스트림을 위한 정확히 한 번(exactly-once) 중복 제거 파이프라인을 구현하세요:

1. 프로듀서(producer) 재시도로 동일한 `event_id`가 여러 번 나타날 수 있는 Kafka 토픽 `raw_events`에서 읽으세요
2. 스키마 `(event_id STRING, user_id STRING, amount DOUBLE, event_ts TIMESTAMP)`로 JSON 페이로드를 파싱하세요
3. `event_ts`에 15분 워터마크를 적용하고 `dropDuplicates(["event_id", "event_ts"])`로 중복을 제거하세요
4. Append 모드를 사용하여 중복 제거된 이벤트를 두 번째 Kafka 토픽 `clean_events`에 쓰세요
5. `foreachBatch`를 사용하여 Kafka와 PostgreSQL 테이블 `clean_events_log` 모두에 동시에 쓰세요
6. 주석으로 설명하세요: `dropDuplicates`에서 `event_id`와 함께 `event_ts`를 포함해야 하는 이유는 무엇인가요? 워터마크 없이는 상태(state) 크기에 어떤 일이 발생하나요?

### 연습 3: 주문 매칭을 위한 스트림-스트림 조인

시간 윈도우 내에서 두 이벤트 스트림을 매칭하는 파이프라인을 구축하세요:

1. 스트림 1: 스키마 `(order_id STRING, user_id STRING, amount DOUBLE, order_ts TIMESTAMP)`의 `orders` 토픽
2. 스트림 2: 스키마 `(payment_id STRING, order_id STRING, payment_method STRING, pay_ts TIMESTAMP)`의 `payments` 토픽
3. 워터마크 적용: `order_ts`에 1시간, `pay_ts`에 2시간
4. 스트림을 내부 조인(inner join): `pay_ts`가 `order_ts`와 `order_ts + 4시간` 사이인 주문과 결제를 매칭하세요
5. 매칭된 쌍을 JSON 형태로 `matched_orders` Kafka 토픽에 쓰세요
6. 4시간 후에도 매칭되지 않은 주문은, 별도의 `foreachBatch` 쿼리를 사용하여 PostgreSQL의 `unmatched_orders` 테이블에 쓰세요
7. 주석으로 설명하세요: 스트림-스트림 조인에서 두 스트림 모두 워터마크가 필요한 이유는 무엇인가요? 상태 보존 기간은 무엇으로 결정되나요?

### 연습 4: foreachBatch 다중 싱크 라이터

각 마이크로배치를 세 가지 대상에 원자적으로 쓰는 `foreachBatch` 함수를 구현하세요:

1. Kafka에서 스트리밍 이벤트를 읽으세요 (연습 1과 동일한 스키마)
2. `process_batch(batch_df, batch_id)` 함수를 구현하세요:
   - 빈 배치는 조기 반환하세요
   - 날짜로 파티셔닝된 Parquet에 원시 이벤트를 쓰세요
   - 사용자별 합계를 집계하고 `batch_id`를 조건에 포함한 `DeltaTable.merge()`를 사용하여 Delta Lake 테이블에 업서트(upsert)하세요 (멱등성 보장)
   - `{batch_id, row_count, total_amount, timestamp}` JSON 요약을 PostgreSQL의 `batch_log` 테이블에 쓰세요
3. 30초 트리거와 체크포인트 위치를 설정하여 `foreachBatch`를 구성하세요
4. PostgreSQL 쓰기가 실패하면 배치를 로컬 오류 파일에 기록하고 스트림을 계속하는 오류 처리를 추가하세요
5. 멱등적(idempotent) `foreachBatch` 구현에서 `batch_id` 파라미터가 중요한 이유를 주석으로 설명하세요

### 연습 5: 엔드 투 엔드 스트리밍 분석 시스템

이커머스 플랫폼을 위한 완전한 스트리밍 분석 시스템을 설계하고 구현하세요:

1. **수집**: 세 개의 Kafka 토픽에서 읽으세요: `clicks`, `cart_events`, `purchases` — 각각 적절한 스키마와 타임스탬프 포함
2. **워터마킹**: 세 스트림 모두에 5분 워터마크를 적용하세요
3. **퍼널 계산**: 텀블링 15분 윈도우를 사용하여 각 윈도우에 대해 계산하세요:
   - `click_count`: 총 클릭 수
   - `cart_count`: 총 장바구니 추가 수
   - `purchase_count`: 총 구매 수
   - `conversion_rate`: `purchase_count / click_count` (0으로 나누기 처리)
4. **정적 강화**: 구매 스트림을 Parquet에서 로드한 정적 `product_catalog` DataFrame과 조인하여 `category`와 `price` 컬럼을 추가하세요
5. **출력**: 퍼널 지표를 Update 모드로 Delta Lake 테이블에 쓰고, 강화된 구매 데이터는 별도의 Parquet 싱크(sink)에 쓰세요
6. **내결함성(Fault Tolerance)**: 각 쿼리에 별도의 체크포인트 위치를 사용하세요; 쿼리 간에 체크포인트 디렉토리를 절대 공유해서는 안 되는 이유를 주석으로 설명하세요

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
