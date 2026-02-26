# Kafka Streams와 ksqlDB

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 유한(배치) 데이터와 무한(스트림) 데이터 처리의 근본적인 차이를 설명하고, 이벤트 시간(Event Time), 처리 시간(Processing Time), 워터마크(Watermark)를 구분할 수 있습니다.
2. Kafka Streams 토폴로지(Topology) 모델에서 소스 프로세서, 상태 비저장 변환(filter, map, flatMap), 상태 저장소(State Store)를 활용한 상태 저장 연산을 설명할 수 있습니다.
3. Kafka Streams 또는 파이썬 Faust를 사용하여 텀블링(Tumbling), 호핑(Hopping), 세션(Session) 윈도우 집계와 스트림-테이블 조인을 구현할 수 있습니다.
4. ksqlDB 쿼리를 작성하여 스트림과 테이블을 생성하고, 실시간 필터링 및 집계를 수행하며, 대화형 접근을 위한 결과를 구체화할 수 있습니다.
5. 이벤트 소싱(Event Sourcing), CQRS, 구체화 뷰(Materialized Views) 등의 스트림 처리 패턴을 실제 데이터 엔지니어링 문제에 적용할 수 있습니다.
6. 적절한 오류 처리, 데드 레터 큐(Dead Letter Queue), 정확히 한 번(Exactly-Once) 시맨틱을 갖춘 스트림 처리 파이프라인을 설계할 수 있습니다.

---

## 개요

Kafka Streams는 Apache Kafka 위에서 실시간 스트림 처리(stream processing) 애플리케이션을 구축하기 위한 클라이언트 라이브러리입니다. ksqlDB는 SQL 인터페이스를 통해 스트림 처리를 확장합니다. 이 레슨에서는 스트림 처리 개념, Faust(파이썬 Kafka Streams), 윈도우 집계(windowed aggregation), 조인(join), 그리고 대화형 쿼리를 위한 ksqlDB를 다룹니다.

---

## 1. 스트림 처리 개념

### 1.1 유한 데이터 vs 무한 데이터(Bounded vs Unbounded Data)

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

### 1.2 스트림 처리 아키텍처(Stream Processing Architectures)

| 패턴 | 설명 | 사용 사례 |
|------|------|----------|
| **이벤트 소싱(Event Sourcing)** | 모든 이벤트를 저장하고 상태를 도출 | 금융 원장, 감사 이력 |
| **CQRS** | 읽기/쓰기 모델 분리 | 읽기 부하가 높은 워크로드 |
| **구체화 뷰(Materialized Views)** | 스트림으로부터 사전 계산된 쿼리 결과 | 실시간 대시보드 |
| **CDC → 스트림** | 데이터베이스 변경을 이벤트로 변환 | 데이터 복제, 동기화 |

---

## 2. Kafka Streams 아키텍처

### 2.1 핵심 개념

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

### 2.2 정확히 한 번 처리(Exactly-Once Semantics)

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

## 3. Faust: 파이썬 Kafka Streams

### 3.1 기본 Faust 애플리케이션

```python
"""
Faust is a Python stream processing library inspired by Kafka Streams.

# pip install faust-streaming  (maintained fork of original faust)
"""

import faust

# Faust 앱 생성 (Kafka에 연결)
# RocksDB는 LSM-트리 성능으로 지속적이고 충돌 복구 가능한 키-값 스토리지를 제공하기 때문에
# 상태 저장소로 선택됩니다 — 고처리량 스트림 집계에 이상적입니다.
# 대안(인메모리 스토어)은 빠르지만 재시작 시 상태를 잃습니다
app = faust.App(
    'my_stream_app',
    broker='kafka://localhost:9092',
    store='rocksdb://',  # 상태 저장소 백엔드
    topic_replication_factor=1,
)

# Faust Records는 타입화된 역직렬화를 제공합니다 — 들어오는 JSON 메시지가
# 타입 검증과 함께 Order 객체로 자동 파싱되어
# 파이프라인 초기에 잘못된 메시지를 포착합니다
class Order(faust.Record):
    order_id: str
    user_id: str
    amount: float
    timestamp: float

# Define input topic
orders_topic = app.topic('orders', value_type=Order)

# @app.agent은 비동기 제너레이터로 실행되는 스트림 프로세서를 생성합니다.
# 각 에이전트 인스턴스는 하나 이상의 Kafka 파티션에서 소비하여
# 더 많은 워커 프로세스를 추가함으로써 수평 확장이 가능합니다
@app.agent(orders_topic)
async def process_orders(orders):
    async for order in orders:
        if order.amount > 1000:
            print(f"High-value order: {order.order_id} = ${order.amount}")

# Run: faust -A myapp worker -l info
```

### 3.2 상태 저장 처리(Stateful Processing): 테이블과 집계

```python
import faust
from datetime import timedelta

app = faust.App('aggregation_demo', broker='kafka://localhost:9092')

class PageView(faust.Record):
    user_id: str
    page: str
    timestamp: float

page_views_topic = app.topic('page_views', value_type=PageView)

# Table = Kafka Streams 용어의 KTable: Kafka 변경 로그 토픽으로 백업된 지속적 키-값 스토어.
# 이 워커가 충돌하면 새 워커가 변경 로그를 재생하여 테이블을 재빌드합니다 — 자동 장애 복구.
# partitions=8은 코-파티셔닝을 위해 소스 토픽의 파티션 수와 일치해야 합니다
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

# Faust는 임시 디버깅과 경량 대시보드를 위해 HTTP를 통해 테이블을 노출합니다.
# 프로덕션에서는 집계를 외부 스토어(Redis, Postgres)로 밀어내는 것을 선호하세요
@app.page('/counts/')
async def get_counts(self, request):
    return self.json({k: v for k, v in page_view_counts.items()})

# 호핑 윈도우: 텀블링 윈도우보다 부드러운 트렌드 곡선을 제공하는 겹치는 시간 윈도우.
# size=1h, step=10min은 각 이벤트가 6개의 겹치는 윈도우에 기여하여 "이동 평균" 효과를 줍니다.
# expires=24h는 무제한 스토리지 성장을 방지하기 위해 RocksDB에서 오래된 윈도우를 자동 삭제합니다
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

### 3.3 스트림 연산(Stream Operations)

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

# Filter — 상태 비저장 연산: 각 이벤트는 독립적으로 평가됩니다.
# 필터링된 이벤트는 별도 토픽으로 전달되어 다운스트림 소비자(예: 알림)가
# 관련 이벤트만 받아 처리 부하와 컨슈머 랙을 줄입니다
@app.agent(events_topic)
async def filter_events(events):
    async for event in events.filter(lambda e: e.value > 100):
        await filtered_topic.send(value=event)

# Group by — event_type으로 스트림을 재파티셔닝하여 같은 타입의 이벤트를
# 같은 Kafka 파티션으로 보냅니다. 이는 상태 저장 집계(count, sum) 전에 필수로,
# 주어진 키의 모든 이벤트가 동일한 워커 인스턴스에서 처리되도록 보장합니다
@app.agent(events_topic)
async def group_by_type(events):
    async for event in events.group_by(Event.event_type):
        print(f"Type: {event.event_type}, Value: {event.value}")

# Map / Transform — 강화는 인플라이트(in-flight)에서 파생 필드를 추가합니다.
# 벽시계 시간 대신 message.timestamp(Kafka 수신 시간)를 사용하면
# 이벤트 재생 또는 재처리 중에도 일관된 결과를 보장합니다
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

## 4. 윈도우 집계(Windowed Aggregations)

### 4.1 윈도우 유형(Window Types)

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

# 텀블링 윈도우: 비겹치는 5분 버킷.
# 각 거래는 정확히 하나의 윈도우에 속하여 의미론이 단순하고
# 출력을 차트로 그리기 쉽습니다 (5분 간격마다 한 데이터 포인트).
# expires=1h는 RocksDB에서 오래된 윈도우를 자동 정리합니다 — 없으면 상태가
# 무한히 증가하여 결국 디스크 공간이 소진됩니다
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
        # .current()는 과거 윈도우가 아닌 현재 활성 윈도우의 값만 반환합니다
        # — 실시간 알림에 중요합니다
        current = tumbling_spending[txn.user_id].current()
        print(f"User {txn.user_id} 5-min spending: ${current:.2f}")

# 호핑 윈도우: 텀블링보다 부드러운 집계를 제공하는 겹치는 윈도우.
# size=10min, step=5min은 윈도우가 50% 겹쳐 경계 근처의 거래가
# 두 윈도우에 기여합니다 — 정확한 윈도우 정렬과 관계없이 지출 급증을
# 포착하고 싶은 사기 탐지에 유용합니다
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

## 5. 스트림 조인(Stream Joins)

### 5.1 조인 패턴(Join Patterns)

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

# 사용자 프로필을 위한 KTable: user_id별 최신 프로필만 저장합니다.
# 이는 천천히 변하는 참조 데이터에 올바른 추상화입니다 —
# KStream을 사용하면 모든 과거 프로필을 누적하여 메모리를 낭비하고
# 조회 시 오래된 데이터를 반환합니다
user_profiles = app.Table('user_profiles', default=None)

@app.agent(profiles_topic)
async def update_profiles(profiles):
    async for profile in profiles:
        # 업서트 의미론: 각 새 프로필 메시지가 이전 것을 덮어씁니다.
        # 이 테이블을 백업하는 변경 로그 토픽이 장애 허용을 제공합니다
        user_profiles[profile.user_id] = profile

# KStream-KTable 조인: 가장 일반적인 스트림 강화 패턴.
# 주문 스트림(무한, 고속도)이 프로필 테이블(최신 스냅샷)로 강화됩니다.
# 이는 윈도우가 필요하고 잠재적으로 매칭을 놓칠 수 있는
# KStream-KStream 조인보다 훨씬 더 효율적입니다
@app.agent(orders_topic)
async def enrich_orders(orders):
    async for order in orders:
        profile = user_profiles.get(order.user_id)
        if profile:
            print(f"Order {order.order_id}: {profile.name} ({profile.tier}) - ${order.amount}")
        else:
            # 누락된 프로필의 우아한 처리: 충돌 대신 로그합니다.
            # 프로덕션에서는 이것들을 조사를 위해 데드 레터 토픽으로 라우팅하세요
            print(f"Order {order.order_id}: Unknown user {order.user_id} - ${order.amount}")
```

---

## 6. ksqlDB 기초

### 6.1 아키텍처와 개념

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

### 6.2 ksqlDB SQL 예제

```sql
-- Kafka 토픽에서 STREAM을 생성합니다.
-- STREAM = 이벤트의 무한 시퀀스 (삽입 의미론, KStream과 유사).
-- TIMESTAMP = 'order_time'은 윈도우가 Kafka의 수신 타임스탬프가 아닌
-- 데이터에 내장된 이벤트 시간을 사용하도록 ksqlDB에 지시합니다 —
-- 이벤트가 순서 없이 도착할 때 올바른 윈도우 집계에 중요합니다
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

-- TABLE = 키별 최신값의 구체화된 뷰 (KTable과 유사).
-- PRIMARY KEY는 업서트 의미론을 위해 어떤 필드를 사용할지 ksqlDB에 알립니다 —
-- 동일한 user_id를 가진 새 메시지가 이전 값을 덮어씁니다
CREATE TABLE user_profiles (
    user_id VARCHAR PRIMARY KEY,
    name VARCHAR,
    email VARCHAR,
    tier VARCHAR
) WITH (
    KAFKA_TOPIC = 'user_profiles',
    VALUE_FORMAT = 'JSON'
);

-- 지속 쿼리: 연속 실행하며 결과를 새 Kafka 토픽에 씁니다.
-- 이것은 일회성 쿼리가 아닙니다 — 실시간으로 모든 새 이벤트를 처리합니다.
-- 출력 토픽은 알림 시스템이나 다른 애플리케이션에서 소비할 수 있습니다
CREATE STREAM high_value_orders AS
SELECT *
FROM orders_stream
WHERE amount > 1000
EMIT CHANGES;

-- TABLE로 구체화된 윈도우 집계.
-- TUMBLING (SIZE 1 HOUR)은 비겹치는 시간별 버킷을 생성합니다.
-- 결과는 실시간 대시보드를 위해 풀 쿼리로 조회 가능합니다
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

-- 스트림-테이블 조인: 모든 주문 이벤트를 사용자의 현재 프로필로 강화합니다.
-- LEFT JOIN은 사용자 프로필이 없을 때도 주문이 출력되도록 보장합니다
-- (예: 프로필 토픽에 아직 없는 신규 사용자), 데이터 손실을 방지합니다
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

-- 풀 쿼리: 구체화된 테이블에 대한 동기적 시점 조회.
-- 현재 값이 필요한 REST API와 대시보드에 적합합니다
SELECT * FROM hourly_user_orders WHERE user_id = 'user_123';

-- 푸시 쿼리: 결과를 Server-Sent Events로 스트리밍하는 장기 실행 쿼리.
-- 풀 쿼리와 달리 푸시 쿼리는 연결을 열어두고 실시간으로 모든
-- 매칭 이벤트를 출력합니다 — 라이브 모니터링 UI에 이상적입니다
SELECT * FROM enriched_orders WHERE amount > 500 EMIT CHANGES;
```

### 6.3 ksqlDB 커넥터(Connectors)

```sql
-- 소스 커넥터: PostgreSQL에서 Kafka 토픽으로 데이터베이스 변경(CDC)을 스트리밍합니다.
-- Debezium은 PostgreSQL의 WAL(Write-Ahead Log)을 통해 행 수준 변경을 캡처합니다.
-- 소스 데이터베이스에 비침습적입니다 (폴링 없음, 트리거 없음).
-- 각 캡처된 변경은 ksqlDB가 실시간으로 처리할 수 있는 Kafka 이벤트가 됩니다
CREATE SOURCE CONNECTOR pg_source WITH (
    'connector.class' = 'io.debezium.connector.postgresql.PostgresConnector',
    'database.hostname' = 'postgres',
    'database.port' = '5432',
    'database.user' = 'replicator',
    'database.password' = 'secret',
    'database.dbname' = 'mydb',
    -- table.include.list는 특정 테이블로 캡처를 제한하여
    -- 전체 데이터베이스 캡처 대비 Kafka 스토리지와 네트워크 오버헤드를 줄입니다
    'table.include.list' = 'public.orders,public.users',
    'topic.prefix' = 'pg'
);

-- 싱크 커넥터: 처리된 스트림 데이터를 전체 텍스트 검색과 시각화를 위해
-- Elasticsearch로 내보냅니다. 이는 Kafka 토픽과 거의 실시간으로 동기화된
-- 구체화된 뷰를 Elasticsearch에 생성합니다.
-- key.ignore=true는 Elasticsearch가 문서 ID를 자동 생성하게 하며
-- 추가 전용 이벤트 데이터에 적합합니다
CREATE SINK CONNECTOR es_sink WITH (
    'connector.class' = 'io.confluent.connect.elasticsearch.ElasticsearchSinkConnector',
    'connection.url' = 'http://elasticsearch:9200',
    'topics' = 'enriched_orders',
    'type.name' = '_doc',
    'key.ignore' = 'true'
);
```

---

## 7. 스트림 애플리케이션 모니터링

### 7.1 주요 지표(Key Metrics)

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

## 8. 연습 문제

### 연습 1: 실시간 사기 탐지(Real-Time Fraud Detection)

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

### 연습 2: 클릭스트림 세션화(Clickstream Sessionization) — ksqlDB

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

## 연습 문제

### 연습 1: 실시간 거래 집계기

여러 윈도우 유형으로 거래 데이터를 동시에 집계하는 Faust 애플리케이션을 구축하세요:

1. `user_id`, `merchant`, `amount`, `category`, `timestamp` 필드를 가진 `Transaction` 레코드를 정의하세요
2. 동일한 거래 스트림에 세 개의 별도 테이블을 생성하세요:
   - 사용자별 총 지출을 추적하는 **텀블링(tumbling)** 5분 테이블
   - 가맹점별 거래 건수를 추적하는 **호핑(hopping)** 10분(step=2분) 테이블
   - 사용자 세션당 거래 건수를 계산하는 **세션(session)** 테이블(30초 갭)
3. 텀블링 윈도우가 업데이트될 때마다, 사용자의 5분 누적 금액이 $500을 초과하면 알림을 출력하세요
4. 현재 텀블링 집계 합계를 JSON으로 반환하는 Faust HTTP 엔드포인트를 `/spending/`에 노출하세요
5. 각 테이블에 `expires`가 설정되어야 하는 이유와, 없을 경우 RocksDB 상태(state)에 어떤 일이 발생하는지 주석으로 설명하세요

### 연습 2: KStream-KTable 데이터 강화 파이프라인

Faust를 사용하여 주문 이벤트를 사용자 프로필 테이블과 조인하는 스트림 강화(enrichment) 파이프라인을 구현하세요:

1. `Order`(order_id, user_id, product_id, amount)와 `UserProfile`(user_id, name, tier, credit_limit) 레코드를 정의하세요
2. 새 프로필 이벤트가 도착할 때마다 최신 상태를 유지하는 `user_profiles` KTable을 관리하세요
3. 각 수신 주문에 대해:
   - KTable에서 사용자의 이름과 등급(tier)으로 주문을 강화(enrich)하세요
   - 사용자 프로필이 없으면 `missing_profile_orders` 토픽(데드 레터 큐)으로 라우팅하세요
   - 주문 금액이 사용자의 `credit_limit`을 초과하면 `credit_alerts` 토픽으로 라우팅하세요
4. 사용자 등급별 신용 한도 초과 알림 수를 집계하는 텀블링 1시간 테이블을 유지하세요
5. 프로필 조회에 KTable이 KStream보다 올바른 추상화인 이유를 주석으로 설명하세요

### 연습 3: ksqlDB 사기 탐지 파이프라인

의심스러운 거래 패턴을 감지하는 완전한 ksqlDB 파이프라인을 작성하세요:

```sql
-- 원시 거래 스트림 시작점: (txn_id, user_id, amount, merchant, country, txn_time)
-- 다음 파이프라인을 구축하세요:

-- 1단계: 이벤트 시간(Event Time) 의미론을 사용하는 소스 스트림 생성
-- 2단계: 사용자별 롤링(rolling) 1시간 지출 합계의 TABLE 생성
-- 3단계: 다음 조건의 거래 STREAM 생성:
--         (a) 단일 금액 > $2000, 또는
--         (b) 사용자의 롤링 1시간 합계가 $5000 초과
-- 4단계: 경보 스트림을 사용자 프로필 테이블과 조인하여 이메일 추가
-- 5단계: 경보 스트림을 지속적으로 모니터링하는 푸시 쿼리(push query) 작성
-- 6단계: 특정 user_id의 현재 롤링 합계를 확인하는 풀 쿼리(pull query) 작성
```

각 단계에서 해당 ksqlDB 구조체(STREAM vs TABLE, EMIT CHANGES vs 시점 조회)를 선택한 이유를 주석으로 설명하세요.

### 연습 4: 데드 레터 큐(Dead Letter Queue)와 오류 처리

연습 2의 Faust 주문 강화 파이프라인을 확장하여 프로덕션 수준의 오류 처리를 추가하세요:

1. 강화 로직을 try/except 블록으로 감싸고, 예외 발생 시 원시 이벤트를 `dead_letter_orders` Faust 토픽에 `error_reason` 필드와 함께 직렬화하세요
2. `dead_letter_orders`를 소비하여 오류 유형별 건수를 테이블로 유지하는 두 번째 Faust 에이전트를 생성하세요
3. 오류 건수를 JSON으로 반환하는 Faust HTTP 엔드포인트를 `/dlq/stats/`에 추가하세요
4. 지수 백오프(exponential backoff) 재시도 로직을 구현하세요: KTable에 사용자 프로필이 없으면 100ms, 200ms, 400ms 간격으로 최대 3번 재시도 후 데드 레터 큐(DLQ)로 라우팅하세요
5. 정확히 한 번(exactly-once) 시맨틱이 재시도 로직과 어떻게 상호작용하는지, 데드 레터 토픽에 멱등적(idempotent) 쓰기가 중요한 이유를 주석으로 설명하세요

### 연습 5: 다중 소스 조인과 세션 분석

클릭스트림(clickstream)과 구매 이벤트를 연관 짓는 고급 파이프라인을 설계하세요:

1. 두 개의 Faust 토픽을 생성하세요: `page_views`(user_id, page, session_id, timestamp)와 `purchases`(user_id, order_id, amount, timestamp)
2. 세션 기반 집계 구현: `page_views`에 30분 세션 윈도우를 적용하여 세션당 페이지 조회 수를 계산하세요
3. 구매 이벤트와 세션 데이터를 조인: 각 구매에 대해 구매 30분 이전에 종료된 세션에서 사용자가 몇 페이지를 조회했는지 조회하세요
4. ksqlDB 동등 구현: 구매 이벤트를 `pages_in_session` 수로 강화하는 STREAM-TABLE 조인을 작성하세요
5. `conversion_events` 토픽(`{user_id, order_id, amount, pages_in_session}`)을 생성하고, ksqlDB TABLE을 사용하여 금액 구간(0-50, 50-200, 200+)별로 전환 사용자의 평균 `pages_in_session`을 계산하세요

---

## 9. 정리

### 핵심 내용

| 개념 | 설명 |
|------|------|
| **KStream** | 무한 이벤트 스트림 (삽입 의미론) |
| **KTable** | 변경 로그 스트림 (업서트, 키별 최신값) |
| **Faust** | 스트림 처리를 위한 파이썬 Kafka Streams 라이브러리 |
| **윈도잉(Windowing)** | 시간 제한 집계를 위한 텀블링·호핑·세션 윈도우 |
| **스트림 조인** | KStream-KStream (윈도우), KStream-KTable (데이터 강화) |
| **ksqlDB** | Kafka Streams의 SQL 인터페이스 (STREAM + TABLE + 쿼리) |
| **푸시 vs 풀(Push vs Pull)** | 연속 업데이트 vs 시점 조회 |
| **정확히 한 번(Exactly-once)** | Kafka 트랜잭션으로 중복 방지 보장 |

### 모범 사례(Best Practices)

1. **이벤트 시간(event time) 사용** — 윈도우 연산 시 처리 시간이 아닌 이벤트 시간 기준
2. **윈도우 만료 설정** — 무제한 상태 성장을 방지
3. **컨슈머 랙(consumer lag) 모니터링** — 시스템 건강 상태의 핵심 지표
4. **데이터 강화에는 KTable 사용** — KStream-KStream 조인 대신 활용
5. **프로토타이핑에는 ksqlDB로 시작** — 프로덕션 환경에서는 Faust 또는 Java로 전환
6. **지연 데이터(late data) 명시적 처리** — 윈도우의 유예 기간(grace period) 설정

### 다음 단계

- **L17**: Spark 구조적 스트리밍(Spark Structured Streaming) — DataFrame 기반 스트림 처리
- **L18**: Debezium을 활용한 CDC(Change Data Capture) — Kafka 스트림으로서의 데이터베이스 변경
