[← 이전: 17. Spark Structured Streaming](17_Spark_Structured_Streaming.md) | [다음: 19. Lakehouse 실전 패턴 →](19_Lakehouse_Practical_Patterns.md)

# Debezium을 활용한 CDC

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 전통적인 배치 ETL의 한계를 설명하고, 세 가지 CDC 접근 방식(타임스탬프 기반, 트리거 기반, 로그 기반)을 비교하여 프로덕션 환경에서 로그 기반 CDC가 선호되는 이유를 밝힐 수 있습니다.
2. Debezium 아키텍처와 데이터베이스의 WAL(Write-Ahead Log)을 읽어 구조화된 변경 이벤트를 Kafka 토픽으로 내보내는 방식을 설명할 수 있습니다.
3. 슬롯 복제(Slot Replication), 커넥터 속성, 토픽 명명 규칙을 포함하여 Kafka Connect를 사용해 PostgreSQL 또는 MySQL용 Debezium 커넥터를 구성할 수 있습니다.
4. Debezium 이벤트 봉투(Event Envelope) 형식(before/after/op 필드)을 해석하여 하위 컨슈머에서 삽입, 업데이트, 삭제 연산을 처리할 수 있습니다.
5. Confluent Schema Registry와 Avro를 활용한 스키마 진화(Schema Evolution) 전략을 구현하여 소비자를 중단시키지 않고 소스 스키마 변경을 안전하게 전파할 수 있습니다.
6. 초기 스냅샷(Snapshot), 컨슈머 지연(Consumer Lag) 모니터링, 커넥터 장애 및 재시작 처리를 고려한 프로덕션 CDC 파이프라인을 설계할 수 있습니다.

---

## 이론과 원리

CDC의 더 깊은 주장은 "near-real-time"이 아니라 — **정확성** 입니다. "updated_at > last_checkpoint인 행"을 폴링하는 것은 삭제를 놓치고, 빠른 sub-second 업데이트를 놓치고, writer와 race합니다. 데이터베이스의 자체 write-ahead log를 읽으면 *모든* mutation이 commit 순서로 캡처됩니다. CDC는 OLTP 데이터베이스의 상태를 다른 곳으로 미러링하는 유일하게 옳은 방법입니다.

- **(A) 폴링(Polling) vs 로그 기반 CDC** — 각 접근이 무엇을 캡처하고 무엇을 놓치는가
- **(B) 진실의 원천으로서의 write-ahead log (WAL)** — PostgreSQL의 logical replication, MySQL의 binlog, MongoDB의 oplog
- **(C) Debezium의 snapshot + stream 모델** — 초기 상태 부트스트래핑, 그 다음 따라잡기
- **(D) 이벤트 형식과 스키마 진화** — Kafka Connect, Debezium 이벤트 envelope, schema registry
- **(E) Transactional outbox 패턴** — 분산 트랜잭션 없이 애플리케이션 수준 이벤트 소싱을 위해 CDC 사용

### A. 폴링 vs 로그 기반 CDC

#### A.1 폴링: 순진한 접근

```sql
SELECT * FROM orders WHERE updated_at > '{last_checkpoint}';
-- last_checkpoint를 결과 집합의 max(updated_at)로 업데이트
```

괜찮아 보임. 무엇을 놓치는가:

- **삭제.** 삭제된 행은 더 이상 존재하지 않음; 쿼리할 `updated_at` 없음. 폴링은 글자 그대로 삭제를 감지할 수 없음.
- **In-flight 트랜잭션.** 한 트랜잭션 내에서 10:00:01.500과 다시 10:00:01.700에 업데이트된 행은 최신만 보여줌. 사이의 업데이트는 손실.
- **Sub-second 변경.** 폴링 간격이 60초이면 그 윈도우 내 모든 변경이 "동시"로 나타남; 순서 손실.
- **행 지연.** 1분 폴링 간격은 다운스트림이 평균 1분 늦게 변경을 보는 것을 의미.
- **Race condition.** 쿼리가 읽는 순간 행이 업데이트되고 있다면? 격리 수준에 따라 옛 또는 새 값을 봄 — 그리고 어느 것인지 알 방법이 없음.

대부분 분석 사용 사례에서 폴링은 "충분히 좋음." 정확성을 요구하는 모든 것에 — 보조 데이터베이스로 복제, 스트리밍 시스템 피딩, 감사 로그 — 폴링은 잘못됨.

#### A.2 로그 기반 CDC: 옳은 접근

모든 관계형 데이터베이스는 내부적으로 *write-ahead log* (WAL)를 작성: 모든 insert/update/delete가 데이터 파일이 건드려지기 *전에* 로그에 append. 이 로그가 데이터베이스를 충돌 안전하게 만드는 것(WAL 재생으로 복구).

로그 기반 CDC:
1. 데이터베이스에 *replication 클라이언트* 로 연결(read replica가 하는 것과 같은 방식).
2. WAL 이벤트 스트리밍: 모든 insert, update, delete를 트랜잭션 commit 순서로.
3. 구조화된 변경 이벤트로 번역.
4. Kafka(또는 다른 sink)에 publish.

이것이 캡처하는 것: **모든 mutation을, commit 순서로, 전체 before/after 상태와 함께**. 삭제 포함. Sub-second 업데이트 포함. 지연: 일반적으로 commit에서 이벤트까지 <1초.

이것이 Debezium이 하는 것입니다. MySQL 복제, PostgreSQL logical replication, Oracle GoldenGate와 같은 근본적 접근.

### B. Write-Ahead Log

WAL 구조는 데이터베이스별로 다양:

#### B.1 PostgreSQL: WAL + logical replication

PostgreSQL은 물리적 WAL(페이지의 바이너리 델타)을 작성. CDC를 위해 **logical replication** 이 필요: 페이지 수준 바이너리 diff가 아닌 컬럼 값을 가진 행 수준 이벤트를 생산하는 WAL의 별도 디코딩된 스트림.

설정:
1. `wal_level = logical` 설정(재시작 필요).
2. *replication slot* 생성 — "Debezium이 어디까지 읽었는지" 기억하는 서버 측 커서.
3. *publication* 생성 — publish할 테이블 선언.
4. Debezium이 연결, slot 읽음, 이벤트 디코딩.

Replication slot은 Debezium 재시작 간 위치 유지. Debezium이 소비하는 한 PostgreSQL은 slot이 아직 읽지 않은 WAL 세그먼트를 유지. **결정적 운영 제약:** Debezium이 너무 오래 멈추면 PostgreSQL 측에 WAL이 누적되어 디스크를 채울 수 있음.

#### B.2 MySQL: binlog

MySQL은 row 형식(`binlog_format=ROW`)으로 모든 커밋된 트랜잭션의 *binary log*(binlog)를 작성. Debezium은 replication 프로토콜을 통해 binlog를 직접 읽음 — read replica가 하는 방식과 같음.

설정: row 형식으로 binlog 활성화, REPLICATION CLIENT와 REPLICATION SLAVE 권한을 가진 사용자를 Debezium에 부여.

#### B.3 MongoDB: oplog

MongoDB의 *operations log*(oplog)는 모든 mutation의 capped 컬렉션. Replica set 멤버가 동기화 유지를 위해 읽음. Debezium은 마치 replica set 멤버인 것처럼 oplog를 읽음.

#### B.4 공통 패턴

셋 모두: 데이터베이스가 자체 내부 목적(복제, 복구)을 위해 모든 변경의 로그를 이미 생산하고 있음. CDC 도구가 그 로그에 tap. **primary에 추가 부하 없음**, 애플리케이션 변경 필요 없음, 전체 mutation 이력 캡처.

### C. Debezium의 Snapshot + Stream 모델

새 CDC consumer는 *과거 상태* 와 *진행 중인 변경* 모두를 필요로 함. Debezium은 세 단계로 처리:

#### C.1 Snapshot 단계

처음 시작 시:
1. 일관된 시점 snapshot 획득(MySQL의 `FLUSH TABLES WITH READ LOCK`이나 PostgreSQL의 snapshot 트랜잭션).
2. 모니터되는 모든 테이블의 모든 행 읽음.
3. 각 행을 `READ` 이벤트로 Kafka에 emit.
4. snapshot 시작 시 WAL/binlog 위치 기록.

이는 다운스트림에 완전한 현재 상태를 제공.

#### C.2 Stream 단계

Snapshot 완료 후:
1. 기록된 위치에서 WAL/binlog 읽기 재개.
2. 각 후속 변경을 `INSERT`, `UPDATE`, 또는 `DELETE` 이벤트로 emit.

핸드오프는 정확함 — snapshot 시작 위치와 stream 시작 위치 사이의 이벤트는 놓치지 않음.

#### C.3 중복 문제

미묘한 이슈: snapshot 동안 WAL은 *또한 새 트랜잭션과 함께 진행* 되고 있었음. 위치 P에서의 Debezium snapshot은 P 이전에 커밋된 행을 포함. 위치 P에서 forward의 stream은 그것들의 효과가 snapshot 동안 발생했다면 같은 트랜잭션을 포함할 것.

현대 Debezium(2.0+)은 이를 정확하게 처리하는 *incremental snapshot* 사용: 스트림을 동시에 소비하면서 청크로 snapshot하고, 기본 키로 deduplicate.

### D. 이벤트 형식과 스키마 진화

#### D.1 Debezium envelope

각 CDC 이벤트:

```json
{
  "before": { "id": 1, "name": "Alice", "email": "alice@old.com" },
  "after":  { "id": 1, "name": "Alice", "email": "alice@new.com" },
  "source": { "ts_ms": 1700000000000, "table": "users", "db": "prod", "lsn": 12345 },
  "op": "u",
  "ts_ms": 1700000000050
}
```

`op`: `c`(create/insert), `u`(update), `d`(delete), `r`(read/snapshot).
`before`와 `after`: 행 상태. insert에 대해 `before`은 null; delete에 대해 `after`은 null.
`source`: 출처 — 정확히 어느 데이터베이스, 테이블, 로그 위치.
`ts_ms`: 이벤트가 작성된 시점.

Consumer는 변경을 결정적으로 적용 가능: update의 경우 "PK X를 가진 행이 시간 `ts_ms`에 `before`에서 `after`로 갔음."

#### D.2 Schema registry와 진화

이벤트 스키마(특히 `before`/`after`)는 형식적 — Avro나 JSON Schema로 정의되고 Confluent Schema Registry에 등록됨. 소스 테이블의 스키마가 변경되면(컬럼 추가/제거) Debezium은 새 스키마로 이벤트를 생산하고, registry가 새 버전을 저장하고, 호환 deserializer를 사용하는 consumer가 변경을 처리.

Schema Registry가 호환성을 강제: 기본적으로 backward-호환 변경(nullable 컬럼 추가, 절대 제거나 retype 안 함)만 허용. 이는 깨는 스키마 변경을 등록 시점에 잡음, consumer를 깨뜨리기 전에.

#### D.3 Kafka Connect

Debezium은 *Kafka Connect connector* 로 실행 — Connect는 운영 관심사(분산 런타임, 실패 시 재시작, 병렬성(파티션당 한 태스크), 메트릭 보고)를 처리하는 프레임워크. Debezium 자체는 소스별 로직(WAL과 대화, 이벤트 디코딩). Kafka Connect가 다른 모든 것을 제공.

### E. Transactional Outbox 패턴

악명 높은 문제를 해결하는 CDC의 특정 응용: 데이터베이스 트랜잭션과 **원자적으로** 어떻게 Kafka에 이벤트를 publish하는가?

순진한 접근:

```python
def create_order(order):
    db.insert("orders", order)              # 트랜잭션 1
    kafka.publish("order_created", order)   # 트랜잭션 안 함
```

DB insert가 성공한 후 Kafka publish가 실패하면 이벤트 없는 주문이 있음. Kafka publish 후 DB commit이 실패하면 주문 없는 이벤트가 있음. DB와 Kafka 간 two-phase commit은 비실용적.

Outbox 패턴:

```python
def create_order(order):
    with db.transaction():
        db.insert("orders", order)
        db.insert("outbox", {"event": "order_created", "payload": order})
```

두 insert가 같은 DB 트랜잭션에 있음; 둘 다 일어나거나 둘 다 안 일어남. `outbox` 테이블이 Debezium에 의해 모니터링됨; outbox에서의 CDC 이벤트가 Kafka 이벤트가 됨. 원자성 보존.

이는 관계형 데이터베이스로 받쳐진 이벤트 구동 아키텍처에서 가장 중요한 단일 패턴입니다.

### From Theory to the Code Below

이어지는 각 절은 위 프레임워크의 한 조각을 운영합니다:

- §1 (CDC 개념)은 §A — 폴링 vs 로그 기반.
- §2 (Debezium 아키텍처)는 §B + §D.3 — connector, Kafka Connect, 소스별 구현.
- §3 (PostgreSQL/MySQL Connector)는 §B.1 + §B.2 — 구체적 설정.
- §4 (이벤트 형식과 스키마 진화)는 §D — envelope, schema registry, 호환성.
- §5 (Snapshot 전략)은 §C — 초기 snapshot 패턴과 incremental snapshot.
- §6 (프로덕션 패턴)은 §E(outbox), 모니터링, 재시작, 지연을 모음.

---

## 개요

변경 데이터 캡처(CDC, Change Data Capture)는 데이터베이스에서 행 수준의 변경 사항을 캡처하여 이벤트로 스트리밍합니다. 주기적인 배치 추출 방식 대신, CDC는 시스템 간 거의 실시간(near-real-time) 데이터 동기화를 제공합니다. 이 레슨에서는 CDC 개념, Debezium 아키텍처, Kafka Connect 통합, 이벤트 형식, 스키마 진화(schema evolution), 그리고 프로덕션 패턴을 다룹니다.

---

## 1. 변경 데이터 캡처(CDC) 기초

### 1.1 왜 CDC인가?

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

### 1.2 CDC 접근 방식

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

## 2. Debezium 아키텍처

### 2.1 구성 요소

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

### 2.2 지원 데이터베이스

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

## 3. Debezium 설정

### 3.1 PostgreSQL 커넥터

```json5
// 참고: JSON은 주석을 지원하지 않습니다 — 여기서는 설명을 위해 JSON5 표기법을 사용합니다.
// 프로덕션에서는 Kafka Connect REST API에 순수 JSON을 제출할 것.
{
  "name": "inventory-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "postgres",
    "database.port": "5432",
    "database.user": "debezium",
    "database.password": "dbz_password",
    "database.dbname": "inventory",

    // topic.prefix는 모든 Kafka 토픽 이름의 첫 번째 세그먼트가 된다
    // (예: dbserver1.public.orders). 안정적인 이름을 선택할 것 — 나중에 변경하면
    // 새 토픽이 생성되어 기존 컨슈머가 모두 중단된다.
    "topic.prefix": "dbserver1",
    "schema.include.list": "public",
    // 테이블을 명시적으로 나열하면 내부 또는 임시 테이블이 의도치 않게
    // 캡처되는 것을 방지한다. 이 항목을 생략하면 스키마의 모든 테이블이 캡처된다.
    "table.include.list": "public.orders,public.customers",

    // pgoutput은 PostgreSQL의 기본 논리적 디코딩 플러그인(v10부터 내장)이다.
    // 이전 decoderbufs 플러그인은 별도의 C 확장 설치가 필요하다.
    "plugin.name": "pgoutput",
    // 각 커넥터는 자체 복제 슬롯이 필요하다 — 커넥터 간에 슬롯을 공유하면
    // 이벤트 누락이나 중복 전달이 발생한다.
    "slot.name": "debezium_slot",
    "publication.name": "dbz_publication",

    "key.converter": "org.apache.kafka.connect.json.JsonConverter",
    // 키에서 schemas.enable=false는 Kafka 메시지를 컴팩트하게 유지한다. 키는
    // 일반적으로 기본 키 값만이므로 스키마 오버헤드가 낭비적이다.
    "key.converter.schemas.enable": false,
    // 값에서 schemas.enable=true는 모든 메시지에 스키마를 포함시킨다.
    // 개발 시에는 편리하지만 메시지 크기를 ~30-50% 증가시킨다.
    // 프로덕션에서는 Avro + 스키마 레지스트리를 사용할 것 (섹션 7.2 참조).
    "value.converter": "org.apache.kafka.connect.json.JsonConverter",
    "value.converter.schemas.enable": true,

    "snapshot.mode": "initial",
    // 툼스톤 레코드(DELETE 후 null 값)는 Kafka 로그 컴팩션이
    // 토픽에서 삭제된 키를 실제로 제거하기 위해 필요하다.
    "tombstones.on.delete": true,
    // "string" 모드는 DECIMAL/NUMERIC 컬럼의 부동소수점 정밀도 손실을 방지한다 —
    // $99.99가 $99.98999...가 되지 않아야 하는 금융 데이터에서 중요하다.
    "decimal.handling.mode": "string",
    "time.precision.mode": "connect",

    // 하트비트는 저활동 기간 동안 WAL 누적을 방지한다.
    // 없으면 조용한 테이블이 복제 슬롯이 WAL 세그먼트를 무기한 보유하게 하여
    // 디스크를 가득 채울 수 있다.
    "heartbeat.interval.ms": 10000,
    "max.batch.size": 2048,
    "poll.interval.ms": 500
  }
}
```

### 3.2 스냅샷 모드(Snapshot Modes)

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

## 4. 변경 이벤트 형식(Change Event Format)

### 4.1 이벤트 구조

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

### 4.2 연산 유형(Operation Types)

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

## 5. Python으로 CDC 이벤트 소비하기

### 5.1 Kafka 컨슈머(Kafka Consumer)

```python
import json
from kafka import KafkaConsumer


def create_cdc_consumer(bootstrap_servers, topic):
    """Create a Kafka consumer for Debezium CDC events."""
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        # "earliest"는 초기 스냅샷 이벤트도 처리할 수 있게 한다.
        # 과거 데이터를 건너뛰고 현재 WAL 위치에서 시작하려면 "latest"를 사용할 것.
        auto_offset_reset="earliest",
        # 수동 커밋(enable_auto_commit=False)은 처리 완료 전 오프셋이
        # 진행되는 것을 방지한다 — 최소 한 번(at-least-once) 전달을 위한 필수 조건이다.
        # 자동 커밋 시 poll과 처리 사이의 충돌은 해당 이벤트를 조용히 건너뛰게 된다.
        enable_auto_commit=False,
        group_id="cdc-consumer-group",
        # null 검사(if m else None)는 Kafka 툼스톤 레코드를 처리한다 —
        # 툼스톤은 null 값을 가지며 행이 삭제되었음을 신호한다.
        value_deserializer=lambda m: json.loads(m.decode("utf-8")) if m else None,
        key_deserializer=lambda m: json.loads(m.decode("utf-8")) if m else None,
    )
    return consumer


def process_change_event(event):
    """Process a single Debezium change event."""
    if event is None:
        return  # 툼스톤 이벤트 (DELETE 이후 null 값)

    # Debezium은 value.converter.schemas.enable 설정에 따라 스키마 봉투(envelope)
    # 유무에 관계없이 이벤트를 내보낼 수 있다. 이 폴백은 두 형식을 모두 처리하여
    # 커넥터 설정에 관계없이 컨슈머가 동작하도록 한다.
    payload = event.get("payload", event)

    op = payload.get("op")
    before = payload.get("before")
    after = payload.get("after")
    source = payload.get("source", {})

    # 소스 메타데이터에서 완전한 테이블 이름을 재구성하면
    # 단일 컨슈머가 여러 테이블의 이벤트를 처리하면서
    # 각 이벤트를 올바른 대상으로 라우팅할 수 있다.
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
        # 토픽 이름은 Debezium 규칙을 따른다: {topic.prefix}.{schema}.{table}
        topic="dbserver1.public.orders",
    )

    try:
        for message in consumer:
            process_change_event(message.value)
            # 메시지별 커밋은 가장 강력한 최소 한 번(at-least-once) 보장을 제공하지만
            # 지연(latency)을 추가한다(이벤트당 Kafka 왕복 1회). 더 높은 처리량을 위해서는
            # N개 메시지마다 또는 T초마다 배치 커밋하되, 충돌 시 미커밋 배치가
            # 재처리됨을 허용할 것.
            consumer.commit()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # close()는 컨슈머 그룹 리밸런싱을 트리거하여 그룹의 다른 컨슈머가
        # 즉시 이 컨슈머의 파티션을 인계받을 수 있게 한다.
        consumer.close()
```

### 5.2 구체화된 뷰(Materialized View) 구성

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
        # 기본 키로 키가 지정된 dict를 사용하면 업데이트와 삭제에서 O(1) 조회가
        # 가능하다 — 높은 처리량의 CDC 스트림을 따라잡기 위해 필수적이다.
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

        # 스냅샷 읽기(op="r")는 스트리밍 시작 전 초기 테이블 상태를
        # 나타내므로 삽입과 동일하게 처리한다.
        if op in ("c", "r"):
            key = after[self.key_field]
            self.data[key] = after
            self.stats["inserts"] += 1

        elif op == "u":
            # Debezium이 트랜잭션 순서를 보존하므로 `after`로 덮어쓰는 것이 올바르다 —
            # 항상 최신 상태를 보게 된다.
            key = after[self.key_field]
            self.data[key] = after
            self.stats["updates"] += 1

        elif op == "d":
            # 삭제 시 `after`가 null이므로 키는 `before`에서 가져온다.
            # 기본값 None을 가진 pop()은 이 행을 본 적 없는 드문 경우
            # (예: 스트림 중간에 컨슈머 시작)에도 안전하게 처리한다.
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

    # auto_offset_reset="earliest"가 여기서 중요하다: 구체화된 뷰는
    # 현재 테이블 상태의 완전한 그림을 구축하기 위해
    # 모든 이벤트(초기 스냅샷 포함)를 처음부터 재처리해야 한다.
    consumer = KafkaConsumer(
        "dbserver1.public.orders",
        bootstrap_servers=["localhost:9092"],
        auto_offset_reset="earliest",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")) if m else None,
    )

    for message in consumer:
        view.apply_event(message.value)

        # 100번째 삽입마다 진행 상황을 로그로 기록한다. 프로덕션에서는
        # 높은 처리량 스트림에서 로그가 넘치지 않도록 적절한 메트릭
        # (Prometheus 카운터)으로 교체할 것.
        if view.stats["inserts"] % 100 == 0:
            print(f"View size: {len(view)}, Stats: {dict(view.stats)}")

            # Query example: find high-value orders
            high_value = view.query(lambda r: r.get("amount", 0) > 1000)
            print(f"High-value orders: {len(high_value)}")
```

---

## 6. 단일 메시지 변환(SMT, Single Message Transforms)

### 6.1 일반적인 변환

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

### 6.2 아웃박스 패턴(Outbox Pattern)

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

## 7. 스키마 진화(Schema Evolution)

### 7.1 스키마 변경 처리

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

### 7.2 스키마 레지스트리(Schema Registry) 통합

```json5
// Avro + 스키마 레지스트리 커넥터 — 프로덕션에서 JSON보다 권장됨.
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

    // Avro 직렬화는 스키마를 메시지마다 반복하지 않고 레지스트리에 한 번만
    // 저장하므로 JSON 대비 메시지 크기를 50-80% 줄인다.
    // 레지스트리는 호환성 규칙도 적용하여 파괴적 스키마 변경이
    // 컨슈머에 도달하기 전에 차단한다.
    "key.converter": "io.confluent.connect.avro.AvroConverter",
    "key.converter.schema.registry.url": "http://schema-registry:8081",
    "value.converter": "io.confluent.connect.avro.AvroConverter",
    "value.converter.schema.registry.url": "http://schema-registry:8081",

    // 이 내부 토픽은 소스 데이터베이스의 DDL 변경(CREATE, ALTER, DROP)을 기록한다.
    // Debezium은 커넥터 시작 전의 스키마를 참조하는 WAL 항목을 올바르게 해석하기 위해
    // 이 이력이 필요하다.
    "schema.history.internal.kafka.bootstrap.servers": "kafka:9092",
    "schema.history.internal.kafka.topic": "schema-changes.inventory"
  }
}
```

---

## 8. 프로덕션 패턴(Production Patterns)

### 8.1 Debezium 모니터링

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

### 8.2 장애 복구(Failure Recovery)

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

### 8.3 Docker Compose 설정

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
    # wal_level=logical은 논리적 복제를 지원하는 최소 WAL 레벨이다 —
    # 없으면 Debezium이 행 수준 변경을 디코드할 수 없다.
    # max_replication_slots와 max_wal_senders는 Debezium 커넥터 수 +
    # 다른 복제 컨슈머(예: 읽기 전용 복제본) 수 이상이어야 한다.
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
      # 복제 팩터 1은 로컬 개발 환경에서만 허용된다.
      # 프로덕션에서는 브로커 장애 없이 데이터 손실을 방지하기 위해 3을 사용할 것.
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    ports:
      - "9092:9092"

  connect:
    image: debezium/connect:2.5
    # kafka와 postgres 모두 Kafka Connect 시작 전에 준비되어야 한다 —
    # 브로커에 접근할 수 없으면 Connect가 커넥터 등록에 실패하고,
    # 소스 데이터베이스가 준비되지 않으면 커넥터도 실패한다.
    depends_on: [kafka, postgres]
    environment:
      BOOTSTRAP_SERVERS: kafka:9092
      GROUP_ID: "debezium-connect"
      # 이 세 내부 토픽은 커넥터 상태를 저장한다. 삭제하면
      # 모든 커넥터가 오프셋 위치를 잃고 재스냅샷을 해야 한다.
      CONFIG_STORAGE_TOPIC: connect_configs
      OFFSET_STORAGE_TOPIC: connect_offsets
      STATUS_STORAGE_TOPIC: connect_statuses
    ports:
      - "8083:8083"
```

---

## 9. 연습 문제

### 연습 1: CDC 파이프라인 구성

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

### 연습 2: 아웃박스 패턴 구현

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

## 연습 문제

### 연습 1: Debezium 커넥터 배포 및 검증

Docker Compose를 사용하여 처음부터 최소한의 CDC 파이프라인을 설정하고 모든 변경 유형을 검증하세요:

1. 8.3절의 Docker Compose 스택(PostgreSQL, Zookeeper, Kafka, Kafka Connect)을 시작하세요
2. `inventory` 데이터베이스에 `orders` 테이블을 생성하세요:
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
3. Kafka Connect REST API를 통해 Debezium PostgreSQL 커넥터를 등록하세요 (`POST /connectors`)
4. `orders` 테이블에 INSERT, UPDATE, DELETE 연산을 수행하고 `kafka-console-consumer`로 생성된 Kafka 메시지를 소비하세요
5. 각 연산에 대해 `op` 필드 값을 기록하고 `before`와 `after` 필드의 구조를 설명하세요
6. 작성한 노트에서 설명하세요: `REPLICA IDENTITY FULL`이 UPDATE와 DELETE 이벤트에서 특히 중요한 이유는 무엇인가요?

### 연습 2: 컨슈머를 활용한 구체화된 캐시 구축

5.2절의 `MaterializedView` 클래스를 확장하여 다중 테이블 CDC 소비를 지원하세요:

1. `MaterializedView`를 수정하여 `table_name` 파라미터를 받고 `(table_name, primary_key)`를 키로 데이터를 저장하세요
2. 여러 Debezium 토픽(예: `dbserver1.public.orders`와 `dbserver1.public.customers`)을 구독하는 `MultiTableConsumer` 클래스를 작성하세요
3. 이벤트 페이로드의 `source.table` 필드를 기반으로 각 CDC 이벤트를 올바른 `MaterializedView` 인스턴스로 라우팅하세요
4. 주문을 조회한 후 customers 뷰에서 해당 고객을 가져오는 `query_join(order_id)` 메서드를 추가하세요 — 메모리 내 크로스 테이블 조인 시뮬레이션
5. 테이블별 삽입, 업데이트, 삭제 건수를 반환하는 `stats()` 메서드를 구현하세요
6. 10개의 주문과 5명의 고객을 삽입하고, 3개의 주문을 업데이트하고, 1명의 고객을 삭제한 후 — 뷰 크기와 통계가 예상과 일치하는지 검증하세요

### 연습 3: 단일 메시지 변환(SMT) 적용

Debezium 커넥터에 세 가지 SMT(Single Message Transform)를 구성하고 효과를 관찰하세요:

1. **RegexRouter**: 모든 캡처 토픽을 `dbserver1.public.*`에서 `cdc.*`로 라우팅하세요 (예: `cdc.orders`, `cdc.customers`)
2. **ExtractNewRecordState**: Debezium 봉투(envelope)를 언랩(unwrap)하여 컨슈머가 `__op`와 `__source_ts_ms` 두 개의 추가 필드와 함께 평탄화된 레코드(`after` 필드만)를 받도록 하세요
3. **InsertField**: 모든 이벤트에 현재 벽시계 시간을 가진 `cdc_processed_at` 필드를 추가하세요
4. 커넥터 설정에서 세 변환을 모두 체인으로 연결하세요 (transforms 속성은 쉼표로 구분된 목록을 허용합니다)
5. 변환된 토픽의 메시지를 소비하고 SMT 적용 전후의 메시지 구조를 비교하세요
6. 주석으로 설명하세요: `ExtractNewRecordState`를 사용하는 경우와 `before`/`after` 필드가 있는 전체 봉투(envelope)를 소비하는 경우를 언제 선택하나요?

### 연습 4: 스키마 진화(Schema Evolution)와 레지스트리 통합

Debezium이 스키마 변경을 처리하는 방식을 테스트하고 Confluent Schema Registry를 통합하세요:

1. 커넥터가 실행 중인 상태에서 `orders` 테이블에 새로운 널 허용(nullable) 컬럼을 추가하세요:
   ```sql
   ALTER TABLE orders ADD COLUMN priority VARCHAR(10);
   ```
2. 새 행을 삽입하고 Kafka 메시지에 이제 `priority` 필드가 포함되는지 확인하세요; 오래된 행(`priority` 없는)은 null로 표시되었는지 검증하세요
3. Avro 직렬화와 Schema Registry를 사용하도록 커넥터를 구성하세요 (`key.converter`와 `value.converter`를 `AvroConverter`로 업데이트)
4. 파괴적인 스키마 변경(not-null 컬럼 삭제)을 시도하고 Schema Registry의 호환성 검사가 호환성 모드에 따라 어떻게 차단하거나 허용하는지 관찰하세요
5. 호환성 모드를 `FULL`로 설정하고 후방 및 전방 호환 변경만 허용되는지 확인하세요
6. 설명하세요: `BACKWARD`, `FORWARD`, `FULL` 호환성 모드의 차이는 무엇인가요? 각 모드가 허용하지만 다른 모드는 허용하지 않을 수 있는 DDL 변경 예시를 하나씩 제시하세요

### 연습 5: 트랜잭션 아웃박스 패턴(Transactional Outbox Pattern) 구현

아웃박스 패턴을 사용하여 주문 이벤트를 원자적으로 발행하는 마이크로서비스를 구축하세요:

1. PostgreSQL에 `orders` 테이블과 `outbox` 테이블을 생성하세요:
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
2. `create_order(customer_id, amount)` Python 함수를 작성하세요:
   - `orders`에 행을 삽입합니다
   - 단일 `BEGIN`/`COMMIT` 트랜잭션 내에서 `outbox`에 해당 이벤트를 삽입합니다
3. `aggregate_type`으로 지정된 토픽(예: `order.events`)으로 이벤트를 라우팅하는 `EventRouter` SMT를 사용하여 `outbox` 테이블만 캡처하는 Debezium 커넥터를 구성하세요
4. `order.events`를 읽어 각 이벤트의 `type`, `aggregate_id`, `payload`를 출력하는 Kafka 컨슈머를 작성하세요
5. 장애를 시뮬레이션하세요: `create_order`를 호출하지만 커밋(commit) 전에 서비스를 종료 — Kafka에 이벤트가 나타나지 않는지 확인하세요
6. 설명하세요: 아웃박스 패턴이 이중 쓰기(dual-write) 문제를 어떻게 해결하나요? 아웃박스 테이블 대신 애플리케이션에서 Kafka로 직접 쓰면 어떤 문제가 발생하나요?

---

## 10. 요약

### 핵심 정리

| 개념 | 설명 |
|------|------|
| **로그 기반 CDC(Log-based CDC)** | 트랜잭션 로그(WAL/binlog) 읽기로 소스 부하 최소화 |
| **Debezium** | Kafka Connect 기반의 오픈소스 CDC 플랫폼 |
| **이벤트 형식(Event format)** | 연산 유형 및 소스 메타데이터를 포함한 변경 전/후 스냅샷 |
| **스냅샷 모드(Snapshot modes)** | 시작 방식에 따라 initial, no_data, schema_only 중 선택 |
| **SMT** | 파이프라인 내 이벤트 변환(평탄화, 라우팅, 필터링) |
| **아웃박스 패턴(Outbox pattern)** | 아웃박스 테이블을 통한 원자적 DB 쓰기 + 이벤트 발행 |
| **스키마 진화(Schema evolution)** | 호환성 모드가 있는 스키마 레지스트리(Schema Registry) 활용 |

### 모범 사례(Best Practices)

1. **캡처 대상 테이블에 REPLICA IDENTITY FULL 설정** — 변경 전/후 이미지 완전 보장
2. **WAL 보존 모니터링** — 무제한 증가는 복제 슬롯(replication slot)이 멈춘 신호
3. **스키마 레지스트리 사용** — 파괴적 변경(breaking change)이 전파되는 것을 방지
4. **ExtractNewRecordState SMT** — 다운스트림 처리 단순화
5. **하트비트 이벤트(Heartbeat events)** — 저활동 기간 중 WAL 누적 방지
6. **프로덕션 적용 전 스테이징 환경에서 스키마 변경 테스트**

### 다음 단계

- **L19**: Lakehouse 실전 패턴 — CDC 스트림을 Delta Lake / Iceberg에 적재
- **L12**로 돌아가기: Great Expectations를 활용한 데이터 품질 검증 — 다운스트림 CDC 데이터 검증

---

[← 이전: 17. Spark Structured Streaming](17_Spark_Structured_Streaming.md) | [다음: 19. Lakehouse 실전 패턴 →](19_Lakehouse_Practical_Patterns.md)
