# NewSQL과 현대 트렌드

**이전**: [14. 분산 데이터베이스](./14_Distributed_Databases.md) | **다음**: [16. 데이터베이스 설계 사례 연구](./16_Database_Design_Case_Study.md)

---

NoSQL 혁명 이후 데이터베이스 환경은 극적으로 진화했습니다. 새로운 세대의 시스템은 두 세계의 장점을 결합하는 것을 목표로 합니다: NoSQL의 수평적 확장성과 가용성, 그리고 관계형 데이터베이스의 ACID 보장과 SQL 인터페이스. NewSQL을 넘어, 특화된 워크로드를 제공하기 위해 완전히 새로운 범주의 데이터베이스가 등장했습니다 -- AI를 위한 벡터 유사도 검색, IoT를 위한 시계열 저장소, 그리고 대규모 그래프 분석. 이 레슨은 데이터베이스 기술의 최첨단과 이러한 시스템을 가능하게 하는 아키텍처 혁신을 조사합니다.

**난이도**: ⭐⭐⭐

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. NewSQL 설계 철학과 ACID를 수평적 확장성과 어떻게 조화시키는지 설명
2. Google Spanner의 TrueTime 메커니즘과 외부 일관성 설명
3. CockroachDB, TiDB, Spanner 아키텍처 비교
4. 벡터 데이터베이스와 유사도 검색 알고리즘(HNSW, IVF) 이해
5. 시계열 데이터베이스 최적화 설명
6. 대규모 분석을 위한 그래프 분석 플랫폼 평가
7. Database-as-a-Service 제공 서비스 평가
8. 데이터 레이크하우스 패러다임과 데이터베이스와의 관계 이해

---

## 목차

1. [NewSQL: 동기와 설계 철학](#1-newsql-동기와-설계-철학)
2. [Google Spanner](#2-google-spanner)
3. [CockroachDB](#3-cockroachdb)
4. [TiDB](#4-tidb)
5. [벡터 데이터베이스](#5-벡터-데이터베이스)
6. [시계열 데이터베이스](#6-시계열-데이터베이스)
7. [대규모 그래프 분석](#7-대규모-그래프-분석)
8. [Database-as-a-Service](#8-database-as-a-service)
9. [데이터 레이크하우스](#9-데이터-레이크하우스)
10. [연습 문제](#10-연습-문제)
11. [참고문헌](#11-참고문헌)

---

## 1. NewSQL: 동기와 설계 철학

### 1.1 SQL과 NoSQL 사이의 간극

2010년대 초반까지, 데이터베이스 세계는 두 진영으로 나뉘었습니다:

```
전통적 SQL                              NoSQL
┌──────────────────────────┐    ┌──────────────────────────┐
│ ✓ ACID 트랜잭션          │    │ ✓ 수평적 확장성           │
│ ✓ SQL 인터페이스         │    │ ✓ 높은 가용성             │
│ ✓ 풍부한 쿼리 기능       │    │ ✓ 유연한 스키마           │
│ ✓ 강한 일관성            │    │ ✓ 낮은 지연시간           │
│ ✗ 수평적 확장            │    │ ✗ ACID 없음 (일반적)      │
│ ✗ 지리적 분산            │    │ ✗ SQL 없음 (또는 제한적)  │
│ ✗ 자동 장애조치          │    │ ✗ 약한 일관성             │
└──────────────────────────┘    └──────────────────────────┘

                    간극: 둘 다 가질 수 있을까?

NewSQL
┌──────────────────────────┐
│ ✓ ACID 트랜잭션          │
│ ✓ SQL 인터페이스         │
│ ✓ 수평적 확장성          │
│ ✓ 높은 가용성            │
│ ✓ 강한 일관성            │
│ ✓ 지리적 분산            │
│ ✓ 자동 장애조치          │
└──────────────────────────┘
```

### 1.2 NewSQL 정의

"NewSQL"이라는 용어는 2011년 451 Group의 Matthew Aslett에 의해 다음과 같은 새로운 클래스의 관계형 데이터베이스 관리 시스템을 설명하기 위해 만들어졌습니다:

1. **ACID 보장 제공** - 읽기-쓰기 트랜잭션에 대해
2. **SQL 사용** - 주요 인터페이스로
3. **수평적 확장** - 공유 없는 아키텍처를 사용하여 범용 하드웨어에서
4. **처리량 달성** - OLTP 워크로드에 대해 NoSQL 시스템과 비교하거나 초과하는

### 1.3 아키텍처 혁신

NewSQL 시스템은 일반적으로 세 가지 주요 혁신을 사용합니다:

**1. 분산 트랜잭션을 사용한 샤딩(Sharding)**: 데이터는 자동으로 노드에 분할(샤딩)되지만, 시스템은 합의 프로토콜(Paxos/Raft)을 통해 투명한 분산 트랜잭션을 제공하여 애플리케이션에서 샤딩이 보이지 않도록 합니다.

**2. 다중 버전 동시성 제어(Multi-version concurrency control, MVCC)**: 잠금 대신, NewSQL 시스템은 전역으로 정렬된 타임스탬프와 함께 MVCC를 사용하여 읽기가 쓰기를 차단하지 않고 진행되도록 합니다.

**3. 합의 기반 복제(Consensus-based replication)**: 각 파티션(샤드)은 Raft 또는 Paxos를 사용하여 여러 노드에 복제되어 높은 가용성과 강한 일관성을 모두 제공합니다.

```
NewSQL 아키텍처 (일반):

Client ──▶ SQL Layer (Parse, Plan, Optimize)
                │
                ▼
           Transaction Layer (Distributed MVCC, 2PC/Raft)
                │
                ▼
           Storage Layer (Sharded, Replicated)
           ┌──────┐  ┌──────┐  ┌──────┐
           │Shard 1│  │Shard 2│  │Shard 3│
           │R1,R2,R3│  │R1,R2,R3│  │R1,R2,R3│  ← 각 샤드는
           └──────┘  └──────┘  └──────┘     3개의 복제본을 가짐 (Raft 그룹)
```

### 1.4 NewSQL vs 샤딩된 PostgreSQL

흔한 질문은: "왜 PostgreSQL을 샤딩하지 않나요?" 이 답은 NewSQL을 특별하게 만드는 것이 무엇인지 보여줍니다:

| 기능 | 샤딩된 PostgreSQL (Citus) | NewSQL (CockroachDB) |
|---------|---------------------------|----------------------|
| **샤드 간 트랜잭션** | 제한적 (샤드 간 2PC) | 완전한 ACID (Raft + MVCC) |
| **자동 리샤딩** | 수동 또는 반자동 | 자동 범위 분할 |
| **스키마 변경** | 롤링 DDL 필요 | 온라인 스키마 변경 |
| **복제** | 비동기/동기 (샤드별) | Raft (범위별) |
| **장애조치** | 수동 또는 스크립트 | 자동 (Raft 리더 선출) |
| **글로벌 분산** | 가능하지만 복잡 | 내장 (지리적 파티셔닝) |
| **호환성** | 완전한 PostgreSQL | PostgreSQL 와이어 프로토콜 (부분집합) |

---

## 2. Google Spanner

### 2.1 개요

2012년 논문에서 소개된 Google Spanner는 **외부적으로 일관된(externally consistent)** (선형화 가능한, linearizable) 트랜잭션을 제공하면서 데이터를 글로벌 규모로 분산하는 최초의 시스템입니다. 이것은 다른 모든 시스템에 영감을 준 원래의 NewSQL 시스템입니다.

```
Google Spanner 아키텍처:

┌─────────────────────────────────────────────────────────────┐
│                        Universe                              │
│                                                             │
│  Zone (US-East)        Zone (EU-West)        Zone (Asia)   │
│  ┌───────────────┐    ┌───────────────┐    ┌─────────────┐ │
│  │ Spanserver 1  │    │ Spanserver 4  │    │ Spanserver 7│ │
│  │ Spanserver 2  │    │ Spanserver 5  │    │ Spanserver 8│ │
│  │ Spanserver 3  │    │ Spanserver 6  │    │ Spanserver 9│ │
│  │               │    │               │    │             │ │
│  │ ┌─TrueTime─┐  │    │ ┌─TrueTime─┐  │    │ ┌─TrueTime┐│ │
│  │ │GPS+Atomic│  │    │ │GPS+Atomic│  │    │ │GPS+Atom.││ │
│  │ │  Clocks  │  │    │ │  Clocks  │  │    │ │ Clocks  ││ │
│  │ └──────────┘  │    │ └──────────┘  │    │ └─────────┘│ │
│  └───────────────┘    └───────────────┘    └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 TrueTime

Spanner의 핵심 혁신은 **TrueTime**입니다. 이것은 단일 타임스탬프 대신 시간 구간을 반환하는 전역 동기화된 시계 API입니다.

```
전통적인 시계:    now() → T           (단일 지점, 알 수 없는 오차)
TrueTime:        TT.now() → [T-ε, T+ε]  (경계가 있는 오차를 가진 구간)

여기서 ε는 시계 불확실성이며, 일반적으로 1-7 밀리초입니다.

TrueTime 보장:
  [earliest, latest]를 반환하는 TT.now()의 모든 호출에 대해:
    earliest ≤ 실제_시간 ≤ latest

TrueTime 구현:
  - GPS 수신기: 나노초 정확도로 시간 제공
  - 원자 시계: 천천히 드리프트, GPS 신호 손실 시 사용
  - 각 데이터센터에 중복성을 위한 여러 시간 소스
  - 타이트한 경계를 계산하기 위한 Marzullo의 알고리즘
```

### 2.3 커밋 대기를 통한 외부 일관성

Spanner는 TrueTime을 사용하여 **외부 일관성(external consistency)** (엄격한 직렬화 가능성 또는 선형화 가능성이라고도 함)을 달성합니다: 트랜잭션 T1이 트랜잭션 T2가 시작하기 전에 커밋되면(실제 시간으로), T1의 커밋 타임스탬프는 T2의 커밋 타임스탬프보다 작습니다.

```
커밋 대기 프로토콜:

1. 트랜잭션 T가 모든 잠금을 획득
2. T가 커밋 타임스탬프 s = TT.now().latest를 선택
3. COMMIT WAIT: T가 TT.after(s)가 true가 될 때까지 대기
   즉, 시계 불확실성이 지나갈 때까지 대기
4. T가 타임스탬프 s로 커밋하고 잠금 해제

작동 원리:
  - T가 실제 시간 t_pick에서 s = TT.now().latest를 선택
  - 커밋 대기 후, 실제 시간 t_commit은 t_commit > s를 만족
  - t_commit 이후에 시작하는 모든 미래 트랜잭션 T'는
    s' = TT.now().latest > t_commit > s를 얻음
  - 따라서 s < s' → T는 T' 이전에 직렬화됨 (외부 일관성)

타임라인:
  ├──────┤ε├──────┤
  |      |s|      |
  |      |        |
  |   TT.now()    |
  |   [T-ε, T+ε]  |
  |               |
  |    Commit     |
  |    Wait       |
  |               |
  |               ├── TT.after(s) = true → COMMIT
```

**커밋 대기 비용**: 모든 쓰기 트랜잭션에 약 7ms 지연시간 추가 (평균 시계 불확실성).

### 2.4 Spanner SQL과 기능

Spanner는 분산 의미론을 위한 확장이 있는 SQL 방언을 지원합니다:

```sql
-- 인터리빙으로 테이블 생성 (자식 행을 부모와 같은 위치에 배치)
CREATE TABLE Customers (
  CustomerId INT64 NOT NULL,
  Name STRING(100),
  Email STRING(256)
) PRIMARY KEY (CustomerId);

CREATE TABLE Orders (
  CustomerId INT64 NOT NULL,
  OrderId INT64 NOT NULL,
  Total NUMERIC,
  CreatedAt TIMESTAMP
) PRIMARY KEY (CustomerId, OrderId),
  INTERLEAVE IN PARENT Customers ON DELETE CASCADE;

-- 인터리빙은 주문을 고객과 물리적으로 같은 위치에 배치
-- 이것은 고객-주문 조인을 위한 네트워크 왕복을 제거합니다!

-- 읽기-쓰기 트랜잭션 (직렬화 가능)
BEGIN TRANSACTION;
  UPDATE Accounts SET Balance = Balance - 100 WHERE AccountId = 1;
  UPDATE Accounts SET Balance = Balance + 100 WHERE AccountId = 2;
COMMIT;

-- 읽기 전용 트랜잭션 (잠금 없음, 커밋 대기 없음)
-- 일관된 읽기를 위해 스냅샷 타임스탬프 사용
SET TRANSACTION READ ONLY;
SELECT * FROM Orders WHERE CustomerId = 42;

-- 오래된 읽기 (제한된 staleness, 더 낮은 지연시간을 위해)
SELECT * FROM Orders
  WHERE CustomerId = 42
  WITH (MAX_STALENESS = 10s);
```

### 2.5 Spanner 요약

| 기능 | 세부사항 |
|---------|--------|
| **일관성** | 외부 일관성 (선형화 가능성) |
| **시계** | TrueTime (GPS + 원자 시계) |
| **복제** | 분할당 다중 Paxos |
| **파티셔닝** | 범위 기반, 자동 분할/병합 |
| **트랜잭션** | Paxos 그룹 간 2PC |
| **SQL** | 확장이 있는 ANSI SQL |
| **가용성** | 99.999% SLA (5 nines) |
| **지연시간** | 약 7ms 쓰기 (커밋 대기), 낮은 ms 읽기 |

---

## 3. CockroachDB

### 3.1 개요

CockroachDB (CRDB)는 Spanner에서 영감을 받았지만 특수 하드웨어 없이(GPS/원자 시계 없이) 실행되도록 설계된 오픈소스 NewSQL 데이터베이스입니다. 바퀴벌레의 회복력을 위해 이름이 지어졌습니다 -- 모든 수준에서 장애에서 살아남도록 설계되었습니다.

### 3.2 아키텍처

```
CockroachDB 아키텍처:

┌──────────────────────────────────────────────────────────┐
│                     SQL Layer                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ Gateway  │  │ Gateway  │  │ Gateway  │  (any node)   │
│  │ Node     │  │ Node     │  │ Node     │               │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
│       │              │              │                    │
│       └──────────────┼──────────────┘                    │
│                      │                                   │
│  ┌───────────────────┴────────────────────────────────┐  │
│  │              Distribution Layer                     │  │
│  │  Key space divided into Ranges (~512MB each)       │  │
│  │  Each Range is a Raft group (3+ replicas)          │  │
│  │  Range 1: [a, f)   Range 2: [f, m)   Range 3: ... │  │
│  └────────────────────────────────────────────────────┘  │
│                      │                                   │
│  ┌───────────────────┴────────────────────────────────┐  │
│  │              Storage Layer (Pebble)                 │  │
│  │  LSM-tree based key-value store                    │  │
│  │  MVCC timestamps on every key                      │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 3.3 다중 활성 가용성

전통적인 주-대기(primary-standby) 설정과 달리, CockroachDB는 전체 데이터베이스에 대한 단일 리더가 없습니다. 각 Range는 자체 Raft 리더를 가지며, 리더는 모든 노드에 분산됩니다:

```
Node 1                  Node 2                  Node 3
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│ Range A: LEADER│     │ Range A: Follower│     │ Range A: Follower│
│ Range B: Follower│     │ Range B: LEADER│     │ Range B: Follower│
│ Range C: Follower│     │ Range C: Follower│     │ Range C: LEADER│
└────────────────┘     └────────────────┘     └────────────────┘

모든 노드는 읽기와 쓰기를 모두 제공합니다 (리드하는 범위에 대해).
단일 장애 지점 없음!
```

### 3.4 TrueTime 없는 직렬화 가능한 격리

CRDB는 TrueTime 없이 범용 하드웨어에서 실행되므로 트랜잭션 정렬을 위해 다른 메커니즘을 사용합니다:

**하이브리드 논리 시계(Hybrid Logical Clocks, HLC)**: 물리적 시계 구성요소와 논리 카운터를 결합합니다.

```
HLC = (physical_time, logical_counter, node_id)

규칙:
1. 로컬 이벤트: HLC.physical = max(HLC.physical, wall_clock)
                HLC.logical += 1
2. 메시지 전송: 현재 HLC 첨부
3. HLC_msg로 메시지 수신:
   HLC.physical = max(HLC.physical, HLC_msg.physical, wall_clock)
   if HLC.physical == HLC_msg.physical:
     HLC.logical = max(HLC.logical, HLC_msg.logical) + 1
   else:
     HLC.logical = 0
```

**시계 스큐 처리**: CRDB는 최대 시계 오프셋(기본값 500ms)을 강제합니다. 시계가 이를 초과하여 드리프트하면 노드가 클러스터에서 제거됩니다. 시계 스큐의 영향을 받을 수 있는 트랜잭션의 경우, CRDB는 **불확실성 구간(uncertainty intervals)**과 **읽기 새로고침(read refreshes)**을 사용합니다:

```
트랜잭션 T1이 타임스탬프 t에서 키 K를 읽음:
  - T1은 타임스탬프 t_w에서 작성된 값 V를 관찰
  - t_w가 T1의 불확실성 구간 [t, t + max_offset] 내에 있으면:
    T1은 V가 T1이 시작하기 전에 또는 후에 작성되었는지 결정할 수 없음
    → T1은 타임스탬프를 t_w 이후로 푸시하고 읽기를 재시도

이것은 "읽기 재시작"이라고 하며 직렬화 가능성을 보장합니다.
```

### 3.5 주요 기능

```sql
-- 지리적 파티셔닝: 특정 지역에 데이터 고정
ALTER TABLE users PARTITION BY LIST (country) (
  PARTITION us VALUES IN ('US'),
  PARTITION eu VALUES IN ('DE', 'FR', 'UK'),
  PARTITION asia VALUES IN ('JP', 'KR', 'SG')
);

ALTER PARTITION us OF TABLE users
  CONFIGURE ZONE USING constraints='[+region=us-east1]';
ALTER PARTITION eu OF TABLE users
  CONFIGURE ZONE USING constraints='[+region=europe-west1]';

-- 리더 따라가기 (가장 가까운 복제본에서 낮은 지연시간 읽기)
ALTER TABLE users CONFIGURE ZONE USING
  lease_preferences='[[+region=us-east1],[+region=europe-west1]]';

-- 온라인 스키마 변경 (다운타임 없음)
ALTER TABLE orders ADD COLUMN discount DECIMAL DEFAULT 0;
-- 백그라운드 작업으로 실행, 테이블 잠금 없음!

-- 변경 데이터 캡처 (CDC)
CREATE CHANGEFEED FOR TABLE orders INTO 'kafka://broker:9092';
```

### 3.6 CockroachDB 요약

| 기능 | 세부사항 |
|---------|--------|
| **일관성** | 직렬화 가능한 격리 |
| **시계** | HLC (하이브리드 논리 시계) |
| **복제** | Range당 Raft |
| **파티셔닝** | 자동 범위 분할 (~512MB) |
| **SQL** | PostgreSQL 와이어 프로토콜 호환 |
| **지리적 분산** | 지리적 파티셔닝, 팔로워 읽기 |
| **스키마 변경** | 온라인, 비차단 |
| **라이선스** | BSL (대부분의 사용에 무료) |

---

## 4. TiDB

### 4.1 개요

TiDB (Ti는 티타늄(Titanium)을 의미)는 PingCAP이 만든 오픈소스 NewSQL 데이터베이스입니다. 구별되는 기능은 **HTAP (Hybrid Transactional/Analytical Processing)**입니다: 단일 시스템에서 OLTP와 OLAP 워크로드를 모두 처리할 수 있습니다.

### 4.2 아키텍처

```
TiDB 아키텍처:

                    ┌─────────┐
                    │  Client │
                    └────┬────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         ┌────────┐ ┌────────┐ ┌────────┐
         │ TiDB   │ │ TiDB   │ │ TiDB   │   SQL Layer
         │ Server │ │ Server │ │ Server │   (stateless)
         └────┬───┘ └────┬───┘ └────┬───┘
              │          │          │
              └──────────┼──────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         ┌────────┐ ┌────────┐ ┌────────┐
         │ TiKV   │ │ TiKV   │ │ TiKV   │   Row Store (OLTP)
         │(Raft)  │ │(Raft)  │ │(Raft)  │   (transactional)
         └────────┘ └────────┘ └────────┘
              │          │          │
              └──────────┼──────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         ┌────────┐ ┌────────┐ ┌────────┐
         │TiFlash │ │TiFlash │ │TiFlash │   Column Store (OLAP)
         │(Raft   │ │(Raft   │ │(Raft   │   (analytical)
         │Learner)│ │Learner)│ │Learner)│
         └────────┘ └────────┘ └────────┘

Placement Driver (PD): 클러스터 메타데이터, 타임스탬프 오라클, 스케줄링
```

### 4.3 HTAP: OLTP와 OLAP의 연결

전통적인 아키텍처는 OLTP (쓰기 중심, 행 지향)와 OLAP (읽기 중심, 열 지향)를 서로 다른 시스템으로 분리하고, 그 사이에 ETL 파이프라인을 두었습니다. TiDB는 둘 다 통합합니다:

```
전통적 아키텍처:
  OLTP (MySQL) ──ETL──▶ OLAP (Data Warehouse)
                        (몇 시간의 지연)

TiDB HTAP 아키텍처:
  ┌──────────────────────────────────────────┐
  │              TiDB                         │
  │                                           │
  │  OLTP 쿼리 ──▶ TiKV (행 저장소)          │
  │  OLAP 쿼리 ──▶ TiFlash (열 저장소)       │
  │                                           │
  │  Raft 복제: TiKV ──▶ TiFlash              │
  │  (실시간, Raft Learner를 통한 비동기)     │
  └──────────────────────────────────────────┘
```

**TiKV** (행 저장소): 데이터를 기본 키로 정렬된 키-값 쌍으로 저장합니다. 포인트 읽기와 쓰기(OLTP)에 최적화되어 있습니다. 복제를 위해 Raft를, 저장을 위해 RocksDB를 사용합니다.

**TiFlash** (열 저장소): 동일한 데이터를 열 형식으로 저장합니다. 스캔, 집계 및 분석 쿼리(OLAP)에 최적화되어 있습니다. Raft Learner 프로토콜을 통해 데이터를 수신합니다(비동기, TiKV 쓰기 지연시간에 영향을 주지 않음).

**쿼리 라우팅**: TiDB SQL 최적화기는 쿼리의 각 부분에 대해 TiKV 또는 TiFlash를 사용할지 자동으로 결정합니다:

```sql
-- OLTP 쿼리 → TiKV로 라우팅
SELECT * FROM orders WHERE order_id = 12345;

-- OLAP 쿼리 → TiFlash로 라우팅
SELECT product_id, SUM(quantity), AVG(price)
FROM order_items
GROUP BY product_id
ORDER BY SUM(quantity) DESC
LIMIT 100;

-- 하이브리드 쿼리 → 포인트 조회는 TiKV, 집계는 TiFlash
SELECT c.name, stats.total_spent
FROM customers c
JOIN (
  SELECT customer_id, SUM(total) as total_spent
  FROM orders
  GROUP BY customer_id
  HAVING SUM(total) > 10000
) stats ON c.id = stats.customer_id
WHERE c.id = 42;
```

### 4.4 TiDB 요약

| 기능 | 세부사항 |
|---------|--------|
| **일관성** | 스냅샷 격리 (기본값), 직렬화 가능 (선택사항) |
| **시계** | 타임스탬프 오라클 (중앙 집중식, PD를 통해) |
| **복제** | Region당 Raft |
| **HTAP** | TiKV (행) + TiFlash (열) |
| **SQL** | MySQL 와이어 프로토콜 호환 |
| **저장소 엔진** | RocksDB (TiKV), ClickHouse 파생 (TiFlash) |
| **라이선스** | Apache 2.0 (오픈소스) |

---

## 5. 벡터 데이터베이스

### 5.1 동기

딥 러닝의 부상은 새로운 데이터 유형을 만들었습니다: **임베딩(embeddings)** -- 텍스트, 이미지, 오디오 또는 모든 객체의 의미론적 의미를 나타내는 고차원 벡터.

```
전통적 DB 쿼리:          벡터 DB 쿼리:
"id = 42인 사용자 찾기"  "이 이미지와 유사한 항목 찾기"
키에 대한 정확한 일치    벡터 공간에서 가장 가까운 이웃

                    ┌─ 벡터 공간 ──────────────────┐
                    │                                   │
                    │    ●(고양이 이미지)                │
                    │      ● (새끼 고양이 이미지)        │
                    │        ● (쿼리: 고양이 사진)       │
                    │                                   │
                    │                     ●(자동차 이미지)│
                    │                   ● (트럭 이미지)   │
                    │                                   │
                    │ ●(일몰 사진)                      │
                    │   ● (해변 사진)                   │
                    │                                   │
                    └───────────────────────────────────┘
```

### 5.2 임베딩

임베딩은 의미론적으로 유사한 객체가 인근 벡터를 갖도록 객체를 고정 차원 벡터로 매핑하는 학습된 표현입니다.

```
텍스트 임베딩 (예: OpenAI text-embedding-3-small, 1536 차원):
  "The cat sat on the mat"   → [0.023, -0.041, 0.108, ..., 0.055]  (1536 dims)
  "A kitten rested on a rug" → [0.021, -0.039, 0.112, ..., 0.051]  (유사!)
  "Stock prices fell today"  → [-0.087, 0.032, -0.005, ..., 0.019] (다름)

이미지 임베딩 (예: CLIP, 512 차원):
  photo_of_cat.jpg → [0.15, -0.23, ..., 0.08]   (512 dims)
  photo_of_dog.jpg → [0.12, -0.19, ..., 0.11]   (벡터 공간에서 근처)
```

### 5.3 유사도 메트릭

| 메트릭 | 공식 | 범위 | 사용 사례 |
|--------|---------|-------|----------|
| **코사인 유사도(Cosine similarity)** | `cos(A,B) = (A . B) / (|A| * |B|)` | [-1, 1] | 텍스트 유사도 (방향이 중요) |
| **유클리드 거리(Euclidean distance, L2)** | `d = sqrt(sum((a_i - b_i)^2))` | [0, inf) | 이미지 유사도, 공간 데이터 |
| **내적(Dot product)** | `A . B = sum(a_i * b_i)` | (-inf, inf) | 추천 (크기가 중요) |
| **맨해튼 거리(Manhattan distance, L1)** | `d = sum(|a_i - b_i|)` | [0, inf) | 희소 벡터 |

### 5.4 근사 최근접 이웃(Approximate Nearest Neighbor, ANN) 검색

고차원에서 정확한 최근접 이웃 검색은 계산적으로 금지됩니다 (차원의 저주). 벡터 데이터베이스는 작은 정확도 손실로 막대한 속도 향상을 교환하는 **근사** 알고리즘을 사용합니다.

**역 파일 인덱스(Inverted File Index, IVF)**:

```
IVF 알고리즘:

1. 훈련: K개의 중심으로 벡터 클러스터링 (K-means)
   ┌────────────────────────────────────┐
   │  ●  ● C1        ●  ●  ●           │
   │    ●       ●        C2  ●         │
   │                                    │
   │       ● C3                         │
   │     ●    ●                         │
   │    ●                  ● ●          │
   │                      ●  C4 ●      │
   └────────────────────────────────────┘
   C1, C2, C3, C4는 클러스터 중심

2. 인덱싱: 각 벡터를 가장 가까운 중심에 할당
   역 리스트:
   C1 → [v1, v5, v12, v33, ...]
   C2 → [v2, v7, v15, v41, ...]
   C3 → [v3, v9, v22, v37, ...]
   C4 → [v4, v11, v19, v45, ...]

3. 검색: 쿼리 q에 대해, nprobe 가장 가까운 중심을 찾고,
   그 클러스터의 벡터만 검색
   nprobe=1: 1개 클러스터 검색 (빠름, 덜 정확)
   nprobe=K: 모든 클러스터 검색 (느림, 정확)
```

**HNSW (Hierarchical Navigable Small World)**:

```
HNSW 알고리즘:

다층 그래프 구축:
- 레이어 0 (하단): 모든 벡터 포함, 조밀하게 연결
- 레이어 1: 벡터의 부분집합 포함, 더 희소한 연결
- 레이어 2: 더 적은 벡터, 더욱 희소함
- ...
- 최상위 레이어: 매우 적은 벡터, 장거리 연결

검색 탐색:

Layer 3:  A ─────────────────── B          (적은 노드, 긴 점프)
          │                     │
Layer 2:  A ──── C ──── D ──── B          (더 많은 노드)
          │      │      │      │
Layer 1:  A ─ E ─ C ─ F ─ D ─ G ─ B      (더욱 많은 노드)
          │   │   │   │   │   │   │
Layer 0:  A E H C I F J D K G L B M      (모든 노드, 짧은 연결)
                              ↑
                          쿼리 지점

검색: 최상위 레이어에서 시작, 탐욕적으로 가장 가까운 이웃으로 이동,
      다음 레이어로 내려가고, 반복.
      레이어 0에서 최종 결과를 위해 로컬 이웃 탐색.

시간 복잡도: 쿼리당 O(log N) (무차별 대입 O(N)과 비교)
```

### 5.5 벡터 데이터베이스 시스템

**Pinecone** (완전 관리형):

```python
import pinecone

# 초기화
pinecone.init(api_key="...", environment="us-east-1-aws")
index = pinecone.Index("product-search")

# 벡터 업서트
index.upsert(vectors=[
    ("prod_1", [0.1, 0.2, ..., 0.5], {"category": "electronics", "price": 99.99}),
    ("prod_2", [0.3, 0.1, ..., 0.8], {"category": "clothing", "price": 49.99}),
])

# 쿼리: 가장 유사한 5개 제품 찾기
results = index.query(
    vector=[0.15, 0.22, ..., 0.48],  # 쿼리 임베딩
    top_k=5,
    filter={"category": {"$eq": "electronics"}},  # 메타데이터 필터
    include_metadata=True
)
```

**Milvus** (오픈소스):

```python
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

# 스키마 정의
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
]
schema = CollectionSchema(fields)
collection = Collection("documents", schema)

# HNSW 인덱스 생성
index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 200}
}
collection.create_index("embedding", index_params)

# 검색
results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 100}},
    limit=10,
    output_fields=["text"]
)
```

**pgvector** (PostgreSQL 확장):

```sql
-- 확장 활성화
CREATE EXTENSION vector;

-- 벡터 열로 테이블 생성
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding vector(1536)  -- 1536차원 벡터
);

-- HNSW 인덱스 생성
CREATE INDEX ON documents
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 200);

-- 삽입
INSERT INTO documents (content, embedding)
VALUES ('The cat sat on the mat', '[0.023, -0.041, ...]');

-- 유사도 검색 (코사인 거리)
SELECT content, 1 - (embedding <=> '[0.025, -0.038, ...]') AS similarity
FROM documents
ORDER BY embedding <=> '[0.025, -0.038, ...]'
LIMIT 5;

-- 하이브리드 검색: 벡터 유사도 + 메타데이터 필터
SELECT content, 1 - (embedding <=> $1) AS similarity
FROM documents
WHERE category = 'science'
ORDER BY embedding <=> $1
LIMIT 10;
```

### 5.6 벡터 데이터베이스 사용 사례

| 사용 사례 | 설명 |
|----------|-------------|
| **의미론적 검색(Semantic search)** | 키워드가 아닌 의미로 문서 찾기 |
| **RAG (Retrieval-Augmented Generation)** | 지식 베이스 임베딩 저장; LLM 프롬프트를 위한 컨텍스트 검색 |
| **추천 시스템(Recommendation systems)** | 사용자 선호도와 유사한 항목 찾기 (임베딩을 통한 협업 필터링) |
| **이미지 검색(Image search)** | 시각적으로 유사한 이미지 찾기 |
| **이상 탐지(Anomaly detection)** | 정상 클러스터에서 멀리 떨어진 벡터 식별 |
| **중복 제거(Deduplication)** | 거의 중복된 문서 또는 이미지 찾기 |
| **멀티모달 검색(Multimodal search)** | 공유 임베딩 공간(CLIP)을 사용하여 텍스트, 이미지 및 오디오에서 검색 |

### 5.7 벡터 데이터베이스 선택

| 시스템 | 유형 | 호스팅 | 강점 | 제한사항 |
|--------|------|---------|-----------|-------------|
| **Pinecone** | 관리형 | 클라우드만 | 간단한 API, 자동 확장 | 벤더 종속, 비용 |
| **Milvus** | 오픈소스 | 자체 호스팅 / Zilliz Cloud | 기능 풍부, GPU 지원 | 운영 복잡성 |
| **Weaviate** | 오픈소스 | 자체 호스팅 / Cloud | GraphQL API, 모듈 | 최신, 작은 생태계 |
| **Qdrant** | 오픈소스 | 자체 호스팅 / Cloud | Rust 기반 (빠름), 필터링 | 작은 커뮤니티 |
| **pgvector** | 확장 | 모든 PostgreSQL | 새로운 인프라 없음, 완전한 SQL | 제한된 규모, GPU 없음 |
| **Chroma** | 오픈소스 | 임베디드 / 서버 | 간단, Python 네이티브 | 제한된 프로덕션 기능 |

---

## 6. 시계열 데이터베이스

### 6.1 시계열 데이터란?

시계열 데이터는 시간으로 인덱싱된 측정값 또는 이벤트로 구성됩니다. 범용 데이터베이스가 잘 처리하지 못하는 고유한 특성을 가지고 있습니다.

```
시계열 데이터 예:

IoT 센서 판독값:
  timestamp            | device_id | temperature | humidity
  2024-11-15T10:00:00  | sensor_42 | 23.5        | 65.2
  2024-11-15T10:00:01  | sensor_42 | 23.6        | 65.1
  2024-11-15T10:00:02  | sensor_42 | 23.5        | 65.3
  ...

애플리케이션 메트릭:
  timestamp            | service   | metric     | value
  2024-11-15T10:00:00  | api-gw    | latency_ms | 42
  2024-11-15T10:00:00  | api-gw    | req_count  | 1547
  2024-11-15T10:00:00  | api-gw    | error_rate | 0.02
  ...
```

### 6.2 시계열 데이터의 특성

| 특성 | 의미 |
|---------------|-------------|
| **쓰기 중심(Write-heavy)** | 지속적인 수집, 쓰기 후 거의 업데이트되지 않음 |
| **시간 순서(Time-ordered)** | 데이터는 자연스럽게 타임스탬프로 정렬됨; 대부분의 쿼리는 시간 범위 포함 |
| **추가 전용(Append-only)** | 새 데이터는 항상 최신; 오래된 데이터는 거의 수정되지 않음 |
| **높은 카디널리티(High cardinality)** | 수백만 개의 고유 시리즈 (device_id x 메트릭 조합) |
| **다운샘플링(Downsampling)** | 오래된 데이터는 집계 가능 (1초 → 1분 → 1시간) |
| **TTL (만료)** | 오래된 데이터는 가치가 감소함; 보존 기간 이후 자동 삭제 |
| **압축(Compression)** | 시간적 지역성으로 높은 압축률 가능 (10-20배) |

### 6.3 TSDB 최적화

**시간 기반 파티셔닝**: 데이터는 시간으로 자동 파티션되어 효율적인 시간 범위 쿼리와 데이터 라이프사이클 관리를 가능하게 합니다.

```
┌─────────────────────────────────────────────────────┐
│  시간 기반 파티셔닝                                  │
│                                                     │
│  Chunk 1        Chunk 2        Chunk 3        ...   │
│  [Nov 1-7]     [Nov 8-14]    [Nov 15-21]          │
│  ┌──────┐      ┌──────┐      ┌──────┐              │
│  │ Data │      │ Data │      │ Data │              │
│  │ Index│      │ Index│      │ Index│              │
│  └──────┘      └──────┘      └──────┘              │
│                                                     │
│  쿼리: WHERE time > 'Nov 10' AND time < 'Nov 20'   │
│  → Chunk 2와 Chunk 3만 스캔!                        │
│                                                     │
│  보존: DROP Chunk 1 (즉시, 행별 없음)               │
└─────────────────────────────────────────────────────┘
```

**델타의 델타 압축(Delta-of-delta compression)**: 연속 타임스탬프는 유사합니다 (예: 매 10초). 델타 사이의 델타를 저장합니다.

```
원시 타임스탬프:     1700000000, 1700000010, 1700000020, 1700000030
델타:                          10,          10,          10
델타의 델타:                                0,           0

압축: 기준 + 첫 델타 + 델타의 델타 저장
  1700000000, 10, 0, 0  (4개의 전체 타임스탬프 대신 4개의 값)

값의 경우: Gorilla 압축 (XOR 기반, Facebook의 Gorilla 논문에서)
  연속 값이 유사하면 XOR이 대부분 0 → 잘 압축됨
```

**다운샘플링**: 오래된 데이터를 자동으로 집계하여 저장 공간 감소.

```
원시 데이터 (1초 해상도):     → 86,400 포인트/일/시리즈
7일 후: 1분으로 다운샘플       → 1,440 포인트/일/시리즈
30일 후: 1시간으로 다운샘플    → 24 포인트/일/시리즈
1년 후: 1일로 다운샘플         → 1 포인트/일/시리즈

저장 공간 감소: 1년에 걸쳐 86,400배!
```

### 6.4 TimescaleDB

TimescaleDB는 PostgreSQL을 위한 시계열 확장으로, 완전한 SQL 기능을 보존합니다.

```sql
-- 하이퍼테이블 생성 (시간으로 자동 파티션)
CREATE TABLE metrics (
  time TIMESTAMPTZ NOT NULL,
  device_id TEXT NOT NULL,
  temperature DOUBLE PRECISION,
  humidity DOUBLE PRECISION
);

SELECT create_hypertable('metrics', 'time');

-- 시간별 자동 파티셔닝 (청크)
-- 각 청크는 별도의 PostgreSQL 테이블

-- 연속 집계 (구체화, 자동 새로고침)
CREATE MATERIALIZED VIEW hourly_stats
WITH (timescaledb.continuous) AS
SELECT
  time_bucket('1 hour', time) AS bucket,
  device_id,
  AVG(temperature) AS avg_temp,
  MAX(temperature) AS max_temp,
  MIN(temperature) AS min_temp
FROM metrics
GROUP BY bucket, device_id;

-- 보존 정책 (오래된 데이터 자동 삭제)
SELECT add_retention_policy('metrics', INTERVAL '90 days');

-- 압축 정책
ALTER TABLE metrics SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'device_id',
  timescaledb.compress_orderby = 'time DESC'
);
SELECT add_compression_policy('metrics', INTERVAL '7 days');

-- 쿼리: 지난 24시간, 5분 평균
SELECT
  time_bucket('5 minutes', time) AS bucket,
  device_id,
  AVG(temperature) AS avg_temp
FROM metrics
WHERE time > NOW() - INTERVAL '24 hours'
  AND device_id = 'sensor_42'
GROUP BY bucket, device_id
ORDER BY bucket;
```

### 6.5 InfluxDB

InfluxDB는 자체 쿼리 언어(Flux)와 라인 프로토콜을 갖춘 목적 구축 시계열 데이터베이스입니다.

```
# InfluxDB Line Protocol (쓰기)
cpu,host=server01,region=us-east usage=0.72,system=0.15 1700000000000000000
cpu,host=server02,region=eu-west usage=0.45,system=0.08 1700000000000000000

# 구조: measurement,tag_key=tag_value field_key=field_value timestamp

# Flux 쿼리: 호스트별 평균 CPU 사용률, 지난 시간, 5분 윈도우
from(bucket: "monitoring")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "cpu" and r._field == "usage")
  |> aggregateWindow(every: 5m, fn: mean)
  |> group(columns: ["host"])
```

### 6.6 TSDB 비교

| 기능 | TimescaleDB | InfluxDB | Prometheus |
|---------|-------------|----------|------------|
| **기반** | PostgreSQL 확장 | 목적 구축 | 목적 구축 |
| **쿼리 언어** | 완전한 SQL | Flux / InfluxQL | PromQL |
| **압축** | 열 압축 | TSM 엔진 | Gorilla + delta |
| **클러스터링** | 자체 관리 또는 클라우드 | 엔터프라이즈만 | Thanos/Cortex |
| **카디널리티** | 좋음 (PostgreSQL 인덱스) | 제한적 (높은 카디널리티 = 느림) | 제한적 |
| **조인** | 완전한 SQL 조인 | 제한적 | 없음 |
| **최적** | 일반 시계열 + SQL | 메트릭, IoT | 인프라 모니터링 |

---

## 7. 대규모 그래프 분석

### 7.1 그래프 데이터베이스를 넘어

Neo4j와 같은 그래프 데이터베이스는 트랜잭션 그래프 쿼리(OLTP)에서 뛰어나지만, 그래프 분석은 전체 그래프를 처리해야 합니다(OLAP): PageRank, 커뮤니티 탐지, 수십억 개의 에지에 걸친 최단 경로.

### 7.2 Apache Spark GraphX

GraphX는 Spark의 그래프 병렬 계산을 위한 API입니다. RDD (Resilient Distributed Dataset) 추상화를 Resilient Distributed Property Graph로 확장합니다.

```python
from pyspark.sql import SparkSession
from graphframes import GraphFrame

spark = SparkSession.builder.appName("GraphAnalytics").getOrCreate()

# 정점 DataFrame
vertices = spark.createDataFrame([
    ("1", "Alice", 30),
    ("2", "Bob", 25),
    ("3", "Charlie", 35),
    ("4", "Diana", 28),
], ["id", "name", "age"])

# 에지 DataFrame
edges = spark.createDataFrame([
    ("1", "2", "follows"),
    ("2", "3", "follows"),
    ("3", "1", "follows"),
    ("1", "4", "follows"),
    ("4", "3", "follows"),
], ["src", "dst", "relationship"])

# 그래프 생성
g = GraphFrame(vertices, edges)

# PageRank
pagerank = g.pageRank(resetProbability=0.15, maxIter=10)
pagerank.vertices.select("id", "name", "pagerank").show()

# 연결된 구성요소
cc = g.connectedComponents()
cc.show()

# 최단 경로
sp = g.shortestPaths(landmarks=["3"])
sp.show()

# 삼각형 개수
tc = g.triangleCount()
tc.show()
```

### 7.3 TigerGraph

TigerGraph는 심층 링크 분석(대규모 다중 홉 쿼리)을 위해 설계된 분산 그래프 분석 플랫폼입니다.

```gsql
// GSQL: TigerGraph의 쿼리 언어

// 스키마 정의
CREATE VERTEX Person (PRIMARY_ID id STRING, name STRING, age INT)
CREATE VERTEX Company (PRIMARY_ID id STRING, name STRING)
CREATE DIRECTED EDGE works_at (FROM Person, TO Company, since DATETIME)
CREATE UNDIRECTED EDGE knows (FROM Person, TO Person, strength FLOAT)

// 쿼리 설치: 2-홉 사기 탐지
CREATE QUERY fraud_ring_detection(VERTEX<Person> seed) FOR GRAPH social {
  // 금융 연결을 공유하는 2홉 내의 모든 사람 찾기
  Start = {seed};

  // 1-홉: 직접 연결
  hop1 = SELECT t FROM Start:s -(knows:e)- Person:t
         WHERE e.strength > 0.8;

  // 2-홉: 연결의 연결
  hop2 = SELECT t FROM hop1:s -(knows:e)- Person:t
         WHERE e.strength > 0.8 AND t != seed;

  // 공유된 금융 패턴 찾기
  suspicious = SELECT t FROM hop2:t -(works_at)- Company:c
               WHERE c.name == "shell_company_pattern";

  PRINT suspicious;
}
```

### 7.4 그래프 분석 비교

| 기능 | Neo4j | Spark GraphX | TigerGraph |
|---------|-------|-------------|------------|
| **유형** | 그래프 DB + 분석 | 분산 계산 | 분산 그래프 DB |
| **규모** | 수십억 개의 노드 | 수십억 개의 에지 | 수조 개의 에지 |
| **쿼리** | Cypher | Scala/Python API | GSQL |
| **실시간** | 예 (OLTP + OLAP) | 아니오 (배치) | 예 |
| **최적** | 혼합 OLTP/OLAP 그래프 | Spark에서 배치 분석 | 심층 링크 분석 |

---

## 8. Database-as-a-Service

### 8.1 서버리스 데이터베이스 트렌드

Database-as-a-Service (DBaaS)는 서버 프로비저닝, 확장, 패치 및 백업 관리를 추상화합니다. 최신 DBaaS 제공 서비스는 사용하지 않을 때 0으로 확장되는 **서버리스** 아키텍처로 더 나아갑니다.

### 8.2 Neon

Neon은 저장소와 계산의 분리를 통해 즉각적인 브랜칭과 0으로의 확장을 가능하게 하는 서버리스 PostgreSQL입니다.

```
Neon 아키텍처:

┌────────────────────────────────────────────────────┐
│  Compute Layer (stateless, scale-to-zero)           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ Postgres │  │ Postgres │  │ Postgres │  (auto) │
│  │ Instance │  │ Instance │  │  (idle)  │         │
│  └────┬─────┘  └────┬─────┘  └──────────┘         │
│       │              │                              │
│       └──────────────┘                              │
│                │                                    │
│  ┌─────────────┴──────────────────────────────────┐ │
│  │  Pageserver (page cache + WAL processing)       │ │
│  └─────────────┬──────────────────────────────────┘ │
│                │                                    │
│  ┌─────────────┴──────────────────────────────────┐ │
│  │  SafeKeeper (WAL durability, consensus)         │ │
│  └─────────────┬──────────────────────────────────┘ │
│                │                                    │
│  ┌─────────────┴──────────────────────────────────┐ │
│  │  Object Storage (S3) - bottomless storage       │ │
│  └────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
```

**주요 기능**:
- **0으로 확장**: 쿼리가 실행되지 않을 때 계산이 0으로 확장됨 (비용 없음)
- **즉각적인 브랜칭**: 전체 데이터베이스의 copy-on-write 브랜치 생성 (데이터를 위한 git 브랜치처럼)
- **무한 저장소**: S3에 저장된 데이터, 상한 없음

```bash
# 브랜치 생성 (즉각적, copy-on-write)
neonctl branches create --name staging --parent main

# 각 브랜치는 자체 계산 엔드포인트를 가짐
# 유용한 경우: 개발, 테스트, 미리보기
```

### 8.3 PlanetScale

PlanetScale은 Vitess (YouTube의 MySQL 샤딩 미들웨어)를 기반으로 구축된 서버리스 MySQL 플랫폼입니다.

**주요 기능**:
- **비차단 스키마 변경**: 스키마 변경이 백그라운드 작업으로 실행됨, 테이블 잠금 없음
- **데이터베이스 브랜칭**: 스키마 변경을 위한 브랜치 생성, 배포 요청을 통한 병합
- **샤딩**: Vitess 기반, 자동 수평 샤딩

```bash
# 스키마 변경을 위한 Git 유사 워크플로우
pscale branch create main add-column
pscale shell main add-column

# 브랜치에서:
mysql> ALTER TABLE users ADD COLUMN avatar_url VARCHAR(500);

# 배포 요청 생성 (풀 리퀘스트처럼)
pscale deploy-request create main add-column

# 검토 및 병합
pscale deploy-request deploy main 1
# 다운타임 없이 프로덕션에 스키마 변경 적용!
```

### 8.4 Supabase

Supabase는 PostgreSQL 기반으로 구축된 오픈소스 Firebase 대안입니다.

```
Supabase 스택:
┌──────────────────────────────────────────────┐
│  Dashboard / Studio (Admin UI)                │
│  ┌─────────────────────────────────────────┐  │
│  │  Auth (GoTrue)                          │  │
│  │  Realtime (WebSocket subscriptions)     │  │
│  │  Storage (S3-compatible)                │  │
│  │  Edge Functions (Deno runtime)          │  │
│  │  PostgREST (auto-generated REST API)    │  │
│  │  pgvector (vector search)               │  │
│  └─────────────────────────────────────────┘  │
│                    │                          │
│           ┌────────┴────────┐                 │
│           │   PostgreSQL    │                 │
│           │  (the source    │                 │
│           │   of truth)     │                 │
│           └─────────────────┘                 │
└──────────────────────────────────────────────┘
```

```javascript
// Supabase 클라이언트 (PostgreSQL 스키마에서 자동 생성된 API)
import { createClient } from '@supabase/supabase-js'

const supabase = createClient('https://xxx.supabase.co', 'anon-key')

// 쿼리 (자동으로 SQL로 변환)
const { data, error } = await supabase
  .from('products')
  .select('name, price, categories(name)')
  .gte('price', 10)
  .order('price', { ascending: true })
  .limit(20)

// 실시간 구독
const subscription = supabase
  .channel('orders')
  .on('postgres_changes',
    { event: 'INSERT', schema: 'public', table: 'orders' },
    (payload) => console.log('New order:', payload.new)
  )
  .subscribe()

// 행 수준 보안 (RLS) - 데이터베이스 수준의 보안
// 사용자는 자신의 주문만 볼 수 있음
// CREATE POLICY "Users see own orders" ON orders
//   FOR SELECT USING (auth.uid() = user_id);
```

### 8.5 DBaaS 비교

| 기능 | Neon | PlanetScale | Supabase |
|---------|------|-------------|----------|
| **엔진** | PostgreSQL | MySQL (Vitess) | PostgreSQL |
| **서버리스** | 예 (0으로 확장) | 예 | 부분적 |
| **브랜칭** | 데이터베이스 브랜칭 | 스키마 브랜칭 | 아니오 |
| **샤딩** | 아니오 (단일 노드) | 예 (Vitess) | 아니오 |
| **자동 생성 API** | 아니오 | 아니오 | 예 (PostgREST) |
| **실시간** | 아니오 | 아니오 | 예 (WebSocket) |
| **벡터 검색** | pgvector | 아니오 | pgvector |
| **오픈소스** | 예 (저장소) | 아니오 (Vitess는 OSS) | 예 |
| **최적** | 개발/테스트, 브랜칭 | 대규모 MySQL | 풀스택 앱 |

---

## 9. 데이터 레이크하우스

### 9.1 데이터 아키텍처의 진화

```
시대 1: 데이터 웨어하우스 (1990년대-2010년대)
  구조화된 데이터 → ETL → 데이터 웨어하우스 (Redshift, Snowflake)
  + 강한 스키마, ACID, SQL
  - 비싸고, 구조화된 데이터만 지원

시대 2: 데이터 레이크 (2010년대)
  모든 데이터 → HDFS/S3에 덤프 → Spark로 처리
  + 저렴한 저장소, 모든 데이터 유형
  - ACID 없음, 스키마 강제 없음, "데이터 늪"

시대 3: 데이터 레이크하우스 (2020년대)
  모든 데이터 → 레이크 저장소 (S3) + 테이블 형식 (Delta/Iceberg)
  + 저렴한 저장소 + ACID + 스키마 + SQL
  두 세계의 장점!
```

### 9.2 레이크하우스란?

데이터 레이크하우스는 데이터 레이크의 저비용 저장소(S3와 같은 객체 저장소)와 데이터 웨어하우스의 데이터 관리 기능(ACID 트랜잭션, 스키마 강제, 인덱싱)을 결합합니다.

```
데이터 레이크하우스 아키텍처:

┌──────────────────────────────────────────────────────┐
│  Query Engines                                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐              │
│  │  Spark  │  │  Trino  │  │ DuckDB  │  ...         │
│  └────┬────┘  └────┬────┘  └────┬────┘              │
│       │            │            │                    │
│       └────────────┼────────────┘                    │
│                    │                                  │
│  ┌─────────────────┴────────────────────────────────┐ │
│  │  Table Format (metadata layer)                    │ │
│  │  ┌────────────┐  ┌────────────┐                   │ │
│  │  │ Delta Lake │  │  Apache    │                   │ │
│  │  │            │  │  Iceberg   │                   │ │
│  │  └────────────┘  └────────────┘                   │ │
│  │  제공: ACID, 스키마 진화, 시간 여행,              │ │
│  │  파티션 가지치기, 파일 수준 통계                  │ │
│  └──────────────────────────────────────────────────┘ │
│                    │                                  │
│  ┌─────────────────┴────────────────────────────────┐ │
│  │  Object Storage                                   │ │
│  │  ┌──────┐  ┌──────┐  ┌──────┐                    │ │
│  │  │  S3  │  │ GCS  │  │ ADLS │                    │ │
│  │  └──────┘  └──────┘  └──────┘                    │ │
│  │  데이터는 Parquet/ORC 파일로 저장                 │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

### 9.3 Delta Lake

Databricks에서 만든 Delta Lake는 데이터 레이크의 Apache Spark에 ACID 트랜잭션을 추가합니다.

**트랜잭션 로그**: Delta Lake는 테이블에 대한 모든 변경을 기록하는 트랜잭션 로그(`_delta_log/`)를 JSON으로 유지합니다. 이는 다음을 제공합니다:

```
my_table/
├── _delta_log/
│   ├── 00000000000000000000.json   ← 초기 테이블 생성
│   ├── 00000000000000000001.json   ← INSERT 1000 행
│   ├── 00000000000000000002.json   ← UPDATE 일부 행
│   └── 00000000000000000003.json   ← DELETE 오래된 행
├── part-00000-xxxx.parquet
├── part-00001-xxxx.parquet
└── part-00002-xxxx.parquet
```

```python
from delta import DeltaTable
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
    .getOrCreate()

# 데이터 쓰기 (ACID)
df.write.format("delta").mode("overwrite").save("/data/events")

# 시간 여행: 특정 버전으로 데이터 읽기
df_v1 = spark.read.format("delta") \
    .option("versionAsOf", 1) \
    .load("/data/events")

# 시간 여행: 특정 타임스탬프로 데이터 읽기
df_yesterday = spark.read.format("delta") \
    .option("timestampAsOf", "2024-11-14") \
    .load("/data/events")

# MERGE (upsert): 기존 행 업데이트, 새 행 삽입
deltaTable = DeltaTable.forPath(spark, "/data/events")
deltaTable.alias("target").merge(
    new_data.alias("source"),
    "target.id = source.id"
).whenMatchedUpdateAll() \
 .whenNotMatchedInsertAll() \
 .execute()

# 스키마 진화
df_new_columns.write.format("delta") \
    .mode("append") \
    .option("mergeSchema", "true") \
    .save("/data/events")
```

### 9.4 Apache Iceberg

Apache Iceberg는 대규모 분석 테이블을 위한 오픈 테이블 형식으로, Hive 테이블의 제한사항을 해결하도록 설계되었습니다.

**Delta Lake보다 주요 이점**:

| 기능 | Delta Lake | Apache Iceberg |
|---------|------------|----------------|
| **엔진 종속** | Spark에 최적화 | 엔진 독립적 (Spark, Trino, Flink, Dremio) |
| **숨겨진 파티셔닝** | 명시적 파티션 열 | 자동 (재작성 없이 파티션 진화) |
| **메타데이터** | JSON 로그 + 체크포인트 | 매니페스트 파일 + 스냅샷 메타데이터 |
| **스키마 진화** | 열 추가/이름변경/삭제 | 중첩 타입을 포함한 완전한 진화 |
| **거버넌스** | Databricks 주도 | Apache Foundation (벤더 중립) |

```sql
-- Spark SQL의 Iceberg
CREATE TABLE catalog.db.events (
  event_id BIGINT,
  user_id BIGINT,
  event_type STRING,
  event_time TIMESTAMP,
  properties MAP<STRING, STRING>
) USING iceberg
PARTITIONED BY (days(event_time));  -- 일별 숨겨진 파티셔닝

-- 시간 여행
SELECT * FROM catalog.db.events VERSION AS OF 12345;
SELECT * FROM catalog.db.events TIMESTAMP AS OF '2024-11-14 00:00:00';

-- 스냅샷 관리
SELECT * FROM catalog.db.events.snapshots;

-- 오래된 스냅샷 만료 (저장 공간 회수)
CALL catalog.system.expire_snapshots('db.events', TIMESTAMP '2024-10-01');
```

### 9.5 레이크하우스 vs 전통적 아키텍처

| 기능 | 데이터 웨어하우스 | 데이터 레이크 | 데이터 레이크하우스 |
|---------|---------------|-----------|----------------|
| **저장 비용** | 높음 | 낮음 | 낮음 |
| **데이터 유형** | 구조화만 | 모든 유형 | 모든 유형 |
| **ACID** | 예 | 아니오 | 예 |
| **스키마** | 쓰기 시 스키마 | 읽기 시 스키마 | 쓰기 시 스키마 (유연) |
| **쿼리 성능** | 우수 | 가변적 | 좋음 (인덱싱 포함) |
| **시간 여행** | 제한적 | 아니오 | 예 |
| **거버넌스** | 강함 | 약함 | 강함 |
| **ML 지원** | 제한적 | 좋음 (원시 데이터) | 좋음 (원시 + 구조화) |

---

## 10. 연습 문제

### 연습 문제 1: NewSQL 비교

다음 차원에 걸쳐 Google Spanner, CockroachDB 및 TiDB를 비교하세요. 표를 작성하세요:

| 차원 | Spanner | CockroachDB | TiDB |
|-----------|---------|-------------|------|
| 시계 메커니즘 | | | |
| 기본 격리 수준 | | | |
| 복제 프로토콜 | | | |
| SQL 호환성 | | | |
| HTAP 지원 | | | |
| 오픈소스? | | | |
| 일반적인 배포 | | | |

### 연습 문제 2: TrueTime과 커밋 대기

Google Spanner의 TrueTime은 `[T-ε, T+ε]`의 시계 불확실성을 보고하며, 여기서 ε는 일반적으로 1-7ms입니다.

1. Spanner가 트랜잭션을 커밋하기 전에 불확실성을 "대기"해야 하는 이유를 설명하세요.
2. ε = 5ms인 경우, 쓰기 트랜잭션의 최소 커밋 지연시간은 무엇입니까?
3. ε가 0 (완벽한 시계)이라면, 프로토콜이 어떻게 단순화될까요?
4. CockroachDB가 동일한 접근 방식을 사용할 수 없는 이유는? 대신 무엇을 합니까?
5. CockroachDB의 HLC 접근 방식이 "읽기 재시작"을 초래할 수 있는 시나리오를 설명하세요. 사용자에게 보이는 영향은 무엇입니까?

### 연습 문제 3: 벡터 데이터베이스 설계

고객 지원 챗봇을 위한 RAG (Retrieval-Augmented Generation) 시스템을 구축하고 있습니다. 지식 베이스에는 다음이 포함됩니다:
- 100,000개의 지원 문서 (평균 500단어씩)
- 문서는 주별로 업데이트됨
- 예상 쿼리 부하: 분당 1,000개 쿼리
- 지연시간 요구사항: 쿼리당 < 100ms

벡터 데이터베이스 구성요소를 설계하세요:
1. 벡터 데이터베이스 시스템을 선택하고 선택을 정당화하세요.
2. 임베딩 모델과 차원 크기를 선택하세요.
3. 인덱싱 전략을 설명하세요 (IVF, HNSW 또는 하이브리드).
4. 문서 업데이트를 어떻게 처리하시겠습니까? (전체 문서를 다시 임베딩? 청크 수준 업데이트?)
5. 하이브리드 검색(벡터 유사도 + 키워드 매칭)을 어떻게 구현하시겠습니까?

### 연습 문제 4: 시계열 스키마 설계

스마트 빌딩 모니터링 시스템을 위한 TimescaleDB 스키마를 설계하세요:
- 50개 층에 걸쳐 500개의 센서
- 각 센서는 10초마다 온도, 습도, CO2 및 점유를 보고
- 일반적인 쿼리: 특정 층의 지난 24시간, 층당 시간당 평균, 이상 탐지
- 데이터 보존: 90일간 원시 데이터, 2년간 시간당 집계

다음에 대한 SQL을 작성하세요:
1. 하이퍼테이블로 테이블 생성
2. 시간당 평균을 위한 연속 집계
3. 보존 정책
4. 압축 정책
5. "지난 6시간 동안 평균 온도가 28C를 초과한 층" 쿼리

### 연습 문제 5: HTAP 시나리오 분석

금융 거래 회사는 현재 다음을 실행합니다:
- 거래 실행을 위한 OLTP 데이터베이스 (PostgreSQL)
- 위험 분석을 위한 OLAP 웨어하우스 (Snowflake) 2시간 ETL 지연

TiDB로 HTAP를 위해 마이그레이션하는 것을 고려하고 있습니다.

1. ETL 파이프라인을 제거하는 것의 이점은 무엇입니까?
2. 트랜잭션과 동일한 시스템에서 분석을 실행하면 어떤 위험이 도입됩니까?
3. TiDB의 아키텍처 (TiKV + TiFlash)가 리소스 경합 위험을 어떻게 완화합니까?
4. 어떤 시나리오에서 여전히 별도의 OLTP 및 OLAP 시스템을 유지하는 것을 권장하시겠습니까?

### 연습 문제 6: 데이터베이스 선택

다음 각 애플리케이션에 대해, 이 레슨에서 다룬 옵션 중 가장 적합한 데이터베이스 기술을 선택하세요. 선택을 정당화하세요.

1. 100억 개의 시퀀스에 걸쳐 유사도 검색을 위한 DNA 시퀀스 임베딩을 저장하는 유전체학 연구 플랫폼.
2. 10,000대의 기계가 있는 공장 모니터링 시스템, 각각 초당 50개의 메트릭을 보고.
3. 30개국에서 운영되는 ACID 트랜잭션이 필요한 글로벌 뱅킹 애플리케이션.
4. 10,000명의 사용자가 있는 협업 문서 편집기 (Google Docs처럼) 구축 중인 스타트업.
5. S3에 저장된 100TB의 이벤트 데이터에 대해 SQL 쿼리를 실행해야 하는 데이터 분석 플랫폼.

### 연습 문제 7: 레이크하우스 설계

차량 공유 회사를 위한 데이터 플랫폼을 설계하고 있습니다. 데이터 소스는 다음을 포함합니다:
- 차량 이벤트 (일 1,000만 건)
- GPS 추적 (차량당 100 포인트)
- 결제 트랜잭션
- 운전자/승객 프로필
- 급증 가격 계산

레이크하우스 아키텍처를 설계하세요:
1. Delta Lake와 Apache Iceberg 중 선택하세요. 정당화하세요.
2. 가장 중요한 세 테이블의 테이블 스키마를 정의하세요.
3. 각 테이블의 파티셔닝 전략을 설명하세요.
4. 결제 트랜잭션 감사를 위한 시간 여행을 어떻게 구현하시겠습니까?
5. (a) 실시간 대시보드, (b) 월간 보고서, (c) ML 피처 엔지니어링에 어떤 쿼리 엔진을 사용하시겠습니까?

### 연습 문제 8: 서버리스 데이터베이스 평가

다음 특성을 가진 SaaS 애플리케이션을 구축하고 있습니다:
- 다중 테넌트 (각 테넌트는 격리된 데이터를 가짐)
- 가변 부하: 평균의 10배 피크
- 개발 팀은 빈번한 스키마 변경이 필요
- 예산이 제한됨 (스타트업)

이 사용 사례에 대해 Neon, PlanetScale 및 Supabase를 평가하세요:
1. 각각 다중 테넌시를 어떻게 처리합니까?
2. 각각 가변 부하를 어떻게 처리합니까?
3. 각각 프로덕션에서 스키마 변경을 어떻게 처리합니까?
4. 귀하의 권장사항은 무엇이며 그 이유는?

### 연습 문제 9: CockroachDB의 일관된 해싱

CockroachDB는 키 공간을 Range (각각 약 512MB)로 나눕니다. 각 Range는 Raft 그룹입니다.

1. Range가 512MB를 초과하여 성장하면 CockroachDB가 어떻게 분할합니까? Raft 그룹에는 무슨 일이 일어납니까?
2. 노드가 클러스터에 합류하면 Range가 어떻게 재균형됩니까?
3. Range 분할과 Range 재균형의 차이는 무엇입니까?
4. CockroachDB는 Range의 Raft 리더가 어느 노드여야 하는지 어떻게 결정합니까? 이것이 지리적 파티셔닝과 어떻게 관련됩니까?

### 연습 문제 10: 에세이 질문

다음 주제에 대해 600단어 에세이를 작성하세요:

"관계형 데이터베이스는 죽었다"는 여러 번 선언되었지만, PostgreSQL 사용은 해마다 계속 증가하고 있습니다. 이 레슨의 자료 (NewSQL, 벡터 데이터베이스, 시계열 데이터베이스, 레이크하우스)를 기반으로, 관계형 모델이 점점 더 관련성이 있는지 또는 덜 관련성이 있는지 논쟁하세요. 관계형 데이터베이스가 어떻게 적응하고 있는지 (예: pgvector, 확장으로서의 TimescaleDB)의 구체적인 예와 특수 시스템에 의해 대체되는 예를 들어 귀하의 주장을 뒷받침하세요.

---

## 11. 참고문헌

1. Corbett, J. et al. (2013). "Spanner: Google's Globally-Distributed Database." ACM TODS.
2. Taft, R. et al. (2020). "CockroachDB: The Resilient Geo-Distributed SQL Database." ACM SIGMOD.
3. Huang, D. et al. (2020). "TiDB: A Raft-based HTAP Database." VLDB.
4. Aslett, M. (2011). "How Will the Database Incumbents Respond to NoSQL and NewSQL?" The 451 Group.
5. Johnson, J., Douze, M., Jegou, H. (2019). "Billion-Scale Similarity Search with GPUs." IEEE Trans. on Big Data.
6. Malkov, Y. & Yashunin, D. (2018). "Efficient and Robust Approximate Nearest Neighbor using Hierarchical Navigable Small World Graphs." IEEE TPAMI.
7. Armbrust, M. et al. (2021). "Lakehouse: A New Generation of Open Platforms that Unify Data Warehousing and Advanced Analytics." CIDR.
8. Armbrust, M. et al. (2020). "Delta Lake: High-Performance ACID Table Storage over Cloud Object Stores." VLDB.
9. Apache Iceberg Documentation. https://iceberg.apache.org/
10. Pelkonen, T. et al. (2015). "Gorilla: A Fast, Scalable, In-Memory Time Series Database." VLDB.
11. Neon Documentation. https://neon.tech/docs
12. CockroachDB Architecture Documentation. https://www.cockroachlabs.com/docs/stable/architecture/overview.html

---

**이전**: [14. 분산 데이터베이스](./14_Distributed_Databases.md) | **다음**: [16. 데이터베이스 설계 사례 연구](./16_Database_Design_Case_Study.md)
