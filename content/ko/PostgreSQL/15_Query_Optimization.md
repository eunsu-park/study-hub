# 15. PostgreSQL 쿼리 최적화 심화

**이전**: [JSON과 JSONB](./14_JSON_JSONB.md) | **다음**: [복제와 고가용성](./16_Replication_HA.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. EXPLAIN ANALYZE 출력에서 비용 추정치, 실제 실행 시간, 버퍼 통계, 반복 횟수를 읽고 해석할 수 있다
2. PostgreSQL 쿼리 플래너(Query Planner)가 SQL을 실행 계획(Execution Plan)으로 변환하는 과정을 설명할 수 있다
3. 쿼리 패턴에 따라 적절한 인덱스 타입(B-tree, Hash, GIN, GiST, BRIN)을 선택할 수 있다
4. 복합 인덱스, 부분 인덱스, 커버링 인덱스를 효과적으로 설계할 수 있다
5. Nested Loop, Hash Join, Merge Join 알고리즘을 비교하고 각각이 선택되는 조건을 예측할 수 있다
6. 테이블 통계와 비용 파라미터를 활용하여 플래너 결정을 이해하고 영향을 줄 수 있다
7. 쿼리 리팩토링, 구체화된 뷰(Materialized View), 파티셔닝 등 고급 최적화 기법을 적용할 수 있다

---

5밀리초와 5초의 차이는 반응하는 애플리케이션과 답답한 사용자 경험의 차이입니다. PostgreSQL의 쿼리 옵티마이저(Query Optimizer)는 매우 정교하지만, 최선의 성능을 발휘하려면 정확한 통계, 잘 선택된 인덱스, 적절히 구조화된 쿼리가 필요합니다. 플래너가 어떻게 생각하는지, 그리고 플래너가 생성하는 실행 계획을 어떻게 읽는지 이해하면 추측에 의존하지 않고 체계적으로 성능 병목을 진단하고 해결할 수 있습니다.

## 목차

1. [EXPLAIN ANALYZE 심화](#1-explain-analyze-심화)
2. [쿼리 플래너](#2-쿼리-플래너)
3. [인덱스 전략](#3-인덱스-전략)
4. [조인 최적화](#4-조인-최적화)
5. [통계와 비용 추정](#5-통계와-비용-추정)
6. [고급 최적화 기법](#6-고급-최적화-기법)
7. [연습 문제](#7-연습-문제)

---

## 1. EXPLAIN ANALYZE 심화

> **비유 -- GPS로 이해하는 쿼리 옵티마이저**: GPS가 현재 교통 상황을 바탕으로 고속도로, 골목길, 유료도로 등 여러 경로를 평가하여 가장 빠른 길을 선택하듯, PostgreSQL의 쿼리 플래너도 순차 스캔(Sequential Scan), 인덱스 스캔(Index Scan), 해시 조인(Hash Join), 병합 조인(Merge Join) 등 다양한 실행 전략을 평가하고 가장 낮은 비용을 가진 방법을 선택합니다. EXPLAIN 출력을 읽는 것은 GPS가 선택한 경로를 확인하는 것과 같습니다 -- 데이터베이스가 데이터에 도달하기 위해 어떤 "도로"를 이용할지 정확히 알려줍니다.

### 이론: Plan tree와 EXPLAIN

planner의 출력은 plan 노드의 트리. 잎은 scan(Seq Scan, Index Scan, …), 내부 노드는 연산(Nested Loop, Hash Join, Sort, Aggregate, …), root가 최종 결과 행을 만듦.

#### C.1 EXPLAIN 출력 읽기

```
Sort  (cost=22.07..22.32 rows=100 width=64) (actual time=0.305..0.312 rows=100 loops=1)
  Sort Key: created_at DESC
  ->  Hash Join  (cost=10.00..18.50 rows=100 width=64) (actual time=0.150..0.270 rows=100 loops=1)
        Hash Cond: (orders.user_id = users.id)
        ->  Seq Scan on orders  (cost=0.00..7.00 rows=500 width=32) (actual time=0.005..0.080 rows=500 loops=1)
        ->  Hash  (cost=8.00..8.00 rows=200 width=32) (actual time=0.040..0.040 rows=200 loops=1)
              ->  Seq Scan on users  (cost=0.00..8.00 rows=200 width=32) (actual time=0.002..0.020 rows=200 loops=1)
```

각 행이 표시:

- **노드 타입** (Sort, Hash Join, Seq Scan, …).
- **`cost=startup..total`** — startup cost는 "첫 행 전까지", total cost는 "모든 행에 대해".
- **`rows=N`** — 추정 행 수.
- **`width=W`** — 평균 행 폭(바이트).
- **`actual time=...`** — `EXPLAIN ANALYZE`에서만 — 실제 ms.
- **`actual rows=N`** — 실제 행 수.
- **`loops=N`** — 이 노드가 실행된 횟수(중첩 join의 inner loop에서).

#### C.2 무엇을 살펴볼 것인가

- **추정 `rows` vs `actual rows`** — 큰 불일치는 stale하거나 부족한 통계를 시사.
- **근처 대안보다 훨씬 높은 cost** — 작은 인덱스 하나가 cost를 2-3 자릿수 줄일 수 있음.
- **`Rows Removed by Filter`** — 높은 수는 scan이 많은 행을 반환했고 filter가 폐기했다는 뜻. filter 컬럼을 cover하는 인덱스로 종종 해결 가능.
- **index-only scan에서 `Heap Fetches`** — 0이 아니면 visibility map이 stale, `VACUUM`이 도움될 수 있음.

### 1.1 EXPLAIN 옵션

```sql
-- 기본 실행 계획
EXPLAIN SELECT * FROM users WHERE id = 1;

-- 실제 실행 + 시간 측정
EXPLAIN ANALYZE SELECT * FROM users WHERE id = 1;

-- 버퍼 정보 포함
EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM users WHERE id = 1;

-- 상세 출력
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) SELECT ...;
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT ...;
EXPLAIN (ANALYZE, BUFFERS, FORMAT YAML) SELECT ...;

-- 실행 없이 계획만 (ANALYZE 없이)
EXPLAIN (COSTS, VERBOSE) SELECT * FROM users;

-- 타이밍 비활성화 (오버헤드 감소)
EXPLAIN (ANALYZE, TIMING OFF) SELECT * FROM users;

-- 설정 정보 포함
EXPLAIN (ANALYZE, SETTINGS) SELECT * FROM users;
```

### 1.2 실행 계획 읽기

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT u.name, COUNT(o.id)
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at > '2024-01-01'
GROUP BY u.name;

/*
HashAggregate  (cost=1234.56..1234.78 rows=100 width=40)
               (actual time=45.123..45.456 loops=1)
  Group Key: u.name
  Batches: 1  Memory Usage: 24kB
  Buffers: shared hit=500 read=100
  ->  Hash Right Join  (cost=100.00..1200.00 rows=5000 width=36)
                       (actual time=5.123..40.456 loops=1)
        Hash Cond: (o.user_id = u.id)
        Buffers: shared hit=400 read=80
        ->  Seq Scan on orders o  (cost=0.00..800.00 rows=30000 width=8)
                                  (actual time=0.015..15.123 loops=1)
              Buffers: shared hit=300 read=50
        ->  Hash  (cost=80.00..80.00 rows=1000 width=36)
                  (actual time=3.456..3.456 loops=1)
              Buckets: 1024  Batches: 1  Memory Usage: 72kB
              Buffers: shared hit=100 read=30
              ->  Index Scan using idx_users_created on users u
                  (cost=0.29..80.00 rows=1000 width=36)
                  (actual time=0.030..2.345 loops=1)
                    Index Cond: (created_at > '2024-01-01')
                    Buffers: shared hit=100 read=30
Planning Time: 0.456 ms
Execution Time: 46.789 ms
*/
```

### 1.3 주요 지표 해석

```
┌─────────────────────────────────────────────────────────────┐
│                   실행 계획 지표 해석                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  cost=시작비용..총비용                                      │
│  • 시작비용: 첫 행 반환까지 비용                           │
│  • 총비용: 모든 행 반환까지 비용                           │
│  • 단위: 추상적 비용 단위                                   │
│                                                             │
│  rows=예상행수                                              │
│  • 플래너가 추정한 행 수                                    │
│                                                             │
│  width=행너비                                               │
│  • 행당 평균 바이트 수                                      │
│                                                             │
│  actual time=시작..종료                                     │
│  • 실제 실행 시간 (밀리초)                                  │
│                                                             │
│  loops=반복횟수                                             │
│  • 노드가 실행된 횟수                                       │
│  • 실제 시간 = time × loops                                │
│                                                             │
│  Buffers:                                                   │
│  • shared hit: 캐시에서 읽은 블록                          │
│  • shared read: 디스크에서 읽은 블록                       │
│  • shared written: 디스크에 쓴 블록                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.4 문제 식별

```sql
-- 문제: 예상 vs 실제 행 수 차이
-- 예상: rows=100, 실제: rows=10000
-- 원인: 오래된 통계, ANALYZE 필요

ANALYZE users;

-- 문제: 높은 시작 비용
-- Sort, Hash 등에서 발생
-- 해결: 적절한 인덱스 추가

-- 문제: loops가 큰 Nested Loop
-- 해결: JOIN 방식 변경 또는 인덱스

-- 문제: Seq Scan on 대형 테이블
-- 해결: 적절한 인덱스 추가
```

---

## 2. 쿼리 플래너

### 이론: Cost Model

모든 plan 노드는 단위 없는 숫자로 표현된 cost를 가집니다. planner는 가장 낮은 **total cost**의 plan을 선택. cost는 세 가지의 가중합 — sequential하게 읽은 page, random하게 읽은 page, 처리한 행당 CPU 시간.

#### A.1 상수

| 매개변수 | 기본값 | 의미 |
|---------|--------|------|
| `seq_page_cost` | 1.0 | sequential 1 page 읽기 비용 |
| `random_page_cost` | 4.0 | random offset 1 page 읽기 비용 |
| `cpu_tuple_cost` | 0.01 | 1 tuple 처리 비용 |
| `cpu_index_tuple_cost` | 0.005 | 1 인덱스 항목 처리 비용 |
| `cpu_operator_cost` | 0.0025 | operator/함수 호출 비용 |
| `parallel_tuple_cost` | 0.1 | worker→leader로 1 tuple 전송 비용 |
| `parallel_setup_cost` | 1000 | 병렬 worker 시작 비용 |

이들은 *상대적* 비용. 기본값은 HDD 시대의 random I/O가 sequential보다 4× 비싸다고 가정. **SSD 기반 스토리지에서는 `random_page_cost`를 ~1.1로 낮추세요**. 이 한 변경이 planner를 (회전 디스크에서 최적이었던) sequential scan으로부터 (random I/O가 저렴할 때 최적인) index scan 쪽으로 흔들 수 있습니다.

#### A.2 Sequential scan 공식

```
seq_scan_cost = seq_page_cost × pages_in_table
              + cpu_tuple_cost × rows_in_table
              + cpu_operator_cost × rows_in_table × operators_per_row
```

10000 page, 1000000 행 테이블에서, `1.0 × 10000 + 0.01 × 1000000 + ... ≈ 20000`.

#### A.3 Index scan 공식

```
index_scan_cost = index_pages_read × random_page_cost      (B-tree 하강)
                + cpu_index_tuple_cost × matching_index_entries
                + matching_rows × random_page_cost          (heap fetch)
                + cpu_tuple_cost × matching_rows
                + cpu_operator_cost × matching_rows
```

`matching_rows × random_page_cost` 항이 selectivity가 높을 때 index scan을 비싸게 만드는 것 — 10000-page 테이블의 10%면, `100,000 × 4.0 = 400,000` 대 sequential scan의 `20,000`. planner가 그 범위에서 sequential scan을 정확히 선택.

### 2.1 플래너 동작

```
┌─────────────────────────────────────────────────────────────┐
│                    쿼리 플래너 과정                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  SQL Query                                                  │
│      │                                                      │
│      ▼                                                      │
│  ┌─────────┐                                               │
│  │ Parser  │ → 구문 분석 → Parse Tree                      │
│  └─────────┘                                               │
│      │                                                      │
│      ▼                                                      │
│  ┌─────────┐                                               │
│  │Analyzer │ → 의미 분석 → Query Tree                      │
│  └─────────┘                                               │
│      │                                                      │
│      ▼                                                      │
│  ┌─────────┐                                               │
│  │Rewriter │ → 규칙 적용 (VIEW 등)                        │
│  └─────────┘                                               │
│      │                                                      │
│      ▼                                                      │
│  ┌─────────┐    ┌──────────────┐                          │
│  │Planner  │◄───│  Statistics  │                          │
│  └─────────┘    └──────────────┘                          │
│      │                                                      │
│      ▼ 최적 실행 계획 선택                                 │
│  ┌─────────┐                                               │
│  │Executor │ → 실행 → 결과                                │
│  └─────────┘                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 플래너 설정

```sql
-- 플래너 설정 확인
SHOW seq_page_cost;      -- 순차 페이지 읽기 비용 (기본 1.0)
SHOW random_page_cost;   -- 랜덤 페이지 읽기 비용 (기본 4.0)
SHOW cpu_tuple_cost;     -- 튜플 처리 비용 (기본 0.01)
SHOW cpu_index_tuple_cost;
SHOW cpu_operator_cost;

-- SSD에서는 random_page_cost 낮춤
SET random_page_cost = 1.1;

-- 특정 계획 비활성화 (테스트용)
SET enable_seqscan = off;
SET enable_indexscan = off;
SET enable_bitmapscan = off;
SET enable_hashjoin = off;
SET enable_mergejoin = off;
SET enable_nestloop = off;

-- 병렬 쿼리 설정
SET max_parallel_workers_per_gather = 4;
SET parallel_tuple_cost = 0.01;
SET parallel_setup_cost = 1000;
```

### 2.3 플래너 힌트 (pg_hint_plan)

```sql
-- pg_hint_plan 확장 설치 필요
CREATE EXTENSION pg_hint_plan;

-- 인덱스 힌트
/*+ IndexScan(users idx_users_email) */
SELECT * FROM users WHERE email = 'test@example.com';

-- 조인 순서 힌트
/*+ Leading(orders users) */
SELECT * FROM users u JOIN orders o ON u.id = o.user_id;

-- 조인 방법 힌트
/*+ HashJoin(users orders) */
SELECT * FROM users u JOIN orders o ON u.id = o.user_id;

/*+ NestLoop(users orders) */
SELECT * FROM users u JOIN orders o ON u.id = o.user_id;

-- Seq Scan 강제
/*+ SeqScan(users) */
SELECT * FROM users WHERE id > 100;

-- 병렬 쿼리 비활성화
/*+ Parallel(users 0) */
SELECT COUNT(*) FROM users;
```

---

## 3. 인덱스 전략

### 3.1 인덱스 타입 선택

```sql
-- B-tree (기본, 대부분의 경우)
CREATE INDEX idx_users_email ON users(email);

-- 적합: =, <, >, <=, >=, BETWEEN, IN, IS NULL
-- LIKE 'abc%' (앞부분 매칭)

-- Hash (동등 비교만)
CREATE INDEX idx_users_email_hash ON users USING HASH (email);
-- 적합: = 만
-- PostgreSQL 10+ 에서 WAL 지원

-- GiST (기하학, 범위, 전문 검색)
CREATE INDEX idx_locations_point ON locations USING GIST (point);
CREATE INDEX idx_events_range ON events USING GIST (time_range);

-- GIN (배열, JSONB, 전문 검색)
CREATE INDEX idx_posts_tags ON posts USING GIN (tags);
CREATE INDEX idx_products_attrs ON products USING GIN (attributes);
CREATE INDEX idx_docs_search ON documents USING GIN (to_tsvector('english', content));

-- BRIN (대용량 순차 데이터)
CREATE INDEX idx_logs_time ON logs USING BRIN (created_at);
-- 적합: 물리적으로 정렬된 데이터 (시계열 등)
-- 매우 작은 크기, 대용량 테이블에 효과적
```

### 3.2 복합 인덱스

```sql
-- 복합 인덱스 순서 중요!
CREATE INDEX idx_orders_user_date ON orders(user_id, created_at);

-- 이 쿼리는 인덱스 사용 가능:
SELECT * FROM orders WHERE user_id = 1;
SELECT * FROM orders WHERE user_id = 1 AND created_at > '2024-01-01';

-- 이 쿼리는 인덱스 사용 불가 (첫 번째 컬럼 없음):
SELECT * FROM orders WHERE created_at > '2024-01-01';

-- 정렬 최적화
CREATE INDEX idx_orders_user_date_desc ON orders(user_id, created_at DESC);

-- INCLUDE (커버링 인덱스, PostgreSQL 11+)
CREATE INDEX idx_orders_covering ON orders(user_id)
INCLUDE (status, total);
-- 인덱스만으로 쿼리 가능 (Index Only Scan)
```

### 3.3 부분 인덱스

```sql
-- 특정 조건에만 인덱스
CREATE INDEX idx_orders_pending ON orders(created_at)
WHERE status = 'pending';

-- NULL 제외
CREATE INDEX idx_users_email_notnull ON users(email)
WHERE email IS NOT NULL;

-- 최근 데이터만
CREATE INDEX idx_logs_recent ON logs(level, message)
WHERE created_at > '2024-01-01';

-- 삭제되지 않은 행만
CREATE INDEX idx_active_products ON products(category_id)
WHERE deleted_at IS NULL;
```

### 3.4 인덱스 관리

```sql
-- 인덱스 사용 통계
SELECT
    schemaname,
    relname AS table_name,
    indexrelname AS index_name,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- 사용되지 않는 인덱스 찾기
SELECT
    schemaname || '.' || relname AS table,
    indexrelname AS index,
    pg_size_pretty(pg_relation_size(i.indexrelid)) AS size,
    idx_scan
FROM pg_stat_user_indexes ui
JOIN pg_index i ON ui.indexrelid = i.indexrelid
WHERE idx_scan = 0
AND NOT indisunique
ORDER BY pg_relation_size(i.indexrelid) DESC;

-- 중복 인덱스 찾기
SELECT
    a.indrelid::regclass AS table_name,
    a.indexrelid::regclass AS index1,
    b.indexrelid::regclass AS index2
FROM pg_index a
JOIN pg_index b ON a.indrelid = b.indrelid
AND a.indexrelid < b.indexrelid
AND (
    (a.indkey::text LIKE b.indkey::text || '%')
    OR (b.indkey::text LIKE a.indkey::text || '%')
);

-- 인덱스 재구성
REINDEX INDEX idx_users_email;
REINDEX TABLE users;
REINDEX DATABASE mydb CONCURRENTLY;  -- PostgreSQL 12+

-- 동시 인덱스 생성 (락 최소화)
CREATE INDEX CONCURRENTLY idx_users_email ON users(email);
```

---

## 4. 조인 최적화

### 이론: Join-Order 탐색과 GEQO

N-way join의 경우, planner는 가능한 많은 join 순서를 고려해야 합니다.

#### D.1 탐색 공간

`N` 테이블의 경우, 고려할 *tree* 수(left-deep, bushy 등)는 지수적으로 자랍니다 — bushy tree에 대해 대략 `(2N)! / N!`. 10 테이블이면 수백만, 20 테이블이면 천문학적.

#### D.2 Dynamic programming (기본)

planner는 **bottom-up dynamic programming**을 사용 — 모든 테이블 쌍에 대해 가장 저렴한 join 찾기, 모든 삼중 조에 대해 가장 저렴한 확장 찾기, 계속. 중간 join의 cost가 재사용되므로 알고리즘은 O(2^N · N^2) — ~12 테이블까지 다룰 만하지만 그 이상은 지수적.

#### D.3 GEQO — 큰 join을 위한 fallback

`from_collapse_limit + join_collapse_limit`만큼의 joinable 테이블이 `geqo_threshold`(기본 12)를 초과하면, PostgreSQL은 **Genetic Query Optimizer**로 전환 — join 순서를 chromosome으로 표현하고, crossover와 mutation으로 population을 진화시키고, 설정 가능한 generation 수만큼 실행. 결과는 *최적이 보장되지 않지만* ≥20 테이블에 대해 exhaustive DP보다 계산이 훨씬 빠름.

planning 시간을 비용으로 exhaustive 탐색을 원한다면 `SET geqo = off;`로 GEQO를 비활성화할 수 있고, `geqo_threshold`를 올려 경계를 밀어낼 수 있습니다.

### 4.1 조인 방식 비교

```
┌─────────────────────────────────────────────────────────────┐
│                      조인 방식 비교                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Nested Loop Join                                           │
│  ─────────────────                                          │
│  for each row in outer:                                     │
│      for each row in inner:                                 │
│          if match: emit                                     │
│                                                             │
│  • 적합: 소규모 테이블, 인덱스 있을 때                     │
│  • 비용: O(N × M), 인덱스 시 O(N × log M)                  │
│                                                             │
│  Hash Join                                                  │
│  ─────────────────                                          │
│  build hash table from inner                                │
│  for each row in outer:                                     │
│      probe hash table                                       │
│                                                             │
│  • 적합: 대규모 테이블, 동등 조인                          │
│  • 비용: O(N + M)                                          │
│  • 메모리 필요 (work_mem)                                  │
│                                                             │
│  Merge Join                                                 │
│  ─────────────────                                          │
│  sort both tables                                           │
│  merge sorted lists                                         │
│                                                             │
│  • 적합: 이미 정렬된 데이터, 범위 조인                     │
│  • 비용: O(N log N + M log M + N + M)                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 조인 순서 최적화

```sql
-- 조인 순서는 성능에 큰 영향
-- 플래너가 자동 최적화하지만, 테이블 많으면 제한

-- 조인 가능한 테이블 수 제한
SHOW join_collapse_limit;  -- 기본 8
SHOW from_collapse_limit;  -- 기본 8

-- 많은 테이블 조인 시 순서 중요
-- 작은 테이블/필터링 많은 테이블 먼저

-- 좋은 예: 필터링 먼저
SELECT *
FROM orders o
JOIN users u ON o.user_id = u.id
WHERE o.status = 'pending'  -- 필터링
AND o.created_at > '2024-01-01';

-- 조인 순서 명시 (테스트용)
SET join_collapse_limit = 1;
SELECT * FROM t1, t2, t3
WHERE t1.id = t2.t1_id AND t2.id = t3.t2_id;
RESET join_collapse_limit;
```

### 4.3 조인 성능 개선

```sql
-- 적절한 인덱스
CREATE INDEX idx_orders_user ON orders(user_id);

-- 조인 컬럼 타입 일치
-- 나쁨: orders.user_id (int) JOIN users.id (bigint) → 형변환
-- 좋음: 같은 타입 사용

-- 불필요한 조인 제거
-- 나쁨
SELECT o.* FROM orders o
JOIN users u ON o.user_id = u.id;  -- users에서 아무것도 안 가져옴

-- 좋음 (조인 제거)
SELECT o.* FROM orders o
WHERE EXISTS (SELECT 1 FROM users u WHERE u.id = o.user_id);

-- 서브쿼리 → 조인 변환
-- 나쁨 (상관 서브쿼리)
SELECT *,
    (SELECT name FROM users WHERE id = o.user_id) AS user_name
FROM orders o;

-- 좋음
SELECT o.*, u.name AS user_name
FROM orders o
JOIN users u ON o.user_id = u.id;
```

---

## 5. 통계와 비용 추정

### 이론: 통계 — 행 수가 어디서 오는가

cost 공식은 `matching_rows`와 `pages_in_table`을 알아야 합니다. planner는 `ANALYZE`가 채운 `pg_statistic`에서 이를 얻음.

#### B.1 `ANALYZE`가 수집하는 것

각 컬럼에 대해 `ANALYZE`는 일부 행을 sampling(`default_statistics_target` × 300, 기본 30000)하고 다음을 계산:

- **distinct value 수** (`n_distinct`)
- **Most common values (MCVs)** — 보통 가장 빈번한 100개 값과 그 빈도.
- **Histogram** — 나머지 값을 빈도가 같은 N bucket(기본 100 bucket)으로 분할.
- **Null fraction** (`null_frac`)
- **Correlation** — 물리적 행 순서와 논리적 컬럼 순서 사이(정렬된 출력을 만드는 index scan의 비용 추정에 사용).
- **평균 컬럼 폭** (`avg_width`).

`pg_statistic`(또는 사람이 읽기 쉬운 `pg_stats` view)에 저장.

#### B.2 Selectivity 추정

`WHERE col = value`에 대해:

- `value`가 MCV 목록에 있으면, selectivity = 그 빈도. 정확.
- 아니면, selectivity = `(1 - sum_of_MCV_frequencies - null_frac) / (n_distinct - count_of_MCVs)`. long tail 평균.

`WHERE col BETWEEN a AND b`에 대해:

- selectivity = (a와 b 사이 histogram bucket 수) / 전체 bucket, 경계 bucket 안에서는 linear interpolation.

`WHERE col1 = ? AND col2 = ?`에 대해:

- 기본적으로 planner는 컬럼이 **independent**라고 가정하고 컬럼별 selectivity를 곱함. 이것이 나쁜 추정의 가장 흔한 원천 — 컬럼이 correlated일 때(예 — `country = 'KR' AND city = 'Seoul'`), 가정이 극단적으로 잘못된 행 수를 만듦.

#### B.3 Extended statistics — correlation 과소평가 수정

`CREATE STATISTICS s_country_city (dependencies, ndistinct) ON country, city FROM addresses;`는 그 컬럼들에 대한 joint statistics를 추적하라고 planner에 알림. 다음 `ANALYZE` 후, 결합 술어의 selectivity 추정이 independence 가정 대신 실제 correlation을 사용.

#### B.4 나쁜 통계가 나쁜 plan을 만드는 이유

planner가 join이 10행을 반환할 것이라고 추정했는데 실제로 10000행을 반환하면, Nested Loop을 선택(10행에 좋고, 10000행에 끔찍). 해결책은 거의 "`random_page_cost` 튜닝"이 아닙니다 — `ANALYZE`(특히 큰 데이터 변경 후), 그리고 correlated 컬럼에는 `CREATE STATISTICS`.

### 5.1 통계 수집

```sql
-- 테이블 통계 수집
ANALYZE users;
ANALYZE;  -- 전체 데이터베이스

-- 자동 ANALYZE 설정
SHOW autovacuum_analyze_threshold;     -- 기본 50
SHOW autovacuum_analyze_scale_factor;  -- 기본 0.1

-- 특정 컬럼 통계 상세도
ALTER TABLE users ALTER COLUMN email SET STATISTICS 1000;
-- 기본 100, 최대 10000
ANALYZE users;

-- 통계 확인
SELECT
    attname,
    n_distinct,
    most_common_vals,
    most_common_freqs,
    histogram_bounds
FROM pg_stats
WHERE tablename = 'users';
```

### 5.2 행 수 추정

```sql
-- 테이블 행 수 추정
SELECT reltuples::bigint AS estimate
FROM pg_class
WHERE relname = 'users';

-- 정확한 행 수 (느림)
SELECT COUNT(*) FROM users;

-- 조건부 행 수 추정
EXPLAIN SELECT * FROM users WHERE status = 'active';
-- rows=xxx 확인

-- 추정 정확도 개선
-- 1. ANALYZE 실행
-- 2. 통계 상세도 증가
-- 3. 확장 통계 (PostgreSQL 10+)
CREATE STATISTICS stts_user_country_status (dependencies)
ON country, status FROM users;
ANALYZE users;
```

### 5.3 비용 계산

```sql
-- 비용 = (페이지 수 × 페이지 비용) + (행 수 × 행 비용)

-- 페이지 수 확인
SELECT relpages FROM pg_class WHERE relname = 'users';

-- 비용 파라미터
SHOW seq_page_cost;        -- 1.0
SHOW random_page_cost;     -- 4.0
SHOW cpu_tuple_cost;       -- 0.01
SHOW cpu_index_tuple_cost; -- 0.005
SHOW cpu_operator_cost;    -- 0.0025

-- Seq Scan 비용 계산 예
-- cost = (relpages × seq_page_cost) + (reltuples × cpu_tuple_cost)
-- cost = (1000 × 1.0) + (100000 × 0.01) = 2000

-- Index Scan 비용은 더 복잡
-- 선택도(selectivity)에 따라 다름
```

---

## 6. 고급 최적화 기법

### 6.1 쿼리 리팩토링

```sql
-- OR → UNION (인덱스 활용)
-- 나쁨
SELECT * FROM products
WHERE category_id = 1 OR brand_id = 2;

-- 좋음
SELECT * FROM products WHERE category_id = 1
UNION
SELECT * FROM products WHERE brand_id = 2;

-- IN → EXISTS (대량 데이터)
-- 나쁨 (서브쿼리 결과 많을 때)
SELECT * FROM users
WHERE id IN (SELECT user_id FROM orders WHERE amount > 1000);

-- 좋음
SELECT * FROM users u
WHERE EXISTS (
    SELECT 1 FROM orders o
    WHERE o.user_id = u.id AND o.amount > 1000
);

-- NOT IN → NOT EXISTS (NULL 처리)
-- NOT IN은 NULL 있으면 항상 빈 결과
SELECT * FROM users
WHERE id NOT IN (SELECT user_id FROM orders);  -- orders.user_id에 NULL 있으면 문제

-- 안전한 방법
SELECT * FROM users u
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id);

-- DISTINCT → GROUP BY (인덱스 활용)
SELECT DISTINCT user_id FROM orders;
-- →
SELECT user_id FROM orders GROUP BY user_id;
```

### 6.2 Materialized View

```sql
-- 복잡한 집계 결과 저장
CREATE MATERIALIZED VIEW mv_daily_sales AS
SELECT
    date_trunc('day', created_at) AS day,
    COUNT(*) AS order_count,
    SUM(total) AS total_sales
FROM orders
GROUP BY date_trunc('day', created_at);

-- 인덱스 추가
CREATE UNIQUE INDEX idx_mv_daily_sales_day ON mv_daily_sales(day);

-- 새로고침
REFRESH MATERIALIZED VIEW mv_daily_sales;
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_sales;  -- UNIQUE 인덱스 필요

-- 자동 새로고침 (pg_cron 또는 트리거 사용)
```

### 6.3 파티셔닝

```sql
-- 범위 파티셔닝
CREATE TABLE orders (
    id BIGSERIAL,
    created_at TIMESTAMP NOT NULL,
    user_id INT,
    total DECIMAL(10,2)
) PARTITION BY RANGE (created_at);

CREATE TABLE orders_2024_q1 PARTITION OF orders
FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

CREATE TABLE orders_2024_q2 PARTITION OF orders
FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');

-- 파티션 프루닝 확인
EXPLAIN SELECT * FROM orders WHERE created_at = '2024-02-15';
-- orders_2024_q1만 스캔

-- 리스트 파티셔닝
CREATE TABLE logs (
    id BIGSERIAL,
    level VARCHAR(10),
    message TEXT
) PARTITION BY LIST (level);

CREATE TABLE logs_error PARTITION OF logs FOR VALUES IN ('ERROR', 'FATAL');
CREATE TABLE logs_info PARTITION OF logs FOR VALUES IN ('INFO', 'DEBUG');

-- 해시 파티셔닝
CREATE TABLE events (
    id BIGSERIAL,
    user_id INT
) PARTITION BY HASH (user_id);

CREATE TABLE events_p0 PARTITION OF events FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE events_p1 PARTITION OF events FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE events_p2 PARTITION OF events FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE events_p3 PARTITION OF events FOR VALUES WITH (MODULUS 4, REMAINDER 3);
```

### 6.4 쿼리 캐싱

```sql
-- Prepared Statement (쿼리 계획 캐싱)
PREPARE get_user(int) AS
SELECT * FROM users WHERE id = $1;

EXECUTE get_user(1);
EXECUTE get_user(2);

DEALLOCATE get_user;

-- PgBouncer 등 커넥션 풀러에서 prepared statement 주의

-- 결과 캐싱 (애플리케이션 레벨)
-- Redis, Memcached 사용 권장
```

---

## 7. 연습 문제

### 연습 1: 실행 계획 분석
```sql
-- 다음 쿼리의 실행 계획 분석 및 최적화:
SELECT u.name, COUNT(o.id), SUM(o.total)
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.country = 'US'
AND o.created_at > NOW() - INTERVAL '1 year'
GROUP BY u.name
HAVING COUNT(o.id) > 10
ORDER BY SUM(o.total) DESC
LIMIT 100;

-- 분석 및 개선 방안 제시:
```

### 연습 2: 인덱스 설계
```sql
-- 다음 쿼리들을 위한 최적 인덱스 설계:
-- 1. SELECT * FROM orders WHERE user_id = ? AND status = 'pending' ORDER BY created_at DESC
-- 2. SELECT * FROM products WHERE category_id = ? AND price BETWEEN ? AND ?
-- 3. SELECT * FROM logs WHERE level = 'ERROR' AND created_at > NOW() - INTERVAL '1 day'

-- 인덱스 생성문 작성:
```

### 연습 3: 조인 최적화
```sql
-- 5개 테이블 조인 쿼리 최적화:
SELECT *
FROM orders o
JOIN users u ON o.user_id = u.id
JOIN products p ON o.product_id = p.id
JOIN categories c ON p.category_id = c.id
JOIN suppliers s ON p.supplier_id = s.id
WHERE c.name = 'Electronics'
AND o.created_at > '2024-01-01';

-- 최적화 전략 수립:
```

### 연습 4: 파티셔닝 설계
```sql
-- 대용량 로그 테이블 파티셔닝:
-- 요구사항:
-- - 일별 데이터 100만 행
-- - 3개월 보관
-- - 자주 조회: level, created_at, user_id

-- 파티션 설계:
```

---

## 참고 자료

- [PostgreSQL EXPLAIN](https://www.postgresql.org/docs/current/using-explain.html)
- [Query Planning](https://www.postgresql.org/docs/current/planner-optimizer.html)
- [Index Types](https://www.postgresql.org/docs/current/indexes-types.html)
- [Use The Index, Luke](https://use-the-index-luke.com/)

---

**이전**: [JSON과 JSONB](./14_JSON_JSONB.md) | **다음**: [복제와 고가용성](./16_Replication_HA.md)
