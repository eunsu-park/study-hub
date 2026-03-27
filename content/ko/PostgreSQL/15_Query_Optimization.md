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
