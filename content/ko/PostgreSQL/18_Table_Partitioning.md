# 18. 테이블 파티셔닝 (Table Partitioning)

**이전**: [윈도우 함수](./17_Window_Functions.md) | **다음**: [전문 검색](./19_Full_Text_Search.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 테이블 파티셔닝(Table Partitioning)의 개념을 설명하고, 의미 있는 성능 이점을 제공하는 시나리오를 식별한다
2. 적절한 기본 키와 인덱스 설계를 갖춘 시계열 데이터용 범위 파티션(Range Partition) 테이블을 생성한다
3. 지역이나 상태 등 범주형 데이터에 리스트 파티셔닝(List Partitioning)을 구현한다
4. 자연적인 범위나 범주가 없을 때 균등한 데이터 분산을 위한 해시 파티셔닝(Hash Partitioning)을 설정한다
5. EXPLAIN ANALYZE를 사용하여 파티션 프루닝(Partition Pruning) 동작을 검증하고, 프루닝을 무력화하는 일반적인 함정을 피한다
6. 파티션 추가, 분리, 삭제, pg_cron을 이용한 자동화 생성 등 동적으로 파티션을 관리한다
7. 기존의 비파티션 테이블을 최소 다운타임으로 파티션 테이블로 전환한다

---

테이블이 수백만 행에서 수십억 행으로 증가하면, 인덱스 자체가 방대해지고 VACUUM 및 백업 같은 유지보수 작업이 길어지면서 잘 인덱싱된 쿼리조차 느려질 수 있습니다. 파티셔닝(Partitioning)은 하나의 논리적 테이블을 데이터의 일부를 담는 더 작은 물리적 조각으로 분할합니다. 이를 통해 PostgreSQL은 관련 파티션만 스캔하고, 더 작은 인덱스를 유지하며, 값비싼 DELETE 문 대신 파티션 전체를 즉시 삭제할 수 있습니다. 시계열 데이터, 이벤트 로그, 대용량 트랜잭션 데이터를 다루는 시스템이라면 파티셔닝은 필수적인 확장 전략입니다.

## 목차
1. [파티셔닝 개요](#1-파티셔닝-개요)
2. [Range 파티셔닝](#2-range-파티셔닝)
3. [List 파티셔닝](#3-list-파티셔닝)
4. [Hash 파티셔닝](#4-hash-파티셔닝)
5. [파티션 프루닝](#5-파티션-프루닝)
6. [파티션 관리](#6-파티션-관리)
7. [연습 문제](#7-연습-문제)

---

## 1. 파티셔닝 개요

### 1.1 파티셔닝이란?

```
┌─────────────────────────────────────────────────────────────────┐
│                    테이블 파티셔닝 개념                          │
│                                                                 │
│   일반 테이블                   파티션 테이블                    │
│   ┌───────────────┐            ┌───────────────┐               │
│   │   orders      │            │ orders (부모) │               │
│   │   (1억 행)    │            └───────┬───────┘               │
│   │               │                    │                       │
│   │   모든 데이터 │            ┌───────┼───────┐               │
│   │   하나의 파일 │            │       │       │               │
│   └───────────────┘        ┌───┴───┐ ┌─┴──┐ ┌──┴──┐           │
│                            │2024_Q1│ │Q2  │ │ Q3  │ ...       │
│                            │2500만 │ │    │ │     │           │
│                            └───────┘ └────┘ └─────┘           │
│                            (분할 저장)                         │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 파티셔닝의 장점

```
┌─────────────────┬───────────────────────────────────────────────┐
│ 장점            │ 설명                                           │
├─────────────────┼───────────────────────────────────────────────┤
│ 쿼리 성능       │ 파티션 프루닝으로 스캔 범위 축소               │
│ 유지보수 용이   │ 파티션 단위로 VACUUM, 백업, 삭제 가능          │
│ 데이터 아카이빙 │ 오래된 파티션을 별도 테이블스페이스로 이동     │
│ 대량 삭제       │ DROP PARTITION으로 빠른 삭제 (DELETE 대비)     │
│ 인덱스 크기     │ 파티션별 작은 인덱스 (메모리 효율)             │
│ 병렬 처리       │ 파티션 단위 병렬 스캔 가능                     │
└─────────────────┴───────────────────────────────────────────────┘
```

### 1.3 파티셔닝 유형

```
┌─────────────────────────────────────────────────────────────────┐
│                    파티셔닝 유형                                 │
│                                                                 │
│   Range: 연속 범위 기준                                         │
│   ├── 날짜별 (월별, 분기별, 연도별)                             │
│   └── 숫자 범위 (ID 범위, 금액 범위)                            │
│                                                                 │
│   List: 이산 값 목록 기준                                       │
│   ├── 지역 (국가, 도시)                                        │
│   ├── 상태 (활성, 비활성, 보류)                                 │
│   └── 카테고리                                                  │
│                                                                 │
│   Hash: 해시 값 기준                                            │
│   └── 균등 분산이 필요한 경우 (특정 기준 없음)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Range 파티셔닝

### 2.1 기본 구조

```sql
-- 파티션 키 선택이 핵심: WHERE 절에서 가장 자주 사용되는 컬럼을 선택.
-- order_date가 이상적 — 쿼리가 거의 항상 시간 범위로 필터링하여
-- 파티션 프루닝으로 관련 없는 월을 완전히 건너뛸 수 있음
CREATE TABLE orders (
    id BIGSERIAL,
    customer_id INT NOT NULL,
    order_date DATE NOT NULL,
    amount NUMERIC(10,2),
    status VARCHAR(20)
) PARTITION BY RANGE (order_date);

-- 파티션 생성 (월별)
CREATE TABLE orders_2024_01 PARTITION OF orders
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE orders_2024_02 PARTITION OF orders
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

CREATE TABLE orders_2024_03 PARTITION OF orders
    FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');

-- 기본 파티션 (범위에 맞지 않는 데이터용)
CREATE TABLE orders_default PARTITION OF orders DEFAULT;
```

### 2.2 인덱스 생성

```sql
-- 부모 테이블에 인덱스 생성 (자동으로 파티션에도 적용)
CREATE INDEX idx_orders_customer ON orders (customer_id);
CREATE INDEX idx_orders_date ON orders (order_date);

-- 파티션별 개별 인덱스 확인
SELECT
    schemaname,
    tablename,
    indexname
FROM pg_indexes
WHERE tablename LIKE 'orders%';
```

### 2.3 PRIMARY KEY와 UNIQUE 제약

```sql
-- 파티션 테이블의 PK/UNIQUE는 파티션 키를 포함해야 함
CREATE TABLE orders (
    id BIGSERIAL,
    order_date DATE NOT NULL,
    customer_id INT NOT NULL,
    amount NUMERIC(10,2),
    PRIMARY KEY (id, order_date)  -- 파티션 키 포함
) PARTITION BY RANGE (order_date);

-- 복합 UNIQUE 제약
ALTER TABLE orders ADD CONSTRAINT orders_unique
    UNIQUE (id, order_date);
```

### 2.4 분기별 파티셔닝 예시

```sql
-- 분기별 파티션
CREATE TABLE sales (
    id BIGSERIAL,
    sale_date DATE NOT NULL,
    product_id INT,
    amount NUMERIC(10,2),
    PRIMARY KEY (id, sale_date)
) PARTITION BY RANGE (sale_date);

-- 2024년 분기별 파티션
CREATE TABLE sales_2024_q1 PARTITION OF sales
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');
CREATE TABLE sales_2024_q2 PARTITION OF sales
    FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');
CREATE TABLE sales_2024_q3 PARTITION OF sales
    FOR VALUES FROM ('2024-07-01') TO ('2024-10-01');
CREATE TABLE sales_2024_q4 PARTITION OF sales
    FOR VALUES FROM ('2024-10-01') TO ('2025-01-01');
```

---

## 3. List 파티셔닝

### 3.1 지역별 파티셔닝

```sql
-- 지역별 파티션
CREATE TABLE customers (
    id SERIAL,
    name VARCHAR(100),
    email VARCHAR(255),
    region VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (id, region)
) PARTITION BY LIST (region);

-- 대륙별 파티션
CREATE TABLE customers_asia PARTITION OF customers
    FOR VALUES IN ('KR', 'JP', 'CN', 'SG', 'IN');

CREATE TABLE customers_europe PARTITION OF customers
    FOR VALUES IN ('UK', 'DE', 'FR', 'IT', 'ES');

CREATE TABLE customers_americas PARTITION OF customers
    FOR VALUES IN ('US', 'CA', 'MX', 'BR');

CREATE TABLE customers_others PARTITION OF customers DEFAULT;
```

### 3.2 상태별 파티셔닝

```sql
-- 주문 상태별 파티션
CREATE TABLE order_items (
    id BIGSERIAL,
    order_id BIGINT,
    status VARCHAR(20) NOT NULL,
    product_id INT,
    quantity INT,
    PRIMARY KEY (id, status)
) PARTITION BY LIST (status);

CREATE TABLE order_items_pending PARTITION OF order_items
    FOR VALUES IN ('pending', 'processing');

CREATE TABLE order_items_completed PARTITION OF order_items
    FOR VALUES IN ('shipped', 'delivered');

CREATE TABLE order_items_cancelled PARTITION OF order_items
    FOR VALUES IN ('cancelled', 'refunded');
```

### 3.3 다중 컬럼 List 파티셔닝

```sql
-- PostgreSQL 11+ 다중 컬럼 파티션
CREATE TABLE events (
    id BIGSERIAL,
    event_type VARCHAR(20) NOT NULL,
    event_date DATE NOT NULL,
    data JSONB,
    PRIMARY KEY (id, event_type, event_date)
) PARTITION BY LIST (event_type);

-- 이벤트 타입별 파티션 → 내부에서 Range 서브파티션
CREATE TABLE events_click PARTITION OF events
    FOR VALUES IN ('click')
    PARTITION BY RANGE (event_date);

CREATE TABLE events_click_2024_01 PARTITION OF events_click
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

---

## 4. Hash 파티셔닝

### 4.1 기본 Hash 파티셔닝

```sql
-- 해시 파티셔닝 (균등 분산)
CREATE TABLE logs (
    id BIGSERIAL,
    user_id INT NOT NULL,
    action VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (id, user_id)
) PARTITION BY HASH (user_id);

-- 4개 파티션으로 분산
CREATE TABLE logs_p0 PARTITION OF logs
    FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE logs_p1 PARTITION OF logs
    FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE logs_p2 PARTITION OF logs
    FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE logs_p3 PARTITION OF logs
    FOR VALUES WITH (MODULUS 4, REMAINDER 3);
```

### 4.2 Hash 파티션 생성 자동화

```sql
-- 동적 파티션 생성 함수
CREATE OR REPLACE FUNCTION create_hash_partitions(
    parent_table TEXT,
    num_partitions INT
) RETURNS VOID AS $$
DECLARE
    i INT;
BEGIN
    FOR i IN 0..num_partitions-1 LOOP
        EXECUTE format(
            'CREATE TABLE %I PARTITION OF %I FOR VALUES WITH (MODULUS %s, REMAINDER %s)',
            parent_table || '_p' || i,
            parent_table,
            num_partitions,
            i
        );
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- 사용
SELECT create_hash_partitions('logs', 8);
```

### 4.3 Hash vs Range/List 선택 기준

```
┌─────────────────────────────────────────────────────────────────┐
│                    파티셔닝 유형 선택 가이드                      │
│                                                                 │
│   Range 선택:                                                   │
│   - 시간 기반 데이터 (로그, 트랜잭션)                            │
│   - 범위 쿼리가 빈번한 경우                                      │
│   - 오래된 데이터 아카이빙/삭제 필요                             │
│                                                                 │
│   List 선택:                                                    │
│   - 명확한 카테고리 구분                                        │
│   - 지역, 상태, 타입 등 이산 값                                  │
│   - 특정 카테고리만 자주 조회                                    │
│                                                                 │
│   Hash 선택:                                                    │
│   - 명확한 분류 기준 없음                                        │
│   - 데이터 균등 분산이 목표                                      │
│   - 범위 쿼리가 필요 없음                                        │
│   - 파티션 수가 고정                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 파티션 프루닝

### 5.1 프루닝 동작 확인

```sql
-- 파티션 프루닝이 핵심 성능 이점: 플래너가 매칭 행을 포함할 수 없는 파티션을
-- 실행 전에 제거하여, 12개 파티션 테이블을 WHERE 절이 한 달과 일치할 때
-- 단일 파티션 스캔으로 변환
EXPLAIN (ANALYZE, COSTS OFF)
SELECT * FROM orders
WHERE order_date = '2024-02-15';

-- 결과 예시:
-- Append
--   ->  Seq Scan on orders_2024_02  -- 2월 파티션만 스캔
--         Filter: (order_date = '2024-02-15'::date)
```

### 5.2 프루닝 설정

```sql
-- 프루닝 활성화 확인
SHOW enable_partition_pruning;  -- on (기본값)

-- 런타임 프루닝 (조인, 서브쿼리에서)
SET enable_partition_pruning = on;
```

### 5.3 프루닝이 작동하지 않는 경우

```sql
-- 1. 파티션 키에 함수 적용 시 프루닝 실패 — 키가 EXTRACT() 등으로 감싸지면
-- 플래너가 어떤 파티션을 건너뛸지 추론 불가
SELECT * FROM orders
WHERE EXTRACT(YEAR FROM order_date) = 2024;

-- 범위 조건으로 재작성하면 플래너가 파티션 경계를 직접 매칭 가능
SELECT * FROM orders
WHERE order_date >= '2024-01-01' AND order_date < '2025-01-01';

-- 2. 암시적 형변환
-- 나쁜 예 (문자열 비교)
SELECT * FROM orders WHERE order_date = '2024-02-15';  -- 문자열

-- 좋은 예 (명시적 타입)
SELECT * FROM orders WHERE order_date = DATE '2024-02-15';

-- 3. OR 조건의 부분적 프루닝
SELECT * FROM orders
WHERE order_date = '2024-01-15' OR customer_id = 123;
-- customer_id 조건이 모든 파티션 스캔 유발
```

### 5.4 파티션 제외 힌트

```sql
-- 특정 파티션만 직접 쿼리
SELECT * FROM orders_2024_02  -- 파티션 직접 참조
WHERE customer_id = 123;

-- constraint_exclusion 설정
SET constraint_exclusion = partition;  -- 기본값
```

---

## 6. 파티션 관리

### 6.1 파티션 추가

```sql
-- 새 파티션 추가
CREATE TABLE orders_2024_04 PARTITION OF orders
    FOR VALUES FROM ('2024-04-01') TO ('2024-05-01');

-- 또는 기존 테이블을 파티션으로 연결
CREATE TABLE orders_2024_05 (LIKE orders INCLUDING ALL);
ALTER TABLE orders ATTACH PARTITION orders_2024_05
    FOR VALUES FROM ('2024-05-01') TO ('2024-06-01');
```

### 6.2 파티션 분리 및 삭제

```sql
-- 파티션 분리 (데이터 보존, 독립 테이블로)
ALTER TABLE orders DETACH PARTITION orders_2024_01;

-- 분리된 테이블은 독립적으로 존재
SELECT * FROM orders_2024_01;

-- 파티션 삭제 (데이터도 삭제)
DROP TABLE orders_2024_01;
```

### 6.3 자동 파티션 생성

```sql
-- 월별 파티션 자동 생성 함수
CREATE OR REPLACE FUNCTION create_monthly_partition(
    parent_table TEXT,
    partition_date DATE
) RETURNS VOID AS $$
DECLARE
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    start_date := DATE_TRUNC('month', partition_date);
    end_date := start_date + INTERVAL '1 month';
    partition_name := parent_table || '_' || TO_CHAR(start_date, 'YYYY_MM');

    -- 이미 존재하면 스킵
    IF NOT EXISTS (
        SELECT 1 FROM pg_tables WHERE tablename = partition_name
    ) THEN
        EXECUTE format(
            'CREATE TABLE %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
            partition_name,
            parent_table,
            start_date,
            end_date
        );
        RAISE NOTICE 'Created partition: %', partition_name;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- 향후 3개월 파티션 미리 생성
DO $$
BEGIN
    FOR i IN 0..2 LOOP
        PERFORM create_monthly_partition(
            'orders',
            CURRENT_DATE + (i || ' months')::interval
        );
    END LOOP;
END;
$$;
```

### 6.4 pg_cron을 사용한 자동화

```sql
-- pg_cron 확장 설치 (별도 설치 필요)
CREATE EXTENSION pg_cron;

-- 매월 1일 새 파티션 생성
SELECT cron.schedule(
    'create-partition',
    '0 0 1 * *',  -- 매월 1일 00:00
    $$SELECT create_monthly_partition('orders', CURRENT_DATE + INTERVAL '2 months')$$
);

-- 오래된 파티션 자동 삭제 (12개월 이전)
SELECT cron.schedule(
    'drop-old-partition',
    '0 1 1 * *',  -- 매월 1일 01:00
    $$DROP TABLE IF EXISTS orders_$$ || TO_CHAR(CURRENT_DATE - INTERVAL '12 months', 'YYYY_MM')
);
```

### 6.5 파티션 정보 조회

```sql
-- 파티션 목록 및 범위 확인
SELECT
    parent.relname AS parent,
    child.relname AS partition,
    pg_get_expr(child.relpartbound, child.oid) AS bounds
FROM pg_inherits
JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
JOIN pg_class child ON pg_inherits.inhrelid = child.oid
WHERE parent.relname = 'orders';

-- 파티션별 행 수
SELECT
    schemaname,
    relname AS partition_name,
    n_live_tup AS row_count
FROM pg_stat_user_tables
WHERE relname LIKE 'orders_%'
ORDER BY relname;

-- 파티션별 크기
SELECT
    child.relname AS partition,
    pg_size_pretty(pg_relation_size(child.oid)) AS size
FROM pg_inherits
JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
JOIN pg_class child ON pg_inherits.inhrelid = child.oid
WHERE parent.relname = 'orders'
ORDER BY child.relname;
```

### 6.6 기존 테이블 파티션 변환

```sql
-- 1. 새 파티션 테이블 생성
CREATE TABLE orders_new (LIKE orders INCLUDING ALL)
    PARTITION BY RANGE (order_date);

-- 2. 파티션 생성
CREATE TABLE orders_new_2024_01 PARTITION OF orders_new
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
-- ... 필요한 파티션 생성

-- 3. 데이터 마이그레이션
INSERT INTO orders_new SELECT * FROM orders;

-- 4. 테이블 교체 (다운타임 최소화)
BEGIN;
ALTER TABLE orders RENAME TO orders_old;
ALTER TABLE orders_new RENAME TO orders;
COMMIT;

-- 5. 확인 후 이전 테이블 삭제
DROP TABLE orders_old;
```

---

## 7. 연습 문제

### 연습 1: 월별 로그 파티셔닝
access_logs 테이블을 월별로 파티셔닝하세요.

```sql
-- 예시 답안
CREATE TABLE access_logs (
    id BIGSERIAL,
    user_id INT,
    action VARCHAR(50),
    ip_address INET,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- 2024년 월별 파티션
DO $$
DECLARE
    start_date DATE := '2024-01-01';
BEGIN
    FOR i IN 0..11 LOOP
        EXECUTE format(
            'CREATE TABLE access_logs_%s PARTITION OF access_logs
             FOR VALUES FROM (%L) TO (%L)',
            TO_CHAR(start_date + (i || ' months')::interval, 'YYYY_MM'),
            start_date + (i || ' months')::interval,
            start_date + ((i+1) || ' months')::interval
        );
    END LOOP;
END;
$$;
```

### 연습 2: 지역별 주문 파티셔닝
국가 코드 기반으로 주문을 파티셔닝하세요.

```sql
-- 예시 답안
CREATE TABLE regional_orders (
    id BIGSERIAL,
    country_code CHAR(2) NOT NULL,
    customer_id INT,
    total NUMERIC(10,2),
    order_date TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (id, country_code)
) PARTITION BY LIST (country_code);

CREATE TABLE regional_orders_kr PARTITION OF regional_orders
    FOR VALUES IN ('KR');
CREATE TABLE regional_orders_us PARTITION OF regional_orders
    FOR VALUES IN ('US');
CREATE TABLE regional_orders_others PARTITION OF regional_orders DEFAULT;
```

### 연습 3: 파티션 유지보수 쿼리
90일 이전 데이터가 있는 파티션을 식별하고 처리하는 쿼리를 작성하세요.

```sql
-- 예시 답안: 오래된 파티션 식별
WITH partition_info AS (
    SELECT
        child.relname AS partition_name,
        pg_get_expr(child.relpartbound, child.oid) AS bounds,
        (regexp_match(
            pg_get_expr(child.relpartbound, child.oid),
            $$FROM \('([^']+)'\)$$
        ))[1]::date AS start_date
    FROM pg_inherits
    JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
    JOIN pg_class child ON pg_inherits.inhrelid = child.oid
    WHERE parent.relname = 'orders'
      AND child.relname != 'orders_default'
)
SELECT *
FROM partition_info
WHERE start_date < CURRENT_DATE - INTERVAL '90 days';
```

---

**이전**: [윈도우 함수](./17_Window_Functions.md) | **다음**: [전문 검색](./19_Full_Text_Search.md)

## 참고 자료
- [PostgreSQL Table Partitioning](https://www.postgresql.org/docs/current/ddl-partitioning.html)
- [Partition Pruning](https://www.postgresql.org/docs/current/ddl-partitioning.html#DDL-PARTITION-PRUNING)
- [pg_partman Extension](https://github.com/pgpartman/pg_partman)
- [Best Practices for Partitioning](https://www.postgresql.org/docs/current/ddl-partitioning.html#DDL-PARTITIONING-OVERVIEW)
