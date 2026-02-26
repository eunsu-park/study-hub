# 데이터 모델링 기초

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 차원 모델링(Dimensional Modeling)의 개념을 설명하고, 팩트 테이블(Fact Table)과 디멘전 테이블(Dimension Table)을 구분할 수 있다
2. 스타 스키마(Star Schema)와 스노우플레이크 스키마(Snowflake Schema) 구조를 설계하고 SQL로 구현할 수 있다
3. 천천히 변하는 차원(SCD, Slowly Changing Dimension) 전략을 적용하여 이력 데이터 변경을 처리할 수 있다
4. 날짜 차원(Date Dimension)과 대리 키(Surrogate Key) 등 일반적인 디멘전 테이블 패턴을 구현할 수 있다
5. 차원 모델링(Dimensional Modeling)과 데이터 볼트(Data Vault) 모델링을 비교하고 상황에 적합한 방식을 선택할 수 있다
6. 분석용 데이터 모델에서 정규화(Normalization)와 비정규화(Denormalization)의 트레이드오프를 평가할 수 있다

---

## 개요

데이터 모델링은 데이터의 구조, 관계, 제약 조건을 정의하는 과정입니다. 데이터 웨어하우스와 분석 시스템에서는 차원 모델링(Dimensional Modeling)이 널리 사용됩니다.

---

## 1. 차원 모델링 (Dimensional Modeling)

### 1.1 차원 모델링 개념

차원 모델링은 비즈니스 프로세스를 **팩트(Fact)**와 **디멘전(Dimension)**으로 분리하여 모델링하는 기법입니다.

```
┌──────────────────────────────────────────────────────────────┐
│                    차원 모델링 구조                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐                                           │
│   │  Dimension   │  WHO, WHAT, WHERE, WHEN, HOW              │
│   │   (차원)     │  - Customer (누가)                         │
│   │              │  - Product (무엇을)                        │
│   │              │  - Location (어디서)                       │
│   │              │  - Time (언제)                             │
│   └──────┬───────┘                                           │
│          │                                                   │
│          ↓                                                   │
│   ┌──────────────┐                                           │
│   │    Fact      │  MEASURES (측정값)                         │
│   │   (팩트)     │  - Sales Amount (판매금액)                  │
│   │              │  - Quantity (수량)                         │
│   │              │  - Profit (이익)                           │
│   └──────────────┘                                           │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 팩트 vs 디멘전

| 구분 | 팩트 테이블 | 디멘전 테이블 |
|------|------------|--------------|
| **내용** | 측정 가능한 수치 데이터 | 설명적 속성 데이터 |
| **예시** | 판매금액, 수량, 이익 | 고객명, 상품명, 날짜 |
| **레코드 수** | 매우 많음 (수억 건) | 상대적으로 적음 |
| **변경 빈도** | 계속 추가됨 | 가끔 변경됨 |
| **분석 역할** | 집계 대상 | 필터/그룹 기준 |

---

## 2. 스타 스키마 (Star Schema)

### 2.1 스타 스키마 구조

스타 스키마는 중앙에 팩트 테이블이 있고, 주변에 디멘전 테이블이 연결된 형태입니다.

```
                    ┌─────────────────┐
                    │   dim_customer  │
                    │  - customer_sk  │
                    │  - customer_id  │
                    │  - name         │
                    │  - email        │
                    └────────┬────────┘
                             │
┌─────────────────┐          │          ┌─────────────────┐
│   dim_product   │          │          │    dim_date     │
│  - product_sk   │          │          │  - date_sk      │
│  - product_id   │          │          │  - full_date    │
│  - name         │          ↓          │  - year         │
│  - category     │   ┌─────────────┐   │  - quarter      │
│  - price        │───│ fact_sales  │───│  - month        │
└─────────────────┘   │ - date_sk   │   └─────────────────┘
                      │ - customer_sk│
                      │ - product_sk │
                      │ - store_sk   │
┌─────────────────┐   │ - quantity   │
│   dim_store     │   │ - amount     │
│  - store_sk     │   │ - discount   │
│  - store_id     │───└─────────────┘
│  - store_name   │
│  - city         │
└─────────────────┘
```

### 2.2 스타 스키마 SQL 구현

```sql
-- 1. 디멘전 테이블 생성
-- 팩트 테이블의 외래 키 제약 조건이 즉시 참조할 수 있도록
-- 디멘전 테이블을 팩트 테이블보다 먼저 생성한다.

-- 날짜 디멘전: 원시 날짜에 대한 조인을 계산하는 대신 미리 채워진 조회 테이블 —
-- 이렇게 하면 쿼리 시간의 날짜 부분 추출 비용을 피하고,
-- 분석가가 month_name, is_weekend 같은 친숙한 속성으로 필터링할 수 있다.
CREATE TABLE dim_date (
    date_sk         INT PRIMARY KEY,           -- Surrogate Key (빠른 조인을 위한 YYYYMMDD 정수)
    full_date       DATE NOT NULL,
    year            INT NOT NULL,
    quarter         INT NOT NULL,
    month           INT NOT NULL,
    month_name      VARCHAR(20) NOT NULL,
    week            INT NOT NULL,
    day_of_week     INT NOT NULL,
    day_name        VARCHAR(20) NOT NULL,
    is_weekend      BOOLEAN NOT NULL,
    is_holiday      BOOLEAN DEFAULT FALSE      -- 휴일 캘린더에서 채워짐; 플래그로 유지하여
                                               -- BI 쿼리에서 서브쿼리 없이 휴일을 제외할 수 있다
);

-- 고객 디멘전
-- 대리 키(customer_sk)는 웨어하우스를 소스 시스템의 자연 키에서 분리한다 —
-- 소스에서 고객 번호를 재부여해도 기존 팩트 행은 여전히 올바르게 조인된다.
CREATE TABLE dim_customer (
    customer_sk     INT PRIMARY KEY,           -- Surrogate Key (대리 키)
    customer_id     VARCHAR(50) NOT NULL,      -- Natural Key (소스 시스템 키)
    first_name      VARCHAR(100) NOT NULL,
    last_name       VARCHAR(100) NOT NULL,
    email           VARCHAR(200),
    phone           VARCHAR(50),
    city            VARCHAR(100),
    country         VARCHAR(100),
    customer_segment VARCHAR(50),              -- Gold, Silver, Bronze
    created_at      DATE NOT NULL,
    -- SCD Type 2 지원 컬럼: effective_date/end_date는 유효 범위를 형성하여
    -- 하나의 customer_id가 속성 변경을 추적하는 여러 행을 가질 수 있다.
    -- is_current 플래그는 최신 버전을 찾기 위해 모든 행을 스캔하는 것을 피하게 해준다.
    effective_date  DATE NOT NULL,
    end_date        DATE,
    is_current      BOOLEAN DEFAULT TRUE
);

-- 상품 디멘전
CREATE TABLE dim_product (
    product_sk      INT PRIMARY KEY,           -- Surrogate Key
    product_id      VARCHAR(50) NOT NULL,      -- Natural Key
    product_name    VARCHAR(200) NOT NULL,
    category        VARCHAR(100),
    subcategory     VARCHAR(100),
    brand           VARCHAR(100),
    unit_price      DECIMAL(10, 2),
    cost_price      DECIMAL(10, 2),
    -- SCD Type 2 지원 컬럼
    effective_date  DATE NOT NULL,
    end_date        DATE,
    is_current      BOOLEAN DEFAULT TRUE
);

-- 매장 디멘전
CREATE TABLE dim_store (
    store_sk        INT PRIMARY KEY,           -- Surrogate Key
    store_id        VARCHAR(50) NOT NULL,      -- Natural Key
    store_name      VARCHAR(200) NOT NULL,
    store_type      VARCHAR(50),               -- Online, Retail
    city            VARCHAR(100),
    state           VARCHAR(100),
    country         VARCHAR(100),
    region          VARCHAR(50),
    opened_date     DATE
);


-- 2. 팩트 테이블 생성
-- BIGINT PK는 수십억 행을 수용한다; 팩트 테이블은 모든 트랜잭션이
-- 새 행을 생성하므로 디멘전보다 훨씬 빠르게 성장한다.

CREATE TABLE fact_sales (
    sales_sk        BIGINT PRIMARY KEY,        -- Surrogate Key (대리 키)
    -- 디멘전 외래 키: 스타 스키마는 모든 FK를 하나의 팩트 테이블에 유지하여
    -- 대부분의 분석 쿼리가 디멘전당 하나의 조인만 필요하다 (다중 홉 조인 불필요).
    date_sk         INT NOT NULL REFERENCES dim_date(date_sk),
    customer_sk     INT NOT NULL REFERENCES dim_customer(customer_sk),
    product_sk      INT NOT NULL REFERENCES dim_product(product_sk),
    store_sk        INT NOT NULL REFERENCES dim_store(store_sk),
    -- 측정값: 가산(additive) 값(수량, 금액)과 파생(profit) 값 모두 저장.
    -- profit을 미리 계산해 두면 모든 쿼리에서 반복 계산을 피할 수 있다.
    quantity        INT NOT NULL,
    unit_price      DECIMAL(10, 2) NOT NULL,
    discount_amount DECIMAL(10, 2) DEFAULT 0,
    sales_amount    DECIMAL(12, 2) NOT NULL,   -- quantity * unit_price - discount
    cost_amount     DECIMAL(12, 2),
    profit_amount   DECIMAL(12, 2),            -- sales_amount - cost_amount
    -- 메타 데이터
    transaction_id  VARCHAR(50),               -- 계통/감사를 위해 OLTP로 역추적
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 각 FK 컬럼에 개별 인덱스 생성: 분석 쿼리는 일반적으로 하나의 디멘전으로
-- 필터링하거나 그룹화한다 (예: "날짜별 매출" 또는 "상품별 매출").
-- 복합 인덱스는 특정 쿼리 패턴에서만 도움이 된다.
CREATE INDEX idx_fact_sales_date ON fact_sales(date_sk);
CREATE INDEX idx_fact_sales_customer ON fact_sales(customer_sk);
CREATE INDEX idx_fact_sales_product ON fact_sales(product_sk);
CREATE INDEX idx_fact_sales_store ON fact_sales(store_sk);
```

### 2.3 스타 스키마 쿼리 예시

```sql
-- 월별, 카테고리별 매출 집계
-- 스타 스키마의 장점: 각 분석 질문은 디멘전당 하나의 조인만 추가하면 된다 —
-- 여기서 두 개의 조인으로 단일 스캔에서 시간 + 상품 슬라이싱이 가능하다.
SELECT
    d.year,
    d.month,
    d.month_name,
    p.category,
    SUM(f.sales_amount) AS total_sales,
    SUM(f.quantity) AS total_quantity,
    SUM(f.profit_amount) AS total_profit,
    COUNT(DISTINCT f.customer_sk) AS unique_customers
FROM fact_sales f
JOIN dim_date d ON f.date_sk = d.date_sk
JOIN dim_product p ON f.product_sk = p.product_sk
WHERE d.year = 2024
GROUP BY d.year, d.month, d.month_name, p.category
ORDER BY d.year, d.month, total_sales DESC;


-- 지역별 상위 10개 상품
-- QUALIFY는 Snowflake/BigQuery 확장으로 윈도우 함수 결과를 필터링 —
-- 순위로 필터링하기 위해 CTE로 감쌀 필요가 없다.
SELECT
    s.region,
    p.product_name,
    SUM(f.sales_amount) AS total_sales,
    RANK() OVER (PARTITION BY s.region ORDER BY SUM(f.sales_amount) DESC) AS rank
FROM fact_sales f
JOIN dim_store s ON f.store_sk = s.store_sk
JOIN dim_product p ON f.product_sk = p.product_sk
GROUP BY s.region, p.product_name
QUALIFY rank <= 10;


-- 고객 세그먼트별 구매 패턴
-- is_current = TRUE로 필터링: SCD Type 2에서 고객은 여러 행을 가질 수 있다;
-- 각 고객을 *최신* 세그먼트 아래에서 한 번만 집계하기 위해 필요하다.
SELECT
    c.customer_segment,
    COUNT(DISTINCT f.customer_sk) AS customer_count,
    AVG(f.sales_amount) AS avg_order_value,
    SUM(f.sales_amount) / COUNT(DISTINCT f.customer_sk) AS revenue_per_customer
FROM fact_sales f
JOIN dim_customer c ON f.customer_sk = c.customer_sk
WHERE c.is_current = TRUE
GROUP BY c.customer_segment
ORDER BY revenue_per_customer DESC;
```

---

## 3. 스노우플레이크 스키마 (Snowflake Schema)

### 3.1 스노우플레이크 스키마 구조

디멘전 테이블을 정규화하여 중복을 제거한 형태입니다.

```
┌──────────────┐
│ dim_category │
│ - category_sk│
│ - category   │
└──────┬───────┘
       │
       ↓
┌──────────────┐     ┌──────────────┐
│dim_subcategory│    │  dim_brand   │
│-subcategory_sk│    │ - brand_sk   │
│- category_sk │     │ - brand_name │
│- subcategory │     └──────┬───────┘
└──────┬───────┘            │
       │                    │
       └──────────┬─────────┘
                  ↓
          ┌─────────────┐
          │ dim_product │
          │- product_sk │
          │-subcategory_sk
          │- brand_sk   │────→ ┌─────────────┐
          │- product_name      │ fact_sales  │
          └─────────────┘      └─────────────┘
```

### 3.2 스노우플레이크 vs 스타 스키마

| 특성 | 스타 스키마 | 스노우플레이크 스키마 |
|------|------------|---------------------|
| **정규화** | 비정규화 | 정규화 |
| **저장 공간** | 더 많음 | 더 적음 |
| **쿼리 성능** | 더 빠름 (조인 적음) | 더 느림 (조인 많음) |
| **유지보수** | 중복 관리 필요 | 관리 용이 |
| **복잡성** | 단순 | 복잡 |
| **권장 사용** | OLAP, 분석 | 저장 공간 제한 시 |

---

## 4. 팩트 테이블 유형

### 4.1 트랜잭션 팩트 (Transaction Fact)

개별 트랜잭션을 기록합니다. 가장 일반적인 형태입니다.

```sql
-- 트랜잭션 팩트 예시: 개별 주문
CREATE TABLE fact_order_line (
    order_line_sk   BIGINT PRIMARY KEY,
    date_sk         INT NOT NULL,
    customer_sk     INT NOT NULL,
    product_sk      INT NOT NULL,
    order_id        VARCHAR(50) NOT NULL,
    line_number     INT NOT NULL,
    quantity        INT NOT NULL,
    unit_price      DECIMAL(10, 2) NOT NULL,
    line_amount     DECIMAL(12, 2) NOT NULL
);
```

### 4.2 주기적 스냅샷 팩트 (Periodic Snapshot Fact)

일정 기간의 집계 데이터를 기록합니다.

```sql
-- 주기적 스냅샷: 일일 재고 현황
CREATE TABLE fact_daily_inventory (
    inventory_sk    BIGINT PRIMARY KEY,
    date_sk         INT NOT NULL,
    product_sk      INT NOT NULL,
    warehouse_sk    INT NOT NULL,
    -- 스냅샷 측정값
    quantity_on_hand INT NOT NULL,
    quantity_reserved INT DEFAULT 0,
    quantity_available INT NOT NULL,
    days_of_supply  INT,
    inventory_value DECIMAL(12, 2)
);


-- 일일 계정 잔액 스냅샷
CREATE TABLE fact_daily_account_balance (
    balance_sk      BIGINT PRIMARY KEY,
    date_sk         INT NOT NULL,
    account_sk      INT NOT NULL,
    customer_sk     INT NOT NULL,
    opening_balance DECIMAL(15, 2) NOT NULL,
    total_credits   DECIMAL(15, 2) DEFAULT 0,
    total_debits    DECIMAL(15, 2) DEFAULT 0,
    closing_balance DECIMAL(15, 2) NOT NULL
);
```

### 4.3 누적 스냅샷 팩트 (Accumulating Snapshot Fact)

프로세스의 시작부터 종료까지 추적합니다.

```sql
-- 누적 스냅샷: 주문이 각 마일스톤을 통과할 때 동일한 행을 업데이트하여
-- 다단계 프로세스를 추적한다. 이벤트당 하나의 불변 행을 갖는 트랜잭션 팩트와
-- 다른 점은 기존 행을 *업데이트*하므로 단계 간 리드 타임 측정이 쉽다.
CREATE TABLE fact_order_fulfillment (
    order_fulfillment_sk BIGINT PRIMARY KEY,
    order_id        VARCHAR(50) UNIQUE NOT NULL,

    -- 마일스톤 날짜 FK: NULL 허용 — 이후 단계가 아직 발생하지 않았기 때문.
    -- NULL ship_date_sk는 주문이 아직 배송되지 않았음을 의미 — SLA 모니터링에
    -- 유용하다 (예: "3일 전 주문 중 ship_date_sk가 NULL인 것").
    order_date_sk       INT NOT NULL,
    payment_date_sk     INT,
    ship_date_sk        INT,
    delivery_date_sk    INT,

    -- 디멘전 외래 키
    customer_sk     INT NOT NULL,
    product_sk      INT NOT NULL,
    warehouse_sk    INT,
    carrier_sk      INT,

    -- 측정값
    order_amount    DECIMAL(12, 2) NOT NULL,
    shipping_cost   DECIMAL(10, 2),

    -- 미리 계산된 리드 타임은 쿼리 시간의 날짜 계산을 피하고
    -- 평균 리드 타임 대시보드를 단순한 AVG() 집계로 만든다.
    days_to_payment     INT,  -- order -> payment
    days_to_ship        INT,  -- payment -> ship
    days_to_delivery    INT,  -- ship -> delivery
    total_lead_time     INT   -- order -> delivery
);
```

---

## 5. SCD (Slowly Changing Dimensions)

### 5.1 SCD 유형 개요

| 유형 | 설명 | 히스토리 | 사용 사례 |
|------|------|----------|----------|
| **Type 0** | 변경 안 함 | 없음 | 고정 속성 (생년월일) |
| **Type 1** | 덮어쓰기 | 없음 | 오류 수정, 히스토리 불필요 |
| **Type 2** | 새 행 추가 | 전체 보관 | 가격 변경, 주소 변경 |
| **Type 3** | 컬럼 추가 | 이전 값만 | 제한적 히스토리 필요 |
| **Type 4** | 히스토리 테이블 분리 | 전체 보관 | 자주 변경되는 속성 |

### 5.2 SCD Type 1: 덮어쓰기

```sql
-- SCD Type 1: 기존 값 덮어쓰기 (히스토리 없음)
UPDATE dim_customer
SET
    email = 'new_email@example.com',
    phone = '010-1234-5678'
WHERE customer_id = 'C001';
```

### 5.3 SCD Type 2: 새 행 추가

```python
# SCD Type 2 구현 예시
import pandas as pd
from datetime import date

def scd_type2_update(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    natural_key: str,
    tracked_columns: list[str]
) -> pd.DataFrame:
    """SCD Type 2 업데이트 로직"""

    today = date.today()
    result_rows = []

    for _, source_row in source_df.iterrows():
        # is_current로 필터링하여 만료된 이력 행을 매칭하지 않도록 —
        # 각 엔티티의 최신 버전만 비교 대상이 되어야 한다.
        current_mask = (
            (target_df[natural_key] == source_row[natural_key]) &
            (target_df['is_current'] == True)
        )
        current_record = target_df[current_mask]

        if current_record.empty:
            # 완전히 새로운 엔티티 — 개방형 유효 기간으로 삽입
            new_row = source_row.copy()
            new_row['effective_date'] = today
            new_row['end_date'] = None
            new_row['is_current'] = True
            result_rows.append(new_row)
        else:
            # tracked_columns만 비교: 일부 속성(예: last_login)은 자주 변경되지만
            # 새 SCD 행을 만들 필요는 없다.
            current_row = current_record.iloc[0]
            has_changes = False

            for col in tracked_columns:
                if current_row[col] != source_row[col]:
                    has_changes = True
                    break

            if has_changes:
                # 기존 행을 삭제하지 않고 만료(expire) 처리 — 이렇게 하면 전체
                # 이력 체인이 보존되어 과거 어느 시점에서도 조인이 가능하다.
                target_df.loc[current_mask, 'end_date'] = today
                target_df.loc[current_mask, 'is_current'] = False

                # 업데이트된 속성 값으로 새 "현재" 행 삽입.
                # 새 행의 대리 키는 다르므로 변경 *이전*에 기록된 팩트 행은
                # 여전히 이전 속성으로 조인된다.
                new_row = source_row.copy()
                new_row['effective_date'] = today
                new_row['end_date'] = None
                new_row['is_current'] = True
                result_rows.append(new_row)

    # 새 레코드 추가
    if result_rows:
        new_records = pd.DataFrame(result_rows)
        target_df = pd.concat([target_df, new_records], ignore_index=True)

    return target_df


# 사용 예시
"""
-- SQL로 SCD Type 2 구현
-- 1. 변경된 레코드 만료
UPDATE dim_customer
SET
    end_date = CURRENT_DATE,
    is_current = FALSE
WHERE customer_id IN (
    SELECT customer_id FROM staging_customer
    WHERE customer_id IN (SELECT customer_id FROM dim_customer WHERE is_current = TRUE)
    AND (email != (SELECT email FROM dim_customer d WHERE d.customer_id = staging_customer.customer_id AND d.is_current = TRUE)
         OR phone != (SELECT phone FROM dim_customer d WHERE d.customer_id = staging_customer.customer_id AND d.is_current = TRUE))
);

-- 2. 새 레코드 삽입
INSERT INTO dim_customer (customer_id, email, phone, effective_date, end_date, is_current)
SELECT
    customer_id,
    email,
    phone,
    CURRENT_DATE,
    NULL,
    TRUE
FROM staging_customer
WHERE customer_id IN (
    SELECT customer_id FROM dim_customer WHERE is_current = FALSE AND end_date = CURRENT_DATE
);
"""
```

### 5.4 SCD Type 2 SQL 구현

```sql
-- 두 단계 UPDATE + INSERT 방식의 SCD Type 2 (PostgreSQL 15+)
-- 단일 MERGE 대신 두 단계(UPDATE 후 INSERT) 방식: 각 단계를 독립적으로
-- 검증할 수 있어 감사 및 디버깅이 더 쉽다.
WITH changes AS (
    -- CTE는 "무엇이 변경되었는가"와 "어떻게 처리할 것인가"를 분리 —
    -- WHERE 절은 추적 대상 컬럼만 나열하여 추적하지 않는 변경은 무시된다.
    SELECT
        s.customer_id,
        s.email,
        s.phone,
        s.city
    FROM staging_customer s
    JOIN dim_customer d ON s.customer_id = d.customer_id AND d.is_current = TRUE
    WHERE s.email != d.email OR s.phone != d.phone OR s.city != d.city
)
-- 1단계: 기존 레코드 만료
-- end_date를 어제로 설정하여 새 행의 effective_date(오늘)와 겹치지 않도록 한다.
-- 이렇게 하면 시점 쿼리(point-in-time query)가 명확해진다.
UPDATE dim_customer
SET
    end_date = CURRENT_DATE - INTERVAL '1 day',
    is_current = FALSE
FROM changes
WHERE dim_customer.customer_id = changes.customer_id
  AND dim_customer.is_current = TRUE;

-- 2단계: 변경된 레코드의 새 "현재" 버전 삽입
INSERT INTO dim_customer (
    customer_id, email, phone, city,
    effective_date, end_date, is_current
)
SELECT
    customer_id, email, phone, city,
    CURRENT_DATE, NULL, TRUE
FROM staging_customer
WHERE customer_id IN (
    -- 방금 만료된 행에 재조인하여 실제로 변경된 레코드에만 삽입하고,
    -- 모든 스테이징 레코드에 삽입하지 않도록 보장한다.
    SELECT customer_id FROM dim_customer
    WHERE end_date = CURRENT_DATE - INTERVAL '1 day'
);
```

---

## 6. 디멘전 테이블 설계 패턴

### 6.1 날짜 디멘전 생성

```python
import pandas as pd
from datetime import date, timedelta

def generate_date_dimension(start_date: str, end_date: str) -> pd.DataFrame:
    """날짜 디멘전 테이블 생성"""

    # 10년 이상의 넓은 날짜 범위를 미리 생성하여 일반 운영 중
    # 테이블 확장이 필요 없도록 한다 — 팩트 행이 미래 날짜(예: 예약 배송)를
    # 참조할 경우 FK 위반을 방지한다.
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    records = []
    for i, d in enumerate(date_range):
        record = {
            # date_sk를 YYYYMMDD 정수로: 사람이 읽기 쉽고 대부분의 열 지향 엔진에서
            # DATE 컬럼보다 조인이 더 빠르다.
            'date_sk': int(d.strftime('%Y%m%d')),
            'full_date': d.date(),
            'year': d.year,
            'quarter': (d.month - 1) // 3 + 1,
            'month': d.month,
            'month_name': d.strftime('%B'),
            'week': d.isocalendar()[1],
            'day_of_week': d.weekday() + 1,  # 1=Monday (ISO 규약)
            'day_name': d.strftime('%A'),
            'day_of_month': d.day,
            'day_of_year': d.timetuple().tm_yday,
            'is_weekend': d.weekday() >= 5,
            'is_month_start': d.day == 1,
            'is_month_end': (d + timedelta(days=1)).day == 1,
            # 회계연도는 4월 시작 가정 — 조직의 회계 캘린더에 맞게 이 상수를 조정한다.
            'fiscal_year': d.year if d.month >= 4 else d.year - 1,
            'fiscal_quarter': ((d.month - 4) % 12) // 3 + 1
        }
        records.append(record)

    return pd.DataFrame(records)


# 사용 예시: 11년 범위는 이력 백필 + 몇 년 앞의 미래를 커버한다
date_dim = generate_date_dimension('2020-01-01', '2030-12-31')
print(date_dim.head())
```

### 6.2 정크 디멘전 (Junk Dimension)

여러 저-카디널리티 플래그/상태를 하나의 디멘전으로 통합합니다.

```sql
-- 정크 디멘전(Junk Dimension): 저-카디널리티 플래그를 하나의 테이블로 통합하여
-- 팩트 테이블에 많은 좁은 Boolean/열거형 컬럼이 생기는 것을 방지한다. 정크 디멘전 없이는
-- 각 플래그가 팩트 행을 복잡하게 하거나 자체적인 작은 디멘전 테이블이 필요해 — 둘 다 낭비적이다.
CREATE TABLE dim_order_flags (
    order_flags_sk  INT PRIMARY KEY,
    is_gift_wrapped BOOLEAN,
    is_expedited    BOOLEAN,
    is_return       BOOLEAN,
    payment_method  VARCHAR(20),  -- Credit, Debit, Cash, PayPal
    order_channel   VARCHAR(20)   -- Web, Mobile, Store, Phone
);

-- 모든 조합을 미리 생성 (카르테시안 곱):
-- 2 * 2 * 2 * 4 * 4 = 128행 — 메모리/캐시에 충분히 들어가므로
-- order_flags_sk로 조인하는 비용이 사실상 없다. 새 플래그 값
-- (예: 5번째 결제 방법)은 이 테이블 재생성을 필요로 한다.
INSERT INTO dim_order_flags (order_flags_sk, is_gift_wrapped, is_expedited, is_return, payment_method, order_channel)
SELECT
    ROW_NUMBER() OVER () as order_flags_sk,
    gift, expedited, return_flag, payment, channel
FROM
    (VALUES (TRUE), (FALSE)) AS gift(gift),
    (VALUES (TRUE), (FALSE)) AS expedited(expedited),
    (VALUES (TRUE), (FALSE)) AS return_flag(return_flag),
    (VALUES ('Credit'), ('Debit'), ('Cash'), ('PayPal')) AS payment(payment),
    (VALUES ('Web'), ('Mobile'), ('Store'), ('Phone')) AS channel(channel);
```

---

## 연습 문제

### 문제 1: 스타 스키마 설계
온라인 서점의 판매 분석을 위한 스타 스키마를 설계하세요. 필요한 팩트 테이블과 디멘전 테이블을 정의하세요.

### 문제 2: SCD Type 2
고객의 등급(Bronze, Silver, Gold)이 변경될 때 히스토리를 보관하는 SCD Type 2 SQL을 작성하세요.

---

## 요약

| 개념 | 설명 |
|------|------|
| **차원 모델링** | 팩트와 디멘전으로 데이터 구조화 |
| **스타 스키마** | 비정규화된 디멘전, 빠른 쿼리 |
| **스노우플레이크** | 정규화된 디멘전, 저장 공간 절약 |
| **팩트 테이블** | 측정 가능한 수치 데이터 저장 |
| **디멘전 테이블** | 설명적 속성 데이터 저장 |
| **SCD** | 디멘전 변경 이력 관리 전략 |

---

## 참고 자료

- [The Data Warehouse Toolkit (Kimball)](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/books/data-warehouse-dw-toolkit/)
- [Dimensional Modeling Techniques](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/)
