[← 이전: 1. 데이터 엔지니어링 개요](01_Data_Engineering_Overview.md) | [다음: 3. ETL vs ELT →](03_ETL_vs_ELT.md)

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

## 이론과 원리

데이터 모델링은 기법(스타 스키마, SCD Type 2)이 *왜* 그렇게 하는지보다 훨씬 먼저 가르쳐지는 주제 중 하나입니다. 그 "왜"는 단 하나의 아키텍처 결정입니다: 트랜잭션(OLTP)을 최적화할 것인가, 분석(OLAP)을 최적화할 것인가? 이 선택이 정규화/비정규화 여부, 행 저장(row-store)/컬럼 저장(column-store) 여부, 그리고 차원 모델링이 적용되는지조차 결정합니다.

- **(A) OLTP vs OLAP** — 동일한 비즈니스 데이터에 대해 별도의 모델을 정당화하는 워크로드 이중성
- **(B) 정규화 vs 비정규화** — Codd의 정규형 vs Kimball의 스타 스키마, 각각이 이기는 시점
- **(C) 차원 모델링(Dimensional Modeling)** — 측정으로서의 팩트(Fact), 맥락으로서의 차원(Dimension), 그리고 스타 vs 스노우플레이크의 기하학
- **(D) 천천히 변하는 차원(Slowly Changing Dimensions)** — 시간에 따라 차원 속성이 변한다는 사실을 처리하는 6가지 표준 방법

### A. OLTP vs OLAP: 두 워크로드, 두 모델

동일한 비즈니스 데이터 — 고객, 주문, 상품 — 가 완전히 다른 두 가지 접근 패턴으로 쿼리됩니다.

#### A.1 OLTP (Online Transaction Processing)

운영 시스템. 예: 전자상거래 결제, 은행 송금, 티켓 예매.

- **워크로드:** 동시 사용자가 많고, 각각 몇 행만 건드림. "주문 1개 삽입, 재고 행 2개 업데이트, 결제 기록 1개 작성."
- **지연 시간:** 밀리초. 사용자가 화면을 보고 있음.
- **동시성(Concurrency):** 수백 ~ 수천의 동시 트랜잭션.
- **저장 레이아웃:** 행 저장(row-store: PostgreSQL, MySQL, SQL Server) — 한 행의 모든 컬럼이 함께 저장되므로 "이 주문 한 건"을 읽는 것은 단일 디스크 시크.
- **스키마:** 고도로 정규화됨(3NF 또는 BCNF) — 중복 최소화, 모든 사실은 한 번만 저장, 외래 키로 무결성 강제.

정규화된 스키마는 쓰기를 빠르고 일관되게 만듭니다: 새 주문 삽입은 `orders`의 한 행을 건드리고 재고 카운터 하나를 증가시킵니다. 데이터 중복이 없으므로 불일치 위험이 없습니다.

#### A.2 OLAP (Online Analytical Processing)

분석 시스템. 예: "월별 상품 카테고리·지역별 매출", "전년 대비 성장률", "고객 코호트(Cohort) 잔존율".

- **워크로드:** 분석가가 몇 명, 각자 수백만 ~ 수십억 행을 건드리는 쿼리 실행. "3년치 매출을 상품별로 합산."
- **지연 시간:** 초 ~ 분. 분석가가 임시 리포트를 실행.
- **동시성:** 수십 개의 동시 쿼리.
- **저장 레이아웃:** 컬럼 저장(column-store: Snowflake, BigQuery, Redshift, ClickHouse, Parquet) — 단일 컬럼의 모든 값이 연속적으로 저장되므로, 50개 중 3개 컬럼만 건드리는 쿼리는 데이터의 6%만 읽음.
- **스키마:** 비정규화(스타 스키마) — 대규모에서 조인은 비싸므로, 차원 속성을 팩트 테이블에 중복하거나 미리 조인된 와이드 테이블을 사용.

컬럼 저장소는 압축으로 우위를 가중시킵니다: `country_code` 컬럼은 ~200개의 고유값만 가지므로, 사전 인코딩(dictionary encoding) + RLE(run-length encoding)로 50-100배 줄어듭니다. Spark/Snowflake는 또한 *predicate pushdown(술어 푸시다운)* 을 수행합니다: 옵티마이저가 `WHERE country = 'US'` 필터를 파일 포맷에 밀어 넣어 매치되지 않는 컬럼은 압축 해제조차 되지 않습니다.

#### A.3 왜 두 모델인가, 하나가 아니라

단일 정규화 OLTP 스키마가 두 워크로드를 모두 처리할 수 있을까요? 원칙적으로는 가능하지만 — 실무에서는 두 가지 이유로 불가능합니다:

1. **성능.** "3년간 상품별 매출 합산" 같은 쿼리를 정규화 OLTP 스키마에 실행하면 `orders + line_items + products + categories`를 조인해야 하고, 행 저장 레이아웃에서 수십억 행을 스캔합니다. 같은 쿼리를 비정규화된 컬럼 팩트 테이블에 실행하면 1개 팩트 테이블의 3개 컬럼만 읽음 — 자릿수 단위로 더 빠릅니다.
2. **워크로드 격리.** 수십억 행을 스캔하는 OLAP 쿼리는 OLTP 트랜잭션을 막아 운영 시스템을 다운시킬 것입니다.

표준 아키텍처: OLTP 데이터베이스가 진실의 원천이고, ETL/ELT 파이프라인(레슨 3, 12)이 데이터를 분석용으로 모델링된 별도의 OLAP 웨어하우스로 옮깁니다.

### B. 정규화 vs 비정규화

Codd의 정규형(1NF부터 BCNF까지)은 중복을 제거합니다. 각 사실은 정확히 한 번 저장되며, 업데이트는 한 행만 건드립니다. 이것은 OLTP에 옳은 선택입니다.

OLAP의 경우, Kimball은 트레이드오프를 뒤집었습니다: 저장은 싸고 조인은 비싸므로, *비정규화하라*. 모든 주문 행에 고객의 이름과 도시를 중복; 모든 라인 아이템에 상품의 카테고리를 중복. 쿼리는 단일 테이블 스캔이나 작은 차원 테이블에 대한 단일 조인이 됩니다.

| 속성 | 정규화 (3NF) | 비정규화 (스타) |
|------|--------------|------------------|
| 저장 | 최소 | 더 큼 (중복) |
| 쓰기 비용 | 낮음 (한 행 업데이트) | 높음 (중복된 모든 사본을 업데이트해야 함) |
| 분석 읽기 비용 | 높음 (많은 조인) | 낮음 (조인 1-2개) |
| 갱신 이상(Update Anomaly) | 불가능 (단일 원천) | 가능 (업데이트 불완전 시) |
| 적합한 용도 | OLTP | OLAP |

비대칭성: OLAP에서 쓰기는 ETL 배치당 한 번 발생합니다(멱등 덮어쓰기 또는 upsert). 읽기는 수천 번 발생합니다. 쓰기를 희생하여 읽기를 최적화하는 것이 옳은 거래입니다.

### C. 차원 모델링

Kimball의 차원 모델은 웨어하우스 스키마 설계의 표준입니다.

#### C.1 팩트와 차원

- **팩트 테이블(Fact table)** 은 비즈니스 프로세스의 *측정값* 을 저장합니다: 판매 단위 수, 매출, 통화 시간. 행은 불변 이벤트이고, 컬럼은 대부분 숫자형.
- **차원 테이블(Dimension table)** 은 그 측정값에 대한 *맥락* 을 저장합니다: 누가(고객), 무엇을(상품), 어디서(매장), 언제(날짜). 행은 엔티티이고, 컬럼은 서술적 속성.

팩트 테이블은 각 차원을 가리키는 외래 키를 가집니다. "판매" 팩트는 고객, 상품, 매장, 날짜에 대한 FK를 가집니다 — "고객 X가 상품 Y를 매장 Z에서 날짜 D에 $W로 구매" 라는 답을 줍니다.

#### C.2 스타 스키마(Star Schema)

```
        ┌──────────────┐
        │  dim_date    │
        └──────┬───────┘
               │
┌────────┐    ┌─▼────────┐    ┌──────────┐
│dim_cust│───▶│fact_sales│◀───│dim_product│
└────────┘    └─┬────────┘    └──────────┘
                │
        ┌───────▼──────┐
        │  dim_store   │
        └──────────────┘
```

중심에 팩트, 그 주위를 차원이 별처럼 둘러쌉니다. 각 차원은 완전히 비정규화 — 상품 차원은 category_name, subcategory_name, brand_name을 추가 정규화 테이블이 아닌 컬럼으로 가집니다. 쿼리는 팩트에서 필요한 각 차원으로의 단일 조인입니다.

#### C.3 스노우플레이크 스키마(Snowflake Schema)

같은 아이디어이지만 차원이 정규화됨: `dim_product`가 `dim_category`를 참조하고, 이는 `dim_department`를 참조합니다. 쿼리 단순성을 저장 효율성과 더 쉬운 차원 업데이트와 맞바꿉니다. 대부분의 현대 웨어하우스(Snowflake, BigQuery)는 스타를 선호합니다 — 저장은 싸고, 옵티마이저가 와이드 테이블을 잘 처리하며, 쿼리가 더 단순합니다.

#### C.4 팩트 그레인(Granularity)

가장 중요한 결정 하나: 각 팩트 행이 어떤 그레인(grain)에 사는가?

- **트랜잭션 그레인:** 이벤트당 한 행 (주문의 라인 아이템당 한 행). 가장 유연.
- **주기적 스냅샷:** 엔티티당 기간당 한 행 (계좌별·일별 잔액).
- **누적 스냅샷(Accumulating snapshot):** 프로세스 인스턴스당 한 행, 각 마일스톤의 타임스탬프 포함 (주문당 한 행, `placed_at`, `paid_at`, `shipped_at`, `delivered_at` 포함).

항상 가지고 있는 *가장 세밀한* 그레인으로 모델링하세요. 위로 집계하는 것은 항상 가능하지만, 분해하는 것은 불가능합니다.

### D. 천천히 변하는 차원 (Slowly Changing Dimensions)

차원 속성은 시간에 따라 변합니다 — 고객이 도시를 옮기고, 상품이 새 카테고리를 받습니다. 모든 팩트 행을 다시 쓰지 않고 어떻게 이력을 보존할까요? 6가지 표준 SCD 타입:

| 타입 | 동작 | 트레이드오프 |
|------|------|--------------|
| **Type 0** | 절대 변경 안 함 (예: 생년월일) | 단순; 진정 불변 속성에만 |
| **Type 1** | 덮어쓰기, 이력 없음 | 가장 단순; 과거 손실 — 팩트가 새 값으로 보임 |
| **Type 2** | 변경마다 새 행 + `valid_from` / `valid_to` / `is_current` | 완전한 이력; 팩트 행은 팩트 시점에 활성이었던 버전을 참조 |
| **Type 3** | `previous_value` 컬럼 추가 | 제한된 이력 (이전 값 1개만); 단순 |
| **Type 4** | 이력을 별도 테이블로 이동 | 현재 차원이 작게 유지; 이력이 필요한 쿼리는 이력 테이블 조인 |
| **Type 6** | Type 1 + 2 + 3 하이브리드 | 가장 유연, 가장 복잡 |

Type 2가 주력입니다. 팩트 테이블의 차원 FK는 팩트 이벤트 시점에 활성이었던 버전의 *대리 키(surrogate key)* 를 참조하므로, 2022년 판매는 고객이 2023년에 이사하더라도 영원히 2022년 고객 도시를 보여줍니다.

#### D.1 대리 키 원칙

차원 테이블은 자연 비즈니스 키(소스 시스템의 customer_id)가 아닌 *대리 키(surrogate key)* — 의미 없는 정수 — 를 기본 키로 사용합니다. 왜?

1. **SCD Type 2가 필요로 함.** 자연 키가 PK라면 동일 고객에 대해 여러 행을 가질 수 없습니다. 대리 키는 한 고객이 여러 행(과거 버전당 하나)을 가질 수 있게 합니다.
2. **웨어하우스를 소스로부터 분리.** 소스 시스템의 customer_id 형식이 바뀔 수 있지만, 대리 키는 영원히 안정적.
3. **더 작은 팩트 테이블.** 대리 키는 4바이트 정수; 자연 키는 36바이트 UUID 문자열일 수 있음.

### From Theory to the Practice Below

이어지는 각 절은 위 프레임워크의 한 조각을 운영합니다:

- §1 (차원 모델링)은 §C — 팩트, 차원, 스타 스키마 — 를 운영합니다.
- §2 (스타 vs 스노우플레이크)는 구체적 스키마로 본 §C.2 vs §C.3입니다.
- §3 (천천히 변하는 차원)은 SQL로 본 §D — Type 1, 2, 3 구현입니다.
- §4 (공통 패턴) — 날짜 차원, 대리 키, 정크 차원(junk dimension) — 은 §D.1과 §C를 실무에 적용합니다.
- §5 (Data Vault)는 감사 가능성/규제가 높은 환경에서 §C의 대안입니다.
- §6 (정규화 트레이드오프)는 §B — 워크로드별로 3NF vs 스타 vs Data Vault 선택입니다.

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

---

[← 이전: 1. 데이터 엔지니어링 개요](01_Data_Engineering_Overview.md) | [다음: 3. ETL vs ELT →](03_ETL_vs_ELT.md)
