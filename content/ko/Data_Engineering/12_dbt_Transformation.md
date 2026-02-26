# dbt 변환 도구

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. ELT 파이프라인에서 dbt의 역할과 전통적인 ETL 변환 도구와의 차이점을 설명할 수 있습니다.
2. dbt 모델, 구체화(Materialization) 방식(뷰, 테이블, 증분, 임시)을 정의하고, ref() 함수를 이용해 모델 간 의존성을 관리할 수 있습니다.
3. dbt 테스트(스키마 테스트 및 커스텀 단일 테스트)를 구현하여 데이터 품질과 무결성을 검증할 수 있습니다.
4. 소스(Source), 스테이징(Staging), 중간(Intermediate), 마트(Mart) 모델로 dbt 프로젝트를 모범 사례에 따라 구성할 수 있습니다.
5. dbt docs를 활용하여 자동화된 문서와 데이터 계보(Data Lineage) 그래프를 생성하고 게시할 수 있습니다.
6. 대용량 데이터셋을 효율적으로 처리하고 서서히 변하는 차원(SCD, Slowly Changing Dimension)을 추적하기 위해 증분(Incremental) 모델과 스냅샷(Snapshot)을 구성할 수 있습니다.

---

## 개요

dbt(data build tool)는 SQL 기반의 데이터 변환 도구입니다. ELT 패턴에서 Transform 단계를 담당하며, 소프트웨어 엔지니어링 모범 사례(버전 관리, 테스트, 문서화)를 데이터 변환에 적용합니다.

---

## 1. dbt 개요

### 1.1 dbt란?

```
┌────────────────────────────────────────────────────────────────┐
│                        dbt 역할                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   ELT 파이프라인에서 T(Transform) 담당                          │
│                                                                │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│   │ Extract  │ →  │   Load   │ →  │Transform │               │
│   │ (Fivetran│    │  (to DW) │    │  (dbt)   │               │
│   │  Airbyte)│    │          │    │          │               │
│   └──────────┘    └──────────┘    └──────────┘               │
│                                                                │
│   dbt 핵심 기능:                                               │
│   - SQL 기반 모델 정의                                         │
│   - 의존성 자동 관리                                           │
│   - 테스트 및 문서화                                           │
│   - Jinja 템플릿 지원                                          │
│   - 버전 관리 (Git)                                            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 dbt Core vs dbt Cloud

| 특성 | dbt Core | dbt Cloud |
|------|----------|-----------|
| **비용** | 무료 (오픈소스) | 유료 (SaaS) |
| **실행** | CLI | Web UI + API |
| **스케줄링** | 외부 도구 필요 (Airflow) | 내장 스케줄러 |
| **IDE** | VS Code 등 | 내장 IDE |
| **협업** | Git 사용 | 내장 협업 기능 |

### 1.3 설치

```bash
# dbt Core 설치
pip install dbt-core

# 데이터베이스별 어댑터 설치
pip install dbt-postgres      # PostgreSQL
pip install dbt-snowflake     # Snowflake
pip install dbt-bigquery      # BigQuery
pip install dbt-redshift      # Redshift
pip install dbt-databricks    # Databricks

# 버전 확인
dbt --version
```

---

## 2. 프로젝트 구조

### 2.1 프로젝트 초기화

```bash
# 새 프로젝트 생성
dbt init my_project
cd my_project

# 프로젝트 구조
my_project/
├── dbt_project.yml          # 프로젝트 설정
├── profiles.yml             # 연결 설정 (~/.dbt/profiles.yml)
├── models/                  # SQL 모델
│   ├── staging/            # 스테이징 모델
│   ├── intermediate/       # 중간 모델
│   └── marts/              # 최종 모델
├── tests/                   # 커스텀 테스트
├── macros/                  # 재사용 매크로
├── seeds/                   # 시드 데이터 (CSV)
├── snapshots/               # SCD 스냅샷
├── analyses/                # 분석 쿼리
└── target/                  # 컴파일된 결과
```

### 2.2 설정 파일

```yaml
# dbt_project.yml
name: 'my_project'
version: '1.0.0'
config-version: 2

profile: 'my_project'

model-paths: ["models"]
analysis-paths: ["analyses"]
test-paths: ["tests"]
seed-paths: ["seeds"]
macro-paths: ["macros"]
snapshot-paths: ["snapshots"]

target-path: "target"
clean-targets:
  - "target"
  - "dbt_packages"

# 구체화(materialization) 전략은 메달리온 패턴을 따릅니다:
# - staging을 view로: 개발 중 빠른 반복, 항상 소스 데이터를 반영하며,
#   1:1 소스 미러의 스토리지 중복을 피할 수 있습니다
# - intermediate를 ephemeral로: CTE로 인라인화 — 직접 쿼리되지 않는
#   중간 로직의 웨어하우스 클러터를 줄입니다
# - marts를 table로: BI 쿼리를 위해 사전 계산되어 스토리지를 속도와 교환합니다
models:
  my_project:
    staging:
      +materialized: view
      +schema: staging
    intermediate:
      +materialized: ephemeral
    marts:
      +materialized: table
      +schema: marts
```

```yaml
# profiles.yml (~/.dbt/profiles.yml)
# 프로젝트 저장소 외부(~/.dbt/)에 저장하여 자격증명이
# 버전 관리에 커밋되는 것을 방지합니다. env_var()는 런타임에
# 환경에서 시크릿을 읽어 로컬 개발과 CI/CD를 모두 지원합니다.
my_project:
  target: dev
  outputs:
    dev:
      type: postgres
      host: localhost
      port: 5432
      user: postgres
      password: "{{ env_var('DB_PASSWORD') }}"
      dbname: analytics
      schema: dbt_dev
      # 개발용 4개 스레드: 속도와 로컬 리소스 한계 사이의 균형입니다.
      # 각 스레드는 하나의 모델을 동시에 실행합니다.
      threads: 4

    prod:
      type: postgres
      host: prod-db.example.com
      port: 5432
      user: "{{ env_var('PROD_USER') }}"
      password: "{{ env_var('PROD_PASSWORD') }}"
      dbname: analytics
      schema: dbt_prod
      # 프로덕션용 8개 스레드: 프로덕션 웨어하우스는 더 많은 컴퓨팅 용량을 보유합니다.
      # 스로틀링을 피하기 위해 웨어하우스의 동시 쿼리 한계에 맞게 조정하세요.
      threads: 8
```

---

## 3. 모델 (Models)

### 3.1 기본 모델

```sql
-- models/staging/stg_orders.sql
-- 스테이징 모델: 소스 데이터 정제
-- 스테이징 모델은 원본 소스 위의 얇은 정제 레이어입니다. 타입 변환과 null 필터링을
-- 처리하여 하위 모델이 이 로직을 반복하지 않고 정제된 일관된 타입의 입력을
-- 신뢰할 수 있게 합니다.

SELECT
    order_id,
    customer_id,
    -- 명시적 CAST는 웨어하우스 엔진마다 다른 암묵적 타입 강제 변환 문제를
    -- 방지합니다 (예: Snowflake vs PostgreSQL 날짜 처리)
    CAST(order_date AS DATE) AS order_date,
    CAST(amount AS DECIMAL(10, 2)) AS amount,
    status,
    -- loaded_at은 데이터 신선도 모니터링을 가능하게 합니다: 이 타임스탬프가
    -- 오래되었다면 파이프라인이 상위에서 실패했을 가능성이 높습니다
    CURRENT_TIMESTAMP AS loaded_at
FROM {{ source('raw', 'orders') }}
WHERE order_id IS NOT NULL
```

```sql
-- models/staging/stg_customers.sql
SELECT
    customer_id,
    TRIM(first_name) AS first_name,
    TRIM(last_name) AS last_name,
    LOWER(email) AS email,
    created_at
FROM {{ source('raw', 'customers') }}
```

```sql
-- models/marts/core/fct_orders.sql
-- 팩트 테이블: 주문

{{
    config(
        materialized='table',
        unique_key='order_id',
        -- 월별 파티션 분할은 파티션 프루닝(partition pruning)을 가능하게 합니다:
        -- order_date로 필터링하는 쿼리는 관련 월 파티션만 스캔하여
        -- BigQuery/Snowflake의 대용량 테이블에서 쿼리 비용을 10-100배 절감합니다.
        partition_by={
            'field': 'order_date',
            'data_type': 'date',
            'granularity': 'month'
        }
    )
}}

-- ref()는 명시적 의존성 그래프를 생성합니다: dbt는 fct_orders 전에
-- stg_orders를 빌드하고, `dbt run --select +fct_orders`는 자동으로
-- 모든 상위 모델을 포함합니다.
WITH orders AS (
    SELECT * FROM {{ ref('stg_orders') }}
),

customers AS (
    SELECT * FROM {{ ref('stg_customers') }}
)

SELECT
    o.order_id,
    o.customer_id,
    c.first_name,
    c.last_name,
    c.email,
    o.order_date,
    o.amount,
    o.status,
    -- 사전 계산된 파생 컬럼은 모든 하위 쿼리에서 반복적인 EXTRACT() 호출을 피하고
    -- year/month로 간단한 GROUP BY 필터를 가능하게 합니다
    EXTRACT(YEAR FROM o.order_date) AS order_year,
    EXTRACT(MONTH FROM o.order_date) AS order_month,
    -- 비즈니스 정의 티어: 임계값은 비즈니스 팀과 합의한 세분화와 일치해야 합니다.
    -- 이를 변경하려면 테이블을 재구축하고 하위 대시보드를 업데이트해야 합니다.
    CASE
        WHEN o.amount > 1000 THEN 'high'
        WHEN o.amount > 100 THEN 'medium'
        ELSE 'low'
    END AS order_tier
-- LEFT JOIN(INNER 아님)은 고객 데이터가 없어도 주문을 보존하여,
-- 소스의 참조 무결성 격차로 인한 자동 데이터 손실을 방지합니다
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
```

### 3.2 소스 정의

```yaml
# models/staging/_sources.yml
version: 2

sources:
  - name: raw
    description: "원본 데이터 소스"
    database: raw_db
    schema: public
    tables:
      - name: orders
        description: "주문 원본 테이블"
        columns:
          - name: order_id
            description: "주문 고유 ID"
            tests:
              - unique
              - not_null
          - name: customer_id
            description: "고객 ID"
          - name: amount
            description: "주문 금액"

      - name: customers
        description: "고객 원본 테이블"
        freshness:
          warn_after: {count: 12, period: hour}
          error_after: {count: 24, period: hour}
        loaded_at_field: updated_at
```

### 3.3 Materialization 유형

```sql
-- View (기본값): 스토리지 비용 없음, 항상 최신 상태이지만 모든 쿼리마다 재계산합니다.
-- 최적 용도: 반복 속도가 중요한 스테이징 모델이나 개발 환경.
{{ config(materialized='view') }}

-- Table: dbt 실행마다 재빌드되는 물리적 테이블. 빠른 읽기를 위해 스토리지를 교환합니다.
-- 최적 용도: BI 도구가 자주 쿼리하는 마트 모델.
{{ config(materialized='table') }}

-- Incremental: 마지막 실행 이후 새로운/변경된 행만 처리하여,
-- 대용량 팩트 테이블(예: 10억+ 행 이벤트 테이블)의 컴퓨팅 비용을 크게 절감합니다.
-- unique_key는 업서트(upsert) 시맨틱을 가능하게 합니다(기존 업데이트, 새 삽입).
-- on_schema_change='append_new_columns'는 전체 새로 고침 없이 새 소스 컬럼을 자동 추가하여
-- 상위 스키마 진화로 인한 파이프라인 중단을 방지합니다.
{{ config(
    materialized='incremental',
    unique_key='order_id',
    on_schema_change='append_new_columns'
) }}

SELECT *
FROM {{ source('raw', 'orders') }}
{% if is_incremental() %}
-- 이미 로드된 최신 날짜보다 새로운 행만 가져옵니다.
-- {{ this }}는 현재 구체화된 테이블을 참조하여 모델을 멱등적으로 만드는
-- 자기 참조 필터를 생성합니다.
WHERE order_date > (SELECT MAX(order_date) FROM {{ this }})
{% endif %}

-- Ephemeral: CTE로 컴파일되어 인라인화 — 웨어하우스 객체가 전혀 생성되지 않습니다.
-- 최적 용도: 하나의 하위 모델만 참조하는 중간 변환.
-- 많은 모델이 참조하는 모델에는 피하세요 (CTE 중복이 발생합니다).
{{ config(materialized='ephemeral') }}
```

---

## 4. 테스트

### 4.1 스키마 테스트

```yaml
# models/marts/core/_schema.yml
version: 2

models:
  - name: fct_orders
    description: "주문 팩트 테이블"
    columns:
      - name: order_id
        description: "주문 고유 ID"
        tests:
          - unique
          - not_null

      - name: customer_id
        tests:
          - not_null
          - relationships:
              to: ref('dim_customers')
              field: customer_id

      - name: amount
        tests:
          - not_null
          - dbt_utils.expression_is_true:
              expression: ">= 0"

      - name: status
        tests:
          - accepted_values:
              values: ['pending', 'completed', 'cancelled', 'refunded']
```

### 4.2 커스텀 테스트

```sql
-- tests/assert_positive_amounts.sql
-- 모든 주문 금액이 양수인지 확인

SELECT
    order_id,
    amount
FROM {{ ref('fct_orders') }}
WHERE amount < 0
```

```sql
-- macros/test_row_count_equal.sql
{% test row_count_equal(model, compare_model) %}

WITH model_count AS (
    SELECT COUNT(*) AS cnt FROM {{ model }}
),

compare_count AS (
    SELECT COUNT(*) AS cnt FROM {{ compare_model }}
)

SELECT
    m.cnt AS model_count,
    c.cnt AS compare_count
FROM model_count m
CROSS JOIN compare_count c
WHERE m.cnt != c.cnt

{% endtest %}
```

### 4.3 테스트 실행

```bash
# 모든 테스트 실행
dbt test

# 특정 모델 테스트
dbt test --select fct_orders

# 소스 freshness 테스트
dbt source freshness
```

---

## 5. 문서화

### 5.1 문서 정의

```yaml
# models/marts/core/_schema.yml
version: 2

models:
  - name: fct_orders
    description: |
      ## 주문 팩트 테이블

      이 테이블은 모든 주문 트랜잭션을 포함합니다.

      ### 사용 사례
      - 일별/월별 매출 분석
      - 고객 구매 패턴 분석
      - 재구매율 계산

      ### 갱신 주기
      - 매일 06:00 UTC

    meta:
      owner: "data-team@company.com"
      contains_pii: false

    columns:
      - name: order_id
        description: "주문 고유 식별자 (UUID)"
        meta:
          dimension: true

      - name: amount
        description: "주문 총액 (USD)"
        meta:
          measure: true
          aggregation: sum
```

### 5.2 문서 생성 및 서빙

```bash
# 문서 생성
dbt docs generate

# 문서 서버 실행
dbt docs serve --port 8080

# http://localhost:8080 에서 확인
```

---

## 6. Jinja 템플릿

### 6.1 기본 Jinja 문법

```sql
-- 변수
{% set my_var = 'value' %}
SELECT '{{ my_var }}' AS col

-- 조건문
SELECT
    CASE
        {% if target.name == 'prod' %}
        WHEN amount > 1000 THEN 'high'
        {% else %}
        WHEN amount > 100 THEN 'high'
        {% endif %}
        ELSE 'low'
    END AS tier
FROM orders

-- 반복문
SELECT
    order_id,
    {% for col in ['amount', 'quantity', 'discount'] %}
    SUM({{ col }}) AS total_{{ col }}{% if not loop.last %},{% endif %}
    {% endfor %}
FROM order_items
GROUP BY order_id
```

### 6.2 Macros

```sql
-- macros/generate_schema_name.sql
-- dbt의 기본 스키마 명명을 재정의하여 커스텀 스키마에 타겟 스키마를
-- 접두사로 붙입니다 (예: dbt_dev_staging). 이는 각 환경이 자체
-- target.schema 접두사를 가지므로 dev와 prod 모델이 같은 데이터베이스에서
-- 충돌하는 것을 방지합니다.
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- set default_schema = target.schema -%}
    {%- if custom_schema_name is none -%}
        {{ default_schema }}
    {%- else -%}
        {{ default_schema }}_{{ custom_schema_name }}
    {%- endif -%}
{%- endmacro %}


-- macros/cents_to_dollars.sql
-- 통화 변환 로직을 중앙화합니다: 정밀도나 제수가 변경되면
-- (예: 센트에서 베이시스 포인트로 전환), 돈을 다루는
-- 모든 모델 대신 여기서 한 번만 업데이트하면 됩니다.
{% macro cents_to_dollars(column_name, precision=2) %}
    ROUND({{ column_name }} / 100.0, {{ precision }})
{% endmacro %}


-- macros/limit_data_in_dev.sql
-- 개발 중 자동으로 쿼리 결과를 제한하여 반복 속도를 높이고
-- 웨어하우스 비용을 절감합니다. 프로덕션(target.name != 'dev')에서는
-- LIMIT이 적용되지 않아 프로덕션 모델은 전체 데이터셋을 처리합니다.
{% macro limit_data_in_dev() %}
    {% if target.name == 'dev' %}
        LIMIT 1000
    {% endif %}
{% endmacro %}
```

```sql
-- 매크로 사용
SELECT
    order_id,
    {{ cents_to_dollars('amount_cents') }} AS amount_dollars
FROM orders
{{ limit_data_in_dev() }}
```

### 6.3 dbt 내장 함수

```sql
-- ref(): 다른 모델 참조
SELECT * FROM {{ ref('stg_orders') }}

-- source(): 소스 테이블 참조
SELECT * FROM {{ source('raw', 'orders') }}

-- this: 현재 모델 참조 (incremental에서 유용)
{% if is_incremental() %}
SELECT MAX(updated_at) FROM {{ this }}
{% endif %}

-- config(): 설정 값 접근
{{ config.get('materialized') }}

-- target: 타겟 환경 정보
{{ target.name }}    -- dev, prod
{{ target.schema }}  -- dbt_dev
{{ target.type }}    -- postgres, snowflake
```

---

## 7. 증분 처리 (Incremental)

### 7.1 기본 증분 모델

```sql
-- models/marts/fct_events.sql
{{
    config(
        materialized='incremental',
        unique_key='event_id',
        -- delete+insert는 merge보다 성능이 좋습니다
        -- 네이티브 MERGE 지원이 없는 웨어하우스(예: 구 PostgreSQL)에서.
        -- unique_key와 일치하는 행을 삭제한 후 새 버전을 삽입하여
        -- 두 단계로 업서트 시맨틱을 달성합니다.
        incremental_strategy='delete+insert'
    )
}}

SELECT
    event_id,
    user_id,
    event_type,
    event_data,
    created_at
FROM {{ source('raw', 'events') }}

{% if is_incremental() %}
-- 이미 로드된 최신 날짜보다 새로운 이벤트만 처리합니다.
-- 이벤트가 대략 시간순으로 도착한다고 가정합니다; 순서가 맞지 않는
-- 이벤트의 경우 룩백 윈도우를 고려하세요: created_at > MAX(created_at) - INTERVAL '1 hour'
WHERE created_at > (SELECT MAX(created_at) FROM {{ this }})
{% endif %}
```

### 7.2 증분 전략

```sql
-- Append (기본): 새 데이터만 추가, 기존 행은 업데이트하지 않습니다.
-- 가장 빠른 전략(중복 제거 오버헤드 없음) — 중복이 상위에서 처리되거나
-- 문제가 되지 않는 불변 이벤트 로그에 이상적입니다.
{{ config(
    materialized='incremental',
    incremental_strategy='append'
) }}

-- Delete+Insert: 키로 삭제 후 삽입합니다.
-- 모든 웨어하우스에서 작동합니다 (MERGE 불필요). MERGE 지원이 제한된
-- PostgreSQL 같은 데이터베이스에서 업서트 시맨틱이 필요할 때 사용하세요.
{{ config(
    materialized='incremental',
    unique_key='id',
    incremental_strategy='delete+insert'
) }}

-- Merge (Snowflake, BigQuery): MERGE 문을 사용합니다.
-- 가장 유연함: 원자적으로 새 행을 삽입하고 기존 행을 업데이트합니다.
-- merge_update_columns는 업데이트되는 컬럼을 제한하여,
-- 다른 프로세스가 관리하는 컬럼의 우발적 덮어쓰기를 방지합니다.
{{ config(
    materialized='incremental',
    unique_key='id',
    incremental_strategy='merge',
    merge_update_columns=['name', 'amount', 'updated_at']
) }}
```

---

## 8. 실행 명령

### 8.1 기본 명령

```bash
# 연결 테스트
dbt debug

# 모든 모델 실행
dbt run

# 특정 모델만 실행
dbt run --select fct_orders
dbt run --select staging.*
dbt run --select +fct_orders+  # 의존성 포함

# 테스트 실행
dbt test

# 빌드 (run + test)
dbt build

# Seed 데이터 로드
dbt seed

# 컴파일만 (실행 안 함)
dbt compile

# 정리
dbt clean
```

### 8.2 선택자 (Selectors)

```bash
# 모델 이름으로
dbt run --select my_model

# 경로로
dbt run --select models/staging/*

# 태그로
dbt run --select tag:daily

# 상위 의존성 포함
dbt run --select +my_model

# 하위 의존성 포함
dbt run --select my_model+

# 양방향
dbt run --select +my_model+

# 특정 모델 제외
dbt run --exclude my_model
```

---

## 9. dbt 고급: 시맨틱 레이어(Semantic Layer)와 메트릭(Metrics)

### 9.1 시맨틱 레이어: 메트릭의 단일 진실 공급원(Single Source of Truth)

전통적인 분석 워크플로우에서는 "매출" 또는 "활성 사용자"와 같은 비즈니스 메트릭이 각 BI 도구, 대시보드, 임시 쿼리에서 독립적으로 정의됩니다. 이로 인해 팀 간에 수치가 일치하지 않는 문제가 발생합니다. CFO의 매출 수치가 제품 팀의 수치와 다른 이유는 각자가 약간 다른 SQL 정의를 사용했기 때문입니다.

> **비유**: 시맨틱 레이어를 **비즈니스 메트릭의 사전**이라고 생각하세요. 사전이 각 단어에 대한 유일한 권위 있는 정의를 제공하듯, 시맨틱 레이어는 각 메트릭에 대한 하나의 표준 정의를 제공합니다. 이것이 없으면, 모든 분석가가 자신만의 "방언"으로 매출을 작성하게 되어 아무도 그 단어의 의미에 동의하지 못합니다.

**dbt 시맨틱 레이어(dbt Semantic Layer)**(dbt Cloud에서 사용 가능, MetricFlow 기반)는 dbt 프로젝트에서 메트릭을 한 번 정의하고, 다운스트림 도구가 소비할 수 있는 쿼리 API를 통해 이를 노출함으로써 이 문제를 해결합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                  Traditional vs Semantic Layer                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Traditional:                                                  │
│   ┌────────┐  each tool defines    ┌──────────┐                │
│   │Looker  │──its own "revenue"──→ │Dashboard1│  (revenue=$1M) │
│   │Tableau │──its own "revenue"──→ │Dashboard2│  (revenue=$1.2M)│
│   │Notebook│──its own "revenue"──→ │Report    │  (revenue=$980K)│
│   └────────┘                       └──────────┘                │
│                                                                 │
│   Semantic Layer:                                               │
│   ┌────────┐      ┌───────────┐    ┌──────────┐               │
│   │Looker  │─┐    │  dbt      │    │Dashboard1│               │
│   │Tableau │─┼───→│  Semantic │───→│Dashboard2│  (all=$1M)    │
│   │Notebook│─┘    │  Layer    │    │Report    │               │
│   └────────┘      └───────────┘    └──────────┘               │
│                   (single def)                                  │
└─────────────────────────────────────────────────────────────────┘
```

| 측면 | 전통적 접근 방식 | 시맨틱 레이어 접근 방식 |
|------|----------------|----------------------|
| **메트릭 정의** | 모든 BI 도구 / 쿼리에서 중복 정의 | dbt YAML에서 한 번만 정의 |
| **일관성** | 팀 간 불일치 발생 가능 | 단일 진실 공급원 보장 |
| **거버넌스(Governance)** | 누가 무엇을 정의했는지 감사하기 어려움 | Git으로 버전 관리 |
| **유지보수** | N곳에서 로직 변경 필요 | 한 번 변경하면 전체 적용 |
| **인사이트 도출 시간** | 분석가가 정확한 SQL을 알아야 함 | 메트릭 이름 + 차원으로 쿼리 |
| **테스트** | 수동 / 도구별 | dbt 테스트가 메트릭 정의에 적용 |

### 9.2 MetricFlow와 시맨틱 모델(Semantic Models)

dbt는 2023년에 시맨틱 레이어를 구동하기 위해 **Transform**(MetricFlow를 만든 회사)을 인수했습니다. MetricFlow는 메트릭 요청을 최적화된 SQL로 컴파일합니다. 구성 요소는 **시맨틱 모델(semantic models)**과 **메트릭(metrics)**입니다.

#### 시맨틱 모델 정의

시맨틱 모델은 dbt 모델을 시맨틱 개념에 매핑합니다: 엔티티(Entity, 조인 키), 차원(Dimension, 그룹화 컬럼), 측정값(Measure, 집계 가능한 컬럼).

```yaml
# models/marts/core/_semantic_models.yml
# Why semantic models? They declare the "meaning" of columns so that
# MetricFlow knows how to join tables and aggregate measures automatically.

semantic_models:
  - name: orders
    defaults:
      agg_time_dimension: order_date
    description: "Order transactions for metric computation"
    model: ref('fct_orders')

    entities:
      # Entities define join keys — MetricFlow uses these to
      # automatically join semantic models when a metric needs
      # dimensions from multiple tables.
      - name: order_id
        type: primary
      - name: customer_id
        type: foreign

    dimensions:
      - name: order_date
        type: time
        type_params:
          time_granularity: day
      - name: order_tier
        type: categorical
      - name: status
        type: categorical

    measures:
      # Measures are the raw aggregatable building blocks.
      # Metrics (defined separately) reference these measures.
      - name: order_count
        agg: count
        expr: order_id
      - name: total_revenue
        agg: sum
        expr: amount
      - name: average_order_value
        agg: average
        expr: amount
```

#### 메트릭 정의

메트릭은 시맨틱 모델의 측정값을 참조하며 네 가지 유형이 있습니다:

```yaml
# models/marts/core/_metrics.yml
metrics:
  # --- Simple metric: directly references one measure ---
  - name: revenue
    description: "Total revenue from completed orders"
    type: simple
    label: "Revenue"
    type_params:
      measure: total_revenue
    filter: |
      {{ Dimension('order_id__status') }} = 'completed'

  - name: order_count
    description: "Total number of orders"
    type: simple
    label: "Order Count"
    type_params:
      measure: order_count

  # --- Derived metric: combines other metrics with arithmetic ---
  # Why derived? AOV = revenue / order_count, but defining it as
  # derived ensures both numerator and denominator use the exact
  # same filters and grain, preventing subtle mismatches.
  - name: average_order_value
    description: "Average order value (revenue / orders)"
    type: derived
    label: "AOV"
    type_params:
      expr: revenue / order_count
      metrics:
        - name: revenue
        - name: order_count

  # --- Cumulative metric: running total over time ---
  - name: cumulative_revenue
    description: "Cumulative revenue year-to-date"
    type: cumulative
    label: "Cumulative Revenue"
    type_params:
      measure: total_revenue
      window: 1 year

  # --- Conversion metric: measures funnel conversion rates ---
  - name: checkout_conversion_rate
    description: "Ratio of orders to cart-creation events"
    type: conversion
    label: "Checkout Conversion"
    type_params:
      entity: customer_id
      calculation: conversions / opportunities
      base_measure: cart_creations    # opportunities
      conversion_measure: order_count # conversions
      window: 7 days
```

### 9.3 시맨틱 레이어 쿼리

#### dbt Cloud 시맨틱 레이어 API

메트릭이 정의되면 다운스트림 도구는 원시 SQL을 작성하지 않고도 **시맨틱 레이어 API**(GraphQL 또는 JDBC)를 통해 메트릭을 쿼리합니다. MetricFlow는 각 요청을 최적화된 SQL로 컴파일합니다.

```graphql
# Example: Query revenue by month and order tier
# The API consumer never writes SQL — just declares
# what metric, dimensions, and time grain they need.
{
  createQuery(
    metrics: [{name: "revenue"}]
    groupBy: [
      {name: "metric_time", grain: MONTH},
      {name: "order_tier"}
    ]
    where: [{sql: "{{ TimeDimension('metric_time', 'MONTH') }} >= '2024-01-01'"}]
    orderBy: [{name: "metric_time"}]
  ) {
    queryId
    result {
      data
    }
  }
}
```

```bash
# Using the dbt Cloud CLI to query metrics locally during development
# Why query locally? To validate metric definitions before deploying.
dbt sl query --metrics revenue --group-by metric_time__month,order_tier \
  --where "metric_time__month >= '2024-01-01'" --order-by metric_time__month
```

#### BI 도구 연동

| BI 도구 | 연동 방법 | 비고 |
|---------|----------|------|
| **Looker** | dbt 시맨틱 레이어 연결 | Looker 2024.2부터 네이티브 연동 |
| **Tableau** | JDBC 커넥터 | 메트릭이 데이터 소스 필드로 표시 |
| **Hex** | 네이티브 dbt 연동 | 노트북에서 직접 메트릭 쿼리 가능 |
| **Google Sheets** | Google Sheets 애드온 | 스프레드시트로 메트릭 가져오기 |
| **커스텀 앱** | GraphQL / JDBC API | 메트릭 기반 내부 도구 구축 |

### 9.4 dbt Mesh: 멀티 프로젝트 dbt

조직이 성장함에 따라 단일 모놀리식(monolithic) dbt 프로젝트는 관리하기 어려워집니다. **dbt Mesh**(dbt 1.6+에서 도입)는 여러 dbt 프로젝트가 명확한 소유권 경계를 유지하면서 서로의 모델을 참조할 수 있게 합니다.

#### 크로스 프로젝트 참조(Cross-Project References)

```yaml
# project_b/dbt_project.yml
# Why cross-project refs? Team A owns the core orders model,
# Team B (marketing analytics) needs to build on top of it
# without duplicating the SQL or breaking Team A's contract.
name: marketing_analytics
version: '1.0.0'

dependencies:
  - project: core_analytics
    # This declares a dependency on another dbt project.
    # dbt resolves the ref() at compile time across projects.
```

```sql
-- project_b/models/marts/marketing/mkt_campaign_attribution.sql
-- Cross-project ref: access core_analytics.fct_orders from this project.
-- The two-argument ref() tells dbt the model lives in another project.
SELECT
    o.order_id,
    o.customer_id,
    o.amount,
    c.campaign_id,
    c.utm_source
FROM {{ ref('core_analytics', 'fct_orders') }} o
LEFT JOIN {{ ref('stg_campaign_touches') }} c
    ON o.customer_id = c.customer_id
    AND o.order_date BETWEEN c.touch_date AND c.touch_date + INTERVAL '7 days'
```

#### 퍼블릭 모델(Public Models)과 컨트랙트(Contracts)

```yaml
# project_a/models/marts/core/_models.yml
# Why contracts? Public models are consumed by other projects,
# so their schema is a promise — breaking it would cascade failures.
models:
  - name: fct_orders
    access: public          # Exposed to other projects (default is "protected")
    group: core_team        # Ownership group
    latest_version: 2       # Enables model versioning for safe migrations

    config:
      contract:
        enforced: true      # Columns must match the declared schema exactly

    columns:
      - name: order_id
        data_type: varchar
        description: "Primary key"
        constraints:
          - type: not_null
          - type: primary_key
      - name: amount
        data_type: numeric
        description: "Order amount in USD"
        constraints:
          - type: not_null
```

#### 그룹(Groups)과 접근 제어(Access Control)

```yaml
# dbt_project.yml — define ownership groups
groups:
  - name: core_team
    owner:
      name: "Core Data Team"
      email: "core-data@company.com"

  - name: marketing_team
    owner:
      name: "Marketing Analytics"
      email: "mkt-analytics@company.com"
```

```yaml
# models/marts/core/_models.yml
# Access levels control who can ref() a model:
#   - private:   only within the same group
#   - protected: only within the same project (default)
#   - public:    any project can reference it
models:
  - name: fct_orders
    access: public
    group: core_team

  - name: int_order_enriched
    access: protected      # Other projects cannot ref() this
    group: core_team

  - name: _stg_orders_deduped
    access: private        # Only core_team models can ref() this
    group: core_team
```

---

## 연습 문제

### 문제 1: 스테이징 모델
원본 products 테이블에서 stg_products 모델을 생성하세요. 가격을 달러로 변환하고 NULL 값을 처리하세요.

### 문제 2: 증분 모델
일별 판매 집계 테이블을 증분으로 처리하는 모델을 작성하세요.

### 문제 3: 테스트 작성
fct_sales 모델에 대한 테스트를 작성하세요 (unique, not_null, 금액 양수 확인).

---

## 요약

| 개념 | 설명 |
|------|------|
| **Model** | SQL 기반 데이터 변환 정의 |
| **Source** | 원본 데이터 참조 |
| **ref()** | 모델 간 참조 (의존성 자동 관리) |
| **Test** | 데이터 품질 검증 |
| **Materialization** | view, table, incremental, ephemeral |
| **Macro** | 재사용 가능한 SQL 템플릿 |

---

## 참고 자료

- [dbt Documentation](https://docs.getdbt.com/)
- [dbt Learn](https://courses.getdbt.com/)
- [dbt Best Practices](https://docs.getdbt.com/guides/best-practices)
