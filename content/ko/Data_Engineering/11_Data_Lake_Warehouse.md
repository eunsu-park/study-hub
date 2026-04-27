[← 이전: 10. Kafka 스트리밍](10_Kafka_Streaming.md) | [다음: 12. dbt 변환 도구 →](12_dbt_Transformation.md)

# Data Lake와 Data Warehouse

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Data Lake, Data Warehouse, Lakehouse 시스템의 아키텍처 차이를 Schema-on-Write와 Schema-on-Read 방식을 포함하여 비교할 수 있습니다.
2. 분석 쿼리 성능에서 차원 모델링(Dimensional Modeling, 스타 스키마 및 눈송이 스키마)의 역할을 설명할 수 있습니다.
3. 데이터 볼륨, 다양성, 쿼리 패턴에 따른 스토리지 아키텍처 선택 시 트레이드오프(trade-off)를 평가할 수 있습니다.
4. Delta Lake 또는 Apache Iceberg와 같은 오픈 테이블 형식을 활용하여 Lakehouse 아키텍처를 설계할 수 있습니다.
5. 메달리온 아키텍처(Medallion Architecture)의 브론즈, 실버, 골드 레이어 간 데이터를 이동하는 ETL/ELT 파이프라인을 구현할 수 있습니다.
6. 주요 클라우드 데이터 웨어하우스 및 데이터 레이크 솔루션의 비용과 확장성 영향을 분석할 수 있습니다.

---

## 개요

데이터 저장소 아키텍처는 조직의 데이터 전략에 핵심적입니다. Data Lake, Data Warehouse, 그리고 둘을 결합한 Lakehouse 아키텍처의 특성과 사용 사례를 이해합니다.

---

## 1. Data Warehouse

### 1.1 개념

```
┌────────────────────────────────────────────────────────────────┐
│                    Data Warehouse                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   특징:                                                        │
│   - 구조화된 데이터 (스키마 정의 필수)                           │
│   - Schema-on-Write (쓰기 시 스키마 적용)                       │
│   - 분석 최적화 (OLAP)                                         │
│   - SQL 기반 쿼리                                              │
│                                                                │
│   ┌──────────────────────────────────────────────────────┐    │
│   │                    Data Warehouse                     │    │
│   │   ┌─────────────────────────────────────────────────┐│    │
│   │   │  Dim Tables    │    Fact Tables                 ││    │
│   │   │  ┌──────────┐  │  ┌──────────┐                 ││    │
│   │   │  │dim_date  │  │  │fact_sales│                 ││    │
│   │   │  │dim_product│  │  │fact_orders│                ││    │
│   │   │  │dim_customer│ │                               ││    │
│   │   │  └──────────┘  │  └──────────┘                 ││    │
│   │   └─────────────────────────────────────────────────┘│    │
│   └──────────────────────────────────────────────────────┘    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 주요 솔루션

| 솔루션 | 유형 | 특징 |
|--------|------|------|
| **Snowflake** | 클라우드 | 분리된 스토리지/컴퓨팅, 자동 확장 |
| **BigQuery** | 클라우드 (GCP) | 서버리스, 페타바이트 규모 |
| **Redshift** | 클라우드 (AWS) | Columnar, MPP 아키텍처 |
| **Synapse** | 클라우드 (Azure) | 통합 분석 플랫폼 |
| **PostgreSQL** | 온프레미스 | 소규모, 오픈소스 |

### 1.3 Data Warehouse SQL 예시

```sql
-- Snowflake/BigQuery 스타일 분석 쿼리

-- 월별 매출 트렌드
-- 왜 스타 스키마 조인인가? 팩트를 차원에 조인하면 전체 데이터셋을
-- 비정규화하지 않고도 모든 차원 속성으로 메트릭을 슬라이싱할 수 있습니다.
SELECT
    d.year,
    d.month,
    d.month_name,
    SUM(f.sales_amount) AS total_sales,
    COUNT(DISTINCT f.customer_sk) AS unique_customers,
    AVG(f.sales_amount) AS avg_order_value,
    -- NULLIF는 이전 달 매출이 없을 때(예: 데이터셋의 첫 달 또는 계절적 공백)
    -- 0으로 나누기를 방지합니다.
    -- LAG 윈도우 함수는 셀프 조인 없이 연속 월을 비교하며,
    -- 컬럼형 웨어하우스에서 훨씬 더 효율적입니다.
    (SUM(f.sales_amount) - LAG(SUM(f.sales_amount)) OVER (ORDER BY d.year, d.month))
        / NULLIF(LAG(SUM(f.sales_amount)) OVER (ORDER BY d.year, d.month), 0) * 100
        AS mom_growth_pct
FROM fact_sales f
JOIN dim_date d ON f.date_sk = d.date_sk
WHERE d.year >= 2023
GROUP BY d.year, d.month, d.month_name
ORDER BY d.year, d.month;

-- 고객 세그먼트별 LTV (Life Time Value)
-- CTE는 고객별 집계를 격리하여 외부 쿼리가 세그먼트 수준 통계를
-- 깔끔하게 계산할 수 있게 합니다 — 중첩 서브쿼리를 피하고
-- 로직을 독립적으로 테스트할 수 있습니다.
WITH customer_metrics AS (
    SELECT
        c.customer_sk,
        c.customer_segment,
        MIN(d.full_date) AS first_purchase_date,
        MAX(d.full_date) AS last_purchase_date,
        COUNT(DISTINCT f.order_id) AS total_orders,
        SUM(f.sales_amount) AS total_revenue
    FROM fact_sales f
    JOIN dim_customer c ON f.customer_sk = c.customer_sk
    JOIN dim_date d ON f.date_sk = d.date_sk
    GROUP BY c.customer_sk, c.customer_segment
)
SELECT
    customer_segment,
    COUNT(*) AS customer_count,
    AVG(total_orders) AS avg_orders,
    AVG(total_revenue) AS avg_ltv,
    -- 중앙값(median)은 매출 분포가 치우쳐 있을 때 평균보다 더 견고합니다;
    -- 소수의 고래 고객(whale customer)이 평균을 크게 왜곡할 수 있습니다.
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_revenue) AS median_ltv
FROM customer_metrics
GROUP BY customer_segment
ORDER BY avg_ltv DESC;
```

---

## 2. Data Lake

### 2.1 개념

```
┌────────────────────────────────────────────────────────────────┐
│                      Data Lake                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   특징:                                                        │
│   - 모든 형태의 데이터 (구조화, 반구조화, 비구조화)               │
│   - Schema-on-Read (읽기 시 스키마 적용)                        │
│   - 원본 데이터 보존                                           │
│   - 저비용 스토리지                                            │
│                                                                │
│   ┌──────────────────────────────────────────────────────┐    │
│   │                     Data Lake                         │    │
│   │  ┌────────────────────────────────────────────────┐  │    │
│   │  │  Raw Zone (Bronze)                              │  │    │
│   │  │  - 원본 데이터 (JSON, CSV, Logs, Images)        │  │    │
│   │  └────────────────────────────────────────────────┘  │    │
│   │                         ↓                             │    │
│   │  ┌────────────────────────────────────────────────┐  │    │
│   │  │  Processed Zone (Silver)                        │  │    │
│   │  │  - 정제된 데이터 (Parquet, Delta)               │  │    │
│   │  └────────────────────────────────────────────────┘  │    │
│   │                         ↓                             │    │
│   │  ┌────────────────────────────────────────────────┐  │    │
│   │  │  Curated Zone (Gold)                            │  │    │
│   │  │  - 분석/ML 준비 데이터                          │  │    │
│   │  └────────────────────────────────────────────────┘  │    │
│   └──────────────────────────────────────────────────────┘    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 주요 스토리지

| 스토리지 | 클라우드 | 특징 |
|----------|----------|------|
| **S3** | AWS | 객체 스토리지, 높은 내구성 |
| **GCS** | GCP | Google Cloud Storage |
| **ADLS** | Azure | Azure Data Lake Storage |
| **HDFS** | 온프레미스 | Hadoop Distributed File System |

### 2.3 Data Lake 파일 구조

```
s3://my-data-lake/
├── raw/                          # Bronze 레이어
│   ├── orders/
│   │   ├── year=2024/
│   │   │   ├── month=01/
│   │   │   │   ├── day=15/
│   │   │   │   │   ├── orders_20240115_001.json
│   │   │   │   │   └── orders_20240115_002.json
│   ├── customers/
│   │   └── snapshot_20240115.csv
│   └── logs/
│       └── app_logs_20240115.log
│
├── processed/                    # Silver 레이어
│   ├── orders/
│   │   └── year=2024/
│   │       └── month=01/
│   │           └── part-00000.parquet
│   └── customers/
│       └── part-00000.parquet
│
└── curated/                      # Gold 레이어
    ├── fact_sales/
    │   └── year=2024/
    │       └── month=01/
    └── dim_customers/
        └── current/
```

```python
# PySpark로 Data Lake 계층 처리
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder \
    .appName("DataLakeProcessing") \
    .getOrCreate()

# Raw → Processed (Bronze → Silver)
# Bronze는 감사(auditability)와 재처리를 위해 변경 불가능한 원본 데이터를 보관합니다.
# Silver는 정제 및 중복 제거를 적용하여 하위 소비자가
# 이 단계를 반복하지 않아도 되는 단일 정제 진실 공급원이 됩니다.
def process_raw_orders():
    # Raw JSON 읽기 — Schema-on-Read는 여기서 어떤 형태든 수용하고
    # 처리 중에만 구조를 적용한다는 의미입니다 (Data Lake 철학)
    raw_df = spark.read.json("s3://my-data-lake/raw/orders/")

    # NULL 필터링 및 중복 제거: 원본 소스는 최소 한 번 전달(at-least-once delivery)로
    # 중복이 자주 포함됩니다; 비즈니스 키로 중복 제거하여
    # 하위 집계에서 부풀려진 카운트를 방지합니다
    processed_df = raw_df \
        .filter(col("order_id").isNotNull()) \
        .withColumn("processed_at", current_timestamp()) \
        .dropDuplicates(["order_id"])

    # Parquet은 컬럼형 압축(JSON 대비 10-30배)과
    # 효율적인 분석 쿼리를 위한 프레디케이트 푸시다운(predicate pushdown)을 제공합니다.
    # year/month로 파티션 분할하면 파티션 프루닝(partition pruning)이 가능해
    # 날짜로 필터링하는 쿼리는 관련 디렉토리만 스캔합니다.
    processed_df.write \
        .mode("overwrite") \
        .partitionBy("year", "month") \
        .parquet("s3://my-data-lake/processed/orders/")

# Processed → Curated (Silver → Gold)
# Gold 레이어는 빠른 쿼리에 최적화된 비즈니스 준비 집계와 차원 모델을 포함합니다
# — BI 도구가 여기서 직접 읽습니다.
def create_fact_sales():
    orders = spark.read.parquet("s3://my-data-lake/processed/orders/")
    customers = spark.read.parquet("s3://my-data-lake/processed/customers/")

    # Gold 레이어에서 조인하여 차원 모델을 사전 계산합니다;
    # 여기서 한 번 수행하면 모든 하위 쿼리나 대시보드에서
    # 반복적인 비용이 많이 드는 조인을 피할 수 있습니다
    fact_sales = orders \
        .join(customers, "customer_id") \
        .select(
            col("order_id"),
            col("customer_sk"),
            col("order_date"),
            col("amount").alias("sales_amount")
        )

    fact_sales.write \
        .mode("overwrite") \
        .partitionBy("year", "month") \
        .parquet("s3://my-data-lake/curated/fact_sales/")
```

---

## 3. Data Warehouse vs Data Lake

### 이론: Schema-on-Write vs Schema-on-Read

이 아키텍처들이 자신을 배열하는 근본 축.

#### B.1 Schema-on-write (웨어하우스 기본)

적재 시점에 검증하고 구조화. 잘못된 행은 거부되거나 격리. 웨어하우스 테이블은 항상 선언된 스키마를 따름.

- **쓰기 시점에 비용 지불.** Producer가 따라야 함. 새 필드는 마이그레이션 요구.
- **읽기 시점의 신뢰.** 모든 쿼리가 깨끗하고 타입된 데이터를 봄. 분석가가 빠르게 움직임.
- **실패 모드:** 스키마 마이그레이션이 고통스러움. 업스트림 소스의 컬럼 이름 변경이 로더를 깨뜨림.

#### B.2 Schema-on-read (레이크 기본)

원시 바이트 던져 넣음; consumer가 읽기 시점에 파싱하고 검증. 새 필드 추가는 쓰기 시점에 무료.

- **읽기 시점에 비용 지불.** 모든 consumer가 파싱/검증 비용 지불. 스키마 버그가 늦게 표면화.
- **쓰기 시점의 유연성.** Producer가 빠르게 움직임. 새 필드가 마이그레이션 없이 JSON에 나타남.
- **실패 모드:** 쓰레기 누적. 다른 consumer가 필드를 다르게 파싱. 신뢰가 침식.

#### B.3 레이크하우스: bronze에서 schema-on-read, silver/gold에서 schema-on-write

실용적 종합. 원시 계층(bronze)은 무엇이든 받음; 검증은 bronze→silver 변환 중에 발생. Silver와 gold 계층은 타입되고 검증되고 스키마 강제됨. "데이터 스왐프" 위험이 bronze로 제한됨; 다운스트림 모든 것은 깨끗.

### 이론: 저장 경제학

레이크하우스를 가능하게 만드는 경제 현실.

- **객체 스토리지:** $0.023/GB-월(S3). 사실상 무제한 용량. 다중 리전 복제 내장.
- **웨어하우스 저장**(관리형): 일반적으로 raw 객체 스토리지보다 5-20배 더 비쌈.
- **컴퓨트:** 탄력적, 초당 청구. 시간당 100노드를 띄우고, 시간당 비용 지불, 종료.

이 체제에서 **저장은 무료; 컴퓨트가 청구서**. 최적 아키텍처: 모든 것을 lake 가격에 raw(bronze)로 저장, 탄력적 컴퓨트로 on-demand로 변환, 대부분 쿼리가 닿는 silver/gold 계층만 구체화.

저장이 지배 비용이었던 옛 웨어하우스 모델은 시대에 뒤떨어짐. 레이크하우스 아키텍처는 클라우드 가격이 실제로 작동하는 방식과 정렬됩니다.

### 3.1 비교

| 특성 | Data Warehouse | Data Lake |
|------|----------------|-----------|
| **데이터 유형** | 구조화 | 모든 유형 |
| **스키마** | Schema-on-Write | Schema-on-Read |
| **사용자** | 비즈니스 분석가 | 데이터 과학자, 엔지니어 |
| **처리** | OLAP | 배치, 스트리밍, ML |
| **비용** | 높음 | 낮음 |
| **쿼리 성능** | 최적화됨 | 가변적 |
| **데이터 품질** | 높음 (정제됨) | 가변적 |

### 3.2 선택 기준

```python
def choose_architecture(requirements: dict) -> str:
    """아키텍처 선택 가이드"""

    # 일치하는 요소 수로 가중 점수를 계산합니다.
    # 실제로는 요소별 가중치가 달라야 하지만(예: 거버넌스 준수는 협상 불가일 수 있음),
    # 동일한 가중치는 이 휴리스틱을 단순하고 확장하기 쉽게 유지합니다.
    warehouse_factors = [
        requirements.get('structured_data_only', False),
        requirements.get('sql_analytics_primary', False),
        requirements.get('strict_governance', False),
        requirements.get('fast_query_response', False),
    ]

    # Lake는 더 넓은 범위의 사용 사례를 다루므로 요소가 더 많습니다;
    # 이는 요구사항이 혼재할 때 자연스럽게 Lake 쪽으로 점수를 치우치게 하며,
    # Lake/Lakehouse로 시작하는 업계 트렌드를 반영합니다.
    lake_factors = [
        requirements.get('unstructured_data', False),
        requirements.get('ml_workloads', False),
        requirements.get('raw_data_preservation', False),
        requirements.get('cost_sensitive', False),
        requirements.get('schema_flexibility', False),
    ]

    if sum(warehouse_factors) > sum(lake_factors):
        return "Data Warehouse 권장"
    elif sum(lake_factors) > sum(warehouse_factors):
        return "Data Lake 권장"
    else:
        # 타이브레이킹(tie-breaking)은 Lakehouse 방향으로: 어느 쪽도 지배하지 않을 때,
        # Lakehouse는 SQL 성능과 원본 데이터 유연성을 모두 제공합니다
        return "Lakehouse 고려"
```

---

## 4. Lakehouse

### 이론: 웨어하우스-레이크-레이크하우스 진화

세 가지 아키텍처 파도, 각각 이전의 고통을 다룸.

#### A.1 1차: 데이터 웨어하우스 (1990년대-2010년대)

Teradata, Oracle, SQL Server, 그다음 Redshift, Snowflake, BigQuery. 메시지: 분석에 튜닝된 관계형 데이터베이스. 쓰기 시 스키마 강제, 컬럼 저장, MPP 실행, 통계가 있는 옵티마이저. 분석가가 SQL을 실행, 빠른 결과 받음.

다음 파도를 추진한 한계:
- **저장과 컴퓨트가 결합됨**(초기 세대). 저장 추가는 컴퓨트 하드웨어 추가를 요구. 비용이 나쁘게 확장.
- **스키마 경직성.** 컬럼 추가는 페타바이트 위에 ALTER TABLE을 의미; 비-SQL 데이터(이미지, 자유 텍스트 로그) 거부.
- **벤더 락인.** 데이터가 독점 포맷에 살았음; ML이나 비-SQL 처리를 위해 꺼내는 것이 고통.

#### A.2 2차: 데이터 레이크 (2010년대)

Hadoop HDFS, 그다음 S3/GCS/ADLS. 그냥 파일(Parquet, JSON, Avro, CSV)을 저렴한 객체 스토리지에 던져 넣기. 여러 컴퓨트 엔진(Spark, Hive, Presto, Python 노트북)이 같은 파일을 읽음. 유연한 스키마, 저렴한 저장, 벤더 락인 없음.

레이크가 이긴 것: 저장 비용(웨어하우스보다 10-100배 저렴), 스키마 유연성, 다중 엔진 분석, ML 친화성.

레이크가 잃은 것:
- **ACID 트랜잭션 없음.** 동시 라이터가 서로 손상시킴. 쓰기 중 리더는 절반만 쓰여진 데이터를 봄.
- **스키마 강제 없음.** 쓰레기 누적("data swamp").
- **세밀한 update / delete 없음.** Parquet 파일에서 한 행 삭제는 파일 재작성을 요구.
- **인덱싱이나 통계 없음.** 누군가 외부에서 인덱스를 만들지 않으면 모든 쿼리가 풀 스캔.

"데이터 스왐프"는 실제 조직 문제가 되었습니다: 누구도 신뢰하거나 효율적으로 쿼리할 수 없는 페타바이트의 데이터.

#### A.3 3차: 레이크하우스 (2020-)

종합. 레이크의 저장 레이아웃(S3의 파일)을 유지하지만 ACID 트랜잭션, 스키마 진화, 시간 여행, 인덱싱을 제공하는 *테이블 포맷* 을 추가. 세 가지 오픈 구현: **Delta Lake**(Databricks), **Apache Iceberg**(Netflix 기원, 현재 Apache), **Apache Hudi**(Uber 기원, 현재 Apache).

메시지: 레이크 경제학(저렴한 저장, 다중 엔진, 개방형 포맷) 위의 웨어하우스 의미론(ACID, SQL, 성능).

### 이론: 메달리온 원칙

레이크하우스 파이프라인의 표준 조직 패턴.

#### D.1 세 계층

- **Bronze:** 원시 적재 데이터, append-only, schema-on-read. 진실의 원천; 절대 삭제하지 않음. CDC 피드, 원시 API 응답, 덤프된 파일. Bronze 데이터는 지저분함; 그것이 그 목적.
- **Silver:** 정제, 중복 제거, 조인, 스키마에 맞춰짐. 타입 검사, null 처리, 비즈니스 키 강제. 분석가가 바로 쓸 수 있음.
- **Gold:** 비즈니스 집계, 비정규화, 특정 소비를 위해 최적화 — 대시보드, ML 피처, 리포트. 종종 사전 조인된 와이드 테이블.

#### D.2 멱등 재구축

메달리온의 슈퍼파워: 각 계층이 **이전 계층에서 재구축 가능**. Silver에 버그가 있으면 bronze→silver 코드 수정, silver 드롭, bronze에서 재구축. Bronze는 절대 변하지 않음(append-only); silver와 gold는 파생됨.

이것이 아키텍처를 살아남게 만드는 것입니다. 프로덕션 gold의 버그? 한 시간 안에 silver에서 재구축. Silver의 버그? 하루 안에 bronze에서 재구축. Bronze는 당신의 재생 로그.

#### D.3 오케스트레이션 모델

각 계층은 오케스트레이션된 잡(Airflow, Dagster, Prefect)으로 만들어짐. 자연스러운 패턴: bronze 적재는 지속적이거나 시간당 실행; silver와 gold는 스케줄에 따라 증분 재구축. 다운스트림 계층의 실패는 업스트림에 영향 없음 — bronze는 계속 누적.

### 4.1 개념

Lakehouse는 Data Lake의 유연성과 Data Warehouse의 성능/관리 기능을 결합한 아키텍처입니다.

```
┌────────────────────────────────────────────────────────────────┐
│                      Lakehouse Architecture                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   ┌────────────────────────────────────────────────────────┐  │
│   │                   Applications                          │  │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │  │
│   │  │    BI    │ │    ML    │ │  SQL     │ │ Streaming│  │  │
│   │  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │  │
│   └────────────────────────────────────────────────────────┘  │
│                              ↓                                 │
│   ┌────────────────────────────────────────────────────────┐  │
│   │                  Query Engine                           │  │
│   │        (Spark, Presto, Trino, Dremio)                  │  │
│   └────────────────────────────────────────────────────────┘  │
│                              ↓                                 │
│   ┌────────────────────────────────────────────────────────┐  │
│   │              Lakehouse Format Layer                     │  │
│   │     ┌──────────────────────────────────────────────┐   │  │
│   │     │  ACID Transactions │ Schema Enforcement      │   │  │
│   │     │  Time Travel       │ Unified Batch/Streaming │   │  │
│   │     └──────────────────────────────────────────────┘   │  │
│   │           Delta Lake / Apache Iceberg / Apache Hudi    │  │
│   └────────────────────────────────────────────────────────┘  │
│                              ↓                                 │
│   ┌────────────────────────────────────────────────────────┐  │
│   │              Object Storage (Data Lake)                 │  │
│   │                  S3 / GCS / ADLS / HDFS                 │  │
│   └────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 4.2 핵심 기능

| 기능 | 설명 |
|------|------|
| **ACID 트랜잭션** | 데이터 무결성 보장 |
| **스키마 진화** | 스키마 변경 지원 |
| **타임 트래블** | 과거 데이터 버전 조회 |
| **Upsert/Merge** | 효율적인 데이터 갱신 |
| **통합 처리** | 배치 + 스트리밍 단일 테이블 |

---

## 5. Delta Lake

### 이론: 테이블 포맷: Delta Lake, Iceberg, Hudi

레이크하우스를 작동하게 만드는 기술. 세 가지 모두 같은 일반적 접근으로 같은 문제를 해결: 실제 데이터 파일과 함께 각 커밋을 기록하는 *트랜잭션 로그*.

#### C.1 트랜잭션 로그 패턴

```
/table_root/
  _delta_log/                ← 트랜잭션 로그 (Delta Lake 명명; Iceberg는 metadata/ 사용)
    000000.json              ← 커밋 0: "added file part-001.parquet"
    000001.json              ← 커밋 1: "added file part-002.parquet, removed part-001.parquet"
  part-001.parquet           ← 실제 데이터
  part-002.parquet
```

현재 상태에서 테이블을 읽으려면 reader는:
1. 로그 파일 목록.
2. 순서대로 재생하여 현재 "live" 데이터 파일 집합 계산.
3. 그 파일들만 읽음.

쓰려면 writer는:
1. 새 데이터 파일을 stage.
2. 변경을 기록하는 새 로그 항목을 원자적으로 append.
3. Optimistic concurrency: 다른 writer의 로그 항목이 이기면 재시도.

이는 제공:
- **ACID 쓰기.** 로그 항목은 원자적; 부분 쓰기는 보이지 않음.
- **스냅샷 읽기.** 시간 T의 reader는 동시 writer와 무관하게 일관된 스냅샷을 봄.
- **시간 여행.** 옛 커밋 번호 시점의 테이블을 읽음 — 디버깅, 감사, 재생에 유용.
- **스키마 진화.** 스키마는 로그에 있음; 옛 데이터 파일은 자체 스키마를 가짐; reader가 병합.
- **Update와 delete.** "영향받은 파일 재작성 + 옛것을 로그에 removed로 기록"으로 구현.

#### C.2 실제 차이

- **Delta Lake:** 가장 단순, Spark/Databricks와 긴밀하게 결합(이제 좋은 third-party 지원). Databricks 고객의 기본.
- **Apache Iceberg:** 처음부터 다중 엔진을 위해 설계. Trino, Flink, Snowflake에서 더 나은 지원. 가장 "개방적" 선택.
- **Apache Hudi:** 스트리밍 + 빈번한 upsert에 가장 좋음. 증분 처리 primitive가 내장되어 있음.

선택은 종종 raw 능력보다는 생태계로 귀결됩니다. 셋 모두 수렴 중.

### 5.1 Delta Lake 기본

```python
from pyspark.sql import SparkSession
from delta import *

# Delta Lake 설정
# 두 확장 모두 필요합니다: DeltaSparkSessionExtension은 Delta 전용 SQL 명령
# (MERGE, OPTIMIZE)을 추가하고, DeltaCatalog는 Spark가 경로뿐만 아니라
# 이름으로 Delta 테이블을 확인할 수 있게 합니다.
spark = SparkSession.builder \
    .appName("DeltaLake") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# Delta 테이블 생성
# Delta로 쓰면(일반 Parquet 대신) ACID 트랜잭션, 스키마 강제,
# 타임 트래블을 제공하는 _delta_log 트랜잭션 로그가 추가됩니다 —
# 모두 일반 Parquet 파일로는 불가능한 기능입니다.
df = spark.createDataFrame([
    (1, "Alice", 100),
    (2, "Bob", 200),
], ["id", "name", "amount"])

df.write.format("delta").save("/data/delta/users")

# 읽기
delta_df = spark.read.format("delta").load("/data/delta/users")

# Delta 경로 위에 SQL 테이블을 등록하면 BI 도구와 SQL 분석가가
# 물리적 파일 위치를 몰라도 데이터를 쿼리할 수 있습니다.
spark.sql("CREATE TABLE users USING DELTA LOCATION '/data/delta/users'")
spark.sql("SELECT * FROM users").show()
```

### 5.2 Delta Lake 고급 기능

```python
from delta.tables import DeltaTable

# MERGE (Upsert)
# MERGE는 Lakehouse를 일반 Data Lake와 구별하는 핵심 연산입니다.
# 이것 없이는 어떤 업데이트든 전체 읽기 → 필터 → 합집합 → 전체 쓰기가 필요하며,
# 비용이 많이 들고 원자적이지 않습니다.
delta_table = DeltaTable.forPath(spark, "/data/delta/users")

new_data = spark.createDataFrame([
    (1, "Alice Updated", 150),  # 업데이트
    (3, "Charlie", 300),        # 삽입
], ["id", "name", "amount"])

# 머지 조건은 매칭을 위한 비즈니스 키를 정의합니다.
# whenMatchedUpdate는 기존 레코드를 처리(SCD Type 1 덮어쓰기)하고,
# whenNotMatchedInsert는 새 레코드를 처리합니다 — 모두 단일 원자적 패스로 수행됩니다.
delta_table.alias("target").merge(
    new_data.alias("source"),
    "target.id = source.id"
).whenMatchedUpdate(set={
    "name": "source.name",
    "amount": "source.amount"
}).whenNotMatchedInsert(values={
    "id": "source.id",
    "name": "source.name",
    "amount": "source.amount"
}).execute()

# Time Travel (과거 버전 조회)
# 타임 트래블은 데이터 문제 디버깅에 매우 유용합니다 — 현재 상태와
# 이전 버전을 비교하여 정확히 무엇이 변경되었는지 확인할 수 있습니다.
# 버전 번호로
df_v0 = spark.read.format("delta") \
    .option("versionAsOf", 0) \
    .load("/data/delta/users")

# 타임스탬프로 — "어제 데이터가 맞았는데"라는 상황에서
# 정확한 버전 번호를 모를 때 유용합니다
df_yesterday = spark.read.format("delta") \
    .option("timestampAsOf", "2024-01-14") \
    .load("/data/delta/users")

# 히스토리 확인
delta_table.history().show()

# Vacuum (오래된 파일 정리)
# 168시간(7일)은 스토리지 비용과 타임 트래블 필요성 사이의 균형입니다.
# 보존 기간이 짧을수록 스토리지가 절약되지만 이전 버전 쿼리나 롤백이 불가능합니다.
# 동시 읽기가 이전 파일에 의존하지 않음을 확인하지 않고
# 기본 7일 임계값 이하로 vacuum을 실행하지 마세요.
delta_table.vacuum(retentionHours=168)  # 7일 보존

# 스키마 진화 — mergeSchema=true는 기존 읽기를 깨지 않고
# 유입 데이터의 새 컬럼을 추가할 수 있게 합니다.
# 이것 없이 새 컬럼이 있는 쓰기는 스키마 불일치 오류로 실패합니다.
spark.read.format("delta") \
    .option("mergeSchema", "true") \
    .load("/data/delta/users")

# Z-Order 최적화 (쿼리 성능)
# Z-ordering은 지정된 컬럼을 기반으로 관련 데이터를 같은 파일에 배치하여,
# 해당 컬럼의 필터에 대한 쿼리 프루닝을 크게 향상시킵니다.
# WHERE 절에 가장 자주 나타나는 컬럼을 선택하세요.
delta_table.optimize().executeZOrderBy("date", "customer_id")
```

---

## 6. Apache Iceberg

### 6.1 Iceberg 기본

```python
from pyspark.sql import SparkSession

# Iceberg는 카탈로그 중심 설계를 사용합니다: 모든 테이블 메타데이터가 카탈로그에
# (여기서는 Hive Metastore) 저장되어 테이블이 엔진에 독립적이 됩니다 — 같은 테이블을
# Spark, Trino, Flink, Dremio에서 데이터 복제 없이 쿼리할 수 있습니다.
spark = SparkSession.builder \
    .appName("Iceberg") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.iceberg.type", "hive") \
    .config("spark.sql.catalog.iceberg.uri", "thrift://localhost:9083") \
    .getOrCreate()

# Iceberg 테이블 생성
# bucket(16, id)는 "숨겨진 파티션(hidden partition)"입니다: Iceberg가 id를 16개의 버킷으로
# 해싱하여 균등한 데이터 분산을 보장합니다. Hive 스타일 파티셔닝과 달리, 쿼리 시
# 파티셔닝 체계를 알 필요가 없습니다 — Iceberg가 필터 프레디케이트를 자동으로
# 재작성합니다 (파티션 진화, partition evolution).
spark.sql("""
    CREATE TABLE iceberg.db.users (
        id INT,
        name STRING,
        amount DECIMAL(10, 2)
    ) USING ICEBERG
    PARTITIONED BY (bucket(16, id))
""")

# 데이터 삽입
spark.sql("""
    INSERT INTO iceberg.db.users VALUES
    (1, 'Alice', 100.00),
    (2, 'Bob', 200.00)
""")

# Time Travel — Iceberg는 스냅샷 기반 모델을 사용합니다 (Delta의 로그 기반과 달리),
# 스냅샷당 매니페스트 목록을 저장합니다. 어떤 엔진도 스냅샷 메타데이터를
# 직접 읽을 수 있어 멀티 엔진 타임 트래블이 가능합니다.
spark.sql("SELECT * FROM iceberg.db.users VERSION AS OF 1").show()
spark.sql("SELECT * FROM iceberg.db.users TIMESTAMP AS OF '2024-01-15'").show()

# 스냅샷 메타데이터 테이블은 작업 유형, 추가/삭제된 파일,
# 요약 통계를 포함한 전체 버전 이력을 제공합니다
spark.sql("SELECT * FROM iceberg.db.users.snapshots").show()
```

### 6.2 Delta Lake vs Iceberg 비교

| 특성 | Delta Lake | Iceberg |
|------|------------|---------|
| **개발사** | Databricks | Netflix → Apache |
| **호환성** | Spark 중심 | 엔진 독립적 |
| **메타데이터** | 트랜잭션 로그 | 스냅샷 기반 |
| **파티션 진화** | 제한적 | 강력한 지원 |
| **숨겨진 파티션** | 미지원 | 지원 |
| **커뮤니티** | Databricks 생태계 | 다양한 벤더 |

---

## 7. 모던 데이터 스택

### 7.1 아키텍처 패턴

```
┌─────────────────────────────────────────────────────────────────┐
│                   Modern Data Stack                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Data Sources                                                  │
│   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                  │
│   │ SaaS   │ │Database│ │  API   │ │  IoT   │                  │
│   └────┬───┘ └───┬────┘ └───┬────┘ └───┬────┘                  │
│        └─────────┴──────────┴──────────┘                        │
│                         ↓                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              Ingestion (EL)                              │  │
│   │        Fivetran / Airbyte / Stitch                       │  │
│   └─────────────────────────────────────────────────────────┘  │
│                         ↓                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │           Cloud Data Warehouse / Lakehouse              │  │
│   │        Snowflake / BigQuery / Databricks                │  │
│   └─────────────────────────────────────────────────────────┘  │
│                         ↓                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              Transformation (T)                          │  │
│   │                      dbt                                 │  │
│   └─────────────────────────────────────────────────────────┘  │
│                         ↓                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                 BI / Analytics                           │  │
│   │        Looker / Tableau / Metabase / Mode               │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 연습 문제

### 문제 1: 아키텍처 선택
다음 요구사항에 맞는 아키텍처를 선택하고 이유를 설명하세요:
- 일일 10TB의 로그 데이터
- ML 모델 학습에 사용
- 원본 데이터 5년 보존 필요

### 문제 2: Delta Lake 구현
고객 데이터에 대한 SCD Type 2를 Delta Lake MERGE로 구현하세요.

---

## 요약

| 아키텍처 | 특징 | 사용 사례 |
|----------|------|----------|
| **Data Warehouse** | 구조화, SQL 최적화 | BI, 리포팅 |
| **Data Lake** | 모든 데이터, 저비용 | ML, 원본 보존 |
| **Lakehouse** | Lake + Warehouse 장점 | 통합 분석 |

---

## 참고 자료

- [Delta Lake Documentation](https://docs.delta.io/)
- [Apache Iceberg Documentation](https://iceberg.apache.org/)
- [Databricks Lakehouse](https://www.databricks.com/product/data-lakehouse)

---

[← 이전: 10. Kafka 스트리밍](10_Kafka_Streaming.md) | [다음: 12. dbt 변환 도구 →](12_dbt_Transformation.md)
