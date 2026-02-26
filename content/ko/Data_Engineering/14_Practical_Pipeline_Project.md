# 실전 파이프라인 프로젝트

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Airflow, Spark, dbt, Great Expectations 등 여러 도구를 통합하여 실제 비즈니스 요구사항을 충족하는 종단간(End-to-End) 데이터 파이프라인을 설계할 수 있습니다.
2. 클라우드 데이터 레이크에 메달리온 아키텍처(Medallion Architecture, 브론즈/실버/골드 레이어)를 구현하여 원시 데이터를 분석 가능한 데이터셋으로 단계적으로 정제할 수 있습니다.
3. 수집, 변환, 품질 검증, 알림 작업을 조율하는 다단계 파이프라인 DAG를 Airflow에서 오케스트레이션할 수 있습니다.
4. 관계형 데이터베이스, 객체 스토리지(S3), 이벤트 스트림(Kafka) 등 이종 소스에 대한 데이터 수집 패턴을 설계할 수 있습니다.
5. 파이프라인 장애와 데이터 품질 저하를 감지하고 Slack 또는 이메일로 알림을 전달하는 모니터링 및 알림 전략을 적용할 수 있습니다.
6. 프로덕션 분석 파이프라인을 구축할 때 배치(Batch)와 스트리밍(Streaming) 수집 방식의 트레이드오프를 평가할 수 있습니다.

---

## 개요

이 레슨에서는 지금까지 배운 모든 기술을 통합하여 실제 데이터 파이프라인을 구축합니다. Airflow로 오케스트레이션, Spark로 대규모 처리, dbt로 변환, Great Expectations로 품질 검증을 수행하는 E2E 파이프라인을 설계합니다.

---

## 1. 프로젝트 개요

### 1.1 시나리오

```
┌────────────────────────────────────────────────────────────────┐
│                    E-Commerce 분석 파이프라인                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   비즈니스 요구사항:                                            │
│   - 일일 매출 분석 대시보드                                     │
│   - 고객 세그먼테이션                                           │
│   - 재고 최적화 알림                                            │
│                                                                │
│   데이터 소스:                                                  │
│   - PostgreSQL: 주문, 고객, 상품                                │
│   - S3: 클릭스트림 로그 (JSON)                                  │
│   - Kafka: 실시간 재고 이벤트                                   │
│                                                                │
│   출력:                                                        │
│   - Data Warehouse: Snowflake/BigQuery                         │
│   - BI Dashboard: Looker/Tableau                               │
│   - Alert System: Slack/Email                                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Data Sources                                                  │
│   ┌────────┐ ┌────────┐ ┌────────┐                             │
│   │PostgreSQL│ S3 Logs│ │ Kafka  │                              │
│   └────┬───┘ └───┬────┘ └───┬────┘                             │
│        │         │          │                                   │
│        └─────────┴──────────┘                                   │
│                   ↓                                             │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                    Airflow                               │  │
│   │    (Orchestration)                                       │  │
│   └─────────────────────────────────────────────────────────┘  │
│                   ↓                                             │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                 Data Lake (S3)                           │  │
│   │   ┌─────────┐  ┌─────────┐  ┌─────────┐                 │  │
│   │   │ Bronze  │→│ Silver  │→│  Gold   │                  │  │
│   │   │  (Raw)  │  │(Cleaned)│  │(Curated)│                  │  │
│   │   └─────────┘  └─────────┘  └─────────┘                 │  │
│   └─────────────────────────────────────────────────────────┘  │
│                   ↓                                             │
│   ┌───────────────────────────────────────────────────────────┐│
│   │ Spark (Processing) │ dbt (Transform) │ GE (Quality)      ││
│   └───────────────────────────────────────────────────────────┘│
│                   ↓                                             │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │               Data Warehouse                             │  │
│   │         (Snowflake / BigQuery)                           │  │
│   └─────────────────────────────────────────────────────────┘  │
│                   ↓                                             │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐                      │
│   │ BI Tool  │ │ ML Models│ │ Alerts   │                      │
│   └──────────┘ └──────────┘ └──────────┘                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 프로젝트 구조

### 2.1 디렉토리 구조

```
ecommerce_pipeline/
├── airflow/
│   ├── dags/
│   │   ├── daily_etl_dag.py
│   │   ├── hourly_streaming_dag.py
│   │   └── data_quality_dag.py
│   └── plugins/
│       └── custom_operators.py
│
├── spark/
│   ├── jobs/
│   │   ├── extract_postgres.py
│   │   ├── process_clickstream.py
│   │   └── aggregate_daily.py
│   └── utils/
│       └── spark_utils.py
│
├── dbt/
│   ├── dbt_project.yml
│   ├── models/
│   │   ├── staging/
│   │   │   ├── stg_orders.sql
│   │   │   ├── stg_customers.sql
│   │   │   └── stg_products.sql
│   │   ├── intermediate/
│   │   │   └── int_order_items.sql
│   │   └── marts/
│   │       ├── fct_orders.sql
│   │       ├── dim_customers.sql
│   │       └── agg_daily_sales.sql
│   └── tests/
│       └── assert_positive_amounts.sql
│
├── great_expectations/
│   ├── expectations/
│   │   ├── orders_suite.json
│   │   └── customers_suite.json
│   └── checkpoints/
│       └── daily_checkpoint.yml
│
├── docker/
│   ├── docker-compose.yml
│   └── Dockerfile.spark
│
├── tests/
│   ├── test_spark_jobs.py
│   └── test_dbt_models.py
│
└── requirements.txt
```

---

## 3. Airflow DAG 구현

### 3.1 메인 ETL DAG

```python
# airflow/dags/daily_etl_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.dbt.cloud.operators.dbt import DbtCloudRunJobOperator
from airflow.utils.task_group import TaskGroup

# default_args는 DAG의 모든 태스크에 상속되어 반복 설정을 줄여줍니다.
# retries=2, 5분 지연은 일시적 실패(네트워크 순단, 임시 리소스 경합)를
# 사람의 개입 없이 처리합니다.
# depends_on_past=False는 각 일별 실행이 독립적임을 의미합니다 — 월요일 실패가
# 화요일 실행을 차단하지 않습니다
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'email': ['data-alerts@company.com'],
    'email_on_failure': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='daily_ecommerce_pipeline',
    default_args=default_args,
    description='일일 이커머스 데이터 파이프라인',
    schedule_interval='0 6 * * *',  # 매일 오전 6시
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['production', 'etl', 'daily'],
    max_active_runs=1,
) as dag:

    start = EmptyOperator(task_id='start')

    # ============================================
    # Extract: 데이터 소스에서 추출
    # ============================================
    # TaskGroup은 관련 태스크를 Airflow UI에서 접을 수 있는 단위로 정리합니다.
    # 이는 순전히 시각적 목적입니다(스케줄링 경계가 아님) — 내부 모든 태스크는
    # 독립적으로 실행됩니다. 그룹화 없이 15개 이상의 태스크는 Graph 뷰에서
    # 읽기 어려워집니다
    with TaskGroup(group_id='extract') as extract_group:

        extract_orders = SparkSubmitOperator(
            task_id='extract_orders',
            application='/opt/spark/jobs/extract_postgres.py',
            conn_id='spark_default',
            application_args=[
                '--table', 'orders',
                '--date', '{{ ds }}',
                '--output', 's3://data-lake/bronze/orders/{{ ds }}/'
            ],
        )

        extract_customers = SparkSubmitOperator(
            task_id='extract_customers',
            application='/opt/spark/jobs/extract_postgres.py',
            conn_id='spark_default',
            application_args=[
                '--table', 'customers',
                '--output', 's3://data-lake/bronze/customers/'
            ],
        )

        extract_products = SparkSubmitOperator(
            task_id='extract_products',
            application='/opt/spark/jobs/extract_postgres.py',
            conn_id='spark_default',
            application_args=[
                '--table', 'products',
                '--output', 's3://data-lake/bronze/products/'
            ],
        )

        extract_clickstream = SparkSubmitOperator(
            task_id='extract_clickstream',
            application='/opt/spark/jobs/process_clickstream.py',
            conn_id='spark_default',
            application_args=[
                '--date', '{{ ds }}',
                '--input', 's3://raw-logs/clickstream/{{ ds }}/',
                '--output', 's3://data-lake/bronze/clickstream/{{ ds }}/'
            ],
        )

    # ============================================
    # Quality Check: Bronze 레이어 품질 검증
    # ============================================
    # Bronze와 Silver 사이의 품질 게이트: 데이터 문제(파일 누락, 스키마 드리프트,
    # null 급증)가 다운스트림으로 전파되기 전에 포착합니다. Bronze에서 불량 데이터를
    # 수정하는 것이 대시보드와 ML 모델에 이미 공급된 손상된 Gold 집계를 수정하는
    # 것보다 10배 저렴합니다
    with TaskGroup(group_id='quality_bronze') as quality_bronze:

        def run_great_expectations(checkpoint_name: str, **kwargs):
            import great_expectations as gx
            context = gx.get_context()
            result = context.run_checkpoint(checkpoint_name=checkpoint_name)
            if not result.success:
                raise ValueError(f"Quality check failed: {checkpoint_name}")

        check_orders = PythonOperator(
            task_id='check_orders_quality',
            python_callable=run_great_expectations,
            op_kwargs={'checkpoint_name': 'bronze_orders_checkpoint'},
        )

        check_customers = PythonOperator(
            task_id='check_customers_quality',
            python_callable=run_great_expectations,
            op_kwargs={'checkpoint_name': 'bronze_customers_checkpoint'},
        )

    # ============================================
    # Transform: Spark로 Silver 레이어 생성
    # ============================================
    # Silver 레이어 = 정제, 중복 제거, 타입 표준화된 데이터.
    # Spark는 데이터 레이크의 파일 기반 변환을 처리하므로 여기서 사용합니다(dbt 아님).
    # dbt는 웨어하우스 SQL 엔진에서 동작합니다.
    # 이 분리는 각 도구가 강점 영역에서 작동하게 합니다
    with TaskGroup(group_id='transform_spark') as transform_spark:

        process_orders = SparkSubmitOperator(
            task_id='process_orders',
            application='/opt/spark/jobs/process_orders.py',
            application_args=[
                '--input', 's3://data-lake/bronze/orders/{{ ds }}/',
                '--output', 's3://data-lake/silver/orders/{{ ds }}/'
            ],
        )

        aggregate_daily = SparkSubmitOperator(
            task_id='aggregate_daily',
            application='/opt/spark/jobs/aggregate_daily.py',
            application_args=[
                '--date', '{{ ds }}',
                '--output', 's3://data-lake/silver/daily_aggregates/{{ ds }}/'
            ],
        )

    # ============================================
    # Transform: dbt로 Gold 레이어 생성
    # ============================================
    # Gold 레이어 = 비즈니스 준비 집계와 차원 모델.
    # dbt는 웨어하우스 측 SQL 변환(스테이징 → 마트)을 처리하고,
    # Spark는 위의 파일 기반 레이크 변환을 처리합니다.
    # 이 분리를 통해 각 도구는 자신의 강점 영역에서 작동합니다
    with TaskGroup(group_id='transform_dbt') as transform_dbt:

        def run_dbt_command(command: str, **kwargs):
            # subprocess 실행: dbt CLI가 표준 인터페이스이기 때문입니다.
            # 프로덕션에서는 셸 인젝션 위험을 피하고 더 나은 오류 보고를 위해
            # DbtCloudRunJobOperator 또는 dbt Python API를 선호하세요
            import subprocess
            result = subprocess.run(
                f"cd /opt/dbt && dbt {command} --profiles-dir /opt/dbt",
                shell=True,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise Exception(f"dbt failed: {result.stderr}")
            print(result.stdout)

        dbt_run = PythonOperator(
            task_id='dbt_run',
            python_callable=run_dbt_command,
            op_kwargs={'command': 'run --select staging marts'},
        )

        # dbt test는 dbt run 후에 실행되어 새로 구체화된 모델을 검증합니다.
        # 이는 변환 로직 자체에서 도입된 데이터 품질 저하를 포착합니다
        # (예: 예상치 못한 NULL을 생성하는 JOIN)
        dbt_test = PythonOperator(
            task_id='dbt_test',
            python_callable=run_dbt_command,
            op_kwargs={'command': 'test'},
        )

        dbt_run >> dbt_test

    # ============================================
    # Quality Check: Gold 레이어 품질 검증
    # ============================================
    # Gold 레이어의 두 번째 품질 게이트는 변환에서 도입된 문제(잘못된 조인,
    # 집계 버그)를 포착합니다. Bronze와 Gold 모두에 품질 게이트를 두면
    # 심층 방어(Defense-in-depth)가 됩니다: Bronze는 소스 문제를 포착하고,
    # Gold는 변환 문제를 포착합니다
    quality_gold = PythonOperator(
        task_id='quality_gold',
        python_callable=run_great_expectations,
        op_kwargs={'checkpoint_name': 'gold_checkpoint'},
    )

    # ============================================
    # Notify: 완료 알림
    # ============================================
    def send_completion_notification(**kwargs):
        import requests
        webhook_url = "https://hooks.slack.com/services/xxx"
        message = {
            "text": f"Daily pipeline completed for {kwargs['ds']}"
        }
        requests.post(webhook_url, json=message)

    notify = PythonOperator(
        task_id='notify_completion',
        python_callable=send_completion_notification,
    )

    end = EmptyOperator(task_id='end')

    # 태스크 의존성은 메달리온 아키텍처를 따릅니다:
    # 추출 → 품질(Bronze) → 변환(Silver) → 변환(Gold) → 품질(Gold) → 알림
    # 이 선형 체인은 각 레이어 전환 전에 데이터 품질이 검증되어
    # 불량 데이터가 다운스트림 소비자에게 전파되지 않도록 합니다
    start >> extract_group >> quality_bronze >> transform_spark >> transform_dbt >> quality_gold >> notify >> end
```

---

## 4. Spark 처리 작업

### 4.1 데이터 추출 작업

```python
# spark/jobs/extract_postgres.py
from pyspark.sql import SparkSession
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', required=True)
    parser.add_argument('--date', required=False)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    spark = SparkSession.builder \
        .appName(f"Extract {args.table}") \
        .getOrCreate()

    # JDBC 읽기 설정
    # 프로덕션에서는 시크릿 매니저(AWS Secrets Manager, Vault)에 자격 증명을 저장하고
    # 환경 변수로 주입하세요 — 절대 비밀번호를 하드코딩하지 마세요
    jdbc_url = "jdbc:postgresql://postgres:5432/ecommerce"
    properties = {
        "user": "postgres",
        "password": "password",
        "driver": "org.postgresql.Driver"
    }

    # 날짜가 지정된 경우 증분 추출: 해당 날짜에 업데이트된 행만 가져와
    # 대형 테이블에 비용이 많이 드는 전체 테이블 스캔을 방지합니다.
    # 서브쿼리 별칭 "AS t"는 Spark의 JDBC 리더가 푸시다운 쿼리에 요구합니다
    if args.date:
        query = f"""
            (SELECT * FROM {args.table}
             WHERE DATE(updated_at) = '{args.date}') AS t
        """
    else:
        query = args.table

    # 데이터 읽기
    df = spark.read.jdbc(
        url=jdbc_url,
        table=query,
        properties=properties
    )

    # Bronze 레이어에 Parquet으로 저장.
    # mode("overwrite")는 각 날짜 파티션이 자체 경로에 기록되기 때문에 안전합니다
    # (Airflow 템플릿으로), 그 날의 데이터만 덮어씁니다
    df.write \
        .mode("overwrite") \
        .parquet(args.output)

    print(f"Extracted {df.count()} rows from {args.table}")
    spark.stop()


if __name__ == "__main__":
    main()
```

### 4.2 클릭스트림 처리

```python
# spark/jobs/process_clickstream.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    # AQE(Adaptive Query Execution)는 실제 데이터 크기에 따라 런타임에
    # 셔플 파티션과 조인 전략을 동적으로 조정합니다 — 변동하는 일별 볼륨에 대해
    # spark.sql.shuffle.partitions를 수동으로 튜닝할 필요가 없습니다
    spark = SparkSession.builder \
        .appName("Process Clickstream") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

    # 스키마를 명시적으로 정의하면 전체 JSON 데이터셋을 두 번 읽어야 하는
    # 비용이 많이 드는 스키마 추론 패스를 방지합니다. 대형 클릭스트림 로그에서
    # (일일 수백만 이벤트) 이것이 처리 시간을 수 분 절약할 수 있습니다
    schema = StructType([
        StructField("event_id", StringType()),
        StructField("user_id", StringType()),
        StructField("session_id", StringType()),
        StructField("event_type", StringType()),
        StructField("page_url", StringType()),
        StructField("timestamp", TimestampType()),
        StructField("properties", MapType(StringType(), StringType())),
    ])

    # JSON 읽기
    df = spark.read.schema(schema).json(args.input)

    # 정제 및 변환:
    # - Null 필터는 다운스트림에서 조인 실패를 일으키는 잘못된 이벤트를 제거합니다
    # - event_date/hour 추출은 효율적인 쿼리를 위한 시간 기반 파티셔닝을 가능하게 합니다
    # - getItem("product_id")는 중첩된 properties 맵을 최상위 컬럼으로 평탄화합니다.
    #   중첩 JSON을 반복적으로 파싱하는 것보다 훨씬 빠르게 쿼리할 수 있습니다
    processed_df = df \
        .filter(col("event_id").isNotNull()) \
        .filter(col("user_id").isNotNull()) \
        .withColumn("event_date", to_date(col("timestamp"))) \
        .withColumn("event_hour", hour(col("timestamp"))) \
        .withColumn("product_id", col("properties").getItem("product_id")) \
        .dropDuplicates(["event_id"]) \
        .select(
            "event_id",
            "user_id",
            "session_id",
            "event_type",
            "page_url",
            "product_id",
            "event_date",
            "event_hour",
            "timestamp"
        )

    # 이중 파티셔닝(날짜 + 시간)으로 일별 배치 쿼리와 시간별 드릴다운 쿼리 모두
    # 효율적으로 프루닝할 수 있습니다. 과도한 파티셔닝(예: 분 단위)은 너무 많은
    # 소형 파일을 생성하여 읽기 성능을 저하시킵니다
    processed_df.write \
        .mode("overwrite") \
        .partitionBy("event_date", "event_hour") \
        .parquet(args.output)

    print(f"Processed {processed_df.count()} events")
    spark.stop()


if __name__ == "__main__":
    main()
```

### 4.3 일별 집계

```python
# spark/jobs/aggregate_daily.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    spark = SparkSession.builder \
        .appName("Daily Aggregation") \
        .getOrCreate()

    # Silver 레이어 읽기 — customers와 products는 전체 스냅샷으로 로드합니다
    # (날짜 파티션 없음): 천천히 변하는 디멘션이기 때문입니다.
    # orders는 증분 처리를 위해 날짜 파티셔닝됩니다
    orders = spark.read.parquet(f"s3://data-lake/silver/orders/{args.date}/")
    customers = spark.read.parquet("s3://data-lake/silver/customers/")
    products = spark.read.parquet("s3://data-lake/silver/products/")

    # 일별 매출 집계 — Gold 레이어에서 이 집계를 사전 계산하면
    # BI 대시보드가 원시 팩트 테이블에 비용이 많이 드는 GROUP BY를 실행하지 않고
    # 직접 읽을 수 있어 쿼리 지연이 분에서 초로 줄어듭니다
    daily_sales = orders \
        .filter(col("order_date") == args.date) \
        .join(products, "product_id") \
        .groupBy(
            col("order_date"),
            col("category"),
            col("region")
        ) \
        .agg(
            count("order_id").alias("order_count"),
            sum("amount").alias("total_revenue"),
            avg("amount").alias("avg_order_value"),
            # countDistinct는 count보다 비용이 많이 들지만
            # 주문량 대비 고객 도달 범위를 이해하는 데 중요합니다
            countDistinct("customer_id").alias("unique_customers")
        )

    # 고객 세그먼트 집계 — daily_sales와 분리됩니다.
    # 다른 세분도(세그먼트 수준 vs 카테고리/지역)를 가지며
    # 다른 비즈니스 질문(세그먼테이션 vs 상품 분석)을 처리하기 때문입니다
    customer_segments = orders \
        .filter(col("order_date") == args.date) \
        .join(customers, "customer_id") \
        .groupBy("customer_segment") \
        .agg(
            count("order_id").alias("orders"),
            sum("amount").alias("revenue")
        )

    # 저장
    daily_sales.write \
        .mode("overwrite") \
        .parquet(f"{args.output}/daily_sales/")

    customer_segments.write \
        .mode("overwrite") \
        .parquet(f"{args.output}/customer_segments/")

    spark.stop()


if __name__ == "__main__":
    main()
```

---

## 5. dbt 모델

### 5.1 스테이징 모델

```sql
-- dbt/models/staging/stg_orders.sql
{{
    config(
        -- view로 구체화하면 빠른 반복이 가능합니다: 개발 시 즉시 재빌드됩니다.
        -- 높은 쿼리 볼륨의 프로덕션에서는 모든 쿼리마다 Silver를 다시 읽는 것을
        -- 피하기 위해 table 또는 incremental로 전환하는 것을 고려하세요
        materialized='view',
        schema='staging'
    )
}}

WITH source AS (
    SELECT * FROM {{ source('silver', 'orders') }}
),

cleaned AS (
    SELECT
        order_id,
        customer_id,
        product_id,
        -- 명시적 CAST는 소스 시스템이 데이터를 어떻게 직렬화했는지와 관계없이
        -- 일관된 타입을 보장합니다 (일부 소스는 날짜를 문자열로 내보냅니다)
        CAST(order_date AS DATE) AS order_date,
        CAST(amount AS DECIMAL(12, 2)) AS amount,
        CAST(quantity AS INT) AS quantity,
        status,
        CURRENT_TIMESTAMP AS loaded_at
    FROM source
    -- 스테이징 레이어에서 null PK와 비양수 금액을 필터링하여
    -- 모든 다운스트림 모델이 매번 재검증 없이 이 불변 조건을 신뢰할 수 있습니다
    WHERE order_id IS NOT NULL
      AND amount > 0
)

SELECT * FROM cleaned
```

### 5.2 마트 모델

```sql
-- dbt/models/marts/fct_orders.sql
{{
    config(
        -- 증분 구체화는 마지막 실행 이후 새 데이터만 처리합니다.
        -- 일일 수백만 행이 증가하는 팩트 테이블에 매우 중요합니다.
        -- 필요할 때 전체 재빌드(dbt run --full-refresh)를 강제할 수 있습니다
        materialized='incremental',
        unique_key='order_id',
        schema='marts',
        -- 일별 파티셔닝은 파이프라인의 일일 스케줄과 일치합니다;
        -- 대부분의 BI 쿼리는 날짜 범위로 필터링하므로 파티션 프루닝이
        -- 클라우드 웨어하우스에서 주요 비용 절감을 제공합니다
        partition_by={
            'field': 'order_date',
            'data_type': 'date',
            'granularity': 'day'
        }
    )
}}

WITH orders AS (
    SELECT * FROM {{ ref('stg_orders') }}
),

customers AS (
    SELECT * FROM {{ ref('dim_customers') }}
),

products AS (
    SELECT * FROM {{ ref('dim_products') }}
)

SELECT
    o.order_id,
    o.order_date,

    -- 고객 정보 — 쿼리 편의를 위해 팩트 테이블에 비정규화합니다.
    -- BI 도구에서 수동 JOIN 없이 세그먼트와 지역으로 필터링/그룹화할 수 있습니다
    o.customer_id,
    c.customer_name,
    c.customer_segment,
    c.region,

    -- 상품 정보
    o.product_id,
    p.product_name,
    p.category,

    -- 측정값 — 팩트 수준에서 비용과 수익을 계산하면 Gold 레이어 요약에서
    -- 상품 비용을 다시 조인하지 않고 직접 집계할 수 있습니다
    o.quantity,
    o.amount AS order_amount,
    p.unit_cost * o.quantity AS cost_amount,
    o.amount - (p.unit_cost * o.quantity) AS profit_amount,

    -- 상태
    o.status,

    -- 메타데이터
    o.loaded_at

-- LEFT JOIN은 디멘션 데이터가 없을 때도 주문을 보존합니다 (예:
-- dim_customers에 아직 없는 신규 고객). INNER JOIN은 이런 주문들을
-- 조용히 제거하여 매출 과소 집계를 야기합니다
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN products p ON o.product_id = p.product_id

{% if is_incremental() %}
WHERE o.order_date > (SELECT MAX(order_date) FROM {{ this }})
{% endif %}
```

```sql
-- dbt/models/marts/agg_daily_sales.sql
{{
    config(
        materialized='table',
        schema='marts'
    )
}}

WITH orders AS (
    SELECT * FROM {{ ref('fct_orders') }}
)

SELECT
    order_date,
    category,
    region,
    customer_segment,

    -- 주문 메트릭
    COUNT(DISTINCT order_id) AS total_orders,
    COUNT(DISTINCT customer_id) AS unique_customers,
    SUM(quantity) AS total_quantity,

    -- 매출 메트릭
    SUM(order_amount) AS total_revenue,
    AVG(order_amount) AS avg_order_value,
    SUM(profit_amount) AS total_profit,

    -- 총 매출이 0일 때(예: 취소 주문만 있는 날, 완료된 판매가 없는 새로 출시된 지역)
    -- 0으로 나누기를 방지합니다. NULLIF는 0 대신 NULL을 반환하여
    -- 나누기 결과를 데이터베이스 오류 대신 NULL로 만들어 BI 도구에 더 안전합니다
    ROUND(SUM(profit_amount) / NULLIF(SUM(order_amount), 0) * 100, 2) AS profit_margin_pct,

    -- 기간 비교 (dbt_utils 사용 시)
    -- {{ dbt_utils.date_spine(...) }}

    CURRENT_TIMESTAMP AS updated_at

FROM orders
GROUP BY
    order_date,
    category,
    region,
    customer_segment
```

---

## 6. 품질 검증

### 6.1 Great Expectations Suite

```python
# great_expectations/create_expectations.py
import great_expectations as gx

context = gx.get_context()

# Orders Suite
orders_suite = context.add_expectation_suite("orders_suite")
validator = context.get_validator(
    batch_request={"datasource": "orders_datasource", ...},
    expectation_suite_name="orders_suite"
)

# 기본 검증
validator.expect_column_values_to_not_be_null("order_id")
validator.expect_column_values_to_be_unique("order_id")
validator.expect_column_values_to_not_be_null("customer_id")
validator.expect_column_values_to_not_be_null("amount")

# 값 범위
validator.expect_column_values_to_be_between("amount", min_value=0, max_value=1000000)
validator.expect_column_values_to_be_between("quantity", min_value=1, max_value=100)

# 허용 값
validator.expect_column_values_to_be_in_set(
    "status",
    ["pending", "processing", "shipped", "delivered", "cancelled"]
)

# 테이블 수준
validator.expect_table_row_count_to_be_between(min_value=1000, max_value=10000000)

# 참조 무결성
# validator.expect_column_values_to_be_in_set(
#     "customer_id",
#     customer_ids_from_dim_table
# )

validator.save_expectation_suite(discard_failed_expectations=False)
```

### 6.2 dbt 테스트

```yaml
# dbt/models/marts/_schema.yml
version: 2

models:
  - name: fct_orders
    description: "주문 팩트 테이블"
    tests:
      - dbt_utils.recency:
          datepart: day
          field: order_date
          interval: 1
    columns:
      - name: order_id
        tests:
          - unique
          - not_null
      - name: customer_id
        tests:
          - not_null
          - relationships:
              to: ref('dim_customers')
              field: customer_id
      - name: order_amount
        tests:
          - not_null
          - dbt_utils.expression_is_true:
              expression: ">= 0"
      - name: profit_amount
        tests:
          - dbt_utils.expression_is_true:
              expression: "<= order_amount"

  - name: agg_daily_sales
    tests:
      - dbt_utils.unique_combination_of_columns:
          combination_of_columns:
            - order_date
            - category
            - region
            - customer_segment
```

---

## 7. 모니터링 및 알림

### 7.1 모니터링 대시보드

```python
# monitoring/metrics_collector.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import json

@dataclass
class PipelineMetrics:
    """파이프라인 메트릭"""
    pipeline_name: str
    run_date: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    records_processed: int = 0
    quality_score: float = 0.0
    errors: list = None

    def to_dict(self):
        return {
            "pipeline_name": self.pipeline_name,
            "run_date": self.run_date,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (self.end_time - self.start_time).seconds if self.end_time else None,
            "status": self.status,
            "records_processed": self.records_processed,
            "quality_score": self.quality_score,
            "errors": self.errors or []
        }


def push_metrics_to_prometheus(metrics: PipelineMetrics):
    """Prometheus에 메트릭 푸시"""
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    # 푸시마다 새 레지스트리를 사용하여 이전 실행의 오래된 메트릭을 방지합니다.
    # 없으면 실패한 파이프라인의 메트릭 레이블이 남아 Grafana 대시보드를 혼동시킬 수 있습니다
    registry = CollectorRegistry()

    # Gauge(Counter 아님): 파이프라인 소요 시간은 순간값이며
    # 계속 증가하는 카운터가 아닙니다. Gauge를 통해 Prometheus가
    # 실행이 점점 느려지는지 추적할 수 있습니다
    duration = Gauge(
        'pipeline_duration_seconds',
        'Pipeline duration',
        ['pipeline_name'],
        registry=registry
    )
    duration.labels(pipeline_name=metrics.pipeline_name).set(
        (metrics.end_time - metrics.start_time).seconds if metrics.end_time else 0
    )

    records = Gauge(
        'pipeline_records_processed',
        'Records processed',
        ['pipeline_name'],
        registry=registry
    )
    records.labels(pipeline_name=metrics.pipeline_name).set(metrics.records_processed)

    quality = Gauge(
        'pipeline_quality_score',
        'Quality score',
        ['pipeline_name'],
        registry=registry
    )
    # 품질 점수로 저하 추세에 대한 Grafana 알림을 활성화합니다 — 예:
    # 3회 연속 실행에서 품질이 95% 미만으로 떨어지면 알림
    quality.labels(pipeline_name=metrics.pipeline_name).set(metrics.quality_score)

    # 파이프라인 작업은 단기 실행이어서 Prometheus가 스크랩할 때 실행 중이 아닐 수 있으므로
    # 풀(pull) 대신 푸시 게이트웨이를 사용합니다. 푸시 게이트웨이는 Prometheus가
    # 수집할 때까지 메트릭을 보관하는 중간 저장소 역할을 합니다
    push_to_gateway('localhost:9091', job='data_pipeline', registry=registry)
```

### 7.2 알림 설정

```python
# monitoring/alerts.py
import requests
from typing import Optional

class AlertManager:
    """알림 관리"""

    def __init__(self, slack_webhook: str, pagerduty_key: Optional[str] = None):
        self.slack_webhook = slack_webhook
        self.pagerduty_key = pagerduty_key

    def send_slack_alert(self, message: str, severity: str = "info"):
        """Slack 알림"""
        # 색상 코딩 첨부는 Slack에서 즉각적인 시각적 심각도를 제공합니다 —
        # 당직 엔지니어가 전체 메시지를 읽지 않고도 빨간색(오류) vs 주황색(경고)을
        # 한눈에 분류할 수 있습니다
        color = {
            "info": "#36a64f",
            "warning": "#ffa500",
            "error": "#ff0000"
        }.get(severity, "#36a64f")

        payload = {
            "attachments": [{
                "color": color,
                "text": message,
                "footer": "Data Pipeline Alert"
            }]
        }
        requests.post(self.slack_webhook, json=payload)

    def send_pagerduty_alert(self, message: str):
        """PagerDuty 알림 (심각한 경우)"""
        # PagerDuty는 즉각적인 인간 대응이 필요한 심각한 실패(예: 1시간 이상
        # 파이프라인 중단)에만 사용합니다. 중요하지 않은 이슈를 Slack만으로 전달하면
        # 알림 피로를 방지하고 PagerDuty 신호를 의미있게 유지합니다
        if not self.pagerduty_key:
            return

        payload = {
            "routing_key": self.pagerduty_key,
            "event_action": "trigger",
            "payload": {
                "summary": message,
                "severity": "critical",
                "source": "data-pipeline"
            }
        }
        requests.post(
            "https://events.pagerduty.com/v2/enqueue",
            json=payload
        )


# Airflow에서 사용
def alert_on_failure(context):
    """Task 실패 시 알림"""
    alert_manager = AlertManager(
        slack_webhook="https://hooks.slack.com/services/xxx"
    )

    message = f"""
    Pipeline Failed!
    DAG: {context['dag'].dag_id}
    Task: {context['task'].task_id}
    Execution Date: {context['ds']}
    Error: {context.get('exception', 'Unknown')}
    """

    alert_manager.send_slack_alert(message, severity="error")
```

---

## 8. 배포

### 8.1 Docker Compose

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres_data:/var/lib/postgresql/data

  airflow-webserver:
    image: apache/airflow:2.7.0
    depends_on:
      - postgres
    environment:
      AIRFLOW__CORE__EXECUTOR: CeleryExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./dbt:/opt/dbt
    ports:
      - "8080:8080"
    command: webserver

  airflow-scheduler:
    image: apache/airflow:2.7.0
    depends_on:
      - airflow-webserver
    volumes:
      - ./airflow/dags:/opt/airflow/dags
    command: scheduler

  spark-master:
    image: bitnami/spark:3.4
    environment:
      - SPARK_MODE=master
    ports:
      - "7077:7077"
      - "8081:8080"
    volumes:
      - ./spark/jobs:/opt/spark/jobs

  spark-worker:
    image: bitnami/spark:3.4
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER_URL=spark://spark-master:7077
    depends_on:
      - spark-master

volumes:
  postgres_data:
```

---

## 연습 문제

### 문제 1: 파이프라인 확장
실시간 재고 이벤트를 Kafka에서 처리하여 재고 부족 알림을 보내는 스트리밍 파이프라인을 추가하세요.

### 문제 2: 품질 대시보드
일별 데이터 품질 점수를 Grafana 대시보드로 시각화하세요.

### 문제 3: 비용 최적화
대용량 데이터 처리 시 Spark 파티션 수와 리소스 설정을 최적화하세요.

---

## 요약

이 프로젝트에서 다룬 핵심 통합:

| 도구 | 역할 |
|------|------|
| **Airflow** | 파이프라인 오케스트레이션 |
| **Spark** | 대규모 데이터 처리 |
| **dbt** | SQL 기반 변환 |
| **Great Expectations** | 데이터 품질 검증 |
| **Data Lake** | 계층화된 스토리지 |

---

## 참고 자료

- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Spark Performance Tuning](https://spark.apache.org/docs/latest/sql-performance-tuning.html)
- [dbt Best Practices](https://docs.getdbt.com/guides/best-practices)
- [Data Engineering Cookbook](https://github.com/andkret/Cookbook)

---

## 예제 코드

Docker Compose, 합성 데이터, 모든 파이프라인 구성 요소를 포함한 실행 가능한 예제 프로젝트:

**[`examples/Data_Engineering/practical_pipeline/`](../../../examples/Data_Engineering/practical_pipeline/)**

Airflow DAG, Spark 작업, dbt 모델, Great Expectations 스위트 포함 — 모두 로컬 실행에 맞게 조정되었습니다.
