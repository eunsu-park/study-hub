# 실전 파이프라인 프로젝트

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
    with TaskGroup(group_id='transform_dbt') as transform_dbt:

        def run_dbt_command(command: str, **kwargs):
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

        dbt_test = PythonOperator(
            task_id='dbt_test',
            python_callable=run_dbt_command,
            op_kwargs={'command': 'test'},
        )

        dbt_run >> dbt_test

    # ============================================
    # Quality Check: Gold 레이어 품질 검증
    # ============================================
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

    # Task 의존성
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
    jdbc_url = "jdbc:postgresql://postgres:5432/ecommerce"
    properties = {
        "user": "postgres",
        "password": "password",
        "driver": "org.postgresql.Driver"
    }

    # 증분 추출 (날짜 지정 시)
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

    # Bronze 레이어에 저장
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

    spark = SparkSession.builder \
        .appName("Process Clickstream") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

    # JSON 스키마 정의
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

    # 정제 및 변환
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

    # 파티션으로 저장
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

    # Silver 레이어 읽기
    orders = spark.read.parquet(f"s3://data-lake/silver/orders/{args.date}/")
    customers = spark.read.parquet("s3://data-lake/silver/customers/")
    products = spark.read.parquet("s3://data-lake/silver/products/")

    # 일별 매출 집계
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
            countDistinct("customer_id").alias("unique_customers")
        )

    # 고객 세그먼트별 집계
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
        CAST(order_date AS DATE) AS order_date,
        CAST(amount AS DECIMAL(12, 2)) AS amount,
        CAST(quantity AS INT) AS quantity,
        status,
        CURRENT_TIMESTAMP AS loaded_at
    FROM source
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
        materialized='incremental',
        unique_key='order_id',
        schema='marts',
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

    -- 고객 정보
    o.customer_id,
    c.customer_name,
    c.customer_segment,
    c.region,

    -- 상품 정보
    o.product_id,
    p.product_name,
    p.category,

    -- 측정값
    o.quantity,
    o.amount AS order_amount,
    p.unit_cost * o.quantity AS cost_amount,
    o.amount - (p.unit_cost * o.quantity) AS profit_amount,

    -- 상태
    o.status,

    -- 메타데이터
    o.loaded_at

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

    -- 수익률
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

    registry = CollectorRegistry()

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
    quality.labels(pipeline_name=metrics.pipeline_name).set(metrics.quality_score)

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
