"""
Daily E-Commerce ETL DAG

Orchestrates: Extract (Spark) → Quality Check (GE) → Transform (dbt) → Notify

Adapted from Data_Engineering Lesson 14 for local execution.
Cloud references (S3, Snowflake) replaced with local filesystem + PostgreSQL.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

default_args = {
    "owner": "data-team",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

DATA_LAKE = "/opt/data-lake"
SPARK_JOBS = "/opt/spark/jobs"
DBT_DIR = "/opt/dbt"


with DAG(
    dag_id="daily_ecommerce_pipeline",
    default_args=default_args,
    description="Daily e-commerce data pipeline (local)",
    schedule="0 6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["etl", "daily", "ecommerce"],
    max_active_runs=1,
) as dag:

    start = EmptyOperator(task_id="start")

    # ── Extract: Spark jobs pull from source DB ──────────────────────
    with TaskGroup(group_id="extract") as extract_group:

        extract_orders = BashOperator(
            task_id="extract_orders",
            bash_command=(
                f"spark-submit {SPARK_JOBS}/extract_postgres.py "
                f"--table orders --date {{{{ ds }}}} "
                f"--output {DATA_LAKE}/bronze/orders/{{{{ ds }}}}/"
            ),
        )

        extract_customers = BashOperator(
            task_id="extract_customers",
            bash_command=(
                f"spark-submit {SPARK_JOBS}/extract_postgres.py "
                f"--table customers "
                f"--output {DATA_LAKE}/bronze/customers/"
            ),
        )

        extract_products = BashOperator(
            task_id="extract_products",
            bash_command=(
                f"spark-submit {SPARK_JOBS}/extract_postgres.py "
                f"--table products "
                f"--output {DATA_LAKE}/bronze/products/"
            ),
        )

    # ── Quality Check: Bronze layer ──────────────────────────────────
    with TaskGroup(group_id="quality_bronze") as quality_bronze:

        def run_great_expectations(suite_name: str, **kwargs):
            import great_expectations as gx

            context = gx.get_context(
                context_root_dir="/opt/great_expectations"
            )
            result = context.run_checkpoint(
                checkpoint_name=f"bronze_{suite_name}_checkpoint"
            )
            if not result.success:
                raise ValueError(f"Quality check failed: {suite_name}")

        check_orders = PythonOperator(
            task_id="check_orders_quality",
            python_callable=run_great_expectations,
            op_kwargs={"suite_name": "orders"},
        )

    # ── Transform: Spark aggregation → Silver layer ──────────────────
    with TaskGroup(group_id="transform_spark") as transform_spark:

        aggregate_daily = BashOperator(
            task_id="aggregate_daily",
            bash_command=(
                f"spark-submit {SPARK_JOBS}/aggregate_daily.py "
                f"--date {{{{ ds }}}} "
                f"--output {DATA_LAKE}/silver/daily_aggregates/{{{{ ds }}}}/"
            ),
        )

    # ── Transform: dbt → Gold layer ──────────────────────────────────
    with TaskGroup(group_id="transform_dbt") as transform_dbt:

        dbt_run = BashOperator(
            task_id="dbt_run",
            bash_command=f"cd {DBT_DIR} && dbt run --profiles-dir {DBT_DIR}",
        )

        dbt_test = BashOperator(
            task_id="dbt_test",
            bash_command=f"cd {DBT_DIR} && dbt test --profiles-dir {DBT_DIR}",
        )

        dbt_run >> dbt_test

    # ── Notify ───────────────────────────────────────────────────────
    def send_notification(**kwargs):
        print(f"Pipeline completed for {kwargs['ds']}")

    notify = PythonOperator(
        task_id="notify_completion",
        python_callable=send_notification,
    )

    end = EmptyOperator(task_id="end")

    # Task dependencies
    start >> extract_group >> quality_bronze >> transform_spark >> transform_dbt >> notify >> end
