"""
Standalone Data Quality DAG

Runs Great Expectations validation on all data layers independently.
Can be triggered manually or on a separate schedule.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def validate_layer(layer: str, suite: str, **kwargs):
    """Run Great Expectations checkpoint for a data layer."""
    import great_expectations as gx

    context = gx.get_context(context_root_dir="/opt/great_expectations")
    checkpoint_name = f"{layer}_{suite}_checkpoint"

    result = context.run_checkpoint(checkpoint_name=checkpoint_name)

    stats = {
        "success": result.success,
        "evaluated_expectations": result.statistics.get("evaluated_expectations", 0),
        "successful_expectations": result.statistics.get("successful_expectations", 0),
    }
    print(f"[{checkpoint_name}] {stats}")

    if not result.success:
        raise ValueError(f"Quality check failed: {checkpoint_name}")


with DAG(
    dag_id="data_quality_validation",
    default_args={
        "owner": "data-team",
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
    },
    description="Standalone data quality validation",
    schedule="30 7 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["quality", "validation"],
) as dag:

    check_bronze_orders = PythonOperator(
        task_id="check_bronze_orders",
        python_callable=validate_layer,
        op_kwargs={"layer": "bronze", "suite": "orders"},
    )

    check_bronze_customers = PythonOperator(
        task_id="check_bronze_customers",
        python_callable=validate_layer,
        op_kwargs={"layer": "bronze", "suite": "customers"},
    )

    check_gold_facts = PythonOperator(
        task_id="check_gold_facts",
        python_callable=validate_layer,
        op_kwargs={"layer": "gold", "suite": "fct_orders"},
    )

    [check_bronze_orders, check_bronze_customers] >> check_gold_facts
