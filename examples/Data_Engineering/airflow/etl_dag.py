"""
Airflow ETL Pipeline DAG Example

This DAG demonstrates the structure of a real ETL pipeline:
- Extract: Pull data from sources
- Transform: Cleanse and transform data
- Load: Load into destination
- Quality Check: Data quality validation

Run: airflow dags test etl_pipeline 2024-01-01
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
import json
import os


default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


# ============================================
# Extract Functions
# ============================================
def extract_orders(**context):
    """Extract order data"""
    ds = context['ds']
    print(f"Extracting orders for {ds}")

    # Simulation: in practice, this would be a DB query
    orders = [
        {'order_id': 1, 'customer_id': 101, 'amount': 150.00, 'status': 'completed'},
        {'order_id': 2, 'customer_id': 102, 'amount': 250.50, 'status': 'completed'},
        {'order_id': 3, 'customer_id': 101, 'amount': 75.25, 'status': 'pending'},
        {'order_id': 4, 'customer_id': 103, 'amount': 320.00, 'status': 'completed'},
        {'order_id': 5, 'customer_id': 102, 'amount': 99.99, 'status': 'cancelled'},
    ]

    # Pass data via XCom
    context['ti'].xcom_push(key='raw_orders', value=orders)
    print(f"Extracted {len(orders)} orders")
    return len(orders)


def extract_customers(**context):
    """Extract customer data"""
    customers = [
        {'customer_id': 101, 'name': 'Alice', 'segment': 'Gold'},
        {'customer_id': 102, 'name': 'Bob', 'segment': 'Silver'},
        {'customer_id': 103, 'name': 'Charlie', 'segment': 'Bronze'},
    ]

    context['ti'].xcom_push(key='raw_customers', value=customers)
    print(f"Extracted {len(customers)} customers")
    return len(customers)


# ============================================
# Transform Functions
# ============================================
def transform_orders(**context):
    """Transform order data"""
    ti = context['ti']
    orders = ti.xcom_pull(task_ids='extract.extract_orders', key='raw_orders')

    # Transformation logic
    transformed = []
    for order in orders:
        # Include only completed orders
        if order['status'] == 'completed':
            transformed.append({
                'order_id': order['order_id'],
                'customer_id': order['customer_id'],
                'amount': order['amount'],
                'order_date': context['ds'],
            })

    ti.xcom_push(key='transformed_orders', value=transformed)
    print(f"Transformed {len(transformed)} orders (from {len(orders)})")
    return len(transformed)


def enrich_orders(**context):
    """Enrich order data with customer information"""
    ti = context['ti']
    orders = ti.xcom_pull(task_ids='transform.transform_orders', key='transformed_orders')
    customers = ti.xcom_pull(task_ids='extract.extract_customers', key='raw_customers')

    # Customer information mapping
    customer_map = {c['customer_id']: c for c in customers}

    enriched = []
    for order in orders:
        customer = customer_map.get(order['customer_id'], {})
        enriched.append({
            **order,
            'customer_name': customer.get('name', 'Unknown'),
            'customer_segment': customer.get('segment', 'Unknown'),
        })

    ti.xcom_push(key='enriched_orders', value=enriched)
    print(f"Enriched {len(enriched)} orders")
    return enriched


# ============================================
# Load Functions
# ============================================
def load_to_warehouse(**context):
    """Load into data warehouse"""
    ti = context['ti']
    enriched_orders = ti.xcom_pull(task_ids='transform.enrich_orders', key='enriched_orders')

    # Simulation: in practice, this would be a DB INSERT
    print(f"Loading {len(enriched_orders)} records to warehouse")
    for order in enriched_orders:
        print(f"  INSERT: {order}")

    return len(enriched_orders)


# ============================================
# Quality Check Functions
# ============================================
def check_row_count(**context):
    """Validate row count"""
    ti = context['ti']
    enriched_orders = ti.xcom_pull(task_ids='transform.enrich_orders', key='enriched_orders')

    row_count = len(enriched_orders)
    print(f"Row count check: {row_count}")

    if row_count == 0:
        raise ValueError("No data to load!")

    ti.xcom_push(key='row_count', value=row_count)
    return row_count


def check_data_quality(**context):
    """Validate data quality"""
    ti = context['ti']
    enriched_orders = ti.xcom_pull(task_ids='transform.enrich_orders', key='enriched_orders')

    errors = []

    for order in enriched_orders:
        # NULL check
        if order.get('order_id') is None:
            errors.append(f"Missing order_id")

        # Value range check
        if order.get('amount', 0) < 0:
            errors.append(f"Negative amount: {order['order_id']}")

    if errors:
        print(f"Quality issues found: {errors}")
        ti.xcom_push(key='quality_issues', value=errors)
        return 'has_issues'
    else:
        print("Quality check passed")
        return 'no_issues'


def decide_next_step(**context):
    """Branch based on quality results"""
    ti = context['ti']
    quality_result = ti.xcom_pull(task_ids='quality.check_data_quality')

    if quality_result == 'has_issues':
        return 'quality.handle_issues'
    else:
        return 'load'


def handle_quality_issues(**context):
    """Handle quality issues"""
    ti = context['ti']
    issues = ti.xcom_pull(task_ids='quality.check_data_quality', key='quality_issues')
    print(f"Handling quality issues: {issues}")
    # In practice, this would send alerts, log records, etc.


# ============================================
# Notification Function
# ============================================
def send_success_notification(**context):
    """Send success notification"""
    ti = context['ti']
    row_count = ti.xcom_pull(task_ids='quality.check_row_count', key='row_count')

    message = f"""
    ETL Pipeline Completed Successfully!
    Date: {context['ds']}
    Records Loaded: {row_count}
    """
    print(message)
    # In practice, this would send notifications via Slack, Email, etc.


# DAG definition
with DAG(
    dag_id='etl_pipeline',
    default_args=default_args,
    description='ETL Pipeline Example',
    schedule_interval='0 6 * * *',  # Daily at 6 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['etl', 'production'],
) as dag:

    start = EmptyOperator(task_id='start')

    # Extract TaskGroup
    with TaskGroup(group_id='extract') as extract_group:
        extract_orders_task = PythonOperator(
            task_id='extract_orders',
            python_callable=extract_orders,
        )
        extract_customers_task = PythonOperator(
            task_id='extract_customers',
            python_callable=extract_customers,
        )

    # Transform TaskGroup
    with TaskGroup(group_id='transform') as transform_group:
        transform_task = PythonOperator(
            task_id='transform_orders',
            python_callable=transform_orders,
        )
        enrich_task = PythonOperator(
            task_id='enrich_orders',
            python_callable=enrich_orders,
        )
        transform_task >> enrich_task

    # Quality TaskGroup
    with TaskGroup(group_id='quality') as quality_group:
        row_count_task = PythonOperator(
            task_id='check_row_count',
            python_callable=check_row_count,
        )
        quality_task = PythonOperator(
            task_id='check_data_quality',
            python_callable=check_data_quality,
        )
        handle_issues_task = PythonOperator(
            task_id='handle_issues',
            python_callable=handle_quality_issues,
            trigger_rule=TriggerRule.NONE_FAILED,
        )
        [row_count_task, quality_task]

    # Branch
    branch_task = BranchPythonOperator(
        task_id='branch_on_quality',
        python_callable=decide_next_step,
    )

    # Load
    load_task = PythonOperator(
        task_id='load',
        python_callable=load_to_warehouse,
    )

    # Notify
    notify_task = PythonOperator(
        task_id='notify',
        python_callable=send_success_notification,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    end = EmptyOperator(
        task_id='end',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # Task dependencies
    start >> extract_group >> transform_group >> quality_group >> branch_task
    branch_task >> [load_task, handle_issues_task]
    [load_task, handle_issues_task] >> notify_task >> end


if __name__ == "__main__":
    dag.test()
