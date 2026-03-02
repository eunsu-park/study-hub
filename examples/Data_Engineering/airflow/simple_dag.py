"""
Airflow Basic DAG Example

This DAG demonstrates a basic Airflow workflow:
- Execute Python functions with PythonOperator
- Execute shell commands with BashOperator
- Define task dependencies

Run: airflow dags test simple_dag 2024-01-01
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator


# Default arguments
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'email': ['data-alerts@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# Python function definitions
def print_hello():
    """Print a greeting message"""
    print("Hello from Airflow!")
    return "hello_returned"


def print_date(**context):
    """Print the execution date"""
    execution_date = context['ds']
    print(f"Execution date: {execution_date}")
    return execution_date


def process_data(value: int, multiplier: int = 2, **context):
    """Data processing example"""
    result = value * multiplier
    print(f"Processing: {value} * {multiplier} = {result}")

    # Store result via XCom
    context['ti'].xcom_push(key='processed_value', value=result)
    return result


def summarize(**context):
    """Summarize results from previous tasks"""
    ti = context['ti']

    # Retrieve values from XCom
    hello_result = ti.xcom_pull(task_ids='hello_task')
    processed_value = ti.xcom_pull(task_ids='process_task', key='processed_value')

    print(f"Summary:")
    print(f"  - Hello result: {hello_result}")
    print(f"  - Processed value: {processed_value}")
    print(f"  - Execution date: {context['ds']}")


# DAG definition
with DAG(
    dag_id='simple_dag',
    default_args=default_args,
    description='Simple Airflow DAG Example',
    schedule_interval='@daily',  # Run daily
    start_date=datetime(2024, 1, 1),
    catchup=False,  # Skip past runs
    tags=['example', 'tutorial'],
) as dag:

    # Task definitions
    start = EmptyOperator(task_id='start')

    hello_task = PythonOperator(
        task_id='hello_task',
        python_callable=print_hello,
    )

    date_task = PythonOperator(
        task_id='date_task',
        python_callable=print_date,
    )

    bash_task = BashOperator(
        task_id='bash_task',
        bash_command='echo "Current time: $(date)" && sleep 2',
    )

    process_task = PythonOperator(
        task_id='process_task',
        python_callable=process_data,
        op_kwargs={'value': 10, 'multiplier': 5},
    )

    summary_task = PythonOperator(
        task_id='summary_task',
        python_callable=summarize,
    )

    end = EmptyOperator(task_id='end')

    # Task dependency definitions
    #     +- hello_task -+
    # start -+             +- process_task - summary_task - end
    #     +- date_task --+
    #             +- bash_task --+

    start >> [hello_task, date_task]
    hello_task >> process_task
    date_task >> [bash_task, process_task]
    [bash_task, process_task] >> summary_task >> end


if __name__ == "__main__":
    # Local testing
    dag.test()
