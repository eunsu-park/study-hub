# Apache Airflow Basics

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the Apache Airflow architecture and describe the roles of each core component (Web Server, Scheduler, Executor, Worker, Metadata DB)
2. Define a DAG (Directed Acyclic Graph) in Python and configure task dependencies using Airflow operators
3. Implement common Airflow operators including PythonOperator, BashOperator, and PostgresOperator
4. Use XComs and Airflow Variables to share data and configuration between tasks
5. Apply scheduling with cron expressions and configure backfilling and catchup behavior
6. Debug and monitor DAG runs using the Airflow Web UI and logs

---

## Overview

Apache Airflow is a platform for programmatically authoring, scheduling, and monitoring workflows. It manages complex data pipelines by defining DAGs (Directed Acyclic Graphs) in Python.

---

## 1. Airflow Architecture

Before diving into components, it helps to understand the problem Airflow solves. A plain cron job can schedule a single script, but it has no built-in support for complex task dependencies, automatic retries on failure, backfilling historical date ranges, or a centralized UI for observability. Airflow addresses all of these: it models pipelines as DAGs with explicit dependencies, provides configurable retry/alerting policies, supports backfill with a single CLI command, and ships with a web UI that shows task status, logs, and execution history in one place.

### 1.1 Core Components

```
┌──────────────────────────────────────────────────────────────┐
│                    Airflow Architecture                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐         ┌─────────────┐                   │
│   │  Web Server │         │  Scheduler  │                   │
│   │    (UI)     │         │             │                   │
│   └──────┬──────┘         └──────┬──────┘                   │
│          │                       │                          │
│          │    ┌─────────────┐    │                          │
│          └───→│  Metadata   │←───┘                          │
│               │  Database   │                               │
│               │ (PostgreSQL)│                               │
│               └──────┬──────┘                               │
│                      │                                      │
│          ┌───────────┴───────────┐                          │
│          ↓                       ↓                          │
│   ┌─────────────┐         ┌─────────────┐                   │
│   │   Worker    │         │   Worker    │                   │
│   │  (Celery)   │         │  (Celery)   │                   │
│   └─────────────┘         └─────────────┘                   │
│                                                              │
│   DAGs Folder: /opt/airflow/dags/                           │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 Component Roles

| Component | Role |
|-----------|------|
| **Web Server** | Provide UI, visualize DAGs, view logs |
| **Scheduler** | Parse DAGs, schedule tasks, trigger execution |
| **Executor** | Determine task execution method (Local, Celery, K8s) |
| **Worker** | Execute actual tasks (Celery/K8s Executor) |
| **Metadata DB** | Store DAG metadata and execution history |

### 1.3 Executor Types

```python
# airflow.cfg settings
# The executor determines how many tasks can run in parallel and whether
# they run on the same machine or across a cluster.  Choosing the wrong
# executor is the #1 cause of "my DAGs are slow" complaints.
executor_types = {
    "SequentialExecutor": "Single process, for development",
    "LocalExecutor": "Multi-process, single machine",
    "CeleryExecutor": "Distributed processing, production",
    "KubernetesExecutor": "Run as K8s Pods"
}

# Recommended configuration:
# Development → LocalExecutor (no external broker needed, still parallel)
# Production  → CeleryExecutor (persistent workers, lower cold-start)
#            or KubernetesExecutor (per-task isolation, auto-scaling to zero)
```

---

## 2. Installation and Environment Setup

### 2.1 Docker Compose Installation (Recommended)

```yaml
# docker-compose.yaml
version: '3.8'

# YAML anchor (&airflow-common) avoids duplicating config across services —
# all Airflow components share the same image, env vars, and volume mounts.
x-airflow-common: &airflow-common
  image: apache/airflow:2.7.0
  environment:
    &airflow-common-env
    AIRFLOW__CORE__EXECUTOR: CeleryExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://airflow:airflow@postgres/airflow
    # Redis as Celery broker: lightweight, no authentication for local dev.
    # In production, use a managed Redis or RabbitMQ with TLS.
    AIRFLOW__CELERY__BROKER_URL: redis://:@redis:6379/0
    AIRFLOW__CORE__FERNET_KEY: ''
    # Pause new DAGs by default so they don't start running immediately on
    # deploy — gives operators time to review before enabling.
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'true'
    AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
  volumes:
    # Mount local directories so DAG code changes are picked up without
    # rebuilding the Docker image — essential for a fast dev loop.
    - ./dags:/opt/airflow/dags
    - ./logs:/opt/airflow/logs
    - ./plugins:/opt/airflow/plugins

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres-db-volume:/var/lib/postgresql/data

  redis:
    image: redis:latest

  airflow-webserver:
    <<: *airflow-common
    command: webserver
    ports:
      - 8080:8080
    depends_on:
      - postgres
      - redis

  airflow-scheduler:
    <<: *airflow-common
    command: scheduler
    depends_on:
      - postgres
      - redis

  airflow-worker:
    <<: *airflow-common
    command: celery worker
    depends_on:
      - airflow-scheduler

  airflow-init:
    <<: *airflow-common
    entrypoint: /bin/bash
    command:
      - -c
      - |
        airflow db init
        airflow users create \
          --username admin \
          --password admin \
          --firstname Admin \
          --lastname User \
          --role Admin \
          --email admin@example.com

volumes:
  postgres-db-volume:
```

### 2.2 pip Installation (Local Development)

```bash
# Create virtual environment
python -m venv airflow-venv
source airflow-venv/bin/activate

# Install Airflow
pip install "apache-airflow[celery,postgres,redis]==2.7.0" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.0/constraints-3.9.txt"

# Initialize
export AIRFLOW_HOME=~/airflow
airflow db init

# Create user
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Start services
airflow webserver --port 8080 &
airflow scheduler &
```

---

## 3. DAG (Directed Acyclic Graph)

### 3.1 Basic DAG Structure

```python
# dags/simple_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# default_args are inherited by every task in the DAG, reducing boilerplate.
# Override per-task when a specific operator needs different retry behavior.
default_args = {
    'owner': 'data_team',
    # depends_on_past=False: each run is independent. Set to True only when
    # a task genuinely needs the previous day's run to have succeeded first
    # (e.g., incremental aggregation that reads yesterday's output).
    'depends_on_past': False,
    'email': ['data-team@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    # 3 retries with 5-min delay: gives transient issues (network blips,
    # temporary DB locks) time to resolve without human intervention.
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
with DAG(
    dag_id='simple_example_dag',
    default_args=default_args,
    description='Simple example DAG',
    schedule_interval='0 9 * * *',  # Daily at 9 AM
    start_date=datetime(2024, 1, 1),
    # Prevent backfill flooding: without catchup=False, Airflow schedules
    # ALL missed runs since start_date on first deployment — if start_date
    # is 2024-01-01 and today is 2024-06-15, that's ~165 concurrent runs.
    catchup=False,
    tags=['example', 'tutorial'],
) as dag:

    # Task 1: Execute Python function
    def print_hello():
        print("Hello, Airflow!")
        return "Hello returned"

    task_hello = PythonOperator(
        task_id='print_hello',
        python_callable=print_hello,
    )

    # Task 2: Execute Bash command
    task_date = BashOperator(
        task_id='print_date',
        bash_command='date',
    )

    # Task 3: Python function (with arguments)
    def greet(name, **kwargs):
        execution_date = kwargs['ds']
        print(f"Hello, {name}! Today is {execution_date}")

    task_greet = PythonOperator(
        task_id='greet_user',
        python_callable=greet,
        op_kwargs={'name': 'Data Engineer'},
    )

    # Define task dependencies
    task_hello >> task_date >> task_greet
    # Or: task_hello.set_downstream(task_date)
```

### 3.2 DAG Parameters

```python
from airflow import DAG

dag = DAG(
    # Required parameters
    dag_id='my_dag',                    # Unique identifier (must be unique across all DAGs)
    start_date=datetime(2024, 1, 1),    # Earliest data_interval the scheduler will create

    # Schedule related
    schedule_interval='@daily',         # Execution frequency
    # schedule_interval='0 0 * * *'     # Cron expression (more precise control)
    # schedule_interval=timedelta(days=1)  # timedelta for non-calendar intervals

    # Execution control
    catchup=False,                      # See note above about backfill flooding
    # max_active_runs=1: prevents overlapping runs for non-idempotent pipelines.
    # Increase for idempotent DAGs that can safely run in parallel.
    max_active_runs=1,
    # max_active_tasks limits parallelism *within* a single run — useful to
    # avoid overwhelming a shared resource (e.g., a database connection pool).
    max_active_tasks=10,

    # Other
    default_args=default_args,          # Default arguments
    description='DAG description',
    tags=['production', 'etl'],         # Tags enable filtering in the web UI
    doc_md="""
    ## DAG Documentation
    This DAG performs daily ETL.
    """
)

# Schedule presets
schedule_presets = {
    '@once': 'Run once',
    '@hourly': 'Every hour (0 * * * *)',
    '@daily': 'Daily at midnight (0 0 * * *)',
    '@weekly': 'Every Sunday (0 0 * * 0)',
    '@monthly': 'First of month (0 0 1 * *)',
    '@yearly': 'January 1st (0 0 1 1 *)',
    None: 'Manual trigger only'
}
```

---

## 4. Operator Types

### 4.1 Main Operators

```python
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.email import EmailOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.http.operators.http import SimpleHttpOperator

# 1. PythonOperator — best when you need complex logic, library imports, or
# DataFrame manipulation.  Use this over BashOperator when the task involves
# more than a one-liner shell command.
def my_function(arg1, arg2):
    return arg1 + arg2

python_task = PythonOperator(
    task_id='python_task',
    python_callable=my_function,
    op_args=[1, 2],              # Positional arguments
    op_kwargs={'arg1': 1},       # Keyword arguments
)


# 2. BashOperator — ideal for calling CLI tools (dbt run, spark-submit),
# running shell scripts, or quick file operations.  Prefer this over
# PythonOperator when the task is essentially a shell command.
bash_task = BashOperator(
    task_id='bash_task',
    bash_command='echo "Hello" && date',
    env={'MY_VAR': 'value'},     # Environment variables
    cwd='/tmp',                  # Working directory
)


# 3. EmptyOperator — zero-cost DAG structure nodes.  Use as start/end
# markers or to fan-in/fan-out parallel branches without running any logic.
start = EmptyOperator(task_id='start')
end = EmptyOperator(task_id='end')


# 4. PostgresOperator — executes SQL directly against a managed connection.
# Prefer this over PythonOperator + psycopg2 for simple SQL statements
# because it handles connection lifecycle and templating automatically.
sql_task = PostgresOperator(
    task_id='sql_task',
    postgres_conn_id='my_postgres',
    sql="""
        INSERT INTO logs (message, created_at)
        VALUES ('Task executed', NOW());
    """,
)


# 5. EmailOperator — sends notification emails via the configured SMTP
# connection.  Use for success summaries or reports; for failure alerts,
# prefer email_on_failure in default_args (fires automatically).
email_task = EmailOperator(
    task_id='send_email',
    to='user@example.com',
    subject='Airflow Notification',
    html_content='<h1>Task completed!</h1>',
)


# 6. SimpleHttpOperator — calls external REST APIs.  The response_check
# lambda lets you define custom success criteria beyond HTTP 2xx status.
http_task = SimpleHttpOperator(
    task_id='http_task',
    http_conn_id='my_api',
    endpoint='/api/data',
    method='GET',
    response_check=lambda response: response.status_code == 200,
)
```

### 4.2 Branch Operator

```python
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator

def choose_branch(**kwargs):
    """Choose task to execute based on condition"""
    execution_date = kwargs['ds']
    day_of_week = datetime.strptime(execution_date, '%Y-%m-%d').weekday()

    if day_of_week < 5:  # Weekday
        return 'weekday_task'
    else:  # Weekend
        return 'weekend_task'

with DAG('branch_example', ...) as dag:

    branch_task = BranchPythonOperator(
        task_id='branch',
        python_callable=choose_branch,
    )

    weekday_task = EmptyOperator(task_id='weekday_task')
    weekend_task = EmptyOperator(task_id='weekend_task')
    # trigger_rule='none_failed_min_one_success': the join task runs as long
    # as at least one branch succeeded and none failed.  Default 'all_success'
    # would never trigger because the un-chosen branch is always "skipped".
    join_task = EmptyOperator(task_id='join', trigger_rule='none_failed_min_one_success')

    branch_task >> [weekday_task, weekend_task] >> join_task
```

### 4.3 Custom Operator

```python
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from typing import Any

class MyCustomOperator(BaseOperator):
    """Custom operator example"""

    # template_fields: Airflow renders Jinja templates in these fields before
    # execute() runs, enabling dynamic values like {{ ds }} or {{ params.x }}.
    # Any field NOT listed here will be treated as a literal string.
    template_fields = ['param']

    @apply_defaults
    def __init__(
        self,
        param: str,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.param = param

    def execute(self, context: dict) -> Any:
        """Task execution logic"""
        self.log.info(f"Executing with param: {self.param}")

        # context dict provides runtime metadata (dates, task instance,
        # DAG run info) — avoids hardcoding values that change per run.
        execution_date = context['ds']
        task_instance = context['ti']

        # Business logic
        result = f"Processed {self.param} on {execution_date}"

        # Returning a value automatically pushes it to XCom with
        # key='return_value', making it available to downstream tasks.
        return result


# Usage
custom_task = MyCustomOperator(
    task_id='custom_task',
    param='my_value',
)
```

---

## 5. Task Dependencies

### 5.1 Dependency Definition Methods

```python
from airflow import DAG
from airflow.operators.empty import EmptyOperator

with DAG('dependency_example', ...) as dag:

    task_a = EmptyOperator(task_id='task_a')
    task_b = EmptyOperator(task_id='task_b')
    task_c = EmptyOperator(task_id='task_c')
    task_d = EmptyOperator(task_id='task_d')
    task_e = EmptyOperator(task_id='task_e')

    # Method 1: >> operator (recommended)
    task_a >> task_b >> task_c

    # Method 2: << operator (reverse)
    task_c << task_b << task_a  # Same as above

    # Method 3: set_downstream / set_upstream
    task_a.set_downstream(task_b)
    task_b.set_downstream(task_c)

    # Parallel execution
    task_a >> [task_b, task_c] >> task_d

    # Complex dependencies
    #     ┌→ B ─┐
    # A ──┤     ├──→ E
    #     └→ C → D ─┘

    task_a >> task_b >> task_e
    task_a >> task_c >> task_d >> task_e
```

### 5.2 Trigger Rules

```python
from airflow.utils.trigger_rule import TriggerRule

# Trigger rule types
trigger_rules = {
    'all_success': 'All upstream tasks succeeded (default)',
    'all_failed': 'All upstream tasks failed',
    'all_done': 'All upstream tasks completed (success/failure irrelevant)',
    'one_success': 'At least one succeeded',
    'one_failed': 'At least one failed',
    'none_failed': 'No failures (skips allowed)',
    'none_failed_min_one_success': 'No failures and at least one success',
    'none_skipped': 'No skips',
    'always': 'Always run',
}

# Usage example: join after a branch — runs even if one branch was skipped,
# as long as none actually *failed*.
task_join = EmptyOperator(
    task_id='join',
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
)

# Error handling task: only runs when at least one upstream failed —
# useful for cleanup or alert tasks that should not run on success.
task_error_handler = EmptyOperator(
    task_id='error_handler',
    trigger_rule=TriggerRule.ONE_FAILED,
)
```

---

## 6. Scheduling

### 6.1 Cron Expressions

```python
# Cron format: minute hour day month day_of_week
cron_examples = {
    '0 0 * * *': 'Daily at midnight',
    '0 9 * * 1-5': 'Weekdays at 9 AM',
    '0 */2 * * *': 'Every 2 hours',
    '30 8 1 * *': 'First of month at 8:30 AM',
    '0 0 * * 0': 'Every Sunday at midnight',
}

# Use in DAG
dag = DAG(
    dag_id='scheduled_dag',
    schedule_interval='0 9 * * 1-5',  # Weekdays at 9 AM
    start_date=datetime(2024, 1, 1),
    ...
)
```

### 6.2 Data Interval

```python
# Airflow 2.0+ data interval concept
# Understanding this is critical: the DAG runs AFTER the interval ends,
# not at the start. This "end-of-period" convention ensures the full
# day's data exists before the pipeline processes it.
"""
schedule_interval = @daily, start_date = 2024-01-01

Execution time: 2024-01-02 00:00
data_interval_start: 2024-01-01 00:00
data_interval_end: 2024-01-02 00:00
logical_date (execution_date): 2024-01-01 00:00

→ Runs on 2024-01-02 to process 2024-01-01 data
"""

def process_daily_data(**kwargs):
    # Data period to process
    data_interval_start = kwargs['data_interval_start']
    data_interval_end = kwargs['data_interval_end']

    print(f"Processing data from {data_interval_start} to {data_interval_end}")

# Using Jinja templates
sql_task = PostgresOperator(
    task_id='load_data',
    sql="""
        SELECT * FROM sales
        WHERE sale_date >= '{{ data_interval_start }}'
          AND sale_date < '{{ data_interval_end }}'
    """,
)
```

---

## 7. Basic DAG Writing Example

### 7.1 Daily ETL DAG

```python
# dags/daily_etl_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    'owner': 'data_team',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': ['data-alerts@company.com'],
}

def extract_data(**kwargs):
    """Extract data"""
    import pandas as pd

    # kwargs['ds'] is the logical date (YYYY-MM-DD) — Airflow injects this
    # automatically, so the same DAG code works for any date during backfills.
    ds = kwargs['ds']

    # Filter by ds to ensure idempotent extraction: re-running this task
    # for the same date always pulls the same data partition.
    query = f"""
        SELECT * FROM source_table
        WHERE date = '{ds}'
    """

    # Parquet preserves column types across the E→T boundary;
    # CSV would lose datetime/decimal precision.
    # df = pd.read_sql(query, source_conn)
    # df.to_parquet(f'/tmp/extract_{ds}.parquet')

    print(f"Extracted data for {ds}")
    return f"/tmp/extract_{ds}.parquet"


def transform_data(**kwargs):
    """Transform data"""
    import pandas as pd

    ti = kwargs['ti']
    extract_path = ti.xcom_pull(task_ids='extract')

    # df = pd.read_parquet(extract_path)
    # Transformation logic
    # df['new_column'] = df['column'].apply(transform_func)
    # df.to_parquet(f'/tmp/transform_{kwargs["ds"]}.parquet')

    print("Data transformed")
    return f"/tmp/transform_{kwargs['ds']}.parquet"


with DAG(
    dag_id='daily_etl_pipeline',
    default_args=default_args,
    description='Daily ETL pipeline',
    schedule_interval='0 6 * * *',  # Daily at 6 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['etl', 'daily', 'production'],
) as dag:

    start = EmptyOperator(task_id='start')

    extract = PythonOperator(
        task_id='extract',
        python_callable=extract_data,
    )

    transform = PythonOperator(
        task_id='transform',
        python_callable=transform_data,
    )

    load = PostgresOperator(
        task_id='load',
        postgres_conn_id='warehouse',
        sql="""
            COPY target_table FROM '/tmp/transform_{{ ds }}.parquet'
            WITH (FORMAT 'parquet');
        """,
    )

    # Post-load validation: fail loudly if no rows were loaded.
    # The 1/0 trick causes a division-by-zero error that Airflow interprets
    # as a task failure, triggering the configured retry and alert policies.
    validate = PostgresOperator(
        task_id='validate',
        postgres_conn_id='warehouse',
        sql="""
            SELECT
                CASE WHEN COUNT(*) > 0 THEN 1
                     ELSE 1/0  -- Intentional error to fail the task
                END
            FROM target_table
            WHERE date = '{{ ds }}';
        """,
    )

    end = EmptyOperator(task_id='end')

    # Define dependencies
    start >> extract >> transform >> load >> validate >> end
```

---

## Practice Problems

### Problem 1: Basic DAG Creation
Create a DAG that runs hourly. It should include two tasks: one that logs the current time and another that creates a temporary file.

### Problem 2: Conditional Execution
Create a DAG using BranchPythonOperator that executes different tasks on weekdays versus weekends.

---

## Summary

| Concept | Description |
|------|------|
| **DAG** | Directed Acyclic Graph defining task dependencies |
| **Operator** | Task execution type (Python, Bash, SQL, etc.) |
| **Task** | Individual work unit within a DAG |
| **Scheduler** | DAG parsing and task scheduling |
| **Executor** | Task execution method (Local, Celery, K8s) |

---

## References

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Astronomer Guides](https://www.astronomer.io/guides/)
