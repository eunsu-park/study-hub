"""
Exercise Solutions: Lesson 04 - Apache Airflow Basics

Covers:
  - Problem 1: Basic DAG Creation (hourly schedule, two tasks)
  - Problem 2: Conditional Execution (BranchPythonOperator, weekday vs weekend)

Note: These solutions simulate Airflow concepts in pure Python.
      No running Airflow instance is required.
"""

from datetime import datetime, timedelta
import tempfile
import os


# ---------------------------------------------------------------------------
# Simulated Airflow primitives
# ---------------------------------------------------------------------------

class SimulatedTaskInstance:
    """Minimal simulation of an Airflow TaskInstance for XCom."""
    def __init__(self):
        self.xcoms: dict[str, object] = {}

    def xcom_push(self, key: str, value: object) -> None:
        self.xcoms[key] = value

    def xcom_pull(self, task_ids: str, key: str = "return_value") -> object:
        return self.xcoms.get(f"{task_ids}_{key}")


class SimulatedDAG:
    """Simulates an Airflow DAG: stores tasks, executes them in order.

    In real Airflow the scheduler reads DAG files, builds a dependency
    graph, and dispatches tasks to workers.  Here we run tasks in
    sequence within the current process.
    """
    def __init__(self, dag_id: str, schedule: str, start_date: datetime):
        self.dag_id = dag_id
        self.schedule = schedule
        self.start_date = start_date
        self.tasks: list[dict] = []
        self.ti = SimulatedTaskInstance()

    def add_task(self, task_id: str, callable_fn, **kwargs) -> None:
        self.tasks.append({"task_id": task_id, "callable": callable_fn, **kwargs})

    def run(self, execution_date: datetime | None = None) -> None:
        exec_date = execution_date or datetime.now()
        print(f"\n[DAG: {self.dag_id}]  schedule={self.schedule}  exec_date={exec_date}")
        print("-" * 60)
        for task in self.tasks:
            task_id = task["task_id"]
            fn = task["callable"]
            print(f"  Running task: {task_id}")
            result = fn(execution_date=exec_date, ti=self.ti)
            if result is not None:
                self.ti.xcoms[f"{task_id}_return_value"] = result
            print(f"  -> {task_id} completed")
        print("-" * 60)
        print(f"[DAG: {self.dag_id}] All tasks finished.\n")


# ---------------------------------------------------------------------------
# Problem 1: Basic DAG Creation
# Create a DAG that runs hourly with two tasks:
#   1. log_current_time  - logs the current time
#   2. create_temp_file  - creates a temporary file
# ---------------------------------------------------------------------------

def log_current_time(execution_date: datetime, **kwargs) -> str:
    """Task 1: Log the current time.

    In Airflow you would use a PythonOperator:
        PythonOperator(
            task_id='log_current_time',
            python_callable=log_current_time,
        )
    """
    now = datetime.now().isoformat()
    print(f"    [log_current_time] Current time: {now}")
    print(f"    [log_current_time] Execution date: {execution_date}")
    return now


def create_temp_file(execution_date: datetime, **kwargs) -> str:
    """Task 2: Create a temporary file with a timestamp.

    In Airflow you would use a BashOperator or PythonOperator:
        BashOperator(
            task_id='create_temp_file',
            bash_command='echo "{{ ts }}" > /tmp/airflow_{{ ds }}.txt',
        )
    """
    content = f"Execution date: {execution_date}\nCreated at: {datetime.now()}\n"
    fd, path = tempfile.mkstemp(prefix="airflow_exercise_", suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    print(f"    [create_temp_file] Created: {path}")
    # Clean up right away for the exercise
    os.remove(path)
    print(f"    [create_temp_file] Cleaned up temp file")
    return path


def problem1_basic_dag():
    """
    Airflow DAG definition equivalent:

        from airflow import DAG
        from airflow.operators.python import PythonOperator
        from datetime import datetime, timedelta

        default_args = {
            'owner': 'data_team',
            'retries': 1,
            'retry_delay': timedelta(minutes=5),
        }

        with DAG(
            dag_id='hourly_log_and_file',
            default_args=default_args,
            schedule_interval='@hourly',
            start_date=datetime(2024, 1, 1),
            catchup=False,
        ) as dag:
            t1 = PythonOperator(
                task_id='log_current_time',
                python_callable=log_current_time,
            )
            t2 = PythonOperator(
                task_id='create_temp_file',
                python_callable=create_temp_file,
            )
            t1 >> t2
    """
    dag = SimulatedDAG(
        dag_id="hourly_log_and_file",
        schedule="@hourly",
        start_date=datetime(2024, 1, 1),
    )
    dag.add_task("log_current_time", log_current_time)
    dag.add_task("create_temp_file", create_temp_file)
    dag.run()


# ---------------------------------------------------------------------------
# Problem 2: Conditional Execution
# Create a DAG using BranchPythonOperator that executes different tasks
# on weekdays versus weekends.
# ---------------------------------------------------------------------------

def branch_weekday_or_weekend(execution_date: datetime, **kwargs) -> str:
    """Branch function: decides which downstream task to execute.

    In Airflow this would be:
        BranchPythonOperator(
            task_id='branch_check',
            python_callable=branch_weekday_or_weekend,
        )

    Returns the task_id to execute next.
    Monday=0, Sunday=6 in Python's weekday().
    """
    day = execution_date.weekday()
    if day < 5:
        branch = "weekday_task"
        print(f"    [branch] {execution_date.strftime('%A')} is a weekday -> {branch}")
    else:
        branch = "weekend_task"
        print(f"    [branch] {execution_date.strftime('%A')} is a weekend -> {branch}")
    return branch


def weekday_task(execution_date: datetime, **kwargs) -> None:
    """Runs on weekdays: full data processing."""
    print(f"    [weekday_task] Running full ETL pipeline for {execution_date.strftime('%A')}")
    print("    [weekday_task] Steps: extract -> validate -> transform -> load")


def weekend_task(execution_date: datetime, **kwargs) -> None:
    """Runs on weekends: lightweight maintenance."""
    print(f"    [weekend_task] Running maintenance for {execution_date.strftime('%A')}")
    print("    [weekend_task] Steps: cleanup temp files -> compact tables -> send summary")


def problem2_conditional_execution():
    """
    Airflow DAG definition equivalent:

        from airflow import DAG
        from airflow.operators.python import PythonOperator, BranchPythonOperator
        from airflow.operators.empty import EmptyOperator

        with DAG(
            dag_id='weekday_weekend_branch',
            schedule_interval='@daily',
            start_date=datetime(2024, 1, 1),
            catchup=False,
        ) as dag:
            branch = BranchPythonOperator(
                task_id='branch_check',
                python_callable=branch_weekday_or_weekend,
            )
            weekday = PythonOperator(
                task_id='weekday_task',
                python_callable=weekday_task,
            )
            weekend = PythonOperator(
                task_id='weekend_task',
                python_callable=weekend_task,
            )
            join = EmptyOperator(
                task_id='join',
                trigger_rule='none_failed_min_one_success',
            )
            branch >> [weekday, weekend] >> join
    """
    # Simulate for a weekday
    print("--- Weekday Execution ---")
    dag_wd = SimulatedDAG("weekday_weekend_branch", "@daily", datetime(2024, 1, 1))
    exec_date_wd = datetime(2024, 11, 18)  # Monday
    # Simulate branching
    branch_result = branch_weekday_or_weekend(execution_date=exec_date_wd)
    if branch_result == "weekday_task":
        dag_wd.add_task("weekday_task", weekday_task)
    else:
        dag_wd.add_task("weekend_task", weekend_task)
    dag_wd.run(execution_date=exec_date_wd)

    # Simulate for a weekend
    print("--- Weekend Execution ---")
    dag_we = SimulatedDAG("weekday_weekend_branch", "@daily", datetime(2024, 1, 1))
    exec_date_we = datetime(2024, 11, 16)  # Saturday
    branch_result = branch_weekday_or_weekend(execution_date=exec_date_we)
    if branch_result == "weekday_task":
        dag_we.add_task("weekday_task", weekday_task)
    else:
        dag_we.add_task("weekend_task", weekend_task)
    dag_we.run(execution_date=exec_date_we)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Problem 1: Basic DAG Creation (Hourly Schedule)")
    print("=" * 70)
    problem1_basic_dag()

    print("=" * 70)
    print("Problem 2: Conditional Execution (Weekday vs Weekend)")
    print("=" * 70)
    problem2_conditional_execution()
