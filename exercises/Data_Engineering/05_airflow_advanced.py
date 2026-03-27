"""
Exercise Solutions: Lesson 05 - Airflow Advanced

Covers:
  - Problem 1: Using XCom (two tasks return numbers, third sums them)
  - Problem 2: Dynamic DAG (ETL tasks for each table in a list)
  - Problem 3: Using Sensors (wait for file, then process)

Note: Pure Python simulation of Airflow concepts.
"""

from datetime import datetime
import os
import tempfile
import time


# ---------------------------------------------------------------------------
# Simulated Airflow primitives
# ---------------------------------------------------------------------------

class XComStore:
    """Simulates the Airflow XCom backend (metadata DB in real Airflow)."""
    def __init__(self):
        self._store: dict[str, object] = {}

    def push(self, task_id: str, key: str, value: object) -> None:
        self._store[f"{task_id}.{key}"] = value

    def pull(self, task_id: str, key: str = "return_value") -> object:
        return self._store.get(f"{task_id}.{key}")


# ---------------------------------------------------------------------------
# Problem 1: Using XCom
# Two tasks each return a number; a third task sums them via XCom.
# ---------------------------------------------------------------------------

def generate_number_a(xcom: XComStore) -> int:
    """Task that produces the first number.

    In Airflow, returning a value from a PythonOperator automatically
    pushes it to XCom with key='return_value'.
    """
    value = 42
    xcom.push("generate_number_a", "return_value", value)
    print(f"  [generate_number_a] Pushed {value} to XCom")
    return value


def generate_number_b(xcom: XComStore) -> int:
    """Task that produces the second number."""
    value = 58
    xcom.push("generate_number_b", "return_value", value)
    print(f"  [generate_number_b] Pushed {value} to XCom")
    return value


def sum_numbers(xcom: XComStore) -> int:
    """Task that pulls both numbers from XCom and computes the sum.

    Airflow equivalent:
        def sum_numbers(**context):
            ti = context['ti']
            a = ti.xcom_pull(task_ids='generate_number_a')
            b = ti.xcom_pull(task_ids='generate_number_b')
            total = a + b
            print(f'Sum: {a} + {b} = {total}')
            return total
    """
    a = xcom.pull("generate_number_a")
    b = xcom.pull("generate_number_b")
    total = a + b
    print(f"  [sum_numbers] Pulled a={a}, b={b} -> sum={total}")
    return total


def problem1_xcom():
    """
    Airflow DAG definition equivalent:

        with DAG('xcom_sum', schedule='@daily', start_date=...) as dag:
            t_a = PythonOperator(task_id='generate_number_a', python_callable=...)
            t_b = PythonOperator(task_id='generate_number_b', python_callable=...)
            t_sum = PythonOperator(task_id='sum_numbers', python_callable=...)
            [t_a, t_b] >> t_sum
    """
    xcom = XComStore()
    print("\n  Task 1a: generate_number_a")
    generate_number_a(xcom)
    print("  Task 1b: generate_number_b")
    generate_number_b(xcom)
    print("  Task 2:  sum_numbers")
    result = sum_numbers(xcom)
    print(f"\n  Final result: {result}")


# ---------------------------------------------------------------------------
# Problem 2: Dynamic DAG
# Dynamically generate ETL tasks for each table in a list.
# ---------------------------------------------------------------------------

def create_etl_task(table_name: str):
    """Factory function that creates an ETL callable for one table.

    In Airflow you would loop over the table list and create operators
    dynamically inside the DAG file:

        TABLE_LIST = ['users', 'orders', 'products']
        for table in TABLE_LIST:
            extract = PythonOperator(
                task_id=f'extract_{table}',
                python_callable=extract_fn,
                op_kwargs={'table': table},
            )
            transform = PythonOperator(
                task_id=f'transform_{table}',
                python_callable=transform_fn,
                op_kwargs={'table': table},
            )
            load = PythonOperator(
                task_id=f'load_{table}',
                python_callable=load_fn,
                op_kwargs={'table': table},
            )
            extract >> transform >> load
    """
    def extract():
        rows = {"users": 1500, "orders": 8300, "products": 420}
        count = rows.get(table_name, 100)
        print(f"    [extract_{table_name}] Extracted {count} rows from source DB")
        return count

    def transform(row_count: int):
        # Simulate cleaning: drop 5% bad rows
        clean_count = int(row_count * 0.95)
        print(f"    [transform_{table_name}] Cleaned {row_count} -> {clean_count} rows")
        return clean_count

    def load(row_count: int):
        print(f"    [load_{table_name}] Loaded {row_count} rows into warehouse")

    return extract, transform, load


def problem2_dynamic_dag():
    """
    Airflow DAG definition equivalent:

        TABLE_LIST = ['users', 'orders', 'products']

        with DAG('dynamic_etl', schedule='@daily', start_date=...) as dag:
            for table in TABLE_LIST:
                ext = PythonOperator(task_id=f'extract_{table}', ...)
                trn = PythonOperator(task_id=f'transform_{table}', ...)
                lod = PythonOperator(task_id=f'load_{table}', ...)
                ext >> trn >> lod

    Adding a new table requires only appending to TABLE_LIST.
    """
    table_list = ["users", "orders", "products"]

    print(f"\n  Dynamically generating ETL tasks for: {table_list}")
    print(f"  (Each table gets its own extract -> transform -> load chain)\n")

    for table in table_list:
        extract_fn, transform_fn, load_fn = create_etl_task(table)
        print(f"  --- ETL: {table} ---")
        row_count = extract_fn()
        clean_count = transform_fn(row_count)
        load_fn(clean_count)
        print()


# ---------------------------------------------------------------------------
# Problem 3: Using Sensors
# Wait for a file to be created, then process it.
# ---------------------------------------------------------------------------

class FileSensor:
    """Simulates Airflow's FileSensor.

    In Airflow:
        FileSensor(
            task_id='wait_for_file',
            filepath='/data/input/daily_data.csv',
            poke_interval=30,
            timeout=3600,
            mode='poke',  # or 'reschedule' to free up worker slot
        )

    Poke mode: The sensor occupies a worker slot and polls at intervals.
    Reschedule mode: The sensor releases the slot between checks (preferred
    for long waits to avoid blocking the worker pool).
    """
    def __init__(self, filepath: str, poke_interval: float = 0.5,
                 timeout: float = 10.0):
        self.filepath = filepath
        self.poke_interval = poke_interval
        self.timeout = timeout

    def poke(self) -> bool:
        """Check once whether the file exists."""
        return os.path.exists(self.filepath)

    def execute(self) -> bool:
        """Block until the file appears or timeout is reached."""
        start = time.time()
        attempt = 0
        while time.time() - start < self.timeout:
            attempt += 1
            if self.poke():
                print(f"    [FileSensor] File found after {attempt} poke(s): {self.filepath}")
                return True
            elapsed = time.time() - start
            print(f"    [FileSensor] Poke #{attempt} - not found ({elapsed:.1f}s elapsed)")
            time.sleep(self.poke_interval)
        raise TimeoutError(
            f"FileSensor timed out after {self.timeout}s waiting for {self.filepath}"
        )


def process_file(filepath: str) -> dict:
    """Task that processes the file after the sensor succeeds."""
    with open(filepath, "r") as f:
        content = f.read()
    lines = content.strip().split("\n")
    print(f"    [process_file] Read {len(lines)} lines from {filepath}")
    return {"filepath": filepath, "line_count": len(lines)}


def problem3_sensor():
    """
    Airflow DAG definition equivalent:

        with DAG('sensor_then_process', schedule='@daily', start_date=...) as dag:
            wait = FileSensor(
                task_id='wait_for_file',
                filepath='/data/input/{{ ds }}.csv',
                poke_interval=30,
                timeout=3600,
            )
            process = PythonOperator(
                task_id='process_file',
                python_callable=process_file,
                op_kwargs={'filepath': '/data/input/{{ ds }}.csv'},
            )
            wait >> process
    """
    # Create a temp file to simulate file arrival
    fd, filepath = tempfile.mkstemp(prefix="sensor_test_", suffix=".csv")
    os.close(fd)
    os.remove(filepath)  # Remove first so the sensor has to wait

    # Schedule file creation after a short delay (simulating external process)
    import threading

    def create_file_later():
        time.sleep(1.5)  # Simulate delay
        with open(filepath, "w") as f:
            f.write("id,name,amount\n1,Alice,100\n2,Bob,200\n3,Carol,150\n")
        print(f"    [external_system] File created: {filepath}")

    print("\n  Starting FileSensor (file will appear after ~1.5 seconds)...")
    thread = threading.Thread(target=create_file_later, daemon=True)
    thread.start()

    sensor = FileSensor(filepath=filepath, poke_interval=0.5, timeout=10.0)
    try:
        sensor.execute()
        result = process_file(filepath)
        print(f"  Result: {result}")
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Problem 1: Using XCom (Sum of Two Numbers)")
    print("=" * 70)
    problem1_xcom()

    print()
    print("=" * 70)
    print("Problem 2: Dynamic DAG (ETL per Table)")
    print("=" * 70)
    problem2_dynamic_dag()

    print()
    print("=" * 70)
    print("Problem 3: Using Sensors (Wait for File)")
    print("=" * 70)
    problem3_sensor()
