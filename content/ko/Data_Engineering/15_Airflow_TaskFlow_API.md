# Airflow TaskFlow API

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Airflow 2.x에서 전통적인 오퍼레이터(Operator) 기반 패턴과 TaskFlow API의 핵심 차이를 설명하고, 각 방식의 사용 시점을 정당화할 수 있습니다.
2. `@dag` 및 `@task` 데코레이터를 사용하여 DAG와 태스크를 정의하고, 함수 반환값을 통해 태스크 간 XCom을 자동으로 전달할 수 있습니다.
3. `expand()`와 `partial()`을 사용한 동적 태스크 매핑(Dynamic Task Mapping)으로 런타임에 파라미터화된 태스크 인스턴스를 생성할 수 있습니다.
4. 태스크 그룹(Task Group) 구성 및 의존성 관리를 적용하여 유지보수 가능하고 모듈화된 DAG 구조를 구축할 수 있습니다.
5. 프로덕션 수준의 파이프라인 안정성을 위해 태스크 수준의 재시도, 타임아웃, SLA 콜백, 실패 처리기를 구성할 수 있습니다.
6. TaskFlow API를 단일 DAG 내에서 외부 오퍼레이터(BashOperator, SparkSubmitOperator) 및 센서(Sensor)와 통합할 수 있습니다.

---

## 개요

Airflow 2.0에서 도입된 TaskFlow API는 데코레이터(decorator)를 사용하여 DAG를 파이썬 네이티브 방식으로 정의할 수 있는 방법을 제공합니다. 기존의 오퍼레이터(Operator) 기반 패턴을 `@task` 데코레이터로 대체하여 자동 XCom 전달, 더 깔끔한 코드, 그리고 향상된 타입 안전성(type safety)을 지원합니다. 이것은 Airflow DAG를 작성하는 현대적인 표준입니다.

---

## 1. TaskFlow vs 전통적인 오퍼레이터

### 1.1 비교

```python
"""
=== Traditional Operator Pattern (Airflow 1.x style) ===
보일러플레이트가 얼마나 많이 필요한지 주목하세요: 모든 함수에서 수동 xcom_push/pull,
문자열 기반 task_id 참조(오류 발생 가능), 태스크 간 전달되는 데이터에 타입 안전성 없음.
아래의 TaskFlow 패턴이 이 모든 문제를 해결합니다.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def extract_fn(**kwargs):
    data = {"user_count": 100, "revenue": 5000}
    # XCom에 수동으로 푸시해야 합니다 — 잊기 쉽거나 키 이름을 오타낼 수 있습니다
    kwargs['ti'].xcom_push(key='extracted_data', value=data)

def transform_fn(**kwargs):
    ti = kwargs['ti']
    # 정확한 task_id 문자열을 지정하여 수동으로 풀어야 합니다 — 태스크 이름을 변경하면
    # 오류 대신 None을 조용히 반환합니다
    data = ti.xcom_pull(task_ids='extract', key='extracted_data')
    data['revenue_per_user'] = data['revenue'] / data['user_count']
    ti.xcom_push(key='transformed_data', value=data)

def load_fn(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='transform', key='transformed_data')
    print(f"Loading data: {data}")

with DAG('traditional_etl', start_date=datetime(2024, 1, 1), schedule='@daily') as dag:
    extract = PythonOperator(task_id='extract', python_callable=extract_fn)
    transform = PythonOperator(task_id='transform', python_callable=transform_fn)
    load = PythonOperator(task_id='load', python_callable=load_fn)
    extract >> transform >> load
```

```python
"""
=== TaskFlow API Pattern (Airflow 2.x modern style) ===
위와 동일한 ETL 로직이지만 XCom 전달이 자동(반환값을 통해)이고,
의존성은 함수 호출 체인에서 추론되며, 타입 힌트로 태스크 간 데이터 계약을 문서화합니다.
"""
from airflow.decorators import dag, task
from datetime import datetime

# catchup=False는 배포 시 start_date 이후 모든 과거 날짜를 Airflow가 실행하는 것을 방지합니다
# — 없으면 오늘 start_date=2024-01-01로 생성된 DAG가 즉시 400개 이상의 백필 실행을 대기열에 넣습니다
@dag(schedule='@daily', start_date=datetime(2024, 1, 1), catchup=False)
def taskflow_etl():

    @task()
    def extract() -> dict:
        # 반환값은 자동으로 XCom에 직렬화됩니다 (기본적으로 JSON으로)
        return {"user_count": 100, "revenue": 5000}

    @task()
    def transform(data: dict) -> dict:
        # `data`는 XCom에서 자동으로 역직렬화됩니다 — 수동 풀 불필요
        data['revenue_per_user'] = data['revenue'] / data['user_count']
        return data

    @task()
    def load(data: dict):
        print(f"Loading data: {data}")

    # 의존성은 함수 호출에서 추론됩니다 — `raw_data`를 transform()에 전달하면
    # Airflow에게 transform이 extract에 의존함을 알립니다
    raw_data = extract()
    transformed = transform(raw_data)
    load(transformed)

taskflow_etl()  # Instantiate the DAG
```

### 1.2 주요 차이점

| 항목 | 전통적 방식 | TaskFlow API |
|------|------------|--------------|
| **태스크 정의** | `PythonOperator(...)` | `@task()` 데코레이터 |
| **XCom 전달** | 수동 `xcom_push/pull` | 반환값을 통한 자동 전달 |
| **의존성** | 명시적 `>>` 연산자 | 함수 호출에서 자동 추론 |
| **타입 힌트** | 강제하지 않음 | 지원 (반환 타입 어노테이션) |
| **코드 가독성** | 보일러플레이트(boilerplate) 많음 | 파이써닉(Pythonic)하고 간결 |
| **DAG 정의** | `with DAG(...) as dag:` | `@dag()` 데코레이터 |

---

## 2. @task 데코레이터 기초

### 2.1 반환값과 자동 XCom

```python
from airflow.decorators import dag, task
from datetime import datetime
import json

@dag(schedule='@daily', start_date=datetime(2024, 1, 1), catchup=False,
     tags=['taskflow', 'example'])
def xcom_demo():

    @task()
    def generate_data() -> dict:
        """Return value is automatically stored as XCom."""
        return {
            "users": [
                {"id": 1, "name": "Alice", "score": 95},
                {"id": 2, "name": "Bob", "score": 87},
                {"id": 3, "name": "Charlie", "score": 92},
            ]
        }

    @task()
    def compute_stats(data: dict) -> dict:
        """Input is automatically pulled from XCom."""
        scores = [u["score"] for u in data["users"]]
        return {
            "mean": sum(scores) / len(scores),
            "max": max(scores),
            "min": min(scores),
            "count": len(scores),
        }

    @task()
    def report(stats: dict):
        print(f"Stats: mean={stats['mean']:.1f}, "
              f"max={stats['max']}, min={stats['min']}, "
              f"count={stats['count']}")

    data = generate_data()
    stats = compute_stats(data)
    report(stats)

xcom_demo()
```

### 2.2 다중 출력(Multiple Outputs)

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(schedule='@daily', start_date=datetime(2024, 1, 1), catchup=False)
def multiple_outputs_demo():

    # multiple_outputs=True는 딕셔너리의 각 키를 별도의 XCom 항목으로 저장하여
    # 다운스트림 태스크가 전체 딕셔너리 대신 특정 키에 의존할 수 있습니다.
    # 불필요한 데이터 풀을 방지하고 부분 의존성을 가능하게 합니다:
    # process_users는 "users"가 변경될 때만 재실행됩니다
    @task(multiple_outputs=True)
    def split_data() -> dict:
        """반환된 딕셔너리의 각 키가 별도의 XCom이 됩니다."""
        return {
            "users": [{"id": 1}, {"id": 2}],
            "metadata": {"source": "api", "timestamp": "2024-01-01"},
            "count": 2,
        }

    @task()
    def process_users(users: list):
        print(f"Processing {len(users)} users")

    @task()
    def process_metadata(metadata: dict):
        print(f"Source: {metadata['source']}")

    # 딕셔너리 키 접근은 세분화된 의존성을 만듭니다: process_users는
    # "metadata"나 "count"가 아닌 "users" XCom에만 의존합니다
    result = split_data()
    process_users(result["users"])
    process_metadata(result["metadata"])

multiple_outputs_demo()
```

### 2.3 커스텀 파라미터가 있는 태스크

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(schedule='@daily', start_date=datetime(2024, 1, 1), catchup=False)
def custom_params_demo():

    @task(
        task_id='custom_extract',
        retries=3,
        retry_delay=60,  # seconds
        # execution_timeout은 실행이 무기한 워커 슬롯을 점유하는 것을 방지합니다
        # (예: 멈춘 API 호출)
        execution_timeout=300,  # 5 minutes
        # pool은 동시성을 제한합니다: 'data_pool'은 5개 슬롯이 있어
        # 이 DAG가 병렬 읽기로 소스 데이터베이스를 압도하는 것을 방지합니다
        pool='data_pool',
        # queue는 특정 워커로 태스크를 라우팅합니다 (예: 더 많은 메모리나
        # 데이터 소스에 대한 네트워크 접근이 있는 워커)
        queue='high_priority',
    )
    def extract(source: str, limit: int = 100) -> list:
        """Task with custom Airflow parameters and function arguments."""
        print(f"Extracting from {source} with limit {limit}")
        return [{"id": i} for i in range(limit)]

    @task(trigger_rule='all_success')
    def validate(records: list) -> bool:
        assert len(records) > 0, "No records extracted"
        return True

    data = extract(source="api_v2", limit=50)
    validate(data)

custom_params_demo()
```

---

## 3. 다양한 런타임을 사용하는 TaskFlow

### 3.1 런타임 데코레이터

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(schedule='@daily', start_date=datetime(2024, 1, 1), catchup=False)
def runtime_demo():

    # Standard Python task (Airflow 워커 프로세스에서 실행)
    # 가장 빠른 시작이지만 워커의 Python 환경을 공유합니다 —
    # Airflow 자체와의 의존성 충돌이 가능합니다
    @task()
    def python_task():
        return {"source": "python"}

    # 가상 환경 태스크: 실행마다 격리된 Python 환경을 생성합니다.
    # 의존성 충돌을 해결합니다 (예: 태스크에 pandas 2.1이 필요하지만
    # Airflow가 구버전을 요구하는 경우). 트레이드오프: 매 실행마다
    # 가상 환경 생성을 위한 ~10-30초 시작 오버헤드
    @task.virtualenv(
        requirements=["pandas==2.1.0", "numpy==1.25.0"],
        python_version="3.11",
        system_site_packages=False,
    )
    def virtualenv_task():
        import pandas as pd
        import numpy as np
        df = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
        return {"rows": len(df), "mean_a": float(df['a'].mean())}

    # Docker 태스크: 완전한 컨테이너 격리 — 호스트 환경에 관계없이
    # 재현 가능한 실행을 보장합니다. 복잡한 네이티브 의존성(C 라이브러리,
    # GPU 드라이버)이 있는 태스크에 이상적입니다. 가상 환경보다 시작 비용이 높습니다
    @task.docker(
        image="python:3.11-slim",
        auto_remove="success",
        mount_tmp_dir=False,
    )
    def docker_task():
        import json
        result = {"environment": "docker", "status": "ok"}
        print(json.dumps(result))

    # Branch task (conditional execution)
    @task.branch()
    def decide_path() -> str:
        import random
        return "fast_path" if random.random() > 0.5 else "slow_path"

    @task(task_id="fast_path")
    def fast():
        return "fast result"

    @task(task_id="slow_path")
    def slow():
        return "slow result"

    # Build DAG
    py_result = python_task()
    venv_result = virtualenv_task()
    branch = decide_path()
    fast()
    slow()

runtime_demo()
```

---

## 4. 동적 태스크 매핑(Dynamic Task Mapping)

### 4.1 동적 병렬 처리를 위한 expand()

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(schedule='@daily', start_date=datetime(2024, 1, 1), catchup=False)
def dynamic_mapping_demo():

    @task()
    def get_partitions() -> list[str]:
        """Returns a list — each element generates a mapped task instance."""
        return ["us-east", "us-west", "eu-west", "ap-south"]

    @task()
    def process_partition(partition: str) -> dict:
        """This task runs once per partition (4 parallel instances)."""
        import random
        records = random.randint(100, 1000)
        return {"partition": partition, "records": records}

    @task()
    def aggregate(results: list[dict]) -> dict:
        """Receives a list of all mapped task outputs."""
        total = sum(r["records"] for r in results)
        return {"total_records": total, "partitions": len(results)}

    # expand()을 사용한 동적 매핑: 태스크 인스턴스 수가 런타임에
    # 업스트림 태스크의 출력에서 결정됩니다 — DAG 파일에 하드코딩되지 않습니다.
    # 파티션 목록이 변경될 때(예: 새 지역 추가) DAG 코드 변경과 재배포 없이
    # 자동으로 처리됩니다
    partitions = get_partitions()
    processed = process_partition.expand(partition=partitions)
    aggregate(processed)

dynamic_mapping_demo()
```

### 4.2 partial()과 함께 사용하는 expand()

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(schedule='@daily', start_date=datetime(2024, 1, 1), catchup=False)
def partial_expand_demo():

    @task()
    def get_files() -> list[str]:
        return ["data_2024_01.csv", "data_2024_02.csv", "data_2024_03.csv"]

    @task()
    def process_file(file_path: str, output_format: str, validate: bool) -> dict:
        """Process a file with shared configuration."""
        return {
            "file": file_path,
            "format": output_format,
            "validated": validate,
            "status": "success",
        }

    files = get_files()

    # partial()은 모든 매핑된 인스턴스에 공유되는 인자를 고정하고, expand()는
    # 인스턴스별 인자를 변경합니다. 이 분리는 모든 매핑된 호출에 설정을 반복하는 것을
    # 피하고 공유 설정을 한 곳에서 쉽게 변경할 수 있게 합니다.
    # partial() 없이는 공유 값을 변하는 목록과 zip해야 합니다
    results = process_file.partial(
        output_format="parquet",  # 모든 인스턴스에 동일
        validate=True,            # 모든 인스턴스에 동일
    ).expand(
        file_path=files,          # 각 인스턴스마다 다름
    )

partial_expand_demo()
```

### 4.3 여러 인자에 대한 매핑

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(schedule='@daily', start_date=datetime(2024, 1, 1), catchup=False)
def zip_mapping_demo():

    @task()
    def get_configs() -> list[dict]:
        return [
            {"table": "users", "schema": "public"},
            {"table": "orders", "schema": "sales"},
            {"table": "products", "schema": "catalog"},
        ]

    @task()
    def extract_table(config: dict) -> dict:
        return {
            "table": f"{config['schema']}.{config['table']}",
            "rows": 1000,
        }

    @task()
    def summarize(extractions: list[dict]):
        for e in extractions:
            print(f"Extracted {e['rows']} from {e['table']}")

    configs = get_configs()
    results = extract_table.expand(config=configs)
    summarize(results)

zip_mapping_demo()
```

---

## 5. TaskGroup과 함께 사용하는 TaskFlow

### 5.1 태스크 구성하기

```python
from airflow.decorators import dag, task, task_group
from datetime import datetime

@dag(schedule='@daily', start_date=datetime(2024, 1, 1), catchup=False)
def taskgroup_demo():

    # @task_group은 Airflow UI Graph 뷰에 시각적 경계를 만듭니다.
    # extract_group 내에서 세 가지 추출이 모두 병렬로 실행됩니다 (서로 의존성 없음),
    # 처리량을 최대화합니다. 그룹 자체는 다운스트림 그룹에 연결할 때
    # 단일 노드로 작동합니다
    @task_group()
    def extract_group():
        """관련 추출 태스크를 묶습니다."""

        @task()
        def extract_users() -> list:
            return [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

        @task()
        def extract_orders() -> list:
            return [{"id": 101, "user_id": 1, "amount": 50.0}]

        @task()
        def extract_products() -> list:
            return [{"id": "P1", "name": "Widget", "price": 25.0}]

        # 태스크 출력의 딕셔너리를 반환하면 transform_group이
        # 키로 각 데이터셋에 접근할 수 있어 명확한 데이터 계약을 유지합니다
        return {
            "users": extract_users(),
            "orders": extract_orders(),
            "products": extract_products(),
        }

    @task_group()
    def transform_group(data: dict):
        """Group related transformation tasks."""

        @task()
        def enrich_orders(orders: list, users: list) -> list:
            # O(n*m) 중첩 루프 대신 O(1) 사용자 조회를 위한 룩업 맵을 구축합니다
            # — 대규모 데이터셋에서 중요합니다
            user_map = {u["id"]: u["name"] for u in users}
            for order in orders:
                order["user_name"] = user_map.get(order["user_id"], "Unknown")
            return orders

        @task()
        def calculate_totals(orders: list) -> dict:
            return {"total_revenue": sum(o["amount"] for o in orders)}

        enriched = enrich_orders(data["orders"], data["users"])
        return calculate_totals(enriched)

    @task()
    def load(totals: dict):
        print(f"Total revenue: ${totals['total_revenue']:.2f}")

    raw_data = extract_group()
    totals = transform_group(raw_data)
    load(totals)

taskgroup_demo()
```

---

## 6. TaskFlow와 전통적인 오퍼레이터 혼용

### 6.1 하이브리드 DAG(Hybrid DAG)

```python
from airflow.decorators import dag, task
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime

@dag(schedule='@daily', start_date=datetime(2024, 1, 1), catchup=False)
def hybrid_dag():

    # 센서(전통적 오퍼레이터) — 외부 의존성을 기다립니다.
    # 센서는 Python 로직이 아닌 외부 상태를 폴링하기 때문에
    # TaskFlow 동등물이 없습니다. poke_interval=60은 1분마다 확인하고;
    # timeout=3600은 1시간 후에도 파일이 나타나지 않으면 태스크를 실패시킵니다
    wait_for_file = FileSensor(
        task_id='wait_for_file',
        filepath='/data/input/daily_{{ ds }}.csv',
        poke_interval=60,
        timeout=3600,
    )

    # TaskFlow 태스크 — ds=None은 특별한 Airflow 패턴입니다: 런타임에
    # Airflow가 논리적 실행 날짜를 문자열로 주입합니다. 이를 통해
    # 하드코딩 없이 올바른 날짜의 파일을 처리할 수 있습니다
    @task()
    def process_file(ds=None) -> dict:
        """ds는 Airflow에 의해 자동으로 주입됩니다 (논리적 날짜)."""
        return {"file": f"/data/input/daily_{ds}.csv", "rows": 1000}

    # SQL 태스크 — PostgresOperator가 @task()보다 SQL에 선호됩니다:
    # Airflow 커넥션 매니저(암호화된 자격 증명, 커넥션 풀링)를 사용하고
    # Airflow UI에서 디버깅을 위해 SQL을 렌더링합니다
    create_table = PostgresOperator(
        task_id='create_staging_table',
        postgres_conn_id='warehouse',
        sql="""
            CREATE TABLE IF NOT EXISTS staging.daily_data (
                id SERIAL PRIMARY KEY,
                data JSONB,
                loaded_at TIMESTAMP DEFAULT NOW()
            );
        """,
    )

    # TaskFlow task
    @task()
    def load_to_staging(metadata: dict):
        print(f"Loading {metadata['rows']} rows from {metadata['file']}")

    # Bash task (traditional operator)
    cleanup = BashOperator(
        task_id='cleanup',
        bash_command='echo "Cleaning up temp files"',
    )

    # 전통적 오퍼레이터와 TaskFlow 태스크를 동일한 의존성 체인에서 혼용하는 것은
    # 완전히 지원됩니다. >> 연산자가 둘을 원활하게 연결합니다 — Airflow는
    # 어떻게 정의되었는지에 관계없이 둘 다 1급(first-class) Task 인스턴스로 취급합니다
    metadata = process_file()
    wait_for_file >> metadata >> create_table >> load_to_staging(metadata) >> cleanup

hybrid_dag()
```

---

## 7. TaskFlow DAG 테스트

### 7.1 태스크 단위 테스트(Unit Testing)

```python
"""
TaskFlow tasks can be tested as regular Python functions.
"""
import pytest

# The decorated task function
from airflow.decorators import task

@task()
def calculate_metrics(data: list) -> dict:
    if not data:
        raise ValueError("Empty data")
    values = [d["value"] for d in data]
    return {
        "mean": sum(values) / len(values),
        "count": len(values),
        "total": sum(values),
    }

# 내부 함수를 직접 호출하여 테스트합니다.
# .function은 Airflow 데코레이터를 우회하므로 Airflow 환경을 시작하지 않고도
# 순수 비즈니스 로직을 테스트할 수 있습니다 — 테스트가 밀리초 안에 실행됩니다
def test_calculate_metrics():
    data = [{"value": 10}, {"value": 20}, {"value": 30}]
    result = calculate_metrics.function(data)
    assert result["mean"] == 20.0
    assert result["count"] == 3
    assert result["total"] == 60

def test_calculate_metrics_empty():
    # 오류 경로 테스트가 중요합니다: 빈 데이터가 조용히 통과하면
    # 다운스트림 태스크가 쓰레기를 처리하고 잘못된 결과를 생성합니다
    with pytest.raises(ValueError, match="Empty data"):
        calculate_metrics.function([])

# Run: pytest test_tasks.py -v
```

### 7.2 DAG 유효성 테스트

```python
"""
Test that DAG loads without errors and has correct structure.
"""
import pytest
from airflow.models import DagBag

def test_dag_loaded():
    """Test DAG file can be imported without errors."""
    dagbag = DagBag(dag_folder='dags/', include_examples=False)
    assert len(dagbag.import_errors) == 0, f"DAG import errors: {dagbag.import_errors}"

def test_dag_structure():
    """Test DAG has expected tasks."""
    dagbag = DagBag(dag_folder='dags/', include_examples=False)
    dag = dagbag.get_dag('taskflow_etl')

    assert dag is not None
    task_ids = [t.task_id for t in dag.tasks]
    assert 'extract' in task_ids
    assert 'transform' in task_ids
    assert 'load' in task_ids

def test_dag_dependencies():
    """Test task dependencies are correct."""
    dagbag = DagBag(dag_folder='dags/', include_examples=False)
    dag = dagbag.get_dag('taskflow_etl')

    extract = dag.get_task('extract')
    transform = dag.get_task('transform')
    assert 'extract' in [t.task_id for t in transform.upstream_list]
```

---

## 8. 마이그레이션 가이드: 전통적 방식에서 TaskFlow로

### 8.1 단계별 마이그레이션

```python
"""
Migration Checklist:

1. Replace `with DAG(...):` with `@dag(...)` decorator
2. Replace PythonOperator with @task() decorator
3. Remove manual xcom_push/xcom_pull
4. Return values instead of pushing XCom
5. Accept function parameters instead of pulling XCom
6. Let dependency be inferred from function calls
7. Remove explicit >> operators (unless mixing with traditional)

Common Pitfalls:
- @task functions must be serializable (no lambdas, no closures over non-serializable objects)
- Return values are stored in XCom (default: metadata DB) — avoid large returns
  → Use custom XCom backend for large data (S3, GCS)
- @task functions run in the worker process — avoid global state
"""
```

### 8.2 대용량 데이터를 위한 커스텀 XCom 백엔드

```python
"""
Default XCom stores data in the Airflow metadata DB (serialized JSON/pickle).
For large datasets, use a custom XCom backend:

# airflow.cfg
[core]
xcom_backend = airflow.providers.amazon.aws.xcom_backends.s3.S3XComBackend

# Environment variables
AIRFLOW__CORE__XCOM_BACKEND=airflow.providers.amazon.aws.xcom_backends.s3.S3XComBackend
XCOM_BACKEND_BUCKET_NAME=my-airflow-xcom
XCOM_BACKEND_PREFIX=xcom/

Now @task return values are automatically stored in S3 instead of the DB.

Alternative backends:
  - S3: airflow.providers.amazon.aws.xcom_backends.s3.S3XComBackend
  - GCS: airflow.providers.google.cloud.xcom_backends.gcs.GCSXComBackend
  - Custom: Subclass BaseXCom and implement serialize/deserialize
"""
```

---

## 9. 모범 사례(Best Practices)

### 9.1 TaskFlow 가이드라인

```python
"""
1. PREFER TaskFlow for Python tasks
   ✓ @task() for Python logic
   ✓ Traditional operators for external systems (SQL, Bash, sensors)

2. KEEP tasks small and focused
   ✓ Each task does one thing
   ✗ Avoid monolithic tasks that do extract + transform + load

3. USE type hints
   ✓ def extract() -> dict:
   ✓ def transform(data: dict) -> list:

4. AVOID large XCom values
   ✓ Pass metadata/references (file paths, S3 keys)
   ✗ Pass entire DataFrames or large datasets

5. USE task_group for organization
   ✓ Group related tasks (extract_group, transform_group)
   ✓ Keeps the DAG graph readable

6. USE dynamic mapping for parallel processing
   ✓ process_partition.expand(partition=get_partitions())
   ✗ Hardcoded parallel tasks (process_1, process_2, ...)

7. TEST tasks as functions
   ✓ calculate_metrics.function(test_data)
   ✓ pytest for both function logic and DAG structure
"""
```

---

## 10. 연습 문제

### 연습 1: 전통적인 DAG를 TaskFlow로 변환하기

```python
"""
Convert this traditional DAG to TaskFlow API:

with DAG('legacy_pipeline', ...) as dag:
    def fetch(**ctx):
        ctx['ti'].xcom_push(key='records', value=[1, 2, 3, 4, 5])

    def double(**ctx):
        records = ctx['ti'].xcom_pull(task_ids='fetch', key='records')
        ctx['ti'].xcom_push(key='doubled', value=[r * 2 for r in records])

    def save(**ctx):
        data = ctx['ti'].xcom_pull(task_ids='double', key='doubled')
        print(f"Saving {data}")

    t1 = PythonOperator(task_id='fetch', python_callable=fetch)
    t2 = PythonOperator(task_id='double', python_callable=double)
    t3 = PythonOperator(task_id='save', python_callable=save)
    t1 >> t2 >> t3

Requirements:
1. Use @dag and @task decorators
2. Use return values instead of xcom_push/pull
3. Add type hints
4. Add a task_group for fetch + double
"""
```

### 연습 2: 동적 ETL 파이프라인

```python
"""
Build a TaskFlow DAG that:
1. Reads a config file listing 5 database tables to sync
2. Uses dynamic task mapping to process each table in parallel
3. Each mapped task: extracts row count + schema info
4. An aggregation task collects all results
5. A final task generates a sync report

Use: @task, expand(), partial(), task_group
"""
```

---

## 연습 문제

### 연습 1: 레거시 DAG를 TaskFlow로 마이그레이션

다음 전통적인 DAG를 TaskFlow API로 변환하세요:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def score_users(**kwargs):
    ti = kwargs['ti']
    users = ti.xcom_pull(task_ids='load_users', key='users')
    scores = {u['id']: u['clicks'] * 2 + u['purchases'] * 10 for u in users}
    ti.xcom_push(key='scores', value=scores)

def filter_vip(**kwargs):
    ti = kwargs['ti']
    scores = ti.xcom_pull(task_ids='score_users', key='scores')
    vips = {uid: s for uid, s in scores.items() if s >= 50}
    ti.xcom_push(key='vips', value=vips)

def send_campaign(**kwargs):
    ti = kwargs['ti']
    vips = ti.xcom_pull(task_ids='filter_vip', key='vips')
    print(f"Sending campaign to {len(vips)} VIP users")

with DAG('legacy_campaign', start_date=datetime(2024, 1, 1), schedule='@weekly') as dag:
    t1 = PythonOperator(task_id='load_users', python_callable=lambda **kw: kw['ti'].xcom_push(key='users', value=[]))
    t2 = PythonOperator(task_id='score_users', python_callable=score_users)
    t3 = PythonOperator(task_id='filter_vip', python_callable=filter_vip)
    t4 = PythonOperator(task_id='send_campaign', python_callable=send_campaign)
    t1 >> t2 >> t3 >> t4
```

요구 사항:
1. `@dag`와 `@task` 데코레이터를 사용하고 적절한 타입 힌트를 추가하세요
2. 모든 `xcom_push`/`xcom_pull`을 자동 반환값 전달 방식으로 교체하세요
3. `score_users`와 `filter_vip`를 `scoring_group`이라는 `@task_group`으로 묶으세요
4. `send_campaign` 태스크에 `retries=2`와 `execution_timeout=120`을 추가하세요
5. `.function()`을 사용하여 점수 계산 로직을 검증하는 단위 테스트(unit test)를 작성하세요

### 연습 2: 동적 다중 지역 ETL

여러 지역의 데이터를 동적으로 처리하는 TaskFlow DAG를 구축하세요:

1. `get_regions()` 태스크가 지역 설정 목록을 반환합니다: `[{"name": "us-east", "bucket": "s3://us-east/data"}, ...]`
2. `expand()`를 사용하여 지역별 행 수를 시뮬레이션하는 `process_region(config: dict)` 태스크를 팬아웃(fan-out)하세요
3. `partial()`을 사용하여 모든 매핑된 인스턴스에 공통 `output_format="parquet"` 파라미터를 고정하세요
4. `aggregate_results(results: list[dict])` 태스크가 모든 출력을 수집하고 총 행 수를 계산합니다
5. `generate_report(summary: dict)` 태스크가 형식화된 요약 표를 출력합니다

`get_regions()`에 새 지역을 추가하면 DAG 코드 변경 없이 자동으로 새 태스크 인스턴스가 생성되는지 확인하세요.

### 연습 3: SLA와 알림을 갖춘 하이브리드 DAG

TaskFlow와 전통적인 오퍼레이터(Operator)를 결합하고 프로덕션 수준의 안정성을 갖춘 하이브리드 DAG를 설계하세요:

1. `FileSensor`가 일별 입력 파일(`/data/input/{{ ds }}.json`)을 기다립니다
2. `@task`가 파일을 읽고 파싱된 레코드를 반환합니다 (`ds` 컨텍스트 변수 사용)
3. `pandas==2.1.0`이 설치된 `@task.virtualenv`가 레코드에 대한 통계 분석을 수행합니다
4. `PostgresOperator`가 요약 결과를 스테이징(staging) 테이블에 씁니다
5. `@task`가 스테이징 테이블을 읽고 지표가 임계값을 초과하면 알림을 전송합니다
6. DAG 전체에 2시간 SLA를 구성하고 실패한 태스크 ID와 실행 날짜를 로그로 남기는 `on_failure_callback`을 구현하세요
7. 5개 이상의 태스크 ID가 모두 존재하고 의존성 순서가 올바른지 검증하는 `DagBag` 테스트를 작성하세요

### 연습 4: 대용량 페이로드를 위한 커스텀 XCom 백엔드

기본 XCom 백엔드(backend)는 Airflow 메타데이터 데이터베이스에 데이터를 저장하여 대형 데이터프레임의 병목이 됩니다. 이 문제를 해결하세요:

1. 대형 데이터프레임을 직접 XCom으로 전달하면 성능 문제가 발생하는 이유를 주석으로 설명하세요
2. `extract()` 태스크가 대규모 데이터셋을 생성하지만 XCom으로는 **파일 경로**만 반환하도록 DAG를 재설계하세요 (로컬 Parquet 파일에 쓰기)
3. `transform(path: str)` 태스크가 파일을 읽어 집계를 수행하고 결과를 새 경로에 씁니다
4. `load(path: str)` 태스크가 최종 파일을 읽고 행 수를 출력합니다
5. `expand()`를 사용하여 모든 중간 파일을 삭제하는 `cleanup(paths: list[str])` 태스크를 추가하세요

### 연습 5: 분기 및 병렬 병합 패턴

실제 A/B 결정을 모델링하는 조건부 처리 DAG를 구현하세요:

1. `@task.branch` 태스크가 `processing_mode`라는 Airflow Variable을 검사하고 `"fast_path"` 또는 `"full_path"`를 반환합니다
2. `fast_path` 그룹은 5초 이내에 결과를 근사치로 계산하는 단일 `@task`를 사용합니다
3. `full_path` 그룹은 데이터 세그먼트별 3개의 병렬 `@task` 인스턴스를 사용하고 이어서 집계(aggregation) 태스크를 실행합니다
4. 두 분기는 `trigger_rule="none_failed_min_one_success"`를 사용하는 `report(result: dict)` 태스크에서 합쳐집니다
5. `.function()`을 사용하여 분기 결정 로직과 빠른 경로(fast-path) 집계 로직 모두에 대한 테스트를 작성하세요

---

## 11. 정리

### 핵심 내용

| 개념 | 설명 |
|------|------|
| **@task 데코레이터** | 파이썬 함수를 데코레이터로 감싸 태스크로 정의 |
| **자동 XCom** | 반환값이 자동으로 저장되고 전달됨 |
| **@dag 데코레이터** | 함수를 데코레이터로 감싸 DAG로 정의 |
| **동적 매핑** | `expand()`가 런타임에 병렬 태스크 인스턴스를 생성 |
| **partial()** | 매핑된 태스크들에 공통 인자를 공유 |
| **task_group** | 관련 태스크를 시각적·논리적으로 구성 |
| **런타임 변형** | `@task.virtualenv`, `@task.docker`, `@task.branch` |
| **하이브리드 DAG** | TaskFlow와 전통적인 오퍼레이터를 자유롭게 혼용 |

### 모범 사례

1. **TaskFlow 사용** — 모든 파이썬 태스크에 적용, 현대적인 표준
2. **반환값을 작게 유지** — 데이터 자체가 아닌 참조(경로, S3 키 등)를 전달
3. **동적 매핑 활용** — 하드코딩된 병렬 태스크 대신 사용
4. **태스크를 함수로 테스트** — `my_task.function(args)`로 단위 테스트
5. **점진적 마이그레이션** — 동일한 DAG 내에서 구방식과 신방식을 혼용 가능

### 다음 단계

- **L16**: Kafka Streams와 ksqlDB — 실시간 스트림 처리(stream processing)
- **L17**: Spark 구조적 스트리밍(Spark Structured Streaming) — Spark DataFrame 기반 스트리밍
