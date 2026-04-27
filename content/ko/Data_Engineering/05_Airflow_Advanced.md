[← 이전: 4. Apache Airflow 기초](04_Apache_Airflow_Basics.md) | [다음: 6. Prefect 모던 오케스트레이션 →](06_Prefect_Modern_Orchestration.md)

# Airflow 심화

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. XCom을 사용하여 Airflow 태스크 간 데이터를 공유하고, 외부 저장소와 비교한 XCom의 한계를 이해할 수 있다
2. Python을 사용하여 동적 DAG(Dynamic DAG)를 프로그래밍 방식으로 생성하여 가변적인 태스크 수나 데이터 소스를 처리할 수 있다
3. 센서(Sensor)를 구현하여 외부 조건을 기다리는 이벤트 기반 워크플로우를 구축할 수 있다
4. 커스텀 훅(Hook)과 오퍼레이터(Operator)를 작성하여 Airflow의 통합 기능을 확장할 수 있다
5. TaskGroup을 사용하여 복잡한 DAG를 구조화하고, BranchPythonOperator로 분기(Branching)를 적용할 수 있다
6. 에러 처리, 재시도(Retry), 알림(Alerting)에 대한 Airflow 모범 사례를 적용하여 프로덕션 수준의 파이프라인을 설계하고 구현할 수 있다

---

## 개요

이 문서에서는 Airflow의 고급 기능인 XCom을 통한 데이터 공유, 동적 DAG 생성, Sensor, Hook, TaskGroup 등을 다룹니다. 이러한 기능을 활용하면 더 유연하고 강력한 파이프라인을 구축할 수 있습니다.

---

## 1. XCom (Cross-Communication)

### 이론: XCom: 태스크 간 작은 메시지

XCom(cross-communication)은 한 태스크가 다른 태스크에 값을 전달하는 Airflow 메커니즘입니다. 개념적으로 `(dag_id, run_id, task_id, key)`로 키된 키-값 저장소. 기본적으로 메타데이터 DB에 저장됩니다.

#### A.1 크기 제약

기본 백엔드는 메타데이터 Postgres/MySQL — 스케줄러가 끊임없이 두드리는 같은 DB. 수 KB보다 큰 XCom 값은 이 DB를 중독시킵니다:

- 모든 태스크 목록 쿼리가 부풀어 오른 XCom 행을 스캔하므로 스케줄러가 느려집니다.
- 백업이 부풀어 오릅니다.
- DB 연결이 포화됩니다.

**원칙:** XCom은 *페이로드(payload)* 가 아닌 *포인터(pointer)* 를 위한 것입니다. `s3://bucket/output_2024_03_15.parquet`을 전달하고, parquet 바이트는 절대 전달하지 마세요. 메타데이터, ID, 상태 플래그의 작은 dict — 괜찮음. 수 KB보다 클 수 있는 모든 것 — S3/GCS에 쓰고 경로를 전달하세요.

더 큰 객체를 전달해야 하는 워크플로우의 경우, **커스텀 XCom 백엔드** 사용: `BaseXCom`을 서브클래스, `serialize_value`에서 S3/GCS로 직렬화, DB에는 경로만 저장, 읽을 때 역직렬화. 이 패턴은 API를 보존하면서 페이로드를 메타데이터 DB 밖으로 옮깁니다.

#### A.2 Pull 의미론과 멱등성

태스크 B가 태스크 A의 XCom을 pull할 때, *동일한 DAG 실행* 의 값을 pull합니다. 이는 의미합니다:

- 태스크 B는 동일한 논리 날짜 내에서 태스크 A에 암묵적으로 의존.
- 태스크 B의 재실행은 동일한 XCom 값을 봄(태스크 A가 멱등하다고 가정).
- 날짜 D의 백필은 프로덕션의 가장 최근이 아닌 날짜 D의 태스크 A 실행에서 XCom 값을 pull.

이것이 XCom을 재시도와 백필 하에서 안전하게 만드는 것입니다 — 글로벌 상태가 아닌 실행에 스코프됨.

### 1.1 XCom 기본 사용법

XCom은 Task 간에 작은 데이터를 공유하는 메커니즘입니다.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def push_data(**kwargs):
    """XCom으로 데이터 푸시"""
    ti = kwargs['ti']

    # 방법 1: 명시적 키로 xcom_push — 태스크가 여러 개의 별개 값
    # (예: 상태 + 메트릭)을 게시해야 할 때 사용한다.
    ti.xcom_push(key='my_key', value={'status': 'success', 'count': 100})

    # 방법 2: return 값 — 하나의 출력이 일반적인 경우에 더 간단하다.
    # key='return_value'로 자동 저장된다.
    return {'result': 'completed', 'rows': 500}

def pull_data(**kwargs):
    """XCom에서 데이터 가져오기"""
    ti = kwargs['ti']

    # 명시적 키로 가져오기 — 상위 태스크가 여러 값을 푸시했고
    # 특정 값이 필요할 때 반드시 사용한다.
    custom_data = ti.xcom_pull(key='my_key', task_ids='push_task')
    print(f"Custom data: {custom_data}")

    # 반환 값 가져오기 (키 생략 시 'return_value' 기본값) —
    # 단순한 태스크 간 통신을 위한 가장 일반적인 패턴이다.
    return_value = ti.xcom_pull(task_ids='push_task')
    print(f"Return value: {return_value}")

    # 여러 태스크에서 한 번에 가져오기 — 하위 태스크가 병렬 상위
    # 태스크들의 결과를 집계하는 팬인(fan-in) 패턴에 유용하다.
    multiple_results = ti.xcom_pull(task_ids=['task1', 'task2'])

with DAG('xcom_example', start_date=datetime(2024, 1, 1), schedule_interval=None) as dag:

    push_task = PythonOperator(
        task_id='push_task',
        python_callable=push_data,
    )

    pull_task = PythonOperator(
        task_id='pull_task',
        python_callable=pull_data,
    )

    push_task >> pull_task
```

### 1.2 Jinja 템플릿에서 XCom 사용

```python
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator

# Bash에서 XCom 사용
bash_task = BashOperator(
    task_id='bash_with_xcom',
    bash_command='echo "Result: {{ ti.xcom_pull(task_ids="push_task") }}"',
)

# SQL에서 XCom 사용
sql_task = PostgresOperator(
    task_id='sql_with_xcom',
    postgres_conn_id='my_postgres',
    sql="""
        INSERT INTO process_log (task_id, result_count, processed_at)
        VALUES (
            'data_load',
            {{ ti.xcom_pull(task_ids='count_task', key='row_count') }},
            NOW()
        );
    """,
)
```

### 1.3 XCom 제한 사항 및 대안

```python
# XCom 제한: 값은 메타데이터 DB에 직렬화된다 (기본 1GB).
# 큰 XCom 값은 DB를 부풀리고, UI를 느리게 하며, 직렬화 중 OOM 오류를
# 일으킬 수 있다. 경험상: XCom 값은 ~50 KB 미만으로 유지한다.

# 권장 패턴: XCom으로 *경로*(포인터)를 전달하고, 실제 데이터는
# 외부 스토리지(S3, GCS)에 저장한다. 이렇게 하면 데이터 크기와 XCom 한계를 분리한다.
class LargeDataHandler:
    """대용량 데이터 전달 패턴"""

    @staticmethod
    def save_to_storage(data, path: str):
        """데이터를 외부 스토리지에 저장하고 경로만 XCom으로 전달"""
        import pandas as pd

        # 외부 스토리지(S3/GCS)는 임의의 큰 파일을 처리하고;
        # XCom은 ~50바이트의 경로 문자열만 저장한다.
        data.to_parquet(path)
        return path

    @staticmethod
    def load_from_storage(path: str):
        """경로에서 데이터 로드"""
        import pandas as pd
        return pd.read_parquet(path)

# 사용 예시
def produce_large_data(**kwargs):
    import pandas as pd

    df = pd.DataFrame({'col': range(1000000)})

    # 날짜 파티션 경로는 같은 파일을 덮어써 중복을 만들지 않으므로
    # 멱등성 재실행을 보장한다.
    path = f"s3://bucket/data/{kwargs['ds']}/output.parquet"
    df.to_parquet(path)

    # DataFrame이 아닌 경로 문자열(~50바이트)만 XCom에 저장된다.
    return path

def consume_large_data(**kwargs):
    import pandas as pd

    ti = kwargs['ti']
    # 경로를 가져온 후 스토리지에서 실제 데이터를 읽는다 —
    # 메타데이터 DB를 가볍게 유지하고 파이프라인의 확장성을 보장한다.
    path = ti.xcom_pull(task_ids='produce_task')

    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows from {path}")
```

---

## 2. 동적 DAG 생성

### 이론: 동적 태스크 매핑

병렬 태스크 수가 런타임에만 알려지는 경우가 있습니다: "오늘 도착한 모든 파일을 처리하라." Airflow가 이를 처리한 두 가지 방법:

#### C.1 옛 방식: 동적 DAG 생성

파싱 시점에 파일 패턴당 DAG 하나 생성. `for filename in glob("/data/*.csv"): create_dag(filename)`. 문제:

- DAG 파일이 몇 초마다 파싱됨 — 파싱마다 파일 목록 실행.
- 파일이 나타나/사라짐에 따라 DAG가 왔다 갔다; UI가 어수선해짐.
- DAG 간 의존성이 어색.

#### C.2 새 방식: 동적 태스크 매핑 (Airflow 2.3+)

런타임에 N개의 병렬 인스턴스로 확장되는 단일 태스크:

```python
@task
def list_files() -> list[str]:
    return [...]

@task
def process(filename: str):
    ...

process.expand(filename=list_files())
```

런타임에 `list_files`가 먼저 실행; 그 출력(리스트)이 `process` 인스턴스가 몇 개 생성되는지 결정. 각각이 한 요소를 받음. 확장은 `list_files` 성공 후 스케줄러에서 발생. UI에서 N개의 매핑된 인스턴스가 있는 단일 "process" 노드로 보임.

이것이 "런타임에 결정되는 컬렉션에 대한 fan out"의 옳은 패턴입니다. 더 깨끗한 코드, 더 나은 UI, 파싱 시점 작업 없음.

### 2.1 설정 기반 동적 DAG

```python
# dags/dynamic_dag_factory.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# 설정 정의
DAG_CONFIGS = [
    {
        'dag_id': 'etl_customers',
        'table': 'customers',
        'schedule': '0 1 * * *',
    },
    {
        'dag_id': 'etl_orders',
        'table': 'orders',
        'schedule': '0 2 * * *',
    },
    {
        'dag_id': 'etl_products',
        'table': 'products',
        'schedule': '0 3 * * *',
    },
]

def create_dag(config: dict) -> DAG:
    """설정 기반으로 DAG 생성"""

    def extract_table(table_name: str, **kwargs):
        print(f"Extracting {table_name} for {kwargs['ds']}")

    def load_table(table_name: str, **kwargs):
        print(f"Loading {table_name} for {kwargs['ds']}")

    dag = DAG(
        dag_id=config['dag_id'],
        schedule_interval=config['schedule'],
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['dynamic', 'etl'],
    )

    with dag:
        extract = PythonOperator(
            task_id='extract',
            python_callable=extract_table,
            op_kwargs={'table_name': config['table']},
        )

        load = PythonOperator(
            task_id='load',
            python_callable=load_table,
            op_kwargs={'table_name': config['table']},
        )

        extract >> load

    return dag

# DAG들을 globals()에 등록: Airflow의 DagBag 파서는 모듈의
# 전역 네임스페이스에서 DAG 객체를 검사 — DAG가 globals()에 없으면
# 스케줄러는 그것을 인식하지 못한다. 이 루프는 설정 항목당 하나의 DAG를 생성한다.
for config in DAG_CONFIGS:
    dag_id = config['dag_id']
    globals()[dag_id] = create_dag(config)
```

### 2.2 YAML/JSON 기반 동적 DAG

```python
# dags/yaml_driven_dag.py
import yaml
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# YAML 설정 로드
config_path = Path(__file__).parent / 'configs' / 'dag_configs.yaml'

# configs/dag_configs.yaml 예시:
"""
dags:
  - id: sales_etl
    schedule: "0 6 * * *"
    tasks:
      - name: extract
        type: python
        function: extract_sales
      - name: transform
        type: python
        function: transform_sales
      - name: load
        type: python
        function: load_sales
"""

def load_config():
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_task_callable(func_name: str):
    """함수명으로 callable 생성"""
    def task_func(**kwargs):
        print(f"Executing {func_name} for {kwargs['ds']}")
    return task_func

def create_dag_from_yaml(dag_config: dict) -> DAG:
    """YAML 설정으로 DAG 생성"""

    dag = DAG(
        dag_id=dag_config['id'],
        schedule_interval=dag_config['schedule'],
        start_date=datetime(2024, 1, 1),
        catchup=False,
    )

    with dag:
        tasks = {}
        for task_config in dag_config['tasks']:
            task = PythonOperator(
                task_id=task_config['name'],
                python_callable=create_task_callable(task_config['function']),
            )
            tasks[task_config['name']] = task

        # 순차 의존성 설정
        task_list = list(tasks.values())
        for i in range(len(task_list) - 1):
            task_list[i] >> task_list[i + 1]

    return dag

# DAG 생성 및 등록
try:
    config = load_config()
    for dag_config in config.get('dags', []):
        dag_id = dag_config['id']
        globals()[dag_id] = create_dag_from_yaml(dag_config)
except Exception as e:
    print(f"Error loading DAG config: {e}")
```

### 2.3 동적 Task 생성

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime

# 처리할 테이블 목록
TABLES = ['users', 'orders', 'products', 'reviews', 'inventory']

with DAG(
    dag_id='dynamic_tasks_example',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',
    catchup=False,
) as dag:

    start = EmptyOperator(task_id='start')
    end = EmptyOperator(task_id='end')

    # 동적으로 태스크 생성: TABLES 목록에 새 테이블을 추가하면
    # ETL 태스크가 자동으로 생성 — 오퍼레이터 보일러플레이트를 복사할 필요 없다.
    for table in TABLES:
        # 기본 인자 `table_name=table`이 루프 변수를 값으로 캡처한다;
        # 없으면 Python의 늦은 바인딩 클로저로 인해 모든 태스크가
        # 마지막 테이블을 참조하게 된다.
        def process_table(table_name=table, **kwargs):
            print(f"Processing table: {table_name}")

        task = PythonOperator(
            task_id=f'process_{table}',
            python_callable=process_table,
            op_kwargs={'table_name': table},
        )

        # 팬아웃 / 팬인: 모든 테이블 태스크가 시작/끝 사이에서 병렬로 실행된다.
        start >> task >> end
```

---

## 3. Sensor

### 이론: Sensor: 일급으로서의 대기

많은 파이프라인이 외부 이벤트를 기다립니다: "S3에 파일이 나타날 때까지 대기", "업스트림 API가 COMPLETE를 반환할 때까지 대기". Sensor는 실행 의미론이 "조건이 참이 될 때까지 폴링"인 오퍼레이터입니다.

#### B.1 poke vs reschedule 구분

Sensor는 두 가지 모드를 가집니다:

- **`poke` 모드**(기본): 워커가 슬롯을 잡고 있고, `poke_interval` 초 동안 sleep하고, 검사를 재시도, 성공이나 `timeout`까지. **그 시간 내내 워커 슬롯을 잡고 있음.** 6시간 동안 파일을 기다리는 sensor는 6시간 동안 워커 슬롯 하나를 막음.
- **`reschedule` 모드**: 검사 사이에 태스크가 *재스케줄됨* — 워커 슬롯이 해제되고, 태스크가 `up_for_reschedule`로 돌아가고, 스케줄러가 `poke_interval` 후 재큐잉. **검사 사이에 슬롯을 해제.**

장기 대기에는 항상 `reschedule` 모드를 사용하세요. 페널티: 더 많은 스케줄러 부하(인터벌당 한 번의 재큐잉), 하지만 워커 굶주림 없음.

현대 Airflow(2.2+)에서, **deferrable 오퍼레이터 / trigger** 는 세 번째 모드를 제공합니다: 태스크가 폴링을 한 프로세스로 수천 개의 대기를 잡을 수 있는 asyncio 기반 `Triggerer` 프로세스에 위임. 이는 고동시성 대기 워크로드에 옳은 선택입니다.

#### B.2 Sensor 안티패턴

- **장기 poke 모드 sensor.** 워커가 인질로 잡힘; sensor fleet이 워커 풀을 소진하면 프로덕션이 정지.
- **타임아웃 없는 sensor.** 업스트림 버그가 sensor가 영원히 대기하게 만들면, 큐가 up_for_reschedule 태스크로 채워짐. *항상 `timeout`을 설정하세요.*
- **구독 대신 폴링.** 업스트림 시스템에 webhook이나 pub/sub이 있으면 그것을 선호하세요 — sensor는 fallback이지 기본이 아님.

### 3.1 내장 Sensor

```python
from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.sensors.time_delta import TimeDeltaSensor
from airflow.providers.http.sensors.http import HttpSensor
from airflow.providers.postgres.sensors.postgres import SqlSensor
from datetime import datetime, timedelta

with DAG('sensor_examples', start_date=datetime(2024, 1, 1), schedule_interval='@daily') as dag:

    # 1. FileSensor — 직접 제어하지 않는 스케줄로 파일을 드롭하는 외부 시스템
    # (예: 벤더 SFTP 업로드)을 기다릴 때 사용한다. 파일이 나타날 때까지
    # DAG를 블록하여 "파일을 찾을 수 없음" 실패를 방지한다.
    wait_for_file = FileSensor(
        task_id='wait_for_file',
        filepath='/data/input/{{ ds }}/data.csv',
        poke_interval=60,           # 60초마다 확인 — 반응성과 I/O 부하 균형
        timeout=3600,               # 1시간 후 포기
        mode='poke',                # 워커 슬롯 점유; 짧은 예상 대기에 사용
    )

    # 2. ExternalTaskSensor — DAG를 합치지 않고 크로스-DAG 의존성을 생성한다.
    # execution_delta=0은 "같은 논리적 날짜 대기"를 의미하며,
    # 연쇄된 일별 파이프라인에서 가장 일반적인 패턴이다.
    wait_for_upstream = ExternalTaskSensor(
        task_id='wait_for_upstream',
        external_dag_id='upstream_dag',
        external_task_id='final_task',
        execution_delta=timedelta(hours=0),
        timeout=7200,
        # reschedule 모드: 확인 사이에 워커 슬롯을 해제하며, 상위 DAG가
        # 완료되는 데 몇 시간이 걸릴 수 있을 때 중요하다.
        mode='reschedule',
    )

    # 3. HttpSensor — 요청을 보내기 전에 API가 사용 가능해질 때까지
    # 기다리는 데 유용하다 (예: 배포 후).
    wait_for_api = HttpSensor(
        task_id='wait_for_api',
        http_conn_id='my_api',
        endpoint='/health',
        request_params={},
        response_check=lambda response: response.status_code == 200,
        poke_interval=30,
        timeout=600,
    )

    # 4. SqlSensor — 데이터베이스 조건을 폴링한다. Airflow를 사용하지 않는
    # 상위 배치 작업(예: 제어 테이블에 "완료" 행을 기록하는 Spark 작업)을
    # 기다리는 데 이상적이다.
    wait_for_data = SqlSensor(
        task_id='wait_for_data',
        conn_id='my_postgres',
        sql="""
            SELECT COUNT(*) > 0
            FROM staging_table
            WHERE date = '{{ ds }}'
        """,
        poke_interval=300,          # 5분 간격: DB 친화적인 폴링 속도
        timeout=3600,
    )

    # 5. TimeDeltaSensor - 시간 대기
    wait_30_minutes = TimeDeltaSensor(
        task_id='wait_30_minutes',
        delta=timedelta(minutes=30),
    )
```

### 3.2 커스텀 Sensor

```python
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults
import boto3

class S3KeySensorCustom(BaseSensorOperator):
    """S3 키 존재 확인 커스텀 Sensor"""

    # bucket_key를 템플릿화하여 날짜를 하드코딩하지 않고
    # {{ ds }}로 날짜 파티션 파일을 기다릴 수 있다.
    template_fields = ['bucket_key']

    @apply_defaults
    def __init__(
        self,
        bucket_name: str,
        bucket_key: str,
        aws_conn_id: str = 'aws_default',
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.bucket_name = bucket_name
        self.bucket_key = bucket_key
        self.aws_conn_id = aws_conn_id

    def poke(self, context) -> bool:
        """조건 확인 (True 반환 시 성공)"""
        self.log.info(f"Checking for s3://{self.bucket_name}/{self.bucket_key}")

        s3 = boto3.client('s3')

        try:
            # head_object는 list_objects보다 저렴하다: 파일 내용을 다운로드하지 않고
            # 프리픽스를 스캔하지 않아 단일 키를 메타데이터와 함께 확인한다.
            s3.head_object(Bucket=self.bucket_name, Key=self.bucket_key)
            self.log.info("File found!")
            return True
        except s3.exceptions.ClientError as e:
            # 404 = 파일이 아직 없음 → False를 반환하여 계속 대기한다.
            # 다른 오류(403 권한, 500 서버 오류)는 예상치 못한 것으로
            # 태스크를 즉시 실패시키기 위해 전파되어야 한다.
            if e.response['Error']['Code'] == '404':
                self.log.info("File not found, waiting...")
                return False
            raise

# 사용
wait_for_s3 = S3KeySensorCustom(
    task_id='wait_for_s3_file',
    bucket_name='my-bucket',
    bucket_key='data/{{ ds }}/input.parquet',
    poke_interval=60,
    timeout=3600,
    mode='reschedule',
)
```

### 3.3 Sensor 모드

```python
# poke vs reschedule 모드 비교
sensor_modes = {
    'poke': {
        'description': '워커 슬롯을 점유하고 대기',
        'pros': '빠른 반응 시간',
        'cons': '워커 리소스 낭비',
        'use_case': '짧은 대기 시간 예상'
    },
    'reschedule': {
        'description': '워커 반환 후 재스케줄',
        'pros': '워커 리소스 효율적 사용',
        'cons': '다소 느린 반응 시간',
        'use_case': '긴 대기 시간 예상'
    }
}

# 긴 대기 센서의 권장 프로덕션 설정:
wait_for_file = FileSensor(
    task_id='wait_for_file',
    filepath='/data/input.csv',
    poke_interval=300,      # 5분: 합리적인 지연 시간에 충분히 짧고,
                            # 재시도로 스케줄러에 부하를 주지 않을 만큼 충분히 길다
    timeout=86400,          # 24시간: 일별 파일에 넉넉한 타임아웃
    mode='reschedule',      # poke 사이에 워커 슬롯을 해제한다 — 워커 용량이
                            # 제한적일 때 필수적이다
    soft_fail=True,         # 타임아웃 시 "실패" 대신 "스킵"으로 표시하여
                            # 하위 태스크가 trigger_rule로 처리 방법을 결정할 수 있게 한다
)
```

---

## 4. Hook과 Connection

### 4.1 Connection 설정

```python
# Airflow UI 또는 CLI로 Connection 설정
# Admin > Connections > Add

# CLI로 Connection 추가
"""
airflow connections add 'my_postgres' \
    --conn-type 'postgres' \
    --conn-host 'localhost' \
    --conn-port '5432' \
    --conn-login 'user' \
    --conn-password 'password' \
    --conn-schema 'mydb'

airflow connections add 'my_s3' \
    --conn-type 'aws' \
    --conn-extra '{"aws_access_key_id": "xxx", "aws_secret_access_key": "yyy", "region_name": "us-east-1"}'
"""

# 환경 변수로 Connection 설정
# AIRFLOW_CONN_MY_POSTGRES='postgresql://user:password@localhost:5432/mydb'
```

### 4.2 Hook 사용

```python
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.http.hooks.http import HttpHook

def use_postgres_hook(**kwargs):
    """PostgreSQL Hook 사용"""
    hook = PostgresHook(postgres_conn_id='my_postgres')

    # SQL 실행
    records = hook.get_records("SELECT * FROM users LIMIT 10")

    # DataFrame으로 반환
    df = hook.get_pandas_df("SELECT * FROM users")

    # 삽입
    hook.insert_rows(
        table='users',
        rows=[(1, 'John'), (2, 'Jane')],
        target_fields=['id', 'name']
    )

    # 직접 연결 사용
    conn = hook.get_conn()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET active = true")
    conn.commit()

def use_s3_hook(**kwargs):
    """S3 Hook 사용"""
    hook = S3Hook(aws_conn_id='my_s3')

    # 파일 업로드
    hook.load_file(
        filename='/tmp/data.csv',
        key='data/output.csv',
        bucket_name='my-bucket',
        replace=True
    )

    # 파일 다운로드
    hook.download_file(
        key='data/input.csv',
        bucket_name='my-bucket',
        local_path='/tmp/input.csv'
    )

    # 파일 목록 조회
    keys = hook.list_keys(
        bucket_name='my-bucket',
        prefix='data/',
        delimiter='/'
    )

def use_http_hook(**kwargs):
    """HTTP Hook 사용"""
    hook = HttpHook(http_conn_id='my_api', method='GET')

    response = hook.run(
        endpoint='/api/data',
        headers={'Authorization': 'Bearer token'},
        data={'param': 'value'}
    )

    return response.json()
```

### 4.3 커스텀 Hook

```python
from airflow.hooks.base import BaseHook
from typing import Any
import requests

class MyCustomHook(BaseHook):
    """커스텀 API Hook"""

    # 이 클래스 속성들은 Airflow의 커넥션 UI가 필드를 자동으로 채우고
    # 커넥션 타입을 검증할 수 있게 한다 — 없으면 사용자가 정확한
    # conn_id 형식을 직접 기억해야 한다.
    conn_name_attr = 'my_custom_conn_id'
    default_conn_name = 'my_custom_default'
    conn_type = 'http'
    hook_name = 'My Custom Hook'

    def __init__(self, my_custom_conn_id: str = default_conn_name):
        super().__init__()
        self.my_custom_conn_id = my_custom_conn_id
        self.base_url = None
        self.api_key = None

    def get_conn(self):
        """Connection 설정 로드"""
        # 자격 증명은 DAG 코드가 아닌 Airflow의 암호화된 커넥션 저장소에 보관된다 —
        # 시크릿을 버전 관리에서 제외하기 위함이다.
        conn = self.get_connection(self.my_custom_conn_id)
        self.base_url = f"https://{conn.host}"
        self.api_key = conn.password
        return conn

    def make_request(self, endpoint: str, method: str = 'GET', data: dict = None) -> Any:
        """API 요청"""
        # 지연 커넥션(lazy connection): 자격 증명은 훅 인스턴스화 시점이 아닌
        # 요청이 실제로 만들어질 때만 로드된다 — 훅이 생성되었지만 사용되지
        # 않는 경우(예: 스킵된 분기)의 불필요한 DB 조회를 방지한다.
        self.get_conn()

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        url = f"{self.base_url}{endpoint}"

        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=data
        )

        # raise_for_status()는 HTTP 4xx/5xx를 Python 예외로 변환하며,
        # Airflow가 이를 잡아 태스크의 재시도 정책에 따라 재시도한다.
        response.raise_for_status()
        return response.json()

# 사용
def call_custom_api(**kwargs):
    hook = MyCustomHook(my_custom_conn_id='my_api')
    result = hook.make_request('/users', method='GET')
    return result
```

---

## 5. TaskGroup

### 이론: TaskGroup vs SubDAG

50개 태스크 DAG는 읽기 어렵습니다. 시각적으로 압축하는 두 가지 방법:

#### D.1 SubDAG (deprecated)

본문이 자체 DAG인 태스크. 한 노드처럼 보이고, 클릭하면 서브 그래프로 확장. 문제:

- 각 SubDAG가 자체 스케줄러 오버헤드, 메타데이터 행 등을 가진 실제 DAG 실행.
- 부모와의 동시성 상호작용이 미묘 — SubDAG 슬롯과 부모 슬롯이 데드락 가능.
- TaskGroup을 위해 Airflow 3에서 제거됨.

#### D.2 TaskGroup (현재)

순수 시각적 그룹화. TaskGroup 내부의 태스크는 같은 DAG에 남아 있음; 그룹은 단지 UI 렌더링 힌트. 추가 DAG 실행 없음, 추가 메타데이터 없음, 동시성 깜짝 놀라움 없음.

```python
with TaskGroup("ingest") as ingest:
    extract_users = ...
    extract_orders = ...
    extract_products = ...
```

UI는 "ingest"를 접을 수 있는 클러스터로 보여줍니다. 클릭해서 확장. 단계별(ingest, transform, load)이나 데이터 도메인별로 그룹화하기에 좋음.

### 5.1 TaskGroup 기본 사용

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime

with DAG('taskgroup_example', start_date=datetime(2024, 1, 1), schedule_interval='@daily') as dag:

    start = EmptyOperator(task_id='start')

    # TaskGroup은 관련 태스크를 UI에서 시각적으로 접어서 복잡한 DAG를
    # 탐색하기 쉽게 만든다. 또한 개별 태스크가 아닌 *그룹* 단위로
    # 의존성을 설정할 수 있게 한다 (extract_group >> transform_group).
    with TaskGroup(group_id='extract_group') as extract_group:
        extract_users = PythonOperator(
            task_id='extract_users',
            python_callable=lambda: print("Extracting users")
        )
        extract_orders = PythonOperator(
            task_id='extract_orders',
            python_callable=lambda: print("Extracting orders")
        )
        extract_products = PythonOperator(
            task_id='extract_products',
            python_callable=lambda: print("Extracting products")
        )

    with TaskGroup(group_id='transform_group') as transform_group:
        transform_users = PythonOperator(
            task_id='transform_users',
            python_callable=lambda: print("Transforming users")
        )
        transform_orders = PythonOperator(
            task_id='transform_orders',
            python_callable=lambda: print("Transforming orders")
        )

    with TaskGroup(group_id='load_group') as load_group:
        load_warehouse = PythonOperator(
            task_id='load_warehouse',
            python_callable=lambda: print("Loading to warehouse")
        )

    end = EmptyOperator(task_id='end')

    # TaskGroup 간 의존성
    start >> extract_group >> transform_group >> load_group >> end
```

### 5.2 중첩 TaskGroup

```python
from airflow.utils.task_group import TaskGroup

with DAG('nested_taskgroup', ...) as dag:

    with TaskGroup(group_id='data_processing') as data_processing:

        with TaskGroup(group_id='source_a') as source_a:
            extract_a = PythonOperator(task_id='extract', ...)
            transform_a = PythonOperator(task_id='transform', ...)
            extract_a >> transform_a

        with TaskGroup(group_id='source_b') as source_b:
            extract_b = PythonOperator(task_id='extract', ...)
            transform_b = PythonOperator(task_id='transform', ...)
            extract_b >> transform_b

        # 병렬 실행 후 조인
        join = EmptyOperator(task_id='join')
        [source_a, source_b] >> join
```

### 5.3 동적 TaskGroup

```python
from airflow.utils.task_group import TaskGroup

# 새 소스를 추가하려면 이 목록에 추가하기만 하면 된다 — 아래 루프가
# extract→load 파이프라인을 자동으로 생성한다.
SOURCES = ['mysql', 'postgres', 'mongodb']

with DAG('dynamic_taskgroup', ...) as dag:

    start = EmptyOperator(task_id='start')

    task_groups = []
    for source in SOURCES:
        # 각 소스는 자체 TaskGroup을 갖는다: 장애를 격리하고
        # (MongoDB 오류가 MySQL 파이프라인을 막지 않음) UI에서
        # 소스별 진행 상황을 한눈에 볼 수 있다.
        with TaskGroup(group_id=f'process_{source}') as tg:
            extract = PythonOperator(
                task_id='extract',
                # 기본 인자 `s=source`는 루프 변수를 값으로 캡처한다
                python_callable=lambda s=source: print(f"Extract from {s}")
            )
            load = PythonOperator(
                task_id='load',
                python_callable=lambda s=source: print(f"Load {s}")
            )
            extract >> load

        task_groups.append(tg)

    end = EmptyOperator(task_id='end')

    # 모든 소스 그룹이 start와 end 사이에서 병렬로 실행된다
    start >> task_groups >> end
```

---

## 6. 분기 처리와 조건부 실행

### 6.1 BranchPythonOperator

```python
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator

def choose_branch(**kwargs):
    """조건에 따라 다음 Task 선택"""
    ti = kwargs['ti']
    data_count = ti.xcom_pull(task_ids='count_data')

    if data_count > 1000:
        return 'process_large'
    elif data_count > 0:
        return 'process_small'
    else:
        return 'skip_processing'

with DAG('branch_example', ...) as dag:

    count_data = PythonOperator(
        task_id='count_data',
        python_callable=lambda: 500,  # 예시 반환값
    )

    branch = BranchPythonOperator(
        task_id='branch',
        python_callable=choose_branch,
    )

    process_large = EmptyOperator(task_id='process_large')
    process_small = EmptyOperator(task_id='process_small')
    skip_processing = EmptyOperator(task_id='skip_processing')

    # 분기 후 합류: 선택되지 않은 분기가 "스킵"으로 표시되기 때문에
    # trigger_rule이 여기서 필수적이다. 기본 'all_success' 규칙은
    # 스킵을 성공이 아닌 것으로 취급한다. 'none_failed_min_one_success'는
    # 선택된 분기가 성공하는 한 합류 태스크가 실행되도록 허용한다.
    join = EmptyOperator(
        task_id='join',
        trigger_rule='none_failed_min_one_success'
    )

    count_data >> branch >> [process_large, process_small, skip_processing] >> join
```

### 6.2 ShortCircuitOperator

```python
from airflow.operators.python import ShortCircuitOperator

def check_condition(**kwargs):
    """조건 확인 - False 반환 시 이후 Task 스킵"""
    ds = kwargs['ds']
    # ShortCircuit vs Branch: "실행할까?" 라는 단일 게이트(예/아니오)가
    # 있을 때 ShortCircuit을 사용하고, 여러 대안 경로 중 하나를 선택해야
    # 할 때 Branch를 사용한다.
    day_of_week = datetime.strptime(ds, '%Y-%m-%d').weekday()
    return day_of_week < 5  # True → 계속 실행; False → 모든 하위 태스크 스킵

with DAG('shortcircuit_example', ...) as dag:

    check = ShortCircuitOperator(
        task_id='check_weekday',
        python_callable=check_condition,
    )

    # check가 False 반환 시 아래 Task들은 스킵됨
    process = PythonOperator(task_id='process', ...)
    load = PythonOperator(task_id='load', ...)

    check >> process >> load
```

---

## 연습 문제

### 문제 1: XCom 활용
두 개의 Task에서 각각 숫자를 반환하고, 세 번째 Task에서 두 숫자의 합을 계산하는 DAG를 작성하세요.

### 문제 2: 동적 DAG
테이블 목록(users, orders, products)을 기반으로 각 테이블에 대한 ETL Task를 동적으로 생성하는 DAG를 작성하세요.

### 문제 3: Sensor 사용
파일이 생성될 때까지 대기한 후 처리하는 DAG를 작성하세요.

---

## 요약

| 기능 | 설명 |
|------|------|
| **XCom** | Task 간 데이터 공유 메커니즘 |
| **동적 DAG** | 설정 기반으로 DAG/Task 동적 생성 |
| **Sensor** | 조건 충족까지 대기하는 Operator |
| **Hook** | 외부 시스템 연결 인터페이스 |
| **TaskGroup** | 관련 Task를 그룹화하여 시각화 |
| **Branch** | 조건에 따른 분기 처리 |

---

## 참고 자료

- [Airflow XCom Guide](https://airflow.apache.org/docs/apache-airflow/stable/concepts/xcoms.html)
- [Dynamic DAGs](https://airflow.apache.org/docs/apache-airflow/stable/howto/dynamic-dag-generation.html)
- [Airflow Sensors](https://airflow.apache.org/docs/apache-airflow/stable/concepts/sensors.html)

---

[← 이전: 4. Apache Airflow 기초](04_Apache_Airflow_Basics.md) | [다음: 6. Prefect 모던 오케스트레이션 →](06_Prefect_Modern_Orchestration.md)
