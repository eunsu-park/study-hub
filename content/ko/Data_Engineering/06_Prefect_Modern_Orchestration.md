[← 이전: 5. Airflow 심화](05_Airflow_Advanced.md) | [다음: 7. Apache Spark 기초 →](07_Apache_Spark_Basics.md)

# Prefect 모던 오케스트레이션

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 워크플로우 정의 방식, 스케줄링 모델, 동적 워크플로우 지원 등 핵심 측면에서 Prefect와 Airflow를 비교할 수 있다
2. Python 데코레이터를 사용하여 Prefect 플로우(Flow)와 태스크(Task)를 정의하고 의존성을 설정할 수 있다
3. Prefect 플로우에서 캐싱(Caching), 재시도(Retry), 동시성 제어(Concurrency Control)를 구현할 수 있다
4. 워크 풀(Work Pool)을 사용하여 Prefect 플로우를 배포하고 Prefect Cloud UI 또는 API를 통해 스케줄을 설정할 수 있다
5. 블록(Block)을 사용하여 Prefect를 외부 시스템과 통합하고, 파라미터화된 재사용 가능한 플로우를 구축할 수 있다
6. 플로우 실행 상태를 모니터링하고 프로덕션 파이프라인을 위한 알림(Notification)과 자동화(Automation)를 설정할 수 있다

---

## 개요

Prefect는 현대적인 워크플로우 오케스트레이션 도구로, Python 네이티브 방식으로 데이터 파이프라인을 구축합니다. Airflow와 비교하여 더 간단한 설정과 동적 워크플로우를 지원합니다.

---

## 1. Prefect 개요

### 1.1 Prefect vs Airflow

```
┌────────────────────────────────────────────────────────────────┐
│                   Prefect vs Airflow 비교                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Airflow:                    Prefect:                          │
│  ┌──────────────┐           ┌──────────────┐                  │
│  │ DAG (Static) │           │ Flow (Dynamic)│                  │
│  │              │           │               │                  │
│  │ - 정적 정의  │           │ - 동적 생성   │                  │
│  │ - 파일 기반  │           │ - Python 코드 │                  │
│  │ - Scheduler  │           │ - 이벤트 기반 │                  │
│  └──────────────┘           └──────────────┘                  │
│                                                                │
│  실행 모델:                  실행 모델:                         │
│  Scheduler → Worker         Trigger → Work Pool → Worker       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

| 특성 | Airflow | Prefect |
|------|---------|---------|
| **정의 방식** | DAG 파일 | Python 데코레이터 |
| **스케줄링** | Scheduler 프로세스 | 이벤트 기반, 서버리스 |
| **동적 워크플로우** | 제한적 | 네이티브 지원 |
| **로컬 실행** | 복잡한 설정 | 즉시 가능 |
| **상태 관리** | DB 필수 | 선택적 |
| **학습 곡선** | 가파름 | 완만함 |

### 1.2 Prefect 아키텍처

```
┌────────────────────────────────────────────────────────────────┐
│                    Prefect Architecture                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   ┌─────────────────────────────────────────────┐             │
│   │              Prefect Cloud / Server         │             │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐       │             │
│   │  │  UI     │ │  API    │ │ Automations    │             │
│   │  └─────────┘ └─────────┘ └─────────┘       │             │
│   └─────────────────────────────────────────────┘             │
│                          ↑ ↓                                   │
│   ┌─────────────────────────────────────────────┐             │
│   │               Work Pools                     │             │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐       │             │
│   │  │ Process │ │ Docker  │ │  K8s    │       │             │
│   │  └─────────┘ └─────────┘ └─────────┘       │             │
│   └─────────────────────────────────────────────┘             │
│                          ↑ ↓                                   │
│   ┌─────────────────────────────────────────────┐             │
│   │               Workers                        │             │
│   │         (Flow 실행 에이전트)                  │             │
│   └─────────────────────────────────────────────┘             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. 설치 및 시작하기

### 2.1 설치

```bash
# 기본 설치
pip install prefect

# 추가 통합 설치
pip install "prefect[aws]"      # AWS 통합
pip install "prefect[gcp]"      # GCP 통합
pip install "prefect[dask]"     # Dask 통합

# 버전 확인
prefect version
```

### 2.2 Prefect Cloud 연결 (선택사항)

```bash
# Prefect Cloud 로그인
prefect cloud login

# 또는 API 키 사용
prefect cloud login --key YOUR_API_KEY

# Self-hosted 서버 연결
prefect config set PREFECT_API_URL="http://localhost:4200/api"
```

### 2.3 로컬 서버 실행

```bash
# Prefect 서버 시작 (UI 포함)
prefect server start

# UI 접속: http://localhost:4200
```

---

## 3. Flow와 Task 기본

### 이론: 명령적 오케스트레이션

Airflow에서 DAG 파일은 몇 초마다 파싱되는 *명세* 입니다; 태스크는 기호적으로 참조되고 의존성 그래프는 어떤 태스크가 실행되기 전에 스케줄러에 의해 구체화됩니다. Prefect에서 flow는 *Python 함수* 입니다 — 호출하면 실제로 실행되고, 그 과정에서 API에 상태를 기록합니다.

#### A.1 DAG가 함수일 때 무엇이 바뀌는가

```python
@flow
def daily_pipeline(date: str):
    raw = extract(date)
    if len(raw) > 0:
        clean = transform(raw)
        load(clean)
    else:
        log("nothing to do")
```

이 코드를 보세요: 평범한 Python입니다. `if`, 함수 호출, 조건부 분기 — 모두 예상대로 작동. "DAG"는 런타임 호출 그래프가 결국 무엇이 되든 그것입니다.

이는 의미합니다:

- **조건부 실행이 자명함.** `BranchPythonOperator` 의례 없음.
- **루프가 자명함.** `for partition in partitions: process(partition)`이 직접 작동.
- **테스트가 자명함.** 단위 테스트에서 flow 함수 호출; 실행됨.
- **타입 검사가 작동.** mypy가 `extract`이 DataFrame을 반환함을 이해.

트레이드오프: 실행 *전에* 검사할 정적 그래프가 없습니다. UI는 런타임에 구성된 대로 그래프를 보여줍니다.

#### A.2 무엇을 포기하는가

- **사전 비행 정적 검증.** Airflow는 파싱 시점에 "태스크 X가 누락된 태스크 Y에 의존한다"라고 알려줄 수 있음. Prefect는 안 됨 — 그래프 오류를 발견하려면 실제로 실행해야 함.
- **장거리 스케줄링 보장.** Airflow 스케줄러는 다음 N개 실행에 대해 추론. Prefect는 각 실행을 독립적으로 스케줄.
- **확립된 오퍼레이터 생태계.** Airflow는 1000+개의 사전 빌드된 오퍼레이터(모든 클라우드, DB, API). Prefect의 컬렉션은 성장 중이지만 더 작음.

옳은 멘탈 모델: Prefect는 파이프라인 코드가 *나머지 Python 애플리케이션처럼 보이기를* 원하는 팀을 위한 것; Airflow는 중앙화된 선언적 오케스트레이션 계층을 원하는 팀을 위한 것.

### 이론: 일급 개념으로서의 상태

모든 Prefect 태스크와 flow는 **상태** 를 가집니다 — `Pending`, `Running`, `Completed`, `Failed`, `Crashed`, `Cancelled`, `Retrying`, `Paused`. 상태 전이는 API를 통해 기록되고 UI를 구동합니다.

#### C.1 프로그래밍 가능한 신호로서의 상태

상태가 대부분 관찰되는 Airflow와 달리, Prefect는 전이를 가로채고 반응할 수 있게 합니다:

```python
def alert_on_failure(flow, flow_run, state):
    if state.is_failed():
        send_slack(...)

@flow(on_failure=[alert_on_failure])
def my_flow(): ...
```

오케스트레이션에 영향을 주기 위해 상태를 명시적으로 반환할 수도 있습니다: `return Completed(message="nothing to do")`은 다운스트림을 우아하게 스킵.

#### C.2 Crashed vs Failed

미묘하지만 중요: `Failed`는 태스크가 실행되어 raise되었음; `Crashed`는 워커가 죽었음(OOM, 네트워크 끊김). 이 구분은 재시도 로직에 중요 — `Failed` 태스크는 `Crashed` 태스크와 다른 처리를 필요로 할 가능성.

#### C.3 재시도는 동작이 아닌 정책

`@task(retries=3, retry_delay_seconds=10)`은 재시도 정책을 선언. 런타임은 `Failed`에서 자동으로 `Retrying`으로 전이하고 태스크를 재실행. Airflow와 같은 멱등성 계약: 태스크는 재실행 안전해야 함.

### 3.1 기본 Flow 작성

```python
from prefect import flow, task
from prefect.logging import get_run_logger

# @task 데코레이터는 함수를 추적 가능한 단위로 변환한다 — Prefect가 상태
# (Pending → Running → Completed/Failed), 로그, 반환 값을 자동으로 기록한다.
# Airflow 오퍼레이터와 달리, 태스크는 보일러플레이트 없는 순수한 Python 함수다.
@task
def extract_data(source: str) -> dict:
    """데이터 추출 Task"""
    # get_run_logger()는 현재 태스크 실행에 스코프된 로거를 반환하므로, 로그 메시지에
    # 태스크 이름/실행 ID가 자동으로 포함된다 — 병렬 플로우 디버깅 시 필수적이다
    logger = get_run_logger()
    logger.info(f"Extracting from {source}")

    # 실제로는 DB, API 등에서 추출
    data = {"source": source, "records": [1, 2, 3, 4, 5]}
    return data

@task
def transform_data(data: dict) -> dict:
    """데이터 변환 Task"""
    logger = get_run_logger()
    logger.info(f"Transforming {len(data['records'])} records")

    # 변환 로직
    data["records"] = [x * 2 for x in data["records"]]
    data["transformed"] = True
    return data

@task
def load_data(data: dict, destination: str) -> bool:
    """데이터 적재 Task"""
    logger = get_run_logger()
    logger.info(f"Loading to {destination}")

    # 실제로는 DB, 파일 등에 저장
    print(f"Loaded data: {data}")
    return True

# @flow는 오케스트레이션 로직을 감싼다. Prefect는 함수 호출 간의 데이터 흐름에서
# 태스크 의존성을 추론한다 — Airflow의 >> 연산자처럼 명시적으로 연결할 필요가 없다.
# 기본 파라미터를 지정하면 프로그래밍 방식으로도, 배포를 통해서도 플로우를 호출할 수 있다.
@flow(name="ETL Pipeline")
def etl_pipeline(source: str = "database", destination: str = "warehouse"):
    """ETL 파이프라인 Flow"""
    # Prefect가 데이터 의존성을 자동으로 추적한다: transform은 extract 완료 후,
    # load는 transform 완료 후 실행된다. 명시적인 의존성 연결이 필요 없다.
    raw_data = extract_data(source)
    transformed = transform_data(raw_data)
    result = load_data(transformed, destination)
    return result

# 플로우는 일반 Python 함수다 — 스케줄러, 데이터베이스, 웹 서버 없이
# 로컬에서 직접 실행하여 테스트할 수 있다 (Airflow의 전체 스택과 달리)
if __name__ == "__main__":
    etl_pipeline()
```

### 3.2 Task 옵션

```python
from prefect import task
from datetime import timedelta

@task(
    name="My Task",
    description="Task 설명",
    tags=["etl", "production"],         # 태그는 Prefect UI 필터링 및 자동화 트리거에 활용된다
    retries=3,                          # 자동 재시도로 일시적 장애(네트워크 타임아웃, API 속도 제한) 처리
    retry_delay_seconds=60,             # 고정 60초 대기로 다운스트림 서비스가 복구할 시간을 준다
    timeout_seconds=3600,               # 1시간 타임아웃으로 좀비 태스크가 리소스를 무기한 점유하는 것을 방지
    cache_key_fn=lambda: "static_key",  # 정적 키는 입력에 상관없이 동일한 결과를 의미 — 멱등적 조회에 사용
    cache_expiration=timedelta(hours=1), # 1시간 만료로 신선도와 불필요한 API 호출 방지 간의 균형을 맞춘다
    log_prints=True,                    # print()를 Prefect 로그로 리다이렉트 — print를 사용하는 레거시 코드에 유용
)
def my_task(param: str) -> str:
    print(f"Processing: {param}")
    return f"Result: {param}"

# 지수 백오프(exponential backoff)는 복구 중인 서비스에 대한 재시도 폭주를 방지한다 —
# 각 재시도는 점점 더 오래 대기(10초, 20초, 40초, 80초, 160초)하여 부하를 줄인다
from prefect.tasks import exponential_backoff

@task(
    retries=5,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
)
def flaky_task():
    """불안정한 외부 API 호출"""
    import random
    if random.random() < 0.7:
        raise Exception("Random failure")
    return "Success"
```

### 3.3 Flow 옵션

```python
from prefect import flow
from prefect.task_runners import ConcurrentTaskRunner, SequentialTaskRunner

@flow(
    name="My Flow",
    description="Flow 설명",
    version="1.0.0",
    retries=2,                           # 플로우 수준 재시도는 전체 플로우를 재실행한다 (개별 태스크가 아님)
    retry_delay_seconds=300,             # 5분 대기로 의존 시스템이 안정될 시간을 준다
    timeout_seconds=7200,                # 2시간 하드 제한으로 런어웨이 파이프라인이 워크 풀을 차단하는 것을 방지
    task_runner=ConcurrentTaskRunner(),  # 스레드로 태스크 실행 — I/O 바운드 태스크(API 호출, DB 쿼리)에 사용
    log_prints=True,
    persist_result=True,                 # 플로우 반환 값을 스토리지에 저장 — 플로우-오브-플로우 패턴에 필요
)
def my_flow():
    pass

# SequentialTaskRunner는 한 번에 하나씩 실행을 강제한다 — 태스크가 공유 상태를
# 가질 때 사용 (예: 동일 파일에 쓰기)하여 동시성으로 인한 손상을 방지한다
@flow(task_runner=SequentialTaskRunner())
def sequential_flow():
    pass
```

---

## 4. 동적 워크플로우

### 4.1 동적 Task 생성

```python
from prefect import flow, task

@task
def process_item(item: str) -> str:
    return f"Processed: {item}"

# Airflow DAG에서는 태스크 수가 파싱 시점에 정의되어야 하지만, Prefect는
# 런타임에 태스크를 생성한다 — 루프 본문이 정적 그래프가 아닌 실제 Python이다.
@flow
def dynamic_tasks_flow(items: list[str]):
    """동적으로 Task 수 결정"""
    results = []
    for item in items:
        # 각 호출은 별도의 태스크 실행을 생성한다 — Prefect가 독립적으로 추적한다
        result = process_item(item)
        results.append(result)
    return results

# 실행
dynamic_tasks_flow(["a", "b", "c", "d"])

# .submit()은 플로우의 task_runner를 통해 동시 실행을 가능하게 한다.
# I/O 바운드 워크로드(API 호출, 파일 다운로드)에서 순차 대기가 시간을 낭비하는
# 경우에 사용한다.
@flow
def parallel_tasks_flow(items: list[str]):
    """병렬로 Task 실행"""
    futures = []
    for item in items:
        # .submit()은 직접 호출과 달리 결과를 기다리지 않고 즉시 PrefectFuture를 반환한다
        future = process_item.submit(item)
        futures.append(future)

    # .result()는 퓨처가 완료될 때까지 블로킹한다 — 모두 제출한 후 결과를
    # 수집하여 병렬성을 최대화한다
    results = [f.result() for f in futures]
    return results
```

### 4.2 조건부 실행

```python
from prefect import flow, task

@task
def check_condition(data: dict) -> bool:
    return data.get("count", 0) > 100

@task
def process_large(data: dict):
    print(f"Processing large dataset: {data['count']} records")

@task
def process_small(data: dict):
    print(f"Processing small dataset: {data['count']} records")

@flow
def conditional_flow(data: dict):
    """조건에 따른 분기"""
    is_large = check_condition(data)

    if is_large:
        process_large(data)
    else:
        process_small(data)

# 실행
conditional_flow({"count": 150})  # process_large 실행
conditional_flow({"count": 50})   # process_small 실행
```

### 4.3 서브플로우

```python
from prefect import flow, task

@task
def extract(source: str) -> list:
    return [1, 2, 3, 4, 5]

@task
def transform(data: list) -> list:
    return [x * 2 for x in data]

@task
def load(data: list, target: str):
    print(f"Loading {len(data)} records to {target}")

# 서브플로우는 다른 플로우에서 호출되는 플로우다. 각 서브플로우는 독립적인 상태
# 추적, 재시도, 로그를 갖는 자체 플로우 실행을 가져 — 구성 가능하고 재사용 가능하게 만든다.
@flow(name="ETL Subflow")
def etl_subflow(source: str, target: str):
    """재사용 가능한 ETL 서브플로우"""
    data = extract(source)
    transformed = transform(data)
    load(transformed, target)
    return len(transformed)

# 부모 플로우는 서브플로우를 함수 호출처럼 오케스트레이션한다. 서브플로우가 실패하면
# 해당 서브플로우만 재시도된다 — 이미 완료된 서브플로우는 재실행되지 않는다.
@flow(name="Main Pipeline")
def main_pipeline():
    """여러 서브플로우 오케스트레이션"""
    # 각 호출은 UI 계층 구조에서 볼 수 있는 중첩 플로우 실행을 생성한다.
    # 병렬 서브플로우 실행을 위해서는 ConcurrentTaskRunner와 함께 .submit()을 사용한다.
    count_a = etl_subflow("source_a", "target_a")
    count_b = etl_subflow("source_b", "target_b")
    count_c = etl_subflow("source_c", "target_c")

    print(f"Total processed: {count_a + count_b + count_c}")

main_pipeline()
```

---

## 5. 배포 (Deployment)

### 이론: 하이브리드 실행 모델

Prefect의 가장 아키텍처적으로 중요한 선택: **컨트롤 플레인(오케스트레이션 API)과 실행 플레인(코드가 실행되는 곳)이 별개의 프로세스이며, 종종 별개의 네트워크에 있음.**

```
┌──────────────────┐                ┌──────────────────────┐
│ Prefect Cloud    │ ←── HTTPS ───  │  워커 (당신의 인프라)│
│ (오케스트레이션) │   상태 이벤트   │   - flow run pull    │
│ - 스케줄         │                │   - 실행             │
│ - 상태 관리      │   ingress 없음  │   - 상태 보고        │
│ - UI / API       │                │                      │
└──────────────────┘                └──────────────────────┘
```

#### B.1 왜 이것이 중요한가

Airflow의 아키텍처는 스케줄러가 워커에 접근할 수 있다고 가정. 기업 환경에서는 종종 VPN, 방화벽 구멍, 또는 스케줄러를 기업 네트워크 내에 두는 것을 의미합니다. Prefect는 관계를 뒤집습니다: 워커가 API에 *밖으로* 접근. 인바운드 방화벽 구멍 없음; 오케스트레이션 UI는 클라우드에 살 수 있고 flow 코드는 온프렘에서 실행될 수 있습니다.

#### B.2 코드 지역성

결정적으로, **Prefect Cloud는 당신의 데이터를 절대 보지 않고 당신의 코드를 절대 실행하지 않습니다**. 워커는 당신이 통제하는 인프라에서 코드를 실행; 상태 이벤트(시작됨, 완료됨, 실패함)만 API로 다시 흐름. 이것은 실제 컴플라이언스 이점: 데이터가 고객 네트워크를 절대 떠나지 않음.

#### B.3 트레이드오프

대가: 어딘가에서 워커를 실행해야 함. Prefect Cloud는 "내가 코드만 푸시하는 완전 관리형 오케스트레이션"이 아니라 "내가 워커를 실행하는 관리형 컨트롤 플레인"입니다. Airflow와 비교하세요, MWAA / Composer도 워커 풀을 제공하라고 요구합니다. 둘 다 같은 근본적 제약; Prefect가 그것을 명시적으로 만듭니다.

### 이론: 워크 풀과 워커

Prefect 2.x는 워크 풀을 도입했습니다 — "이 flow가 어디서 실행되는가"를 표현하는 현대적 방법.

#### D.1 모델

- **deployment** 는 "이 flow, 이 스케줄에, 이 매개변수로"를 말합니다.
- **워크 풀(work pool)** 은 deployment가 실행을 푸시하는 큐(인프라 설정 포함)입니다.
- **워커** 는 특정 워크 풀을 폴링하고 실행합니다.

이는 Airflow의 익스큐터 + 큐 개념과 비슷하지만 더 명시적. "작은 Python 태스크"를 위해 process 워커가 받치는 워크 풀 하나, "Spark 잡"을 위해 Kubernetes 워커가 받치는 다른 풀, "AWS batch"를 위해 ECS 워커가 받치는 또 다른 풀을 가질 수 있습니다.

#### D.2 코드와 인프라의 분리

`kubernetes_job.with_options(image="my-spark:1.0").submit(work)`라고 말하는 flow는 인프라를 인라인으로 선언; 워커는 단지 pull하고 디스패치. flow 코드가 아닌 deployment의 워크 풀을 변경하여 flow를 인프라 간에 옮길 수 있습니다.

#### D.3 Pull 모델 함의

워커가 워크 풀에서 pull; API가 push하지 않음. 풀을 폴링하는 워커가 없으면 실행이 큐잉되지만 실행되지 않음. 누락된/죽은 워커의 첫 번째 신호는 "스케줄된 실행이 시작되지 않음"입니다. 모니터링은 항상 "각 활성 워크 풀에 최소 하나의 워커가 폴링하고 있는가?"를 포함해야 합니다.

### 5.1 Deployment 생성

```python
from prefect import flow
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

@flow
def my_etl_flow(date: str = None):
    """일일 ETL 플로우"""
    from datetime import datetime
    date = date or datetime.now().strftime("%Y-%m-%d")
    print(f"Running ETL for {date}")

# 배포는 플로우 정의와 실행 스케줄링을 분리한다 — 동일한 플로우가 서로 다른
# 스케줄, 파라미터, 인프라를 가진 여러 배포를 가질 수 있다.
deployment = Deployment.build_from_flow(
    flow=my_etl_flow,
    name="daily-etl",
    version="1.0",
    tags=["production", "etl"],           # 태그는 Prefect Cloud에서 필터링 및 RBAC를 가능하게 한다
    schedule=CronSchedule(cron="0 6 * * *"),  # 오전 6시 실행으로 업무 시간 전에 데이터가 준비된다
    parameters={"date": None},            # None은 일일 실행 시 datetime.now() 폴백을 트리거한다
    work_pool_name="default-agent-pool",  # 올바른 인프라로 실행을 라우팅한다
)

# apply()는 배포를 Prefect 서버/클라우드에 등록한다 — 워커가
# 예약된 실행을 폴링하여 지정된 워크 풀에서 실행한다
deployment.apply()
```

### 5.2 CLI로 Deployment 생성

```bash
# prefect.yaml 생성
prefect init

# Deployment 빌드 및 적용
prefect deploy --name daily-etl
```

```yaml
# prefect.yaml 예시
name: my-project
prefect-version: 2.14.0

deployments:
  - name: daily-etl
    entrypoint: flows/etl.py:my_etl_flow
    work_pool:
      name: default-agent-pool
    schedule:
      cron: "0 6 * * *"
    parameters:
      date: null
    tags:
      - production
      - etl
```

### 5.3 Work Pool 및 Worker

```bash
# Work Pool 생성
prefect work-pool create my-pool --type process

# Worker 시작
prefect worker start --pool my-pool

# Docker 기반 Work Pool
prefect work-pool create docker-pool --type docker

# Kubernetes 기반 Work Pool
prefect work-pool create k8s-pool --type kubernetes
```

---

## 6. Airflow와의 비교 예제

### 6.1 Airflow 버전

```python
# Airflow DAG
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Airflow 태스크는 XCom(크로스 커뮤니케이션)을 통해 데이터를 전달한다 — 메타데이터
# DB에 저장되는 암묵적인 키-값 저장소다. 이 간접 참조는 데이터 흐름 추적을 어렵게 하고
# 페이로드 크기를 제한한다 (DB에서 기본 48KB, 설정 가능).
def extract(**kwargs):
    ti = kwargs['ti']
    data = [1, 2, 3, 4, 5]
    ti.xcom_push(key='data', value=data)

def transform(**kwargs):
    ti = kwargs['ti']
    # 정확한 task_id 문자열을 알아야 한다 — 태스크 이름 변경 시 런타임 에러의 원인이 된다
    data = ti.xcom_pull(key='data', task_ids='extract')
    result = [x * 2 for x in data]
    ti.xcom_push(key='result', value=result)

def load(**kwargs):
    ti = kwargs['ti']
    result = ti.xcom_pull(key='result', task_ids='transform')
    print(f"Loading: {result}")

with DAG(
    'etl_airflow',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',
    catchup=False,  # 이것 없이는 Airflow가 start_date 이후 누락된 모든 실행을 백필한다
) as dag:
    t1 = PythonOperator(task_id='extract', python_callable=extract)
    t2 = PythonOperator(task_id='transform', python_callable=transform)
    t3 = PythonOperator(task_id='load', python_callable=load)

    # 의존성을 명시적으로 선언해야 한다 — Airflow는 코드에서 추론할 수 없다
    t1 >> t2 >> t3
```

### 6.2 Prefect 버전

```python
# Prefect Flow — 동일한 ETL 로직을 보일러플레이트 없이 구현한다. 데이터는 함수
# 반환 값으로 전달되므로 (XCom이 아닌), 코드를 순수한 Python으로 테스트할 수 있다.
from prefect import flow, task

@task
def extract() -> list:
    return [1, 2, 3, 4, 5]

# 타입 힌트는 이중 역할을 한다: Python 타입 검사 + Prefect 스키마 유효성 검증
@task
def transform(data: list) -> list:
    return [x * 2 for x in data]

@task
def load(data: list):
    print(f"Loading: {data}")

@flow
def etl_prefect():
    # 의존성은 데이터 흐름에서 암묵적으로 결정된다 — Prefect가 DAG를 자동으로 구성한다.
    # 이는 명시적인 >> 순서가 실제 데이터 의존성과 불일치하는 버그 유형을 제거한다.
    data = extract()
    transformed = transform(data)
    load(transformed)

# 스케줄러/웹서버/DB 없이 일반 Python 스크립트처럼 실행하고 디버깅할 수 있다
etl_prefect()
```

### 6.3 주요 차이점

```python
"""
1. 데이터 전달:
   - Airflow: XCom 사용 (명시적 push/pull)
   - Prefect: 함수 반환값 직접 사용 (자연스러운 Python)

2. 의존성:
   - Airflow: >> 연산자로 명시
   - Prefect: 함수 호출 순서로 자동 추론

3. 스케줄링:
   - Airflow: Scheduler 프로세스 필수
   - Prefect: 선택적, 이벤트 기반 가능

4. 로컬 테스트:
   - Airflow: 복잡한 설정 필요
   - Prefect: 일반 Python 함수처럼 실행

5. 동적 워크플로우:
   - Airflow: 제한적 지원
   - Prefect: 네이티브 Python 제어문 사용
"""
```

---

## 7. 고급 기능

### 7.1 상태 핸들러

```python
from prefect import flow, task
from prefect.states import State, Completed, Failed

# 상태 핸들러는 상태 전환 시 실행된다 — 메인 태스크 로직을 복잡하게 만들지 않고
# 알림, 정리, 커스텀 로깅에 사용한다.
def custom_state_handler(task, task_run, state: State):
    """Task 상태 변경 시 호출"""
    if state.is_failed():
        # 프로덕션에서는 여기에 Slack/PagerDuty를 통합한다. state를 반환하면
        # (예외를 발생시키지 않고) Prefect가 정상적인 재시도/실패 처리를 계속한다.
        print(f"Task {task.name} failed!")
    return state

# on_failure 훅은 실패 시에만 실행된다 — 오류 알림만 필요한 경우
# 모든 상태 전환을 확인하는 것보다 효율적이다
@task(on_failure=[custom_state_handler])
def risky_task():
    raise ValueError("Something went wrong")

# 플로우 수준 핸들러는 플로우 내 모든 태스크의 실패를 감지한다 — 태스크별
# 알림 대신 단일 "파이프라인 실패" 알림을 보내는 데 유용하다
@flow(on_failure=[lambda flow, flow_run, state: print("Flow failed!")])
def my_flow():
    risky_task()
```

### 7.2 결과 저장소

```python
from prefect import flow, task
from prefect.filesystems import S3, LocalFileSystem
from prefect.serializers import JSONSerializer

# 결과 저장소는 플로우 실행 수명 이후에도 태스크 출력을 유지한다 — 실행 간
# 캐싱, 과거 결과 디버깅, 플로우-오브-플로우 데이터 전달을 가능하게 한다.
# LocalFileSystem은 개발 시 편리하고, 프로덕션에서는 S3/GCS를 사용한다.
@task(result_storage=LocalFileSystem(basepath="/tmp/prefect"))
def save_locally():
    return {"data": [1, 2, 3]}

# JSON 직렬화를 사용한 S3 저장소 — JSONSerializer는 기본 pickle과 달리
# 사람이 읽을 수 있는 결과를 생성하여 디버깅에 유용하다.
# persist_result=True는 실제로 스토리지에 쓰기 위해 필요하다. 없으면 결과는
# 메모리에 남아 프로세스 종료 시 사라진다.
@task(
    persist_result=True,
    result_storage=S3(bucket_path="my-bucket/results"),
    result_serializer=JSONSerializer(),
)
def save_to_s3():
    return {"large": "data"}
```

### 7.3 비밀 관리

```python
from prefect.blocks.system import Secret

# Block으로 비밀 저장 (UI 또는 CLI)
# prefect block register -m prefect.blocks.system

# 코드에서 사용
@task
def use_secret():
    api_key = Secret.load("my-api-key").get()
    # API 호출에 사용
    return f"Using key: {api_key[:4]}..."

# 환경 변수 사용
import os

@task
def use_env_var():
    return os.getenv("MY_SECRET")
```

---

## 연습 문제

### 문제 1: 기본 Flow 작성
3개의 Task(데이터 추출, 변환, 적재)로 구성된 ETL Flow를 작성하세요.

### 문제 2: 동적 Task
파일 목록을 입력받아 각 파일을 병렬로 처리하는 Flow를 작성하세요.

### 문제 3: 조건부 실행
데이터 크기에 따라 다른 처리 방식을 선택하는 Flow를 작성하세요.

---

## 요약

| 개념 | 설명 |
|------|------|
| **Flow** | 워크플로우 정의 (Airflow의 DAG) |
| **Task** | 개별 작업 단위 |
| **Deployment** | Flow의 배포 설정 |
| **Work Pool** | Worker 그룹 관리 |
| **Worker** | Flow 실행 에이전트 |

---

## 참고 자료

- [Prefect Documentation](https://docs.prefect.io/)
- [Prefect Tutorials](https://docs.prefect.io/tutorials/)
- [Prefect GitHub](https://github.com/PrefectHQ/prefect)

---

[← 이전: 5. Airflow 심화](05_Airflow_Advanced.md) | [다음: 7. Apache Spark 기초 →](07_Apache_Spark_Basics.md)
