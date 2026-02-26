# 데이터 품질과 거버넌스

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 데이터 품질의 6가지 핵심 차원(정확성, 완전성, 일관성, 적시성, 유일성, 유효성)을 정의하고, 각 차원이 하위 분석에 미치는 영향을 설명할 수 있습니다.
2. Great Expectations 또는 dbt 테스트와 같은 도구를 사용하여 프로덕션 파이프라인의 이상을 감지하는 자동화된 데이터 품질 검사를 구현할 수 있습니다.
3. 데이터 카탈로그(Data Catalog), 계보(Lineage) 추적, 소유권(Ownership) 정책을 포함한 데이터 거버넌스(Data Governance) 프레임워크를 설계할 수 있습니다.
4. 파이프라인 건전성을 실시간으로 모니터링하기 위한 데이터 품질 메트릭 수집 및 알림 패턴을 적용할 수 있습니다.
5. 데이터 옵저버빌리티(Data Observability) 플랫폼을 평가하고, 스키마 변경, 널(Null) 비율 급증, 분포 변화 감지 능력을 비교할 수 있습니다.
6. 데이터 거버넌스 정책과 GDPR 또는 CCPA와 같은 규제 준수 요건 사이의 관계를 분석할 수 있습니다.

---

## 개요

데이터 품질은 데이터의 정확성, 완전성, 일관성을 보장하는 것이고, 데이터 거버넌스는 데이터 자산을 체계적으로 관리하는 프레임워크입니다. 신뢰할 수 있는 데이터 파이프라인을 위해 필수적입니다.

---

## 1. 데이터 품질 차원

### 1.1 품질 차원 정의

```
┌────────────────────────────────────────────────────────────────┐
│                   데이터 품질 6대 차원                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   1. 정확성 (Accuracy)                                         │
│      - 데이터가 실제 값을 올바르게 반영하는가?                    │
│      - 예: 고객 이메일이 유효한 형식인가?                        │
│                                                                │
│   2. 완전성 (Completeness)                                     │
│      - 필요한 모든 데이터가 존재하는가?                          │
│      - 예: 필수 필드에 NULL이 없는가?                           │
│                                                                │
│   3. 일관성 (Consistency)                                      │
│      - 데이터가 여러 시스템 간 일치하는가?                       │
│      - 예: 주문 수가 주문 테이블과 집계 테이블에서 동일한가?      │
│                                                                │
│   4. 적시성 (Timeliness)                                       │
│      - 데이터가 적절한 시간 내에 제공되는가?                     │
│      - 예: 실시간 대시보드가 5분 내 갱신되는가?                  │
│                                                                │
│   5. 유일성 (Uniqueness)                                       │
│      - 중복 데이터가 없는가?                                    │
│      - 예: 동일한 주문이 중복 기록되지 않았는가?                 │
│                                                                │
│   6. 유효성 (Validity)                                         │
│      - 데이터가 정의된 규칙을 준수하는가?                        │
│      - 예: 날짜가 올바른 형식인가?                              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 품질 메트릭 예시

```python
from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass
class DataQualityMetrics:
    """데이터 품질 메트릭"""
    table_name: str
    row_count: int
    null_count: dict[str, int]
    duplicate_count: int
    freshness_hours: float
    schema_valid: bool

def calculate_quality_metrics(df: pd.DataFrame, table_name: str) -> DataQualityMetrics:
    """품질 메트릭 계산"""

    # 완전성: 전체가 아닌 컬럼별 NULL 수를 계산합니다,
    # 단일 중요 컬럼(예: 기본 키)이 50% null인 것이
    # 여러 선택적 컬럼에 null이 조금씩 있는 것보다 훨씬 심각하기 때문입니다
    null_count = {col: df[col].isna().sum() for col in df.columns}

    # 유일성: 행 전체 중복 제거는 최소 한 번 전달 시스템(Kafka, S3 이벤트)에서
    # 흔히 발생하는 정확한 중복 수집을 감지합니다
    duplicate_count = df.duplicated().sum()

    return DataQualityMetrics(
        table_name=table_name,
        row_count=len(df),
        null_count=null_count,
        duplicate_count=duplicate_count,
        freshness_hours=0,  # 별도 계산 필요
        schema_valid=True    # 별도 검증 필요
    )


def quality_score(metrics: DataQualityMetrics) -> float:
    """0-100 품질 점수 계산"""
    scores = []

    # 완전성 점수 (NULL 비율)
    # 테이블이 비어있거나 컬럼이 없을 때 0으로 나누기 방지
    total_cells = metrics.row_count * len(metrics.null_count)
    total_nulls = sum(metrics.null_count.values())
    completeness = (1 - total_nulls / total_cells) * 100 if total_cells > 0 else 100
    scores.append(completeness)

    # 유일성 점수 (중복 비율)
    # 여기서는 두 차원의 단순 평균을 사용합니다; 프로덕션에서는
    # 비즈니스 영향에 따라 가중치를 부여해야 합니다 (예: 금융 데이터에서
    # 유일성 실패는 치명적일 수 있지만, 로그의 완전성 격차는 허용 가능합니다)
    uniqueness = (1 - metrics.duplicate_count / metrics.row_count) * 100 if metrics.row_count > 0 else 100
    scores.append(uniqueness)

    return sum(scores) / len(scores)
```

---

## 2. Great Expectations

### 2.1 설치 및 초기화

```bash
# 설치
pip install great_expectations

# 프로젝트 초기화
great_expectations init
```

### 2.2 기본 사용법

```python
import great_expectations as gx
import pandas as pd

# GX는 모든 설정, 데이터 소스, Expectation Suite를 관리하는 중앙 진입점으로
# "Context"를 사용합니다 — dbt가 profiles.yml을 단일 설정 허브로 쓰는 것과 유사합니다
context = gx.get_context()

# 데이터 소스 추가 — 로컬/개발 환경 검증을 위한 Pandas 데이터 소스
# 프로덕션에서는 데이터를 메모리에 로드하지 않고 제자리에서 검증하기 위해
# SQLAlchemy 또는 Spark 데이터 소스로 교체하세요
datasource = context.sources.add_pandas("my_datasource")

# 데이터 에셋 정의
data_asset = datasource.add_dataframe_asset(name="orders")

# DataFrame 로드
df = pd.read_csv("orders.csv")

# Batch Request — "어떤 데이터"와 "어떤 Expectations"을 분리합니다.
# 동일한 Expectation Suite로 서로 다른 배치를 검증할 수 있습니다
# (예: 어제 데이터, 오늘 데이터, 백필 배치)
batch_request = data_asset.build_batch_request(dataframe=df)

# Expectation Suite 생성
suite = context.add_expectation_suite("orders_suite")

# Validator 생성 — 데이터(배치)와 규칙(Suite) 사이의 다리 역할
# Validator는 Expectations을 구체적인 SQL/Pandas 검사로 구체화합니다
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name="orders_suite"
)
```

### 2.3 Expectations 정의

```python
# 각 Expectation은 6가지 품질 차원 중 하나에 매핑됩니다.
# 차원별로 Expectations을 구성하면 어떤 품질 측면이 실패했는지 명확히 파악되어
# 타겟 수정이 가능합니다.

# 완전성: 기본 키는 절대 null이어서는 안 됩니다 — order_id가 null이면
# 다운스트림 조인과 집계가 신뢰할 수 없게 됩니다
validator.expect_column_values_to_not_be_null("order_id")

# 유일성: order_id 중복은 매출/건수 지표를 부풀립니다.
# 실패 시 소스의 최소 한 번 전달 방식을 조사하세요.
validator.expect_column_values_to_be_unique("order_id")

# 유효성: 범위 검사는 데이터 손상(음수 금액)과
# 단위 오류(센트 vs 달러)를 나타낼 수 있는 이상값(>100만)을 감지합니다
validator.expect_column_values_to_be_between(
    "amount",
    min_value=0,
    max_value=1000000
)

# 유효성: 열거형 검사는 소스 시스템의 스키마 드리프트(Schema Drift)나
# 처리되지 않은 비즈니스 상태 전환을 나타내는 예상치 못한 status 값을 감지합니다
validator.expect_column_values_to_be_in_set(
    "status",
    ["pending", "completed", "cancelled", "refunded"]
)

# 정확성: 정규식으로 이메일 형식 검증 — 데이터 입력 오류와
# 이메일 기반 고객 매칭을 깨뜨리는 쓰레기 값을 포착합니다
validator.expect_column_values_to_match_regex(
    "email",
    r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
)

# 완전성: 볼륨 검사는 업스트림 추출 실패를 감지합니다.
# 행 수가 1000 미만으로 떨어지면 소스 쿼리가 실패했거나
# 부분 결과를 반환했을 가능성이 높습니다
validator.expect_table_row_count_to_be_between(
    min_value=1000,
    max_value=1000000
)

# 일관성: 스키마 드리프트 감지 — 업스트림 시스템이 데이터 팀과
# 조율 없이 컬럼을 추가, 제거, 또는 이름 변경할 때 포착합니다
validator.expect_table_columns_to_match_set(
    ["order_id", "customer_id", "amount", "status", "order_date"]
)

# 유효성: 날짜 형식 검사는 다운스트림 파싱 실패를 예방합니다.
# 혼합된 형식(MM/DD vs YYYY-MM-DD)은 흔한 조용한 데이터 손상입니다
validator.expect_column_values_to_match_strftime_format(
    "order_date",
    "%Y-%m-%d"
)

# 일관성: 테이블 간 참조 무결성 — 고아(Orphan) customer_id는
# orders와 customers 추출 작업 간의 동기화 지연을 나타냅니다
validator.expect_column_values_to_be_in_set(
    "customer_id",
    customer_ids_list  # 고객 테이블의 ID 목록
)

# discard_failed_expectations=False는 실패한 Expectations을 포함한 모든 것을 보존합니다.
# 이는 Suite를 반복적으로 개발할 때 중요합니다
validator.save_expectation_suite(discard_failed_expectations=False)
```

### 2.4 검증 실행

```python
# Checkpoint 생성 및 실행
checkpoint = context.add_or_update_checkpoint(
    name="orders_checkpoint",
    validations=[
        {
            "batch_request": batch_request,
            "expectation_suite_name": "orders_suite"
        }
    ]
)

# 검증 실행
result = checkpoint.run()

# 결과 확인
print(f"Success: {result.success}")
print(f"Statistics: {result.statistics}")

# 실패한 Expectations 확인
for validation_result in result.list_validation_results():
    for exp_result in validation_result.results:
        if not exp_result.success:
            print(f"Failed: {exp_result.expectation_config.expectation_type}")
            print(f"  Column: {exp_result.expectation_config.kwargs.get('column')}")
            print(f"  Result: {exp_result.result}")
```

### 2.5 데이터 문서 생성

```python
# Data Docs 빌드 및 열기
context.build_data_docs()
context.open_data_docs()
```

---

## 3. Airflow 통합

### 3.1 Great Expectations Operator

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import great_expectations as gx

def validate_data(**kwargs):
    """Great Expectations 검증 Task"""
    context = gx.get_context()

    # Checkpoint는 배치 요청 + Expectation Suite + 액션(예: 결과 저장,
    # Slack 알림)을 묶습니다. Checkpoint를 통해 실행하면 수동 실행과
    # 예약된 실행 모두에서 일관된 검증 동작이 보장됩니다
    result = context.run_checkpoint(
        checkpoint_name="orders_checkpoint"
    )

    # 실패 시 예외를 발생시키면 Airflow가 이 태스크를 FAILED로 표시하고,
    # default_args에 설정된 이메일/Slack 알림을 트리거하며 다운스트림 태스크를 차단합니다
    # — 파이프라인의 품질 게이트 역할을 합니다
    if not result.success:
        raise ValueError("Data quality check failed!")

    return result.statistics


with DAG(
    'data_quality_dag',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',
) as dag:

    validate = PythonOperator(
        task_id='validate_orders',
        python_callable=validate_data,
    )
```

### 3.2 커스텀 품질 검사

```python
from airflow.operators.python import PythonOperator, BranchPythonOperator

def check_row_count(**kwargs):
    """행 수 검증"""
    import pandas as pd

    df = pd.read_parquet(f"/data/{kwargs['ds']}/orders.parquet")
    row_count = len(df)

    # XCom에 메트릭을 푸시하면 다운스트림 태스크(브랜칭 결정 등)가
    # 파일을 다시 읽지 않고 카운트에 접근할 수 있습니다 — 중복 I/O를 방지합니다
    kwargs['ti'].xcom_push(key='row_count', value=row_count)

    # 1000 임계값은 상식적 최솟값입니다: 실제 일일 주문량은 ~10,000+입니다.
    # 1000 미만이면 보통 진짜 낮은 비즈니스 활동이 아니라
    # 부분 추출 실패를 의미합니다
    if row_count < 1000:
        raise ValueError(f"Row count too low: {row_count}")

    return row_count


def check_freshness(**kwargs):
    """데이터 신선도 검증"""
    from datetime import datetime, timedelta

    # 파일 mtime을 데이터 신선도의 프록시로 사용합니다 — 단순하지만 효과적입니다.
    # 프로덕션에서는 더 정확한 신선도 측정을 위해 데이터 내부의
    # MAX(event_timestamp)를 확인하는 것이 좋습니다
    import os
    file_path = f"/data/{kwargs['ds']}/orders.parquet"
    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))

    age_hours = (datetime.now() - mtime).total_seconds() / 3600

    # 24시간 임계값은 일일 파이프라인 SLA와 일치합니다.
    # 데이터가 24시간보다 오래됐다면 파이프라인이 실행되지 않았거나
    # 업스트림 소스가 오래됐을 가능성이 높습니다
    if age_hours > 24:
        raise ValueError(f"Data too old: {age_hours:.1f} hours")

    return age_hours


def decide_next_step(**kwargs):
    """품질 결과에 따른 분기"""
    ti = kwargs['ti']
    row_count = ti.xcom_pull(task_ids='check_row_count', key='row_count')

    # 적응형 처리: 대규모 배치는 분산 컴퓨팅을 위해 Spark를 사용하고,
    # 소규모 배치는 오버헤드를 줄이기 위해 Pandas를 사용합니다
    # — 비용을 최적화하고 소규모 데이터셋에서 Spark 기동 지연을 방지합니다
    if row_count > 10000:
        return 'process_large_batch'
    else:
        return 'process_small_batch'


with DAG('quality_checks_dag', ...) as dag:

    check_rows = PythonOperator(
        task_id='check_row_count',
        python_callable=check_row_count,
    )

    check_fresh = PythonOperator(
        task_id='check_freshness',
        python_callable=check_freshness,
    )

    branch = BranchPythonOperator(
        task_id='decide_processing',
        python_callable=decide_next_step,
    )

    # 두 검사가 병렬로 실행(팬인)된 후 브랜칭 — 둘 중 하나라도
    # 실패하면 branch 태스크는 절대 실행되지 않아, 불량 데이터가
    # 다운스트림으로 흐르는 것을 방지합니다
    [check_rows, check_fresh] >> branch
```

---

## 4. 데이터 카탈로그

### 4.1 카탈로그 개념

```
┌────────────────────────────────────────────────────────────────┐
│                    데이터 카탈로그                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   메타데이터 관리 시스템:                                       │
│                                                                │
│   ┌────────────────────────────────────────────────────────┐  │
│   │  기술 메타데이터                                        │  │
│   │  - 스키마, 데이터 타입, 파티션                          │  │
│   │  - 위치, 형식, 크기                                     │  │
│   │  - 생성일, 수정일                                       │  │
│   └────────────────────────────────────────────────────────┘  │
│                                                                │
│   ┌────────────────────────────────────────────────────────┐  │
│   │  비즈니스 메타데이터                                    │  │
│   │  - 설명, 정의, 용어                                     │  │
│   │  - 소유자, 관리자                                       │  │
│   │  - 태그, 분류                                           │  │
│   └────────────────────────────────────────────────────────┘  │
│                                                                │
│   ┌────────────────────────────────────────────────────────┐  │
│   │  운영 메타데이터                                        │  │
│   │  - 사용 빈도, 쿼리 패턴                                 │  │
│   │  - 품질 점수, 이슈                                      │  │
│   │  - 접근 권한                                            │  │
│   └────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 4.2 카탈로그 도구

| 도구 | 유형 | 특징 |
|------|------|------|
| **DataHub** | 오픈소스 | LinkedIn 개발, 범용 |
| **Apache Atlas** | 오픈소스 | Hadoop 생태계 |
| **Amundsen** | 오픈소스 | Lyft 개발, 검색 중심 |
| **OpenMetadata** | 오픈소스 | 올인원 플랫폼 |
| **Atlan** | 상용 | 협업 중심 |
| **Alation** | 상용 | 엔터프라이즈 |

### 4.3 DataHub 예시

```python
# DataHub 메타데이터 수집 예시
# 프로그래밍 방식의 메타데이터 전송은 CI/CD 파이프라인의 일부로
# 카탈로그를 자동으로 업데이트합니다 — 스키마나 소유권이 변경될 때
# 수동 데이터 입력이 필요하지 않습니다
from datahub.emitter.mce_builder import make_dataset_urn
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.metadata.schema_classes import (
    DatasetPropertiesClass,
    SchemaMetadataClass,
    SchemaFieldClass,
    StringTypeClass,
    NumberTypeClass,
)

# REST Emitter는 DataHub GMS(Generalized Metadata Service)에 메타데이터를 전송합니다.
# 비동기, 고처리량 전송을 위해서는 KafkaEmitter를 사용하세요
emitter = DatahubRestEmitter(gms_server="http://localhost:8080")

# URN(Uniform Resource Name)은 모든 플랫폼에서 각 데이터셋을 고유하게 식별합니다.
# 3부 구조(플랫폼.이름.환경)는 dev와 prod 환경에서 동일한 테이블 이름이
# 존재할 때 명칭 충돌을 방지합니다
dataset_urn = make_dataset_urn(
    platform="postgres",
    name="analytics.public.fact_orders",
    env="PROD"
)

# Custom Properties는 DataHub UI에서 필터링/검색할 수 있는 거버넌스 메타데이터를 저장합니다.
# 여기서 PII 상태를 태깅하면 다운스트림에서 자동 접근 제어 적용이 가능합니다
properties = DatasetPropertiesClass(
    description="주문 팩트 테이블",
    customProperties={
        "owner": "data-team@company.com",
        "sla": "daily",
        "pii": "false"
    }
)

# 스키마 정의 — 필드 수준 메타데이터를 등록하면 영향 분석이 가능합니다:
# "amount"를 "order_amount"로 이름 변경 시 DataHub가 이 필드를 참조하는
# 모든 대시보드와 다운스트림 모델을 식별할 수 있습니다
schema = SchemaMetadataClass(
    schemaName="fact_orders",
    platform=f"urn:li:dataPlatform:postgres",
    fields=[
        SchemaFieldClass(
            fieldPath="order_id",
            type=StringTypeClass(),
            description="주문 고유 ID"
        ),
        SchemaFieldClass(
            fieldPath="amount",
            type=NumberTypeClass(),
            description="주문 금액"
        ),
    ]
)

# 메타데이터 emit
emitter.emit_mce(properties)
emitter.emit_mce(schema)
```

---

## 5. 데이터 리니지

### 5.1 리니지 개념

```
┌────────────────────────────────────────────────────────────────┐
│                     데이터 리니지 (Lineage)                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   데이터의 출처와 변환 과정을 추적:                              │
│                                                                │
│   Raw Sources          Staging           Marts                 │
│   ┌──────────┐        ┌──────────┐      ┌──────────┐          │
│   │ orders   │───────→│stg_orders│─────→│fct_orders│          │
│   │ (raw)    │        │          │      │          │          │
│   └──────────┘        └──────────┘      └────┬─────┘          │
│                                               │                │
│   ┌──────────┐        ┌──────────┐           │                │
│   │customers │───────→│stg_customers│────────→│                │
│   │ (raw)    │        │          │           │                │
│   └──────────┘        └──────────┘           │                │
│                                               ↓                │
│                                         ┌──────────┐          │
│                                         │ dashboard│          │
│                                         │ (BI)     │          │
│                                         └──────────┘          │
│                                                                │
│   활용:                                                        │
│   - 영향 분석: 소스 변경 시 영향받는 대상 파악                   │
│   - 근본 원인 분석: 데이터 이슈의 원인 추적                     │
│   - 규정 준수: 데이터 흐름 감사                                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 dbt 리니지

```bash
# dbt 리니지 생성
dbt docs generate

# 리니지 확인 (docs 서버)
dbt docs serve
```

```yaml
# dbt 모델 메타데이터
version: 2

models:
  - name: fct_orders
    description: "주문 팩트 테이블"
    meta:
      owner: "data-team"
      upstream:
        - stg_orders
        - stg_customers
      downstream:
        - sales_dashboard
        - ml_model_features
```

### 5.3 OpenLineage

```python
# OpenLineage를 사용한 리니지 추적
# OpenLineage는 특정 벤더에 종속되지 않는 리니지 이벤트 오픈 표준입니다.
# 이를 통해 이종 파이프라인(Spark + Airflow + dbt)이 단일 백엔드(예: Marquez, DataHub)에
# 벤더 종속 없이 리니지를 전송할 수 있습니다
from openlineage.client import OpenLineageClient
from openlineage.client.run import Run, Job, RunEvent, RunState
from openlineage.client.facet import (
    SqlJobFacet,
    SchemaDatasetFacet,
    SchemaField,
)
from datetime import datetime
import uuid

client = OpenLineageClient(url="http://localhost:5000")

# Job 정의 — Namespace + name은 조직 전체에서 작업을 고유하게 식별합니다.
# 팀 간 리니지 발견을 위해 일관된 네이밍 컨벤션(예: team.pipeline_name)을 사용하세요
job = Job(
    namespace="my_pipeline",
    name="transform_orders"
)

# 실행당 UUID는 각 실행을 독립적으로 추적할 수 있게 합니다.
# run_id를 통해 특정 파이프라인 실패 디버깅이 가능합니다
run_id = str(uuid.uuid4())
run = Run(runId=run_id)

# 입력 데이터셋 — 스키마 패싯과 함께 입력을 선언하면 자동 영향 분석이 가능합니다:
# raw.orders가 변경되면 리니지 그래프가 영향받는 다운스트림 작업과 데이터셋을 보여줍니다
input_datasets = [
    {
        "namespace": "postgres",
        "name": "raw.orders",
        "facets": {
            "schema": SchemaDatasetFacet(
                fields=[
                    SchemaField(name="order_id", type="string"),
                    SchemaField(name="amount", type="decimal"),
                ]
            )
        }
    }
]

# 출력 데이터셋
output_datasets = [
    {
        "namespace": "postgres",
        "name": "analytics.fct_orders",
    }
]

# START 이벤트는 실행 시작을 표시합니다 — 리니지 소비자가 진행 중인 작업을 추적하고
# 중단/멈춤 파이프라인을 감지할 수 있습니다
client.emit(
    RunEvent(
        eventType=RunState.START,
        eventTime=datetime.now().isoformat(),
        run=run,
        job=job,
        inputs=input_datasets,
        outputs=output_datasets,
    )
)

# ... 실제 변환 작업 ...

# COMPLETE 이벤트로 실행을 닫습니다. START와 COMPLETE 모두 전송하면 소요 시간 추적이 됩니다.
# COMPLETE가 수신되지 않으면 모니터링이 해당 실행을 잠재적 실패로 플래그할 수 있습니다
client.emit(
    RunEvent(
        eventType=RunState.COMPLETE,
        eventTime=datetime.now().isoformat(),
        run=run,
        job=job,
    )
)
```

---

## 6. 거버넌스 프레임워크

### 6.1 데이터 거버넌스 구성 요소

```
┌────────────────────────────────────────────────────────────────┐
│                 데이터 거버넌스 프레임워크                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   1. 조직 (Organization)                                       │
│      - 데이터 스튜어드 지정                                     │
│      - 역할과 책임 정의                                         │
│      - 거버넌스 위원회                                          │
│                                                                │
│   2. 정책 (Policies)                                           │
│      - 데이터 분류 정책                                         │
│      - 접근 제어 정책                                           │
│      - 보존/삭제 정책                                           │
│      - 품질 기준                                                │
│                                                                │
│   3. 프로세스 (Processes)                                      │
│      - 데이터 요청/승인 프로세스                                │
│      - 이슈 관리 프로세스                                       │
│      - 변경 관리 프로세스                                       │
│                                                                │
│   4. 기술 (Technology)                                         │
│      - 데이터 카탈로그                                          │
│      - 품질 모니터링                                            │
│      - 접근 제어 시스템                                         │
│      - 감사 로그                                                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 6.2 데이터 분류

```python
from enum import Enum

class DataClassification(Enum):
    """데이터 민감도 분류
    최소 민감에서 최대 민감까지 4단계. 이 계층 구조는 접근 제어 정책에 직접 매핑됩니다:
    예를 들어 RESTRICTED는 미사용 데이터 암호화와 비권한 사용자에 대한
    컬럼 수준 마스킹이 필요합니다.
    """
    PUBLIC = "public"           # 공개 가능
    INTERNAL = "internal"       # 내부 사용
    CONFIDENTIAL = "confidential"  # 기밀
    RESTRICTED = "restricted"   # 제한적 (PII, 금융)

class DataClassifier:
    """자동 데이터 분류"""

    # 흔한 PII 유형에 대한 정규식 패턴. 거짓 음성(False Negative)을 최소화하기 위해
    # 의도적으로 폭넓게 작성했습니다 — 비-PII를 과분류(제한)하는 것이
    # 실제 PII를 과소분류(노출)하는 것보다 낫습니다. 거짓 양성은
    # 수동 검토 시 화이트리스트에 추가할 수 있습니다
    PII_PATTERNS = {
        'email': r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
        'phone': r'\d{3}-\d{3,4}-\d{4}',
        'ssn': r'\d{3}-\d{2}-\d{4}',
        'credit_card': r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}',
    }

    PII_COLUMN_NAMES = [
        'email', 'phone', 'ssn', 'social_security',
        'credit_card', 'password', 'address'
    ]

    @classmethod
    def classify_column(cls, column_name: str, sample_values: list) -> DataClassification:
        """컬럼 분류"""
        column_lower = column_name.lower()

        # 컬럼명 기반 검사를 먼저 실행합니다. O(1)이고
        # 데이터를 읽지 않고도 가장 일반적인 케이스를 포착하기 때문입니다
        if any(pii in column_lower for pii in cls.PII_COLUMN_NAMES):
            return DataClassification.RESTRICTED

        # 값 기반 검사를 폴백으로 사용합니다: 일반적인 이름의 컬럼(예:
        # 실제로 주민번호가 들어있는 'field_1')에서 PII를 포착합니다.
        # 100개 샘플링은 정확도와 성능 사이의 균형을 맞춥니다 —
        # 대형 테이블에서 모든 행을 스캔하면 너무 느립니다
        import re
        for value in sample_values[:100]:  # 샘플링
            if value is None:
                continue
            for pii_type, pattern in cls.PII_PATTERNS.items():
                if re.match(pattern, str(value)):
                    return DataClassification.RESTRICTED

        # 기본값을 INTERNAL(PUBLIC이 아님)로 설정합니다 — 안전한 기본값입니다:
        # 데이터는 사람이 검토한 후 명시적으로 PUBLIC으로 승격되어야 합니다
        return DataClassification.INTERNAL
```

---

## 연습 문제

### 문제 1: Great Expectations
주문 데이터에 대한 Expectation Suite를 작성하세요 (NULL 체크, 유니크, 값 범위, 참조 무결성).

### 문제 2: 품질 대시보드
일별 데이터 품질 점수를 계산하고 시각화하는 파이프라인을 설계하세요.

### 문제 3: 리니지 추적
ETL 파이프라인의 리니지를 자동으로 추적하는 시스템을 설계하세요.

---

## 요약

| 개념 | 설명 |
|------|------|
| **데이터 품질** | 정확성, 완전성, 일관성, 적시성 보장 |
| **Great Expectations** | Python 기반 데이터 품질 프레임워크 |
| **데이터 카탈로그** | 메타데이터 관리 시스템 |
| **데이터 리니지** | 데이터 출처와 변환 추적 |
| **데이터 거버넌스** | 데이터 자산의 체계적 관리 |

---

## 참고 자료

- [Great Expectations Documentation](https://docs.greatexpectations.io/)
- [DataHub Documentation](https://datahubproject.io/docs/)
- [OpenLineage](https://openlineage.io/)
- [DMBOK (Data Management Body of Knowledge)](https://www.dama.org/cpages/body-of-knowledge)
