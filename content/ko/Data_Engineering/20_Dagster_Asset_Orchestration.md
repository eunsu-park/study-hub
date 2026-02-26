[← 이전: 19. Lakehouse 실전 패턴](19_Lakehouse_Practical_Patterns.md) | [다음: 21. 데이터 버전 관리와 데이터 계약 →](21_Data_Versioning_and_Contracts.md)

# Dagster — 자산 기반 오케스트레이션(Asset-Based Orchestration)

## 학습 목표

1. Dagster의 소프트웨어 정의 자산(Software-Defined Assets) 철학과 태스크 기반 DAG와의 차이점 이해
2. 핵심 Dagster 개념 마스터: Assets, Ops, Jobs, Graphs, Resources, IO Managers
3. 핵심 측면(개발자 경험, 테스트, 관측성)에서 Dagster와 Airflow 비교
4. `dagster.yaml`, `definitions.py`를 포함한 올바른 구조의 Dagster 프로젝트 설정
5. Dagster의 자산 그래프(asset graph)를 사용하여 엔드투엔드 데이터 파이프라인 구축
6. 증분 처리(incremental processing)와 시간 윈도우(time-windowed) 처리를 위한 파티션 자산(partitioned assets) 구현
7. 실세계 파이프라인을 위해 Dagster를 dbt, Spark, pandas와 통합

---

## 개요

대부분의 오케스트레이션 도구들 — Airflow, Prefect, Luigi — 은 **태스크(task)** 단위로 생각합니다: "이 Python 함수를 실행하고, 그다음 저 함수를 실행하라." Dagster는 이 사고 모델을 완전히 뒤집습니다. "어떤 단계를 실행해야 하는가?"라고 묻는 대신, Dagster는 "어떤 데이터 자산(data asset)이 존재해야 하며, 어떻게 파생되는가?"라고 묻습니다. 이 겉보기에는 작은 변화가 우리가 데이터 파이프라인을 구축하고, 테스트하고, 디버그하고, 관측하는 방식에 심대한 영향을 미칩니다.

Dagster는 Nick Schrock(Facebook에서 GraphQL 공동 창시자)이 만들었으며 2019년에 처음 출시되었습니다. 2022년 Dagster 1.0으로 프로덕션 수준의 성숙도에 도달했으며, 이후 가장 빠르게 성장하는 오케스트레이션 프레임워크 중 하나가 되었습니다. 자산 중심 모델은 현대 데이터 스택 — dbt 모델, ML 피처, 분석 테이블이 모두 소비자가 의존하는 **자산(assets)** 인 환경 — 과 자연스럽게 맞아떨어집니다.

이 레슨에서는 Dagster에 대한 철학부터 프로덕션 배포까지 완전한 이해를 구축합니다. 마지막에는 테스트 가능하고, 관측 가능하며, 유지보수 가능한 자산 기반 파이프라인을 설계하고 구현할 수 있게 됩니다.

> **비유**: Dagster 자산은 각 요리(자산)가 자신의 재료(업스트림 자산)를 선언하는 레시피 책과 같습니다. 주방(Dagster)이 자동으로 조리 순서를 파악합니다. 소스 레시피를 업데이트하면, 주방은 어떤 요리를 다시 준비해야 하는지 압니다 — 모든 단계를 일일이 나열하지 않아도 됩니다.

---

## 1. 소프트웨어 정의 자산(Software-Defined Asset) 철학

### 1.1 태스크 기반 vs 자산 기반 사고

Dagster와 전통적인 오케스트레이터 사이의 근본적인 차이는 무엇을 주요 추상화(primary abstraction)로 보는가에 있습니다.

```python
"""
태스크 기반 오케스트레이션(Airflow 모델):
─────────────────────────────────────────
"어떤 TASK를 어떤 순서로 실행해야 하는가?"

  extract_orders() → clean_orders() → compute_metrics() → load_dashboard()
       Task 1           Task 2            Task 3             Task 4

- DAG는 계산(COMPUTATION) 단계를 설명한다
- 데이터는 태스크 실행의 부산물(side effect)이다
- Task 3가 실패하면, 어느 TASK가 실패했는지 알 수 있다
- 하지만 어떤 DATASET이 오래됐는지(stale)? 직접적으로는 알 수 없다.


자산 기반 오케스트레이션(Dagster 모델):
──────────────────────────────────────────
"어떤 DATA ASSET이 존재해야 하며, 어떻게 파생되는가?"

  raw_orders → cleaned_orders → order_metrics → dashboard_summary
    Asset 1       Asset 2          Asset 3          Asset 4

- 그래프는 DATA 의존성을 설명한다
- 계산은 자산을 생성하기 위한 수단이다
- order_metrics가 오래됐다면, 즉시 볼 수 있다:
  - 마지막으로 구체화(materialized)된 시점
  - 의존하는 업스트림 자산
  - 다운스트림에서 소비하는 주체
"""
```

왜 이것이 중요한가? 실제 시나리오를 생각해봅시다: 오전 3시에 일일 파이프라인이 실패합니다. 태스크 기반 시스템에서는 "`compute_revenue` Task 실패"를 볼 수 있습니다. 그러면 어떤 데이터셋이 영향을 받는지 직접 파악해야 합니다. Dagster에서는 "자산 `daily_revenue`가 어제부터 오래됨(stale)"을 볼 수 있으며 — 모든 다운스트림 소비자도 잠재적으로 오래된 것으로 시각적으로 표시됩니다.

### 1.2 Dagster의 세 가지 기둥

Dagster는 서로를 강화하는 세 가지 핵심 아이디어를 기반으로 합니다:

```python
"""
Dagster의 세 가지 기둥:

1. 소프트웨어 정의 자산(SOFTWARE-DEFINED ASSETS)
   - 자산은 부산물이 아니라 1등 시민(first-class citizen)
   - 각 자산은 선언한다: 무엇을 생성하는지, 무엇에 의존하는지, 어떻게 계산하는지
   - 자산 그래프가 곧 파이프라인 정의

2. 선언적 데이터 관리(DECLARATIVE DATA MANAGEMENT)
   - 무엇이 존재해야 하는지 선언하면, Dagster가 어떻게 도달할지 파악
   - 조정(Reconciliation): 원하는 상태 vs 실제 상태 비교
   - 신선도 정책(Freshness policies): "이 자산은 1시간 이내의 것이어야 한다"

3. 설계상 테스트 가능(TESTABLE BY DESIGN)
   - 모든 자산은 Python 함수 — 독립적으로 단위 테스트 가능
   - 리소스는 주입 가능(injectable) — 프로덕션 DB를 테스트 DB로 교체
   - 파이프라인을 로컬에서 실행하기 위해 스케줄러를 실행할 필요 없음

세 가지가 함께 다음과 같은 개발 경험을 만든다:
  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
  │ Python로       │ ──→ │  로컬에서 테스트│ ──→ │  자신 있게     │
  │ 자산 정의      │     │  (pytest)       │     │  배포           │
  └────────────────┘     └────────────────┘     └────────────────┘
"""
```

### 1.3 Dagster를 선택해야 할 때

Dagster가 뛰어난 시나리오와 덜 이상적인 시나리오가 있습니다:

| 시나리오 | Dagster | Airflow |
|----------|---------|---------|
| 데이터 중심 파이프라인 (ELT, 분석) | 우수 | 양호 |
| ML 피처 파이프라인 | 우수 | 보통 |
| dbt 통합 | 네이티브 (`dagster-dbt`) | 오퍼레이터 경유 |
| 일반 태스크 오케스트레이션 (비데이터) | 보통 | 우수 |
| 기존 Airflow 투자 | 마이그레이션 비용 | 유지 |
| 팀 규모 < 5명, 신규 프로젝트 | 권장 | 가능 |
| 테스트 & 로컬 개발 | 우수 | 설정 필요 |
| 이벤트 기반 / 센서 중심 | 양호 | 양호 |
| 성숙한 생태계 / 커뮤니티 규모 | 성장 중 | 매우 큼 |

---

## 2. 핵심 개념 심층 분석

### 2.1 Assets — 기반

**자산(asset)** 은 데이터 플랫폼의 영속적인 객체입니다 — 테이블, 파일, 모델 아티팩트, 대시보드. Dagster에서는 데코레이터가 적용된 Python 함수로 자산을 정의합니다.

```python
import dagster as dg
import pandas as pd

# 왜 @asset 데코레이터인가? Dagster에게 알려준다:
# 1. 이 함수는 명명된 데이터 자산을 생성(PRODUCES)한다
# 2. 파라미터는 업스트림 의존성(이름으로)이다
# 3. 반환값이 자산의 데이터 그 자체이다

@dg.asset(
    description="이커머스 API에서 수집된 원시 주문 데이터",
    metadata={"source": "api.store.com/orders", "owner": "data-team"},
    group_name="bronze",          # Dagster UI에서의 시각적 그룹화
)
def raw_orders() -> pd.DataFrame:
    """이커머스 API에서 원시 주문 데이터를 수집한다.

    이 자산은 업스트림 의존성이 없다 — SOURCE 자산이다.
    """
    # 프로덕션에서는 API를 호출하거나 S3에서 읽을 것이다
    return pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "customer_id": [101, 102, 101, 103, 102],
        "amount": [99.99, 149.50, 25.00, 299.99, 75.00],
        "status": ["completed", "completed", "refunded", "completed", "pending"],
        "created_at": pd.date_range("2024-01-01", periods=5, freq="D"),
    })


@dg.asset(
    description="유효하지 않은 레코드가 제거되고 타입이 표준화된 주문",
    group_name="silver",
)
def cleaned_orders(raw_orders: pd.DataFrame) -> pd.DataFrame:
    """주문 데이터를 정리하고 검증한다.

    파라미터 이름이 중요한 이유:
      - `raw_orders`는 업스트림 자산 이름과 정확히 일치한다
      - Dagster가 이를 의존성으로 자동 인식한다
      - Airflow의 >>처럼 수동으로 의존성을 연결할 필요 없다
    """
    df = raw_orders.copy()

    # 실버 레이어에서 완료된 주문만 필터링하는 이유?
    # Bronze는 모든 데이터를 보존; Silver는 비즈니스 규칙을 적용
    df = df[df["status"] == "completed"]

    # 타입 표준화 — 다운스트림 자산이 일관된 타입을 받도록 보장
    df["amount"] = df["amount"].astype(float)
    df["created_at"] = pd.to_datetime(df["created_at"])

    return df


@dg.asset(
    description="고객별 집계된 주문 지표",
    group_name="gold",
)
def order_metrics(cleaned_orders: pd.DataFrame) -> pd.DataFrame:
    """고객별 주문 지표를 계산한다.

    골드 레이어에서 집계하는 이유?
      - 골드 자산은 비즈니스 소비자(BI 도구, 대시보드)에게 서비스된다
      - 사전 집계는 분석가의 반복 계산을 방지한다
      - 명확한 소유권: 데이터 팀이 gold를 소유하고, 분석가가 소비한다
    """
    metrics = cleaned_orders.groupby("customer_id").agg(
        total_orders=("order_id", "count"),
        total_revenue=("amount", "sum"),
        avg_order_value=("amount", "mean"),
        first_order=("created_at", "min"),
        last_order=("created_at", "max"),
    ).reset_index()

    return metrics
```

Dagster가 이 코드에서 구성하는 자산 그래프:

```
raw_orders ──→ cleaned_orders ──→ order_metrics
 (bronze)         (silver)           (gold)
```

### 2.2 Ops, Graphs, Jobs — 계산 레이어

자산이 주요 추상화이지만, Dagster는 계산 단계에 대한 세밀한 제어가 필요한 경우를 위해 **ops**(오퍼레이션)도 지원합니다.

```python
import dagster as dg

# 자산 대신 Ops를 사용하는 경우:
# - 계산이 영속적인 데이터 자산을 생성하지 않을 때
# - 단계별로 세밀한 재시도/타임아웃이 필요할 때
# - Airflow에서 마이그레이션 시 (ops는 태스크에 대응)

@dg.op(
    description="소스 API에 도달 가능한지 검증",
    retry_policy=dg.RetryPolicy(max_retries=3, delay=10),
)
def check_api_health(context: dg.OpExecutionContext) -> bool:
    """외부 API가 응답하는지 확인한다."""
    context.log.info("API 상태 확인 중...")
    # 프로덕션에서: requests.get("https://api.store.com/health")
    return True


@dg.op
def extract_data(context: dg.OpExecutionContext, api_healthy: bool) -> dict:
    """API가 정상인 경우에만 데이터를 추출한다."""
    if not api_healthy:
        raise dg.Failure(description="API가 정상 상태가 아닙니다")
    context.log.info("API에서 데이터 추출 중")
    return {"orders": [1, 2, 3], "extracted_at": "2024-01-15"}


@dg.op
def validate_data(context: dg.OpExecutionContext, raw_data: dict) -> dict:
    """추출된 데이터가 기대치를 충족하는지 검증한다."""
    assert len(raw_data["orders"]) > 0, "추출된 주문 없음"
    context.log.info(f"{len(raw_data['orders'])}개 주문 검증 완료")
    return raw_data


# Graph는 Ops를 계산 DAG로 구성한다
# Graph와 Job을 분리하는 이유?
# - Graph = 논리적 계산 (재사용 가능)
# - Job = Graph + 구성 (환경별)

@dg.graph
def etl_graph():
    """ETL 계산 그래프를 정의한다."""
    healthy = check_api_health()
    raw = extract_data(healthy)
    validate_data(raw)


# Job = 특정 리소스/설정에 바인딩된 Graph
etl_job = etl_graph.to_job(
    name="etl_job",
    description="주문 데이터를 위한 일일 ETL 잡",
    config={
        "ops": {
            "check_api_health": {"config": {}},
        }
    },
)
```

**무엇을 언제 사용하는가:**

| 개념 | 사용 시기 | 생각하는 방식 |
|---------|----------|----------------|
| `@asset` | 영속적 데이터 생성 | "어떤 데이터가 존재해야 하는가?" |
| `@op` | 계산의 한 단계 | "어떤 계산이 실행돼야 하는가?" |
| `@graph` | ops 구성 | "단계들이 어떻게 연결되는가?" |
| `Job` | 그래프 실행 | "언제, 어떻게 실행하는가?" |

### 2.3 Resources — 의존성 주입(Dependency Injection)

리소스(Resource)는 Dagster의 의존성 주입 메커니즘입니다. 자산 코드를 변경하지 않고 환경(dev, staging, production)간에 구현체를 교체할 수 있게 해줍니다.

```python
import dagster as dg

# 리소스가 필요한 이유?
# 1. 자산을 인프라(S3 클라이언트, DB 연결)에서 분리
# 2. 테스트 가능 — 테스트에서 mock 리소스 주입
# 3. 자산 간 연결 공유 (커넥션 풀링)
# 4. 환경별 구성 (dev는 로컬 파일, prod는 S3)


class DatabaseResource(dg.ConfigurableResource):
    """설정 가능한 데이터베이스 연결 리소스.

    ConfigurableResource를 사용하는 이유?
      - Pydantic을 통한 타입 안전 구성
      - 런타임이 아닌 파이프라인 시작 시 검증
      - Dagster UI에서 자기 문서화
    """
    host: str
    port: int = 5432
    database: str
    username: str
    password: str

    def get_connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    def execute_query(self, query: str) -> list:
        """SQL 쿼리를 실행한다 (설명을 위해 단순화)."""
        # 프로덕션에서: sqlalchemy 또는 psycopg2 사용
        return [{"result": "mock_data"}]


class S3Resource(dg.ConfigurableResource):
    """데이터 읽기/쓰기를 위한 S3 클라이언트 리소스."""
    bucket: str
    region: str = "us-east-1"
    endpoint_url: str = ""  # dev에서 MinIO/LocalStack용

    def read_parquet(self, key: str) -> "pd.DataFrame":
        """S3에서 Parquet 파일을 읽는다."""
        import pandas as pd
        path = f"s3://{self.bucket}/{key}"
        return pd.read_parquet(path)

    def write_parquet(self, df: "pd.DataFrame", key: str) -> None:
        """DataFrame을 Parquet으로 S3에 쓴다."""
        path = f"s3://{self.bucket}/{key}"
        df.to_parquet(path, index=False)


# 자산에서 리소스 사용:

@dg.asset
def revenue_report(
    context: dg.AssetExecutionContext,
    database: DatabaseResource,      # 이름으로 자동 주입
    s3: S3Resource,                   # 이름으로 자동 주입
) -> None:
    """데이터베이스에서 S3로 매출 보고서를 생성한다.

    리소스를 파라미터로 선언하는 이유?
      - Dagster가 런타임에 자동으로 주입한다
      - 테스트에서 mock 리소스를 제공할 수 있다
      - 자산이 자기 문서화된다: 의존성을 볼 수 있다
    """
    results = database.execute_query("SELECT * FROM daily_revenue")
    import pandas as pd
    df = pd.DataFrame(results)
    s3.write_parquet(df, "reports/revenue/latest.parquet")
    context.log.info(f"{len(df)}행을 S3에 기록했습니다")
```

### 2.4 IO Managers — 자산 영속성 제어

IO 매니저(IO Manager)는 자산이 **어떻게** 저장되고 로드되는지를 정의합니다. "무엇"(자산 로직)과 "어디"(스토리지)를 분리합니다.

```python
import dagster as dg
import pandas as pd
from pathlib import Path


class ParquetIOManager(dg.ConfigurableIOManager):
    """로컬 파일시스템에 자산을 Parquet 파일로 저장한다.

    IO 매니저가 필요한 이유?
      - 자산 함수는 DataFrame을 반환; IO 매니저가 영속성을 처리
      - 설정으로 로컬 Parquet(dev)와 S3 Parquet(prod) 간 교체 가능
      - 업스트림 자산은 다운스트림 실행 전에 자동으로 로드됨
    """
    base_path: str

    def handle_output(self, context: dg.OutputContext, obj: pd.DataFrame) -> None:
        """자산의 출력을 Parquet 파일에 저장한다."""
        # 경로에 asset_key를 사용하는 이유?
        # 각 자산은 고유하고 결정론적인 파일 경로를 갖는다
        path = Path(self.base_path) / f"{context.asset_key.to_python_identifier()}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        obj.to_parquet(path, index=False)
        context.log.info(f"{len(obj)}행을 {path}에 기록했습니다")

        # Dagster UI에서 보이는 메타데이터 첨부
        context.add_output_metadata({
            "num_rows": len(obj),
            "columns": list(obj.columns),
            "path": str(path),
        })

    def load_input(self, context: dg.InputContext) -> pd.DataFrame:
        """Parquet 파일에서 업스트림 자산을 로드한다."""
        path = Path(self.base_path) / f"{context.asset_key.to_python_identifier()}.parquet"
        df = pd.read_parquet(path)
        context.log.info(f"{path}에서 {len(df)}행을 로드했습니다")
        return df


class CsvIOManager(dg.ConfigurableIOManager):
    """자산을 CSV 파일로 저장한다 (디버깅 / 소규모 데이터셋에 유용)."""
    base_path: str

    def handle_output(self, context: dg.OutputContext, obj: pd.DataFrame) -> None:
        path = Path(self.base_path) / f"{context.asset_key.to_python_identifier()}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        obj.to_csv(path, index=False)

    def load_input(self, context: dg.InputContext) -> pd.DataFrame:
        path = Path(self.base_path) / f"{context.asset_key.to_python_identifier()}.csv"
        return pd.read_csv(path)
```

---

## 3. Dagster vs Airflow — 상세 비교

트레이드오프를 이해하면 팀과 프로젝트에 맞는 정보에 기반한 선택을 할 수 있습니다.

### 3.1 기능 비교

| 차원 | Dagster | Airflow |
|-----------|---------|---------|
| **주요 추상화** | 소프트웨어 정의 자산 | 태스크 DAG |
| **데이터 인식** | 1등 시민 (계보, 신선도) | 제한적 (메타데이터 플러그인 경유) |
| **로컬 개발** | `dagster dev` (즉시) | Docker Compose 또는 독립 실행형 |
| **테스트** | `pytest` 네이티브, 인프로세스 | 스케줄러 실행 필요 |
| **동적 파이프라인** | 네이티브 (Python 제어 흐름) | 2.4+에서 `expand()` |
| **백필(Backfill)** | 자산 수준 파티션 백필 | DAG 수준 재실행 |
| **UI** | 자산 그래프 + 계보 + 신선도 | 태스크 그래프 + 간트 차트 |
| **구성** | Pydantic 기반, 타입 안전 | Airflow Variables/Connections |
| **스케줄링** | Cron, 센서, 신선도 정책 | Cron, 타임테이블, 센서 |
| **커뮤니티** | GitHub 별 ~10K, 성장 중 | GitHub 별 ~37K, 성숙 |
| **관리형 클라우드** | Dagster Cloud (Dagster+) | Astronomer, MWAA, Cloud Composer |
| **플러그인 생태계** | ~50 통합 | ~1000+ 프로바이더 |

### 3.2 코드 비교 — 동일한 파이프라인을 두 시스템으로

```python
"""
동일한 파이프라인 — "주문 수집, 정리, 집계" — 을 두 시스템으로 구현.
이는 철학적 차이를 잘 보여준다.
"""

# ── Airflow 버전 ──────────────────────────────────────────────
# 초점: 어떤 TASK를, 어떤 순서로 실행할 것인가

# from airflow.decorators import dag, task
# @dag(schedule='@daily', start_date=datetime(2024, 1, 1))
# def order_pipeline():
#     @task()
#     def extract():
#         return fetch_orders()
#
#     @task()
#     def transform(raw):
#         return clean(raw)
#
#     @task()
#     def load(cleaned):
#         write_to_warehouse(cleaned)
#
#     load(transform(extract()))
#
# order_pipeline()

# 핵심 관찰:
# - TASK(동사)를 정의한다: extract, transform, load
# - 데이터 흐름은 반환값을 통해 암묵적이다
# - "이 task가 무엇을 생성하는가?"에 대한 내장 개념 없음
# - 테스트 시 Airflow 컨텍스트 필요

# ── Dagster 버전 ──────────────────────────────────────────────
# 초점: 어떤 DATA가 존재해야 하는지, 어떻게 파생되는지

# import dagster as dg
# @dg.asset
# def raw_orders():
#     return fetch_orders()
#
# @dg.asset
# def cleaned_orders(raw_orders):
#     return clean(raw_orders)
#
# @dg.asset
# def order_metrics(cleaned_orders):
#     return aggregate(cleaned_orders)

# 핵심 관찰:
# - ASSET(명사)을 정의한다: raw_orders, cleaned_orders, order_metrics
# - 의존성은 함수 파라미터를 통해 명시적이다
# - 각 자산은 독립적으로 구체화(materializable)하고 테스트 가능
# - 테스트에 오케스트레이터 컨텍스트 불필요
```

### 3.3 마이그레이션 경로: Airflow에서 Dagster로

Dagster는 점진적 마이그레이션을 위해 `dagster-airflow`를 제공합니다:

```python
"""
마이그레이션 전략 (점진적, 빅뱅이 아닌):

Phase 1: Dagster를 Airflow와 함께 실행
  - 기존 Airflow DAG를 Dagster 잡으로 래핑
  - dagster-airflow를 사용하여 DAG 임포트
  - 두 시스템이 병렬로 실행

Phase 2: 중요 파이프라인 변환
  - 고가치 DAG를 Dagster 자산으로 재작성
  - 레거시 DAG는 Airflow에 유지
  - Dagster 센서가 Airflow 완료를 감시

Phase 3: 완전 마이그레이션
  - 모든 파이프라인을 Dagster로
  - Airflow 해체
  - 모니터링 통합

예상 일정: 5-10명 엔지니어 팀 기준 3-6개월
"""

# 예시: Airflow DAG를 Dagster로 임포트
# from dagster_airflow import make_dagster_definitions_from_airflow_dags
#
# definitions = make_dagster_definitions_from_airflow_dags(
#     airflow_dags_path="/path/to/airflow/dags",
#     connections=[...],
# )
```

---

## 4. Dagster 프로젝트 구조

### 4.1 표준 레이아웃

잘 조직된 Dagster 프로젝트는 다음 구조를 따릅니다:

```
my_dagster_project/
├── my_dagster_project/          # Python 패키지
│   ├── __init__.py
│   ├── definitions.py           # 진입점 — Dagster가 이것을 로드
│   ├── assets/                  # 자산 정의
│   │   ├── __init__.py
│   │   ├── bronze.py            # 원시 수집 자산
│   │   ├── silver.py            # 정리/검증된 자산
│   │   └── gold.py              # 비즈니스 수준 집계
│   ├── resources/               # 리소스 정의
│   │   ├── __init__.py
│   │   ├── database.py
│   │   └── storage.py
│   ├── jobs.py                  # 잡 정의 (ops 사용 시)
│   ├── schedules.py             # 스케줄 정의
│   └── sensors.py               # 센서 정의
├── tests/                       # 단위 및 통합 테스트
│   ├── test_assets.py
│   └── test_resources.py
├── dagster.yaml                 # Dagster 인스턴스 구성
├── pyproject.toml               # 프로젝트 메타데이터 + Dagster 구성
└── workspace.yaml               # 멀티 프로젝트 워크스페이스 구성
```

### 4.2 Definitions 객체

`definitions.py` 파일은 진입점입니다 — 모든 자산, 리소스, 잡, 스케줄, 센서에 대해 Dagster에게 알려줍니다.

```python
# definitions.py — Dagster 프로젝트의 중앙 레지스트리

import dagster as dg
from my_dagster_project.assets.bronze import raw_orders, raw_customers
from my_dagster_project.assets.silver import cleaned_orders, cleaned_customers
from my_dagster_project.assets.gold import order_metrics, customer_lifetime_value

# 단일 Definitions 객체를 사용하는 이유?
# 1. Dagster가 한 곳에서 프로젝트의 모든 것을 알 수 있다
# 2. 검증이 런타임이 아닌 임포트 시에 발생한다
# 3. UI/CLI가 이것을 읽어 자산 그래프를 렌더링한다

defs = dg.Definitions(
    assets=[
        # Bronze 레이어 (원시 수집)
        raw_orders,
        raw_customers,
        # Silver 레이어 (정리됨)
        cleaned_orders,
        cleaned_customers,
        # Gold 레이어 (집계됨)
        order_metrics,
        customer_lifetime_value,
    ],
    resources={
        # 여기서 리소스를 정의하는 이유?
        # - 자산은 이름(문자열 키)으로 리소스를 참조한다
        # - 다른 환경을 위한 구현체 교체
        "database": DatabaseResource(
            host="localhost",
            database="analytics",
            username="dagster",
            password=dg.EnvVar("DB_PASSWORD"),   # 환경에서 읽음
        ),
        "io_manager": ParquetIOManager(
            base_path="/data/dagster/assets",
        ),
    },
    schedules=[daily_refresh_schedule],
    sensors=[new_file_sensor],
)
```

### 4.3 구성 파일

```yaml
# dagster.yaml — Dagster 인스턴스 구성
# Dagster INSTANCE를 구성한다 (스토리지, 텔레메트리 등)
# 코드를 구성하는 definitions.py와 혼동하지 말 것

storage:
  # Dagster가 실행 이력, 이벤트 로그 등을 저장하는 곳
  postgres:
    postgres_url:
      env: DAGSTER_PG_URL
    # SQLite 대신 Postgres를 사용하는 이유?
    # - 동시 접근 (다수의 워커)
    # - 재시작 간 영속성
    # - 프로덕션 배포에 필수

run_coordinator:
  module: dagster.core.run_coordinator
  class: QueuedRunCoordinator
  config:
    max_concurrent_runs: 10
    # 동시 실행을 제한하는 이유?
    # - 리소스 고갈 방지
    # - 데이터베이스 연결 수 제어
    # - 예측 가능한 파이프라인 실행

telemetry:
  enabled: false
```

```toml
# pyproject.toml — 프로젝트 구성
[project]
name = "my_dagster_project"
version = "0.1.0"
dependencies = [
    "dagster>=1.6",
    "dagster-pandas",
    "dagster-dbt",
    "pandas",
    "pyarrow",
]

[tool.dagster]
module_name = "my_dagster_project.definitions"
# module_name을 지정하는 이유?
# - `dagster dev`가 이것을 사용하여 Definitions를 찾는다
# - 단일 프로젝트 설정에서 workspace.yaml 불필요
```

---

## 5. Dagster 자산으로 데이터 파이프라인 구축

### 5.1 다중 소스 수집 파이프라인

여러 데이터 소스를 가진 현실적인 이커머스 분석 파이프라인을 구축해봅니다.

```python
import dagster as dg
import pandas as pd
from datetime import datetime, timedelta


# ── 소스 자산 (Bronze 레이어) ──────────────────────────────────

@dg.asset(
    group_name="bronze",
    description="트랜잭셔널 데이터베이스에서 가져온 원시 주문 데이터",
    metadata={"source": "postgres://orders_db", "refresh": "daily"},
)
def raw_orders(database: DatabaseResource) -> pd.DataFrame:
    """OLTP 데이터베이스에서 주문 데이터를 수집한다.

    프라이머리가 아닌 레플리카에서 가져오는 이유?
      - 프로덕션 데이터베이스에 쿼리 부하를 주지 않기 위해
      - 읽기 레플리카는 오래 걸리는 분석 쿼리를 처리할 수 있다
      - 프로덕션 테이블 잠금 위험 없음
    """
    return database.execute_query("""
        SELECT order_id, customer_id, product_id, quantity, unit_price,
               discount, status, created_at, updated_at
        FROM orders
        WHERE updated_at >= CURRENT_DATE - INTERVAL '1 day'
    """)


@dg.asset(
    group_name="bronze",
    description="제품 API에서 가져온 원시 제품 카탈로그",
)
def raw_products() -> pd.DataFrame:
    """제품 카탈로그를 수집한다 (드물게 변경됨)."""
    # 제품을 별도 자산으로 분리하는 이유?
    # - 갱신 빈도가 다르다 (주별 vs 주문은 일별)
    # - 소스 시스템이 다르다 (API vs 데이터베이스)
    # - 독립적 구체화(Independent materialization)
    return pd.DataFrame({
        "product_id": [1, 2, 3],
        "name": ["Widget A", "Widget B", "Premium Widget"],
        "category": ["basic", "basic", "premium"],
        "cost": [10.0, 15.0, 50.0],
    })


# ── 정리된 자산 (Silver 레이어) ─────────────────────────────────

@dg.asset(
    group_name="silver",
    description="제품 정보가 보강된 검증된 주문",
)
def enriched_orders(
    raw_orders: pd.DataFrame,
    raw_products: pd.DataFrame,
) -> pd.DataFrame:
    """주문을 제품 데이터와 조인하고 파생 필드를 계산한다.

    실버 레이어에서 조인하는 이유?
      - Bronze는 재생 능력(replay capability)을 위해 원시 조인되지 않은 데이터를 저장
      - Silver는 "단일 진실 공급원(single source of truth)" 뷰를 만드는 곳
      - 다운스트림 골드 자산이 이 조인을 반복할 필요 없음
    """
    df = raw_orders.merge(raw_products, on="product_id", how="left")

    # 라인 아이템별 합계 금액 계산
    df["line_total"] = df["quantity"] * df["unit_price"] * (1 - df["discount"])

    # 비즈니스 주의를 위한 고가 주문 플래그
    # 왜 $500인가? 이는 구성 가능한 비즈니스 규칙이다
    df["is_high_value"] = df["line_total"] > 500

    return df


# ── 비즈니스 자산 (Gold 레이어) ──────────────────────────────────

@dg.asset(
    group_name="gold",
    description="제품 카테고리별 일일 매출 지표",
)
def daily_category_revenue(enriched_orders: pd.DataFrame) -> pd.DataFrame:
    """BI 대시보드를 위해 카테고리별 매출을 집계한다.

    비즈니스 로직:
      - 완료된 주문만 매출로 계산
      - 매출 = line_total의 합계 (이미 할인 적용됨)
      - AOV(평균 주문 가치) = 매출 / 고유 주문 수
    """
    completed = enriched_orders[enriched_orders["status"] == "completed"]

    metrics = completed.groupby("category").agg(
        total_revenue=("line_total", "sum"),
        order_count=("order_id", "nunique"),
        avg_discount=("discount", "mean"),
    ).reset_index()

    metrics["aov"] = metrics["total_revenue"] / metrics["order_count"]
    return metrics


@dg.asset(
    group_name="gold",
    description="구매 행동에 기반한 고객 세분화",
)
def customer_segments(enriched_orders: pd.DataFrame) -> pd.DataFrame:
    """고객을 가치 계층으로 세분화한다.

    RFM-lite 세분화를 사용하는 이유?
      - 최신성(Recency), 빈도(Frequency), 금액(Monetary) — 표준 리테일 분석
      - 타겟팅 및 리텐션 캠페인에 간단하지만 효과적
      - 골드 레이어는 비즈니스 로직에 적합한 곳
    """
    customer_stats = enriched_orders.groupby("customer_id").agg(
        total_spent=("line_total", "sum"),
        order_count=("order_id", "nunique"),
        last_order=("created_at", "max"),
    ).reset_index()

    # 총 지출에 따른 세분화
    # 이 임계값은 비즈니스에 맞게 조정되어야 한다
    conditions = [
        customer_stats["total_spent"] >= 1000,
        customer_stats["total_spent"] >= 500,
        customer_stats["total_spent"] >= 100,
    ]
    labels = ["platinum", "gold", "silver"]
    customer_stats["segment"] = pd.np.select(conditions, labels, default="bronze")

    return customer_stats
```

### 5.2 전체 자산 그래프

```
                    ┌──────────────┐
                    │ raw_products  │  (bronze)
                    └──────┬───────┘
                           │
┌──────────────┐    ┌──────▼───────────┐    ┌────────────────────────┐
│  raw_orders  │───→│ enriched_orders  │───→│ daily_category_revenue │
│  (bronze)    │    │    (silver)       │    │        (gold)          │
└──────────────┘    └──────┬───────────┘    └────────────────────────┘
                           │
                    ┌──────▼───────────┐
                    │customer_segments │
                    │     (gold)       │
                    └──────────────────┘
```

---

## 6. 파티션 자산과 증분 처리

### 6.1 파티션이 필요한 이유

프로덕션에서는 매 실행마다 전체 데이터셋을 구체화하는 경우가 드뭅니다. 대신, **파티션(partitions)** — 일반적으로 날짜, 지역, 또는 카테고리 기준 — 으로 데이터를 처리합니다. 이를 통해:

- **증분 처리(Incremental processing)**: 새 데이터/변경된 데이터만 처리 (실행당 $O(n)$ 대신 $O(\Delta n)$)
- **타겟 백필(Targeted backfills)**: 모든 것을 건드리지 않고 특정 날짜 범위만 재처리
- **병렬 실행(Parallel execution)**: 다른 파티션을 동시에 실행 가능
- **비용 절감(Cost control)**: 필요한 것만 처리 (클라우드에서 특히 중요)

### 6.2 시간 파티션 자산

```python
import dagster as dg
import pandas as pd
from datetime import datetime

# 일일 파티션 체계 정의
# DailyPartitionsDefinition을 사용하는 이유?
# - 대부분의 데이터 파이프라인은 일별 배치로 운영된다
# - 각 파티션은 하루치 데이터를 나타낸다
# - Dagster가 파티션별 구체화 상태를 추적한다

daily_partitions = dg.DailyPartitionsDefinition(
    start_date="2024-01-01",
    # start_date를 설정하는 이유?
    # - Dagster가 인식할 가장 이른 파티션을 정의한다
    # - 아주 오래된 시점까지 실수로 백필되는 것을 방지
    # - 데이터 소스가 데이터를 수집하기 시작한 시점과 일치
)


@dg.asset(
    partitions_def=daily_partitions,
    group_name="bronze",
    description="주문 날짜별로 파티션된 일일 주문 수집",
)
def daily_raw_orders(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """특정 날짜 파티션의 주문을 수집한다.

    수집 시 파티션을 나누는 이유?
      - 소스 쿼리가 제한됨: WHERE date = '2024-01-15'
      - 전체 테이블을 다시 읽을 필요 없음
      - 특정 날짜 백필 = 해당 파티션 하나만 재실행
    """
    # context.partition_key가 날짜 문자열을 제공한다, 예: "2024-01-15"
    partition_date = context.partition_key
    context.log.info(f"{partition_date} 주문 수집 중")

    # 프로덕션에서: partition_date로 WHERE 절 쿼리
    return pd.DataFrame({
        "order_id": range(100),
        "amount": [50.0] * 100,
        "order_date": [partition_date] * 100,
    })


@dg.asset(
    partitions_def=daily_partitions,
    group_name="silver",
)
def daily_cleaned_orders(
    context: dg.AssetExecutionContext,
    daily_raw_orders: pd.DataFrame,
) -> pd.DataFrame:
    """특정 파티션의 주문을 정리한다.

    핵심 인사이트: Dagster가 이 자산의 "2024-01-15" 파티션을 구체화할 때,
    업스트림 daily_raw_orders에서 "2024-01-15" 파티션을 자동으로 로드한다.
    파티션 정렬이 자동이다!
    """
    partition_date = context.partition_key
    df = daily_raw_orders.copy()
    df = df[df["amount"] > 0]  # 유효하지 않은 금액 제거
    context.log.info(
        f"{partition_date} {len(df)}개 주문 정리 완료 "
        f"({len(daily_raw_orders) - len(df)}개 유효하지 않은 주문 제거)"
    )
    return df
```

### 6.3 다차원 파티션(Multi-Dimensional Partitions)

복잡한 시나리오를 위해 Dagster는 다차원 파티션을 지원합니다:

```python
import dagster as dg

# 다차원 파티션이 필요한 이유?
# - 데이터가 시간과 카테고리 모두에 따라 달라진다
# - 날짜별, 지역별, 또는 둘 다로 백필 가능
# - 예: 1월 "europe" 데이터 전체 재처리

region_time_partitions = dg.MultiPartitionsDefinition({
    "date": dg.DailyPartitionsDefinition(start_date="2024-01-01"),
    "region": dg.StaticPartitionsDefinition(["us", "europe", "asia"]),
})


@dg.asset(partitions_def=region_time_partitions)
def regional_orders(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """날짜와 지역 모두로 파티션된 주문.

    각 구체화는 하나의 (날짜, 지역) 쌍을 처리한다.
    총 파티션 수 = 날짜 수 × 지역 수. 365일 × 3개 지역 = 1,095개 파티션.
    Dagster가 각각을 독립적으로 추적한다.
    """
    keys = context.partition_key.keys_by_dimension
    date = keys["date"]
    region = keys["region"]
    context.log.info(f"{date} {region} 주문 처리 중")

    return pd.DataFrame({
        "order_id": range(50),
        "region": [region] * 50,
        "date": [date] * 50,
    })
```

### 6.4 파티션 간 매핑

다운스트림 자산이 업스트림 파티션들을 집계하는 경우(예: 일별 데이터에서 주별 지표), 파티션 매핑(partition mapping)을 사용합니다:

```python
import dagster as dg

weekly_partitions = dg.WeeklyPartitionsDefinition(start_date="2024-01-01")

@dg.asset(
    partitions_def=weekly_partitions,
    ins={
        "daily_cleaned_orders": dg.AssetIn(
            partition_mapping=dg.TimeWindowPartitionMapping(
                # TimeWindowPartitionMapping을 사용하는 이유?
                # - 하나의 주별 파티션을 7개의 일별 업스트림 파티션에 매핑
                # - Dagster가 7일치를 자동으로 로드
                start_offset=0,
                end_offset=0,
            ),
        ),
    },
)
def weekly_order_summary(daily_cleaned_orders: pd.DataFrame) -> pd.DataFrame:
    """일별 주문을 주별 요약으로 집계한다.

    이 자산은 7개의 일별 파티션을 읽어 1개의 주별 파티션을 생성한다.
    Dagster가 파티션 매핑을 통해 팬인(fan-in)을 자동으로 처리한다.
    """
    return daily_cleaned_orders.groupby("order_date").agg(
        daily_revenue=("amount", "sum"),
        daily_orders=("order_id", "count"),
    ).reset_index()
```

---

## 7. 센서와 스케줄

### 7.1 스케줄(Schedules) — 시간 기반 트리거

```python
import dagster as dg

# 간단한 Cron 기반 스케줄
daily_refresh = dg.ScheduleDefinition(
    name="daily_asset_refresh",
    # 잡 대신 자산 선택을 타겟으로 하는 이유?
    # - 자산 구체화를 직접 스케줄링
    # - 별도의 잡 객체를 만들 필요 없음
    target=dg.AssetSelection.groups("bronze", "silver", "gold"),
    cron_schedule="0 6 * * *",    # 매일 UTC 오전 6시
    default_status=dg.DefaultScheduleStatus.RUNNING,
)


# 파티션 인식 스케줄
@dg.schedule(
    cron_schedule="0 6 * * *",
    job_name="daily_etl_job",
)
def daily_partition_schedule(context: dg.ScheduleEvaluationContext):
    """전날 파티션을 자동으로 타겟으로 하는 스케줄.

    왜 전날인가? N일의 데이터는 보통 N+1일에 완성된다.
    오전 6시 실행은 늦게 도착하는 이벤트를 위한 여유를 제공한다.
    """
    yesterday = (context.scheduled_execution_time - timedelta(days=1)).strftime("%Y-%m-%d")
    return dg.RunRequest(
        partition_key=yesterday,
        tags={"source": "daily_schedule"},
    )
```

### 7.2 센서(Sensors) — 이벤트 기반 트리거

센서는 외부 조건을 폴링하고 조건이 충족될 때 실행을 트리거합니다.

```python
import dagster as dg
from pathlib import Path

@dg.sensor(
    job_name="ingest_new_files_job",
    minimum_interval_seconds=60,  # 60초마다 폴링
)
def new_file_sensor(context: dg.SensorEvaluationContext):
    """랜딩 존에서 새 파일을 감시하고 수집을 트리거한다.

    스케줄 대신 센서를 사용하는 이유?
      - 데이터가 예측할 수 없는 시간에 도착한다 (파트너 업로드, API 푸시)
      - 이벤트 기반: 데이터가 나타나면 즉시 처리
      - 고정 스케줄로 "혹시 몰라" 실행하는 것보다 효율적
    """
    landing_zone = Path("/data/landing/orders/")
    last_mtime = float(context.cursor or "0")

    new_files = []
    max_mtime = last_mtime

    # 수정 시간(mtime)을 커서로 사용하면 처리된 파일의 별도 데이터베이스를
    # 유지할 필요가 없다. 트레이드오프: 파일이 동일한 mtime으로 제자리에서
    # 수정되면 재처리되지 않는다.
    for f in landing_zone.glob("*.csv"):
        if f.stat().st_mtime > last_mtime:
            new_files.append(str(f))
            max_mtime = max(max_mtime, f.stat().st_mtime)

    if new_files:
        context.log.info(f"{len(new_files)}개의 새 파일 발견")
        # 커서는 센서 틱(tick) 간에 영속적으로 유지된다. 데몬이 재시작되어도
        # 커서는 마지막으로 커밋된 값에서 재개되므로 재스캔이 필요 없다.
        context.update_cursor(str(max_mtime))
        # run_key는 중복을 방지한다: 같은 키가 다시 생성되더라도(예: 런이
        # 시작되기 전에 센서 틱이 두 번 실행되는 경우) Dagster가 두 번째
        # 런을 시작하는 대신 중복을 건너뛴다.
        yield dg.RunRequest(
            run_key=f"new_files_{max_mtime}",
            run_config={
                "ops": {
                    "ingest": {"config": {"files": new_files}}
                }
            },
        )


# 신선도 기반 센서: 자산이 오래됐을 때 트리거
@dg.freshness_policy_sensor(
    asset_selection=dg.AssetSelection.groups("gold"),
)
def freshness_sensor(context: dg.FreshnessPolicySensorContext):
    """오래된 골드 자산을 자동으로 구체화한다.

    이것이 Dagster의 선언적(DECLARATIVE) 스케줄링이다:
      - "오전 6시에 실행"이 아닌 "이 자산은 2시간 이내의 것이어야 한다"고 말한다
      - Dagster가 신선도를 확인하고 필요시 구체화를 트리거한다
    """
    pass  # 센서 프레임워크가 로직을 처리한다
```

---

## 8. Dagster에서의 테스트

Dagster의 가장 강력한 장점 중 하나는 테스트 스토리입니다. 자산은 순수한 Python 함수입니다 — 오케스트레이션 인프라 없이 `pytest`로 테스트할 수 있습니다.

### 8.1 자산 단위 테스트

```python
import pytest
import pandas as pd

# 자산 테스트는 함수를 호출하는 것만큼 간단하다!

def test_cleaned_orders():
    """cleaned_orders가 완료되지 않은 주문을 제거하는지 테스트한다."""
    # Arrange: 테스트 입력 생성
    raw = pd.DataFrame({
        "order_id": [1, 2, 3],
        "customer_id": [101, 102, 103],
        "amount": [100.0, 200.0, 50.0],
        "status": ["completed", "refunded", "completed"],
        "created_at": pd.date_range("2024-01-01", periods=3),
    })

    # Act: 자산 함수를 직접 호출
    # 이것이 가능한 이유? @asset은 그냥 Python 함수이기 때문!
    result = cleaned_orders(raw)

    # Assert
    assert len(result) == 2              # 환불된 주문 제거
    assert "refunded" not in result["status"].values
    assert result["amount"].dtype == float


def test_order_metrics_aggregation():
    """order_metrics가 고객별로 올바르게 집계하는지 테스트한다."""
    cleaned = pd.DataFrame({
        "order_id": [1, 2, 3],
        "customer_id": [101, 101, 102],
        "amount": [100.0, 200.0, 50.0],
        "status": ["completed"] * 3,
        "created_at": pd.date_range("2024-01-01", periods=3),
    })

    result = order_metrics(cleaned)

    assert len(result) == 2  # 두 명의 고유 고객
    cust_101 = result[result["customer_id"] == 101].iloc[0]
    assert cust_101["total_orders"] == 2
    assert cust_101["total_revenue"] == 300.0
    assert cust_101["avg_order_value"] == 150.0
```

### 8.2 리소스를 사용한 테스트 (모킹)

```python
import dagster as dg

def test_revenue_report_with_mock_resources():
    """외부 리소스에 의존하는 자산을 테스트한다.

    리소스를 모킹하는 이유?
      - 테스트에 실제 데이터베이스나 S3 버킷이 불필요
      - 테스트가 초 단위가 아닌 밀리초 단위로 실행
      - 결정론적: 네트워크 문제로 인한 불안정한 테스트 없음
    """
    # mock 리소스 생성
    mock_db = MockDatabaseResource(data=[{"revenue": 1000}])
    mock_s3 = MockS3Resource()

    # mock 리소스로 자산 실행
    result = dg.materialize(
        assets=[revenue_report],
        resources={
            "database": mock_db,
            "s3": mock_s3,
        },
    )

    assert result.success
    assert mock_s3.last_written_path == "reports/revenue/latest.parquet"


def test_full_pipeline_integration():
    """전체 자산 그래프를 엔드투엔드로 테스트한다.

    통합 테스트에 materialize()를 사용하는 이유?
      - 올바른 순서로 전체 자산 그래프를 실행한다
      - IO 매니저를 사용하여 자산 간 데이터를 전달한다
      - 개별 자산이 아닌 전체 파이프라인을 검증한다
    """
    result = dg.materialize(
        assets=[raw_orders, cleaned_orders, order_metrics],
        resources={
            "io_manager": dg.mem_io_manager,
            # mem_io_manager를 사용하는 이유?
            # - 자산을 메모리에 저장한다 (디스크 I/O 없음)
            # - 빠른 통합 테스트에 완벽하다
            # - 정리(cleanup)가 필요 없다
        },
    )

    assert result.success
    # 모든 자산이 구체화됐는지 확인
    assert result.output_for_node("raw_orders") is not None
    assert result.output_for_node("order_metrics") is not None
```

---

## 9. Dagster Cloud vs OSS 배포

### 9.1 배포 옵션

```python
"""
Dagster 배포 비교:

┌─────────────────────────────────────────────────────────────────────┐
│                    Dagster OSS (자체 호스팅)                         │
├─────────────────────────────────────────────────────────────────────┤
│ 직접 관리해야 할 컴포넌트:                                            │
│ ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│ │ Dagster   │  │ Dagster   │  │PostgreSQL│  │ 사용자 코드│           │
│ │ Webserver │  │ Daemon    │  │ (스토리지)│  │ (gRPC)   │           │
│ └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
│                                                                     │
│ 장점:                          단점:                                 │
│ ✓ 완전한 제어                  ✗ 운영 부담                            │
│ ✓ 벤더 종속 없음               ✗ 스케일링 직접 처리                   │
│ ✓ 무료                         ✗ 내장 알림/인사이트 없음              │
│ ✓ 에어갭(air-gapped) 환경      ✗ 업그레이드 직접 관리                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    Dagster Cloud (관리형)                            │
├─────────────────────────────────────────────────────────────────────┤
│ Dagster 관리:       │  사용자 관리:                                   │
│ ┌──────────┐        │  ┌──────────────┐                              │
│ │ Webserver │        │  │ 사용자 코드  │ (내 인프라에서 실행)         │
│ │ Daemon    │        │  │ (하이브리드  │                              │
│ │ Storage   │        │  │  또는 서버리스)│                             │
│ │ Monitoring│        │  └──────────────┘                              │
│ └──────────┘        │                                                │
│                                                                     │
│ 배포 모드:                                                            │
│ 1. 하이브리드: 에이전트가 내 클라우드에서 실행, 코드는 내 VPC에 유지  │
│ 2. 서버리스: Dagster가 모든 것을 실행 (가장 간단)                    │
│                                                                     │
│ 장점:                          단점:                                 │
│ ✓ 운영 오버헤드 제로            ✗ 비용 (컴퓨팅 사용량 과금)            │
│ ✓ 내장 알림                    ✗ 인프라 제어 감소                     │
│ ✓ 브랜치 배포                  ✗ 인터넷 연결 필요                     │
│ ✓ 인사이트 & 비용 추적                                                │
└─────────────────────────────────────────────────────────────────────┘
"""
```

### 9.2 Docker Compose를 사용한 OSS 배포

```yaml
# docker-compose.yaml for Dagster OSS
# Docker Compose를 사용하는 이유? 소규모 팀과 dev/staging 환경에 적합하다.
# 프로덕션 대규모 배포에는 Kubernetes 사용 (Helm 차트: dagster/dagster).

version: "3.8"
services:
  # 웹서버는 Dagster UI(자산 그래프, 런 이력, 로그)를 제공한다.
  # 상태가 없어(stateless) 로드 밸런서 뒤에서 수평 확장이 가능하다.
  dagster-webserver:
    image: dagster/dagster-k8s:latest
    command: ["dagster-webserver", "-h", "0.0.0.0", "-p", "3000"]
    ports:
      - "3000:3000"
    environment:
      DAGSTER_PG_URL: "postgresql://dagster:dagster@postgres:5432/dagster"
    depends_on:
      - postgres

  # 데몬은 스케줄, 센서, 런 큐잉을 처리한다. 웹서버와 달리
  # 데몬은 단일 인스턴스여야 한다 — 여러 데몬을 실행하면
  # 스케줄 트리거 중복 및 센서 평가 중복이 발생한다.
  dagster-daemon:
    image: dagster/dagster-k8s:latest
    command: ["dagster-daemon", "run"]
    environment:
      DAGSTER_PG_URL: "postgresql://dagster:dagster@postgres:5432/dagster"
    depends_on:
      - postgres

  # Dagster는 런 메타데이터, 이벤트 로그, 스케줄 상태를 Postgres에 저장한다.
  # SQLite는 로컬 개발에서 지원되지만 동시 쓰기 안전성이 부족하므로,
  # 웹서버 + 데몬을 함께 실행할 때는 항상 Postgres를 사용할 것.
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: dagster
      POSTGRES_PASSWORD: dagster
      POSTGRES_DB: dagster
    volumes:
      - dagster-pg-data:/var/lib/postgresql/data

volumes:
  dagster-pg-data:
```

---

## 10. dbt, Spark, pandas와의 통합

### 10.1 dagster-dbt: 1등급 dbt 지원

Dagster의 dbt 통합은 가장 강력한 셀링 포인트 중 하나입니다. 각 dbt 모델이 자동으로 Dagster 자산이 되어, 통합된 계보 그래프(unified lineage graph)를 만듭니다.

```python
from dagster_dbt import DbtCliResource, dbt_assets
from dagster import AssetExecutionContext, Definitions
from pathlib import Path

# dbt 프로젝트를 지정
DBT_PROJECT_DIR = Path(__file__).parent / "dbt_project"

# dbt 매니페스트를 파싱하는 이유?
# - Dagster가 dbt의 manifest.json을 읽어 모든 모델을 발견한다
# - 각 모델이 자동으로 Dagster 자산이 된다
# - 모델 간 의존성이 자산 의존성이 된다
# - dbt 테스트가 Dagster 자산 체크가 된다

@dbt_assets(manifest=DBT_PROJECT_DIR / "target" / "manifest.json")
def my_dbt_assets(context: AssetExecutionContext, dbt: DbtCliResource):
    """Dagster 자산으로서의 모든 dbt 모델.

    수동으로 dbt를 실행하는 대신 @dbt_assets를 사용하는 이유?
      - 통합 계보: Python 자산 + dbt 모델이 하나의 그래프에
      - 선택적 실행: 특정 dbt 모델 구체화
      - 관측: Dagster가 dbt 모델의 신선도를 추적
    """
    yield from dbt.cli(["build"], context=context).stream()


# Python 자산을 dbt 자산과 결합
@dg.asset(
    deps=["stg_orders"],  # 이것은 dbt 모델 이름이다!
    description="dbt 스테이징 테이블에서 파생된 ML 피처",
)
def ml_features(database: DatabaseResource) -> pd.DataFrame:
    """dbt로 변환된 데이터에서 ML 피처를 계산한다.

    Python과 dbt를 혼합하는 이유?
      - dbt는 SQL 변환(스테이징, 조인, 집계)에 탁월하다
      - Python은 ML 피처 엔지니어링(임베딩, 커스텀 로직)에 탁월하다
      - Dagster가 두 가지를 단일 의존성 그래프로 통합한다
    """
    staged = database.execute_query("SELECT * FROM stg_orders")
    return compute_features(pd.DataFrame(staged))
```

### 10.2 Spark 통합

```python
from dagster_spark import SparkResource
import dagster as dg

@dg.asset(
    group_name="silver",
    description="Spark를 통한 대규모 주문 정리",
)
def spark_cleaned_orders(
    context: dg.AssetExecutionContext,
    spark: SparkResource,
) -> None:
    """대규모 데이터셋에 Spark를 사용하여 주문을 정리한다.

    이 자산에 Spark를 사용하는 이유?
      - 데이터 볼륨이 단일 머신 메모리를 초과할 때 (>10 GB)
      - 분산 조인이나 집계가 필요할 때
      - Dagster가 오케스트레이션; Spark가 무거운 작업을 처리
    """
    session = spark.spark_session

    raw = session.read.parquet("s3://data-lake/bronze/orders/")

    # 여러 필터를 AND로 합치지 않고 체이닝하는 것은 기능적으로 동일하지만
    # 가독성이 더 좋다. Spark의 옵티마이저(Catalyst)가 자동으로 이를
    # 단일 조건부 푸시다운(predicate pushdown)으로 병합한다.
    cleaned = (
        raw
        .filter("status = 'completed'")
        .filter("amount > 0")
        # order_id에서 dropDuplicates는 키당 임의 행을 유지한다.
        # 최신 버전이 필요하다면 중복 제거 전에 orderBy를 추가하거나
        # 윈도우 함수를 대신 사용할 것.
        .dropDuplicates(["order_id"])
        .withColumn("processed_at", dg.F.current_timestamp())
    )

    # mode("overwrite")는 전체 Silver 테이블을 교체한다. 증분 처리에는
    # Delta Lake MERGE를 사용하는 것이 더 낫지만 — 데이터셋이 단일 Spark
    # 작업에 맞는 경우(<100 GB)에는 전체 덮어쓰기도 허용된다.
    cleaned.write.mode("overwrite").parquet("s3://data-lake/silver/orders/")
    context.log.info(f"{cleaned.count()}개의 정리된 주문을 기록했습니다")
```

---

## 요약

```
Dagster 핵심 개념:
─────────────────────
Asset           = "어떤 데이터가 존재해야 하는가?"
Op              = "어떤 계산이 실행돼야 하는가?"
Graph           = "Ops들이 어떻게 연결되는가?"
Job             = "그래프를 언제/어떻게 실행하는가?"
Resource        = "어떤 인프라가 필요한가?"
IO Manager      = "자산 데이터를 어떻게/어디에 저장하는가?"
Partition       = "증분 처리를 위해 데이터를 어떻게 분할하는가?"
Sensor          = "외부 이벤트에 언제 반응해야 하는가?"
Schedule        = "타이머에 따라 언제 실행해야 하는가?"
Definitions     = "내 Dagster 프로젝트의 전체 레지스트리"

Airflow 대신 Dagster를 선택해야 할 때:
  ✓ 데이터 중심 파이프라인 (분석, ML 피처, ELT)
  ✓ 강력한 테스트 요구사항
  ✓ dbt 통합 필요
  ✓ 소규모 팀의 그린필드 프로젝트
  ✓ 자산 신선도 추적이 중요한 경우

Airflow를 유지해야 할 때:
  ✓ 기존 Airflow 투자가 큰 경우
  ✓ 범용 태스크 오케스트레이션 (비데이터)
  ✓ 광범위한 프로바이더 생태계 필요 (1000+ 오퍼레이터)
  ✓ 팀이 이미 Airflow 전문가인 경우
```

---

## 연습 문제

### 연습 1: 자신만의 자산 파이프라인 구축

블로그 분석 플랫폼을 위한 Dagster 자산 파이프라인을 만드세요:

1. mock 페이지 뷰 데이터를 생성하는 `raw_page_views` 소스 자산을 정의하세요 (컬럼: `page_url`, `user_id`, `timestamp`, `session_id`)
2. 봇 트래픽을 제거하고("bot"을 포함하는 user agent) 세션별로 중복을 제거하는 `cleaned_page_views` 자산을 정의하세요
3. 고유 방문자 수로 페이지를 순위 매기는 `page_popularity` 골드 자산을 정의하세요
4. 각 자산에 메타데이터와 설명을 추가하세요
5. `pytest`를 사용하여 세 가지 자산을 모두 테스트하세요

### 연습 2: 파티션된 증분 파이프라인

연습 1을 시간 파티셔닝으로 확장하세요:

1. `raw_page_views`와 `cleaned_page_views`에 `DailyPartitionsDefinition`을 추가하세요
2. 일별 자산에서 `TimeWindowPartitionMapping`을 사용하여 `WeeklyPartitionsDefinition`을 가진 `weekly_page_report` 자산을 만드세요
3. UTC 오전 2시에 전날 파티션을 구체화하는 스케줄을 작성하세요
4. `dagster dev`를 사용하여 지난 7일치 백필을 시뮬레이션하세요

### 연습 3: 리소스 주입과 테스트

의존성 주입을 연습하세요:

1. `PostgresResource`와 `S3Resource` (mock 구현체)를 만드세요
2. Postgres에서 읽고 S3에 쓰는 자산 `user_profiles`를 만드세요
3. mock 리소스를 사용하여 자산 로직을 검증하는 테스트를 작성하세요
4. `mem_io_manager`와 함께 `dg.materialize()`를 사용하는 두 번째 테스트를 작성하세요

### 연습 4: Dagster-dbt 통합 설계

하이브리드 파이프라인을 (코드 또는 종이 위에) 설계하세요:

1. dbt 처리: 스테이징 모델 (`stg_users`, `stg_events`), 마트 (`fct_user_activity`)
2. Python 처리: ML 피처 계산, 모델 스코어링
3. dbt 자산과 Python 자산 모두를 보여주는 자산 의존성 그래프를 그리세요
4. 어떤 자산이 파티션되어야 하는지, 그 이유를 파악하세요

### 연습 5: 센서 기반 파이프라인

이벤트 기반 수집 파이프라인을 구축하세요:

1. 새 CSV 파일을 위한 디렉토리를 감시하는 센서를 만드세요
2. 새 파일이 나타나면, 데이터를 수집, 정리, 집계하는 잡을 트리거하세요
3. `context.cursor`를 사용하여 처리된 파일을 추적하세요
4. 엣지 케이스를 처리하세요: 빈 파일, 중복 파일, 잘못된 형식의 CSV

---

## 참고 자료

- [Dagster 공식 문서](https://docs.dagster.io/)
- [Dagster GitHub 저장소](https://github.com/dagster-io/dagster)
- [dagster-dbt 통합 가이드](https://docs.dagster.io/integrations/dbt)
- [소프트웨어 정의 자산 개념](https://docs.dagster.io/concepts/assets/software-defined-assets)
- [Dagster Cloud](https://dagster.io/cloud)
- [Dagster vs Airflow — 공식 비교](https://dagster.io/vs/dagster-vs-airflow)
- [Dagster University (무료 코스)](https://courses.dagster.io/)

---

[← 이전: 19. Lakehouse 실전 패턴](19_Lakehouse_Practical_Patterns.md) | [다음: 21. 데이터 버전 관리와 데이터 계약 →](21_Data_Versioning_and_Contracts.md)
