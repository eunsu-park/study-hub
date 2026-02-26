# 데이터 엔지니어링 개요

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 데이터 엔지니어링(Data Engineering)을 정의하고, 데이터 과학자(Data Scientist) 및 데이터 분석가(Data Analyst)와 비교하여 데이터 엔지니어의 핵심 역할을 설명할 수 있다
2. 데이터 파이프라인(Data Pipeline)의 구성 요소를 설명하고, Python으로 기본 ETL 파이프라인을 구현할 수 있다
3. 배치 처리(Batch Processing)와 스트리밍 처리(Stream Processing)를 비교하고, 주어진 사용 사례에 적합한 방식을 선택할 수 있다
4. 데이터 웨어하우스(Data Warehouse), 데이터 레이크(Data Lake), 람다(Lambda), 카파(Kappa) 아키텍처 등 주요 데이터 아키텍처 패턴을 설명할 수 있다
5. 데이터 엔지니어링 생태계의 핵심 도구를 파악하고 클라우드 서비스와 매핑할 수 있다
6. 멱등성(Idempotency), 원자성(Atomicity), 재시도(Retry) 로직을 포함한 파이프라인 설계 모범 사례를 적용할 수 있다

---

## 개요

데이터 엔지니어링은 조직의 데이터를 수집, 저장, 처리, 전달하는 시스템을 설계하고 구축하는 분야입니다. 데이터 엔지니어는 데이터 파이프라인을 구축하여 원시 데이터를 분석 가능한 형태로 변환합니다.

---

## 1. 데이터 엔지니어의 역할

### 1.1 핵심 책임

```
┌─────────────────────────────────────────────────────────────┐
│                    데이터 엔지니어 역할                        │
├─────────────────────────────────────────────────────────────┤
│  1. 데이터 수집 (Ingestion)                                   │
│     - 다양한 소스에서 데이터 추출                               │
│     - API, 데이터베이스, 파일, 스트리밍                         │
│                                                             │
│  2. 데이터 저장 (Storage)                                     │
│     - Data Lake, Data Warehouse 설계                         │
│     - 스키마 설계 및 최적화                                    │
│                                                             │
│  3. 데이터 변환 (Transformation)                              │
│     - ETL/ELT 파이프라인 구축                                 │
│     - 데이터 품질 보장                                        │
│                                                             │
│  4. 데이터 전달 (Serving)                                     │
│     - 분석가/과학자에게 데이터 제공                            │
│     - BI 도구, API, 대시보드 연동                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 데이터 엔지니어 vs 데이터 과학자 vs 데이터 분석가

| 역할 | 주요 업무 | 필요 기술 |
|------|----------|----------|
| **데이터 엔지니어** | 파이프라인 구축, 인프라 관리 | Python, SQL, Spark, Airflow, Kafka |
| **데이터 과학자** | 모델 개발, 예측 분석 | Python, ML/DL, 통계, 수학 |
| **데이터 분석가** | 비즈니스 인사이트 도출 | SQL, BI 도구, 시각화, 통계 |

### 1.3 데이터 엔지니어 필수 기술

```python
# 데이터 엔지니어 기술 스택 예시
# 벤더 묶음이 아닌 기능별로 분류 — 팀이 소유권을 분할하는 방식을 반영한다
# (예: "플랫폼" 팀이 인프라를, "데이터" 팀이 오케스트레이션+처리를 담당).
# 카테고리를 알면 클라우드 공급자 간에 이전 가능한 기술을 파악하는 데 도움이 된다.
tech_stack = {
    "programming": ["Python", "SQL", "Scala", "Java"],
    "databases": ["PostgreSQL", "MySQL", "MongoDB", "Redis"],
    "big_data": ["Spark", "Hadoop", "Flink", "Hive"],
    "orchestration": ["Airflow", "Prefect", "Dagster"],
    "streaming": ["Kafka", "Kinesis", "Pub/Sub"],
    "cloud": ["AWS", "GCP", "Azure"],
    "infrastructure": ["Docker", "Kubernetes", "Terraform"],
    "storage": ["S3", "GCS", "HDFS", "Delta Lake"]
}
```

---

## 2. 데이터 파이프라인 개념

### 2.1 파이프라인이란?

데이터 파이프라인은 데이터를 소스에서 목적지까지 이동시키는 일련의 처리 단계입니다.

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Source  │ → │  Extract │ → │Transform │ → │   Load   │
│ (소스)   │    │  (추출)  │    │ (변환)   │    │ (적재)   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
     ↓               ↓               ↓               ↓
  Database        Raw Data      Cleaned Data    Warehouse
  API, Files      Staging       Processed       Analytics
```

### 2.2 파이프라인 구성 요소

```python
# 간단한 파이프라인 예시
from datetime import datetime
import pandas as pd

class DataPipeline:
    """기본 데이터 파이프라인 클래스"""

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None

    def extract(self, source: str) -> pd.DataFrame:
        """데이터 추출 단계"""
        print(f"[{datetime.now()}] Extracting from {source}")
        # 실제로는 DB, API, 파일 등에서 데이터 추출
        data = pd.read_csv(source)
        return data

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 변환 단계"""
        print(f"[{datetime.now()}] Transforming data")
        # 대체 삽입이 아닌 삭제 선택: 이 단순 파이프라인은 상위 시스템이 부분
        # 데이터를 처리한다고 가정하며, 삭제는 채울 값을 추측할 필요 없이
        # 로직을 멱등성(idempotent)으로 유지한다.
        df = df.dropna()
        # 처리 시간을 기록해 하위 소비자가 오래된 데이터를 감지하고
        # 같은 소스 파일에 대한 두 번의 실행을 구분할 수 있게 한다.
        df['processed_at'] = datetime.now()
        return df

    def load(self, df: pd.DataFrame, destination: str):
        """데이터 적재 단계"""
        print(f"[{datetime.now()}] Loading to {destination}")
        # CSV 대신 Parquet 선택: 열 지향(columnar) 형식은 압축이 더 잘 되고
        # 데이터 타입을 보존하여 재로드 시 타입 추론 문제를 방지한다.
        df.to_parquet(destination, index=False)

    def run(self, source: str, destination: str):
        """전체 파이프라인 실행"""
        self.start_time = datetime.now()
        print(f"Pipeline '{self.name}' started")

        # 순차적 E→T→L: 각 단계가 이전 단계의 출력을 받으므로
        # 단계 사이에 검증이나 체크포인트를 추가하기 쉽다.
        raw_data = self.extract(source)
        transformed_data = self.transform(raw_data)
        self.load(transformed_data, destination)

        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).seconds
        print(f"Pipeline completed in {duration} seconds")


# 파이프라인 실행
if __name__ == "__main__":
    pipeline = DataPipeline("daily_sales")
    pipeline.run("sales_raw.csv", "sales_processed.parquet")
```

### 2.3 파이프라인 유형

| 유형 | 설명 | 사용 사례 |
|------|------|----------|
| **배치 (Batch)** | 정해진 시간에 대량 데이터 처리 | 일일 보고서, 월간 집계 |
| **스트리밍 (Streaming)** | 실시간 데이터 처리 | 실시간 대시보드, 이상 탐지 |
| **마이크로배치 (Micro-batch)** | 짧은 간격의 작은 배치 | 준실시간 분석 (5-15분) |
| **이벤트 기반 (Event-driven)** | 특정 이벤트 발생 시 처리 | 트리거 기반 처리 |

---

## 3. 배치 처리 vs 스트리밍 처리

### 3.1 배치 처리 (Batch Processing)

```python
# 배치 처리 예시: 일일 매출 집계
from datetime import datetime, timedelta
import pandas as pd

def daily_sales_batch():
    """일일 매출 배치 처리"""

    # 전날 데이터 처리: 하루가 완전히 끝나야만 집계할 수 있으며,
    # 그렇지 않으면 합계가 부분적이어서 오해를 불러일으킨다.
    yesterday = datetime.now() - timedelta(days=1)
    date_str = yesterday.strftime('%Y-%m-%d')

    # 소스 DB 수준에서 사전 집계하여 데이터 전송을 최소화 —
    # 원시 트랜잭션 대신 집계된 행만 네트워크를 통해 전송된다.
    query = f"""
    SELECT
        product_id,
        SUM(quantity) as total_quantity,
        SUM(amount) as total_amount
    FROM sales
    WHERE DATE(created_at) = '{date_str}'
    GROUP BY product_id
    """

    # 날짜 파티션 출력 파일로 멱등성(idempotent) 재실행 가능:
    # 같은 날짜로 재실행해도 같은 파일을 덮어쓸 뿐이다.
    print(f"Processing batch for {date_str}")
    # df = execute_query(query)
    # df.to_parquet(f"sales_summary_{date_str}.parquet")

    return {"status": "success", "date": date_str}

# 배치 처리 특징
batch_characteristics = {
    "latency": "높음 (분~시간)",
    "throughput": "높음 (대량 처리에 효율적)",
    "use_cases": ["일일 보고서", "주간 집계", "데이터 마이그레이션"],
    "tools": ["Spark", "Airflow", "dbt", "AWS Glue"]
}
```

### 3.2 스트리밍 처리 (Stream Processing)

```python
# 스트리밍 처리 예시: 실시간 이벤트 처리
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Any
import json

@dataclass
class Event:
    """스트리밍 이벤트"""
    event_type: str
    data: dict
    timestamp: datetime

class StreamProcessor:
    """간단한 스트림 프로세서"""

    def __init__(self):
        # 이벤트 타입당 여러 핸들러: 옵저버(observer) 패턴을 사용하면
        # 기존 핸들러 코드를 수정하지 않고 새로운 반응(알림, 로깅, 메트릭)을
        # 추가할 수 있다 — 다운타임 비용이 큰 스트리밍에서 매우 중요하다.
        self.handlers: dict[str, list[Callable]] = {}

    def register_handler(self, event_type: str, handler: Callable):
        """이벤트 핸들러 등록"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    def process(self, event: Event):
        """이벤트 처리"""
        # 팬아웃(fan-out): 등록된 모든 핸들러가 같은 이벤트에 대해 실행되어
        # 독립적인 처리 경로(예: 로그 AND 알림 동시 실행)를 구현한다.
        handlers = self.handlers.get(event.event_type, [])
        for handler in handlers:
            handler(event)

    def consume(self, stream):
        """스트림에서 이벤트 소비 (시뮬레이션)"""
        for message in stream:
            # 타임스탬프는 소스 시간이 아닌 소비 시점에 부여 — 프로덕션에서는
            # 클럭 편차(clock-skew) 문제를 피하기 위해 소스 이벤트 타임스탬프를 사용한다.
            event = Event(
                event_type=message['type'],
                data=message['data'],
                timestamp=datetime.now()
            )
            self.process(event)


# 핸들러 예시
def log_handler(event: Event):
    """이벤트 로깅"""
    print(f"[{event.timestamp}] {event.event_type}: {event.data}")

def alert_handler(event: Event):
    """이상 탐지 알림"""
    if event.data.get('amount', 0) > 10000:
        print(f"ALERT: High value transaction detected!")

# 스트리밍 특징
streaming_characteristics = {
    "latency": "낮음 (밀리초~초)",
    "throughput": "중간 (레코드 단위)",
    "use_cases": ["실시간 대시보드", "이상 탐지", "알림"],
    "tools": ["Kafka", "Flink", "Spark Streaming", "Kinesis"]
}
```

### 3.3 배치 vs 스트리밍 비교

| 특성 | 배치 처리 | 스트리밍 처리 |
|------|----------|--------------|
| **지연 시간** | 분~시간 | 밀리초~초 |
| **데이터 처리량** | 대량 | 소량/연속 |
| **복잡성** | 상대적 단순 | 상대적 복잡 |
| **재처리** | 용이 | 어려움 |
| **비용** | 저렴 | 고가 |
| **사용 사례** | 보고서, 집계 | 실시간 분석, 알림 |

---

## 4. 데이터 아키텍처 패턴

### 4.1 전통적인 데이터 웨어하우스 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│                  전통적 Data Warehouse 아키텍처                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐   ┌─────────┐   ┌─────────────────────────┐    │
│  │ Source 1│   │ Source 2│   │       Source N          │    │
│  │  (ERP)  │   │  (CRM)  │   │      (Other)            │    │
│  └────┬────┘   └────┬────┘   └───────────┬─────────────┘    │
│       │             │                     │                  │
│       └─────────────┼─────────────────────┘                  │
│                     ↓                                        │
│           ┌─────────────────┐                                │
│           │   ETL Process   │                                │
│           │ (Extract-Transform-Load)                         │
│           └────────┬────────┘                                │
│                    ↓                                         │
│           ┌─────────────────┐                                │
│           │  Data Warehouse │                                │
│           │   (Star Schema) │                                │
│           └────────┬────────┘                                │
│                    ↓                                         │
│           ┌─────────────────┐                                │
│           │    BI Tools     │                                │
│           │ (Tableau, Power BI)                              │
│           └─────────────────┘                                │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 모던 데이터 레이크 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│                  Modern Data Lake 아키텍처                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Sources                                                     │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                    │
│  │ API │ │ DB  │ │ IoT │ │ Log │ │Files│                    │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘                    │
│     └───────┴───────┴───────┴───────┘                        │
│                     ↓                                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    Data Lake                         │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │    │
│  │  │  Bronze │→│  Silver │→│  Gold   │              │    │
│  │  │   Raw   │  │ Cleaned │  │Curated │              │    │
│  │  └─────────┘  └─────────┘  └─────────┘             │    │
│  └─────────────────────────────────────────────────────┘    │
│                     ↓                                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │BI/Reports│ │ ML/AI    │ │ Data Apps│ │ API      │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└──────────────────────────────────────────────────────────────┘
```

### 4.3 람다 아키텍처 (Lambda Architecture)

배치와 스트리밍을 결합한 하이브리드 아키텍처입니다.

> **람다 vs 카파 선택 기준?**
> - **람다**: 스키마 변경 후 과거 집계 수정 등 이력 재처리와 실시간 결과가 동시에 필요할 때 사용한다. 배치 레이어는 스피드 레이어의 근사 결과가 수렴하는 "정보의 원천(source of truth)" 역할을 한다.
> - **카파**: 스트리밍 우선 방식으로 충분하고 이벤트 로그가 재실행 가능한 경우(예: 보존 기간이 긴 Kafka)에 사용한다. 카파는 같은 로직에 대해 두 개의 별도 코드베이스를 유지하는 운영 부담을 피할 수 있다.

```python
# 람다 아키텍처 개념 구현
class LambdaArchitecture:
    """람다 아키텍처: 배치 + 스트리밍 레이어"""

    def __init__(self):
        # 두 병렬 레이어는 근본적인 긴장을 해소한다: 배치는 *정확한* 결과(전체
        # 데이터셋 재계산)를, 스피드는 *적시* 결과(서브초 지연)를 제공한다.
        # 둘 중 하나만으로는 정확성과 실시간성 모두를 원하는 사기 탐지
        # 대시보드 같은 사용 사례를 충족할 수 없다.
        self.batch_layer = BatchLayer()
        self.speed_layer = SpeedLayer()
        self.serving_layer = ServingLayer()

    def ingest(self, data):
        """데이터 수집: 두 레이어에 동시 전달"""
        # 이중 쓰기(dual-write): 같은 이벤트가 두 레이어 모두에 공급되어 동기화를 유지한다.
        # 배치 레이어는 재처리를 위한 불변 마스터 사본을 저장하고,
        # 스피드 레이어는 저지연 쿼리를 위해 즉시 처리한다.
        self.batch_layer.append(data)
        self.speed_layer.process(data)

    def query(self, params):
        """쿼리: 배치 뷰 + 실시간 뷰 병합"""
        batch_result = self.serving_layer.get_batch_view(params)
        realtime_result = self.speed_layer.get_realtime_view(params)

        # 병합 로직: 배치 뷰는 마지막 배치 실행까지의 모든 데이터를,
        # 실시간 뷰는 그 이후의 간격만 커버한다. 병합하면 쿼리 결과가
        # *완전하면서도* *최신* 상태를 유지한다.
        return self.merge_views(batch_result, realtime_result)


class BatchLayer:
    """배치 레이어: 전체 데이터셋 처리"""

    def append(self, data):
        """마스터 데이터셋에 추가"""
        # 추가 전용(append-only) 불변 저장소: 나중에 비즈니스 로직이 변경되더라도
        # 원시 이벤트 이력을 보존하여 전체 재계산이 가능하다.
        pass

    def compute_batch_views(self):
        """배치 뷰 계산 (주기적 실행)"""
        # *전체* 데이터셋을 재계산 — 비용이 크지만 정확성을 보장한다.
        # 일반적으로 마스터 데이터셋에서 Spark나 MapReduce를 사용하여
        # 스케줄(예: 시간/일별)에 따라 실행된다.
        pass


class SpeedLayer:
    """스피드 레이어: 실시간 데이터 처리"""

    def process(self, data):
        """실시간 처리"""
        # 증분 업데이트만 — 빠르지만 근사값일 수 있다.
        # 배치 레이어가 따라잡으면 그 결과가 스피드 레이어를 대체하므로,
        # 근사 오류는 자동으로 수정된다.
        pass

    def get_realtime_view(self, params):
        """실시간 뷰 반환"""
        pass


class ServingLayer:
    """서빙 레이어: 쿼리 처리"""

    def get_batch_view(self, params):
        """배치 뷰 반환"""
        # 저지연 저장소(예: Cassandra, HBase)에서 사전 계산된 배치 결과를 제공 —
        # 무거운 계산은 배치 실행 시 이미 완료되었으므로 읽기가 빠르다.
        pass
```

### 4.4 카파 아키텍처 (Kappa Architecture)

스트리밍만 사용하는 단순화된 아키텍처입니다.

```
┌──────────────────────────────────────────────────────────────┐
│                    Kappa Architecture                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Sources                                                     │
│  ┌─────┐ ┌─────┐ ┌─────┐                                    │
│  │Event│ │Event│ │Event│                                    │
│  └──┬──┘ └──┬──┘ └──┬──┘                                    │
│     └───────┴───────┘                                        │
│             ↓                                                │
│  ┌─────────────────────────────────────┐                    │
│  │         Message Queue (Kafka)       │                    │
│  │         - Event Log                 │                    │
│  │         - Replayable                │                    │
│  └─────────────────┬───────────────────┘                    │
│                    ↓                                         │
│  ┌─────────────────────────────────────┐                    │
│  │      Stream Processing Layer        │                    │
│  │      (Flink, Spark Streaming)       │                    │
│  └─────────────────┬───────────────────┘                    │
│                    ↓                                         │
│  ┌─────────────────────────────────────┐                    │
│  │          Serving Layer              │                    │
│  │    (Database, Cache, API)           │                    │
│  └─────────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. 데이터 엔지니어링 도구 생태계

### 5.1 주요 도구 분류

```python
# 벤더 묶음이 아닌 기능 레이어별로 구성 — 이는 실제 데이터 플랫폼을
# 구축하는 방식을 반영한다. 레이어당 하나의 도구를 선택하고
# 전체 설계를 바꾸지 않고도 벤더를 교체할 수 있다.
data_engineering_tools = {
    "orchestration": {
        # 배치 오케스트레이터는 DAG를 스케줄링하고, 스트리밍 오케스트레이터는
        # 장시간 실행되는 토폴로지를 관리한다 — 실패/재시도 모델이 다르다.
        "batch": ["Apache Airflow", "Prefect", "Dagster", "Luigi"],
        "streaming": ["Apache Kafka", "Apache Flink", "Spark Streaming"]
    },
    "processing": {
        "batch": ["Apache Spark", "Apache Hive", "Presto/Trino"],
        "streaming": ["Apache Kafka Streams", "Apache Flink", "Apache Storm"]
    },
    "storage": {
        # 레이크 vs 웨어하우스 vs DB는 지연/비용/유연성 간의 트레이드오프:
        # 레이크는 원시 데이터에 가장 저렴하고, 웨어하우스는 분석을 최적화하며,
        # OLTP 데이터베이스는 저지연 애플리케이션 읽기를 담당한다.
        "data_lake": ["S3", "GCS", "HDFS", "Azure Blob"],
        "data_warehouse": ["Snowflake", "BigQuery", "Redshift", "Databricks"],
        "databases": ["PostgreSQL", "MySQL", "MongoDB", "Cassandra"]
    },
    "transformation": {
        # SQL 기반 도구(dbt)는 분석가가 Python 없이 변환을 소유할 수 있게 하고,
        # 코드 기반 도구(PySpark)는 순수 SQL에서 다루기 어려운 ML 피처 엔지니어링이나
        # 복잡한 비즈니스 로직을 처리한다.
        "sql_based": ["dbt", "SQLMesh"],
        "code_based": ["PySpark", "Pandas", "Polars"]
    },
    "quality": {
        "testing": ["Great Expectations", "dbt tests", "Soda"],
        "monitoring": ["Monte Carlo", "Datadog", "Grafana"]
    },
    "catalog": ["Apache Atlas", "DataHub", "Amundsen", "OpenMetadata"]
}
```

### 5.2 클라우드 서비스 매핑

| 기능 | AWS | GCP | Azure |
|------|-----|-----|-------|
| **오케스트레이션** | Step Functions, MWAA | Cloud Composer | Data Factory |
| **스트리밍** | Kinesis | Pub/Sub, Dataflow | Event Hubs |
| **배치 처리** | EMR, Glue | Dataproc, Dataflow | HDInsight |
| **Data Lake** | S3 + Lake Formation | GCS + BigLake | ADLS + Synapse |
| **Data Warehouse** | Redshift | BigQuery | Synapse Analytics |

---

## 6. 데이터 엔지니어링 모범 사례

### 6.1 파이프라인 설계 원칙

```python
# 좋은 파이프라인 설계 원칙
# 위반이 프로덕션 장애를 일으키는 빈도 순으로 나열:
# 멱등성 실패는 중복 데이터를, 원자성 실패는 부분 로드를,
# 모니터링 부재는 며칠간 인지하지 못하는 무소음 장애를 야기한다.
pipeline_best_practices = {
    "idempotency": "같은 입력에 같은 결과 보장",
    "atomicity": "전체 성공 또는 전체 실패",
    "incremental": "증분 처리로 효율성 확보",
    "monitoring": "모든 단계에서 모니터링",
    "error_handling": "실패 시 재시도 및 알림",
    "documentation": "코드와 문서화 함께 관리"
}

# 멱등성(Idempotency) 예시
def idempotent_upsert(df, table_name, key_columns):
    """멱등성을 보장하는 upsert 함수"""
    # INSERT만 사용하는 대신 DELETE 후 INSERT: 파이프라인을 재실행하면
    # (예: 부분 실패 후) 이 방식은 중복 행을 방지한다.
    # MERGE/UPSERT도 대안이 되지만, DELETE+INSERT가 대부분의 SQL 방언에서
    # 동일하게 작동하여 더 이해하기 쉽다.
    delete_query = f"""
    DELETE FROM {table_name}
    WHERE (key1, key2) IN (
        SELECT DISTINCT key1, key2 FROM staging_table
    )
    """
    # execute(delete_query)
    # insert_dataframe(df, table_name)
    pass
```

### 6.2 에러 처리와 재시도

```python
import time
from functools import wraps
from typing import Callable, Type

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    # 프로덕션에서는 특정 예외만 잡는다(예: ConnectionError).
    # TypeError 같은 프로그래밍 버그는 재시도하지 않도록 방지한다.
    exceptions: tuple[Type[Exception], ...] = (Exception,)
):
    """재시도 데코레이터"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        print(f"Attempt {attempt} failed: {e}")
                        # 선형 백오프(delay * attempt): 일시적 장애가 해소될 시간을
                        # 주면서 이미 어려움을 겪는 서비스에 부하를 주지 않는다.
                        # 진정한 지수 백오프는 delay * (2 ** attempt) + 랜덤 지터 사용.
                        time.sleep(delay * attempt)
            raise last_exception
        return wrapper
    return decorator


# max_attempts=3, delay=2.0 → 2초 후, 4초 후 재시도 후 포기.
# 최악의 경우 총 대기: 6초. 빠른 복구와 이미 어려움을 겪는
# API 과부하 방지 사이에서 균형을 맞춘다.
@retry(max_attempts=3, delay=2.0)
def fetch_data_from_api(url: str):
    """API에서 데이터 가져오기 (재시도 포함)"""
    import requests
    # timeout=30: API가 살아있지만 느린 경우 파이프라인이 무기한 걸리는 것을 방지.
    # 30초는 대부분의 REST API에 충분히 넉넉하다.
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()
```

---

## 연습 문제

### 문제 1: 파이프라인 설계
온라인 쇼핑몰의 일일 매출 리포트를 생성하는 파이프라인을 설계하세요.

```python
# 풀이 예시
class DailySalesReportPipeline:
    def extract(self):
        """주문, 상품, 고객 데이터 추출"""
        pass

    def transform(self):
        """매출 집계, 카테고리별 분석"""
        pass

    def load(self):
        """리포트 테이블 적재"""
        pass
```

### 문제 2: 배치 vs 스트리밍 선택
다음 사례에서 배치와 스트리밍 중 적합한 방식을 선택하고 이유를 설명하세요:
- 일일 판매 보고서 생성
- 실시간 재고 부족 알림
- 월간 고객 세그먼테이션

---

## 요약

| 개념 | 설명 |
|------|------|
| **데이터 파이프라인** | 소스에서 목적지까지 데이터 이동 및 변환 |
| **배치 처리** | 대량 데이터를 주기적으로 처리 |
| **스트리밍 처리** | 실시간으로 데이터 처리 |
| **Data Lake** | 원시 데이터를 저장하는 저장소 |
| **Data Warehouse** | 정제된 데이터를 저장하는 분석용 저장소 |
| **ETL/ELT** | 데이터 추출, 변환, 적재 프로세스 |

---

## 참고 자료

- [Fundamentals of Data Engineering (O'Reilly)](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108298/)
- [The Data Engineering Cookbook](https://github.com/andkret/Cookbook)
- [Data Engineering Weekly Newsletter](https://dataengineeringweekly.com/)
