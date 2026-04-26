[← 이전: 2. 데이터 모델링 기초](02_Data_Modeling_Basics.md) | [다음: 4. Apache Airflow 기초 →](04_Apache_Airflow_Basics.md)

# ETL vs ELT

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. ETL(Extract, Transform, Load)과 ELT(Extract, Load, Transform) 패턴을 설명하고, 두 방식의 근본적인 차이점을 기술할 수 있다
2. Python에서 pandas와 SQLAlchemy를 사용하여 ETL 파이프라인을 구현할 수 있다
3. 데이터 웨어하우스 내에서 SQL 기반 변환을 활용한 ELT 파이프라인을 구현할 수 있다
4. 성능, 확장성, 비용, 사용 사례 등의 측면에서 ETL과 ELT를 비교할 수 있다
5. 데이터 볼륨, 변환 복잡도, 인프라 조건을 고려하여 ETL과 ELT 중 적합한 방식을 선택할 수 있다
6. 실제 시나리오를 분석하고 주어진 데이터 엔지니어링 문제에 맞는 패턴을 선택할 수 있다

---

## 이론과 원리

ETL과 ELT 논쟁은 두 글자의 자리바꿈처럼 보이지만, 실제로는 *컴퓨트가 어디에 사는가* 의 자리바꿈입니다. ETL은 변환을 웨어하우스 *이전* 의 전용 서버에 두고, ELT는 변환을 웨어하우스 *내부* 에 둡니다. 이 단일 변경은 클라우드 웨어하우스가 컴퓨트를 저렴하고 탄력적으로 만든 결과이며, 팀이 코드를 어떻게 구조화하는지, 누가 변환을 소유하는지, 어떤 실패 모드가 나타나는지로 연쇄됩니다.

- **(A) 컴퓨트 지역성(Compute Locality) 질문** — 2015-2018년에 무엇이 바뀌어 기본값이 ETL에서 ELT로 뒤집혔는가
- **(B) Schema-on-write vs schema-on-read** — 변환이 구조를 강제하는 시점이 누가 비용을 지불하는지를 결정
- **(C) 멱등 적재 패턴(Idempotent Loading Patterns)** — 전체 새로고침(full refresh), 증분 추가(incremental append), merge/upsert, 그리고 재시도 안전성에 대한 함의
- **(D) 데이터 계보(Data Lineage)와 수정 비용 곡선** — 오류를 늦게 잡는 것이 기하급수적으로 비싼 이유

### A. 컴퓨트 지역성 질문

30년 동안 기본값은 ETL이었습니다. 전용 ETL 서버(Informatica, DataStage, Talend)가 소스에서 추출하고, 자체 메모리와 CPU에서 변환하고, *깨끗하고 모델링된* 데이터를 웨어하우스로 적재했습니다. 웨어하우스는 큐레이션된 결과만 저장했습니다.

ETL이 역사적으로 이긴 이유:

- **웨어하우스 컴퓨트가 부족하고 비쌌음.** Teradata, Oracle Exadata는 노드-시간당 청구. 웨어하우스 사이클을 원시 데이터 파싱에 쓰는 것은 비싼 자원의 낭비.
- **웨어하우스 저장이 부족했음.** 원시 데이터를 적재하지 않은 이유는 저렴하게 둘 곳이 없었기 때문.
- **스키마 규율.** 큐레이션되고 모델링된 출력만 적재하면 웨어하우스가 깨끗하게 유지됨.

무엇이 바뀌었는가:

- **클라우드 웨어하우스가 저장과 컴퓨트를 분리.** Snowflake, BigQuery, Redshift Spectrum은 저장(GB-월당 센트)과 컴퓨트(초당 탄력적 클러스터)를 별도로 청구.
- **저장이 사실상 무료가 됨.** S3급 저장이 $0.023/GB-월이면 모든 원시 바이트를 던져 넣는 비용은 사실상 없음.
- **컴퓨트가 탄력적이 됨.** 1 TB를 변환해야 하는가? 100노드 클러스터를 띄우고, 30분 비용을 내고, 종료. 고정 용량 ETL 서버는 유물.

새 기본값은 ELT가 되었습니다: 원시 데이터를 레이크/웨어하우스로 던져 넣고, 웨어하우스 내에서 SQL로 변환합니다(흔히 dbt를 통해 — 레슨 12). 이는 아키텍처를 압축합니다: 별도 ETL 서버 없음, 별도 변환 언어 없음, 별도 운영 팀 없음.

#### A.1 ETL이 여전히 이기는 경우

ELT가 항상 옳지는 않습니다. ETL이 여전히 이기는 경우:

- **PII가 적재되기 전에 마스킹되어야 함.** GDPR / HIPAA가 원시 PII가 웨어하우스에 적재되는 것을 금지할 수 있음. 비행 중에 마스킹(ETL); 원시 값은 절대 저장하지 않음.
- **소스 데이터가 거대하지만 일부만 필요함.** 추출 시점에 필터링하면 전체 스트림을 저장하고 변환하는 비용을 피함.
- **변환이 비-SQL 로직을 요구함.** 이미지 처리, 복잡한 Python ML 피처 엔지니어링, dbt가 깔끔하게 표현할 수 없는 모든 것.
- **웨어하우스가 쿼리당 청구함.** BigQuery는 스캔된 바이트당 청구; 원시 테이블을 여러 번 다시 스캔하는 ELT는 한 번 사전 집계하는 ETL보다 비쌀 수 있음.

옳은 프레이밍은 "ETL vs ELT"가 아니라 "각 변환이 어디에 살며, 왜 그런가?"입니다.

### B. Schema-on-Write vs Schema-on-Read

ETL/ELT와 정렬되어 있지만 개념적으로는 독립적인 두 번째 축:

- **Schema-on-write (ETL 기본값):** 적재 시점에 검증하고 구조화. 잘못된 행은 거부됨. 웨어하우스 테이블은 항상 선언된 스키마를 따름. 비용: 경직성 — 새 필드 추가는 스키마 마이그레이션을 요구.
- **Schema-on-read (ELT/lake 기본값):** 원시 바이트를 던져 넣음; 각 소비자가 읽을 때 파싱·검증. 새 필드 추가는 무료(JSON을 그냥 던져 넣기). 비용: 모든 소비자가 파싱 비용을 지불; 잘못된 행은 늦게 드러남.

ELT 파이프라인은 일반적으로 둘을 *결합* 합니다: bronze 계층(원시 레이크)에서는 schema-on-read, silver/gold 계층(dbt가 만드는 모델링된 웨어하우스 테이블)에서는 schema-on-write. 이것이 메달리온 패턴입니다(레슨 1, 11, 19).

### C. 멱등 적재 패턴

ETL이든 ELT든 *어떻게* 적재하는지가 중요합니다. 세 가지 패턴이 지배적이며, 각각 다른 멱등성 속성을 가집니다.

#### C.1 전체 새로고침 (truncate + insert)

```sql
TRUNCATE TABLE silver.dim_customer;
INSERT INTO silver.dim_customer SELECT ... FROM bronze.customer;
```

장점: 자명하게 멱등(재실행 → 동일한 최종 상태). 단순한 SQL.
단점: 1행만 변경되어도 O(N) 작업. truncate 동안 테이블 잠금. 이력 손실(어제 상태의 기록 없음).

소형 차원 테이블(< 수백만 행)에서 재구축이 저렴할 때 사용.

#### C.2 증분 추가 (insert new only)

```sql
INSERT INTO silver.fact_sales
SELECT * FROM bronze.sales WHERE event_date = '{run_date}';
```

장점: O(델타) 작업. 이력 보존.
단점: 날짜로 파티셔닝되고 삽입 전에 파티션이 삭제되지 않으면 멱등하지 않음. 그렇지 않으면 재실행이 중복을 추가.

이를 멱등하게 만드는 패턴:

```sql
DELETE FROM silver.fact_sales WHERE event_date = '{run_date}';
INSERT INTO silver.fact_sales
SELECT * FROM bronze.sales WHERE event_date = '{run_date}';
```

한 트랜잭션 안의 두 문장(또는 테이블 포맷 ACID 보장과 함께) → 재실행이 동일한 상태를 생산.

#### C.3 Merge / upsert (업데이트가 있는 증분)

```sql
MERGE INTO silver.dim_customer AS target
USING (SELECT * FROM bronze.customer WHERE updated_at >= '{last_run}') AS source
ON target.customer_id = source.customer_id
WHEN MATCHED THEN UPDATE SET ...
WHEN NOT MATCHED THEN INSERT ...;
```

장점: 삽입과 업데이트를 처리. 멱등(MERGE는 집합 기반; 재실행이 동일한 최종 상태를 생산). O(델타) 작업.
단점: MERGE 지원 필요(현대 웨어하우스, Delta/Iceberg). 더 복잡한 SQL. 워터마크 로직(`last_run`)이 정확히 추적되어야 함.

대부분의 현대 dbt 증분 모델은 이 패턴을 사용합니다.

#### C.4 워터마크 계약

증분 적재는 "마지막 실행 이후 무엇이 새것인가"를 알아야 합니다. 흔한 두 가지 워터마크:

- **이벤트 타임스탬프:** `WHERE event_at >= '{last_run}'`. 이벤트가 불변이고 이벤트 시간이 소스에서 설정되면 신뢰 가능.
- **소스 측 last-modified:** `WHERE updated_at >= '{last_run}'`. 어떤 변경에서든 소스 행이 `updated_at`을 업데이트해야 함; 삭제는 놓침(이를 위해 CDC 필요 — 레슨 18).

워터마크는 증분 적재를 정확하게 만드는 계약입니다. 잘못 잡으면 데이터를 조용히 놓치거나 중복 적재합니다.

### D. 데이터 계보와 수정 비용 곡선

추출에서 잡힌 파이프라인 오류는 수정 비용이 저렴합니다. 같은 오류가 임원 대시보드 발행 후에 잡히면 평판 비용이 비쌉니다. 수정 비용 곡선은 기하급수적이며, 계보(lineage)가 그것을 압축합니다.

```
비용
  │           ╱─── 프로덕션 대시보드에서 잡힘
  │         ╱
  │       ╱
  │     ╱  ─── 스테이징에서 잡힘
  │   ╱
  │ ╱     ─── 변환에서 잡힘
  │       ─── 추출에서 잡힘
  └─────────────────────────────────── 단계
```

계보 추적 — 어떤 소스 컬럼이 어떤 파생 테이블로 흘러가는지 아는 것 — 은 버그가 발견되었을 때, 재구축이 필요한 모든 다운스트림 테이블을 즉시 식별할 수 있게 합니다. dbt가 자동 생성하는 계보 그래프(레슨 12)와 Dagster의 자산 그래프(레슨 20)는 이를 일급(first-class)으로 만듭니다. 계보가 없으면 영향 반경(blast radius)을 찾는 것은 수동 고고학입니다.

따름정리: 가능한 한 일찍 검증하라. 추출에서 스키마 검사; 각 변환에서 데이터 품질 테스트(Great Expectations, dbt tests); 팩트 메트릭이 주간 평균에서 3시그마 이상 움직이면 알림. 레슨 13 참조.

### From Theory to the Code Below

이어지는 각 절은 위 프레임워크의 한 조각을 운영합니다:

- §1 (ETL)은 §A의 "변환이 웨어하우스 외부에 산다" 패턴을 구체적인 Python으로 보여줍니다.
- §2 (ELT)는 §A의 "변환이 웨어하우스 내부에 산다" 패턴을 SQL로 보여줍니다.
- §3 (비교)는 §A.1 — 각 패턴이 이기는 시점입니다.
- §4 (실세계 시나리오)는 §A 트레이드오프를 프로젝트 결정에 실용적으로 매핑합니다.
- §5 (하이브리드)는 §A가 이진(binary)이 아님을 인정합니다 — 많은 파이프라인이 일부 단계는 ETL, 다른 단계는 ELT를 사용.
- §6 (모범 사례)는 §C(멱등 적재) + §D(계보와 검증)를 구체화합니다.

---

## 개요

ETL(Extract, Transform, Load)과 ELT(Extract, Load, Transform)는 데이터 파이프라인의 두 가지 주요 패턴입니다. 전통적인 ETL은 변환 후 적재하고, 모던 ELT는 적재 후 변환합니다.

---

## 1. ETL (Extract, Transform, Load)

### 1.1 ETL 프로세스

```
┌─────────────────────────────────────────────────────────────┐
│                      ETL Process                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐    ┌──────────────┐    ┌──────────┐         │
│   │ Sources  │ → │ ETL Server   │ → │ Target   │          │
│   │          │    │              │    │ (DW)     │          │
│   │ - DB     │    │ 1. Extract   │    │          │          │
│   │ - Files  │    │ 2. Transform │    │ Clean    │          │
│   │ - APIs   │    │ 3. Load      │    │ Data     │          │
│   └──────────┘    └──────────────┘    └──────────┘         │
│                                                             │
│   변환이 중간 서버에서 수행됨                                  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 ETL 예시 코드

```python
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

class ETLPipeline:
    """전통적인 ETL 파이프라인"""

    def __init__(self, source_conn: str, target_conn: str):
        # 소스와 타겟을 별도 엔진으로 분리: 소스 DB에 실수로 쓰는 것을 방지하고
        # 커넥션 풀링을 독립적으로 만든다.
        self.source_engine = create_engine(source_conn)
        self.target_engine = create_engine(target_conn)

    def extract(self, query: str) -> pd.DataFrame:
        """
        Extract: 소스에서 데이터 추출
        """
        print(f"[Extract] Starting at {datetime.now()}")
        df = pd.read_sql(query, self.source_engine)
        print(f"[Extract] Extracted {len(df)} rows")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform: 데이터 정제 및 변환
        - 이 단계가 ETL 서버에서 수행됨 (리소스 소모)
        """
        print(f"[Transform] Starting at {datetime.now()}")

        # 1. 필수 필드(customer_id, amount)가 없는 행은 삭제하고 선택적 필드(email)는
        # 채운다 — 판매 레코드를 잃는 것이 이메일 플레이스홀더를 갖는 것보다 나쁘다.
        df = df.dropna(subset=['customer_id', 'amount'])
        df['email'] = df['email'].fillna('unknown@example.com')

        # 2. 명시적 타입 변환은 소스 측 스키마 드리프트를 조기에 잡는다:
        # order_date가 예상치 못한 형식으로 도착하면 조용히 손상된 데이터를
        # 적재하는 대신 명확히 실패한다.
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['amount'] = df['amount'].astype(float)

        # 3. 파생 컬럼 생성
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month
        df['day_of_week'] = df['order_date'].dt.dayofweek

        # 4. 비즈니스 로직 적용 — ETL 서버에서 세그멘팅하면 모든 소비자가 동일한
        # 세그먼트 정의를 보게 된다. ELT에서는 각 분석가가 임계값을 다르게 적용할 수 있다.
        df['customer_segment'] = df['total_purchases'].apply(
            lambda x: 'Gold' if x > 10000 else ('Silver' if x > 5000 else 'Bronze')
        )

        # 5. 빠른 실패(fail-fast) 검증: 음수 금액이 하위 매출 보고서를 오염시키는 것보다
        # 파이프라인을 중단하는 편이 더 저렴하다.
        assert df['amount'].min() >= 0, "Negative amounts found"

        print(f"[Transform] Transformed {len(df)} rows")
        return df

    def load(self, df: pd.DataFrame, table_name: str):
        """
        Load: 타겟 데이터 웨어하우스에 적재
        """
        print(f"[Load] Starting at {datetime.now()}")

        # 전체 새로고침(if_exists='replace'): 단순하고 멱등성이 있지만,
        # 대형 테이블에는 적합하지 않다 — 그런 경우 증분 upsert를 사용한다.
        # chunksize=10000은 DB 메모리 한계를 초과할 수 있는 거대한 단일
        # INSERT 구문 생성을 방지한다.
        df.to_sql(
            table_name,
            self.target_engine,
            if_exists='replace',
            index=False,
            chunksize=10000
        )

        print(f"[Load] Loaded {len(df)} rows to {table_name}")

    def run(self, source_query: str, target_table: str):
        """ETL 파이프라인 실행"""
        start_time = datetime.now()
        print(f"ETL Pipeline started at {start_time}")

        # E-T-L 순서로 실행
        raw_data = self.extract(source_query)
        transformed_data = self.transform(raw_data)
        self.load(transformed_data, target_table)

        end_time = datetime.now()
        print(f"ETL Pipeline completed in {(end_time - start_time).seconds} seconds")


# 사용 예시
if __name__ == "__main__":
    pipeline = ETLPipeline(
        source_conn="postgresql://user:pass@source-db:5432/sales",
        target_conn="postgresql://user:pass@warehouse:5432/analytics"
    )

    pipeline.run(
        source_query="""
            SELECT
                o.order_id,
                o.customer_id,
                c.email,
                o.order_date,
                o.amount,
                c.total_purchases
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            WHERE o.order_date >= CURRENT_DATE - INTERVAL '1 day'
        """,
        target_table="fact_daily_orders"
    )
```

### 1.3 ETL 도구

| 도구 | 유형 | 특징 |
|------|------|------|
| **Informatica** | 상용 | 엔터프라이즈급, GUI 기반 |
| **Talend** | 오픈소스/상용 | Java 기반, 다양한 커넥터 |
| **SSIS** | 상용 (MS) | SQL Server 통합 |
| **Pentaho** | 오픈소스 | 경량, 사용 편의 |
| **Apache NiFi** | 오픈소스 | 데이터 플로우, 실시간 |

---

## 2. ELT (Extract, Load, Transform)

### 2.1 ELT 프로세스

```
┌─────────────────────────────────────────────────────────────┐
│                      ELT Process                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │ Sources  │ → │ Load Raw     │ → │ Transform    │      │
│   │          │    │ (Data Lake)  │    │ (in DW)      │      │
│   │ - DB     │    │              │    │              │      │
│   │ - Files  │    │ Raw Zone     │    │ SQL/Spark    │      │
│   │ - APIs   │    │ (as-is)      │    │ (DW 리소스)  │      │
│   └──────────┘    └──────────────┘    └──────────────┘     │
│                                                             │
│   변환이 타겟 시스템(DW/Lake)에서 수행됨                       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 ELT 예시 코드

```python
import pandas as pd
from datetime import datetime

class ELTPipeline:
    """모던 ELT 파이프라인"""

    def __init__(self, source_conn: str, warehouse_conn: str):
        self.source_conn = source_conn
        self.warehouse_conn = warehouse_conn

    def extract_and_load(self, source_query: str, raw_table: str):
        """
        Extract & Load: 원본 데이터를 그대로 적재
        - 변환 없이 raw 데이터를 빠르게 적재
        """
        print(f"[Extract & Load] Starting at {datetime.now()}")

        # 소스에서 데이터 추출
        df = pd.read_sql(source_query, self.source_conn)

        # Raw 테이블에 그대로 적재 (변환 없음)
        df.to_sql(
            raw_table,
            self.warehouse_conn,
            if_exists='replace',
            index=False
        )

        print(f"[Extract & Load] Loaded {len(df)} rows to {raw_table}")

    def transform_in_warehouse(self, transform_sql: str):
        """
        Transform: 웨어하우스 내에서 SQL로 변환
        - DW의 컴퓨팅 파워 활용
        - SQL 기반 변환 (dbt 등 사용)
        """
        print(f"[Transform] Starting at {datetime.now()}")

        # 웨어하우스에서 SQL 실행
        with self.warehouse_conn.connect() as conn:
            conn.execute(transform_sql)

        print(f"[Transform] Transformation completed")


# dbt 모델 예시 (SQL 기반 변환)
# 이 SQL은 위 Python ETL transform()과 *같은* 로직을 적용한다는 점에 주목 —
# 핵심 차이점은 실행 *위치*: 웨어하우스 엔진 내부에서 MPP(대규모 병렬 처리) 컴퓨팅을 활용한다.
DBT_MODEL_EXAMPLE = """
-- models/staging/stg_orders.sql
-- dbt를 사용한 ELT 변환

WITH source AS (
    -- {{ source() }} 매크로는 계통 추적과 신선도 검사를 제공하여
    -- raw 테이블이 업데이트되지 않으면 dbt가 알림을 보낼 수 있다.
    SELECT * FROM {{ source('raw', 'orders_raw') }}
),

cleaned AS (
    SELECT
        order_id,
        customer_id,
        COALESCE(email, 'unknown@example.com') AS email,
        CAST(order_date AS DATE) AS order_date,
        CAST(amount AS DECIMAL(10, 2)) AS amount,
        total_purchases,
        -- 파생 컬럼은 쿼리 시점에 계산 — ETL 서버 불필요;
        -- 웨어하우스가 여러 노드에서 병렬로 처리한다.
        EXTRACT(YEAR FROM order_date) AS order_year,
        EXTRACT(MONTH FROM order_date) AS order_month,
        EXTRACT(DOW FROM order_date) AS day_of_week,
        -- 비즈니스 로직
        CASE
            WHEN total_purchases > 10000 THEN 'Gold'
            WHEN total_purchases > 5000 THEN 'Silver'
            ELSE 'Bronze'
        END AS customer_segment,
        -- loaded_at은 각 행이 처리된 *시점*을 추적하여
        -- 증분 모델이 새로 도착한 데이터만 처리할 수 있게 한다.
        CURRENT_TIMESTAMP AS loaded_at
    FROM source
    -- 품질 필터를 SQL로 처리: 스테이징 레이어에 진입하기 전 불량 행을 거부하며,
    -- ETL 검증 방식과 동일한 효과다.
    WHERE customer_id IS NOT NULL
      AND amount IS NOT NULL
      AND amount >= 0
)

SELECT * FROM cleaned
"""


# 실제 ELT 파이프라인 (Snowflake/BigQuery 스타일)
# 3계층 아키텍처(raw → staging → mart)는 메달리온(medallion) 패턴을 따른다:
# 각 계층은 데이터 품질을 높이면서 디버깅과 재처리를 위해 원시 사본을 보존한다.
class ModernELTWithSQL:
    """SQL 기반 모던 ELT"""

    def __init__(self, warehouse):
        self.warehouse = warehouse

    def extract_load(self, source: str, target_raw: str):
        """원본 → Raw 레이어"""
        # COPY INTO는 웨어하우스 네이티브 대량 적재 명령 — 클라우드 스토리지에서
        # Parquet 파일을 직접 병렬로 읽기 때문에 행별 INSERT보다 수 배 빠르다.
        copy_sql = f"""
        COPY INTO {target_raw}
        FROM @{source}
        FILE_FORMAT = (TYPE = 'PARQUET')
        """
        self.warehouse.execute(copy_sql)

    def transform_staging(self):
        """Raw → Staging 레이어"""
        # PARSE_JSON + 명시적 변환: raw 계층은 반정형 데이터를 그대로 저장하고,
        # 스테이징 계층은 스키마를 강제 적용하여 타입 불일치를 조기에 잡으면서
        # 재처리를 위해 원시 사본은 건드리지 않는다.
        staging_sql = """
        CREATE OR REPLACE TABLE staging.orders AS
        SELECT
            order_id,
            customer_id,
            PARSE_JSON(raw_data):email::STRING AS email,
            TO_DATE(raw_data:order_date) AS order_date,
            raw_data:amount::NUMBER(10,2) AS amount
        FROM raw.orders_raw
        """
        self.warehouse.execute(staging_sql)

    def transform_mart(self):
        """Staging → Mart 레이어"""
        # Mart는 스테이징 데이터와 디멘전 테이블을 조인하고 분석 컬럼을 추가한다.
        # cumulative_amount 같은 윈도우 함수는 웨어하우스의 분산 컴퓨팅에서
        # 효율적으로 실행 — 외부 처리가 필요 없다.
        mart_sql = """
        CREATE OR REPLACE TABLE mart.fact_orders AS
        SELECT
            o.order_id,
            d.date_sk,
            c.customer_sk,
            o.amount,
            -- 고객별 누적 합계: 전체 팩트 테이블을 다시 스캔하지 않고
            -- 생애가치(lifetime-value) 쿼리를 가능하게 한다.
            SUM(o.amount) OVER (
                PARTITION BY o.customer_id
                ORDER BY o.order_date
            ) AS cumulative_amount
        FROM staging.orders o
        JOIN dim_date d ON o.order_date = d.full_date
        JOIN dim_customer c ON o.customer_id = c.customer_id
        """
        self.warehouse.execute(mart_sql)
```

### 2.3 ELT 도구

| 도구 | 유형 | 특징 |
|------|------|------|
| **dbt** | 오픈소스 | SQL 기반 변환, 테스트, 문서화 |
| **Fivetran** | 상용 | 자동 스키마 관리, 150+ 커넥터 |
| **Airbyte** | 오픈소스 | 커스텀 커넥터, EL 특화 |
| **Stitch** | 상용 | 간편한 설정, SaaS 친화적 |
| **AWS Glue** | 클라우드 | 서버리스, Spark 기반 |

---

## 3. ETL vs ELT 비교

### 3.1 상세 비교

| 특성 | ETL | ELT |
|------|-----|-----|
| **변환 위치** | 중간 서버 | 타겟 시스템 (DW/Lake) |
| **데이터 이동** | 변환된 데이터만 | 원본 데이터 전체 |
| **스키마** | 미리 정의 필요 | 유연 (Schema-on-Read) |
| **처리 속도** | 느림 (중간 처리) | 빠름 (병렬 처리) |
| **비용** | 별도 인프라 필요 | DW 리소스 사용 |
| **유연성** | 낮음 | 높음 (원본 보존) |
| **복잡한 변환** | 적합 | 제한적 |
| **실시간 처리** | 어려움 | 비교적 용이 |

### 3.2 선택 기준

```python
def choose_etl_or_elt(requirements: dict) -> str:
    """ETL/ELT 선택 가이드"""

    # ETL 선호 요인: 변환이 웨어하우스 외부에서 반드시 이루어져야 하는
    # 시나리오 (개인정보, 비-SQL 로직, 또는 레거시 형식 파싱).
    etl_factors = [
        requirements.get('data_privacy', False),      # 적재 전 PII 마스킹 필요
        requirements.get('complex_transforms', False), # SQL로 표현하기 어려운 로직 (ML, NLP)
        requirements.get('legacy_systems', False),    # 고정폭/메인프레임 형식
        requirements.get('small_data', False),        # DW 오버헤드가 정당화되지 않는 소규모
    ]

    # ELT 선호 요인: 웨어하우스 엔진이 병렬화할 수 있고 원시 데이터를
    # 보존해야 하는 시나리오.
    elt_factors = [
        requirements.get('big_data', False),          # 웨어하우스 MPP가 규모를 처리
        requirements.get('cloud_dw', False),          # 쿼리당 과금 컴퓨팅
        requirements.get('data_lake', False),         # Schema-on-read 유연성
        requirements.get('flexible_schema', False),   # 소스가 자주 변경됨
        requirements.get('raw_data_access', False),   # 분석가가 원시 데이터 필요
        requirements.get('sql_transforms', False),    # 변환이 SQL로 표현 가능
    ]

    # 단순 점수 계산 — 실제로는 요인에 가중치를 다르게 적용한다
    # (예: data_privacy는 점수를 완전히 무시하는 하드 제약일 수 있다).
    etl_score = sum(etl_factors)
    elt_score = sum(elt_factors)

    if etl_score > elt_score:
        return "ETL 권장"
    elif elt_score > etl_score:
        return "ELT 권장"
    else:
        return "하이브리드 고려"


# 사용 예시
project_requirements = {
    'big_data': True,
    'cloud_dw': True,  # Snowflake, BigQuery
    'sql_transforms': True,
    'raw_data_access': True
}

recommendation = choose_etl_or_elt(project_requirements)
print(recommendation)  # "ELT 권장"
```

---

## 4. 하이브리드 접근법

### 4.1 ETLT 패턴

```
┌─────────────────────────────────────────────────────────────┐
│                    ETLT (Hybrid) Pattern                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Sources → [E] → [T] → [L] → [T] → Mart                   │
│                    ↑          ↑                             │
│                    │          │                             │
│            Light Transform   Heavy Transform                │
│            (마스킹, 검증)    (집계, 조인)                    │
│            ETL Server        Data Warehouse                 │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 하이브리드 구현 예시

```python
class HybridPipeline:
    """ETL + ELT 하이브리드 파이프라인"""

    def __init__(self, source, staging_area, warehouse):
        self.source = source
        self.staging = staging_area
        self.warehouse = warehouse

    def extract_with_light_transform(self):
        """
        E + Light T: 추출하면서 가벼운 변환 수행
        - PII 마스킹 (개인정보 보호)
        - 기본 데이터 타입 변환
        - 필수 필드 검증
        """
        # PII 마스킹은 데이터가 소스 시스템을 떠나기 전에 반드시 이루어져야 한다:
        # 원시 PII가 웨어하우스에 한번 적재되면 접근 제어가 훨씬 어려워지고
        # GDPR/CCPA 데이터 최소화 요건을 준수하기 어렵다.
        query = """
        SELECT
            order_id,
            -- 소스에서 해시화하여 웨어하우스가 원시 이메일을 보지 못하게 한다;
            -- MD5는 익명화(pseudonymization)에 충분하다 (보안 목적은 아님).
            MD5(customer_email) AS customer_email_hash,
            SUBSTRING(phone, 1, 3) || '****' || SUBSTRING(phone, -4) AS phone_masked,
            -- 기본 타입 변환으로 잘못된 데이터를 조기에 잡아
            -- 웨어하우스 스토리지와 컴퓨팅을 소비하기 전에 처리한다.
            CAST(order_date AS DATE) AS order_date,
            CAST(amount AS DECIMAL(10, 2)) AS amount
        FROM orders
        WHERE amount IS NOT NULL
        """
        return self.source.execute(query)

    def load_to_staging(self, data):
        """L: 스테이징 영역에 적재"""
        self.staging.load(data, 'orders_staging')

    def transform_in_warehouse(self):
        """
        Heavy T: 웨어하우스에서 복잡한 변환
        - 조인, 집계, 윈도우 함수를 DW로 밀어 넣는다.
          웨어하우스는 이런 연산을 많은 노드에서 병렬화할 수 있기 때문 —
          ETL 서버에서 처리하면 단일 머신 병목이 발생한다.
        """
        heavy_transform_sql = """
        CREATE TABLE mart.order_analysis AS
        SELECT
            o.order_date,
            c.customer_segment,
            p.product_category,
            COUNT(*) AS order_count,
            SUM(o.amount) AS total_amount,
            AVG(o.amount) AS avg_order_value,
            -- 7일 롤링 합계: ROWS BETWEEN 6 PRECEDING AND CURRENT ROW는
            -- 정확히 7행을 제공한다 (오늘 + 이전 6일).
            -- 정렬 키가 있는 열 지향 엔진에서 효율적으로 실행된다.
            SUM(o.amount) OVER (
                PARTITION BY c.customer_segment
                ORDER BY o.order_date
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ) AS rolling_7day_amount
        FROM orders_staging o
        JOIN dim_customer c ON o.customer_id = c.customer_id
        JOIN dim_product p ON o.product_id = p.product_id
        GROUP BY o.order_date, c.customer_segment, p.product_category
        """
        self.warehouse.execute(heavy_transform_sql)

    def run(self):
        """하이브리드 파이프라인 실행"""
        # 1단계: ETL — ETL 서버에서 가벼운 변환 (PII 마스킹)
        data = self.extract_with_light_transform()

        # 2단계: 마스킹된 데이터를 스테이징 영역에 적재
        self.load_to_staging(data)

        # 3단계: ELT — DW 엔진에서 무거운 변환
        self.transform_in_warehouse()
```

---

## 5. 실무 사례

### 5.1 ETL 사용 사례

```python
# 사례 1: 개인정보 처리 (GDPR 준수)
# 여기서 ETL이 올바른 선택: PII는 소스 환경을 떠나기 *전에* 마스킹되어야 한다 —
# ELT 방식은 접근 제어가 있더라도 원시 PII가 웨어하우스에 적재되어
# 데이터 최소화 원칙을 위반한다.
class GDPRCompliantETL:
    """GDPR 준수 ETL - 개인정보 마스킹 후 적재"""

    def transform(self, df):
        # ETL 서버에서 마스킹하여 웨어하우스가 원시 PII를 저장하지 않게 한다.
        # 부분 마스킹(마지막 4자리 유지)은 지원 에스컬레이션 시 레코드를 확인할 수 있어
        # 개인정보와 실용성 사이의 균형을 맞춘다.
        df['email'] = df['email'].apply(self.mask_email)
        df['ssn'] = df['ssn'].apply(lambda x: 'XXX-XX-' + x[-4:])
        df['credit_card'] = df['credit_card'].apply(lambda x: '**** **** **** ' + x[-4:])

        # 동의 필터: 옵트인한 사용자의 데이터만 전송한다.
        # 국경 간 전송 *전에* 이 필터를 적용하면
        # GDPR 6조(처리의 적법 근거)를 준수할 수 있다.
        df = df[df['consent_given'] == True]

        return df

    def mask_email(self, email):
        if pd.isna(email):
            return None
        local, domain = email.split('@')
        return local[:2] + '***@' + domain


# 사례 2: 레거시 시스템 통합
# 여기서 ETL이 필요: 메인프레임 고정폭 형식은 현대 웨어하우스에 직접 적재할 수 없다 —
# 먼저 중간 서버에서 구조적 파싱이 필요하다.
class LegacySystemETL:
    """레거시 메인프레임 데이터 통합"""

    def transform(self, raw_data):
        # 고정폭 필드 오프셋은 메인프레임 COBOL 카피북에 정의되어 있다;
        # 1바이트만 어긋나도 모든 하위 필드가 손상된다.
        records = []
        for line in raw_data.split('\n'):
            record = {
                'account_no': line[0:10].strip(),
                'account_type': line[10:12],
                # 메인프레임은 통화를 정수(소수점 없이)로 저장하므로,
                # 실제 달러 금액을 복원하려면 100으로 나눈다.
                'balance': int(line[12:24]) / 100,
                'status': 'A' if line[24:25] == '1' else 'I',
                'date': self.parse_legacy_date(line[25:33])
            }
            records.append(record)
        return pd.DataFrame(records)

    def parse_legacy_date(self, date_str):
        # YYYYMMDD → YYYY-MM-DD
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
```

### 5.2 ELT 사용 사례

```sql
-- 사례 1: dbt를 활용한 이커머스 분석

-- models/staging/stg_orders.sql
WITH raw_orders AS (
    SELECT * FROM {{ source('raw', 'orders') }}
)
SELECT
    order_id,
    customer_id,
    order_date,
    status,
    total_amount
FROM raw_orders
WHERE order_date IS NOT NULL

-- models/marts/core/fct_orders.sql
WITH orders AS (
    SELECT * FROM {{ ref('stg_orders') }}
),
customers AS (
    SELECT * FROM {{ ref('dim_customers') }}
),
products AS (
    SELECT * FROM {{ ref('dim_products') }}
)
SELECT
    o.order_id,
    o.order_date,
    c.customer_segment,
    p.product_category,
    o.total_amount,
    -- DW에서 윈도우 함수 활용
    ROW_NUMBER() OVER (PARTITION BY o.customer_id ORDER BY o.order_date) AS order_sequence,
    LAG(o.order_date) OVER (PARTITION BY o.customer_id ORDER BY o.order_date) AS prev_order_date
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id


-- 사례 2: BigQuery ELT
-- 대용량 로그 분석 (서버리스 처리)
CREATE OR REPLACE TABLE analytics.user_behavior AS
SELECT
    user_id,
    DATE(timestamp) AS event_date,
    event_type,
    COUNT(*) AS event_count,
    COUNTIF(event_type = 'purchase') AS purchase_count,
    SUM(CASE WHEN event_type = 'purchase' THEN revenue ELSE 0 END) AS total_revenue,
    -- 세션 분석 (복잡한 윈도우 함수)
    ARRAY_AGG(
        STRUCT(timestamp, event_type, page_url)
        ORDER BY timestamp
    ) AS event_sequence
FROM raw.events
WHERE DATE(timestamp) = CURRENT_DATE() - 1
GROUP BY user_id, DATE(timestamp), event_type;
```

---

## 6. 도구 선택 가이드

### 6.1 데이터 규모별 권장 도구

| 데이터 규모 | ETL 도구 | ELT 도구 |
|-------------|----------|----------|
| **소규모** (< 1GB) | Python + Pandas | dbt + PostgreSQL |
| **중규모** (1GB-100GB) | Airflow + Python | dbt + Snowflake |
| **대규모** (> 100GB) | Spark | dbt + BigQuery/Databricks |

### 6.2 아키텍처별 권장

```python
# 타겟 시스템의 강점에 맞게 방식을 선택한다:
# 전통적 DW는 쓰기 시 스키마를 강제(ETL이 자연스럽게 맞음),
# 클라우드 DW는 탄력적 컴퓨팅을 제공(ELT가 이를 활용),
# 데이터 레이크는 어떤 형식도 허용(Schema-on-read의 ELT).
architecture_recommendations = {
    "traditional_dw": {
        "approach": "ETL",
        "tools": ["Informatica", "Talend", "SSIS"],
        "reason": "스키마 엄격, 변환 후 적재"
    },
    "cloud_dw": {
        "approach": "ELT",
        "tools": ["dbt", "Fivetran + dbt", "Airbyte + dbt"],
        # 클라우드 DW는 컴퓨팅 초당 과금 — DW 내부에서 변환을 실행하면
        # 별도의 ETL 클러스터 비용을 방지한다.
        "reason": "DW 컴퓨팅 파워 활용, 원본 보존"
    },
    "data_lake": {
        "approach": "ELT",
        "tools": ["Spark", "AWS Glue", "Databricks"],
        "reason": "스키마 유연, 대용량 처리"
    },
    "hybrid": {
        "approach": "ETLT",
        "tools": ["Airflow + dbt", "Prefect + dbt"],
        # 오케스트레이터(Airflow/Prefect)가 가벼운 ETL 변환(마스킹, 검증)을 처리하고,
        # dbt가 DW에서 무거운 SQL 변환을 처리한다.
        "reason": "민감 정보 처리 + DW 변환"
    }
}
```

---

## 연습 문제

### 문제 1: ETL vs ELT 선택
다음 상황에서 ETL과 ELT 중 어떤 방식이 적합한지 선택하고 이유를 설명하세요:
- 일일 100GB의 로그 데이터를 BigQuery에 적재
- 개인정보가 포함된 고객 데이터를 처리

### 문제 2: ELT SQL 작성
Raw 테이블 `raw_orders`에서 일별 매출 집계 테이블을 생성하는 ELT SQL을 작성하세요.

---

## 요약

| 개념 | 설명 |
|------|------|
| **ETL** | 변환 후 적재, 중간 서버에서 처리 |
| **ELT** | 적재 후 변환, 타겟 시스템에서 처리 |
| **ETL 장점** | 데이터 품질 보장, 민감 정보 처리 |
| **ELT 장점** | 빠른 적재, 유연한 스키마, 원본 보존 |
| **하이브리드** | ETL + ELT 조합, 상황에 맞게 선택 |

---

## 참고 자료

- [dbt Documentation](https://docs.getdbt.com/)
- [Modern Data Stack](https://www.moderndatastack.xyz/)
- [ETL vs ELT: The Difference](https://www.fivetran.com/blog/etl-vs-elt)

---

[← 이전: 2. 데이터 모델링 기초](02_Data_Modeling_Basics.md) | [다음: 4. Apache Airflow 기초 →](04_Apache_Airflow_Basics.md)
