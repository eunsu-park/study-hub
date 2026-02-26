[← 이전: 20. Dagster 자산 기반 오케스트레이션](20_Dagster_Asset_Orchestration.md) | [다음: 개요 →](00_Overview.md)

# 데이터 버전 관리와 데이터 계약(Data Versioning and Data Contracts)

## 학습 목표

1. 재현성(Reproducibility), 롤백(Rollback), 감사 추적(Auditability)을 위해 데이터 버전 관리가 왜 필수적인지 이해한다
2. lakeFS의 브랜칭(Branching), 커밋(Committing), 머징(Merging), 비교(Diffing) 개념을 Git처럼 다루는 방법을 익힌다
3. DVC와 lakeFS를 서로 다른 데이터 버전 관리 사례에 맞게 비교한다
4. 데이터 생산자(Producer)와 소비자(Consumer) 사이의 데이터 계약(Data Contract)을 정의하고 적용한다
5. Pydantic, JSON Schema, Avro를 사용해 스키마 계약(Schema Contract)을 구현한다
6. 자동화된 유효성 검사 게이트(Validation Gate)를 갖춘 계약 우선(Contract-First) 데이터 파이프라인을 구축한다
7. 데이터 계약을 데이터 메시(Data Mesh) 원칙 및 도메인 소유권(Domain Ownership)과 연결한다

---

## 개요

현대 데이터 플랫폼은 두 가지 관련된 문제를 겪고 있다. 첫째, **데이터는 기본적으로 가변적(Mutable)**이다 — Spark 잡이 파티션을 덮어쓰면, 명시적으로 보존하지 않는 한 이전 버전은 사라진다. "실행 취소" 버튼도 없고, 변경 기록도 없으며, 오늘의 출력과 어제의 출력을 비교할 방법도 없다. 둘째, **데이터 인터페이스가 암묵적(Implicit)**이다 — 업스트림 팀이 경고 없이 컬럼명, 데이터 타입, 비즈니스 로직을 변경하면 다운스트림 파이프라인이 새벽 3시에 조용히 망가진다.

데이터 버전 관리와 데이터 계약은 이러한 문제에 대한 해답이다. 버전 관리는 데이터에 Git이 코드에 제공하는 것과 동일한 안전망을 제공한다: 브랜치, 커밋, 비교, 롤백. 계약은 데이터 생산자와 소비자 간의 합의를 공식화하여 변경이 의도적이고, 소통되고, 검증되도록 보장한다.

이 두 가지 실천 방식은 함께 취약하고 신뢰 기반의 데이터 생태계를 견고하고 엔지니어링 수준의 플랫폼으로 변환시킨다. 이 레슨에서는 lakeFS로 데이터 레이크를 브랜칭하는 것부터 자동화된 테스트로 스키마 계약을 적용하는 것까지 두 영역 모두에서 실용적인 기술을 쌓을 것이다.

> **비유**: 데이터 계약은 부서 간 SLA(서비스 수준 협약)와 같다 — 재무팀은 합의된 형식으로 정제된 거래 데이터를 전달하겠다고 약속하고, 분석팀은 정확히 무엇을 기대해야 하는지 안다. 재무팀이 형식을 변경해야 할 경우, 업데이트를 제안하고, 양측이 검토하며, 전환이 조율된다. 더 이상 갑작스러운 파손은 없다.

---

## 1. 데이터 버전 관리가 중요한 이유

### 1.1 가변 데이터의 문제점

```python
"""
가변 데이터 문제(The Mutable Data Problem):
═════════════════════════════════════════

시나리오: 일일 ETL 파이프라인이 s3://data-lake/gold/revenue/에 쓴다

월요일:    ETL 실행 → $1.2M 매출 기록 ✓
화요일:    ETL 실행 → $1.5M 매출 기록 ✓ (월요일 데이터 덮어씀)
수요일:    버그 배포 → $0 매출 기록 ✗ (화요일 데이터 덮어씀)
목요일:    버그 발견!

버전 관리 없이는 답할 수 없는 질문들:
  1. 화요일 데이터는 어떻게 생겼나? (없어짐 — 덮어씀)
  2. 버그가 정확히 언제 데이터를 오염시켰나? (기록 없음)
  3. 화요일 버전으로 롤백할 수 있나? (백업이 있다면 수동 복원)
  4. 어떤 다운스트림 대시보드가 $0 데이터를 소비했나? (알 수 없음)

버전 관리 사용 시 (lakeFS, Delta Lake 타임 트래블, Iceberg 스냅샷):
  1. 화요일 데이터 → 커밋 abc123 체크아웃
  2. 버그 도입 → 커밋 간 비교로 정확한 변경 확인
  3. 롤백 → 커밋 abc123으로 되돌리기 (즉시, 제로 복사)
  4. 계보 → 버전 메타데이터가 소비자 추적

버전 관리 비용: ~1-5% 스토리지 오버헤드 (카피-온-라이트(Copy-on-Write) / 메타데이터 전용)
버전 관리 가치: 인시던트당 수 시간~수 일 절약
"""
```

### 1.2 데이터 버전 관리의 활용 사례

| 활용 사례 | 버전 관리 없이 | 버전 관리 사용 시 |
|----------|---------------|-----------------|
| **버그 롤백** | 백업에서 복원 (수 시간) | 커밋 되돌리기 (수 초) |
| **재현 가능한 ML** | "어떤 데이터로 모델 v2를 훈련했나?" | 커밋 `abc123`에 고정 |
| **데이터 변경 A/B 테스트** | 두 파이프라인을 병렬 실행 | 브랜치, 테스트, 머지 |
| **규제 감사** | 수동 스냅샷 내보내기 | 전체 커밋 기록 |
| **스키마 마이그레이션** | 위험한 일괄 업데이트 | 브랜치, 마이그레이션, 검증, 머지 |
| **다팀 협업** | 충돌이 조용히 덮어써짐 | 머지 충돌 감지 |

### 1.3 버전 관리 스펙트럼

모든 버전 관리 방식이 동등하지는 않다. 올바른 선택은 규모와 사용 사례에 따라 다르다.

```python
"""
데이터 버전 관리 방식 (단순에서 포괄적으로):

Level 0: 버전 관리 없음
  └── 데이터를 제자리에 덮어씀. 기록 없음.
  └── 위험: 높음. 복구: 백업 복원 (있다면).

Level 1: 타임스탬프 기반 스냅샷
  └── s3://bucket/table/snapshot=2024-01-15/
  └── 스냅샷마다 전체 복사. 스토리지: $O(n \times s)$ (여기서 $s$ = 스냅샷 수)
  └── 단순하지만 비용이 많이 듦. 비교 기능 없음.

Level 2: 포맷 수준 버전 관리 (Delta Lake, Iceberg)
  └── 트랜잭션 로그가 파일 수준 변경을 추적
  └── 스냅샷 ID를 통한 타임 트래블
  └── 스토리지: $O(n + \Delta)$ (변경된 파일만 저장)
  └── 단일 테이블 범위로 제한

Level 3: 저장소 수준 버전 관리 (lakeFS)
  └── 전체 데이터 레이크에 걸친 Git 방식 브랜치/커밋
  └── 크로스 테이블 원자적 커밋
  └── 스토리지: $O(n + \Delta)$ (카피-온-라이트 방식)
  └── 모든 데이터셋에 걸쳐 전체 비교, 머지, 롤백 가능

Level 4: 전체 계보 + 버전 관리
  └── 데이터 버전 관리 + 생성 방법 추적 (파이프라인 코드 버전)
  └── 도구: lakeFS + Dagster, DVC + Git, Pachyderm
  └── 완전한 재현성 지원: 동일 코드 + 동일 데이터 = 동일 결과
"""
```

---

## 2. lakeFS: 데이터 레이크를 위한 Git

### 2.1 lakeFS 아키텍처

lakeFS는 데이터를 복사하지 않고 기존 오브젝트 스토어(S3, GCS, Azure Blob) 위에 Git 방식의 버전 관리 레이어를 추가한다.

```python
"""
lakeFS 아키텍처:
════════════════

┌─────────────────────────────────────────────────────────────┐
│                        lakeFS Server                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  S3-Compatible│  │  Version     │  │  Merge       │     │
│  │  API Gateway  │  │  Control     │  │  Engine      │     │
│  │  (read/write) │  │  (commits,   │  │  (3-way diff │     │
│  │               │  │   branches)  │  │   + merge)   │     │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘     │
│         │                  │                                 │
│  ┌──────▼──────────────────▼───────┐                       │
│  │        Metadata Store            │                       │
│  │  (PostgreSQL / DynamoDB)         │                       │
│  │  - Branch pointers               │                       │
│  │  - Commit objects                 │                       │
│  │  - Object deduplication index     │                       │
│  └──────────────┬──────────────────┘                       │
└─────────────────┼───────────────────────────────────────────┘
                  │
    ┌─────────────▼─────────────┐
    │   Underlying Object Store  │
    │   (S3 / GCS / Azure Blob / │
    │    MinIO / local)           │
    │                             │
    │   Actual data files live    │
    │   here, unchanged.          │
    │   lakeFS only manages       │
    │   METADATA (pointers).      │
    └────────────────────────────┘

핵심 인사이트: lakeFS는 브랜치/커밋 시 데이터를 복사하지 않는다.
카피-온-라이트(Copy-on-Write) 방식을 사용한다:
  - 브랜치 생성 = 포인터 생성 (즉시, ~0 바이트)
  - 새 데이터 쓰기 = 오브젝트 스토어에 새 파일 + 메타데이터 업데이트
  - 변경되지 않은 파일 = 브랜치 간 공유 (중복 없음)

스토리지 오버헤드 공식:
  전체 스토리지 = 원본 데이터 + 변경된 데이터 (델타)
  아님: 원본 데이터 × 브랜치 수
"""
```

### 2.2 핵심 연산

```python
import lakefs
from lakefs.client import Client

# lakeFS 클라이언트 초기화
# S3 호환 API를 사용하는 이유? 기존 Spark/pandas 코드가 변경 없이 동작.
# 엔드포인트 URL만 s3.amazonaws.com에서 lakefs.example.com으로 바꾸면 됨.

client = Client(
    host="http://localhost:8000",
    username="AKIAIOSFODNN7EXAMPLE",
    password="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
)


# ── 저장소 연산 ─────────────────────────────────────────

def setup_repository():
    """S3로 백업된 lakeFS 저장소 생성.

    저장소가 필요한 이유?
      - 저장소는 하나의 스토리지 네임스페이스(예: S3 버킷 접두사)에 매핑
      - 모든 브랜치, 커밋, 태그는 저장소 안에 존재
      - Git 저장소처럼 생각하되, 데이터 파일용
    """
    repo = lakefs.Repository("analytics-lake", client=client)
    repo.create(
        storage_namespace="s3://my-bucket/analytics-lake/",
        default_branch="main",
    )
    print(f"Repository created: {repo.id}")
    return repo


# ── 브랜칭 ──────────────────────────────────────────────

def create_feature_branch(repo):
    """격리된 실험을 위한 브랜치 생성.

    데이터를 브랜칭하는 이유?
      - 프로덕션에 영향 없이 새 ETL 변환을 테스트
      - 여러 팀이 서로 다른 데이터 변경을 동시에 진행 가능
      - 실패한 실험은 버려짐 (브랜치 삭제), 롤백 불필요
    """
    main = repo.branch("main")
    dev_branch = repo.branch("feature/new-cleaning-logic").create(source_reference="main")
    print(f"Branch created: {dev_branch.id}")

    # 이 시점에서 dev_branch는 main과 동일한 데이터를 가짐
    # 데이터는 복사되지 않음 — 단지 포인터만 생성됨
    return dev_branch


# ── 변경 사항 커밋 ────────────────────────────────────

def write_and_commit(branch):
    """브랜치에 데이터를 쓰고 변경 사항을 커밋.

    명시적 커밋이 필요한 이유?
      - 커밋되지 않은 쓰기는 "스테이징"되어 있지만 다른 브랜치에서 보이지 않음
      - 커밋은 원자적: 커밋의 모든 파일이 함께 성공하거나 실패
      - 커밋 메시지는 데이터가 변경된 이유를 기록 (Git과 동일)
    """
    import io
    import pandas as pd

    # 샘플 데이터 생성
    df = pd.DataFrame({
        "order_id": [1, 2, 3],
        "amount": [100.0, 200.0, 300.0],
        "status": ["completed", "completed", "pending"],
    })

    # 브랜치에 데이터 업로드
    # Parquet을 사용하는 이유? 컬럼형 포맷은 분석 쿼리에 효율적
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    branch.object("orders/daily/2024-01-15.parquet").upload(
        data=buffer,
        content_type="application/octet-stream",
    )

    # 변경 사항 커밋
    commit = branch.commit(
        message="Add orders for 2024-01-15",
        metadata={
            "pipeline": "daily_order_ingestion",
            "source": "postgres_replica",
            "row_count": "3",
        },
    )
    print(f"Committed: {commit.id} - {commit.message}")
    return commit


# ── 비교(Diff) ─────────────────────────────────────────

def compare_branches(repo):
    """두 브랜치 간의 데이터를 비교.

    머지 전에 비교하는 이유?
      - 변경된 내용 정확히 파악 (추가된 파일, 수정된 파일, 삭제된 파일)
      - 프로덕션에 도달하기 전에 의도치 않은 변경 발견
      - 코드 리뷰처럼 데이터 변경 검토 (데이터 PR)
    """
    main = repo.branch("main")
    feature = repo.branch("feature/new-cleaning-logic")

    diff = main.diff(other_ref=feature)

    for change in diff:
        # change.type: 'added', 'removed', 'changed'
        # change.path: 오브젝트 키
        print(f"  [{change.type}] {change.path}")

    return diff


# ── 머징 ────────────────────────────────────────────────

def merge_to_main(repo):
    """검증 후 피처 브랜치를 main에 머지.

    덮어쓰기 대신 머지를 사용하는 이유?
      - main의 커밋 기록 보존
      - 브랜치 생성 이후 main이 업데이트된 경우 충돌 감지
      - 원자적: 전체 머지가 성공하거나 실패 (부분 업데이트 없음)
    """
    feature = repo.branch("feature/new-cleaning-logic")
    main = repo.branch("main")

    try:
        merge_result = feature.merge_into(main)
        print(f"Merge successful: {merge_result}")
    except lakefs.exceptions.ConflictException as e:
        # 충돌을 처리하는 이유?
        # 브랜치를 만든 이후 누군가 main의 같은 파일을 업데이트했다면,
        # lakeFS가 충돌을 감지함 (Git과 동일).
        print(f"Merge conflict: {e}")
        print("Resolve manually: update the conflicting files on the feature branch")


# ── 롤백 ───────────────────────────────────────────────

def rollback_to_commit(repo, commit_id: str):
    """main 브랜치를 특정 커밋으로 롤백.

    전진 수정 대신 롤백을 사용하는 이유?
      - 즉각적인 복구: 프로덕션 데이터가 수 초 내에 복원됨
      - 전진 수정은 시간이 걸림: 조사, 수정, 파이프라인 재실행
      - 롤백은 사용자가 올바른 데이터를 보는 동안 조사할 시간을 확보
    """
    main = repo.branch("main")
    main.revert(parent_number=1, reference=commit_id)
    print(f"Rolled back main to commit {commit_id}")
```

### 2.3 lakeFS와 Spark 통합

```python
"""
lakeFS + Spark 통합:

lakeFS의 장점은 Spark가 표준 S3 API로 읽고 쓴다는 것.
엔드포인트 URL만 바꾸면 됨. 코드 변경 불필요!
"""

from pyspark.sql import SparkSession

# Spark의 S3 엔드포인트를 lakeFS로 지정하면 기존 Spark 잡에 코드 변경이 없다 —
# 엔드포인트 URL만 바뀐다. lakeFS가 S3 호환성을 선택한 이유가 이것이다:
# 이미 S3를 사용하는 팀의 도입 마찰을 없인다.
# path.style.access=true는 lakeFS가 실제 S3에서 사용하는
# 가상 호스팅 방식 URL(bucket.endpoint)을 지원하지 않기 때문에 필요하다.

spark = SparkSession.builder \
    .appName("lakeFS-Spark") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://lakefs:8000") \
    .config("spark.hadoop.fs.s3a.access.key", "AKIAIOSFODNN7EXAMPLE") \
    .config("spark.hadoop.fs.s3a.secret.key", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .getOrCreate()

# URI 형식: s3a://<저장소>/<브랜치>/경로/데이터
# "main"에서 읽으면 항상 최신 커밋 상태를 가져온다.
main_orders = spark.read.parquet("s3a://analytics-lake/main/orders/")

# 슬래시가 포함된 브랜치 이름은 URI에서 ~를 구분자로 사용한다
# (feature~new-cleaning-logic은 "feature/new-cleaning-logic" 브랜치를 나타냄).
# 이는 뒤따르는 객체 경로와의 모호성을 방지한다.
feature_orders = spark.read.parquet("s3a://analytics-lake/feature~new-cleaning-logic/orders/")

# 브랜치 대신 커밋 해시를 참조하면 불변 접근이 가능하다 —
# 브랜치가 진행되더라도 항상 정확히 동일한 데이터를 읽는다.
# 재현 가능한 ML 훈련과 규제 감사에 필수적이다.
historical = spark.read.parquet("s3a://analytics-lake/abc123def/orders/")

# 브랜치에 쓰기는 명시적으로 커밋될 때까지 "스테이징" 상태다
# (해당 브랜치에서만 보임). 이는 실패한 Spark 잡이 다른 사용자에게
# 부분적으로 쓰인 데이터를 노출하지 않는다는 것을 의미한다.
result_df = main_orders.filter("amount > 100")
result_df.write.mode("overwrite").parquet(
    "s3a://analytics-lake/feature~new-cleaning-logic/orders_filtered/"
)
# 이후 lakeFS API를 통해 커밋 (위 참조)
```

---

## 3. DVC vs lakeFS 비교

DVC(Data Version Control)와 lakeFS 모두 데이터를 버전 관리하지만, 서로 다른 사용 사례를 대상으로 한다.

### 3.1 아키텍처 비교

```python
"""
DVC (Data Version Control):
════════════════════════════
- Git 확장: Git의 .dvc 파일이 데이터 파일 해시를 추적
- 데이터는 원격 스토리지에 존재 (S3, GCS, NFS)
- 버전 관리 = .dvc 포인터 파일을 포함하는 Git 커밋
- 최적 용도: ML 실험, 소규모 팀, 모델/데이터셋 버전 관리

  Git 저장소:                원격 스토리지:
  ┌──────────────┐        ┌──────────────┐
  │ data.csv.dvc │──hash─→│ data.csv     │
  │ model.pkl.dvc│──hash─→│ model.pkl    │
  │ pipeline.py  │        │              │
  └──────────────┘        └──────────────┘

lakeFS:
═══════
- 독립형 서버: 오브젝트 스토어를 직접 관리
- Git 의존성 없음 (자체 커밋/브랜치 모델 보유)
- S3 호환 API: Spark/Presto/Trino에 투명하게 동작
- 최적 용도: 데이터 레이크 버전 관리, 다팀, 프로덕션 파이프라인

  lakeFS 서버:              오브젝트 스토어:
  ┌──────────────┐        ┌──────────────┐
  │ branches     │──meta─→│ actual data   │
  │ commits      │        │ files (shared │
  │ merge engine │        │ via CoW)      │
  └──────────────┘        └──────────────┘
"""
```

### 3.2 기능 비교표

| 기능 | DVC | lakeFS |
|------|-----|--------|
| **주요 사용 사례** | ML 실험 | 데이터 레이크 관리 |
| **버전 모델** | Git 기반 (.dvc 파일) | 독립형 (자체 브랜치/커밋) |
| **브랜칭** | Git 브랜치를 통해 | 네이티브, 즉시 (제로 복사) |
| **API** | CLI + Python | S3 호환 + REST + Python |
| **Spark/Trino 통합** | 수동 경로 관리 | 투명 (S3 API) |
| **원자적 다중 파일 커밋** | 예 (Git 경유) | 예 (네이티브) |
| **머지 충돌** | 파일 수준 (Git 경유) | 오브젝트 수준 (3-way 머지) |
| **파이프라인 추적** | `dvc.yaml` 파이프라인 | 외부 (Dagster/Airflow) |
| **데이터 비교** | 제한적 (해시 비교) | 풍부 (파일 수준 + 선택적 내용) |
| **규모** | 수백 개 파일 | 수백만 개 오브젝트 |
| **팀 규모** | 소규모 (1-10명) | 모든 규모 (엔터프라이즈 지원) |
| **배포** | 없음 (클라이언트 전용) | 서버 필요 |
| **비용** | 무료 (오픈소스) | 무료 오픈소스 / lakeFS Cloud |

### 3.3 결정 가이드

```python
"""
DVC를 선택하는 경우:
  ✓ 팀이 이미 Git을 많이 활용하는 경우
  ✓ ML 데이터셋과 모델을 버전 관리할 경우
  ✓ 데이터셋 수가 수백 개 수준 (수백만이 아닌)
  ✓ 파이프라인 재현성이 필요한 경우 (dvc repro)
  ✓ Spark/Trino를 위한 실시간 브랜치 전환이 불필요한 경우

lakeFS를 선택하는 경우:
  ✓ 수백만 개 오브젝트가 있는 데이터 레이크가 있는 경우
  ✓ 여러 팀이 공유 스토리지에 읽고 쓰는 경우
  ✓ Spark/Trino/Presto가 버전 관리된 데이터를 투명하게 읽어야 하는 경우
  ✓ 크로스 테이블 원자적 커밋이 필요한 경우
  ✓ 데이터 파이프라인 CI/CD 테스트를 위한 즉각적인 브랜칭이 필요한 경우
  ✓ 사전 머지 훅(Pre-merge Hook)으로 데이터 품질 게이트를 적용하려는 경우

둘 다 선택하는 경우 (ML에는 DVC, 데이터 레이크에는 lakeFS):
  ✓ ML 팀은 실험 추적 + 모델 버전 관리에 DVC 사용
  ✓ 데이터 엔지니어링 팀은 공유 데이터 레이크에 lakeFS 사용
  ✓ ML 훈련은 Spark를 통해 lakeFS에서 읽고, DVC로 모델 버전 관리
"""
```

---

## 4. 데이터 계약(Data Contracts)

### 4.1 문제: 암묵적 인터페이스

일반적인 데이터 플랫폼에서 데이터 생산자와 소비자 사이의 인터페이스는 암묵적이다 — 오늘 파이프라인이 생성하는 것만으로 정의된다.

```python
"""
암묵적 인터페이스 문제(The Implicit Interface Problem):
═══════════════════════════════════════════════════

팀 A (생산자):                        팀 B (소비자):
┌───────────────────┐                ┌───────────────────┐
│ orders 테이블 쓰기 │                │ orders 테이블 읽기 │
│ 컬럼:              │                │ 기대:              │
│  - order_id (int)  │──── table ───→│  - order_id (int)  │
│  - amount (float)  │               │  - amount (float)  │
│  - status (string) │               │  - status (string) │
└───────────────────┘                └───────────────────┘

1일차: 모든 것이 작동 ✓

30일차: 팀 A가 "amount" → "total_amount"로 이름 변경
  - 팀 A: "컬럼명을 개선했어요!"
  - 팀 B: KeyError: 'amount'로 파이프라인 실패
  - 누군가 알아챌 때까지 6시간 동안 대시보드에 데이터 없음

60일차: 팀 A가 nullable "discount" 컬럼 추가
  - 팀 B의 집계가 모든 지표에 NaN 반환
  - 근본 원인 파악에 3시간 소요

근본 원인: 팀 A가 무엇을 전달하겠다고 약속하는지 정의하는 계약이 없음.
"""
```

### 4.2 데이터 계약이란?

데이터 계약은 데이터 생산자와 소비자 간의 공식적인 합의로, 다음을 명시한다:

```python
"""
데이터 계약 구성 요소:

┌─────────────────────────────────────────────────────────────┐
│                       DATA CONTRACT                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 스키마(SCHEMA)                                          │
│     - 컬럼명, 데이터 타입, 널 허용 여부                     │
│     - 기본 키, 외래 키                                      │
│     - 유효한 값 범위와 열거형(Enum)                         │
│                                                             │
│  2. 의미론(SEMANTICS)                                       │
│     - 각 필드의 비즈니스적 의미                             │
│     - 계산 로직 (예: "amount = quantity × price")           │
│     - 시간적 의미론 (이벤트 시간 vs 처리 시간)             │
│                                                             │
│  3. SLA (서비스 수준 협약)                                  │
│     - 최신성: "소스 변경 후 1시간 내 업데이트"             │
│     - 완전성: "소스 레코드의 >99% 존재"                    │
│     - 가용성: "99.9% 시간 동안 쿼리 가능"                  │
│                                                             │
│  4. 진화 정책(EVOLUTION POLICY)                             │
│     - 변경이 제안되고 검토되는 방법                         │
│     - 하위 호환성 요구 사항                                 │
│     - 사용 중단 타임라인                                    │
│                                                             │
│  5. 소유권(OWNERSHIP)                                       │
│     - 생산자 팀과 연락처                                    │
│     - 소비자 팀 (등록된)                                   │
│     - 위반 시 에스컬레이션 경로                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
"""
```

### 4.3 다양한 형식으로 스키마 계약 구현

```python
# ── 방법 1: Pydantic 모델 (Python 네이티브) ───────────────────
# Pydantic을 사용하는 이유? 타입 안전, 자동 검증, 명확한 에러 메시지.
# 최적 용도: Python 파이프라인, FastAPI 데이터 서비스.

from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderStatus(str, Enum):
    """유효한 주문 상태 — 계약은 오직 이 값들만 보장."""
    PENDING = "pending"
    COMPLETED = "completed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


class OrderRecord(BaseModel):
    """orders 데이터셋의 스키마 계약.

    이 Pydantic 모델 자체가 계약이다. 데이터가 일치하지 않으면
    명확한 에러 메시지와 함께 검증 실패.

    버전: 2.0
    소유자: data-platform-team
    소비자: analytics, ML, finance
    """
    order_id: int = Field(..., gt=0, description="Unique order identifier")
    customer_id: int = Field(..., gt=0, description="Customer FK")
    amount: float = Field(..., ge=0, description="Order total in USD")
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")
    status: OrderStatus
    created_at: datetime
    updated_at: Optional[datetime] = None

    # 필드 검증기를 사용하는 이유? 비즈니스 규칙을 계약에 인코딩.
    # 스키마는 타입뿐만 아니라 의미론적 정확성도 보장.
    @field_validator("amount")
    @classmethod
    def amount_must_be_reasonable(cls, v: float) -> float:
        """비즈니스 규칙: $1M 이상의 주문은 수동 검토 필요."""
        if v > 1_000_000:
            raise ValueError(f"Amount ${v:,.2f} exceeds $1M limit — needs review")
        return round(v, 2)


def validate_dataframe(df, model_class):
    """DataFrame의 모든 행을 Pydantic 계약에 대해 검증.

    파이프라인 경계에서 검증하는 이유?
      - 조기 실패: 데이터가 대시보드에 퍼지기 전에 문제 발견
      - 명확한 에러 메시지: 어떤 행, 어떤 필드, 무엇이 문제인지
      - 빠른 실패: 파이프라인 실패가 다운스트림 손상 데이터보다 낫다
    """
    errors = []
    valid_records = []

    for idx, row in df.iterrows():
        try:
            record = model_class(**row.to_dict())
            valid_records.append(record.model_dump())
        except Exception as e:
            errors.append({"row": idx, "error": str(e)})

    if errors:
        error_rate = len(errors) / len(df)
        print(f"Validation: {len(errors)}/{len(df)} rows failed ({error_rate:.1%})")
        # 임계값이 필요한 이유? 일부 불량 레코드는 예상됨 (데이터 품질)
        # 그러나 >5% 실패 시 체계적인 문제가 있는 것
        if error_rate > 0.05:
            raise ValueError(
                f"Contract violation: {error_rate:.1%} error rate exceeds 5% threshold"
            )

    return valid_records, errors
```

```python
# ── 방법 2: JSON Schema (언어 독립적) ───────────────────
# JSON Schema를 사용하는 이유? Python, Java, Go, JavaScript에서 동작.
# 최적 용도: 크로스 팀 계약, API 경계, 스키마 레지스트리.

import json
from jsonschema import validate, ValidationError

# JSON Schema 계약은 언어 독립적이다 — 동일한 스키마 파일을 Python,
# Java, Go, JavaScript 소비자 모두가 검증할 수 있다. 이 때문에 생산자와
# 소비자가 서로 다른 언어를 사용하는 크로스 팀 계약에 JSON Schema가 이상적이다.
ORDER_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "OrderRecord",
    "description": "Schema contract for the orders dataset (v2.0)",
    "type": "object",
    # "required"는 모든 레코드에 반드시 존재해야 하는 필드를 열거한다.
    # 여기에 없는 필드는 선택적 — 있을 수도 없을 수도 있다.
    # 하위 호환성을 최대화하려면 required 목록을 최소화하는 것이 좋다.
    "required": ["order_id", "customer_id", "amount", "status", "created_at"],
    "properties": {
        "order_id": {
            "type": "integer",
            "minimum": 1,
            "description": "Unique order identifier",
        },
        "customer_id": {
            "type": "integer",
            "minimum": 1,
            "description": "Customer foreign key",
        },
        "amount": {
            "type": "number",
            "minimum": 0,
            "maximum": 1000000,
            "description": "Order total in USD",
        },
        "status": {
            "type": "string",
            "enum": ["pending", "completed", "refunded", "cancelled"],
        },
        "created_at": {
            "type": "string",
            "format": "date-time",
        },
    },
    # additionalProperties: False는 엄격 모드다 — 예상치 못한 컬럼이 포함된
    # 레코드를 거부한다. 이는 새 컬럼이 조용히 나타나는 "스키마 드리프트
    # (Schema Drift)"를 방지한다: 다운스트림 소비자가 문서화되지 않은 필드에
    # 의도치 않게 의존하는 상황을 막는다. 생산자가 새 컬럼을 점진적으로
    # 추가하는 마이그레이션 기간에는 True로 설정한다.
    "additionalProperties": False,
}


def validate_record_json_schema(record: dict) -> bool:
    """JSON Schema 계약에 대해 단일 레코드를 검증."""
    try:
        validate(instance=record, schema=ORDER_SCHEMA)
        return True
    except ValidationError as e:
        print(f"Contract violation: {e.message}")
        print(f"  Path: {'.'.join(str(p) for p in e.absolute_path)}")
        return False
```

```python
# ── 방법 3: Avro 스키마 (스트리밍 / Kafka) ──────────────────
# Avro를 사용하는 이유? 네이티브 스키마 진화, 압축 바이너리 형식, Kafka 표준.
# 최적 용도: Kafka 토픽, 스키마 레지스트리 통합, 크로스 언어 스트리밍.

# Avro의 압축 바이너리 직렬화는 JSON 대비 Kafka 메시지 크기를 50-80% 줄인다.
# 스키마는 스키마 레지스트리에 한 번 등록하고, 각 메시지에는 4바이트 스키마 ID만 전송된다.
AVRO_SCHEMA = {
    "type": "record",
    "name": "OrderRecord",
    # namespace는 여러 팀이 같은 레지스트리에 스키마를 게시할 때 이름 충돌을 방지한다
    # — "com.company.analytics.OrderRecord"는 payments 팀에도 "OrderRecord"가
    # 있더라도 유일하다.
    "namespace": "com.company.analytics",
    "doc": "Schema contract for order events (v2.0)",
    "fields": [
        {"name": "order_id", "type": "long", "doc": "Unique order identifier"},
        {"name": "customer_id", "type": "long", "doc": "Customer FK"},
        {"name": "amount", "type": "double", "doc": "Order total in USD"},
        {
            "name": "status",
            # Avro enum은 string보다 제한적이다 — 리더가 "symbols"에 없는 값을
            # 거부한다. 이는 잘못된 status 값을 다운스트림이 아닌 직렬화 시점에
            # 잡아낸다.
            "type": {
                "type": "enum",
                "name": "OrderStatus",
                "symbols": ["PENDING", "COMPLETED", "REFUNDED", "CANCELLED"],
            },
        },
        {"name": "created_at", "type": "long", "logicalType": "timestamp-millis"},
        {
            "name": "updated_at",
            # Union 타입 ["null", "long"]은 이 필드를 nullable로 만든다.
            # default=None이 유효하려면 null이 Union의 첫 번째에 와야 한다.
            # 이는 Avro 고유의 순서 요구 사항이다.
            "type": ["null", "long"],
            "default": None,
            "logicalType": "timestamp-millis",
        },
    ],
}

"""
Avro 스키마 진화 규칙(Avro Schema Evolution Rules):
──────────────────────────────────────────────────
BACKWARD 호환 변경 (소비자가 구 스키마로 새 데이터를 읽을 수 있음):
  ✓ 기본값이 있는 필드 추가
  ✓ 기본값이 있는 필드 제거
  ✗ 필드 이름 변경 (파괴적!)
  ✗ 필드 타입 변경 (파괴적!)

FORWARD 호환 변경 (새 스키마를 가진 소비자가 구 데이터를 읽을 수 있음):
  ✓ 기본값이 있는 필드 추가
  ✓ 필드 제거
  ✗ 기본값 없이 필수 필드 추가 (파괴적!)

FULL 호환 = backward와 forward 모두 호환.

이것이 중요한 이유?
  - Kafka 소비자는 서로 다른 코드 버전을 실행할 수 있음
  - 생산자 업데이트가 소비자 동시 업데이트를 요구해서는 안 됨
  - 스키마 레지스트리가 쓰기 시점에 호환성을 강제
"""
```

---

## 5. 계약 테스팅과 적용

### 5.1 파이프라인에서의 계약 검증

```python
"""
파이프라인에서 계약을 적용하는 위치:

  생산자                    계약 게이트                 소비자
  ┌────────┐    ┌─────────────────────────────┐    ┌────────┐
  │ 소스    │───→│ 1. 스키마 검증              │───→│ 대상   │
  │ 시스템  │    │ 2. 의미론적 검사            │    │ 시스템 │
  │         │    │ 3. 최신성 검증              │    │        │
  └────────┘    │ 4. 완전성 검사              │    └────────┘
                │                             │
                │ PASS → 데이터 흐름 통과      │
                │ FAIL → 알림 + 격리          │
                └─────────────────────────────┘

경계에서 검증하는 이유 (소비자 내부가 아닌)?
  - 빠른 실패: 데이터가 퍼지기 전에 문제 발견
  - 단일 적용 지점: 하나의 게이트, 많은 소비자
  - 명확한 책임: 생산자가 위반을 수정
"""
```

### 5.2 Great Expectations 통합

```python
import great_expectations as gx

# Great Expectations를 사용하는 이유?
# - 선언적 데이터 품질 검사 (기대치, Expectations)
# - 풍부한 내장 기대치 (200개 이상)
# - 이해관계자 소통을 위한 HTML 데이터 문서
# - Airflow, Dagster, dbt와 통합

def create_order_contract_suite():
    """orders 데이터셋의 계약 기대치 정의.

    이 기대치들은 계약을 실행 가능한 테스트로 인코딩.
    각 기대치는 계약 조항에 매핑됨.
    """
    context = gx.get_context()

    # 데이터 소스와 배치 생성
    datasource = context.data_sources.add_pandas("orders_source")

    # 기대치 스위트는 하나의 데이터셋에 대한 모든 계약 검사를 그룹화한다.
    # 스위트 이름에 버전을 붙이면(v2) 마이그레이션 기간 동안
    # 이전 계약과 새 계약을 병렬로 실행할 수 있다.
    suite = context.suites.add_expectation_suite("orders_contract_v2")

    # 컬럼 순서 검사는 조용한 스키마 드리프트를 잡아낸다 — 생산자가 컬럼을
    # 재정렬하거나 이름을 바꾸면, 다운스트림 쿼리가 위치 기반으로 잘못된 컬럼을
    # 읽는 사태가 벌어지기 전에 이 기대치가 즉시 실패한다.
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchOrderedList(
            column_list=[
                "order_id", "customer_id", "amount",
                "status", "created_at", "updated_at",
            ]
        )
    )

    # 타입 기대치
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(
            column="amount", type_="float64"
        )
    )

    # mostly=0.99는 행의 최대 1%가 범위를 벗어나도 허용한다.
    # 이는 단 하나의 이상치 때문에 전체 검증이 실패하는 것을 막으면서도
    # 체계적인 데이터 품질 문제는 여전히 잡아낸다.
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="amount", min_value=0, max_value=1_000_000,
            mostly=0.99,
        )
    )

    # 열거형 기대치
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="status",
            value_set=["pending", "completed", "refunded", "cancelled"],
        )
    )

    # 유일성 기대치
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(column="order_id")
    )

    # 서로 다른 "mostly" 임계값은 비즈니스 중요도를 반영한다:
    # order_id는 기본 키이므로 1.0 (무관용),
    # amount는 0.99 — 소수의 null amount는 합법적일 수 있기 때문이다
    # (예: 무료 프로모션 주문).
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="order_id", mostly=1.0,
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="amount", mostly=0.99,
        )
    )

    return suite


def run_contract_validation(df, suite_name: str = "orders_contract_v2"):
    """DataFrame에 대해 계약 검증 실행.

    검증 결과는 다음과 같이 활용 가능:
    - 프로그래밍 방식으로 확인 (합격/불합격)
    - HTML 데이터 문서로 렌더링 (이해관계자용)
    - 알림 발송 (웹훅/이메일)
    """
    context = gx.get_context()
    # Checkpoint는 데이터 소스, 기대치 스위트, 검증 액션(결과 저장, 알림 발송)을
    # 하나의 재사용 가능한 단위로 결합한다.
    # 이는 "무엇을 검증할지"와 "언제/어떻게 검증할지"를 분리한다.
    result = context.run_checkpoint(
        checkpoint_name="orders_checkpoint",
        batch_request={
            "datasource_name": "orders_source",
            "data_asset_name": "orders",
            # batch_data는 인메모리 DataFrame을 받는다 — 디스크에 먼저 쓸 필요가 없다.
            # 이를 통해 스트리밍 foreachBatch 핸들러나 CI/CD 테스트 파이프라인 안에서도
            # 계약 검사를 수행할 수 있다.
            "batch_data": df,
        },
    )

    if not result.success:
        failed = [
            r for r in result.results
            if not r.success
        ]
        print(f"CONTRACT VIOLATION: {len(failed)} expectations failed")
        for f in failed:
            print(f"  - {f.expectation_config.expectation_type}: {f.result}")
        # 예외를 발생시키면 파이프라인이 중단되어 계약을 위반한 데이터가
        # 다운스트림으로 전파되지 않는다. 이것이 "빠른 실패(fail-fast)" 패턴이다:
        # 파이프라인 실패가 조용히 손상된 대시보드보다 낫다.
        raise ContractViolationError(f"{len(failed)} contract expectations failed")

    print("Contract validation PASSED")
    return result
```

### 5.3 Soda 통합

```python
# Soda는 YAML 기반 계약 정의 언어를 제공
# Soda를 사용하는 이유? 더 간단한 구문, 데이터 계약 특화

"""
# soda_contract.yaml — 데이터 계약 정의

dataset: orders
owner: data-platform-team
version: 2.0

schema:
  # 엄격한 스키마 검사 — 컬럼이 정확히 일치하지 않으면 실패
  fail:
    when mismatching columns:
      - order_id: integer
      - customer_id: integer
      - amount: float
      - status: string
      - created_at: timestamp
      - updated_at: timestamp

checks:
  # 유일성
  - duplicate_count(order_id) = 0

  # 완전성
  - missing_count(order_id) = 0
  - missing_percent(amount) < 1%

  # 값 범위
  - min(amount) >= 0
  - max(amount) < 1000000

  # 열거형 값
  - invalid_count(status) = 0:
      valid values: [pending, completed, refunded, cancelled]

  # 최신성 (데이터는 2시간 미만이어야 함)
  - freshness(created_at) < 2h

  # 행 수 (이상 감지)
  - row_count > 0
  - anomaly score for row_count < 3  # 정상의 3 표준편차 이상이면 알림
"""
```

---

## 6. 계약 우선 파이프라인 구축

### 6.1 계약 우선 접근법

파이프라인을 구축한 후 계약을 정의하는 대신, 계약 우선 접근법은 계약을 먼저 정의하고 이를 충족시키는 파이프라인을 구축한다.

```python
"""
계약 우선 개발 사이클(Contract-First Development Cycle):

1. 계약 정의 (생산자 + 소비자 합의)
   ├── 스키마: 컬럼, 타입, 제약 조건
   ├── SLA: 최신성, 완전성, 가용성
   └── 진화: 변경 제안 방법

2. 생산자 파이프라인 구현
   └── 계약에 맞는 데이터를 생성해야 함

3. 모든 파이프라인 실행 시 검증
   └── 계약 검사가 파이프라인 단계로 실행됨

4. 비즈니스 요구 변경 시 계약 진화
   ├── 변경 제안 (PR/RFC)
   ├── 소비자 영향 평가
   ├── 하위 호환성과 함께 구현
   └── 전환 기간 후 이전 버전 사용 중단

이는 API 우선 개발과 유사:
  - API 우선: OpenAPI 스펙 정의 → 엔드포인트 구현 → 스펙에 대해 테스트
  - 계약 우선: 데이터 계약 정의 → 파이프라인 구현 → 계약에 대해 검증
"""
```

### 6.2 계약 우선 파이프라인 구현

```python
import dagster as dg
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# 1단계: 계약을 먼저 정의
class OrderContract(BaseModel):
    """ORDER 계약 — 생산자와 소비자가 합의한 내용.

    이 모델은 다음 역할을 한다:
    - 문서화 (필드 설명)
    - 검증 로직 (타입 검사, 제약 조건)
    - 스키마 진화 추적기 (버전 필드)
    """
    class Config:
        json_schema_extra = {
            "version": "2.0",
            "owner": "data-platform-team",
            "consumers": ["analytics", "ml-team", "finance"],
            "sla": {
                "freshness": "1 hour",
                "completeness": "99%",
                "availability": "99.9%",
            },
        }

    order_id: int = Field(..., gt=0)
    customer_id: int = Field(..., gt=0)
    amount: float = Field(..., ge=0, le=1_000_000)
    status: str = Field(..., pattern=r"^(pending|completed|refunded|cancelled)$")
    created_at: datetime
    updated_at: Optional[datetime] = None


# 2단계: Dagster 자산 검사로서의 계약 검증
@dg.asset_check(asset=dg.AssetKey("cleaned_orders"))
def orders_contract_check(context: dg.AssetCheckExecutionContext, cleaned_orders: pd.DataFrame):
    """cleaned_orders를 OrderContract에 대해 검증.

    인라인 검증이 아닌 자산 검사를 사용하는 이유?
      - Dagster UI에서 가시적: 초록색 체크 = 계약 충족
      - 이력 추적: 계약이 언제 실패하기 시작했는지 확인
      - 차단: 실패 시 다운스트림 구체화 방지 가능
    """
    errors = []
    for idx, row in cleaned_orders.iterrows():
        try:
            OrderContract(**row.to_dict())
        except Exception as e:
            errors.append({"row": idx, "error": str(e)})

    error_rate = len(errors) / len(cleaned_orders) if len(cleaned_orders) > 0 else 0

    if error_rate > 0.01:  # >1% 오류율 = 계약 위반
        yield dg.AssetCheckResult(
            passed=False,
            metadata={
                "error_count": len(errors),
                "error_rate": f"{error_rate:.2%}",
                "sample_errors": str(errors[:5]),
            },
        )
    else:
        yield dg.AssetCheckResult(
            passed=True,
            metadata={
                "validated_rows": len(cleaned_orders),
                "error_count": len(errors),
                "contract_version": "2.0",
            },
        )
```

---

## 7. 스키마 진화(Schema Evolution)

### 7.1 파괴적 변경 vs 비파괴적 변경 관리

스키마 진화는 불가피하다. 핵심은 소비자를 깨뜨리는 변경과 그렇지 않은 변경을 구별하는 것이다.

```python
"""
스키마 진화 분류(Schema Evolution Classification):
═════════════════════════════════════════════════

비파괴적(NON-BREAKING) (소비자 조율 없이 안전하게 배포 가능):
  ✓ 기본값이 있는 선택적 컬럼 추가
  ✓ 타입 확장 (int32 → int64, float32 → float64)
  ✓ 새 열거형 값 추가 (소비자가 'default' 처리를 사용하는 경우)
  ✓ 최대 길이 제약 증가
  ✓ NOT NULL을 nullable로 완화

파괴적(BREAKING) (소비자 조율 필요):
  ✗ 컬럼 제거
  ✗ 컬럼 이름 변경
  ✗ 타입 축소 (int64 → int32)
  ✗ 기본값 없이 NOT NULL 컬럼 추가
  ✗ 컬럼의 의미론적 의미 변경
  ✗ 열거형 값 제거

파괴적 변경 마이그레이션 전략:
  1단계: 이전 컬럼 옆에 새 컬럼 추가 (둘 다 데이터 채움)
  2단계: 소비자가 새 컬럼으로 마이그레이션 (계약 레지스트리로 추적)
  3단계: 이전 컬럼 사용 중단 (경고 기간: 30-90일)
  4단계: 이전 컬럼 제거 (모든 소비자 마이그레이션 후)

타임라인: $T_{migration} \approx 30 + N_{consumers} \times 7$ 일
  여기서 $N_{consumers}$는 소비 팀의 수
"""
```

### 7.2 스키마 진화 구현

```python
from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class ChangeType(Enum):
    ADD_COLUMN = "add_column"
    REMOVE_COLUMN = "remove_column"
    RENAME_COLUMN = "rename_column"
    CHANGE_TYPE = "change_type"
    ADD_CONSTRAINT = "add_constraint"
    REMOVE_CONSTRAINT = "remove_constraint"


@dataclass
class SchemaChange:
    """데이터 계약에 대한 제안된 변경을 나타냄."""
    change_type: ChangeType
    column_name: str
    details: dict = field(default_factory=dict)
    is_breaking: bool = False
    migration_plan: str = ""


def assess_schema_changes(
    old_schema: dict,
    new_schema: dict,
) -> list[SchemaChange]:
    """두 스키마 버전을 비교하고 각 변경을 분류.

    자동화된 평가가 필요한 이유?
      - 사람은 대형 스키마에서 파괴적 변경을 놓칠 수 있음
      - 모든 계약 업데이트에 걸쳐 일관된 분류
      - CI/CD 게이트에 연동 (승인 없이 파괴적 변경 차단)
    """
    changes = []

    old_cols = set(old_schema.get("required", []))
    new_cols = set(new_schema.get("required", []))

    old_props = old_schema.get("properties", {})
    new_props = new_schema.get("properties", {})

    # 컬럼 제거는 항상 파괴적이다 — 현재 소비자가 해당 컬럼을 사용하지 않더라도,
    # 컬럼을 제거하면 Avro/Parquet 리더에서 스키마 불일치가 발생하여
    # 해당 컬럼이 포함된 이전 데이터를 소비자가 읽지 못하게 된다.
    for col in old_props:
        if col not in new_props:
            changes.append(SchemaChange(
                change_type=ChangeType.REMOVE_COLUMN,
                column_name=col,
                is_breaking=True,
                migration_plan=f"Phase out '{col}' over 30 days. "
                              f"Notify consumers: analytics, ML, finance.",
            ))

    # 선택적(OPTIONAL) 컬럼 추가는 기존 소비자가 이를 무시할 수 있으므로 비파괴적이다.
    # 하지만 기본값 없이 필수(REQUIRED) 컬럼을 추가하면, 모든 소비자가 새 스키마를
    # 읽기 전에 코드를 업데이트해야 한다.
    for col in new_props:
        if col not in old_props:
            is_required = col in new_cols
            changes.append(SchemaChange(
                change_type=ChangeType.ADD_COLUMN,
                column_name=col,
                details={"required": is_required, "schema": new_props[col]},
                is_breaking=is_required and "default" not in new_props[col],
                migration_plan="" if not is_required else
                    f"Add '{col}' with default value first, "
                    f"then make required after consumers adopt.",
            ))

    # 타입 변경은 확장(int32 → int64)이라도 파괴적이다 — 다운스트림 소비자가
    # 원래 타입을 Pydantic 모델, Spark 스키마, 데이터베이스 DDL에 하드코딩했을 수 있기
    # 때문이다. 안전한 패턴은 병렬 컬럼이다: 기존 컬럼을 유지하면서 새 타입 컬럼을
    # 추가하고, 모든 소비자가 마이그레이션한 후 기존 컬럼을 사용 중단한다.
    for col in old_props:
        if col in new_props:
            old_type = old_props[col].get("type")
            new_type = new_props[col].get("type")
            if old_type != new_type:
                changes.append(SchemaChange(
                    change_type=ChangeType.CHANGE_TYPE,
                    column_name=col,
                    details={"old_type": old_type, "new_type": new_type},
                    is_breaking=True,
                    migration_plan=f"Add '{col}_v2' ({new_type}) alongside "
                                  f"'{col}' ({old_type}). Deprecate after migration.",
                ))

    return changes


# 사용 예:
# changes = assess_schema_changes(ORDER_SCHEMA_V1, ORDER_SCHEMA_V2)
# breaking = [c for c in changes if c.is_breaking]
# if breaking:
#     print(f"BLOCKED: {len(breaking)} breaking changes require consumer approval")
```

---

## 8. 데이터 메시(Data Mesh)와 도메인 소유권

### 8.1 데이터 메시 원칙

데이터 계약은 데이터 소유권이 도메인 팀으로 분산되는 **데이터 메시(Data Mesh)** 아키텍처의 핵심이다.

```python
"""
데이터 메시 아키텍처(Data Mesh Architecture):
═══════════════════════════════════════════

전통적 방식 (중앙화):
  ┌───────────┐    ┌────────────────┐    ┌───────────┐
  │ Team A    │───→│ Central Data   │───→│ Team C    │
  │ (source)  │    │ Team           │    │ (consumer)│
  └───────────┘    │                │    └───────────┘
  ┌───────────┐    │ - Owns ALL     │    ┌───────────┐
  │ Team B    │───→│   pipelines    │───→│ Team D    │
  │ (source)  │    │ - Bottleneck   │    │ (consumer)│
  └───────────┘    └────────────────┘    └───────────┘

  문제: 중앙 팀 = 병목. 조직 성장에 따라 확장 불가.

데이터 메시 (분산화):
  ┌───────────────────┐        ┌───────────────────┐
  │ Orders Domain     │        │ Products Domain    │
  │ ┌──────────────┐  │        │ ┌──────────────┐  │
  │ │ Data Product: │  │        │ │ Data Product: │  │
  │ │ orders_clean  │──┼────────┼→│ product_catalog│ │
  │ │ CONTRACT: v2  │  │        │ │ CONTRACT: v3  │  │
  │ └──────────────┘  │        │ └──────────────┘  │
  └───────────────────┘        └───────────────────┘
          │                            │
          └──────────┬─────────────────┘
                     ▼
  ┌───────────────────────────┐
  │ Analytics Domain          │
  │ ┌──────────────────────┐  │
  │ │ Data Product:        │  │
  │ │ revenue_dashboard    │  │
  │ │ Consumes: orders v2  │  │
  │ │           products v3│  │
  │ └──────────────────────┘  │
  └───────────────────────────┘

데이터 메시의 4가지 원칙:
  1. 도메인 소유권(Domain Ownership): 각 도메인이 자체 데이터 파이프라인과 계약 소유
  2. 데이터를 제품으로(Data as a Product): SLA가 있는 제품으로 데이터 출력 처리
  3. 셀프 서비스 플랫폼(Self-Serve Platform): 스토리지, 컴퓨팅, 계약을 위한 공유 인프라
  4. 연합 거버넌스(Federated Governance): 공유 표준 (명명, 보안, 컴플라이언스)
"""
```

### 8.2 데이터 제품과 계약

```python
"""
데이터 제품(Data Product) = 데이터 + 계약 + 메타데이터 + SLA

예시: Orders 도메인이 "orders_clean" 데이터 제품 게시

  ┌────────────────────────────────────────────┐
  │ Data Product: orders_clean                 │
  ├────────────────────────────────────────────┤
  │ Owner: Orders Team (@orders-eng)           │
  │ Contract version: 2.0                      │
  │ Schema: OrderContract (Pydantic model)     │
  │ Location: s3://data-lake/gold/orders_clean │
  │ Format: Parquet (partitioned by date)      │
  │ Freshness SLA: < 1 hour from source       │
  │ Completeness SLA: > 99%                   │
  │ Availability: 99.9%                        │
  │ Access: self-service via data catalog      │
  │ Consumers: analytics, ML, finance          │
  │                                            │
  │ Change policy:                             │
  │   Non-breaking: deploy freely              │
  │   Breaking: 30-day notice + RFC            │
  └────────────────────────────────────────────┘

계약은 데이터 제품의 API이다.
계약이 없으면 데이터 메시는 데이터 혼돈으로 전락한다.
"""
```

---

## 9. 버전 관리된 ML 데이터셋

### 9.1 버전 관리된 데이터로 재현 가능한 훈련

데이터 버전 관리의 가장 영향력 있는 활용 중 하나는 ML 재현성이다.

```python
"""
ML 재현성 문제(The ML Reproducibility Problem):
═══════════════════════════════════════════════

지난달에 model_v3을 훈련했다. 성과가 좋았다.
오늘 동일한 데이터 + 새로운 피처로 재훈련해야 한다.
그러나 훈련 데이터가 변경되었다 (새 레코드, 수정, 스키마 업데이트).

버전 관리 없이:
  - "model_v3를 훈련한 정확한 데이터는?" → 알 수 없음
  - "model_v3 결과를 재현할 수 있나?" → 불가능
  - "model_v3와 model_v4 훈련 데이터 사이에 무엇이 변했나?" → 수동 비교

버전 관리 사용 시 (lakeFS + DVC):
  - model_v3는 lakeFS 커밋 abc123에서 훈련 (또는 DVC 태그 v3-data)
  - 재현: 정확한 버전 체크아웃, 재훈련 → 동일한 결과
  - 비교: 커밋 abc123 vs def456 비교 → 데이터 변경 확인
  - 감사: 원본 데이터 → 훈련 세트 → 모델 → 예측까지 전체 계보
"""

# ── lakeFS 방식: 훈련 데이터를 커밋에 고정 ────────────────

def train_model_with_versioned_data(
    lakefs_repo: str,
    commit_id: str,
    model_version: str,
):
    """고정된 데이터 버전을 사용하여 ML 모델 훈련.

    브랜치가 아닌 커밋에 고정하는 이유?
      - 브랜치는 이동함 (새 커밋이 추가됨)
      - 커밋은 불변 (동일한 데이터가 영원히)
      - 재현 가능: 동일 커밋 = 동일 훈련 데이터 = 동일 모델
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("ML-Training").getOrCreate()

    # 커밋 해시(브랜치가 아님)로 읽으면 불변성이 보장된다:
    # 내일 브랜치에 새 데이터가 추가되더라도, 이 훈련 실행은 항상 동일한 파일을 읽는다.
    # 이것이 ML 재현성의 토대다.
    training_data = spark.read.parquet(
        f"s3a://{lakefs_repo}/{commit_id}/ml/training/features/"
    )

    # MLflow에 데이터 버전을 기록하면 양방향 링크가 생성된다:
    # 모델 → 데이터 ("이 모델은 어떤 데이터로 훈련됐나?") 와
    # 데이터 → 모델 ("이 데이터 버전으로 훈련된 모델은?").
    # 이 링크 없이는 모델 품질 저하의 원인 조사가 추측에 불과하다.
    import mlflow
    with mlflow.start_run():
        mlflow.log_param("data_lakefs_repo", lakefs_repo)
        mlflow.log_param("data_lakefs_commit", commit_id)
        mlflow.log_param("model_version", model_version)
        # 행 수는 빠른 정상성 검사 역할을 한다 — 기대 수치와 크게 다르면
        # 데이터 버전이 잘못되었거나 피처 파이프라인에 버그가 있을 수 있다.
        mlflow.log_metric("training_rows", training_data.count())

        # 모델 훈련...
        # model = train(training_data)
        # mlflow.sklearn.log_model(model, "model")


# ── DVC 방식: Git에서 데이터셋 버전 태그 지정 ─────────────

"""
# ML 데이터 버전 관리를 위한 DVC 워크플로우:

# 1. DVC로 데이터셋 추적
$ dvc add data/training_features.parquet
# 생성: data/training_features.parquet.dvc (Git의 포인터 파일)

# 2. Git에 포인터 커밋
$ git add data/training_features.parquet.dvc
$ git commit -m "Training data v3: added click features"
$ git tag data-v3

# 3. 데이터 버전을 참조하여 모델 훈련
$ python train.py --data-version data-v3

# 4. 나중에 재현:
$ git checkout data-v3
$ dvc checkout          # 정확한 데이터 버전 다운로드
$ python train.py       # 동일한 결과

# 5. 데이터 버전 비교:
$ dvc diff data-v2 data-v3
#   Modified: data/training_features.parquet
#   +5000 rows, +2 columns (click_count, session_duration)
"""
```

### 9.2 데이터셋 계보(Lineage)

```python
"""
버전 관리를 통한 완전한 ML 계보:

  원본 데이터             피처 스토어            모델 레지스트리
  (lakeFS)             (lakeFS/DVC)           (MLflow)
  ┌─────────┐         ┌─────────────┐        ┌─────────────┐
  │commit a1 │────────→│commit f1    │───────→│ model_v1    │
  │(raw data)│ ETL +   │(features v1)│ Train  │ metric: 0.85│
  └─────────┘ Feature  └─────────────┘        └─────────────┘
              Eng.

  ┌─────────┐         ┌─────────────┐        ┌─────────────┐
  │commit a2 │────────→│commit f2    │───────→│ model_v2    │
  │(+new data)│ ETL + │(features v2)│ Train  │ metric: 0.87│
  └─────────┘ Feature  └─────────────┘        └─────────────┘
              Eng.     (+click feats)

완전한 추적 가능성:
  model_v2 → features f2에서 훈련 → 원본 데이터 a2에서 파생
  f1→f2 비교: +2 컬럼 (click_count, session_duration)
  a1→a2 비교: 1월분 +50K 새 행
"""
```

---

## 요약

```
데이터 버전 관리 핵심 개념:
──────────────────────────
lakeFS          = 데이터 레이크를 위한 Git 방식 버전 관리 (브랜치, 커밋, 머지)
DVC             = 데이터셋과 ML 모델 버전 관리를 위한 Git 확장
카피-온-라이트  = 전체 복사가 아닌 변경분만 저장 (효율적인 브랜칭)
타임 트래블     = 과거 데이터 버전 쿼리 (Delta/Iceberg에서도 사용 가능)
원자적 커밋     = 여러 파일에 대한 전부 또는 전무 방식 변경

데이터 계약 핵심 개념:
─────────────────────
스키마 계약     = 컬럼, 타입, 제약 조건의 공식 정의
의미론 계약     = 비즈니스 의미와 계산 로직
SLA             = 최신성, 완전성, 가용성 보장
진화 정책       = 파괴적 vs 비파괴적 변경 관리 방법
계약 테스팅     = 파이프라인 경계에서의 자동화된 검증

계약 도구:
  Pydantic          → Python 네이티브, Python 파이프라인에 적합
  JSON Schema       → 언어 독립적, API 경계
  Avro              → 스트리밍/Kafka, 내장 진화 지원
  Great Expectations → 풍부한 기대치 라이브러리, 데이터 문서
  Soda              → YAML 기반, 계약 특화

데이터 메시 연결:
  데이터 제품 = 데이터 + 계약 + SLA + 소유권
  도메인 팀이 자체 데이터 제품과 계약을 소유
  계약이 분산화된 셀프 서비스 데이터 소비를 가능하게 함
```

---

## 연습 문제

### 연습 1: lakeFS 브랜치-머지 워크플로우

lakeFS를 사용하여 데이터 파이프라인 변경을 시뮬레이션한다:

1. lakeFS 저장소 생성 (Docker 퀵스타트 사용 또는 API 모킹)
2. `main` 브랜치에 초기 주문 데이터를 쓰고 커밋
3. `feature/add-discount-column` 브랜치 생성
4. 피처 브랜치의 주문 데이터에 `discount` 컬럼 추가
5. diff API를 사용하여 브랜치 비교
6. 피처 브랜치를 `main`에 머지
7. 머지된 데이터에 새 컬럼이 있는지 확인

### 연습 2: Pydantic 계약 구축

`user_events` 데이터셋에 대한 완전한 데이터 계약을 정의한다:

1. 필드가 있는 Pydantic 모델 생성: `event_id`, `user_id`, `event_type` (열거형: click, view, purchase), `timestamp`, `page_url`, `metadata` (선택적 dict)
2. 비즈니스 규칙에 대한 필드 검증기 추가 (예: `page_url`은 `/`로 시작해야 함)
3. 유효한 레코드와 오류 세부 정보를 반환하는 `validate_dataframe` 함수 작성
4. 유효하고 유효하지 않은 행이 모두 포함된 DataFrame으로 테스트
5. 설정 가능한 오류 임계값 구현 (>X% 실패 시 전체 배치 거부)

### 연습 3: 스키마 진화 평가

스키마 진화 분석기를 구현한다:

1. `ORDER_SCHEMA_V1`과 `ORDER_SCHEMA_V2` 정의 (V2는 컬럼 이름을 변경하고 새 선택적 컬럼 추가)
2. 버전 간 모든 변경 사항을 감지하는 함수 작성
3. 각 변경을 파괴적 또는 비파괴적으로 분류
4. 파괴적 변경에 대한 마이그레이션 계획 생성
5. 분류기가 파괴적 변경을 올바르게 식별하는지 검증하는 테스트 작성

### 연습 4: Dagster를 사용한 계약 우선 파이프라인

계약 우선 Dagster 파이프라인 구축:

1. `transactions` 데이터셋에 대한 계약 (Pydantic 모델) 정의
2. 모의 거래 데이터를 생성하는 Dagster 자산 생성
3. 계약에 대해 자산을 검증하는 `@dg.asset_check` 추가
4. `transactions`를 소비하는 다운스트림 자산 생성 (검사 통과 시에만 실행)
5. 자산과 계약 검사 모두에 대한 테스트 작성

### 연습 5: DVC + lakeFS 통합 설계

ML 플랫폼을 위한 하이브리드 버전 관리 전략 설계:

1. 아키텍처를 그림으로 표현: 원본 데이터 (lakeFS), 피처 (lakeFS), 모델 (DVC + MLflow)
2. 버전 관리 워크플로우 정의: 데이터 과학자가 새 피처 세트를 생성하고, 모델을 훈련하고, 프로덕션으로 승격하는 방법
3. 데이터 변경을 머지하기 전에 데이터 계약을 검증하는 CI/CD 파이프라인의 의사 코드 작성
4. 어떤 아티팩트가 어디에 버전 관리되는지와 그 이유 식별

---

## 참고 자료

- [lakeFS 공식 문서](https://docs.lakefs.io/)
- [lakeFS GitHub 저장소](https://github.com/treeverse/lakeFS)
- [DVC 공식 문서](https://dvc.org/doc)
- [Data Contracts — PayPal Engineering Blog](https://medium.com/paypal-tech/the-next-big-thing-in-data-engineering-data-contracts-17a55e7a0b89)
- [Great Expectations 공식 문서](https://docs.greatexpectations.io/)
- [Soda 데이터 계약](https://docs.soda.io/soda/data-contracts.html)
- [Avro 스키마 진화](https://avro.apache.org/docs/current/specification/)
- [Zhamak Dehghani의 Data Mesh](https://www.datamesh-architecture.com/)
- [JSON Schema 명세](https://json-schema.org/)
- [Confluent 스키마 레지스트리](https://docs.confluent.io/platform/current/schema-registry/)

---

[← 이전: 20. Dagster 자산 기반 오케스트레이션](20_Dagster_Asset_Orchestration.md) | [다음: 개요 →](00_Overview.md)
