# Feature Store

## 1. Feature Store 개념

Feature Store는 ML 피처를 중앙에서 관리, 저장, 서빙하는 플랫폼입니다.

### 1.1 Feature Store의 역할

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Feature Store 아키텍처                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │                Feature Engineering                       │      │
│   │                                                          │      │
│   │   Raw Data → Transform → Features → Feature Store        │      │
│   │                                                          │      │
│   └─────────────────────────────────────────────────────────┘      │
│                          │                                          │
│                          ▼                                          │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │                  Feature Store                           │      │
│   │  ┌──────────────────┐  ┌──────────────────┐             │      │
│   │  │  Offline Store   │  │   Online Store   │             │      │
│   │  │                  │  │                  │             │      │
│   │  │ - 학습용         │  │ - 추론용         │             │      │
│   │  │ - 배치 처리      │  │ - 저지연         │             │      │
│   │  │ - 대용량        │  │ - 키-값 조회     │             │      │
│   │  │                  │  │                  │             │      │
│   │  │ (S3, BigQuery,  │  │ (Redis, DynamoDB)│             │      │
│   │  │  Parquet)       │  │                  │             │      │
│   │  └──────────────────┘  └──────────────────┘             │      │
│   │                                                          │      │
│   │  ┌──────────────────────────────────────────┐           │      │
│   │  │           Feature Registry               │           │      │
│   │  │  - 피처 메타데이터                       │           │      │
│   │  │  - 버전 관리                             │           │      │
│   │  │  - 스키마 정의                           │           │      │
│   │  └──────────────────────────────────────────┘           │      │
│   └─────────────────────────────────────────────────────────┘      │
│                          │                                          │
│            ┌─────────────┴─────────────┐                           │
│            ▼                           ▼                           │
│   ┌─────────────────┐        ┌─────────────────┐                   │
│   │    Training     │        │    Inference    │                   │
│   │  (Offline)      │        │   (Online)      │                   │
│   └─────────────────┘        └─────────────────┘                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Feature Store의 장점

```python
"""
Feature Store 도입 이점
"""

benefits = {
    "재사용성": {
        "description": "한 번 정의한 피처를 여러 모델에서 재사용",
        "example": "user_total_purchases를 이탈 예측, 추천, 사기 탐지 모델에서 모두 사용"
    },
    "일관성": {
        "description": "학습과 추론에서 동일한 피처 계산 보장",
        "example": "학습/서빙 스큐(skew) 방지"
    },
    "시점 정확성": {
        "description": "특정 시점의 피처 값을 정확히 조회",
        "example": "예측 시점에 알 수 있었던 정보만 사용 (데이터 누수 방지)"
    },
    "피처 검색": {
        "description": "중앙 레지스트리에서 기존 피처 검색",
        "example": "팀 간 피처 공유 및 발견"
    },
    "거버넌스": {
        "description": "피처 버전 관리, 접근 제어, 리니지 추적",
        "example": "규제 준수, 감사 대응"
    }
}
```

---

## 2. Feast 개요

Feast(Feature Store)는 가장 널리 사용되는 오픈소스 Feature Store입니다.

### 2.1 설치

```bash
# Feast 설치
pip install feast

# 특정 저장소 지원 추가
pip install feast[redis]    # Redis 온라인 스토어
pip install feast[gcp]      # GCP 지원
pip install feast[aws]      # AWS 지원
```

### 2.2 프로젝트 구조

```
feature_repo/
├── feature_store.yaml      # 프로젝트 설정
├── features/
│   ├── user_features.py    # 사용자 피처 정의
│   └── product_features.py # 상품 피처 정의
├── data/
│   └── user_data.parquet   # 오프라인 데이터
└── tests/
    └── test_features.py
```

### 2.3 기본 설정

```yaml
# feature_store.yaml
project: churn_prediction
registry: data/registry.db
provider: local

online_store:
  type: redis
  connection_string: "localhost:6379"

offline_store:
  type: file

entity_key_serialization_version: 2
```

---

## 3. 피처 정의

### 3.1 Entity와 Feature View

```python
"""
features/user_features.py - 사용자 피처 정의
"""

from datetime import timedelta
from feast import Entity, Feature, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64, String

# Entity 정의 (피처의 키)
user = Entity(
    name="user_id",
    description="Customer ID",
    value_type=ValueType.INT64
)

# 데이터 소스 정의
user_source = FileSource(
    path="data/user_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp"
)

# Feature View 정의
user_features = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=1),  # Time to Live
    schema=[
        Field(name="total_purchases", dtype=Int64),
        Field(name="avg_purchase_amount", dtype=Float32),
        Field(name="days_since_last_purchase", dtype=Int64),
        Field(name="customer_segment", dtype=String),
        Field(name="tenure_months", dtype=Int64),
    ],
    online=True,   # 온라인 스토어 활성화
    source=user_source,
    tags={
        "team": "ml-platform",
        "owner": "data-science"
    }
)

# 집계 피처 (On-Demand Feature)
from feast import on_demand_feature_view
import pandas as pd

@on_demand_feature_view(
    sources=[user_features],
    schema=[
        Field(name="purchase_frequency", dtype=Float32),
        Field(name="is_high_value", dtype=Int64),
    ]
)
def user_derived_features(inputs: pd.DataFrame) -> pd.DataFrame:
    """실시간 계산 피처"""
    df = pd.DataFrame()
    df["purchase_frequency"] = inputs["total_purchases"] / inputs["tenure_months"]
    df["is_high_value"] = (inputs["avg_purchase_amount"] > 100).astype(int)
    return df
```

### 3.2 복잡한 피처 정의

```python
"""
features/transaction_features.py - 트랜잭션 피처
"""

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from feast.aggregation import Aggregation
from datetime import timedelta

# 트랜잭션 소스
transaction_source = FileSource(
    path="data/transactions.parquet",
    timestamp_field="transaction_timestamp"
)

# 트랜잭션 피처 뷰
transaction_features = FeatureView(
    name="transaction_features",
    entities=[user],
    ttl=timedelta(days=7),
    schema=[
        Field(name="transaction_amount", dtype=Float32),
        Field(name="transaction_count_7d", dtype=Int64),
        Field(name="avg_transaction_7d", dtype=Float32),
    ],
    source=transaction_source,
    online=True
)

# 시간 윈도우 집계 (StreamFeatureView - Feast 0.26+)
from feast import StreamFeatureView, PushSource

push_source = PushSource(
    name="transaction_push_source",
    batch_source=transaction_source
)

streaming_features = StreamFeatureView(
    name="transaction_streaming_features",
    entities=[user],
    ttl=timedelta(hours=1),
    schema=[
        Field(name="transaction_count_1h", dtype=Int64),
        Field(name="total_amount_1h", dtype=Float32),
    ],
    source=push_source,
    aggregations=[
        Aggregation(
            column="transaction_amount",
            function="count",
            time_window=timedelta(hours=1)
        ),
        Aggregation(
            column="transaction_amount",
            function="sum",
            time_window=timedelta(hours=1)
        )
    ]
)
```

---

## 4. 온라인/오프라인 스토어

### 4.1 Feast 레지스트리 적용

```bash
# 피처 정의를 레지스트리에 적용
feast apply

# 현재 등록된 피처 확인
feast feature-views list
feast entities list
```

### 4.2 오프라인 스토어 (학습용)

```python
"""
오프라인 피처 조회 - 학습 데이터 생성
"""

from feast import FeatureStore
import pandas as pd

# Feature Store 초기화
store = FeatureStore(repo_path="./feature_repo")

# 엔티티 데이터 (학습할 샘플)
entity_df = pd.DataFrame({
    "user_id": [1001, 1002, 1003, 1004, 1005],
    "event_timestamp": pd.to_datetime([
        "2024-01-15 10:00:00",
        "2024-01-15 11:00:00",
        "2024-01-15 12:00:00",
        "2024-01-15 13:00:00",
        "2024-01-15 14:00:00"
    ])
})

# 히스토리컬 피처 조회 (Point-in-time 정확)
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_features:total_purchases",
        "user_features:avg_purchase_amount",
        "user_features:days_since_last_purchase",
        "user_features:tenure_months",
        "user_derived_features:purchase_frequency",
        "transaction_features:transaction_count_7d"
    ]
).to_df()

print(training_df.head())
print(f"Training data shape: {training_df.shape}")

# 결과를 학습에 사용
# X = training_df.drop(["user_id", "event_timestamp"], axis=1)
# y = training_df["label"]  # 레이블은 별도로 조인
```

### 4.3 온라인 스토어 (추론용)

```python
"""
온라인 피처 조회 - 실시간 추론
"""

from feast import FeatureStore

store = FeatureStore(repo_path="./feature_repo")

# 온라인 스토어에 피처 로드 (materialization)
# 오프라인 → 온라인 동기화
store.materialize_incremental(end_date=datetime.now())

# 또는 전체 기간 재동기화
# store.materialize(
#     start_date=datetime(2024, 1, 1),
#     end_date=datetime.now()
# )

# 온라인 피처 조회 (저지연)
feature_vector = store.get_online_features(
    features=[
        "user_features:total_purchases",
        "user_features:avg_purchase_amount",
        "user_features:days_since_last_purchase",
        "user_derived_features:purchase_frequency"
    ],
    entity_rows=[
        {"user_id": 1001},
        {"user_id": 1002}
    ]
).to_dict()

print(feature_vector)
# {
#     "user_id": [1001, 1002],
#     "total_purchases": [45, 23],
#     "avg_purchase_amount": [67.5, 89.2],
#     ...
# }
```

### 4.4 Feature Server

```bash
# Feature Server 시작 (REST API)
feast serve -p 6566

# API 호출
curl -X POST "http://localhost:6566/get-online-features" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      "user_features:total_purchases",
      "user_features:avg_purchase_amount"
    ],
    "entities": {
      "user_id": [1001, 1002]
    }
  }'
```

---

## 5. 피처 서빙 통합

### 5.1 추론 파이프라인 통합

```python
"""
모델 추론에 Feature Store 통합
"""

from feast import FeatureStore
import joblib
import numpy as np

class ModelWithFeatureStore:
    """Feature Store 통합 모델 서버"""

    def __init__(self, model_path: str, feature_repo_path: str):
        self.model = joblib.load(model_path)
        self.store = FeatureStore(repo_path=feature_repo_path)
        self.feature_list = [
            "user_features:total_purchases",
            "user_features:avg_purchase_amount",
            "user_features:days_since_last_purchase",
            "user_features:tenure_months",
            "user_derived_features:purchase_frequency"
        ]

    def predict(self, user_id: int) -> dict:
        """피처 조회 후 예측"""
        # 1. Feature Store에서 피처 조회
        features = self.store.get_online_features(
            features=self.feature_list,
            entity_rows=[{"user_id": user_id}]
        ).to_dict()

        # 2. 피처 벡터 생성
        feature_names = [f.split(":")[1] for f in self.feature_list]
        feature_vector = np.array([
            [features[name][0] for name in feature_names]
        ])

        # 3. 예측
        prediction = self.model.predict(feature_vector)[0]
        probability = self.model.predict_proba(feature_vector)[0]

        return {
            "user_id": user_id,
            "prediction": int(prediction),
            "probability": probability.tolist(),
            "features_used": dict(zip(feature_names, feature_vector[0].tolist()))
        }

# 사용
model_server = ModelWithFeatureStore(
    model_path="models/churn_model.pkl",
    feature_repo_path="./feature_repo"
)

result = model_server.predict(user_id=1001)
print(result)
```

### 5.2 FastAPI 통합

```python
"""
FastAPI + Feature Store
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from feast import FeatureStore
import joblib
import numpy as np

app = FastAPI()
store = FeatureStore(repo_path="./feature_repo")
model = joblib.load("models/churn_model.pkl")

FEATURES = [
    "user_features:total_purchases",
    "user_features:avg_purchase_amount",
    "user_features:tenure_months"
]

class PredictionRequest(BaseModel):
    user_id: int

class PredictionResponse(BaseModel):
    user_id: int
    prediction: int
    probability: list
    features: dict

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """피처 조회 및 예측"""
    try:
        # Feature Store에서 조회
        features = store.get_online_features(
            features=FEATURES,
            entity_rows=[{"user_id": request.user_id}]
        ).to_dict()

        # 피처 벡터 생성
        feature_names = [f.split(":")[1] for f in FEATURES]
        feature_vector = np.array([
            [features[name][0] for name in feature_names]
        ])

        # 예측
        pred = model.predict(feature_vector)[0]
        prob = model.predict_proba(feature_vector)[0]

        return PredictionResponse(
            user_id=request.user_id,
            prediction=int(pred),
            probability=prob.tolist(),
            features={name: features[name][0] for name in feature_names}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

---

## 6. 피처 엔지니어링 파이프라인

### 6.1 배치 피처 파이프라인

```python
"""
배치 피처 계산 파이프라인
"""

import pandas as pd
from datetime import datetime, timedelta

class FeatureEngineeringPipeline:
    """피처 엔지니어링 파이프라인"""

    def __init__(self, raw_data_path: str, output_path: str):
        self.raw_data_path = raw_data_path
        self.output_path = output_path

    def load_raw_data(self) -> pd.DataFrame:
        """원시 데이터 로드"""
        return pd.read_parquet(self.raw_data_path)

    def compute_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """사용자 피처 계산"""
        user_features = df.groupby("user_id").agg({
            "transaction_amount": ["count", "sum", "mean"],
            "transaction_date": ["min", "max"]
        }).reset_index()

        user_features.columns = [
            "user_id", "total_purchases", "total_amount",
            "avg_purchase_amount", "first_purchase_date", "last_purchase_date"
        ]

        # 추가 피처 계산
        today = datetime.now()
        user_features["days_since_last_purchase"] = (
            today - pd.to_datetime(user_features["last_purchase_date"])
        ).dt.days

        user_features["tenure_months"] = (
            pd.to_datetime(user_features["last_purchase_date"]) -
            pd.to_datetime(user_features["first_purchase_date"])
        ).dt.days // 30

        # 타임스탬프 추가
        user_features["event_timestamp"] = today
        user_features["created_timestamp"] = today

        return user_features

    def validate_features(self, df: pd.DataFrame) -> bool:
        """피처 검증"""
        # Null 체크
        if df.isnull().sum().sum() > 0:
            print("Warning: Null values detected")
            return False

        # 범위 체크
        if (df["avg_purchase_amount"] < 0).any():
            print("Warning: Negative values in avg_purchase_amount")
            return False

        return True

    def save_features(self, df: pd.DataFrame):
        """피처 저장"""
        df.to_parquet(
            self.output_path,
            index=False,
            engine="pyarrow"
        )
        print(f"Features saved to {self.output_path}")

    def run(self):
        """파이프라인 실행"""
        print("Loading raw data...")
        raw_df = self.load_raw_data()

        print("Computing features...")
        features_df = self.compute_user_features(raw_df)

        print("Validating features...")
        if not self.validate_features(features_df):
            raise ValueError("Feature validation failed")

        print("Saving features...")
        self.save_features(features_df)

        return features_df

# 실행
pipeline = FeatureEngineeringPipeline(
    raw_data_path="data/raw_transactions.parquet",
    output_path="feature_repo/data/user_features.parquet"
)
features = pipeline.run()

# Feast 피처 동기화
store = FeatureStore(repo_path="./feature_repo")
store.materialize_incremental(end_date=datetime.now())
```

### 6.2 스트리밍 피처 업데이트

```python
"""
스트리밍 피처 업데이트 (Feast Push)
"""

from feast import FeatureStore
import pandas as pd
from datetime import datetime

store = FeatureStore(repo_path="./feature_repo")

def process_streaming_event(event: dict):
    """실시간 이벤트 처리 및 피처 업데이트"""
    # 이벤트에서 피처 추출
    feature_df = pd.DataFrame([{
        "user_id": event["user_id"],
        "transaction_amount": event["amount"],
        "transaction_timestamp": datetime.now()
    }])

    # Feature Store에 푸시
    store.push(
        push_source_name="transaction_push_source",
        df=feature_df
    )

# Kafka Consumer 예시
# for message in kafka_consumer:
#     event = json.loads(message.value)
#     process_streaming_event(event)
```

---

## 연습 문제

### 문제 1: Feature Store 설정
Feast Feature Store를 설정하고 기본 피처를 정의하세요.

### 문제 2: 학습 데이터 생성
get_historical_features를 사용하여 Point-in-time 정확한 학습 데이터를 생성하세요.

### 문제 3: 온라인 서빙
온라인 스토어를 설정하고 실시간 추론에 통합하세요.

---

## 요약

| 구성 요소 | 설명 | 사용 사례 |
|----------|------|----------|
| Offline Store | 대용량 히스토리컬 데이터 | 학습 데이터 생성 |
| Online Store | 저지연 키-값 조회 | 실시간 추론 |
| Feature Registry | 피처 메타데이터 관리 | 피처 발견, 거버넌스 |
| Materialization | 오프라인 → 온라인 동기화 | 피처 서빙 준비 |

---

## 참고 자료

- [Feast Documentation](https://docs.feast.dev/)
- [Feature Store for ML](https://www.featurestore.org/)
- [Tecton Feature Platform](https://www.tecton.ai/)

---

## 예제 코드

구체화(Materialization)와 서빙을 포함한 독립 실행 가능한 Feast 피처 스토어(Feature Store) 예제:

**[`examples/MLOps/feature_store/`](../../../examples/MLOps/feature_store/)**

L11 이론과 L12 실습 프로젝트를 연결 — 동일한 엔티티/피처 스키마, 실행 가능한 오프라인/온라인 조회 데모.
