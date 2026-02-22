# 11. Feature Stores

## 1. Feature Store Concept

A Feature Store is a platform that centrally manages, stores, and serves ML features.

### 1.1 Feature Store Role

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Feature Store Architecture                       │
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
│   │  │ - Training       │  │ - Inference      │             │      │
│   │  │ - Batch          │  │ - Low latency    │             │      │
│   │  │ - Large scale    │  │ - Key-value      │             │      │
│   │  │                  │  │                  │             │      │
│   │  │ (S3, BigQuery,   │  │ (Redis, DynamoDB)│             │      │
│   │  │  Parquet)        │  │                  │             │      │
│   │  └──────────────────┘  └──────────────────┘             │      │
│   │                                                          │      │
│   │  ┌──────────────────────────────────────────┐           │      │
│   │  │           Feature Registry               │           │      │
│   │  │  - Feature metadata                      │           │      │
│   │  │  - Version management                    │           │      │
│   │  │  - Schema definition                     │           │      │
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

### 1.2 Feature Store Benefits

```python
"""
Feature Store adoption benefits
"""

benefits = {
    "reusability": {
        "description": "Reuse features defined once across multiple models",
        "example": "Use user_total_purchases in churn, recommendation, and fraud models"
    },
    "consistency": {
        "description": "Ensure identical feature computation in training and inference",
        "example": "Prevent training/serving skew"
    },
    "point_in_time_accuracy": {
        "description": "Retrieve exact feature values at specific points in time",
        "example": "Use only information available at prediction time (prevent data leakage)"
    },
    "feature_discovery": {
        "description": "Search existing features in central registry",
        "example": "Share and discover features across teams"
    },
    "governance": {
        "description": "Feature versioning, access control, lineage tracking",
        "example": "Regulatory compliance, audit response"
    }
}
```

---

## 2. Feast Overview

Feast (Feature Store) is the most widely used open-source Feature Store.

### 2.1 Installation

```bash
# Install Feast
pip install feast

# Add support for specific stores
pip install feast[redis]    # Redis online store
pip install feast[gcp]      # GCP support
pip install feast[aws]      # AWS support
```

### 2.2 Project Structure

```
feature_repo/
├── feature_store.yaml      # Project configuration
├── features/
│   ├── user_features.py    # User feature definitions
│   └── product_features.py # Product feature definitions
├── data/
│   └── user_data.parquet   # Offline data
└── tests/
    └── test_features.py
```

### 2.3 Basic Configuration

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

## 3. Feature Definition

### 3.1 Entity and Feature View

```python
"""
features/user_features.py - User feature definitions
"""

from datetime import timedelta
from feast import Entity, Feature, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64, String

# Define Entity (feature key)
user = Entity(
    name="user_id",
    description="Customer ID",
    value_type=ValueType.INT64
)

# Define data source
user_source = FileSource(
    path="data/user_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp"
)

# Define Feature View
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
    online=True,   # Enable online store
    source=user_source,
    tags={
        "team": "ml-platform",
        "owner": "data-science"
    }
)

# Derived features (On-Demand Feature)
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
    """Real-time computed features"""
    df = pd.DataFrame()
    df["purchase_frequency"] = inputs["total_purchases"] / inputs["tenure_months"]
    df["is_high_value"] = (inputs["avg_purchase_amount"] > 100).astype(int)
    return df
```

### 3.2 Complex Feature Definitions

```python
"""
features/transaction_features.py - Transaction features
"""

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from feast.aggregation import Aggregation
from datetime import timedelta

# Transaction source
transaction_source = FileSource(
    path="data/transactions.parquet",
    timestamp_field="transaction_timestamp"
)

# Transaction feature view
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

# Time window aggregation (StreamFeatureView - Feast 0.26+)
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

## 4. Online/Offline Stores

### 4.1 Apply Feast Registry

```bash
# Apply feature definitions to registry
feast apply

# List registered features
feast feature-views list
feast entities list
```

### 4.2 Offline Store (Training)

```python
"""
Offline feature retrieval - Generate training data
"""

from feast import FeatureStore
import pandas as pd

# Initialize Feature Store
store = FeatureStore(repo_path="./feature_repo")

# Entity data (samples to train on)
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

# Retrieve historical features (Point-in-time accurate)
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

# Use result for training
# X = training_df.drop(["user_id", "event_timestamp"], axis=1)
# y = training_df["label"]  # Join label separately
```

### 4.3 Online Store (Inference)

```python
"""
Online feature retrieval - Real-time inference
"""

from feast import FeatureStore

store = FeatureStore(repo_path="./feature_repo")

# Load features to online store (materialization)
# Sync offline → online
store.materialize_incremental(end_date=datetime.now())

# Or full re-sync for entire period
# store.materialize(
#     start_date=datetime(2024, 1, 1),
#     end_date=datetime.now()
# )

# Online feature retrieval (low latency)
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
# Start Feature Server (REST API)
feast serve -p 6566

# API call
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

## 5. Feature Serving Integration

### 5.1 Inference Pipeline Integration

```python
"""
Integrate Feature Store with model inference
"""

from feast import FeatureStore
import joblib
import numpy as np

class ModelWithFeatureStore:
    """Model server with Feature Store integration"""

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
        """Retrieve features and predict"""
        # 1. Get features from Feature Store
        features = self.store.get_online_features(
            features=self.feature_list,
            entity_rows=[{"user_id": user_id}]
        ).to_dict()

        # 2. Create feature vector
        feature_names = [f.split(":")[1] for f in self.feature_list]
        feature_vector = np.array([
            [features[name][0] for name in feature_names]
        ])

        # 3. Predict
        prediction = self.model.predict(feature_vector)[0]
        probability = self.model.predict_proba(feature_vector)[0]

        return {
            "user_id": user_id,
            "prediction": int(prediction),
            "probability": probability.tolist(),
            "features_used": dict(zip(feature_names, feature_vector[0].tolist()))
        }

# Usage
model_server = ModelWithFeatureStore(
    model_path="models/churn_model.pkl",
    feature_repo_path="./feature_repo"
)

result = model_server.predict(user_id=1001)
print(result)
```

### 5.2 FastAPI Integration

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
    """Retrieve features and predict"""
    try:
        # Retrieve from Feature Store
        features = store.get_online_features(
            features=FEATURES,
            entity_rows=[{"user_id": request.user_id}]
        ).to_dict()

        # Create feature vector
        feature_names = [f.split(":")[1] for f in FEATURES]
        feature_vector = np.array([
            [features[name][0] for name in feature_names]
        ])

        # Predict
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

## 6. Feature Engineering Pipeline

### 6.1 Batch Feature Pipeline

```python
"""
Batch feature computation pipeline
"""

import pandas as pd
from datetime import datetime, timedelta

class FeatureEngineeringPipeline:
    """Feature engineering pipeline"""

    def __init__(self, raw_data_path: str, output_path: str):
        self.raw_data_path = raw_data_path
        self.output_path = output_path

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data"""
        return pd.read_parquet(self.raw_data_path)

    def compute_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute user features"""
        user_features = df.groupby("user_id").agg({
            "transaction_amount": ["count", "sum", "mean"],
            "transaction_date": ["min", "max"]
        }).reset_index()

        user_features.columns = [
            "user_id", "total_purchases", "total_amount",
            "avg_purchase_amount", "first_purchase_date", "last_purchase_date"
        ]

        # Compute additional features
        today = datetime.now()
        user_features["days_since_last_purchase"] = (
            today - pd.to_datetime(user_features["last_purchase_date"])
        ).dt.days

        user_features["tenure_months"] = (
            pd.to_datetime(user_features["last_purchase_date"]) -
            pd.to_datetime(user_features["first_purchase_date"])
        ).dt.days // 30

        # Add timestamps
        user_features["event_timestamp"] = today
        user_features["created_timestamp"] = today

        return user_features

    def validate_features(self, df: pd.DataFrame) -> bool:
        """Validate features"""
        # Null check
        if df.isnull().sum().sum() > 0:
            print("Warning: Null values detected")
            return False

        # Range check
        if (df["avg_purchase_amount"] < 0).any():
            print("Warning: Negative values in avg_purchase_amount")
            return False

        return True

    def save_features(self, df: pd.DataFrame):
        """Save features"""
        df.to_parquet(
            self.output_path,
            index=False,
            engine="pyarrow"
        )
        print(f"Features saved to {self.output_path}")

    def run(self):
        """Run pipeline"""
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

# Execute
pipeline = FeatureEngineeringPipeline(
    raw_data_path="data/raw_transactions.parquet",
    output_path="feature_repo/data/user_features.parquet"
)
features = pipeline.run()

# Sync Feast features
store = FeatureStore(repo_path="./feature_repo")
store.materialize_incremental(end_date=datetime.now())
```

### 6.2 Streaming Feature Update

```python
"""
Streaming feature update (Feast Push)
"""

from feast import FeatureStore
import pandas as pd
from datetime import datetime

store = FeatureStore(repo_path="./feature_repo")

def process_streaming_event(event: dict):
    """Process real-time events and update features"""
    # Extract features from event
    feature_df = pd.DataFrame([{
        "user_id": event["user_id"],
        "transaction_amount": event["amount"],
        "transaction_timestamp": datetime.now()
    }])

    # Push to Feature Store
    store.push(
        push_source_name="transaction_push_source",
        df=feature_df
    )

# Kafka Consumer example
# for message in kafka_consumer:
#     event = json.loads(message.value)
#     process_streaming_event(event)
```

---

## Practice Exercises

### Exercise 1: Feature Store Setup
Set up Feast Feature Store and define basic features.

### Exercise 2: Training Data Generation
Use get_historical_features to generate point-in-time accurate training data.

### Exercise 3: Online Serving
Configure online store and integrate with real-time inference.

---

## Summary

| Component | Description | Use Case |
|-----------|-------------|----------|
| Offline Store | Large-scale historical data | Training data generation |
| Online Store | Low-latency key-value retrieval | Real-time inference |
| Feature Registry | Feature metadata management | Feature discovery, governance |
| Materialization | Offline → Online sync | Prepare features for serving |

---

## References

- [Feast Documentation](https://docs.feast.dev/)
- [Feature Store for ML](https://www.featurestore.org/)
- [Tecton Feature Platform](https://www.tecton.ai/)

---

## Example Code

Standalone Feast feature store examples with materialization and serving:

**[`examples/MLOps/feature_store/`](../../../examples/MLOps/feature_store/)**

Bridges L11 theory with L12 practical project — same entity/feature schema, runnable offline and online retrieval demos.
