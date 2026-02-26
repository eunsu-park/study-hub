# 12. Practical MLOps Project

## Learning Objectives

After completing this lesson, you will be able to:

1. Architect a complete end-to-end MLOps pipeline that integrates data sourcing, a feature store, model training, model registry, serving, and drift monitoring into a cohesive system
2. Select and justify a technology stack (Kubernetes, Airflow, MLflow, Feast, TorchServe, Prometheus/Grafana) for a production MLOps project
3. Implement a CI/CD pipeline with GitHub Actions that automatically runs tests, retrains the model, and deploys it upon code or data changes
4. Integrate all MLOps components — feature store, experiment tracker, model registry, serving endpoint, and monitoring — into a working project
5. Evaluate the production readiness of an MLOps pipeline by assessing observability, rollback capability, and failure recovery procedures

---

## 1. E2E MLOps Pipeline Design

Build a complete MLOps pipeline that operates in a real production environment.

### 1.1 Project Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     E2E MLOps Pipeline                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     │
│   │  Data   │────▶│ Feature │────▶│ Model   │────▶│  Model  │     │
│   │ Source  │     │  Store  │     │Training │     │Registry │     │
│   └─────────┘     └─────────┘     └─────────┘     └────┬────┘     │
│                                                         │          │
│                                                         ▼          │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     │
│   │Retrain  │◀────│  Drift  │◀────│  Model  │◀────│  Model  │     │
│   │Trigger  │     │Detection│     │ Monitor │     │ Serving │     │
│   └────┬────┘     └─────────┘     └─────────┘     └─────────┘     │
│        │                                                            │
│        └──────────────────────────────────────────────────────────▶│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Technology Stack

```yaml
# Project technology stack
infrastructure:
  orchestration: Kubernetes
  ci_cd: GitHub Actions
  iac: Terraform

data_pipeline:
  batch: Apache Airflow
  streaming: Kafka
  storage: S3, PostgreSQL

ml_platform:
  experiment_tracking: MLflow
  feature_store: Feast
  model_registry: MLflow Model Registry

serving:
  inference_server: TorchServe
  api_gateway: Kong
  load_balancer: AWS ALB

monitoring:
  metrics: Prometheus
  visualization: Grafana
  drift_detection: Evidently
  alerting: PagerDuty, Slack
```

---

## 2. Project Structure

### 2.1 Directory Structure

```
mlops-project/
├── .github/
│   └── workflows/
│       ├── ci.yaml              # CI pipeline
│       ├── train.yaml           # Training pipeline
│       └── deploy.yaml          # Deployment pipeline
├── data/
│   ├── raw/                     # Raw data
│   └── processed/               # Processed data
├── features/
│   ├── feature_store.yaml       # Feast config
│   ├── entities.py              # Entity definitions
│   └── feature_views.py         # Feature View definitions
├── src/
│   ├── data/
│   │   ├── ingestion.py         # Data collection
│   │   ├── validation.py        # Data validation
│   │   └── preprocessing.py     # Preprocessing
│   ├── features/
│   │   └── engineering.py       # Feature engineering
│   ├── training/
│   │   ├── train.py             # Training script
│   │   ├── evaluate.py          # Evaluation script
│   │   └── hyperparameter.py    # Hyperparameter tuning
│   ├── serving/
│   │   ├── handler.py           # Model handler
│   │   └── api.py               # API server
│   └── monitoring/
│       ├── drift.py             # Drift detection
│       └── metrics.py           # Metrics collection
├── pipelines/
│   ├── training_pipeline.py     # Kubeflow training pipeline
│   └── serving_pipeline.py      # Deployment pipeline
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
├── configs/
│   ├── training_config.yaml
│   ├── serving_config.yaml
│   └── monitoring_config.yaml
├── docker/
│   ├── Dockerfile.train
│   ├── Dockerfile.serve
│   └── docker-compose.yaml
├── kubernetes/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
├── MLproject                    # MLflow project
├── pyproject.toml               # Dependency management
└── README.md
```

### 2.2 Configuration Files

```yaml
# configs/training_config.yaml
project:
  name: churn-prediction
  version: "1.0.0"

data:
  source: s3://bucket/data/
  train_path: train.parquet
  test_path: test.parquet
  validation_split: 0.2

features:
  store_path: ./features
  entity: user_id
  features:
    - user_features:total_purchases
    - user_features:avg_purchase_amount
    - user_features:tenure_months

model:
  type: random_forest
  params:
    n_estimators: 200
    max_depth: 10
    random_state: 42

training:
  experiment_name: churn-prediction
  tracking_uri: http://mlflow:5000
  epochs: 100
  batch_size: 32

quality_gates:
  accuracy: 0.85
  precision: 0.80
  recall: 0.75
```

---

## 3. Data Pipeline

### 3.1 Data Ingestion and Validation

```python
"""
src/data/ingestion.py - Data collection
"""

import pandas as pd
from typing import Dict, Any
import great_expectations as ge
from datetime import datetime

class DataIngestion:
    """Data ingestion class"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def ingest(self, source_path: str) -> pd.DataFrame:
        """Ingest data"""
        df = pd.read_parquet(source_path)
        df["ingestion_timestamp"] = datetime.now()
        return df

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate data"""
        # Great Expectations provides declarative assertions — fail fast before expensive training
        ge_df = ge.from_pandas(df)

        results = [
            ge_df.expect_column_to_exist("user_id"),
            ge_df.expect_column_to_exist("target"),
            ge_df.expect_column_values_to_not_be_null("user_id"),
            ge_df.expect_column_values_to_be_between(
                "age", min_value=18, max_value=120
            ),
            ge_df.expect_table_row_count_to_be_between(
                min_value=1000, max_value=None
            )
        ]

        return all(r.success for r in results)

    def save(self, df: pd.DataFrame, output_path: str):
        """Save validated data"""
        df.to_parquet(output_path, index=False)
```

### 3.2 Feature Engineering

```python
"""
src/features/engineering.py - Feature engineering
"""

import pandas as pd
from feast import FeatureStore
from datetime import datetime
from typing import List

class FeatureEngineering:
    """Feature engineering class"""

    def __init__(self, feature_store_path: str):
        self.store = FeatureStore(repo_path=feature_store_path)

    def get_training_features(
        self,
        entity_df: pd.DataFrame,
        feature_list: List[str]
    ) -> pd.DataFrame:
        """Retrieve training features"""
        training_df = self.store.get_historical_features(
            entity_df=entity_df,
            features=feature_list
        ).to_df()

        return training_df

    def create_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Create features"""
        features = raw_df.copy()

        # Aggregation features
        # clip(lower=1) prevents division by zero for new users with 0 tenure
        features["purchase_frequency"] = (
            features["total_purchases"] / features["tenure_months"].clip(lower=1)
        )

        # Categorical feature encoding
        features = pd.get_dummies(
            features,
            columns=["customer_segment"],
            prefix="segment"
        )

        return features

    def materialize(self):
        """Sync features"""
        self.store.materialize_incremental(end_date=datetime.now())
```

---

## 4. Training Pipeline

### 4.1 Model Training

```python
"""
src/training/train.py - Model training
"""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import yaml
from typing import Dict, Any

class ModelTrainer:
    """Model training class"""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        mlflow.set_tracking_uri(self.config["training"]["tracking_uri"])
        mlflow.set_experiment(self.config["training"]["experiment_name"])

    def prepare_data(self, df: pd.DataFrame):
        """Prepare data"""
        feature_columns = [
            col for col in df.columns
            if col not in ["user_id", "target", "event_timestamp"]
        ]

        X = df[feature_columns]
        y = df["target"]

        return train_test_split(
            X, y,
            test_size=self.config["data"]["validation_split"],
            random_state=42,
            stratify=y
        )

    def train(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """Train model"""
        # Context manager ensures run closure even if training crashes mid-epoch
        with mlflow.start_run() as run:
            params = self.config["model"]["params"]
            mlflow.log_params(params)
            mlflow.log_param("model_type", self.config["model"]["type"])

            # Train model
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            # CV detects overfitting early — high cv_std signals unstable performance
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            mlflow.log_metric("cv_mean", cv_scores.mean())
            mlflow.log_metric("cv_std", cv_scores.std())

            # Validation
            y_pred = model.predict(X_val)
            metrics = self._calculate_metrics(y_val, y_pred)

            for name, value in metrics.items():
                mlflow.log_metric(name, value)

            # Signature enforces input/output schema — catches shape mismatches at serving time
            signature = mlflow.models.infer_signature(X_train, y_pred)
            mlflow.sklearn.log_model(
                model, "model",
                signature=signature,
                registered_model_name=self.config["project"]["name"]
            )

            return {
                "run_id": run.info.run_id,
                "metrics": metrics,
                "model": model
            }

    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate metrics"""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro"),
            "recall": recall_score(y_true, y_pred, average="macro"),
            "f1_score": f1_score(y_true, y_pred, average="macro")
        }

    def validate_quality_gates(self, metrics: Dict[str, float]) -> bool:
        """Validate quality gates"""
        # Quality gates prevent deploying a model worse than current production
        gates = self.config["quality_gates"]
        passed = all(
            metrics.get(metric, 0) >= threshold
            for metric, threshold in gates.items()
        )
        return passed
```

### 4.2 Kubeflow Pipeline

```python
"""
pipelines/training_pipeline.py - Kubeflow training pipeline
"""

from kfp import dsl
from kfp import compiler
from kfp.dsl import Input, Output, Dataset, Model, Metrics

@dsl.component(packages_to_install=["pandas", "pyarrow"])
def ingest_data(
    source_path: str,
    output_data: Output[Dataset]
):
    """Data ingestion component"""
    import pandas as pd
    df = pd.read_parquet(source_path)
    df.to_parquet(output_data.path, index=False)

@dsl.component(packages_to_install=["pandas", "great-expectations"])
def validate_data(
    input_data: Input[Dataset],
    output_data: Output[Dataset]
) -> bool:
    """Data validation component"""
    import pandas as pd
    import great_expectations as ge

    df = pd.read_parquet(input_data.path)
    ge_df = ge.from_pandas(df)

    is_valid = ge_df.expect_table_row_count_to_be_between(
        min_value=1000
    ).success

    if is_valid:
        df.to_parquet(output_data.path, index=False)

    return is_valid

@dsl.component(packages_to_install=["pandas", "scikit-learn", "mlflow", "feast"])
def train_model(
    input_data: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    mlflow_uri: str,
    experiment_name: str
):
    """Model training component"""
    import pandas as pd
    import mlflow
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    df = pd.read_parquet(input_data.path)
    X = df.drop(["target", "user_id"], axis=1, errors="ignore")
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    with mlflow.start_run():
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, clf.predict(X_test))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(clf, "model")

        metrics.log_metric("accuracy", float(accuracy))
        joblib.dump(clf, model.path)

@dsl.component
def deploy_model(
    model: Input[Model],
    accuracy: float,
    min_accuracy: float = 0.85
) -> str:
    """Model deployment component"""
    if accuracy < min_accuracy:
        return f"Deployment skipped: accuracy {accuracy} < {min_accuracy}"

    # Deployment logic
    return "Model deployed successfully"

@dsl.pipeline(
    name="E2E Training Pipeline",
    description="Complete training pipeline with validation and deployment"
)
def training_pipeline(
    data_source: str = "s3://bucket/data.parquet",
    mlflow_uri: str = "http://mlflow:5000",
    experiment_name: str = "churn-prediction"
):
    # 1. Data ingestion
    ingest_task = ingest_data(source_path=data_source)

    # 2. Data validation
    validate_task = validate_data(
        input_data=ingest_task.outputs["output_data"]
    )

    # Conditional gate: skip expensive GPU training if data validation fails
    with dsl.Condition(validate_task.output == True):
        train_task = train_model(
            input_data=validate_task.outputs["output_data"],
            mlflow_uri=mlflow_uri,
            experiment_name=experiment_name
        )

        # 4. Deployment
        deploy_model(
            model=train_task.outputs["model"],
            accuracy=train_task.outputs["metrics"].accuracy
        )

# Compile
compiler.Compiler().compile(
    training_pipeline,
    "training_pipeline.yaml"
)
```

---

## 5. Deployment and Serving

### 5.1 Model Serving API

```python
"""
src/serving/api.py - Model serving API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from feast import FeatureStore
import mlflow
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
import time

app = FastAPI(title="Churn Prediction API")

# Prometheus metrics
PREDICTIONS = Counter("predictions_total", "Total predictions", ["status"])
LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")

# Load model and Feature Store
model = None
store = None

class PredictionRequest(BaseModel):
    user_id: int

class PredictionResponse(BaseModel):
    user_id: int
    churn_probability: float
    prediction: str
    features: dict

# Startup event loads model once — avoids 100-500ms model deserialization per request
@app.on_event("startup")
async def load_resources():
    global model, store
    # URI "models:/name/Production" always resolves to the latest Production stage
    model = mlflow.sklearn.load_model("models:/churn-prediction/Production")
    store = FeatureStore(repo_path="./features")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()

    try:
        # Retrieve features from Feature Store
        features = store.get_online_features(
            features=[
                "user_features:total_purchases",
                "user_features:avg_purchase_amount",
                "user_features:tenure_months"
            ],
            entity_rows=[{"user_id": request.user_id}]
        ).to_dict()

        # Create feature vector
        feature_vector = np.array([[
            features["total_purchases"][0],
            features["avg_purchase_amount"][0],
            features["tenure_months"][0]
        ]])

        # Predict
        probability = model.predict_proba(feature_vector)[0][1]
        prediction = "High Risk" if probability > 0.5 else "Low Risk"

        PREDICTIONS.labels(status="success").inc()
        LATENCY.observe(time.time() - start_time)

        return PredictionResponse(
            user_id=request.user_id,
            churn_probability=float(probability),
            prediction=prediction,
            features={
                "total_purchases": features["total_purchases"][0],
                "avg_purchase_amount": features["avg_purchase_amount"][0],
                "tenure_months": features["tenure_months"][0]
            }
        )

    except Exception as e:
        # Track error rate separately — spikes indicate feature store or model issues
        PREDICTIONS.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### 5.2 Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-prediction-api
  labels:
    app: churn-prediction
spec:
  replicas: 3
  selector:
    matchLabels:
      app: churn-prediction
  template:
    metadata:
      labels:
        app: churn-prediction
      # Prometheus annotations enable auto-discovery — no manual scrape config needed
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
        - name: api
          image: churn-prediction-api:latest
          ports:
            - containerPort: 8000
          env:
            - name: MLFLOW_TRACKING_URI
              value: "http://mlflow:5000"
            - name: FEATURE_STORE_PATH
              value: "/app/features"
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
          # Liveness restarts hung pods; readiness gates traffic until model is loaded
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
          volumeMounts:
            - name: feature-store
              mountPath: /app/features
      volumes:
        - name: feature-store
          configMap:
            name: feature-store-config
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: churn-prediction-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: churn-prediction-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

---

## 6. Automated Retraining

### 6.1 Retraining Trigger

```python
"""
src/monitoring/retrain_trigger.py - Automated retraining trigger
"""

from datetime import datetime, timedelta
from typing import Dict, Any
import mlflow
from src.monitoring.drift import DriftDetector
from src.training.train import ModelTrainer

class RetrainingOrchestrator:
    """Automated retraining orchestrator"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.drift_detector = DriftDetector()
        self.trainer = ModelTrainer(config["training_config"])
        self.last_retrain = None
        self.cooldown = timedelta(hours=config.get("cooldown_hours", 24))

    def check_and_retrain(
        self,
        reference_data,
        current_data,
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Check retraining need and execute"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "retrained": False,
            "reason": None
        }

        # Cooldown check
        if self._in_cooldown():
            result["reason"] = "In cooldown period"
            return result

        # Check retraining conditions
        should_retrain, reason = self._should_retrain(
            reference_data, current_data, current_metrics
        )

        if should_retrain:
            result["retrained"] = True
            result["reason"] = reason
            result["training_result"] = self._execute_retraining()
            self.last_retrain = datetime.now()

        return result

    def _should_retrain(
        self,
        reference_data,
        current_data,
        metrics: Dict[str, float]
    ) -> tuple[bool, str]:
        """Check retraining conditions"""
        # Priority order: performance (reactive) → drift (proactive) → schedule (safety net)
        for metric, threshold in self.config["quality_thresholds"].items():
            if metrics.get(metric, 1.0) < threshold:
                return True, f"Performance degradation: {metric}={metrics[metric]}"

        # 2. Data drift
        drift_result = self.drift_detector.detect(reference_data, current_data)
        if drift_result["is_drift"]:
            return True, f"Data drift detected: {drift_result['drift_score']}"

        # 3. Scheduled retraining
        if self.config.get("scheduled_retrain_days"):
            days_since = (datetime.now() - self.last_retrain).days if self.last_retrain else float("inf")
            if days_since >= self.config["scheduled_retrain_days"]:
                return True, "Scheduled retraining"

        return False, None

    def _in_cooldown(self) -> bool:
        """Check cooldown period"""
        if self.last_retrain is None:
            return False
        return datetime.now() - self.last_retrain < self.cooldown

    def _execute_retraining(self) -> Dict[str, Any]:
        """Execute retraining"""
        # Train with new data
        training_result = self.trainer.train_on_latest_data()

        # Quality gate prevents deploying a worse model — retraining doesn't guarantee improvement
        if self.trainer.validate_quality_gates(training_result["metrics"]):
            self._deploy_model(training_result["run_id"])
            training_result["deployed"] = True
        else:
            training_result["deployed"] = False

        return training_result

    def _deploy_model(self, run_id: str):
        """Deploy model"""
        client = mlflow.tracking.MlflowClient()
        model_uri = f"runs:/{run_id}/model"

        # Register model and promote to Production
        result = mlflow.register_model(model_uri, self.config["model_name"])
        # archive_existing_versions=True ensures only one Production model at a time
        client.transition_model_version_stage(
            name=self.config["model_name"],
            version=result.version,
            stage="Production",
            archive_existing_versions=True
        )
```

### 6.2 GitHub Actions CI/CD

```yaml
# .github/workflows/train.yaml
name: Model Training Pipeline

on:
  schedule:
    - cron: "0 2 * * *"  # Daily at 2 AM
  workflow_dispatch:
    inputs:
      force_retrain:
        description: 'Force retraining'
        required: false
        default: 'false'

jobs:
  check-drift:
    runs-on: ubuntu-latest
    outputs:
      should_retrain: ${{ steps.drift.outputs.should_retrain }}
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Check drift
        id: drift
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_URI }}
        run: |
          result=$(python scripts/check_drift.py)
          echo "should_retrain=$result" >> $GITHUB_OUTPUT

  # Train only runs if drift detected or manually forced — saves compute costs
  train:
    needs: check-drift
    if: needs.check-drift.outputs.should_retrain == 'true' || github.event.inputs.force_retrain == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Train model
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_URI }}
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          python src/training/train.py --config configs/training_config.yaml

      - name: Validate model
        run: |
          python scripts/validate_model.py

  deploy:
    needs: train
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Deploy to production
        env:
          KUBECONFIG: ${{ secrets.KUBECONFIG }}
        run: |
          kubectl rollout restart deployment/churn-prediction-api
```

---

## 7. Checklist

### Production Readiness Checklist

```markdown
## MLOps Production Checklist

### Data
- [ ] Implement data validation pipeline
- [ ] Data versioning (DVC)
- [ ] Data schema validation
- [ ] PII data masking

### Features
- [ ] Set up Feature Store
- [ ] Feature versioning
- [ ] Online/offline sync
- [ ] Feature documentation

### Training
- [ ] Experiment tracking (MLflow)
- [ ] Hyperparameter tuning
- [ ] Define quality gates
- [ ] Set up model registry

### Serving
- [ ] Implement REST API
- [ ] Health check endpoints
- [ ] Horizontal scaling (HPA)
- [ ] Load balancing

### Monitoring
- [ ] Collect performance metrics
- [ ] Drift detection
- [ ] Alert configuration
- [ ] Dashboard setup

### CI/CD
- [ ] Automated testing
- [ ] Automated deployment
- [ ] Rollback strategy
- [ ] Automated retraining trigger
```

---

## Practice Exercise

### Project Assignment
Build a complete customer churn prediction MLOps system from scratch using the structure above.

1. Set up Feature Store and define features
2. Implement training pipeline
3. Implement model serving API
4. Set up drift monitoring
5. Implement automated retraining trigger

---

## Summary

| Stage | Key Technology | Output |
|-------|---------------|--------|
| Data | Great Expectations, DVC | Validated data |
| Features | Feast | Feature Store |
| Training | MLflow, Kubeflow | Trained model |
| Serving | FastAPI, K8s | API endpoint |
| Monitoring | Evidently, Prometheus | Dashboard, alerts |
| Automation | GitHub Actions | CI/CD pipeline |

---

## References

- [Made With ML - MLOps](https://madewithml.com/)
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/)
- [MLOps Community](https://mlops.community/)

---

## Example Code

Runnable end-to-end MLOps project with synthetic data and Docker Compose infrastructure:

**[`examples/MLOps/practical_project/`](../../../examples/MLOps/practical_project/)**

Includes Feast feature store, MLflow training, FastAPI serving, and Evidently drift detection — all adapted for local execution.
