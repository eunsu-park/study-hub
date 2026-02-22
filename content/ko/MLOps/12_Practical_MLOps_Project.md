# 실전 MLOps 프로젝트

## 1. E2E MLOps 파이프라인 설계

실제 프로덕션 환경에서 작동하는 완전한 MLOps 파이프라인을 구축합니다.

### 1.1 프로젝트 개요

```
┌─────────────────────────────────────────────────────────────────────┐
│                     E2E MLOps 파이프라인                             │
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

### 1.2 기술 스택

```yaml
# 프로젝트 기술 스택
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

## 2. 프로젝트 구조

### 2.1 디렉토리 구조

```
mlops-project/
├── .github/
│   └── workflows/
│       ├── ci.yaml              # CI 파이프라인
│       ├── train.yaml           # 학습 파이프라인
│       └── deploy.yaml          # 배포 파이프라인
├── data/
│   ├── raw/                     # 원시 데이터
│   └── processed/               # 전처리된 데이터
├── features/
│   ├── feature_store.yaml       # Feast 설정
│   ├── entities.py              # Entity 정의
│   └── feature_views.py         # Feature View 정의
├── src/
│   ├── data/
│   │   ├── ingestion.py         # 데이터 수집
│   │   ├── validation.py        # 데이터 검증
│   │   └── preprocessing.py     # 전처리
│   ├── features/
│   │   └── engineering.py       # 피처 엔지니어링
│   ├── training/
│   │   ├── train.py             # 학습 스크립트
│   │   ├── evaluate.py          # 평가 스크립트
│   │   └── hyperparameter.py    # 하이퍼파라미터 튜닝
│   ├── serving/
│   │   ├── handler.py           # 모델 핸들러
│   │   └── api.py               # API 서버
│   └── monitoring/
│       ├── drift.py             # 드리프트 감지
│       └── metrics.py           # 메트릭 수집
├── pipelines/
│   ├── training_pipeline.py     # Kubeflow 학습 파이프라인
│   └── serving_pipeline.py      # 배포 파이프라인
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
├── MLproject                    # MLflow 프로젝트
├── pyproject.toml               # 의존성 관리
└── README.md
```

### 2.2 설정 파일

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

## 3. 데이터 파이프라인

### 3.1 데이터 수집 및 검증

```python
"""
src/data/ingestion.py - 데이터 수집
"""

import pandas as pd
from typing import Dict, Any
import great_expectations as ge
from datetime import datetime

class DataIngestion:
    """데이터 수집 클래스"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def ingest(self, source_path: str) -> pd.DataFrame:
        """데이터 수집"""
        df = pd.read_parquet(source_path)
        df["ingestion_timestamp"] = datetime.now()
        return df

    def validate(self, df: pd.DataFrame) -> bool:
        """데이터 검증"""
        ge_df = ge.from_pandas(df)

        # 기본 검증
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
        """검증된 데이터 저장"""
        df.to_parquet(output_path, index=False)
```

### 3.2 피처 엔지니어링

```python
"""
src/features/engineering.py - 피처 엔지니어링
"""

import pandas as pd
from feast import FeatureStore
from datetime import datetime
from typing import List

class FeatureEngineering:
    """피처 엔지니어링 클래스"""

    def __init__(self, feature_store_path: str):
        self.store = FeatureStore(repo_path=feature_store_path)

    def get_training_features(
        self,
        entity_df: pd.DataFrame,
        feature_list: List[str]
    ) -> pd.DataFrame:
        """학습용 피처 조회"""
        training_df = self.store.get_historical_features(
            entity_df=entity_df,
            features=feature_list
        ).to_df()

        return training_df

    def create_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """피처 생성"""
        features = raw_df.copy()

        # 집계 피처
        features["purchase_frequency"] = (
            features["total_purchases"] / features["tenure_months"].clip(lower=1)
        )

        # 범주형 피처 인코딩
        features = pd.get_dummies(
            features,
            columns=["customer_segment"],
            prefix="segment"
        )

        return features

    def materialize(self):
        """피처 동기화"""
        self.store.materialize_incremental(end_date=datetime.now())
```

---

## 4. 학습 파이프라인

### 4.1 모델 학습

```python
"""
src/training/train.py - 모델 학습
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
    """모델 학습 클래스"""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        mlflow.set_tracking_uri(self.config["training"]["tracking_uri"])
        mlflow.set_experiment(self.config["training"]["experiment_name"])

    def prepare_data(self, df: pd.DataFrame):
        """데이터 준비"""
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
        """모델 학습"""
        with mlflow.start_run() as run:
            # 파라미터 로깅
            params = self.config["model"]["params"]
            mlflow.log_params(params)
            mlflow.log_param("model_type", self.config["model"]["type"])

            # 모델 학습
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            # 교차 검증
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            mlflow.log_metric("cv_mean", cv_scores.mean())
            mlflow.log_metric("cv_std", cv_scores.std())

            # 검증
            y_pred = model.predict(X_val)
            metrics = self._calculate_metrics(y_val, y_pred)

            for name, value in metrics.items():
                mlflow.log_metric(name, value)

            # 모델 저장
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
        """메트릭 계산"""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro"),
            "recall": recall_score(y_true, y_pred, average="macro"),
            "f1_score": f1_score(y_true, y_pred, average="macro")
        }

    def validate_quality_gates(self, metrics: Dict[str, float]) -> bool:
        """품질 게이트 검증"""
        gates = self.config["quality_gates"]
        passed = all(
            metrics.get(metric, 0) >= threshold
            for metric, threshold in gates.items()
        )
        return passed
```

### 4.2 Kubeflow 파이프라인

```python
"""
pipelines/training_pipeline.py - Kubeflow 학습 파이프라인
"""

from kfp import dsl
from kfp import compiler
from kfp.dsl import Input, Output, Dataset, Model, Metrics

@dsl.component(packages_to_install=["pandas", "pyarrow"])
def ingest_data(
    source_path: str,
    output_data: Output[Dataset]
):
    """데이터 수집 컴포넌트"""
    import pandas as pd
    df = pd.read_parquet(source_path)
    df.to_parquet(output_data.path, index=False)

@dsl.component(packages_to_install=["pandas", "great-expectations"])
def validate_data(
    input_data: Input[Dataset],
    output_data: Output[Dataset]
) -> bool:
    """데이터 검증 컴포넌트"""
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
    """모델 학습 컴포넌트"""
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
    """모델 배포 컴포넌트"""
    if accuracy < min_accuracy:
        return f"Deployment skipped: accuracy {accuracy} < {min_accuracy}"

    # 배포 로직
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
    # 1. 데이터 수집
    ingest_task = ingest_data(source_path=data_source)

    # 2. 데이터 검증
    validate_task = validate_data(
        input_data=ingest_task.outputs["output_data"]
    )

    # 3. 조건부 학습
    with dsl.Condition(validate_task.output == True):
        train_task = train_model(
            input_data=validate_task.outputs["output_data"],
            mlflow_uri=mlflow_uri,
            experiment_name=experiment_name
        )

        # 4. 배포
        deploy_model(
            model=train_task.outputs["model"],
            accuracy=train_task.outputs["metrics"].accuracy
        )

# 컴파일
compiler.Compiler().compile(
    training_pipeline,
    "training_pipeline.yaml"
)
```

---

## 5. 배포 및 서빙

### 5.1 모델 서빙 API

```python
"""
src/serving/api.py - 모델 서빙 API
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

# Prometheus 메트릭
PREDICTIONS = Counter("predictions_total", "Total predictions", ["status"])
LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")

# 모델 및 Feature Store 로드
model = None
store = None

class PredictionRequest(BaseModel):
    user_id: int

class PredictionResponse(BaseModel):
    user_id: int
    churn_probability: float
    prediction: str
    features: dict

@app.on_event("startup")
async def load_resources():
    global model, store
    model = mlflow.sklearn.load_model("models:/churn-prediction/Production")
    store = FeatureStore(repo_path="./features")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()

    try:
        # Feature Store에서 피처 조회
        features = store.get_online_features(
            features=[
                "user_features:total_purchases",
                "user_features:avg_purchase_amount",
                "user_features:tenure_months"
            ],
            entity_rows=[{"user_id": request.user_id}]
        ).to_dict()

        # 피처 벡터 생성
        feature_vector = np.array([[
            features["total_purchases"][0],
            features["avg_purchase_amount"][0],
            features["tenure_months"][0]
        ]])

        # 예측
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
        PREDICTIONS.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### 5.2 Kubernetes 배포

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

## 6. 자동 재학습

### 6.1 재학습 트리거

```python
"""
src/monitoring/retrain_trigger.py - 자동 재학습 트리거
"""

from datetime import datetime, timedelta
from typing import Dict, Any
import mlflow
from src.monitoring.drift import DriftDetector
from src.training.train import ModelTrainer

class RetrainingOrchestrator:
    """자동 재학습 오케스트레이터"""

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
        """재학습 필요 여부 확인 및 실행"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "retrained": False,
            "reason": None
        }

        # 쿨다운 체크
        if self._in_cooldown():
            result["reason"] = "In cooldown period"
            return result

        # 재학습 조건 확인
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
        """재학습 조건 확인"""
        # 1. 성능 저하
        for metric, threshold in self.config["quality_thresholds"].items():
            if metrics.get(metric, 1.0) < threshold:
                return True, f"Performance degradation: {metric}={metrics[metric]}"

        # 2. 데이터 드리프트
        drift_result = self.drift_detector.detect(reference_data, current_data)
        if drift_result["is_drift"]:
            return True, f"Data drift detected: {drift_result['drift_score']}"

        # 3. 예정된 재학습
        if self.config.get("scheduled_retrain_days"):
            days_since = (datetime.now() - self.last_retrain).days if self.last_retrain else float("inf")
            if days_since >= self.config["scheduled_retrain_days"]:
                return True, "Scheduled retraining"

        return False, None

    def _in_cooldown(self) -> bool:
        """쿨다운 기간 확인"""
        if self.last_retrain is None:
            return False
        return datetime.now() - self.last_retrain < self.cooldown

    def _execute_retraining(self) -> Dict[str, Any]:
        """재학습 실행"""
        # 새 데이터로 학습
        training_result = self.trainer.train_on_latest_data()

        # 품질 게이트 통과 시 배포
        if self.trainer.validate_quality_gates(training_result["metrics"]):
            self._deploy_model(training_result["run_id"])
            training_result["deployed"] = True
        else:
            training_result["deployed"] = False

        return training_result

    def _deploy_model(self, run_id: str):
        """모델 배포"""
        client = mlflow.tracking.MlflowClient()
        model_uri = f"runs:/{run_id}/model"

        # 모델 등록 및 Production으로 승격
        result = mlflow.register_model(model_uri, self.config["model_name"])
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
    - cron: "0 2 * * *"  # 매일 오전 2시
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

## 7. 체크리스트

### 프로덕션 준비 체크리스트

```markdown
## MLOps 프로덕션 체크리스트

### 데이터
- [ ] 데이터 검증 파이프라인 구현
- [ ] 데이터 버전 관리 (DVC)
- [ ] 데이터 스키마 검증
- [ ] PII 데이터 마스킹

### 피처
- [ ] Feature Store 설정
- [ ] 피처 버전 관리
- [ ] 온라인/오프라인 동기화
- [ ] 피처 문서화

### 학습
- [ ] 실험 추적 (MLflow)
- [ ] 하이퍼파라미터 튜닝
- [ ] 품질 게이트 정의
- [ ] 모델 레지스트리 설정

### 서빙
- [ ] REST API 구현
- [ ] 헬스 체크 엔드포인트
- [ ] 수평 확장 (HPA)
- [ ] 로드 밸런싱

### 모니터링
- [ ] 성능 메트릭 수집
- [ ] 드리프트 감지
- [ ] 알림 설정
- [ ] 대시보드 구성

### CI/CD
- [ ] 자동 테스트
- [ ] 자동 배포
- [ ] 롤백 전략
- [ ] 자동 재학습 트리거
```

---

## 연습 문제

### 프로젝트 과제
위의 구조를 참고하여 고객 이탈 예측 MLOps 시스템을 처음부터 구축하세요.

1. Feature Store 설정 및 피처 정의
2. 학습 파이프라인 구현
3. 모델 서빙 API 구현
4. 드리프트 모니터링 설정
5. 자동 재학습 트리거 구현

---

## 요약

| 단계 | 핵심 기술 | 산출물 |
|------|----------|--------|
| 데이터 | Great Expectations, DVC | 검증된 데이터 |
| 피처 | Feast | Feature Store |
| 학습 | MLflow, Kubeflow | 학습된 모델 |
| 서빙 | FastAPI, K8s | API 엔드포인트 |
| 모니터링 | Evidently, Prometheus | 대시보드, 알림 |
| 자동화 | GitHub Actions | CI/CD 파이프라인 |

---

## 참고 자료

- [Made With ML - MLOps](https://madewithml.com/)
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/)
- [MLOps Community](https://mlops.community/)

---

## 예제 코드

합성 데이터와 Docker Compose 인프라를 포함한 실행 가능한 E2E MLOps 프로젝트:

**[`examples/MLOps/practical_project/`](../../../examples/MLOps/practical_project/)**

Feast 피처 스토어(Feature Store), MLflow 학습, FastAPI 서빙, Evidently 드리프트 감지 포함 — 모두 로컬 실행에 맞게 조정되었습니다.
