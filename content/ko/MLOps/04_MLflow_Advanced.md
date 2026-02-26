# MLflow 고급

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 엔트리 포인트(entry point), 파라미터, 재현 가능한 환경을 정의하는 MLproject 파일로 MLflow Projects를 생성할 수 있다
2. `mlflow.pyfunc.PythonModel`을 상속하여 비표준 추론 로직을 패키징하는 커스텀 MLflow 모델을 구현할 수 있다
3. 데이터 전처리, 학습, 평가를 부모-자식 실행(parent-child run)으로 연결하는 다단계 MLflow 파이프라인을 구축할 수 있다
4. 프로덕션 배포를 위해 원격 아티팩트 저장소(S3, GCS)와 데이터베이스 기반 트래킹 서버로 MLflow를 구성할 수 있다
5. Optuna 또는 Ray Tune을 사용하여 하이퍼파라미터 튜닝(hyperparameter tuning) 워크플로우에 MLflow 트래킹을 통합할 수 있다

---

## 1. MLflow Projects

MLflow Projects는 재현 가능한 ML 코드 패키징 형식입니다.

### 1.1 프로젝트 구조

```
my_ml_project/
├── MLproject              # 프로젝트 정의 파일
├── conda.yaml             # Conda 환경 정의
├── requirements.txt       # pip 의존성 (선택)
├── train.py               # 학습 스크립트
├── evaluate.py            # 평가 스크립트
└── data/
    └── sample_data.csv
```

### 1.2 MLproject 파일

```yaml
# MLproject
name: churn-prediction

# 환경 정의 (3가지 옵션)
# 옵션 1: Conda
conda_env: conda.yaml

# 옵션 2: Docker
# docker_env:
#   image: my-docker-image:latest

# 옵션 3: System (현재 환경 사용)
# python_env: python_env.yaml

# 엔트리 포인트 분리 — CI/CD에서 전체 파이프라인 실행 없이
# 학습/평가/탐색을 독립적으로 트리거 가능
entry_points:
  main:
    parameters:
      data_path: {type: str, default: "data/train.csv"}
      n_estimators: {type: int, default: 100}
      max_depth: {type: int, default: 5}
      learning_rate: {type: float, default: 0.1}
    command: "python train.py --data-path {data_path} --n-estimators {n_estimators} --max-depth {max_depth} --learning-rate {learning_rate}"

  evaluate:
    parameters:
      model_path: {type: str}
      test_data: {type: str}
    command: "python evaluate.py --model-path {model_path} --test-data {test_data}"

  hyperparameter_search:
    parameters:
      n_trials: {type: int, default: 50}
    command: "python hyperparam_search.py --n-trials {n_trials}"
```

### 1.3 conda.yaml

```yaml
# conda.yaml
name: churn-prediction-env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pip
  - scikit-learn=1.2.0
  - pandas=1.5.0
  - numpy=1.23.0
  - pip:
    - mlflow>=2.0
    - xgboost>=1.7
```

### 1.4 프로젝트 실행

```bash
# 로컬 실행
mlflow run . -P n_estimators=200 -P max_depth=10

# Git에서 직접 실행
mlflow run https://github.com/user/ml-project.git -P data_path=s3://bucket/data.csv

# 특정 브랜치/태그
mlflow run https://github.com/user/ml-project.git --version main

# Docker 환경에서 실행
mlflow run . --env-manager docker

# 특정 엔트리 포인트 실행
mlflow run . -e evaluate -P model_path=models/model.pkl -P test_data=data/test.csv

# 실험 지정
mlflow run . --experiment-name "production-training"
```

### 1.5 train.py 예시

```python
"""
train.py - MLflow Project 학습 스크립트
"""

import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main(args):
    # autolog이 모델 파라미터, 메트릭, 아티팩트를 자동으로 캡처 —
    # 보일러플레이트를 줄이고 중요한 정보의 로깅 누락을 방지
    mlflow.sklearn.autolog()

    # 컨텍스트 매니저(context manager)로 학습 중 크래시 시에도 실행 종료를 보장
    with mlflow.start_run():
        # 데이터 로드
        df = pd.read_csv(args.data_path)
        X = df.drop("target", axis=1)
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 데이터 출처 로깅 — 모델 품질 저하 시 디버깅에 필수;
        # autolog은 데이터 경로 같은 커스텀 메타데이터를 캡처하지 않음
        mlflow.log_param("data_path", args.data_path)
        mlflow.log_param("train_size", len(X_train))

        # 모델 학습
        model = GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=42
        )
        model.fit(X_train, y_train)

        # 평가
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='macro'),
            "recall": recall_score(y_test, y_pred, average='macro'),
            "f1": f1_score(y_test, y_pred, average='macro')
        }

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        print(f"Model trained with accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
```

---

## 2. MLflow Models

### 2.1 모델 플레이버 (Flavors)

```python
"""
MLflow 모델 플레이버
"""

import mlflow

# 지원되는 플레이버
flavors = {
    "sklearn": "scikit-learn 모델",
    "pytorch": "PyTorch 모델",
    "tensorflow": "TensorFlow/Keras 모델",
    "xgboost": "XGBoost 모델",
    "lightgbm": "LightGBM 모델",
    "catboost": "CatBoost 모델",
    "transformers": "HuggingFace Transformers",
    "langchain": "LangChain 모델",
    "onnx": "ONNX 모델",
    "pyfunc": "Python 함수 (커스텀)"
}
```

### 2.2 모델 시그니처

```python
"""
모델 시그니처 정의
"""

import mlflow
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec

# 방법 1: 자동 추론
signature = infer_signature(X_train, model.predict(X_train))

# 방법 2: 명시적 정의
input_schema = Schema([
    ColSpec("double", "feature_1"),
    ColSpec("double", "feature_2"),
    ColSpec("string", "category")
])
output_schema = Schema([ColSpec("long", "prediction")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# 모델 저장 시 시그니처 포함
mlflow.sklearn.log_model(
    model,
    "model",
    signature=signature,
    input_example=X_train[:5]  # 입력 예시
)
```

### 2.3 커스텀 모델 (pyfunc)

```python
"""
커스텀 MLflow 모델 (pyfunc)
"""

import mlflow
import mlflow.pyfunc
import pandas as pd

class CustomModel(mlflow.pyfunc.PythonModel):
    """커스텀 MLflow 모델"""

    # pyfunc이 전처리 + 모델 + 후처리를 하나의 아티팩트로 래핑 —
    # 서빙 환경에서 학습과 정확히 동일한 파이프라인 사용을 보장
    def __init__(self, preprocessor, model, threshold=0.5):
        self.preprocessor = preprocessor
        self.model = model
        self.threshold = threshold

    def load_context(self, context):
        """아티팩트 로드"""
        # 모델 로드 시 한 번만 호출(요청마다가 아님) — 여기서 무거운 초기화를 수행하여
        # 추론 중 반복적인 I/O 오버헤드를 방지
        import joblib
        # context.artifacts에서 추가 파일 로드 가능
        pass

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """예측 수행"""
        # 전처리
        processed = self.preprocessor.transform(model_input)

        # 예측
        probabilities = self.model.predict_proba(processed)[:, 1]

        # 레이블과 확률 모두 반환 — 호출자가 추론을 다시 실행하지 않고도
        # 비즈니스별 임계값을 적용할 수 있음
        predictions = (probabilities >= self.threshold).astype(int)

        return pd.DataFrame({
            "prediction": predictions,
            "probability": probabilities
        })


# 커스텀 모델 저장
custom_model = CustomModel(preprocessor, trained_model, threshold=0.6)

# Conda 환경 정의
conda_env = {
    "channels": ["conda-forge"],
    "dependencies": [
        "python=3.9",
        "pip",
        {"pip": ["mlflow", "scikit-learn", "pandas"]}
    ],
    "name": "custom_model_env"
}

# 외부 아티팩트를 모델과 번들링 — 전처리기와 설정이 모델과 함께 이동하므로
# 서빙 시 "아티팩트 누락" 오류를 제거
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=custom_model,
        conda_env=conda_env,
        artifacts={
            "preprocessor": "artifacts/preprocessor.pkl",
            "config": "artifacts/config.yaml"
        },
        signature=signature,
        input_example=sample_input
    )
```

### 2.4 모델 포맷 저장

```
model/
├── MLmodel                    # 모델 메타데이터
├── model.pkl                  # 직렬화된 모델
├── conda.yaml                 # Conda 환경
├── python_env.yaml            # Python 환경
├── requirements.txt           # pip 의존성
├── input_example.json         # 입력 예시
└── registered_model_meta      # 레지스트리 메타데이터
```

```yaml
# MLmodel 파일 내용
artifact_path: model
flavors:
  python_function:
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    predict_fn: predict
    python_version: 3.9.0
  sklearn:
    code: null
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: 1.2.0
mlflow_version: 2.8.0
model_uuid: a1b2c3d4-e5f6-7890-abcd-ef1234567890
signature:
  inputs: '[{"name": "feature_1", "type": "double"}, ...]'
  outputs: '[{"type": "long"}]'
```

---

## 3. Model Registry

### 3.1 모델 등록

```python
"""
Model Registry 사용법
"""

import mlflow
from mlflow.tracking import MlflowClient

# 방법 1: log_model 시 직접 등록
with mlflow.start_run():
    # 학습...
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="ChurnPredictionModel"  # 자동 등록
    )

# 방법 2: 기존 run에서 등록
result = mlflow.register_model(
    model_uri="runs:/RUN_ID/model",
    name="ChurnPredictionModel"
)
print(f"Version: {result.version}")

# 방법 3: MlflowClient 사용
client = MlflowClient()
client.create_registered_model(
    name="ChurnPredictionModel",
    description="Customer churn prediction model",
    tags={"team": "ML", "project": "retention"}
)

# 버전 추가
client.create_model_version(
    name="ChurnPredictionModel",
    source="runs:/RUN_ID/model",
    run_id="RUN_ID",
    description="Initial version with RF"
)
```

### 3.2 모델 스테이지 관리

```python
"""
모델 스테이지 전환
"""

from mlflow.tracking import MlflowClient

client = MlflowClient()

# 스테이지 종류: None, Staging, Production, Archived

# Staging으로 전환
client.transition_model_version_stage(
    name="ChurnPredictionModel",
    version=1,
    stage="Staging",
    archive_existing_versions=False
)

# Production으로 승격
client.transition_model_version_stage(
    name="ChurnPredictionModel",
    version=1,
    stage="Production",
    # 자동 아카이브로 다중 Production 버전을 방지 — load_model("Production")이
    # 항상 정확히 하나의 모델을 반환하도록 보장
    archive_existing_versions=True
)

# 모델 로드 (스테이지별)
staging_model = mlflow.pyfunc.load_model("models:/ChurnPredictionModel/Staging")
prod_model = mlflow.pyfunc.load_model("models:/ChurnPredictionModel/Production")

# 특정 버전 로드
model_v1 = mlflow.pyfunc.load_model("models:/ChurnPredictionModel/1")
```

### 3.3 모델 메타데이터 관리

```python
"""
모델 버전 메타데이터
"""

from mlflow.tracking import MlflowClient

client = MlflowClient()

# 모델 설명 업데이트
client.update_registered_model(
    name="ChurnPredictionModel",
    description="Updated description"
)

# 버전 설명 업데이트
client.update_model_version(
    name="ChurnPredictionModel",
    version=1,
    description="Improved feature engineering"
)

# 태그 추가
client.set_registered_model_tag(
    name="ChurnPredictionModel",
    key="task",
    value="binary_classification"
)

client.set_model_version_tag(
    name="ChurnPredictionModel",
    version=1,
    key="validated",
    value="true"
)

# 모델 정보 조회
model = client.get_registered_model("ChurnPredictionModel")
print(f"Name: {model.name}")
print(f"Description: {model.description}")
print(f"Latest versions: {model.latest_versions}")

# 버전 정보 조회
version = client.get_model_version("ChurnPredictionModel", 1)
print(f"Version: {version.version}")
print(f"Stage: {version.current_stage}")
print(f"Source: {version.source}")
```

### 3.4 모델 검색

```python
"""
등록된 모델 검색
"""

from mlflow.tracking import MlflowClient

client = MlflowClient()

# 모든 모델 조회
models = client.search_registered_models()
for m in models:
    print(f"Model: {m.name}, Latest: {m.latest_versions}")

# 필터링 검색
models = client.search_registered_models(
    filter_string="name LIKE '%Churn%'"
)

# 버전 검색
versions = client.search_model_versions(
    filter_string="name='ChurnPredictionModel' and current_stage='Production'"
)
for v in versions:
    print(f"Version {v.version}: {v.current_stage}")
```

---

## 4. MLflow Serving

### 4.1 로컬 서빙

```bash
# 모델 서빙 (run ID 기반)
mlflow models serve -m "runs:/RUN_ID/model" -p 5001 --no-conda

# 모델 서빙 (Registry 기반)
mlflow models serve -m "models:/ChurnPredictionModel/Production" -p 5001

# 환경 옵션
mlflow models serve -m "models:/MyModel/1" \
    --env-manager local \
    --host 0.0.0.0 \
    --port 5001
```

### 4.2 REST API 호출

```python
"""
MLflow 서빙 API 호출
"""

import requests
import json

# 엔드포인트
url = "http://localhost:5001/invocations"

# 입력 데이터 (여러 형식 지원)
# 형식 1: split orientation
data_split = {
    "dataframe_split": {
        "columns": ["feature_1", "feature_2", "feature_3"],
        "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    }
}

# 형식 2: records orientation
data_records = {
    "dataframe_records": [
        {"feature_1": 1.0, "feature_2": 2.0, "feature_3": 3.0},
        {"feature_1": 4.0, "feature_2": 5.0, "feature_3": 6.0}
    ]
}

# 형식 3: instances (TensorFlow Serving 호환)
data_instances = {
    "instances": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
}

# API 호출
response = requests.post(
    url,
    headers={"Content-Type": "application/json"},
    data=json.dumps(data_split)
)

print(f"Status: {response.status_code}")
print(f"Predictions: {response.json()}")
```

### 4.3 Docker 이미지 빌드

```bash
# Docker 이미지 생성
mlflow models build-docker \
    -m "models:/ChurnPredictionModel/Production" \
    -n "churn-model:latest"

# 이미지 실행
docker run -p 5001:8080 churn-model:latest

# Dockerfile 직접 생성
mlflow models generate-dockerfile \
    -m "models:/ChurnPredictionModel/Production" \
    -d ./docker-build
```

### 4.4 배치 추론

```python
"""
배치 추론 수행
"""

import mlflow
import pandas as pd

# 모델 로드
model = mlflow.pyfunc.load_model("models:/ChurnPredictionModel/Production")

# 대량 데이터 로드
batch_data = pd.read_parquet("s3://bucket/batch_data.parquet")

# 배치 예측
predictions = model.predict(batch_data)

# 결과 저장
results = batch_data.copy()
results["prediction"] = predictions
results.to_parquet("s3://bucket/predictions.parquet")
```

---

## 5. 고급 설정

### 5.1 원격 Tracking Server

```bash
# 메타데이터(실행, 파라미터, 메트릭)는 PostgreSQL, 대형 아티팩트(모델, 데이터)는 S3에 저장
# — 관심사 분리: 메타데이터는 관계형 쿼리, 바이너리는 저렴한 대량 스토리지 활용
mlflow server \
    --backend-store-uri postgresql://user:password@host:5432/mlflow \
    --default-artifact-root s3://mlflow-artifacts/ \
    --host 0.0.0.0 \
    --port 5000

# 환경 변수 설정
export MLFLOW_TRACKING_URI=http://mlflow-server:5000
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
```

### 5.2 인증 설정

```python
"""
MLflow 인증 설정
"""

import os
import mlflow

# 기본 인증
os.environ["MLFLOW_TRACKING_USERNAME"] = "user"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"

# 토큰 기반 인증
os.environ["MLFLOW_TRACKING_TOKEN"] = "your-token"

# Azure ML 통합
os.environ["AZURE_TENANT_ID"] = "tenant-id"
os.environ["AZURE_CLIENT_ID"] = "client-id"
os.environ["AZURE_CLIENT_SECRET"] = "client-secret"
```

### 5.3 플러그인 사용

```python
"""
MLflow 플러그인 예시
"""

# Databricks 플러그인
# pip install databricks-cli

import mlflow
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/user@email.com/my-experiment")

# Google Cloud 플러그인
# pip install mlflow[google-cloud]
mlflow.set_tracking_uri("gs://bucket/mlflow")
```

---

## 6. 실전 워크플로우

```python
"""
전체 MLflow 워크플로우 예시
"""

import mlflow
from mlflow.tracking import MlflowClient

# 1. 설정
mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("production-churn-model")
client = MlflowClient()

# 2. 학습 및 실험
with mlflow.start_run(run_name="rf-optimized") as run:
    # 학습 코드...
    mlflow.sklearn.log_model(model, "model", signature=signature)
    run_id = run.info.run_id

# 3. 모델 등록
model_version = mlflow.register_model(
    f"runs:/{run_id}/model",
    "ChurnPredictionModel"
)

# 4. Staging 전환
client.transition_model_version_stage(
    name="ChurnPredictionModel",
    version=model_version.version,
    stage="Staging"
)

# 5. Staging에서 먼저 테스트 — 실제 트래픽에 영향을 주기 전에
# 프로덕션과 유사한 환경에서 모델 로드와 임계값 검증을 확인
staging_model = mlflow.pyfunc.load_model("models:/ChurnPredictionModel/Staging")
test_results = evaluate_model(staging_model, test_data)

# 6. 메트릭 기반 승격 게이트 — 회귀(regression)가 프로덕션에 도달하는 것을 방지
if test_results["accuracy"] > 0.9:
    client.transition_model_version_stage(
        name="ChurnPredictionModel",
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Model v{model_version.version} promoted to Production!")
```

---

## 연습 문제

### 문제 1: MLflow Project 생성
완전한 MLflow Project를 생성하고 로컬에서 실행하세요.

### 문제 2: 커스텀 pyfunc 모델
전처리와 후처리를 포함하는 커스텀 pyfunc 모델을 작성하세요.

### 문제 3: Model Registry 워크플로우
모델을 등록하고 Staging -> Production 전환을 자동화하세요.

---

## 요약

| 기능 | 설명 |
|------|------|
| MLflow Projects | 재현 가능한 코드 패키징 |
| MLflow Models | 표준화된 모델 포맷 |
| Model Registry | 모델 버전 및 스테이지 관리 |
| MLflow Serving | REST API 서빙 |
| pyfunc | 커스텀 모델 래퍼 |

---

## 참고 자료

- [MLflow Projects](https://mlflow.org/docs/latest/projects.html)
- [MLflow Models](https://mlflow.org/docs/latest/models.html)
- [Model Registry](https://mlflow.org/docs/latest/model-registry.html)
