# MLflow 기초

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. MLflow의 네 가지 핵심 컴포넌트인 Tracking, Projects, Models, Model Registry를 설명하고 각각이 ML 라이프사이클에서 수행하는 역할을 서술할 수 있다
2. 머신러닝 학습 실행에서 파라미터(parameter), 메트릭(metric), 아티팩트(artifact)를 로깅하여 MLflow로 실험 추적(experiment tracking)을 구현할 수 있다
3. MLflow의 모델 포맷을 사용하여 ML 모델을 패키징하고 MLflow 모델 서버로 로컬에서 서빙할 수 있다
4. MLflow Model Registry를 사용하여 모델 버전을 관리하고 스테이징(staging)에서 프로덕션(production)으로의 스테이지 전환(stage transition)을 관리할 수 있다
5. 중앙 집중식 MLflow 트래킹 서버를 설정하고 팀 기반 ML 워크플로우에 통합할 수 있다

---

## 1. MLflow 개요

MLflow는 머신러닝 라이프사이클을 관리하기 위한 오픈소스 플랫폼입니다. 실험 추적, 모델 패키징, 배포를 통합적으로 지원합니다.

### 1.1 MLflow의 4가지 컴포넌트

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MLflow 컴포넌트                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│   │  Tracking   │    │  Projects   │    │   Models    │            │
│   │             │    │             │    │             │            │
│   │ - 실험 추적  │    │ - 재현 가능한│    │ - 모델 포맷  │            │
│   │ - 메트릭    │    │   프로젝트   │    │ - 다양한    │            │
│   │ - 파라미터  │    │ - 의존성 관리│    │   플레이버   │            │
│   │ - 아티팩트  │    │             │    │             │            │
│   └─────────────┘    └─────────────┘    └─────────────┘            │
│                                                                     │
│   ┌─────────────────────────────────────────────────────┐          │
│   │                   Model Registry                     │          │
│   │                                                      │          │
│   │  - 모델 버전 관리  - 스테이지 전환  - 모델 설명      │          │
│   │                                                      │          │
│   └─────────────────────────────────────────────────────┘          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 설치 및 설정

```bash
# MLflow 설치
pip install mlflow

# 추가 의존성 (선택)
pip install mlflow[extras]  # 모든 추가 기능
pip install mlflow[sklearn]  # scikit-learn 지원
pip install mlflow[pytorch]  # PyTorch 지원

# 버전 확인
mlflow --version
```

---

## 2. MLflow Tracking

### 2.1 기본 개념

```python
"""
MLflow Tracking 기본 개념
"""

# 핵심 용어
mlflow_concepts = {
    "Experiment": "관련 실행들의 그룹 (예: 'churn-prediction')",
    "Run": "하나의 학습 실행 (파라미터, 메트릭, 아티팩트 포함)",
    "Parameters": "입력 설정 (learning_rate, epochs 등)",
    "Metrics": "출력 결과 (accuracy, loss 등)",
    "Artifacts": "파일 (모델, 그래프, 데이터 등)",
    "Tags": "실행에 대한 메타데이터"
}
```

### 2.2 첫 번째 실험

```python
"""
MLflow 기본 사용법
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 데이터 준비
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 실험 설정
mlflow.set_experiment("iris-classification")

# 실행 시작
with mlflow.start_run(run_name="random-forest-baseline"):
    # 1. 파라미터 로깅
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
    }
    mlflow.log_params(params)

    # 2. 모델 학습
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # 3. 예측
    y_pred = model.predict(X_test)

    # 4. 메트릭 로깅
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='macro'),
        "recall": recall_score(y_test, y_pred, average='macro'),
        "f1": f1_score(y_test, y_pred, average='macro')
    }
    mlflow.log_metrics(metrics)

    # 5. 모델 로깅
    mlflow.sklearn.log_model(model, "model")

    # 6. 태그 추가
    mlflow.set_tag("model_type", "RandomForest")
    mlflow.set_tag("developer", "ML Team")

    # 실행 정보 출력
    run = mlflow.active_run()
    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")
    print(f"Metrics: {metrics}")
```

### 2.3 파라미터 및 메트릭 로깅

```python
"""
다양한 로깅 방법
"""

import mlflow
import numpy as np

with mlflow.start_run():
    # 단일 파라미터
    mlflow.log_param("learning_rate", 0.001)

    # 다중 파라미터
    mlflow.log_params({
        "batch_size": 32,
        "epochs": 100,
        "optimizer": "adam"
    })

    # 단일 메트릭
    mlflow.log_metric("accuracy", 0.95)

    # 다중 메트릭
    mlflow.log_metrics({
        "precision": 0.93,
        "recall": 0.91,
        "f1": 0.92
    })

    # 스텝별 메트릭 (학습 곡선)
    for epoch in range(100):
        train_loss = 1.0 / (epoch + 1) + np.random.random() * 0.1
        val_loss = 1.0 / (epoch + 1) + np.random.random() * 0.15
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

    # 태그 (검색 가능한 메타데이터)
    mlflow.set_tag("data_version", "v2.0")
    mlflow.set_tag("experiment_type", "baseline")

    # 다중 태그
    mlflow.set_tags({
        "feature_set": "full",
        "preprocessing": "standardized"
    })
```

### 2.4 아티팩트 로깅

```python
"""
아티팩트 로깅
"""

import mlflow
import matplotlib.pyplot as plt
import pandas as pd
import json

with mlflow.start_run():
    # 1. 파일 로깅
    # 단일 파일
    with open("config.json", "w") as f:
        json.dump({"key": "value"}, f)
    mlflow.log_artifact("config.json")

    # 디렉토리 전체
    mlflow.log_artifacts("./outputs", artifact_path="results")

    # 2. 그래프 로깅
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_title("Training Curve")
    mlflow.log_figure(fig, "training_curve.png")
    plt.close()

    # 3. DataFrame 로깅 (CSV)
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    df.to_csv("data.csv", index=False)
    mlflow.log_artifact("data.csv")

    # 4. 딕셔너리를 JSON으로
    results = {"accuracy": 0.95, "model": "RF"}
    mlflow.log_dict(results, "results.json")

    # 5. 텍스트 로깅
    mlflow.log_text("This is a log message", "log.txt")
```

---

## 3. MLflow UI

### 3.1 서버 시작

```bash
# 로컬 서버 시작 (기본)
mlflow ui

# 포트 지정
mlflow ui --port 5000

# 호스트 지정 (외부 접속 허용)
mlflow ui --host 0.0.0.0 --port 5000

# 백엔드 저장소 지정
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000
```

### 3.2 Tracking URI 설정

```python
"""
Tracking URI 설정 방법
"""

import mlflow

# 방법 1: 코드에서 설정
mlflow.set_tracking_uri("http://localhost:5000")

# 방법 2: 환경 변수
# export MLFLOW_TRACKING_URI=http://localhost:5000

# 방법 3: 파일 기반 (기본값)
mlflow.set_tracking_uri("file:///path/to/mlruns")

# 현재 설정 확인
print(mlflow.get_tracking_uri())
```

### 3.3 UI 기능 활용

```python
"""
UI에서 활용할 수 있는 기능들
"""

# 1. 실험 비교를 위한 구조화된 로깅
experiments_to_compare = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 10}
]

for params in experiments_to_compare:
    with mlflow.start_run():
        mlflow.log_params(params)
        # 학습 및 평가
        accuracy = train_and_evaluate(params)
        mlflow.log_metric("accuracy", accuracy)

# 2. 검색 가능한 태그 추가
with mlflow.start_run():
    mlflow.set_tags({
        "model_type": "RandomForest",
        "feature_version": "v2",
        "data_split": "stratified"
    })

# 3. 실행 검색 (API)
runs = mlflow.search_runs(
    experiment_names=["iris-classification"],
    filter_string="metrics.accuracy > 0.9 and params.max_depth = '5'",
    order_by=["metrics.accuracy DESC"]
)
print(runs[["run_id", "params.n_estimators", "metrics.accuracy"]])
```

---

## 4. 기본 사용 예제

### 4.1 scikit-learn 모델

```python
"""
scikit-learn 모델 전체 예제
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 데이터 로드
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# 실험 설정
mlflow.set_experiment("wine-classification")

# 하이퍼파라미터 그리드
param_grid = [
    {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3},
    {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 5},
    {"n_estimators": 200, "learning_rate": 0.01, "max_depth": 7}
]

for params in param_grid:
    with mlflow.start_run(run_name=f"gb-n{params['n_estimators']}"):
        # 파라미터 로깅
        mlflow.log_params(params)
        mlflow.log_param("test_size", 0.2)

        # 파이프라인 생성
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GradientBoostingClassifier(**params, random_state=42))
        ])

        # 교차 검증
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())

        # 최종 학습
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # 메트릭 로깅
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        mlflow.log_metrics({
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred, average='macro'),
            "test_recall": recall_score(y_test, y_pred, average='macro')
        })

        # Confusion Matrix 시각화
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close()

        # Feature Importance
        classifier = pipeline.named_steps['classifier']
        fig, ax = plt.subplots(figsize=(10, 6))
        importance = classifier.feature_importances_
        indices = np.argsort(importance)[::-1]
        ax.barh(range(len(importance)), importance[indices])
        ax.set_yticks(range(len(importance)))
        ax.set_yticklabels([wine.feature_names[i] for i in indices])
        ax.set_title('Feature Importance')
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close()

        # 모델 로깅
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"Params: {params}")
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### 4.2 PyTorch 모델

```python
"""
PyTorch 모델 MLflow 추적
"""

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

# 데이터 준비
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 모델 정의
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 실험 설정
mlflow.set_experiment("pytorch-classification")

# 하이퍼파라미터
params = {
    "hidden_dim": 64,
    "learning_rate": 0.001,
    "epochs": 50,
    "batch_size": 32
}

with mlflow.start_run():
    mlflow.log_params(params)

    # 모델 초기화
    model = SimpleNN(20, params["hidden_dim"], 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    # 학습
    model.train()
    for epoch in range(params["epochs"]):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

        # 검증 (매 10 에폭)
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                outputs = model(X_test_t)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y_test_t).float().mean().item()
                mlflow.log_metric("val_accuracy", accuracy, step=epoch)
            model.train()

    # 최종 평가
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs, 1)
        test_accuracy = (predicted == y_test_t).float().mean().item()

    mlflow.log_metric("test_accuracy", test_accuracy)

    # 모델 로깅
    mlflow.pytorch.log_model(model, "model")

    print(f"Test Accuracy: {test_accuracy:.4f}")
```

---

## 5. Autologging

### 5.1 자동 로깅 설정

```python
"""
MLflow Autologging
"""

import mlflow

# 전체 프레임워크 자동 로깅
mlflow.autolog()

# 특정 프레임워크만
mlflow.sklearn.autolog()
mlflow.pytorch.autolog()
mlflow.tensorflow.autolog()
mlflow.xgboost.autolog()
mlflow.lightgbm.autolog()

# 자동 로깅 비활성화
mlflow.autolog(disable=True)
```

### 5.2 Autologging 예시

```python
"""
sklearn autologging 예시
"""

import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Autologging 활성화
mlflow.sklearn.autolog(
    log_input_examples=True,      # 입력 예시 로깅
    log_model_signatures=True,    # 모델 시그니처 로깅
    log_models=True,              # 모델 로깅
    log_datasets=True,            # 데이터셋 정보 로깅
    silent=False                  # 로깅 메시지 출력
)

# 데이터 준비
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 실험 (자동으로 모든 것이 로깅됨)
mlflow.set_experiment("autolog-demo")

# 모델 학습 (자동으로 run 생성 및 로깅)
model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

# 자동으로 로깅되는 항목:
# - 파라미터: n_estimators, max_depth, ...
# - 메트릭: training_score, ...
# - 아티팩트: model, feature_importance, ...
```

---

## 6. 모델 로드 및 예측

### 6.1 저장된 모델 로드

```python
"""
저장된 모델 로드 방법
"""

import mlflow
import mlflow.sklearn

# 방법 1: Run ID로 로드
model = mlflow.sklearn.load_model("runs:/RUN_ID/model")

# 방법 2: 아티팩트 경로로 로드
model = mlflow.sklearn.load_model("file:///path/to/mlruns/0/run_id/artifacts/model")

# 방법 3: Model Registry에서 로드 (다음 레슨에서 자세히)
model = mlflow.sklearn.load_model("models:/MyModel/Production")

# 방법 4: pyfunc으로 로드 (프레임워크 무관)
model = mlflow.pyfunc.load_model("runs:/RUN_ID/model")

# 예측
predictions = model.predict(X_test)
```

### 6.2 최근 실행 결과 조회

```python
"""
실험 결과 조회
"""

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 실험 조회
experiment = client.get_experiment_by_name("iris-classification")
print(f"Experiment ID: {experiment.experiment_id}")

# 실행 검색
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.accuracy DESC"],
    max_results=5
)

for run in runs:
    print(f"Run ID: {run.info.run_id}")
    print(f"  Accuracy: {run.data.metrics.get('accuracy')}")
    print(f"  Params: {run.data.params}")

# 최고 성능 모델 로드
best_run = runs[0]
best_model = mlflow.sklearn.load_model(f"runs:/{best_run.info.run_id}/model")
```

---

## 연습 문제

### 문제 1: 기본 실험 추적
Titanic 데이터셋을 사용하여 생존 예측 모델을 학습하고, MLflow로 실험을 추적하세요.

```python
# 힌트
import mlflow
from sklearn.datasets import fetch_openml

titanic = fetch_openml("titanic", version=1, as_frame=True)
# 전처리 후 모델 학습
# mlflow로 파라미터, 메트릭 로깅
```

### 문제 2: 하이퍼파라미터 비교
서로 다른 하이퍼파라미터로 5개 이상의 실험을 실행하고, MLflow UI에서 비교하세요.

---

## 요약

| 기능 | 메서드 | 설명 |
|------|--------|------|
| 실험 설정 | `mlflow.set_experiment()` | 실험 그룹 지정 |
| 실행 시작 | `mlflow.start_run()` | 새 실행 시작 |
| 파라미터 | `mlflow.log_param(s)()` | 입력 파라미터 로깅 |
| 메트릭 | `mlflow.log_metric(s)()` | 출력 메트릭 로깅 |
| 아티팩트 | `mlflow.log_artifact(s)()` | 파일 로깅 |
| 모델 | `mlflow.sklearn.log_model()` | 모델 저장 |
| 자동 로깅 | `mlflow.autolog()` | 자동 추적 활성화 |

---

## 참고 자료

- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Quickstart](https://mlflow.org/docs/latest/quickstart.html)
