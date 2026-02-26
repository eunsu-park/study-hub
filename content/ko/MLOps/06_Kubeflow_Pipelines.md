# Kubeflow Pipelines

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Kubeflow 생태계 컴포넌트인 Pipelines, Katib, Training Operators, KServe, Notebooks를 설명하고 각각이 ML 플랫폼에서 수행하는 역할을 서술할 수 있다
2. `@component` 데코레이터를 사용하여 Python 함수를 컨테이너화된 스텝으로 정의하고 KFP SDK로 Kubeflow Pipeline 컴포넌트와 파이프라인을 구축할 수 있다
3. 제어 흐름(control flow) 구조를 사용하여 Kubeflow Pipelines에서 조건부 분기(conditional branching)와 병렬 실행(parallel execution)을 구현할 수 있다
4. Kubernetes 클러스터에서 베이지안 최적화(Bayesian optimization)를 사용한 자동 하이퍼파라미터 튜닝을 위해 Katib를 구성할 수 있다
5. Kubernetes에서 A/B 테스트(A/B testing)와 카나리 롤아웃(canary rollout)을 위해 KServe를 사용하여 학습된 모델을 배포하고 관리할 수 있다

---

## 1. Kubeflow 개요

Kubeflow는 Kubernetes 위에서 ML 워크플로우를 구축, 배포, 관리하기 위한 오픈소스 플랫폼입니다.

### 1.1 Kubeflow 컴포넌트

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Kubeflow 생태계                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│   │  Pipelines  │    │   Katib     │    │  Training   │            │
│   │             │    │             │    │  Operators  │            │
│   │ - 워크플로우 │    │ - AutoML    │    │             │            │
│   │ - 파이프라인 │    │ - 하이퍼    │    │ - TFJob     │            │
│   │   오케스트   │    │   파라미터  │    │ - PyTorchJob│            │
│   │   레이션    │    │   튜닝      │    │ - MXJob     │            │
│   └─────────────┘    └─────────────┘    └─────────────┘            │
│                                                                     │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│   │   KServe    │    │  Notebooks  │    │   Central   │            │
│   │             │    │             │    │  Dashboard  │            │
│   │ - 모델 서빙  │    │ - Jupyter   │    │             │            │
│   │ - A/B 테스트│    │   환경      │    │ - UI       │            │
│   │ - 카나리    │    │             │    │ - 관리      │            │
│   └─────────────┘    └─────────────┘    └─────────────┘            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 설치

```bash
# KFP SDK 설치
pip install kfp

# 버전 확인
python -c "import kfp; print(kfp.__version__)"

# Kubeflow Pipelines 클러스터 설치 (minikube 예시)
# https://www.kubeflow.org/docs/started/installing-kubeflow/
```

---

## 2. Kubeflow Pipelines SDK

### 2.1 기본 개념

```python
"""
KFP 기본 개념
"""

# 핵심 용어
kfp_concepts = {
    "Pipeline": "ML 워크플로우를 정의하는 DAG (Directed Acyclic Graph)",
    "Component": "파이프라인의 개별 단계 (함수 또는 컨테이너)",
    "Run": "파이프라인의 한 번 실행",
    "Experiment": "관련 실행들의 그룹",
    "Artifact": "컴포넌트 간 전달되는 데이터"
}
```

### 2.2 간단한 파이프라인

```python
"""
KFP v2 기본 파이프라인
"""

from kfp import dsl
from kfp import compiler

# 각 @dsl.component는 자체 컨테이너에서 실행 — 스텝 간 의존성 충돌
# (예: 다른 sklearn 버전)이 절대 발생하지 않도록 격리
@dsl.component
def preprocess_data(input_path: str, output_path: dsl.OutputPath(str)):
    """데이터 전처리 컴포넌트"""
    import pandas as pd

    df = pd.read_csv(input_path)
    # 전처리 로직
    df_processed = df.dropna()
    df_processed.to_csv(output_path, index=False)

@dsl.component
def train_model(
    data_path: str,
    n_estimators: int,
    model_path: dsl.OutputPath(str)
):
    """모델 학습 컴포넌트"""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X, y)

    joblib.dump(model, model_path)

@dsl.component
def evaluate_model(
    model_path: str,
    test_data_path: str
) -> float:
    """모델 평가 컴포넌트"""
    import pandas as pd
    from sklearn.metrics import accuracy_score
    import joblib

    model = joblib.load(model_path)
    df = pd.read_csv(test_data_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    return accuracy

# 파이프라인 정의
@dsl.pipeline(
    name="ML Training Pipeline",
    description="A simple ML training pipeline"
)
def ml_pipeline(
    input_data_path: str = "gs://bucket/data.csv",
    n_estimators: int = 100
):
    # 출력으로 스텝이 연결됨 — KFP가 자동으로 DAG 의존성을 생성하므로
    # 명시적 순서 지정 없이 2단계가 1단계 완료를 대기
    preprocess_task = preprocess_data(input_path=input_data_path)

    # 학습이 전처리 출력을 소비 — 아티팩트가 클라우드 스토리지를 통해 전달되므로
    # 스텝이 다른 머신에서 실행 가능
    train_task = train_model(
        data_path=preprocess_task.outputs["output_path"],
        n_estimators=n_estimators
    )

    # 단계 3: 평가
    evaluate_task = evaluate_model(
        model_path=train_task.outputs["model_path"],
        test_data_path=preprocess_task.outputs["output_path"]
    )

# YAML로 컴파일 — 컴파일된 스펙은 이식 가능하고 git에 버전 관리되어
# 파이프라인 변경이 코드와 동일한 리뷰 프로세스를 거칠 수 있음
compiler.Compiler().compile(
    pipeline_func=ml_pipeline,
    package_path="ml_pipeline.yaml"
)
```

---

## 3. 컴포넌트 작성

### 3.1 Python 함수 컴포넌트

```python
"""
Python 함수 기반 컴포넌트
"""

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics

# base_image와 packages를 명시적으로 지정 — 재현성 보장;
# 이것 없이는 기본 이미지를 상속하여 시간이 지나면서 드리프트 발생 가능
@dsl.component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas", "scikit-learn"]
)
def train_sklearn_model(
    training_data: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    n_estimators: int = 100,
    max_depth: int = 5
):
    """scikit-learn 모델 학습"""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import joblib
    import json

    # 데이터 로드
    df = pd.read_csv(training_data.path)
    X = df.drop("target", axis=1)
    y = df["target"]

    # 모델 학습
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    clf.fit(X, y)

    # 교차 검증
    cv_scores = cross_val_score(clf, X, y, cv=5)

    # 모델 저장
    joblib.dump(clf, model.path)

    # 메트릭 로깅
    metrics.log_metric("cv_mean", float(cv_scores.mean()))
    metrics.log_metric("cv_std", float(cv_scores.std()))
    metrics.log_metric("n_features", int(X.shape[1]))

# GPU 사용 컴포넌트
@dsl.component(
    base_image="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
    packages_to_install=["transformers"]
)
def train_pytorch_model(
    data: Input[Dataset],
    model: Output[Model],
    epochs: int = 10,
    learning_rate: float = 0.001
):
    """PyTorch 모델 학습 (GPU)"""
    import torch
    # GPU 학습 코드...
    pass
```

### 3.2 컨테이너 기반 컴포넌트

```python
"""
Docker 컨테이너 기반 컴포넌트
"""

from kfp import dsl
from kfp.dsl import ContainerSpec

# 방법 1: container_spec 직접 정의
@dsl.container_component
def custom_training_component(
    data_path: str,
    output_path: dsl.OutputPath(str),
    epochs: int
):
    return ContainerSpec(
        image="gcr.io/my-project/training-image:latest",
        command=["python", "train.py"],
        args=[
            "--data-path", data_path,
            "--output-path", output_path,
            "--epochs", str(epochs)
        ]
    )

# 방법 2: YAML 컴포넌트 정의
component_yaml = """
name: Training Component
description: Custom training component
inputs:
  - name: data_path
    type: String
  - name: epochs
    type: Integer
    default: '10'
outputs:
  - name: model_path
    type: String
implementation:
  container:
    image: gcr.io/my-project/training:latest
    command:
      - python
      - train.py
    args:
      - --data-path
      - {inputValue: data_path}
      - --epochs
      - {inputValue: epochs}
      - --output-path
      - {outputPath: model_path}
"""

# YAML에서 컴포넌트 로드
from kfp.components import load_component_from_text
training_op = load_component_from_text(component_yaml)
```

### 3.3 재사용 가능한 컴포넌트

```python
"""
재사용 가능한 컴포넌트 라이브러리
"""

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model

# components/data_processing.py
@dsl.component(packages_to_install=["pandas", "numpy"])
def load_and_split_data(
    input_path: str,
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    test_size: float = 0.2,
    random_state: int = 42
):
    """데이터 로드 및 분할"""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_path)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    train_df.to_csv(train_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)

@dsl.component(packages_to_install=["pandas", "scikit-learn"])
def feature_engineering(
    input_data: Input[Dataset],
    output_data: Output[Dataset],
    numerical_features: list,
    categorical_features: list
):
    """피처 엔지니어링"""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    df = pd.read_csv(input_data.path)

    # 수치형 스케일링
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # 범주형 인코딩
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    df.to_csv(output_data.path, index=False)
```

---

## 4. 파이프라인 고급 기능

### 4.1 조건부 실행

```python
"""
조건부 실행과 분기
"""

from kfp import dsl

@dsl.component
def check_data_quality(data_path: str) -> bool:
    """데이터 품질 검사"""
    import pandas as pd
    df = pd.read_csv(data_path)
    # 품질 검사 로직
    return df.isnull().sum().sum() == 0

@dsl.component
def clean_data(data_path: str, output_path: dsl.OutputPath(str)):
    """데이터 정제"""
    import pandas as pd
    df = pd.read_csv(data_path)
    df = df.dropna()
    df.to_csv(output_path, index=False)

@dsl.component
def train_model(data_path: str):
    """모델 학습"""
    pass

@dsl.pipeline(name="conditional-pipeline")
def conditional_pipeline(data_path: str):
    # 데이터 품질 검사
    quality_check = check_data_quality(data_path=data_path)

    # 조건부 실행
    with dsl.Condition(quality_check.output == False, name="need-cleaning"):
        clean_task = clean_data(data_path=data_path)
        train_model(data_path=clean_task.outputs["output_path"])

    with dsl.Condition(quality_check.output == True, name="no-cleaning"):
        train_model(data_path=data_path)
```

### 4.2 반복 실행

```python
"""
ParallelFor를 사용한 반복 실행
"""

from kfp import dsl
from typing import List

@dsl.component
def train_with_params(
    data_path: str,
    params: dict
) -> float:
    """파라미터로 모델 학습"""
    # 학습 로직
    return 0.95

@dsl.component
def select_best_model(
    accuracies: List[float],
    param_sets: List[dict]
) -> dict:
    """최고 성능 모델 선택"""
    best_idx = accuracies.index(max(accuracies))
    return param_sets[best_idx]

@dsl.pipeline(name="hyperparameter-search")
def hyperparameter_search_pipeline(data_path: str):
    # 하이퍼파라미터 조합
    param_sets = [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 10}
    ]

    # 병렬 학습
    with dsl.ParallelFor(param_sets) as params:
        train_task = train_with_params(
            data_path=data_path,
            params=params
        )

    # 결과 수집 (ParallelFor 밖에서)
    # select_best_model(...)
```

### 4.3 리소스 설정

```python
"""
컴포넌트 리소스 설정
"""

from kfp import dsl
from kfp import kubernetes

@dsl.component
def gpu_training(data_path: str):
    """GPU 학습"""
    pass

@dsl.pipeline(name="resource-pipeline")
def resource_pipeline(data_path: str):
    # GPU 학습 태스크
    train_task = gpu_training(data_path=data_path)

    # request vs limit: request는 스케줄링을 위한 최소 리소스 예약,
    # limit는 최대치를 제한하여 한 학습 작업이 다른 pod의 리소스를 빼앗지 못하도록 방지
    train_task.set_cpu_limit("4")
    train_task.set_memory_limit("16Gi")
    train_task.set_cpu_request("2")
    train_task.set_memory_request("8Gi")

    # GPU 설정
    kubernetes.add_node_selector(
        train_task,
        label_key="cloud.google.com/gke-accelerator",
        label_value="nvidia-tesla-t4"
    )
    train_task.set_accelerator_type("nvidia.com/gpu")
    train_task.set_accelerator_limit(1)

    # 환경 변수
    train_task.set_env_variable("CUDA_VISIBLE_DEVICES", "0")

    # 볼륨 마운트
    kubernetes.mount_pvc(
        train_task,
        pvc_name="data-pvc",
        mount_path="/data"
    )
```

---

## 5. Kubernetes 통합

### 5.1 시크릿 및 ConfigMap

```python
"""
Kubernetes 리소스 연동
"""

from kfp import dsl
from kfp import kubernetes

@dsl.component
def component_with_secrets():
    """시크릿을 사용하는 컴포넌트"""
    import os
    api_key = os.environ.get("API_KEY")
    # ...

@dsl.pipeline(name="k8s-resources-pipeline")
def k8s_pipeline():
    task = component_with_secrets()

    # 시크릿에서 환경 변수
    kubernetes.use_secret_as_env(
        task,
        secret_name="api-credentials",
        secret_key_to_env={"api-key": "API_KEY"}
    )

    # ConfigMap에서 환경 변수
    kubernetes.use_config_map_as_env(
        task,
        config_map_name="app-config",
        config_map_key_to_env={"setting": "APP_SETTING"}
    )

    # 시크릿을 파일로 마운트
    kubernetes.use_secret_as_volume(
        task,
        secret_name="tls-certs",
        mount_path="/certs"
    )
```

### 5.2 서비스 어카운트

```python
"""
서비스 어카운트 설정
"""

from kfp import dsl
from kfp import kubernetes

@dsl.pipeline(name="sa-pipeline")
def service_account_pipeline():
    task = some_component()

    # 특정 서비스 어카운트 사용
    kubernetes.set_service_account(
        task,
        service_account="ml-pipeline-sa"
    )

    # 이미지 풀 시크릿
    kubernetes.add_image_pull_secret(
        task,
        secret_name="docker-registry-secret"
    )
```

---

## 6. 파이프라인 실행

### 6.1 SDK로 실행

```python
"""
KFP SDK로 파이프라인 실행
"""

from kfp import compiler
from kfp.client import Client

# 파이프라인 컴파일
compiler.Compiler().compile(
    pipeline_func=ml_pipeline,
    package_path="pipeline.yaml"
)

# KFP 클라이언트 생성
client = Client(host="http://kubeflow-pipelines-api:8888")

# 실험 생성 또는 가져오기
experiment = client.create_experiment(
    name="my-experiment",
    description="ML training experiments"
)

# 파이프라인 실행
run = client.create_run_from_pipeline_package(
    pipeline_file="pipeline.yaml",
    experiment_id=experiment.experiment_id,
    run_name="training-run-001",
    arguments={
        "input_data_path": "gs://bucket/data.csv",
        "n_estimators": 200
    }
)

print(f"Run ID: {run.run_id}")
print(f"Run URL: {client.get_run(run.run_id).display_url}")

# 실행 완료 대기
client.wait_for_run_completion(run.run_id, timeout=3600)

# 실행 상태 확인
run_details = client.get_run(run.run_id)
print(f"Status: {run_details.state}")
```

### 6.2 스케줄링

```python
"""
파이프라인 스케줄링
"""

from kfp.client import Client

client = Client(host="http://kubeflow-pipelines-api:8888")

# 반복 실행 (cron) 생성
recurring_run = client.create_recurring_run(
    experiment_id=experiment.experiment_id,
    job_name="daily-training",
    pipeline_package_path="pipeline.yaml",
    cron_expression="0 2 * * *",  # 매일 오전 2시
    max_concurrency=1,
    arguments={
        "input_data_path": "gs://bucket/daily_data/"
    }
)

print(f"Recurring Run ID: {recurring_run.id}")

# 반복 실행 비활성화
client.disable_recurring_run(recurring_run.id)

# 반복 실행 활성화
client.enable_recurring_run(recurring_run.id)
```

### 6.3 실행 결과 조회

```python
"""
실행 결과 및 아티팩트 조회
"""

from kfp.client import Client

client = Client()

# 특정 실행 결과
run = client.get_run("run-id")
print(f"State: {run.state}")
print(f"Created: {run.created_at}")
print(f"Finished: {run.finished_at}")

# 실행 목록 조회
runs = client.list_runs(
    experiment_id=experiment.experiment_id,
    page_size=10,
    sort_by="created_at desc"
)

for r in runs.runs:
    print(f"{r.name}: {r.state}")

# 파이프라인 출력 조회
# (아티팩트는 보통 GCS, S3 등에 저장됨)
```

---

## 7. 실전 파이프라인 예제

```python
"""
완전한 ML 파이프라인 예제
"""

from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Model, Metrics

@dsl.component(packages_to_install=["pandas", "scikit-learn"])
def ingest_data(
    source_path: str,
    output_data: Output[Dataset]
):
    """데이터 수집"""
    import pandas as pd
    df = pd.read_csv(source_path)
    df.to_csv(output_data.path, index=False)

@dsl.component(packages_to_install=["pandas", "great-expectations"])
def validate_data(
    input_data: Input[Dataset],
    validation_report: Output[Dataset]
) -> bool:
    """데이터 검증"""
    import pandas as pd
    df = pd.read_csv(input_data.path)

    # 검증 로직
    is_valid = (
        len(df) > 100 and
        df.isnull().sum().sum() / df.size < 0.1
    )

    # 리포트 저장
    report = {"valid": is_valid, "rows": len(df)}
    pd.DataFrame([report]).to_csv(validation_report.path, index=False)

    return is_valid

@dsl.component(packages_to_install=["pandas", "scikit-learn", "joblib"])
def train_and_evaluate(
    train_data: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    n_estimators: int = 100
) -> float:
    """모델 학습 및 평가"""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    import joblib

    df = pd.read_csv(train_data.path)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    joblib.dump(clf, model.path)

    metrics.log_metric("accuracy", float(accuracy))
    metrics.log_metric("f1_score", float(f1))

    return accuracy

@dsl.component
def deploy_model(
    model: Input[Model],
    accuracy: float,
    min_accuracy: float = 0.8
) -> str:
    """모델 배포"""
    if accuracy < min_accuracy:
        return "Model not deployed: accuracy below threshold"

    # 배포 로직 (예: 모델을 서빙 인프라에 배포)
    return f"Model deployed with accuracy {accuracy}"

@dsl.pipeline(
    name="End-to-End ML Pipeline",
    description="Complete ML pipeline with data validation, training, and deployment"
)
def e2e_ml_pipeline(
    data_source: str = "gs://bucket/data.csv",
    n_estimators: int = 100,
    min_accuracy: float = 0.8
):
    # 1. 데이터 수집
    ingest_task = ingest_data(source_path=data_source)

    # 2. 데이터 검증
    validate_task = validate_data(
        input_data=ingest_task.outputs["output_data"]
    )

    # 3. 검증 통과 시 학습
    # 조건부 게이트 — 데이터 품질이 낮으면 비용이 큰 학습을 건너뛰어
    # 손상된 데이터에 GPU 시간을 낭비하는 것을 방지
    with dsl.Condition(validate_task.output == True, name="data-valid"):
        train_task = train_and_evaluate(
            train_data=ingest_task.outputs["output_data"],
            n_estimators=n_estimators
        )

        # 4. 배포
        deploy_model(
            model=train_task.outputs["model"],
            accuracy=train_task.output,
            min_accuracy=min_accuracy
        )

# 컴파일
compiler.Compiler().compile(
    pipeline_func=e2e_ml_pipeline,
    package_path="e2e_pipeline.yaml"
)
```

---

## 연습 문제

### 문제 1: 기본 파이프라인
3단계 파이프라인을 작성하세요: 데이터 로드 -> 전처리 -> 모델 학습

### 문제 2: 하이퍼파라미터 검색
ParallelFor를 사용하여 여러 하이퍼파라미터 조합을 병렬로 실험하는 파이프라인을 작성하세요.

### 문제 3: 스케줄링
매주 월요일 새벽에 실행되는 재학습 파이프라인을 설정하세요.

---

## 요약

| 개념 | 설명 |
|------|------|
| Pipeline | ML 워크플로우 DAG |
| Component | 파이프라인의 개별 단계 |
| @dsl.component | Python 함수 컴포넌트 |
| @dsl.pipeline | 파이프라인 정의 |
| dsl.Condition | 조건부 실행 |
| dsl.ParallelFor | 병렬 반복 실행 |

---

## 참고 자료

- [Kubeflow Pipelines Documentation](https://www.kubeflow.org/docs/components/pipelines/)
- [KFP SDK v2](https://kubeflow-pipelines.readthedocs.io/)
- [Kubeflow Examples](https://github.com/kubeflow/examples)
