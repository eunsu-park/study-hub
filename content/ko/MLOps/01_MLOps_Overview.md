# MLOps 개요

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. MLOps를 정의하고 머신러닝 모델 개발과 신뢰할 수 있는 프로덕션 배포 사이의 간극을 어떻게 해소하는지 설명할 수 있다
2. MLOps의 네 가지 핵심 원칙인 재현성(reproducibility), 자동화(automation), 모니터링(monitoring), 버전 관리(versioning)를 설명하고 각각이 왜 필요한지 서술할 수 있다
3. MLOps의 세 가지 성숙도 수준(maturity level)을 파악하고 특정 조직이나 프로젝트에 적합한 수준을 결정할 수 있다
4. 데이터 검증, 피처 엔지니어링(feature engineering), 학습, 평가, 배포를 포함하는 엔드투엔드(end-to-end) ML 파이프라인을 설계할 수 있다
5. DevOps와 MLOps의 핵심 차이를 설명하고 데이터 및 모델 관리가 도입하는 고유한 과제를 서술할 수 있다

---

## 1. MLOps란?

MLOps(Machine Learning Operations)는 머신러닝 모델의 개발, 배포, 운영을 자동화하고 효율적으로 관리하기 위한 실무 방법론입니다.

```
기존 ML 프로젝트의 문제점:
- 모델이 노트북에서만 작동
- 실험 재현 불가능
- 배포 후 성능 저하 감지 어려움
- 수동 재학습 필요

MLOps의 목표:
- 자동화된 ML 파이프라인
- 재현 가능한 실험
- 지속적인 모델 모니터링
- 자동 재학습 및 배포
```

### 1.1 MLOps의 핵심 원칙

```python
"""
MLOps의 핵심 원칙
"""

# 1. 재현성 (Reproducibility)
# - 동일한 데이터와 코드로 동일한 결과를 얻을 수 있어야 함
experiment_config = {
    "data_version": "v1.2.0",
    "code_version": "git:abc123",
    "random_seed": 42,
    "hyperparameters": {"lr": 0.001, "epochs": 100}
}

# 2. 자동화 (Automation)
# - 수동 작업 최소화, 파이프라인으로 자동화
pipeline_stages = [
    "data_validation",
    "feature_engineering",
    "model_training",
    "model_evaluation",
    "model_deployment"
]

# 3. 모니터링 (Monitoring)
# - 모델 성능과 데이터 품질 지속 감시
monitoring_metrics = {
    "model_accuracy": 0.95,
    "prediction_latency_p99": "50ms",
    "data_drift_score": 0.02
}

# 4. 버전 관리 (Versioning)
# - 데이터, 코드, 모델 모두 버전 관리
versions = {
    "data": "s3://bucket/data/v2.0/",
    "model": "models/classifier_v3.2.1",
    "config": "configs/production.yaml"
}
```

---

## 2. DevOps vs MLOps

### 2.1 전통적 DevOps

```
개발자 → 코드 작성 → 빌드 → 테스트 → 배포 → 모니터링
                   ↑                        │
                   └────────── 피드백 ───────┘
```

### 2.2 MLOps의 추가 요소

```
데이터 과학자 → 데이터 → 피처 엔지니어링 → 모델 학습 → 검증 → 배포 → 모니터링
                 ↑                                              │
                 │            ← 데이터 드리프트 감지 ←───────────┘
                 │            ← 모델 성능 저하 감지 ←───────────┘
                 └─────────────── 재학습 트리거 ────────────────┘
```

### 2.3 주요 차이점

```python
"""
DevOps vs MLOps 차이점
"""

# DevOps: 코드 중심
devops_artifacts = {
    "source": "application_code",
    "build": "docker_image",
    "test": "unit_tests, integration_tests",
    "deploy": "kubernetes_manifests"
}

# MLOps: 데이터 + 코드 + 모델
mlops_artifacts = {
    "data": "training_data, validation_data",
    "features": "feature_definitions, feature_store",
    "code": "training_scripts, serving_code",
    "model": "model_weights, model_metadata",
    "experiments": "hyperparameters, metrics, artifacts"
}

# MLOps 고유의 과제
mlops_challenges = [
    "데이터 품질 관리",
    "피처 일관성 유지",
    "모델 버전 관리",
    "A/B 테스트",
    "드리프트 감지",
    "모델 해석 가능성"
]
```

| 구분 | DevOps | MLOps |
|------|--------|-------|
| 주요 산출물 | 애플리케이션 코드 | 데이터 + 모델 + 코드 |
| 테스트 | 유닛/통합 테스트 | + 데이터 검증, 모델 검증 |
| 배포 단위 | 컨테이너/서비스 | 모델 + 서빙 인프라 |
| 모니터링 | 시스템 메트릭 | + 모델 성능, 데이터 드리프트 |
| 롤백 | 이전 버전 배포 | 모델 버전 롤백 + 데이터 고려 |

---

## 3. MLOps 성숙도 레벨

### 3.1 Google의 MLOps 성숙도 모델

```
Level 0: 수동 프로세스
├── 주피터 노트북에서 실험
├── 수동으로 모델 배포
├── 모니터링 없음
└── 재학습 트리거 없음

Level 1: ML 파이프라인 자동화
├── 자동화된 학습 파이프라인
├── 실험 추적
├── 모델 레지스트리
└── 기본 모니터링

Level 2: CI/CD 파이프라인
├── 자동 테스트 (코드 + 데이터 + 모델)
├── 지속적 학습 (CT)
├── 자동 재배포
└── 완전한 모니터링 및 알림
```

### 3.2 성숙도별 특징

```python
"""
MLOps 성숙도 레벨별 구현
"""

# Level 0: 수동 ML
class Level0_ManualML:
    """
    - 데이터 과학자가 노트북에서 실험
    - 엔지니어가 수동으로 모델 배포
    - 드문 배포 (분기별/연간)
    """
    def train(self):
        # 노트북에서 실행
        model = train_model(data)
        model.save("model.pkl")
        # 수동으로 서버에 복사

# Level 1: ML 파이프라인
class Level1_Pipeline:
    """
    - 자동화된 학습 파이프라인
    - 실험 추적 (MLflow)
    - 모델 레지스트리
    """
    def train_pipeline(self):
        with mlflow.start_run():
            data = load_data()
            model = train_model(data)
            mlflow.log_metrics(evaluate(model))
            mlflow.sklearn.log_model(model, "model")

# Level 2: CI/CD/CT
class Level2_CICDCT:
    """
    - 코드 변경 시 자동 테스트
    - 데이터 변경 시 자동 재학습
    - 성능 저하 시 자동 롤백
    """
    def continuous_training(self):
        if detect_drift() or new_data_available():
            trigger_training_pipeline()
        if model_performance_degraded():
            rollback_to_previous_version()
```

### 3.3 성숙도 체크리스트

```yaml
# mlops_maturity_checklist.yaml
level_0:
  - 노트북 기반 실험
  - 수동 모델 배포
  - 문서화 없음

level_1:
  - 자동화된 학습 파이프라인
  - 실험 추적 시스템 (MLflow/W&B)
  - 모델 버전 관리
  - 기본 모니터링

level_2:
  - CI/CD for ML
  - 자동화된 테스트 (데이터/모델)
  - 지속적 학습 (Continuous Training)
  - 드리프트 감지 및 알림
  - A/B 테스트 인프라
  - 피처 스토어
```

---

## 4. MLOps 도구 생태계

### 4.1 카테고리별 도구

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MLOps 도구 생태계                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [실험 추적]          [파이프라인]         [피처 스토어]              │
│  - MLflow             - Kubeflow          - Feast                   │
│  - Weights & Biases   - Airflow           - Tecton                  │
│  - Neptune            - Prefect           - Hopsworks               │
│  - Comet ML           - Dagster                                     │
│                                                                     │
│  [모델 레지스트리]     [서빙]              [모니터링]                 │
│  - MLflow Registry    - TorchServe        - Evidently               │
│  - Vertex AI          - Triton            - Grafana                 │
│  - SageMaker          - TFServing         - Prometheus              │
│  - Neptune            - BentoML           - WhyLabs                 │
│                       - Seldon                                      │
│                                                                     │
│  [데이터 버전]         [라벨링]            [인프라]                   │
│  - DVC                - Label Studio      - Kubernetes              │
│  - Delta Lake         - Labelbox          - Docker                  │
│  - LakeFS             - Prodigy           - Terraform               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 주요 도구 비교

```python
"""
MLOps 도구 비교
"""

# 실험 추적 도구
experiment_tracking = {
    "MLflow": {
        "type": "오픈소스",
        "features": ["실험 추적", "모델 레지스트리", "서빙"],
        "deployment": "self-hosted / managed",
        "best_for": "엔터프라이즈, 커스터마이징"
    },
    "Weights & Biases": {
        "type": "상용 (무료 티어 있음)",
        "features": ["실험 추적", "하이퍼파라미터 튜닝", "아티팩트"],
        "deployment": "SaaS / self-hosted",
        "best_for": "딥러닝, 협업, 시각화"
    },
    "Neptune": {
        "type": "상용 (무료 티어 있음)",
        "features": ["실험 추적", "메타데이터 관리"],
        "deployment": "SaaS",
        "best_for": "대규모 실험 관리"
    }
}

# 서빙 도구
serving_tools = {
    "TorchServe": {
        "supported": ["PyTorch"],
        "features": ["REST/gRPC", "배치 추론", "A/B 테스트"],
        "best_for": "PyTorch 모델"
    },
    "Triton": {
        "supported": ["PyTorch", "TensorFlow", "ONNX", "TensorRT"],
        "features": ["멀티모델", "GPU 최적화", "동적 배칭"],
        "best_for": "고성능 추론, 멀티 프레임워크"
    },
    "TFServing": {
        "supported": ["TensorFlow"],
        "features": ["REST/gRPC", "버전 관리"],
        "best_for": "TensorFlow 모델"
    }
}
```

### 4.3 클라우드 관리형 서비스

```python
"""
클라우드 MLOps 플랫폼
"""

cloud_platforms = {
    "AWS SageMaker": {
        "components": [
            "SageMaker Studio",      # 개발 환경
            "SageMaker Pipelines",   # 파이프라인
            "Model Registry",         # 모델 관리
            "SageMaker Endpoints",   # 서빙
            "Model Monitor"          # 모니터링
        ]
    },
    "Google Vertex AI": {
        "components": [
            "Workbench",             # 개발 환경
            "Vertex Pipelines",      # 파이프라인
            "Model Registry",         # 모델 관리
            "Prediction",            # 서빙
            "Model Monitoring"       # 모니터링
        ]
    },
    "Azure ML": {
        "components": [
            "Azure ML Studio",       # 개발 환경
            "Azure Pipelines",       # 파이프라인
            "Model Registry",         # 모델 관리
            "Managed Endpoints",     # 서빙
            "Data Drift"             # 모니터링
        ]
    }
}
```

---

## 5. MLOps 아키텍처 패턴

### 5.1 기본 아키텍처

```
┌──────────────────────────────────────────────────────────────────┐
│                        MLOps 기본 아키텍처                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────┐     ┌─────────────┐     ┌─────────────┐           │
│   │  Data   │────▶│   Feature   │────▶│   Model     │           │
│   │  Lake   │     │   Store     │     │  Training   │           │
│   └─────────┘     └─────────────┘     └──────┬──────┘           │
│                                               │                  │
│                                               ▼                  │
│   ┌─────────┐     ┌─────────────┐     ┌──────────────┐          │
│   │ Monitor │◀────│   Model     │◀────│    Model     │          │
│   │  /Alert │     │   Serving   │     │   Registry   │          │
│   └────┬────┘     └─────────────┘     └──────────────┘          │
│        │                                                         │
│        └────────── Retrain Trigger ─────────────────▶ Training  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 코드 예시

```python
"""
MLOps 파이프라인 기본 구조
"""

from typing import Dict, Any

class MLOpsPipeline:
    """기본 MLOps 파이프라인"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiment_tracker = None
        self.model_registry = None
        self.feature_store = None

    def data_ingestion(self):
        """데이터 수집 및 검증"""
        raw_data = self.load_data(self.config["data_source"])
        validated_data = self.validate_data(raw_data)
        return validated_data

    def feature_engineering(self, data):
        """피처 엔지니어링"""
        features = self.feature_store.get_features(
            entity_ids=data["entity_ids"],
            feature_list=self.config["features"]
        )
        return features

    def train(self, features):
        """모델 학습"""
        with self.experiment_tracker.start_run():
            model = self.train_model(features)
            metrics = self.evaluate(model)
            self.experiment_tracker.log_metrics(metrics)
            return model

    def register(self, model):
        """모델 등록"""
        if self.passes_quality_gate(model):
            self.model_registry.register(
                model=model,
                stage="staging"
            )

    def deploy(self, model_version: str):
        """모델 배포"""
        model = self.model_registry.load(model_version)
        self.serving_endpoint.update(model)

    def monitor(self):
        """모델 모니터링"""
        if self.detect_drift():
            self.trigger_retraining()
```

---

## 6. 시작하기

### 6.1 첫 MLOps 프로젝트

```python
"""
MLOps 시작하기: MLflow로 실험 추적
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# MLflow 설정
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris-classification")

# 데이터 준비
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# MLflow 실험 시작
with mlflow.start_run(run_name="random-forest-v1"):
    # 하이퍼파라미터 로깅
    params = {"n_estimators": 100, "max_depth": 5}
    mlflow.log_params(params)

    # 모델 학습
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    # 평가 및 메트릭 로깅
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # 모델 저장
    mlflow.sklearn.log_model(model, "model")

    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")
```

### 6.2 로컬 환경 설정

```bash
# MLflow 서버 시작
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000

# 브라우저에서 확인
# http://localhost:5000
```

---

## 연습 문제

### 문제 1: MLOps 성숙도 평가
현재 팀의 ML 프로세스를 평가하고 성숙도 레벨을 결정하세요.

### 문제 2: 도구 선택
다음 상황에 적합한 MLOps 도구를 선택하세요:
- 소규모 팀, PyTorch 사용, 예산 제한
- 대규모 팀, 멀티 프레임워크, 고성능 필요

---

## 요약

| 개념 | 설명 |
|------|------|
| MLOps | ML 모델의 개발, 배포, 운영 자동화 |
| DevOps vs MLOps | MLOps는 데이터와 모델 관리가 추가됨 |
| 성숙도 Level 0 | 수동 프로세스, 노트북 기반 |
| 성숙도 Level 1 | 자동화된 파이프라인, 실험 추적 |
| 성숙도 Level 2 | CI/CD/CT, 지속적 학습 |
| 핵심 도구 | MLflow, W&B, Kubeflow, Feast, Triton |

---

## 참고 자료

- [Google MLOps Whitepaper](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [MLOps Principles](https://ml-ops.org/)
- [Made With ML - MLOps](https://madewithml.com/courses/mlops/)
