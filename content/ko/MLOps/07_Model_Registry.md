# 모델 레지스트리

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. ML 라이프사이클에서 모델 레지스트리(Model Registry)의 역할을 정의하고 데이터 사이언티스트와 ML 엔지니어 간 협업을 어떻게 가능하게 하는지 설명할 수 있다
2. 시맨틱 버전닝(semantic versioning) 관례를 사용하여 모델 버전을 관리하고 MLflow Model Registry API로 모델을 등록할 수 있다
3. 모델 스테이지 전환(Staging → Production → Archived)을 관리하고 프로덕션 승격을 위한 승인 워크플로우(approval workflow)를 구현할 수 있다
4. 클라우드 관리형 모델 레지스트리(AWS SageMaker, Google Vertex AI, Azure ML)와 자체 호스팅 MLflow를 비교하고 트레이드오프를 평가할 수 있다
5. 접근 제어(access control), 감사 로그(audit logging), 롤백(rollback) 절차를 포함하는 모델 레지스트리 거버넌스 프레임워크를 설계할 수 있다

---

## 1. 모델 레지스트리 개념

모델 레지스트리는 ML 모델의 중앙 저장소로, 버전 관리, 메타데이터 추적, 배포 관리를 담당합니다.

### 1.1 모델 레지스트리의 역할

```
┌─────────────────────────────────────────────────────────────────────┐
│                     모델 레지스트리 기능                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────┐          │
│   │                  Model Registry                      │          │
│   │                                                      │          │
│   │   ┌───────────────┐  ┌───────────────┐              │          │
│   │   │ Version       │  │ Metadata      │              │          │
│   │   │ Management    │  │ Tracking      │              │          │
│   │   │               │  │               │              │          │
│   │   │ - v1, v2, v3  │  │ - 메트릭      │              │          │
│   │   │ - 변경 이력   │  │ - 파라미터    │              │          │
│   │   │ - 롤백       │  │ - 의존성      │              │          │
│   │   └───────────────┘  └───────────────┘              │          │
│   │                                                      │          │
│   │   ┌───────────────┐  ┌───────────────┐              │          │
│   │   │ Stage         │  │ Access        │              │          │
│   │   │ Management    │  │ Control       │              │          │
│   │   │               │  │               │              │          │
│   │   │ - Development │  │ - 권한 관리   │              │          │
│   │   │ - Staging     │  │ - 승인 워크플로│              │          │
│   │   │ - Production  │  │ - 감사 로그   │              │          │
│   │   └───────────────┘  └───────────────┘              │          │
│   │                                                      │          │
│   └─────────────────────────────────────────────────────┘          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 레지스트리 선택

```python
"""
모델 레지스트리 옵션
"""

registry_options = {
    "MLflow Model Registry": {
        "type": "오픈소스",
        "features": ["버전 관리", "스테이지", "태그", "설명"],
        "integration": "MLflow 생태계",
        "hosting": "Self-hosted"
    },
    "AWS SageMaker Model Registry": {
        "type": "관리형",
        "features": ["버전 관리", "승인 워크플로우", "메트릭 추적"],
        "integration": "AWS 생태계",
        "hosting": "AWS"
    },
    "Google Vertex AI Model Registry": {
        "type": "관리형",
        "features": ["버전 관리", "자동 배포", "모니터링"],
        "integration": "GCP 생태계",
        "hosting": "GCP"
    },
    "Azure ML Model Registry": {
        "type": "관리형",
        "features": ["버전 관리", "배포 관리", "설명 가능성"],
        "integration": "Azure 생태계",
        "hosting": "Azure"
    }
}
```

---

## 2. 버전 관리

### 2.1 시맨틱 버전 관리

```python
"""
모델 시맨틱 버전 관리 전략
"""

# 버전 형식: MAJOR.MINOR.PATCH
version_strategy = {
    "MAJOR": {
        "trigger": "호환되지 않는 변경",
        "examples": [
            "입력/출력 스키마 변경",
            "모델 아키텍처 변경",
            "피처 집합 변경"
        ]
    },
    "MINOR": {
        "trigger": "기능 추가 (호환 유지)",
        "examples": [
            "새로운 피처 추가 (선택적)",
            "하이퍼파라미터 최적화",
            "추가 출력 필드"
        ]
    },
    "PATCH": {
        "trigger": "버그 수정, 재학습",
        "examples": [
            "동일 데이터/설정으로 재학습",
            "버그 수정",
            "데이터 업데이트 (스키마 동일)"
        ]
    }
}

# 버전 예시
versions = [
    ("1.0.0", "초기 프로덕션 릴리스"),
    ("1.0.1", "새 데이터로 재학습"),
    ("1.1.0", "하이퍼파라미터 튜닝"),
    ("1.2.0", "새 피처 추가"),
    ("2.0.0", "모델 아키텍처 변경 (RF -> XGBoost)")
]
```

### 2.2 MLflow 버전 관리

```python
"""
MLflow Model Registry 버전 관리
"""

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 1. 모델 등록 (첫 버전)
model_name = "ChurnPredictionModel"

# 방법 1: log_model 시 직접 등록
with mlflow.start_run():
    # 학습...
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name=model_name
    )

# 방법 2: 기존 run에서 등록
result = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name=model_name
)
print(f"Version: {result.version}")

# 2. 버전 설명 추가
client.update_model_version(
    name=model_name,
    version=result.version,
    description="""
    ## Changes
    - Improved feature engineering
    - Added customer segment features

    ## Performance
    - Accuracy: 0.92 (+2% from v1)
    - F1 Score: 0.89
    """
)

# 3. 태그 추가
client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key="validated",
    value="true"
)

client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key="data_version",
    value="2024-01-15"
)

# 4. 버전 목록 조회
versions = client.search_model_versions(f"name='{model_name}'")
for v in versions:
    print(f"v{v.version}: {v.current_stage} - {v.description[:50]}...")
```

### 2.3 버전 비교

```python
"""
모델 버전 비교
"""

from mlflow.tracking import MlflowClient
import pandas as pd

client = MlflowClient()
model_name = "ChurnPredictionModel"

def compare_versions(model_name: str, versions: list):
    """여러 버전의 메트릭 비교"""
    comparison = []

    for version in versions:
        # 버전 정보 가져오기
        model_version = client.get_model_version(model_name, version)

        # 연결된 run에서 메트릭 가져오기
        run = client.get_run(model_version.run_id)

        comparison.append({
            "version": version,
            "stage": model_version.current_stage,
            "accuracy": run.data.metrics.get("accuracy"),
            "f1_score": run.data.metrics.get("f1_score"),
            "created_at": model_version.creation_timestamp
        })

    return pd.DataFrame(comparison)

# 버전 비교
df = compare_versions(model_name, ["1", "2", "3"])
print(df)

# 가장 좋은 버전 찾기
best_version = df.loc[df["accuracy"].idxmax(), "version"]
print(f"Best version: {best_version}")
```

---

## 3. 스테이지 전환

### 3.1 스테이지 개념

```
┌─────────────────────────────────────────────────────────────────────┐
│                      모델 스테이지 워크플로우                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐      ┌──────────┐      ┌──────────┐                 │
│   │  None    │ ───▶ │ Staging  │ ───▶ │Production│                 │
│   │(개발중)  │      │(테스트)   │      │(운영중)  │                 │
│   └──────────┘      └────┬─────┘      └────┬─────┘                 │
│                          │                  │                       │
│                          │                  │                       │
│                          ▼                  ▼                       │
│                    ┌──────────────────────────┐                     │
│                    │       Archived           │                     │
│                    │     (아카이브)           │                     │
│                    └──────────────────────────┘                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 스테이지 전환 코드

```python
"""
MLflow 스테이지 전환
"""

from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "ChurnPredictionModel"

# 1. Staging으로 전환
client.transition_model_version_stage(
    name=model_name,
    version="2",
    stage="Staging",
    archive_existing_versions=False
)

# 2. Production으로 승격 (기존 Production 자동 아카이브)
client.transition_model_version_stage(
    name=model_name,
    version="2",
    stage="Production",
    archive_existing_versions=True
)

# 3. 모델 로드 (스테이지별)
import mlflow

staging_model = mlflow.pyfunc.load_model(
    f"models:/{model_name}/Staging"
)

production_model = mlflow.pyfunc.load_model(
    f"models:/{model_name}/Production"
)

# 4. 현재 Production 버전 확인
prod_versions = client.get_latest_versions(
    model_name,
    stages=["Production"]
)
if prod_versions:
    print(f"Current Production: v{prod_versions[0].version}")
```

### 3.3 승인 워크플로우

```python
"""
모델 승격 승인 워크플로우
"""

from mlflow.tracking import MlflowClient
from typing import Dict, Any

class ModelApprovalWorkflow:
    """모델 승인 워크플로우"""

    def __init__(self, client: MlflowClient):
        self.client = client

    def validate_for_staging(
        self,
        model_name: str,
        version: str,
        quality_gates: Dict[str, float]
    ) -> Dict[str, Any]:
        """Staging 승격 검증"""
        # 품질 게이트를 코드로 검증 — 수동 확인에 의존하면 사람의 실수로
        # 품질 미달 모델이 배포될 수 있음
        model_version = self.client.get_model_version(model_name, version)
        run = self.client.get_run(model_version.run_id)
        metrics = run.data.metrics

        results = {
            "passed": True,
            "checks": {}
        }

        for metric_name, threshold in quality_gates.items():
            actual = metrics.get(metric_name, 0)
            passed = actual >= threshold
            results["checks"][metric_name] = {
                "threshold": threshold,
                "actual": actual,
                "passed": passed
            }
            if not passed:
                results["passed"] = False

        return results

    def validate_for_production(
        self,
        model_name: str,
        version: str
    ) -> Dict[str, Any]:
        """Production 승격 검증"""
        # Staging을 거치지 않은 직접 Production 배포를 방지 —
        # 검증 태그와 설명이 있는지 확인하여 거버넌스(governance)를 강제
        model_version = self.client.get_model_version(model_name, version)

        checks = {
            "is_in_staging": model_version.current_stage == "Staging",
            "has_validation_tag": "validated" in model_version.tags,
            "has_description": bool(model_version.description)
        }

        return {
            "passed": all(checks.values()),
            "checks": checks
        }

    def promote_to_staging(self, model_name: str, version: str) -> bool:
        """Staging으로 승격"""
        validation = self.validate_for_staging(
            model_name, version,
            {"accuracy": 0.85, "f1_score": 0.80}
        )

        if validation["passed"]:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Staging"
            )
            return True
        return False

    def promote_to_production(self, model_name: str, version: str) -> bool:
        """Production으로 승격"""
        validation = self.validate_for_production(model_name, version)

        if validation["passed"]:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True
            )
            return True
        return False

# 사용 예시
client = MlflowClient()
workflow = ModelApprovalWorkflow(client)

# Staging 검증 및 승격
staging_result = workflow.promote_to_staging("ChurnModel", "3")
print(f"Staging promotion: {'Success' if staging_result else 'Failed'}")

# Production 검증 및 승격
if staging_result:
    prod_result = workflow.promote_to_production("ChurnModel", "3")
    print(f"Production promotion: {'Success' if prod_result else 'Failed'}")
```

---

## 4. CI/CD 통합

### 4.1 GitHub Actions 통합

```yaml
# .github/workflows/model-promotion.yaml
name: Model Promotion Pipeline

on:
  workflow_dispatch:
    inputs:
      model_name:
        description: 'Model name'
        required: true
      version:
        description: 'Model version to promote'
        required: true
      target_stage:
        description: 'Target stage (Staging/Production)'
        required: true
        default: 'Staging'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install mlflow

      - name: Validate model
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          python scripts/validate_model.py \
            --model-name ${{ github.event.inputs.model_name }} \
            --version ${{ github.event.inputs.version }} \
            --stage ${{ github.event.inputs.target_stage }}

  promote:
    needs: validate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Promote model
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          python scripts/promote_model.py \
            --model-name ${{ github.event.inputs.model_name }} \
            --version ${{ github.event.inputs.version }} \
            --stage ${{ github.event.inputs.target_stage }}

  deploy:
    needs: promote
    if: ${{ github.event.inputs.target_stage == 'Production' }}
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          # 배포 스크립트
          echo "Deploying model to production..."
```

### 4.2 검증 스크립트

```python
"""
scripts/validate_model.py
"""

import argparse
import mlflow
from mlflow.tracking import MlflowClient
import sys

def validate_model(model_name: str, version: str, stage: str) -> bool:
    """모델 검증"""
    client = MlflowClient()

    # 버전 정보 가져오기
    model_version = client.get_model_version(model_name, version)
    run = client.get_run(model_version.run_id)
    metrics = run.data.metrics

    # 검증 기준
    if stage == "Staging":
        requirements = {
            "accuracy": 0.85,
            "f1_score": 0.80
        }
    else:  # Production
        requirements = {
            "accuracy": 0.90,
            "f1_score": 0.85
        }
        # Production은 Staging 거쳐야 함
        if model_version.current_stage != "Staging":
            print("Error: Model must be in Staging before Production")
            return False

    # 메트릭 검증
    for metric, threshold in requirements.items():
        actual = metrics.get(metric, 0)
        if actual < threshold:
            print(f"Failed: {metric} = {actual} < {threshold}")
            return False
        print(f"Passed: {metric} = {actual} >= {threshold}")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()

    if not validate_model(args.model_name, args.version, args.stage):
        sys.exit(1)
```

### 4.3 배포 자동화

```python
"""
scripts/promote_model.py
"""

import argparse
import mlflow
from mlflow.tracking import MlflowClient

def promote_model(model_name: str, version: str, stage: str):
    """모델 승격"""
    client = MlflowClient()

    # 스테이지 전환
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=(stage == "Production")
    )

    print(f"Model {model_name} v{version} promoted to {stage}")

    # 태그 업데이트
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="promoted_at",
        value=str(datetime.now())
    )

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="promoted_by",
        value="CI/CD Pipeline"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()

    promote_model(args.model_name, args.version, args.stage)
```

---

## 5. 모델 메타데이터 관리

### 5.1 메타데이터 스키마

```python
"""
모델 메타데이터 구조
"""

model_metadata = {
    # 기본 정보
    "name": "ChurnPredictionModel",
    "version": "2.1.0",
    "created_at": "2024-01-15T10:30:00Z",
    "created_by": "ml-team",

    # 성능 메트릭
    "metrics": {
        "accuracy": 0.92,
        "precision": 0.89,
        "recall": 0.91,
        "f1_score": 0.90,
        "auc_roc": 0.95
    },

    # 학습 정보
    "training": {
        "framework": "sklearn",
        "algorithm": "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 10
        },
        "training_data": "s3://bucket/data/train_v2.parquet",
        "data_version": "2024-01-10"
    },

    # 피처 정보
    "features": {
        "input_schema": {
            "age": "float64",
            "tenure": "int64",
            "monthly_charges": "float64",
            "total_charges": "float64"
        },
        "output_schema": {
            "prediction": "int64",
            "probability": "float64"
        }
    },

    # 의존성
    "dependencies": {
        "python": "3.9",
        "sklearn": "1.2.0",
        "pandas": "1.5.0"
    },

    # 배포 정보
    "deployment": {
        "serving_framework": "mlflow",
        "endpoint": "https://api.example.com/predict",
        "latency_p99": "45ms",
        "throughput": "1000 req/s"
    }
}
```

### 5.2 메타데이터 저장

```python
"""
MLflow에 메타데이터 저장
"""

import mlflow
from mlflow.tracking import MlflowClient
import json

client = MlflowClient()

def save_model_metadata(model_name: str, version: str, metadata: dict):
    """모델 메타데이터 저장"""

    # 태그로 저장 (간단한 key-value)
    for key, value in metadata.get("metrics", {}).items():
        client.set_model_version_tag(
            model_name, version,
            f"metric_{key}", str(value)
        )

    # 복잡한 데이터는 JSON으로
    client.set_model_version_tag(
        model_name, version,
        "features_schema",
        json.dumps(metadata.get("features", {}))
    )

    client.set_model_version_tag(
        model_name, version,
        "dependencies",
        json.dumps(metadata.get("dependencies", {}))
    )

    # 설명에 포함
    description = f"""
    ## Model Information
    - Framework: {metadata['training']['framework']}
    - Algorithm: {metadata['training']['algorithm']}

    ## Performance
    - Accuracy: {metadata['metrics']['accuracy']}
    - F1 Score: {metadata['metrics']['f1_score']}

    ## Data
    - Training Data: {metadata['training']['training_data']}
    - Data Version: {metadata['training']['data_version']}
    """

    client.update_model_version(model_name, version, description=description)

# 사용
save_model_metadata("ChurnModel", "3", model_metadata)
```

---

## 6. 롤백 전략

```python
"""
모델 롤백 전략
"""

from mlflow.tracking import MlflowClient
from datetime import datetime

class ModelRollback:
    """모델 롤백 관리"""

    def __init__(self, client: MlflowClient):
        self.client = client

    def rollback_to_previous(self, model_name: str) -> str:
        """이전 Production 버전으로 롤백"""
        # 아카이브된 버전 중 가장 최근 찾기 — 현재 피처 스키마와
        # 데이터 분포에 호환될 가능성이 가장 높은 버전
        versions = self.client.search_model_versions(
            f"name='{model_name}'"
        )

        archived = [
            v for v in versions
            if v.current_stage == "Archived"
        ]

        if not archived:
            raise ValueError("No archived versions available")

        # 가장 최근 아카이브 버전
        latest_archived = max(archived, key=lambda x: int(x.version))

        # 삭제가 아닌 아카이브 — 사후 분석(post-mortem)을 위해 아티팩트를 보존
        current_prod = self.get_production_version(model_name)
        if current_prod:
            self.client.transition_model_version_stage(
                model_name, current_prod.version, "Archived"
            )

            # 롤백 사유를 태그로 기록 — 왜 이 버전이 프로덕션에서 제거되었는지 감사 추적(audit trail) 생성
            self.client.set_model_version_tag(
                model_name, current_prod.version,
                "rollback_reason", "Performance degradation"
            )

        # 이전 버전 Production으로
        self.client.transition_model_version_stage(
            model_name, latest_archived.version, "Production"
        )

        self.client.set_model_version_tag(
            model_name, latest_archived.version,
            "rolled_back_at", str(datetime.now())
        )

        return latest_archived.version

    def rollback_to_version(self, model_name: str, version: str) -> None:
        """특정 버전으로 롤백"""
        # 현재 Production 아카이브
        current_prod = self.get_production_version(model_name)
        if current_prod:
            self.client.transition_model_version_stage(
                model_name, current_prod.version, "Archived"
            )

        # 지정된 버전을 Production으로
        self.client.transition_model_version_stage(
            model_name, version, "Production"
        )

    def get_production_version(self, model_name: str):
        """현재 Production 버전 조회"""
        versions = self.client.get_latest_versions(
            model_name, stages=["Production"]
        )
        return versions[0] if versions else None

# 사용 예시
client = MlflowClient()
rollback = ModelRollback(client)

# 이전 버전으로 롤백
previous_version = rollback.rollback_to_previous("ChurnModel")
print(f"Rolled back to v{previous_version}")

# 특정 버전으로 롤백
rollback.rollback_to_version("ChurnModel", "2")
```

---

## 연습 문제

### 문제 1: 버전 관리
모델을 3개 버전 등록하고, 메트릭 기반으로 가장 좋은 버전을 자동 선택하세요.

### 문제 2: 승인 워크플로우
Staging -> Production 승격을 위한 검증 스크립트를 작성하세요.

### 문제 3: 롤백
성능 저하 감지 시 자동으로 이전 버전으로 롤백하는 로직을 구현하세요.

---

## 요약

| 개념 | 설명 |
|------|------|
| 모델 레지스트리 | 모델의 중앙 저장소 |
| 버전 관리 | 모델 변경 이력 추적 |
| 스테이지 | None -> Staging -> Production |
| 메타데이터 | 모델 관련 정보 (메트릭, 의존성 등) |
| 롤백 | 이전 버전으로 복원 |

---

## 참고 자료

- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [AWS SageMaker Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)
- [Google Vertex AI Model Registry](https://cloud.google.com/vertex-ai/docs/model-registry/)
