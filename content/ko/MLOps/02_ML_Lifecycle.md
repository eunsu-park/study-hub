# ML 프로젝트 라이프사이클

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. ML 프로젝트 라이프사이클의 각 단계인 문제 정의, 데이터 수집, 피처 엔지니어링(feature engineering), 모델 학습, 검증, 배포, 모니터링을 설명하고 각 단계가 어떻게 연결되는지 서술할 수 있다
2. 타겟 변수, 성공 지표(success metric), 운영 제약사항을 정의하여 비즈니스 목표를 잘 범위가 설정된 ML 문제로 전환할 수 있다
3. ML 파이프라인 각 단계에서 데이터 무결성을 보장하기 위한 데이터 검증 및 품질 검사를 구현할 수 있다
4. 재현 가능하고 신뢰할 수 있는 ML 시스템을 구축하기 위해 피처 엔지니어링과 모델 검증 모범 사례를 적용할 수 있다
5. 데이터 드리프트(data drift)와 성능 저하 같은 모니터링 신호를 기반으로 재학습(retraining) 전략을 설계할 수 있다

---

## 1. ML 프로젝트 단계 개요

머신러닝 프로젝트는 단순히 모델을 학습시키는 것을 넘어, 데이터 수집부터 모니터링까지 전체 생명주기를 관리해야 합니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ML 프로젝트 라이프사이클                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    │
│   │ 문제정의  │───▶│ 데이터   │───▶│  피처    │───▶│  모델    │    │
│   │          │    │ 수집/준비 │    │ 엔지니어링│    │  학습    │    │
│   └──────────┘    └──────────┘    └──────────┘    └────┬─────┘    │
│                                                         │          │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐         │          │
│   │ 모니터링  │◀───│  배포    │◀───│  검증    │◀────────┘          │
│   │          │    │          │    │          │                     │
│   └────┬─────┘    └──────────┘    └──────────┘                     │
│        │                                                            │
│        └──────────────── 재학습 ─────────────────────────────▶     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 문제 정의 및 범위 설정

### 2.1 비즈니스 목표 정의

```python
"""
ML 프로젝트 문제 정의 템플릿
"""

project_definition = {
    # 비즈니스 목표
    "business_objective": "고객 이탈률 30% 감소",

    # ML 문제 정의
    "ml_problem": {
        "type": "binary_classification",
        "target": "is_churned",
        "success_metric": "precision_at_recall_80",
        "baseline": 0.65
    },

    # 제약사항
    "constraints": {
        "latency": "< 100ms",
        "throughput": "1000 req/s",
        "model_size": "< 500MB",
        "interpretability": "high"  # 규제 요구사항
    },

    # 데이터 요구사항
    "data_requirements": {
        "historical_period": "2 years",
        "minimum_samples": 100000,
        "features": ["usage_patterns", "demographics", "support_tickets"]
    }
}
```

### 2.2 성공 기준 정의

```python
"""
모델 성능 기준 정의
"""

success_criteria = {
    # 오프라인 메트릭 (모델 품질)
    "offline_metrics": {
        "accuracy": {"min": 0.85, "target": 0.90},
        "precision": {"min": 0.80, "target": 0.85},
        "recall": {"min": 0.75, "target": 0.80},
        "auc_roc": {"min": 0.85, "target": 0.90}
    },

    # 온라인 메트릭 (비즈니스 영향)
    "online_metrics": {
        "churn_rate_reduction": {"target": "30%"},
        "false_positive_cost": {"max": "$10K/month"}
    },

    # 시스템 메트릭
    "system_metrics": {
        "p99_latency": {"max": "100ms"},
        "availability": {"min": "99.9%"},
        "throughput": {"min": "1000 req/s"}
    }
}
```

---

## 3. 데이터 수집 및 준비

### 3.1 데이터 파이프라인

```python
"""
데이터 수집 파이프라인 예시
"""

from typing import Dict, Any
import pandas as pd
from datetime import datetime, timedelta

class DataPipeline:
    """데이터 수집 및 준비 파이프라인"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_sources = config["data_sources"]

    def extract(self) -> Dict[str, pd.DataFrame]:
        """다양한 소스에서 데이터 추출"""
        # 소스별 분리 추출 — 독립적 재시도와 스키마 검증이 가능
        data = {}

        # 데이터베이스에서 추출
        data["transactions"] = self.query_database(
            query="SELECT * FROM transactions WHERE date > ?",
            params=[self.config["start_date"]]
        )

        # S3에서 추출
        data["user_events"] = self.read_from_s3(
            bucket="data-lake",
            prefix=f"events/{self.config['date_partition']}/"
        )

        # API에서 추출
        data["external_features"] = self.fetch_from_api(
            endpoint=self.config["external_api"]
        )

        return data

    def transform(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """데이터 변환 및 전처리"""

        # left join으로 이벤트 트래킹이 불완전해도 모든 거래 데이터를 보존
        df = raw_data["transactions"].merge(
            raw_data["user_events"],
            on="user_id",
            how="left"
        )

        # 순서 중요: null이 이상치 탐지를 왜곡할 수 있으므로 결측치를 먼저 처리
        df = self.handle_missing(df)

        # 이상치 처리
        df = self.handle_outliers(df)

        # 데이터 타입 변환
        df = self.convert_types(df)

        return df

    def validate(self, df: pd.DataFrame) -> bool:
        """데이터 품질 검증"""
        # 검증을 transform과 별도 단계로 분리 — 비용이 큰 후속 처리 전에
        # 조기 실패(fail-fast)를 가능하게 함
        validations = {
            "row_count": len(df) > self.config["min_rows"],
            "null_ratio": df.isnull().mean().max() < 0.1,
            "schema_match": self.check_schema(df),
            "value_ranges": self.check_value_ranges(df)
        }

        return all(validations.values())

    def load(self, df: pd.DataFrame, destination: str):
        """처리된 데이터 저장"""
        # 데이터 자체에 버전 메타데이터 포함 — 파일명 규칙에만 의존하지 않고
        # 계보 추적(lineage tracing)을 가능하게 함
        df["_data_version"] = self.config["version"]
        df["_processed_at"] = datetime.now()

        # CSV 대신 Parquet — 컬럼형 포맷으로 3-10배 압축률과 스키마 검증 제공
        df.to_parquet(
            f"{destination}/data_v{self.config['version']}.parquet",
            index=False
        )
```

### 3.2 데이터 버전 관리 (DVC)

```yaml
# dvc.yaml - DVC 파이프라인 정의
# deps/outs 선언으로 변경되지 않은 스테이지를 건너뛰어 대규모 파이프라인에서 수 시간을 절약
stages:
  prepare_data:
    cmd: python src/data/prepare.py
    deps:
      - src/data/prepare.py
      - data/raw/
    outs:
      - data/processed/train.parquet
      - data/processed/test.parquet

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/train.parquet
    # params를 별도 추적 — 코드 변경 없이 하이퍼파라미터만 변경해도 DVC가 감지
    params:
      - train.epochs
      - train.learning_rate
    outs:
      - models/model.pkl
    metrics:
      # cache: false로 메트릭을 git에 보관(DVC 저장소가 아닌) — 커밋 간 쉬운 비교 가능
      - metrics/train_metrics.json:
          cache: false
```

```bash
# DVC 기본 명령어
# 데이터 추적 시작
dvc add data/raw/dataset.csv

# 파이프라인 실행
dvc repro

# 버전 간 차이 확인
dvc diff

# 데이터 가져오기 (원격 저장소)
dvc pull
```

---

## 4. 피처 엔지니어링

### 4.1 피처 정의 및 계산

```python
"""
피처 엔지니어링 파이프라인
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class FeatureEngineer:
    """피처 엔지니어링 클래스"""

    def __init__(self, feature_config: dict):
        self.config = feature_config
        self.encoders = {}
        self.scalers = {}

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """피처 생성"""
        # 원본을 변경하지 않고 새 DataFrame에 빌드 — 피처 계보(lineage) 추적 용이
        features = pd.DataFrame()

        # 시간 기반 피처 — 주기적 행동 패턴 포착(예: 주말 vs 평일 사용량)
        features["hour"] = df["timestamp"].dt.hour
        features["day_of_week"] = df["timestamp"].dt.dayofweek
        features["is_weekend"] = features["day_of_week"].isin([5, 6]).astype(int)

        # 롤링 집계로 최근 행동 트렌드 포착 — 누적 합계보다 예측력이 높음
        features["total_purchases_30d"] = self.rolling_aggregate(
            df, "purchase_amount", window=30, agg="sum"
        )
        features["avg_session_duration_7d"] = self.rolling_aggregate(
            df, "session_duration", window=7, agg="mean"
        )

        # 비율 피처 — 시간으로 정규화하여 가입 기간이 다른 사용자 간 비교를 가능하게 함
        features["purchase_frequency"] = (
            df["purchase_count"] / df["days_since_signup"]
        ).fillna(0)

        # 상호작용 피처 — 피처 간 교차 비율이 비선형 관계를 드러내는 경우가 많음
        features["value_per_session"] = (
            df["total_purchase_value"] / df["session_count"]
        ).fillna(0)

        return features

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """범주형 변수 인코딩"""
        for col in self.config["categorical_features"]:
            # 첫 호출에서만 fit_transform, 추론 시에는 transform만 사용
            # — 테스트/운영 데이터로부터의 데이터 누수(data leakage) 방지
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col])
            else:
                df[col] = self.encoders[col].transform(df[col])
        return df

    def scale_numericals(self, df: pd.DataFrame) -> pd.DataFrame:
        """수치형 변수 스케일링"""
        numerical_cols = self.config["numerical_features"]

        if "standard" not in self.scalers:
            self.scalers["standard"] = StandardScaler()
            df[numerical_cols] = self.scalers["standard"].fit_transform(
                df[numerical_cols]
            )
        else:
            df[numerical_cols] = self.scalers["standard"].transform(
                df[numerical_cols]
            )
        return df

    def save_transformers(self, path: str):
        """인코더/스케일러 저장"""
        # 모델과 함께 변환기(transformer)를 저장 — 추론 시 재학습 없이
        # 일관된 인코딩/스케일링을 보장하기 위해 필수
        import joblib
        joblib.dump({
            "encoders": self.encoders,
            "scalers": self.scalers
        }, path)
```

### 4.2 피처 스토어 연동

```python
"""
Feature Store 사용 예시 (Feast)
"""

from feast import FeatureStore

# Feature Store 초기화
fs = FeatureStore(repo_path="./feature_repo")

# 오프라인 스토어(offline store)로 학습 — point-in-time 조인으로 미래 데이터 누수 방지
training_df = fs.get_historical_features(
    entity_df=entity_df,  # entity_id, event_timestamp
    features=[
        "user_features:total_purchases",
        "user_features:avg_session_duration",
        "product_features:category",
        "product_features:price_range"
    ]
).to_df()

# 온라인 스토어(online store)로 추론 — 사전 구체화된 키-값 조회로 저지연(<10ms) 서빙
feature_vector = fs.get_online_features(
    features=[
        "user_features:total_purchases",
        "user_features:avg_session_duration"
    ],
    entity_rows=[{"user_id": 12345}]
).to_dict()
```

---

## 5. 모델 학습

### 5.1 실험 관리

```python
"""
실험 관리가 포함된 모델 학습
"""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import optuna

class ModelTrainer:
    """모델 학습 클래스"""

    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)

    def train_with_tracking(
        self,
        X_train, y_train,
        X_val, y_val,
        params: dict
    ):
        """MLflow 추적이 포함된 학습"""
        # 컨텍스트 매니저(context manager)로 학습 중 크래시가 발생해도 실행 종료를 보장
        with mlflow.start_run():
            # 파라미터를 먼저 기록 — 학습이 실패해도 어떤 설정이 문제였는지 진단 가능
            mlflow.log_params(params)

            # 데이터셋 크기 기록 — 실행 간 데이터 파이프라인의 암묵적 문제 감지
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size", len(X_val))

            # 모델 학습
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            # 검증
            val_predictions = model.predict(X_val)
            val_proba = model.predict_proba(X_val)[:, 1]

            # 메트릭 계산 및 로깅
            metrics = self.calculate_metrics(y_val, val_predictions, val_proba)
            mlflow.log_metrics(metrics)

            # 시그니처(signature)와 함께 모델 저장 — 서빙 시 입력 스키마를 강제하여
            # 스키마 불일치로 인한 무음 예측 오류(silent prediction error)를 방지
            mlflow.sklearn.log_model(
                model, "model",
                signature=mlflow.models.infer_signature(X_train, val_predictions)
            )

            # 피처 중요도는 디버깅과 이해관계자 신뢰 구축에 활용
            self.log_feature_importance(model, X_train.columns)

            return model, metrics

    def hyperparameter_tuning(self, X, y, n_trials: int = 100):
        """Optuna를 이용한 하이퍼파라미터 튜닝"""
        # GridSearch 대신 Optuna — 베이지안 최적화(Bayesian optimization)로
        # 전수 탐색보다 훨씬 적은 시행 횟수로 좋은 파라미터를 탐색
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10)
            }

            model = RandomForestClassifier(**params, random_state=42)
            scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        # 최적 파라미터 로깅
        with mlflow.start_run(run_name="best_params"):
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_auc", study.best_value)

        return study.best_params
```

### 5.2 학습 파이프라인

```yaml
# training_pipeline.yaml
pipeline:
  name: "churn-prediction-training"
  # 비첨두 시간대에 스케줄링 — 서빙 워크로드와 GPU/CPU 경합 방지
  schedule: "0 2 * * *"  # 매일 오전 2시

  stages:
    # 검증을 피처 엔지니어링과 분리 — 불량 데이터에 대한 피처 계산 전에
    # 조기 실패(fail fast) 가능
    - name: data_validation
      script: src/validate_data.py
      inputs:
        - data/raw/
      outputs:
        - reports/data_validation.html

    - name: feature_engineering
      script: src/feature_engineering.py
      inputs:
        - data/raw/
      outputs:
        - data/features/

    - name: train
      script: src/train.py
      inputs:
        - data/features/
      params:
        - config/train_config.yaml
      outputs:
        - models/

    - name: evaluate
      script: src/evaluate.py
      inputs:
        - models/
        - data/features/test.parquet
      outputs:
        - reports/evaluation.html
```

---

## 6. 모델 검증 및 테스트

### 6.1 모델 품질 게이트

```python
"""
모델 품질 검증
"""

from typing import Dict, Any
import numpy as np

class ModelValidator:
    """모델 검증 클래스"""

    def __init__(self, quality_gates: Dict[str, float]):
        self.quality_gates = quality_gates

    def validate(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """품질 게이트 검증"""
        results = {
            "passed": True,
            "details": {}
        }

        for metric_name, threshold in self.quality_gates.items():
            actual_value = metrics.get(metric_name, 0)
            passed = actual_value >= threshold

            results["details"][metric_name] = {
                "threshold": threshold,
                "actual": actual_value,
                "passed": passed
            }

            if not passed:
                results["passed"] = False

        return results

    def compare_with_baseline(
        self,
        new_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        min_improvement: float = 0.01
    ) -> Dict[str, Any]:
        """베이스라인 모델과 비교"""
        # 항상 프로덕션 베이스라인과 비교 — 절대 임계값은 통과하지만
        # 현재 서빙 중인 모델보다 나쁜 모델 배포를 방지
        results = {"improved": True, "details": {}}

        for metric_name in new_metrics:
            new_val = new_metrics[metric_name]
            baseline_val = baseline_metrics.get(metric_name, 0)
            improvement = (new_val - baseline_val) / baseline_val if baseline_val else 0

            results["details"][metric_name] = {
                "new": new_val,
                "baseline": baseline_val,
                "improvement": f"{improvement:.2%}"
            }

            # 성능 저하 체크
            if new_val < baseline_val * (1 - min_improvement):
                results["improved"] = False

        return results

# 사용 예시
validator = ModelValidator({
    "accuracy": 0.85,
    "precision": 0.80,
    "recall": 0.75,
    "auc_roc": 0.85
})

validation_result = validator.validate(model_metrics)
if not validation_result["passed"]:
    raise ValueError(f"Model failed quality gates: {validation_result}")
```

### 6.2 A/B 테스트 준비

```python
"""
A/B 테스트 설정
"""

ab_test_config = {
    "experiment_name": "churn_model_v2",
    "variants": {
        "control": {
            "model_version": "v1.2.3",
            "traffic_percentage": 50
        },
        "treatment": {
            "model_version": "v2.0.0",
            "traffic_percentage": 50
        }
    },
    "metrics": {
        "primary": "conversion_rate",
        "secondary": ["latency_p99", "error_rate"]
    },
    "duration_days": 14,
    "min_sample_size": 10000
}
```

---

## 7. 배포

### 7.1 배포 전략

```python
"""
모델 배포 전략
"""

deployment_strategies = {
    "blue_green": {
        "description": "새 버전을 별도 환경에 배포 후 트래픽 전환",
        "rollback": "즉시 가능 (이전 환경으로 트래픽 전환)",
        "use_case": "다운타임 최소화 필요시"
    },
    "canary": {
        "description": "일부 트래픽만 새 버전으로 점진적 전환",
        "rollback": "트래픽 비율 조정으로 가능",
        "use_case": "리스크 최소화, A/B 테스트"
    },
    "shadow": {
        "description": "실제 트래픽 복제하여 새 모델 테스트 (결과 미반영)",
        "rollback": "불필요 (프로덕션 영향 없음)",
        "use_case": "새 모델 검증"
    }
}
```

### 7.2 배포 코드

```python
"""
모델 배포 자동화
"""

import mlflow
from mlflow.tracking import MlflowClient

class ModelDeployer:
    """모델 배포 클래스"""

    def __init__(self, registry_uri: str):
        self.client = MlflowClient(registry_uri)

    def promote_to_production(
        self,
        model_name: str,
        version: str,
        archive_current: bool = True
    ):
        """모델을 프로덕션으로 승격"""
        # 승격 전에 아카이브 — 프로덕션 버전이 정확히 하나만 존재하도록 보장하고
        # 롤백 이력을 유지
        if archive_current:
            current_prod = self.get_production_model(model_name)
            if current_prod:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=current_prod.version,
                    stage="Archived"
                )

        # 새 버전을 프로덕션으로
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )

        print(f"Model {model_name} v{version} promoted to Production")

    def rollback(self, model_name: str):
        """이전 버전으로 롤백"""
        # 가장 최근 아카이브 버전 선택 — 현재 피처 스키마 및 데이터 분포와
        # 호환될 가능성이 가장 높음
        versions = self.client.search_model_versions(
            f"name='{model_name}'"
        )

        archived = [v for v in versions if v.current_stage == "Archived"]
        if not archived:
            raise ValueError("No archived version available for rollback")

        latest_archived = max(archived, key=lambda x: int(x.version))

        # 현재 프로덕션 아카이브
        current_prod = self.get_production_model(model_name)
        if current_prod:
            self.client.transition_model_version_stage(
                name=model_name,
                version=current_prod.version,
                stage="Archived"
            )

        # 롤백 실행
        self.client.transition_model_version_stage(
            name=model_name,
            version=latest_archived.version,
            stage="Production"
        )

        print(f"Rolled back to v{latest_archived.version}")
```

---

## 8. 모니터링

### 8.1 모니터링 메트릭

```python
"""
모델 모니터링 설정
"""

monitoring_config = {
    # 모델 성능 메트릭
    "model_metrics": {
        "accuracy": {"threshold": 0.85, "alert_on": "below"},
        "latency_p99": {"threshold": 100, "alert_on": "above", "unit": "ms"},
        "error_rate": {"threshold": 0.01, "alert_on": "above"}
    },

    # 데이터 드리프트 메트릭
    "drift_metrics": {
        "psi": {"threshold": 0.1, "alert_on": "above"},  # Population Stability Index
        "ks_statistic": {"threshold": 0.1, "alert_on": "above"}
    },

    # 시스템 메트릭
    "system_metrics": {
        "cpu_usage": {"threshold": 80, "alert_on": "above", "unit": "%"},
        "memory_usage": {"threshold": 80, "alert_on": "above", "unit": "%"},
        "gpu_utilization": {"threshold": 90, "alert_on": "above", "unit": "%"}
    }
}
```

### 8.2 재학습 트리거

```python
"""
자동 재학습 트리거
"""

class RetrainingTrigger:
    """재학습 트리거 클래스"""

    def __init__(self, config: dict):
        self.config = config

    def check_triggers(self, metrics: dict) -> dict:
        """재학습 필요 여부 확인"""
        triggers = {
            "should_retrain": False,
            "reasons": []
        }

        # 1. 성능 저하 — 가장 직접적인 신호; 개념 드리프트(concept drift)를 포착
        if metrics.get("accuracy", 1.0) < self.config["min_accuracy"]:
            triggers["should_retrain"] = True
            triggers["reasons"].append("accuracy_degradation")

        # 2. 데이터 드리프트(PSI) — 성능 하락 전 조기 경고;
        # 레이블이 아직 없어도 분포 변화를 감지
        if metrics.get("psi", 0) > self.config["max_psi"]:
            triggers["should_retrain"] = True
            triggers["reasons"].append("data_drift")

        # 3. 시간 기반 재학습 — 메트릭이 감지하지 못하는 점진적 드리프트에 대한 안전망
        days_since_training = metrics.get("days_since_training", 0)
        if days_since_training > self.config["max_days_without_training"]:
            triggers["should_retrain"] = True
            triggers["reasons"].append("scheduled_retrain")

        # 4. 새 데이터 임계치
        if metrics.get("new_data_count", 0) > self.config["new_data_threshold"]:
            triggers["should_retrain"] = True
            triggers["reasons"].append("new_data_available")

        return triggers

# 설정 예시
retrain_config = {
    "min_accuracy": 0.85,
    "max_psi": 0.1,
    "max_days_without_training": 30,
    "new_data_threshold": 100000
}

trigger = RetrainingTrigger(retrain_config)
result = trigger.check_triggers(current_metrics)

if result["should_retrain"]:
    print(f"Triggering retrain due to: {result['reasons']}")
    # trigger_training_pipeline()
```

---

## 9. 버전 관리 전략

### 9.1 전체 버전 관리

```yaml
# version_management.yaml
versioning:
  # 데이터 버전
  data:
    strategy: "semantic"  # v1.0.0
    storage: "dvc"
    format: "parquet"

  # 코드 버전
  code:
    strategy: "git"
    branching: "git-flow"

  # 모델 버전
  model:
    strategy: "semantic"
    registry: "mlflow"
    stages: ["None", "Staging", "Production", "Archived"]

  # 피처 버전
  features:
    strategy: "semantic"
    store: "feast"

  # 연결 관계 추적
  lineage:
    data_version -> code_version -> model_version
    features_version -> model_version
```

### 9.2 시맨틱 버전 관리

```python
"""
모델 시맨틱 버전 관리
"""

# 버전 형식: MAJOR.MINOR.PATCH
# MAJOR: 호환되지 않는 변경 (새 아키텍처, 피처 스키마 변경)
# MINOR: 기능 추가 (새 피처, 하이퍼파라미터 변경)
# PATCH: 버그 수정, 재학습

version_examples = {
    "1.0.0": "초기 프로덕션 릴리스",
    "1.0.1": "동일 데이터/피처로 재학습",
    "1.1.0": "새 피처 추가",
    "1.2.0": "하이퍼파라미터 최적화",
    "2.0.0": "모델 아키텍처 변경 (RF -> XGBoost)"
}
```

---

## 연습 문제

### 문제 1: 파이프라인 설계
이커머스 추천 시스템의 ML 파이프라인을 설계하세요. 데이터 수집부터 모니터링까지 각 단계를 정의하세요.

### 문제 2: 재학습 정책
다음 상황에서 재학습 정책을 설계하세요:
- 일일 신규 주문 10만 건
- 계절성이 강한 상품 판매
- 모델 추론 latency 50ms 이하 요구

---

## 요약

| 단계 | 주요 활동 | 핵심 산출물 |
|------|----------|------------|
| 문제 정의 | 비즈니스 목표, ML 문제 정의 | 프로젝트 문서 |
| 데이터 준비 | 수집, 검증, 버전 관리 | 검증된 데이터셋 |
| 피처 엔지니어링 | 피처 생성, 변환 | 피처 파이프라인 |
| 모델 학습 | 학습, 실험 관리 | 학습된 모델, 메트릭 |
| 검증 | 품질 게이트, A/B 테스트 | 검증 리포트 |
| 배포 | Blue/Green, Canary | 서빙 엔드포인트 |
| 모니터링 | 성능, 드리프트 감지 | 대시보드, 알림 |

---

## 참고 자료

- [MLOps Principles - ML System Design](https://ml-ops.org/)
- [Google MLOps Maturity Model](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Data Version Control (DVC)](https://dvc.org/doc)
