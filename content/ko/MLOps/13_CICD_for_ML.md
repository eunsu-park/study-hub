# 머신러닝을 위한 CI/CD

## 개요

ML CI/CD(지속적 통합/지속적 배포)는 전통적인 소프트웨어 CI/CD에 데이터 검증, 모델 학습, 평가 게이트(Evaluation Gate), 그리고 ML 시스템에 특화된 배포 전략을 더한 것입니다. 이 레슨에서는 GitHub Actions를 활용한 ML 파이프라인 자동화, 데이터·모델 검증 게이트, 배포 전략(섀도우, 카나리, 블루-그린), 롤백 패턴, 그리고 ML 시스템에 고유한 테스트 전략을 다룹니다.

---

## 1. ML CI/CD vs 전통적인 CI/CD

### 1.1 주요 차이점

```python
"""
Traditional Software CI/CD:
  Code → Build → Unit Test → Integration Test → Deploy
  Trigger: Code change (git push)
  Artifact: Application binary / container image

ML CI/CD:
  Code → Build → Unit Test → Data Validation → Train → Evaluate → Deploy
  Triggers: Code change, Data change, Schedule, Drift alert
  Artifacts: Model file, Feature pipeline, Serving config

  ┌──────────────────────────────────────────────────────────────┐
  │                    ML CI/CD Pipeline                         │
  │                                                              │
  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌────┐ │
  │  │ Code │→ │ Data │→ │Train │→ │ Eval │→ │Stage │→ │Prod│ │
  │  │ Test │  │Valid.│  │Model │  │ Gate │  │ Test │  │    │ │
  │  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └────┘ │
  │                                                              │
  │  Extra ML concerns:                                         │
  │  - Data quality checks before training                      │
  │  - Model performance thresholds (accuracy, latency)         │
  │  - A/B testing in production                                │
  │  - Automated rollback on performance degradation            │
  └──────────────────────────────────────────────────────────────┘
"""
```

### 1.2 ML 파이프라인 트리거(Trigger)

```python
"""
Trigger Types:

1. Code Change (git push):
   - Retrains with existing data + new code
   - Runs full test suite + training

2. Data Change (new data arrives):
   - Retrains with new data + existing code
   - Data validation first, then training

3. Scheduled (cron):
   - Periodic retraining (daily/weekly)
   - Ensures model stays fresh

4. Drift Alert (monitoring):
   - Data drift or model performance degradation detected
   - Triggers emergency retraining pipeline

5. Manual (on-demand):
   - Hyperparameter tuning experiments
   - A/B test setup
"""
```

---

## 2. ML을 위한 GitHub Actions

### 2.1 ML 파이프라인 워크플로우

```yaml
# .github/workflows/ml_pipeline.yaml
name: ML Pipeline

on:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'configs/**'
      - 'requirements.txt'
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6am
  workflow_dispatch:       # Manual trigger
    inputs:
      skip_training:
        description: 'Skip training (deploy existing model)'
        type: boolean
        default: false

env:
  MODEL_REGISTRY: mlflow
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}

jobs:
  # ── Stage 1: Code Quality ──
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt
      - name: Lint
        run: ruff check src/ tests/
      - name: Type check
        run: mypy src/ --ignore-missing-imports
      - name: Unit tests
        run: pytest tests/unit/ -v --cov=src --cov-report=xml

  # ── Stage 2: Data Validation ──
  data-validation:
    runs-on: ubuntu-latest
    needs: code-quality
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Validate training data
        run: python scripts/validate_data.py
        env:
          DATA_PATH: ${{ secrets.TRAINING_DATA_PATH }}
      - name: Check data freshness
        run: python scripts/check_data_freshness.py --max-age-days 7

  # ── Stage 3: Model Training ──
  train:
    runs-on: ubuntu-latest
    needs: data-validation
    if: ${{ !inputs.skip_training }}
    outputs:
      model_version: ${{ steps.register.outputs.model_version }}
      run_id: ${{ steps.train.outputs.run_id }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Train model
        id: train
        run: |
          RUN_ID=$(python scripts/train.py --config configs/production.yaml)
          echo "run_id=$RUN_ID" >> $GITHUB_OUTPUT
      - name: Register model
        id: register
        run: |
          VERSION=$(python scripts/register_model.py \
            --run-id ${{ steps.train.outputs.run_id }} \
            --model-name production-model)
          echo "model_version=$VERSION" >> $GITHUB_OUTPUT

  # ── Stage 4: Model Evaluation Gate ──
  evaluate:
    runs-on: ubuntu-latest
    needs: train
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Evaluate model
        run: |
          python scripts/evaluate.py \
            --model-version ${{ needs.train.outputs.model_version }} \
            --min-accuracy 0.92 \
            --max-latency-ms 50 \
            --min-improvement 0.005
      - name: Bias and fairness check
        run: python scripts/fairness_check.py --model-version ${{ needs.train.outputs.model_version }}

  # ── Stage 5: Staging Deployment ──
  deploy-staging:
    runs-on: ubuntu-latest
    needs: evaluate
    environment: staging
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to staging
        run: |
          python scripts/deploy.py \
            --model-version ${{ needs.train.outputs.model_version }} \
            --environment staging
      - name: Integration tests
        run: pytest tests/integration/ -v --timeout=300
      - name: Load test
        run: python scripts/load_test.py --target staging --rps 100 --duration 60

  # ── Stage 6: Production Deployment ──
  deploy-production:
    runs-on: ubuntu-latest
    needs: deploy-staging
    environment: production
    steps:
      - uses: actions/checkout@v4
      - name: Canary deploy (10%)
        run: |
          python scripts/deploy.py \
            --model-version ${{ needs.train.outputs.model_version }} \
            --environment production \
            --strategy canary \
            --traffic-percent 10
      - name: Monitor canary (15 min)
        run: python scripts/monitor_canary.py --duration-minutes 15 --threshold 0.01
      - name: Full rollout
        run: |
          python scripts/deploy.py \
            --model-version ${{ needs.train.outputs.model_version }} \
            --environment production \
            --strategy canary \
            --traffic-percent 100
```

---

## 3. 데이터 검증 게이트(Data Validation Gate)

### 3.1 자동화된 데이터 검사

```python
"""
Data validation ensures training data meets quality standards
before expensive model training begins.
"""

import pandas as pd
import numpy as np
import json
import sys


def validate_training_data(data_path, config_path="configs/data_validation.json"):
    """Validate training data before model training.

    Returns exit code 0 if all checks pass, 1 if any fail.
    """
    df = pd.read_parquet(data_path)

    with open(config_path) as f:
        config = json.load(f)

    results = []

    # 1. Row count check
    min_rows = config.get("min_rows", 1000)
    results.append({
        "check": "row_count",
        "passed": len(df) >= min_rows,
        "detail": f"{len(df)} rows (min: {min_rows})",
    })

    # 2. Missing value check
    max_null_pct = config.get("max_null_percent", 5.0)
    null_pcts = (df.isnull().sum() / len(df) * 100).to_dict()
    high_null_cols = {k: v for k, v in null_pcts.items() if v > max_null_pct}
    results.append({
        "check": "missing_values",
        "passed": len(high_null_cols) == 0,
        "detail": f"High null columns: {high_null_cols}" if high_null_cols else "OK",
    })

    # 3. Schema check
    expected_columns = set(config.get("required_columns", []))
    actual_columns = set(df.columns)
    missing = expected_columns - actual_columns
    results.append({
        "check": "schema",
        "passed": len(missing) == 0,
        "detail": f"Missing columns: {missing}" if missing else "OK",
    })

    # 4. Target distribution check (detect label shift)
    target_col = config.get("target_column", "target")
    if target_col in df.columns:
        value_counts = df[target_col].value_counts(normalize=True)
        min_class_pct = config.get("min_class_percent", 1.0)
        rare_classes = value_counts[value_counts < min_class_pct / 100]
        results.append({
            "check": "target_distribution",
            "passed": len(rare_classes) == 0,
            "detail": f"Rare classes (<{min_class_pct}%): {rare_classes.to_dict()}"
                      if len(rare_classes) > 0 else "OK",
        })

    # 5. Feature range check
    for feature, bounds in config.get("feature_ranges", {}).items():
        if feature in df.columns:
            out_of_range = (
                (df[feature] < bounds["min"]) | (df[feature] > bounds["max"])
            ).sum()
            results.append({
                "check": f"range_{feature}",
                "passed": out_of_range == 0,
                "detail": f"{out_of_range} values out of [{bounds['min']}, {bounds['max']}]",
            })

    # 6. Duplicate check
    dup_count = df.duplicated().sum()
    max_dup_pct = config.get("max_duplicate_percent", 1.0)
    dup_pct = dup_count / len(df) * 100
    results.append({
        "check": "duplicates",
        "passed": dup_pct <= max_dup_pct,
        "detail": f"{dup_count} duplicates ({dup_pct:.2f}%)",
    })

    # Report
    all_passed = all(r["passed"] for r in results)
    print(f"\n{'='*60}")
    print(f"Data Validation Report: {'PASSED' if all_passed else 'FAILED'}")
    print(f"{'='*60}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['check']}: {r['detail']}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/training.parquet"
    sys.exit(validate_training_data(data_path))
```

---

## 4. 모델 평가 게이트(Model Evaluation Gate)

### 4.1 평가 기준

```python
"""
Model Evaluation Gate:
  A model must pass ALL criteria before deployment.

  ┌─────────────────────────────────────────────────┐
  │              Evaluation Gate                     │
  │                                                  │
  │  ✓ Accuracy >= 0.92                             │
  │  ✓ Latency p99 <= 50ms                          │
  │  ✓ Improvement >= 0.5% over current production  │
  │  ✓ Fairness: demographic parity >= 0.8          │
  │  ✓ No data leakage detected                     │
  │  ✓ Model size <= 500MB                          │
  │                                                  │
  │  All pass → Proceed to deployment               │
  │  Any fail → Block and notify team               │
  └─────────────────────────────────────────────────┘
"""

import json
import time
import sys


def evaluate_model(model_version, config_path="configs/eval_gate.json"):
    """Evaluate model against deployment criteria."""
    with open(config_path) as f:
        config = json.load(f)

    # Load model and test data (simplified)
    # model = load_model(model_version)
    # X_test, y_test = load_test_data()

    results = []

    # 1. Accuracy check
    # accuracy = compute_accuracy(model, X_test, y_test)
    accuracy = 0.935  # placeholder
    min_accuracy = config.get("min_accuracy", 0.90)
    results.append({
        "check": "accuracy",
        "value": accuracy,
        "threshold": min_accuracy,
        "passed": accuracy >= min_accuracy,
    })

    # 2. Latency check
    # latencies = benchmark_latency(model, X_test[:1000])
    # p99_latency = np.percentile(latencies, 99)
    p99_latency = 42.0  # placeholder (ms)
    max_latency = config.get("max_latency_p99_ms", 50)
    results.append({
        "check": "latency_p99",
        "value": f"{p99_latency:.1f}ms",
        "threshold": f"{max_latency}ms",
        "passed": p99_latency <= max_latency,
    })

    # 3. Improvement over production
    # prod_accuracy = get_production_model_accuracy()
    prod_accuracy = 0.928  # placeholder
    min_improvement = config.get("min_improvement", 0.005)
    improvement = accuracy - prod_accuracy
    results.append({
        "check": "improvement",
        "value": f"{improvement:.4f}",
        "threshold": f">= {min_improvement}",
        "passed": improvement >= min_improvement,
    })

    # 4. Model size check
    # model_size_mb = get_model_size(model_version)
    model_size_mb = 245.0  # placeholder
    max_size_mb = config.get("max_model_size_mb", 500)
    results.append({
        "check": "model_size",
        "value": f"{model_size_mb:.0f}MB",
        "threshold": f"<= {max_size_mb}MB",
        "passed": model_size_mb <= max_size_mb,
    })

    # Report
    all_passed = all(r["passed"] for r in results)
    print(f"\n{'='*60}")
    print(f"Model Evaluation Gate: {'PASSED' if all_passed else 'BLOCKED'}")
    print(f"Model Version: {model_version}")
    print(f"{'='*60}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['check']}: {r['value']} (threshold: {r['threshold']})")

    if not all_passed:
        print("\nDeployment BLOCKED. Fix failing checks before retrying.")
        sys.exit(1)

    print("\nModel approved for deployment.")
    return results
```

---

## 5. 배포 전략(Deployment Strategies)

### 5.1 전략 비교

```python
"""
ML Deployment Strategies:

1. Direct Replacement (Big Bang):
   [v1 100%] → [v2 100%]
   ✓ Simple
   ✗ Risky — no rollback time

2. Shadow Deployment:
   [v1 100%] → [v1 100% + v2 shadow]
   Both models receive all traffic, but only v1 serves responses.
   v2 predictions are logged for comparison.
   ✓ Zero-risk testing with production traffic
   ✗ Double compute cost

3. Canary Deployment:
   [v1 100%] → [v1 90% + v2 10%] → [v1 0% + v2 100%]
   Gradually shift traffic to the new model.
   ✓ Early detection of issues
   ✓ Automatic rollback on degradation
   ✗ Requires traffic splitting infrastructure

4. Blue-Green:
   [Blue v1 active] → [Green v2 ready] → [switch] → [Green v2 active]
   Two identical environments; instant switch.
   ✓ Instant rollback (switch back)
   ✗ Double infrastructure cost

5. A/B Testing:
   [v1 50% + v2 50%] for N days → analyze → promote winner
   Statistical comparison of business metrics.
   ✓ Data-driven decisions
   ✗ Requires longer evaluation period
"""
```

### 5.2 카나리 배포(Canary Deployment) 구현

```python
import time
import sys


class CanaryDeployer:
    """Canary deployment with automatic rollback."""

    def __init__(self, model_version, steps=None):
        self.model_version = model_version
        self.steps = steps or [5, 10, 25, 50, 100]  # Traffic percentages
        self.current_step = 0

    def deploy_canary(self, traffic_percent):
        """Deploy model to a percentage of traffic."""
        print(f"Deploying v{self.model_version} to {traffic_percent}% of traffic...")
        # In production: update load balancer / service mesh / feature flag
        # Example: kubectl set traffic split

    def check_metrics(self, duration_seconds=300):
        """Monitor canary metrics for the specified duration.

        Returns True if metrics are healthy, False if rollback needed.
        """
        print(f"Monitoring canary for {duration_seconds}s...")
        # In production: query Prometheus/Datadog for error rate, latency, accuracy
        # Simplified check:
        # error_rate = get_error_rate(self.model_version)
        # latency_p99 = get_latency_p99(self.model_version)
        # return error_rate < 0.01 and latency_p99 < 100
        return True  # placeholder

    def rollback(self):
        """Rollback: route all traffic to previous version."""
        print(f"ROLLBACK: Removing v{self.model_version} from traffic")
        self.deploy_canary(0)

    def run(self, check_duration=300):
        """Execute the full canary rollout."""
        for pct in self.steps:
            self.deploy_canary(pct)

            if pct < 100:
                healthy = self.check_metrics(check_duration)
                if not healthy:
                    self.rollback()
                    print(f"Canary FAILED at {pct}% traffic. Rolled back.")
                    return False

        print(f"Canary deployment COMPLETE: v{self.model_version} at 100%")
        return True
```

---

## 6. ML 테스트 전략(Testing Strategy for ML)

### 6.1 ML 테스트 피라미드(Test Pyramid)

```python
"""
ML Test Pyramid (bottom to top):

                    ┌───────────┐
                    │  E2E /    │  ← Full pipeline tests
                    │  System   │     (expensive, slow)
                   ┌┴───────────┴┐
                   │  Integration │  ← Data pipeline + model tests
                  ┌┴──────────────┴┐
                  │   Model Tests   │  ← Training, evaluation, inference
                 ┌┴────────────────┴┐
                 │  Data Tests       │  ← Schema, quality, distribution
                ┌┴──────────────────┴┐
                │   Unit Tests        │  ← Feature logic, transforms
                └─────────────────────┘

ML-Specific Test Types:

1. Data Tests:
   - Schema validation (column names, types)
   - No unexpected nulls
   - Feature distributions within expected range
   - No data leakage between train/test

2. Model Tests:
   - Model trains without errors
   - Predictions are within expected range
   - Model beats baseline (e.g., random, majority class)
   - Invariance tests (model output stable to irrelevant changes)
   - Directional tests (known feature changes → expected prediction change)

3. Infrastructure Tests:
   - Model serialization/deserialization roundtrip
   - Serving endpoint returns predictions
   - Latency under load
   - Memory usage within limits
"""
```

### 6.2 ML 테스트 예제

```python
import pytest
import numpy as np
import pickle


class TestFeatureEngineering:
    """Unit tests for feature engineering functions."""

    def test_cyclical_encoding(self):
        from src.features import cyclical_encode
        result = cyclical_encode(hour=0, period=24)
        assert abs(result["sin"] - 0.0) < 1e-6
        assert abs(result["cos"] - 1.0) < 1e-6

        result = cyclical_encode(hour=6, period=24)
        assert abs(result["sin"] - 1.0) < 1e-6

    def test_null_handling(self):
        from src.features import handle_nulls
        import pandas as pd
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, 2, 3]})
        result = handle_nulls(df)
        assert result.isnull().sum().sum() == 0


class TestModel:
    """Model-level tests."""

    def test_model_trains(self, sample_data):
        """Model trains without errors on sample data."""
        from src.train import train_model
        model = train_model(sample_data["X_train"], sample_data["y_train"])
        assert model is not None

    def test_predictions_in_range(self, trained_model, sample_data):
        """Predictions are within expected bounds."""
        preds = trained_model.predict(sample_data["X_test"])
        assert preds.min() >= 0, "Negative predictions"
        assert preds.max() <= 1, "Predictions above 1"

    def test_beats_baseline(self, trained_model, sample_data):
        """Model accuracy exceeds random baseline."""
        from sklearn.metrics import accuracy_score
        preds = trained_model.predict(sample_data["X_test"])
        accuracy = accuracy_score(sample_data["y_test"], preds)
        baseline = max(sample_data["y_test"].value_counts(normalize=True))
        assert accuracy > baseline, f"Model ({accuracy:.3f}) worse than baseline ({baseline:.3f})"

    def test_invariance(self, trained_model):
        """Irrelevant feature changes don't affect predictions."""
        x1 = np.array([[25, 50000, 1]])  # age, income, employed
        x2 = np.array([[25, 50000, 1]])  # Same features
        pred1 = trained_model.predict_proba(x1)
        pred2 = trained_model.predict_proba(x2)
        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_directionality(self, trained_model):
        """Known feature changes produce expected prediction changes."""
        # Higher income should increase credit approval probability
        x_low = np.array([[30, 30000, 1]])
        x_high = np.array([[30, 100000, 1]])
        pred_low = trained_model.predict_proba(x_low)[0][1]
        pred_high = trained_model.predict_proba(x_high)[0][1]
        assert pred_high > pred_low, "Higher income should increase approval prob"

    def test_serialization_roundtrip(self, trained_model, tmp_path):
        """Model survives save/load cycle."""
        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(trained_model, f)
        with open(model_path, "rb") as f:
            loaded = pickle.load(f)

        x = np.array([[30, 50000, 1]])
        np.testing.assert_array_equal(
            trained_model.predict(x),
            loaded.predict(x),
        )
```

---

## 7. 롤백 패턴(Rollback Patterns)

### 7.1 자동 롤백(Automated Rollback)

```python
"""
Rollback Triggers:

1. Error rate spike (> 2x baseline)
2. Latency degradation (p99 > threshold)
3. Accuracy drop (below minimum threshold)
4. Data quality issues in predictions
5. Manual override (human-in-the-loop)

Rollback Mechanisms:

1. Model version rollback:
   - Switch serving to previous model version
   - Model Registry tracks all versions

2. Traffic rollback:
   - Route 100% traffic back to old model
   - Used with canary / blue-green deployments

3. Feature flag rollback:
   - Disable new model via feature flag
   - Fastest rollback (no deployment needed)
"""


class AutoRollback:
    """Monitor model and trigger automatic rollback."""

    def __init__(self, model_version, previous_version,
                 error_threshold=0.02, latency_threshold_ms=100):
        self.model_version = model_version
        self.previous_version = previous_version
        self.error_threshold = error_threshold
        self.latency_threshold_ms = latency_threshold_ms

    def check_health(self):
        """Check if the deployed model is healthy."""
        # In production: query monitoring system
        # metrics = get_model_metrics(self.model_version)
        metrics = {"error_rate": 0.005, "latency_p99_ms": 45}  # placeholder

        issues = []
        if metrics["error_rate"] > self.error_threshold:
            issues.append(f"Error rate {metrics['error_rate']:.3f} > {self.error_threshold}")
        if metrics["latency_p99_ms"] > self.latency_threshold_ms:
            issues.append(f"Latency {metrics['latency_p99_ms']}ms > {self.latency_threshold_ms}ms")

        return len(issues) == 0, issues

    def rollback(self, reason):
        """Rollback to previous model version."""
        print(f"ROLLBACK triggered: {reason}")
        print(f"Rolling back from v{self.model_version} to v{self.previous_version}")
        # In production: update serving config, notify team
        # deploy_model(self.previous_version)
        # send_alert(f"Model rollback: {reason}")
```

---

## 8. 연습 문제

### 연습 1: ML CI/CD 파이프라인 구축

```python
"""
Create a GitHub Actions workflow for an ML project:
1. Code quality: lint, type check, unit tests
2. Data validation: schema check, null check, distribution check
3. Training: train model, log to MLflow
4. Evaluation gate: accuracy > 0.90, latency < 100ms
5. Deploy to staging with integration tests
6. Canary deploy to production (10% → 50% → 100%)
7. Add rollback on evaluation gate failure
"""
```

### 연습 2: ML 테스트 스위트(Test Suite)

```python
"""
Write a comprehensive test suite for an ML pipeline:
1. Data tests: schema, nulls, distribution, leakage
2. Feature tests: encoding, scaling, null handling
3. Model tests: trains, beats baseline, invariance, directional
4. Integration tests: end-to-end pipeline, serialization
5. Performance tests: latency, throughput, memory
6. Run tests in CI and fail the pipeline on any failure
"""
```

---

## 9. 요약

### 핵심 정리

| 개념 | 설명 |
|------|------|
| **ML CI/CD** | 전통적 CI/CD에 데이터 검증, 학습, 평가 게이트를 추가한 파이프라인 |
| **트리거(Triggers)** | 코드 변경, 데이터 변경, 스케줄, 드리프트(Drift) 알림 |
| **데이터 검증** | 학습 전 스키마(Schema), 결측값, 분포, 최신성(Freshness) 검사 |
| **평가 게이트(Evaluation Gate)** | 정확도, 지연 시간, 성능 향상, 공정성 임계값 |
| **카나리 배포(Canary Deploy)** | 각 단계에서 모니터링하며 트래픽을 점진적으로 전환 |
| **ML 테스트** | 데이터 테스트, 모델 테스트(불변성, 방향성), 인프라 테스트 |
| **롤백(Rollback)** | 에러율·지연 시간·정확도 저하 시 자동 롤백 |

### 모범 사례

1. **학습 전 게이트** — 불필요한 컴퓨팅 낭비를 막기 위해 데이터 품질을 먼저 검증한다
2. **배포 전 게이트** — 신규 모델은 반드시 현재 프로덕션(Production) 모델보다 나아야 한다
3. **항상 카나리 배포** — 100% 즉시 배포는 금물; 점진적 롤아웃을 사용한다
4. **불변성(Invariance) 테스트** — 관련 없는 피처 변경에도 모델 출력이 안정적인지 확인한다
5. **롤백 자동화** — 프로덕션 장애에서 사람의 개입(Human-in-the-loop)은 너무 느리다
6. **모든 것을 버전 관리** — 코드, 데이터, 모델, 설정, 파이프라인 정의 모두 버전 관리한다

### 다음 단계

- **L14**: DVC — 데이터와 ML 실험을 위한 버전 관리
- **L15**: LLMOps — LLM 애플리케이션에 특화된 CI/CD 패턴
