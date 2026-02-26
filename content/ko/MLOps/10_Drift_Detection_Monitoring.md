# 드리프트 감지 & 모니터링

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 데이터 드리프트(data drift), 개념 드리프트(concept drift), 레이블 드리프트(label drift)를 구별하고 각 유형이 프로덕션 모델 성능을 어떻게 저하시키는지 설명할 수 있다
2. KS 검정(KS test), PSI(Population Stability Index), 카이제곱 검정(chi-squared test) 등 통계적 드리프트 감지 검정을 구현하여 입력 피처의 분포 변화를 감지할 수 있다
3. 예측 로그를 수집하고 드리프트 메트릭을 계산하며 임계값 초과 시 알림을 트리거하는 모델 모니터링 파이프라인을 구축할 수 있다
4. Evidently AI 또는 유사 도구를 사용하여 자동화된 데이터 품질 및 드리프트 모니터링 리포트를 생성할 수 있다
5. 불필요한 재학습 오버헤드를 최소화하면서 감지된 드리프트에 대응하는 재학습 트리거(retraining trigger) 전략을 설계할 수 있다

---

## 1. 드리프트 개념

드리프트(Drift)는 시간이 지남에 따라 데이터나 모델 성능이 변화하는 현상입니다.

### 1.1 드리프트 유형

```
┌─────────────────────────────────────────────────────────────────────┐
│                        드리프트 유형                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │                  Data Drift                              │      │
│   │                                                          │      │
│   │   입력 데이터의 분포가 변화                               │      │
│   │   P(X)_train ≠ P(X)_production                          │      │
│   │                                                          │      │
│   │   예: 고객 연령대 분포 변화                               │      │
│   └─────────────────────────────────────────────────────────┘      │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │                  Concept Drift                           │      │
│   │                                                          │      │
│   │   입력과 출력 간의 관계가 변화                            │      │
│   │   P(Y|X)_train ≠ P(Y|X)_production                      │      │
│   │                                                          │      │
│   │   예: 경제 상황 변화로 이탈 패턴 변화                      │      │
│   └─────────────────────────────────────────────────────────┘      │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │                  Label Drift                             │      │
│   │                                                          │      │
│   │   타겟 변수의 분포가 변화                                 │      │
│   │   P(Y)_train ≠ P(Y)_production                          │      │
│   │                                                          │      │
│   │   예: 사기 비율 증가                                      │      │
│   └─────────────────────────────────────────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 드리프트의 원인

```python
"""
드리프트 원인 분석
"""

drift_causes = {
    "데이터 수집 변화": {
        "examples": ["센서 오작동", "데이터 소스 변경", "로깅 버그"],
        "detection": "스키마 검증, 통계 모니터링"
    },
    "외부 환경 변화": {
        "examples": ["계절성", "경쟁사 행동", "경제 상황", "규제 변화"],
        "detection": "도메인 지식, 장기 트렌드 분석"
    },
    "사용자 행동 변화": {
        "examples": ["신규 사용자 유입", "사용 패턴 변화", "세대 교체"],
        "detection": "코호트 분석, A/B 테스트"
    },
    "피처 엔지니어링 오류": {
        "examples": ["업스트림 데이터 변경", "전처리 버그"],
        "detection": "피처 스토어 검증, 단위 테스트"
    }
}
```

---

## 2. 드리프트 감지 기법

### 2.1 통계적 검정

```python
"""
드리프트 감지 통계 기법
"""

import numpy as np
from scipy import stats
from typing import Tuple

def kolmogorov_smirnov_test(
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float = 0.05
) -> Tuple[float, bool]:
    """KS 검정 - 두 분포의 차이 검정"""
    # 비모수(non-parametric) 검정: t-test나 카이제곱과 달리 분포 가정 불필요
    statistic, p_value = stats.ks_2samp(reference, current)
    is_drift = p_value < threshold
    return statistic, is_drift

def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10
) -> float:
    """PSI - 분포 안정성 지수"""
    # 비닝(binning)으로 요약 통계량(평균/표준편차)이 놓치는 분포 형태 변화를 포착
    bins = np.histogram_bin_edges(reference, bins=n_bins)
    ref_hist, _ = np.histogram(reference, bins=bins, density=True)
    cur_hist, _ = np.histogram(current, bins=bins, density=True)

    # log(0)은 -inf를 생성하여 PSI 값을 오염시키므로 소량값으로 대체
    ref_hist = np.where(ref_hist == 0, 0.0001, ref_hist)
    cur_hist = np.where(cur_hist == 0, 0.0001, cur_hist)

    # PSI 계산
    psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))

    return psi

def wasserstein_distance(
    reference: np.ndarray,
    current: np.ndarray
) -> float:
    """Wasserstein 거리 (Earth Mover's Distance)"""
    return stats.wasserstein_distance(reference, current)

def jensen_shannon_divergence(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10
) -> float:
    """Jensen-Shannon Divergence"""
    # 대칭적이고 [0,1] 범위 — KL 발산(divergence)보다 임계값 설정이 용이
    from scipy.spatial.distance import jensenshannon

    bins = np.histogram_bin_edges(reference, bins=n_bins)
    ref_hist, _ = np.histogram(reference, bins=bins, density=True)
    cur_hist, _ = np.histogram(current, bins=bins, density=True)

    return jensenshannon(ref_hist, cur_hist)

# 해석 기준
psi_thresholds = {
    "PSI < 0.1": "변화 없음",
    "0.1 <= PSI < 0.2": "약간의 변화",
    "PSI >= 0.2": "심각한 변화 - 재학습 필요"
}
```

### 2.2 다변량 드리프트 감지

```python
"""
다변량 드리프트 감지
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def domain_classifier_drift(
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float = 0.55
) -> Tuple[float, bool]:
    """
    도메인 분류기 기반 드리프트 감지
    - 참조 데이터와 현재 데이터를 구분하는 분류기 학습
    - AUC가 0.5에 가까우면 드리프트 없음
    """
    # 개별 피처 검정이 놓치는 다변량 드리프트(예: 상관관계 변화)를 포착
    X = np.vstack([reference, current])
    y = np.hstack([
        np.zeros(len(reference)),
        np.ones(len(current))
    ])

    # 교차 검증(cross-validation)으로 과적합 방지 — AUC=0.5면 학습 가능한 드리프트 없음
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    mean_auc = scores.mean()

    # 0.5에서 멀수록 드리프트 가능성 높음
    drift_score = abs(mean_auc - 0.5) * 2
    is_drift = mean_auc > threshold

    return drift_score, is_drift

def multivariate_drift_pca(
    reference: np.ndarray,
    current: np.ndarray,
    n_components: int = 5
) -> float:
    """
    PCA 기반 다변량 드리프트 감지
    """
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import mahalanobis

    # PCA로 차원의 저주(curse of dimensionality) 완화 — 피처 수 > 샘플 수일 때 Mahalanobis 실패
    pca = PCA(n_components=n_components)
    ref_pca = pca.fit_transform(reference)
    cur_pca = pca.transform(current)

    # 각 차원의 평균과 공분산
    ref_mean = np.mean(ref_pca, axis=0)
    cur_mean = np.mean(cur_pca, axis=0)
    ref_cov = np.cov(ref_pca.T)

    # Mahalanobis 거리
    try:
        distance = mahalanobis(ref_mean, cur_mean, np.linalg.inv(ref_cov))
    except np.linalg.LinAlgError:
        # 공분산 행렬이 특이(singular)하면(공선성 피처) 유클리드 거리로 폴백
        distance = np.linalg.norm(ref_mean - cur_mean)

    return distance
```

---

## 3. Evidently AI

### 3.1 Evidently 기본 사용

```python
"""
Evidently AI 드리프트 감지
"""

import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DataDriftTable,
    ColumnDriftMetric
)

# 데이터 준비
reference_data = pd.read_csv("reference_data.csv")
current_data = pd.read_csv("current_data.csv")

# 명시적 매핑으로 Evidently가 컬럼 타입별로 적절한 통계 검정을 선택하도록 보장
column_mapping = ColumnMapping(
    target="target",
    prediction="prediction",
    numerical_features=["age", "tenure", "monthly_charges"],
    categorical_features=["gender", "contract_type"]
)

# 데이터 드리프트 리포트
drift_report = Report(metrics=[
    DatasetDriftMetric(),
    DataDriftTable(),
])

drift_report.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=column_mapping
)

# HTML 리포트 저장
drift_report.save_html("drift_report.html")

# JSON 결과 (프로그래밍 사용)
result = drift_report.as_dict()
print(f"Dataset drift detected: {result['metrics'][0]['result']['dataset_drift']}")
```

### 3.2 상세 드리프트 분석

```python
"""
Evidently 상세 분석
"""

from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric,
    ColumnSummaryMetric,
    ColumnQuantileMetric,
    ColumnValueRangeMetric
)

# 특정 컬럼 상세 분석
column_report = Report(metrics=[
    ColumnDriftMetric(column_name="monthly_charges"),
    ColumnSummaryMetric(column_name="monthly_charges"),
    ColumnQuantileMetric(column_name="monthly_charges", quantile=0.95),
    ColumnValueRangeMetric(column_name="monthly_charges"),
])

column_report.run(
    reference_data=reference_data,
    current_data=current_data
)

# 결과 추출
result = column_report.as_dict()

for metric in result["metrics"]:
    metric_name = metric["metric"]
    metric_result = metric["result"]
    print(f"{metric_name}: {metric_result}")
```

### 3.3 모델 성능 모니터링

```python
"""
Evidently 모델 성능 모니터링
"""

from evidently.report import Report
from evidently.metric_preset import ClassificationPreset, RegressionPreset
from evidently.metrics import (
    ClassificationQualityMetric,
    ClassificationClassBalance,
    ClassificationConfusionMatrix
)

# 분류 모델 성능 리포트
classification_report = Report(metrics=[
    ClassificationPreset(),
])

classification_report.run(
    reference_data=reference_data,  # 학습 데이터 + 예측
    current_data=current_data,       # 현재 데이터 + 예측
    column_mapping=column_mapping
)

classification_report.save_html("model_performance_report.html")

# 결과 추출
result = classification_report.as_dict()
performance = result["metrics"][0]["result"]["current"]

print(f"Accuracy: {performance['accuracy']}")
print(f"Precision: {performance['precision']}")
print(f"Recall: {performance['recall']}")
```

### 3.4 실시간 모니터링

```python
"""
Evidently Test Suite (자동화된 검사)
"""

from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset, DataQualityTestPreset
from evidently.tests import (
    TestColumnDrift,
    TestShareOfMissingValues,
    TestColumnValueRange,
    TestNumberOfRows
)

# 테스트 스위트는 pass/fail 반환 — CI/CD에 통합하여 데이터 품질로 배포를 게이트
test_suite = TestSuite(tests=[
    # 프리셋은 전체 컬럼 커버; 개별 테스트는 핵심 피처에 더 엄격한 임계값 적용
    DataDriftTestPreset(),

    # 개별 테스트
    TestColumnDrift(column_name="monthly_charges", stattest_threshold=0.1),
    TestShareOfMissingValues(column_name="age", lt=0.05),
    # 도메인 제약 조건으로 데이터 파이프라인 버그 포착 (예: 인코딩 오류로 음수 나이)
    TestColumnValueRange(column_name="age", left=18, right=100),
    TestNumberOfRows(gte=1000),
])

test_suite.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=column_mapping
)

# 결과 확인
result = test_suite.as_dict()

print(f"Tests passed: {result['summary']['success_tests']}")
print(f"Tests failed: {result['summary']['failed_tests']}")

# 실패한 테스트 상세
for test in result["tests"]:
    if test["status"] == "FAIL":
        print(f"FAILED: {test['name']} - {test['description']}")
```

---

## 4. 모니터링 파이프라인

### 4.1 완전한 모니터링 시스템

```python
"""
프로덕션 모니터링 파이프라인
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric

class ModelMonitor:
    """ML 모델 모니터링 시스템"""

    def __init__(
        self,
        reference_data: pd.DataFrame,
        column_mapping,
        alert_thresholds: Dict[str, float]
    ):
        # 참조 데이터를 초기화 시 한 번 로드 — 매 모니터링 주기마다 스토리지 재읽기 방지
        self.reference_data = reference_data
        self.column_mapping = column_mapping
        self.thresholds = alert_thresholds
        # 이력으로 추세 분석 가능 — 단일 시점 드리프트는 노이즈, 지속적 드리프트가 진짜 신호
        self.monitoring_history = []

    def check_data_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """데이터 드리프트 검사"""
        report = Report(metrics=[DatasetDriftMetric()])
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )

        result = report.as_dict()
        drift_result = result["metrics"][0]["result"]

        return {
            "timestamp": datetime.now().isoformat(),
            "dataset_drift": drift_result["dataset_drift"],
            "drift_share": drift_result["drift_share"],
            "number_of_drifted_columns": drift_result["number_of_drifted_columns"],
            "columns": drift_result.get("drift_by_columns", {})
        }

    def check_model_performance(
        self,
        predictions: pd.Series,
        actuals: pd.Series
    ) -> Dict[str, float]:
        """모델 성능 검사"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics = {
            "accuracy": accuracy_score(actuals, predictions),
            "precision": precision_score(actuals, predictions, average="macro"),
            "recall": recall_score(actuals, predictions, average="macro"),
            "f1_score": f1_score(actuals, predictions, average="macro")
        }

        return metrics

    def generate_alerts(self, drift_result: Dict, performance: Dict) -> List[str]:
        """알림 생성"""
        alerts = []

        # 데이터 드리프트 알림
        if drift_result["dataset_drift"]:
            alerts.append(f"DATA_DRIFT: {drift_result['drift_share']:.1%} of features drifted")

        # 성능 저하 알림
        for metric, value in performance.items():
            if metric in self.thresholds and value < self.thresholds[metric]:
                alerts.append(
                    f"PERFORMANCE_DEGRADATION: {metric} = {value:.4f} "
                    f"(threshold: {self.thresholds[metric]})"
                )

        return alerts

    def run_monitoring(
        self,
        current_data: pd.DataFrame,
        predictions: pd.Series = None,
        actuals: pd.Series = None
    ) -> Dict[str, Any]:
        """전체 모니터링 실행"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "data_drift": self.check_data_drift(current_data),
            "alerts": []
        }

        # 실제 레이블(actuals)은 지연 도착(수시간/수일)하는 경우가 많음 — 드리프트 감지는 레이블 없이도 동작
        if predictions is not None and actuals is not None:
            result["performance"] = self.check_model_performance(predictions, actuals)
            result["alerts"] = self.generate_alerts(
                result["data_drift"],
                result["performance"]
            )
        else:
            result["alerts"] = self.generate_alerts(result["data_drift"], {})

        self.monitoring_history.append(result)

        return result

# 사용 예시
monitor = ModelMonitor(
    reference_data=reference_data,
    column_mapping=column_mapping,
    alert_thresholds={
        "accuracy": 0.85,
        "f1_score": 0.80
    }
)

# 매시간 모니터링
result = monitor.run_monitoring(
    current_data=hourly_data,
    predictions=predictions,
    actuals=actuals
)

if result["alerts"]:
    for alert in result["alerts"]:
        print(f"ALERT: {alert}")
        # send_slack_notification(alert)
        # send_email_alert(alert)
```

### 4.2 Prometheus + Grafana 통합

```python
"""
Prometheus 메트릭 노출
"""

from prometheus_client import Gauge, Counter, Histogram, start_http_server
import time

# Gauge(Counter 아님): 드리프트 점수는 시간에 따라 증가/감소할 수 있으므로
DRIFT_SCORE = Gauge(
    "model_drift_score",
    "Current drift score",
    ["feature"]
)

PREDICTION_ACCURACY = Gauge(
    "model_accuracy",
    "Current model accuracy"
)

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total predictions made",
    ["model_version"]
)

# Histogram 버킷을 SLA 티어에 맞춤 — 백분위수 알림(p50, p95, p99) 활성화
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
)

class PrometheusMonitor:
    """Prometheus 메트릭 업데이트"""

    def __init__(self, port: int = 9090):
        start_http_server(port)

    def update_drift_metrics(self, drift_result: dict):
        """드리프트 메트릭 업데이트"""
        for feature, is_drifted in drift_result.get("columns", {}).items():
            DRIFT_SCORE.labels(feature=feature).set(
                1 if is_drifted else 0
            )

    def update_performance_metrics(self, metrics: dict):
        """성능 메트릭 업데이트"""
        PREDICTION_ACCURACY.set(metrics.get("accuracy", 0))

    def record_prediction(self, model_version: str, latency: float):
        """예측 기록"""
        PREDICTIONS_TOTAL.labels(model_version=model_version).inc()
        PREDICTION_LATENCY.observe(latency)

# 사용
prom_monitor = PrometheusMonitor(port=9090)
prom_monitor.update_drift_metrics(drift_result)
prom_monitor.update_performance_metrics(performance_metrics)
```

---

## 5. 알림 설정

### 5.1 Slack 알림

```python
"""
Slack 알림 연동
"""

import requests
import json

class SlackAlerter:
    """Slack 알림 발송"""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "warning"
    ):
        """알림 발송"""
        color = {
            "info": "#36a64f",
            "warning": "#ff9800",
            "critical": "#ff0000"
        }.get(severity, "#808080")

        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": title,
                    "text": message,
                    "fields": [
                        {
                            "title": "Severity",
                            "value": severity.upper(),
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": datetime.now().isoformat(),
                            "short": True
                        }
                    ]
                }
            ]
        }

        response = requests.post(
            self.webhook_url,
            json=payload
        )

        return response.status_code == 200

    def send_drift_alert(self, drift_result: dict):
        """드리프트 알림"""
        if drift_result["dataset_drift"]:
            drifted_cols = drift_result.get("number_of_drifted_columns", 0)
            drift_share = drift_result.get("drift_share", 0)

            # 심각도 에스컬레이션: 50% 이상 피처 드리프트는 업스트림 데이터 파이프라인 장애 가능성
            self.send_alert(
                title="Data Drift Detected",
                message=f"{drifted_cols} features drifted ({drift_share:.1%})",
                severity="critical" if drift_share > 0.5 else "warning"
            )

# 사용
alerter = SlackAlerter(webhook_url="https://hooks.slack.com/services/...")
alerter.send_drift_alert(drift_result)
```

### 5.2 자동 재학습 트리거

```python
"""
자동 재학습 트리거
"""

class RetrainingTrigger:
    """자동 재학습 트리거"""

    def __init__(
        self,
        drift_threshold: float = 0.3,
        performance_threshold: float = 0.85,
        cooldown_hours: int = 24
    ):
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        # 쿨다운(cooldown)으로 드리프트가 지속적이지만 데이터가 안정화되지 않았을 때 재학습 루프 방지
        self.cooldown_hours = cooldown_hours
        self.last_retrain = None

    def should_retrain(
        self,
        drift_score: float,
        performance: float
    ) -> tuple[bool, str]:
        """재학습 필요 여부 판단"""
        # 쿨다운 체크
        if self.last_retrain:
            hours_since = (datetime.now() - self.last_retrain).total_seconds() / 3600
            if hours_since < self.cooldown_hours:
                return False, f"In cooldown period ({hours_since:.1f}h)"

        # 드리프트 먼저 확인: 성능 저하 전에 감지하는 선제적 신호
        if drift_score > self.drift_threshold:
            return True, f"High drift score: {drift_score:.2f}"

        # 성능 확인은 두 번째: 레이블이 있을 때 사용하는 반응적 신호
        if performance < self.performance_threshold:
            return True, f"Low performance: {performance:.4f}"

        return False, "No retraining needed"

    def trigger_retraining(self, reason: str):
        """재학습 트리거"""
        self.last_retrain = datetime.now()

        # 파이프라인 트리거 (예: Airflow, Kubeflow)
        # trigger_training_pipeline()

        print(f"Retraining triggered: {reason}")
        return True

# 사용
trigger = RetrainingTrigger(
    drift_threshold=0.3,
    performance_threshold=0.85
)

should_retrain, reason = trigger.should_retrain(
    drift_score=0.35,
    performance=0.82
)

if should_retrain:
    trigger.trigger_retraining(reason)
```

---

## 연습 문제

### 문제 1: 드리프트 감지
합성 데이터로 데이터 드리프트를 시뮬레이션하고 감지하세요.

### 문제 2: Evidently 리포트
실제 데이터셋에 대해 완전한 Evidently 리포트를 생성하세요.

### 문제 3: 알림 시스템
드리프트 감지 시 자동으로 알림을 보내는 시스템을 구축하세요.

---

## 요약

| 드리프트 유형 | 설명 | 감지 방법 |
|--------------|------|----------|
| Data Drift | 입력 분포 변화 | PSI, KS Test |
| Concept Drift | 입출력 관계 변화 | 성능 모니터링 |
| Label Drift | 타겟 분포 변화 | 클래스 분포 비교 |

---

## 참고 자료

- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [Data Drift Detection](https://www.evidentlyai.com/blog/machine-learning-monitoring-data-and-concept-drift)
- [Prometheus Monitoring](https://prometheus.io/docs/)
