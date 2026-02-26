# 10. Drift Detection & Monitoring

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between data drift, concept drift, and label drift, and explain how each type degrades model performance in production
2. Implement statistical drift detection tests (KS test, PSI, chi-squared) to detect distribution shifts in input features
3. Build a model monitoring pipeline that collects prediction logs, computes drift metrics, and triggers alerts when thresholds are exceeded
4. Use Evidently AI or similar tools to generate automated data quality and drift monitoring reports
5. Design a retraining trigger strategy that responds to detected drift while minimizing unnecessary retraining overhead

---

## 1. Drift Concepts

Drift is the phenomenon where data or model performance changes over time.

### 1.1 Drift Types

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Drift Types                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │                  Data Drift                              │      │
│   │                                                          │      │
│   │   Input data distribution changes                        │      │
│   │   P(X)_train ≠ P(X)_production                          │      │
│   │                                                          │      │
│   │   Example: Customer age distribution changes             │      │
│   └─────────────────────────────────────────────────────────┘      │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │                  Concept Drift                           │      │
│   │                                                          │      │
│   │   Relationship between input and output changes          │      │
│   │   P(Y|X)_train ≠ P(Y|X)_production                      │      │
│   │                                                          │      │
│   │   Example: Economic changes alter churn patterns         │      │
│   └─────────────────────────────────────────────────────────┘      │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │                  Label Drift                             │      │
│   │                                                          │      │
│   │   Target variable distribution changes                   │      │
│   │   P(Y)_train ≠ P(Y)_production                          │      │
│   │                                                          │      │
│   │   Example: Fraud rate increases                          │      │
│   └─────────────────────────────────────────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Drift Causes

```python
"""
Drift cause analysis
"""

drift_causes = {
    "data_collection_changes": {
        "examples": ["Sensor malfunction", "Data source change", "Logging bug"],
        "detection": "Schema validation, statistical monitoring"
    },
    "external_environment_changes": {
        "examples": ["Seasonality", "Competitor actions", "Economic conditions", "Regulatory changes"],
        "detection": "Domain knowledge, long-term trend analysis"
    },
    "user_behavior_changes": {
        "examples": ["New user influx", "Usage pattern changes", "Generational shifts"],
        "detection": "Cohort analysis, A/B testing"
    },
    "feature_engineering_errors": {
        "examples": ["Upstream data changes", "Preprocessing bugs"],
        "detection": "Feature store validation, unit tests"
    }
}
```

---

## 2. Drift Detection Techniques

### 2.1 Statistical Tests

```python
"""
Drift detection statistical methods
"""

import numpy as np
from scipy import stats
from typing import Tuple

def kolmogorov_smirnov_test(
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float = 0.05
) -> Tuple[float, bool]:
    """KS test - Test difference between two distributions"""
    # Non-parametric: no distribution assumption needed, unlike t-test or chi-squared
    statistic, p_value = stats.ks_2samp(reference, current)
    is_drift = p_value < threshold
    return statistic, is_drift

def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10
) -> float:
    """PSI - Population Stability Index"""
    # Binning captures shape changes that summary stats (mean/std) miss
    bins = np.histogram_bin_edges(reference, bins=n_bins)
    ref_hist, _ = np.histogram(reference, bins=bins, density=True)
    cur_hist, _ = np.histogram(current, bins=bins, density=True)

    # Prevent log(0) which would produce -inf and corrupt the PSI value
    ref_hist = np.where(ref_hist == 0, 0.0001, ref_hist)
    cur_hist = np.where(cur_hist == 0, 0.0001, cur_hist)

    # Calculate PSI
    psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))

    return psi

def wasserstein_distance(
    reference: np.ndarray,
    current: np.ndarray
) -> float:
    """Wasserstein distance (Earth Mover's Distance)"""
    return stats.wasserstein_distance(reference, current)

def jensen_shannon_divergence(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10
) -> float:
    """Jensen-Shannon Divergence"""
    # Symmetric and bounded [0,1] — easier to threshold than KL divergence
    from scipy.spatial.distance import jensenshannon

    bins = np.histogram_bin_edges(reference, bins=n_bins)
    ref_hist, _ = np.histogram(reference, bins=bins, density=True)
    cur_hist, _ = np.histogram(current, bins=bins, density=True)

    return jensenshannon(ref_hist, cur_hist)

# Interpretation guidelines
psi_thresholds = {
    "PSI < 0.1": "No change",
    "0.1 <= PSI < 0.2": "Slight change",
    "PSI >= 0.2": "Significant change - retraining required"
}
```

### 2.2 Multivariate Drift Detection

```python
"""
Multivariate drift detection
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
    Domain classifier-based drift detection
    - Train classifier to distinguish reference from current data
    - AUC close to 0.5 indicates no drift
    """
    # Catches multivariate drift that per-feature tests miss (e.g., correlation shifts)
    X = np.vstack([reference, current])
    y = np.hstack([
        np.zeros(len(reference)),
        np.ones(len(current))
    ])

    # Cross-validation prevents overfitting to noise — AUC=0.5 means no learnable drift
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    mean_auc = scores.mean()

    # Further from 0.5, higher drift probability
    drift_score = abs(mean_auc - 0.5) * 2
    is_drift = mean_auc > threshold

    return drift_score, is_drift

def multivariate_drift_pca(
    reference: np.ndarray,
    current: np.ndarray,
    n_components: int = 5
) -> float:
    """
    PCA-based multivariate drift detection
    """
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import mahalanobis

    # PCA reduces curse of dimensionality — Mahalanobis fails with more features than samples
    pca = PCA(n_components=n_components)
    ref_pca = pca.fit_transform(reference)
    cur_pca = pca.transform(current)

    # Mean and covariance for each dimension
    ref_mean = np.mean(ref_pca, axis=0)
    cur_mean = np.mean(cur_pca, axis=0)
    ref_cov = np.cov(ref_pca.T)

    # Mahalanobis distance
    try:
        distance = mahalanobis(ref_mean, cur_mean, np.linalg.inv(ref_cov))
    except np.linalg.LinAlgError:
        # Singular covariance (collinear features) — fall back to Euclidean
        distance = np.linalg.norm(ref_mean - cur_mean)

    return distance
```

---

## 3. Evidently AI

### 3.1 Evidently Basics

```python
"""
Evidently AI drift detection
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

# Prepare data
reference_data = pd.read_csv("reference_data.csv")
current_data = pd.read_csv("current_data.csv")

# Explicit mapping ensures Evidently picks the right statistical test per column type
column_mapping = ColumnMapping(
    target="target",
    prediction="prediction",
    numerical_features=["age", "tenure", "monthly_charges"],
    categorical_features=["gender", "contract_type"]
)

# Data drift report
drift_report = Report(metrics=[
    DatasetDriftMetric(),
    DataDriftTable(),
])

drift_report.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=column_mapping
)

# Save HTML report
drift_report.save_html("drift_report.html")

# JSON results (for programming use)
result = drift_report.as_dict()
print(f"Dataset drift detected: {result['metrics'][0]['result']['dataset_drift']}")
```

### 3.2 Detailed Drift Analysis

```python
"""
Evidently detailed analysis
"""

from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric,
    ColumnSummaryMetric,
    ColumnQuantileMetric,
    ColumnValueRangeMetric
)

# Detailed analysis for specific column
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

# Extract results
result = column_report.as_dict()

for metric in result["metrics"]:
    metric_name = metric["metric"]
    metric_result = metric["result"]
    print(f"{metric_name}: {metric_result}")
```

### 3.3 Model Performance Monitoring

```python
"""
Evidently model performance monitoring
"""

from evidently.report import Report
from evidently.metric_preset import ClassificationPreset, RegressionPreset
from evidently.metrics import (
    ClassificationQualityMetric,
    ClassificationClassBalance,
    ClassificationConfusionMatrix
)

# Classification model performance report
classification_report = Report(metrics=[
    ClassificationPreset(),
])

classification_report.run(
    reference_data=reference_data,  # Training data + predictions
    current_data=current_data,       # Current data + predictions
    column_mapping=column_mapping
)

classification_report.save_html("model_performance_report.html")

# Extract results
result = classification_report.as_dict()
performance = result["metrics"][0]["result"]["current"]

print(f"Accuracy: {performance['accuracy']}")
print(f"Precision: {performance['precision']}")
print(f"Recall: {performance['recall']}")
```

### 3.4 Real-time Monitoring

```python
"""
Evidently Test Suite (automated testing)
"""

from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset, DataQualityTestPreset
from evidently.tests import (
    TestColumnDrift,
    TestShareOfMissingValues,
    TestColumnValueRange,
    TestNumberOfRows
)

# Test suite returns pass/fail — integrate into CI/CD to gate deployments on data quality
test_suite = TestSuite(tests=[
    # Preset covers all columns; individual tests add stricter thresholds for critical features
    DataDriftTestPreset(),

    # Individual tests
    TestColumnDrift(column_name="monthly_charges", stattest_threshold=0.1),
    TestShareOfMissingValues(column_name="age", lt=0.05),
    # Domain constraints catch data pipeline bugs (e.g., negative age from encoding errors)
    TestColumnValueRange(column_name="age", left=18, right=100),
    TestNumberOfRows(gte=1000),
])

test_suite.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=column_mapping
)

# Check results
result = test_suite.as_dict()

print(f"Tests passed: {result['summary']['success_tests']}")
print(f"Tests failed: {result['summary']['failed_tests']}")

# Failed test details
for test in result["tests"]:
    if test["status"] == "FAIL":
        print(f"FAILED: {test['name']} - {test['description']}")
```

---

## 4. Monitoring Pipeline

### 4.1 Complete Monitoring System

```python
"""
Production monitoring pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric

class ModelMonitor:
    """ML model monitoring system"""

    def __init__(
        self,
        reference_data: pd.DataFrame,
        column_mapping,
        alert_thresholds: Dict[str, float]
    ):
        # Reference data loaded once at init — avoids re-reading from storage each monitoring cycle
        self.reference_data = reference_data
        self.column_mapping = column_mapping
        self.thresholds = alert_thresholds
        # History enables trend analysis — single-point drift is noise, sustained drift is real
        self.monitoring_history = []

    def check_data_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Check data drift"""
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
        """Check model performance"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics = {
            "accuracy": accuracy_score(actuals, predictions),
            "precision": precision_score(actuals, predictions, average="macro"),
            "recall": recall_score(actuals, predictions, average="macro"),
            "f1_score": f1_score(actuals, predictions, average="macro")
        }

        return metrics

    def generate_alerts(self, drift_result: Dict, performance: Dict) -> List[str]:
        """Generate alerts"""
        alerts = []

        # Data drift alerts
        if drift_result["dataset_drift"]:
            alerts.append(f"DATA_DRIFT: {drift_result['drift_share']:.1%} of features drifted")

        # Performance degradation alerts
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
        """Run full monitoring"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "data_drift": self.check_data_drift(current_data),
            "alerts": []
        }

        # Actuals often arrive delayed (hours/days) — drift detection works without them
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

# Example usage
monitor = ModelMonitor(
    reference_data=reference_data,
    column_mapping=column_mapping,
    alert_thresholds={
        "accuracy": 0.85,
        "f1_score": 0.80
    }
)

# Hourly monitoring
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

### 4.2 Prometheus + Grafana Integration

```python
"""
Expose Prometheus metrics
"""

from prometheus_client import Gauge, Counter, Histogram, start_http_server
import time

# Gauge (not Counter) because drift score can increase or decrease over time
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

# Histogram buckets aligned with SLA tiers — enables percentile alerting (p50, p95, p99)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
)

class PrometheusMonitor:
    """Update Prometheus metrics"""

    def __init__(self, port: int = 9090):
        start_http_server(port)

    def update_drift_metrics(self, drift_result: dict):
        """Update drift metrics"""
        for feature, is_drifted in drift_result.get("columns", {}).items():
            DRIFT_SCORE.labels(feature=feature).set(
                1 if is_drifted else 0
            )

    def update_performance_metrics(self, metrics: dict):
        """Update performance metrics"""
        PREDICTION_ACCURACY.set(metrics.get("accuracy", 0))

    def record_prediction(self, model_version: str, latency: float):
        """Record prediction"""
        PREDICTIONS_TOTAL.labels(model_version=model_version).inc()
        PREDICTION_LATENCY.observe(latency)

# Usage
prom_monitor = PrometheusMonitor(port=9090)
prom_monitor.update_drift_metrics(drift_result)
prom_monitor.update_performance_metrics(performance_metrics)
```

---

## 5. Alert Configuration

### 5.1 Slack Alerts

```python
"""
Slack integration
"""

import requests
import json

class SlackAlerter:
    """Send Slack alerts"""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "warning"
    ):
        """Send alert"""
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
        """Send drift alert"""
        if drift_result["dataset_drift"]:
            drifted_cols = drift_result.get("number_of_drifted_columns", 0)
            drift_share = drift_result.get("drift_share", 0)

            # Severity escalation: >50% features drifted likely means upstream data pipeline broke
            self.send_alert(
                title="Data Drift Detected",
                message=f"{drifted_cols} features drifted ({drift_share:.1%})",
                severity="critical" if drift_share > 0.5 else "warning"
            )

# Usage
alerter = SlackAlerter(webhook_url="https://hooks.slack.com/services/...")
alerter.send_drift_alert(drift_result)
```

### 5.2 Automated Retraining Trigger

```python
"""
Automated retraining trigger
"""

class RetrainingTrigger:
    """Automated retraining trigger"""

    def __init__(
        self,
        drift_threshold: float = 0.3,
        performance_threshold: float = 0.85,
        cooldown_hours: int = 24
    ):
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        # Cooldown prevents retraining loops when drift is persistent but data hasn't stabilized
        self.cooldown_hours = cooldown_hours
        self.last_retrain = None

    def should_retrain(
        self,
        drift_score: float,
        performance: float
    ) -> tuple[bool, str]:
        """Determine if retraining needed"""
        # Cooldown check
        if self.last_retrain:
            hours_since = (datetime.now() - self.last_retrain).total_seconds() / 3600
            if hours_since < self.cooldown_hours:
                return False, f"In cooldown period ({hours_since:.1f}h)"

        # Drift check first: proactive signal before performance degrades
        if drift_score > self.drift_threshold:
            return True, f"High drift score: {drift_score:.2f}"

        # Performance check second: reactive signal when labels are available
        if performance < self.performance_threshold:
            return True, f"Low performance: {performance:.4f}"

        return False, "No retraining needed"

    def trigger_retraining(self, reason: str):
        """Trigger retraining"""
        self.last_retrain = datetime.now()

        # Trigger pipeline (e.g., Airflow, Kubeflow)
        # trigger_training_pipeline()

        print(f"Retraining triggered: {reason}")
        return True

# Usage
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

## Practice Exercises

### Exercise 1: Drift Detection
Simulate and detect data drift using synthetic data.

### Exercise 2: Evidently Reports
Generate complete Evidently reports for a real dataset.

### Exercise 3: Alert System
Build a system that automatically sends alerts when drift is detected.

---

## Summary

| Drift Type | Description | Detection Method |
|-----------|-------------|------------------|
| Data Drift | Input distribution change | PSI, KS Test |
| Concept Drift | Input-output relationship change | Performance monitoring |
| Label Drift | Target distribution change | Class distribution comparison |

---

## References

- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [Data Drift Detection](https://www.evidentlyai.com/blog/machine-learning-monitoring-data-and-concept-drift)
- [Prometheus Monitoring](https://prometheus.io/docs/)
