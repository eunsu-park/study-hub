"""
Drift Detection Example
=======================

Example of data drift detection using Evidently AI.

How to run:
    pip install evidently pandas numpy scikit-learn
    python drift_detection.py
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from scipy import stats
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings("ignore")

# Evidently import (optional)
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
    from evidently.test_suite import TestSuite
    from evidently.test_preset import DataDriftTestPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("Evidently is not installed. Using basic statistical methods.")


# ============================================================
# Basic Statistical Drift Detection
# ============================================================

class StatisticalDriftDetector:
    """Drift detection using statistical methods"""

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def ks_test(self, reference: np.ndarray, current: np.ndarray) -> Dict[str, Any]:
        """Kolmogorov-Smirnov test"""
        statistic, p_value = stats.ks_2samp(reference, current)
        return {
            "test": "ks",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": p_value < self.significance_level
        }

    def psi(self, reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """Population Stability Index"""
        # Generate histograms
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)

        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)

        # Convert to proportions
        ref_pct = (ref_counts + 1) / (len(reference) + n_bins)
        cur_pct = (cur_counts + 1) / (len(current) + n_bins)

        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def wasserstein_distance(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Wasserstein distance"""
        return float(stats.wasserstein_distance(reference, current))

    def detect_column_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        column: str
    ) -> Dict[str, Any]:
        """Detect drift in a single column"""
        ref_col = reference[column].dropna().values
        cur_col = current[column].dropna().values

        ks_result = self.ks_test(ref_col, cur_col)
        psi_value = self.psi(ref_col, cur_col)
        wasserstein = self.wasserstein_distance(ref_col, cur_col)

        # PSI interpretation
        if psi_value < 0.1:
            psi_status = "no_drift"
        elif psi_value < 0.2:
            psi_status = "slight_drift"
        else:
            psi_status = "significant_drift"

        return {
            "column": column,
            "ks_test": ks_result,
            "psi": {
                "value": psi_value,
                "status": psi_status
            },
            "wasserstein_distance": wasserstein,
            "drift_detected": ks_result["drift_detected"] or psi_value >= 0.2
        }

    def detect_dataset_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        numerical_columns: list
    ) -> Dict[str, Any]:
        """Detect drift across the entire dataset"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "columns": {},
            "summary": {
                "total_columns": len(numerical_columns),
                "drifted_columns": 0,
                "drift_detected": False
            }
        }

        for col in numerical_columns:
            if col in reference.columns and col in current.columns:
                col_result = self.detect_column_drift(reference, current, col)
                results["columns"][col] = col_result
                if col_result["drift_detected"]:
                    results["summary"]["drifted_columns"] += 1

        drift_share = results["summary"]["drifted_columns"] / results["summary"]["total_columns"]
        results["summary"]["drift_share"] = drift_share
        results["summary"]["drift_detected"] = drift_share > 0.5

        return results


# ============================================================
# Evidently-Based Drift Detection
# ============================================================

class EvidentlyDriftDetector:
    """Drift detection using Evidently AI"""

    def __init__(self):
        if not EVIDENTLY_AVAILABLE:
            raise ImportError("Evidently is not installed")

    def create_report(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        column_mapping: ColumnMapping = None
    ) -> Report:
        """Generate drift report"""
        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftPreset()
        ])

        report.run(
            reference_data=reference,
            current_data=current,
            column_mapping=column_mapping
        )

        return report

    def run_tests(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        column_mapping: ColumnMapping = None
    ) -> TestSuite:
        """Run drift tests"""
        test_suite = TestSuite(tests=[
            DataDriftTestPreset()
        ])

        test_suite.run(
            reference_data=reference,
            current_data=current,
            column_mapping=column_mapping
        )

        return test_suite

    def get_drift_summary(self, report: Report) -> Dict[str, Any]:
        """Extract drift summary from report"""
        result = report.as_dict()

        # Extract DatasetDriftMetric results
        for metric in result.get("metrics", []):
            if "DatasetDriftMetric" in str(metric.get("metric", "")):
                drift_result = metric.get("result", {})
                return {
                    "dataset_drift": drift_result.get("dataset_drift", False),
                    "drift_share": drift_result.get("drift_share", 0),
                    "number_of_columns": drift_result.get("number_of_columns", 0),
                    "number_of_drifted_columns": drift_result.get("number_of_drifted_columns", 0)
                }

        return {"error": "Could not extract drift summary"}


# ============================================================
# Drift Monitoring System
# ============================================================

class DriftMonitor:
    """Drift monitoring system"""

    def __init__(
        self,
        reference_data: pd.DataFrame,
        numerical_columns: list,
        alert_threshold: float = 0.3
    ):
        self.reference_data = reference_data
        self.numerical_columns = numerical_columns
        self.alert_threshold = alert_threshold
        self.detector = StatisticalDriftDetector()
        self.history = []

    def check(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Check for drift"""
        result = self.detector.detect_dataset_drift(
            self.reference_data,
            current_data,
            self.numerical_columns
        )

        # Add to history
        self.history.append({
            "timestamp": result["timestamp"],
            "drift_share": result["summary"]["drift_share"],
            "drift_detected": result["summary"]["drift_detected"]
        })

        # Generate alerts
        result["alerts"] = self._generate_alerts(result)

        return result

    def _generate_alerts(self, result: Dict) -> list:
        """Generate alerts"""
        alerts = []

        if result["summary"]["drift_detected"]:
            alerts.append({
                "level": "critical",
                "message": f"Dataset drift detected: {result['summary']['drift_share']:.1%} of columns drifted"
            })

        for col, col_result in result["columns"].items():
            if col_result["drift_detected"]:
                psi = col_result["psi"]["value"]
                if psi >= 0.25:
                    alerts.append({
                        "level": "warning",
                        "message": f"Significant drift in '{col}': PSI={psi:.3f}"
                    })

        return alerts

    def get_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """Analyze drift trend"""
        if len(self.history) < 2:
            return {"message": "Not enough data for trend analysis"}

        recent = self.history[-window_size:]
        drift_shares = [h["drift_share"] for h in recent]

        return {
            "window_size": len(recent),
            "avg_drift_share": np.mean(drift_shares),
            "max_drift_share": max(drift_shares),
            "drift_count": sum(1 for h in recent if h["drift_detected"]),
            "trend": "increasing" if len(drift_shares) > 1 and drift_shares[-1] > drift_shares[0] else "stable"
        }


# ============================================================
# Example Execution
# ============================================================

def generate_sample_data(n_samples: int = 1000, drift: bool = False) -> pd.DataFrame:
    """Generate sample data"""
    np.random.seed(42 if not drift else 123)

    data = {
        "feature_1": np.random.normal(0, 1, n_samples),
        "feature_2": np.random.normal(5, 2, n_samples),
        "feature_3": np.random.exponential(2, n_samples),
        "feature_4": np.random.uniform(0, 10, n_samples)
    }

    if drift:
        # Add drift to some features
        data["feature_1"] = np.random.normal(0.5, 1.2, n_samples)  # Mean and variance shift
        data["feature_3"] = np.random.exponential(3, n_samples)    # Distribution change

    return pd.DataFrame(data)


def main():
    """Main execution function"""
    print("="*60)
    print("Drift Detection Example")
    print("="*60)

    # 1. Generate data
    print("\n[1] Generating data...")
    reference_data = generate_sample_data(1000, drift=False)
    current_data_no_drift = generate_sample_data(500, drift=False)
    current_data_with_drift = generate_sample_data(500, drift=True)

    print(f"  Reference data: {len(reference_data)} samples")
    print(f"  Current data (no drift): {len(current_data_no_drift)} samples")
    print(f"  Current data (with drift): {len(current_data_with_drift)} samples")

    # 2. Basic statistical drift detection
    print("\n[2] Statistical drift detection...")
    detector = StatisticalDriftDetector()
    numerical_cols = ["feature_1", "feature_2", "feature_3", "feature_4"]

    # Data without drift
    print("\n  --- Data without drift ---")
    result_no_drift = detector.detect_dataset_drift(
        reference_data, current_data_no_drift, numerical_cols
    )
    print(f"  Drift detected: {result_no_drift['summary']['drift_detected']}")
    print(f"  Drift share: {result_no_drift['summary']['drift_share']:.1%}")

    # Data with drift
    print("\n  --- Data with drift ---")
    result_with_drift = detector.detect_dataset_drift(
        reference_data, current_data_with_drift, numerical_cols
    )
    print(f"  Drift detected: {result_with_drift['summary']['drift_detected']}")
    print(f"  Drift share: {result_with_drift['summary']['drift_share']:.1%}")

    # Per-column details
    print("\n  Per-column details:")
    for col, col_result in result_with_drift["columns"].items():
        drift_status = "DRIFT" if col_result["drift_detected"] else "OK"
        psi = col_result["psi"]["value"]
        print(f"    {col}: PSI={psi:.4f} [{drift_status}]")

    # 3. Evidently-based detection (if installed)
    if EVIDENTLY_AVAILABLE:
        print("\n[3] Evidently drift detection...")
        evidently_detector = EvidentlyDriftDetector()

        report = evidently_detector.create_report(
            reference_data, current_data_with_drift
        )

        summary = evidently_detector.get_drift_summary(report)
        print(f"  Dataset Drift: {summary.get('dataset_drift', 'N/A')}")
        print(f"  Drift Share: {summary.get('drift_share', 0):.1%}")
        print(f"  Drifted Columns: {summary.get('number_of_drifted_columns', 0)}/{summary.get('number_of_columns', 0)}")

        # Save HTML report
        report.save_html("drift_report.html")
        print("\n  HTML report saved: drift_report.html")
    else:
        print("\n[3] Evidently installation required (pip install evidently)")

    # 4. Monitoring system simulation
    print("\n[4] Monitoring system simulation...")
    monitor = DriftMonitor(
        reference_data=reference_data,
        numerical_columns=numerical_cols
    )

    # Check with data from multiple time points
    for i in range(5):
        # Gradual drift over time
        drift_factor = i * 0.1
        test_data = reference_data.copy()
        test_data["feature_1"] = test_data["feature_1"] + drift_factor
        test_data["feature_3"] = test_data["feature_3"] * (1 + drift_factor)

        result = monitor.check(test_data.sample(500))
        print(f"\n  Time point {i+1}:")
        print(f"    Drift detected: {result['summary']['drift_detected']}")
        print(f"    Drift share: {result['summary']['drift_share']:.1%}")
        if result["alerts"]:
            for alert in result["alerts"]:
                print(f"    [{alert['level'].upper()}] {alert['message']}")

    # Trend analysis
    trend = monitor.get_trend()
    print(f"\n  Trend analysis:")
    print(f"    Average drift share: {trend['avg_drift_share']:.1%}")
    print(f"    Drift detection count: {trend['drift_count']}")
    print(f"    Trend: {trend['trend']}")

    print("\n" + "="*60)
    print("Example complete!")


if __name__ == "__main__":
    main()
