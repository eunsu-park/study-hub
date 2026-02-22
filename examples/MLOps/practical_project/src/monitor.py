"""
Model monitoring with Evidently for data/prediction drift detection.

Adapted from MLOps Lesson 12 §5 (monitoring section).
Compares reference (training) data with current (production) data.

Usage:
    python src/monitor.py [--reference data/churn.parquet] [--current data/current.parquet]
"""

import argparse
from datetime import datetime

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report


class DriftDetector:
    """Detect data and prediction drift using Evidently."""

    def __init__(self, numerical_features=None, categorical_features=None):
        self.numerical_features = numerical_features or [
            "age", "tenure_months", "total_purchases",
            "avg_purchase_amount", "days_since_last_purchase", "support_tickets",
        ]
        self.categorical_features = categorical_features or ["customer_segment"]

        self.column_mapping = ColumnMapping(
            target="target",
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
        )

    def detect(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
    ) -> dict:
        """Run drift detection and return summary."""
        report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
        ])

        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping,
        )

        result = report.as_dict()

        # Extract drift summary
        data_drift = result["metrics"][0]["result"]
        drift_share = data_drift.get("share_of_drifted_columns", 0)
        is_drift = data_drift.get("dataset_drift", False)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "is_drift": is_drift,
            "drift_share": drift_share,
            "n_drifted_columns": data_drift.get("number_of_drifted_columns", 0),
            "n_columns": data_drift.get("number_of_columns", 0),
        }

        return summary

    def save_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        output_path: str = "drift_report.html",
    ):
        """Generate and save an HTML drift report."""
        report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
        ])

        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping,
        )

        report.save_html(output_path)
        print(f"Drift report saved -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Drift detection")
    parser.add_argument("--reference", default="data/churn.parquet")
    parser.add_argument("--current", default="data/churn.parquet",
                        help="In production, this would be recent inference data")
    parser.add_argument("--report", default="drift_report.html")
    args = parser.parse_args()

    reference = pd.read_parquet(args.reference)
    current = pd.read_parquet(args.current)

    detector = DriftDetector()
    summary = detector.detect(reference, current)

    print("Drift Detection Results:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if summary["is_drift"]:
        print("\n  WARNING: Data drift detected! Consider retraining.")

    detector.save_report(reference, current, args.report)


if __name__ == "__main__":
    main()
