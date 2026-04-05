"""
Model quality gate tests.

Verifies that a trained model meets minimum performance thresholds.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.generate_churn_data import generate_churn_data


@pytest.fixture
def trained_model():
    """Train a model on synthetic data for testing."""
    df = generate_churn_data(n_rows=2000, seed=42)

    feature_cols = [
        "age", "tenure_months", "total_purchases",
        "avg_purchase_amount", "days_since_last_purchase", "support_tickets",
    ]
    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test


class TestModelQuality:
    """Quality gate tests — model must exceed these thresholds."""

    def test_accuracy_above_threshold(self, trained_model):
        model, X_test, y_test = trained_model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        assert acc >= 0.60, f"Accuracy {acc:.4f} below threshold 0.60"

    def test_precision_above_threshold(self, trained_model):
        model, X_test, y_test = trained_model
        y_pred = model.predict(X_test)
        prec = precision_score(y_test, y_pred, average="macro")
        assert prec >= 0.55, f"Precision {prec:.4f} below threshold 0.55"

    def test_recall_above_threshold(self, trained_model):
        model, X_test, y_test = trained_model
        y_pred = model.predict(X_test)
        rec = recall_score(y_test, y_pred, average="macro")
        assert rec >= 0.55, f"Recall {rec:.4f} below threshold 0.55"

    def test_model_predicts_both_classes(self, trained_model):
        model, X_test, _ = trained_model
        y_pred = model.predict(X_test)
        assert len(set(y_pred)) > 1, "Model predicts only one class"
