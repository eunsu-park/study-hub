"""
DVC Pipeline Example — Training Stage
======================================
Demonstrates a DVC-compatible training script that:
- Reads hyperparameters from params.yaml
- Loads data from DVC-tracked files
- Trains a model and saves it
- Outputs metrics in JSON format (DVC metrics)
- Outputs plots in CSV format (DVC plots)

Usage:
    # Run via DVC pipeline
    dvc repro

    # Run standalone
    python src/train.py

    # Run as DVC experiment with modified params
    dvc exp run --set-param train.n_estimators=500
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)


def load_params(params_file="params.yaml"):
    """Load hyperparameters from DVC params file."""
    with open(params_file) as f:
        params = yaml.safe_load(f)
    return params


def load_data(features_dir="data/features"):
    """Load feature-engineered data."""
    X_train = pd.read_csv(f"{features_dir}/X_train.csv")
    y_train = pd.read_csv(f"{features_dir}/y_train.csv").values.ravel()
    X_test = pd.read_csv(f"{features_dir}/X_test.csv")
    y_test = pd.read_csv(f"{features_dir}/y_test.csv").values.ravel()
    return X_train, y_train, X_test, y_test


def create_model(params):
    """Create model based on params."""
    model_type = params.get("model_type", "gradient_boosting")

    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 200),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 6),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            subsample=params.get("subsample", 0.8),
            random_state=42,
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 10),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_metrics(y_test, y_pred, y_proba, output_dir="metrics"):
    """Save evaluation metrics in DVC-compatible JSON format."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "auc": round(roc_auc_score(y_test, y_proba), 4),
        "n_test_samples": len(y_test),
        "positive_rate": round(y_test.mean(), 4),
    }

    with open(f"{output_dir}/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics: {metrics}")
    return metrics


def save_plots(y_test, y_pred, y_proba, feature_names, feature_importances,
               output_dir="plots"):
    """Save plots in CSV format for DVC plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        f"{output_dir}/roc_curve.csv", index=False
    )

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pd.DataFrame({"precision": precision, "recall": recall}).to_csv(
        f"{output_dir}/precision_recall.csv", index=False
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_data = []
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            cm_data.append({"actual": str(i), "predicted": str(j), "count": int(val)})
    pd.DataFrame(cm_data).to_csv(
        f"{output_dir}/confusion_matrix.csv", index=False
    )

    # Feature importance
    if feature_importances is not None:
        imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance": feature_importances,
        }).sort_values("importance", ascending=False)
        imp_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)


def train():
    """Main training function."""
    # Load params
    all_params = load_params()
    train_params = all_params.get("train", {})

    print(f"Training with params: {train_params}")

    # Load data
    X_train, y_train, X_test, y_test = load_data()
    print(f"Data: {X_train.shape[0]} train, {X_test.shape[0]} test, "
          f"{X_train.shape[1]} features")

    # Train
    model = create_model(train_params)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Save model
    Path("models").mkdir(parents=True, exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save metrics
    save_metrics(y_test, y_pred, y_proba)

    # Save plots
    importances = getattr(model, "feature_importances_", None)
    save_plots(y_test, y_pred, y_proba, X_train.columns.tolist(), importances)

    print("Training complete.")


if __name__ == "__main__":
    train()
