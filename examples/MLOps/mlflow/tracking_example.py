"""
MLflow Tracking Example
=======================

Example of experiment tracking using MLflow.

How to run:
    # Start MLflow server
    mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

    # Run the script
    python tracking_example.py
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# MLflow configuration
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "iris-classification-demo"


def setup_mlflow():
    """Initialize MLflow"""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow Tracking URI: {TRACKING_URI}")
    print(f"Experiment: {EXPERIMENT_NAME}")


def load_data():
    """Load and split data"""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target
    )
    return X_train, X_test, y_train, y_test, iris.target_names


def calculate_metrics(y_true, y_pred):
    """Calculate metrics"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro")
    }


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Visualize confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return fig


def train_and_log(model, model_name, params, X_train, X_test, y_train, y_test, class_names):
    """Train model and log to MLflow"""
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # Train model
        model.fit(X_train, y_train)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())

        # Predict and evaluate
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log confusion matrix
        fig = plot_confusion_matrix(y_test, y_pred, class_names)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # Feature importance (if applicable)
        if hasattr(model, "feature_importances_"):
            fig, ax = plt.subplots(figsize=(10, 6))
            importance = model.feature_importances_
            feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
            indices = np.argsort(importance)[::-1]
            ax.bar(range(len(importance)), importance[indices])
            ax.set_xticks(range(len(importance)))
            ax.set_xticklabels([feature_names[i] for i in indices], rotation=45)
            ax.set_title("Feature Importance")
            mlflow.log_figure(fig, "feature_importance.png")
            plt.close(fig)

        # Save model
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)

        # Add tags
        mlflow.set_tag("validated", "true")
        mlflow.set_tag("dataset", "iris")

        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_macro']:.4f}")
        print(f"  CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return mlflow.active_run().info.run_id


def main():
    """Main execution function"""
    # MLflow setup
    setup_mlflow()

    # Load data
    X_train, X_test, y_train, y_test, class_names = load_data()

    # Define models
    models = [
        (
            "RandomForest",
            RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            {"n_estimators": 100, "max_depth": 5}
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
            {"n_estimators": 100, "max_depth": 3}
        ),
        (
            "LogisticRegression",
            LogisticRegression(max_iter=200, random_state=42),
            {"max_iter": 200}
        )
    ]

    # Train and log models
    run_ids = []
    for model_name, model, params in models:
        run_id = train_and_log(
            model, model_name, params,
            X_train, X_test, y_train, y_test,
            class_names
        )
        run_ids.append((model_name, run_id))

    # Print results
    print("\n" + "=" * 50)
    print("Experiment complete!")
    print(f"View results in MLflow UI: {TRACKING_URI}")
    print("\nRegistered runs:")
    for name, run_id in run_ids:
        print(f"  - {name}: {run_id}")


if __name__ == "__main__":
    main()
