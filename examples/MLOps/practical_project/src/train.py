"""
Model training with MLflow experiment tracking.

Adapted from MLOps Lesson 12 §4.1.
Trains a RandomForest classifier, logs metrics/params to MLflow,
registers the model, and checks quality gates.

Usage:
    python src/train.py --config configs/training_config.yaml
"""

import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split


class ModelTrainer:
    """Model training with MLflow tracking."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        mlflow.set_tracking_uri(self.config["training"]["tracking_uri"])
        mlflow.set_experiment(self.config["training"]["experiment_name"])

    def prepare_data(self, df: pd.DataFrame):
        """Split into train/validation sets."""
        feature_cols = [
            c for c in df.columns
            if c not in ["user_id", "target", "event_timestamp", "ingestion_timestamp"]
        ]

        # One-hot encode categorical columns
        X = pd.get_dummies(df[feature_cols], drop_first=True)
        y = df["target"]

        return train_test_split(
            X, y,
            test_size=self.config["data"]["validation_split"],
            random_state=42,
            stratify=y,
        )

    def train(self, X_train, y_train, X_val, y_val) -> dict:
        """Train model and log to MLflow."""
        with mlflow.start_run() as run:
            params = self.config["model"]["params"]
            mlflow.log_params(params)
            mlflow.log_param("model_type", self.config["model"]["type"])

            # Train
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            mlflow.log_metric("cv_mean", cv_scores.mean())
            mlflow.log_metric("cv_std", cv_scores.std())

            # Validation metrics
            y_pred = model.predict(X_val)
            metrics = {
                "accuracy": accuracy_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred, average="macro"),
                "recall": recall_score(y_val, y_pred, average="macro"),
                "f1_score": f1_score(y_val, y_pred, average="macro"),
            }
            for name, value in metrics.items():
                mlflow.log_metric(name, value)

            # Log model
            signature = mlflow.models.infer_signature(X_train, y_pred)
            mlflow.sklearn.log_model(
                model, "model",
                signature=signature,
                registered_model_name=self.config["project"]["name"],
            )

            print(f"Run ID: {run.info.run_id}")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            return {"run_id": run.info.run_id, "metrics": metrics, "model": model}

    def validate_quality_gates(self, metrics: dict) -> bool:
        """Check if metrics pass quality gates."""
        gates = self.config["quality_gates"]
        for metric, threshold in gates.items():
            actual = metrics.get(metric, 0)
            status = "PASS" if actual >= threshold else "FAIL"
            print(f"  Gate {metric}: {actual:.4f} >= {threshold} -> {status}")

        return all(metrics.get(m, 0) >= t for m, t in gates.items())


def main():
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--data", default="data/churn.parquet")
    args = parser.parse_args()

    trainer = ModelTrainer(args.config)
    df = pd.read_parquet(args.data)
    X_train, X_val, y_train, y_val = trainer.prepare_data(df)

    result = trainer.train(X_train, y_train, X_val, y_val)

    print("\nQuality Gates:")
    passed = trainer.validate_quality_gates(result["metrics"])
    print(f"\nOverall: {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()
