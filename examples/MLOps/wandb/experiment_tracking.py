"""
Weights & Biases Experiment Tracking Example
============================================

Example of experiment tracking using W&B.

How to run:
    # Log in to W&B
    wandb login

    # Run the script
    python experiment_tracking.py
"""

import wandb
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# W&B project settings
PROJECT_NAME = "breast-cancer-classification"
ENTITY = None  # Team name (None for personal)


def load_data():
    """Load and split data"""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target
    )
    return X_train, X_test, y_train, y_test, data.feature_names, data.target_names


def calculate_metrics(y_true, y_pred, y_proba=None):
    """Calculate metrics"""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics


def train_with_wandb(model_name, model, params, X_train, X_test, y_train, y_test, feature_names):
    """Train model with W&B experiment tracking"""

    # Initialize W&B
    run = wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY,
        name=model_name,
        config=params,
        tags=["baseline", model_name.lower()],
        notes=f"Training {model_name} on breast cancer dataset"
    )

    # Log additional config
    wandb.config.update({
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": X_train.shape[1]
    })

    # Train model
    model.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    wandb.log({
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std()
    })

    # Learning curve simulation (for some models)
    if hasattr(model, "n_estimators"):
        # Record step-wise performance
        for i in range(1, params.get("n_estimators", 100) + 1, 10):
            partial_model = type(model)(**{**params, "n_estimators": i})
            partial_model.fit(X_train, y_train)
            train_score = partial_model.score(X_train, y_train)
            val_score = partial_model.score(X_test, y_test)
            wandb.log({
                "train_accuracy": train_score,
                "val_accuracy": val_score,
                "n_estimators": i
            })

    # Final predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Log metrics
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    wandb.log(metrics)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()

    # Feature importance (if applicable)
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:15]  # Top 15

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(indices)), importance[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha="right")
        plt.title(f"Feature Importance - {model_name}")
        plt.tight_layout()
        wandb.log({"feature_importance": wandb.Image(plt)})
        plt.close()

        # Also log as table
        importance_data = [
            [feature_names[i], importance[i]]
            for i in indices
        ]
        table = wandb.Table(columns=["feature", "importance"], data=importance_data)
        wandb.log({"feature_importance_table": table})

    # ROC Curve (if probability prediction is available)
    if y_proba is not None:
        wandb.log({
            "roc_curve": wandb.plot.roc_curve(y_test, np.column_stack([1-y_proba, y_proba]))
        })

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    wandb.log({
        "classification_report": report
    })

    # Save model artifact
    artifact = wandb.Artifact(
        name=f"{model_name.lower()}-model",
        type="model",
        description=f"{model_name} trained on breast cancer dataset"
    )
    # In production, save actual model file
    # artifact.add_file("model.pkl")
    wandb.log_artifact(artifact)

    # Print results
    print(f"\n{model_name} results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  ROC AUC: {metrics.get('roc_auc', 'N/A')}")
    print(f"  CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # End run
    wandb.finish()

    return metrics


def hyperparameter_sweep():
    """Hyperparameter sweep example"""

    # Sweep configuration
    sweep_config = {
        "name": "rf-hyperparameter-sweep",
        "method": "bayes",  # random, grid, bayes
        "metric": {
            "name": "val_accuracy",
            "goal": "maximize"
        },
        "parameters": {
            "n_estimators": {
                "values": [50, 100, 150, 200]
            },
            "max_depth": {
                "values": [3, 5, 7, 10, None]
            },
            "min_samples_split": {
                "distribution": "int_uniform",
                "min": 2,
                "max": 20
            },
            "min_samples_leaf": {
                "distribution": "int_uniform",
                "min": 1,
                "max": 10
            }
        }
    }

    def train_sweep():
        """Training function to run in sweep"""
        wandb.init()
        config = wandb.config

        X_train, X_test, y_train, y_test, _, _ = load_data()

        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            random_state=42
        )

        model.fit(X_train, y_train)
        val_accuracy = model.score(X_test, y_test)

        wandb.log({"val_accuracy": val_accuracy})
        wandb.finish()

    # Create and run sweep
    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
    print(f"\nSweep ID: {sweep_id}")
    print("To run the sweep:")
    print(f"  wandb agent {sweep_id}")

    # Run locally (optional)
    # wandb.agent(sweep_id, function=train_sweep, count=20)


def main():
    """Main execution function"""
    print("="*60)
    print("Weights & Biases Experiment Tracking Example")
    print("="*60)

    # Load data
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data()
    print(f"\nDataset:")
    print(f"  Training data: {len(X_train)} samples")
    print(f"  Test data: {len(X_test)} samples")
    print(f"  Number of features: {len(feature_names)}")

    # Define models
    models = [
        (
            "RandomForest",
            RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            {"n_estimators": 100, "max_depth": 10}
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
            {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}
        ),
        (
            "LogisticRegression",
            LogisticRegression(max_iter=1000, random_state=42),
            {"max_iter": 1000, "solver": "lbfgs"}
        )
    ]

    # Train models
    results = {}
    for model_name, model, params in models:
        print(f"\n{'='*40}")
        print(f"Training {model_name}...")
        metrics = train_with_wandb(
            model_name, model, params,
            X_train, X_test, y_train, y_test,
            feature_names
        )
        results[model_name] = metrics

    # Results summary
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"\n{'Model':<20} {'Accuracy':<12} {'F1 Score':<12} {'ROC AUC':<12}")
    print("-"*60)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['accuracy']:<12.4f} {metrics['f1_score']:<12.4f} {metrics.get('roc_auc', 0):<12.4f}")

    print(f"\nView detailed results on the W&B dashboard:")
    print(f"  https://wandb.ai/{ENTITY or 'your-username'}/{PROJECT_NAME}")


if __name__ == "__main__":
    main()
