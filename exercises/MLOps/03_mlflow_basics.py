"""
Exercise Solutions: MLflow Basics
===========================================
Lesson 03 from MLOps topic.

Exercises
---------
1. Basic Experiment Tracking — Log parameters, metrics, and artifacts for a
   Titanic dataset classification experiment (simulated without real MLflow).
2. Hyperparameter Comparison — Run multiple configurations and compare results,
   selecting the best model based on a primary metric.
"""

import math
import random
import json
import os
import tempfile
from datetime import datetime


# ============================================================
# Simulated MLflow (pure Python, no real MLflow dependency)
# ============================================================

class SimulatedMLflow:
    """A minimal simulation of MLflow's tracking API.

    This captures the core concepts:
    - Experiments group related runs
    - Runs log parameters, metrics, artifacts, and tags
    - Metrics can be logged at each step (for training curves)
    """

    def __init__(self):
        self.experiments = {}
        self.active_run = None
        self.run_counter = 0

    def set_experiment(self, name):
        if name not in self.experiments:
            self.experiments[name] = {"runs": [], "name": name}
        self._current_experiment = name

    def start_run(self, run_name=None):
        self.run_counter += 1
        run = {
            "run_id": f"run_{self.run_counter:04d}",
            "run_name": run_name or f"run_{self.run_counter}",
            "parameters": {},
            "metrics": {},
            "metric_history": {},
            "artifacts": [],
            "tags": {},
            "start_time": datetime.now().isoformat(),
            "status": "RUNNING",
        }
        self.active_run = run
        return run

    def log_param(self, key, value):
        self.active_run["parameters"][key] = value

    def log_params(self, params):
        self.active_run["parameters"].update(params)

    def log_metric(self, key, value, step=None):
        self.active_run["metrics"][key] = value
        if key not in self.active_run["metric_history"]:
            self.active_run["metric_history"][key] = []
        self.active_run["metric_history"][key].append({
            "step": step, "value": value
        })

    def log_artifact(self, path):
        self.active_run["artifacts"].append(path)

    def set_tag(self, key, value):
        self.active_run["tags"][key] = value

    def end_run(self):
        self.active_run["status"] = "FINISHED"
        self.active_run["end_time"] = datetime.now().isoformat()
        exp_name = self._current_experiment
        self.experiments[exp_name]["runs"].append(self.active_run)
        run = self.active_run
        self.active_run = None
        return run

    def search_runs(self, experiment_name, order_by=None):
        runs = self.experiments.get(experiment_name, {}).get("runs", [])
        if order_by:
            metric_name = order_by.replace("metrics.", "").replace(" DESC", "").strip()
            descending = "DESC" in order_by
            runs = sorted(
                runs,
                key=lambda r: r["metrics"].get(metric_name, 0),
                reverse=descending,
            )
        return runs


# Simulated dataset — Titanic-like features
def generate_titanic_data(n_samples=500, seed=42):
    """Generate a simulated Titanic-like dataset.

    Features: pclass, age, fare, sex_male, embarked_S, embarked_C, sibsp, parch
    Target: survived (binary)
    """
    random.seed(seed)
    data = []
    for _ in range(n_samples):
        pclass = random.choice([1, 2, 3])
        age = max(1, random.gauss(30, 14))
        fare = max(0, random.gauss(35, 50) * (4 - pclass) / 3)
        sex_male = random.choice([0, 1])
        embarked_s = random.choice([0, 1])
        embarked_c = 1 - embarked_s if random.random() < 0.3 else 0
        sibsp = random.choice([0, 0, 0, 1, 1, 2])
        parch = random.choice([0, 0, 0, 1, 1, 2])

        # Survival probability influenced by class, sex, age
        logit = (
            1.5
            - 0.5 * pclass
            + 1.2 * (1 - sex_male)
            - 0.02 * age
            + 0.005 * fare
            - 0.3 * sibsp
        )
        prob = 1 / (1 + math.exp(-logit))
        survived = 1 if random.random() < prob else 0

        data.append({
            "features": [pclass, age, fare, sex_male, embarked_s, embarked_c, sibsp, parch],
            "label": survived,
        })
    return data


def train_test_split_sim(data, test_ratio=0.2, seed=42):
    """Split data into train and test sets."""
    random.seed(seed)
    shuffled = data[:]
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def train_logistic_regression(train_data, lr=0.01, epochs=100, regularization=0.01):
    """Train a simple logistic regression from scratch.

    Returns weights, bias, and per-epoch loss history.
    """
    n_features = len(train_data[0]["features"])
    weights = [random.gauss(0, 0.01) for _ in range(n_features)]
    bias = 0.0
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0
        for sample in train_data:
            x = sample["features"]
            y = sample["label"]

            # Forward pass
            z = sum(w * xi for w, xi in zip(weights, x)) + bias
            z = max(-500, min(500, z))  # Clip for numerical stability
            pred = 1 / (1 + math.exp(-z))

            # Binary cross-entropy loss
            eps = 1e-7
            loss = -(y * math.log(pred + eps) + (1 - y) * math.log(1 - pred + eps))
            total_loss += loss

            # Gradient descent
            error = pred - y
            for j in range(n_features):
                weights[j] -= lr * (error * x[j] + regularization * weights[j])
            bias -= lr * error

        avg_loss = total_loss / len(train_data)
        loss_history.append(avg_loss)

    return weights, bias, loss_history


def evaluate_model(weights, bias, test_data, threshold=0.5):
    """Evaluate model and return metrics."""
    tp = fp = tn = fn = 0
    for sample in test_data:
        x = sample["features"]
        y = sample["label"]
        z = sum(w * xi for w, xi in zip(weights, x)) + bias
        z = max(-500, min(500, z))
        pred_prob = 1 / (1 + math.exp(-z))
        pred = 1 if pred_prob >= threshold else 0

        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 0:
            tn += 1
        else:
            fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# ============================================================
# Exercise 1: Basic Experiment Tracking
# ============================================================

def exercise_1_basic_tracking():
    """Log parameters, metrics, and artifacts for a Titanic classification experiment.

    This demonstrates the core MLflow workflow:
    1. Create/set an experiment
    2. Start a run
    3. Log parameters (hyperparameters, data info)
    4. Train the model, logging metrics at each epoch
    5. Log final evaluation metrics
    6. Save and log artifacts (model weights, confusion matrix)
    7. End the run
    """
    mlflow = SimulatedMLflow()
    mlflow.set_experiment("titanic-survival-prediction")

    # --- Generate and split data ---
    data = generate_titanic_data(n_samples=500)
    train_data, test_data = train_test_split_sim(data)

    # --- Start a tracked run ---
    mlflow.start_run(run_name="logistic_regression_baseline")

    # Log data parameters
    mlflow.log_param("dataset", "titanic_simulated")
    mlflow.log_param("n_samples", len(data))
    mlflow.log_param("train_size", len(train_data))
    mlflow.log_param("test_size", len(test_data))
    mlflow.log_param("n_features", 8)

    # Log model hyperparameters
    mlflow.log_params({
        "model_type": "logistic_regression",
        "learning_rate": 0.01,
        "epochs": 100,
        "regularization": 0.01,
        "threshold": 0.5,
    })

    # Tags for organization
    mlflow.set_tag("author", "mlops_student")
    mlflow.set_tag("stage", "experimentation")
    mlflow.set_tag("framework", "pure_python")

    # --- Train with epoch-level metric logging ---
    print("Training logistic regression...")
    weights, bias, loss_history = train_logistic_regression(
        train_data, lr=0.01, epochs=100, regularization=0.01
    )

    # Log loss at each epoch (simulates mlflow.log_metric with step parameter)
    for epoch, loss in enumerate(loss_history):
        mlflow.log_metric("train_loss", loss, step=epoch)
    mlflow.log_metric("final_train_loss", loss_history[-1])

    # --- Evaluate and log final metrics ---
    metrics = evaluate_model(weights, bias, test_data)
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            mlflow.log_metric(metric_name, metric_value)

    # --- Save artifacts ---
    with tempfile.TemporaryDirectory() as tmpdir:
        # Model weights artifact
        model_path = os.path.join(tmpdir, "model_weights.json")
        with open(model_path, "w") as f:
            json.dump({"weights": weights, "bias": bias}, f, indent=2)
        mlflow.log_artifact(model_path)

        # Confusion matrix artifact
        cm_path = os.path.join(tmpdir, "confusion_matrix.json")
        with open(cm_path, "w") as f:
            json.dump({
                "tp": metrics["tp"], "fp": metrics["fp"],
                "tn": metrics["tn"], "fn": metrics["fn"],
            }, f, indent=2)
        mlflow.log_artifact(cm_path)

    run = mlflow.end_run()

    # --- Display results ---
    print(f"\nRun: {run['run_name']} ({run['run_id']})")
    print(f"Status: {run['status']}")
    print(f"\nParameters:")
    for k, v in run["parameters"].items():
        print(f"  {k}: {v}")
    print(f"\nMetrics:")
    for k, v in sorted(run["metrics"].items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    print(f"\nArtifacts: {len(run['artifacts'])} files logged")
    print(f"Tags: {run['tags']}")
    print(f"\nTraining curve (first 5, last 5 epochs):")
    for entry in run["metric_history"]["train_loss"][:5]:
        print(f"  Epoch {entry['step']:3d}: loss = {entry['value']:.4f}")
    print("  ...")
    for entry in run["metric_history"]["train_loss"][-5:]:
        print(f"  Epoch {entry['step']:3d}: loss = {entry['value']:.4f}")

    return run


# ============================================================
# Exercise 2: Hyperparameter Comparison
# ============================================================

def exercise_2_hyperparameter_comparison():
    """Run multiple configurations and select the best model.

    This demonstrates:
    - Running multiple experiments with different hyperparameters
    - Comparing runs across a common experiment
    - Selecting the best model based on a primary metric
    - Logging sufficient metadata for reproducibility
    """
    mlflow = SimulatedMLflow()
    mlflow.set_experiment("titanic-hyperparam-search")

    data = generate_titanic_data(n_samples=500)
    train_data, test_data = train_test_split_sim(data)

    # --- Define hyperparameter grid ---
    configs = [
        {"lr": 0.001, "epochs": 200, "reg": 0.001, "threshold": 0.5},
        {"lr": 0.01,  "epochs": 100, "reg": 0.01,  "threshold": 0.5},
        {"lr": 0.01,  "epochs": 200, "reg": 0.01,  "threshold": 0.5},
        {"lr": 0.05,  "epochs": 100, "reg": 0.001, "threshold": 0.5},
        {"lr": 0.01,  "epochs": 100, "reg": 0.1,   "threshold": 0.5},
        {"lr": 0.01,  "epochs": 100, "reg": 0.01,  "threshold": 0.4},
        {"lr": 0.01,  "epochs": 100, "reg": 0.01,  "threshold": 0.6},
    ]

    print("Hyperparameter Search")
    print("=" * 60)
    print(f"Total configurations: {len(configs)}")
    print(f"Primary metric: f1 (maximize)")
    print()

    # --- Run each configuration ---
    for i, config in enumerate(configs, 1):
        run_name = f"config_{i:02d}_lr{config['lr']}_ep{config['epochs']}"
        mlflow.start_run(run_name=run_name)

        mlflow.log_params({
            "learning_rate": config["lr"],
            "epochs": config["epochs"],
            "regularization": config["reg"],
            "threshold": config["threshold"],
        })

        weights, bias, loss_history = train_logistic_regression(
            train_data,
            lr=config["lr"],
            epochs=config["epochs"],
            regularization=config["reg"],
        )

        metrics = evaluate_model(weights, bias, test_data, threshold=config["threshold"])
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                mlflow.log_metric(metric_name, metric_value)
        mlflow.log_metric("final_loss", loss_history[-1])

        mlflow.end_run()

    # --- Compare all runs ---
    print(f"{'Run':<40s} {'Accuracy':>10s} {'Precision':>10s} "
          f"{'Recall':>10s} {'F1':>10s} {'Loss':>10s}")
    print("-" * 90)

    runs = mlflow.search_runs("titanic-hyperparam-search", order_by="metrics.f1 DESC")
    for run in runs:
        m = run["metrics"]
        print(f"{run['run_name']:<40s} "
              f"{m.get('accuracy', 0):>10.4f} "
              f"{m.get('precision', 0):>10.4f} "
              f"{m.get('recall', 0):>10.4f} "
              f"{m.get('f1', 0):>10.4f} "
              f"{m.get('final_loss', 0):>10.4f}")

    # --- Select best model ---
    best_run = runs[0]
    print()
    print(f"Best model: {best_run['run_name']}")
    print(f"  F1 Score: {best_run['metrics']['f1']:.4f}")
    print(f"  Parameters:")
    for k, v in best_run["parameters"].items():
        print(f"    {k}: {v}")

    # --- Analysis: impact of each hyperparameter ---
    print("\nHyperparameter Impact Analysis:")
    print("-" * 40)

    # Group by learning rate
    lr_groups = {}
    for run in runs:
        lr = run["parameters"]["learning_rate"]
        if lr not in lr_groups:
            lr_groups[lr] = []
        lr_groups[lr].append(run["metrics"]["f1"])

    print("  Learning Rate → Mean F1:")
    for lr in sorted(lr_groups.keys()):
        mean_f1 = sum(lr_groups[lr]) / len(lr_groups[lr])
        print(f"    lr={lr}: {mean_f1:.4f} (n={len(lr_groups[lr])})")

    return best_run


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Exercise 1: Basic Experiment Tracking")
    print("=" * 60)
    exercise_1_basic_tracking()

    print("\n\n")
    print("Exercise 2: Hyperparameter Comparison")
    print("=" * 60)
    exercise_2_hyperparameter_comparison()
