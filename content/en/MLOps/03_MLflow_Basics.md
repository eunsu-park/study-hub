# MLflow Basics

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the four core components of MLflow — Tracking, Projects, Models, and Model Registry — and explain the role each plays in the ML lifecycle
2. Implement experiment tracking with MLflow by logging parameters, metrics, and artifacts from a machine learning training run
3. Package ML models using MLflow's model format and serve them locally with the MLflow model server
4. Use the MLflow Model Registry to version models and manage stage transitions from staging to production
5. Set up a centralized MLflow tracking server and integrate it into a team-based ML workflow

---

## 1. MLflow Overview

MLflow is an open-source platform for managing the machine learning lifecycle. It provides integrated support for experiment tracking, model packaging, and deployment.

### 1.1 Four Components of MLflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MLflow Components                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│   │  Tracking   │    │  Projects   │    │   Models    │            │
│   │             │    │             │    │             │            │
│   │ - Experiment│    │ - Reproducible│  │ - Model     │            │
│   │   tracking  │    │   projects  │    │   formats   │            │
│   │ - Metrics   │    │ - Dependency│    │ - Various   │            │
│   │ - Parameters│    │   management│    │   flavors   │            │
│   │ - Artifacts │    │             │    │             │            │
│   └─────────────┘    └─────────────┘    └─────────────┘            │
│                                                                     │
│   ┌─────────────────────────────────────────────────────┐          │
│   │                   Model Registry                     │          │
│   │                                                      │          │
│   │  - Model versioning  - Stage transitions  - Descriptions  │    │
│   │                                                      │          │
│   └─────────────────────────────────────────────────────┘          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Installation and Setup

```bash
# Install MLflow
pip install mlflow

# Additional dependencies (optional)
pip install mlflow[extras]  # All additional features
pip install mlflow[sklearn]  # scikit-learn support
pip install mlflow[pytorch]  # PyTorch support

# Check version
mlflow --version
```

---

## 2. MLflow Tracking

### 2.1 Basic Concepts

```python
"""
MLflow Tracking Basic Concepts
"""

# Key terms
mlflow_concepts = {
    "Experiment": "Group of related runs (e.g., 'churn-prediction')",
    "Run": "Single training execution (includes parameters, metrics, artifacts)",
    "Parameters": "Input settings (learning_rate, epochs, etc.)",
    "Metrics": "Output results (accuracy, loss, etc.)",
    "Artifacts": "Files (models, plots, data, etc.)",
    "Tags": "Metadata about the run"
}
```

### 2.2 First Experiment

```python
"""
Basic MLflow Usage
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Prepare data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Set experiment
mlflow.set_experiment("iris-classification")

# The context manager guarantees mlflow.end_run() is called even if training raises an exception;
# without it, a crashed run is left in RUNNING state and pollutes experiment comparisons.
with mlflow.start_run(run_name="random-forest-baseline"):
    # Log parameters before training so the run is traceable even if training fails;
    # this allows filtering failed runs by their hyperparameters to identify bad configs.
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
    }
    mlflow.log_params(params)

    # 2. Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # 3. Predict
    y_pred = model.predict(X_test)

    # 4. Log metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='macro'),
        "recall": recall_score(y_test, y_pred, average='macro'),
        "f1": f1_score(y_test, y_pred, average='macro')
    }
    mlflow.log_metrics(metrics)

    # 5. Log model
    mlflow.sklearn.log_model(model, "model")

    # 6. Add tags
    mlflow.set_tag("model_type", "RandomForest")
    mlflow.set_tag("developer", "ML Team")

    # Print run information
    run = mlflow.active_run()
    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")
    print(f"Metrics: {metrics}")
```

### 2.3 Logging Parameters and Metrics

```python
"""
Various Logging Methods
"""

import mlflow
import numpy as np

with mlflow.start_run():
    # Single parameter
    mlflow.log_param("learning_rate", 0.001)

    # Multiple parameters
    mlflow.log_params({
        "batch_size": 32,
        "epochs": 100,
        "optimizer": "adam"
    })

    # Single metric
    mlflow.log_metric("accuracy", 0.95)

    # Multiple metrics
    mlflow.log_metrics({
        "precision": 0.93,
        "recall": 0.91,
        "f1": 0.92
    })

    # Step-wise metrics (training curves)
    for epoch in range(100):
        train_loss = 1.0 / (epoch + 1) + np.random.random() * 0.1
        val_loss = 1.0 / (epoch + 1) + np.random.random() * 0.15
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

    # Tags (searchable metadata)
    mlflow.set_tag("data_version", "v2.0")
    mlflow.set_tag("experiment_type", "baseline")

    # Multiple tags
    mlflow.set_tags({
        "feature_set": "full",
        "preprocessing": "standardized"
    })
```

### 2.4 Logging Artifacts

```python
"""
Artifact Logging
"""

import mlflow
import matplotlib.pyplot as plt
import pandas as pd
import json

with mlflow.start_run():
    # 1. Log files
    # Single file
    with open("config.json", "w") as f:
        json.dump({"key": "value"}, f)
    mlflow.log_artifact("config.json")

    # Entire directory
    mlflow.log_artifacts("./outputs", artifact_path="results")

    # 2. Log plots
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_title("Training Curve")
    mlflow.log_figure(fig, "training_curve.png")
    plt.close()

    # 3. Log DataFrame (CSV)
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    df.to_csv("data.csv", index=False)
    mlflow.log_artifact("data.csv")

    # 4. Log dictionary as JSON
    results = {"accuracy": 0.95, "model": "RF"}
    mlflow.log_dict(results, "results.json")

    # 5. Log text
    mlflow.log_text("This is a log message", "log.txt")
```

---

## 3. MLflow UI

### 3.1 Starting the Server

```bash
# Start local server (default)
mlflow ui

# Specify port
mlflow ui --port 5000

# Specify host (allow external access)
mlflow ui --host 0.0.0.0 --port 5000

# Specify backend store
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000
```

### 3.2 Setting Tracking URI

```python
"""
How to Set Tracking URI
"""

import mlflow

# Method 1: Set in code
mlflow.set_tracking_uri("http://localhost:5000")

# Method 2: Environment variable
# export MLFLOW_TRACKING_URI=http://localhost:5000

# Method 3: File-based (default)
mlflow.set_tracking_uri("file:///path/to/mlruns")

# Check current setting
print(mlflow.get_tracking_uri())
```

### 3.3 Using UI Features

```python
"""
Features Available in UI
"""

# 1. Structured logging for experiment comparison
experiments_to_compare = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 10}
]

for params in experiments_to_compare:
    with mlflow.start_run():
        mlflow.log_params(params)
        # Train and evaluate
        accuracy = train_and_evaluate(params)
        mlflow.log_metric("accuracy", accuracy)

# 2. Add searchable tags
with mlflow.start_run():
    mlflow.set_tags({
        "model_type": "RandomForest",
        "feature_version": "v2",
        "data_split": "stratified"
    })

# 3. Search runs (API)
runs = mlflow.search_runs(
    experiment_names=["iris-classification"],
    filter_string="metrics.accuracy > 0.9 and params.max_depth = '5'",
    order_by=["metrics.accuracy DESC"]
)
print(runs[["run_id", "params.n_estimators", "metrics.accuracy"]])
```

---

## 4. Basic Usage Examples

### 4.1 scikit-learn Model

```python
"""
Complete scikit-learn Model Example
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# Set experiment
mlflow.set_experiment("wine-classification")

# Hyperparameter grid
param_grid = [
    {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3},
    {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 5},
    {"n_estimators": 200, "learning_rate": 0.01, "max_depth": 7}
]

for params in param_grid:
    with mlflow.start_run(run_name=f"gb-n{params['n_estimators']}"):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("test_size", 0.2)

        # Create pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GradientBoostingClassifier(**params, random_state=42))
        ])

        # Cross validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())

        # Final training
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Log metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        mlflow.log_metrics({
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred, average='macro'),
            "test_recall": recall_score(y_test, y_pred, average='macro')
        })

        # Confusion Matrix visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close()

        # Feature Importance
        classifier = pipeline.named_steps['classifier']
        fig, ax = plt.subplots(figsize=(10, 6))
        importance = classifier.feature_importances_
        indices = np.argsort(importance)[::-1]
        ax.barh(range(len(importance)), importance[indices])
        ax.set_yticks(range(len(importance)))
        ax.set_yticklabels([wine.feature_names[i] for i in indices])
        ax.set_title('Feature Importance')
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close()

        # Log model
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"Params: {params}")
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### 4.2 PyTorch Model

```python
"""
PyTorch Model MLflow Tracking
"""

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

# Prepare data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Set experiment
mlflow.set_experiment("pytorch-classification")

# Hyperparameters
params = {
    "hidden_dim": 64,
    "learning_rate": 0.001,
    "epochs": 50,
    "batch_size": 32
}

with mlflow.start_run():
    mlflow.log_params(params)

    # Initialize model
    model = SimpleNN(20, params["hidden_dim"], 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    # Training
    model.train()
    for epoch in range(params["epochs"]):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

        # Validation (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                outputs = model(X_test_t)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y_test_t).float().mean().item()
                mlflow.log_metric("val_accuracy", accuracy, step=epoch)
            model.train()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs, 1)
        test_accuracy = (predicted == y_test_t).float().mean().item()

    mlflow.log_metric("test_accuracy", test_accuracy)

    # Log model
    mlflow.pytorch.log_model(model, "model")

    print(f"Test Accuracy: {test_accuracy:.4f}")
```

---

## 5. Autologging

### 5.1 Setting Up Autologging

```python
"""
MLflow Autologging
"""

import mlflow

# Enable autologging for all frameworks
mlflow.autolog()

# Enable for specific frameworks only
mlflow.sklearn.autolog()
mlflow.pytorch.autolog()
mlflow.tensorflow.autolog()
mlflow.xgboost.autolog()
mlflow.lightgbm.autolog()

# Disable autologging
mlflow.autolog(disable=True)
```

### 5.2 Autologging Example

```python
"""
sklearn autologging example
"""

import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Enable autologging
mlflow.sklearn.autolog(
    log_input_examples=True,      # Log input examples
    log_model_signatures=True,    # Log model signatures
    log_models=True,              # Log models
    log_datasets=True,            # Log dataset information
    silent=False                  # Print logging messages
)

# Prepare data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Set experiment (everything is automatically logged)
mlflow.set_experiment("autolog-demo")

# Train model (automatically creates run and logs)
model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

# Automatically logged items:
# - Parameters: n_estimators, max_depth, ...
# - Metrics: training_score, ...
# - Artifacts: model, feature_importance, ...
```

---

## 6. Loading Models and Making Predictions

### 6.1 Loading Saved Models

```python
"""
How to Load Saved Models
"""

import mlflow
import mlflow.sklearn

# Method 1: Load by Run ID
model = mlflow.sklearn.load_model("runs:/RUN_ID/model")

# Method 2: Load by artifact path
model = mlflow.sklearn.load_model("file:///path/to/mlruns/0/run_id/artifacts/model")

# Method 3: Load from Model Registry (covered in next lesson)
# The "Production" stage is a gate in the registry: Staging means "passed automated tests,
# under review"; promoting to Production is the explicit approval that enables this URI to resolve.
# Only one version is active in Production at a time, enabling instant rollback by demoting it.
model = mlflow.sklearn.load_model("models:/MyModel/Production")

# Method 4: Load as pyfunc (framework-agnostic)
model = mlflow.pyfunc.load_model("runs:/RUN_ID/model")

# Make predictions
predictions = model.predict(X_test)
```

### 6.2 Querying Recent Run Results

```python
"""
Querying Experiment Results
"""

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get experiment
experiment = client.get_experiment_by_name("iris-classification")
print(f"Experiment ID: {experiment.experiment_id}")

# Search runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.accuracy DESC"],
    max_results=5
)

for run in runs:
    print(f"Run ID: {run.info.run_id}")
    print(f"  Accuracy: {run.data.metrics.get('accuracy')}")
    print(f"  Params: {run.data.params}")

# Load best performing model
best_run = runs[0]
best_model = mlflow.sklearn.load_model(f"runs:/{best_run.info.run_id}/model")
```

---

## Exercises

### Exercise 1: Basic Experiment Tracking
Use the Titanic dataset to train a survival prediction model and track the experiment with MLflow.

```python
# Hint
import mlflow
from sklearn.datasets import fetch_openml

titanic = fetch_openml("titanic", version=1, as_frame=True)
# After preprocessing, train model
# Log parameters and metrics with mlflow
```

### Exercise 2: Hyperparameter Comparison
Run at least 5 experiments with different hyperparameters and compare them in the MLflow UI.

---

## Summary

| Feature | Method | Description |
|------|--------|------|
| Set experiment | `mlflow.set_experiment()` | Specify experiment group |
| Start run | `mlflow.start_run()` | Start new run |
| Parameters | `mlflow.log_param(s)()` | Log input parameters |
| Metrics | `mlflow.log_metric(s)()` | Log output metrics |
| Artifacts | `mlflow.log_artifact(s)()` | Log files |
| Model | `mlflow.sklearn.log_model()` | Save model |
| Autologging | `mlflow.autolog()` | Enable automatic tracking |

---

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Quickstart](https://mlflow.org/docs/latest/quickstart.html)
