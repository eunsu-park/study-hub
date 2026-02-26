# Weights & Biases (W&B)

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the core features of Weights & Biases — Experiments, Sweeps, Artifacts, Tables, and Reports — and explain when to use each
2. Instrument a training loop with `wandb.init`, `wandb.log`, and `wandb.finish` to track metrics, hyperparameters, and model checkpoints
3. Configure and run W&B Sweeps for automated hyperparameter optimization using grid, random, and Bayesian search strategies
4. Use W&B Artifacts to version datasets and models, enabling reproducible experiment lineage tracking
5. Create W&B Reports that document and share experiment results and visualizations with collaborators

---

## 1. W&B Overview

Weights & Biases is a platform for ML experiment tracking, hyperparameter tuning, and model management.

### 1.1 Core Features

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Weights & Biases Features                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│   │ Experiments │    │   Sweeps    │    │  Artifacts  │            │
│   │             │    │             │    │             │            │
│   │ - Experiment│    │ - Hyper     │    │ - Datasets  │            │
│   │   tracking  │    │   parameter │    │ - Models    │            │
│   │ - Metrics   │    │   tuning    │    │ - Version   │            │
│   │ - Visualization│  │             │    │   control   │            │
│   └─────────────┘    └─────────────┘    └─────────────┘            │
│                                                                     │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│   │   Tables    │    │   Reports   │    │   Models    │            │
│   │             │    │             │    │             │            │
│   │ - Data      │    │ - Documentation│  │ - Model     │            │
│   │   visualization│  │ - Sharing   │    │   registry  │            │
│   └─────────────┘    └─────────────┘    └─────────────┘            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Installation and Setup

```bash
# Install
pip install wandb

# Login
wandb login
# Enter API key (https://wandb.ai/authorize)

# Set as environment variable
export WANDB_API_KEY=your-api-key
```

```python
# Login from Python
import wandb
wandb.login(key="your-api-key")
```

---

## 2. Basic Experiment Tracking

### 2.1 First Experiment

```python
"""
Basic W&B Usage
"""

import wandb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Prepare data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Initialize W&B — all subsequent wandb.log() calls are associated with this run;
# config is stored as a snapshot, so changing params later won't affect the logged record
wandb.init(
    project="iris-classification",    # Project name
    name="random-forest-baseline",    # Run name
    config={                          # Hyperparameters
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
    },
    tags=["baseline", "random-forest"],
    notes="Initial baseline experiment"
)

# Access config via wandb.config — ensures hyperparams come from the single source of truth,
# enabling Sweeps to override values without changing code
config = wandb.config

# Train model
model = RandomForestClassifier(
    n_estimators=config.n_estimators,
    max_depth=config.max_depth,
    random_state=config.random_state
)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log metrics
wandb.log({
    "accuracy": accuracy,
    "test_size": len(X_test),
    "train_size": len(X_train)
})

# Finish run
wandb.finish()
```

### 2.2 Logging Training Process

```python
"""
Real-time Training Process Logging
"""

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Initialize
wandb.init(project="pytorch-training")

# Define model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=wandb.config.get("lr", 0.001))

# wandb.watch hooks into autograd to log gradients and parameter norms —
# helps diagnose vanishing/exploding gradients without manual instrumentation
wandb.watch(model, criterion, log="all", log_freq=100)

# Training loop
for epoch in range(wandb.config.get("epochs", 10)):
    model.train()
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Batch-level logging (optional)
        if batch_idx % 100 == 0:
            wandb.log({
                "batch_loss": loss.item(),
                "epoch": epoch,
                "batch": batch_idx
            })

    # Epoch-level logging
    avg_loss = train_loss / len(train_loader)
    val_accuracy = evaluate(model, val_loader)

    wandb.log({
        "epoch": epoch,
        "train_loss": avg_loss,
        "val_accuracy": val_accuracy
    })

    # Save only when improving — avoids storing redundant checkpoints;
    # wandb.save uploads to cloud so checkpoints survive even if the machine is preempted
    if val_accuracy > best_accuracy:
        torch.save(model.state_dict(), "best_model.pth")
        wandb.save("best_model.pth")
        best_accuracy = val_accuracy

wandb.finish()
```

### 2.3 Logging Various Data Types

```python
"""
Logging Various Data Types
"""

import wandb
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

wandb.init(project="data-logging-demo")

# 1. Log images
images = wandb.Image(
    np.random.rand(100, 100, 3),
    caption="Random Image"
)
wandb.log({"random_image": images})

# PIL image
pil_image = Image.open("sample.png")
wandb.log({"pil_image": wandb.Image(pil_image)})

# Multiple images
wandb.log({
    "examples": [wandb.Image(img, caption=f"Sample {i}")
                 for i, img in enumerate(image_batch[:10])]
})

# 2. Log plots
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
ax.set_title("Training Curve")
wandb.log({"plot": wandb.Image(fig)})
plt.close()

# Or use plotly
import plotly.express as px
fig = px.scatter(x=[1, 2, 3], y=[1, 4, 9])
wandb.log({"plotly_chart": fig})

# 3. Histogram
wandb.log({"predictions": wandb.Histogram(predictions)})

# 4. Table
columns = ["id", "image", "prediction", "label"]
data = [
    [i, wandb.Image(img), pred, label]
    for i, (img, pred, label) in enumerate(zip(images, preds, labels))
]
table = wandb.Table(columns=columns, data=data)
wandb.log({"predictions_table": table})

# 5. Confusion Matrix
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        y_true=y_true,
        preds=y_pred,
        class_names=class_names
    )
})

# 6. ROC Curve
wandb.log({
    "roc_curve": wandb.plot.roc_curve(
        y_true, y_scores, labels=class_names
    )
})

# 7. PR Curve
wandb.log({
    "pr_curve": wandb.plot.pr_curve(
        y_true, y_scores, labels=class_names
    )
})

wandb.finish()
```

---

## 3. Sweeps (Hyperparameter Tuning)

### 3.1 Sweep Configuration

```python
"""
W&B Sweeps Configuration
"""

import wandb

# Sweep configuration
sweep_config = {
    "name": "hyperparam-sweep",
    # Bayesian method models metric as a function of params — converges faster
    # than random/grid by exploring promising regions more densely
    "method": "bayes",
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-1
        },
        "batch_size": {
            "values": [16, 32, 64, 128]
        },
        "epochs": {
            "value": 50  # Fixed value
        },
        "optimizer": {
            "values": ["adam", "sgd", "rmsprop"]
        },
        "hidden_dim": {
            "distribution": "int_uniform",
            "min": 32,
            "max": 256
        },
        "dropout": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.5
        }
    },
    # Hyperband early termination kills underperforming runs early —
    # saves compute by stopping runs that are unlikely to beat the current best
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 5,
        "eta": 3
    }
}

# Create sweep
sweep_id = wandb.sweep(sweep_config, project="my-project")
print(f"Sweep ID: {sweep_id}")
```

### 3.2 Running Sweep Agent

```python
"""
Sweep Training Function
"""

import wandb
import torch

def train_sweep():
    """Training function to run in sweep"""
    # Initialize W&B (sweep provides config)
    wandb.init()
    config = wandb.config

    # Create model
    model = create_model(
        hidden_dim=config.hidden_dim,
        dropout=config.dropout
    )

    # Setup optimizer
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Training
    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_accuracy = evaluate(model, val_loader)

        wandb.log({
            "train_loss": train_loss,
            "val_accuracy": val_accuracy,
            "epoch": epoch
        })

    wandb.finish()

# Run sweep
wandb.agent(
    sweep_id,
    function=train_sweep,
    count=50  # Maximum number of runs
)
```

### 3.3 Running Sweep from CLI

```bash
# Create sweep.yaml file
# Start sweep
wandb sweep sweep.yaml

# Run agent (can run in parallel on multiple machines)
wandb agent username/project/sweep_id
```

```yaml
# sweep.yaml
name: hyperparameter-sweep
method: bayes
metric:
  name: val_accuracy
  goal: maximize
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 0.00001
    max: 0.1
  batch_size:
    values: [16, 32, 64]
  hidden_dim:
    distribution: int_uniform
    min: 64
    max: 512
```

---

## 4. Artifacts

### 4.1 Dataset Version Control

```python
"""
Managing Datasets with W&B Artifacts
"""

import wandb

# Create and upload artifact
wandb.init(project="dataset-versioning")

# Create dataset artifact
dataset_artifact = wandb.Artifact(
    name="mnist-dataset",
    type="dataset",
    description="MNIST dataset for classification",
    metadata={
        "size": 70000,
        "classes": 10,
        "source": "torchvision"
    }
)

# Add files/directories
dataset_artifact.add_file("data/train.csv")
dataset_artifact.add_dir("data/images/")

# add_reference stores a pointer, not the data — avoids duplicating large datasets
# while still tracking them in the lineage graph
dataset_artifact.add_reference("s3://bucket/large_data/")

# Upload
wandb.log_artifact(dataset_artifact)
wandb.finish()
```

### 4.2 Model Artifacts

```python
"""
Managing Model Artifacts
"""

import wandb
import torch

wandb.init(project="model-artifacts")

# After training...

# Create model artifact
model_artifact = wandb.Artifact(
    name="churn-model",
    type="model",
    description="Customer churn prediction model",
    metadata={
        "accuracy": 0.95,
        "framework": "pytorch",
        "architecture": "MLP"
    }
)

# Save and add model file
torch.save(model.state_dict(), "model.pth")
model_artifact.add_file("model.pth")

# Bundle config alongside the model — ensures inference uses the same hyperparams
model_artifact.add_file("config.yaml")

# Upload
wandb.log_artifact(model_artifact)

# Aliases act as mutable pointers (like git tags) — "production" always points to
# the current serving version, making downstream consumers version-agnostic
wandb.run.link_artifact(model_artifact, "model-registry/churn-model", aliases=["latest", "production"])

wandb.finish()
```

### 4.3 Using Artifacts

```python
"""
Downloading and Using Artifacts
"""

import wandb

wandb.init(project="using-artifacts")

# Download artifact
artifact = wandb.use_artifact("mnist-dataset:latest")  # or :v0, :v1, etc.
artifact_dir = artifact.download()

print(f"Downloaded to: {artifact_dir}")

# Direct access to artifact files
with artifact.file("train.csv") as f:
    df = pd.read_csv(f)

# Dependency tracking (this run uses this artifact)
# use_artifact() handles this automatically

wandb.finish()
```

### 4.4 Artifact Lineage

```python
"""
Tracking Artifact Lineage
"""

import wandb

# Data → Training → Model lineage
wandb.init(project="lineage-demo")

# 1. Input artifact (dataset)
dataset = wandb.use_artifact("processed-data:latest")

# 2. Perform training
# ...

# 3. Output artifact (model)
model_artifact = wandb.Artifact("trained-model", type="model")
model_artifact.add_file("model.pth")
wandb.log_artifact(model_artifact)

# Can view entire lineage graph in W&B UI
# Dataset → (training run) → Model

wandb.finish()
```

---

## 5. Comparison with MLflow

### 5.1 Feature Comparison

```python
"""
MLflow vs W&B Comparison
"""

comparison = {
    "Experiment Tracking": {
        "MLflow": "Open source, self-hosted",
        "W&B": "SaaS-based, free tier available"
    },
    "Visualization": {
        "MLflow": "Basic visualization",
        "W&B": "Rich visualization, real-time updates"
    },
    "Collaboration": {
        "MLflow": "Limited",
        "W&B": "Team features, report sharing"
    },
    "Hyperparameter Tuning": {
        "MLflow": "Requires external tools (Optuna, etc.)",
        "W&B": "Built-in Sweeps"
    },
    "Model Registry": {
        "MLflow": "Full functionality",
        "W&B": "Model Registry (recently added)"
    },
    "Deployment": {
        "MLflow": "MLflow Serving",
        "W&B": "No direct support (integrate other tools)"
    },
    "Cost": {
        "MLflow": "Free (infrastructure costs only)",
        "W&B": "Free tier + paid plans"
    }
}
```

### 5.2 Using Together

```python
"""
Using MLflow and W&B Together
"""

import mlflow
import wandb

# Dual tracking: MLflow for model registry/serving, W&B for visualization/collaboration
# — each platform's strength complements the other
wandb.init(project="dual-tracking")
mlflow.set_experiment("dual-tracking")

with mlflow.start_run():
    # Common configuration
    params = {"lr": 0.001, "epochs": 100}

    # Log parameters to both
    mlflow.log_params(params)
    wandb.config.update(params)

    # Training loop
    for epoch in range(params["epochs"]):
        loss = train_one_epoch()
        accuracy = evaluate()

        # Log metrics to both
        mlflow.log_metrics({"loss": loss, "accuracy": accuracy}, step=epoch)
        wandb.log({"loss": loss, "accuracy": accuracy, "epoch": epoch})

    # Save model
    mlflow.sklearn.log_model(model, "model")
    wandb.save("model.pkl")

wandb.finish()
```

---

## 6. Advanced Features

### 6.1 Team Collaboration

```python
"""
Team Project Setup
"""

import wandb

# Log to team project
wandb.init(
    entity="team-name",           # Team name
    project="shared-project",     # Project name
    group="experiment-group",     # Experiment group (group related experiments)
    job_type="training"           # Job type
)
```

### 6.2 Report Creation

```python
"""
W&B Reports API
"""

import wandb

# Reports are typically created in UI, but also available via API
api = wandb.Api()

# Query all runs in project
runs = api.runs("username/project")

for run in runs:
    print(f"Run: {run.name}")
    print(f"  Config: {run.config}")
    print(f"  Summary: {run.summary}")
    print(f"  History: {run.history().shape}")
```

### 6.3 Alert Configuration

```python
"""
W&B Alerts
"""

import wandb

wandb.init(project="alerting-demo")

# Trigger alerts during training
for epoch in range(100):
    accuracy = train_and_evaluate()

    if accuracy > 0.95:
        wandb.alert(
            title="High Accuracy Achieved!",
            text=f"Model achieved {accuracy:.2%} accuracy at epoch {epoch}",
            level=wandb.AlertLevel.INFO
        )

    if accuracy < 0.5:
        wandb.alert(
            title="Training Issue",
            text=f"Accuracy dropped to {accuracy:.2%}",
            level=wandb.AlertLevel.WARN
        )

    wandb.log({"accuracy": accuracy, "epoch": epoch})

wandb.finish()
```

---

## Exercises

### Exercise 1: Basic Experiment Tracking
Train a CNN model on the MNIST dataset and track experiments with W&B.

### Exercise 2: Run Sweeps
Execute a Bayesian optimization sweep for 3 or more hyperparameters.

### Exercise 3: Artifacts
Save datasets and models as artifacts and verify lineage.

---

## Summary

| Feature | W&B | MLflow |
|------|-----|--------|
| Experiment Tracking | wandb.log() | mlflow.log_metrics() |
| Hyperparameter Tuning | Sweeps | External tools |
| Data/Model Versioning | Artifacts | Model Registry |
| Visualization | Rich dashboards | Basic UI |
| Collaboration | Teams, Reports | Limited |
| Hosting | SaaS / Self-hosted | Self-hosted |

---

## References

- [W&B Documentation](https://docs.wandb.ai/)
- [W&B Sweeps](https://docs.wandb.ai/guides/sweeps)
- [W&B Artifacts](https://docs.wandb.ai/guides/artifacts)
- [W&B Reports](https://docs.wandb.ai/guides/reports)
