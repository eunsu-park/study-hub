# MLOps Overview

## Learning Objectives

After completing this lesson, you will be able to:

1. Define MLOps and explain how it addresses the gap between machine learning model development and reliable production deployment
2. Describe the four core principles of MLOps — reproducibility, automation, monitoring, and versioning — and explain why each is necessary
3. Identify the three maturity levels of MLOps and determine which level is appropriate for a given organization or project
4. Design an end-to-end ML pipeline that covers data validation, feature engineering, training, evaluation, and deployment
5. Explain the key differences between DevOps and MLOps and describe the unique challenges that data and model management introduce

---

## 1. What is MLOps?

MLOps (Machine Learning Operations) is a practical methodology for automating and efficiently managing the development, deployment, and operation of machine learning models.

```
Problems with Traditional ML Projects:
- Models only work in notebooks
- Experiments cannot be reproduced
- Difficult to detect performance degradation after deployment
- Manual retraining required

Goals of MLOps:
- Automated ML pipelines
- Reproducible experiments
- Continuous model monitoring
- Automated retraining and deployment
```

### 1.1 Core Principles of MLOps

```python
"""
Core Principles of MLOps
"""

# 1. Reproducibility
# Without this, debugging a production failure requires guessing which data/code version was used;
# pinning all three (data, code, seed) makes any past run exactly reconstructable.
experiment_config = {
    "data_version": "v1.2.0",
    "code_version": "git:abc123",
    "random_seed": 42,
    "hyperparameters": {"lr": 0.001, "epochs": 100}
}

# 2. Automation
# Manual hand-offs between stages introduce human error and make the pipeline non-repeatable;
# a defined stage list is the contract that every run follows the same sequence.
pipeline_stages = [
    "data_validation",
    "feature_engineering",
    "model_training",
    "model_evaluation",
    "model_deployment"
]

# 3. Monitoring
# Models silently degrade as the real world drifts away from training data;
# these three metrics together catch degradation at the model, latency, and data layers independently.
monitoring_metrics = {
    "model_accuracy": 0.95,
    "prediction_latency_p99": "50ms",
    "data_drift_score": 0.02
}

# 4. Versioning
# Data, model, and config must be versioned together — a model artifact alone is meaningless
# without knowing which data it was trained on and which config was active at deployment.
versions = {
    "data": "s3://bucket/data/v2.0/",
    "model": "models/classifier_v3.2.1",
    "config": "configs/production.yaml"
}
```

---

## 2. DevOps vs MLOps

### 2.1 Traditional DevOps

```
Developer → Write Code → Build → Test → Deploy → Monitor
                   ↑                        │
                   └────────── Feedback ────┘
```

### 2.2 Additional Elements in MLOps

```
Data Scientist → Data → Feature Engineering → Model Training → Validation → Deploy → Monitor
                 ↑                                                          │
                 │            ← Detect Data Drift ←──────────────────────┘
                 │            ← Detect Model Performance Degradation ←────┘
                 └─────────────── Retrain Trigger ────────────────────────┘
```

### 2.3 Key Differences

```python
"""
DevOps vs MLOps Differences
"""

# DevOps: Code-centric
devops_artifacts = {
    "source": "application_code",
    "build": "docker_image",
    "test": "unit_tests, integration_tests",
    "deploy": "kubernetes_manifests"
}

# MLOps: Data + Code + Model
mlops_artifacts = {
    "data": "training_data, validation_data",
    "features": "feature_definitions, feature_store",
    "code": "training_scripts, serving_code",
    "model": "model_weights, model_metadata",
    "experiments": "hyperparameters, metrics, artifacts"
}

# MLOps-specific Challenges
mlops_challenges = [
    "Data quality management",
    "Feature consistency maintenance",
    "Model version control",
    "A/B testing",
    "Drift detection",
    "Model interpretability"
]
```

| Category | DevOps | MLOps |
|------|--------|-------|
| Main Artifacts | Application code | Data + Model + Code |
| Testing | Unit/Integration tests | + Data validation, Model validation |
| Deployment Unit | Container/Service | Model + Serving infrastructure |
| Monitoring | System metrics | + Model performance, Data drift |
| Rollback | Deploy previous version | Model version rollback + Data considerations |

---

## 3. MLOps Maturity Levels

### 3.1 Google's MLOps Maturity Model

```
Level 0: Manual Process
├── Experimentation in Jupyter notebooks
├── Manual model deployment
├── No monitoring
└── No retraining triggers

Level 1: ML Pipeline Automation
├── Automated training pipeline
├── Experiment tracking
├── Model registry
└── Basic monitoring

Level 2: CI/CD Pipeline
├── Automated testing (code + data + model)
├── Continuous Training (CT)
├── Automated redeployment
└── Complete monitoring and alerting
```

### 3.2 Characteristics by Maturity Level

```python
"""
Implementation by MLOps Maturity Level
"""

# Level 0: Manual ML
class Level0_ManualML:
    """
    - Data scientists experiment in notebooks
    - Engineers manually deploy models
    - Infrequent deployment (quarterly/annual)
    """
    def train(self):
        # Run in notebook
        model = train_model(data)
        model.save("model.pkl")
        # Manually copy to server

# Level 1: ML Pipeline
class Level1_Pipeline:
    """
    - Automated training pipeline
    - Experiment tracking (MLflow)
    - Model registry
    """
    def train_pipeline(self):
        with mlflow.start_run():
            data = load_data()
            model = train_model(data)
            mlflow.log_metrics(evaluate(model))
            mlflow.sklearn.log_model(model, "model")

# Level 2: CI/CD/CT
class Level2_CICDCT:
    """
    - Automated testing on code changes
    - Automated retraining on data changes
    - Automated rollback on performance degradation
    """
    def continuous_training(self):
        # CT (Continuous Training) means the pipeline re-executes automatically when
        # triggered by data events or drift detection — not by a human pressing "run".
        # This is what separates Level 2 from Level 1's scheduled or manual retraining.
        if detect_drift() or new_data_available():
            trigger_training_pipeline()
        if model_performance_degraded():
            rollback_to_previous_version()
```

### 3.3 Maturity Checklist

```yaml
# mlops_maturity_checklist.yaml
level_0:
  - Notebook-based experimentation
  - Manual model deployment
  - No documentation

level_1:
  - Automated training pipeline
  - Experiment tracking system (MLflow/W&B)
  - Model version control
  - Basic monitoring

level_2:
  - CI/CD for ML
  - Automated testing (data/model)
  - Continuous Training
  - Drift detection and alerting
  - A/B testing infrastructure
  - Feature store
```

---

## 4. MLOps Tool Ecosystem

### 4.1 Tools by Category

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MLOps Tool Ecosystem                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [Experiment Tracking]  [Pipeline]           [Feature Store]       │
│  - MLflow               - Kubeflow           - Feast               │
│  - Weights & Biases     - Airflow            - Tecton              │
│  - Neptune              - Prefect            - Hopsworks           │
│  - Comet ML             - Dagster                                  │
│                                                                     │
│  [Model Registry]       [Serving]            [Monitoring]          │
│  - MLflow Registry      - TorchServe         - Evidently           │
│  - Vertex AI            - Triton             - Grafana             │
│  - SageMaker            - TFServing          - Prometheus          │
│  - Neptune              - BentoML            - WhyLabs             │
│                         - Seldon                                   │
│                                                                     │
│  [Data Versioning]      [Labeling]           [Infrastructure]      │
│  - DVC                  - Label Studio       - Kubernetes          │
│  - Delta Lake           - Labelbox           - Docker              │
│  - LakeFS               - Prodigy            - Terraform           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Major Tool Comparison

```python
"""
MLOps Tool Comparison
"""

# Experiment tracking tools
experiment_tracking = {
    "MLflow": {
        "type": "Open source",
        "features": ["Experiment tracking", "Model registry", "Serving"],
        "deployment": "self-hosted / managed",
        "best_for": "Enterprise, customization"
    },
    "Weights & Biases": {
        "type": "Commercial (free tier available)",
        "features": ["Experiment tracking", "Hyperparameter tuning", "Artifacts"],
        "deployment": "SaaS / self-hosted",
        "best_for": "Deep learning, collaboration, visualization"
    },
    "Neptune": {
        "type": "Commercial (free tier available)",
        "features": ["Experiment tracking", "Metadata management"],
        "deployment": "SaaS",
        "best_for": "Large-scale experiment management"
    }
}

# Serving tools
serving_tools = {
    "TorchServe": {
        "supported": ["PyTorch"],
        "features": ["REST/gRPC", "Batch inference", "A/B testing"],
        "best_for": "PyTorch models"
    },
    "Triton": {
        "supported": ["PyTorch", "TensorFlow", "ONNX", "TensorRT"],
        "features": ["Multi-model", "GPU optimization", "Dynamic batching"],
        "best_for": "High-performance inference, multi-framework"
    },
    "TFServing": {
        "supported": ["TensorFlow"],
        "features": ["REST/gRPC", "Version management"],
        "best_for": "TensorFlow models"
    }
}
```

### 4.3 Cloud Managed Services

```python
"""
Cloud MLOps Platforms
"""

cloud_platforms = {
    "AWS SageMaker": {
        "components": [
            "SageMaker Studio",      # Development environment
            "SageMaker Pipelines",   # Pipeline
            "Model Registry",         # Model management
            "SageMaker Endpoints",   # Serving
            "Model Monitor"          # Monitoring
        ]
    },
    "Google Vertex AI": {
        "components": [
            "Workbench",             # Development environment
            "Vertex Pipelines",      # Pipeline
            "Model Registry",         # Model management
            "Prediction",            # Serving
            "Model Monitoring"       # Monitoring
        ]
    },
    "Azure ML": {
        "components": [
            "Azure ML Studio",       # Development environment
            "Azure Pipelines",       # Pipeline
            "Model Registry",         # Model management
            "Managed Endpoints",     # Serving
            "Data Drift"             # Monitoring
        ]
    }
}
```

---

## 5. MLOps Architecture Patterns

### 5.1 Basic Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Basic MLOps Architecture                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────┐     ┌─────────────┐     ┌─────────────┐           │
│   │  Data   │────▶│   Feature   │────▶│   Model     │           │
│   │  Lake   │     │   Store     │     │  Training   │           │
│   └─────────┘     └─────────────┘     └──────┬──────┘           │
│                                               │                  │
│                                               ▼                  │
│   ┌─────────┐     ┌─────────────┐     ┌──────────────┐          │
│   │ Monitor │◀────│   Model     │◀────│    Model     │          │
│   │  /Alert │     │   Serving   │     │   Registry   │          │
│   └────┬────┘     └─────────────┘     └──────────────┘          │
│        │                                                         │
│        └────────── Retrain Trigger ─────────────────▶ Training  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 Code Example

```python
"""
Basic MLOps Pipeline Structure
"""

from typing import Dict, Any

class MLOpsPipeline:
    """Basic MLOps Pipeline"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiment_tracker = None
        self.model_registry = None
        self.feature_store = None

    def data_ingestion(self):
        """Data collection and validation"""
        # Validation is a separate step from ingestion because bad data silently corrupts models;
        # catching schema errors and outliers here prevents expensive downstream retraining.
        raw_data = self.load_data(self.config["data_source"])
        validated_data = self.validate_data(raw_data)
        return validated_data

    def feature_engineering(self, data):
        """Feature engineering"""
        # Reading from a feature store (not computing inline) ensures training and serving
        # use the exact same feature logic, eliminating training-serving skew.
        features = self.feature_store.get_features(
            entity_ids=data["entity_ids"],
            feature_list=self.config["features"]
        )
        return features

    def train(self, features):
        """Model training"""
        # The experiment tracker context wraps training so every artifact — metrics,
        # parameters, the model file — is atomically associated with one run ID.
        with self.experiment_tracker.start_run():
            model = self.train_model(features)
            metrics = self.evaluate(model)
            self.experiment_tracker.log_metrics(metrics)
            return model

    def register(self, model):
        """Model registration"""
        # Registering to "staging" first enforces a human or automated quality gate
        # before any model can reach production; skipping this step risks deploying
        # a regression silently.
        if self.passes_quality_gate(model):
            self.model_registry.register(
                model=model,
                stage="staging"
            )

    def deploy(self, model_version: str):
        """Model deployment"""
        model = self.model_registry.load(model_version)
        self.serving_endpoint.update(model)

    def monitor(self):
        """Model monitoring"""
        # Closing the feedback loop: drift detection here is what enables Level 2 CT —
        # the pipeline retriggers itself without human intervention.
        if self.detect_drift():
            self.trigger_retraining()
```

---

## 6. Getting Started

### 6.1 First MLOps Project

```python
"""
Getting Started with MLOps: Experiment Tracking with MLflow
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris-classification")

# Prepare data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# The context manager ensures the run is closed cleanly even if training raises an exception;
# a run left open in RUNNING state cannot be compared in the UI until it is ended.
with mlflow.start_run(run_name="random-forest-v1"):
    # Log hyperparameters before training so the run is identifiable even if it fails mid-run.
    params = {"n_estimators": 100, "max_depth": 5}
    mlflow.log_params(params)

    # Train model
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate and log metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # Save model
    mlflow.sklearn.log_model(model, "model")

    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")
```

### 6.2 Local Environment Setup

```bash
# Start MLflow server
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000

# Check in browser
# http://localhost:5000
```

---

## Exercises

### Exercise 1: MLOps Maturity Assessment
Evaluate your team's current ML process and determine the maturity level.

### Exercise 2: Tool Selection
Choose appropriate MLOps tools for the following situations:
- Small team, using PyTorch, budget constraints
- Large team, multi-framework, high performance requirements

---

## Summary

| Concept | Description |
|------|------|
| MLOps | Automation of ML model development, deployment, and operation |
| DevOps vs MLOps | MLOps adds data and model management |
| Maturity Level 0 | Manual process, notebook-based |
| Maturity Level 1 | Automated pipeline, experiment tracking |
| Maturity Level 2 | CI/CD/CT, continuous training |
| Core Tools | MLflow, W&B, Kubeflow, Feast, Triton |

---

## References

- [Google MLOps Whitepaper](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [MLOps Principles](https://ml-ops.org/)
- [Made With ML - MLOps](https://madewithml.com/courses/mlops/)
