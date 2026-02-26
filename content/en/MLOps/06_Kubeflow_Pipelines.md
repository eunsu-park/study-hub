# Kubeflow Pipelines

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the Kubeflow ecosystem components — Pipelines, Katib, Training Operators, KServe, and Notebooks — and explain the role each plays in the ML platform
2. Build Kubeflow Pipeline components and pipelines using the KFP SDK, defining containerized steps as Python functions with the `@component` decorator
3. Implement conditional branching and parallel execution in Kubeflow Pipelines using control flow constructs
4. Configure Katib for automated hyperparameter tuning using Bayesian optimization on a Kubernetes cluster
5. Deploy and manage trained models using KServe for A/B testing and canary rollouts on Kubernetes

---

## 1. Kubeflow Overview

Kubeflow is an open-source platform for building, deploying, and managing ML workflows on Kubernetes.

### 1.1 Kubeflow Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Kubeflow Ecosystem                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│   │  Pipelines  │    │   Katib     │    │  Training   │            │
│   │             │    │             │    │  Operators  │            │
│   │ - Workflow  │    │ - AutoML    │    │             │            │
│   │ - Pipeline  │    │ - Hyper     │    │ - TFJob     │            │
│   │   orchestr  │    │   parameter │    │ - PyTorchJob│            │
│   │   ation     │    │   tuning    │    │ - MXJob     │            │
│   └─────────────┘    └─────────────┘    └─────────────┘            │
│                                                                     │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│   │   KServe    │    │  Notebooks  │    │   Central   │            │
│   │             │    │             │    │  Dashboard  │            │
│   │ - Model     │    │ - Jupyter   │    │             │            │
│   │   serving   │    │   environment│   │ - UI       │            │
│   │ - A/B test  │    │             │    │ - Management│            │
│   │ - Canary    │    │             │    │             │            │
│   └─────────────┘    └─────────────┘    └─────────────┘            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Installation

```bash
# Install KFP SDK
pip install kfp

# Check version
python -c "import kfp; print(kfp.__version__)"

# Install Kubeflow Pipelines cluster (minikube example)
# https://www.kubeflow.org/docs/started/installing-kubeflow/
```

---

## 2. Kubeflow Pipelines SDK

### 2.1 Basic Concepts

```python
"""
KFP Basic Concepts
"""

# Key terms
kfp_concepts = {
    "Pipeline": "DAG (Directed Acyclic Graph) defining ML workflow",
    "Component": "Individual step in pipeline (function or container)",
    "Run": "Single execution of a pipeline",
    "Experiment": "Group of related runs",
    "Artifact": "Data passed between components"
}
```

### 2.2 Simple Pipeline

```python
"""
KFP v2 Basic Pipeline
"""

from kfp import dsl
from kfp import compiler

# Each @dsl.component runs in its own container — provides isolation so
# dependency conflicts between steps (e.g., different sklearn versions) never occur
@dsl.component
def preprocess_data(input_path: str, output_path: dsl.OutputPath(str)):
    """Data preprocessing component"""
    import pandas as pd

    df = pd.read_csv(input_path)
    # Preprocessing logic
    df_processed = df.dropna()
    df_processed.to_csv(output_path, index=False)

@dsl.component
def train_model(
    data_path: str,
    n_estimators: int,
    model_path: dsl.OutputPath(str)
):
    """Model training component"""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X, y)

    joblib.dump(model, model_path)

@dsl.component
def evaluate_model(
    model_path: str,
    test_data_path: str
) -> float:
    """Model evaluation component"""
    import pandas as pd
    from sklearn.metrics import accuracy_score
    import joblib

    model = joblib.load(model_path)
    df = pd.read_csv(test_data_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    return accuracy

# Define pipeline
@dsl.pipeline(
    name="ML Training Pipeline",
    description="A simple ML training pipeline"
)
def ml_pipeline(
    input_data_path: str = "gs://bucket/data.csv",
    n_estimators: int = 100
):
    # Steps are connected through outputs — KFP automatically creates the DAG dependency,
    # so step 2 waits for step 1 to finish without explicit ordering
    preprocess_task = preprocess_data(input_path=input_data_path)

    # Training consumes preprocessing output — artifacts are passed via cloud storage,
    # not in-memory, enabling steps to run on different machines
    train_task = train_model(
        data_path=preprocess_task.outputs["output_path"],
        n_estimators=n_estimators
    )

    # Step 3: Evaluation
    evaluate_task = evaluate_model(
        model_path=train_task.outputs["model_path"],
        test_data_path=preprocess_task.outputs["output_path"]
    )

# Compile to YAML — the compiled spec is portable and can be versioned in git,
# allowing pipeline changes to go through the same review process as code
compiler.Compiler().compile(
    pipeline_func=ml_pipeline,
    package_path="ml_pipeline.yaml"
)
```

---

## 3. Component Development

### 3.1 Python Function Components

```python
"""
Python Function-Based Components
"""

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics

# Specify base_image and packages explicitly — ensures reproducibility;
# without this, the component inherits the default image which may drift over time
@dsl.component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas", "scikit-learn"]
)
def train_sklearn_model(
    training_data: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    n_estimators: int = 100,
    max_depth: int = 5
):
    """Train scikit-learn model"""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import joblib
    import json

    # Load data
    df = pd.read_csv(training_data.path)
    X = df.drop("target", axis=1)
    y = df["target"]

    # Train model
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    clf.fit(X, y)

    # Cross validation
    cv_scores = cross_val_score(clf, X, y, cv=5)

    # Save model
    joblib.dump(clf, model.path)

    # Log metrics
    metrics.log_metric("cv_mean", float(cv_scores.mean()))
    metrics.log_metric("cv_std", float(cv_scores.std()))
    metrics.log_metric("n_features", int(X.shape[1]))

# GPU-enabled component
@dsl.component(
    base_image="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
    packages_to_install=["transformers"]
)
def train_pytorch_model(
    data: Input[Dataset],
    model: Output[Model],
    epochs: int = 10,
    learning_rate: float = 0.001
):
    """Train PyTorch model (GPU)"""
    import torch
    # GPU training code...
    pass
```

### 3.2 Container-Based Components

```python
"""
Docker Container-Based Components
"""

from kfp import dsl
from kfp.dsl import ContainerSpec

# Method 1: Define container_spec directly
@dsl.container_component
def custom_training_component(
    data_path: str,
    output_path: dsl.OutputPath(str),
    epochs: int
):
    return ContainerSpec(
        image="gcr.io/my-project/training-image:latest",
        command=["python", "train.py"],
        args=[
            "--data-path", data_path,
            "--output-path", output_path,
            "--epochs", str(epochs)
        ]
    )

# Method 2: YAML component definition
component_yaml = """
name: Training Component
description: Custom training component
inputs:
  - name: data_path
    type: String
  - name: epochs
    type: Integer
    default: '10'
outputs:
  - name: model_path
    type: String
implementation:
  container:
    image: gcr.io/my-project/training:latest
    command:
      - python
      - train.py
    args:
      - --data-path
      - {inputValue: data_path}
      - --epochs
      - {inputValue: epochs}
      - --output-path
      - {outputPath: model_path}
"""

# Load component from YAML
from kfp.components import load_component_from_text
training_op = load_component_from_text(component_yaml)
```

### 3.3 Reusable Components

```python
"""
Reusable Component Library
"""

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model

# components/data_processing.py
@dsl.component(packages_to_install=["pandas", "numpy"])
def load_and_split_data(
    input_path: str,
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    test_size: float = 0.2,
    random_state: int = 42
):
    """Load and split data"""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_path)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    train_df.to_csv(train_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)

@dsl.component(packages_to_install=["pandas", "scikit-learn"])
def feature_engineering(
    input_data: Input[Dataset],
    output_data: Output[Dataset],
    numerical_features: list,
    categorical_features: list
):
    """Feature engineering"""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    df = pd.read_csv(input_data.path)

    # Scale numerical features
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Encode categorical features
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    df.to_csv(output_data.path, index=False)
```

---

## 4. Advanced Pipeline Features

### 4.1 Conditional Execution

```python
"""
Conditional Execution and Branching
"""

from kfp import dsl

@dsl.component
def check_data_quality(data_path: str) -> bool:
    """Check data quality"""
    import pandas as pd
    df = pd.read_csv(data_path)
    # Quality check logic
    return df.isnull().sum().sum() == 0

@dsl.component
def clean_data(data_path: str, output_path: dsl.OutputPath(str)):
    """Clean data"""
    import pandas as pd
    df = pd.read_csv(data_path)
    df = df.dropna()
    df.to_csv(output_path, index=False)

@dsl.component
def train_model(data_path: str):
    """Train model"""
    pass

@dsl.pipeline(name="conditional-pipeline")
def conditional_pipeline(data_path: str):
    # Check data quality
    quality_check = check_data_quality(data_path=data_path)

    # Conditional execution
    with dsl.Condition(quality_check.output == False, name="need-cleaning"):
        clean_task = clean_data(data_path=data_path)
        train_model(data_path=clean_task.outputs["output_path"])

    with dsl.Condition(quality_check.output == True, name="no-cleaning"):
        train_model(data_path=data_path)
```

### 4.2 Looping with ParallelFor

```python
"""
Looping Execution with ParallelFor
"""

from kfp import dsl
from typing import List

@dsl.component
def train_with_params(
    data_path: str,
    params: dict
) -> float:
    """Train model with parameters"""
    # Training logic
    return 0.95

@dsl.component
def select_best_model(
    accuracies: List[float],
    param_sets: List[dict]
) -> dict:
    """Select best performing model"""
    best_idx = accuracies.index(max(accuracies))
    return param_sets[best_idx]

@dsl.pipeline(name="hyperparameter-search")
def hyperparameter_search_pipeline(data_path: str):
    # Hyperparameter combinations
    param_sets = [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 10}
    ]

    # Parallel training
    with dsl.ParallelFor(param_sets) as params:
        train_task = train_with_params(
            data_path=data_path,
            params=params
        )

    # Collect results (outside ParallelFor)
    # select_best_model(...)
```

### 4.3 Resource Configuration

```python
"""
Component Resource Configuration
"""

from kfp import dsl
from kfp import kubernetes

@dsl.component
def gpu_training(data_path: str):
    """GPU training"""
    pass

@dsl.pipeline(name="resource-pipeline")
def resource_pipeline(data_path: str):
    # GPU training task
    train_task = gpu_training(data_path=data_path)

    # request vs limit: request reserves minimum resources for scheduling,
    # limit caps maximum to prevent one training job from starving other pods
    train_task.set_cpu_limit("4")
    train_task.set_memory_limit("16Gi")
    train_task.set_cpu_request("2")
    train_task.set_memory_request("8Gi")

    # GPU configuration
    kubernetes.add_node_selector(
        train_task,
        label_key="cloud.google.com/gke-accelerator",
        label_value="nvidia-tesla-t4"
    )
    train_task.set_accelerator_type("nvidia.com/gpu")
    train_task.set_accelerator_limit(1)

    # Environment variables
    train_task.set_env_variable("CUDA_VISIBLE_DEVICES", "0")

    # Volume mounts
    kubernetes.mount_pvc(
        train_task,
        pvc_name="data-pvc",
        mount_path="/data"
    )
```

---

## 5. Kubernetes Integration

### 5.1 Secrets and ConfigMaps

```python
"""
Kubernetes Resource Integration
"""

from kfp import dsl
from kfp import kubernetes

@dsl.component
def component_with_secrets():
    """Component using secrets"""
    import os
    api_key = os.environ.get("API_KEY")
    # ...

@dsl.pipeline(name="k8s-resources-pipeline")
def k8s_pipeline():
    task = component_with_secrets()

    # Environment variables from secret
    kubernetes.use_secret_as_env(
        task,
        secret_name="api-credentials",
        secret_key_to_env={"api-key": "API_KEY"}
    )

    # Environment variables from ConfigMap
    kubernetes.use_config_map_as_env(
        task,
        config_map_name="app-config",
        config_map_key_to_env={"setting": "APP_SETTING"}
    )

    # Mount secret as volume
    kubernetes.use_secret_as_volume(
        task,
        secret_name="tls-certs",
        mount_path="/certs"
    )
```

### 5.2 Service Accounts

```python
"""
Service Account Configuration
"""

from kfp import dsl
from kfp import kubernetes

@dsl.pipeline(name="sa-pipeline")
def service_account_pipeline():
    task = some_component()

    # Use specific service account
    kubernetes.set_service_account(
        task,
        service_account="ml-pipeline-sa"
    )

    # Image pull secret
    kubernetes.add_image_pull_secret(
        task,
        secret_name="docker-registry-secret"
    )
```

---

## 6. Running Pipelines

### 6.1 Running with SDK

```python
"""
Running Pipelines with KFP SDK
"""

from kfp import compiler
from kfp.client import Client

# Compile pipeline
compiler.Compiler().compile(
    pipeline_func=ml_pipeline,
    package_path="pipeline.yaml"
)

# Create KFP client
client = Client(host="http://kubeflow-pipelines-api:8888")

# Create or get experiment
experiment = client.create_experiment(
    name="my-experiment",
    description="ML training experiments"
)

# Run pipeline
run = client.create_run_from_pipeline_package(
    pipeline_file="pipeline.yaml",
    experiment_id=experiment.experiment_id,
    run_name="training-run-001",
    arguments={
        "input_data_path": "gs://bucket/data.csv",
        "n_estimators": 200
    }
)

print(f"Run ID: {run.run_id}")
print(f"Run URL: {client.get_run(run.run_id).display_url}")

# Wait for completion
client.wait_for_run_completion(run.run_id, timeout=3600)

# Check run status
run_details = client.get_run(run.run_id)
print(f"Status: {run_details.state}")
```

### 6.2 Scheduling

```python
"""
Pipeline Scheduling
"""

from kfp.client import Client

client = Client(host="http://kubeflow-pipelines-api:8888")

# Create recurring run (cron)
recurring_run = client.create_recurring_run(
    experiment_id=experiment.experiment_id,
    job_name="daily-training",
    pipeline_package_path="pipeline.yaml",
    cron_expression="0 2 * * *",  # Daily at 2 AM
    max_concurrency=1,
    arguments={
        "input_data_path": "gs://bucket/daily_data/"
    }
)

print(f"Recurring Run ID: {recurring_run.id}")

# Disable recurring run
client.disable_recurring_run(recurring_run.id)

# Enable recurring run
client.enable_recurring_run(recurring_run.id)
```

### 6.3 Querying Run Results

```python
"""
Query Run Results and Artifacts
"""

from kfp.client import Client

client = Client()

# Get specific run
run = client.get_run("run-id")
print(f"State: {run.state}")
print(f"Created: {run.created_at}")
print(f"Finished: {run.finished_at}")

# List runs
runs = client.list_runs(
    experiment_id=experiment.experiment_id,
    page_size=10,
    sort_by="created_at desc"
)

for r in runs.runs:
    print(f"{r.name}: {r.state}")

# Query pipeline outputs
# (Artifacts are typically stored in GCS, S3, etc.)
```

---

## 7. Complete Pipeline Example

```python
"""
Complete ML Pipeline Example
"""

from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Model, Metrics

@dsl.component(packages_to_install=["pandas", "scikit-learn"])
def ingest_data(
    source_path: str,
    output_data: Output[Dataset]
):
    """Ingest data"""
    import pandas as pd
    df = pd.read_csv(source_path)
    df.to_csv(output_data.path, index=False)

@dsl.component(packages_to_install=["pandas", "great-expectations"])
def validate_data(
    input_data: Input[Dataset],
    validation_report: Output[Dataset]
) -> bool:
    """Validate data"""
    import pandas as pd
    df = pd.read_csv(input_data.path)

    # Validation logic
    is_valid = (
        len(df) > 100 and
        df.isnull().sum().sum() / df.size < 0.1
    )

    # Save report
    report = {"valid": is_valid, "rows": len(df)}
    pd.DataFrame([report]).to_csv(validation_report.path, index=False)

    return is_valid

@dsl.component(packages_to_install=["pandas", "scikit-learn", "joblib"])
def train_and_evaluate(
    train_data: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    n_estimators: int = 100
) -> float:
    """Train and evaluate model"""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    import joblib

    df = pd.read_csv(train_data.path)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    joblib.dump(clf, model.path)

    metrics.log_metric("accuracy", float(accuracy))
    metrics.log_metric("f1_score", float(f1))

    return accuracy

@dsl.component
def deploy_model(
    model: Input[Model],
    accuracy: float,
    min_accuracy: float = 0.8
) -> str:
    """Deploy model"""
    if accuracy < min_accuracy:
        return "Model not deployed: accuracy below threshold"

    # Deployment logic (e.g., deploy model to serving infrastructure)
    return f"Model deployed with accuracy {accuracy}"

@dsl.pipeline(
    name="End-to-End ML Pipeline",
    description="Complete ML pipeline with data validation, training, and deployment"
)
def e2e_ml_pipeline(
    data_source: str = "gs://bucket/data.csv",
    n_estimators: int = 100,
    min_accuracy: float = 0.8
):
    # 1. Data ingestion
    ingest_task = ingest_data(source_path=data_source)

    # 2. Data validation
    validate_task = validate_data(
        input_data=ingest_task.outputs["output_data"]
    )

    # 3. Conditional gate — skips expensive training if data quality is poor,
    # preventing wasted GPU hours on corrupted data
    with dsl.Condition(validate_task.output == True, name="data-valid"):
        train_task = train_and_evaluate(
            train_data=ingest_task.outputs["output_data"],
            n_estimators=n_estimators
        )

        # 4. Deploy
        deploy_model(
            model=train_task.outputs["model"],
            accuracy=train_task.output,
            min_accuracy=min_accuracy
        )

# Compile
compiler.Compiler().compile(
    pipeline_func=e2e_ml_pipeline,
    package_path="e2e_pipeline.yaml"
)
```

---

## Exercises

### Exercise 1: Basic Pipeline
Create a 3-step pipeline: Load data -> Preprocess -> Train model

### Exercise 2: Hyperparameter Search
Create a pipeline that experiments with multiple hyperparameter combinations in parallel using ParallelFor.

### Exercise 3: Scheduling
Set up a retraining pipeline that runs every Monday morning.

---

## Summary

| Concept | Description |
|------|------|
| Pipeline | ML workflow DAG |
| Component | Individual step in pipeline |
| @dsl.component | Python function component |
| @dsl.pipeline | Pipeline definition |
| dsl.Condition | Conditional execution |
| dsl.ParallelFor | Parallel looping |

---

## References

- [Kubeflow Pipelines Documentation](https://www.kubeflow.org/docs/components/pipelines/)
- [KFP SDK v2](https://kubeflow-pipelines.readthedocs.io/)
- [Kubeflow Examples](https://github.com/kubeflow/examples)
