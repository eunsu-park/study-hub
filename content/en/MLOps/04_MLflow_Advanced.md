# MLflow Advanced

## Learning Objectives

After completing this lesson, you will be able to:

1. Create MLflow Projects with MLproject files that define entry points, parameters, and reproducible environments for ML code
2. Implement custom MLflow models by subclassing `mlflow.pyfunc.PythonModel` to package non-standard inference logic
3. Build multi-step MLflow pipelines that chain data preprocessing, training, and evaluation as parent-child runs
4. Configure MLflow with remote artifact stores (S3, GCS) and database-backed tracking servers for production deployments
5. Integrate MLflow tracking into hyperparameter tuning workflows using Optuna or Ray Tune

---

## 1. MLflow Projects

MLflow Projects is a packaging format for reproducible ML code.

### 1.1 Project Structure

```
my_ml_project/
├── MLproject              # Project definition file
├── conda.yaml             # Conda environment definition
├── requirements.txt       # pip dependencies (optional)
├── train.py               # Training script
├── evaluate.py            # Evaluation script
└── data/
    └── sample_data.csv
```

### 1.2 MLproject File

```yaml
# MLproject
name: churn-prediction

# Environment definition (3 options)
# Option 1: Conda
conda_env: conda.yaml

# Option 2: Docker
# docker_env:
#   image: my-docker-image:latest

# Option 3: System (use current environment)
# python_env: python_env.yaml

# Entry points — separate train/evaluate/search so each can be triggered
# independently in CI/CD without running the full pipeline
entry_points:
  main:
    parameters:
      data_path: {type: str, default: "data/train.csv"}
      n_estimators: {type: int, default: 100}
      max_depth: {type: int, default: 5}
      learning_rate: {type: float, default: 0.1}
    command: "python train.py --data-path {data_path} --n-estimators {n_estimators} --max-depth {max_depth} --learning-rate {learning_rate}"

  evaluate:
    parameters:
      model_path: {type: str}
      test_data: {type: str}
    command: "python evaluate.py --model-path {model_path} --test-data {test_data}"

  hyperparameter_search:
    parameters:
      n_trials: {type: int, default: 50}
    command: "python hyperparam_search.py --n-trials {n_trials}"
```

### 1.3 conda.yaml

```yaml
# conda.yaml
name: churn-prediction-env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pip
  - scikit-learn=1.2.0
  - pandas=1.5.0
  - numpy=1.23.0
  - pip:
    - mlflow>=2.0
    - xgboost>=1.7
```

### 1.4 Running Projects

```bash
# Run locally
mlflow run . -P n_estimators=200 -P max_depth=10

# Run directly from Git
mlflow run https://github.com/user/ml-project.git -P data_path=s3://bucket/data.csv

# Specific branch/tag
mlflow run https://github.com/user/ml-project.git --version main

# Run in Docker environment
mlflow run . --env-manager docker

# Run specific entry point
mlflow run . -e evaluate -P model_path=models/model.pkl -P test_data=data/test.csv

# Specify experiment
mlflow run . --experiment-name "production-training"
```

### 1.5 train.py Example

```python
"""
train.py - MLflow Project Training Script
"""

import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main(args):
    # autolog captures model params, metrics, and artifacts automatically —
    # reduces boilerplate and prevents forgetting to log important details
    mlflow.sklearn.autolog()

    # Context manager ensures run closure even if training crashes mid-way
    with mlflow.start_run():
        # Load data
        df = pd.read_csv(args.data_path)
        X = df.drop("target", axis=1)
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Log data provenance — essential for debugging when model quality degrades;
        # autolog doesn't capture custom metadata like data paths
        mlflow.log_param("data_path", args.data_path)
        mlflow.log_param("train_size", len(X_train))

        # Train model
        model = GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='macro'),
            "recall": recall_score(y_test, y_pred, average='macro'),
            "f1": f1_score(y_test, y_pred, average='macro')
        }

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        print(f"Model trained with accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
```

---

## 2. MLflow Models

### 2.1 Model Flavors

```python
"""
MLflow Model Flavors
"""

import mlflow

# Supported flavors
flavors = {
    "sklearn": "scikit-learn models",
    "pytorch": "PyTorch models",
    "tensorflow": "TensorFlow/Keras models",
    "xgboost": "XGBoost models",
    "lightgbm": "LightGBM models",
    "catboost": "CatBoost models",
    "transformers": "HuggingFace Transformers",
    "langchain": "LangChain models",
    "onnx": "ONNX models",
    "pyfunc": "Python functions (custom)"
}
```

### 2.2 Model Signatures

```python
"""
Defining Model Signatures
"""

import mlflow
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec

# Method 1: Automatic inference
signature = infer_signature(X_train, model.predict(X_train))

# Method 2: Explicit definition
input_schema = Schema([
    ColSpec("double", "feature_1"),
    ColSpec("double", "feature_2"),
    ColSpec("string", "category")
])
output_schema = Schema([ColSpec("long", "prediction")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Include signature when saving model
mlflow.sklearn.log_model(
    model,
    "model",
    signature=signature,
    input_example=X_train[:5]  # Input example
)
```

### 2.3 Custom Models (pyfunc)

```python
"""
Custom MLflow Model (pyfunc)
"""

import mlflow
import mlflow.pyfunc
import pandas as pd

class CustomModel(mlflow.pyfunc.PythonModel):
    """Custom MLflow Model"""

    # pyfunc wraps preprocessing + model + postprocessing into one artifact —
    # guarantees the serving environment uses the exact same pipeline as training
    def __init__(self, preprocessor, model, threshold=0.5):
        self.preprocessor = preprocessor
        self.model = model
        self.threshold = threshold

    def load_context(self, context):
        """Load artifacts"""
        # Called once at model load time (not per-request) — heavy initialization here
        # avoids repeated I/O overhead during inference
        import joblib
        # Can load additional files from context.artifacts
        pass

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """Perform prediction"""
        # Preprocessing
        processed = self.preprocessor.transform(model_input)

        # Prediction
        probabilities = self.model.predict_proba(processed)[:, 1]

        # Return both label and probability — callers can apply their own
        # business-specific threshold without re-running inference
        predictions = (probabilities >= self.threshold).astype(int)

        return pd.DataFrame({
            "prediction": predictions,
            "probability": probabilities
        })


# Save custom model
custom_model = CustomModel(preprocessor, trained_model, threshold=0.6)

# Define Conda environment
conda_env = {
    "channels": ["conda-forge"],
    "dependencies": [
        "python=3.9",
        "pip",
        {"pip": ["mlflow", "scikit-learn", "pandas"]}
    ],
    "name": "custom_model_env"
}

# Bundle external artifacts with the model — ensures preprocessor and config
# travel together with the model, eliminating "missing artifact" errors at serving time
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=custom_model,
        conda_env=conda_env,
        artifacts={
            "preprocessor": "artifacts/preprocessor.pkl",
            "config": "artifacts/config.yaml"
        },
        signature=signature,
        input_example=sample_input
    )
```

### 2.4 Model Format Structure

```
model/
├── MLmodel                    # Model metadata
├── model.pkl                  # Serialized model
├── conda.yaml                 # Conda environment
├── python_env.yaml            # Python environment
├── requirements.txt           # pip dependencies
├── input_example.json         # Input example
└── registered_model_meta      # Registry metadata
```

```yaml
# MLmodel file contents
artifact_path: model
flavors:
  python_function:
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    predict_fn: predict
    python_version: 3.9.0
  sklearn:
    code: null
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: 1.2.0
mlflow_version: 2.8.0
model_uuid: a1b2c3d4-e5f6-7890-abcd-ef1234567890
signature:
  inputs: '[{"name": "feature_1", "type": "double"}, ...]'
  outputs: '[{"type": "long"}]'
```

---

## 3. Model Registry

### 3.1 Model Registration

```python
"""
Using Model Registry
"""

import mlflow
from mlflow.tracking import MlflowClient

# Method 1: Register directly when logging model
with mlflow.start_run():
    # Training...
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="ChurnPredictionModel"  # Auto-register
    )

# Method 2: Register from existing run
result = mlflow.register_model(
    model_uri="runs:/RUN_ID/model",
    name="ChurnPredictionModel"
)
print(f"Version: {result.version}")

# Method 3: Using MlflowClient
client = MlflowClient()
client.create_registered_model(
    name="ChurnPredictionModel",
    description="Customer churn prediction model",
    tags={"team": "ML", "project": "retention"}
)

# Add version
client.create_model_version(
    name="ChurnPredictionModel",
    source="runs:/RUN_ID/model",
    run_id="RUN_ID",
    description="Initial version with RF"
)
```

### 3.2 Model Stage Management

```python
"""
Model Stage Transitions
"""

from mlflow.tracking import MlflowClient

client = MlflowClient()

# Stages: None, Staging, Production, Archived

# Transition to Staging
client.transition_model_version_stage(
    name="ChurnPredictionModel",
    version=1,
    stage="Staging",
    archive_existing_versions=False
)

# Promote to Production
client.transition_model_version_stage(
    name="ChurnPredictionModel",
    version=1,
    stage="Production",
    # Auto-archive prevents multiple Production versions — ensures load_model("Production")
    # always returns exactly one model
    archive_existing_versions=True
)

# Load model (by stage)
staging_model = mlflow.pyfunc.load_model("models:/ChurnPredictionModel/Staging")
prod_model = mlflow.pyfunc.load_model("models:/ChurnPredictionModel/Production")

# Load specific version
model_v1 = mlflow.pyfunc.load_model("models:/ChurnPredictionModel/1")
```

### 3.3 Model Metadata Management

```python
"""
Model Version Metadata
"""

from mlflow.tracking import MlflowClient

client = MlflowClient()

# Update model description
client.update_registered_model(
    name="ChurnPredictionModel",
    description="Updated description"
)

# Update version description
client.update_model_version(
    name="ChurnPredictionModel",
    version=1,
    description="Improved feature engineering"
)

# Add tags
client.set_registered_model_tag(
    name="ChurnPredictionModel",
    key="task",
    value="binary_classification"
)

client.set_model_version_tag(
    name="ChurnPredictionModel",
    version=1,
    key="validated",
    value="true"
)

# Get model information
model = client.get_registered_model("ChurnPredictionModel")
print(f"Name: {model.name}")
print(f"Description: {model.description}")
print(f"Latest versions: {model.latest_versions}")

# Get version information
version = client.get_model_version("ChurnPredictionModel", 1)
print(f"Version: {version.version}")
print(f"Stage: {version.current_stage}")
print(f"Source: {version.source}")
```

### 3.4 Model Search

```python
"""
Searching Registered Models
"""

from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get all models
models = client.search_registered_models()
for m in models:
    print(f"Model: {m.name}, Latest: {m.latest_versions}")

# Filtered search
models = client.search_registered_models(
    filter_string="name LIKE '%Churn%'"
)

# Search versions
versions = client.search_model_versions(
    filter_string="name='ChurnPredictionModel' and current_stage='Production'"
)
for v in versions:
    print(f"Version {v.version}: {v.current_stage}")
```

---

## 4. MLflow Serving

### 4.1 Local Serving

```bash
# Serve model (run ID based)
mlflow models serve -m "runs:/RUN_ID/model" -p 5001 --no-conda

# Serve model (Registry based)
mlflow models serve -m "models:/ChurnPredictionModel/Production" -p 5001

# Environment options
mlflow models serve -m "models:/MyModel/1" \
    --env-manager local \
    --host 0.0.0.0 \
    --port 5001
```

### 4.2 REST API Calls

```python
"""
Calling MLflow Serving API
"""

import requests
import json

# Endpoint
url = "http://localhost:5001/invocations"

# Input data (multiple formats supported)
# Format 1: split orientation
data_split = {
    "dataframe_split": {
        "columns": ["feature_1", "feature_2", "feature_3"],
        "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    }
}

# Format 2: records orientation
data_records = {
    "dataframe_records": [
        {"feature_1": 1.0, "feature_2": 2.0, "feature_3": 3.0},
        {"feature_1": 4.0, "feature_2": 5.0, "feature_3": 6.0}
    ]
}

# Format 3: instances (TensorFlow Serving compatible)
data_instances = {
    "instances": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
}

# API call
response = requests.post(
    url,
    headers={"Content-Type": "application/json"},
    data=json.dumps(data_split)
)

print(f"Status: {response.status_code}")
print(f"Predictions: {response.json()}")
```

### 4.3 Building Docker Images

```bash
# Create Docker image
mlflow models build-docker \
    -m "models:/ChurnPredictionModel/Production" \
    -n "churn-model:latest"

# Run image
docker run -p 5001:8080 churn-model:latest

# Generate Dockerfile directly
mlflow models generate-dockerfile \
    -m "models:/ChurnPredictionModel/Production" \
    -d ./docker-build
```

### 4.4 Batch Inference

```python
"""
Performing Batch Inference
"""

import mlflow
import pandas as pd

# Load model
model = mlflow.pyfunc.load_model("models:/ChurnPredictionModel/Production")

# Load batch data
batch_data = pd.read_parquet("s3://bucket/batch_data.parquet")

# Batch prediction
predictions = model.predict(batch_data)

# Save results
results = batch_data.copy()
results["prediction"] = predictions
results.to_parquet("s3://bucket/predictions.parquet")
```

---

## 5. Advanced Configuration

### 5.1 Remote Tracking Server

```bash
# PostgreSQL for metadata (runs, params, metrics) + S3 for large artifacts (models, data)
# — separating concerns: relational queries on metadata, cheap bulk storage for binaries
mlflow server \
    --backend-store-uri postgresql://user:password@host:5432/mlflow \
    --default-artifact-root s3://mlflow-artifacts/ \
    --host 0.0.0.0 \
    --port 5000

# Set environment variables
export MLFLOW_TRACKING_URI=http://mlflow-server:5000
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
```

### 5.2 Authentication Setup

```python
"""
MLflow Authentication Setup
"""

import os
import mlflow

# Basic authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = "user"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"

# Token-based authentication
os.environ["MLFLOW_TRACKING_TOKEN"] = "your-token"

# Azure ML integration
os.environ["AZURE_TENANT_ID"] = "tenant-id"
os.environ["AZURE_CLIENT_ID"] = "client-id"
os.environ["AZURE_CLIENT_SECRET"] = "client-secret"
```

### 5.3 Using Plugins

```python
"""
MLflow Plugin Examples
"""

# Databricks plugin
# pip install databricks-cli

import mlflow
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/user@email.com/my-experiment")

# Google Cloud plugin
# pip install mlflow[google-cloud]
mlflow.set_tracking_uri("gs://bucket/mlflow")
```

---

## 6. Complete Workflow

```python
"""
Complete MLflow Workflow Example
"""

import mlflow
from mlflow.tracking import MlflowClient

# 1. Setup
mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("production-churn-model")
client = MlflowClient()

# 2. Training and experimentation
with mlflow.start_run(run_name="rf-optimized") as run:
    # Training code...
    mlflow.sklearn.log_model(model, "model", signature=signature)
    run_id = run.info.run_id

# 3. Register model
model_version = mlflow.register_model(
    f"runs:/{run_id}/model",
    "ChurnPredictionModel"
)

# 4. Transition to Staging
client.transition_model_version_stage(
    name="ChurnPredictionModel",
    version=model_version.version,
    stage="Staging"
)

# 5. Test in Staging first — validates the model loads correctly and meets thresholds
# in an environment that mirrors production before affecting real traffic
staging_model = mlflow.pyfunc.load_model("models:/ChurnPredictionModel/Staging")
test_results = evaluate_model(staging_model, test_data)

# 6. Gate promotion on metrics — prevents regressions from reaching production
if test_results["accuracy"] > 0.9:
    client.transition_model_version_stage(
        name="ChurnPredictionModel",
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Model v{model_version.version} promoted to Production!")
```

---

## Exercises

### Exercise 1: Create MLflow Project
Create a complete MLflow Project and run it locally.

### Exercise 2: Custom pyfunc Model
Write a custom pyfunc model that includes preprocessing and post-processing.

### Exercise 3: Model Registry Workflow
Automate model registration and Staging -> Production transitions.

---

## Summary

| Feature | Description |
|------|------|
| MLflow Projects | Reproducible code packaging |
| MLflow Models | Standardized model format |
| Model Registry | Model version and stage management |
| MLflow Serving | REST API serving |
| pyfunc | Custom model wrapper |

---

## References

- [MLflow Projects](https://mlflow.org/docs/latest/projects.html)
- [MLflow Models](https://mlflow.org/docs/latest/models.html)
- [Model Registry](https://mlflow.org/docs/latest/model-registry.html)
