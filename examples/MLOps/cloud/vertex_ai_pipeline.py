"""
MLOps — Google Vertex AI Pipeline Example
===========================================
Demonstrates:
- Dataset creation and management
- Custom training job configuration
- KFP v2 pipeline components (preprocess, train, evaluate)
- Model upload and evaluation with quality gates
- Endpoint deployment with traffic splitting
- Model monitoring configuration

Prerequisites: pip install google-cloud-aiplatform kfp

Note: This script requires GCP credentials and a Vertex AI-enabled project.
It is designed as a reference implementation — read the code and comments
to understand the patterns, then adapt for your environment.

Run: python vertex_ai_pipeline.py <example>
Available: training, pipeline, deploy, monitor, all
"""

import sys
import json
from datetime import datetime


# ── 1. Vertex AI Custom Training ────────────────────────────────

# Why Vertex AI custom training over AutoML?
#   AutoML is great for quick baselines (no code, fast).
#   Custom training gives you full control over the model architecture,
#   training loop, data preprocessing, and dependency management.

def create_training_example():
    """Demonstrate Vertex AI custom training job configuration."""
    print("=" * 60)
    print("1. VERTEX AI CUSTOM TRAINING")
    print("=" * 60)

    training_config = {
        "display_name": "sklearn-classifier-training",
        "script_path": "src/train.py",

        # Why pre-built containers?
        #   Google maintains optimized containers for common frameworks.
        #   No need to build and push Docker images for standard setups.
        "container_uri": "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-3:latest",

        # Why a separate serving container?
        #   Training and serving have different dependencies.
        #   The serving container is optimized for inference (smaller, faster).
        "serving_container_uri": "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",

        "requirements": ["pandas>=2.0", "scikit-learn>=1.3"],

        "training_args": {
            # Why command-line args instead of config files?
            #   - Tracked by Vertex AI Experiments automatically
            #   - Can be overridden without changing code
            #   - Visible in the console UI
            "n-estimators": 100,
            "max-depth": 10,
            "test-size": 0.2,
        },

        "compute": {
            "replica_count": 1,
            "machine_type": "n1-standard-8",     # 8 vCPU, 30 GB RAM

            # GPU options (uncomment as needed):
            # "accelerator_type": "NVIDIA_TESLA_T4",
            # "accelerator_count": 1,

            # For large models:
            # "machine_type": "a2-highgpu-1g",   # A100 GPU
            # "accelerator_type": "NVIDIA_TESLA_A100",
            # "accelerator_count": 1,
        },

        "output_dir": "gs://my-ml-bucket/models/",
    }

    print("\nTraining Configuration:")
    print(json.dumps(training_config, indent=2))

    # Reference: Vertex AI SDK code
    print("\n--- Vertex AI SDK code (reference) ---")
    print("""
from google.cloud import aiplatform

aiplatform.init(
    project="my-gcp-project",
    location="us-central1",
    staging_bucket="gs://my-ml-bucket",
)

job = aiplatform.CustomTrainingJob(
    display_name="sklearn-classifier-training",
    script_path="src/train.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-3:latest",
    requirements=["pandas>=2.0", "scikit-learn>=1.3"],
    model_serving_container_image_uri=
        "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
)

# run() returns a Model object that can be deployed directly
model = job.run(
    model_display_name="sklearn-classifier-v1",
    args=["--n-estimators", "100", "--max-depth", "10"],
    replica_count=1,
    machine_type="n1-standard-8",
    base_output_dir="gs://my-ml-bucket/models/",
)
    """)

    # Reference: train.py entry point for Vertex AI
    print("\n--- train.py template (Vertex AI) ---")
    print("""
import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    args = parser.parse_args()

    # Vertex AI environment variables:
    #   AIP_MODEL_DIR: where to save the model (GCS path)
    #   AIP_DATA_FORMAT: format of training data
    model_dir = os.environ.get('AIP_MODEL_DIR', '/tmp/model')

    # Load data (from GCS or local)
    train_df = pd.read_csv('gs://my-bucket/data/train.csv')
    X = train_df.drop('target', axis=1)
    y = train_df['target']

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    clf.fit(X, y)

    # Save to AIP_MODEL_DIR — Vertex AI picks this up as the model artifact
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, 'model.joblib'))

if __name__ == '__main__':
    main()
    """)

    return training_config


# ── 2. KFP v2 Pipeline ──────────────────────────────────────────

# Why KFP v2 (Kubeflow Pipelines)?
#   Vertex AI Pipelines uses KFP v2 as its SDK. This means:
#   - Pipelines are portable (can also run on self-hosted Kubeflow)
#   - Components are reusable across teams and projects
#   - Visual DAG in the Vertex AI console

def create_pipeline_example():
    """Demonstrate Vertex AI Pipeline with KFP v2 components."""
    print("\n" + "=" * 60)
    print("2. VERTEX AI PIPELINE (KFP v2)")
    print("=" * 60)

    # Pipeline component definitions (as reference)
    pipeline_config = {
        "name": "vertex-ml-pipeline",
        "description": "End-to-end ML pipeline on Vertex AI",
        "parameters": {
            "input_uri": {
                "type": "str",
                "default": "gs://my-bucket/data/raw.csv",
            },
            "accuracy_threshold": {
                "type": "float",
                "default": 0.85,
            },
            "n_estimators": {
                "type": "int",
                "default": 100,
            },
        },
        "components": [
            {
                "name": "preprocess_data",
                "type": "ContainerComponent",
                "base_image": "python:3.11",
                "packages": ["pandas", "scikit-learn", "pyarrow"],
                "inputs": ["input_uri (str)", "test_size (float)"],
                "outputs": ["train_data (Dataset)", "test_data (Dataset)"],
                "description": (
                    # Why a separate preprocess component?
                    #   - Caching: if data hasn't changed, skip this step
                    #   - Reusability: same preprocessor for multiple models
                    #   - Different compute: CPU-only, less memory
                    "Split and clean raw data into train/test datasets. "
                    "Results are cached — reruns only if input changes."
                ),
            },
            {
                "name": "train_model",
                "type": "ContainerComponent",
                "base_image": "python:3.11",
                "packages": ["pandas", "scikit-learn", "joblib", "pyarrow"],
                "inputs": ["train_data (Dataset)", "n_estimators (int)", "max_depth (int)"],
                "outputs": ["model_artifact (Model)", "metrics (Metrics)"],
                "description": (
                    # Why output Metrics alongside Model?
                    #   Metrics are logged to Vertex AI Experiments automatically.
                    #   This enables experiment comparison in the console.
                    "Train a RandomForest classifier and log metrics."
                ),
            },
            {
                "name": "evaluate_model",
                "type": "ContainerComponent",
                "inputs": [
                    "test_data (Dataset)",
                    "model_artifact (Model)",
                    "accuracy_threshold (float)",
                ],
                "outputs": ["metrics (Metrics)"],
                "returns": "bool (deploy_approved)",
                "description": (
                    # Why return a boolean?
                    #   KFP Condition step uses this to decide deployment.
                    #   Only models meeting the threshold get deployed.
                    "Evaluate model on test data. Returns True if accuracy "
                    "meets the threshold."
                ),
            },
        ],
        "dag": {
            "preprocess_data": {"depends_on": []},
            "train_model": {"depends_on": ["preprocess_data"]},
            "evaluate_model": {"depends_on": ["preprocess_data", "train_model"]},
            "deploy_model": {
                "depends_on": ["evaluate_model"],
                "condition": "evaluate_model.output == True",
            },
        },
    }

    print("\nPipeline Configuration:")
    print(json.dumps(pipeline_config, indent=2))

    # Reference: Full KFP v2 pipeline code
    print("\n--- KFP v2 Pipeline code (reference) ---")
    print("""
from kfp.v2 import dsl, compiler
from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics

@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "pyarrow"],
)
def preprocess_data(
    input_uri: str,
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    test_size: float = 0.2,
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_uri)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df.to_parquet(train_data.path)
    test_df.to_parquet(test_data.path)

@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "joblib", "pyarrow"],
)
def train_model(
    train_data: Input[Dataset],
    model_artifact: Output[Model],
    metrics: Output[Metrics],
    n_estimators: int = 100,
):
    import pandas as pd, joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    train_df = pd.read_parquet(train_data.path)
    X, y = train_df.drop('target', axis=1), train_df['target']
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X, y)
    metrics.log_metric("train_accuracy", accuracy_score(y, clf.predict(X)))
    joblib.dump(clf, model_artifact.path + ".joblib")

@dsl.pipeline(name="vertex-ml-pipeline")
def ml_pipeline(input_uri: str, accuracy_threshold: float = 0.85):
    prep = preprocess_data(input_uri=input_uri)
    train = train_model(train_data=prep.outputs["train_data"])
    evaluate = evaluate_model(
        test_data=prep.outputs["test_data"],
        model_artifact=train.outputs["model_artifact"],
    )
    with dsl.Condition(evaluate.output == True):
        deploy_model(model=train.outputs["model_artifact"])

# Compile to YAML (can run on Vertex AI or self-hosted Kubeflow)
compiler.Compiler().compile(ml_pipeline, "pipeline.yaml")
    """)

    return pipeline_config


# ── 3. Vertex AI Endpoint Deployment ─────────────────────────────

# Why managed endpoints?
#   Vertex AI handles auto-scaling, health checks, traffic splitting,
#   and logging. You deploy a model; it handles the rest.

def create_deployment_example():
    """Demonstrate Vertex AI endpoint deployment configuration."""
    print("\n" + "=" * 60)
    print("3. VERTEX AI ENDPOINT DEPLOYMENT")
    print("=" * 60)

    deployment_config = {
        "endpoint": {
            "display_name": "sklearn-classifier-endpoint",
            # Why a separate Endpoint resource?
            #   An Endpoint can host multiple model versions simultaneously
            #   for A/B testing and gradual rollout.
        },
        "deployment": {
            "model_display_name": "sklearn-classifier-v1",
            "machine_type": "n1-standard-4",     # 4 vCPU, 15 GB
            "min_replica_count": 1,
            "max_replica_count": 5,

            # Why auto-scaling?
            #   ML traffic is often spiky. Scale from 1 instance at rest
            #   to 5 during peak. Pay only for what you use.
            "autoscaling_target_cpu_utilization": 60,

            # Traffic split for A/B testing
            "traffic_split": {"model-v1": 80, "model-v2": 20},
            # 80% goes to v1, 20% to v2
            # Monitor metrics, then shift when confident
        },
        "prediction": {
            # Why specify content type?
            #   Vertex AI needs to know how to serialize/deserialize
            #   the request/response for the serving container.
            "content_type": "application/json",
            "example_request": {
                "instances": [
                    {"age": 35, "income": 60000, "score": 75},
                    {"age": 28, "income": 45000, "score": 82},
                ],
            },
            "example_response": {
                "predictions": [0, 1],
            },
        },
    }

    print("\nDeployment Configuration:")
    print(json.dumps(deployment_config, indent=2))

    # Reference: Deployment SDK code
    print("\n--- Vertex AI Deployment code (reference) ---")
    print("""
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

# Upload model to registry
model = aiplatform.Model.upload(
    display_name="sklearn-classifier-v1",
    artifact_uri="gs://my-bucket/models/",
    serving_container_image_uri=
        "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
)

# Create endpoint
endpoint = aiplatform.Endpoint.create(
    display_name="sklearn-classifier-endpoint"
)

# Deploy model to endpoint
model.deploy(
    endpoint=endpoint,
    deployed_model_display_name="sklearn-v1",
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=5,
    traffic_percentage=100,
)

# Make predictions
prediction = endpoint.predict(
    instances=[{"age": 35, "income": 60000, "score": 75}]
)
print(prediction.predictions)
    """)

    return deployment_config


# ── 4. Vertex AI Model Monitoring ────────────────────────────────

# Why model monitoring?
#   Models degrade over time as data distributions shift.
#   Vertex AI Model Monitoring detects this automatically
#   and alerts before users are affected.

def create_monitoring_example():
    """Demonstrate Vertex AI Model Monitoring configuration."""
    print("\n" + "=" * 60)
    print("4. VERTEX AI MODEL MONITORING")
    print("=" * 60)

    monitoring_config = {
        "display_name": "sklearn-classifier-monitoring",

        "skew_detection": {
            # Why skew detection?
            #   Training-serving skew means the model sees different data
            #   in production than in training. Common causes:
            #   - Feature pipeline bugs
            #   - Different preprocessing in train vs serve
            #   - Data source changes
            "training_dataset": "gs://my-bucket/data/train.csv",
            "thresholds": {
                # Jensen-Shannon divergence thresholds per feature
                # Why JSD? It's symmetric and bounded [0, 1]
                "age": 0.3,
                "income": 0.3,
                "score": 0.3,
            },
            "default_threshold": 0.3,
        },

        "drift_detection": {
            # Why drift detection separately from skew?
            #   Skew = training data vs current serving data (pipeline bug)
            #   Drift = serving data over time (world changed)
            #   Same model, different root causes, different responses
            "thresholds": {
                "age": 0.3,
                "income": 0.3,
            },
            "default_threshold": 0.3,
        },

        "sampling": {
            # Why sample instead of monitoring every request?
            #   Cost and performance. At 10K req/sec, logging everything
            #   is expensive. A 10% sample is statistically sufficient
            #   for drift detection.
            "sampling_rate": 0.1,   # Monitor 10% of requests
        },

        "alerting": {
            "email": ["ml-team@company.com"],
            "notification_channels": ["projects/my-project/notificationChannels/123"],
            # Why multiple channels?
            #   Email for humans; Cloud Monitoring for automated response
            #   (trigger retraining via Cloud Functions)
        },

        "monitoring_schedule": {
            # How often to compute drift statistics
            "frequency": "1h",
            # Why hourly? Balance between detection speed and cost.
            # Real-time monitoring is possible but expensive.
        },
    }

    print("\nMonitoring Configuration:")
    print(json.dumps(monitoring_config, indent=2))

    # Reference: Monitoring SDK code
    print("\n--- Vertex AI Monitoring code (reference) ---")
    print("""
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

# Get the deployed endpoint
endpoint = aiplatform.Endpoint("projects/.../endpoints/123")

# Create monitoring job
# Why a monitoring job instead of custom code?
#   Vertex AI handles:
#   - Sampling prediction requests automatically
#   - Computing statistical tests (JSD, L-infinity)
#   - Comparing against training baseline
#   - Sending alerts through Cloud Monitoring
monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
    display_name="sklearn-monitoring",
    endpoint=endpoint,
    logging_sampling_strategy={
        "random_sample_config": {"sample_rate": 0.1}
    },
    schedule_config={"monitor_interval": {"seconds": 3600}},
    objective_configs=[{
        "training_dataset": {
            "gcs_source": {"uris": ["gs://my-bucket/data/train.csv"]},
            "data_format": "csv",
            "target_field": "target",
        },
        "training_prediction_skew_detection_config": {
            "skew_thresholds": {
                "age": {"value": 0.3},
                "income": {"value": 0.3},
            },
        },
        "prediction_drift_detection_config": {
            "drift_thresholds": {
                "age": {"value": 0.3},
                "income": {"value": 0.3},
            },
        },
    }],
    alert_config={
        "email_alert_config": {
            "user_emails": ["ml-team@company.com"]
        },
    },
)
    """)

    # Monitoring response pattern
    print("\n--- Monitoring Response Pattern ---")
    print("""
When Vertex AI detects drift:

  1. Alert fires (email + Cloud Monitoring)
     → "Feature 'income' drift detected: JSD = 0.42 > threshold 0.30"

  2. Automated response options:
     a) Cloud Function triggered → starts retraining pipeline
     b) Slack notification → human reviews and decides
     c) PagerDuty → on-call ML engineer investigates

  3. Investigation checklist:
     - Is the drift real or a monitoring false positive?
     - Is it data drift (input changed) or concept drift (relationship changed)?
     - Is the feature pipeline broken?
     - Has the upstream data source changed schema?

  4. Resolution:
     - Fix pipeline bug → redeploy same model
     - Real drift → retrain on recent data → deploy new model
     - Concept drift → may need new features or architecture
    """)

    return monitoring_config


# ── Main ─────────────────────────────────────────────────────────

DEMOS = {
    "training": create_training_example,
    "pipeline": create_pipeline_example,
    "deploy": create_deployment_example,
    "monitor": create_monitoring_example,
}


def main():
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"

    if choice == "all":
        for fn in DEMOS.values():
            fn()
    elif choice in DEMOS:
        DEMOS[choice]()
    else:
        print(f"Unknown: {choice}")
        print(f"Available: {', '.join(DEMOS.keys())}, all")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Vertex AI pipeline demo completed.")


if __name__ == "__main__":
    main()
