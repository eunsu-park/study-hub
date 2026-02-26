"""
MLOps — AWS SageMaker Pipeline Example
========================================
Demonstrates:
- Data processing step with ScriptProcessor
- Training step with hyperparameter tuning
- Model evaluation step with quality gates
- Conditional model registration (only if accuracy meets threshold)
- Endpoint deployment with auto-scaling configuration
- Cost optimization with spot instances

Prerequisites: pip install sagemaker boto3

Note: This script requires AWS credentials and a SageMaker execution role.
It is designed as a reference implementation — read the code and comments
to understand the patterns, then adapt for your environment.

Run: python sagemaker_pipeline.py <example>
Available: pipeline, training, deploy, cost, all
"""

import sys
import json
from datetime import datetime


# ── 1. SageMaker Training Job ───────────────────────────────────

# Why managed training over EC2?
#   SageMaker handles instance provisioning, shutdown, log collection,
#   and metric tracking. You only pay for training time, not idle time.

def create_training_job_example():
    """Demonstrate SageMaker training job configuration.

    Architecture:
      1. Upload code + data to S3
      2. SageMaker provisions ml.* instance(s)
      3. Pulls your container + data
      4. Runs training, streams logs to CloudWatch
      5. Saves model artifact (model.tar.gz) to S3
      6. Terminates instance — no idle cost
    """
    print("=" * 60)
    print("1. SAGEMAKER TRAINING JOB")
    print("=" * 60)

    # This is a reference configuration — requires SageMaker SDK
    training_config = {
        "estimator": {
            # Why SKLearn estimator instead of generic?
            #   Pre-built container with sklearn installed
            #   No need to build and push a Docker image
            "framework": "SKLearn",
            "entry_point": "train.py",
            "source_dir": "src/",
            "framework_version": "1.2-1",
            "py_version": "py3",
            "instance_type": "ml.m5.xlarge",   # 4 vCPU, 16 GB RAM
            "instance_count": 1,
            "output_path": "s3://my-bucket/models/",
        },
        "hyperparameters": {
            # Why pass hyperparameters here instead of hardcoding?
            #   - Tracked automatically by SageMaker Experiments
            #   - Can be tuned with HyperparameterTuner
            #   - Visible in SageMaker Studio UI
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "test_size": 0.2,
        },
        "spot_training": {
            # Why spot instances?
            #   Up to 90% cost reduction for interruptible workloads.
            #   SageMaker handles checkpointing automatically.
            "use_spot_instances": True,
            "max_wait": 7200,    # Max wait for spot capacity (seconds)
            "max_run": 3600,     # Max training time (seconds)
            # Savings: ~70% for ml.m5.xlarge ($0.23 → $0.07/hr)
        },
        "input_channels": {
            # Why channels? SageMaker mounts data to specific paths:
            #   /opt/ml/input/data/train/ and /opt/ml/input/data/test/
            "train": "s3://my-bucket/data/train/",
            "test": "s3://my-bucket/data/test/",
        },
    }

    print("\nTraining Configuration:")
    print(json.dumps(training_config, indent=2))

    # Reference: train.py entry point structure
    print("\n--- train.py template ---")
    print("""
import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)
    args = parser.parse_args()

    # SageMaker mounts data to these paths automatically
    train_dir = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train')
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')

    # Load data
    train_df = pd.read_csv(os.path.join(train_dir, 'train.csv'))
    X = train_df.drop('target', axis=1)
    y = train_df['target']

    # Train
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    clf.fit(X, y)

    # Save — SageMaker packages SM_MODEL_DIR as model.tar.gz
    joblib.dump(clf, os.path.join(model_dir, 'model.pkl'))

if __name__ == '__main__':
    main()
    """)

    return training_config


# ── 2. SageMaker Pipeline ───────────────────────────────────────

# Why SageMaker Pipelines over Airflow/Step Functions?
#   - Native integration with SageMaker steps (no glue code)
#   - Built-in step caching (skip unchanged steps)
#   - Visual DAG in SageMaker Studio
#   - Parameterized for reuse across environments

def create_pipeline_example():
    """Demonstrate SageMaker Pipeline configuration.

    Pipeline DAG:
      ┌────────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐
      │ Processing │──▶│ Training │──▶│ Evaluate  │──▶│ Condition│
      │ (data prep)│   │          │   │ (metrics) │   │ (acc>85%)│
      └────────────┘   └──────────┘   └───────────┘   └──────┬───┘
                                                         YES  │ NO
                                                       ┌──────▼─────┐
                                                       │  Register  │
                                                       │  Model     │
                                                       └────────────┘
    """
    print("\n" + "=" * 60)
    print("2. SAGEMAKER PIPELINE")
    print("=" * 60)

    pipeline_config = {
        "name": "ml-training-pipeline",
        "parameters": {
            # Why pipeline parameters?
            #   Change these without modifying pipeline code.
            #   Same pipeline works for dev/staging/prod.
            "InputData": {
                "type": "String",
                "default": "s3://my-bucket/data/raw/",
            },
            "AccuracyThreshold": {
                "type": "Float",
                "default": 0.85,
            },
            "InstanceType": {
                "type": "String",
                "default": "ml.m5.xlarge",
            },
        },
        "steps": [
            {
                "name": "PreprocessData",
                "type": "ProcessingStep",
                "config": {
                    # Why a separate processing step?
                    #   - Decouples data prep from training
                    #   - Different compute needs (CPU vs GPU)
                    #   - Results are cached — reuse if data unchanged
                    "processor": "ScriptProcessor",
                    "instance_type": "ml.m5.xlarge",
                    "instance_count": 1,
                    "code": "src/preprocess.py",
                    "inputs": ["s3://my-bucket/data/raw/"],
                    "outputs": ["train.csv", "test.csv"],
                },
            },
            {
                "name": "TrainModel",
                "type": "TrainingStep",
                "depends_on": ["PreprocessData"],
                "config": {
                    "estimator": "SKLearn",
                    "instance_type": "ml.m5.xlarge",
                    "entry_point": "train.py",
                    "hyperparameters": {
                        "n_estimators": 100,
                        "max_depth": 10,
                    },
                },
            },
            {
                "name": "EvaluateModel",
                "type": "ProcessingStep",
                "depends_on": ["TrainModel"],
                "config": {
                    # Why evaluate in a separate step?
                    #   - Clean separation of concerns
                    #   - Evaluation output feeds into the condition step
                    #   - Can use different instance type
                    "code": "src/evaluate.py",
                    "outputs": ["evaluation_report.json"],
                },
            },
            {
                "name": "CheckAccuracy",
                "type": "ConditionStep",
                "depends_on": ["EvaluateModel"],
                "config": {
                    # Why conditional registration?
                    #   Only models meeting the quality threshold enter
                    #   the registry. This prevents bad models from being
                    #   accidentally deployed.
                    "condition": "accuracy >= AccuracyThreshold",
                    "if_steps": ["RegisterModel"],
                    "else_steps": [],
                },
            },
            {
                "name": "RegisterModel",
                "type": "RegisterModel",
                "config": {
                    "model_package_group": "my-model-group",
                    "approval_status": "PendingManualApproval",
                    "inference_instances": ["ml.m5.large", "ml.m5.xlarge"],
                },
            },
        ],
    }

    print("\nPipeline Configuration:")
    print(json.dumps(pipeline_config, indent=2))

    # Reference: Pipeline SDK code structure
    print("\n--- Pipeline SDK code (reference) ---")
    print("""
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat

# 1. Define parameters
input_data = ParameterString(name="InputData", default_value="s3://...")
threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.85)

# 2. Define steps (ProcessingStep, TrainingStep, etc.)
# 3. Define conditions (ConditionStep)
# 4. Assemble pipeline
pipeline = Pipeline(
    name="ml-pipeline",
    parameters=[input_data, threshold],
    steps=[processing_step, training_step, condition_step],
)

# 5. Create/update and start
pipeline.upsert(role_arn=role)
execution = pipeline.start()
    """)

    return pipeline_config


# ── 3. SageMaker Endpoint Deployment ────────────────────────────

# Why managed endpoints over self-hosted serving?
#   - Auto-scaling based on traffic
#   - Built-in A/B testing (traffic splitting)
#   - Automatic health checks and rollback
#   - CloudWatch metrics for monitoring

def create_deployment_example():
    """Demonstrate SageMaker endpoint deployment configuration."""
    print("\n" + "=" * 60)
    print("3. SAGEMAKER ENDPOINT DEPLOYMENT")
    print("=" * 60)

    endpoint_config = {
        "endpoint_name": "sklearn-classifier-prod",
        "model_config": {
            # Why a Model object separate from the estimator?
            #   The Model decouples the training artifact from serving.
            #   You can deploy the same model with different configurations.
            "model_data": "s3://my-bucket/models/model.tar.gz",
            "image_uri": "sklearn-inference:latest",
            "role": "arn:aws:iam::123456789012:role/SageMakerRole",
        },
        "production_variant": {
            "variant_name": "primary",
            "instance_type": "ml.m5.large",
            "initial_instance_count": 2,
            # Why 2 instances? High availability — if one fails,
            # the other continues serving while replacement spins up
        },
        "auto_scaling": {
            # Why auto-scaling?
            #   ML traffic is often spiky (batch jobs, time-of-day patterns).
            #   Pay for minimum capacity at rest, scale up for peaks.
            "min_capacity": 2,
            "max_capacity": 10,
            "target_invocations_per_instance": 100,
            # Scale up when average invocations per instance exceeds 100
            "scale_in_cooldown": 300,   # Wait 5 min before scaling down
            "scale_out_cooldown": 60,   # Scale up quickly (1 min)
        },
        "ab_testing": {
            # Why A/B testing at the endpoint level?
            #   Compare model versions on real traffic without risk.
            #   SageMaker handles traffic splitting automatically.
            "variant_a": {"name": "model-v1", "weight": 80},
            "variant_b": {"name": "model-v2", "weight": 20},
            # 80/20 split: v1 serves 80% of traffic, v2 gets 20%
            # Monitor metrics, then shift traffic when confident
        },
    }

    print("\nEndpoint Configuration:")
    print(json.dumps(endpoint_config, indent=2))

    # Reference: Deployment SDK code
    print("\n--- Deployment SDK code (reference) ---")
    print("""
from sagemaker import Model
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

# Create model
model = Model(
    model_data='s3://my-bucket/models/model.tar.gz',
    image_uri='...',
    role=role,
)

# Deploy with auto-scaling
predictor = model.deploy(
    initial_instance_count=2,
    instance_type='ml.m5.large',
    endpoint_name='sklearn-classifier-prod',
    serializer=CSVSerializer(),
    deserializer=JSONDeserializer(),
)

# Invoke
result = predictor.predict([[25, 50000, 3]])
    """)

    return endpoint_config


# ── 4. Cost Optimization ────────────────────────────────────────

# Why track ML costs explicitly?
#   Cloud ML spend can grow 10x in months without anyone noticing.
#   Training, endpoints, and notebooks all accumulate cost.

def cost_optimization_example():
    """Demonstrate SageMaker cost estimation and optimization strategies."""
    print("\n" + "=" * 60)
    print("4. COST OPTIMIZATION")
    print("=" * 60)

    # Approximate SageMaker pricing (USD/hour, as of 2025)
    pricing = {
        "ml.m5.xlarge":     {"on_demand": 0.23,  "spot": 0.069},
        "ml.m5.2xlarge":    {"on_demand": 0.46,  "spot": 0.138},
        "ml.g4dn.xlarge":   {"on_demand": 0.736, "spot": 0.221},
        "ml.g5.xlarge":     {"on_demand": 1.006, "spot": 0.302},
        "ml.p3.2xlarge":    {"on_demand": 3.825, "spot": 1.148},
        "ml.p4d.24xlarge":  {"on_demand": 32.77, "spot": 9.831},
        "ml.inf2.xlarge":   {"on_demand": 0.758, "spot": None},
    }

    print("\nSageMaker Instance Pricing (approx.):")
    print(f"{'Instance':<20} {'On-Demand':>10} {'Spot':>10} {'Savings':>10}")
    print("-" * 52)
    for inst, prices in pricing.items():
        spot_str = f"${prices['spot']:.3f}" if prices['spot'] else "N/A"
        savings = ""
        if prices['spot']:
            savings = f"{(1 - prices['spot']/prices['on_demand'])*100:.0f}%"
        print(f"{inst:<20} ${prices['on_demand']:.3f}     {spot_str:>10} {savings:>10}")

    # Cost scenario analysis
    print("\n--- Monthly Cost Scenarios ---")
    scenarios = [
        {
            "name": "Small team (1 daily training)",
            "training_instance": "ml.m5.xlarge",
            "training_hours_per_day": 2,
            "endpoint_instance": "ml.m5.large",
            "endpoint_count": 1,
            "notebook_hours_per_day": 8,
        },
        {
            "name": "Medium team (5 daily trainings)",
            "training_instance": "ml.g4dn.xlarge",
            "training_hours_per_day": 10,
            "endpoint_instance": "ml.m5.xlarge",
            "endpoint_count": 2,
            "notebook_hours_per_day": 40,
        },
        {
            "name": "Large team (20 daily trainings)",
            "training_instance": "ml.p3.2xlarge",
            "training_hours_per_day": 40,
            "endpoint_instance": "ml.g4dn.xlarge",
            "endpoint_count": 4,
            "notebook_hours_per_day": 160,
        },
    ]

    for scenario in scenarios:
        train_inst = scenario["training_instance"]
        train_rate = pricing.get(train_inst, {}).get("on_demand", 0)
        spot_rate = pricing.get(train_inst, {}).get("spot", train_rate)

        # Monthly costs (30 days)
        train_cost_ondemand = train_rate * scenario["training_hours_per_day"] * 30
        train_cost_spot = spot_rate * scenario["training_hours_per_day"] * 30

        # Endpoint cost (24/7)
        endpoint_rate = pricing.get(scenario["endpoint_instance"], {}).get("on_demand", 0)
        endpoint_cost = endpoint_rate * 24 * 30 * scenario["endpoint_count"]

        # Notebook cost (business hours)
        notebook_rate = 0.23  # ml.m5.xlarge for notebooks
        notebook_cost = notebook_rate * scenario["notebook_hours_per_day"] * 22  # Weekdays

        total_ondemand = train_cost_ondemand + endpoint_cost + notebook_cost
        total_optimized = train_cost_spot + endpoint_cost * 0.7 + notebook_cost * 0.5

        print(f"\n{scenario['name']}:")
        print(f"  Training (on-demand):  ${train_cost_ondemand:,.0f}/month")
        print(f"  Training (spot):       ${train_cost_spot:,.0f}/month")
        print(f"  Endpoint (24/7):       ${endpoint_cost:,.0f}/month")
        print(f"  Notebooks:             ${notebook_cost:,.0f}/month")
        print(f"  Total (on-demand):     ${total_ondemand:,.0f}/month")
        print(f"  Total (optimized):     ${total_optimized:,.0f}/month")
        print(f"  Potential savings:     ${total_ondemand - total_optimized:,.0f}/month "
              f"({(1 - total_optimized/max(total_ondemand,1))*100:.0f}%)")

    # Optimization recommendations
    print("\n--- Key Optimization Strategies ---")
    strategies = [
        ("Spot training", "60-70% savings", "Enable use_spot_instances=True with checkpointing"),
        ("Auto-shutdown notebooks", "20-40% savings", "Set lifecycle config: 1hr idle → stop"),
        ("Right-size endpoints", "30-50% savings", "Profile GPU utilization; use Inf2 for inference"),
        ("Reserved capacity", "30-40% savings", "1-year commitment for stable endpoint workloads"),
        ("Multi-model endpoints", "50-70% savings", "Serve multiple models on one instance"),
    ]

    print(f"\n{'Strategy':<25} {'Savings':<15} {'How'}")
    print("-" * 80)
    for strategy, savings, how in strategies:
        print(f"{strategy:<25} {savings:<15} {how}")


# ── Main ─────────────────────────────────────────────────────────

DEMOS = {
    "training": create_training_job_example,
    "pipeline": create_pipeline_example,
    "deploy": create_deployment_example,
    "cost": cost_optimization_example,
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
    print("SageMaker pipeline demo completed.")


if __name__ == "__main__":
    main()
