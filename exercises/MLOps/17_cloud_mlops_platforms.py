"""
Exercise Solutions: Cloud MLOps Platforms
===========================================
Lesson 17 from MLOps topic.

Exercises
---------
1. SageMaker Training Job — Simulate creating a SageMaker training job
   with spot instances, custom metrics, and artifact retrieval.
2. Vertex AI Pipeline — Simulate a 3-step KFP v2 pipeline with
   conditional deployment based on accuracy threshold.
3. Platform Comparison — Evaluate AWS SageMaker, Google Vertex AI, and
   Azure ML across a feature matrix for a hypothetical project.
4. Vendor-Neutral Architecture — Design a cloud-agnostic ML architecture
   using open-source tools with optional cloud-native substitutions.
5. Cost Optimization Audit — Profile an ML workload and calculate potential
   savings from spot instances, right-sizing, and mixed precision.
"""

import math
import random
import json
from datetime import datetime, timedelta


# ============================================================
# Exercise 1: SageMaker Training Job
# ============================================================

def exercise_1_sagemaker_training():
    """Simulate creating a SageMaker training job.

    Demonstrates:
    - Estimator configuration with instance type and spot training
    - Custom metric definitions for CloudWatch
    - Training with hyperparameters
    - Model artifact retrieval from S3
    """

    class SageMakerEstimator:
        """Simulated SageMaker Estimator."""

        def __init__(self, entry_point, framework, framework_version,
                     instance_type, instance_count, role, hyperparameters=None,
                     use_spot_instances=False, max_wait=None, max_run=None,
                     metric_definitions=None):
            self.entry_point = entry_point
            self.framework = framework
            self.framework_version = framework_version
            self.instance_type = instance_type
            self.instance_count = instance_count
            self.role = role
            self.hyperparameters = hyperparameters or {}
            self.use_spot_instances = use_spot_instances
            self.max_wait = max_wait
            self.max_run = max_run
            self.metric_definitions = metric_definitions or []
            self.training_job = None

        def fit(self, inputs, job_name=None):
            """Run training job (simulated)."""
            random.seed(42)

            self.training_job = {
                "job_name": job_name or f"training-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "status": "InProgress",
                "instance_type": self.instance_type,
                "instance_count": self.instance_count,
                "spot_training": self.use_spot_instances,
                "inputs": inputs,
                "start_time": datetime.now().isoformat(),
            }

            # Simulate training
            n_epochs = self.hyperparameters.get("epochs", 10)
            lr = self.hyperparameters.get("learning_rate", 0.01)

            metrics_history = []
            for epoch in range(1, n_epochs + 1):
                progress = 1 - math.exp(-0.3 * epoch)
                train_loss = max(0.01, 0.8 * (1 - progress) + random.gauss(0, 0.02))
                val_accuracy = min(0.99, 0.7 + 0.25 * progress + random.gauss(0, 0.01))

                metrics_history.append({
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "val_accuracy": round(val_accuracy, 4),
                })

            # Training results
            final_metrics = metrics_history[-1]
            training_time_min = random.uniform(15, 45) * n_epochs / 10

            # Spot savings
            on_demand_cost = training_time_min / 60 * self._get_instance_cost()
            spot_cost = on_demand_cost * 0.3 if self.use_spot_instances else on_demand_cost

            self.training_job.update({
                "status": "Completed",
                "end_time": datetime.now().isoformat(),
                "training_time_minutes": round(training_time_min, 1),
                "metrics_history": metrics_history,
                "final_metrics": final_metrics,
                "model_artifact": f"s3://sagemaker-us-east-1/output/{self.training_job['job_name']}/output/model.tar.gz",
                "billable_seconds": round(training_time_min * 60),
                "on_demand_cost": round(on_demand_cost, 2),
                "actual_cost": round(spot_cost, 2),
                "spot_savings": round(on_demand_cost - spot_cost, 2),
            })

            return self.training_job

        def _get_instance_cost(self):
            """Hourly cost by instance type (approximate)."""
            costs = {
                "ml.m5.xlarge": 0.23,
                "ml.m5.2xlarge": 0.46,
                "ml.c5.4xlarge": 0.68,
                "ml.p3.2xlarge": 3.06,
                "ml.p3.8xlarge": 12.24,
                "ml.g4dn.xlarge": 0.52,
                "ml.g5.xlarge": 1.01,
            }
            return costs.get(self.instance_type, 1.0)

    # --- Configure and run training ---
    print("SageMaker Training Job")
    print("=" * 60)

    metric_definitions = [
        {"Name": "train:loss", "Regex": r"train_loss: ([0-9.]+)"},
        {"Name": "val:accuracy", "Regex": r"val_accuracy: ([0-9.]+)"},
    ]

    estimator = SageMakerEstimator(
        entry_point="train.py",
        framework="sklearn",
        framework_version="1.4-1",
        instance_type="ml.m5.2xlarge",
        instance_count=1,
        role="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
        hyperparameters={
            "n_estimators": 100,
            "max_depth": 10,
            "learning_rate": 0.01,
            "epochs": 20,
        },
        use_spot_instances=True,
        max_wait=7200,  # 2 hours max wait for spot
        max_run=3600,   # 1 hour max training time
        metric_definitions=metric_definitions,
    )

    print(f"\n  Configuration:")
    print(f"    Instance: {estimator.instance_type} x{estimator.instance_count}")
    print(f"    Spot: {estimator.use_spot_instances}")
    print(f"    Framework: {estimator.framework} {estimator.framework_version}")
    print(f"    Hyperparameters: {json.dumps(estimator.hyperparameters, indent=4)}")

    job = estimator.fit(
        {"training": "s3://ml-data/train.csv", "validation": "s3://ml-data/val.csv"},
        job_name="churn-sklearn-spot-001",
    )

    print(f"\n  Training Results:")
    print(f"    Job: {job['job_name']}")
    print(f"    Status: {job['status']}")
    print(f"    Duration: {job['training_time_minutes']:.1f} minutes")
    print(f"    Final loss: {job['final_metrics']['train_loss']:.4f}")
    print(f"    Final accuracy: {job['final_metrics']['val_accuracy']:.4f}")
    print(f"\n  Cost:")
    print(f"    On-demand: ${job['on_demand_cost']:.2f}")
    print(f"    Spot:      ${job['actual_cost']:.2f}")
    print(f"    Savings:   ${job['spot_savings']:.2f} ({job['spot_savings']/max(job['on_demand_cost'], 0.01):.0%})")
    print(f"\n  Model artifact: {job['model_artifact']}")

    return estimator


# ============================================================
# Exercise 2: Vertex AI Pipeline
# ============================================================

def exercise_2_vertex_pipeline():
    """Simulate a 3-step Vertex AI pipeline with conditional deployment.

    Pipeline structure:
    preprocess -> train -> evaluate -> [conditional] deploy
    """

    class PipelineComponent:
        def __init__(self, name, func):
            self.name = name
            self.func = func
            self.outputs = {}

        def __call__(self, **kwargs):
            self.outputs = self.func(**kwargs)
            return self.outputs

    def preprocess(data_uri, test_ratio=0.2):
        """Preprocess component."""
        random.seed(42)
        n = 1000
        data = []
        for _ in range(n):
            x = [random.gauss(0, 1) for _ in range(5)]
            y = 1 if sum(w * xi for w, xi in zip([0.5, -0.3, 0.8, 0.1, -0.6], x)) > 0 else 0
            data.append((x, y))
        split = int(n * (1 - test_ratio))
        return {
            "train_data": data[:split],
            "test_data": data[split:],
            "n_train": split,
            "n_test": n - split,
        }

    def train(train_data, n_estimators=100, max_depth=10):
        """Train component."""
        w = [0.0] * 5
        b = 0.0
        for _ in range(100):
            for x, y in train_data:
                z = max(-500, min(500, sum(wi * xi for wi, xi in zip(w, x)) + b))
                p = 1 / (1 + math.exp(-z))
                e = p - y
                for j in range(5):
                    w[j] -= 0.01 * (e * x[j] + 0.001 * w[j])
                b -= 0.01 * e
        return {"weights": w, "bias": b}

    def evaluate(model, test_data):
        """Evaluate component."""
        w, b = model["weights"], model["bias"]
        tp = fp = tn = fn = 0
        for x, y in test_data:
            z = max(-500, min(500, sum(wi * xi for wi, xi in zip(w, x)) + b))
            pred = 1 if (1 / (1 + math.exp(-z))) >= 0.5 else 0
            if pred == 1 and y == 1: tp += 1
            elif pred == 1 and y == 0: fp += 1
            elif pred == 0 and y == 0: tn += 1
            else: fn += 1
        acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        return {"accuracy": round(acc, 4)}

    # --- Run pipeline ---
    print("Vertex AI Pipeline")
    print("=" * 60)

    accuracy_threshold = 0.80

    print(f"\n  Pipeline: preprocess -> train -> evaluate -> [conditional deploy]")
    print(f"  Deployment threshold: accuracy >= {accuracy_threshold}")

    # Step 1: Preprocess
    print(f"\n  Step 1: Preprocess")
    prep_result = preprocess("gs://ml-data/raw/", test_ratio=0.2)
    print(f"    Train: {prep_result['n_train']}, Test: {prep_result['n_test']}")

    # Step 2: Train
    print(f"\n  Step 2: Train")
    model = train(prep_result["train_data"])
    print(f"    Model trained (5 weights + bias)")

    # Step 3: Evaluate
    print(f"\n  Step 3: Evaluate")
    eval_result = evaluate(model, prep_result["test_data"])
    print(f"    Accuracy: {eval_result['accuracy']:.4f}")

    # Conditional deployment
    print(f"\n  Step 4: Conditional Deployment")
    if eval_result["accuracy"] >= accuracy_threshold:
        print(f"    Accuracy {eval_result['accuracy']:.4f} >= {accuracy_threshold}")
        print(f"    -> DEPLOYING to endpoint: projects/my-project/locations/"
              f"us-central1/endpoints/churn-predictor")

        endpoint_config = {
            "endpoint": "churn-predictor",
            "machine_type": "n1-standard-4",
            "min_replicas": 1,
            "max_replicas": 5,
            "traffic_split": {"new_model": 100},
        }
        print(f"    Config: {json.dumps(endpoint_config, indent=4)}")
    else:
        print(f"    Accuracy {eval_result['accuracy']:.4f} < {accuracy_threshold}")
        print(f"    -> SKIPPING deployment")

    # Reusable components
    print(f"\n  Reusable Components:")
    reusable = [
        ("preprocess", "Any tabular data preprocessing with train/test split"),
        ("evaluate", "Generic evaluation for binary classification"),
    ]
    for comp, use_case in reusable:
        print(f"    - {comp}: {use_case}")

    # Pipeline YAML
    print(f"\n  Compiled Pipeline YAML (abbreviated):")
    pipeline_yaml = """
    pipelineSpec:
      components:
        preprocess: {inputDefinitions: {...}, outputDefinitions: {...}}
        train: {inputDefinitions: {...}, outputDefinitions: {...}}
        evaluate: {inputDefinitions: {...}, outputDefinitions: {...}}
      dag:
        tasks:
          preprocess-task: {componentRef: preprocess}
          train-task: {componentRef: train, dependentTasks: [preprocess-task]}
          evaluate-task: {componentRef: evaluate, dependentTasks: [train-task]}
          deploy-condition: {condition: metrics.accuracy >= 0.80}
    """
    print(pipeline_yaml)

    return eval_result


# ============================================================
# Exercise 3: Platform Comparison
# ============================================================

def exercise_3_platform_comparison():
    """Evaluate AWS, GCP, and Azure ML platforms for a hypothetical project."""

    print("Platform Comparison")
    print("=" * 60)

    # Project profile
    project = {
        "team_size": 8,
        "current_cloud": "AWS (primary), some GCP",
        "model_type": "Tabular classification + NLP text analysis",
        "scale": "~10K predictions/day, 50 training jobs/week",
        "compliance": "SOC2",
        "budget": "medium ($5K-10K/month for ML infra)",
    }

    print(f"\n  Project Profile:")
    for k, v in project.items():
        print(f"    {k}: {v}")

    # Feature matrix
    features = {
        "Training": {
            "AWS SageMaker": {"score": 9, "notes": "Built-in algorithms, spot training, distributed"},
            "Vertex AI": {"score": 8, "notes": "AutoML, custom training, TPU support"},
            "Azure ML": {"score": 8, "notes": "Designer UI, compute clusters, HyperDrive"},
        },
        "Experiment Tracking": {
            "AWS SageMaker": {"score": 7, "notes": "SageMaker Experiments (basic)"},
            "Vertex AI": {"score": 7, "notes": "Vertex AI Experiments (Vizier integration)"},
            "Azure ML": {"score": 8, "notes": "MLflow integration, rich UI"},
        },
        "Model Registry": {
            "AWS SageMaker": {"score": 8, "notes": "Model Registry with approval workflows"},
            "Vertex AI": {"score": 7, "notes": "Model Registry + Endpoints"},
            "Azure ML": {"score": 8, "notes": "Model Registry with CI/CD triggers"},
        },
        "Serving": {
            "AWS SageMaker": {"score": 9, "notes": "Real-time, batch, async, multi-model endpoints"},
            "Vertex AI": {"score": 8, "notes": "Online prediction, batch, custom containers"},
            "Azure ML": {"score": 8, "notes": "Managed endpoints, batch endpoints"},
        },
        "Pipeline Orchestration": {
            "AWS SageMaker": {"score": 7, "notes": "SageMaker Pipelines (limited)"},
            "Vertex AI": {"score": 9, "notes": "KFP v2 native, excellent caching"},
            "Azure ML": {"score": 8, "notes": "Designer + SDK pipelines"},
        },
        "Feature Store": {
            "AWS SageMaker": {"score": 8, "notes": "SageMaker Feature Store (online + offline)"},
            "Vertex AI": {"score": 8, "notes": "Vertex AI Feature Store"},
            "Azure ML": {"score": 6, "notes": "Limited (use Azure Synapse)"},
        },
        "Monitoring": {
            "AWS SageMaker": {"score": 8, "notes": "Model Monitor, data quality, bias detection"},
            "Vertex AI": {"score": 7, "notes": "Model Monitoring (drift detection)"},
            "Azure ML": {"score": 7, "notes": "Data collector, monitoring"},
        },
        "Cost (per $1K budget)": {
            "AWS SageMaker": {"score": 7, "notes": "Spot discounts, reserved instances"},
            "Vertex AI": {"score": 8, "notes": "Preemptible VMs, sustained use discounts"},
            "Azure ML": {"score": 7, "notes": "Spot VMs, reserved instances"},
        },
    }

    # Display comparison
    print(f"\n  Feature Matrix (1-10 scale):")
    print(f"  {'Feature':<25s} {'SageMaker':>10s} {'Vertex AI':>10s} {'Azure ML':>10s}")
    print(f"  {'-'*55}")

    totals = {"AWS SageMaker": 0, "Vertex AI": 0, "Azure ML": 0}
    for feature, platforms in features.items():
        scores = []
        for platform in ["AWS SageMaker", "Vertex AI", "Azure ML"]:
            score = platforms[platform]["score"]
            totals[platform] += score
            scores.append(f"{score:>10d}")
        print(f"  {feature:<25s} {''.join(scores)}")

    print(f"  {'-'*55}")
    print(f"  {'TOTAL':<25s}", end="")
    for platform in ["AWS SageMaker", "Vertex AI", "Azure ML"]:
        print(f"{totals[platform]:>10d}", end="")
    print()

    # Cost estimation
    print(f"\n  Monthly Cost Estimate:")
    print(f"  {'Component':<30s} {'SageMaker':>10s} {'Vertex AI':>10s} {'Azure ML':>10s}")
    print(f"  {'-'*60}")
    costs = {
        "Training (50 jobs/week)": [1800, 1500, 1700],
        "Serving (10K pred/day)": [800, 700, 750],
        "Storage (S3/GCS/Blob)": [200, 180, 190],
        "Feature Store": [400, 350, 500],
        "Monitoring": [300, 250, 280],
    }
    totals = [0, 0, 0]
    for component, vals in costs.items():
        print(f"  {component:<30s}", end="")
        for i, v in enumerate(vals):
            print(f"{'$' + str(v):>10s}", end="")
            totals[i] += v
        print()
    print(f"  {'-'*60}")
    print(f"  {'TOTAL':<30s}", end="")
    for t in totals:
        print(f"{'$' + str(t):>10s}", end="")
    print()

    # Recommendation
    print(f"\n  Recommendation:")
    print(f"    Given: AWS primary cloud, SOC2 compliance, medium budget")
    print(f"    Primary: AWS SageMaker")
    print(f"      - Best ecosystem integration with existing AWS infrastructure")
    print(f"      - Strong serving capabilities for 10K pred/day scale")
    print(f"      - Spot training for cost optimization")
    print(f"    Alternative: Vertex AI (if migrating pipelines, KFP v2 is superior)")

    return features


# ============================================================
# Exercise 4: Vendor-Neutral Architecture
# ============================================================

def exercise_4_vendor_neutral():
    """Design a vendor-neutral ML architecture."""

    print("Vendor-Neutral ML Architecture")
    print("=" * 60)

    layers = {
        "Experiment Tracking": {
            "open_source": "MLflow",
            "cloud_substitute": {
                "AWS": "SageMaker Experiments",
                "GCP": "Vertex AI Experiments",
                "Azure": "Azure ML (MLflow-compatible)",
            },
            "why_open": "MLflow is the de facto standard; all clouds support it natively",
        },
        "Pipeline Orchestration": {
            "open_source": "Kubeflow Pipelines (KFP v2)",
            "cloud_substitute": {
                "AWS": "SageMaker Pipelines",
                "GCP": "Vertex AI Pipelines (KFP v2 native)",
                "Azure": "Azure ML Pipelines",
            },
            "why_open": "KFP runs on any K8s cluster; GCP has native support",
        },
        "Feature Store": {
            "open_source": "Feast",
            "cloud_substitute": {
                "AWS": "SageMaker Feature Store",
                "GCP": "Vertex AI Feature Store",
                "Azure": "Azure Synapse + custom",
            },
            "why_open": "Feast supports multiple backends (Redis, BigQuery, Redshift)",
        },
        "Model Serving": {
            "open_source": "Triton Inference Server / TorchServe",
            "cloud_substitute": {
                "AWS": "SageMaker Endpoints",
                "GCP": "Vertex AI Endpoints",
                "Azure": "Azure ML Managed Endpoints",
            },
            "why_open": "Triton supports multi-framework; runs anywhere with Docker",
        },
        "Monitoring": {
            "open_source": "Evidently AI + Prometheus/Grafana",
            "cloud_substitute": {
                "AWS": "SageMaker Model Monitor",
                "GCP": "Vertex AI Model Monitoring",
                "Azure": "Azure ML Data Collector",
            },
            "why_open": "Evidently provides ML-specific drift detection; Prometheus is universal",
        },
        "Model Registry": {
            "open_source": "MLflow Model Registry",
            "cloud_substitute": {
                "AWS": "SageMaker Model Registry",
                "GCP": "Vertex AI Model Registry",
                "Azure": "Azure ML Model Registry",
            },
            "why_open": "MLflow registry integrates with MLflow tracking seamlessly",
        },
        "Data Versioning": {
            "open_source": "DVC",
            "cloud_substitute": {
                "AWS": "S3 versioning + SageMaker Data Wrangler",
                "GCP": "GCS versioning + Vertex AI Datasets",
                "Azure": "Azure Blob versioning",
            },
            "why_open": "DVC provides Git-native workflow for data; works with any storage",
        },
    }

    for layer_name, config in layers.items():
        print(f"\n  {layer_name}:")
        print(f"    Open Source: {config['open_source']}")
        print(f"    Cloud Options:")
        for cloud, service in config["cloud_substitute"].items():
            print(f"      {cloud}: {service}")
        print(f"    Rationale: {config['why_open']}")

    print(f"\n  Architecture Diagram:")
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │                    Developer Workflow                     │
    │  Git + DVC ──> MLflow Tracking ──> Experiment Analysis   │
    └──────────────────────┬──────────────────────────────────┘
                           │
    ┌──────────────────────┴──────────────────────────────────┐
    │                  Pipeline Layer (KFP v2)                  │
    │  Data Prep ──> Feature Eng ──> Training ──> Evaluation   │
    │                     │                         │           │
    │                 Feast (Features)        MLflow (Registry)  │
    └──────────────────────┬──────────────────────────────────┘
                           │
    ┌──────────────────────┴──────────────────────────────────┐
    │                  Serving Layer                            │
    │  Triton/TorchServe ──> Load Balancer ──> API Gateway     │
    └──────────────────────┬──────────────────────────────────┘
                           │
    ┌──────────────────────┴──────────────────────────────────┐
    │               Monitoring Layer                            │
    │  Evidently (Drift) ──> Prometheus ──> Grafana (Alerts)   │
    └─────────────────────────────────────────────────────────┘
    """)

    return layers


# ============================================================
# Exercise 5: Cost Optimization Audit
# ============================================================

def exercise_5_cost_optimization():
    """Perform a cost optimization audit for ML workloads."""

    print("Cost Optimization Audit")
    print("=" * 60)

    # --- Current workload profile ---
    workload = {
        "training": {
            "instance_type": "ml.p3.2xlarge",
            "hourly_cost": 3.06,
            "hours_per_job": 2,
            "jobs_per_week": 50,
            "gpu_utilization_pct": 45,
        },
        "serving": {
            "instance_type": "ml.m5.2xlarge",
            "hourly_cost": 0.46,
            "instances": 3,
            "hours_per_day": 24,
            "avg_cpu_utilization_pct": 30,
        },
        "notebooks": {
            "instance_type": "ml.m5.xlarge",
            "hourly_cost": 0.23,
            "instances": 8,
            "hours_per_day": 10,
        },
        "storage": {
            "s3_gb": 500,
            "cost_per_gb": 0.023,
        },
    }

    # --- Calculate current costs ---
    print(f"\n  Current Monthly Cost Breakdown:")
    print(f"  {'Component':<30s} {'Monthly Cost':>12s} {'Details':>30s}")
    print(f"  {'-'*72}")

    training_cost = (workload["training"]["hourly_cost"] *
                     workload["training"]["hours_per_job"] *
                     workload["training"]["jobs_per_week"] * 4.3)
    serving_cost = (workload["serving"]["hourly_cost"] *
                    workload["serving"]["instances"] *
                    workload["serving"]["hours_per_day"] * 30)
    notebook_cost = (workload["notebooks"]["hourly_cost"] *
                     workload["notebooks"]["instances"] *
                     workload["notebooks"]["hours_per_day"] * 22)
    storage_cost = workload["storage"]["s3_gb"] * workload["storage"]["cost_per_gb"]

    total_current = training_cost + serving_cost + notebook_cost + storage_cost

    costs = [
        ("Training (GPU)", training_cost, f"{workload['training']['jobs_per_week']} jobs/week"),
        ("Serving (24/7)", serving_cost, f"{workload['serving']['instances']} instances"),
        ("Notebooks", notebook_cost, f"{workload['notebooks']['instances']} users"),
        ("Storage (S3)", storage_cost, f"{workload['storage']['s3_gb']} GB"),
    ]

    for name, cost, details in costs:
        print(f"  {name:<30s} ${cost:>11,.2f} {details:>30s}")
    print(f"  {'-'*72}")
    print(f"  {'TOTAL':<30s} ${total_current:>11,.2f}")

    # --- Optimization opportunities ---
    print(f"\n\n  Optimization Opportunities:")
    print(f"  {'='*60}")

    optimizations = []

    # 1. Spot instances for training
    spot_saving = training_cost * 0.70  # 70% discount
    optimizations.append({
        "name": "Spot instances for training",
        "current": training_cost,
        "optimized": training_cost * 0.30,
        "savings": spot_saving,
        "risk": "Job interruption (mitigate with checkpointing)",
        "effort": "Low (1-2 days)",
    })

    # 2. Right-sizing serving instances
    rightsize_saving = serving_cost * 0.40  # Drop to smaller instances
    optimizations.append({
        "name": "Right-size serving (m5.2xl -> m5.xl)",
        "current": serving_cost,
        "optimized": serving_cost * 0.60,
        "savings": rightsize_saving,
        "risk": "May need auto-scaling for peak loads",
        "effort": "Low (1 day)",
    })

    # 3. Auto-shutdown notebooks
    auto_shutdown_saving = notebook_cost * 0.30
    optimizations.append({
        "name": "Auto-shutdown idle notebooks",
        "current": notebook_cost,
        "optimized": notebook_cost * 0.70,
        "savings": auto_shutdown_saving,
        "risk": "None (lifecycle configs handle this)",
        "effort": "Low (configuration change)",
    })

    # 4. Mixed precision training
    mixed_prec_saving = training_cost * 0.30 * 0.40  # 40% faster with mixed precision
    optimizations.append({
        "name": "Mixed precision training (FP16)",
        "current": training_cost * 0.30,  # After spot
        "optimized": training_cost * 0.30 * 0.60,
        "savings": mixed_prec_saving,
        "risk": "Minimal accuracy impact (< 0.1%)",
        "effort": "Medium (code changes + validation)",
    })

    # 5. S3 Intelligent-Tiering
    storage_saving = storage_cost * 0.30
    optimizations.append({
        "name": "S3 Intelligent-Tiering",
        "current": storage_cost,
        "optimized": storage_cost * 0.70,
        "savings": storage_saving,
        "risk": "None (automatic)",
        "effort": "Low (bucket policy)",
    })

    total_savings = sum(o["savings"] for o in optimizations)

    for i, opt in enumerate(optimizations, 1):
        pct = opt["savings"] / opt["current"] * 100 if opt["current"] > 0 else 0
        print(f"\n  {i}. {opt['name']}")
        print(f"     Savings: ${opt['savings']:,.2f}/month ({pct:.0f}%)")
        print(f"     Risk: {opt['risk']}")
        print(f"     Effort: {opt['effort']}")

    # --- 90-day plan ---
    print(f"\n\n  90-Day Cost Optimization Plan:")
    print(f"  {'='*60}")

    phases = [
        {
            "name": "Week 1-2: Quick Wins",
            "actions": [
                "Enable auto-shutdown for idle notebooks",
                "Enable S3 Intelligent-Tiering",
                "Switch training jobs to spot instances",
            ],
            "expected_savings": auto_shutdown_saving + storage_saving + spot_saving,
        },
        {
            "name": "Week 3-4: Right-Sizing",
            "actions": [
                "Analyze serving metrics (CPU/memory utilization)",
                "Right-size serving instances",
                "Set up auto-scaling for serving endpoints",
            ],
            "expected_savings": rightsize_saving,
        },
        {
            "name": "Week 5-8: Code Optimization",
            "actions": [
                "Implement mixed precision training",
                "Validate model accuracy with FP16",
                "Optimize data loading pipeline",
            ],
            "expected_savings": mixed_prec_saving,
        },
        {
            "name": "Week 9-12: Monitoring & Governance",
            "actions": [
                "Set up cost alerts and dashboards",
                "Implement tagging strategy for cost attribution",
                "Monthly cost review process",
            ],
            "expected_savings": 0,
        },
    ]

    cumulative = 0
    for phase in phases:
        cumulative += phase["expected_savings"]
        print(f"\n  {phase['name']}")
        for action in phase["actions"]:
            print(f"    - {action}")
        if phase["expected_savings"] > 0:
            print(f"    Expected savings: ${phase['expected_savings']:,.2f}/month")
        print(f"    Cumulative: ${cumulative:,.2f}/month")

    print(f"\n  Summary:")
    print(f"    Current monthly cost:  ${total_current:>10,.2f}")
    print(f"    Optimized monthly cost: ${total_current - total_savings:>10,.2f}")
    print(f"    Total monthly savings: ${total_savings:>10,.2f} "
          f"({total_savings/total_current:.0%})")
    print(f"    Annual savings:        ${total_savings * 12:>10,.2f}")

    return optimizations


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Exercise 1: SageMaker Training Job")
    print("=" * 60)
    exercise_1_sagemaker_training()

    print("\n\n")
    print("Exercise 2: Vertex AI Pipeline")
    print("=" * 60)
    exercise_2_vertex_pipeline()

    print("\n\n")
    print("Exercise 3: Platform Comparison")
    print("=" * 60)
    exercise_3_platform_comparison()

    print("\n\n")
    print("Exercise 4: Vendor-Neutral Architecture")
    print("=" * 60)
    exercise_4_vendor_neutral()

    print("\n\n")
    print("Exercise 5: Cost Optimization Audit")
    print("=" * 60)
    exercise_5_cost_optimization()
