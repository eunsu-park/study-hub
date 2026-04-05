[← Previous: 16. Model Testing and Validation](16_Model_Testing_and_Validation.md) | [Next: Overview →](00_Overview.md)

# Cloud MLOps Platforms

## Learning Objectives

1. Understand why cloud-native MLOps platforms accelerate the path from experiment to production
2. Navigate the AWS SageMaker ecosystem: Studio, Pipelines, Feature Store, and Clarify
3. Build and deploy ML workflows on Google Vertex AI with Pipelines and Model Monitoring
4. Use Azure Machine Learning for training, deployment, and responsible AI analysis
5. Compare the three major platforms across key dimensions using a structured feature matrix
6. Evaluate multi-cloud and vendor-neutral strategies using MLflow and Kubeflow as abstraction layers
7. Apply cost optimization strategies to reduce cloud ML spend without sacrificing capability

---

## Overview

Building an ML platform from scratch requires assembling dozens of components: experiment tracking, pipeline orchestration, model serving, feature stores, monitoring, and more. Cloud providers offer managed versions of these components that integrate tightly with their compute, storage, and networking infrastructure. The trade-off is clear: faster time-to-production at the cost of vendor dependency.

This lesson provides a practical tour of the three major cloud MLOps platforms — AWS SageMaker, Google Vertex AI, and Azure Machine Learning. Rather than exhaustive documentation (which changes rapidly), we focus on architectural patterns, SDK usage, and the decision frameworks that help you choose the right platform (or avoid choosing one at all).

> **Analogy**: Choosing a cloud MLOps platform is like choosing an airline. All three get you to the destination (model in production), but they differ in comfort (UX and developer experience), routes (integration ecosystem), loyalty programs (ecosystem lock-in), and pricing (pay-per-use vs reserved capacity). Some travelers prefer one airline's lounge; others prioritize the cheapest ticket. The best choice depends on where you're starting from and how often you fly.

---

## 1. Why Cloud-Native MLOps

### 1.1 The Build vs Buy Decision

```python
"""
Build vs Buy Decision Framework for ML Platforms:

  Self-Managed (Open Source):
  ┌──────────────────────────────────────────────────────────────┐
  │  MLflow + Kubeflow + Feast + Evidently + Triton + Airflow    │
  │                                                              │
  │  Pros:                          Cons:                        │
  │  ✓ No vendor lock-in            ✗ Complex to integrate       │
  │  ✓ Full customization            ✗ Operational overhead       │
  │  ✓ Cost control at scale         ✗ Slower time-to-value      │
  │  ✓ Multi-cloud portable          ✗ Need ML platform team     │
  └──────────────────────────────────────────────────────────────┘

  Cloud-Managed:
  ┌──────────────────────────────────────────────────────────────┐
  │  SageMaker / Vertex AI / Azure ML                            │
  │                                                              │
  │  Pros:                          Cons:                        │
  │  ✓ Integrated end-to-end        ✗ Vendor lock-in             │
  │  ✓ Managed infrastructure       ✗ Higher unit costs          │
  │  ✓ Built-in security/compliance ✗ Less customizable          │
  │  ✓ Faster time-to-value         ✗ Platform-specific APIs     │
  └──────────────────────────────────────────────────────────────┘

  Decision criteria:
  ┌───────────────────────┬──────────────┬──────────────────┐
  │ Factor                │ Self-Managed │ Cloud-Managed     │
  ├───────────────────────┼──────────────┼──────────────────┤
  │ Team size             │ > 5 ML eng.  │ Any size          │
  │ Time to production    │ 3-6 months   │ 2-6 weeks         │
  │ Annual ML spend       │ > $500K      │ Any               │
  │ Multi-cloud required? │ Yes          │ Not critical      │
  │ Custom hardware (TPU) │ Specific     │ Provider has it   │
  │ Compliance needs      │ Sovereign    │ Provider certified│
  └───────────────────────┴──────────────┴──────────────────┘
"""
```

### 1.2 Cloud MLOps Architecture Patterns

```python
"""
Common architecture pattern across all cloud MLOps platforms:

  ┌──────────────────────────────────────────────────────────────────┐
  │                     Cloud MLOps Platform                         │
  │                                                                  │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐│
  │  │ Notebook  │  │ Pipeline │  │ Model    │  │ Serving          ││
  │  │ / IDE     │  │ Orchest. │  │ Registry │  │ (Endpoint)       ││
  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘│
  │                                                                  │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐│
  │  │ Feature  │  │ Experiment│  │ Monitoring│ │ Responsible AI   ││
  │  │ Store    │  │ Tracking │  │ & Alerts │  │ (Bias/Explain)   ││
  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘│
  │                                                                  │
  │  ─────────────── Shared Infrastructure ─────────────────────── │
  │  │ Compute (CPU/GPU/TPU) │ Storage (S3/GCS/Blob) │ IAM/VPC │  │
  └──────────────────────────────────────────────────────────────────┘

  Key principle: each component is a managed service that handles
  provisioning, scaling, patching, and monitoring automatically.
"""
```

---

## 2. AWS SageMaker

### 2.1 SageMaker Ecosystem Overview

```python
"""
AWS SageMaker: the most comprehensive cloud ML platform.

  SageMaker Ecosystem (as of 2025):
  ┌──────────────────────────────────────────────────────────────────┐
  │                        SageMaker Studio                          │
  │  (Unified IDE: notebooks, experiments, pipelines, endpoints)     │
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  Development:                  Operations:                       │
  │  ├─ Studio Notebooks           ├─ Pipelines (orchestration)      │
  │  ├─ Processing (data prep)     ├─ Model Registry                 │
  │  ├─ Training (distributed)     ├─ Endpoints (real-time/batch)    │
  │  ├─ Tuning (HPO)               ├─ Model Monitor                  │
  │  └─ Autopilot (AutoML)         └─ Edge Manager                   │
  │                                                                  │
  │  Data & Features:              Responsible AI:                   │
  │  ├─ Feature Store              ├─ Clarify (bias detection)       │
  │  ├─ Data Wrangler              ├─ Debugger (training issues)     │
  │  └─ Ground Truth (labeling)    └─ Model Dashboard                │
  │                                                                  │
  │  Infrastructure:                                                 │
  │  ├─ ml.* instance types (CPU, GPU, Inf, Trn)                     │
  │  ├─ Spot Training (up to 90% cost reduction)                     │
  │  └─ Multi-model / multi-container endpoints                      │
  └──────────────────────────────────────────────────────────────────┘

  Key differentiators:
    - Deepest integration with AWS services (S3, IAM, CloudWatch, Lambda)
    - Most instance type options (including custom silicon: Inferentia, Trainium)
    - Largest marketplace of built-in algorithms and pre-trained models
    - Mature model hosting with auto-scaling and A/B testing
"""
```

### 2.2 SageMaker Training Job

```python
"""
SageMaker Training: managed training with automatic instance provisioning.

Why managed training?
  - No need to provision EC2 instances manually
  - Automatic shutdown after training completes (no idle costs)
  - Built-in distributed training support
  - Spot instance support for cost reduction
  - Automatic metric logging to CloudWatch
"""
import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker.inputs import TrainingInput


def create_training_job():
    """Create a SageMaker training job.

    Architecture:
      1. Upload training data to S3
      2. SageMaker provisions instance(s)
      3. Pulls your training script + data
      4. Runs training, logs metrics
      5. Saves model artifact to S3
      6. Terminates instance
    """
    # Why use a SageMaker session?
    #   Manages AWS credentials, default bucket, region
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()  # IAM role with S3/ECR permissions

    # Define the estimator — tells SageMaker HOW to train
    # Why SKLearn estimator instead of a generic Estimator?
    #   SKLearn pulls a pre-built AWS container (no Dockerfile needed);
    #   it also auto-injects the framework version into CloudWatch metadata,
    #   making it easy to reproduce results months later
    estimator = SKLearn(
        entry_point="train.py",              # Your training script
        source_dir="src/",                   # Directory with dependencies
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",        # 4 vCPU, 16 GB RAM
        framework_version="1.2-1",           # scikit-learn version
        py_version="py3",
        # Why output_path to S3 instead of local disk?
        #   The training instance is ephemeral; S3 is the only durable
        #   storage that survives instance termination
        output_path=f"s3://{session.default_bucket()}/models/",

        # Why hyperparameters here instead of in code?
        #   - Tracked by SageMaker Experiments automatically
        #   - Can be tuned with HyperparameterTuner
        #   - Visible in Studio UI
        hyperparameters={
            "n_estimators": 100,
            "max_depth": 10,
            "test_size": 0.2,
        },

        # Spot training — use unused EC2 capacity at up to 90% discount
        # Why not always use spot?
        #   - Can be interrupted (SageMaker handles checkpointing)
        #   - Good for training; risky for time-sensitive jobs
        use_spot_instances=True,
        # Why max_wait > max_run? The gap (7200 - 3600 = 3600 s) is the
        #   maximum time SageMaker will queue waiting for a spot instance
        #   before failing; set it large enough to ride out brief capacity crunches
        max_wait=7200,                       # Max wait for spot (seconds)
        max_run=3600,                        # Max training time (seconds)
    )

    # Define input data channels
    # Why channels? SageMaker mounts data to specific paths in the container
    train_input = TrainingInput(
        s3_data="s3://my-bucket/data/train/",
        # Why specify content_type? SageMaker uses it to pick the correct
        #   data-loading optimisation (pipe mode vs file mode); CSV triggers
        #   SageMaker's record-IO optimised streaming for large datasets
        content_type="text/csv",
    )
    test_input = TrainingInput(
        s3_data="s3://my-bucket/data/test/",
        content_type="text/csv",
    )

    # Launch training — this is asynchronous by default
    # Why pass a job_name? Without it SageMaker auto-generates a UUID;
    #   a meaningful name makes CloudWatch log filtering and cost allocation
    #   tags far easier to manage across dozens of experiments
    estimator.fit(
        inputs={"train": train_input, "test": test_input},
        job_name="sklearn-classifier-v1",
        wait=True,  # Set False for async (check status in Studio)
    )

    return estimator
```

### 2.3 SageMaker Pipelines

```python
"""
SageMaker Pipelines: native ML workflow orchestration.

Why SageMaker Pipelines over Airflow/Step Functions?
  - Native integration with SageMaker steps (no glue code)
  - Built-in caching (skip unchanged steps)
  - Visual DAG in Studio
  - Parameterized pipelines for reuse
  - Direct integration with Model Registry

Pipeline Architecture:
  ┌────────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐
  │ Processing │──▶│ Training │──▶│ Evaluate  │──▶│ Register │
  │ (data prep)│   │          │   │ (metrics) │   │ (if good)│
  └────────────┘   └──────────┘   └───────────┘   └──────────┘
                                        │
                                  ┌─────▼─────┐
                                  │ Condition  │
                                  │ (acc>0.85?)│
                                  └─────┬──────┘
                                  YES   │   NO
                                  ┌─────▼─────┐
                                  │  Deploy    │
                                  │  Endpoint  │
                                  └────────────┘
"""
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.processing import ScriptProcessor


def create_sagemaker_pipeline(role, session):
    """Create a SageMaker Pipeline with conditional model registration.

    This demonstrates the core pipeline pattern:
      Data prep → Train → Evaluate → Conditionally register & deploy
    """
    # Pipeline parameters — change these without modifying code
    # Why parameters? Enable reuse across environments (dev/staging/prod)
    input_data = ParameterString(
        name="InputData",
        default_value="s3://my-bucket/data/raw/",
    )
    accuracy_threshold = ParameterFloat(
        name="AccuracyThreshold",
        default_value=0.85,
    )

    # Step 1: Data Processing
    # Why a separate processing step?
    #   - Decouples data prep from training (different compute needs)
    #   - Processing results are cached (reuse if data hasn't changed)
    #   - Can run on cheaper instances than training
    processor = ScriptProcessor(
        role=role,
        image_uri=sagemaker.image_uris.retrieve("sklearn", session.boto_region_name),
        instance_count=1,
        instance_type="ml.m5.xlarge",
    )

    processing_step = ProcessingStep(
        name="PreprocessData",
        processor=processor,
        code="src/preprocess.py",
    )

    # Step 2: Training
    estimator = SKLearn(
        entry_point="train.py",
        source_dir="src/",
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        framework_version="1.2-1",
    )

    training_step = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        depends_on=[processing_step],
    )

    # Step 3: Conditional registration
    # Why conditional? Only register models that meet quality thresholds
    # This prevents bad models from entering the registry
    condition = ConditionGreaterThanOrEqualTo(
        left=training_step.properties.FinalMetricDataList[0].Value,
        right=accuracy_threshold,
    )

    # Why RegisterModel instead of a custom deploy step?
    #   RegisterModel writes versioned metadata (metrics, lineage, approval status)
    #   into the Model Registry; a human or CI gate can then approve before
    #   the model is ever promoted to a live endpoint
    register_step = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        # Why list inference and transform instances separately?
        #   Real-time endpoints and batch transform have different latency/
        #   throughput profiles; specifying both lets the registry surface
        #   the right deployment options in Studio
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        # Why a model package group? Groups bundle all versions of one model
        #   family so Studio can compare v1/v2/v3 side-by-side before approval
        model_package_group_name="my-model-group",
    )

    condition_step = ConditionStep(
        name="CheckAccuracy",
        conditions=[condition],
        if_steps=[register_step],
        # Why empty else_steps rather than a failure step?
        #   Failing the pipeline would block retries; silent skip lets the
        #   pipeline complete cleanly so its execution history is preserved
        else_steps=[],  # Do nothing if accuracy is below threshold
    )

    # Assemble pipeline
    # Why declare parameters at the Pipeline level?
    #   SageMaker surfaces them in Studio's "Run pipeline" dialog, enabling
    #   non-engineers to trigger experiments with different thresholds without
    #   touching code or redeploying the pipeline definition
    pipeline = Pipeline(
        name="ml-training-pipeline",
        parameters=[input_data, accuracy_threshold],
        steps=[processing_step, training_step, condition_step],
        sagemaker_session=session,
    )

    return pipeline
```

### 2.4 SageMaker Clarify (Bias and Explainability)

```python
"""
SageMaker Clarify: bias detection and model explainability.

Why Clarify?
  - Pre-training bias detection (data imbalances)
  - Post-training bias detection (model discrimination)
  - SHAP-based feature importance explanations
  - Integrates with Model Monitor for ongoing bias tracking

Bias Metrics Computed by Clarify:
  - Class Imbalance (CI): difference in label proportions
  - Difference in Proportions of Labels (DPL)
  - Disparate Impact (DI): ratio of positive outcome rates
  - Kullback-Leibler Divergence (KL): distribution difference
"""
from sagemaker.clarify import (
    SageMakerClarifyProcessor,
    DataConfig,
    BiasConfig,
    ModelConfig,
    SHAPConfig,
)


def run_bias_analysis(role, session, model_name, data_s3_uri):
    """Run SageMaker Clarify bias analysis.

    Why run bias analysis?
      - Legal compliance (EU AI Act, ECOA)
      - Detect discrimination before deployment
      - Generate explainability reports for model cards
    """
    # Why a dedicated Clarify processor instead of running bias checks in train.py?
    #   Clarify spins up an isolated compute job so bias analysis never shares
    #   memory or CPU with training; this also means it can be re-run on existing
    #   model artifacts without re-training
    clarify = SageMakerClarifyProcessor(
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        sagemaker_session=session,
    )

    # Data configuration — tells Clarify about your dataset structure
    data_config = DataConfig(
        s3_data_input_path=data_s3_uri,
        # Why a separate S3 output path for Clarify?
        #   Clarify produces large JSON reports; keeping them separate from model
        #   artifacts avoids polluting the model registry and simplifies audits
        s3_output_path=f"s3://{session.default_bucket()}/clarify-output/",
        label="target",                   # Target column name
        headers=["age", "income", "gender", "education", "target"],
        dataset_type="text/csv",
    )

    # Bias configuration — which attribute to check for bias
    # Why specify facet_name?
    #   Clarify computes bias metrics comparing the facet group
    #   (e.g., gender=Female) against the baseline (gender=Male)
    bias_config = BiasConfig(
        label_values_or_threshold=[1],    # Positive outcome label
        facet_name="gender",              # Protected attribute
        facet_values_or_threshold=["Female"],  # Disadvantaged group
    )

    # Model configuration for post-training bias
    # Why a separate ModelConfig with its own instance type?
    #   Clarify invokes the model endpoint to get predictions on perturbed inputs;
    #   a smaller instance (ml.m5.large vs xlarge) is sufficient for inference
    #   and keeps the analysis cost low
    model_config = ModelConfig(
        model_name=model_name,
        instance_count=1,
        instance_type="ml.m5.large",
        content_type="text/csv",
        accept_type="text/csv",
    )

    # Run pre-training bias analysis (data-only, no model needed)
    # Why run pre-training bias? Catching class imbalance before training is
    #   cheaper than fixing a discriminatory model after deployment
    clarify.run_pre_training_bias(
        data_config=data_config,
        data_bias_config=bias_config,
    )

    # Run post-training bias analysis (requires deployed model)
    clarify.run_post_training_bias(
        data_config=data_config,
        data_bias_config=bias_config,
        model_config=model_config,
    )

    # SHAP explainability — feature importance for individual predictions
    # Why SHAP? It provides consistent, theoretically-grounded explanations
    shap_config = SHAPConfig(
        # Why baseline=None? Clarify computes a mean baseline from the dataset,
        #   which is more representative than a hand-crafted zero vector and
        #   avoids introducing researcher bias into the explanation
        baseline=None,                    # Clarify auto-computes baseline
        num_samples=100,                  # Number of SHAP samples
        agg_method="mean_abs",            # Aggregation for global importance
    )

    clarify.run_explainability(
        data_config=data_config,
        model_config=model_config,
        explainability_config=shap_config,
    )
```

---

## 3. Google Vertex AI

### 3.1 Vertex AI Ecosystem Overview

```python
"""
Google Vertex AI: unified ML platform built on Google's AI infrastructure.

  Vertex AI Ecosystem (as of 2025):
  ┌──────────────────────────────────────────────────────────────────┐
  │                      Vertex AI Workbench                         │
  │  (Managed JupyterLab with BigQuery/GCS integration)              │
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  Development:                  Operations:                       │
  │  ├─ Custom Training (any fw)   ├─ Pipelines (KFP + TFX)         │
  │  ├─ AutoML (no-code)           ├─ Model Registry                 │
  │  ├─ Hyperparameter Tuning      ├─ Endpoints (online/batch)       │
  │  └─ Neural Architecture Search ├─ Model Monitoring               │
  │                                └─ Experiments (built-in tracking)│
  │                                                                  │
  │  Data & Features:              Responsible AI:                   │
  │  ├─ Feature Store              ├─ Explainable AI (feature attr.) │
  │  ├─ Managed Datasets           ├─ Model Evaluation               │
  │  └─ BigQuery ML integration    └─ What-If Tool                   │
  │                                                                  │
  │  Infrastructure:                                                 │
  │  ├─ TPU v5 / A100 / H100 GPUs                                   │
  │  ├─ Reduction Server (distributed training optimization)         │
  │  └─ Generative AI (Model Garden, Gemini API)                     │
  └──────────────────────────────────────────────────────────────────┘

  Key differentiators:
    - Native TPU support (best for large model training)
    - BigQuery ML: train models with SQL (no Python needed)
    - Generative AI integration (Gemini, PaLM, Imagen)
    - KFP-based Pipelines (portable to open-source Kubeflow)
    - Tight integration with BigQuery, Dataflow, Pub/Sub
"""
```

### 3.2 Vertex AI Custom Training

```python
"""
Vertex AI Custom Training: run any training code on managed infrastructure.

Why custom training over AutoML?
  - AutoML: fast prototyping, no ML expertise needed, limited customization
  - Custom: full control over architecture, training loop, dependencies
  - Use AutoML as a baseline, then beat it with custom training

Training options:
  1. Pre-built containers (TensorFlow, PyTorch, scikit-learn, XGBoost)
  2. Custom containers (bring your own Dockerfile)
  3. Python package (upload .tar.gz with setup.py)
"""
from google.cloud import aiplatform


def create_vertex_training_job():
    """Create a Vertex AI custom training job.

    Architecture:
      1. Package training code as a Python source distribution
      2. Vertex AI provisions machine(s) with specified accelerators
      3. Installs your package + dependencies
      4. Runs training, streams logs to Cloud Logging
      5. Saves artifacts to GCS
      6. Terminates machine(s)
    """
    # Initialize the Vertex AI SDK
    # Why project and location? Vertex AI resources are regional
    aiplatform.init(
        project="my-gcp-project",
        location="us-central1",              # Region with best GPU availability
        staging_bucket="gs://my-ml-bucket",  # GCS bucket for staging artifacts
    )

    # Define the training job
    # Why CustomTrainingJob over CustomContainerTrainingJob?
    #   Use CustomTrainingJob when using pre-built containers
    #   Use CustomContainerTrainingJob for custom Docker images
    job = aiplatform.CustomTrainingJob(
        display_name="sklearn-classifier-training",
        script_path="src/train.py",          # Entry point script
        # Why a Google-hosted container URI instead of Docker Hub?
        #   GCP containers are stored in Artifact Registry within the same
        #   network region, so pulls are faster and avoid egress costs;
        #   they also receive regular security patches from Google
        container_uri="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-3:latest",
        requirements=["pandas>=2.0", "scikit-learn>=1.3"],
        # Why specify model_serving_container separately from training container?
        #   Training and serving have different dependencies (e.g., no need
        #   for pandas or data loaders at serving time); a lean serving image
        #   reduces cold-start latency and attack surface
        model_serving_container_image_uri=(
            "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
        ),
    )

    # Run the training job
    # Why return a Model object?
    #   Vertex AI automatically creates a Model resource from training output.
    #   This Model can be directly deployed to an Endpoint.
    model = job.run(
        model_display_name="sklearn-classifier-v1",

        # Training arguments — passed to your script as command-line args
        # Why command-line args instead of environment variables?
        #   Args appear in Vertex AI job metadata and experiment comparisons;
        #   env vars are invisible to the platform's tracking layer
        args=[
            "--n-estimators", "100",
            "--max-depth", "10",
            "--test-size", "0.2",
        ],

        # Compute configuration
        # Why replica_count=1 here but not hardcoded to 1 in the function?
        #   Keeping it as a parameter makes it trivial to switch to multi-worker
        #   distributed training by bumping the count without changing the script
        replica_count=1,
        machine_type="n1-standard-8",       # 8 vCPU, 30 GB RAM
        # accelerator_type="NVIDIA_TESLA_T4",  # Uncomment for GPU
        # accelerator_count=1,

        # Why base_output_dir on GCS instead of local disk?
        #   Vertex AI training VMs are ephemeral; GCS is the only storage that
        #   persists after the job ends and is accessible by serving endpoints
        base_output_dir="gs://my-ml-bucket/models/",
    )

    return model
```

### 3.3 Vertex AI Pipelines

```python
"""
Vertex AI Pipelines: serverless ML pipeline orchestration.

Why Vertex AI Pipelines?
  - Based on Kubeflow Pipelines (KFP) v2 SDK — portable
  - Serverless: no cluster to manage (unlike self-hosted Kubeflow)
  - Cached steps (skip unchanged components)
  - Visual DAG in Vertex AI console
  - Direct integration with all Vertex AI services

Portability benefit:
  Pipelines written with KFP SDK can run on:
    1. Vertex AI Pipelines (managed)
    2. Self-hosted Kubeflow Pipelines (open source)
    3. Any KFP-compatible runner
"""
from kfp.v2 import dsl, compiler
from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics
from google.cloud import aiplatform


@component(
    base_image="python:3.11",
    # Why packages_to_install instead of a pre-built image?
    #   KFP dynamically pip-installs into the base image at runtime;
    #   this avoids maintaining a separate Dockerfile for lightweight components
    #   while still allowing exact version pinning for reproducibility
    packages_to_install=["pandas", "scikit-learn", "pyarrow"],
)
def preprocess_data(
    input_uri: str,
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    test_size: float = 0.2,
):
    """Preprocess raw data into train/test splits.

    Why a separate component?
      - Different compute needs (CPU-only, less memory)
      - Results are cached (rerun only if input changes)
      - Can be reused across multiple pipelines
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_uri)

    # Clean and preprocess
    # Why dropna on the target column specifically (not all columns)?
    #   Rows with missing features can be imputed; rows missing the label
    #   are completely unlearnable and must be removed to avoid silent NaN
    #   propagation into loss functions
    df = df.dropna(subset=["target"])
    # Why median fill instead of mean? Median is robust to outliers, which are
    #   common in financial/healthcare data and would skew a mean imputation
    df = df.fillna(df.median(numeric_only=True))

    # Split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    # Save outputs — KFP handles artifact storage automatically
    # Why Parquet instead of CSV? Parquet preserves dtypes (no silent int→float
    #   conversion on re-read) and is 3-10x smaller, reducing inter-step I/O cost
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
    max_depth: int = 10,
):
    """Train a scikit-learn model.

    Why output both Model and Metrics?
      - Model: the serialized model file for deployment
      - Metrics: logged to Vertex AI Experiments for tracking/comparison
    """
    import pandas as pd
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score

    train_df = pd.read_parquet(train_data.path)
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        # Why random_state=42? Ensures bit-for-bit reproducibility across reruns;
        #   without it, two identical pipeline executions can produce models that
        #   differ in accuracy by 1-2% due to non-deterministic tree building
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Log training metrics — visible in Vertex AI Experiments
    # Why log hyperparameters as metrics too (n_estimators, max_depth)?
    #   Vertex AI Experiments can only sort/filter on logged metrics, not on
    #   component parameters; logging them here enables side-by-side comparison
    #   of runs with different hyperparameter settings in the Experiments UI
    y_pred = clf.predict(X_train)
    metrics.log_metric("train_accuracy", accuracy_score(y_train, y_pred))
    metrics.log_metric("train_f1", f1_score(y_train, y_pred, average="weighted"))
    metrics.log_metric("n_estimators", n_estimators)
    metrics.log_metric("max_depth", max_depth)

    # Save model
    # Why append ".joblib" to model_artifact.path?
    #   KFP sets model_artifact.path to a directory path; joblib.dump
    #   needs a file path, so we add an extension to create a valid filename
    joblib.dump(clf, model_artifact.path + ".joblib")


@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "joblib", "pyarrow"],
)
def evaluate_model(
    test_data: Input[Dataset],
    model_artifact: Input[Model],
    metrics: Output[Metrics],
    accuracy_threshold: float = 0.85,
) -> bool:
    """Evaluate model and decide whether to deploy.

    Why return a boolean?
      The pipeline uses this to conditionally deploy.
      Only models meeting the threshold get promoted.
    """
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    test_df = pd.read_parquet(test_data.path)
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    clf = joblib.load(model_artifact.path + ".joblib")
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Why log both test_accuracy AND accuracy_threshold as metrics?
    #   Logging the threshold alongside the result makes it self-documenting
    #   in Vertex AI Experiments — reviewers see "0.87 vs threshold 0.85"
    #   without needing to look up the pipeline parameter values separately
    metrics.log_metric("test_accuracy", accuracy)
    metrics.log_metric("test_f1", f1)
    metrics.log_metric("accuracy_threshold", accuracy_threshold)
    # Why log deploy_approved as 0/1 integer instead of boolean?
    #   Vertex AI Experiments stores all metrics as floats; an integer 1/0
    #   makes filtering ("show me approved runs") work in the UI without parsing
    metrics.log_metric("deploy_approved", int(accuracy >= accuracy_threshold))

    return accuracy >= accuracy_threshold


@dsl.pipeline(
    name="vertex-ml-pipeline",
    description="End-to-end ML pipeline on Vertex AI",
)
def ml_pipeline(
    input_uri: str = "gs://my-bucket/data/raw.csv",
    accuracy_threshold: float = 0.85,
    n_estimators: int = 100,
    max_depth: int = 10,
):
    """Assemble the pipeline DAG.

    Why a pipeline function?
      KFP compiles this into a YAML spec that Vertex AI executes.
      The function defines the DAG; each component call creates a step.
    """
    # Step 1: Preprocess
    preprocess_task = preprocess_data(input_uri=input_uri)

    # Step 2: Train (depends on preprocess output)
    # Why pass outputs by name ("train_data") instead of by variable reference?
    #   KFP compiles the pipeline to YAML; using the named output key ensures
    #   the compiler can verify artifact type compatibility at compile time
    train_task = train_model(
        train_data=preprocess_task.outputs["train_data"],
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    # Step 3: Evaluate
    # Why pass test_data from preprocess_task rather than train_task?
    #   Evaluation must use held-out data the model never saw; wiring it from
    #   the preprocess step (not train) makes this data isolation explicit in
    #   the DAG and visible in the pipeline graph
    eval_task = evaluate_model(
        test_data=preprocess_task.outputs["test_data"],
        model_artifact=train_task.outputs["model_artifact"],
        accuracy_threshold=accuracy_threshold,
    )

    # Step 4: Conditional deployment
    # Why dsl.Condition instead of an if statement?
    #   Python if statements execute at compile time (always same branch);
    #   dsl.Condition creates a runtime branch that Vertex AI evaluates using
    #   the actual output value after the evaluate_model step completes
    with dsl.Condition(eval_task.output == True):  # noqa: E712
        # Deploy only if evaluation passes
        deploy_model(
            model=train_task.outputs["model_artifact"],
            project="my-gcp-project",
            location="us-central1",
        )
```

### 3.4 Vertex AI Model Monitoring

```python
"""
Vertex AI Model Monitoring: automated drift detection and alerting.

Why managed monitoring?
  - Automatic skew/drift detection on served predictions
  - No separate monitoring infrastructure to maintain
  - Alerts via Cloud Monitoring → PagerDuty/Slack
  - Integrated with Vertex AI Endpoints

Monitoring Types:
  1. Training-serving skew: training data vs live requests
  2. Prediction drift: live requests over time
  3. Feature attribution drift: SHAP values changing

  ┌──────────┐   ┌─────────────┐   ┌──────────────┐   ┌─────────┐
  │ Endpoint │──▶│ Sample      │──▶│ Compute      │──▶│ Alert   │
  │ Traffic  │   │ Requests    │   │ Statistics   │   │ (drift) │
  └──────────┘   └─────────────┘   └──────────────┘   └─────────┘
"""


def setup_model_monitoring(endpoint_name, project, location):
    """Configure Vertex AI Model Monitoring for an endpoint.

    This sets up automatic monitoring that:
      1. Samples a percentage of prediction requests
      2. Computes statistical tests against training data
      3. Alerts if drift exceeds thresholds
    """
    aiplatform.init(project=project, location=location)

    endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)

    # Configure monitoring
    # Why specify skew and drift thresholds separately?
    #   Skew = training data vs serving data (catches pipeline bugs)
    #   Drift = serving data over time (catches world changes)
    # Why point training_dataset at the original CSV, not a processed version?
    #   Vertex AI computes baseline statistics from this file once and caches them;
    #   using the same data distribution the model was trained on ensures skew
    #   detection measures the gap between training and production, not between
    #   two different preprocessing runs
    monitoring_config = {
        "objective_configs": [{
            "training_dataset": {
                "gcs_source": {"uris": ["gs://my-bucket/data/train.csv"]},
                "data_format": "csv",
                "target_field": "target",
            },
            "training_prediction_skew_detection_config": {
                "skew_thresholds": {
                    # Per-feature thresholds (Jensen-Shannon divergence)
                    # Why 0.3 as the JS divergence threshold?
                    #   JS divergence ranges 0-1; 0.3 is empirically a good
                    #   starting point — it catches meaningful distribution
                    #   shifts without triggering alerts on normal daily variation;
                    #   tune per-feature based on observed variance in staging
                    "age": {"value": 0.3},
                    "income": {"value": 0.3},
                },
                "default_skew_threshold": {"value": 0.3},
            },
            "prediction_drift_detection_config": {
                "drift_thresholds": {
                    "age": {"value": 0.3},
                    "income": {"value": 0.3},
                },
                "default_drift_threshold": {"value": 0.3},
            },
        }],
        # Why store monitoring stats in GCS rather than in the endpoint?
        #   GCS retention allows historical drift trend analysis; stats stored
        #   only in the endpoint are lost when the endpoint is updated or deleted
        "stats_anomalies_base_directory": "gs://my-bucket/monitoring/",
    }

    # Alert configuration — where to send drift alerts
    # Why email + Cloud Monitoring?
    #   Email for humans; Cloud Monitoring for automated response
    #   (e.g., trigger retraining pipeline via Cloud Functions)
    alert_config = {
        "email_alert_config": {
            "user_emails": ["ml-team@company.com"],
        },
    }

    return monitoring_config, alert_config
```

---

## 4. Azure Machine Learning

### 4.1 Azure ML Ecosystem Overview

```python
"""
Azure Machine Learning: enterprise-grade ML platform with Microsoft integration.

  Azure ML Ecosystem (as of 2025):
  ┌──────────────────────────────────────────────────────────────────┐
  │                       Azure ML Studio                            │
  │  (Web-based IDE: designer, notebooks, AutoML, endpoints)         │
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  Development:                  Operations:                       │
  │  ├─ Compute Instances (dev)    ├─ Pipelines (component-based)    │
  │  ├─ Compute Clusters (train)   ├─ Model Registry (with lineage) │
  │  ├─ AutoML                     ├─ Managed Endpoints              │
  │  ├─ Designer (drag-and-drop)   ├─ Batch Endpoints                │
  │  └─ VS Code integration       └─ Model Monitoring               │
  │                                                                  │
  │  Data & Features:              Responsible AI:                   │
  │  ├─ Managed Feature Store      ├─ Responsible AI Dashboard       │
  │  ├─ Data Assets (versioned)    ├─ Fairlearn integration          │
  │  └─ Azure Data Factory link    ├─ InterpretML (explainability)   │
  │                                └─ Error Analysis                  │
  │                                                                  │
  │  Infrastructure:                                                 │
  │  ├─ NCv3/ND series (NVIDIA GPUs)                                │
  │  ├─ Low-priority VMs (cost reduction)                            │
  │  └─ Azure Arc (hybrid cloud + edge)                              │
  └──────────────────────────────────────────────────────────────────┘

  Key differentiators:
    - Deepest enterprise integration (Active Directory, Azure DevOps)
    - Responsible AI Dashboard (error analysis + fairness + explainability)
    - VS Code integration (develop locally, run remotely)
    - Azure Arc: run ML workloads on-premises or multi-cloud
    - Managed Feature Store (newest addition, competitive with Feast)
"""
```

### 4.2 Azure ML Training with SDK v2

```python
"""
Azure ML SDK v2: Python-first, declarative ML workflow definitions.

Why SDK v2 over SDK v1?
  - Declarative YAML-based configuration (GitOps-friendly)
  - Consistent API for all resource types
  - Better integration with Azure CLI (az ml)
  - Component-based pipelines (reusable, testable)
"""
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import (
    Environment,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
)
from azure.identity import DefaultAzureCredential


def create_azure_training_job():
    """Create an Azure ML training job using SDK v2.

    Architecture:
      1. Define compute target (cluster or instance)
      2. Define environment (conda/Docker)
      3. Define command job (script + inputs + compute)
      4. Submit and monitor
    """
    # Authenticate — uses managed identity in Azure, local creds in dev
    # Why DefaultAzureCredential?
    #   Tries multiple auth methods in order:
    #   environment variables → managed identity → VS Code → Azure CLI
    credential = DefaultAzureCredential()

    # Why scope MLClient to a specific workspace (not subscription-wide)?
    #   Azure ML resources are workspace-scoped; a workspace-level client prevents
    #   jobs, environments, and models from accidentally leaking across projects
    #   that share the same Azure subscription
    ml_client = MLClient(
        credential=credential,
        subscription_id="<subscription-id>",
        resource_group_name="ml-rg",
        workspace_name="ml-workspace",
    )

    # Define the environment — what software is installed
    # Why a custom environment?
    #   Azure ML curated environments exist for common frameworks,
    #   but custom environments ensure reproducibility
    env = Environment(
        name="sklearn-training-env",
        # Why conda_file instead of pip requirements.txt?
        #   Conda resolves the full dependency graph including native libraries
        #   (MKL, OpenBLAS) that pip ignores; this prevents silent version
        #   conflicts between scikit-learn and NumPy's BLAS backend
        conda_file="environments/conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
    )

    # Define the training job
    # Why 'command' instead of 'Estimator'?
    #   SDK v2 uses 'command' as the primary abstraction
    #   It's more flexible and consistent with Azure CLI
    training_job = command(
        code="src/",                           # Source code directory
        # Why use ${{inputs.*}} template syntax instead of hardcoded values?
        #   Azure ML substitutes these at runtime; the same job definition can
        #   be re-run with different hyperparameters from the CLI or Studio
        #   without editing Python code — enabling GitOps-style parameter sweeps
        command="python train.py "
                "--n-estimators ${{inputs.n_estimators}} "
                "--max-depth ${{inputs.max_depth}} "
                "--data-path ${{inputs.training_data}}",
        inputs={
            # Why reference the data asset by name:version ("training-data:1")?
            #   Azure ML data assets are versioned; pinning to version 1 ensures
            #   the training job is reproducible even if the dataset is updated later
            "training_data": Input(type="uri_folder", path="azureml:training-data:1"),
            "n_estimators": 100,
            "max_depth": 10,
        },
        outputs={
            "model": Output(type="uri_folder"),
        },
        environment=env,
        compute="gpu-cluster",                 # Pre-created compute cluster
        display_name="sklearn-classifier-training",

        # Resource limits
        # Why set limits? Prevents runaway jobs from consuming the budget
        instance_count=1,
    )

    # Submit the job — returns a RunDetails object for tracking
    # Why create_or_update instead of create?
    #   Idempotent submission allows CI pipelines to safely retry on transient
    #   Azure API failures without risk of duplicate job creation
    returned_job = ml_client.jobs.create_or_update(training_job)
    print(f"Job submitted: {returned_job.studio_url}")

    return returned_job
```

### 4.3 Azure ML Pipelines (Component-Based)

```python
"""
Azure ML Pipelines: component-based ML workflow orchestration.

Why component-based?
  - Components are versioned, reusable, testable units
  - Like functions in a programming language
  - Can mix Python components with CLI components
  - Visual pipeline designer in Studio (drag-and-drop)
"""
from azure.ai.ml import dsl, Input, Output
from azure.ai.ml.entities import CommandComponent


# Define reusable components
# Why define as components?
#   - Version controlled in the workspace
#   - Discoverable by other team members
#   - Testable independently
#   - Can be shared across pipelines

preprocess_component = CommandComponent(
    name="preprocess_data",
    # Why version="1.0" explicitly?
    #   Azure ML stores every registered component version; without explicit
    #   versioning, the workspace auto-increments but pipelines referencing
    #   "latest" can silently pick up breaking changes in a new version
    version="1.0",
    display_name="Preprocess Training Data",
    description="Clean data, handle missing values, split train/test",
    inputs={
        # Why uri_folder instead of uri_file for raw_data?
        #   Datasets often arrive as partitioned files (e.g., date-sharded CSVs);
        #   uri_folder lets the component read the whole directory without knowing
        #   the individual file names in advance
        "raw_data": Input(type="uri_folder"),
        "test_size": Input(type="number", default=0.2),
    },
    outputs={
        "train_data": Output(type="uri_folder"),
        "test_data": Output(type="uri_folder"),
    },
    code="src/components/preprocess/",
    command=(
        "python preprocess.py "
        "--raw-data ${{inputs.raw_data}} "
        "--test-size ${{inputs.test_size}} "
        "--train-output ${{outputs.train_data}} "
        "--test-output ${{outputs.test_data}}"
    ),
    # Why pin to azureml:sklearn-env:1 (versioned) rather than "latest"?
    #   Pinning ensures the component always runs in the same environment;
    #   "latest" can silently break when the environment is updated for another task
    environment="azureml:sklearn-env:1",
)


@dsl.pipeline(
    name="azure-ml-training-pipeline",
    description="End-to-end ML pipeline on Azure ML",
    default_compute="cpu-cluster",
)
def azure_ml_pipeline(
    raw_data: Input,
    accuracy_threshold: float = 0.85,
):
    """Azure ML pipeline with preprocessing, training, and evaluation.

    Why @dsl.pipeline decorator?
      Tells Azure ML SDK this function defines a pipeline DAG.
      Each component call creates a step in the DAG.
    """
    # Step 1: Preprocess
    preprocess_step = preprocess_component(
        raw_data=raw_data,
        test_size=0.2,
    )

    # Step 2: Train
    train_step = command(
        name="train_model",
        code="src/components/train/",
        command="python train.py --data ${{inputs.data}} --output ${{outputs.model}}",
        # Why wire inputs from preprocess_step.outputs rather than raw_data directly?
        #   Data lineage: Azure ML traces the provenance chain from raw data →
        #   processed data → model; this makes the Studio lineage graph accurate
        #   and is required for regulatory audit trails in finance/healthcare
        inputs={"data": preprocess_step.outputs.train_data},
        outputs={"model": Output(type="uri_folder")},
        environment="azureml:sklearn-env:1",
    )

    # Step 3: Evaluate
    eval_step = command(
        name="evaluate_model",
        code="src/components/evaluate/",
        command=(
            "python evaluate.py "
            "--model ${{inputs.model}} "
            "--test-data ${{inputs.test_data}} "
            "--threshold ${{inputs.threshold}}"
        ),
        inputs={
            "model": train_step.outputs.model,
            # Why test_data from preprocess_step, not train_step?
            #   Enforces that evaluation uses data the model never saw;
            #   explicit wiring also means Azure ML validates this dependency
            #   at pipeline compile time before any compute is consumed
            "test_data": preprocess_step.outputs.test_data,
            # Why pass threshold as a pipeline parameter rather than hardcoding?
            #   Different deployment environments (staging vs prod) may require
            #   different quality gates without changing the component code
            "threshold": accuracy_threshold,
        },
        environment="azureml:sklearn-env:1",
    )

    # Why return the model output at the pipeline level?
    #   Surfacing outputs at the pipeline boundary lets downstream pipelines
    #   or Azure ML deployment jobs consume the model artifact directly without
    #   needing to know the internal step structure
    return {
        "model": train_step.outputs.model,
    }
```

### 4.4 Azure Responsible AI Dashboard

```python
"""
Azure Responsible AI Dashboard: unified view of model fairness,
interpretability, and error analysis.

Why a dashboard instead of individual tools?
  - Combines 4 tools in one view: Error Analysis, Fairlearn,
    InterpretML, and Counterfactual explanations
  - Decision-makers (not just data scientists) can explore results
  - Generates compliance-ready reports

  Dashboard Components:
  ┌──────────────────────────────────────────────────────────────┐
  │                Responsible AI Dashboard                      │
  │                                                              │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐│
  │  │ Error    │  │ Fairness │  │ Explain- │  │ Counterfact- ││
  │  │ Analysis │  │ (Fairlrn)│  │ ability  │  │ uals         ││
  │  │          │  │          │  │(InterpML)│  │ (DiCE)       ││
  │  │ Tree map │  │ Disparity│  │ Feature  │  │ "What-if"    ││
  │  │ of errors│  │ metrics  │  │ importance│ │ scenarios    ││
  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘│
  └──────────────────────────────────────────────────────────────┘
"""
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
)


def create_responsible_ai_insights(ml_client, model, train_data, test_data):
    """Create a Responsible AI Dashboard for a model.

    Why run RAI insights?
      - Error Analysis: find cohorts where the model fails
      - Fairness: detect bias across protected groups
      - Explainability: understand feature contributions
      - Counterfactuals: generate "what-if" scenarios
    """
    from azure.ai.ml import Input
    from azure.ai.ml.entities import (
        RaiInsightsComponent,
    )

    # The RAI Dashboard is built by running a pipeline that
    # computes all analyses and stores results in the workspace

    # Error Analysis component
    # Why error analysis?
    #   Traditional metrics give one number (e.g., accuracy = 90%).
    #   Error analysis shows WHERE the 10% errors are concentrated.
    #   E.g., "90% of errors are on patients over 70 with diabetes"
    error_analysis_config = {
        # Why max_depth=4? A decision tree of depth 4 can represent up to 16 error
        #   cohorts — enough to find meaningful patterns without overfitting to
        #   noise in the test set; deeper trees risk memorising individual errors
        "max_depth": 4,           # Decision tree depth for error patterns
        # Why num_leaves=31? Matches LightGBM's default; empirically captures
        #   the major error clusters while keeping the tree interpretable for
        #   non-technical stakeholders reviewing the dashboard
        "num_leaves": 31,         # Number of error cohorts to identify
    }

    # Fairness component
    fairness_config = {
        "sensitive_features": ["gender", "race"],
        # Why include false_positive_rate and false_negative_rate alongside accuracy?
        #   Overall accuracy can be equal across groups while one group suffers
        #   disproportionately high false negatives (e.g., loan denials for
        #   qualified applicants) — the disaggregated rates expose this disparity
        "metrics": [
            "accuracy_score",
            "false_positive_rate",
            "false_negative_rate",
            "selection_rate",
        ],
    }

    # Explainability component
    explainability_config = {
        # Why "mimic" explainer instead of SHAP TreeExplainer?
        #   Mimic trains a simple surrogate (e.g., linear) model to approximate
        #   the complex model globally; it's 10-100x faster than SHAP on large
        #   datasets and still produces reliable feature importance rankings
        "method": "mimic",        # Use InterpretML's mimic explainer
        # Why max_features=10? Displaying all features overwhelms reviewers;
        #   top-10 covers the dominant explanatory factors and keeps the
        #   dashboard readable for compliance officers who are not data scientists
        "max_features": 10,       # Top 10 features to explain
    }

    print("RAI Dashboard configuration created.")
    print(f"  Error Analysis: max_depth={error_analysis_config['max_depth']}")
    print(f"  Fairness: features={fairness_config['sensitive_features']}")
    print(f"  Explainability: method={explainability_config['method']}")

    return {
        "error_analysis": error_analysis_config,
        "fairness": fairness_config,
        "explainability": explainability_config,
    }
```

---

## 5. Platform Comparison

### 5.1 Feature Matrix

```python
"""
Cloud MLOps Platform Comparison (as of 2025):

┌──────────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ Feature              │ AWS SageMaker    │ Google Vertex AI │ Azure ML         │
├──────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ IDE                  │ SageMaker Studio │ Workbench        │ ML Studio + VSC  │
│ AutoML               │ Autopilot        │ AutoML           │ AutoML           │
│ Notebook             │ Studio Notebooks │ Managed Notebook │ Compute Instance │
├──────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Experiment Tracking  │ Experiments      │ Experiments      │ MLflow / Jobs    │
│ Pipeline Orchestr.   │ SM Pipelines     │ Vertex Pipelines │ Azure ML Pipeline│
│ Pipeline SDK         │ SageMaker SDK    │ KFP v2 (open)    │ Azure ML SDK v2  │
├──────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Model Registry       │ Model Registry   │ Model Registry   │ Model Registry   │
│ Model Serving        │ Endpoints        │ Endpoints        │ Managed Endpoint │
│ Batch Inference      │ Batch Transform  │ Batch Prediction │ Batch Endpoint   │
│ Edge Deployment      │ Edge Manager     │ Edge (limited)   │ Azure Arc        │
├──────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Feature Store        │ Feature Store    │ Feature Store    │ Feature Store    │
│ Data Versioning      │ Limited          │ Managed Datasets │ Data Assets      │
│ Monitoring           │ Model Monitor    │ Model Monitoring │ Data Collector   │
├──────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Bias Detection       │ Clarify          │ What-If Tool     │ Fairlearn / RAI  │
│ Explainability       │ Clarify (SHAP)   │ Explainable AI   │ InterpretML      │
├──────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Custom Hardware      │ Inf2, Trn1       │ TPU v5           │ ND-series GPUs   │
│ Spot/Low-Priority    │ Spot Training    │ Spot VMs         │ Low-priority VMs │
│ Distributed Training │ Data Parallel    │ Reduction Server │ Distributed      │
├──────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Gen AI Integration   │ Bedrock / JumpSt │ Model Garden     │ Azure OpenAI     │
│ SQL ML               │ Redshift ML      │ BigQuery ML      │ SQL ML           │
├──────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Pricing Model        │ Per-instance-hr  │ Per-instance-hr  │ Per-instance-hr  │
│ Free Tier            │ 250 hrs Studio   │ $300 credit      │ $200 credit      │
│ Ecosystem Lock-in    │ High (SM SDK)    │ Medium (KFP)     │ Medium (AzML SDK)│
└──────────────────────┴──────────────────┴──────────────────┴──────────────────┘

Key Observations:
  1. All three are converging — feature parity is increasing
  2. SageMaker: most mature, deepest service integration
  3. Vertex AI: best for large models (TPU), most portable (KFP)
  4. Azure ML: best for enterprise (AD, compliance), best RAI tooling
"""
```

### 5.2 When to Choose Which Platform

```python
"""
Decision Framework — When to Choose Which Platform:

  Choose AWS SageMaker when:
    ✓ Already heavily invested in AWS
    ✓ Need the widest variety of instance types
    ✓ Want the most mature ML marketplace
    ✓ Running on Inferentia/Trainium chips for cost optimization
    ✓ Large team already familiar with SageMaker

  Choose Google Vertex AI when:
    ✓ Training very large models (TPU access)
    ✓ Heavy BigQuery / data warehouse usage
    ✓ Want pipeline portability (KFP is open source)
    ✓ Using Google's generative AI models (Gemini)
    ✓ Prefer simpler, more opinionated API design

  Choose Azure Machine Learning when:
    ✓ Enterprise with Microsoft stack (AD, Teams, DevOps)
    ✓ Regulatory requirements need Responsible AI Dashboard
    ✓ Hybrid cloud / on-premise requirements (Azure Arc)
    ✓ VS Code-first development workflow
    ✓ Using Azure OpenAI for LLM applications

  Consider vendor-neutral (MLflow + Kubeflow) when:
    ✓ Multi-cloud strategy is mandatory
    ✓ Large ML platform team (> 5 engineers)
    ✓ Need full control over every component
    ✓ Running on-premise or hybrid
    ✓ Cost optimization at scale (> $500K/year ML spend)
"""
```

---

## 6. Multi-Cloud and Vendor-Neutral Strategies

### 6.1 MLflow as an Abstraction Layer

```python
"""
MLflow as a vendor-neutral abstraction layer:

  MLflow provides a consistent API across all cloud platforms.
  Your training code stays the same; only the backend changes.

  Architecture:
  ┌──────────────────────────────────────────────────────────────┐
  │                    Your ML Code (unchanged)                   │
  │   import mlflow                                               │
  │   mlflow.log_metric("accuracy", 0.95)                         │
  │   mlflow.sklearn.log_model(model, "model")                    │
  └──────────────────┬───────────────────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ SageMaker│ │ Vertex AI│ │ Azure ML │
  │ (backend)│ │ (backend)│ │ (backend)│
  │          │ │          │ │          │
  │ S3 store │ │ GCS store│ │ Blob     │
  │ SM Deploy│ │ Vertex   │ │ AzML     │
  │          │ │ Endpoint │ │ Endpoint │
  └──────────┘ └──────────┘ └──────────┘

  What MLflow abstracts:
    ✓ Experiment tracking (runs, metrics, artifacts)
    ✓ Model packaging (MLmodel format)
    ✓ Model serving (REST API)
    ✓ Model registry (staging → production)

  What MLflow does NOT abstract:
    ✗ Pipeline orchestration (need Kubeflow/Airflow)
    ✗ Feature stores (need Feast or cloud-native)
    ✗ Data versioning (need DVC)
    ✗ Monitoring (need Evidently or cloud-native)
"""
import mlflow


def portable_training_code():
    """Training code that works on any cloud platform.

    Why use MLflow for portability?
      - Same code runs on SageMaker, Vertex AI, Azure ML, or local
      - Backend configuration is separate from training logic
      - Model artifacts are stored in MLflow's standard format
    """
    # This code runs identically regardless of cloud provider
    # Only the tracking URI and artifact store change
    # Why set_tracking_uri via code rather than the MLFLOW_TRACKING_URI env var?
    #   Code-level configuration is explicit and version-controlled; env vars
    #   are invisible in code reviews and can be silently overridden in CI systems
    mlflow.set_tracking_uri("http://mlflow-server:5000")  # Change per environment

    with mlflow.start_run(run_name="portable-training"):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split

        # Generate sample data
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        # Train
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # Log — these calls work on ANY backend
        accuracy = clf.score(X_test, y_test)
        # Why log_param for n_estimators AND log_metric for accuracy in the same run?
        #   MLflow separates params (hyperparameters, set before training) from
        #   metrics (results, computed after training); this separation enables
        #   the parallel coordinates plot in the MLflow UI for hyperparameter analysis
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        # Why log_model instead of just saving the .pkl file to disk?
        #   mlflow.sklearn.log_model wraps the model in an MLmodel spec that
        #   records the Python environment, input schema, and serving interface;
        #   any cloud's MLflow integration can then deploy it without extra config
        mlflow.sklearn.log_model(clf, "model")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Model logged to: {mlflow.get_artifact_uri()}")
```

### 6.2 Kubeflow for Pipeline Portability

```python
"""
Kubeflow Pipelines: portable ML pipelines across cloud and on-premise.

  Portability matrix:
  ┌────────────────────┬──────────────────────────────────────────┐
  │ Platform           │ KFP Pipeline Runs On?                    │
  ├────────────────────┼──────────────────────────────────────────┤
  │ Vertex AI Pipelines│ ✓ Native (KFP v2 is the SDK)            │
  │ AWS (self-hosted)  │ ✓ Kubeflow on EKS                       │
  │ Azure (self-hosted)│ ✓ Kubeflow on AKS                       │
  │ On-premise         │ ✓ Kubeflow on any K8s cluster            │
  │ SageMaker Pipelines│ ✗ Different SDK (not KFP-compatible)     │
  │ Azure ML Pipelines │ ✗ Different SDK (Azure ML SDK v2)        │
  └────────────────────┴──────────────────────────────────────────┘

  Strategy:
    - Write pipelines using KFP v2 SDK
    - Run on Vertex AI Pipelines for managed experience
    - Fall back to self-hosted Kubeflow if needed for portability

  Trade-off:
    KFP gives you portability at the cost of cloud-native features.
    SageMaker/Azure ML pipelines have deeper integration with their
    respective platforms but lock you in.
"""
```

### 6.3 Abstraction Layer Decision Framework

```python
"""
Choosing Your Abstraction Strategy:

  ┌─────────────────────────────────────────────────────────────────┐
  │ Question                        │ Recommendation                │
  ├─────────────────────────────────┼───────────────────────────────┤
  │ Single cloud, small team?       │ Go cloud-native (SageMaker/   │
  │                                 │ Vertex AI/Azure ML)            │
  ├─────────────────────────────────┼───────────────────────────────┤
  │ Multi-cloud required?           │ MLflow (tracking) +            │
  │                                 │ Kubeflow (pipelines) +         │
  │                                 │ Feast (features)               │
  ├─────────────────────────────────┼───────────────────────────────┤
  │ Single cloud, might switch?     │ Use cloud-native BUT use       │
  │                                 │ MLflow for experiment tracking  │
  │                                 │ (easiest to abstract)          │
  ├─────────────────────────────────┼───────────────────────────────┤
  │ On-premise + cloud hybrid?      │ Kubeflow everywhere +          │
  │                                 │ cloud storage backends          │
  ├─────────────────────────────────┼───────────────────────────────┤
  │ Regulatory / data sovereignty?  │ On-prem Kubeflow with          │
  │                                 │ cloud for non-sensitive work    │
  └─────────────────────────────────┴───────────────────────────────┘

  The "pit of success" strategy:
    1. Start with cloud-native (fastest time-to-value)
    2. Add MLflow for experiment tracking (easy, high ROI)
    3. Wrap cloud-specific serving behind an API gateway
    4. Move to Kubeflow ONLY if multi-cloud becomes real requirement
    5. Avoid premature abstraction — it has real costs
"""
```

---

## 7. Cost Optimization Strategies

### 7.1 Training Cost Optimization

```python
"""
ML Cloud Cost Optimization Strategies:

  Training is the largest cost center for most ML teams.
  Here are strategies ordered by effort vs impact:

  ┌─────────────────────────────────────────────────────────────────┐
  │ Strategy                │ Effort │ Savings   │ Risk            │
  ├─────────────────────────┼────────┼───────────┼─────────────────┤
  │ Spot/preemptible VMs    │ Low    │ 60-90%    │ Job interruption│
  │ Right-size instances    │ Low    │ 30-50%    │ Slower training │
  │ Auto-shutdown notebooks │ Low    │ 20-40%    │ None            │
  │ Reserved instances      │ Medium │ 30-60%    │ Commitment      │
  │ Mixed precision (fp16)  │ Medium │ 40-60%    │ Numerical issues│
  │ Model distillation      │ High   │ 50-80%    │ Quality loss    │
  │ Architecture search     │ High   │ Variable  │ Time investment │
  └─────────────────────────┴────────┴───────────┴─────────────────┘
"""


def estimate_training_cost(instance_type, hours, provider="aws",
                           spot=False):
    """Estimate training cost for a given configuration.

    Why estimate before running?
      Cloud ML costs can surprise teams — a multi-GPU training job
      can cost hundreds of dollars per hour. Estimation prevents
      budget overruns.
    """
    # Approximate hourly rates (USD, as of 2025)
    # Why approximate? Prices change; check current pricing
    # Why store both on_demand and spot in the same dict entry?
    #   Enables direct savings comparison without a second lookup; the ratio
    #   (spot/on_demand) varies dramatically by instance family — GPU instances
    #   often have deeper spot discounts than CPU instances
    pricing = {
        "aws": {
            "ml.m5.xlarge":    {"on_demand": 0.23, "spot": 0.07},
            "ml.g4dn.xlarge":  {"on_demand": 0.74, "spot": 0.22},
            "ml.p3.2xlarge":   {"on_demand": 3.83, "spot": 1.15},
            # Why include p4d.24xlarge? At $32/hr on-demand, a 2-hour run
            #   ($65) is easy to underestimate; having this in the table makes
            #   the cost impact of large-scale training immediately visible
            "ml.p4d.24xlarge": {"on_demand": 32.77, "spot": 9.83},
        },
        "gcp": {
            "n1-standard-8":   {"on_demand": 0.38, "spot": 0.08},
            "n1-standard-8+T4":{"on_demand": 0.73, "spot": 0.22},
            "a2-highgpu-1g":   {"on_demand": 3.67, "spot": 1.10},
        },
        "azure": {
            "Standard_D4s_v3": {"on_demand": 0.19, "spot": 0.04},
            "Standard_NC6":    {"on_demand": 0.90, "spot": 0.18},
            "Standard_ND40rs_v2": {"on_demand": 22.03, "spot": 6.61},
        },
    }

    provider_pricing = pricing.get(provider, {})
    instance_pricing = provider_pricing.get(instance_type, {})

    if not instance_pricing:
        return {"error": f"Unknown instance: {instance_type} on {provider}"}

    rate_type = "spot" if spot else "on_demand"
    hourly_rate = instance_pricing[rate_type]
    total_cost = hourly_rate * hours

    return {
        "provider": provider,
        "instance_type": instance_type,
        "rate_type": rate_type,
        "hourly_rate": hourly_rate,
        "hours": hours,
        "total_cost": round(total_cost, 2),
        # Why compute savings_vs_on_demand only when spot=True?
        #   When running on-demand the saving is definitionally 0%;
        #   including a 0% saving in the output makes budget reports cleaner
        #   and avoids misleading "0% saving" lines for on-demand jobs
        "savings_vs_on_demand": (
            round((1 - instance_pricing["spot"] / instance_pricing["on_demand"]) * 100, 1)
            if spot else 0
        ),
    }


def cost_optimization_recommendations(monthly_spend, team_size,
                                       training_frequency):
    """Generate cost optimization recommendations based on usage profile.

    Why profile-based recommendations?
      A startup training once a week has different optimization strategies
      than an enterprise training 100 models daily. One-size-fits-all
      advice wastes effort on low-impact changes.
    """
    recommendations = []

    # Low-hanging fruit — always applicable
    # Why 10% savings estimate for notebook auto-shutdown?
    #   Industry surveys show notebooks are idle 30-50% of working hours;
    #   assuming they run 8hr/day, a 1hr idle cutoff recovers ~10% of
    #   notebook compute spend with zero model quality impact
    recommendations.append({
        "strategy": "Auto-shutdown idle notebooks",
        "effort": "LOW",
        "estimated_savings": round(monthly_spend * 0.10, 2),
        "how": "Set lifecycle policies: 1hr idle → stop instance",
    })

    # Spot instances — high impact, some risk
    # Why gate spot recommendations on daily training frequency?
    #   Teams training weekly can absorb a spot interruption with manual retry;
    #   daily teams need automated checkpointing to make spot viable, so only
    #   recommend when the payoff justifies the engineering investment
    if training_frequency == "daily":
        recommendations.append({
            "strategy": "Use spot instances for training",
            "effort": "LOW",
            "estimated_savings": round(monthly_spend * 0.40, 2),
            "how": "Enable spot/preemptible with checkpointing",
            "risk": "Jobs may be interrupted; needs checkpoint logic",
        })

    # Reserved capacity — for predictable workloads
    # Why $5000/month as the threshold for reserved instance recommendations?
    #   Below $5K the 1-year commitment savings (~$1.5K) don't justify the
    #   flexibility cost; above $5K the savings compound significantly and
    #   offset the risk of being locked into a specific instance type
    if monthly_spend > 5000:
        recommendations.append({
            "strategy": "Purchase reserved instances (1yr)",
            "effort": "MEDIUM",
            "estimated_savings": round(monthly_spend * 0.30, 2),
            "how": "Commit to 1-year reserved capacity for stable workloads",
            "risk": "Lock-in to specific instance types",
        })

    # Mixed precision — for GPU workloads
    if monthly_spend > 2000:
        recommendations.append({
            "strategy": "Enable mixed precision training (fp16/bf16)",
            "effort": "MEDIUM",
            "estimated_savings": round(monthly_spend * 0.25, 2),
            "how": "Use PyTorch AMP or TensorFlow mixed precision API",
            "risk": "Rare numerical instability; test thoroughly",
        })

    # Right-sizing — needs profiling
    recommendations.append({
        "strategy": "Right-size instances based on GPU utilization",
        "effort": "MEDIUM",
        "estimated_savings": round(monthly_spend * 0.20, 2),
        "how": "Profile GPU/CPU utilization; downsize if < 60% utilized",
    })

    return {
        "current_monthly_spend": monthly_spend,
        # Why sum estimated_savings across all recommendations?
        #   This overstates total savings (strategies overlap), but it gives
        #   a useful upper-bound "opportunity size" to motivate the work;
        #   real savings require prioritising one strategy at a time
        "potential_monthly_savings": sum(r["estimated_savings"] for r in recommendations),
        # Why sort by estimated_savings descending?
        #   Teams should tackle the highest-ROI optimisation first;
        #   sorted output makes the prioritisation decision self-evident
        "recommendations": sorted(recommendations,
                                   key=lambda r: r["estimated_savings"],
                                   reverse=True),
    }
```

---

## 8. Practical Migration Patterns

### 8.1 Migration from Local to Cloud

```python
"""
Migration Path: Local Development → Cloud MLOps

  Most teams follow this progression:
  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ Stage 0  │──▶│ Stage 1  │──▶│ Stage 2  │──▶│ Stage 3  │
  │ Local    │   │ Cloud    │   │ Pipeline │   │ Full     │
  │ Jupyter  │   │ Notebook │   │ Auto.    │   │ MLOps    │
  └──────────┘   └──────────┘   └──────────┘   └──────────┘

  Stage 0: Local Development
    - Jupyter on laptop
    - No experiment tracking
    - Manual model deployment
    - Time: days to start

  Stage 1: Cloud Notebooks
    - SageMaker Studio / Vertex Workbench / Azure ML Compute
    - MLflow for experiment tracking
    - Manual pipeline execution
    - Benefit: better compute, collaboration, tracked experiments

  Stage 2: Pipeline Automation
    - SageMaker Pipelines / Vertex Pipelines / Azure ML Pipelines
    - Automated training on schedule or trigger
    - Model registry with approval gates
    - Benefit: reproducibility, audit trail, faster iteration

  Stage 3: Full MLOps
    - CI/CD for ML code and data
    - Automated testing (data, model, infra)
    - Production monitoring and auto-retraining
    - Feature store for consistent features
    - Benefit: reliable, scalable, maintainable ML systems

  Key principle: don't skip stages.
  Each stage builds the habits and infrastructure for the next.
"""
```

---

## Summary

| Aspect | AWS SageMaker | Google Vertex AI | Azure ML |
|--------|---------------|------------------|----------|
| **Best for** | AWS-native teams | Large models, data teams | Enterprise, Microsoft stack |
| **Strength** | Breadth, maturity | TPU, BigQuery ML, KFP portability | RAI Dashboard, VS Code, Arc |
| **Lock-in level** | High (SM SDK) | Medium (KFP portable) | Medium (AzML SDK) |
| **Unique feature** | Inferentia/Trainium | BigQuery ML | Responsible AI Dashboard |
| **Pipeline SDK** | Proprietary | KFP v2 (open source) | Proprietary |
| **Cost strategy** | Spot + Savings Plans | Spot + CUDs | Low-priority + Reservations |

The meta-lesson: **choose based on where your data already lives and what your team already knows.** The differences between platforms are smaller than the cost of switching. Start with one platform, use MLflow for portability insurance, and only go multi-cloud when business requirements demand it.

---

## Exercises

### Exercise 1: SageMaker Training Job
Using the SageMaker Python SDK:
- Create a training job for a scikit-learn classifier
- Configure spot training with a 2-hour max wait
- Log custom metrics to CloudWatch
- Retrieve the trained model artifact from S3

### Exercise 2: Vertex AI Pipeline
Using the KFP v2 SDK:
- Create a 3-step pipeline (preprocess, train, evaluate)
- Add conditional deployment based on accuracy threshold
- Compile the pipeline to YAML
- Identify which components could be reused in other pipelines

### Exercise 3: Platform Comparison
For your current (or hypothetical) ML project:
- Evaluate all three platforms using the feature matrix in Section 5
- Estimate monthly costs on each platform using the pricing estimator
- Recommend a platform with justification based on your team size, existing cloud usage, and ML workload characteristics
- Document what would need to change if you switched platforms

### Exercise 4: Vendor-Neutral Architecture
Design a vendor-neutral ML architecture that uses:
- MLflow for experiment tracking and model registry
- Kubeflow Pipelines for orchestration
- Feast for feature store
- Evidently for monitoring
- Document where cloud-native components could substitute for better integration

### Exercise 5: Cost Optimization Audit
Perform a cost optimization audit:
- Profile your (hypothetical) ML workload: instance types, hours, frequency
- Apply the cost estimation function from Section 7
- Calculate potential savings from spot instances, right-sizing, and mixed precision
- Create a 90-day cost optimization plan with prioritized actions

---

[← Previous: 16. Model Testing and Validation](16_Model_Testing_and_Validation.md) | [Next: Overview →](00_Overview.md)
