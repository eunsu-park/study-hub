# MLOps Learning Guide

## Overview

MLOps (Machine Learning Operations) is a practical field that automates and streamlines the development, deployment, and operation of machine learning models. This learning material is designed for learners who have completed the basics of Deep Learning and covers the overall processes and tools needed to operate ML systems reliably in production.

### Target Audience
- Those who have completed Deep Learning basics
- Developers who want to deploy ML models to production
- Software engineers collaborating with data science teams
- DevOps/SRE engineers responsible for operating ML systems

---

## Learning Roadmap

```
MLOps Overview → ML Lifecycle → MLflow Basics → MLflow Advanced → W&B
     │                              │              │         │
     │                              ↓              ↓         ↓
     │                         Experiment Tracking ──── Model Registry ──→ Kubeflow
     │                                                            │
     ↓                                                            ↓
Feature Store ← Drift/Monitoring ← TorchServe/Triton ← Model Serving Basics
     │
     ↓
Practical MLOps Project (E2E Pipeline)
```

---

## File List

| File | Topic | Difficulty | Key Content |
|------|------|--------|----------|
| [01_MLOps_Overview.md](./01_MLOps_Overview.md) | MLOps Overview | ⭐ | MLOps definition, DevOps vs MLOps, maturity levels, tool ecosystem |
| [02_ML_Lifecycle.md](./02_ML_Lifecycle.md) | ML Lifecycle | ⭐⭐ | Project phases, retraining triggers, version control strategies |
| [03_MLflow_Basics.md](./03_MLflow_Basics.md) | MLflow Basics | ⭐⭐ | Tracking, experiment management, metrics/parameters logging, UI |
| [04_MLflow_Advanced.md](./04_MLflow_Advanced.md) | MLflow Advanced | ⭐⭐⭐ | Projects, Models, Registry, Serving |
| [05_Weights_and_Biases.md](./05_Weights_and_Biases.md) | Weights & Biases | ⭐⭐ | Experiment logging, Sweeps, Artifacts, MLflow comparison |
| [06_Kubeflow_Pipelines.md](./06_Kubeflow_Pipelines.md) | Kubeflow Pipelines | ⭐⭐⭐ | Pipeline SDK, component authoring, K8s integration |
| [07_Model_Registry.md](./07_Model_Registry.md) | Model Registry | ⭐⭐ | Version control, stage transitions, CI/CD integration |
| [08_Model_Serving_Basics.md](./08_Model_Serving_Basics.md) | Model Serving Basics | ⭐⭐ | REST API, gRPC, batch vs real-time inference |
| [09_TorchServe_Triton.md](./09_TorchServe_Triton.md) | TorchServe & Triton | ⭐⭐⭐ | Handler authoring, model optimization, multi-model serving |
| [10_Drift_Detection_Monitoring.md](./10_Drift_Detection_Monitoring.md) | Drift & Monitoring | ⭐⭐⭐ | Data/model drift, Evidently AI, alert configuration |
| [11_Feature_Stores.md](./11_Feature_Stores.md) | Feature Store | ⭐⭐⭐ | Feast, online/offline stores, feature serving |
| [12_Practical_MLOps_Project.md](./12_Practical_MLOps_Project.md) | Practical MLOps Project | ⭐⭐⭐⭐ | E2E pipeline, automated retraining, project structure |
| [13_CICD_for_ML.md](./13_CICD_for_ML.md) | CI/CD for ML | ⭐⭐⭐ | ML pipeline CI/CD, data/model validation gates, deployment strategies, rollback |
| [14_DVC_Data_Version_Control.md](./14_DVC_Data_Version_Control.md) | DVC Data Version Control | ⭐⭐⭐ | DVC pipelines, experiment tracking, CML, remote storage |
| [15_LLMOps.md](./15_LLMOps.md) | LLMOps | ⭐⭐⭐⭐ | Prompt versioning, RAG operations, LLM evaluation, guardrails, cost management |
| [16_Model_Testing_and_Validation.md](./16_Model_Testing_and_Validation.md) | Model Testing & Validation | ⭐⭐⭐ | Data/model/behavioral/infra tests, fairness, shadow deployment, CI/CD automation |
| [17_Cloud_MLOps_Platforms.md](./17_Cloud_MLOps_Platforms.md) | Cloud MLOps Platforms | ⭐⭐⭐⭐ | SageMaker, Vertex AI, Azure ML, platform comparison, cost optimization |

---

## Environment Setup

### Installing Required Libraries

```bash
# Basic ML libraries
pip install numpy pandas scikit-learn torch torchvision

# MLOps tools
pip install mlflow wandb feast evidently

# Serving tools
pip install torchserve torch-model-archiver

# Kubeflow (Python SDK)
pip install kfp
```

### Docker Setup

```bash
# MLflow server (Docker)
docker run -d \
  --name mlflow-server \
  -p 5000:5000 \
  -v $(pwd)/mlruns:/mlruns \
  ghcr.io/mlflow/mlflow:latest \
  mlflow server --host 0.0.0.0 --backend-store-uri /mlruns

# Triton Inference Server
docker run --gpus all -d \
  --name triton-server \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

### Version Check

```python
import mlflow
import wandb
import feast

print(f"MLflow: {mlflow.__version__}")
print(f"W&B: {wandb.__version__}")
print(f"Feast: {feast.__version__}")
```

### Recommended Versions
- Python: 3.9+
- MLflow: 2.8+
- Weights & Biases: 0.16+
- Feast: 0.35+
- Kubernetes: 1.25+

---

## Recommended Learning Order

### Stage 1: Basic Theory (01-02)
- Understand MLOps concepts and necessity
- Learn ML project lifecycle

### Stage 2: Experiment Management (03-05)
- Track experiments with MLflow
- Utilize Weights & Biases
- Manage metrics, parameters, and artifacts

### Stage 3: Pipelines (06-07)
- Build ML pipelines using Kubeflow
- Operate model registry

### Stage 4: Model Serving (08-09)
- Deploy REST/gRPC APIs
- Utilize TorchServe, Triton

### Stage 5: Monitoring & Feature Store (10-11)
- Detect drift
- Build feature stores

### Stage 6: Practical Project (12)
- Build E2E MLOps pipeline
- Implement automated retraining system

### Stage 7: Advanced Topics (13-17)
- CI/CD pipelines for ML systems
- Data version control with DVC
- LLMOps: operating LLM-based applications
- Model testing and validation strategies
- Cloud MLOps platforms (SageMaker, Vertex AI, Azure ML)

---

## Related Resources

### Official Documentation
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Weights & Biases Docs](https://docs.wandb.ai/)
- [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/)
- [Feast Documentation](https://docs.feast.dev/)
- [TorchServe](https://pytorch.org/serve/)
- [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/)

### Recommended Books
- "Introducing MLOps" - Mark Treveil
- "Machine Learning Engineering" - Andriy Burkov
- "Designing Machine Learning Systems" - Chip Huyen

### Related Learning Materials
- [Docker folder](../Docker/): Container basics
- [Deep_Learning folder](../Deep_Learning/): Deep learning model training
- [Machine_Learning folder](../Machine_Learning/): ML algorithm basics
