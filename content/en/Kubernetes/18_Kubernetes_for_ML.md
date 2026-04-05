# 18. Kubernetes for ML

**Previous**: [17. Production Operations](./17_Production_Operations.md) | **Next**: [19. Capstone: Production Cluster](./19_Capstone_Production_Cluster.md)

## Learning Objectives

- Configure GPU scheduling in Kubernetes using device plugins and resource limits
- Deploy and operate the NVIDIA GPU Operator for automated GPU management
- Use Kubeflow components for ML workflows including Notebooks, Pipelines, and Training Operator
- Serve models at scale with KServe, Seldon Core, and NVIDIA Triton
- Optimize ML workloads with spot instances, resource quotas, and cost management

---

Machine learning workloads have unique infrastructure demands: GPUs for training, large datasets that must be loaded efficiently, long-running distributed training jobs, and low-latency model serving. Kubernetes has become the platform of choice for ML infrastructure because it provides scheduling, resource management, and scalability in a standardized way. This lesson covers the complete lifecycle of ML on Kubernetes — from GPU scheduling to distributed training to production model serving.

## Table of Contents

- [1. GPU Scheduling in Kubernetes](#1-gpu-scheduling-in-kubernetes)
- [2. NVIDIA GPU Operator](#2-nvidia-gpu-operator)
- [3. Kubeflow Components](#3-kubeflow-components)
- [4. Distributed Training on Kubernetes](#4-distributed-training-on-kubernetes)
- [5. Model Serving](#5-model-serving)
- [6. ML Experiment Tracking](#6-ml-experiment-tracking)
- [7. Spot and Preemptible Instances for Training](#7-spot-and-preemptible-instances-for-training)
- [8. Resource Quotas for ML Teams](#8-resource-quotas-for-ml-teams)
- [Exercises](#exercises)

---

## 1. GPU Scheduling in Kubernetes

### 1.1 How GPU Scheduling Works

Kubernetes treats GPUs as **extended resources** advertised by device plugins. The kubelet discovers GPUs through the device plugin framework, and the scheduler assigns them to pods using standard resource requests.

```
GPU Scheduling Flow:
┌──────────────┐    Register     ┌──────────────┐
│ GPU Device   │ ──────────────► │   Kubelet    │
│ Plugin       │    Advertise    │              │
│ (DaemonSet)  │    nvidia.com/  │  Node:       │
│              │    gpu: 4       │  Allocatable: │
└──────────────┘                 │  nvidia.com/ │
                                 │   gpu: 4     │
                                 └──────┬───────┘
                                        │ Reports
                                        ▼
                                 ┌──────────────┐
                                 │  API Server   │
                                 └──────┬───────┘
                                        │
                                        ▼
┌──────────────┐    Schedule     ┌──────────────┐
│  Scheduler   │ ──────────────► │  Pod gets    │
│  checks GPU  │    to node      │  GPU device  │
│  availability│    with GPU     │  assigned    │
└──────────────┘                 └──────────────┘
```

### 1.2 Requesting GPUs in Pod Specs

```yaml
# Simple GPU pod
apiVersion: v1
kind: Pod
metadata:
  name: gpu-training
  namespace: ml-team
spec:
  restartPolicy: Never
  containers:
    - name: trainer
      image: nvcr.io/nvidia/pytorch:24.01-py3
      command: ["python", "train.py"]
      resources:
        limits:
          nvidia.com/gpu: 1      # Request exactly 1 GPU
          # GPUs cannot be overcommitted:
          # requests are implicitly set equal to limits
        requests:
          cpu: "4"
          memory: 16Gi
      volumeMounts:
        - name: dataset
          mountPath: /data
        - name: shm                # Shared memory for PyTorch DataLoader
          mountPath: /dev/shm
  volumes:
    - name: dataset
      persistentVolumeClaim:
        claimName: training-dataset
    - name: shm
      emptyDir:
        medium: Memory
        sizeLimit: 8Gi
```

### 1.3 Multi-GPU Training Pod

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: multi-gpu-training
  namespace: ml-team
spec:
  restartPolicy: Never
  containers:
    - name: trainer
      image: nvcr.io/nvidia/pytorch:24.01-py3
      command:
        - python
        - -m
        - torch.distributed.launch
        - --nproc_per_node=4
        - train.py
        - --batch-size=256
        - --epochs=100
      resources:
        limits:
          nvidia.com/gpu: 4       # 4 GPUs on a single node
        requests:
          cpu: "16"
          memory: 64Gi
      env:
        - name: NCCL_DEBUG
          value: INFO
        - name: NCCL_SOCKET_IFNAME
          value: eth0
      volumeMounts:
        - name: shm
          mountPath: /dev/shm
  volumes:
    - name: shm
      emptyDir:
        medium: Memory
        sizeLimit: 32Gi
  # Schedule on nodes with 4+ GPUs
  nodeSelector:
    gpu-count: "4"
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
```

### 1.4 GPU Node Labels and Taints

```bash
# View GPU node labels (set by NVIDIA device plugin / GPU operator)
kubectl get nodes -l nvidia.com/gpu.present=true \
  -o custom-columns=\
"NAME:.metadata.name,\
GPU_PRODUCT:.metadata.labels.nvidia\.com/gpu\.product,\
GPU_COUNT:.status.allocatable.nvidia\.com/gpu,\
GPU_MEM:.metadata.labels.nvidia\.com/gpu\.memory"

# Taint GPU nodes to prevent non-GPU workloads
kubectl taint nodes gpu-node-1 nvidia.com/gpu=present:NoSchedule

# Node affinity for specific GPU types
# (e.g., schedule only on A100 nodes)
```

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: a100-training
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values:
                  - NVIDIA-A100-SXM4-80GB
                  - NVIDIA-A100-SXM4-40GB
  containers:
    - name: trainer
      image: nvcr.io/nvidia/pytorch:24.01-py3
      resources:
        limits:
          nvidia.com/gpu: 8
```

### 1.5 GPU Sharing Strategies

```
GPU Sharing Options:
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Time-Slicing (NVIDIA)                                   │
│  ├── Multiple pods share a GPU via time-multiplexing     │
│  ├── No memory isolation                                 │
│  └── Good for inference, dev/test                        │
│                                                          │
│  Multi-Instance GPU (MIG) - A100/H100 only               │
│  ├── Hardware-level partitioning                         │
│  ├── Full memory and compute isolation                   │
│  ├── Up to 7 instances per A100                          │
│  └── Each instance is a separate resource                │
│                                                          │
│  Multi-Process Service (MPS)                             │
│  ├── CUDA context sharing                                │
│  ├── Better utilization for small models                 │
│  └── Limited isolation                                   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

```yaml
# GPU time-slicing configuration (ConfigMap for GPU Operator)
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: gpu-operator
data:
  any: |
    version: v1
    sharing:
      timeSlicing:
        renameByDefault: false
        failRequestsGreaterThanOne: false
        resources:
          - name: nvidia.com/gpu
            replicas: 4    # Each physical GPU appears as 4 virtual GPUs
```

### 1.6 NVIDIA MIG (Multi-Instance GPU) Partitioning

MIG (Multi-Instance GPU) provides **hardware-level** partitioning on A100 and
H100 GPUs. Unlike time-slicing, MIG instances have dedicated memory, L2 cache,
and compute engines — providing full isolation suitable for multi-tenant
inference or smaller training jobs.

**A100 80 GB MIG profiles** (commonly used):

| Profile | GPU Instances | Memory per Instance | Compute Slices |
|---------|--------------|--------------------:|----------------|
| `1g.10gb` | 7 per GPU | 10 GB | 1 |
| `2g.20gb` | 3 per GPU | 20 GB | 2 |
| `3g.40gb` | 2 per GPU | 40 GB | 3 |
| `4g.40gb` | 1 per GPU | 40 GB | 4 |
| `7g.80gb` | 1 per GPU | 80 GB | 7 (full GPU) |

#### Enabling MIG via the GPU Operator

```bash
# Enable MIG on a node (requires node reboot)
kubectl label node gpu-node-1 nvidia.com/mig.config=all-1g.10gb

# The MIG Manager DaemonSet reconfigures the GPU and exposes instances
# Each instance appears as a separate resource:
kubectl describe node gpu-node-1 | grep nvidia.com/mig
# Allocatable:
#   nvidia.com/mig-1g.10gb: 7
```

```yaml
# Request a specific MIG instance in a Pod spec
apiVersion: v1
kind: Pod
metadata:
  name: inference-job
spec:
  containers:
    - name: model-server
      image: nvcr.io/nvidia/tritonserver:24.01-py3
      resources:
        limits:
          nvidia.com/mig-1g.10gb: 1     # Request 1 MIG instance (10 GB)
```

#### Mixed MIG Strategies

The GPU Operator's MIG Manager supports mixed strategies (different profiles on
the same node):

```bash
# Apply a mixed MIG config via a ConfigMap
kubectl label node gpu-node-1 nvidia.com/mig.config=mixed

# custom-mig-config ConfigMap (deployed with GPU Operator):
# Defines: [3g.40gb x2, 1g.10gb x1] on a single A100 80GB
```

MIG instances are reset when the node label changes; running workloads using
those GPU instances will be evicted.

---

## 2. NVIDIA GPU Operator

### 2.1 Architecture

The GPU Operator automates the management of all NVIDIA software components needed for GPU-enabled Kubernetes nodes:

```
NVIDIA GPU Operator Components:
┌─────────────────────────────────────────────────────────┐
│                   GPU Operator                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Manages (as DaemonSets):                          │ │
│  │ ┌───────────────┐ ┌──────────────────────┐        │ │
│  │ │ NVIDIA Driver │ │ Container Toolkit     │        │ │
│  │ │ (kernel module│ │ (nvidia-container-    │        │ │
│  │ │  installed on │ │  runtime + hook)      │        │ │
│  │ │  host)        │ │                       │        │ │
│  │ └───────────────┘ └──────────────────────┘        │ │
│  │ ┌───────────────┐ ┌──────────────────────┐        │ │
│  │ │ Device Plugin │ │ DCGM Exporter         │        │ │
│  │ │ (advertises   │ │ (GPU metrics for      │        │ │
│  │ │  GPUs to      │ │  Prometheus)          │        │ │
│  │ │  kubelet)     │ │                       │        │ │
│  │ └───────────────┘ └──────────────────────┘        │ │
│  │ ┌───────────────┐ ┌──────────────────────┐        │ │
│  │ │ GFD (GPU      │ │ MIG Manager           │        │ │
│  │ │  Feature      │ │ (partition A100/H100  │        │ │
│  │ │  Discovery)   │ │  GPUs into slices)    │        │ │
│  │ └───────────────┘ └──────────────────────┘        │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Installation

```bash
# Add the NVIDIA Helm repository
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# Install the GPU Operator
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=true \
  --set toolkit.enabled=true \
  --set devicePlugin.enabled=true \
  --set dcgmExporter.enabled=true \
  --set gfd.enabled=true \
  --set migManager.enabled=false \
  --set operator.defaultRuntime=containerd

# Verify installation
kubectl get pods -n gpu-operator
kubectl get nodes -o json | \
  jq '.items[] | {name: .metadata.name, gpus: .status.allocatable["nvidia.com/gpu"]}'
```

### 2.3 Custom GPU Operator Configuration

```yaml
# gpu-operator-values.yaml
operator:
  defaultRuntime: containerd

driver:
  enabled: true
  version: "550.54.15"          # Pin driver version
  manager:
    env:
      - name: ENABLE_GPU_DIRECT_STORAGE
        value: "true"

toolkit:
  enabled: true

devicePlugin:
  enabled: true
  config:
    name: time-slicing-config    # Enable time-slicing

dcgmExporter:
  enabled: true
  serviceMonitor:
    enabled: true                # Auto-create ServiceMonitor for Prometheus

gfd:
  enabled: true

migManager:
  enabled: true                  # Enable for A100/H100
  config:
    name: mig-config
    default: all-balanced        # MIG profile

nodeStatusExporter:
  enabled: true

validator:
  plugin:
    env:
      - name: WITH_WORKLOAD
        value: "true"
```

### 2.4 GPU Monitoring with DCGM

```bash
# Verify GPU metrics are being exported
kubectl port-forward -n gpu-operator svc/nvidia-dcgm-exporter 9400:9400
curl localhost:9400/metrics | grep DCGM

# Key GPU metrics:
# DCGM_FI_DEV_GPU_UTIL          - GPU utilization %
# DCGM_FI_DEV_MEM_COPY_UTIL     - Memory bandwidth utilization %
# DCGM_FI_DEV_FB_USED           - GPU memory used (MB)
# DCGM_FI_DEV_FB_FREE           - GPU memory free (MB)
# DCGM_FI_DEV_GPU_TEMP          - GPU temperature (C)
# DCGM_FI_DEV_POWER_USAGE       - Power draw (W)
# DCGM_FI_DEV_SM_CLOCK          - SM clock frequency (MHz)
```

```yaml
# Prometheus alerting for GPU health
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: gpu-alerts
  namespace: monitoring
spec:
  groups:
    - name: gpu-health
      rules:
        - alert: GPUTemperatureHigh
          expr: DCGM_FI_DEV_GPU_TEMP > 85
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "GPU temperature above 85C on {{ $labels.gpu }}"

        - alert: GPUMemoryExhausted
          expr: |
            DCGM_FI_DEV_FB_FREE
            / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE)
            < 0.05
          for: 10m
          labels:
            severity: critical
          annotations:
            summary: "GPU memory < 5% free on {{ $labels.gpu }}"

        - alert: GPUUnderutilized
          expr: DCGM_FI_DEV_GPU_UTIL < 10
          for: 30m
          labels:
            severity: info
          annotations:
            summary: "GPU utilization below 10% for 30 minutes"
```

---

## 3. Kubeflow Components

### 3.1 Kubeflow Architecture

Kubeflow provides a suite of ML tools that run natively on Kubernetes:

```
Kubeflow Ecosystem:
┌──────────────────────────────────────────────────────────┐
│                     Kubeflow Platform                    │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
│  │ Notebooks  │  │ Pipelines  │  │ Training Operator  │ │
│  │ (Jupyter   │  │ (ML DAGs   │  │ (TFJob, PyTorch-   │ │
│  │  in pods)  │  │  with Argo │  │  Job, MPIJob, etc.)│ │
│  │            │  │  Workflows)│  │                    │ │
│  └────────────┘  └────────────┘  └────────────────────┘ │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
│  │ KServe     │  │ Katib      │  │ Feature Store      │ │
│  │ (Model     │  │ (Hyper-    │  │ (Feast)            │ │
│  │  Serving)  │  │  parameter │  │                    │ │
│  │            │  │  Tuning)   │  │                    │ │
│  └────────────┘  └────────────┘  └────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### 3.2 Kubeflow Notebooks

```yaml
# Jupyter Notebook server with GPU
apiVersion: kubeflow.org/v1
kind: Notebook
metadata:
  name: research-notebook
  namespace: ml-team
  labels:
    app: research-notebook
spec:
  template:
    spec:
      containers:
        - name: notebook
          image: kubeflownotebookswg/jupyter-pytorch-cuda-full:v1.8.0
          resources:
            requests:
              cpu: "2"
              memory: 8Gi
            limits:
              cpu: "4"
              memory: 16Gi
              nvidia.com/gpu: 1
          ports:
            - containerPort: 8888
              name: notebook-port
              protocol: TCP
          volumeMounts:
            - name: workspace
              mountPath: /home/jovyan
            - name: datasets
              mountPath: /data
              readOnly: true
            - name: shm
              mountPath: /dev/shm
      volumes:
        - name: workspace
          persistentVolumeClaim:
            claimName: research-workspace-pvc
        - name: datasets
          persistentVolumeClaim:
            claimName: shared-datasets
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 4Gi
```

### 3.3 Kubeflow Pipelines

```python
# pipeline.py - Define an ML pipeline with Kubeflow Pipelines SDK
from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Model, Metrics

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn"]
)
def preprocess_data(
    raw_data_path: str,
    output_dataset: Output[Dataset],
):
    """Load and preprocess training data."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(raw_data_path)
    df = df.dropna()
    df.to_csv(output_dataset.path, index=False)

@dsl.component(
    base_image="nvcr.io/nvidia/pytorch:24.01-py3",
)
def train_model(
    dataset: Input[Dataset],
    model_output: Output[Model],
    metrics_output: Output[Metrics],
    epochs: int = 50,
    learning_rate: float = 0.001,
):
    """Train a PyTorch model on GPU."""
    import torch
    import torch.nn as nn

    # Training logic here...
    accuracy = 0.95  # placeholder
    metrics_output.log_metric("accuracy", accuracy)
    metrics_output.log_metric("epochs", epochs)

    torch.save(model.state_dict(), model_output.path)

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["kserve"]
)
def deploy_model(
    model: Input[Model],
    model_name: str,
    namespace: str = "ml-serving",
):
    """Deploy model to KServe."""
    from kubernetes import client, config
    config.load_incluster_config()
    # Deploy KServe InferenceService...

@dsl.pipeline(
    name="ml-training-pipeline",
    description="End-to-end ML training and deployment pipeline"
)
def ml_pipeline(
    raw_data_path: str = "gs://my-bucket/data/train.csv",
    epochs: int = 50,
    learning_rate: float = 0.001,
):
    preprocess_task = preprocess_data(raw_data_path=raw_data_path)

    train_task = train_model(
        dataset=preprocess_task.outputs["output_dataset"],
        epochs=epochs,
        learning_rate=learning_rate,
    ).set_accelerator_type("nvidia.com/gpu")\
     .set_accelerator_limit(1)\
     .set_cpu_request("4")\
     .set_memory_request("16Gi")

    deploy_task = deploy_model(
        model=train_task.outputs["model_output"],
        model_name="my-model",
    )

if __name__ == "__main__":
    compiler.Compiler().compile(ml_pipeline, "pipeline.yaml")
```

### 3.4 Training Operator

```yaml
# PyTorchJob for distributed training
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: resnet-training
  namespace: ml-team
spec:
  elasticPolicy:
    rdzvBackend: c10d
    minReplicas: 2
    maxReplicas: 8
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: registry.example.com/ml/resnet-trainer:v1.0
              command:
                - python
                - -m
                - torch.distributed.run
                - --nnodes=1:8
                - --nproc_per_node=4
                - --rdzv_backend=c10d
                - --rdzv_endpoint=$(MASTER_ADDR):$(MASTER_PORT)
                - train.py
                - --model=resnet50
                - --batch-size=64
                - --epochs=100
              resources:
                limits:
                  nvidia.com/gpu: 4
                requests:
                  cpu: "8"
                  memory: 32Gi
              volumeMounts:
                - name: shm
                  mountPath: /dev/shm
                - name: data
                  mountPath: /data
          volumes:
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: 16Gi
            - name: data
              persistentVolumeClaim:
                claimName: training-data
    Worker:
      replicas: 3
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: registry.example.com/ml/resnet-trainer:v1.0
              command:
                - python
                - -m
                - torch.distributed.run
                - --nnodes=1:8
                - --nproc_per_node=4
                - --rdzv_backend=c10d
                - --rdzv_endpoint=$(MASTER_ADDR):$(MASTER_PORT)
                - train.py
                - --model=resnet50
                - --batch-size=64
                - --epochs=100
              resources:
                limits:
                  nvidia.com/gpu: 4
                requests:
                  cpu: "8"
                  memory: 32Gi
              volumeMounts:
                - name: shm
                  mountPath: /dev/shm
                - name: data
                  mountPath: /data
                  readOnly: true
          volumes:
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: 16Gi
            - name: data
              persistentVolumeClaim:
                claimName: training-data
```

---

## 4. Distributed Training on Kubernetes

### 4.1 Distributed Training Patterns

```
Distributed Training Strategies:
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Data Parallelism                                        │
│  ├── Each worker has a full model copy                   │
│  ├── Data is split across workers                        │
│  ├── Gradients are synchronized (AllReduce)              │
│  └── Most common for standard training                   │
│                                                          │
│  Model Parallelism                                       │
│  ├── Model is split across GPUs/nodes                    │
│  ├── Each GPU holds part of the model                    │
│  ├── Required for models that do not fit on 1 GPU        │
│  └── Complex communication patterns                      │
│                                                          │
│  Pipeline Parallelism                                    │
│  ├── Model layers split across GPUs                      │
│  ├── Micro-batches pipeline through stages               │
│  ├── Reduces bubble time vs naive model parallel         │
│  └── Used in Megatron-LM, DeepSpeed                     │
│                                                          │
│  Hybrid (3D Parallelism)                                 │
│  ├── Data + Model + Pipeline combined                    │
│  ├── Used for LLM training (100B+ parameters)            │
│  └── Requires high-bandwidth interconnect (NVLink)       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 4.2 Network Configuration for Distributed Training

```yaml
# High-performance networking for training
apiVersion: v1
kind: Pod
metadata:
  name: training-worker
  annotations:
    k8s.v1.cni.cncf.io/networks: rdma-network  # Secondary RDMA network
spec:
  containers:
    - name: trainer
      image: nvcr.io/nvidia/pytorch:24.01-py3
      env:
        # NCCL configuration for optimal distributed training
        - name: NCCL_DEBUG
          value: INFO
        - name: NCCL_SOCKET_IFNAME
          value: eth0
        - name: NCCL_IB_DISABLE
          value: "0"          # Enable InfiniBand if available
        - name: NCCL_NET_GDR_LEVEL
          value: "5"          # GPUDirect RDMA level
        - name: NCCL_P2P_LEVEL
          value: NVL           # NVLink for intra-node
      resources:
        limits:
          nvidia.com/gpu: 8
          rdma/rdma_shared_device_a: 1  # RDMA device
```

### 4.3 MPI-Based Distributed Training

```yaml
# MPIJob for Horovod-based distributed training
apiVersion: kubeflow.org/v2beta1
kind: MPIJob
metadata:
  name: horovod-training
  namespace: ml-team
spec:
  slotsPerWorker: 4       # GPUs per worker
  runPolicy:
    cleanPodPolicy: Running
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        spec:
          containers:
            - name: launcher
              image: registry.example.com/ml/horovod-trainer:v1.0
              command:
                - mpirun
                - --allow-run-as-root
                - -np
                - "16"          # Total processes (4 workers * 4 GPUs)
                - -bind-to
                - none
                - -map-by
                - slot
                - -x
                - NCCL_DEBUG=INFO
                - -x
                - LD_LIBRARY_PATH
                - python
                - train.py
              resources:
                requests:
                  cpu: "1"
                  memory: 4Gi
    Worker:
      replicas: 4
      template:
        spec:
          containers:
            - name: worker
              image: registry.example.com/ml/horovod-trainer:v1.0
              resources:
                limits:
                  nvidia.com/gpu: 4
                requests:
                  cpu: "8"
                  memory: 32Gi
              volumeMounts:
                - name: shm
                  mountPath: /dev/shm
          volumes:
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: 16Gi
```

### 4.4 Training Job Monitoring

```go
// Go program to monitor training job progress
package main

import (
    "context"
    "fmt"
    "time"

    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
    "k8s.io/apimachinery/pkg/runtime/schema"
    "k8s.io/client-go/dynamic"
    "k8s.io/client-go/tools/clientcmd"
)

func main() {
    config, _ := clientcmd.BuildConfigFromFlags("",
        "/root/.kube/config")
    dynClient, _ := dynamic.NewForConfig(config)

    pytorchJobGVR := schema.GroupVersionResource{
        Group:    "kubeflow.org",
        Version:  "v1",
        Resource: "pytorchjobs",
    }

    // Watch PyTorchJob status
    for {
        job, err := dynClient.Resource(pytorchJobGVR).
            Namespace("ml-team").
            Get(context.TODO(), "resnet-training", metav1.GetOptions{})
        if err != nil {
            fmt.Printf("Error: %v\n", err)
            time.Sleep(10 * time.Second)
            continue
        }

        conditions, found, _ := unstructured.NestedSlice(
            job.Object, "status", "conditions",
        )
        if found {
            for _, c := range conditions {
                cond := c.(map[string]interface{})
                fmt.Printf("Condition: type=%s status=%s reason=%s\n",
                    cond["type"], cond["status"], cond["reason"])
            }
        }

        replicaStatuses, found, _ := unstructured.NestedMap(
            job.Object, "status", "replicaStatuses",
        )
        if found {
            for role, status := range replicaStatuses {
                s := status.(map[string]interface{})
                fmt.Printf("  %s: active=%v succeeded=%v failed=%v\n",
                    role, s["active"], s["succeeded"], s["failed"])
            }
        }

        time.Sleep(30 * time.Second)
    }
}
```

### 4.5 Volcano Batch Scheduler

Volcano (volcano.sh) is a CNCF graduated batch scheduler designed for ML and
HPC workloads. It extends the default Kubernetes scheduler with **gang
scheduling** (all-or-nothing pod allocation), fair queuing, and job-level
preemption.

#### Core Concepts

| Resource | Purpose |
|----------|---------|
| `Queue` | Named capacity pool; jobs compete within a queue |
| `PodGroup` | Minimum number of pods that must be co-scheduled (gang unit) |
| `Job` | Batch job with multiple tasks/roles (master + workers) |
| `VolcanoJob` | `batch.volcano.sh/v1alpha1 Job` — higher-level job CRD |

Gang scheduling ensures that a training job only starts when **all** required
pods can be scheduled simultaneously, preventing resource deadlocks (one job
holding partial resources while blocking another).

#### Installation

```bash
helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
helm install volcano volcano-sh/volcano \
  --namespace volcano-system \
  --create-namespace
```

#### Queue and VolcanoJob Example

```yaml
# Queue: reserve capacity for the ML team
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: ml-team
spec:
  weight: 1
  capability:
    cpu: "64"
    memory: 256Gi
    nvidia.com/gpu: "16"
---
# VolcanoJob: distributed PyTorch training with gang scheduling
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: pytorch-dist-train
  namespace: ml-workloads
spec:
  minAvailable: 5          # Gang constraint: all 5 pods must be schedulable together
  schedulerName: volcano
  queue: ml-team
  policies:
    - event: PodEvicted
      action: RestartJob   # Restart the whole job if any pod is evicted

  tasks:
    - replicas: 1
      name: master
      policies:
        - event: TaskCompleted
          action: CompleteJob
      template:
        spec:
          containers:
            - name: train
              image: nvcr.io/nvidia/pytorch:24.01-py3
              command: ["torchrun", "--nproc_per_node=4", "train.py"]
              resources:
                limits:
                  nvidia.com/gpu: 4

    - replicas: 4
      name: worker
      template:
        spec:
          containers:
            - name: train
              image: nvcr.io/nvidia/pytorch:24.01-py3
              command: ["torchrun", "--nproc_per_node=4", "train.py"]
              resources:
                limits:
                  nvidia.com/gpu: 4
```

#### PodGroup (low-level gang primitive)

```yaml
# Standalone PodGroup used directly with standard Pods
apiVersion: scheduling.volcano.sh/v1beta1
kind: PodGroup
metadata:
  name: inference-gang
  namespace: ml-workloads
spec:
  minMember: 3             # Start only when at least 3 pods can be scheduled
  queue: ml-team
  minResources:
    cpu: "12"
    nvidia.com/gpu: "3"
---
# Reference the PodGroup from your Pods
apiVersion: v1
kind: Pod
metadata:
  name: inference-0
  namespace: ml-workloads
  annotations:
    scheduling.volcano.sh/pod-group-name: inference-gang
spec:
  schedulerName: volcano
  containers:
    - name: server
      image: nvcr.io/nvidia/tritonserver:24.01-py3
      resources:
        limits:
          nvidia.com/gpu: 1
```

```bash
# Monitor Volcano queue utilization
kubectl get queues
kubectl describe queue ml-team

# Watch Volcano job status
kubectl get vcjob -n ml-workloads
kubectl describe vcjob pytorch-dist-train -n ml-workloads
```

---

## 5. Model Serving

### 5.1 KServe (formerly KFServing)

KServe provides a Kubernetes-native platform for model serving with auto-scaling, canary deployments, and multi-framework support:

```yaml
# KServe InferenceService for a PyTorch model
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: resnet-classifier
  namespace: ml-serving
  annotations:
    serving.kserve.io/enable-prometheus-scraping: "true"
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 10
    scaleTarget: 5             # Target concurrent requests per replica
    scaleMetric: concurrency
    pytorch:
      storageUri: "gs://models/resnet50/v1"
      protocolVersion: v2
      resources:
        requests:
          cpu: "2"
          memory: 4Gi
        limits:
          nvidia.com/gpu: 1
          memory: 8Gi
  transformer:
    minReplicas: 1
    containers:
      - name: preprocessor
        image: registry.example.com/ml/image-preprocessor:v1.0
        resources:
          requests:
            cpu: "1"
            memory: 2Gi
---
# Canary deployment: 90% to v1, 10% to v2
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: resnet-classifier
  namespace: ml-serving
spec:
  predictor:
    canaryTrafficPercent: 10
    pytorch:
      storageUri: "gs://models/resnet50/v2"
      protocolVersion: v2
      resources:
        limits:
          nvidia.com/gpu: 1
```

### 5.2 NVIDIA Triton Inference Server

```yaml
# Triton Inference Server deployment for multi-model serving
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-inference
  namespace: ml-serving
spec:
  replicas: 3
  selector:
    matchLabels:
      app: triton-inference
  template:
    metadata:
      labels:
        app: triton-inference
    spec:
      containers:
        - name: triton
          image: nvcr.io/nvidia/tritonserver:24.01-py3
          args:
            - tritonserver
            - --model-repository=s3://models/triton-repo
            - --model-control-mode=poll
            - --repository-poll-secs=60
            - --strict-model-config=false
            - --log-verbose=0
          ports:
            - containerPort: 8000    # HTTP
              name: http
            - containerPort: 8001    # gRPC
              name: grpc
            - containerPort: 8002    # Metrics
              name: metrics
          resources:
            requests:
              cpu: "4"
              memory: 8Gi
            limits:
              nvidia.com/gpu: 1
              memory: 16Gi
          livenessProbe:
            httpGet:
              path: /v2/health/live
              port: 8000
            initialDelaySeconds: 30
          readinessProbe:
            httpGet:
              path: /v2/health/ready
              port: 8000
            initialDelaySeconds: 30
          volumeMounts:
            - name: shm
              mountPath: /dev/shm
      volumes:
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 4Gi
---
apiVersion: v1
kind: Service
metadata:
  name: triton-inference
  namespace: ml-serving
spec:
  selector:
    app: triton-inference
  ports:
    - name: http
      port: 8000
      targetPort: 8000
    - name: grpc
      port: 8001
      targetPort: 8001
    - name: metrics
      port: 8002
      targetPort: 8002
---
# HPA for Triton based on GPU utilization
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: triton-hpa
  namespace: ml-serving
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: triton-inference
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Pods
      pods:
        metric:
          name: DCGM_FI_DEV_GPU_UTIL
        target:
          type: AverageValue
          averageValue: "70"
```

### 5.3 Seldon Core

```yaml
# Seldon Core deployment with A/B testing
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: recommendation-model
  namespace: ml-serving
spec:
  predictors:
    - name: model-a
      replicas: 2
      traffic: 80
      graph:
        name: classifier
        implementation: SKLEARN_SERVER
        modelUri: gs://models/sklearn/v1
        resources:
          requests:
            cpu: "1"
            memory: 2Gi
      componentSpecs:
        - spec:
            containers:
              - name: classifier
                resources:
                  requests:
                    cpu: "1"
                    memory: 2Gi
    - name: model-b
      replicas: 1
      traffic: 20
      graph:
        name: classifier
        implementation: SKLEARN_SERVER
        modelUri: gs://models/sklearn/v2
        resources:
          requests:
            cpu: "1"
            memory: 2Gi
```

### 5.4 Model Serving Performance

```bash
# Test inference latency
kubectl run load-test --image=curlimages/curl --rm -it -- \
  curl -s -w "\n%{time_total}s\n" \
  -X POST http://triton-inference:8000/v2/models/resnet50/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs":[{"name":"input","shape":[1,3,224,224],"datatype":"FP32","data":[...]}]}'

# Load testing with vegeta
kubectl run vegeta --image=peterevans/vegeta --rm -it -- \
  sh -c 'echo "POST http://triton-inference:8000/v2/models/resnet50/infer" | \
  vegeta attack -rate=100 -duration=60s | vegeta report'
```

---

## 6. ML Experiment Tracking

### 6.1 MLflow on Kubernetes

```yaml
# MLflow tracking server deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-server
  namespace: ml-platform
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
        - name: mlflow
          image: ghcr.io/mlflow/mlflow:2.10.0
          command:
            - mlflow
            - server
            - --host=0.0.0.0
            - --port=5000
            - --backend-store-uri=postgresql://mlflow:password@postgres:5432/mlflow
            - --default-artifact-root=s3://mlflow-artifacts/
            - --serve-artifacts
          ports:
            - containerPort: 5000
          env:
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: mlflow-s3-credentials
                  key: access-key
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: mlflow-s3-credentials
                  key: secret-key
          resources:
            requests:
              cpu: 500m
              memory: 1Gi
            limits:
              cpu: "2"
              memory: 4Gi
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-server
  namespace: ml-platform
spec:
  selector:
    app: mlflow
  ports:
    - port: 5000
      targetPort: 5000
```

### 6.2 Integrating MLflow with Training Jobs

```python
# In training code: log experiments to MLflow
import mlflow
import mlflow.pytorch
import os

# MLflow tracking URI from environment
mlflow.set_tracking_uri(os.environ.get(
    "MLFLOW_TRACKING_URI", "http://mlflow-server.ml-platform:5000"
))

mlflow.set_experiment("resnet-training")

with mlflow.start_run(run_name="gpu-training-v2"):
    # Log parameters
    mlflow.log_params({
        "model": "resnet50",
        "batch_size": 64,
        "learning_rate": 0.001,
        "epochs": 100,
        "optimizer": "adam",
        "gpu_count": 4,
    })

    # Training loop
    for epoch in range(100):
        train_loss = train_one_epoch(model, dataloader, optimizer)
        val_loss, val_acc = validate(model, val_loader)

        # Log metrics per epoch
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }, step=epoch)

    # Log the model artifact
    mlflow.pytorch.log_model(model, "model")

    # Log GPU utilization info
    mlflow.log_param("gpu_type", "NVIDIA-A100-80GB")
    mlflow.log_metric("training_time_hours", 4.5)
```

---

## 7. Spot and Preemptible Instances for Training

### 7.1 Why Spot Instances for ML

```
Cost Comparison (approximate):
┌────────────────────────────────────────────────────────┐
│  Instance Type     On-Demand     Spot        Savings   │
├────────────────────────────────────────────────────────┤
│  p3.2xlarge        $3.06/hr      $0.92/hr    70%      │
│  (1x V100)                                            │
│                                                        │
│  p3.8xlarge        $12.24/hr     $3.67/hr    70%      │
│  (4x V100)                                            │
│                                                        │
│  p4d.24xlarge      $32.77/hr     $12.78/hr   61%      │
│  (8x A100)                                            │
│                                                        │
│  100-epoch training on 4x A100:                        │
│  On-Demand: $32.77 * 24h = $786                        │
│  Spot:      $12.78 * 24h * 1.2 (interruptions) = $368  │
│  Savings:   $418 per training run                      │
└────────────────────────────────────────────────────────┘
```

### 7.2 Spot-Tolerant Training Architecture

```yaml
# Node pool with spot instances for training
# (AWS EKS Managed Node Group)
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: ml-cluster
  region: us-east-1

managedNodeGroups:
  - name: gpu-spot
    instanceTypes:
      - p3.8xlarge
      - p3.16xlarge
    spot: true
    minSize: 0
    maxSize: 10
    desiredCapacity: 0    # Scale to zero when idle
    labels:
      workload-type: ml-training
      instance-lifecycle: spot
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
      - key: instance-lifecycle
        value: spot
        effect: NoSchedule
    tags:
      k8s.io/cluster-autoscaler/enabled: "true"

  - name: gpu-ondemand
    instanceTypes:
      - p3.8xlarge
    minSize: 0
    maxSize: 4
    desiredCapacity: 0
    labels:
      workload-type: ml-serving
      instance-lifecycle: on-demand
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
```

### 7.3 Checkpointing for Spot Interruptions

```yaml
# Training job with checkpoint support
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: spot-safe-training
  namespace: ml-team
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure     # Restart on spot interruption
      template:
        spec:
          tolerations:
            - key: instance-lifecycle
              value: spot
              operator: Equal
              effect: NoSchedule
          nodeSelector:
            instance-lifecycle: spot
          # Spot interruption handler as sidecar
          containers:
            - name: pytorch
              image: registry.example.com/ml/trainer:v1.0
              command:
                - python
                - train.py
                - --checkpoint-dir=/checkpoints
                - --checkpoint-interval=300   # Every 5 minutes
                - --resume-from-checkpoint     # Auto-resume
              resources:
                limits:
                  nvidia.com/gpu: 4
              volumeMounts:
                - name: checkpoints
                  mountPath: /checkpoints
                - name: shm
                  mountPath: /dev/shm
            - name: spot-handler
              image: registry.example.com/ml/spot-handler:v1.0
              command:
                - /spot-handler
                - --checkpoint-dir=/checkpoints
                - --grace-period=120
              volumeMounts:
                - name: checkpoints
                  mountPath: /checkpoints
          volumes:
            - name: checkpoints
              persistentVolumeClaim:
                claimName: training-checkpoints  # Survives pod restart
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: 16Gi
```

### 7.4 AWS Spot Interruption Handler

```yaml
# AWS Node Termination Handler (DaemonSet)
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: aws-node-termination-handler
  namespace: kube-system
spec:
  selector:
    matchLabels:
      app: aws-node-termination-handler
  template:
    metadata:
      labels:
        app: aws-node-termination-handler
    spec:
      nodeSelector:
        instance-lifecycle: spot
      tolerations:
        - operator: Exists
      serviceAccountName: aws-node-termination-handler
      containers:
        - name: handler
          image: public.ecr.aws/aws-ec2/aws-node-termination-handler:v1.22.0
          env:
            - name: NODE_NAME
              valueFrom:
                fieldRef:
                  fieldPath: spec.nodeName
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: ENABLE_SPOT_INTERRUPTION_DRAINING
              value: "true"
            - name: ENABLE_REBALANCE_DRAINING
              value: "true"
            - name: EMIT_KUBERNETES_EVENTS
              value: "true"
          resources:
            requests:
              cpu: 50m
              memory: 64Mi
```

---

## 8. Resource Quotas for ML Teams

### 8.1 Namespace Isolation for ML Teams

```yaml
# Namespace with resource quotas for ML team
apiVersion: v1
kind: Namespace
metadata:
  name: ml-team-alpha
  labels:
    team: alpha
    environment: research
---
# GPU quota for the team
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: ml-team-alpha
spec:
  hard:
    requests.nvidia.com/gpu: "8"     # Max 8 GPUs total
    limits.nvidia.com/gpu: "8"
    requests.cpu: "32"
    requests.memory: 128Gi
    limits.cpu: "64"
    limits.memory: 256Gi
    pods: "50"
    persistentvolumeclaims: "20"
---
# Priority-based quota for training vs serving
apiVersion: v1
kind: ResourceQuota
metadata:
  name: training-quota
  namespace: ml-team-alpha
spec:
  hard:
    requests.nvidia.com/gpu: "6"     # 6 GPUs for training
  scopeSelector:
    matchExpressions:
      - scopeName: PriorityClass
        operator: In
        values:
          - ml-training
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: serving-quota
  namespace: ml-team-alpha
spec:
  hard:
    requests.nvidia.com/gpu: "2"     # 2 GPUs for serving
  scopeSelector:
    matchExpressions:
      - scopeName: PriorityClass
        operator: In
        values:
          - ml-serving
```

### 8.2 Priority Classes for ML Workloads

```yaml
# High priority for production serving
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: ml-serving
value: 1000000
globalDefault: false
description: "Production model serving workloads"
preemptionPolicy: PreemptLowerPriority
---
# Medium priority for training
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: ml-training
value: 100000
globalDefault: false
description: "ML training jobs"
preemptionPolicy: PreemptLowerPriority
---
# Low priority for experiments
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: ml-experiment
value: 10000
globalDefault: false
description: "Experimental and development ML workloads"
preemptionPolicy: PreemptLowerPriority
```

### 8.3 LimitRange for Default Resources

```yaml
# Default resource limits for ML namespace
apiVersion: v1
kind: LimitRange
metadata:
  name: ml-defaults
  namespace: ml-team-alpha
spec:
  limits:
    - type: Container
      default:
        cpu: "2"
        memory: 4Gi
      defaultRequest:
        cpu: 500m
        memory: 1Gi
      max:
        cpu: "32"
        memory: 128Gi
        nvidia.com/gpu: "4"
      min:
        cpu: 100m
        memory: 128Mi
    - type: PersistentVolumeClaim
      max:
        storage: 1Ti
      min:
        storage: 1Gi
```

### 8.4 GPU Utilization Tracking

```yaml
# Prometheus rules for tracking GPU utilization per team
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: gpu-utilization-tracking
  namespace: monitoring
spec:
  groups:
    - name: gpu-team-metrics
      interval: 60s
      rules:
        # GPU utilization per namespace
        - record: namespace:gpu_utilization:avg
          expr: |
            avg by (namespace) (
              DCGM_FI_DEV_GPU_UTIL
              * on (pod) group_left(namespace)
              kube_pod_info
            )

        # GPU hours consumed per namespace (for chargeback)
        - record: namespace:gpu_hours:increase1h
          expr: |
            sum by (namespace) (
              increase(
                DCGM_FI_DEV_GPU_UTIL{} [1h]
              ) / 100 / 3600
              * on (pod) group_left(namespace)
              kube_pod_info
            )

        - alert: GPUQuotaAlmostExhausted
          expr: |
            sum by (namespace) (
              kube_resourcequota{resource="requests.nvidia.com/gpu", type="used"}
            )
            /
            sum by (namespace) (
              kube_resourcequota{resource="requests.nvidia.com/gpu", type="hard"}
            )
            > 0.9
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "GPU quota > 90% used in {{ $labels.namespace }}"
```

---

## Exercises

### Exercise 1: GPU Pod Configuration

Write a complete Pod specification for a PyTorch training job that: (a) requests 2 NVIDIA A100 GPUs, (b) has 16 GiB shared memory for DataLoader workers, (c) mounts a PVC at `/data`, (d) only schedules on nodes with A100 GPUs, and (e) tolerates the GPU node taint.

<details><summary>Show Answer</summary>

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pytorch-training
  namespace: ml-team
  labels:
    app: pytorch-training
    workload: training
spec:
  restartPolicy: Never
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values:
                  - NVIDIA-A100-SXM4-80GB
                  - NVIDIA-A100-SXM4-40GB
                  - NVIDIA-A100-PCIE-40GB
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
  containers:
    - name: trainer
      image: nvcr.io/nvidia/pytorch:24.01-py3
      command:
        - python
        - -m
        - torch.distributed.launch
        - --nproc_per_node=2
        - train.py
        - --batch-size=128
        - --epochs=100
        - --data-dir=/data
      resources:
        requests:
          cpu: "8"
          memory: 32Gi
        limits:
          nvidia.com/gpu: 2
          memory: 64Gi
      env:
        - name: NCCL_DEBUG
          value: INFO
        - name: NCCL_SOCKET_IFNAME
          value: eth0
        - name: PYTHONUNBUFFERED
          value: "1"
      volumeMounts:
        - name: training-data
          mountPath: /data
          readOnly: true
        - name: shm
          mountPath: /dev/shm
        - name: output
          mountPath: /output
  volumes:
    - name: training-data
      persistentVolumeClaim:
        claimName: training-dataset-pvc
    - name: shm
      emptyDir:
        medium: Memory
        sizeLimit: 16Gi
    - name: output
      persistentVolumeClaim:
        claimName: training-output-pvc
```

</details>

### Exercise 2: KServe InferenceService with Autoscaling

Create a KServe InferenceService that: (a) serves a TensorFlow model from S3, (b) scales from 0 to 10 replicas based on concurrent requests, (c) uses 1 GPU per replica, (d) has a canary deployment sending 20% traffic to a v2 model, and (e) sets a 5-second request timeout.

<details><summary>Show Answer</summary>

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: image-classifier
  namespace: ml-serving
  annotations:
    serving.kserve.io/enable-prometheus-scraping: "true"
spec:
  predictor:
    minReplicas: 0
    maxReplicas: 10
    scaleTarget: 10            # Target 10 concurrent requests per replica
    scaleMetric: concurrency
    canaryTrafficPercent: 20   # 20% to canary (v2)
    timeout: 5                 # 5-second request timeout
    tensorflow:
      storageUri: "s3://models/image-classifier/v2"
      runtimeVersion: "2.14.0"
      resources:
        requests:
          cpu: "2"
          memory: 4Gi
        limits:
          nvidia.com/gpu: 1
          memory: 8Gi
      env:
        - name: TF_FORCE_GPU_ALLOW_GROWTH
          value: "true"
---
# The stable (v1) version is the previously deployed version.
# When the InferenceService is first created with canaryTrafficPercent,
# we need the previous revision to exist. Alternatively, use the
# following pattern with explicit traffic splitting:
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: image-classifier-split
  namespace: ml-serving
spec:
  predictor:
    minReplicas: 0
    maxReplicas: 10
    scaleTarget: 10
    scaleMetric: concurrency
    timeout: 5
    tensorflow:
      storageUri: "s3://models/image-classifier/v1"
      runtimeVersion: "2.14.0"
      resources:
        requests:
          cpu: "2"
          memory: 4Gi
        limits:
          nvidia.com/gpu: 1
          memory: 8Gi
```

To promote the canary after validation:

```bash
# Check canary metrics
kubectl get inferenceservice image-classifier -n ml-serving \
  -o jsonpath='{.status.components.predictor}'

# Promote canary (set traffic to 0, which makes the canary the new default)
kubectl patch inferenceservice image-classifier -n ml-serving \
  --type='json' \
  -p='[{"op": "remove", "path": "/spec/predictor/canaryTrafficPercent"}]'
```

</details>

### Exercise 3: Spot-Tolerant Training Setup

Design a complete spot-tolerant training setup that includes: (a) a node pool configuration for GPU spot instances, (b) a PyTorchJob that checkpoints every 10 minutes, (c) a PVC for checkpoints that persists across pod restarts, and (d) a PriorityClass that allows training to be preempted by serving workloads.

<details><summary>Show Answer</summary>

```yaml
# (d) PriorityClass for training (lower than serving)
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: spot-training
value: 50000
globalDefault: false
description: "Spot-based ML training - can be preempted by serving"
preemptionPolicy: PreemptLowerPriority
---
# (c) Persistent checkpoint storage
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-checkpoints
  namespace: ml-team
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: gp3-encrypted
  resources:
    requests:
      storage: 100Gi
---
# (b) PyTorchJob with checkpointing
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: spot-tolerant-training
  namespace: ml-team
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          priorityClassName: spot-training
          tolerations:
            - key: nvidia.com/gpu
              operator: Exists
              effect: NoSchedule
            - key: instance-lifecycle
              value: spot
              operator: Equal
              effect: NoSchedule
          nodeSelector:
            instance-lifecycle: spot
          terminationGracePeriodSeconds: 120
          containers:
            - name: pytorch
              image: registry.example.com/ml/trainer:v1.0
              command:
                - python
                - train.py
                - --checkpoint-dir=/checkpoints
                - --checkpoint-interval-sec=600
                - --resume-from-latest
                - --batch-size=128
                - --epochs=200
              resources:
                requests:
                  cpu: "8"
                  memory: 32Gi
                limits:
                  nvidia.com/gpu: 4
                  memory: 64Gi
              env:
                - name: NCCL_DEBUG
                  value: INFO
              volumeMounts:
                - name: checkpoints
                  mountPath: /checkpoints
                - name: data
                  mountPath: /data
                  readOnly: true
                - name: shm
                  mountPath: /dev/shm
              lifecycle:
                preStop:
                  exec:
                    command:
                      - /bin/sh
                      - -c
                      - |
                        echo "Received termination signal, saving checkpoint..."
                        kill -SIGUSR1 $(cat /tmp/trainer.pid)
                        sleep 90
          volumes:
            - name: checkpoints
              persistentVolumeClaim:
                claimName: training-checkpoints
            - name: data
              persistentVolumeClaim:
                claimName: training-dataset
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: 16Gi
```

**(a) Node pool configuration (EKS eksctl):**

```yaml
# eksctl ClusterConfig snippet
managedNodeGroups:
  - name: gpu-spot-training
    instanceTypes:
      - p3.8xlarge       # 4x V100
      - p3.16xlarge      # 8x V100
    spot: true
    minSize: 0
    maxSize: 8
    desiredCapacity: 0
    labels:
      workload-type: ml-training
      instance-lifecycle: spot
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
      - key: instance-lifecycle
        value: spot
        effect: NoSchedule
    tags:
      k8s.io/cluster-autoscaler/enabled: "true"
      k8s.io/cluster-autoscaler/ml-cluster: "owned"
    volumeSize: 200
    volumeType: gp3
```

</details>

### Exercise 4: GPU Resource Quota Design

Design a resource quota system for 3 ML teams that share a cluster with 24 GPUs total. Team A (research) gets 12 GPUs, Team B (production) gets 8 GPUs, Team C (experimentation) gets 4 GPUs. Include quota definitions, LimitRanges, and a mechanism to allow temporary GPU borrowing between teams.

<details><summary>Show Answer</summary>

```yaml
# Team A - Research (12 GPUs)
apiVersion: v1
kind: Namespace
metadata:
  name: ml-research
  labels:
    team: research
    gpu-tier: high
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: ml-research
spec:
  hard:
    requests.nvidia.com/gpu: "12"
    limits.nvidia.com/gpu: "12"
    requests.cpu: "96"
    requests.memory: 384Gi
    pods: "100"
---
apiVersion: v1
kind: LimitRange
metadata:
  name: gpu-limits
  namespace: ml-research
spec:
  limits:
    - type: Container
      max:
        nvidia.com/gpu: "8"
      default:
        cpu: "2"
        memory: 8Gi
      defaultRequest:
        cpu: "1"
        memory: 4Gi
---
# Team B - Production (8 GPUs)
apiVersion: v1
kind: Namespace
metadata:
  name: ml-production
  labels:
    team: production
    gpu-tier: medium
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: ml-production
spec:
  hard:
    requests.nvidia.com/gpu: "8"
    limits.nvidia.com/gpu: "8"
    requests.cpu: "64"
    requests.memory: 256Gi
    pods: "200"
---
apiVersion: v1
kind: LimitRange
metadata:
  name: gpu-limits
  namespace: ml-production
spec:
  limits:
    - type: Container
      max:
        nvidia.com/gpu: "4"
      default:
        cpu: "2"
        memory: 4Gi
      defaultRequest:
        cpu: 500m
        memory: 2Gi
---
# Team C - Experimentation (4 GPUs)
apiVersion: v1
kind: Namespace
metadata:
  name: ml-experiments
  labels:
    team: experiments
    gpu-tier: low
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: ml-experiments
spec:
  hard:
    requests.nvidia.com/gpu: "4"
    limits.nvidia.com/gpu: "4"
    requests.cpu: "32"
    requests.memory: 128Gi
    pods: "50"
---
apiVersion: v1
kind: LimitRange
metadata:
  name: gpu-limits
  namespace: ml-experiments
spec:
  limits:
    - type: Container
      max:
        nvidia.com/gpu: "2"
      default:
        cpu: "1"
        memory: 4Gi
      defaultRequest:
        cpu: 500m
        memory: 2Gi
---
# GPU Borrowing: Overflow namespace with lower priority
# When a team needs more GPUs temporarily, they submit to the
# overflow namespace with a lower PriorityClass.
# These pods can be preempted when the owning team needs their GPUs back.
apiVersion: v1
kind: Namespace
metadata:
  name: ml-overflow
  labels:
    purpose: gpu-borrowing
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: overflow-quota
  namespace: ml-overflow
spec:
  hard:
    requests.nvidia.com/gpu: "24"   # Can use any idle GPU
    limits.nvidia.com/gpu: "24"
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: gpu-overflow
value: 1000                          # Very low priority
globalDefault: false
description: "Overflow GPU workloads - will be preempted by regular team workloads"
preemptionPolicy: Never              # Do not preempt others
---
# Teams use this PriorityClass for overflow workloads:
# spec:
#   priorityClassName: gpu-overflow
```

This design allows:
- Each team has a guaranteed GPU allocation.
- The overflow namespace lets any team temporarily use idle GPUs from other teams.
- Overflow workloads have the lowest priority and use `preemptionPolicy: Never`, so they will be evicted when the owning team submits work.

</details>

### Exercise 5: End-to-End ML Pipeline

Design a Kubeflow Pipeline that: (a) preprocesses data from a PVC, (b) trains a model using a PyTorchJob with 2 GPUs, (c) evaluates the model and records metrics to MLflow, (d) conditionally deploys to KServe if accuracy exceeds 95%, and (e) sends a Slack notification with results. Write the pipeline definition in Python.

<details><summary>Show Answer</summary>

```python
from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Model, Metrics, Condition

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "pyarrow"]
)
def preprocess(
    data_path: str,
    output_dataset: Output[Dataset],
    train_split: Output[Dataset],
    val_split: Output[Dataset],
):
    """Preprocess raw data and create train/val splits."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(data_path)

    # Clean and transform
    df = df.dropna()
    df = df[df["label"].isin(range(1000))]  # ImageNet classes

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    df.to_parquet(output_dataset.path, index=False)
    train_df.to_parquet(train_split.path, index=False)
    val_df.to_parquet(val_split.path, index=False)


@dsl.component(base_image="nvcr.io/nvidia/pytorch:24.01-py3")
def train_model(
    train_data: Input[Dataset],
    model_output: Output[Model],
    epochs: int = 50,
    learning_rate: float = 0.001,
    batch_size: int = 64,
) -> float:
    """Train PyTorch model on GPU."""
    import torch
    import torch.nn as nn

    # Load data, build model, train...
    # (simplified for brevity)
    accuracy = 0.962  # placeholder
    torch.save({}, model_output.path + "/model.pt")
    return accuracy


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["mlflow", "boto3"]
)
def evaluate_and_log(
    model: Input[Model],
    val_data: Input[Dataset],
    mlflow_uri: str,
    metrics_output: Output[Metrics],
) -> float:
    """Evaluate model and log to MLflow."""
    import mlflow

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("production-pipeline")

    with mlflow.start_run():
        accuracy = 0.962  # from evaluation
        precision = 0.958
        recall = 0.961

        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        })
        mlflow.log_artifact(model.path)

    metrics_output.log_metric("accuracy", accuracy)
    metrics_output.log_metric("precision", precision)
    metrics_output.log_metric("recall", recall)

    return accuracy


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["kubernetes", "kserve"]
)
def deploy_model(
    model: Input[Model],
    model_name: str,
    namespace: str,
):
    """Deploy model to KServe."""
    from kubernetes import client, config
    import json

    config.load_incluster_config()
    api = client.CustomObjectsApi()

    isvc = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": model_name,
            "namespace": namespace,
        },
        "spec": {
            "predictor": {
                "minReplicas": 1,
                "maxReplicas": 5,
                "pytorch": {
                    "storageUri": model.uri,
                    "resources": {
                        "limits": {"nvidia.com/gpu": "1"},
                        "requests": {"cpu": "2", "memory": "4Gi"},
                    },
                },
            },
        },
    }

    try:
        api.create_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            body=isvc,
        )
    except client.ApiException as e:
        if e.status == 409:  # Already exists, patch instead
            api.patch_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                name=model_name,
                body=isvc,
            )


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["requests"]
)
def send_notification(
    model_name: str,
    accuracy: float,
    deployed: bool,
    slack_webhook_url: str,
):
    """Send Slack notification with results."""
    import requests
    import json

    status = "DEPLOYED" if deployed else "NOT DEPLOYED (accuracy < 95%)"
    message = {
        "text": (
            f"ML Pipeline Complete\n"
            f"Model: {model_name}\n"
            f"Accuracy: {accuracy:.4f}\n"
            f"Status: {status}"
        )
    }
    requests.post(slack_webhook_url, json=message)


@dsl.pipeline(
    name="e2e-ml-pipeline",
    description="End-to-end ML training, evaluation, and deployment"
)
def ml_pipeline(
    data_path: str = "/data/raw/dataset.csv",
    model_name: str = "image-classifier",
    serving_namespace: str = "ml-serving",
    mlflow_uri: str = "http://mlflow-server.ml-platform:5000",
    slack_webhook: str = "https://hooks.slack.com/services/...",
    epochs: int = 50,
    learning_rate: float = 0.001,
    accuracy_threshold: float = 0.95,
):
    # Step 1: Preprocess
    preprocess_task = preprocess(data_path=data_path)

    # Step 2: Train (with GPU)
    train_task = train_model(
        train_data=preprocess_task.outputs["train_split"],
        epochs=epochs,
        learning_rate=learning_rate,
    ).set_accelerator_type("nvidia.com/gpu") \
     .set_accelerator_limit(2) \
     .set_cpu_request("8") \
     .set_memory_request("32Gi")

    # Step 3: Evaluate and log
    eval_task = evaluate_and_log(
        model=train_task.outputs["model_output"],
        val_data=preprocess_task.outputs["val_split"],
        mlflow_uri=mlflow_uri,
    )

    # Step 4: Conditional deployment
    with Condition(
        eval_task.output > accuracy_threshold,
        name="accuracy-check"
    ):
        deploy_task = deploy_model(
            model=train_task.outputs["model_output"],
            model_name=model_name,
            namespace=serving_namespace,
        )

    # Step 5: Notification (always runs)
    notify_task = send_notification(
        model_name=model_name,
        accuracy=eval_task.output,
        deployed=(eval_task.output > accuracy_threshold),
        slack_webhook_url=slack_webhook,
    )

if __name__ == "__main__":
    compiler.Compiler().compile(ml_pipeline, "e2e_pipeline.yaml")
```

</details>

---

**Previous**: [17. Production Operations](./17_Production_Operations.md) | **Next**: [19. Capstone: Production Cluster](./19_Capstone_Production_Cluster.md)
