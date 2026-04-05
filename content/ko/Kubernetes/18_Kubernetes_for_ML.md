# 18. ML을 위한 Kubernetes(Kubernetes for ML)

**이전**: [17. 프로덕션 운영](./17_Production_Operations.md) | **다음**: [19. 캡스톤: 프로덕션 클러스터](./19_Capstone_Production_Cluster.md)

## 학습 목표

- 디바이스 플러그인(Device Plugin)과 리소스 제한을 사용한 Kubernetes에서의 GPU 스케줄링 구성
- 자동화된 GPU 관리를 위한 NVIDIA GPU Operator 배포 및 운영
- Notebooks, Pipelines, Training Operator를 포함한 ML 워크플로우용 Kubeflow 컴포넌트 사용
- KServe, Seldon Core, NVIDIA Triton을 사용한 대규모 모델 서빙
- 스팟 인스턴스(Spot Instance), 리소스 쿼터, 비용 관리를 통한 ML 워크로드 최적화

---

머신러닝 워크로드는 고유한 인프라 요구사항을 가지고 있습니다: 학습을 위한 GPU, 효율적으로 로드해야 하는 대규모 데이터셋, 장기 실행되는 분산 학습 작업, 저지연 모델 서빙. Kubernetes는 표준화된 방식으로 스케줄링, 리소스 관리, 확장성을 제공하기 때문에 ML 인프라의 선택 플랫폼이 되었습니다. 이 레슨에서는 GPU 스케줄링부터 분산 학습, 프로덕션 모델 서빙까지 Kubernetes에서의 ML의 전체 라이프사이클을 다룹니다.

## 목차

- [1. Kubernetes에서의 GPU 스케줄링](#1-kubernetes에서의-gpu-스케줄링)
- [2. NVIDIA GPU Operator](#2-nvidia-gpu-operator)
- [3. Kubeflow 컴포넌트](#3-kubeflow-컴포넌트)
- [4. Kubernetes에서의 분산 학습](#4-kubernetes에서의-분산-학습)
- [5. 모델 서빙](#5-모델-서빙)
- [6. ML 실험 추적](#6-ml-실험-추적)
- [7. 학습을 위한 스팟 및 선점형 인스턴스](#7-학습을-위한-스팟-및-선점형-인스턴스)
- [8. ML 팀을 위한 리소스 쿼터](#8-ml-팀을-위한-리소스-쿼터)
- [연습문제](#연습문제)

---

## 1. Kubernetes에서의 GPU 스케줄링

### 1.1 GPU 스케줄링 작동 방식

Kubernetes는 GPU를 디바이스 플러그인에 의해 광고되는 **확장 리소스(Extended Resources)**로 취급합니다. kubelet은 디바이스 플러그인 프레임워크를 통해 GPU를 발견하고, 스케줄러는 표준 리소스 요청을 사용하여 파드에 할당합니다.

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

### 1.2 파드 스펙에서 GPU 요청

```yaml
# 간단한 GPU 파드
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
          nvidia.com/gpu: 1      # 정확히 1 GPU 요청
          # GPU는 오버커밋 불가:
          # requests는 암묵적으로 limits와 동일하게 설정
        requests:
          cpu: "4"
          memory: 16Gi
      volumeMounts:
        - name: dataset
          mountPath: /data
        - name: shm                # PyTorch DataLoader를 위한 공유 메모리
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

### 1.3 멀티 GPU 학습 파드

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
          nvidia.com/gpu: 4       # 단일 노드에서 4 GPU
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
  # 4개 이상 GPU가 있는 노드에 스케줄링
  nodeSelector:
    gpu-count: "4"
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
```

### 1.4 GPU 노드 레이블과 테인트

```bash
# GPU 노드 레이블 확인 (NVIDIA 디바이스 플러그인 / GPU Operator가 설정)
kubectl get nodes -l nvidia.com/gpu.present=true \
  -o custom-columns=\
"NAME:.metadata.name,\
GPU_PRODUCT:.metadata.labels.nvidia\.com/gpu\.product,\
GPU_COUNT:.status.allocatable.nvidia\.com/gpu,\
GPU_MEM:.metadata.labels.nvidia\.com/gpu\.memory"

# 비GPU 워크로드 방지를 위해 GPU 노드에 테인트
kubectl taint nodes gpu-node-1 nvidia.com/gpu=present:NoSchedule

# 특정 GPU 유형에 대한 노드 어피니티
# (예: A100 노드에만 스케줄링)
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

### 1.5 GPU 공유 전략

```
GPU Sharing Options:
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  타임 슬라이싱 (Time-Slicing, NVIDIA)                     │
│  ├── 여러 파드가 시간 다중화로 GPU 공유                    │
│  ├── 메모리 격리 없음                                     │
│  └── 추론, 개발/테스트에 적합                              │
│                                                          │
│  멀티 인스턴스 GPU (MIG) - A100/H100 전용                  │
│  ├── 하드웨어 수준 파티셔닝                                │
│  ├── 완전한 메모리 및 컴퓨팅 격리                          │
│  ├── A100당 최대 7개 인스턴스                              │
│  └── 각 인스턴스가 별도의 리소스                           │
│                                                          │
│  멀티 프로세스 서비스 (MPS)                                │
│  ├── CUDA 컨텍스트 공유                                   │
│  ├── 소규모 모델의 더 나은 활용도                          │
│  └── 제한적인 격리                                        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

```yaml
# GPU 타임 슬라이싱 구성 (GPU Operator용 ConfigMap)
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
            replicas: 4    # 각 물리적 GPU가 4개 가상 GPU로 표시
```

### 1.6 NVIDIA MIG (Multi-Instance GPU) 파티셔닝

MIG(Multi-Instance GPU)는 A100 및 H100 GPU에서 **하드웨어 수준**의 파티셔닝을 제공합니다. 타임슬라이싱과 달리 MIG 인스턴스는 전용 메모리, L2 캐시, 컴퓨팅 엔진을 가지고 있어 멀티테넌트 추론이나 소규모 훈련 작업에 적합한 완전한 격리를 제공합니다.

**A100 80 GB MIG 프로파일** (일반적으로 사용):

| 프로파일 | GPU 인스턴스 수 | 인스턴스당 메모리 | 컴퓨팅 슬라이스 |
|---------|--------------|--------------------:|----------------|
| `1g.10gb` | GPU당 7개 | 10 GB | 1 |
| `2g.20gb` | GPU당 3개 | 20 GB | 2 |
| `3g.40gb` | GPU당 2개 | 40 GB | 3 |
| `4g.40gb` | GPU당 1개 | 40 GB | 4 |
| `7g.80gb` | GPU당 1개 | 80 GB | 7 (전체 GPU) |

#### GPU Operator를 통한 MIG 활성화

```bash
# 노드에서 MIG 활성화 (노드 재부팅 필요)
kubectl label node gpu-node-1 nvidia.com/mig.config=all-1g.10gb

# MIG Manager DaemonSet이 GPU를 재구성하고 인스턴스를 노출시킴
# 각 인스턴스는 별도 리소스로 표시됨:
kubectl describe node gpu-node-1 | grep nvidia.com/mig
# Allocatable:
#   nvidia.com/mig-1g.10gb: 7
```

```yaml
# Pod spec에서 특정 MIG 인스턴스 요청
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
          nvidia.com/mig-1g.10gb: 1     # MIG 인스턴스 1개 요청 (10 GB)
```

#### 혼합 MIG 전략

GPU Operator의 MIG Manager는 혼합 전략(같은 노드에서 다른 프로파일)을 지원합니다:

```bash
# ConfigMap을 통해 혼합 MIG 구성 적용
kubectl label node gpu-node-1 nvidia.com/mig.config=mixed

# custom-mig-config ConfigMap (GPU Operator와 함께 배포):
# 단일 A100 80GB에서 [3g.40gb x2, 1g.10gb x1]을 정의
```

MIG 인스턴스는 노드 레이블이 변경될 때 초기화됩니다; 해당 GPU 인스턴스를 사용하는 실행 중인 워크로드는 축출됩니다.

---

## 2. NVIDIA GPU Operator

### 2.1 아키텍처

GPU Operator는 GPU 지원 Kubernetes 노드에 필요한 모든 NVIDIA 소프트웨어 컴포넌트 관리를 자동화합니다:

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

### 2.2 설치

```bash
# NVIDIA Helm 레포지토리 추가
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# GPU Operator 설치
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

# 설치 확인
kubectl get pods -n gpu-operator
kubectl get nodes -o json | \
  jq '.items[] | {name: .metadata.name, gpus: .status.allocatable["nvidia.com/gpu"]}'
```

### 2.3 커스텀 GPU Operator 구성

```yaml
# gpu-operator-values.yaml
operator:
  defaultRuntime: containerd

driver:
  enabled: true
  version: "550.54.15"          # 드라이버 버전 고정
  manager:
    env:
      - name: ENABLE_GPU_DIRECT_STORAGE
        value: "true"

toolkit:
  enabled: true

devicePlugin:
  enabled: true
  config:
    name: time-slicing-config    # 타임슬라이싱 활성화

dcgmExporter:
  enabled: true
  serviceMonitor:
    enabled: true                # Prometheus용 ServiceMonitor 자동 생성

gfd:
  enabled: true

migManager:
  enabled: true                  # A100/H100에 활성화
  config:
    name: mig-config
    default: all-balanced        # MIG 프로파일

nodeStatusExporter:
  enabled: true

validator:
  plugin:
    env:
      - name: WITH_WORKLOAD
        value: "true"
```

### 2.4 GPU 모니터링 (DCGM)

```bash
# GPU 메트릭 내보내기 확인
kubectl port-forward -n gpu-operator svc/nvidia-dcgm-exporter 9400:9400
curl localhost:9400/metrics | grep DCGM

# 주요 GPU 메트릭:
# DCGM_FI_DEV_GPU_UTIL          - GPU 활용도 %
# DCGM_FI_DEV_MEM_COPY_UTIL     - 메모리 대역폭 활용도 %
# DCGM_FI_DEV_FB_USED           - GPU 메모리 사용량 (MB)
# DCGM_FI_DEV_FB_FREE           - GPU 메모리 여유량 (MB)
# DCGM_FI_DEV_GPU_TEMP          - GPU 온도 (C)
# DCGM_FI_DEV_POWER_USAGE       - 전력 소비 (W)
# DCGM_FI_DEV_SM_CLOCK          - SM 클럭 주파수 (MHz)
```

```yaml
# GPU 헬스에 대한 Prometheus 알림
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

## 3. Kubeflow 컴포넌트

### 3.1 Kubeflow 아키텍처

Kubeflow는 Kubernetes에서 네이티브로 실행되는 ML 도구 모음을 제공합니다:

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
# GPU가 있는 Jupyter Notebook 서버
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
# pipeline.py - Kubeflow Pipelines SDK로 ML 파이프라인 정의
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
# 분산 학습을 위한 PyTorchJob
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

## 4. Kubernetes에서의 분산 학습

### 4.1 PyTorchJob을 사용한 분산 학습

```yaml
# 멀티 노드 분산 학습
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: distributed-training
  namespace: ml-team
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: registry.example.com/ml/trainer:v1.0
              command:
                - python
                - -m
                - torch.distributed.run
                - --nproc_per_node=4
                - --nnodes=3
                - train.py
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
    Worker:
      replicas: 2
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: registry.example.com/ml/trainer:v1.0
              command:
                - python
                - -m
                - torch.distributed.run
                - --nproc_per_node=4
                - --nnodes=3
                - train.py
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

### 4.2 분산 학습을 위한 네트워크 구성

```yaml
# 학습을 위한 고성능 네트워킹
apiVersion: v1
kind: Pod
metadata:
  name: training-worker
  annotations:
    k8s.v1.cni.cncf.io/networks: rdma-network  # 보조 RDMA 네트워크
spec:
  containers:
    - name: trainer
      image: nvcr.io/nvidia/pytorch:24.01-py3
      env:
        # 최적 분산 학습을 위한 NCCL 구성
        - name: NCCL_DEBUG
          value: INFO
        - name: NCCL_SOCKET_IFNAME
          value: eth0
        - name: NCCL_IB_DISABLE
          value: "0"          # InfiniBand 사용 가능 시 활성화
        - name: NCCL_NET_GDR_LEVEL
          value: "5"          # GPUDirect RDMA 레벨
        - name: NCCL_P2P_LEVEL
          value: NVL           # 노드 내 NVLink 사용
      resources:
        limits:
          nvidia.com/gpu: 8
          rdma/rdma_shared_device_a: 1  # RDMA 디바이스
```

### 4.3 MPI 기반 분산 학습

```yaml
# Horovod 기반 분산 학습을 위한 MPIJob
apiVersion: kubeflow.org/v2beta1
kind: MPIJob
metadata:
  name: horovod-training
  namespace: ml-team
spec:
  slotsPerWorker: 4       # 워커당 GPU 수
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
                - "16"          # 총 프로세스 수 (4 워커 * 4 GPU)
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

### 4.4 훈련 작업 모니터링

```go
// 훈련 작업 진행 상황을 모니터링하는 Go 프로그램
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

    // PyTorchJob 상태 감시
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

### 4.5 Volcano 배치 스케줄러

Volcano(volcano.sh)는 ML 및 HPC 워크로드를 위해 설계된 CNCF 졸업 배치 스케줄러입니다. 기본 Kubernetes 스케줄러를 **갱 스케줄링**(전부-또는-없음 파드 할당), 공정한 큐잉, 작업 수준 선점으로 확장합니다.

#### 핵심 개념

| 리소스 | 목적 |
|--------|------|
| `Queue` | 명명된 용량 풀; 작업이 큐 내에서 경쟁 |
| `PodGroup` | 동시에 스케줄링되어야 하는 최소 파드 수 (갱 단위) |
| `Job` | 여러 태스크/역할을 가진 배치 작업 (마스터 + 워커) |
| `VolcanoJob` | `batch.volcano.sh/v1alpha1 Job` — 고수준 작업 CRD |

갱 스케줄링은 훈련 작업이 필요한 **모든** 파드를 동시에 스케줄링할 수 있을 때만 시작되도록 보장하여 리소스 데드락(한 작업이 부분 리소스를 점유하면서 다른 작업을 차단)을 방지합니다.

#### 설치

```bash
helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
helm install volcano volcano-sh/volcano \
  --namespace volcano-system \
  --create-namespace
```

#### Queue와 VolcanoJob 예시

```yaml
# Queue: ML 팀을 위한 용량 예약
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
# VolcanoJob: 갱 스케줄링을 사용한 분산 PyTorch 훈련
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: pytorch-dist-train
  namespace: ml-workloads
spec:
  minAvailable: 5          # 갱 제약: 5개 파드 모두 함께 스케줄 가능해야 함
  schedulerName: volcano
  queue: ml-team
  policies:
    - event: PodEvicted
      action: RestartJob   # 파드가 축출되면 전체 작업 재시작

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

#### PodGroup (저수준 갱 프리미티브)

```yaml
# 표준 파드와 직접 사용하는 독립형 PodGroup
apiVersion: scheduling.volcano.sh/v1beta1
kind: PodGroup
metadata:
  name: inference-gang
  namespace: ml-workloads
spec:
  minMember: 3             # 최소 3개 파드를 스케줄할 수 있을 때만 시작
  queue: ml-team
  minResources:
    cpu: "12"
    nvidia.com/gpu: "3"
---
# 파드에서 PodGroup 참조
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
# Volcano 큐 활용도 모니터링
kubectl get queues
kubectl describe queue ml-team

# Volcano 작업 상태 감시
kubectl get vcjob -n ml-workloads
kubectl describe vcjob pytorch-dist-train -n ml-workloads
```

---

## 5. 모델 서빙

### 5.1 KServe InferenceService

```yaml
# GPU를 사용한 KServe 모델 서빙
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llm-serving
  namespace: ml-serving
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 5
    scaleTarget: 10
    scaleMetric: concurrency
    pytorch:
      storageUri: "s3://models/llm/v1"
      resources:
        requests:
          cpu: "4"
          memory: 16Gi
        limits:
          nvidia.com/gpu: 1
          memory: 32Gi
```

### 5.2 NVIDIA Triton Inference Server

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-inference
  namespace: ml-serving
spec:
  replicas: 2
  selector:
    matchLabels:
      app: triton
  template:
    metadata:
      labels:
        app: triton
    spec:
      containers:
        - name: triton
          image: nvcr.io/nvidia/tritonserver:24.01-py3
          args:
            - tritonserver
            - --model-repository=s3://models/triton-repo
            - --model-control-mode=poll
            - --repository-poll-secs=30
          ports:
            - containerPort: 8000
              name: http
            - containerPort: 8001
              name: grpc
            - containerPort: 8002
              name: metrics
          resources:
            requests:
              cpu: "4"
              memory: 8Gi
            limits:
              nvidia.com/gpu: 1
              memory: 16Gi
          readinessProbe:
            httpGet:
              path: /v2/health/ready
              port: 8000
            initialDelaySeconds: 30
          livenessProbe:
            httpGet:
              path: /v2/health/live
              port: 8000
            initialDelaySeconds: 30
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
```

### 5.3 Seldon Core

```yaml
# A/B 테스트가 있는 Seldon Core 배포
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

### 5.4 모델 서빙 성능

```bash
# 추론 지연시간 테스트
kubectl run load-test --image=curlimages/curl --rm -it -- \
  curl -s -w "\n%{time_total}s\n" \
  -X POST http://triton-inference:8000/v2/models/resnet50/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs":[{"name":"input","shape":[1,3,224,224],"datatype":"FP32","data":[...]}]}'

# vegeta를 사용한 부하 테스트
kubectl run vegeta --image=peterevans/vegeta --rm -it -- \
  sh -c 'echo "POST http://triton-inference:8000/v2/models/resnet50/infer" | \
  vegeta attack -rate=100 -duration=60s | vegeta report'
```

---

## 6. ML 실험 추적

### 6.1 Kubernetes에서의 MLflow

```yaml
# MLflow 추적 서버 배포
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

### 6.2 MLflow와 학습 작업 통합

```python
# 학습 코드에서: MLflow에 실험 기록
import mlflow
import mlflow.pytorch
import os

# 환경변수에서 MLflow 추적 URI 가져오기
mlflow.set_tracking_uri(os.environ.get(
    "MLFLOW_TRACKING_URI", "http://mlflow-server.ml-platform:5000"
))

mlflow.set_experiment("resnet-training")

with mlflow.start_run(run_name="gpu-training-v2"):
    # 파라미터 기록
    mlflow.log_params({
        "model": "resnet50",
        "batch_size": 64,
        "learning_rate": 0.001,
        "epochs": 100,
        "optimizer": "adam",
        "gpu_count": 4,
    })

    # 학습 루프
    for epoch in range(100):
        train_loss = train_one_epoch(model, dataloader, optimizer)
        val_loss, val_acc = validate(model, val_loader)

        # 에포크별 메트릭 기록
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }, step=epoch)

    # 모델 아티팩트 기록
    mlflow.pytorch.log_model(model, "model")

    # GPU 활용 정보 기록
    mlflow.log_param("gpu_type", "NVIDIA-A100-80GB")
    mlflow.log_metric("training_time_hours", 4.5)
```

---

## 7. 학습을 위한 스팟 및 선점형 인스턴스

### 7.1 ML에 스팟 인스턴스를 사용하는 이유

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
│  4x A100에서 100에포크 학습:                            │
│  On-Demand: $32.77 * 24h = $786                        │
│  Spot:      $12.78 * 24h * 1.2 (중단) = $368           │
│  절감:   학습 실행당 $418                               │
└────────────────────────────────────────────────────────┘
```

### 7.2 스팟 허용 학습 아키텍처

```yaml
# 학습을 위한 스팟 인스턴스 노드 풀
# (AWS EKS 관리형 노드 그룹)
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
    desiredCapacity: 0    # 유휴 시 0으로 스케일다운
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

### 7.3 스팟 중단에 대비한 체크포인팅

```yaml
# 체크포인트 지원이 있는 학습 작업
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: spot-safe-training
  namespace: ml-team
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure     # 스팟 중단 시 재시작
      template:
        spec:
          tolerations:
            - key: instance-lifecycle
              value: spot
              operator: Equal
              effect: NoSchedule
          nodeSelector:
            instance-lifecycle: spot
          containers:
            - name: pytorch
              image: registry.example.com/ml/trainer:v1.0
              command:
                - python
                - train.py
                - --checkpoint-dir=/checkpoints
                - --checkpoint-interval=300   # 5분마다
                - --resume-from-checkpoint     # 자동 재개
              resources:
                limits:
                  nvidia.com/gpu: 4
              volumeMounts:
                - name: checkpoints
                  mountPath: /checkpoints
                - name: shm
                  mountPath: /dev/shm
          volumes:
            - name: checkpoints
              persistentVolumeClaim:
                claimName: training-checkpoints  # 파드 재시작에도 유지
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: 16Gi
```

### 7.4 AWS 스팟 중단 핸들러

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

## 8. ML 팀을 위한 리소스 쿼터

### 8.1 ML 팀을 위한 네임스페이스 격리

```yaml
# ML 팀을 위한 리소스 쿼터가 있는 네임스페이스
apiVersion: v1
kind: Namespace
metadata:
  name: ml-team-alpha
  labels:
    team: alpha
    environment: research
---
# 팀의 GPU 쿼터
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: ml-team-alpha
spec:
  hard:
    requests.nvidia.com/gpu: "8"     # 총 최대 8 GPU
    limits.nvidia.com/gpu: "8"
    requests.cpu: "32"
    requests.memory: 128Gi
    limits.cpu: "64"
    limits.memory: 256Gi
    pods: "50"
    persistentvolumeclaims: "20"
```

### 8.2 ML 워크로드를 위한 우선순위 클래스(PriorityClass)

```yaml
# 프로덕션 서빙에 높은 우선순위
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: ml-serving
value: 1000000
globalDefault: false
description: "프로덕션 모델 서빙 워크로드"
preemptionPolicy: PreemptLowerPriority
---
# 학습에 중간 우선순위
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: ml-training
value: 100000
globalDefault: false
description: "ML 학습 작업"
preemptionPolicy: PreemptLowerPriority
---
# 실험에 낮은 우선순위
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: ml-experiment
value: 10000
globalDefault: false
description: "실험 및 개발 ML 워크로드"
preemptionPolicy: PreemptLowerPriority
```

### 8.3 기본 리소스를 위한 LimitRange

```yaml
# ML 네임스페이스에 대한 기본 리소스 제한
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

### 8.4 GPU 활용도 추적

```yaml
# 팀별 GPU 활용도 추적을 위한 Prometheus 규칙
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
        # 네임스페이스별 GPU 활용도
        - record: namespace:gpu_utilization:avg
          expr: |
            avg by (namespace) (
              DCGM_FI_DEV_GPU_UTIL
              * on (pod) group_left(namespace)
              kube_pod_info
            )

        # 네임스페이스별 GPU 시간 소비 (차지백(chargeback)용)
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

## 연습문제

### 연습문제 1: GPU 파드 구성

다음 조건의 PyTorch 학습 작업을 위한 완전한 Pod 스펙을 작성하세요: (a) 2개의 NVIDIA A100 GPU 요청, (b) DataLoader 워커를 위한 16 GiB 공유 메모리, (c) `/data`에 PVC 마운트, (d) A100 GPU가 있는 노드에만 스케줄링, (e) GPU 노드 테인트 허용.

<details><summary>정답 보기</summary>

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

### 연습문제 2: KServe InferenceService와 오토스케일링

다음 조건의 KServe InferenceService를 작성하세요: (a) S3의 TensorFlow 모델 서빙, (b) 동시 요청 기반으로 0-10 레플리카 스케일링, (c) 레플리카당 1 GPU 사용, (d) v2 모델에 20% 트래픽을 보내는 카나리 배포, (e) 5초 요청 타임아웃 설정.

<details><summary>정답 보기</summary>

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
    scaleTarget: 10            # 레플리카당 10개 동시 요청 대상
    scaleMetric: concurrency
    canaryTrafficPercent: 20   # 카나리(v2)에 20%
    timeout: 5                 # 5초 요청 타임아웃
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
```

카나리 검증 후 프로모트:

```bash
# 카나리 메트릭 확인
kubectl get inferenceservice image-classifier -n ml-serving \
  -o jsonpath='{.status.components.predictor}'

# 카나리 프로모트 (트래픽을 0으로 설정하면 카나리가 새 기본값이 됨)
kubectl patch inferenceservice image-classifier -n ml-serving \
  --type='json' \
  -p='[{"op": "remove", "path": "/spec/predictor/canaryTrafficPercent"}]'
```

</details>

### 연습문제 3: 스팟 허용 학습 설정

다음을 포함하는 완전한 스팟 허용 학습 설정을 설계하세요: (a) GPU 스팟 인스턴스용 노드 풀 구성, (b) 10분마다 체크포인트하는 PyTorchJob, (c) 파드 재시작에도 유지되는 체크포인트용 PVC, (d) 서빙 워크로드에 의해 선점될 수 있는 PriorityClass.

<details><summary>정답 보기</summary>

```yaml
# (d) 학습용 PriorityClass (서빙보다 낮음)
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: spot-training
value: 50000
globalDefault: false
description: "스팟 기반 ML 학습 - 서빙에 의해 선점 가능"
preemptionPolicy: PreemptLowerPriority
---
# (c) 영구 체크포인트 스토리지
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
# (b) 체크포인팅이 있는 PyTorchJob
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
              volumeMounts:
                - name: checkpoints
                  mountPath: /checkpoints
                - name: data
                  mountPath: /data
                  readOnly: true
                - name: shm
                  mountPath: /dev/shm
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

</details>

### 연습문제 4: GPU 리소스 쿼터 설계

총 24개 GPU를 공유하는 3개 ML 팀을 위한 리소스 쿼터 시스템을 설계하세요. Team A(연구)는 12 GPU, Team B(프로덕션)는 8 GPU, Team C(실험)는 4 GPU를 받습니다. 쿼터 정의, LimitRanges, 팀 간 임시 GPU 차용을 허용하는 메커니즘을 포함하세요.

<details><summary>정답 보기</summary>

```yaml
# Team A - 연구 (12 GPU)
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
# Team B - 프로덕션 (8 GPU)
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
# Team C - 실험 (4 GPU)
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
# GPU 차용: 낮은 우선순위의 오버플로 네임스페이스
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
    requests.nvidia.com/gpu: "24"   # 유휴 GPU 모두 사용 가능
    limits.nvidia.com/gpu: "24"
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: gpu-overflow
value: 1000                          # 매우 낮은 우선순위
globalDefault: false
description: "오버플로 GPU 워크로드 - 일반 팀 워크로드에 의해 선점됨"
preemptionPolicy: Never              # 다른 워크로드를 선점하지 않음
```

이 설계의 특징:
- 각 팀에 보장된 GPU 할당이 있습니다.
- 오버플로 네임스페이스를 통해 어떤 팀이든 다른 팀의 유휴 GPU를 임시로 사용할 수 있습니다.
- 오버플로 워크로드는 가장 낮은 우선순위이며 `preemptionPolicy: Never`를 사용하므로, 소유 팀이 작업을 제출하면 퇴거됩니다.

</details>

### 연습문제 5: 엔드투엔드 ML 파이프라인

다음 조건의 Kubeflow Pipeline을 설계하세요: (a) PVC에서 데이터 전처리, (b) 2 GPU를 사용한 PyTorchJob으로 모델 학습, (c) 모델 평가 후 MLflow에 메트릭 기록, (d) 정확도가 95%를 초과하면 조건부로 KServe에 배포, (e) 결과와 함께 Slack 알림 전송. 파이프라인 정의를 Python으로 작성하세요.

<details><summary>정답 보기</summary>

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
    """원시 데이터 전처리 및 학습/검증 분할 생성."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(data_path)
    df = df.dropna()
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
) -> float:
    """GPU에서 PyTorch 모델 학습."""
    import torch
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
    """모델 평가 및 MLflow에 기록."""
    import mlflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("production-pipeline")
    with mlflow.start_run():
        accuracy = 0.962
        mlflow.log_metrics({"accuracy": accuracy})
        mlflow.log_artifact(model.path)
    metrics_output.log_metric("accuracy", accuracy)
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
    """KServe에 모델 배포."""
    from kubernetes import client, config
    config.load_incluster_config()
    # InferenceService 생성/업데이트 로직...


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
    """결과와 함께 Slack 알림 전송."""
    import requests
    status = "DEPLOYED" if deployed else "NOT DEPLOYED (accuracy < 95%)"
    message = {"text": f"ML Pipeline Complete\nModel: {model_name}\nAccuracy: {accuracy:.4f}\nStatus: {status}"}
    requests.post(slack_webhook_url, json=message)


@dsl.pipeline(
    name="e2e-ml-pipeline",
    description="엔드투엔드 ML 학습, 평가, 배포"
)
def ml_pipeline(
    data_path: str = "/data/raw/dataset.csv",
    model_name: str = "image-classifier",
    serving_namespace: str = "ml-serving",
    mlflow_uri: str = "http://mlflow-server.ml-platform:5000",
    slack_webhook: str = "https://hooks.slack.com/services/...",
    epochs: int = 50,
    accuracy_threshold: float = 0.95,
):
    preprocess_task = preprocess(data_path=data_path)

    train_task = train_model(
        train_data=preprocess_task.outputs["train_split"],
        epochs=epochs,
    ).set_accelerator_type("nvidia.com/gpu") \
     .set_accelerator_limit(2) \
     .set_cpu_request("8") \
     .set_memory_request("32Gi")

    eval_task = evaluate_and_log(
        model=train_task.outputs["model_output"],
        val_data=preprocess_task.outputs["val_split"],
        mlflow_uri=mlflow_uri,
    )

    with Condition(eval_task.output > accuracy_threshold, name="accuracy-check"):
        deploy_task = deploy_model(
            model=train_task.outputs["model_output"],
            model_name=model_name,
            namespace=serving_namespace,
        )

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

**이전**: [17. 프로덕션 운영](./17_Production_Operations.md) | **다음**: [19. 캡스톤: 프로덕션 클러스터](./19_Capstone_Production_Cluster.md)
