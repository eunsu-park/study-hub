# MLOps 학습 가이드

## 개요

MLOps(Machine Learning Operations)는 머신러닝 모델의 개발, 배포, 운영을 자동화하고 효율화하는 실무 분야입니다. 이 학습 자료는 Deep Learning 기초를 이수한 학습자를 대상으로, 실무에서 ML 시스템을 안정적으로 운영하기 위한 전반적인 프로세스와 도구를 다룹니다.

### 대상 독자
- Deep Learning 기초 완료자
- ML 모델을 프로덕션에 배포하고자 하는 개발자
- 데이터 과학팀과 협업하는 소프트웨어 엔지니어
- ML 시스템 운영을 담당하는 DevOps/SRE 엔지니어

---

## 학습 로드맵

```
MLOps 개요 → ML 라이프사이클 → MLflow 기초 → MLflow 고급 → W&B
     │                              │              │         │
     │                              ↓              ↓         ↓
     │                         실험 추적 ──── 모델 레지스트리 ──→ Kubeflow
     │                                                            │
     ↓                                                            ↓
Feature Store ← 드리프트/모니터링 ← TorchServe/Triton ← 모델 서빙 기초
     │
     ↓
실전 MLOps 프로젝트 (E2E 파이프라인)
```

---

## 파일 목록

| 파일 | 주제 | 난이도 | 핵심 내용 |
|------|------|--------|----------|
| [01_MLOps_Overview.md](./01_MLOps_Overview.md) | MLOps 개요 | ⭐ | MLOps 정의, DevOps vs MLOps, 성숙도 레벨, 도구 생태계 |
| [02_ML_Lifecycle.md](./02_ML_Lifecycle.md) | ML 라이프사이클 | ⭐⭐ | 프로젝트 단계, 재학습 트리거, 버전 관리 전략 |
| [03_MLflow_Basics.md](./03_MLflow_Basics.md) | MLflow 기초 | ⭐⭐ | Tracking, 실험 관리, 메트릭/파라미터 로깅, UI |
| [04_MLflow_Advanced.md](./04_MLflow_Advanced.md) | MLflow 고급 | ⭐⭐⭐ | Projects, Models, Registry, Serving |
| [05_Weights_and_Biases.md](./05_Weights_and_Biases.md) | Weights & Biases | ⭐⭐ | 실험 로깅, Sweeps, Artifacts, MLflow 비교 |
| [06_Kubeflow_Pipelines.md](./06_Kubeflow_Pipelines.md) | Kubeflow Pipelines | ⭐⭐⭐ | Pipeline SDK, 컴포넌트 작성, K8s 통합 |
| [07_Model_Registry.md](./07_Model_Registry.md) | 모델 레지스트리 | ⭐⭐ | 버전 관리, 스테이지 전환, CI/CD 통합 |
| [08_Model_Serving_Basics.md](./08_Model_Serving_Basics.md) | 모델 서빙 기초 | ⭐⭐ | REST API, gRPC, 배치 vs 실시간 추론 |
| [09_TorchServe_Triton.md](./09_TorchServe_Triton.md) | TorchServe & Triton | ⭐⭐⭐ | 핸들러 작성, 모델 최적화, 멀티모델 서빙 |
| [10_Drift_Detection_Monitoring.md](./10_Drift_Detection_Monitoring.md) | 드리프트 & 모니터링 | ⭐⭐⭐ | 데이터/모델 드리프트, Evidently AI, 알림 설정 |
| [11_Feature_Stores.md](./11_Feature_Stores.md) | Feature Store | ⭐⭐⭐ | Feast, 온라인/오프라인 스토어, 피처 서빙 |
| [12_Practical_MLOps_Project.md](./12_Practical_MLOps_Project.md) | 실전 MLOps 프로젝트 | ⭐⭐⭐⭐ | E2E 파이프라인, 자동 재학습, 프로젝트 구조 |
| [13_CICD_for_ML.md](./13_CICD_for_ML.md) | ML을 위한 CI/CD | ⭐⭐⭐ | ML 파이프라인 CI/CD, 데이터/모델 검증 게이트, 배포 전략, 롤백 |
| [14_DVC_Data_Version_Control.md](./14_DVC_Data_Version_Control.md) | DVC 데이터 버전 관리 | ⭐⭐⭐ | DVC 파이프라인, 실험 추적, CML, 원격 저장소 |
| [15_LLMOps.md](./15_LLMOps.md) | LLMOps | ⭐⭐⭐⭐ | 프롬프트 버전 관리, RAG 운영, LLM 평가, 가드레일, 비용 관리 |
| [16_Model_Testing_and_Validation.md](./16_Model_Testing_and_Validation.md) | 모델 테스트와 검증 | ⭐⭐⭐ | 데이터/모델/행동/인프라 테스트, 공정성, 섀도우 배포, CI/CD 자동화 |
| [17_Cloud_MLOps_Platforms.md](./17_Cloud_MLOps_Platforms.md) | 클라우드 MLOps 플랫폼 | ⭐⭐⭐⭐ | SageMaker, Vertex AI, Azure ML, 플랫폼 비교, 비용 최적화 |

---

## 환경 설정

### 필수 라이브러리 설치

```bash
# 기본 ML 라이브러리
pip install numpy pandas scikit-learn torch torchvision

# MLOps 도구
pip install mlflow wandb feast evidently

# 서빙 도구
pip install torchserve torch-model-archiver

# Kubeflow (Python SDK)
pip install kfp
```

### Docker 설정

```bash
# MLflow 서버 (Docker)
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

### 버전 확인

```python
import mlflow
import wandb
import feast

print(f"MLflow: {mlflow.__version__}")
print(f"W&B: {wandb.__version__}")
print(f"Feast: {feast.__version__}")
```

### 권장 버전
- Python: 3.9+
- MLflow: 2.8+
- Weights & Biases: 0.16+
- Feast: 0.35+
- Kubernetes: 1.25+

---

## 학습 순서 권장

### 1단계: 기초 이론 (01-02)
- MLOps 개념과 필요성 이해
- ML 프로젝트 라이프사이클 학습

### 2단계: 실험 관리 (03-05)
- MLflow로 실험 추적
- Weights & Biases 활용
- 메트릭, 파라미터, 아티팩트 관리

### 3단계: 파이프라인 (06-07)
- Kubeflow를 이용한 ML 파이프라인 구축
- 모델 레지스트리 운영

### 4단계: 모델 서빙 (08-09)
- REST/gRPC API 배포
- TorchServe, Triton 활용

### 5단계: 모니터링 & Feature Store (10-11)
- 드리프트 감지
- Feature Store 구축

### 6단계: 실전 프로젝트 (12)
- E2E MLOps 파이프라인 구축
- 자동 재학습 시스템

### 7단계: 심화 주제 (13-15)
- ML 시스템을 위한 CI/CD 파이프라인
- DVC를 이용한 데이터 버전 관리
- LLMOps: LLM 기반 애플리케이션 운영

---

## 관련 자료

### 공식 문서
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Weights & Biases Docs](https://docs.wandb.ai/)
- [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/)
- [Feast Documentation](https://docs.feast.dev/)
- [TorchServe](https://pytorch.org/serve/)
- [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/)

### 추천 도서
- "Introducing MLOps" - Mark Treveil
- "Machine Learning Engineering" - Andriy Burkov
- "Designing Machine Learning Systems" - Chip Huyen

### 연관 학습 자료
- [Docker 폴더](../Docker/): 컨테이너 기초
- [Deep_Learning 폴더](../Deep_Learning/): 딥러닝 모델 학습
- [Machine_Learning 폴더](../Machine_Learning/): ML 알고리즘 기초
