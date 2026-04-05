[← 이전: 16. 모델 테스트와 검증](16_Model_Testing_and_Validation.md) | [다음: 개요 →](00_Overview.md)

# 클라우드 MLOps 플랫폼(Cloud MLOps Platforms)

## 학습 목표

1. 클라우드 네이티브(Cloud-Native) MLOps 플랫폼이 실험에서 프로덕션까지의 경로를 단축하는 이유 이해
2. AWS SageMaker 생태계 탐색: Studio, Pipelines, Feature Store, Clarify
3. Google Vertex AI의 Pipelines 및 Model Monitoring으로 ML 워크플로우 구축 및 배포
4. Azure Machine Learning을 활용한 학습, 배포, 책임 AI 분석
5. 구조화된 기능 매트릭스를 이용한 세 주요 플랫폼 비교
6. MLflow와 Kubeflow를 추상화 레이어로 활용하는 멀티 클라우드(Multi-Cloud) 및 벤더 중립 전략 평가
7. 성능을 희생하지 않으면서 클라우드 ML 비용을 절감하는 비용 최적화(Cost Optimization) 전략 적용

---

## 개요

ML 플랫폼을 처음부터 구축하려면 수십 개의 컴포넌트를 조합해야 합니다. 실험 추적, 파이프라인 오케스트레이션(Pipeline Orchestration), 모델 서빙(Model Serving), 피처 스토어(Feature Store), 모니터링 등이 필요합니다. 클라우드 공급업체는 이러한 컴포넌트들의 관리형 버전을 제공하며, 이는 컴퓨팅, 스토리지, 네트워킹 인프라와 긴밀하게 통합되어 있습니다. 트레이드오프는 명확합니다. 벤더 의존성이 생기지만 프로덕션 배포까지의 시간이 단축됩니다.

이 레슨에서는 세 가지 주요 클라우드 MLOps 플랫폼인 AWS SageMaker, Google Vertex AI, Azure Machine Learning을 실용적인 관점에서 살펴봅니다. 빠르게 변화하는 상세 문서 대신, 아키텍처 패턴, SDK 사용법, 그리고 적합한 플랫폼을 선택하는(또는 아예 선택하지 않는) 의사결정 프레임워크에 초점을 맞춥니다.

> **비유**: 클라우드 MLOps 플랫폼을 선택하는 것은 항공사를 선택하는 것과 같습니다. 세 가지 모두 목적지(프로덕션의 모델)에 데려다주지만, 편의성(UX와 개발자 경험), 노선(통합 생태계), 마일리지 프로그램(생태계 종속성), 가격(사용량 기반 vs 예약 용량)이 다릅니다. 어떤 여행자는 특정 항공사의 라운지를 선호하고, 다른 여행자는 가장 저렴한 티켓을 우선시합니다. 최선의 선택은 출발지와 비행 빈도에 따라 달라집니다.

---

## 1. 클라우드 네이티브 MLOps를 사용하는 이유

### 1.1 자체 구축 vs 구매 결정

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

### 1.2 클라우드 MLOps 아키텍처 패턴

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

### 2.1 SageMaker 생태계 개요

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

### 2.2 SageMaker 학습 작업(Training Job)

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
    # SKLearn 추정기(Estimator)를 사용하는 이유? (vs 일반 Estimator)
    #   SKLearn은 사전 구축된 AWS 컨테이너를 사용해 Dockerfile이 필요 없으며,
    #   프레임워크 버전을 CloudWatch 메타데이터에 자동 기록해
    #   수개월 후에도 결과를 재현할 수 있게 해준다
    estimator = SKLearn(
        entry_point="train.py",              # Your training script
        source_dir="src/",                   # Directory with dependencies
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",        # 4 vCPU, 16 GB RAM
        framework_version="1.2-1",           # scikit-learn version
        py_version="py3",
        # output_path를 로컬 디스크 대신 S3로 지정하는 이유?
        #   학습 인스턴스(Instance)는 임시적이어서 인스턴스 종료 후 소멸되지만,
        #   S3는 인스턴스 종료 이후에도 유지되는 유일한 영구 스토리지다
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
        # max_wait > max_run으로 설정하는 이유? (7200 - 3600 = 3600초 여유)
        #   이 차이(3600초)가 SageMaker가 스팟 인스턴스(Spot Instance)를
        #   기다리는 최대 대기 시간이다. 짧은 용량 부족 상황을 견딜 만큼
        #   충분히 크게 설정해야 한다
        max_wait=7200,                       # Max wait for spot (seconds)
        max_run=3600,                        # Max training time (seconds)
    )

    # Define input data channels
    # Why channels? SageMaker mounts data to specific paths in the container
    train_input = TrainingInput(
        s3_data="s3://my-bucket/data/train/",
        # content_type을 명시하는 이유?
        #   SageMaker가 최적의 데이터 로딩 방식(파이프(Pipe) 모드 vs 파일(File) 모드)을
        #   선택하는 데 사용된다. CSV는 대용량 데이터셋에 최적화된
        #   SageMaker의 스트리밍 로딩 방식을 트리거한다
        content_type="text/csv",
    )
    test_input = TrainingInput(
        s3_data="s3://my-bucket/data/test/",
        content_type="text/csv",
    )

    # Launch training — this is asynchronous by default
    # job_name을 명시하는 이유?
    #   지정하지 않으면 SageMaker가 UUID를 자동 생성하는데,
    #   의미 있는 이름이 있어야 수십 개의 실험에서 CloudWatch 로그 필터링과
    #   비용 할당 태그 관리가 훨씬 쉬워진다
    estimator.fit(
        inputs={"train": train_input, "test": test_input},
        job_name="sklearn-classifier-v1",
        wait=True,  # Set False for async (check status in Studio)
    )

    return estimator
```

### 2.3 SageMaker 파이프라인(Pipelines)

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

    # 커스텀 배포 단계 대신 RegisterModel을 사용하는 이유?
    #   RegisterModel은 메트릭(Metrics), 계보(Lineage), 승인 상태를
    #   모델 레지스트리(Model Registry)에 버전별로 기록한다.
    #   담당자 또는 CI 게이트가 승인해야만 라이브 엔드포인트로 배포된다
    register_step = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        # 추론 인스턴스(Inference)와 변환 인스턴스(Transform)를 구분하는 이유?
        #   실시간 엔드포인트(Endpoint)와 배치 변환(Batch Transform)은
        #   지연(Latency)/처리량(Throughput) 프로파일이 다르다.
        #   둘 다 지정하면 Studio에서 상황에 맞는 배포 옵션을 제안받을 수 있다
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        # 모델 패키지 그룹(Model Package Group)을 사용하는 이유?
        #   그룹은 한 모델 패밀리의 모든 버전(v1/v2/v3)을 묶어서,
        #   Studio에서 승인 전 버전별 비교를 바로 할 수 있게 해준다
        model_package_group_name="my-model-group",
    )

    condition_step = ConditionStep(
        name="CheckAccuracy",
        conditions=[condition],
        if_steps=[register_step],
        # else_steps를 실패로 처리하지 않고 빈 리스트로 두는 이유?
        #   파이프라인을 실패시키면 재시도가 차단된다.
        #   조용히 건너뛰면 파이프라인이 정상 완료되어 실행 이력이 보존된다
        else_steps=[],  # Do nothing if accuracy is below threshold
    )

    # Assemble pipeline
    # Pipeline 레벨에서 파라미터를 선언하는 이유?
    #   SageMaker가 Studio의 "파이프라인 실행" 대화상자에 파라미터를 노출시켜,
    #   비엔지니어도 코드나 파이프라인 정의를 수정하지 않고
    #   임계값을 바꿔가며 실험을 실행할 수 있게 된다
    pipeline = Pipeline(
        name="ml-training-pipeline",
        parameters=[input_data, accuracy_threshold],
        steps=[processing_step, training_step, condition_step],
        sagemaker_session=session,
    )

    return pipeline
```

### 2.4 SageMaker Clarify (편향 및 설명 가능성)

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
    # train.py 안에서 편향(Bias) 분석하지 않고 별도 Clarify 프로세서를 사용하는 이유?
    #   Clarify는 격리된 컴퓨팅 작업을 별도로 실행해 학습과 메모리/CPU를 공유하지 않는다.
    #   덕분에 재학습 없이 기존 모델 아티팩트(Artifact)에 대해 편향 분석을 재실행할 수 있다
    clarify = SageMakerClarifyProcessor(
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        sagemaker_session=session,
    )

    # Data configuration — tells Clarify about your dataset structure
    data_config = DataConfig(
        s3_data_input_path=data_s3_uri,
        # Clarify 출력을 별도 S3 경로에 저장하는 이유?
        #   Clarify는 대용량 JSON 리포트를 생성하는데,
        #   모델 아티팩트와 분리해야 모델 레지스트리가 오염되지 않고
        #   감사(Audit) 시 찾기도 쉽다
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
    # ModelConfig에 별도 인스턴스 타입을 지정하는 이유?
    #   Clarify는 퍼터베이션(Perturbation)된 입력으로 모델 엔드포인트를 호출하는데,
    #   추론에는 더 작은 인스턴스(ml.m5.large)로 충분해
    #   분석 비용을 낮게 유지할 수 있다
    model_config = ModelConfig(
        model_name=model_name,
        instance_count=1,
        instance_type="ml.m5.large",
        content_type="text/csv",
        accept_type="text/csv",
    )

    # Run pre-training bias analysis (data-only, no model needed)
    # 사전 학습 편향(Pre-training Bias) 분석을 실행하는 이유?
    #   클래스 불균형(Class Imbalance)을 학습 전에 발견하면
    #   차별적인 모델을 배포 후 수정하는 것보다 훨씬 비용이 적게 든다
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
        # baseline=None으로 설정하는 이유?
        #   Clarify가 데이터셋에서 평균 기준선(Mean Baseline)을 자동 계산한다.
        #   직접 지정한 영벡터(Zero Vector)보다 더 대표성이 있으며
        #   연구자 편향이 설명에 개입되지 않도록 방지한다
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

### 3.1 Vertex AI 생태계 개요

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

### 3.2 Vertex AI 커스텀 학습(Custom Training)

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
        # Docker Hub 대신 Google 호스팅 컨테이너 URI를 사용하는 이유?
        #   GCP 컨테이너는 동일 리전의 Artifact Registry에 저장되어
        #   풀(Pull) 속도가 빠르고 이그레스(Egress) 비용도 없다.
        #   또한 Google이 정기적으로 보안 패치를 제공한다
        container_uri="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-3:latest",
        requirements=["pandas>=2.0", "scikit-learn>=1.3"],
        # 학습 컨테이너와 서빙(Serving) 컨테이너를 별도로 지정하는 이유?
        #   학습과 서빙은 의존성이 다르다 (예: 서빙 시 pandas나 데이터 로더 불필요).
        #   가벼운 서빙 이미지는 콜드 스타트(Cold Start) 지연을 줄이고
        #   공격 표면(Attack Surface)도 최소화한다
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
        # 환경 변수 대신 커맨드라인 인수(Command-line Args)를 사용하는 이유?
        #   인수는 Vertex AI 작업 메타데이터와 실험(Experiment) 비교 화면에 표시되지만,
        #   환경 변수는 플랫폼 추적 레이어에서 보이지 않는다
        args=[
            "--n-estimators", "100",
            "--max-depth", "10",
            "--test-size", "0.2",
        ],

        # Compute configuration
        # replica_count를 함수에 하드코딩하지 않는 이유?
        #   파라미터로 유지하면 스크립트 수정 없이 숫자만 바꿔
        #   다중 워커 분산 학습(Multi-Worker Distributed Training)으로 전환할 수 있다
        replica_count=1,
        machine_type="n1-standard-8",       # 8 vCPU, 30 GB RAM
        # accelerator_type="NVIDIA_TESLA_T4",  # Uncomment for GPU
        # accelerator_count=1,

        # base_output_dir을 로컬 디스크 대신 GCS로 지정하는 이유?
        #   Vertex AI 학습 VM은 임시적(Ephemeral)이어서 작업 종료 후 소멸된다.
        #   GCS만이 작업 이후에도 유지되며 서빙 엔드포인트에서 접근 가능한 스토리지다
        base_output_dir="gs://my-ml-bucket/models/",
    )

    return model
```

### 3.3 Vertex AI 파이프라인(Pipelines)

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
    # 사전 빌드 이미지 대신 packages_to_install을 사용하는 이유?
    #   KFP는 런타임에 베이스 이미지에 동적으로 pip install을 수행한다.
    #   가벼운 컴포넌트를 위한 별도 Dockerfile 관리 부담을 없애면서도
    #   정확한 버전 고정(Version Pinning)이 가능해 재현성이 보장된다
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
    # 모든 컬럼이 아닌 target 컬럼만 dropna 하는 이유?
    #   피처(Feature)가 누락된 행은 대치(Imputation)가 가능하지만,
    #   레이블(Label)이 누락된 행은 학습에 전혀 사용할 수 없다.
    #   또한 누락 레이블이 손실 함수(Loss Function)에 조용히 전파되는
    #   NaN 버그를 예방한다
    df = df.dropna(subset=["target"])
    # 평균 대신 중앙값(Median)으로 결측값을 채우는 이유?
    #   중앙값은 이상값(Outlier)에 강건하다. 금융·의료 데이터에서 흔한
    #   극단값이 있으면 평균 대치가 왜곡될 수 있다
    df = df.fillna(df.median(numeric_only=True))

    # Split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    # Save outputs — KFP handles artifact storage automatically
    # CSV 대신 Parquet을 사용하는 이유?
    #   Parquet은 데이터 타입(dtype)을 보존해 재로딩 시
    #   int→float 변환 같은 조용한 오류가 생기지 않으며,
    #   크기도 3~10배 작아 단계 간 I/O 비용을 절감한다
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
        # random_state=42로 설정하는 이유?
        #   동일한 파이프라인을 재실행해도 비트 단위로 동일한 결과를 보장한다.
        #   설정하지 않으면 비결정적 트리 구축으로 인해 동일 조건의 두 실행이
        #   정확도에서 1~2% 차이를 보일 수 있다
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Log training metrics — visible in Vertex AI Experiments
    # n_estimators, max_depth도 메트릭(Metric)으로 기록하는 이유?
    #   Vertex AI 실험(Experiments)은 기록된 메트릭으로만 정렬·필터링이 가능하며,
    #   컴포넌트 파라미터는 추적 레이어에서 보이지 않는다.
    #   메트릭으로 기록해야 Experiments UI에서 하이퍼파라미터별 비교가 가능하다
    y_pred = clf.predict(X_train)
    metrics.log_metric("train_accuracy", accuracy_score(y_train, y_pred))
    metrics.log_metric("train_f1", f1_score(y_train, y_pred, average="weighted"))
    metrics.log_metric("n_estimators", n_estimators)
    metrics.log_metric("max_depth", max_depth)

    # Save model
    # model_artifact.path에 ".joblib"을 덧붙이는 이유?
    #   KFP는 model_artifact.path를 디렉토리 경로로 설정한다.
    #   joblib.dump는 파일 경로가 필요하므로 확장자를 추가해 유효한 파일명을 만든다
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

    # test_accuracy와 accuracy_threshold를 함께 기록하는 이유?
    #   Vertex AI 실험(Experiments)에서 임계값과 결과를 나란히 보여줘
    #   리뷰어가 파이프라인 파라미터를 따로 찾지 않아도
    #   "0.87 vs 임계값 0.85"를 한눈에 확인할 수 있다
    metrics.log_metric("test_accuracy", accuracy)
    metrics.log_metric("test_f1", f1)
    metrics.log_metric("accuracy_threshold", accuracy_threshold)
    # deploy_approved를 bool 대신 0/1 정수로 기록하는 이유?
    #   Vertex AI 실험은 모든 메트릭을 float으로 저장한다.
    #   정수 1/0으로 기록해야 UI에서 "승인된 실행만 표시" 같은
    #   필터링이 파싱 없이 바로 작동한다
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
    # 변수 참조 대신 이름("train_data")으로 출력(Output)을 전달하는 이유?
    #   KFP는 파이프라인을 YAML로 컴파일한다. 명명된 출력 키를 사용하면
    #   컴파일 시점에 아티팩트 타입 호환성을 검증할 수 있어
    #   런타임 오류를 조기에 발견할 수 있다
    train_task = train_model(
        train_data=preprocess_task.outputs["train_data"],
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    # Step 3: Evaluate
    # test_data를 train_task가 아닌 preprocess_task에서 가져오는 이유?
    #   모델이 본 적 없는 데이터로 평가해야 한다는 데이터 격리를 명시적으로 표현한다.
    #   DAG에서 이렇게 연결하면 파이프라인 그래프에서도 격리가 시각적으로 드러나
    #   감사(Audit) 시 데이터 유출이 없었음을 쉽게 확인할 수 있다
    eval_task = evaluate_model(
        test_data=preprocess_task.outputs["test_data"],
        model_artifact=train_task.outputs["model_artifact"],
        accuracy_threshold=accuracy_threshold,
    )

    # Step 4: Conditional deployment
    # Python if문 대신 dsl.Condition을 사용하는 이유?
    #   Python if문은 컴파일 시점에 실행되어 항상 같은 분기를 택한다.
    #   dsl.Condition은 evaluate_model 단계가 완료된 후
    #   실제 출력값을 Vertex AI가 런타임에 평가하는 분기를 만든다
    with dsl.Condition(eval_task.output == True):  # noqa: E712
        # Deploy only if evaluation passes
        deploy_model(
            model=train_task.outputs["model_artifact"],
            project="my-gcp-project",
            location="us-central1",
        )
```

### 3.4 Vertex AI 모델 모니터링(Model Monitoring)

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
    # training_dataset을 처리된 버전이 아닌 원본 CSV로 지정하는 이유?
    #   Vertex AI는 이 파일로 기준 통계(Baseline Statistics)를 한 번 계산하고 캐시한다.
    #   모델이 학습에 사용한 것과 동일한 분포를 기준으로 삼아야
    #   스큐(Skew) 감지가 학습과 프로덕션 간의 차이를 정확히 측정한다
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
                    # JS 발산(Jensen-Shannon Divergence) 임계값을 0.3으로 설정하는 이유?
                    #   JS 발산은 0~1 범위이며, 0.3은 실무에서 좋은 출발점이다.
                    #   일상적인 변동으로 인한 불필요한 알람 없이 의미 있는
                    #   분포 변화를 잡아낼 수 있다. 피처별 분산을 스테이징 환경에서
                    #   관찰한 후 개별 조정하는 것이 권장된다
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
        # 모니터링 통계를 엔드포인트가 아닌 GCS에 저장하는 이유?
        #   GCS 보존을 통해 시계열 드리프트(Drift) 추이 분석이 가능하다.
        #   엔드포인트에만 저장된 통계는 엔드포인트 업데이트나 삭제 시 소멸된다
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

### 4.1 Azure ML 생태계 개요

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

### 4.2 SDK v2를 사용한 Azure ML 학습

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

    # MLClient를 구독(Subscription) 전체가 아닌 특정 워크스페이스로 범위를 제한하는 이유?
    #   Azure ML 리소스는 워크스페이스 범위이므로, 워크스페이스 수준 클라이언트를 사용하면
    #   같은 Azure 구독을 공유하는 프로젝트 간에 작업, 환경, 모델이
    #   실수로 섞이는 것을 방지할 수 있다
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
        # pip requirements.txt 대신 conda_file을 사용하는 이유?
        #   Conda는 네이티브 라이브러리(MKL, OpenBLAS)까지 포함한 전체 의존성 그래프를 해결한다.
        #   pip만 사용하면 scikit-learn과 NumPy의 BLAS 백엔드 간
        #   조용한 버전 충돌이 발생할 수 있다
        conda_file="environments/conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
    )

    # Define the training job
    # Why 'command' instead of 'Estimator'?
    #   SDK v2 uses 'command' as the primary abstraction
    #   It's more flexible and consistent with Azure CLI
    training_job = command(
        code="src/",                           # Source code directory
        # 하드코딩 대신 ${{inputs.*}} 템플릿 문법을 사용하는 이유?
        #   Azure ML이 런타임에 이 값을 대입하므로, 동일한 작업 정의를
        #   CLI나 Studio에서 다른 하이퍼파라미터로 재실행할 수 있다.
        #   코드 수정 없이 GitOps 방식의 파라미터 스윕(Sweep)이 가능해진다
        command="python train.py "
                "--n-estimators ${{inputs.n_estimators}} "
                "--max-depth ${{inputs.max_depth}} "
                "--data-path ${{inputs.training_data}}",
        inputs={
            # 데이터 자산을 이름:버전("training-data:1")으로 참조하는 이유?
            #   Azure ML 데이터 자산은 버전 관리된다. 버전 1로 고정하면
            #   나중에 데이터셋이 업데이트되어도 학습 작업이 재현 가능하게 유지된다
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
    # create 대신 create_or_update를 사용하는 이유?
    #   멱등성(Idempotent) 제출이므로 CI 파이프라인에서 일시적인 Azure API 오류 시
    #   중복 작업 생성 위험 없이 안전하게 재시도할 수 있다
    returned_job = ml_client.jobs.create_or_update(training_job)
    print(f"Job submitted: {returned_job.studio_url}")

    return returned_job
```

### 4.3 Azure ML 파이프라인(컴포넌트 기반)

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
    # version="1.0"을 명시하는 이유?
    #   Azure ML은 등록된 모든 컴포넌트 버전을 저장한다.
    #   명시적 버전 없이 "latest"를 참조하면 새 버전에 포함된 호환성 변경으로
    #   파이프라인이 조용히 깨질 수 있다
    version="1.0",
    display_name="Preprocess Training Data",
    description="Clean data, handle missing values, split train/test",
    inputs={
        # raw_data에 uri_file 대신 uri_folder를 사용하는 이유?
        #   데이터셋은 날짜별 파티션 CSV처럼 여러 파일로 나뉘어 오는 경우가 많다.
        #   uri_folder를 사용하면 파일명을 미리 알지 못해도
        #   디렉토리 전체를 읽을 수 있다
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
    # "latest" 대신 azureml:sklearn-env:1(버전 고정)을 사용하는 이유?
    #   버전 고정은 컴포넌트가 항상 동일한 환경에서 실행됨을 보장한다.
    #   "latest"는 다른 작업을 위해 환경이 업데이트될 때 조용히 깨질 수 있다
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
        # preprocess_step.outputs.train_data로 연결하는 이유?
        #   데이터 계보(Data Lineage): Azure ML이 원시 데이터 → 처리 데이터 → 모델의
        #   출처 체인을 추적한다. Studio 계보 그래프가 정확해지며,
        #   금융·의료 규제 감사 시 필수적인 증적 자료가 된다
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
            # test_data를 train_step이 아닌 preprocess_step에서 가져오는 이유?
            #   모델이 본 적 없는 데이터로 평가한다는 원칙을 명시적으로 표현한다.
            #   Azure ML은 이 의존성을 컴파일 시점에 검증해 컴퓨팅 소비 전에
            #   데이터 격리 오류를 발견할 수 있다
            "test_data": preprocess_step.outputs.test_data,
            # threshold를 파이프라인 파라미터로 전달하는 이유?
            #   스테이징과 프로덕션이 다른 품질 기준을 가질 수 있다.
            #   파라미터로 분리하면 컴포넌트 코드 변경 없이 환경별 기준을 조정할 수 있다
            "threshold": accuracy_threshold,
        },
        environment="azureml:sklearn-env:1",
    )

    # 파이프라인 레벨에서 model 출력을 반환하는 이유?
    #   파이프라인 경계에서 출력을 노출하면 다운스트림 파이프라인이나
    #   Azure ML 배포 작업이 내부 단계 구조를 알지 못해도
    #   모델 아티팩트를 직접 사용할 수 있다
    return {
        "model": train_step.outputs.model,
    }
```

### 4.4 Azure 책임 AI 대시보드(Responsible AI Dashboard)

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
        # max_depth=4로 설정하는 이유?
        #   깊이 4의 결정 트리는 최대 16개의 오류 코호트를 나타낼 수 있다.
        #   의미 있는 패턴을 발견하기에 충분하면서, 테스트 셋의 노이즈를
        #   과적합(Overfitting)하지 않을 정도로 얕게 유지된다
        "max_depth": 4,           # Decision tree depth for error patterns
        # num_leaves=31로 설정하는 이유?
        #   LightGBM의 기본값과 일치한다. 경험적으로 주요 오류 클러스터를 포착하면서도
        #   대시보드를 검토하는 비기술 이해관계자가 해석할 수 있을 정도로 유지된다
        "num_leaves": 31,         # Number of error cohorts to identify
    }

    # Fairness component
    fairness_config = {
        "sensitive_features": ["gender", "race"],
        # 정확도(accuracy_score) 외에 false_positive_rate, false_negative_rate를 추가하는 이유?
        #   전체 정확도가 그룹 간에 동일해도, 한 그룹이 불균형하게 높은 거짓 음성(FN)을 겪을 수 있다.
        #   (예: 자격 있는 지원자에 대한 대출 거절)
        #   세분화된 비율(Disaggregated Rates)이 이 불균형을 드러낸다
        "metrics": [
            "accuracy_score",
            "false_positive_rate",
            "false_negative_rate",
            "selection_rate",
        ],
    }

    # Explainability component
    explainability_config = {
        # SHAP TreeExplainer 대신 "mimic" 설명기를 사용하는 이유?
        #   Mimic은 복잡한 모델을 전역적으로 근사하는 간단한 대리(Surrogate) 모델을 학습한다.
        #   대용량 데이터셋에서 SHAP보다 10~100배 빠르면서도
        #   신뢰할 수 있는 피처 중요도 순위를 제공한다
        "method": "mimic",        # Use InterpretML's mimic explainer
        # max_features=10으로 제한하는 이유?
        #   모든 피처를 표시하면 리뷰어가 압도된다.
        #   상위 10개면 주요 설명 요인을 커버하면서
        #   데이터 과학자가 아닌 컴플라이언스 담당자도 읽을 수 있는 대시보드가 유지된다
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

## 5. 플랫폼 비교

### 5.1 기능 매트릭스(Feature Matrix)

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

### 5.2 플랫폼 선택 기준

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

## 6. 멀티 클라우드 및 벤더 중립 전략

### 6.1 추상화 레이어로서의 MLflow

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
    # 환경 변수(MLFLOW_TRACKING_URI) 대신 코드에서 set_tracking_uri를 호출하는 이유?
    #   코드 수준 설정은 명시적이고 버전 관리된다.
    #   환경 변수는 코드 리뷰에서 보이지 않으며
    #   CI 시스템에서 조용히 덮어씌워질 수 있다
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
        # 같은 실행에서 log_param(n_estimators)과 log_metric(accuracy) 모두 기록하는 이유?
        #   MLflow는 파라미터(학습 전 설정 하이퍼파라미터)와
        #   메트릭(학습 후 계산된 결과)을 구분한다.
        #   이 구분이 MLflow UI의 병렬 좌표(Parallel Coordinates) 플롯을 통한
        #   하이퍼파라미터 분석을 가능하게 한다
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        # 모델을 .pkl 파일로 직접 저장하지 않고 log_model을 사용하는 이유?
        #   mlflow.sklearn.log_model은 모델을 Python 환경, 입력 스키마,
        #   서빙 인터페이스가 기록된 MLmodel 스펙으로 감싼다.
        #   어느 클라우드의 MLflow 통합이든 추가 설정 없이 바로 배포할 수 있다
        mlflow.sklearn.log_model(clf, "model")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Model logged to: {mlflow.get_artifact_uri()}")
```

### 6.2 파이프라인 이식성을 위한 Kubeflow

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

### 6.3 추상화 레이어 의사결정 프레임워크

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

## 7. 비용 최적화 전략

### 7.1 학습 비용 최적화

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
    # on_demand와 spot을 같은 딕셔너리 항목에 저장하는 이유?
    #   두 번 검색하지 않고 직접 절감액 비교가 가능하다.
    #   (spot/on_demand) 비율은 인스턴스 패밀리마다 크게 다르며,
    #   GPU 인스턴스는 CPU 인스턴스보다 스팟 할인이 더 깊은 경우가 많다
    pricing = {
        "aws": {
            "ml.m5.xlarge":    {"on_demand": 0.23, "spot": 0.07},
            "ml.g4dn.xlarge":  {"on_demand": 0.74, "spot": 0.22},
            "ml.p3.2xlarge":   {"on_demand": 3.83, "spot": 1.15},
            # ml.p4d.24xlarge를 포함하는 이유?
            #   온디맨드 $32/시간이면 2시간 실행에 $65가 드는데,
            #   이 표가 없으면 대규모 학습 작업의 비용을 쉽게 과소평가한다
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
        # spot=True일 때만 savings_vs_on_demand를 계산하는 이유?
        #   온디맨드 실행 시 절감액은 정의상 0%이다.
        #   0% 절감 항목을 출력에 포함하면 예산 보고서가 지저분해지고
        #   온디맨드 작업에 대해 오해를 줄 수 있다
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
    # 노트북 자동 종료의 절감액을 10%로 추정하는 이유?
    #   업계 조사에 따르면 노트북은 업무 시간의 30~50%가 유휴 상태이다.
    #   하루 8시간 실행 기준, 1시간 유휴 종료 정책으로
    #   노트북 컴퓨팅 비용의 약 10%를 모델 품질 손실 없이 회수할 수 있다
    recommendations.append({
        "strategy": "Auto-shutdown idle notebooks",
        "effort": "LOW",
        "estimated_savings": round(monthly_spend * 0.10, 2),
        "how": "Set lifecycle policies: 1hr idle → stop instance",
    })

    # Spot instances — high impact, some risk
    # 스팟 인스턴스 추천을 daily 학습 빈도에만 적용하는 이유?
    #   주 1회 학습하는 팀은 스팟 중단 시 수동 재시도로 충분하다.
    #   매일 학습하는 팀은 스팟을 실용적으로 사용하기 위해
    #   자동화된 체크포인팅(Checkpointing)이 필요하므로,
    #   투자 대비 효과가 클 때만 권장한다
    if training_frequency == "daily":
        recommendations.append({
            "strategy": "Use spot instances for training",
            "effort": "LOW",
            "estimated_savings": round(monthly_spend * 0.40, 2),
            "how": "Enable spot/preemptible with checkpointing",
            "risk": "Jobs may be interrupted; needs checkpoint logic",
        })

    # Reserved capacity — for predictable workloads
    # 예약 인스턴스 권장 임계값을 월 $5,000으로 설정하는 이유?
    #   $5K 미만에서 1년 약정 절감액(~$1.5K)은 유연성 포기 비용을 정당화하지 못한다.
    #   $5K 초과 시 절감액이 복리로 커지고 특정 인스턴스 타입 고정 위험을
    #   상쇄하기에 충분하다
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
        # 모든 추천 절감액을 합산하는 이유?
        #   전략들이 겹치므로 실제 절감액은 이보다 적지만,
        #   합계는 유용한 "기회 규모(Opportunity Size)" 상한을 제시해
        #   작업을 동기부여하는 데 쓰인다. 실제 절감은 한 번에 하나씩 우선순위를 정해야 한다
        "potential_monthly_savings": sum(r["estimated_savings"] for r in recommendations),
        # estimated_savings 내림차순으로 정렬하는 이유?
        #   팀이 ROI가 가장 높은 최적화를 먼저 다뤄야 하기 때문이다.
        #   정렬된 출력이 우선순위 결정을 자명하게 만든다
        "recommendations": sorted(recommendations,
                                   key=lambda r: r["estimated_savings"],
                                   reverse=True),
    }
```

---

## 8. 실용적인 마이그레이션 패턴

### 8.1 로컬에서 클라우드로 마이그레이션

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

## 요약

| 항목 | AWS SageMaker | Google Vertex AI | Azure ML |
|------|---------------|------------------|----------|
| **최적 대상** | AWS 네이티브 팀 | 대형 모델, 데이터 팀 | 엔터프라이즈, Microsoft 스택 |
| **강점** | 넓은 범위, 성숙도 | TPU, BigQuery ML, KFP 이식성 | RAI 대시보드, VS Code, Arc |
| **종속성 수준** | 높음 (SM SDK) | 중간 (KFP 이식 가능) | 중간 (AzML SDK) |
| **고유 기능** | Inferentia/Trainium | BigQuery ML | 책임 AI 대시보드 |
| **파이프라인 SDK** | 독점(Proprietary) | KFP v2 (오픈소스) | 독점(Proprietary) |
| **비용 전략** | 스팟 + Savings Plans | 스팟 + CUD | 저우선순위 + 예약 |

핵심 교훈: **데이터가 이미 어디에 있는지, 팀이 이미 무엇을 알고 있는지를 기준으로 선택하세요.** 플랫폼 간 차이는 전환 비용보다 훨씬 작습니다. 하나의 플랫폼으로 시작하고, 이식성 보험으로 MLflow를 사용하며, 비즈니스 요구사항이 실제로 요구할 때만 멀티 클라우드로 전환하세요.

---

## 연습 문제

### 연습 1: SageMaker 학습 작업
SageMaker Python SDK를 사용하여:
- scikit-learn 분류기에 대한 학습 작업 생성
- 최대 대기 시간 2시간의 스팟 학습 구성
- CloudWatch에 커스텀 메트릭 기록
- S3에서 학습된 모델 아티팩트 가져오기

### 연습 2: Vertex AI 파이프라인
KFP v2 SDK를 사용하여:
- 3단계 파이프라인 생성 (전처리, 학습, 평가)
- 정확도 임계값 기반 조건부 배포 추가
- 파이프라인을 YAML로 컴파일
- 다른 파이프라인에서 재사용 가능한 컴포넌트 식별

### 연습 3: 플랫폼 비교
현재(또는 가상의) ML 프로젝트에 대해:
- 섹션 5의 기능 매트릭스를 이용해 세 플랫폼 모두 평가
- 가격 추정기를 사용하여 각 플랫폼의 월간 비용 추산
- 팀 규모, 기존 클라우드 사용량, ML 워크로드 특성에 기반한 플랫폼 추천과 근거 제시
- 플랫폼을 전환할 경우 변경해야 할 사항 문서화

### 연습 4: 벤더 중립 아키텍처
다음을 사용하는 벤더 중립 ML 아키텍처 설계:
- 실험 추적 및 모델 레지스트리를 위한 MLflow
- 오케스트레이션을 위한 Kubeflow Pipelines
- 피처 스토어를 위한 Feast
- 모니터링을 위한 Evidently
- 더 나은 통합을 위해 클라우드 네이티브 컴포넌트로 대체할 수 있는 곳 문서화

### 연습 5: 비용 최적화 감사
비용 최적화 감사 수행:
- (가상의) ML 워크로드 프로파일링: 인스턴스 유형, 사용 시간, 빈도
- 섹션 7의 비용 추정 함수 적용
- 스팟 인스턴스, 인스턴스 크기 최적화, 혼합 정밀도 학습에서의 잠재적 절감액 계산
- 우선순위가 지정된 작업을 포함한 90일 비용 최적화 계획 수립

---

[← 이전: 16. 모델 테스트와 검증](16_Model_Testing_and_Validation.md) | [다음: 개요 →](00_Overview.md)
