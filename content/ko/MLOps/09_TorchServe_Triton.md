# TorchServe & Triton Inference Server

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. TorchServe 아키텍처의 프론트엔드(frontend), 백엔드(backend), 모델 스토어(model store), 워커(worker) 컴포넌트를 설명하고 요청이 어떻게 처리되는지 서술할 수 있다
2. PyTorch 모델을 `.mar` 아카이브 파일로 패키징하고 커스텀 요청 핸들러(request handler)를 사용하여 TorchServe로 배포할 수 있다
3. Triton Inference Server 아키텍처와 여러 백엔드(PyTorch, TensorFlow, ONNX, TensorRT) 지원을 설명할 수 있다
4. 앙상블 파이프라인(ensemble pipeline)과 동적 배칭(dynamic batching)으로 Triton 모델 리포지토리를 구성하여 GPU 처리량을 최대화할 수 있다
5. TorchServe와 Triton Inference Server를 성능, 유연성, 운영 복잡성 측면에서 비교하여 배포 시나리오에 적합한 도구를 선택할 수 있다

---

## 1. TorchServe 개요

TorchServe는 PyTorch 모델을 프로덕션 환경에서 서빙하기 위한 공식 도구입니다.

### 1.1 TorchServe 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TorchServe 아키텍처                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐                                                   │
│   │   Client    │                                                   │
│   └──────┬──────┘                                                   │
│          │                                                          │
│          ▼                                                          │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │                    Frontend                              │      │
│   │  ┌──────────────┐  ┌──────────────┐                     │      │
│   │  │ REST API     │  │ gRPC API     │                     │      │
│   │  │ :8080        │  │ :7070        │                     │      │
│   │  └──────────────┘  └──────────────┘                     │      │
│   └─────────────────────────────────────────────────────────┘      │
│          │                                                          │
│          ▼                                                          │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │                    Backend                               │      │
│   │  ┌──────────────────────────────────────────┐           │      │
│   │  │              Model Store                  │           │      │
│   │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐    │           │      │
│   │  │  │ Model A │ │ Model B │ │ Model C │    │           │      │
│   │  │  └─────────┘ └─────────┘ └─────────┘    │           │      │
│   │  └──────────────────────────────────────────┘           │      │
│   │                                                          │      │
│   │  ┌──────────────┐  ┌──────────────┐                     │      │
│   │  │ Worker 1     │  │ Worker 2     │  ...                │      │
│   │  └──────────────┘  └──────────────┘                     │      │
│   └─────────────────────────────────────────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 설치

```bash
# TorchServe 설치
pip install torchserve torch-model-archiver torch-workflow-archiver

# 버전 확인
torchserve --version
```

---

## 2. 핸들러 작성

### 2.1 기본 핸들러

```python
"""
custom_handler.py - TorchServe 커스텀 핸들러
"""

import torch
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler
import json
import logging

logger = logging.getLogger(__name__)

class CustomHandler(BaseHandler):
    """커스텀 TorchServe 핸들러"""

    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, context):
        """모델 초기화"""
        # 시작 시 한 번만 호출됨(요청마다가 아님) — 여기서 모델 로드와 GPU 메모리 할당을 수행하여
        # 매 추론 호출마다 100-500ms 오버헤드를 방지
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # GPU 자동 감지 — CPU로 폴백하여 동일한 핸들러가 양쪽 환경에서 동작
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 모델 로드
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = f"{model_dir}/{serialized_file}"

        # TorchScript 모델 — 직렬화된 연산 그래프로 런타임에 Python이 불필요하여
        # 더 빠른 추론과 C++ 배포가 가능
        self.model = torch.jit.load(model_pt_path, map_location=self.device)
        self.model.eval()

        # 추가 설정 로드 (있는 경우)
        self.class_names = self._load_class_names(model_dir)

        self.initialized = True
        logger.info("Model initialized successfully")

    def _load_class_names(self, model_dir):
        """클래스 이름 로드"""
        try:
            with open(f"{model_dir}/index_to_name.json") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def preprocess(self, data):
        """입력 전처리"""
        inputs = []

        for row in data:
            # JSON 입력 처리
            if isinstance(row, dict):
                features = row.get("data") or row.get("body")
            else:
                features = row.get("body")

            if isinstance(features, (bytes, bytearray)):
                features = json.loads(features.decode("utf-8"))

            tensor = torch.tensor(features, dtype=torch.float32)
            inputs.append(tensor)

        # 배치로 묶기
        return torch.stack(inputs).to(self.device)

    def inference(self, data):
        """추론 수행"""
        # torch.no_grad()로 그래디언트 추적 비활성화 — 메모리 사용량을 ~50% 줄이고
        # 서빙 시에는 역전파가 불필요하므로 추론 속도도 향상
        with torch.no_grad():
            outputs = self.model(data)
            probabilities = F.softmax(outputs, dim=1)
        return probabilities

    def postprocess(self, data):
        """출력 후처리"""
        results = []

        for prob in data:
            prob_list = prob.cpu().numpy().tolist()
            prediction = int(torch.argmax(prob).item())

            result = {
                "prediction": prediction,
                "probabilities": prob_list
            }

            # 클래스 이름 추가
            if self.class_names:
                result["class_name"] = self.class_names.get(str(prediction))

            results.append(result)

        return results
```

### 2.2 이미지 분류 핸들러

```python
"""
image_classifier_handler.py - 이미지 분류 핸들러
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from ts.torch_handler.vision_handler import VisionHandler
from PIL import Image
import io
import base64

class ImageClassifierHandler(VisionHandler):
    """이미지 분류 핸들러"""

    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess(self, data):
        """이미지 전처리"""
        images = []

        for row in data:
            image_data = row.get("data") or row.get("body")

            # Base64 디코딩
            if isinstance(image_data, str):
                image_data = base64.b64decode(image_data)

            # 이미지 로드
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image = self.transform(image)
            images.append(image)

        return torch.stack(images).to(self.device)

    def postprocess(self, data):
        """결과 후처리"""
        probabilities = F.softmax(data, dim=1)
        top_k = torch.topk(probabilities, 5)

        results = []
        for probs, indices in zip(top_k.values, top_k.indices):
            result = {
                "predictions": [
                    {
                        "class_id": int(idx),
                        "class_name": self.mapping.get(str(int(idx)), "unknown"),
                        "probability": float(prob)
                    }
                    for prob, idx in zip(probs, indices)
                ]
            }
            results.append(result)

        return results
```

---

## 3. 모델 아카이브 및 배포

### 3.1 모델 아카이브 생성

```bash
# 모델을 TorchScript로 저장
python -c "
import torch
model = YourModel()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
scripted = torch.jit.script(model)
scripted.save('model.pt')
"

# MAR 파일 생성
torch-model-archiver \
    --model-name my_model \
    --version 1.0 \
    --serialized-file model.pt \
    --handler custom_handler.py \
    --extra-files "index_to_name.json,config.json" \
    --export-path model_store

# 생성된 파일: model_store/my_model.mar
```

### 3.2 TorchServe 시작

```bash
# 기본 시작
torchserve --start \
    --model-store model_store \
    --models my_model=my_model.mar

# 설정 파일과 함께
torchserve --start \
    --model-store model_store \
    --models my_model=my_model.mar \
    --ts-config config.properties

# Docker로 실행
docker run -d \
    --name torchserve \
    -p 8080:8080 \
    -p 8081:8081 \
    -p 8082:8082 \
    -v $(pwd)/model_store:/home/model-server/model-store \
    pytorch/torchserve:latest \
    torchserve --start --model-store /home/model-server/model-store
```

### 3.3 설정 파일

```properties
# config.properties
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082

# 워커 설정
default_workers_per_model=4
job_queue_size=1000

# 배치 설정
max_batch_delay=100
batch_size=32

# GPU 설정
number_of_gpu=1

# 모델 설정
model_store=/home/model-server/model-store
load_models=my_model.mar
```

### 3.4 API 호출

```python
"""
TorchServe API 호출
"""

import requests
import json

# 예측 요청
def predict(data):
    response = requests.post(
        "http://localhost:8080/predictions/my_model",
        json=data
    )
    return response.json()

# 단일 예측
result = predict({"data": [1.0, 2.0, 3.0, 4.0]})
print(result)

# 배치 예측
batch_data = [
    {"data": [1.0, 2.0, 3.0, 4.0]},
    {"data": [5.0, 6.0, 7.0, 8.0]}
]
results = [predict(d) for d in batch_data]

# 관리 API
# 모델 목록
models = requests.get("http://localhost:8081/models").json()

# 모델 상세 정보
model_info = requests.get("http://localhost:8081/models/my_model").json()

# 워커 스케일링
requests.put(
    "http://localhost:8081/models/my_model",
    params={"min_worker": 2, "max_worker": 4}
)

# 모델 등록
requests.post(
    "http://localhost:8081/models",
    params={
        "url": "my_model_v2.mar",
        "initial_workers": 2
    }
)
```

---

## 4. Triton Inference Server

### 4.1 Triton 개요

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Triton Inference Server                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   지원 프레임워크:                                                   │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│   │ PyTorch  │ │TensorFlow│ │   ONNX   │ │ TensorRT │              │
│   └──────────┘ └──────────┘ └──────────┘ └──────────┘              │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐                           │
│   │  Python  │ │   DALI   │ │   vLLM   │                           │
│   └──────────┘ └──────────┘ └──────────┘                           │
│                                                                     │
│   주요 기능:                                                        │
│   - 동적 배칭 (Dynamic Batching)                                    │
│   - 모델 앙상블                                                     │
│   - 멀티 모델 서빙                                                  │
│   - GPU 스케줄링                                                    │
│   - 모델 버전 관리                                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 모델 저장소 구조

```
model_repository/
├── model_a/
│   ├── config.pbtxt
│   ├── 1/
│   │   └── model.onnx
│   └── 2/
│       └── model.onnx
├── model_b/
│   ├── config.pbtxt
│   └── 1/
│       └── model.pt
└── ensemble_model/
    └── config.pbtxt
```

### 4.3 모델 설정

```protobuf
# config.pbtxt - ONNX 모델
name: "churn_predictor"
platform: "onnxruntime_onnx"
max_batch_size: 32

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [4]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [2]
  }
]

# 동일 GPU에 다중 인스턴스 — 컴퓨팅과 데이터 전송을 겹쳐서(overlap)
# 한 인스턴스가 I/O를 대기하는 동안 GPU를 계속 활용
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]

# 동적 배칭(dynamic batching)으로 수신 요청을 배치로 묶음 — GPU 커널 실행 오버헤드를
# 여러 입력에 분산하여 최소한의 지연 비용으로 처리량을 3-5배 향상
dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  # 100us 최대 대기 — 배치 크기와 지연 시간 사이의 균형; 처리량 우선 워크로드에서는 증가
  max_queue_delay_microseconds: 100
}

# 버전 정책
version_policy: {
  latest: {
    num_versions: 2
  }
}
```

```protobuf
# config.pbtxt - PyTorch 모델
name: "image_classifier"
platform: "pytorch_libtorch"
max_batch_size: 64

input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]

output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [1000]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

### 4.4 Triton 실행

```bash
# Docker로 실행
docker run --gpus all -d \
    --name triton \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -v $(pwd)/model_repository:/models \
    nvcr.io/nvidia/tritonserver:23.10-py3 \
    tritonserver --model-repository=/models

# 헬스 체크
curl -v localhost:8000/v2/health/ready

# 모델 메타데이터
curl localhost:8000/v2/models/churn_predictor
```

### 4.5 Python 클라이언트

```python
"""
Triton Python 클라이언트
"""

import numpy as np
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient

# HTTP 클라이언트
def triton_http_inference(model_name: str, input_data: np.ndarray):
    """HTTP를 통한 추론"""
    client = httpclient.InferenceServerClient(url="localhost:8000")

    # 입력 설정
    inputs = [
        httpclient.InferInput("input", input_data.shape, "FP32")
    ]
    inputs[0].set_data_from_numpy(input_data)

    # 출력 설정
    outputs = [
        httpclient.InferRequestedOutput("output")
    ]

    # 추론
    response = client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs
    )

    return response.as_numpy("output")

# gRPC 클라이언트 (더 빠름)
def triton_grpc_inference(model_name: str, input_data: np.ndarray):
    """gRPC를 통한 추론"""
    client = grpcclient.InferenceServerClient(url="localhost:8001")

    inputs = [
        grpcclient.InferInput("input", input_data.shape, "FP32")
    ]
    inputs[0].set_data_from_numpy(input_data)

    outputs = [
        grpcclient.InferRequestedOutput("output")
    ]

    response = client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs
    )

    return response.as_numpy("output")

# 사용 예시
data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
result = triton_http_inference("churn_predictor", data)
print(f"Predictions: {result}")
```

---

## 5. 모델 최적화

### 5.1 ONNX 변환

```python
"""
PyTorch 모델을 ONNX로 변환
"""

import torch
import torch.onnx

# 모델 로드
model = YourModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# 더미 입력
dummy_input = torch.randn(1, 4)

# ONNX로 내보내기
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

# ONNX 모델 검증
import onnx
import onnxruntime as ort

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

# 추론 테스트
session = ort.InferenceSession("model.onnx")
result = session.run(
    None,
    {"input": dummy_input.numpy()}
)
print(f"ONNX output: {result}")
```

### 5.2 TensorRT 최적화

```python
"""
TensorRT 최적화
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def build_engine(onnx_path: str, engine_path: str):
    """ONNX에서 TensorRT 엔진 빌드"""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # ONNX 파싱
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # 빌더 설정
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # FP16 최적화 (지원되는 경우)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # 동적 배치 설정
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 4), (16, 4), (64, 4))  # min, opt, max
    config.add_optimization_profile(profile)

    # 엔진 빌드
    engine = builder.build_serialized_network(network, config)

    # 저장
    with open(engine_path, "wb") as f:
        f.write(engine)

    return engine

# 엔진 빌드
build_engine("model.onnx", "model.engine")
```

### 5.3 양자화

```python
"""
모델 양자화
"""

import torch
from torch.quantization import quantize_dynamic, quantize_static

# 동적 양자화 (가장 간단)
model_int8 = quantize_dynamic(
    model,
    {torch.nn.Linear},  # 양자화할 레이어 타입
    dtype=torch.qint8
)

# 정적 양자화 (더 좋은 성능)
model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
model_prepared = torch.quantization.prepare(model)

# 캘리브레이션 (대표 데이터로)
with torch.no_grad():
    for data in calibration_data:
        model_prepared(data)

model_quantized = torch.quantization.convert(model_prepared)

# 크기 비교
import os
torch.save(model.state_dict(), "model_fp32.pth")
torch.save(model_quantized.state_dict(), "model_int8.pth")

print(f"FP32 size: {os.path.getsize('model_fp32.pth') / 1e6:.2f} MB")
print(f"INT8 size: {os.path.getsize('model_int8.pth') / 1e6:.2f} MB")
```

---

## 6. 멀티모델 서빙

### 6.1 Triton 앙상블

```protobuf
# ensemble_model/config.pbtxt
name: "ensemble_pipeline"
platform: "ensemble"
max_batch_size: 32

input [
  {
    name: "raw_input"
    data_type: TYPE_FP32
    dims: [4]
  }
]

output [
  {
    name: "final_output"
    data_type: TYPE_FP32
    dims: [2]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "preprocessor"
      model_version: -1
      input_map {
        key: "raw_input"
        value: "raw_input"
      }
      output_map {
        key: "processed_output"
        value: "processed"
      }
    },
    {
      model_name: "classifier"
      model_version: -1
      input_map {
        key: "processed"
        value: "input"
      }
      output_map {
        key: "output"
        value: "final_output"
      }
    }
  ]
}
```

### 6.2 A/B 테스트

```python
"""
TorchServe A/B 테스트 설정
"""

import random
import requests

class ABTestRouter:
    """A/B 테스트 라우터"""

    def __init__(self, model_a: str, model_b: str, traffic_split: float = 0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self.base_url = "http://localhost:8080/predictions"

    def predict(self, data: dict) -> dict:
        """A/B 분배된 예측"""
        # 트래픽 분배
        if random.random() < self.traffic_split:
            model = self.model_a
            variant = "A"
        else:
            model = self.model_b
            variant = "B"

        # 예측
        response = requests.post(
            f"{self.base_url}/{model}",
            json=data
        )

        result = response.json()
        result["variant"] = variant
        result["model"] = model

        return result

# 사용
router = ABTestRouter("model_v1", "model_v2", traffic_split=0.8)
result = router.predict({"data": [1.0, 2.0, 3.0, 4.0]})
```

---

## 연습 문제

### 문제 1: TorchServe 배포
PyTorch 이미지 분류 모델을 TorchServe로 배포하세요.

### 문제 2: ONNX 변환
PyTorch 모델을 ONNX로 변환하고 Triton에서 서빙하세요.

### 문제 3: 성능 최적화
TensorRT로 모델을 최적화하고 추론 속도를 비교하세요.

---

## 요약

| 도구 | 장점 | 적합한 상황 |
|------|------|------------|
| TorchServe | PyTorch 네이티브, 간단 | PyTorch 모델 |
| Triton | 멀티 프레임워크, 고성능 | 복잡한 요구사항 |
| ONNX Runtime | 범용, 크로스 플랫폼 | 경량 배포 |
| TensorRT | GPU 최적화 | 최고 성능 필요 |

---

## 참고 자료

- [TorchServe Documentation](https://pytorch.org/serve/)
- [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [TensorRT](https://developer.nvidia.com/tensorrt)
