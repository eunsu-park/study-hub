[이전: 실전 텍스트 분류 프로젝트](./40_Practical_Text_Classification.md) | [다음: 강화학습 소개](./42_Reinforcement_Learning_Intro.md)

---

# 41. 모델 저장 및 배포

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. state_dict와 체크포인트를 사용한 PyTorch 모델 저장 및 로드
2. TorchScript(tracing, scripting)를 사용한 모델 내보내기
3. 크로스 프레임워크 배포를 위한 ONNX 변환
4. `torch.export()`를 사용한 그래프 캡처 (PyTorch 2.x)
5. 추론 최적화 기법 적용 (양자화, 컴파일)
6. REST API, Docker, 모바일, 클라우드 플랫폼을 통한 모델 배포

---

## 이론과 원리

모델 배포는 근본적으로 *학습 시 관심사를 추론 시 관심사에서 분리하는 것*입니다. 학습 시에는 유연성을 원함(Python, eager 모드, 전체 autograd). 추론 시에는 속도와 이식성을 원함(컴파일된 그래프, 고정 형상, Python 없음). 이 섹션은 각 export와 최적화 단계 아래의 수학/CS를 설명합니다: 그래프 캡처, 양자화, 그리고 속도/정확도 트레이드오프.

이 섹션에서 다루는 내용:

- **A.** State dict vs 전체 체크포인트
- **B.** 그래프 캡처: TorchScript vs ONNX vs torch.export
- **C.** 양자화: int8, 혼합 정밀도, 무엇이 희생되는가
- **D.** 추론 최적화: 컴파일, 배치, 커널 융합

### A. State Dict vs 전체 체크포인트

모델을 저장하는 두 방법:

```python
# State dict (선호): 파라미터만
torch.save(model.state_dict(), "model.pt")
# 로딩: 모델 아키텍처 인스턴스화, 그 다음 파라미터 로딩
model = MyModel(); model.load_state_dict(torch.load("model.pt"))

# 전체 pickle (권장 안 함): 실제 Python 클래스 저장
torch.save(model, "model.pt")
# 로딩: 원래 클래스 정의가 사용 가능해야 함
model = torch.load("model.pt")
```

State dict이 이김, 그것이 *미래 보호*되기 때문: 파라미터 이름과 형상이 일치하는 한 Python 클래스 구현을 변경하고, 헬퍼 메서드를 추가하는 등을 할 수 있음. Pickle된 전체 모델은 코드를 재구성하는 순간 깨짐.

파라미터 외에, "체크포인트"는 보통 옵티마이저 상태(`optimizer.state_dict()`), 에폭 번호, 스케줄러 상태, 최고 검증 메트릭을 포함 — 학습을 멈춘 곳에서 *재개*하는 데 필요한 모든 것. 이것이 `torch.save({"model": ..., "opt": ..., "epoch": ...}, "ckpt.pt")`이 일반적으로 보유하는 것.

### B. 그래프 캡처: TorchScript, ONNX, torch.export

PyTorch는 기본적으로 *eager 모드*에서 실행: 모든 연산이 즉시 디스패치되고, Python이 루프 안에 있음. 배포의 경우 보통 *캡처된 그래프*를 원함: 최적화, 직렬화, Python 없이 실행될 수 있는 계산의 정적 표현.

세 캡처 접근:

- **TorchScript 트레이싱** (`torch.jit.trace`): 예제 입력에서 모델 실행, 연산 기록. 데이터 의존 제어 흐름(`if x.sum() > 0:`) 캡처할 수 없음.
- **TorchScript 스크립팅** (`torch.jit.script`): Python 소스 코드의 정적 분석, 제어 흐름 지원하지만 Python의 부분집합만.
- **ONNX export** (`torch.onnx.export`): 트레이스하고 크로스 프레임워크 포맷으로 export. ONNX Runtime, TensorRT, 모바일 등에서 실행 가능.
- **torch.export** (PyTorch 2.x): 새 공식 캡처, FX 그래프 기반, 동적 형상의 더 신뢰할 수 있는 처리.

캡처된 그래프가 당신이 배송하는 것. 그래프를 *만든* Python 스크립트는 추론 시 더 이상 필요 없음.

### C. 양자화

양자화는 파라미터 정밀도를 fp32(4바이트)에서 int8(1바이트) 또는 심지어 int4로 감소. 세 종류:

- **사후 학습 정적 양자화 (PTQ)**: fp32로 학습, 그 다음 가중치와 활성화를 int8로 변환. 활성화 범위를 찾기 위해 작은 데이터셋으로 보정. 보통 0.5-2% 정확도 손실.
- **양자화 인식 학습 (QAT)**: 학습 중 int8 효과 시뮬레이션(fake-quantize, 그 다음 fp32 업데이트). 일반적으로 PTQ 대비 정확도 손실의 대부분을 회복.
- **동적 양자화**: 가중치를 int8로 저장, 활성화를 즉석 양자화. 더 적은 메모리 절약, 더 적은 속도 향상, 적용하기 더 쉬움.

수학: int8은 256 값. fp32를 int8에 매핑하려면 `quantize(x) = round(x / scale + zero_point)`. 양자화 오차를 최소화하기 위해 텐서별 또는 채널별 스케일이 선택됨. 현대 하드웨어(NVIDIA Tensor Core, 모바일 NPU)는 fp32보다 int8에서 훨씬 빠름 — 일반적으로 2-4배 추론 속도 향상.

LLM의 경우 **4비트 양자화**(GPTQ, AWQ)가 표준이 됨: fp32 대비 8배 메모리 절약과 매우 작은 품질 손실, 소비자 GPU에서 70B 파라미터 모델 가능.

### D. 추론 최적화

양자화 외에, 여러 기법이 추론을 가속:

- **커널 융합**: 여러 op(예: LayerNorm + Linear + GELU)를 하나의 CUDA 커널로 결합, 메모리 트래픽 감소. `torch.compile()`이 자동으로 함.
- **배치**: 여러 요청을 함께 처리. 처리량이 GPU 포화까지 배치 크기와 거의 선형으로 스케일.
- **KV 캐싱** (자기 회귀 LM의 경우): 이전 토큰의 키/값을 캐시하여 재계산되지 않게 함. 토큰당 비용을 O(T^2)에서 O(T)로 감소.
- **추론 디코딩**: 작은 "초안" 모델이 K 토큰 생성; 큰 모델이 그것들을 병렬로 검증. 수락되면 큰 모델 한 번 패스로 K 토큰을 얻음.

프로덕션의 경우, 중요도 순서는 보통: 배치 먼저(무료), 그 다음 양자화(작은 정확도 비용), 그 다음 컴파일(무료), 그 다음 LM에 KV 캐시(필수), 그 다음 추론 디코딩 같은 고급 트릭.

### 이론에서 아래 코드로

| 이론 개념 | 본 레슨의 코드 구성 |
|-----------|---------------------|
| State dict 저장/로딩 | `torch.save(model.state_dict(), ...); model.load_state_dict(...)` |
| TorchScript 트레이스 | `torch.jit.trace(model, example_input)` |
| ONNX export | `torch.onnx.export(model, example, "model.onnx")` |
| 동적 양자화 | `torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)` |
| 컴파일 | `model = torch.compile(model)` |

---


## 1. PyTorch 모델 저장

### state_dict 저장 (권장)

```python
# 저장
torch.save(model.state_dict(), 'model_weights.pth')

# 로드
model = MyModel()  # 같은 구조 필요
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

### 전체 모델 저장

```python
# 저장
torch.save(model, 'model_full.pth')

# 로드
model = torch.load('model_full.pth')
model.eval()
```

### 체크포인트 저장

```python
# 저장
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'best_acc': best_acc
}
torch.save(checkpoint, 'checkpoint.pth')

# 로드
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

---

## 2. TorchScript

### 개념

```
Python 의존성 없이 모델 실행
- C++에서 로드 가능
- 모바일 배포
- 서버 최적화
```

### Tracing

```python
# 예시 입력으로 추적
model.eval()
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# 저장
traced_model.save('model_traced.pt')

# 로드
loaded_model = torch.jit.load('model_traced.pt')
output = loaded_model(example_input)
```

### Scripting

```python
# 제어 흐름 있는 모델
class MyModel(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x * 2
        return x

scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')
```

### 비교

| 방법 | 장점 | 단점 |
|------|------|------|
| Trace | 간단, 대부분 동작 | 동적 제어 흐름 불가 |
| Script | 동적 제어 흐름 지원 | 일부 Python 기능 제한 |

---

## 3. ONNX 변환

### 변환

```python
import torch.onnx

model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    opset_version=11
)
```

### ONNX Runtime 추론

```python
import onnxruntime as ort
import numpy as np

# 세션 생성
session = ort.InferenceSession("model.onnx")

# 추론
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
result = session.run([output_name], {input_name: input_data})
```

### 검증

```python
import onnx

# 모델 로드 및 검증
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX 모델 검증 통과")
```

---

## 4. torch.export() (PyTorch 2.x)

### 개념

`torch.export()`는 TorchScript의 PyTorch 2.x 후속 기능입니다. TorchDynamo를 사용하여 전체 연산 그래프를 캡처하고 깔끔하고 이식 가능한 `ExportedProgram`을 생성합니다.

```
TorchScript (레거시) → torch.export() (PyTorch 2.x 권장)
- 안전한 그래프 캡처 (무음 오류 없음)
- torch.compile() 생태계와 호환
- 동적 형상(dynamic shapes) 지원 개선
```

### 기본 사용법

```python
import torch

model = MyModel().eval()
example_input = (torch.randn(1, 3, 224, 224),)

# 모델 내보내기
exported = torch.export.export(model, example_input)

# 내보낸 모델 실행
output = exported.module()(torch.randn(1, 3, 224, 224))
```

### 동적 형상(Dynamic Shapes)

```python
from torch.export import Dim

# 동적 차원 정의
batch = Dim("batch", min=1, max=32)
dynamic_shapes = {"x": {0: batch}}

exported = torch.export.export(
    model,
    (torch.randn(4, 3, 224, 224),),
    dynamic_shapes=dynamic_shapes,
)

# [1, 32] 범위의 모든 배치 크기에서 동작
output = exported.module()(torch.randn(8, 3, 224, 224))
```

### 저장과 로드

```python
# 저장
torch.export.save(exported, "model_exported.pt2")

# 로드
loaded = torch.export.load("model_exported.pt2")
output = loaded.module()(input_data)
```

### torch.export() vs TorchScript

| 기능 | TorchScript | torch.export() |
|------|------------|----------------|
| 그래프 캡처 | Tracing 또는 Scripting | TorchDynamo (자동) |
| Python 지원 | 제한된 부분 집합 | 전체 Python 시맨틱스 |
| 동적 형상 | 수동 어노테이션 | 퍼스트 클래스 지원 |
| 정확성 | 무음 실패 가능 | 안전한 캡처 또는 명확한 에러 |
| 생태계 | 레거시 | torch.compile과 통합 |

---

## 5. 추론 최적화

### eval 모드

```python
model.eval()  # Dropout, BatchNorm 비활성화
```

### no_grad

```python
with torch.no_grad():
    output = model(input)
```

### 추론 모드 (PyTorch 2.0+)

```python
with torch.inference_mode():
    output = model(input)
```

### 양자화 (Quantization)

```python
# 동적 양자화 (간단)
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# 정적 양자화 (더 최적화)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model)
# 캘리브레이션 데이터로 실행
model_quantized = torch.quantization.convert(model_prepared)
```

---

## 6. 배포 옵션

### Flask API

```python
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)
model = torch.load('model.pth')
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    tensor = torch.tensor(data).float()

    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).tolist()

    return jsonify({'prediction': pred})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### FastAPI (권장)

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()
model = torch.jit.load('model_traced.pt')
model.eval()

class InputData(BaseModel):
    data: list

@app.post("/predict")
async def predict(input_data: InputData):
    tensor = torch.tensor(input_data.data).float()

    with torch.inference_mode():
        output = model(tensor)
        pred = output.argmax(dim=1).tolist()

    return {"prediction": pred}
```

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model_traced.pt .
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 7. 모바일 배포

### PyTorch Mobile

```python
# 모바일용 최적화
traced_model = torch.jit.trace(model, example_input)
optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)
optimized_model._save_for_lite_interpreter("model_mobile.ptl")
```

### Android/iOS

```kotlin
// Android (Kotlin)
val module = LiteModuleLoader.load(assetFilePath(this, "model_mobile.ptl"))
val inputTensor = Tensor.fromBlob(inputArray, longArrayOf(1, 3, 224, 224))
val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
```

---

## 8. 클라우드 배포

### AWS SageMaker

```python
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data='s3://bucket/model.tar.gz',
    role=role,
    framework_version='2.0',
    py_version='py310',
    entry_point='inference.py'
)

predictor = model.deploy(
    instance_type='ml.m5.large',
    initial_instance_count=1
)
```

### Hugging Face Hub

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="model.pt",
    path_in_repo="pytorch_model.bin",
    repo_id="username/model-name",
    repo_type="model"
)
```

---

## 9. 베스트 프랙티스

### 저장 전 체크리스트

```python
# 1. eval 모드
model.eval()

# 2. GPU → CPU (범용성)
model.cpu()

# 3. 검증
with torch.no_grad():
    test_output = model(test_input.cpu())
    assert test_output.shape == expected_shape
```

### 버전 관리

```python
save_dict = {
    'model_state_dict': model.state_dict(),
    'model_config': {
        'input_size': 784,
        'hidden_size': 256,
        'num_classes': 10
    },
    'pytorch_version': torch.__version__,
    'training_date': datetime.now().isoformat()
}
torch.save(save_dict, 'model_v1.0.pth')
```

---

## 정리

### 저장 방법 선택

| 용도 | 방법 |
|------|------|
| 학습 재개 | 체크포인트 (state_dict + optimizer) |
| Python 배포 | state_dict |
| C++ 배포 | TorchScript |
| 범용 배포 | ONNX |
| PyTorch 2.x 생태계 | torch.export() |
| 모바일 | PyTorch Mobile |

### 핵심 코드

```python
# 저장
torch.save(model.state_dict(), 'model.pth')

# TorchScript (레거시)
traced = torch.jit.trace(model.eval(), example_input)
traced.save('model.pt')

# torch.export() (PyTorch 2.x 권장)
exported = torch.export.export(model.eval(), (example_input,))
torch.export.save(exported, 'model.pt2')

# ONNX
torch.onnx.export(model, example_input, 'model.onnx')
```

---

## 다음 단계

[실전 이미지 분류 프로젝트](./39_Practical_Image_Classification.md)에서 실전 프로젝트를 진행합니다.
