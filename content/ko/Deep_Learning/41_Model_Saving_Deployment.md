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
