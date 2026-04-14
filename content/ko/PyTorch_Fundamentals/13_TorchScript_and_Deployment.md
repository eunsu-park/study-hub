# TorchScript와 배포 (TorchScript and Deployment)

**이전**: [커스텀 레이어와 함수](./12_Custom_Layers_and_Functions.md) | **다음**: [PyTorch 생태계](./14_PyTorch_Ecosystem.md)

---

## 학습 목표 (Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Eager 모드와 TorchScript의 차이를 설명할 수 있습니다
2. 트레이싱과 스크립팅으로 모델을 TorchScript로 변환할 수 있습니다
3. `torch.jit.trace`와 `torch.jit.script`를 언제 사용할지 이해할 수 있습니다
4. `torch.export` (PyTorch 2.x)로 그래프를 캡처할 수 있습니다
5. 크로스 플랫폼 배포를 위해 모델을 ONNX로 내보낼 수 있습니다
6. TorchServe로 PyTorch 모델을 배포할 수 있습니다
7. 양자화 기초로 추론용 모델을 최적화할 수 있습니다

---

## 1. TorchScript 개요

TorchScript는 Python 런타임 없이 PyTorch 모델을 직렬화하고 최적화하는 방법입니다.

### 두 가지 변환 방법

| 방법 | 방식 | 사용 시기 |
|------|------|----------|
| **트레이싱** | 샘플 입력으로 모델 실행, 연산 기록 | 제어 흐름 없는 모델 |
| **스크립팅** | Python 소스 코드를 IR로 파싱 | 동적 제어 흐름이 있는 모델 |

---

## 2. 트레이싱

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

model = SimpleModel()
model.eval()

# 예제 입력으로 트레이싱
traced_model = torch.jit.trace(model, torch.randn(1, 784))

# 저장 (Python 모델 정의 필요 없음!)
traced_model.save('traced_model.pt')

# 로드
loaded = torch.jit.load('traced_model.pt')
output = loaded(torch.randn(1, 784))
```

> **주의**: 모델에 입력 데이터에 따른 `if`, `for`, `while`이 있으면 트레이싱 대신 스크립팅을 사용하세요.

---

## 3. 스크립팅

```python
class DynamicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        if x.sum() > 0:
            return torch.relu(self.linear(x))
        else:
            return torch.sigmoid(self.linear(x))

scripted_model = torch.jit.script(DynamicModel().eval())
# 두 분기 모두 올바르게 작동
scripted_model.save('scripted_model.pt')
```

---

## 4. torch.export (PyTorch 2.x)

```python
model = SimpleModel().eval()
exported = torch.export.export(model, (torch.randn(1, 784),))

# 동적 shape
from torch.export import Dim
batch = Dim("batch", min=1, max=256)
exported = torch.export.export(
    model, (torch.randn(1, 784),),
    dynamic_shapes={"x": {0: batch}},
)
```

---

## 5. torch.compile (런타임 최적화)

```python
model = SimpleModel().to(device)
compiled_model = torch.compile(model)

# 첫 번째 호출: 컴파일 (느림)
# 이후 호출: 최적화 (빠름)
output = compiled_model(torch.randn(32, 784, device=device))

# 컴파일 모드
model = torch.compile(model, mode="reduce-overhead")  # CPU 오버헤드 최소화
model = torch.compile(model, mode="max-autotune")      # 최대 최적화
```

### 비교

| 기능 | torch.compile | TorchScript | torch.export |
|------|--------------|-------------|-------------|
| **목표** | 런타임 속도 | 직렬화 | 그래프 캡처 |
| **디스크 저장** | 아니오 | 예 | 예 |
| **Python 필요** | 예 | 아니오 | 아니오 |
| **성능** | 최고 | 양호 | 양호 |

---

## 6. ONNX 내보내기

```python
model.eval()
torch.onnx.export(
    model, torch.randn(1, 784), "model.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=17,
)

# ONNX Runtime으로 추론
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {"input": np.random.randn(10, 784).astype(np.float32)})
```

---

## 7. TorchServe

```bash
# 모델 아카이브 생성
torch-model-archiver \
    --model-name my_model \
    --version 1.0 \
    --serialized-file model_weights.pt \
    --handler handler.py \
    --export-path model_store/

# TorchServe 시작
torchserve --start --model-store model_store --models my_model=my_model.mar
```

---

## 8. 양자화 (간략 소개)

```python
# 동적 양자화
quantized_model = torch.ao.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# 모델 크기 비교
import os
torch.save(model.state_dict(), 'fp32_model.pt')
torch.save(quantized_model.state_dict(), 'int8_model.pt')
print(f"FP32: {os.path.getsize('fp32_model.pt')/1024:.1f} KB")
print(f"INT8: {os.path.getsize('int8_model.pt')/1024:.1f} KB")
```

---

## 요약

| 도구 | 용도 | 핵심 명령 |
|------|------|----------|
| 트레이싱 | 제어 흐름 없는 모델 직렬화 | `torch.jit.trace(model, input)` |
| 스크립팅 | 제어 흐름 있는 모델 직렬화 | `torch.jit.script(model)` |
| torch.export | 현대적 그래프 캡처 | `torch.export.export(model, args)` |
| torch.compile | 런타임 최적화 (직렬화 없음) | `torch.compile(model)` |
| ONNX | 크로스 프레임워크 내보내기 | `torch.onnx.export(model, ...)` |
| TorchServe | 모델 서빙 | `torch-model-archiver` + `torchserve` |
| 양자화 | 모델 압축 | `quantize_dynamic(model, ...)` |

---

**다음**: [PyTorch 생태계](./14_PyTorch_Ecosystem.md) -- PyTorch를 확장하는 라이브러리와 도구.
