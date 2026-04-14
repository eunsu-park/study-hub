# 모델 저장과 로드 (Model Saving and Loading)

**이전**: [학습 루프](./08_Training_Loop.md) | **다음**: [GPU 학습](./10_GPU_Training.md)

---

## 학습 목표 (Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `state_dict`를 사용하여 모델 가중치를 저장하고 로드할 수 있습니다
2. 완전한 학습 체크포인트(모델, 옵티마이저, 에포크, 손실)를 저장할 수 있습니다
3. 체크포인트에서 학습을 재개할 수 있습니다
4. 크로스 프레임워크 배포를 위해 모델을 ONNX 형식으로 내보낼 수 있습니다
5. 다른 하드웨어에서 모델을 로드할 때 장치 매핑을 처리할 수 있습니다
6. `torch.save`, `state_dict`, `torch.jit.save`의 차이를 이해할 수 있습니다
7. 학습 중 최적 모델 저장을 구현할 수 있습니다

---

## 1. state_dict 저장과 로드 (권장)

```python
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

# 저장
torch.save(model.state_dict(), 'model_weights.pt')

# 로드
model2 = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
model2.load_state_dict(torch.load('model_weights.pt'))
model2.eval()
```

### 왜 torch.save(model) 대신 state_dict인가?

```python
# 권장하지 않음: 전체 모델 저장 (클래스 정의 포함)
torch.save(model, 'entire_model.pt')
# 문제: 로드 시 정확히 같은 클래스 정의가 import 가능해야 함

# 권장: state dict만 저장
torch.save(model.state_dict(), 'model_weights.pt')
```

---

## 2. 학습 체크포인트

```python
# 완전한 체크포인트 저장
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'best_val_loss': best_val_loss,
}
torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')

# 체크포인트에서 학습 재개
checkpoint = torch.load('checkpoint_epoch_25.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

## 3. 장치 매핑

```python
# GPU에서 저장한 모델을 CPU에서 로드
model.load_state_dict(torch.load('model_weights.pt', map_location='cpu'))

# 장치 불가지론 로딩
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('model_weights.pt', map_location=device))
model.to(device)

# PyTorch 2.x: 보안을 위해 weights_only=True 사용
state_dict = torch.load('model_weights.pt', weights_only=True)
```

---

## 4. 부분 로딩 (전이 학습)

```python
old_state = torch.load('old_model.pt')
new_model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 20))

# 일치하는 키만 로드
new_state = new_model.state_dict()
pretrained = {k: v for k, v in old_state.items()
              if k in new_state and v.shape == new_state[k].shape}
new_state.update(pretrained)
new_model.load_state_dict(new_state)

# 또는 strict=False 사용
model.load_state_dict(old_state, strict=False)
```

---

## 5. ONNX 내보내기

```python
model.eval()
dummy_input = torch.randn(1, 784)

torch.onnx.export(
    model, dummy_input, 'model.onnx',
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=17,
)

# ONNX Runtime으로 추론
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('model.onnx')
input_data = np.random.randn(5, 784).astype(np.float32)
outputs = session.run(None, {'input': input_data})
```

---

## 6. SafeTensors 형식

```python
# safetensors: pickle 기반 torch.save보다 안전하고 빠른 대안
from safetensors.torch import save_file, load_file

save_file(model.state_dict(), 'model.safetensors')
state_dict = load_file('model.safetensors')
model.load_state_dict(state_dict)
```

---

## 요약

| 개념 | 핵심 내용 |
|------|----------|
| state_dict | 저장/로드의 권장 방법; 아키텍처 독립적 |
| 체크포인트 | 학습 재개를 위해 모델 + 옵티마이저 + 에포크 저장 |
| map_location | 로드 시 CPU/GPU 장치 차이 처리 |
| weights_only=True | 보안: pickle 기반 공격 방지 |
| strict=False | 전이 학습을 위한 부분 로딩 허용 |
| ONNX | 크로스 프레임워크 배포; dynamic_axes로 가변 배치 |
| SafeTensors | pickle 기반 직렬화보다 안전하고 빠른 대안 |

---

**다음**: [GPU 학습](./10_GPU_Training.md) -- 장치 관리와 혼합 정밀도로 GPU에서 학습.
