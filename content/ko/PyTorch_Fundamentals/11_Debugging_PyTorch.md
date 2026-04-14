# PyTorch 디버깅 (Debugging PyTorch)

**이전**: [GPU 학습](./10_GPU_Training.md) | **다음**: [커스텀 레이어와 함수](./12_Custom_Layers_and_Functions.md)

---

## 학습 목표 (Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. shape 불일치 에러를 체계적으로 진단하고 수정할 수 있습니다
2. 그래디언트를 검사하여 소실, 폭발, None 그래디언트 문제를 식별할 수 있습니다
3. 순전파 및 역전파 훅으로 중간값을 검사할 수 있습니다
4. `torch.autograd.detect_anomaly()`로 NaN을 생성하는 연산을 찾을 수 있습니다
5. 장치 불일치 및 dtype 에러를 디버깅할 수 있습니다
6. PyTorch에서 중단점과 print 기반 디버깅을 효과적으로 사용할 수 있습니다
7. 모델 성능을 프로파일링하여 병목을 찾을 수 있습니다

---

## 1. Shape 불일치 에러

```python
class DebugModel(nn.Module):
    def forward(self, x):
        print(f"입력:        {x.shape}")
        x = self.pool(torch.relu(self.conv1(x)))
        print(f"conv1 후:    {x.shape}")
        x = x.flatten(1)
        print(f"flatten 후:  {x.shape}")
        x = self.fc(x)
        print(f"출력:        {x.shape}")
        return x
```

### 일반적인 Shape 수정

```python
# 1. 누락된 배치 차원
x = torch.randn(784)           # [784]
x = x.unsqueeze(0)             # [1, 784]

# 2. CrossEntropyLoss shape 불일치
output = torch.randn(32, 10)   # 로짓: [배치, 클래스]
target = torch.randint(0, 10, (32,))  # 레이블: [배치] (원-핫 아님)
```

---

## 2. 그래디언트 디버깅

```python
def check_gradients(model, threshold=1e-7):
    """소실 또는 폭발 그래디언트를 확인합니다."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm < threshold:
                print(f"소실: {name} grad_norm={grad_norm:.2e}")
            elif grad_norm > 1000:
                print(f"폭발: {name} grad_norm={grad_norm:.2e}")
            elif torch.isnan(param.grad).any():
                print(f"NaN:  {name}")
            else:
                print(f"정상: {name} grad_norm={grad_norm:.4f}")

loss.backward()
check_gradients(model)
```

---

## 3. 훅 (Hooks)

### 3.1 순전파 훅

```python
activations = {}

def save_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

model[0].register_forward_hook(save_activation('linear1'))

output = model(torch.randn(1, 784))

for name, act in activations.items():
    print(f"{name}: shape={act.shape}, "
          f"mean={act.mean():.4f}, std={act.std():.4f}")
```

### 3.2 역전파 훅

```python
gradient_info = {}

def save_gradient(name):
    def hook(module, grad_input, grad_output):
        gradient_info[name] = {
            'grad_output': [g.norm().item() if g is not None else None
                           for g in grad_output],
        }
    return hook

model[0].register_full_backward_hook(save_gradient('linear1'))
```

---

## 4. 이상 감지 (Anomaly Detection)

```python
# 역전파 중 NaN이나 Inf를 생성하는 연산을 감지
with torch.autograd.detect_anomaly():
    x = torch.randn(3, requires_grad=True)
    y = x / 0  # 문제를 일으킬 것
    y.sum().backward()

# 주의: 상당히 느려짐, 디버깅에만 사용
```

---

## 5. 디버깅 체크리스트

```
1. Shape: 모든 연산 후 .shape 출력
2. Device: 모든 텐서의 .device 출력
3. dtype: 모든 텐서의 .dtype 출력
4. 값: .min(), .max(), .mean()으로 NaN/Inf 확인
5. 그래디언트: backward 후 .grad가 None이 아닌지 확인
6. 모드: model.training이 올바른지 확인 (train vs eval)
```

### breakpoint() 사용

```python
class MyModel(nn.Module):
    def forward(self, x):
        x = self.layer1(x)
        if torch.isnan(x).any():
            breakpoint()  # Python 디버거로 진입
        x = self.layer2(x)
        return x
```

### 최소 재현 예제

```python
# 디버깅에 막혔을 때, 가능한 가장 작은 예제로 축소

model = nn.Linear(3, 2)  # 가장 단순한 모델
x = torch.randn(1, 3)    # 단일 샘플
y = torch.tensor([0])     # 단일 레이블

output = model(x)
loss = nn.CrossEntropyLoss()(output, y)
loss.backward()

# 작동하면? 점진적으로 복잡도를 추가
```

---

## 6. 성능 프로파일링

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(5):
        output = model(torch.randn(32, 784).to(device))
        output.sum().backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

## 요약

| 문제 | 진단 도구 |
|------|----------|
| Shape 불일치 | 각 연산 후 `.shape` 출력 |
| None 그래디언트 | `requires_grad`, `detach()` 사용 확인 |
| 소실/폭발 그래디언트 | 레이어별 그래디언트 노름 모니터링 |
| 역전파 중 NaN | `torch.autograd.detect_anomaly()` |
| 장치 불일치 | 모든 텐서의 `.device` 출력 |
| dtype 불일치 | `.dtype` 출력; `.float()` 일관 사용 |
| 중간값 | 순전파 훅 |
| 그래디언트 흐름 | 역전파 훅 |

---

**다음**: [커스텀 레이어와 함수](./12_Custom_Layers_and_Functions.md) -- 커스텀 autograd 함수와 레이어 구현.
