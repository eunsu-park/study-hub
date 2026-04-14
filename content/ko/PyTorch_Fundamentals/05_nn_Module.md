# nn.Module

**이전**: [자동 미분](./04_Autograd.md) | **다음**: [손실 함수와 옵티마이저](./06_Loss_Functions_and_Optimizers.md)

---

## 학습 목표 (Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `nn.Module`을 상속하여 커스텀 신경망 클래스를 정의할 수 있습니다
2. `forward()` 메서드를 구현하고 `__call__`이 이를 감싸는 이유를 이해할 수 있습니다
3. 직렬화와 장치 관리를 위해 파라미터와 버퍼를 올바르게 등록할 수 있습니다
4. 내장 레이어(`nn.Linear`, `nn.Conv2d`, `nn.ReLU`, `nn.Dropout` 등)를 사용할 수 있습니다
5. `nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict`로 모듈을 합성할 수 있습니다
6. `parameters()`, `named_parameters()`, `state_dict()`로 모델의 파라미터를 검사할 수 있습니다
7. `.to()`로 모델 전체를 장치와 dtype 간에 이동할 수 있습니다
8. 커스텀 가중치 초기화 전략을 적용할 수 있습니다

---

`nn.Module`은 PyTorch의 모든 신경망 구성 요소의 기본 클래스입니다. 모든 레이어, 손실 함수, 모델이 `nn.Module`입니다. 모듈을 정의하고, 합성하고, 검사하는 방법을 배우는 것이 모든 아키텍처를 구축하는 관문입니다.

---

## 1. 모듈 정의

### 1.1 기본 구조

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()  # 반드시 부모 __init__ 호출
        self.linear1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = SimpleNet()
x = torch.randn(32, 784)
output = model(x)          # model.forward(x)를 훅과 함께 호출
print(output.shape)        # torch.Size([32, 10])
```

### 1.2 왜 `model(x)`이지 `model.forward(x)`가 아닌가

항상 `model(x)`를 호출하고, `model.forward(x)`를 직접 호출하지 마세요:

```python
# model(x)가 하는 일:
# 1. 등록된 forward pre-hook 실행
# 2. self.forward(x) 호출
# 3. 등록된 forward hook 실행
# 4. 결과 반환

output = model(x)          # 올바른 방법
output = model.forward(x)  # 잘못된 방법 (훅을 건너뜀)
```

---

## 2. 파라미터와 버퍼

### 2.1 nn.Parameter

파라미터는 그래디언트 계산을 위해 자동으로 등록되는 텐서입니다:

```python
class CustomLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.T + self.bias
```

### 2.2 버퍼

버퍼는 모델과 함께 저장되어야 하지만 파라미터가 아닌(그래디언트 없는) 텐서입니다:

```python
class BatchNormManual(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
```

> **규칙**: 학습 가능한 텐서에는 `nn.Parameter`, 저장/이동이 필요한 비학습 텐서에는 `register_buffer`, 그 외에는 일반 속성을 사용하세요.

---

## 3. 내장 레이어

### 3.1 선형 레이어

```python
linear = nn.Linear(in_features=128, out_features=64)  # y = xW^T + b
linear_no_bias = nn.Linear(128, 64, bias=False)        # 편향 없이
```

### 3.2 활성화 함수

```python
# 모듈로 (nn.Sequential에서 사용)
relu = nn.ReLU()
gelu = nn.GELU()

# 함수로 (forward()에서 사용)
import torch.nn.functional as F
y = F.relu(x)
y = F.gelu(x)
y = F.softmax(x, dim=-1)
```

### 3.3 정규화

```python
bn = nn.BatchNorm1d(num_features=128)   # 배치 정규화
ln = nn.LayerNorm(normalized_shape=128)  # 레이어 정규화
gn = nn.GroupNorm(num_groups=8, num_channels=64)  # 그룹 정규화
```

### 3.4 드롭아웃

```python
dropout = nn.Dropout(p=0.5)  # 50%의 원소를 랜덤으로 0으로

# 드롭아웃은 train 모드에서 활성, eval 모드에서 비활성
model.train()  # 드롭아웃 활성
model.eval()   # 드롭아웃 비활성
```

---

## 4. 모듈 합성

### 4.1 nn.Sequential

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

output = model(torch.randn(32, 784))  # 모든 레이어를 순서대로 통과
```

### 4.2 nn.ModuleList

동적 모듈 컬렉션에 사용합니다:

```python
class MultiHeadModel(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.shared = nn.Linear(784, 256)
        self.heads = nn.ModuleList([
            nn.Linear(256, 10) for _ in range(n_heads)
        ])

    def forward(self, x):
        x = F.relu(self.shared(x))
        return [head(x) for head in self.heads]
```

> **경고**: 일반 Python 리스트(`self.heads = [...]`)는 모듈을 등록하지 않습니다. 항상 `nn.ModuleList`를 사용하세요.

### 4.3 중첩 모듈

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)  # 잔차 연결

class DeepNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_blocks, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(n_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.input_proj(x))
        x = self.blocks(x)
        return self.output_proj(x)
```

---

## 5. 모델 검사

```python
model = DeepNet(784, 256, 4, 10)

# 파라미터 수 세기
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"전체: {total:,}  학습 가능: {trainable:,}")

# state_dict
sd = model.state_dict()
for key, value in sd.items():
    print(f"{key}: {value.shape}")
```

---

## 6. 장치와 dtype 관리

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNet().to(device)  # 모든 파라미터와 버퍼를 이동

# dtype 변환
model = model.to(torch.float16)  # 반정밀도
model = model.float()            # float32로 복귀
```

---

## 7. 가중치 초기화

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

model = SimpleNet()
model.apply(init_weights)  # 모든 하위 모듈에 재귀적으로 적용
```

---

## 8. train() vs eval()

```python
model.train()  # 학습 모드: 드롭아웃 활성, BatchNorm이 배치 통계 사용
model.eval()   # 평가 모드: 드롭아웃 비활성, BatchNorm이 러닝 통계 사용

print(model.training)  # True 또는 False
```

> **중요**: `model.eval()`은 그래디언트 계산을 비활성화하지 않습니다. 그래디언트를 비활성화하려면 여전히 `torch.no_grad()`가 필요합니다. `eval()`은 Dropout과 BatchNorm 같은 레이어의 동작만 변경합니다.

---

## 요약

| 개념 | 핵심 내용 |
|------|----------|
| nn.Module | 모든 신경망 구성 요소의 기본 클래스 |
| forward() | 계산 정의; `model(x)`로 호출, `model.forward(x)` 사용 금지 |
| nn.Parameter | 학습 가능한 텐서, 모듈에 자동 등록 |
| register_buffer | 모델과 함께 저장되는 비학습 텐서 |
| nn.Sequential | 레이어를 순서대로 연결 |
| nn.ModuleList/Dict | 동적 하위 모듈 컬렉션 (올바르게 등록됨) |
| state_dict | 모든 파라미터와 버퍼의 직렬화 가능한 스냅샷 |
| train()/eval() | Dropout, BatchNorm 등의 동작 전환 |

---

**다음**: [손실 함수와 옵티마이저](./06_Loss_Functions_and_Optimizers.md) -- 손실 함수와 최적화 알고리즘의 선택과 구성.
