# 커스텀 레이어와 함수 (Custom Layers and Functions)

**이전**: [PyTorch 디버깅](./11_Debugging_PyTorch.md) | **다음**: [TorchScript와 배포](./13_TorchScript_and_Deployment.md)

---

## 학습 목표 (Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `torch.autograd.Function`으로 커스텀 autograd 함수를 구현할 수 있습니다
2. `forward()`와 `backward()`를 정의할 수 있습니다
3. `torch.autograd.gradcheck`으로 커스텀 그래디언트를 검증할 수 있습니다
4. `nn.Module` 서브클래스로 재사용 가능한 커스텀 레이어를 구축할 수 있습니다
5. 커스텀 함수와 표준 모듈을 모델에 결합할 수 있습니다
6. PyTorch가 기본 제공하지 않는 연산을 구현할 수 있습니다
7. 커스텀 Function vs 커스텀 Module을 언제 사용할지 이해할 수 있습니다

---

## 1. 커스텀 autograd.Function

```python
import torch
from torch.autograd import Function

class MyReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# 사용법
x = torch.randn(5, requires_grad=True)
y = MyReLU.apply(x)  # .apply() 사용, 직접 호출 아님
loss = y.sum()
loss.backward()
```

---

## 2. 다중 입력 함수

```python
class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input @ weight.T
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_output @ weight           # dL/dinput
        grad_weight = grad_output.T @ input          # dL/dweight
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(dim=0)       # dL/dbias
        # forward 입력과 같은 순서로 한 개씩 그래디언트 반환
        return grad_input, grad_weight, grad_bias
```

---

## 3. 그래디언트 검증

```python
from torch.autograd import gradcheck

# double precision으로 테스트 (수치 정확도)
x = torch.randn(3, 4, dtype=torch.double, requires_grad=True)

result = gradcheck(MyReLU.apply, (x,), eps=1e-6, atol=1e-4)
print(f"그래디언트 검증 통과: {result}")
```

---

## 4. 실용적인 커스텀 함수

### 4.1 직통 추정기 (Straight-Through Estimator)

양자화와 이산 연산에 사용합니다:

```python
class StraightThroughEstimator(Function):
    @staticmethod
    def forward(ctx, input):
        return input.round()  # 가장 가까운 정수로 양자화

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # 그래디언트를 그대로 전달 (항등)

ste = StraightThroughEstimator.apply
x = torch.tensor([0.3, 0.7, 1.2], requires_grad=True)
y = ste(x)
print(y)  # tensor([0., 1., 1.])
y.sum().backward()
print(x.grad)  # tensor([1., 1., 1.])  -- 그래디언트가 통과
```

### 4.2 수치 안정 Log-Sum-Exp

```python
class StableLogSumExp(Function):
    @staticmethod
    def forward(ctx, input, dim):
        max_val = input.max(dim=dim, keepdim=True).values
        exp_shifted = (input - max_val).exp()
        sum_exp = exp_shifted.sum(dim=dim, keepdim=True)
        output = max_val + sum_exp.log()
        ctx.save_for_backward(input, output)
        ctx.dim = dim
        return output.squeeze(dim)

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        grad_input = torch.softmax(input, dim=ctx.dim)
        grad_input = grad_input * grad_output.unsqueeze(ctx.dim)
        return grad_input, None  # dim에 대해서는 None
```

---

## 5. 커스텀 레이어 (nn.Module)

### 5.1 커스텀 함수를 모듈로 감싸기

```python
import torch.nn as nn

class SwishLayer(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

model = nn.Sequential(
    nn.Linear(784, 256),
    SwishLayer(),
    nn.Linear(256, 10),
)
```

### 5.2 파라미터가 있는 커스텀 레이어

```python
class ScaleShift(nn.Module):
    """학습 가능한 요소별 스케일과 시프트: y = gamma * x + beta."""

    def __init__(self, num_features):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return self.gamma * x + self.beta

    def extra_repr(self):
        return f"num_features={self.gamma.shape[0]}"
```

---

## 6. 언제 무엇을 사용할까

| 시나리오 | 사용 |
|---------|------|
| 학습 가능한 파라미터가 있는 새 레이어 | `nn.Module` 서브클래스 |
| 커스텀 forward + 표준 backward | 내장 연산을 사용한 `nn.Module` |
| 커스텀 backward (안정성, 효율) | `autograd.Function` |
| STE가 있는 미분 불가능 연산 | `autograd.Function` |
| 외부 C/CUDA 코드 래핑 | `autograd.Function` + C 확장 |

---

## 요약

| 개념 | 핵심 내용 |
|------|----------|
| autograd.Function | 커스텀 forward와 backward; `.apply()`로 호출 |
| ctx.save_for_backward | backward에서 필요한 텐서 저장 |
| 그래디언트 반환 순서 | forward 입력당 하나의 그래디언트; 미분 불가능에는 None |
| gradcheck | 유한 차분과 비교하여 커스텀 그래디언트 검증; float64 사용 |
| 커스텀 Module | 학습 가능한 가중치에 nn.Parameter 사용한 nn.Module |
| STE | 미분 불가능 연산을 통해 그래디언트 전달 |

---

**다음**: [TorchScript와 배포](./13_TorchScript_and_Deployment.md) -- PyTorch 모델 컴파일과 배포.
