# 자동 미분 (Autograd)

**이전**: [텐서 연산](./03_Tensor_Operations.md) | **다음**: [nn.Module](./05_nn_Module.md)

---

## 학습 목표 (Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. PyTorch의 autograd 엔진이 계산 그래프를 구축하고 순회하는 방법을 설명할 수 있습니다
2. `requires_grad`를 사용하여 텐서를 그래디언트 계산 대상으로 표시할 수 있습니다
3. `backward()`로 그래디언트를 계산하고 `.grad`로 접근할 수 있습니다
4. 그래프에서 리프(leaf) 텐서와 비리프(non-leaf) 텐서의 차이를 이해할 수 있습니다
5. `torch.no_grad()`와 `detach()`를 사용하여 그래디언트 계산을 제어할 수 있습니다
6. 고차 그래디언트와 야코비안-벡터 곱을 계산할 수 있습니다
7. 그래디언트 문제(None, 누적 버그, 인플레이스 에러)를 디버깅할 수 있습니다
8. `retain_graph`와 `create_graph`를 고급 autograd 시나리오에 적용할 수 있습니다

---

자동 미분(autograd)은 신경망 학습을 가능하게 하는 엔진입니다. 수동으로 그래디언트를 계산하는 대신, PyTorch가 `requires_grad=True`인 텐서에 대한 모든 연산을 기록하고, 역방향으로 순회하여 모든 그래디언트를 동시에 계산합니다.

---

## 1. 계산 그래프

### 1.1 작동 방식

`requires_grad=True`인 텐서에 연산을 수행하면, PyTorch는 각 연산을 **방향성 비순환 그래프(DAG)**에 기록합니다:

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

z = x * y          # MulBackward0
w = z + x          # AddBackward0
loss = w ** 2       # PowBackward0

print(loss)              # tensor(64., grad_fn=<PowBackward0>)
```

### 1.2 순전파와 역전파

```python
x = torch.tensor(2.0, requires_grad=True)

# 순전파: 출력 계산
y = x ** 3 + 2 * x ** 2 + x
# y = x^3 + 2x^2 + x
# dy/dx = 3x^2 + 4x + 1

# 역전파: 그래디언트 계산
y.backward()

print(x.grad)  # tensor(21.)  (3*4 + 4*2 + 1 = 21)
```

---

## 2. 리프 텐서와 grad

### 2.1 리프 vs 비리프

```python
# 리프 텐서: 사용자가 직접 생성
a = torch.tensor(1.0, requires_grad=True)   # 리프
b = torch.randn(3, requires_grad=True)      # 리프

# 비리프 텐서: 연산의 결과
d = a * 2     # 비리프 (grad_fn 있음)
print(a.is_leaf)  # True
print(d.is_leaf)  # False
```

### 2.2 그래디언트 누적

**그래디언트는 기본적으로 누적됩니다** -- 교체되는 것이 아니라 더해집니다:

```python
x = torch.tensor(2.0, requires_grad=True)

# 첫 번째 backward
y1 = x ** 2
y1.backward()
print(x.grad)  # tensor(4.)

# 그래디언트를 초기화하지 않고 두 번째 backward
y2 = x ** 3
y2.backward()
print(x.grad)  # tensor(16.)  12가 아님! (4 + 12 = 16, 누적!)

# 반드시 backward 전에 그래디언트를 초기화
x.grad.zero_()
y3 = x ** 3
y3.backward()
print(x.grad)  # tensor(12.)  올바른 값!
```

> **핵심 규칙**: 학습 루프에서 `loss.backward()` 전에 항상 `optimizer.zero_grad()`(또는 수동으로 그래디언트 초기화)를 호출하세요. 이를 잊는 것은 가장 흔한 PyTorch 버그 중 하나입니다.

---

## 3. 그래디언트 계산 제어

### 3.1 torch.no_grad()

추론과 파라미터 업데이트 중에 그래디언트 추적을 비활성화합니다:

```python
x = torch.tensor(2.0, requires_grad=True)

with torch.no_grad():
    y = x * 3
    print(y.requires_grad)  # False

# 일반적 사용:
# 1. 평가/추론
model.eval()
with torch.no_grad():
    predictions = model(test_data)

# 2. 수동 파라미터 업데이트
with torch.no_grad():
    param -= learning_rate * param.grad
```

### 3.2 detach()

데이터를 공유하지만 그래프에서 분리된 텐서를 생성합니다:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

y_detached = y.detach()
print(y_detached.requires_grad)  # False

# 사용 사례: 모델 출력을 다른 계산의 고정 입력으로 사용
features = encoder(x)
features_fixed = features.detach()  # encoder로의 그래디언트 흐름 중단
output = decoder(features_fixed)
```

---

## 4. backward() 상세

### 4.1 스칼라 출력

출력이 스칼라이면 `backward()`에 인자가 필요 없습니다:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x ** 2).sum()  # 스칼라
loss.backward()
print(x.grad)  # tensor([2., 4., 6.])
```

### 4.2 비스칼라 출력 (야코비안-벡터 곱)

출력이 스칼라가 아니면 `gradient` 인자를 제공해야 합니다:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # 스칼라가 아님 -- shape [3]

# 야코비안-벡터 곱의 "벡터"를 제공해야 함
y.backward(torch.ones_like(y))
print(x.grad)  # tensor([2., 4., 6.])
```

### 4.3 retain_graph

기본적으로 계산 그래프는 `backward()` 후에 해제됩니다:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

y.backward()       # 그래프 해제됨
# y.backward()     # 에러: 그래프 이미 해제됨

# 여러 번 backward하려면 retain_graph 사용
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward(retain_graph=True)  # 그래프 유지
x.grad.zero_()
y.backward()  # 이제 작동 (이전 호출에서 그래프 유지됨)
```

---

## 5. 고차 그래디언트

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3  # y = x^3, dy/dx = 3x^2, d2y/dx2 = 6x

# 1차 미분
grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
print(grad1)  # tensor(12.)  (3 * 4 = 12)

# 2차 미분
grad2 = torch.autograd.grad(grad1, x)[0]
print(grad2)  # tensor(12.)  (6 * 2 = 12)
```

---

## 6. 일반적인 Autograd 에러

### 6.1 None 그래디언트

```python
x = torch.tensor(2.0)  # requires_grad=False (기본값)
y = x ** 2
# y.backward()  -- x.grad는 None (requires_grad=False이므로)

# 수정:
x = torch.tensor(2.0, requires_grad=True)
```

### 6.2 리프 변수에 대한 인플레이스 연산

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
# x.add_(1)  # 에러: 리프 Variable에 대한 인플레이스 연산

# 수정: 아웃-오브-플레이스 연산 사용
y = x + 1
```

### 6.3 그래디언트 미초기화

```python
# 증상: 손실이 예상대로 감소하지 않음
# 원인: 반복 간에 그래디언트가 누적됨

# 수정: backward 전에 그래디언트 초기화
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 7. Autograd 내부

### 7.1 grad_fn 체인

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 3       # MulBackward0
z = y + 1       # AddBackward0
w = z ** 2       # PowBackward0

# 그래프 탐색
print(w.grad_fn)                           # PowBackward0
print(w.grad_fn.next_functions)            # ((AddBackward0, 0),)
```

### 7.2 훅

backward 중에 실행되는 함수를 등록합니다:

```python
x = torch.tensor(2.0, requires_grad=True)

def print_grad(grad):
    print(f"그래디언트: {grad}")

x.register_hook(print_grad)

y = x ** 2
y.backward()  # 출력: "그래디언트: 4.0"
```

---

## 요약

| 개념 | 핵심 내용 |
|------|----------|
| 계산 그래프 | 순전파 중 동적으로 구축; 역전파 중 역순으로 순회 |
| requires_grad | 최적화하려는 파라미터에 True로 설정 |
| backward() | 그래디언트를 계산하고 리프 텐서의 `.grad`에 저장 |
| 그래디언트 누적 | 기본적으로 그래디언트가 더해짐; 항상 backward 전에 초기화 |
| torch.no_grad() | 그래프 구성 비활성화; 추론과 파라미터 업데이트에 사용 |
| detach() | 그래디언트 흐름 차단; 그래프와 독립된 텐서 생성 |
| retain_graph | 여러 번 backward를 위해 backward 후 그래프 유지 |
| create_graph | 고차 그래디언트 계산 활성화 |

---

**다음**: [nn.Module](./05_nn_Module.md) -- PyTorch의 모듈 시스템으로 신경망 아키텍처 구축.
