# 03. 역전파 이해

[이전: 신경망 기초](./02_Neural_Network_Basics.md) | [다음: 훈련 기법](./04_Training_Techniques.md)

---

## 학습 목표

- 역전파(Backpropagation) 알고리즘의 원리 이해
- 체인 룰(Chain Rule)을 이용한 기울기 계산
- NumPy로 역전파 직접 구현

---

## 1. 역전파란?

역전파는 신경망의 가중치를 학습하기 위한 알고리즘입니다.

```
Forward Pass:  Input ──▶ Hidden Layer ──▶ Output ──▶ Loss
Backward Pass: Input ◀── Hidden Layer ◀── Output ◀── Loss
```

### 직관: 공헌도 배분(Credit Assignment)

신경망을 여러 작업 스테이션이 있는 조립 라인이라고 생각해보세요. 최종 제품(예측)에 결함이 있으면(높은 손실), **어느 스테이션이 얼마나 책임이 있는지** 파악해야 합니다 — 이것이 *공헌도 배분 문제(Credit Assignment Problem)*입니다. 역전파(Backpropagation)는 오류를 체인을 따라 역추적하여 이를 해결합니다: 각 스테이션(층)이 입력에 대한 국소 민감도("내 입력이 ε만큼 변하면, 출력은 ∂output/∂input × ε만큼 변한다")를 보고하고, 이 국소 민감도들이 체인을 따라 곱해져 각 매개변수의 최종 오차 기여도를 산출합니다.

### 핵심 아이디어

1. **순전파**: 입력에서 출력까지 값 계산
2. **손실 계산**: 예측과 정답의 차이
3. **역전파**: 손실에서 입력 방향으로 그래디언트를 전파 — 각 층이 들어오는 그래디언트에 자신의 국소 야코비안(Jacobian)을 곱함
4. **가중치 업데이트**: 기울기를 이용해 가중치 조정

---

## 2. 체인 룰 (Chain Rule)

신경망에서 연쇄법칙(chain rule)이 왜 중요할까요? 신경망은 깊이 중첩된 함수 합성입니다: `L(softmax(W2 * relu(W1 * x + b1) + b2), y)`. `dL/dW1`을 계산하려면 직접 미분할 수 없습니다 — 각 층의 국소 미분의 곱으로 분해해야 합니다. 연쇄법칙이 바로 이 분해를 제공하여, 전체 식을 펼치지 않고도 층별로 기울기를 계산할 수 있게 합니다.

합성 함수의 미분 법칙입니다.

### 수식

```
y = f(g(x))

dy/dx = (dy/dg) × (dg/dx)
```

### 예시

```
z = x²
y = sin(z)
L = y²

dL/dx = (dL/dy) × (dy/dz) × (dz/dx)
      = 2y × cos(z) × 2x
```

---

## 3. 단일 뉴런의 역전파

### 순전파

```python
z = w*x + b      # Linear transformation
a = sigmoid(z)    # Activation
L = (a - y)²     # Loss (MSE)
```

### 역전파 (기울기 계산)

```python
dL/da = 2(a - y)                    # Gradient of loss w.r.t. activation
da/dz = sigmoid(z) * (1 - sigmoid(z))  # Sigmoid derivative
dz/dw = x                           # Gradient of linear transform w.r.t. weight
dz/db = 1                           # Gradient of linear transform w.r.t. bias

# Apply chain rule
dL/dw = (dL/da) × (da/dz) × (dz/dw)
dL/db = (dL/da) × (da/dz) × (dz/db)
```

---

## 4. 손실 함수

### MSE (Mean Squared Error)

```python
L = (1/n) × Σ(y_pred - y_true)²
dL/dy_pred = (2/n) × (y_pred - y_true)
```

### Cross-Entropy (분류)

```python
L = -Σ y_true × log(y_pred)
dL/dy_pred = -y_true / y_pred  # Simplified when combined with softmax
```

### Softmax + Cross-Entropy 결합

```python
# Amazing result: becomes very simple
dL/dz = y_pred - y_true  # Gradient w.r.t. softmax input
```

---

## 5. MLP 역전파

2층 MLP의 역전파 과정입니다.

### 구조

```
Input(x) → [W1, b1] → ReLU → [W2, b2] → Output(y)
```

### 순전파

```python
# Why save z1 and a1?  The backward pass needs these intermediate values
# to compute gradients — this is the memory-compute tradeoff of backprop.
z1 = x @ W1 + b1      # Linear transform: project input into hidden space
a1 = relu(z1)          # Non-linearity: enable learning of non-linear patterns
z2 = a1 @ W2 + b2      # Linear transform: project hidden representation to output
y_pred = z2            # Or softmax(z2) for classification
```

### 역전파

```python
# Output layer
dL/dz2 = y_pred - y_true  # (for softmax + CE)

# Why transpose?  In the forward pass, z2 = a1 @ W2, where a1 is (batch, H)
# and W2 is (H, out).  To get dL/dW2, we need to "undo" the matmul so that
# each element W2[i,j] gets credited for how much it contributed to the loss.
# By the chain rule: dL/dW2[i,j] = Σ_batch a1[:,i] × dL/dz2[:,j].
# In matrix form that's exactly a1.T @ dL/dz2  (transpose aligns the
# batch dimension for the dot product).
dL/dW2 = a1.T @ dL/dz2     # (H, batch) @ (batch, out) → (H, out)
dL/db2 = sum(dL/dz2, axis=0)

# Hidden layer — propagating the gradient backwards through W2:
# In the forward pass, z2 = a1 @ W2.  To compute dL/da1, we need the
# gradient w.r.t. a1 (the *input* to this matmul).  Again by chain rule:
# dL/da1[:,i] = Σ_j dL/dz2[:,j] × W2[i,j], which in matrix form is
# dL/dz2 @ W2.T — the transpose of W2 "routes" each output gradient
# back to the input dimension that produced it.
dL/da1 = dL/dz2 @ W2.T     # (batch, out) @ (out, H) → (batch, H)
dL/dz1 = dL/da1 * relu_derivative(z1)  # element-wise: ReLU passes gradient where z1 > 0, blocks where z1 ≤ 0
dL/dW1 = x.T @ dL/dz1
dL/db1 = sum(dL/dz1, axis=0)
```

---

## 6. NumPy 구현 핵심

```python
class MLP:
    def backward(self, x, y_true, y_pred, cache):
        """Backpropagation: compute gradients"""
        # Why unpack cache?  Forward pass saved intermediate activations (a1)
        # and pre-activation values (z1) — we need them here to compute gradients.
        # Without caching, we'd have to recompute the forward pass.
        a1, z1 = cache

        # --- Output layer gradients ---
        # Why start here?  Backprop works from the loss backward.
        # For softmax + cross-entropy, the combined gradient simplifies to (y_pred - y_true).
        dz2 = y_pred - y_true
        # Why a1.T?  dL/dW2[i,j] = sum_over_batch(a1[:,i] * dz2[:,j]),
        # which is the matrix product a1.T @ dz2.
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        # --- Hidden layer gradients (chain rule) ---
        da1 = dz2 @ self.W2.T  # Route output-layer error back through W2
        # Why (z1 > 0)?  This is ReLU's derivative: gradient flows where neurons
        # were active (z1 > 0), and is blocked where they were off (z1 <= 0).
        dz1 = da1 * (z1 > 0)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # Why return a dict?  The optimizer needs each parameter's gradient separately
        # to apply the update rule (e.g., W -= lr * grad).
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
```

---

## 7. PyTorch의 자동 미분

PyTorch에서는 이 모든 과정이 자동입니다.

```python
# Forward pass
y_pred = model(x)
loss = criterion(y_pred, y_true)

# Backward pass (automatic!)
loss.backward()

# Access gradients
print(model.fc1.weight.grad)
```

### 계산 그래프

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y * 3
z.backward()

# x.grad = dz/dx = dz/dy × dy/dx = 3 × 2x = 12
```

---

## 8. 기울기 소실/폭발 문제

### 기울기 소실 (Vanishing Gradient)

- 원인: 시그모이드/tanh의 미분이 0에 가까움
- 해결: ReLU, 잔차 연결(Residual Connection)

### 기울기 폭발 (Exploding Gradient)

- 원인: 깊은 네트워크에서 기울기 누적
- 해결: Gradient Clipping, Batch Normalization

```python
# Gradient Clipping in PyTorch
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 9. 수치 기울기 검증

역전파 구현이 올바른지 확인하는 방법입니다.

```python
def numerical_gradient(f, x, h=1e-5):
    """Compute gradient using numerical differentiation"""
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy()
        x_plus.flat[i] += h
        x_minus = x.copy()
        x_minus.flat[i] -= h
        grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# Verification
analytical_grad = backward(...)  # Analytical gradient
numerical_grad = numerical_gradient(loss_fn, weights)
diff = np.linalg.norm(analytical_grad - numerical_grad)
assert diff < 1e-5, "Gradient check failed!"
```

---

## 정리

### 역전파의 핵심

1. **체인 룰**: 합성 함수 미분의 핵심
2. **국소적 계산**: 각 층에서 독립적으로 기울기 계산
3. **기울기 전파**: 출력에서 입력 방향으로 전파

### NumPy로 배우는 것

- 행렬의 전치와 곱셈의 의미
- 활성화 함수 미분의 역할
- 배치 처리에서의 기울기 합산

### PyTorch로 넘어가면

- `loss.backward()` 한 줄로 모든 기울기 계산
- 계산 그래프 자동 구성
- GPU 가속

---

## 연습 문제

### 연습 1: 체인 룰(Chain Rule) 손으로 계산하기

아래 합성 함수에 대해 체인 룰을 이용하여 기울기 `dL/dx`를 분석적으로 구하세요.

```
x → z = 3x + 1 → a = relu(z) → L = a²
```

1. 각 중간 도함수 `dL/da`, `da/dz`, `dz/dx`를 작성하세요.
2. 체인 룰을 적용하여 `dL/dx`를 구하세요.
3. 유한 차분 공식 `(f(x+h) - f(x-h)) / (2h)` (h=1e-5, x=2.0)으로 수치적으로 검증하세요.

### 연습 2: 단일 뉴런 역전파(Backpropagation)

시그모이드 활성화와 MSE 손실을 사용하는 단일 뉴런의 순전파와 역전파를 직접 구현하세요.

1. `w=0.5`, `b=0.1`, `x=1.0`, `y_true=0.8`을 사용하세요.
2. 순전파 계산: `z = w*x + b`, `a = sigmoid(z)`, `L = (a - y_true)²`
3. 기울기 `dL/dw`와 `dL/db`를 단계별로 손으로 계산하세요.
4. NumPy로 구현하여 손으로 계산한 값과 일치하는지 검증하세요.

### 연습 3: 2층 MLP 기울기 검사(Gradient Check)

수치 기울기 검증으로 역전파 구현이 올바른지 확인하세요.

1. 난수 가중치로 2층 MLP(입력=4, 은닉=8, 출력=2)를 NumPy로 구성하세요.
2. 분석적 역전파를 구현하세요.
3. 중앙 차분 공식으로 `numerical_gradient`를 구현하세요.
4. 상대 오차 `||분석적 - 수치적|| / (||분석적|| + ||수치적||)`를 계산하세요.
5. 상대 오차가 `1e-5` 미만임을 확인하세요.

### 연습 4: 시그모이드 vs ReLU에서의 기울기 소실(Vanishing Gradient)

기울기 소실 문제를 경험적으로 관찰하세요.

1. PyTorch로 시그모이드 활성화를 사용하는 10개 은닉층을 가진 깊은 MLP를 구성하세요.
2. `torch.nn.init.normal_(std=1.0)`으로 가중치를 초기화하세요.
3. 순전파를 실행하고 `loss.backward()`를 호출하세요.
4. 각 층의 가중치 기울기 노름(Norm)을 출력하세요.
5. ReLU 활성화로 1~4단계를 반복하고 층별 기울기 노름을 비교하여 차이를 설명하세요.

### 연습 5: 계산 그래프와 `retain_graph`

PyTorch의 동적 계산 그래프를 탐구하세요.

1. `x`와 `y`가 기울기를 필요로 하도록 설정하고 `z = x**2 + y**2`를 계산하세요.
2. `z.backward()`를 호출하고 `x.grad`와 `y.grad`를 출력하세요.
3. `z.backward()`를 다시 호출하여 발생하는 오류를 확인하세요.
4. 계산을 재생성하고 `z.backward(retain_graph=True)`를 두 번 호출하세요. 기울기가 누적됨(예상값의 두 배)을 확인하고, 학습 루프에서 `optimizer.zero_grad()`가 필요한 이유를 설명하세요.

---

## 다음 단계

[학습 기법](./04_Training_Techniques.md)에서 기울기를 이용한 가중치 업데이트 방법을 학습합니다.
