# 03. 행렬 미분 (Matrix Calculus)

## 학습 목표

- 스칼라-벡터 미분과 그래디언트의 개념을 이해하고 계산할 수 있다
- 야코비안 행렬의 정의와 체인 룰 적용 방법을 학습한다
- 헤시안 행렬의 의미와 최적화에서의 역할을 이해한다
- 주요 행렬 미분 항등식을 유도하고 활용할 수 있다
- 머신러닝의 손실 함수 그래디언트를 직접 유도할 수 있다
- PyTorch의 자동 미분 기능을 이해하고 검증에 활용할 수 있다

---

## 1. 스칼라-벡터 미분 (Scalar-by-Vector Derivatives)

### 1.1 그래디언트의 정의

스칼라 함수 $f: \mathbb{R}^n \to \mathbb{R}$에 대해 그래디언트는 모든 편미분을 모아놓은 벡터입니다:

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

그래디언트는 함수가 가장 가파르게 증가하는 방향을 가리킵니다.

### 1.2 기본 예제

**예제 1**: $f(\mathbf{x}) = \mathbf{a}^T \mathbf{x}$

$$\frac{\partial}{\partial \mathbf{x}}(\mathbf{a}^T \mathbf{x}) = \mathbf{a}$$

**예제 2**: $f(\mathbf{x}) = \mathbf{x}^T \mathbf{x}$

$$\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T \mathbf{x}) = 2\mathbf{x}$$

**예제 3**: $f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ (이차 형식)

$$\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T A \mathbf{x}) = (A + A^T)\mathbf{x}$$

$A$가 대칭이면 $2A\mathbf{x}$가 됩니다.

### 1.3 Python 구현: 이차 형식의 미분

```python
import numpy as np
import torch
from sympy import symbols, Matrix, diff, simplify

# SymPy로 심볼릭 계산
print("=== SymPy 심볼릭 미분 ===")
x1, x2 = symbols('x1 x2')
x = Matrix([x1, x2])
A = Matrix([[2, 1], [1, 3]])

# f(x) = x^T A x
f = (x.T * A * x)[0]
print(f"f(x) = {f}")

# 그래디언트 계산
grad_f = Matrix([diff(f, x1), diff(f, x2)])
print(f"∇f = {simplify(grad_f)}")
print(f"(A + A^T)x = {simplify((A + A.T) * x)}")

# PyTorch로 수치 계산 및 검증
print("\n=== PyTorch 자동 미분 ===")
x_val = torch.tensor([1.0, 2.0], requires_grad=True)
A_torch = torch.tensor([[2.0, 1.0], [1.0, 3.0]])

# 순전파
f_val = x_val @ A_torch @ x_val
print(f"f(x) = {f_val.item():.4f}")

# 역전파
f_val.backward()
print(f"∇f (autograd) = {x_val.grad}")

# 공식으로 계산
grad_formula = (A_torch + A_torch.T) @ x_val.detach()
print(f"∇f (공식)      = {grad_formula}")
print(f"차이: {torch.norm(x_val.grad - grad_formula).item():.2e}")
```

### 1.4 분자 레이아웃 vs 분모 레이아웃

행렬 미분에는 두 가지 표기법이 있습니다:

- **분자 레이아웃** (Numerator layout): $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$의 $(i,j)$ 원소가 $\frac{\partial y_i}{\partial x_j}$
- **분모 레이아웃** (Denominator layout): $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$의 $(i,j)$ 원소가 $\frac{\partial y_j}{\partial x_i}$

이 문서에서는 분자 레이아웃을 사용합니다.

**작동 예제 — 두 레이아웃에서 $\frac{d}{d\mathbf{x}}(A\mathbf{x})$**

$A \in \mathbb{R}^{m \times n}$이고 $\mathbf{y} = A\mathbf{x}$라 하면, $y_i = \sum_k A_{ik} x_k$입니다.

| 레이아웃 | 결과 | 크기 |
|----------|------|------|
| **분자** | $\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = A$ | $m \times n$ |
| **분모** | $\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = A^T$ | $n \times m$ |

두 레이아웃은 서로 전치 관계입니다. **PyTorch와 NumPy는 야코비안에 대해 분자 레이아웃을 따릅니다**: `torch.autograd.functional.jacobian`은 $(i,j)$ 원소가 $\partial y_i / \partial x_j$인 텐서를 반환하며, 분자 표기법과 일치합니다. 논문을 읽을 때는 저자가 어떤 레이아웃을 사용하는지 확인한 후 그래디언트 공식을 적용하세요.

## 2. 벡터-벡터 미분: 야코비안 (Jacobian)

### 2.1 야코비안 행렬의 정의

벡터 함수 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$에 대해 야코비안 행렬은:

$$J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$$

크기는 $m \times n$입니다.

### 2.2 체인 룰 with 야코비안

$\mathbf{z} = \mathbf{g}(\mathbf{f}(\mathbf{x}))$일 때:

$$\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{y}} \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$$

여기서 $\mathbf{y} = \mathbf{f}(\mathbf{x})$이고, 우변은 야코비안 행렬의 곱입니다.

### 2.3 야코비안 계산 예제

```python
import torch

# 함수 정의: f: R^2 -> R^3
def vector_function(x):
    """
    f([x1, x2]) = [x1^2 + x2,
                   x1 * x2,
                   sin(x1) + cos(x2)]
    """
    return torch.stack([
        x[0]**2 + x[1],
        x[0] * x[1],
        torch.sin(x[0]) + torch.cos(x[1])
    ])

x = torch.tensor([1.0, 2.0], requires_grad=True)

# PyTorch의 야코비안 계산
from torch.autograd.functional import jacobian

J = jacobian(vector_function, x)
print("야코비안 행렬 (3x2):")
print(J)

# 수동 계산으로 검증
x1, x2 = x[0].item(), x[1].item()
J_manual = torch.tensor([
    [2*x1, 1],
    [x2, x1],
    [np.cos(x1), -np.sin(x2)]
])
print("\n수동 계산:")
print(J_manual)
print(f"\n차이: {torch.norm(J - J_manual).item():.2e}")
```

### 2.4 체인 룰 실습

```python
# 합성 함수의 야코비안: h(x) = g(f(x))
def f(x):
    """f: R^2 -> R^2"""
    return torch.stack([x[0]**2, x[0] + x[1]])

def g(y):
    """g: R^2 -> R^2"""
    return torch.stack([y[0] * y[1], y[0] - y[1]])

def h(x):
    """h = g ∘ f"""
    return g(f(x))

x = torch.tensor([1.0, 2.0])

# 방법 1: 직접 계산
J_h = jacobian(h, x)
print("J_h (직접):")
print(J_h)

# 방법 2: 체인 룰
J_f = jacobian(f, x)
y = f(x)
J_g = jacobian(g, y)
J_chain = J_g @ J_f
print("\nJ_g @ J_f (체인 룰):")
print(J_chain)

print(f"\n차이: {torch.norm(J_h - J_chain).item():.2e}")
```

## 3. 헤시안 행렬 (Hessian Matrix)

### 3.1 헤시안의 정의

스칼라 함수 $f: \mathbb{R}^n \to \mathbb{R}$의 헤시안 행렬은 2차 편미분으로 구성됩니다:

$$H = \nabla^2 f = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix}$$

슈바르츠 정리에 의해 $f$가 클래스 $C^2$이면 $H$는 대칭입니다.

### 3.2 헤시안의 성질과 최적화

- **양정치 (Positive Definite)**: 모든 고유값 > 0 → 극솟값
- **음정치 (Negative Definite)**: 모든 고유값 < 0 → 극댓값
- **부정부호 (Indefinite)**: 양수/음수 고유값 혼재 → 안장점

### 3.3 뉴턴 방법에서의 역할

뉴턴 방법의 업데이트 규칙:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - H^{-1}(\mathbf{x}_k) \nabla f(\mathbf{x}_k)$$

헤시안의 역행렬을 사용하여 2차 정보를 활용합니다.

### 3.4 헤시안 계산 예제

```python
import torch
import numpy as np

# 함수 정의: f(x, y) = x^2 + xy + 2y^2
def f(x):
    return x[0]**2 + x[0]*x[1] + 2*x[1]**2

x = torch.tensor([1.0, 2.0], requires_grad=True)

# 그래디언트 계산
y = f(x)
grad = torch.autograd.grad(y, x, create_graph=True)[0]
print("∇f =", grad)

# 헤시안 계산 (각 그래디언트 성분을 다시 미분)
hessian = torch.zeros(2, 2)
for i in range(2):
    hessian[i] = torch.autograd.grad(grad[i], x, retain_graph=True)[0]

print("\n헤시안 행렬:")
print(hessian)

# 수동 계산: H = [[2, 1], [1, 4]]
H_manual = torch.tensor([[2.0, 1.0], [1.0, 4.0]])
print("\n수동 계산:")
print(H_manual)

# 고유값으로 정부호 판정
eigenvalues = torch.linalg.eigvalsh(hessian)
print(f"\n고유값: {eigenvalues}")
print("양정치 (극솟값):", torch.all(eigenvalues > 0).item())
```

### 3.5 헤시안과 볼록성

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 볼록 함수: f(x, y) = x^2 + 2y^2
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z_convex = X**2 + 2*Y**2

# 안장점 함수: f(x, y) = x^2 - y^2
Z_saddle = X**2 - Y**2

fig = plt.figure(figsize=(14, 6))

# 볼록 함수
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z_convex, cmap='viridis', alpha=0.8)
ax1.set_title('볼록 함수: $f(x,y) = x^2 + 2y^2$\n헤시안 양정치', fontsize=12)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')

# 안장점 함수
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z_saddle, cmap='plasma', alpha=0.8)
ax2.set_title('안장점: $f(x,y) = x^2 - y^2$\n헤시안 부정부호', fontsize=12)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f(x,y)')

plt.tight_layout()
plt.savefig('hessian_surfaces.png', dpi=150)
plt.close()

print("헤시안과 함수 형태 시각화 저장: hessian_surfaces.png")
```

## 4. 행렬 미분 항등식

### 4.1 주요 항등식 목록

| 함수 | 미분 결과 |
|------|-----------|
| $\mathbf{a}^T \mathbf{x}$ | $\mathbf{a}$ |
| $\mathbf{x}^T \mathbf{x}$ | $2\mathbf{x}$ |
| $\mathbf{x}^T A \mathbf{x}$ | $(A + A^T)\mathbf{x}$ |
| $\mathbf{a}^T X \mathbf{b}$ | $\mathbf{a}\mathbf{b}^T$ |
| $\text{tr}(AB)$ | $B^T$ (w.r.t. $A$) |
| $\log \|A\|$ | $A^{-T}$ (역행렬의 전치) |
| $\mathbf{x}^T A^{-1} \mathbf{x}$ | $-A^{-1}\mathbf{x}\mathbf{x}^T A^{-1}$ (w.r.t. $A$) |

### 4.2 항등식 유도: $\mathbf{x}^T A \mathbf{x}$

인덱스 표기법을 사용한 유도:

$$f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} = \sum_{i,j} x_i A_{ij} x_j$$

$x_k$로 미분:

$$\frac{\partial f}{\partial x_k} = \sum_j A_{kj} x_j + \sum_i x_i A_{ik} = (A\mathbf{x})_k + (A^T\mathbf{x})_k$$

따라서:

$$\nabla f = (A + A^T)\mathbf{x}$$

### 4.3 항등식 유도: $\text{tr}(AB)$

트레이스의 성질 $\text{tr}(AB) = \sum_{ij} A_{ij} B_{ji}$를 사용:

$$\frac{\partial}{\partial A_{kl}} \text{tr}(AB) = \frac{\partial}{\partial A_{kl}} \sum_{ij} A_{ij} B_{ji} = B_{lk}$$

따라서:

$$\frac{\partial \text{tr}(AB)}{\partial A} = B^T$$

### 4.4 항등식 검증 코드

```python
import torch

# 항등식 1: ∂(x^T a)/∂x = a
x = torch.randn(5, requires_grad=True)
a = torch.randn(5)
f = x @ a
f.backward()
print("항등식 1: ∂(x^T a)/∂x = a")
print(f"autograd: {x.grad}")
print(f"공식:     {a}")
print(f"차이: {torch.norm(x.grad - a).item():.2e}\n")

# 항등식 2: ∂(x^T A x)/∂x = (A + A^T)x
x = torch.randn(5, requires_grad=True)
A = torch.randn(5, 5)
f = x @ A @ x
f.backward()
print("항등식 2: ∂(x^T A x)/∂x = (A + A^T)x")
print(f"autograd: {x.grad}")
expected = (A + A.T) @ x.detach()
print(f"공식:     {expected}")
print(f"차이: {torch.norm(x.grad - expected).item():.2e}\n")

# 항등식 3: ∂tr(AB)/∂A = B^T
A = torch.randn(4, 4, requires_grad=True)
B = torch.randn(4, 4)
f = torch.trace(A @ B)
f.backward()
print("항등식 3: ∂tr(AB)/∂A = B^T")
print(f"autograd:\n{A.grad}")
print(f"공식:\n{B.T}")
print(f"차이: {torch.norm(A.grad - B.T).item():.2e}")
```

## 5. ML에서의 행렬 미분 응용

### 5.1 MSE 손실의 그래디언트 유도

회귀 문제에서 손실 함수:

$$L(\mathbf{w}) = \frac{1}{2n} \|\mathbf{y} - X\mathbf{w}\|^2$$

그래디언트:

$$\nabla_\mathbf{w} L = -\frac{1}{n} X^T (\mathbf{y} - X\mathbf{w})$$

유도 과정:

$$\nabla_\mathbf{w} L = \nabla_\mathbf{w} \frac{1}{2n}(\mathbf{y} - X\mathbf{w})^T(\mathbf{y} - X\mathbf{w})$$

$\mathbf{r} = \mathbf{y} - X\mathbf{w}$로 놓으면:

$$\nabla_\mathbf{w} L = \frac{1}{n} \nabla_\mathbf{w} \mathbf{r}^T \mathbf{r} = \frac{1}{n} \cdot 2 \mathbf{r}^T \nabla_\mathbf{w} \mathbf{r} = -\frac{1}{n} X^T \mathbf{r}$$

### 5.2 MSE 그래디언트 구현 및 검증

```python
import torch
import torch.nn as nn

# 데이터 생성
n, d = 100, 10
X = torch.randn(n, d)
y = torch.randn(n)
w = torch.randn(d, requires_grad=True)

# 방법 1: PyTorch autograd
pred = X @ w
loss = 0.5 * torch.mean((y - pred)**2)
loss.backward()
grad_autograd = w.grad.clone()

# 방법 2: 수동 유도 공식
residual = y - X @ w.detach()
grad_formula = -X.T @ residual / n

print("MSE 그래디언트 비교:")
print(f"autograd: {grad_autograd[:5]}")
print(f"공식:     {grad_formula[:5]}")
print(f"차이: {torch.norm(grad_autograd - grad_formula).item():.2e}")
```

### 5.3 소프트맥스 교차 엔트로피 그래디언트

소프트맥스 함수:

$$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

교차 엔트로피 손실:

$$L = -\sum_i y_i \log \sigma(\mathbf{z})_i$$

그래디언트 (원-핫 레이블 $\mathbf{y}$에 대해):

$$\frac{\partial L}{\partial \mathbf{z}} = \sigma(\mathbf{z}) - \mathbf{y}$$

이 간결한 형태는 소프트맥스의 야코비안 계산에서 유도됩니다.

### 5.4 소프트맥스 그래디언트 검증

```python
import torch
import torch.nn.functional as F

# 로짓과 타겟
logits = torch.randn(5, requires_grad=True)
target_class = 2  # 클래스 2가 정답

# 방법 1: PyTorch autograd
loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_class]))
loss.backward()
grad_autograd = logits.grad.clone()

# 방법 2: 수동 계산
probs = F.softmax(logits.detach(), dim=0)
y_onehot = torch.zeros(5)
y_onehot[target_class] = 1.0
grad_formula = probs - y_onehot

print("소프트맥스 교차 엔트로피 그래디언트:")
print(f"autograd: {grad_autograd}")
print(f"공식:     {grad_formula}")
print(f"차이: {torch.norm(grad_autograd - grad_formula).item():.2e}")
```

### 5.5 역전파: 야코비안의 연쇄 곱

신경망에서 역전파는 체인 룰의 반복 적용입니다:

$$\frac{\partial L}{\partial \mathbf{w}_1} = \frac{\partial L}{\partial \mathbf{z}_L} \frac{\partial \mathbf{z}_L}{\partial \mathbf{z}_{L-1}} \cdots \frac{\partial \mathbf{z}_2}{\partial \mathbf{z}_1} \frac{\partial \mathbf{z}_1}{\partial \mathbf{w}_1}$$

각 항은 야코비안이고, 오른쪽에서 왼쪽으로 계산 (리버스 모드).

### 5.6 선형 레이어의 그래디언트 유도

선형 레이어: $\mathbf{z} = W\mathbf{x} + \mathbf{b}$

손실 $L$에 대한 그래디언트:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \mathbf{z}} \mathbf{x}^T$$

$$\frac{\partial L}{\partial \mathbf{b}} = \frac{\partial L}{\partial \mathbf{z}}$$

$$\frac{\partial L}{\partial \mathbf{x}} = W^T \frac{\partial L}{\partial \mathbf{z}}$$

```python
# 선형 레이어 그래디언트 수동 구현
class LinearLayer:
    def __init__(self, in_dim, out_dim):
        self.W = torch.randn(out_dim, in_dim, requires_grad=False)
        self.b = torch.randn(out_dim, requires_grad=False)
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return self.W @ x + self.b

    def backward(self, dL_dz):
        """dL_dz: 손실에 대한 출력의 그래디언트"""
        self.dW = torch.outer(dL_dz, self.x)  # (out_dim, in_dim)
        self.db = dL_dz  # (out_dim,)
        dL_dx = self.W.T @ dL_dz  # (in_dim,)
        return dL_dx

# 테스트
layer = LinearLayer(5, 3)
x = torch.randn(5)
z = layer.forward(x)
dL_dz = torch.randn(3)  # 가짜 그래디언트
dL_dx = layer.backward(dL_dz)

print("선형 레이어 역전파:")
print(f"dW 형태: {layer.dW.shape}")
print(f"db 형태: {layer.db.shape}")
print(f"dx 형태: {dL_dx.shape}")

# PyTorch로 검증
W_torch = layer.W.clone().requires_grad_(True)
b_torch = layer.b.clone().requires_grad_(True)
x_torch = x.clone().requires_grad_(True)

z_torch = W_torch @ x_torch + b_torch
z_torch.backward(dL_dz)

print(f"\ndW 차이: {torch.norm(layer.dW - W_torch.grad).item():.2e}")
print(f"db 차이: {torch.norm(layer.db - b_torch.grad).item():.2e}")
print(f"dx 차이: {torch.norm(dL_dx - x_torch.grad).item():.2e}")
```

## 6. 자동 미분 (Automatic Differentiation)

### 6.1 포워드 모드 vs 리버스 모드

**포워드 모드 (Forward Mode)**:
- 입력에서 출력 방향으로 미분 전파
- $n$개 입력, 1개 출력일 때 효율적
- 방향 도함수 계산에 유용

**리버스 모드 (Reverse Mode)**:
- 출력에서 입력 방향으로 미분 전파 (역전파)
- 1개 출력, $n$개 입력일 때 효율적
- 딥러닝에서 사용 (손실 함수는 스칼라)

### 6.2 계산 그래프

계산 그래프는 연산을 노드로, 데이터 흐름을 엣지로 표현합니다.

```python
# 계산 그래프 예시: f(x, y) = (x + y) * (x - y)
import torch

x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

# 중간 변수 저장
a = x + y  # a = 5
b = x - y  # b = 1
f = a * b  # f = 5

print("계산 그래프:")
print(f"x={x.item()}, y={y.item()}")
print(f"a = x + y = {a.item()}")
print(f"b = x - y = {b.item()}")
print(f"f = a * b = {f.item()}")

# 역전파
f.backward()
print(f"\n∂f/∂x = {x.grad.item()}")
print(f"∂f/∂y = {y.grad.item()}")

# 수동 계산 검증
# f = (x+y)(x-y) = x^2 - y^2
# ∂f/∂x = 2x = 6
# ∂f/∂y = -2y = -4
print(f"\n수동 계산: ∂f/∂x = 2x = {2*x.item()}")
print(f"수동 계산: ∂f/∂y = -2y = {-2*y.item()}")
```

### 6.3 PyTorch Autograd 내부 동작

```python
# 계산 그래프 시각화 (간단한 예)
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
w = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = torch.tensor([0.5, 1.0], requires_grad=True)

# 순전파
z = w @ x + b  # 선형 변환
a = torch.relu(z)  # 활성화
loss = a.sum()  # 손실

print("계산 그래프 추적:")
print(f"grad_fn of z: {z.grad_fn}")
print(f"grad_fn of a: {a.grad_fn}")
print(f"grad_fn of loss: {loss.grad_fn}")

# 역전파
loss.backward()

print("\n그래디언트:")
print(f"∂L/∂x: {x.grad}")
print(f"∂L/∂w:\n{w.grad}")
print(f"∂L/∂b: {b.grad}")
```

### 6.4 고차 미분

```python
# 2차 미분 (헤시안 대각선)
x = torch.tensor(2.0, requires_grad=True)
y = x**4

# 1차 미분
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"f(x) = x^4, f'(x) = 4x^3")
print(f"f'(2) = {dy_dx.item()} (예상: {4*2**3})")

# 2차 미분
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
print(f"f''(x) = 12x^2")
print(f"f''(2) = {d2y_dx2.item()} (예상: {12*2**2})")
```

### 6.5 자동 미분의 한계와 수동 구현

자동 미분은 편리하지만 때로는 수동 구현이 필요합니다:

- 메모리 효율성 (그래디언트 체크포인팅)
- 커스텀 역전파 로직
- 수치 안정성 개선

```python
# 커스텀 autograd 함수
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x < 0] = 0
        return grad_input

# 사용
x = torch.randn(5, requires_grad=True)
y = MyReLU.apply(x)
loss = y.sum()
loss.backward()

print("커스텀 ReLU 그래디언트:")
print(f"x: {x.detach()}")
print(f"y: {y.detach()}")
print(f"∂L/∂x: {x.grad}")
```

## 연습 문제

### 문제 1: 행렬 미분 항등식 유도
$\mathbf{x} \in \mathbb{R}^n$, $A \in \mathbb{R}^{n \times n}$일 때, 다음을 증명하시오:

$$\frac{\partial}{\partial \mathbf{x}} (\mathbf{x}^T A \mathbf{x}) = (A + A^T)\mathbf{x}$$

인덱스 표기법을 사용하여 단계별로 유도하고, PyTorch로 검증하는 코드를 작성하시오.

### 문제 2: 로지스틱 회귀 그래디언트
로지스틱 회귀의 손실 함수는:

$$L(\mathbf{w}) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log \sigma(\mathbf{w}^T \mathbf{x}_i) + (1-y_i) \log(1-\sigma(\mathbf{w}^T \mathbf{x}_i)) \right]$$

여기서 $\sigma(z) = 1/(1+e^{-z})$. 그래디언트 $\nabla_\mathbf{w} L$을 유도하시오. 결과가 다음과 같음을 보이시오:

$$\nabla_\mathbf{w} L = \frac{1}{n} X^T (\boldsymbol{\sigma} - \mathbf{y})$$

여기서 $\boldsymbol{\sigma} = [\sigma(\mathbf{w}^T \mathbf{x}_1), \ldots, \sigma(\mathbf{w}^T \mathbf{x}_n)]^T$.

### 문제 3: 배치 정규화 그래디언트
배치 정규화는 다음과 같이 정의됩니다:

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

여기서 $\mu = \frac{1}{n}\sum_i x_i$, $\sigma^2 = \frac{1}{n}\sum_i (x_i - \mu)^2$.

손실 $L$에 대한 $x_i$의 그래디언트를 유도하시오. PyTorch의 `BatchNorm1d`와 비교하여 검증하시오.

### 문제 4: 소프트맥스 야코비안
소프트맥스 함수 $\sigma(\mathbf{z})_i = e^{z_i} / \sum_j e^{z_j}$의 야코비안을 계산하시오.

$$\frac{\partial \sigma_i}{\partial z_j} = ?$$

힌트: $i=j$인 경우와 $i \neq j$인 경우를 나누어 계산하시오. 결과가 다음과 같음을 보이시오:

$$\frac{\partial \sigma_i}{\partial z_j} = \sigma_i(\delta_{ij} - \sigma_j)$$

### 문제 5: L2 정규화 그래디언트
릿지 회귀의 손실 함수:

$$L(\mathbf{w}) = \frac{1}{2n}\|\mathbf{y} - X\mathbf{w}\|^2 + \frac{\lambda}{2}\|\mathbf{w}\|^2$$

그래디언트를 유도하고, 정규 방정식 (Normal Equation)을 구하시오. PyTorch로 경사 하강법과 해석해를 비교하시오.

## 참고 자료

### 온라인 자료
- [Matrix Calculus for Deep Learning](https://explained.ai/matrix-calculus/) - 상세한 행렬 미분 튜토리얼
- [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) - 행렬 미분 공식집
- [PyTorch Autograd Documentation](https://pytorch.org/docs/stable/autograd.html)

### 교재
- Magnus & Neudecker, *Matrix Differential Calculus with Applications in Statistics and Econometrics*
- Goodfellow et al., *Deep Learning*, Chapter 6 (Numerical Computation)
- Boyd & Vandenberghe, *Convex Optimization*, Appendix A

### 논문
- Griewank & Walther, *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation* (2008)
- Baydin et al., *Automatic Differentiation in Machine Learning: a Survey* (JMLR 2018)

---

**다음 레슨**: [04. 노름과 거리 메트릭](04_Norms_and_Distances.md)
