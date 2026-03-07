# 20강: 고급 주제 (Advanced Topics)

[이전: 19강](./19_Randomized_Linear_Algebra.md) | [개요](./00_Overview.md)

---

## 학습 목표

- 고유 분해와 Pade 근사를 사용하여 행렬 함수(지수, 로그, 제곱근)를 계산하고 적용할 수 있다
- Kronecker 곱의 성질과 행렬 방정식의 벡터화에 대한 응용을 이해할 수 있다
- 블록 행렬을 다루고 블록 구조를 활용하여 효율적으로 계산할 수 있다
- 행렬 미적분을 적용하여 스칼라, 벡터, 행렬 표현식을 행렬에 대해 미분할 수 있다
- Moore-Penrose 유사역행렬을 계산하고 과대결정 및 과소결정 시스템에 사용할 수 있다

---

## 1. 행렬 함수

### 1.1 행렬 함수의 정의

대각화 가능한 행렬 $A = V\Lambda V^{-1}$에 대해, 함수 $f(A)$는 다음과 같이 정의됩니다:

$$f(A) = V f(\Lambda) V^{-1} = V \begin{bmatrix} f(\lambda_1) & & \\ & \ddots & \\ & & f(\lambda_n) \end{bmatrix} V^{-1}$$

이는 모든 스칼라 함수를 행렬로 확장합니다: 지수, 로그, 제곱근, 삼각함수 등.

```python
import numpy as np
from scipy.linalg import expm, logm, sqrtm, funm
import matplotlib.pyplot as plt

# Matrix exponential via eigendecomposition
A = np.array([[1, 1],
              [0, 2]])

eigenvalues, V = np.linalg.eig(A)
exp_Lambda = np.diag(np.exp(eigenvalues))
exp_A_eigen = V @ exp_Lambda @ np.linalg.inv(V)

# Compare with scipy
exp_A_scipy = expm(A)

print(f"Matrix A:\n{A}")
print(f"\nEigenvalues: {eigenvalues}")
print(f"\nexp(A) via eigendecomposition:\n{np.round(exp_A_eigen.real, 6)}")
print(f"exp(A) via scipy.linalg.expm:\n{np.round(exp_A_scipy, 6)}")
print(f"Match: {np.allclose(exp_A_eigen, exp_A_scipy)}")
```

### 1.2 행렬 지수

행렬 지수 $e^A$는 거듭제곱 급수로 정의됩니다:

$$e^A = \sum_{k=0}^{\infty} \frac{A^k}{k!} = I + A + \frac{A^2}{2!} + \frac{A^3}{3!} + \cdots$$

이는 선형 상미분방정식 시스템을 푸는 데 기본적입니다: $\dot{\mathbf{x}} = A\mathbf{x}$의 해는 $\mathbf{x}(t) = e^{At}\mathbf{x}(0)$입니다.

```python
def matrix_exp_series(A, n_terms=20):
    """Compute exp(A) using the Taylor series (for illustration only)."""
    result = np.eye(A.shape[0])
    term = np.eye(A.shape[0])
    for k in range(1, n_terms + 1):
        term = term @ A / k
        result = result + term
    return result

# Compare methods
A = np.array([[0, -1],
              [1, 0]])  # Rotation generator

print("Methods for computing exp(A):")
methods = {
    "Taylor (20 terms)": matrix_exp_series(A, 20),
    "Taylor (50 terms)": matrix_exp_series(A, 50),
    "scipy expm (Pade)": expm(A),
    "Eigendecomposition": None,
}

# Eigendecomposition method
eigenvalues, V = np.linalg.eig(A)
methods["Eigendecomposition"] = (V @ np.diag(np.exp(eigenvalues)) @
                                  np.linalg.inv(V)).real

for name, result in methods.items():
    print(f"  {name:25s}: {np.round(result, 8)}")

# The exact answer for this skew-symmetric matrix is a rotation by 1 radian
theta = 1.0
R_exact = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
print(f"\n  Exact (rotation by 1 rad): {np.round(R_exact, 8)}")
```

### 1.3 ODE를 위한 행렬 지수

```python
# Solve dx/dt = Ax using matrix exponential
A = np.array([[-0.5, 1],
              [-1, -0.5]])  # Damped oscillator

x0 = np.array([1.0, 0.0])

# Solve at multiple time points
t_values = np.linspace(0, 10, 200)
trajectory = np.zeros((len(t_values), 2))

for i, t in enumerate(t_values):
    trajectory[i] = expm(A * t) @ x0

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(t_values, trajectory[:, 0], label='x1(t)')
ax1.plot(t_values, trajectory[:, 1], label='x2(t)')
ax1.set_xlabel('t')
ax1.set_ylabel('x(t)')
ax1.set_title('Solution of dx/dt = Ax')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(trajectory[:, 0], trajectory[:, 1])
ax2.plot(x0[0], x0[1], 'go', markersize=10, label='Start')
ax2.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('Phase Portrait')
ax2.legend()
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 1.4 행렬 로그와 행렬 제곱근

행렬 로그는 행렬 지수의 역입니다: $B = e^A$이면 $A = \log(B)$입니다.

행렬 제곱근은 $S^2 = A$ (즉, $S = A^{1/2}$)를 만족합니다.

```python
# Matrix logarithm
A = np.array([[2, 1],
              [0, 3]])

# Compute exp(A) then take log to recover A
exp_A = expm(A)
log_exp_A = logm(exp_A)
print(f"A:\n{A}")
print(f"log(exp(A)):\n{np.round(log_exp_A.real, 6)}")
print(f"Recovered A? {np.allclose(A, log_exp_A.real)}")

# Matrix square root
A_pd = np.array([[4, 2],
                 [2, 3]])

sqrt_A = sqrtm(A_pd)
print(f"\nA:\n{A_pd}")
print(f"sqrt(A):\n{np.round(sqrt_A.real, 6)}")
print(f"sqrt(A) @ sqrt(A) = A? {np.allclose(sqrt_A @ sqrt_A, A_pd)}")

# For SPD matrices, the square root is also SPD
eigenvalues = np.linalg.eigvalsh(sqrt_A.real)
print(f"sqrt(A) eigenvalues: {eigenvalues}")
print(f"sqrt(A) is PD? {np.all(eigenvalues > 0)}")
```

### 1.5 임의 함수의 적용

```python
# Apply any scalar function to a matrix
# f(A) = V diag(f(lambda_1), ..., f(lambda_n)) V^(-1)

def matrix_function(A, f):
    """Apply scalar function f to matrix A via eigendecomposition."""
    eigenvalues, V = np.linalg.eig(A)
    f_eigenvalues = np.array([f(lam) for lam in eigenvalues])
    return (V @ np.diag(f_eigenvalues) @ np.linalg.inv(V)).real

A = np.array([[4, 1],
              [1, 3]])

# Matrix sine
sin_A = matrix_function(A, np.sin)
sin_A_scipy = funm(A, np.sin)
print(f"sin(A) via eigen:\n{np.round(sin_A, 6)}")
print(f"sin(A) via scipy:\n{np.round(sin_A_scipy, 6)}")
print(f"Match: {np.allclose(sin_A, sin_A_scipy)}")

# Matrix sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

sigmoid_A = matrix_function(A, sigmoid)
print(f"\nsigmoid(A):\n{np.round(sigmoid_A, 6)}")
```

---

## 2. Kronecker 곱

### 2.1 정의

$m \times n$ 행렬 $A$와 $p \times q$ 행렬 $B$의 **Kronecker 곱** $A \otimes B$는 $mp \times nq$ 블록 행렬입니다:

$$A \otimes B = \begin{bmatrix} a_{11}B & a_{12}B & \cdots & a_{1n}B \\ a_{21}B & a_{22}B & \cdots & a_{2n}B \\ \vdots & & \ddots & \vdots \\ a_{m1}B & a_{m2}B & \cdots & a_{mn}B \end{bmatrix}$$

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[0, 5],
              [6, 7]])

# Kronecker product
AB = np.kron(A, B)
print(f"A ({A.shape}) otimes B ({B.shape}) = ({AB.shape})")
print(f"A x B:\n{AB}")
```

### 2.2 성질

Kronecker 곱은 많은 유용한 대수적 성질을 가지고 있습니다:

```python
# Key properties of Kronecker products
A = np.random.randn(2, 3)
B = np.random.randn(4, 5)
C = np.random.randn(3, 2)
D = np.random.randn(5, 4)

# Property 1: (A x B)(C x D) = (AC) x (BD)
# This is the "mixed product" property
lhs = np.kron(A, B) @ np.kron(C, D)
rhs = np.kron(A @ C, B @ D)
print(f"Mixed product: {np.allclose(lhs, rhs)}")

# Property 2: (A x B)^T = A^T x B^T
lhs = np.kron(A, B).T
rhs = np.kron(A.T, B.T)
print(f"Transpose: {np.allclose(lhs, rhs)}")

# Property 3: (A x B)^(-1) = A^(-1) x B^(-1) (for square invertible matrices)
A_sq = np.random.randn(3, 3)
B_sq = np.random.randn(2, 2)
lhs = np.linalg.inv(np.kron(A_sq, B_sq))
rhs = np.kron(np.linalg.inv(A_sq), np.linalg.inv(B_sq))
print(f"Inverse: {np.allclose(lhs, rhs)}")

# Property 4: Eigenvalues of A x B are products of eigenvalues
eig_A = np.linalg.eigvals(A_sq)
eig_B = np.linalg.eigvals(B_sq)
eig_AB = np.sort(np.linalg.eigvals(np.kron(A_sq, B_sq)))
eig_products = np.sort(np.array([a * b for a in eig_A for b in eig_B]))
print(f"Eigenvalue products: {np.allclose(eig_AB, eig_products)}")

# Property 5: tr(A x B) = tr(A) * tr(B)
print(f"Trace product: {np.isclose(np.trace(np.kron(A_sq, B_sq)), np.trace(A_sq) * np.trace(B_sq))}")
```

### 2.3 벡터화와 Vec 연산자

**vec** 연산자는 행렬의 모든 열을 하나의 벡터로 쌓습니다. Kronecker 곱은 다음을 통해 행렬 방정식을 벡터 방정식으로 연결합니다:

$$\text{vec}(AXB) = (B^T \otimes A) \text{vec}(X)$$

이 항등식은 **Sylvester 방정식** $AX + XB = C$ 같은 행렬 방정식을 표준 선형 시스템으로 변환합니다.

```python
def vec(X):
    """Vectorize a matrix (column-major order)."""
    return X.flatten('F')

def unvec(x, shape):
    """Reshape vector back to matrix (column-major)."""
    return x.reshape(shape, order='F')

# Verify: vec(AXB) = (B^T kron A) vec(X)
A = np.random.randn(3, 3)
X = np.random.randn(3, 4)
B = np.random.randn(4, 4)

lhs = vec(A @ X @ B)
rhs = np.kron(B.T, A) @ vec(X)
print(f"vec(AXB) = (B^T kron A) vec(X): {np.allclose(lhs, rhs)}")

# Solve the Sylvester equation: AX + XB = C
from scipy.linalg import solve_sylvester

A_syl = np.random.randn(3, 3)
B_syl = np.random.randn(4, 4)
C_syl = np.random.randn(3, 4)

# Method 1: scipy (direct Bartels-Stewart algorithm)
X_scipy = solve_sylvester(A_syl, B_syl, C_syl)

# Method 2: via Kronecker product and vectorization
# AX + XB = C  =>  (I_n kron A + B^T kron I_m) vec(X) = vec(C)
m, n = A_syl.shape[0], B_syl.shape[0]
M = np.kron(np.eye(n), A_syl) + np.kron(B_syl.T, np.eye(m))
X_kron = unvec(np.linalg.solve(M, vec(C_syl)), (m, n))

print(f"\nSylvester equation AX + XB = C:")
print(f"  scipy solution matches Kronecker: {np.allclose(X_scipy, X_kron)}")
print(f"  Residual: {np.linalg.norm(A_syl @ X_scipy + X_scipy @ B_syl - C_syl):.2e}")
```

---

## 3. 블록 행렬

### 3.1 블록 행렬 연산

**블록 행렬**은 행렬을 부분행렬(블록)로 분할합니다. 많은 연산이 블록 단위로 수행될 수 있어 효율적인 알고리즘이 가능합니다.

$$M = \begin{bmatrix} A & B \\ C & D \end{bmatrix}$$

```python
# Block matrix construction and operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5], [6]])
C = np.array([[7, 8]])
D = np.array([[9]])

# Construct block matrix
M = np.block([[A, B],
              [C, D]])
print(f"Block matrix M:\n{M}")

# Block multiplication
X = np.block([[A, B],
              [C, D]])
Y = np.block([[D, C],
              [B.T, A.T[:1, :]]])  # Need compatible sizes

# Block diagonal matrix
from scipy.linalg import block_diag
blocks = [np.ones((2, 2)), 2 * np.eye(3), np.array([[5]])]
BD = block_diag(*blocks)
print(f"\nBlock diagonal:\n{BD}")
print(f"Shape: {BD.shape}")
```

### 3.2 블록 행렬 역행렬

블록 행렬 $M = \begin{bmatrix} A & B \\ C & D \end{bmatrix}$에서 $A$와 $D$가 정방이고 가역일 때:

$$M^{-1} = \begin{bmatrix} A^{-1} + A^{-1}BS^{-1}CA^{-1} & -A^{-1}BS^{-1} \\ -S^{-1}CA^{-1} & S^{-1} \end{bmatrix}$$

여기서 $S = D - CA^{-1}B$는 $M$에서 $A$의 **Schur 보수**입니다.

```python
def block_inverse(A, B, C, D):
    """Invert a block matrix using the Schur complement."""
    A_inv = np.linalg.inv(A)
    S = D - C @ A_inv @ B  # Schur complement
    S_inv = np.linalg.inv(S)

    top_left = A_inv + A_inv @ B @ S_inv @ C @ A_inv
    top_right = -A_inv @ B @ S_inv
    bot_left = -S_inv @ C @ A_inv
    bot_right = S_inv

    return np.block([[top_left, top_right],
                     [bot_left, bot_right]])

# Test
A = np.array([[4, 1], [1, 3]])
B = np.array([[1, 0], [0, 1]])
C = np.array([[2, 1], [1, 2]])
D = np.array([[5, 1], [1, 4]])

M = np.block([[A, B], [C, D]])
M_inv_block = block_inverse(A, B, C, D)
M_inv_direct = np.linalg.inv(M)

print(f"Block inverse matches direct: {np.allclose(M_inv_block, M_inv_direct)}")
print(f"M @ M^(-1) = I: {np.allclose(M @ M_inv_block, np.eye(4))}")
```

### 3.3 Woodbury 항등식

**Woodbury 행렬 항등식**은 저랭크 수정이 가해질 때 역행렬을 효율적으로 업데이트합니다:

$$(A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}$$

이는 $A^{-1}$이 알려져 있거나 계산이 쉽고 $U, V$의 열 수가 적을 때 매우 유용합니다.

```python
def woodbury_inverse(A_inv, U, C, V):
    """Compute (A + UCV)^(-1) using the Woodbury identity."""
    # (A + UCV)^(-1) = A^(-1) - A^(-1)U (C^(-1) + V A^(-1) U)^(-1) V A^(-1)
    C_inv = np.linalg.inv(C)
    middle = np.linalg.inv(C_inv + V @ A_inv @ U)
    return A_inv - A_inv @ U @ middle @ V @ A_inv

# Example: updating a diagonal matrix with a low-rank correction
n = 100
k = 3  # Rank of update

A = 5 * np.eye(n)  # Easy to invert
A_inv = np.eye(n) / 5

U = np.random.randn(n, k)
C = np.eye(k)
V = np.random.randn(k, n)

# Method 1: Direct inverse (expensive for large n)
import time
start = time.time()
M = A + U @ C @ V
M_inv_direct = np.linalg.inv(M)
t_direct = time.time() - start

# Method 2: Woodbury (uses A^(-1) which is trivial for diagonal A)
start = time.time()
M_inv_woodbury = woodbury_inverse(A_inv, U, C, V)
t_woodbury = time.time() - start

print(f"Direct inverse:  {t_direct*1000:.2f} ms")
print(f"Woodbury:        {t_woodbury*1000:.2f} ms")
print(f"Match: {np.allclose(M_inv_direct, M_inv_woodbury)}")
```

---

## 4. 행렬 미적분

### 4.1 행렬의 스칼라 함수에 대한 미분

스칼라 함수 $f(X)$에서 $X$가 행렬일 때, 미분 $\frac{\partial f}{\partial X}$는 $X$와 같은 형상의 행렬입니다:

$$\left(\frac{\partial f}{\partial X}\right)_{ij} = \frac{\partial f}{\partial X_{ij}}$$

```python
def numerical_matrix_gradient(f, X, eps=1e-7):
    """Compute gradient of scalar function f w.r.t. matrix X."""
    grad = np.zeros_like(X)
    f0 = f(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_plus = X.copy()
            X_plus[i, j] += eps
            grad[i, j] = (f(X_plus) - f0) / eps
    return grad

# Example 1: f(X) = tr(X)
X = np.random.randn(3, 3)
f_trace = lambda X: np.trace(X)
grad_trace = numerical_matrix_gradient(f_trace, X)
print(f"d/dX tr(X) = I?")
print(f"Numerical gradient:\n{np.round(grad_trace, 4)}")
print(f"Expected (I):\n{np.eye(3)}")

# Example 2: f(X) = tr(AX)
A = np.random.randn(3, 3)
f_trAX = lambda X: np.trace(A @ X)
grad_trAX = numerical_matrix_gradient(f_trAX, X)
print(f"\nd/dX tr(AX) = A^T?")
print(f"Match: {np.allclose(grad_trAX, A.T)}")
```

### 4.2 주요 행렬 미분 공식

자주 필요한 행렬 미분의 참조표입니다:

| 함수 $f(X)$ | 그래디언트 $\frac{\partial f}{\partial X}$ |
|---|---|
| $\text{tr}(X)$ | $I$ |
| $\text{tr}(AX)$ | $A^T$ |
| $\text{tr}(X^TAX)$ | $(A + A^T)X$ |
| $\text{tr}(AXB)$ | $A^TB^T$ |
| $\mathbf{a}^TX\mathbf{b}$ | $\mathbf{a}\mathbf{b}^T$ |
| $\|X\mathbf{w} - \mathbf{y}\|^2$ | $2(X\mathbf{w} - \mathbf{y})\mathbf{w}^T$ ($X$에 대해) |
| $\log\det(X)$ | $X^{-T}$ |
| $\det(X)$ | $\det(X) X^{-T}$ |

```python
# Verify several matrix derivative formulas

X = np.random.randn(3, 3)
A = np.random.randn(3, 3)
w = np.random.randn(3)
b = np.random.randn(3)

# 1. d/dX tr(X^T A X) = (A + A^T) X
f1 = lambda X: np.trace(X.T @ A @ X)
grad_num = numerical_matrix_gradient(f1, X)
grad_exact = (A + A.T) @ X
print(f"d/dX tr(X^TAX) = (A+A^T)X: {np.allclose(grad_num, grad_exact, atol=1e-5)}")

# 2. d/dX a^T X b = a b^T
a = np.random.randn(3)
f2 = lambda X: a @ X @ b
grad_num = numerical_matrix_gradient(f2, X)
grad_exact = np.outer(a, b)
print(f"d/dX a^TXb = ab^T: {np.allclose(grad_num, grad_exact, atol=1e-5)}")

# 3. d/dX log det(X) = X^{-T}  (for X positive definite)
X_pd = X.T @ X + 3 * np.eye(3)  # Make PD
f3 = lambda X: np.log(np.linalg.det(X))
grad_num = numerical_matrix_gradient(f3, X_pd)
grad_exact = np.linalg.inv(X_pd).T
print(f"d/dX log det(X) = X^(-T): {np.allclose(grad_num, grad_exact, atol=1e-4)}")

# 4. Gradient of linear regression loss: ||Xw - y||^2 w.r.t. w
X_data = np.random.randn(10, 3)
y_data = np.random.randn(10)
f4 = lambda w: np.sum((X_data @ w - y_data)**2)
grad_num_w = np.zeros(3)
for j in range(3):
    w_plus = w.copy(); w_plus[j] += 1e-7
    grad_num_w[j] = (f4(w_plus) - f4(w)) / 1e-7
grad_exact_w = 2 * X_data.T @ (X_data @ w - y_data)
print(f"d/dw ||Xw-y||^2 = 2X^T(Xw-y): {np.allclose(grad_num_w, grad_exact_w, atol=1e-4)}")
```

### 4.3 행렬 표현식의 연쇄 법칙

합성 함수에 대해 연쇄 법칙은 자연스럽게 확장됩니다. $L = g(f(X))$에서 $f: \mathbb{R}^{m \times n} \to \mathbb{R}^{p \times q}$이고 $g: \mathbb{R}^{p \times q} \to \mathbb{R}$일 때:

$$\frac{\partial L}{\partial X_{ij}} = \sum_{k,l} \frac{\partial L}{\partial f_{kl}} \frac{\partial f_{kl}}{\partial X_{ij}}$$

```python
# Chain rule example: L = ||sigma(WX + b) - Y||^2
# where sigma is element-wise sigmoid

def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))

def sigmoid_grad(Z):
    s = sigmoid(Z)
    return s * (1 - s)

# Forward pass
np.random.seed(42)
n, d, m = 10, 3, 2
X = np.random.randn(d, n)     # Features (d x n)
W = np.random.randn(m, d)     # Weights
b = np.random.randn(m, 1)
Y = np.random.randn(m, n)     # Targets

Z = W @ X + b                 # Pre-activation (m x n)
H = sigmoid(Z)                # Activation (m x n)
L = np.sum((H - Y)**2)        # Loss (scalar)

# Backward pass (chain rule)
dL_dH = 2 * (H - Y)                    # (m x n)
dL_dZ = dL_dH * sigmoid_grad(Z)         # (m x n) element-wise
dL_dW = dL_dZ @ X.T                     # (m x d)
dL_db = dL_dZ.sum(axis=1, keepdims=True)  # (m x 1)

# Verify with numerical gradient
def loss_fn(W_flat):
    W_r = W_flat.reshape(m, d)
    Z = W_r @ X + b
    H = sigmoid(Z)
    return np.sum((H - Y)**2)

grad_num = np.zeros_like(W)
W_flat = W.flatten()
for i in range(len(W_flat)):
    W_plus = W_flat.copy(); W_plus[i] += 1e-5
    W_minus = W_flat.copy(); W_minus[i] -= 1e-5
    grad_num.flat[i] = (loss_fn(W_plus) - loss_fn(W_minus)) / (2e-5)

print(f"Analytical gradient matches numerical: "
      f"{np.allclose(dL_dW, grad_num, atol=1e-4)}")
print(f"Max difference: {np.max(np.abs(dL_dW - grad_num)):.2e}")
```

---

## 5. Moore-Penrose 유사역행렬

### 5.1 정의

행렬 $A$의 **Moore-Penrose 유사역행렬** $A^+$는 다음 네 가지 조건을 만족하는 유일한 행렬입니다:

1. $AA^+A = A$
2. $A^+AA^+ = A^+$
3. $(AA^+)^T = AA^+$
4. $(A^+A)^T = A^+A$

SVD $A = U\Sigma V^T$를 가진 행렬의 유사역행렬은 다음과 같습니다:

$$A^+ = V\Sigma^+ U^T$$

여기서 $\Sigma^+$는 0이 아닌 특이값을 역수로 변환합니다.

```python
# Pseudoinverse computation
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# Method 1: numpy
A_pinv_np = np.linalg.pinv(A)

# Method 2: via SVD
U, S, Vt = np.linalg.svd(A, full_matrices=False)
S_pinv = np.diag(1.0 / S)
A_pinv_svd = Vt.T @ S_pinv @ U.T

print(f"A shape: {A.shape}")
print(f"A+ shape: {A_pinv_np.shape}")
print(f"Methods match: {np.allclose(A_pinv_np, A_pinv_svd)}")

# Verify Moore-Penrose conditions
print(f"\nMoore-Penrose conditions:")
print(f"1. A A+ A = A: {np.allclose(A @ A_pinv_np @ A, A)}")
print(f"2. A+ A A+ = A+: {np.allclose(A_pinv_np @ A @ A_pinv_np, A_pinv_np)}")
print(f"3. (A A+)^T = A A+: {np.allclose((A @ A_pinv_np).T, A @ A_pinv_np)}")
print(f"4. (A+ A)^T = A+ A: {np.allclose((A_pinv_np @ A).T, A_pinv_np @ A)}")
```

### 5.2 응용: 과대결정 및 과소결정 시스템

유사역행렬은 비일관적이거나 과소결정된 시스템에 대한 "최적" 해를 제공합니다:

- **과대결정** ($m > n$): $x = A^+b$는 최소자승해 ($\|Ax - b\|$를 최소화)
- **과소결정** ($m < n$): $x = A^+b$는 최소 노름 해 (모든 해 중 $\|x\|$를 최소화)

```python
# Overdetermined system (least squares)
A_over = np.array([[1, 1],
                   [1, 2],
                   [1, 3],
                   [1, 4]])
b_over = np.array([2.1, 3.9, 6.2, 7.8])

x_pinv = np.linalg.pinv(A_over) @ b_over
x_lstsq = np.linalg.lstsq(A_over, b_over, rcond=None)[0]

print("Overdetermined system (4 equations, 2 unknowns):")
print(f"  Pseudoinverse solution: {x_pinv}")
print(f"  Least squares solution: {x_lstsq}")
print(f"  Match: {np.allclose(x_pinv, x_lstsq)}")
print(f"  Residual: {np.linalg.norm(A_over @ x_pinv - b_over):.4f}")

# Underdetermined system (minimum norm)
A_under = np.array([[1, 2, 3],
                    [4, 5, 6]])
b_under = np.array([14, 32])

x_pinv = np.linalg.pinv(A_under) @ b_under
print(f"\nUnderdetermined system (2 equations, 3 unknowns):")
print(f"  Pseudoinverse solution: {x_pinv}")
print(f"  Solution norm: {np.linalg.norm(x_pinv):.4f}")
print(f"  Satisfies Ax=b: {np.allclose(A_under @ x_pinv, b_under)}")

# Verify it is minimum norm: any other solution has larger norm
# General solution: x = x_pinv + null(A) @ z
null = np.linalg.svd(A_under)[2][2:]  # Null space
for _ in range(100):
    z = np.random.randn(null.shape[0])
    x_other = x_pinv + null.T @ z
    assert np.linalg.norm(x_other) >= np.linalg.norm(x_pinv) - 1e-10
print(f"  Verified: minimum norm among all solutions")
```

### 5.3 랭크 결핍 행렬의 유사역행렬

```python
# Rank-deficient matrix
A_rank_def = np.array([[1, 2, 3],
                       [2, 4, 6],
                       [3, 6, 9]])

print(f"Rank: {np.linalg.matrix_rank(A_rank_def)}")
print(f"Singular values: {np.linalg.svd(A_rank_def, compute_uv=False)}")

# Regular inverse does not exist
try:
    A_inv = np.linalg.inv(A_rank_def)
    print("Inverse computed (with warning)")
except np.linalg.LinAlgError:
    print("Inverse does not exist (singular matrix)")

# Pseudoinverse always exists
A_pinv = np.linalg.pinv(A_rank_def)
print(f"\nPseudoinverse:\n{np.round(A_pinv, 6)}")

# It satisfies the Moore-Penrose conditions
print(f"A A+ A = A: {np.allclose(A_rank_def @ A_pinv @ A_rank_def, A_rank_def)}")

# Solve Ax = b for a consistent system
b = np.array([6, 12, 18])  # b is in the column space of A
x_pinv = A_pinv @ b
print(f"\nSolution: {x_pinv}")
print(f"Residual: {np.linalg.norm(A_rank_def @ x_pinv - b):.2e}")
```

---

## 6. 추가 고급 주제

### 6.1 행렬 완성

부분적으로 관측된 행렬이 주어졌을 때, **행렬 완성**은 행렬이 저랭크라는 가정 하에 누락된 항목을 채웁니다:

```python
def matrix_completion_svt(M_observed, mask, rank=5, max_iter=100, tol=1e-5):
    """Singular Value Thresholding for matrix completion.

    Args:
        M_observed: observed entries (0 where missing)
        mask: boolean matrix (True = observed)
        rank: target rank
        max_iter: maximum iterations
        tol: convergence tolerance
    """
    M = M_observed.copy()

    for iteration in range(max_iter):
        # SVD
        U, S, Vt = np.linalg.svd(M, full_matrices=False)

        # Truncate to rank
        S_trunc = S[:rank]
        M_approx = U[:, :rank] @ np.diag(S_trunc) @ Vt[:rank, :]

        # Update: keep observed entries, fill missing with approximation
        M_new = np.where(mask, M_observed, M_approx)

        # Check convergence
        change = np.linalg.norm(M_new - M) / (np.linalg.norm(M) + 1e-10)
        M = M_new

        if change < tol:
            print(f"Converged in {iteration + 1} iterations")
            break

    return M

# Example: movie recommendation
np.random.seed(42)
n_users, n_movies = 20, 30
true_rank = 3

# True low-rank matrix
U_true = np.random.randn(n_users, true_rank)
V_true = np.random.randn(n_movies, true_rank)
M_true = U_true @ V_true.T

# Observe only 30% of entries
mask = np.random.rand(n_users, n_movies) < 0.3
M_observed = M_true * mask

# Complete the matrix
M_completed = matrix_completion_svt(M_observed, mask, rank=true_rank)

# Evaluate on missing entries
missing_mask = ~mask
error = np.sqrt(np.mean((M_completed[missing_mask] - M_true[missing_mask])**2))
print(f"RMSE on missing entries: {error:.4f}")
print(f"Observed fraction: {mask.mean():.1%}")
```

### 6.2 행렬 미분방정식

행렬 지수는 미분방정식 시스템과 연결되며, 제어 이론, 양자역학, 네트워크 분석에 응용됩니다:

```python
# Network diffusion: dx/dt = -L x (heat equation on a graph)
from scipy.linalg import expm

# Create a small graph
n = 6
edges = [(0,1), (0,2), (1,2), (1,3), (2,4), (3,4), (3,5), (4,5)]
A_adj = np.zeros((n, n))
for i, j in edges:
    A_adj[i, j] = A_adj[j, i] = 1

D = np.diag(A_adj.sum(axis=1))
L = D - A_adj  # Graph Laplacian

# Initial heat: concentrated at node 0
x0 = np.zeros(n)
x0[0] = 1.0

# Diffuse over time
times = [0, 0.1, 0.5, 1.0, 2.0, 5.0]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for ax, t in zip(axes.flat, times):
    x_t = expm(-L * t) @ x0
    ax.bar(range(n), x_t, color='steelblue', edgecolor='black')
    ax.set_ylim(0, 1.1)
    ax.set_title(f't = {t}')
    ax.set_xlabel('Node')
    ax.set_ylabel('Heat')
    ax.grid(True, alpha=0.3)

plt.suptitle('Heat Diffusion on a Graph (via Matrix Exponential)')
plt.tight_layout()
plt.show()

# At t -> infinity, heat is uniformly distributed
x_inf = expm(-L * 100) @ x0
print(f"Steady state: {np.round(x_inf, 4)}")
print(f"Sum preserved: {x_inf.sum():.4f} (should be {x0.sum():.1f})")
```

---

## 연습 문제

### 문제 1: 행렬 지수

1. 행렬 $A = \begin{bmatrix} 0 & -\omega \\ \omega & 0 \end{bmatrix}$에 대해, $e^{At}$를 해석적으로 계산하고 $\omega = 2$, $t = \pi/4$에 대해 `scipy.linalg.expm`으로 검증하시오
2. ODE 시스템 $\dot{x} = Ax$를 $A = \begin{bmatrix} -1 & 2 \\ 0 & -3 \end{bmatrix}$이고 초기 조건 $x(0) = [1, 1]^T$로 풀기
3. $t \in [0, 5]$에 대한 궤적을 그리시오

### 문제 2: Kronecker 곱 응용

1. Kronecker 곱을 사용하여 Lyapunov 방정식 $AX + XA^T = Q$를 주어진 $A$와 $Q$에 대해 풀기
2. `scipy.linalg.solve_continuous_lyapunov`와 비교하여 해를 검증하시오
3. Kronecker 접근법은 행렬 크기에 따라 어떻게 확장되는가? $n = 10, 50, 100, 200$에 대해 벤치마크하시오

### 문제 3: 행렬 미적분

다음 함수의 그래디언트를 해석적으로 유도한 후, 수치적으로 검증하시오:

1. $f(W) = \|XW - Y\|_F^2$ (행렬 최소자승)
2. $f(W) = \text{tr}(W^T A W B)$ (이차 형식)
3. $f(\Sigma) = \log\det(\Sigma) + \text{tr}(\Sigma^{-1}S)$ (가우시안의 음의 로그 우도)

### 문제 4: 유사역행렬과 투영

1. 랭크 2인 $(5, 3)$ 형상의 랭크 결핍 행렬 $A$에 대해 $A^+$를 계산하시오
2. $P = AA^+$가 $A$의 열공간으로의 투영임을 보이시오
3. $P' = A^+A$가 $A$의 행공간으로의 투영임을 보이시오
4. $P^2 = P$와 $P'^2 = P'$를 검증하시오

### 문제 5: 행렬 완성

랭크 5인 $50 \times 50$ 행렬을 생성하시오. 70%의 항목을 랜덤으로 제거하시오. SVT 알고리즘을 적용하고 다음을 측정하시오:

1. 반복 횟수에 따른 누락 항목의 RMSE
2. 가정된 랭크(1-10)에 따른 RMSE
3. 누락 비율이 50%에서 90%로 증가함에 따라 성능이 어떻게 저하되는가?

---

[이전: 19강](./19_Randomized_Linear_Algebra.md) | [개요](./00_Overview.md)

**License**: CC BY-NC 4.0
