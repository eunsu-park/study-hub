# 레슨 11: 양정부호 행렬

[이전: 레슨 10](./10_Matrix_Norms_and_Condition_Numbers.md) | [개요](./00_Overview.md) | [다음: 레슨 12](./12_Sparse_Matrices.md)

---

## 학습 목표

- 양정부호, 양의 준정부호, 음정부호, 부정부호 행렬을 정의할 수 있습니다
- 양정부호 판별의 여러 검사법을 적용할 수 있습니다: 고유값 검사, Cholesky 검사, 선행 주소행렬식, 피봇 검사
- 이차형식과 양정부호 행렬 사이의 관계를 이해할 수 있습니다
- 최적화와 볼록성에서 헤시안 행렬의 역할을 설명할 수 있습니다
- 공분산 행렬, 커널 방법, 준정부호 프로그래밍에서의 응용을 인식할 수 있습니다
- Python으로 양정부호 검사와 Cholesky 분해를 구현할 수 있습니다

---

## 1. 정의와 이차형식

### 1.1 이차형식

**이차형식**(quadratic form)은 대칭 행렬 $A \in \mathbb{R}^{n \times n}$에 의해 정의되는 벡터 $\mathbf{x} \in \mathbb{R}^n$의 스칼라 값 함수입니다:

$$Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} = \sum_{i=1}^{n} \sum_{j=1}^{n} a_{ij} x_i x_j$$

예를 들어, $2 \times 2$ 대칭 행렬의 경우:

$$Q(x_1, x_2) = \begin{bmatrix} x_1 & x_2 \end{bmatrix} \begin{bmatrix} a & b \\ b & c \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = a x_1^2 + 2b x_1 x_2 + c x_2^2$$

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a symmetric matrix
A = np.array([[3, 1],
              [1, 2]])

# Evaluate quadratic form at a point
x = np.array([1, 2])
Q = x.T @ A @ x
print(f"Q(x) = x^T A x = {Q}")

# Visualize the quadratic form surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = A[0, 0] * X1**2 + 2 * A[0, 1] * X1 * X2 + A[1, 1] * X2**2

ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Q(x)')
ax.set_title('Quadratic Form: x^T A x (Positive Definite)')
plt.tight_layout()
plt.show()
```

### 1.2 대칭 행렬의 분류

대칭 행렬 $A$는 이차형식에 따라 다음과 같이 분류됩니다:

| 분류 | 조건 | 고유값 |
|------|------|--------|
| **양정부호** (PD) | 모든 $\mathbf{x} \neq \mathbf{0}$에 대해 $\mathbf{x}^T A \mathbf{x} > 0$ | 모든 $\lambda_i > 0$ |
| **양의 준정부호** (PSD) | 모든 $\mathbf{x}$에 대해 $\mathbf{x}^T A \mathbf{x} \geq 0$ | 모든 $\lambda_i \geq 0$ |
| **음정부호** (ND) | 모든 $\mathbf{x} \neq \mathbf{0}$에 대해 $\mathbf{x}^T A \mathbf{x} < 0$ | 모든 $\lambda_i < 0$ |
| **음의 준정부호** (NSD) | 모든 $\mathbf{x}$에 대해 $\mathbf{x}^T A \mathbf{x} \leq 0$ | 모든 $\lambda_i \leq 0$ |
| **부정부호** | $Q$가 양수와 음수 모두를 취함 | 부호가 혼합 |

양정부호에 $A \succ 0$, 양의 준정부호에 $A \succeq 0$으로 씁니다. 이 표기법은 대칭 행렬에 대한 **뢰브너 순서**(Loewner order)라는 반순서를 정의합니다.

```python
# Classify different matrices
matrices = {
    "Positive definite": np.array([[2, -1], [-1, 2]]),
    "Positive semidefinite": np.array([[1, 1], [1, 1]]),
    "Negative definite": np.array([[-3, 1], [1, -4]]),
    "Indefinite": np.array([[1, 0], [0, -1]]),
}

for name, M in matrices.items():
    eigenvalues = np.linalg.eigvalsh(M)
    print(f"{name}:")
    print(f"  Matrix:\n{M}")
    print(f"  Eigenvalues: {eigenvalues}")

    # Test with random vectors
    n_tests = 10000
    results = []
    for _ in range(n_tests):
        x = np.random.randn(2)
        results.append(x.T @ M @ x)
    results = np.array(results)
    print(f"  Min Q(x): {results.min():.4f}, Max Q(x): {results.max():.4f}")
    print()
```

---

## 2. 양정부호 판별 검사

대칭 행렬이 양정부호인지 판별하는 여러 동치 검사가 있습니다. 각각 서로 다른 계산상의 장점을 가집니다.

### 2.1 고유값 검사

대칭 행렬 $A$가 양정부호인 것은 **모든 고유값이 순양수**인 것과 동치입니다.

```python
def is_positive_definite_eigenvalue(A, tol=1e-10):
    """Check positive definiteness via eigenvalues."""
    if not np.allclose(A, A.T):
        raise ValueError("Matrix must be symmetric")
    eigenvalues = np.linalg.eigvalsh(A)
    return np.all(eigenvalues > tol), eigenvalues

# Example
A = np.array([[4, 2, 1],
              [2, 5, 3],
              [1, 3, 6]])

is_pd, eigenvalues = is_positive_definite_eigenvalue(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Is positive definite? {is_pd}")
```

### 2.2 Cholesky 분해 검사

대칭 행렬 $A$가 양정부호인 것은 **Cholesky 분해** $A = LL^T$를 가지는 것과 동치이며, 여기서 $L$은 순양수 대각 원소를 가진 하삼각 행렬입니다.

Cholesky 분해는 본질적으로 대칭성을 활용하는 가우스 소거법입니다. 일반 LU 분해의 약 절반의 계산량을 필요로 합니다.

```python
def is_positive_definite_cholesky(A):
    """Check positive definiteness via Cholesky decomposition."""
    try:
        L = np.linalg.cholesky(A)
        return True, L
    except np.linalg.LinAlgError:
        return False, None

# Positive definite matrix
A_pd = np.array([[4, 2, 1],
                 [2, 5, 3],
                 [1, 3, 6]])

is_pd, L = is_positive_definite_cholesky(A_pd)
print(f"Is PD (Cholesky)? {is_pd}")
if L is not None:
    print(f"L:\n{L}")
    print(f"Reconstruction error: {np.linalg.norm(A_pd - L @ L.T):.2e}")

# Not positive definite
A_indef = np.array([[1, 2],
                    [2, 1]])
is_pd, _ = is_positive_definite_cholesky(A_indef)
print(f"\nIndefinite matrix is PD? {is_pd}")
```

### 2.3 선행 주소행렬식 (실베스터 판정법)

대칭 행렬 $A$가 양정부호인 것은 **모든 선행 주소행렬식이 양수**인 것과 동치입니다. $k$번째 선행 주소행렬식은 왼쪽 위 $k \times k$ 부분행렬의 행렬식입니다.

$$\Delta_1 = a_{11} > 0, \quad \Delta_2 = \det\begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} > 0, \quad \ldots, \quad \Delta_n = \det(A) > 0$$

```python
def is_positive_definite_minors(A, tol=1e-10):
    """Check positive definiteness via leading principal minors (Sylvester)."""
    n = A.shape[0]
    minors = []
    for k in range(1, n + 1):
        submatrix = A[:k, :k]
        det_k = np.linalg.det(submatrix)
        minors.append(det_k)
        if det_k <= tol:
            return False, minors
    return True, minors

A = np.array([[4, 2, 1],
              [2, 5, 3],
              [1, 3, 6]])

is_pd, minors = is_positive_definite_minors(A)
print(f"Leading principal minors: {minors}")
print(f"Is positive definite? {is_pd}")

# Compare: indefinite matrix
A_indef = np.array([[1, 3],
                    [3, 1]])
is_pd, minors = is_positive_definite_minors(A_indef)
print(f"\nIndefinite matrix minors: {minors}")
print(f"Is positive definite? {is_pd}")
```

### 2.4 피봇 검사

가우스 소거법(행 교환 없이) 수행 중, 대칭 행렬이 양정부호인 것은 **모든 피봇이 순양수**인 것과 동치입니다.

```python
def is_positive_definite_pivots(A, tol=1e-10):
    """Check positive definiteness via Gaussian elimination pivots."""
    n = A.shape[0]
    U = A.astype(float).copy()
    pivots = []

    for i in range(n):
        if abs(U[i, i]) <= tol:
            return False, pivots
        pivots.append(U[i, i])
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            U[j, i:] -= factor * U[i, i:]

    all_positive = all(p > tol for p in pivots)
    return all_positive, pivots

A = np.array([[4, 2, 1],
              [2, 5, 3],
              [1, 3, 6]])

is_pd, pivots = is_positive_definite_pivots(A)
print(f"Pivots: {pivots}")
print(f"Is positive definite? {is_pd}")
```

### 2.5 검사법 비교

| 검사 | 복잡도 | 장점 | 단점 |
|------|--------|------|------|
| 고유값 | $O(n^3)$ | 모든 고유값 제공 | 전체 고유분해 필요 |
| Cholesky | $O(n^3/3)$ | 가장 빠름; 유용한 분해 제공 | 예/아니오 답변만 |
| 실베스터 판정법 | $O(n^4)$ | 이론적 우아함 | 큰 $n$에서 수치적으로 불안정 |
| 피봇 | $O(n^3/3)$ | 가우스 소거법과 연결 | 신중한 구현 필요 |

실전에서는 **Cholesky 검사**가 표준 방법입니다. 분해가 성공하면 행렬은 양정부호입니다. Cholesky 인자 $L$ 자체가 선형 시스템 풀기와 다변량 가우시안 샘플링에 유용합니다.

---

## 3. 양정부호 행렬의 성질

### 3.1 대수적 성질

양정부호 행렬은 많은 유용한 폐포 성질을 만족합니다:

```python
import numpy as np
from scipy.linalg import sqrtm

# Start with a positive definite matrix
A = np.array([[4, 1],
              [1, 3]])

eigenvalues_A = np.linalg.eigvalsh(A)
print(f"A eigenvalues: {eigenvalues_A} (all positive)")

# Property 1: Sum of PD matrices is PD
B = np.array([[2, 0.5],
              [0.5, 2]])
C = A + B
print(f"\nA + B eigenvalues: {np.linalg.eigvalsh(C)} (all positive)")

# Property 2: Scalar multiple (positive scalar) preserves PD
print(f"3A eigenvalues: {np.linalg.eigvalsh(3 * A)} (all positive)")

# Property 3: Inverse of PD is PD
A_inv = np.linalg.inv(A)
print(f"A^(-1) eigenvalues: {np.linalg.eigvalsh(A_inv)} (all positive)")

# Property 4: Diagonal entries are positive
print(f"A diagonal: {np.diag(A)} (all positive)")

# Property 5: det(A) > 0 for PD
print(f"det(A) = {np.linalg.det(A):.4f} (positive)")

# Property 6: Congruence transformation preserves PD
# If A is PD and P is invertible, then P^T A P is PD
P = np.array([[1, 2],
              [0, 1]])
P_A_P = P.T @ A @ P
print(f"\nP^T A P eigenvalues: {np.linalg.eigvalsh(P_A_P)} (all positive)")

# Property 7: A^(1/2) exists and is PD (matrix square root)
A_sqrt = sqrtm(A).real
print(f"\nsqrt(A):\n{A_sqrt}")
print(f"sqrt(A) @ sqrt(A) = A? {np.allclose(A_sqrt @ A_sqrt, A)}")
print(f"sqrt(A) eigenvalues: {np.linalg.eigvalsh(A_sqrt)} (all positive)")
```

### 3.2 기하학적 해석: 타원체

양정부호 행렬 $A$는 타원체를 정의합니다:

$$\mathcal{E} = \{ \mathbf{x} \in \mathbb{R}^n : \mathbf{x}^T A \mathbf{x} \leq 1 \}$$

$A$의 고유벡터는 타원체의 주축을 제공하고, 고유값은 축의 길이를 결정합니다 (고유벡터 $\mathbf{v}_i$를 따른 반축 길이는 $1/\sqrt{\lambda_i}$).

```python
# Visualize ellipsoids for different PD matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

matrices = [
    np.array([[2, 0], [0, 2]]),    # Circle (identity-like)
    np.array([[4, 1], [1, 1]]),    # Tilted ellipse
    np.array([[5, 3], [3, 5]]),    # Tilted ellipse (different orientation)
]
titles = ["A = 2I (circle)", "Tilted ellipse", "Rotated ellipse"]

theta = np.linspace(0, 2 * np.pi, 200)
circle = np.array([np.cos(theta), np.sin(theta)])

for ax, A, title in zip(axes, matrices, titles):
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # The ellipsoid x^T A x = 1 is the image of the unit circle under A^(-1/2)
    A_inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    ellipse = A_inv_sqrt @ circle

    ax.plot(ellipse[0], ellipse[1], 'b-', linewidth=2)

    # Draw principal axes
    for i in range(2):
        v = eigenvectors[:, i] / np.sqrt(eigenvalues[i])
        ax.arrow(0, 0, v[0], v[1], head_width=0.05, head_length=0.03,
                 fc=f'C{i+1}', ec=f'C{i+1}', linewidth=2,
                 label=f'lambda={eigenvalues[i]:.2f}')
        ax.arrow(0, 0, -v[0], -v[1], head_width=0.05, head_length=0.03,
                 fc=f'C{i+1}', ec=f'C{i+1}', linewidth=2)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.show()
```

---

## 4. 헤시안과 최적화

### 4.1 헤시안 행렬

스칼라 값 함수 $f: \mathbb{R}^n \to \mathbb{R}$에 대해, 헤시안 행렬 $H$는 모든 2차 편미분을 포함합니다:

$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

헤시안은 항상 대칭입니다 (연속인 2차 도함수를 가정할 때, 슈바르츠 정리에 의해). 임계점 ($\nabla f = \mathbf{0}$인 곳)에서:

- $H \succ 0$ (양정부호) $\Rightarrow$ **극소점**
- $H \prec 0$ (음정부호) $\Rightarrow$ **극대점**
- $H$가 부정부호 $\Rightarrow$ **안장점**

```python
from scipy.optimize import minimize

# Example: f(x, y) = 3x^2 + 2xy + 2y^2 - 6x - 4y
# Gradient: [6x + 2y - 6, 2x + 4y - 4]
# Hessian: [[6, 2], [2, 4]]

def f(x):
    return 3*x[0]**2 + 2*x[0]*x[1] + 2*x[1]**2 - 6*x[0] - 4*x[1]

def grad_f(x):
    return np.array([6*x[0] + 2*x[1] - 6, 2*x[0] + 4*x[1] - 4])

# Hessian is constant for a quadratic function
H = np.array([[6, 2],
              [2, 4]])

eigenvalues_H = np.linalg.eigvalsh(H)
print(f"Hessian eigenvalues: {eigenvalues_H}")
print(f"Hessian is PD: {np.all(eigenvalues_H > 0)}")
print("=> Critical point is a local minimum")

# Find the critical point
result = minimize(f, x0=[0, 0], jac=grad_f)
print(f"\nMinimum at: {result.x}")
print(f"Minimum value: {result.fun:.4f}")

# Verify: solve gradient = 0
x_star = np.linalg.solve(H, np.array([6, 4]))
print(f"Analytic solution: {x_star}")
```

### 4.2 볼록성과 양의 준정부호성

두 번 미분 가능한 함수 $f$가 **볼록**(convex)인 것은 모든 $x$에서 헤시안 $H(x) \succeq 0$인 것과 동치입니다. 볼록성은 최적화의 초석입니다: 볼록 함수의 모든 극소점이 전역 최소점입니다.

```python
# Visualize convex vs non-convex functions
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)

# Convex function: f(x,y) = x^2 + y^2 (Hessian = 2I, PD)
Z_convex = X**2 + Y**2
axes[0].contour(X, Y, Z_convex, levels=20, cmap='viridis')
axes[0].set_title('Convex: f = x^2 + y^2\nHessian = 2I (PD)')
axes[0].set_aspect('equal')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

# Non-convex (saddle): f(x,y) = x^2 - y^2 (Hessian indefinite)
Z_saddle = X**2 - Y**2
axes[1].contour(X, Y, Z_saddle, levels=20, cmap='RdBu_r')
axes[1].set_title('Saddle: f = x^2 - y^2\nHessian indefinite')
axes[1].set_aspect('equal')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

plt.tight_layout()
plt.show()
```

### 4.3 뉴턴 방법과 헤시안

최적화를 위한 뉴턴 방법은 헤시안을 사용하여 갱신 방향을 계산합니다:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - H^{-1} \nabla f(\mathbf{x}_k)$$

이것은 $H$가 양정부호일 때 잘 작동합니다. 그렇지 않으면 뉴턴 방향이 오르막을 가리킬 수 있으며, 수정(Levenberg-Marquardt, 신뢰 영역)이 필요합니다.

```python
def newtons_method(f, grad_f, hess_f, x0, tol=1e-8, max_iter=100):
    """Newton's method for unconstrained optimization."""
    x = np.array(x0, dtype=float)
    history = [x.copy()]

    for i in range(max_iter):
        g = grad_f(x)
        H = hess_f(x)

        if np.linalg.norm(g) < tol:
            print(f"Converged in {i} iterations")
            break

        # Newton step: solve H @ dx = -g
        try:
            dx = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            print("Hessian is singular; cannot proceed")
            break

        x = x + dx
        history.append(x.copy())

    return x, np.array(history)

# Rosenbrock-like function (modified for demonstration)
def f_opt(x):
    return 5 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad_f_opt(x):
    dx0 = -20 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    dx1 = 10 * (x[1] - x[0]**2)
    return np.array([dx0, dx1])

def hess_f_opt(x):
    h00 = -20 * (x[1] - 3 * x[0]**2) + 2
    h01 = -20 * x[0]
    h11 = 10
    return np.array([[h00, h01],
                     [h01, h11]])

x_min, history = newtons_method(f_opt, grad_f_opt, hess_f_opt, x0=[-1, 1])
print(f"Minimum at: {x_min}")
print(f"Hessian at minimum:\n{hess_f_opt(x_min)}")
print(f"Hessian eigenvalues: {np.linalg.eigvalsh(hess_f_opt(x_min))}")
```

---

## 5. 실전에서의 양정부호 행렬

### 5.1 공분산 행렬

모든 공분산 행렬은 양의 준정부호입니다. 데이터에 완전한 선형 종속성이 없으면 공분산 행렬은 양정부호입니다.

$$\text{Cov}(X) = E[(X - \mu)(X - \mu)^T]$$

이것이 PSD인 이유는 모든 벡터 $\mathbf{a}$에 대해:

$$\mathbf{a}^T \text{Cov}(X) \mathbf{a} = E[(\mathbf{a}^T(X - \mu))^2] \geq 0$$

```python
# Generate multivariate data and verify covariance is PSD
np.random.seed(42)
n_samples = 500
mean = np.array([1, 2, 3])
cov_true = np.array([[2, 1, 0.5],
                     [1, 3, 1],
                     [0.5, 1, 2]])

# Check the true covariance is PD
L = np.linalg.cholesky(cov_true)
print(f"True covariance is PD (Cholesky exists)")

# Generate samples using Cholesky
Z = np.random.randn(n_samples, 3)
X = mean + (L @ Z.T).T

# Empirical covariance
cov_empirical = np.cov(X.T)
print(f"\nTrue covariance:\n{cov_true}")
print(f"\nEmpirical covariance:\n{np.round(cov_empirical, 3)}")

# Verify PD
eigenvalues = np.linalg.eigvalsh(cov_empirical)
print(f"\nEmpirical covariance eigenvalues: {eigenvalues}")
print(f"Is PD? {np.all(eigenvalues > 0)}")
```

### 5.2 커널 행렬 (그람 행렬)

커널 함수 $k(x, y)$가 유효한 것은 모든 입력 점 집합에 대해 커널 행렬(그람 행렬)이 양의 준정부호인 것과 동치입니다. 이것이 **머서 조건**(Mercer's condition)입니다.

$$K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j) \quad \Rightarrow \quad K \succeq 0$$

```python
from scipy.spatial.distance import cdist

# RBF (Gaussian) kernel
def rbf_kernel(X, gamma=1.0):
    """Compute the RBF kernel matrix."""
    sq_dists = cdist(X, X, 'sqeuclidean')
    return np.exp(-gamma * sq_dists)

# Generate data points
np.random.seed(42)
X = np.random.randn(50, 2)

# Compute kernel matrix
K = rbf_kernel(X, gamma=0.5)

# Verify PSD
eigenvalues = np.linalg.eigvalsh(K)
print(f"Kernel matrix shape: {K.shape}")
print(f"Min eigenvalue: {eigenvalues.min():.6e}")
print(f"Max eigenvalue: {eigenvalues.max():.6e}")
print(f"All eigenvalues >= 0? {np.all(eigenvalues >= -1e-10)}")

# Visualize eigenvalue spectrum
plt.figure(figsize=(8, 4))
plt.plot(eigenvalues, 'b.-')
plt.xlabel('Index')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalue Spectrum of RBF Kernel Matrix')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 5.3 양정부호성 보장을 위한 정규화

실전에서 PSD여야 할 행렬이 수치 오차로 인해 이 성질을 잃을 수 있습니다. 작은 대각 섭동(ridge)을 추가하면 양정부호성이 복원됩니다:

$$A_{\text{reg}} = A + \epsilon I$$

$A$의 최소 고유값이 $\lambda_{\min}$이면, $A_{\text{reg}}$의 최소 고유값은 $\lambda_{\min} + \epsilon$입니다.

```python
def make_positive_definite(A, epsilon=1e-6):
    """Add ridge to ensure positive definiteness."""
    eigenvalues = np.linalg.eigvalsh(A)
    if np.all(eigenvalues > 0):
        return A
    # Shift eigenvalues so minimum is epsilon
    shift = max(0, -eigenvalues.min()) + epsilon
    return A + shift * np.eye(A.shape[0])

# Near-singular covariance (e.g., highly correlated features)
A_near_singular = np.array([[1.0, 0.9999],
                            [0.9999, 1.0]])

eigenvalues = np.linalg.eigvalsh(A_near_singular)
print(f"Original eigenvalues: {eigenvalues}")
print(f"Condition number: {eigenvalues.max() / eigenvalues.min():.0f}")

# Regularize
A_reg = make_positive_definite(A_near_singular, epsilon=1e-4)
eigenvalues_reg = np.linalg.eigvalsh(A_reg)
print(f"\nRegularized eigenvalues: {eigenvalues_reg}")
print(f"Condition number: {eigenvalues_reg.max() / eigenvalues_reg.min():.0f}")
```

---

## 6. 준정부호 프로그래밍 (SDP)

### 6.1 SDP란?

준정부호 프로그래밍은 선형 프로그래밍의 일반화로, 제약 조건이 행렬의 양의 준정부호성을 포함합니다:

$$\min_X \text{tr}(CX) \quad \text{subject to} \quad \text{tr}(A_i X) = b_i, \quad X \succeq 0$$

SDP는 선형 프로그래밍, 이차 프로그래밍, 이차 원뿔 프로그래밍을 특수한 경우로 포함하는 강력한 프레임워크입니다. 제어 이론, 조합 최적화, 양자 정보에 응용됩니다.

### 6.2 CVXPY를 사용한 SDP 예제

```python
# pip install cvxpy
import cvxpy as cp

# Simple SDP: find the nearest correlation matrix
# Given a matrix that is almost but not quite a valid correlation matrix,
# find the nearest PSD matrix with unit diagonal

# Corrupted correlation matrix (not PSD)
C = np.array([[1.0,  0.9,  0.7],
              [0.9,  1.0, -0.4],
              [0.7, -0.4,  1.0]])

eigenvalues_C = np.linalg.eigvalsh(C)
print(f"Original eigenvalues: {eigenvalues_C}")
print(f"Is PSD? {np.all(eigenvalues_C >= 0)}")

# SDP formulation: minimize ||X - C||_F subject to X PSD, diag(X) = 1
X = cp.Variable((3, 3), symmetric=True)
objective = cp.Minimize(cp.norm(X - C, 'fro'))
constraints = [X >> 0]  # X is PSD
for i in range(3):
    constraints.append(X[i, i] == 1)

prob = cp.Problem(objective, constraints)
prob.solve()

X_opt = X.value
print(f"\nNearest correlation matrix:\n{np.round(X_opt, 4)}")
print(f"Eigenvalues: {np.linalg.eigvalsh(X_opt)}")
print(f"Frobenius distance: {np.linalg.norm(X_opt - C):.4f}")
```

### 6.3 MAX-CUT 완화

조합론의 고전적인 MAX-CUT 문제는 SDP 완화(Goemans-Williamson 알고리즘, 0.878 근사비 달성)를 사용하여 근사할 수 있습니다:

```python
# MAX-CUT SDP relaxation for a small graph
# Graph with 4 nodes
W = np.array([[0, 1, 0, 1],
              [1, 0, 1, 0],
              [0, 1, 0, 1],
              [1, 0, 1, 0]])  # Cycle graph C4

n = W.shape[0]
L = np.diag(W.sum(axis=1)) - W  # Laplacian matrix

# SDP relaxation: max (1/4) tr(LX) s.t. diag(X)=1, X PSD
X = cp.Variable((n, n), symmetric=True)
objective = cp.Maximize(0.25 * cp.trace(L @ X))
constraints = [X >> 0]
for i in range(n):
    constraints.append(X[i, i] == 1)

prob = cp.Problem(objective, constraints)
prob.solve()

print(f"SDP optimal value (upper bound on MAX-CUT): {prob.value:.4f}")
print(f"SDP solution X:\n{np.round(X.value, 3)}")

# Round the SDP solution to get an actual cut
# Use random hyperplane rounding
from scipy.linalg import sqrtm
V = sqrtm(X.value).real
r = np.random.randn(n)
cut = np.sign(V @ r)
cut_value = 0.25 * sum(W[i, j] * (1 - cut[i] * cut[j])
                        for i in range(n) for j in range(i+1, n))
print(f"Rounded cut: {cut}")
print(f"Cut value: {cut_value:.0f}")
```

---

## 7. 슈어 보행렬과 블록 양정부호성

### 7.1 슈어 보행렬 (Schur Complement)

블록 행렬에 대해:

$$M = \begin{bmatrix} A & B \\ B^T & C \end{bmatrix}$$

여기서 $A$가 가역이면, $M$에서 $A$의 **슈어 보행렬**은:

$$S = C - B^T A^{-1} B$$

**핵심 결과**: $M \succ 0$인 것은 $A \succ 0$이고 $S \succ 0$인 것과 동치입니다.

```python
# Schur complement example
A = np.array([[4, 1],
              [1, 3]])
B = np.array([[1, 0],
              [0, 1]])
C = np.array([[5, 1],
              [1, 4]])

# Form the block matrix
M = np.block([[A, B],
              [B.T, C]])

print(f"Block matrix M:\n{M}")
print(f"M eigenvalues: {np.linalg.eigvalsh(M)}")
print(f"M is PD? {np.all(np.linalg.eigvalsh(M) > 0)}")

# Check via Schur complement
A_inv = np.linalg.inv(A)
S = C - B.T @ A_inv @ B  # Schur complement

print(f"\nA eigenvalues: {np.linalg.eigvalsh(A)}")
print(f"A is PD? {np.all(np.linalg.eigvalsh(A) > 0)}")

print(f"\nSchur complement S = C - B^T A^(-1) B:")
print(f"{S}")
print(f"S eigenvalues: {np.linalg.eigvalsh(S)}")
print(f"S is PD? {np.all(np.linalg.eigvalsh(S) > 0)}")
print(f"\nConclusion: M is PD (both A and S are PD)")
```

---

## 연습 문제

### 연습 문제 1: 양정부호 판별

다음 각 행렬이 양정부호, 양의 준정부호, 음정부호, 부정부호 중 어디에 해당하는지 판별하세요. 각 행렬에 대해 최소 두 가지 다른 검사를 사용하세요.

$$A = \begin{bmatrix} 5 & 2 \\ 2 & 1 \end{bmatrix}, \quad
B = \begin{bmatrix} 1 & -1 & 0 \\ -1 & 3 & -1 \\ 0 & -1 & 2 \end{bmatrix}, \quad
C = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 5 & 7 \\ 3 & 7 & 10 \end{bmatrix}$$

### 연습 문제 2: 수작업 Cholesky 분해

다음 행렬의 Cholesky 분해를 수작업으로 계산하세요 ($L$의 각 원소를 단계별로 유도), 그런 다음 `np.linalg.cholesky`로 검증하세요:

$$A = \begin{bmatrix} 4 & 2 & -2 \\ 2 & 10 & 2 \\ -2 & 2 & 5 \end{bmatrix}$$

### 연습 문제 3: 헤시안 분석

함수 $f(x, y) = x^3 + y^3 - 3xy$에 대해 임계점에서 헤시안을 계산하고 각각을 극소점, 극대점 또는 안장점으로 분류하세요.

### 연습 문제 4: 가장 가까운 양정부호 행렬

임의의 대칭 행렬을 받아 프로베니우스 노름에서 가장 가까운 양의 준정부호 행렬을 반환하는 함수를 작성하세요. 힌트: 고유분해를 계산하고 음의 고유값을 0으로 설정하세요.

### 연습 문제 5: 공분산과 샘플링

목표 공분산 행렬 $\Sigma = \begin{bmatrix} 3 & 1 & 0.5 \\ 1 & 2 & 0.8 \\ 0.5 & 0.8 & 1.5 \end{bmatrix}$이 주어졌을 때:

1. $\Sigma$가 양정부호인지 검증하세요
2. Cholesky 분해를 계산하세요
3. Cholesky 인자를 사용하여 $\mathcal{N}(\mathbf{0}, \Sigma)$에서 10,000개의 샘플을 생성하세요
4. 경험적 공분산을 $\Sigma$와 비교하세요

---

[이전: 레슨 10](./10_Matrix_Norms_and_Condition_Numbers.md) | [개요](./00_Overview.md) | [다음: 레슨 12](./12_Sparse_Matrices.md)

**License**: CC BY-NC 4.0
