# 레슨 13: 수치 선형대수

[이전: 레슨 12](./12_Sparse_Matrices.md) | [개요](./00_Overview.md) | [다음: 레슨 14](./14_Iterative_Methods.md)

---

## 학습 목표

- IEEE 754 부동소수점 표현과 선형대수에 대한 함의를 이해할 수 있습니다
- 머신 엡실론(machine epsilon)을 정의하고 계산에 미치는 영향을 시연할 수 있습니다
- 조건수(condition number)를 계산하고 해석하여 문제의 민감도를 평가할 수 있습니다
- 전진 안정성, 후진 안정성, 수치 안정성을 구별할 수 있습니다
- 수치 알고리즘에서 파국적 소거(catastrophic cancellation)를 식별하고 방지할 수 있습니다
- 기본 알고리즘(가우스 소거법, 그람-슈미트, 하우스홀더)의 수치 안정성을 평가할 수 있습니다

---

## 1. 부동소수점 산술

### 1.1 IEEE 754 표현

모든 부동소수점 수는 다음과 같이 저장됩니다:

$$x = (-1)^s \times m \times 2^e$$

여기서 $s$는 부호 비트, $m$은 가수(significand), $e$는 지수입니다.

| 형식 | 총 비트 | 가수 비트 | 지수 비트 | 머신 엡실론 |
|------|---------|----------|----------|------------|
| float16 (반정밀도) | 16 | 10 | 5 | $\approx 9.77 \times 10^{-4}$ |
| float32 (단정밀도) | 32 | 23 | 8 | $\approx 1.19 \times 10^{-7}$ |
| float64 (배정밀도) | 64 | 52 | 11 | $\approx 2.22 \times 10^{-16}$ |

```python
import numpy as np

# Inspect floating-point properties
for dtype in [np.float16, np.float32, np.float64]:
    info = np.finfo(dtype)
    print(f"{dtype.__name__:>10}: eps={info.eps:.3e}, "
          f"min={info.tiny:.3e}, max={info.max:.3e}, "
          f"mantissa bits={info.nmant}")
```

### 1.2 주요 부동소수점 특성

부동소수점 산술은 실수 산술과 같은 대수 법칙을 따르지 않습니다. 핵심적인 차이는 모든 연산의 결과가 가장 가까운 표현 가능한 수로 **반올림**된다는 것입니다.

```python
# 1. Associativity fails
a = 1e16
b = -1e16
c = 1.0
print(f"(a + b) + c = {(a + b) + c}")  # 1.0
print(f"a + (b + c) = {a + (b + c)}")  # 0.0 (c is lost when added to a)

# 2. Distributivity fails
x = 1e-15
y = 1.0
z = -1.0
print(f"\nx * (y + z) = {x * (y + z)}")   # 0.0
print(f"x*y + x*z = {x * y + x * z}")     # 0.0 (but intermediate values differ)

# 3. Not all decimal fractions are representable
print(f"\n0.1 + 0.2 = {0.1 + 0.2}")  # 0.30000000000000004
print(f"0.1 + 0.2 == 0.3: {0.1 + 0.2 == 0.3}")  # False

# 4. Spacing between consecutive floats
x = 1.0
eps = np.spacing(x)
print(f"\nSpacing at 1.0: {eps}")
print(f"1.0 + eps/2 == 1.0: {1.0 + eps / 2 == 1.0}")  # True (rounded down)
print(f"1.0 + eps == 1.0: {1.0 + eps == 1.0}")          # False
```

### 1.3 부동소수점 산술의 표준 모델

IEEE 754 표준은 모든 산술 연산 $\circ \in \{+, -, \times, /\}$에 대해 다음을 보장합니다:

$$\text{fl}(a \circ b) = (a \circ b)(1 + \delta), \quad |\delta| \leq \epsilon_{\text{mach}}$$

여기서 $\epsilon_{\text{mach}}$는 머신 엡실론입니다. 이는 각 개별 연산이 최대 $\epsilon_{\text{mach}}$의 상대 오차를 도입함을 의미하지만, 오차는 많은 연산에 걸쳐 누적될 수 있습니다.

---

## 2. 머신 엡실론

### 2.1 정의

머신 엡실론($\epsilon_{\text{mach}}$)은 다음을 만족하는 가장 작은 양수입니다:

$$\text{fl}(1 + \epsilon_{\text{mach}}) > 1$$

동치적으로, 1.0과 다음 표현 가능한 부동소수점 수 사이 간격의 절반입니다.

```python
def compute_machine_epsilon(dtype=np.float64):
    """Compute machine epsilon by successive halving."""
    eps = dtype(1.0)
    while dtype(1.0) + eps > dtype(1.0):
        eps_prev = eps
        eps = dtype(eps / 2.0)
    return eps_prev

# Compute for different precisions
for dtype in [np.float16, np.float32, np.float64]:
    computed = compute_machine_epsilon(dtype)
    actual = np.finfo(dtype).eps
    print(f"{dtype.__name__:>10}: computed={computed:.3e}, "
          f"numpy={actual:.3e}, match={np.isclose(computed, actual)}")
```

### 2.2 상대 오차 vs 절대 오차

계산값 $\hat{x}$가 참값 $x$를 근사할 때:

- **절대 오차**: $|x - \hat{x}|$
- **상대 오차**: $\frac{|x - \hat{x}|}{|x|}$ ($x \neq 0$인 경우)

부동소수점 산술은 절대 오차가 아닌 **상대 오차**를 제어합니다. 이는 작은 수는 작은 절대 오차를 가지고, 큰 수는 큰 절대 오차를 가짐을 의미합니다.

```python
# Relative error is bounded, absolute error scales with magnitude
for magnitude in [1e-10, 1e0, 1e10]:
    x = magnitude
    x_plus = np.nextafter(x, np.inf)
    abs_error = x_plus - x
    rel_error = abs_error / x
    print(f"x = {x:.0e}: abs_error = {abs_error:.3e}, "
          f"rel_error = {rel_error:.3e}")
```

---

## 3. 조건수

### 3.1 정의

**조건수**(condition number)는 입력의 작은 섭동에 대한 문제 출력의 민감도를 측정합니다. 행렬 $A$에 대해 조건수(2-노름 기준)는:

$$\kappa(A) = \|A\| \cdot \|A^{-1}\| = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}$$

여기서 $\sigma_{\max}$와 $\sigma_{\min}$은 최대 및 최소 특이값입니다.

| $\kappa(A)$ | 해석 |
|---|---|
| $\approx 1$ | 잘 조건화됨 |
| $10^3$ | 보통; 정확도 3자리 손실 |
| $10^{16}$ | 불량 조건화; float64에서 사실상 특이 |
| $\infty$ | 특이 행렬 |

```python
# Condition number examples
matrices = {
    "Identity": np.eye(4),
    "Well-conditioned": np.array([[2, 1], [1, 2]]),
    "Moderate": np.array([[1, 1], [1, 1.0001]]),
    "Hilbert (ill-conditioned)": np.array([
        [1, 1/2, 1/3, 1/4],
        [1/2, 1/3, 1/4, 1/5],
        [1/3, 1/4, 1/5, 1/6],
        [1/4, 1/5, 1/6, 1/7]
    ]),
}

for name, A in matrices.items():
    cond = np.linalg.cond(A)
    sv = np.linalg.svd(A, compute_uv=False)
    print(f"{name:30s}: cond = {cond:.2e}, "
          f"sigma_max = {sv[0]:.4f}, sigma_min = {sv[-1]:.4e}")
```

### 3.2 조건수와 해의 정확도

선형 시스템 $Ax = b$에서, 우변의 섭동 $\delta b$는 해의 섭동 $\delta x$를 야기하며 다음으로 제한됩니다:

$$\frac{\|\delta x\|}{\|x\|} \leq \kappa(A) \frac{\|\delta b\|}{\|b\|}$$

이는 $\kappa(A) = 10^k$이면 약 $k$자리의 정확도를 잃음을 의미합니다.

```python
# Demonstrate condition number effect on solution accuracy
from scipy.linalg import hilbert

for n in [3, 5, 8, 10, 12]:
    H = hilbert(n)
    x_true = np.ones(n)
    b = H @ x_true

    # Solve
    x_computed = np.linalg.solve(H, b)

    # Errors
    rel_error = np.linalg.norm(x_computed - x_true) / np.linalg.norm(x_true)
    cond = np.linalg.cond(H)
    digits_lost = np.log10(cond)

    print(f"n={n:2d}: cond(H) = {cond:.2e}, "
          f"rel_error = {rel_error:.2e}, "
          f"digits lost ~ {digits_lost:.1f}")
```

### 3.3 일반적인 연산의 조건수

```python
# Condition number of matrix multiplication
A = np.random.randn(4, 4)
B = np.random.randn(4, 4)

print(f"cond(A) = {np.linalg.cond(A):.2f}")
print(f"cond(B) = {np.linalg.cond(B):.2f}")
print(f"cond(AB) = {np.linalg.cond(A @ B):.2f}")
print(f"cond(A) * cond(B) = {np.linalg.cond(A) * np.linalg.cond(B):.2f}")
print("Note: cond(AB) <= cond(A) * cond(B)")

# Condition number of eigenvalue problems
# Eigenvalues of symmetric matrices are well-conditioned
# Eigenvalues of non-symmetric matrices can be ill-conditioned
A_sym = np.array([[2, 1], [1, 3]])
A_nonsym = np.array([[1, 1000], [0, 2]])

print(f"\nSymmetric: cond = {np.linalg.cond(A_sym):.2f}")
print(f"Non-symmetric: cond = {np.linalg.cond(A_nonsym):.2f}")
```

---

## 4. 수치 안정성

### 4.1 전진 안정성과 후진 안정성

알고리즘은 다음과 같습니다:

- **전진 안정**(forward stable): 정확한 입력에 대해 계산된 결과 $\hat{y}$가 $\|\hat{y} - y\| / \|y\| = O(\epsilon_{\text{mach}})$를 만족합니다.
- **후진 안정**(backward stable): 계산된 결과가 약간 섭동된 문제의 정확한 답입니다: $\hat{y} = f(x + \delta x)$ 여기서 $\|\delta x\| / \|x\| = O(\epsilon_{\text{mach}})$.

후진 안정성이 최고 기준입니다: 알고리즘이 입력 데이터에 이미 존재하는 오차보다 나쁘지 않은 오차를 도입한다는 것을 의미합니다.

### 4.2 예제: 합산 알고리즘

```python
def naive_sum(x):
    """Left-to-right summation. Error grows as O(n * eps)."""
    s = 0.0
    for xi in x:
        s += xi
    return s

def kahan_sum(x):
    """Kahan compensated summation. Error is O(eps) regardless of n."""
    s = 0.0
    c = 0.0  # Running compensation for lost low-order bits
    for xi in x:
        y = xi - c
        t = s + y
        c = (t - s) - y  # Algebraically zero, but captures rounding error
        s = t
    return s

def pairwise_sum(x):
    """Pairwise (recursive) summation. Error grows as O(log(n) * eps)."""
    n = len(x)
    if n <= 256:
        return sum(x)
    mid = n // 2
    return pairwise_sum(x[:mid]) + pairwise_sum(x[mid:])

# Test with a known-difficult case
np.random.seed(42)
n = 100000
# Mix of large and small numbers
x = np.concatenate([
    np.random.randn(n // 2) * 1e8,
    np.random.randn(n // 2) * 1e-8
])
np.random.shuffle(x)

# "True" sum using higher precision
true_sum = np.float64(np.sum(x.astype(np.float128)))

results = {
    "np.sum": np.sum(x),
    "naive_sum": naive_sum(x),
    "kahan_sum": kahan_sum(x),
    "pairwise_sum": pairwise_sum(x),
}

for name, result in results.items():
    rel_error = abs(result - true_sum) / abs(true_sum)
    print(f"{name:15s}: result = {result:.10e}, rel_error = {rel_error:.2e}")
```

---

## 5. 파국적 소거

### 5.1 파국적 소거란?

**파국적 소거**(catastrophic cancellation)는 거의 같은 두 수를 뺄 때 발생합니다. 각 수가 완전한 정밀도를 가질 수 있지만, 그 차이는 훨씬 적은 유효 숫자를 가집니다.

```python
# Classic example: quadratic formula
# Solve x^2 - 200x + 1 = 0
a, b, c = 1.0, -200.0, 1.0

# Standard formula (suffers from cancellation for one root)
disc = np.sqrt(b**2 - 4*a*c)
x1_standard = (-b + disc) / (2*a)
x2_standard = (-b - disc) / (2*a)

# True roots (computed with higher precision)
# x = 100 +/- sqrt(9999) = 100 +/- 99.99500...
x1_true = 199.994999874993750
x2_true = 0.005000125006250

print("Standard quadratic formula:")
print(f"  x1 = {x1_standard:.15f} (error = {abs(x1_standard - x1_true):.2e})")
print(f"  x2 = {x2_standard:.15f} (error = {abs(x2_standard - x2_true):.2e})")

# Numerically stable alternative: use Vieta's formula for the small root
# x1 * x2 = c/a, so x2 = c / (a * x1)
x2_stable = c / (a * x1_standard)
print(f"\nStable formula:")
print(f"  x2 = {x2_stable:.15f} (error = {abs(x2_stable - x2_true):.2e})")
```

### 5.2 소거의 추가 예제

```python
# Example 1: Computing variance
# Naive: Var = E[X^2] - (E[X])^2 can suffer from cancellation
np.random.seed(42)
x = np.random.randn(10000) + 1e8  # Large mean, small variance

# Naive formula
mean_x = np.mean(x)
var_naive = np.mean(x**2) - mean_x**2
var_stable = np.var(x)  # Uses centered formula: mean((x - mean)^2)

print(f"Naive variance:  {var_naive:.6f}")
print(f"Stable variance: {var_stable:.6f}")
print(f"Relative diff: {abs(var_naive - var_stable) / var_stable:.2e}")

# Example 2: exp(x) - 1 for small x
x_small = 1e-15
print(f"\nexp({x_small}) - 1:")
print(f"  Naive:  {np.exp(x_small) - 1:.6e}")  # Cancellation!
print(f"  expm1:  {np.expm1(x_small):.6e}")     # Numerically stable

# Example 3: log(1 + x) for small x
print(f"\nlog(1 + {x_small}):")
print(f"  Naive:  {np.log(1 + x_small):.6e}")   # Cancellation!
print(f"  log1p:  {np.log1p(x_small):.6e}")      # Numerically stable
```

### 5.3 선형대수에서 소거 방지

```python
# Cancellation in determinants
# det(A) can be catastrophically inaccurate for ill-conditioned matrices

# Wilkinson matrix: nearly singular
n = 10
A = np.eye(n)
A[:, -1] = 1
A[-1, :] = -1 * np.ones(n)
A[-1, -1] = 1

# Compare determinant methods
det_numpy = np.linalg.det(A)
sign, logdet = np.linalg.slogdet(A)  # log-space computation
det_slog = sign * np.exp(logdet)

print(f"np.linalg.det: {det_numpy:.6e}")
print(f"slogdet:       {det_slog:.6e}")
print(f"log|det|:      {logdet:.6f}")

# For large matrices, always use slogdet
print("\nFor products of many matrices, log-determinant avoids overflow:")
A = np.random.randn(100, 100)
sign, logdet = np.linalg.slogdet(A)
print(f"sign = {sign}, log|det| = {logdet:.2f}")
```

---

## 6. 기본 알고리즘의 안정성

### 6.1 부분 피봇팅을 사용한 가우스 소거법

피봇팅 없이 가우스 소거법은 불안정할 수 있습니다. **부분 피봇팅**(가장 큰 원소를 대각에 놓기 위해 행을 교환)은 실전에서 후진 안정적입니다.

```python
from scipy.linalg import lu, solve

# Without pivoting: potential instability
A_bad = np.array([[1e-20, 1.0],
                  [1.0, 1.0]])
b = np.array([1.0, 2.0])

# LU with pivoting (default in scipy)
P, L, U = lu(A_bad)
print("LU with pivoting:")
print(f"P:\n{P}")
print(f"L:\n{L}")
print(f"U:\n{U}")

x = solve(A_bad, b)
print(f"Solution: {x}")
print(f"Residual: {np.linalg.norm(A_bad @ x - b):.2e}")

# Growth factor: ratio of largest entry in U to largest in A
growth = np.max(np.abs(U)) / np.max(np.abs(A_bad))
print(f"Growth factor: {growth:.2e}")
```

### 6.2 고전적 vs 수정 그람-슈미트

고전적 그람-슈미트(CGS) 과정은 불량 조건 행렬에서 직교성을 잃습니다. 수정 그람-슈미트(MGS)가 더 안정적이고, 하우스홀더 반사가 후진 안정적입니다.

```python
def classical_gram_schmidt(A):
    """Classical Gram-Schmidt: numerically unstable."""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R

def modified_gram_schmidt(A):
    """Modified Gram-Schmidt: better numerical stability."""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = A.copy().astype(float)
    for j in range(n):
        R[j, j] = np.linalg.norm(V[:, j])
        Q[:, j] = V[:, j] / R[j, j]
        for i in range(j + 1, n):
            R[j, i] = Q[:, j] @ V[:, i]
            V[:, i] -= R[j, i] * Q[:, j]
    return Q, R

# Compare on ill-conditioned matrix
n = 50
# Construct an ill-conditioned matrix
singular_values = np.logspace(0, -12, n)
U, _ = np.linalg.qr(np.random.randn(n, n))
Vt, _ = np.linalg.qr(np.random.randn(n, n))
A = U @ np.diag(singular_values) @ Vt
print(f"Condition number: {np.linalg.cond(A):.2e}")

# Classical Gram-Schmidt
Q_cgs, R_cgs = classical_gram_schmidt(A)
orth_error_cgs = np.linalg.norm(Q_cgs.T @ Q_cgs - np.eye(n))

# Modified Gram-Schmidt
Q_mgs, R_mgs = modified_gram_schmidt(A)
orth_error_mgs = np.linalg.norm(Q_mgs.T @ Q_mgs - np.eye(n))

# Householder (NumPy's QR, backward stable)
Q_hh, R_hh = np.linalg.qr(A)
orth_error_hh = np.linalg.norm(Q_hh.T @ Q_hh - np.eye(n))

print(f"\nOrthogonality error (||Q^T Q - I||):")
print(f"  Classical GS:     {orth_error_cgs:.2e}")
print(f"  Modified GS:      {orth_error_mgs:.2e}")
print(f"  Householder (QR): {orth_error_hh:.2e}")
```

### 6.3 하우스홀더 반사

하우스홀더 반사는 후진 안정적 QR 인수분해의 근간입니다. 하우스홀더 반사기는 다음 형태의 직교 행렬입니다:

$$H = I - 2\frac{\mathbf{v}\mathbf{v}^T}{\mathbf{v}^T\mathbf{v}}$$

```python
def householder_qr(A):
    """QR factorization using Householder reflections (backward stable)."""
    m, n = A.shape
    R = A.copy().astype(float)
    Q = np.eye(m)

    for j in range(min(m - 1, n)):
        # Compute Householder vector
        x = R[j:, j]
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x) * np.sign(x[0])
        v = x + e1
        v = v / np.linalg.norm(v)

        # Apply reflection: R[j:, j:] = R[j:, j:] - 2 v (v^T R[j:, j:])
        R[j:, j:] -= 2 * np.outer(v, v @ R[j:, j:])

        # Accumulate Q
        Q[j:, :] -= 2 * np.outer(v, v @ Q[j:, :])

    return Q.T, R

# Test
A = np.random.randn(5, 4)
Q, R = householder_qr(A)
print(f"||QR - A|| = {np.linalg.norm(Q @ R - A):.2e}")
print(f"||Q^T Q - I|| = {np.linalg.norm(Q.T @ Q - np.eye(5)):.2e}")
```

---

## 7. 혼합 정밀도와 오차 분석

### 7.1 혼합 정밀도 산술

최신 하드웨어는 여러 부동소수점 정밀도를 지원합니다. 계산의 일부에 낮은 정밀도(float16, bfloat16)를 사용하면 **반복 세분화**(iterative refinement)를 통해 정확도를 유지하면서 속도를 극적으로 향상시킬 수 있습니다.

```python
# Effect of precision on linear algebra
A = np.random.randn(100, 100)
b = np.random.randn(100)
x_true = np.linalg.solve(A, b)

for dtype in [np.float16, np.float32, np.float64]:
    A_d = A.astype(dtype)
    b_d = b.astype(dtype)

    if dtype == np.float16:
        # float16 does not have good LAPACK support; use manual solve
        x_d = np.linalg.solve(A_d.astype(np.float32), b_d.astype(np.float32))
    else:
        x_d = np.linalg.solve(A_d, b_d)

    rel_error = np.linalg.norm(x_d - x_true) / np.linalg.norm(x_true)
    residual = np.linalg.norm(A @ x_d.astype(np.float64) - b)
    print(f"{dtype.__name__:>10}: rel_error = {rel_error:.2e}, "
          f"residual = {residual:.2e}")
```

### 7.2 반복 세분화

반복 세분화는 저정밀도에서 시스템을 풀고, 고정밀도에서 계산한 잔차를 사용하여 해를 보정합니다:

1. 저정밀도에서 $A\hat{x} = b$를 풉니다
2. 고정밀도에서 잔차 $r = b - A\hat{x}$를 계산합니다
3. 저정밀도에서 $A\delta x = r$를 풉니다
4. $\hat{x} \leftarrow \hat{x} + \delta x$를 갱신합니다
5. 수렴할 때까지 반복합니다

```python
def iterative_refinement(A, b, solve_low, max_iter=10, tol=1e-14):
    """Iterative refinement using low-precision solve with high-precision residual."""
    x = solve_low(A, b)
    residuals = []

    for i in range(max_iter):
        r = b - A @ x  # High precision residual
        residual_norm = np.linalg.norm(r) / np.linalg.norm(b)
        residuals.append(residual_norm)

        if residual_norm < tol:
            print(f"Converged in {i+1} iterations")
            break

        dx = solve_low(A, r)
        x = x + dx

    return x, residuals

# Simulate low-precision solve by rounding
def solve_low_precision(A, b):
    """Simulate float32 solve."""
    return np.linalg.solve(A.astype(np.float32), b.astype(np.float32)).astype(np.float64)

np.random.seed(42)
A = np.random.randn(200, 200)
b = np.random.randn(200)

x_refined, residuals = iterative_refinement(A, b, solve_low_precision)
x_direct = np.linalg.solve(A, b)

print(f"\nDirect solve error:  {np.linalg.norm(A @ x_direct - b):.2e}")
print(f"Refined solve error: {np.linalg.norm(A @ x_refined - b):.2e}")

# Plot convergence
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.semilogy(residuals, 'bo-')
plt.xlabel('Iteration')
plt.ylabel('Relative residual')
plt.title('Iterative Refinement Convergence')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 8. 실전 지침

### 8.1 경험 법칙

1. **부동소수점 수를 `==`로 비교하지 마세요**. `np.isclose()`나 `np.allclose()`를 사용하세요.
2. 선형 시스템을 풀기 전에 **조건수를 확인하세요**. $\kappa \cdot \epsilon_{\text{mach}} \approx 1$이면 답이 무의미합니다.
3. **안정적인 알고리즘을 선호하세요**: 그람-슈미트보다 하우스홀더 QR, 비피봇 LU보다 피봇 LU.
4. **$A^{-1}$를 명시적으로 계산하지 마세요**. 대신 $Ax = b$를 풀어야 합니다.
5. 큰 행렬에서 오버플로를 피하려면 **`det` 대신 `slogdet`를 사용하세요**.
6. 소거 경계 근처의 표현식에는 **`expm1`, `log1p`를 사용하세요**.
7. 오버플로/언더플로를 피하려면 계산 전에 **데이터를 스케일링하세요**.

```python
# Rule 1: Never use == for floats
x = 0.1 + 0.2
print(f"0.1 + 0.2 == 0.3: {x == 0.3}")
print(f"np.isclose(0.1+0.2, 0.3): {np.isclose(x, 0.3)}")

# Rule 4: Solve Ax=b, don't compute A^{-1}b
A = np.random.randn(500, 500)
b = np.random.randn(500)

# BAD: x = A^{-1} b
x_inv = np.linalg.inv(A) @ b
res_inv = np.linalg.norm(A @ x_inv - b)

# GOOD: solve directly
x_solve = np.linalg.solve(A, b)
res_solve = np.linalg.norm(A @ x_solve - b)

print(f"\nUsing A^(-1)b: residual = {res_inv:.2e}")
print(f"Using solve:   residual = {res_solve:.2e}")

# Rule 7: Scaling
# Problem: compute x^T A x when x has entries of very different magnitude
x_unscaled = np.array([1e15, 1e-15])
A_small = np.array([[1, 0.5], [0.5, 1]])

# Without scaling (overflow risk)
Q_unscaled = x_unscaled @ A_small @ x_unscaled
print(f"\nUnscaled Q: {Q_unscaled:.6e}")

# With scaling
scale = np.linalg.norm(x_unscaled)
x_scaled = x_unscaled / scale
Q_scaled = (scale**2) * (x_scaled @ A_small @ x_scaled)
print(f"Scaled Q:   {Q_scaled:.6e}")
```

---

## 연습 문제

### 연습 문제 1: 머신 엡실론 탐구

1. 연속 반감법을 사용하여 `float32`의 머신 엡실론을 계산하는 함수를 작성하세요
2. `1.0 + eps/2`가 `1.0`으로 반올림되지만 `1.0 + eps`는 그렇지 않음을 검증하세요 (계산한 eps에 대해)
3. $x = 1000$과 $x = 0.001$에서 연속 부동소수점 수 사이의 간격을 계산하세요. 어떻게 다릅니까?

### 연습 문제 2: 조건수와 해의 정확도

크기 $n = 2, 4, 6, 8, 10, 12, 14$의 힐베르트 행렬에 대해:

1. 조건수를 계산하세요
2. $x_{\text{true}} = \mathbf{1}$이고 $b = Hx_{\text{true}}$인 $Hx = b$를 풀어보세요
3. 상대 오차 대 $n$을 도표로 그리고 이론적 한계 $\kappa(H) \cdot \epsilon_{\text{mach}}$를 겹쳐 표시하세요

### 연습 문제 3: 파국적 소거

함수 $f(x) = \frac{1 - \cos(x)}{x^2}$은 작은 $x$에서 소거를 겪습니다. 수치적으로 안정한 공식을 사용하여 재작성하고 $x = 10^{-k}$, $k = 1, \ldots, 16$에서 두 방법을 비교하세요.

### 연습 문제 4: 그람-슈미트 비교

고전적 그람-슈미트, 수정 그람-슈미트를 구현하고 NumPy의 하우스홀더 QR을 사용하세요. 조건수 $10^{12}$인 행렬에 대해 다음을 측정하세요:

1. 직교성 손실: $\|Q^TQ - I\|$
2. 인수분해 오차: $\|QR - A\|$

어느 방법이 가장 안정적입니까? 어느 것이 가장 빠릅니까?

### 연습 문제 5: 반복 세분화

중간 정도로 불량 조건화된 행렬($\kappa \approx 10^8$)로 선형 시스템을 다음 방법으로 풀어보세요:

1. float32에서 직접 풀기
2. float64에서 직접 풀기
3. float32 풀기 + 반복 세분화 (잔차는 float64)

각 세분화 단계 후 상대 오차와 잔차를 비교하세요.

---

[이전: 레슨 12](./12_Sparse_Matrices.md) | [개요](./00_Overview.md) | [다음: 레슨 14](./14_Iterative_Methods.md)

**License**: CC BY-NC 4.0
