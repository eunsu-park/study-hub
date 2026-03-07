# 레슨 14: 반복법

[이전: 레슨 13](./13_Numerical_Linear_Algebra.md) | [개요](./00_Overview.md) | [다음: 레슨 15](./15_Tensors_and_Multilinear_Algebra.md)

---

## 학습 목표

- 선형 시스템 풀이에서 반복법이 직접법보다 선호되는 시점을 이해할 수 있습니다
- 고전적 반복법을 구현하고 분석할 수 있습니다: 야코비(Jacobi), 가우스-자이델(Gauss-Seidel), SOR
- 대칭 양정부호 시스템을 위한 켤레 기울기법(CG)을 유도하고 구현할 수 있습니다
- 비대칭 시스템을 위한 크릴로프 부분공간법을 이해할 수 있습니다: GMRES와 BiCGSTAB
- 전처리(preconditioning) 기법을 적용하여 수렴을 가속화할 수 있습니다
- 행렬 특성에 기반하여 적절한 솔버를 선택할 수 있습니다

---

## 1. 직접법 vs 반복법

### 1.1 반복법을 사용하는 시점

**직접법**(LU, Cholesky, QR)은 고정된 연산 횟수로 (반올림 오차까지의) 정확한 해를 계산합니다. **반복법**은 해에 수렴하는 근사의 수열 $x^{(0)}, x^{(1)}, x^{(2)}, \ldots$을 생성합니다.

| 기준 | 직접법 | 반복법 |
|------|--------|--------|
| 행렬 크기 | 소~중 ($n < 10^4$) | 대 ($n > 10^4$) |
| 희소성 | 밀집 또는 보통 | 희소 (희소성 보존) |
| 필요 정확도 | 머신 정밀도 | 조절 가능한 허용 오차 |
| 다중 우변 | 효율적 (한 번 인수분해) | 각 우변을 처음부터 |
| 메모리 | $O(n^2)$ 이상 (채움) | $O(\text{nnz})$ + 소수 벡터 |

```python
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg, gmres, bicgstab
import time
import matplotlib.pyplot as plt

# Compare direct vs iterative for increasing matrix sizes
def build_poisson_2d(n):
    """Build 2D Poisson system (n^2 x n^2 SPD matrix)."""
    T = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n, n))
    I = sparse.eye(n)
    A = -(sparse.kron(I, T) + sparse.kron(T, I))
    return A.tocsr()

sizes = [20, 40, 60, 80, 100]
times_direct = []
times_iterative = []

for n in sizes:
    A = build_poisson_2d(n)
    b = np.random.randn(n**2)

    # Direct solve
    start = time.time()
    x_direct = spsolve(A.tocsc(), b)
    times_direct.append(time.time() - start)

    # Iterative solve (CG)
    start = time.time()
    x_iter, info = cg(A, b, tol=1e-10)
    times_iterative.append(time.time() - start)

    print(f"n={n:3d} ({n**2:5d} unknowns): direct={times_direct[-1]:.4f}s, "
          f"CG={times_iterative[-1]:.4f}s, "
          f"||x_d - x_i||/||x_d|| = {np.linalg.norm(x_direct - x_iter) / np.linalg.norm(x_direct):.2e}")
```

---

## 2. 고전적 반복법

### 2.1 분할 프레임워크

고전적 반복법은 행렬을 $A = M - N$으로 분할하여 다음 반복을 이끌어냅니다:

$$M x^{(k+1)} = N x^{(k)} + b \quad \Rightarrow \quad x^{(k+1)} = M^{-1} N x^{(k)} + M^{-1} b$$

반복이 수렴하는 것은 **스펙트럼 반경**(spectral radius) $\rho(M^{-1}N) < 1$인 것과 동치입니다.

### 2.2 야코비 방법

$A = D - (L + U)$로 분할합니다. 여기서 $D$는 대각, $L$은 순하삼각, $U$는 순상삼각입니다:

$$x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j \neq i} a_{ij} x_j^{(k)} \right)$$

```python
def jacobi(A, b, x0=None, tol=1e-8, max_iter=1000):
    """Jacobi iterative method."""
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    D_inv = 1.0 / A.diagonal()
    residuals = []

    for k in range(max_iter):
        r = b - A @ x
        residual = np.linalg.norm(r)
        residuals.append(residual)

        if residual < tol * np.linalg.norm(b):
            print(f"Jacobi converged in {k+1} iterations")
            return x, residuals

        # x_new = D^{-1} (b - (L+U) x)
        x_new = D_inv * (b - A @ x + A.diagonal() * x)
        x = x_new

    print(f"Jacobi did not converge in {max_iter} iterations")
    return x, residuals

# Test on a diagonally dominant system
n = 50
A = build_poisson_2d(n)
b = np.ones(n**2)

x_jacobi, res_jacobi = jacobi(A, b, tol=1e-8, max_iter=5000)
```

### 2.3 가우스-자이델 방법

모든 이전 값을 사용하는 대신, 가우스-자이델은 갱신된 값을 사용 가능해지면 즉시 사용합니다:

$$x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j < i} a_{ij} x_j^{(k+1)} - \sum_{j > i} a_{ij} x_j^{(k)} \right)$$

```python
def gauss_seidel(A, b, x0=None, tol=1e-8, max_iter=1000):
    """Gauss-Seidel iterative method (works with dense or sparse matrices)."""
    A_dense = A.toarray() if sparse.issparse(A) else A
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    residuals = []

    for k in range(max_iter):
        r = b - (A @ x if sparse.issparse(A) else A_dense @ x)
        residual = np.linalg.norm(r)
        residuals.append(residual)

        if residual < tol * np.linalg.norm(b):
            print(f"Gauss-Seidel converged in {k+1} iterations")
            return x, residuals

        for i in range(n):
            sigma = A_dense[i, :] @ x - A_dense[i, i] * x[i]
            x[i] = (b[i] - sigma) / A_dense[i, i]

    print(f"Gauss-Seidel did not converge in {max_iter} iterations")
    return x, residuals

# Test (smaller system for GS since it is serial)
n_small = 15
A_small = build_poisson_2d(n_small)
b_small = np.ones(n_small**2)

x_gs, res_gs = gauss_seidel(A_small, b_small, tol=1e-8, max_iter=5000)
```

### 2.4 연속 과완화법 (SOR)

SOR은 가우스-자이델을 가속하기 위해 완화 매개변수 $\omega$를 도입합니다:

$$x_i^{(k+1)} = (1 - \omega) x_i^{(k)} + \frac{\omega}{a_{ii}} \left( b_i - \sum_{j < i} a_{ij} x_j^{(k+1)} - \sum_{j > i} a_{ij} x_j^{(k)} \right)$$

$\omega = 1$이면 SOR은 가우스-자이델로 축소됩니다. 최적 $\omega$는 $(1, 2)$에 있으며 야코비 반복 행렬의 스펙트럼 반경에 의존합니다.

```python
def sor(A, b, omega=1.5, x0=None, tol=1e-8, max_iter=1000):
    """Successive Over-Relaxation."""
    A_dense = A.toarray() if sparse.issparse(A) else A
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    residuals = []

    for k in range(max_iter):
        r = b - (A @ x if sparse.issparse(A) else A_dense @ x)
        residual = np.linalg.norm(r)
        residuals.append(residual)

        if residual < tol * np.linalg.norm(b):
            print(f"SOR (omega={omega}) converged in {k+1} iterations")
            return x, residuals

        for i in range(n):
            sigma = A_dense[i, :] @ x - A_dense[i, i] * x[i]
            x_gs = (b[i] - sigma) / A_dense[i, i]
            x[i] = (1 - omega) * x[i] + omega * x_gs

    print(f"SOR did not converge in {max_iter} iterations")
    return x, residuals

# Compare convergence for different omega values
omegas = [0.8, 1.0, 1.2, 1.5, 1.8]
results = {}
for omega in omegas:
    x_sor, res_sor = sor(A_small, b_small, omega=omega, tol=1e-8, max_iter=3000)
    results[omega] = res_sor

# Plot convergence
plt.figure(figsize=(10, 6))
for omega, res in results.items():
    label = f'omega={omega}' + (' (GS)' if omega == 1.0 else '')
    plt.semilogy(res[:500], label=label)
plt.xlabel('Iteration')
plt.ylabel('Residual norm')
plt.title('SOR Convergence for Different omega')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 3. 켤레 기울기법 (CG)

### 3.1 CG 유도

켤레 기울기법(Conjugate Gradient, CG)은 **대칭 양정부호**(SPD) 시스템을 위한 최적의 크릴로프 부분공간법입니다. 크릴로프 부분공간에서 오차의 $A$-노름을 최소화합니다:

$$\mathcal{K}_k(A, r_0) = \text{span}\{r_0, Ar_0, A^2 r_0, \ldots, A^{k-1} r_0\}$$

핵심 통찰: CG는 $i \neq j$일 때 $p_i^T A p_j = 0$을 만족하는 **켤레 방향**(conjugate directions) $p_0, p_1, \ldots$의 수열을 생성합니다.

### 3.2 알고리즘

```
Given: SPD matrix A, right-hand side b, initial guess x_0
r_0 = b - A x_0
p_0 = r_0

For k = 0, 1, 2, ...
    alpha_k = (r_k^T r_k) / (p_k^T A p_k)
    x_{k+1} = x_k + alpha_k * p_k
    r_{k+1} = r_k - alpha_k * A p_k
    beta_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
    p_{k+1} = r_{k+1} + beta_k * p_k

    If ||r_{k+1}|| < tol: stop
```

```python
def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None):
    """Conjugate Gradient method for SPD matrices."""
    n = len(b)
    if max_iter is None:
        max_iter = 2 * n
    x = np.zeros(n) if x0 is None else x0.copy()

    r = b - A @ x
    p = r.copy()
    rs_old = r @ r
    residuals = [np.sqrt(rs_old)]

    for k in range(max_iter):
        Ap = A @ p
        alpha = rs_old / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        residuals.append(np.sqrt(rs_new))

        if np.sqrt(rs_new) < tol * np.linalg.norm(b):
            print(f"CG converged in {k+1} iterations")
            return x, residuals

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    print(f"CG did not converge in {max_iter} iterations")
    return x, residuals

# Test on 2D Poisson system
n = 30
A = build_poisson_2d(n)
b = np.ones(n**2)

x_cg, res_cg = conjugate_gradient(A, b, tol=1e-10)

# Compare with scipy
x_scipy, info = cg(A, b, tol=1e-10)
print(f"||x_cg - x_scipy|| = {np.linalg.norm(x_cg - x_scipy):.2e}")
```

### 3.3 수렴 분석

CG는 (정확한 산술에서) 최대 $n$회 반복에서 수렴합니다. 실전에서 수렴 속도는 **조건수** $\kappa(A) = \lambda_{\max} / \lambda_{\min}$에 의존합니다:

$$\|e_k\|_A \leq 2 \left( \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \right)^k \|e_0\|_A$$

조건수가 작을수록 더 빠른 수렴을 이끕니다.

```python
# Demonstrate CG convergence vs condition number
fig, ax = plt.subplots(figsize=(10, 6))

for n in [10, 20, 40]:
    A = build_poisson_2d(n)
    b = np.ones(n**2)
    cond = np.linalg.cond(A.toarray()) if n <= 20 else None

    x, residuals = conjugate_gradient(A, b, tol=1e-12)

    label = f'n={n} (N={n**2})'
    if cond is not None:
        label += f', kappa={cond:.0f}'
    ax.semilogy(np.array(residuals) / residuals[0], label=label)

ax.set_xlabel('Iteration')
ax.set_ylabel('Relative residual')
ax.set_title('CG Convergence for 2D Poisson Problems')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 4. GMRES (일반화 최소 잔차법)

### 4.1 개요

GMRES는 (SPD만이 아닌) **모든 비특이 행렬**에서 작동합니다. 크릴로프 부분공간에서 잔차의 2-노름을 최소화하는 해를 찾습니다:

$$x_k = \arg\min_{x \in x_0 + \mathcal{K}_k} \|b - Ax\|_2$$

GMRES는 **아르놀디 과정**(Arnoldi process)을 사용하여 크릴로프 부분공간의 정규직교 기저를 구축하고 각 단계에서 소규모 최소제곱 문제를 풉니다.

```python
def gmres_simple(A, b, x0=None, tol=1e-10, max_iter=None):
    """Simplified GMRES (full, no restarts)."""
    n = len(b)
    if max_iter is None:
        max_iter = min(n, 200)

    x = np.zeros(n) if x0 is None else x0.copy()
    r0 = b - A @ x
    beta = np.linalg.norm(r0)
    residuals = [beta]

    if beta < tol:
        return x, residuals

    # Arnoldi vectors
    V = np.zeros((n, max_iter + 1))
    H = np.zeros((max_iter + 1, max_iter))
    V[:, 0] = r0 / beta

    for j in range(max_iter):
        # Arnoldi step
        w = A @ V[:, j]
        for i in range(j + 1):
            H[i, j] = w @ V[:, i]
            w = w - H[i, j] * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)

        if H[j + 1, j] < 1e-14:
            # Lucky breakdown: exact solution found
            break
        V[:, j + 1] = w / H[j + 1, j]

        # Solve least squares: min ||beta*e1 - H_j*y||
        e1 = np.zeros(j + 2)
        e1[0] = beta
        y, _, _, _ = np.linalg.lstsq(H[:j + 2, :j + 1], e1, rcond=None)
        x_k = x + V[:, :j + 1] @ y
        res = np.linalg.norm(b - A @ x_k)
        residuals.append(res)

        if res < tol * np.linalg.norm(b):
            print(f"GMRES converged in {j+1} iterations")
            return x_k, residuals

    print(f"GMRES did not converge in {max_iter} iterations")
    return x_k, residuals

# Test on non-symmetric system
np.random.seed(42)
n = 100
A_nonsym = sparse.random(n, n, density=0.1, format='csr', random_state=42)
A_nonsym = A_nonsym + 5 * sparse.eye(n)  # Make diagonally dominant
b = np.random.randn(n)

x_gmres, res_gmres = gmres_simple(A_nonsym.toarray(), b, tol=1e-10)

# Compare with scipy
x_scipy, info = gmres(A_nonsym, b, atol=1e-10)
print(f"||x_gmres - x_scipy|| = {np.linalg.norm(x_gmres - x_scipy):.2e}")
```

### 4.2 재시작 GMRES

완전 GMRES는 모든 아르놀디 벡터를 저장하므로 $k$회 반복에 $O(n \cdot k)$ 메모리가 필요합니다. **재시작 GMRES(m)**은 $m$회 반복 후 현재 해를 새 초기 추측으로 사용하여 재시작합니다.

```python
from scipy.sparse.linalg import gmres as scipy_gmres

# GMRES with different restart values
n = 50
A = build_poisson_2d(n)
# Make it non-symmetric by adding a convection term
A_conv = A + 0.5 * sparse.diags([np.ones(n**2 - 1)], [1], shape=(n**2, n**2))
b = np.ones(n**2)

for restart in [10, 20, 50, 100]:
    residuals = []
    def callback(rk):
        residuals.append(rk)

    x, info = scipy_gmres(A_conv, b, restart=restart, atol=1e-10,
                           callback=callback, callback_type='pr_norm')
    print(f"GMRES({restart:3d}): {len(residuals):4d} iterations, "
          f"info={info}, residual={np.linalg.norm(A_conv @ x - b):.2e}")
```

---

## 5. BiCGSTAB

### 5.1 개요

BiCGSTAB (쌍켤레 기울기 안정화법)은 **비대칭 시스템**을 위해 설계되었습니다. BiCG 방법과 안정화를 결합하여 순수 BiCG의 불규칙한 수렴을 방지합니다.

**GMRES 대비 장점**:
- 고정 저장: 반복 횟수에 관계없이 소수의 벡터만 필요
- 재시작 불필요

**단점**:
- 정체하거나 실패할 수 있음
- 최적성 보장 없음 (GMRES와 달리)

```python
# BiCGSTAB example
n = 50
A_conv = build_poisson_2d(n) + 0.5 * sparse.diags(
    [np.ones(n**2 - 1)], [1], shape=(n**2, n**2))
b = np.ones(n**2)

# Collect residual history
res_history_bicgstab = []
def callback_bicgstab(x):
    res_history_bicgstab.append(np.linalg.norm(A_conv @ x - b))

x_bicg, info = bicgstab(A_conv, b, tol=1e-10, callback=callback_bicgstab)
print(f"BiCGSTAB: {len(res_history_bicgstab)} iterations, info={info}")
print(f"Residual: {np.linalg.norm(A_conv @ x_bicg - b):.2e}")
```

### 5.2 솔버 비교

```python
# Compare CG, GMRES, BiCGSTAB on an SPD system
n = 40
A_spd = build_poisson_2d(n)
b = np.ones(n**2)

solvers = {}

# CG (only for SPD)
res_cg = []
def cb_cg(x):
    res_cg.append(np.linalg.norm(A_spd @ x - b))
x_cg, _ = cg(A_spd, b, tol=1e-12, callback=cb_cg)
solvers['CG'] = res_cg

# GMRES
res_gmres = []
def cb_gmres(rk):
    res_gmres.append(rk)
x_gm, _ = scipy_gmres(A_spd, b, atol=1e-12, callback=cb_gmres,
                        callback_type='pr_norm')
solvers['GMRES'] = res_gmres

# BiCGSTAB
res_bicg = []
def cb_bicg(x):
    res_bicg.append(np.linalg.norm(A_spd @ x - b))
x_bi, _ = bicgstab(A_spd, b, tol=1e-12, callback=cb_bicg)
solvers['BiCGSTAB'] = res_bicg

# Plot comparison
plt.figure(figsize=(10, 6))
for name, res in solvers.items():
    plt.semilogy(res, label=f'{name} ({len(res)} iters)')
plt.xlabel('Iteration')
plt.ylabel('Residual norm')
plt.title('Solver Comparison on 2D Poisson (SPD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 6. 전처리 (Preconditioning)

### 6.1 전처리가 필요한 이유

전처리는 시스템 $Ax = b$를 더 작은 조건수를 가진 동치 시스템으로 변환합니다. $Ax = b$를 푸는 대신:

$$M^{-1}Ax = M^{-1}b \quad \text{(좌 전처리)}$$

를 풉니다. 좋은 전처리기 $M$은 다음을 만족합니다:
1. $M \approx A$ ($M^{-1}A \approx I$이므로 $\kappa(M^{-1}A) \approx 1$)
2. $Mz = r$를 푸는 것이 저비용

### 6.2 야코비 (대각) 전처리기

가장 단순한 전처리기: $M = \text{diag}(A)$.

```python
from scipy.sparse.linalg import LinearOperator

def jacobi_preconditioner(A):
    """Diagonal (Jacobi) preconditioner."""
    d = A.diagonal()
    d_inv = 1.0 / d
    n = A.shape[0]
    return LinearOperator((n, n), matvec=lambda x: d_inv * x)

# Unpreconditioned CG
n = 50
A = build_poisson_2d(n)
b = np.ones(n**2)

res_unprecond = []
def cb1(x):
    res_unprecond.append(np.linalg.norm(A @ x - b))
cg(A, b, tol=1e-10, callback=cb1)

# Jacobi preconditioned CG
M_jacobi = jacobi_preconditioner(A)
res_jacobi = []
def cb2(x):
    res_jacobi.append(np.linalg.norm(A @ x - b))
cg(A, b, tol=1e-10, M=M_jacobi, callback=cb2)

print(f"Unpreconditioned CG: {len(res_unprecond)} iterations")
print(f"Jacobi PCG:          {len(res_jacobi)} iterations")
```

### 6.3 불완전 LU (ILU) 전처리기

ILU는 $A$의 희소 패턴과 일치하는 비영 원소만 유지하는 근사 LU 인수분해(ILU(0))를 계산하거나, 제한된 채움을 허용합니다(ILUT).

```python
from scipy.sparse.linalg import spilu

# ILU preconditioner
n = 50
A = build_poisson_2d(n)
A_csc = A.tocsc()
b = np.ones(n**2)

# Compute ILU(0)
ilu = spilu(A_csc, drop_tol=0.0)
M_ilu = LinearOperator(A.shape, matvec=ilu.solve)

# Compare convergence
res_none, res_jac, res_ilu = [], [], []

def cb_none(x):
    res_none.append(np.linalg.norm(A @ x - b))
def cb_jac(x):
    res_jac.append(np.linalg.norm(A @ x - b))
def cb_ilu(x):
    res_ilu.append(np.linalg.norm(A @ x - b))

cg(A, b, tol=1e-10, callback=cb_none)
cg(A, b, tol=1e-10, M=jacobi_preconditioner(A), callback=cb_jac)
cg(A, b, tol=1e-10, M=M_ilu, callback=cb_ilu)

print(f"No preconditioner: {len(res_none)} iterations")
print(f"Jacobi:            {len(res_jac)} iterations")
print(f"ILU(0):            {len(res_ilu)} iterations")

# Plot
plt.figure(figsize=(10, 6))
plt.semilogy(res_none, label=f'None ({len(res_none)} iters)')
plt.semilogy(res_jac, label=f'Jacobi ({len(res_jac)} iters)')
plt.semilogy(res_ilu, label=f'ILU(0) ({len(res_ilu)} iters)')
plt.xlabel('Iteration')
plt.ylabel('Residual norm')
plt.title('Effect of Preconditioning on CG Convergence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 6.4 SSOR 전처리기

대칭 연속 과완화(SSOR)는 전진 및 후진 가우스-자이델 소인(sweep)을 완화 매개변수 $\omega$와 결합합니다:

$$M_{\text{SSOR}} = \frac{1}{\omega(2-\omega)} (D + \omega L) D^{-1} (D + \omega U)$$

```python
def ssor_preconditioner(A, omega=1.0):
    """SSOR preconditioner (dense implementation for clarity)."""
    A_dense = A.toarray() if sparse.issparse(A) else A
    n = A_dense.shape[0]
    D = np.diag(np.diag(A_dense))
    L = np.tril(A_dense, -1)
    U = np.triu(A_dense, 1)

    # M = (D + omega*L) D^{-1} (D + omega*U) / (omega*(2-omega))
    D_inv = np.diag(1.0 / np.diag(A_dense))
    M1 = D + omega * L
    M2 = D + omega * U
    scale = omega * (2 - omega)

    def solve(r):
        # Forward solve: (D + omega*L) y = r
        y = np.linalg.solve(M1, r)
        # Scale: z = D y
        z = D @ y
        # Backward solve: (D + omega*U) x = z
        x = np.linalg.solve(M2, z)
        return x / scale

    return LinearOperator((n, n), matvec=solve)
```

---

## 7. 수렴 기준

### 7.1 정지 기준

반복법은 정확도와 계산 비용의 균형을 맞추기 위해 적절한 정지 기준이 필요합니다:

```python
def cg_with_monitoring(A, b, tol=1e-10, max_iter=1000):
    """CG with detailed convergence monitoring."""
    n = len(b)
    x = np.zeros(n)
    r = b.copy()
    p = r.copy()
    rs_old = r @ r
    b_norm = np.linalg.norm(b)

    history = {
        'residual': [],
        'relative_residual': [],
        'a_norm_error': [],
    }

    # Compute exact solution for error tracking
    x_exact = spsolve(A.tocsc(), b)

    for k in range(max_iter):
        Ap = A @ p
        alpha = rs_old / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r

        # Monitor different quantities
        res_norm = np.sqrt(rs_new)
        error = x - x_exact
        a_norm_error = np.sqrt(error @ (A @ error))

        history['residual'].append(res_norm)
        history['relative_residual'].append(res_norm / b_norm)
        history['a_norm_error'].append(a_norm_error)

        # Stopping criterion: relative residual
        if res_norm / b_norm < tol:
            print(f"Converged in {k+1} iterations")
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x, history

n = 30
A = build_poisson_2d(n)
b = np.random.randn(n**2)

x, history = cg_with_monitoring(A, b, tol=1e-12)

# Plot different convergence measures
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(history['residual'], label='||r_k||')
ax.semilogy(history['relative_residual'], label='||r_k||/||b||')
ax.semilogy(history['a_norm_error'], label='||e_k||_A')
ax.set_xlabel('Iteration')
ax.set_ylabel('Value')
ax.set_title('CG Convergence Measures')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 8. 올바른 솔버 선택

### 8.1 결정 흐름도

```
행렬이 대칭인가?
+-- 예: 양정부호인가?
|   +-- 예 -> CG (+ 전처리기)
|   +-- 아니오 -> MINRES
+-- 아니오: 정방인가?
    +-- 예 -> GMRES 또는 BiCGSTAB
    |   +-- 최적성 필요? -> GMRES
    |   +-- 고정 메모리 필요? -> BiCGSTAB
    +-- 아니오 -> LSQR 또는 LSMR (최소제곱)
```

### 8.2 요약 표

| 솔버 | 행렬 유형 | 메모리 | 수렴 | 비고 |
|------|----------|--------|------|------|
| CG | SPD | $O(n)$ | 단조 | SPD에 최적 |
| MINRES | 대칭 부정부호 | $O(n)$ | 단조 | 안장점 문제용 |
| GMRES | 일반 | $O(nk)$ | 최적 | 큰 $k$에는 재시작 |
| BiCGSTAB | 일반 | $O(n)$ | 불규칙 | 재시작 불필요 |
| LSQR | 직사각 | $O(n)$ | 단조 | 최소제곱용 |

```python
# Demonstrate solver selection
from scipy.sparse.linalg import minres, lsqr

# Case 1: SPD matrix -> CG
A_spd = build_poisson_2d(20)
b = np.ones(400)
x_cg, info = cg(A_spd, b, tol=1e-10)
print(f"SPD system with CG: info={info}")

# Case 2: Non-symmetric -> GMRES
A_nonsym = A_spd + 0.3 * sparse.random(400, 400, density=0.01, format='csr')
x_gm, info = scipy_gmres(A_nonsym, b, atol=1e-10)
print(f"Non-symmetric with GMRES: info={info}")

# Case 3: Rectangular (overdetermined) -> LSQR
m, n_cols = 500, 100
A_rect = sparse.random(m, n_cols, density=0.05, format='csr')
b_rect = np.random.randn(m)
result = lsqr(A_rect, b_rect)
x_lsqr = result[0]
print(f"Least squares with LSQR: ||Ax-b|| = {np.linalg.norm(A_rect @ x_lsqr - b_rect):.4f}")
```

---

## 연습 문제

### 연습 문제 1: 야코비 수렴

대각에 4, 하위 및 상위 대각에 -1을 가진 $100 \times 100$ 삼대각 행렬의 시스템 $Ax = b$에 대해 야코비 방법을 구현하세요. 수렴(잔차 vs. 반복)을 도표로 그리세요. 야코비 반복 행렬의 스펙트럼 반경을 계산하고 관찰된 수렴 속도와 연결하세요.

### 연습 문제 2: CG 구현

CG를 처음부터 구현하고 크기 $50 \times 50$의 2D 포아송 시스템에서 `scipy.sparse.linalg.cg`와 비교 검증하세요. 오차의 A-노름 $\|e_k\|_A = \sqrt{(x_k - x^*)^T A (x_k - x^*)}$를 추적하고 단조 감소를 검증하세요.

### 연습 문제 3: 전처리기 비교

크기 $80 \times 80$의 2D 포아송 시스템에서 다음 전처리기를 CG와 비교하세요:

1. 전처리기 없음
2. 야코비 (대각 스케일링)
3. ILU(0) (영 채움 불완전 LU)

각각의 반복 횟수와 총 시간을 보고하세요. 어느 것이 가장 효과적입니까?

### 연습 문제 4: GMRES vs BiCGSTAB

2D 포아송 시스템에 대류항을 추가하여 비대칭 대류-확산 시스템을 생성하세요. GMRES(20), GMRES(50), BiCGSTAB을 반복 횟수, 시간, 최종 잔차 측면에서 비교하세요.

### 연습 문제 5: 최적 SOR 매개변수

대각에 2, 부대각에 -1을 가진 $n \times n$ 삼대각 행렬에서 최적 SOR 매개변수는 해석적으로 알려져 있습니다: $\omega^* = 2 / (1 + \sin(\pi / (n+1)))$. $n = 50$에 대해 $\omega$ 값을 1.0에서 1.99까지 변화시키며 SOR을 실행하고 수렴까지의 반복 횟수 vs. $\omega$를 도표로 그려 이를 실험적으로 검증하세요.

---

[이전: 레슨 13](./13_Numerical_Linear_Algebra.md) | [개요](./00_Overview.md) | [다음: 레슨 15](./15_Tensors_and_Multilinear_Algebra.md)

**License**: CC BY-NC 4.0
