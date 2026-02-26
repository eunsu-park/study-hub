# 11. 라플라스와 포아송 방정식 (Laplace and Poisson Equations)

## 학습 목표
- 타원형 PDE의 특성 이해
- 5점 스텐실 유한차분법 구현
- 반복법 (Jacobi, Gauss-Seidel, SOR) 학습
- 수렴 분석 및 최적화
- scipy.sparse를 이용한 효율적 구현

---

**이 레슨이 중요한 이유:** 라플라스(Laplace)와 포아송(Poisson) 방정식은 시스템이 정상 상태(Steady State)에 도달할 때마다 나타납니다. 시간에 따라 전진하는 포물선형 및 쌍곡선형 PDE와 달리, 타원형 방정식은 영역 내 모든 점이 다른 모든 점에 동시에 영향을 미치는 평형 배치를 기술합니다. 이 전역적 결합은 시간 스텝핑으로 해를 구할 수 없음을 의미하며, 대신 전체 영역을 동시에 풀어야 합니다. 이는 자연스럽게 대규모 희소 선형 시스템과 반복 솔버로 이어집니다.

## 1. 라플라스/포아송 방정식 이론

### 1.1 정의

라플라스 방정식은 소스가 없는 정상 상태 현상을 기술하고, 포아송 방정식은 소스 항 $f(x,y)$를 포함합니다:

$$\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0 \quad \text{(라플라스)}$$

$$\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = f(x,y) \quad \text{(포아송)}$$

라플라스 방정식의 핵심 성질은 **평균값 성질(Mean Value Property)**입니다: 임의의 내부 점에서의 값은 이웃 값들의 평균과 같습니다. 이것이 바로 야코비(Jacobi) 반복법이 수치적으로 활용하는 성질입니다.

```
라플라스 방정식 (동차):
∇²u = ∂²u/∂x² + ∂²u/∂y² = 0

포아송 방정식 (비동차):
∇²u = ∂²u/∂x² + ∂²u/∂y² = f(x, y)
```

### 1.2 물리적 응용

| 분야 | 방정식 | u의 의미 |
|------|--------|----------|
| 정상 열전도 | ∇²T = 0 | 온도 |
| 정전기학 | ∇²φ = -ρ/ε | 전위 |
| 유체 역학 | ∇²ψ = -ω | 유동함수 |
| 중력장 | ∇²φ = 4πGρ | 중력 퍼텐셜 |
| 막의 변형 | ∇²w = p/T | 변위 |

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def elliptic_pde_examples():
    """타원형 PDE 예제"""
    print("="*60)
    print("타원형 PDE (라플라스/포아송) 응용")
    print("="*60)

    print("\n[1] 정상 열전도")
    print("    -k∇²T = Q (열원)")
    print("    경계에서 온도가 고정되면 내부 온도 분포 결정")

    print("\n[2] 정전기학")
    print("    ∇²φ = -ρ/ε₀")
    print("    전하 분포가 주어지면 전위 φ 결정")

    print("\n[3] 탄성 막")
    print("    T∇²w = -p (w: 변위, T: 장력, p: 압력)")
    print("    경계가 고정된 막에 균일 압력 적용")

elliptic_pde_examples()
```

### 1.3 최대값 원리

라플라스 방정식의 해는 영역 내부에서 극값을 가지지 않습니다.
(최대/최소는 경계에서만 발생)

```python
def maximum_principle_demo():
    """최대값 원리 시연"""
    # 해석해: u(x,y) = x² - y² (조화 함수)
    x = np.linspace(-1, 1, 50)
    y = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(x, y)

    u = X**2 - Y**2  # 라플라시안 = 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 등고선도
    ax1 = axes[0]
    c = ax1.contourf(X, Y, u, levels=20, cmap='RdBu_r')
    plt.colorbar(c, ax=ax1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('u(x,y) = x² - y² (조화 함수)')
    ax1.set_aspect('equal')

    # 경계와 내부 값 비교
    ax2 = axes[1]
    # 경계 값들
    boundary_values = []
    # 위/아래 경계
    boundary_values.extend(u[0, :].tolist())
    boundary_values.extend(u[-1, :].tolist())
    # 좌/우 경계
    boundary_values.extend(u[:, 0].tolist())
    boundary_values.extend(u[:, -1].tolist())

    # 내부 값들
    interior_values = u[1:-1, 1:-1].flatten()

    ax2.hist(interior_values, bins=30, alpha=0.7, label='내부', density=True)
    ax2.axvline(x=np.min(boundary_values), color='r', linestyle='--',
               label=f'경계 최소: {np.min(boundary_values):.2f}')
    ax2.axvline(x=np.max(boundary_values), color='g', linestyle='--',
               label=f'경계 최대: {np.max(boundary_values):.2f}')
    ax2.set_xlabel('u 값')
    ax2.set_ylabel('빈도')
    ax2.set_title('최대값 원리: 내부값은 경계값 사이')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('maximum_principle.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"경계 값 범위: [{np.min(boundary_values):.2f}, {np.max(boundary_values):.2f}]")
    print(f"내부 값 범위: [{np.min(interior_values):.2f}, {np.max(interior_values):.2f}]")

# maximum_principle_demo()
```

---

## 2. 5점 스텐실 (Five-Point Stencil)

### 2.1 이산화

2D 라플라시안의 중심차분:

```
∇²u ≈ (u_{i+1,j} - 2u_{i,j} + u_{i-1,j})/Δx²
     + (u_{i,j+1} - 2u_{i,j} + u_{i,j-1})/Δy²

등간격 격자 (Δx = Δy = h):
∇²u ≈ (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}) / h²
```

### 2.2 스텐실 시각화

```
           [i, j+1]
              |
              |
  [i-1, j]---[i, j]---[i+1, j]
              |
              |
           [i, j-1]

계수: 이웃 4개 = 1, 중심 = -4
```

### 2.3 행렬 형태

포아송 방정식 ∇²u = f를 이산화하면 선형 시스템 Au = b를 얻습니다.

```python
from scipy.sparse import diags, kron, eye, csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np

def create_laplacian_2d(nx, ny, dx, dy):
    """
    2D 라플라시안 행렬 생성

    Parameters:
    -----------
    nx, ny : int - 각 방향 격자점 수 (경계 포함)
    dx, dy : float - 격자 간격

    Returns:
    --------
    A : sparse matrix - 라플라시안 행렬 (내부점에 대해)
    """
    # 내부점 수
    mx = nx - 2
    my = ny - 2
    n = mx * my

    # 1D 라플라시안 연산자
    # d²/dx² ≈ (1, -2, 1) / dx²
    Dxx = diags([1, -2, 1], [-1, 0, 1], shape=(mx, mx)) / dx**2
    Dyy = diags([1, -2, 1], [-1, 0, 1], shape=(my, my)) / dy**2

    # 2D 라플라시안: 크로네커 곱
    # ∇² = d²/dx² ⊗ I_y + I_x ⊗ d²/dy²
    Ix = eye(mx)
    Iy = eye(my)

    L = kron(Iy, Dxx) + kron(Dyy, Ix)

    return L.tocsr()


def create_laplacian_2d_explicit(nx, ny, h):
    """
    2D 라플라시안 행렬 (명시적 구성, 등간격)

    내부점 인덱싱: k = (j-1)*mx + (i-1)
    """
    mx = nx - 2
    my = ny - 2
    n = mx * my

    # COO 형식으로 구성
    rows = []
    cols = []
    data = []

    for j in range(my):
        for i in range(mx):
            k = j * mx + i  # 현재 점의 1D 인덱스

            # 중심점: -4
            rows.append(k)
            cols.append(k)
            data.append(-4.0 / h**2)

            # 왼쪽 이웃 (i-1, j)
            if i > 0:
                rows.append(k)
                cols.append(k - 1)
                data.append(1.0 / h**2)

            # 오른쪽 이웃 (i+1, j)
            if i < mx - 1:
                rows.append(k)
                cols.append(k + 1)
                data.append(1.0 / h**2)

            # 아래 이웃 (i, j-1)
            if j > 0:
                rows.append(k)
                cols.append(k - mx)
                data.append(1.0 / h**2)

            # 위 이웃 (i, j+1)
            if j < my - 1:
                rows.append(k)
                cols.append(k + mx)
                data.append(1.0 / h**2)

    from scipy.sparse import coo_matrix
    A = coo_matrix((data, (rows, cols)), shape=(n, n))

    return A.tocsr()


# 테스트
nx = ny = 5
h = 1.0 / (nx - 1)

A1 = create_laplacian_2d(nx, ny, h, h)
A2 = create_laplacian_2d_explicit(nx, ny, h)

print(f"라플라시안 행렬 크기: {A1.shape}")
print(f"비영 요소 수: {A1.nnz}")
print(f"\n행렬 비교 오차: {np.max(np.abs(A1 - A2))}")
```

---

## 3. 직접 풀이법

### 3.1 희소 행렬 직접 풀이

```python
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve

class PoissonSolverDirect:
    """
    2D 포아송 방정식 직접 풀이

    ∇²u = f(x, y)
    """

    def __init__(self, Lx=1.0, Ly=1.0, nx=51, ny=51):
        """
        Parameters:
        -----------
        Lx, Ly : float - 영역 크기
        nx, ny : int - 격자점 수
        """
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny

        # 격자 생성
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # 라플라시안 행렬
        self.A = create_laplacian_2d(nx, ny, self.dx, self.dy)

        print(f"포아송 직접 풀이기 설정")
        print(f"  격자: {nx} x {ny}")
        print(f"  내부점 수: {(nx-2)*(ny-2)}")

    def solve(self, f_func, bc_func):
        """
        포아송 방정식 풀이

        Parameters:
        -----------
        f_func : callable - 소스항 f(x, y)
        bc_func : callable - 경계조건 u(x, y) at boundary

        Returns:
        --------
        u : 2D array - 해
        """
        mx = self.nx - 2
        my = self.ny - 2

        # 소스항 (내부점)
        X_inner = self.X[1:-1, 1:-1]
        Y_inner = self.Y[1:-1, 1:-1]
        f = f_func(X_inner, Y_inner).flatten()

        # 경계조건 기여
        b = f.copy()

        # 경계조건 적용
        u_bc = bc_func(self.X, self.Y)

        # 아래 경계 (j=0)
        b[:mx] -= u_bc[0, 1:-1] / self.dy**2

        # 위 경계 (j=ny-1)
        b[-mx:] -= u_bc[-1, 1:-1] / self.dy**2

        # 좌측 경계 (i=0)
        for j in range(my):
            b[j*mx] -= u_bc[j+1, 0] / self.dx**2

        # 우측 경계 (i=nx-1)
        for j in range(my):
            b[j*mx + mx - 1] -= u_bc[j+1, -1] / self.dx**2

        # 선형 시스템 풀이
        u_inner = spsolve(self.A, b)

        # 전체 해 재구성
        u = u_bc.copy()
        u[1:-1, 1:-1] = u_inner.reshape((my, mx))

        return u


def demo_poisson_direct():
    """포아송 직접 풀이 데모"""
    # 문제: ∇²u = -2π²sin(πx)sin(πy)
    # 경계조건: u = 0
    # 해석해: u(x,y) = sin(πx)sin(πy)

    solver = PoissonSolverDirect(Lx=1.0, Ly=1.0, nx=51, ny=51)

    def f_func(X, Y):
        return -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)

    def bc_func(X, Y):
        return np.zeros_like(X)

    u = solver.solve(f_func, bc_func)

    # 해석해
    u_exact = np.sin(np.pi * solver.X) * np.sin(np.pi * solver.Y)

    # 오차
    error = np.max(np.abs(u - u_exact))
    print(f"\n최대 오차: {error:.2e}")

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    c1 = axes[0].contourf(solver.X, solver.Y, u, levels=30, cmap='viridis')
    plt.colorbar(c1, ax=axes[0])
    axes[0].set_title('수치해')
    axes[0].set_aspect('equal')

    c2 = axes[1].contourf(solver.X, solver.Y, u_exact, levels=30, cmap='viridis')
    plt.colorbar(c2, ax=axes[1])
    axes[1].set_title('해석해')
    axes[1].set_aspect('equal')

    c3 = axes[2].contourf(solver.X, solver.Y, np.abs(u - u_exact), levels=30, cmap='hot')
    plt.colorbar(c3, ax=axes[2])
    axes[2].set_title(f'오차 (최대: {error:.2e})')
    axes[2].set_aspect('equal')

    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig('poisson_direct.png', dpi=150, bbox_inches='tight')
    plt.show()

    return solver, u, u_exact

# solver, u, u_exact = demo_poisson_direct()
```

---

## 4. 반복법 (Iterative Methods)

### 4.1 반복법 개요

대규모 시스템에서 직접 풀이보다 반복법이 효율적일 수 있습니다.

```
Au = b 를 u^(k+1) = Mu^(k) + c 형태로 변환

수렴 조건: 반복 행렬 M의 스펙트럼 반경 ρ(M) < 1
```

### 4.2 Jacobi 반복법

각 점의 새 값을 이웃들의 이전 값으로 계산:

```
u_{i,j}^(k+1) = (1/4) · (u_{i+1,j}^(k) + u_{i-1,j}^(k) + u_{i,j+1}^(k) + u_{i,j-1}^(k) - h²f_{i,j})
```

```python
class JacobiSolver:
    """
    Jacobi 반복법으로 포아송/라플라스 방정식 풀이
    """

    def __init__(self, Lx=1.0, Ly=1.0, nx=51, ny=51):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny

        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # 등간격 가정
        assert abs(self.dx - self.dy) < 1e-10, "등간격 격자 필요"
        self.h = self.dx

    def solve(self, f, u_bc, tol=1e-6, max_iter=10000, verbose=True):
        """
        Jacobi 반복법 실행

        Parameters:
        -----------
        f : 2D array - 소스항
        u_bc : 2D array - 경계조건이 설정된 초기값
        tol : float - 수렴 허용 오차
        max_iter : int - 최대 반복 횟수

        Returns:
        --------
        u : 2D array - 해
        residuals : list - 반복별 잔차
        """
        u = u_bc.copy()
        u_new = u.copy()
        h2 = self.h ** 2

        residuals = []

        for k in range(max_iter):
            # Jacobi 업데이트 (내부점만)
            u_new[1:-1, 1:-1] = 0.25 * (
                u[1:-1, 2:] + u[1:-1, :-2] +  # 좌우 이웃
                u[2:, 1:-1] + u[:-2, 1:-1] -  # 상하 이웃
                h2 * f[1:-1, 1:-1]            # 소스항
            )

            # 잔차 계산
            residual = np.max(np.abs(u_new - u))
            residuals.append(residual)

            # 수렴 확인
            if residual < tol:
                if verbose:
                    print(f"Jacobi 수렴: {k+1} 반복, 잔차 = {residual:.2e}")
                return u_new, residuals

            u = u_new.copy()

        if verbose:
            print(f"Jacobi: 최대 반복 도달, 잔차 = {residuals[-1]:.2e}")

        return u_new, residuals


def demo_jacobi():
    """Jacobi 반복법 데모"""
    solver = JacobiSolver(Lx=1.0, Ly=1.0, nx=51, ny=51)

    # 라플라스 방정식 (f = 0)
    # 경계조건: 위쪽 온도 100, 나머지 0
    f = np.zeros((solver.ny, solver.nx))

    u_bc = np.zeros((solver.ny, solver.nx))
    u_bc[-1, :] = 100  # 위쪽 경계

    u, residuals = solver.solve(f, u_bc, tol=1e-6, max_iter=10000)

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 해
    c = axes[0].contourf(solver.X, solver.Y, u, levels=30, cmap='hot')
    plt.colorbar(c, ax=axes[0], label='온도')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('라플라스 방정식 해 (Jacobi)')
    axes[0].set_aspect('equal')

    # 수렴 이력
    axes[1].semilogy(residuals, 'b-')
    axes[1].set_xlabel('반복 횟수')
    axes[1].set_ylabel('잔차')
    axes[1].set_title(f'Jacobi 수렴 ({len(residuals)} 반복)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('jacobi_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

    return solver, u, residuals

# solver, u, residuals = demo_jacobi()
```

### 4.3 Gauss-Seidel 반복법

새로 계산된 값을 즉시 사용:

```
u_{i,j}^(k+1) = (1/4) · (u_{i+1,j}^(k) + u_{i-1,j}^(k+1) + u_{i,j+1}^(k) + u_{i,j-1}^(k+1) - h²f_{i,j})
```

```python
class GaussSeidelSolver:
    """
    Gauss-Seidel 반복법
    """

    def __init__(self, Lx=1.0, Ly=1.0, nx=51, ny=51):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny

        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.h = self.dx

    def solve(self, f, u_bc, tol=1e-6, max_iter=10000, verbose=True):
        """
        Gauss-Seidel 반복법 실행
        """
        u = u_bc.copy()
        h2 = self.h ** 2

        residuals = []

        for k in range(max_iter):
            max_change = 0.0

            # Gauss-Seidel 업데이트 (순서대로 즉시 업데이트)
            for j in range(1, self.ny - 1):
                for i in range(1, self.nx - 1):
                    u_old = u[j, i]
                    u[j, i] = 0.25 * (
                        u[j, i+1] + u[j, i-1] +  # 좌우
                        u[j+1, i] + u[j-1, i] -  # 상하
                        h2 * f[j, i]
                    )
                    max_change = max(max_change, abs(u[j, i] - u_old))

            residuals.append(max_change)

            if max_change < tol:
                if verbose:
                    print(f"Gauss-Seidel 수렴: {k+1} 반복, 잔차 = {max_change:.2e}")
                return u, residuals

        if verbose:
            print(f"Gauss-Seidel: 최대 반복 도달, 잔차 = {residuals[-1]:.2e}")

        return u, residuals


def compare_jacobi_gs():
    """Jacobi vs Gauss-Seidel 비교"""
    nx = ny = 51

    # 동일한 문제 설정
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    f = np.zeros((ny, nx))
    u_bc = np.zeros((ny, nx))
    u_bc[-1, :] = 100

    # Jacobi
    jacobi = JacobiSolver(nx=nx, ny=ny)
    u_jacobi, res_jacobi = jacobi.solve(f, u_bc, tol=1e-6, verbose=False)

    # Gauss-Seidel
    gs = GaussSeidelSolver(nx=nx, ny=ny)
    u_gs, res_gs = gs.solve(f, u_bc, tol=1e-6, verbose=False)

    print(f"Jacobi: {len(res_jacobi)} 반복")
    print(f"Gauss-Seidel: {len(res_gs)} 반복")
    print(f"속도 향상: {len(res_jacobi) / len(res_gs):.2f}배")

    # 시각화
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(res_jacobi, 'b-', label=f'Jacobi ({len(res_jacobi)} iter)')
    ax.semilogy(res_gs, 'r-', label=f'Gauss-Seidel ({len(res_gs)} iter)')
    ax.set_xlabel('반복 횟수')
    ax.set_ylabel('잔차')
    ax.set_title('Jacobi vs Gauss-Seidel 수렴 비교')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('jacobi_vs_gs.png', dpi=150, bbox_inches='tight')
    plt.show()

    return res_jacobi, res_gs

# res_jacobi, res_gs = compare_jacobi_gs()
```

### 4.4 SOR (Successive Over-Relaxation)

과완화 파라미터 ω를 도입하여 수렴 가속:

```
u_{i,j}^(k+1) = (1-ω)·u_{i,j}^(k) + ω·(Gauss-Seidel 값)

최적 ω ≈ 2 / (1 + sin(πh))  (h: 격자 간격, 정사각 영역)
```

```python
class SORSolver:
    """
    SOR (Successive Over-Relaxation) 방법
    """

    def __init__(self, Lx=1.0, Ly=1.0, nx=51, ny=51):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny

        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.h = self.dx

        # 최적 완화 파라미터 (정사각 영역)
        self.omega_opt = 2 / (1 + np.sin(np.pi * self.h))
        print(f"SOR 최적 ω = {self.omega_opt:.4f}")

    def solve(self, f, u_bc, omega=None, tol=1e-6, max_iter=10000, verbose=True):
        """
        SOR 반복법 실행

        Parameters:
        -----------
        omega : float - 완화 파라미터 (None이면 최적값 사용)
        """
        if omega is None:
            omega = self.omega_opt

        u = u_bc.copy()
        h2 = self.h ** 2

        residuals = []

        for k in range(max_iter):
            max_change = 0.0

            for j in range(1, self.ny - 1):
                for i in range(1, self.nx - 1):
                    u_old = u[j, i]

                    # Gauss-Seidel 값
                    u_gs = 0.25 * (
                        u[j, i+1] + u[j, i-1] +
                        u[j+1, i] + u[j-1, i] -
                        h2 * f[j, i]
                    )

                    # SOR 업데이트
                    u[j, i] = (1 - omega) * u_old + omega * u_gs

                    max_change = max(max_change, abs(u[j, i] - u_old))

            residuals.append(max_change)

            if max_change < tol:
                if verbose:
                    print(f"SOR (ω={omega:.3f}) 수렴: {k+1} 반복, 잔차 = {max_change:.2e}")
                return u, residuals

        if verbose:
            print(f"SOR: 최대 반복 도달, 잔차 = {residuals[-1]:.2e}")

        return u, residuals


def demo_sor():
    """SOR 데모: 다양한 ω 값 비교"""
    nx = ny = 51

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)

    f = np.zeros((ny, nx))
    u_bc = np.zeros((ny, nx))
    u_bc[-1, :] = 100

    sor_solver = SORSolver(nx=nx, ny=ny)

    omega_values = [1.0, 1.2, 1.5, 1.7, sor_solver.omega_opt, 1.95]
    results = {}

    for omega in omega_values:
        u, res = sor_solver.solve(f, u_bc.copy(), omega=omega, tol=1e-6, verbose=False)
        results[omega] = (u, res)
        print(f"ω = {omega:.3f}: {len(res)} 반복")

    # 시각화
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(omega_values)))

    for idx, omega in enumerate(omega_values):
        u, res = results[omega]
        label = f'ω = {omega:.3f}' + (' (최적)' if abs(omega - sor_solver.omega_opt) < 0.01 else '')
        ax.semilogy(res, color=colors[idx], label=f'{label}: {len(res)} iter', linewidth=2)

    ax.set_xlabel('반복 횟수')
    ax.set_ylabel('잔차')
    ax.set_title('SOR: 완화 파라미터 ω에 따른 수렴')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sor_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results

# results = demo_sor()
```

---

## 5. 수렴 분석

### 5.1 이론적 수렴률

| 방법 | 반복 행렬 스펙트럼 반경 | 수렴률 |
|------|------------------------|--------|
| Jacobi | cos(πh) | 느림 |
| Gauss-Seidel | cos²(πh) | 2배 빠름 |
| SOR (최적) | 1 - 2πh | 훨씬 빠름 |

```python
def convergence_rate_analysis():
    """수렴률 이론 분석"""
    h_values = np.array([1/10, 1/20, 1/40, 1/80, 1/160])

    # 이론적 스펙트럼 반경
    rho_jacobi = np.cos(np.pi * h_values)
    rho_gs = np.cos(np.pi * h_values)**2
    omega_opt = 2 / (1 + np.sin(np.pi * h_values))
    rho_sor = omega_opt - 1  # 최적 SOR

    # 수렴에 필요한 반복 횟수 (잔차를 1e-6으로 줄이기)
    target_reduction = -np.log(1e-6)  # ln(10^6)

    iter_jacobi = target_reduction / (-np.log(rho_jacobi))
    iter_gs = target_reduction / (-np.log(rho_gs))
    iter_sor = target_reduction / (-np.log(rho_sor))

    print("수렴 이론 분석")
    print("=" * 70)
    print(f"{'h':<10} {'n=1/h':<10} {'Jacobi':<12} {'G-S':<12} {'SOR':<12}")
    print("-" * 70)

    for i, h in enumerate(h_values):
        n = int(1/h)
        print(f"{h:<10.4f} {n:<10} {iter_jacobi[i]:<12.0f} {iter_gs[i]:<12.0f} {iter_sor[i]:<12.0f}")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_values = 1 / h_values

    ax1 = axes[0]
    ax1.semilogy(n_values, 1 - rho_jacobi, 'o-', label='Jacobi')
    ax1.semilogy(n_values, 1 - rho_gs, 's-', label='Gauss-Seidel')
    ax1.semilogy(n_values, 1 - rho_sor, '^-', label='SOR (최적)')
    ax1.set_xlabel('n = 1/h')
    ax1.set_ylabel('1 - ρ (수렴 인자)')
    ax1.set_title('스펙트럼 반경')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.loglog(n_values, iter_jacobi, 'o-', label='Jacobi')
    ax2.loglog(n_values, iter_gs, 's-', label='Gauss-Seidel')
    ax2.loglog(n_values, iter_sor, '^-', label='SOR (최적)')
    ax2.loglog(n_values, n_values**2, 'k--', alpha=0.5, label='O(n²)')
    ax2.loglog(n_values, n_values, 'k:', alpha=0.5, label='O(n)')
    ax2.set_xlabel('n = 1/h')
    ax2.set_ylabel('필요 반복 횟수')
    ax2.set_title('수렴에 필요한 반복')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

# convergence_rate_analysis()
```

### 5.2 격자 크기별 비교

```python
def grid_size_comparison():
    """격자 크기별 반복법 비교"""
    grid_sizes = [21, 41, 61, 81]

    results = {'Jacobi': [], 'GS': [], 'SOR': []}

    for n in grid_sizes:
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)

        f = np.zeros((n, n))
        u_bc = np.zeros((n, n))
        u_bc[-1, :] = 100

        # Jacobi
        solver = JacobiSolver(nx=n, ny=n)
        _, res = solver.solve(f, u_bc.copy(), tol=1e-6, verbose=False)
        results['Jacobi'].append(len(res))

        # Gauss-Seidel
        solver = GaussSeidelSolver(nx=n, ny=n)
        _, res = solver.solve(f, u_bc.copy(), tol=1e-6, verbose=False)
        results['GS'].append(len(res))

        # SOR
        solver = SORSolver(nx=n, ny=n)
        _, res = solver.solve(f, u_bc.copy(), tol=1e-6, verbose=False)
        results['SOR'].append(len(res))

        print(f"n={n}: Jacobi={results['Jacobi'][-1]}, GS={results['GS'][-1]}, SOR={results['SOR'][-1]}")

    # 시각화
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(grid_sizes, results['Jacobi'], 'o-', label='Jacobi', linewidth=2, markersize=8)
    ax.loglog(grid_sizes, results['GS'], 's-', label='Gauss-Seidel', linewidth=2, markersize=8)
    ax.loglog(grid_sizes, results['SOR'], '^-', label='SOR (최적)', linewidth=2, markersize=8)

    # 기준선
    n_ref = np.array(grid_sizes)
    ax.loglog(n_ref, 0.5 * n_ref**2, 'k--', alpha=0.5, label='O(n²)')
    ax.loglog(n_ref, 2 * n_ref, 'k:', alpha=0.5, label='O(n)')

    ax.set_xlabel('격자 크기 n')
    ax.set_ylabel('반복 횟수')
    ax.set_title('격자 크기에 따른 수렴 비교')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('grid_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results

# results = grid_size_comparison()
```

---

## 6. 실전 응용 예제

### 6.1 열전도 문제

```python
def heat_conduction_example():
    """2D 정상 열전도 문제"""
    # 정사각형 판, 각 변의 온도가 다름
    nx = ny = 51

    solver = SORSolver(Lx=1.0, Ly=1.0, nx=nx, ny=ny)

    # 소스항: 내부 열원 없음
    f = np.zeros((ny, nx))

    # 경계조건
    u_bc = np.zeros((ny, nx))
    u_bc[0, :] = 0      # 아래: 0°C
    u_bc[-1, :] = 100   # 위: 100°C
    u_bc[:, 0] = 50     # 왼쪽: 50°C
    u_bc[:, -1] = 50    # 오른쪽: 50°C

    # 모서리 처리 (평균)
    u_bc[0, 0] = 25
    u_bc[0, -1] = 25
    u_bc[-1, 0] = 75
    u_bc[-1, -1] = 75

    u, _ = solver.solve(f, u_bc, tol=1e-6)

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 등고선
    c1 = axes[0].contourf(solver.X, solver.Y, u, levels=30, cmap='hot')
    plt.colorbar(c1, ax=axes[0], label='온도 (°C)')
    axes[0].contour(solver.X, solver.Y, u, levels=10, colors='white', alpha=0.5)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('정상 열전도 (온도 분포)')
    axes[0].set_aspect('equal')

    # 열유속 벡터
    # q = -k∇T
    qx, qy = np.gradient(u, solver.dx, solver.dy)
    qx = -qx
    qy = -qy

    skip = 3
    axes[1].contourf(solver.X, solver.Y, u, levels=30, cmap='hot', alpha=0.5)
    axes[1].quiver(solver.X[::skip, ::skip], solver.Y[::skip, ::skip],
                   qx[::skip, ::skip], qy[::skip, ::skip],
                   scale=500, color='blue')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('열유속 벡터')
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('heat_conduction.png', dpi=150, bbox_inches='tight')
    plt.show()

    return solver, u

# solver, u = heat_conduction_example()
```

### 6.2 정전기 문제

```python
def electrostatic_example():
    """정전기 문제: 두 도체 사이의 전위"""
    nx = ny = 81

    solver = SORSolver(Lx=1.0, Ly=1.0, nx=nx, ny=ny)

    # 경계조건: 모든 외부 경계 접지 (0V)
    u_bc = np.zeros((ny, nx))

    # 내부에 두 개의 원형 도체
    # 도체 1: 중심 (0.3, 0.5), 반지름 0.1, 전위 +100V
    # 도체 2: 중심 (0.7, 0.5), 반지름 0.1, 전위 -100V

    def is_inside_conductor(X, Y, cx, cy, r):
        return (X - cx)**2 + (Y - cy)**2 <= r**2

    conductor1_mask = is_inside_conductor(solver.X, solver.Y, 0.3, 0.5, 0.08)
    conductor2_mask = is_inside_conductor(solver.X, solver.Y, 0.7, 0.5, 0.08)

    u_bc[conductor1_mask] = 100
    u_bc[conductor2_mask] = -100

    # 소스항: 전하 없음 (라플라스)
    f = np.zeros((ny, nx))

    # 풀이 (도체 내부는 경계조건으로 고정)
    u = u_bc.copy()

    for _ in range(5000):
        u_new = u.copy()

        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                # 도체 내부는 건너뜀
                if conductor1_mask[j, i] or conductor2_mask[j, i]:
                    continue

                u_new[j, i] = 0.25 * (u[j, i+1] + u[j, i-1] + u[j+1, i] + u[j-1, i])

        if np.max(np.abs(u_new - u)) < 1e-6:
            break

        u = u_new.copy()

    # 전기장 계산: E = -∇φ
    Ey, Ex = np.gradient(u, solver.dy, solver.dx)
    Ex = -Ex
    Ey = -Ey
    E_mag = np.sqrt(Ex**2 + Ey**2)

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 전위
    c1 = axes[0].contourf(solver.X, solver.Y, u, levels=30, cmap='RdBu_r')
    plt.colorbar(c1, ax=axes[0], label='전위 (V)')
    axes[0].contour(solver.X, solver.Y, u, levels=20, colors='k', alpha=0.3)

    # 도체 표시
    theta = np.linspace(0, 2*np.pi, 50)
    axes[0].plot(0.3 + 0.08*np.cos(theta), 0.5 + 0.08*np.sin(theta), 'r-', linewidth=2)
    axes[0].plot(0.7 + 0.08*np.cos(theta), 0.5 + 0.08*np.sin(theta), 'b-', linewidth=2)

    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('전위 분포')
    axes[0].set_aspect('equal')

    # 전기장
    axes[1].contourf(solver.X, solver.Y, E_mag, levels=30, cmap='hot')
    skip = 4
    axes[1].streamplot(solver.X, solver.Y, Ex, Ey, color='white', density=1.5, linewidth=0.5)

    axes[1].plot(0.3 + 0.08*np.cos(theta), 0.5 + 0.08*np.sin(theta), 'r-', linewidth=2)
    axes[1].plot(0.7 + 0.08*np.cos(theta), 0.5 + 0.08*np.sin(theta), 'b-', linewidth=2)

    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('전기장 (유선)')
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('electrostatic.png', dpi=150, bbox_inches='tight')
    plt.show()

    return solver, u

# solver, u = electrostatic_example()
```

---

## 7. 요약

### 반복법 비교

| 방법 | 특징 | 수렴 속도 | 메모리 |
|------|------|----------|--------|
| Jacobi | 병렬화 용이 | O(n²) 반복 | 2배 저장 |
| Gauss-Seidel | 순차 업데이트 | 2배 빠름 | 제자리 |
| SOR | 과완화 가속 | O(n) 반복 | 제자리 |
| 직접법 | 한 번에 풀이 | - | O(n²) 저장 |

### 최적 완화 파라미터

```
ω_opt = 2 / (1 + sin(πh))

예: h = 1/50 → ω_opt ≈ 1.937
```

### 다음 단계

1. **12장**: 이류방정식 - 1차 쌍곡선형 PDE
2. 다중격자법 (Multigrid) - 더 빠른 수렴
3. 켤레기울기법 (CG) - 대규모 시스템

---

## 연습문제

### 연습 1: SOR 최적 ω 탐색
다양한 격자 크기에서 실험적으로 최적 ω를 찾고 이론값과 비교하시오.

### 연습 2: L자형 영역
L자형 영역에서 라플라스 방정식을 풀어보시오.

### 연습 3: 비균질 소스
f(x,y) = sin(2πx)·sin(2πy)인 포아송 방정식을 풀고 해석해와 비교하시오.

### 연습 4: Red-Black Gauss-Seidel
체스판 패턴으로 업데이트하는 Red-Black G-S를 구현하시오.

---

## 참고 자료

1. **교재**: LeVeque, "Finite Difference Methods"
2. **반복법**: Saad, "Iterative Methods for Sparse Linear Systems"
3. **Python**: scipy.sparse, scipy.sparse.linalg
