# 09. 열방정식 (Heat Equation)

## 학습 목표
- 1D/2D 열방정식의 물리적 의미 이해
- FTCS (Forward Time Central Space) 양해법 구현
- BTCS (Backward Time Central Space) 음해법 구현
- Crank-Nicolson 방법 이해 및 구현
- 다양한 경계조건 처리 방법 학습

---

## 1. 열방정식 이론

### 1.1 물리적 배경

열방정식은 열전도 현상을 기술하는 포물선형 PDE입니다.

```
1D 열방정식:
∂u/∂t = α · ∂²u/∂x²

여기서:
- u(x,t): 온도
- α: 열확산계수 (thermal diffusivity)
- x: 공간 좌표
- t: 시간
```

### 1.2 열확산계수

```python
"""
열확산계수 α = k / (ρ·c)

여기서:
- k: 열전도도 (W/m·K)
- ρ: 밀도 (kg/m³)
- c: 비열 (J/kg·K)
"""

# 물질별 열확산계수 (m²/s)
thermal_diffusivity = {
    '구리': 1.11e-4,
    '알루미늄': 9.7e-5,
    '철': 2.3e-5,
    '콘크리트': 7.5e-7,
    '물': 1.43e-7,
    '공기': 2.2e-5,
}

for material, alpha in thermal_diffusivity.items():
    print(f"{material}: α = {alpha:.2e} m²/s")
```

### 1.3 해석해 (분리변수법)

경계조건 u(0,t) = u(L,t) = 0, 초기조건 u(x,0) = f(x)인 경우:

```
u(x,t) = Σ Bₙ · sin(nπx/L) · exp(-α(nπ/L)²t)

Bₙ = (2/L) ∫₀^L f(x)·sin(nπx/L) dx
```

```python
import numpy as np
import matplotlib.pyplot as plt

def exact_solution_heat(x, t, alpha, L, n_terms=50):
    """
    열방정식 해석해 (푸리에 급수)

    초기조건: u(x,0) = sin(πx/L) (첫 번째 모드만)
    경계조건: u(0,t) = u(L,t) = 0
    """
    # 단순 초기조건의 경우
    return np.sin(np.pi * x / L) * np.exp(-alpha * (np.pi / L)**2 * t)

# 시각화
L = 1.0
alpha = 0.01
x = np.linspace(0, L, 101)

fig, ax = plt.subplots(figsize=(10, 6))

times = [0, 0.5, 1.0, 2.0, 5.0]
for t in times:
    u = exact_solution_heat(x, t, alpha, L)
    ax.plot(x, u, label=f't = {t}')

ax.set_xlabel('x')
ax.set_ylabel('u(x,t)')
ax.set_title('1D 열방정식 해석해 (시간에 따른 변화)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig('heat_exact.png', dpi=150)
# plt.show()
```

---

## 2. FTCS 양해법 (Explicit Scheme)

### 2.1 이산화

FTCS = Forward Time, Central Space

```
시간: 전방차분
∂u/∂t ≈ (u_i^{n+1} - u_i^n) / Δt

공간: 중심차분
∂²u/∂x² ≈ (u_{i+1}^n - 2u_i^n + u_{i-1}^n) / Δx²

결합:
u_i^{n+1} = u_i^n + r·(u_{i+1}^n - 2u_i^n + u_{i-1}^n)

여기서 r = α·Δt/Δx² (안정 조건: r ≤ 0.5)
```

### 2.2 FTCS 스텐실 시각화

```
시간 n+1:         [i]
                   ↑
시간 n:    [i-1]--[i]--[i+1]
```

### 2.3 FTCS 구현

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class HeatEquation1D_FTCS:
    """
    1D 열방정식 FTCS 양해법

    ∂u/∂t = α · ∂²u/∂x²
    """

    def __init__(self, L=1.0, alpha=0.01, nx=51, T=1.0, safety=0.4):
        """
        Parameters:
        -----------
        L : float - 영역 길이
        alpha : float - 열확산계수
        nx : int - 공간 격자점 수
        T : float - 최종 시간
        safety : float - CFL 안전계수 (0 < safety ≤ 0.5)
        """
        self.L = L
        self.alpha = alpha
        self.nx = nx
        self.T = T

        # 격자 생성
        self.dx = L / (nx - 1)
        self.x = np.linspace(0, L, nx)

        # 안정 조건에 따른 시간 간격 결정
        self.dt = safety * self.dx**2 / alpha
        self.nt = int(np.ceil(T / self.dt))
        self.dt = T / self.nt  # 정확히 T에 도달하도록 조정

        self.r = alpha * self.dt / self.dx**2

        print(f"FTCS 열방정식 설정")
        print(f"  dx = {self.dx:.4f}, dt = {self.dt:.6f}")
        print(f"  r = α·dt/dx² = {self.r:.4f}")
        print(f"  시간 스텝 수: {self.nt}")
        print(f"  안정성: {'OK' if self.r <= 0.5 else 'WARNING!'}")

    def set_initial_condition(self, func):
        """초기조건 설정"""
        self.u = func(self.x)
        self.u0 = self.u.copy()

    def set_boundary_conditions(self, left_type='dirichlet', left_value=0,
                                  right_type='dirichlet', right_value=0):
        """경계조건 설정"""
        self.bc = {
            'left': {'type': left_type, 'value': left_value},
            'right': {'type': right_type, 'value': right_value}
        }

    def apply_bc(self, u):
        """경계조건 적용"""
        # 왼쪽 경계
        if self.bc['left']['type'] == 'dirichlet':
            u[0] = self.bc['left']['value']
        elif self.bc['left']['type'] == 'neumann':
            # ∂u/∂x = flux => u[0] = u[1] - flux * dx
            u[0] = u[1] - self.bc['left']['value'] * self.dx

        # 오른쪽 경계
        if self.bc['right']['type'] == 'dirichlet':
            u[-1] = self.bc['right']['value']
        elif self.bc['right']['type'] == 'neumann':
            # ∂u/∂x = flux => u[-1] = u[-2] + flux * dx
            u[-1] = u[-2] + self.bc['right']['value'] * self.dx

        return u

    def step(self):
        """한 시간 스텝 진행 (FTCS)"""
        u_new = self.u.copy()

        # 내부점 업데이트
        u_new[1:-1] = self.u[1:-1] + self.r * (
            self.u[2:] - 2*self.u[1:-1] + self.u[:-2]
        )

        # 경계조건 적용
        u_new = self.apply_bc(u_new)

        self.u = u_new

    def solve(self, save_interval=None):
        """전체 시간 구간 풀이"""
        if save_interval is None:
            save_interval = max(1, self.nt // 100)

        history = [self.u.copy()]
        times = [0]

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), np.array(history)


def demo_ftcs():
    """FTCS 데모"""
    # 문제 설정
    solver = HeatEquation1D_FTCS(L=1.0, alpha=0.01, nx=51, T=2.0, safety=0.4)

    # 초기조건: 사인파
    solver.set_initial_condition(lambda x: np.sin(np.pi * x))

    # 경계조건: 양끝 고정
    solver.set_boundary_conditions('dirichlet', 0, 'dirichlet', 0)

    # 풀이
    times, history = solver.solve(save_interval=20)

    # 해석해와 비교
    u_exact = exact_solution_heat(solver.x, times[-1], solver.alpha, solver.L)

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 시간에 따른 해
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
    for i, (t, u) in enumerate(zip(times[::10], history[::10])):
        ax1.plot(solver.x, u, color=colors[i*10] if i*10 < len(colors) else colors[-1],
                label=f't={t:.2f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u')
    ax1.set_title('FTCS 열방정식 해')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 최종 시간에서 비교
    ax2 = axes[1]
    ax2.plot(solver.x, history[-1], 'b-', label='FTCS 수치해', linewidth=2)
    ax2.plot(solver.x, u_exact, 'r--', label='해석해', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('u')
    ax2.set_title(f't = {times[-1]:.2f}에서 해석해 비교')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    error = np.max(np.abs(history[-1] - u_exact))
    print(f"\n최종 시간에서 최대 오차: {error:.2e}")

    plt.tight_layout()
    plt.savefig('heat_ftcs.png', dpi=150, bbox_inches='tight')
    plt.show()

    return solver, times, history

# solver, times, history = demo_ftcs()
```

---

## 3. BTCS 음해법 (Implicit Scheme)

### 3.1 이산화

BTCS = Backward Time, Central Space

```
시간: 후방차분 (n+1 시점에서 평가)
∂u/∂t ≈ (u_i^{n+1} - u_i^n) / Δt

공간: 중심차분 (n+1 시점에서)
∂²u/∂x² ≈ (u_{i+1}^{n+1} - 2u_i^{n+1} + u_{i-1}^{n+1}) / Δx²

정리:
-r·u_{i-1}^{n+1} + (1+2r)·u_i^{n+1} - r·u_{i+1}^{n+1} = u_i^n
```

### 3.2 행렬 형태

```
A · u^{n+1} = u^n

여기서 A는 삼중대각 행렬:
    | 1+2r  -r    0   ...  |
    | -r   1+2r  -r   ...  |
A = |  0    -r  1+2r  ...  |
    | ...               -r |
    |             -r  1+2r |
```

### 3.3 BTCS 구현

```python
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

class HeatEquation1D_BTCS:
    """
    1D 열방정식 BTCS 음해법

    무조건 안정 (unconditionally stable)
    """

    def __init__(self, L=1.0, alpha=0.01, nx=51, T=1.0, nt=100):
        """
        Parameters:
        -----------
        L : float - 영역 길이
        alpha : float - 열확산계수
        nx : int - 공간 격자점 수
        T : float - 최종 시간
        nt : int - 시간 스텝 수
        """
        self.L = L
        self.alpha = alpha
        self.nx = nx
        self.T = T
        self.nt = nt

        # 격자 생성
        self.dx = L / (nx - 1)
        self.dt = T / nt
        self.x = np.linspace(0, L, nx)

        self.r = alpha * self.dt / self.dx**2

        print(f"BTCS 열방정식 설정")
        print(f"  dx = {self.dx:.4f}, dt = {self.dt:.6f}")
        print(f"  r = α·dt/dx² = {self.r:.4f}")
        print(f"  BTCS는 무조건 안정 (r에 제한 없음)")

        # 행렬 A 생성 (내부점만)
        self._build_matrix()

    def _build_matrix(self):
        """BTCS 행렬 생성"""
        n = self.nx - 2  # 내부점 수

        main_diag = (1 + 2*self.r) * np.ones(n)
        off_diag = -self.r * np.ones(n - 1)

        self.A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')

    def set_initial_condition(self, func):
        """초기조건 설정"""
        self.u = func(self.x)
        self.u0 = self.u.copy()

    def set_boundary_conditions(self, left_value=0, right_value=0):
        """디리클레 경계조건 설정"""
        self.u_left = left_value
        self.u_right = right_value

    def step(self):
        """한 시간 스텝 진행 (BTCS)"""
        # 우변 벡터 (내부점)
        b = self.u[1:-1].copy()

        # 경계조건 기여
        b[0] += self.r * self.u_left
        b[-1] += self.r * self.u_right

        # 선형 시스템 풀이
        u_inner = spsolve(self.A, b)

        # 전체 해 업데이트
        self.u[1:-1] = u_inner
        self.u[0] = self.u_left
        self.u[-1] = self.u_right

    def solve(self, save_interval=None):
        """전체 시간 구간 풀이"""
        if save_interval is None:
            save_interval = max(1, self.nt // 100)

        history = [self.u.copy()]
        times = [0]

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), np.array(history)


def compare_ftcs_btcs():
    """FTCS vs BTCS 비교"""
    L = 1.0
    alpha = 0.01
    nx = 51
    T = 2.0

    # FTCS (CFL 제한됨)
    ftcs = HeatEquation1D_FTCS(L, alpha, nx, T, safety=0.4)
    ftcs.set_initial_condition(lambda x: np.sin(np.pi * x))
    ftcs.set_boundary_conditions('dirichlet', 0, 'dirichlet', 0)
    times_ftcs, history_ftcs = ftcs.solve()

    # BTCS (큰 시간 간격 사용 가능)
    btcs = HeatEquation1D_BTCS(L, alpha, nx, T, nt=50)  # 훨씬 적은 시간 스텝
    btcs.set_initial_condition(lambda x: np.sin(np.pi * x))
    btcs.set_boundary_conditions(0, 0)
    times_btcs, history_btcs = btcs.solve()

    # 해석해
    u_exact = exact_solution_heat(ftcs.x, T, alpha, L)

    # 비교
    print(f"\n비교 결과:")
    print(f"  FTCS 시간 스텝 수: {ftcs.nt}")
    print(f"  BTCS 시간 스텝 수: {btcs.nt}")
    print(f"  FTCS 최대 오차: {np.max(np.abs(history_ftcs[-1] - u_exact)):.2e}")
    print(f"  BTCS 최대 오차: {np.max(np.abs(history_btcs[-1] - u_exact)):.2e}")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(ftcs.x, history_ftcs[-1], 'b-', label=f'FTCS (dt={ftcs.dt:.5f})', linewidth=2)
    ax1.plot(btcs.x, history_btcs[-1], 'g--', label=f'BTCS (dt={btcs.dt:.4f})', linewidth=2)
    ax1.plot(ftcs.x, u_exact, 'r:', label='해석해', linewidth=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('u')
    ax1.set_title(f't = {T}에서 비교')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.semilogy(ftcs.x, np.abs(history_ftcs[-1] - u_exact) + 1e-16, 'b-', label='FTCS 오차')
    ax2.semilogy(btcs.x, np.abs(history_btcs[-1] - u_exact) + 1e-16, 'g--', label='BTCS 오차')
    ax2.set_xlabel('x')
    ax2.set_ylabel('|오차|')
    ax2.set_title('수치 오차 비교')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('heat_ftcs_vs_btcs.png', dpi=150, bbox_inches='tight')
    plt.show()

# compare_ftcs_btcs()
```

---

## 4. Crank-Nicolson 방법

### 4.1 이론

Crank-Nicolson = FTCS와 BTCS의 평균 (2차 정확도)

```
(u_i^{n+1} - u_i^n) / Δt = (α/2) · [(∂²u/∂x²)^n + (∂²u/∂x²)^{n+1}]

정리:
-r/2·u_{i-1}^{n+1} + (1+r)·u_i^{n+1} - r/2·u_{i+1}^{n+1}
    = r/2·u_{i-1}^n + (1-r)·u_i^n + r/2·u_{i+1}^n
```

### 4.2 행렬 형태

```
A · u^{n+1} = B · u^n

A: (1+r) 대각선, -r/2 비대각선
B: (1-r) 대각선, r/2 비대각선
```

### 4.3 Crank-Nicolson 구현

```python
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

class HeatEquation1D_CrankNicolson:
    """
    1D 열방정식 Crank-Nicolson 방법

    - 무조건 안정
    - 2차 정확도 (시간, 공간 모두)
    """

    def __init__(self, L=1.0, alpha=0.01, nx=51, T=1.0, nt=100):
        """
        Parameters:
        -----------
        L : float - 영역 길이
        alpha : float - 열확산계수
        nx : int - 공간 격자점 수
        T : float - 최종 시간
        nt : int - 시간 스텝 수
        """
        self.L = L
        self.alpha = alpha
        self.nx = nx
        self.T = T
        self.nt = nt

        # 격자 생성
        self.dx = L / (nx - 1)
        self.dt = T / nt
        self.x = np.linspace(0, L, nx)

        self.r = alpha * self.dt / self.dx**2

        print(f"Crank-Nicolson 열방정식 설정")
        print(f"  dx = {self.dx:.4f}, dt = {self.dt:.6f}")
        print(f"  r = α·dt/dx² = {self.r:.4f}")
        print(f"  2차 정확도 & 무조건 안정")

        # 행렬 생성
        self._build_matrices()

    def _build_matrices(self):
        """Crank-Nicolson 행렬 A, B 생성"""
        n = self.nx - 2  # 내부점 수
        r = self.r

        # A 행렬: 좌변 (암시적 부분)
        main_A = (1 + r) * np.ones(n)
        off_A = (-r/2) * np.ones(n - 1)
        self.A = diags([off_A, main_A, off_A], [-1, 0, 1], format='csr')

        # B 행렬: 우변 (명시적 부분)
        main_B = (1 - r) * np.ones(n)
        off_B = (r/2) * np.ones(n - 1)
        self.B = diags([off_B, main_B, off_B], [-1, 0, 1], format='csr')

    def set_initial_condition(self, func):
        """초기조건 설정"""
        self.u = func(self.x)
        self.u0 = self.u.copy()

    def set_boundary_conditions(self, left_value=0, right_value=0):
        """디리클레 경계조건 설정"""
        self.u_left = left_value
        self.u_right = right_value

    def step(self):
        """한 시간 스텝 진행 (Crank-Nicolson)"""
        r = self.r

        # 우변: B·u^n + 경계조건 기여
        b = self.B @ self.u[1:-1]

        # 경계조건 기여 (좌변과 우변 모두에서)
        b[0] += (r/2) * (self.u_left + self.u_left)  # n과 n+1 시점의 BC
        b[-1] += (r/2) * (self.u_right + self.u_right)

        # 선형 시스템 풀이
        u_inner = spsolve(self.A, b)

        # 전체 해 업데이트
        self.u[1:-1] = u_inner
        self.u[0] = self.u_left
        self.u[-1] = self.u_right

    def solve(self, save_interval=None):
        """전체 시간 구간 풀이"""
        if save_interval is None:
            save_interval = max(1, self.nt // 100)

        history = [self.u.copy()]
        times = [0]

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), np.array(history)


def compare_all_schemes():
    """FTCS, BTCS, Crank-Nicolson 비교"""
    L = 1.0
    alpha = 0.01
    nx = 51
    T = 2.0

    # 동일한 (큰) 시간 간격으로 비교
    nt = 40  # FTCS는 불안정할 것

    # Crank-Nicolson
    cn = HeatEquation1D_CrankNicolson(L, alpha, nx, T, nt)
    cn.set_initial_condition(lambda x: np.sin(np.pi * x))
    cn.set_boundary_conditions(0, 0)
    times_cn, history_cn = cn.solve()

    # BTCS
    btcs = HeatEquation1D_BTCS(L, alpha, nx, T, nt)
    btcs.set_initial_condition(lambda x: np.sin(np.pi * x))
    btcs.set_boundary_conditions(0, 0)
    times_btcs, history_btcs = btcs.solve()

    # FTCS (안정한 설정)
    ftcs = HeatEquation1D_FTCS(L, alpha, nx, T, safety=0.4)
    ftcs.set_initial_condition(lambda x: np.sin(np.pi * x))
    ftcs.set_boundary_conditions('dirichlet', 0, 'dirichlet', 0)
    times_ftcs, history_ftcs = ftcs.solve()

    # 해석해
    u_exact = exact_solution_heat(cn.x, T, alpha, L)

    # 오차 비교
    print(f"\n정확도 비교 (t = {T}):")
    print(f"  FTCS (dt={ftcs.dt:.5f}, {ftcs.nt} steps): {np.max(np.abs(history_ftcs[-1] - u_exact)):.2e}")
    print(f"  BTCS (dt={btcs.dt:.4f}, {btcs.nt} steps): {np.max(np.abs(history_btcs[-1] - u_exact)):.2e}")
    print(f"  C-N  (dt={cn.dt:.4f}, {cn.nt} steps): {np.max(np.abs(history_cn[-1] - u_exact)):.2e}")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(cn.x, history_ftcs[-1], 'b-', label='FTCS', linewidth=2)
    ax1.plot(cn.x, history_btcs[-1], 'g--', label='BTCS', linewidth=2)
    ax1.plot(cn.x, history_cn[-1], 'm:', label='Crank-Nicolson', linewidth=2)
    ax1.plot(cn.x, u_exact, 'r-.', label='해석해', linewidth=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('u')
    ax1.set_title(f't = {T}에서 세 스킴 비교')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 정확도 수렴 테스트
    ax2 = axes[1]
    dt_values = []
    errors_btcs = []
    errors_cn = []

    for nt_test in [20, 40, 80, 160, 320]:
        dt_test = T / nt_test
        dt_values.append(dt_test)

        # BTCS
        solver = HeatEquation1D_BTCS(L, alpha, nx, T, nt_test)
        solver.set_initial_condition(lambda x: np.sin(np.pi * x))
        solver.set_boundary_conditions(0, 0)
        _, hist = solver.solve()
        errors_btcs.append(np.max(np.abs(hist[-1] - u_exact)))

        # Crank-Nicolson
        solver = HeatEquation1D_CrankNicolson(L, alpha, nx, T, nt_test)
        solver.set_initial_condition(lambda x: np.sin(np.pi * x))
        solver.set_boundary_conditions(0, 0)
        _, hist = solver.solve()
        errors_cn.append(np.max(np.abs(hist[-1] - u_exact)))

    ax2.loglog(dt_values, errors_btcs, 'gs-', label='BTCS (1차)', linewidth=2)
    ax2.loglog(dt_values, errors_cn, 'mo-', label='Crank-Nicolson (2차)', linewidth=2)

    # 기준선
    dt_ref = np.array(dt_values)
    ax2.loglog(dt_ref, 0.5*dt_ref, 'k--', alpha=0.5, label='O(Δt)')
    ax2.loglog(dt_ref, 0.5*dt_ref**2, 'k:', alpha=0.5, label='O(Δt²)')

    ax2.set_xlabel('Δt')
    ax2.set_ylabel('최대 오차')
    ax2.set_title('시간 정확도 수렴')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('heat_scheme_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# compare_all_schemes()
```

---

## 5. 2D 열방정식

### 5.1 2D 열방정식

```
∂u/∂t = α · (∂²u/∂x² + ∂²u/∂y²) = α · ∇²u
```

### 5.2 FTCS 2D 구현

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class HeatEquation2D_FTCS:
    """
    2D 열방정식 FTCS 양해법

    ∂u/∂t = α · (∂²u/∂x² + ∂²u/∂y²)
    """

    def __init__(self, Lx=1.0, Ly=1.0, alpha=0.01, nx=51, ny=51, T=0.5, safety=0.2):
        """
        Parameters:
        -----------
        Lx, Ly : float - 영역 크기
        alpha : float - 열확산계수
        nx, ny : int - 격자점 수
        T : float - 최종 시간
        safety : float - CFL 안전계수
        """
        self.Lx = Lx
        self.Ly = Ly
        self.alpha = alpha
        self.nx = nx
        self.ny = ny
        self.T = T

        # 격자 생성
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # 2D CFL 조건: r_x + r_y ≤ 0.5
        dt_cfl = safety * 0.5 / (alpha * (1/self.dx**2 + 1/self.dy**2))
        self.dt = dt_cfl
        self.nt = int(np.ceil(T / self.dt))
        self.dt = T / self.nt

        self.rx = alpha * self.dt / self.dx**2
        self.ry = alpha * self.dt / self.dy**2

        print(f"2D FTCS 열방정식 설정")
        print(f"  격자: {nx} x {ny}")
        print(f"  dx = {self.dx:.4f}, dy = {self.dy:.4f}, dt = {self.dt:.6f}")
        print(f"  r_x = {self.rx:.4f}, r_y = {self.ry:.4f}")
        print(f"  r_x + r_y = {self.rx + self.ry:.4f} (≤ 0.5이어야 안정)")

    def set_initial_condition(self, func):
        """초기조건 설정: u(x,y,0) = func(X, Y)"""
        self.u = func(self.X, self.Y)
        self.u0 = self.u.copy()

    def set_boundary_conditions(self, bc_value=0):
        """디리클레 경계조건 (모든 경계에서 동일 값)"""
        self.bc_value = bc_value

    def apply_bc(self, u):
        """경계조건 적용"""
        u[0, :] = self.bc_value   # 아래
        u[-1, :] = self.bc_value  # 위
        u[:, 0] = self.bc_value   # 왼쪽
        u[:, -1] = self.bc_value  # 오른쪽
        return u

    def step(self):
        """한 시간 스텝 진행 (2D FTCS)"""
        u_new = self.u.copy()

        # 내부점 업데이트
        u_new[1:-1, 1:-1] = self.u[1:-1, 1:-1] + \
            self.rx * (self.u[1:-1, 2:] - 2*self.u[1:-1, 1:-1] + self.u[1:-1, :-2]) + \
            self.ry * (self.u[2:, 1:-1] - 2*self.u[1:-1, 1:-1] + self.u[:-2, 1:-1])

        # 경계조건
        u_new = self.apply_bc(u_new)

        self.u = u_new

    def solve(self, save_interval=None):
        """전체 시간 구간 풀이"""
        if save_interval is None:
            save_interval = max(1, self.nt // 50)

        history = [self.u.copy()]
        times = [0]

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), history


def demo_heat_2d():
    """2D 열방정식 데모"""
    # 문제 설정
    solver = HeatEquation2D_FTCS(Lx=1.0, Ly=1.0, alpha=0.01, nx=51, ny=51, T=0.5)

    # 초기조건: 가우시안 열점
    def ic(X, Y):
        x0, y0 = 0.5, 0.5
        sigma = 0.1
        return np.exp(-((X-x0)**2 + (Y-y0)**2) / (2*sigma**2))

    solver.set_initial_condition(ic)
    solver.set_boundary_conditions(0)

    # 풀이
    times, history = solver.solve(save_interval=20)

    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 선택된 시간에서의 해
    plot_indices = [0, len(times)//4, len(times)//2, 3*len(times)//4, -1]
    for idx, i in enumerate(plot_indices[:5]):
        if idx < 5:
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            c = ax.contourf(solver.X, solver.Y, history[i], levels=30, cmap='hot')
            plt.colorbar(c, ax=ax)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f't = {times[i]:.3f}')
            ax.set_aspect('equal')

    # 빈 subplot 처리
    if len(plot_indices) < 6:
        axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('heat_2d.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 중심점에서의 시간 변화
    fig2, ax = plt.subplots(figsize=(10, 5))
    center_values = [h[solver.ny//2, solver.nx//2] for h in history]
    ax.plot(times, center_values, 'b-', linewidth=2)
    ax.set_xlabel('시간 t')
    ax.set_ylabel('u(0.5, 0.5, t)')
    ax.set_title('중심점에서의 온도 변화')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('heat_2d_center.png', dpi=150, bbox_inches='tight')
    plt.show()

    return solver, times, history

# solver, times, history = demo_heat_2d()
```

### 5.3 2D Crank-Nicolson (ADI 방법)

대규모 2D 문제에서는 ADI (Alternating Direction Implicit) 방법이 효율적입니다.

```python
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

class HeatEquation2D_ADI:
    """
    2D 열방정식 ADI (Alternating Direction Implicit) 방법

    각 시간 스텝을 두 반스텝으로 나눔:
    1. x-방향 암시적, y-방향 명시적
    2. y-방향 암시적, x-방향 명시적

    무조건 안정 + 2차 정확도
    """

    def __init__(self, Lx=1.0, Ly=1.0, alpha=0.01, nx=51, ny=51, T=0.5, nt=100):
        self.Lx = Lx
        self.Ly = Ly
        self.alpha = alpha
        self.nx = nx
        self.ny = ny
        self.T = T
        self.nt = nt

        # 격자 생성
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.dt = T / nt
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.rx = alpha * self.dt / (2 * self.dx**2)
        self.ry = alpha * self.dt / (2 * self.dy**2)

        print(f"2D ADI 열방정식 설정")
        print(f"  격자: {nx} x {ny}")
        print(f"  r_x = {self.rx:.4f}, r_y = {self.ry:.4f}")

        # 행렬 생성
        self._build_matrices()

    def _build_matrices(self):
        """ADI 삼중대각 행렬 생성"""
        # x-방향 (각 y에 대해)
        mx = self.nx - 2
        main_x = (1 + 2*self.rx) * np.ones(mx)
        off_x = -self.rx * np.ones(mx - 1)
        self.Ax = diags([off_x, main_x, off_x], [-1, 0, 1], format='csr')

        # y-방향 (각 x에 대해)
        my = self.ny - 2
        main_y = (1 + 2*self.ry) * np.ones(my)
        off_y = -self.ry * np.ones(my - 1)
        self.Ay = diags([off_y, main_y, off_y], [-1, 0, 1], format='csr')

    def set_initial_condition(self, func):
        """초기조건 설정"""
        self.u = func(self.X, self.Y)
        self.u0 = self.u.copy()

    def set_boundary_conditions(self, bc_value=0):
        """디리클레 경계조건"""
        self.bc_value = bc_value

    def step(self):
        """한 시간 스텝 (ADI 두 반스텝)"""
        u = self.u
        bc = self.bc_value

        # 중간 해 배열
        u_half = np.zeros_like(u)
        u_new = np.zeros_like(u)

        # 반스텝 1: x-암시적, y-명시적
        for j in range(1, self.ny - 1):
            # y-명시적 부분 (우변)
            b = u[j, 1:-1] + self.ry * (u[j+1, 1:-1] - 2*u[j, 1:-1] + u[j-1, 1:-1])
            # 경계조건
            b[0] += self.rx * bc
            b[-1] += self.rx * bc
            # x-암시적 풀이
            u_half[j, 1:-1] = spsolve(self.Ax, b)

        # 경계조건 적용
        u_half[0, :] = bc
        u_half[-1, :] = bc
        u_half[:, 0] = bc
        u_half[:, -1] = bc

        # 반스텝 2: y-암시적, x-명시적
        for i in range(1, self.nx - 1):
            # x-명시적 부분 (우변)
            b = u_half[1:-1, i] + self.rx * (u_half[1:-1, i+1] - 2*u_half[1:-1, i] + u_half[1:-1, i-1])
            # 경계조건
            b[0] += self.ry * bc
            b[-1] += self.ry * bc
            # y-암시적 풀이
            u_new[1:-1, i] = spsolve(self.Ay, b)

        # 경계조건 적용
        u_new[0, :] = bc
        u_new[-1, :] = bc
        u_new[:, 0] = bc
        u_new[:, -1] = bc

        self.u = u_new

    def solve(self, save_interval=None):
        """전체 시간 구간 풀이"""
        if save_interval is None:
            save_interval = max(1, self.nt // 50)

        history = [self.u.copy()]
        times = [0]

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), history


def compare_2d_methods():
    """2D FTCS vs ADI 비교"""
    Lx = Ly = 1.0
    alpha = 0.01
    nx = ny = 41
    T = 0.3

    # 초기조건
    def ic(X, Y):
        return np.sin(np.pi * X) * np.sin(np.pi * Y)

    # FTCS
    ftcs = HeatEquation2D_FTCS(Lx, Ly, alpha, nx, ny, T, safety=0.2)
    ftcs.set_initial_condition(ic)
    ftcs.set_boundary_conditions(0)
    times_ftcs, history_ftcs = ftcs.solve()

    # ADI
    adi = HeatEquation2D_ADI(Lx, Ly, alpha, nx, ny, T, nt=30)
    adi.set_initial_condition(ic)
    adi.set_boundary_conditions(0)
    times_adi, history_adi = adi.solve()

    # 해석해: u = sin(πx)sin(πy)exp(-2απ²t)
    u_exact = np.sin(np.pi * adi.X) * np.sin(np.pi * adi.Y) * \
              np.exp(-2 * alpha * np.pi**2 * T)

    print(f"\n비교 결과 (t = {T}):")
    print(f"  FTCS 시간 스텝: {ftcs.nt}")
    print(f"  ADI 시간 스텝: {adi.nt}")
    print(f"  FTCS 최대 오차: {np.max(np.abs(history_ftcs[-1] - u_exact)):.2e}")
    print(f"  ADI 최대 오차: {np.max(np.abs(history_adi[-1] - u_exact)):.2e}")

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    c1 = axes[0].contourf(adi.X, adi.Y, history_ftcs[-1], levels=30, cmap='hot')
    plt.colorbar(c1, ax=axes[0])
    axes[0].set_title(f'FTCS (dt={ftcs.dt:.5f})')
    axes[0].set_aspect('equal')

    c2 = axes[1].contourf(adi.X, adi.Y, history_adi[-1], levels=30, cmap='hot')
    plt.colorbar(c2, ax=axes[1])
    axes[1].set_title(f'ADI (dt={adi.dt:.4f})')
    axes[1].set_aspect('equal')

    c3 = axes[2].contourf(adi.X, adi.Y, u_exact, levels=30, cmap='hot')
    plt.colorbar(c3, ax=axes[2])
    axes[2].set_title('해석해')
    axes[2].set_aspect('equal')

    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig('heat_2d_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# compare_2d_methods()
```

---

## 6. 다양한 경계조건 처리

### 6.1 노이만 경계조건

```python
class HeatEquation1D_Neumann:
    """
    노이만 경계조건을 가진 1D 열방정식

    ∂u/∂x|_{x=0} = flux_left
    ∂u/∂x|_{x=L} = flux_right
    """

    def __init__(self, L=1.0, alpha=0.01, nx=51, T=1.0, safety=0.4):
        self.L = L
        self.alpha = alpha
        self.nx = nx
        self.T = T

        self.dx = L / (nx - 1)
        self.dt = safety * self.dx**2 / alpha
        self.nt = int(np.ceil(T / self.dt))
        self.dt = T / self.nt

        self.x = np.linspace(0, L, nx)
        self.r = alpha * self.dt / self.dx**2

        print(f"노이만 BC 열방정식: r = {self.r:.4f}")

    def set_initial_condition(self, func):
        self.u = func(self.x)

    def set_boundary_conditions(self, flux_left=0, flux_right=0):
        """노이만 경계조건 설정"""
        self.flux_left = flux_left
        self.flux_right = flux_right

    def step(self):
        """한 시간 스텝 (노이만 BC 포함)"""
        u_new = self.u.copy()

        # 내부점
        u_new[1:-1] = self.u[1:-1] + self.r * (
            self.u[2:] - 2*self.u[1:-1] + self.u[:-2]
        )

        # 왼쪽 노이만 BC: ∂u/∂x = flux_left
        # 고스트 노드 방법: u[-1] = u[1] - 2*dx*flux_left
        # u_new[0] = u[0] + r*(u[1] - 2*u[0] + u[-1])
        #          = u[0] + r*(u[1] - 2*u[0] + u[1] - 2*dx*flux_left)
        #          = u[0] + r*(2*u[1] - 2*u[0] - 2*dx*flux_left)
        u_new[0] = self.u[0] + self.r * (
            2*self.u[1] - 2*self.u[0] - 2*self.dx*self.flux_left
        )

        # 오른쪽 노이만 BC: ∂u/∂x = flux_right
        u_new[-1] = self.u[-1] + self.r * (
            2*self.u[-2] - 2*self.u[-1] + 2*self.dx*self.flux_right
        )

        self.u = u_new

    def solve(self):
        history = [self.u.copy()]
        times = [0]

        save_interval = max(1, self.nt // 100)

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), np.array(history)


def demo_neumann_bc():
    """노이만 경계조건 데모: 단열 양단"""
    solver = HeatEquation1D_Neumann(L=1.0, alpha=0.01, nx=51, T=5.0)

    # 초기조건: 왼쪽 반은 뜨겁고 오른쪽 반은 차가움
    solver.set_initial_condition(lambda x: np.where(x < 0.5, 1.0, 0.0))

    # 양단 단열 (flux = 0)
    solver.set_boundary_conditions(flux_left=0, flux_right=0)

    times, history = solver.solve()

    # 에너지 보존 확인
    energies = [np.trapz(h, solver.x) for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 온도 분포
    ax1 = axes[0]
    for i in range(0, len(times), len(times)//5):
        ax1.plot(solver.x, history[i], label=f't = {times[i]:.2f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u')
    ax1.set_title('단열 경계조건 (∂u/∂x = 0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 총 에너지
    ax2 = axes[1]
    ax2.plot(times, energies, 'b-', linewidth=2)
    ax2.axhline(y=energies[0], color='r', linestyle='--', label='초기 에너지')
    ax2.set_xlabel('시간 t')
    ax2.set_ylabel('총 에너지 (∫u dx)')
    ax2.set_title('에너지 보존 확인')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('heat_neumann.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"초기 에너지: {energies[0]:.6f}")
    print(f"최종 에너지: {energies[-1]:.6f}")
    print(f"에너지 변화: {(energies[-1] - energies[0]) / energies[0] * 100:.4f}%")

# demo_neumann_bc()
```

### 6.2 로빈 경계조건

```python
def demo_robin_bc():
    """로빈 경계조건: 대류 열전달"""
    L = 1.0
    alpha = 0.01
    nx = 51
    T = 2.0

    dx = L / (nx - 1)
    dt = 0.4 * dx**2 / alpha
    nt = int(np.ceil(T / dt))
    dt = T / nt
    r = alpha * dt / dx**2

    x = np.linspace(0, L, nx)

    # 초기조건: 균일 온도
    u = np.ones(nx)

    # 로빈 BC 파라미터
    # -k·∂u/∂x = h·(u - T_inf) at x = 0
    # 여기서 k=열전도도, h=열전달계수, T_inf=주변온도
    k = 1.0
    h = 10.0  # 열전달계수
    T_inf = 0.0  # 주변 온도
    Bi = h * dx / k  # Biot 수

    history = [u.copy()]
    times = [0]

    for n in range(nt):
        u_new = u.copy()

        # 내부점
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])

        # 왼쪽 로빈 BC: h(u - T_inf) + k·∂u/∂x = 0
        # (u[1] - u[0])/dx = (h/k)(u[0] - T_inf)
        # 고스트 노드: u[-1] = u[1] - 2*dx*(h/k)*(u[0] - T_inf)
        u_new[0] = u[0] + r * (2*u[1] - 2*u[0] - 2*dx*(h/k)*(u[0] - T_inf))

        # 오른쪽: 디리클레 (고정 온도)
        u_new[-1] = 1.0

        u = u_new

        if (n + 1) % (nt // 50) == 0:
            history.append(u.copy())
            times.append((n + 1) * dt)

    # 시각화
    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(0, len(times), len(times)//5):
        ax.plot(x, history[i], label=f't = {times[i]:.2f}')

    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title(f'로빈 경계조건 (대류 열전달)\nBiot 수 = {Bi:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('heat_robin.png', dpi=150, bbox_inches='tight')
    plt.show()

# demo_robin_bc()
```

---

## 7. 요약

### 스킴 비교표

| 스킴 | 정확도 | 안정성 | 계산 비용 | 특징 |
|------|--------|--------|-----------|------|
| FTCS | O(Δt, Δx²) | 조건부 (r≤0.5) | 낮음 | 단순, 명시적 |
| BTCS | O(Δt, Δx²) | 무조건 | 중간 | 암시적, 행렬 풀이 |
| Crank-Nicolson | O(Δt², Δx²) | 무조건 | 중간 | 2차 정확도 |
| ADI (2D) | O(Δt², Δx²) | 무조건 | 중간 | 효율적인 2D |

### CFL 조건

```
1D FTCS: r = α·Δt/Δx² ≤ 0.5
2D FTCS: r_x + r_y ≤ 0.5
```

### 다음 단계

1. **10장**: 파동방정식 - 쌍곡선형 PDE
2. **11장**: 라플라스/포아송 - 타원형 PDE
3. **12장**: 이류방정식 - 1차 쌍곡선형 PDE

---

## 연습문제

### 연습 1: FTCS 안정성 실험
r = 0.3, 0.5, 0.6에서 FTCS를 실행하고 안정/불안정 동작을 확인하시오.

### 연습 2: 수렴 차수 확인
Crank-Nicolson의 시간 정확도가 2차임을 수치적으로 확인하시오.

### 연습 3: 비균질 경계조건
u(0,t) = 0, u(L,t) = 100인 경우의 정상상태 해를 구하시오.

### 연습 4: 2D 열방정식
가우시안 초기조건이 아닌 사각형 열점 초기조건으로 2D 열방정식을 풀어보시오.

---

## 참고 자료

1. **교재**: LeVeque, "Finite Difference Methods for Ordinary and Partial Differential Equations"
2. **Python**: scipy.sparse, numpy
3. **시각화**: matplotlib.animation
