# 10. 파동방정식 (Wave Equation)

## 학습 목표
- 1D/2D 파동방정식의 물리적 의미 이해
- CTCS (Central Time Central Space) 방법 구현
- 다양한 경계조건 (고정, 자유, 흡수) 처리
- 파동 전파 애니메이션 시각화

---

## 1. 파동방정식 이론

### 1.1 물리적 배경

파동방정식은 파동 현상을 기술하는 쌍곡선형 PDE입니다.

```
1D 파동방정식:
∂²u/∂t² = c² · ∂²u/∂x²

여기서:
- u(x,t): 변위 (displacement)
- c: 파동 전파 속도
- x: 공간 좌표
- t: 시간
```

### 1.2 응용 분야

| 분야 | 물리량 u | 전파속도 c |
|------|----------|-----------|
| 현의 진동 | 변위 | √(T/ρ) (T: 장력, ρ: 선밀도) |
| 음파 | 압력 | √(γP/ρ) (공기 중 ~340 m/s) |
| 전자기파 | 전기장/자기장 | 광속 (~3×10⁸ m/s) |
| 지진파 | 지반 변위 | 수 km/s |

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# 물질별 파동 속도 (예시)
wave_speeds = {
    '기타 현 (E)': 329.0,   # Hz를 m/s로 환산
    '공기 (음파, 20°C)': 343.0,
    '물 (음파)': 1481.0,
    '강철 (종파)': 5960.0,
    '빛 (진공)': 299792458.0,
}

for material, c in wave_speeds.items():
    print(f"{material}: c = {c:.0f} m/s")
```

### 1.3 달랑베르 해 (D'Alembert's Solution)

무한 영역에서의 해석해:

```
u(x,t) = f(x - ct) + g(x + ct)

- f(x - ct): 오른쪽으로 이동하는 파
- g(x + ct): 왼쪽으로 이동하는 파
```

```python
import numpy as np
import matplotlib.pyplot as plt

def dalembert_demo():
    """달랑베르 해 시각화"""
    c = 1.0  # 파동 속도
    x = np.linspace(-5, 5, 500)

    # 초기 조건: 가우시안 펄스
    def f(x):
        return np.exp(-x**2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    times = [0, 0.5, 1.0, 1.5, 2.0, 2.5]

    for idx, t in enumerate(times):
        ax = axes[idx // 3, idx % 3]

        # 오른쪽 이동 파
        u_right = 0.5 * f(x - c*t)
        # 왼쪽 이동 파
        u_left = 0.5 * f(x + c*t)
        # 총 해
        u_total = u_right + u_left

        ax.plot(x, u_right, 'b--', alpha=0.5, label='오른쪽 이동')
        ax.plot(x, u_left, 'r--', alpha=0.5, label='왼쪽 이동')
        ax.plot(x, u_total, 'k-', linewidth=2, label='총합')

        ax.set_xlim(-5, 5)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title(f't = {t:.1f}')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()

    plt.suptitle("달랑베르 해: u = f(x-ct) + f(x+ct)", fontsize=14)
    plt.tight_layout()
    plt.savefig('wave_dalembert.png', dpi=150, bbox_inches='tight')
    plt.show()

# dalembert_demo()
```

---

## 2. CTCS 방법 (Central Time Central Space)

### 2.1 이산화

CTCS는 시간과 공간 모두에 중심차분을 사용합니다.

```
시간: 중심차분
∂²u/∂t² ≈ (u_i^{n+1} - 2u_i^n + u_i^{n-1}) / Δt²

공간: 중심차분
∂²u/∂x² ≈ (u_{i+1}^n - 2u_i^n + u_{i-1}^n) / Δx²

결합 (정리하면):
u_i^{n+1} = 2u_i^n - u_i^{n-1} + C² · (u_{i+1}^n - 2u_i^n + u_{i-1}^n)

여기서 C = c·Δt/Δx (Courant 수)
```

### 2.2 CTCS 스텐실

```
시간 n+1:         [i]
                   ↑
시간 n:    [i-1]--[i]--[i+1]
                   ↓
시간 n-1:         [i]
```

### 2.3 안정성 조건

```
CFL 조건: C = c·Δt/Δx ≤ 1

물리적 의미:
- 수치적 정보 전파 속도(Δx/Δt) ≥ 물리적 전파 속도(c)
- C = 1일 때 가장 정확함 (수치 분산 없음)
```

### 2.4 CTCS 구현

```python
import numpy as np
import matplotlib.pyplot as plt

class WaveEquation1D:
    """
    1D 파동방정식 CTCS 방법

    ∂²u/∂t² = c² · ∂²u/∂x²
    """

    def __init__(self, L=1.0, c=1.0, nx=101, T=2.0, courant=0.9):
        """
        Parameters:
        -----------
        L : float - 영역 길이
        c : float - 파동 전파 속도
        nx : int - 공간 격자점 수
        T : float - 최종 시간
        courant : float - Courant 수 (≤ 1)
        """
        self.L = L
        self.c = c
        self.nx = nx
        self.T = T

        # 격자 생성
        self.dx = L / (nx - 1)
        self.x = np.linspace(0, L, nx)

        # CFL 조건에 따른 시간 간격
        self.dt = courant * self.dx / c
        self.nt = int(np.ceil(T / self.dt))
        self.dt = T / self.nt

        self.C = c * self.dt / self.dx  # Courant 수
        self.C2 = self.C ** 2

        print(f"1D 파동방정식 CTCS 설정")
        print(f"  L = {L}, c = {c}")
        print(f"  nx = {nx}, dx = {self.dx:.4f}")
        print(f"  dt = {self.dt:.6f}, nt = {self.nt}")
        print(f"  Courant 수 C = {self.C:.4f}")
        print(f"  안정성: {'OK' if self.C <= 1 else 'WARNING!'}")

    def set_initial_conditions(self, u0_func, v0_func=None):
        """
        초기조건 설정

        Parameters:
        -----------
        u0_func : callable - u(x, 0) = u0_func(x)
        v0_func : callable - ∂u/∂t(x, 0) = v0_func(x)
        """
        self.u = u0_func(self.x)
        self.u0 = self.u.copy()

        if v0_func is None:
            v0_func = lambda x: np.zeros_like(x)

        # 첫 번째 시간 스텝 (초기 속도 사용)
        # u^1 ≈ u^0 + dt·v^0 + (dt²/2)·c²·∂²u^0/∂x²
        self.u_prev = self.u.copy()

        # 공간 2차 미분
        d2u = np.zeros_like(self.u)
        d2u[1:-1] = (self.u[2:] - 2*self.u[1:-1] + self.u[:-2]) / self.dx**2

        self.u = self.u_prev + self.dt * v0_func(self.x) + \
                 0.5 * self.dt**2 * self.c**2 * d2u

    def set_boundary_conditions(self, bc_type='fixed', left_value=0, right_value=0):
        """
        경계조건 설정

        Parameters:
        -----------
        bc_type : str - 'fixed' (고정), 'free' (자유), 'absorbing' (흡수)
        """
        self.bc_type = bc_type
        self.bc_left = left_value
        self.bc_right = right_value

    def apply_bc(self, u, u_prev=None):
        """경계조건 적용"""
        if self.bc_type == 'fixed':
            # 디리클레: u = 상수
            u[0] = self.bc_left
            u[-1] = self.bc_right

        elif self.bc_type == 'free':
            # 노이만: ∂u/∂x = 0
            u[0] = u[1]
            u[-1] = u[-2]

        elif self.bc_type == 'absorbing':
            # 흡수 경계 (1차 Sommerfeld)
            # ∂u/∂t ± c·∂u/∂x = 0
            if u_prev is not None:
                # 왼쪽: ∂u/∂t - c·∂u/∂x = 0 (오른쪽으로 나가는 파)
                u[0] = u_prev[0] + self.C * (u[1] - u_prev[1])
                # 오른쪽: ∂u/∂t + c·∂u/∂x = 0 (왼쪽으로 나가는 파)
                u[-1] = u_prev[-1] - self.C * (u[-1] - u_prev[-2])

        return u

    def step(self):
        """한 시간 스텝 진행 (CTCS)"""
        u_new = np.zeros_like(self.u)

        # 내부점 업데이트
        u_new[1:-1] = (2*self.u[1:-1] - self.u_prev[1:-1] +
                       self.C2 * (self.u[2:] - 2*self.u[1:-1] + self.u[:-2]))

        # 경계조건 적용
        u_new = self.apply_bc(u_new, self.u_prev)

        # 업데이트
        self.u_prev = self.u.copy()
        self.u = u_new

    def solve(self, save_interval=None):
        """전체 시간 구간 풀이"""
        if save_interval is None:
            save_interval = max(1, self.nt // 200)

        history = [self.u0.copy()]
        times = [0]

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), np.array(history)


def demo_wave_1d():
    """1D 파동방정식 데모"""
    L = 1.0
    c = 1.0

    # 세 가지 경계조건 비교
    bc_types = ['fixed', 'free', 'absorbing']
    results = {}

    for bc in bc_types:
        solver = WaveEquation1D(L=L, c=c, nx=201, T=3.0, courant=0.9)

        # 초기조건: 가우시안 펄스
        def u0(x):
            x0 = 0.3
            sigma = 0.05
            return np.exp(-(x - x0)**2 / (2 * sigma**2))

        solver.set_initial_conditions(u0)
        solver.set_boundary_conditions(bc_type=bc)

        times, history = solver.solve(save_interval=10)
        results[bc] = (solver, times, history)

        print(f"\n{bc} 경계조건 완료")

    # 시각화
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))

    for row, bc in enumerate(bc_types):
        solver, times, history = results[bc]

        # 여러 시간 스냅샷
        time_indices = [0, len(times)//4, len(times)//2, len(times)-1]

        for col, ti in enumerate(time_indices):
            ax = axes[row, col]
            ax.plot(solver.x, history[ti], 'b-', linewidth=1.5)
            ax.set_xlim(0, L)
            ax.set_ylim(-1.5, 1.5)
            ax.set_xlabel('x')
            ax.set_ylabel('u')
            ax.set_title(f't = {times[ti]:.2f}')
            ax.grid(True, alpha=0.3)

            if col == 0:
                ax.set_ylabel(f'{bc}\nu')

    plt.suptitle('1D 파동방정식: 경계조건 비교', fontsize=14)
    plt.tight_layout()
    plt.savefig('wave_1d_bc_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results

# results = demo_wave_1d()
```

---

## 3. 경계조건 상세

### 3.1 고정 경계 (Fixed/Dirichlet)

```
u(0, t) = 0, u(L, t) = 0

물리적 의미: 현의 양 끝이 고정됨
결과: 파동이 경계에서 반사되며 위상이 반전됨
```

### 3.2 자유 경계 (Free/Neumann)

```
∂u/∂x(0, t) = 0, ∂u/∂x(L, t) = 0

물리적 의미: 경계에서 힘이 없음 (자유롭게 움직임)
결과: 파동이 경계에서 반사되며 위상이 유지됨
```

### 3.3 흡수 경계 (Absorbing/Sommerfeld)

```
∂u/∂t + c·∂u/∂x = 0 (오른쪽 경계)
∂u/∂t - c·∂u/∂x = 0 (왼쪽 경계)

물리적 의미: 파동이 경계를 통해 빠져나감 (무한 영역 근사)
```

```python
def boundary_condition_comparison():
    """경계조건 효과 비교"""
    L = 1.0
    c = 1.0
    nx = 201
    T = 4.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    bc_names = {'fixed': '고정 경계 (반전 반사)',
                'free': '자유 경계 (동상 반사)',
                'absorbing': '흡수 경계 (반사 없음)'}

    for idx, bc_type in enumerate(['fixed', 'free', 'absorbing']):
        solver = WaveEquation1D(L=L, c=c, nx=nx, T=T, courant=0.95)

        # 초기조건: 중앙으로 이동하는 가우시안 펄스
        def u0(x):
            x0 = 0.2
            sigma = 0.05
            return np.exp(-(x - x0)**2 / (2 * sigma**2))

        solver.set_initial_conditions(u0)
        solver.set_boundary_conditions(bc_type=bc_type)

        times, history = solver.solve(save_interval=5)

        # 시공간 등고선도
        ax = axes[idx]
        X, T_grid = np.meshgrid(solver.x, times)
        c_plot = ax.contourf(X, T_grid, history, levels=30, cmap='RdBu_r')
        plt.colorbar(c_plot, ax=ax)

        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_title(bc_names[bc_type])

    plt.tight_layout()
    plt.savefig('wave_bc_spacetime.png', dpi=150, bbox_inches='tight')
    plt.show()

# boundary_condition_comparison()
```

---

## 4. 정상파와 고유모드

### 4.1 정상파 해석해

고정 경계조건에서의 고유모드:

```
u_n(x, t) = sin(nπx/L) · cos(nπct/L)

고유진동수: f_n = nc/(2L)
```

```python
def standing_waves():
    """정상파 고유모드"""
    L = 1.0
    c = 1.0
    x = np.linspace(0, L, 200)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for n in range(1, 7):
        ax = axes[(n-1)//3, (n-1)%3]

        # 여러 시간에서의 정상파
        for phase in np.linspace(0, np.pi, 5):
            u = np.sin(n * np.pi * x / L) * np.cos(phase)
            alpha = 0.2 + 0.8 * (1 - abs(np.cos(phase)))
            ax.plot(x, u, 'b-', alpha=alpha)

        # 포락선
        ax.plot(x, np.sin(n * np.pi * x / L), 'r--', linewidth=2, label='포락선')
        ax.plot(x, -np.sin(n * np.pi * x / L), 'r--', linewidth=2)

        f_n = n * c / (2 * L)
        ax.set_title(f'모드 {n}: f = {f_n:.2f} Hz')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_xlim(0, L)
        ax.set_ylim(-1.3, 1.3)
        ax.grid(True, alpha=0.3)

        # 마디점 표시
        nodes = np.linspace(0, L, n+1)
        for node in nodes:
            ax.axvline(x=node, color='green', linestyle=':', alpha=0.5)

    plt.suptitle('정상파 고유모드 (고정-고정 경계)', fontsize=14)
    plt.tight_layout()
    plt.savefig('wave_standing_modes.png', dpi=150, bbox_inches='tight')
    plt.show()

# standing_waves()
```

### 4.2 수치 해와 해석해 비교

```python
def compare_with_exact():
    """수치해와 해석해 비교"""
    L = 1.0
    c = 1.0

    # 해석해: u(x,t) = sin(πx)·cos(πct)
    def exact_solution(x, t):
        return np.sin(np.pi * x / L) * np.cos(np.pi * c * t / L)

    # Courant 수에 따른 비교
    courant_values = [0.5, 0.8, 1.0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    T = 2.0

    for idx, C in enumerate(courant_values):
        solver = WaveEquation1D(L=L, c=c, nx=51, T=T, courant=C)

        # 초기조건: 첫 번째 고유모드
        solver.set_initial_conditions(
            u0_func=lambda x: np.sin(np.pi * x / L),
            v0_func=lambda x: np.zeros_like(x)
        )
        solver.set_boundary_conditions(bc_type='fixed')

        times, history = solver.solve()

        # 최종 시간에서 비교
        u_exact = exact_solution(solver.x, T)
        u_numerical = history[-1]

        ax = axes[idx]
        ax.plot(solver.x, u_exact, 'b-', label='해석해', linewidth=2)
        ax.plot(solver.x, u_numerical, 'ro', label='수치해', markersize=4)

        error = np.max(np.abs(u_numerical - u_exact))
        ax.set_title(f'C = {C}\n최대 오차: {error:.2e}')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f't = {T}에서 해석해 비교', fontsize=14)
    plt.tight_layout()
    plt.savefig('wave_exact_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# compare_with_exact()
```

---

## 5. 2D 파동방정식

### 5.1 2D 파동방정식

```
∂²u/∂t² = c² · (∂²u/∂x² + ∂²u/∂y²) = c² · ∇²u
```

### 5.2 CTCS 2D 구현

```python
import numpy as np
import matplotlib.pyplot as plt

class WaveEquation2D:
    """
    2D 파동방정식 CTCS 방법

    ∂²u/∂t² = c² · (∂²u/∂x² + ∂²u/∂y²)
    """

    def __init__(self, Lx=1.0, Ly=1.0, c=1.0, nx=101, ny=101, T=2.0, courant=0.5):
        """
        Parameters:
        -----------
        Lx, Ly : float - 영역 크기
        c : float - 파동 속도
        nx, ny : int - 격자점 수
        T : float - 최종 시간
        courant : float - Courant 수 (2D에서 C ≤ 1/√2)
        """
        self.Lx = Lx
        self.Ly = Ly
        self.c = c
        self.nx = nx
        self.ny = ny
        self.T = T

        # 격자 생성
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # 2D CFL: C² ≤ 1/(1/dx² + 1/dy²) · dt²
        # 간소화: C_x² + C_y² ≤ 1, 등간격이면 C ≤ 1/√2
        dt_max = courant / (c * np.sqrt(1/self.dx**2 + 1/self.dy**2))
        self.dt = dt_max
        self.nt = int(np.ceil(T / self.dt))
        self.dt = T / self.nt

        self.Cx = c * self.dt / self.dx
        self.Cy = c * self.dt / self.dy
        self.Cx2 = self.Cx ** 2
        self.Cy2 = self.Cy ** 2

        print(f"2D 파동방정식 CTCS 설정")
        print(f"  격자: {nx} x {ny}")
        print(f"  Cx = {self.Cx:.4f}, Cy = {self.Cy:.4f}")
        print(f"  Cx² + Cy² = {self.Cx2 + self.Cy2:.4f} (≤ 1이어야 안정)")

    def set_initial_conditions(self, u0_func, v0_func=None):
        """초기조건 설정"""
        self.u = u0_func(self.X, self.Y)
        self.u0 = self.u.copy()

        if v0_func is None:
            v0_func = lambda X, Y: np.zeros_like(X)

        # 첫 번째 시간 스텝
        self.u_prev = self.u.copy()

        d2u_dx2 = np.zeros_like(self.u)
        d2u_dy2 = np.zeros_like(self.u)
        d2u_dx2[:, 1:-1] = (self.u[:, 2:] - 2*self.u[:, 1:-1] + self.u[:, :-2]) / self.dx**2
        d2u_dy2[1:-1, :] = (self.u[2:, :] - 2*self.u[1:-1, :] + self.u[:-2, :]) / self.dy**2

        self.u = (self.u_prev + self.dt * v0_func(self.X, self.Y) +
                  0.5 * self.dt**2 * self.c**2 * (d2u_dx2 + d2u_dy2))

    def set_boundary_conditions(self, bc_type='fixed'):
        """경계조건 설정"""
        self.bc_type = bc_type

    def apply_bc(self, u):
        """경계조건 적용"""
        if self.bc_type == 'fixed':
            u[0, :] = 0
            u[-1, :] = 0
            u[:, 0] = 0
            u[:, -1] = 0
        elif self.bc_type == 'free':
            u[0, :] = u[1, :]
            u[-1, :] = u[-2, :]
            u[:, 0] = u[:, 1]
            u[:, -1] = u[:, -2]
        return u

    def step(self):
        """한 시간 스텝 진행"""
        u_new = np.zeros_like(self.u)

        # 내부점 업데이트
        u_new[1:-1, 1:-1] = (
            2*self.u[1:-1, 1:-1] - self.u_prev[1:-1, 1:-1] +
            self.Cx2 * (self.u[1:-1, 2:] - 2*self.u[1:-1, 1:-1] + self.u[1:-1, :-2]) +
            self.Cy2 * (self.u[2:, 1:-1] - 2*self.u[1:-1, 1:-1] + self.u[:-2, 1:-1])
        )

        u_new = self.apply_bc(u_new)

        self.u_prev = self.u.copy()
        self.u = u_new

    def solve(self, save_interval=None):
        """전체 시간 구간 풀이"""
        if save_interval is None:
            save_interval = max(1, self.nt // 100)

        history = [self.u0.copy()]
        times = [0]

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), history


def demo_wave_2d():
    """2D 파동방정식 데모"""
    solver = WaveEquation2D(Lx=1.0, Ly=1.0, c=1.0, nx=101, ny=101, T=2.0, courant=0.4)

    # 초기조건: 중앙의 가우시안 펄스
    def u0(X, Y):
        x0, y0 = 0.5, 0.5
        sigma = 0.1
        return np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

    solver.set_initial_conditions(u0)
    solver.set_boundary_conditions('fixed')

    times, history = solver.solve(save_interval=10)

    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    plot_times = [0, len(times)//5, 2*len(times)//5,
                  3*len(times)//5, 4*len(times)//5, len(times)-1]

    vmax = np.max(np.abs(history[0]))

    for idx, ti in enumerate(plot_times):
        ax = axes[idx // 3, idx % 3]
        c_plot = ax.contourf(solver.X, solver.Y, history[ti],
                            levels=30, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(c_plot, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f't = {times[ti]:.3f}')
        ax.set_aspect('equal')

    plt.suptitle('2D 파동방정식 (고정 경계)', fontsize=14)
    plt.tight_layout()
    plt.savefig('wave_2d.png', dpi=150, bbox_inches='tight')
    plt.show()

    return solver, times, history

# solver, times, history = demo_wave_2d()
```

---

## 6. 애니메이션 시각화

### 6.1 1D 파동 애니메이션

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_wave_animation_1d():
    """1D 파동 애니메이션 생성"""
    # 시뮬레이션
    solver = WaveEquation1D(L=1.0, c=1.0, nx=201, T=4.0, courant=0.95)

    def u0(x):
        # 두 개의 가우시안 펄스
        return (np.exp(-(x - 0.3)**2 / 0.01) +
                0.5 * np.exp(-(x - 0.7)**2 / 0.01))

    solver.set_initial_conditions(u0)
    solver.set_boundary_conditions('fixed')

    times, history = solver.solve(save_interval=2)

    # 애니메이션 생성
    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot([], [], 'b-', linewidth=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    ax.set_xlim(0, solver.L)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title('1D 파동방정식 (고정 경계)')
    ax.grid(True, alpha=0.3)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(frame):
        line.set_data(solver.x, history[frame])
        time_text.set_text(f't = {times[frame]:.3f}')
        return line, time_text

    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(times), interval=30, blit=True)

    # GIF로 저장
    # anim.save('wave_1d_animation.gif', writer='pillow', fps=30)
    plt.show()

    return anim

# anim = create_wave_animation_1d()
```

### 6.2 2D 파동 애니메이션

```python
def create_wave_animation_2d():
    """2D 파동 애니메이션 생성"""
    solver = WaveEquation2D(Lx=1.0, Ly=1.0, c=1.0, nx=81, ny=81, T=2.0, courant=0.4)

    def u0(X, Y):
        # 중앙에서 벗어난 초기 펄스
        x0, y0 = 0.3, 0.3
        sigma = 0.08
        return np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

    solver.set_initial_conditions(u0)
    solver.set_boundary_conditions('fixed')

    times, history = solver.solve(save_interval=5)

    # 애니메이션
    fig, ax = plt.subplots(figsize=(8, 8))
    vmax = np.max(np.abs(history[0]))

    c_plot = ax.contourf(solver.X, solver.Y, history[0],
                        levels=30, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(c_plot, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    title = ax.set_title(f't = 0.000')

    def animate(frame):
        ax.clear()
        c_plot = ax.contourf(solver.X, solver.Y, history[frame],
                            levels=30, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f't = {times[frame]:.3f}')
        ax.set_aspect('equal')
        return c_plot.collections

    anim = FuncAnimation(fig, animate, frames=len(times), interval=50)

    # GIF로 저장
    # anim.save('wave_2d_animation.gif', writer='pillow', fps=20)
    plt.show()

    return anim

# anim = create_wave_animation_2d()
```

---

## 7. 수치 분산 분석

### 7.1 분산 관계

```python
def dispersion_analysis():
    """수치 분산 분석"""
    # 연속 분산 관계: ω = c·k
    # CTCS 분산 관계: sin(ω·dt/2) = C·sin(k·dx/2)

    k_dx = np.linspace(0.01, np.pi, 100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 여러 Courant 수에 대한 분산
    ax1 = axes[0]
    courant_values = [0.5, 0.8, 0.95, 1.0]

    for C in courant_values:
        # 수치 분산 관계로부터 ω 계산
        # ω_num · dt/2 = arcsin(C · sin(k·dx/2))
        arg = C * np.sin(k_dx / 2)
        arg = np.clip(arg, -1, 1)  # arcsin 범위 내로
        omega_num = 2 * np.arcsin(arg)

        # 정규화된 위상속도
        c_phase = omega_num / k_dx

        ax1.plot(k_dx, c_phase / C, label=f'C = {C}', linewidth=2)

    ax1.axhline(y=1, color='r', linestyle='--', label='정확 (c_num/c = 1)')
    ax1.set_xlabel('kΔx')
    ax1.set_ylabel('c_numerical / c_exact')
    ax1.set_title('수치 위상속도 (분산)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, np.pi)
    ax1.set_ylim(0.5, 1.1)

    # 군속도
    ax2 = axes[1]
    for C in courant_values:
        # 군속도 = dω/dk
        arg = C * np.sin(k_dx / 2)
        arg = np.clip(arg, -1, 1)

        # 수치적 미분
        omega_num = 2 * np.arcsin(arg)
        d_omega = np.gradient(omega_num, k_dx[1] - k_dx[0])

        ax2.plot(k_dx, d_omega / C, label=f'C = {C}', linewidth=2)

    ax2.axhline(y=1, color='r', linestyle='--', label='정확')
    ax2.set_xlabel('kΔx')
    ax2.set_ylabel('c_group / c_exact')
    ax2.set_title('수치 군속도')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, np.pi)
    ax2.set_ylim(0, 1.5)

    plt.tight_layout()
    plt.savefig('wave_dispersion.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("관찰:")
    print("1. C = 1.0일 때 수치 분산이 가장 적음")
    print("2. 작은 k (긴 파장)에서는 모든 C에서 정확")
    print("3. 큰 k (짧은 파장)에서 분산 오차 증가")

# dispersion_analysis()
```

---

## 8. 응용 예제

### 8.1 기타 현의 진동

```python
def guitar_string_simulation():
    """기타 현 진동 시뮬레이션"""
    L = 0.65  # 기타 현 길이 (m)
    T = 73.0  # 장력 (N)
    mu = 3.75e-4  # 선밀도 (kg/m)

    c = np.sqrt(T / mu)  # 파동 속도
    f1 = c / (2 * L)  # 기본 진동수

    print(f"기타 현 파라미터:")
    print(f"  길이: {L} m")
    print(f"  장력: {T} N")
    print(f"  선밀도: {mu} kg/m")
    print(f"  파동 속도: {c:.1f} m/s")
    print(f"  기본 진동수: {f1:.1f} Hz (음: {freq_to_note(f1)})")

    # 시뮬레이션
    solver = WaveEquation1D(L=L, c=c, nx=201, T=0.01, courant=0.9)

    # 초기조건: 뜯은 위치 (삼각형)
    pluck_position = 0.2  # L에서의 상대 위치

    def u0(x):
        peak = L * pluck_position
        amplitude = 0.005  # 5mm
        return np.where(x < peak,
                       amplitude * x / peak,
                       amplitude * (L - x) / (L - peak))

    solver.set_initial_conditions(u0)
    solver.set_boundary_conditions('fixed')

    times, history = solver.solve(save_interval=2)

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 여러 시간에서의 현 모양
    ax1 = axes[0, 0]
    for i in range(0, len(times), len(times)//6):
        ax1.plot(solver.x * 1000, history[i] * 1000, label=f't = {times[i]*1000:.2f} ms')
    ax1.set_xlabel('위치 (mm)')
    ax1.set_ylabel('변위 (mm)')
    ax1.set_title('현의 변위')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 특정 위치에서의 시간 변화
    ax2 = axes[0, 1]
    monitor_position = int(0.25 * solver.nx)
    displacements = [h[monitor_position] for h in history]
    ax2.plot(np.array(times) * 1000, np.array(displacements) * 1000, 'b-')
    ax2.set_xlabel('시간 (ms)')
    ax2.set_ylabel('변위 (mm)')
    ax2.set_title(f'위치 x = {solver.x[monitor_position]*1000:.1f} mm에서의 진동')
    ax2.grid(True, alpha=0.3)

    # 주파수 스펙트럼
    ax3 = axes[1, 0]
    from scipy.fft import fft, fftfreq

    signal = np.array(displacements)
    n = len(signal)
    dt = times[1] - times[0]

    freqs = fftfreq(n, dt)
    spectrum = np.abs(fft(signal))

    positive_mask = freqs > 0
    ax3.plot(freqs[positive_mask], spectrum[positive_mask])
    ax3.set_xlabel('주파수 (Hz)')
    ax3.set_ylabel('진폭')
    ax3.set_title('주파수 스펙트럼')
    ax3.set_xlim(0, 5000)
    ax3.grid(True, alpha=0.3)

    # 기대 고조파 표시
    for n in range(1, 6):
        f_n = n * f1
        if f_n < 5000:
            ax3.axvline(x=f_n, color='r', linestyle='--', alpha=0.5)
            ax3.text(f_n, ax3.get_ylim()[1]*0.9, f'{n}f₁', rotation=90)

    # 시공간도
    ax4 = axes[1, 1]
    X, T_grid = np.meshgrid(solver.x * 1000, np.array(times) * 1000)
    c_plot = ax4.contourf(X, T_grid, np.array(history) * 1000,
                         levels=30, cmap='RdBu_r')
    plt.colorbar(c_plot, ax=ax4, label='변위 (mm)')
    ax4.set_xlabel('위치 (mm)')
    ax4.set_ylabel('시간 (ms)')
    ax4.set_title('시공간 변위')

    plt.tight_layout()
    plt.savefig('guitar_string.png', dpi=150, bbox_inches='tight')
    plt.show()


def freq_to_note(freq):
    """주파수를 음이름으로 변환"""
    A4 = 440.0
    notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

    if freq <= 0:
        return "N/A"

    semitones = 12 * np.log2(freq / A4)
    note_idx = int(round(semitones)) % 12
    octave = 4 + int(round(semitones + 9) // 12)

    return f"{notes[note_idx]}{octave}"

# guitar_string_simulation()
```

### 8.2 드럼 헤드 진동 (2D)

```python
def drum_head_simulation():
    """드럼 헤드 진동 시뮬레이션 (원형 막)"""
    # 사각형 영역에서 원형 마스크 사용
    R = 0.15  # 반지름 (m)
    c = 100.0  # 파동 속도 (m/s)

    solver = WaveEquation2D(Lx=2*R, Ly=2*R, c=c, nx=101, ny=101, T=0.02, courant=0.4)

    # 원형 마스크 생성
    center_x, center_y = R, R
    distance = np.sqrt((solver.X - center_x)**2 + (solver.Y - center_y)**2)
    mask = distance <= R

    # 초기조건: 중앙을 누름
    def u0(X, Y):
        r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        u = np.exp(-(r**2) / (0.03**2)) * 0.005  # 5mm 진폭
        u[~mask] = 0
        return u

    solver.set_initial_conditions(u0)

    # 커스텀 경계조건: 원형 경계에서 고정
    def circular_bc(u):
        u[~mask] = 0
        # 원형 경계 근처도 0으로
        boundary = (distance > R * 0.95) & (distance <= R)
        u[boundary] = 0
        return u

    # 풀이 (경계조건 수정)
    original_apply_bc = solver.apply_bc
    solver.apply_bc = circular_bc

    times, history = solver.solve(save_interval=5)

    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    plot_times = [0, len(times)//5, 2*len(times)//5,
                  3*len(times)//5, 4*len(times)//5, len(times)-1]

    vmax = np.max(np.abs(history[0]))

    for idx, ti in enumerate(plot_times):
        ax = axes[idx // 3, idx % 3]

        # 원형 마스크 적용된 데이터
        data = history[ti].copy()
        data[~mask] = np.nan

        c_plot = ax.contourf(solver.X * 1000, solver.Y * 1000, data * 1000,
                            levels=30, cmap='RdBu_r', vmin=-vmax*1000, vmax=vmax*1000)

        # 원 테두리
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(center_x*1000 + R*1000*np.cos(theta),
               center_y*1000 + R*1000*np.sin(theta), 'k-', linewidth=2)

        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(f't = {times[ti]*1000:.2f} ms')
        ax.set_aspect('equal')

    plt.suptitle('드럼 헤드 진동 (원형 막)', fontsize=14)
    plt.tight_layout()
    plt.savefig('drum_head.png', dpi=150, bbox_inches='tight')
    plt.show()

# drum_head_simulation()
```

---

## 9. 요약

### 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| 파동방정식 | ∂²u/∂t² = c²·∇²u, 쌍곡선형 PDE |
| CTCS | 시간/공간 모두 중심차분, 2차 정확도 |
| Courant 수 | C = c·Δt/Δx, 안정성: C ≤ 1 |
| 고정 경계 | 파동 반전 반사 |
| 자유 경계 | 파동 동상 반사 |
| 흡수 경계 | 파동 투과 (반사 없음) |

### CFL 조건

```
1D: C = c·Δt/Δx ≤ 1
2D: Cx² + Cy² ≤ 1 (등간격: C ≤ 1/√2)
```

### 다음 단계

1. **11장**: 라플라스/포아송 방정식 (타원형)
2. **12장**: 이류방정식 (1차 쌍곡선형)

---

## 연습문제

### 연습 1: Courant 수 실험
C = 0.5, 0.8, 1.0, 1.1에서 수치해의 안정성을 확인하시오.

### 연습 2: 정상파 모드
초기조건 u(x,0) = sin(2πx/L)에서 시작하여 두 번째 정상파 모드를 확인하시오.

### 연습 3: 흡수 경계 개선
2차 Sommerfeld 흡수 경계조건을 구현하고 1차와 비교하시오.

### 연습 4: 원형 막 고유모드
원형 막의 첫 번째 고유모드(베셀 함수)를 수치해와 비교하시오.

---

## 참고 자료

1. **교재**: LeVeque, "Finite Difference Methods"
2. **물리**: Morse & Ingard, "Theoretical Acoustics"
3. **수치 분산**: Trefethen, "Finite Difference and Spectral Methods"
