# 12. 이류방정식 (Advection Equation)

## 학습 목표
- 1차 쌍곡선형 PDE인 이류방정식 이해
- 풍상법 (Upwind Scheme) 구현
- FTCS의 불안정성 분석
- Lax-Friedrichs, Lax-Wendroff 방법 학습
- 수치 분산과 수치 확산 이해
- Courant 수의 중요성 파악

---

## 1. 이류방정식 이론

### 1.1 정의와 물리적 의미

```
1D 선형 이류방정식:
∂u/∂t + c · ∂u/∂x = 0

여기서:
- u(x,t): 이류되는 양 (농도, 온도 등)
- c: 이류 속도 (양수면 오른쪽으로 이동)
```

### 1.2 해석해

이류방정식의 해는 초기 프로파일이 속도 c로 이동하는 것입니다:

```
u(x, t) = u₀(x - ct)

여기서 u₀(x)는 초기조건
```

```python
import numpy as np
import matplotlib.pyplot as plt

def exact_advection():
    """이류방정식 해석해 시각화"""
    c = 1.0  # 이류 속도
    L = 4.0
    x = np.linspace(0, L, 500)

    # 초기조건: 가우시안 펄스
    def u0(x):
        x0 = 1.0
        sigma = 0.2
        return np.exp(-(x - x0)**2 / (2 * sigma**2))

    fig, ax = plt.subplots(figsize=(12, 5))

    times = [0, 0.5, 1.0, 1.5, 2.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))

    for t, color in zip(times, colors):
        u = u0(x - c * t)  # 해석해
        ax.plot(x, u, color=color, linewidth=2, label=f't = {t}')

    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title(f'이류방정식 해석해: u(x,t) = u₀(x - ct), c = {c}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, L)

    # 이동 방향 표시
    ax.annotate('', xy=(3, 0.8), xytext=(2, 0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(2.5, 0.85, f'c = {c}', color='red', ha='center')

    plt.tight_layout()
    plt.savefig('advection_exact.png', dpi=150, bbox_inches='tight')
    plt.show()

# exact_advection()
```

### 1.3 특성선

이류방정식의 특성선은 직선 x - ct = const 입니다.

```python
def characteristic_lines():
    """특성선 시각화"""
    c = 1.0
    L = 2.0
    T = 2.0

    fig, ax = plt.subplots(figsize=(10, 6))

    # 특성선 (x - ct = const)
    for x0 in np.linspace(0, L, 11):
        t = np.linspace(0, T, 100)
        x = x0 + c * t
        ax.plot(x, t, 'b-', alpha=0.7)

    # 영역 표시
    ax.axvline(x=0, color='k', linestyle='-', linewidth=2)
    ax.axvline(x=L, color='k', linestyle='-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=2)
    ax.axhline(y=T, color='k', linestyle='--', linewidth=1)

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title(f'특성선: dx/dt = c = {c}')
    ax.set_xlim(-0.5, L + 1)
    ax.set_ylim(-0.1, T + 0.1)
    ax.grid(True, alpha=0.3)

    # 방향 표시
    ax.annotate('', xy=(1.5, 1.5), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(1.2, 0.8, '정보 전파', color='red', fontsize=12)

    plt.tight_layout()
    plt.savefig('characteristic_lines.png', dpi=150, bbox_inches='tight')
    plt.show()

# characteristic_lines()
```

---

## 2. FTCS 불안정성

### 2.1 FTCS 스킴

```
(u_i^{n+1} - u_i^n) / Δt + c · (u_{i+1}^n - u_{i-1}^n) / (2Δx) = 0

정리:
u_i^{n+1} = u_i^n - (C/2) · (u_{i+1}^n - u_{i-1}^n)

여기서 C = c·Δt/Δx (Courant 수)
```

### 2.2 불안정성 원인

von Neumann 분석:
```
증폭인자 G = 1 - iC·sin(kΔx)
|G|² = 1 + C²·sin²(kΔx) > 1  (항상!)

→ FTCS는 이류방정식에 대해 무조건 불안정!
```

```python
class AdvectionFTCS:
    """
    이류방정식 FTCS (불안정)

    주의: 교육 목적으로만 사용. 실제로는 불안정!
    """

    def __init__(self, L=4.0, c=1.0, nx=101, T=1.0, courant=0.5):
        self.L = L
        self.c = c
        self.nx = nx
        self.T = T

        self.dx = L / (nx - 1)
        self.dt = courant * self.dx / abs(c)
        self.nt = int(np.ceil(T / self.dt))
        self.dt = T / self.nt

        self.x = np.linspace(0, L, nx)
        self.C = c * self.dt / self.dx

        print(f"FTCS 이류방정식 (불안정)")
        print(f"  C = {self.C:.4f}")

    def set_initial_condition(self, func):
        self.u = func(self.x)
        self.u0 = self.u.copy()

    def step(self):
        """FTCS 스텝 (불안정)"""
        u_new = self.u.copy()

        # 내부점
        u_new[1:-1] = self.u[1:-1] - (self.C / 2) * (self.u[2:] - self.u[:-2])

        # 주기적 경계조건
        u_new[0] = self.u[0] - (self.C / 2) * (self.u[1] - self.u[-2])
        u_new[-1] = u_new[0]

        self.u = u_new

    def solve(self, save_interval=1):
        history = [self.u0.copy()]
        times = [0]

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), np.array(history)


def demo_ftcs_instability():
    """FTCS 불안정성 시연"""
    solver = AdvectionFTCS(L=4.0, c=1.0, nx=101, T=1.0, courant=0.5)

    def u0(x):
        return np.exp(-(x - 1.0)**2 / 0.08)

    solver.set_initial_condition(u0)
    times, history = solver.solve(save_interval=5)

    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    time_indices = [0, 5, 10, 15, 20, min(25, len(times)-1)]

    for idx, ti in enumerate(time_indices):
        ax = axes[idx // 3, idx % 3]

        # 수치해
        ax.plot(solver.x, history[ti], 'b-', label='수치해', linewidth=2)

        # 해석해
        u_exact = u0(solver.x - solver.c * times[ti])
        ax.plot(solver.x, u_exact, 'r--', label='해석해', linewidth=2)

        ax.set_xlim(0, solver.L)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title(f't = {times[ti]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('FTCS 이류방정식: 무조건 불안정!', fontsize=14, color='red')
    plt.tight_layout()
    plt.savefig('advection_ftcs_unstable.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n최종 시간에서 최대값: {np.max(np.abs(history[-1])):.2e}")
    print("→ 수치 해가 폭발적으로 증가!")

# demo_ftcs_instability()
```

---

## 3. 풍상법 (Upwind Scheme)

### 3.1 풍상법 원리

정보가 흐르는 방향(풍상)에서 공간 미분을 근사:

```
c > 0 (오른쪽으로 이동):
∂u/∂x ≈ (u_i - u_{i-1}) / Δx  (후방차분)

c < 0 (왼쪽으로 이동):
∂u/∂x ≈ (u_{i+1} - u_i) / Δx  (전방차분)
```

### 3.2 스킴

```
c > 0:
u_i^{n+1} = u_i^n - C · (u_i^n - u_{i-1}^n)
         = (1-C)·u_i^n + C·u_{i-1}^n

안정 조건: 0 ≤ C ≤ 1
```

```python
class AdvectionUpwind:
    """
    이류방정식 풍상법 (Upwind)

    조건부 안정: 0 ≤ C ≤ 1
    1차 정확도
    """

    def __init__(self, L=4.0, c=1.0, nx=101, T=1.0, courant=0.8):
        self.L = L
        self.c = c
        self.nx = nx
        self.T = T

        self.dx = L / (nx - 1)
        self.dt = courant * self.dx / abs(c)
        self.nt = int(np.ceil(T / self.dt))
        self.dt = T / self.nt

        self.x = np.linspace(0, L, nx)
        self.C = c * self.dt / self.dx

        print(f"Upwind 이류방정식")
        print(f"  C = {self.C:.4f}")
        print(f"  안정성: {'OK' if 0 <= self.C <= 1 else 'WARNING!'}")

    def set_initial_condition(self, func):
        self.u = func(self.x)
        self.u0 = self.u.copy()

    def step(self):
        """Upwind 스텝"""
        u_new = self.u.copy()

        if self.c > 0:
            # 후방차분 (정보가 왼쪽에서 옴)
            u_new[1:] = self.u[1:] - self.C * (self.u[1:] - self.u[:-1])
            # 왼쪽 경계: 유입 조건 (외부에서 들어오는 값)
            u_new[0] = self.u0[0]  # 또는 특정 값
        else:
            # 전방차분 (정보가 오른쪽에서 옴)
            u_new[:-1] = self.u[:-1] - self.C * (self.u[1:] - self.u[:-1])
            # 오른쪽 경계
            u_new[-1] = self.u0[-1]

        self.u = u_new

    def solve(self, save_interval=1):
        history = [self.u0.copy()]
        times = [0]

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), np.array(history)


def demo_upwind():
    """풍상법 데모"""
    L = 4.0
    c = 1.0
    T = 2.0

    def u0(x):
        return np.exp(-(x - 1.0)**2 / 0.08)

    # 여러 Courant 수 비교
    courant_values = [0.5, 0.8, 0.95]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, C in enumerate(courant_values):
        solver = AdvectionUpwind(L=L, c=c, nx=101, T=T, courant=C)
        solver.set_initial_condition(u0)
        times, history = solver.solve()

        ax = axes[idx]

        # 초기, 중간, 최종
        ax.plot(solver.x, u0(solver.x), 'k--', label='초기', alpha=0.5)
        ax.plot(solver.x, history[-1], 'b-', label='수치해', linewidth=2)
        ax.plot(solver.x, u0(solver.x - c * T), 'r--', label='해석해', linewidth=2)

        error = np.max(np.abs(history[-1] - u0(solver.x - c * T)))
        ax.set_title(f'C = {C}\n최대 오차: {error:.4f}')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, L)

    plt.suptitle('풍상법 (Upwind): Courant 수에 따른 정확도', fontsize=12)
    plt.tight_layout()
    plt.savefig('advection_upwind.png', dpi=150, bbox_inches='tight')
    plt.show()

# demo_upwind()
```

### 3.3 수치 확산

풍상법은 안정하지만 수치 확산(numerical diffusion)을 도입합니다.

```
Modified equation:
∂u/∂t + c·∂u/∂x = (c·Δx/2)·(1 - C)·∂²u/∂x²

→ 인공 확산계수: D_num = (c·Δx/2)·(1 - C)
→ C = 1일 때 확산 없음 (정확해)
```

```python
def numerical_diffusion_demo():
    """수치 확산 시연"""
    L = 4.0
    c = 1.0
    T = 2.0

    def u0(x):
        # 불연속 초기조건 (계단 함수)
        return np.where((x > 0.5) & (x < 1.5), 1.0, 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    courant_values = [0.3, 0.6, 0.95]

    for idx, C in enumerate(courant_values):
        solver = AdvectionUpwind(L=L, c=c, nx=201, T=T, courant=C)
        solver.set_initial_condition(u0)
        times, history = solver.solve()

        ax = axes[idx]

        ax.plot(solver.x, u0(solver.x), 'k--', label='초기', alpha=0.5)
        ax.plot(solver.x, history[-1], 'b-', label='수치해', linewidth=2)
        ax.plot(solver.x, u0(solver.x - c * T), 'r--', label='해석해', linewidth=2)

        # 인공 확산계수
        D_num = (c * solver.dx / 2) * (1 - C)
        ax.set_title(f'C = {C}\n수치 확산: D = {D_num:.4f}')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, L)
        ax.set_ylim(-0.2, 1.4)

    plt.suptitle('풍상법의 수치 확산 (불연속 초기조건)', fontsize=12)
    plt.tight_layout()
    plt.savefig('numerical_diffusion.png', dpi=150, bbox_inches='tight')
    plt.show()

# numerical_diffusion_demo()
```

---

## 4. Lax-Friedrichs 스킴

### 4.1 스킴

FTCS를 수정하여 안정화:

```
u_i^{n+1} = (1/2)·(u_{i+1}^n + u_{i-1}^n) - (C/2)·(u_{i+1}^n - u_{i-1}^n)

= (1/2)·(1+C)·u_{i-1}^n + (1/2)·(1-C)·u_{i+1}^n
```

### 4.2 구현

```python
class AdvectionLaxFriedrichs:
    """
    이류방정식 Lax-Friedrichs 스킴

    조건부 안정: |C| ≤ 1
    1차 정확도
    """

    def __init__(self, L=4.0, c=1.0, nx=101, T=1.0, courant=0.8):
        self.L = L
        self.c = c
        self.nx = nx
        self.T = T

        self.dx = L / (nx - 1)
        self.dt = courant * self.dx / abs(c)
        self.nt = int(np.ceil(T / self.dt))
        self.dt = T / self.nt

        self.x = np.linspace(0, L, nx)
        self.C = c * self.dt / self.dx

        print(f"Lax-Friedrichs 이류방정식")
        print(f"  C = {self.C:.4f}")

    def set_initial_condition(self, func):
        self.u = func(self.x)
        self.u0 = self.u.copy()

    def step(self):
        """Lax-Friedrichs 스텝"""
        u_new = np.zeros_like(self.u)

        # 내부점
        u_new[1:-1] = (0.5 * (self.u[2:] + self.u[:-2]) -
                       (self.C / 2) * (self.u[2:] - self.u[:-2]))

        # 경계: 외삽 또는 고정
        u_new[0] = self.u[0]
        u_new[-1] = self.u[-1]

        self.u = u_new

    def solve(self, save_interval=1):
        history = [self.u0.copy()]
        times = [0]

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), np.array(history)
```

---

## 5. Lax-Wendroff 스킴

### 5.1 이론

2차 정확도를 위해 테일러 전개 사용:

```
u^{n+1} = u^n + Δt·∂u/∂t + (Δt²/2)·∂²u/∂t²

이류방정식에서:
∂u/∂t = -c·∂u/∂x
∂²u/∂t² = c²·∂²u/∂x²

결과:
u_i^{n+1} = u_i^n - (C/2)·(u_{i+1}^n - u_{i-1}^n)
          + (C²/2)·(u_{i+1}^n - 2u_i^n + u_{i-1}^n)
```

### 5.2 구현

```python
class AdvectionLaxWendroff:
    """
    이류방정식 Lax-Wendroff 스킴

    조건부 안정: |C| ≤ 1
    2차 정확도
    """

    def __init__(self, L=4.0, c=1.0, nx=101, T=1.0, courant=0.8):
        self.L = L
        self.c = c
        self.nx = nx
        self.T = T

        self.dx = L / (nx - 1)
        self.dt = courant * self.dx / abs(c)
        self.nt = int(np.ceil(T / self.dt))
        self.dt = T / self.nt

        self.x = np.linspace(0, L, nx)
        self.C = c * self.dt / self.dx
        self.C2 = self.C ** 2

        print(f"Lax-Wendroff 이류방정식")
        print(f"  C = {self.C:.4f}")

    def set_initial_condition(self, func):
        self.u = func(self.x)
        self.u0 = self.u.copy()

    def step(self):
        """Lax-Wendroff 스텝"""
        u_new = np.zeros_like(self.u)

        # 내부점
        u_new[1:-1] = (self.u[1:-1]
                       - (self.C / 2) * (self.u[2:] - self.u[:-2])
                       + (self.C2 / 2) * (self.u[2:] - 2*self.u[1:-1] + self.u[:-2]))

        # 경계
        u_new[0] = self.u[0]
        u_new[-1] = self.u[-1]

        self.u = u_new

    def solve(self, save_interval=1):
        history = [self.u0.copy()]
        times = [0]

        for n in range(self.nt):
            self.step()
            if (n + 1) % save_interval == 0:
                history.append(self.u.copy())
                times.append((n + 1) * self.dt)

        return np.array(times), np.array(history)


def compare_all_schemes():
    """모든 스킴 비교"""
    L = 4.0
    c = 1.0
    T = 2.0
    C = 0.8

    def u0(x):
        return np.exp(-(x - 1.0)**2 / 0.08)

    schemes = {
        'Upwind': AdvectionUpwind(L, c, nx=101, T=T, courant=C),
        'Lax-Friedrichs': AdvectionLaxFriedrichs(L, c, nx=101, T=T, courant=C),
        'Lax-Wendroff': AdvectionLaxWendroff(L, c, nx=101, T=T, courant=C),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (name, solver) in enumerate(schemes.items()):
        solver.set_initial_condition(u0)
        times, history = solver.solve()

        ax = axes[idx]

        ax.plot(solver.x, u0(solver.x), 'k--', alpha=0.5, label='초기')
        ax.plot(solver.x, history[-1], 'b-', linewidth=2, label='수치해')
        ax.plot(solver.x, u0(solver.x - c * T), 'r--', linewidth=2, label='해석해')

        error = np.max(np.abs(history[-1] - u0(solver.x - c * T)))
        ax.set_title(f'{name}\n최대 오차: {error:.4f}')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, L)

    plt.suptitle(f'이류 스킴 비교 (C = {C})', fontsize=12)
    plt.tight_layout()
    plt.savefig('advection_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# compare_all_schemes()
```

---

## 6. 수치 분산 분석

### 6.1 분산 관계

```python
def dispersion_analysis():
    """수치 분산 관계 분석"""
    k_dx = np.linspace(0.01, np.pi, 200)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    C = 0.8

    # 정확한 분산 관계: ω = c·k
    # 정규화: ω·dt = C·k·dx
    omega_exact = C * k_dx

    # 각 스킴의 증폭인자로부터 ω 계산

    # Upwind: G = 1 - C + C·exp(-i·k·dx)
    G_upwind = 1 - C + C * np.exp(-1j * k_dx)
    omega_upwind = -np.angle(G_upwind)

    # Lax-Friedrichs: G = cos(k·dx) - i·C·sin(k·dx)
    G_lf = np.cos(k_dx) - 1j * C * np.sin(k_dx)
    omega_lf = -np.angle(G_lf)

    # Lax-Wendroff: G = 1 - i·C·sin(k·dx) - C²·(1 - cos(k·dx))
    G_lw = 1 - 1j * C * np.sin(k_dx) - C**2 * (1 - np.cos(k_dx))
    omega_lw = -np.angle(G_lw)

    # 위상속도 비교
    ax1 = axes[0]
    ax1.plot(k_dx, omega_exact / k_dx / C, 'k-', linewidth=2, label='정확')
    ax1.plot(k_dx, omega_upwind / k_dx / C, 'b-', linewidth=2, label='Upwind')
    ax1.plot(k_dx, omega_lf / k_dx / C, 'g-', linewidth=2, label='Lax-Friedrichs')
    ax1.plot(k_dx, omega_lw / k_dx / C, 'r-', linewidth=2, label='Lax-Wendroff')

    ax1.set_xlabel('kΔx')
    ax1.set_ylabel('c_num / c')
    ax1.set_title(f'위상속도 오차 (C = {C})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, np.pi)
    ax1.set_ylim(0, 1.2)

    # 진폭 감쇠 (증폭인자 크기)
    ax2 = axes[1]
    ax2.axhline(y=1, color='k', linestyle='-', linewidth=2, label='정확')
    ax2.plot(k_dx, np.abs(G_upwind), 'b-', linewidth=2, label='Upwind')
    ax2.plot(k_dx, np.abs(G_lf), 'g-', linewidth=2, label='Lax-Friedrichs')
    ax2.plot(k_dx, np.abs(G_lw), 'r-', linewidth=2, label='Lax-Wendroff')

    ax2.set_xlabel('kΔx')
    ax2.set_ylabel('|G|')
    ax2.set_title(f'증폭인자 크기 (진폭 감쇠)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, np.pi)
    ax2.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig('advection_dispersion.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("관찰:")
    print("1. Upwind, Lax-Friedrichs: 큰 감쇠 (수치 확산)")
    print("2. Lax-Wendroff: 작은 감쇠, 위상 오차 (수치 분산)")
    print("3. 모든 스킴: 짧은 파장(큰 k)에서 오차 증가")

# dispersion_analysis()
```

### 6.2 분산 vs 확산 효과

```python
def dispersion_diffusion_demo():
    """수치 분산 vs 수치 확산 시각화"""
    L = 8.0
    c = 1.0
    T = 4.0
    C = 0.8

    # 초기조건: 사각 펄스 (불연속)
    def u0_square(x):
        return np.where((x > 1.0) & (x < 2.0), 1.0, 0.0)

    # 초기조건: 가우시안 (매끄러움)
    def u0_gauss(x):
        return np.exp(-(x - 1.5)**2 / 0.1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for row, (u0, name) in enumerate([(u0_square, '사각 펄스'), (u0_gauss, '가우시안')]):
        # Upwind (확산 우세)
        solver1 = AdvectionUpwind(L, c, nx=201, T=T, courant=C)
        solver1.set_initial_condition(u0)
        _, hist1 = solver1.solve()

        ax1 = axes[row, 0]
        ax1.plot(solver1.x, u0(solver1.x), 'k--', alpha=0.5, label='초기')
        ax1.plot(solver1.x, hist1[-1], 'b-', linewidth=2, label='수치해')
        ax1.plot(solver1.x, u0(solver1.x - c * T), 'r--', linewidth=2, label='해석해')
        ax1.set_title(f'Upwind ({name})\n수치 확산')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, L)
        ax1.set_ylim(-0.3, 1.3)

        # Lax-Wendroff (분산 우세)
        solver2 = AdvectionLaxWendroff(L, c, nx=201, T=T, courant=C)
        solver2.set_initial_condition(u0)
        _, hist2 = solver2.solve()

        ax2 = axes[row, 1]
        ax2.plot(solver2.x, u0(solver2.x), 'k--', alpha=0.5, label='초기')
        ax2.plot(solver2.x, hist2[-1], 'b-', linewidth=2, label='수치해')
        ax2.plot(solver2.x, u0(solver2.x - c * T), 'r--', linewidth=2, label='해석해')
        ax2.set_title(f'Lax-Wendroff ({name})\n수치 분산 (진동)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, L)
        ax2.set_ylim(-0.3, 1.3)

        # 오차 비교
        ax3 = axes[row, 2]
        exact = u0(solver1.x - c * T)
        ax3.plot(solver1.x, hist1[-1] - exact, 'b-', label='Upwind 오차')
        ax3.plot(solver2.x, hist2[-1] - exact, 'r-', label='Lax-Wendroff 오차')
        ax3.set_title(f'오차 비교 ({name})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, L)
        ax3.set_ylim(-0.5, 0.5)

        for ax in axes[row]:
            ax.set_xlabel('x')
            ax.set_ylabel('u')

    plt.tight_layout()
    plt.savefig('dispersion_vs_diffusion.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("핵심 관찰:")
    print("- Upwind: 매끄럽지만 퍼짐 (확산)")
    print("- Lax-Wendroff: 날카롭지만 진동 (분산)")
    print("- 불연속 해에서 두 효과 모두 문제")

# dispersion_diffusion_demo()
```

---

## 7. Courant 수의 중요성

### 7.1 안정성과 정확도

```python
def courant_number_study():
    """Courant 수에 따른 안정성과 정확도"""
    L = 4.0
    c = 1.0
    T = 1.0

    def u0(x):
        return np.exp(-(x - 1.0)**2 / 0.08)

    courant_values = [0.2, 0.5, 0.8, 0.95, 1.0, 1.05]
    errors = {'Upwind': [], 'Lax-Wendroff': []}
    stable = {'Upwind': [], 'Lax-Wendroff': []}

    for C in courant_values:
        try:
            # Upwind
            solver1 = AdvectionUpwind(L, c, nx=101, T=T, courant=C)
            solver1.set_initial_condition(u0)
            _, hist1 = solver1.solve()
            err1 = np.max(np.abs(hist1[-1] - u0(solver1.x - c * T)))
            errors['Upwind'].append(err1)
            stable['Upwind'].append(err1 < 10)

            # Lax-Wendroff
            solver2 = AdvectionLaxWendroff(L, c, nx=101, T=T, courant=C)
            solver2.set_initial_condition(u0)
            _, hist2 = solver2.solve()
            err2 = np.max(np.abs(hist2[-1] - u0(solver2.x - c * T)))
            errors['Lax-Wendroff'].append(err2)
            stable['Lax-Wendroff'].append(err2 < 10)

        except:
            errors['Upwind'].append(np.nan)
            errors['Lax-Wendroff'].append(np.nan)
            stable['Upwind'].append(False)
            stable['Lax-Wendroff'].append(False)

    # 시각화
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(courant_values))
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, errors['Upwind'], width, label='Upwind', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, errors['Lax-Wendroff'], width, label='Lax-Wendroff', alpha=0.8)

    # 불안정 표시
    for i, C in enumerate(courant_values):
        if not stable['Upwind'][i]:
            bars1[i].set_color('red')
        if not stable['Lax-Wendroff'][i]:
            bars2[i].set_color('red')

    ax.set_xlabel('Courant 수 C')
    ax.set_ylabel('최대 오차')
    ax.set_title('Courant 수에 따른 정확도 (빨간색 = 불안정)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(courant_values)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 안정 영역 표시
    ax.axvline(x=4.5, color='red', linestyle='--', alpha=0.5)
    ax.text(4.7, ax.get_ylim()[1]*0.9, 'C > 1\n불안정', color='red', fontsize=10)

    plt.tight_layout()
    plt.savefig('courant_study.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n결과 요약:")
    for C, e1, e2 in zip(courant_values, errors['Upwind'], errors['Lax-Wendroff']):
        status = 'UNSTABLE' if C > 1 else ''
        print(f"C = {C}: Upwind = {e1:.4f}, L-W = {e2:.4f} {status}")

# courant_number_study()
```

---

## 8. 종합 예제: 오염 물질 이동

```python
def pollution_transport():
    """오염 물질 이동 시뮬레이션"""
    L = 10.0  # 강 길이 (km)
    c = 1.0   # 유속 (km/h)
    T = 8.0   # 시뮬레이션 시간 (h)

    # 초기 오염 분포: 0~2 km 구간에서 농도 높음
    def u0(x):
        return np.where((x > 0) & (x < 2), np.sin(np.pi * x / 2)**2, 0)

    # 고해상도 참조해
    solver_ref = AdvectionUpwind(L, c, nx=1001, T=T, courant=0.99)
    solver_ref.set_initial_condition(u0)
    _, hist_ref = solver_ref.solve()

    # 저해상도 비교
    solvers = {
        'Upwind (거친 격자)': AdvectionUpwind(L, c, nx=51, T=T, courant=0.8),
        'Lax-Wendroff (거친 격자)': AdvectionLaxWendroff(L, c, nx=51, T=T, courant=0.8),
        '고해상도 참조': solver_ref,
    }

    results = {}
    for name, solver in solvers.items():
        if name != '고해상도 참조':
            solver.set_initial_condition(u0)
            _, hist = solver.solve()
            results[name] = (solver, hist)
        else:
            results[name] = (solver, hist_ref)

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # t = 0
    ax1 = axes[0, 0]
    ax1.fill_between(solver_ref.x, 0, hist_ref[0], alpha=0.3, color='blue')
    ax1.plot(solver_ref.x, hist_ref[0], 'b-', linewidth=2)
    ax1.set_title('초기 오염 분포 (t = 0)')
    ax1.set_xlabel('거리 (km)')
    ax1.set_ylabel('오염 농도')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, L)

    # t = T/2
    ax2 = axes[0, 1]
    mid_idx = len(hist_ref) // 2
    ax2.plot(solver_ref.x, hist_ref[mid_idx], 'k--', linewidth=2, label='참조해')
    for name, (solver, hist) in results.items():
        if name != '고해상도 참조':
            mid = len(hist) // 2
            ax2.plot(solver.x, hist[mid], linewidth=2, label=name)
    ax2.set_title(f't = {T/2} 시간')
    ax2.set_xlabel('거리 (km)')
    ax2.set_ylabel('오염 농도')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, L)

    # t = T
    ax3 = axes[1, 0]
    ax3.plot(solver_ref.x, hist_ref[-1], 'k--', linewidth=2, label='참조해')
    for name, (solver, hist) in results.items():
        if name != '고해상도 참조':
            ax3.plot(solver.x, hist[-1], linewidth=2, label=name)
    ax3.set_title(f't = {T} 시간 (최종)')
    ax3.set_xlabel('거리 (km)')
    ax3.set_ylabel('오염 농도')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, L)

    # 시공간도
    ax4 = axes[1, 1]
    solver = results['Upwind (거친 격자)'][0]
    hist = results['Upwind (거친 격자)'][1]
    times = np.linspace(0, T, len(hist))
    X, Time = np.meshgrid(solver.x, times)
    c = ax4.contourf(X, Time, hist, levels=20, cmap='YlOrRd')
    plt.colorbar(c, ax=ax4, label='농도')
    ax4.set_xlabel('거리 (km)')
    ax4.set_ylabel('시간 (h)')
    ax4.set_title('시공간 오염 분포 (Upwind)')

    plt.suptitle('강의 오염 물질 이동 시뮬레이션', fontsize=14)
    plt.tight_layout()
    plt.savefig('pollution_transport.png', dpi=150, bbox_inches='tight')
    plt.show()

# pollution_transport()
```

---

## 9. 요약

### 스킴 비교표

| 스킴 | 정확도 | 안정성 | 특성 |
|------|--------|--------|------|
| FTCS | O(Δt, Δx²) | **무조건 불안정** | 사용 금지 |
| Upwind | O(Δt, Δx) | C ≤ 1 | 수치 확산 |
| Lax-Friedrichs | O(Δt, Δx) | C ≤ 1 | 큰 수치 확산 |
| Lax-Wendroff | O(Δt², Δx²) | C ≤ 1 | 수치 분산 (진동) |

### CFL 조건

```
C = c·Δt/Δx ≤ 1

물리적 의미:
- 수치적 정보 전파 속도 ≥ 물리적 전파 속도
- C = 1: Upwind가 정확해
```

### 수치 오차 종류

| 유형 | 원인 | 효과 | 대표 스킴 |
|------|------|------|----------|
| 수치 확산 | 홀수차 절단오차 | 해가 퍼짐 | Upwind |
| 수치 분산 | 짝수차 절단오차 | 진동 발생 | Lax-Wendroff |

---

## 연습문제

### 연습 1: FTCS 불안정성 확인
여러 Courant 수에서 FTCS를 실행하고 불안정성을 확인하시오.

### 연습 2: 역방향 이류
c < 0일 때 Upwind 스킴을 수정하고 테스트하시오.

### 연습 3: Beam-Warming 스킴
2차 풍상법(Beam-Warming)을 구현하고 Lax-Wendroff와 비교하시오.

### 연습 4: 2D 이류
∂u/∂t + c_x·∂u/∂x + c_y·∂u/∂y = 0을 풀어보시오.

---

## 참고 자료

1. **교재**: LeVeque, "Numerical Methods for Conservation Laws"
2. **CFD**: Versteeg & Malalasekera, "An Introduction to CFD"
3. **수치분석**: Strikwerda, "Finite Difference Schemes and PDEs"
