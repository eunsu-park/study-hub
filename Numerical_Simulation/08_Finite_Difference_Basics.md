# 08. 유한차분법 기초 (Finite Difference Method Basics)

## 학습 목표
- 유한차분법의 기본 원리 이해
- 격자/메쉬 생성 방법 학습
- 전방/후방/중심 차분의 유도와 정확도 분석
- 절단오차(Truncation Error) 개념 이해
- CFL 조건과 von Neumann 안정성 분석

---

## 1. 유한차분법 소개

### 1.1 기본 아이디어

유한차분법(Finite Difference Method, FDM)은 미분을 유한한 차분으로 근사하는 방법입니다.

```
미분의 정의:
f'(x) = lim[h→0] (f(x+h) - f(x)) / h

유한차분 근사 (h가 작은 유한값):
f'(x) ≈ (f(x+h) - f(x)) / h
```

```python
import numpy as np
import matplotlib.pyplot as plt

def finite_difference_demo():
    """유한차분 기본 개념 시연"""

    # 테스트 함수: f(x) = sin(x)
    # 정확한 미분: f'(x) = cos(x)
    f = lambda x: np.sin(x)
    f_exact = lambda x: np.cos(x)

    x = np.pi / 4  # 테스트 점
    h_values = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

    print(f"테스트 함수: f(x) = sin(x), x = π/4")
    print(f"정확한 미분값: f'(π/4) = cos(π/4) = {f_exact(x):.10f}")
    print()
    print(f"{'h':<10} {'전방차분':<15} {'후방차분':<15} {'중심차분':<15}")
    print("-" * 55)

    for h in h_values:
        # 전방차분 (Forward Difference)
        forward = (f(x + h) - f(x)) / h

        # 후방차분 (Backward Difference)
        backward = (f(x) - f(x - h)) / h

        # 중심차분 (Central Difference)
        central = (f(x + h) - f(x - h)) / (2 * h)

        print(f"{h:<10.4f} {forward:<15.10f} {backward:<15.10f} {central:<15.10f}")

    print()
    print("관찰: 중심차분이 가장 정확합니다 (2차 정확도)")

finite_difference_demo()
```

### 1.2 왜 유한차분법인가?

| 장점 | 단점 |
|------|------|
| 구현이 간단함 | 복잡한 기하형상에 부적합 |
| 직관적 이해 가능 | 불규칙 격자 처리 어려움 |
| 계산 효율적 | 국소 해상도 조절 제한 |
| 고차 정확도 가능 | 경계조건 처리 복잡할 수 있음 |

---

## 2. 격자/메쉬 생성

### 2.1 1D 균등 격자

```python
import numpy as np

def create_1d_grid(x_min, x_max, nx):
    """
    1D 균등 격자 생성

    Parameters:
    -----------
    x_min : float - 시작점
    x_max : float - 끝점
    nx : int - 격자점 수

    Returns:
    --------
    x : array - 격자점 좌표
    dx : float - 격자 간격
    """
    x = np.linspace(x_min, x_max, nx)
    dx = (x_max - x_min) / (nx - 1)
    return x, dx

# 예시
x, dx = create_1d_grid(0, 1, 11)
print(f"격자점: {x}")
print(f"격자 간격 dx = {dx}")
print(f"내부 격자점 수: {len(x) - 2}")  # 경계 제외
```

### 2.2 2D 균등 격자

```python
def create_2d_grid(x_range, y_range, nx, ny):
    """
    2D 균등 격자 생성

    Parameters:
    -----------
    x_range : tuple - (x_min, x_max)
    y_range : tuple - (y_min, y_max)
    nx, ny : int - 각 방향 격자점 수

    Returns:
    --------
    X, Y : 2D arrays - 격자점 좌표
    dx, dy : float - 격자 간격
    """
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)

    dx = (x_range[1] - x_range[0]) / (nx - 1)
    dy = (y_range[1] - y_range[0]) / (ny - 1)

    X, Y = np.meshgrid(x, y)

    return X, Y, dx, dy

# 예시
X, Y, dx, dy = create_2d_grid((0, 1), (0, 1), 11, 11)
print(f"격자 크기: {X.shape}")
print(f"dx = {dx}, dy = {dy}")
```

### 2.3 시공간 격자

```python
def create_spacetime_grid(x_range, t_range, nx, nt):
    """
    시공간 격자 생성 (1D 공간 + 시간)

    Parameters:
    -----------
    x_range : tuple - (x_min, x_max)
    t_range : tuple - (t_start, t_end)
    nx : int - 공간 격자점 수
    nt : int - 시간 스텝 수

    Returns:
    --------
    x, t : arrays - 좌표
    dx, dt : float - 간격
    """
    x = np.linspace(x_range[0], x_range[1], nx)
    t = np.linspace(t_range[0], t_range[1], nt + 1)

    dx = (x_range[1] - x_range[0]) / (nx - 1)
    dt = (t_range[1] - t_range[0]) / nt

    return x, t, dx, dt

# 예시
x, t, dx, dt = create_spacetime_grid((0, 1), (0, 0.5), 51, 100)
print(f"공간 격자점: {len(x)}, dx = {dx:.4f}")
print(f"시간 스텝: {len(t)-1}, dt = {dt:.6f}")
```

### 2.4 격자 시각화

```python
import matplotlib.pyplot as plt

def visualize_grids():
    """격자 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1D 격자
    ax1 = axes[0]
    x, dx = create_1d_grid(0, 1, 11)
    ax1.scatter(x, np.zeros_like(x), s=50, c='blue')
    for i, xi in enumerate(x):
        ax1.axvline(x=xi, color='gray', linestyle='--', alpha=0.3)
        ax1.annotate(f'$x_{{{i}}}$', (xi, 0.02), ha='center', fontsize=8)
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 0.2)
    ax1.set_title('1D 균등 격자', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_yticks([])
    ax1.annotate(f'dx = {dx:.2f}', (0.5, 0.1), ha='center', fontsize=10)

    # 2D 격자
    ax2 = axes[1]
    X, Y, dx, dy = create_2d_grid((0, 1), (0, 1), 6, 6)
    ax2.scatter(X, Y, s=30, c='blue')
    for i in range(X.shape[0]):
        ax2.axhline(y=Y[i, 0], color='gray', linestyle='--', alpha=0.3)
    for j in range(X.shape[1]):
        ax2.axvline(x=X[0, j], color='gray', linestyle='--', alpha=0.3)
    ax2.set_title('2D 균등 격자', fontsize=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')

    # 시공간 격자
    ax3 = axes[2]
    x_st = np.linspace(0, 1, 6)
    t_st = np.linspace(0, 0.5, 4)
    X_st, T_st = np.meshgrid(x_st, t_st)
    ax3.scatter(X_st, T_st, s=30, c='blue')
    for i in range(len(t_st)):
        ax3.axhline(y=t_st[i], color='gray', linestyle='--', alpha=0.3)
    for j in range(len(x_st)):
        ax3.axvline(x=x_st[j], color='gray', linestyle='--', alpha=0.3)
    ax3.set_title('시공간 격자', fontsize=12)
    ax3.set_xlabel('x (공간)')
    ax3.set_ylabel('t (시간)')

    # 시간 진행 방향 표시
    ax3.annotate('', xy=(1.1, 0.4), xytext=(1.1, 0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax3.text(1.15, 0.25, '시간 진행', rotation=90, va='center', color='red')

    plt.tight_layout()
    plt.savefig('grids.png', dpi=150, bbox_inches='tight')
    plt.show()

# visualize_grids()
```

---

## 3. 차분 공식 유도

### 3.1 테일러 전개 기반 유도

테일러 급수를 사용하여 차분 공식을 유도합니다.

```
f(x+h) = f(x) + h·f'(x) + (h²/2!)·f''(x) + (h³/3!)·f'''(x) + O(h⁴)
f(x-h) = f(x) - h·f'(x) + (h²/2!)·f''(x) - (h³/3!)·f'''(x) + O(h⁴)
```

```python
def derive_difference_formulas():
    """차분 공식 유도 과정"""

    print("=" * 60)
    print("차분 공식 유도 (테일러 전개)")
    print("=" * 60)

    # 전방차분 유도
    print("\n[1] 전방차분 (Forward Difference)")
    print("    f(x+h) = f(x) + h·f'(x) + (h²/2)·f''(x) + O(h³)")
    print("    정리하면:")
    print("    f'(x) = [f(x+h) - f(x)] / h - (h/2)·f''(ξ)")
    print("    ≈ [f(x+h) - f(x)] / h + O(h)")
    print("    → 1차 정확도 (First-order accurate)")

    # 후방차분 유도
    print("\n[2] 후방차분 (Backward Difference)")
    print("    f(x-h) = f(x) - h·f'(x) + (h²/2)·f''(x) + O(h³)")
    print("    정리하면:")
    print("    f'(x) = [f(x) - f(x-h)] / h + (h/2)·f''(ξ)")
    print("    ≈ [f(x) - f(x-h)] / h + O(h)")
    print("    → 1차 정확도")

    # 중심차분 유도
    print("\n[3] 중심차분 (Central Difference)")
    print("    f(x+h) - f(x-h) = 2h·f'(x) + (2h³/6)·f'''(x) + O(h⁵)")
    print("    정리하면:")
    print("    f'(x) = [f(x+h) - f(x-h)] / 2h - (h²/6)·f'''(ξ)")
    print("    ≈ [f(x+h) - f(x-h)] / 2h + O(h²)")
    print("    → 2차 정확도 (Second-order accurate)")

    # 2차 미분 중심차분
    print("\n[4] 2차 미분 중심차분")
    print("    f(x+h) + f(x-h) = 2f(x) + h²·f''(x) + (h⁴/12)·f''''(x) + O(h⁶)")
    print("    정리하면:")
    print("    f''(x) = [f(x+h) - 2f(x) + f(x-h)] / h² + O(h²)")
    print("    → 2차 정확도")

derive_difference_formulas()
```

### 3.2 차분 공식 요약

#### 1차 미분 (∂u/∂x)

| 이름 | 공식 | 정확도 | 스텐실 |
|------|------|--------|--------|
| 전방차분 | (u_{i+1} - u_i) / Δx | O(Δx) | [i, i+1] |
| 후방차분 | (u_i - u_{i-1}) / Δx | O(Δx) | [i-1, i] |
| 중심차분 | (u_{i+1} - u_{i-1}) / 2Δx | O(Δx²) | [i-1, i+1] |

#### 2차 미분 (∂²u/∂x²)

| 이름 | 공식 | 정확도 |
|------|------|--------|
| 중심차분 | (u_{i+1} - 2u_i + u_{i-1}) / Δx² | O(Δx²) |

```python
def difference_operators():
    """차분 연산자 구현"""

    def forward_diff(u, dx, i):
        """전방차분: ∂u/∂x ≈ (u[i+1] - u[i]) / dx"""
        return (u[i+1] - u[i]) / dx

    def backward_diff(u, dx, i):
        """후방차분: ∂u/∂x ≈ (u[i] - u[i-1]) / dx"""
        return (u[i] - u[i-1]) / dx

    def central_diff_1st(u, dx, i):
        """중심차분 (1차 미분): ∂u/∂x ≈ (u[i+1] - u[i-1]) / (2*dx)"""
        return (u[i+1] - u[i-1]) / (2 * dx)

    def central_diff_2nd(u, dx, i):
        """중심차분 (2차 미분): ∂²u/∂x² ≈ (u[i+1] - 2*u[i] + u[i-1]) / dx²"""
        return (u[i+1] - 2*u[i] + u[i-1]) / dx**2

    return forward_diff, backward_diff, central_diff_1st, central_diff_2nd

# 벡터화된 버전
def apply_diff_operators(u, dx):
    """전체 배열에 차분 연산자 적용"""

    # 1차 미분 (중심차분, 내부점)
    du_dx = np.zeros_like(u)
    du_dx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
    # 경계: 전방/후방 차분
    du_dx[0] = (u[1] - u[0]) / dx
    du_dx[-1] = (u[-1] - u[-2]) / dx

    # 2차 미분 (중심차분, 내부점)
    d2u_dx2 = np.zeros_like(u)
    d2u_dx2[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2

    return du_dx, d2u_dx2

# 테스트
x = np.linspace(0, np.pi, 101)
dx = x[1] - x[0]
u = np.sin(x)  # u = sin(x)

du_dx, d2u_dx2 = apply_diff_operators(u, dx)

# 비교
print(f"위치 x = π/2에서:")
print(f"  수치 ∂u/∂x = {du_dx[50]:.6f}, 정확값 = {np.cos(x[50]):.6f}")
print(f"  수치 ∂²u/∂x² = {d2u_dx2[50]:.6f}, 정확값 = {-np.sin(x[50]):.6f}")
```

---

## 4. 절단오차 분석 (Truncation Error Analysis)

### 4.1 절단오차란?

절단오차는 미분을 차분으로 근사할 때 잘라낸(truncate) 고차 항들입니다.

```python
def truncation_error_analysis():
    """절단오차 분석"""

    # f(x) = sin(x), f''(x) = -sin(x)
    f = lambda x: np.sin(x)
    f_exact = lambda x: np.cos(x)
    f_2nd = lambda x: -np.sin(x)

    x = np.pi / 4
    h_values = np.array([0.2, 0.1, 0.05, 0.025, 0.0125])

    errors_forward = []
    errors_central = []

    for h in h_values:
        # 전방차분 오차
        forward = (f(x + h) - f(x)) / h
        err_forward = abs(forward - f_exact(x))
        errors_forward.append(err_forward)

        # 중심차분 오차
        central = (f(x + h) - f(x - h)) / (2 * h)
        err_central = abs(central - f_exact(x))
        errors_central.append(err_central)

    errors_forward = np.array(errors_forward)
    errors_central = np.array(errors_central)

    # 수렴 차수 계산
    order_forward = np.log(errors_forward[:-1] / errors_forward[1:]) / np.log(2)
    order_central = np.log(errors_central[:-1] / errors_central[1:]) / np.log(2)

    print("절단오차 분석")
    print("=" * 70)
    print(f"{'h':<12} {'전방차분 오차':<18} {'차수':<8} {'중심차분 오차':<18} {'차수':<8}")
    print("-" * 70)

    for i, h in enumerate(h_values):
        order_f = order_forward[i-1] if i > 0 else '-'
        order_c = order_central[i-1] if i > 0 else '-'
        if i > 0:
            print(f"{h:<12.4f} {errors_forward[i]:<18.2e} {order_f:<8.2f} {errors_central[i]:<18.2e} {order_c:<8.2f}")
        else:
            print(f"{h:<12.4f} {errors_forward[i]:<18.2e} {'-':<8} {errors_central[i]:<18.2e} {'-':<8}")

    print()
    print("결론:")
    print(f"  전방차분: O(h) - 1차 정확도 (h를 반으로 줄이면 오차도 반)")
    print(f"  중심차분: O(h²) - 2차 정확도 (h를 반으로 줄이면 오차가 1/4)")

    return h_values, errors_forward, errors_central

h_values, errors_forward, errors_central = truncation_error_analysis()
```

### 4.2 오차 수렴 시각화

```python
def plot_convergence():
    """오차 수렴 시각화"""
    fig, ax = plt.subplots(figsize=(10, 6))

    h_values, errors_forward, errors_central = truncation_error_analysis()

    # 로그-로그 플롯
    ax.loglog(h_values, errors_forward, 'o-', label='전방차분 (1차)', linewidth=2, markersize=8)
    ax.loglog(h_values, errors_central, 's-', label='중심차분 (2차)', linewidth=2, markersize=8)

    # 기준선
    ax.loglog(h_values, h_values * 0.5, 'k--', alpha=0.5, label='O(h)')
    ax.loglog(h_values, h_values**2 * 2, 'k:', alpha=0.5, label='O(h²)')

    ax.set_xlabel('h (격자 간격)', fontsize=12)
    ax.set_ylabel('오차', fontsize=12)
    ax.set_title('차분법 절단오차 수렴', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('truncation_error.png', dpi=150, bbox_inches='tight')
    plt.show()

# plot_convergence()
```

---

## 5. CFL 조건 (Courant-Friedrichs-Lewy Condition)

### 5.1 CFL 조건이란?

CFL 조건은 시간에 의존하는 PDE의 수치해법에서 안정성을 보장하는 조건입니다.

```
물리적 의미:
- 수치 정보 전파 속도 ≥ 물리적 정보 전파 속도

수학적 조건:
- 이류방정식: c·Δt/Δx ≤ 1 (Courant 수 ≤ 1)
- 열방정식: α·Δt/Δx² ≤ 1/2
- 파동방정식: c·Δt/Δx ≤ 1
```

```python
def cfl_condition_demo():
    """CFL 조건 시연"""

    print("CFL 조건 (Courant-Friedrichs-Lewy)")
    print("=" * 60)

    # 이류방정식
    print("\n[1] 이류방정식: ∂u/∂t + c·∂u/∂x = 0")
    print("    CFL 조건: C = c·Δt/Δx ≤ 1")
    print("    → Courant 수 C가 1보다 작거나 같아야 안정")

    c = 1.0  # 전파속도
    dx = 0.1
    dt_max = dx / c
    print(f"    예: c = {c}, Δx = {dx}")
    print(f"    최대 허용 Δt = {dt_max}")

    # 열방정식
    print("\n[2] 열방정식: ∂u/∂t = α·∂²u/∂x² (FTCS)")
    print("    CFL 조건: r = α·Δt/Δx² ≤ 1/2")

    alpha = 0.01  # 열확산계수
    dx = 0.1
    dt_max = 0.5 * dx**2 / alpha
    print(f"    예: α = {alpha}, Δx = {dx}")
    print(f"    최대 허용 Δt = {dt_max}")

    # 파동방정식
    print("\n[3] 파동방정식: ∂²u/∂t² = c²·∂²u/∂x²")
    print("    CFL 조건: C = c·Δt/Δx ≤ 1")

    c = 1.0
    dx = 0.1
    dt_max = dx / c
    print(f"    예: c = {c}, Δx = {dx}")
    print(f"    최대 허용 Δt = {dt_max}")

cfl_condition_demo()
```

### 5.2 CFL 조건 계산기

```python
class CFLCalculator:
    """CFL 조건 계산 및 검증"""

    @staticmethod
    def advection(c, dx, dt):
        """
        이류방정식 CFL 수 계산

        Parameters:
        -----------
        c : float - 전파속도
        dx : float - 공간 격자 간격
        dt : float - 시간 간격

        Returns:
        --------
        C : float - Courant 수
        stable : bool - 안정성 여부
        """
        C = abs(c) * dt / dx
        stable = C <= 1.0
        return C, stable

    @staticmethod
    def heat_ftcs(alpha, dx, dt):
        """
        열방정식 (FTCS) CFL 수 계산

        Parameters:
        -----------
        alpha : float - 열확산계수
        """
        r = alpha * dt / dx**2
        stable = r <= 0.5
        return r, stable

    @staticmethod
    def wave(c, dx, dt):
        """파동방정식 CFL 수 계산"""
        C = c * dt / dx
        stable = C <= 1.0
        return C, stable

    @staticmethod
    def max_dt_advection(c, dx, safety=0.9):
        """이류방정식 최대 허용 dt 계산"""
        return safety * dx / abs(c)

    @staticmethod
    def max_dt_heat(alpha, dx, safety=0.9):
        """열방정식 (FTCS) 최대 허용 dt 계산"""
        return safety * 0.5 * dx**2 / alpha

    @staticmethod
    def max_dt_wave(c, dx, safety=0.9):
        """파동방정식 최대 허용 dt 계산"""
        return safety * dx / c

# 사용 예시
cfl = CFLCalculator()

# 열방정식 예시
alpha = 0.01
dx = 0.02
dt = 0.001

r, stable = cfl.heat_ftcs(alpha, dx, dt)
print(f"열방정식 CFL 분석:")
print(f"  α = {alpha}, Δx = {dx}, Δt = {dt}")
print(f"  r = α·Δt/Δx² = {r:.4f}")
print(f"  안정: {stable}")
print(f"  권장 최대 Δt = {cfl.max_dt_heat(alpha, dx):.6f}")
```

---

## 6. von Neumann 안정성 분석

### 6.1 분석 원리

von Neumann 안정성 분석은 푸리에 모드의 시간에 따른 성장을 분석합니다.

```
가정: u_j^n = G^n · e^{i·k·j·Δx}

여기서:
- G: 증폭인자 (amplification factor)
- k: 파수 (wave number)
- j: 공간 인덱스
- n: 시간 스텝

안정 조건: |G| ≤ 1 (모든 k에 대해)
```

```python
def von_neumann_analysis():
    """von Neumann 안정성 분석"""

    print("von Neumann 안정성 분석")
    print("=" * 60)

    # FTCS 열방정식 분석
    print("\n[예제] FTCS 열방정식")
    print("    u_j^{n+1} = u_j^n + r·(u_{j+1}^n - 2·u_j^n + u_{j-1}^n)")
    print("    여기서 r = α·Δt/Δx²")
    print()
    print("    u_j^n = G^n·e^{ikjΔx} 대입:")
    print("    G·e^{ikjΔx} = e^{ikjΔx} + r·(e^{ik(j+1)Δx} - 2·e^{ikjΔx} + e^{ik(j-1)Δx})")
    print()
    print("    양변을 e^{ikjΔx}로 나누면:")
    print("    G = 1 + r·(e^{ikΔx} + e^{-ikΔx} - 2)")
    print("      = 1 + r·(2cos(kΔx) - 2)")
    print("      = 1 - 2r·(1 - cos(kΔx))")
    print("      = 1 - 4r·sin²(kΔx/2)")
    print()
    print("    안정 조건 |G| ≤ 1:")
    print("    -1 ≤ 1 - 4r·sin²(kΔx/2) ≤ 1")
    print()
    print("    왼쪽 부등식에서:")
    print("    -2 ≤ -4r·sin²(kΔx/2)")
    print("    4r·sin²(kΔx/2) ≤ 2")
    print()
    print("    sin²(kΔx/2)의 최대값 = 1 (kΔx = π):")
    print("    4r ≤ 2")
    print("    r ≤ 1/2")
    print()
    print("    결론: FTCS 열방정식은 r = α·Δt/Δx² ≤ 1/2 일 때 안정")

von_neumann_analysis()
```

### 6.2 증폭인자 시각화

```python
def plot_amplification_factor():
    """증폭인자 시각화"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # k*dx 범위
    kh = np.linspace(0, np.pi, 100)

    # FTCS 열방정식 증폭인자
    ax1 = axes[0]
    r_values = [0.1, 0.25, 0.5, 0.6, 0.8]

    for r in r_values:
        G = 1 - 4 * r * np.sin(kh / 2)**2
        label = f'r = {r}' + (' (불안정)' if r > 0.5 else '')
        linestyle = '--' if r > 0.5 else '-'
        ax1.plot(kh, G, label=label, linestyle=linestyle, linewidth=2)

    ax1.axhline(y=1, color='red', linestyle=':', alpha=0.7)
    ax1.axhline(y=-1, color='red', linestyle=':', alpha=0.7)
    ax1.fill_between(kh, -1, 1, alpha=0.1, color='green', label='안정 영역')
    ax1.set_xlabel('kΔx', fontsize=12)
    ax1.set_ylabel('G (증폭인자)', fontsize=12)
    ax1.set_title('FTCS 열방정식 증폭인자\nG = 1 - 4r·sin²(kΔx/2)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, np.pi)
    ax1.set_ylim(-2, 1.5)

    # 여러 스킴 비교 (이류방정식)
    ax2 = axes[1]
    C = 0.8  # Courant 수

    # FTCS (불안정)
    G_ftcs = 1 - 1j * C * np.sin(kh)
    ax2.plot(kh, np.abs(G_ftcs), label='FTCS (불안정)', linewidth=2)

    # 풍상법 (Upwind)
    G_upwind = 1 - C * (1 - np.cos(kh)) - 1j * C * np.sin(kh)
    ax2.plot(kh, np.abs(G_upwind), label='Upwind', linewidth=2)

    # Lax-Friedrichs
    G_lax = np.cos(kh) - 1j * C * np.sin(kh)
    ax2.plot(kh, np.abs(G_lax), label='Lax-Friedrichs', linewidth=2)

    ax2.axhline(y=1, color='red', linestyle=':', alpha=0.7, label='안정 한계')
    ax2.set_xlabel('kΔx', fontsize=12)
    ax2.set_ylabel('|G| (증폭인자 크기)', fontsize=12)
    ax2.set_title(f'이류방정식 증폭인자 비교 (C = {C})', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, np.pi)
    ax2.set_ylim(0, 2)

    plt.tight_layout()
    plt.savefig('amplification_factor.png', dpi=150, bbox_inches='tight')
    plt.show()

# plot_amplification_factor()
```

### 6.3 수치 실험으로 확인

```python
def stability_experiment():
    """안정성 수치 실험"""

    # 파라미터
    L = 1.0
    nx = 51
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)

    alpha = 0.01

    # 초기조건
    u0 = np.sin(np.pi * x)

    # FTCS 스킴
    def ftcs_step(u, r):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
        u_new[0] = 0  # 경계조건
        u_new[-1] = 0
        return u_new

    # 여러 r 값으로 실험
    r_values = [0.4, 0.5, 0.6]
    n_steps = 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, r in enumerate(r_values):
        dt = r * dx**2 / alpha
        u = u0.copy()

        ax = axes[idx]
        ax.plot(x, u0, 'b--', label='초기', alpha=0.5)

        for step in range(n_steps):
            u = ftcs_step(u, r)
            if step in [20, 50, 99]:
                ax.plot(x, u, label=f'step {step+1}')

        stable = r <= 0.5
        status = "안정" if stable else "불안정"
        ax.set_title(f'r = {r} ({status})\ndt = {dt:.6f}', fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if stable:
            ax.set_ylim(-1.5, 1.5)
        else:
            ax.set_ylim(-10, 10)

    plt.tight_layout()
    plt.savefig('stability_experiment.png', dpi=150, bbox_inches='tight')
    plt.show()

# stability_experiment()
```

---

## 7. 고차 정확도 차분 공식

### 7.1 4차 정확도 공식

```python
def high_order_formulas():
    """고차 정확도 차분 공식"""

    print("고차 정확도 차분 공식")
    print("=" * 60)

    # 1차 미분 4차 정확도
    print("\n[1] 1차 미분 (4차 정확도)")
    print("    f'(x) ≈ [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / 12h")
    print("    절단오차: O(h⁴)")

    # 2차 미분 4차 정확도
    print("\n[2] 2차 미분 (4차 정확도)")
    print("    f''(x) ≈ [-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)] / 12h²")
    print("    절단오차: O(h⁴)")

    # 수치 검증
    f = lambda x: np.sin(x)
    f_1 = lambda x: np.cos(x)
    f_2 = lambda x: -np.sin(x)

    x = np.pi / 4
    h = 0.1

    # 1차 미분
    d1_2nd = (f(x + h) - f(x - h)) / (2 * h)  # 2차 정확도
    d1_4th = (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12 * h)  # 4차 정확도

    print(f"\n수치 검증 (h = {h}):")
    print(f"  정확한 f'(π/4) = {f_1(x):.10f}")
    print(f"  2차 정확도: {d1_2nd:.10f}, 오차: {abs(d1_2nd - f_1(x)):.2e}")
    print(f"  4차 정확도: {d1_4th:.10f}, 오차: {abs(d1_4th - f_1(x)):.2e}")

    # 2차 미분
    d2_2nd = (f(x + h) - 2*f(x) + f(x - h)) / h**2
    d2_4th = (-f(x + 2*h) + 16*f(x + h) - 30*f(x) + 16*f(x - h) - f(x - 2*h)) / (12 * h**2)

    print(f"\n  정확한 f''(π/4) = {f_2(x):.10f}")
    print(f"  2차 정확도: {d2_2nd:.10f}, 오차: {abs(d2_2nd - f_2(x)):.2e}")
    print(f"  4차 정확도: {d2_4th:.10f}, 오차: {abs(d2_4th - f_2(x)):.2e}")

high_order_formulas()
```

### 7.2 차분 계수 생성기

```python
from scipy.special import factorial

def compute_fd_coefficients(derivative_order, accuracy_order, positions=None):
    """
    유한차분 계수 계산

    Parameters:
    -----------
    derivative_order : int - 미분 차수 (1=1차 미분, 2=2차 미분, ...)
    accuracy_order : int - 정확도 차수
    positions : list - 스텐실 위치 (기본: 중심차분)

    Returns:
    --------
    coeffs : array - 차분 계수
    positions : array - 스텐실 위치
    """
    import numpy as np

    if positions is None:
        # 중심차분 스텐실
        n_points = derivative_order + accuracy_order
        if n_points % 2 == 0:
            n_points += 1
        half = n_points // 2
        positions = np.arange(-half, half + 1)

    n = len(positions)
    positions = np.array(positions, dtype=float)

    # Vandermonde 행렬 구성
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = positions[j] ** i

    # 우변 벡터
    b = np.zeros(n)
    b[derivative_order] = factorial(derivative_order)

    # 계수 계산
    coeffs = np.linalg.solve(A, b)

    return coeffs, positions

# 예시: 1차 미분, 2차 정확도
coeffs, pos = compute_fd_coefficients(1, 2)
print("1차 미분 (2차 정확도):")
print(f"  위치: {pos}")
print(f"  계수: {coeffs}")
print(f"  공식: ({coeffs[0]:.1f}·f[i-1] + {coeffs[1]:.1f}·f[i] + {coeffs[2]:.1f}·f[i+1]) / h")

# 2차 미분, 2차 정확도
coeffs, pos = compute_fd_coefficients(2, 2)
print("\n2차 미분 (2차 정확도):")
print(f"  위치: {pos}")
print(f"  계수: {coeffs}")
```

---

## 8. 희소 행렬을 이용한 효율적 구현

### 8.1 scipy.sparse 사용

```python
from scipy import sparse
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve

def create_laplacian_1d(nx, dx, bc_type='dirichlet'):
    """
    1D 라플라시안 행렬 생성 (희소 행렬)

    Parameters:
    -----------
    nx : int - 격자점 수
    dx : float - 격자 간격
    bc_type : str - 경계조건 유형

    Returns:
    --------
    L : sparse matrix - 라플라시안 행렬
    """
    n = nx - 2  # 내부점 수 (디리클레 경계조건)

    # 대각선 요소
    main_diag = -2.0 * np.ones(n)
    off_diag = np.ones(n - 1)

    # 희소 행렬 생성
    L = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')
    L = L / dx**2

    return L

# 예시
nx = 101
dx = 1.0 / (nx - 1)
L = create_laplacian_1d(nx, dx)

print(f"라플라시안 행렬 크기: {L.shape}")
print(f"비영 요소 수: {L.nnz}")
print(f"밀도: {L.nnz / (L.shape[0] * L.shape[1]) * 100:.2f}%")
print(f"\n행렬 일부:")
print(L.toarray()[:5, :5])
```

### 8.2 2D 라플라시안 행렬

```python
def create_laplacian_2d(nx, ny, dx, dy):
    """
    2D 라플라시안 행렬 생성 (5점 스텐실)

    d²u/dx² + d²u/dy² ≈ (u_{i+1,j} - 2u_{i,j} + u_{i-1,j})/dx²
                       + (u_{i,j+1} - 2u_{i,j} + u_{i,j-1})/dy²

    행렬 인덱싱: 내부점을 행 우선으로 1D로 펼침
    k = (j-1)*(nx-2) + (i-1)  (i, j는 1-based 내부점 인덱스)
    """
    mx = nx - 2  # x 방향 내부점 수
    my = ny - 2  # y 방향 내부점 수
    n = mx * my  # 총 내부점 수

    # 계수
    cx = 1.0 / dx**2
    cy = 1.0 / dy**2
    cc = -2.0 * (cx + cy)

    # 대각선 구성
    main_diag = cc * np.ones(n)
    x_diag = cx * np.ones(n - 1)
    y_diag = cy * np.ones(n - mx)

    # x 방향 이웃 연결 (행 경계에서 끊김)
    for j in range(my):
        if j < my - 1:
            x_diag[j * mx + mx - 1] = 0

    # 희소 행렬 생성
    diagonals = [y_diag, x_diag, main_diag, x_diag, y_diag]
    offsets = [-mx, -1, 0, 1, mx]

    L = diags(diagonals, offsets, shape=(n, n), format='csr')

    return L

# 예시
nx, ny = 11, 11
dx = dy = 1.0 / (nx - 1)
L = create_laplacian_2d(nx, ny, dx, dy)

print(f"2D 라플라시안 행렬 크기: {L.shape}")
print(f"비영 요소 수: {L.nnz}")
print(f"밀도: {L.nnz / (L.shape[0] * L.shape[1]) * 100:.2f}%")
```

### 8.3 희소 행렬 시각화

```python
def visualize_sparse_matrices():
    """희소 행렬 구조 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1D 라플라시안
    L1d = create_laplacian_1d(21, 0.05)
    ax1 = axes[0]
    ax1.spy(L1d, markersize=3)
    ax1.set_title(f'1D 라플라시안 (n={L1d.shape[0]})\n삼중 대각 구조', fontsize=12)

    # 2D 라플라시안 (작은 크기)
    L2d = create_laplacian_2d(7, 7, 0.1, 0.1)
    ax2 = axes[1]
    ax2.spy(L2d, markersize=5)
    ax2.set_title(f'2D 라플라시안 ({L2d.shape[0]}x{L2d.shape[0]})\n5중 대각 구조', fontsize=12)

    # 희소성 비교
    ax3 = axes[2]
    sizes = [11, 21, 41, 81, 161]
    densities_1d = []
    densities_2d = []

    for s in sizes:
        L1 = create_laplacian_1d(s, 1.0/(s-1))
        densities_1d.append(L1.nnz / (L1.shape[0]**2) * 100)

        L2 = create_laplacian_2d(s, s, 1.0/(s-1), 1.0/(s-1))
        densities_2d.append(L2.nnz / (L2.shape[0]**2) * 100)

    ax3.semilogy(sizes, densities_1d, 'o-', label='1D 라플라시안', linewidth=2)
    ax3.semilogy(sizes, densities_2d, 's-', label='2D 라플라시안', linewidth=2)
    ax3.set_xlabel('격자 크기 n', fontsize=12)
    ax3.set_ylabel('행렬 밀도 (%)', fontsize=12)
    ax3.set_title('희소 행렬 밀도\n(격자 크기에 따른 변화)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sparse_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()

# visualize_sparse_matrices()
```

---

## 9. 종합 예제: 포아송 방정식 풀이

```python
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
import matplotlib.pyplot as plt

def solve_poisson_1d():
    """
    1D 포아송 방정식 풀이

    -d²u/dx² = f(x), 0 < x < 1
    경계조건: u(0) = 0, u(1) = 0
    소스항: f(x) = π²·sin(πx)
    해석해: u(x) = sin(πx)
    """
    # 파라미터
    nx = 101
    L = 1.0
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)

    # 소스항 (내부점만)
    x_inner = x[1:-1]
    f = np.pi**2 * np.sin(np.pi * x_inner)

    # 라플라시안 행렬 (-d²/dx²)
    n = nx - 2
    main_diag = 2.0 * np.ones(n)
    off_diag = -1.0 * np.ones(n - 1)
    A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')
    A = A / dx**2

    # 선형 시스템 풀이
    u_inner = spsolve(A, f)

    # 전체 해 (경계조건 포함)
    u = np.zeros(nx)
    u[1:-1] = u_inner
    u[0] = 0  # 디리클레 BC
    u[-1] = 0

    # 해석해
    u_exact = np.sin(np.pi * x)

    # 오차
    error = np.max(np.abs(u - u_exact))
    print(f"1D 포아송 방정식 풀이")
    print(f"  격자점: {nx}")
    print(f"  최대 오차: {error:.2e}")

    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(x, u_exact, 'b-', label='해석해', linewidth=2)
    ax1.plot(x, u, 'ro', label='수치해', markersize=4, markevery=5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x)')
    ax1.set_title('1D 포아송 방정식 해')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, np.abs(u - u_exact), 'g-', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('|오차|')
    ax2.set_title(f'수치해 오차 (최대: {error:.2e})')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('poisson_1d.png', dpi=150, bbox_inches='tight')
    plt.show()

    return x, u, u_exact

# x, u, u_exact = solve_poisson_1d()
```

---

## 10. 요약

### 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| 유한차분법 | 미분을 유한한 차분으로 근사 |
| 전방차분 | (u_{i+1} - u_i)/Δx, O(Δx) |
| 후방차분 | (u_i - u_{i-1})/Δx, O(Δx) |
| 중심차분 | (u_{i+1} - u_{i-1})/(2Δx), O(Δx²) |
| 절단오차 | 테일러 전개에서 잘린 고차 항 |
| CFL 조건 | 수치 안정성을 위한 Δt/Δx 제한 |
| von Neumann 분석 | 푸리에 모드 증폭인자 분석 |
| 희소 행렬 | 대규모 시스템의 효율적 저장/계산 |

### 주요 공식

```
1차 미분 (중심): f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
2차 미분 (중심): f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²

CFL 조건:
- 이류방정식: c·Δt/Δx ≤ 1
- 열방정식 (FTCS): α·Δt/Δx² ≤ 0.5
- 파동방정식: c·Δt/Δx ≤ 1
```

### 다음 단계

1. **09장**: 열방정식 - FTCS, BTCS, Crank-Nicolson
2. **10장**: 파동방정식 - CTCS, 경계조건 처리
3. **11장**: 라플라스/포아송 - 반복법
4. **12장**: 이류방정식 - 풍상법, 수치 확산

---

## 연습문제

### 연습 1: 차분 정확도 확인
f(x) = e^x에 대해 x = 1에서 전방/후방/중심 차분을 계산하고, h = 0.1, 0.01, 0.001에서 오차를 비교하시오.

### 연습 2: 2차 미분 수치해
f(x) = x⁴에 대해 중심차분으로 f''(x)를 계산하고 정확값 12x²와 비교하시오.

### 연습 3: CFL 조건 계산
열확산계수 α = 0.05, 격자 간격 Δx = 0.01일 때, FTCS 방법이 안정하기 위한 최대 시간 간격 Δt를 구하시오.

### 연습 4: 희소 행렬 포아송 방정식
위의 1D 포아송 예제를 수정하여 f(x) = 1 (상수)인 경우의 해를 구하시오. (해석해: u(x) = x(1-x)/2)

---

## 참고 자료

1. **교재**:
   - LeVeque, "Finite Difference Methods for Ordinary and Partial Differential Equations"
   - Strikwerda, "Finite Difference Schemes and Partial Differential Equations"

2. **Python 라이브러리**:
   - scipy.sparse: 희소 행렬 연산
   - numpy: 배열 연산

3. **온라인**:
   - MIT OCW 18.336: Numerical Methods for PDEs
