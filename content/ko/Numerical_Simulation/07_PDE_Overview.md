# 07. 편미분방정식 개요 (Partial Differential Equations Overview)

## 학습 목표
- 편미분방정식(PDE)의 기본 개념과 분류 이해
- 포물선형, 쌍곡선형, 타원형 PDE의 특성 파악
- 경계조건과 초기조건의 역할 이해
- 적정조건 문제(Well-posed problem)의 개념 학습

---

**이 레슨이 중요한 이유:** ODE가 시간에 따른 변화만을 기술하는 반면, 대부분의 물리 현상 -- 열 전도, 파동 전파, 유체 흐름, 전자기장 -- 은 공간과 시간 모두에서의 변화를 포함합니다. PDE는 이러한 다차원 현상을 위한 수학적 언어입니다. PDE 분류를 이해하는 것은 각 유형(타원형, 포물선형, 쌍곡선형)이 근본적으로 다른 수치 전략과 경계조건 처리를 요구하기 때문에 필수적입니다.

## 1. 편미분방정식이란?

### 1.1 정의

편미분방정식(Partial Differential Equation, PDE)은 여러 독립변수에 대한 편미분을 포함하는 방정식입니다.

```
일반적인 2차 PDE 형태:
A·∂²u/∂x² + B·∂²u/∂x∂y + C·∂²u/∂y² + D·∂u/∂x + E·∂u/∂y + F·u = G
```

여기서 A, B, C, D, E, F, G는 x, y의 함수일 수 있습니다.

### 1.2 ODE vs PDE 비교

| 특성 | ODE | PDE |
|------|-----|-----|
| 독립변수 | 1개 (보통 t) | 2개 이상 (보통 x, y, z, t) |
| 미분 종류 | 상미분 | 편미분 |
| 해의 형태 | 함수 y(t) | 함수 u(x, y, ...) |
| 경계조건 | 초기조건 | 경계조건 + 초기조건 |
| 해법 난이도 | 상대적 쉬움 | 복잡함 |

### 1.3 물리적 응용 예시

```python
"""
주요 물리 현상과 대응하는 PDE
"""

# 열전도 (Heat Conduction)
# ∂u/∂t = α · ∂²u/∂x²
# 시간에 따른 온도 분포 변화

# 파동 전파 (Wave Propagation)
# ∂²u/∂t² = c² · ∂²u/∂x²
# 소리, 빛, 진동의 전파

# 정상 상태 열분포 (Steady-State)
# ∂²u/∂x² + ∂²u/∂y² = 0
# 시간 변화 없는 온도 분포

# 유체 이류 (Advection)
# ∂u/∂t + v · ∂u/∂x = 0
# 물질의 이동

# 확산 (Diffusion)
# ∂u/∂t = D · ∇²u
# 농도 확산 현상
```

---

## 2. PDE 분류

### 2.1 2차 선형 PDE의 분류

2차 선형 PDE의 일반형:
```
A·∂²u/∂x² + B·∂²u/∂x∂y + C·∂²u/∂y² + (하위 항들) = 0
```

**판별식 Δ = B² - 4AC**에 따라 분류:

| 분류 | 조건 | 대표 방정식 | 물리 현상 |
|------|------|-------------|-----------|
| **타원형 (Elliptic)** | Δ < 0 | 라플라스, 포아송 | 정상상태 |
| **포물선형 (Parabolic)** | Δ = 0 | 열방정식 | 확산 |
| **쌍곡선형 (Hyperbolic)** | Δ > 0 | 파동방정식 | 파동 전파 |

```python
import numpy as np

def classify_pde(A, B, C):
    """
    2차 선형 PDE 분류

    Parameters:
    -----------
    A : float - ∂²u/∂x² 계수
    B : float - ∂²u/∂x∂y 계수
    C : float - ∂²u/∂y² 계수

    Returns:
    --------
    str : PDE 분류
    """
    delta = B**2 - 4*A*C

    if delta < 0:
        return "타원형 (Elliptic)"
    elif delta == 0:
        return "포물선형 (Parabolic)"
    else:
        return "쌍곡선형 (Hyperbolic)"

# 예시
print("라플라스 방정식 (A=1, B=0, C=1):", classify_pde(1, 0, 1))
print("열방정식 (A=1, B=0, C=0):", classify_pde(1, 0, 0))
print("파동방정식 (A=1, B=0, C=-1):", classify_pde(1, 0, -1))
```

### 2.2 표준형 (Canonical Forms)

#### 타원형 표준형 (Laplace 방정식)
```
∂²u/∂x² + ∂²u/∂y² = 0
```

#### 포물선형 표준형 (열방정식)
```
∂u/∂t = α · ∂²u/∂x²
```

#### 쌍곡선형 표준형 (파동방정식)
```
∂²u/∂t² = c² · ∂²u/∂x²
```

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_pde_types():
    """PDE 유형별 특성 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x, t)

    # 타원형: 정상상태 - 시간 독립
    # u = x(1-x) (경계조건 만족하는 해)
    U_elliptic = X * (1 - X)
    ax1 = axes[0]
    c1 = ax1.contourf(X, T, U_elliptic, levels=20, cmap='coolwarm')
    ax1.set_title('타원형 (Elliptic)\n정상상태 - 시간 독립', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y (또는 t)')
    plt.colorbar(c1, ax=ax1)

    # 포물선형: 확산 - 시간에 따라 평활화
    # 초기 삼각파가 시간에 따라 평탄해짐
    U_parabolic = np.exp(-np.pi**2 * T * 0.1) * np.sin(np.pi * X)
    ax2 = axes[1]
    c2 = ax2.contourf(X, T, U_parabolic, levels=20, cmap='coolwarm')
    ax2.set_title('포물선형 (Parabolic)\n확산 - 시간에 따라 평활화', fontsize=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    plt.colorbar(c2, ax=ax2)

    # 쌍곡선형: 파동 - 진동 전파
    c_speed = 1.0
    U_hyperbolic = np.sin(2*np.pi*X) * np.cos(2*np.pi*c_speed*T)
    ax3 = axes[2]
    c3 = ax3.contourf(X, T, U_hyperbolic, levels=20, cmap='coolwarm')
    ax3.set_title('쌍곡선형 (Hyperbolic)\n파동 - 진동 전파', fontsize=12)
    ax3.set_xlabel('x')
    ax3.set_ylabel('t')
    plt.colorbar(c3, ax=ax3)

    plt.tight_layout()
    plt.savefig('pde_types.png', dpi=150, bbox_inches='tight')
    plt.show()

# visualize_pde_types()
```

---

## 3. 경계조건 (Boundary Conditions)

### 3.1 경계조건의 종류

PDE를 풀기 위해서는 적절한 경계조건이 필요합니다.

```python
"""
세 가지 주요 경계조건 유형
"""
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_boundary_conditions():
    """경계조건 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    x = np.linspace(0, 1, 100)

    # 1. 디리클레 경계조건 (Dirichlet BC)
    # u(0) = a, u(L) = b - 경계에서 함수값 지정
    ax1 = axes[0]
    u_dirichlet = 0 + (1 - 0) * x  # 선형 보간 예시
    ax1.plot(x, u_dirichlet, 'b-', linewidth=2)
    ax1.scatter([0, 1], [0, 1], color='red', s=100, zorder=5, label='경계값')
    ax1.set_title('디리클레 경계조건 (Dirichlet)\nu(0) = 0, u(L) = 1', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x)')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 노이만 경계조건 (Neumann BC)
    # du/dx|₀ = a, du/dx|_L = b - 경계에서 미분값(기울기) 지정
    ax2 = axes[1]
    # du/dx = 0 at both ends (단열 조건)
    u_neumann = np.cos(np.pi * x)  # 양끝에서 기울기 0
    ax2.plot(x, u_neumann, 'b-', linewidth=2)
    ax2.annotate('', xy=(0.05, u_neumann[5]), xytext=(0, u_neumann[0]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax2.annotate('', xy=(0.95, u_neumann[-6]), xytext=(1, u_neumann[-1]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax2.set_title('노이만 경계조건 (Neumann)\n∂u/∂x|₀ = 0, ∂u/∂x|_L = 0', fontsize=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('u(x)')
    ax2.grid(True, alpha=0.3)

    # 3. 로빈 경계조건 (Robin/Mixed BC)
    # a·u + b·du/dx = c - 함수값과 미분값의 선형 조합
    ax3 = axes[2]
    u_robin = np.exp(-x) * np.cos(2*np.pi*x)
    ax3.plot(x, u_robin, 'b-', linewidth=2)
    ax3.scatter([0], [u_robin[0]], color='green', s=100, zorder=5)
    ax3.set_title('로빈 경계조건 (Robin)\nα·u + β·∂u/∂n = γ', fontsize=12)
    ax3.set_xlabel('x')
    ax3.set_ylabel('u(x)')
    ax3.grid(True, alpha=0.3)
    ax3.annotate('혼합 조건', xy=(0, u_robin[0]), xytext=(0.2, 0.8),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')

    plt.tight_layout()
    plt.savefig('boundary_conditions.png', dpi=150, bbox_inches='tight')
    plt.show()

# demonstrate_boundary_conditions()
```

### 3.2 경계조건 상세

#### 디리클레 경계조건 (Dirichlet BC)
- **정의**: 경계에서 함수값 자체를 지정
- **수식**: u(경계) = g
- **물리적 의미**: 온도 고정, 변위 고정

```python
def apply_dirichlet_bc(u, left_value, right_value):
    """디리클레 경계조건 적용"""
    u[0] = left_value    # 왼쪽 경계
    u[-1] = right_value  # 오른쪽 경계
    return u

# 예시: 막대 양끝 온도 고정
u = np.zeros(100)
u = apply_dirichlet_bc(u, left_value=100.0, right_value=0.0)
print(f"왼쪽 경계: {u[0]}°C, 오른쪽 경계: {u[-1]}°C")
```

#### 노이만 경계조건 (Neumann BC)
- **정의**: 경계에서 법선방향 미분값을 지정
- **수식**: ∂u/∂n|경계 = h
- **물리적 의미**: 열유속 지정, 단열 조건(h=0)

```python
def apply_neumann_bc(u, dx, left_flux, right_flux):
    """
    노이만 경계조건 적용 (1차 정확도)

    Parameters:
    -----------
    u : array - 해 배열
    dx : float - 격자 간격
    left_flux : float - 왼쪽 경계 유속 (∂u/∂x)
    right_flux : float - 오른쪽 경계 유속 (∂u/∂x)
    """
    # 왼쪽: ∂u/∂x|₀ = left_flux
    # (u[1] - u[0])/dx = left_flux
    u[0] = u[1] - dx * left_flux

    # 오른쪽: ∂u/∂x|_L = right_flux
    # (u[-1] - u[-2])/dx = right_flux
    u[-1] = u[-2] + dx * right_flux

    return u

# 예시: 단열 경계조건 (열유속 = 0)
u = np.linspace(100, 0, 100)
dx = 0.01
u = apply_neumann_bc(u, dx, left_flux=0.0, right_flux=0.0)
print(f"단열 경계 적용 완료")
```

#### 로빈 경계조건 (Robin/Mixed BC)
- **정의**: 함수값과 미분값의 선형 조합
- **수식**: α·u + β·∂u/∂n = γ
- **물리적 의미**: 대류 열전달

```python
def apply_robin_bc(u, dx, alpha, beta, gamma):
    """
    로빈 경계조건 적용 (왼쪽 경계)
    α·u + β·∂u/∂x = γ
    """
    # α·u[0] + β·(u[1] - u[0])/dx = γ
    # (α - β/dx)·u[0] + (β/dx)·u[1] = γ
    # u[0] = (γ - (β/dx)·u[1]) / (α - β/dx)

    if abs(alpha - beta/dx) > 1e-10:
        u[0] = (gamma - (beta/dx) * u[1]) / (alpha - beta/dx)

    return u

# 예시: 대류 열전달
# h·(u - T_inf) + k·∂u/∂x = 0
# 여기서 h는 열전달계수, k는 열전도도, T_inf는 주변 온도
```

---

## 4. 초기조건 (Initial Conditions)

### 4.1 초기조건의 역할

시간에 의존하는 PDE(포물선형, 쌍곡선형)에서는 초기 상태를 지정해야 합니다.

```python
def demonstrate_initial_conditions():
    """다양한 초기조건 예시"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    x = np.linspace(0, 1, 200)
    L = 1.0

    # 1. 사인 함수 초기조건
    u1 = np.sin(np.pi * x / L)
    axes[0, 0].plot(x, u1, 'b-', linewidth=2)
    axes[0, 0].set_title('사인 초기조건\nu(x,0) = sin(πx/L)', fontsize=12)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('u')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-0.2, 1.2)

    # 2. 가우시안 펄스 초기조건
    x0 = 0.5  # 중심
    sigma = 0.1  # 폭
    u2 = np.exp(-(x - x0)**2 / (2 * sigma**2))
    axes[0, 1].plot(x, u2, 'b-', linewidth=2)
    axes[0, 1].set_title('가우시안 펄스\nu(x,0) = exp(-(x-x₀)²/2σ²)', fontsize=12)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('u')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(-0.2, 1.2)

    # 3. 계단 함수 (불연속)
    u3 = np.where(x < 0.5, 1.0, 0.0)
    axes[1, 0].plot(x, u3, 'b-', linewidth=2)
    axes[1, 0].set_title('계단 함수\nu(x,0) = H(0.5-x)', fontsize=12)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('u')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(-0.2, 1.2)

    # 4. 삼각파
    u4 = np.where(x < 0.5, 2*x, 2*(1-x))
    axes[1, 1].plot(x, u4, 'b-', linewidth=2)
    axes[1, 1].set_title('삼각파\nu(x,0) = 2min(x, 1-x)', fontsize=12)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('u')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(-0.2, 1.2)

    plt.tight_layout()
    plt.savefig('initial_conditions.png', dpi=150, bbox_inches='tight')
    plt.show()

# demonstrate_initial_conditions()
```

### 4.2 포물선형 vs 쌍곡선형 초기조건

| PDE 유형 | 필요한 초기조건 | 예시 |
|----------|-----------------|------|
| 포물선형 (열방정식) | u(x, 0) | 초기 온도 분포 |
| 쌍곡선형 (파동방정식) | u(x, 0) 및 ∂u/∂t(x, 0) | 초기 변위 + 초기 속도 |

```python
class PDEProblem:
    """PDE 문제 정의 클래스"""

    def __init__(self, pde_type, domain, nx, nt=None):
        """
        Parameters:
        -----------
        pde_type : str - 'parabolic', 'hyperbolic', 'elliptic'
        domain : tuple - (x_min, x_max) 또는 ((x_min, x_max), (y_min, y_max))
        nx : int - x 방향 격자점 수
        nt : int - 시간 스텝 수 (시간 의존 문제의 경우)
        """
        self.pde_type = pde_type
        self.domain = domain
        self.nx = nx
        self.nt = nt

        # 격자 생성
        self.x = np.linspace(domain[0], domain[1], nx)
        self.dx = self.x[1] - self.x[0]

        # 초기조건 및 경계조건 저장
        self.initial_condition = None
        self.initial_velocity = None  # 쌍곡선형용
        self.bc_left = {'type': 'dirichlet', 'value': 0}
        self.bc_right = {'type': 'dirichlet', 'value': 0}

    def set_initial_condition(self, func):
        """초기조건 설정"""
        self.initial_condition = func(self.x)
        return self

    def set_initial_velocity(self, func):
        """초기 속도 설정 (파동방정식용)"""
        if self.pde_type != 'hyperbolic':
            print("경고: 초기 속도는 쌍곡선형 PDE에만 필요합니다.")
        self.initial_velocity = func(self.x)
        return self

    def set_boundary_condition(self, side, bc_type, value=0, flux=0, alpha=1, beta=0, gamma=0):
        """
        경계조건 설정

        Parameters:
        -----------
        side : str - 'left' 또는 'right'
        bc_type : str - 'dirichlet', 'neumann', 'robin'
        """
        bc = {'type': bc_type}

        if bc_type == 'dirichlet':
            bc['value'] = value
        elif bc_type == 'neumann':
            bc['flux'] = flux
        elif bc_type == 'robin':
            bc['alpha'] = alpha
            bc['beta'] = beta
            bc['gamma'] = gamma

        if side == 'left':
            self.bc_left = bc
        else:
            self.bc_right = bc

        return self

    def summary(self):
        """문제 요약 출력"""
        print(f"\n{'='*50}")
        print(f"PDE 문제 요약")
        print(f"{'='*50}")
        print(f"유형: {self.pde_type}")
        print(f"정의역: [{self.domain[0]}, {self.domain[1]}]")
        print(f"격자점: {self.nx}")
        print(f"격자 간격 (dx): {self.dx:.6f}")
        print(f"\n왼쪽 경계조건: {self.bc_left}")
        print(f"오른쪽 경계조건: {self.bc_right}")

        if self.initial_condition is not None:
            print(f"\n초기조건: 설정됨")
        if self.initial_velocity is not None:
            print(f"초기 속도: 설정됨")
        print(f"{'='*50}\n")

# 사용 예시
problem = PDEProblem('parabolic', (0, 1), 101)
problem.set_initial_condition(lambda x: np.sin(np.pi * x))
problem.set_boundary_condition('left', 'dirichlet', value=0)
problem.set_boundary_condition('right', 'dirichlet', value=0)
problem.summary()
```

---

## 5. 적정조건 문제 (Well-Posed Problems)

### 5.1 Hadamard의 적정조건

PDE 문제가 "잘 정의되었다(well-posed)"라고 하려면 세 가지 조건을 만족해야 합니다:

1. **존재성 (Existence)**: 해가 존재해야 함
2. **유일성 (Uniqueness)**: 해가 유일해야 함
3. **연속의존성 (Stability)**: 초기/경계조건의 작은 변화에 해가 연속적으로 의존해야 함

```python
def demonstrate_well_posedness():
    """적정조건 시연"""

    print("="*60)
    print("적정조건 문제 (Well-Posed Problem) 예시")
    print("="*60)

    # 열방정식: 적정조건 만족
    print("\n[1] 열방정식 (Well-Posed)")
    print("    ∂u/∂t = α·∂²u/∂x², 0 < x < L, t > 0")
    print("    경계: u(0,t) = u(L,t) = 0")
    print("    초기: u(x,0) = f(x)")
    print("    → 존재성: O, 유일성: O, 안정성: O")

    # 역방향 열방정식: 부적정조건 (Ill-Posed)
    print("\n[2] 역방향 열방정식 (Ill-Posed)")
    print("    ∂u/∂t = -α·∂²u/∂x² (시간 역방향)")
    print("    → 초기조건의 작은 오차가 기하급수적으로 증폭")
    print("    → 안정성 조건 위반!")

    # 라플라스 방정식 + 적절한 경계조건: Well-Posed
    print("\n[3] 라플라스 방정식 (Well-Posed with Dirichlet BC)")
    print("    ∂²u/∂x² + ∂²u/∂y² = 0 in Ω")
    print("    경계: u = g on ∂Ω")
    print("    → 존재성: O, 유일성: O, 안정성: O")

    # Cauchy 문제 for Laplace: Ill-Posed
    print("\n[4] 라플라스 Cauchy 문제 (Ill-Posed)")
    print("    ∂²u/∂x² + ∂²u/∂y² = 0")
    print("    u(x,0) = 0, ∂u/∂y(x,0) = (1/n)sin(nx)")
    print("    → 해: u = (1/n²)sin(nx)sinh(ny)")
    print("    → n→∞일 때 초기조건→0이지만 해는 폭발!")

demonstrate_well_posedness()
```

### 5.2 각 PDE 유형별 적정조건 경계조건

```python
def required_conditions_table():
    """각 PDE 유형별 필요한 조건"""

    conditions = """
    ┌─────────────────┬──────────────────────┬──────────────────────┐
    │     PDE 유형    │      경계조건         │      초기조건         │
    ├─────────────────┼──────────────────────┼──────────────────────┤
    │   타원형        │ 전체 경계에서         │ 불필요               │
    │   (Elliptic)    │ Dirichlet/Neumann/   │                      │
    │                 │ Robin 조건            │                      │
    ├─────────────────┼──────────────────────┼──────────────────────┤
    │   포물선형      │ 공간 경계에서         │ t=0에서              │
    │   (Parabolic)   │ Dirichlet/Neumann    │ u(x,0) = f(x)        │
    ├─────────────────┼──────────────────────┼──────────────────────┤
    │   쌍곡선형      │ 공간 경계에서         │ t=0에서              │
    │   (Hyperbolic)  │ Dirichlet/Neumann    │ u(x,0) = f(x)        │
    │                 │                      │ ∂u/∂t(x,0) = g(x)    │
    └─────────────────┴──────────────────────┴──────────────────────┘
    """
    print(conditions)

required_conditions_table()
```

### 5.3 안정성의 수치적 중요성

```python
import numpy as np
import matplotlib.pyplot as plt

def stability_demonstration():
    """수치 안정성의 중요성 시연"""

    # 파라미터
    L = 1.0
    nx = 50
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)

    alpha = 0.01  # 열확산계수

    # CFL 안정성 조건: dt <= dx² / (2*alpha)
    dt_stable = 0.4 * dx**2 / (2 * alpha)  # 안정
    dt_unstable = 1.5 * dx**2 / (2 * alpha)  # 불안정

    print(f"격자 간격 dx = {dx:.4f}")
    print(f"열확산계수 α = {alpha}")
    print(f"안정성 조건: dt ≤ {dx**2 / (2*alpha):.6f}")
    print(f"안정한 dt = {dt_stable:.6f}")
    print(f"불안정한 dt = {dt_unstable:.6f}")

    # 초기조건
    u0 = np.sin(np.pi * x)

    # FTCS (Forward Time Central Space) 스킴
    def ftcs_step(u, alpha, dt, dx):
        u_new = u.copy()
        r = alpha * dt / dx**2
        for i in range(1, len(u)-1):
            u_new[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1])
        # 경계조건 (Dirichlet)
        u_new[0] = 0
        u_new[-1] = 0
        return u_new

    # 시뮬레이션
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 안정한 경우
    u_stable = u0.copy()
    ax1 = axes[0]
    ax1.plot(x, u_stable, 'b-', label='t=0', alpha=0.8)

    for step in range(100):
        u_stable = ftcs_step(u_stable, alpha, dt_stable, dx)
        if step in [10, 30, 60, 99]:
            ax1.plot(x, u_stable, label=f't={step*dt_stable:.4f}', alpha=0.8)

    ax1.set_title(f'안정한 경우\ndt = {dt_stable:.6f} (CFL 조건 만족)', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('u')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.5, 1.5)

    # 불안정한 경우
    u_unstable = u0.copy()
    ax2 = axes[1]
    ax2.plot(x, u_unstable, 'b-', label='t=0', alpha=0.8)

    for step in range(10):  # 몇 스텝만 해도 폭발
        u_unstable = ftcs_step(u_unstable, alpha, dt_unstable, dx)
        if step in [1, 3, 5, 9]:
            u_clipped = np.clip(u_unstable, -10, 10)  # 시각화를 위해 클리핑
            ax2.plot(x, u_clipped, label=f't={step*dt_unstable:.4f}', alpha=0.8)

    ax2.set_title(f'불안정한 경우\ndt = {dt_unstable:.6f} (CFL 조건 위반)', fontsize=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('u')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-10, 10)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('stability_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n불안정 시뮬레이션 후 최대값: {np.max(np.abs(u_unstable)):.2e}")
    print("→ CFL 조건을 위반하면 해가 폭발합니다!")

# stability_demonstration()
```

---

## 6. 종합 예제: 1D 열방정식 문제 정의

```python
import numpy as np
import matplotlib.pyplot as plt

class HeatEquation1D:
    """
    1D 열방정식 문제 정의 및 해석해 비교

    ∂u/∂t = α · ∂²u/∂x²

    경계조건: u(0,t) = u(L,t) = 0 (Dirichlet)
    초기조건: u(x,0) = sin(πx/L)

    해석해: u(x,t) = sin(πx/L) · exp(-α(π/L)²t)
    """

    def __init__(self, L=1.0, alpha=0.01, nx=51, T=1.0, nt=1000):
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
        self.t = np.linspace(0, T, nt + 1)

        # CFL 조건 확인
        self.r = alpha * self.dt / self.dx**2
        self.cfl_satisfied = self.r <= 0.5

        print(f"열방정식 1D 문제 설정")
        print(f"  영역: [0, {L}]")
        print(f"  열확산계수 α = {alpha}")
        print(f"  공간 격자: nx = {nx}, dx = {self.dx:.4f}")
        print(f"  시간 격자: nt = {nt}, dt = {self.dt:.6f}")
        print(f"  CFL 수: r = α·dt/dx² = {self.r:.4f}")
        print(f"  CFL 조건 만족: {self.cfl_satisfied}")

    def initial_condition(self, x):
        """초기조건: u(x,0) = sin(πx/L)"""
        return np.sin(np.pi * x / self.L)

    def exact_solution(self, x, t):
        """해석해: u(x,t) = sin(πx/L) · exp(-α(π/L)²t)"""
        return np.sin(np.pi * x / self.L) * np.exp(-self.alpha * (np.pi / self.L)**2 * t)

    def boundary_conditions(self):
        """경계조건 반환"""
        return {
            'left': {'type': 'dirichlet', 'value': 0.0},
            'right': {'type': 'dirichlet', 'value': 0.0}
        }

    def plot_exact_solution(self):
        """해석해 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 여러 시간에서의 해
        times = [0, 0.1, 0.3, 0.5, 1.0]
        for t in times:
            u = self.exact_solution(self.x, t)
            ax1.plot(self.x, u, label=f't = {t}')

        ax1.set_title('1D 열방정식 해석해 (시간 변화)', fontsize=12)
        ax1.set_xlabel('x')
        ax1.set_ylabel('u(x,t)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 시공간 등고선도
        X, T = np.meshgrid(self.x, self.t)
        U = self.exact_solution(X, T)

        c = ax2.contourf(X, T, U, levels=20, cmap='coolwarm')
        plt.colorbar(c, ax=ax2, label='u(x,t)')
        ax2.set_title('1D 열방정식 시공간 분포', fontsize=12)
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')

        plt.tight_layout()
        plt.savefig('heat_exact.png', dpi=150, bbox_inches='tight')
        plt.show()

# 문제 정의 및 해석해 확인
problem = HeatEquation1D(L=1.0, alpha=0.01, nx=51, T=1.0, nt=1000)
# problem.plot_exact_solution()
```

---

## 7. 요약

### 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| PDE | 여러 독립변수에 대한 편미분을 포함하는 방정식 |
| 타원형 | Δ < 0, 정상상태 (라플라스, 포아송) |
| 포물선형 | Δ = 0, 확산/열전도 (열방정식) |
| 쌍곡선형 | Δ > 0, 파동 전파 (파동방정식) |
| 디리클레 BC | 경계에서 함수값 지정 |
| 노이만 BC | 경계에서 미분값(유속) 지정 |
| 로빈 BC | 함수값과 미분값의 조합 |
| 적정조건 | 존재성, 유일성, 연속의존성 |

### 다음 단계

1. **08장**: 유한차분법 기초 - 공간/시간 이산화
2. **09장**: 열방정식 수치해법 - FTCS, BTCS, Crank-Nicolson
3. **10장**: 파동방정식 수치해법 - CTCS, 경계조건 처리
4. **11장**: 라플라스/포아송 방정식 - 반복법
5. **12장**: 이류방정식 - 풍상법, 수치 확산

---

## 연습문제

### 연습 1: PDE 분류
다음 PDE를 분류하시오 (타원형/포물선형/쌍곡선형):

1. ∂²u/∂x² + 2∂²u/∂y² = 0
2. ∂u/∂t = 4∂²u/∂x²
3. ∂²u/∂t² = 9∂²u/∂x²
4. ∂²u/∂x² - ∂²u/∂y² = 0

### 연습 2: 경계조건 설정
막대의 열전도 문제에서 다음 물리적 상황에 적합한 경계조건 유형을 선택하시오:

1. 왼쪽 끝이 얼음물(0°C)에 담겨 있다
2. 오른쪽 끝이 완벽하게 단열되어 있다
3. 왼쪽 끝에서 공기와 열교환이 일어난다

### 연습 3: 해석해 유도
1D 열방정식 u_t = α·u_xx에서 경계조건 u(0,t) = u(L,t) = 0이고 초기조건이 u(x,0) = sin(2πx/L)일 때 해석해를 구하시오.

### 연습 4: 적정조건 확인
다음 문제들이 적정조건을 만족하는지 판단하시오:

1. 라플라스 방정식 + 디리클레 경계조건
2. 열방정식 + 초기조건 + 디리클레 경계조건
3. 열방정식 (시간 역방향) + 최종조건

---

## 참고 자료

1. **교재**:
   - "Numerical Methods for Engineers" - Chapra & Canale
   - "Numerical Solution of Partial Differential Equations" - Morton & Mayers

2. **온라인**:
   - MIT OCW 18.303: Linear Partial Differential Equations
   - Stanford CME 306: Numerical Solution of PDEs

3. **소프트웨어**:
   - NumPy, SciPy: Python 수치 계산
   - FEniCS, FiPy: PDE 전용 라이브러리
