# 15. 2D MHD 솔버

## 학습 목표

- 차원 분할(Dimensional Splitting)과 비분할(Unsplit) 기법을 사용하여 1D MHD 방법을 2D로 확장하기
- 2D Cartesian 그리드에서 유한 체적법(Finite Volume Method) 구현하기
- Constrained Transport (CT)를 적용하여 $\nabla \cdot B = 0$을 정확히 보존하기
- 자기장 성분을 위한 엇갈린 그리드(Staggered Grid, Yee Mesh) 사용하기
- 고차 재구성 구현하기: PLM, WENO
- 벤치마크 문제 시뮬레이션하기: Orszag-Tang 와류, Kelvin-Helmholtz 불안정성
- Corner Transport Upwind (CTU) 방법 이해하기

## 1. 2D MHD 소개

MHD 시뮬레이션을 1D에서 2D로 확장하면 새로운 과제들이 도입됩니다: 다차원 파동 전파, 기하학적 소스 항, 그리고 다중 차원에서 $\nabla \cdot B = 0$을 보존해야 하는 중요한 요구사항입니다.

### 1.1 2D MHD 방정식

2D Cartesian 좌표 $(x, y)$에서 이상 MHD 방정식은:

```
∂U/∂t + ∂F/∂x + ∂G/∂y = 0
```

여기서 보존 변수는:

```
U = [ρ, ρv_x, ρv_y, ρv_z, B_x, B_y, B_z, E]ᵀ
```

$x$ 방향의 플럭스:

```
F = [ρv_x, ρv_x² + p_T - B_x²/μ₀, ρv_x v_y - B_x B_y/μ₀, ρv_x v_z - B_x B_z/μ₀,
     0, v_x B_y - v_y B_x, v_x B_z - v_z B_x,
     v_x(E + p_T) - B_x(v·B)/μ₀]ᵀ
```

그리고 유사하게 $G$ ($y$ 방향의 플럭스)가 있으며, 총 압력은:

```
p_T = p + B²/(2μ₀)
```

### 1.2 발산 제약 조건(Divergence Constraint)

자기장은 다음을 만족해야 합니다:

```
∇ · B = ∂B_x/∂x + ∂B_y/∂y = 0
```

1D에서는 이것이 $\partial B_x / \partial x = 0$으로 축소되어, $B_x$가 초기에 일정하면 자동으로 만족됩니다. 2D에서는 $\nabla \cdot B = 0$을 보존하기 위해 특수한 수치적 처리가 필요합니다.

**$\nabla \cdot B = 0$ 위반의 결과:**
- 비물리적 단극자 힘
- 수치적 불안정성
- 부정확한 파동 속도와 충격파 구조

### 1.3 2D에서의 과제

1. **계산 비용**: $N_x \times N_y$ 셀, 타임스텝당 $\mathcal{O}(N^2)$ 연산
2. **다차원 효과**: 코너 결합(Corner Coupling), 횡방향 파동
3. **발산 보존**: 특수한 이산화 필요 (CT, divergence cleaning 등)
4. **CFL 조건**: 타임스텝이 2D 파동 전파에 의해 제한됨

## 2. 2D 유한 체적법(Finite Volume Method)

### 2.1 셀 중심 이산화(Cell-Centered Discretization)

영역을 직사각형 셀 $[x_{i-1/2}, x_{i+1/2}] \times [y_{j-1/2}, y_{j+1/2}]$로 나눕니다.

셀 평균 보존 변수:

```
U_{i,j} = (1/ΔxΔy) ∫∫ U(x,y,t) dx dy
```

### 2.2 반이산(Semi-Discrete) 형식

유한 체적 이산화:

```
dU_{i,j}/dt = -(F_{i+1/2,j} - F_{i-1/2,j})/Δx - (G_{i,j+1/2} - G_{i,j-1/2})/Δy
```

셀 면에서의 플럭스는 Riemann 솔버(HLL, HLLD, Roe 등)로부터 계산됩니다.

### 2.3 차원 분할(Dimensional Splitting) (Strang Splitting)

**아이디어**: 2D 진화를 교대하는 1D 스윕으로 분할.

하나의 타임스텝 $\Delta t$에 대해:

1. **$x$에서 반 스텝**: $\Delta t / 2$ 동안 $\partial U / \partial t + \partial F / \partial x = 0$을 사용하여 진화
2. **$y$에서 완전 스텝**: $\Delta t$ 동안 $\partial U / \partial t + \partial G / \partial y = 0$을 사용하여 진화
3. **$x$에서 반 스텝**: $\Delta t / 2$ 동안 $x$에서 다시 진화

이것이 **Strang splitting**입니다 (각 1D 스텝이 2차 정확도이면 시간에 대해 2차 정확도).

**장점:**
- 1D Riemann 솔버 재사용
- 구현 간단

**단점:**
- 비등방성 오차 (방향성 편향)
- 차원 분할에 걸쳐 $\nabla \cdot B = 0$을 보존하기 어려움

### 2.4 비분할 방법(Unsplit Methods)

**Corner Transport Upwind (CTU)** 방법은 횡방향 플럭스 보정을 포함하여 모든 방향을 동시에 업데이트합니다.

**알고리즘** (단순화된 CTU):

1. $x$와 $y$ 방향 모두에서 셀 면의 상태 재구성
2. 모든 면에서 Riemann 문제 풀기
3. 횡방향 플럭스 보정 계산 (예: 상류 코너 상태)
4. 코너 결합을 포함하여 보존 변수 업데이트

CTU 방법은 완전히 다차원적이며 방향성 편향을 줄입니다.

## 3. Constrained Transport (CT)

Constrained Transport는 자기장을 위한 엇갈린 그리드를 사용하고 Faraday 법칙을 적분 형식으로 진화시킴으로써 $\nabla \cdot B = 0$을 기계 정밀도까지 보존하는 수치 방법입니다.

### 3.1 Yee Mesh (엇갈린 그리드)

**셀 중심 양** ($(x_i, y_j)$에서):
- $\rho, p, v_x, v_y, v_z, E$

**면 중심 자기장** (엇갈림):
- $B_x$는 $(x_{i-1/2}, y_j)$ (수직 $x$인 면)
- $B_y$는 $(x_i, y_{j-1/2})$ (수직 $y$인 면)
- $B_z$는 셀 중심 $(x_i, y_j)$ (만약 $B_z$가 존재하지만 2D에서 발산에 영향을 주지 않음)

**모서리 중심 전기장**:
- $E_z$는 $(x_i, y_j)$ (2D $xy$ 평면에서 셀 코너)

### 3.2 적분 형식의 Faraday 법칙

Faraday 법칙:

```
∂B/∂t = -∇ × E
```

2D에서 ($B = (B_x, B_y, B_z)$이고 $E_z$가 유일한 관련 전기장 성분):

```
∂B_x/∂t = -∂E_z/∂y
∂B_y/∂t = ∂E_z/∂x
```

셀 면에 대해 적분:

```
d/dt ∫ B_x dy = -[E_z(top) - E_z(bottom)]
d/dt ∫ B_y dx = [E_z(right) - E_z(left)]
```

이것은 초기에 유지되면 자연스럽게 $\nabla \cdot B = 0$을 보존합니다.

### 3.3 전기장 계산

이상 MHD에서 전기장:

```
E = -v × B
```

2D에서:

```
E_z = v_x B_y - v_y B_x
```

**CT 알고리즘**:

1. **셀 면에서 기본 변수 재구성**
2. **Riemann 문제 풀기**로 면 중심 속도와 자기장 얻기
3. **셀 모서리에서 전기장 계산** 인접한 면으로부터 상류(Upwinded) $v$와 $B$ 사용:
   ```
   E_z(i,j) = [v_x B_y - v_y B_x]_{i,j}
   ```
   평균화 전략: 산술 평균, 상류, Riemann 솔버 기반

4. **자기장 업데이트** 이산 Faraday 법칙 사용:
   ```
   B_x(i-1/2, j)^{n+1} = B_x(i-1/2, j)^n - Δt/Δy [E_z(i,j+1/2) - E_z(i,j-1/2)]
   B_y(i, j-1/2)^{n+1} = B_y(i, j-1/2)^n + Δt/Δx [E_z(i+1/2,j) - E_z(i-1/2,j)]
   ```

### 3.4 발산 없음(Divergence-Free) 보장

이산 업데이트가 Faraday 법칙의 적분 형식으로부터 유도되었으므로, 이산 발산:

```
(∇ · B)_{i,j} = [B_x(i+1/2,j) - B_x(i-1/2,j)]/Δx + [B_y(i,j+1/2) - B_y(i,j-1/2)]/Δy
```

은 기계 정밀도까지 보존됩니다 (초기에 0이면).

## 4. 고차 재구성(Higher-Order Reconstruction)

### 4.1 구간 선형 방법(Piecewise Linear Method, PLM)

2차 공간 정확도는 각 셀 내에서 선형으로 해를 재구성해야 합니다:

```
U(x) = U_i + σ_i (x - x_i)
```

여기서 $\sigma_i$는 기울기로, 이웃 셀로부터 추정됩니다:

```
σ_i ≈ (U_{i+1} - U_{i-1}) / (2Δx)  (중심 차분)
```

**기울기 제한(Slope Limiting)**: 불연속 근처에서 허위 진동을 방지하기 위해 제한자 적용:

```
σ_i = minmod(σ_L, σ_C, σ_R)
```

여기서:
- $\sigma_L = (U_i - U_{i-1}) / \Delta x$
- $\sigma_C = (U_{i+1} - U_{i-1}) / (2 \Delta x)$
- $\sigma_R = (U_{i+1} - U_i) / \Delta x$

**minmod 제한자**:

```
minmod(a, b, c) =
    min(|a|, |b|, |c|) * sign(a)  if sign(a) = sign(b) = sign(c)
    0                              otherwise
```

기타 제한자: MC (monotonized central), van Leer, superbee.

### 4.2 WENO (Weighted Essentially Non-Oscillatory)

WENO 기법은 여러 스텐실의 가중 조합을 사용하여 고차 정확도(5차 이상)를 달성합니다.

**WENO5 재구성** (단순화):

5점 스텐실 $\{U_{i-2}, U_{i-1}, U_i, U_{i+1}, U_{i+2}\}$를 사용하여 세 개의 3점 후보 다항식을 구성한 다음, 매끄러움 기반 가중치로 블렌딩하여 $x_{i+1/2}$에서 재구성된 값을 얻습니다.

**장점**:
- 매끄러운 영역에서 고차 정확도
- 불연속 근처에서 비진동

**단점**:
- 계산적으로 비쌈 (큰 스텐실, 비선형 가중치)
- 복잡한 구현

### 4.3 특성 변수 vs 기본 변수 재구성(Characteristic vs. Primitive Variable Reconstruction)

재구성은 다음에서 수행될 수 있습니다:
- **기본 변수** $(\rho, v_x, v_y, v_z, B_x, B_y, B_z, p)$: 더 간단하지만 비물리적 상태 생성 가능
- **보존 변수** $U$: 보존을 보장하지만 진동 생성 가능
- **특성 변수** $W = L \cdot U$: 파동 분리, 불연속 포착에 최적

MHD의 경우, 특성 재구성이 선호되지만 Jacobian의 고유벡터를 풀어야 하므로 (2D/3D에서 비쌈).

## 5. 시간 적분(Time Integration)

### 5.1 2D에서의 CFL 조건

타임스텝은 다음에 의해 제한됩니다:

```
Δt ≤ CFL * min(Δx, Δy) / max(|λ|)
```

여기서 $\lambda$는 파동 속도입니다 (고속 자기음파, Alfvén, 저속 자기음파, 엔트로피).

안전을 위해 일반적으로 $CFL \approx 0.4-0.8$.

### 5.2 시간 단계 기법(Time Stepping Schemes)

**2차 Runge-Kutta (RK2)**:

```
U* = U^n + Δt L(U^n)
U^{n+1} = 0.5 U^n + 0.5 U* + 0.5 Δt L(U*)
```

여기서 $L(U) = -(∂F/∂x + ∂G/∂y)$는 공간 연산자입니다.

**3차 Runge-Kutta (RK3)** (TVD-RK3):

```
U^{(1)} = U^n + Δt L(U^n)
U^{(2)} = 3/4 U^n + 1/4 U^{(1)} + 1/4 Δt L(U^{(1)})
U^{n+1} = 1/3 U^n + 2/3 U^{(2)} + 2/3 Δt L(U^{(2)})
```

RK3는 WENO 기법과 일반적으로 사용됩니다.

## 6. 벤치마크 문제: Orszag-Tang 와류

**Orszag-Tang 와류**는 충격파, 전류 시트, 복잡한 와류 구조의 형성을 특징으로 하는 표준 2D MHD 테스트 문제입니다.

### 6.1 초기 조건

영역: 주기 경계 조건을 갖는 $[0, 1] \times [0, 1]$.

```
ρ = γ²
p = γ
v_x = -sin(2πy)
v_y = sin(2πx)
v_z = 0
B_x = -sin(2πy) / sqrt(4π)
B_y = sin(4πx) / sqrt(4π)
B_z = 0
```

여기서 $\gamma = 5/3$ (단열 지수).

이 설정은 $\beta \sim 1$ (플라즈마와 자기 압력이 비슷함)을 가지며 난류 cascade를 생성합니다.

### 6.2 진화

와류가 진화하면서:
- 충격파가 형성되고 상호작용
- 반대 방향 장 사이의 경계에서 전류 시트 발달
- 자기 재결합 발생
- 와도가 더 작은 스케일로 cascade

**주요 진단**:
- 밀도 윤곽: 충격파 구조 표시
- 자기장 선: 재결합과 위상 변화 설명
- 전류 밀도 $|j_z| = |\nabla \times B|_z$: 전류 시트 위치

### 6.3 예상 결과

$t \approx 0.5$에서 충격파와 와류의 복잡한 패턴이 나타납니다. 미세 구조를 해결하려면 고해상도 시뮬레이션(512² 이상)이 필요합니다.

## 7. MHD에서 Kelvin-Helmholtz 불안정성

**Kelvin-Helmholtz (KH) 불안정성**은 상대적인 전단 운동에 있는 두 유체 사이의 경계면에서 발생합니다.

### 7.1 설정

영역: $x$에서 주기, $y$에서 반사 또는 주기인 $[0, 1] \times [-1, 1]$.

**속도 전단층**:

```
v_x(y) = -V₀ tanh(y / a)
v_y = δv₀ sin(2πx)  (섭동)
```

여기서 $V_0$는 전단 속도, $a$는 전단층 두께, $\delta v_0 \ll V_0$는 섭동 진폭입니다.

**자기장** ($x$ 방향으로 균일):

```
B_x = B₀
B_y = 0
```

### 7.2 선형 안정성 분석

자기장이 없는 경우, 모드 $k$에 대한 KH 성장률은:

```
γ_KH ~ k V₀ / 2  (얇은 전단층에 대해)
```

자기장이 있으면, 자기 장력이 짧은 파장을 안정화합니다. 분산 관계:

```
γ² = k² V₀² - k² v_A²
```

여기서 $v_A = B_0 / \sqrt{\mu_0 \rho}$는 Alfvén 속도입니다.

**안정화 조건**:

```
B₀ > sqrt(μ₀ ρ) V₀  (v_A > V₀)
```

$v_A > V_0$이면, KH 모드가 억제됩니다.

### 7.3 수치 시뮬레이션

초기 조건:

```
ρ = 1
p = 1
v_x = -V₀ tanh(y / a)
v_y = δv₀ sin(2πx)
B_x = B₀
B_y = 0
```

전형적인 매개변수: $V_0 = 1$, $a = 0.1$, $\delta v_0 = 0.01$, $B_0 = 0$에서 $2$.

**관찰**:
- $B_0 = 0$: 고전적인 KH 롤 발달
- $B_0 = 0.5$: KH 성장 느려지지만 여전히 불안정
- $B_0 = 2$: KH 억제, 자기 장력이 안정화

## 8. Python 구현: CT를 사용한 2D MHD 솔버

### 8.1 데이터 구조

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class MHD2D:
    def __init__(self, Nx, Ny, Lx, Ly, gamma=5/3):
        self.Nx, self.Ny = Nx, Ny
        self.Lx, self.Ly = Lx, Ly
        self.dx, self.dy = Lx / Nx, Ly / Ny
        self.gamma = gamma

        # Cell centers
        self.x = np.linspace(0.5*self.dx, Lx - 0.5*self.dx, Nx)
        self.y = np.linspace(0.5*self.dy, Ly - 0.5*self.dy, Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # Staggered grid for magnetic field (CT)
        # Bx at (i-1/2, j), By at (i, j-1/2)
        self.x_Bx = np.linspace(0, Lx, Nx+1)
        self.y_By = np.linspace(0, Ly, Ny+1)

        # Conserved variables (cell-centered)
        self.rho = np.ones((Nx, Ny))
        self.mx = np.zeros((Nx, Ny))
        self.my = np.zeros((Nx, Ny))
        self.mz = np.zeros((Nx, Ny))
        self.E = np.ones((Nx, Ny))

        # Magnetic field (staggered)
        self.Bx = np.zeros((Nx+1, Ny))  # Face-centered in x
        self.By = np.zeros((Nx, Ny+1))  # Face-centered in y
        self.Bz = np.zeros((Nx, Ny))    # Cell-centered (if needed)

        # Electric field (edge-centered)
        self.Ez = np.zeros((Nx+1, Ny+1))

        self.t = 0.0

    def primitive_variables(self):
        """Compute primitive variables from conserved."""
        vx = self.mx / self.rho
        vy = self.my / self.rho
        vz = self.mz / self.rho

        # Average magnetic field to cell centers
        Bx_cc = 0.5 * (self.Bx[:-1, :] + self.Bx[1:, :])
        By_cc = 0.5 * (self.By[:, :-1] + self.By[:, 1:])
        Bz_cc = self.Bz

        B2 = Bx_cc**2 + By_cc**2 + Bz_cc**2
        p = (self.gamma - 1) * (self.E - 0.5 * self.rho * (vx**2 + vy**2 + vz**2) - 0.5 * B2)

        return self.rho, vx, vy, vz, p, Bx_cc, By_cc, Bz_cc

    def compute_dt(self, CFL=0.4):
        """Compute timestep based on CFL condition."""
        rho, vx, vy, vz, p, Bx, By, Bz = self.primitive_variables()

        # Fast magnetosonic speed
        cs = np.sqrt(self.gamma * p / rho)
        va = np.sqrt((Bx**2 + By**2 + Bz**2) / rho)
        cf = np.sqrt(cs**2 + va**2)

        dt_x = self.dx / np.max(np.abs(vx) + cf)
        dt_y = self.dy / np.max(np.abs(vy) + cf)

        return CFL * min(dt_x, dt_y)

    def check_divergence(self):
        """Check divergence of B (should be ~0)."""
        div_B = (self.Bx[1:, :] - self.Bx[:-1, :]) / self.dx + \
                (self.By[:, 1:] - self.By[:, :-1]) / self.dy
        return np.max(np.abs(div_B))
```

### 8.2 자기장을 위한 CT 업데이트

```python
def update_magnetic_field_CT(self):
    """Update magnetic field using Constrained Transport."""
    # Compute electric field Ez at cell corners (edges in 2D)
    # Ez = vx * By - vy * Bx

    # Need velocities and B fields at corners
    # Simple averaging (can be improved with Riemann solver values)

    # Average vx to y-edges
    vx_avg_y = 0.5 * (self.mx[:-1, :] / self.rho[:-1, :] + self.mx[1:, :] / self.rho[1:, :])
    vx_avg_y = np.pad(vx_avg_y, ((1,0), (0,0)), mode='wrap')  # Periodic

    # Average vy to x-edges
    vy_avg_x = 0.5 * (self.my[:, :-1] / self.rho[:, :-1] + self.my[:, 1:] / self.rho[:, 1:])
    vy_avg_x = np.pad(vy_avg_x, ((0,0), (1,0)), mode='wrap')

    # Average Bx to corners (from faces)
    Bx_corner = 0.5 * (self.Bx[:, :-1] + self.Bx[:, 1:])
    Bx_corner = np.pad(Bx_corner, ((0,0), (0,1)), mode='wrap')

    # Average By to corners (from faces)
    By_corner = 0.5 * (self.By[:-1, :] + self.By[1:, :])
    By_corner = np.pad(By_corner, ((0,1), (0,0)), mode='wrap')

    # Compute Ez at corners
    # This is simplified; production codes use Riemann solver at faces
    self.Ez = vx_avg_y * By_corner - vy_avg_x * Bx_corner

    # Update Bx using Ez (discrete Faraday's law)
    # ∂Bx/∂t = -∂Ez/∂y
    dt = self.compute_dt()
    self.Bx[:, :] -= dt / self.dy * (self.Ez[:, 1:] - self.Ez[:, :-1])

    # Update By
    # ∂By/∂t = ∂Ez/∂x
    self.By[:, :] += dt / self.dx * (self.Ez[1:, :] - self.Ez[:-1, :])
```

### 8.3 완전한 2D MHD 솔버 (단순화)

Riemann 솔버, CT, 고차 재구성을 갖춘 완전한 2D MHD 솔버의 복잡성 때문에, 여기서는 개념적 골격을 제공합니다. Athena, FLASH, Pluto와 같은 생산 코드는 이것을 수천 줄로 구현합니다.

**단순화된 알고리즘**:

```python
def step(self, dt):
    """Single timestep using operator splitting."""
    # Step 1: Half step in x direction
    self.step_x(dt / 2)

    # Step 2: Full step in y direction
    self.step_y(dt)

    # Step 3: Half step in x direction
    self.step_x(dt / 2)

    # Update magnetic field using CT
    self.update_magnetic_field_CT()

    self.t += dt

def step_x(self, dt):
    """1D sweep in x direction (simplified)."""
    # For each j, solve 1D Riemann problems along x
    for j in range(self.Ny):
        # Extract 1D slice
        rho_1d = self.rho[:, j]
        mx_1d = self.mx[:, j]
        # ... (other variables)

        # Reconstruct, solve Riemann problem, update
        # (Reuse 1D MHD solver)

        # Update conserved variables
        # self.rho[:, j] = ...
        pass

def step_y(self, dt):
    """1D sweep in y direction."""
    # Similar to step_x but along y
    pass
```

**참고**: 견고한 2D MHD 솔버를 구현하려면 다음이 필요합니다:
- 1D Riemann 솔버 (HLL, HLLD 등)
- 재구성 (PLM, WENO)
- Riemann 솔버 면 상태로부터 CT 전기장 계산
- 경계 조건
- 소스 항 (중력 등, 해당되는 경우)

교육 목적으로 Orszag-Tang 와류 설정과 간단한 forward-Euler 업데이트를 시연합니다.

### 8.4 Orszag-Tang 와류 설정

```python
def init_orszag_tang(mhd, gamma=5/3):
    """Initialize Orszag-Tang vortex."""
    X, Y = mhd.X, mhd.Y

    # Density and pressure
    mhd.rho[:, :] = gamma**2
    p = gamma * np.ones_like(mhd.rho)

    # Velocity
    vx = -np.sin(2 * np.pi * Y)
    vy = np.sin(2 * np.pi * X)
    vz = np.zeros_like(vx)

    mhd.mx = mhd.rho * vx
    mhd.my = mhd.rho * vy
    mhd.mz = mhd.rho * vz

    # Magnetic field (staggered grid)
    # Bx at (i-1/2, j)
    X_Bx, Y_Bx = np.meshgrid(mhd.x_Bx, mhd.y, indexing='ij')
    mhd.Bx[:, :] = -np.sin(2 * np.pi * Y_Bx) / np.sqrt(4 * np.pi)

    # By at (i, j-1/2)
    X_By, Y_By = np.meshgrid(mhd.x, mhd.y_By, indexing='ij')
    mhd.By[:, :] = np.sin(4 * np.pi * X_By) / np.sqrt(4 * np.pi)

    # Bz (cell-centered, zero)
    mhd.Bz[:, :] = 0.0

    # Total energy
    Bx_cc = 0.5 * (mhd.Bx[:-1, :] + mhd.Bx[1:, :])
    By_cc = 0.5 * (mhd.By[:, :-1] + mhd.By[:, 1:])
    B2 = Bx_cc**2 + By_cc**2

    mhd.E = p / (gamma - 1) + 0.5 * mhd.rho * (vx**2 + vy**2 + vz**2) + 0.5 * B2

# Initialize
Nx, Ny = 128, 128
mhd = MHD2D(Nx, Ny, Lx=1.0, Ly=1.0, gamma=5/3)
init_orszag_tang(mhd)

print(f"Orszag-Tang vortex initialized on {Nx}×{Ny} grid")
print(f"Initial div(B) max: {mhd.check_divergence():.3e}")
```

### 8.5 간단한 Forward Euler 업데이트 (시연용)

```python
def simple_update(mhd, dt):
    """
    Simplified forward Euler update (NOT recommended for production).
    For demonstration only.
    """
    rho, vx, vy, vz, p, Bx, By, Bz = mhd.primitive_variables()

    # Compute fluxes (very simplified, ignoring Riemann solver)
    # This will NOT capture shocks correctly!

    # Flux in x direction (at i+1/2, j)
    Fx_rho = mhd.mx  # rho * vx
    # ... (other flux components)

    # Update conserved variables (forward Euler, very crude)
    # drho/dt = -(dFx/dx + dFy/dy)

    # For proper implementation, use Riemann solver at each face
    # This is just a placeholder

    # Update magnetic field via CT
    mhd.update_magnetic_field_CT()

    mhd.t += dt

# Note: This is NOT a working solver, just a skeleton
# For actual simulation, use established codes or implement full Riemann solver
```

### 8.6 시각화

```python
def plot_orszag_tang(mhd):
    """Plot density and magnetic field."""
    rho, vx, vy, vz, p, Bx, By, Bz = mhd.primitive_variables()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Density
    im1 = ax1.contourf(mhd.X, mhd.Y, rho, levels=50, cmap='viridis')
    ax1.set_title(f'Density at t = {mhd.t:.3f}', fontsize=14)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    plt.colorbar(im1, ax=ax1)

    # Magnetic field lines
    ax2.contourf(mhd.X, mhd.Y, np.sqrt(Bx**2 + By**2), levels=50, cmap='plasma')
    ax2.streamplot(mhd.X.T, mhd.Y.T, Bx.T, By.T, color='white', linewidth=0.5, density=1.5)
    ax2.set_title(f'Magnetic Field at t = {mhd.t:.3f}', fontsize=14)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'orszag_tang_t{mhd.t:.3f}.png', dpi=150)
    plt.show()

plot_orszag_tang(mhd)
```

### 8.7 Kelvin-Helmholtz 불안정성 설정

```python
def init_kelvin_helmholtz(mhd, V0=1.0, a=0.1, dv=0.01, B0=0.0):
    """Initialize Kelvin-Helmholtz instability."""
    X, Y = mhd.X, mhd.Y

    # Density (uniform)
    mhd.rho[:, :] = 1.0

    # Pressure (uniform)
    p = 1.0 * np.ones_like(mhd.rho)

    # Velocity shear
    vx = -V0 * np.tanh(Y / a)
    vy = dv * np.sin(2 * np.pi * X)
    vz = np.zeros_like(vx)

    mhd.mx = mhd.rho * vx
    mhd.my = mhd.rho * vy
    mhd.mz = mhd.rho * vz

    # Magnetic field (uniform in x)
    mhd.Bx[:, :] = B0
    mhd.By[:, :] = 0.0
    mhd.Bz[:, :] = 0.0

    # Total energy
    Bx_cc = 0.5 * (mhd.Bx[:-1, :] + mhd.Bx[1:, :])
    By_cc = 0.5 * (mhd.By[:, :-1] + mhd.By[:, 1:])
    B2 = Bx_cc**2 + By_cc**2

    mhd.E = p / (mhd.gamma - 1) + 0.5 * mhd.rho * (vx**2 + vy**2) + 0.5 * B2

# Initialize KH instability
mhd_kh = MHD2D(Nx=256, Ny=256, Lx=1.0, Ly=2.0, gamma=5/3)
init_kelvin_helmholtz(mhd_kh, V0=1.0, a=0.1, dv=0.01, B0=0.5)

print(f"Kelvin-Helmholtz initialized: V0=1.0, B0=0.5")
print(f"Alfvén speed: vA = {0.5 / np.sqrt(1.0):.2f} (should stabilize if vA > V0)")
```

### 8.8 생산급 코드

실제 연구를 위해서는 확립된 MHD 코드를 사용하세요:

**Athena++**: https://github.com/PrincetonUniversity/athena
- C++, 현대적 AMR (adaptive mesh refinement)
- MHD, radiation, GR 옵션
- 발산 없는 B를 위한 CT

**PLUTO**: http://plutocode.ph.unito.it/
- 모듈형, 다양한 물리 모듈 지원
- MHD, 상대론적 MHD, radiation
- 다중 Riemann 솔버, CT

**FLASH**: https://flash.rochester.edu/
- 대규모 천체물리학 시뮬레이션
- AMR, MHD, 유체역학
- 초신성, 별 형성에 널리 사용

**예제**: Orszag-Tang 와류를 위한 Athena 실행:

```bash
# Athena input file (athinput.orszag_tang)
<problem>
problem_id = OrszagTang
gamma = 1.666667

<mesh>
nx1 = 256
nx2 = 256
x1min = 0.0
x1max = 1.0
x2min = 0.0
x2max = 1.0
ix1_bc = periodic
ox1_bc = periodic
ix2_bc = periodic
ox2_bc = periodic

<hydro>
evolution = mhd

<time>
tlim = 0.5
```

실행:

```bash
./athena -i athinput.orszag_tang
```

## 9. 고급 주제

### 9.1 Adaptive Mesh Refinement (AMR)

AMR은 관심 영역(충격파, 전류 시트)에서 그리드를 동적으로 정제하고 다른 곳에서는 조대화하여 계산 비용을 절감합니다.

**MHD AMR의 과제**:
- 정제 경계를 넘어 $\nabla \cdot B = 0$ 보존
- 엇갈린 장에 대한 prolongation과 restriction 연산자

**해결책**:
- 조대-미세 경계에서 플럭스 보정
- 발산 보존 prolongation (예: Balsara 방법)

### 9.2 발산 청소(Divergence Cleaning)

CT의 대안: 0이 아닌 $\nabla \cdot B$를 허용하되 감쇠 메커니즘 추가.

**쌍곡선 발산 청소** (Dedner et al. 2002):

보조 스칼라 장 $\psi$를 추가하고 진화:

```
∂B/∂t + ∇ψ = ... (usual MHD terms)
∂ψ/∂t + c_h² ∇·B = -c_h² ψ / τ
```

여기서 $c_h$는 청소 속도(일반적으로 $c_h \sim c_{fast}$)이고 $\tau$는 감쇠 시간 척도입니다.

이것은 발산 오차를 파동으로 영역 밖으로 전파합니다.

**장점**: 비구조화 그리드에서 작동, CT보다 구현 쉬움
**단점**: $\nabla \cdot B = 0$을 정확히 보장하지 않음, 매개변수 튜닝 필요

### 9.3 양성성 보존 방법(Positivity-Preserving Methods)

각 업데이트 후 $\rho > 0$과 $p > 0$을 보장하는 것이 안정성에 중요합니다.

**기법**:
- 재구성된 상태를 물리적 범위로 제한
- 음의 밀도/압력을 방지하기 위해 플럭스 조정
- 양성성 보존 Riemann 솔버

### 9.4 고차 방법

2차를 넘어:
- **WENO5**: 5차 재구성
- **Discontinuous Galerkin (DG)**: 셀 내 고차, 플럭스를 통해 결합
- **스펙트럴 방법**: 매끄러운 문제에 대해 (충격파에 이상적이지 않음)

절충: 고차는 수치 소산을 줄이지만 비용과 복잡성을 증가시킵니다.

## 10. 요약

이 레슨은 2D MHD를 위한 고급 수치 기법을 다루었습니다:

1. **2D MHD 방정식**: 1D로부터 확장, 2D에서 8개의 보존 변수
2. **유한 체적법**: 셀 중심 이산화, 반이산 형식
3. **차원 분할**: Strang splitting (2차), 1D 솔버 재사용
4. **비분할 방법**: 다차원 결합을 위한 CTU
5. **Constrained Transport**: 엇갈린 그리드 (Yee mesh), $\nabla \cdot B = 0$ 정확히 보존
6. **고차 재구성**: PLM (제한자), WENO (5차)
7. **Orszag-Tang 와류**: 벤치마크 문제, 난류 MHD
8. **Kelvin-Helmholtz 불안정성**: 자기장이 전단층 안정화
9. **Python 구현**: CT를 사용한 2D MHD를 위한 골격 코드

생산 시뮬레이션을 위해서는 광범위하게 테스트되고 최적화된 확립된 코드(Athena, PLUTO, FLASH)를 사용하세요.

## 연습 문제

1. **CFL 조건**: $\Delta x = \Delta y = 0.01$, 고속 자기음파 속도 $c_f = 2$, 최대 유동 속도 $|v| = 1$인 2D MHD 시뮬레이션에 대해 $CFL = 0.5$에 대한 최대 타임스텝을 계산하세요.

2. **발산 보존**: 왜 $B$를 업데이트하기 위한 표준 유한 체적법은 $\nabla \cdot B = 0$을 보존하지 않지만 Constrained Transport는 보존하는지 설명하세요. Yee mesh를 스케치하고 $B_x$, $B_y$, $E_z$가 어디에 위치하는지 표시하세요.

3. **Orszag-Tang 와류**: 왜 Orszag-Tang 와류가 2D MHD 코드에 대한 좋은 테스트 문제입니까? 어떤 물리적 과정을 테스트합니까 (최소 3개 나열)?

4. **Kelvin-Helmholtz 안정화**: $V_0 = 2$ m/s, $\rho = 1$ kg/m³인 전단층에 대해, KH 불안정성을 억제하기 위해 필요한 최소 자기장 $B_0$을 계산하세요 (즉, $v_A \geq V_0$). Tesla로 표현하세요 (진공 투자율 $\mu_0 = 4\pi \times 10^{-7}$ H/m 가정).

5. **PLM 재구성**: 셀 중심 값 $U_{i-1} = 1.0$, $U_i = 1.5$, $U_{i+1} = 2.5$가 주어졌을 때, (a) 중심 차분, (b) minmod 제한자를 사용하여 기울기 $\sigma_i$를 계산하세요. 셀 경계면 $i+1/2$에서 좌측과 우측 상태는 무엇입니까?

6. **CT 전기장**: Constrained Transport에서 셀 코너의 전기장 $E_z$는 인접한 면의 속도와 자기장으로부터 계산됩니다. 4개의 주변 면 중심에서 $v_x$, $v_y$, $B_x$, $B_y$에 대해 $E_z(i, j)$의 공식을 작성하세요. (간단한 평균화 가정.)

7. **차원 분할 오차**: Strang splitting ($L_x^{1/2} L_y L_x^{1/2}$)은 시간에 대해 2차 정확도입니다. 간단한 splitting ($L_x L_y$)을 사용하면 정확도의 차수는 무엇입니까? 왜 Strang splitting이 선호됩니까?

8. **WENO 장점**: WENO 기법은 매끄러운 영역에서 5차 정확하지만 불연속 근처에서 더 낮은 차수로 감소합니다. 왜 이것이 항상 2차 PLM을 사용하는 것에 비해 유익합니까?

9. **AMR 정제 기준**: MHD 시뮬레이션에서 전류 시트 근처에서 그리드를 정제하고 싶습니다. $|\nabla \times B|$에 기반한 정제를 트리거하기 위한 기준을 제안하세요. 조건을 수학적으로 작성하세요.

10. **계산 비용**: 동일한 물리와 Riemann 솔버를 가정하여 $256 \times 256$ 그리드에서의 2D MHD 시뮬레이션과 256 셀이 있는 1D 시뮬레이션의 계산 비용(타임스텝당 연산)을 비교하세요. 비율을 추정하세요 (상수 무시).

---

**이전:** [우주 기상 MHD](./14_Space_Weather.md) | **다음:** [상대론적 MHD](./16_Relativistic_MHD.md)
