# 3. 플라스마 기술 계층 구조

## 학습 목표

- N-체 모델에서 kinetic 모델, fluid 모델까지의 플라스마 기술 계층 구조 이해
- Klimontovich 방정식 유도와 통계적 기술과의 관계 파악
- moment 방정식에서의 closure 문제와 일반적인 closure 기법 설명
- 물리적 regime에 따라 입자, kinetic, fluid 기술을 언제 사용할지 판단
- 서로 다른 기술 수준의 간단한 수치 비교 구현
- 계층 구조 전반에 걸친 정확도와 계산 효율성 간의 trade-off 이해

## 1. 계층 구조 개요

플라스마는 여러 수준의 세밀도(detail)로 기술될 수 있으며, 정확도와 계산 가능성(computational tractability) 사이의 trade-off를 만드는 **모델 계층 구조**를 형성합니다:

```
Hierarchy of Plasma Descriptions:

┌─────────────────────────────────────────────────────────────┐
│  Level 1: N-Body (Microscopic)                              │
│  Track all particles individually: {x_i(t), v_i(t)}         │
│  Phase space: 6N dimensions                                 │
│  Exact but intractable for N ~ 10^20                        │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼ Ensemble averaging
┌─────────────────────────────────────────────────────────────┐
│  Level 2: Kinetic (Statistical)                             │
│  Distribution function: f(x, v, t)                          │
│  Phase space: 6 dimensions + time                           │
│  Vlasov (collisionless) or Boltzmann (collisional)         │
│  Retains velocity space information                         │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼ Velocity moments
┌─────────────────────────────────────────────────────────────┐
│  Level 3: Fluid (Macroscopic)                               │
│  Moments: n(x,t), u(x,t), T(x,t), ...                       │
│  Configuration space: 3 dimensions + time                   │
│  MHD (magnetohydrodynamics)                                 │
│  Requires closure approximation                             │
└─────────────────────────────────────────────────────────────┘
```

각 수준은 그 위의 수준으로부터 **coarse-graining** 또는 **averaging**을 통해 얻어지며, 정보를 희생하는 대신 계산 효율성을 얻습니다.

### 1.1 각 수준을 언제 사용할 것인가

| 기술 방법 | 최적 용도 | 예시 |
|-------------|----------|----------|
| **N-body** | 소수 입자, 강한 상관관계 | Dusty plasmas, molecular dynamics, validation |
| **Kinetic** | 파동-입자 상호작용, non-Maxwellian 분포 | Landau damping, beam instabilities, magnetic reconnection |
| **Fluid** | 큰 스케일, 저주파수, 높은 충돌성 | MHD equilibria, macroscopic stability, turbulence |

선택은 다음에 의존합니다:
- **스케일 분리**: 잘 분리된 빠른/느린 시간스케일이 있는가?
- **충돌성**: 분포가 Maxwellian(fluid)인가 non-Maxwellian(kinetic)인가?
- **계산 자원**: Kinetic 시뮬레이션이 fluid보다 훨씬 비쌉니다

## 2. N-Body 기술

### 2.1 정확한 운동 방정식

전하 $q_\alpha$와 질량 $m_\alpha$를 가진 $N$ 개의 입자에 대해, 정확한 역학은 다음에 의해 지배됩니다:

$$m_\alpha \frac{d\mathbf{v}_\alpha}{dt} = q_\alpha \left(\mathbf{E}(\mathbf{x}_\alpha, t) + \mathbf{v}_\alpha \times \mathbf{B}(\mathbf{x}_\alpha, t)\right)$$

$$\frac{d\mathbf{x}_\alpha}{dt} = \mathbf{v}_\alpha$$

여기서 장(field)들은 모든 전하에 의해 결정됩니다:

$$\mathbf{E}(\mathbf{x}, t) = \sum_{\beta=1}^{N} \frac{q_\beta}{4\pi\epsilon_0} \frac{\mathbf{x} - \mathbf{x}_\beta}{|\mathbf{x} - \mathbf{x}_\beta|^3} + \mathbf{E}_{ext}$$

$$\mathbf{B}(\mathbf{x}, t) = \mathbf{B}_{ext} + \text{(relativistic corrections)}$$

이것은 **6N-차원** 동역학 시스템입니다. $N \sim 10^{20}$ 입자를 가진 일반적인 플라스마의 경우, 이것은 완전히 다룰 수 없습니다(intractable).

### 2.2 Liouville 정리

N-입자 분포 함수 $F_N(\mathbf{x}_1, \mathbf{v}_1, \ldots, \mathbf{x}_N, \mathbf{v}_N, t)$는 **Liouville 방정식**을 만족합니다:

$$\frac{dF_N}{dt} = \frac{\partial F_N}{\partial t} + \sum_{\alpha=1}^{N} \left(\mathbf{v}_\alpha \cdot \frac{\partial F_N}{\partial \mathbf{x}_\alpha} + \frac{\mathbf{F}_\alpha}{m_\alpha} \cdot \frac{\partial F_N}{\partial \mathbf{v}_\alpha}\right) = 0$$

**해석:** 확률 밀도는 위상 공간의 궤적을 따라 보존됩니다(위상 공간에서의 비압축성 흐름).

이것은 정확하지만 여전히 6N 변수를 포함합니다.

## 3. Klimontovich 방정식

### 3.1 미시적 분포 함수

N-body에서 kinetic으로 연결하기 위해, **Klimontovich 미시적 밀도**를 도입합니다:

$$f^{micro}(\mathbf{x}, \mathbf{v}, t) = \sum_{\alpha=1}^{N} \delta(\mathbf{x} - \mathbf{x}_\alpha(t)) \delta(\mathbf{v} - \mathbf{v}_\alpha(t))$$

이것은 delta 함수들의 합으로, 모든 입자의 정확한 위치와 속도를 나타내는 "거친(grainy)" 분포 함수입니다.

### 3.2 Klimontovich 방정식

미시적 분포는 다음을 만족합니다:

$$\frac{\partial f^{micro}}{\partial t} + \mathbf{v} \cdot \nabla f^{micro} + \frac{q}{m}(\mathbf{E}^{micro} + \mathbf{v} \times \mathbf{B}^{micro}) \cdot \nabla_v f^{micro} = 0$$

여기서 $\mathbf{E}^{micro}$와 $\mathbf{B}^{micro}$는 모든 입자(고려 중인 입자 포함)에 의해 생성된 미시적 장입니다.

이것이 **Klimontovich 방정식**입니다 - 여전히 정확하지만, 이제 6N-차원 위상 공간 대신 $(x, v, t)$ 공간에 있습니다.

**핵심 포인트:** $f^{micro}$는 극도로 특이적입니다(delta 함수들의 합). 따라서 직접 유용하지 않습니다. 앙상블 평균(ensemble averaging)을 통해 **smoothing**이 필요합니다.

## 4. Klimontovich에서 Vlasov/Boltzmann으로

### 4.1 앙상블 평균

실현(realization)들의 앙상블에 대한 **통계적 평균**을 수행합니다:

$$f(\mathbf{x}, \mathbf{v}, t) = \langle f^{micro}(\mathbf{x}, \mathbf{v}, t) \rangle$$

이것은 거친 미시적 분포를 **매끄러운 평균 분포**로 대체합니다.

장들도 유사하게 분해됩니다:

$$\mathbf{E}^{micro} = \mathbf{E} + \delta\mathbf{E}$$
$$\mathbf{B}^{micro} = \mathbf{B} + \delta\mathbf{B}$$

여기서 $\mathbf{E}, \mathbf{B}$는 평활화된(평균) 장이고 $\delta\mathbf{E}, \delta\mathbf{B}$는 요동(fluctuation)입니다.

### 4.2 Vlasov 방정식 (무충돌 극한)

Klimontovich 방정식을 평균하고 **상관관계를 무시**하면(평균장 근사), **Vlasov 방정식**을 얻습니다:

$$\frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \frac{q}{m}(\mathbf{E} + \mathbf{v} \times \mathbf{B}) \cdot \nabla_v f = 0$$

이것은 Maxwell 방정식과 결합됩니다:

$$\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}, \quad \rho = \sum_s q_s \int f_s d^3v$$

$$\nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \epsilon_0 \mu_0 \frac{\partial \mathbf{E}}{\partial t}, \quad \mathbf{J} = \sum_s q_s \int \mathbf{v} f_s d^3v$$

이것이 무충돌 플라스마 역학을 기술하는 **Vlasov-Maxwell 시스템**입니다.

**가정:**
- 평균장: 각 입자는 개별 입자 장이 아닌 평활화된 장에 반응
- 충돌 없음: $f$는 가역적으로 진화
- 유효 조건: $n\lambda_D^3 \gg 1$ (약한 결합)

### 4.3 Boltzmann 방정식 (충돌 포함)

충돌이 중요할 때, 앙상블 평균은 **충돌항(collision term)**을 도입합니다:

$$\frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \frac{q}{m}(\mathbf{E} + \mathbf{v} \times \mathbf{B}) \cdot \nabla_v f = C[f]$$

여기서 $C[f]$는 **충돌 연산자**로, 일반적으로 Boltzmann 충돌 적분입니다:

$$C[f_a] = \sum_b \int d^3v' \int d\Omega \, \sigma(\Omega) \, |\mathbf{v} - \mathbf{v}'| \left(f_a' f_b'^* - f_a f_b\right)$$

여기서 $f_a'$는 $f_a(\mathbf{v}')$를 나타내고, $f_b'^*$는 충돌 후의 $f_b(\mathbf{v}'^*)$를 나타냅니다.

Coulomb 충돌의 경우, 충돌 적분은 더 복잡합니다(Landau 또는 Fokker-Planck 형태).

## 5. BBGKY 계층 구조

### 5.1 축약 분포 함수

대안적인 체계적 접근법은 **축약 분포 함수(reduced distribution functions)**를 사용합니다:

- **1-입자 분포** $f_1(\mathbf{x}_1, \mathbf{v}_1, t)$: $(\mathbf{x}_1, \mathbf{v}_1)$에서 임의의 입자를 찾을 확률
- **2-입자 분포** $f_2(\mathbf{x}_1, \mathbf{v}_1, \mathbf{x}_2, \mathbf{v}_2, t)$: 결합 확률
- 등등...

이것들은 N-입자 분포를 적분하여 얻습니다:

$$f_1(\mathbf{x}_1, \mathbf{v}_1, t) = \int d^3x_2 \cdots d^3x_N \, d^3v_2 \cdots d^3v_N \, F_N$$

### 5.2 BBGKY 방정식

$f_1$에 대한 방정식은 $f_2$를 포함하고; $f_2$에 대한 방정식은 $f_3$를 포함하며; 등등:

$$\frac{\partial f_1}{\partial t} + \mathbf{v}_1 \cdot \nabla_1 f_1 + \frac{\mathbf{F}_1^{ext}}{m} \cdot \nabla_{v_1} f_1 = \int d^3x_2 d^3v_2 \, \frac{\mathbf{F}_{12}}{m} \cdot \nabla_{v_1} f_2$$

이것이 **BBGKY 계층 구조** (Bogoliubov-Born-Green-Kirkwood-Yvon)입니다.

**Closure:** $f_1$을 풀기 위해, $f_1$ 관점에서 $f_2$에 대한 근사가 필요합니다. 일반적인 근사:
- **평균장**: $f_2(\mathbf{x}_1, \mathbf{v}_1, \mathbf{x}_2, \mathbf{v}_2) \approx f_1(\mathbf{x}_1, \mathbf{v}_1) f_1(\mathbf{x}_2, \mathbf{v}_2)$ (상관관계 없음)
- **Boltzmann**: 2-체 상관관계 포함하지만 고차 항은 무시

평균장 closure는 Vlasov 방정식을 복원합니다.

## 6. Fluid Moment와 Closure

### 6.1 Moment 정의

kinetic 분포 $f(\mathbf{x}, \mathbf{v}, t)$로부터, 속도에 대한 적분으로 **fluid moment**를 정의합니다:

**밀도 (0차 moment):**

$$n(\mathbf{x}, t) = \int f(\mathbf{x}, \mathbf{v}, t) \, d^3v$$

**평균 속도 (1차 moment):**

$$\mathbf{u}(\mathbf{x}, t) = \frac{1}{n} \int \mathbf{v} \, f(\mathbf{x}, \mathbf{v}, t) \, d^3v$$

**압력 텐서 (2차 moment):**

$$\mathsf{P}(\mathbf{x}, t) = m \int (\mathbf{v} - \mathbf{u})(\mathbf{v} - \mathbf{u}) \, f(\mathbf{x}, \mathbf{v}, t) \, d^3v$$

등방성 분포의 경우, $\mathsf{P} = p \mathsf{I}$이며 스칼라 압력 $p = nk_B T$입니다.

**열 흐름 (3차 moment):**

$$\mathbf{q}(\mathbf{x}, t) = \frac{m}{2} \int |\mathbf{v} - \mathbf{u}|^2 (\mathbf{v} - \mathbf{u}) \, f(\mathbf{x}, \mathbf{v}, t) \, d^3v$$

### 6.2 Moment 방정식

Vlasov/Boltzmann 방정식의 moment를 취하면 fluid 방정식의 계층 구조가 생성됩니다:

**0차 moment (연속 방정식):**

$$\frac{\partial n}{\partial t} + \nabla \cdot (n\mathbf{u}) = 0$$

**1차 moment (운동량):**

$$mn\left(\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u}\right) = qn(\mathbf{E} + \mathbf{u} \times \mathbf{B}) - \nabla \cdot \mathsf{P} + \mathbf{R}$$

여기서 $\mathbf{R}$은 충돌로 인한 운동량 전달입니다.

**2차 moment (에너지/압력):**

$$\frac{\partial}{\partial t}\left(\frac{3}{2}p\right) + \nabla \cdot \left(\frac{3}{2}p\mathbf{u}\right) + \mathsf{P}:\nabla\mathbf{u} + \nabla \cdot \mathbf{q} = Q$$

여기서 $Q$는 가열/냉각을 나타냅니다.

### 6.3 Closure 문제

**문제:** 각 moment 방정식은 그 다음 고차 moment를 도입합니다:
- 연속 방정식은 $\mathbf{u}$를 포함
- 운동량 방정식은 $\mathsf{P}$를 포함
- 에너지 방정식은 $\mathbf{q}$를 포함
- 등등...

이것은 **무한 계층 구조**입니다 - 방정식보다 미지수가 더 많습니다.

**해결책:** **Closure** - 고차 moment를 저차 moment 관점에서 근사합니다.

### 6.4 일반적인 Closure 방법

**1. 단열 closure (MHD):**

$\mathbf{q} = 0$ (열 전도 없음)을 가정하고 단열 상태 방정식을 가진 등방성 압력:

$$p \propto n^\gamma$$

여기서 $\gamma$는 단열 지수(monatomic gas의 경우 5/3)입니다.

**2. 등온 closure:**

$T = \text{const}$를 가정하여, $p = nk_B T$.

**3. Braginskii closure:**

자화된 플라스마의 경우, kinetic theory에서 유도된 평행 및 수직 열 흐름과 점성 텐서를 사용합니다.

**4. Moment closure (예: 13-moment):**

현상학적 closure와 함께 더 많은 moment(예: $\mathsf{P}$, $\mathbf{q}$, 4차 moment)를 유지합니다.

## 7. 기술 방법 비교

### 7.1 계산 비용

| 방법 | 변수 | 차원 | 일반적인 격자 크기 | 확장성 |
|--------|-----------|----------------|-------------------|-------------|
| N-body | $6N$ | 6N-차원 | — | $\mathcal{O}(N^2)$ 또는 $\mathcal{O}(N\log N)$ |
| PIC (particle-in-cell) | $N$ 입자 + 장 | 3D + 입자 | $10^6$ 셀, $10^9$ 입자 | $\mathcal{O}(N)$ |
| Vlasov (continuum) | $f(x,v,t)$ | 6D + 시간 | $10^6$ 격자점 (3D-3V) | 격자 의존적 |
| Gyrokinetic | $f(x,v_\parallel,\mu,t)$ | 5D + 시간 | $10^5$ 격자점 | 격자 의존적 |
| Fluid (MHD) | $n, \mathbf{u}, T, \mathbf{B}$ | 3D + 시간 | $10^5$ 셀 | 격자 의존적 |

**PIC** (Particle-In-Cell)는 하이브리드입니다: 입자(kinetic)를 사용하지만 격자에서 장을 계산하여 비용을 $\mathcal{O}(N)$으로 줄입니다.

### 7.2 물리적 충실도

```
Physical Fidelity:

High ┌────────────────────────────────────────────────┐
  ↑  │ N-body                                         │
  │  │  • Exact (within classical EM)                 │
  │  │  • All correlations included                   │
  │  │  • Intractable for large N                     │
  │  └────────────────────────────────────────────────┘
  │                   ↓
  │  ┌────────────────────────────────────────────────┐
  │  │ Kinetic (Vlasov/Boltzmann)                     │
  │  │  • Retains velocity distribution               │
  │  │  • Captures wave-particle interactions         │
  │  │  • Non-Maxwellian effects                      │
  │  │  • Expensive: 6D phase space                   │
  │  └────────────────────────────────────────────────┘
  │                   ↓
  │  ┌────────────────────────────────────────────────┐
  │  │ Fluid (MHD)                                    │
  │  │  • Only moments (n, u, T)                      │
  │  │  • Assumes local thermodynamic equilibrium     │
  │  │  • Closure approximation required              │
  │  │  • Fast: 3D only                               │
Low └────────────────────────────────────────────────┘
```

### 7.3 유효 영역

**Fluid (MHD)가 유효한 경우:**
- 높은 충돌성: $\nu \gg \omega$ (충돌 주파수가 파동 주파수를 초과)
- Maxwellian에 가까운 분포
- 길이 스케일 $\gg r_{L,i}$ (이온 Larmor 반지름)
- 시간 스케일 $\gg \omega_{ci}^{-1}$ (이온 gyroperiod)

**Kinetic이 필요한 경우:**
- 파동-입자 공명(Landau damping, cyclotron resonance)
- Non-Maxwellian 특성(beam, loss cone)
- 무충돌 충격파
- 자기 재결합(전자 스케일)

## 8. 수치 예제

### 8.1 1차원 플라스마 진동

간단한 1D 플라스마 진동에 대해 N-body, Vlasov, fluid 기술을 비교하겠습니다.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Physical constants
e = 1.602176634e-19
m_e = 9.1093837015e-31
epsilon_0 = 8.8541878128e-12
k_B = 1.380649e-23

# Plasma parameters
n0 = 1e18  # m^-3
T = 1e5    # K (~ 10 eV)
L = 1.0    # Domain length [m]
omega_pe = np.sqrt(n0 * e**2 / (epsilon_0 * m_e))

print(f"Plasma frequency: ω_pe = {omega_pe:.3e} rad/s")
print(f"Plasma period: τ_pe = {2*np.pi/omega_pe:.3e} s")

# Perturbation
k = 2 * np.pi / L  # Wavenumber
amplitude = 0.01   # Perturbation amplitude

class NBodyPlasma1D:
    """1D N-body electrostatic plasma simulation."""

    def __init__(self, N, L, n0, T, amplitude, k):
        self.N = N
        self.L = L
        self.n0 = n0
        self.T = T
        self.v_th = np.sqrt(k_B * T / m_e)

        # Initialize positions (perturbed uniform distribution)
        self.x = np.linspace(0, L, N, endpoint=False)
        self.x += amplitude * L / k * np.sin(k * self.x)
        self.x = self.x % L  # Periodic

        # Initialize velocities (Maxwellian)
        self.v = np.random.normal(0, self.v_th, N)

        # Charge per particle
        self.q = -e
        self.m = m_e

    def compute_field(self, Ng=128):
        """Compute electric field using Poisson solver on grid."""
        # Deposit charge to grid
        rho_grid, edges = np.histogram(self.x, bins=Ng, range=(0, self.L))
        dx = self.L / Ng
        rho_grid = self.q * rho_grid / dx  # Charge density

        # Background neutralizing charge
        rho_grid -= self.q * self.N / self.L

        # Solve Poisson equation: d^2 phi / dx^2 = -rho / epsilon_0
        # Using FFT
        rho_k = np.fft.rfft(rho_grid)
        k_modes = 2 * np.pi * np.fft.rfftfreq(Ng, d=dx)
        k_modes[0] = 1  # Avoid division by zero (set DC to zero)

        phi_k = -rho_k / (epsilon_0 * k_modes**2)
        phi_k[0] = 0  # No DC potential

        # Electric field: E = -d phi / dx
        E_k = 1j * k_modes * phi_k
        E_grid = np.fft.irfft(E_k, n=Ng)

        return E_grid, edges

    def step(self, dt):
        """Advance system by dt using leapfrog."""
        # Compute field
        E_grid, edges = self.compute_field()

        # Interpolate field to particle positions
        Ng = len(E_grid)
        dx = self.L / Ng
        indices = np.floor(self.x / dx).astype(int) % Ng
        E_particles = E_grid[indices]

        # Push velocities (half step)
        self.v += (self.q / self.m) * E_particles * (dt / 2)

        # Push positions
        self.x += self.v * dt
        self.x = self.x % L  # Periodic BC

        # Push velocities (half step)
        E_grid, _ = self.compute_field()
        indices = np.floor(self.x / dx).astype(int) % Ng
        E_particles = E_grid[indices]
        self.v += (self.q / self.m) * E_particles * (dt / 2)


class VlasovPlasma1D:
    """1D Vlasov-Poisson simulation on phase space grid."""

    def __init__(self, Nx, Nv, L, v_max, n0, T, amplitude, k):
        self.Nx = Nx
        self.Nv = Nv
        self.L = L
        self.v_max = v_max

        self.x = np.linspace(0, L, Nx, endpoint=False)
        self.v = np.linspace(-v_max, v_max, Nv)
        self.dx = L / Nx
        self.dv = 2 * v_max / Nv

        X, V = np.meshgrid(self.x, self.v, indexing='ij')

        # Initial distribution: perturbed Maxwellian
        v_th = np.sqrt(k_B * T / m_e)
        f_max = (1 / (np.sqrt(2*np.pi) * v_th)) * np.exp(-V**2 / (2*v_th**2))
        density_pert = n0 * (1 + amplitude * np.sin(k * X))

        self.f = density_pert[:, np.newaxis] * f_max

    def compute_field(self):
        """Compute electric field from Poisson equation."""
        # Density
        n = np.trapz(self.f, self.v, axis=1)

        # Charge density (with neutralizing background)
        rho = -e * (n - n0)

        # Solve Poisson via FFT
        rho_k = np.fft.rfft(rho)
        k_modes = 2 * np.pi * np.fft.rfftfreq(self.Nx, d=self.dx)
        k_modes[0] = 1

        phi_k = -rho_k / (epsilon_0 * k_modes**2)
        phi_k[0] = 0

        E_k = 1j * k_modes * phi_k
        E = np.fft.irfft(E_k, n=self.Nx)

        return E

    def step(self, dt):
        """Advance Vlasov equation using splitting."""
        E = self.compute_field()

        # Step 1: Advection in x (v * df/dx)
        for j in range(self.Nv):
            v_val = self.v[j]
            shift = int(np.round(v_val * dt / self.dx))
            self.f[:, j] = np.roll(self.f[:, j], -shift)

        # Step 2: Acceleration in v ((qE/m) * df/dv)
        for i in range(self.Nx):
            accel = -e * E[i] / m_e
            shift = int(np.round(accel * dt / self.dv))
            self.f[i, :] = np.roll(self.f[i, :], -shift)


class FluidPlasma1D:
    """1D cold fluid (two-fluid) model."""

    def __init__(self, Nx, L, n0, amplitude, k):
        self.Nx = Nx
        self.L = L
        self.dx = L / Nx

        self.x = np.linspace(0, L, Nx, endpoint=False)

        # Initialize density and velocity
        self.n = n0 * (1 + amplitude * np.sin(k * self.x))
        self.u = np.zeros(Nx)  # Initially at rest

    def compute_field(self):
        """Compute electric field."""
        rho = -e * (self.n - n0)

        rho_k = np.fft.rfft(rho)
        k_modes = 2 * np.pi * np.fft.rfftfreq(self.Nx, d=self.dx)
        k_modes[0] = 1

        phi_k = -rho_k / (epsilon_0 * k_modes**2)
        phi_k[0] = 0

        E_k = 1j * k_modes * phi_k
        E = np.fft.irfft(E_k, n=self.Nx)

        return E

    def step(self, dt):
        """Advance using simple Euler (for demonstration)."""
        E = self.compute_field()

        # Continuity: dn/dt = -d(nu)/dx
        nu = self.n * self.u
        nu_k = np.fft.rfft(nu)
        k_modes = 2 * np.pi * np.fft.rfftfreq(self.Nx, d=self.dx)
        d_nu_dx = np.fft.irfft(1j * k_modes * nu_k, n=self.Nx)

        self.n -= d_nu_dx * dt

        # Momentum (cold, neglecting pressure): du/dt = qE/m - u * du/dx
        u_k = np.fft.rfft(self.u)
        du_dx = np.fft.irfft(1j * k_modes * u_k, n=self.Nx)

        self.u += (-e * E / m_e - self.u * du_dx) * dt


# Run simulations
print("\nRunning simulations...")

# N-body
N_particles = 10000
nbody = NBodyPlasma1D(N_particles, L, n0, T, amplitude, k)

# Vlasov
Nx_vlasov = 64
Nv_vlasov = 64
v_max = 5 * np.sqrt(k_B * T / m_e)
vlasov = VlasovPlasma1D(Nx_vlasov, Nv_vlasov, L, v_max, n0, T, amplitude, k)

# Fluid
Nx_fluid = 128
fluid = FluidPlasma1D(Nx_fluid, L, n0, amplitude, k)

# Time stepping
dt = 0.01 / omega_pe
Nt = 200
times = np.arange(Nt) * dt

# Storage for diagnostics
nbody_density_history = []
vlasov_density_history = []
fluid_density_history = []

for step in range(Nt):
    # N-body
    nbody.step(dt)
    rho_nb, _ = np.histogram(nbody.x, bins=64, range=(0, L))
    rho_nb = rho_nb / (L/64) * N_particles / n0  # Normalize
    nbody_density_history.append(rho_nb)

    # Vlasov
    vlasov.step(dt)
    n_vl = np.trapz(vlasov.f, vlasov.v, axis=1) / n0
    vlasov_density_history.append(n_vl)

    # Fluid
    fluid.step(dt)
    fluid_density_history.append(fluid.n / n0)

print("Simulations complete.")

# Plot comparison at t = π/ω_pe (quarter period)
idx = int(np.pi / (omega_pe * dt) / 2)

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

x_plot = np.linspace(0, L, 64)

ax = axes[0]
ax.plot(x_plot, nbody_density_history[idx], 'o-', label='N-body', alpha=0.7)
ax.plot(x_plot, nbody_density_history[0], '--', label='Initial', alpha=0.5)
ax.set_ylabel('Normalized Density', fontsize=11)
ax.set_title('N-body Simulation', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
x_vlasov = vlasov.x
ax.plot(x_vlasov, vlasov_density_history[idx], 's-', label='Vlasov', alpha=0.7, markersize=4)
ax.plot(x_vlasov, vlasov_density_history[0], '--', label='Initial', alpha=0.5)
ax.set_ylabel('Normalized Density', fontsize=11)
ax.set_title('Vlasov Simulation', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2]
x_fluid = fluid.x
ax.plot(x_fluid, fluid_density_history[idx], '^-', label='Fluid', alpha=0.7, markersize=3)
ax.plot(x_fluid, fluid_density_history[0], '--', label='Initial', alpha=0.5)
ax.set_xlabel('Position [m]', fontsize=11)
ax.set_ylabel('Normalized Density', fontsize=11)
ax.set_title('Fluid Simulation', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plasma_oscillation_comparison.png', dpi=150)
plt.show()

print(f"\nPlot shows density at t = {times[idx]:.2e} s ≈ π/(2ω_pe)")
```

### 8.2 위상 공간 진화 (Vlasov)

```python
def plot_phase_space_evolution():
    """Visualize phase space evolution for Vlasov simulation."""

    # Reinitialize
    vlasov2 = VlasovPlasma1D(64, 128, L, v_max, n0, T, amplitude, k)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    times_plot = [0, int(0.25*Nt), int(0.5*Nt), int(0.75*Nt)]
    titles = ['t = 0', 't = π/(2ω_pe)', 't = π/ω_pe', 't = 3π/(2ω_pe)']

    for ax, t_idx, title in zip(axes.flat, times_plot, titles):
        # Advance to desired time
        for _ in range(t_idx):
            vlasov2.step(dt)

        X, V = np.meshgrid(vlasov2.x, vlasov2.v, indexing='ij')

        contour = ax.contourf(X, V / np.sqrt(k_B * T / m_e),
                              vlasov2.f.T, levels=20, cmap='viridis')
        ax.set_xlabel('Position x [m]', fontsize=11)
        ax.set_ylabel(r'Velocity $v/v_{th}$', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        plt.colorbar(contour, ax=ax, label=r'$f(x,v)$')

    plt.tight_layout()
    plt.savefig('vlasov_phase_space.png', dpi=150)
    plt.show()

plot_phase_space_evolution()
```

### 8.3 에너지 보존 확인

```python
def compare_energy_conservation():
    """Compare energy conservation across methods."""

    # Reinitialize
    nbody3 = NBodyPlasma1D(10000, L, n0, T, amplitude, k)
    vlasov3 = VlasovPlasma1D(64, 64, L, v_max, n0, T, amplitude, k)
    fluid3 = FluidPlasma1D(128, L, n0, amplitude, k)

    nbody_KE = []
    vlasov_KE = []
    fluid_KE = []

    for step in range(Nt):
        # N-body kinetic energy
        KE_nb = 0.5 * m_e * np.sum(nbody3.v**2)
        nbody_KE.append(KE_nb)
        nbody3.step(dt)

        # Vlasov kinetic energy
        V_grid = vlasov3.v[np.newaxis, :]
        KE_vl = 0.5 * m_e * np.sum(vlasov3.f * V_grid**2) * vlasov3.dx * vlasov3.dv
        vlasov_KE.append(KE_vl)
        vlasov3.step(dt)

        # Fluid kinetic energy
        KE_fl = 0.5 * m_e * np.sum(fluid3.n * fluid3.u**2) * fluid3.dx
        fluid_KE.append(KE_fl)
        fluid3.step(dt)

    # Normalize to initial value
    nbody_KE = np.array(nbody_KE) / nbody_KE[0]
    vlasov_KE = np.array(vlasov_KE) / vlasov_KE[0]
    fluid_KE = np.array(fluid_KE) / fluid_KE[0]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(times * omega_pe, nbody_KE, label='N-body', linewidth=2, alpha=0.8)
    ax.plot(times * omega_pe, vlasov_KE, label='Vlasov', linewidth=2, alpha=0.8)
    ax.plot(times * omega_pe, fluid_KE, label='Fluid', linewidth=2, alpha=0.8)

    ax.set_xlabel(r'Time $\omega_{pe} t$', fontsize=12)
    ax.set_ylabel('Normalized Kinetic Energy', fontsize=12)
    ax.set_title('Energy Conservation Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('energy_conservation_comparison.png', dpi=150)
    plt.show()

compare_energy_conservation()
```

## 요약

플라스마 기술의 계층 구조는 다양한 세밀도 수준에서 플라스마를 모델링하기 위한 체계적인 프레임워크를 제공합니다:

1. **N-body**: 모든 입자의 정확한 고전 역학, 실제 플라스마($N \sim 10^{20}$)에는 다룰 수 없음.

2. **Klimontovich 방정식**: 정확한 미시적 분포 함수(delta 함수들의 합), N-body와 통계적 기술을 연결.

3. **Kinetic (Vlasov/Boltzmann)**: 6D 위상 공간에서 매끄러운 분포 함수 $f(x, v, t)$, 앙상블 평균으로 얻음. 파동-입자 상호작용에 필수적인 속도 공간 정보 유지.

4. **Fluid (MHD)**: 3D 구성 공간에서 속도 moment $n, \mathbf{u}, T$, 속도에 대한 적분으로 얻음. moment 계층 구조를 절단하기 위해 closure 필요.

5. **Closure 문제**: Moment 방정식은 무한 계층 구조를 형성; 시스템을 닫기 위해 근사(단열, 등온, Braginskii)가 필요.

6. **영역 의존성**: 강한 상관관계에는 입자 방법, non-Maxwellian 분포와 파동-입자 공명에는 kinetic, 대규모 MHD 현상에는 fluid.

각 수준이 언제 적절한지, 그리고 그것들 사이를 어떻게 탐색할지 이해하는 것은 효율적이고 정확한 플라스마 모델링에 필수적입니다.

## 연습 문제

### 문제 1: Klimontovich에서 Vlasov로

Klimontovich 방정식에서 시작:

$$\frac{\partial f^{micro}}{\partial t} + \mathbf{v} \cdot \nabla f^{micro} + \frac{q}{m}(\mathbf{E}^{micro} + \mathbf{v} \times \mathbf{B}^{micro}) \cdot \nabla_v f^{micro} = 0$$

(a) $f^{micro} = \langle f^{micro} \rangle + \delta f$ 와 $\mathbf{E}^{micro} = \langle \mathbf{E}^{micro} \rangle + \delta \mathbf{E}$ 로 쓰세요.

(b) Klimontovich 방정식을 앙상블 평균하세요. 어떤 가정이 Vlasov 방정식으로 이끕니까? (힌트: $\langle \delta f \delta \mathbf{E} \rangle$ 무시)

(c) 물리적으로, $\langle \delta f \delta \mathbf{E} \rangle$ 항은 무엇을 나타냅니까? 왜 약하게 결합된 플라스마에서 무시될 수 있습니까?

### 문제 2: Moment Closure

전자 분포 $f_e(x, v, t)$를 가진 1D 정전기 플라스마에 대해:

(a) 연속 방정식(0차 moment)을 유도하세요:
   $$\frac{\partial n_e}{\partial t} + \frac{\partial (n_e u_e)}{\partial x} = 0$$

(b) 운동량 방정식(1차 moment)을 유도하여, 압력 $p_e$를 포함함을 보이세요.

(c) 상수 $T_e$로 등온 closure $p_e = n_e k_B T_e$를 가정하세요. Poisson 방정식과 결합하여, 작은 섭동이 다음을 만족함을 보이세요:
   $$\frac{\partial^2 n_e}{\partial t^2} = \frac{k_B T_e}{m_e} \frac{\partial^2 n_e}{\partial x^2} + \omega_{pe}^2 (n_e - n_0)$$

(d) 평면파 $n_e \propto e^{i(kx - \omega t)}$에 대해, 분산 관계를 구하세요. kinetic theory의 Langmuir 파동과 비교하세요.

### 문제 3: 무충돌 vs 충돌 영역

주파수 $\omega \approx \omega_{pe}$를 가진 플라스마 진동을 고려하세요.

(a) Knudsen 수 $Kn = \lambda_{mfp}/L$과 충돌 주파수 $\nu_{ei}$를 사용하여, 진동이 무충돌이기 위한 기준을 결정하세요.

(b) $n_e = 10^{20}$ m$^{-3}$, $T_e = 10$ keV, $L = 1$ m의 tokamak에 대해, $\nu_{ei}$와 $\omega_{pe}$를 계산하세요. 플라스마 진동이 무충돌입니까?

(c) $n_e = 10^{16}$ m$^{-3}$, $T_e = 2$ eV, $L = 0.1$ m의 glow discharge에 대해 반복하세요.

(d) 이 결과들을 기반으로, 어떤 시스템이 플라스마 진동에 대해 kinetic 기술을 요구합니까?

### 문제 4: 위상 공간 밀도 보존

(a) Vlasov 방정식이 다음과 같이 쓰일 수 있음을 보이세요:
   $$\frac{df}{dt} = \frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \mathbf{a} \cdot \nabla_v f = 0$$
   여기서 $\mathbf{a} = (q/m)(\mathbf{E} + \mathbf{v} \times \mathbf{B})$.

(b) 이것을 $(x, v)$ 위상 공간의 입자 궤적을 따라 $df/dt = 0$로 해석하세요.

(c) 이것을 사용하여 위상 공간의 부피가 보존됨을 논하세요(**Liouville 정리**).

(d) Boltzmann 방정식(충돌 포함)이 왜 위상 공간 부피 보존을 위반하는지 설명하세요.

### 문제 5: Fluid vs Kinetic Landau Damping

Langmuir 파동의 감쇠는 fluid와 kinetic 처리 사이에서 다릅니다.

(a) 문제 2에서, Langmuir 파동에 대한 fluid 분산 관계는:
   $$\omega^2 = \omega_{pe}^2 + 3k^2 v_{te}^2$$
   이것이 **감쇠가 없음**(실수 $\omega$)을 예측함을 보이세요.

(b) Landau의 kinetic theory는:
   $$\omega^2 \approx \omega_{pe}^2 + 3k^2 v_{te}^2 - i\sqrt{\frac{\pi}{8}} \frac{\omega_{pe}}{(kv_{te})^3} e^{-1/(2k^2\lambda_D^2)}$$
   $k\lambda_D \ll 1$에 대해. 감쇠를 담당하는 허수부를 식별하세요.

(c) 물리적으로, 왜 fluid 모델은 Landau damping을 놓칩니까? (힌트: $v \approx \omega/k$에서의 공명 입자들.)

(d) $n_e = 10^{19}$ m$^{-3}$, $T_e = 100$ eV, $k = 100$ m$^{-1}$에 대해, Landau damping rate $\gamma = \text{Im}(\omega)$를 추정하세요.

---

**이전:** [Coulomb Collisions](./02_Coulomb_Collisions.md) | **다음:** [Single Particle Motion I](./04_Single_Particle_Motion_I.md)
