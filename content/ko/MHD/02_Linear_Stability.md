# 2. 선형 안정성 이론

## 학습 목표

- 평형 주변에서 이상 MHD 방정식의 선형화 이해
- 힘 연산자 유도 및 정규 모드에 대한 고유값 문제 수식화
- 고유값 문제를 풀지 않고 안정성을 결정하는 에너지 원리 적용
- 특정 평형에 대한 성장률 및 안정성 경계 계산
- 외부 kink 안정성에 대한 Kruskal-Shafranov 기준 이해
- 국소 interchange 안정성에 대한 Suydam 기준 적용
- MHD 안정성 분석을 위한 수치 고유값 솔버 구현

## 1. MHD 안정성 소개

MHD 평형은 힘 균형을 만족하지만 작은 섭동에 대해 **불안정**할 수 있습니다. 안정성 분석은 섭동이 성장하는지 (불안정), 감쇠/진동하는지 (안정)를 결정합니다.

**선형 안정성 이론**은 무한소 섭동의 진화를 조사합니다:

```
평형 상태: (p₀, ρ₀, B₀, v₀=0)
섭동 상태:   (p₀+p₁, ρ₀+ρ₁, B₀+B₁, v₁)

여기서 |섭동| << |평형|
```

주요 질문:
- 섭동이 지수적으로 성장합니까? (불안정)
- 성장 없이 진동합니까? (한계 안정)
- 감쇠합니까? (안정)

**성장률** $\gamma$ 또는 **주파수** $\omega$는 고유값 문제를 풀어 결정됩니다.

## 2. 이상 MHD의 선형화

### 2.1 섭동 전개

평형으로부터의 플라즈마 변위 $\boldsymbol{\xi}(\mathbf{x}, t)$를 고려하세요. Lagrangian 변위는 섭동된 양과 관계됩니다:

$$
\mathbf{x}' = \mathbf{x} + \boldsymbol{\xi}(\mathbf{x}, t)
$$

섭동된 양 (Eulerian 설명):

$$
\begin{aligned}
\rho_1 &= -\nabla\cdot(\rho_0\boldsymbol{\xi}) \\
\mathbf{v}_1 &= \frac{\partial\boldsymbol{\xi}}{\partial t} \\
\mathbf{B}_1 &= \nabla\times(\boldsymbol{\xi}\times\mathbf{B}_0) \\
p_1 &= -\boldsymbol{\xi}\cdot\nabla p_0 - \gamma p_0\nabla\cdot\boldsymbol{\xi}
\end{aligned}
$$

여기서 $\gamma$는 단열 지수입니다.

### 2.2 선형화된 운동량 방정식

이상 MHD 운동량 방정식:

$$
\rho\frac{\partial\mathbf{v}}{\partial t} = -\nabla p + \mathbf{J}\times\mathbf{B}
$$

선형화 (1차 항만 유지):

$$
\rho_0\frac{\partial^2\boldsymbol{\xi}}{\partial t^2} = -\nabla p_1 + \mathbf{J}_1\times\mathbf{B}_0 + \mathbf{J}_0\times\mathbf{B}_1
$$

$\mathbf{J} = \nabla\times\mathbf{B}/\mu_0$ 사용:

$$
\rho_0\frac{\partial^2\boldsymbol{\xi}}{\partial t^2} = -\nabla p_1 + \frac{1}{\mu_0}(\nabla\times\mathbf{B}_1)\times\mathbf{B}_0 + \frac{1}{\mu_0}(\nabla\times\mathbf{B}_0)\times\mathbf{B}_1
$$

이것은 간결하게 다음과 같이 쓸 수 있습니다:

$$
\rho_0\frac{\partial^2\boldsymbol{\xi}}{\partial t^2} = \mathbf{F}(\boldsymbol{\xi})
$$

여기서 $\mathbf{F}(\boldsymbol{\xi})$는 **힘 연산자** ($\boldsymbol{\xi}$에 선형)입니다.

### 2.3 정규 모드 분석

정규 모드 시간 의존성 가정:

$$
\boldsymbol{\xi}(\mathbf{x}, t) = \hat{\boldsymbol{\xi}}(\mathbf{x})e^{-i\omega t}
$$

대입:

$$
-\omega^2\rho_0\hat{\boldsymbol{\xi}} = \mathbf{F}(\hat{\boldsymbol{\xi}})
$$

이것은 **고유값 문제**입니다:

$$
\mathbf{F}(\hat{\boldsymbol{\xi}}) = -\omega^2\rho_0\hat{\boldsymbol{\xi}}
$$

- 고유값: $\omega^2$
- 고유 함수: $\hat{\boldsymbol{\xi}}(\mathbf{x})$

**안정성 기준**:
- 모든 $\omega^2 > 0$이면: **안정** (진동 모드)
- $\omega^2 < 0$인 것이 있으면: **불안정** (지수 성장, $\gamma = \sqrt{-\omega^2}$)
- $\omega^2 = 0$이면: **한계 안정성**

### 2.4 힘 연산자의 자기수반 성질

힘 연산자 $\mathbf{F}$는 중요한 성질을 가집니다: **자기수반** (에르미트):

$$
\int \boldsymbol{\xi}_1^*\cdot\mathbf{F}(\boldsymbol{\xi}_2)\, dV = \int \boldsymbol{\xi}_2^*\cdot\mathbf{F}(\boldsymbol{\xi}_1)\, dV
$$

**결과**: 모든 고유값 $\omega^2$는 **실수**입니다.

증명: 고유 모드 $\mathbf{F}(\hat{\boldsymbol{\xi}}) = -\omega^2\rho_0\hat{\boldsymbol{\xi}}$를 고려하세요.

$\hat{\boldsymbol{\xi}}^*$와 내적:

$$
\int \hat{\boldsymbol{\xi}}^*\cdot\mathbf{F}(\hat{\boldsymbol{\xi}})\, dV = -\omega^2\int \rho_0|\hat{\boldsymbol{\xi}}|^2\, dV
$$

$\mathbf{F}$가 자기수반이면 좌변은 실수이므로 $\omega^2$는 실수입니다.

이는 모드가 다음 중 하나임을 의미합니다:
- **진동**: $\omega$ 실수, $\boldsymbol{\xi} \propto e^{-i\omega t}$
- **성장/감쇠**: $\omega$ 허수, $\boldsymbol{\xi} \propto e^{\gamma t}$, $\gamma = |\omega|$

## 3. 에너지 원리

### 3.1 위치 에너지 범함수

고유값 문제를 직접 풀기보다, **Bernstein, Frieman, Kruskal, Kulsrud (1958)**는 안정성이 위치 에너지 범함수의 부호로 결정될 수 있음을 보였습니다.

**섭동된 위치 에너지** 정의:

$$
\delta W = -\frac{1}{2}\int \boldsymbol{\xi}^*\cdot\mathbf{F}(\boldsymbol{\xi})\, dV
$$

고유값 방정식으로부터:

$$
\delta W = \frac{1}{2}\omega^2\int \rho_0|\boldsymbol{\xi}|^2\, dV = \frac{1}{2}\omega^2 K
$$

여기서 $K$는 섭동의 운동 에너지입니다.

**에너지 원리**:
- 모든 허용된 $\boldsymbol{\xi}$에 대해 $\delta W > 0$이면: **안정** ($\omega^2 > 0$)
- 어떤 $\boldsymbol{\xi}$에 대해 $\delta W < 0$이면: **불안정** ($\omega^2 < 0$)
- $\min(\delta W) = 0$이면: **한계 안정성**

### 3.2 $\delta W$의 명시적 형태

위치 에너지는 분해될 수 있습니다:

$$
\delta W = \delta W_F + \delta W_S + \delta W_V
$$

여기서:
- $\delta W_F$: 유체 (벌크) 기여
- $\delta W_S$: 표면 기여
- $\delta W_V$: 진공 기여

**유체 기여**:

$$
\delta W_F = \frac{1}{2}\int\left[\frac{|\mathbf{Q}|^2}{\mu_0} + \gamma p_0|\nabla\cdot\boldsymbol{\xi}|^2 + (\boldsymbol{\xi}\cdot\nabla p_0)(\nabla\cdot\boldsymbol{\xi}^*)\right]dV
$$

여기서:

$$
\mathbf{Q} = \nabla\times(\boldsymbol{\xi}\times\mathbf{B}_0) = \mathbf{B}_1
$$

은 섭동된 자기장입니다.

### 3.3 물리적 해석

$\delta W_F$를 물리적 기여로 분해:

$$
\delta W_F = \delta W_{compression} + \delta W_{tension} + \delta W_{pressure}
$$

**자기 압축**:

$$
\delta W_{compression} = \frac{1}{2\mu_0}\int |\mathbf{B}_1|^2\, dV
$$

자기장선 압축/늘이기는 에너지 비용 (안정화).

**자기 장력**:

$\mathbf{B}_1 = \nabla\times(\boldsymbol{\xi}\times\mathbf{B}_0)$로부터, 자기장선 구부리기는 복원력 생성.

**압력 구동**:

압력 구배 항은 곡률이 불리하면 불안정성을 유발할 수 있습니다.

### 3.4 직관적 형태

비압축 섭동 ($\nabla\cdot\boldsymbol{\xi} = 0$)에 대해:

$$
\delta W \approx \frac{1}{2}\int\left[\frac{|\nabla\times(\boldsymbol{\xi}\times\mathbf{B}_0)|^2}{\mu_0} + \boldsymbol{\xi}_\perp\cdot\nabla p_0\, \nabla\cdot\boldsymbol{\xi}_\perp\right]dV
$$

첫 번째 항 (자기 굽힘)은 항상 양수 (안정화).

두 번째 항 (압력 구동)은 $\nabla p_0$와 $\nabla\cdot\boldsymbol{\xi}_\perp$가 같은 부호이면 음수일 수 있습니다.

**불리한 곡률**: 볼록한 자기장선 곡률을 향하는 압력 구배 → 불안정.

## 4. Kruskal-Shafranov 기준

### 4.1 외부 Kink 불안정성

**외부 kink**는 전체 플라즈마 기둥의 대규모 변위입니다. 토로이달 가둠에서 지배적인 불안정성입니다.

**구성**: 나선형 섭동을 가진 원통 플라즈마:

$$
\boldsymbol{\xi} = \hat{\boldsymbol{\xi}}(r)e^{i(m\theta - nz/R_0)}
$$

여기서:
- $m$: 폴로이달 모드 수
- $n$: 토로이달 모드 수 (토로이달 시스템)
- $(m,n) = (1,1)$이 가장 위험한 모드

### 4.2 날카로운 경계 유도

날카로운 경계 모델 고려:
- $r < a$ 내부에 균일한 전류 밀도: $\mathbf{J} = J_0\hat{\mathbf{z}}$
- $r > a$ 외부에 진공

가장자리에서 안전 인자:

$$
q(a) = \frac{aB_z}{R_0 B_θ(a)}
$$

**안정성 조건** (Kruskal and Shafranov, 1958):

$$
q(a) > \frac{m}{n}
$$

$(m,n) = (1,1)$ 모드에 대해:

$$
q(a) > 1
$$

### 4.3 물리적 해석

kink 불안정성은 다음 사이의 경쟁에서 발생합니다:
- **불안정화**: 기둥이 구부러질 때 자기 압력 불균형
- **안정화**: 축방향 장 $B_z$가 구부림에 저항하는 장력 제공

```
Kink 불안정성 (m=1):
======================

전:          후:
   │                ╱│╲
   │               ╱ │ ╲
   │     →        │  │  │  (기둥 구부러짐)
   │               ╲ │ ╱
   │                ╲│╱

안정화: B_z가 장력 제공
        (더 높은 B_z → 더 높은 q → 안정)
```

$q(a) < 1$에 대해, 자기장선은 폴로이달 회전당 1 토로이달 회전 미만을 완성하며, 구성은 kink에 저항할 수 없습니다.

### 4.4 일반화

확산 전류 프로파일 $J(r)$에 대해:

$$
q(a) > \frac{m}{n}\quad\text{(필요하지만 충분하지 않음)}
$$

추가 조건은 전류 프로파일 형태, 압력 구배, 전도 벽 위치를 포함합니다.

## 5. Suydam 기준

### 5.1 국소 Interchange 불안정성

**Suydam 기준**은 국소화된 interchange 모드에 대한 안정성의 **필요 조건**을 제공합니다.

**Interchange 불안정성**: 인접 플럭스 튜브가 압력 구배와 자기 곡률에 의해 위치를 교환.

### 5.2 유도

평형에서 원통 플라즈마 고려. Suydam 기준은:

$$
\frac{r}{4}\left(\frac{q'}{q}\right)^2 + \frac{2\mu_0 p'}{B_z^2} > 0
$$

모든 $r$에 대해.

**물리적 해석**:

- 첫 번째 항: 자기 전단 $q'/q$가 안정화
- 두 번째 항: 압력 구배가 불안정화 가능

$p' < 0$ (압력 외부로 감소) 그리고 전단이 약하면, 위반 → 불안정성.

### 5.3 응용

다음을 가진 screw pinch에 대해:
- $B_z = \text{const}$
- $B_θ(r) = B_{θ0}r/a$ (선형 프로파일)

안전 인자:

$$
q(r) = \frac{rB_z}{R_0 B_θ} = \frac{B_z a}{R_0 B_{θ0}}\frac{1}{r}
$$

따라서:

$$
\frac{q'}{q} = -\frac{1}{r}
$$

Suydam 기준은:

$$
\frac{1}{4r^2} + \frac{2\mu_0 p'}{B_z^2} > 0
$$

$|p'|$이 너무 크면 기준이 위반됩니다.

### 5.4 제한

Suydam 기준은 **필요**하지만 **충분하지 않음**입니다. 위반하면 불안정성을 보장하지만, 만족해도 안정성을 보장하지 않습니다 (전역 모드가 여전히 존재 가능).

## 6. 성장률 계산

### 6.1 간단한 기하학에 대한 해석적 성장률

$m=0$ (sausage 모드)를 가진 날카로운 경계 Z-pinch에 대해, 성장률은:

$$
\gamma^2 = \frac{k_z^2 B_θ^2(a)}{\mu_0\rho_0}
$$

여기서 $k_z$는 축방향 파수입니다.

**해석**: 안정화 없음 → $k_z \neq 0$에 대해 항상 불안정.

축방향 장 $B_z$ 추가는 짧은 파장 안정화:

$$
\gamma^2 = \frac{B_θ^2(a)}{\mu_0\rho_0}\left(k_z^2 - \frac{B_z^2}{B_θ^2(a)}k_z^2\right)
$$

안정화를 위한 임계 파수:

$$
k_z > k_c = \frac{B_θ(a)}{B_z}
$$

### 6.2 고유값 문제 수식화

일반 기하학에 대해, 고유값 문제:

$$
\mathbf{F}(\hat{\boldsymbol{\xi}}) = -\omega^2\rho_0\hat{\boldsymbol{\xi}}
$$

수치적으로 이산화하고 풀어야 합니다.

**방법**:
1. 기저 함수 또는 그리드 선택
2. $\mathbf{F}$를 이산 공간에 투영 → 행렬 $\mathbf{A}$
3. 행렬 고유값 문제 풀이: $\mathbf{A}\mathbf{x} = \lambda\mathbf{x}$
4. 고유값 $\lambda = -\omega^2$
5. $\lambda > 0$이면: 불안정, $\gamma = \sqrt{\lambda}$

## 7. 수치 구현

### 7.1 유한 요소 이산화

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

class MHDStabilitySolver:
    """
    Solve MHD stability eigenvalue problem for cylindrical geometry
    """

    def __init__(self, nr, r_max, equilibrium):
        """
        nr: number of radial grid points
        r_max: maximum radius
        equilibrium: dict with Bz(r), Btheta(r), p(r), rho(r)
        """
        self.nr = nr
        self.r_max = r_max
        self.r = np.linspace(0, r_max, nr)
        self.dr = self.r[1] - self.r[0]

        self.equilibrium = equilibrium
        self.mu0 = 4*np.pi*1e-7

    def compute_force_operator(self, m, kz):
        """
        Compute force operator matrix for mode (m, kz)
        Simplified 1D radial eigenvalue problem
        """
        nr = self.nr
        r = self.r

        # Get equilibrium profiles
        Bz = np.array([self.equilibrium['Bz'](ri) for ri in r])
        Btheta = np.array([self.equilibrium['Btheta'](ri) for ri in r])
        p = np.array([self.equilibrium['p'](ri) for ri in r])
        rho = np.array([self.equilibrium['rho'](ri) for ri in r])

        # Build matrix (simplified model)
        # Full MHD stability is very complex; here we implement a toy model

        A = np.zeros((nr, nr))

        for i in range(1, nr-1):
            # 라플라시안 구조 d²ξ/dr² + (1/r)dξ/dr - (m²/r²)ξ는
            # 힘 연산자를 m번째 푸리에 고조파에 투영하는 데서 나옵니다;
            # -m²/r² 항은 방위각 방향으로 원통 주위에서 변위를 구부리는
            # 원심 비용입니다.

            # 자기 장력 계수 B²/(μ₀ρ)는 v_A² (알벤 속도 제곱)과 같습니다:
            # 자기장선 굽힘은 v_A²에 비례하는 복원력을 생성하여
            # 높은 m의 짧은 파장 섭동을 안정화합니다 — 이것이 근본적인 안정화 메커니즘입니다.
            tension_coef = (Bz[i]**2 + Btheta[i]**2) / (self.mu0 * rho[i])

            # Diagonal
            A[i, i] = -2*tension_coef/self.dr**2 - m**2*tension_coef/r[i]**2

            # 비대칭 비대각 항 tension/(2r dr)은 라플라시안의
            # 원통형 (1/r)dξ/dr 부분에서 나옵니다; 1차이며
            # r → ∞인 직교 좌표 기하학에서는 사라집니다.
            A[i, i+1] = tension_coef/self.dr**2 + tension_coef/(2*r[i]*self.dr)
            A[i, i-1] = tension_coef/self.dr**2 - tension_coef/(2*r[i]*self.dr)

            # 압력 구배 항 -∇p/ρ는 불안정화 구동으로 작용합니다:
            # 압력이 바깥쪽으로 감소하면 (dp/dr < 0), 이 항은 A에서 양수가 되어
            # 복원력을 줄이고 잠재적으로 성장을 구동합니다.
            if i > 0:
                dpdx = (p[i+1] - p[i-1]) / (2*self.dr)
                A[i, i] += -dpdx / rho[i]

        # Dirichlet 조건 ξ(0) = ξ(r_max) = 0은 축에서의 정칙성 (특이 변위 없음)과
        # 경계에서 섭동 없음(플라즈마 표면이 이 단순화된 모델에서 고정됨)을
        # 강제합니다.
        A[0, 0] = 1.0
        A[-1, -1] = 1.0

        return A

    def solve_stability(self, m, kz):
        """
        Solve eigenvalue problem for mode (m, kz)
        Returns: eigenvalues (growth rates squared), eigenvectors
        """
        A = self.compute_force_operator(m, kz)

        # 힘 연산자 F가 자기수반(에르미트)이므로 ω²의 실수 고유값이 보장됩니다.
        # eigh는 일반 복소 고유값 해석기보다 빠르고 수치적으로 안정적인
        # 이 대칭성을 활용합니다.
        # A ξ = λ ξ, where λ = -ω²
        eigenvalues, eigenvectors = eigh(A)

        # λ > 0은 ω² < 0을 의미하며, 즉 ω가 순허수 → 지수 성장입니다.
        # 부호 관례(λ = -ω²)는 고유값 방정식 F(ξ) = -ω²ρξ에서 나옵니다:
        # 양정치 F (복원력)는 λ < 0 (안정 진동)을 줌으로써 불안정을 나타내고,
        # 음반정치 F는 λ > 0 (불안정)을 줍니다.
        growth_rates_squared = eigenvalues

        return growth_rates_squared, eigenvectors

    def stability_scan(self, m_values, kz_values):
        """
        Scan stability over mode numbers
        Returns: growth rate map
        """
        growth_map = np.zeros((len(m_values), len(kz_values)))

        for i, m in enumerate(m_values):
            for j, kz in enumerate(kz_values):
                eigenvalues, _ = self.solve_stability(m, kz)

                # Maximum growth rate for this mode
                max_growth_sq = np.max(eigenvalues)
                growth_map[i, j] = np.sqrt(max(max_growth_sq, 0))

        return growth_map

    def plot_growth_rate(self, m_values, kz_values, growth_map):
        """Plot growth rate map"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Contour plot
        KZ, M = np.meshgrid(kz_values, m_values)
        levels = 20

        CS = ax.contourf(KZ, M, growth_map, levels=levels, cmap='hot')
        ax.contour(KZ, M, growth_map, levels=[0], colors='cyan', linewidths=2)

        cbar = plt.colorbar(CS, ax=ax)
        cbar.set_label('Growth rate γ [1/s]', fontsize=12)

        ax.set_xlabel('Axial wavenumber kz [1/m]', fontsize=12)
        ax.set_ylabel('Poloidal mode number m', fontsize=12)
        ax.set_title('MHD Instability Growth Rate Map', fontsize=14)

        plt.tight_layout()
        return fig

    def plot_eigenmode(self, m, kz, mode_index=0):
        """Plot eigenmode structure"""
        eigenvalues, eigenvectors = self.solve_stability(m, kz)

        # Sort by eigenvalue (most unstable first)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select mode
        eigenvalue = eigenvalues[mode_index]
        eigenmode = eigenvectors[:, mode_index]

        gamma = np.sqrt(max(eigenvalue, 0))

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.r, eigenmode.real, 'b-', linewidth=2, label='Real part')
        ax.plot(self.r, eigenmode.imag, 'r--', linewidth=2, label='Imaginary part')

        ax.set_xlabel('Radius r [m]', fontsize=12)
        ax.set_ylabel('Displacement ξ (normalized)', fontsize=12)
        ax.set_title(f'Eigenmode (m={m}, kz={kz:.2f}): γ = {gamma:.2e} s⁻¹', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

# Example: Z-pinch stability
def example_zpinch_stability():
    """Analyze stability of a Z-pinch"""

    # Equilibrium parameters
    a = 0.1  # Plasma radius [m]
    I = 1e6  # Current [A]
    n = 1e20 # Density [m^-3]
    T = 1e6  # Temperature [K]
    mu0 = 4*np.pi*1e-7
    kB = 1.38e-23
    mp = 1.67e-27

    # Equilibrium profiles
    def Bz(r):
        return 0.5  # Weak axial field [T]

    def Btheta(r):
        # Enclosed current
        if r < a:
            I_enc = I * (r/a)**2
        else:
            I_enc = I
        return mu0 * I_enc / (2*np.pi*r) if r > 0 else 0

    def p(r):
        # Parabolic pressure
        p0 = 2 * n * kB * T
        if r < a:
            return p0 * (1 - (r/a)**2)
        else:
            return 0

    def rho(r):
        # Uniform density
        return n * mp if r < a else 0.01 * n * mp

    equilibrium = {
        'Bz': Bz,
        'Btheta': Btheta,
        'p': p,
        'rho': rho
    }

    # Solver
    solver = MHDStabilitySolver(nr=100, r_max=2*a, equilibrium=equilibrium)

    print("=== Z-Pinch Stability Analysis ===")
    print(f"Plasma radius: {a*100} cm")
    print(f"Current: {I/1e6} MA")
    print(f"Density: {n:.2e} m^-3")
    print(f"Temperature: {T/1e6} MK")
    print(f"Axial field: {Bz(0)} T")
    print(f"Azimuthal field at edge: {Btheta(a):.3f} T")

    # Stability scan
    m_values = [0, 1, 2, 3]  # Poloidal mode numbers
    kz_values = np.linspace(1, 100, 50)  # Axial wavenumbers [1/m]

    print("\nScanning stability...")
    growth_map = solver.stability_scan(m_values, kz_values)

    # Find most unstable mode
    max_idx = np.unravel_index(np.argmax(growth_map), growth_map.shape)
    max_growth = growth_map[max_idx]
    m_unstable = m_values[max_idx[0]]
    kz_unstable = kz_values[max_idx[1]]

    print(f"\nMost unstable mode: m={m_unstable}, kz={kz_unstable:.1f} m^-1")
    print(f"Growth rate: γ = {max_growth:.2e} s^-1")
    print(f"Growth time: τ = {1/max_growth:.2e} s")

    # Plot growth rate map
    fig1 = solver.plot_growth_rate(m_values, kz_values, growth_map)
    plt.savefig('/tmp/zpinch_growth_map.png', dpi=150)
    print("\nGrowth rate map saved to /tmp/zpinch_growth_map.png")

    # Plot unstable eigenmode
    fig2 = solver.plot_eigenmode(m_unstable, kz_unstable, mode_index=0)
    plt.savefig('/tmp/zpinch_eigenmode.png', dpi=150)
    print("Eigenmode structure saved to /tmp/zpinch_eigenmode.png")

    plt.close('all')

if __name__ == "__main__":
    example_zpinch_stability()
```

### 7.2 안전 인자 분석

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_q_profile(r, Bz, Btheta, R0):
    """
    Compute safety factor q(r) = r*Bz / (R0*Btheta)
    """
    q = r * Bz / (R0 * Btheta + 1e-10)  # Avoid division by zero
    return q

def check_kruskal_shafranov(q_edge, m, n):
    """
    Check Kruskal-Shafranov criterion: q(a) > m/n
    """
    q_crit = m / n
    margin = q_edge - q_crit

    stable = margin > 0

    return stable, margin, q_crit

def plot_q_and_stability(r, q, R0, a):
    """
    Plot q-profile with stability boundaries
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # q-profile
    ax.plot(r/a, q, 'b-', linewidth=2, label='q(r)')

    # Rational surfaces
    rational_q = [1, 1.5, 2, 2.5, 3]
    colors = ['red', 'orange', 'green', 'cyan', 'magenta']

    for q_val, color in zip(rational_q, colors):
        ax.axhline(y=q_val, color=color, linestyle='--', alpha=0.7,
                   label=f'q = {q_val}')

        # Find radial location
        idx = np.argmin(np.abs(q - q_val))
        if idx > 0 and idx < len(r)-1:
            r_res = r[idx] / a
            ax.plot(r_res, q_val, 'o', color=color, markersize=8)

    ax.set_xlabel('r/a (normalized radius)', fontsize=12)
    ax.set_ylabel('Safety factor q', fontsize=12)
    ax.set_title('Safety Factor Profile and Rational Surfaces', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim([0, max(q)*1.1])

    plt.tight_layout()
    return fig

def example_tokamak_q_profile():
    """
    Compute and analyze q-profile for a tokamak
    """
    # Tokamak parameters
    R0 = 3.0  # Major radius [m]
    a = 1.0   # Minor radius [m]
    Bt0 = 5.0 # Toroidal field [T]
    Ip = 2e6  # Plasma current [A]

    mu0 = 4*np.pi*1e-7

    # Radial grid
    r = np.linspace(0.01, a, 200)

    # Current density profile (parabolic)
    alpha = 2.0
    j0 = Ip / (np.pi * a**2 * (1 - 1/(alpha+1)))

    def j_profile(r_val):
        return j0 * (1 - (r_val/a)**alpha)

    # Enclosed current
    def I_enclosed(r_val):
        # Integrate j(r') from 0 to r
        from scipy.integrate import quad
        result, _ = quad(lambda rp: j_profile(rp) * 2*np.pi*rp, 0, r_val)
        return result

    # Poloidal field
    I_enc = np.array([I_enclosed(ri) for ri in r])
    Btheta = mu0 * I_enc / (2*np.pi*r)

    # Toroidal field (1/R dependence)
    Bz = Bt0 * R0 / (R0 + r)

    # Safety factor
    q = compute_q_profile(r, Bz, Btheta, R0)

    print("=== Tokamak q-Profile Analysis ===")
    print(f"Major radius R0 = {R0} m")
    print(f"Minor radius a = {a} m")
    print(f"Aspect ratio A = {R0/a}")
    print(f"Toroidal field Bt0 = {Bt0} T")
    print(f"Plasma current Ip = {Ip/1e6} MA")

    print(f"\nq(0) ≈ {q[0]:.2f}")
    print(f"q(a) = {q[-1]:.2f}")

    # Check Kruskal-Shafranov for (1,1) mode
    stable_11, margin_11, q_crit_11 = check_kruskal_shafranov(q[-1], 1, 1)

    print(f"\nKruskal-Shafranov (m=1, n=1):")
    print(f"  Critical q: {q_crit_11}")
    print(f"  Edge q: {q[-1]:.2f}")
    print(f"  Margin: {margin_11:.2f}")
    print(f"  Status: {'STABLE' if stable_11 else 'UNSTABLE'}")

    # Check for other modes
    modes = [(1, 1), (2, 1), (3, 1), (3, 2)]
    print("\nStability check for resonant modes:")
    for m, n in modes:
        stable, margin, q_crit = check_kruskal_shafranov(q[-1], m, n)
        status = "✓ STABLE" if stable else "✗ UNSTABLE"
        print(f"  (m={m}, n={n}): q_crit={q_crit:.2f}, margin={margin:+.2f} → {status}")

    # Find rational surfaces
    print("\nRational surface locations (r/a):")
    for q_rational in [1, 2, 3]:
        idx = np.argmin(np.abs(q - q_rational))
        if q[idx] > 0.8*q_rational and q[idx] < 1.2*q_rational:
            print(f"  q = {q_rational}: r/a ≈ {r[idx]/a:.3f}")
        else:
            print(f"  q = {q_rational}: not found in plasma")

    # Plot
    fig = plot_q_and_stability(r, q, R0, a)
    plt.savefig('/tmp/tokamak_q_profile.png', dpi=150)
    print("\nq-profile plot saved to /tmp/tokamak_q_profile.png")
    plt.close()

if __name__ == "__main__":
    example_tokamak_q_profile()
```

### 7.3 에너지 원리 계산

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

def compute_delta_W(r, xi_r, xi_theta, Bz, Btheta, p, rho, m, kz, gamma_adiabatic=5/3):
    """
    Compute potential energy δW for a given displacement

    Parameters:
    -----------
    r: radial grid
    xi_r, xi_theta: displacement components
    Bz, Btheta: equilibrium fields
    p: pressure
    rho: density
    m: poloidal mode number
    kz: axial wavenumber
    gamma_adiabatic: adiabatic index

    Returns:
    --------
    delta_W: potential energy
    """
    mu0 = 4*np.pi*1e-7

    # Perturbed magnetic field (simplified)
    # B1_r ~ -ikz * xi_r * Bz + (im/r) * xi_theta * Bz
    # This is a simplified model; full calculation is complex

    # div_xi는 유체가 얼마나 압축되는지 측정합니다: 0이 아닌 발산은
    # 자기 압축과 음향 압축 모두를 통해 에너지 비용을 지불하므로,
    # 가장 위험한(불안정) 섭동은 비압축(div_xi → 0)인 경향이 있습니다 —
    # 이러한 안정화 항을 피하기 때문입니다.
    dxi_r_dr = np.gradient(xi_r, r)
    div_xi = dxi_r_dr + xi_r/r + (1j*m/r)*xi_theta

    # kz*Bz와 (m/r)*Bz 항은 자기장선 굽힘을 포착합니다: kz 방향 또는
    # m번째 방위각 고조파 주위로 플라즈마를 변위시키면 자기장선이 늘어나고
    # 구부러져 양의 일을 합니다(B1_perp_sq > 0), 이는 항상 안정화입니다 —
    # 이것이 B_z가 Z-pinch를 더 안정하게 만드는 이유입니다.
    B1_perp_sq = np.abs((kz*Bz)**2 * xi_r**2 + (m*Bz/r)**2 * xi_theta**2)

    delta_W_magnetic = 0.5 * simps(B1_perp_sq / mu0 * 2*np.pi*r, r)

    # 단열 압축 δW_p = (γp/2)|div_ξ|²은 음향 복원력을 적분합니다:
    # 플라즈마를 단열적으로 압축하면 압력이 높아져 추가 압축에 저항합니다.
    # 이 항은 항상 양수(안정화)입니다.
    delta_W_pressure = 0.5 * simps(gamma_adiabatic * p * np.abs(div_xi)**2 * 2*np.pi*r, r)

    # 압력 구배 구동 ξ_r (dp/dr) div_ξ는 dp/dr < 0 (바깥쪽으로 감소하는 압력)이고
    # 변위가 바깥쪽이면 (ξ_r > 0이고 div_ξ > 0) 음수일 수 있습니다:
    # 이것이 낮은 압력의 플라즈마가 바깥쪽으로 흘러 자유 에너지를 방출하는
    # 교환/ballooning 메커니즘입니다.
    dp_dr = np.gradient(p, r)
    delta_W_drive = simps(xi_r * dp_dr * div_xi.real * 2*np.pi*r, r)

    delta_W = delta_W_magnetic + delta_W_pressure + delta_W_drive

    return delta_W

def example_energy_principle():
    """
    Use energy principle to assess stability
    """
    # Set up equilibrium (similar to previous examples)
    a = 0.1
    r = np.linspace(0.01, a, 100)

    # Fields
    Bz0 = 1.0
    Btheta0 = 0.5

    Bz = Bz0 * np.ones_like(r)
    Btheta = Btheta0 * r / a

    # Pressure
    p0 = 1e5
    p = p0 * (1 - (r/a)**2)**2

    # Density
    rho = 1e-3 * np.ones_like(r)

    # Test displacement (trial function)
    # Try a simple form: xi_r ~ sin(πr/a)
    xi_r = np.sin(np.pi * r / a)
    xi_theta = np.zeros_like(r)

    # Mode numbers
    m = 1
    kz = 10  # [1/m]

    # Compute δW
    delta_W = compute_delta_W(r, xi_r, xi_theta, Bz, Btheta, p, rho, m, kz)

    print("=== Energy Principle Stability Test ===")
    print(f"Mode: m={m}, kz={kz} m^-1")
    print(f"Trial displacement: ξ_r ~ sin(πr/a)")
    print(f"\nPotential energy: δW = {delta_W:.3e} J")

    if delta_W > 0:
        print("Result: STABLE (δW > 0)")
    elif delta_W < 0:
        print("Result: UNSTABLE (δW < 0)")
    else:
        print("Result: MARGINAL (δW = 0)")

    # Try multiple trial functions
    print("\n=== Testing Multiple Trial Functions ===")

    trial_functions = {
        'sin(πr/a)': lambda r: np.sin(np.pi*r/a),
        'sin(2πr/a)': lambda r: np.sin(2*np.pi*r/a),
        '(r/a)(1-r/a)': lambda r: (r/a)*(1-r/a),
        '(1-(r/a)²)': lambda r: 1-(r/a)**2,
    }

    results = []

    for name, func in trial_functions.items():
        xi_r_trial = func(r)
        xi_theta_trial = np.zeros_like(r)

        dW = compute_delta_W(r, xi_r_trial, xi_theta_trial, Bz, Btheta, p, rho, m, kz)

        results.append((name, dW))
        status = "STABLE" if dW > 0 else "UNSTABLE"
        print(f"  {name:20s}: δW = {dW:+.3e} J → {status}")

    # Find most dangerous (minimum δW)
    min_idx = np.argmin([dW for _, dW in results])
    min_name, min_dW = results[min_idx]

    print(f"\nMost dangerous trial function: {min_name}")
    print(f"Minimum δW: {min_dW:.3e} J")

    if min_dW < 0:
        print("⚠ UNSTABLE configuration detected!")
    else:
        print("✓ Stable for all tested trial functions")

if __name__ == "__main__":
    example_energy_principle()
```

## 8. 고급 주제

### 8.1 저항성 불안정성

**저항성 MHD**에서, 유한 저항성은 자기 재결합을 허용합니다. 이상 MHD 안정성 분석을 수정해야 합니다.

**Tearing 모드**: $q = m/n$인 유리수 표면에서의 재결합이 자기 섬 형성으로 이어집니다 (강의 4에서 다룸).

### 8.2 운동학적 효과

MHD는 유체 근사를 가정합니다. 입자 자이로반지름과 비교 가능한 파장에 대해, 운동학적 효과 (Landau 감쇠, 파동-입자 공명)가 안정성을 수정합니다.

### 8.3 전도 벽 안정화

플라즈마 근처에 전도 벽을 배치하면 영상 전류를 유도하여 외부 kink 모드를 안정화할 수 있습니다.

**벽 있음**: 모드는 저항성 벽 시간척도에서 성장 (느림)
**벽 없음**: 모드는 Alfvén 시간척도에서 성장 (빠름)

## 요약

이 강의에서 MHD 선형 안정성 이론을 개발했습니다:

1. **선형화**: 섭동 분석이 자기수반 힘 연산자를 가진 고유값 문제 $\mathbf{F}(\hat{\boldsymbol{\xi}}) = -\omega^2\rho_0\hat{\boldsymbol{\xi}}$로 이어집니다.

2. **에너지 원리**: 고유값 문제를 풀지 않고 위치 에너지 $\delta W$의 부호로 안정성 결정. 자기 압축, 장력, 압력 구동으로 분해.

3. **Kruskal-Shafranov 기준**: 외부 kink 안정성은 $q(a) > m/n$ 필요. 토카막에 대해 $q(a) > 1$이 필요.

4. **Suydam 기준**: 국소 interchange 안정성은 압력 구배 구동을 극복하기 위한 충분한 자기 전단 필요.

5. **성장률**: 수치 고유값 솔버가 성장률과 모드 구조를 계산.

6. **수치 구현**: 힘 연산자 이산화, 고유값 솔버, 에너지 원리 계산.

이 도구들은 핵융합 플라즈마에서 MHD 불안정성을 이해하고 예측하는 기초를 형성하며, 운영 영역을 제한하고 제어 전략을 동기 부여합니다.

## 연습 문제

### 문제 1: Sausage 모드에 대한 에너지 원리

$r=a$에서 날카로운 경계를 가진 균일 전류 밀도의 Z-pinch를 고려하세요. 평형은:
- $B_θ(r) = \frac{\mu_0 I r}{2\pi a^2}$ for $r < a$
- $B_θ(r) = \frac{\mu_0 I}{2\pi r}$ for $r > a$
- $p(r) = p_0$ (일정) for $r < a$

시험 변위 $\xi_r = \xi_0\sin(\pi r/a)$, $\xi_θ = \xi_z = 0$를 가진 $m=0$ (sausage) 모드에 대해:

**(a)** $\nabla\cdot\boldsymbol{\xi}$를 계산하세요.

**(b)** 섭동된 자기장 $\mathbf{B}_1$을 추정하세요.

**(c)** 자기 압축 에너지 $\delta W_{mag} = \frac{1}{2\mu_0}\int |\mathbf{B}_1|^2 dV$를 계산하세요.

**(d)** 압력 압축 에너지 $\delta W_p = \frac{\gamma}{2}\int p_0 |\nabla\cdot\boldsymbol{\xi}|^2 dV$를 계산하세요.

**(e)** $\delta W > 0$ (안정) 또는 $\delta W < 0$ (불안정)인지 결정하세요.

### 문제 2: 토카막에 대한 Kruskal-Shafranov

토카막이:
- 주요 반지름 $R_0 = 2$ m
- 소반지름 $a = 0.5$ m
- 토로이달 장 $B_t = 4$ T (일정)
- 전류 밀도 $J_z(r) = J_0(1 - r^2/a^2)$

**(a)** 총 플라즈마 전류 $I_p = \int J_z(r) 2\pi r\, dr$를 구하세요.

**(b)** 폴로이달 장 $B_θ(r) = \mu_0 I(r)/(2\pi r)$ (여기서 $I(r)$은 둘러싸인 전류)를 계산하세요.

**(c)** 안전 인자 $q(r) = rB_t/(R_0 B_θ(r))$를 계산하세요.

**(d)** $q(0)$ (축 상, l'Hôpital 규칙 사용)과 $q(a)$ (가장자리)를 평가하세요.

**(e)** $(m,n) = (1,1)$ 모드에 대해 Kruskal-Shafranov 기준을 확인하세요. 구성이 안정합니까?

**(f)** $q(a) = 3$을 달성하기 위해 필요한 최소 가장자리 전류는 무엇입니까?

### 문제 3: Suydam 기준 적용

Screw pinch가:
- $B_z = 1$ T (일정)
- $B_θ(r) = B_{θ0}(r/a)$ (선형)
- $p(r) = p_0(1 - r^2/a^2)$
- 주요 반지름 $R_0 = 10a$

**(a)** 안전 인자 $q(r)$과 그 도함수 $q'(r)$을 계산하세요.

**(b)** 압력 구배 $p'(r)$을 계산하세요.

**(c)** $r = a/2$에서 Suydam 기준을 평가하세요:
$$
\frac{r}{4}\left(\frac{q'}{q}\right)^2 + \frac{2\mu_0 p'}{B_z^2}
$$

**(d)** 모든 반지름에서 Suydam 안정성에 대해 허용되는 최대 $p_0$를 결정하세요.

**(e)** $p_0$가 이 한계를 초과하면 무슨 일이 일어납니까?

### 문제 4: 성장률 추정

$\delta W < 0$인 불안정 모드에 대해, 성장률은 다음으로 추정할 수 있습니다:

$$
\gamma^2 \sim \frac{|\delta W|}{K}
$$

여기서 $K = \frac{1}{2}\int\rho_0|\boldsymbol{\xi}|^2 dV$는 섭동의 운동 에너지입니다.

반지름 $a = 0.1$ m, 길이 $L = 1$ m, 밀도 $\rho_0 = 10^{-6}$ kg/m³인 원통 플라즈마를 고려하세요.

변위는 $\boldsymbol{\xi} = \xi_0\sin(\pi r/a)\hat{\mathbf{r}}$, $\xi_0 = 0.01$ m입니다.

에너지 원리 계산으로부터: $\delta W = -10^3$ J.

**(a)** 운동 에너지 $K$를 계산하세요.

**(b)** 성장률 $\gamma$를 추정하세요.

**(c)** 성장 시간 $\tau = 1/\gamma$를 계산하세요.

**(d)** Alfvén 속도가 $v_A = 10^6$ m/s이면, $\gamma$를 Alfvén 주파수 $\omega_A = v_A/a$와 비교하세요.

**(e)** 이것은 빠른 (Alfvén 시간척도) 또는 느린 불안정성입니까?

### 문제 5: 고유값 문제 설정

다음을 가진 원통 플라즈마에 대한 고유값 문제를 설정하세요 (풀지 마세요):
- $B_z = B_0 = \text{const}$
- $B_θ(r) = 0$
- $p(r) = p_0(1-r^2/a^2)$
- $\rho = \rho_0 = \text{const}$

$m=1$ 섭동 $\boldsymbol{\xi} = \xi_r(r)e^{i(\theta - kz z)}\hat{\mathbf{r}} + \xi_θ(r)e^{i(\theta - kz z)}\hat{\boldsymbol{\theta}}$에 대해:

**(a)** 선형화된 운동량 방정식을 성분 형태로 쓰세요.

**(b)** $\mathbf{B}_1$을 $\xi_r, \xi_θ$로 표현하세요.

**(c)** $\xi_r(r)$과 $\xi_θ(r)$에 대한 결합 ODE를 유도하세요.

**(d)** $r=0$과 $r=a$에서 경계 조건은 무엇입니까?

**(e)** 수치 해를 위해 이 시스템을 어떻게 이산화하겠습니까?

---

**이전**: [MHD Equilibria](./01_MHD_Equilibria.md) | **다음**: [Pressure-Driven Instabilities](./03_Pressure_Driven_Instabilities.md)
