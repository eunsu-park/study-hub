# 1. 플라즈마 소개

## 학습 목표

- 플라즈마를 물질의 네 번째 상태로 이해하고 자연 및 실험실 예시 식별하기
- Debye 차폐 개념을 유도하고 다양한 플라즈마에 대해 Debye 길이를 계산하기
- 집단적 행동 및 준중성 조건에 대한 플라즈마 기준 설명하기
- 특성 플라즈마 주파수(플라즈마 주파수 및 gyrofrequency)를 제1원리로부터 계산하기
- 플라즈마 베타 매개변수를 계산하고 열압력 대 자기압력의 상대적 중요성 해석하기
- Python 도구를 적용하여 다양한 물리 시스템에 걸친 플라즈마 매개변수 계산 및 비교하기

## 1. 물질의 네 번째 상태

### 1.1 플라즈마란?

플라즈마는 고체, 액체, 기체에 이어 **물질의 네 번째 상태**라고 불립니다. 장거리 전자기 상호작용으로 인해 집단적 행동을 나타내는 자유 전하 입자(전자와 이온)의 집합으로 구성됩니다.

```
State Transitions:

Solid → Liquid → Gas → Plasma
  ↑         ↑        ↑       ↑
 Heat     Heat    Heat   Ionization
            (add energy →)

Key characteristics:
- Solid:  molecules in fixed lattice, short-range forces
- Liquid: molecules mobile, short-range forces
- Gas:    molecules independent, rare collisions
- Plasma: ions + electrons, long-range EM forces, collective behavior
```

기체에서 플라즈마로의 전이는 **이온화**를 통해 발생합니다. 이는 중성 원자나 분자가 다음과 같은 이유로 전자를 잃는 과정입니다:
- 열에너지(고온)
- 전자기 복사(광이온화)
- 에너지가 높은 입자와의 충돌
- 전기장(avalanche breakdown)

### 1.2 이온화 및 이온화도

**이온화도** $\alpha$는 다음과 같이 정의됩니다:

$$\alpha = \frac{n_i}{n_i + n_n}$$

여기서 $n_i$는 이온 밀도이고 $n_n$은 중성 입자 밀도입니다.

- **완전 이온화 플라즈마**: $\alpha \approx 1$ (거의 모든 입자가 전하를 띰)
- **부분 이온화 플라즈마**: $0 < \alpha < 1$ (전하를 띤 입자와 중성 입자의 혼합)
- **약이온화 플라즈마**: $\alpha \ll 1$ (대부분의 입자가 중성 상태이지만 집단적 효과가 지배적)

플라즈마 기준(아래에서 논의)이 충족되면 약이온화 플라즈마($\alpha \sim 10^{-6}$인 경우도 있음)도 플라즈마 행동을 나타낼 수 있습니다.

### 1.3 플라즈마의 예

플라즈마는 우주에서 가장 일반적인 물질 상태로, 가시 물질의 99% 이상을 차지합니다.

**자연 플라즈마:**

| System | Temperature (eV) | Density (m⁻³) | Magnetic Field (T) |
|--------|------------------|---------------|-------------------|
| Interstellar medium | 1 | 10⁶ | 10⁻¹⁰ |
| Solar wind (1 AU) | 10 | 10⁷ | 10⁻⁹ |
| Solar corona | 100 | 10¹⁴ | 10⁻² |
| Solar core | 1000 | 10³² | — |
| Lightning | 2-3 | 10²² | — |
| Ionosphere (F-layer) | 0.1 | 10¹² | 10⁻⁵ |

**실험실 플라즈마:**

| System | Temperature (eV) | Density (m⁻³) | Magnetic Field (T) |
|--------|------------------|---------------|-------------------|
| Tokamak core | 10,000 | 10²⁰ | 5 |
| Tokamak edge | 100 | 10¹⁹ | 5 |
| Fluorescent lamp | 1-2 | 10¹⁶ | — |
| Neon sign | 1-3 | 10¹⁴ | — |
| Arc discharge | 1-2 | 10²² | — |
| Plasma processing | 3-5 | 10¹⁶ | 10⁻² |

*참고: 1 eV ≈ 11,600 K. 플라즈마 온도는 관례상 전자볼트로 표현됩니다.*

## 2. Debye 차폐

### 2.1 문제: 장거리 Coulomb 힘

점전하 $q$로부터의 Coulomb 퍼텐셜은 다음과 같습니다:

$$\phi(r) = \frac{1}{4\pi\epsilon_0} \frac{q}{r}$$

이 $1/r$ 의존성은 힘이 **장거리**임을 의미합니다—무한대까지 확장됩니다. 플라즈마의 모든 입자가 차폐되지 않은 Coulomb 힘을 통해 다른 모든 입자와 상호작용한다면, 시스템은 다루기 어려울 것입니다.

그러나 플라즈마는 놀라운 집단 현상인 **Debye 차폐**를 나타냅니다.

### 2.2 Debye 길이의 유도

플라즈마에 시험 전하 $Q$를 삽입하는 경우를 고려합시다. 이동 가능한 전자와 이온이 재배열되어 이 전하를 차폐할 것입니다.

**가정:**
1. 플라즈마가 열평형 상태(Boltzmann 분포)
2. 정전기 퍼텐셜이 작음: $|e\phi| \ll k_B T$
3. 외부 장이 없고 자기장이 없음

전자와 이온 밀도는 Boltzmann 관계를 통해 퍼텐셜 $\phi$에 반응합니다:

$$n_e = n_0 \exp\left(\frac{e\phi}{k_B T_e}\right) \approx n_0 \left(1 + \frac{e\phi}{k_B T_e}\right)$$

$$n_i = n_0 \exp\left(-\frac{Ze\phi}{k_B T_i}\right) \approx n_0 \left(1 - \frac{Ze\phi}{k_B T_i}\right)$$

여기서 $n_0$는 배경 밀도, $Z$는 이온 전하 상태이고, 작은 $\phi$에 대해 선형화했습니다.

전하 밀도는 다음과 같습니다:

$$\rho = e(Zn_i - n_e) = -en_0\left(\frac{e}{k_B T_e} + \frac{Ze}{k_B T_i}\right)\phi$$

Poisson 방정식은 퍼텐셜과 전하 밀도를 연결합니다:

$$\nabla^2 \phi = -\frac{\rho}{\epsilon_0} = \frac{n_0 e}{\epsilon_0}\left(\frac{e}{k_B T_e} + \frac{Ze}{k_B T_i}\right)\phi$$

**Debye 길이** $\lambda_D$를 정의합니다:

$$\frac{1}{\lambda_D^2} = \frac{n_0 e^2}{\epsilon_0 k_B T_e} + \frac{Zn_0 e^2}{\epsilon_0 k_B T_i} = \frac{n_0 e^2}{\epsilon_0 k_B}\left(\frac{1}{T_e} + \frac{Z}{T_i}\right)$$

**단일 종** 경우(또는 $T_e \approx T_i = T$이고 $Z=1$일 때):

$$\lambda_D = \sqrt{\frac{\epsilon_0 k_B T}{n_0 e^2}}$$

**전자만**의 경우(이동성이 높아 주로 지배적):

$$\lambda_{De} = \sqrt{\frac{\epsilon_0 k_B T_e}{n_e e^2}} \approx 7.43 \times 10^3 \sqrt{\frac{T_e[\text{eV}]}{n_e[\text{m}^{-3}]}} \quad [\text{m}]$$

**물리적 해석:** Debye 길이는 이동 전하 운반체에 의해 전하 불균형이 차폐되는 거리입니다.

Poisson 방정식은 다음과 같이 됩니다:

$$\nabla^2 \phi = \frac{\phi}{\lambda_D^2}$$

구형 대칭의 경우(원점의 시험 전하):

$$\frac{1}{r^2}\frac{d}{dr}\left(r^2 \frac{d\phi}{dr}\right) = \frac{\phi}{\lambda_D^2}$$

해:

$$\phi(r) = \frac{Q}{4\pi\epsilon_0 r} e^{-r/\lambda_D}$$

이것이 **Debye-Hückel 퍼텐셜**입니다—지수 인자로 차폐된 Coulomb 퍼텐셜입니다.

```
Shielded Potential:

φ(r) ∝ (1/r) × exp(-r/λ_D)

         |
    Q    |    ___
  -----  |   /   \___
    |    |  /        \____
    |    | /              \______
----+----+-------------------------- r
    0   λ_D

Unshielded: 1/r (dashed)
Shielded:   (1/r)exp(-r/λ_D) (solid)

For r ≪ λ_D: full Coulomb force
For r ≫ λ_D: exponentially suppressed
```

### 2.3 플라즈마 매개변수

Debye 구 내의 입자 수는 다음과 같습니다:

$$N_D = n \cdot \frac{4}{3}\pi\lambda_D^3 = n\lambda_D^3 \cdot \frac{4\pi}{3}$$

시스템이 **집단적 플라즈마 행동**을 나타내려면 다음이 필요합니다:

$$n\lambda_D^3 \gg 1$$

**해석:** 집단 차폐가 개별 입자 상호작용을 지배하려면 Debye 구 내에 많은 입자가 있어야 합니다.

$n\lambda_D^3 \ll 1$일 때, 시스템은 이진 충돌이 지배하는 **약결합 기체**처럼 행동합니다(진정한 플라즈마가 아님).

**플라즈마 결합 매개변수:**

$$\Gamma = \frac{\text{potential energy}}{\text{kinetic energy}} = \frac{e^2}{4\pi\epsilon_0 a k_B T}$$

여기서 $a = (3/4\pi n)^{1/3}$은 평균 입자 간 거리입니다.

- **약결합 플라즈마**: $\Gamma \ll 1$ (핵융합 장치를 포함한 대부분의 플라즈마)
- **강결합 플라즈마**: $\Gamma \gtrsim 1$ (예: 백색왜성 내부, 먼지 플라즈마)

약결합 플라즈마의 경우:

$$\Gamma \sim \frac{1}{N_D^{1/3}}$$

따라서 $n\lambda_D^3 \gg 1$은 $\Gamma \ll 1$과 동등합니다.

## 3. 준중성

### 3.1 전하 중성 조건

Debye 길이보다 훨씬 큰 길이 스케일($L \gg \lambda_D$)에서 플라즈마는 **준중성**입니다:

$$n_e \approx Z_i n_i$$

**왜?** 전하 불균형은 큰 전기장을 생성합니다:

$$E \sim \frac{e(n_i - n_e)L}{\epsilon_0}$$

이러한 장들은 입자를 가속시켜 플라즈마 주기 $\omega_{pe}^{-1}$(아래에 정의됨) 정도의 시간 스케일로 중성을 회복합니다.

### 3.2 플라즈마 근사

대부분의 플라즈마 계산에서 다음을 가정할 수 있습니다:
1. **준중성**: $L \gg \lambda_D$에 대해 $n_e = Zn_i$
2. **집단적 행동**: $n\lambda_D^3 \gg 1$

이 두 조건이 **플라즈마 영역**을 정의합니다:

```
Plasma Criteria:

    n λ_D³ ≫ 1         (many particles in Debye sphere)
    L ≫ λ_D            (system size ≫ Debye length)

Together these imply:
    - Long-range collective interactions
    - Quasi-neutrality
    - Plasma oscillations and waves
```

## 4. 플라즈마 주파수

### 4.1 전자 플라즈마 진동

이동하지 않는 이온이 있는 차가운 균일한 플라즈마($T=0$)를 고려합시다. 모든 전자를 이온에 대해 작은 거리 $\delta x$만큼 변위시킵니다.

**전하 분리는 전기장을 생성합니다:**

변위된 전자 슬래브에 대해 Gauss 법칙을 사용하면:

$$E = \frac{n_e e \delta x}{\epsilon_0}$$

**전자의 운동 방정식:**

$$m_e \frac{d^2(\delta x)}{dt^2} = -eE = -\frac{n_e e^2}{\epsilon_0} \delta x$$

이것은 각주파수가 다음과 같은 단순 조화 운동입니다:

$$\omega_{pe}^2 = \frac{n_e e^2}{\epsilon_0 m_e}$$

**전자 플라즈마 주파수**를 정의합니다:

$$\omega_{pe} = \sqrt{\frac{n_e e^2}{\epsilon_0 m_e}} \approx 5.64 \times 10^4 \sqrt{n_e[\text{m}^{-3}]} \quad [\text{rad/s}]$$

**물리적 해석:** 이온 배경에 대한 전자의 자연 진동 주파수입니다. 이것은 대부분의 플라즈마에서 가장 빠른 시간 스케일입니다.

### 4.2 이온 플라즈마 주파수

마찬가지로 이온의 경우:

$$\omega_{pi} = \sqrt{\frac{n_i Z^2 e^2}{\epsilon_0 m_i}}$$

$m_i \gg m_e$(수소의 경우 일반적으로 $m_i/m_e \sim 1836$)이므로:

$$\frac{\omega_{pi}}{\omega_{pe}} = \sqrt{\frac{m_e}{m_i}} \approx \frac{1}{43} \quad \text{(양성자의 경우)}$$

이온 진동은 전자 진동보다 훨씬 느립니다.

### 4.3 플라즈마 주기

**플라즈마 주기**는 다음과 같습니다:

$$\tau_{pe} = \frac{2\pi}{\omega_{pe}} \approx 1.11 \times 10^{-4} \frac{1}{\sqrt{n_e[\text{m}^{-3}]}} \quad [\text{s}]$$

일반적인 핵융합 플라즈마($n_e \sim 10^{20}$ m$^{-3}$)의 경우:

$$\tau_{pe} \sim 10^{-14} \text{ s} = 10 \text{ fs}$$

이것이 정전기 현상의 기본 시간 스케일을 설정합니다.

## 5. Gyrofrequency 및 Larmor 반경

### 5.1 Cyclotron 운동

자기장 $\mathbf{B}$에서 전하 입자는 Lorentz 힘을 받습니다:

$$\mathbf{F} = q\mathbf{v} \times \mathbf{B}$$

$\mathbf{B}$에 수직인 운동의 경우, 힘은 속도에 수직이어서 원운동을 유발합니다.

**Gyrofrequency**(또는 cyclotron 주파수)는 다음과 같습니다:

$$\omega_c = \frac{|q|B}{m}$$

**전자의 경우:**

$$\omega_{ce} = \frac{eB}{m_e} \approx 1.76 \times 10^{11} B[\text{T}] \quad [\text{rad/s}]$$

**이온의 경우:**

$$\omega_{ci} = \frac{ZeB}{m_i}$$

양성자의 경우($Z=1$, $m_i = 1836 m_e$):

$$\omega_{ci} \approx 9.58 \times 10^7 B[\text{T}] \quad [\text{rad/s}]$$

참고: $\omega_{ci}/\omega_{ce} = m_e/m_i \ll 1$ — 이온은 전자보다 훨씬 느리게 회전합니다.

### 5.2 Larmor 반경

원형 궤도의 반경(gyroradius 또는 Larmor 반경)은 구심 가속도를 평형시켜 구합니다:

$$m\frac{v_\perp^2}{r_L} = qv_\perp B$$

$$r_L = \frac{mv_\perp}{qB} = \frac{v_\perp}{\omega_c}$$

**열적 Larmor 반경:** 열속도 $v_{th} = \sqrt{k_B T/m}$를 사용하면:

$$r_{L,thermal} = \frac{v_{th}}{\omega_c} = \frac{\sqrt{k_B T m}}{qB}$$

**전자의 경우:**

$$r_{Le} = \frac{\sqrt{2k_B T_e m_e}}{eB} \approx 2.28 \times 10^{-6} \frac{\sqrt{T_e[\text{eV}]}}{B[\text{T}]} \quad [\text{m}]$$

**이온의 경우(양성자):**

$$r_{Li} = \frac{\sqrt{2k_B T_i m_p}}{eB} \approx 9.77 \times 10^{-5} \frac{\sqrt{T_i[\text{eV}]}}{B[\text{T}]} \quad [\text{m}]$$

Larmor 반경은 수직 입자 수송의 스케일을 설정하고 자기장 효과가 중요한 시기를 결정합니다.

## 6. 플라즈마 베타

### 6.1 정의

**플라즈마 베타**는 열압력 대 자기압력의 비율입니다:

$$\beta = \frac{p}{p_B} = \frac{2\mu_0 nk_B T}{B^2}$$

여기서:
- 열압력: $p = nk_B T$ (종에 대해 합산)
- 자기압력: $p_B = B^2/(2\mu_0)$

전자와 이온이 있는 플라즈마의 경우:

$$\beta = \frac{2\mu_0 (n_e k_B T_e + n_i k_B T_i)}{B^2}$$

준중성($n_e = n_i = n$)과 $T_e \approx T_i = T$를 가정하면:

$$\beta = \frac{4\mu_0 n k_B T}{B^2}$$

### 6.2 물리적 해석

- **$\beta \ll 1$**: 자기압력이 지배(자기화된 플라즈마)
  - 입자가 자기력선에 단단히 결합됨
  - 자기력선이 대략 강체
  - 예: tokamak 중심부, 태양 코로나, 자기권

- **$\beta \sim 1$**: 열압력과 자기압력이 비슷함
  - 자기력선이 플라즈마 압력에 의해 왜곡될 수 있음
  - 예: tokamak 가장자리, reversed-field pinch

- **$\beta \gg 1$**: 열압력이 지배(자기화되지 않은 플라즈마)
  - 자기장이 역학에 거의 영향을 주지 않음
  - 예: 초기 우주, 일부 레이저 플라즈마

### 6.3 임계 베타 및 MHD 안정성

자기 제약 핵융합에서는 플라즈마가 MHD 불안정해지는 **임계 베타** $\beta_c$가 있습니다. Tokamak의 경우:

$$\beta_c \sim \frac{I_p}{aB_0}$$

여기서 $I_p$는 플라즈마 전류, $a$는 소반경, $B_0$는 토로이달 장입니다. 일반적인 값: $\beta_c \sim 0.02 - 0.05$ (2-5%).

고$\beta$ 플라즈마는 핵융합에 바람직하지만(높은 압력 → 높은 핵융합 출력) 안정성 한계가 존재합니다.

## 7. 특성 스케일 및 순서

### 7.1 플라즈마 매개변수 요약

밀도 $n$, 온도 $T$, 자기장 $B$로 특성화된 플라즈마의 경우:

| Parameter | Symbol | Formula | Units |
|-----------|--------|---------|-------|
| Debye length | $\lambda_D$ | $\sqrt{\epsilon_0 k_B T/(n e^2)}$ | m |
| Plasma frequency | $\omega_{pe}$ | $\sqrt{n e^2/(\epsilon_0 m_e)}$ | rad/s |
| Electron gyrofrequency | $\omega_{ce}$ | $eB/m_e$ | rad/s |
| Ion gyrofrequency | $\omega_{ci}$ | $eB/m_i$ | rad/s |
| Electron Larmor radius | $r_{Le}$ | $v_{te}/\omega_{ce}$ | m |
| Ion Larmor radius | $r_{Li}$ | $v_{ti}/\omega_{ci}$ | m |
| Plasma beta | $\beta$ | $2\mu_0 nk_B T/B^2$ | — |

여기서 $v_{te} = \sqrt{k_B T_e/m_e}$와 $v_{ti} = \sqrt{k_B T_i/m_i}$는 열속도입니다.

### 7.2 일반적인 순서

자기화된 플라즈마의 경우:

**길이 스케일:**
$$r_{Le} \ll r_{Li} \ll L_\parallel \sim L_\perp$$

여기서 $L_\parallel$과 $L_\perp$는 특성 평행 및 수직 스케일 길이입니다.

**주파수 순서:**
$$\omega_{ci} \ll \omega_{ce} \ll \omega_{pe}$$

**시간 스케일:**
$$\tau_{pe} \ll \tau_{ce} \ll \tau_{ci}$$

이러한 순서는 플라즈마 모델을 단순화하기 위해 **점근 전개**에서 활용됩니다:
- **Drift-kinetic 이론**: 빠른 gyro-운동에 대해 평균화
- **Gyrokinetic 이론**: gyro-위상에 대해 평균화
- **MHD**: 시간 스케일 $\gg \tau_{ce}$, 길이 스케일 $\gg r_{Li}$ 가정

## 8. 계산 도구

### 8.1 플라즈마 매개변수 계산기

다음은 모든 기본 플라즈마 매개변수를 계산하는 Python 함수입니다:

```python
import numpy as np
import matplotlib.pyplot as plt

# Physical constants
e = 1.602176634e-19      # Elementary charge [C]
m_e = 9.1093837015e-31   # Electron mass [kg]
m_p = 1.672621898e-27    # Proton mass [kg]
epsilon_0 = 8.8541878128e-12  # Permittivity [F/m]
k_B = 1.380649e-23       # Boltzmann constant [J/K]
mu_0 = 4e-7 * np.pi      # Permeability [H/m]
eV_to_K = 11604.518      # Conversion factor eV to K

class PlasmaParameters:
    """Compute and store fundamental plasma parameters."""

    def __init__(self, n_e, T_e, B=0, T_i=None, Z=1, A=1):
        """
        Parameters:
        -----------
        n_e : float
            Electron density [m^-3]
        T_e : float
            Electron temperature [eV]
        B : float, optional
            Magnetic field strength [T]
        T_i : float, optional
            Ion temperature [eV]. If None, assumes T_i = T_e
        Z : int, optional
            Ion charge state (default: 1)
        A : int, optional
            Ion mass number (default: 1 for hydrogen)
        """
        self.n_e = n_e
        self.T_e = T_e
        self.B = B
        self.T_i = T_i if T_i is not None else T_e
        self.Z = Z
        self.A = A
        self.m_i = A * m_p

        # Compute all parameters
        self._compute_parameters()

    def _compute_parameters(self):
        """Compute all plasma parameters."""

        # Debye length [m]
        # T_e는 eV 단위로 주어지므로 eV_to_K를 곱해 켈빈으로, 다시 k_B를 곱해
        # 줄 단위로 변환한다. 이렇게 하면 단위 변환이 클래스 전체에 분산되지 않는다.
        self.lambda_D = np.sqrt(epsilon_0 * k_B * self.T_e * eV_to_K /
                                (self.n_e * e**2))

        # Plasma parameter (number of particles in Debye sphere)
        # N_D >> 1은 집단적 플라즈마 행동의 핵심 기준이다;
        # lambda_D 계산 직후 여기에 함께 두면 차폐 스케일을 구한 즉시
        # 플라즈마 기준을 확인하기가 매우 쉬워진다.
        self.N_D = self.n_e * (4*np.pi/3) * self.lambda_D**3

        # Electron plasma frequency [rad/s]
        # omega_pe는 밀도에만 의존하고 온도에는 의존하지 않는다.
        # 이온 배경의 복원력에서 비롯되며, 열압력에서 비롯되지 않기 때문이다.
        self.omega_pe = np.sqrt(self.n_e * e**2 / (epsilon_0 * m_e))
        self.f_pe = self.omega_pe / (2*np.pi)  # [Hz]

        # Ion plasma frequency [rad/s]
        # omega_pi << omega_pe (sqrt(m_e/m_i) 만큼 작다); 둘 다 유지하면
        # 많은 유체 근사의 토대가 되는 전자/이온 시간스케일 분리를 쉽게 검증할 수 있다.
        self.omega_pi = np.sqrt(self.n_e * self.Z**2 * e**2 /
                                (epsilon_0 * self.m_i))
        self.f_pi = self.omega_pi / (2*np.pi)  # [Hz]

        # Thermal velocities [m/s]
        # lambda_D와 동일한 eV_to_K * k_B 패턴을 사용해 단위 변환을 일관되게 유지한다;
        # v_te와 v_ti는 이후 모든 drift의 속도 스케일을 결정한다.
        self.v_te = np.sqrt(k_B * self.T_e * eV_to_K / m_e)
        self.v_ti = np.sqrt(k_B * self.T_i * eV_to_K / self.m_i)

        if self.B > 0:
            # Gyrofrequency와 Larmor 반경은 B > 0일 때만 의미가 있다;
            # 이 조건으로 보호하면 0으로 나누기를 방지하고,
            # B = 0일 때 자기화 효과가 적용되지 않음을 호출자에게 명확히 알린다.
            self.omega_ce = e * self.B / m_e
            self.f_ce = self.omega_ce / (2*np.pi)  # [Hz]

            # Ion gyrofrequency [rad/s]
            self.omega_ci = self.Z * e * self.B / self.m_i
            self.f_ci = self.omega_ci / (2*np.pi)  # [Hz]

            # Larmor radii [m]
            # r_Le = v_te / omega_ce를 사용하는 것은, 열속도가 이미 온도 의존성을
            # 깔끔하게 담고 있어 정확한 공식 mv/qB보다 간결하기 때문이다.
            self.r_Le = self.v_te / self.omega_ce
            self.r_Li = self.v_ti / self.omega_ci

            # Plasma beta
            # 열압력에 T_e + T_i를 합산해야 beta가 두 종을 모두 반영한다;
            # beta < 1이면 자기압력이 지배적이어서 플라즈마를 가두고 있다는 의미다.
            p_thermal = self.n_e * k_B * (self.T_e + self.T_i) * eV_to_K
            p_magnetic = self.B**2 / (2 * mu_0)
            self.beta = p_thermal / p_magnetic
        else:
            self.omega_ce = None
            self.omega_ci = None
            self.r_Le = None
            self.r_Li = None
            self.beta = None

    def print_summary(self):
        """Print a formatted summary of plasma parameters."""
        print("="*60)
        print("PLASMA PARAMETERS")
        print("="*60)
        print(f"Input Parameters:")
        print(f"  Electron density:     n_e = {self.n_e:.3e} m^-3")
        # 온도를 eV와 K 두 단위로 모두 출력해, 두 표기법을 사용하는 표와 직접 비교할 수 있게 한다
        # (플라즈마 문헌은 eV, 열역학 문헌은 K를 사용한다).
        print(f"  Electron temperature: T_e = {self.T_e:.3f} eV ({self.T_e*eV_to_K:.3e} K)")
        print(f"  Ion temperature:      T_i = {self.T_i:.3f} eV ({self.T_i*eV_to_K:.3e} K)")
        print(f"  Magnetic field:       B   = {self.B:.3f} T")
        print(f"  Ion charge/mass:      Z   = {self.Z}, A = {self.A}")
        print("-"*60)

        print(f"Debye Shielding:")
        print(f"  Debye length:         λ_D = {self.lambda_D:.3e} m")
        print(f"  Plasma parameter:     N_D = {self.N_D:.3e}")
        # 임계값으로 1이 아닌 100을 사용하는 것은 보수적인 선택이다: N_D >> 1이 이론적
        # 요건이지만, N_D ~ 100이어야 통계적 평균이 신뢰할 만하다.
        print(f"  Plasma criterion:     N_D >> 1? {self.N_D > 100}")
        print("-"*60)

        print(f"Plasma Frequencies:")
        print(f"  Electron plasma freq: ω_pe = {self.omega_pe:.3e} rad/s ({self.f_pe:.3e} Hz)")
        print(f"  Ion plasma freq:      ω_pi = {self.omega_pi:.3e} rad/s ({self.f_pi:.3e} Hz)")
        # 비율을 출력하면 전자가 이온보다 얼마나 빠르게 진동하는지 즉시 확인할 수 있고,
        # 플라즈마 이론 전반에서 활용되는 전자/이온 시간스케일 분리의 근거를 제공한다.
        print(f"  Ratio:                ω_pe/ω_pi = {self.omega_pe/self.omega_pi:.2f}")
        print("-"*60)

        print(f"Thermal Velocities:")
        print(f"  Electron thermal vel: v_te = {self.v_te:.3e} m/s")
        print(f"  Ion thermal vel:      v_ti = {self.v_ti:.3e} m/s")
        print(f"  Ratio:                v_te/v_ti = {self.v_te/self.v_ti:.2f}")
        print("-"*60)

        if self.B > 0:
            print(f"Magnetic Field Effects:")
            print(f"  Electron gyrofreq:    ω_ce = {self.omega_ce:.3e} rad/s ({self.f_ce:.3e} Hz)")
            print(f"  Ion gyrofreq:         ω_ci = {self.omega_ci:.3e} rad/s ({self.f_ci:.3e} Hz)")
            print(f"  Electron Larmor:      r_Le = {self.r_Le:.3e} m")
            print(f"  Ion Larmor:           r_Li = {self.r_Li:.3e} m")
            print(f"  Plasma beta:          β    = {self.beta:.3e}")
            print(f"  Regime:               ", end="")
            # beta 임계값(0.01, 0.1, 10)은 실질적으로 의미 있는 경계에 해당한다:
            # β < 0.01은 자기권 심부와 토카막 중심부에서 전형적으로, 자기력선 굽힘이 무시 가능하며;
            # β > 10은 자기장이 역학적으로 무관함을 의미한다.
            if self.beta < 0.01:
                print("Strongly magnetized (β << 1)")
            elif self.beta < 0.1:
                print("Magnetized (β < 1)")
            elif self.beta < 10:
                print("Moderate β")
            else:
                print("Weakly magnetized (β > 1)")
            print("-"*60)

            print(f"Frequency Orderings:")
            print(f"  ω_ci/ω_ce = {self.omega_ci/self.omega_ce:.3e}")
            print(f"  ω_ce/ω_pe = {self.omega_ce/self.omega_pe:.3e}")
            print(f"  ω_pi/ω_pe = {self.omega_pi/self.omega_pe:.3e}")
            print("-"*60)

            print(f"Length Scale Orderings:")
            print(f"  r_Le/λ_D  = {self.r_Le/self.lambda_D:.3e}")
            print(f"  r_Li/r_Le = {self.r_Li/self.r_Le:.3e}")

        print("="*60)


# Example usage
if __name__ == "__main__":
    # 세 가지 예시는 밀도에서 ~13차수, 온도에서 4차수에 걸쳐 있으며,
    # 질적으로 서로 다른 영역을 대표하도록 선택되었다: 핵융합(강한 자기화, 무충돌),
    # 우주(약한 자기화, 무충돌), 산업용 플라즈마(비자기화, 부분 충돌).
    print("\n### Example 1: Tokamak Core ###\n")
    tokamak = PlasmaParameters(n_e=1e20, T_e=10000, B=5, T_i=10000, Z=1, A=2)
    tokamak.print_summary()

    print("\n### Example 2: Solar Wind (1 AU) ###\n")
    solar_wind = PlasmaParameters(n_e=1e7, T_e=10, B=1e-9, T_i=10, Z=1, A=1)
    solar_wind.print_summary()

    print("\n### Example 3: Fluorescent Lamp ###\n")
    fluorescent = PlasmaParameters(n_e=1e16, T_e=2, B=0, T_i=0.1, Z=1, A=40)
    fluorescent.print_summary()
```

### 8.2 플라즈마 매개변수 공간 시각화

다양한 플라즈마의 매개변수 공간을 시각화해 봅시다:

```python
def plot_plasma_parameter_space():
    """Plot various plasmas in (n, T) space with characteristic regions."""

    # Define various plasmas
    plasmas = {
        'Interstellar medium': (1e6, 1),
        'Solar wind': (1e7, 10),
        'Ionosphere': (1e12, 0.1),
        'Solar corona': (1e14, 100),
        'Glow discharge': (1e16, 3),
        'Tokamak edge': (1e19, 100),
        'Tokamak core': (1e20, 10000),
        'Laser plasma': (1e26, 1000),
        'White dwarf': (1e36, 10000),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: n-T space with Debye length contours
    n_range = np.logspace(4, 38, 100)
    T_range = np.logspace(-2, 5, 100)
    N, T = np.meshgrid(n_range, T_range)

    # Debye length [m]
    lambda_D = np.sqrt(epsilon_0 * k_B * T * eV_to_K / (N * e**2))

    contour1 = ax1.contour(N, T, lambda_D,
                           levels=[1e-12, 1e-9, 1e-6, 1e-3, 1, 1e3],
                           colors='gray', alpha=0.5, linewidths=0.8)
    ax1.clabel(contour1, inline=True, fontsize=8, fmt='λ_D=%gm')

    # Plot plasmas
    for name, (n, T_eV) in plasmas.items():
        ax1.scatter(n, T_eV, s=100, alpha=0.7, edgecolors='black', linewidths=1.5)
        ax1.annotate(name, (n, T_eV), fontsize=9,
                    xytext=(5, 5), textcoords='offset points')

    # Plasma parameter N_D = 1 line
    T_for_ND1 = 3/(4*np.pi) * (e**2 / (epsilon_0 * k_B * eV_to_K)) * n_range**(-2/3)
    ax1.plot(n_range, T_for_ND1, 'r--', linewidth=2,
             label=r'$n\lambda_D^3 = 1$ (plasma boundary)')

    ax1.set_xlabel(r'Electron Density $n_e$ [m$^{-3}$]', fontsize=12)
    ax1.set_ylabel(r'Temperature $T$ [eV]', fontsize=12)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1e4, 1e38)
    ax1.set_ylim(1e-2, 1e5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_title('Plasma Parameter Space', fontsize=14, fontweight='bold')

    # Plot 2: Coupling parameter space
    # Compute coupling parameter Γ
    a = (3/(4*np.pi*N))**(1/3)  # mean inter-particle spacing
    Gamma = e**2 / (4*np.pi*epsilon_0 * a * k_B * T * eV_to_K)

    contour2 = ax2.contourf(N, T, Gamma,
                            levels=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
                            cmap='RdYlBu_r', alpha=0.7)
    cbar = plt.colorbar(contour2, ax=ax2, label=r'Coupling Parameter $\Gamma$')

    # Mark Γ = 1 (strongly coupled boundary)
    cs = ax2.contour(N, T, Gamma, levels=[1], colors='red', linewidths=3)
    ax2.clabel(cs, inline=True, fontsize=12, fmt=r'$\Gamma=1$')

    # Plot plasmas
    for name, (n, T_eV) in plasmas.items():
        ax2.scatter(n, T_eV, s=100, alpha=0.9, edgecolors='black',
                   linewidths=1.5, zorder=10)
        ax2.annotate(name, (n, T_eV), fontsize=9,
                    xytext=(5, 5), textcoords='offset points')

    ax2.set_xlabel(r'Electron Density $n_e$ [m$^{-3}$]', fontsize=12)
    ax2.set_ylabel(r'Temperature $T$ [eV]', fontsize=12)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(1e4, 1e38)
    ax2.set_ylim(1e-2, 1e5)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Plasma Coupling Parameter Space', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('plasma_parameter_space.png', dpi=150, bbox_inches='tight')
    plt.show()

# Run visualization
plot_plasma_parameter_space()
```

### 8.3 주파수 및 길이 스케일 비교

```python
def compare_plasma_scales():
    """Compare characteristic frequencies and length scales for various plasmas."""

    plasma_configs = {
        'Tokamak core': {'n_e': 1e20, 'T_e': 10000, 'B': 5, 'T_i': 10000},
        'Tokamak edge': {'n_e': 1e19, 'T_e': 100, 'B': 5, 'T_i': 100},
        'Solar wind': {'n_e': 1e7, 'T_e': 10, 'B': 1e-9, 'T_i': 10},
        'Ionosphere': {'n_e': 1e12, 'T_e': 0.1, 'B': 5e-5, 'T_i': 0.1},
        'Fluorescent': {'n_e': 1e16, 'T_e': 2, 'B': 0, 'T_i': 0.1},
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = list(plasma_configs.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    # Compute parameters
    results = {}
    for name, config in plasma_configs.items():
        pp = PlasmaParameters(**config)
        results[name] = pp

    # Plot 1: Frequencies
    ax = axes[0, 0]
    x = np.arange(len(names))
    width = 0.2

    omega_pe_vals = [results[n].omega_pe for n in names]
    omega_pi_vals = [results[n].omega_pi for n in names]
    omega_ce_vals = [results[n].omega_ce if results[n].omega_ce else 0 for n in names]
    omega_ci_vals = [results[n].omega_ci if results[n].omega_ci else 0 for n in names]

    ax.bar(x - 1.5*width, omega_pe_vals, width, label=r'$\omega_{pe}$', alpha=0.8)
    ax.bar(x - 0.5*width, omega_pi_vals, width, label=r'$\omega_{pi}$', alpha=0.8)
    ax.bar(x + 0.5*width, omega_ce_vals, width, label=r'$\omega_{ce}$', alpha=0.8)
    ax.bar(x + 1.5*width, omega_ci_vals, width, label=r'$\omega_{ci}$', alpha=0.8)

    ax.set_ylabel('Frequency [rad/s]', fontsize=11)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Characteristic Frequencies', fontsize=12, fontweight='bold')

    # Plot 2: Length scales
    ax = axes[0, 1]
    lambda_D_vals = [results[n].lambda_D for n in names]
    r_Le_vals = [results[n].r_Le if results[n].r_Le else np.nan for n in names]
    r_Li_vals = [results[n].r_Li if results[n].r_Li else np.nan for n in names]

    ax.bar(x - width, lambda_D_vals, width, label=r'$\lambda_D$', alpha=0.8)
    ax.bar(x, r_Le_vals, width, label=r'$r_{Le}$', alpha=0.8)
    ax.bar(x + width, r_Li_vals, width, label=r'$r_{Li}$', alpha=0.8)

    ax.set_ylabel('Length [m]', fontsize=11)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Characteristic Length Scales', fontsize=12, fontweight='bold')

    # Plot 3: Plasma parameter and Beta
    ax = axes[1, 0]
    N_D_vals = [results[n].N_D for n in names]

    ax.bar(x, N_D_vals, color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, label=r'$N_D = 1$')
    ax.set_ylabel(r'Plasma Parameter $N_D$', fontsize=11)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Plasma Parameter (Debye sphere population)', fontsize=12, fontweight='bold')

    # Plot 4: Beta
    ax = axes[1, 1]
    beta_vals = [results[n].beta if results[n].beta else np.nan for n in names]

    ax.bar(x, beta_vals, color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, label=r'$\beta = 1$')
    ax.set_ylabel(r'Plasma Beta $\beta$', fontsize=11)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Plasma Beta (thermal/magnetic pressure)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('plasma_scales_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# Run comparison
compare_plasma_scales()
```

### 8.4 대화형 매개변수 탐색기

```python
def interactive_parameter_scan():
    """
    Scan plasma parameters to understand scaling relationships.
    """
    # Vary density at fixed temperature
    n_vals = np.logspace(14, 22, 50)
    T_fixed = 1000  # eV
    B_fixed = 2     # T

    lambda_D = []
    omega_pe = []
    r_Le = []
    beta = []

    for n in n_vals:
        pp = PlasmaParameters(n_e=n, T_e=T_fixed, B=B_fixed)
        lambda_D.append(pp.lambda_D)
        omega_pe.append(pp.omega_pe)
        r_Le.append(pp.r_Le)
        beta.append(pp.beta)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.loglog(n_vals, lambda_D, 'b-', linewidth=2)
    ax.set_xlabel(r'Density $n_e$ [m$^{-3}$]', fontsize=11)
    ax.set_ylabel(r'Debye Length $\lambda_D$ [m]', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(r'$\lambda_D \propto n^{-1/2}$', fontsize=12)

    ax = axes[0, 1]
    ax.loglog(n_vals, omega_pe, 'r-', linewidth=2)
    ax.set_xlabel(r'Density $n_e$ [m$^{-3}$]', fontsize=11)
    ax.set_ylabel(r'Plasma Frequency $\omega_{pe}$ [rad/s]', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(r'$\omega_{pe} \propto n^{1/2}$', fontsize=12)

    ax = axes[1, 0]
    ax.loglog(n_vals, r_Le, 'g-', linewidth=2)
    ax.set_xlabel(r'Density $n_e$ [m$^{-3}$]', fontsize=11)
    ax.set_ylabel(r'Electron Larmor Radius $r_{Le}$ [m]', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(r'$r_{Le}$ independent of $n$ (fixed $T, B$)', fontsize=12)

    ax = axes[1, 1]
    ax.loglog(n_vals, beta, 'm-', linewidth=2)
    ax.set_xlabel(r'Density $n_e$ [m$^{-3}$]', fontsize=11)
    ax.set_ylabel(r'Plasma Beta $\beta$', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(r'$\beta \propto n$ (fixed $T, B$)', fontsize=12)

    plt.suptitle(f'Parameter Scaling with Density (T={T_fixed} eV, B={B_fixed} T)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('parameter_scaling_density.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Now vary temperature at fixed density
    T_vals = np.logspace(0, 4, 50)
    n_fixed = 1e19  # m^-3

    lambda_D = []
    v_te = []
    r_Le = []
    beta = []

    for T in T_vals:
        pp = PlasmaParameters(n_e=n_fixed, T_e=T, B=B_fixed)
        lambda_D.append(pp.lambda_D)
        v_te.append(pp.v_te)
        r_Le.append(pp.r_Le)
        beta.append(pp.beta)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.loglog(T_vals, lambda_D, 'b-', linewidth=2)
    ax.set_xlabel(r'Temperature $T_e$ [eV]', fontsize=11)
    ax.set_ylabel(r'Debye Length $\lambda_D$ [m]', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(r'$\lambda_D \propto T^{1/2}$', fontsize=12)

    ax = axes[0, 1]
    ax.loglog(T_vals, v_te, 'r-', linewidth=2)
    ax.set_xlabel(r'Temperature $T_e$ [eV]', fontsize=11)
    ax.set_ylabel(r'Thermal Velocity $v_{te}$ [m/s]', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(r'$v_{te} \propto T^{1/2}$', fontsize=12)

    ax = axes[1, 0]
    ax.loglog(T_vals, r_Le, 'g-', linewidth=2)
    ax.set_xlabel(r'Temperature $T_e$ [eV]', fontsize=11)
    ax.set_ylabel(r'Electron Larmor Radius $r_{Le}$ [m]', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(r'$r_{Le} \propto T^{1/2}$', fontsize=12)

    ax = axes[1, 1]
    ax.loglog(T_vals, beta, 'm-', linewidth=2)
    ax.set_xlabel(r'Temperature $T_e$ [eV]', fontsize=11)
    ax.set_ylabel(r'Plasma Beta $\beta$', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(r'$\beta \propto T$ (fixed $n, B$)', fontsize=12)

    plt.suptitle(f'Parameter Scaling with Temperature (n={n_fixed:.0e} m⁻³, B={B_fixed} T)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('parameter_scaling_temperature.png', dpi=150, bbox_inches='tight')
    plt.show()

# Run interactive scans
interactive_parameter_scan()
```

## 요약

이 레슨에서는 플라즈마를 물질의 네 번째 상태로 소개하고 플라즈마 행동을 특성화하는 기본 매개변수를 유도했습니다:

1. **Debye 차폐**: 플라즈마는 Debye 길이 $\lambda_D = \sqrt{\epsilon_0 k_B T/(ne^2)}$에 걸쳐 전하 불균형을 차폐하여, 장거리 Coulomb 힘을 지수적으로 차폐된 상호작용으로 변환합니다.

2. **플라즈마 기준**: 집단적 행동을 위해서는 $n\lambda_D^3 \gg 1$(Debye 구에 많은 입자)과 $L \gg \lambda_D$ 스케일에서의 준중성이 필요합니다.

3. **플라즈마 주파수**: 자연 정전기 진동 주파수 $\omega_{pe} = \sqrt{ne^2/(\epsilon_0 m_e)}$는 전자 역학의 기본 시간 스케일을 설정합니다.

4. **Gyrofrequency 및 Larmor 반경**: 자기장에서 입자는 $\omega_c = |q|B/m$에서 반경 $r_L = v_\perp/\omega_c$로 회전하여, 비등방성과 새로운 길이/시간 스케일을 도입합니다.

5. **플라즈마 베타**: 비율 $\beta = 2\mu_0 nk_B T/B^2$는 열압력 또는 자기압력이 지배하는지를 결정하며, 제약 및 안정성에 중대한 영향을 미칩니다.

이러한 매개변수는 성간 매질에서 핵융합 장치에 이르기까지 밀도와 온도에서 40차수에 걸친 플라즈마 행동을 이해하는 기초를 형성합니다.

## 연습 문제

### 문제 1: 플라즈마 매개변수 계산

$n_e = 10^{17}$ m$^{-3}$, $T_e = 3$ eV, $B = 0.01$ T인 저온 플라즈마를 고려하세요.

(a) Debye 길이, 플라즈마 주파수, 전자 gyrofrequency, 전자 Larmor 반경을 계산하세요.

(b) 이 시스템이 플라즈마 기준 $n\lambda_D^3 \gg 1$을 만족하는지 확인하세요.

(c) 시간 스케일 $\omega_{pe}^{-1}$과 $\omega_{ce}^{-1}$을 비교하세요. 어느 것이 더 빠릅니까?

(d) 이온이 아르곤(A=40)이면, 이온 gyrofrequency와 Larmor 반경을 계산하세요. 전자 값과 비교하세요.

### 문제 2: Tokamak에서의 Debye 차폐

토카막의 중심부 매개변수: $n_e = 10^{20}$ m$^{-3}$, $T_e = 15$ keV, $T_i = 12$ keV.

(a) 전자와 이온의 Debye 길이 기여를 별도로 계산한 다음 유효 Debye 길이를 구하세요.

(b) 시험 전하로부터의 퍼텐셜이 Coulomb 값의 $1/e$로 감소하는 거리는 얼마입니까?

(c) 토카막의 소반경이 $a = 1$ m이면, 준중성이 훌륭한 근사임을 확인하세요.

(d) 하나의 Debye 구 내에 몇 개의 전자가 있습니까?

### 문제 3: 스케일링 분석

자기장 $B = 2$ T의 수소 플라즈마에 대해:

(a) 전자 Larmor 반경이 Debye 길이와 같아지는 밀도 $n$을 구하세요. $T_e = 1$ keV라고 가정합니다.

(b) $n$, $T_e$, $B$의 관점에서 비율 $r_{Le}/\lambda_D$에 대한 일반 식을 유도하세요.

(c) (a)에서 구한 밀도에 대해 플라즈마 베타를 계산하세요. 이것은 자기화된 플라즈마입니까, 아니면 자기화되지 않은 플라즈마입니까?

### 문제 4: 태양 코로나 매개변수

태양 코로나는 $n_e \sim 10^{14}$ m$^{-3}$, $T_e \sim 100$ eV, $B \sim 10^{-2}$ T입니다.

(a) 모든 특성 플라즈마 매개변수를 계산하세요.

(b) 플라즈마 베타를 결정하세요. 이것이 코로나의 자기 제약에 대해 무엇을 말해줍니까?

(c) Alfvén 속도는 $v_A = B/\sqrt{\mu_0 n_i m_i}$입니다(양성자의 경우). $v_A$를 전자 열속도와 비교하세요. 이것이 파동 전파에 대해 무엇을 의미합니까?

(d) 밀도가 $n \propto r^{-2}$로, $B \propto r^{-2}$로 떨어진다고 가정할 때, 플라즈마가 자기화되지 않게 되는($\beta \sim 1$) 태양으로부터의 거리(태양 반지름 $R_\odot$ 단위)는 얼마입니까?

### 문제 5: 실험실 플라즈마 영역

$10^{16} \le n \le 10^{21}$ m$^{-3}$과 $1 \le T_e \le 1000$ eV로 $n$과 $T$를 독립적으로 변화시킬 수 있는 플라즈마 실험을 설계하고 있습니다. $B = 1$ T의 자기장을 사용할 수 있습니다.

(a) $(n, T)$ 공간에서 일정한 $\lambda_D$, $\omega_{pe}$, $r_{Le}$, $\beta$의 등고선을 보여주는 플롯을 만드세요.

(b) 다음 영역을 식별하세요:
   - 플라즈마가 약결합($\Gamma < 0.1$)
   - 전자 Larmor 반경이 1 mm보다 작음
   - 플라즈마 베타가 0.01보다 작음

(c) $n = 10^{19}$ m$^{-3}$과 $T_e = 100$ eV에 대해 충돌 주파수를 계산하세요(레슨 2의 개념이 필요합니다—추정으로 $\ln\Lambda \approx 15$인 SI 단위로 $\nu_{ei} \sim 10^{-6} n_e \ln\Lambda / T_e^{3/2}$ 사용).

(d) $\nu_{ei}$를 $\omega_{ce}$와 비교하여 이 플라즈마가 충돌성인지 무충돌성인지 결정하세요.

---

**이전:** [개요](./00_Overview.md) | **다음:** [Coulomb 충돌](./02_Coulomb_Collisions.md)
