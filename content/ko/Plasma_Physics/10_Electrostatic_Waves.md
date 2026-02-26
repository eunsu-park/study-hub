# 10. 정전기 파동

## 학습 목표

- 정전기 파동 분산 관계에 대한 일반 프레임워크 이해
- 따뜻한 플라즈마에서 Langmuir 파동에 대한 Bohm-Gross 분산 유도
- 이온 음향 파동 및 그 감쇠 조건 분석
- 자화 플라즈마에서 upper hybrid 및 lower hybrid 공명 계산
- Bernstein modes와 파동 가열에서의 역할 학습
- 다양한 플라즈마 매개변수에 대한 정전기 분산 관계를 수치적으로 풀이

## 서론

이전 강의에서 우리는 자기장 섭동을 포함하는 MHD 및 전자기 파동을 연구했습니다. 이제 우리는 **정전기 파동**에 초점을 맞춥니다:

$$\mathbf{B}_1 = 0, \quad \mathbf{E}_1 = -\nabla\phi_1$$

이러한 파동은 전기장 변동만을 포함하며 스칼라 포텐셜로부터 유도될 수 있습니다. 이들은 전하 분리에서 발생하며 Poisson 방정식에 의해 지배됩니다:

$$\nabla \cdot \mathbf{E}_1 = \frac{\rho_1}{\epsilon_0}$$

정전기 파동은 다음에 중요합니다:
- 플라즈마 가열 및 전류 구동 (electron cyclotron, lower hybrid)
- 진단 (Thomson scattering, collective modes)
- 불안정성 이해 (two-stream, drift waves)
- 파동-입자 상호작용 (Landau damping, quasilinear theory)

핵심 도구는 전자기 섭동에 대한 플라즈마의 응답을 인코딩하는 **유전 함수** $\epsilon(\mathbf{k}, \omega)$입니다.

## 1. 일반 정전기 분산 프레임워크

### 1.1 유전 함수와 감수성

섭동 $\propto e^{i(\mathbf{k}\cdot\mathbf{x} - \omega t)}$에 대해, Poisson 방정식은 다음이 됩니다:

$$\mathbf{k} \cdot \mathbf{E}_1 = \frac{\rho_1}{\epsilon_0}$$

유도된 전하 밀도 $\rho_1$은 플라즈마 응답을 통해 전기장에 의존합니다. **유전 함수** $\epsilon(\mathbf{k}, \omega)$를 다음과 같이 정의합니다:

$$\mathbf{k} \cdot \mathbf{D} = \mathbf{k} \cdot (\epsilon_0 \epsilon \mathbf{E}_1) = 0$$

정전기 파동의 경우, 이는 다음을 제공합니다:

$$\boxed{\epsilon(\mathbf{k}, \omega) = 0}$$

이것이 **정전기 분산 관계**입니다.

유전 함수는 각 종의 **감수성** $\chi_s$와 관련됩니다:

$$\epsilon = 1 + \sum_s \chi_s(\mathbf{k}, \omega)$$

감수성은 종 $s$가 전기장에 어떻게 응답하는지 설명합니다:

$$\chi_s = \frac{n_{1s}/n_0}{\epsilon_0 E_1 / (e_s n_0)}$$

선형화된 Vlasov 방정식으로부터, 종 $s$에 대한 감수성은:

$$\chi_s(\mathbf{k}, \omega) = -\frac{\omega_{ps}^2}{k^2} \int \frac{\mathbf{k} \cdot \partial f_0/\partial \mathbf{v}}{\omega - \mathbf{k}\cdot\mathbf{v}} d^3v$$

여기서 $\omega_{ps}^2 = n_0 e_s^2 / (\epsilon_0 m_s)$는 플라즈마 주파수입니다.

### 1.2 차가운 플라즈마 한계

**차가운 플라즈마** ($T = 0$, $f_0 = n_0 \delta(\mathbf{v})$)의 경우, 적분이 단순화됩니다. 입자는 다음을 통해 파동 전기장에 응답합니다:

$$m \frac{d\mathbf{v}_1}{dt} = e \mathbf{E}_1$$

조화 섭동 $\propto e^{-i\omega t}$에 대해:
$$-i\omega m \mathbf{v}_1 = e \mathbf{E}_1$$

연속성으로부터의 밀도 섭동은:
$$-i\omega n_1 + n_0 i\mathbf{k}\cdot\mathbf{v}_1 = 0$$

결합하면:
$$n_1 = \frac{n_0 \mathbf{k}\cdot\mathbf{v}_1}{\omega} = \frac{n_0 e \mathbf{k}\cdot\mathbf{E}_1}{m\omega^2}$$

감수성은:
$$\chi_s = \frac{n_1 e_s}{\epsilon_0 E_1 n_0} = -\frac{\omega_{ps}^2}{\omega^2}$$

여러 차가운 종에 대한 유전 함수:

$$\epsilon = 1 - \sum_s \frac{\omega_{ps}^2}{\omega^2}$$

### 1.3 따뜻한 플라즈마: 운동 효과

열 속도 $v_{th}$를 가진 **따뜻한 플라즈마**의 경우, 다음일 때 운동 효과가 중요해집니다:

$$\frac{\omega}{k} \sim v_{th}$$

위상 속도 $v_\phi = \omega/k$가 입자 속도와 비교할 만합니다. 이는 다음으로 이어집니다:
- **Landau damping**: 공명 파동-입자 상호작용
- **열 분산**: 파동 주파수가 $k v_{th}$에 의존

Maxwellian 분포의 경우:
$$f_0(\mathbf{v}) = n_0 \left(\frac{m}{2\pi k_B T}\right)^{3/2} \exp\left(-\frac{m v^2}{2k_B T}\right)$$

1D 감수성 ($\mathbf{k} = k \hat{z}$에 대해)은 **플라즈마 분산 함수** $Z(\zeta)$를 포함합니다:

$$\chi_s = -\frac{\omega_{ps}^2}{k^2 v_{th,s}^2} \left[1 + \zeta_s Z(\zeta_s)\right]$$

여기서 $\zeta_s = \omega/(k v_{th,s})$이고:

$$Z(\zeta) = \frac{1}{\sqrt{\pi}} \int_{-\infty}^{\infty} \frac{e^{-x^2}}{x - \zeta} dx$$

큰 $|\zeta|$ (위상 속도 $\gg v_{th}$)의 경우:
$$Z(\zeta) \approx -\frac{1}{\zeta}\left(1 + \frac{1}{2\zeta^2} + \frac{3}{4\zeta^4} + \cdots\right)$$

이는 차가운 플라즈마 결과에 대한 열 보정을 제공합니다.

## 2. Langmuir 파동 (전자 플라즈마 진동)

### 2.1 차가운 플라즈마: 순수 진동

정지 이온을 가진 무자화 플라즈마에서, 차가운 플라즈마 분산은:

$$\epsilon = 1 - \frac{\omega_{pe}^2}{\omega^2} = 0$$

이는 다음을 제공합니다:
$$\boxed{\omega = \omega_{pe}}$$

이것이 1929년 Langmuir와 Tonks에 의해 처음 관찰된 **Langmuir 파동** (또는 전자 플라즈마 진동)입니다.

특성:
- **무분산**: $\omega$가 $k$와 무관
- **전파 없음**: 군속도 $v_g = d\omega/dk = 0$
- **정상 진동**: 전자가 집단적으로 진동
- **고주파**: 전형적인 플라즈마에 대해 $\omega_{pe} \sim 10^{9}-10^{11}$ rad/s

물리적 그림:
```
Time t=0:           Time t=T/4:         Time t=T/2:
-  +  -  +  -       - - -  +  +  +      -  +  -  +  -
Ions (stationary)   Electrons          Back to equilibrium
                    displaced
```

### 2.2 따뜻한 플라즈마: Bohm-Gross 분산

전자 열 운동을 포함하면, 분산은 다음이 됩니다:

$$1 - \frac{\omega_{pe}^2}{k^2 v_{th,e}^2}[1 + \zeta_e Z(\zeta_e)] = 0$$

$\omega \gg k v_{th,e}$ (약한 열 효과)의 경우, 1차까지 전개하면:

$$1 - \frac{\omega_{pe}^2}{\omega^2}\left(1 + \frac{3k^2 v_{th,e}^2}{\omega^2}\right) = 0$$

$\omega^2$에 대해 풀면:

$$\boxed{\omega^2 = \omega_{pe}^2 + 3k^2 v_{th,e}^2}$$

이것이 **Bohm-Gross 분산 관계** (1949)이며, 여기서 $v_{th,e} = \sqrt{k_B T_e / m_e}$입니다.

특성:
- **분산성**: $\omega$가 $k$에 의존
- **위상 속도**: $v_\phi = \omega/k = \sqrt{\omega_{pe}^2/k^2 + 3v_{th,e}^2}$
- **군속도**: $v_g = d\omega/dk = 3k v_{th,e}^2/\omega$
- **차단**: $k = 0$에서 최소 주파수 $\omega_{pe}$

열 보정은 다음일 때 중요합니다:
$$k\lambda_D \sim 1$$

여기서 $\lambda_D = v_{th,e}/\omega_{pe}$는 Debye 길이입니다.

### 2.3 Langmuir 파동의 Landau Damping

$\omega/k \sim v_{th,e}$의 경우, $Z(\zeta)$의 허수 부분이 감쇠를 제공합니다:

$$\text{Im}\,\omega = -\sqrt{\frac{\pi}{8}} \frac{\omega_{pe}}{k^3 \lambda_D^3} \exp\left(-\frac{\omega_{pe}^2}{2k^2 v_{th,e}^2} - \frac{3}{2}\right)$$

감쇠율은 $k\lambda_D \ll 1$에 대해 지수적으로 감소합니다.

물리적 그림: $v \approx \omega/k$로 움직이는 입자는 파동과 공명적으로 에너지를 교환할 수 있습니다. 빠른 입자보다 느린 입자가 더 많으면 (Maxwellian은 높은 $v$에서 음의 기울기를 가짐), 파동은 에너지를 잃습니다.

## 3. 이온 음향 파동

### 3.1 유도: 이유체 모델

다음과 같은 **저주파** 진동을 고려합니다:
- 전자가 단열적으로 응답 (빠름, 등온)
- 이온이 동역학적으로 응답 (느림, 관성)

**전자 응답** (Boltzmann 관계):
$$n_e = n_0 \exp\left(\frac{e\phi}{k_B T_e}\right) \approx n_0\left(1 + \frac{e\phi}{k_B T_e}\right)$$

**이온 연속성 및 운동량**:
$$\frac{\partial n_i}{\partial t} + \nabla\cdot(n_i \mathbf{v}_i) = 0$$

$$m_i n_i \frac{\partial \mathbf{v}_i}{\partial t} = -e n_i \nabla\phi - \nabla(n_i k_B T_i)$$

**Poisson 방정식**:
$$\epsilon_0 \nabla^2 \phi = e(n_e - n_i)$$

$n_e = n_0 + n_{e1}$, $n_i = n_0 + n_{i1}$, $\phi = \phi_1$로 선형화하고 $e^{i(kx - \omega t)}$를 가정하면:

$$-\omega^2 m_i n_{i1} = -k^2 e n_{i1} \phi_1 - k^2 n_{i1} k_B T_i$$

$$-k^2 \epsilon_0 \phi_1 = e\left(\frac{e n_0 \phi_1}{k_B T_e} - n_{i1}\right)$$

결합하면:
$$-k^2 \epsilon_0 \phi_1 = e\left(\frac{e n_0 \phi_1}{k_B T_e} - \frac{k^2 e \phi_1 (n_0 + k_B T_i k^2/\omega^2 m_i)}{\omega^2 m_i - k^2 k_B T_i}\right)$$

대수학 후, 분산 관계는:

$$\omega^2 = \frac{k^2 k_B T_e}{m_i (1 + k^2 \lambda_D^2)} \left(\frac{1 + 3T_i/T_e}{1 + k^2\lambda_D^2}\right)$$

$k\lambda_D \ll 1$의 경우:

$$\boxed{\omega \approx k c_s}$$

여기서 **이온 음향 속도**는:

$$\boxed{c_s = \sqrt{\frac{k_B T_e}{m_i}}}$$

$T_i \ll T_e$를 가정합니다.

### 3.2 물리적 그림

이온 음향 파동은 중성 기체의 음파와 유사하지만:
- 복원력: 전자 압력 (이온 압력이 아님)
- 관성: 이온 질량
- 전자가 열역학적 저장소로 작용

```
Ion density:     High    Low     High    Low
                  |       |       |       |
                  v       ^       v       ^
                 +++     ---     +++     ---
Electron cloud: (responds adiabatically)

Wave propagates at c_s ~ √(Te/mi)
```

전형적인 속도:
- $T_e = 10$ eV, 수소: $c_s \approx 50$ km/s
- $T_e = 1$ keV, 중수소: $c_s \approx 200$ km/s

전자 열 속도 ($v_{th,e} \sim 10^3$ km/s)보다 훨씬 느리지만, 이온 열 속도 ($v_{th,i} \sim 10$ km/s)보다 빠릅니다.

### 3.3 이온 Landau Damping

$\omega/k \sim v_{th,i}$의 경우, 파동 위상 속도 근처에서 움직이는 이온이 파동을 감쇠시킬 수 있습니다:

$$\gamma \approx -\sqrt{\frac{\pi}{8}} \frac{c_s}{k\lambda_D} \left(\frac{T_i}{T_e}\right)^{3/2} \exp\left(-\frac{1}{2k^2\lambda_D^2} - \frac{3}{2}\right)$$

**약한 감쇠**의 조건은:

$$T_e \gg T_i$$

$T_e \sim T_i$인 경우, 파동은 심하게 감쇠되어 효과적으로 전파되지 않습니다.

이것은 실험실 플라즈마에서 관찰됩니다:
- 뜨거운 전자, 차가운 이온: 강한 이온 음향 파동
- 비교할 만한 온도: 파동이 감쇠됨

### 3.4 난류에서의 역할

이온 음향 난류는 플라즈마에 편재합니다:
- 전류에 의해 구동됨 (이온 음향 불안정성)
- 비선형 상호작용이 mode의 스펙트럼을 생성
- 이상 저항성, 가열로 이어짐

다음에서 관찰됨:
- 전리층 (레이더 산란)
- 태양풍
- 토카막 가장자리 플라즈마

## 4. 자화 플라즈마의 파동

### 4.1 Upper Hybrid 공명

자화 플라즈마에서, 전자는 $\omega_{ce}$에서 회전하면서 $\omega_{pe}$에서 진동합니다. $\mathbf{B}$에 수직으로 전파하는 파동 ($\mathbf{k} \perp \mathbf{B}$)의 경우, 두 운동이 결합됩니다.

$\mathbf{B}$를 가로지르는 정전기 파동에 대한 분산 관계는:

$$\epsilon_{\perp} = 1 - \frac{\omega_{pe}^2}{\omega^2 - \omega_{ce}^2} = 0$$

이는 다음을 제공합니다:

$$\boxed{\omega_{UH}^2 = \omega_{pe}^2 + \omega_{ce}^2}$$

이것이 **upper hybrid 주파수**입니다.

물리적 그림:
- 전자가 $\omega_{pe}$에서 진동
- 자기장이 $\mathbf{v} \times \mathbf{B}$로부터 복원력을 추가
- 유효 주파수는 둘의 기하 평균

upper hybrid layer ($\omega = \omega_{UH}$인 곳)는 다음에 사용됩니다:
- 플라즈마 가열 (upper hybrid resonance heating)
- 전류 구동
- 진단 (reflectometry)

### 4.2 Lower Hybrid 공명

이온 운동을 포함하는 더 낮은 주파수의 경우, **lower hybrid 주파수**는 이온 cyclotron 및 이온 플라즈마 진동의 결합으로부터 발생합니다:

$$\frac{1}{\omega_{LH}^2} = \frac{1}{\omega_{ci}\omega_{ce}} + \frac{1}{\omega_{pi}^2 + \omega_{ci}^2}$$

$\omega_{pi}^2 \gg \omega_{ci}^2$의 경우:

$$\boxed{\omega_{LH} \approx \sqrt{\frac{\omega_{ci}\omega_{ce}}{1 + \omega_{pe}^2/\omega_{ce}^2}} \approx \sqrt{\omega_{ci}\omega_{ce}}}$$

($\omega_{pe}^2 \ll \omega_{ce}^2$를 가정).

전형적인 값:
- $B = 5$ T, $n = 10^{20}$ m$^{-3}$, 수소
- $f_{ce} \approx 140$ GHz, $f_{ci} \approx 76$ MHz
- $f_{LH} \approx 3.3$ GHz

Lower hybrid 파동은 다음에 광범위하게 사용됩니다:
- **전류 구동**: Landau damping을 통해 효율적으로 전류를 구동할 수 있음
- 가열 (ECRH 또는 ICRH보다 덜 효율적이지만)
- 토카막에서 플라즈마 시작

파동은 $\mathbf{B}$에 대한 특정 각도로 전파할 수 있어 국소화된 증착을 허용합니다.

### 4.3 따뜻한 플라즈마 보정

유한 온도를 포함하면, 차단 및 공명이 이동합니다:

$$\omega_{UH}^2 = \omega_{pe}^2 + \omega_{ce}^2 + 3k^2 v_{th,e}^2$$

lower hybrid의 경우:
$$\omega_{LH}^2 \approx \omega_{ci}\omega_{ce} \left(1 + \frac{\omega_{pe}^2}{\omega_{ce}^2}\right)^{-1}\left(1 + k^2\lambda_D^2\right)$$

열 보정은 $k\lambda_D \sim 1$ 또는 $k\rho_L \sim 1$ (여기서 $\rho_L$은 Larmor 반경)일 때 중요해집니다.

## 5. Bernstein Modes

### 5.1 전자 Bernstein Modes

전자 cyclotron 주파수의 고조파에서, 전자기 파동이 차단될 때도 **정전기** modes가 전파할 수 있습니다. 이것이 **Bernstein modes**입니다.

분산 관계는 Bessel 함수를 포함합니다. $n$번째 고조파 근처에서:

$$\omega \approx n\omega_{ce}\left(1 + \frac{k^2 v_{th,e}^2}{2\omega_{ce}^2}\right)$$

이러한 modes는 $\omega \approx n\omega_{ce}$, $n = 1, 2, 3, \ldots$에 대해 존재합니다.

특성:
- 정전기 ($\mathbf{B}_1$ 없음)
- EM 파동이 할 수 없는 곳에서 전파 가능
- 공명 전자와 강한 상호작용
- 가열 및 전류 구동에 사용

$\mathbf{k} \perp \mathbf{B}$에 대한 뜨거운 플라즈마 유전체는:

$$\epsilon_{\perp} = 1 + \frac{\omega_{pe}^2}{k^2 v_{th,e}^2} \sum_{n=-\infty}^{\infty} \frac{I_n(\lambda) e^{-\lambda}}{\omega/\omega_{ce} - n} \left(\frac{\omega}{k v_{th,e}}\right)Z\left(\frac{\omega - n\omega_{ce}}{k v_{th,e}}\right)$$

여기서 $\lambda = k^2 \rho_L^2$이고 $I_n$은 수정된 Bessel 함수입니다.

### 5.2 이온 Bernstein Modes

마찬가지로, 이온 Bernstein modes는 $\omega_{ci}$의 고조파 근처에 존재합니다:

$$\omega \approx n\omega_{ci}\left(1 + \frac{k^2 v_{th,i}^2}{2\omega_{ci}^2}\right)$$

이것들은 다음에 중요합니다:
- 밀집 플라즈마에서 이온 가열
- Mode 변환 가열 (EM 파동이 정전기 Bernstein mode로 변환)
- 작은 스케일에서 kinetic Alfvén 파동 이해

### 5.3 Mode 변환

불균질 플라즈마에서, 전자기 파동은 특정 위치에서 정전기 파동으로 **mode 변환**할 수 있습니다:

```
 EM wave (O-mode or X-mode)
       |
       v
  Cutoff/Resonance layer
       |
       v
Bernstein wave (electrostatic)
       |
       v
   Absorption via Landau/cyclotron damping
```

이것은 **mode 변환 가열** 체계에서 사용되며, 직접 EM 파동 주입의 차단 문제를 피합니다.

## 6. Python 구현

### 6.1 Langmuir 파동 분산

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz

def plasma_dispersion_Z(zeta):
    """
    Plasma dispersion function Z(ζ).

    Z(ζ) = (1/√π) ∫ exp(-x²)/(x - ζ) dx

    Uses Faddeeva function: Z(ζ) = i√π w(ζ) where w(z) = exp(-z²)erfc(-iz)
    """
    return 1j * np.sqrt(np.pi) * wofz(zeta)

def langmuir_dispersion_cold(k, omega_pe):
    """
    Cold plasma dispersion: ω = ω_pe (independent of k).
    """
    return omega_pe * np.ones_like(k)

def langmuir_dispersion_warm(k, omega_pe, v_th):
    """
    Warm plasma dispersion (Bohm-Gross): ω² = ω_pe² + 3k²v_th².
    """
    return np.sqrt(omega_pe**2 + 3 * k**2 * v_th**2)

def langmuir_dispersion_kinetic(k, omega_pe, v_th, num_points=100):
    """
    Solve full kinetic dispersion using plasma dispersion function.

    ε = 1 - (ω_pe²/k²v_th²)[1 + ζZ(ζ)] = 0
    where ζ = ω/(kv_th)
    """
    omega_solutions = []

    for k_val in k:
        if k_val == 0:
            omega_solutions.append(omega_pe)
            continue

        # 탐색 범위가 ω_pe에서 시작하는 이유는 Langmuir 파동은 항상 차단 주파수보다
        # 위에서 전파하기 때문입니다 — ω_pe 아래에서는 k^2 < 0이므로 실수 파동이 없습니다.
        # 상한 ω_pe * 1.5는 열 보정이 ω를 ~ω_pe * (1 + 3/2 * (k λ_D)^2) 까지만
        # 올리기 때문에 충분합니다.
        omega_range = np.linspace(omega_pe, omega_pe * 1.5, num_points)

        epsilon = []
        for omega in omega_range:
            zeta = omega / (k_val * v_th)
            Z_val = plasma_dispersion_Z(zeta)
            # ε의 실수 부분만 사용하는 이유는 실수 주파수를 찾기 위해서입니다;
            # Im(ε)은 Landau 감쇠율을 주지만 ζ >> 1에서 지수적으로 작으며
            # 이를 추출하려면 별도의 풀이 방법이 필요합니다.
            eps = 1 - (omega_pe**2 / (k_val**2 * v_th**2)) * (1 + zeta * Z_val)
            epsilon.append(np.real(eps))

        epsilon = np.array(epsilon)

        # Re(ε)의 부호 변화는 영점을 표시합니다: ε = 0이 분산 관계입니다.
        # 첫 번째 부호 변화를 사용하면 더 작은 ζ값에서 Z(ζ)의 진동하는 꼬리로 인한
        # 가짜 근(spurious root)을 피할 수 있습니다.
        sign_changes = np.where(np.diff(np.sign(epsilon)))[0]
        if len(sign_changes) > 0:
            idx = sign_changes[0]
            omega_solutions.append(omega_range[idx])
        else:
            omega_solutions.append(omega_range[-1])

    return np.array(omega_solutions)

# Parameters
n = 1e19  # m^-3
T_e = 10  # eV
m_e = 9.109e-31  # kg
e = 1.602e-19  # C
epsilon_0 = 8.854e-12  # F/m
k_B = 1.381e-23  # J/K

omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
v_th = np.sqrt(2 * k_B * T_e * e / m_e)
lambda_D = v_th / omega_pe

print(f"Plasma frequency: f_pe = {omega_pe / (2*np.pi) / 1e9:.2f} GHz")
print(f"Thermal velocity: v_th = {v_th / 1e6:.2f} Mm/s")
print(f"Debye length: λ_D = {lambda_D * 1e6:.2f} μm")

# Wavenumber array
# k를 1/λ_D로 정규화하는 것이 자연스러운 선택입니다. k λ_D가 Bohm-Gross 보정과
# 강한 Landau 감쇠 시작을 모두 제어하는 단일 무차원 매개변수이기 때문입니다;
# 범위 0.01-2는 차가운 플라즈마 regime에서 운동 regime까지 포괄합니다.
k = np.linspace(0.01, 2, 100) / lambda_D  # Normalize to 1/λ_D

# Compute dispersions
omega_cold = langmuir_dispersion_cold(k, omega_pe)
omega_warm = langmuir_dispersion_warm(k, omega_pe, v_th)
omega_kinetic = langmuir_dispersion_kinetic(k, omega_pe, v_th)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Dispersion relation
ax1.plot(k * lambda_D, omega_cold / omega_pe, 'k--', linewidth=2, label='Cold plasma')
ax1.plot(k * lambda_D, omega_warm / omega_pe, 'b-', linewidth=2, label='Bohm-Gross')
ax1.plot(k * lambda_D, omega_kinetic / omega_pe, 'r:', linewidth=2, label='Kinetic (full)')
ax1.set_xlabel('$k\\lambda_D$', fontsize=13)
ax1.set_ylabel('$\\omega / \\omega_{pe}$', fontsize=13)
ax1.set_title('Langmuir Wave Dispersion', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.95, 1.4])

# Phase and group velocity
v_phase_warm = omega_warm / k
v_group_warm = 3 * k * v_th**2 / omega_warm

ax2.plot(k * lambda_D, v_phase_warm / v_th, 'b-', linewidth=2, label='$v_\\phi$ (Bohm-Gross)')
ax2.plot(k * lambda_D, v_group_warm / v_th, 'r--', linewidth=2, label='$v_g$ (Bohm-Gross)')
ax2.axhline(1, color='k', linestyle=':', alpha=0.5, label='$v_{th}$')
ax2.set_xlabel('$k\\lambda_D$', fontsize=13)
ax2.set_ylabel('Velocity / $v_{th}$', fontsize=13)
ax2.set_title('Phase and Group Velocity', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('langmuir_dispersion.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 6.2 이온 음향 파동 분산

```python
def ion_acoustic_dispersion(k, n, T_e, T_i, m_i, Z=1):
    """
    Ion acoustic wave dispersion with damping.

    Parameters:
    -----------
    k : array
        Wavenumber (m^-1)
    n : float
        Density (m^-3)
    T_e : float
        Electron temperature (eV)
    T_i : float
        Ion temperature (eV)
    m_i : float
        Ion mass (kg)
    Z : int
        Ion charge number

    Returns:
    --------
    omega_real : array
        Real frequency
    gamma : array
        Damping rate
    """
    T_e_J = T_e * e
    T_i_J = T_i * e

    # c_s는 T_i 대신 T_e를 사용합니다. 이온 음향 파동의 복원력이
    # 전자 압력이기 때문입니다: 뜨거운 전자가 Boltzmann 장벽을 형성하여 이온을 당겨 돌려보냅니다.
    c_s = np.sqrt(k_B * T_e_J / m_i)

    # Debye length
    v_th_e = np.sqrt(2 * k_B * T_e_J / m_e)
    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    lambda_D = v_th_e / omega_pe

    # 1/√(1 + k²λ_D²) 인자는 유한 Debye 길이 보정입니다: k λ_D → 1이 되면
    # 전하 차폐가 복원력을 약화시키고 ω가 포화하여
    # ω ~ ω_pe * √(m_e/m_i)를 넘지 않도록 합니다.
    omega_real = k * c_s / np.sqrt(1 + k**2 * lambda_D**2)

    # Damping rate (approximate, for T_e >> T_i)
    v_th_i = np.sqrt(2 * k_B * T_i_J / m_i)
    zeta_i = omega_real / (k * v_th_i)

    # 이온 Landau 감쇠는 T_i << T_e (즉, ζ_i >> 1)일 때만 지수적으로 작습니다.
    # (T_i/T_e)^(3/2) 앞 계수는 v ~ c_s에서 이온 꼬리 밀도가
    # T_i가 감소함에 따라 어떻게 줄어드는지 반영합니다 — 공명 이온이 적을수록 감쇠가 약해집니다.
    gamma = -np.sqrt(np.pi/8) * (omega_real / (k * lambda_D)) * \
            (T_i / T_e)**(3/2) * np.exp(-1/(2*k**2*lambda_D**2) - 3/2)

    return omega_real, gamma

# Parameters
n = 1e19  # m^-3
T_e = 10  # eV
m_i = 1.673e-27  # kg (proton)
T_i_array = np.array([0.1, 0.5, 1.0, 5.0])  # eV

# Compute Debye length
v_th_e = np.sqrt(2 * k_B * T_e * e / m_e)
omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
lambda_D = v_th_e / omega_pe
c_s = np.sqrt(k_B * T_e * e / m_i)

print(f"Ion acoustic speed: c_s = {c_s / 1e3:.1f} km/s")

k = np.linspace(0.01, 3, 200) / lambda_D

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = ['blue', 'green', 'orange', 'red']

for i, T_i in enumerate(T_i_array):
    omega_real, gamma = ion_acoustic_dispersion(k, n, T_e, T_i, m_i)

    ax1.plot(k * lambda_D, omega_real * lambda_D / c_s, color=colors[i],
             linewidth=2, label=f'$T_i = {T_i}$ eV')

    ax2.semilogy(k * lambda_D, np.abs(gamma) / omega_real, color=colors[i],
                 linewidth=2, label=f'$T_i = {T_i}$ eV')

# Add linear dispersion line
k_linear = k[k * lambda_D < 0.5]
ax1.plot(k_linear * lambda_D, k_linear * lambda_D, 'k--', alpha=0.5,
         label='$\\omega = kc_s$')

ax1.set_xlabel('$k\\lambda_D$', fontsize=13)
ax1.set_ylabel('$\\omega\\lambda_D / c_s$', fontsize=13)
ax1.set_title('Ion Acoustic Dispersion', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('$k\\lambda_D$', fontsize=13)
ax2.set_ylabel('$|\\gamma| / \\omega$', fontsize=13)
ax2.set_title('Landau Damping Rate', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([1e-6, 1])

plt.tight_layout()
plt.savefig('ion_acoustic_dispersion.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 6.3 Upper 및 Lower Hybrid 주파수

```python
def hybrid_frequencies(n, B, Z=1, A=1):
    """
    Compute upper and lower hybrid frequencies.

    Parameters:
    -----------
    n : array
        Density (m^-3)
    B : array
        Magnetic field (T)
    Z : int
        Ion charge
    A : float
        Ion mass number

    Returns:
    --------
    dict with frequencies
    """
    m_i = A * 1.673e-27

    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    omega_pi = np.sqrt(n * Z**2 * e**2 / (epsilon_0 * m_i))
    omega_ce = e * B / m_e
    omega_ci = Z * e * B / m_i

    # Upper hybrid 주파수는 플라즈마 주파수와 사이클로트론 주파수의 직교 합(quadrature sum)입니다.
    # 수직 전파에서 두 복원 메커니즘(공간 전하 진동과 Lorentz 힘)이
    # 독립적으로 작용하여 에너지가 단순히 더해지기 때문입니다.
    omega_UH = np.sqrt(omega_pe**2 + omega_ce**2)

    # lower hybrid 공식은 조화 합(harmonic addition, 1/ω²)을 사용합니다.
    # 두 극한 regime이 직렬로 기여하기 때문입니다: 낮은 밀도(ω_pi ≪ ω_ci)에서 이온은
    # 자화되어 ω_LH → √(ω_ci ω_ce)이고; 높은 밀도에서는 비자화되어 ω_LH → ω_pi입니다.
    term1 = 1 / (omega_ci * omega_ce)
    term2 = 1 / (omega_pi**2 + omega_ci**2)
    omega_LH = 1 / np.sqrt(term1 + term2)

    return {
        'omega_pe': omega_pe,
        'omega_pi': omega_pi,
        'omega_ce': omega_ce,
        'omega_ci': omega_ci,
        'omega_UH': omega_UH,
        'omega_LH': omega_LH
    }

# Parameter ranges
# 고정된 n에서 B를 스캔하면 두 주파수가 어떻게 다르게 스케일하는지 보여줍니다:
# ω_UH ~ √(ω_pe² + ω_ce²)는 낮은 B에서 ω_pe가 지배하고 높은 B에서 ω_ce가 지배하는 반면,
# ω_LH ~ √(ω_ci ω_ce)는 B에 대해 전체적으로 선형으로 증가합니다.
B_array = np.linspace(0.5, 10, 100)  # T
n_ref = 1e20  # m^-3

freq = hybrid_frequencies(n_ref, B_array, Z=1, A=2)  # Deuterium

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Upper hybrid
ax1.plot(B_array, freq['omega_pe'] / (2*np.pi) / 1e9, 'b--',
         linewidth=2, label='$f_{pe}$')
ax1.plot(B_array, freq['omega_ce'] / (2*np.pi) / 1e9, 'r--',
         linewidth=2, label='$f_{ce}$')
ax1.plot(B_array, freq['omega_UH'] / (2*np.pi) / 1e9, 'k-',
         linewidth=2.5, label='$f_{UH}$')
ax1.set_xlabel('Magnetic Field (T)', fontsize=13)
ax1.set_ylabel('Frequency (GHz)', fontsize=13)
ax1.set_title(f'Upper Hybrid Frequency ($n = {n_ref:.0e}$ m$^{{-3}}$)', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Lower hybrid
ax2.plot(B_array, freq['omega_ci'] / (2*np.pi) / 1e6, 'r--',
         linewidth=2, label='$f_{ci}$')
ax2.plot(B_array, freq['omega_LH'] / (2*np.pi) / 1e6, 'k-',
         linewidth=2.5, label='$f_{LH}$')
ax2.set_xlabel('Magnetic Field (T)', fontsize=13)
ax2.set_ylabel('Frequency (MHz)', fontsize=13)
ax2.set_title(f'Lower Hybrid Frequency ($n = {n_ref:.0e}$ m$^{{-3}}$)', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hybrid_frequencies.png', dpi=150, bbox_inches='tight')
plt.show()

# Print typical ITER values
B_ITER = 5.3  # T
n_ITER = 1e20  # m^-3
freq_ITER = hybrid_frequencies(n_ITER, B_ITER, Z=1, A=2)

print(f"\nITER-like parameters (B = {B_ITER} T, n = {n_ITER:.0e} m^-3):")
print(f"  f_pe = {freq_ITER['omega_pe'][0] / (2*np.pi) / 1e9:.1f} GHz")
print(f"  f_ce = {freq_ITER['omega_ce'][0] / (2*np.pi) / 1e9:.1f} GHz")
print(f"  f_UH = {freq_ITER['omega_UH'][0] / (2*np.pi) / 1e9:.1f} GHz")
print(f"  f_ci = {freq_ITER['omega_ci'][0] / (2*np.pi) / 1e6:.1f} MHz")
print(f"  f_LH = {freq_ITER['omega_LH'][0] / (2*np.pi) / 1e6:.1f} MHz")
```

### 6.4 Bernstein Mode 분산

```python
from scipy.special import iv  # Modified Bessel function

def bernstein_mode_approximate(n_harmonic, k_perp, rho_L, omega_c):
    """
    Approximate Bernstein mode dispersion near n-th harmonic.

    ω ≈ nω_c(1 + k²v_th²/(2ω_c²))
    """
    lambda_val = (k_perp * rho_L)**2
    omega = n_harmonic * omega_c * (1 + lambda_val / 2)
    return omega

# Parameters
B = 2.0  # T
T_e = 5e3  # eV (5 keV)
n = 5e19  # m^-3

omega_ce = e * B / m_e
v_th_e = np.sqrt(2 * k_B * T_e * e / m_e)
rho_L = v_th_e / omega_ce

print(f"Electron cyclotron frequency: f_ce = {omega_ce / (2*np.pi) / 1e9:.2f} GHz")
print(f"Larmor radius: ρ_L = {rho_L * 1e3:.2f} mm")

# Wavenumber
k_perp = np.linspace(0, 100, 500) / rho_L

fig, ax = plt.subplots(figsize=(10, 7))

harmonics = [1, 2, 3, 4, 5]
colors = plt.cm.viridis(np.linspace(0, 1, len(harmonics)))

for i, n_harm in enumerate(harmonics):
    omega = bernstein_mode_approximate(n_harm, k_perp, rho_L, omega_ce)
    ax.plot(k_perp * rho_L, omega / omega_ce, color=colors[i],
            linewidth=2, label=f'n = {n_harm}')

    # Mark the harmonic
    ax.axhline(n_harm, color=colors[i], linestyle='--', alpha=0.3)

ax.set_xlabel('$k_\\perp \\rho_L$', fontsize=13)
ax.set_ylabel('$\\omega / \\omega_{ce}$', fontsize=13)
ax.set_title('Electron Bernstein Mode Dispersion', fontsize=14)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 10])
ax.set_ylim([0, 6])

plt.tight_layout()
plt.savefig('bernstein_modes.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 요약

정전기 파동은 플라즈마의 기본 modes이며 다음으로 특징지어집니다:

**일반 프레임워크**:
- 분산 관계: $\epsilon(\mathbf{k}, \omega) = 0$
- 감수성으로부터의 유전 함수: $\epsilon = 1 + \sum_s \chi_s$
- 차가운 플라즈마: 대수적 분산
- 따뜻한 플라즈마: 플라즈마 분산 함수 $Z(\zeta)$를 통한 운동 효과

**Langmuir 파동**:
- 차가운 플라즈마: $\omega = \omega_{pe}$ (전파 없음)
- 따뜻한 플라즈마 (Bohm-Gross): $\omega^2 = \omega_{pe}^2 + 3k^2v_{th}^2$
- 군속도: $v_g = 3kv_{th}^2/\omega$
- $\omega/k \sim v_{th}$일 때 Landau damping

**이온 음향 파동**:
- 분산: $\omega = kc_s$ 여기서 $c_s = \sqrt{k_B T_e/m_i}$
- 전자 압력 복원력, 이온 관성
- $T_e \gg T_i$가 약한 감쇠를 요구
- 난류 및 수송에 중요

**자화 플라즈마 공명**:
- Upper hybrid: $\omega_{UH}^2 = \omega_{pe}^2 + \omega_{ce}^2$
- Lower hybrid: $\omega_{LH} \approx \sqrt{\omega_{ci}\omega_{ce}}$
- 플라즈마 가열 및 전류 구동에 사용

**Bernstein modes**:
- Cyclotron 고조파 근처의 정전기 modes: $\omega \approx n\omega_c$
- EM 파동이 차단되는 곳에 존재
- Mode 변환이 과밀 플라즈마에서 가열을 허용

응용은 다음을 포함합니다:
- 플라즈마 진단 (Thomson scattering, reflectometry)
- 가열 및 전류 구동 (ECRH, LHCD)
- 불안정성 및 난류 이해
- 천체물리 현상 (태양 전파 폭발, 펄사 방출)

## 연습 문제

### 문제 1: Langmuir 파동 특성
플라즈마가 $n = 5 \times 10^{18}$ m$^{-3}$와 $T_e = 5$ eV를 가집니다.

(a) 플라즈마 주파수 $f_{pe}$와 Debye 길이 $\lambda_D$를 계산하시오.

(b) $k = 0.1/\lambda_D$를 가진 Langmuir 파동에 대해, Bohm-Gross 관계를 사용하여 주파수를 계산하시오.

(c) 위상 속도 $v_\phi$와 군속도 $v_g$를 계산하시오. 이들이 $v_{th,e}$와 어떻게 비교됩니까?

(d) 어떤 파수에서 $\omega$에 대한 열 보정이 $\omega_{pe}$의 10%가 됩니까?

### 문제 2: 이온 음향 파동 감쇠
아르곤 플라즈마 ($A = 40$)가 $n = 10^{19}$ m$^{-3}$, $T_e = 20$ eV, $T_i = 2$ eV를 가집니다.

(a) 이온 음향 속도 $c_s$를 계산하시오.

(b) $k\lambda_D = 0.3$에 대해, 실수 주파수 $\omega$와 Landau damping 율 $\gamma$를 구하시오.

(c) 감쇠 길이 $L_d = v_g/|\gamma|$를 계산하시오. 여기서 $v_g = d\omega/dk$입니다.

(d) 이온 온도가 $T_i = 10$ eV로 증가하면, 감쇠는 어떻게 변합니까? 파동이 여전히 관찰 가능합니까?

### 문제 3: Hybrid 공명
토카막이 $B = 3$ T와 $n = 10^{20}$ m$^{-3}$ (코어)에서 $n = 10^{18}$ m$^{-3}$ (가장자리)로 변하는 밀도를 가집니다.

(a) 코어와 가장자리에서 $f_{UH}$를 계산하시오.

(b) 가열 시스템이 110 GHz에서 작동합니다. upper hybrid 공명 층은 어디에 위치합니까?

(c) 코어와 가장자리에서 $f_{LH}$를 계산하시오 (중수소를 가정).

(d) lower hybrid 시스템이 5 GHz에서 작동하는 경우, 반경의 함수로서 공명 층이 위치한 곳을 스케치하시오.

### 문제 4: Bernstein 파동 가열
$\omega_{pe} > \omega_{ce}$인 밀집 플라즈마를 고려하여, 플라즈마를 electron cyclotron 파동에 대해 과밀하게 만듭니다.

(a) $B = 1.5$ T, $T_e = 3$ keV에 대해, $\omega_{pe} < \omega_{ce}$인 최대 밀도를 구하시오.

(b) $n = 5 \times 10^{20}$ m$^{-3}$ (과밀)인 경우, $\omega_{UH}/\omega_{ce}$를 계산하시오.

(c) $\omega = \omega_{ce}$에서 O-mode 파동은 관통할 수 없습니다. upper hybrid 층에서 Bernstein 파동으로의 mode 변환이 어떻게 가열을 허용할 수 있는지 설명하시오.

(d) 첫 번째 고조파에서 효율적인 전자 가열에 필요한 Bernstein 파동의 수직 파수 $k_\perp \rho_L$을 추정하시오.

### 문제 5: 분산 관계 분석
전자-이온 플라즈마에 대한 정전기 분산 관계는:

$$1 - \frac{\omega_{pe}^2}{k^2 v_{th,e}^2}[1 + \zeta_e Z(\zeta_e)] - \frac{\omega_{pi}^2}{k^2 v_{th,i}^2}[1 + \zeta_i Z(\zeta_i)] = 0$$

여기서 $\zeta_s = \omega/(k v_{th,s})$입니다.

(a) $\omega \gg k v_{th,e}$ 한계에서, 이것이 Bohm-Gross 관계를 준다는 것을 보이기 위해 전개하시오.

(b) $\omega \ll k v_{th,e}$이지만 $\omega \sim k v_{th,i}$ 한계에서, 이것이 전자 Boltzmann 응답을 가진 이온 음향 파동을 준다는 것을 보이시오.

(c) $m_i/m_e = 1836$, $T_e = T_i = 1$ keV에 대해, $0.01 < k\lambda_D < 10$ 범위에서 $\omega(k)$에 대해 수치적으로 풀이하시오. Langmuir 및 이온 음향 분기를 식별하시오.

(d) 어떤 $k\lambda_D$ 값에서 두 분기가 비교할 만한 주파수를 가집니까? 이 regime에서 어떤 물리가 발생합니까?

---

**이전**: [9. Collisional Kinetics](./09_Collisional_Kinetics.md)
**다음**: [11. Electromagnetic Waves](./11_Electromagnetic_Waves.md)
