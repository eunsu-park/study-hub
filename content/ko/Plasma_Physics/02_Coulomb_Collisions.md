# 2. Coulomb 충돌

## 학습 목표

- 전하 입자 간의 Coulomb 산란 물리를 이해하고 Rutherford 단면적 유도하기
- 전자-전자, 이온-이온, 전자-이온 상호작용에 대한 충돌 주파수 계산하기
- Coulomb 로그를 유도하고 충돌 속도 결정에서의 역할 이해하기
- Spitzer 저항률을 계산하고 온도 의존성 이해하기
- Knudsen 수를 사용하여 충돌성 및 무충돌성 플라즈마 영역 구별하기
- Python 도구를 적용하여 충돌 역학 및 수송 특성 분석하기

## 1. Coulomb 산란

### 1.1 이체 문제

전하 $q_1$과 $q_2$, 질량 $m_1$과 $m_2$를 가진 두 전하 입자가 Coulomb 힘을 통해 상호작용하는 경우를 고려합시다. 질량 중심 좌표계에서 이것은 환산 질량을 가진 단일 입자 문제로 축소됩니다:

$$\mu = \frac{m_1 m_2}{m_1 + m_2}$$

입자는 Coulomb 퍼텐셜에서 움직입니다:

$$V(r) = \frac{1}{4\pi\epsilon_0} \frac{q_1 q_2}{r}$$

### 1.2 산란 기하학

```
Scattering Geometry (Lab Frame):

              b (impact parameter)
              ↓
    ●→→→→→→→→→●→→→→→→→
   particle 1  ↑  scattered particle
            target
         (particle 2)

              χ = scattering angle

Classical orbit:
- Hyperbolic trajectory for repulsive force
- Deflection angle χ depends on impact parameter b
- Small b → large deflection
- Large b → small deflection
```

**척력** Coulomb 상호작용($q_1 q_2 > 0$)의 경우, 산란각 $\chi$는 충격 매개변수 $b$와 다음 관계에 있습니다:

$$\tan\left(\frac{\chi}{2}\right) = \frac{b_{90}}{b}$$

여기서 $b_{90}$는 **90° 산란의 충격 매개변수**입니다:

$$b_{90} = \frac{1}{4\pi\epsilon_0} \frac{q_1 q_2}{\mu v_0^2} = \frac{q_1 q_2}{4\pi\epsilon_0 E_{cm}}$$

여기서 $v_0$는 초기 상대 속도이고 $E_{cm} = \frac{1}{2}\mu v_0^2$는 질량 중심 운동 에너지입니다.

**물리적 해석:** $b_{90}$는 Coulomb 퍼텐셜 에너지가 운동 에너지와 같아지는 충격 매개변수입니다:

$$\frac{1}{4\pi\epsilon_0}\frac{q_1 q_2}{b_{90}} = \frac{1}{2}\mu v_0^2$$

### 1.3 Rutherford 단면적

**미분 단면적**은 각도 $\chi$에서 입체각 $d\Omega$으로 산란될 확률을 설명합니다:

$$\frac{d\sigma}{d\Omega} = \left(\frac{b_{90}}{2}\right)^2 \frac{1}{\sin^4(\chi/2)}$$

이것이 고전 산란 이론의 가장 중요한 결과 중 하나인 **Rutherford 산란 공식**입니다.

**주요 특징:**
1. 강한 전방 편향: $\chi \to 0$일 때 $d\sigma/d\Omega \to \infty$ (작은 각도 산란이 지배적)
2. 대칭성: $|\chi|$에만 의존
3. 발산: 전체 단면적 $\sigma_{total} = \int (d\sigma/d\Omega) d\Omega$가 발산!

발산은 다음과 같은 이유로 발생합니다:
- 작은 각도 산란($\chi \ll 1$)이 지배적
- 장거리 Coulomb 힘은 임의로 큰 충격 매개변수를 허용
- 많은 약한 편향이 드문 큰 편향보다 더 중요

### 1.4 운동량 전달 단면적

수송 특성의 경우, $(1 - \cos\chi)$로 가중된 **운동량 전달 단면적**이 필요합니다:

$$\sigma_m = \int (1 - \cos\chi) \frac{d\sigma}{d\Omega} d\Omega$$

이 적분도 발산하지만, 발산은 더 완만합니다($\ln b_{max}$). Debye 차폐로 곧 이를 다루겠습니다.

## 2. Coulomb 로그

### 2.1 Coulomb 상호작용의 차단

Rutherford 공식은 무한 범위의 차폐되지 않은 Coulomb 퍼텐셜을 가정합니다. 플라즈마에서는 두 물리적 효과가 차단을 제공합니다:

**1. 최대 충격 매개변수($b_{max}$):** Debye 차폐

거리 $b > \lambda_D$에서 Coulomb 퍼텐셜은 지수적으로 차폐됩니다. 따라서:

$$b_{max} \sim \lambda_D = \sqrt{\frac{\epsilon_0 k_B T}{n e^2}}$$

**2. 최소 충격 매개변수($b_{min}$):** 고전적 최근접 거리 또는 양자 불확정성

최소 충격 매개변수는 다음 중 큰 값입니다:

(a) **고전적 최근접** $b_{90}$ (퍼텐셜 에너지가 운동 에너지와 같을 때)

(b) **양자 de Broglie 파장** $\lambda_{dB} = \hbar/(mv_{thermal})$ (파동 효과가 중요할 때)

대부분의 플라즈마의 경우, 고전적 한계가 지배합니다:

$$b_{min} \sim b_{90} = \frac{q_1 q_2}{4\pi\epsilon_0 \mu v_{th}^2}$$

여기서 $v_{th} = \sqrt{k_B T/m}$는 열속도입니다.

### 2.2 Coulomb 로그의 정의

**Coulomb 로그**는 다음과 같이 정의됩니다:

$$\ln\Lambda = \ln\left(\frac{b_{max}}{b_{min}}\right)$$

**전자-이온 충돌의 경우:**

$$\ln\Lambda_{ei} \approx \ln\left(\frac{12\pi n_e \lambda_D^3}{Z}\right)$$

$\lambda_D = \sqrt{\epsilon_0 k_B T_e/(n_e e^2)}$를 사용하면:

$$\ln\Lambda_{ei} \approx \begin{cases}
23 - \ln\left(\sqrt{n_e/10^6} \, Z \, T_e^{-3/2}\right) & T_e < 10 Z^2 \text{ eV} \\
24 - \ln\left(\sqrt{n_e/10^6} \, T_e^{-1}\right) & T_e > 10 Z^2 \text{ eV}
\end{cases}$$

여기서 $n_e$는 cm$^{-3}$ 단위이고 $T_e$는 eV 단위입니다.

**일반적인 값:**
- 실험실 플라즈마: $\ln\Lambda \approx 10 - 15$
- 핵융합 플라즈마: $\ln\Lambda \approx 15 - 20$
- 천체물리 플라즈마: $\ln\Lambda \approx 20 - 30$

```
Physical Meaning of ln Λ:

ln Λ ≈ ln(number of particles in Debye sphere) ≈ ln(N_D)

    b_max ~ λ_D
       ↓
  ●───●───●───●     Debye sphere
  ●───●───●───●     contains ~N_D particles
  ●───●───●───●
       ↑
    b_min ~ b_90

ln Λ counts the "effective range" of Coulomb interactions
in units of logarithms (weak dependence on plasma parameters)
```

### 2.3 약한 로그 의존성

$\ln\Lambda$의 놀라운 특징은 플라즈마 매개변수에 대한 **약한 의존성**입니다. $n$과 $T$가 다른 플라즈마에서 많은 차수만큼 변하더라도 $\ln\Lambda$은 2-3배만 변합니다.

| Plasma | $n$ [m$^{-3}$] | $T$ [eV] | $\ln\Lambda$ |
|--------|----------------|----------|--------------|
| Tokamak core | $10^{20}$ | 10,000 | 17 |
| Tokamak edge | $10^{19}$ | 100 | 15 |
| Solar corona | $10^{14}$ | 100 | 19 |
| Ionosphere |  $10^{12}$ | 0.1 | 12 |
| Glow discharge | $10^{16}$ | 2 | 10 |

이 약한 의존성 덕분에 많은 추정에서 $\ln\Lambda \approx 15$를 상수로 취급할 수 있습니다.

## 3. 충돌 주파수

### 3.1 운동량 전달 충돌 주파수

**충돌 주파수** $\nu$는 입자가 운동량 변화 충돌을 겪는 속도입니다. 다음과 같이 정의됩니다:

$$\nu = n \sigma_m v_{th}$$

여기서 $\sigma_m$은 운동량 전달 단면적입니다.

Coulomb 충돌의 경우, 차단을 가진 Rutherford 공식을 적분하면:

$$\sigma_m \sim \pi b_{90}^2 \ln\Lambda$$

### 3.2 전자-이온 충돌 주파수

**전자-이온 충돌 주파수**는 다음과 같습니다:

$$\nu_{ei} = \frac{n_i Z^2 e^4 \ln\Lambda}{4\pi\epsilon_0^2 m_e^2 v_e^3}$$

전자 열속도 $v_e = \sqrt{k_B T_e/m_e}$를 사용하면:

$$\nu_{ei} = \frac{n_i Z^2 e^4 \ln\Lambda}{4\pi\epsilon_0^2 m_e^{1/2} (k_B T_e)^{3/2}}$$

**수치 공식:**

$$\nu_{ei} \approx 2.91 \times 10^{-6} \, \frac{n_e[\text{m}^{-3}] \, Z \, \ln\Lambda}{T_e[\text{eV}]^{3/2}} \quad [\text{s}^{-1}]$$

**주요 스케일링:** $\nu_{ei} \propto n T^{-3/2}$

- 충돌은 밀도에 따라 증가(더 많은 표적)
- 충돌은 온도에 따라 급격히 감소(더 빠른 입자는 충돌 영역에서 더 적은 시간을 보냄)

### 3.3 전자-전자 충돌 주파수

**같은 입자 충돌**(전자-전자 또는 이온-이온)의 경우, 운동학이 다릅니다. 충돌 주파수는 다음과 같습니다:

$$\nu_{ee} \approx \nu_{ei}$$

(1차 수치 인수를 가진 동일한 차수).

더 정확하게:

$$\nu_{ee} = \frac{n_e e^4 \ln\Lambda}{8\pi\epsilon_0^2 m_e^{1/2} (k_B T_e)^{3/2}} \approx 1.45 \times 10^{-6} \, \frac{n_e[\text{m}^{-3}] \, \ln\Lambda}{T_e[\text{eV}]^{3/2}} \quad [\text{s}^{-1}]$$

### 3.4 이온-이온 충돌 주파수

마찬가지로 이온의 경우:

$$\nu_{ii} = \frac{n_i Z^4 e^4 \ln\Lambda}{8\pi\epsilon_0^2 m_i^{1/2} (k_B T_i)^{3/2}}$$

**전자 충돌 주파수와의 비교:**

$$\frac{\nu_{ii}}{\nu_{ei}} \sim \sqrt{\frac{m_e}{m_i}} \left(\frac{T_e}{T_i}\right)^{3/2}$$

$T_e \sim T_i$이고 수소 플라즈마의 경우:

$$\nu_{ii} \sim \frac{\nu_{ei}}{43}$$

이온은 전자보다 훨씬 덜 자주 충돌합니다(느린 열속도).

### 3.5 충돌 주파수의 순서

$T_e \sim T_i$이고 $m_i \gg m_e$인 일반적인 플라즈마의 경우:

$$\nu_{ee} \sim \nu_{ei} \gg \nu_{ie} \gg \nu_{ii}$$

여기서 $\nu_{ie}$는 이온-전자 충돌 주파수입니다.

**물리적 해석:**
- 전자는 전자 및 이온과 자주 충돌
- 이온은 주로 이온과 충돌; 전자는 너무 빠르고 가벼워서 이온을 크게 편향시키지 못함
- 종 간의 운동량 및 에너지 교환은 느린 이온 시간 스케일에서 발생

## 4. Spitzer 저항률

### 4.1 충돌 주파수로부터의 유도

전기 저항률은 전류를 운반하는 전자와 정지해 있는 이온 사이의 운동량 전달로부터 발생합니다.

간단한 Drude 모델에서 전도도는 다음과 같습니다:

$$\sigma_{\parallel} = \frac{n_e e^2}{m_e \nu_{ei}}$$

**Spitzer 저항률**은 다음과 같습니다:

$$\eta = \frac{1}{\sigma_{\parallel}} = \frac{m_e \nu_{ei}}{n_e e^2}$$

$\nu_{ei}$를 대입하면:

$$\eta = \frac{Z \, m_e e^2 \ln\Lambda}{4\pi\epsilon_0^2 (k_B T_e)^{3/2}}$$

**수치 공식:**

$$\eta \approx 5.2 \times 10^{-5} \, \frac{Z \, \ln\Lambda}{T_e[\text{eV}]^{3/2}} \quad [\Omega \cdot \text{m}]$$

### 4.2 온도 의존성

핵심 결과는 **강한 온도 의존성**입니다:

$$\eta \propto T_e^{-3/2}$$

**의미:**
- 뜨거운 플라즈마는 훌륭한 도체(낮은 저항률)
- 저항률은 가열에 따라 급격히 감소
- 핵융합 플라즈마($T_e \sim 10$ keV)의 경우, $\eta \sim 10^{-8}$ Ω·m (실온 구리와 비슷!)

```
Resistivity vs Temperature:

η [Ω⋅m]
 ↑
10⁻⁴│         .
     │       .
10⁻⁵│      .
     │    .
10⁻⁶│   .
     │  .
10⁻⁷│ .
     │.
10⁻⁸│
     └─────────────────→ T_e [eV]
     10   100   1000  10000

Spitzer: η ∝ T^(-3/2)
```

### 4.3 충돌 가열

저항률은 전기 에너지를 열로 소산시킵니다. 출력 밀도는 다음과 같습니다:

$$P_{Ohmic} = \eta J^2 = \eta \frac{I^2}{A^2}$$

여기서 $J$는 전류 밀도이고 $I$는 총 전류입니다.

토카막에서 **Ohmic 가열**은 낮은 온도에서 지배적이지만 $T^{-3/2}$ 스케일링으로 인해 높은 $T$에서는 비효과적이 됩니다.

### 4.4 고전 저항률과의 비교

Spitzer 저항률을 고전 금속과 비교:

| Material | $\eta$ [Ω·m] at 300 K |
|----------|------------------------|
| Copper | $1.7 \times 10^{-8}$ |
| Aluminum | $2.7 \times 10^{-8}$ |
| Plasma ($T_e=10$ keV) | $\sim 10^{-8}$ |

뜨거운 플라즈마는 금속만큼 전도성이 좋습니다! 그러나 물리적 메커니즘은 다릅니다:
- 금속: 전자-포논 산란
- 플라즈마: 전자-이온 Coulomb 충돌

## 5. 평균 자유 경로 및 충돌성

### 5.1 평균 자유 경로

**평균 자유 경로** $\lambda_{mfp}$는 입자가 충돌 사이에 이동하는 평균 거리입니다:

$$\lambda_{mfp} = \frac{v_{th}}{\nu}$$

**전자의 경우:**

$$\lambda_{mfp,e} = \frac{v_{te}}{\nu_{ei}} = \frac{\sqrt{k_B T_e/m_e}}{\nu_{ei}}$$

**수치 추정:**

$$\lambda_{mfp,e} \approx 3.44 \times 10^{11} \, \frac{T_e[\text{eV}]^2}{n_e[\text{m}^{-3}] \, Z \, \ln\Lambda} \quad [\text{m}]$$

### 5.2 Knudsen 수

**Knudsen 수**는 평균 자유 경로를 시스템 크기 $L$과 비교합니다:

$$Kn = \frac{\lambda_{mfp}}{L}$$

**충돌성 영역:**

- **충돌성(유체 유사):** $Kn \ll 1$
  - 시스템 크기 내에서 많은 충돌 발생
  - 국소 열역학적 평형(LTE)
  - 유체(MHD) 설명이 유효

- **무충돌성(운동학적):** $Kn \gg 1$
  - 시스템 크기 내에서 충돌이 거의 없거나 전혀 없음
  - 분포 함수가 비Maxwellian
  - 운동학적(Vlasov) 설명 필요

- **전이:** $Kn \sim 1$
  - 어느 한계도 적용 불가
  - 모델링하기 가장 어려운 영역

### 5.3 예제

**Tokamak 중심부:**
- $n_e = 10^{20}$ m$^{-3}$, $T_e = 10$ keV, $L = 1$ m, $\ln\Lambda = 17$
- $\nu_{ei} \approx 1.7 \times 10^4$ s$^{-1}$
- $v_{te} \approx 4.2 \times 10^7$ m/s
- $\lambda_{mfp} \approx 2500$ m $\gg L$
- $Kn \approx 2500$ → **무충돌성**

**자기권 플라즈마:**
- $n \sim 10^6$ m$^{-3}$, $T \sim 1$ keV, $L \sim 10^7$ m
- $\lambda_{mfp} \sim 10^{15}$ m $\gg L$
- 극도로 무충돌성

**글로우 방전:**
- $n \sim 10^{16}$ m$^{-3}$, $T \sim 2$ eV, $L \sim 0.1$ m
- $\lambda_{mfp} \sim 1$ m $\gtrsim L$
- 전이 영역

## 6. 에너지 균등분배

### 6.1 종 간 에너지 교환

$T_e \ne T_i$일 때, 충돌을 통해 종 사이에 에너지가 전달됩니다. **에너지 교환 주파수**는 다음과 같습니다:

$$\nu_{E,ei} = \frac{2m_e}{m_i} \nu_{ei}$$

인자 $2m_e/m_i \ll 1$은 질량이 매우 다른 입자 간 충돌에서의 비효율적인 에너지 전달을 반영합니다.

### 6.2 균등분배 시간

**균등분배 시간**은 온도가 평형을 이루는 시간입니다:

$$\tau_{eq} = \frac{1}{\nu_{E,ei}} = \frac{m_i}{2m_e \nu_{ei}}$$

**수치 추정:**

$$\tau_{eq} \approx 1.09 \times 10^{13} \, \frac{A \, T_e[\text{eV}]^{3/2}}{n_e[\text{m}^{-3}] \, Z \, \ln\Lambda} \quad [\text{s}]$$

여기서 $A$는 이온 질량수입니다.

토카막의 수소 플라즈마($A=1$)의 경우:
- $n_e = 10^{20}$ m$^{-3}$, $T_e = 10$ keV
- $\tau_{eq} \approx 1$ s

이것은 에너지 제약 시간($\sim 0.1$ s)에 비해 **길기** 때문에 $T_e$와 $T_i$가 크게 다를 수 있습니다.

### 6.3 전자 대 이온 가열

핵융합 실험에서:
- **전자 가열**(예: ECRH, Ohmic): $T_e$를 직접 가열
- **이온 가열**(예: 중성빔 주입, ICRH): $T_i$를 직접 가열

느린 균등분배($\tau_{eq} \gg \tau_E$)로 인해 $T_e$와 $T_i$의 별도 제어가 가능합니다. 일반적인 토카막 프로파일은 중심부에서 $T_e \gtrsim T_i$를 보여줍니다.

## 7. 계산 예제

### 7.1 충돌 주파수 계산기

```python
import numpy as np
import matplotlib.pyplot as plt

# Constants
e = 1.602176634e-19
m_e = 9.1093837015e-31
m_p = 1.672621898e-27
epsilon_0 = 8.8541878128e-12
k_B = 1.380649e-23
eV_to_K = 11604.518

def coulomb_logarithm(n_e, T_e, Z=1):
    """
    Calculate Coulomb logarithm.

    Parameters:
    -----------
    n_e : float
        Electron density [m^-3]
    T_e : float
        Electron temperature [eV]
    Z : int
        Ion charge state

    Returns:
    --------
    ln_Lambda : float
        Coulomb logarithm
    """
    # NRL Plasma Formulary는 ln_Lambda를 CGS(cm^-3) 단위로 정의하므로 변환이 필요하다;
    # SI 공식을 그대로 사용하면 상수가 달라져 잘못된 결과가 나온다.
    n_e_cgs = n_e * 1e-6  # Convert to cm^-3

    # 두 영역을 구분하는 이유는 최소 충격 매개변수(b_min)가 달라지기 때문이다:
    # 낮은 T_e(< 10 Z^2 eV)에서는 b_min이 고전적 90° 산란 길이 b_90이고,
    # 높은 T_e에서는 b_min이 de Broglie 파장(양자 한계)으로 전환된다.
    # 잘못된 분기를 사용하면 ln_Lambda가 ~1-2 과대평가되고,
    # 이 오차는 충돌 주파수와 저항률에 직접 전파된다.
    if T_e < 10 * Z**2:
        ln_Lambda = 23 - np.log(np.sqrt(n_e_cgs) * Z * T_e**(-1.5))
    else:
        ln_Lambda = 24 - np.log(np.sqrt(n_e_cgs) * T_e**(-1))

    return ln_Lambda

def nu_ei(n_e, T_e, Z=1, ln_Lambda=None):
    """
    Electron-ion collision frequency.

    Returns: frequency [s^-1]
    """
    if ln_Lambda is None:
        ln_Lambda = coulomb_logarithm(n_e, T_e, Z)

    # T_e^(-3/2) 스케일링이 핵심 물리다: 더 빠른 전자는 이온 근처에서
    # 머무는 시간이 짧아 유효 단면적이 ∝ v^(-4) ∝ T^(-2)가 되고,
    # 열속도의 T^(1/2) 증가분이 상쇄되어 결과적으로 T^(-3/2)가 된다.
    # 이것이 바로 토카막 Ohmic 가열이 ~1 keV 이상에서 비효율적이 되는 이유다.
    return 2.91e-6 * n_e * Z * ln_Lambda / T_e**1.5

def nu_ee(n_e, T_e, ln_Lambda=None):
    """
    Electron-electron collision frequency.

    Returns: frequency [s^-1]
    """
    if ln_Lambda is None:
        ln_Lambda = coulomb_logarithm(n_e, T_e)

    # nu_ee ~ nu_ei / 2 수치적으로; 동종 입자(like-particle) 충돌은 질량 중심(center-of-mass)
    # 계에서 운동량을 덜 효율적으로 전달하므로 전치인수가 더 작지만(1.45 vs 2.91),
    # T^(-3/2) 물리는 동일하게 적용된다.
    return 1.45e-6 * n_e * ln_Lambda / T_e**1.5

def nu_ii(n_i, T_i, Z=1, A=1, ln_Lambda=None):
    """
    Ion-ion collision frequency.

    Returns: frequency [s^-1]
    """
    if ln_Lambda is None:
        ln_Lambda = coulomb_logarithm(n_i, T_i, Z)

    # sqrt(m_e / m_i) 인수는 이온이 sqrt(m_i/m_e)만큼 느리다는 사실을 반영한다:
    # nu_ii / nu_ei ~ sqrt(m_e/m_i) ~ 1/43 (수소 기준) — 이온은 훨씬 드물게 충돌하므로
    # 이온 수송(transport)은 더 느린 시간 스케일에 지배된다.
    factor = 1.45e-6 * np.sqrt(m_e / (A * m_p))
    return factor * n_i * Z**4 * ln_Lambda / T_i**1.5

def spitzer_resistivity(T_e, Z=1, ln_Lambda=None):
    """
    Spitzer resistivity.

    Parameters:
    -----------
    T_e : float
        Electron temperature [eV]
    Z : int
        Ion charge
    ln_Lambda : float, optional
        Coulomb logarithm

    Returns:
    --------
    eta : float
        Resistivity [Ohm*m]
    """
    # ln_Lambda가 주어지지 않으면 n_e 없이도 사용할 수 있도록 표준 추정값 15를 사용한다;
    # 저항률은 ln_Lambda에 로그적으로만 의존하므로 고정값 사용 시
    # 대부분의 실험실 및 핵융합 플라즈마에서 오차는 30% 미만이다.
    if ln_Lambda is None:
        ln_Lambda = 15  # Typical value

    # eta ∝ T_e^(-3/2)는 nu_ei ∝ T_e^(-3/2)의 직접적인 결과다:
    # 더 뜨거운 플라즈마에서 전자가 덜 자주 산란되므로 전도도가 더 좋다.
    # T_e ~ 10 keV(핵융합 관련)에서 eta는 실온 구리 수준에 근접하여
    # Ohmic 전류 구동은 가능하나 Ohmic 가열은 무시할 만해진다.
    return 5.2e-5 * Z * ln_Lambda / T_e**1.5

def mean_free_path(n_e, T_e, Z=1, ln_Lambda=None):
    """
    Electron mean free path.

    Returns: lambda_mfp [m]
    """
    if ln_Lambda is None:
        ln_Lambda = coulomb_logarithm(n_e, T_e, Z)

    # lambda_mfp = v_te / nu_ei ∝ T_e^2 / n: T^2 스케일링 덕분에 온도가 조금만
    # 올라가도 평균 자유 경로(mean free path)가 급격히 늘어난다. 이 때문에
    # 핵융합급 플라즈마(T ~ 10 keV)는 장치 크기 대비 사실상 무충돌(collisionless)이다.
    return 3.44e11 * T_e**2 / (n_e * Z * ln_Lambda)

# Demonstration
if __name__ == "__main__":
    print("="*70)
    print("COLLISION FREQUENCY ANALYSIS")
    print("="*70)

    # Example: Tokamak parameters
    n_e = 1e20  # m^-3
    T_e = 10000  # eV
    T_i = 8000   # eV
    Z = 1
    A = 2  # Deuterium

    ln_Lambda = coulomb_logarithm(n_e, T_e, Z)

    print(f"\nPlasma Parameters:")
    print(f"  n_e = {n_e:.2e} m^-3")
    print(f"  T_e = {T_e:.0f} eV")
    print(f"  T_i = {T_i:.0f} eV")
    print(f"  Z   = {Z}, A = {A}")
    print(f"  ln Λ = {ln_Lambda:.2f}")
    print("-"*70)

    nu_ei_val = nu_ei(n_e, T_e, Z, ln_Lambda)
    nu_ee_val = nu_ee(n_e, T_e, ln_Lambda)
    nu_ii_val = nu_ii(n_e, T_i, Z, A, ln_Lambda)

    print(f"\nCollision Frequencies:")
    print(f"  ν_ei = {nu_ei_val:.3e} s^-1  (period: {1/nu_ei_val:.3e} s)")
    print(f"  ν_ee = {nu_ee_val:.3e} s^-1  (period: {1/nu_ee_val:.3e} s)")
    print(f"  ν_ii = {nu_ii_val:.3e} s^-1  (period: {1/nu_ii_val:.3e} s)")
    print(f"  Ratio ν_ei/ν_ii = {nu_ei_val/nu_ii_val:.1f}")
    print("-"*70)

    eta = spitzer_resistivity(T_e, Z, ln_Lambda)
    print(f"\nSpitzer Resistivity:")
    print(f"  η = {eta:.3e} Ω·m")
    print(f"  (Copper at 300 K: 1.7e-8 Ω·m)")
    print("-"*70)

    lambda_mfp = mean_free_path(n_e, T_e, Z, ln_Lambda)
    v_te = np.sqrt(k_B * T_e * eV_to_K / m_e)

    print(f"\nMean Free Path:")
    print(f"  λ_mfp = {lambda_mfp:.2e} m")
    print(f"  v_te  = {v_te:.3e} m/s")
    print(f"  For system size L = 1 m:")
    print(f"    Knudsen number Kn = {lambda_mfp/1:.0f}")
    print(f"    Regime: {'Collisionless' if lambda_mfp > 1 else 'Collisional'}")
    print("-"*70)

    # Energy equipartition time
    # tau_eq의 m_i / (2 m_e) 인수는 가벼운 전자와 무거운 이온 사이의
    # 에너지 전달 비효율성을 반영한다: 충돌 후 전자 운동 에너지의 대부분이
    # 돌아오므로 온도 평형에는 많은 충돌이 필요하다 — 이 때문에 보조 가열
    # 토카막에서 상대적으로 높은 밀도에서도 T_e ≠ T_i인 경우가 흔하다.
    tau_eq = (A * m_p) / (2 * m_e * nu_ei_val)
    print(f"\nEnergy Equipartition:")
    print(f"  τ_eq = {tau_eq:.3e} s = {tau_eq*1000:.1f} ms")
    print("="*70)
```

### 7.2 온도에 대한 저항률

```python
def plot_resistivity_vs_temperature():
    """Plot Spitzer resistivity as a function of temperature."""

    T_vals = np.logspace(0, 4, 100)  # 1 eV to 10 keV
    Z_vals = [1, 2, 6]  # H, He, C

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Linear-log plot
    for Z in Z_vals:
        eta_vals = [spitzer_resistivity(T, Z, ln_Lambda=15) for T in T_vals]
        ax1.loglog(T_vals, eta_vals, linewidth=2, label=f'Z={Z}')

    # Add reference: T^(-3/2) scaling
    eta_ref = spitzer_resistivity(10, Z=1) * (T_vals/10)**(-1.5)
    ax1.loglog(T_vals, eta_ref, 'k--', alpha=0.5, linewidth=1.5,
               label=r'$\propto T^{-3/2}$')

    # Copper resistivity (room temp)
    ax1.axhline(y=1.7e-8, color='brown', linestyle=':', linewidth=2,
                label='Copper (300 K)')

    ax1.set_xlabel(r'Temperature $T_e$ [eV]', fontsize=12)
    ax1.set_ylabel(r'Resistivity $\eta$ [Ω·m]', fontsize=12)
    ax1.set_title('Spitzer Resistivity vs Temperature', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 1e4)
    ax1.set_ylim(1e-9, 1e-4)

    # Conductivity plot
    for Z in Z_vals:
        sigma_vals = [1/spitzer_resistivity(T, Z, ln_Lambda=15) for T in T_vals]
        ax2.loglog(T_vals, sigma_vals, linewidth=2, label=f'Z={Z}')

    ax2.set_xlabel(r'Temperature $T_e$ [eV]', fontsize=12)
    ax2.set_ylabel(r'Conductivity $\sigma$ [S/m]', fontsize=12)
    ax2.set_title('Electrical Conductivity vs Temperature', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, 1e4)

    plt.tight_layout()
    plt.savefig('spitzer_resistivity.png', dpi=150)
    plt.show()

plot_resistivity_vs_temperature()
```

### 7.3 충돌 주파수 스케일링

```python
def plot_collision_frequency_scaling():
    """Visualize scaling of collision frequencies with n and T."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Scan 1: Vary density at fixed T
    n_vals = np.logspace(14, 22, 100)
    T_fixed = 100  # eV

    nu_ei_vals = [nu_ei(n, T_fixed) for n in n_vals]
    nu_ee_vals = [nu_ee(n, T_fixed) for n in n_vals]

    ax = axes[0, 0]
    ax.loglog(n_vals, nu_ei_vals, 'b-', linewidth=2, label=r'$\nu_{ei}$')
    ax.loglog(n_vals, nu_ee_vals, 'r--', linewidth=2, label=r'$\nu_{ee}$')
    ax.set_xlabel(r'Density $n_e$ [m$^{-3}$]', fontsize=11)
    ax.set_ylabel(r'Collision Frequency [s$^{-1}$]', fontsize=11)
    ax.set_title(f'Collision Frequency vs Density (T={T_fixed} eV)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Scan 2: Vary temperature at fixed n
    T_vals = np.logspace(0, 4, 100)
    n_fixed = 1e19  # m^-3

    nu_ei_vals = [nu_ei(n_fixed, T) for T in T_vals]
    nu_ee_vals = [nu_ee(n_fixed, T) for T in T_vals]

    ax = axes[0, 1]
    ax.loglog(T_vals, nu_ei_vals, 'b-', linewidth=2, label=r'$\nu_{ei}$')
    ax.loglog(T_vals, nu_ee_vals, 'r--', linewidth=2, label=r'$\nu_{ee}$')

    # Reference line: T^(-3/2)
    nu_ref = nu_ei(n_fixed, 100) * (T_vals/100)**(-1.5)
    ax.loglog(T_vals, nu_ref, 'k:', linewidth=1.5, alpha=0.7,
              label=r'$\propto T^{-3/2}$')

    ax.set_xlabel(r'Temperature $T_e$ [eV]', fontsize=11)
    ax.set_ylabel(r'Collision Frequency [s$^{-1}$]', fontsize=11)
    ax.set_title(f'Collision Frequency vs Temperature (n={n_fixed:.0e} m⁻³)',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Scan 3: Mean free path vs temperature
    lambda_vals = [mean_free_path(n_fixed, T) for T in T_vals]

    ax = axes[1, 0]
    ax.loglog(T_vals, lambda_vals, 'g-', linewidth=2)
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5,
               label='L = 1 m (device size)')
    ax.set_xlabel(r'Temperature $T_e$ [eV]', fontsize=11)
    ax.set_ylabel(r'Mean Free Path $\lambda_{mfp}$ [m]', fontsize=11)
    ax.set_title(f'Mean Free Path vs Temperature (n={n_fixed:.0e} m⁻³)',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Scan 4: Coulomb logarithm landscape
    n_range = np.logspace(14, 22, 50)
    T_range = np.logspace(0, 4, 50)
    N, T = np.meshgrid(n_range, T_range)

    ln_Lambda_map = np.zeros_like(N)
    for i in range(len(T_range)):
        for j in range(len(n_range)):
            ln_Lambda_map[i, j] = coulomb_logarithm(N[i, j], T[i, j])

    ax = axes[1, 1]
    contour = ax.contourf(N, T, ln_Lambda_map, levels=20, cmap='viridis')
    cbar = plt.colorbar(contour, ax=ax, label=r'$\ln\Lambda$')

    cs = ax.contour(N, T, ln_Lambda_map, levels=[10, 15, 20, 25],
                    colors='white', linewidths=1.5, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=9)

    ax.set_xlabel(r'Density $n_e$ [m$^{-3}$]', fontsize=11)
    ax.set_ylabel(r'Temperature $T_e$ [eV]', fontsize=11)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Coulomb Logarithm Landscape', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('collision_frequency_scaling.png', dpi=150)
    plt.show()

plot_collision_frequency_scaling()
```

### 7.4 충돌성 맵

```python
def plot_collisionality_map():
    """
    Create a map showing collisional vs collisionless regimes
    for various system sizes.
    """
    n_range = np.logspace(14, 24, 100)
    T_range = np.logspace(0, 4, 100)
    N, T = np.meshgrid(n_range, T_range)

    # Mean free path
    lambda_mfp_map = 3.44e11 * T**2 / (N * 15)  # ln Λ ≈ 15

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Mean free path contours
    ax = axes[0]
    levels = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1e4]
    contour = ax.contourf(N, T, lambda_mfp_map, levels=levels,
                          cmap='RdYlGn', norm=plt.matplotlib.colors.LogNorm())
    cbar = plt.colorbar(contour, ax=ax, label=r'Mean Free Path $\lambda_{mfp}$ [m]')

    cs = ax.contour(N, T, lambda_mfp_map, levels=levels,
                    colors='black', linewidths=1, alpha=0.4)
    ax.clabel(cs, inline=True, fontsize=9, fmt='%g m')

    # Mark typical system sizes
    system_sizes = {
        'Tokamak': 1,
        'Lab device': 0.1,
        'Magnetosphere': 1e7,
    }

    for name, L in system_sizes.items():
        # Line where lambda_mfp = L (Kn = 1)
        T_Kn1 = np.sqrt(N * 15 * L / 3.44e11)
        valid = (T_Kn1 >= T_range.min()) & (T_Kn1 <= T_range.max())
        ax.plot(N[valid], T_Kn1[valid], 'r--', linewidth=2.5, alpha=0.8)

        # Label
        idx = len(N) // 2
        if valid[idx]:
            ax.annotate(f'Kn=1 (L={L}m)', (N[idx], T_Kn1[idx]),
                       fontsize=10, color='red', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel(r'Density $n_e$ [m$^{-3}$]', fontsize=12)
    ax.set_ylabel(r'Temperature $T_e$ [eV]', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Mean Free Path Landscape', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 2: Knudsen number for L=1m
    L_ref = 1.0  # m
    Kn_map = lambda_mfp_map / L_ref

    ax = axes[1]
    levels_Kn = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    contour = ax.contourf(N, T, Kn_map, levels=levels_Kn,
                          cmap='coolwarm', norm=plt.matplotlib.colors.LogNorm())
    cbar = plt.colorbar(contour, ax=ax, label=f'Knudsen Number (L={L_ref}m)')

    # Mark Kn = 1 (boundary)
    cs_boundary = ax.contour(N, T, Kn_map, levels=[1],
                             colors='black', linewidths=3)
    ax.clabel(cs_boundary, inline=True, fontsize=12, fmt='Kn=1')

    # Shade regions
    ax.text(1e15, 1e3, 'Collisionless\n(Kn >> 1)',
           fontsize=14, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(1e23, 10, 'Collisional\n(Kn << 1)',
           fontsize=14, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    ax.set_xlabel(r'Density $n_e$ [m$^{-3}$]', fontsize=12)
    ax.set_ylabel(r'Temperature $T_e$ [eV]', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f'Collisionality Regime (L={L_ref} m)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('collisionality_map.png', dpi=150)
    plt.show()

plot_collisionality_map()
```

## 요약

Coulomb 충돌은 전하 입자 간의 장거리 전자기 상호작용으로부터 발생하며 플라즈마 수송 및 역학에 심대한 영향을 미칩니다:

1. **Rutherford 산란**은 이진 Coulomb 충돌을 설명하며, 미분 단면적이 전방 방향으로 강하게 편향됩니다.

2. **Coulomb 로그** $\ln\Lambda \approx 10-20$은 Debye 차폐(최대 충격 매개변수)와 양자/고전 효과(최소 충격 매개변수)를 고려하여 발산을 정규화합니다.

3. **충돌 주파수**는 $\nu \propto n T^{-3/2}$로 스케일되며, 전자는 낮은 질량으로 인해 이온보다 훨씬 더 자주 충돌합니다.

4. **Spitzer 저항률** $\eta \propto T_e^{-3/2}$는 온도에 따라 급격히 감소하여 뜨거운 플라즈마를 훌륭한 도체로 만듭니다.

5. **충돌성 영역**은 Knudsen 수 $Kn = \lambda_{mfp}/L$로 특성화되며, 핵융합 플라즈마는 일반적으로 무충돌성($Kn \gg 1$)이고 운동학적 설명이 필요합니다.

6. **에너지 균등분배**는 전자와 이온 사이에서 느린 시간 스케일 $\tau_{eq} \sim (m_i/m_e)\nu_{ei}^{-1}$로 발생하여, 보조 가열 플라즈마에서 별도의 온도 진화를 허용합니다.

충돌 역학을 이해하는 것은 수송, 가열, 전류 구동, 그리고 운동학적 설명과 유체 설명 사이의 전이를 모델링하는 데 필수적입니다.

## 연습 문제

### 문제 1: 글로우 방전의 충돌 주파수

네온 글로우 방전이 $n_e = 10^{16}$ m$^{-3}$, $T_e = 2$ eV, $T_i = 0.05$ eV입니다.

(a) Coulomb 로그를 계산하세요.

(b) 전자-이온 충돌 주파수 $\nu_{ei}$와 충돌 주기를 계산하세요.

(c) 평균 자유 경로를 구하세요. 일반적인 방전관 직경 3 cm와 비교하세요. 이것은 충돌성 플라즈마입니까, 아니면 무충돌성 플라즈마입니까?

(d) 전자-전자 충돌 주파수를 계산하고 $\nu_{ei}$와 비교하세요.

### 문제 2: Tokamak의 Spitzer 저항률

다음 매개변수를 가진 중수소 토카막을 고려하세요:
- 중심부: $n_e = 5 \times 10^{19}$ m$^{-3}$, $T_e = 12$ keV
- 가장자리: $n_e = 2 \times 10^{18}$ m$^{-3}$, $T_e = 100$ eV

(a) 두 위치에서 Spitzer 저항률을 계산하세요.

(b) 전류 밀도 $J = 1$ MA/m$^2$가 중심부를 통과하면, Ohmic 가열 출력 밀도 $P = \eta J^2$를 계산하세요.

(c) 가장자리는 동일한 총 전류를 운반하지만 더 작은 단면을 통과하여 $J_{edge} = 3$ MA/m$^2$입니다. Ohmic 가열 출력 밀도를 비교하세요. 저항 가열이 어디에서 더 중요합니까?

(d) 중심부와 가장자리 모두에 대해 10 m 토로이달 경로를 따라 전압 강하를 추정하세요.

### 문제 3: 온도 평형

전자 사이클로트론 공명 가열(ECRH) 시스템이 $n_e = 10^{20}$ m$^{-3}$, $T_e = 5$ keV, $T_i = 3$ keV, 부피 $V = 10$ m$^3$인 중수소 플라즈마의 전자에 1 MW를 주입합니다.

(a) 에너지 균등분배 시간 $\tau_{eq}$를 계산하세요.

(b) 전자에서 이온으로의 에너지 전달 속도(와트)를 추정하세요.

(c) 에너지 제약 시간이 $\tau_E = 0.1$ s이면, $\tau_{eq}$와 $\tau_E$를 비교하세요. 온도가 평형을 이룰까요?

(d) 모든 입력 출력이 수송을 통해 손실된다고 가정하고 정상 상태 전자 및 이온 온도를 구하세요(복사 및 기타 손실 무시). 출력 균형이 다음을 준다는 사실을 사용하세요:
   $$P_{ECRH} = P_{ei} + P_{loss,e}$$
   $$P_{ei} = P_{loss,i}$$
   여기서 $P_{ei} \propto (T_e - T_i)/\tau_{eq}$이고 $P_{loss} \propto 3nT/\tau_E$입니다.

### 문제 4: 충격 매개변수 추정

$T_e = 10$ eV와 $n_e = 10^{18}$ m$^{-3}$인 수소 플라즈마에서 전자-이온 충돌의 경우:

(a) Debye 길이 $\lambda_D$를 계산하세요.

(b) 열속도를 사용하여 90° 산란 충격 매개변수 $b_{90}$를 계산하세요.

(c) de Broglie 파장 $\lambda_{dB} = \hbar/(m_e v_{th})$를 구하고 $b_{90}$와 비교하세요. 어느 것이 $b_{min}$을 결정합니까?

(d) $\ln\Lambda = \ln(b_{max}/b_{min})$를 계산하고 표준 공식과 비교하세요.

### 문제 5: 충돌성 영역

축을 따라 $B = 0.5$ T인 자기화된 플라즈마 기둥이 길이 $L_\parallel = 2$ m와 반경 $r = 0.1$ m입니다.

(a) $n_e = 10^{18}$ m$^{-3}$과 $T_e = 50$ eV에 대해 평행 및 수직 평균 자유 경로를 계산하세요. (힌트: $\lambda_{\parallel} = v_{th,\parallel}/\nu$이고 $r_L \ll \lambda_{mfp}$이면 $\lambda_\perp \sim r_L$)

(b) Knudsen 수 $Kn_\parallel = \lambda_{mfp}/L_\parallel$과 $Kn_\perp = \lambda_{mfp}/r$를 계산하세요.

(c) 자기력선을 따라 플라즈마가 충돌성입니까, 아니면 무충돌성입니까? 자기력선을 가로질러서는?

(d) $n_e = 10^{20}$ m$^{-3}$과 $T_e = 1$ keV에 대해 반복하세요. 충돌성이 어떻게 변합니까?

---

**이전:** [플라즈마 소개](./01_Introduction_to_Plasma.md) | **다음:** [플라즈마 설명 계층](./03_Plasma_Description_Hierarchy.md)
