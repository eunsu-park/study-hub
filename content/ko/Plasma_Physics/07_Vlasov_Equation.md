# 7. Vlasov 방정식

## 학습 목표

- 위상 공간과 분포 함수 $f(\mathbf{x},\mathbf{v},t)$ 이해하기
- 무충돌 플라즈마에 대한 Liouville 정리로부터 Vlasov 방정식 유도하기
- $f$의 모멘트로부터 거시적 물리량(밀도, 평균 속도, 압력) 계산하기
- 평형 분포 함수(Maxwellian, bi-Maxwellian, kappa) 탐구하기
- Vlasov 방정식으로부터 보존 법칙(입자, 운동량, 에너지, 엔트로피) 분석하기
- Python을 사용하여 Vlasov 방정식의 수치 해 구현하기

## 1. 위상 공간과 분포 함수

### 1.1 위상 공간

단일 입자의 경우, **위상 공간**은 위치와 속도의 6차원 공간입니다:

$$
(\mathbf{x}, \mathbf{v}) = (x, y, z, v_x, v_y, v_z)
$$

$N$개 입자의 경우, 전체 위상 공간은 $6N$차원입니다. 하지만 큰 $N$ (플라즈마는 $\sim 10^{20}$개의 입자를 가집니다!)에 대해, 개별 입자를 추적하는 것은 비실용적입니다.

대신, 우리는 **분포 함수**를 통한 통계적 기술을 사용합니다.

### 1.2 분포 함수

분포 함수 $f(\mathbf{x}, \mathbf{v}, t)$는 위상 공간에서 **입자의 수밀도**를 제공합니다:

$$
dN = f(\mathbf{x}, \mathbf{v}, t) \, d^3x \, d^3v
$$

**해석**: $f\,d^3x\,d^3v$는 시간 $t$에 $\mathbf{x}$ 주위의 미소 부피 $d^3x$ 안에서 $\mathbf{v}$ 주위의 속도 $d^3v$를 가진 입자의 수입니다.

```
    Phase space (6D)

    v_z ↑
        |       • particle
        |     /
        |    /  represented by
        |   /   density f(x,v,t)
        |  /
        | /________________→ v_x
       /
      / v_y
     ↓

    Position space x,y,z (3D)
```

### 1.3 규격화

부피 $V$ 내의 전체 입자 수는:

$$
N(t) = \int_V d^3x \int_{-\infty}^{\infty} f(\mathbf{x}, \mathbf{v}, t) \, d^3v
$$

전체 플라즈마의 경우:

$$
N_{\text{total}} = \int_{\text{all space}} d^3x \int_{-\infty}^{\infty} f(\mathbf{x}, \mathbf{v}, t) \, d^3v
$$

### 1.4 모멘트: 거시적 물리량

거시적 물리량은 속도 공간에 대해 $f$를 적분하여 얻습니다 (**모멘트**):

**수밀도**:
$$
n(\mathbf{x}, t) = \int f(\mathbf{x}, \mathbf{v}, t) \, d^3v
$$

**평균(유체) 속도**:
$$
\mathbf{u}(\mathbf{x}, t) = \frac{1}{n(\mathbf{x}, t)} \int \mathbf{v} f(\mathbf{x}, \mathbf{v}, t) \, d^3v
$$

**압력 텐서**:
$$
\mathbf{P}(\mathbf{x}, t) = m \int (\mathbf{v} - \mathbf{u})(\mathbf{v} - \mathbf{u}) f(\mathbf{x}, \mathbf{v}, t) \, d^3v
$$

여기서 외적 $(\mathbf{v} - \mathbf{u})(\mathbf{v} - \mathbf{u})$는 텐서를 제공합니다.

**스칼라 압력** (등방성 분포의 경우):
$$
P = \frac{1}{3}\text{Tr}(\mathbf{P}) = \frac{m}{3}\int |\mathbf{v} - \mathbf{u}|^2 f \, d^3v
$$

**온도** (운동학적 정의):
$$
T = \frac{P}{nk_B} = \frac{m}{3nk_B}\int |\mathbf{v} - \mathbf{u}|^2 f \, d^3v
$$

**에너지 밀도**:
$$
\mathcal{E} = \frac{m}{2}\int v^2 f \, d^3v = \frac{1}{2}m n u^2 + \frac{3}{2}nk_BT
$$

(운동 에너지 = 평균 흐름 + 열 에너지)

### 1.5 예제: 1D 속도 분포

$f = f(v_x)$가 Gaussian인 1D 문제를 고려합니다:

$$
f(v_x) = n_0 \sqrt{\frac{m}{2\pi k_B T}} \exp\left(-\frac{m(v_x - u)^2}{2k_BT}\right)
$$

모멘트:
- $\int f \, dv_x = n_0$ (밀도)
- $\int v_x f \, dv_x = n_0 u$ (운동량 밀도)
- $\int (v_x - u)^2 f \, dv_x = n_0 k_BT/m$ (분산)

## 2. Vlasov 방정식

### 2.1 Liouville 정리로부터의 유도

고전 역학에서, **Liouville 정리**는 위상 공간 밀도가 궤적을 따라 보존됨을 나타냅니다:

$$
\frac{df}{dt} = 0
$$

전미분을 전개하면:

$$
\frac{\partial f}{\partial t} + \frac{d\mathbf{x}}{dt}\cdot\frac{\partial f}{\partial \mathbf{x}} + \frac{d\mathbf{v}}{dt}\cdot\frac{\partial f}{\partial \mathbf{v}} = 0
$$

전하 입자의 경우:
- $\frac{d\mathbf{x}}{dt} = \mathbf{v}$
- $\frac{d\mathbf{v}}{dt} = \frac{q}{m}(\mathbf{E} + \mathbf{v}\times\mathbf{B})$

대입하면:

$$
\boxed{\frac{\partial f}{\partial t} + \mathbf{v}\cdot\nabla f + \frac{q}{m}(\mathbf{E} + \mathbf{v}\times\mathbf{B})\cdot\frac{\partial f}{\partial \mathbf{v}} = 0}
$$

이것이 **Vlasov 방정식**입니다 (**무충돌 Boltzmann 방정식**이라고도 함).

### 2.2 물리적 해석

Vlasov 방정식은 분포 함수가 입자 궤적에 의해 위상 공간을 통해 **대류**되며, (무충돌 플라즈마에 대해) 소스나 싱크가 없음을 나타냅니다.

```
    Phase space flow

       f(x,v,t)   →   f(x+vδt, v+aδt, t+δt)

    Particles flow along trajectories in (x,v) space
    Distribution function is "painted" on phase space
    and advected by the flow
```

각 항:
- $\frac{\partial f}{\partial t}$: 명시적 시간 의존성
- $\mathbf{v}\cdot\nabla f$: 위치 공간에서의 이류
- $\frac{q}{m}(\mathbf{E}+\mathbf{v}\times\mathbf{B})\cdot\frac{\partial f}{\partial \mathbf{v}}$: 속도 공간에서의 가속

### 2.3 자기 일관성: Vlasov-Maxwell 시스템

전기장과 자기장 $\mathbf{E}$와 $\mathbf{B}$는 플라즈마 자체에 의해 생성됩니다. 이들은 **Maxwell 방정식**을 만족해야 합니다:

$$
\nabla\cdot\mathbf{E} = \frac{\rho}{\epsilon_0} = \frac{1}{\epsilon_0}\sum_s q_s \int f_s \, d^3v
$$

$$
\nabla\times\mathbf{E} = -\frac{\partial\mathbf{B}}{\partial t}
$$

$$
\nabla\cdot\mathbf{B} = 0
$$

$$
\nabla\times\mathbf{B} = \mu_0\mathbf{J} + \mu_0\epsilon_0\frac{\partial\mathbf{E}}{\partial t} = \mu_0\sum_s q_s\int\mathbf{v}f_s\,d^3v + \mu_0\epsilon_0\frac{\partial\mathbf{E}}{\partial t}
$$

결합된 시스템 (Vlasov + Maxwell)은 **Vlasov-Maxwell 시스템**이며, 플라즈마의 자기 일관적 운동학적 기술입니다.

정전기 문제의 경우 ($\mathbf{B}$ 변화 없음), **Vlasov-Poisson 시스템**을 갖습니다:

$$
\frac{\partial f}{\partial t} + \mathbf{v}\cdot\nabla f + \frac{q}{m}\mathbf{E}\cdot\frac{\partial f}{\partial \mathbf{v}} = 0
$$

$$
\nabla\cdot\mathbf{E} = \frac{1}{\epsilon_0}\sum_s q_s \int f_s \, d^3v
$$

### 2.4 다종 플라즈마

여러 종(전자, 이온, 불순물)을 가진 플라즈마의 경우, 각 종 $s$에 대해 별도의 분포 함수 $f_s$를 갖습니다:

$$
\frac{\partial f_s}{\partial t} + \mathbf{v}\cdot\nabla f_s + \frac{q_s}{m_s}(\mathbf{E} + \mathbf{v}\times\mathbf{B})\cdot\frac{\partial f_s}{\partial \mathbf{v}} = 0
$$

$\mathbf{E}$와 $\mathbf{B}$는 모든 종에 대한 합으로 결정됩니다.

## 3. 보존 법칙

### 3.1 입자 보존

Vlasov 방정식을 모든 속도에 대해 적분합니다:

$$
\int \frac{\partial f}{\partial t} d^3v + \int \mathbf{v}\cdot\nabla f \, d^3v + \int \frac{q}{m}(\mathbf{E}+\mathbf{v}\times\mathbf{B})\cdot\frac{\partial f}{\partial \mathbf{v}} d^3v = 0
$$

첫 번째 항:
$$
\int \frac{\partial f}{\partial t} d^3v = \frac{\partial}{\partial t}\int f \, d^3v = \frac{\partial n}{\partial t}
$$

두 번째 항:
$$
\int \mathbf{v}\cdot\nabla f \, d^3v = \nabla\cdot\int \mathbf{v}f \, d^3v = \nabla\cdot(n\mathbf{u})
$$

세 번째 항 (부분 적분, $f\to 0$를 $|\mathbf{v}|\to\infty$일 때 가정):
$$
\int \frac{q}{m}\mathbf{F}\cdot\frac{\partial f}{\partial \mathbf{v}} d^3v = -\frac{q}{m}\int f\nabla_v\cdot\mathbf{F} \, d^3v = 0
$$

왜냐하면 $\nabla_v\cdot(\mathbf{E}+\mathbf{v}\times\mathbf{B}) = \nabla_v\cdot\mathbf{E} + \nabla_v\cdot(\mathbf{v}\times\mathbf{B}) = 0$ (E는 v와 독립적이고, 외적은 속도 공간에서 발산이 0).

결과:
$$
\boxed{\frac{\partial n}{\partial t} + \nabla\cdot(n\mathbf{u}) = 0}
$$

**연속 방정식**: 입자 수가 보존됩니다.

### 3.2 운동량 보존

Vlasov 방정식에 $m\mathbf{v}$를 곱하고 적분합니다:

$$
\int m\mathbf{v}\frac{\partial f}{\partial t} d^3v + \int m\mathbf{v}(\mathbf{v}\cdot\nabla f) d^3v + \int q\mathbf{v}(\mathbf{E}+\mathbf{v}\times\mathbf{B})\cdot\frac{\partial f}{\partial\mathbf{v}} d^3v = 0
$$

약간의 대수적 계산 후 (부분 적분 등):

$$
\frac{\partial}{\partial t}(mn\mathbf{u}) + \nabla\cdot\mathbf{P} = qn(\mathbf{E} + \mathbf{u}\times\mathbf{B})
$$

여기서 $\mathbf{P}$는 운동량 플럭스 텐서입니다 (압력과 흐름을 포함).

**운동량 방정식**: 운동량은 전자기력으로 인해 변화합니다.

### 3.3 에너지 보존

$\frac{1}{2}mv^2$를 곱하고 적분합니다:

$$
\frac{\partial}{\partial t}\left(\frac{1}{2}mn\langle v^2\rangle\right) + \nabla\cdot\mathbf{Q} = qn\mathbf{u}\cdot\mathbf{E}
$$

여기서 $\mathbf{Q}$는 에너지 플럭스입니다.

**에너지 방정식**: 운동 에너지는 전기장에 의해 수행된 일로 인해 변화합니다.

(자기장은 일을 하지 않음: $\mathbf{v}\times\mathbf{B}\perp\mathbf{v}$)

### 3.4 엔트로피와 Casimir 불변량

**엔트로피**를 정의합니다:

$$
S = -k_B \int f \ln f \, d^3x \, d^3v
$$

Vlasov 방정식으로부터, 우리는 다음을 보일 수 있습니다:

$$
\frac{dS}{dt} = 0
$$

**엔트로피는 무충돌 플라즈마에서 보존됩니다** (가역적 동역학). 이것은 엔트로피가 증가하는 충돌 시스템 (H-정리)과 매우 다릅니다.

더 일반적으로, 다음 형태의 모든 범함수:

$$
C = \int G(f) \, d^3x \, d^3v
$$

여기서 $G$는 임의의 함수이며, $f$가 Vlasov 방정식을 만족하면 보존됩니다. 이를 **Casimir 불변량**이라고 합니다.

## 4. 평형 분포 함수

### 4.1 Maxwellian 분포

가장 일반적인 평형은 **Maxwellian**입니다 (열평형):

$$
\boxed{f_0(\mathbf{v}) = n_0 \left(\frac{m}{2\pi k_B T}\right)^{3/2} \exp\left(-\frac{m|\mathbf{v} - \mathbf{u}|^2}{2k_BT}\right)}
$$

여기서:
- $n_0$: 평형 밀도
- $\mathbf{u}$: 평균 드리프트 속도
- $T$: 온도

**성질**:
- $\mathbf{u}$로 움직이는 프레임에서 등방성
- 주어진 밀도와 에너지에 대해 엔트로피 최대화
- $\mathbf{E} = \mathbf{B} = 0$ (또는 균일한 $\mathbf{u} = \mathbf{E}\times\mathbf{B}/B^2$)인 경우 Vlasov 방정식의 정상 해

### 4.2 Jeans 정리

**Jeans 정리**: 운동 상수의 임의의 함수는 Vlasov 방정식의 정상 해입니다.

장 $\mathbf{E}$, $\mathbf{B}$에서 입자에 대해, 에너지 $H = \frac{1}{2}mv^2 + q\Phi$가 보존되면:

$$
f = f(H)
$$

는 정상 해입니다.

더 일반적으로:
$$
f = f(H, \mathbf{P}_{\text{canonical}}, \mu, J, \Phi, \ldots)
$$

여기서 인수는 임의의 운동 상수입니다 (에너지, 정준 운동량, 단열 불변량 등).

### 4.3 Bi-Maxwellian 분포

자화된 플라즈마에서, 비등방성이 발생할 수 있습니다. 일반적인 모델은 **bi-Maxwellian**입니다:

$$
\boxed{f_0(v_\perp, v_\parallel) = n_0 \frac{m}{2\pi k_B} \frac{1}{T_\perp}\frac{1}{\sqrt{2\pi k_B T_\parallel/m}} \exp\left(-\frac{mv_\perp^2}{2k_BT_\perp} - \frac{m v_\parallel^2}{2k_BT_\parallel}\right)}
$$

여기서 $T_\perp$와 $T_\parallel$는 $\mathbf{B}$에 수직 및 평행한 온도입니다.

**비등방성 매개변수**:
$$
A = \frac{T_\perp}{T_\parallel} - 1
$$

- $A > 0$: 수직 가열 (예: 사이클로트론 공명 가열)
- $A < 0$: 평행 가열 (예: 자기 압축)

Bi-Maxwellian 분포는 불안정성을 유발할 수 있습니다 (예: $A > 0$에 대한 전자기 이온 사이클로트론 파동).

### 4.4 Kappa 분포

우주 플라즈마 (태양풍, 자기권)에서, 관측은 **비열적 꼬리**를 보여줍니다 — Maxwellian이 예측하는 것보다 더 많은 고에너지 입자. 일반적인 모델은 **kappa 분포**입니다:

$$
\boxed{f_\kappa(v) = n_0 \frac{1}{(\pi\kappa\theta^2)^{3/2}} \frac{\Gamma(\kappa+1)}{\Gamma(\kappa-1/2)} \left(1 + \frac{v^2}{\kappa\theta^2}\right)^{-(\kappa+1)}}
$$

여기서:
- $\kappa > 3/2$: 스펙트럼 지수 (낮은 $\kappa$ = 더 뚱뚱한 꼬리)
- $\theta^2 = \frac{2k_BT}{m}\frac{\kappa - 3/2}{\kappa}$: 열속도 매개변수
- $\Gamma$: 감마 함수

**극한**:
- $\kappa \to \infty$: Maxwellian으로 회복
- $\kappa \to 3/2$: 멱법칙 꼬리 $f \propto v^{-2(\kappa+1)} = v^{-5}$

Kappa 분포는 다음으로부터 발생합니다:
- 간헐적 입자 가속
- 무충돌 "준평형" 완화
- 난류 가열

### 4.5 드리프트 Maxwellian (빔)

빔이 있는 플라즈마는 두 개의 집단을 갖습니다:

$$
f = f_{\text{bulk}} + f_{\text{beam}}
$$

예를 들어:
$$
f(v) = n_b\left(\frac{m}{2\pi k_B T_b}\right)^{3/2}\exp\left(-\frac{m(v - v_b)^2}{2k_BT_b}\right) + n_c\left(\frac{m}{2\pi k_B T_c}\right)^{3/2}\exp\left(-\frac{mv^2}{2k_BT_c}\right)
$$

여기서 아래첨자 $b$ = 빔, $c$ = 코어.

**Bump-on-tail** (1D 버전):

```
    f(v)
      ↑
      |    Core
      |   /‾‾‾\
      |  /     \___
      | /          \___   Beam (bump)
      |/               \__/‾‾\_______
     ─┴───────────────────────────────→ v
                               v_b

    Positive slope df/dv > 0 at resonance
    → Unstable (two-stream instability)
```

이것은 불안정성 (two-stream, bump-on-tail)을 유발하여, 빔에서 파동으로 에너지를 전달합니다.

## 5. 선형화 및 섭동 이론

### 5.1 평형 + 섭동

작은 진폭 파동의 경우, 분해합니다:

$$
f = f_0(\mathbf{v}) + f_1(\mathbf{x}, \mathbf{v}, t)
$$

$$
\mathbf{E} = \mathbf{E}_0 + \mathbf{E}_1(\mathbf{x}, t)
$$

여기서 아래첨자 $0$ = 평형, $1$ = 섭동이며 $|f_1| \ll f_0$.

### 5.2 선형화된 Vlasov 방정식

Vlasov에 대입하고 1차 항만 유지합니다:

$$
\frac{\partial f_1}{\partial t} + \mathbf{v}\cdot\nabla f_1 + \frac{q}{m}(\mathbf{E}_0 + \mathbf{v}\times\mathbf{B}_0)\cdot\frac{\partial f_1}{\partial\mathbf{v}} = -\frac{q}{m}\mathbf{E}_1\cdot\frac{\partial f_0}{\partial\mathbf{v}}
$$

선형화된 Poisson과 결합:

$$
\nabla\cdot\mathbf{E}_1 = \frac{1}{\epsilon_0}\sum_s q_s \int f_1^{(s)} \, d^3v
$$

이것은 플라즈마 파동과 불안정성의 **선형 운동학 이론**의 기초입니다 (Landau 감쇠에 대해 Lesson 8에서 이것을 풀 것입니다).

### 5.3 BGK 모드

**BGK (Bernstein-Greene-Kruskal) 모드**는 Vlasov-Poisson 시스템의 정확한 비선형 해이며, 갇힌 입자를 가진 정전기 파동 패킷을 나타냅니다.

1D의 경우:
$$
f(x, v, t) = f(v - u(x))
$$

여기서 입자는 파동의 포텐셜 우물에 갇혀 있습니다. 이들은 위상 속도 $v_{\text{ph}}$와 포획 폭에 의해 결정되는 진폭을 가진 진행파 해입니다.

BGK 모드는 다음 사이의 균형을 나타냅니다:
- 입자 포획 (비선형 효과)
- 파동 전파

이들은 다음과 관련이 있습니다:
- 비선형 Landau 감쇠
- 전자 및 이온 홀
- 이중층

## 6. Python 구현

### 6.1 분포 함수 플로팅

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Constants
k_B = 1.380649e-23  # J/K
m_p = 1.67e-27      # kg
m_e = 9.11e-31      # kg
e = 1.6e-19         # C

def maxwellian_1d(v, n, T, m, u=0):
    """
    1D Maxwellian distribution
    """
    return n * np.sqrt(m / (2 * np.pi * k_B * T)) * np.exp(-m * (v - u)**2 / (2 * k_B * T))

def maxwellian_3d(v, n, T, m):
    """
    3D Maxwellian (isotropic, speed distribution)
    f(v) dv = 4π v^2 f(v_vec) dv
    """
    return 4 * np.pi * v**2 * n * (m / (2 * np.pi * k_B * T))**(3/2) * np.exp(-m * v**2 / (2 * k_B * T))

def kappa_1d(v, n, T, m, kappa, u=0):
    """
    1D kappa distribution
    """
    theta_sq = (2 * k_B * T / m) * (kappa - 3/2) / kappa
    norm = n / (np.sqrt(np.pi * kappa * theta_sq)) * gamma(kappa + 1) / gamma(kappa - 1/2)
    return norm * (1 + (v - u)**2 / (kappa * theta_sq))**(-kappa - 1)

def bi_maxwellian_vperp(v_perp, v_para, n, T_perp, T_para, m):
    """
    Bi-Maxwellian: f(v_perp, v_para)
    Here we fix v_para and plot vs v_perp
    """
    return n * (m / (2 * np.pi * k_B * T_perp)) * np.sqrt(m / (2 * np.pi * k_B * T_para)) * \
           np.exp(-m * v_perp**2 / (2 * k_B * T_perp) - m * v_para**2 / (2 * k_B * T_para))

# Parameters
n0 = 1e19  # m^-3
T_eV = 100  # eV
T = T_eV * e / k_B  # Kelvin
m = m_p

v = np.linspace(-5e5, 5e5, 1000)  # m/s

# Plot 1D distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Maxwellian
ax = axes[0, 0]
f_max = maxwellian_1d(v, n0, T, m)
ax.plot(v/1e3, f_max, 'b-', linewidth=2, label='Maxwellian')
ax.set_xlabel('v (km/s)', fontsize=12)
ax.set_ylabel('f(v) (s/m⁴)', fontsize=12)
ax.set_title('1D Maxwellian Distribution', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Drifting Maxwellian (beam)
ax = axes[0, 1]
f_core = maxwellian_1d(v, n0*0.9, T, m, u=0)
f_beam = maxwellian_1d(v, n0*0.1, T*0.5, m, u=2e5)
f_total = f_core + f_beam
ax.plot(v/1e3, f_core, 'b-', linewidth=1, label='Core')
ax.plot(v/1e3, f_beam, 'r-', linewidth=1, label='Beam')
ax.plot(v/1e3, f_total, 'k-', linewidth=2, label='Total')
ax.set_xlabel('v (km/s)', fontsize=12)
ax.set_ylabel('f(v) (s/m⁴)', fontsize=12)
ax.set_title('Bump-on-Tail Distribution', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Kappa distribution comparison
ax = axes[1, 0]
kappa_values = [3, 5, 10, 100]
colors = ['red', 'orange', 'green', 'blue']
for kappa_val, color in zip(kappa_values, colors):
    if kappa_val > 3/2:
        # κ는 분포가 정규화 가능하려면 > 3/2이어야 합니다 (모든 속도에 대한
        # 적분은 지수(power-law exponent) κ+1 > 5/2일 때만 수렴합니다).
        f_kappa = kappa_1d(v, n0, T, m, kappa_val)
        # κ = 100은 선형 척도에서 Maxwellian과 수치적으로 구별할 수 없지만,
        # "κ → ∞"로 표시하여 학생들에게 이론적 극한을 명시적으로 보여줍니다.
        label = f'κ = {kappa_val}' if kappa_val < 100 else 'κ → ∞ (Maxwellian)'
        ax.plot(v/1e3, f_kappa, color=color, linewidth=2, label=label)

ax.set_xlabel('v (km/s)', fontsize=12)
ax.set_ylabel('f(v) (s/m⁴)', fontsize=12)
ax.set_title('Kappa Distributions (Non-thermal Tails)', fontsize=14, fontweight='bold')
ax.set_yscale('log')
# 로그 척도가 여기서 필수적입니다: 초열적(suprathermal) 증강은 f가 수십 배
# 차이나는 꼬리(v >> v_th)에서만 보입니다. 선형 척도에서는 피크 근처에서
# 모든 곡선이 동일하게 보여, 멱법칙(power-law) 꼬리의 핵심 물리가 숨겨집니다.
ax.grid(True, alpha=0.3, which='both')
ax.legend()

# 3D speed distribution
ax = axes[1, 1]
v_speed = np.linspace(0, 6e5, 1000)
f_3d = maxwellian_3d(v_speed, n0, T, m)
v_th = np.sqrt(2 * k_B * T / m)
ax.plot(v_speed/1e3, f_3d, 'b-', linewidth=2)
ax.axvline(x=v_th/1e3, color='r', linestyle='--', linewidth=2,
          label=f'v_th = {v_th/1e3:.1f} km/s')
ax.set_xlabel('Speed v (km/s)', fontsize=12)
ax.set_ylabel('f(v) (s/m⁴)', fontsize=12)
ax.set_title('3D Maxwellian Speed Distribution', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('distribution_functions.png', dpi=150)
print("Saved: distribution_functions.png")

# Bi-Maxwellian
fig, ax = plt.subplots(figsize=(10, 6))

v_perp_array = np.linspace(0, 5e5, 1000)
T_perp_eV = 200
T_para_eV = 50
T_perp = T_perp_eV * e / k_B
T_para = T_para_eV * e / k_B

# Different v_para slices
v_para_values = [0, 1e5, 2e5, 3e5]
colors = ['blue', 'green', 'orange', 'red']

for v_para, color in zip(v_para_values, colors):
    f_bi = bi_maxwellian_vperp(v_perp_array, v_para, n0, T_perp, T_para, m)
    ax.plot(v_perp_array/1e3, f_bi, color=color, linewidth=2,
           label=f'v_para = {v_para/1e3:.0f} km/s')

ax.set_xlabel('v_perp (km/s)', fontsize=12)
ax.set_ylabel('f(v_perp, v_para) (s/m⁴)', fontsize=12)
ax.set_title(f'Bi-Maxwellian: T_perp = {T_perp_eV} eV, T_para = {T_para_eV} eV',
            fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')
ax.legend()
plt.tight_layout()
plt.savefig('bi_maxwellian.png', dpi=150)
print("Saved: bi_maxwellian.png")
```

### 6.2 모멘트 계산

```python
from scipy.integrate import simps

def compute_moments(v_array, f_array):
    """
    Compute moments of 1D distribution function
    """
    # 밀도: Simpson 규칙(simps)을 사용합니다. np.trapz보다 O(h^4) 오차를 가집니다
    # (trapz는 O(h^2)). 이는 f가 매끄럽지만 곡선 형태의 꼬리를 가질 때 중요하며,
    # 이는 Maxwellian 및 kappa 분포를 거친 속도 격자에서 샘플링할 때 전형적입니다.
    n = simps(f_array, v_array)

    # 평균 속도: n (n0가 아닌)으로 나누어 분포가 기준 밀도 n0에서
    # 벗어났을 때도 진정한 평균 속도를 제공합니다.
    u = simps(v_array * f_array, v_array) / n

    # 분산: u ≠ 0인 이동(drifting) 분포에서 열 분산을 올바르게 얻기 위해
    # 계산된 평균 u (0이 아닌)에 대해 상대적으로 계산합니다.
    var = simps((v_array - u)**2 * f_array, v_array) / n

    # Thermal velocity
    v_th = np.sqrt(var)

    return n, u, v_th, var

# Test with Maxwellian
v_test = np.linspace(-1e6, 1e6, 10000)
n_test = 1e19
T_test = 100 * e / k_B
u_test = 1e5  # drifting

f_test = maxwellian_1d(v_test, n_test, T_test, m_p, u=u_test)

n_calc, u_calc, v_th_calc, var_calc = compute_moments(v_test, f_test)

print("\n=== Moment Calculation ===")
print(f"Input:")
print(f"  n = {n_test:.2e} m^-3")
print(f"  u = {u_test:.2e} m/s")
print(f"  T = {T_test*k_B/e:.2f} eV")
print(f"  v_th (expected) = {np.sqrt(2*k_B*T_test/m_p):.2e} m/s")

print(f"\nCalculated from distribution:")
print(f"  n = {n_calc:.2e} m^-3")
print(f"  u = {u_calc:.2e} m/s")
print(f"  v_th = {v_th_calc:.2e} m/s")
print(f"  T = {m_p*var_calc/k_B/2:.2f} K = {m_p*var_calc/e/2:.2f} eV")

errors = [
    abs(n_calc - n_test) / n_test,
    abs(u_calc - u_test) / abs(u_test),
    abs(v_th_calc - np.sqrt(2*k_B*T_test/m_p)) / np.sqrt(2*k_B*T_test/m_p)
]
print(f"\nRelative errors: {errors}")
```

### 6.3 간단한 1D Vlasov 솔버 (연산자 분리)

```python
def vlasov_1d_solver(x, v, f0, E_func, dt, num_steps, q, m):
    """
    Simple 1D Vlasov solver using operator splitting

    x: position grid
    v: velocity grid
    f0: initial distribution f(x,v,t=0)
    E_func: function E(x,t) giving electric field
    dt: timestep
    num_steps: number of steps
    q, m: charge and mass

    Returns: f(x,v,t) at final time
    """
    f = f0.copy()
    dx = x[1] - x[0]
    dv = v[1] - v[0]
    Nx, Nv = len(x), len(v)

    # Storage for snapshots
    snapshots = []
    snapshot_times = []

    for n in range(num_steps):
        t = n * dt

        # Step 1: x 방향 이류(advection) (∂f/∂t + v ∂f/∂x = 0)
        # 연산자 분리(operator splitting)는 6D Vlasov 방정식을 두 개의 1D 이류로
        # 분리합니다 (x와 v 방향 각각). 각각 독립적으로 풀립니다. dt가 작을 때
        # 유효합니다 (Strang 분리는 전반적으로 O(dt^2) 정확도를 줍니다).
        # 풍상(upwind) 방식 사용: 이류 방정식에 대해 안정적(소산적)이기 때문에 선택됩니다.
        # 스텐실(stencil)은 항상 정보가 이동하는 방향인 상류(upstream) 쪽에서
        # 도함수를 취하여 f에 비물리적 진동이 나타나는 것을 방지합니다.
        f_new = np.zeros_like(f)
        for j in range(Nv):
            for i in range(Nx):
                if v[j] > 0:
                    i_up = (i - 1) % Nx  # 입자가 +x 방향으로 이동; 정보는 왼쪽 셀에서 옵니다.
                    # 주기적 경계 조건(periodic BC) (% Nx)은 f가 x=0과 x=L에서 동일하다는
                    # 가정을 강제합니다 — 공간적으로 주기적인 플라즈마 파동에 적합합니다.
                    f_new[i, j] = f[i, j] - v[j] * dt / dx * (f[i, j] - f[i_up, j])
                else:
                    i_up = (i + 1) % Nx
                    f_new[i, j] = f[i, j] - v[j] * dt / dx * (f[i_up, j] - f[i, j])
        f = f_new.copy()

        # Step 2: v 방향 가속(acceleration) (∂f/∂t + (q/m)E ∂f/∂v = 0)
        # 전기장은 현재 시간 t (t+dt가 아닌)에서 재평가되어 명시적 시간 적분을 1차로 유지합니다.
        # t에서 E를 사용하는 것은 v 공간에서의 순방향 Euler 스텝과 동등합니다 —
        # 단순하지만 정확도를 위해 작은 dt가 필요합니다.
        E = E_func(x, t)
        f_new = np.zeros_like(f)
        for i in range(Nx):
            a = q * E[i] / m  # 가속도 = q*E/m (속도 공간에서의 Newton의 법칙)
            for j in range(Nv):
                if a > 0:
                    j_up = max(j - 1, 0)
                    # 양의 가속도는 f를 더 높은 v 방향으로 이동시킵니다: 풍상 셀로 왼쪽
                    # 이웃(더 낮은 v)을 사용합니다. 경계에서 클램핑(j_up = 0)하여
                    # 입자가 속도 공간에서 감싸지 않도록 합니다 —
                    # 위치와 달리 속도는 이 설정에서 물리적 한계를 가집니다.
                    f_new[i, j] = f[i, j] - a * dt / dv * (f[i, j] - f[i, j_up])
                else:
                    j_up = min(j + 1, Nv - 1)
                    f_new[i, j] = f[i, j] - a * dt / dv * (f[i, j_up] - f[i, j])
        f = f_new.copy()

        # Save snapshots
        if n % (num_steps // 10) == 0:
            snapshots.append(f.copy())
            snapshot_times.append(t)

    return f, snapshots, snapshot_times

# Setup 1D problem
Nx, Nv = 128, 128
# Lx = 2π/k는 박스를 정확히 하나의 파장으로 설정합니다. 이를 통해 주기적 경계 조건이
# 파동과 일치합니다: f(x=0) = f(x=Lx). 박스가 파장의 분수를 포함하면
# 인위적인 반사가 발생합니다.
Lx = 2 * np.pi / 0.5  # wavelength
x = np.linspace(0, Lx, Nx)
# 속도 격자는 100 eV에서 ±3×10^5 m/s ≈ ±6 v_th를 커버하여 Maxwellian의 >99.9%를 포함합니다.
# v_th가 너무 적으면 입자가 손실되어 보존이 위반됩니다;
# 너무 많이 늘이면 지수적으로 작은 꼬리에 격자점을 낭비합니다.
v = np.linspace(-3e5, 3e5, Nv)

X, V = np.meshgrid(x, v, indexing='ij')

# Initial condition: perturbed Maxwellian
n0 = 1e19
T0 = 100 * e / k_B
k_wave = 0.5  # wavenumber (1/m)
# 작은 진폭(1%)은 섭동을 선형 영역에 유지하여 시뮬레이션 결과를
# 선형 파동 이론(Landau 감쇠 등)과 직접 비교할 수 있게 합니다.
# 더 큰 진폭은 입자 포획과 비선형 포화를 일으킵니다.
amplitude = 0.01

f0 = np.zeros((Nx, Nv))
for i in range(Nx):
    n_pert = n0 * (1 + amplitude * np.cos(k_wave * x[i]))
    f0[i, :] = maxwellian_1d(v, n_pert, T0, m_e, u=0)

# Electric field (for this demo, use a static wave)
def E_field(x, t):
    E0 = 1e2  # V/m
    return E0 * np.sin(k_wave * x) * np.cos(1e5 * t)

# Solve
print("\nSolving 1D Vlasov equation...")
dt = 1e-8  # s
num_steps = 500

f_final, snapshots, times = vlasov_1d_solver(x, v, f0, E_field, dt, num_steps, -e, m_e)

print(f"Completed {num_steps} steps, final time = {times[-1]:.2e} s")

# Plot phase space evolution
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (snap, t) in enumerate(zip([f0] + snapshots[:5], [0] + times[:5])):
    ax = axes[idx]
    im = ax.contourf(x, v/1e3, snap.T, levels=30, cmap='viridis')
    ax.set_xlabel('x (m)', fontsize=10)
    ax.set_ylabel('v (km/s)', fontsize=10)
    ax.set_title(f't = {t:.2e} s', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='f(x,v)')

plt.tight_layout()
plt.savefig('vlasov_1d_evolution.png', dpi=150)
print("Saved: vlasov_1d_evolution.png")

# Density evolution
fig, ax = plt.subplots(figsize=(10, 6))
for idx, (snap, t) in enumerate(zip([f0] + snapshots[::2], [0] + times[::2])):
    n_x = simps(snap, v, axis=1)
    ax.plot(x, n_x/n0, linewidth=1.5, label=f't = {t:.1e} s')

ax.set_xlabel('x (m)', fontsize=12)
ax.set_ylabel('n(x,t) / n0', fontsize=12)
ax.set_title('Density Oscillation (1D Vlasov)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('vlasov_density.png', dpi=150)
print("Saved: vlasov_density.png")
```

## 요약

이 수업에서, 우리는 Vlasov 방정식을 사용하여 플라즈마의 운동학 이론을 개발했습니다:

1. **위상 공간과 분포 함수**: $f(\mathbf{x}, \mathbf{v}, t)$는 6D 위상 공간에서 플라즈마의 통계적 상태를 기술합니다.

2. **모멘트**: 거시적 물리량 (밀도, 속도, 압력, 온도)은 속도 공간에 대해 $f$를 적분하여 얻습니다.

3. **Vlasov 방정식**:
   $$
   \frac{\partial f}{\partial t} + \mathbf{v}\cdot\nabla f + \frac{q}{m}(\mathbf{E}+\mathbf{v}\times\mathbf{B})\cdot\frac{\partial f}{\partial\mathbf{v}} = 0
   $$
   Liouville 정리로부터 유도된, $f$의 무충돌 진화를 기술합니다.

4. **자기 일관성**: Maxwell 방정식과 결합된 Vlasov 방정식은 Vlasov-Maxwell 시스템을 형성합니다.

5. **보존 법칙**: Vlasov 방정식은 입자, 운동량, 에너지, 엔트로피 (Casimir 불변량)를 보존합니다.

6. **평형 분포**:
   - **Maxwellian**: 열평형
   - **Bi-Maxwellian**: 비등방성 자화 플라즈마
   - **Kappa**: 비열적 꼬리 (우주 플라즈마)
   - **빔**: bump-on-tail (불안정)

7. **선형화**: 작은 진폭 파동에 대한 섭동 이론은 선형 운동학 이론으로 이어집니다 (다음 수업: Landau 감쇠).

Vlasov 방정식은 운동학적 플라즈마 물리학의 기초이며, 유체 모델이 놓치는 현상을 포착합니다:
- Landau 감쇠 (무충돌 파동 감쇠)
- 운동학적 불안정성
- 파동-입자 공명

## 연습 문제

### 문제 1: 드리프트 Maxwellian의 모멘트

1D Maxwellian 분포를 고려합니다:

$$
f(v) = n_0\sqrt{\frac{m}{2\pi k_B T}}\exp\left(-\frac{m(v-u_0)^2}{2k_BT}\right)
$$

(a) 0차 모멘트 (밀도)를 계산합니다: $n = \int f(v) \, dv$.

(b) 1차 모멘트 (평균 속도)를 계산합니다: $\langle v\rangle = \frac{1}{n}\int v f(v) \, dv$.

(c) 2차 모멘트 (온도)를 계산합니다: $T = \frac{m}{k_B}\int (v - \langle v\rangle)^2 f(v) \, dv$.

(d) 제공된 Python 코드를 사용하여 결과를 수치적으로 검증합니다.

---

### 문제 2: Bi-Maxwellian 비등방성

자화된 플라즈마가 $T_\perp = 200$ eV 및 $T_\parallel = 50$ eV인 bi-Maxwellian 분포를 갖습니다.

(a) 비등방성 매개변수 $A = T_\perp/T_\parallel - 1$을 계산합니다.

(b) $n = 10^{19}$ m$^{-3}$에 대해 압력 텐서 성분 $P_\perp = nk_BT_\perp$ 및 $P_\parallel = nk_BT_\parallel$를 계산합니다.

(c) 이 분포는 등방성입니까? 안정합니까? (힌트: 큰 $A > 0$는 사이클로트론 불안정성을 유발할 수 있습니다.)

(d) 비등방성에서 이용 가능한 자유 에너지를 추정합니다: $\Delta W = \frac{3}{2}nk_B(T_\perp - T_\parallel)$.

---

### 문제 3: Kappa 분포 대 Maxwellian

$\kappa = 3$인 kappa 분포에 대해, 동일한 밀도와 온도를 가진 Maxwellian에 비해 $v > 3v_{\text{th}}$인 입자 (초열 입자)의 수의 비율을 계산합니다.

(a) Maxwellian의 경우, $v > 3v_{\text{th}}$인 입자의 비율은 대략 $\exp(-9/2) \approx 0.011$ (1.1%)입니다. 이것을 수치적으로 계산합니다.

(b) kappa 분포의 경우, 다음을 적분하여 동일한 비율을 수치적으로 계산합니다:
   $$
   f_{\text{super}} = \frac{\int_{3v_{th}}^\infty f_\kappa(v) \, dv}{\int_0^\infty f_\kappa(v) \, dv}
   $$

(c) 증강 계수 (kappa/Maxwellian)는 무엇입니까?

(d) 태양풍과 자기권에서 kappa 분포가 관측되는 이유를 설명합니다.

---

### 문제 4: 엔트로피 보존

엔트로피 $S = -k_B\int f\ln f \, d^3x\,d^3v$가 Vlasov 방정식에 의해 보존됨을 분석적으로 보입니다.

(a) $\frac{dS}{dt} = -k_B\int\frac{\partial}{\partial t}(f\ln f) \, d^3x\,d^3v$로 시작합니다.

(b) $\frac{\partial}{\partial t}(f\ln f) = (1 + \ln f)\frac{\partial f}{\partial t}$를 사용하고 Vlasov 방정식을 대입합니다.

(c) 부분 적분하고 Lorentz 가속에 대해 $\nabla_x\cdot\mathbf{v} = 0$ 및 $\nabla_v\cdot\mathbf{a} = 0$를 사용합니다.

(d) 모든 항이 소멸하여, $\frac{dS}{dt} = 0$임을 보입니다.

---

### 문제 5: Langmuir 파동에 대한 선형화된 Vlasov-Poisson

자화되지 않은 플라즈마에서 1D 정전기 섭동을 고려합니다:

$$
f = f_0(v) + f_1(x, v, t)
$$

$$
E = E_1(x, t)
$$

여기서 $f_0$는 정지 상태의 Maxwellian입니다.

(a) $f_1$에 대한 선형화된 Vlasov 방정식을 작성합니다.

(b) $f_1$의 항으로 $E_1$에 대한 선형화된 Poisson 방정식을 작성합니다.

(c) 평면파 해를 가정합니다: $f_1 \propto e^{i(kx - \omega t)}$, $E_1 \propto e^{i(kx - \omega t)}$. 분산 관계를 유도합니다 (실수 $\omega$에 대해 Bohm-Gross 관계를 얻어야 합니다; Lesson 8에서 Landau 감쇠를 다룰 것입니다).

(d) $k\lambda_D \ll 1$에 대해, $\omega^2 \approx \omega_{pe}^2(1 + 3k^2\lambda_D^2)$임을 보입니다. 여기서 $\lambda_D = \sqrt{\epsilon_0 k_BT/(ne^2)}$.

**힌트**: 이 문제는 Lesson 8을 미리 보여줍니다. 완전한 해는 $v = \omega/k$에서 극점을 처리해야 합니다 (Landau 처방).

---

## 내비게이션

- **이전**: [자기 거울과 단열 불변량](./06_Magnetic_Mirrors_Adiabatic_Invariants.md)
- **다음**: [Landau 감쇠](./08_Landau_Damping.md)
