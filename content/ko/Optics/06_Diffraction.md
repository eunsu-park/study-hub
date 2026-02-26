# 06. 회절

[← 이전: 05. 파동광학 — 간섭](05_Wave_Optics_Interference.md) | [다음: 07. 편광 →](07_Polarization.md)

---

## 학습 목표

1. 하위헌스-프레넬 원리(Huygens-Fresnel principle)를 기술하고 이것이 어떻게 회절 현상을 설명하는지 설명한다
2. 단일 슬릿의 프라운호퍼(Fraunhofer) 회절 무늬를 유도하고 분석한다 — 세기 공식과 최솟값 조건 포함
3. 원형 조리개(circular aperture)의 에어리(Airy) 무늬를 계산하고 광학계의 분해능 한계를 결정하는 데 적용한다
4. 회절 격자(diffraction grating)를 분석한다 — 분해 능력, 자유 스펙트럼 범위, 분광학에서의 사용
5. 프라운호퍼(원거리장) 회절과 프레넬(Fresnel, 근거리장) 회절을 구분하고, 프레넬 수(Fresnel number)를 판별 기준으로 사용한다
6. 회절이 모든 광학계의 궁극적인 분해능 한계를 설정하는 방법을 설명한다
7. 회절 이론을 실용 시스템에 적용한다: 분광기, X선 결정학, 음향 유사 현상

---

## 왜 중요한가

회절(Diffraction)은 더 작은 세부를 보기 위해 단순히 배율을 높일 수 없는 이유이다. 이것은 스마트폰 카메라부터 허블 우주 망원경, 실리콘 위에 트랜지스터를 인쇄하는 리소그래피 기계에 이르기까지 모든 광학계의 근본적인 분해능 장벽이다. 회절을 이해하는 것은 분광기(회절 격자로 파장을 분리) 설계, X선 결정학 데이터 해석(회절로 분자 구조 규명), 그리고 광파장보다 작은 특징을 영상화하기 위해 전자 현미경이나 근거리장(near-field) 기법이 필요한 이유를 파악하는 데 필수적이다.

> **비유**: 좁은 틈이 있는 벽 근처에 돌을 연못에 떨어뜨린다고 상상해 보자. 파동은 총알처럼 틈을 직진하지 않는다 — 대신 반원형 패턴으로 반대편에서 퍼진다. 파장에 비해 틈이 좁을수록 파동이 더 많이 퍼진다. 이것이 회절이다. 빛도 마찬가지다: 파장에 비견되는 조리개를 통과하면 직선으로 진행하지 않는다.

---

## 1. 하위헌스-프레넬 원리(Huygens-Fresnel Principle)

### 1.1 하위헌스의 구성법(1690)

크리스티안 하위헌스(Christiaan Huygens)는 파면(wavefront) 위의 모든 점이 이차 구면 파동의 광원 역할을 한다고 제안했다. 이후 시각의 새로운 파면은 이 파동들의 포락면(접선면)이다.

### 1.2 프레넬의 개선(1818)

프레넬(Fresnel)은 두 가지 핵심 요소를 추가하여 하위헌스의 원리를 정량화했다:

1. **위상을 포함한 중첩**: 각 파동은 이동한 거리에 의해 결정되는 위상을 가진다
2. **경사 인자(obliquity factor)**: 후방 전파를 억제하는 경사 인자 $K(\chi) = \frac{1}{2}(1 + \cos\chi)$

하위헌스-프레넬 적분은 조리개 $\Sigma$에서 점 $P$에서의 장을 준다:

$$E(P) = -\frac{i}{\lambda} \iint_\Sigma E(Q) \frac{e^{ikr}}{r} K(\chi) \, dA$$

여기서 $Q$는 조리개 위의 점이고, $r$은 $Q$에서 $P$까지의 거리이며, $K(\chi)$는 경사 인자이다.

### 1.3 프라운호퍼 회절 vs. 프레넬 회절

분류는 **프레넬 수(Fresnel number)**에 따라 달라진다:

$$N_F = \frac{a^2}{\lambda L}$$

여기서 $a$는 조리개 크기, $L$은 관측 거리이다.

| 영역 | 프레넬 수 | 특징 |
|--------|---------------|-----------------|
| 프레넬(근거리장) | $N_F \geq 1$ | 복잡한 무늬, 파면 곡률이 중요 |
| 프라운호퍼(원거리장) | $N_F \ll 1$ | 단순한 무늬, 푸리에 변환에 해당 |

**프라운호퍼 조건**: $L \gg a^2/\lambda$. 가시광($\lambda = 500$ nm)에서 1 mm 슬릿의 경우: $L \gg 2$ m. 조리개에 렌즈를 놓으면 프라운호퍼 무늬를 초점면에 맺히게 할 수 있다 — 이것이 분광기의 표준 구성이다.

---

## 2. 단일 슬릿 프라운호퍼 회절

### 2.1 장치와 기하학

평면파가 폭 $a$의 슬릿을 비춘다. 멀리 있는 스크린(또는 렌즈의 초점면)에서 세기 무늬를 관측한다.

프라운호퍼 근사를 이용한 하위헌스-프레넬 적분에서, 각도 $\theta$ 방향에서의 전기장 진폭:

$$E(\theta) = E_0 \frac{\sin\beta}{\beta}$$

여기서:

$$\beta = \frac{\pi a \sin\theta}{\lambda}$$

함수 $\text{sinc}(\beta/\pi) = \sin(\beta)/\beta$는 **sinc 함수** — 광학과 신호 처리에서 가장 중요한 함수 중 하나이다.

### 2.2 세기 무늬

세기는 $|E|^2$에 비례한다:

$$I(\theta) = I_0 \left(\frac{\sin\beta}{\beta}\right)^2 = I_0 \,\text{sinc}^2\left(\frac{a\sin\theta}{\lambda}\right)$$

**최솟값(세기의 영점)**은 다음에서 발생한다:

$$a\sin\theta = m\lambda, \qquad m = \pm 1, \pm 2, \pm 3, \ldots$$

참고: $m = 0$은 최솟값이 아니라 중앙 최댓값이다.

**중앙 최댓값**: 중앙 봉우리의 각도 반폭(angular half-width):

$$\Delta\theta = \frac{\lambda}{a}$$

이것은 한쪽 첫 번째 최솟값에서 반대쪽 첫 번째 최솟값까지의 전체 폭이다: $2\lambda/a$.

### 2.3 2차 최댓값

2차 최댓값은 $m = 1, 2, 3, \ldots$에 대해 $\beta \approx (m + \frac{1}{2})\pi$에서 근사적으로 발생한다.

| 최댓값 | 위치 $a\sin\theta/\lambda$ | 상대 세기 |
|---------|-------------------------------|-------------------|
| 중앙 | 0 | 1.000 |
| 1차 | $\approx 1.43$ | 0.0472 (4.72%) |
| 2차 | $\approx 2.46$ | 0.0165 (1.65%) |
| 3차 | $\approx 3.47$ | 0.0083 (0.83%) |

2차 최댓값은 빠르게 감소한다 — 중앙 최댓값이 전체 회절 에너지의 약 90%를 포함한다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Single-slit Fraunhofer diffraction: intensity pattern
# The sinc^2 function is the Fourier transform of a rectangular aperture

def single_slit_intensity(theta, a, wavelength):
    """
    Calculate normalized single-slit diffraction intensity.
    theta: angle (radians)
    a: slit width (meters)
    wavelength: wavelength (meters)
    Returns: I/I_0
    """
    beta = np.pi * a * np.sin(theta) / wavelength
    # Handle the singularity at beta = 0 (central maximum)
    with np.errstate(divide='ignore', invalid='ignore'):
        sinc = np.where(np.abs(beta) < 1e-10, 1.0, np.sin(beta) / beta)
    return sinc**2

wavelength = 550e-9  # green light (m)
slit_widths = [5e-6, 20e-6, 100e-6]  # 5 um, 20 um, 100 um

theta = np.linspace(-0.15, 0.15, 2000)  # angle in radians

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: linear scale — shows the narrow central peak and secondary maxima
for a in slit_widths:
    I = single_slit_intensity(theta, a, wavelength)
    ax1.plot(np.rad2deg(theta), I, linewidth=1.5,
             label=f'a = {a*1e6:.0f} μm')

ax1.set_xlabel('Angle (degrees)', fontsize=12)
ax1.set_ylabel('Normalized Intensity $I/I_0$', fontsize=12)
ax1.set_title('Single-Slit Diffraction (Linear Scale)', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-8, 8)

# Right: log scale — reveals the secondary maxima structure
for a in slit_widths:
    I = single_slit_intensity(theta, a, wavelength)
    I_db = 10 * np.log10(np.maximum(I, 1e-10))  # convert to dB
    ax2.plot(np.rad2deg(theta), I_db, linewidth=1.5,
             label=f'a = {a*1e6:.0f} μm')

ax2.set_xlabel('Angle (degrees)', fontsize=12)
ax2.set_ylabel('Intensity (dB)', fontsize=12)
ax2.set_title('Single-Slit Diffraction (Log Scale)', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-8, 8)
ax2.set_ylim(-40, 2)

plt.tight_layout()
plt.savefig('single_slit_diffraction.png', dpi=150)
plt.show()
```

---

## 3. 원형 조리개: 에어리 무늬(Airy Pattern)

### 3.1 에어리 원판(Airy Disk)

지름 $D$인 원형 조리개의 프라운호퍼 회절 무늬는 원형 대칭이며 다음과 같이 기술된다:

$$I(\theta) = I_0 \left[\frac{2J_1(x)}{x}\right]^2$$

여기서 $x = \frac{\pi D\sin\theta}{\lambda}$이고 $J_1$은 제1종 1차 베셀 함수(Bessel function of the first kind)이다.

중앙의 밝은 원판을 **에어리 원판(Airy disk)**, 주변 고리를 **에어리 고리(Airy rings)**라 한다.

### 3.2 주요 특징

**첫 번째 어두운 고리** ($J_1$의 첫 번째 영점, $x = 3.8317$에서):

$$\sin\theta_1 = 1.22\frac{\lambda}{D}$$

이것이 **레일리 기준(Rayleigh criterion)**의 기초이다(레슨 04 참조). 에어리 원판의 각반지름은 원형 조리개를 가진 모든 광학계의 회절 한계 분해능을 결정한다.

초점면에서 **에어리 원판의 반지름**:

$$r_{\text{Airy}} = 1.22\frac{\lambda f}{D} = 1.22\lambda N$$

여기서 $N = f/D$는 f-수(f-number)이다.

### 3.3 원 내 에너지(Encircled Energy)

에어리 무늬는 대부분의 에너지를 중앙 원판에 집중시킨다:

| 특징 | 원 내 에너지 |
|---------|-----------------|
| 에어리 원판 (첫 번째 어두운 고리까지) | 83.8% |
| 두 번째 어두운 고리까지 | 91.0% |
| 세 번째 어두운 고리까지 | 93.8% |

나머지 $\sim 16\%$는 에어리 고리에 분산되어 밝은 점 광원 주위에 희미한 후광을 만든다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1

# Airy pattern: diffraction pattern of a circular aperture
# This is the point spread function (PSF) of any diffraction-limited telescope or camera

def airy_pattern(x):
    """
    Normalized Airy pattern: [2*J1(x)/x]^2
    x = pi * D * sin(theta) / lambda
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(x) < 1e-10, 1.0, (2 * j1(x) / x)**2)
    return result

# 1D profile
x = np.linspace(-15, 15, 1000)
I_airy = airy_pattern(x)

# 2D pattern
xx, yy = np.meshgrid(np.linspace(-10, 10, 500), np.linspace(-10, 10, 500))
rr = np.sqrt(xx**2 + yy**2)
I_2d = airy_pattern(rr)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Left: 1D intensity profile (linear)
ax1.plot(x, I_airy, 'b-', linewidth=2)
ax1.set_xlabel('$x = \\pi D \\sin\\theta / \\lambda$', fontsize=12)
ax1.set_ylabel('$I / I_0$', fontsize=12)
ax1.set_title('Airy Pattern (1D Profile)', fontsize=13)
ax1.axvline(3.83, color='red', linestyle='--', alpha=0.5, label='1st zero (3.83)')
ax1.axvline(-3.83, color='red', linestyle='--', alpha=0.5)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Middle: 2D pattern (log scale to show rings)
im = ax2.imshow(np.log10(np.maximum(I_2d, 1e-6)), extent=[-10, 10, -10, 10],
                cmap='inferno', vmin=-4, vmax=0)
ax2.set_xlabel('$x$', fontsize=12)
ax2.set_ylabel('$y$', fontsize=12)
ax2.set_title('Airy Pattern (2D, Log Scale)', fontsize=13)
# Draw circles at the dark rings
for zero in [3.83, 7.02, 10.17]:
    circle = plt.Circle((0, 0), zero, fill=False, color='white', linewidth=0.5, linestyle='--')
    ax2.add_patch(circle)
plt.colorbar(im, ax=ax2, label='$\\log_{10}(I/I_0)$')

# Right: encircled energy as a function of radius
r_values = np.linspace(0, 15, 500)
# Numerically integrate the Airy pattern in 2D to get encircled energy
# E(r) = integral of I(rho) * 2*pi*rho from 0 to r
dr = r_values[1] - r_values[0]
rho = np.linspace(0, 15, 2000)
I_rho = airy_pattern(rho)
# Cumulative integral: E(r) = integral_0^r I(rho) * rho * d(rho) (normalized)
encircled = np.cumsum(I_rho * rho * (rho[1] - rho[0]))
encircled = encircled / encircled[-1]  # normalize to 1

ax3.plot(rho, encircled * 100, 'b-', linewidth=2)
ax3.axhline(83.8, color='r', linestyle='--', alpha=0.5, label='83.8% (Airy disk)')
ax3.axvline(3.83, color='r', linestyle=':', alpha=0.5)
ax3.set_xlabel('Radius $x = \\pi D \\sin\\theta / \\lambda$', fontsize=12)
ax3.set_ylabel('Encircled Energy (%)', fontsize=12)
ax3.set_title('Encircled Energy', fontsize=13)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 15)
ax3.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('airy_pattern.png', dpi=150)
plt.show()
```

---

## 4. 회절 격자(Diffraction Grating)

### 4.1 다중 슬릿 회절

회절 격자는 거리 $d$ (**격자 주기(grating period)** 또는 **격자 간격(grating spacing)**)만큼 떨어진 $N$개의 평행하고 등간격인 슬릿(또는 홈)으로 구성된다. 세기 무늬는 단일 슬릿 회절과 다중 슬릿 간섭이 결합된 것이다:

$$I(\theta) = I_0 \left(\frac{\sin\beta}{\beta}\right)^2 \left(\frac{\sin(N\gamma)}{\sin\gamma}\right)^2$$

여기서:
- $\beta = \frac{\pi a\sin\theta}{\lambda}$ (단일 슬릿 회절 포락선, $a$ = 슬릿 폭)
- $\gamma = \frac{\pi d\sin\theta}{\lambda}$ (슬릿 간 간섭)

$(\sin N\gamma / \sin\gamma)^2$ 항은 다음에서 날카로운 **주 최댓값(principal maxima)**을 만든다:

$$d\sin\theta = m\lambda, \qquad m = 0, \pm 1, \pm 2, \ldots$$

인접한 주 최댓값 사이에는 $N - 2$개의 2차 최댓값과 $N - 1$개의 최솟값이 있다.

### 4.2 격자의 특성

**각 분산(angular dispersion)** — 단위 파장당 각도 변화량:

$$\frac{d\theta}{d\lambda} = \frac{m}{d\cos\theta}$$

높은 차수 $m$과 작은 격자 주기 $d$가 더 큰 분산을 준다.

**분해 능력(resolving power)** — 가까이 있는 두 파장을 구별하는 능력:

$$\mathcal{R} = \frac{\lambda}{\Delta\lambda} = mN$$

여기서 $m$은 회절 차수이고 $N$은 조명된 홈의 총 수이다. 1000 홈/mm의 격자를 5 cm 폭에 걸쳐 조명하면 $N = 50{,}000$이다. 1차에서: $\mathcal{R} = 50{,}000$ — $\Delta\lambda = 550/50{,}000 = 0.011$ nm의 파장 차이를 분해할 수 있다.

**자유 스펙트럼 범위(free spectral range)** — 다음 차수와 겹치기 전까지 해당 차수에서의 파장 범위:

$$\Delta\lambda_{\text{FSR}} = \frac{\lambda}{m}$$

### 4.3 블레이즈 격자(Blazed Gratings)

단순 격자는 스펙트럼 정보를 제공하지 않는 0차($m = 0$)에 대부분의 빛을 낭비한다. **블레이즈 격자(blazed grating)**는 홈이 블레이즈 각(blaze angle) $\theta_b$만큼 기울어져 있어 단일 슬릿 회절 포락선의 봉우리를 원하는 회절 차수로 향하게 한다. 이로써 해당 차수의 효율이 크게 높아진다.

블레이즈 조건:

$$d(\sin\theta_i + \sin\theta_m) = m\lambda \quad \text{및} \quad \theta_b = \frac{\theta_i + \theta_m}{2}$$

리트로우(Littrow) 구성($\theta_i = \theta_m$)에서: $2d\sin\theta_b = m\lambda$.

```python
import numpy as np
import matplotlib.pyplot as plt

# Diffraction grating: intensity pattern showing principal maxima
# and the single-slit envelope

def grating_intensity(theta, a, d, N, wavelength):
    """
    Calculate the intensity pattern of an N-slit diffraction grating.

    theta: angle (radians)
    a: slit width (m)
    d: grating period / slit spacing (m)
    N: number of slits
    wavelength: wavelength (m)

    Returns: I / I_0 (normalized to single-slit central maximum)
    """
    # Single-slit diffraction factor
    beta = np.pi * a * np.sin(theta) / wavelength
    with np.errstate(divide='ignore', invalid='ignore'):
        sinc_factor = np.where(np.abs(beta) < 1e-12, 1.0, np.sin(beta) / beta)

    # Multi-slit interference factor
    gamma = np.pi * d * np.sin(theta) / wavelength
    with np.errstate(divide='ignore', invalid='ignore'):
        # sin(N*gamma) / sin(gamma) → N when gamma → m*pi
        array_factor = np.where(
            np.abs(np.sin(gamma)) < 1e-12,
            N * np.cos(N * gamma) / np.cos(gamma),  # L'Hopital's rule
            np.sin(N * gamma) / np.sin(gamma)
        )

    return (sinc_factor * array_factor / N)**2 * N**2

wavelength = 550e-9    # green light
d = 2e-6               # grating period: 2 um (500 grooves/mm)
a = d * 0.4            # slit width: 40% of period

theta = np.linspace(-0.3, 0.3, 10000)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Show patterns for different numbers of slits
N_values = [2, 5, 20, 100]

for ax, N in zip(axes.flat, N_values):
    I = grating_intensity(theta, a, d, N, wavelength)
    I_envelope = grating_intensity(theta, a, a, 1, wavelength)  # single-slit envelope

    ax.plot(np.rad2deg(theta), I / I.max(), 'b-', linewidth=1,
            label=f'N = {N} slits')
    ax.plot(np.rad2deg(theta), I_envelope / I_envelope.max(), 'r--', linewidth=1,
            alpha=0.5, label='Single-slit envelope')

    ax.set_xlabel('Angle (degrees)', fontsize=11)
    ax.set_ylabel('Normalized Intensity', fontsize=11)
    ax.set_title(f'N = {N} slits, d = {d*1e6:.1f} μm', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-0.02, 1.1)

    # Mark diffraction orders
    for m in range(-3, 4):
        theta_m = np.arcsin(m * wavelength / d) if abs(m * wavelength / d) < 1 else None
        if theta_m is not None:
            ax.axvline(np.rad2deg(theta_m), color='gray', linestyle=':', alpha=0.3)

plt.suptitle(f'Diffraction Grating Patterns (λ = {wavelength*1e9:.0f} nm)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('diffraction_grating.png', dpi=150)
plt.show()
```

---

## 5. 프레넬 회절(Fresnel Diffraction)

### 5.1 프레넬 영역

프레넬 수 $N_F = a^2/(\lambda L) \geq 1$이 될 만큼 관측점이 충분히 가까우면, 파면의 곡률을 무시할 수 없다. 회절 무늬는 스크린까지의 정확한 거리에 따라 달라지며 꽤 복잡할 수 있다.

### 5.2 프레넬 구역(Fresnel Zones)

프레넬의 독창적인 접근법: 파면을 연속적인 구역들이 경로 길이에서 $\lambda/2$씩 차이나도록 동심 환형 영역(프레넬 구역)으로 나눈다.

$m$번째 프레넬 구역의 반지름:

$$r_m = \sqrt{m\lambda z}$$

여기서 $z$는 조리개에서 관측점까지의 거리이다.

특성:
- 인접한 구역은 거의 같은 진폭이지만 반대 위상에 기여한다
- 모든 구역에서의 전체 장은 첫 번째 구역 기여의 약 절반이다
- 교대 구역을 막는 것(**구역판(zone plate)**)은 수렴 렌즈처럼 작동한다

### 5.3 프레넬 구역판(Fresnel Zone Plate)

프레넬 구역판은 교대 프레넬 구역을 막거나 위상 변이시키는 회절 광학 소자이다. 렌즈처럼 빛을 집속시키며, 초점 거리는:

$$f_m = \frac{r_1^2}{\lambda} = \frac{r_m^2}{m\lambda}$$

구역판은 X선 현미경(X선에서는 $n \approx 1$이어서 굴절 렌즈가 실용적이지 않음)과 천문 전파 망원경에 사용된다.

### 5.4 직선 날에 의한 회절

직선 날(straight edge)에 의한 프레넬 회절은 특징적인 무늬를 만든다: 세기가 기하학적 그림자 경계에서 밝음에서 어둠으로 급격히 변하지 않는다. 대신:

- **그림자 바깥**: 세기는 방해받지 않은 값 주위에서 진동하며, 날에서 멀어질수록 무늬가 덜 두드러진다
- **그림자 경계에서**: 세기는 방해받지 않은 값의 정확히 25%이다 ($I = I_0/4$)
- **그림자 안쪽**: 세기가 영(0)으로 매끄럽게 감소한다

이것은 복소 평면에서 코르누 나선(Cornu spiral)을 그리는 **프레넬 적분(Fresnel integrals)** $C(u)$와 $S(u)$로 수학적으로 기술된다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel

# Fresnel diffraction at a straight edge
# The intensity near the geometric shadow boundary

# Fresnel integrals: S(u) and C(u)
u = np.linspace(-5, 5, 2000)  # dimensionless Fresnel parameter

# The Fresnel integrals give us the complex amplitude
S, C = fresnel(u)

# Complex amplitude: A(u) = (C(u) + 1/2) + i*(S(u) + 1/2) relative to total
# For a semi-infinite screen (straight edge), the field at position u is:
# E(u) = (1 + i)/2 * [(C(u) + 1/2) + i*(S(u) + 1/2)]
# but more directly, the intensity is:
# I/I_0 = 1/2 * [(C(u) + 1/2)^2 + (S(u) + 1/2)^2]
I_norm = 0.5 * ((C + 0.5)**2 + (S + 0.5)**2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: intensity profile at the straight edge
ax1.plot(u, I_norm, 'b-', linewidth=2)
ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Unobstructed ($I_0$)')
ax1.axhline(0.25, color='red', linestyle=':', alpha=0.5, label='Shadow edge ($I_0/4$)')
ax1.axvline(0, color='green', linestyle='--', alpha=0.5, label='Geometric shadow boundary')
ax1.fill_betweenx([0, 1.5], -5, 0, alpha=0.05, color='gray')

ax1.set_xlabel('Fresnel parameter $u$', fontsize=12)
ax1.set_ylabel('$I / I_0$', fontsize=12)
ax1.set_title('Fresnel Diffraction at a Straight Edge', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.5)
ax1.text(-3, 0.1, 'Shadow\nregion', fontsize=10, ha='center', color='gray')
ax1.text(3, 0.8, 'Illuminated\nregion', fontsize=10, ha='center', color='gray')

# Right: Cornu spiral (C(u) vs S(u))
ax2.plot(C, S, 'b-', linewidth=1.5)
ax2.plot(0.5, 0.5, 'ro', markersize=8, label='$u \\to +\\infty$')
ax2.plot(-0.5, -0.5, 'go', markersize=8, label='$u \\to -\\infty$')

# Mark some u values along the spiral
for u_mark in [-3, -2, -1, 0, 1, 2, 3]:
    s_val, c_val = fresnel(u_mark)
    ax2.plot(c_val, s_val, 'ko', markersize=4)
    ax2.annotate(f'u={u_mark}', xy=(c_val, s_val), fontsize=8,
                 xytext=(5, 5), textcoords='offset points')

ax2.set_xlabel('C(u)', fontsize=12)
ax2.set_ylabel('S(u)', fontsize=12)
ax2.set_title('Cornu Spiral', fontsize=13)
ax2.set_aspect('equal')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fresnel_edge_diffraction.png', dpi=150)
plt.show()
```

---

## 6. 회절과 분해능 한계

### 6.1 회절 한계(Diffraction Limit)

모든 영상 시스템은 회절에 의해 설정된 최소 분해 가능 특징 크기를 갖는다:

$$\Delta x_{\min} = \frac{0.61\lambda}{\text{NA}}$$

현미경 대물 렌즈의 경우, 또는 등가로:

$$\theta_{\min} = 1.22\frac{\lambda}{D}$$

조리개 $D$인 망원경이나 카메라의 경우.

이것은 *근본적인* 한계이다 — 렌즈 연마, 수차 보정, 계산 처리 등 어떠한 방법도 기존 원거리장 영상으로는 이를 극복할 수 없다. (STED, PALM/STORM, 구조 조명 등의 초분해능(super-resolution) 기법은 근방 형광 분자를 구별하는 교묘한 방법을 사용하여 회절 한계 이하의 분해능을 달성한다.)

### 6.2 회절과 푸리에 변환의 관계

심오한 결과: 프라운호퍼 회절 무늬는 조리개 함수의 **푸리에 변환(Fourier transform)**이다.

조리개가 투과 함수 $t(x, y)$를 갖는다면, 원거리장 전기장은:

$$E(k_x, k_y) \propto \mathcal{F}\{t(x, y)\}$$

여기서 $k_x = \frac{2\pi}{\lambda}\sin\theta_x$, $k_y = \frac{2\pi}{\lambda}\sin\theta_y$는 공간 주파수이다.

이것의 의미:
- **좁은 조리개** (공간에서 작음) → **넓은 회절 무늬** (공간 주파수에서 퍼짐)
- **넓은 조리개** (공간에서 큼) → **좁은 회절 무늬** (공간 주파수에서 날카로움)

이것은 시간-주파수 불확정 관계의 공간적 유사체이다: $\Delta x \cdot \Delta k_x \geq 2\pi$.

### 6.3 망원경과 카메라의 회절 한계

지름 $D$인 원형 조리개에서, 광학 전달 함수(OTF, optical transfer function)는 차단 공간 주파수 이상에서 영이다:

$$f_{\text{cutoff}} = \frac{D}{\lambda f} = \frac{1}{\lambda N}$$

여기서 $N = f/D$는 f-수이다. $f_{\text{cutoff}}$ 이상의 공간 주파수를 가진 특징은 단순히 전달되지 않는다 — 회복 불가능하게 손실된다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstration: Fraunhofer diffraction as Fourier transform
# Compare the diffraction patterns of different aperture shapes

def compute_diffraction_2d(aperture, pad_factor=4):
    """
    Compute the Fraunhofer diffraction pattern of a 2D aperture.
    Uses FFT (since Fraunhofer diffraction = Fourier transform).

    aperture: 2D numpy array (transmission function)
    pad_factor: zero-padding factor for better resolution
    Returns: intensity pattern (2D)
    """
    N = aperture.shape[0]
    padded = np.zeros((N * pad_factor, N * pad_factor))
    start = (N * pad_factor - N) // 2
    padded[start:start+N, start:start+N] = aperture

    # 2D FFT (shifted so zero frequency is at center)
    E = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    I = np.abs(E)**2
    I = I / I.max()  # normalize
    return I

N = 256
x = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, x)
R = np.sqrt(X**2 + Y**2)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Define four aperture shapes
apertures = {
    'Circular': R <= 0.5,
    'Square': (np.abs(X) <= 0.3) & (np.abs(Y) <= 0.3),
    'Slit (horizontal)': (np.abs(X) <= 0.4) & (np.abs(Y) <= 0.05),
    'Annular': (R >= 0.3) & (R <= 0.5),
}

for i, (name, aperture) in enumerate(apertures.items()):
    aperture = aperture.astype(float)
    I_diff = compute_diffraction_2d(aperture, pad_factor=4)

    # Top row: aperture
    axes[0, i].imshow(aperture, extent=[-1, 1, -1, 1], cmap='gray')
    axes[0, i].set_title(f'Aperture: {name}', fontsize=11)
    axes[0, i].set_xlabel('x')
    axes[0, i].set_ylabel('y')

    # Bottom row: diffraction pattern (log scale)
    M = I_diff.shape[0]
    extent = [-1, 1, -1, 1]  # normalized spatial frequency
    axes[1, i].imshow(np.log10(np.maximum(I_diff, 1e-6)),
                      extent=extent, cmap='inferno', vmin=-4, vmax=0)
    axes[1, i].set_title(f'Diffraction pattern (log)', fontsize=11)
    axes[1, i].set_xlabel('$k_x$')
    axes[1, i].set_ylabel('$k_y$')

plt.suptitle('Fraunhofer Diffraction = Fourier Transform of Aperture',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('diffraction_fourier_transform.png', dpi=150)
plt.show()
```

---

## 7. 회절의 응용

### 7.1 X선 결정학(X-Ray Crystallography)

X선($\lambda \sim 0.1$ nm)을 결정에 조사하면 규칙적인 원자 격자가 3차원 회절 격자 역할을 한다. 보강 간섭의 조건(브래그의 법칙, Bragg's law):

$$2d\sin\theta = n\lambda$$

여기서 $d$는 격자 간격이다. 회절 반점의 위치와 세기를 분석하여 결정 구조를 결정할 수 있다. 이 기법은 DNA, 단백질, 수많은 다른 분자의 구조를 규명했다.

### 7.2 분광기(Spectrometers)

회절 격자는 대부분의 분광기의 핵심이다. 빛은 슬릿으로 들어가 시준(collimate)되고, 격자에서 회절되어 검출기 배열에 집속된다. 각 파장은 서로 다른 위치에 집속된다:

$$x = f_{\text{camera}} \cdot \sin\theta_m \approx f_{\text{camera}} \cdot \frac{m\lambda}{d}$$

격자 방정식이 교정을 결정하고, 분해 능력이 스펙트럼 분해능을 결정하며, 블레이즈 각이 효율을 결정한다.

### 7.3 음향 및 수면파 유사 현상

회절은 모든 파동 현상에 보편적이다. 소리는 모퉁이 주위에서 회절된다(복도 모퉁이 너머의 사람 목소리를 들을 수 있다). 수면파는 항만 입구를 통해 회절된다. 이러한 거시적 유사체는 회절을 직접 관찰 가능하게 하며 강력한 교육적 시연을 제공한다.

### 7.4 일상 속 회절

- **CD/DVD**: 트랙 간격($\sim 1.6$ $\mu$m, CD 기준)이 반사 격자 역할을 하여 무지개 색상을 만든다
- **카메라 스타버스트(star burst)**: 카메라 조리개 날 주위에서의 회절이 밝은 점 광원에서 뾰족한 "스타버스트" 패턴을 만든다
- **홀로그래픽 보안 레이블**: 마이크로 구조 패턴이 회절을 통해 각도 의존적 색상을 만든다

---

## 연습 문제

### 연습 문제 1: 단일 슬릿 분석

폭 0.1 mm인 단일 슬릿을 633 nm 단색광으로 조명한다. 2 m 떨어진 스크린에서 회절 무늬를 관측한다.

(a) 중앙 최댓값의 폭(양쪽 첫 번째 최솟값 사이의 거리)은?

(b) 세 번째 최솟값의 각도는?

(c) 프레넬 수를 계산하라. 이 거리에서 프라운호퍼 근사가 유효한가?

### 연습 문제 2: 에어리 원판과 분해능

허블 우주 망원경의 주 거울 지름은 2.4 m이다.

(a) $\lambda = 500$ nm에서 에어리 원판의 각지름(angular diameter)을 계산하라.

(b) 달 표면(거리 384,000 km)에서 분해할 수 있는 최소 특징 크기는?

(c) 유효 초점 거리 57.6 m에서 검출기 위의 공간 분해능은?

(d) HST의 픽셀 크기 25 $\mu$m과 비교하라. 검출기가 에어리 원판을 적절하게 샘플링하는가?

### 연습 문제 3: 회절 격자 분광학

격자는 600 홈/mm이고 폭이 5 cm이다.

(a) 조명된 총 홈 수는?

(b) 1차와 2차에서의 분해 능력을 계산하라.

(c) 이 격자가 1차에서 나트륨 D 이중선($\lambda_1 = 589.0$ nm, $\lambda_2 = 589.6$ nm)을 분해할 수 있는가? 2차에서는?

(d) 2차에서 나트륨 이중선의 각도 분리는?

(e) 589 nm에서 2차의 자유 스펙트럼 범위는?

### 연습 문제 4: 프레넬 구역판

$\lambda = 550$ nm에서 초점 거리 $f = 1$ m인 렌즈 역할을 하는 프레넬 구역판을 설계하라.

(a) 첫 번째 구역의 반지름은?

(b) 열 번째 구역의 반지름은?

(c) 반지름 1 cm의 판을 만들려면 몇 개의 구역이 필요한가?

(d) 최소 특징 크기(가장 바깥 구역의 폭)는? 표준 리소그래피로 제작 가능한가?

### 연습 문제 5: 회절 한계 사진술

카메라는 50 mm 렌즈와 픽셀 피치 4 $\mu$m인 센서를 갖는다.

(a) 에어리 원판 지름이 픽셀 크기와 같아지는 f-수는? ($\lambda = 550$ nm 사용)

(b) 이 f-수 이하로 조리개를 좁히면 이미지 선명도는 어떻게 되는가?

(c) $f$/2.8과 $f$/16에서 최대 유용 분해능(선쌍/mm)을 계산하라.

---

## 요약

| 개념 | 핵심 공식 / 사실 |
|---------|-------------------|
| 하위헌스-프레넬 원리 | 파면 위의 모든 점이 이차 파동의 광원 |
| 프레넬 수 | $N_F = a^2/(\lambda L)$; $N_F \ll 1$이면 프라운호퍼 |
| 단일 슬릿 최솟값 | $a\sin\theta = m\lambda$, $m = \pm 1, \pm 2, \ldots$ |
| 단일 슬릿 세기 | $I = I_0 (\sin\beta/\beta)^2$, $\beta = \pi a\sin\theta/\lambda$ |
| 에어리 무늬 (원형) | $I = I_0 [2J_1(x)/x]^2$; 첫 번째 영점 $1.22\lambda/D$ |
| 회절 한계 | $\Delta x = 0.61\lambda/\text{NA}$; $\theta_R = 1.22\lambda/D$ |
| 격자 최댓값 | $d\sin\theta = m\lambda$ |
| 격자 분해 능력 | $\mathcal{R} = mN$ ($m$ = 차수, $N$ = 총 홈 수) |
| 격자 각 분산 | $d\theta/d\lambda = m/(d\cos\theta)$ |
| 격자 FSR | $\Delta\lambda_{\text{FSR}} = \lambda/m$ |
| 프레넬 구역 반지름 | $r_m = \sqrt{m\lambda z}$ |
| 프라운호퍼 ↔ 푸리에 | 원거리장 무늬는 조리개의 푸리에 변환 |
| 브래그의 법칙 | $2d\sin\theta = n\lambda$ (X선 회절) |

---

[← 이전: 05. 파동광학 — 간섭](05_Wave_Optics_Interference.md) | [다음: 07. 편광 →](07_Polarization.md)
