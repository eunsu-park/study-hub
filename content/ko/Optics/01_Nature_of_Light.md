# 01. 빛의 본질

[다음: 02. 기하광학 기초 →](02_Geometric_Optics_Fundamentals.md)

---

## 학습 목표

1. 빛의 파동-입자 이중성(wave-particle duality)을 설명하고, 두 모델이 서로를 어떻게 보완하는지 설명한다
2. 전자기 스펙트럼(electromagnetic spectrum)을 파장, 진동수, 광자 에너지와 연관 짓는다
3. 굴절률(refractive index)을 이용해 다양한 매질에서 빛의 속도를 계산한다
4. 분산(dispersion)과 그 물리적 기원을 진동수에 따른 굴절률 변화로 설명한다
5. 뉴턴에서 아인슈타인까지 이어지는 빛에 대한 이해의 역사적 발전 과정을 추적한다
6. 광자 에너지 관계식 $E = h\nu$를 분광학(spectroscopy)과 광자공학(photonics)의 실제 문제에 적용한다
7. 분산 매질에서 위상 속도(phase velocity), 군 속도(group velocity), 신호 속도(signal velocity)를 구별한다

---

## 왜 중요한가

빛은 우주의 주요 전령이다. 우리가 먼 별, 분자 구조, 양자 세계에 대해 알고 있는 거의 모든 것은 빛을 분석함으로써 얻어진다. 빛의 본질을 이해하는 것은 광학(optics), 광자공학(photonics), 통신, 의료 영상, 양자 컴퓨팅으로 가는 관문이다. 카메라 렌즈를 설계하든, 광섬유 네트워크를 구축하든, 천문 스펙트럼을 해석하든, 모든 것은 여기서 시작된다 — 수천 년 동안 인류를 당혹케 한 질문: *빛이란 무엇인가?*

> **비유**: 빛을 앞뒤가 있는 동전으로 생각하자. 한쪽 면인 *파동* 면은 간섭 무늬, 회절, 편광을 설명한다. 다른 쪽 면인 *입자* 면은 광전 효과(photoelectric effect)와 광자 계수(photon counting)를 설명한다. 두 면을 동시에 볼 수는 없지만, 둘 다 똑같이 실재한다. 이 상보성(complementarity)은 우리 이해의 한계가 아니라, 자연의 근본적인 특성이다.

---

## 1. 역사적 발전

### 1.1 입자설(Corpuscular Theory) (뉴턴, 1704)

아이작 뉴턴은 빛이 직선으로 이동하는 작은 입자("미립자")로 이루어져 있다고 제안했다. 이것은 반사(표면에서 입자가 튕겨 나오는 것)와 굴절(더 밀한 매질로 입자가 가속되는 것)을 설명했다. 뉴턴의 권위 덕분에 입자설은 한 세기 이상 지배적이었다.

**장점**: 직선 전파, 선명한 그림자, 반사를 설명했다.
**단점**: 간섭 또는 회절 무늬를 설명하지 못했다.

### 1.2 파동설(Wave Theory) (하위헌스, 1690; 영, 1801; 프레넬, 1818)

크리스티안 하위헌스(Christiaan Huygens)는 빛이 "발광 에테르(luminiferous aether)"라는 가상의 매질을 통해 전파되는 파동이라고 제안했다. 토머스 영(Thomas Young)의 이중 슬릿 실험(1801)은 파동 행동의 극적인 증거를 제공했고, 오귀스탱 장 프레넬(Augustin-Jean Fresnel)은 회절을 설명하는 엄밀한 수학적 파동 이론을 발전시켰다.

**핵심 예측**: 파동은 더 밀한 매질에서 *더 느리게* 이동해야 한다(뉴턴의 예측과 반대). 푸코(Foucault)의 측정(1850)은 이를 확인하여 파동설을 결정적으로 지지했다.

### 1.3 전자기 이론(Electromagnetic Theory) (맥스웰, 1865)

제임스 클러크 맥스웰(James Clerk Maxwell)은 전기와 자기를 네 개의 방정식으로 통합하고, 전자기파가 다음 속도로 전파됨을 보였다:

$$c = \frac{1}{\sqrt{\mu_0 \epsilon_0}} \approx 3 \times 10^8 \text{ m/s}$$

이는 측정된 빛의 속도와 일치했고, 맥스웰은 결론을 내렸다: *"빛은 전자기 교란이다."* 하인리히 헤르츠(Heinrich Hertz)는 1887년에 전자기파를 실험적으로 확인했다.

### 1.4 양자 혁명(Quantum Revolution) (플랑크, 1900; 아인슈타인, 1905)

막스 플랑크(Max Planck)는 에너지가 불연속적인 양자(quanta)로 방출된다고 제안하여 자외선 파국(ultraviolet catastrophe)을 해결했다: $E = h\nu$. 알베르트 아인슈타인(Albert Einstein)은 1905년에 이를 빛 자체에 확장하여, 빛을 광자(photon)의 흐름으로 취급함으로써 광전 효과를 설명했다 — 각 광자는 에너지 $E = h\nu$와 운동량 $p = h/\lambda$를 가진다.

### 1.5 현대적 종합: 양자 전기역학(QED)

리처드 파인만(Richard Feynman), 줄리안 슈윙거(Julian Schwinger), 도모나가 신이치로(Sin-Itiro Tomonaga)는 1940~50년대에 양자 전기역학(Quantum Electrodynamics, QED)을 발전시켜, 빛-물질 상호작용에 대한 가장 완전한 기술을 제공했다. QED에서 광자는 양자화된 전자기장의 들뜸(excitation)이다. 이 이론은 실험과 $10^{12}$분의 1보다 나은 놀라운 정밀도로 일치한다.

---

## 2. 전자기파로서의 빛

### 2.1 맥스웰 방정식과 파동 해

자유 공간에서 맥스웰 방정식은 파동 방정식을 이끌어낸다:

$$\nabla^2 \mathbf{E} = \mu_0 \epsilon_0 \frac{\partial^2 \mathbf{E}}{\partial t^2}$$

단색 평면파(monochromatic plane wave) 해는 다음 형태를 취한다:

$$\mathbf{E}(\mathbf{r}, t) = \mathbf{E}_0 \cos(\mathbf{k} \cdot \mathbf{r} - \omega t + \phi)$$

여기서:
- $\mathbf{E}_0$는 진폭 벡터(편광 방향을 결정한다)
- $\mathbf{k}$는 파동 벡터(wave vector)($|\mathbf{k}| = 2\pi/\lambda$, 전파 방향을 가리킨다)
- $\omega = 2\pi\nu$는 각진동수(angular frequency)
- $\phi$는 초기 위상

자기장 $\mathbf{B}$는 $\mathbf{E}$와 $\mathbf{k}$ 모두에 수직이다:

$$\mathbf{B} = \frac{1}{c} \hat{\mathbf{k}} \times \mathbf{E}$$

### 2.2 전자기 스펙트럼

전자기 스펙트럼(electromagnetic spectrum)은 엄청난 범위의 파장에 걸쳐 있다:

| 영역 | 파장 | 진동수 | 광자 에너지 |
|------|------|--------|------------|
| 전파(Radio) | > 1 m | < 300 MHz | < 1.24 $\mu$eV |
| 마이크로파(Microwave) | 1 mm – 1 m | 300 MHz – 300 GHz | 1.24 $\mu$eV – 1.24 meV |
| 적외선(Infrared) | 700 nm – 1 mm | 300 GHz – 430 THz | 1.24 meV – 1.77 eV |
| **가시광선(Visible)** | **400 – 700 nm** | **430 – 750 THz** | **1.77 – 3.10 eV** |
| 자외선(Ultraviolet) | 10 – 400 nm | 750 THz – 30 PHz | 3.10 – 124 eV |
| X선(X-ray) | 0.01 – 10 nm | 30 PHz – 30 EHz | 124 eV – 124 keV |
| 감마선(Gamma ray) | < 0.01 nm | > 30 EHz | > 124 keV |

가시 스펙트럼 — 우리 눈이 감지할 수 있는 좁은 띠 — 은 이 범위의 극히 작은 부분이다:

| 색상 | 파장 (nm) | 진동수 (THz) |
|------|-----------|-------------|
| 빨강 | 620 – 700 | 430 – 484 |
| 주황 | 590 – 620 | 484 – 508 |
| 노랑 | 570 – 590 | 508 – 526 |
| 초록 | 495 – 570 | 526 – 606 |
| 파랑 | 450 – 495 | 606 – 668 |
| 보라 | 380 – 450 | 668 – 789 |

### 2.3 기본 관계식

파장 $\lambda$, 진동수 $\nu$, 속도 $c$의 관계:

$$c = \lambda \nu$$

진공에서 $c = 299\,792\,458$ m/s (1983년 이후 미터의 정의에 의해 정확한 값).

파수(wave number) $k$와 각진동수 $\omega$:

$$k = \frac{2\pi}{\lambda}, \qquad \omega = 2\pi\nu, \qquad c = \frac{\omega}{k}$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize an electromagnetic plane wave propagating in the z-direction
# E oscillates in the x-direction, B oscillates in the y-direction

z = np.linspace(0, 4 * np.pi, 500)  # spatial coordinate (in units of wavelength/2pi)
t = 0  # snapshot at t = 0

# Normalized fields: E_x and B_y for a plane wave
E_x = np.sin(z)       # electric field component
B_y = np.sin(z)       # magnetic field component (in phase, perpendicular)

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111, projection='3d')

# Plot E-field (red) oscillating in the x-z plane
ax.plot(z, E_x, np.zeros_like(z), color='red', linewidth=2, label='E-field (x)')
# Plot B-field (blue) oscillating in the y-z plane
ax.plot(z, np.zeros_like(z), B_y, color='blue', linewidth=2, label='B-field (y)')

ax.set_xlabel('z (propagation)')
ax.set_ylabel('E_x')
ax.set_zlabel('B_y')
ax.set_title('Electromagnetic Plane Wave')
ax.legend()
plt.tight_layout()
plt.savefig('em_plane_wave.png', dpi=150)
plt.show()
```

---

## 3. 입자로서의 빛: 광자

### 3.1 광자 에너지

각 광자는 에너지를 가진다:

$$E = h\nu = \frac{hc}{\lambda}$$

여기서 $h = 6.626 \times 10^{-34}$ J$\cdot$s는 플랑크 상수(Planck's constant)이다.

계산에 편리한 형태:

$$E \text{ (eV)} = \frac{1240}{\lambda \text{ (nm)}}$$

이는 초록 광자($\lambda = 550$ nm)가 약 2.25 eV의 에너지를 가짐을 의미한다 — 망막에서 광화학 반응을 유발하기에는 충분하지만, 대부분의 원자를 이온화하기에는 부족하다.

### 3.2 광자 운동량

질량이 없음에도 불구하고, 광자는 운동량을 가진다:

$$p = \frac{E}{c} = \frac{h}{\lambda} = \frac{h\nu}{c}$$

이것이 **복사 압력(radiation pressure)**의 기초이다 — 태양빛은 완전히 흡수하는 표면에 약 4.6 $\mu$Pa의 압력을 가한다. 우주선의 태양돛(solar sail)은 이 작지만 지속적인 힘을 활용한다.

### 3.3 광전 효과

광전 효과(photoelectric effect)에 대한 아인슈타인의 설명(1905)은 광자의 첫 번째 직접적인 증거였다:

$$E_k = h\nu - \phi$$

여기서 $E_k$는 방출된 전자의 최대 운동 에너지이고 $\phi$는 물질의 일함수(work function)이다. 핵심 관측:
- 문턱 진동수 $\nu_0 = \phi/h$ 이하에서는 강도와 관계없이 전자가 방출되지 않는다
- 문턱 이상에서는 전자 에너지가 진동수에 선형으로 증가한다
- 방출은 즉각적이다 — 매우 낮은 강도에서도 시간 지연이 없다

```python
import numpy as np
import matplotlib.pyplot as plt

# Photoelectric effect: kinetic energy vs. photon frequency
# Demonstrates that E_k depends linearly on frequency, not intensity

h = 6.626e-34         # Planck's constant (J·s)
eV = 1.602e-19        # electron-volt to Joules conversion

# Work functions for common metals (in eV)
metals = {
    'Cesium (Cs)': 2.1,
    'Sodium (Na)': 2.28,
    'Zinc (Zn)': 4.33,
    'Platinum (Pt)': 5.64,
}

# Frequency range: 0 to 2000 THz
nu = np.linspace(0, 2000e12, 500)
E_photon_eV = h * nu / eV  # photon energy in eV

fig, ax = plt.subplots(figsize=(10, 6))

for metal, phi in metals.items():
    # Kinetic energy is max(0, E_photon - phi)
    # We only plot the region where E_k > 0 (above threshold)
    E_k = E_photon_eV - phi
    mask = E_k > 0
    ax.plot(nu[mask] / 1e12, E_k[mask], linewidth=2, label=f'{metal}, $\\phi$ = {phi} eV')
    # Mark the threshold frequency with a vertical dotted line
    nu_threshold = phi * eV / h
    ax.axvline(nu_threshold / 1e12, linestyle=':', alpha=0.5)

ax.set_xlabel('Frequency (THz)', fontsize=12)
ax.set_ylabel('Max Kinetic Energy (eV)', fontsize=12)
ax.set_title('Photoelectric Effect: $E_k = h\\nu - \\phi$', fontsize=14)
ax.legend(fontsize=10)
ax.set_xlim(0, 2000)
ax.set_ylim(0, 6)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('photoelectric_effect.png', dpi=150)
plt.show()
```

### 3.4 파동-입자 이중성: 드브로이 관계식

루이 드브로이(Louis de Broglie, 1924)는 *모든* 물질이 파동 같은 성질을 가진다고 제안했다:

$$\lambda = \frac{h}{p} = \frac{h}{mv}$$

광자의 경우 이는 $p = h/\lambda$와 일관된다. 전자, 중성자, 심지어 큰 분자에 대해서도 물질파 간섭이 실험적으로 확인되었다. 닐스 보어(Niels Bohr)의 상보성 원리(complementarity principle)는 파동과 입자 기술이 상보적임을 — 실험 장치가 어느 측면이 드러날지를 결정한다고 — 말한다.

---

## 4. 매질에서 빛의 속도

### 4.1 굴절률

빛이 물질에 들어가면 매질 내의 전자와 상호작용한다. 매질에서 빛의 속도는:

$$v = \frac{c}{n}$$

여기서 $n$은 매질의 **굴절률(refractive index)**이다. 일반 물질에서 $v \leq c$이므로, $n \geq 1$이다.

| 물질 | 굴절률 $n$ |
|------|-----------|
| 진공 | 1 (정확한 값) |
| 공기 (표준 상태) | 1.000293 |
| 물 | 1.333 |
| 유리 (크라운) | 1.52 |
| 유리 (플린트) | 1.62 |
| 다이아몬드 | 2.417 |
| 실리콘 | 3.48 (1550 nm에서) |

> **비유**: 마칭 밴드가 포장도로(빠른 매질)에서 비스듬히 진흙 밭(느린 매질)으로 건너가는 상상을 해보자. 진흙에 먼저 닿는 쪽이 느려져서 전체 줄이 회전하게 된다 — 이것이 굴절이다. "굴절률"은 포장도로에서의 걷는 속도와 진흙에서의 걷는 속도의 비와 같다.

### 4.2 굴절률의 미시적 기원

원자 수준에서 굴절률은 전자기장에 의한 속박 전자의 분극(polarization)에서 비롯된다. 로렌츠 진동자 모델(Lorentz oscillator model)은 각 전자를 전기장에 의해 구동되는 감쇠 조화 진동자로 취급한다:

$$n^2(\omega) = 1 + \frac{Nq^2}{\epsilon_0 m_e} \sum_j \frac{f_j}{\omega_{0j}^2 - \omega^2 - i\gamma_j \omega}$$

여기서 $f_j$는 진동자 세기(oscillator strengths), $\omega_{0j}$는 공명 진동수, $\gamma_j$는 감쇠 상수이다. 이 모델은 다음을 포착한다:
- **정상 분산(Normal dispersion)**: $n$이 $\omega$와 함께 증가한다 (공명 진동수에서 멀리 떨어진 경우)
- **이상 분산(Anomalous dispersion)**: $n$이 $\omega$와 함께 감소한다 (공명 진동수 근처)
- **흡수(Absorption)**: $n$의 허수부가 흡수 계수를 준다

### 4.3 위상 속도와 군 속도

분산 매질에서 서로 다른 진동수 성분은 서로 다른 속도로 이동한다:

- **위상 속도(Phase velocity)**: $v_p = \frac{\omega}{k} = \frac{c}{n(\omega)}$ — 단일 진동수 성분의 속도
- **군 속도(Group velocity)**: $v_g = \frac{d\omega}{dk}$ — 파속(wave packet) 외포선(envelope)의 속도

이들의 관계:

$$v_g = v_p - \lambda \frac{dv_p}{d\lambda} = \frac{c}{n + \omega \frac{dn}{d\omega}}$$

군 속도는 일반적으로 에너지와 정보가 전파되는 속도이다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstrate the difference between phase velocity and group velocity
# by showing a wave packet in a dispersive medium

x = np.linspace(0, 100, 2000)  # spatial coordinate

# Central wave number and frequency
k0 = 1.0      # central wave number
omega0 = 1.0   # central frequency (v_p = omega/k = 1.0)

# Dispersion: omega = omega0 + v_g * (k - k0) + 0.5 * beta2 * (k - k0)^2
# For demonstration: v_p = 1.0, v_g = 0.6 (group slower than phase)
v_p = 1.0
v_g = 0.6

# Build a Gaussian wave packet at t=0 and t=30
sigma = 5.0  # width of the envelope
times = [0, 30]

fig, axes = plt.subplots(len(times), 1, figsize=(12, 6), sharex=True)

for ax, t in zip(axes, times):
    # Envelope moves at group velocity
    envelope = np.exp(-((x - 50 - v_g * t) ** 2) / (2 * sigma ** 2))
    # Carrier wave moves at phase velocity
    carrier = np.cos(k0 * (x - v_p * t))
    # The wave packet is the product of envelope and carrier
    wave_packet = envelope * carrier

    ax.plot(x, wave_packet, 'b-', linewidth=1, label='Wave packet')
    ax.plot(x, envelope, 'r--', linewidth=1.5, label='Envelope ($v_g$)')
    ax.plot(x, -envelope, 'r--', linewidth=1.5)
    ax.set_ylabel('Amplitude')
    ax.set_title(f't = {t} (phase moves at $v_p$ = {v_p}, envelope at $v_g$ = {v_g})')
    ax.legend(loc='upper right')
    ax.set_ylim(-1.3, 1.3)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Position x')
plt.tight_layout()
plt.savefig('phase_vs_group_velocity.png', dpi=150)
plt.show()
```

---

## 5. 분산

### 5.1 코시 방정식과 셀마이어 방정식

흡수 공명 진동수에서 멀리 떨어진 투명 물질의 경우, 굴절률은 파장에 따라 변한다. 두 가지 경험적 모델:

**코시 방정식(Cauchy's equation)** (근사):

$$n(\lambda) = A + \frac{B}{\lambda^2} + \frac{C}{\lambda^4} + \cdots$$

**셀마이어 방정식(Sellmeier equation)** (더 정확한, 물리 기반):

$$n^2(\lambda) = 1 + \sum_i \frac{B_i \lambda^2}{\lambda^2 - C_i}$$

여기서 $B_i$와 $C_i$는 공명 진동수와 진동자 세기에 관련된 경험적으로 결정된 상수이다.

### 5.2 색수차 분산과 프리즘

프리즘은 굴절률이 파장에 따라 달라지기 때문에 백색광을 구성 색상으로 분리한다. 보라색 빛($n$ 더 큼)은 빨간색 빛($n$ 더 작음)보다 더 많이 굴절된다.

프리즘의 **각 분산(angular dispersion)**은:

$$\frac{d\theta}{d\lambda} = \frac{d\theta}{dn} \cdot \frac{dn}{d\lambda}$$

**아베 수(Abbe number)** $V_d$는 광학 유리의 분산을 특징짓는다:

$$V_d = \frac{n_d - 1}{n_F - n_C}$$

여기서 $n_d$, $n_F$, $n_C$는 특정 파장(587.6 nm, 486.1 nm, 656.3 nm)에서의 굴절률이다. 아베 수가 높을수록 분산이 낮음을 의미한다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Sellmeier equation for BK7 glass (common optical glass)
# Demonstrates normal dispersion: n increases as wavelength decreases

def sellmeier_bk7(wavelength_um):
    """
    Compute refractive index of BK7 glass using the Sellmeier equation.
    wavelength_um: wavelength in micrometers
    Returns: refractive index n
    """
    # Sellmeier coefficients for Schott BK7 (sourced from Schott catalog)
    B1, B2, B3 = 1.03961212, 0.231792344, 1.01046945
    C1, C2, C3 = 0.00600069867, 0.0200179144, 103.560653  # in um^2

    lam2 = wavelength_um ** 2
    n_sq = 1 + (B1 * lam2 / (lam2 - C1)
                + B2 * lam2 / (lam2 - C2)
                + B3 * lam2 / (lam2 - C3))
    return np.sqrt(n_sq)

# Wavelength range: 300 nm to 2000 nm
wavelengths_nm = np.linspace(300, 2000, 500)
wavelengths_um = wavelengths_nm / 1000.0  # convert to micrometers

n_bk7 = sellmeier_bk7(wavelengths_um)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: n vs wavelength — shows normal dispersion (n decreases with wavelength)
ax1.plot(wavelengths_nm, n_bk7, 'b-', linewidth=2)
ax1.set_xlabel('Wavelength (nm)', fontsize=12)
ax1.set_ylabel('Refractive Index n', fontsize=12)
ax1.set_title('BK7 Glass: Sellmeier Dispersion Curve', fontsize=13)
ax1.axvspan(380, 700, alpha=0.15, color='yellow', label='Visible range')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Right: group index n_g = n - lambda * dn/dlambda
# Group index determines the group velocity: v_g = c / n_g
dn_dlam = np.gradient(n_bk7, wavelengths_um)  # dn/d(lambda in um)
n_group = n_bk7 - wavelengths_um * dn_dlam

ax2.plot(wavelengths_nm, n_group, 'r-', linewidth=2, label='Group index $n_g$')
ax2.plot(wavelengths_nm, n_bk7, 'b--', linewidth=1.5, label='Phase index $n$')
ax2.set_xlabel('Wavelength (nm)', fontsize=12)
ax2.set_ylabel('Index', fontsize=12)
ax2.set_title('Phase Index vs. Group Index', fontsize=13)
ax2.axvspan(380, 700, alpha=0.15, color='yellow')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bk7_dispersion.png', dpi=150)
plt.show()
```

### 5.3 무지개: 자연 속의 분산

무지개는 분산의 아름다운 표현이다. 태양빛이 물방울에 들어가면, 표면에서 굴절되고, 내부에서 반사된 다음, 나올 때 다시 굴절된다. $n_{\text{water}}$가 파장에 따라 달라지기 때문에, 서로 다른 색이 약간 다른 각도로 나온다:

- **1차 무지개(Primary rainbow)**: 내부 반사 한 번, 반태양점(antisolar point)으로부터 약 42$^\circ$에서 보인다. 빨강이 바깥쪽, 보라가 안쪽.
- **2차 무지개(Secondary rainbow)**: 내부 반사 두 번, 약 51$^\circ$에서 보인다. 색이 반전된다(빨강이 안쪽, 보라가 바깥쪽). 추가 반사 손실로 인해 더 희미하다.
- **알렉산더의 어두운 띠(Alexander's dark band)**: 1차와 2차 무지개 사이의 영역은 이 각도 범위로 빛이 향하지 않기 때문에 더 어둡게 보인다.

---

## 6. 에너지와 강도

### 6.1 포인팅 벡터

전자기파의 에너지 흐름은 **포인팅 벡터(Poynting vector)**로 기술된다:

$$\mathbf{S} = \frac{1}{\mu_0} \mathbf{E} \times \mathbf{B}$$

시간 평균 강도(단위 면적당 전력)는:

$$I = \langle |\mathbf{S}| \rangle = \frac{1}{2} c \epsilon_0 E_0^2 = \frac{E_0^2}{2\mu_0 c}$$

맑은 날 지표면에서 태양빛의 경우, $I \approx 1000$ W/m$^2$이다.

### 6.2 광자 플럭스

강도는 광자 플럭스(photon flux) $\Phi$ (단위 면적당 초당 광자 수)로도 표현될 수 있다:

$$I = \Phi \cdot h\nu$$

1 mW 빨간색 레이저 포인터($\lambda = 650$ nm)는 초당 약 $3.3 \times 10^{15}$개의 광자를 방출한다 — 대부분의 실용적 목적으로 고전 파동 광학이 잘 작동하는 이유인 놀라운 숫자이다.

```python
import numpy as np

# Calculate photon flux for a laser beam
# This helps build intuition for why classical optics works at everyday power levels

h = 6.626e-34       # Planck's constant (J·s)
c = 3e8             # speed of light (m/s)

# Laser parameters
power_mW = 1.0                      # laser power in milliwatts
wavelength_nm = 650                  # red laser pointer
beam_diameter_mm = 1.0               # typical beam diameter

# Derived quantities
power_W = power_mW * 1e-3
wavelength_m = wavelength_nm * 1e-9
beam_area = np.pi * (beam_diameter_mm * 1e-3 / 2) ** 2  # beam cross-section area

# Photon energy
E_photon = h * c / wavelength_m
print(f"Photon energy: {E_photon:.3e} J = {E_photon / 1.602e-19:.3f} eV")

# Photon flux (total photons per second)
photon_rate = power_W / E_photon
print(f"Photon emission rate: {photon_rate:.3e} photons/s")

# Intensity (power per unit area)
intensity = power_W / beam_area
print(f"Beam intensity: {intensity:.1f} W/m^2")

# Photon flux density (photons per second per unit area)
flux_density = photon_rate / beam_area
print(f"Photon flux density: {flux_density:.3e} photons/(s·m^2)")

# At this rate, individual photon granularity is utterly undetectable
# by any classical measurement — validating the wave description
```

---

## 7. 광학 경로 길이와 페르마 원리 (미리보기)

매질을 통과하는 **광학 경로 길이(Optical Path Length, OPL)**는:

$$\text{OPL} = \int_A^B n(s) \, ds$$

여기서 $n(s)$는 경로를 따른 굴절률이고 $ds$는 미소 경로 요소이다.

**페르마 원리(Fermat's Principle)**는 빛이 광학 경로 길이가 *정류(stationary)*인 경로(보통은 최솟값)를 따른다고 말한다. 이 단일 원리는 다음을 통합한다:
- **반사(Reflection)**: 입사각 = 반사각
- **굴절(Refraction)**: 스넬 법칙
- **경사 굴절률 매질에서의 곡선 경로** (예: 신기루, GRIN 렌즈)

페르마 원리는 [레슨 02](02_Geometric_Optics_Fundamentals.md)에서 깊이 탐구할 것이다.

---

## 연습 문제

### 연습 1: 광자 에너지 계산

수소 원자가 $n=3$에서 $n=2$로 전이(H-alpha 선)하는 동안 광자를 방출한다.

(a) $\frac{1}{\lambda} = R_H \left(\frac{1}{n_1^2} - \frac{1}{n_2^2}\right)$ ($R_H = 1.097 \times 10^7$ m$^{-1}$)를 이용해 방출된 광자의 파장을 계산하라.

(b) 이 빛은 무슨 색인가?

(c) 광자의 에너지(eV)와 운동량을 계산하라.

### 연습 2: 매질에서의 속도

빛이 공기에서 다이아몬드($n = 2.417$)로 들어간다.

(a) 다이아몬드 내부에서 빛의 속도는 얼마인가?

(b) 다이아몬드 내부에서 589 nm (나트륨 D선) 빛의 파장은 얼마인가?

(c) 빛이 다이아몬드에 들어갈 때 진동수가 변하는가? 왜 그런지 또는 왜 그렇지 않은지 설명하라.

### 연습 3: 분산 분석

BK7 유리에 대한 셀마이어 방정식(위 코드에 주어진 계수)을 사용하여:

(a) 486.1 nm (F선), 587.6 nm (d선), 656.3 nm (C선)에서 굴절률을 계산하라.

(b) 아베 수 $V_d$를 계산하라.

(c) 500 nm에서 군 속도를 계산하라. 위상 속도보다 빠른가 느린가?

### 연습 4: 광자 계수

CCD 카메라 센서에 5 $\mu$m $\times$ 5 $\mu$m 면적의 픽셀이 있다. 1초 노출 동안 한 픽셀에 도달하는 550 nm 빛의 강도는 $10^{-6}$ W/m$^2$이다.

(a) 노출 동안 이 픽셀에 몇 개의 광자가 닿는가?

(b) 양자 효율이 70%라면, 몇 개의 광전자가 생성되는가?

(c) 광자 샷 노이즈(photon shot noise)가 유의미해지는 — 즉 $\sqrt{N} / N > 10\%$인 경우 — 빛의 레벨(강도)은?

### 연습 5: 위상 속도 대 군 속도

$n(\lambda) = 1.5 + \frac{3 \times 10^4}{\lambda^2}$ ($\lambda$는 nm 단위)인 매질에서:

(a) $\lambda = 500$ nm에서 위상 속도를 계산하라.

(b) 이 파장에서 $dn/d\lambda$를 계산하라.

(c) 군 속도를 구하라. 이 매질은 500 nm에서 정상 분산(normal dispersion)인가 이상 분산(anomalous dispersion)인가?

---

## 요약

| 개념 | 핵심 공식 / 사실 |
|------|----------------|
| 파동-입자 이중성 | 빛은 파동(간섭, 회절)과 입자(광전 효과, 콤프턴 산란) 모두로 행동한다 |
| 광자 에너지 | $E = h\nu = hc/\lambda$; 편리한 형태: $E$(eV) = 1240/$\lambda$(nm) |
| 광자 운동량 | $p = h/\lambda = E/c$ |
| 전자기파 관계 | $c = \lambda\nu = \omega/k$ |
| 매질에서의 속도 | $v = c/n$, $n$은 굴절률 |
| 위상 속도 | $v_p = \omega/k = c/n(\omega)$ |
| 군 속도 | $v_g = d\omega/dk = c/(n + \omega \cdot dn/d\omega)$ |
| 포인팅 벡터 | $\mathbf{S} = (\mathbf{E} \times \mathbf{B})/\mu_0$; 강도 $I = \frac{1}{2}c\epsilon_0 E_0^2$ |
| 셀마이어 방정식 | $n^2(\lambda) = 1 + \sum_i B_i\lambda^2/(\lambda^2 - C_i)$ |
| 아베 수 | $V_d = (n_d - 1)/(n_F - n_C)$; 분산을 측정한다 |
| 역사적 발전 | 뉴턴(미립자) → 하위헌스/영/프레넬(파동) → 맥스웰(전자기) → 아인슈타인/플랑크(광자) → QED |

---

[다음: 02. 기하광학 기초 →](02_Geometric_Optics_Fundamentals.md)
