# 13. 복사와 안테나

[← 이전: 12. 도파관과 공동](12_Waveguides_and_Cavities.md) | [다음: 14. 상대론적 전자기학 →](14_Relativistic_Electrodynamics.md)

## 학습 목표

1. 맥스웰 방정식으로부터 지연 퍼텐셜(retarded potential)을 유도하고 지연 시간(retarded time) 개념을 이해한다
2. 원거리장(far-field) 근사에서 진동 전기 쌍극자의 복사장(radiation field)을 계산한다
3. 라머 공식(Larmor formula)을 적용하여 가속하는 전하의 복사 전력을 계산한다
4. 복사 저항(radiation resistance)과 안테나 효율에서 그 역할을 이해한다
5. 반파장 쌍극자 안테나를 분석하고 복사 패턴을 계산한다
6. 간단한 안테나 배열을 설계하고 위상 제어를 통한 빔 조향(beam steering)을 이해한다
7. 지향성(directivity), 이득(gain), 유효 면적(effective area)으로 안테나 성능을 정량화한다

모든 라디오 방송국, 셀 타워, Wi-Fi 공유기, 그리고 하늘의 모든 별은 전자기 복사의 원천이다. 복사는 전하가 가속할 때마다 발생하며, 방출된 장(field)은 에너지와 운동량을 무한히 멀리까지 전달한다. 핵심 과제는 소스로부터 멀리 떨어진 장을 계산하고, 실용적 목적에 맞게 복사 패턴을 형성하는 것이다. 이 레슨에서는 빛의 유한한 속도를 퍼텐셜 형식에 내장한 지연 퍼텐셜에서 출발하여, 쌍극자 복사, 라머 공식을 거쳐, 전자기파를 놀라운 정밀도로 형성하고 지향하는 안테나 공학으로 나아간다.

> **유추**: 연못에 돌을 던지면 물결이 바깥쪽으로 퍼져 나간다. 물결은 돌이 언제, 어디에 떨어졌는지에 대한 정보를 담고 있지만, 먼 지점에는 거리에 비례하는 지연 후에 도달한다. 이것이 바로 지연 퍼텐셜이 기술하는 것이다: 가속하는 전하로부터 나온 전자기 "물결"은 지연 시간 $t_r = t - |\mathbf{r} - \mathbf{r}'|/c$에 멀리 있는 관측자에게 도달하며, 소스의 과거 상태를 전달한다.

---

## 1. 지연 퍼텐셜

### 1.1 인과율의 문제

정전기학에서 스칼라 퍼텐셜은 다음과 같다:

$$V(\mathbf{r}) = \frac{1}{4\pi\epsilon_0}\int \frac{\rho(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} \, d^3r'$$

그러나 시간에 따라 변하는 소스에 대해서는 이것이 옳을 수 없다 — $\mathbf{r}'$에서의 $\rho$ 변화가 $\mathbf{r}$에서의 $V$에 즉각적으로 영향을 미친다고 암시하여 인과율을 위반하기 때문이다.

### 1.2 지연 시간

올바른 해는 **지연 시간(retarded time)**을 사용한다:

$$t_r = t - \frac{|\mathbf{r} - \mathbf{r}'|}{c}$$

이것은 $\mathbf{r}'$에 있는 소스로부터의 신호가 빛의 속도로 이동하여 시각 $t$에 관측점 $\mathbf{r}$에 도달하기 위해 출발해야 하는 시각이다.

### 1.3 지연 퍼텐셜

지연 퍼텐셜은 다음과 같다:

$$\boxed{V(\mathbf{r}, t) = \frac{1}{4\pi\epsilon_0}\int \frac{\rho(\mathbf{r}', t_r)}{|\mathbf{r} - \mathbf{r}'|} \, d^3r'}$$

$$\boxed{\mathbf{A}(\mathbf{r}, t) = \frac{\mu_0}{4\pi}\int \frac{\mathbf{J}(\mathbf{r}', t_r)}{|\mathbf{r} - \mathbf{r}'|} \, d^3r'}$$

이들은 로렌츠 게이지 조건 $\nabla \cdot \mathbf{A} + \mu_0\epsilon_0 \partial V/\partial t = 0$을 만족하며, 비균질 파동 방정식을 따른다:

$$\nabla^2 V - \frac{1}{c^2}\frac{\partial^2 V}{\partial t^2} = -\frac{\rho}{\epsilon_0}$$

$$\nabla^2 \mathbf{A} - \frac{1}{c^2}\frac{\partial^2 \mathbf{A}}{\partial t^2} = -\mu_0 \mathbf{J}$$

### 1.4 리에나르-비헤르트 퍼텐셜 (개요)

궤적 $\mathbf{r}_0(t)$를 따라 속도 $\mathbf{v}(t)$로 운동하는 단일 점전하 $q$에 대해, 지연 퍼텐셜은 다음이 된다:

$$V(\mathbf{r}, t) = \frac{q}{4\pi\epsilon_0} \frac{1}{|\boldsymbol{\mathscr{r}}|(1 - \hat{\boldsymbol{\mathscr{r}}} \cdot \boldsymbol{\beta})}\bigg|_{t_r}$$

$$\mathbf{A}(\mathbf{r}, t) = \frac{\mu_0 q}{4\pi} \frac{\mathbf{v}}{|\boldsymbol{\mathscr{r}}|(1 - \hat{\boldsymbol{\mathscr{r}}} \cdot \boldsymbol{\beta})}\bigg|_{t_r}$$

여기서 $\boldsymbol{\mathscr{r}} = \mathbf{r} - \mathbf{r}_0(t_r)$는 지연 위치에서 장점(field point)까지의 벡터이고, $\boldsymbol{\beta} = \mathbf{v}/c$이다. 분모의 인수 $(1 - \hat{\boldsymbol{\mathscr{r}}} \cdot \boldsymbol{\beta})$는 상대론적 빔 집중(relativistic beaming) 효과를 담당한다.

---

## 2. 진동 전기 쌍극자 복사

### 2.1 설정

가장 단순한 복사 시스템은 진동 전기 쌍극자이다:

$$\mathbf{p}(t) = p_0 \cos(\omega t) \, \hat{z} = \text{Re}[p_0 e^{-i\omega t}] \, \hat{z}$$

이것은 $+q$와 $-q$ 두 전하가 공통 중심을 기준으로 간격 $d(t)$로 진동하는 것을 나타내며, $p_0 = qd_0$이다.

### 2.2 원거리장 근사

**원거리장(far field)** 혹은 복사 영역(radiation zone)에서, $r \gg \lambda \gg d$인 경우, 벡터 퍼텐셜은 다음으로 단순화된다:

$$\mathbf{A}(\mathbf{r}, t) \approx -\frac{\mu_0 p_0 \omega}{4\pi r} \sin(\omega t_r) \, \hat{z}$$

여기서 $t_r = t - r/c$이다.

### 2.3 복사장

원거리 영역에서의 전기장과 자기장은:

$$\mathbf{E} = -\frac{\mu_0 p_0 \omega^2}{4\pi c}\frac{\sin\theta}{r}\cos\left[\omega\left(t - \frac{r}{c}\right)\right] \hat{\theta}$$

$$\mathbf{B} = \frac{1}{c}\hat{r} \times \mathbf{E} = -\frac{\mu_0 p_0 \omega^2}{4\pi c^2}\frac{\sin\theta}{r}\cos\left[\omega\left(t - \frac{r}{c}\right)\right] \hat{\phi}$$

쌍극자 복사의 주요 특징:
- 장은 $1/r^2$ 대신 $1/r$로 감소한다 — 이것이 복사의 특징이다
- $\mathbf{E} \perp \mathbf{B} \perp \hat{r}$ — 장은 횡파(transverse)이다
- $\sin\theta$ 패턴은 **쌍극자 축 방향으로는 복사가 없음** ($\theta = 0$)을 의미하며, **적도면에서 최대 복사** ($\theta = \pi/2$)가 나타남을 의미한다

### 2.4 복사 전력

시간 평균 포인팅 벡터(Poynting vector)는:

$$\langle \mathbf{S} \rangle = \frac{\mu_0 p_0^2 \omega^4}{32\pi^2 c} \frac{\sin^2\theta}{r^2} \hat{r}$$

전체 입체각에 걸쳐 적분하면 총 복사 전력을 얻는다:

$$\boxed{P = \frac{\mu_0 p_0^2 \omega^4}{12\pi c} = \frac{p_0^2 \omega^4}{12\pi \epsilon_0 c^3}}$$

이 $\omega^4$ 의존성은 하늘이 왜 파란지를 설명한다(레일리 산란(Rayleigh scattering)) — 고주파의 파란빛이 저주파의 붉은빛보다 훨씬 강하게 산란되기 때문이다.

```python
import numpy as np
import matplotlib.pyplot as plt

def dipole_radiation_pattern(theta, power_pattern='sin2'):
    """
    Compute the radiation pattern of an oscillating electric dipole.

    The sin^2(theta) pattern is the defining signature of electric
    dipole radiation — it shows zero radiation along the dipole axis
    and maximum radiation perpendicular to it.
    """
    if power_pattern == 'sin2':
        return np.sin(theta)**2
    return np.ones_like(theta)

# Create polar radiation pattern plot
theta = np.linspace(0, 2 * np.pi, 500)
r_pattern = dipole_radiation_pattern(theta)

fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                          subplot_kw={'projection': 'polar'})

# Power pattern
axes[0].plot(theta, r_pattern, 'b-', linewidth=2)
axes[0].fill(theta, r_pattern, alpha=0.2, color='blue')
axes[0].set_title('Dipole Power Pattern $\\sim \\sin^2\\theta$', pad=20)
axes[0].set_theta_zero_location('N')
axes[0].set_theta_direction(-1)

# Field pattern (sqrt of power)
axes[1].plot(theta, np.abs(np.sin(theta)), 'r-', linewidth=2)
axes[1].fill(theta, np.abs(np.sin(theta)), alpha=0.2, color='red')
axes[1].set_title('Dipole Field Pattern $\\sim |\\sin\\theta|$', pad=20)
axes[1].set_theta_zero_location('N')
axes[1].set_theta_direction(-1)

plt.tight_layout()
plt.savefig("dipole_radiation_pattern.png", dpi=150)
plt.show()
```

---

## 3. 라머 공식

### 3.1 가속하는 전하의 복사 전력

가속도 $\mathbf{a}$를 가진 점전하 $q$의 복사 전력:

$$\boxed{P = \frac{q^2 a^2}{6\pi\epsilon_0 c^3} = \frac{\mu_0 q^2 a^2}{6\pi c}}$$

이것이 **라머 공식(Larmor formula)**으로, 비상대론적 운동($v \ll c$)에서 성립한다.

복사 전력의 각도 분포는:

$$\frac{dP}{d\Omega} = \frac{q^2 a^2}{16\pi^2 \epsilon_0 c^3} \sin^2\Theta$$

여기서 $\Theta$는 가속도 벡터와 관측 방향 사이의 각도이다.

### 3.2 쌍극자 복사와의 연관성

쌍극자 $\mathbf{p}(t) = q \mathbf{d}(t)$에 대해, 전하의 가속도는 $\ddot{\mathbf{p}}$와 연관된다:

$$P = \frac{|\ddot{\mathbf{p}}|^2}{6\pi\epsilon_0 c^3}$$

조화 진동 $\ddot{\mathbf{p}} = -\omega^2 p_0 \hat{z}$에 대해, 이것은 쌍극자 공식을 재현한다.

### 3.3 상대론적 일반화

상대론적 속도로 운동하는 전하에 대해, 라머 공식은 다음으로 일반화된다:

$$P = \frac{q^2 \gamma^6}{6\pi\epsilon_0 c}\left(a^2 - \frac{|\mathbf{v} \times \mathbf{a}|^2}{c^2}\right)$$

$\gamma^6$ 인수는 싱크로트론 복사(synchrotron radiation) — 상대론적 속도로 원운동하는 전하의 복사 — 를 매우 강렬하게 만든다.

```python
def larmor_power(q, a, eps0=8.854e-12, c=3e8):
    """
    Compute power radiated by an accelerating charge (Larmor formula).

    Why Larmor: this formula is the foundation of all radiation physics.
    Every antenna, every synchrotron, every X-ray tube operates on the
    principle that accelerating charges radiate.
    """
    return q**2 * a**2 / (6 * np.pi * eps0 * c**3)

# Example: electron in a TV tube (CRT)
q_e = 1.602e-19   # electron charge (C)
m_e = 9.109e-31   # electron mass (kg)

# Electron accelerated through 25 kV over 10 cm
V_accel = 25e3  # volts
d_accel = 0.1   # meters
a_crt = q_e * V_accel / (m_e * d_accel)

P_crt = larmor_power(q_e, a_crt)
print(f"CRT electron acceleration: {a_crt:.2e} m/s²")
print(f"Radiated power: {P_crt:.2e} W")
print(f"This is negligible compared to kinetic energy gain!")

print()

# Synchrotron electron (v ~ c, circular orbit)
gamma = 1000  # Lorentz factor for ~500 MeV electron
v = 3e8 * np.sqrt(1 - 1/gamma**2)
R = 10  # orbit radius (m)
a_sync = v**2 / R  # centripetal acceleration

# Relativistic Larmor: multiply by gamma^4 for circular motion
P_sync = larmor_power(q_e, a_sync) * gamma**4
print(f"Synchrotron electron (γ={gamma}):")
print(f"  Centripetal acceleration: {a_sync:.2e} m/s²")
print(f"  Radiated power: {P_sync:.2e} W = {P_sync*1e6:.1f} μW per electron")
```

---

## 4. 복사 저항

### 4.1 정의

안테나에 전류가 흐를 때, 복사 전력을 다음과 같이 표현할 수 있다:

$$P_{\text{rad}} = \frac{1}{2} I_0^2 R_{\text{rad}}$$

여기서 $I_0$는 전류 최댓값이고, $R_{\text{rad}}$은 **복사 저항(radiation resistance)**이다 — 복사로 방출되는 전력을 나타내는 가상의 저항이다. "가상"이라고 하는 이유는 실제 저항이 에너지를 소산시키는 것이 아니라, 에너지가 전자기파로 운반되어 나가기 때문이다.

### 4.2 짧은 쌍극자 안테나의 복사 저항

길이 $\ell \ll \lambda$이고 균일한 전류 $I_0$를 흘리는 짧은 쌍극자 안테나의 경우:

$$R_{\text{rad}} = \frac{2\pi}{3}\eta_0 \left(\frac{\ell}{\lambda}\right)^2 \approx 790 \left(\frac{\ell}{\lambda}\right)^2 \, \Omega$$

여기서 $\eta_0 = \sqrt{\mu_0/\epsilon_0} \approx 377 \, \Omega$는 자유 공간의 임피던스(impedance of free space)이다.

$\ell = 0.01\lambda$인 경우: $R_{\text{rad}} \approx 0.079 \, \Omega$ — 매우 낮아서, 짧은 안테나는 효율이 나쁘다.

### 4.3 안테나 효율

안테나의 효율은:

$$\eta = \frac{R_{\text{rad}}}{R_{\text{rad}} + R_{\text{loss}}}$$

여기서 $R_{\text{loss}}$는 안테나 도체의 옴 저항이다. 짧은 안테나에서는 $R_{\text{rad}}$가 작아 효율이 낮다. 이것이 AM 라디오 타워(수백 미터 파장에서 동작)가 매우 높아야 하는 이유이다.

---

## 5. 반파장 쌍극자

### 5.1 전류 분포

전체 길이 $L = \lambda/2$인 중심 급전 쌍극자 안테나는 다음의 정현파 전류 분포를 갖는다:

$$I(z) = I_0 \cos\left(\frac{\pi z}{L}\right) = I_0 \cos(k z)$$

여기서 $z$는 중심에서 측정하며 $|z| \leq L/2$이다.

### 5.2 복사 패턴

원거리장 복사 패턴은 모든 전류 요소의 기여를 적분하여 구한다. 결과는:

$$E_\theta \propto \frac{\cos\left(\frac{\pi}{2}\cos\theta\right)}{\sin\theta}$$

이 패턴은 짧은 쌍극자의 $\sin\theta$ 패턴보다 약간 더 지향적이다.

### 5.3 복사 저항과 지향성

반파장 쌍극자의 경우:

$$R_{\text{rad}} = 73.1 \, \Omega$$

이는 짧은 쌍극자의 복사 저항보다 훨씬 크기 때문에, 반파장 쌍극자를 실용적으로 만든다. 입력 임피던스는 $Z_{\text{in}} \approx 73.1 + j42.5 \, \Omega$ (약간 유도성)이다.

**지향성(directivity)**은:

$$D = \frac{U_{\max}}{\bar{U}} = 1.64 = 2.15 \, \text{dBi}$$

여기서 dBi는 "등방성 복사기(isotropic radiator)에 대한 dB"를 의미한다.

```python
def half_wave_dipole_pattern(theta):
    """
    Radiation pattern of a half-wave dipole antenna.

    Why the half-wave dipole: it's the most fundamental practical antenna.
    Its ~73 ohm radiation resistance matches common transmission lines,
    and its pattern is the building block for more complex antenna arrays.
    """
    # Avoid division by zero at theta = 0 and pi
    sin_theta = np.sin(theta)
    sin_theta = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)
    return (np.cos(np.pi / 2 * np.cos(theta)) / sin_theta)**2

def compare_antenna_patterns():
    """Compare short dipole vs half-wave dipole radiation patterns."""
    theta = np.linspace(0, 2 * np.pi, 1000)

    # Patterns (normalized)
    short_dipole = np.sin(theta)**2
    half_wave = half_wave_dipole_pattern(theta)
    half_wave_norm = half_wave / half_wave.max()

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    ax.plot(theta, short_dipole, 'b-', linewidth=2, label='Short dipole')
    ax.plot(theta, half_wave_norm, 'r-', linewidth=2, label='Half-wave dipole')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.legend(loc='lower right')
    ax.set_title('Antenna Radiation Patterns', pad=20)

    plt.tight_layout()
    plt.savefig("antenna_patterns.png", dpi=150)
    plt.show()

compare_antenna_patterns()
```

---

## 6. 안테나 배열

### 6.1 배열 인수

$N$개의 동일한 안테나가 $z$축을 따라 간격 $d$로 배열되고, 각각이 동일한 진폭에 점진적 위상 이동 $\alpha$로 급전될 때, 총 복사 패턴은:

$$\text{배열 패턴} = \text{소자 패턴} \times \text{배열 인수}$$

균일 선형 배열의 **배열 인수(array factor)**는:

$$\text{AF}(\theta) = \sum_{n=0}^{N-1} e^{in(kd\cos\theta + \alpha)} = \frac{\sin\left(\frac{N\psi}{2}\right)}{\sin\left(\frac{\psi}{2}\right)}$$

여기서 $\psi = kd\cos\theta + \alpha$이다.

### 6.2 빔 조향

주 빔 방향 $\theta_0$는 $\psi = 0$일 때 나타난다:

$$kd\cos\theta_0 + \alpha = 0 \implies \theta_0 = \arccos\left(-\frac{\alpha}{kd}\right)$$

위상 이동 $\alpha$를 변화시킴으로써 안테나를 물리적으로 움직이지 않고도 **전자적으로 빔을 조향**할 수 있다 — 이것이 위상 배열 레이더(phased array radar)의 원리이다.

### 6.3 배열 특성

- **빔폭(beamwidth)**: $\sim 1/N$으로 감소한다 (소자가 많을수록 빔이 좁아진다)
- **부엽(sidelobe)**: 균일 배열에서 첫 번째 부엽은 주 빔보다 약 13.2 dB 낮다
- **격자 부엽(grating lobe)**: $d > \lambda/2$이면 추가 주 빔이 나타나며, 일반적으로 바람직하지 않다

```python
def array_factor(theta, N, d_lambda, alpha_deg=0):
    """
    Compute array factor for a uniform linear antenna array.

    Parameters:
        theta     : angle from array axis (radians)
        N         : number of elements
        d_lambda  : element spacing in wavelengths (d/lambda)
        alpha_deg : progressive phase shift (degrees)

    Why phased arrays: they enable electronic beam steering in
    microseconds (vs. seconds for mechanical rotation), making them
    essential for modern radar, 5G, and satellite communications.
    """
    k = 2 * np.pi  # k*lambda = 2*pi, so k*d = 2*pi*d/lambda
    alpha = np.radians(alpha_deg)
    psi = k * d_lambda * np.cos(theta) + alpha

    # Array factor (handle the sin(0)/sin(0) case)
    numerator = np.sin(N * psi / 2)
    denominator = np.sin(psi / 2)

    # Avoid division by zero
    af = np.where(np.abs(denominator) < 1e-10, N, numerator / denominator)

    return np.abs(af)**2 / N**2  # normalized

def plot_array_patterns():
    """Demonstrate beam steering and array factor properties."""
    theta = np.linspace(0, np.pi, 1000)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Effect of number of elements (broadside, alpha=0)
    ax = axes[0, 0]
    for N in [2, 4, 8, 16]:
        af = array_factor(theta, N, d_lambda=0.5, alpha_deg=0)
        ax.plot(np.degrees(theta), 10 * np.log10(af + 1e-10),
                linewidth=1.5, label=f'N = {N}')
    ax.set_xlabel('Angle from array axis (degrees)')
    ax.set_ylabel('Array Factor (dB)')
    ax.set_title('Effect of Number of Elements (d = λ/2, broadside)')
    ax.set_ylim(-30, 3)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Beam steering
    ax = axes[0, 1]
    N = 8
    for alpha in [0, -45, -90, -135]:
        af = array_factor(theta, N, d_lambda=0.5, alpha_deg=alpha)
        ax.plot(np.degrees(theta), 10 * np.log10(af + 1e-10),
                linewidth=1.5, label=f'$\\alpha$ = {alpha}°')
    ax.set_xlabel('Angle from array axis (degrees)')
    ax.set_ylabel('Array Factor (dB)')
    ax.set_title(f'Beam Steering (N={N}, d = λ/2)')
    ax.set_ylim(-30, 3)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Polar plot: broadside array
    ax = axes[1, 0]
    ax_polar = fig.add_subplot(2, 2, 3, projection='polar')
    axes[1, 0].remove()

    for N in [4, 8, 16]:
        theta_full = np.linspace(0, 2 * np.pi, 2000)
        af = array_factor(theta_full, N, d_lambda=0.5)
        af_db = 10 * np.log10(af + 1e-10)
        af_db_plot = np.clip(af_db + 30, 0, 30)  # shift for plotting
        ax_polar.plot(theta_full, af_db_plot, linewidth=1.5, label=f'N={N}')

    ax_polar.set_title('Polar Pattern (broadside)', pad=20)
    ax_polar.set_theta_zero_location('N')
    ax_polar.set_theta_direction(-1)
    ax_polar.legend(loc='lower right')

    # Grating lobes: effect of spacing
    ax = axes[1, 1]
    N = 8
    for d_lam in [0.25, 0.5, 0.75, 1.0]:
        af = array_factor(theta, N, d_lambda=d_lam)
        ax.plot(np.degrees(theta), 10 * np.log10(af + 1e-10),
                linewidth=1.5, label=f'd = {d_lam}λ')
    ax.set_xlabel('Angle from array axis (degrees)')
    ax.set_ylabel('Array Factor (dB)')
    ax.set_title(f'Grating Lobes: Effect of Spacing (N={N})')
    ax.set_ylim(-30, 3)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("antenna_arrays.png", dpi=150)
    plt.show()

plot_array_patterns()
```

---

## 7. 지향성과 이득

### 7.1 정의

**지향성(directivity)**은 최대 복사 세기 대 평균의 비이다:

$$D = \frac{U_{\max}}{\bar{U}} = \frac{4\pi U_{\max}}{P_{\text{rad}}}$$

여기서 $U(\theta, \phi) = r^2 |\langle \mathbf{S} \rangle|$는 복사 세기(radiation intensity, W/sr)이다.

**이득(gain)**은 안테나 효율을 포함한다:

$$G = \eta \, D$$

**유효 면적(effective area)**은 수신 안테나가 포착하는 전력을 입사 전력 밀도와 연관 짓는다:

$$A_e = \frac{\lambda^2}{4\pi} G$$

### 7.2 표준 안테나 파라미터

| 안테나 | 지향성 | 이득 (일반적) | 복사 저항 |
|--------|--------|---------------|-----------|
| 등방성 | 1 (0 dBi) | — | — |
| 짧은 쌍극자 ($\ell \ll \lambda$) | 1.5 (1.76 dBi) | $\sim$0.5–1.0 dBi | $790(\ell/\lambda)^2$ $\Omega$ |
| 반파장 쌍극자 | 1.64 (2.15 dBi) | $\sim$2.0 dBi | 73.1 $\Omega$ |
| 1/4파장 모노폴 | 3.28 (5.15 dBi) | $\sim$4.5 dBi | 36.5 $\Omega$ |

### 7.3 프리스 전송 방정식

거리 $R$로 분리된 두 안테나에 대해, 수신 전력은:

$$\frac{P_r}{P_t} = G_t G_r \left(\frac{\lambda}{4\pi R}\right)^2$$

인수 $(\lambda / 4\pi R)^2$은 **자유 공간 경로 손실(free-space path loss)**이다.

```python
def friis_link_budget(Pt_dBm, Gt_dBi, Gr_dBi, freq_GHz, distance_km):
    """
    Compute received power using the Friis equation.

    Why Friis: this is the fundamental equation for all wireless
    link design — from satellite communications to Bluetooth.
    """
    c = 3e8
    lam = c / (freq_GHz * 1e9)
    R = distance_km * 1e3

    # Free space path loss in dB
    FSPL_dB = 20 * np.log10(4 * np.pi * R / lam)

    Pr_dBm = Pt_dBm + Gt_dBi + Gr_dBi - FSPL_dB

    print(f"Friis Link Budget")
    print(f"=================")
    print(f"Frequency:    {freq_GHz} GHz (λ = {lam*1e3:.1f} mm)")
    print(f"Distance:     {distance_km} km")
    print(f"Tx power:     {Pt_dBm} dBm")
    print(f"Tx gain:      {Gt_dBi} dBi")
    print(f"Rx gain:      {Gr_dBi} dBi")
    print(f"Path loss:    {FSPL_dB:.1f} dB")
    print(f"Rx power:     {Pr_dBm:.1f} dBm = {10**(Pr_dBm/10) * 1e-3:.2e} W")

    return Pr_dBm

# Example: Wi-Fi link
friis_link_budget(Pt_dBm=20, Gt_dBi=3, Gr_dBi=3,
                  freq_GHz=5.0, distance_km=0.05)

print()

# Example: Satellite link
friis_link_budget(Pt_dBm=43, Gt_dBi=40, Gr_dBi=35,
                  freq_GHz=12.0, distance_km=36000)
```

---

## 8. 시각화: 시간에 따른 쌍극자 복사

```python
def animate_dipole_radiation(save=False):
    """
    Visualize the time evolution of electric field lines from
    an oscillating electric dipole.

    Why visualize: static patterns do not convey the dynamic nature
    of radiation — seeing the fields propagate outward gives
    physical intuition about how energy leaves the source.
    """
    # Create field on a grid
    N = 200
    x = np.linspace(-5, 5, N)
    z = np.linspace(-5, 5, N)
    X, Z = np.meshgrid(x, z)
    R = np.sqrt(X**2 + Z**2)
    R = np.where(R < 0.3, 0.3, R)  # avoid singularity at origin

    theta = np.arctan2(X, Z)  # angle from z-axis (dipole axis)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    phases = [0, np.pi / 3, 2 * np.pi / 3]

    for ax, phase in zip(axes, phases):
        # Far-field radiation pattern: E ~ sin(theta)/r * cos(kr - wt)
        k = 2 * np.pi  # wavelength = 1 unit
        E_theta = np.sin(theta) / R * np.cos(k * R - phase)

        # Convert to Cartesian components for vector plot
        Ex = E_theta * np.cos(theta)
        Ez = -E_theta * np.sin(theta)
        E_mag = np.sqrt(Ex**2 + Ez**2)

        # Intensity plot
        im = ax.pcolormesh(X, Z, E_theta, cmap='RdBu_r', shading='auto',
                           vmin=-2, vmax=2)

        # Mark the dipole
        ax.plot(0, 0, 'ko', markersize=10)
        ax.arrow(0, -0.15, 0, 0.3, head_width=0.1, head_length=0.05,
                 fc='yellow', ec='yellow')

        ax.set_xlabel('x / λ')
        ax.set_ylabel('z / λ')
        ax.set_title(f'ωt = {phase/np.pi:.1f}π')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='$E_\\theta$', shrink=0.7)

    plt.suptitle('Oscillating Dipole Radiation (Electric Field)', fontsize=14)
    plt.tight_layout()
    plt.savefig("dipole_radiation_time.png", dpi=150)
    plt.show()

animate_dipole_radiation()
```

---

## 요약

| 개념 | 핵심 공식 | 물리적 의미 |
|------|-----------|-------------|
| 지연 퍼텐셜 | $V = \frac{1}{4\pi\epsilon_0}\int \frac{\rho(\mathbf{r}', t_r)}{|\mathbf{r}-\mathbf{r}'|} d^3r'$ | 인과율: 장은 소스의 과거 상태를 담는다 |
| 쌍극자 복사 | $P = \frac{p_0^2\omega^4}{12\pi\epsilon_0 c^3}$ | 전력은 $\omega^4$에 비례한다 |
| 라머 공식 | $P = \frac{q^2 a^2}{6\pi\epsilon_0 c^3}$ | 가속하는 전하는 복사한다 |
| 복사 저항 | $P_{\text{rad}} = \frac{1}{2}I_0^2 R_{\text{rad}}$ | 복사 전력에 대한 등가 저항 |
| 반파장 쌍극자 | $R_{\text{rad}} = 73.1\,\Omega$ | 전송선과의 실용적 임피던스 매칭 |
| 배열 인수 | $\text{AF} = \frac{\sin(N\psi/2)}{\sin(\psi/2)}$ | 패턴 곱셈 |
| 프리스 방정식 | $P_r/P_t = G_t G_r (\lambda/4\pi R)^2$ | 무선 링크 전력 예산 |

---

## 연습 문제

### 연습 1: 전류가 스위치-온될 때의 지연 퍼텐셜
긴 도선이 $t < 0$에서는 전류가 없다가 $t > 0$에서 전류 $I_0$를 흘린다. 지연 퍼텐셜을 이용하여 도선으로부터 거리 $s$에서의 자기장을 시간의 함수로 계산하라. 장이 도선의 가장 가까운 점에서부터 "켜지기" 시작하여, 더 많은 도선 부분이 기여함에 따라 점차 확장됨을 보여라.

### 연습 2: 자기 쌍극자 복사
진동 자기 쌍극자 $\mathbf{m}(t) = m_0 \cos(\omega t) \hat{z}$는 전기 쌍극자와 유사한 패턴으로 복사하지만 $\mathbf{E}$와 $\mathbf{B}$가 뒤바뀐다. (a) 원거리장 전기장을 적어라. (b) 총 복사 전력을 계산하라. (c) 자기 쌍극자와 전기 쌍극자 복사 전력의 비가 $(m_0 \omega / p_0 c^2)^2$임을 보여라.

### 연습 3: 위상 배열 설계
반파장 간격으로 10 GHz에서 동작하는 8소자 위상 배열을 설계하라. (a) 브로드사이드(broadside) 기준 0, 30, 45, 60도 방향으로 빔을 조향할 때의 배열 인수를 그려라. (b) 각 경우의 3 dB 빔폭을 결정하라. (c) 격자 부엽이 나타나기 전 최대 조향 각도는 얼마인가?

### 연습 4: 위성 링크 예산
정지 위성(고도 36,000 km)이 12 GHz에서 1미터 접시 안테나(이득 $\approx$ 40 dBi)를 통해 20 W로 송신한다. (a) 자유 공간 경로 손실을 계산하라. (b) 지상국에 3미터 접시(이득 $\approx$ 50 dBi)가 있다면 수신 전력은 얼마인가? (c) 이것은 290 K에서 30 MHz 대역폭 수신기의 열 잡음 한계를 초과하는가?

---

[← 이전: 12. 도파관과 공동](12_Waveguides_and_Cavities.md) | [다음: 14. 상대론적 전자기학 →](14_Relativistic_Electrodynamics.md)
