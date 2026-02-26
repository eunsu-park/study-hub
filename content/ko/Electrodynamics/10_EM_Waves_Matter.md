# 10. 물질에서의 전자기파

[← 이전: 09. 진공에서의 전자기파](09_EM_Waves_Vacuum.md) | [다음: 11. 반사와 굴절 →](11_Reflection_and_Refraction.md)

## 학습 목표

1. 맥스웰 방정식으로부터 선형 유전체 및 도전성 매질에서의 파동 방정식을 유도한다
2. 복소 유전율(Complex Permittivity)과 흡수·분산의 관계를 이해한다
3. 드루데 모형(Drude Model)을 적용하여 금속의 전자기 응답을 기술한다
4. 표피 깊이(Skin Depth)를 계산하고 그 실용적 의미를 파악한다
5. 복소 굴절률(Complex Refractive Index)과 흡수 계수(Absorption Coefficient)를 다룬다
6. 크라머스-크로니히 관계식(Kramers-Kronig Relations)을 유도하고 응답 함수의 실수부와 허수부를 연결하는 의미를 해석한다
7. Python을 이용하여 분산 매질에서의 파동 전파를 시뮬레이션한다

전자기파가 물질 속으로 진입하면, 더 이상 빛의 속도로 아무 방해 없이 진행하지 못한다. 파동의 전기장이 유전체의 속박 전하와 도체의 자유 전하를 움직이게 하여 분극(Polarization) 전류와 전도(Conduction) 전류를 만들고, 이것이 다시 맥스웰 방정식에 피드백된다. 그 결과 다양한 현상이 나타난다: 파동이 느려지거나(굴절), 에너지를 잃거나(흡수), 주파수에 따라 다른 속도로 전파된다(분산). 이러한 효과를 이해하는 것은 광섬유, 레이더 흡수체, 그리고 사실상 모든 광자소자 설계의 핵심이다.

> **비유**: 파동(예: "파도타기 응원")이 전파되는 사람들의 군중(매질)을 생각해보자. 진공에서는 파동이 아무 저항 없이 최대 속도로 움직인다. 밀집한 군중(높은 유전율)에서는 사람들이 천천히 반응하므로 파동이 느려진다. 사람들이 서툴러 서로 부딪힌다면(전도율), 진동할 때마다 마찰로 에너지가 손실된다. 이러한 군중 역학의 주파수 의존성이 바로 분산이다.

---

## 1. 선형 매질에서의 맥스웰 방정식

### 1.1 구성 관계

선형·등방·균질(LIH, Linear Isotropic Homogeneous) 매질에서 물질의 응답은 다음으로 표현된다:

$$\mathbf{D} = \epsilon \mathbf{E}, \quad \mathbf{B} = \mu \mathbf{H}, \quad \mathbf{J}_f = \sigma \mathbf{E}$$

여기서 $\epsilon$ 은 유전율(Permittivity), $\mu$ 는 투자율(Permeability), $\sigma$ 는 전도율(Conductivity)이다. 다음과 같이 쓴다:

$$\epsilon = \epsilon_0 \epsilon_r, \quad \mu = \mu_0 \mu_r$$

$\epsilon_r$ 과 $\mu_r$ 은 각각 비유전율(Relative Permittivity)과 비투자율(Relative Permeability)이다.

### 1.2 유전체에서의 파동 방정식

물질 내의 소스 없는 맥스웰 방정식으로부터 출발한다:

$$\nabla \times \mathbf{E} = -\mu \frac{\partial \mathbf{H}}{\partial t}, \quad \nabla \times \mathbf{H} = \epsilon \frac{\partial \mathbf{E}}{\partial t} + \sigma \mathbf{E}$$

첫 번째 식의 회전(Curl)을 취하고 두 번째 식을 대입하면:

$$\nabla^2 \mathbf{E} = \mu \epsilon \frac{\partial^2 \mathbf{E}}{\partial t^2} + \mu \sigma \frac{\partial \mathbf{E}}{\partial t}$$

이것이 **감쇠 파동 방정식(Damped Wave Equation)** 이다. 우변의 첫째 항은 파동 전파(속도 $v = 1/\sqrt{\mu\epsilon}$)를, 둘째 항은 옴 손실(Ohmic Loss)을 나타낸다.

손실 없는 유전체($\sigma = 0$)의 경우:

$$\nabla^2 \mathbf{E} = \mu \epsilon \frac{\partial^2 \mathbf{E}}{\partial t^2}$$

매질에서의 파속(Wave Speed)은:

$$v = \frac{1}{\sqrt{\mu \epsilon}} = \frac{c}{\sqrt{\mu_r \epsilon_r}} = \frac{c}{n}$$

여기서 $n = \sqrt{\mu_r \epsilon_r}$ 은 **굴절률(Refractive Index)** 이다.

### 1.3 도전성 매질에서의 평면파 가정

$\tilde{k}$ 를 복소 파수(Complex Wave Number)로 하여 평면파 해 $\mathbf{E} = \mathbf{E}_0 \, e^{i(\tilde{k}z - \omega t)}$ 를 가정한다. 감쇠 파동 방정식에 대입하면:

$$-\tilde{k}^2 = -\mu \epsilon \omega^2 + i \mu \sigma \omega$$

$$\tilde{k}^2 = \mu \epsilon \omega^2 - i \mu \sigma \omega = \mu \omega^2 \left(\epsilon - i\frac{\sigma}{\omega}\right)$$

이로부터 **복소 유전율(Complex Permittivity)** 을 정의하는 것이 자연스럽다.

---

## 2. 복소 유전율

### 2.1 정의

전도율을 단일 복소량으로 흡수시킨다:

$$\tilde{\epsilon}(\omega) = \epsilon(\omega) + i\frac{\sigma(\omega)}{\omega} = \epsilon'(\omega) + i\epsilon''(\omega)$$

여기서 $\epsilon' = \operatorname{Re}(\tilde{\epsilon})$, $\epsilon'' = \operatorname{Im}(\tilde{\epsilon})$ 다. 이제 파동 방정식이 단순한 형태가 된다:

$$\tilde{k}^2 = \mu \tilde{\epsilon} \, \omega^2$$

실수부 $\epsilon'$ 은 위상 속도를, 허수부 $\epsilon''$ 은 흡수를 결정한다.

### 2.2 부호 규약

교재에 따라 시간 의존성의 부호 규약($e^{-i\omega t}$ 대 $e^{+i\omega t}$)이 다르다. 우리의 규약 $e^{i(\tilde{k}z - \omega t)}$ 에서:

- $\epsilon'' > 0$ 은 **손실** (흡수)에 해당
- $\tilde{k}$ 의 허수부가 양수이면 $z$ 방향으로 지수적 감쇠

이것은 물리학자의 규약(그리피스, 잭슨)이다. 공학에서는 보통 $e^{j(\omega t - kz)}$ 를 사용하므로 $\epsilon''$ 의 부호가 반전된다.

### 2.3 로렌츠 진동자 모형

유전 응답의 가장 단순한 미시적 모형은 속박 전자를 파동의 전기장에 의해 구동되는 감쇠 조화 진동자(Damped Harmonic Oscillator)로 취급한다:

$$m\ddot{x} + m\gamma\dot{x} + m\omega_0^2 x = -eE_0 e^{-i\omega t}$$

여기서 $\omega_0$ 는 고유 진동수, $\gamma$ 는 감쇠율이다. 정상 상태 해로부터 쌍극자 모멘트 $p = -ex$ 를 구하고, 단위 부피당 $N$ 개의 진동자를 합산하면:

$$\epsilon_r(\omega) = 1 + \frac{Ne^2}{m\epsilon_0} \frac{1}{\omega_0^2 - \omega^2 - i\gamma\omega}$$

진동수 $\omega_j$, 진동자 세기 $f_j$ 를 가진 다중 공명의 경우:

$$\epsilon_r(\omega) = 1 + \frac{Ne^2}{m\epsilon_0} \sum_j \frac{f_j}{\omega_j^2 - \omega^2 - i\gamma_j\omega}$$

```python
import numpy as np
import matplotlib.pyplot as plt

def lorentz_permittivity(omega, omega_0, gamma, omega_p):
    """
    Compute complex relative permittivity using the Lorentz oscillator model.

    Parameters:
        omega   : angular frequency array (rad/s)
        omega_0 : resonance frequency (rad/s)
        gamma   : damping rate (rad/s)
        omega_p : plasma frequency = sqrt(Ne^2 / m eps_0) (rad/s)

    Returns:
        Complex epsilon_r(omega)

    Why this model matters: it captures the essential physics of how
    bound electrons respond to an oscillating E-field, producing both
    dispersion (varying n) and absorption near resonance.
    """
    return 1.0 + omega_p**2 / (omega_0**2 - omega**2 - 1j * gamma * omega)

# Parameters for a typical glass-like material
omega_0 = 6e15       # UV resonance ~6 PHz
gamma = 1e14         # damping ~0.1 PHz
omega_p = 4e15       # plasma frequency

omega = np.linspace(1e14, 1.2e16, 5000)
eps = lorentz_permittivity(omega, omega_0, gamma, omega_p)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Real part: determines refractive index
axes[0].plot(omega / 1e15, eps.real, 'b-', linewidth=1.5)
axes[0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
axes[0].axvline(x=omega_0 / 1e15, color='r', linestyle=':', label=f'$\\omega_0$')
axes[0].set_ylabel("$\\epsilon'$ (real part)")
axes[0].set_title("Lorentz Oscillator Model: Complex Permittivity")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Imaginary part: determines absorption
axes[1].plot(omega / 1e15, eps.imag, 'r-', linewidth=1.5)
axes[1].axvline(x=omega_0 / 1e15, color='r', linestyle=':', label=f'$\\omega_0$')
axes[1].set_xlabel("Frequency (PHz)")
axes[1].set_ylabel("$\\epsilon''$ (imaginary part)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lorentz_permittivity.png", dpi=150)
plt.show()
```

공명 근방($\omega \approx \omega_0$)에서 허수부는 최댓값(최대 흡수)을 갖고, 실수부는 1보다 크다가 작아지는 방향으로 급격히 변한다 — 이것이 **이상 분산(Anomalous Dispersion)** 으로, 진동수가 증가할수록 굴절률이 감소한다.

---

## 3. 분산

### 3.1 정상 분산과 이상 분산

공명에서 멀리 떨어진 곳에서는 $\epsilon'$ 이 진동수와 함께 증가한다 — 이것이 **정상 분산(Normal Dispersion)** 이다. 프리즘이 색깔을 분리하는 것은 파란빛(높은 $\omega$)이 더 큰 굴절률을 가져 더 많이 굴절되기 때문이다.

공명 근방에서는 $\epsilon'$ 이 진동수와 함께 **감소**한다 — **이상 분산(Anomalous Dispersion)**. 이 좁은 대역에서 군속도(Group Velocity)가 $c$ 를 초과하거나 음수가 될 수도 있지만, 신호 속도는 $c$ 이하로 유지된다(빛보다 빠른 정보 전달 불가).

### 3.2 위상 속도와 군속도

복소 파수는 $\tilde{k} = k + i\kappa$ 이므로:

$$\mathbf{E} = \mathbf{E}_0 \, e^{-\kappa z} e^{i(kz - \omega t)}$$

- **위상 속도(Phase Velocity)**: $v_p = \omega / k$ (등위상면이 이동하는 속도)
- **군속도(Group Velocity)**: $v_g = d\omega / dk$ (파속의 포락선이 이동하는 속도)

분산 매질에서는 $v_p \neq v_g$ 다. 공명에서 멀리 떨어진 로렌츠 모형의 경우:

$$v_g = \frac{c}{n + \omega \, dn/d\omega}$$

### 3.3 군속도 분산

2차 미분 $d^2k/d\omega^2$ 은 펄스가 전파되면서 얼마나 넓어지는지를 결정한다. 광섬유 설계에서 매우 중요하다:

$$\text{GVD} = \frac{d^2 k}{d\omega^2} = \frac{1}{c}\left(2\frac{dn}{d\omega} + \omega\frac{d^2n}{d\omega^2}\right)$$

- GVD > 0: 정상 분산 (파장이 긴 빛이 더 빨리 전파)
- GVD < 0: 이상 분산 (파장이 짧은 빛이 더 빨리 전파)

```python
def compute_group_velocity(omega, eps_r, mu_r=1.0):
    """
    Compute phase velocity, group velocity, and GVD from complex permittivity.

    Why we compute these numerically: analytical expressions exist for
    simple models, but numerical differentiation generalizes to arbitrary
    epsilon(omega) from experiments or multi-resonance models.
    """
    c = 3e8  # speed of light (m/s)

    # Complex refractive index
    n_complex = np.sqrt(eps_r * mu_r)
    n = n_complex.real  # real refractive index
    kappa = n_complex.imag  # extinction coefficient

    # Phase velocity
    v_phase = c / n

    # Group velocity via numerical derivative dn/domega
    dn_domega = np.gradient(n, omega)
    v_group = c / (n + omega * dn_domega)

    # GVD via second derivative of k = n*omega/c
    k_real = n * omega / c
    d2k_domega2 = np.gradient(np.gradient(k_real, omega), omega)

    return v_phase, v_group, d2k_domega2

# Using the Lorentz model from above (far from resonance region)
mask = (omega < 4e15) | (omega > 8e15)  # avoid resonance region
omega_disp = omega[mask]
eps_disp = eps[mask]

v_ph, v_gr, gvd = compute_group_velocity(omega_disp, eps_disp)

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

axes[0].plot(omega_disp / 1e15, v_ph / 3e8, 'b-', label='$v_p / c$')
axes[0].axhline(y=1, color='gray', linestyle='--')
axes[0].set_ylabel('$v_p / c$')
axes[0].set_title('Dispersion Properties')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(omega_disp / 1e15, v_gr / 3e8, 'r-', label='$v_g / c$')
axes[1].axhline(y=1, color='gray', linestyle='--')
axes[1].set_ylabel('$v_g / c$')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(omega_disp / 1e15, gvd * 1e30, 'g-', label='GVD')
axes[2].axhline(y=0, color='gray', linestyle='--')
axes[2].set_xlabel('Frequency (PHz)')
axes[2].set_ylabel('GVD (fs$^2$/mm)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("dispersion_properties.png", dpi=150)
plt.show()
```

---

## 4. 금속을 위한 드루데 모형

### 4.1 자유 전자 응답

금속에서는 전도 전자가 사실상 자유롭다($\omega_0 = 0$). 운동 방정식이 다음으로 단순화된다:

$$m\ddot{x} + m\gamma\dot{x} = -eE_0 e^{-i\omega t}$$

이로부터 **드루데 유전율(Drude Permittivity)** 을 얻는다:

$$\epsilon_r(\omega) = 1 - \frac{\omega_p^2}{\omega^2 + i\gamma\omega}$$

여기서 $\omega_p = \sqrt{Ne^2 / m\epsilon_0}$ 는 **플라즈마 주파수(Plasma Frequency)** — 전자 기체의 자연 진동 주파수다.

### 4.2 주요 특성

실수부와 허수부를 분리하면:

$$\epsilon'(\omega) = 1 - \frac{\omega_p^2}{\omega^2 + \gamma^2}, \quad \epsilon''(\omega) = \frac{\omega_p^2 \gamma}{\omega(\omega^2 + \gamma^2)}$$

**저주파수** ($\omega \ll \gamma$): 허수부가 지배적이며, 물질은 도체처럼 거동한다. 드루데 전도율은:

$$\sigma(\omega) = \frac{Ne^2}{m(\gamma - i\omega)} \quad \Rightarrow \quad \sigma_0 = \frac{Ne^2}{m\gamma} \quad (\text{직류 전도율})$$

**고주파수** ($\omega \gg \gamma$): 감쇠를 무시할 수 있으며:

$$\epsilon_r(\omega) \approx 1 - \frac{\omega_p^2}{\omega^2}$$

- $\omega < \omega_p$ 인 경우: $\epsilon_r < 0$, 파동이 소멸(금속은 불투명)
- $\omega > \omega_p$ 인 경우: $\epsilon_r > 0$, 파동 전파(금속이 투명해짐)

이것이 금속이 가시광선에서 반짝이지만 X선에 대해서는 투명한 이유이며, 알칼리 금속이 자외선 투명 경계를 갖는 이유다.

### 4.3 주요 금속의 플라즈마 주파수

| 금속 | $\omega_p$ (PHz) | $\lambda_p$ (nm) | 자외선 투명? |
|-------|-------------------|-------------------|-----------------|
| Na    | 8.8               | 214               | 예 ($\lambda < 214$ nm) |
| Al    | 22.9              | 82                | 예 (심자외선) |
| Au    | 13.7              | 138               | 밴드간 전이로 복잡 |
| Ag    | 14.0              | 135               | 가장 깔끔한 드루데 금속 |

```python
def drude_permittivity(omega, omega_p, gamma):
    """
    Drude model for metallic permittivity.

    Why the Drude model: despite its simplicity (free electrons + damping),
    it accurately predicts optical properties of simple metals like Al, Na, Ag
    from IR through visible frequencies.
    """
    return 1.0 - omega_p**2 / (omega**2 + 1j * gamma * omega)

# Silver parameters
omega_p_Ag = 14.0e15   # plasma frequency ~14 PHz
gamma_Ag = 0.032e15    # damping rate ~32 THz (low for Ag)

omega_metal = np.linspace(0.1e15, 25e15, 5000)
eps_Ag = drude_permittivity(omega_metal, omega_p_Ag, gamma_Ag)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(omega_metal / 1e15, eps_Ag.real, 'b-', linewidth=1.5, label="$\\epsilon'$")
ax.plot(omega_metal / 1e15, eps_Ag.imag, 'r--', linewidth=1.5, label="$\\epsilon''$")
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax.axvline(x=omega_p_Ag / 1e15, color='green', linestyle=':', label=f'$\\omega_p$ = {omega_p_Ag/1e15:.1f} PHz')

# Shade the opaque region where epsilon' < 0
ax.axvspan(0, omega_p_Ag / 1e15, alpha=0.1, color='gray', label='Opaque region')

ax.set_xlabel('Frequency (PHz)')
ax.set_ylabel('$\\epsilon_r$')
ax.set_title('Drude Model: Silver Permittivity')
ax.set_ylim(-10, 5)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("drude_silver.png", dpi=150)
plt.show()
```

---

## 5. 표피 깊이

### 5.1 유도

파동이 도전성 매질에 진입할 때, 복소 파수는:

$$\tilde{k} = \omega\sqrt{\mu\tilde{\epsilon}} = \omega\sqrt{\mu\left(\epsilon + i\frac{\sigma}{\omega}\right)}$$

양도체($\sigma \gg \omega\epsilon$)의 경우:

$$\tilde{k} \approx \sqrt{\frac{\mu\sigma\omega}{2}}(1 + i)$$

장(Field)은 $e^{-z/\delta}$ 로 감쇠하며, **표피 깊이(Skin Depth)** 는:

$$\boxed{\delta = \sqrt{\frac{2}{\mu\sigma\omega}}}$$

표피 깊이는 파동이 진입하면서 진폭이 $1/e \approx 37\%$ 로 줄어드는 거리를 나타낸다.

### 5.2 물리적 해석

표피 깊이는 다음 경우 감소한다:
- **높은 주파수** ($\delta \propto 1/\sqrt{\omega}$): 고주파 장은 표면에 집중됨
- **높은 전도율** ($\delta \propto 1/\sqrt{\sigma}$): 더 좋은 도체는 더 많이 반사하고, 더 얇은 층에서 흡수
- **높은 투자율** ($\delta \propto 1/\sqrt{\mu}$): 자성 물질은 표피 깊이가 더 작음

### 5.3 실용적 수치

| 물질 | $\sigma$ (S/m) | $\delta$ at 60 Hz | $\delta$ at 1 MHz | $\delta$ at 1 GHz |
|----------|-----------------|--------------------|--------------------|---------------------|
| 구리   | $5.96 \times 10^7$ | 8.5 mm | 0.066 mm | 2.1 $\mu$m |
| 알루미늄 | $3.77 \times 10^7$ | 10.7 mm | 0.083 mm | 2.6 $\mu$m |
| 해수 | 4.0              | 32.5 m  | 0.25 m   | 7.9 mm |

이것이 잠수함이 통신에 초장파(ELF, Extremely Low Frequency) 라디오를 사용하는 이유이며, 전자레인지가 2.45 GHz에서의 표피 깊이보다 작은 구멍이 뚫린 금속 망을 사용하는 이유다.

```python
def skin_depth(freq, sigma, mu_r=1.0):
    """
    Calculate electromagnetic skin depth.

    Why skin depth matters: it determines shielding effectiveness,
    heating depth in induction furnaces, and the minimum thickness
    of conductive coatings on waveguides.
    """
    mu = 4 * np.pi * 1e-7 * mu_r  # permeability (H/m)
    omega = 2 * np.pi * freq
    return np.sqrt(2.0 / (mu * sigma * omega))

# Compute skin depth vs frequency for copper
freq = np.logspace(1, 12, 1000)  # 10 Hz to 1 THz
sigma_Cu = 5.96e7  # copper conductivity (S/m)
delta_Cu = skin_depth(freq, sigma_Cu)

fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(freq, delta_Cu * 1e3, 'b-', linewidth=2, label='Copper')

# Add reference lines for common frequencies
ref_freqs = {'60 Hz': 60, '1 kHz': 1e3, '1 MHz': 1e6, '1 GHz': 1e9}
for name, f in ref_freqs.items():
    d = skin_depth(f, sigma_Cu) * 1e3
    ax.plot(f, d, 'ro', markersize=8)
    ax.annotate(f'{name}\n$\\delta$ = {d:.2g} mm',
                xy=(f, d), xytext=(f * 3, d * 2),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Skin depth (mm)')
ax.set_title('Skin Depth in Copper vs. Frequency')
ax.grid(True, which='both', alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig("skin_depth_copper.png", dpi=150)
plt.show()
```

---

## 6. 복소 굴절률과 흡수

### 6.1 정의

복소 굴절률은:

$$\tilde{n} = n + i\kappa$$

여기서 $n$ 은 (실수) 굴절률이고, $\kappa$ 는 **소광 계수(Extinction Coefficient)** 다. 이들은 복소 유전율과 다음 관계를 갖는다:

$$\tilde{n}^2 = \tilde{\epsilon}_r \mu_r$$

비자성 물질($\mu_r = 1$)의 경우:

$$n^2 - \kappa^2 = \epsilon', \quad 2n\kappa = \epsilon''$$

이를 풀면:

$$n = \sqrt{\frac{|\tilde{\epsilon}_r| + \epsilon'}{2}}, \quad \kappa = \sqrt{\frac{|\tilde{\epsilon}_r| - \epsilon'}{2}}$$

### 6.2 흡수 계수

파동 세기는 다음과 같이 감쇠한다:

$$I(z) = I_0 \, e^{-\alpha z}$$

**흡수 계수(Absorption Coefficient)** (또는 감쇠 계수)는:

$$\alpha = \frac{2\omega\kappa}{c} = \frac{2\kappa\omega}{c}$$

계수 2는 세기가 $|E|^2$ 에 비례하는 반면, 전기장 진폭은 $e^{-\kappa\omega z/c}$ 로 감쇠하기 때문에 나타난다.

### 6.3 비어-람베르트 법칙

분광학에서 흡수는 종종 다음과 같이 표현된다:

$$A = \log_{10}\left(\frac{I_0}{I}\right) = \frac{\alpha z}{2.303}$$

**몰 흡수 계수(Molar Absorption Coefficient)** $\varepsilon_m$ 은 미시적 흡수와 다음 관계를 갖는다:

$$\alpha = \varepsilon_m c_{\text{mol}} \ln 10$$

여기서 $c_{\text{mol}}$ 은 몰 농도다.

```python
def complex_refractive_index(eps_complex, mu_r=1.0):
    """
    Compute n and kappa from complex permittivity.

    Why separate n and kappa: experiments measure reflectance and
    transmittance, which map directly to n and kappa rather than
    epsilon' and epsilon''.
    """
    n_complex = np.sqrt(eps_complex * mu_r)
    return n_complex.real, n_complex.imag

# Demonstrate absorption in glass with a Lorentz resonance
omega_abs = np.linspace(1e14, 1.2e16, 5000)
eps_glass = lorentz_permittivity(omega_abs, omega_0=6e15, gamma=1e14, omega_p=4e15)

n_glass, kappa_glass = complex_refractive_index(eps_glass)
c = 3e8
alpha_glass = 2 * omega_abs * kappa_glass / c  # absorption coefficient (1/m)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axes[0].plot(omega_abs / 1e15, n_glass, 'b-', linewidth=1.5)
axes[0].set_ylabel('Refractive index $n$')
axes[0].set_title('Complex Refractive Index (Lorentz Model)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(omega_abs / 1e15, alpha_glass * 1e-6, 'r-', linewidth=1.5)
axes[1].set_xlabel('Frequency (PHz)')
axes[1].set_ylabel('Absorption coeff. $\\alpha$ (10$^6$ m$^{-1}$)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("complex_refractive_index.png", dpi=150)
plt.show()
```

---

## 7. 크라머스-크로니히 관계식

### 7.1 인과율과 해석성

유전 응답에 대한 가장 심오한 제약은 **인과율(Causality)** 에서 비롯된다: 시간 $t$ 에서의 분극(Polarization)은 미래의 전기장에 의존할 수 없다. 수학적으로, 응답 함수 $\chi(t)$ 는 $t < 0$ 에서 소멸한다.

이 인과율 조건과, $\chi(\omega)$ 가 복소 $\omega$ 평면의 상반부에서 해석적(Analytic)이고 무한대에서 감소한다는 가정을 결합하면, **크라머스-크로니히(KK) 관계식** 이 도출된다:

$$\epsilon'(\omega) - 1 = \frac{2}{\pi} \, \mathcal{P} \int_0^{\infty} \frac{\omega' \epsilon''(\omega')}{\omega'^2 - \omega^2} \, d\omega'$$

$$\epsilon''(\omega) = -\frac{2\omega}{\pi} \, \mathcal{P} \int_0^{\infty} \frac{\epsilon'(\omega') - 1}{\omega'^2 - \omega^2} \, d\omega'$$

여기서 $\mathcal{P}$ 는 코시 주값(Cauchy Principal Value)을 나타낸다.

### 7.2 물리적 의미

KK 관계식은 **분산과 흡수가 독립적이지 않다**는 것을 의미한다 — 모든 주파수에서 $\epsilon''(\omega)$ 를 알면 $\epsilon'(\omega)$ 를 완전히 결정할 수 있고, 그 역도 성립한다. 이것은 강력한 실험 도구다: 흡수 스펙트럼을 측정하면 별도의 측정 없이 굴절률을 계산할 수 있다.

> **비유**: 크라머스-크로니히 관계식은 전인적 건강 검진과 같다. 시스템이 모든 주파수에서 얼마나 에너지를 흡수하는지("증상")를 알면, 빛을 어떻게 굴절시키는지("진단")를 정확히 추론할 수 있다. 흡수 없이 분산이 있을 수 없듯, 증상 없이 근본 원인이 있을 수 없다.

### 7.3 합산 규칙

KK 관계식은 다음과 같은 합산 규칙(Sum Rules)을 함의한다:

$$\int_0^{\infty} \omega \, \epsilon''(\omega) \, d\omega = \frac{\pi}{2} \omega_p^2$$

이 **f-합산 규칙(f-Sum Rule)** 은 특정 모형에 무관하게 전체 흡수량을 전자 밀도와 연결한다.

```python
from scipy.integrate import simpson

def kramers_kronig_real(omega, eps_imag):
    """
    Compute epsilon'(omega) from epsilon''(omega) using Kramers-Kronig.

    Why KK is useful: in experiments, you often measure only absorption
    (eps'') via transmission. KK lets you recover eps' (refractive index)
    without an independent reflectance measurement.
    """
    eps_real = np.ones_like(omega)  # start with the 1 (vacuum contribution)
    domega = omega[1] - omega[0]

    for i, w in enumerate(omega):
        # Cauchy principal value: skip the singularity at omega' = omega
        integrand = omega * eps_imag / (omega**2 - w**2 + 1e-30)
        # Zero out the singular point
        if i < len(omega):
            integrand[i] = 0.0
        eps_real[i] += (2.0 / np.pi) * simpson(integrand, x=omega)

    return eps_real

# Verify KK on the Lorentz model
omega_kk = np.linspace(0.01e15, 1.5e16, 2000)
eps_lorentz = lorentz_permittivity(omega_kk, omega_0=6e15, gamma=1e14, omega_p=4e15)

# Use exact imaginary part to reconstruct real part
eps_real_kk = kramers_kronig_real(omega_kk, eps_lorentz.imag)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(omega_kk / 1e15, eps_lorentz.real, 'b-', linewidth=2, label="Exact $\\epsilon'$")
ax.plot(omega_kk / 1e15, eps_real_kk, 'r--', linewidth=2, label="KK from $\\epsilon''$")
ax.set_xlabel('Frequency (PHz)')
ax.set_ylabel("$\\epsilon'$")
ax.set_title("Kramers-Kronig Verification: Recovering $\\epsilon'$ from $\\epsilon''$")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("kramers_kronig_verification.png", dpi=150)
plt.show()
```

---

## 8. 분산 매질에서의 파동 전파: 시뮬레이션

가우시안 펄스가 분산 매질을 전파하면서 펄스 폭이 넓어지는 현상을 시뮬레이션해보자.

```python
def simulate_pulse_in_dispersive_medium(n_func, omega_center, bandwidth,
                                         z_max, c=3e8, N=4096):
    """
    Simulate Gaussian pulse propagation in a dispersive medium.

    Strategy:
    1. Construct pulse in frequency domain
    2. Multiply by transfer function exp(i * k(omega) * z)
    3. IFFT back to time domain

    Why frequency-domain approach: it naturally handles arbitrary dispersion
    relations without the numerical instabilities of direct PDE solvers.
    """
    # Frequency grid
    domega = 2 * bandwidth / N
    omega = omega_center + np.linspace(-bandwidth, bandwidth, N)

    # Gaussian pulse spectrum centered at omega_center
    pulse_spectrum = np.exp(-0.5 * ((omega - omega_center) / (bandwidth / 10))**2)

    # Propagation distances
    distances = [0, z_max / 4, z_max / 2, z_max]

    # Time grid (from inverse FFT)
    dt = 2 * np.pi / (N * domega)
    t = np.arange(N) * dt
    t -= t[N // 2]  # center the time axis

    fig, axes = plt.subplots(len(distances), 1, figsize=(10, 10), sharex=True)

    for ax, z in zip(axes, distances):
        # Transfer function: phase accumulated over distance z
        n_vals = n_func(omega)
        k_vals = n_vals * omega / c
        propagated = pulse_spectrum * np.exp(1j * k_vals * z)

        # Inverse FFT to get time-domain pulse
        pulse_time = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(propagated)))
        intensity = np.abs(pulse_time)**2
        intensity /= intensity.max()

        ax.plot(t * 1e15, intensity, 'b-', linewidth=1.5)
        ax.set_ylabel('Intensity (norm.)')
        ax.set_title(f'z = {z*1e6:.0f} $\\mu$m')
        ax.set_xlim(-500, 500)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (fs)')
    plt.suptitle('Gaussian Pulse in Dispersive Medium', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("pulse_dispersion.png", dpi=150)
    plt.show()

# Refractive index function from Lorentz model (away from resonance)
def n_dispersive(omega):
    eps = lorentz_permittivity(omega, omega_0=6e15, gamma=1e14, omega_p=4e15)
    return np.sqrt(eps).real

simulate_pulse_in_dispersive_medium(
    n_func=n_dispersive,
    omega_center=3e15,       # visible light ~3 PHz
    bandwidth=2e15,          # broad bandwidth for short pulse
    z_max=100e-6             # 100 micrometers
)
```

---

## 요약

| 개념 | 핵심 공식 | 물리적 의미 |
|---------|-------------|------------------|
| 복소 유전율 | $\tilde{\epsilon} = \epsilon' + i\epsilon''$ | 분산과 흡수를 통합 |
| 로렌츠 모형 | $\epsilon_r = 1 + \omega_p^2/(\omega_0^2 - \omega^2 - i\gamma\omega)$ | 속박 전자 응답 |
| 드루데 모형 | $\epsilon_r = 1 - \omega_p^2/(\omega^2 + i\gamma\omega)$ | 자유 전자 응답 |
| 표피 깊이 | $\delta = \sqrt{2/\mu\sigma\omega}$ | 도체에서의 침투 깊이 |
| 복소 굴절률 | $\tilde{n} = n + i\kappa$ | 위상 속도와 감쇠 |
| 흡수 계수 | $\alpha = 2\omega\kappa/c$ | 세기 감쇠율 |
| 크라머스-크로니히 | $\epsilon' \leftrightarrow \epsilon''$ (적분 관계) | 인과율이 분산과 흡수를 연결 |

---

## 연습 문제

### 연습 1: 알루미늄의 드루데 모형
알루미늄의 플라즈마 주파수는 $\omega_p = 2.29 \times 10^{16}$ rad/s, 감쇠율은 $\gamma = 1.22 \times 10^{14}$ rad/s다. (a) $\epsilon_r(\omega)$ 를 0에서 $3\omega_p$ 까지 그려라. (b) 알루미늄이 투명해지는 주파수를 구하라. (c) 1 GHz와 가시광선 주파수(500 nm)에서의 표피 깊이를 계산하라.

### 연습 2: 다중 공명 로렌츠 모형
유리는 자외선과 적외선에서 흡수 공명을 갖는다. 두 로렌츠 진동자로 유리를 모형화하라: 하나는 $\omega_1 = 1.5 \times 10^{16}$ rad/s (자외선), $\gamma_1 = 10^{14}$ rad/s, $f_1 = 0.6$, 다른 하나는 $\omega_2 = 6 \times 10^{13}$ rad/s (적외선), $\gamma_2 = 5 \times 10^{12}$ rad/s, $f_2 = 0.4$. $\omega_p = 2 \times 10^{16}$ rad/s를 사용하라. 가시광선 범위에서 $n(\omega)$ 와 $\kappa(\omega)$ 를 그려라. 이 모형은 유리가 가시광선에 투명함을 예측하는가?

### 연습 3: 크라머스-크로니히 수치 검증
은(Silver)의 드루데 모형($\omega_p = 14.0 \times 10^{15}$ rad/s, $\gamma = 3.2 \times 10^{13}$ rad/s)을 사용하라. (a) $\epsilon''(\omega)$ 를 해석적으로 계산하라. (b) 크라머스-크로니히 적분을 수치적으로 사용하여 $\epsilon'(\omega)$ 를 재구성하라. (c) 정확한 드루데 $\epsilon'(\omega)$ 와 비교하라. 수치 오차의 원인을 논하라.

### 연습 4: 펄스 폭 증가
중심 파장 800 nm의 10 fs 가우시안 펄스가 BK7 유리 1 cm를 통과한다(800 nm에서 GVD $\approx 44.7$ fs$^2$/mm). (a) $\tau_{\text{out}} = \tau_{\text{in}} \sqrt{1 + (4\ln 2 \cdot \text{GVD} \cdot L / \tau_{\text{in}}^2)^2}$ 공식을 사용하여 출력 펄스 폭을 추정하라. (b) 전파를 수치적으로 시뮬레이션하고 해석적 추정치와 비교하라.

---

[← 이전: 09. 진공에서의 전자기파](09_EM_Waves_Vacuum.md) | [다음: 11. 반사와 굴절 →](11_Reflection_and_Refraction.md)
