# 18. 응용 — 플라즈모닉스와 메타물질

[← 이전: 17. 전자기 산란](17_Electromagnetic_Scattering.md) | [다음: 개요 →](00_Overview.md)

## 학습 목표

1. 표면 플라즈몬 폴라리톤(surface plasmon polariton, SPP)과 금속-유전체 계면에서의 분산 관계를 이해한다
2. 금속 나노입자에서의 국소 표면 플라즈몬 공명(localized surface plasmon resonance, LSPR)을 분석한다
3. 나노입자 형상이 공명 파장을 결정하는 방식을 설명한다
4. 메타물질(metamaterial)을 정의하고 파장 이하의 구조가 유효 물질 특성을 어떻게 만드는지 이해한다
5. 음의 굴절률(negative refractive index) 조건을 유도하고 베셀라고(Veselago)의 예측을 이해한다
6. 분리 링 공진기(split-ring resonator)와 와이어 배열을 메타물질의 구성 요소로 설명한다
7. 바이오센서, 전자기 클로킹, 초분해능 영상, 광자 결정 등의 응용을 탐구한다

앞의 레슨들은 고전 전자기학의 토대를 쌓았다: 맥스웰 방정식, 파동 전파, 산란, 복사. 이 마지막 레슨은 이러한 토대가 현대 포토닉스의 가장 흥미로운 두 최전선을 어떻게 가능하게 하는지 보여준다. **플라즈모닉스(plasmonics)**는 금속의 자유 전자 집단 진동을 이용하여 회절 한계 이하인 나노미터 규모로 빛을 가두며, 초고감도 바이오센서와 나노스케일 광학 회로를 가능하게 한다. **메타물질(metamaterials)**은 더 대담한 접근을 취하며, 음의 굴절률, 전자기 투명 망토, 완전 렌즈처럼 자연에서는 불가능한 성질을 가진 인공 전자기 물질을 공학적으로 설계한다. 두 분야 모두 현대 나노 제조 기술과 결합된 맥스웰 방정식의 창조적 힘을 잘 보여준다.

> **비유**: 플라즈모닉스는 전자가 연주자인 오케스트라를 지휘하는 것과 같다. 지휘자(입사광)가 올바른 주파수를 맞추면, 금속 나노입자의 모든 전자가 완벽한 조화로 진동하며 강렬한 공명을 일으켜 "소리"(전자기장)를 아주 작은 부피에 집중시킨다. 메타물질은 반면에 악기를 처음부터 새로 만드는 것과 같다 — 파장보다 훨씬 작은 공명 구조를 특정 패턴으로 배열함으로써, 어떤 자연 물질도 낼 수 없는 "음표"(전자기적 특성)를 내는 물질을 만들어낸다.

---

## 1. 표면 플라즈몬 폴라리톤

### 1.1 표면 플라즈몬이란?

**표면 플라즈몬 폴라리톤(surface plasmon polariton, SPP)**은 금속-유전체 계면을 따라 전파하는 전자기파로, 두 매질 모두에서 지수적으로 감쇠하는 장을 갖는다. 이는 표면에서의 전자기장과 자유 전자 플라즈마의 결합 진동이다.

핵심 특징: SPP는 같은 주파수의 자유 공간 빛보다 **짧은** 파장을 갖는다. 이를 통해 회절 한계 이하로 빛을 가둘 수 있다.

### 1.2 분산 관계

금속($\epsilon_m(\omega)$, 보통 음수)과 유전체($\epsilon_d > 0$) 사이의 평면 계면을 생각하자. SPP 분산 관계는 TM 표면파의 경계 조건을 맞춤으로써 얻는다:

$$\boxed{k_{\text{SPP}} = \frac{\omega}{c}\sqrt{\frac{\epsilon_m \epsilon_d}{\epsilon_m + \epsilon_d}}}$$

금속에 드루드 모델(Drude model)을 사용하면($\epsilon_m = 1 - \omega_p^2/\omega^2$, 손실 없는 경우):

- $\omega \to 0$일 때: $k_{\text{SPP}} \approx (\omega/c)\sqrt{\epsilon_d}$ (광선에 접근)
- $\omega \to \omega_{sp} = \omega_p/\sqrt{1 + \epsilon_d}$일 때: $k_{\text{SPP}} \to \infty$ (점근 표면 플라즈몬 주파수)

SPP는 항상 광선 $\omega = ck/\sqrt{\epsilon_d}$의 오른쪽에 위치하므로, **자유 전파하는 빛으로는 직접 여기할 수 없다** — 특별한 결합 메커니즘이 필요하다.

### 1.3 장 프로파일

장은 계면으로부터 지수적으로 감쇠한다:

$$\mathbf{E} \propto e^{ik_{\text{SPP}}x - \kappa_d z} \quad (z > 0, \text{ 유전체})$$

$$\mathbf{E} \propto e^{ik_{\text{SPP}}x + \kappa_m z} \quad (z < 0, \text{ 금속})$$

여기서 $\kappa_{d,m} = \sqrt{k_{\text{SPP}}^2 - \epsilon_{d,m}\omega^2/c^2}$는 감쇠 상수이다. 유전체로의 침투 깊이는 보통 100-500 nm이고, 금속으로의 침투는 스킨 깊이(귀금속의 경우 약 20-30 nm)로 제한된다.

### 1.4 여기 방법

$k_{\text{SPP}} > k_0\sqrt{\epsilon_d}$이므로, 자유 공간 빛은 SPP를 직접 여기할 수 없다. 일반적인 결합 방식:

- **프리즘 결합(크레치만 구성, Kretschmann configuration)**: 프리즘에서의 전반사 에바네선트파가 추가 운동량을 제공
- **회절 격자 결합(grating coupling)**: 주기적인 표면 구조가 광자 파동 벡터에 $\pm n G$를 더함($G = 2\pi/\Lambda$는 격자 벡터)
- **근거리장 여기(near-field excitation)**: 나노 탐침이나 파장 이하 개구부가 넓은 스펙트럼의 파동 벡터를 제공

```python
import numpy as np
import matplotlib.pyplot as plt

def spp_dispersion(omega, omega_p, gamma=0, eps_d=1.0):
    """
    Compute SPP dispersion relation at a metal-dielectric interface.

    Parameters:
        omega   : angular frequency array (rad/s)
        omega_p : metal plasma frequency (rad/s)
        gamma   : metal damping rate (rad/s)
        eps_d   : dielectric constant of the dielectric medium

    Why SPP dispersion: it reveals the fundamental limits of
    light confinement — SPPs can have arbitrarily short wavelengths
    near the surface plasmon frequency, but at the cost of increasing loss.
    """
    c = 3e8

    # Drude permittivity
    eps_m = 1 - omega_p**2 / (omega**2 + 1j * gamma * omega)

    # SPP wave vector (complex for lossy metal)
    k_spp = (omega / c) * np.sqrt(eps_m * eps_d / (eps_m + eps_d))

    # Light line
    k_light = omega * np.sqrt(eps_d) / c

    # Surface plasmon frequency
    omega_sp = omega_p / np.sqrt(1 + eps_d)

    return k_spp, k_light, omega_sp

# Silver parameters
omega_p = 14.0e15   # plasma frequency (rad/s)
gamma = 3.2e13      # damping (rad/s)
omega = np.linspace(0.01e15, 12e15, 2000)

# Air-silver interface
k_spp_air, k_light_air, omega_sp_air = spp_dispersion(omega, omega_p, gamma, eps_d=1.0)

# Glass-silver interface
k_spp_glass, k_light_glass, omega_sp_glass = spp_dispersion(omega, omega_p, gamma, eps_d=2.25)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Dispersion diagram
ax = axes[0]
ax.plot(k_spp_air.real / 1e7, omega / 1e15, 'b-', linewidth=2, label='SPP (air-Ag)')
ax.plot(k_light_air / 1e7, omega / 1e15, 'b--', linewidth=1, alpha=0.5, label='Light line (air)')
ax.plot(k_spp_glass.real / 1e7, omega / 1e15, 'r-', linewidth=2, label='SPP (glass-Ag)')
ax.plot(k_light_glass / 1e7, omega / 1e15, 'r--', linewidth=1, alpha=0.5, label='Light line (glass)')

ax.axhline(y=omega_sp_air / 1e15, color='blue', linestyle=':', alpha=0.5)
ax.axhline(y=omega_sp_glass / 1e15, color='red', linestyle=':', alpha=0.5)

ax.set_xlabel('$k$ (10$^7$ rad/m)')
ax.set_ylabel('Frequency (PHz)')
ax.set_title('Surface Plasmon Polariton Dispersion')
ax.set_xlim(0, 5)
ax.set_ylim(0, 10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Propagation length (1/e decay of intensity)
ax = axes[1]
L_spp_air = 1.0 / (2 * k_spp_air.imag)
L_spp_glass = 1.0 / (2 * k_spp_glass.imag)

valid_air = (omega > 0.5e15) & (omega < omega_sp_air * 0.95)
valid_glass = (omega > 0.5e15) & (omega < omega_sp_glass * 0.95)

ax.semilogy(omega[valid_air] / 1e15, L_spp_air[valid_air] * 1e6,
            'b-', linewidth=2, label='Air-Ag')
ax.semilogy(omega[valid_glass] / 1e15, L_spp_glass[valid_glass] * 1e6,
            'r-', linewidth=2, label='Glass-Ag')

ax.set_xlabel('Frequency (PHz)')
ax.set_ylabel('Propagation length ($\\mu$m)')
ax.set_title('SPP Propagation Length')
ax.legend(fontsize=12)
ax.grid(True, which='both', alpha=0.3)
ax.set_ylim(0.1, 1000)

plt.tight_layout()
plt.savefig("spp_dispersion.png", dpi=150)
plt.show()
```

---

## 2. 국소 표면 플라즈몬 공명 (LSPR)

### 2.1 나노입자 공명

금속 나노입자(반지름 $a \ll \lambda$)에 빛이 조사되면 전도 전자가 집단적으로 진동한다. 준정적 근사(quasi-static approximation)를 사용하면, 구 내부와 외부의 전기장은:

$$\mathbf{E}_{\text{in}} = \frac{3\epsilon_d}{\epsilon_m + 2\epsilon_d}\mathbf{E}_0$$

$$\mathbf{E}_{\text{out}} = \mathbf{E}_0 + \frac{3\hat{r}(\hat{r}\cdot\mathbf{p}) - \mathbf{p}}{4\pi\epsilon_0\epsilon_d r^3}$$

여기서 $\mathbf{p} = 4\pi\epsilon_0\epsilon_d a^3 \frac{\epsilon_m - \epsilon_d}{\epsilon_m + 2\epsilon_d}\mathbf{E}_0$는 유도 쌍극자 모멘트이다.

### 2.2 프뢸리히 조건

분모가 최소가 될 때 공명이 일어난다:

$$\boxed{\text{Re}[\epsilon_m(\omega)] = -2\epsilon_d}$$

이것이 **프뢸리히 조건(Frohlich condition)**이다. 진공 중($\epsilon_d = 1$) 드루드 금속의 경우:

$$\omega_{\text{LSPR}} = \frac{\omega_p}{\sqrt{3}}$$

공명 주파수는 다음에 의존한다:
- **입자 재료**: 금속마다 $\omega_p$와 감쇠가 다름
- **주변 매질**: $\epsilon_d$가 클수록 공명이 적색 편이
- **입자 형상**: 타원체, 막대, 기타 형상은 서로 다른 탈분극 인수를 가짐

### 2.3 장 증강

공명에서 국소 전기장은 입사장에 비해 10-1000배까지 증강될 수 있다. 이 **근거리장 증강(near-field enhancement)**은 다음의 기반이 된다:
- 표면 증강 라만 분광법(SERS): 증강 $\propto |E/E_0|^4$, $10^6$-$10^{10}$배 강한 라만 신호 달성
- 국소 표면 플라즈몬 공명 바이오센서: 분석 분자 결합 시 스펙트럼 편이
- 플라즈모닉 태양 전지: 박막 광전지에서의 흡수 증대

```python
def nanoparticle_polarizability(omega, omega_p, gamma, a, eps_d=1.0):
    """
    Compute the polarizability of a metallic nanosphere.

    Parameters:
        omega   : angular frequency (rad/s)
        omega_p : plasma frequency (rad/s)
        gamma   : damping rate (rad/s)
        a       : sphere radius (m)
        eps_d   : dielectric constant of surrounding medium

    Why polarizability: it determines the absorption and scattering
    cross sections of the nanoparticle, which are the measurable
    quantities in optical experiments.
    """
    eps_m = 1 - omega_p**2 / (omega**2 + 1j * gamma * omega)
    alpha = 4 * np.pi * a**3 * (eps_m - eps_d) / (eps_m + 2 * eps_d)
    return alpha

def nanoparticle_cross_sections(omega, omega_p, gamma, a, eps_d=1.0):
    """Compute absorption and scattering cross sections."""
    c = 3e8
    k = omega * np.sqrt(eps_d) / c
    alpha = nanoparticle_polarizability(omega, omega_p, gamma, a, eps_d)

    # Absorption cross section (dominant for small particles)
    sigma_abs = k * np.imag(alpha)

    # Scattering cross section (proportional to a^6)
    sigma_sca = k**4 / (6 * np.pi) * np.abs(alpha)**2

    sigma_ext = sigma_abs + sigma_sca
    return sigma_ext, sigma_abs, sigma_sca

def plot_lspr():
    """Visualize LSPR for different metals and environments."""
    omega = np.linspace(1e15, 8e15, 2000)
    a = 20e-9  # 20 nm radius

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Silver nanoparticle in different media
    ax = axes[0, 0]
    for eps_d, label, color in [(1.0, 'Vacuum', 'blue'),
                                 (1.77, 'Water', 'green'),
                                 (2.25, 'Glass', 'red')]:
        sigma_ext, _, _ = nanoparticle_cross_sections(
            omega, 14e15, 3.2e13, a, eps_d)
        # Convert to nm² and normalize
        ax.plot(2*np.pi*3e8/omega * 1e9, sigma_ext * 1e18,
                color=color, linewidth=2, label=label)

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('$\\sigma_{\\mathrm{ext}}$ (nm²)')
    ax.set_title('Silver NP (20 nm radius) in Different Media')
    ax.set_xlim(300, 800)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Gold nanoparticle (interband transitions modify the response)
    ax = axes[0, 1]
    omega_p_Au = 13.7e15
    gamma_Au = 1.075e14  # higher damping than Ag
    for a_nm in [10, 20, 40, 80]:
        a_val = a_nm * 1e-9
        sigma_ext, _, _ = nanoparticle_cross_sections(
            omega, omega_p_Au, gamma_Au, a_val, eps_d=1.0)
        ax.plot(2*np.pi*3e8/omega * 1e9, sigma_ext / (np.pi * a_val**2),
                linewidth=2, label=f'{a_nm} nm')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('$Q_{\\mathrm{ext}}$')
    ax.set_title('Gold NP: Size Dependence')
    ax.set_xlim(300, 800)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Field enhancement at resonance
    ax = axes[1, 0]
    eps_d = 1.0
    omega_res = omega[np.argmax(np.abs(
        nanoparticle_polarizability(omega, 14e15, 3.2e13, 20e-9, eps_d)))]

    # Field enhancement as a function of position along the axis
    r = np.linspace(1.01 * a, 10 * a, 200)
    # On-axis enhancement (theta=0): |E/E0|^2 = |1 + 2*alpha/(4*pi*r^3)|^2
    alpha_res = nanoparticle_polarizability(
        np.array([omega_res]), 14e15, 3.2e13, a, eps_d)[0]
    enhancement = np.abs(1 + 2 * alpha_res / (4 * np.pi * r**3))**2

    ax.semilogy(r / a, enhancement, 'b-', linewidth=2)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1, color='red', linestyle=':', label='Particle surface')
    ax.set_xlabel('Distance from center ($r/a$)')
    ax.set_ylabel('$|E/E_0|^2$')
    ax.set_title('Near-Field Enhancement (Ag, on-axis)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    # Biosensor application: spectral shift
    ax = axes[1, 1]
    eps_d_values = np.linspace(1.0, 2.5, 20)
    lambda_res = np.zeros(len(eps_d_values))

    for i, eps_d in enumerate(eps_d_values):
        sigma_ext, _, _ = nanoparticle_cross_sections(
            omega, 14e15, 3.2e13, 20e-9, eps_d)
        lambda_res[i] = 2 * np.pi * 3e8 / omega[np.argmax(sigma_ext)] * 1e9

    ax.plot(eps_d_values, lambda_res, 'bo-', linewidth=2, markersize=6)
    ax.set_xlabel('Surrounding $\\epsilon_d$')
    ax.set_ylabel('LSPR wavelength (nm)')
    ax.set_title('LSPR Sensitivity to Environment')
    ax.grid(True, alpha=0.3)

    # Sensitivity
    sensitivity = np.gradient(lambda_res, eps_d_values)
    ax2 = ax.twinx()
    ax2.plot(eps_d_values, sensitivity, 'r--', linewidth=1.5, label='Sensitivity')
    ax2.set_ylabel('Sensitivity (nm/RIU)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.suptitle('Localized Surface Plasmon Resonance', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("lspr_properties.png", dpi=150)
    plt.show()

plot_lspr()
```

---

## 3. 메타물질

### 3.1 개념

**메타물질(metamaterials)**은 자연에서 발견되지 않는 전자기적 특성을 가진 인공 구조 물질이다. 핵심 아이디어: 파장 이하의 공명 요소(메타 원자)를 주기적 패턴으로 배열한다. 단위 셀이 파장보다 훨씬 작을 때($a \ll \lambda$), 이 매질은 유효 유전율 $\epsilon_{\text{eff}}(\omega)$와 유효 투자율 $\mu_{\text{eff}}(\omega)$로 기술할 수 있다.

### 3.2 음의 굴절률

1968년, 베셀라고(Veselago)는 $\epsilon$과 $\mu$가 **동시에 음수**인 물질이 다음을 가짐을 보였다:

$$n = -\sqrt{|\epsilon_r||\mu_r|}$$

이 **음의 굴절률(negative refractive index)**은 놀라운 결과를 낳는다:
- 스넬의 법칙에 따르면 굴절이 법선의 **같은 쪽**으로 일어남
- 포인팅 벡터(에너지 흐름)가 파동 벡터(위상 전파)와 반평행
- 도플러 효과가 반전됨
- 체렌코프 복사가 뒤쪽으로 방출됨

### 3.3 구성 요소

**와이어 배열(wire arrays, 금속 와이어)**: 와이어 간격과 반지름에 의존하는 플라즈마 주파수 이하에서 음의 $\epsilon_{\text{eff}}$를 제공한다:

$$\epsilon_{\text{eff}} = 1 - \frac{\omega_{p,\text{eff}}^2}{\omega^2}, \quad \omega_{p,\text{eff}}^2 = \frac{2\pi c^2}{a^2 \ln(a/r)}$$

**분리 링 공진기(split-ring resonators, SRRs)**: 공명 주파수 근처에서 음의 $\mu_{\text{eff}}$를 제공한다:

$$\mu_{\text{eff}} = 1 - \frac{F\omega^2}{\omega^2 - \omega_0^2 + i\gamma\omega}$$

여기서 $F$는 충전 인수(filling factor)이고 $\omega_0$는 링의 LC 공명이다.

와이어 배열과 SRR을 결합하면 $\epsilon_{\text{eff}} < 0$이고 $\mu_{\text{eff}} < 0$인 주파수 대역을 달성할 수 있어, 음의 굴절률 메타물질이 만들어진다.

```python
def metamaterial_effective_properties(omega, omega_p_eff, omega_0_srr,
                                      gamma_srr, F=0.5, gamma_wire=0):
    """
    Compute effective permittivity and permeability of a metamaterial
    consisting of wire arrays + split-ring resonators.

    Parameters:
        omega       : angular frequency (rad/s)
        omega_p_eff : effective plasma frequency of wire array (rad/s)
        omega_0_srr : resonance frequency of SRRs (rad/s)
        gamma_srr   : damping of SRRs (rad/s)
        F           : filling factor of SRRs
        gamma_wire  : damping of wire array (rad/s)

    Why effective medium: when meta-atoms are much smaller than lambda,
    the metamaterial behaves like a continuous medium with engineered
    epsilon and mu — exactly as an ordinary dielectric behaves as
    continuous despite being made of discrete atoms.
    """
    # Wire array: effective permittivity (Drude-like)
    eps_eff = 1 - omega_p_eff**2 / (omega**2 + 1j * gamma_wire * omega)

    # SRR: effective permeability (Lorentz-like)
    mu_eff = 1 - F * omega**2 / (omega**2 - omega_0_srr**2 + 1j * gamma_srr * omega)

    # Effective refractive index
    n_eff = np.sqrt(eps_eff * mu_eff)

    # Choose the correct branch: Re(n) < 0 when both eps and mu are negative
    # This is the key physics: the sign of n must be chosen consistently
    # with causality (Im(n) > 0 for passive media)
    n_eff = np.where(
        (np.real(eps_eff) < 0) & (np.real(mu_eff) < 0),
        -np.abs(n_eff.real) + 1j * np.abs(n_eff.imag),
        n_eff
    )

    return eps_eff, mu_eff, n_eff


def plot_metamaterial():
    """Visualize metamaterial effective properties and negative index band."""
    omega = np.linspace(0.1e10, 20e10, 2000)

    omega_p_eff = 12e10   # wire array plasma frequency ~12 GHz
    omega_0_srr = 8e10    # SRR resonance ~8 GHz
    gamma_srr = 0.3e10    # SRR damping
    gamma_wire = 0.1e10   # wire damping

    eps_eff, mu_eff, n_eff = metamaterial_effective_properties(
        omega, omega_p_eff, omega_0_srr, gamma_srr,
        F=0.5, gamma_wire=gamma_wire)

    freq_GHz = omega / (2 * np.pi * 1e9)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Effective permittivity
    ax = axes[0, 0]
    ax.plot(freq_GHz, eps_eff.real, 'b-', linewidth=2, label="$\\epsilon'$")
    ax.plot(freq_GHz, eps_eff.imag, 'b--', linewidth=1.5, label="$\\epsilon''$")
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('$\\epsilon_{\\mathrm{eff}}$')
    ax.set_title('Effective Permittivity (Wire Array)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 3)

    # Effective permeability
    ax = axes[0, 1]
    ax.plot(freq_GHz, mu_eff.real, 'r-', linewidth=2, label="$\\mu'$")
    ax.plot(freq_GHz, mu_eff.imag, 'r--', linewidth=1.5, label="$\\mu''$")
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('$\\mu_{\\mathrm{eff}}$')
    ax.set_title('Effective Permeability (SRRs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 3)

    # Effective refractive index
    ax = axes[1, 0]
    ax.plot(freq_GHz, n_eff.real, 'k-', linewidth=2, label="$n'$ (real)")
    ax.plot(freq_GHz, n_eff.imag, 'k--', linewidth=1.5, label="$n''$ (imag)")
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    # Shade the negative-index band
    neg_idx = (np.real(eps_eff) < 0) & (np.real(mu_eff) < 0)
    if np.any(neg_idx):
        freq_neg = freq_GHz[neg_idx]
        ax.axvspan(freq_neg.min(), freq_neg.max(), alpha=0.2, color='green',
                   label='Negative index band')

    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('$n_{\\mathrm{eff}}$')
    ax.set_title('Effective Refractive Index')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 5)

    # Property map
    ax = axes[1, 1]
    eps_r = eps_eff.real
    mu_r = mu_eff.real

    # Color regions by quadrant
    ax.scatter(eps_r, mu_r, c=freq_GHz, cmap='viridis', s=3, alpha=0.5)

    ax.axhline(y=0, color='black', linewidth=1)
    ax.axvline(x=0, color='black', linewidth=1)

    # Label quadrants
    ax.text(1.5, 1.5, 'Standard\nmaterials', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(-2, -2, 'Negative\nindex', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax.text(-2, 1.5, '$\\epsilon < 0$\n(metals)', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax.text(1.5, -2, '$\\mu < 0$\n(SRRs only)', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    ax.set_xlabel("$\\epsilon'_{\\mathrm{eff}}$")
    ax.set_ylabel("$\\mu'_{\\mathrm{eff}}$")
    ax.set_title('$\\epsilon$-$\\mu$ Phase Space')
    ax.set_xlim(-4, 3)
    ax.set_ylim(-4, 3)
    ax.grid(True, alpha=0.3)
    cb = plt.colorbar(ax.collections[0], ax=ax, label='Frequency (GHz)')

    plt.suptitle('Metamaterial Effective Properties', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("metamaterial_properties.png", dpi=150)
    plt.show()

plot_metamaterial()
```

---

## 4. 전자기 클로킹

### 4.1 변환 광학

**변환 광학(transformation optics)**은 좌표 변환 하에서 맥스웰 방정식이 형태 불변성을 갖는다는 점을 이용한다. 좌표 변환 $\mathbf{r} \to \mathbf{r}'$은 물리 공간에서 변환된 물질 매개변수를 갖는 등가 문제로 대응된다:

$$\epsilon'_{ij} = \frac{J_i^{\ a} J_j^{\ b} \epsilon_{ab}}{\det(\mathbf{J})}, \quad \mu'_{ij} = \frac{J_i^{\ a} J_j^{\ b} \mu_{ab}}{\det(\mathbf{J})}$$

여기서 $J_i^{\ a} = \partial x'^i / \partial x^a$는 변환의 야코비안이다.

### 4.2 원통형 클로크

반지름 $a$인 원통을 은폐하려면, 다음 변환을 이용하여 $0 \leq r \leq b$ 영역을 환형 영역 $a \leq r' \leq b$로 압축한다:

$$r' = a + r\frac{b - a}{b}, \quad \theta' = \theta, \quad z' = z$$

이를 위해 클로크 쉘에서 비등방성이고 비균일한 물질 매개변수가 필요하다:

$$\epsilon_r' = \mu_r' = \frac{r' - a}{r'}, \quad \epsilon_\theta' = \mu_\theta' = \frac{r'}{r' - a}, \quad \epsilon_z' = \mu_z' = \left(\frac{b}{b - a}\right)^2 \frac{r' - a}{r'}$$

내부 경계($r' = a$)에서 $\epsilon_r' = \mu_r' = 0$이고 $\epsilon_\theta' = \mu_\theta' \to \infty$로, 실현하기 매우 어려운 극단적인 매개변수 값이다.

### 4.3 실용적 한계

- **대역폭**: 클로킹은 분산 물질을 요구하여 대역폭이 제한됨
- **손실**: 실제 메타물질에는 손실이 있어 그림자가 생김
- **크기**: 지금까지는 마이크로파 주파수(센티미터 규모)에서만 시연됨
- **불완전한 매개변수**: 단순화된 설계는 성능을 저하시킴

> **비유**: 강물이 바위 주위로 흐르는 것을 상상해보자. 강바닥을 정교하게 형성하면, 물이 바위 주위로 매끄럽게 흘러 하류에서 마치 바위가 없는 것처럼 다시 합쳐진다. 변환 광학은 빛에 대해 같은 일을 한다 — 전자기파가 은폐된 물체 주위로 매끄럽게 흐르도록 "강바닥"(물질 매개변수)을 설계하는 것이다.

---

## 5. 광자 결정

### 5.1 개념

광자 결정(photonic crystals)은 파장에 필적하는 주기를 가진 주기적 유전체 구조이다. 이들은 **광자 밴드갭(photonic band gaps)** — 전자기 모드가 전파할 수 없는 주파수 범위 — 을 만들며, 반도체의 전자 밴드갭과 유사하다.

### 5.2 1차원 광자 결정 (브래그 거울)

높은 굴절률과 낮은 굴절률 층($n_H$, $n_L$)이 교대로 쌓인 다층 구조에서 각 층의 두께가 1/4 파장이면, 설계 파장을 중심으로 반사 대역이 형성된다.

1/4 파장 적층의 밴드갭 폭은 대략 다음과 같다:

$$\frac{\Delta\omega}{\omega_0} = \frac{4}{\pi}\arcsin\left(\frac{n_H - n_L}{n_H + n_L}\right)$$

### 5.3 2차원 및 3차원 광자 결정

2D와 3D에서 밴드 구조는 격자 대칭에 의존한다:
- **유전체 내 공기 구멍의 2D 삼각 격자**: 큰 TE 밴드갭
- **유전체 막대의 2D 사각 격자**: 큰 TM 밴드갭
- **3D 다이아몬드 격자 (우드파일 구조)**: 완전한 3D 밴드갭

```python
def photonic_crystal_1d(n_H, n_L, N_periods, wavelength_range, lambda_design):
    """
    Compute reflectance of a 1D photonic crystal (Bragg mirror)
    using the transfer matrix method.

    Why photonic crystals: they enable complete control over light
    propagation — waveguiding around sharp bends, ultra-high-Q cavities,
    and engineered group velocity for slow light applications.
    """
    d_H = lambda_design / (4 * n_H)
    d_L = lambda_design / (4 * n_L)

    R = np.zeros(len(wavelength_range))

    for idx, lam in enumerate(wavelength_range):
        M = np.eye(2, dtype=complex)

        for _ in range(N_periods):
            # High-index layer
            delta_H = 2 * np.pi * n_H * d_H / lam
            M_H = np.array([
                [np.cos(delta_H), -1j * np.sin(delta_H) / n_H],
                [-1j * n_H * np.sin(delta_H), np.cos(delta_H)]
            ])

            # Low-index layer
            delta_L = 2 * np.pi * n_L * d_L / lam
            M_L = np.array([
                [np.cos(delta_L), -1j * np.sin(delta_L) / n_L],
                [-1j * n_L * np.sin(delta_L), np.cos(delta_L)]
            ])

            M = M @ M_H @ M_L

        # Reflection coefficient (air substrate)
        n_sub = 1.52  # glass substrate
        n_0 = 1.0     # air

        r = ((M[0,0] + M[0,1]*n_sub)*n_0 - (M[1,0] + M[1,1]*n_sub)) / \
            ((M[0,0] + M[0,1]*n_sub)*n_0 + (M[1,0] + M[1,1]*n_sub))

        R[idx] = np.abs(r)**2

    return R


def plot_photonic_crystal():
    """Demonstrate photonic band gap in a 1D photonic crystal."""
    lam = np.linspace(300, 900, 1000) * 1e-9  # nm
    lambda_design = 550e-9  # green

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Effect of number of periods
    ax = axes[0]
    for N in [2, 5, 10, 20]:
        R = photonic_crystal_1d(n_H=2.3, n_L=1.38, N_periods=N,
                                wavelength_range=lam, lambda_design=lambda_design)
        ax.plot(lam * 1e9, R * 100, linewidth=1.5, label=f'{N} periods')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance (%)')
    ax.set_title('1D Photonic Crystal: Effect of Period Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Effect of index contrast
    ax = axes[1]
    for n_H, n_L, label in [(1.5, 1.38, 'Low contrast'),
                              (2.0, 1.38, 'Medium'),
                              (2.3, 1.38, 'TiO$_2$/MgF$_2$'),
                              (3.5, 1.38, 'High contrast')]:
        R = photonic_crystal_1d(n_H=n_H, n_L=n_L, N_periods=10,
                                wavelength_range=lam, lambda_design=lambda_design)
        ax.plot(lam * 1e9, R * 100, linewidth=1.5,
                label=f'$n_H$={n_H}, $n_L$={n_L}')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance (%)')
    ax.set_title('1D Photonic Crystal: Effect of Index Contrast')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Photonic Crystal Band Gaps', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("photonic_crystal.png", dpi=150)
    plt.show()

plot_photonic_crystal()
```

---

## 6. 응용 요약

### 6.1 바이오센서

LSPR 기반 바이오센서는 나노입자 표면에 분자가 결합할 때 공명 파장의 편이를 통해 분자를 감지한다. 전형적인 감도: 굴절률 단위(RIU)당 100-400 nm이며, 검출 한계는 단일 분자에 가까워진다.

### 6.2 초분해능 영상

**베셀라고-펜드리 완전 렌즈(Veselago-Pendry perfect lens)**는 음의 굴절률 물질 슬래브를 이용하여 에바네선트파(파장 이하 공간 정보를 담고 있는)를 증폭하고 완전한 상을 재구성한다. 진정한 완전 렌즈는 아직 실용화되지 않았지만, 은 박막을 이용한 근거리장 초렌즈는 $\lambda/6$ 이하의 분해능을 시연했다.

### 6.3 광학 컴퓨팅

플라즈모닉 도파관은 나노미터 폭의 채널에서 신호를 전달할 수 있어, 고밀도 집적 회로에서 전자 상호 연결을 대체할 가능성이 있다. 현재 과제: 전파 손실(플라즈몬 전파 길이는 보통 10-100 $\mu$m).

### 6.4 에너지 수확

메타물질 완전 흡수체(metamaterial perfect absorber)는 특정 파장을 거의 100% 효율로 흡수하도록 설계할 수 있어, 열광발전(thermophotovoltaics)과 열 방출체에 유용하다.

| 응용 | 기술 | 현황 |
|-------------|-----------|--------|
| LSPR 바이오센서 | 금 나노입자 | 상용 제품 출시 |
| SERS | 나노구조 기판 | 분석화학에서 널리 사용 |
| 메타물질 흡수체 | 분리 링 + 와이어 배열 | 마이크로파/테라헤르츠 시연 |
| 전자기 클로킹 | 변환 광학 | 마이크로파에서 개념 증명 |
| 초렌즈 | 은 박막 | UV에서 근거리장 시연 |
| 광자 결정 | 2D/3D 주기 구조 | 광학 섬유, LED, 레이저 |

---

## 요약

| 개념 | 핵심 공식/원리 | 물리적 의미 |
|---------|----------------------|------------------|
| SPP 분산 | $k_{\text{SPP}} = (\omega/c)\sqrt{\epsilon_m\epsilon_d/(\epsilon_m + \epsilon_d)}$ | 금속-유전체 계면의 표면파 |
| LSPR 조건 | $\text{Re}[\epsilon_m] = -2\epsilon_d$ | 나노입자 공명 |
| 장 증강 | 공명에서 $|E/E_0|^2 \gg 1$ | SERS, 바이오센서의 기반 |
| 음의 굴절률 | $\epsilon < 0$이고 $\mu < 0$ 동시에 | 스넬의 법칙 반전 |
| 와이어 배열 | $\epsilon_{\text{eff}} = 1 - \omega_{p,\text{eff}}^2/\omega^2$ | 인공 플라즈마 |
| SRR | $\mu_{\text{eff}} = 1 - F\omega^2/(\omega^2 - \omega_0^2 + i\gamma\omega)$ | 자기 공명 |
| 변환 광학 | 야코비안으로부터 $\epsilon', \mu'$ | 좌표 기반 물질 설계 |
| 광자 밴드갭 | $\Delta\omega/\omega_0 \propto (n_H - n_L)/(n_H + n_L)$ | 금지 주파수 범위 |

---

## 연습 문제

### 연습 1: SPP 여기 설계
$\lambda = 633$ nm (He-Ne 레이저)에서 50 nm 은 박막의 SPP를 여기하는 크레치만 프리즘 결합기를 설계하라. (a) 은의 드루드 모델을 사용하여 $k_{\text{SPP}}$를 계산하라. (b) BK7 유리 프리즘($n = 1.515$)에서 필요한 입사각을 구하라. (c) 각도 반사율 스펙트럼(SPP 결합 각도에서 날카로운 딥)을 그려라. (d) 은 표면에 10 nm 두께의 단백질 층($n = 1.45$)이 흡착되면 결합 각도는 어떻게 변하는가?

### 연습 2: 나노입자 형상 효과
금 나노로드를 종횡비 $R = a/b$(장축/단축)인 회전 타원체로 모델링할 수 있다. 장축의 LSPR은 $\text{Re}[\epsilon_m] = -(1 + 1/L_a)\epsilon_d$일 때 발생하며, $L_a$는 장축 방향의 탈분극 인수이다. (a) 종횡비 1(구), 2, 3, 4, 5에 대한 $L_a$를 계산하라. (b) 종횡비의 함수로 LSPR 파장을 그려라. (c) 금 나노로드가 가시광선에서 근적외선까지 조정 가능한 이유를 설명하라.

### 연습 3: 메타물질 밴드 구조
$\omega_{p,\text{eff}} = 12$ GHz, $\omega_0 = 8$ GHz, $\gamma = 0.3$ GHz, $F = 0.5$인 메타물질에 대해 다음을 계산하라: (a) $n_{\text{eff}} < 0$인 주파수 대역. (b) 음의 굴절률 대역에서 주파수의 함수로 성능 지수 FOM $= |n'|/n''$. (c) 최대 FOM은 얼마이며, 어느 주파수에서 발생하는가?

### 연습 4: 광자 결정 결함
1차원 광자 결정에 한 층의 두께를 바꾸어 "결함"을 도입하라. 전달 행렬법을 사용하여, 밴드갭 내에 날카로운 투과 피크가 나타남을 보여라. (a) 양쪽 주기 수의 함수로 이 결함 모드의 Q 인수를 계산하라. (b) 결함 층의 두께를 변화시킬 때 결함 모드 주파수는 어떻게 변하는가?

### 연습 5: 클로킹 성능
원통형 전자기 클로크의 2D FDTD 시뮬레이션을 구현하라. 단순화된 축소 매개변수 클로크(오직 $\epsilon_r$과 $\epsilon_\theta$만 변하고 $\mu = 1$)를 사용하라. (a) 클로크 없이 PEC 원통에서 평면파 산란을 시뮬레이션하라. (b) 클로크를 추가하고 산란 감소를 관찰하라. (c) 클로크 유무에 따른 전체 산란 단면적을 측정하라. (d) 주파수가 설계 주파수에서 벗어날수록 클로킹 성능은 어떻게 저하되는가?

---

[← 이전: 17. 전자기 산란](17_Electromagnetic_Scattering.md) | [다음: 개요 →](00_Overview.md)
