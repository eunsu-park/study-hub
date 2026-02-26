# 17. 전자기 산란

[← 이전: 16. 전산 전자기학](16_Computational_Electrodynamics.md) | [다음: 18. 응용 — 플라즈모닉스와 메타물질 →](18_Plasmonics_and_Metamaterials.md)

## 학습 목표

1. 입사장(incident field), 산란장(scattered field), 전체장(total field)으로 전자기 산란 문제를 정식화한다
2. 파장보다 훨씬 작은 입자에 대한 레일리 산란(Rayleigh scattering)을 유도하고, 하늘이 파란 이유를 설명한다
3. 임의 크기 구형 입자에 대한 미 이론(Mie theory)을 이해한다
4. 산란(scattering), 흡수(absorption), 소광(extinction) 단면적을 정의하고 계산한다
5. 약한 산란체에 대한 본 근사(Born approximation)와 광학 정리(optical theorem)를 적용한다
6. 미분 산란 단면적을 계산하고 각도 분포를 이해한다
7. 산란 이론을 레이더(radar), 라이다(lidar), 의료 영상 분야의 실제 응용과 연결한다

전자기파가 장애물 — 빗방울, 나노입자, 항공기 — 과 만나면 파동의 일부가 여러 방향으로 굴절된다. 이것이 **산란(scattering)**이며, 우리가 보는 모든 것을 형성한다: 파란 하늘, 흰 구름, 붉은 노을, 무지개. 산란은 또한 레이더(항공기 탐지)에서 라이다(대기 프로파일링), 의료 영상(광학 간섭 단층 촬영)에 이르는 기술의 토대를 이룬다. 이 레슨에서는 가장 단순한 경우(작은 입자, 레일리 영역)에서 시작하여 구에 대한 정확한 해(미 이론)로 나아가며, 전자기 산란 이론을 기본 원리로부터 전개한다. 그 과정에서 이론과 실험을 연결하는 단면적과 산란 진폭의 언어를 배운다.

> **비유**: 테니스공(입사파)을 농구공(산란체)에 던진다고 하자. 테니스공은 어디에 맞느냐에 따라 다른 방향으로 튕겨 나간다. 같은 방향에서 많은 테니스공을 던지면, 공들은 농구공 주변에서 특징적인 패턴으로 산란된다. "산란 단면적"은 유효 표적 면적으로, 입사 플럭스 중 얼마나 많은 부분이 다른 방향으로 바뀌는지 알려준다. 놀랍게도, 전자기파의 경우 파장에 따라 단면적이 입자의 물리적 크기보다 훨씬 크거나 작을 수 있다.

---

## 1. 산란 문제

### 1.1 설정

입사 평면파 $\mathbf{E}_i = \mathbf{E}_0 e^{i(\mathbf{k}\cdot\mathbf{r} - \omega t)}$가 유한한 크기의 산란체에 입사한다. 전체장은 다음과 같다:

$$\mathbf{E}_{\text{total}} = \mathbf{E}_i + \mathbf{E}_s$$

여기서 $\mathbf{E}_s$는 **산란장(scattered field)**이다. 산란체로부터 멀리서 ($r \to \infty$), 산란장은 발산하는 구면파 형태를 취한다:

$$\mathbf{E}_s \to f(\theta, \phi) \frac{e^{ikr}}{r} \hat{e}_s$$

함수 $f(\theta, \phi)$는 **산란 진폭(scattering amplitude)**으로, 산란 과정의 각도 정보를 모두 담고 있다.

### 1.2 단면적

**미분 산란 단면적(differential scattering cross section)**은 단위 입체각당 산란 파워를 입사 세기로 나눈 값으로 정의된다:

$$\frac{d\sigma_s}{d\Omega} = |f(\theta, \phi)|^2$$

**전체 산란 단면적(total scattering cross section)**은 모든 각도에 대해 적분한다:

$$\sigma_s = \int |f(\theta, \phi)|^2 \, d\Omega$$

산란체가 에너지를 흡수하기도 하면, **흡수 단면적(absorption cross section)** $\sigma_a$가 이를 설명하며, **소광 단면적(extinction cross section)**은 다음과 같다:

$$\sigma_{\text{ext}} = \sigma_s + \sigma_a$$

소광 단면적은 입사 빔이 얼마나 감쇠하는지(산란과 흡수 모두)를 결정한다.

### 1.3 크기 매개변수

산란의 물리는 입자 크기 대 파장의 비율에 결정적으로 의존하며, 이는 **크기 매개변수(size parameter)**로 표현된다:

$$x = \frac{2\pi a}{\lambda} = ka$$

여기서 $a$는 입자 반지름이다. 세 가지 영역이 나타난다:

| 영역 | 조건 | 주요 특징 |
|--------|-----------|-------------|
| 레일리(Rayleigh) | $x \ll 1$ | 산란 $\propto \lambda^{-4}$, 거의 등방성 |
| 미(Mie, 공명) | $x \sim 1$ | 복잡한 공명, 강한 전방 산란 |
| 기하광학(Geometric optics) | $x \gg 1$ | 광선 추적, $\sigma \approx 2\pi a^2$ |

---

## 2. 레일리 산란

### 2.1 유도 쌍극자

작은 입자($a \ll \lambda$)를 균일한 전기장 안에 놓으면 유도 쌍극자 모멘트가 생긴다:

$$\mathbf{p} = \epsilon_0 \alpha_E \mathbf{E}$$

여기서 $\alpha_E$는 전기 분극률(electric polarizability)이다. 반지름 $a$, 상대 유전율 $\epsilon_r$인 유전체 구에 대해:

$$\alpha_E = 4\pi a^3 \frac{\epsilon_r - 1}{\epsilon_r + 2}$$

인수 $(\epsilon_r - 1)/(\epsilon_r + 2)$는 **클라우시우스-모소티(Clausius-Mossotti)** 인수이다.

### 2.2 산란 단면적

진동하는 쌍극자는 레슨 13에서 유도한 패턴으로 복사한다. 전체 산란 파워로부터 다음을 얻는다:

$$\boxed{\sigma_{\text{Rayleigh}} = \frac{8\pi}{3}\left(\frac{2\pi}{\lambda}\right)^4 a^6 \left|\frac{\epsilon_r - 1}{\epsilon_r + 2}\right|^2 = \frac{128\pi^5}{3} \frac{a^6}{\lambda^4}\left|\frac{\epsilon_r - 1}{\epsilon_r + 2}\right|^2}$$

핵심적인 특징은 다음과 같다:
- **$\lambda^{-4}$ 의존성**: 청색광($\lambda \approx 450$ nm)은 적색광($\lambda \approx 700$ nm)보다 $(700/450)^4 \approx 5.7$배 더 많이 산란된다
- **$a^6$ 의존성**: 산란은 입자 크기에 매우 민감하다

### 2.3 하늘이 파란 이유 (그리고 노을이 붉은 이유)

대기로 들어오는 햇빛은 질소 분자와 산소 분자($a \sim 0.1$ nm, $\lambda \sim 500$ nm이므로 $x \sim 10^{-3}$)를 만난다. $\lambda^{-4}$ 법칙에 의해 청색광은 적색광보다 약 6배 더 산란된다. 하늘을 바라볼 때(태양과 반대 방향), 우리는 주로 산란된 청색광을 보게 된다.

노을 때는 직사광선이 훨씬 두꺼운 대기층을 통과한다. 청색광은 이미 산란되어 사라지고, 투과광은 적색과 주황색이 풍부하게 남는다.

```python
import numpy as np
import matplotlib.pyplot as plt

def rayleigh_cross_section(a, wavelength, eps_r):
    """
    Compute Rayleigh scattering cross section for a small sphere.

    Parameters:
        a          : sphere radius (m)
        wavelength : wavelength of light (m)
        eps_r      : relative permittivity of sphere

    Why Rayleigh: it's the simplest scattering theory and explains
    everyday phenomena like sky color, haze, and why fine particles
    scatter light so differently from large ones.
    """
    k = 2 * np.pi / wavelength
    cm = (eps_r - 1) / (eps_r + 2)
    return (128 * np.pi**5 / 3) * a**6 / wavelength**4 * np.abs(cm)**2

def rayleigh_differential(theta, polarization='unpolarized'):
    """
    Differential cross section pattern for Rayleigh scattering.

    For unpolarized incident light, the scattered intensity at angle theta is:
    dσ/dΩ ∝ (1 + cos²θ) / 2
    """
    if polarization == 'parallel':
        return np.cos(theta)**2
    elif polarization == 'perpendicular':
        return np.ones_like(theta)
    else:  # unpolarized
        return 0.5 * (1 + np.cos(theta)**2)

# Demonstrate λ^{-4} dependence
wavelengths = np.linspace(380, 780, 200) * 1e-9  # visible spectrum
a_N2 = 0.1e-9   # effective radius of N2 molecule
eps_r_air = 1.00029  # relative permittivity of air (approximately)

sigma_rayleigh = rayleigh_cross_section(a_N2, wavelengths, eps_r_air)

# Normalize to green (550 nm)
idx_green = np.argmin(np.abs(wavelengths - 550e-9))
sigma_norm = sigma_rayleigh / sigma_rayleigh[idx_green]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Spectrum of scattered light
ax = axes[0]
# Color mapping for visible spectrum
colors = plt.cm.rainbow(np.linspace(0, 1, len(wavelengths)))
for i in range(len(wavelengths) - 1):
    ax.fill_between(wavelengths[i:i+2] * 1e9, 0, sigma_norm[i:i+2],
                    color=colors[i], alpha=0.8)
ax.plot(wavelengths * 1e9, sigma_norm, 'k-', linewidth=1)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Relative scattering cross section')
ax.set_title('Rayleigh Scattering: $\\sigma \\propto \\lambda^{-4}$')
ax.grid(True, alpha=0.3)

# Annotate blue vs red
ax.annotate('Blue scatters\n~6x more', xy=(450, sigma_norm[np.argmin(np.abs(wavelengths - 450e-9))]),
            xytext=(500, 4), arrowprops=dict(arrowstyle='->', color='blue'),
            fontsize=10, color='blue')

# Angular pattern
ax = axes[1]
theta = np.linspace(0, 2 * np.pi, 500)
r_unpol = rayleigh_differential(theta, 'unpolarized')
r_para = rayleigh_differential(theta, 'parallel')
r_perp = rayleigh_differential(theta, 'perpendicular')

ax_polar = fig.add_subplot(1, 3, 2, projection='polar')
axes[1].remove()
ax_polar.plot(theta, r_unpol, 'k-', linewidth=2, label='Unpolarized')
ax_polar.plot(theta, r_para, 'b--', linewidth=1.5, label='$\\parallel$ polarized')
ax_polar.plot(theta, r_perp, 'r:', linewidth=1.5, label='$\\perp$ polarized')
ax_polar.set_title('Rayleigh Angular Pattern', pad=20)
ax_polar.set_theta_zero_location('E')
ax_polar.legend(loc='lower right', fontsize=9)

# Sky color at different sun positions
ax = axes[2]
# Transmission through atmosphere of thickness L
# I/I_0 = exp(-n σ L) where n is number density
n_air = 2.5e25   # molecules/m^3 at sea level
L_zenith = 8000   # effective atmosphere height (m)

sun_angles = [0, 30, 60, 80, 85]  # degrees from zenith
for angle in sun_angles:
    L = L_zenith / max(np.cos(np.radians(angle)), 0.01)
    # Use approximate cross section for N2 (scaled)
    sigma_approx = 5e-31 * (550e-9 / wavelengths)**4
    transmission = np.exp(-n_air * sigma_approx * L)

    ax.plot(wavelengths * 1e9, transmission, linewidth=1.5,
            label=f'Zenith angle = {angle}°')

ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Transmission')
ax.set_title('Direct Sunlight Through Atmosphere')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("rayleigh_scattering.png", dpi=150)
plt.show()
```

---

## 3. 미 이론

### 3.1 구에 대한 정확한 해

미 이론(1908)은 임의 크기의 균일한 구에 의한 평면파 산란에 대한 정확한 해를 제공한다. 입사장, 산란장, 내부장은 **벡터 구면 조화함수(vector spherical harmonics)**로 전개된다:

$$\mathbf{E}_s = \sum_{n=1}^{\infty} E_n \left(i a_n \mathbf{N}_{e1n}^{(3)} - b_n \mathbf{M}_{o1n}^{(3)}\right)$$

여기서 $\mathbf{M}$과 $\mathbf{N}$은 벡터 구면파 함수이고, $a_n$, $b_n$은 **미 계수(Mie coefficients)**이다:

$$a_n = \frac{m\psi_n(mx)\psi_n'(x) - \psi_n(x)\psi_n'(mx)}{m\psi_n(mx)\xi_n'(x) - \xi_n(x)\psi_n'(mx)}$$

$$b_n = \frac{\psi_n(mx)\psi_n'(x) - m\psi_n(x)\psi_n'(mx)}{\psi_n(mx)\xi_n'(x) - m\xi_n(x)\psi_n'(mx)}$$

여기서 $m = n_{\text{sphere}}/n_{\text{medium}}$은 상대 굴절률, $x = ka$는 크기 매개변수이며, $\psi_n$, $\xi_n$은 리카티-베셀 함수(Riccati-Bessel functions)이다.

### 3.2 미 계수로부터 구한 단면적

$$\sigma_{\text{ext}} = \frac{2\pi}{k^2}\sum_{n=1}^{\infty}(2n+1)\,\text{Re}(a_n + b_n)$$

$$\sigma_s = \frac{2\pi}{k^2}\sum_{n=1}^{\infty}(2n+1)(|a_n|^2 + |b_n|^2)$$

$$\sigma_a = \sigma_{\text{ext}} - \sigma_s$$

### 3.3 효율 인수

기하학적 단면적 $\pi a^2$로 정규화하는 것이 관례이다:

$$Q_{\text{ext}} = \frac{\sigma_{\text{ext}}}{\pi a^2}, \quad Q_s = \frac{\sigma_s}{\pi a^2}, \quad Q_a = \frac{\sigma_a}{\pi a^2}$$

놀랍게도 $Q_{\text{ext}}$는 2를 초과할 수 있다. 이는 입자가 자신의 물리적 크기보다 더 넓은 영역을 "그늘지게 한다"는 뜻이다. 이것이 **소광 역설(extinction paradox)**이며, 입자 주위의 회절 역시 전방 빔에서 에너지를 제거한다는 점을 이해하면 해소된다.

```python
from scipy.special import spherical_jn, spherical_yn

def riccati_bessel_jn(n, x):
    """Riccati-Bessel function psi_n(x) = x * j_n(x)."""
    return x * spherical_jn(n, x)

def riccati_bessel_jn_deriv(n, x):
    """Derivative of psi_n(x)."""
    return spherical_jn(n, x) + x * spherical_jn(n, x, derivative=True)

def riccati_bessel_hn(n, x):
    """Riccati-Bessel function xi_n(x) = x * h_n^(1)(x)."""
    return x * (spherical_jn(n, x) + 1j * spherical_yn(n, x))

def riccati_bessel_hn_deriv(n, x):
    """Derivative of xi_n(x)."""
    jn = spherical_jn(n, x)
    jn_d = spherical_jn(n, x, derivative=True)
    yn = spherical_yn(n, x)
    yn_d = spherical_yn(n, x, derivative=True)
    return (jn + 1j * yn) + x * (jn_d + 1j * yn_d)

def mie_coefficients(m, x, n_max=None):
    """
    Compute Mie scattering coefficients a_n and b_n.

    Parameters:
        m     : relative refractive index (complex)
        x     : size parameter (2*pi*a/lambda)
        n_max : number of terms (default: x + 4*x^(1/3) + 2)

    Why Mie theory: it's the exact solution for spheres, serving as
    the benchmark for approximate methods and the foundation for
    understanding scattering by non-spherical particles.
    """
    if n_max is None:
        n_max = int(x + 4 * x**(1/3) + 2)
    n_max = max(n_max, 3)

    mx = m * x
    a_n = np.zeros(n_max, dtype=complex)
    b_n = np.zeros(n_max, dtype=complex)

    for n in range(1, n_max + 1):
        psi_mx = riccati_bessel_jn(n, mx)
        psi_mx_d = riccati_bessel_jn_deriv(n, mx)
        psi_x = riccati_bessel_jn(n, x)
        psi_x_d = riccati_bessel_jn_deriv(n, x)
        xi_x = riccati_bessel_hn(n, x)
        xi_x_d = riccati_bessel_hn_deriv(n, x)

        a_n[n-1] = (m * psi_mx * psi_x_d - psi_x * psi_mx_d) / \
                    (m * psi_mx * xi_x_d - xi_x * psi_mx_d)
        b_n[n-1] = (psi_mx * psi_x_d - m * psi_x * psi_mx_d) / \
                    (psi_mx * xi_x_d - m * xi_x * psi_mx_d)

    return a_n, b_n

def mie_cross_sections(m, x):
    """Compute Mie scattering, extinction, and absorption efficiencies."""
    a_n, b_n = mie_coefficients(m, x)
    n_terms = len(a_n)
    n_arr = np.arange(1, n_terms + 1)

    Q_ext = (2 / x**2) * np.sum((2 * n_arr + 1) * np.real(a_n + b_n))
    Q_sca = (2 / x**2) * np.sum((2 * n_arr + 1) * (np.abs(a_n)**2 + np.abs(b_n)**2))
    Q_abs = Q_ext - Q_sca

    return Q_ext, Q_sca, Q_abs


def plot_mie_efficiency():
    """
    Plot Mie efficiency factors vs size parameter for different
    refractive indices.
    """
    x_range = np.linspace(0.01, 30, 500)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Different refractive indices
    indices = [
        (1.33 + 0j, 'Water droplet ($n$ = 1.33)'),
        (1.50 + 0j, 'Glass sphere ($n$ = 1.50)'),
        (1.50 + 0.1j, 'Absorbing sphere ($n$ = 1.50 + 0.1i)'),
        (2.00 + 1.00j, 'Metallic sphere ($n$ = 2.0 + 1.0i)')
    ]

    for ax, (m, label) in zip(axes.flat, indices):
        Q_ext_arr = np.zeros(len(x_range))
        Q_sca_arr = np.zeros(len(x_range))
        Q_abs_arr = np.zeros(len(x_range))

        for i, x in enumerate(x_range):
            if x < 0.05:
                continue
            try:
                Q_ext_arr[i], Q_sca_arr[i], Q_abs_arr[i] = mie_cross_sections(m, x)
            except (ValueError, ZeroDivisionError):
                pass

        ax.plot(x_range, Q_ext_arr, 'k-', linewidth=2, label='$Q_{\\mathrm{ext}}$')
        ax.plot(x_range, Q_sca_arr, 'b-', linewidth=1.5, label='$Q_{\\mathrm{sca}}$')
        if np.any(Q_abs_arr > 0.01):
            ax.plot(x_range, Q_abs_arr, 'r--', linewidth=1.5, label='$Q_{\\mathrm{abs}}$')
        ax.axhline(y=2, color='gray', linestyle=':', alpha=0.5, label='$Q_{\\mathrm{ext}} = 2$ (limit)')
        ax.set_xlabel('Size parameter $x = 2\\pi a / \\lambda$')
        ax.set_ylabel('Efficiency $Q$')
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(5, 1.1 * np.max(Q_ext_arr)))

    plt.suptitle('Mie Scattering Efficiency vs Size Parameter', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("mie_efficiency.png", dpi=150)
    plt.show()

plot_mie_efficiency()
```

---

## 4. 본 근사

### 4.1 약한 산란

산란체가 "약할" 때(굴절률 대비가 작거나 얇을 때), 산란체 내부의 전체장을 입사장으로 근사할 수 있다. 이것이 **본 근사(Born approximation)**이다:

$$\mathbf{E}_s(\mathbf{r}) \approx \frac{k^2}{4\pi}\int \Delta\epsilon_r(\mathbf{r}') \, \mathbf{E}_i(\mathbf{r}') \frac{e^{ik|\mathbf{r}-\mathbf{r}'|}}{|\mathbf{r}-\mathbf{r}'|} \, d^3r'$$

여기서 $\Delta\epsilon_r = \epsilon_r - 1$은 유전율 대비(dielectric contrast)이다.

### 4.2 원거리장 본 근사

원거리장에서 $|\mathbf{r} - \mathbf{r}'| \approx r - \hat{r}\cdot\mathbf{r}'$이므로:

$$f(\theta, \phi) \propto \int \Delta\epsilon_r(\mathbf{r}') \, e^{-i\mathbf{q}\cdot\mathbf{r}'} \, d^3r'$$

여기서 $\mathbf{q} = \mathbf{k}_s - \mathbf{k}_i$는 **산란 벡터(scattering vector)**이다(탄성 산란의 경우 $|\mathbf{q}| = 2k\sin(\theta/2)$). 산란 진폭은 $\mathbf{q}$에서 평가한 유전율 대비의 **푸리에 변환(Fourier transform)**이다.

이 강력한 결과는 산란 측정을 산란체의 공간 구조와 연결한다 — X선 결정학(X-ray crystallography), 레이더 영상, 역 산란의 기반이다.

### 4.3 유효 범위

본 근사는 다음 조건에서 유효하다:

$$|\Delta\epsilon_r| \cdot k \cdot a \ll 1$$

즉, 산란체를 통과하며 누적되는 위상 변이가 작아야 한다.

```python
def born_scattering_cross_section(k, a, delta_eps):
    """
    Compute the Born approximation scattering cross section for
    a homogeneous dielectric sphere.

    The result is:
    σ = (k^4 / 6π) * |delta_eps|^2 * V^2 * [3(sin(qa) - qa*cos(qa))/(qa)^3]^2

    averaged over angles, where V = 4πa^3/3.

    Why Born approximation: it gives analytical insight into scattering
    by arbitrary shapes, since the cross section is essentially the
    Fourier transform of the object's shape.
    """
    theta = np.linspace(0.001, np.pi, 500)
    q = 2 * k * np.sin(theta / 2)

    # Form factor for a sphere: F(q) = 3[sin(qa) - qa*cos(qa)] / (qa)^3
    qa = q * a
    F = np.where(qa < 0.01, 1.0 - qa**2/10,
                  3 * (np.sin(qa) - qa * np.cos(qa)) / qa**3)

    V = 4 * np.pi * a**3 / 3

    # Differential cross section
    dsigma = (k**4 / (4 * np.pi)**2) * np.abs(delta_eps)**2 * V**2 * F**2

    # Total cross section (integrate over solid angle)
    sigma_total = 2 * np.pi * np.trapz(dsigma * np.sin(theta), theta)

    return theta, dsigma, sigma_total


def compare_born_vs_mie():
    """Compare Born approximation with exact Mie theory."""
    # Weak scatterer: n = 1.05 (delta_eps = n^2 - 1 ≈ 0.1025)
    n_sphere = 1.05
    m = n_sphere
    delta_eps = n_sphere**2 - 1

    x_values = np.linspace(0.1, 10, 50)
    wavelength = 500e-9  # 500 nm

    Q_mie_arr = np.zeros(len(x_values))
    Q_born_arr = np.zeros(len(x_values))

    for i, x in enumerate(x_values):
        a = x * wavelength / (2 * np.pi)
        k = 2 * np.pi / wavelength

        # Mie
        _, Q_mie_arr[i], _ = mie_cross_sections(m, x)

        # Born
        _, _, sigma_born = born_scattering_cross_section(k, a, delta_eps)
        Q_born_arr[i] = sigma_born / (np.pi * a**2)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_values, Q_mie_arr, 'b-', linewidth=2, label='Mie (exact)')
    ax.plot(x_values, Q_born_arr, 'r--', linewidth=2, label='Born approximation')
    ax.set_xlabel('Size parameter $x$')
    ax.set_ylabel('Scattering efficiency $Q_{\\mathrm{sca}}$')
    ax.set_title(f'Born vs Mie: Weak Scatterer ($n$ = {n_sphere})')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Annotate validity region
    ax.axvspan(0, 2, alpha=0.1, color='green')
    ax.text(1, ax.get_ylim()[1] * 0.9, 'Born valid', fontsize=11,
            ha='center', color='green')

    plt.tight_layout()
    plt.savefig("born_vs_mie.png", dpi=150)
    plt.show()

compare_born_vs_mie()
```

---

## 5. 광학 정리

### 5.1 내용

**광학 정리(optical theorem)**는 전체 소광 단면적을 전방 산란 진폭과 연결한다:

$$\boxed{\sigma_{\text{ext}} = \frac{4\pi}{k}\,\text{Im}[f(0)]}$$

여기서 $f(0)$는 정확한 전방 방향($\theta = 0$)의 산란 진폭이다.

### 5.2 물리적 의미

전방 산란파는 입사파와 상쇄 간섭을 일으켜 산란체 뒤에 "그림자"를 만든다. 입사 빔에서 제거되는 파워(소광)는 전적으로 이 전방 간섭에 의해 결정되며 — 산란 파워가 각도에 따라 어떻게 분포하는지와는 무관하다.

### 5.3 결론

- 소광은 항상 흡수 이상이다: $\sigma_{\text{ext}} \geq \sigma_a$
- 순수하게 산란만 하는 입자(흡수 없음)의 경우, $\sigma_{\text{ext}} = \sigma_s$
- 큰 입자 극한($x \to \infty$)에서 $\sigma_{\text{ext}} \to 2\pi a^2$ — 기하학적 단면적의 두 배(소광 역설)

---

## 6. 각도 산란 패턴

### 6.1 작은 입자 (레일리 영역)

$x \ll 1$이면 산란은 전방과 후방 사이에서 거의 대칭적이다:

$$\frac{d\sigma}{d\Omega} \propto (1 + \cos^2\theta)$$

### 6.2 큰 입자 (미 영역)

$x$가 증가함에 따라 산란은 점점 전방으로 집중된다. 전방 로브(forward lobe)는 $\Delta\theta \sim 1/x \sim \lambda/(2\pi a)$로 좁아지며, 큰 각도에서 복잡한 간섭 줄무늬가 나타난다.

```python
def mie_scattering_pattern(m, x, theta):
    """
    Compute Mie angular scattering pattern (intensity functions S1, S2).

    Why angular patterns: they are directly measurable in experiments
    (nephelometers, goniometers) and provide information about
    particle size, shape, and refractive index.
    """
    from scipy.special import lpmv

    a_n, b_n = mie_coefficients(m, x)
    n_max = len(a_n)
    cos_theta = np.cos(theta)

    S1 = np.zeros(len(theta), dtype=complex)
    S2 = np.zeros(len(theta), dtype=complex)

    for n in range(1, n_max + 1):
        # Angular functions pi_n and tau_n
        # pi_n = P_n^1 / sin(theta)
        # tau_n = d/d(theta) P_n^1

        P1 = lpmv(1, n, cos_theta)
        sin_theta = np.sin(theta)
        sin_theta = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)
        pi_n = P1 / sin_theta
        # tau_n requires numerical derivative or recursion
        # Simple finite difference for tau_n
        dtheta = 1e-6
        P1_plus = lpmv(1, n, np.cos(theta + dtheta))
        tau_n = -(P1_plus - P1) / dtheta

        prefactor = (2 * n + 1) / (n * (n + 1))
        S1 += prefactor * (a_n[n-1] * pi_n + b_n[n-1] * tau_n)
        S2 += prefactor * (a_n[n-1] * tau_n + b_n[n-1] * pi_n)

    return np.abs(S1)**2, np.abs(S2)**2


def plot_angular_patterns():
    """Plot angular scattering patterns for different size parameters."""
    theta = np.linspace(0.01, np.pi, 500)
    m = 1.33  # water droplet

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    x_values = [0.1, 1.0, 5.0, 20.0]

    for ax, x in zip(axes.flat, x_values):
        try:
            S1, S2 = mie_scattering_pattern(m, x, theta)
            S_unpol = 0.5 * (S1 + S2)

            ax_polar = fig.add_subplot(2, 2, list(axes.flat).index(ax) + 1,
                                       projection='polar')
            ax.remove()

            # Use log scale for pattern
            pattern_db = 10 * np.log10(S_unpol / S_unpol.max() + 1e-10)
            pattern_plot = np.clip(pattern_db + 40, 0, 40)  # shift for plotting

            ax_polar.plot(theta, pattern_plot, 'b-', linewidth=1.5)
            ax_polar.plot(-theta + 2*np.pi, pattern_plot, 'b-', linewidth=1.5)
            ax_polar.set_title(f'$x$ = {x} ($a/\\lambda$ = {x/(2*np.pi):.3f})',
                              pad=15)
            ax_polar.set_theta_zero_location('E')

        except Exception:
            ax.text(0.5, 0.5, f'x = {x}\n(computation error)',
                    ha='center', va='center', transform=ax.transAxes)

    plt.suptitle(f'Mie Scattering Angular Pattern (water, $n$ = {m})',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("mie_angular_patterns.png", dpi=150)
    plt.show()

plot_angular_patterns()
```

---

## 7. 응용

### 7.1 레이더

레이더(radar)는 마이크로파 펄스를 송신하고 후방 산란 신호($\theta = \pi$)를 감지한다. **레이더 단면적(radar cross section, RCS)**은 다음과 같다:

$$\sigma_{\text{RCS}} = 4\pi r^2 \frac{|\mathbf{E}_s|^2}{|\mathbf{E}_i|^2}\bigg|_{\theta=\pi}$$

스텔스 항공기는 기하학적 구조(경사진 표면)와 재료(레이더 흡수 코팅)를 통해 RCS를 최소화하도록 설계된다.

### 7.2 라이다와 대기 원격 탐사

라이다(lidar)는 짧은 레이저 펄스를 사용하여 대기를 프로파일링한다. 후방 산란광은 다음에 대한 정보를 담고 있다:
- **에어로졸 농도** (입자의 미 산란)
- **분자 밀도** (공기 분자의 레일리 산란)
- **풍속** (후방 산란광의 도플러 편이)
- **온도** (레일리 선의 넓어짐)

### 7.3 의료 영상

광학 간섭 단층 촬영(optical coherence tomography, OCT)은 근적외선 광 산란을 이용하여 마이크로미터 분해능으로 조직을 영상화한다. 전방 산란 영역($x \sim 1-10$, 세포 크기)이 영상화 깊이와 대비를 결정한다.

### 7.4 나노입자 특성 분석

동적 광 산란(dynamic light scattering, DLS)은 산란광 세기의 시간적 변동을 분석하여 용액 내 나노입자의 크기 분포를 측정한다. 입자는 크기에 반비례하는 속도로 확산(스토크스-아인슈타인 관계)하며, 이로 인한 세기 변동은 자기상관 분석으로 해석된다.

---

## 요약

| 개념 | 핵심 공식 | 물리적 의미 |
|---------|-------------|------------------|
| 크기 매개변수 | $x = 2\pi a / \lambda$ | 산란 영역 결정 |
| 레일리 단면적 | $\sigma \propto a^6 / \lambda^4$ | 작은 입자; 파란 하늘 |
| 미 계수 | 리카티-베셀 함수로부터의 $a_n, b_n$ | 구에 대한 정확한 해 |
| 소광 효율 | $Q_{\text{ext}} = \sigma_{\text{ext}} / \pi a^2$ | 2 초과 가능 (역설) |
| 본 근사 | $f \propto \text{FT}[\Delta\epsilon_r]$ at $\mathbf{q}$ | 약한 산란체; 푸리에 관계 |
| 광학 정리 | $\sigma_{\text{ext}} = (4\pi/k)\,\text{Im}[f(0)]$ | 전방 산란이 소광 결정 |
| 전방 집중 | $\Delta\theta \sim \lambda / (2\pi a)$ | 큰 입자는 전방으로 산란 |

---

## 연습 문제

### 연습 1: 노을 시뮬레이션
태양 천정각의 함수로 대기를 통과하는 햇빛의 투과율을 모델링하라. N$_2$와 O$_2$에 대한 레일리 산란 단면적을 사용하고, 대기를 지수 밀도 프로파일($n(h) = n_0 e^{-h/H}$, $H = 8.5$ km)로 모델링하라. 천정각 0, 45, 70, 85, 90도에 대한 투과 스펙트럼을 그려라. 각 각도에서 투과광의 "색온도"를 계산하라.

### 연습 2: 미 공명
$x = 0$부터 $x = 50$까지 크기 매개변수의 함수로 유리구($n = 1.5$)의 $Q_{\text{ext}}$를 그려라. 공명 피크를 확인하고 미 계수 $a_n$, $b_n$의 관점에서 그 기원을 설명하라. 각 공명에서 어느 다극자 차수 $n$이 지배적인가?

### 연습 3: 구름의 불투명도
구름 물방울의 전형적인 반지름은 10 $\mu$m이고 농도는 $\sim 300$ cm$^{-3}$이다. (a) 미 이론($n = 1.33$)을 사용하여 가시광선($\lambda = 550$ nm)에 대한 $Q_{\text{ext}}$를 계산하라. (b) 소광 계수 $\alpha = n_{\text{droplet}} \sigma_{\text{ext}}$를 계산하라. (c) 두께 1 km인 구름의 광학 깊이는 얼마인가? 이것이 구름이 시각적으로 불투명한 것과 일치하는가?

### 연습 4: 본 근사의 유효 범위
$n = 1.05$인 구에 대해, $x = 0$에서 $x = 20$까지 본 근사와 미 이론의 단면적을 비교하라. 본 근사의 오차가 10%를 초과하는 $x$는 얼마인가? $n = 1.2$와 $n = 1.5$에 대해 반복하라. 경험 법칙 $|\Delta\epsilon_r| \cdot x \ll 1$을 검증하라.

### 연습 5: 레이더 단면적
반지름 1 m인 금속 구가 10 GHz 레이더에 의해 조명된다. (a) 크기 매개변수를 계산하라. (b) $n = 10 + 10i$(근사 금속)로 미 이론을 사용하여 $Q_{\text{ext}}$를 계산하라. (c) RCS를 추정하라. (d) 레이더 방정식 $P_r = P_t G^2 \lambda^2 \sigma / (4\pi)^3 R^4$에서 $P_t = 1$ MW, $G = 40$ dBi, $R = 100$ km일 때 $P_r$을 계산하라.

---

[← 이전: 16. 전산 전자기학](16_Computational_Electrodynamics.md) | [다음: 18. 응용 — 플라즈모닉스와 메타물질 →](18_Plasmonics_and_Metamaterials.md)
