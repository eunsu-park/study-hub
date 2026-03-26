# 11. 반사와 굴절

[← 이전: 10. 물질에서의 전자기파](10_EM_Waves_Matter.md) | [다음: 12. 도파관과 공동 →](12_Waveguides_and_Cavities.md)

## 학습 목표

1. 맥스웰 방정식의 경계 조건으로부터 스넬의 법칙(Snell's law) 유도
2. s-편광과 p-편광에 대한 프레넬 방정식(Fresnel equations) 유도
3. 반사율(reflectance)과 투과율(transmittance)의 계산 및 에너지 보존 검증
4. 브루스터 각(Brewster's angle)과 편광 광학 응용 이해
5. 전반사(total internal reflection)와 에바네선트파(evanescent wave)의 물리 분석
6. 박막 간섭을 이용한 단층 반사 방지 코팅 설계
7. Python을 이용한 프레넬 계수의 수치 계산 구현

빛이 두 매질의 경계면에 부딪히면 파동의 일부는 반사되고 일부는 투과된다. 정확한 비율은 입사각, 편광 상태, 두 매질의 굴절률에 따라 달라진다. 프레넬 방정식으로 정량적으로 기술되는 이 현상들은 안경 코팅과 광섬유 커플러에서부터 수면의 눈부심과 다이아몬드의 반짝임까지 모든 것의 기초가 된다. 이 레슨에서는 맥스웰의 경계 조건으로부터 직접 프레넬 방정식을 유도하고, 편광 선글라스가 작동하는 원리와 박막 코팅으로 반사를 없애는 방법에 대한 통찰을 얻는다.

> **비유**: 두께가 달라지는 이음부에 부딪히는 밧줄 파동을 상상해 보자. 이음부에서 파동의 일부는 되돌아 튀어나오고, 일부는 더 두꺼운 밧줄 쪽으로(다른 속도로) 계속 진행한다. 두 밧줄 구간의 "임피던스 불일치(impedance mismatch)"가 반사와 투과에 얼마나 많은 에너지가 배분되는지를 결정한다. 전자기파에서는 굴절률이 밧줄의 두께 역할을 하며, 입사각이 문제에 기하학적 풍부함을 더한다.

---

## 1. 경계면에서의 경계 조건

### 1.1 맥스웰의 경계 조건

자유 표면 전하나 전류가 없는 매질 1과 매질 2 사이의 평면 경계에서 맥스웰 방정식은 다음을 요구한다:

$$\epsilon_1 E_{1\perp} = \epsilon_2 E_{2\perp} \quad \text{(법선 방향 } D \text{ 연속)}$$

$$B_{1\perp} = B_{2\perp} \quad \text{(법선 방향 } B \text{ 연속)}$$

$$E_{1\parallel} = E_{2\parallel} \quad \text{(접선 방향 } E \text{ 연속)}$$

$$\frac{1}{\mu_1} B_{1\parallel} = \frac{1}{\mu_2} B_{2\parallel} \quad \text{(접선 방향 } H \text{ 연속)}$$

이 조건들은 경계면 위의 모든 점과 모든 시각에서 성립해야 한다.

### 1.2 위상 정합 — 스넬의 법칙 유도

$z = 0$인 평면 경계에 $\mathbf{E}_I = \mathbf{E}_0 \, e^{i(\mathbf{k}_I \cdot \mathbf{r} - \omega t)}$인 평면파가 입사한다고 하자. 반사파의 파동 벡터를 $\mathbf{k}_R$, 투과파의 파동 벡터를 $\mathbf{k}_T$라 하자.

경계 조건은 경계면($z = 0$) 위의 **모든** $x$와 $t$에 대해 성립해야 한다. 이는 세 파동의 위상이 $z = 0$에서 모두 일치해야 함을 요구한다:

$$\mathbf{k}_I \cdot \mathbf{r}\big|_{z=0} = \mathbf{k}_R \cdot \mathbf{r}\big|_{z=0} = \mathbf{k}_T \cdot \mathbf{r}\big|_{z=0}$$

이로부터 두 결과가 나온다:

**반사 법칙**: $\theta_R = \theta_I$ (입사각과 반사각이 같다)

**스넬의 법칙**:

$$\boxed{n_1 \sin\theta_I = n_2 \sin\theta_T}$$

여기서 $\theta_I$는 입사각(표면 법선으로부터 측정), $\theta_T$는 투과각(굴절각)이다.

스넬의 법칙은 독립적인 공준이 아니라 맥스웰의 경계 조건이 충족 가능하도록 요구한 결과로 자연스럽게 등장한다는 점에 주목하라.

---

## 2. 프레넬 방정식

### 2.1 편광 규약

입사 전기장을 두 편광 성분으로 분해한다:

- **s-편광(s-polarization)** (독일어 *senkrecht* = 수직에서 유래): $\mathbf{E}$가 입사면에 수직
- **p-편광(p-polarization)** (평행): $\mathbf{E}$가 입사면 내에 있음

입사면(plane of incidence)은 입사 파동 벡터 $\mathbf{k}_I$와 표면 법선 $\hat{n}$으로 정의된다.

### 2.2 s-편광(TE)

s-편광의 경우, 전기장은 입사면 밖을 가리킨다(예: $\hat{y}$ 방향). 접선 방향 $E$와 $H$의 경계 조건을 적용하면:

$$E_I + E_R = E_T$$

$$\frac{1}{\mu_1}(E_I - E_R)\cos\theta_I = \frac{1}{\mu_2} E_T \cos\theta_T$$

비자성 재료($\mu_1 = \mu_2 = \mu_0$)에 대해, s-편광의 **프레넬 반사 계수**는:

$$\boxed{r_s = \frac{n_1 \cos\theta_I - n_2 \cos\theta_T}{n_1 \cos\theta_I + n_2 \cos\theta_T}}$$

**투과 계수**는:

$$\boxed{t_s = \frac{2 n_1 \cos\theta_I}{n_1 \cos\theta_I + n_2 \cos\theta_T}}$$

### 2.3 p-편광(TM)

p-편광의 경우, 자기장이 입사면에 수직이다. 경계 조건으로부터:

$$\boxed{r_p = \frac{n_2 \cos\theta_I - n_1 \cos\theta_T}{n_2 \cos\theta_I + n_1 \cos\theta_T}}$$

$$\boxed{t_p = \frac{2 n_1 \cos\theta_I}{n_2 \cos\theta_I + n_1 \cos\theta_T}}$$

### 2.4 수직 입사

$\theta_I = 0$(수직 입사)에서는 두 편광 모두 같은 결과를 준다:

$$r = \frac{n_1 - n_2}{n_1 + n_2}, \quad t = \frac{2n_1}{n_1 + n_2}$$

공기-유리 경계면($n_1 = 1.0$, $n_2 = 1.5$)의 경우: $r = -0.2$로, 세기의 4%가 반사된다. 음의 부호는 $\pi$ 위상 이동을 나타낸다.

```python
import numpy as np
import matplotlib.pyplot as plt

def fresnel_coefficients(n1, n2, theta_i):
    """
    Compute Fresnel reflection and transmission coefficients.

    Parameters:
        n1, n2   : refractive indices (can be complex for absorbing media)
        theta_i  : angle of incidence (radians), array

    Returns:
        r_s, r_p, t_s, t_p (complex amplitude coefficients)

    Why complex: the coefficients carry phase information, which is
    crucial for thin-film interference calculations and for understanding
    evanescent waves in total internal reflection.
    """
    # Snell's law to find transmission angle
    # Use complex sqrt to handle total internal reflection gracefully
    cos_i = np.cos(theta_i)
    sin_i = np.sin(theta_i)
    cos_t = np.sqrt(1 - (n1 / n2 * sin_i)**2 + 0j)

    # s-polarization (TE)
    r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    t_s = 2 * n1 * cos_i / (n1 * cos_i + n2 * cos_t)

    # p-polarization (TM)
    r_p = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    t_p = 2 * n1 * cos_i / (n2 * cos_i + n1 * cos_t)

    return r_s, r_p, t_s, t_p

# Air to glass (external reflection)
n1, n2 = 1.0, 1.5
theta = np.linspace(0, np.pi / 2 - 0.001, 500)

r_s, r_p, t_s, t_p = fresnel_coefficients(n1, n2, theta)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Amplitude coefficients
axes[0].plot(np.degrees(theta), r_s.real, 'b-', linewidth=2, label='$r_s$')
axes[0].plot(np.degrees(theta), r_p.real, 'r-', linewidth=2, label='$r_p$')
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Angle of incidence (degrees)')
axes[0].set_ylabel('Amplitude coefficient')
axes[0].set_title(f'Fresnel Coefficients: Air ($n_1$={n1}) → Glass ($n_2$={n2})')
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)

# Reflectance and Transmittance (intensity)
R_s = np.abs(r_s)**2
R_p = np.abs(r_p)**2

axes[1].plot(np.degrees(theta), R_s, 'b-', linewidth=2, label='$R_s$')
axes[1].plot(np.degrees(theta), R_p, 'r-', linewidth=2, label='$R_p$')
axes[1].plot(np.degrees(theta), 0.5 * (R_s + R_p), 'k--', linewidth=1.5,
             label='Unpolarized')

# Mark Brewster angle
theta_B = np.arctan(n2 / n1)
axes[1].axvline(x=np.degrees(theta_B), color='green', linestyle=':',
                label=f"Brewster's angle = {np.degrees(theta_B):.1f}°")

axes[1].set_xlabel('Angle of incidence (degrees)')
axes[1].set_ylabel('Reflectance')
axes[1].set_title('Reflectance vs. Angle')
axes[1].set_ylim(0, 1)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fresnel_air_glass.png", dpi=150)
plt.show()
```

---

## 3. 반사율과 투과율

### 3.1 정의

**반사율(reflectance)** $R$과 **투과율(transmittance)** $T$는 입사 *전력* 중 반사되고 투과되는 비율을 나타낸다:

$$R_s = |r_s|^2, \quad R_p = |r_p|^2$$

$$T_s = \frac{n_2 \cos\theta_T}{n_1 \cos\theta_I} |t_s|^2, \quad T_p = \frac{n_2 \cos\theta_T}{n_1 \cos\theta_I} |t_p|^2$$

인수 $n_2 \cos\theta_T / (n_1 \cos\theta_I)$는 굴절 시 빔 단면적과 파동 속도의 변화를 보정한다.

### 3.2 에너지 보존

에너지 보존은 다음을 요구한다:

$$R + T = 1$$

이는 각 편광에 대해 별도로 성립한다. 프레넬 계산의 유용한 검증 수단이다.

```python
def verify_energy_conservation(n1, n2, theta_i):
    """
    Verify R + T = 1 for both polarizations.

    Why verify: this catch numerical errors and ensures that
    the Fresnel equations are implemented correctly — a common
    source of sign and factor-of-2 mistakes.
    """
    r_s, r_p, t_s, t_p = fresnel_coefficients(n1, n2, theta_i)

    cos_i = np.cos(theta_i)
    sin_t = n1 / n2 * np.sin(theta_i)
    cos_t = np.sqrt(1 - sin_t**2 + 0j)

    R_s = np.abs(r_s)**2
    R_p = np.abs(r_p)**2

    # Transmittance includes the beam area and velocity correction
    T_s = (n2 * cos_t.real) / (n1 * cos_i) * np.abs(t_s)**2
    T_p = (n2 * cos_t.real) / (n1 * cos_i) * np.abs(t_p)**2

    print(f"At theta_i = {np.degrees(theta_i):.1f}°:")
    print(f"  R_s + T_s = {(R_s + T_s).real:.6f}")
    print(f"  R_p + T_p = {(R_p + T_p).real:.6f}")

# Check at several angles
for angle_deg in [0, 30, 56.3, 80]:
    verify_energy_conservation(1.0, 1.5, np.radians(angle_deg))
```

---

## 4. 브루스터 각

### 4.1 유도

브루스터 각 $\theta_B$에서 반사된 p-편광이 사라진다: $r_p = 0$. 프레넬 방정식으로부터:

$$n_2 \cos\theta_B = n_1 \cos\theta_T$$

스넬의 법칙과 결합하면:

$$\boxed{\tan\theta_B = \frac{n_2}{n_1}}$$

브루스터 각에서 $\theta_I + \theta_T = 90°$이 성립한다. 반사광과 굴절광이 서로 수직이다.

### 4.2 물리적 해석

반사파는 두 번째 매질에서 유도된 진동 쌍극자에 의해 생성된다. 브루스터 각에서 굴절파는 가상의 반사파 방향에 수직으로 진행한다. 쌍극자는 자신의 진동 축 방향으로 복사하지 않으므로, p-편광 쌍극자는 반사 방향으로 복사를 방출할 수 없다. 따라서 s-편광 빛만 반사된다.

### 4.3 응용

- **편광 선글라스(polarized sunglasses)**: 수평 표면(도로, 수면)에서의 눈부심은 브루스터 각 근방에서 주로 s-편광이다. 편광 렌즈는 s-편광 빛을 차단한다.
- **브루스터 창(Brewster windows)**: 레이저 공동(cavity)은 브루스터 각으로 기울어진 창을 사용하여 p-편광에 대한 반사 손실을 제거하고, 레이저가 선형 편광을 방출하도록 강제한다.
- **유사 브루스터 각(pseudo-Brewster angle)**: 흡수 매질(복소수 $n_2$)의 경우 $R_p$는 최솟값에 도달하지만 0이 되지는 않는다. $R_p$ 최솟값에서의 각도가 유사 브루스터 각이다.

---

## 5. 전반사

### 5.1 임계각

빛이 밀한 매질에서 소한 매질로 진행할 때($n_1 > n_2$), 스넬의 법칙은 다음을 준다:

$$\sin\theta_T = \frac{n_1}{n_2} \sin\theta_I$$

$\sin\theta_T = 1$(즉, $\theta_T = 90°$)이 될 때 **임계각(critical angle)**에 도달한다:

$$\boxed{\theta_c = \arcsin\left(\frac{n_2}{n_1}\right)}$$

$\theta_I > \theta_c$인 경우, 스넬의 법칙은 $\theta_T$에 대한 실수 해를 갖지 않으며 파동은 **전반사(totally internally reflected)**된다.

### 5.2 전반사 시 무슨 일이 일어나는가?

"전(全)" 반사라는 이름에도 불구하고 전기장이 경계면에서 갑자기 멈추지는 않는다. 대신:

- $|r_s| = |r_p| = 1$ — 모든 입사 전력이 반사된다
- 투과 계수 $t$는 0이 아니며, 매질 2에 **에바네선트파(evanescent wave)**가 존재한다
- 프레넬 계수가 복소수가 되어 반사 시 **위상 이동(phase shift)**이 나타난다

### 5.3 에바네선트파

임계각을 넘으면 $\cos\theta_T$가 순허수가 된다:

$$\cos\theta_T = i\sqrt{\left(\frac{n_1}{n_2}\right)^2 \sin^2\theta_I - 1}$$

투과 전기장은 다음과 같이 된다:

$$\mathbf{E}_T \propto e^{ik_x x} e^{-\kappa z}$$

여기서 $\kappa = k_2\sqrt{(n_1/n_2)^2\sin^2\theta_I - 1}$은 감쇠 상수이다. 이것이 **에바네선트파**: 경계면을 따라($x$ 방향으로) 전파하지만 경계면에서 멀어질수록($z$ 방향) 지수적으로 감쇠한다.

침투 깊이(penetration depth)는:

$$d_p = \frac{1}{\kappa} = \frac{\lambda_2}{2\pi\sqrt{(n_1/n_2)^2\sin^2\theta_I - 1}}$$

일반적으로 $d_p \sim \lambda$ — 에바네선트 전기장은 매질 2 속으로 약 한 파장 정도 침투한다.

> **비유**: 에바네선트파는 벽을 통해 새어 나오는 소리와 같다. 벽이 충분히 얇다면(침투 깊이와 비슷한 수준) 일부 에너지가 반대편으로 터널링할 수 있다. 이것이 **좌절 전반사(frustrated total internal reflection)**로, 양자 터널링의 광학적 유사체다.

```python
def plot_evanescent_wave(n1, n2, theta_i_deg, wavelength=500e-9):
    """
    Visualize the evanescent wave field beyond the critical angle.

    Why this matters: evanescent waves are the basis of
    TIRF microscopy, fiber optic sensors, and near-field optics.
    """
    theta_c = np.degrees(np.arcsin(n2 / n1))
    theta_i = np.radians(theta_i_deg)
    k2 = 2 * np.pi * n2 / wavelength

    # Decay constant in medium 2
    kappa = k2 * np.sqrt((n1 / n2)**2 * np.sin(theta_i)**2 - 1)
    # Lateral wave number
    kx = k2 * (n1 / n2) * np.sin(theta_i)

    # Penetration depth
    d_p = 1.0 / kappa

    # Create spatial grid
    x = np.linspace(0, 3 * wavelength, 300)
    z = np.linspace(-2 * wavelength, 3 * wavelength, 400)
    X, Z = np.meshgrid(x, z)

    # Field in medium 1 (z < 0): incident + reflected
    # For simplicity, show only the evanescent field in medium 2
    E = np.zeros_like(X, dtype=complex)

    # Medium 2 (z > 0): evanescent wave
    mask2 = Z > 0
    E[mask2] = np.exp(1j * kx * X[mask2]) * np.exp(-kappa * Z[mask2])

    # Medium 1 (z < 0): simple standing wave pattern
    k1 = 2 * np.pi * n1 / wavelength
    kz1 = k1 * np.cos(theta_i)
    mask1 = Z <= 0
    E[mask1] = (np.exp(1j * (kx * X[mask1] + kz1 * Z[mask1])) +
                np.exp(1j * (kx * X[mask1] - kz1 * Z[mask1])))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Real part of E-field
    im = axes[0].pcolormesh(X * 1e9, Z * 1e9, E.real, cmap='RdBu_r',
                             shading='auto', vmin=-2, vmax=2)
    axes[0].axhline(y=0, color='white', linewidth=2)
    axes[0].set_xlabel('x (nm)')
    axes[0].set_ylabel('z (nm)')
    axes[0].set_title(f'E-field (TIR at {theta_i_deg}°, $\\theta_c$ = {theta_c:.1f}°)')
    plt.colorbar(im, ax=axes[0], label='Re(E)')

    # Intensity vs z at x=0
    z_line = np.linspace(-wavelength, 3 * wavelength, 500)
    I_med1 = np.ones_like(z_line[z_line <= 0])
    I_med2 = np.exp(-2 * kappa * z_line[z_line > 0])

    axes[1].plot(z_line[z_line <= 0] * 1e9, I_med1, 'b-', linewidth=2,
                 label='Medium 1')
    axes[1].plot(z_line[z_line > 0] * 1e9, I_med2, 'r-', linewidth=2,
                 label='Medium 2 (evanescent)')
    axes[1].axvline(x=0, color='black', linewidth=2, label='Interface')
    axes[1].axhline(y=np.exp(-2), color='gray', linestyle='--', alpha=0.5,
                     label=f'$1/e^2$ depth = {d_p*1e9:.0f} nm')
    axes[1].set_xlabel('z (nm)')
    axes[1].set_ylabel('Intensity (normalized)')
    axes[1].set_title('Evanescent Field Decay')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("evanescent_wave.png", dpi=150)
    plt.show()

    print(f"Critical angle: {theta_c:.1f}°")
    print(f"Penetration depth: {d_p*1e9:.1f} nm")

# Glass to air, beyond critical angle
plot_evanescent_wave(n1=1.5, n2=1.0, theta_i_deg=45, wavelength=500e-9)
```

---

## 6. 반사 방지 코팅

### 6.1 문제

단일 공기-유리 경계면은 입사광의 약 4%를 반사한다. 10개의 면을 가진 카메라 렌즈는 반사로 인해 빛의 거의 34%를 잃는다(여러 반사에 의한 고스트 이미지도 생긴다). 반사 방지(AR, anti-reflection) 코팅은 이러한 손실을 극적으로 줄인다.

### 6.2 단층 4분의 1 파장 코팅

가장 간단한 AR 코팅은 굴절률 $n_s$인 기판 위에 두께 $d$, 굴절률 $n_c$인 박막을 입히는 것이다. 박막의 위아래 면에서 반사된 빛 사이의 상쇄 간섭이 반사파를 소거하려면:

**두께 조건** (4분의 1 파장):

$$d = \frac{\lambda}{4 n_c}$$

**굴절률 정합 조건** (수직 입사 시 반사율 0):

$$n_c = \sqrt{n_1 \cdot n_s}$$

공기($n_1 = 1$) 위의 유리($n_s = 1.5$)의 경우: $n_c = \sqrt{1.5} \approx 1.225$. MgF$_2$($n = 1.38$)가 가장 가까운 일반 재료로, 완벽하지는 않지만 반사율을 4%에서 약 1.3%로 줄인다.

### 6.3 전달 행렬법

다층 코팅의 경우 전달 행렬법(transfer matrix method)이 표준 도구다. 두께 $d_j$, 굴절률 $n_j$인 각 층 $j$는 2x2 행렬로 기여한다:

$$M_j = \begin{pmatrix} \cos\delta_j & -i\sin\delta_j / \eta_j \\ -i\eta_j \sin\delta_j & \cos\delta_j \end{pmatrix}$$

여기서 $\delta_j = 2\pi n_j d_j \cos\theta_j / \lambda$는 위상 두께(phase thickness)이고, $\eta_j = n_j \cos\theta_j$(s-편광, p-편광은 $\eta_j = n_j / \cos\theta_j$)이다.

전체 시스템 행렬은 $M = M_1 M_2 \cdots M_N$이고, 반사 계수는:

$$r = \frac{(M_{11} + M_{12}\eta_s)\eta_0 - (M_{21} + M_{22}\eta_s)}{(M_{11} + M_{12}\eta_s)\eta_0 + (M_{21} + M_{22}\eta_s)}$$

```python
def transfer_matrix_reflectance(n_layers, d_layers, n_substrate,
                                 wavelengths, theta_i=0, polarization='s'):
    """
    Compute reflectance of a multilayer thin film using the transfer matrix method.

    Parameters:
        n_layers    : list of refractive indices [n_1, n_2, ...] (can be complex)
        d_layers    : list of layer thicknesses [d_1, d_2, ...] in meters
        n_substrate : substrate refractive index
        wavelengths : array of wavelengths (m)
        theta_i     : angle of incidence (rad)
        polarization: 's' or 'p'

    Why transfer matrices: they compose naturally for any number of layers,
    handle interference exactly, and extend to complex refractive indices
    for absorbing films.
    """
    n0 = 1.0  # ambient medium (air)
    R = np.zeros(len(wavelengths))

    for idx, lam in enumerate(wavelengths):
        # Snell's law in each layer
        sin_i = n0 * np.sin(theta_i)

        # Admittances depend on polarization
        cos_i = np.sqrt(1 - (sin_i / n0)**2 + 0j)
        cos_sub = np.sqrt(1 - (sin_i / n_substrate)**2 + 0j)

        if polarization == 's':
            eta_0 = n0 * cos_i
            eta_s = n_substrate * cos_sub
        else:
            eta_0 = n0 / cos_i
            eta_s = n_substrate / cos_sub

        # Build total transfer matrix
        M = np.eye(2, dtype=complex)
        for n_j, d_j in zip(n_layers, d_layers):
            cos_j = np.sqrt(1 - (sin_i / n_j)**2 + 0j)
            delta_j = 2 * np.pi * n_j * d_j * cos_j / lam

            if polarization == 's':
                eta_j = n_j * cos_j
            else:
                eta_j = n_j / cos_j

            layer_matrix = np.array([
                [np.cos(delta_j), -1j * np.sin(delta_j) / eta_j],
                [-1j * eta_j * np.sin(delta_j), np.cos(delta_j)]
            ])
            M = M @ layer_matrix

        # Reflection coefficient
        r = ((M[0, 0] + M[0, 1] * eta_s) * eta_0 -
             (M[1, 0] + M[1, 1] * eta_s)) / \
            ((M[0, 0] + M[0, 1] * eta_s) * eta_0 +
             (M[1, 0] + M[1, 1] * eta_s))

        R[idx] = np.abs(r)**2

    return R

# Compare: uncoated glass vs single-layer MgF2 vs ideal quarter-wave
wavelengths = np.linspace(350e-9, 800e-9, 500)
n_glass = 1.52
lambda_design = 550e-9  # design wavelength (green)

# Uncoated
R_bare = np.abs((1 - n_glass) / (1 + n_glass))**2 * np.ones(len(wavelengths))

# MgF2 coating (n=1.38)
n_MgF2 = 1.38
d_MgF2 = lambda_design / (4 * n_MgF2)
R_MgF2 = transfer_matrix_reflectance([n_MgF2], [d_MgF2], n_glass, wavelengths)

# Ideal quarter-wave (n = sqrt(n_glass))
n_ideal = np.sqrt(n_glass)
d_ideal = lambda_design / (4 * n_ideal)
R_ideal = transfer_matrix_reflectance([n_ideal], [d_ideal], n_glass, wavelengths)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(wavelengths * 1e9, R_bare * 100, 'k--', linewidth=1.5, label='Uncoated')
ax.plot(wavelengths * 1e9, R_MgF2 * 100, 'b-', linewidth=2,
        label=f'MgF$_2$ (n={n_MgF2})')
ax.plot(wavelengths * 1e9, R_ideal * 100, 'r-', linewidth=2,
        label=f'Ideal (n={n_ideal:.3f})')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Reflectance (%)')
ax.set_title('Anti-Reflection Coating Performance')
ax.set_ylim(0, 5)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Shade visible spectrum
ax.axvspan(380, 750, alpha=0.05, color='yellow')
ax.text(565, 4.5, 'Visible', ha='center', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig("antireflection_coating.png", dpi=150)
plt.show()

print(f"Design wavelength: {lambda_design*1e9:.0f} nm")
print(f"MgF2 thickness: {d_MgF2*1e9:.1f} nm")
print(f"MgF2 reflectance at design λ: {R_MgF2[len(wavelengths)//2]*100:.2f}%")
```

---

## 7. 종합 시각화

이 레슨에서 다룬 모든 현상을 보여 주는 완전한 시각화를 만들어 보자.

```python
def comprehensive_fresnel_plot():
    """
    Complete Fresnel analysis for both external and internal reflection.

    Why both cases matter: external reflection (air→glass) is relevant for
    camera lenses and windows, while internal reflection (glass→air) gives
    total internal reflection, essential for fiber optics and prisms.
    """
    theta = np.linspace(0, np.pi / 2 - 0.001, 1000)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- External reflection: air → glass ---
    n1, n2 = 1.0, 1.5
    r_s, r_p, _, _ = fresnel_coefficients(n1, n2, theta)
    R_s, R_p = np.abs(r_s)**2, np.abs(r_p)**2
    theta_B = np.arctan(n2 / n1)

    axes[0, 0].plot(np.degrees(theta), R_s, 'b-', linewidth=2, label='$R_s$')
    axes[0, 0].plot(np.degrees(theta), R_p, 'r-', linewidth=2, label='$R_p$')
    axes[0, 0].axvline(x=np.degrees(theta_B), color='green', linestyle=':',
                        label=f'$\\theta_B$ = {np.degrees(theta_B):.1f}°')
    axes[0, 0].set_title(f'External: Air → Glass ($n_2/n_1$ = {n2/n1:.1f})')
    axes[0, 0].set_ylabel('Reflectance')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)

    # Phase on reflection (external)
    axes[0, 1].plot(np.degrees(theta), np.angle(r_s) / np.pi, 'b-', linewidth=2,
                     label='$\\phi_s / \\pi$')
    axes[0, 1].plot(np.degrees(theta), np.angle(r_p) / np.pi, 'r-', linewidth=2,
                     label='$\\phi_p / \\pi$')
    axes[0, 1].set_title('Phase Shift on External Reflection')
    axes[0, 1].set_ylabel('Phase / $\\pi$')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # --- Internal reflection: glass → air ---
    n1, n2 = 1.5, 1.0
    r_s, r_p, _, _ = fresnel_coefficients(n1, n2, theta)
    R_s, R_p = np.abs(r_s)**2, np.abs(r_p)**2
    theta_c = np.arcsin(n2 / n1)
    theta_B = np.arctan(n2 / n1)

    axes[1, 0].plot(np.degrees(theta), R_s, 'b-', linewidth=2, label='$R_s$')
    axes[1, 0].plot(np.degrees(theta), R_p, 'r-', linewidth=2, label='$R_p$')
    axes[1, 0].axvline(x=np.degrees(theta_c), color='purple', linestyle='--',
                        label=f'$\\theta_c$ = {np.degrees(theta_c):.1f}°')
    axes[1, 0].axvline(x=np.degrees(theta_B), color='green', linestyle=':',
                        label=f'$\\theta_B$ = {np.degrees(theta_B):.1f}°')
    axes[1, 0].set_title(f'Internal: Glass → Air ($n_1/n_2$ = {n1/n2:.1f})')
    axes[1, 0].set_xlabel('Angle of incidence (degrees)')
    axes[1, 0].set_ylabel('Reflectance')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)

    # Phase on reflection (internal) — shows Goos-Hanchen-like phase shifts
    axes[1, 1].plot(np.degrees(theta), np.angle(r_s) / np.pi, 'b-', linewidth=2,
                     label='$\\phi_s / \\pi$')
    axes[1, 1].plot(np.degrees(theta), np.angle(r_p) / np.pi, 'r-', linewidth=2,
                     label='$\\phi_p / \\pi$')
    axes[1, 1].axvline(x=np.degrees(theta_c), color='purple', linestyle='--',
                        label=f'$\\theta_c$ = {np.degrees(theta_c):.1f}°')
    axes[1, 1].set_title('Phase Shift on Internal Reflection')
    axes[1, 1].set_xlabel('Angle of incidence (degrees)')
    axes[1, 1].set_ylabel('Phase / $\\pi$')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("fresnel_comprehensive.png", dpi=150)
    plt.show()

comprehensive_fresnel_plot()
```

---

## 요약

| 개념 | 핵심 공식 | 물리적 의미 |
|------|-----------|-------------|
| 스넬의 법칙 | $n_1 \sin\theta_I = n_2 \sin\theta_T$ | 경계면에서의 위상 정합 |
| 프레넬 $r_s$ | $(n_1\cos\theta_I - n_2\cos\theta_T)/(n_1\cos\theta_I + n_2\cos\theta_T)$ | s-편광 반사 진폭 |
| 프레넬 $r_p$ | $(n_2\cos\theta_I - n_1\cos\theta_T)/(n_2\cos\theta_I + n_1\cos\theta_T)$ | p-편광 반사 진폭 |
| 브루스터 각 | $\tan\theta_B = n_2/n_1$ | p-편광 반사율이 0이 되는 각 |
| 임계각 | $\sin\theta_c = n_2/n_1$ | 전반사 시작 조건 |
| 침투 깊이 | $d_p = \lambda/(2\pi\sqrt{(n_1/n_2)^2\sin^2\theta_I - 1})$ | 에바네선트파의 침투 범위 |
| 4분의 1 파장 코팅 | $d = \lambda/(4n_c)$, $n_c = \sqrt{n_s}$ | 설계 파장 $\lambda$에서 반사율 0 |

---

## 연습 문제

### 연습 문제 1: 다이아몬드의 광채
다이아몬드의 굴절률은 $n = 2.42$이다. (a) 다이아몬드-공기 임계각을 계산하라. (b) 공기에서 다이아몬드로 입사하는 경우 브루스터 각을 구하라. (c) $R_s$와 $R_p$를 0-90도 범위에서 플롯하라. (d) 다이아몬드가 그토록 눈부시게 빛나는 이유를 설명하라(힌트: 임계각이 작아 상단으로 들어온 빛의 대부분이 패싯에서 전반사된다).

### 연습 문제 2: 다층 유전체 거울
5층 유전체 거울을 설계하라(고/저 굴절률 층 교대 배치: $n_H = 2.3$, $n_L = 1.38$, 유리 기판 $n_s = 1.52$). $\lambda = 1064$ nm(Nd:YAG 레이저)에서 $R > 99\%$를 달성해야 한다. 각 층에는 4분의 1 파장 두께를 사용하라. 전달 행렬법을 이용해 800~1300 nm 범위에서 $R(\lambda)$를 플롯하라.

### 연습 문제 3: 좌절 전반사
두 유리 프리즘($n = 1.5$)이 가변 폭 $d$의 공기 간극으로 분리되어 있다. 빔이 45도로 첫 번째 프리즘에 입사한다(임계각 이상). 전달 행렬법(두 경계면 + 공기 간극 층)을 이용해 $d/\lambda$가 0에서 3까지 변할 때 투과율 $T$를 계산하라. $d \to 0$이면 $T \to 1$이고 $d \gg \lambda$이면 $T \to 0$임을 확인하라.

### 연습 문제 4: 반사에 의한 편광
비편광 빛이 다양한 각도에서 유리창($n = 1.5$)에 반사된다. (a) 편광도 $P = (R_s - R_p)/(R_s + R_p)$를 각도의 함수로 플롯하라. (b) $P$가 최대가 되는 각도는? (c) 60도에서 $P$를 계산하고, 편광 선글라스가 효과적인 이유를 설명하라.

---

[← 이전: 10. 물질에서의 전자기파](10_EM_Waves_Matter.md) | [다음: 12. 도파관과 공동 →](12_Waveguides_and_Cavities.md)
