# 15. 다중극 전개

[← 이전: 14. 상대론적 전자기학](14_Relativistic_Electrodynamics.md) | [다음: 16. 전산 전자기학 →](16_Computational_Electrodynamics.md)

## 학습 목표

1. 다중극 전개(Multipole Expansion)를 원거리 장에 대한 체계적인 근사 방법으로 이해한다
2. $1/r$의 거듭제곱 형태로 단극(Monopole), 쌍극(Dipole), 사중극(Quadrupole) 항을 유도한다
3. 전개식과 르장드르 다항식(Legendre Polynomials) 및 구면 조화 함수(Spherical Harmonics)의 연관성을 파악한다
4. 주어진 전하 및 전류 분포에 대해 전기·자기 다중극 모멘트를 계산한다
5. 다중극 복사: 전기 쌍극, 자기 쌍극, 전기 사중극 복사를 분석한다
6. 어느 다중극이 복사를 방출하는지를 결정하는 선택 규칙(Selection Rules)을 이해한다
7. Python으로 다중극 계산을 구현하고 복사 패턴을 시각화한다

전하 분포를 멀리서 관측할 때, 세부 형태는 점점 덜 중요해진다. 중요한 것은 점차 거칠어지는 일련의 기술들이다: 총 전하(단극), 양전하와 음전하의 분리(쌍극), 그 분리의 비대칭성(사중극), 이런 식으로 계속된다. 다중극 전개는 이러한 계층 구조를 제공하며, 임의의 전하 분포를 $1/r$의 점차 높은 거듭제곱으로 감소하는 기여들의 급수로 분해한다. 복사의 경우, 다중극 전개는 가속하는 전하가 주로 쌍극처럼 복사함을 보여주며, 파장에 비해 소스의 크기가 클 때 사중극 이상의 보정이 중요해진다.

> **비유**: 비행기에서 도시를 내려다보는 것을 상상해 보자. 3만 피트(단극 수준)에서는 "여기에 도시가 있다"는 사실, 즉 하나의 점만 보인다. 1만 피트(쌍극 수준)로 내려오면 "도시가 남북 방향으로 길게 뻗어 있다"는 것을 알 수 있다. 5천 피트(사중극 수준)에서는 "밀집된 중심부에서 네 방향으로 교외가 뻗어 있다"는 것이 보인다. 다중극 전개의 각 수준은 점점 세밀한 공간적 디테일을 더해준다. 마치 점점 확대하여 들여다보는 것처럼.

---

## 1. 정전기학에서의 다중극 전개

### 1.1 기본 설정

크기 $d$의 영역에 국한된 전하 분포 $\rho(\mathbf{r}')$를 생각하자. 소스로부터 멀리 떨어진 점 $\mathbf{r}$ ($r \gg d$)에서의 퍼텐셜을 구하고자 한다.

정확한 퍼텐셜은 다음과 같다:

$$V(\mathbf{r}) = \frac{1}{4\pi\epsilon_0}\int \frac{\rho(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} \, d^3r'$$

### 1.2 $1/|\mathbf{r} - \mathbf{r}'|$의 전개

핵심적인 수학적 항등식은 다음과 같다:

$$\frac{1}{|\mathbf{r} - \mathbf{r}'|} = \sum_{\ell=0}^{\infty} \frac{r'^{\ell}}{r^{\ell+1}} P_\ell(\cos\alpha)$$

여기서 $\alpha$는 $\mathbf{r}$과 $\mathbf{r}'$ 사이의 각도이고, $P_\ell$은 **르장드르 다항식(Legendre Polynomials)**이다. 이 식은 $r > r'$일 때 유효하다.

처음 몇 개의 르장드르 다항식은 다음과 같다:

$$P_0(x) = 1, \quad P_1(x) = x, \quad P_2(x) = \frac{1}{2}(3x^2 - 1), \quad P_3(x) = \frac{1}{2}(5x^3 - 3x)$$

### 1.3 항별 전개

퍼텐셜에 대입하면:

$$V(\mathbf{r}) = \frac{1}{4\pi\epsilon_0}\sum_{\ell=0}^{\infty} \frac{1}{r^{\ell+1}} \int r'^{\ell} P_\ell(\cos\alpha) \, \rho(\mathbf{r}') \, d^3r'$$

**단극(Monopole, $\ell = 0$)**:

$$V_0 = \frac{1}{4\pi\epsilon_0} \frac{Q}{r}, \quad Q = \int \rho(\mathbf{r}') \, d^3r'$$

이것은 총 전하로, 멀리서 보면 점전하처럼 보인다.

**쌍극(Dipole, $\ell = 1$)**:

$$V_1 = \frac{1}{4\pi\epsilon_0} \frac{\mathbf{p} \cdot \hat{r}}{r^2}, \quad \mathbf{p} = \int \mathbf{r}' \rho(\mathbf{r}') \, d^3r'$$

쌍극 모멘트 $\mathbf{p}$는 양전하와 음전하의 분리를 나타낸다.

**사중극(Quadrupole, $\ell = 2$)**:

$$V_2 = \frac{1}{4\pi\epsilon_0} \frac{1}{r^3} \sum_{ij} \frac{1}{2} Q_{ij} \hat{r}_i \hat{r}_j$$

여기서 사중극 모멘트 텐서(Quadrupole Moment Tensor)는 다음과 같다:

$$Q_{ij} = \int (3r'_i r'_j - r'^2 \delta_{ij}) \rho(\mathbf{r}') \, d^3r'$$

사중극 텐서는 대칭이고 대각합이 0(Traceless)이다(독립 성분 5개).

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

def multipole_potential(charges, positions, r_eval, theta_eval, max_ell=10):
    """
    Compute electrostatic potential using multipole expansion.

    Parameters:
        charges   : list of charges
        positions : list of position vectors [[x,y,z], ...]
        r_eval    : radial distances for evaluation
        theta_eval: polar angles for evaluation
        max_ell   : maximum multipole order

    Why multipole expansion: for distant fields, only the first few
    terms contribute significantly, giving both physical insight
    (what "shape" does the charge distribution have?) and computational
    efficiency (O(L) vs O(N) for N charges).
    """
    eps_0 = 8.854e-12
    charges = np.array(charges)
    positions = np.array(positions)

    V = np.zeros((len(r_eval), len(theta_eval)))

    for ell in range(max_ell + 1):
        P_ell = legendre(ell)

        # Compute multipole moment: q_ell = sum_i q_i * r_i^ell * P_ell(cos theta_i)
        for q, pos in zip(charges, positions):
            r_i = np.linalg.norm(pos)
            if r_i < 1e-15:
                cos_alpha_i = 0
            else:
                cos_alpha_i = pos[2] / r_i  # z/r for alignment with z-axis

            q_ell = q * r_i**ell * P_ell(cos_alpha_i)

            for j, theta in enumerate(theta_eval):
                V[:, j] += (1 / (4 * np.pi * eps_0)) * q_ell * P_ell(np.cos(theta)) / r_eval**(ell + 1)

    return V

def compare_exact_vs_multipole():
    """
    Compare exact potential with successive multipole approximations
    for a simple charge distribution.
    """
    eps_0 = 8.854e-12

    # Two charges: +q at (0,0,d/2) and -q at (0,0,-d/2)
    # This is a pure dipole
    d = 0.1   # separation (m)
    q = 1e-9  # charge (C)

    charges = [q, -q]
    positions = [[0, 0, d/2], [0, 0, -d/2]]

    # Evaluate along different angles at fixed r
    theta = np.linspace(0.01, np.pi - 0.01, 200)
    r_eval = np.array([1.0])  # 1 meter away

    # Exact potential
    V_exact = np.zeros(len(theta))
    for qi, pos in zip(charges, positions):
        for j, th in enumerate(theta):
            r_point = np.array([r_eval[0] * np.sin(th), 0, r_eval[0] * np.cos(th)])
            dist = np.linalg.norm(r_point - np.array(pos))
            V_exact[j] += qi / (4 * np.pi * eps_0 * dist)

    # Multipole approximations
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Angular pattern comparison
    ax = axes[0]
    ax.plot(np.degrees(theta), V_exact * 1e9, 'k-', linewidth=2, label='Exact')

    for max_l, color, style in [(0, 'red', '--'), (1, 'blue', '-.'), (2, 'green', ':')]:
        V_approx = np.zeros(len(theta))
        for ell in range(max_l + 1):
            P_ell = legendre(ell)
            for qi, pos in zip(charges, positions):
                r_i = np.linalg.norm(pos)
                cos_alpha_i = pos[2] / r_i if r_i > 0 else 0
                q_ell = qi * r_i**ell * P_ell(cos_alpha_i)
                for j, th in enumerate(theta):
                    V_approx[j] += q_ell * P_ell(np.cos(th)) / (4 * np.pi * eps_0 * r_eval[0]**(ell + 1))

        label = f'Up to $\\ell$ = {max_l} ({"monopole" if max_l==0 else "dipole" if max_l==1 else "quadrupole"})'
        ax.plot(np.degrees(theta), V_approx * 1e9, color=color, linestyle=style,
                linewidth=1.5, label=label)

    ax.set_xlabel('Polar angle $\\theta$ (degrees)')
    ax.set_ylabel('Potential (nV at r = 1 m)')
    ax.set_title('Multipole Approximation: Dipole Configuration')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Radial falloff comparison
    ax = axes[1]
    r_range = np.logspace(-0.5, 1.5, 100)  # 0.3 to 30 meters
    theta_fixed = np.pi / 4  # 45 degrees

    V_exact_r = np.zeros(len(r_range))
    V_mono_r = np.zeros(len(r_range))
    V_dipo_r = np.zeros(len(r_range))

    for i, r in enumerate(r_range):
        r_point = np.array([r * np.sin(theta_fixed), 0, r * np.cos(theta_fixed)])
        for qi, pos in zip(charges, positions):
            dist = np.linalg.norm(r_point - np.array(pos))
            V_exact_r[i] += qi / (4 * np.pi * eps_0 * dist)

        # Dipole approximation: p = q*d along z
        p = q * d
        V_dipo_r[i] = p * np.cos(theta_fixed) / (4 * np.pi * eps_0 * r**2)

    ax.loglog(r_range, np.abs(V_exact_r), 'k-', linewidth=2, label='Exact')
    ax.loglog(r_range, np.abs(V_dipo_r), 'b--', linewidth=1.5, label='Dipole approx')
    ax.loglog(r_range, np.abs(V_exact_r[0]) * (r_range[0]/r_range)**2,
              'gray', linestyle=':', alpha=0.5, label='$\\sim 1/r^2$ guide')
    ax.set_xlabel('Distance r (m)')
    ax.set_ylabel('|Potential| (V)')
    ax.set_title('Radial Falloff at $\\theta$ = 45°')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig("multipole_expansion.png", dpi=150)
    plt.show()

compare_exact_vs_multipole()
```

---

## 2. 구면 조화 함수

### 2.1 덧셈 정리

르장드르 다항식은 **덧셈 정리(Addition Theorem)**를 사용해 분해할 수 있다:

$$P_\ell(\cos\alpha) = \frac{4\pi}{2\ell + 1}\sum_{m=-\ell}^{\ell} Y_\ell^{m*}(\theta', \phi') \, Y_\ell^m(\theta, \phi)$$

이를 통해 다중극 전개를 구면 조화 함수로 표현할 수 있다:

$$V(\mathbf{r}) = \frac{1}{4\pi\epsilon_0}\sum_{\ell=0}^{\infty}\sum_{m=-\ell}^{\ell} \frac{4\pi}{2\ell + 1} \frac{q_{\ell m}}{r^{\ell+1}} Y_\ell^m(\theta, \phi)$$

여기서 **구면 다중극 모멘트(Spherical Multipole Moments)**는 다음과 같다:

$$q_{\ell m} = \int r'^{\ell} Y_\ell^{m*}(\theta', \phi') \, \rho(\mathbf{r}') \, d^3r'$$

### 2.2 구면 조화 함수의 성질

구면 조화 함수 $Y_\ell^m(\theta, \phi)$는 구면 위에서 완전 정규직교 기저를 이룬다:

$$\int_0^{2\pi}\int_0^{\pi} Y_\ell^{m*}(\theta, \phi) Y_{\ell'}^{m'}(\theta, \phi) \sin\theta \, d\theta \, d\phi = \delta_{\ell\ell'}\delta_{mm'}$$

처음 몇 개는 다음과 같다:

$$Y_0^0 = \frac{1}{\sqrt{4\pi}}, \quad Y_1^0 = \sqrt{\frac{3}{4\pi}}\cos\theta, \quad Y_1^{\pm 1} = \mp\sqrt{\frac{3}{8\pi}}\sin\theta \, e^{\pm i\phi}$$

$$Y_2^0 = \sqrt{\frac{5}{16\pi}}(3\cos^2\theta - 1), \quad Y_2^{\pm 1} = \mp\sqrt{\frac{15}{8\pi}}\sin\theta\cos\theta \, e^{\pm i\phi}$$

### 2.3 독립 성분의 수

각 차수 $\ell$마다 $2\ell + 1$개의 독립 모멘트가 있다:
- $\ell = 0$ (단극): 1개 ($Q$)
- $\ell = 1$ (쌍극): 3개 ($p_x, p_y, p_z$)
- $\ell = 2$ (사중극): 5개 (대각합 없는 대칭 텐서)
- $\ell = 3$ (팔중극): 7개

```python
from scipy.special import sph_harm

def plot_spherical_harmonics(max_ell=3):
    """
    Visualize spherical harmonics Y_l^m on the sphere.

    Why spherical harmonics: they are the angular basis functions
    for the multipole expansion, analogous to sine and cosine
    being the basis for Fourier series. Each Y_l^m represents
    a specific angular pattern of the potential.
    """
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 200)
    THETA, PHI = np.meshgrid(theta, phi)

    fig, axes = plt.subplots(max_ell + 1, 2 * max_ell + 1, figsize=(20, 12),
                              subplot_kw={'projection': 'polar'})

    for ell in range(max_ell + 1):
        for m_idx, m in enumerate(range(-ell, ell + 1)):
            col = m_idx + (max_ell - ell)  # center alignment
            ax = axes[ell, col]

            # Compute Y_l^m
            # scipy sph_harm uses (m, l, phi, theta) convention
            Y = sph_harm(abs(m), ell, PHI, THETA)
            if m < 0:
                Y = np.sqrt(2) * (-1)**m * Y.imag
            elif m > 0:
                Y = np.sqrt(2) * (-1)**m * Y.real
            else:
                Y = Y.real

            # Plot in polar coordinates (theta is the radial axis)
            # Average over phi for a 2D cross-section
            Y_cross = Y[0, :]  # phi = 0 cross-section
            r_plot = np.abs(Y_cross)

            ax.plot(theta, r_plot, 'b-', linewidth=1.5)
            ax.fill_between(theta, 0, r_plot,
                           where=(Y_cross >= 0), alpha=0.3, color='blue')
            ax.fill_between(theta, 0, r_plot,
                           where=(Y_cross < 0), alpha=0.3, color='red')
            ax.set_title(f'$Y_{{{ell}}}^{{{m}}}$', fontsize=10, pad=5)
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        # Hide unused subplots
        for m_idx in range(2 * max_ell + 1):
            if m_idx < (max_ell - ell) or m_idx > (max_ell + ell):
                axes[ell, m_idx].set_visible(False)

    plt.suptitle('Spherical Harmonics $Y_\\ell^m$ (cross-sections at $\\phi=0$)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("spherical_harmonics.png", dpi=150)
    plt.show()

plot_spherical_harmonics(max_ell=3)
```

---

## 3. 자기 다중극

### 3.1 자기 벡터 퍼텐셜의 전개

국소화된 전류 분포의 자기 벡터 퍼텐셜(Magnetic Vector Potential)은 다음과 같다:

$$\mathbf{A}(\mathbf{r}) = \frac{\mu_0}{4\pi}\int \frac{\mathbf{J}(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} \, d^3r'$$

앞서와 같이 $1/|\mathbf{r} - \mathbf{r}'|$를 전개하면, 앞서는 항들은 다음과 같다:

**자기 단극(Magnetic Monopole, $\ell = 0$)**: $\nabla \cdot \mathbf{B} = 0$이기 때문에, 즉 자기 전하가 존재하지 않기 때문에 이 항은 항등적으로 사라진다.

**자기 쌍극(Magnetic Dipole, $\ell = 1$)**:

$$\mathbf{A}_{\text{dip}} = \frac{\mu_0}{4\pi}\frac{\mathbf{m} \times \hat{r}}{r^2}$$

여기서 자기 쌍극 모멘트는 다음과 같다:

$$\mathbf{m} = \frac{1}{2}\int \mathbf{r}' \times \mathbf{J}(\mathbf{r}') \, d^3r'$$

평면 전류 루프의 경우: $\mathbf{m} = I \mathbf{A}$ (전류 × 넓이 벡터).

### 3.2 자기 사중극

자기 사중극 텐서(Magnetic Quadrupole Tensor)는 다음과 같다:

$$M_{ij} = \frac{1}{3}\int (r'_i J_j + r'_j J_i) \, d^3r'$$

전기 사중극과 달리, 자기 사중극은 일반적으로 대각합이 0이 아니며, 복사 특성도 근본적으로 다르다.

---

## 4. 다중극 복사

### 4.1 복사의 계층 구조

특성 크기 $d$이고 파장이 $\lambda$인 진동 소스에서 각 다중극 차수의 복사 일률은 다음과 같이 스케일된다:

$$P_\ell \sim \left(\frac{d}{\lambda}\right)^{2\ell}$$

일반적으로 $d \ll \lambda$ (장파장 극한)이므로, 복사는 소멸하지 않는 가장 낮은 차수의 다중극이 지배한다:

1. **전기 쌍극(E1, Electric Dipole)**: $P \propto |\ddot{\mathbf{p}}|^2 \propto \omega^4 p_0^2$ — 대부분의 안테나와 원자 전이에서 우세
2. **자기 쌍극(M1, Magnetic Dipole)**: $P \propto |\ddot{\mathbf{m}}|^2/c^2 \propto \omega^4 m_0^2/c^2$ — $(v/c)^2$만큼 억제됨
3. **전기 사중극(E2, Electric Quadrupole)**: $P \propto |\dddot{Q}|^2/c^2 \propto \omega^6 Q_0^2 d^2/c^2$ — $(d/\lambda)^2$만큼 억제됨

### 4.2 복사 패턴

각 다중극은 특유의 각도 분포를 갖는다:

| 다중극 | 일률 패턴 | 반전성(Parity) | 편광 |
|--------|-----------|----------------|------|
| E1 | $\sin^2\theta$ | 홀수($-1$) | $\hat{\theta}$ |
| M1 | $\sin^2\theta$ | 짝수($+1$) | $\hat{\phi}$ |
| E2 | $\sin^2\theta\cos^2\theta$ ($m=0$의 경우) | 짝수($+1$) | 혼합 |

E1과 M1의 패턴은 형태상 동일하지만, **반전성(Parity)**이 다르다. M1 복사의 전기장은 $\mathbf{r} \to -\mathbf{r}$ 변환 시 E1과 반대 부호를 갖는다.

### 4.3 전기 사중극 복사

진동하는 선형 사중극($z = \pm d/2$에 $+q$, 원점에 $-2q$)의 경우:

$$P_{\text{E2}} = \frac{\mu_0 \omega^6 Q_0^2}{1440\pi c^3}$$

여기서 $Q_0 = qd^2$는 사중극 모멘트 진폭이다.

```python
def multipole_radiation_patterns():
    """
    Compare radiation patterns of different multipole orders.

    Why compare: understanding the angular patterns helps identify
    the dominant multipole in experimental measurements (e.g., from
    the angular distribution of emitted radiation in atomic physics).
    """
    theta = np.linspace(0, 2 * np.pi, 500)

    # Power patterns for different multipoles
    E1 = np.sin(theta)**2                          # Electric dipole
    M1 = np.sin(theta)**2                          # Magnetic dipole (same shape!)
    E2_m0 = np.sin(theta)**2 * np.cos(theta)**2   # Electric quadrupole (m=0)
    E2_m1 = (1 + np.cos(theta)**2)**2 * np.sin(theta)**2 / 4  # E2 (m=1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12),
                              subplot_kw={'projection': 'polar'})

    patterns = [
        (E1, 'Electric Dipole (E1)', 'blue'),
        (M1, 'Magnetic Dipole (M1)', 'red'),
        (E2_m0, 'Electric Quadrupole E2 (m=0)', 'green'),
        (E2_m1, 'Electric Quadrupole E2 (m=1)', 'purple')
    ]

    for ax, (pattern, title, color) in zip(axes.flat, patterns):
        pattern_norm = pattern / pattern.max() if pattern.max() > 0 else pattern
        ax.plot(theta, pattern_norm, color=color, linewidth=2)
        ax.fill(theta, pattern_norm, alpha=0.2, color=color)
        ax.set_title(title, pad=15, fontsize=11)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

    plt.suptitle('Multipole Radiation Patterns (Power)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("multipole_radiation_patterns.png", dpi=150)
    plt.show()

multipole_radiation_patterns()
```

---

## 5. 선택 규칙

### 5.1 반전성 선택 규칙

각 다중극은 공간 반전($\mathbf{r} \to -\mathbf{r}$) 하에서 명확한 반전성을 갖는다:

- **차수 $\ell$의 전기 다중극**: 반전성 $(-1)^\ell$
- **차수 $\ell$의 자기 다중극**: 반전성 $(-1)^{\ell+1}$

양자역학에서, 명확한 반전성을 가진 상태 사이의 전이는 방출된 광자가 적절한 반전성을 운반해야 한다:

$$\text{E1}: \Delta\ell = \pm 1, \quad \Delta m = 0, \pm 1, \quad \text{반전성 변화}$$

$$\text{M1}: \Delta\ell = 0, \quad \Delta m = 0, \pm 1, \quad \text{반전성 유지}$$

$$\text{E2}: \Delta\ell = 0, \pm 2, \quad \Delta m = 0, \pm 1, \pm 2, \quad \text{반전성 유지}$$

### 5.2 선택 규칙이 중요한 이유

- 수소의 $2s \to 1s$ 전이는 E1에 대해 **금지**되어 있다(동일한 반전성). 이는 두 광자 방출로 일어나며 매우 느리다.
- $2p \to 1s$ 전이는 E1에 **허용**되어 있다(반전성 변화, $\Delta\ell = 1$). 수명은 약 $1.6$ ns이다.
- 핵 감마선(Gamma-ray) 전이는 핵 구조상 높은 다중극을 선호하기 때문에 M1 또는 E2를 통해 일어나는 경우가 많다.

### 5.3 상대적 세기

원자 전이($d \sim a_0 \sim 0.5$ 옹스트롬, $\lambda \sim 5000$ 옹스트롬)의 경우:

$$\frac{P_{\text{M1}}}{P_{\text{E1}}} \sim \left(\frac{v}{c}\right)^2 \sim \alpha^2 \approx 5 \times 10^{-5}$$

$$\frac{P_{\text{E2}}}{P_{\text{E1}}} \sim \left(\frac{d}{\lambda}\right)^2 \sim \left(\frac{a_0}{\lambda}\right)^2 \approx 10^{-7}$$

여기서 $\alpha \approx 1/137$은 미세 구조 상수(Fine-Structure Constant)이다.

```python
def multipole_power_scaling():
    """
    Show how radiated power scales with frequency and source size
    for different multipole orders.

    Why scaling matters: it explains why E1 dominates in atomic physics,
    why nuclear transitions are often M1 or E2, and why radio antennas
    are almost always analyzed as E1 radiators.
    """
    # Ratio d/lambda
    d_over_lambda = np.logspace(-4, 0, 100)

    # Relative power (normalized to E1 = 1 at d/lambda = 0.01)
    E1 = np.ones_like(d_over_lambda)  # reference
    M1 = 0.1 * np.ones_like(d_over_lambda)  # suppressed by (v/c)^2 ~ 0.1 for illustration
    E2 = d_over_lambda**2 / d_over_lambda[0]**2
    M2 = 0.1 * d_over_lambda**2 / d_over_lambda[0]**2
    E3 = d_over_lambda**4 / d_over_lambda[0]**4

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.loglog(d_over_lambda, E1, 'b-', linewidth=2, label='E1 (electric dipole)')
    ax.loglog(d_over_lambda, M1, 'r--', linewidth=2, label='M1 (magnetic dipole)')
    ax.loglog(d_over_lambda, E2, 'g-.', linewidth=2, label='E2 (electric quadrupole)')
    ax.loglog(d_over_lambda, M2, 'm:', linewidth=2, label='M2 (magnetic quadrupole)')
    ax.loglog(d_over_lambda, E3, 'k--', linewidth=1.5, alpha=0.5,
              label='E3 (electric octupole)')

    # Mark typical regimes
    ax.axvspan(1e-4, 1e-2, alpha=0.1, color='blue', label='Atoms ($d/\\lambda \\sim 10^{-3}$)')
    ax.axvspan(1e-2, 1e-1, alpha=0.1, color='green', label='Nuclei ($d/\\lambda \\sim 10^{-2}$)')

    ax.set_xlabel('Source size / wavelength ($d / \\lambda$)')
    ax.set_ylabel('Relative radiated power')
    ax.set_title('Multipole Radiation Power Scaling')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(1e-10, 1e3)

    plt.tight_layout()
    plt.savefig("multipole_scaling.png", dpi=150)
    plt.show()

multipole_power_scaling()
```

---

## 6. 다중극 모멘트 계산

### 6.1 예시: 선형 사중극

$z$축 위에 놓인 세 전하를 생각하자: $z = +d$에 $+q$, 원점에 $-2q$, $z = -d$에 $+q$.

- 단극: $Q = q - 2q + q = 0$
- 쌍극: $\mathbf{p} = q(d\hat{z}) + (-2q)(0) + q(-d\hat{z}) = 0$
- 사중극: $Q_{zz} = q(3d^2 - d^2) + 0 + q(3d^2 - d^2) = 4qd^2$

퍼텐셜은 $1/r^3$으로 감소하는데, 이것이 소멸하지 않는 첫 번째 항이다.

### 6.2 예시: 정사각형 사중극

$xy$평면의 정사각형($a$ 변) 꼭짓점에 놓인 네 전하: $(\pm a/2, a/2, 0)$에 $+q$, $(\pm a/2, -a/2, 0)$에 $-q$.

- 단극: $Q = 0$
- 쌍극: $\mathbf{p} = 2qa\hat{y}$ ($y$ 방향의 알짜 쌍극)

이 배열은 쌍극 모멘트가 0이 아니므로, 사중극 기여가 부차적(Subdominant)이다.

```python
def compute_multipole_moments(charges, positions, max_ell=4):
    """
    Compute electric multipole moments q_{ell,m} for a discrete charge distribution.

    Why compute moments: they characterize the far-field behavior of
    any charge distribution. The moments are the "fingerprint" of the
    source as seen from a distance.
    """
    from scipy.special import sph_harm

    moments = {}

    for ell in range(max_ell + 1):
        for m in range(-ell, ell + 1):
            q_lm = 0 + 0j
            for q, pos in zip(charges, positions):
                r = np.linalg.norm(pos)
                if r < 1e-15:
                    # Y_l^m(0,0) is nonzero only for m=0
                    if m == 0:
                        theta_i, phi_i = 0, 0
                    else:
                        continue
                else:
                    theta_i = np.arccos(pos[2] / r)
                    phi_i = np.arctan2(pos[1], pos[0])

                Y_lm_conj = np.conj(sph_harm(m, ell, phi_i, theta_i))
                q_lm += q * r**ell * Y_lm_conj

            moments[(ell, m)] = q_lm

    return moments

# Linear quadrupole
d = 0.1  # meters
q = 1e-9  # coulombs

charges_lq = [q, -2*q, q]
positions_lq = [[0, 0, d], [0, 0, 0], [0, 0, -d]]

print("Linear Quadrupole (charges: +q, -2q, +q along z-axis)")
print("=" * 60)
moments_lq = compute_multipole_moments(charges_lq, positions_lq)
for (ell, m), val in sorted(moments_lq.items()):
    if abs(val) > 1e-25:
        print(f"  q({ell},{m:+d}) = {val:.4e}")

print("\n")

# Dipole
charges_dp = [q, -q]
positions_dp = [[0, 0, d/2], [0, 0, -d/2]]

print("Pure Dipole (charges: +q, -q along z-axis)")
print("=" * 60)
moments_dp = compute_multipole_moments(charges_dp, positions_dp)
for (ell, m), val in sorted(moments_dp.items()):
    if abs(val) > 1e-25:
        print(f"  q({ell},{m:+d}) = {val:.4e}")
```

---

## 7. 다중극 전개의 응용

### 7.1 중력 다중극

다중극 전개는 중력을 포함한 모든 $1/r$ 장에 적용된다. 지구의 중력 퍼텐셜은 다음과 같다:

$$U = -\frac{GM}{r}\left[1 - \sum_{\ell=2}^{\infty} \left(\frac{R_E}{r}\right)^\ell J_\ell P_\ell(\cos\theta)\right]$$

$J_2 \approx 1.08 \times 10^{-3}$ 항(지구 편평도(Oblateness))은 인공위성 궤도의 세차 운동(Precession)을 유발한다.

### 7.2 핵 다중극 모멘트

핵의 전하 분포는 다중극 모멘트로 특성화된다:
- **사중극 모멘트**: 핵의 변형을 측정한다. 장축형(Prolate) 핵은 $Q > 0$, 단축형(Oblate) 핵은 $Q < 0$이다.
- **자기 쌍극 모멘트**: 핵 스핀(Nuclear Spin) 및 구성과 관련된다.

### 7.3 안테나 설계

안테나 공학자들은 다중극 전개를 사용해 안테나 원거리장(Far-Field) 패턴을 특성화한다. 계수 $a_{\ell m}$과 $b_{\ell m}$ (E 및 M 다중극)은 복사 패턴을 완전히 결정한다.

---

## 요약

| 개념 | 핵심 공식 | 물리적 의미 |
|------|-----------|-------------|
| 단극 퍼텐셜 | $V_0 = Q/(4\pi\epsilon_0 r)$ | 총 전하; $\sim 1/r$ |
| 쌍극 퍼텐셜 | $V_1 = \mathbf{p}\cdot\hat{r}/(4\pi\epsilon_0 r^2)$ | 전하 분리; $\sim 1/r^2$ |
| 사중극 퍼텐셜 | $V_2 \sim Q_{ij}\hat{r}_i\hat{r}_j / r^3$ | 전하 비대칭성; $\sim 1/r^3$ |
| 구면 조화 함수 | $q_{\ell m} = \int r'^\ell Y_\ell^{m*} \rho \, d^3r'$ | 소스의 각도 분해 |
| E1 복사 일률 | $P \propto \omega^4 p_0^2$ | $d \ll \lambda$일 때 우세 |
| 일률 스케일링 | $P_\ell \propto (d/\lambda)^{2\ell}$ | 높은 다중극일수록 억제됨 |
| E1 선택 규칙 | $\Delta\ell = \pm 1$, 반전성 변화 | 원자 전이 지배 |

---

## 연습 문제

### 연습 1: 사중극 퍼텐셜 지도
$z$축 위에 간격 $d$로 배치된 선형 사중극($+q$, $-2q$, $+q$)의 정전기 퍼텐셜 2D 등고선 그래프를 그려라. $r = 2d, 5d, 10d$ 거리에서 쿨롱 법칙으로 계산한 정확한 퍼텐셜과 비교하라. 사중극 근사가 5% 이내의 정확도를 가지려면 최소 어느 정도 거리가 필요한가?

### 연습 2: 고리의 다중극 모멘트
반지름 $R$이고 총 전하 $Q$인 균일하게 대전된 고리가 $xy$평면에 놓여 있다. (a) 대칭성에 의해 $q_{\ell m} = 0$ ($m \neq 0$)이므로, $\ell = 6$까지 모든 다중극 모멘트 $q_{\ell 0}$를 계산하라. (b) 짝수 $\ell$만 기여함을 보여라. (c) 축 위에서의 정확한 퍼텐셜과 $\ell = 4$에서 끊은 다중극 급수를 비교하라.

### 연습 3: 자기 쌍극 복사
수소 원자의 전자가 $2s$에서 $1s$ 상태로 전이한다. 두 상태 모두 $\ell = 0$임을 이용해 E1 복사가 왜 금지되는지 설명하라. M1 전이율을 $2p \to 1s$ E1 전이율과 비교하여 추정하라. $2s \to 1s$ 붕괴의 실제 메커니즘은 무엇인가?

### 연습 4: E2 복사 패턴
$z$축을 따라 진동하는 전기 사중극의 전체 3D 복사 패턴을 그려라. $m = 0$의 경우 $dP/d\Omega \propto \sin^2\theta\cos^2\theta$ 공식을 사용하라. E1 패턴과 비교하고 영점 방향(Null Directions)을 찾아라.

### 연습 5: 지구의 중력 다중극
지구에 대해 $J_2 = 1.0826 \times 10^{-3}$, $J_4 = -1.62 \times 10^{-6}$이다. (a) $J_4$ 항까지 포함하여 적도와 극에서 지표면의 중력 퍼텐셜을 계산하라. (b) 적도와 극 사이의 지오이드(Geoid) 높이 차이는 얼마인가? (c) $J_2$가 저궤도 위성의 궤도 세차 운동률에 어떤 영향을 미치는가?

---

[← 이전: 14. 상대론적 전자기학](14_Relativistic_Electrodynamics.md) | [다음: 16. 전산 전자기학 →](16_Computational_Electrodynamics.md)
