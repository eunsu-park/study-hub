# 12. 도파관과 공동

[← 이전: 11. 반사와 굴절](11_Reflection_and_Refraction.md) | [다음: 13. 복사와 안테나 →](13_Radiation_and_Antennas.md)

## 학습 목표

1. 맥스웰 방정식으로부터 안내파(guided wave)의 횡방향 전기장 방정식 유도
2. 직사각형 도파관에서 TE 모드와 TM 모드의 해를 구하고 차단 주파수 파악
3. 도파관 분산 관계와 위상 속도 및 군 속도 개념 이해
4. 직사각형 공동 공진기(cavity resonator)의 분석과 공진 주파수 및 품질 인수 Q 계산
5. 원형 도파관과 광섬유의 기본 원리 기술
6. Python을 이용한 도파관 파라미터의 수치 계산 및 모드 패턴 시각화
7. 도파관 이론과 마이크로파 공학 및 포토닉스의 실제 응용 연결

도파관(waveguide)은 물이 파이프를 통해 흐르듯 전자기파를 가두고 안내하는 중공(hollow) 또는 유전체 구조물이다. 자유 공간 전파와 달리 도파관은 경계 조건을 부과하여 횡방향 전기장 패턴을 이산적인 **모드(mode)**로 양자화하며, 각 모드는 최소 동작 주파수를 갖는다. 이 모드 구조가 마이크로파 전송, 레이더 배관, 입자 가속기, 광섬유 통신을 이해하는 핵심이다. 이 레슨에서는 제1원리로부터 모드 구조를 유도하고, 도파관 분산의 풍부한 물리를 탐구하며, 3차원에서 파동이 갇히는 공진 공동으로 분석을 확장한다.

> **비유**: 도파관은 소리를 위한 복도와 같다. 좁은 복도에서 소리를 지르면 벽과의 상쇄 간섭 없이 전파할 수 있는 특정 공간 패턴(모드)의 음파만 살아남는다. 매우 낮은 주파수의 소리(복도 폭보다 훨씬 긴 파장)는 전혀 전파할 수 없다 — "차단 주파수 아래"에 있기 때문이다. 마찬가지로, 마이크로파 도파관은 우세 모드(dominant mode)의 차단 주파수보다 높은 주파수만 통과시킨다.

---

## 1. 안내파의 일반 이론

### 1.1 설정

$z$축 방향으로 뻗어 있으며 $xy$평면에서 균일한 단면을 가진 도파관을 고려하자. 다음 형태의 해를 구한다:

$$\mathbf{E}(x, y, z, t) = \mathbf{E}(x, y) \, e^{i(k_z z - \omega t)}$$

$$\mathbf{H}(x, y, z, t) = \mathbf{H}(x, y) \, e^{i(k_z z - \omega t)}$$

여기서 $k_z$는 도파관을 따르는 전파 상수이다.

### 1.2 횡방향 및 종방향 분해

전기장을 횡방향 성분($\mathbf{E}_T$, $\mathbf{H}_T$)과 종방향 성분($E_z$, $H_z$)으로 분해한다. 맥스웰의 회전 방정식은 **횡방향 전기장을 $E_z$와 $H_z$만으로** 표현해 준다:

$$\mathbf{E}_T = \frac{i}{k_c^2}\left(k_z \nabla_T E_z - \omega\mu \, \hat{z} \times \nabla_T H_z\right)$$

$$\mathbf{H}_T = \frac{i}{k_c^2}\left(k_z \nabla_T H_z + \omega\epsilon \, \hat{z} \times \nabla_T E_z\right)$$

여기서 $k_c^2 = k^2 - k_z^2$는 **차단 파수(cutoff wave number)**이고 $k = \omega\sqrt{\mu\epsilon}$이다.

이것은 강력한 결과다: $E_z$와 $H_z$(스칼라 문제)를 풀면 6개의 전기장 성분 전부가 결정된다.

### 1.3 모드 분류

- **TE 모드(Transverse Electric)**: $E_z = 0$, 전기장은 $H_z$로 결정
- **TM 모드(Transverse Magnetic)**: $H_z = 0$, 전기장은 $E_z$로 결정
- **TEM 모드**: $E_z = H_z = 0$, 두 개 이상의 도체 필요(예: 동축 케이블)

중공 도파관에서는 TEM 모드가 불가능하다 — $E_z$ 또는 $H_z$ 중 적어도 하나는 0이 아니어야 한다.

---

## 2. 직사각형 도파관

### 2.1 기하 구조

관례에 따라 폭 $a$($x$ 방향), 높이 $b$($y$ 방향)($a > b$)인 직사각형 도파관을 고려하자. 벽은 완전 도체이다.

### 2.2 TM 모드($H_z = 0$)

종방향 전기장은 다음을 만족한다:

$$\left(\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + k_c^2\right) E_z = 0$$

모든 벽에서 $E_z = 0$(완전 도체에서 접선 방향 $E$가 0)인 경계 조건을 가진다. 해는:

$$E_z^{mn} = E_0 \sin\left(\frac{m\pi x}{a}\right) \sin\left(\frac{n\pi y}{b}\right)$$

$m = 1, 2, 3, \ldots$, $n = 1, 2, 3, \ldots$ (비자명 해를 위해 둘 다 최소 1이어야 한다).

차단 파수는:

$$k_c^{mn} = \sqrt{\left(\frac{m\pi}{a}\right)^2 + \left(\frac{n\pi}{b}\right)^2}$$

### 2.3 TE 모드($E_z = 0$)

종방향 자기장은 같은 헬름홀츠 방정식을 만족하지만, 벽에서 $\partial H_z / \partial n = 0$인 노이만(Neumann) 경계 조건을 가진다:

$$H_z^{mn} = H_0 \cos\left(\frac{m\pi x}{a}\right) \cos\left(\frac{n\pi y}{b}\right)$$

$m = 0, 1, 2, \ldots$, $n = 0, 1, 2, \ldots$ (단 $m = n = 0$은 제외). 코사인 함수는 자연스럽게 노이만 조건을 만족한다.

### 2.4 차단 주파수

모드 $(m, n)$의 차단 주파수는:

$$\boxed{f_c^{mn} = \frac{1}{2\sqrt{\mu\epsilon}} \sqrt{\left(\frac{m}{a}\right)^2 + \left(\frac{n}{b}\right)^2}}$$

차단 이하($f < f_c$)에서 $k_z$는 허수가 되어 파동이 에바네선트(evanescent)해진다. $a > b$의 경우:

- **우세 모드(dominant mode)**: TE$_{10}$, $f_c = 1/(2a\sqrt{\mu\epsilon})$ — 가장 낮은 차단 주파수
- **첫 번째 TM 모드**: TM$_{11}$, TE$_{10}$보다 높은 차단 주파수

**단일 모드 대역폭(single-mode bandwidth)**은 우세 모드만 전파하는 주파수 범위 $[f_c^{10}, f_c^{20}]$(또는 $a/b$ 비율에 따라 $[f_c^{10}, f_c^{01}]$)이다. 표준 $a = 2b$ 비율의 경우 한 옥타브: $f_c^{20} = 2f_c^{10}$.

```python
import numpy as np
import matplotlib.pyplot as plt

def cutoff_frequencies(a, b, max_m=5, max_n=5, mu=4*np.pi*1e-7, eps=8.854e-12):
    """
    Compute cutoff frequencies for rectangular waveguide modes.

    Parameters:
        a, b  : waveguide dimensions (m), a > b
        max_m, max_n : maximum mode indices

    Why mode ordering matters: in practice, we want single-mode operation
    (only the dominant mode propagating). The cutoff frequencies determine
    the usable bandwidth of the waveguide.
    """
    modes = []
    for m in range(max_m + 1):
        for n in range(max_n + 1):
            if m == 0 and n == 0:
                continue  # no TEM mode in hollow waveguide

            fc = 0.5 / np.sqrt(mu * eps) * np.sqrt((m / a)**2 + (n / b)**2)

            # Determine mode type
            if m == 0 or n == 0:
                mode_type = 'TE'  # TE modes allow m=0 or n=0
            else:
                mode_type = 'TE/TM'  # both exist for m,n >= 1

            modes.append({
                'm': m, 'n': n,
                'fc_GHz': fc / 1e9,
                'type': mode_type
            })

    # Sort by cutoff frequency
    modes.sort(key=lambda x: x['fc_GHz'])
    return modes

# WR-90 waveguide (X-band): a = 22.86 mm, b = 10.16 mm
a = 22.86e-3  # m
b = 10.16e-3  # m

modes = cutoff_frequencies(a, b)

print("Rectangular Waveguide WR-90 Mode Table")
print("=" * 55)
print(f"{'Mode':<12} {'Type':<8} {'f_c (GHz)':>10}")
print("-" * 55)
for mode in modes[:12]:
    name = f"({mode['m']},{mode['n']})"
    print(f"{name:<12} {mode['type']:<8} {mode['fc_GHz']:>10.3f}")

print(f"\nDominant mode: TE10, f_c = {modes[0]['fc_GHz']:.3f} GHz")
print(f"Single-mode band: {modes[0]['fc_GHz']:.3f} - {modes[1]['fc_GHz']:.3f} GHz")
```

---

## 3. 분산 관계

### 3.1 도파관 분산

전파 모드의 전파 상수는:

$$k_z = \sqrt{k^2 - k_c^2} = \frac{\omega}{c}\sqrt{1 - \left(\frac{f_c}{f}\right)^2}$$

이로부터 **도파관 분산 관계(waveguide dispersion relation)**가 나온다:

$$\boxed{\omega^2 = \omega_c^2 + k_z^2 c^2}$$

이는 상대론적 에너지-운동량 관계 $E^2 = (mc^2)^2 + (pc)^2$와 같은 형태로, $\omega_c$가 "정지 질량" 역할을 한다. 차단 주파수는 안내된 광자에 대한 유효 질량처럼 작용한다.

### 3.2 위상 속도와 군 속도

**위상 속도(phase velocity)** (도파관을 따라 일정 위상면이 이동하는 속도):

$$v_p = \frac{\omega}{k_z} = \frac{c}{\sqrt{1 - (f_c/f)^2}} > c$$

**군 속도(group velocity)** (에너지 전송 속도):

$$v_g = \frac{d\omega}{dk_z} = c\sqrt{1 - \left(\frac{f_c}{f}\right)^2} < c$$

주목할 만한 관계:

$$\boxed{v_p \cdot v_g = c^2}$$

위상 속도는 $c$를 초과하지만, 정보와 에너지를 실어 나르는 군 속도는 항상 $c$보다 작다. 차단 주파수에서 $v_g \to 0$이고 $v_p \to \infty$이다.

```python
def waveguide_dispersion(a, b, m, n, f_max_GHz=20):
    """
    Plot the dispersion relation and velocities for a waveguide mode.

    Why dispersion matters: the frequency-dependent group velocity causes
    pulse broadening in waveguide communication systems, analogous to
    chromatic dispersion in optical fibers.
    """
    c = 3e8  # speed of light (m/s)
    fc = 0.5 * c * np.sqrt((m / a)**2 + (n / b)**2)
    fc_GHz = fc / 1e9

    f = np.linspace(fc * 1.01, f_max_GHz * 1e9, 1000)
    omega = 2 * np.pi * f
    omega_c = 2 * np.pi * fc

    # Propagation constant
    kz = (omega / c) * np.sqrt(1 - (fc / f)**2)

    # Phase and group velocity
    vp = omega / kz
    vg = c**2 / vp  # from vp * vg = c^2

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Dispersion diagram (omega vs kz)
    axes[0].plot(kz, f / 1e9, 'b-', linewidth=2)
    axes[0].plot(kz, kz * c / (2 * np.pi * 1e9), 'k--', alpha=0.5,
                 label='Light line ($\\omega = ck_z$)')
    axes[0].axhline(y=fc_GHz, color='r', linestyle=':', label=f'$f_c$ = {fc_GHz:.2f} GHz')
    axes[0].set_xlabel('$k_z$ (rad/m)')
    axes[0].set_ylabel('Frequency (GHz)')
    axes[0].set_title(f'Dispersion: TE$_{{{m}{n}}}$')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Phase and group velocity
    axes[1].plot(f / 1e9, vp / c, 'b-', linewidth=2, label='$v_p / c$')
    axes[1].plot(f / 1e9, vg / c, 'r-', linewidth=2, label='$v_g / c$')
    axes[1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(x=fc_GHz, color='green', linestyle=':', label=f'$f_c$')
    axes[1].set_xlabel('Frequency (GHz)')
    axes[1].set_ylabel('Velocity / c')
    axes[1].set_title('Phase and Group Velocity')
    axes[1].set_ylim(0, 5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Guide wavelength vs free-space wavelength
    lambda_g = 2 * np.pi / kz
    lambda_0 = c / f

    axes[2].plot(lambda_0 * 1e3, lambda_g * 1e3, 'b-', linewidth=2)
    axes[2].plot(lambda_0 * 1e3, lambda_0 * 1e3, 'k--', alpha=0.5,
                 label='$\\lambda_g = \\lambda_0$')
    axes[2].set_xlabel('Free-space wavelength (mm)')
    axes[2].set_ylabel('Guide wavelength (mm)')
    axes[2].set_title('Guide Wavelength')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'WR-90 Rectangular Waveguide: TE$_{{{m}{n}}}$ Mode', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("waveguide_dispersion.png", dpi=150)
    plt.show()

waveguide_dispersion(a=22.86e-3, b=10.16e-3, m=1, n=0)
```

---

## 4. 모드 패턴

### 4.1 TE$_{mn}$ 모드의 전기장 분포

TE$_{mn}$ 모드에서 횡방향 전기장은:

$$H_x = -\frac{ik_z}{k_c^2} \frac{m\pi}{a} H_0 \sin\left(\frac{m\pi x}{a}\right)\cos\left(\frac{n\pi y}{b}\right)$$

$$H_y = -\frac{ik_z}{k_c^2} \frac{n\pi}{b} H_0 \cos\left(\frac{m\pi x}{a}\right)\sin\left(\frac{n\pi y}{b}\right)$$

$$E_x = \frac{i\omega\mu}{k_c^2} \frac{n\pi}{b} H_0 \cos\left(\frac{m\pi x}{a}\right)\sin\left(\frac{n\pi y}{b}\right)$$

$$E_y = -\frac{i\omega\mu}{k_c^2} \frac{m\pi}{a} H_0 \sin\left(\frac{m\pi x}{a}\right)\cos\left(\frac{n\pi y}{b}\right)$$

우세 모드 TE$_{10}$은 특히 단순한 패턴을 갖는다: $E_y = E_0 \sin(\pi x / a)$로, 넓은 방향을 가로지르는 단일 반파장 변화이다.

```python
def plot_waveguide_modes(a, b, modes_to_plot=None):
    """
    Visualize electric field patterns for rectangular waveguide modes.

    Why visualize: the mode pattern determines coupling efficiency
    to antennas, the location of slots for radiation, and the
    current distribution on the waveguide walls.
    """
    if modes_to_plot is None:
        modes_to_plot = [('TE', 1, 0), ('TE', 2, 0), ('TE', 0, 1),
                         ('TE', 1, 1), ('TM', 1, 1), ('TE', 2, 1)]

    x = np.linspace(0, a, 100)
    y = np.linspace(0, b, 80)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, (mode_type, m, n) in enumerate(modes_to_plot):
        if idx >= len(axes):
            break

        kc2 = (m * np.pi / a)**2 + (n * np.pi / b)**2

        if mode_type == 'TE':
            # Electric field components for TE_mn
            if kc2 == 0:
                continue
            Ex = (n * np.pi / b) * np.cos(m * np.pi * X / a) * np.sin(n * np.pi * Y / b)
            Ey = -(m * np.pi / a) * np.sin(m * np.pi * X / a) * np.cos(n * np.pi * Y / b)
        else:  # TM
            # Electric field components for TM_mn
            Ex = (m * np.pi / a) * np.cos(m * np.pi * X / a) * np.sin(n * np.pi * Y / b)
            Ey = (n * np.pi / b) * np.sin(m * np.pi * X / a) * np.cos(n * np.pi * Y / b)

        E_mag = np.sqrt(Ex**2 + Ey**2)

        ax = axes[idx]
        im = ax.pcolormesh(X * 1e3, Y * 1e3, E_mag, cmap='hot', shading='auto')

        # Add vector arrows (subsample for clarity)
        skip = 8
        scale = E_mag.max() if E_mag.max() > 0 else 1
        ax.quiver(X[::skip, ::skip] * 1e3, Y[::skip, ::skip] * 1e3,
                  Ex[::skip, ::skip] / scale, Ey[::skip, ::skip] / scale,
                  color='cyan', alpha=0.7, scale=15)

        c_val = 3e8
        fc = 0.5 * c_val * np.sqrt((m / a)**2 + (n / b)**2)
        ax.set_title(f'{mode_type}$_{{{m}{n}}}$ ($f_c$ = {fc/1e9:.2f} GHz)')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_aspect('equal')

    plt.suptitle('Rectangular Waveguide Mode Patterns (Electric Field)', fontsize=14)
    plt.tight_layout()
    plt.savefig("waveguide_modes.png", dpi=150)
    plt.show()

plot_waveguide_modes(a=22.86e-3, b=10.16e-3)
```

---

## 5. 직사각형 공동 공진기

### 5.1 도파관에서 공동으로

공진 공동(resonant cavity)은 도파관의 양 끝을 도체 벽($z = 0$과 $z = d$)으로 막아 형성된다. 추가 경계 조건으로 $k_z$가 양자화된다:

$$k_z = \frac{p\pi}{d}, \quad p = 0, 1, 2, \ldots$$

(TM 모드는 $p = 0$이 허용되고; TE 모드는 $H_z$의 특정 경계 조건에서 비자명 해를 위해 $p \geq 1$이어야 한다.)

### 5.2 공진 주파수

직사각형 공동의 공진 주파수는:

$$\boxed{f_{mnp} = \frac{1}{2\sqrt{\mu\epsilon}}\sqrt{\left(\frac{m}{a}\right)^2 + \left(\frac{n}{b}\right)^2 + \left(\frac{p}{d}\right)^2}}$$

공동의 **우세 모드**는 치수에 따라 달라진다. $a > d > b$의 경우 가장 낮은 공진이 일반적으로 TE$_{101}$이다.

### 5.3 품질 인수 Q

품질 인수(quality factor)는 공동이 얼마나 잘 에너지를 저장하는지를 측정한다:

$$\boxed{Q = \omega_0 \frac{W_{\text{stored}}}{P_{\text{loss}}} = \frac{f_0}{\Delta f}}$$

여기서 $W_{\text{stored}}$는 시간 평균 저장 에너지이고, $P_{\text{loss}}$는 벽에서 소산되는 전력이다.

벽 전도율이 $\sigma$인 직사각형 공동의 TE$_{101}$ 모드에 대해:

$$Q = \frac{(k a)^2 b d \, \mu \omega}{4 R_s \left[b d (a^2 + d^2)/a^2 d^2 \cdot \pi^2/2 + a(b/d + d/b) + \ldots\right]}$$

여기서 $R_s = \sqrt{\omega\mu / 2\sigma}$는 표면 저항이다. 구리 공동의 전형적인 값: $Q \sim 10^3$~$10^5$.

입자 가속기에 사용되는 초전도 공동은 $Q > 10^{10}$을 달성한다.

```python
def cavity_resonances(a, b, d, max_mode=4):
    """
    Compute resonant frequencies of a rectangular cavity.

    Why cavities matter: microwave ovens use cavity resonances to
    heat food, particle accelerators use superconducting cavities
    to accelerate charged particles, and radar systems use cavity
    filters for frequency selection.
    """
    c = 3e8
    resonances = []

    for m in range(max_mode + 1):
        for n in range(max_mode + 1):
            for p in range(max_mode + 1):
                if m == 0 and n == 0:
                    continue

                # TE modes: need at least two nonzero indices
                # TM modes: m >= 1 and n >= 1, p can be 0
                if m >= 1 and n >= 1:
                    f = 0.5 * c * np.sqrt((m/a)**2 + (n/b)**2 + (p/d)**2)
                    resonances.append(('TM' if p == 0 or True else 'TM',
                                       m, n, p, f / 1e9))

                if (m >= 1 or n >= 1) and p >= 1:
                    f = 0.5 * c * np.sqrt((m/a)**2 + (n/b)**2 + (p/d)**2)
                    resonances.append(('TE', m, n, p, f / 1e9))

    # Remove duplicates (TE and TM can share the same frequency)
    seen = set()
    unique = []
    for mode in resonances:
        key = (mode[1], mode[2], mode[3])
        if key not in seen:
            seen.add(key)
            unique.append(mode)

    unique.sort(key=lambda x: x[4])
    return unique

# Cavity dimensions
a_cav = 30e-3   # 30 mm
b_cav = 15e-3   # 15 mm
d_cav = 25e-3   # 25 mm

resonances = cavity_resonances(a_cav, b_cav, d_cav)

print(f"Rectangular Cavity: {a_cav*1e3:.0f} x {b_cav*1e3:.0f} x {d_cav*1e3:.0f} mm")
print("=" * 50)
print(f"{'Mode (m,n,p)':<15} {'f (GHz)':>10}")
print("-" * 50)
for mode in resonances[:10]:
    print(f"({mode[1]},{mode[2]},{mode[3]}){'':<10} {mode[4]:>10.3f}")
```

---

## 6. 광섬유 (간략 소개)

### 6.1 계단형 굴절률 광섬유

광섬유(optical fiber)는 클래딩(cladding, $n_2 < n_1$)으로 둘러싸인 코어(core, 굴절률 $n_1$)로 구성된다. 빛은 전반사에 의해 갇힌다. 핵심 파라미터는 **개구수(numerical aperture, NA)**이다:

$$\text{NA} = \sqrt{n_1^2 - n_2^2} = n_1 \sin\theta_{\max}$$

여기서 $\theta_{\max}$는 최대 수광각이다.

### 6.2 V-수와 모드 수

**V-수(V-number)**는 광섬유가 지원하는 모드 수를 결정한다:

$$V = \frac{2\pi a}{\lambda} \text{NA} = \frac{2\pi a}{\lambda}\sqrt{n_1^2 - n_2^2}$$

여기서 $a$는 코어 반경이다. $V < 2.405$이면 기본 LP$_{01}$ 모드만 전파한다(단일 모드 광섬유). 큰 $V$에 대한 근사 모드 수는:

$$N_{\text{modes}} \approx \frac{V^2}{2}$$

### 6.3 광섬유 분산

광섬유의 전체 분산은 두 가지 기여로 구성된다:

- **재료 분산(material dispersion)**: 굴절률이 파장에 따라 변한다(레슨 10의 로렌츠 모델)
- **도파관 분산(waveguide dispersion)**: 모드의 유효 굴절률이 비율 $a/\lambda$에 따라 달라진다

표준 실리카 광섬유의 영(zero) 분산 파장은 1.3 $\mu$m 근처이며, 초기 광섬유 시스템이 이 파장에서 동작한 이유다.

```python
def fiber_modes_and_na(n_core, n_clad, core_radius, wavelength):
    """
    Compute fiber parameters: NA, V-number, and number of modes.

    Why these parameters: NA determines coupling efficiency from a source,
    V-number determines single-mode vs multimode operation, and the
    number of modes affects bandwidth (modal dispersion).
    """
    NA = np.sqrt(n_core**2 - n_clad**2)
    V = 2 * np.pi * core_radius / wavelength * NA
    N_modes = max(1, int(V**2 / 2))

    theta_max = np.arcsin(NA)

    print(f"Optical Fiber Parameters")
    print(f"========================")
    print(f"Core index:    {n_core:.4f}")
    print(f"Cladding index: {n_clad:.4f}")
    print(f"Core radius:   {core_radius*1e6:.1f} μm")
    print(f"Wavelength:    {wavelength*1e9:.0f} nm")
    print(f"NA:            {NA:.4f}")
    print(f"V-number:      {V:.3f}")
    print(f"Single-mode:   {'Yes' if V < 2.405 else 'No'}")
    print(f"Approx modes:  {N_modes}")
    print(f"Max acceptance angle: {np.degrees(theta_max):.1f}°")

    return NA, V, N_modes

# Standard single-mode fiber (SMF-28)
fiber_modes_and_na(n_core=1.4681, n_clad=1.4629,
                   core_radius=4.1e-6, wavelength=1550e-9)

print()

# Multimode fiber
fiber_modes_and_na(n_core=1.480, n_clad=1.460,
                   core_radius=25e-6, wavelength=850e-9)
```

---

## 7. 원형 도파관 (간략 개요)

### 7.1 모드 구조

원통 좌표계$(r, \phi, z)$에서 해는 베셀 함수(Bessel function)를 포함한다. TE$_{mn}$ 및 TM$_{mn}$ 모드의 차단 주파수는:

$$f_c^{\text{TE}} = \frac{x'_{mn}}{2\pi a \sqrt{\mu\epsilon}}, \quad f_c^{\text{TM}} = \frac{x_{mn}}{2\pi a \sqrt{\mu\epsilon}}$$

여기서 $x_{mn}$은 $J_m(x)$의 $n$번째 영점(zero)이고, $x'_{mn}$은 $J'_m(x)$의 $n$번째 영점이다.

| 모드 | $x_{mn}$ 또는 $x'_{mn}$ | 상대 차단 주파수 |
|------|------------------------|------------------|
| TE$_{11}$ | $x'_{11} = 1.841$ | 1.000 (우세 모드) |
| TM$_{01}$ | $x_{01} = 2.405$ | 1.306 |
| TE$_{21}$ | $x'_{21} = 3.054$ | 1.659 |
| TM$_{11}$ | $x_{11} = 3.832$ | 2.081 |
| TE$_{01}$ | $x'_{01} = 3.832$ | 2.081 |

TE$_{01}$ 모드는 특별하다: 감쇠(attenuation)가 주파수에 따라 **감소**하여 장거리 마이크로파 전송에 매력적이다(단, 모드 변환 문제로 실용화가 제한되었다).

---

## 8. 도파관 감쇠

### 8.1 손실 원인

실제 도파관은 유한한 벽 전도율을 가져 옴 손실이 발생한다. 감쇠 상수(attenuation constant)는:

$$\alpha = \frac{P_{\text{loss per unit length}}}{2 P_{\text{transmitted}}}$$

TE$_{10}$ 모드에 대해:

$$\alpha = \frac{R_s}{a^3 b k \eta} \left(2b\pi^2 + a^3 k^2\right) \cdot \frac{1}{\sqrt{1 - (f_c/f)^2}}$$

여기서 $R_s = \sqrt{\pi f \mu / \sigma}$는 표면 저항이고 $\eta = \sqrt{\mu/\epsilon}$는 고유 임피던스(intrinsic impedance)이다.

주요 특징:
- 차단 주파수에서 감쇠는 무한대이다($f \to f_c$)
- 중간 주파수에서 감쇠가 최솟값을 가진다
- 고차 모드(higher-order mode)는 일반적으로 감쇠가 더 크다

---

## 요약

| 개념 | 핵심 공식 | 물리적 의미 |
|------|-----------|-------------|
| 차단 주파수 | $f_c^{mn} = \frac{c}{2}\sqrt{(m/a)^2 + (n/b)^2}$ | 모드 전파를 위한 최소 주파수 |
| 분산 관계 | $\omega^2 = \omega_c^2 + k_z^2 c^2$ | 질량 있는 입자와 유사한 분산 |
| 위상 속도 | $v_p = c/\sqrt{1 - (f_c/f)^2}$ | 항상 $> c$ |
| 군 속도 | $v_g = c\sqrt{1 - (f_c/f)^2}$ | 항상 $< c$, 에너지 전달 |
| $v_p \cdot v_g$ 관계 | $v_p \cdot v_g = c^2$ | 기하 평균이 $c$와 같음 |
| 공동 공진 | $f_{mnp} = \frac{c}{2}\sqrt{(m/a)^2 + (n/b)^2 + (p/d)^2}$ | 3차원 정상파 |
| 품질 인수 | $Q = \omega_0 W / P_{\text{loss}}$ | 에너지 저장 효율 |
| 광섬유 V-수 | $V = (2\pi a/\lambda)\sqrt{n_1^2 - n_2^2}$ | 모드 수 파라미터 |

---

## 연습 문제

### 연습 문제 1: WR-284 도파관
WR-284 도파관의 치수는 $a = 72.14$ mm, $b = 34.04$ mm이다. (a) 처음 8개 모드의 차단 주파수를 계산하고 TE 또는 TM으로 구분하라. (b) 사용 가능한 단일 모드 주파수 범위를 구하라. (c) 3 GHz에서 TE$_{10}$ 모드의 위상 속도, 군 속도, 도파관 파장을 계산하라.

### 연습 문제 2: 마이크로파 공동 설계
기본 TE$_{101}$ 모드가 정확히 2.45 GHz(전자레인지 주파수)에서 공진하는 직사각형 공동 공진기를 설계하라. $a > d > b$인 치수 $a$, $b$, $d$를 선택하라. (a) 구리 벽($\sigma = 5.96 \times 10^7$ S/m)을 가정하여 Q-인수를 계산하라. (b) 소스가 꺼진 후 공동에 에너지가 얼마나 오래 남아 있는가($\tau = Q / \omega_0$)?

### 연습 문제 3: 단일 모드 광섬유 설계
1310 nm에서 동작하는 계단형 굴절률 단일 모드 광섬유를 설계하라. 코어와 클래딩 굴절률은 $n_1 = 1.468$, $n_2 = 1.463$이다. (a) 단일 모드 동작($V < 2.405$)을 위한 최대 코어 반경을 구하라. (b) NA와 수광각을 계산하라. (c) 이 광섬유를 850 nm에서 사용하면 몇 개의 모드가 지원되는가?

### 연습 문제 4: 모드 시각화
$a = 2b$인 직사각형 도파관에서 TE$_{21}$ 모드와 TM$_{21}$ 모드를 시각화하는 Python 프로그램을 작성하라. 전기장 벡터 패턴과 자기장 선 모두를 플롯하라. 전기장이 최대가 되는 위치와 0이 되는 위치를 확인하라.

### 연습 문제 5: 원형 도파관 모드
반경 $a = 15$ mm인 원형 도파관에서 첫 6개 모드의 차단 주파수를 계산하라. 베셀 함수를 이용해 TE$_{11}$ 모드의 반경 방향 전기장 패턴을 플롯하라. 비슷한 면적의 직사각형 도파관과 단일 모드 대역폭을 비교하라.

---

[← 이전: 11. 반사와 굴절](11_Reflection_and_Refraction.md) | [다음: 13. 복사와 안테나 →](13_Radiation_and_Antennas.md)
