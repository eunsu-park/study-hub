# 진공에서의 전자기파

[← 이전: 08. 맥스웰 방정식 — 적분 형태](08_Maxwells_Equations_Integral.md)

---

## 학습 목표

1. 맥스웰 방정식의 평면파 해를 유도하고, 네 가지 방정식을 모두 만족함을 확인한다
2. 횡파 조건(Transversality Condition) $\mathbf{E} \perp \mathbf{B} \perp \mathbf{k}$ 를 증명한다
3. 선형, 원형, 타원형 편광(Polarization)을 설명하고 수학적으로 표현한다
4. 스토크스 매개변수(Stokes Parameters)를 정의하고, 임의의 편광 상태를 기술하는 데 활용한다
5. 전자기파의 에너지, 운동량, 복사 압력을 계산한다
6. 자유 공간 임피던스(Impedance of Free Space)를 정의하고 그 물리적 의미를 설명한다
7. Python으로 편광 상태와 에너지 전달을 수치적으로 시뮬레이션한다

---

맥스웰의 이론을 불멸로 만든 물리적 예언, 전자기파에 드디어 도달했다. 7강에서 유도한 파동 방정식은 진동하는 전기장과 자기장이 진공을 빛의 속도로 전파하는 해를 허용한다. 이 파동은 에너지, 운동량, 그리고 정보를 운반한다. 빛, 라디오파, X선, 그 밖의 모든 전자기 복사가 바로 이것이다. 이 강의에서는 진공 속 평면파의 상세한 성질 — 구조, 편광, 에너지론 — 을 전개하며, 파동 광학과 복사 이론 전체의 기초를 마련한다.

---

## 평면파 해

진공에서의 파동 방정식은 다음과 같다:

$$\nabla^2 \mathbf{E} = \mu_0\epsilon_0 \frac{\partial^2 \mathbf{E}}{\partial t^2} = \frac{1}{c^2}\frac{\partial^2 \mathbf{E}}{\partial t^2}$$

가장 단순한 해는 **단색 평면파(Monochromatic Plane Wave)** 다 — 전파 방향에 수직인 무한한 평면 위에서 균일하게 진동하는 사인파다.

$\hat{z}$ 방향으로 전파하는 파동:

$$\tilde{\mathbf{E}} = \tilde{E}_0 \, e^{i(kz - \omega t)}$$

여기서:
- $\tilde{E}_0$ 는 복소 진폭 벡터(진폭과 위상을 동시에 인코딩)
- $k = 2\pi/\lambda$ 는 파수(Wave Number)
- $\omega = 2\pi f$ 는 각주파수(Angular Frequency)
- 분산 관계(Dispersion Relation): $\omega = ck$

일반적으로 $\hat{k}$ 방향으로 전파할 때:

$$\tilde{\mathbf{E}}(\mathbf{r}, t) = \tilde{\mathbf{E}}_0 \, e^{i(\mathbf{k}\cdot\mathbf{r} - \omega t)}$$

여기서 $\mathbf{k}$ 는 **파동 벡터(Wave Vector)** (전파 방향을 가리키며, 크기는 $k = \omega/c$) 다.

물리적 장(Field)은 실수부를 취한다:

$$\mathbf{E} = \text{Re}(\tilde{\mathbf{E}}) = E_0 \cos(\mathbf{k}\cdot\mathbf{r} - \omega t + \phi)$$

> **비유**: 평면파는 수영장에 아주 긴 막대를 수평으로 떨어뜨렸을 때 생기는 파동과 같다. 파마루가 평행한 선(3D에서는 평면)을 이루며 함께 앞으로 나아간다. 주어진 파마루 위의 모든 점은 일제히 진동한다 — 이것이 평면파의 "평면"이다.

---

## 횡파성: E, B, 그리고 k

맥스웰 방정식은 $\mathbf{E}$ 와 $\mathbf{B}$ 가 $\mathbf{k}$ 에 대해 어떤 방향을 가져야 하는지를 강하게 제약한다.

### 가우스 법칙으로부터

$$\nabla \cdot \mathbf{E} = 0 \implies i\mathbf{k} \cdot \tilde{\mathbf{E}}_0 = 0 \implies \mathbf{k} \cdot \mathbf{E}_0 = 0$$

$\mathbf{E}$ 는 $\mathbf{k}$ 에 수직이다 — 전기장은 전파 방향에 횡(Transverse)으로 진동한다.

### 패러데이 법칙으로부터

$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t} \implies i\mathbf{k} \times \tilde{\mathbf{E}}_0 = i\omega \tilde{\mathbf{B}}_0$$

$$\tilde{\mathbf{B}}_0 = \frac{1}{\omega}\mathbf{k} \times \tilde{\mathbf{E}}_0 = \frac{1}{c}\hat{k} \times \tilde{\mathbf{E}}_0$$

이로부터 알 수 있는 것:
1. $\mathbf{B}$ 도 $\mathbf{k}$ 에 수직이다 (횡파)
2. $\mathbf{B}$ 는 $\mathbf{E}$ 에 수직이다
3. $|\mathbf{B}| = |\mathbf{E}|/c$

따라서 세 벡터 $\mathbf{E}$, $\mathbf{B}$, $\mathbf{k}$ 는 **오른손 좌표계의 직교 삼조(Right-Handed Orthogonal Triad)** 를 이룬다:

$$\boxed{\mathbf{E} \perp \mathbf{B} \perp \mathbf{k}, \qquad B_0 = E_0/c, \qquad \hat{k} = \hat{E} \times \hat{B}}$$

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Visualize a plane EM wave: E in x, B in y, propagation in z
# Why 3D: the orthogonal structure of E, B, k demands three dimensions

c = 1.0            # normalized speed of light
lambda_0 = 1.0     # wavelength
k = 2 * np.pi / lambda_0
omega = c * k
E0 = 1.0
B0 = E0 / c

z = np.linspace(0, 3 * lambda_0, 500)
t = 0  # snapshot at t = 0

Ex = E0 * np.sin(k * z - omega * t)
By = B0 * np.sin(k * z - omega * t)

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot E field (oscillates in x-direction)
ax.plot(z, Ex, np.zeros_like(z), 'b-', linewidth=2, label='E (x-direction)')

# Plot B field (oscillates in y-direction)
ax.plot(z, np.zeros_like(z), By, 'r-', linewidth=2, label='B (y-direction)')

# Draw arrows at selected points to show vector nature
N_arrows = 30
z_arrows = np.linspace(0, 3*lambda_0, N_arrows)
for zi in z_arrows:
    ex = E0 * np.sin(k * zi)
    by = B0 * np.sin(k * zi)
    # Why arrows: they show that E and B are perpendicular at every point
    ax.quiver(zi, 0, 0, 0, ex, 0, color='blue', alpha=0.3, arrow_length_ratio=0.3)
    ax.quiver(zi, 0, 0, 0, 0, by, color='red', alpha=0.3, arrow_length_ratio=0.3)

# Propagation direction arrow
ax.quiver(0, 0, 0, 0.5, 0, 0, color='green', linewidth=3, arrow_length_ratio=0.3)
ax.text(0.3, 0, 0.15, 'k', fontsize=14, color='green', fontweight='bold')

ax.set_xlabel('z (propagation)')
ax.set_ylabel('x (E-field)')
ax.set_zlabel('y (B-field)')
ax.set_title('Plane Electromagnetic Wave: E ⊥ B ⊥ k', fontsize=14)
ax.legend(loc='upper right')

# Adjust view angle for best visualization
ax.view_init(elev=20, azim=-60)
plt.tight_layout()
plt.savefig('plane_wave_3d.png', dpi=150)
plt.show()

# Verify transversality numerically
print("Transversality verification:")
print(f"E · k = E_x * k_z = {E0} * 0 = 0  (E ⊥ k) ✓")
print(f"B · k = B_y * k_z = {B0} * 0 = 0  (B ⊥ k) ✓")
print(f"E · B = E_x * B_y = varies, but E_vec · B_vec = 0  (E ⊥ B) ✓")
print(f"|B₀|/|E₀| = {B0/E0} = 1/c ✓")
```

---

## 편광

전자기파의 **편광(Polarization)** 은 전기장 벡터가 전파 방향에 수직인 평면 위에서 그리는 궤적을 말한다.

$\hat{z}$ 방향으로 전파하는 파동에서 가장 일반적인 전기장은:

$$\mathbf{E}(z,t) = E_{0x}\cos(kz - \omega t)\hat{x} + E_{0y}\cos(kz - \omega t + \delta)\hat{y}$$

여기서 $\delta$ 는 $x$ 성분과 $y$ 성분 사이의 위상 차이다.

### 선형 편광(Linear Polarization) ($\delta = 0$ 또는 $\pi$)

$$\mathbf{E} = (E_{0x}\hat{x} + E_{0y}\hat{y})\cos(kz - \omega t)$$

$\mathbf{E}$ 의 끝점은 고정된 직선을 따라 앞뒤로 진동한다. 편광 방향은 $x$ 축과 $\alpha = \arctan(E_{0y}/E_{0x})$ 의 각도를 이룬다.

### 원형 편광(Circular Polarization) ($\delta = \pm\pi/2$, $E_{0x} = E_{0y}$)

$$\mathbf{E} = E_0[\cos(kz-\omega t)\hat{x} \mp \sin(kz-\omega t)\hat{y}]$$

$\mathbf{E}$ 의 끝점이 원을 그린다. 관례:
- $\delta = -\pi/2$: **우원형(Right Circular)** (파동 진행 방향에서 볼 때 시계 방향)
- $\delta = +\pi/2$: **좌원형(Left Circular)** (파동 진행 방향에서 볼 때 반시계 방향)

### 타원형 편광(Elliptical Polarization) (일반적인 $\delta$, 임의의 진폭)

가장 일반적인 경우다. $\mathbf{E}$ 의 끝점이 타원을 그린다. 선형 편광과 원형 편광은 이것의 특수한 경우다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize different polarization states
# Why multiple panels: comparing states side by side reveals the pattern

omega = 2 * np.pi
t = np.linspace(0, 1, 500)   # one period

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

polarization_cases = [
    ('Linear (horizontal)', 1.0, 0.0, 0.0),
    ('Linear (45°)', 1.0, 1.0, 0.0),
    ('Linear (vertical)', 0.0, 1.0, 0.0),
    ('Right Circular', 1.0, 1.0, -np.pi/2),
    ('Left Circular', 1.0, 1.0, np.pi/2),
    ('Elliptical', 1.0, 0.5, np.pi/4),
]

for idx, (name, E0x, E0y, delta) in enumerate(polarization_cases):
    ax = axes[idx // 3, idx % 3]

    Ex = E0x * np.cos(omega * t)
    Ey = E0y * np.cos(omega * t + delta)

    # Why color by time: it shows the direction of rotation
    colors = plt.cm.viridis(t / t[-1])
    for i in range(len(t) - 1):
        ax.plot([Ex[i], Ex[i+1]], [Ey[i], Ey[i+1]], color=colors[i], linewidth=2)

    # Mark starting point
    ax.plot(Ex[0], Ey[0], 'ro', markersize=8, zorder=5)
    # Arrow showing direction at t=0
    if idx >= 3 and E0y > 0:
        ax.annotate('', xy=(Ex[10], Ey[10]), xytext=(Ex[0], Ey[0]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel('$E_x$')
    ax.set_ylabel('$E_y$')
    ax.set_title(f'{name}\n$E_{{0x}}$={E0x}, $E_{{0y}}$={E0y}, δ={delta/np.pi:.2f}π')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', alpha=0.3)
    ax.axvline(x=0, color='gray', alpha=0.3)

plt.suptitle('Polarization States of EM Waves', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('polarization_states.png', dpi=150)
plt.show()
```

---

## 스토크스 매개변수

**스토크스 매개변수(Stokes Parameters)** 는 부분 편광된 빛을 포함하여 편광 상태를 완전하게 기술하는 방법이다. 복소 진폭 $\tilde{E}_x = E_{0x}$, $\tilde{E}_y = E_{0y}e^{i\delta}$ 를 갖는 파동에 대해:

$$S_0 = E_{0x}^2 + E_{0y}^2 \qquad \text{(전체 세기)}$$
$$S_1 = E_{0x}^2 - E_{0y}^2 \qquad \text{(수평 대 수직 선호도)}$$
$$S_2 = 2E_{0x}E_{0y}\cos\delta \qquad \text{(+45° 대 -45° 선호도)}$$
$$S_3 = 2E_{0x}E_{0y}\sin\delta \qquad \text{(우원형 대 좌원형 선호도)}$$

완전 편광인 경우: $S_0^2 = S_1^2 + S_2^2 + S_3^2$.

부분 편광된 빛의 **편광도(Degree of Polarization)**: $\Pi = \frac{\sqrt{S_1^2+S_2^2+S_3^2}}{S_0}$, $0 \leq \Pi \leq 1$.

### 푸앵카레 구

정규화된 스토크스 매개변수 $(S_1/S_0, S_2/S_0, S_3/S_0)$ 는 **푸앵카레 구(Poincare Sphere)** 위의 한 점을 정의한다:
- 북극: 우원형 편광
- 남극: 좌원형 편광
- 적도: 선형 편광들
- 내부: 부분 편광

```python
import numpy as np
import matplotlib.pyplot as plt

# Stokes parameters for various polarization states
# Why Stokes: they are directly measurable (unlike complex amplitudes)

def stokes_parameters(E0x, E0y, delta):
    """Compute Stokes parameters for a fully polarized wave."""
    S0 = E0x**2 + E0y**2
    S1 = E0x**2 - E0y**2
    S2 = 2 * E0x * E0y * np.cos(delta)
    S3 = 2 * E0x * E0y * np.sin(delta)
    return S0, S1, S2, S3

# Define several polarization states
states = {
    'H (horizontal)':    (1.0, 0.0, 0.0),
    'V (vertical)':      (0.0, 1.0, 0.0),
    '+45° linear':       (1.0, 1.0, 0.0),
    '-45° linear':       (1.0, 1.0, np.pi),
    'Right circular':    (1.0, 1.0, -np.pi/2),
    'Left circular':     (1.0, 1.0, np.pi/2),
    'Elliptical (1)':    (1.0, 0.5, np.pi/4),
    'Elliptical (2)':    (0.8, 0.6, np.pi/3),
}

print(f"{'State':<22} {'S₀':>6} {'S₁':>6} {'S₂':>6} {'S₃':>6}  {'Check':>8}")
print("=" * 70)

for name, (E0x, E0y, delta) in states.items():
    S0, S1, S2, S3 = stokes_parameters(E0x, E0y, delta)
    check = np.sqrt(S1**2 + S2**2 + S3**2)
    # Why check: for fully polarized light, √(S₁²+S₂²+S₃²) = S₀
    print(f"{name:<22} {S0:6.2f} {S1:6.2f} {S2:6.2f} {S3:6.2f}  {check:8.4f}")

# Poincare sphere visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Draw sphere
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(xs, ys, zs, alpha=0.1, color='lightblue')

# Plot polarization states on the sphere
colors = plt.cm.tab10(np.linspace(0, 1, len(states)))
for idx, (name, (E0x, E0y, delta)) in enumerate(states.items()):
    S0, S1, S2, S3 = stokes_parameters(E0x, E0y, delta)
    if S0 > 0:
        # Normalize to unit sphere
        s1, s2, s3 = S1/S0, S2/S0, S3/S0
        ax.scatter(s1, s2, s3, color=colors[idx], s=100, zorder=5)
        ax.text(s1*1.15, s2*1.15, s3*1.15, name, fontsize=8, color=colors[idx])

# Label poles and equator
ax.set_xlabel('$S_1/S_0$')
ax.set_ylabel('$S_2/S_0$')
ax.set_zlabel('$S_3/S_0$')
ax.set_title('Poincare Sphere', fontsize=14)

plt.tight_layout()
plt.savefig('poincare_sphere.png', dpi=150)
plt.show()
```

---

## 전자기파의 에너지와 운동량

### 에너지 밀도

평면파의 순간 에너지 밀도:

$$u = \frac{1}{2}\left(\epsilon_0 E^2 + \frac{B^2}{\mu_0}\right)$$

$B = E/c$ 이고 $c = 1/\sqrt{\mu_0\epsilon_0}$ 인 평면파에 대해:

$$\frac{B^2}{\mu_0} = \frac{E^2}{\mu_0 c^2} = \epsilon_0 E^2$$

따라서 전기 에너지와 자기 에너지의 기여는 **정확히 같다**:

$$u = \epsilon_0 E^2 = \frac{B^2}{\mu_0}$$

> **비유**: 역학적 파동(예: 진동하는 줄)에서는 운동 에너지와 퍼텐셜 에너지가 평균적으로 같다. 전자기파에서는 전기 에너지 밀도와 자기 에너지 밀도가 이 역할을 맡는다 — 파동 에너지를 똑같이 나눠 갖는 완벽한 파트너들이다.

### 시간 평균 에너지 밀도

$$\langle u \rangle = \frac{1}{2}\epsilon_0 E_0^2$$

### 포인팅 벡터와 세기

$$\mathbf{S} = \frac{1}{\mu_0}\mathbf{E}\times\mathbf{B} = \frac{E^2}{\mu_0 c}\hat{k} = cu\,\hat{k}$$

세기(Intensity, 단위 넓이당 시간 평균 전력):

$$I = \langle|\mathbf{S}|\rangle = \frac{1}{2}\frac{E_0^2}{\mu_0 c} = \frac{1}{2}\epsilon_0 c E_0^2 = \frac{c}{2\mu_0}B_0^2$$

### 운동량

전자기파는 운동량을 운반한다. 운동량 밀도:

$$\mathbf{g} = \frac{\mathbf{S}}{c^2} = \frac{u}{c}\hat{k}$$

세기 $I$ 의 파동이 표면에 부딪힐 때 전달되는 운동량:

$$\text{복사 압력 (흡수)} = \frac{I}{c}$$
$$\text{복사 압력 (완전 반사)} = \frac{2I}{c}$$

```python
import numpy as np

# Energy and momentum of electromagnetic waves — practical calculations
# Why real numbers: connecting abstract formulas to tangible quantities

c = 3e8
mu_0 = 4 * np.pi * 1e-7
epsilon_0 = 8.854e-12

print("Electromagnetic Wave Properties")
print("=" * 60)

# Example 1: Sunlight
I_sun = 1361  # W/m² (solar constant at Earth)
E0_sun = np.sqrt(2 * I_sun / (epsilon_0 * c))
B0_sun = E0_sun / c
P_abs_sun = I_sun / c
P_ref_sun = 2 * I_sun / c

print(f"\n1. Sunlight at Earth (I = {I_sun} W/m²)")
print(f"   E₀ = {E0_sun:.1f} V/m")
print(f"   B₀ = {B0_sun*1e6:.3f} μT")
print(f"   Radiation pressure (absorbing): {P_abs_sun*1e6:.3f} μPa")
print(f"   Radiation pressure (reflecting): {P_ref_sun*1e6:.3f} μPa")
print(f"   Force on 10×10 m sail: {P_ref_sun * 100:.6f} N")

# Example 2: Laser pointer
P_laser = 5e-3   # 5 mW
A_spot = np.pi * (0.5e-3)**2   # 0.5 mm radius spot
I_laser = P_laser / A_spot
E0_laser = np.sqrt(2 * I_laser / (epsilon_0 * c))
B0_laser = E0_laser / c

print(f"\n2. 5 mW Laser Pointer (spot radius 0.5 mm)")
print(f"   Intensity: I = {I_laser:.1f} W/m²")
print(f"   E₀ = {E0_laser:.1f} V/m")
print(f"   B₀ = {B0_laser*1e6:.3f} μT")

# Example 3: Cell phone signal
P_phone = 1.0   # 1 W transmitted power
d = 1.0          # 1 m distance
I_phone = P_phone / (4 * np.pi * d**2)   # isotropic radiation
E0_phone = np.sqrt(2 * I_phone / (epsilon_0 * c))

print(f"\n3. Cell Phone (1 W, 1 m away)")
print(f"   Intensity: I = {I_phone:.2f} W/m²")
print(f"   E₀ = {E0_phone:.2f} V/m")

# Example 4: Microwave oven
P_oven = 1000    # 1 kW
A_oven = 0.3 * 0.3  # 30 cm × 30 cm cavity
I_oven = P_oven / A_oven
E0_oven = np.sqrt(2 * I_oven / (epsilon_0 * c))

print(f"\n4. Microwave Oven (1 kW, 30×30 cm)")
print(f"   Intensity: I = {I_oven:.0f} W/m²")
print(f"   E₀ = {E0_oven:.0f} V/m")

# Verify energy equipartition
print(f"\n--- Energy Equipartition Verification ---")
print(f"For any plane wave:")
print(f"  Electric energy density: u_E = ε₀E²/2")
print(f"  Magnetic energy density: u_B = B²/(2μ₀) = E²/(2μ₀c²) = ε₀E²/2 = u_E  ✓")
print(f"  The two contributions are EXACTLY equal!")
```

---

## 자유 공간 임피던스

전기장 진폭 대 자기장 진폭의 비가 **자유 공간 임피던스(Impedance of Free Space)** 를 정의한다:

$$Z_0 = \frac{E_0}{H_0} = \frac{E_0}{B_0/\mu_0} = \mu_0 c = \sqrt{\frac{\mu_0}{\epsilon_0}}$$

$$\boxed{Z_0 = \sqrt{\frac{\mu_0}{\epsilon_0}} \approx 376.73 \; \Omega}$$

이것은 자연의 기본 상수다. 전자기파 전파에서 전송선로(Transmission Line)의 특성 임피던스(Characteristic Impedance)와 유사한 역할을 한다.

평면파의 세기를 $Z_0$ 로 우아하게 쓸 수 있다:

$$I = \frac{E_0^2}{2Z_0}$$

유전율 $\epsilon$ 과 투자율 $\mu$ 를 가진 매질에서:

$$Z = \sqrt{\frac{\mu}{\epsilon}} = \frac{Z_0}{\sqrt{\epsilon_r \mu_r}}$$

파동이 서로 다른 임피던스의 매질 경계를 지날 때 부분 반사가 일어난다 — 마치 줄의 파동이 밀도 변화를 만날 때와 같다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Impedance of free space and its role in wave propagation
# Why impedance: it determines reflection and transmission at interfaces

mu_0 = 4 * np.pi * 1e-7
epsilon_0 = 8.854e-12
c = 1 / np.sqrt(mu_0 * epsilon_0)

Z_0 = np.sqrt(mu_0 / epsilon_0)
print(f"Impedance of free space: Z₀ = {Z_0:.4f} Ω")
print(f"                         Z₀ ≈ 120π = {120*np.pi:.4f} Ω")

# Impedance and reflection at interfaces
# Reflection coefficient: r = (Z₂ - Z₁)/(Z₂ + Z₁)
# Transmission coefficient: t = 2Z₂/(Z₂ + Z₁)
# Why these formulas: they follow from matching boundary conditions at the interface

materials = {
    'Vacuum': 1.0,
    'Air': 1.0006,
    'Glass (n=1.5)': 1.5**2,
    'Water (n=1.33)': 1.33**2,
    'Silicon (n=3.42)': 3.42**2,
    'Diamond (n=2.42)': 2.42**2,
}

print(f"\n{'Material':<22} {'εᵣ':>6} {'Z (Ω)':>10} {'r (from vacuum)':>16} {'R (%)':>8}")
print("=" * 70)

Z1 = Z_0  # incoming wave in vacuum
for name, eps_r in materials.items():
    Z2 = Z_0 / np.sqrt(eps_r)
    r = (Z2 - Z1) / (Z2 + Z1)     # amplitude reflection coefficient
    R = r**2                         # power reflectance
    print(f"{name:<22} {eps_r:6.3f} {Z2:10.2f} {r:16.4f} {R*100:8.2f}")

# Visualize reflection at glass interface
n_glass = 1.5
Z_glass = Z_0 / n_glass
r = (Z_glass - Z_0) / (Z_glass + Z_0)   # negative: phase flip on reflection
t = 2 * Z_glass / (Z_glass + Z_0)

print(f"\nNormal incidence reflection at air-glass interface:")
print(f"  r = {r:.4f} (negative = π phase shift)")
print(f"  t = {t:.4f}")
print(f"  R = {r**2*100:.2f}%")
print(f"  T = {(1-r**2)*100:.2f}%")
print(f"  R + T = {(r**2 + 1-r**2)*100:.1f}% (energy conservation) ✓")

# Plot: time snapshot of incident, reflected, and transmitted waves
z = np.linspace(-10, 10, 1000)
k1 = 1.0           # wave number in medium 1
k2 = n_glass * k1   # wave number in medium 2

# Incident wave (z < 0)
E_inc = np.where(z < 0, np.sin(k1 * z), 0)

# Reflected wave (z < 0, traveling backward)
E_ref = np.where(z < 0, r * np.sin(-k1 * z), 0)

# Transmitted wave (z > 0)
E_trans = np.where(z >= 0, t * np.sin(k2 * z), 0)

# Total field
E_total = E_inc + E_ref + E_trans

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(z, E_inc, 'b-', linewidth=1.5, label='Incident')
axes[0].plot(z, E_ref, 'r-', linewidth=1.5, label='Reflected')
axes[0].plot(z, E_trans, 'g-', linewidth=1.5, label='Transmitted')
axes[0].axvline(x=0, color='gray', linewidth=3, alpha=0.5)
axes[0].fill_between([0, 10], [-1.5, -1.5], [1.5, 1.5], alpha=0.05, color='blue')
axes[0].text(-5, 1.2, 'Vacuum', fontsize=12)
axes[0].text(3, 1.2, f'Glass (n={n_glass})', fontsize=12)
axes[0].set_ylabel('E')
axes[0].set_title('Individual Waves at Interface')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(z, E_total, 'k-', linewidth=2)
axes[1].axvline(x=0, color='gray', linewidth=3, alpha=0.5)
axes[1].fill_between([0, 10], [-1.5, -1.5], [1.5, 1.5], alpha=0.05, color='blue')
axes[1].set_xlabel('z')
axes[1].set_ylabel('E')
axes[1].set_title('Total Field (standing wave pattern in medium 1)')
axes[1].grid(True, alpha=0.3)

plt.suptitle(f'Reflection at Dielectric Interface (R = {r**2*100:.1f}%)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wave_reflection.png', dpi=150)
plt.show()
```

---

## 전자기 스펙트럼

모든 전자기파는 동일한 근본적 성질을 공유한다 — 다른 점은 오직 주파수(혹은 파장)뿐이다:

$$c = f\lambda$$

| 영역 | 주파수 | 파장 | 발생원 |
|---|---|---|---|
| 라디오파 | < 300 MHz | > 1 m | 안테나 |
| 마이크로파 | 300 MHz - 300 GHz | 1 mm - 1 m | 클라이스트론, 마그네트론 |
| 적외선 | 300 GHz - 430 THz | 700 nm - 1 mm | 열복사 |
| 가시광선 | 430 - 750 THz | 400 - 700 nm | 원자, 분자 |
| 자외선 | 750 THz - 30 PHz | 10 - 400 nm | 고온별, 방전 |
| X선 | 30 PHz - 30 EHz | 0.01 - 10 nm | 내각 전자 |
| 감마선 | > 30 EHz | < 0.01 nm | 핵전이 |

```python
import numpy as np
import matplotlib.pyplot as plt

# The electromagnetic spectrum — frequencies and wavelengths
# Why log scale: the spectrum spans ~20 orders of magnitude

c = 3e8

# Define spectrum regions
regions = [
    ('Radio', 1e3, 3e8, 'red'),
    ('Microwave', 3e8, 3e11, 'orange'),
    ('Infrared', 3e11, 4.3e14, 'darkred'),
    ('Visible', 4.3e14, 7.5e14, 'green'),
    ('UV', 7.5e14, 3e16, 'purple'),
    ('X-ray', 3e16, 3e19, 'blue'),
    ('Gamma', 3e19, 3e22, 'black'),
]

fig, ax = plt.subplots(figsize=(14, 4))

for name, f_low, f_high, color in regions:
    ax.barh(0, np.log10(f_high) - np.log10(f_low), left=np.log10(f_low),
            height=0.6, color=color, alpha=0.6, edgecolor='black')
    f_mid = np.sqrt(f_low * f_high)
    ax.text(np.log10(f_mid), 0, name, ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')

ax.set_xlabel('log₁₀(frequency / Hz)')
ax.set_yticks([])
ax.set_xlim(2, 23)
ax.set_title('The Electromagnetic Spectrum', fontsize=14, fontweight='bold')

# Add wavelength axis on top
ax2 = ax.twiny()
ax2.set_xlim(2, 23)
# λ = c/f, so log₁₀(λ) = log₁₀(c) - log₁₀(f) = 8.477 - log₁₀(f)
tick_freqs = np.arange(3, 23, 2)
tick_lambdas = [f'{c/10**f:.0e}' for f in tick_freqs]
ax2.set_xticks(tick_freqs)
ax2.set_xticklabels(tick_lambdas, fontsize=8)
ax2.set_xlabel('Wavelength (m)')

plt.tight_layout()
plt.savefig('em_spectrum.png', dpi=150)
plt.show()

# Photon energy E = hf
h = 6.626e-34   # Planck's constant
print("\nPhoton energies across the spectrum:")
print(f"{'Region':<12} {'f (Hz)':>12} {'λ':>12} {'E (eV)':>12}")
print("=" * 50)
for name, f_low, f_high, _ in regions:
    f = np.sqrt(f_low * f_high)   # geometric mean
    lam = c / f
    E_eV = h * f / 1.6e-19

    if lam >= 1:
        lam_str = f"{lam:.1f} m"
    elif lam >= 1e-3:
        lam_str = f"{lam*1e3:.1f} mm"
    elif lam >= 1e-6:
        lam_str = f"{lam*1e6:.1f} μm"
    elif lam >= 1e-9:
        lam_str = f"{lam*1e9:.1f} nm"
    else:
        lam_str = f"{lam*1e12:.2f} pm"

    print(f"{name:<12} {f:12.2e} {lam_str:>12} {E_eV:12.4e}")
```

---

## 요약

| 개념 | 핵심 방정식 |
|---|---|
| 평면파 | $\tilde{\mathbf{E}} = \tilde{\mathbf{E}}_0\,e^{i(\mathbf{k}\cdot\mathbf{r}-\omega t)}$ |
| 분산 관계 | $\omega = ck$, $c = 1/\sqrt{\mu_0\epsilon_0}$ |
| 횡파성 | $\mathbf{E} \perp \mathbf{B} \perp \mathbf{k}$ |
| E-B 관계 | $\mathbf{B} = \frac{1}{c}\hat{k}\times\mathbf{E}$, $B_0 = E_0/c$ |
| 에너지 밀도 | $u = \epsilon_0 E^2 = B^2/\mu_0$ |
| 에너지 등분배 | $u_E = u_B$ (전기 = 자기) |
| 세기 | $I = \frac{1}{2}\epsilon_0 c E_0^2 = E_0^2/(2Z_0)$ |
| 포인팅 벡터 | $\mathbf{S} = cu\,\hat{k}$ |
| 복사 압력 | $P = I/c$ (흡수), $2I/c$ (반사) |
| 자유 공간 임피던스 | $Z_0 = \sqrt{\mu_0/\epsilon_0} \approx 377\;\Omega$ |
| 스토크스 매개변수 | $S_0, S_1, S_2, S_3$ 로 편광 기술 |
| 반사 계수 | $r = (Z_2-Z_1)/(Z_2+Z_1)$ |

---

## 연습 문제

### 연습 1: 파동 검증
$\mathbf{E} = E_0\cos(kz - \omega t)\hat{x}$ 와 $\mathbf{B} = (E_0/c)\cos(kz - \omega t)\hat{y}$ 가 진공에서 네 가지 맥스웰 방정식을 모두 만족함을 확인하라. 각 방정식을 명시적으로 검토하라.

### 연습 2: 원형 편광 분해
임의의 선형 편광파를 같은 진폭의 두 원형 편광파로 분해할 수 있음을 보여라. 분해를 명시적으로 쓰고 수치적으로 검증하라.

### 연습 3: 태양 돛
넓이 $A = 100$ m$^2$, 반사율 $R = 0.95$ 인 태양 돛이 태양으로부터 1 AU ($I = 1361$ W/m$^2$) 떨어진 곳에 있다. (a) 복사력, (b) 돛 질량이 1 kg일 때의 가속도, (c) 중력을 무시했을 때 화성 궤도(1.5 AU)까지 도달하는 데 걸리는 시간을 계산하라. 복사력과 태양의 중력을 비교하라.

### 연습 4: 편광자 체인
이상적인 선형 편광자 세 개를 직렬로 배치한다. 첫 번째는 $0°$, 두 번째는 $45°$, 세 번째는 $90°$ 다. 말뤼스의 법칙($I = I_0\cos^2\theta$)을 사용하여 무편광 입사광 중 세 편광자를 모두 통과하는 비율을 계산하라. $0°$ 와 $90°$ 두 편광자만 있을 때는 어떻게 되는가? 스토크스 벡터로 시뮬레이션하라.

### 연습 5: 정재파
반대 방향으로 진행하는 같은 진폭, 같은 주파수의 두 평면파가 정재파(Standing Wave)를 만든다. $\mathbf{E}$ 와 $\mathbf{B}$ 의 정재파 패턴을 유도하고, 시간 평균 포인팅 벡터가 0임을 보여라. 수치적으로 검증하고 공간 포락선을 그려라.

---

[← 이전: 08. 맥스웰 방정식 — 적분 형태](08_Maxwells_Equations_Integral.md)
