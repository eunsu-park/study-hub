# 정자기학

[← 이전: 03. 도체와 유전체](03_Conductors_and_Dielectrics.md) | [다음: 05. 자기 벡터 퍼텐셜 →](05_Magnetic_Vector_Potential.md)

---

## 학습 목표

1. 로런츠 힘 법칙을 서술하고 움직이는 전하와 전류 도선에 작용하는 자기력을 계산한다
2. 비오-사바르 법칙을 적용하여 전류 분포로부터 자기장을 계산한다
3. 앙페르 법칙을 적분 형태와 미분 형태로 유도하고 적용한다
4. $\nabla \cdot \mathbf{B} = 0$ (자기 단극자 없음)의 의미와 물리적 결과를 설명한다
5. 자기 쌍극자 모멘트(magnetic dipole moment)를 정의하고 자기 쌍극자의 장을 계산한다
6. 외부 자기장 속 자기 쌍극자에 작용하는 힘과 돌림힘을 계산한다
7. Python을 이용하여 비오-사바르 계산을 수치적으로 구현한다

---

이제 정지한 전하의 세계에서 움직이는 전하의 세계 — 전류 — 로 넘어간다. 움직이는 전하는 근본적으로 새로운 종류의 장인 자기장 $\mathbf{B}$를 만들어낸다. 전하로부터 방사형으로 뻗어나가는 전기장과 달리, 자기장은 전류 주위를 닫힌 고리 형태로 감싼다. 방사형 대 순환형이라는 이 위상학적 차이는 수학에 고스란히 담겨 있다: 정전기학에서 $\nabla \cdot \mathbf{E} = \rho/\epsilon_0$이고 $\nabla \times \mathbf{E} = 0$인 반면, 정자기학에서는 $\nabla \cdot \mathbf{B} = 0$이고 $\nabla \times \mathbf{B} = \mu_0 \mathbf{J}$이다. 모든 것이 "맞바뀌어" 있다.

---

## 로런츠 힘 법칙

자기장 $\mathbf{B}$ 속에서 속도 $\mathbf{v}$로 움직이는 전하 $q$에 작용하는 자기력은:

$$\mathbf{F}_{\text{자기}} = q\mathbf{v} \times \mathbf{B}$$

전기력과 합치면, 완전한 **로런츠 힘(Lorentz force)**이 된다:

$$\mathbf{F} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$$

자기력의 핵심 성질:
- 자기력은 항상 $\mathbf{v}$에 **수직** — 전하에 일을 하지 않는다
- 운동의 **방향**을 바꾸지만, **속력**은 바꾸지 않는다
- 균일한 $\mathbf{B}$ 장 속에서 대전 입자는 **원운동**(또는 $\mathbf{v}$의 $\mathbf{B}$ 방향 성분이 있으면 나선 운동)을 한다

### 사이클로트론 운동

균일한 자기장 $\mathbf{B}$에 수직으로 운동하는 전하 $q$의 원형 궤도 반지름과 진동수:

$$r = \frac{mv}{qB} \qquad \text{(사이클로트론 반지름)}$$

$$\omega_c = \frac{qB}{m} \qquad \text{(사이클로트론 진동수)}$$

사이클로트론 진동수는 속력에 무관하다 — 이 놀라운 사실이 사이클로트론 입자 가속기의 원리이다.

> **비유**: 자기력은 항상 당신의 진행 방향에 수직으로 미는 손과 같다. 똑바로 걷고 있을 때 누군가가 계속 옆으로(속도에 수직으로) 밀면, 당신은 원을 그리며 걷게 된다. 이 힘은 당신을 빠르게도 느리게도 하지 않는다 — 오직 경로를 구부릴 뿐이다.

### 전류 도선에 작용하는 힘

자기장 속에서 전류 $I$가 흐르는 도선에 작용하는 힘:

$$\mathbf{F} = I \int d\mathbf{l} \times \mathbf{B}$$

균일한 자기장 속의 길이 $L$인 직선 도선에 대해:

$$F = BIL\sin\theta$$

여기서 $\theta$는 도선과 자기장 사이의 각도이다.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Simulate charged particle motion in a magnetic field
# Why 3D: the helical motion requires three spatial dimensions

q = 1.6e-19    # proton charge (C)
m = 1.67e-27   # proton mass (kg)
B = np.array([0, 0, 1e-3])  # magnetic field in z-direction (1 mT)

# Initial conditions: velocity has both perpendicular and parallel components
v0 = np.array([1e5, 0, 3e4])  # m/s — mostly perpendicular, some parallel

# Time parameters
# Why ω_c: we need the cyclotron period to choose appropriate time steps
omega_c = q * np.linalg.norm(B) / m
T_c = 2 * np.pi / omega_c   # cyclotron period
dt = T_c / 200               # 200 steps per period for smooth curves
N_steps = 1000

# Integrate equations of motion: dv/dt = (q/m)(v × B)
# Why Euler-Cromer: it preserves energy better than simple Euler for oscillatory motion
pos = np.zeros((N_steps, 3))
vel = np.zeros((N_steps, 3))
pos[0] = np.array([0, 0, 0])
vel[0] = v0

for i in range(N_steps - 1):
    # Lorentz force (no electric field)
    a = (q / m) * np.cross(vel[i], B)
    vel[i+1] = vel[i] + a * dt
    pos[i+1] = pos[i] + vel[i+1] * dt  # Euler-Cromer: use updated velocity

# Analytical cyclotron radius
r_c = m * np.sqrt(v0[0]**2 + v0[1]**2) / (q * np.linalg.norm(B))
print(f"Cyclotron frequency: ω_c = {omega_c:.4e} rad/s")
print(f"Cyclotron period:    T_c = {T_c:.4e} s")
print(f"Cyclotron radius:    r_c = {r_c:.4f} m")

fig = plt.figure(figsize=(12, 5))

# 3D helical trajectory
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', linewidth=0.8)
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_zlabel('z (m)')
ax1.set_title('Helical Motion in Uniform B')

# xy projection — circular motion
ax2 = fig.add_subplot(122)
ax2.plot(pos[:, 0], pos[:, 1], 'b-', linewidth=0.8)
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_title('xy-Projection (Circular)')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

# Draw expected cyclotron circle for comparison
theta_circ = np.linspace(0, 2*np.pi, 200)
ax2.plot(r_c * np.cos(theta_circ) + r_c, r_c * np.sin(theta_circ),
         'r--', alpha=0.5, label=f'r_c = {r_c:.3f} m')
ax2.legend()

plt.tight_layout()
plt.savefig('cyclotron_motion.png', dpi=150)
plt.show()
```

---

## 비오-사바르 법칙

비오-사바르 법칙(Biot-Savart law)은 정상 전류가 만드는 자기장을 준다:

$$\mathbf{B}(\mathbf{r}) = \frac{\mu_0}{4\pi} \int \frac{I \, d\mathbf{l}' \times \hat{\boldsymbol{\mathscr{r}}}}{{|\boldsymbol{\mathscr{r}}|}^2}$$

여기서:
- $\mu_0 = 4\pi \times 10^{-7}$ T$\cdot$m/A는 자유 공간의 투자율(permeability of free space)이다
- $d\mathbf{l}'$는 도선을 따르는 전류 요소이다
- $\boldsymbol{\mathscr{r}} = \mathbf{r} - \mathbf{r}'$는 소스 점에서 장 점(field point)으로의 벡터이다

부피 전류 밀도 $\mathbf{J}$에 대해:

$$\mathbf{B}(\mathbf{r}) = \frac{\mu_0}{4\pi} \int \frac{\mathbf{J}(\mathbf{r}') \times \hat{\boldsymbol{\mathscr{r}}}}{|\boldsymbol{\mathscr{r}}|^2} \, d\tau'$$

### 예: 원형 고리의 자기장

반지름 $R$인 원형 고리에 전류 $I$가 흐를 때, 중심으로부터 축 위 거리 $z$에서의 자기장:

$$B_z = \frac{\mu_0 I R^2}{2(R^2 + z^2)^{3/2}}$$

고리 중심($z = 0$)에서:

$$B_{\text{중심}} = \frac{\mu_0 I}{2R}$$

### 예: 무한 직선 도선

전류 $I$가 흐르는 무한히 긴 직선 도선. 비오-사바르 법칙(또는 앙페르 법칙을 이용하면 더 쉽게):

$$B = \frac{\mu_0 I}{2\pi s}$$

여기서 $s$는 도선으로부터의 수직 거리이다. 자기장은 도선을 감싸며 돈다 — 방향은 오른손 법칙으로 결정된다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Biot-Savart law: magnetic field of a circular current loop
# Why numerical: the off-axis field has no simple closed form

mu_0 = 4 * np.pi * 1e-7   # permeability of free space (T·m/A)

def biot_savart_loop(R, I, field_points, N_segments=1000):
    """
    Compute B field from a circular loop of radius R carrying current I.

    Parameters:
        R: loop radius (m)
        I: current (A)
        field_points: array of shape (M, 3) — points where B is evaluated
        N_segments: number of segments to discretize the loop

    Returns:
        B: array of shape (M, 3) — magnetic field at each point
    """
    # Discretize the loop into small current elements
    # Why many segments: accuracy improves with finer discretization
    phi = np.linspace(0, 2 * np.pi, N_segments, endpoint=False)
    dphi = 2 * np.pi / N_segments

    # Position of each current element on the loop (in xy-plane)
    loop_x = R * np.cos(phi)
    loop_y = R * np.sin(phi)
    loop_z = np.zeros_like(phi)

    # Current element direction: dl' = R dphi * (-sin φ, cos φ, 0)
    dl_x = -R * np.sin(phi) * dphi
    dl_y = R * np.cos(phi) * dphi
    dl_z = np.zeros_like(phi)

    B = np.zeros_like(field_points)

    for i in range(len(field_points)):
        # Separation vector: r - r'
        rx = field_points[i, 0] - loop_x
        ry = field_points[i, 1] - loop_y
        rz = field_points[i, 2] - loop_z
        r_mag = np.sqrt(rx**2 + ry**2 + rz**2)
        r_mag = np.maximum(r_mag, 1e-10)

        # Cross product dl' × r_hat = dl' × (r/|r|)
        # Why cross product: this is the core of Biot-Savart
        cross_x = dl_y * rz - dl_z * ry
        cross_y = dl_z * rx - dl_x * rz
        cross_z = dl_x * ry - dl_y * rx

        # Sum contributions from all segments: B = (μ₀I/4π) Σ (dl'×r̂)/r²
        B[i, 0] = (mu_0 * I / (4 * np.pi)) * np.sum(cross_x / r_mag**3)
        B[i, 1] = (mu_0 * I / (4 * np.pi)) * np.sum(cross_y / r_mag**3)
        B[i, 2] = (mu_0 * I / (4 * np.pi)) * np.sum(cross_z / r_mag**3)

    return B

# Compute on-axis field and compare with analytic formula
R = 0.1    # 10 cm radius
I = 1.0    # 1 A current

z_vals = np.linspace(-0.5, 0.5, 200)
field_pts = np.column_stack([np.zeros_like(z_vals), np.zeros_like(z_vals), z_vals])
B = biot_savart_loop(R, I, field_pts)

# Analytic on-axis formula
B_analytic = mu_0 * I * R**2 / (2 * (R**2 + z_vals**2)**1.5)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(z_vals * 100, B[:, 2] * 1e6, 'b-', linewidth=2, label='Biot-Savart (numerical)')
axes[0].plot(z_vals * 100, B_analytic * 1e6, 'r--', linewidth=2, label='Analytic')
axes[0].set_xlabel('z (cm)')
axes[0].set_ylabel('$B_z$ ($\\mu$T)')
axes[0].set_title('On-Axis Field of Circular Loop')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Compute field in the xz-plane for a 2D cross-section
x_grid = np.linspace(-0.3, 0.3, 40)
z_grid = np.linspace(-0.3, 0.3, 40)
XG, ZG = np.meshgrid(x_grid, z_grid)

pts = np.column_stack([XG.ravel(), np.zeros(XG.size), ZG.ravel()])
B_grid = biot_savart_loop(R, I, pts, N_segments=500)
Bx = B_grid[:, 0].reshape(XG.shape)
Bz = B_grid[:, 2].reshape(ZG.shape)
B_mag = np.sqrt(Bx**2 + Bz**2)

axes[1].streamplot(XG, ZG, Bx, Bz, color=np.log10(B_mag + 1e-12),
                   cmap='viridis', density=2)
axes[1].plot([-R, R], [0, 0], 'ro', markersize=8)
axes[1].set_xlabel('x (m)')
axes[1].set_ylabel('z (m)')
axes[1].set_title('B Field of Circular Loop (xz-plane)')
axes[1].set_aspect('equal')

plt.suptitle('Biot-Savart Law: Current Loop', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('biot_savart_loop.png', dpi=150)
plt.show()
```

---

## 앙페르 법칙

앙페르 법칙(Ampere's law)은 닫힌 경로를 따른 $\mathbf{B}$의 순환(circulation)을 그 경로를 통과하는 전류와 연결한다:

### 적분 형태

$$\oint \mathbf{B} \cdot d\mathbf{l} = \mu_0 I_{\text{enc}}$$

### 미분 형태

$$\nabla \times \mathbf{B} = \mu_0 \mathbf{J}$$

앙페르 법칙은 정자기학에서 가우스 법칙이 정전기학에서 하는 역할과 같다 — 항상 성립하지만, 대칭성 덕분에 $\mathbf{B}$를 적분 밖으로 꺼낼 수 있을 때 가장 유용하다.

### 적용 예

**긴 직선 도선**: 반지름 $s$인 원형 앙페르 경로:
$$B(2\pi s) = \mu_0 I \implies B = \frac{\mu_0 I}{2\pi s}$$

**무한 솔레노이드** (단위 길이당 $n$번 감김, 전류 $I$): 내부 자기장은 균일하고 외부는 0:
$$B = \mu_0 n I \quad \text{(내부)}, \qquad B = 0 \quad \text{(외부)}$$

**토로이달 솔레노이드** (총 $N$번 감김, 반지름 $R$): 토러스 내부에서 중심으로부터 거리 $s$에서:
$$B = \frac{\mu_0 N I}{2\pi s}$$

> **비유**: 앙페르 법칙은, 닫힌 경로를 따라 걸으면서 $\mathbf{B}$가 얼마나 당신을 "따라오는지"(선적분)를 측정하면, 그 합이 경로를 통과하는 전류에 비례한다고 말한다 — 마치 고리의 원주를 따라 흐름을 측정하여 고리를 통과하는 수도관의 수를 세는 것과 같다.

---

## 자기 단극자는 없다

자기장의 발산은 항상 0이다:

$$\nabla \cdot \mathbf{B} = 0$$

발산 정리를 통한 적분 형태:

$$\oint \mathbf{B} \cdot d\mathbf{a} = 0$$

이것이 의미하는 바:
- **자기력선에는 시작도 끝도 없다** — 항상 닫힌 고리를 형성한다
- **자기 단극자(magnetic monopole)는 없다** (독립된 N극이나 S극은 존재하지 않는다)
- $\mathbf{B}$는 **솔레노이드형(solenoidal)** 장이다

이는 발산이 $\rho/\epsilon_0$인 $\mathbf{E}$와 극명하게 대조된다. 만약 자기 단극자가 존재한다면, 자기 전하 밀도 $\rho_m$에 대해 $\nabla \cdot \mathbf{B} = \mu_0 \rho_m$이 될 것이다. 광범위한 탐색에도 불구하고 아직까지 단극자는 발견된 적이 없다(일부 대통일이론(grand unified theory)에서는 예측하지만).

---

## 자기 쌍극자

### 자기 쌍극자 모멘트

면적 $A$인 작은 전류 고리에 전류 $I$가 흐를 때, 자기 쌍극자 모멘트는:

$$\mathbf{m} = I \mathbf{A} = IA\hat{n}$$

여기서 $\hat{n}$은 고리에 수직인 단위 벡터(전류 방향에 대해 오른손 법칙으로 결정)이다.

### 자기 쌍극자의 장

고리에서 멀리 떨어진 곳($r \gg \sqrt{A}$)에서, 장은 특유의 쌍극자 패턴을 보인다:

$$\mathbf{B}_{\text{쌍극}} = \frac{\mu_0}{4\pi r^3}\left[3(\mathbf{m} \cdot \hat{r})\hat{r} - \mathbf{m}\right] = \frac{\mu_0 m}{4\pi r^3}\left(2\cos\theta\,\hat{r} + \sin\theta\,\hat{\theta}\right)$$

이것은 전기 쌍극자 장과 정확히 같은 각도 구조를 갖는다! 장은 $1/r^3$으로 감소한다.

### 쌍극자에 작용하는 돌림힘과 힘

외부 자기장 $\mathbf{B}$ 속의 자기 쌍극자 $\mathbf{m}$에는:

**돌림힘(torque)**:
$$\boldsymbol{\tau} = \mathbf{m} \times \mathbf{B}$$

이 돌림힘은 $\mathbf{m}$을 $\mathbf{B}$ 방향으로 정렬시키려 한다 — 이것이 나침반 바늘이 작동하는 원리이다.

**퍼텐셜 에너지**:
$$U = -\mathbf{m} \cdot \mathbf{B}$$

$\mathbf{m} \parallel \mathbf{B}$ (같은 방향)일 때 에너지가 최소, 반평행일 때 최대이다.

**힘** (불균일 장에서):
$$\mathbf{F} = \nabla(\mathbf{m} \cdot \mathbf{B})$$

쌍극자는 자기장이 더 강한 영역으로 당겨진다 — 자석이 철 가루를 끌어당기는 이유가 바로 이것이다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Compare electric and magnetic dipole fields — they have the same structure!
# Why compare: seeing the structural identity deepens understanding of both

mu_0 = 4 * np.pi * 1e-7
epsilon_0 = 8.854e-12

# Both dipoles pointing in the z-direction
m = 1.0   # magnetic dipole moment (A·m²)
p = 1.0   # electric dipole moment (C·m) — normalized for visual comparison

# Grid in the xz-plane
x = np.linspace(-2, 2, 40)
z = np.linspace(-2, 2, 40)
X, Z = np.meshgrid(x, z)

r = np.sqrt(X**2 + Z**2)
r = np.maximum(r, 0.3)  # exclude region near origin

# Angles: cos θ = z/r, sin θ = x/r (in xz-plane, θ measured from z-axis)
cos_theta = Z / r
sin_theta = X / r

# Magnetic dipole field: B = (μ₀m/4πr³)(2cosθ r̂ + sinθ θ̂)
# Convert to Cartesian: r̂ = sinθ x̂ + cosθ ẑ, θ̂ = cosθ x̂ - sinθ ẑ
prefactor_B = mu_0 * m / (4 * np.pi * r**3)
Br = prefactor_B * 2 * cos_theta     # radial component
Bt = prefactor_B * sin_theta          # theta component

# To Cartesian
Bx = Br * sin_theta + Bt * cos_theta
Bz = Br * cos_theta - Bt * sin_theta

# Electric dipole field (same structure with different prefactor)
prefactor_E = p / (4 * np.pi * epsilon_0 * r**3)
Er = prefactor_E * 2 * cos_theta
Et = prefactor_E * sin_theta
Ex = Er * sin_theta + Et * cos_theta
Ez = Er * cos_theta - Et * sin_theta

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

B_mag = np.sqrt(Bx**2 + Bz**2)
axes[0].streamplot(X, Z, Bx, Bz, color=np.log10(B_mag), cmap='plasma', density=2)
axes[0].set_xlabel('x')
axes[0].set_ylabel('z')
axes[0].set_title('Magnetic Dipole Field')
axes[0].set_aspect('equal')

E_mag = np.sqrt(Ex**2 + Ez**2)
axes[1].streamplot(X, Z, Ex, Ez, color=np.log10(E_mag), cmap='plasma', density=2)
axes[1].set_xlabel('x')
axes[1].set_ylabel('z')
axes[1].set_title('Electric Dipole Field')
axes[1].set_aspect('equal')

plt.suptitle('Dipole Fields: Magnetic vs Electric (Same Structure!)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('dipole_comparison.png', dpi=150)
plt.show()
```

---

## 자성 재료 (간략 개요)

유전체가 전기장에 분극 $\mathbf{P}$로 반응하는 것처럼, 자성 재료는 자기장에 **자화(magnetization)** $\mathbf{M}$ (단위 부피당 자기 쌍극자 모멘트)로 반응한다:

$$\mathbf{B} = \mu_0(\mathbf{H} + \mathbf{M})$$

여기서 $\mathbf{H}$는 자기장 세기(magnetic field intensity)로, 정전기학의 $\mathbf{D}$에 대응한다.

**선형** 자성 재료에서: $\mathbf{M} = \chi_m \mathbf{H}$이고 $\mathbf{B} = \mu_0(1+\chi_m)\mathbf{H} = \mu\mathbf{H}$.

세 가지 분류:

| 종류 | $\chi_m$ | 예시 | 메커니즘 |
|---|---|---|---|
| **반자성(Diamagnetic)** | $\sim -10^{-5}$ | Cu, Ag, H$_2$O | 궤도 전자 반응 |
| **상자성(Paramagnetic)** | $\sim 10^{-5}$ to $10^{-3}$ | Al, O$_2$, Pt | 영구 모멘트의 정렬 |
| **강자성(Ferromagnetic)** | $\sim 10^2$ to $10^5$ | Fe, Co, Ni | 자구(domain) 정렬 + 교환 상호작용 |

강자성 재료는 **이력 현상(hysteresis)**을 보인다 — 자화 상태는 현재 자기장뿐만 아니라 이력에도 의존한다. 이것이 영구 자석이 존재하고 자기 데이터 저장이 가능한 이유이다.

---

## 평행 도선 사이의 힘

거리 $d$만큼 떨어져 있는 두 평행 도선에 전류 $I_1$과 $I_2$가 흐를 때:

$$\frac{F}{L} = \frac{\mu_0 I_1 I_2}{2\pi d}$$

- **평행 전류는 서로 당긴다** (같은 방향)
- **반평행 전류는 서로 밀어낸다** (반대 방향)

이 힘이 전류의 SI 단위를 정의한다: 암페어는 1 m 간격으로 1 A의 전류가 흐르는 두 도선이 단위 길이당 $2 \times 10^{-7}$ N의 힘을 받도록 정의된다.

---

## 비교: 정전기학 대 정자기학

| 성질 | 정전기학 | 정자기학 |
|---|---|---|
| 소스 | 전하 $\rho$ | 전류 $\mathbf{J}$ |
| 장 | $\mathbf{E}$ | $\mathbf{B}$ |
| 힘 법칙 | $\mathbf{F} = q\mathbf{E}$ | $\mathbf{F} = q\mathbf{v} \times \mathbf{B}$ |
| 발산 | $\nabla \cdot \mathbf{E} = \rho/\epsilon_0$ | $\nabla \cdot \mathbf{B} = 0$ |
| 회전 | $\nabla \times \mathbf{E} = 0$ | $\nabla \times \mathbf{B} = \mu_0\mathbf{J}$ |
| 소스 법칙 | 가우스 법칙 | 앙페르 법칙 |
| 퍼텐셜 | $V$ (스칼라) | $\mathbf{A}$ (벡터) |
| 쌍극자 모멘트 | $\mathbf{p} = q\mathbf{d}$ | $\mathbf{m} = I\mathbf{A}$ |

---

## 요약

| 개념 | 핵심 방정식 |
|---|---|
| 로런츠 힘 | $\mathbf{F} = q(\mathbf{E} + \mathbf{v}\times\mathbf{B})$ |
| 사이클로트론 반지름 | $r = mv/(qB)$ |
| 비오-사바르 | $\mathbf{B} = \frac{\mu_0}{4\pi}\int \frac{I\,d\mathbf{l}'\times\hat{\boldsymbol{\mathscr{r}}}}{|\boldsymbol{\mathscr{r}}|^2}$ |
| 앙페르 법칙 (적분) | $\oint \mathbf{B}\cdot d\mathbf{l} = \mu_0 I_{\text{enc}}$ |
| 앙페르 법칙 (미분) | $\nabla\times\mathbf{B} = \mu_0\mathbf{J}$ |
| 단극자 없음 | $\nabla \cdot \mathbf{B} = 0$ |
| 솔레노이드 | $B = \mu_0 n I$ |
| 쌍극자 모멘트 | $\mathbf{m} = IA\hat{n}$ |
| 쌍극자 돌림힘 | $\boldsymbol{\tau} = \mathbf{m}\times\mathbf{B}$ |
| 쌍극자에 작용하는 힘 | $\mathbf{F} = \nabla(\mathbf{m}\cdot\mathbf{B})$ |

---

## 연습 문제

### 연습 1: 사이클로트론 시뮬레이션
사이클로트론 시뮬레이션을 수정하여 균일한 전기장 $\mathbf{E} = E_0 \hat{x}$를 추가하라. **$\mathbf{E} \times \mathbf{B}$ 드리프트** — 입자가 $\mathbf{E}$와 $\mathbf{B}$ 모두에 수직인 방향으로 $v_d = E/B$의 속도로 이동하는 현상 — 을 관찰하라. 이를 수치적으로 검증하라.

### 연습 2: 헬름홀츠 코일
반지름 $R$인 두 동축 원형 고리가 거리 $R$만큼 떨어져 있고, 각각 같은 방향으로 전류 $I$가 흐른다. 축을 따라 $B_z$를 계산하고 그래프로 나타내라. 중간점 근처에서 자기장이 매우 균일함을 보여라 ($B_z$의 1차 및 2차 도함수가 그곳에서 0이 됨을 확인하라).

### 연습 3: 유한 도선의 비오-사바르
길이 $2L$인 직선 도선에 전류 $I$가 흐른다. 비오-사바르 법칙을 이용하여 도선에서 수직 거리 $s$에서의 자기장을 유도하라. $L \to \infty$ 극한에서 $B = \mu_0 I/(2\pi s)$를 복원함을 보여라.

### 연습 4: 전류 고리 사이의 힘
반지름 $R$인 두 동축 원형 고리에 전류 $I_1$과 $I_2$가 흐른다. 두 고리 사이의 힘을 간격 $d$의 함수로 수치적으로 계산하라. 먼 간격에서 $F = \nabla(\mathbf{m} \cdot \mathbf{B})$를 이용한 결과와 완전한 수치 결과를 비교하라.

### 연습 5: 솔레노이드의 자기장 (수치)
$N$개의 원형 고리를 z축을 따라 균등하게 배치하여 솔레노이드를 모델링하라. 비오-사바르 법칙을 이용하여 전 영역에서 자기장을 계산하라. 축을 따라 $B_z$를 그래프로 나타내고 내부 자기장이 $\mu_0 n I$에 수렴함을 검증하라. 끝 부분 근처의 퍼짐 자기장(fringing field)을 살펴보라.

---

[← 이전: 03. 도체와 유전체](03_Conductors_and_Dielectrics.md) | [다음: 05. 자기 벡터 퍼텐셜 →](05_Magnetic_Vector_Potential.md)
