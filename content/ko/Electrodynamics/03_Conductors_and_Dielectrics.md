# 도체와 유전체

[← 이전: 02. 전기 퍼텐셜과 에너지](02_Electric_Potential_and_Energy.md) | [다음: 04. 정자기학 →](04_Magnetostatics.md)

---

## 학습 목표

1. 도체의 정전기적 성질을 설명하고 도체 표면에서의 경계 조건을 유도한다
2. 접지된 도체가 있는 문제에 영상 전하법(method of image charges)을 적용한다
3. 유전체 분극(dielectric polarization)을 설명하고 자유 전하(free charge)와 속박 전하(bound charge)를 구분한다
4. 변위장(displacement field) $\mathbf{D}$를 정의하고 $\mathbf{E}$, $\mathbf{P}$와의 관계를 서술한다
5. 유전체 경계면에서의 경계 조건을 유도하고 적용한다
6. 표준 기하학적 구조와 유전체 채움에 대해 전기용량을 계산한다
7. Python으로 영상 전하 문제와 전기용량 문제를 수치적으로 풀이한다

---

실제 정전기 문제에서 전하가 빈 공간에 떠 있는 경우는 거의 없다. 전하는 도체(금속, 전선, 전극) 위에 존재하며, 전기장은 유전체(유리, 플라스틱, 생체 조직)를 통과한다. 재료가 전기장에 어떻게 반응하는지를 이해하는 것은 회로 설계에서 세포막이 전압을 유지하는 원리에 이르기까지 모든 것의 기초가 된다. 이 레슨에서는 정전기학에서의 두 가지 대표적인 재료 — 전하를 자유롭게 재분배하는 도체와, 전도는 하지 않지만 분극되는 유전체 — 를 소개하고, 각각을 다루는 수학적 도구를 발전시킨다.

---

## 정전기적 평형 상태의 도체

도체는 자유롭게 움직일 수 있는 전하(일반적으로 전자)를 포함한다. 도체가 정전기적 평형에 도달하면:

1. **도체 내부에서 $\mathbf{E} = 0$.** 전기장이 있다면 자유 전하가 이동하여 전기장을 상쇄한다. 평형은 내부 전기장이 0임을 요구한다.

2. **도체 내부에서 $\rho = 0$.** 가우스 법칙으로부터: $\nabla \cdot \mathbf{E} = \rho/\epsilon_0 = 0$.

3. **모든 여분의 전하는 표면에 분포한다.** 부피 전하가 없으므로, 알짜 전하는 반드시 도체 표면에만 위치한다.

4. **도체는 등전위면이다.** 내부에서 $\mathbf{E} = -\nabla V = 0$이므로, $V$는 전체적으로 일정해야 한다.

5. **표면 바로 바깥의 $\mathbf{E}$는 표면에 수직이다.** 접선 방향 성분이 있다면 표면 전하가 흐르게 된다.

> **비유**: 전기장 속의 도체는 기울어진 탁자 위의 물과 같다. 물(자유 전하)은 표면이 완벽하게 수평이 될 때(등전위)까지 재분배된다. 물이 기울기를 없애려고 흐르듯이, 전하는 접선 방향의 전기장을 없애려고 흐른다.

### 표면 전하와 표면 바로 바깥의 전기장

표면 전하 밀도 $\sigma$를 갖는 도체 표면에서:

$$\mathbf{E}_{\text{표면 바로 바깥}} = \frac{\sigma}{\epsilon_0} \hat{n}$$

여기서 $\hat{n}$은 외부를 향하는 단위 법선 벡터이다. 이는 도체 표면을 둘러싸는 납작한 가우스 면(pillbox)으로부터 유도된다(내부의 전기장은 0이다).

표면 전하에 작용하는 **단위 면적당 힘**은:

$$\mathbf{f} = \frac{\sigma^2}{2\epsilon_0} \hat{n}$$

이것이 **정전기 압력(electrostatic pressure)**이다 — $\sigma$의 부호에 관계없이 항상 바깥 방향으로 작용한다.

---

## 영상 전하법

영상 전하법은 영리한 기법이다: 복잡한 경계값 문제(도체 근처의 전하)를, 동일한 경계 조건을 만족하는 더 단순한 문제(전하와 허구의 "영상" 전하의 쌍)로 대체한다. 유일성 정리에 의해, 이것이 정확한 해가 된다.

### 접지된 평면 위의 점 전하

전하 $+q$가 무한히 넓은 접지된 도체 평면($V = 0$) 위 높이 $d$에 위치한다.

**영상 해**: 도체를 제거하고, 평면 아래 $d$ 거리에(거울 위치) 영상 전하 $-q$를 놓는다. $+q$와 $-q$ 쌍의 퍼텐셜은 자동으로 평면에서 $V = 0$을 만족한다.

$z > 0$인 점 $(x, y, z)$에서의 퍼텐셜:

$$V = \frac{q}{4\pi\epsilon_0}\left[\frac{1}{\sqrt{x^2 + y^2 + (z-d)^2}} - \frac{1}{\sqrt{x^2 + y^2 + (z+d)^2}}\right]$$

평면($z = 0$) 위의 **유도 표면 전하 밀도**:

$$\sigma = -\epsilon_0 \frac{\partial V}{\partial z}\bigg|_{z=0} = \frac{-qd}{2\pi(x^2 + y^2 + d^2)^{3/2}}$$

총 유도 전하는 $Q_{\text{유도}} = \int \sigma \, da = -q$이며, 예상과 일치한다.

평면 쪽을 향한 **전하에 작용하는 힘**:

$$F = \frac{1}{4\pi\epsilon_0}\frac{q^2}{(2d)^2} = \frac{q^2}{16\pi\epsilon_0 d^2}$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Image charge method: point charge above a grounded conducting plane
# Why image charges: they replace complex boundary conditions with simple geometry

epsilon_0 = 8.854e-12
k_e = 1 / (4 * np.pi * epsilon_0)

q = 1e-9    # charge (1 nC)
d = 0.3     # height above plane (30 cm)

# Grid in the xz-plane (y=0)
x = np.linspace(-1, 1, 300)
z = np.linspace(-0.5, 1.0, 300)
X, Z = np.meshgrid(x, z)

# Real charge at (0, 0, d) and image charge at (0, 0, -d)
r_real = np.sqrt(X**2 + (Z - d)**2)
r_image = np.sqrt(X**2 + (Z + d)**2)
r_real = np.maximum(r_real, 1e-4)
r_image = np.maximum(r_image, 1e-4)

# Potential is valid only above the plane (z > 0)
V = k_e * q / r_real - k_e * q / r_image

# Mask the region below the conductor (z < 0) — V=0 there physically
V[Z < 0] = 0

# Why clip: avoid extreme values near the charge
V_clipped = np.clip(V, -200, 200)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Equipotential lines
levels = np.linspace(-150, 150, 31)
cs = axes[0].contour(X, Z, V_clipped, levels=levels, cmap='RdBu_r')
axes[0].axhline(y=0, color='gray', linewidth=3, label='Conductor')
axes[0].plot(0, d, 'ro', markersize=10, label=f'+q at z={d}')
axes[0].plot(0, -d, 'bx', markersize=10, label=f'-q (image)')
axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('z (m)')
axes[0].set_title('Equipotentials')
axes[0].legend()
axes[0].set_aspect('equal')

# Induced surface charge density on the plane
x_surf = np.linspace(-1, 1, 500)
# Why this formula: σ = -qd / (2π(x²+d²)^(3/2)), derived from -ε₀ ∂V/∂z|_{z=0}
sigma = -q * d / (2 * np.pi * (x_surf**2 + d**2)**1.5)

axes[1].plot(x_surf * 100, sigma * 1e9, 'b-', linewidth=2)
axes[1].fill_between(x_surf * 100, sigma * 1e9, alpha=0.3)
axes[1].set_xlabel('x (cm)')
axes[1].set_ylabel(r'$\sigma$ (nC/m$^2$)')
axes[1].set_title('Induced Surface Charge Density')
axes[1].grid(True, alpha=0.3)

# Verify total induced charge equals -q
total_sigma = np.trapz(sigma * 2 * np.pi * np.abs(x_surf), x_surf)
axes[1].text(0.95, 0.95, f'Total induced charge: {total_sigma/q:.3f}q',
             transform=axes[1].transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat'))

plt.suptitle('Method of Image Charges', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('image_charges.png', dpi=150)
plt.show()
```

### 접지된 구에 대한 영상 전하

반지름 $R$인 접지된 구의 중심으로부터 거리 $a$에 전하 $q$가 있다 ($a > R$). 영상 전하는:

$$q' = -\frac{R}{a}q, \quad \text{위치: 중심에서 } b = \frac{R^2}{a}$$

이는 자명하지 않은 결과다 — 영상 전하는 크기가 다르고, 구 내부에 위치한다.

---

## 유전체

### 분극

전기장을 유전체 재료에 가하면 원자 또는 분자의 전하가 약간 이동하여 작은 쌍극자들이 생성된다. 이것이 **분극(polarization)**이다. 분극 벡터 $\mathbf{P}$는 단위 부피당 쌍극자 모멘트이다:

$$\mathbf{P} = \frac{\text{쌍극자 모멘트}}{\text{부피}} = n \langle \mathbf{p} \rangle$$

여기서 $n$은 분자 수 밀도이고, $\langle \mathbf{p} \rangle$는 평균 분자 쌍극자 모멘트이다.

### 속박 전하

분극은 **속박 전하(bound charge)**라고 불리는 유효 전하를 만들어낸다:

$$\rho_b = -\nabla \cdot \mathbf{P} \qquad \text{(부피 속박 전하)}$$
$$\sigma_b = \mathbf{P} \cdot \hat{n} \qquad \text{(표면 속박 전하)}$$

물리적 그림: 쌍극자들이 정렬하면, 내부의 전하들은 쌍으로 상쇄되지만, 표면과 $\mathbf{P}$가 불균일한 곳에서는 상쇄되지 않은 전하가 나타난다.

> **비유**: 손을 잡고 줄을 선 사람들을 상상해보자. 각 사람은 전기적으로 중성이지만, 모두가 오른쪽으로 약간 기울어지면(분극), 맨 오른쪽 사람의 오른손에는 파트너가 없다 — 이것이 상쇄되지 않은 표면 전하처럼 튀어나온다. 줄 내부에서는 모든 왼손이 오른손을 잡고 있으므로 알짜 전하가 없다.

### 변위장 D

총 전하 = 자유 전하 + 속박 전하: $\rho = \rho_f + \rho_b$. 가우스 법칙은 다음이 된다:

$$\nabla \cdot \mathbf{E} = \frac{\rho_f + \rho_b}{\epsilon_0} = \frac{\rho_f - \nabla \cdot \mathbf{P}}{\epsilon_0}$$

**변위장(displacement field)**을 정의한다:

$$\mathbf{D} = \epsilon_0 \mathbf{E} + \mathbf{P}$$

그러면 $\mathbf{D}$에 대한 가우스 법칙은 자유 전하만 포함한다:

$$\nabla \cdot \mathbf{D} = \rho_f$$
$$\oint \mathbf{D} \cdot d\mathbf{a} = Q_{f,\text{enc}}$$

### 선형 유전체

대부분의 재료에서(보통의 전기장 세기에서) $\mathbf{P}$는 $\mathbf{E}$에 비례한다:

$$\mathbf{P} = \epsilon_0 \chi_e \mathbf{E}$$

여기서 $\chi_e$는 **전기 감수율(electric susceptibility)**이다. 그러면:

$$\mathbf{D} = \epsilon_0(1 + \chi_e)\mathbf{E} = \epsilon \mathbf{E}$$

여기서 $\epsilon = \epsilon_0 \epsilon_r$은 **유전율(permittivity)**이고, $\epsilon_r = 1 + \chi_e$는 **비유전율(relative permittivity)** (유전 상수)이다.

| 재료 | $\epsilon_r$ |
|---|---|
| 진공 | 1 |
| 공기 | 1.0006 |
| 종이 | 3.7 |
| 유리 | 4-10 |
| 실리콘 | 11.7 |
| 물 | 80 |
| 티탄산바륨 | ~1000-10000 |

---

## 유전체 경계면에서의 경계 조건

두 유전체($\epsilon_1$과 $\epsilon_2$)의 경계면에서, 자유 표면 전하 $\sigma_f$가 있을 때:

**법선 성분** (납작한 가우스 면으로부터):
$$D_1^{\perp} - D_2^{\perp} = \sigma_f$$
$$\epsilon_1 E_1^{\perp} - \epsilon_2 E_2^{\perp} = \sigma_f$$

자유 표면 전하가 없는 경우($\sigma_f = 0$):
$$\epsilon_1 E_1^{\perp} = \epsilon_2 E_2^{\perp}$$

**접선 성분** (루프로부터, $\nabla \times \mathbf{E} = 0$이므로):
$$E_1^{\parallel} = E_2^{\parallel}$$

이 조건들은 유전체 경계면에서 전기력선이 어떻게 꺾이는지를 결정한다:

$$\frac{\tan\theta_1}{\tan\theta_2} = \frac{\epsilon_1}{\epsilon_2}$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstrate field line bending at a dielectric interface
# Why this visualization: seeing refraction of E-field lines builds intuition

epsilon_r1 = 1.0    # air
epsilon_r2 = 4.0    # glass

# Incident angles (measured from the normal)
theta1 = np.linspace(5, 85, 100)
theta1_rad = np.radians(theta1)

# Snell's-law analogue for E-field: tan(θ₂)/tan(θ₁) = ε₂/ε₁
# Why tangent (not sine): this comes from continuity of E∥ and ε E⊥
tan_theta2 = (epsilon_r2 / epsilon_r1) * np.tan(theta1_rad)
theta2_rad = np.arctan(tan_theta2)
theta2 = np.degrees(theta2_rad)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot refraction angles
axes[0].plot(theta1, theta2, 'b-', linewidth=2)
axes[0].plot(theta1, theta1, 'k--', alpha=0.5, label='No refraction')
axes[0].set_xlabel(r'$\theta_1$ (degrees) — angle in medium 1')
axes[0].set_ylabel(r'$\theta_2$ (degrees) — angle in medium 2')
axes[0].set_title(f'E-field Refraction (εᵣ₁={epsilon_r1}, εᵣ₂={epsilon_r2})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Draw field lines crossing the interface
axes[1].axhline(y=0, color='gray', linewidth=3, label='Interface')
axes[1].fill_between([-2, 2], [-2, -2], [0, 0], alpha=0.1, color='blue',
                      label=f'Medium 2 (εᵣ={epsilon_r2})')
axes[1].fill_between([-2, 2], [0, 0], [2, 2], alpha=0.1, color='yellow',
                      label=f'Medium 1 (εᵣ={epsilon_r1})')

# Draw a few representative field lines
for theta1_deg in [20, 40, 60]:
    t1 = np.radians(theta1_deg)
    t2 = np.arctan((epsilon_r2 / epsilon_r1) * np.tan(t1))

    # Line in medium 1 (above interface)
    L = 1.5
    x_start = -L * np.sin(t1)
    y_start = L * np.cos(t1)
    axes[1].annotate('', xy=(0, 0), xytext=(x_start, y_start),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Line in medium 2 (below interface)
    x_end = L * np.sin(t2)
    y_end = -L * np.cos(t2)
    axes[1].annotate('', xy=(x_end, y_end), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))

    axes[1].text(x_start - 0.1, y_start, f'{theta1_deg}°', fontsize=9)

axes[1].set_xlim(-2, 2)
axes[1].set_ylim(-2, 2)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Field Lines at Dielectric Interface')
axes[1].legend(loc='lower right')
axes[1].set_aspect('equal')

plt.tight_layout()
plt.savefig('dielectric_refraction.png', dpi=150)
plt.show()
```

---

## 전기용량

**축전기(capacitor)**는 도체 사이의 전기장에 전하와 에너지를 저장하는 소자이다. 전하 $+Q$와 $-Q$를 띤 두 도체에 대해:

$$C = \frac{Q}{V} \qquad [F = C/V]$$

여기서 $V = V_+ - V_-$는 전위차이다.

### 표준 기하 구조

**평행판** (면적 $A$, 간격 $d$):
$$C = \frac{\epsilon_0 A}{d}$$

**동축 원통** (내부 반지름 $a$, 외부 반지름 $b$, 길이 $L$):
$$C = \frac{2\pi\epsilon_0 L}{\ln(b/a)}$$

**동심 구** (내부 반지름 $a$, 외부 반지름 $b$):
$$C = 4\pi\epsilon_0 \frac{ab}{b-a}$$

### 유전체의 효과

유전 상수 $\epsilon_r$인 유전체로 축전기를 채우면 전기용량이 $\epsilon_r$배 증가한다:

$$C_{\text{유전체}} = \epsilon_r C_{\text{진공}}$$

이것이 유전체를 사용하는 주된 이유이다 — 절연을 유지하면서 전기용량을 증가시킨다.

### 축전기에 저장된 에너지

$$W = \frac{1}{2}CV^2 = \frac{Q^2}{2C} = \frac{1}{2}QV$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Capacitance calculations for three standard geometries
# Why compare geometries: each has different scaling behavior

epsilon_0 = 8.854e-12

# --- Parallel Plate ---
A = 0.01                  # plate area (100 cm²)
d = np.linspace(0.001, 0.01, 100)  # separation (1 mm to 1 cm)
C_pp = epsilon_0 * A / d

# --- Coaxial Cable ---
a = 0.001                 # inner radius (1 mm)
b = np.linspace(0.002, 0.02, 100)  # outer radius
L = 1.0                   # length (1 m)
C_coax = 2 * np.pi * epsilon_0 * L / np.log(b / a)

# --- Concentric Spheres ---
a_s = 0.05                # inner radius (5 cm)
b_s = np.linspace(0.06, 0.5, 100)  # outer radius
C_sphere = 4 * np.pi * epsilon_0 * a_s * b_s / (b_s - a_s)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(d * 1000, C_pp * 1e12, 'b-', linewidth=2)
axes[0].set_xlabel('Separation d (mm)')
axes[0].set_ylabel('C (pF)')
axes[0].set_title('Parallel Plate')
axes[0].grid(True, alpha=0.3)

axes[1].plot(b * 1000, C_coax * 1e12, 'r-', linewidth=2)
axes[1].set_xlabel('Outer radius b (mm)')
axes[1].set_ylabel('C (pF)')
axes[1].set_title(f'Coaxial (a={a*1000:.0f} mm, L={L} m)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(b_s * 100, C_sphere * 1e12, 'g-', linewidth=2)
axes[2].set_xlabel('Outer radius b (cm)')
axes[2].set_ylabel('C (pF)')
axes[2].set_title(f'Concentric Spheres (a={a_s*100:.0f} cm)')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Capacitance of Standard Geometries', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('capacitance_geometries.png', dpi=150)
plt.show()

# Dielectric effect on parallel plate capacitor
print("Effect of dielectric on parallel plate capacitor:")
print(f"{'Material':<20} {'εᵣ':>6} {'C (pF)':>10}")
print("-" * 38)
d_fixed = 0.001  # 1 mm gap
for name, eps_r in [('Vacuum', 1), ('Paper', 3.7), ('Glass', 5),
                     ('Silicon', 11.7), ('Water', 80)]:
    C = eps_r * epsilon_0 * A / d_fixed
    print(f"{name:<20} {eps_r:>6.1f} {C*1e12:>10.2f}")
```

---

## 축전기의 직렬·병렬 연결

축전기는 직렬과 병렬로 조합할 수 있다:

**병렬** (각각 동일한 전압):
$$C_{\text{합성}} = C_1 + C_2 + C_3 + \cdots$$

**직렬** (각각 동일한 전하):
$$\frac{1}{C_{\text{합성}}} = \frac{1}{C_1} + \frac{1}{C_2} + \frac{1}{C_3} + \cdots$$

패턴을 주목하자: 축전기는 저항과 반대 방식으로 합성된다. 병렬 축전기는 직접 더하고(직렬 저항처럼), 직렬 축전기는 역수로 더한다(병렬 저항처럼).

### 왜 그럴까?

병렬에서는 각 축전기가 동일한 전압원으로부터 독립적으로 전하를 저장하므로, 총 저장 전하(따라서 총 전기용량)는 합이 된다.

직렬에서는 동일한 전하 $Q$가 각 축전기에 나타나지만(사이의 고립된 도체에서의 전하 보존), 전압은 더해진다: $V_{\text{합}} = V_1 + V_2 + \cdots = Q/C_1 + Q/C_2 + \cdots$

---

## 유전체에 작용하는 힘

유전체 슬래브는 가장자리의 퍼짐 전기장(fringing field)에 의해 평행판 축전기 안으로 당겨진다. 폭 $w$, 간격 $d$인 평행판 축전기에, 두께 $d$이고 유전율 $\epsilon_r$인 유전체 슬래브가 거리 $x$만큼 삽입되어 있을 때:

일정 전압 $V$에서:

$$F = \frac{\epsilon_0(\epsilon_r - 1)wV^2}{2d}$$

힘은 $x$에 무관하다 — 슬래브가 미끄러져 들어가는 동안 일정하다. 이는 단위 변위당 에너지 변화가 일정하기 때문이다.

---

## 실용적인 축전기

### 축전기의 에너지 밀도

전기장 $E = V/d$인 평행판 축전기의 판 사이:

$$u = \frac{1}{2}\epsilon_0 E^2 = \frac{1}{2}\epsilon_0\frac{V^2}{d^2}$$

1 mm 간격에서 1000 V일 때: $u \approx 4.4 \times 10^{-3}$ J/m$^3$. 유전체($\epsilon_r = 1000$, 티탄산바륨의 경우)를 사용하면: $u \approx 4.4$ J/m$^3$.

### 절연 파괴

모든 유전체는 절연 파괴(도전 상태로 전환)가 일어나기 전까지 견딜 수 있는 최대 전기장 세기가 있다. 대표적인 값:

| 재료 | 절연 파괴 전기장 (MV/m) |
|---|---|
| 공기 | 3 |
| 종이 | 16 |
| 유리 | 10-40 |
| 테플론 | 60 |
| 운모 | 100-200 |

이것이 축전기의 최대 전압을 제한한다. 1 mm 공기 간격의 경우: $V_{\max} = 3 \times 10^6 \times 10^{-3} = 3000$ V.

최대 에너지 밀도는 유전 상수와 절연 파괴 전기장 모두에 의해 결정된다 — 높은 $\epsilon_r$과 높은 절연 파괴 전기장을 동시에 갖는 재료는 소형 에너지 저장에 매우 가치 있다.

---

## 요약

| 개념 | 핵심 방정식 |
|---|---|
| 도체 내부의 E | $\mathbf{E} = 0$ |
| 표면 전하 전기장 | $E = \sigma/\epsilon_0$ |
| 영상 전하 (평면) | $q' = -q$ (거울 위치) |
| 영상 전하 (구) | $q' = -(R/a)q$, 위치 $b = R^2/a$ |
| 분극 | $\mathbf{P} = \epsilon_0 \chi_e \mathbf{E}$ |
| 속박 전하 | $\rho_b = -\nabla \cdot \mathbf{P}$, $\sigma_b = \mathbf{P}\cdot\hat{n}$ |
| 변위장 | $\mathbf{D} = \epsilon_0\mathbf{E} + \mathbf{P} = \epsilon\mathbf{E}$ |
| D에 대한 가우스 법칙 | $\nabla \cdot \mathbf{D} = \rho_f$ |
| 경계 조건 (법선) | $\epsilon_1 E_1^\perp - \epsilon_2 E_2^\perp = \sigma_f$ |
| 경계 조건 (접선) | $E_1^\parallel = E_2^\parallel$ |
| 평행판 전기용량 | $C = \epsilon_0 A/d$ |
| 축전기 에너지 | $W = \frac{1}{2}CV^2$ |

---

## 연습 문제

### 연습 1: 영상 전하 — 접지된 구
전하 $q = 5$ nC가 반지름 $R = 0.2$ m인 접지된 도체 구의 중심으로부터 $a = 0.5$ m에 위치한다. 영상 전하의 크기와 위치를 구하라. 두 전하와 구의 중심을 포함하는 평면에서의 퍼텐셜을 계산하고 그래프로 나타내라.

### 연습 2: 유전체 구
반지름 $R$이고 유전 상수 $\epsilon_r$인 유전체 구가 균일한 외부 전기장 $\mathbf{E}_0 = E_0 \hat{z}$ 속에 놓여 있다. 구 내부의 전기장이 균일함을 보이고 그 크기를 구하라: $E_{\text{내부}} = \frac{3}{\epsilon_r + 2}E_0$. 내부와 외부의 전기력선을 그려라.

### 연습 3: 다층 축전기
두께 $d_1, d_2, d_3$이고 유전 상수 $\epsilon_1, \epsilon_2, \epsilon_3$인 세 유전체 슬래브가 평행판 축전기의 판 사이에 겹쳐 쌓여 있다. 유효 전기용량을 유도하라. $d_1=d_2=d_3=1$ mm, $\epsilon_1=2, \epsilon_2=5, \epsilon_3=10$으로 수치 검증하라.

### 연습 4: 대전된 도체의 에너지
반지름 $R$인 도체 구가 총 전하 $Q$를 갖는다. 전기장에 저장된 정전 에너지를 계산하라. 동일한 $R$과 $Q$를 갖는 균일하게 대전된 절연체 구의 에너지와 어떻게 다른가?

### 연습 5: 분극과 속박 전하
반지름 $R$, 길이 $L$인 유전체 원통이 분극 $\mathbf{P} = P_0 \hat{z}$ (균일)를 갖는다. 부피 속박 전하와 표면 속박 전하를 계산하라. 총 속박 전하가 0임을 보여라. 속박 전하 분포를 스케치하라.

---

[← 이전: 02. 전기 퍼텐셜과 에너지](02_Electric_Potential_and_Energy.md) | [다음: 04. 정자기학 →](04_Magnetostatics.md)
