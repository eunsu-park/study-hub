# 맥스웰 방정식 — 적분 형태

[← 이전: 07. 맥스웰 방정식 — 미분 형태](07_Maxwells_Equations_Differential.md) | [다음: 09. 진공에서의 전자기파 →](09_EM_Waves_Vacuum.md)

---

## 학습 목표

1. 스토크스 정리(Stokes' theorem)와 발산 정리(divergence theorem)를 이용하여 맥스웰 방정식의 미분 형태와 적분 형태를 상호 변환한다
2. 연속 방정식(continuity equation)을 유도하고 전하 보존을 설명한다
3. 전자기 에너지 보존에 관한 포인팅 정리(Poynting's theorem)를 서술하고 유도한다
4. 포인팅 벡터(Poynting vector)를 계산하고 전자기 시스템에서 에너지 흐름을 해석한다
5. 맥스웰 응력 텐서(Maxwell stress tensor)를 소개하고 전자기 운동량을 설명한다
6. 에너지 및 운동량 보존을 실제 문제에 적용한다
7. 에너지 흐름을 수치적으로 계산하고 포인팅 정리를 검증한다

---

맥스웰 방정식의 미분 형태는 우아하고 간결하지만, 적분 형태는 측정과 직접 연결된다: 전압계는 $\mathbf{E}$의 선적분을, 자속계는 $\mathbf{B}$의 면적분을 측정한다. 더 중요하게는, 적분 형태가 자연스럽게 보존 법칙으로 이어진다 — 전하 보존과 전자기 에너지 보존이다. 이 레슨에서 유도하는 포인팅 정리는 전자기장이 에너지와 운동량을 실어 나르며, 이것이 공간을 통해 실재하는 물리량으로 흐름을 밝혀준다. 포인팅 벡터 $\mathbf{S} = \frac{1}{\mu_0}\mathbf{E} \times \mathbf{B}$는 에너지가 어디로 향하는지 정확히 알려준다 — "배터리에서 전구까지 에너지는 어떻게 전달되는가?"와 같은 질문에 답한다.

---

## 맥스웰 방정식의 적분 형태

미분 형태와 적분 형태는 벡터 해석학의 두 기본 정리로 연결된다:

- **발산 정리(Divergence theorem)**: $\int_V (\nabla \cdot \mathbf{F}) \, d\tau = \oint_S \mathbf{F} \cdot d\mathbf{a}$
- **스토크스 정리(Stokes' theorem)**: $\int_S (\nabla \times \mathbf{F}) \cdot d\mathbf{a} = \oint_C \mathbf{F} \cdot d\mathbf{l}$

이를 맥스웰 방정식에 적용하면:

### (i) 가우스 법칙

$$\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0} \xrightarrow{\text{발산 정리}} \boxed{\oint_S \mathbf{E} \cdot d\mathbf{a} = \frac{Q_{\text{enc}}}{\epsilon_0}}$$

닫힌 면을 통과하는 전기 선속(flux)의 합은 내부 전하를 $\epsilon_0$으로 나눈 값과 같다.

### (ii) 자기 단극자 없음

$$\nabla \cdot \mathbf{B} = 0 \xrightarrow{\text{발산 정리}} \boxed{\oint_S \mathbf{B} \cdot d\mathbf{a} = 0}$$

임의의 닫힌 면을 통과하는 자기 선속의 합은 0이다. 부피로 들어가는 자기력선은 반드시 나와야 한다.

### (iii) 패러데이 법칙

$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t} \xrightarrow{\text{스토크스 정리}} \boxed{\oint_C \mathbf{E} \cdot d\mathbf{l} = -\frac{d}{dt}\int_S \mathbf{B} \cdot d\mathbf{a}}$$

폐경로를 따른 기전력(EMF)은 그 경로를 통과하는 자기 선속의 변화율의 음수와 같다.

### (iv) 앙페르-맥스웰 법칙

$$\nabla \times \mathbf{B} = \mu_0\mathbf{J} + \mu_0\epsilon_0\frac{\partial \mathbf{E}}{\partial t} \xrightarrow{\text{스토크스 정리}} \boxed{\oint_C \mathbf{B} \cdot d\mathbf{l} = \mu_0 I_{\text{enc}} + \mu_0\epsilon_0\frac{d}{dt}\int_S \mathbf{E} \cdot d\mathbf{a}}$$

폐경로를 따른 $\mathbf{B}$의 순환은 $\mu_0$에 경로를 통과하는 전체 전류(실제 전류 + 변위 전류)를 곱한 값과 같다.

> **비유**: 미분 형태는 현미경과 같다 — 한 점에서 일어나는 현상을 서술한다. 적분 형태는 위성에서 바라본 광경과 같다 — 한 영역 전체에 걸친 누적 효과를 서술한다. 발산 정리와 스토크스 정리는 두 관점을 연결하는 "줌" 버튼이다.

---

## 전하 보존: 연속 방정식

앙페르-맥스웰 방정식의 발산을 취하면:

$$\nabla \cdot (\nabla \times \mathbf{B}) = \mu_0\nabla \cdot \mathbf{J} + \mu_0\epsilon_0\frac{\partial}{\partial t}(\nabla \cdot \mathbf{E})$$

좌변은 사라진다(회전의 발산은 0). 우변에 가우스 법칙을 적용하면:

$$0 = \mu_0\nabla \cdot \mathbf{J} + \mu_0\frac{\partial \rho}{\partial t}$$

$$\boxed{\nabla \cdot \mathbf{J} + \frac{\partial \rho}{\partial t} = 0} \qquad \text{(연속 방정식)}$$

발산 정리를 이용한 적분 형태:

$$\oint_S \mathbf{J} \cdot d\mathbf{a} = -\frac{dQ_{\text{enc}}}{dt}$$

**물리적 의미**: 부피에서 흘러나가는 순전류는 내부 전하의 감소율과 같다. 전하는 생성되거나 소멸하지 않는다 — 전하는 **국소적으로 보존**된다. 이것이 전자기학에서 가장 근본적인 보존 법칙이다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstrate charge conservation: expanding charged sphere
# Why this model: it cleanly illustrates the continuity equation

# A uniformly charged sphere expands radially
# ρ(r,t) = ρ₀(R₀/R(t))³ for r < R(t), where R(t) = R₀ + vt
# J(r,t) = ρ(r,t) v r̂

R0 = 1.0         # initial radius
rho_0 = 1.0      # initial charge density
v = 0.5           # expansion velocity

t_values = [0, 0.5, 1.0, 1.5, 2.0]
r = np.linspace(0, 4, 1000)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

total_charge_values = []

for t_val in t_values:
    R_t = R0 + v * t_val
    # Charge density: total charge Q = ρ₀(4π/3)R₀³ is conserved
    # So ρ(t) = ρ₀(R₀/R(t))³
    rho_t = rho_0 * (R0 / R_t)**3

    rho_profile = np.where(r < R_t, rho_t, 0)
    axes[0].plot(r, rho_profile, linewidth=2, label=f't = {t_val:.1f}')

    # Current density J = ρv at r < R(t)
    J_profile = np.where(r < R_t, rho_t * v, 0)
    axes[1].plot(r, J_profile, linewidth=2, label=f't = {t_val:.1f}')

    # Verify total charge is conserved
    # Q = ∫ρ 4πr² dr
    Q = np.trapz(rho_profile * 4 * np.pi * r**2, r)
    total_charge_values.append(Q)

Q_exact = rho_0 * (4/3) * np.pi * R0**3
axes[0].set_xlabel('r')
axes[0].set_ylabel(r'$\rho(r)$')
axes[0].set_title('Charge Density (expanding sphere)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('r')
axes[1].set_ylabel(r'$J_r(r)$')
axes[1].set_title('Current Density')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Continuity Equation: Charge Conservation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('continuity_equation.png', dpi=150)
plt.show()

# Verify charge conservation
print("Charge conservation verification:")
print(f"Exact total charge: Q = {Q_exact:.4f}")
for t_val, Q_num in zip(t_values, total_charge_values):
    print(f"  t = {t_val:.1f}: Q = {Q_num:.4f} (error: {abs(Q_num-Q_exact)/Q_exact:.2e})")
```

---

## 포인팅 정리: 에너지 보존

전자기장에 저장된 에너지는 얼마이며, 어떻게 흐르는가? 포인팅 정리가 이 두 질문 모두에 답한다.

### 유도

전자기장이 전하에 하는 일률부터 시작한다. 단위 부피당 받는 힘은 $\mathbf{f} = \rho\mathbf{E} + \mathbf{J}\times\mathbf{B}$이다. 단위 부피당 전달되는 일률은:

$$\frac{dW_{\text{mech}}}{dt \, d\tau} = \mathbf{f} \cdot \mathbf{v} = \rho\mathbf{v}\cdot\mathbf{E} + (\mathbf{J}\times\mathbf{B})\cdot\mathbf{v}$$

$\mathbf{J} = \rho\mathbf{v}$이고 $(\mathbf{J}\times\mathbf{B})\cdot\mathbf{v} = (\mathbf{v}\times\mathbf{B})\cdot\mathbf{J} = 0$(자기력은 일을 하지 않는다)이므로:

$$\frac{dW_{\text{mech}}}{dt \, d\tau} = \mathbf{J} \cdot \mathbf{E}$$

이제 앙페르-맥스웰 법칙을 이용하여 $\mathbf{J}$를 소거하고, 패러데이 법칙과 벡터 항등식 $\nabla\cdot(\mathbf{E}\times\mathbf{B}) = \mathbf{B}\cdot(\nabla\times\mathbf{E}) - \mathbf{E}\cdot(\nabla\times\mathbf{B})$를 이용하여 정리하면:

$$\boxed{-\frac{\partial u}{\partial t} = \nabla \cdot \mathbf{S} + \mathbf{J} \cdot \mathbf{E}}$$

여기서:

$$u = \frac{1}{2}\left(\epsilon_0 E^2 + \frac{B^2}{\mu_0}\right) \qquad \text{(에너지 밀도)}$$

$$\mathbf{S} = \frac{1}{\mu_0}\mathbf{E} \times \mathbf{B} \qquad \text{(포인팅 벡터)}$$

### 적분 형태

$$\boxed{-\frac{dU_{\text{em}}}{dt} = \oint_S \mathbf{S} \cdot d\mathbf{a} + \int_V \mathbf{J}\cdot\mathbf{E} \, d\tau}$$

**물리적 의미**: 부피 내 전자기 에너지의 감소율은 면을 통해 복사되어 나가는 에너지(포인팅 선속)와 내부 전하에 전달되는 에너지(줄 가열)의 합과 같다.

> **비유**: 포인팅 정리는 전자기 에너지에 대한 은행 명세서와 같다. 장에 저장된 에너지는 통장 잔액($U_{\text{em}}$)이다. 포인팅 벡터는 은행 벽을 통해 드나드는 돈($\mathbf{S}$)이다. $\mathbf{J}\cdot\mathbf{E}$ 항은 전하들이 인출하는 돈(역학적 일이나 열로 소비된다)이다. 정리는 이렇게 말한다: 잔액 감소 = 유출 + 인출.

---

## 포인팅 벡터

포인팅 벡터 $\mathbf{S} = \frac{1}{\mu_0}\mathbf{E}\times\mathbf{B}$의 단위는 W/m$^2$이며, **에너지 선속(energy flux)** — 단위 면적당 에너지 흐름률 — 을 나타낸다.

### 에너지 흐름 방향

$\mathbf{S}$는 에너지가 전파하는 방향을 가리킨다. $\hat{z}$ 방향으로 전파하는 평면 전자기파에서 $\mathbf{E} = E_0\hat{x}$, $\mathbf{B} = B_0\hat{y}$이면:

$$\mathbf{S} = \frac{E_0 B_0}{\mu_0}\hat{z} = \frac{E_0^2}{\mu_0 c}\hat{z}$$

에너지는 파동의 진행 방향으로 흐른다 — 예상과 일치한다.

### 시간 평균 포인팅 벡터

진동하는 장에서는 시간 평균이 더 유용하다:

$$\langle\mathbf{S}\rangle = \frac{1}{2\mu_0}\text{Re}(\tilde{\mathbf{E}} \times \tilde{\mathbf{B}}^*)$$

여기서 $\tilde{\mathbf{E}}$와 $\tilde{\mathbf{B}}$는 복소 진폭이다.

### 세기(Intensity)

전자기파의 **세기(intensity)**는 시간 평균 포인팅 벡터의 크기이다:

$$I = |\langle\mathbf{S}\rangle| = \frac{1}{2}\epsilon_0 c E_0^2 = \frac{E_0^2}{2\mu_0 c}$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Poynting vector of a plane electromagnetic wave
# Why visualize: seeing E, B, and S together clarifies energy flow

c = 3e8               # speed of light (m/s)
mu_0 = 4 * np.pi * 1e-7
epsilon_0 = 8.854e-12
f = 1e9               # frequency (1 GHz)
omega = 2 * np.pi * f
k = omega / c
E0 = 100              # electric field amplitude (V/m)
B0 = E0 / c           # magnetic field amplitude (T)

# Spatial profile at t = 0
z = np.linspace(0, 3 * c / f, 1000)  # 3 wavelengths

E_x = E0 * np.sin(k * z)           # E polarized in x
B_y = B0 * np.sin(k * z)           # B polarized in y
S_z = (1 / mu_0) * E_x * B_y       # Poynting vector in z

# Time-averaged intensity
I_avg = E0**2 / (2 * mu_0 * c)

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axes[0].plot(z * 1e2, E_x, 'b-', linewidth=2)
axes[0].set_ylabel('$E_x$ (V/m)')
axes[0].set_title('Electric Field')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='gray', alpha=0.3)

axes[1].plot(z * 1e2, B_y * 1e6, 'r-', linewidth=2)
axes[1].set_ylabel('$B_y$ ($\\mu$T)')
axes[1].set_title('Magnetic Field')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='gray', alpha=0.3)

axes[2].plot(z * 1e2, S_z, 'g-', linewidth=2, label='$S_z$ (instantaneous)')
axes[2].axhline(y=I_avg, color='orange', linestyle='--', linewidth=2,
                label=f'$\\langle S \\rangle$ = {I_avg:.2f} W/m²')
axes[2].set_ylabel('$S_z$ (W/m$^2$)')
axes[2].set_xlabel('z (cm)')
axes[2].set_title('Poynting Vector (Energy Flux)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('Plane EM Wave: E, B, and Poynting Vector',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('poynting_vector.png', dpi=150)
plt.show()

print(f"Wavelength:          λ = {c/f*100:.2f} cm")
print(f"E₀ = {E0} V/m")
print(f"B₀ = {B0*1e6:.4f} μT")
print(f"Time-averaged intensity: ⟨S⟩ = {I_avg:.4f} W/m²")
```

---

## 저항 도선에서의 에너지 흐름

포인팅 벡터의 놀라운 응용: 배터리에서 전구까지 에너지는 어떻게 전달되는가?

단위 길이당 저항 $R/L$을 가지며 전류 $I$가 흐르는 긴 직선 도선에서:

- **도선 내부**: $\mathbf{E}$는 도선을 따라 향하고(전류를 구동), $\mathbf{B}$는 도선을 감아 돈다
- **포인팅 벡터** $\mathbf{S} = \frac{1}{\mu_0}\mathbf{E}\times\mathbf{B}$는 **반지름 방향 안쪽**을 향한다!

에너지는 도선을 통해 흐르는 것이 아니라 — 도선 주위의 전자기장을 통해 흘러 도선 속으로 반지름 방향으로 들어간다. 도선은 장으로부터 에너지를 흡수하여 열로 변환하는 "싱크(sink)" 역할을 한다.

반지름 $a$(도선 표면), 길이 $l$인 원통형 면으로 들어오는 전체 일률:

$$P = \oint \mathbf{S}\cdot d\mathbf{a} = \frac{E \cdot B}{2\mu_0}(2\pi a l) = \frac{(V/l)(\mu_0 I/2\pi a)}{\mu_0}(2\pi a l) = VI = I^2R$$

이것이 줄 법칙(Joule's law)이다 — 포인팅 정리로 확인된다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Poynting vector around a current-carrying resistive wire
# Why this example: it reveals that energy flows through the field, not the wire

mu_0 = 4 * np.pi * 1e-7

I = 1.0          # current (A)
V_per_m = 0.1    # voltage drop per meter (V/m) — determines E inside
a_wire = 0.001   # wire radius (1 mm)

# Grid around the wire (cross-section in xy-plane, wire along z)
x = np.linspace(-0.02, 0.02, 40)
y = np.linspace(-0.02, 0.02, 40)
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)
r = np.maximum(r, a_wire)   # clip to wire surface

# E field: along z everywhere (uniform in this simple model)
# B field: circles the wire, B = μ₀I/(2πr)
# Why E along z: driven by the battery/voltage source
E_z = V_per_m   # uniform E field along wire

B_phi = mu_0 * I / (2 * np.pi * r)  # magnitude
# Convert B_φ to Cartesian: B_x = -B_φ sin(φ), B_y = B_φ cos(φ)
phi = np.arctan2(Y, X)
B_x = -B_phi * np.sin(phi)
B_y = B_phi * np.cos(phi)

# Poynting vector: S = (1/μ₀)(E × B)
# E = E_z ẑ, B = B_x x̂ + B_y ŷ
# Why cross product: S = (E_z/μ₀)(ẑ × (B_x x̂ + B_y ŷ)) = (E_z/μ₀)(B_x ŷ - B_y x̂)
# Wait — let's be careful: ẑ × x̂ = ŷ, ẑ × ŷ = -x̂
S_x = -(E_z / mu_0) * B_y   # radially inward component (x)
S_y = (E_z / mu_0) * B_x    # radially inward component (y)

# Verify: S should point radially INWARD (toward the wire)
S_r = (S_x * X + S_y * Y) / r   # radial component

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Poynting vector field
S_mag = np.sqrt(S_x**2 + S_y**2)
axes[0].streamplot(X, Y, S_x, S_y, color=np.log10(S_mag + 1e-10),
                   cmap='viridis', density=2)
circle = plt.Circle((0, 0), a_wire, color='gray', fill=True, alpha=0.5)
axes[0].add_patch(circle)
axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('y (m)')
axes[0].set_title('Poynting Vector S (Energy flows INTO the wire)')
axes[0].set_aspect('equal')

# Radial component of S
r_line = np.linspace(a_wire, 0.02, 100)
S_r_line = -E_z * mu_0 * I / (2 * np.pi * r_line * mu_0)
# Simplified: S_r = -E_z * I / (2π r) — negative means inward
axes[1].plot(r_line * 1000, -S_r_line, 'r-', linewidth=2)
axes[1].set_xlabel('r (mm)')
axes[1].set_ylabel('|S_r| (W/m²)')
axes[1].set_title('Radial Energy Flux (inward)')
axes[1].grid(True, alpha=0.3)

# Total power entering the wire (per meter length)
P_per_m = I * V_per_m  # = I²R per meter
axes[1].text(0.95, 0.95, f'Power/m = IV = {P_per_m:.3f} W/m\n= I²R (Joule heating)',
             transform=axes[1].transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat'))

plt.suptitle('Energy Flow Around a Resistive Wire', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('poynting_wire.png', dpi=150)
plt.show()
```

---

## 맥스웰 응력 텐서(Maxwell Stress Tensor)

전자기장은 에너지뿐 아니라 **운동량**도 운반한다. 운동량 밀도는:

$$\mathbf{g} = \mu_0\epsilon_0 \mathbf{S} = \epsilon_0(\mathbf{E}\times\mathbf{B})$$

**맥스웰 응력 텐서** $\overleftrightarrow{T}$는 면에 작용하는 단위 면적당 전자기력을 서술한다:

$$T_{ij} = \epsilon_0\left(E_i E_j - \frac{1}{2}\delta_{ij}E^2\right) + \frac{1}{\mu_0}\left(B_i B_j - \frac{1}{2}\delta_{ij}B^2\right)$$

면 $S$로 둘러싸인 부피 $V$ 내 전하에 작용하는 힘은:

$$\mathbf{F} = \oint_S \overleftrightarrow{T} \cdot d\mathbf{a} - \mu_0\epsilon_0\frac{d}{dt}\int_V \mathbf{S} \, d\tau$$

정적 장에서는 두 번째 항이 사라지고:

$$F_i = \oint_S \sum_j T_{ij} \, da_j$$

### 물리적 해석

응력 텐서는 다음을 알려준다:
- **대각 성분** ($T_{xx}, T_{yy}, T_{zz}$): 압력 (장선을 따라 장력, 장선에 수직으로 압축)
- **비대각 성분**: 전단 응력(shear stress)

전기 및 자기 장선은 **장력을 받은 고무줄**처럼 행동한다 — 길이 방향으로는 당기고 옆으로는 민다. 이것이 전자기력의 역학적 기원이다.

### 복사 압력(Radiation Pressure)

전자기파가 면에 부딪히면 운동량을 전달하고 압력을 가한다:

**완전 흡수**:
$$P_{\text{rad}} = \frac{I}{c} = \frac{\langle S \rangle}{c}$$

**완전 반사**:
$$P_{\text{rad}} = \frac{2I}{c}$$

지구에서 태양광($I = 1361$ W/m$^2$)의 경우: $P_{\text{rad}} \approx 4.5\,\mu$Pa(흡수), $9\,\mu$Pa(반사). 매우 작지만 — 소행성의 궤도에 측정 가능한 영향(야르콥스키 효과, Yarkovsky effect)을 미치고 태양 돛(solar sail)을 추진할 수 있을 만큼은 크다.

### 각운동량(Angular Momentum)

원편광(circularly polarized) 빛도 각운동량을 운반한다. 각운동량 밀도는:

$$\ell_z = \pm \frac{u}{\omega}$$

여기서 $\pm$는 우원편광/좌원편광에 대응한다. 각 광자는 $\pm\hbar$의 각운동량을 지닌다 — 이것이 광자의 스핀-1 성질이다.

```python
import numpy as np

# Maxwell stress tensor: compute force between two parallel charged plates
# Why stress tensor: it gives the force without knowing the charge distribution in detail

epsilon_0 = 8.854e-12
mu_0 = 4 * np.pi * 1e-7

# Parallel plate capacitor
sigma = 1e-6      # surface charge density (μC/m²)
A = 0.01          # plate area (100 cm²)

# Electric field between plates
E = sigma / epsilon_0   # V/m (field in gap, no B field)

# Maxwell stress tensor between the plates
# With E = E_z ẑ only:
# T_zz = ε₀(E_z² - E²/2) = ε₀ E²/2
# This is the pressure (force per unit area) on the plate
T_zz = epsilon_0 * E**2 / 2

# Force on one plate
F = T_zz * A

# Compare with direct Coulomb force: F = σ²A/(2ε₀)
F_coulomb = sigma**2 * A / (2 * epsilon_0)

print("Maxwell Stress Tensor: Force Between Capacitor Plates")
print("=" * 55)
print(f"Surface charge density: σ = {sigma*1e6:.1f} μC/m²")
print(f"Plate area:             A = {A*1e4:.0f} cm²")
print(f"Electric field:         E = {E:.2f} V/m")
print(f"")
print(f"Stress T_zz = ε₀E²/2 = {T_zz:.4f} Pa")
print(f"Force (stress tensor): F = {F:.6f} N")
print(f"Force (Coulomb):       F = {F_coulomb:.6f} N")
print(f"Agreement:             {abs(F - F_coulomb)/F_coulomb:.2e} relative error")
print(f"")
print(f"Electromagnetic momentum density:")
print(f"  g = ε₀(E × B) = 0 (no B field in this static case)")
```

---

## 보존 법칙 요약

| 보존 법칙 | 미분 형태 | 적분 형태 |
|---|---|---|
| **전하(Charge)** | $\nabla\cdot\mathbf{J} + \partial\rho/\partial t = 0$ | $\oint \mathbf{J}\cdot d\mathbf{a} = -dQ_{\text{enc}}/dt$ |
| **에너지(Energy)** | $\nabla\cdot\mathbf{S} + \partial u/\partial t = -\mathbf{J}\cdot\mathbf{E}$ | $\oint \mathbf{S}\cdot d\mathbf{a} = -dU_{\text{em}}/dt - \int\mathbf{J}\cdot\mathbf{E}\,d\tau$ |
| **운동량(Momentum)** | $\nabla\cdot\overleftrightarrow{T} - \mu_0\epsilon_0\partial\mathbf{S}/\partial t = \mathbf{f}$ | $\oint \overleftrightarrow{T}\cdot d\mathbf{a} = \mathbf{F}_{\text{mech}} + d\mathbf{p}_{\text{em}}/dt$ |

각 보존 법칙은 동일한 구조를 갖는다: 면을 통과하는 선속이 부피 내부의 변화율과 균형을 이룬다.

---

## 맥스웰 방정식: 완전한 요약

| # | 미분 형태 | 적분 형태 | 이름 |
|---|---|---|---|
| I | $\nabla\cdot\mathbf{E} = \rho/\epsilon_0$ | $\oint\mathbf{E}\cdot d\mathbf{a} = Q_{\text{enc}}/\epsilon_0$ | 가우스 법칙(Gauss) |
| II | $\nabla\cdot\mathbf{B} = 0$ | $\oint\mathbf{B}\cdot d\mathbf{a} = 0$ | 자기 단극자 없음 |
| III | $\nabla\times\mathbf{E} = -\partial\mathbf{B}/\partial t$ | $\oint\mathbf{E}\cdot d\mathbf{l} = -d\Phi_B/dt$ | 패러데이 법칙(Faraday) |
| IV | $\nabla\times\mathbf{B} = \mu_0\mathbf{J}+\mu_0\epsilon_0\partial\mathbf{E}/\partial t$ | $\oint\mathbf{B}\cdot d\mathbf{l} = \mu_0(I_{\text{enc}}+\epsilon_0 d\Phi_E/dt)$ | 앙페르-맥스웰(Ampere-Maxwell) |

**보조**: $\mathbf{F} = q(\mathbf{E}+\mathbf{v}\times\mathbf{B})$ (로렌츠 힘, Lorentz force), $\nabla\cdot\mathbf{J}+\partial\rho/\partial t = 0$ (연속 방정식)

---

## 요약

| 개념 | 핵심 방정식 |
|---|---|
| 연속 방정식(Continuity equation) | $\nabla\cdot\mathbf{J} + \partial\rho/\partial t = 0$ |
| 에너지 밀도(Energy density) | $u = \frac{1}{2}(\epsilon_0 E^2 + B^2/\mu_0)$ |
| 포인팅 벡터(Poynting vector) | $\mathbf{S} = \frac{1}{\mu_0}\mathbf{E}\times\mathbf{B}$ |
| 포인팅 정리(Poynting's theorem) | $-\partial u/\partial t = \nabla\cdot\mathbf{S} + \mathbf{J}\cdot\mathbf{E}$ |
| 파동 세기(Wave intensity) | $I = \frac{1}{2}\epsilon_0 c E_0^2$ |
| 전자기 운동량 밀도(EM momentum density) | $\mathbf{g} = \epsilon_0(\mathbf{E}\times\mathbf{B}) = \mathbf{S}/c^2$ |
| 응력 텐서(Stress tensor) | $T_{ij} = \epsilon_0(E_iE_j - \frac{1}{2}\delta_{ij}E^2) + \frac{1}{\mu_0}(B_iB_j - \frac{1}{2}\delta_{ij}B^2)$ |
| 복사 압력(Radiation pressure) | $P = I/c$ (흡수), $P = 2I/c$ (반사) |

---

## 연습 문제

### 연습 1: 축전기에 저장된 에너지
평행판 축전기($A = 100$ cm$^2$, $d = 1$ mm)를 $V = 1000$ V로 충전한다. (a) $W = \frac{1}{2}CV^2$와 (b) $W = \frac{\epsilon_0}{2}\int E^2 \, d\tau$를 이용하여 저장된 전체 에너지를 각각 구하고, 두 결과가 일치함을 보여라.

### 연습 2: 충전 중인 축전기의 포인팅 벡터
충전 과정에서 원형 축전기의 두 판 사이 포인팅 벡터는 반지름 방향 안쪽을 향한다. 판의 가장자리에서 $\mathbf{S}$를 계산하고, 안쪽으로의 전체 선속이 두 판 사이 장에너지의 증가율과 같음을 확인하라.

### 연습 3: 태양 복사 압력
태양은 지구 거리에서 약 $I = 1361$ W/m$^2$를 전달한다. (a) 완전 흡수면과 완전 반사면에 작용하는 복사 압력을 각각 계산하라. (b) $10 \times 10$ m 태양 돛에 작용하는 전체 힘을 구하라. (c) 전기장 및 자기장 진폭을 구하라.

### 연습 4: 응력 텐서를 이용한 힘 계산
맥스웰 응력 텐서를 이용하여 간격 $d$만큼 떨어진, 전류 $I_1$과 $I_2$가 흐르는 두 평행 무한 도선 사이의 단위 길이당 힘을 계산하라. $F/L = \mu_0 I_1 I_2/(2\pi d)$와 일치함을 확인하라.

### 연습 5: 수치적 포인팅 정리 검증
레슨 7의 1차원 FDTD 시뮬레이션에서 각 시간 단계마다 에너지 밀도 $u(x,t)$와 포인팅 벡터 $S(x,t)$를 계산하라. 포인팅 정리를 검증하라: 영역 내 전체 에너지와 경계를 통해 빠져나간 에너지의 합이 초기 에너지와 같아야 한다.

---

[← 이전: 07. 맥스웰 방정식 — 미분 형태](07_Maxwells_Equations_Differential.md) | [다음: 09. 진공에서의 전자기파 →](09_EM_Waves_Vacuum.md)
