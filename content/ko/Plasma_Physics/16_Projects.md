# 16. 프로젝트

## 학습 목표

- 다양한 장 기하학을 가진 보리스 알고리즘을 사용하여 완전한 3D 입자 궤도 적분기 구현
- 시각화 도구를 갖춘 냉각 및 온난 플라즈마의 일반 분산 관계 해결기 개발
- 반라그랑주 방법을 사용하여 운동 플라즈마 현상을 연구하는 1D 블라소프-푸아송 해결기 생성
- 단일 입자 궤도, 파동 이론, 운동 이론의 지식 종합
- 수치 결과를 해석적 예측과 대조 검증하고 선형 이론을 넘어선 비선형 물리 탐구
- 연구에 사용되는 계산 플라즈마 물리 방법에 대한 실무 경험 획득

## 프로젝트 1: 입자 궤도 시뮬레이터

### 1.1 개요

**목표**: 다양한 전자기장 구성을 처리하고 입자 궤적, 표류, 불변량을 시각화할 수 있는 다목적 3D 입자 궤도 적분기를 구축합니다.

**난이도**: ⭐⭐⭐

**예상 시간**: 10–15시간

**개발 기술**:
- 운동 방정식의 수치 적분
- 보리스 알고리즘(E&M 장에서 입자를 위한 도약 방법)
- 3D 시각화
- 해석적 표류 속도와 대조 검증
- 입자 구속 및 손실 메커니즘 이해

### 1.2 물리 배경

전하를 띤 입자의 상대론적 운동 방정식은:

$$\frac{d\mathbf{p}}{dt} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$$

여기서 $\mathbf{p} = \gamma m \mathbf{v}$는 운동량이고 $\gamma = 1/\sqrt{1 - v^2/c^2}$는 로렌츠 인자입니다.

비상대론적 입자($v \ll c$)의 경우:

$$m \frac{d\mathbf{v}}{dt} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$$

**보리스 알고리즘**은 정적 장에서 에너지를 보존하는 2차 정확한 시간 가역적 스킴입니다. 이것은 입자-셀(PIC) 코드의 주력 방법입니다.

**보리스 알고리즘** (한 시간 단계 $\Delta t$):

1. 전기장에 의한 절반 가속:
   $$\mathbf{v}^{-} = \mathbf{v}^n + \frac{q \Delta t}{2m} \mathbf{E}$$

2. 자기장에 의한 회전:
   $$\mathbf{v}^{+} = \mathbf{v}^{-} + \mathbf{v}^{-} \times \mathbf{t} + (\mathbf{v}^{-} \times \mathbf{t}) \times \mathbf{s}$$
   여기서:
   $$\mathbf{t} = \frac{q \Delta t}{2m} \mathbf{B}, \quad \mathbf{s} = \frac{2\mathbf{t}}{1 + t^2}$$

3. 전기장에 의한 절반 가속:
   $$\mathbf{v}^{n+1} = \mathbf{v}^{+} + \frac{q \Delta t}{2m} \mathbf{E}$$

4. 위치 업데이트:
   $$\mathbf{x}^{n+1} = \mathbf{x}^n + \mathbf{v}^{n+1} \Delta t$$

### 1.3 구현 가이드

**단계 1: 기본 인프라**

속성을 가진 `Particle` 클래스 생성:
- `q`, `m`: 전하와 질량
- `r`: 위치 벡터 [x, y, z]
- `v`: 속도 벡터 [vx, vy, vz]
- `history`: 궤적을 저장하는 리스트

**단계 2: 장 구성**

다양한 기하학에 대해 $\mathbf{E}(\mathbf{r}, t)$와 $\mathbf{B}(\mathbf{r}, t)$를 반환하는 함수 구현:

1. **균일한 B**: $\mathbf{B} = B_0 \hat{\mathbf{z}}$ (선회)
2. **균일한 E + B**: $\mathbf{E} = E_0 \hat{\mathbf{x}}$, $\mathbf{B} = B_0 \hat{\mathbf{z}}$ (E×B 표류)
3. **경사 B**: $\mathbf{B} = B_0 (1 + \alpha x) \hat{\mathbf{z}}$ (grad-B 표류)
4. **곡선 B**: $\mathbf{B} = \frac{B_0 R_0}{R_0 + x} \hat{\mathbf{z}}$ (곡률 표류, 단순화)
5. **자기 거울**: $\mathbf{B} = B_0 (1 + (z/L)^2) \hat{\mathbf{z}}$ (거울 힘)
6. **토카막(단순화)**: 토로이달 + 폴로이달 장

**단계 3: 보리스 적분기**

보리스 알고리즘을 함수로 구현:

```python
def boris_step(r, v, E, B, q, m, dt):
    """
    One step of Boris algorithm.

    Parameters:
    r, v: position and velocity (3D arrays)
    E, B: electric and magnetic fields at position r (3D arrays)
    q, m: charge and mass
    dt: time step

    Returns:
    r_new, v_new
    """
    # Half electric acceleration
    v_minus = v + (q * dt / (2 * m)) * E

    # Magnetic rotation
    t = (q * dt / (2 * m)) * B
    s = 2 * t / (1 + np.dot(t, t))
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)

    # Half electric acceleration
    v_new = v_plus + (q * dt / (2 * m)) * E

    # Position update
    r_new = r + v_new * dt

    return r_new, v_new
```

**단계 4: 적분 루프**

```python
def integrate_orbit(particle, E_func, B_func, t_final, dt):
    """
    Integrate particle orbit from t=0 to t=t_final.

    Parameters:
    particle: Particle object
    E_func, B_func: functions returning E(r, t) and B(r, t)
    t_final: final time
    dt: time step
    """
    t = 0
    while t < t_final:
        E = E_func(particle.r, t)
        B = B_func(particle.r, t)

        particle.r, particle.v = boris_step(particle.r, particle.v, E, B,
                                             particle.q, particle.m, dt)

        particle.history['t'].append(t)
        particle.history['r'].append(particle.r.copy())
        particle.history['v'].append(particle.v.copy())

        t += dt
```

**단계 5: 시각화**

`matplotlib`를 사용하여 3D 궤적 플롯:

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

r_history = np.array(particle.history['r'])
ax.plot(r_history[:, 0], r_history[:, 1], r_history[:, 2], linewidth=1)

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title('Particle Trajectory')
plt.show()
```

**단계 6: 분석 및 검증**

각 장 구성에 대해 계산:

1. **선회 반지름**: $\rho = m v_\perp / (q B)$
2. **표류 속도**: 수치와 해석 비교:
   - E×B 표류: $\mathbf{v}_E = \mathbf{E} \times \mathbf{B} / B^2$
   - Grad-B 표류: $\mathbf{v}_{\nabla B} = \pm \frac{m v_\perp^2}{2 q B^3} \mathbf{B} \times \nabla B$
   - 곡률 표류: $\mathbf{v}_\kappa = \frac{m v_\parallel^2}{q B^3} \mathbf{B} \times (\mathbf{B} \cdot \nabla)\mathbf{B}$
3. **단열 불변량**:
   - 자기 모멘트: $\mu = m v_\perp^2 / (2B)$
   - 종방향 작용: $J = \oint v_\parallel ds$ (바운스 운동의 경우)
4. **에너지 보존**: 정적 장에서 $E_{kin} = \frac{1}{2} m v^2$가 보존되는지 확인

### 1.4 완전한 코드

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Physical constants
e = 1.6e-19
m_e = 9.11e-31
m_p = 1.67e-27

class Particle:
    def __init__(self, q, m, r0, v0):
        self.q = q
        self.m = m
        self.r = np.array(r0, dtype=float)
        self.v = np.array(v0, dtype=float)
        self.history = {'t': [], 'r': [], 'v': []}

    def kinetic_energy(self):
        return 0.5 * self.m * np.dot(self.v, self.v)

    def magnetic_moment(self, B):
        B_mag = np.linalg.norm(B)
        v_perp = np.linalg.norm(self.v - np.dot(self.v, B) * B / B_mag**2 * B_mag)
        return self.m * v_perp**2 / (2 * B_mag)

def boris_step(r, v, E, B, q, m, dt):
    """Boris algorithm: one time step."""
    # Half electric push
    v_minus = v + (q * dt / (2 * m)) * E

    # Magnetic rotation
    # t_vec = (q dt/2m) B는 반 회전각 벡터입니다; Boris 트릭(Boris trick)은
    # 완전한 회전을 두 번의 외적으로 분해하여 완전한 회전 행렬 계산을 피하면서
    # 동시에 정확한 시간 가역성(time-reversibility)을 유지합니다.
    t_vec = (q * dt / (2 * m)) * B
    t_mag_sq = np.dot(t_vec, t_vec)
    # s_vec = 2t/(1+|t|²)는 회전에 대한 배각 공식(double-angle formula)입니다:
    # 이 정확한 형태가 정적 B 필드에서 운동 에너지를 기계 정밀도 수준까지 보존하며,
    # 이것이 단순 오일러 적분(Euler integration) 대신 Boris 알고리즘을 선호하는 이유입니다.
    s_vec = 2 * t_vec / (1 + t_mag_sq)

    v_prime = v_minus + np.cross(v_minus, t_vec)
    v_plus = v_minus + np.cross(v_prime, s_vec)

    # Half electric push
    v_new = v_plus + (q * dt / (2 * m)) * E

    # 이미 갱신된 속도를 사용한 위치 업데이트(도약 개구리 순서, leap-frog ordering)는
    # 위치와 속도 모두에서 2차 정확도(second-order accuracy)를 보장합니다.
    r_new = r + v_new * dt

    return r_new, v_new

def integrate_orbit(particle, E_func, B_func, t_final, dt):
    """Integrate particle orbit."""
    t = 0
    while t < t_final:
        E = E_func(particle.r, t)
        B = B_func(particle.r, t)

        particle.r, particle.v = boris_step(particle.r, particle.v, E, B,
                                             particle.q, particle.m, dt)

        particle.history['t'].append(t)
        particle.history['r'].append(particle.r.copy())
        particle.history['v'].append(particle.v.copy())

        t += dt

# Field configurations
def uniform_B(r, t, B0=0.1):
    """Uniform magnetic field in z direction."""
    return np.array([0, 0, 0]), np.array([0, 0, B0])

def E_cross_B(r, t, E0=1000, B0=0.1):
    """Uniform E and B fields (E×B drift)."""
    return np.array([E0, 0, 0]), np.array([0, 0, B0])

def gradient_B(r, t, B0=0.1, alpha=0.1):
    """Gradient in B field."""
    x, y, z = r
    B = B0 * (1 + alpha * x)
    return np.array([0, 0, 0]), np.array([0, 0, B])

def magnetic_mirror(r, t, B0=0.1, L=1.0):
    """Magnetic mirror field."""
    x, y, z = r
    B_mag = B0 * (1 + (z / L)**2)
    # Simplified: B field in z direction with magnitude varying
    # For full mirror: need radial component too
    Bz = B_mag
    Br = -B0 * (z / L**2) * np.sqrt(x**2 + y**2)  # from div B = 0
    theta = np.arctan2(y, x)
    Bx = Br * np.cos(theta)
    By = Br * np.sin(theta)
    return np.array([0, 0, 0]), np.array([Bx, By, Bz])

# Test Case 1: Gyration in uniform B
print("Test 1: Gyration in uniform B field")
print("=" * 50)

B0 = 0.1  # T
v_perp = 1e6  # m/s
electron = Particle(q=-e, m=m_e, r0=[0, 0, 0], v0=[v_perp, 0, 0])

omega_c = e * B0 / m_e
rho_c = m_e * v_perp / (e * B0)
T_c = 2 * np.pi / omega_c

print(f"Cyclotron frequency: {omega_c:.3e} rad/s")
print(f"Gyroradius: {rho_c * 1e3:.3f} mm")
print(f"Cyclotron period: {T_c * 1e9:.3f} ns")

dt = T_c / 100
t_final = 3 * T_c

integrate_orbit(electron, lambda r, t: uniform_B(r, t, B0), lambda r, t: (np.zeros(3), np.zeros(3)),
                t_final, dt)

r_hist = np.array(electron.history['r'])

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121)
ax1.plot(r_hist[:, 0] * 1e3, r_hist[:, 1] * 1e3, 'b-', linewidth=1)
ax1.plot(r_hist[0, 0] * 1e3, r_hist[0, 1] * 1e3, 'go', markersize=8, label='Start')
ax1.set_xlabel('x (mm)')
ax1.set_ylabel('y (mm)')
ax1.set_title('Electron Gyration in Uniform B')
ax1.set_aspect('equal')
ax1.legend()
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(r_hist[:, 0] * 1e3, r_hist[:, 1] * 1e3, r_hist[:, 2] * 1e3, 'b-', linewidth=1)
ax2.set_xlabel('x (mm)')
ax2.set_ylabel('y (mm)')
ax2.set_zlabel('z (mm)')
ax2.set_title('3D View')

plt.tight_layout()
plt.savefig('project1_gyration.png', dpi=150)
plt.show()

# Verify gyroradius
x_max = np.max(np.abs(r_hist[:, 0]))
print(f"\nNumerical gyroradius: {x_max * 1e3:.3f} mm")
print(f"Theoretical: {rho_c * 1e3:.3f} mm")
print(f"Relative error: {100 * (x_max - rho_c) / rho_c:.2f}%")

# Test Case 2: E×B drift
print("\n\nTest 2: E×B drift")
print("=" * 50)

E0 = 1000  # V/m
B0 = 0.1   # T

v_ExB = E0 / B0
print(f"E×B drift velocity: {v_ExB:.2f} m/s")

proton = Particle(q=e, m=m_p, r0=[0, 0, 0], v0=[0, 1e5, 0])

t_final = 1e-4
dt = 1e-7

integrate_orbit(proton, lambda r, t: E_cross_B(r, t, E0, B0),
                lambda r, t: (np.zeros(3), np.zeros(3)), t_final, dt)

r_hist = np.array(proton.history['r'])
t_hist = np.array(proton.history['t'])

# Calculate drift velocity
drift_y = (r_hist[-1, 1] - r_hist[0, 1]) / t_final

print(f"\nNumerical drift velocity: {drift_y:.2f} m/s")
print(f"Theoretical E×B drift: {v_ExB:.2f} m/s")
print(f"Relative error: {100 * (drift_y - v_ExB) / v_ExB:.2f}%")

plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.plot(r_hist[:, 0] * 1e3, r_hist[:, 1] * 1e3, 'b-', linewidth=1)
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('E×B Drift (x-y plane)')
plt.grid(alpha=0.3)

plt.subplot(122)
plt.plot(t_hist * 1e6, r_hist[:, 1] * 1e3, 'b-', linewidth=2)
plt.xlabel('Time (μs)')
plt.ylabel('y position (mm)')
plt.title('Drift Motion')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('project1_ExB_drift.png', dpi=150)
plt.show()

# Test Case 3: Magnetic Mirror
print("\n\nTest 3: Magnetic Mirror")
print("=" * 50)

B0 = 0.1
L = 0.5
v_parallel = 1e5
v_perp = 5e4

electron_mirror = Particle(q=-e, m=m_e, r0=[0, 0, -0.4], v0=[v_perp, 0, v_parallel])

t_final = 5e-6
dt = 1e-9

integrate_orbit(electron_mirror, lambda r, t: magnetic_mirror(r, t, B0, L),
                lambda r, t: (np.zeros(3), np.zeros(3)), t_final, dt)

r_hist = np.array(electron_mirror.history['r'])
v_hist = np.array(electron_mirror.history['v'])

# Calculate magnetic moment
B_field = np.array([magnetic_mirror(r, 0, B0, L)[1] for r in r_hist])
B_mag = np.linalg.norm(B_field, axis=1)

mu_values = []
for i, (v, B) in enumerate(zip(v_hist, B_field)):
    B_unit = B / B_mag[i]
    v_par = np.dot(v, B_unit)
    v_perp_mag = np.sqrt(np.dot(v, v) - v_par**2)
    mu = m_e * v_perp_mag**2 / (2 * B_mag[i])
    mu_values.append(mu)

mu_values = np.array(mu_values)

print(f"Magnetic moment μ:")
print(f"  Mean: {np.mean(mu_values):.3e} J/T")
print(f"  Std: {np.std(mu_values):.3e} J/T")
print(f"  Variation: {100 * np.std(mu_values) / np.mean(mu_values):.2f}%")

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121)
ax1.plot(r_hist[:, 2], r_hist[:, 0] * 1e3, 'b-', linewidth=1)
ax1.set_xlabel('z (m)')
ax1.set_ylabel('x (mm)')
ax1.set_title('Mirror Bounce Motion (side view)')
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(122)
ax2.plot(np.array(electron_mirror.history['t']) * 1e6, mu_values / mu_values[0], 'b-', linewidth=1)
ax2.axhline(1.0, color='r', linestyle='--', label='Perfect conservation')
ax2.set_xlabel('Time (μs)')
ax2.set_ylabel('μ / μ₀')
ax2.set_title('Magnetic Moment Conservation')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('project1_mirror.png', dpi=150)
plt.show()

print("\nProject 1 complete!")
```

### 1.5 확장

1. **손실 원뿔**: 거울 구성에서 피치각을 변경하고 손실 원뿔 각도를 결정합니다.
2. **토카막 궤도**: 토로이달 장 $B_\phi \propto 1/R$과 폴로이달 장 $B_\theta$를 구현하여 바나나 궤도를 봅니다.
3. **푸앵카레 단면**: 주기 궤도의 경우, 위상 공간(예: $x=0$ 교차점에서 $v_\parallel$ 대 $z$)을 플롯합니다.
4. **상대론적 입자**: 상대론적 에너지로 확장하고 비상대론적과 비교합니다.
5. **통계 앙상블**: 서로 다른 초기 조건으로 많은 입자를 실행하고 표류 통계를 계산합니다.

---

## 프로젝트 2: 분산 관계 해결기

### 2.1 개요

**목표**: 플라즈마에서 정전기 및 전자기파 분산 관계를 위한 일반 해결기를 만듭니다. 분산 다이어그램, CMA 다이어그램을 생성하고 파동 모드를 식별합니다.

**난이도**: ⭐⭐⭐⭐

**예상 시간**: 12–20시간

**개발 기술**:
- 복잡한 분산 관계를 위한 근 찾기
- 냉각 및 온난 플라즈마 파동 이론 이해
- 다차원 데이터 시각화(ω-k 다이어그램, CMA 다이어그램)
- 파동 모드 및 공명/차단 해석

### 2.2 물리 배경

플라즈마에서 전자기파의 분산 관계는 플라즈마 전류 $\mathbf{J}$를 포함한 맥스웰 방정식에서 유도됩니다:

$$\nabla \times \nabla \times \mathbf{E} + \frac{\omega^2}{c^2} \overleftrightarrow{\epsilon} \cdot \mathbf{E} = 0$$

여기서 $\overleftrightarrow{\epsilon}$는 유전 텐서입니다.

**냉각, 자화 플라즈마**의 경우, 주 좌표계에서 유전 텐서는:

$$\overleftrightarrow{\epsilon} = \begin{pmatrix}
S & -iD & 0 \\
iD & S & 0 \\
0 & 0 & P
\end{pmatrix}$$

여기서 **스틱스 매개변수**는:

$$S = 1 - \sum_s \frac{\omega_{ps}^2}{\omega^2 - \omega_{cs}^2}$$
$$D = \sum_s \frac{\omega_{cs}}{\omega} \frac{\omega_{ps}^2}{\omega^2 - \omega_{cs}^2}$$
$$P = 1 - \sum_s \frac{\omega_{ps}^2}{\omega^2}$$

자기장에 대한 각도 $\theta$로 전파하는 경우, 분산 관계는:

$$A n^4 - B n^2 + C = 0$$

여기서 $n = ck/\omega$는 굴절률이고:

$$A = S \sin^2\theta + P \cos^2\theta$$
$$B = (S^2 - D^2) \sin^2\theta + PS(1 + \cos^2\theta)$$
$$C = P(S^2 - D^2)$$

**온난 플라즈마** 보정은 열 항(봄-그로스, 란다우 감쇠 등)을 추가합니다.

### 2.3 구현 가이드

**단계 1: 스틱스 매개변수**

```python
def stix_parameters(omega, omega_ps, omega_cs):
    """
    Calculate Stix parameters S, D, P.

    Parameters:
    omega: wave frequency (rad/s)
    omega_ps: plasma frequencies (array for each species)
    omega_cs: cyclotron frequencies (array, signed)

    Returns:
    S, D, P
    """
    S = 1 - np.sum(omega_ps**2 / (omega**2 - omega_cs**2))
    D = np.sum((omega_cs / omega) * omega_ps**2 / (omega**2 - omega_cs**2))
    P = 1 - np.sum(omega_ps**2 / omega**2)

    return S, D, P
```

**단계 2: 분산 관계**

```python
def cold_plasma_dispersion(omega, k, theta, omega_ps, omega_cs):
    """
    Cold plasma dispersion relation: A n^4 - B n^2 + C = 0.

    Returns the LHS (should be zero for a wave mode).
    """
    S, D, P = stix_parameters(omega, omega_ps, omega_cs)

    c = 3e8
    n = c * k / omega  # refractive index

    sin2 = np.sin(theta)**2
    cos2 = np.cos(theta)**2

    A = S * sin2 + P * cos2
    B = (S**2 - D**2) * sin2 + P * S * (1 + cos2)
    C = P * (S**2 - D**2)

    return A * n**4 - B * n**2 + C
```

**단계 3: 근 찾기**

주어진 $k$와 $\theta$에 대해, 분산 관계가 만족되도록 $\omega$를 찾습니다:

```python
from scipy.optimize import fsolve, brentq

def find_omega(k, theta, omega_guess, omega_ps, omega_cs):
    """
    Find wave frequency ω for given k and θ.
    """
    def dispersion_func(omega):
        if omega <= 0:
            return 1e10  # penalize negative frequencies
        return cold_plasma_dispersion(omega, k, theta, omega_ps, omega_cs)

    omega_solution = fsolve(dispersion_func, omega_guess)[0]

    return omega_solution
```

**단계 4: 분산 다이어그램 생성**

```python
def dispersion_diagram(k_range, theta, omega_ps, omega_cs, omega_guesses):
    """
    Generate ω(k) dispersion diagram for multiple modes.

    Parameters:
    k_range: array of wavenumbers
    theta: propagation angle
    omega_guesses: list of initial guesses for different modes

    Returns:
    omegas: list of arrays, one per mode
    """
    omegas = [[] for _ in omega_guesses]

    for k in k_range:
        for i, omega_guess in enumerate(omega_guesses):
            try:
                omega_sol = find_omega(k, theta, omega_guess, omega_ps, omega_cs)
                if omega_sol > 0 and np.abs(cold_plasma_dispersion(omega_sol, k, theta,
                                                                     omega_ps, omega_cs)) < 1e-3:
                    omegas[i].append(omega_sol)
                else:
                    omegas[i].append(np.nan)
            except:
                omegas[i].append(np.nan)

    return [np.array(omega_list) for omega_list in omegas]
```

**단계 5: CMA 다이어그램**

**클레모-멀렐리-앨리스(CMA) 다이어그램**은 $(\omega_{pe}^2/\omega_{ce}^2, \omega^2/\omega_{ce}^2)$ 공간에서 서로 다른 파동 모드가 존재하는 영역을 보여줍니다.

차단과 공명:
- **R 차단**: $\omega = \omega_R = \frac{1}{2}(\omega_{ce} + \sqrt{\omega_{ce}^2 + 4\omega_{pe}^2})$
- **L 차단**: $\omega = \omega_L = \frac{1}{2}(-\omega_{ce} + \sqrt{\omega_{ce}^2 + 4\omega_{pe}^2})$
- **P 차단**: $\omega = \omega_{pe}$
- **상부 하이브리드**: $\omega_{UH}^2 = \omega_{pe}^2 + \omega_{ce}^2$
- **하부 하이브리드**: $\omega_{LH}^2 = \frac{\omega_{pi}^2}{1 + \omega_{pe}^2 / \omega_{ce}^2}$

### 2.4 완전한 코드

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, brentq

# Constants
e = 1.6e-19
m_e = 9.11e-31
m_p = 1.67e-27
c = 3e8

def stix_parameters(omega, omega_ps, omega_cs):
    """Calculate Stix parameters S, D, P."""
    S = 1 - np.sum(omega_ps**2 / (omega**2 - omega_cs**2 + 1e-10))
    D = np.sum((omega_cs / omega) * omega_ps**2 / (omega**2 - omega_cs**2 + 1e-10))
    P = 1 - np.sum(omega_ps**2 / (omega**2 + 1e-10))

    return S, D, P

def cold_plasma_dispersion(omega, k, theta, omega_ps, omega_cs):
    """Cold plasma dispersion relation."""
    S, D, P = stix_parameters(omega, omega_ps, omega_cs)

    n = c * k / (omega + 1e-10)

    sin2 = np.sin(theta)**2
    cos2 = np.cos(theta)**2

    A = S * sin2 + P * cos2
    B = (S**2 - D**2) * sin2 + P * S * (1 + cos2)
    C = P * (S**2 - D**2)

    return A * n**4 - B * n**2 + C

# Plasma parameters
n = 1e18  # m^-3
B = 0.05  # T

omega_pe = np.sqrt(n * e**2 / (m_e * 8.85e-12))
omega_pi = np.sqrt(n * e**2 / (m_p * 8.85e-12))
omega_ce = e * B / m_e
omega_ci = e * B / m_p

omega_ps = np.array([omega_pe, omega_pi])
omega_cs = np.array([omega_ce, -omega_ci])  # electrons +, ions -

print("Plasma parameters:")
print(f"  ω_pe / (2π) = {omega_pe / (2 * np.pi):.3e} Hz")
print(f"  ω_ce / (2π) = {omega_ce / (2 * np.pi):.3e} Hz")
print(f"  ω_pi / (2π) = {omega_pi / (2 * np.pi):.3e} Hz")
print(f"  ω_ci / (2π) = {omega_ci / (2 * np.pi):.3e} Hz")
print()

# Cutoff frequencies
omega_R = 0.5 * (omega_ce + np.sqrt(omega_ce**2 + 4 * omega_pe**2))
omega_L = 0.5 * (-omega_ce + np.sqrt(omega_ce**2 + 4 * omega_pe**2))
omega_UH = np.sqrt(omega_pe**2 + omega_ce**2)
omega_LH = omega_pi / np.sqrt(1 + omega_pe**2 / omega_ce**2)

print("Characteristic frequencies:")
print(f"  R cutoff: {omega_R / (2 * np.pi):.3e} Hz")
print(f"  L cutoff: {omega_L / (2 * np.pi):.3e} Hz")
print(f"  Upper hybrid: {omega_UH / (2 * np.pi):.3e} Hz")
print(f"  Lower hybrid: {omega_LH / (2 * np.pi):.3e} Hz")
print()

# Dispersion diagram: parallel propagation (θ = 0)
k_range = np.linspace(1, 1000, 300)
theta = 0  # parallel

# Find O-mode (P = 0 → ω = ω_pe) and X-mode (more complex)
omega_O = []
omega_R_mode = []
omega_L_mode = []

for k in k_range:
    # O-mode: ω² = ω_pe² + k² c²
    omega_O.append(np.sqrt(omega_pe**2 + (k * c)**2))

    # R-mode: solve S - n² = 0
    def R_dispersion(omega):
        S, D, P = stix_parameters(omega, omega_ps, omega_cs)
        n = c * k / omega
        return S - n**2

    # L-mode: solve S - n² = 0 (but different branch)

    try:
        omega_R_sol = fsolve(R_dispersion, omega_R * 1.5)[0]
        if omega_R_sol > omega_R and omega_R_sol < 10 * omega_ce:
            omega_R_mode.append(omega_R_sol)
        else:
            omega_R_mode.append(np.nan)
    except:
        omega_R_mode.append(np.nan)

    try:
        omega_L_sol = fsolve(R_dispersion, omega_L * 0.9)[0]
        if omega_L_sol > 0 and omega_L_sol < omega_L * 1.5:
            omega_L_mode.append(omega_L_sol)
        else:
            omega_L_mode.append(np.nan)
    except:
        omega_L_mode.append(np.nan)

omega_O = np.array(omega_O)
omega_R_mode = np.array(omega_R_mode)
omega_L_mode = np.array(omega_L_mode)

# Plot dispersion diagram
plt.figure(figsize=(12, 7))

plt.plot(k_range, omega_O / omega_ce, 'b-', linewidth=2, label='O-mode')
plt.plot(k_range, omega_R_mode / omega_ce, 'r-', linewidth=2, label='R-mode (X-mode branch)')
plt.plot(k_range, omega_L_mode / omega_ce, 'g-', linewidth=2, label='L-mode (X-mode branch)')

# Light line
plt.plot(k_range, k_range * c / omega_ce, 'k--', linewidth=1, alpha=0.5, label='Light line ω = ck')

# Cutoffs and resonances
plt.axhline(omega_R / omega_ce, color='r', linestyle=':', alpha=0.7, label=f'R cutoff')
plt.axhline(omega_L / omega_ce, color='g', linestyle=':', alpha=0.7, label=f'L cutoff')
plt.axhline(omega_pe / omega_ce, color='b', linestyle=':', alpha=0.7, label=f'ω_pe')
plt.axhline(omega_UH / omega_ce, color='m', linestyle=':', alpha=0.7, label=f'Upper hybrid')
plt.axhline(1.0, color='orange', linestyle=':', alpha=0.7, label=f'ω_ce')

plt.xlabel('Wavenumber k (m⁻¹)', fontsize=12)
plt.ylabel('ω / ω_ce', fontsize=12)
plt.title('Cold Plasma Dispersion: Parallel Propagation (θ = 0)', fontsize=14)
plt.legend(fontsize=10, loc='upper left')
plt.grid(alpha=0.3)
plt.xlim(0, 1000)
plt.ylim(0, 20)
plt.tight_layout()
plt.savefig('project2_dispersion_parallel.png', dpi=150)
plt.show()

print("Project 2: Dispersion diagram complete!")
```

### 2.5 확장

1. **비스듬한 전파**: $\theta$를 0에서 $\pi/2$로 변경하고 $\omega(k, \theta)$를 3D 표면으로 플롯합니다.
2. **온난 플라즈마**: 열 보정 추가(봄-그로스: $\omega^2 = \omega_{pe}^2 + 3k^2 v_{te}^2$).
3. **이온 음향파**: 이온 응답 포함, 란다우 감쇠와 함께 IA 분산 플롯.
4. **CMA 다이어그램**: $(X, Y)$ 공간에서 전파/감쇠 영역 플롯 ($X = \omega_{pe}^2/\omega^2$, $Y = \omega_{ce}/\omega$).
5. **공명 원뿔**: 휘슬러파의 경우, 주파수 대 공명 원뿔 각도 플롯.

---

## 프로젝트 3: 1D 블라소프-푸아송 해결기

### 3.1 개요

**목표**: 반라그랑주(분할) 방법을 사용하여 1D-1V 블라소프-푸아송 해결기를 구현합니다. 랭뮤어 진동, 란다우 감쇠, 쌍류 불안정성, 범프-온-테일 불안정성을 시뮬레이션합니다.

**난이도**: ⭐⭐⭐⭐⭐

**예상 시간**: 20–30시간

**개발 기술**:
- PDE를 위한 고급 수치 방법
- 운동 플라즈마 시뮬레이션
- FFT 기반 푸아송 해결기
- 위상 공간 시각화
- 선형 이론과 비교(란다우 감쇠율, 성장률)

### 3.2 물리 배경

**1D 블라소프 방정식**은 전자 분포 함수 $f(x, v, t)$의 진화를 설명합니다:

$$\frac{\partial f}{\partial t} + v \frac{\partial f}{\partial x} + \frac{q E}{m} \frac{\partial f}{\partial v} = 0$$

전기장을 위한 **푸아송 방정식**과 결합:

$$\frac{\partial E}{\partial x} = \frac{q}{\epsilon_0} (n_i - n_e)$$

여기서 전자 밀도는:
$$n_e(x, t) = \int f(x, v, t) \, dv$$

이온은 고정된 중성화 배경으로 가정: $n_i = n_0 = \text{const}$.

**분할 방법**: 블라소프 방정식을 두 단계로 분할:

1. **x에서 이류**(자유 흐름):
   $$\frac{\partial f}{\partial t} + v \frac{\partial f}{\partial x} = 0$$

2. **v에서 이류**(가속):
   $$\frac{\partial f}{\partial t} + \frac{q E}{m} \frac{\partial f}{\partial v} = 0$$

각 단계는 특성의 **역추적**으로 정확히 해결됩니다.

### 3.3 구현 가이드

**단계 1: 초기화**

위상 공간 그리드 설정:
- $x \in [0, L]$에서 $N_x$ 그리드 점
- $v \in [v_{min}, v_{max}]$에서 $N_v$ 그리드 점
- 초기 분포: 란다우 감쇠를 위한 $f_0(x, v) = f_0(v) (1 + \alpha \cos(kx))$

**단계 2: 푸아송 해결기(FFT)**

```python
def solve_poisson_fft(rho, dx, L):
    """
    Solve Poisson equation: d²φ/dx² = -ρ/ε₀ using FFT.

    Returns electric field E = -dφ/dx.
    """
    epsilon_0 = 8.85e-12
    Nx = len(rho)

    # Fourier transform of rho
    rho_k = np.fft.fft(rho)

    # Wavenumbers
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    # k[0] = 0 (DC 모드)는 φ_k = -ρ_k/(ε₀k²)에서 0으로 나누기를 발생시킵니다.
    # k[0] = 1로 임시 설정하면 분모가 유한해지며; 아래에서 phi_k[0]이
    # 즉시 덮어써지므로 물리적 영향은 전혀 없습니다.
    k[0] = 1  # avoid division by zero (DC component is arbitrary for periodic)

    # Fourier transform of potential: φ_k = -ρ_k / (ε₀ k²)
    phi_k = -rho_k / (epsilon_0 * k**2)
    # DC 퍼텐셜을 0으로 설정하면 가산적 게이지 자유도(additive gauge freedom)가
    # 고정됩니다: 주기적 영역에서는 퍼텐셜의 차이만이 물리적이며, 비-zero 평균값은
    # 동역학에 영향을 주지 않으면서 모든 입자 에너지를 이동시킬 뿐입니다.
    phi_k[0] = 0  # set DC component to zero

    # 푸리에 공간에서의 미분(스펙트럴 미분, spectral differentiation)은 dφ/dx에 대한
    # 유한 차분 격자법(finite-difference stencil)의 절단 오차를 완전히 제거합니다.
    # Electric field: E = -dφ/dx → E_k = i k φ_k
    E_k = 1j * k * phi_k

    # Inverse FFT
    E = np.real(np.fft.ifft(E_k))

    return E
```

**단계 3: x에서 이류**

$\partial f / \partial t + v \partial f / \partial x = 0$을 해결:

각 $v_j$에 대해, 특성은 $x(t) = x_0 + v_j \Delta t$입니다. $f$를 업데이트하기 위해 역추적:

$$f^{n+1}(x_i, v_j) = f^n(x_i - v_j \Delta t, v_j)$$

비그리드 값에 대해 보간(예: 3차 스플라인) 사용.

**단계 4: v에서 이류**

$\partial f / \partial t + (qE/m) \partial f / \partial v = 0$을 해결:

각 $x_i$에 대해, 특성은 $v(t) = v_0 + (q E_i / m) \Delta t$입니다. 업데이트:

$$f^{n+1}(x_i, v_j) = f^n(x_i, v_j - \frac{qE_i}{m} \Delta t)$$

다시 보간 사용.

**단계 5: 시간 단계 루프**

```python
for n in range(Nt):
    # 1. Compute density
    n_e = np.trapz(f, v_grid, axis=1)

    # 2. Solve Poisson
    rho = e * (n_0 - n_e)
    E = solve_poisson_fft(rho, dx, L)

    # 3. Advection in x (half step)
    f = advect_x(f, v_grid, dt/2)

    # 4. Advection in v (full step)
    f = advect_v(f, E, dt)

    # 5. Advection in x (half step)
    f = advect_x(f, v_grid, dt/2)

    # 6. Diagnostics
    energy[n] = compute_energy(f, E)
```

### 3.4 테스트 케이스

**테스트 1: 플라즈마 진동**

작은 정현파 밀도 교란으로 초기화:
$$n_e(x, 0) = n_0 (1 + \epsilon \cos(kx)), \quad f(x, v, 0) = \frac{n_e(x, 0)}{\sqrt{2\pi v_{th}^2}} e^{-v^2/(2v_{th}^2)}$$

전기장은 $\omega_{pe}$에서 진동해야 합니다. 검증:
$$\omega_{pe} = \sqrt{\frac{n_0 e^2}{\epsilon_0 m_e}}$$

**테스트 2: 란다우 감쇠**

동일한 초기 조건 사용. 전기장 진폭은 지수적으로 감쇠해야 합니다:
$$E(t) \propto e^{-\gamma_L t}$$

여기서 란다우 감쇠율은 ($k\lambda_D \ll 1$의 경우):
$$\gamma_L \approx \sqrt{\frac{\pi}{8}} \omega_{pe} (k\lambda_D)^{-3} e^{-1/(2k^2\lambda_D^2)}$$

**테스트 3: 쌍류 불안정성**

두 개의 반대 방향 빔으로 초기화:
$$f(x, v, 0) = \frac{n_0}{2} \left[ \frac{1}{\sqrt{2\pi v_{th}^2}} e^{-(v - v_0)^2 / (2v_{th}^2)} + \frac{1}{\sqrt{2\pi v_{th}^2}} e^{-(v + v_0)^2 / (2v_{th}^2)} \right]$$

$x$에 작은 교란 포함. 불안정성은 지수적으로 성장합니다. 성장률을 측정하고 이론과 비교합니다.

**테스트 4: 범프-온-테일**

벌크 맥스웰리안에 더 높은 속도에서 작은 범프를 추가하여 초기화:
$$f(x, v, 0) = \frac{n_b}{\sqrt{2\pi v_{th}^2}} e^{-v^2/(2v_{th}^2)} + \frac{n_{bump}}{\sqrt{2\pi v_{bump}^2}} e^{-(v - v_{bump})^2/(2v_{bump}^2)}$$

이것은 불안정하고 범프를 평탄화하는 파동을 구동합니다(준선형 확산).

### 3.5 완전한 코드(단순화)

길이 때문에 간소화된 버전을 제공합니다:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Constants
e = 1.6e-19
m_e = 9.11e-31
epsilon_0 = 8.85e-12

# Grid parameters
Nx = 128
Nv = 128
L = 2 * np.pi / 1e5  # spatial domain (one wavelength)
v_max = 5e6  # m/s

x_grid = np.linspace(0, L, Nx, endpoint=False)
v_grid = np.linspace(-v_max, v_max, Nv)
dx = x_grid[1] - x_grid[0]
dv = v_grid[1] - v_grid[0]

# Plasma parameters
n_0 = 1e16  # m^-3
T_e = 1  # eV
v_th = np.sqrt(e * T_e / m_e)
omega_pe = np.sqrt(n_0 * e**2 / (epsilon_0 * m_e))

# Time step (CFL condition)
dt = 0.1 * min(dx / v_max, dv / (e * 1e3 / m_e))  # conservative
Nt = 500

print(f"ω_pe = {omega_pe:.3e} rad/s")
print(f"T_pe = {2 * np.pi / omega_pe:.3e} s")
print(f"dt = {dt:.3e} s")
print(f"Total time = {Nt * dt:.3e} s ({Nt * dt * omega_pe / (2 * np.pi):.2f} plasma periods)")

# Initial distribution: Maxwellian with perturbation
k_pert = 2 * np.pi / L
alpha = 0.01

f = np.zeros((Nx, Nv))
for i, x in enumerate(x_grid):
    n_local = n_0 * (1 + alpha * np.cos(k_pert * x))
    f[i, :] = n_local / (np.sqrt(2 * np.pi) * v_th) * np.exp(-v_grid**2 / (2 * v_th**2))

# Advection functions (using linear interpolation for simplicity)
def advect_x(f, v_grid, dt):
    """Advect in x: f(x - v*dt, v)."""
    f_new = np.zeros_like(f)
    for j, v in enumerate(v_grid):
        shift = v * dt
        x_old = (x_grid - shift) % L  # periodic boundary
        f_new[:, j] = np.interp(x_old, x_grid, f[:, j], period=L)
    return f_new

def advect_v(f, E, dt):
    """Advect in v: f(x, v - (qE/m)*dt)."""
    f_new = np.zeros_like(f)
    for i, x_i in enumerate(x_grid):
        accel = -e * E[i] / m_e
        v_old = v_grid - accel * dt
        # Interpolate (with extrapolation for boundary)
        f_new[i, :] = np.interp(v_old, v_grid, f[i, :], left=0, right=0)
    return f_new

def solve_poisson_fft(rho, dx):
    """Solve Poisson equation using FFT."""
    Nx = len(rho)
    rho_k = np.fft.fft(rho)
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    k[0] = 1  # avoid division by zero
    phi_k = -rho_k / (epsilon_0 * k**2)
    phi_k[0] = 0
    E_k = 1j * k * phi_k
    E = np.real(np.fft.ifft(E_k))
    return E

# Diagnostics
E_history = []
energy_history = []

# Time-stepping loop
for n in range(Nt):
    # 속도 공간에서 f를 적분하여 밀도를 계산합니다: n_e = ∫f dv.
    # 이 모멘트 축소(moment reduction)는 정확하며, 운동론적 Vlasov 시뮬레이션이
    # 어떠한 닫힘 가정(closure assumption) 없이도 유체 모멘트를 복원할 수 있는 이유입니다.
    n_e = np.trapz(f, v_grid, axis=1)

    # Solve Poisson
    # rho = e(n_i - n_e): 이온은 고정 배경(n_i = n_0)이므로, 전자 밀도 요동만이
    # 자기 일관적(self-consistent) 전기장을 구동합니다.
    rho = e * (n_0 - n_e)
    E = solve_poisson_fft(rho, dx)

    # f를 진행시키기 전에 진단량(diagnostics)을 저장하면, 방금 계산된 전기장에
    # 대응하는 상태를 기록하게 됩니다(일관된 스냅샷, consistent snapshot).
    E_history.append(np.max(np.abs(E)))
    field_energy = 0.5 * epsilon_0 * np.sum(E**2) * dx
    kinetic_energy = 0.5 * m_e * np.sum(f * (v_grid[np.newaxis, :]**2) * dx * dv)
    energy_history.append(field_energy + kinetic_energy)

    # Strang 분할(Strang splitting, 반-x, 전체-v, 반-x)은 시간에서 2차 정확도를
    # 달성합니다: 1차 분할은 O(dt) 정확도만 제공하지만, 이 대칭적 배열은 선두
    # 오차 항을 상쇄하여 동일한 비용으로 룽게-쿠타 2(Runge-Kutta 2)의 정확도를 맞춥니다.
    f = advect_x(f, v_grid, dt / 2)
    f = advect_v(f, E, dt)
    f = advect_x(f, v_grid, dt / 2)

    # Print progress
    if n % 50 == 0:
        print(f"Step {n}/{Nt}, E_max = {E_history[-1]:.3e} V/m")

# Plot results
t_grid = np.arange(Nt) * dt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Electric field amplitude vs. time
axes[0, 0].semilogy(t_grid * omega_pe / (2 * np.pi), E_history, 'b-', linewidth=1)
axes[0, 0].set_xlabel('Time (plasma periods)', fontsize=11)
axes[0, 0].set_ylabel('Max |E| (V/m)', fontsize=11)
axes[0, 0].set_title('Electric Field Amplitude (Landau Damping)', fontsize=12)
axes[0, 0].grid(alpha=0.3)

# Fit exponential decay to measure damping rate
t_fit = t_grid[50:200]
E_fit = np.array(E_history[50:200])
log_E = np.log(E_fit)
p = np.polyfit(t_fit, log_E, 1)
gamma_numerical = -p[0]

# Theoretical Landau damping rate
k = k_pert
lambda_D = v_th / omega_pe
kLD = k * lambda_D
gamma_theory = np.sqrt(np.pi / 8) * omega_pe * (kLD)**(-3) * np.exp(-1 / (2 * kLD**2))

axes[0, 0].plot(t_fit * omega_pe / (2 * np.pi), np.exp(p[0] * t_fit + p[1]), 'r--',
                linewidth=2, label=f'Fit: γ = {gamma_numerical:.2e} s⁻¹')
axes[0, 0].legend(fontsize=10)

print(f"\nLandau damping:")
print(f"  Numerical γ = {gamma_numerical:.3e} s⁻¹ ({gamma_numerical/omega_pe:.3e} ω_pe)")
print(f"  Theoretical γ = {gamma_theory:.3e} s⁻¹ ({gamma_theory/omega_pe:.3e} ω_pe)")
print(f"  Relative error = {100 * (gamma_numerical - gamma_theory) / gamma_theory:.1f}%")

# Energy conservation
axes[0, 1].plot(t_grid * omega_pe / (2 * np.pi), np.array(energy_history) / energy_history[0], 'g-',
                linewidth=2)
axes[0, 1].axhline(1.0, color='k', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('Time (plasma periods)', fontsize=11)
axes[0, 1].set_ylabel('Total energy / E(0)', fontsize=11)
axes[0, 1].set_title('Energy Conservation', fontsize=12)
axes[0, 1].grid(alpha=0.3)

# Phase space at initial time
axes[1, 0].contourf(x_grid, v_grid / 1e6, f.T, levels=20, cmap='viridis')
axes[1, 0].set_xlabel('x (m)', fontsize=11)
axes[1, 0].set_ylabel('v (10⁶ m/s)', fontsize=11)
axes[1, 0].set_title('Phase Space f(x, v) at Final Time', fontsize=12)

# Density profile
n_e_final = np.trapz(f, v_grid, axis=1)
axes[1, 1].plot(x_grid, n_e_final / n_0, 'b-', linewidth=2, label='Final')
axes[1, 1].axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Equilibrium')
axes[1, 1].set_xlabel('x (m)', fontsize=11)
axes[1, 1].set_ylabel('n_e / n_0', fontsize=11)
axes[1, 1].set_title('Density Profile', fontsize=12)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('project3_vlasov_landau.png', dpi=150)
plt.show()

print("\nProject 3: Vlasov-Poisson simulation complete!")
```

### 3.6 확장

1. **고차 보간**: 더 나은 정확도를 위해 3차 또는 5차 스플라인 사용.
2. **쌍류 불안정성**: 초기 조건 수정, 성장률 측정, 이론과 비교.
3. **범프-온-테일**: 평탄화 관측, 파동 에너지 포화 측정.
4. **전자 포획**: 포획된 입자에 의해 형성된 위상 공간 소용돌이 플롯.
5. **2D 블라소프**: 2D-2V로 확장(계산 집약적!).
6. **PIC 비교**: 간단한 1D PIC 코드를 구현하고 블라소프와 비교.

---

## 결론

이 세 프로젝트는 전체 강좌의 자료를 종합합니다:

- **프로젝트 1**(입자 궤도)은 레슨 3–4의 단일 입자 이론을 실현합니다.
- **프로젝트 2**(분산 해결기)는 레슨 9–10의 파동 이론을 구현합니다.
- **프로젝트 3**(블라소프-푸아송)은 레슨 6–8의 운동 이론을 다루고 란다우 감쇠와 같은 기본 플라즈마 현상을 보여줍니다.

이러한 프로젝트를 완료하면 현대 플라즈마 물리 연구에서 사용되는 계산 도구에 대한 실무 경험을 갖게 됩니다. 이러한 방법은 다음으로 확장됩니다:

- **입자-셀(PIC)** 코드: 무충돌 플라즈마 시뮬레이션(예: 레이저-플라즈마 상호작용, 우주 물리)
- **자이로운동론** 코드: 토카막 난류(예: GENE, GYRO, GS2)
- **MHD** 코드: 핵융합 평형 및 안정성(예: NIMROD, M3D-C1)

플라즈마 물리 과정을 완료하신 것을 축하합니다! 이제 플라즈마 물리의 이론과 계산 모두에 대한 견고한 기초를 갖추셨습니다.

---

## 연습 문제

### 연습 1: 보리스 알고리즘(Boris Algorithm) 수렴 연구

보리스 알고리즘은 시간에 대해 2차 정확도를 가집니다. 시간 단계 크기에 따라 선회 반지름 오차가 어떻게 변하는지 측정하여 이를 실험적으로 검증하세요.

**단계**:
1. $\mathbf{B} = 0.1\,\text{T}\,\hat{\mathbf{z}}$의 균일한 자기장에서 $v_\perp = 10^6\,\text{m/s}$, 초기 평행 속도 없이 선회하는 전자를 설정합니다.
2. 해석적 선회 반지름 $\rho_c = m_e v_\perp / (eB)$와 사이클로트론 주기 $T_c = 2\pi m_e / (eB)$를 계산합니다.
3. $N \in \{10, 20, 50, 100, 200, 500\}$에 대해 $\Delta t = T_c / N$의 시간 단계를 사용하여 정확히 10 사이클로트론 주기 동안 시뮬레이션을 실행합니다.
4. 10주기 후, 위치 오차를 측정합니다: 입자의 위치가 출발점에서 벗어난 편차(궤도는 정확히 닫혀야 함).
5. 위치 오차 대 $\Delta t$를 로그-로그 눈금에 플롯하고 기울기를 맞춥니다. 기울기가 2차 수렴(second-order convergence)에 일치하는 2에 가까운지 확인합니다.
6. 각 시간 단계 크기에서 에너지 변동 $\Delta E_{kin} / E_{kin,0}$을 측정하고 모든 $N$에 대해 유계 상태(증가하지 않음)임을 검증합니다.

**기대 결과**: 위치 오차 $\propto (\Delta t)^2$; 에너지는 $\Delta t$ 크기에 무관하게 기계 정밀도 수준에서 보존됩니다.

---

### 연습 2: 자기 거울(Magnetic Mirror)에서의 손실 원뿔(Loss Cone)

자기 거울은 피치각(pitch angle)이 손실 원뿔 각도를 초과하는 입자만 가둡니다. 손실 원뿔 각도를 해석적으로 유도하고 입자 궤도 시뮬레이터를 사용하여 수치적으로 검증하세요.

**단계**:
1. 거울비(mirror ratio) $R_m = B_{max}/B_{min}$ ($z=0$에서 $B_{min} = 0.1\,\text{T}$, 거울점에서 $B_{max} = 0.5\,\text{T}$ 사용)의 자기 거울에 대해 손실 원뿔 반각(half-angle)을 유도합니다:
   $$\sin^2\alpha_{lc} = \frac{B_{min}}{B_{max}} = \frac{1}{R_m}$$
2. 중간면($z=0$)에서 같은 속력 $v = 5 \times 10^6\,\text{m/s}$이지만 피치각 $\alpha$가 $0°$에서 $90°$ 사이에서 균일하게 샘플링된 50개의 전자를 발사합니다($\alpha$는 $\mathbf{v}$와 $\mathbf{B}$ 사이의 각도).
3. 1.3절의 자기 거울 장을 사용하여 각 궤도를 $t_{final} = 50\,\mu\text{s}$ 동안 적분합니다.
4. 각 입자를 구속됨(바운싱) 또는 손실됨($|z| > z_{mirror}$ 도달)으로 분류하고 초기 피치각을 기록합니다.
5. 피치각에 대한 구속 결과를 플롯합니다. 이론적 손실 원뿔 경계를 표시하고 수치 결과와 비교합니다.
6. 등방성 분포(isotropic distribution)에서 구속될 입자의 비율을 추정합니다.

**힌트**: 단열 불변량(adiabatic invariant) $\mu = m v_\perp^2 / (2B)$는 보존됩니다. 이를 사용하여 거울점에서의 피치각을 유도하고 반사 조건을 결정하세요.

---

### 연습 3: CMA 다이어그램에서 차단과 공명 식별

클레모-멀렐리-앨리스(Clemmow-Mullaly-Allis, CMA) 다이어그램은 차단(cutoff)과 공명(resonance)에 따라 모든 냉각 플라즈마 파동 모드를 정리합니다. 이 다이어그램을 수치적으로 구성하고 명명된 파동 모드를 식별하세요.

**단계**:
1. 고정 자기장 $B_0 = 0.05\,\text{T}$를 선택합니다. 무차원 축을 정의합니다:
   - $X = \omega_{pe}^2 / \omega^2$ (밀도 매개변수, $n$ 또는 $\omega$ 변경으로 조정)
   - $Y = \omega_{ce} / \omega$ (자화 매개변수)
2. $X \in [0, 4]$, $Y \in [0, 3]$의 $(X, Y)$ 값 그리드에서 각 점에 대해 스틱스 매개변수(Stix parameters) $S$, $D$, $P$를 계산합니다($\omega$는 고정하고 $n$을 변경하여 $X$를 조정).
3. 차단선을 그립니다:
   - $P = 0$ (O모드 차단: $\omega = \omega_{pe}$, 즉 $X = 1$)
   - $R = S + D = 0$ (R 차단)
   - $L = S - D = 0$ (L 차단)
4. 공명선을 그립니다:
   - $S = 0$ (상부 및 하부 하이브리드 공명)
   - $\tan^2\theta = -P/S$ ($\theta = 0$ 및 $\theta = \pi/2$에 대해, 평행 및 수직 공명)
5. 파동 모드가 전파하는 영역(두 $n_\pm^2 > 0$, 하나만 양수, 또는 둘 다 음수/감쇠)에 따라 색을 칠합니다.
6. 영역에 표준 이름으로 레이블을 붙입니다: O모드, X모드, R파, L파, 휘슬러 모드(whistler mode), 하부 하이브리드파.

**참고**: 다이어그램은 T. H. Stix, *Waves in Plasmas* (AIP, 1992)의 그림 1-8을 재현해야 합니다.

---

### 연습 4: 쌍류 불안정성(Two-Stream Instability) 성장률 측정

쌍류 불안정성은 가장 중요한 운동 플라즈마 불안정성 중 하나입니다. 성장률을 수치적으로 측정하고 해석적 예측과 비교하세요.

**단계**:
1. 프로젝트 3의 블라소프-푸아송(Vlasov-Poisson) 해결기를 수정하여 쌍빔(two-beam) 초기 조건을 수용하도록 합니다:
   $$f_0(x, v) = \frac{n_0}{2}\left[\mathcal{M}(v - v_0) + \mathcal{M}(v + v_0)\right](1 + \alpha\cos(k_0 x))$$
   여기서 $\mathcal{M}(v) = (2\pi v_{th}^2)^{-1/2}\exp(-v^2/2v_{th}^2)$, 빔 속력 $v_0 = 3v_{th}$, $v_{th} = 10^6\,\text{m/s}$, $\alpha = 0.01$.
2. 교란 파수(perturbation wavenumber) $k_0$를 불안정 대역에 선택합니다: 대칭 빔의 경우 불안정성은 $k_0 \approx \omega_{pe} / v_0$ 근방에서 가장 강합니다. 냉각 빔 분산 관계(cold-beam dispersion relation)를 사용하여 이것이 불안정 영역에 있음을 검증합니다:
   $$1 = \frac{\omega_{pe}^2/2}{(\omega - kv_0)^2} + \frac{\omega_{pe}^2/2}{(\omega + kv_0)^2}$$
3. 시뮬레이션을 실행하고 각 시간 단계에서 $\max_x |E(x, t)|$를 기록합니다.
4. 선형 성장 구간($\ln|E|$가 시간에 선형으로 증가하는 구간)을 식별하고, 직선 적합(linear fit)으로 수치 성장률 $\gamma_{num}$을 측정합니다.
5. $\gamma_{num}$을 냉각 빔 분산 관계를 수치적으로 풀어(`numpy.roots`를 다항식 형태에 사용) 얻은 $\omega$의 허수부(imaginary part)와 비교합니다.
6. 위상 공간 $f(x, v)$를 세 시점에서 플롯합니다: 초기, 포화 중간, 포화 후. 포화 시 나타나는 소용돌이 구조(위상 공간 홀, phase space holes)를 설명합니다.

**기대 결과**: 선형 구간에서 $|E| \propto e^{\gamma t}$이며, $v_0 \gg v_{th}$이고 $k = \omega_{pe}/v_0$일 때 $\gamma \approx \omega_{pe}/2\sqrt{2}$.

---

### 연습 5: 통합 소프로젝트 — 휘슬러파(Whistler Wave) 전파와 분산

휘슬러파는 전자 사이클로트론 주파수 아래에서 전파하는 우원 편파(right-hand circularly polarized) 전자기파입니다. 이 파는 복사대(radiation belt) 역학과 전리층(ionospheric) 통신에서 중요한 역할을 합니다. 세 가지 주요 프로젝트를 모두 연결하는 통합 소프로젝트를 설계하고 수행하세요.

**A부 — 분산 다이어그램** (프로젝트 2 코드 사용):
1. 휘슬러파 분기($\omega < \omega_{ce}$, 평행 전파, R모드)를 계산하고 $\omega$-$k$ 다이어그램에 플롯합니다.
2. 군속도(group velocity) $v_g = d\omega/dk$를 주파수의 함수로 겹쳐 플롯하고 최대 군속도의 주파수를 식별합니다.
3. $\omega_{ce} \ll \omega_{pe}$ 한계에서 군속도가 $v_g \approx 2c\sqrt{\omega/\omega_{pe}^2 \cdot \omega_{ce}}$임을 보이고 수치적으로 검증합니다.

**B부 — 시간 분산** (해석적 계산):
1. 번개 방전이 광대역 충격을 생성합니다. 두 주파수 $f_1 = 5\,\text{kHz}$와 $f_2 = 10\,\text{kHz}$가 전리층을 통해 전파하는 것을 고려합니다($n_e = 10^{10}\,\text{m}^{-3}$, $B_0 = 5 \times 10^{-5}\,\text{T}$, $L = 1000\,\text{km}$ 사용).
2. 각 주파수의 이동 시간 $t = L/v_g(\omega)$를 계산합니다.
3. 도착 시간 차이가 "휘슬러 분산(whistler dispersion)"입니다. 시간 분리 $\Delta t = t_1 - t_2$를 추정하고 관측된 휘슬러 분산(일반적으로 1~10초)과 비교합니다.

**C부 — 입자 공명** (해석적):
1. 전자는 사이클로트론 공명 조건 $\omega - k v_\parallel = \omega_{ce}$를 통해 휘슬러파와 공명할 수 있습니다. 이 전리층 플라즈마에서 $f = 5\,\text{kHz}$의 휘슬러파에 대해 공명 전자 에너지(keV 단위)를 구합니다.
2. 이 공명이 어떻게 피치각 산란(pitch angle scattering)과 복사대 전자의 대기권 손실로 이어지는지 논의합니다(파동-입자 상호작용(wave-particle interaction)의 실제 적용).

**제출물**: 세 개의 그림(분산 다이어그램, 군속도 곡선, 주파수 대 이동 시간)을 적절한 주석과 함께 생성하는 단일 Python 스크립트, 그리고 파동-입자 공명에 대한 짧은 서면 논의.

---

**이전**: [플라즈마 진단](./15_Plasma_Diagnostics.md) | **다음**: 없음(마지막 레슨)
