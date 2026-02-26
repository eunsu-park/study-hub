# 02. 기하광학 기초

[← 이전: 01. 빛의 본질](01_Nature_of_Light.md) | [다음: 03. 거울과 렌즈 →](03_Mirrors_and_Lenses.md)

---

## 학습 목표

1. 빛의 광선 모델(ray model)을 적용하고 그 유효 범위(특성 크기 $\gg \lambda$)를 이해한다
2. 페르마의 최소 시간 원리(Fermat's principle of least time)로부터 스넬 법칙과 반사 법칙을 유도한다
3. 스넬 법칙을 이용해 평면 및 곡면 경계면에서의 굴절 문제를 푼다
4. 전반사(total internal reflection)의 임계각을 계산하고 그 응용을 설명한다
5. 프리즘 분산을 분석하고 다중 면 광학 시스템에서 광선을 추적한다
6. 에이코날 방정식(eikonal equation)을 파동 광학과 기하 광학 사이의 다리로 기술한다
7. 기하 광학을 이용해 자연 현상(신기루, 무지개, 광섬유)을 설명한다

---

## 왜 중요한가

기하 광학(geometric optics)은 광학 설계의 핵심 도구이다. 얼굴의 안경에서 스마트폰 카메라, 우주 망원경의 거울까지, 거의 모든 실용적 광학 시스템은 광선 추적(ray tracing) — 각 면에서 반사와 굴절을 체계적으로 적용하는 것 — 을 이용하여 설계된다. 심지어 고급 계산 광학 및 렌즈 설계 소프트웨어(Zemax, Code V)도 파동 보정을 추가하기 전에 기하 광학에서 시작한다. 기하 광학을 마스터하면 앞으로 접하게 될 대부분의 광학 시스템을 이해하고 설계하는 도구를 갖게 된다.

> **비유**: 기하 광학은 파동 광학에 대해 뉴턴 역학이 양자역학에 대해 가지는 관계와 같다. 빛의 파장보다 물체가 훨씬 클 때 탁월하게 작동하는 근사이다 — 뉴턴의 법칙이 물체가 원자 크기보다 훨씬 클 때 탁월하게 작동하는 것처럼. 적용 가능한 모든 곳에서 더 단순한 이론을 사용하고, 반드시 필요할 때만 완전한 이론을 도입한다.

---

## 1. 빛의 광선 모델

### 1.1 광선이 유효한 경우

**기하 광학**(기하 광학(ray optics)이라고도 함)에서 빛은 광선으로 전파된다 — 균질 매질에서는 직선, 경계면에서는 꺾이고, 경사 굴절률 매질에서는 곡선을 그린다. 이 모델은 다음 조건에서 유효하다:

$$\text{특성 크기} \gg \lambda$$

가시광선($\lambda \sim 400$–$700$ nm)의 경우, 기하 광학은 약 10 $\mu$m보다 큰 물체에서 유효하다. 이 크기 이하에서는 회절이 중요해지고 파동 광학(레슨 05~06)이 필요하다.

### 1.2 부호 규약

기하 광학 계산에는 일관된 부호 규약이 필요하다. 여기서는 **실상(real)은 양수** 규약을 사용한다:

| 물리량 | 양수(+) | 음수(-) |
|--------|---------|---------|
| 물체 거리 $s$ | 물체가 입사 쪽에 있을 때 | 물체가 출사 쪽에 있을 때(허상) |
| 상 거리 $s'$ | 상이 출사 쪽에 있을 때(실상) | 상이 입사 쪽에 있을 때(허상) |
| 초점 거리 $f$ | 수렴형(오목 거울, 볼록 렌즈) | 발산형(볼록 거울, 오목 렌즈) |
| 곡률 반지름 $R$ | 곡률 중심이 출사 쪽에 있을 때 | 곡률 중심이 입사 쪽에 있을 때 |
| 높이 $y$ | 광축 위 | 광축 아래 |

빛은 규약상 왼쪽에서 오른쪽으로 이동한다.

---

## 2. 페르마 원리

### 2.1 진술

**페르마 원리(Fermat's Principle)** (최소 시간의 원리): 빛은 두 점 사이를 광학 경로 길이(OPL)가 *정류(stationary)*인 경로 — 즉 1차 변분이 사라지는 경로 — 를 따라 이동한다:

$$\delta \int_A^B n(\mathbf{r}) \, ds = 0$$

대부분의 경우 "정류"는 *최솟값*(최소 시간의 경로)을 의미하지만, 극댓값이나 안장점일 수도 있다(예: 오목 거울의 반사).

$A$에서 $B$까지의 **광학 경로 길이(Optical Path Length, OPL)**는:

$$\text{OPL} = \int_A^B n(\mathbf{r}) \, ds$$

여기서 $n(\mathbf{r})$은 공간적으로 변하는 굴절률이다. 균질 매질에서는 $\text{OPL} = n \cdot d$ ($d$는 기하학적 거리).

### 2.2 물리적 해석

진공에서 빛은 항상 최단 기하학적 경로(직선)를 취한다. 매질이 있는 경우, 빛은 꺾인 경로를 취할 수 있다 — 더 빠른 매질을 통과하는 꺾인 경로의 *시간*이 더 느린 매질을 일부 통과하는 직선 경로의 *시간*보다 짧을 수 있기 때문이다.

> **비유**: 해변에 있는 인명구조원이 바다에서 익사하는 수영객을 발견했다고 상상해보자. 인명구조원은 물에서 수영하는 것보다 모래에서 달리는 것이 더 빠르다. 수영객에게 가는 가장 빠른 경로는 *직선*이 아니다; 대신 각도가 딱 맞을 때까지 해변을 따라 달린 다음 물로 들어간다. 페르마 원리는 빛도 같은 방식으로 경로를 "선택"한다고 말한다 — 이동 시간을 최소화하여.

```python
import numpy as np
import matplotlib.pyplot as plt

# Fermat's principle: find the path that minimizes travel time
# A lifeguard analogy — running on beach (fast) then swimming (slow)

# Setup
beach_length = 20.0     # meters (horizontal extent)
lifeguard_pos = (0, 5)  # position on beach (x=0, y=5 from water line)
swimmer_pos = (15, -8)  # position in water (x=15, y=-8 from water line)

v_sand = 5.0   # running speed on sand (m/s)
v_water = 1.5  # swimming speed in water (m/s)

# Water line is at y = 0. The lifeguard enters water at point (x_entry, 0).
# We vary x_entry to find the minimum-time path.

x_entry = np.linspace(0, 20, 500)

# Distance on sand from lifeguard to entry point
d_sand = np.sqrt((x_entry - lifeguard_pos[0])**2 + lifeguard_pos[1]**2)
# Distance in water from entry point to swimmer
d_water = np.sqrt((swimmer_pos[0] - x_entry)**2 + swimmer_pos[1]**2)

# Total travel time = distance/speed for each segment
time_total = d_sand / v_sand + d_water / v_water

# Find the optimal entry point (minimum time)
idx_opt = np.argmin(time_total)
x_opt = x_entry[idx_opt]
t_opt = time_total[idx_opt]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: time vs entry point — shows clear minimum
ax1.plot(x_entry, time_total, 'b-', linewidth=2)
ax1.axvline(x_opt, color='r', linestyle='--', label=f'Optimal: x = {x_opt:.1f} m')
ax1.set_xlabel('Entry point x (m)', fontsize=12)
ax1.set_ylabel('Total time (s)', fontsize=12)
ax1.set_title("Fermat's Principle: Minimum Travel Time", fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Right panel: physical path — lifeguard → entry → swimmer
ax2.axhspan(-10, 0, alpha=0.2, color='cyan', label='Water')
ax2.axhspan(0, 8, alpha=0.15, color='sandybrown', label='Beach')
ax2.axhline(0, color='blue', linewidth=1)

# Draw three paths: straight line, optimal, and a suboptimal
for x_e, color, label, ls in [(swimmer_pos[0], 'gray', 'Straight line', '--'),
                                (x_opt, 'red', f'Optimal (t={t_opt:.2f}s)', '-'),
                                (3.0, 'orange', 'Suboptimal', ':')]:
    ax2.plot([lifeguard_pos[0], x_e], [lifeguard_pos[1], 0], color=color, linestyle=ls, linewidth=2)
    ax2.plot([x_e, swimmer_pos[0]], [0, swimmer_pos[1]], color=color, linestyle=ls, linewidth=2)

ax2.plot(*lifeguard_pos, 'ko', markersize=10, label='Lifeguard')
ax2.plot(*swimmer_pos, 'bx', markersize=12, markeredgewidth=3, label='Swimmer')
ax2.set_xlabel('x (m)', fontsize=12)
ax2.set_ylabel('y (m)', fontsize=12)
ax2.set_title('Path Geometry (Top View)', fontsize=13)
ax2.legend(fontsize=9, loc='upper left')
ax2.set_xlim(-2, 22)
ax2.set_ylim(-10, 8)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fermat_principle.png', dpi=150)
plt.show()

# Verify: at the optimal point, sin(theta_sand)/sin(theta_water) = v_sand/v_water
# This is the lifeguard version of Snell's law!
theta_sand = np.arctan(x_opt / lifeguard_pos[1])
theta_water = np.arctan((swimmer_pos[0] - x_opt) / abs(swimmer_pos[1]))
print(f"sin(θ_sand)/sin(θ_water) = {np.sin(theta_sand)/np.sin(theta_water):.3f}")
print(f"v_sand/v_water = {v_sand/v_water:.3f}")
```

---

## 3. 반사 법칙

### 3.1 진술

광선이 매끈한 표면에 닿으면, 반사 광선은 **입사면(plane of incidence)**(입사 광선과 표면 법선을 포함하는 면)에 놓이고:

$$\theta_i = \theta_r$$

입사각 $\theta_i$는 반사각 $\theta_r$과 같고, 둘 다 표면 법선에서 측정된다.

### 3.2 페르마 원리로부터의 유도

점 $A$에서 거울 표면까지, 다시 점 $B$까지 이동하는 광선을 생각해보자. 거울이 $x$축을 따라 있고, $A = (0, a)$, $B = (d, b)$라 하자. 광선은 거울의 $P = (x, 0)$에서 반사된다.

전체 경로 길이는:

$$L(x) = \sqrt{x^2 + a^2} + \sqrt{(d-x)^2 + b^2}$$

$dL/dx = 0$으로 놓으면:

$$\frac{x}{\sqrt{x^2 + a^2}} = \frac{d-x}{\sqrt{(d-x)^2 + b^2}}$$

좌변은 $\sin\theta_i$이고 우변은 $\sin\theta_r$이므로, $\theta_i = \theta_r$을 얻는다.

### 3.3 반사의 종류

- **정반사(Specular reflection)**: 매끈한 표면(거울, 잔잔한 물)에서. 선명한 상을 만든다.
- **난반사(Diffuse reflection)**: 거친 표면(종이, 벽)에서. 빛을 모든 방향으로 산란시킨다. 이것이 우리가 어떤 시점에서든 물체를 볼 수 있는 이유이다 — 주변 빛을 난반사로 산란시키기 때문이다.

정반사와 난반사의 경계는 파장에 대한 표면 거칠기에 달려 있다. 표면 불규칙성이 $\lambda$보다 훨씬 작으면 반사는 정반사이다.

---

## 4. 굴절의 스넬 법칙

### 4.1 진술

빛이 굴절률 $n_1$인 매질에서 굴절률 $n_2$인 매질로 통과할 때, 굴절된 광선은 다음을 만족한다:

$$n_1 \sin\theta_1 = n_2 \sin\theta_2$$

여기서 $\theta_1$은 입사각, $\theta_2$는 굴절각이며, 둘 다 법선에서 측정된다.

### 4.2 페르마 원리로부터의 유도

매질 1의 $A$에서 매질 2의 $B$로 가는 광선이 점 $P$에서 경계면을 통과하는 것을 생각해보자. 광학 경로 길이는:

$$\text{OPL}(x) = n_1 \sqrt{x^2 + a^2} + n_2 \sqrt{(d-x)^2 + b^2}$$

$d(\text{OPL})/dx = 0$으로 놓으면:

$$n_1 \frac{x}{\sqrt{x^2 + a^2}} = n_2 \frac{d-x}{\sqrt{(d-x)^2 + b^2}}$$

$$n_1 \sin\theta_1 = n_2 \sin\theta_2$$

### 4.3 스넬 법칙의 결과

**덜 밀한 매질에서 더 밀한 매질로** ($n_1 < n_2$): 광선이 법선 *쪽으로* 굴절된다 ($\theta_2 < \theta_1$).

**더 밀한 매질에서 덜 밀한 매질로** ($n_1 > n_2$): 광선이 법선 *반대로* 굴절된다 ($\theta_2 > \theta_1$).

**수직 입사(normal incidence)** ($\theta_1 = 0$): 굴절이 발생하지 않는다. $\theta_2 = 0$.

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize refraction at a flat interface using Snell's law
# Shows how the refracted angle depends on the angle of incidence

n1, n2 = 1.0, 1.5  # air to glass

theta1_deg = np.linspace(0, 89, 200)
theta1_rad = np.deg2rad(theta1_deg)

# Snell's law: n1 * sin(theta1) = n2 * sin(theta2)
sin_theta2 = (n1 / n2) * np.sin(theta1_rad)
theta2_rad = np.arcsin(sin_theta2)
theta2_deg = np.rad2deg(theta2_rad)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: angle relationship
ax1.plot(theta1_deg, theta2_deg, 'b-', linewidth=2, label=f'$n_1={n1}$ → $n_2={n2}$ (air→glass)')
ax1.plot(theta1_deg, theta1_deg, 'k--', alpha=0.3, label='No refraction ($n_1=n_2$)')
ax1.set_xlabel('Angle of incidence $\\theta_1$ (degrees)', fontsize=12)
ax1.set_ylabel('Angle of refraction $\\theta_2$ (degrees)', fontsize=12)
ax1.set_title("Snell's Law: Refraction Angles", fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Right: ray diagram showing incident and refracted rays
ax2.axhspan(-5, 0, alpha=0.15, color='lightblue', label=f'Medium 2 (n={n2})')
ax2.axhspan(0, 5, alpha=0.05, color='white', label=f'Medium 1 (n={n1})')
ax2.axhline(0, color='gray', linewidth=2)

# Draw normal (dashed vertical)
ax2.plot([0, 0], [-4, 4], 'k--', linewidth=1, alpha=0.5, label='Normal')

# Draw rays for several angles
for theta1_val in [15, 30, 45, 60, 75]:
    t1 = np.deg2rad(theta1_val)
    t2 = np.arcsin((n1 / n2) * np.sin(t1))

    # Incident ray: comes from upper-left, hits origin
    x_inc = -4 * np.sin(t1)
    y_inc = 4 * np.cos(t1)
    ax2.annotate('', xy=(0, 0), xytext=(x_inc, y_inc),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    # Refracted ray: leaves origin, goes into lower medium
    x_ref = 4 * np.sin(t2)
    y_ref = -4 * np.cos(t2)
    ax2.annotate('', xy=(x_ref, y_ref), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax2.text(x_inc - 0.3, y_inc + 0.2, f'{theta1_val}°', fontsize=8, color='blue')

ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Ray Diagram: Air → Glass', fontsize=13)
ax2.set_aspect('equal')
ax2.legend(fontsize=9, loc='lower left')

plt.tight_layout()
plt.savefig('snells_law.png', dpi=150)
plt.show()
```

---

## 5. 전반사

### 5.1 임계각

빛이 더 밀한 매질에서 덜 밀한 매질로 이동할 때 ($n_1 > n_2$), 모든 빛이 반사되는 **임계각(critical angle)** $\theta_c$가 존재한다:

$$\sin\theta_c = \frac{n_2}{n_1}$$

$\theta_1 > \theta_c$이면, 스넬 법칙은 $\sin\theta_2 > 1$을 주는데, 이는 실수 해가 없다 — 투과 광선이 존재하지 않는다.

| 경계면 | $n_1$ | $n_2$ | 임계각 |
|--------|-------|-------|--------|
| 유리 → 공기 | 1.50 | 1.00 | 41.8° |
| 물 → 공기 | 1.33 | 1.00 | 48.8° |
| 다이아몬드 → 공기 | 2.42 | 1.00 | 24.4° |
| 광섬유 (코어 → 클래딩) | 1.48 | 1.46 | 80.6° |

다이아몬드의 매우 낮은 임계각은 보석 내부에서 튀는 빛이 빠져나가기 전에 많은 전반사를 겪는다는 것을 의미하며, 특징적인 광채와 "파이어(fire)"를 만들어낸다.

### 5.2 전반사의 응용

**광섬유(Optical fibers)**: 빛이 코어-클래딩 경계면에서의 반복적인 전반사에 의해 가는 유리 섬유를 따라 안내된다. 이것이 현대 통신의 근간이다.

**프리즘**: 직각 프리즘은 흡수 손실이 없는 전반사를 이용하여 빛을 90° 또는 180° 방향으로 전환할 수 있으며, 이는 금속 거울보다 더 효율적이다.

**지문 센서**: 좌절된 전반사(Frustrated Total Internal Reflection, FTIR)는 피부 융선이 유리 표면에 닿는 곳을 감지하여 TIR 조건을 방해한다.

### 5.3 에바네센트파

전반사가 일어날 때도 전자기장은 경계면에서 급격히 사라지지 않는다. 대신, **에바네센트파(evanescent wave)**가 두 번째 매질로 지수적으로 감쇠하며 침투한다:

$$E(z) = E_0 \exp\left(-\frac{z}{\delta}\right)$$

여기서 침투 깊이는:

$$\delta = \frac{\lambda}{2\pi\sqrt{n_1^2 \sin^2\theta_1 - n_2^2}}$$

에바네센트파는 정상적인 TIR 조건에서 두 번째 매질로 순 에너지를 운반하지 않는다. 그러나 다른 밀한 매질이 가까이 (약 $\lambda$ 이내) 가져오면, 에바네센트파가 그 매질에 결합될 수 있다 — 이것이 **좌절된 전반사(FTIR)**로, 양자 터널링(quantum tunneling)과 유사하다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Evanescent wave: intensity decays exponentially beyond the interface
# during total internal reflection

n1 = 1.50   # glass
n2 = 1.00   # air
wavelength = 550e-9  # green light (550 nm)

# Calculate penetration depth for several angles beyond the critical angle
theta_c = np.arcsin(n2 / n1)
angles_deg = [42, 50, 60, 70, 80]  # all > critical angle (41.8°)

z = np.linspace(0, 2000, 500)  # distance into medium 2, in nm

fig, ax = plt.subplots(figsize=(10, 6))

for angle_deg in angles_deg:
    theta = np.deg2rad(angle_deg)

    # Penetration depth: how far the evanescent field extends
    denom = np.sqrt(n1**2 * np.sin(theta)**2 - n2**2)
    delta = wavelength / (2 * np.pi * denom)  # in meters
    delta_nm = delta * 1e9  # convert to nm for plotting

    # Intensity decays as exp(-2z/delta) since I ~ E^2
    intensity = np.exp(-2 * z / delta_nm)

    ax.plot(z, intensity, linewidth=2,
            label=f'$\\theta$ = {angle_deg}° ($\\delta$ = {delta_nm:.0f} nm)')

ax.set_xlabel('Distance into medium 2 (nm)', fontsize=12)
ax.set_ylabel('Relative Intensity $I/I_0$', fontsize=12)
ax.set_title(f'Evanescent Wave Decay (glass→air, $\\theta_c$ = {np.rad2deg(theta_c):.1f}°)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig('evanescent_wave.png', dpi=150)
plt.show()
```

---

## 6. 프리즘 광학

### 6.1 프리즘을 통한 굴절

꼭짓각(apex angle) $A$를 가진 프리즘은 편차각(deviation angle) $\delta$만큼 빛을 굴절시킨다. 최소 편차에서 광선에 대해:

$$n = \frac{\sin\left(\frac{A + \delta_{\min}}{2}\right)}{\sin\left(\frac{A}{2}\right)}$$

이것은 높은 정확도로 굴절률을 측정하는 표준 방법이다.

### 6.2 프리즘 분산

$n$이 파장에 따라 달라지기 때문에, 프리즘은 백색광을 스펙트럼 성분으로 분리한다:

$$\frac{d\delta}{d\lambda} = \frac{d\delta}{dn} \cdot \frac{dn}{d\lambda}$$

기하 인수 $d\delta/dn$는 프리즘 각도와 방향에 달려 있다. 물질 인수 $dn/d\lambda$는 유리 종류에 달려 있다(플린트 유리가 크라운 유리보다 분산이 높다).

### 6.3 무지개 형성

빗방울은 구형 프리즘 역할을 한다. 태양의 평행 광선이 다른 높이로 빗방울에 들어가서:

1. 앞면에서 **굴절(Refraction)**
2. 뒷면에서 **반사(Reflection)** (1차는 한 번, 2차는 두 번)
3. 출구에서 다시 앞면에서 **굴절(Refraction)**

편차각은 충돌 매개변수(impact parameter)에 따라 달라지고, **최소 편차**(데카르트 광선)가 약 138°에서 발생한다 — 즉 빛의 최대 집중이 반태양점으로부터 $180° - 138° = 42°$에서 나타난다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Rainbow optics: trace rays through a spherical water droplet
# and find the Descartes minimum deviation angle

def rainbow_deviation(b, n, k=1):
    """
    Calculate the deviation angle for a ray hitting a spherical droplet.

    b: impact parameter (0 to 1, normalized to droplet radius)
    n: refractive index of water
    k: number of internal reflections (1 for primary, 2 for secondary)

    Returns: deviation angle in degrees
    """
    # Angle of incidence from the impact parameter
    theta_i = np.arcsin(b)
    # Angle of refraction (Snell's law)
    theta_r = np.arcsin(b / n)
    # Total deviation: accounts for entry refraction, k internal reflections, exit refraction
    deviation = 2 * theta_i - 2 * (k + 1) * theta_r + k * np.pi
    return np.rad2deg(deviation)

b = np.linspace(0.01, 0.99, 1000)  # impact parameter

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate for different wavelengths (different n values for water)
wavelengths = {
    'Red (700nm)': 1.3312,
    'Yellow (580nm)': 1.3335,
    'Blue (450nm)': 1.3400,
}

for label, n_water in wavelengths.items():
    dev = rainbow_deviation(b, n_water, k=1)
    ax.plot(b, dev, linewidth=2, label=label)

    # Find the minimum deviation (Descartes ray)
    idx_min = np.argmin(np.abs(np.gradient(dev)))
    ax.plot(b[idx_min], dev[idx_min], 'o', markersize=8)
    ax.annotate(f'  min = {dev[idx_min]:.1f}°',
                xy=(b[idx_min], dev[idx_min]), fontsize=9)

ax.set_xlabel('Impact parameter b (normalized)', fontsize=12)
ax.set_ylabel('Deviation angle (degrees)', fontsize=12)
ax.set_title('Primary Rainbow: Deviation Angle vs Impact Parameter', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(135, 180)
plt.tight_layout()
plt.savefig('rainbow_deviation.png', dpi=150)
plt.show()
```

---

## 7. 에이코날 방정식

### 7.1 파동에서 광선으로

에이코날 방정식(eikonal equation)은 파동 광학과 기하 광학 사이를 연결하는 엄밀한 수학적 다리를 제공한다. 파동 방정식에서 출발하여 천천히 변하는 외포선(envelope)을 가정하면:

$$\mathbf{E}(\mathbf{r}) = \mathbf{E}_0(\mathbf{r}) \exp\left(i k_0 S(\mathbf{r})\right)$$

여기서 $S(\mathbf{r})$은 **에이코날(eikonal)**(광학 경로 함수)이고 $k_0 = 2\pi/\lambda_0$은 자유 공간 파수이다. $\lambda \to 0$ 극한(기하 광학 극한)에서 파동 방정식은 **에이코날 방정식**으로 환원된다:

$$|\nabla S|^2 = n^2(\mathbf{r})$$

### 7.2 물리적 의미

$S$가 일정한 면이 **파면(wavefronts)**이고, 광선은 이 파면에 수직인 곡선이다:

$$\frac{d\mathbf{r}}{ds} = \frac{\nabla S}{n}$$

이 방정식은 광선이 굴절률이 높은 영역 쪽으로 굽는다고 알려준다. 균질 매질($n$ = 상수)에서 $\nabla S$는 일정하고 광선은 직선이다.

### 7.3 경사 굴절률 매질에서의 광선 방정식

$n(\mathbf{r})$이 공간적으로 변하는 매질에서 광선 경로는:

$$\frac{d}{ds}\left(n \frac{d\mathbf{r}}{ds}\right) = \nabla n$$

이것이 **광선 방정식(ray equation)** — 초기 위치와 방향이 주어지면 완전한 광선 경로를 결정하는 2차 ODE이다.

**응용: 신기루(Mirages)**

뜨거운 사막에서 지표 근처의 공기는 더 뜨겁고 밀도가 낮아, $n$이 지표 쪽으로 감소한다. 광선 방정식은 광선이 표면에서 *위쪽으로* 굽는다고 예측한다. 관찰자는 물웅덩이처럼 보이는 곳에 반사된 하늘을 본다 — 신기루이다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Ray tracing in a graded-index medium (mirage simulation)
# Temperature gradient near hot ground creates a vertical n gradient

def n_profile(y):
    """
    Refractive index profile for air above a hot surface.
    n decreases toward the ground due to temperature-induced density decrease.
    This is a simplified exponential model.
    """
    n_ground = 1.000250   # slightly lower n near the hot ground
    n_inf = 1.000293      # standard air n at altitude
    scale_height = 2.0     # meters — height over which the gradient relaxes
    return n_inf - (n_inf - n_ground) * np.exp(-y / scale_height)

def dn_dy(y):
    """Derivative of n with respect to y (vertical gradient)."""
    n_ground = 1.000250
    n_inf = 1.000293
    scale_height = 2.0
    return (n_inf - n_ground) / scale_height * np.exp(-y / scale_height)

def ray_equations(s, state):
    """
    ODE system for the ray equation in 2D: d/ds(n dr/ds) = grad(n)
    state = [x, y, dx/ds, dy/ds]
    Using the parameterization where ds is arc length.
    """
    x, y, dx_ds, dy_ds = state
    n = n_profile(y)
    dndx = 0        # n doesn't depend on x (horizontal homogeneity)
    dndy = dn_dy(y)

    # d/ds(n * dx/ds) = dn/dx => n * d2x/ds2 + dn/ds * dx/ds = dn/dx
    # But dn/ds = (dn/dy) * (dy/ds), and similarly we need the full equations
    # Simplified: d2x/ds2 = (1/n)(dndx - dndy*dy_ds*dx_ds/n)
    # d2y/ds2 = (1/n)(dndy - (dndy*dy_ds^2 + dndx*dx_ds*dy_ds)/n)

    # Cleaner approach: let p = n*dx/ds, q = n*dy/ds
    # dp/ds = dndx, dq/ds = dndy
    # But we track (x, y, vx, vy) where vx=dx/ds, vy=dy/ds
    d2x_ds2 = (dndx - (dndx * dx_ds + dndy * dy_ds) * dx_ds) / n
    d2y_ds2 = (dndy - (dndx * dx_ds + dndy * dy_ds) * dy_ds) / n

    return [dx_ds, dy_ds, d2x_ds2, d2y_ds2]

# Launch rays from x=0 at various downward angles
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Top panel: ray paths (mirage effect)
for angle_deg in [-0.02, -0.04, -0.06, -0.08, -0.10]:
    angle_rad = np.deg2rad(angle_deg)
    y0 = 1.7  # observer eye height (meters)

    # Initial conditions: position (0, y0), direction (cos(a), sin(a))
    initial = [0, y0, np.cos(angle_rad), np.sin(angle_rad)]

    # Integrate the ray equation along the arc length parameter
    sol = solve_ivp(ray_equations, [0, 5000], initial,
                    max_step=1.0, events=None, dense_output=True)

    # Only plot rays that stay above ground (y > 0)
    mask = sol.y[1] > 0
    ax1.plot(sol.y[0][mask], sol.y[1][mask], linewidth=1.5,
             label=f'{angle_deg}°')

ax1.set_xlabel('Horizontal distance (m)', fontsize=12)
ax1.set_ylabel('Height (m)', fontsize=12)
ax1.set_title('Ray Paths in a Mirage (Hot Ground)', fontsize=13)
ax1.legend(fontsize=9, title='Launch angle')
ax1.set_ylim(0, 2)
ax1.grid(True, alpha=0.3)

# Bottom panel: refractive index profile
y_range = np.linspace(0, 5, 200)
ax2.plot(n_profile(y_range), y_range, 'b-', linewidth=2)
ax2.set_xlabel('Refractive index n', fontsize=12)
ax2.set_ylabel('Height (m)', fontsize=12)
ax2.set_title('Refractive Index Profile (n decreases near hot ground)', fontsize=13)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mirage_ray_tracing.png', dpi=150)
plt.show()
```

---

## 8. 곡면에서의 굴절

### 8.1 단일 굴절면

굴절률 $n_1$과 $n_2$인 매질을 분리하는 반지름 $R$인 구면 경계에 대해:

$$\frac{n_1}{s} + \frac{n_2}{s'} = \frac{n_2 - n_1}{R}$$

여기서 $s$는 물체 거리, $s'$는 상 거리이다. 이것이 박형 렌즈 방정식(레슨 03)의 기초이다.

### 8.2 겉보기 깊이

수영장을 들여다볼 때, 물체는 실제보다 표면에 더 가까이 있는 것처럼 보인다. 법선에 가까운 시야에서:

$$\text{겉보기 깊이} = \frac{\text{실제 깊이}}{n}$$

물($n = 1.33$) 속 2 m 깊이의 동전은 약 1.5 m 깊이에 있는 것처럼 보인다.

---

## 연습 문제

### 연습 1: 페르마 원리 유도

빛이 공기($n_1 = 1$)의 점 $A = (0, 3)$에서 유리($n_2 = 1.5$)의 점 $B = (6, -4)$로 이동하며, $y = 0$인 평면 경계면을 통과한다.

(a) 통과 점 $x$의 함수로 광학 경로 길이를 써라.

(b) 미분하여 0으로 놓아 최적 $x$를 구하라.

(c) 결과 각도가 스넬 법칙을 만족하는지 확인하라.

### 연습 2: 전반사

광섬유의 코어 굴절률은 1.48이고 클래딩 굴절률은 1.46이다.

(a) 코어-클래딩 경계면에서 임계각을 계산하라.

(b) 공기에서 광섬유로 들어가는 빛의 최대 수용각(개구수, numerical aperture)은 얼마인가?

(c) 광섬유가 구부러지면 임계각 조건은 어떻게 변하는가? 어느 굴곡 반지름에서 빛이 새기 시작하는가?

### 연습 3: 프리즘 분석

$n = 1.62$ (589 nm)인 플린트 유리로 만들어진 60° 정삼각형 프리즘이 있다.

(a) 최소 편차각 $\delta_{\min}$을 계산하라.

(b) 최소 편차에서 입사각은 얼마인가?

(c) 450 nm에서 $n = 1.64$이면, 최소 편차에서 589 nm와 450 nm 빛 사이의 각 분산은 얼마인가?

### 연습 4: 에바네센트파

633 nm 빛이 유리($n_1 = 1.52$)에서 유리-공기 경계면에 45°로 닿는다.

(a) 전반사가 일어나는지 확인하라.

(b) 에바네센트파 침투 깊이 $\delta$를 계산하라.

(c) 두 번째 유리 표면이 첫 번째로부터 $\delta/2$ 거리에 놓이면, 강도의 약 몇 분율이 터널링하는가?

### 연습 5: 신기루 추정

뜨거운 도로에서 공기 온도 경구배(temperature gradient)는 $n(y) = n_0(1 + \alpha y)$ ($y$는 도로 위 높이, $n_0 = 1.000250$, $\alpha = 1.7 \times 10^{-5}$ m$^{-1}$)로 근사할 수 있는 굴절률 프로파일을 만든다.

(a) $y = 1.5$ m에서 도로 표면($y = 0$)을 아슬아슬하게 스치도록 수평으로 광선을 발사하려면 어느 각도로 해야 하는가?

(b) 신기루는 얼마나 멀리(수평으로) 나타나는가?

---

## 요약

| 개념 | 핵심 공식 / 사실 |
|------|----------------|
| 광선 모델 유효 범위 | 특성 크기 $\gg \lambda$ |
| 페르마 원리 | $\delta \int n \, ds = 0$ — 빛은 정류 광학 경로 길이의 경로를 취한다 |
| 광학 경로 길이 | $\text{OPL} = \int n \, ds = n \cdot d$ (균질 매질) |
| 반사 법칙 | $\theta_i = \theta_r$ |
| 스넬 법칙 | $n_1 \sin\theta_1 = n_2 \sin\theta_2$ |
| 임계각 | $\sin\theta_c = n_2/n_1$ ($n_1 > n_2$ 필요) |
| 에바네센트파 깊이 | $\delta = \lambda / (2\pi\sqrt{n_1^2\sin^2\theta - n_2^2})$ |
| 프리즘 최소 편차 | $n = \sin\frac{A+\delta_{\min}}{2} / \sin\frac{A}{2}$ |
| 에이코날 방정식 | $|\nabla S|^2 = n^2(\mathbf{r})$ — 파동에서 기하 광학으로의 다리 |
| 광선 방정식 | $\frac{d}{ds}(n\frac{d\mathbf{r}}{ds}) = \nabla n$ — 광선은 더 높은 $n$ 쪽으로 굽는다 |
| 곡면 굴절 | $n_1/s + n_2/s' = (n_2 - n_1)/R$ |
| 겉보기 깊이 | 겉보기 깊이 $=$ 실제 깊이 $/n$ |

---

[← 이전: 01. 빛의 본질](01_Nature_of_Light.md) | [다음: 03. 거울과 렌즈 →](03_Mirrors_and_Lenses.md)
