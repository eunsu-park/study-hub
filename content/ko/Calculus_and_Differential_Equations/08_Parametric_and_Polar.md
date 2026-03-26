# 08. 매개변수 곡선과 극좌표

## 학습 목표

- 매개변수 방정식(parametric equations)을 사용하여 곡선을 표현하고 속도, 가속도, 기울기를 계산할 수 있다
- 매개변수로 정의된 곡선의 호의 길이(arc length)를 계산할 수 있다
- 직교 좌표계와 극좌표계(polar coordinate system) 사이를 변환하고 표준 극곡선을 식별할 수 있다
- 극좌표에서 면적과 호의 길이를 계산할 수 있다
- Python (Matplotlib)을 사용하여 매개변수 곡선과 극곡선을 시각화할 수 있다

---

## 1. 매개변수 방정식

때때로 평면의 곡선은 $y = f(x)$로 설명할 수 없다. **매개변수 표현(parametric representation)** 은 세 번째 변수 -- **매개변수(parameter)** $t$ -- 를 사용하여 두 좌표를 동시에 설명한다:

$$x = f(t), \quad y = g(t), \quad a \le t \le b$$

$t$를 시간이라 생각하자: 시계가 $a$부터 $b$까지 돌아가면서 점 $(x(t), y(t))$가 경로를 그린다. 이것은 정확히 애니메이션이 작동하는 방식이다 -- 각 프레임은 점이 어디에 나타나는지를 결정하는 시간 표시를 가진다.

**예시: 원**

단위원 $x^2 + y^2 = 1$은 $x$의 함수가 아니지만 (수직선 검정에 실패), 매개변수로는:

$$x = \cos t, \quad y = \sin t, \quad 0 \le t \le 2\pi$$

### 1.1 속도와 가속도

$t$가 시간을 나타내면, **속도 벡터(velocity vector)** 는:

$$\mathbf{v}(t) = \left(\frac{dx}{dt},\, \frac{dy}{dt}\right)$$

**속력(speed)** (속도의 크기)은:

$$\|\mathbf{v}(t)\| = \sqrt{\left(\frac{dx}{dt}\right)^2 + \left(\frac{dy}{dt}\right)^2}$$

**가속도 벡터(acceleration vector)** 는:

$$\mathbf{a}(t) = \left(\frac{d^2x}{dt^2},\, \frac{d^2y}{dt^2}\right)$$

**예시: 포사체 운동(Projectile Motion)**

각도 $\theta$, 초기 속력 $v_0$으로 발사된 공 (공기 저항 무시):

$$x(t) = v_0 \cos\theta \cdot t, \quad y(t) = v_0 \sin\theta \cdot t - \tfrac{1}{2}g t^2$$

- $v_0$: 초기 속력 (m/s)
- $\theta$: 발사 각도 (라디안)
- $g$: 중력 가속도 ($\approx 9.8\,\text{m/s}^2$)

속도 성분은 $\dot{x} = v_0 \cos\theta$ (일정)와 $\dot{y} = v_0 \sin\theta - gt$ (선형 감소)이다.

### 1.2 매개변수 곡선의 기울기

매개변수를 소거하지 않고 기울기 $dy/dx$를 구하려면, **연쇄 법칙(chain rule)** 을 사용한다:

$$\frac{dy}{dx} = \frac{dy/dt}{dx/dt} \quad \text{(단, } dx/dt \neq 0\text{)}$$

이것은 강력한 기법이다: $y$를 $x$로 풀지 않고, $t$에 대한 도함수를 나눈다.

**2계 도함수** (곡률)의 경우:

$$\frac{d^2y}{dx^2} = \frac{\frac{d}{dt}\!\left(\frac{dy}{dx}\right)}{dx/dt}$$

주의: 이것은 $\frac{d^2y/dt^2}{d^2x/dt^2}$가 아니다 -- 흔한 실수이다.

**예시:** $x = t^2$, $y = t^3$인 경우:

$$\frac{dy}{dx} = \frac{3t^2}{2t} = \frac{3t}{2}$$

$$\frac{d^2y}{dx^2} = \frac{\frac{d}{dt}(3t/2)}{2t} = \frac{3/2}{2t} = \frac{3}{4t}$$

### 1.3 매개변수 곡선의 호의 길이

$t = a$부터 $t = b$까지 매개변수 곡선의 길이는 무한소 거리 요소 $ds$를 합산하여 구한다:

$$L = \int_a^b \sqrt{\left(\frac{dx}{dt}\right)^2 + \left(\frac{dy}{dt}\right)^2}\, dt$$

**직관:** 각 순간에 점은 수평으로 $dx$, 수직으로 $dy$만큼 이동한다. 피타고라스 정리에 의해 이동한 작은 거리는 $\sqrt{dx^2 + dy^2}$이다. $dt$로 나누고 적분하면 총 거리를 얻는다.

**예시: 원의 둘레**

$x = R\cos t$, $y = R\sin t$, $0 \le t \le 2\pi$인 경우:

$$L = \int_0^{2\pi} \sqrt{R^2\sin^2 t + R^2\cos^2 t}\, dt = \int_0^{2\pi} R\, dt = 2\pi R$$

잘 알려진 공식과 정확히 일치한다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# --- Parametric curve examples ---

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Projectile motion
v0, theta, g = 20.0, np.radians(45), 9.8
t_flight = 2 * v0 * np.sin(theta) / g  # total flight time
t = np.linspace(0, t_flight, 200)
x_proj = v0 * np.cos(theta) * t
y_proj = v0 * np.sin(theta) * t - 0.5 * g * t**2

axes[0].plot(x_proj, y_proj, 'b-', linewidth=2)
axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('y (m)')
axes[0].set_title('Projectile Motion')
axes[0].set_aspect('equal')
axes[0].grid(True)

# 2. Cycloid: the curve traced by a point on a rolling wheel
t = np.linspace(0, 4 * np.pi, 500)
R = 1.0
x_cyc = R * (t - np.sin(t))
y_cyc = R * (1 - np.cos(t))

axes[1].plot(x_cyc, y_cyc, 'r-', linewidth=2)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Cycloid (Rolling Wheel)')
axes[1].set_aspect('equal')
axes[1].grid(True)

# 3. Lissajous figure: superposition of two perpendicular oscillations
t = np.linspace(0, 2 * np.pi, 1000)
x_lis = np.sin(3 * t)
y_lis = np.sin(4 * t + np.pi / 4)

axes[2].plot(x_lis, y_lis, 'g-', linewidth=1)
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
axes[2].set_title('Lissajous Figure (3:4)')
axes[2].set_aspect('equal')
axes[2].grid(True)

plt.tight_layout()
plt.show()

# --- Arc length computation ---
# Arc length of one arch of cycloid: x = t - sin(t), y = 1 - cos(t), 0 <= t <= 2pi
def cycloid_speed(t):
    """Speed along the cycloid: sqrt((dx/dt)^2 + (dy/dt)^2)."""
    dxdt = 1 - np.cos(t)
    dydt = np.sin(t)
    return np.sqrt(dxdt**2 + dydt**2)

arc_length, _ = quad(cycloid_speed, 0, 2 * np.pi)
print(f"Arc length of one cycloid arch: {arc_length:.6f}")
print(f"Theoretical value (8R = 8):     {8 * R:.6f}")
```

---

## 2. 극좌표

### 2.1 극좌표계

점의 위치를 수평, 수직 거리 $(x, y)$로 나타내는 대신, **극좌표(polar coordinates)** 는 다음을 사용한다:

- $r$: 원점(**극(pole)**)으로부터의 거리
- $\theta$: 양의 $x$축으로부터 반시계 방향으로 측정된 각도

**변환 공식:**

$$x = r\cos\theta, \quad y = r\sin\theta$$

$$r = \sqrt{x^2 + y^2}, \quad \theta = \arctan\!\left(\frac{y}{x}\right)$$

$\arctan$ 공식은 사분면에 주의가 필요하다; Python에서는 `np.arctan2(y, x)`를 사용한다.

**비유:** 극좌표를 레이더로 생각하라: "얼마나 먼지" ($r$)와 "어느 방향인지" ($\theta$)를 지정하는 것이지, "얼마나 동쪽인지"와 "얼마나 북쪽인지"가 아니다.

### 2.2 일반적인 극곡선

| 곡선 | 방정식 | 모양 |
|-------|----------|-------|
| 원 | $r = a$ | 원점 중심 반지름 $a$인 원 |
| 심장형(Cardioid) | $r = a(1 + \cos\theta)$ | 하트 모양 곡선 |
| 장미($n$잎) | $r = a\cos(n\theta)$ | $n$이 홀수이면 $n$잎, 짝수이면 $2n$잎 |
| 리마송(Limacon) | $r = a + b\cos\theta$ | $a/b$ 비율에 따라 변화 |
| 아르키메데스 나선 | $r = a\theta$ | 균일 간격 나선 |
| 로그 나선 | $r = ae^{b\theta}$ | 자기 유사 나선 (앵무조개 껍질) |

### 2.3 극좌표에서의 면적

$\theta = \alpha$부터 $\theta = \beta$까지 극곡선 $r = f(\theta)$로 둘러싸인 면적은:

$$A = \frac{1}{2}\int_\alpha^\beta r^2\, d\theta = \frac{1}{2}\int_\alpha^\beta [f(\theta)]^2\, d\theta$$

**유도:** 각도 $\theta$에서 각도 너비 $d\theta$인 얇은 "파이 조각"은 밑변 $r\,d\theta$와 높이 $r$인 삼각형으로 근사되어 면적 $\tfrac{1}{2}r^2\,d\theta$를 준다.

**예시:** 심장형 $r = 1 + \cos\theta$로 둘러싸인 면적:

$$A = \frac{1}{2}\int_0^{2\pi}(1 + \cos\theta)^2\, d\theta = \frac{1}{2}\int_0^{2\pi}(1 + 2\cos\theta + \cos^2\theta)\, d\theta = \frac{3\pi}{2}$$

### 2.4 극좌표에서의 호의 길이

극곡선 $r = f(\theta)$의 $\theta = \alpha$부터 $\theta = \beta$까지의 호의 길이는:

$$L = \int_\alpha^\beta \sqrt{r^2 + \left(\frac{dr}{d\theta}\right)^2}\, d\theta$$

**유도:** 매개변수 호의 길이 공식에 $x = r\cos\theta$, $y = r\sin\theta$과 매개변수 $\theta$를 넣고, $\cos^2\theta + \sin^2\theta = 1$을 사용하여 전개 및 단순화하면 이 공식이 나온다.

**예시:** 아르키메데스 나선 $r = \theta$의 $\theta = 0$부터 $\theta = 2\pi$까지 호의 길이:

$$L = \int_0^{2\pi} \sqrt{\theta^2 + 1}\, d\theta$$

이 적분은 치환 $\theta = \sinh u$를 사용하거나 수치적으로 계산할 수 있다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

fig, axes = plt.subplots(2, 3, figsize=(15, 10),
                          subplot_kw={'projection': 'polar'})

# 1. Cardioid
theta = np.linspace(0, 2 * np.pi, 500)
r_card = 1 + np.cos(theta)
axes[0, 0].plot(theta, r_card, 'b-', linewidth=2)
axes[0, 0].set_title('Cardioid\n$r = 1 + \\cos\\theta$', pad=15)

# 2. Rose curve (3 petals)
r_rose3 = np.cos(3 * theta)
axes[0, 1].plot(theta, r_rose3, 'r-', linewidth=2)
axes[0, 1].set_title('Rose (3 petals)\n$r = \\cos(3\\theta)$', pad=15)

# 3. Rose curve (4 petals, n=2 even gives 2n=4)
r_rose4 = np.cos(2 * theta)
axes[0, 2].plot(theta, r_rose4, 'g-', linewidth=2)
axes[0, 2].set_title('Rose (4 petals)\n$r = \\cos(2\\theta)$', pad=15)

# 4. Limaçon with inner loop (a/b < 1)
r_lima = 0.5 + np.cos(theta)
axes[1, 0].plot(theta, r_lima, 'm-', linewidth=2)
axes[1, 0].set_title('Limaçon\n$r = 0.5 + \\cos\\theta$', pad=15)

# 5. Archimedean spiral
theta_sp = np.linspace(0, 4 * np.pi, 500)
r_arch = 0.3 * theta_sp
axes[1, 1].plot(theta_sp, r_arch, 'c-', linewidth=2)
axes[1, 1].set_title('Archimedean Spiral\n$r = 0.3\\theta$', pad=15)

# 6. Logarithmic spiral
r_log = np.exp(0.1 * theta_sp)
axes[1, 2].plot(theta_sp, r_log, color='orange', linewidth=2)
axes[1, 2].set_title('Log Spiral\n$r = e^{0.1\\theta}$', pad=15)

plt.tight_layout()
plt.show()

# --- Area and arc length computations ---

# Area of cardioid r = 1 + cos(theta)
area_card, _ = quad(lambda th: 0.5 * (1 + np.cos(th))**2, 0, 2 * np.pi)
print(f"Cardioid area:       {area_card:.6f}")
print(f"Theoretical (3π/2):  {1.5 * np.pi:.6f}")

# Arc length of Archimedean spiral r = theta, 0 to 2pi
def spiral_ds(th):
    """Integrand for polar arc length: sqrt(r^2 + (dr/dtheta)^2)."""
    r = th
    drdt = 1.0
    return np.sqrt(r**2 + drdt**2)

arc_spiral, _ = quad(spiral_ds, 0, 2 * np.pi)
print(f"Archimedean spiral arc length (0 to 2π): {arc_spiral:.6f}")
```

---

## 3. 극곡선의 기울기와 접선

극곡선 $r = f(\theta)$는 $\theta$를 매개변수로 하는 매개변수 곡선으로 볼 수 있다:

$$x = r\cos\theta = f(\theta)\cos\theta, \quad y = r\sin\theta = f(\theta)\sin\theta$$

매개변수 기울기 공식을 적용하면:

$$\frac{dy}{dx} = \frac{dy/d\theta}{dx/d\theta} = \frac{f'(\theta)\sin\theta + f(\theta)\cos\theta}{f'(\theta)\cos\theta - f(\theta)\sin\theta}$$

여기서 $f'(\theta) = dr/d\theta$.

**수평 접선(horizontal tangent)** 은 $dy/d\theta = 0$ (그리고 $dx/d\theta \neq 0$)일 때 발생한다.

**수직 접선(vertical tangent)** 은 $dx/d\theta = 0$ (그리고 $dy/d\theta \neq 0$)일 때 발생한다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Find tangent lines on the cardioid r = 1 + cos(theta)
theta_vals = np.linspace(0, 2 * np.pi, 1000)
r_vals = 1 + np.cos(theta_vals)

# Convert to Cartesian for plotting
x = r_vals * np.cos(theta_vals)
y = r_vals * np.sin(theta_vals)

# Compute dy/dtheta and dx/dtheta
dr = -np.sin(theta_vals)
dxdt = dr * np.cos(theta_vals) - r_vals * np.sin(theta_vals)
dydt = dr * np.sin(theta_vals) + r_vals * np.cos(theta_vals)

# Find approximate horizontal tangent points (dy/dtheta ≈ 0)
horiz_mask = np.abs(dydt) < 0.01
# Filter out points where dx/dtheta is also near zero (cusps)
horiz_mask &= np.abs(dxdt) > 0.1

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, y, 'b-', linewidth=2, label='Cardioid')

# Mark special points
ax.plot(x[horiz_mask], y[horiz_mask], 'ro', markersize=6,
        label='Horizontal tangent')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Tangent Lines on Cardioid')
ax.set_aspect('equal')
ax.legend()
ax.grid(True)
plt.show()
```

---

## 4. 응용과 연결

**매개변수 곡선이 나타나는 곳:**
- 로봇 공학: 관절 각도가 말단 효과기(end-effector) 경로를 매개변수화
- 컴퓨터 그래픽스: 베지에(Bezier) 곡선과 스플라인은 매개변수적
- 물리학: 궤도 역학은 시간을 매개변수로 사용
- 공학: CNC 가공 경로는 매개변수적으로 지정

**극좌표는 원형 대칭이 있는 문제를 단순화한다:**
- 안테나 방사 패턴
- 행성 궤도 (케플러의 법칙은 극좌표 형태를 사용)
- 자연의 나선 구조 (앵무조개 껍질, 은하)

---

## 연습 문제

**1.** 입자가 $0 \le t \le 2\pi$에 대해 곡선 $x = e^t\cos t$, $y = e^t\sin t$를 따라 이동한다.
   - (a) $t = 0$에서의 속도와 속력을 구하라.
   - (b) 속력이 $\sqrt{2}\,e^t$임을 보여라.
   - (c) 총 호의 길이를 구하라.

**2.** 매개변수 곡선 $x = 2\cos t + \cos 2t$, $y = 2\sin t + \sin 2t$ (**에피사이클로이드(epicycloid)**)에 대해:
   - (a) $t = \pi/4$에서 $dy/dx$를 구하라.
   - (b) $[0, 2\pi)$에서 접선이 수평인 $t$의 모든 값을 구하라.
   - (c) 총 호의 길이를 수치적으로 계산하라.

**3.** 장미 곡선 $r = \sin(3\theta)$의 한 잎으로 둘러싸인 면적을 구하라. (힌트: 한 잎은 $0 \le \theta \le \pi/3$ 범위이다.)

**4.** 로그 나선 $r = e^{0.2\theta}$의 $\theta = 0$부터 $\theta = 4\pi$까지 호의 길이를 계산하라. `scipy.integrate.quad`로 수치적으로 검증하라.

**5.** 위성이 극좌표 궤도 $r = \frac{p}{1 + e\cos\theta}$ (극좌표의 원뿔 곡선)을 따르며, 반통경(semi-latus rectum) $p = 7000\,\text{km}$, 이심률(eccentricity) $e = 0.1$이다.
   - (a) 근지점(periapsis, 최근접)과 원지점(apoapsis, 최원거리)을 구하라.
   - (b) 극좌표로 궤도를 그려라.
   - (c) $\theta = 0$부터 $\theta = \pi$까지 반지름 벡터가 쓸어낸 면적을 구하라 (케플러의 제2법칙).

---

## 참고 자료

- **James Stewart**, *Calculus: Early Transcendentals*, 9th Edition, Chapters 10.1-10.4
- **Gilbert Strang**, *Calculus*, Chapter 9 (Polar Coordinates and Complex Numbers)
- **3Blue1Brown**, "Parametric Curves" (시각적 직관)
- **Matplotlib 극좌표 플롯 문서**: https://matplotlib.org/stable/gallery/pie_and_polar_charts/index.html

---

[이전: 수열과 급수](./07_Sequences_and_Series.md) | [다음: 다변수 함수](./09_Multivariable_Functions.md)
