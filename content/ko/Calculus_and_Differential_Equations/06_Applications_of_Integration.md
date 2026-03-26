# 적분의 응용

## 학습 목표

- 어떤 함수가 위에 있는지 식별하고 차이를 적분하여 두 곡선 사이의 면적을 **계산**할 수 있다
- 디스크(disk), 와셔(washer), 셸(shell) 방법을 사용하여 회전체(solid of revolution)의 부피를 **계산**할 수 있다
- 직교 좌표와 매개변수 형태로 주어진 곡선에 대한 호의 길이(arc length) 공식을 **유도**하고 계산할 수 있다
- 회전체의 겉넓이(surface area)를 **계산**하고 호의 길이와의 관계를 설명할 수 있다
- 일(work), 정수압(hydrostatic force), 무게 중심(center of mass)을 포함하는 물리적 문제에 적분을 **적용**할 수 있다

## 소개

적분은 면적을 계산해야 하는 필요에서 탄생했지만, 그 응용은 훨씬 더 넓다. 무한히 많은 무한소 기여를 합산해야 할 때마다 -- 고체를 얇은 디스크로 자르거나, 곡선을 작은 선분으로 펼치거나, 표면에 걸쳐 힘을 합산하거나 -- 적분이 그 도구이다.

이 레슨은 정적분의 기하학적 및 물리적 응용을 다룬다. 이 문제들은 모두 같은 전략을 공유한다: 물체를 얇은 조각으로 자르고, 각 조각에 대한 식을 쓰고, 적분하여 모두 합산한다.

## 곡선 사이의 면적

### 두 곡선: 위에서 아래를 빼기

$[a, b]$에서 $f(x) \geq g(x)$이면, 곡선 사이의 면적은:

$$A = \int_a^b [f(x) - g(x)] \, dx$$

**왜 위에서 아래를 빼는가?** 각 얇은 수직 띠의 높이는 $f(x) - g(x)$이고 너비는 $dx$이다. 이 모든 띠를 $a$부터 $b$까지 합산(적분)한다.

**중요:** 곡선이 $[a, b]$ 내에서 교차하면, 각 교차점에서 적분을 분할하고 어떤 함수가 "위에" 있는지 바꿔야 한다.

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x_sym = sp.Symbol('x')

# Find area between y = x^2 and y = x + 2
f = x_sym + 2
g = x_sym**2

# Step 1: Find intersection points
intersections = sp.solve(f - g, x_sym)
a, b = float(intersections[0]), float(intersections[1])
print(f"Intersection points: x = {intersections}")

# Step 2: Determine which is on top (f > g between intersections)
# At x = 0: f(0) = 2, g(0) = 0, so f is on top

# Step 3: Integrate
area = sp.integrate(f - g, (x_sym, intersections[0], intersections[1]))
print(f"Area = integral from {a} to {b} of [(x+2) - x^2] dx = {area} = {float(area):.4f}")

# Visualization
x_vals = np.linspace(-2.5, 3.5, 500)
f_vals = x_vals + 2
g_vals = x_vals**2

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x_vals, f_vals, 'b-', linewidth=2, label='$y = x + 2$')
ax.plot(x_vals, g_vals, 'r-', linewidth=2, label='$y = x^2$')

# Shade the area between curves
x_fill = np.linspace(a, b, 300)
ax.fill_between(x_fill, x_fill + 2, x_fill**2, alpha=0.3, color='green',
                label=f'Area = {float(area):.2f}')

ax.plot([a, b], [a+2, b+2], 'ko', markersize=8, zorder=5)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Area Between Two Curves')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('area_between_curves.png', dpi=150)
plt.show()
```

## 회전체의 부피

영역이 축을 중심으로 회전하면 3D 고체를 쓸어낸다. 이 고체를 얇은 조각으로 잘라 부피를 계산한다.

### 디스크 방법

$x$축을 중심으로 회전할 때, 축에 수직인 각 단면은 **디스크(disk)** (채워진 원)이다:

$$V = \int_a^b \pi [f(x)]^2 \, dx$$

- $f(x)$: 위치 $x$에서 디스크의 반지름
- $[f(x)]^2$: 원형 단면의 면적 ($\pi r^2$)
- $dx$: 각 디스크의 두께

**비유:** 무한히 많은 얇은 동전을 쌓는다고 상상하라. 각 동전은 곡선에 의해 결정되는 다른 반지름을 가진다.

### 와셔 방법

가운데에 구멍이 있으면 (두 곡선 사이의 영역이 회전), 각 단면은 **와셔(washer)** (고리)이다:

$$V = \int_a^b \pi \left([R(x)]^2 - [r(x)]^2\right) dx$$

- $R(x)$: 외부 반지름 (축에서 더 먼 쪽)
- $r(x)$: 내부 반지름 (축에 더 가까운 쪽)

### 셸 방법

때때로 디스크 대신 **원통형 셸(cylindrical shell)** 을 사용하는 것이 더 쉽다. $y$축을 중심으로 회전할 때, 위치 $x$의 얇은 수직 띠는 원통형 셸을 쓸어낸다:

$$V = \int_a^b 2\pi x \cdot f(x) \, dx$$

- $2\pi x$: 셸의 둘레 (축에서 거리 $x$)
- $f(x)$: 셸의 높이
- $dx$: 두께

**언제 어떤 것을 사용하는가:**
- **디스크/와셔**: 회전축에 수직인 단면이 단순할 때
- **셸**: 회전축에 평행한 단면이 더 단순할 때

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp

x_sym = sp.Symbol('x')

# Example: Rotate y = sqrt(x) from x=0 to x=4 around the x-axis
# Disk method: V = pi * integral_0^4 (sqrt(x))^2 dx = pi * integral_0^4 x dx
V_disk = sp.pi * sp.integrate(x_sym, (x_sym, 0, 4))
print(f"Disk method: V = pi * integral_0^4 x dx = {V_disk} = {float(V_disk):.4f}")

# Shell method for the same solid (rotating around x-axis):
# We must express x as a function of y: x = y^2, and y ranges from 0 to 2
y_sym = sp.Symbol('y')
V_shell = 2 * sp.pi * sp.integrate(y_sym * (4 - y_sym**2), (y_sym, 0, 2))
print(f"Shell method: V = 2pi * integral_0^2 y*(4-y^2) dy = {V_shell} = {float(V_shell):.4f}")
print(f"Both methods agree: {V_disk == V_shell}")

# 3D visualization of the solid of revolution
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create the surface of revolution
theta = np.linspace(0, 2*np.pi, 100)
x_3d = np.linspace(0, 4, 100)
Theta, X = np.meshgrid(theta, x_3d)

# y = sqrt(x) rotated around x-axis gives r = sqrt(x)
R = np.sqrt(X)
Y = R * np.cos(Theta)
Z = R * np.sin(Theta)

ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
ax.set_title('Solid of Revolution: $y = \\sqrt{x}$ rotated around $x$-axis')
plt.tight_layout()
plt.savefig('volume_revolution_3d.png', dpi=150)
plt.show()
```

### 와셔 예시

$y = x$와 $y = x^2$ ($x=0$부터 $x=1$까지) 사이의 영역을 $x$축 중심으로 회전하여 얻은 부피를 구하라.

$$V = \int_0^1 \pi\left[(x)^2 - (x^2)^2\right] dx = \pi \int_0^1 (x^2 - x^4) \, dx = \pi\left[\frac{x^3}{3} - \frac{x^5}{5}\right]_0^1 = \frac{2\pi}{15}$$

```python
import sympy as sp

x = sp.Symbol('x')

# Washer method: region between y=x and y=x^2, rotated about x-axis
# Outer radius R(x) = x (farther from axis), inner radius r(x) = x^2
V_washer = sp.pi * sp.integrate(x**2 - x**4, (x, 0, 1))
print(f"Washer volume: {V_washer} = {float(V_washer):.6f}")
```

## 호의 길이

### 직교 좌표 형태

$x = a$부터 $x = b$까지 곡선 $y = f(x)$의 길이는:

$$L = \int_a^b \sqrt{1 + [f'(x)]^2} \, dx$$

**유도:** 곡선의 작은 조각은 수평 길이 $dx$와 수직 길이 $dy = f'(x) \, dx$를 가진다. 피타고라스 정리에 의해, 호의 길이 요소는:

$$ds = \sqrt{dx^2 + dy^2} = \sqrt{1 + \left(\frac{dy}{dx}\right)^2} \, dx$$

$ds$를 적분하면 총 호의 길이를 얻는다.

### 매개변수 형태

곡선이 $x = x(t)$, $y = y(t)$ ($t \in [\alpha, \beta]$)로 주어지면:

$$L = \int_\alpha^\beta \sqrt{\left(\frac{dx}{dt}\right)^2 + \left(\frac{dy}{dt}\right)^2} \, dt$$

```python
import numpy as np
import sympy as sp
from scipy import integrate

x = sp.Symbol('x')

# Arc length of y = x^(3/2) from x=0 to x=4
f = x**sp.Rational(3, 2)
f_prime = sp.diff(f, x)
integrand = sp.sqrt(1 + f_prime**2)
print(f"f(x) = {f}")
print(f"f'(x) = {f_prime}")
print(f"Arc length integrand: sqrt(1 + (f')^2) = {sp.simplify(integrand)}")

# Symbolic integration
L_exact = sp.integrate(integrand, (x, 0, 4))
print(f"Exact arc length: {L_exact} = {float(L_exact):.6f}")

# Numerical verification using scipy
f_numeric = lambda t: np.sqrt(1 + (1.5 * np.sqrt(t))**2)
L_numerical, _ = integrate.quad(f_numeric, 0, 4)
print(f"Numerical arc length: {L_numerical:.6f}")

# Parametric example: circle x = cos(t), y = sin(t), t in [0, 2pi]
t = sp.Symbol('t')
x_param = sp.cos(t)
y_param = sp.sin(t)
ds = sp.sqrt(sp.diff(x_param, t)**2 + sp.diff(y_param, t)**2)
L_circle = sp.integrate(ds, (t, 0, 2*sp.pi))
print(f"\nCircle circumference (parametric): {L_circle} = {float(L_circle):.6f}")
```

## 회전체의 겉넓이

곡선 $y = f(x)$가 $x$축 중심으로 회전할 때, 겉넓이는:

$$S = \int_a^b 2\pi f(x) \sqrt{1 + [f'(x)]^2} \, dx$$

**직관:** 각 호의 길이 요소 $ds$는 둘레 $2\pi f(x)$인 얇은 띠(절두체)를 쓸어낸다. 이 띠의 면적은 $2\pi f(x) \, ds$이다.

이것은 두 가지 아이디어를 결합한다:
- 호의 길이 요소 $ds = \sqrt{1 + [f'(x)]^2} \, dx$
- 각 점이 그리는 원의 둘레 $2\pi r$

```python
import numpy as np
import sympy as sp

x = sp.Symbol('x')

# Surface area of y = sqrt(x) from x=0 to x=1, rotated about x-axis
f = sp.sqrt(x)
f_prime = sp.diff(f, x)
integrand = 2 * sp.pi * f * sp.sqrt(1 + f_prime**2)

print(f"Surface area integrand: {sp.simplify(integrand)}")
SA = sp.integrate(integrand, (x, 0, 1))
print(f"Surface area = {SA} = {float(SA):.6f}")

# Verify: surface area of a sphere (rotate y = sqrt(r^2 - x^2) about x-axis)
r = sp.Symbol('r', positive=True)
f_sphere = sp.sqrt(r**2 - x**2)
f_sphere_prime = sp.diff(f_sphere, x)
integrand_sphere = 2 * sp.pi * f_sphere * sp.sqrt(1 + f_sphere_prime**2)
SA_sphere = sp.integrate(sp.simplify(integrand_sphere), (x, -r, r))
print(f"\nSurface area of sphere: {SA_sphere}")
# Should give 4*pi*r^2, the well-known formula
```

## 물리적 응용

### 일 (Work)

물리학에서, **일(work)** 은 거리에 대한 힘의 적분이다:

$$W = \int_a^b F(x) \, dx$$

- $F(x)$: 위치의 함수로서의 힘 (뉴턴 단위)
- $dx$: 무한소 변위
- $W$: 총 일 (줄 단위)

**예시: 스프링의 일.** 훅의 법칙(Hooke's law)은 $F(x) = kx$라고 말한다. 여기서 $k$는 스프링 상수이고 $x$는 평형 위치로부터의 변위이다. 스프링을 $x = 0$에서 $x = d$까지 늘리는 데 드는 일:

$$W = \int_0^d kx \, dx = \frac{1}{2} k d^2$$

```python
import sympy as sp

x = sp.Symbol('x')
k = sp.Symbol('k', positive=True)
d = sp.Symbol('d', positive=True)

# Work to stretch a spring from 0 to d
W_spring = sp.integrate(k * x, (x, 0, d))
print(f"Work to stretch spring: W = {W_spring}")

# Numerical example: k = 200 N/m, stretch 0.3 m
W_numeric = W_spring.subs([(k, 200), (d, 0.3)])
print(f"With k=200 N/m, d=0.3 m: W = {float(W_numeric)} J")
```

### 정수압과 힘

댐이나 잠긴 판은 깊이에 따라 증가하는 압력을 경험한다. 수면 아래 깊이 $h$에서 수평 띠에 작용하는 힘:

$$dF = \rho g h \cdot w(h) \, dh$$

여기서 $\rho$는 유체 밀도 (물의 경우 $\approx 1000$ kg/m$^3$), $g \approx 9.8$ m/s$^2$, $w(h)$는 깊이 $h$에서 판의 너비이다.

총 힘:

$$F = \int_0^H \rho g h \cdot w(h) \, dh$$

### 무게 중심

밀도 $\rho$를 가진 박판(lamina, 얇은 평판)이 $[a, b]$에서 $y = f(x)$로 경계지어지면:

$$\bar{x} = \frac{\int_a^b x \cdot f(x) \, dx}{\int_a^b f(x) \, dx}, \qquad \bar{y} = \frac{\frac{1}{2}\int_a^b [f(x)]^2 \, dx}{\int_a^b f(x) \, dx}$$

- $\bar{x}$: 무게 중심의 $x$-좌표 ($x$의 가중 평균)
- $\bar{y}$: $y$-좌표 (각 수평 띠의 중심은 높이 $f(x)/2$에 있다)

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x = sp.Symbol('x')

# Center of mass of a semicircular lamina: y = sqrt(1 - x^2)
f = sp.sqrt(1 - x**2)

# Total area (mass, assuming uniform density)
area = sp.integrate(f, (x, -1, 1))  # Should be pi/2
print(f"Area = {area}")

# x-bar: by symmetry, should be 0
x_bar = sp.integrate(x * f, (x, -1, 1)) / area
print(f"x_bar = {x_bar}")

# y-bar
y_bar = sp.Rational(1, 2) * sp.integrate(f**2, (x, -1, 1)) / area
print(f"y_bar = {y_bar} = {float(y_bar):.6f}")
# y_bar = 4/(3*pi) ≈ 0.4244

# Visualize with center of mass marked
theta = np.linspace(0, np.pi, 200)
x_circle = np.cos(theta)
y_circle = np.sin(theta)

fig, ax = plt.subplots(figsize=(8, 6))
ax.fill(np.append(x_circle, x_circle[-1]),
        np.append(y_circle, 0), alpha=0.3, color='blue')
ax.plot(x_circle, y_circle, 'b-', linewidth=2)
ax.plot([-1, 1], [0, 0], 'b-', linewidth=2)
ax.plot(float(x_bar), float(y_bar), 'r*', markersize=15, zorder=5,
        label=f'Center of mass: (0, {float(y_bar):.4f})')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Center of Mass of a Semicircular Lamina')
ax.legend()
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('center_of_mass.png', dpi=150)
plt.show()
```

## 요약

- **곡선 사이의 면적**: $\int_a^b [f(x) - g(x)] \, dx$ -- 항상 아래 함수를 위 함수에서 뺀다
- **디스크 방법**: $V = \pi \int [f(x)]^2 \, dx$ -- 구멍이 없는 고체, 축에 수직으로 자르기
- **와셔 방법**: $V = \pi \int [R^2 - r^2] \, dx$ -- 구멍이 있는 고체
- **셸 방법**: $V = 2\pi \int x \cdot f(x) \, dx$ -- 원통형 셸, $y$축 중심 회전 시 종종 더 쉬움
- **호의 길이**: $L = \int \sqrt{1 + [f'(x)]^2} \, dx$ -- 무한소에 적용된 피타고라스 정리
- **겉넓이**: $S = 2\pi \int f(x) \sqrt{1 + [f'(x)]^2} \, dx$ -- 호의 길이 곱하기 둘레
- **물리적 응용**: 일(work), 정수압, 무게 중심 모두 "자르고, 근사하고, 적분하기" 패러다임을 따른다

## 연습 문제

### 문제 1: 곡선 사이의 면적

$x = 0$과 $x = \pi/2$ 사이에서 $y = \sin x$와 $y = \cos x$로 둘러싸인 영역의 면적을 구하라. (힌트: 곡선이 어디서 교차하는지, 각 부분 구간에서 어떤 것이 위에 있는지 결정하라.)

### 문제 2: 디스크/와셔에 의한 부피

$y = \sqrt{x}$, $y = 0$, $x = 4$로 경계지어진 영역이 $x$축 중심으로 회전한다. 다음을 사용하여 부피를 구하라:
(a) 디스크 방법
(b) Python으로 답을 검증하라

### 문제 3: 셸 방법에 의한 부피

$y = x - x^2$와 $y = 0$으로 경계지어진 영역이 $y$축 중심으로 회전한다. 셸 방법을 사용하여 부피를 구하라. 그런 다음 ($y$에 대한) 와셔 방법으로 같은 부피를 계산하고 일치하는지 검증하라.

### 문제 4: 호의 길이

$x = 1$부터 $x = 2$까지 $y = \frac{x^2}{2} - \frac{\ln x}{4}$의 호의 길이를 계산하라. (이 적분은 깔끔하게 단순화된다 -- 왜 그런지 보여라.)

### 문제 5: 물리적 응용

원뿔형 탱크 (꼭짓점이 아래)의 높이가 6 m이고 꼭대기 반지름이 3 m이며, 물 ($\rho = 1000$ kg/m$^3$)로 채워져 있다. 모든 물을 탱크 꼭대기로 퍼올리는 데 필요한 일을 계산하라.

(힌트: 바닥으로부터 높이 $y$에 있는 얇은 수평 물 조각은 반지름 $r = y/2$이며, 거리 $(6 - y)$만큼 들어 올려야 한다.)

## 참고 자료

- Stewart, *Calculus: Early Transcendentals*, Ch. 6 (Applications of Integration)
- [3Blue1Brown: Volumes of Revolution](https://www.youtube.com/watch?v=rjLJIVoQxz4)
- [Paul's Online Notes: Applications of Integrals](https://tutorial.math.lamar.edu/Classes/CalcI/AreaBetweenCurves.aspx)

---

[이전: 적분 기법](./05_Integration_Techniques.md) | [다음: 수열과 급수](./07_Sequences_and_Series.md)
