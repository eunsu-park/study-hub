# 도함수의 응용

## 학습 목표

- 임계점(critical point)을 **찾고** 1계 및 2계 도함수 판정법을 사용하여 극대, 극소, 또는 안장점으로 분류할 수 있다
- 실세계 제약 조건을 수학적 공식으로 변환하여 최적화(optimization) 문제를 **풀** 수 있다
- 여러 양이 동시에 변하는 문제에 관련 변화율(related rates) 기법을 **적용**할 수 있다
- 로피탈의 법칙(L'Hopital's rule)을 사용하여 부정형(indeterminate form)을 **계산**할 수 있다
- 한 점 근처에서 함수값을 근사하기 위해 선형 근사(linear approximation)와 테일러 다항식(Taylor polynomial)을 **구성**할 수 있다

## 소개

도함수를 계산하는 방법을 아는 것은 지도를 읽는 방법을 아는 것과 같다. 진정한 힘은 그 기술을 사용하여 항해하는 것에서 나온다 -- 가장 높은 산을 찾고, 그림자가 얼마나 빠르게 움직이는지 예측하고, 계산기 없이 함수값을 추정하는 것. 이 레슨은 과학과 공학 전반에 걸쳐 나타나는 도함수의 주요 응용을 다룬다.

## 임계점과 극값

### 임계점 찾기

$f$의 **임계점(critical point)** 은 $f'(c) = 0$이거나 $f'(c)$가 존재하지 않는 정의역 내의 값 $c$이다.

**왜 중요한가:** 극값 정리(Extreme Value Theorem)는 닫힌 구간 $[a, b]$에서 연속인 함수가 최댓값과 최솟값을 모두 달성함을 보장한다. 이 극값은 임계점이나 끝점에서 발생한다.

### 1계 도함수 판정법

임계점 $c$ 양쪽에서 $f'(x)$의 부호를 조사한다:

| $c$ 이전의 $f'$ | $c$ 이후의 $f'$ | 결론 |
|------------------|-----------------|------------|
| $+$ (증가) | $-$ (감소) | $c$에서 **극대(local maximum)** |
| $-$ (감소) | $+$ (증가) | $c$에서 **극소(local minimum)** |
| 같은 부호 | 같은 부호 | 어느 쪽도 아님 (변곡점 가능) |

### 2계 도함수 판정법

$f'(c) = 0$이고 $c$에서 $f''$이 존재하면:

- $f''(c) > 0$: **극소(local minimum)** (곡선이 위로 볼록, 그릇 모양)
- $f''(c) < 0$: **극대(local maximum)** (곡선이 아래로 볼록, 언덕 모양)
- $f''(c) = 0$: 판정 **불가(inconclusive)** (1계 도함수 판정법을 대신 사용)

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x = sp.Symbol('x')

# Example: f(x) = x^3 - 3x^2 + 1
f = x**3 - 3*x**2 + 1
f_prime = sp.diff(f, x)
f_double_prime = sp.diff(f, x, 2)

print(f"f(x) = {f}")
print(f"f'(x) = {f_prime}")
print(f"f''(x) = {f_double_prime}")

# Find critical points: where f'(x) = 0
critical_points = sp.solve(f_prime, x)
print(f"\nCritical points: {critical_points}")

for cp in critical_points:
    second_deriv_val = f_double_prime.subs(x, cp)
    if second_deriv_val > 0:
        classification = "local minimum"
    elif second_deriv_val < 0:
        classification = "local maximum"
    else:
        classification = "inconclusive"
    print(f"  x = {cp}: f''({cp}) = {second_deriv_val} --> {classification}")
    print(f"    f({cp}) = {f.subs(x, cp)}")

# Visualization
x_vals = np.linspace(-1, 4, 500)
f_np = lambda t: t**3 - 3*t**2 + 1

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Function plot
ax1.plot(x_vals, f_np(x_vals), 'b-', linewidth=2, label='$f(x) = x^3 - 3x^2 + 1$')
for cp in critical_points:
    cp_float = float(cp)
    ax1.plot(cp_float, f_np(cp_float), 'ro', markersize=10, zorder=5)
    ax1.annotate(f'({cp_float}, {f_np(cp_float):.0f})',
                 (cp_float, f_np(cp_float)), textcoords="offset points",
                 xytext=(15, 10), fontsize=11)
ax1.set_ylabel('$f(x)$')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title('Critical Points and Extrema')

# First derivative plot
f_prime_np = lambda t: 3*t**2 - 6*t
ax2.plot(x_vals, f_prime_np(x_vals), 'r-', linewidth=2, label="$f'(x) = 3x^2 - 6x$")
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.fill_between(x_vals, f_prime_np(x_vals), 0,
                  where=(f_prime_np(x_vals) > 0), alpha=0.2, color='green',
                  label='$f\' > 0$ (increasing)')
ax2.fill_between(x_vals, f_prime_np(x_vals), 0,
                  where=(f_prime_np(x_vals) < 0), alpha=0.2, color='red',
                  label='$f\' < 0$ (decreasing)')
ax2.set_xlabel('$x$')
ax2.set_ylabel("$f'(x)$")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('critical_points.png', dpi=150)
plt.show()
```

## 최적화 문제

최적화(optimization)는 아마도 도함수의 가장 실용적인 응용일 것이다. 전략은 항상 같다:

1. **그림을 그리고** 변수를 식별한다
2. 최적화할 양을 하나의 변수의 함수로 **쓴다**
3. 정의역을 **찾는다** (물리적 제약 조건)
4. **미분**하고, 0과 같다고 놓고, 풀다
5. 결과가 최대/최소인지 **검증**한다 (2계 도함수 판정법 또는 끝점 검사)

### 예제: 최대 울타리 면적

한 농부가 200미터의 울타리를 가지고 있으며, 강에 접해 있는 가능한 가장 큰 직사각형 면적을 울타리로 둘러싸려 한다 (강변에는 울타리가 필요 없다).

**설정:** $x$ = 폭 (강에 수직), $y$ = 길이 (강에 평행).

제약 조건: $2x + y = 200$이므로 $y = 200 - 2x$.

목적: $A = xy = x(200 - 2x) = 200x - 2x^2$를 최대화.

$$A'(x) = 200 - 4x = 0 \implies x = 50$$

$$A''(x) = -4 < 0 \quad \text{(최대 확인)}$$

따라서 $x = 50$ m, $y = 100$ m, $A_{\max} = 5000$ m$^2$.

```python
import numpy as np
import matplotlib.pyplot as plt

# Optimization: maximize area A(x) = x(200 - 2x)
x = np.linspace(0, 100, 500)
A = x * (200 - 2*x)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, A, 'b-', linewidth=2)
ax.plot(50, 5000, 'ro', markersize=10, zorder=5)
ax.annotate('Maximum: (50, 5000)', (50, 5000),
            textcoords="offset points", xytext=(20, -20),
            fontsize=12, arrowprops=dict(arrowstyle='->', color='red'))
ax.set_xlabel('Width $x$ (meters)')
ax.set_ylabel('Area $A$ (m$^2$)')
ax.set_title('Fencing Problem: Area vs Width')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('optimization_fencing.png', dpi=150)
plt.show()
```

### 예제: 재료 비용 최소화

부피 $V = 500$ cm$^3$인 뚜껑 없는 원통형 캔을 최소 재료로 설계하라.

**설정:** 변수는 반지름 $r$과 높이 $h$이다. 캔에는 바닥 ($\pi r^2$)과 옆면 ($2\pi r h$)이 있다.

제약 조건: $\pi r^2 h = 500$이므로 $h = \frac{500}{\pi r^2}$.

표면적: $S = \pi r^2 + 2\pi r h = \pi r^2 + \frac{1000}{r}$

$$S'(r) = 2\pi r - \frac{1000}{r^2} = 0 \implies r^3 = \frac{500}{\pi} \implies r = \left(\frac{500}{\pi}\right)^{1/3}$$

```python
import numpy as np
import sympy as sp

r = sp.Symbol('r', positive=True)
V = 500

# Surface area as a function of r only (h eliminated using volume constraint)
S = sp.pi * r**2 + 1000 / r
dS = sp.diff(S, r)
r_opt = sp.solve(dS, r)[0]
h_opt = V / (sp.pi * r_opt**2)

print(f"Optimal radius: r = {r_opt} = {float(r_opt):.4f} cm")
print(f"Optimal height: h = {float(h_opt):.4f} cm")
print(f"Minimum surface area: {float(S.subs(r, r_opt)):.4f} cm^2")
print(f"Ratio h/r = {float(h_opt / r_opt):.4f}")
# Notice: h/r = 2 at the optimum -- the height equals the diameter!
```

## 관련 변화율

관련 변화율(related rates) 문제에서는 여러 양이 시간에 따라 변하며, 하나의 변화율을 알고 다른 것의 변화율을 구하고자 한다.

**전략:** 양들의 관계식을 쓰고, 연쇄 법칙을 사용하여 양변을 시간 $t$에 대해 미분한 다음, 알려진 값을 대입한다.

### 예제: 확장하는 원

연못에 떨어진 돌이 반지름이 3 cm/s로 증가하는 원형 물결을 만든다. $r = 10$ cm일 때 면적은 얼마나 빠르게 증가하는가?

**방정식:** $A = \pi r^2$

**$t$에 대해 미분:**

$$\frac{dA}{dt} = 2\pi r \frac{dr}{dt}$$

- $\frac{dr}{dt} = 3$ cm/s (주어진 것)
- $r = 10$ cm (주어진 순간)

$$\frac{dA}{dt} = 2\pi(10)(3) = 60\pi \approx 188.5 \text{ cm}^2/\text{s}$$

**직관:** 원이 커질수록 면적이 더 빠르게 증가하는데, 이는 둘레 (바깥으로 밀려나는 "경계")가 더 길기 때문이다.

### 예제: 사다리 문제

10미터 사다리가 벽에 기대어 있다. 밑부분이 1 m/s로 미끄러져 나간다. 밑부분이 벽에서 6 m 떨어져 있을 때 꼭대기는 얼마나 빠르게 미끄러져 내려오는가?

$$x^2 + y^2 = 100$$

$$2x \frac{dx}{dt} + 2y \frac{dy}{dt} = 0$$

$x = 6$일 때: $y = \sqrt{100 - 36} = 8$.

$$2(6)(1) + 2(8)\frac{dy}{dt} = 0 \implies \frac{dy}{dt} = -\frac{3}{4} \text{ m/s}$$

음의 부호는 $y$가 감소한다는 것을 의미한다 (꼭대기가 *아래로* 미끄러진다).

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize the ladder problem over time
# x(t) = initial_x + t (bottom moves at 1 m/s)
# y(t) = sqrt(100 - x(t)^2) (Pythagorean constraint)

t = np.linspace(0, 3.5, 100)
x0 = 6.0
x_t = x0 + t
y_t = np.sqrt(np.maximum(100 - x_t**2, 0))

# Rate of top sliding: dy/dt = -x * (dx/dt) / y
dx_dt = 1.0
dy_dt = -x_t * dx_dt / np.where(y_t > 0, y_t, np.nan)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Position plot
for i in range(0, len(t), 15):
    alpha = 0.3 + 0.7 * i / len(t)
    ax1.plot([x_t[i], 0], [0, y_t[i]], 'b-', alpha=alpha, linewidth=2)
ax1.set_xlabel('$x$ (m)')
ax1.set_ylabel('$y$ (m)')
ax1.set_title('Ladder sliding along wall')
ax1.set_aspect('equal')
ax1.set_xlim(-0.5, 11)
ax1.set_ylim(-0.5, 11)
ax1.grid(True, alpha=0.3)

# Rate plot -- shows dy/dt accelerates as the ladder falls
ax2.plot(t, dy_dt, 'r-', linewidth=2)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('$dy/dt$ (m/s)')
ax2.set_title('Rate of top sliding down')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linewidth=0.5)
# Note: the top accelerates downward -- the rate becomes more negative
# over time. As y -> 0, the speed approaches infinity (the ladder
# "slaps" the ground), which is physically unrealistic but mathematically exact.

plt.tight_layout()
plt.savefig('related_rates_ladder.png', dpi=150)
plt.show()
```

## 로피탈의 법칙

극한이 **부정형(indeterminate form)** ($\frac{0}{0}$ 또는 $\frac{\infty}{\infty}$)을 산출할 때, 로피탈의 법칙(L'Hopital's rule)은 우아한 탈출구를 제공한다:

$$\text{만약 } \lim_{x \to a} \frac{f(x)}{g(x)} \text{ 가 } \frac{0}{0} \text{ 또는 } \frac{\infty}{\infty}\text{이면, } \lim_{x \to a} \frac{f(x)}{g(x)} = \lim_{x \to a} \frac{f'(x)}{g'(x)}$$

단, 우변의 극한이 존재해야 한다.

**예시:**

$$\lim_{x \to 0} \frac{\sin x}{x} = \lim_{x \to 0} \frac{\cos x}{1} = 1$$

$$\lim_{x \to \infty} \frac{\ln x}{x} = \lim_{x \to \infty} \frac{1/x}{1} = 0$$

$$\lim_{x \to 0} \frac{e^x - 1 - x}{x^2} \stackrel{\frac{0}{0}}{=} \lim_{x \to 0} \frac{e^x - 1}{2x} \stackrel{\frac{0}{0}}{=} \lim_{x \to 0} \frac{e^x}{2} = \frac{1}{2}$$

**주의:** 다른 부정형 ($0 \cdot \infty$, $\infty - \infty$, $0^0$, $1^\infty$, $\infty^0$)은 먼저 $\frac{0}{0}$ 또는 $\frac{\infty}{\infty}$ 형태로 변환해야 한다.

```python
import sympy as sp

x = sp.Symbol('x')

# Verify L'Hopital examples with SymPy
examples = [
    (sp.sin(x) / x, 0, "sin(x)/x"),
    (sp.log(x) / x, sp.oo, "ln(x)/x"),
    ((sp.exp(x) - 1 - x) / x**2, 0, "(e^x - 1 - x)/x^2"),
    ((1 - sp.cos(x)) / x**2, 0, "(1 - cos(x))/x^2"),
]

for expr, point, name in examples:
    result = sp.limit(expr, x, point)
    print(f"lim {name} as x->{point}: {result}")
```

## 선형 근사와 테일러 다항식

### 선형 근사

$x = a$ 근처에서, 미분 가능한 함수는 접선으로 잘 근사된다:

$$f(x) \approx f(a) + f'(a)(x - a) \quad \text{($x$가 $a$ 근처일 때)}$$

이것을 **선형화(linearization)** 또는 **1차 테일러 근사(first-order Taylor approximation)** 라 한다. 이것은 미분방정식이 종종 선형화로 풀릴 수 있는 이유이며, 뉴턴 방법이 작동하는 이유의 기초이다.

**예시:** 계산기 없이 $\sqrt{4.1}$을 추정하라.

$f(x) = \sqrt{x}$, $a = 4$라 하자. 그러면 $f(4) = 2$, $f'(x) = \frac{1}{2\sqrt{x}}$, $f'(4) = \frac{1}{4}$.

$$\sqrt{4.1} \approx 2 + \frac{1}{4}(4.1 - 4) = 2 + 0.025 = 2.025$$

실제 값: $\sqrt{4.1} = 2.02485...$. 선형 근사는 소수점 3자리까지 정확하다.

### 테일러 다항식

더 나은 정확도를 위해, 고차 항을 포함한다. $a$를 중심으로 한 $f$의 **$n$차 테일러 다항식**은:

$$T_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(a)}{k!}(x - a)^k = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots$$

- $f^{(k)}(a)$: $a$에서 평가된 $k$계 도함수
- $k!$: 팩토리얼, 계수가 함수의 행동과 일치하도록 보장
- $(x - a)^k$: 각 항은 더 미세한 국소적 세부 사항을 포착

$a = 0$일 때, 이를 **매클로린 다항식(Maclaurin polynomial)** 이라 한다.

### 뉴턴 방법

뉴턴 방법(Newton's method)은 선형 근사를 반복적으로 사용하여 근을 찾는다:

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

**기하학적 관점:** 각 단계에서, 곡선을 접선으로 대체하고 접선이 $x$축과 만나는 점을 찾는다.

```python
import numpy as np
import matplotlib.pyplot as plt

def newtons_method(f, df, x0, tol=1e-12, max_iter=50, verbose=True):
    """
    Newton's method for finding roots of f(x) = 0.

    Each iteration replaces f with its tangent line at the current
    estimate and solves for where the tangent crosses zero. This
    gives quadratic convergence -- the number of correct digits
    roughly doubles each iteration.
    """
    x = x0
    history = [x]

    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-15:
            print("WARNING: derivative near zero, method may fail")
            break
        x_new = x - fx / dfx
        history.append(x_new)

        if verbose:
            print(f"  Iter {i+1}: x = {x_new:.15f}, f(x) = {f(x_new):.2e}")

        if abs(x_new - x) < tol:
            break
        x = x_new

    return x_new, history

# Find sqrt(2) using Newton's method on f(x) = x^2 - 2
print("Finding sqrt(2):")
f = lambda x: x**2 - 2
df = lambda x: 2*x
root, hist = newtons_method(f, df, x0=1.0)
print(f"\nResult: {root:.15f}")
print(f"Actual: {np.sqrt(2):.15f}")

# Visualize convergence -- note how fast it converges
errors = [abs(x - np.sqrt(2)) for x in hist]
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(range(len(errors)), errors, 'bo-', linewidth=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('|Error|')
ax.set_title("Newton's Method: Quadratic Convergence")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('newtons_method.png', dpi=150)
plt.show()
```

## 요약

- **임계점(critical points)** 은 $f'(x) = 0$이거나 $f'$이 존재하지 않는 곳에서 발생한다; 1계 또는 2계 도함수 판정법을 사용하여 분류한다
- **최적화(optimization)** 는 실세계 제약 조건을 미적분학으로 변환한다: 목적 함수를 하나의 변수의 함수로 표현한 다음 임계점을 찾는다
- **관련 변화율(related rates)** 은 연쇄 법칙을 사용하여 연결된 양들의 변화율을 연결한다
- **로피탈의 법칙(L'Hopital's rule)** 은 분자와 분모를 미분하여 $\frac{0}{0}$과 $\frac{\infty}{\infty}$ 부정형을 해결한다
- **선형 근사(linear approximation)** ($f(x) \approx f(a) + f'(a)(x-a)$)는 가장 단순한 테일러 다항식이며 뉴턴 방법의 기초이다
- **뉴턴 방법(Newton's method)** 은 2차 수렴(quadratic convergence)을 달성하여, 각 반복에서 정확한 자릿수가 대략 두 배가 된다

## 연습 문제

### 문제 1: 극값 찾기와 분류

$f(x) = x^4 - 4x^3 + 4x^2$의 모든 임계점을 찾고 각각을 극대, 극소, 또는 어느 것도 아닌 것으로 분류하라. 함수를 그려 검증하라.

### 문제 2: 최적화

정사각형 밑면과 열린 윗면을 가진 직사각형 상자의 부피가 32,000 cm$^3$이어야 한다. 사용되는 재료의 양을 최소화하는 치수를 구하라 (즉, 표면적을 최소화하라).

### 문제 3: 관련 변화율

공기가 100 cm$^3$/s의 속도로 구형 풍선에 주입된다. 지름이 50 cm일 때 반지름은 얼마나 빠르게 증가하는가? ($V = \frac{4}{3}\pi r^3$)

### 문제 4: 로피탈의 법칙

각 극한을 계산하라:

(a) $\lim_{x \to 0} \frac{e^x - 1 - x - x^2/2}{x^3}$

(b) $\lim_{x \to 0^+} x \ln x$ (힌트: $\frac{\ln x}{1/x}$로 다시 쓰라)

(c) $\lim_{x \to \infty} x^{1/x}$ (힌트: $y = x^{1/x}$로 놓고 $\ln$을 취하라)

### 문제 5: 테일러 다항식 근사

(a) $f(x) = e^x$의 4차 매클로린 다항식을 쓰라. 이를 사용하여 $e^{0.5}$를 추정하고 실제 값과 비교하라.

(b) $[-2\pi, 2\pi]$에서 $\sin(x)$를 1, 3, 5, 7, 9차 테일러 다항식과 함께 그리는 Python 코드를 작성하라. 고차 다항식이 더 넓은 범위에서 함수를 근사하는 것을 관찰하라.

## 참고 자료

- Stewart, *Calculus: Early Transcendentals*, Ch. 4 (Applications of Differentiation)
- [3Blue1Brown: Optimization](https://www.youtube.com/watch?v=IHZwWFHWa-w)
- [Paul's Online Notes: Applications of Derivatives](https://tutorial.math.lamar.edu/Classes/CalcI/DerivAppsIntro.aspx)

---

[이전: 도함수의 기초](./02_Derivatives_Fundamentals.md) | [다음: 적분의 기초](./04_Integration_Fundamentals.md)
