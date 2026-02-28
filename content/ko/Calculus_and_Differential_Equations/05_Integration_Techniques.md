# 적분 기법

## 학습 목표

- 합성 함수를 포함하는 적분을 계산하기 위해 $u$-치환($u$-substitution)을 **적용**할 수 있다
- 부분적분(integration by parts)을 **수행**하고 적용 시기를 인식할 수 있다 (LIATE 규칙)
- 부분분수(partial fractions)를 사용하여 유리함수를 **분해**하고 각 항을 적분할 수 있다
- 거듭제곱 축소(power-reduction)와 삼각 치환(trigonometric substitution)을 사용하여 삼각 적분을 **계산**할 수 있다
- 이상 적분(improper integral, 제1종 및 제2종)의 수렴(convergence) 또는 발산(divergence)을 **판정**할 수 있다

## 소개

미적분학의 기본 정리는 모든 정적분이 역도함수를 찾아 계산할 수 있다고 말한다. 문제는? 역도함수를 찾는 것이 도함수를 찾는 것보다 훨씬 어려운 경우가 많다는 것이다. 미분은 알고리즘적(규칙을 기계적으로 적용)인 반면, 적분은 일종의 예술이다 -- 패턴 인식, 영리한 치환, 때로는 창의적인 기법이 필요하다.

이렇게 생각해 보자: 미분은 꽃병을 깨뜨리는 것과 같고(쉽고 체계적), 적분은 조각들을 다시 맞추는 것과 같다(조각들이 어떻게 맞는지에 대한 통찰이 필요하다).

이 레슨은 실제로 만나는 대다수의 적분을 처리하는 필수 기법들을 다룬다.

## 치환법 (u-치환)

### 아이디어

$u$-치환은 연쇄 법칙(chain rule)의 적분 대응물이다. 합성 함수와 그 내부 도함수를 보면 단순화할 수 있다.

**연쇄 법칙 (미분):** $\frac{d}{dx} F(g(x)) = F'(g(x)) \cdot g'(x)$

**치환법 (적분):** $\int F'(g(x)) \cdot g'(x) \, dx = F(g(x)) + C$

### 절차

1. 내부 함수 $u = g(x)$를 **식별**한다
2. $du = g'(x) \, dx$를 **계산**한다
3. 적분을 $u$에 관해 완전히 **다시 쓴다**
4. $u$에 대해 **적분**한다
5. $x$로 다시 **대입**한다

### 예시 1: 기본 치환

$$\int 2x \cos(x^2) \, dx$$

$u = x^2$으로 놓으면, $du = 2x \, dx$이다. 적분은 다음이 된다:

$$\int \cos(u) \, du = \sin(u) + C = \sin(x^2) + C$$

### 예시 2: 상수 조정

$$\int x e^{3x^2} \, dx$$

$u = 3x^2$으로 놓으면, $du = 6x \, dx$이므로 $x \, dx = \frac{du}{6}$.

$$\int e^u \cdot \frac{du}{6} = \frac{1}{6} e^u + C = \frac{1}{6} e^{3x^2} + C$$

### 예시 3: 정적분에서의 치환

정적분의 경우, 변수와 함께 적분 한계도 바꾼다:

$$\int_0^1 x \sqrt{1 - x^2} \, dx$$

$u = 1 - x^2$, $du = -2x \, dx$로 놓자. $x = 0$일 때: $u = 1$. $x = 1$일 때: $u = 0$.

$$\int_1^0 \sqrt{u} \cdot \left(-\frac{du}{2}\right) = \frac{1}{2} \int_0^1 u^{1/2} \, du = \frac{1}{2} \cdot \frac{2}{3} u^{3/2} \Big|_0^1 = \frac{1}{3}$$

```python
import sympy as sp

x = sp.Symbol('x')

# SymPy handles substitution automatically
integrals = [
    2*x * sp.cos(x**2),
    x * sp.exp(3*x**2),
    x * sp.sqrt(1 - x**2),
]

for expr in integrals:
    result = sp.integrate(expr, x)
    print(f"integral of {expr} dx = {result}")

# Definite integral with substitution
print(f"\nintegral_0^1 x*sqrt(1-x^2) dx = "
      f"{sp.integrate(x * sp.sqrt(1 - x**2), (x, 0, 1))}")
```

## 부분적분

### 공식

부분적분(integration by parts)은 곱의 법칙(product rule)의 적분 대응물이다:

$$\int u \, dv = uv - \int v \, du$$

**유도:** 곱의 법칙 $\frac{d}{dx}(uv) = u\frac{dv}{dx} + v\frac{du}{dx}$에서 시작하여 양변을 적분하고 재배열한다.

### LIATE 규칙

어떤 인수를 $u$ (미분할 것)로, 어떤 것을 $dv$ (적분할 것)로 선택할 때 **LIATE** 우선순위를 사용한다:

| 우선순위 | 유형 | 예시 |
|----------|------|----------|
| 1 (최고) | **L**ogarithmic (로그) | $\ln x$, $\log x$ |
| 2 | **I**nverse trig (역삼각) | $\arctan x$, $\arcsin x$ |
| 3 | **A**lgebraic (대수) | $x^2$, $3x + 1$ |
| 4 | **T**rigonometric (삼각) | $\sin x$, $\cos x$ |
| 5 (최저) | **E**xponential (지수) | $e^x$, $2^x$ |

목록에서 더 높은 인수를 $u$로 선택한다 (미분하면 단순해진다).

### 예시 1: $\int x e^x \, dx$

LIATE 사용: $u = x$ (대수), $dv = e^x \, dx$ (지수).

그러면 $du = dx$, $v = e^x$.

$$\int x e^x \, dx = x e^x - \int e^x \, dx = x e^x - e^x + C = e^x(x - 1) + C$$

### 예시 2: $\int x^2 \sin x \, dx$ (두 번 적용)

첫 번째 적용: $u = x^2$, $dv = \sin x \, dx$이면 $du = 2x \, dx$, $v = -\cos x$.

$$\int x^2 \sin x \, dx = -x^2 \cos x + \int 2x \cos x \, dx$$

$\int 2x \cos x \, dx$에 대한 두 번째 적용: $u = 2x$, $dv = \cos x \, dx$.

$$= -x^2 \cos x + 2x \sin x - \int 2 \sin x \, dx = -x^2 \cos x + 2x \sin x + 2\cos x + C$$

### 예시 3: $\int \ln x \, dx$ (로그)

$u = \ln x$, $dv = dx$이면 $du = \frac{1}{x} dx$, $v = x$.

$$\int \ln x \, dx = x \ln x - \int x \cdot \frac{1}{x} \, dx = x \ln x - x + C$$

```python
import sympy as sp

x = sp.Symbol('x')

# Integration by parts examples
by_parts_examples = [
    (x * sp.exp(x), "x * e^x"),
    (x**2 * sp.sin(x), "x^2 * sin(x)"),
    (sp.log(x), "ln(x)"),
    (x * sp.log(x), "x * ln(x)"),
    (sp.exp(x) * sp.sin(x), "e^x * sin(x)"),  # Requires the "cycling" trick
]

for expr, name in by_parts_examples:
    result = sp.integrate(expr, x)
    print(f"integral of {name} dx = {result}")
```

## 부분분수 분해

### 방법

임의의 유리함수 $\frac{P(x)}{Q(x)}$ ($\deg P < \deg Q$인 경우)는 각각을 개별적으로 쉽게 적분할 수 있는 더 단순한 분수로 분해할 수 있다.

**1단계:** 분모를 완전히 인수분해한다.
**2단계:** 인수 유형에 따라 분해를 쓴다:

| 인수 유형 | 분해 |
|-------------|---------------|
| 일차: $(ax + b)$ | $\frac{A}{ax + b}$ |
| 반복 일차: $(ax + b)^n$ | $\frac{A_1}{ax+b} + \frac{A_2}{(ax+b)^2} + \cdots + \frac{A_n}{(ax+b)^n}$ |
| 기약 이차: $(ax^2+bx+c)$ | $\frac{Ax + B}{ax^2+bx+c}$ |

**3단계:** 양변에 분모를 곱하고 계수를 비교하여 상수를 구한다.

### 예시: $\int \frac{x+5}{x^2+x-2} \, dx$

인수분해: $x^2 + x - 2 = (x+2)(x-1)$.

분해: $\frac{x+5}{(x+2)(x-1)} = \frac{A}{x+2} + \frac{B}{x-1}$

양변에 곱하면: $x + 5 = A(x-1) + B(x+2)$

$x = 1$ 대입: $6 = 3B \implies B = 2$.
$x = -2$ 대입: $3 = -3A \implies A = -1$.

$$\int \frac{x+5}{x^2+x-2} \, dx = \int \left(\frac{-1}{x+2} + \frac{2}{x-1}\right) dx = -\ln|x+2| + 2\ln|x-1| + C$$

```python
import sympy as sp

x = sp.Symbol('x')

# Partial fraction decomposition
expr = (x + 5) / (x**2 + x - 2)
decomposed = sp.apart(expr, x)
print(f"Partial fractions of {expr}:")
print(f"  = {decomposed}")

# Integrate
result = sp.integrate(expr, x)
print(f"  integral = {result}")

# More complex example: repeated and irreducible factors
expr2 = (2*x**2 + 3) / (x**3 - x)
decomposed2 = sp.apart(expr2, x)
print(f"\nPartial fractions of {expr2}:")
print(f"  = {decomposed2}")
print(f"  integral = {sp.integrate(expr2, x)}")
```

## 삼각 적분

### 사인과 코사인의 거듭제곱

$\int \sin^m x \cos^n x \, dx$에 대해:

- $m$이 홀수이면: $\sin x$ 하나를 남기고, 나머지를 $\sin^2 x = 1 - \cos^2 x$로 $\cos$로 변환, $u = \cos x$ 치환
- $n$이 홀수이면: $\cos x$ 하나를 남기고, 나머지를 $\cos^2 x = 1 - \sin^2 x$로 $\sin$으로 변환, $u = \sin x$ 치환
- 둘 다 짝수이면: 거듭제곱 축소 항등식 사용:
  - $\sin^2 x = \frac{1 - \cos 2x}{2}$
  - $\cos^2 x = \frac{1 + \cos 2x}{2}$

**예시:** $\int \sin^3 x \cos^2 x \, dx$

$m = 3$이 홀수이므로, $\sin x$ 하나를 남긴다:

$$\int \sin^2 x \cos^2 x \sin x \, dx = \int (1 - \cos^2 x) \cos^2 x \sin x \, dx$$

$u = \cos x$, $du = -\sin x \, dx$로 놓으면:

$$-\int (1 - u^2) u^2 \, du = -\int (u^2 - u^4) \, du = -\frac{u^3}{3} + \frac{u^5}{5} + C = -\frac{\cos^3 x}{3} + \frac{\cos^5 x}{5} + C$$

### 삼각 치환

$\sqrt{a^2 - x^2}$, $\sqrt{a^2 + x^2}$, 또는 $\sqrt{x^2 - a^2}$를 포함하는 적분에 대해:

| 식 | 치환 | 사용되는 항등식 |
|------------|-------------|---------------|
| $\sqrt{a^2 - x^2}$ | $x = a\sin\theta$ | $1 - \sin^2\theta = \cos^2\theta$ |
| $\sqrt{a^2 + x^2}$ | $x = a\tan\theta$ | $1 + \tan^2\theta = \sec^2\theta$ |
| $\sqrt{x^2 - a^2}$ | $x = a\sec\theta$ | $\sec^2\theta - 1 = \tan^2\theta$ |

**예시:** $\int \frac{dx}{\sqrt{4 - x^2}}$

$x = 2\sin\theta$, $dx = 2\cos\theta \, d\theta$로 놓으면:

$$\int \frac{2\cos\theta \, d\theta}{\sqrt{4 - 4\sin^2\theta}} = \int \frac{2\cos\theta \, d\theta}{2\cos\theta} = \int d\theta = \theta + C = \arcsin\frac{x}{2} + C$$

```python
import sympy as sp

x = sp.Symbol('x')

# Trigonometric integrals
trig_examples = [
    (sp.sin(x)**3 * sp.cos(x)**2, "sin^3(x) cos^2(x)"),
    (sp.sin(x)**2 * sp.cos(x)**2, "sin^2(x) cos^2(x)"),
    (1 / sp.sqrt(4 - x**2), "1/sqrt(4 - x^2)"),
    (sp.sqrt(1 + x**2), "sqrt(1 + x^2)"),
]

for expr, name in trig_examples:
    result = sp.integrate(expr, x)
    print(f"integral of {name} dx = {sp.simplify(result)}")
```

## 이상 적분

적분이 **이상(improper)** 인 경우는 다음 중 하나이다:
- **제1종:** 적분 한계 중 하나 또는 둘이 무한
- **제2종:** 피적분함수가 $[a, b]$ 내에서 무한 불연속을 가짐

### 제1종: 무한 한계

$$\int_a^{\infty} f(x) \, dx = \lim_{t \to \infty} \int_a^t f(x) \, dx$$

극한이 존재하고 유한하면, 적분은 **수렴(converges)**; 그렇지 않으면 **발산(diverges)**한다.

**예시 (수렴):**

$$\int_1^{\infty} \frac{1}{x^2} \, dx = \lim_{t \to \infty} \left[-\frac{1}{x}\right]_1^t = \lim_{t \to \infty} \left(-\frac{1}{t} + 1\right) = 1$$

**예시 (발산):**

$$\int_1^{\infty} \frac{1}{x} \, dx = \lim_{t \to \infty} [\ln x]_1^t = \lim_{t \to \infty} \ln t = \infty$$

### $\int_1^{\infty} \frac{1}{x^p} dx$에 대한 $p$-판정법

- $p > 1$이면 수렴
- $p \leq 1$이면 발산

이것은 가장 자주 사용되는 수렴 판정법 중 하나이며, $p$-급수 판정법과 유사하다.

### 제2종: 불연속 피적분함수

$f$가 $c \in [a, b]$에서 불연속이면:

$$\int_a^b f(x) \, dx = \lim_{\epsilon \to 0^+} \int_a^{c-\epsilon} f(x) \, dx + \lim_{\epsilon \to 0^+} \int_{c+\epsilon}^b f(x) \, dx$$

**예시:**

$$\int_0^1 \frac{1}{\sqrt{x}} \, dx = \lim_{\epsilon \to 0^+} \int_\epsilon^1 x^{-1/2} \, dx = \lim_{\epsilon \to 0^+} [2\sqrt{x}]_\epsilon^1 = 2 - \lim_{\epsilon \to 0^+} 2\sqrt{\epsilon} = 2$$

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x = sp.Symbol('x')

# Type I: Convergent vs divergent
print("Type I improper integrals:")
print(f"  integral_1^inf 1/x^2 dx = {sp.integrate(1/x**2, (x, 1, sp.oo))}")
print(f"  integral_1^inf 1/x dx = {sp.integrate(1/x, (x, 1, sp.oo))}")

# The Gaussian integral -- one of the most important improper integrals
print(f"\n  integral_0^inf e^(-x^2) dx = {sp.integrate(sp.exp(-x**2), (x, 0, sp.oo))}")

# Type II: Discontinuous integrand
print(f"\nType II improper integrals:")
print(f"  integral_0^1 1/sqrt(x) dx = {sp.integrate(1/sp.sqrt(x), (x, 0, 1))}")

# Visualize convergence: partial integrals approaching the limit
t_values = np.linspace(1, 50, 200)

# 1/x^2 converges to 1
partial_convergent = 1 - 1/t_values

# 1/x diverges
partial_divergent = np.log(t_values)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(t_values, partial_convergent, 'b-', linewidth=2)
ax1.axhline(y=1, color='red', linestyle='--', label='Limit = 1')
ax1.set_xlabel('Upper limit $t$')
ax1.set_ylabel('$\\int_1^t x^{-2} \\, dx$')
ax1.set_title('Convergent: $\\int_1^\\infty x^{-2} \\, dx = 1$')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(t_values, partial_divergent, 'r-', linewidth=2)
ax2.set_xlabel('Upper limit $t$')
ax2.set_ylabel('$\\int_1^t x^{-1} \\, dx$')
ax2.set_title('Divergent: $\\int_1^\\infty x^{-1} \\, dx = \\infty$')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('improper_integrals.png', dpi=150)
plt.show()
```

## 올바른 기법 선택하기

적절한 적분 방법을 선택하기 위한 결정 가이드:

```
피적분함수가 유리함수 P(x)/Q(x)인가?
  예 --> 부분분수 (deg P >= deg Q이면, 먼저 다항식 나눗셈 수행)

sqrt(a^2 - x^2), sqrt(a^2 + x^2), 또는 sqrt(x^2 - a^2)를 포함하는가?
  예 --> 삼각 치환

"함수와 그 도함수" 패턴이 있는가?
  예 --> u-치환

두 가지 다른 유형의 함수의 곱인가?
  예 --> 부분적분 (u 선택에 LIATE 사용)

sin과 cos의 거듭제곱을 포함하는가?
  예 --> 홀수 거듭제곱 또는 거듭제곱 축소 전략 사용

위의 어느 것도 아닌가?
  --> SymPy 또는 수치 적분 시도
```

## 요약

- **$u$-치환**은 연쇄 법칙을 역으로 적용한다: $f(g(x)) \cdot g'(x)$ 패턴을 찾는다
- **부분적분**은 곱의 법칙을 역으로 적용한다: $u$와 $dv$를 선택하기 위해 LIATE 규칙을 사용한다
- **부분분수**는 유리함수를 로그와 아크탄젠트로 적분되는 더 단순한 항으로 분해한다
- **삼각 적분**은 항등식과 치환을 사용하여 거듭제곱과 근호를 처리한다
- **이상 적분**은 정적분을 무한 영역이나 특이 피적분함수로 확장한다; 수렴 여부는 피적분함수가 얼마나 빠르게 감소하는지에 달려 있다
- 수작업 기법이 실패하면, **SymPy**와 **scipy.integrate.quad**가 신뢰할 수 있는 계산적 대안을 제공한다

## 연습 문제

### 문제 1: 치환법

각 적분을 계산하라:

(a) $\int \frac{e^{\sqrt{x}}}{\sqrt{x}} \, dx$

(b) $\int_0^{\pi/2} \cos x \cdot e^{\sin x} \, dx$

(c) $\int \frac{x}{(x^2+1)^3} \, dx$

### 문제 2: 부분적분

다음을 계산하라:

(a) $\int x^2 e^{-x} \, dx$

(b) $\int e^x \cos x \, dx$ (힌트: 부분적분을 두 번 적용한 후, 적분을 대수적으로 풀라)

(c) $\int \arctan x \, dx$

### 문제 3: 부분분수

다음을 계산하라:

(a) $\int \frac{3x+1}{x^2-5x+6} \, dx$

(b) $\int \frac{x^2 + 1}{x(x-1)^2} \, dx$

### 문제 4: 삼각 치환

$x = 3\sin\theta$ 치환을 사용하여 $\int \frac{x^2}{\sqrt{9-x^2}} \, dx$를 계산하라.

### 문제 5: 이상 적분의 수렴

각 적분이 수렴하는지 발산하는지 판정하라. 수렴하면 값을 구하라.

(a) $\int_0^{\infty} x e^{-x} \, dx$

(b) $\int_0^1 \frac{1}{x^{2/3}} \, dx$

(c) $\int_2^{\infty} \frac{1}{x \ln^2 x} \, dx$

SymPy를 사용하여 답을 검증하라.

## 참고 자료

- Stewart, *Calculus: Early Transcendentals*, Ch. 7 (Techniques of Integration)
- [Paul's Online Notes: Integration Techniques](https://tutorial.math.lamar.edu/Classes/CalcII/IntTechIntro.aspx)
- [MIT OCW 18.01: Techniques of Integration](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/)

---

[이전: 적분의 기초](./04_Integration_Fundamentals.md) | [다음: 적분의 응용](./06_Applications_of_Integration.md)
